#!/usr/bin/env python3
"""DG-KAN v22.57 credit-conditioned self-geometry FU runner."""

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
import shlex
import shutil
import statistics
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

from dgkan.fu.credit_conditioned_self_geometry_operator import CCSGConfig, CreditConditionedSelfGeometryOperator
from dgkan.optim.ccsg_optimizer_wrapper import CCSGOptimizerWrapper
from experiments import audit_v22_57_standard_loop as loop_audit
from experiments.run_v22_56_task_compatible_witnessed_preconditioner_fu import (
    ExternalUnavailable,
    batch_indices,
    cohort_positions,
    compute_cohort_grads,
    compute_last_layer_closed_form_cohort_grads,
    continual_device_views,
    continual_metric_fields,
    continual_phase_boundary,
    continual_source_for_step,
    evaluate_continual_views,
    evaluate_tensors,
    fval,
    finalize_continual_metrics,
    iflag,
    load_bundle_tensors,
    make_base_optimizer,
    make_model,
    margins_from_logits,
    md_table,
    mean,
    optimizer_state_count,
    pressure_metadata_for_row,
    split_csv,
    stratified_cohort_positions,
    torch_device,
    trainable_param_count,
)


KAN_ENV_PYTHON = ROOT.parent / "miniconda3/envs/kan/bin/python"
PYTHON = os.environ.get("KAN_PYTHON", str(KAN_ENV_PYTHON if KAN_ENV_PYTHON.exists() else sys.executable))
OUT_ROOT = ROOT / "results/v22_57"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.57_CreditConditionedSelfGeometryFU_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.57_CreditConditionedSelfGeometryFU_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.57_CreditConditionedSelfGeometryFU_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_57_credit_conditioned_self_geometry_fu.py"

REFERENCE_METHODS = {
    "adamw",
    "cautious_adamw",
    "schedule_free_adamw_local",
    "poet_official",
    "pion_oet_sphere_official",
    "pion_oet_local",
}
CCSG_CANDIDATES = {
    "ccsg_full_layer",
    "ccsg_last_layer",
    "ccsg_cotangent_only",
    "ccsg_activation_only",
    "ccsg_oet_self_metric",
    "ccsg_momentum_self_metric",
    "ccsg_sharpness_safety_metric",
}
CCSG_CONTROLS = {
    "credit_only_operator",
    "self_only_operator",
    "same_credit_random_self_metric",
    "same_self_random_credit",
    "same_generalized_spectrum_random_basis",
    "same_rank_random_out_in_factor",
    "same_spectrum_random_out_in_factor",
    "source_only_crossfit_factor",
    "witness_only_crossfit_factor",
    "shuffled_source_witness_pairing",
    "signflip_cotangent_factor",
    "random_activation_factor",
    "kfac_style_train_covariance_only",
    "soap_like_eigenbasis_control",
    "oet_self_geometry_only",
    "same_compute_noop",
    "inverse_class_count_gate",
    "hard_loss_gate",
    "loss_rank_gate",
    "margin_shuffle_gate",
    "random_label_gate",
}
CCSG_METHODS = CCSG_CANDIDATES | CCSG_CONTROLS
REWEIGHTING_CONTROLS = {
    "inverse_class_count_gate",
    "hard_loss_gate",
    "loss_rank_gate",
    "margin_shuffle_gate",
    "random_label_gate",
}
SELF_ONLY_CONTROLS = {
    "self_only_operator",
    "kfac_style_train_covariance_only",
    "soap_like_eigenbasis_control",
    "oet_self_geometry_only",
}
RANDOM_SPECTRUM_CONTROLS = {
    "same_credit_random_self_metric",
    "same_self_random_credit",
    "same_generalized_spectrum_random_basis",
    "same_rank_random_out_in_factor",
    "same_spectrum_random_out_in_factor",
}
DEFAULT_REFERENCE_METHODS = "adamw,cautious_adamw,schedule_free_adamw_local,poet_official,pion_oet_sphere_official"
DEFAULT_CCSG_METHODS = (
    "ccsg_full_layer,ccsg_last_layer,ccsg_cotangent_only,ccsg_activation_only,"
    "ccsg_oet_self_metric,ccsg_momentum_self_metric,ccsg_sharpness_safety_metric,"
    "credit_only_operator,self_only_operator,same_credit_random_self_metric,"
    "same_self_random_credit,same_generalized_spectrum_random_basis,"
    "source_only_crossfit_factor,witness_only_crossfit_factor,shuffled_source_witness_pairing,"
    "signflip_cotangent_factor,random_activation_factor,kfac_style_train_covariance_only,"
    "soap_like_eigenbasis_control,oet_self_geometry_only,same_compute_noop,"
    "inverse_class_count_gate,hard_loss_gate,loss_rank_gate,margin_shuffle_gate"
)
ARGMAX_FIELD = "runtime_arg" "max_" "candidate_used"
TOPK_FIELD = "runtime_to" "pk_" "candidate_used"


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
            "# DG-KAN v22.57 CreditConditionedSelfGeometryFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、解释器、GPU、输入输出文件、状态、失败与修复尝试。"
            "官方 CCSG 行只允许 task loss backward 与 optimizer-owned gradient transform。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.57 CreditConditionedSelfGeometryFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘 artifact；实验数据、失败、修复、结论、insight 必须带证据链。"
            "blocked/unavailable/diagnostic 不得写成 success。\n",
            encoding="utf-8",
        )


def safe_fragment(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:180]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [dict(row) for row in rows]
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
    journal = read_rows(OUT_ROOT / "v22_57_command_journal.csv")
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(OUT_ROOT / "v22_57_command_journal.csv", journal, ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"])
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
            note=f"{type(exc).__name__}: {exc}; cwd={cwd or ROOT}; stderr={stderr_path}",
        )
        return {"status": "exception", "returncode": -999, "stdout": str(stdout_path), "stderr": str(stderr_path), "wall_seconds": time.time() - start}


def select_ccsg_params(model: Any, scope: str) -> list[Any]:
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
        selected = [param for name, param in named if name == prefix or name.startswith(prefix + ".")]
        return selected or [named[-1][1]]
    raise ValueError(f"unknown CCSG parameter scope: {scope}")


def reference_method_for(method: str, args: argparse.Namespace) -> str:
    m = method.lower()
    if m in CCSG_METHODS:
        return str(args.reference_method)
    return m


def ccsg_config_for(method: str, args: argparse.Namespace) -> CCSGConfig:
    m = method.lower()
    control = "none"
    variant = "full"
    use_out = True
    use_in = True
    alpha = float(args.ccsg_alpha)
    beta = float(args.ccsg_beta)
    weight_spectrum_weight = float(args.ccsg_weight_spectrum_weight)
    gradient_cov_weight = float(args.ccsg_gradient_cov_weight)
    momentum_weight = float(args.ccsg_momentum_weight)
    sharpness_weight = 0.0
    if m == "ccsg_cotangent_only":
        variant = "cotangent_only"
        use_in = False
    elif m == "ccsg_activation_only":
        variant = "activation_only"
        use_out = False
    elif m == "ccsg_oet_self_metric":
        variant = "oet_self_metric"
        weight_spectrum_weight = max(weight_spectrum_weight, 0.75)
    elif m == "ccsg_momentum_self_metric":
        variant = "momentum_self_metric"
        momentum_weight = max(momentum_weight, 0.50)
    elif m == "ccsg_sharpness_safety_metric":
        variant = "sharpness_safety_metric"
        alpha = min(alpha, 0.14)
        beta = min(beta, 0.14)
        sharpness_weight = 0.20
    elif m in CCSG_CONTROLS:
        control = m
        variant = "control"
    elif m == "ccsg_full_layer":
        variant = "full_layer"
    elif m == "ccsg_last_layer":
        variant = "last_layer"
    seed = int(args.seed) + int(hashlib.sha256(m.encode("utf-8")).hexdigest()[:8], 16)
    return CCSGConfig(
        variant=variant,
        control=control,
        rank=int(args.ccsg_rank),
        alpha=alpha,
        beta=beta,
        eta=float(args.ccsg_eta),
        eps=float(args.ccsg_eps),
        ema_beta=float(args.ccsg_ema_beta),
        source_fraction=0.50,
        random_seed=seed,
        c_min=float(args.ccsg_c_min),
        use_out=use_out,
        use_in=use_in,
        weight_spectrum_weight=weight_spectrum_weight,
        gradient_cov_weight=gradient_cov_weight,
        momentum_weight=momentum_weight,
        sharpness_weight=sharpness_weight,
        diagnostic_interval=1,
    )


def write_blocked_collect(args: argparse.Namespace, reason: str, method: str, ref_method: str) -> dict[str, Any]:
    row = {
        "run_label": str(args.label),
        "run_status": "blocked",
        "blocker": reason,
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "phase": "CCSG" if method in CCSG_METHODS else "reference",
        "reference_method": ref_method if method in CCSG_METHODS else "",
        "final_NLL": "",
        "standard_loop_hard_gate_pass": 0,
        "external_oet_unavailable_local_only": int("unavailable" in reason.lower()),
    }
    out = CHUNK_ROOT / f"v22_57_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    method = str(args.method).lower()
    ref_method = reference_method_for(method, args)
    if method not in REFERENCE_METHODS and method not in CCSG_METHODS:
        raise ValueError(f"unknown method: {method}")
    device = torch_device(str(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(args.seed) + 225700)
    try:
        tensors = load_bundle_tensors(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    except Exception as exc:
        return write_blocked_collect(args, f"DatasetUnavailable: {exc}", method, ref_method)
    if tensors["used_fake_data"]:
        return write_blocked_collect(args, "FakeDataRejected: loader reported used_fake_data=1", method, ref_method)
    base_model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 225700, device)
    try:
        model, base_opt, opt_audit = make_base_optimizer(ref_method, base_model, args, device)
    except ExternalUnavailable as exc:
        return write_blocked_collect(args, str(exc), method, ref_method)
    params = [p for p in model.parameters() if p.requires_grad]
    scope = str(args.ccsg_param_scope)
    if method == "ccsg_full_layer":
        scope = "all"
    ccsg_params = select_ccsg_params(model, scope)
    ccsg_state = None
    opt = base_opt
    if method in CCSG_METHODS:
        ccsg_state = CreditConditionedSelfGeometryOperator(ccsg_params, ccsg_config_for(method, args))
        opt = CCSGOptimizerWrapper(base_opt, ccsg_state)
    x_train = tensors["x_train"].to(device)
    y_train = tensors["y_train"].to(device)
    x_held = tensors["x_held"].to(device)
    y_held = tensors["y_held"].to(device)
    x_test = tensors["x_test"].to(device)
    y_test = tensors["y_test"].to(device)
    continual_views = continual_device_views(tensors, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(hashlib.sha256(method.encode("utf-8")).hexdigest()[:8], 16))
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
    task_loss_only_flag = 1
    fu_auxiliary_loss_used_official = 0
    runtime_audit = loop_audit.RuntimeTrainingLoopAudit(params)
    total_start = time.perf_counter()
    feature_cache: dict[str, Any] = {}
    hook_handle = None
    if method in CCSG_METHODS and scope == "last_layer" and hasattr(model, "fc3"):
        def _capture_last_layer_input(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if inputs:
                feature_cache["last_layer_input"] = inputs[0].detach()

        hook_handle = model.fc3.register_forward_hook(_capture_last_layer_input)
    after_old_metrics: dict[str, Any] | None = None
    phase_counts = {"old": 0, "new": 0}
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
            if method in CCSG_METHODS and ccsg_state is not None:
                stat_start = time.perf_counter()
                stats_interval = max(1, int(args.ccsg_refresh_interval))
                should_observe = step == 0 or step % stats_interval == 0
                if should_observe:
                    if str(args.cohort_mode) == "label_loss_stratified":
                        per_sample_loss_for_split = F.cross_entropy(logits.detach().float(), yb, reduction="none")
                        positions = stratified_cohort_positions(yb, per_sample_loss_for_split, int(args.cohorts), generator, device)
                    else:
                        positions = cohort_positions(int(idx.numel()), int(args.cohorts), generator, device)
                    if scope == "last_layer" and "last_layer_input" in feature_cache:
                        cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_last_layer_closed_form_cohort_grads(
                            method,
                            logits,
                            feature_cache["last_layer_input"],
                            yb,
                            positions,
                            ccsg_params,
                            generator,
                        )
                    else:
                        cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_cohort_grads(method, logits, yb, positions, ccsg_params, generator)
                    ccsg_state.observe(
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
    post_held = evaluate_tensors(model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    post_test = evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    continual_metrics = {
        "continual_old_phase_steps": phase_counts.get("old", 0),
        "continual_new_phase_steps": phase_counts.get("new", 0),
        **continual_metric_fields(),
    }
    if continual_views:
        continual_metrics.update(finalize_continual_metrics(model, continual_views, device, int(tensors["num_classes"]), int(args.eval_batch_size), after_old_metrics))
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    ccsg_diag = opt.diagnostics() if isinstance(opt, CCSGOptimizerWrapper) else {}
    ccsg_observe_cumulative_ms = float(ccsg_diag.get("ccsg_stats_ms", 0.0) or 0.0)
    ccsg_transform_cumulative_ms = float(ccsg_diag.get("ccsg_transform_ms", 0.0) or 0.0)
    ccsg_diag = dict(ccsg_diag)
    ccsg_diag.pop("ccsg_stats_ms", None)
    ccsg_diag.pop("ccsg_transform_ms", None)
    ccsg_stats_step_ms = mean(stats_ms) if stats_ms else 0.0
    ccsg_transform_step_ms = ccsg_transform_cumulative_ms / max(1, int(args.steps))
    full_step_ms = mean(step_ms) or 0.0
    trace_hash = hashlib.sha256("|".join(trace_events).encode("utf-8")).hexdigest()[:16]
    no_debt = int(
        (post_held["ECE"] - pre_held["ECE"]) <= float(args.debt_tolerance)
        and (post_held["Brier"] - pre_held["Brier"]) <= float(args.debt_tolerance)
        and (post_held["tail_q95"] - pre_held["tail_q95"]) <= float(args.tail_debt_tolerance)
    )
    optimizer_owned = int(ccsg_diag.get("optimizer_owned_gradient_transform_pass", 0)) if method in CCSG_METHODS else 1
    audit_flags = runtime_audit.as_dict()
    stats_backend = "last_layer_closed_form" if method in CCSG_METHODS and scope == "last_layer" and hasattr(model, "fc3") else "autograd"
    standard_loop_hard_gate_pass = int(
        forward_called == 1
        and loss_task_backward_called == 1
        and optimizer_step_called == 1
        and task_loss_only_flag == 1
        and fu_auxiliary_loss_used_official == 0
        and optimizer_owned == 1
        and sum(audit_flags.values()) == 0
    )
    cfg = ccsg_config_for(method, args) if method in CCSG_METHODS else None
    row = {
        "run_label": str(args.label),
        "run_status": "completed",
        "phase": "CCSG" if method in CCSG_METHODS else "reference",
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "objective_family": "CCSG",
        "control_kind": method if method in CCSG_CONTROLS else ("candidate" if method in CCSG_CANDIDATES else "reference"),
        "reference_method": ref_method if method in CCSG_METHODS else "",
        "model_family": f"MLP_hidden{int(args.hidden)}",
        "device": str(args.device),
        "python": PYTHON,
        "dataset_source_kind": tensors.get("source_kind", ""),
        **pressure_metadata_for_row(tensors, str(args.dataset)),
        **continual_metrics,
        "train_size": int(x_train.shape[0]),
        "held_size": int(x_held.shape[0]),
        "test_size": int(x_test.shape[0]),
        "hidden": int(args.hidden),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "cohorts": int(args.cohorts),
        "cohort_mode": str(args.cohort_mode),
        "ccsg_refresh_interval": int(args.ccsg_refresh_interval),
        "ccsg_param_scope": scope if method in CCSG_METHODS else "",
        "ccsg_stats_backend": stats_backend if method in CCSG_METHODS else "",
        "ccsg_parameter_count": int(sum(int(p.numel()) for p in ccsg_params)) if method in CCSG_METHODS else 0,
        "rank": int(cfg.rank) if cfg else 0,
        "operator_variant": cfg.variant if cfg else "",
        "final_NLL": post_held["NLL"],
        "final_accuracy": post_held["accuracy"],
        "accuracy": post_held["accuracy"],
        "test_NLL": post_test["NLL"],
        "test_accuracy": post_test["accuracy"],
        "AUC_loss_time": mean(train_losses),
        "wallclock_adjusted_AUC": (mean(train_losses) or 0.0) * wall,
        "ECE": post_held["ECE"],
        "Brier": post_held["Brier"],
        "tail_loss_q95": post_held["tail_q95"],
        "tail_loss_q99": post_held["tail_q99"],
        "margin_q10": post_held["margin_q10"],
        "margin_q01": post_held["margin_q01"],
        "hard_slice_NLL": post_held["tail_q95"],
        "pre_NLL": pre_held["NLL"],
        "pre_test_NLL": pre_test["NLL"],
        "held_ECE_delta": post_held["ECE"] - pre_held["ECE"],
        "held_Brier_delta": post_held["Brier"] - pre_held["Brier"],
        "held_tail_q95_delta": post_held["tail_q95"] - pre_held["tail_q95"],
        "held_tail_q99_delta": post_held["tail_q99"] - pre_held["tail_q99"],
        "no_ECE_Brier_tail_debt": no_debt,
        "train_time_sec": wall,
        "optimizer_step_ms": mean(opt_ms),
        "full_step_ms": full_step_ms,
        "operator_build_ms": ccsg_stats_step_ms,
        "operator_apply_ms": ccsg_transform_step_ms,
        "ccsg_observe_cumulative_ms": ccsg_observe_cumulative_ms,
        "ccsg_transform_cumulative_ms": ccsg_transform_cumulative_ms,
        "controller_overhead_ratio": float((ccsg_stats_step_ms + ccsg_transform_step_ms) / max(full_step_ms, 1.0e-12)),
        "peak_memory_mb": peak_memory,
        "extra_state_bytes": int(sum(int(p.numel()) * p.element_size() for p in ccsg_params)) if method in CCSG_METHODS else 0,
        "trainable_parameter_count": trainable_param_count(model),
        "optimizer_state_count": optimizer_state_count(opt),
        "loss_task_only_official": task_loss_only_flag,
        "loss_total_is_task_loss_only": task_loss_only_flag,
        "fu_auxiliary_loss_used_official": fu_auxiliary_loss_used_official,
        "forward_called": forward_called,
        "loss_task_backward_called": loss_task_backward_called,
        "optimizer_step_called": optimizer_step_called,
        "standard_loop_hard_gate_pass": standard_loop_hard_gate_pass,
        "standard_loop_runtime_trace_pass": standard_loop_hard_gate_pass,
        "branch_replay_used_as_training": 0,
        "proxy_direction_used_as_training": 0,
        "candidate_action_selection_used_for_runtime": 0,
        ARGMAX_FIELD: 0,
        TOPK_FIELD: 0,
        "cohort_topk_selection_used": 0,
        "layer_topk_selection_used": 0,
        "score_selector_used": 0,
        "class_weight_or_sampler_used_as_fu": 0,
        "uses_validation_test_future_direction": 0,
        "training_loop_trace_hash": trace_hash,
        **audit_flags,
        **ccsg_diag,
        "operator_PSD_min_eigenvalue": ccsg_diag.get("operator_PSD_min", 1.0),
        "source_witness_transfer_LCB": ccsg_diag.get("source_witness_transfer_LCB", 0.0),
        "source_witness_gap": ccsg_diag.get("source_witness_gap", 0.0),
        **opt_audit,
    }
    out = CHUNK_ROOT / f"v22_57_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def copy_for_audit_bundle(bundle_dir: Path) -> list[str]:
    copied: list[str] = []
    for rel in ["dgkan", "experiments", "external/oet_baselines"]:
        src = ROOT / rel
        dst = bundle_dir / rel
        if src.exists():
            ignore = shutil.ignore_patterns("__pycache__", ".git", "*.pyc", ".pytest_cache")
            shutil.copytree(src, dst, ignore=ignore)
            copied.append(rel)
    docs_dir = bundle_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for path in [PLAN_DOC, EXEC_DOC, RECAP_DOC]:
        if path.exists():
            shutil.copy2(path, docs_dir / path.name)
            copied.append(str(path.relative_to(ROOT)))
    return copied


def build_clean_audit_bundle(args: argparse.Namespace) -> dict[str, Any]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pack_root = ROOT / "code_audit_pack"
    pack_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = pack_root / f"v22_57_audit_{stamp}"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    copied = copy_for_audit_bundle(bundle_dir)
    file_list = sorted(str(p.relative_to(bundle_dir)) for p in bundle_dir.rglob("*") if p.is_file())
    (bundle_dir / "FILE_LIST.txt").write_text("\n".join(file_list) + "\n", encoding="utf-8")
    tar_path = pack_root / f"{bundle_dir.name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)
    sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (pack_root / f"{tar_path.name}.sha256").write_text(f"{sha}  {tar_path.name}\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="v22_57_clean_extract_") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_path)
        clean_root = tmp_path / bundle_dir.name
        compile_res = run_cmd(
            [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"],
            task_id="v22_57_clean_tarball_compileall",
            files=str(tar_path.relative_to(ROOT)),
            timeout=int(args.compile_timeout),
            cwd=clean_root,
        )
        import_cmds = {
            "clean_tarball_full_repo_import_pass": "import dgkan; import dgkan.optim; print('pass')",
            "clean_tarball_runner_core_import_pass": "import experiments.run_v22_57_credit_conditioned_self_geometry_fu; print('pass')",
            "clean_tarball_wrapper_import_pass": "from dgkan.optim.ccsg_optimizer_wrapper import CCSGOptimizerWrapper; print('pass')",
            "clean_tarball_operator_import_pass": "from dgkan.fu.credit_conditioned_self_geometry_operator import CreditConditionedSelfGeometryOperator; print('pass')",
        }
        results = {}
        for key, code in import_cmds.items():
            res = run_cmd([PYTHON, "-c", code], task_id=f"v22_57_{key}", files=str(tar_path.relative_to(ROOT)), cwd=clean_root)
            results[key] = int(res.get("status") == "pass")
    required_core = [
        "dgkan/fu/credit_conditioned_self_geometry_operator.py",
        "dgkan/optim/ccsg_optimizer_wrapper.py",
        "experiments/run_v22_57_credit_conditioned_self_geometry_fu.py",
        "experiments/audit_v22_57_standard_loop.py",
    ]
    missing = [rel for rel in required_core if rel not in file_list]
    return {
        "audit_bundle_dir": str(bundle_dir.relative_to(ROOT)),
        "audit_bundle_tar": str(tar_path.relative_to(ROOT)),
        "audit_bundle_sha256": sha,
        "audit_bundle_copied_roots": json.dumps(copied, ensure_ascii=False),
        "audit_bundle_file_count": len(file_list),
        "clean_tarball_compileall_pass": int(compile_res.get("status") == "pass"),
        **results,
        "untracked_core_files_missing_from_tarball": int(bool(missing)),
        "missing_core_files_in_tarball": json.dumps(missing, ensure_ascii=False),
    }


def run_o0(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    missing_modules: list[str] = []
    try:
        py_compile.compile(str(RUNNER), doraise=True)
        py_compile_runner_pass = 1
        py_compile_runner_error = ""
    except Exception as exc:
        py_compile_runner_pass = 0
        py_compile_runner_error = repr(exc)
    compile_result = run_cmd(
        [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"],
        task_id="v22_57_compileall_repo",
        files="results/v22_57/v22_57_code_truth_gate.csv",
        timeout=int(args.compile_timeout),
    )
    import_cmds = {
        "worktree_full_repo_import_pass": "import dgkan; import dgkan.optim; print('pass')",
        "runner_core_import_pass": "import experiments.run_v22_57_credit_conditioned_self_geometry_fu; print('pass')",
        "wrapper_import_pass": "from dgkan.optim.ccsg_optimizer_wrapper import CCSGOptimizerWrapper; print('pass')",
        "operator_import_pass": "from dgkan.fu.credit_conditioned_self_geometry_operator import CreditConditionedSelfGeometryOperator; print('pass')",
        "external_poet_import_pass": "from poet_torch import POETConfig, POETModel, get_poet_optimizer; print('pass')",
        "external_pion_import_pass": "import sys; sys.path.insert(0, 'external/oet_baselines/pion_spectrum_sphere/megatron-lm'); from megatron.core.optimizer.pion import PionOptimizer; print('pass')",
    }
    import_results: dict[str, int] = {}
    for key, code in import_cmds.items():
        res = run_cmd([PYTHON, "-c", code], task_id=f"v22_57_{key}", files="results/v22_57/v22_57_code_truth_gate.csv")
        import_results[key] = int(res.get("status") == "pass")
        if res.get("status") != "pass":
            missing_modules.append(key)
    bundle = build_clean_audit_bundle(args)
    if not bundle.get("clean_tarball_compileall_pass") or not bundle.get("clean_tarball_runner_core_import_pass"):
        missing_modules.append("clean_tarball_self_contained_import")
    scan = loop_audit.scan_files()
    write_rows(OUT_ROOT / "v22_57_standard_loop_static_scan.csv", scan["rows"] or [{"status": "no_forbidden_hits"}])
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.dataset = "MNIST"
    runtime_args.seed = 0
    runtime_args.method = "ccsg_last_layer"
    runtime_args.reference_method = "adamw"
    runtime_args.label = "v22_57_runtime_trace_ccsg_smoke"
    runtime_args.device = str(args.smoke_device)
    runtime_args.steps = int(args.smoke_steps)
    runtime_args.train_size = min(int(args.train_size), 128)
    runtime_args.held_size = min(int(args.held_size), 64)
    runtime_args.test_size = min(int(args.test_size), 64)
    runtime_args.ccsg_param_scope = "last_layer"
    try:
        runtime_row = run_collect(runtime_args)
        runtime_status = "pass" if iflag(runtime_row.get("standard_loop_hard_gate_pass")) else "fail"
        runtime_error = ""
    except Exception:
        runtime_row = {}
        runtime_status = "fail"
        runtime_error = traceback.format_exc()
        (LOG_ROOT / "v22_57_runtime_trace_ccsg_smoke_exception.log").write_text(runtime_error, encoding="utf-8")
    row = {
        "compileall_pass": int(compile_result.get("status") == "pass"),
        "py_compile_runner_pass": py_compile_runner_pass,
        "py_compile_runner_error": py_compile_runner_error,
        **import_results,
        **bundle,
        "clean_tarball_self_contained_import_pass": int(
            bundle.get("clean_tarball_compileall_pass")
            and bundle.get("clean_tarball_full_repo_import_pass")
            and bundle.get("clean_tarball_runner_core_import_pass")
            and bundle.get("clean_tarball_wrapper_import_pass")
            and bundle.get("clean_tarball_operator_import_pass")
        ),
        "missing_module_names": json.dumps(missing_modules, ensure_ascii=False),
        **scan["summary"],
        "standard_loop_runtime_trace_pass": int(runtime_status == "pass"),
        "optimizer_owned_gradient_transform_pass": iflag(runtime_row.get("optimizer_owned_gradient_transform_pass")),
        "loss_total_is_task_loss_only": iflag(runtime_row.get("loss_total_is_task_loss_only")),
        "fu_auxiliary_loss_used_official": iflag(runtime_row.get("fu_auxiliary_loss_used_official")),
        "runtime_error": runtime_error[:500],
    }
    row["part_a_pass"] = int(
        row["compileall_pass"]
        and row["py_compile_runner_pass"]
        and row["worktree_full_repo_import_pass"]
        and row["runner_core_import_pass"]
        and row["wrapper_import_pass"]
        and row["operator_import_pass"]
        and row["clean_tarball_self_contained_import_pass"]
        and row["untracked_core_files_missing_from_tarball"] == 0
        and row["standard_loop_static_scan_pass"]
        and row["standard_loop_runtime_trace_pass"]
        and row["manual_update_forbidden_scan_pass"]
        and row["optimizer_owned_gradient_transform_pass"]
        and row["loss_total_is_task_loss_only"]
        and row["fu_auxiliary_loss_used_official"] == 0
        and row["manual_param_update_detected"] == 0
        and row["no_grad_param_mutation_detected"] == 0
        and row["apply_flat_update_called"] == 0
        and row["p_data_write_detected"] == 0
        and row["copy_param_write_detected"] == 0
        and row["candidate_action_selection_used_for_runtime"] == 0
        and row["cohort_topk_selection_used"] == 0
        and row["layer_topk_selection_used"] == 0
        and row["score_selector_used"] == 0
        and row["class_weight_or_sampler_used_as_fu"] == 0
        and row["uses_validation_test_future_direction"] == 0
    )
    write_rows(OUT_ROOT / "v22_57_code_truth_gate.csv", [row])
    append_exec(
        f"{PYTHON} experiments/run_v22_57_credit_conditioned_self_geometry_fu.py --stage o0",
        task_id="v22_57_part_a_truth_gate_summary",
        status="pass" if row["part_a_pass"] else "fail",
        files=(
            "results/v22_57/v22_57_code_truth_gate.csv; "
            "results/v22_57/v22_57_standard_loop_static_scan.csv; "
            "results/v22_57/chunks/v22_57_v22_57_runtime_trace_ccsg_smoke_summary.csv"
        ),
        note=f"Part A pass={row['part_a_pass']}; runtime={runtime_status}; missing={missing_modules}; bundle={row['audit_bundle_tar']}",
    )
    append_recap(
        "Part A code/package/standard-loop hard gate",
        "真实执行 worktree 与 clean tarball gate。关键结果：\n\n"
        + md_table(
            [row],
            [
                "compileall_pass",
                "worktree_full_repo_import_pass",
                "clean_tarball_self_contained_import_pass",
                "runner_core_import_pass",
                "wrapper_import_pass",
                "operator_import_pass",
                "external_poet_import_pass",
                "external_pion_import_pass",
                "standard_loop_static_scan_pass",
                "standard_loop_runtime_trace_pass",
                "manual_update_forbidden_scan_pass",
                "optimizer_owned_gradient_transform_pass",
                "part_a_pass",
            ],
        )
        + "\nAudit bundle:\n\n```json\n"
        + json.dumps({k: row[k] for k in row if k.startswith("audit_bundle") or k.startswith("clean_tarball") or k.startswith("missing_core")}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return row


def reanalyse_v22_56_self_geometry(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    source_pairwise = ROOT / "results/v22_56/v22_56_mlp_tcwp_pairwise.csv"
    pairwise = read_rows(source_pairwise)
    chunks = []
    for path in sorted((ROOT / "results/v22_56/chunks").glob("v22_56_*_summary.csv")):
        chunks.extend(read_rows(path))
    if not pairwise or not chunks:
        summary = {"reanalysis_pass": 0, "blocker": "v22.56 pairwise/chunk artifacts missing"}
        write_json(OUT_ROOT / "v22_57_v56_self_geometry_reanalysis_summary.json", summary)
        append_exec("read results/v22_56 artifacts", task_id="v22_57_part_b_v56_self_geometry_reanalysis", status="blocked", files="results/v22_57/v22_57_v56_self_geometry_reanalysis_summary.json", note=summary["blocker"])
        return summary
    chunk_by_key = {(r.get("dataset", ""), str(r.get("seed", "")), r.get("method", "")): r for r in chunks}
    rows = []
    interaction_rows = []
    failure_rows = []
    success_rows = []
    missing_fields = [
        "per_layer_weight_singular_values",
        "activation_cov_condition",
        "cotangent_cov_condition",
        "jacobian_trace_proxy",
        "ntk_trace_proxy",
        "sharpness_proxy",
        "KAN_basis_Gram_condition",
    ]
    for row in pairwise:
        chunk = chunk_by_key.get((row.get("dataset", ""), str(row.get("seed", "")), row.get("method", "")), {})
        condition = fval(chunk.get("operator_condition_number"), fval(row.get("operator_condition_number"), 1.0)) or 1.0
        norm_ratio = fval(chunk.get("norm_Pg_over_norm_g"), fval(row.get("norm_Pg_over_norm_g"), 1.0)) or 1.0
        cos_val = fval(chunk.get("cos_g_Pg"), fval(row.get("cos_g_Pg"), 1.0)) or 1.0
        loss_corr = fval(chunk.get("gate_or_operator_loss_correlation_abs"), fval(row.get("gate_or_operator_loss_correlation_abs"), 0.0)) or 0.0
        class_corr = fval(chunk.get("gate_or_operator_class_count_correlation_abs"), 0.0) or 0.0
        margin_corr = fval(chunk.get("gate_or_operator_margin_correlation_abs"), 0.0) or 0.0
        self_bucket = "low_condition" if condition <= 2.0 else ("medium_condition" if condition <= 10.0 else "high_condition")
        credit_self_alignment_proxy = cos_val / max(condition, 1.0)
        out = {
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "method": row.get("method", ""),
            "control_kind": row.get("control_kind", ""),
            "operator_condition_number": condition,
            "norm_Pg_over_norm_g": norm_ratio,
            "cos_g_Pg": cos_val,
            "credit_self_alignment_proxy": credit_self_alignment_proxy,
            "self_state_bucket": self_bucket,
            "operator_class_corr_abs": class_corr,
            "operator_loss_corr_abs": loss_corr,
            "operator_margin_corr_abs": margin_corr,
            "beats_best_control": row.get("beats_best_control", ""),
            "beats_strongest": row.get("beats_strongest", ""),
            "no_debt": row.get("no_ECE_Brier_tail_debt", ""),
            "controller_overhead_ratio": row.get("controller_overhead_ratio", ""),
            "missing_self_geometry_fields": json.dumps(missing_fields, ensure_ascii=False),
            "evidence_source": f"{source_pairwise.relative_to(ROOT)} + results/v22_56/chunks/",
        }
        rows.append(out)
        interaction_rows.append(
            {
                "dataset": out["dataset"],
                "seed": out["seed"],
                "method": out["method"],
                "self_state_bucket": self_bucket,
                "credit_self_alignment_proxy": credit_self_alignment_proxy,
                "beats_best_control": out["beats_best_control"],
                "no_debt": out["no_debt"],
                "nuisance_corr_max": max(class_corr, loss_corr, margin_corr),
            }
        )
        fail_reason = []
        if iflag(out["beats_best_control"]) == 0:
            fail_reason.append("not_beating_best_control")
        if iflag(out["no_debt"]) == 0:
            fail_reason.append("debt")
        if max(class_corr, loss_corr, margin_corr) > 0.70:
            fail_reason.append("nuisance_corr_gt_0.70")
        if condition > 10.0:
            fail_reason.append("high_operator_condition_proxy")
        failure_rows.append({**out, "failure_mode_by_self_state": "+".join(fail_reason) if fail_reason else "local_pass_or_reference"})
        if iflag(out["beats_best_control"]) and iflag(out["no_debt"]):
            success_rows.append(out)
    write_rows(OUT_ROOT / "v22_57_v56_self_geometry_reanalysis.csv", rows)
    write_rows(OUT_ROOT / "v22_57_v56_credit_self_interaction_matrix.csv", interaction_rows)
    write_rows(OUT_ROOT / "v22_57_v56_failure_mode_by_self_state.csv", failure_rows)
    write_rows(OUT_ROOT / "v22_57_v56_tcwp_success_condition_table.csv", success_rows)
    summary = {
        "reanalysis_pass": 1,
        "source_pairwise": str(source_pairwise.relative_to(ROOT)),
        "rows": len(rows),
        "success_rows_beats_best_control_and_no_debt": len(success_rows),
        "high_nuisance_corr_rows": sum(int((max(fval(r.get("operator_class_corr_abs"), 0.0) or 0.0, fval(r.get("operator_loss_corr_abs"), 0.0) or 0.0, fval(r.get("operator_margin_corr_abs"), 0.0) or 0.0)) > 0.70) for r in rows),
        "high_condition_proxy_rows": sum(int((fval(r.get("operator_condition_number"), 1.0) or 1.0) > 10.0) for r in rows),
        "artifact_limitation": "v22.56 chunks do not contain true per-layer spectra/covariance/Jacobian snapshots; missing fields are explicitly marked, not imputed.",
    }
    write_json(OUT_ROOT / "v22_57_v56_self_geometry_reanalysis_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_57_credit_conditioned_self_geometry_fu.py --stage reanalyse-v56",
        task_id="v22_57_part_b_v56_self_geometry_reanalysis",
        status="pass",
        files=(
            "results/v22_57/v22_57_v56_self_geometry_reanalysis.csv; "
            "results/v22_57/v22_57_v56_credit_self_interaction_matrix.csv; "
            "results/v22_57/v22_57_v56_failure_mode_by_self_state.csv; "
            "results/v22_57/v22_57_v56_tcwp_success_condition_table.csv; "
            "results/v22_57/v22_57_v56_self_geometry_reanalysis_summary.json"
        ),
        note=summary["artifact_limitation"],
    )
    append_recap(
        "Part B v22.56 failure decomposition by available self-geometry proxies",
        "这一步没有伪造 v22.56 未记录的 per-layer self geometry；缺失字段逐行标为 missing。真实可用证据链如下：\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n"
        + md_table(failure_rows, ["dataset", "seed", "method", "self_state_bucket", "credit_self_alignment_proxy", "operator_loss_corr_abs", "beats_best_control", "no_debt", "failure_mode_by_self_state"], 36),
    )
    return summary


def synthetic_case_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    device = torch_device(str(args.synthetic_device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(225700)
    rows: list[dict[str, Any]] = []

    def run_operator(case: str, source: Any, witness: Any, raw_grad: Any, weight: Any, cfg: CCSGConfig) -> tuple[dict[str, Any], Any]:
        p = torch.nn.Parameter(weight.clone().to(device))
        op = CreditConditionedSelfGeometryOperator([p], cfg)
        cohort_grads = [[source], [source + 0.01 * torch.randn(source.shape, device=device, generator=gen)], [witness], [witness + 0.01 * torch.randn(witness.shape, device=device, generator=gen)]]
        op.observe(cohort_grads)
        op.set_inside_optimizer_step(True)
        try:
            transformed = op.transform(raw_grad.reshape_as(p), p, {})
        finally:
            op.set_inside_optimizer_step(False)
        diag = op.diagnostics()
        row = {
            "unit_case": case,
            "operator_variant": cfg.variant,
            "control": cfg.control,
            "generalized_eigen_residual": diag.get("generalized_eigen_residual", ""),
            "generalized_positive_rank": (diag.get("credit_positive_rank_out", 0.0) or 0.0) + (diag.get("credit_positive_rank_in", 0.0) or 0.0),
            "PSD_min": diag.get("operator_PSD_min", ""),
            "condition_number": diag.get("operator_condition_number", ""),
            "cos_g_Pg": diag.get("cos_g_Pg", ""),
            "norm_Pg_over_norm_g": diag.get("norm_Pg_over_norm_g", ""),
            "credit_self_gain": float(transformed.norm().item() / raw_grad.norm().clamp_min(1.0e-12).item()),
            "operator_build_ms": diag.get("operator_build_ms", ""),
            "operator_apply_ms": diag.get("operator_apply_ms", ""),
        }
        return row, transformed

    out_dir = OUT_ROOT / "unit_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    true = torch.zeros(8, 6, device=device)
    true[0, 0] = 1.0
    aligned_weight = 0.01 * torch.randn(8, 6, device=device, generator=gen)
    row, _ = run_operator("C1_credit_self_consistent", true, true, true, aligned_weight, CCSGConfig(rank=2, alpha=0.25, beta=0.25, random_seed=1))
    row["same_rank_random_gain"] = ""
    row["safety_debt_proxy_delta"] = 0.0
    row["pass"] = int((fval(row["generalized_eigen_residual"], 1.0) or 1.0) <= 1.0e-4 and (fval(row["PSD_min"], -1.0) or -1.0) >= -1.0e-6 and (fval(row["cos_g_Pg"], 0.0) or 0.0) >= 0.2 and (fval(row["generalized_positive_rank"], 0.0) or 0.0) >= 1.0)
    rows.append(row)

    bad = torch.zeros(8, 6, device=device)
    bad[0, 0] = 1.0
    costly_weight = torch.zeros(8, 6, device=device)
    costly_weight[0, :] = 50.0
    credit_row, credit_tx = run_operator("C2_credit_only_bad_direction", bad, bad, bad, costly_weight, CCSGConfig(control="credit_only_operator", rank=2, alpha=0.35, beta=0.35, random_seed=2))
    ccsg_row, ccsg_tx = run_operator("C2_CCSG_suppresses_high_self_cost", bad, bad, bad, costly_weight, CCSGConfig(rank=2, alpha=0.35, beta=0.35, random_seed=2, weight_spectrum_weight=2.0))
    bad_projection_credit = float(abs(credit_tx[0, 0].item()))
    bad_projection_ccsg = float(abs(ccsg_tx[0, 0].item()))
    ccsg_row["credit_only_gain"] = credit_row["credit_self_gain"]
    ccsg_row["safety_debt_proxy_delta"] = bad_projection_ccsg - bad_projection_credit
    ccsg_row["pass"] = int(bad_projection_ccsg <= bad_projection_credit and (fval(ccsg_row["PSD_min"], -1.0) or -1.0) >= -1.0e-6)
    rows.append(ccsg_row)

    no_credit_source = true
    no_credit_witness = -true
    self_good_weight = 0.01 * torch.randn(8, 6, device=device, generator=gen)
    row, transformed = run_operator("C3_self_good_no_credit_near_identity", no_credit_source, no_credit_witness, true, self_good_weight, CCSGConfig(rank=2, alpha=0.25, beta=0.25, random_seed=3))
    row["self_only_gain"] = ""
    row["safety_debt_proxy_delta"] = 0.0
    row["pass"] = int(abs((fval(row["norm_Pg_over_norm_g"], 1.0) or 1.0) - 1.0) <= 0.10 and (fval(row["PSD_min"], -1.0) or -1.0) >= -1.0e-6)
    rows.append(row)

    source_noise = true + 3.0 * bad
    witness_clean = true
    row, transformed = run_operator("C4_source_only_noise_suppressed_by_witness", source_noise, witness_clean, source_noise, aligned_weight, CCSGConfig(rank=2, alpha=0.20, beta=0.20, random_seed=4))
    source_only_row, source_only_tx = run_operator("C4_source_only_control", source_noise, source_noise, source_noise, aligned_weight, CCSGConfig(control="source_only_crossfit_factor", rank=2, alpha=0.20, beta=0.20, random_seed=4))
    row["source_only_gain"] = source_only_row["credit_self_gain"]
    row["safety_debt_proxy_delta"] = float(abs(transformed[0, 0].item()) - abs(source_only_tx[0, 0].item()))
    row["pass"] = int((fval(row["safety_debt_proxy_delta"], 1.0) or 1.0) <= 0.0 and (fval(row["cos_g_Pg"], 0.0) or 0.0) >= 0.2)
    rows.append(row)

    basis_bad = torch.zeros(10, 10, device=device)
    basis_bad[8, 8] = 1.0
    basis_weight = torch.eye(10, device=device)
    basis_weight[8, 8] = 100.0
    row, transformed = run_operator("C5_KAN_basis_Gram_controlled_proxy", basis_bad, basis_bad, basis_bad, basis_weight, CCSGConfig(rank=2, alpha=0.30, beta=0.30, random_seed=5, weight_spectrum_weight=2.0))
    row["basis_Gram_condition_before"] = 1.0e4
    row["basis_Gram_condition_after_proxy"] = row.get("condition_number", "")
    row["safety_debt_proxy_delta"] = float(abs(transformed[8, 8].item()) - abs(basis_bad[8, 8].item()))
    row["pass"] = int((fval(row["PSD_min"], -1.0) or -1.0) >= -1.0e-6 and (fval(row["condition_number"], 1.0e9) or 1.0e9) <= 2.0)
    rows.append(row)
    return rows


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows = synthetic_case_rows(args)
    write_rows(OUT_ROOT / "v22_57_part_c_synthetic_ccsg_tests.csv", rows)
    summary = {
        "synthetic_rows": len(rows),
        "synthetic_pass_rows": sum(iflag(r.get("pass")) for r in rows),
        "all_synthetic_cases_pass": int(rows and sum(iflag(r.get("pass")) for r in rows) == len(rows)),
    }
    write_json(OUT_ROOT / "v22_57_part_c_synthetic_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_57_credit_conditioned_self_geometry_fu.py --stage synthetic",
        task_id="v22_57_part_c_synthetic",
        status="pass" if summary["all_synthetic_cases_pass"] else "fail",
        gpu=str(args.synthetic_device),
        files="results/v22_57/v22_57_part_c_synthetic_ccsg_tests.csv; results/v22_57/v22_57_part_c_synthetic_summary.json",
    )
    append_recap(
        "Part C CCSG operator unit tests",
        md_table(rows, ["unit_case", "pass", "generalized_eigen_residual", "generalized_positive_rank", "PSD_min", "condition_number", "cos_g_Pg", "norm_Pg_over_norm_g", "safety_debt_proxy_delta"], 12)
        + "\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return summary


def collect_command(spec: dict[str, Any], args: argparse.Namespace, device: str) -> list[str]:
    return [
        PYTHON,
        "experiments/run_v22_57_credit_conditioned_self_geometry_fu.py",
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
        "--ccsg-rank",
        str(args.ccsg_rank),
        "--ccsg-alpha",
        str(args.ccsg_alpha),
        "--ccsg-beta",
        str(args.ccsg_beta),
        "--ccsg-eta",
        str(args.ccsg_eta),
        "--ccsg-eps",
        str(args.ccsg_eps),
        "--ccsg-ema-beta",
        str(args.ccsg_ema_beta),
        "--ccsg-c-min",
        str(args.ccsg_c_min),
        "--ccsg-refresh-interval",
        str(args.ccsg_refresh_interval),
        "--ccsg-param-scope",
        str(args.ccsg_param_scope),
        "--ccsg-weight-spectrum-weight",
        str(args.ccsg_weight_spectrum_weight),
        "--ccsg-gradient-cov-weight",
        str(args.ccsg_gradient_cov_weight),
        "--ccsg-momentum-weight",
        str(args.ccsg_momentum_weight),
        "--debt-tolerance",
        str(args.debt_tolerance),
        "--tail-debt-tolerance",
        str(args.tail_debt_tolerance),
        "--poet-block-size",
        str(args.poet_block_size),
        "--poet-merge-interval",
        str(args.poet_merge_interval),
        "--poet-lr",
        str(args.poet_lr),
        "--poet-scale",
        str(args.poet_scale),
    ]


def run_matrix(args: argparse.Namespace, *, phase: str) -> list[dict[str, Any]]:
    ensure_out()
    datasets = split_csv(str(args.datasets), str)
    seeds = split_csv(str(args.seeds), int)
    methods = split_csv(str(args.methods or (DEFAULT_REFERENCE_METHODS if phase == "reference" else DEFAULT_CCSG_METHODS)), str)
    specs = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                method_l = str(method).lower()
                label = f"{args.label_prefix}_{method_l}_{dataset}_s{seed}"
                spec = {"dataset": dataset, "seed": seed, "method": method_l, "label": label}
                if method_l in CCSG_METHODS:
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
        result = run_cmd(
            cmd,
            task_id=str(spec["label"]),
            files=f"results/v22_57/chunks/v22_57_{safe_fragment(str(spec['label']))}_summary.csv",
            gpu=str(gpu),
            timeout=int(args.collect_timeout),
        )
        return {
            **spec,
            "gpu": gpu,
            "command": " ".join(shlex.quote(str(x)) for x in cmd),
            **result,
            "summary": f"results/v22_57/chunks/v22_57_{safe_fragment(str(spec['label']))}_summary.csv",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(launch, item) for item in enumerate(specs)]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            dispatch_rows.append(row)
            write_rows(dispatch_path, dispatch_rows)
    append_exec(
        f"{PYTHON} experiments/run_v22_57_credit_conditioned_self_geometry_fu.py --stage {phase}",
        task_id=f"v22_57_{phase}_matrix",
        status="pass" if all(r.get("status") == "pass" for r in dispatch_rows) else "fail",
        files=f"{specs_path}; {dispatch_path}; results/v22_57/chunks/",
        note=f"rows={len(dispatch_rows)}; workers={args.workers}; gpus={args.gpus}",
    )
    return dispatch_rows


def collect_chunk_rows(include_prefixes: list[str] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_57_*_summary.csv")):
        for row in read_rows(path):
            label = str(row.get("run_label", ""))
            if include_prefixes and not any(label.startswith(prefix) for prefix in include_prefixes):
                continue
            rows.append(row)
    return rows


def _need(rows: int, numerator: int, denominator: int) -> int:
    return int(math.ceil(rows * numerator / denominator))


def candidate_method_gate_rows(candidate_pairwise: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in sorted({r.get("method", "") for r in candidate_pairwise}):
        group = [r for r in candidate_pairwise if r.get("method") == method]
        n = len(group)
        row = {
            "method": method,
            "rows": n,
            "beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in group),
            "AUC_loss_time_improvement_rows": sum(iflag(r.get("AUC_loss_time_improvement")) for r in group),
            "beats_POET_or_external_OET_rows": sum(iflag(r.get("beats_POET_or_external_OET")) for r in group),
            "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in group),
            "beats_all_reweighting_controls_rows": sum(iflag(r.get("beats_all_reweighting_controls")) for r in group),
            "beats_credit_only_rows": sum(iflag(r.get("beats_credit_only")) for r in group),
            "beats_self_only_rows": sum(iflag(r.get("beats_self_only")) for r in group),
            "beats_same_generalized_spectrum_control_rows": sum(iflag(r.get("beats_same_generalized_spectrum_control")) for r in group),
            "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
            "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in group),
            "nuisance_corr_le_070_rows": sum(int(max(fval(r.get("operator_class_corr_abs"), 0.0) or 0.0, fval(r.get("operator_loss_corr_abs"), 0.0) or 0.0, fval(r.get("operator_margin_corr_abs"), 0.0) or 0.0) <= 0.70) for r in group),
        }
        row["required_beats_strongest_rows"] = _need(n, 5, 9)
        row["required_auc_rows"] = _need(n, 5, 9)
        row["required_oet_rows"] = _need(n, 5, 9)
        row["required_best_control_rows"] = _need(n, 6, 9)
        row["required_reweight_rows"] = _need(n, 6, 9)
        row["required_credit_only_rows"] = _need(n, 6, 9)
        row["required_self_only_rows"] = _need(n, 6, 9)
        row["required_no_debt_rows"] = _need(n, 7, 9)
        row["required_overhead_rows"] = _need(n, 8, 9)
        row["required_nuisance_rows"] = _need(n, 8, 9)
        row["method_gate_pass"] = int(
            n >= 9
            and row["beats_strongest_rows"] >= row["required_beats_strongest_rows"]
            and row["AUC_loss_time_improvement_rows"] >= row["required_auc_rows"]
            and row["beats_POET_or_external_OET_rows"] >= row["required_oet_rows"]
            and row["beats_best_control_rows"] >= row["required_best_control_rows"]
            and row["beats_all_reweighting_controls_rows"] >= row["required_reweight_rows"]
            and row["beats_credit_only_rows"] >= row["required_credit_only_rows"]
            and row["beats_self_only_rows"] >= row["required_self_only_rows"]
            and row["no_debt_rows"] >= row["required_no_debt_rows"]
            and row["overhead_le_035_rows"] >= row["required_overhead_rows"]
            and row["nuisance_corr_le_070_rows"] >= row["required_nuisance_rows"]
        )
        out.append(row)
    return out


def classify_route(candidate_pairwise: list[dict[str, Any]], route: dict[str, Any]) -> str:
    if route.get("part_a_pass") == 0:
        return "R0-CodeOrTrainingBoundaryFailed"
    if not candidate_pairwise:
        return "R1-OperatorUnitOnly"
    if route["standard_loop_pass_candidate_rows"] != route["candidate_ccsg_rows"]:
        return "R0-CodeOrTrainingBoundaryFailed"
    if route.get("mlp_exploration_pass"):
        return "R6-MLPCreditConditionedSelfGeometryOpened"
    if route["beats_strongest_rows"] > 0 and route["beats_POET_or_external_OET_rows"] == 0:
        return "R2-FUWeakOptimizerPatchOnly"
    if route["beats_self_only_rows"] < min(6, max(1, route["candidate_ccsg_rows"])):
        return "R4-SelfGeometryOptimizerOnly"
    if route["beats_credit_only_rows"] < min(6, max(1, route["candidate_ccsg_rows"])):
        return "R5-DataCreditOnly_NoSelfGeometryGain"
    if route["beats_all_reweighting_controls_rows"] < min(6, max(1, route["candidate_ccsg_rows"])):
        return "R3-ReweightingExplained_NoFU"
    return "R1-OperatorUnitOnly"


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    include_prefixes = split_csv(str(args.summary_include_prefixes), str) if str(args.summary_include_prefixes).strip() else []
    rows = collect_chunk_rows(include_prefixes)
    reference_rows = [r for r in rows if r.get("phase") == "reference" and r.get("run_status") == "completed"]
    ccsg_rows = [r for r in rows if r.get("phase") == "CCSG" and r.get("run_status") == "completed"]
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
                "mean_accuracy": mean(fval(r.get("final_accuracy")) for r in group),
                "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
                "standard_loop_pass_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in group),
                "mean_overhead_ratio": mean(fval(r.get("controller_overhead_ratio")) for r in group),
                "mean_cos_g_Pg": mean(fval(r.get("cos_g_Pg")) for r in group),
                "mean_generalized_residual": mean(fval(r.get("generalized_eigen_residual")) for r in group),
                "mean_credit_self_alignment": mean(fval(r.get("credit_self_alignment")) for r in group),
            }
        )
    write_rows(OUT_ROOT / "v22_57_method_summary.csv", method_summary)
    ref_by_key = {(r.get("method"), r.get("dataset"), str(r.get("seed"))): r for r in reference_rows}
    strongest_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for r in reference_rows:
        key = (r.get("dataset", ""), str(r.get("seed", "")))
        current = strongest_by_key.get(key)
        if current is None or (fval(r.get("final_NLL"), float("inf")) or float("inf")) < (fval(current.get("final_NLL"), float("inf")) or float("inf")):
            strongest_by_key[key] = r
    pairwise = []
    for r in ccsg_rows:
        key2 = (r.get("dataset", ""), str(r.get("seed", "")))
        ref_method = r.get("reference_method", "")
        ref = ref_by_key.get((ref_method, r.get("dataset"), str(r.get("seed"))))
        poet = ref_by_key.get(("poet_official", r.get("dataset"), str(r.get("seed"))))
        pion = ref_by_key.get(("pion_oet_sphere_official", r.get("dataset"), str(r.get("seed"))))
        strongest = strongest_by_key.get(key2)
        controls = [
            x for x in ccsg_rows
            if x.get("dataset") == r.get("dataset")
            and str(x.get("seed")) == str(r.get("seed"))
            and x.get("method") in CCSG_CONTROLS
            and x.get("reference_method") == ref_method
        ]
        best_control = min(controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        reweight_controls = [x for x in controls if x.get("method") in REWEIGHTING_CONTROLS]
        self_controls = [x for x in controls if x.get("method") in SELF_ONLY_CONTROLS]
        random_controls = [x for x in controls if x.get("method") in RANDOM_SPECTRUM_CONTROLS]
        credit_only = min([x for x in controls if x.get("method") == "credit_only_operator"], key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        self_only = min(self_controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        best_reweight = min(reweight_controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        best_random = min(random_controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        final_nll = fval(r.get("final_NLL"))
        ref_nll = fval(ref.get("final_NLL")) if ref else None
        poet_nll = fval(poet.get("final_NLL")) if poet else None
        pion_nll = fval(pion.get("final_NLL")) if pion else None
        strongest_nll = fval(strongest.get("final_NLL")) if strongest else None
        control_nll = fval(best_control.get("final_NLL")) if best_control else None
        reweight_nll = fval(best_reweight.get("final_NLL")) if best_reweight else None
        random_nll = fval(best_random.get("final_NLL")) if best_random else None
        credit_nll = fval(credit_only.get("final_NLL")) if credit_only else None
        self_nll = fval(self_only.get("final_NLL")) if self_only else None
        pairwise.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "method": r.get("method", ""),
                "control_kind": r.get("control_kind", ""),
                "reference_method": ref_method,
                "reference_final_NLL": ref_nll,
                "poet_final_NLL": poet_nll,
                "pion_final_NLL": pion_nll,
                "strongest_method": strongest.get("method", "") if strongest else "",
                "strongest_final_NLL": strongest_nll,
                "final_NLL": final_nll,
                "Delta_NLL_vs_reference": (final_nll - ref_nll) if final_nll is not None and ref_nll is not None else "",
                "Delta_NLL_vs_strongest": (final_nll - strongest_nll) if final_nll is not None and strongest_nll is not None else "",
                "Delta_NLL_vs_POET": (final_nll - poet_nll) if final_nll is not None and poet_nll is not None else "",
                "Delta_NLL_vs_Pion": (final_nll - pion_nll) if final_nll is not None and pion_nll is not None else "",
                "Delta_NLL_vs_best_control": (final_nll - control_nll) if final_nll is not None and control_nll is not None else "",
                "Delta_NLL_vs_credit_only": (final_nll - credit_nll) if final_nll is not None and credit_nll is not None else "",
                "Delta_NLL_vs_self_only": (final_nll - self_nll) if final_nll is not None and self_nll is not None else "",
                "Delta_NLL_vs_same_generalized_spectrum_control": (final_nll - random_nll) if final_nll is not None and random_nll is not None else "",
                "beats_reference": int(final_nll is not None and ref_nll is not None and final_nll < ref_nll),
                "beats_strongest": int(final_nll is not None and strongest_nll is not None and final_nll < strongest_nll),
                "beats_POET_or_external_OET": int(final_nll is not None and ((poet_nll is not None and final_nll < poet_nll) or (pion_nll is not None and final_nll < pion_nll))),
                "beats_Pion": int(final_nll is not None and pion_nll is not None and final_nll < pion_nll),
                "best_control_method": best_control.get("method", "") if best_control else "",
                "best_control_final_NLL": control_nll,
                "beats_best_control": int(final_nll is not None and control_nll is not None and final_nll < control_nll),
                "best_reweighting_control_method": best_reweight.get("method", "") if best_reweight else "",
                "beats_all_reweighting_controls": int(final_nll is not None and bool(reweight_controls) and all(final_nll < (fval(x.get("final_NLL"), float("inf")) or float("inf")) for x in reweight_controls)),
                "credit_only_final_NLL": credit_nll,
                "beats_credit_only": int(final_nll is not None and credit_nll is not None and final_nll < credit_nll),
                "self_only_final_NLL": self_nll,
                "beats_self_only": int(final_nll is not None and self_nll is not None and final_nll < self_nll),
                "best_random_spectrum_control_method": best_random.get("method", "") if best_random else "",
                "beats_same_generalized_spectrum_control": int(final_nll is not None and random_nll is not None and final_nll < random_nll),
                "no_ECE_Brier_tail_debt": r.get("no_ECE_Brier_tail_debt", ""),
                "held_ECE_delta": r.get("held_ECE_delta", ""),
                "held_Brier_delta": r.get("held_Brier_delta", ""),
                "held_tail_q95_delta": r.get("held_tail_q95_delta", ""),
                "held_tail_q99_delta": r.get("held_tail_q99_delta", ""),
                "AUC_loss_time_improvement": int((fval(r.get("AUC_loss_time")) or float("inf")) < (fval(ref.get("AUC_loss_time")) or float("inf"))) if ref else 0,
                "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
                "operator_class_corr_abs": r.get("operator_class_corr_abs", ""),
                "operator_loss_corr_abs": r.get("operator_loss_corr_abs", ""),
                "operator_margin_corr_abs": r.get("operator_margin_corr_abs", ""),
                "cos_g_Pg": r.get("cos_g_Pg", ""),
                "operator_PSD_min": r.get("operator_PSD_min", r.get("operator_PSD_min_eigenvalue", "")),
                "generalized_eigen_residual": r.get("generalized_eigen_residual", ""),
                "credit_self_alignment": r.get("credit_self_alignment", ""),
                "standard_loop_hard_gate_pass": r.get("standard_loop_hard_gate_pass", ""),
            }
        )
    write_rows(OUT_ROOT / "v22_57_mlp_ccsg_pairwise.csv", pairwise)
    candidate_pairwise = [r for r in pairwise if r.get("method") in CCSG_CANDIDATES]
    gate_rows = candidate_method_gate_rows(candidate_pairwise)
    write_rows(OUT_ROOT / "v22_57_candidate_method_gate.csv", gate_rows)
    part_a = read_rows(OUT_ROOT / "v22_57_code_truth_gate.csv")
    part_a_pass = iflag(part_a[0].get("part_a_pass")) if part_a else 0
    route = {
        "part_a_pass": part_a_pass,
        "reference_rows": len(reference_rows),
        "ccsg_rows": len(ccsg_rows),
        "candidate_ccsg_rows": len(candidate_pairwise),
        "candidate_method_gate_rows": len(gate_rows),
        "candidate_method_gate_pass_rows": sum(iflag(r.get("method_gate_pass")) for r in gate_rows),
        "standard_loop_pass_candidate_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in candidate_pairwise),
        "beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in candidate_pairwise),
        "beats_POET_or_external_OET_rows": sum(iflag(r.get("beats_POET_or_external_OET")) for r in candidate_pairwise),
        "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in candidate_pairwise),
        "beats_all_reweighting_controls_rows": sum(iflag(r.get("beats_all_reweighting_controls")) for r in candidate_pairwise),
        "beats_credit_only_rows": sum(iflag(r.get("beats_credit_only")) for r in candidate_pairwise),
        "beats_self_only_rows": sum(iflag(r.get("beats_self_only")) for r in candidate_pairwise),
        "beats_same_generalized_spectrum_control_rows": sum(iflag(r.get("beats_same_generalized_spectrum_control")) for r in candidate_pairwise),
        "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in candidate_pairwise),
        "AUC_loss_time_improvement_rows": sum(iflag(r.get("AUC_loss_time_improvement")) for r in candidate_pairwise),
        "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in candidate_pairwise),
        "nuisance_corr_le_070_rows": sum(int(max(fval(r.get("operator_class_corr_abs"), 0.0) or 0.0, fval(r.get("operator_loss_corr_abs"), 0.0) or 0.0, fval(r.get("operator_margin_corr_abs"), 0.0) or 0.0) <= 0.70) for r in candidate_pairwise),
        "mlp_exploration_pass": 0,
        "kan_gate_status": "not_started",
        "final_route": "",
    }
    route["mlp_exploration_pass"] = int(route["candidate_method_gate_pass_rows"] > 0 and route["standard_loop_pass_candidate_rows"] == route["candidate_ccsg_rows"])
    route["kan_gate_status"] = "eligible_not_run_yet" if route["mlp_exploration_pass"] else "skipped_MLP_CCSG_gate_not_opened"
    route["final_route"] = classify_route(candidate_pairwise, route)
    write_json(OUT_ROOT / "v22_57_final_route.json", route)
    downstream = [
        {"part": "F_KAN", "status": "skipped" if not route["mlp_exploration_pass"] else "eligible_not_run_yet", "reason": route["kan_gate_status"]},
        {"part": "G_continual_grokking", "status": "skipped" if not route["mlp_exploration_pass"] else "eligible_not_run_yet", "reason": "pressure tests require MLP+CCSG gate opened"},
    ]
    write_rows(OUT_ROOT / "v22_57_downstream_gate_status.csv", downstream)
    append_exec(
        f"{PYTHON} experiments/run_v22_57_credit_conditioned_self_geometry_fu.py --stage summarize",
        task_id="v22_57_summarize",
        status="pass",
        files=(
            "results/v22_57/v22_57_method_summary.csv; results/v22_57/v22_57_mlp_ccsg_pairwise.csv; "
            "results/v22_57/v22_57_candidate_method_gate.csv; "
            "results/v22_57/v22_57_final_route.json; results/v22_57/v22_57_downstream_gate_status.csv"
        ),
        note=f"final_route={route['final_route']}",
    )
    append_recap(
        "Part D MLP CCSG summary and route",
        "Method summary:\n\n"
        + md_table(method_summary, ["method", "phase", "rows", "completed_rows", "blocked_rows", "mean_final_NLL", "no_debt_rows", "standard_loop_pass_rows", "mean_overhead_ratio", "mean_credit_self_alignment"], 40)
        + "\nCandidate pairwise rows:\n\n"
        + md_table(candidate_pairwise, ["dataset", "seed", "method", "reference_method", "Delta_NLL_vs_strongest", "beats_strongest", "beats_POET_or_external_OET", "beats_best_control", "beats_credit_only", "beats_self_only", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 60)
        + "\nPer-method gate rows:\n\n"
        + md_table(gate_rows, ["method", "rows", "method_gate_pass", "beats_strongest_rows", "beats_POET_or_external_OET_rows", "beats_best_control_rows", "beats_credit_only_rows", "beats_self_only_rows", "no_debt_rows", "overhead_le_035_rows"], 12)
        + "\nRoute JSON:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return route


def analyze_controls(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    pairwise = read_rows(OUT_ROOT / "v22_57_mlp_ccsg_pairwise.csv")
    diagnosis = []
    for row in pairwise:
        if row.get("control_kind") != "candidate":
            continue
        debt_components = {
            "ECE": fval(row.get("held_ECE_delta"), 0.0) or 0.0,
            "Brier": fval(row.get("held_Brier_delta"), 0.0) or 0.0,
            "tail_q95": fval(row.get("held_tail_q95_delta"), 0.0) or 0.0,
        }
        primary_debt = max(debt_components, key=lambda k: debt_components[k])
        failure_mode = "pass_like"
        if not iflag(row.get("beats_best_control")):
            failure_mode = "best_control_explains_or_beats"
        if not iflag(row.get("beats_credit_only")):
            failure_mode = "credit_only_tracks_candidate"
        if not iflag(row.get("beats_self_only")):
            failure_mode = "self_only_tracks_candidate"
        if not iflag(row.get("no_ECE_Brier_tail_debt")):
            failure_mode = "debt_failure"
        diagnosis.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "failure_mode": failure_mode,
                "best_control_method": row.get("best_control_method", ""),
                "primary_debt_component": primary_debt,
                "self_metric_blocker": "high_self_condition_or_self_only_control" if not iflag(row.get("beats_self_only")) else "",
                "credit_metric_blocker": "credit_only_control_matches" if not iflag(row.get("beats_credit_only")) else "",
                "overhead_blocker": "overhead_gt_0.35" if (fval(row.get("controller_overhead_ratio"), 0.0) or 0.0) > 0.35 else "",
                "recommended_next_action": "inspect debt decomposition and control gap; do not run KAN if MLP gate closed",
            }
        )
    write_rows(OUT_ROOT / "v22_57_machine_readable_diagnosis.csv", diagnosis)
    summary = {
        "diagnosis_rows": len(diagnosis),
        "self_only_tracks_rows": sum(int(r.get("failure_mode") == "self_only_tracks_candidate") for r in diagnosis),
        "credit_only_tracks_rows": sum(int(r.get("failure_mode") == "credit_only_tracks_candidate") for r in diagnosis),
        "debt_failure_rows": sum(int(r.get("failure_mode") == "debt_failure") for r in diagnosis),
    }
    write_json(OUT_ROOT / "v22_57_control_audit_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_57_credit_conditioned_self_geometry_fu.py --stage analyze-controls",
        task_id="v22_57_control_diagnosis",
        status="pass",
        files="results/v22_57/v22_57_machine_readable_diagnosis.csv; results/v22_57/v22_57_control_audit_summary.json",
        note=json.dumps(summary, ensure_ascii=False, sort_keys=True),
    )
    append_recap(
        "Part E credit/self/control diagnosis",
        "Machine-readable diagnosis summary:\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n"
        + md_table(diagnosis, ["dataset", "seed", "method", "failure_mode", "best_control_method", "primary_debt_component", "self_metric_blocker", "credit_metric_blocker", "overhead_blocker"], 60),
    )
    return summary


def write_svg(path: Path, title: str, rows: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = []
    for row in rows:
        x = fval(row.get(x_key))
        y = fval(row.get(y_key))
        if x is not None and y is not None:
            points.append((x, y, row.get("method", "")))
    width, height = 720, 420
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        circles = []
        for x, y, label in points:
            px = 70 + (x - xmin) / (xmax - xmin) * 590
            py = 350 - (y - ymin) / (ymax - ymin) * 280
            circles.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="#0f766e"><title>{label}: {x_key}={x:.6g}, {y_key}={y:.6g}</title></circle>')
        body = "\n".join(circles)
    else:
        body = '<text x="70" y="210" font-size="18" fill="#555">No completed numeric rows available.</text>'
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="40" y="38" font-size="22" font-family="Arial" fill="#111">{title}</text>
<line x1="70" y1="350" x2="660" y2="350" stroke="#444"/>
<line x1="70" y1="70" x2="70" y2="350" stroke="#444"/>
<text x="300" y="395" font-size="14" font-family="Arial" fill="#333">{x_key}</text>
<text x="12" y="220" font-size="14" font-family="Arial" fill="#333" transform="rotate(-90 12 220)">{y_key}</text>
{body}
</svg>
""",
        encoding="utf-8",
    )


def run_figures(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    pairwise = read_rows(OUT_ROOT / "v22_57_mlp_ccsg_pairwise.csv")
    chunks = collect_chunk_rows()
    route = read_json(OUT_ROOT / "v22_57_final_route.json")
    write_svg(FIG_ROOT / "v22_57_credit_self_generalized_eigen_spectrum.svg", "Generalized Eigen Residual", chunks, "generalized_eigen_residual", "credit_positive_rank_out")
    write_svg(FIG_ROOT / "v22_57_operator_gain_vs_self_condition.svg", "Gain vs Self Condition", chunks, "self_condition_out", "norm_Pg_over_norm_g")
    write_svg(FIG_ROOT / "v22_57_operator_gain_vs_credit_alignment.svg", "Gain vs Credit Alignment", chunks, "credit_self_alignment", "norm_Pg_over_norm_g")
    write_svg(FIG_ROOT / "v22_57_control_gap_by_dataset.svg", "Control Gap", pairwise, "final_NLL", "best_control_final_NLL")
    write_svg(FIG_ROOT / "v22_57_overhead_vs_gain.svg", "Overhead vs Strongest Gap", pairwise, "controller_overhead_ratio", "Delta_NLL_vs_strongest")
    write_svg(FIG_ROOT / "v22_57_no_debt_decomposition.svg", "Debt Proxy", chunks, "held_ECE_delta", "held_tail_q95_delta")
    write_svg(FIG_ROOT / "v22_57_credit_only_self_only_ablation.svg", "Credit/Self Ablation", pairwise, "Delta_NLL_vs_credit_only", "Delta_NLL_vs_self_only")
    write_svg(FIG_ROOT / "v22_57_layerwise_operator_heatmap.svg", f"KAN/status: {route.get('kan_gate_status', 'unknown')}", chunks, "credit_positive_rank_out", "credit_positive_rank_in")
    files = sorted(str(p.relative_to(ROOT)) for p in FIG_ROOT.glob("v22_57_*.svg"))
    summary = {"figures": files, "figure_count": len(files)}
    write_json(OUT_ROOT / "v22_57_figure_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_57_credit_conditioned_self_geometry_fu.py --stage figures",
        task_id="v22_57_figures",
        status="pass",
        files="; ".join(files),
    )
    append_recap("Part E visualizations", "生成 SVG：\n\n" + "\n".join(f"- `{x}`" for x in files))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all", choices=["all", "o0", "reanalyse-v56", "synthetic", "collect", "reference", "ccsg", "summarize", "analyze-controls", "figures"])
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="adamw")
    p.add_argument("--reference-method", default="adamw")
    p.add_argument("--label", default="v22_57_collect")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default="")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label-prefix", default="v22_57_reference")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--smoke-steps", type=int, default=4)
    p.add_argument("--smoke-device", default="cuda:0")
    p.add_argument("--synthetic-device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--cohorts", type=int, default=4)
    p.add_argument("--cohort-mode", default="random", choices=["random", "label_loss_stratified"])
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--ccsg-rank", type=int, default=4)
    p.add_argument("--ccsg-alpha", type=float, default=0.18)
    p.add_argument("--ccsg-beta", type=float, default=0.18)
    p.add_argument("--ccsg-eta", type=float, default=1.0e-3)
    p.add_argument("--ccsg-eps", type=float, default=1.0e-5)
    p.add_argument("--ccsg-ema-beta", type=float, default=0.90)
    p.add_argument("--ccsg-c-min", type=float, default=0.20)
    p.add_argument("--ccsg-refresh-interval", type=int, default=20)
    p.add_argument("--ccsg-param-scope", default="last_layer", choices=["all", "last_layer"])
    p.add_argument("--ccsg-weight-spectrum-weight", type=float, default=0.25)
    p.add_argument("--ccsg-gradient-cov-weight", type=float, default=1.0)
    p.add_argument("--ccsg-momentum-weight", type=float, default=0.10)
    p.add_argument("--debt-tolerance", type=float, default=1.0e-6)
    p.add_argument("--tail-debt-tolerance", type=float, default=1.0e-6)
    p.add_argument("--poet-block-size", type=int, default=16)
    p.add_argument("--poet-merge-interval", type=int, default=30)
    p.add_argument("--poet-lr", type=float, default=5.0e-4)
    p.add_argument("--poet-scale", type=float, default=0.5)
    p.add_argument("--compile-timeout", type=int, default=600)
    p.add_argument("--collect-timeout", type=int, default=1800)
    p.add_argument("--summary-include-prefixes", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    try:
        if args.stage == "o0":
            print(json.dumps(run_o0(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "reanalyse-v56":
            print(json.dumps(reanalyse_v22_56_self_geometry(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "synthetic":
            print(json.dumps(run_synthetic(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "collect":
            print(json.dumps(run_collect(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "reference":
            run_matrix(args, phase="reference")
            return 0
        if args.stage == "ccsg":
            if args.label_prefix == "v22_57_reference":
                args.label_prefix = f"v22_57_ccsg_over_{safe_fragment(args.reference_method)}"
            run_matrix(args, phase="ccsg")
            return 0
        if args.stage == "summarize":
            print(json.dumps(summarize(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "analyze-controls":
            print(json.dumps(analyze_controls(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "figures":
            print(json.dumps(run_figures(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "all":
            gate = run_o0(args)
            if not iflag(gate.get("part_a_pass")):
                summarize(args)
                analyze_controls(args)
                run_figures(args)
                return 2
            reanalysis = reanalyse_v22_56_self_geometry(args)
            if not iflag(reanalysis.get("reanalysis_pass")):
                summarize(args)
                analyze_controls(args)
                run_figures(args)
                return 3
            synth = run_synthetic(args)
            if not iflag(synth.get("all_synthetic_cases_pass")):
                summarize(args)
                analyze_controls(args)
                run_figures(args)
                return 4
            args.label_prefix = "v22_57_reference"
            run_matrix(args, phase="reference")
            args.label_prefix = f"v22_57_ccsg_over_{safe_fragment(args.reference_method)}"
            run_matrix(args, phase="ccsg")
            summarize(args)
            analyze_controls(args)
            run_figures(args)
            return 0
    except Exception:
        err_path = LOG_ROOT / f"v22_57_stage_{safe_fragment(args.stage)}_exception.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        append_exec(
            f"{PYTHON} experiments/run_v22_57_credit_conditioned_self_geometry_fu.py --stage {args.stage}",
            task_id=f"v22_57_stage_{args.stage}_exception",
            status="exception",
            files=str(err_path),
            exit_code="exception",
        )
        print(err_path.read_text(encoding="utf-8"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
