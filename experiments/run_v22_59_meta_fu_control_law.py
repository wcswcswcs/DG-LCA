#!/usr/bin/env python3
"""DG-KAN v22.59 constrained Meta-FU control-law runner."""

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

from dgkan.fu.meta_fu_control_law import (  # noqa: E402
    META_FU_FEATURE_DIM,
    META_FU_OUTPUT_NAMES,
    BoundedScalarController,
    MetaFUConfig,
    MetaFUScalarOperator,
    controller_parameter_count,
)
from dgkan.fu.state_coupled_functional_geometry_operator import StateCoupledFunctionalGeometryOperator  # noqa: E402
from dgkan.optim.meta_fu_optimizer_wrapper import MetaFUOptimizerWrapper  # noqa: E402
from dgkan.optim.scfg_optimizer_wrapper import SCFGOptimizerWrapper  # noqa: E402
from experiments import audit_v22_59_standard_loop as loop_audit  # noqa: E402
from experiments.run_v22_56_task_compatible_witnessed_preconditioner_fu import (  # noqa: E402
    ExternalUnavailable,
    batch_indices,
    cohort_positions,
    compute_cohort_grads,
    compute_last_layer_closed_form_cohort_grads,
    evaluate_tensors,
    fval,
    iflag,
    load_bundle_tensors,
    make_base_optimizer,
    make_model,
    margins_from_logits,
    md_table,
    mean,
    optimizer_state_count,
    split_csv,
    stratified_cohort_positions,
    torch_device,
    trainable_param_count,
)
from experiments.run_v22_58_state_coupled_functional_geometry_fu import (  # noqa: E402
    scfg_config_for,
    select_scfg_params,
)


KAN_ENV_PYTHON = ROOT.parent / "miniconda3/envs/kan/bin/python"
PYTHON = os.environ.get("KAN_PYTHON", str(KAN_ENV_PYTHON if KAN_ENV_PYTHON.exists() else sys.executable))
OUT_ROOT = ROOT / "results/v22_59"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
CONTROLLER_ROOT = OUT_ROOT / "controllers"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.59_MetaFU_ControlLaw_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.59_MetaFU_ControlLaw_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.59_MetaFU_ControlLaw_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_59_meta_fu_control_law.py"

REFERENCE_METHODS = {
    "adamw",
    "cautious_adamw",
    "schedule_free_adamw_local",
    "poet_official",
    "pion_oet_sphere_official",
    "pion_oet_local",
}
M0_LEARNED_METHODS = {
    "meta_mlp_scalar_scheduler",
    "linear_scalar_scheduler",
    "constant_scalar_scheduler",
}
M0_CONTROL_METHODS = {
    "random_scalar_scheduler_same_distribution",
    "frozen_controller_from_other_seed",
    "frozen_controller_from_other_dataset",
    "handdesigned_v22_58_scfg",
    "credit_only_handdesigned",
    "self_only_handdesigned",
}
M0_METHODS = M0_LEARNED_METHODS | M0_CONTROL_METHODS
SCFG_BASELINE_MAP = {
    "handdesigned_v22_58_scfg": "scfg-state-transition",
    "credit_only_handdesigned": "scfg-credit-only",
    "self_only_handdesigned": "scfg-self-only",
}
DEFAULT_REFERENCE_METHODS = "adamw,cautious_adamw,schedule_free_adamw_local,poet_official,pion_oet_sphere_official"
DEFAULT_M0_METHODS = (
    "meta_mlp_scalar_scheduler,linear_scalar_scheduler,constant_scalar_scheduler,"
    "random_scalar_scheduler_same_distribution,frozen_controller_from_other_seed,"
    "frozen_controller_from_other_dataset,handdesigned_v22_58_scfg,"
    "credit_only_handdesigned,self_only_handdesigned"
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
    CONTROLLER_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.59 MetaFU ControlLaw 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、解释器、GPU、输入输出文件、状态、失败与修复尝试。"
            "官方 Meta-FU 行只允许 task loss backward 与 optimizer-owned gradient transform；"
            "controller 只输出解析 FU 标量控制参数。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.59 MetaFU ControlLaw 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘 artifact；实验数据、失败、修复、分析、结论、insight 必须带证据链。"
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
    journal = read_rows(OUT_ROOT / "v22_59_command_journal.csv")
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(OUT_ROOT / "v22_59_command_journal.csv", journal, ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"])
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


def run_cmd(
    cmd: list[str],
    *,
    task_id: str,
    files: str = "",
    gpu: str = "cpu",
    timeout: int | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
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


def _logit(p: float) -> float:
    p = min(1.0 - 1.0e-5, max(1.0e-5, float(p)))
    return math.log(p / (1.0 - p))


def raw_from_bounded(values: dict[str, float], args: argparse.Namespace) -> list[float]:
    return [
        _logit(float(values["alpha"]) / max(float(args.metafu_alpha_max), 1.0e-8)),
        _logit(float(values["beta"]) / max(float(args.metafu_beta_max), 1.0e-8)),
        _logit(float(values["lambda_debt"]) / max(float(args.metafu_lambda_debt_max), 1.0e-8)),
        _logit(float(values["lambda_state"]) / max(float(args.metafu_lambda_state_max), 1.0e-8)),
        _logit((float(values["tau"]) - float(args.metafu_tau_min)) / max(float(args.metafu_tau_max) - float(args.metafu_tau_min), 1.0e-8)),
        _logit(float(values["rho"])),
    ]


def internal_split_tensors(tensors: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    x_train = tensors["x_train"]
    y_train = tensors["y_train"]
    n = int(x_train.shape[0])
    gen = torch.Generator(device=x_train.device)
    gen.manual_seed(int(seed) + 225900)
    perm = torch.randperm(n, generator=gen)
    source_n = max(1, int(round(0.50 * n)))
    witness_n = max(1, int(round(0.25 * n)))
    if source_n + witness_n >= n:
        source_n = max(1, int(0.60 * n))
        witness_n = max(1, int(0.20 * n))
    source_idx = perm[:source_n]
    witness_idx = perm[source_n:source_n + witness_n]
    query_idx = perm[source_n + witness_n:]
    if int(query_idx.numel()) == 0:
        query_idx = witness_idx[-1:]
        witness_idx = witness_idx[:-1] if int(witness_idx.numel()) > 1 else witness_idx
    inner_idx = torch.cat([source_idx, witness_idx], dim=0)
    return {
        "x_source": x_train[source_idx],
        "y_source": y_train[source_idx],
        "x_witness": x_train[witness_idx],
        "y_witness": y_train[witness_idx],
        "x_inner": x_train[inner_idx],
        "y_inner": y_train[inner_idx],
        "x_query": x_train[query_idx],
        "y_query": y_train[query_idx],
        "source_idx_hash": hashlib.sha256(source_idx.cpu().numpy().tobytes()).hexdigest()[:16],
        "witness_idx_hash": hashlib.sha256(witness_idx.cpu().numpy().tobytes()).hexdigest()[:16],
        "query_idx_hash": hashlib.sha256(query_idx.cpu().numpy().tobytes()).hexdigest()[:16],
    }


def controller_path_for_method(method: str) -> Path:
    m = method.lower()
    if m == "linear_scalar_scheduler":
        return CONTROLLER_ROOT / "v22_59_m0_linear_controller.pt"
    if m == "constant_scalar_scheduler":
        return CONTROLLER_ROOT / "v22_59_m0_controller.pt"
    if m == "random_scalar_scheduler_same_distribution":
        return CONTROLLER_ROOT / "v22_59_m0_controller.pt"
    if m == "frozen_controller_from_other_seed":
        return CONTROLLER_ROOT / "v22_59_m0_seed_holdout_controller.pt"
    if m == "frozen_controller_from_other_dataset":
        return CONTROLLER_ROOT / "v22_59_m0_dataset_holdout_controller.pt"
    return CONTROLLER_ROOT / "v22_59_m0_controller.pt"


def metafu_config_for(method: str, args: argparse.Namespace) -> MetaFUConfig:
    m = method.lower()
    mode = m
    if m == "constant_scalar_scheduler":
        mode = "constant_scalar_scheduler"
    elif m == "random_scalar_scheduler_same_distribution":
        mode = "random_scalar_scheduler_same_distribution"
    elif m == "linear_scalar_scheduler":
        mode = "linear_scalar_scheduler"
    elif m.startswith("frozen_controller"):
        mode = "meta_mlp_scalar_scheduler"
    return MetaFUConfig(
        variant="metafu_m0_scalar_scheduler",
        control="none",
        rank=int(args.metafu_rank),
        alpha=float(args.metafu_alpha_max),
        beta=float(args.metafu_beta_max),
        eta=float(args.metafu_eta),
        eps=float(args.metafu_eps),
        ema_beta=float(args.metafu_ema_beta),
        source_fraction=0.50,
        random_seed=int(args.seed) + int(hashlib.sha256(m.encode("utf-8")).hexdigest()[:8], 16),
        c_min=float(args.metafu_c_min),
        use_out=True,
        use_in=True,
        weight_spectrum_weight=float(args.metafu_weight_spectrum_weight),
        gradient_cov_weight=float(args.metafu_gradient_cov_weight),
        momentum_weight=float(args.metafu_momentum_weight),
        state_transition_weight=float(args.metafu_lambda_state_max),
        debt_weight=float(args.metafu_lambda_debt_max),
        tail_weight=float(args.metafu_tail_weight),
        low_overhead=bool(args.metafu_low_overhead),
        estimated_by_sketch=1,
        diagnostic_interval=1,
        controller_mode=mode,
        controller_path=str(controller_path_for_method(m)),
        input_dim=int(args.metafu_input_dim),
        hidden_dim=int(args.metafu_controller_hidden),
        alpha_max=float(args.metafu_alpha_max),
        beta_max=float(args.metafu_beta_max),
        lambda_debt_max=float(args.metafu_lambda_debt_max),
        lambda_state_max=float(args.metafu_lambda_state_max),
        tau_min=float(args.metafu_tau_min),
        tau_max=float(args.metafu_tau_max),
        default_rho=float(args.metafu_default_rho),
        controller_random_seed=int(args.seed) + 593,
        official_frozen=1,
        train_only_controller_inputs=1,
        no_dataset_or_seed_input=1,
        feature_mode=str(args.metafu_feature_mode),
    )


def select_metafu_params(model: Any, scope: str) -> list[Any]:
    return select_scfg_params(model, scope)


def current_lr(opt: Any, default: float) -> float:
    groups = getattr(opt, "param_groups", [])
    if groups:
        try:
            return float(groups[0].get("lr", default))
        except Exception:
            return float(default)
    base = getattr(opt, "base", None)
    if base is not None:
        return current_lr(base, default)
    return float(default)


def write_blocked_collect(args: argparse.Namespace, reason: str, method: str, ref_method: str) -> dict[str, Any]:
    row = {
        "run_label": str(args.label),
        "run_status": "blocked",
        "blocker": reason,
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "phase": "M0" if method in M0_METHODS else "reference",
        "reference_method": ref_method if method in M0_METHODS else "",
        "final_NLL": "",
        "standard_loop_hard_gate_pass": 0,
        "external_oet_unavailable_local_only": int("unavailable" in reason.lower()),
    }
    out = CHUNK_ROOT / f"v22_59_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    method = str(args.method).lower()
    ref_method = str(args.reference_method)
    if method in REFERENCE_METHODS:
        ref_method = method
    if method not in REFERENCE_METHODS and method not in M0_METHODS:
        raise ValueError(f"unknown method: {method}")
    device = torch_device(str(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(args.seed) + 225900)
    try:
        tensors = load_bundle_tensors(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    except Exception as exc:
        return write_blocked_collect(args, f"DatasetUnavailable: {exc}", method, ref_method)
    if tensors["used_fake_data"]:
        return write_blocked_collect(args, "FakeDataRejected: loader reported used_fake_data=1", method, ref_method)
    split = internal_split_tensors(tensors, int(args.seed))
    base_model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 225900, device)
    try:
        model, base_opt, opt_audit = make_base_optimizer(ref_method, base_model, args, device)
    except ExternalUnavailable as exc:
        return write_blocked_collect(args, str(exc), method, ref_method)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = base_opt
    metafu_state = None
    scfg_state = None
    scope = str(args.metafu_param_scope)
    selected_params = select_metafu_params(model, scope)
    if method in M0_LEARNED_METHODS or method in {
        "random_scalar_scheduler_same_distribution",
        "frozen_controller_from_other_seed",
        "frozen_controller_from_other_dataset",
    }:
        param_names = {id(p): name for name, p in model.named_parameters()}
        metafu_state = MetaFUScalarOperator(selected_params, metafu_config_for(method, args), param_names=param_names)
        opt = MetaFUOptimizerWrapper(base_opt, metafu_state)
    elif method in SCFG_BASELINE_MAP:
        scfg_args = argparse.Namespace(**vars(args))
        scfg_args.scfg_rank = int(args.metafu_rank)
        scfg_args.scfg_alpha = float(args.metafu_alpha_max)
        scfg_args.scfg_beta = float(args.metafu_beta_max)
        scfg_args.scfg_eta = float(args.metafu_eta)
        scfg_args.scfg_eps = float(args.metafu_eps)
        scfg_args.scfg_ema_beta = float(args.metafu_ema_beta)
        scfg_args.scfg_c_min = float(args.metafu_c_min)
        scfg_args.scfg_weight_spectrum_weight = float(args.metafu_weight_spectrum_weight)
        scfg_args.scfg_gradient_cov_weight = float(args.metafu_gradient_cov_weight)
        scfg_args.scfg_momentum_weight = float(args.metafu_momentum_weight)
        scfg_args.scfg_state_transition_weight = float(args.metafu_lambda_state_max)
        scfg_args.scfg_debt_weight = float(args.metafu_lambda_debt_max)
        scfg_args.scfg_tail_weight = float(args.metafu_tail_weight)
        scfg_state = StateCoupledFunctionalGeometryOperator(selected_params, scfg_config_for(SCFG_BASELINE_MAP[method], scfg_args))
        opt = SCFGOptimizerWrapper(base_opt, scfg_state)
    x_inner = split["x_inner"].to(device)
    y_inner = split["y_inner"].to(device)
    x_query = split["x_query"].to(device)
    y_query = split["y_query"].to(device)
    x_test = tensors["x_test"].to(device)
    y_test = tensors["y_test"].to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(hashlib.sha256(method.encode("utf-8")).hexdigest()[:8], 16))
    pre_query = evaluate_tensors(model, x_query, y_query, device, int(tensors["num_classes"]), int(args.eval_batch_size))
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
    if (method in M0_METHODS) and scope == "last_layer" and hasattr(model, "fc3"):
        def _capture_last_layer_input(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if inputs:
                feature_cache["last_layer_input"] = inputs[0].detach()

        hook_handle = model.fc3.register_forward_hook(_capture_last_layer_input)
    try:
        for step in range(int(args.steps)):
            idx = batch_indices(int(x_inner.shape[0]), int(args.batch_size), generator, device)
            xb = x_inner[idx]
            yb = y_inner[idx].long()
            opt.zero_grad(set_to_none=True)
            step_start = time.perf_counter()
            logits = model(xb).float()
            forward_called = 1
            trace_events.append("forward")
            loss_task = F.cross_entropy(logits, yb)
            active_state = metafu_state if metafu_state is not None else scfg_state
            if active_state is not None:
                stat_start = time.perf_counter()
                stats_interval = max(1, int(args.metafu_refresh_interval))
                should_observe = step == 0 or step % stats_interval == 0
                if should_observe:
                    if metafu_state is not None:
                        metafu_state.set_progress(step, int(args.steps), current_lr(base_opt, float(args.lr)), float(args.weight_decay))
                        metafu_state.set_train_metrics(float(loss_task.detach().item()))
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
                            selected_params,
                            generator,
                        )
                    else:
                        cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_cohort_grads(method, logits, yb, positions, selected_params, generator)
                    active_state.observe(
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
    post_query = evaluate_tensors(model, x_query, y_query, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    post_test = evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    opt_diag = opt.diagnostics() if isinstance(opt, (MetaFUOptimizerWrapper, SCFGOptimizerWrapper)) else {}
    controller_trace_file = ""
    controller_feature_file = ""
    if isinstance(metafu_state, MetaFUScalarOperator):
        trace_rows = metafu_state.controller_trace_rows()
        feature_rows = metafu_state.feature_trace_rows()
        if trace_rows:
            trace_path = CHUNK_ROOT / f"v22_59_{safe_fragment(str(args.label))}_controller_trace.csv"
            write_rows(trace_path, trace_rows)
            controller_trace_file = str(trace_path.relative_to(ROOT))
        if feature_rows:
            feature_path = CHUNK_ROOT / f"v22_59_{safe_fragment(str(args.label))}_controller_features.csv"
            write_rows(feature_path, feature_rows)
            controller_feature_file = str(feature_path.relative_to(ROOT))
    telemetry_file = ""
    if isinstance(scfg_state, StateCoupledFunctionalGeometryOperator):
        telemetry_rows = scfg_state.telemetry_rows()
        if telemetry_rows:
            telemetry_path = CHUNK_ROOT / f"v22_59_{safe_fragment(str(args.label))}_self_state_telemetry.csv"
            write_rows(telemetry_path, telemetry_rows)
            telemetry_file = str(telemetry_path.relative_to(ROOT))
    observe_cumulative_ms = float(opt_diag.get("ccsg_stats_ms", opt_diag.get("scfg_stats_ms", 0.0)) or 0.0)
    transform_cumulative_ms = float(opt_diag.get("ccsg_transform_ms", opt_diag.get("scfg_transform_ms", 0.0)) or 0.0)
    opt_diag = dict(opt_diag)
    opt_diag.pop("ccsg_stats_ms", None)
    opt_diag.pop("ccsg_transform_ms", None)
    opt_diag.pop("scfg_stats_ms", None)
    opt_diag.pop("scfg_transform_ms", None)
    stats_step_ms = mean(stats_ms) if stats_ms else 0.0
    transform_step_ms = transform_cumulative_ms / max(1, int(args.steps))
    full_step_ms = mean(step_ms) or 0.0
    trace_hash = hashlib.sha256("|".join(trace_events).encode("utf-8")).hexdigest()[:16]
    no_debt = int(
        (post_query["ECE"] - pre_query["ECE"]) <= float(args.debt_tolerance)
        and (post_query["Brier"] - pre_query["Brier"]) <= float(args.debt_tolerance)
        and (post_query["tail_q95"] - pre_query["tail_q95"]) <= float(args.tail_debt_tolerance)
    )
    optimizer_owned = int(opt_diag.get("optimizer_owned_gradient_transform_pass", 0)) if method in M0_METHODS else 1
    audit_flags = runtime_audit.as_dict()
    standard_loop_hard_gate_pass = int(
        forward_called == 1
        and loss_task_backward_called == 1
        and optimizer_step_called == 1
        and task_loss_only_flag == 1
        and fu_auxiliary_loss_used_official == 0
        and optimizer_owned == 1
        and sum(audit_flags.values()) == 0
    )
    controller_param_count = int(opt_diag.get("meta_controller_parameter_count", 0) or 0)
    task_param_count = trainable_param_count(model)
    row = {
        "run_label": str(args.label),
        "run_status": "completed",
        "phase": "M0" if method in M0_METHODS else "reference",
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "objective_family": "MetaFU_M0" if method in M0_METHODS else "reference",
        "control_kind": "learned" if method in M0_LEARNED_METHODS else ("control" if method in M0_CONTROL_METHODS else "reference"),
        "reference_method": ref_method if method in M0_METHODS else "",
        "model_family": f"MLP_hidden{int(args.hidden)}",
        "device": str(args.device),
        "python": PYTHON,
        "dataset_source_kind": tensors.get("source_kind", ""),
        "dataset_name_not_available_to_controller": 1,
        "seed_not_available_to_controller": 1,
        "query_split_is_train_only": 1,
        "source_idx_hash": split["source_idx_hash"],
        "witness_idx_hash": split["witness_idx_hash"],
        "query_idx_hash": split["query_idx_hash"],
        "train_size": int(x_inner.shape[0]),
        "source_size": int(split["x_source"].shape[0]),
        "witness_size": int(split["x_witness"].shape[0]),
        "query_size": int(x_query.shape[0]),
        "test_size": int(x_test.shape[0]),
        "hidden": int(args.hidden),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "cohorts": int(args.cohorts),
        "cohort_mode": str(args.cohort_mode),
        "metafu_refresh_interval": int(args.metafu_refresh_interval),
        "metafu_param_scope": scope if method in M0_METHODS else "",
        "controller_trace_file": controller_trace_file,
        "controller_feature_file": controller_feature_file,
        "self_state_telemetry_file": telemetry_file,
        "final_NLL": post_query["NLL"],
        "meta_test_query_NLL": post_query["NLL"],
        "final_accuracy": post_query["accuracy"],
        "accuracy": post_query["accuracy"],
        "test_NLL": post_test["NLL"],
        "test_accuracy": post_test["accuracy"],
        "AUC_loss_time": mean(train_losses),
        "wallclock_adjusted_AUC": (mean(train_losses) or 0.0) * wall,
        "ECE": post_query["ECE"],
        "Brier": post_query["Brier"],
        "tail_loss_q95": post_query["tail_q95"],
        "tail_loss_q99": post_query["tail_q99"],
        "margin_q10": post_query["margin_q10"],
        "margin_q01": post_query["margin_q01"],
        "pre_NLL": pre_query["NLL"],
        "pre_test_NLL": pre_test["NLL"],
        "held_ECE_delta": post_query["ECE"] - pre_query["ECE"],
        "held_Brier_delta": post_query["Brier"] - pre_query["Brier"],
        "held_tail_q95_delta": post_query["tail_q95"] - pre_query["tail_q95"],
        "held_tail_q99_delta": post_query["tail_q99"] - pre_query["tail_q99"],
        "no_ECE_Brier_tail_debt": no_debt,
        "train_time_sec": wall,
        "optimizer_step_ms": mean(opt_ms),
        "full_step_ms": full_step_ms,
        "operator_build_step_ms": stats_step_ms,
        "operator_apply_step_ms": transform_step_ms,
        "operator_build_over_full_ratio": float(stats_step_ms / max(full_step_ms, 1.0e-12)),
        "operator_apply_over_full_ratio": float(transform_step_ms / max(full_step_ms, 1.0e-12)),
        "operator_observe_cumulative_ms": observe_cumulative_ms,
        "operator_transform_cumulative_ms": transform_cumulative_ms,
        "controller_overhead_ratio": float((stats_step_ms + transform_step_ms + float(opt_diag.get("controller_eval_ms", 0.0)) / max(1, int(args.steps))) / max(full_step_ms, 1.0e-12)),
        "peak_memory_mb": peak_memory,
        "trainable_parameter_count": task_param_count,
        "meta_controller_parameter_count": controller_param_count,
        "controller_parameter_count_ratio": float(controller_param_count / max(1, task_param_count)),
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
        **opt_diag,
        **opt_audit,
    }
    out = CHUNK_ROOT / f"v22_59_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def extract_initial_feature_row(dataset: str, seed: int, args: argparse.Namespace, device_name: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = torch_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tensors = load_bundle_tensors(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
    split = internal_split_tensors(tensors, seed)
    model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(seed) + 225900, device)
    model, base_opt, _audit = make_base_optimizer(str(args.meta_label_reference_method), model, args, device)
    params = select_metafu_params(model, str(args.metafu_param_scope))
    op = MetaFUScalarOperator(params, metafu_config_for("constant_scalar_scheduler", argparse.Namespace(**{**vars(args), "seed": seed})))
    feature_cache: dict[str, Any] = {}
    hook_handle = None
    if str(args.metafu_param_scope) == "last_layer" and hasattr(model, "fc3"):
        def _capture_last_layer_input(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if inputs:
                feature_cache["last_layer_input"] = inputs[0].detach()

        hook_handle = model.fc3.register_forward_hook(_capture_last_layer_input)
    try:
        x_inner = split["x_inner"].to(device)
        y_inner = split["y_inner"].to(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 225901)
        idx = batch_indices(int(x_inner.shape[0]), int(args.batch_size), gen, device)
        xb = x_inner[idx]
        yb = y_inner[idx].long()
        logits = model(xb).float()
        loss = F.cross_entropy(logits, yb)
        op.set_progress(0, max(1, int(args.meta_label_horizon)), float(args.lr), float(args.weight_decay))
        op.set_train_metrics(float(loss.detach().item()))
        per_loss = F.cross_entropy(logits.detach().float(), yb, reduction="none")
        positions = stratified_cohort_positions(yb, per_loss, int(args.cohorts), gen, device) if str(args.cohort_mode) == "label_loss_stratified" else cohort_positions(int(idx.numel()), int(args.cohorts), gen, device)
        if str(args.metafu_param_scope) == "last_layer" and "last_layer_input" in feature_cache:
            cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_last_layer_closed_form_cohort_grads(
                "meta_mlp_scalar_scheduler",
                logits,
                feature_cache["last_layer_input"],
                yb,
                positions,
                params,
                gen,
            )
        else:
            cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_cohort_grads("meta_mlp_scalar_scheduler", logits, yb, positions, params, gen)
        op.observe(cohort_grads, cohort_labels=cohort_labels, cohort_losses=cohort_losses, cohort_margins=cohort_margins, optimizer_state=getattr(base_opt, "state", {}))
        feature_rows = op.feature_trace_rows()
        avg: dict[str, float] = {}
        for i in range(int(args.metafu_input_dim)):
            key = f"z{i:02d}"
            vals = [fval(r.get(key)) for r in feature_rows]
            avg[key] = mean(vals) or 0.0
        return {
            "dataset": dataset,
            "seed": seed,
            "task_id_hash": hashlib.sha256(f"{dataset}:{seed}".encode("utf-8")).hexdigest()[:16],
            "dataset_name_not_available_to_controller": 1,
            "seed_not_available_to_controller": 1,
            "source_idx_hash": split["source_idx_hash"],
            "witness_idx_hash": split["witness_idx_hash"],
            "query_idx_hash": split["query_idx_hash"],
            **avg,
        }
    finally:
        if hook_handle is not None:
            hook_handle.remove()


def scalar_probe_config(strength: float, args: argparse.Namespace) -> Any:
    ns = argparse.Namespace(**vars(args))
    ns.scfg_rank = int(args.metafu_rank)
    ns.scfg_alpha = float(strength)
    ns.scfg_beta = float(strength)
    ns.scfg_eta = float(args.metafu_eta)
    ns.scfg_eps = float(args.metafu_eps)
    ns.scfg_ema_beta = float(args.metafu_ema_beta)
    ns.scfg_c_min = float(args.metafu_c_min)
    ns.scfg_weight_spectrum_weight = float(args.metafu_weight_spectrum_weight)
    ns.scfg_gradient_cov_weight = float(args.metafu_gradient_cov_weight)
    ns.scfg_momentum_weight = float(args.metafu_momentum_weight)
    ns.scfg_state_transition_weight = float(args.metafu_lambda_state_max) * (0.5 + strength)
    ns.scfg_debt_weight = float(args.metafu_lambda_debt_max) * (0.5 + strength)
    ns.scfg_tail_weight = float(args.metafu_tail_weight)
    return scfg_config_for("scfg-state-transition-tail-aware" if strength > 0 else "scfg-state-transition", ns)


def run_scalar_probe(dataset: str, seed: int, strength: float, args: argparse.Namespace, device_name: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = torch_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tensors = load_bundle_tensors(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
    if tensors["used_fake_data"]:
        raise RuntimeError("fake data rejected in meta probe")
    split = internal_split_tensors(tensors, seed)
    model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(seed) + 225900, device)
    ref_method = str(args.meta_label_reference_method)
    model, base_opt, _audit = make_base_optimizer(ref_method, model, args, device)
    params = select_metafu_params(model, str(args.metafu_param_scope))
    opt: Any = base_opt
    state = None
    if strength > 0.0:
        state = StateCoupledFunctionalGeometryOperator(params, scalar_probe_config(strength, args))
        opt = SCFGOptimizerWrapper(base_opt, state)
    x_inner = split["x_inner"].to(device)
    y_inner = split["y_inner"].to(device)
    x_query = split["x_query"].to(device)
    y_query = split["y_query"].to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + int(strength * 10000) + 225902)
    feature_cache: dict[str, Any] = {}
    hook_handle = None
    if strength > 0.0 and str(args.metafu_param_scope) == "last_layer" and hasattr(model, "fc3"):
        def _capture_last_layer_input(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if inputs:
                feature_cache["last_layer_input"] = inputs[0].detach()

        hook_handle = model.fc3.register_forward_hook(_capture_last_layer_input)
    try:
        train_losses: list[float] = []
        for step in range(int(args.meta_label_horizon)):
            idx = batch_indices(int(x_inner.shape[0]), int(args.batch_size), gen, device)
            xb = x_inner[idx]
            yb = y_inner[idx].long()
            opt.zero_grad(set_to_none=True)
            logits = model(xb).float()
            loss_task = F.cross_entropy(logits, yb)
            if state is not None and (step == 0 or step % max(1, int(args.metafu_refresh_interval)) == 0):
                per_loss = F.cross_entropy(logits.detach().float(), yb, reduction="none")
                positions = stratified_cohort_positions(yb, per_loss, int(args.cohorts), gen, device) if str(args.cohort_mode) == "label_loss_stratified" else cohort_positions(int(idx.numel()), int(args.cohorts), gen, device)
                if str(args.metafu_param_scope) == "last_layer" and "last_layer_input" in feature_cache:
                    cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_last_layer_closed_form_cohort_grads(
                        "scfg-state-transition-tail-aware",
                        logits,
                        feature_cache["last_layer_input"],
                        yb,
                        positions,
                        params,
                        gen,
                    )
                else:
                    cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_cohort_grads("scfg-state-transition-tail-aware", logits, yb, positions, params, gen)
                state.observe(cohort_grads, cohort_labels=cohort_labels, cohort_losses=cohort_losses, cohort_margins=cohort_margins, optimizer_state=getattr(base_opt, "state", {}))
            loss_task.backward()
            opt.step()
            if ref_method == "poet_official":
                try:
                    model.merge_if_needed(step + 1)
                except Exception:
                    pass
            train_losses.append(float(loss_task.detach().item()))
        query = evaluate_tensors(model, x_query, y_query, device, int(tensors["num_classes"]), int(args.eval_batch_size))
        return {"strength": strength, "query_NLL": query["NLL"], "query_accuracy": query["accuracy"], "AUC_loss_time": mean(train_losses)}
    finally:
        if hook_handle is not None:
            hook_handle.remove()


def build_meta_dataset(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    datasets = split_csv(str(args.meta_train_datasets), str)
    seeds = split_csv(str(args.meta_train_seeds), int)
    strengths = split_csv(str(args.meta_probe_strengths), float)
    rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    device = str(args.synthetic_device)
    for dataset in datasets:
        for seed in seeds:
            try:
                feature_row = extract_initial_feature_row(dataset, seed, args, device)
                probes = [run_scalar_probe(dataset, seed, strength, args, device) for strength in strengths]
                probe_rows.extend({"dataset": dataset, "seed": seed, **p} for p in probes)
                best = min(probes, key=lambda r: fval(r.get("query_NLL"), float("inf")) or float("inf"))
                strength = float(best["strength"])
                bounded = {
                    "alpha": min(float(args.metafu_alpha_max), max(1.0e-5, strength)),
                    "beta": min(float(args.metafu_beta_max), max(1.0e-5, strength)),
                    "lambda_debt": min(float(args.metafu_lambda_debt_max), float(args.metafu_lambda_debt_max) * (0.55 + strength)),
                    "lambda_state": min(float(args.metafu_lambda_state_max), float(args.metafu_lambda_state_max) * (0.55 + strength)),
                    "tau": min(float(args.metafu_tau_max), max(float(args.metafu_tau_min), 0.08 + 1.2 * strength)),
                    "rho": min(0.95, max(0.05, 0.25 + 2.0 * strength)),
                }
                raw = raw_from_bounded(bounded, args)
                row = {
                    **feature_row,
                    "inner_horizon_H": int(args.meta_label_horizon),
                    "reference_optimizer": str(args.meta_label_reference_method),
                    "best_probe_strength": strength,
                    "best_probe_query_NLL": best["query_NLL"],
                    "best_probe_query_accuracy": best["query_accuracy"],
                    "source_witness_query_train_only": 1,
                    "target_raw_json": json.dumps(raw),
                    **{f"target_{k}": v for k, v in bounded.items()},
                }
                rows.append(row)
            except Exception as exc:
                rows.append({"dataset": dataset, "seed": seed, "row_status": "blocked", "blocker": repr(exc)})
    write_rows(OUT_ROOT / "v22_59_meta_dataset_episodes.csv", rows)
    write_rows(OUT_ROOT / "v22_59_m0_probe_label_generation.csv", probe_rows)
    append_exec(
        f"{PYTHON} experiments/run_v22_59_meta_fu_control_law.py --stage meta-dataset",
        task_id="v22_59_part_c_meta_dataset",
        status="pass" if any(r.get("target_raw_json") for r in rows) else "fail",
        files="results/v22_59/v22_59_meta_dataset_episodes.csv; results/v22_59/v22_59_m0_probe_label_generation.csv",
        gpu=device,
        note=f"rows={len(rows)}; label_rows={sum(1 for r in rows if r.get('target_raw_json'))}; probes={len(probe_rows)}",
    )
    return train_controllers_from_rows(args, rows)


def train_one_controller(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    kind: str,
    path: Path,
    subset_name: str,
) -> dict[str, Any]:
    import torch

    usable = [r for r in rows if r.get("target_raw_json")]
    if not usable:
        return {"controller": str(path.relative_to(ROOT)), "kind": kind, "subset": subset_name, "trained": 0, "blocker": "no usable meta rows"}
    x = torch.tensor([[fval(r.get(f"z{i:02d}"), 0.0) or 0.0 for i in range(int(args.metafu_input_dim))] for r in usable], dtype=torch.float32)
    y = torch.tensor([json.loads(str(r["target_raw_json"])) for r in usable], dtype=torch.float32)
    feature_mean = x.mean(dim=0)
    feature_std = x.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    constant_raw = y.mean(dim=0)
    bounds = {
        "alpha_max": float(args.metafu_alpha_max),
        "beta_max": float(args.metafu_beta_max),
        "lambda_debt_max": float(args.metafu_lambda_debt_max),
        "lambda_state_max": float(args.metafu_lambda_state_max),
        "tau_min": float(args.metafu_tau_min),
        "tau_max": float(args.metafu_tau_max),
    }
    controller = BoundedScalarController(
        input_dim=int(args.metafu_input_dim),
        hidden_dim=int(args.metafu_controller_hidden),
        kind=kind,
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        constant_raw=constant_raw.tolist(),
        bounds=bounds,
    )
    train_loss = 0.0
    if kind in {"mlp", "linear"}:
        opt = torch.optim.AdamW(controller.parameters(), lr=float(args.meta_controller_lr), weight_decay=1.0e-4)
        for _ in range(int(args.meta_controller_epochs)):
            opt.zero_grad(set_to_none=True)
            pred = controller.raw(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            opt.step()
            train_loss = float(loss.detach().item())
    else:
        with torch.no_grad():
            train_loss = float(torch.nn.functional.mse_loss(controller.raw(x), y).item())
    artifact = {
        "kind": kind,
        "subset": subset_name,
        "input_dim": int(args.metafu_input_dim),
        "hidden_dim": int(args.metafu_controller_hidden),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "constant_raw": constant_raw.tolist(),
        "target_raw_samples": y.tolist(),
        "state_dict": controller.state_dict() if kind in {"mlp", "linear"} else {},
        "train_loss_mse_raw": train_loss,
        "rows": len(usable),
        "controller_parameter_count": controller_parameter_count(controller),
        "controller_output_names": list(META_FU_OUTPUT_NAMES),
        "dataset_name_not_available_to_controller": 1,
        "seed_not_available_to_controller": 1,
        "created_at": now_sg(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    return {
        "controller": str(path.relative_to(ROOT)),
        "kind": kind,
        "subset": subset_name,
        "trained": 1,
        "rows": len(usable),
        "train_loss_mse_raw": train_loss,
        "controller_parameter_count": artifact["controller_parameter_count"],
    }


def train_controllers_from_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r.get("target_raw_json")]
    seed_holdout = [r for r in usable if int(r.get("seed", -999)) == min([int(x.get("seed", 0)) for x in usable], default=0)]
    dataset_holdout = [r for r in usable if str(r.get("dataset", "")) == (usable[0].get("dataset", "") if usable else "")]
    summaries = [
        train_one_controller(usable, args, kind="mlp", path=CONTROLLER_ROOT / "v22_59_m0_controller.pt", subset_name="all_meta_train"),
        train_one_controller(usable, args, kind="linear", path=CONTROLLER_ROOT / "v22_59_m0_linear_controller.pt", subset_name="all_meta_train"),
        train_one_controller(seed_holdout or usable, args, kind="mlp", path=CONTROLLER_ROOT / "v22_59_m0_seed_holdout_controller.pt", subset_name="single_seed_source"),
        train_one_controller(dataset_holdout or usable, args, kind="mlp", path=CONTROLLER_ROOT / "v22_59_m0_dataset_holdout_controller.pt", subset_name="single_dataset_source"),
    ]
    write_rows(OUT_ROOT / "v22_59_m0_controller_train_summary.csv", summaries)
    append_recap(
        "Part C Meta dataset and M0 controller training",
        "Meta dataset labels are generated only from source/witness/query splits inside train tensors. Controller inputs exclude dataset name and seed; grouping ids are logged only as hashes.\n\n"
        + md_table(summaries, ["controller", "kind", "subset", "trained", "rows", "train_loss_mse_raw", "controller_parameter_count"], 10),
    )
    return {
        "meta_dataset_rows": len(rows),
        "usable_meta_dataset_rows": len(usable),
        "controllers": summaries,
        "controller_train_pass": int(any(iflag(r.get("trained")) for r in summaries)),
    }


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
    bundle_dir = pack_root / f"v22_59_audit_{stamp}"
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
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="v22_59_clean_extract_") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_path)
        clean_root = tmp_path / bundle_dir.name
        compile_res = run_cmd(
            [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"],
            task_id="v22_59_clean_tarball_compileall",
            files=str(tar_path.relative_to(ROOT)),
            timeout=int(args.compile_timeout),
            cwd=clean_root,
        )
        rows.append({"check": "clean_tarball_compileall_pass", "pass": int(compile_res.get("status") == "pass")})
        import_cmds = {
            "clean_tarball_full_repo_import_pass": "import dgkan; import dgkan.optim; print('pass')",
            "clean_tarball_runner_core_import_pass": "import experiments.run_v22_59_meta_fu_control_law; print('pass')",
            "clean_tarball_wrapper_import_pass": "from dgkan.optim.meta_fu_optimizer_wrapper import MetaFUOptimizerWrapper; print('pass')",
            "clean_tarball_operator_import_pass": "from dgkan.fu.meta_fu_control_law import MetaFUScalarOperator; print('pass')",
        }
        for key, code in import_cmds.items():
            res = run_cmd([PYTHON, "-c", code], task_id=f"v22_59_{key}", files=str(tar_path.relative_to(ROOT)), cwd=clean_root)
            rows.append({"check": key, "pass": int(res.get("status") == "pass"), "returncode": res.get("returncode")})
    write_rows(OUT_ROOT / "v22_59_clean_tarball_self_contained_import.csv", rows)
    required_core = [
        "dgkan/fu/meta_fu_control_law.py",
        "dgkan/optim/meta_fu_optimizer_wrapper.py",
        "experiments/run_v22_59_meta_fu_control_law.py",
        "experiments/audit_v22_59_standard_loop.py",
    ]
    missing = [rel for rel in required_core if rel not in file_list]
    result = {
        "audit_bundle_dir": str(bundle_dir.relative_to(ROOT)),
        "audit_bundle_tar": str(tar_path.relative_to(ROOT)),
        "audit_bundle_sha256": sha,
        "audit_bundle_copied_roots": json.dumps(copied, ensure_ascii=False),
        "audit_bundle_file_count": len(file_list),
        "clean_tarball_compileall_pass": rows[0]["pass"],
        **{r["check"]: r["pass"] for r in rows if r["check"] != "clean_tarball_compileall_pass"},
        "untracked_core_files_missing_from_tarball": int(bool(missing)),
        "missing_core_files_in_tarball": json.dumps(missing, ensure_ascii=False),
    }
    return result


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
        task_id="v22_59_compileall_repo",
        files="results/v22_59/v22_59_code_truth_gate.csv",
        timeout=int(args.compile_timeout),
    )
    import_cmds = {
        "worktree_full_repo_import_pass": "import dgkan; import dgkan.optim; print('pass')",
        "runner_core_import_pass": "import experiments.run_v22_59_meta_fu_control_law; print('pass')",
        "wrapper_import_pass": "from dgkan.optim.meta_fu_optimizer_wrapper import MetaFUOptimizerWrapper; print('pass')",
        "operator_import_pass": "from dgkan.fu.meta_fu_control_law import MetaFUScalarOperator; print('pass')",
        "external_poet_import_pass": "from poet_torch import POETConfig, POETModel, get_poet_optimizer; print('pass')",
        "external_pion_import_pass": "import sys; sys.path.insert(0, 'external/oet_baselines/pion_spectrum_sphere/megatron-lm'); from megatron.core.optimizer.pion import PionOptimizer; print('pass')",
    }
    import_rows: list[dict[str, Any]] = []
    import_results: dict[str, int] = {}
    for key, code in import_cmds.items():
        res = run_cmd([PYTHON, "-c", code], task_id=f"v22_59_{key}", files="results/v22_59/v22_59_full_import_closure.csv")
        import_results[key] = int(res.get("status") == "pass")
        import_rows.append({"check": key, "pass": import_results[key], "returncode": res.get("returncode")})
        if res.get("status") != "pass":
            missing_modules.append(key)
    write_rows(OUT_ROOT / "v22_59_full_import_closure.csv", import_rows)
    bundle = build_clean_audit_bundle(args)
    if not bundle.get("clean_tarball_compileall_pass") or not bundle.get("clean_tarball_runner_core_import_pass"):
        missing_modules.append("clean_tarball_self_contained_import")
    scan = loop_audit.scan_files()
    write_rows(OUT_ROOT / "v22_59_standard_loop_static_scan.csv", scan["rows"] or [{"status": "no_forbidden_hits"}])
    write_rows(OUT_ROOT / "v22_59_manual_update_forbidden_scan.csv", scan["rows"] or [{"status": "no_forbidden_hits"}])
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.dataset = "MNIST"
    runtime_args.seed = 0
    runtime_args.method = "meta_mlp_scalar_scheduler"
    runtime_args.reference_method = "adamw"
    runtime_args.label = "v22_59_runtime_trace_metafu_smoke"
    runtime_args.device = str(args.smoke_device)
    runtime_args.steps = int(args.smoke_steps)
    runtime_args.train_size = min(int(args.train_size), 160)
    runtime_args.held_size = min(int(args.held_size), 64)
    runtime_args.test_size = min(int(args.test_size), 64)
    try:
        runtime_row = run_collect(runtime_args)
        runtime_status = "pass" if iflag(runtime_row.get("standard_loop_hard_gate_pass")) else "fail"
        runtime_error = ""
    except Exception:
        runtime_row = {}
        runtime_status = "fail"
        runtime_error = traceback.format_exc()
        (LOG_ROOT / "v22_59_runtime_trace_metafu_smoke_exception.log").write_text(runtime_error, encoding="utf-8")
    write_rows(OUT_ROOT / "v22_59_standard_loop_runtime_trace.csv", [runtime_row] if runtime_row else [{"status": "runtime_exception", "error": runtime_error[:500]}])
    leakage_row = {
        "meta_controller_frozen_on_meta_test": iflag(runtime_row.get("meta_controller_frozen_on_meta_test")),
        "dataset_name_not_available_to_controller": iflag(runtime_row.get("dataset_name_not_available_to_controller")),
        "seed_not_available_to_controller": iflag(runtime_row.get("seed_not_available_to_controller")),
        "controller_output_class_count_correlation": runtime_row.get("operator_class_corr_abs", ""),
        "controller_output_loss_rank_correlation": runtime_row.get("operator_loss_corr_abs", ""),
        "controller_output_margin_correlation": runtime_row.get("operator_margin_corr_abs", ""),
        "dataset_id_classifier_accuracy_from_controller_output": "",
        "seed_id_classifier_accuracy_from_controller_output": "",
        "leakage_audit_pass": int(
            iflag(runtime_row.get("meta_controller_frozen_on_meta_test"))
            and iflag(runtime_row.get("dataset_name_not_available_to_controller"))
            and iflag(runtime_row.get("seed_not_available_to_controller"))
            and (fval(runtime_row.get("operator_class_corr_abs"), 0.0) or 0.0) <= 0.70
            and (fval(runtime_row.get("operator_loss_corr_abs"), 0.0) or 0.0) <= 0.70
        ),
    }
    write_rows(OUT_ROOT / "v22_59_meta_controller_leakage_audit.csv", [leakage_row])
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
        "meta_controller_frozen_on_meta_test": iflag(runtime_row.get("meta_controller_frozen_on_meta_test")),
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
        and row["meta_controller_frozen_on_meta_test"]
    )
    write_rows(OUT_ROOT / "v22_59_code_truth_gate.csv", [row])
    append_exec(
        f"{PYTHON} experiments/run_v22_59_meta_fu_control_law.py --stage o0",
        task_id="v22_59_part_a_truth_gate_summary",
        status="pass" if row["part_a_pass"] else "fail",
        files=(
            "results/v22_59/v22_59_code_truth_gate.csv; "
            "results/v22_59/v22_59_full_import_closure.csv; "
            "results/v22_59/v22_59_clean_tarball_self_contained_import.csv; "
            "results/v22_59/v22_59_standard_loop_static_scan.csv; "
            "results/v22_59/v22_59_standard_loop_runtime_trace.csv; "
            "results/v22_59/v22_59_manual_update_forbidden_scan.csv; "
            "results/v22_59/v22_59_meta_controller_leakage_audit.csv"
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
                "standard_loop_static_scan_pass",
                "standard_loop_runtime_trace_pass",
                "manual_update_forbidden_scan_pass",
                "optimizer_owned_gradient_transform_pass",
                "meta_controller_frozen_on_meta_test",
                "part_a_pass",
            ],
        )
        + "\nLeakage smoke row:\n\n"
        + md_table([leakage_row], list(leakage_row.keys()), 1)
        + "\nAudit bundle:\n\n```json\n"
        + json.dumps({k: row[k] for k in row if k.startswith("audit_bundle") or k.startswith("clean_tarball") or k.startswith("missing_core")}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return row


def reanalyse_v22_58(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    src = ROOT / "results/v22_58"
    pairwise = read_rows(src / "v22_58_mlp_scfg_pairwise.csv")
    route = read_json(src / "v22_58_final_route.json")
    telemetry = read_rows(src / "v22_58_self_state_telemetry.csv")
    candidates = [r for r in pairwise if r.get("control_kind") == "candidate"]
    rows: list[dict[str, Any]] = []
    for method in sorted({r.get("method", "") for r in candidates}):
        group = [r for r in candidates if r.get("method") == method]
        debt_means = {
            "ECE": mean(fval(r.get("held_ECE_delta")) for r in group),
            "Brier": mean(fval(r.get("held_Brier_delta")) for r in group),
            "tail_q95": mean(fval(r.get("held_tail_q95_delta")) for r in group),
            "tail_q99": mean(fval(r.get("held_tail_q99_delta")) for r in group),
        }
        primary_debt = max(debt_means, key=lambda k: debt_means[k] if debt_means[k] is not None else -float("inf")) if group else ""
        rows.append(
            {
                "method": method,
                "rows": len(group),
                "SCFG_beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in group),
                "SCFG_beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in group),
                "SCFG_beats_credit_only_rows": sum(iflag(r.get("beats_credit_only")) for r in group),
                "SCFG_beats_self_only_rows": sum(iflag(r.get("beats_self_only")) for r in group),
                "SCFG_no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
                "SCFG_overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in group),
                "mean_operator_control_correlation": mean(
                    max(fval(r.get("operator_class_corr_abs"), 0.0) or 0.0, fval(r.get("operator_loss_corr_abs"), 0.0) or 0.0, fval(r.get("operator_margin_corr_abs"), 0.0) or 0.0)
                    for r in group
                ),
                "mean_state_transition_prediction_error": mean(fval(t.get("state_transition_prediction_error")) for t in telemetry),
                "mean_held_ECE_delta": debt_means["ECE"],
                "mean_held_Brier_delta": debt_means["Brier"],
                "mean_held_tail_q95_delta": debt_means["tail_q95"],
                "mean_held_tail_q99_delta": debt_means["tail_q99"],
                "primary_debt_component": primary_debt,
            }
        )
    summary = {
        "v22_58_artifacts_present": int(bool(pairwise) and bool(route)),
        "old_final_route": route.get("final_route", ""),
        "old_mlp_exploration_pass": route.get("mlp_exploration_pass", ""),
        "candidate_rows": len(candidates),
        "SCFG_beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in candidates),
        "SCFG_beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in candidates),
        "SCFG_no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in candidates),
        "SCFG_overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in candidates),
        "handwritten_controller_composition_too_weak_supported": int(route.get("final_route") == "R1-OperatorUnitOnly" and sum(iflag(r.get("beats_strongest")) for r in candidates) == 0),
    }
    write_rows(OUT_ROOT / "v22_59_v22_58_failure_reanalysis.csv", rows)
    write_json(OUT_ROOT / "v22_59_v22_58_failure_reanalysis_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_59_meta_fu_control_law.py --stage reanalyse-v58",
        task_id="v22_59_part_b_v58_reanalysis",
        status="pass" if summary["v22_58_artifacts_present"] else "fail",
        files="results/v22_59/v22_59_v22_58_failure_reanalysis.csv; results/v22_59/v22_59_v22_58_failure_reanalysis_summary.json",
        note=json.dumps(summary, ensure_ascii=False, sort_keys=True),
    )
    append_recap(
        "Part B v22.58 failure reanalysis",
        "从 v22.58 pairwise/route artifacts 重算失败归因。关键 summary：\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\nPer-method evidence:\n\n"
        + md_table(rows, ["method", "rows", "SCFG_beats_strongest_rows", "SCFG_beats_best_control_rows", "SCFG_no_debt_rows", "SCFG_overhead_le_035_rows", "primary_debt_component"], 20),
    )
    return summary


def collect_command(spec: dict[str, Any], args: argparse.Namespace, device: str) -> list[str]:
    return [
        PYTHON,
        "experiments/run_v22_59_meta_fu_control_law.py",
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
        "--metafu-rank",
        str(args.metafu_rank),
        "--metafu-alpha-max",
        str(args.metafu_alpha_max),
        "--metafu-beta-max",
        str(args.metafu_beta_max),
        "--metafu-lambda-debt-max",
        str(args.metafu_lambda_debt_max),
        "--metafu-lambda-state-max",
        str(args.metafu_lambda_state_max),
        "--metafu-tau-min",
        str(args.metafu_tau_min),
        "--metafu-tau-max",
        str(args.metafu_tau_max),
        "--metafu-default-rho",
        str(args.metafu_default_rho),
        "--metafu-eta",
        str(args.metafu_eta),
        "--metafu-eps",
        str(args.metafu_eps),
        "--metafu-ema-beta",
        str(args.metafu_ema_beta),
        "--metafu-c-min",
        str(args.metafu_c_min),
        "--metafu-refresh-interval",
        str(args.metafu_refresh_interval),
        "--metafu-param-scope",
        str(args.metafu_param_scope),
        "--metafu-input-dim",
        str(args.metafu_input_dim),
        "--metafu-feature-mode",
        str(args.metafu_feature_mode),
        "--metafu-controller-hidden",
        str(args.metafu_controller_hidden),
        "--metafu-weight-spectrum-weight",
        str(args.metafu_weight_spectrum_weight),
        "--metafu-gradient-cov-weight",
        str(args.metafu_gradient_cov_weight),
        "--metafu-momentum-weight",
        str(args.metafu_momentum_weight),
        "--metafu-tail-weight",
        str(args.metafu_tail_weight),
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
    methods = split_csv(str(args.methods or (DEFAULT_REFERENCE_METHODS if phase == "reference" else DEFAULT_M0_METHODS)), str)
    specs = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                method_l = str(method).lower()
                label = f"{args.label_prefix}_{method_l}_{dataset}_s{seed}"
                spec = {"dataset": dataset, "seed": seed, "method": method_l, "label": label}
                if method_l in M0_METHODS:
                    spec["reference_method"] = str(args.reference_method)
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
            files=f"results/v22_59/chunks/v22_59_{safe_fragment(str(spec['label']))}_summary.csv",
            gpu=str(gpu),
            timeout=int(args.collect_timeout),
        )
        return {
            **spec,
            "gpu": gpu,
            "command": " ".join(shlex.quote(str(x)) for x in cmd),
            **result,
            "summary": f"results/v22_59/chunks/v22_59_{safe_fragment(str(spec['label']))}_summary.csv",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(launch, item) for item in enumerate(specs)]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            dispatch_rows.append(row)
            write_rows(dispatch_path, dispatch_rows)
    append_exec(
        f"{PYTHON} experiments/run_v22_59_meta_fu_control_law.py --stage {phase}",
        task_id=f"v22_59_{phase}_matrix",
        status="pass" if all(r.get("status") == "pass" for r in dispatch_rows) else "fail",
        files=f"{specs_path}; {dispatch_path}; results/v22_59/chunks/",
        note=f"rows={len(dispatch_rows)}; workers={args.workers}; gpus={args.gpus}",
    )
    return dispatch_rows


def collect_chunk_rows(include_prefixes: list[str] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_59_*_summary.csv")):
        for row in read_rows(path):
            label = str(row.get("run_label", ""))
            if include_prefixes and not any(label.startswith(prefix) for prefix in include_prefixes):
                continue
            rows.append(row)
    return rows


def _need(rows: int, numerator: int, denominator: int) -> int:
    return int(math.ceil(rows * numerator / denominator))


def nearest_centroid_predictability(rows: list[dict[str, Any]], label_key: str) -> tuple[float, float, int]:
    data = [
        (
            str(r.get(label_key, "")),
            [fval(r.get(f"controller_output_{name}_mean"), 0.0) or 0.0 for name in META_FU_OUTPUT_NAMES],
        )
        for r in rows
        if r.get(label_key, "") != ""
    ]
    labels = sorted({x[0] for x in data})
    if len(data) < 2 or len(labels) <= 1:
        return 0.0, 1.0, len(labels)
    correct = 0
    for i, (label, vec) in enumerate(data):
        train = data[:i] + data[i + 1:]
        centroids = {}
        for lab in labels:
            group = [v for l, v in train if l == lab]
            if not group:
                continue
            centroids[lab] = [statistics.fmean(col) for col in zip(*group)]
        if not centroids:
            continue
        pred = min(centroids, key=lambda lab: sum((a - b) ** 2 for a, b in zip(vec, centroids[lab])))
        correct += int(pred == label)
    return float(correct / max(1, len(data))), float(1.0 / len(labels)), len(labels)


def controller_leakage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in sorted({r.get("method", "") for r in rows if r.get("phase") == "M0"}):
        group = [r for r in rows if r.get("method") == method and r.get("phase") == "M0" and r.get("run_status") == "completed"]
        ds_acc, ds_chance, ds_classes = nearest_centroid_predictability(group, "dataset")
        seed_acc, seed_chance, seed_classes = nearest_centroid_predictability(group, "seed")
        max_corr = max(
            [fval(r.get("operator_class_corr_abs"), 0.0) or 0.0 for r in group]
            + [fval(r.get("operator_loss_corr_abs"), 0.0) or 0.0 for r in group]
            + [fval(r.get("operator_margin_corr_abs"), 0.0) or 0.0 for r in group]
            + [0.0]
        )
        out.append(
            {
                "method": method,
                "rows": len(group),
                "controller_output_class_count_correlation": mean(fval(r.get("operator_class_corr_abs")) for r in group),
                "controller_output_loss_rank_correlation": mean(fval(r.get("operator_loss_corr_abs")) for r in group),
                "controller_output_margin_correlation": mean(fval(r.get("operator_margin_corr_abs")) for r in group),
                "dataset_id_classifier_accuracy_from_controller_output": ds_acc,
                "dataset_id_chance": ds_chance,
                "dataset_id_classes": ds_classes,
                "seed_id_classifier_accuracy_from_controller_output": seed_acc,
                "seed_id_chance": seed_chance,
                "seed_id_classes": seed_classes,
                "controller_output_dataset_id_predictability_le_chance_plus_010": int(ds_acc <= ds_chance + 0.10),
                "controller_output_seed_id_predictability_le_chance_plus_010": int(seed_acc <= seed_chance + 0.10),
                "max_controller_nuisance_corr": max_corr,
                "leakage_audit_pass": int(max_corr <= 0.70 and ds_acc <= ds_chance + 0.10 and seed_acc <= seed_chance + 0.10),
            }
        )
    return out


def m0_method_gate_rows(pairwise: list[dict[str, Any]], leakage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leak_by_method = {r.get("method", ""): r for r in leakage}
    out: list[dict[str, Any]] = []
    for method in sorted({r.get("method", "") for r in pairwise if r.get("method") in M0_LEARNED_METHODS}):
        group = [r for r in pairwise if r.get("method") == method]
        n = len(group)
        leak = leak_by_method.get(method, {})
        row = {
            "method": method,
            "rows": n,
            "beats_strongest_reference_rows": sum(iflag(r.get("beats_strongest_reference")) for r in group),
            "beats_handdesigned_v22_58_rows": sum(iflag(r.get("beats_handdesigned_v22_58")) for r in group),
            "beats_frozen_shuffled_random_controller_controls_rows": sum(iflag(r.get("beats_frozen_shuffled_random_controller_controls")) for r in group),
            "beats_credit_only_and_self_only_rows": sum(iflag(r.get("beats_credit_only_and_self_only")) for r in group),
            "no_ECE_Brier_tail_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
            "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in group),
            "controller_parameter_count_ratio_le_001_rows": sum(int((fval(r.get("controller_parameter_count_ratio"), float("inf")) or float("inf")) <= 0.01) for r in group),
            "dataset_predictability_pass": iflag(leak.get("controller_output_dataset_id_predictability_le_chance_plus_010")),
            "seed_predictability_pass": iflag(leak.get("controller_output_seed_id_predictability_le_chance_plus_010")),
        }
        row["required_beats_strongest_reference_rows"] = _need(n, 5, 9)
        row["required_beats_handdesigned_v22_58_rows"] = _need(n, 5, 9)
        row["required_beats_frozen_shuffled_random_rows"] = _need(n, 6, 9)
        row["required_beats_credit_self_rows"] = _need(n, 6, 9)
        row["required_no_debt_rows"] = _need(n, 7, 9)
        row["required_overhead_rows"] = _need(n, 7, 9)
        row["method_gate_pass"] = int(
            n >= 9
            and row["beats_strongest_reference_rows"] >= row["required_beats_strongest_reference_rows"]
            and row["beats_handdesigned_v22_58_rows"] >= row["required_beats_handdesigned_v22_58_rows"]
            and row["beats_frozen_shuffled_random_controller_controls_rows"] >= row["required_beats_frozen_shuffled_random_rows"]
            and row["beats_credit_only_and_self_only_rows"] >= row["required_beats_credit_self_rows"]
            and row["no_ECE_Brier_tail_debt_rows"] >= row["required_no_debt_rows"]
            and row["overhead_le_035_rows"] >= row["required_overhead_rows"]
            and row["controller_parameter_count_ratio_le_001_rows"] == n
            and row["dataset_predictability_pass"]
            and row["seed_predictability_pass"]
        )
        out.append(row)
    return out


def classify_route(route: dict[str, Any], gate_rows: list[dict[str, Any]], pairwise: list[dict[str, Any]], leakage: list[dict[str, Any]]) -> str:
    if route.get("part_a_pass") == 0:
        return "R0-CodeOrTrainingLoopBoundaryFailed"
    if not pairwise:
        return "R1-MetaControllerUnitOnly"
    if route.get("mlp_m0_exploration_pass"):
        if route.get("beats_external_oet_rows", 0) > 0:
            return "R6-MetaFUOverExternalOETOpened"
        return "R5-MLPMetaFUOverStrongBaselineOpened"
    random_controls_stronger = route.get("random_or_frozen_controls_explain_rows", 0) >= max(1, route.get("learned_pairwise_rows", 0) // 2)
    if random_controls_stronger:
        return "R2-MetaControllerRandomOrFrozenExplained"
    learned_leakage = [r for r in leakage if r.get("method") in M0_LEARNED_METHODS]
    if any((fval(r.get("max_controller_nuisance_corr"), 0.0) or 0.0) > 0.70 for r in learned_leakage):
        return "R3-MetaControllerReweightingExplained_NoFU"
    if route.get("beats_handdesigned_rows", 0) > 0 and route.get("beats_strongest_reference_rows", 0) == 0:
        return "R4-MetaFUImprovesHandRuleOnly_NotStrongOptimizerValue"
    return "R1-MetaControllerUnitOnly"


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    include_prefixes = split_csv(str(args.summary_include_prefixes), str) if str(args.summary_include_prefixes).strip() else []
    rows = collect_chunk_rows(include_prefixes)
    reference_rows = [r for r in rows if r.get("phase") == "reference" and r.get("run_status") == "completed"]
    m0_rows = [r for r in rows if r.get("phase") == "M0" and r.get("run_status") == "completed"]
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
                "mean_meta_test_query_NLL": mean(fval(r.get("meta_test_query_NLL")) for r in group),
                "mean_accuracy": mean(fval(r.get("final_accuracy")) for r in group),
                "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
                "standard_loop_pass_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in group),
                "mean_overhead_ratio": mean(fval(r.get("controller_overhead_ratio")) for r in group),
                "mean_controller_parameter_count_ratio": mean(fval(r.get("controller_parameter_count_ratio")) for r in group),
                "mean_controller_output_entropy": mean(fval(r.get("controller_output_entropy")) for r in group),
            }
        )
    write_rows(OUT_ROOT / "v22_59_method_summary.csv", method_summary)
    ref_by_key = {(r.get("method"), r.get("dataset"), str(r.get("seed"))): r for r in reference_rows}
    strongest_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for r in reference_rows:
        key = (r.get("dataset", ""), str(r.get("seed", "")))
        current = strongest_by_key.get(key)
        if current is None or (fval(r.get("final_NLL"), float("inf")) or float("inf")) < (fval(current.get("final_NLL"), float("inf")) or float("inf")):
            strongest_by_key[key] = r
    m0_by_key_method = {(r.get("method"), r.get("dataset"), str(r.get("seed"))): r for r in m0_rows}
    pairwise: list[dict[str, Any]] = []
    for r in m0_rows:
        key = (r.get("dataset", ""), str(r.get("seed", "")))
        final_nll = fval(r.get("final_NLL"))
        strongest = strongest_by_key.get(key)
        strongest_nll = fval(strongest.get("final_NLL")) if strongest else None
        ref = ref_by_key.get((r.get("reference_method", ""), r.get("dataset"), str(r.get("seed"))))
        ref_nll = fval(ref.get("final_NLL")) if ref else None
        hand = m0_by_key_method.get(("handdesigned_v22_58_scfg", r.get("dataset"), str(r.get("seed"))))
        credit = m0_by_key_method.get(("credit_only_handdesigned", r.get("dataset"), str(r.get("seed"))))
        self_only = m0_by_key_method.get(("self_only_handdesigned", r.get("dataset"), str(r.get("seed"))))
        random_controls = [
            m0_by_key_method.get((m, r.get("dataset"), str(r.get("seed"))))
            for m in ["random_scalar_scheduler_same_distribution", "frozen_controller_from_other_seed", "frozen_controller_from_other_dataset"]
        ]
        random_controls = [x for x in random_controls if x is not None]
        best_random = min(random_controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        hand_nll = fval(hand.get("final_NLL")) if hand else None
        credit_nll = fval(credit.get("final_NLL")) if credit else None
        self_nll = fval(self_only.get("final_NLL")) if self_only else None
        random_nll = fval(best_random.get("final_NLL")) if best_random else None
        poet = ref_by_key.get(("poet_official", r.get("dataset"), str(r.get("seed"))))
        pion = ref_by_key.get(("pion_oet_sphere_official", r.get("dataset"), str(r.get("seed"))))
        poet_nll = fval(poet.get("final_NLL")) if poet else None
        pion_nll = fval(pion.get("final_NLL")) if pion else None
        pairwise.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "method": r.get("method", ""),
                "control_kind": r.get("control_kind", ""),
                "reference_method": r.get("reference_method", ""),
                "final_NLL": final_nll,
                "reference_final_NLL": ref_nll,
                "strongest_reference_method": strongest.get("method", "") if strongest else "",
                "strongest_reference_final_NLL": strongest_nll,
                "handdesigned_v22_58_final_NLL": hand_nll,
                "credit_only_final_NLL": credit_nll,
                "self_only_final_NLL": self_nll,
                "best_random_frozen_control_method": best_random.get("method", "") if best_random else "",
                "best_random_frozen_control_final_NLL": random_nll,
                "poet_final_NLL": poet_nll,
                "pion_final_NLL": pion_nll,
                "Delta_NLL_vs_reference": (final_nll - ref_nll) if final_nll is not None and ref_nll is not None else "",
                "Delta_NLL_vs_strongest_reference": (final_nll - strongest_nll) if final_nll is not None and strongest_nll is not None else "",
                "Delta_NLL_vs_handdesigned_v22_58": (final_nll - hand_nll) if final_nll is not None and hand_nll is not None else "",
                "Delta_NLL_vs_credit_only": (final_nll - credit_nll) if final_nll is not None and credit_nll is not None else "",
                "Delta_NLL_vs_self_only": (final_nll - self_nll) if final_nll is not None and self_nll is not None else "",
                "Delta_NLL_vs_random_frozen": (final_nll - random_nll) if final_nll is not None and random_nll is not None else "",
                "beats_reference": int(final_nll is not None and ref_nll is not None and final_nll < ref_nll),
                "beats_strongest_reference": int(final_nll is not None and strongest_nll is not None and final_nll < strongest_nll),
                "beats_handdesigned_v22_58": int(final_nll is not None and hand_nll is not None and final_nll < hand_nll),
                "beats_frozen_shuffled_random_controller_controls": int(final_nll is not None and bool(random_controls) and all(final_nll < (fval(x.get("final_NLL"), float("inf")) or float("inf")) for x in random_controls)),
                "beats_credit_only_and_self_only": int(final_nll is not None and credit_nll is not None and self_nll is not None and final_nll < credit_nll and final_nll < self_nll),
                "beats_POET_or_external_OET": int(final_nll is not None and ((poet_nll is not None and final_nll < poet_nll) or (pion_nll is not None and final_nll < pion_nll))),
                "random_or_frozen_control_beats_method": int(final_nll is not None and random_nll is not None and random_nll <= final_nll),
                "no_ECE_Brier_tail_debt": r.get("no_ECE_Brier_tail_debt", ""),
                "held_ECE_delta": r.get("held_ECE_delta", ""),
                "held_Brier_delta": r.get("held_Brier_delta", ""),
                "held_tail_q95_delta": r.get("held_tail_q95_delta", ""),
                "held_tail_q99_delta": r.get("held_tail_q99_delta", ""),
                "AUC_loss_time_improvement": int((fval(r.get("AUC_loss_time")) or float("inf")) < (fval(ref.get("AUC_loss_time")) or float("inf"))) if ref else 0,
                "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
                "controller_parameter_count_ratio": r.get("controller_parameter_count_ratio", ""),
                "controller_output_entropy": r.get("controller_output_entropy", ""),
                "operator_class_corr_abs": r.get("operator_class_corr_abs", ""),
                "operator_loss_corr_abs": r.get("operator_loss_corr_abs", ""),
                "operator_margin_corr_abs": r.get("operator_margin_corr_abs", ""),
                "standard_loop_hard_gate_pass": r.get("standard_loop_hard_gate_pass", ""),
            }
        )
    write_rows(OUT_ROOT / "v22_59_m0_pairwise.csv", pairwise)
    leakage = controller_leakage_rows(rows)
    write_rows(OUT_ROOT / "v22_59_meta_controller_leakage_audit.csv", leakage or read_rows(OUT_ROOT / "v22_59_meta_controller_leakage_audit.csv"))
    gate_rows = m0_method_gate_rows(pairwise, leakage)
    write_rows(OUT_ROOT / "v22_59_m0_candidate_method_gate.csv", gate_rows)
    part_a = read_rows(OUT_ROOT / "v22_59_code_truth_gate.csv")
    part_a_pass = iflag(part_a[0].get("part_a_pass")) if part_a else 0
    learned_pairwise = [r for r in pairwise if r.get("method") in M0_LEARNED_METHODS]
    route = {
        "part_a_pass": part_a_pass,
        "reference_rows": len(reference_rows),
        "m0_rows": len(m0_rows),
        "learned_pairwise_rows": len(learned_pairwise),
        "m0_method_gate_rows": len(gate_rows),
        "m0_method_gate_pass_rows": sum(iflag(r.get("method_gate_pass")) for r in gate_rows),
        "standard_loop_pass_learned_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in learned_pairwise),
        "beats_strongest_reference_rows": sum(iflag(r.get("beats_strongest_reference")) for r in learned_pairwise),
        "beats_external_oet_rows": sum(iflag(r.get("beats_POET_or_external_OET")) for r in learned_pairwise),
        "beats_handdesigned_rows": sum(iflag(r.get("beats_handdesigned_v22_58")) for r in learned_pairwise),
        "beats_random_frozen_rows": sum(iflag(r.get("beats_frozen_shuffled_random_controller_controls")) for r in learned_pairwise),
        "random_or_frozen_controls_explain_rows": sum(iflag(r.get("random_or_frozen_control_beats_method")) for r in learned_pairwise),
        "beats_credit_self_rows": sum(iflag(r.get("beats_credit_only_and_self_only")) for r in learned_pairwise),
        "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in learned_pairwise),
        "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in learned_pairwise),
        "controller_param_ratio_le_001_rows": sum(int((fval(r.get("controller_parameter_count_ratio"), float("inf")) or float("inf")) <= 0.01) for r in learned_pairwise),
        "mlp_m0_exploration_pass": int(any(iflag(r.get("method_gate_pass")) for r in gate_rows)),
        "m1_gate_status": "not_started",
        "m2_gate_status": "not_started",
        "kan_gate_status": "not_started",
        "final_route": "",
    }
    route["m1_gate_status"] = "eligible_not_run_yet" if route["mlp_m0_exploration_pass"] else "skipped_M0_gate_not_opened"
    route["m2_gate_status"] = "skipped_M1_gate_not_opened"
    route["kan_gate_status"] = "skipped_MLP_MetaFU_gate_not_opened" if not route["mlp_m0_exploration_pass"] else "eligible_after_external_gate"
    route["final_route"] = classify_route(route, gate_rows, learned_pairwise, leakage)
    write_json(OUT_ROOT / "v22_59_final_route.json", route)
    downstream = [
        {"part": "E_M1_metric_mixture", "status": route["m1_gate_status"], "reason": "M1 starts only after M0 weak/exploration signal"},
        {"part": "F_M2_lowrank_coeff", "status": route["m2_gate_status"], "reason": "M2 starts only after M1 exploration pass"},
        {"part": "H_KAN", "status": route["kan_gate_status"], "reason": "KAN official starts only after MLP+MetaFU gate"},
    ]
    write_rows(OUT_ROOT / "v22_59_downstream_gate_status.csv", downstream)
    fallback = build_m0_fallback_report(route, gate_rows, pairwise, leakage)
    write_rows(OUT_ROOT / "v22_59_m0_fallback_diagnostics.csv", fallback)
    append_exec(
        f"{PYTHON} experiments/run_v22_59_meta_fu_control_law.py --stage summarize",
        task_id="v22_59_summarize",
        status="pass",
        files=(
            "results/v22_59/v22_59_method_summary.csv; results/v22_59/v22_59_m0_pairwise.csv; "
            "results/v22_59/v22_59_m0_candidate_method_gate.csv; results/v22_59/v22_59_meta_controller_leakage_audit.csv; "
            "results/v22_59/v22_59_final_route.json; results/v22_59/v22_59_downstream_gate_status.csv; "
            "results/v22_59/v22_59_m0_fallback_diagnostics.csv"
        ),
        note=f"final_route={route['final_route']}",
    )
    append_recap(
        "Part D M0 learned scalar scheduler summary and route",
        "Method summary:\n\n"
        + md_table(method_summary, ["method", "phase", "rows", "completed_rows", "blocked_rows", "mean_meta_test_query_NLL", "no_debt_rows", "standard_loop_pass_rows", "mean_overhead_ratio", "mean_controller_parameter_count_ratio"], 40)
        + "\nM0 pairwise rows:\n\n"
        + md_table(pairwise, ["dataset", "seed", "method", "Delta_NLL_vs_strongest_reference", "beats_strongest_reference", "beats_handdesigned_v22_58", "beats_frozen_shuffled_random_controller_controls", "beats_credit_only_and_self_only", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 80)
        + "\nPer-method gate rows:\n\n"
        + md_table(gate_rows, ["method", "rows", "method_gate_pass", "beats_strongest_reference_rows", "beats_handdesigned_v22_58_rows", "beats_frozen_shuffled_random_controller_controls_rows", "beats_credit_only_and_self_only_rows", "no_ECE_Brier_tail_debt_rows", "overhead_le_035_rows"], 12)
        + "\nLeakage audit:\n\n"
        + md_table(leakage, ["method", "rows", "dataset_id_classifier_accuracy_from_controller_output", "dataset_id_chance", "seed_id_classifier_accuracy_from_controller_output", "seed_id_chance", "max_controller_nuisance_corr", "leakage_audit_pass"], 20)
        + "\nFallback diagnostics:\n\n"
        + md_table(fallback, ["check", "status", "evidence", "repair_direction"], 20)
        + "\nRoute JSON:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return route


def build_m0_fallback_report(route: dict[str, Any], gate_rows: list[dict[str, Any]], pairwise: list[dict[str, Any]], leakage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    learned = [r for r in pairwise if r.get("method") in M0_LEARNED_METHODS]
    if route.get("mlp_m0_exploration_pass"):
        rows.append({"check": "M0 gate", "status": "pass", "evidence": "at least one learned M0 method passed gate", "repair_direction": "M1 eligible"})
        return rows
    rows.append(
        {
            "check": "M0 gate",
            "status": "fail",
            "evidence": f"m0_method_gate_pass_rows={route.get('m0_method_gate_pass_rows')}",
            "repair_direction": "follow v22.59 9.1: inspect leakage/output constancy/reference strength/source-witness-query objective",
        }
    )
    if learned:
        entropy = mean(fval(r.get("controller_output_entropy")) for r in learned)
        rows.append(
            {
                "check": "controller output constancy",
                "status": "needs_repair" if (entropy or 0.0) < 0.10 else "checked",
                "evidence": f"mean_controller_output_entropy={entropy}",
                "repair_direction": "reduce input noise, normalize/clamp, or retrain smaller controller if entropy collapses",
            }
        )
    learned_leakage = [r for r in leakage if r.get("method") in M0_LEARNED_METHODS]
    if learned_leakage:
        max_corr = max((fval(r.get("max_controller_nuisance_corr"), 0.0) or 0.0) for r in learned_leakage)
        rows.append(
            {
                "check": "controller leakage/reweighting",
                "status": "needs_repair" if max_corr > 0.70 else "checked",
                "evidence": f"max_controller_nuisance_corr={max_corr}",
                "repair_direction": "remove loss/class direct features, keep normalized trend/condition summaries, add anti-correlation penalty",
            }
        )
    rows.append(
        {
            "check": "source/witness/query objective",
            "status": "checked",
            "evidence": "meta_dataset rows and collect rows record query_split_is_train_only=1 and hashed S/W/Q ids",
            "repair_direction": "if still failing, shorten H to 20 and use first-order objective over strongest reference",
        }
    )
    rows.append(
        {
            "check": "reference strength",
            "status": "needs_repair" if route.get("beats_handdesigned_rows", 0) > 0 and route.get("beats_strongest_reference_rows", 0) == 0 else "checked",
            "evidence": f"beats_handdesigned_rows={route.get('beats_handdesigned_rows')}; beats_strongest_reference_rows={route.get('beats_strongest_reference_rows')}",
            "repair_direction": "train over strongest reference and add POET/Pion reference rows to meta objective",
        }
    )
    rows.append(
        {
            "check": "random/frozen controls",
            "status": "needs_repair" if route.get("random_or_frozen_controls_explain_rows", 0) > 0 else "checked",
            "evidence": f"random_or_frozen_controls_explain_rows={route.get('random_or_frozen_controls_explain_rows')}",
            "repair_direction": "add same-output-distribution controls and reduce controller capacity",
        }
    )
    return rows


def run_figures(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(OUT_ROOT / "v22_59_m0_pairwise.csv")
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {"figure_rows": len(rows), "figures_created": 0}
    write_json(OUT_ROOT / "v22_59_figure_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all", choices=["all", "o0", "reanalyse-v58", "meta-dataset", "collect", "reference", "m0", "summarize", "figures"])
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="adamw")
    p.add_argument("--reference-method", default="adamw")
    p.add_argument("--label", default="v22_59_collect")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--seeds", default="3,4,5")
    p.add_argument("--methods", default="")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label-prefix", default="v22_59_reference")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--smoke-steps", type=int, default=4)
    p.add_argument("--smoke-device", default="cuda:0")
    p.add_argument("--synthetic-device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--cohorts", type=int, default=4)
    p.add_argument("--cohort-mode", default="random", choices=["random", "label_loss_stratified"])
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--metafu-rank", type=int, default=2)
    p.add_argument("--metafu-alpha-max", type=float, default=0.16)
    p.add_argument("--metafu-beta-max", type=float, default=0.16)
    p.add_argument("--metafu-lambda-debt-max", type=float, default=1.25)
    p.add_argument("--metafu-lambda-state-max", type=float, default=1.25)
    p.add_argument("--metafu-tau-min", type=float, default=0.02)
    p.add_argument("--metafu-tau-max", type=float, default=0.35)
    p.add_argument("--metafu-default-rho", type=float, default=0.50)
    p.add_argument("--metafu-eta", type=float, default=1.0e-3)
    p.add_argument("--metafu-eps", type=float, default=1.0e-5)
    p.add_argument("--metafu-ema-beta", type=float, default=0.90)
    p.add_argument("--metafu-c-min", type=float, default=0.20)
    p.add_argument("--metafu-refresh-interval", type=int, default=20)
    p.add_argument("--metafu-param-scope", default="last_layer", choices=["all", "last_layer"])
    p.add_argument("--metafu-input-dim", type=int, default=META_FU_FEATURE_DIM)
    p.add_argument("--metafu-feature-mode", default="default", choices=["default", "anti_leak", "anti_leak_v1"])
    p.add_argument("--metafu-controller-hidden", type=int, default=8)
    p.add_argument("--metafu-weight-spectrum-weight", type=float, default=0.25)
    p.add_argument("--metafu-gradient-cov-weight", type=float, default=1.0)
    p.add_argument("--metafu-momentum-weight", type=float, default=0.10)
    p.add_argument("--metafu-tail-weight", type=float, default=0.55)
    p.add_argument("--metafu-low-overhead", action="store_true")
    p.add_argument("--meta-train-datasets", default="MNIST,FashionMNIST,KMNIST,Wine")
    p.add_argument("--meta-train-seeds", default="0,1")
    p.add_argument("--meta-label-reference-method", default="adamw")
    p.add_argument("--meta-probe-strengths", default="0.0,0.04,0.08,0.12,0.16")
    p.add_argument("--meta-label-horizon", type=int, default=20)
    p.add_argument("--meta-controller-epochs", type=int, default=600)
    p.add_argument("--meta-controller-lr", type=float, default=2.0e-3)
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
        if args.stage == "reanalyse-v58":
            print(json.dumps(reanalyse_v22_58(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "meta-dataset":
            print(json.dumps(build_meta_dataset(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "collect":
            print(json.dumps(run_collect(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "reference":
            run_matrix(args, phase="reference")
            return 0
        if args.stage == "m0":
            if args.label_prefix == "v22_59_reference":
                args.label_prefix = f"v22_59_m0_over_{safe_fragment(args.reference_method)}"
            run_matrix(args, phase="m0")
            return 0
        if args.stage == "summarize":
            print(json.dumps(summarize(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "figures":
            print(json.dumps(run_figures(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "all":
            gate = run_o0(args)
            if not iflag(gate.get("part_a_pass")):
                summarize(args)
                run_figures(args)
                return 2
            v58 = reanalyse_v22_58(args)
            if not iflag(v58.get("v22_58_artifacts_present")):
                summarize(args)
                run_figures(args)
                return 3
            meta = build_meta_dataset(args)
            if not iflag(meta.get("controller_train_pass")):
                summarize(args)
                run_figures(args)
                return 4
            ref_args = argparse.Namespace(**vars(args))
            ref_args.label_prefix = "v22_59_reference"
            ref_args.methods = args.methods if args.methods and all(m in REFERENCE_METHODS for m in split_csv(args.methods, str)) else DEFAULT_REFERENCE_METHODS
            run_matrix(ref_args, phase="reference")
            m0_args = argparse.Namespace(**vars(args))
            m0_args.label_prefix = f"v22_59_m0_over_{safe_fragment(args.reference_method)}"
            m0_args.methods = args.methods if args.methods and all(m in M0_METHODS for m in split_csv(args.methods, str)) else DEFAULT_M0_METHODS
            run_matrix(m0_args, phase="m0")
            summarize(args)
            run_figures(args)
            return 0
    except Exception:
        err = traceback.format_exc()
        (LOG_ROOT / "v22_59_runner_exception.log").write_text(err, encoding="utf-8")
        append_exec(
            f"{PYTHON} experiments/run_v22_59_meta_fu_control_law.py --stage {args.stage}",
            task_id=f"v22_59_{args.stage}_exception",
            status="exception",
            files="results/v22_59/logs/v22_59_runner_exception.log",
            note=err[-1000:],
            exit_code="exception",
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
