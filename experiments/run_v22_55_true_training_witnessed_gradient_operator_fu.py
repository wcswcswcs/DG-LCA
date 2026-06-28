#!/usr/bin/env python3
"""DG-KAN v22.55 true-training witnessed gradient operator FU runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import py_compile
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from dgkan.fu.witnessed_gradient_operator import WGOConfig, WitnessedGradientOperator
from dgkan.optim.witnessed_optimizer_wrapper import WitnessedOptimizerWrapper
from experiments import audit_standard_training_loop as loop_audit
from experiments import run_v22_49A_broader_related_work_map as v49
from experiments.run_v22_54_true_training_witnessed_objective_fu import (
    CautiousAdamW,
    ScheduleFreeAdamWLocal,
    make_poet_model,
)


KAN_ENV_PYTHON = ROOT.parent / "miniconda3/envs/kan/bin/python"
PYTHON = os.environ.get("KAN_PYTHON", str(KAN_ENV_PYTHON if KAN_ENV_PYTHON.exists() else sys.executable))
OUT_ROOT = ROOT / "results/v22_55"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.55_TrueTrainingWitnessedGradientOperatorFU_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.55_TrueTrainingWitnessedGradientOperatorFU_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.55_TrueTrainingWitnessedGradientOperatorFU_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py"

REFERENCE_METHODS = {
    "adamw",
    "cautious_adamw",
    "schedule_free_adamw_local",
    "poet_official",
    "pion_oet_sphere_official",
    "pion_oet_local",
}
WGO_CANDIDATES = {
    "wgo_diag_snr",
    "wgo_block_lowrank_r4",
    "wgo_safety_diag",
    "wgo_residual_deleak_diag",
    "wgo_residual_safety_diag",
    "wgo_poet",
}
WGO_CONTROLS = {
    "same_gate_distribution_random",
    "same_gate_entropy_random",
    "same_block_norm_random_gate",
    "shuffled_witness_gate",
    "frozen_gate_from_previous_seed",
    "source_only_gate",
    "witness_only_gate",
    "inverse_class_count_gate",
    "hard_loss_gate",
    "loss_rank_gate",
    "random_label_gate",
    "same_compute_noop_gate",
    "margin_shuffle_gate",
}
WGO_METHODS = WGO_CANDIDATES | WGO_CONTROLS
REWEIGHTING_CONTROLS = {"inverse_class_count_gate", "hard_loss_gate", "loss_rank_gate", "margin_shuffle_gate", "source_only_gate", "witness_only_gate"}
DEFAULT_REFERENCE_METHODS = "adamw,cautious_adamw,schedule_free_adamw_local,poet_official,pion_oet_sphere_official"
DEFAULT_WGO_METHODS = (
    "wgo_diag_snr,wgo_block_lowrank_r4,wgo_safety_diag,wgo_poet,"
    "same_gate_distribution_random,same_gate_entropy_random,same_block_norm_random_gate,"
    "shuffled_witness_gate,frozen_gate_from_previous_seed,source_only_gate,witness_only_gate,"
    "inverse_class_count_gate,hard_loss_gate,loss_rank_gate,random_label_gate,same_compute_noop_gate"
)


class ExternalUnavailable(RuntimeError):
    pass


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
            "# DG-KAN v22.55 TrueTrainingWitnessedGradientOperatorFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、文件、GPU、状态、失败与修复尝试。"
            "官方 WGO 只允许 task loss backward 与 optimizer-owned gradient transform。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.55 TrueTrainingWitnessedGradientOperatorFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘 artifact；结论、失败、修复、insight 必须带证据链。"
            "不得编造数据；blocked/unavailable 必须原样记录。\n",
            encoding="utf-8",
        )


def safe_fragment(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:180]


def split_csv(text: str, cast: Any = str) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


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


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8", errors="replace"))


def append_exec(
    command: str,
    *,
    task_id: str,
    status: str,
    gpu: str = "",
    files: str = "",
    note: str = "",
    exit_code: Any = "n/a",
) -> None:
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
    journal = read_rows(OUT_ROOT / "v22_55_command_journal.csv")
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(
        OUT_ROOT / "v22_55_command_journal.csv",
        journal,
        ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"],
    )
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


def run_cmd(cmd: list[str], *, task_id: str, files: str = "", gpu: str = "cpu", timeout: int | None = None) -> dict[str, Any]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr_path = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
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
            " ".join(shlex.quote(str(x)) for x in cmd),
            task_id=task_id,
            status=status,
            gpu=gpu,
            files=files,
            exit_code=proc.returncode,
            note=f"stdout={stdout_path}; stderr={stderr_path}; wall_seconds={time.time() - start:.3f}",
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


def fval(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def iflag(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.fmean(clean)) if clean else None


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 20) -> str:
    if not rows:
        return "_no rows_\n"
    shown = rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in shown:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.6g}"
            vals.append(str(val).replace("\n", " ")[:120])
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > limit:
        lines.append(f"\n_... {len(rows) - limit} more rows omitted_")
    return "\n".join(lines) + "\n"


def torch_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_bundle_tensors(dataset: str, train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    return v49.load_bundle_tensors(dataset, train_size, held_size, test_size, seed)


def make_model(input_dim: int, num_classes: int, hidden: int, seed: int, device: Any) -> Any:
    return v49.make_model(input_dim, num_classes, hidden, seed, device)


def evaluate_tensors(model: Any, x: Any, y: Any, device: Any, num_classes: int, batch_size: int) -> dict[str, float]:
    return v49.evaluate_tensors(model, x, y, device, num_classes, batch_size)


def optimizer_state_count(opt: Any) -> int:
    total = 0
    state = getattr(opt, "state", {})
    if isinstance(state, dict):
        for item in state.values():
            if isinstance(item, dict):
                for value in item.values():
                    total += int(value.numel()) if hasattr(value, "numel") else 1
            else:
                total += int(item.numel()) if hasattr(item, "numel") else 1
    base = getattr(opt, "base", None)
    if base is not None and base is not opt:
        total += optimizer_state_count(base)
    return int(total)


def trainable_param_count(model: Any) -> int:
    return int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))


def select_wgo_params(model: Any, scope: str) -> list[Any]:
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
    raise ValueError(f"unknown WGO parameter scope: {scope}")


def batch_indices(n: int, batch_size: int, generator: Any, device: Any) -> Any:
    import torch

    if batch_size >= n:
        return torch.arange(n, device=device)
    return torch.randperm(n, generator=generator, device=device)[:batch_size]


def cohort_positions(count: int, cohorts: int, generator: Any, device: Any) -> list[Any]:
    import torch

    if count <= 0:
        return []
    idx = torch.randperm(count, generator=generator, device=device)
    return [chunk for chunk in torch.chunk(idx, max(1, int(cohorts))) if int(chunk.numel()) > 0]


def stratified_cohort_positions(labels: Any, losses: Any, cohorts: int, generator: Any, device: Any) -> list[Any]:
    import torch

    k = max(1, int(cohorts))
    buckets: list[list[Any]] = [[] for _ in range(k)]
    labels = labels.detach().long().reshape(-1)
    losses = losses.detach().float().reshape(-1)
    for label in torch.unique(labels).tolist():
        label_pos = torch.nonzero(labels == int(label), as_tuple=False).reshape(-1)
        if int(label_pos.numel()) == 0:
            continue
        order = torch.argsort(losses[label_pos], stable=True)
        ordered = label_pos[order]
        if int(ordered.numel()) > 1:
            shift = int(torch.randint(0, k, (1,), generator=generator, device=device).item())
        else:
            shift = 0
        for j, pos in enumerate(ordered):
            buckets[(j + shift) % k].append(pos)
    out = []
    for bucket in buckets:
        if bucket:
            out.append(torch.stack(bucket).to(device=device))
    return out


def flatten_grads(items: Iterable[Any | None], params: list[Any]) -> Any:
    import torch

    vecs = []
    for tensor, param in zip(items, params):
        if tensor is None:
            vecs.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
        else:
            vecs.append(tensor.detach().reshape(-1))
    if not vecs:
        return torch.empty(0)
    return torch.cat(vecs)


def vector_norm(vec: Any) -> float:
    if vec is None or int(vec.numel()) == 0:
        return 0.0
    val = float(vec.norm().detach().item())
    return val if math.isfinite(val) else 0.0


def make_base_optimizer(method: str, model: Any, args: argparse.Namespace, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    if method == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        return model, opt, {"optimizer_step_source": "torch.optim.AdamW", "external_optimizer_available": 1}
    if method == "cautious_adamw":
        opt = CautiousAdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        return model, opt, {"optimizer_step_source": "CautiousAdamW_grad_gate_over_AdamW", "external_optimizer_available": 1}
    if method == "schedule_free_adamw_local":
        opt = ScheduleFreeAdamWLocal(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        return model, opt, {"optimizer_step_source": "ScheduleFreeAdamWLocal_over_AdamW", "external_optimizer_available": 1}
    if method == "poet_official":
        try:
            wrapped, opt, audit = make_poet_model(model, args)
        except Exception as exc:
            raise ExternalUnavailable(f"POET official unavailable: {exc}") from exc
        audit.update({"optimizer_step_source": "poet_torch.get_poet_optimizer", "external_optimizer_available": 1})
        return wrapped, opt, audit
    if method in {"pion_oet_sphere_official", "pion_oet_local"}:
        try:
            from experiments.run_v22_53O_pion_poet_external_oet_baseline import MatrixGeometryOptimizer

            opt = MatrixGeometryOptimizer(method, model, float(args.lr), float(args.weight_decay), device)
        except Exception as exc:
            raise ExternalUnavailable(f"{method} unavailable: {exc}") from exc
        return model, opt, {"optimizer_step_source": f"{method}_optimizer_step", "external_optimizer_available": 1}
    raise ValueError(f"unknown optimizer method: {method}")


def reference_method_for(method: str, args: argparse.Namespace) -> str:
    if method == "wgo_poet":
        return "poet_official"
    if method in WGO_METHODS:
        return str(args.reference_method)
    return method


def wgo_config_for(method: str, args: argparse.Namespace) -> WGOConfig:
    variant = "diag_snr"
    control = "none"
    rank = 4
    if method == "wgo_block_lowrank_r4":
        variant = "block_lowrank"
        rank = 4
    elif method == "wgo_safety_diag":
        variant = "safety_diag"
    elif method == "wgo_residual_deleak_diag":
        variant = "residual_deleak_diag"
    elif method == "wgo_residual_safety_diag":
        variant = "residual_safety_diag"
    elif method in WGO_CONTROLS:
        control = method
    return WGOConfig(
        variant=variant,
        control=control,
        lambda_var=float(args.wgo_lambda_var),
        lambda_debt=float(args.wgo_lambda_debt),
        ema_beta=float(args.wgo_ema_beta),
        block_size=int(args.wgo_block_size),
        lowrank_rank=rank,
        clip_min=float(args.wgo_clip_min),
        clip_max=float(args.wgo_clip_max),
        random_seed=int(args.seed) + int(hashlib.sha256(method.encode("utf-8")).hexdigest()[:8], 16),
    )


def margins_from_logits(logits: Any, labels: Any) -> Any:
    import torch

    probs = torch.softmax(logits.detach().float(), dim=1)
    true = probs.gather(1, labels.long().view(-1, 1)).squeeze(1)
    masked = probs.clone()
    masked.scatter_(1, labels.long().view(-1, 1), -1.0)
    other = masked.max(dim=1).values
    return true - other


def compute_cohort_grads(method: str, logits: Any, yb: Any, cohorts: list[Any], params: list[Any], generator: Any) -> tuple[list[list[Any | None]], list[Any], list[Any], list[Any]]:
    import torch
    import torch.nn.functional as F

    labels_for_gate = yb
    if method == "random_label_gate":
        labels_for_gate = yb[torch.randperm(int(yb.numel()), generator=generator, device=yb.device)]
    per_loss = F.cross_entropy(logits.float(), labels_for_gate.long(), reduction="none")
    margins = margins_from_logits(logits, labels_for_gate.long())
    cohort_grads: list[list[Any | None]] = []
    cohort_labels: list[Any] = []
    cohort_losses: list[Any] = []
    cohort_margins: list[Any] = []
    for pos in cohorts:
        if int(pos.numel()) == 0:
            continue
        loss = per_loss[pos].mean()
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        cohort_grads.append(list(grads))
        cohort_labels.append(labels_for_gate[pos].detach())
        cohort_losses.append(per_loss[pos].detach())
        cohort_margins.append(margins[pos].detach())
    return cohort_grads, cohort_labels, cohort_losses, cohort_margins


def compute_last_layer_closed_form_cohort_grads(
    method: str,
    logits: Any,
    features: Any,
    yb: Any,
    cohorts: list[Any],
    params: list[Any],
    generator: Any,
) -> tuple[list[list[Any | None]], list[Any], list[Any], list[Any]]:
    import torch
    import torch.nn.functional as F

    labels_for_gate = yb
    if method == "random_label_gate":
        labels_for_gate = yb[torch.randperm(int(yb.numel()), generator=generator, device=yb.device)]
    logits_f = logits.detach().float()
    features_f = features.detach().float()
    per_loss = F.cross_entropy(logits_f, labels_for_gate.long(), reduction="none")
    margins = margins_from_logits(logits_f, labels_for_gate.long())
    probs = torch.softmax(logits_f, dim=1)
    one_hot = F.one_hot(labels_for_gate.long(), num_classes=int(probs.shape[1])).to(dtype=probs.dtype, device=probs.device)
    errors = probs - one_hot
    cohort_grads: list[list[Any | None]] = []
    cohort_labels: list[Any] = []
    cohort_losses: list[Any] = []
    cohort_margins: list[Any] = []
    for pos in cohorts:
        if int(pos.numel()) == 0:
            continue
        err = errors[pos]
        feat = features_f[pos]
        denom = max(1, int(pos.numel()))
        weight_grad = err.transpose(0, 1).matmul(feat).div(float(denom))
        bias_grad = err.mean(dim=0)
        grads: list[Any | None] = []
        for param in params:
            if param.ndim == 2 and tuple(param.shape) == tuple(weight_grad.shape):
                grads.append(weight_grad.to(device=param.device, dtype=param.dtype))
            elif param.ndim == 1 and tuple(param.shape) == tuple(bias_grad.shape):
                grads.append(bias_grad.to(device=param.device, dtype=param.dtype))
            else:
                grads.append(torch.zeros_like(param))
        cohort_grads.append(grads)
        cohort_labels.append(labels_for_gate[pos].detach())
        cohort_losses.append(per_loss[pos].detach())
        cohort_margins.append(margins[pos].detach())
    return cohort_grads, cohort_labels, cohort_losses, cohort_margins


def write_blocked_collect(args: argparse.Namespace, reason: str, method: str, ref_method: str) -> dict[str, Any]:
    row = {
        "run_label": str(args.label),
        "run_status": "blocked",
        "blocker": reason,
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "phase": "WGO" if method in WGO_METHODS else "reference",
        "reference_method": ref_method if method in WGO_METHODS else "",
        "final_NLL": "",
        "standard_loop_hard_gate_pass": 0,
        "external_oet_unavailable_local_only": int("unavailable" in reason.lower()),
    }
    out = CHUNK_ROOT / f"v22_55_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    try:
        import torch._functorch.config as functorch_config

        functorch_config.donated_buffer = False
    except Exception:
        pass
    method = str(args.method).lower()
    ref_method = reference_method_for(method, args)
    if method not in REFERENCE_METHODS and method not in WGO_METHODS:
        raise ValueError(f"unknown method: {method}")
    device = torch_device(str(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(args.seed) + 225500)
    try:
        tensors = load_bundle_tensors(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    except Exception as exc:
        return write_blocked_collect(args, f"DatasetUnavailable: {exc}", method, ref_method)
    if tensors["used_fake_data"]:
        return write_blocked_collect(args, "FakeDataRejected: loader reported used_fake_data=1", method, ref_method)
    base_model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 225500, device)
    try:
        model, base_opt, opt_audit = make_base_optimizer(ref_method, base_model, args, device)
    except ExternalUnavailable as exc:
        return write_blocked_collect(args, str(exc), method, ref_method)
    params = [p for p in model.parameters() if p.requires_grad]
    wgo_params = select_wgo_params(model, str(args.wgo_param_scope))
    wgo_state = None
    opt = base_opt
    if method in WGO_METHODS:
        wgo_state = WitnessedGradientOperator(wgo_params, wgo_config_for(method, args))
        opt = WitnessedOptimizerWrapper(base_opt, wgo_state)
    x_train = tensors["x_train"].to(device)
    y_train = tensors["y_train"].to(device)
    x_held = tensors["x_held"].to(device)
    y_held = tensors["y_held"].to(device)
    x_test = tensors["x_test"].to(device)
    y_test = tensors["y_test"].to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(hashlib.sha256(method.encode("utf-8")).hexdigest()[:8], 16))
    pre_held = evaluate_tensors(model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    pre_test = evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    train_losses: list[float] = []
    step_ms: list[float] = []
    opt_ms: list[float] = []
    stats_ms: list[float] = []
    transform_ms_before = 0.0
    trace_events: list[str] = []
    forward_called = 0
    loss_task_backward_called = 0
    optimizer_step_called = 0
    loss_total_is_task_loss_only_for_official_wgo = 1
    fu_auxiliary_loss_used_official = 0
    runtime_audit = loop_audit.RuntimeTrainingLoopAudit(params)
    total_start = time.perf_counter()
    feature_cache: dict[str, Any] = {}
    hook_handle = None
    if method in WGO_METHODS and str(args.wgo_param_scope) == "last_layer" and hasattr(model, "fc3"):
        def _capture_last_layer_input(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if inputs:
                feature_cache["last_layer_input"] = inputs[0].detach()

        hook_handle = model.fc3.register_forward_hook(_capture_last_layer_input)
    try:
        for step in range(int(args.steps)):
            idx = batch_indices(int(x_train.shape[0]), int(args.batch_size), generator, device)
            xb = x_train[idx]
            yb = y_train[idx].long()
            opt.zero_grad(set_to_none=True)
            step_start = time.perf_counter()
            logits = model(xb).float()
            forward_called = 1
            trace_events.append("forward")
            loss_task = F.cross_entropy(logits, yb)
            if method in WGO_METHODS and wgo_state is not None:
                stat_start = time.perf_counter()
                stats_interval = max(1, int(args.wgo_stats_interval))
                should_observe = step == 0 or step % stats_interval == 0
                if should_observe:
                    if str(args.cohort_mode) == "label_loss_stratified":
                        per_sample_loss_for_split = F.cross_entropy(logits.detach().float(), yb, reduction="none")
                        positions = stratified_cohort_positions(yb, per_sample_loss_for_split, int(args.cohorts), generator, device)
                    else:
                        positions = cohort_positions(int(idx.numel()), int(args.cohorts), generator, device)
                    if str(args.wgo_param_scope) == "last_layer" and "last_layer_input" in feature_cache:
                        cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_last_layer_closed_form_cohort_grads(
                            method,
                            logits,
                            feature_cache["last_layer_input"],
                            yb,
                            positions,
                            wgo_params,
                            generator,
                        )
                    else:
                        cohort_grads, cohort_labels, cohort_losses, cohort_margins = compute_cohort_grads(method, logits, yb, positions, wgo_params, generator)
                    wgo_state.observe(
                        cohort_grads,
                        cohort_labels=cohort_labels,
                        cohort_losses=cohort_losses,
                        cohort_margins=cohort_margins,
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
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    wgo_diag = opt.diagnostics() if isinstance(opt, WitnessedOptimizerWrapper) else {}
    transform_ms_after = float(wgo_diag.get("wgo_transform_ms", 0.0) or 0.0)
    wgo_observe_cumulative_ms = float(wgo_diag.get("wgo_stats_ms", 0.0) or 0.0)
    wgo_transform_cumulative_ms = transform_ms_after - transform_ms_before
    wgo_diag = dict(wgo_diag)
    wgo_diag.pop("wgo_stats_ms", None)
    wgo_diag.pop("wgo_transform_ms", None)
    wgo_stats_step_ms = mean(stats_ms) if stats_ms else 0.0
    wgo_transform_step_ms = wgo_transform_cumulative_ms / max(1, int(args.steps))
    full_step_ms = mean(step_ms) or 0.0
    trace_hash = hashlib.sha256("|".join(trace_events).encode("utf-8")).hexdigest()[:16]
    no_debt = int(
        (post_held["ECE"] - pre_held["ECE"]) <= float(args.debt_tolerance)
        and (post_held["Brier"] - pre_held["Brier"]) <= float(args.debt_tolerance)
        and (post_held["tail_q95"] - pre_held["tail_q95"]) <= float(args.tail_debt_tolerance)
    )
    optimizer_owned = int(wgo_diag.get("optimizer_owned_gradient_transform_pass", 0)) if method in WGO_METHODS else 1
    audit_flags = runtime_audit.as_dict()
    wgo_stats_backend = "last_layer_closed_form" if method in WGO_METHODS and str(args.wgo_param_scope) == "last_layer" and hasattr(model, "fc3") else "autograd"
    standard_loop_hard_gate_pass = int(
        forward_called == 1
        and loss_task_backward_called == 1
        and optimizer_step_called == 1
        and loss_total_is_task_loss_only_for_official_wgo == 1
        and fu_auxiliary_loss_used_official == 0
        and optimizer_owned == 1
        and sum(audit_flags.values()) == 0
    )
    row = {
        "run_label": str(args.label),
        "run_status": "completed",
        "phase": "WGO" if method in WGO_METHODS else "reference",
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "objective_family": "WGO",
        "control_kind": method if method in WGO_CONTROLS else ("candidate" if method in WGO_CANDIDATES else "reference"),
        "reference_method": ref_method if method in WGO_METHODS else "",
        "model_family": f"MLP_hidden{int(args.hidden)}",
        "device": str(args.device),
        "python": PYTHON,
        "train_size": int(args.train_size),
        "held_size": int(args.held_size),
        "test_size": int(args.test_size),
        "hidden": int(args.hidden),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "cohorts": int(args.cohorts),
        "cohort_mode": str(args.cohort_mode),
        "wgo_stats_interval": int(args.wgo_stats_interval),
        "wgo_param_scope": str(args.wgo_param_scope),
        "wgo_clip_min": float(args.wgo_clip_min),
        "wgo_clip_max": float(args.wgo_clip_max),
        "wgo_stats_backend": wgo_stats_backend if method in WGO_METHODS else "",
        "wgo_parameter_count": int(sum(int(p.numel()) for p in wgo_params)) if method in WGO_METHODS else 0,
        "diagnostic_only": int(
            str(args.cohort_mode) != "random"
            or int(args.wgo_stats_interval) != 1
            or str(args.wgo_param_scope) != "all"
            or float(args.wgo_clip_min) != 0.0
            or float(args.wgo_clip_max) != 1.0
            or method in {"wgo_residual_deleak_diag", "wgo_residual_safety_diag"}
        ),
        "final_NLL": post_held["NLL"],
        "final_accuracy": post_held["accuracy"],
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
        "no_ECE_Brier_tail_debt": no_debt,
        "train_time_sec": wall,
        "optimizer_step_ms": mean(opt_ms),
        "full_step_ms": full_step_ms,
        "wgo_stats_ms": wgo_stats_step_ms,
        "wgo_transform_ms": wgo_transform_step_ms,
        "wgo_observe_cumulative_ms": wgo_observe_cumulative_ms,
        "wgo_transform_cumulative_ms": wgo_transform_cumulative_ms,
        "controller_overhead_ratio": float((wgo_stats_step_ms + wgo_transform_step_ms) / max(full_step_ms, 1.0e-12)),
        "peak_memory_mb": peak_memory,
        "extra_state_bytes": int(sum(int(p.numel()) * p.element_size() for p in wgo_params)) if method in WGO_METHODS else 0,
        "trainable_parameter_count": trainable_param_count(model),
        "optimizer_state_count": optimizer_state_count(opt),
        "loss_task_only_official": loss_total_is_task_loss_only_for_official_wgo,
        "loss_total_is_task_loss_only_for_official_WGO": loss_total_is_task_loss_only_for_official_wgo,
        "fu_auxiliary_loss_used_official": fu_auxiliary_loss_used_official,
        "forward_called": forward_called,
        "loss_task_backward_called": loss_task_backward_called,
        "optimizer_step_called": optimizer_step_called,
        "standard_loop_hard_gate_pass": standard_loop_hard_gate_pass,
        "standard_loop_runtime_trace_pass": standard_loop_hard_gate_pass,
        "branch_replay_used_as_training": 0,
        "proxy_direction_used_as_training": 0,
        "candidate_action_selection_used_for_runtime": 0,
        "cohort_topk_selection_used": 0,
        "score_selector_used": 0,
        "class_weight_or_sampler_used_as_fu": 0,
        "uses_validation_test_future_direction": 0,
        "training_loop_trace_hash": trace_hash,
        **audit_flags,
        **wgo_diag,
        **opt_audit,
    }
    out = CHUNK_ROOT / f"v22_55_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def run_o0(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    try:
        py_compile.compile(str(RUNNER), doraise=True)
        py_compile_runner_pass = 1
        py_compile_runner_error = ""
    except Exception as exc:
        py_compile_runner_pass = 0
        py_compile_runner_error = repr(exc)
    compile_result = run_cmd(
        [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"],
        task_id="v22_55_compileall_repo",
        files="results/v22_55/v22_55_code_truth_gate.csv",
        timeout=int(args.compile_timeout),
    )
    full_import = run_cmd(
        [PYTHON, "-c", "import dgkan; import experiments.run_v22_55_true_training_witnessed_gradient_operator_fu; print('pass')"],
        task_id="v22_55_full_repo_runner_import",
        files="results/v22_55/v22_55_code_truth_gate.csv",
    )
    poet_import = run_cmd(
        [PYTHON, "-c", "from poet_torch import POETConfig, POETModel, get_poet_optimizer; print('pass')"],
        task_id="v22_55_external_poet_import",
        files="results/v22_55/v22_55_code_truth_gate.csv",
    )
    pion_code = (
        "import sys; "
        "sys.path.insert(0, 'external/oet_baselines/pion_spectrum_sphere/megatron-lm'); "
        "from megatron.core.optimizer.pion import PionOptimizer; "
        "print('pass', PionOptimizer.__name__)"
    )
    pion_import = run_cmd(
        [PYTHON, "-c", pion_code],
        task_id="v22_55_external_pion_import",
        files="results/v22_55/v22_55_code_truth_gate.csv",
    )
    scan = loop_audit.scan_files()
    write_rows(OUT_ROOT / "v22_55_standard_loop_static_scan.csv", scan["rows"] or [{"status": "no_forbidden_hits"}])
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.dataset = "MNIST"
    runtime_args.seed = 0
    runtime_args.method = "wgo_diag_snr"
    runtime_args.reference_method = "adamw"
    runtime_args.label = "v22_55_runtime_trace_wgo_smoke"
    runtime_args.device = str(args.smoke_device)
    runtime_args.steps = int(args.smoke_steps)
    runtime_args.train_size = min(int(args.train_size), 128)
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
        (LOG_ROOT / "v22_55_runtime_trace_wgo_smoke_exception.log").write_text(runtime_error, encoding="utf-8")
    row = {
        "compileall_pass": int(compile_result.get("status") == "pass"),
        "py_compile_runner_pass": py_compile_runner_pass,
        "py_compile_runner_error": py_compile_runner_error,
        "full_repo_clean_import_pass": int(full_import.get("status") == "pass"),
        "external_poet_import_pass": int(poet_import.get("status") == "pass"),
        "external_pion_import_pass": int(pion_import.get("status") == "pass"),
        **scan["summary"],
        "standard_loop_runtime_trace_pass": int(runtime_status == "pass"),
        "optimizer_owned_gradient_transform_pass": iflag(runtime_row.get("optimizer_owned_gradient_transform_pass")),
        "loss_total_is_task_loss_only_for_official_WGO": iflag(runtime_row.get("loss_total_is_task_loss_only_for_official_WGO")),
        "fu_auxiliary_loss_used_official": iflag(runtime_row.get("fu_auxiliary_loss_used_official")),
        "runtime_error": runtime_error[:500],
    }
    row["part_a_pass"] = int(
        row["compileall_pass"]
        and row["py_compile_runner_pass"]
        and row["full_repo_clean_import_pass"]
        and row["standard_loop_static_scan_pass"]
        and row["standard_loop_runtime_trace_pass"]
        and row["manual_update_forbidden_scan_pass"]
        and row["optimizer_owned_gradient_transform_pass"]
        and row["loss_total_is_task_loss_only_for_official_WGO"]
        and row["fu_auxiliary_loss_used_official"] == 0
        and row["manual_param_update_detected"] == 0
        and row["no_grad_param_mutation_detected"] == 0
        and row["apply_flat_update_called"] == 0
        and row["p_data_write_detected"] == 0
        and row["copy_param_write_detected"] == 0
        and row["candidate_action_selection_used_for_runtime"] == 0
        and row["cohort_topk_selection_used"] == 0
        and row["score_selector_used"] == 0
        and row["class_weight_or_sampler_used_as_fu"] == 0
        and row["uses_validation_test_future_direction"] == 0
    )
    write_rows(OUT_ROOT / "v22_55_code_truth_gate.csv", [row])
    append_exec(
        f"{PYTHON} experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py --stage o0",
        task_id="v22_55_part_a_truth_gate_summary",
        status="pass" if row["part_a_pass"] else "fail",
        files=(
            "results/v22_55/v22_55_code_truth_gate.csv; "
            "results/v22_55/v22_55_standard_loop_static_scan.csv; "
            "results/v22_55/chunks/v22_55_v22_55_runtime_trace_wgo_smoke_summary.csv"
        ),
        note=f"Part A pass={row['part_a_pass']}; runtime={runtime_status}",
    )
    append_recap(
        "Part A standard-loop hard gate",
        md_table(
            [row],
            [
                "compileall_pass",
                "full_repo_clean_import_pass",
                "external_poet_import_pass",
                "external_pion_import_pass",
                "standard_loop_static_scan_pass",
                "standard_loop_runtime_trace_pass",
                "manual_update_forbidden_scan_pass",
                "optimizer_owned_gradient_transform_pass",
                "part_a_pass",
            ],
        ),
    )
    return row


def reanalyse_v22_54(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    src = ROOT / "results/v22_54"
    method_summary = read_rows(src / "v22_54_method_summary.csv")
    pairwise = read_rows(src / "v22_54_mlp_true_training_fu_pairwise.csv")
    final_route = read_json(src / "v22_54_final_route.json")
    repair_summaries = sorted(src.glob("v22_54_repair_*_summary.csv"))
    if not method_summary or not pairwise:
        raise RuntimeError("v22.54 artifacts missing; cannot run independent reanalysis")
    poet_row = next((r for r in method_summary if r.get("method") == "poet_official"), {})
    candidates = {"owl", "dal", "cwbl", "wctb", "wmrp", "fwnp", "cfrp", "csrp", "rcop"}
    candidate_pairwise = [r for r in pairwise if r.get("method") in candidates]
    repair_rows = []
    for path in repair_summaries:
        rows = read_rows(path)
        repair_rows.extend({"source_file": str(path.relative_to(ROOT)), **r} for r in rows)
    repair_names = ["wctb", "wmrp", "fwnp", "cfrp", "csrp", "rcop"]
    repair_summary = []
    for name in repair_names:
        group = [r for r in candidate_pairwise if str(r.get("method", "")).startswith(name)]
        extra = [r for r in repair_rows if name in r.get("source_file", "")]
        repair_summary.append(
            {
                "repair": name,
                "pairwise_rows": len(group),
                "beats_POET_rows": sum(iflag(r.get("beats_reference")) for r in group if r.get("reference_method") == "poet_official"),
                "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in group),
                "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
                "standard_loop_pass_rows": sum(iflag(r.get("standard_training_loop_pass")) for r in group),
                "summary_artifact_rows": len(extra),
            }
        )
    summary = {
        "poet_official_mean_final_NLL": fval(poet_row.get("mean_final_NLL")),
        "poet_official_no_debt_rows": poet_row.get("no_debt_rows", ""),
        "owl_dal_cwbl_beat_POET_rows": sum(iflag(r.get("beats_reference")) for r in candidate_pairwise if r.get("method") in {"owl", "dal", "cwbl"} and r.get("reference_method") == "poet_official"),
        "repair_beat_POET_rows": sum(item["beats_POET_rows"] for item in repair_summary),
        "repair_beats_best_control_rows": sum(item["beats_best_control_rows"] for item in repair_summary),
        "repair_no_debt_rows": sum(item["no_debt_rows"] for item in repair_summary),
        "standard_loop_pass_rows": sum(iflag(r.get("standard_training_loop_pass")) for r in candidate_pairwise),
        "witness_LCB_positive_rows": sum(iflag(r.get("witness_LCB_positive")) for r in candidate_pairwise),
        "first_order_task_interference_rows": sum(int(abs(fval(r.get("first_order_task_interference"), 0.0) or 0.0) <= 1.0e-4) for r in candidate_pairwise),
        "v22_54_final_route": final_route.get("final_route", ""),
        "reanalysis_pass": int(final_route.get("final_route") == "R4-AuxiliaryLossOrReweightingExplained_NoFU" and fval(poet_row.get("mean_final_NLL")) is not None),
    }
    write_rows(OUT_ROOT / "v22_55_part_b_v22_54_repair_reanalysis.csv", repair_summary)
    write_json(OUT_ROOT / "v22_55_part_b_v22_54_reanalysis_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py --stage reanalyse-v54",
        task_id="v22_55_part_b_v22_54_reanalysis",
        status="pass" if summary["reanalysis_pass"] else "fail",
        files="results/v22_55/v22_55_part_b_v22_54_reanalysis_summary.json; results/v22_55/v22_55_part_b_v22_54_repair_reanalysis.csv",
        note=f"v22_54_final_route={summary['v22_54_final_route']}",
    )
    append_recap(
        "Part B v22.54 independent reanalysis",
        "真实读取 `results/v22_54/` artifacts 后重算关键计数：\n\n"
        + "```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n"
        + md_table(repair_summary, ["repair", "pairwise_rows", "beats_POET_rows", "beats_best_control_rows", "no_debt_rows", "standard_loop_pass_rows"]),
    )
    return summary


def synthetic_case_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    device = torch_device(str(args.synthetic_device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(225500)
    rows: list[dict[str, Any]] = []

    def make_param(n: int = 64) -> Any:
        return torch.nn.Parameter(torch.zeros(n, device=device))

    def observe_from_vectors(name: str, vectors: list[Any], cfg: WGOConfig, labels: list[Any] | None = None, losses: list[Any] | None = None, margins: list[Any] | None = None) -> tuple[WitnessedGradientOperator, dict[str, Any]]:
        p = make_param(int(vectors[0].numel()))
        op = WitnessedGradientOperator([p], cfg)
        grads = [[v.reshape_as(p)] for v in vectors]
        diag = op.observe(grads, cohort_labels=labels, cohort_losses=losses, cohort_margins=margins)
        transformed = op.transform(torch.stack(vectors).mean(dim=0).reshape_as(p), p, {})
        diag["direction_cosine_with_raw"] = float(F.cosine_similarity(transformed.reshape(1, -1), torch.stack(vectors).mean(dim=0).reshape(1, -1)).item())
        return op, diag

    base = torch.randn(64, device=device, generator=gen)
    aligned = [base + 0.01 * torch.randn(64, device=device, generator=gen) for _ in range(4)]
    _, c1 = observe_from_vectors("C1", aligned, WGOConfig(random_seed=1, lambda_var=0.05))
    rows.append({"case": "C1_aligned", "pass": int(c1["gate_mean"] >= 0.8 and c1["direction_cosine_with_raw"] >= 0.95), **c1})

    noise = torch.zeros(64, device=device)
    noise[:8] = 8.0
    c2_base = base.clone()
    c2_base[:8] = 0.0
    c2_jitter = 0.01 * torch.randn(64, device=device, generator=gen)
    c2_jitter[:8] = 0.0
    vectors = [c2_base + noise, c2_base + noise * 0.8, c2_base, c2_base + c2_jitter]
    op, c2 = observe_from_vectors("C2", vectors, WGOConfig(random_seed=2, lambda_var=0.25))
    gate = op._flat_gate if op._flat_gate is not None else torch.ones(64, device=device)
    c2["gate_noise_direction"] = float(gate[:8].mean().item())
    rows.append({"case": "C2_source_specific_noise", "pass": int(c2["gate_noise_direction"] <= 0.2), **c2})

    conflict = base.clone()
    conflict[:8] = -base[:8]
    vectors = [base, base + 0.02 * torch.randn(64, device=device, generator=gen), conflict, base]
    labels = [torch.tensor([0, 0], device=device), torch.tensor([0, 1], device=device), torch.tensor([1, 1], device=device), torch.tensor([1, 0], device=device)]
    losses = [torch.tensor([0.6, 0.7], device=device), torch.tensor([0.7, 0.8], device=device), torch.tensor([1.2, 1.1], device=device), torch.tensor([0.8, 0.9], device=device)]
    margins = [torch.tensor([0.4, 0.3], device=device), torch.tensor([0.3, 0.2], device=device), torch.tensor([-0.1, -0.2], device=device), torch.tensor([0.2, 0.1], device=device)]
    op, c3 = observe_from_vectors("C3", vectors, WGOConfig(random_seed=3, lambda_var=0.20), labels, losses, margins)
    gate = op._flat_gate if op._flat_gate is not None else torch.ones(64, device=device)
    c3["minority_conflict_gate"] = float(gate[:8].mean().item())
    rows.append({"case": "C3_minority_conflict", "pass": int(c3["minority_conflict_gate"] < float(gate[8:].mean().item()) and abs(c3["gate_class_count_correlation"]) <= 0.7), **c3})

    labels_imb = [torch.zeros(8, dtype=torch.long, device=device), torch.zeros(8, dtype=torch.long, device=device), torch.ones(8, dtype=torch.long, device=device), torch.zeros(8, dtype=torch.long, device=device)]
    vectors = [base.clone() for _ in range(4)]
    _, c4 = observe_from_vectors("C4", vectors, WGOConfig(random_seed=4, lambda_var=0.05), labels_imb, losses, margins)
    rows.append({"case": "C4_class_weighting_trap", "pass": int(abs(c4["gate_class_count_correlation"]) <= 0.7), **c4})

    vectors = [base + noise, base + noise, base - noise, base - noise]
    op, c5 = observe_from_vectors("C5", vectors, WGOConfig(variant="safety_diag", random_seed=5, lambda_var=0.05, lambda_debt=0.75))
    gate = op._flat_gate if op._flat_gate is not None else torch.ones(64, device=device)
    c5["debt_heavy_component_gate"] = float(gate[:8].mean().item())
    rows.append({"case": "C5_safety_debt_conflict", "pass": int(c5["debt_heavy_component_gate"] <= 0.5), **c5})

    c6 = {"case": "C6_poet_wrapper_smoke", "pass": 0, "status": ""}
    try:
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).to(device)
        ns = argparse.Namespace(lr=1.0e-3, weight_decay=0.0, poet_block_size=4, poet_merge_interval=2, poet_lr=5.0e-4, poet_scale=0.5)
        wrapped, base_opt, _ = make_poet_model(model, ns)
        params = [p for p in wrapped.parameters() if p.requires_grad]
        op = WitnessedGradientOperator(params, WGOConfig(random_seed=6))
        opt = WitnessedOptimizerWrapper(base_opt, op)
        xb = torch.randn(16, 4, device=device, generator=gen)
        yb = torch.randint(0, 2, (16,), device=device, generator=gen)
        logits = wrapped(xb)
        loss = F.cross_entropy(logits, yb)
        cohort_grads = []
        for chunk in torch.chunk(torch.arange(16, device=device), 4):
            cg = torch.autograd.grad(F.cross_entropy(logits[chunk], yb[chunk]), params, retain_graph=True, allow_unused=True)
            cohort_grads.append(list(cg))
        op.observe(cohort_grads)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        wrapped.merge_if_needed(1)
        d = opt.diagnostics()
        c6.update(d)
        c6["pass"] = int(iflag(d.get("optimizer_owned_gradient_transform_pass")) == 1 and iflag(d.get("optimizer_step_called")) == 1)
        c6["status"] = "completed"
    except Exception as exc:
        c6["status"] = f"blocked_or_unavailable: {exc}"
    rows.append(c6)
    return rows


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows = synthetic_case_rows(args)
    write_rows(OUT_ROOT / "v22_55_part_c_synthetic_wgo_tests.csv", rows)
    summary = {
        "synthetic_rows": len(rows),
        "synthetic_pass_rows": sum(iflag(r.get("pass")) for r in rows),
        "all_synthetic_cases_pass": int(rows and sum(iflag(r.get("pass")) for r in rows) == len(rows)),
    }
    write_json(OUT_ROOT / "v22_55_part_c_synthetic_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py --stage synthetic",
        task_id="v22_55_part_c_synthetic",
        status="pass" if summary["all_synthetic_cases_pass"] else "fail",
        gpu=str(args.synthetic_device),
        files="results/v22_55/v22_55_part_c_synthetic_wgo_tests.csv; results/v22_55/v22_55_part_c_synthetic_summary.json",
    )
    append_recap(
        "Part C synthetic/mechanistic tests",
        md_table(rows, ["case", "pass", "gate_mean", "gate_std", "gate_noise_direction", "minority_conflict_gate", "debt_heavy_component_gate", "status"], 12)
        + "\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return summary


def collect_command(spec: dict[str, Any], args: argparse.Namespace, device: str) -> list[str]:
    cmd = [
        PYTHON,
        "experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py",
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
        "--wgo-lambda-var",
        str(args.wgo_lambda_var),
        "--wgo-lambda-debt",
        str(args.wgo_lambda_debt),
        "--wgo-ema-beta",
        str(args.wgo_ema_beta),
        "--wgo-block-size",
        str(args.wgo_block_size),
        "--wgo-clip-min",
        str(args.wgo_clip_min),
        "--wgo-clip-max",
        str(args.wgo_clip_max),
        "--wgo-stats-interval",
        str(args.wgo_stats_interval),
        "--wgo-param-scope",
        str(args.wgo_param_scope),
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
    return cmd


def run_matrix(args: argparse.Namespace, *, phase: str) -> list[dict[str, Any]]:
    ensure_out()
    datasets = split_csv(str(args.datasets), str)
    seeds = split_csv(str(args.seeds), int)
    methods = split_csv(str(args.methods or (DEFAULT_REFERENCE_METHODS if phase == "reference" else DEFAULT_WGO_METHODS)), str)
    specs = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                method_l = str(method).lower()
                label = f"{args.label_prefix}_{method_l}_{dataset}_s{seed}"
                spec = {"dataset": dataset, "seed": seed, "method": method_l, "label": label}
                if method_l in WGO_METHODS:
                    spec["reference_method"] = "poet_official" if method_l == "wgo_poet" else str(args.reference_method)
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
            files=f"results/v22_55/chunks/v22_55_{safe_fragment(str(spec['label']))}_summary.csv",
            gpu=str(gpu),
            timeout=int(args.collect_timeout),
        )
        return {
            **spec,
            "gpu": gpu,
            "command": " ".join(shlex.quote(str(x)) for x in cmd),
            **result,
            "summary": f"results/v22_55/chunks/v22_55_{safe_fragment(str(spec['label']))}_summary.csv",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(launch, item) for item in enumerate(specs)]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            dispatch_rows.append(row)
            write_rows(dispatch_path, dispatch_rows)
    append_exec(
        f"{PYTHON} experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py --stage {phase}",
        task_id=f"v22_55_{phase}_matrix",
        status="pass" if all(r.get("status") == "pass" for r in dispatch_rows) else "fail",
        files=f"{specs_path}; {dispatch_path}; results/v22_55/chunks/",
        note=f"rows={len(dispatch_rows)}; workers={args.workers}; gpus={args.gpus}",
    )
    return dispatch_rows


def collect_chunk_rows(include_prefixes: list[str] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_55_*_summary.csv")):
        for row in read_rows(path):
            label = str(row.get("run_label", ""))
            if include_prefixes and not any(label.startswith(prefix) for prefix in include_prefixes):
                continue
            rows.append(row)
    return rows


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    include_prefixes = split_csv(str(args.summary_include_prefixes), str) if str(args.summary_include_prefixes).strip() else []
    rows = collect_chunk_rows(include_prefixes)
    reference_rows = [r for r in rows if r.get("phase") == "reference" and r.get("run_status") == "completed"]
    wgo_rows = [r for r in rows if r.get("phase") == "WGO" and r.get("run_status") == "completed"]
    method_summary = []
    for method in sorted({r.get("method", "") for r in rows}):
        group = [r for r in rows if r.get("method") == method]
        method_summary.append(
            {
                "method": method,
                "phase": group[0].get("phase", "") if group else "",
                "rows": len(group),
                "completed_rows": sum(int(r.get("run_status") == "completed") for r in group),
                "mean_final_NLL": mean(fval(r.get("final_NLL")) for r in group),
                "mean_accuracy": mean(fval(r.get("final_accuracy")) for r in group),
                "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
                "standard_loop_pass_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in group),
                "mean_gate_mean": mean(fval(r.get("gate_mean")) for r in group),
                "mean_gate_std": mean(fval(r.get("gate_std")) for r in group),
                "mean_overhead_ratio": mean(fval(r.get("controller_overhead_ratio")) for r in group),
                "mean_gate_class_count_correlation_abs": mean(abs(fval(r.get("gate_class_count_correlation"), 0.0) or 0.0) for r in group),
                "mean_gate_loss_correlation_abs": mean(abs(fval(r.get("gate_loss_correlation"), 0.0) or 0.0) for r in group),
            }
        )
    write_rows(OUT_ROOT / "v22_55_method_summary.csv", method_summary)
    ref_by_key = {(r.get("method"), r.get("dataset"), str(r.get("seed"))): r for r in reference_rows}
    strongest_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for r in reference_rows:
        key = (r.get("dataset", ""), str(r.get("seed", "")))
        current = strongest_by_key.get(key)
        if current is None or (fval(r.get("final_NLL"), float("inf")) or float("inf")) < (fval(current.get("final_NLL"), float("inf")) or float("inf")):
            strongest_by_key[key] = r
    pairwise = []
    for r in wgo_rows:
        key2 = (r.get("dataset", ""), str(r.get("seed", "")))
        ref_method = r.get("reference_method", "")
        ref = ref_by_key.get((ref_method, r.get("dataset"), str(r.get("seed"))))
        poet = ref_by_key.get(("poet_official", r.get("dataset"), str(r.get("seed"))))
        strongest = strongest_by_key.get(key2)
        controls = [
            x for x in wgo_rows
            if x.get("dataset") == r.get("dataset")
            and str(x.get("seed")) == str(r.get("seed"))
            and x.get("method") in WGO_CONTROLS
            and x.get("reference_method") == ref_method
        ]
        best_control = min(controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        reweight_controls = [x for x in controls if x.get("method") in REWEIGHTING_CONTROLS]
        best_reweight = min(reweight_controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        final_nll = fval(r.get("final_NLL"))
        ref_nll = fval(ref.get("final_NLL")) if ref else None
        poet_nll = fval(poet.get("final_NLL")) if poet else None
        strongest_nll = fval(strongest.get("final_NLL")) if strongest else None
        control_nll = fval(best_control.get("final_NLL")) if best_control else None
        reweight_nll = fval(best_reweight.get("final_NLL")) if best_reweight else None
        pairwise.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "method": r.get("method", ""),
                "control_kind": r.get("control_kind", ""),
                "reference_method": ref_method,
                "reference_final_NLL": ref_nll,
                "poet_final_NLL": poet_nll,
                "strongest_method": strongest.get("method", "") if strongest else "",
                "strongest_final_NLL": strongest_nll,
                "final_NLL": final_nll,
                "Delta_NLL_vs_AdamW": _delta_vs(ref_by_key, r, "adamw"),
                "Delta_NLL_vs_Cautious": _delta_vs(ref_by_key, r, "cautious_adamw"),
                "Delta_NLL_vs_ScheduleFree": _delta_vs(ref_by_key, r, "schedule_free_adamw_local"),
                "Delta_NLL_vs_POET": (final_nll - poet_nll) if final_nll is not None and poet_nll is not None else "",
                "Delta_NLL_vs_Pion": _delta_vs(ref_by_key, r, "pion_oet_sphere_official"),
                "Delta_NLL_vs_strongest": (final_nll - strongest_nll) if final_nll is not None and strongest_nll is not None else "",
                "beats_reference": int(final_nll is not None and ref_nll is not None and final_nll < ref_nll),
                "beats_POET_or_external_OET": int(final_nll is not None and poet_nll is not None and final_nll < poet_nll),
                "beats_strongest": int(final_nll is not None and strongest_nll is not None and final_nll < strongest_nll),
                "best_control_method": best_control.get("method", "") if best_control else "",
                "best_control_final_NLL": control_nll,
                "beats_best_control": int(final_nll is not None and control_nll is not None and final_nll < control_nll),
                "beats_all_controls": int(final_nll is not None and bool(controls) and all(final_nll < (fval(x.get("final_NLL"), float("inf")) or float("inf")) for x in controls)),
                "best_reweighting_control_method": best_reweight.get("method", "") if best_reweight else "",
                "beats_all_reweighting_controls": int(final_nll is not None and bool(reweight_controls) and all(final_nll < (fval(x.get("final_NLL"), float("inf")) or float("inf")) for x in reweight_controls)),
                "no_ECE_Brier_tail_debt": r.get("no_ECE_Brier_tail_debt", ""),
                "AUC_loss_time_improvement": int((fval(r.get("AUC_loss_time")) or float("inf")) < (fval(ref.get("AUC_loss_time")) or float("inf"))) if ref else 0,
                "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
                "gate_class_count_correlation_abs": abs(fval(r.get("gate_class_count_correlation"), 0.0) or 0.0),
                "gate_loss_correlation_abs": abs(fval(r.get("gate_loss_correlation"), 0.0) or 0.0),
                "standard_loop_hard_gate_pass": r.get("standard_loop_hard_gate_pass", ""),
                "gate_mean": r.get("gate_mean", ""),
                "gate_std": r.get("gate_std", ""),
            }
        )
    write_rows(OUT_ROOT / "v22_55_mlp_wgo_pairwise.csv", pairwise)
    candidate_pairwise = [r for r in pairwise if r.get("method") in WGO_CANDIDATES]
    tier0_total = len([r for r in candidate_pairwise if r.get("dataset") in {"MNIST", "FashionMNIST", "KMNIST"}])
    route = {
        "reference_rows": len(reference_rows),
        "wgo_rows": len(wgo_rows),
        "candidate_wgo_rows": len(candidate_pairwise),
        "tier0_candidate_rows": tier0_total,
        "standard_loop_pass_candidate_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in candidate_pairwise),
        "beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in candidate_pairwise),
        "beats_POET_or_external_OET_rows": sum(iflag(r.get("beats_POET_or_external_OET")) for r in candidate_pairwise),
        "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in candidate_pairwise),
        "beats_all_reweighting_controls_rows": sum(iflag(r.get("beats_all_reweighting_controls")) for r in candidate_pairwise),
        "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in candidate_pairwise),
        "AUC_loss_time_improvement_rows": sum(iflag(r.get("AUC_loss_time_improvement")) for r in candidate_pairwise),
        "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in candidate_pairwise),
        "gate_class_corr_le_070_rows": sum(int((fval(r.get("gate_class_count_correlation_abs"), float("inf")) or float("inf")) <= 0.70) for r in candidate_pairwise),
        "gate_loss_corr_le_070_rows": sum(int((fval(r.get("gate_loss_correlation_abs"), float("inf")) or float("inf")) <= 0.70) for r in candidate_pairwise),
        "mlp_exploration_pass": 0,
        "final_route": "",
    }
    route["mlp_exploration_pass"] = int(
        tier0_total >= 9
        and route["beats_strongest_rows"] >= 5
        and route["beats_POET_or_external_OET_rows"] >= 5
        and route["beats_best_control_rows"] >= 6
        and route["beats_all_reweighting_controls_rows"] >= 6
        and route["no_debt_rows"] >= 7
        and route["AUC_loss_time_improvement_rows"] >= 5
        and route["overhead_le_035_rows"] >= 8
        and route["gate_class_corr_le_070_rows"] >= 8
        and route["gate_loss_corr_le_070_rows"] >= 8
        and route["standard_loop_pass_candidate_rows"] == len(candidate_pairwise)
    )
    route["final_route"] = classify_route(candidate_pairwise, route)
    write_json(OUT_ROOT / "v22_55_final_route.json", route)
    skipped = [
        {"part": "G_layerwise", "status": "skipped", "reason": "MLP+WGO exploration gate not opened" if not route["mlp_exploration_pass"] else "eligible_not_run_by_default"},
        {"part": "H_KAN", "status": "skipped", "reason": "KAN official matrix starts only after MLP+WGO gate passes" if not route["mlp_exploration_pass"] else "eligible_not_run_by_default"},
        {"part": "I_continual_grokking", "status": "skipped", "reason": "pressure tests cannot override standard MLP+WGO failure" if not route["mlp_exploration_pass"] else "eligible_not_run_by_default"},
    ]
    write_rows(OUT_ROOT / "v22_55_downstream_gate_status.csv", skipped)
    append_exec(
        f"{PYTHON} experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py --stage summarize",
        task_id="v22_55_summarize",
        status="pass",
        files=(
            "results/v22_55/v22_55_method_summary.csv; results/v22_55/v22_55_mlp_wgo_pairwise.csv; "
            "results/v22_55/v22_55_final_route.json; results/v22_55/v22_55_downstream_gate_status.csv"
        ),
        note=f"final_route={route['final_route']}",
    )
    append_recap(
        "v22.55 MLP WGO summary and route",
        "Method summary:\n\n"
        + md_table(method_summary, ["method", "phase", "rows", "completed_rows", "mean_final_NLL", "no_debt_rows", "standard_loop_pass_rows", "mean_gate_mean", "mean_overhead_ratio"], 32)
        + "\nCandidate pairwise rows:\n\n"
        + md_table(candidate_pairwise, ["dataset", "seed", "method", "reference_method", "Delta_NLL_vs_strongest", "beats_strongest", "beats_best_control", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 40)
        + "\nRoute JSON:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return route


def _delta_vs(ref_by_key: dict[tuple[str, str, str], dict[str, str]], row: dict[str, str], ref_method: str) -> Any:
    ref = ref_by_key.get((ref_method, row.get("dataset", ""), str(row.get("seed", ""))))
    final_nll = fval(row.get("final_NLL"))
    ref_nll = fval(ref.get("final_NLL")) if ref else None
    return (final_nll - ref_nll) if final_nll is not None and ref_nll is not None else ""


def classify_route(candidate_pairwise: list[dict[str, Any]], route: dict[str, Any]) -> str:
    if not candidate_pairwise:
        return "R0-TrainingLoopBoundaryFailed" if route.get("reference_rows", 0) == 0 else "R1-WGOUnitOnly"
    if route["standard_loop_pass_candidate_rows"] != len(candidate_pairwise):
        return "R0-TrainingLoopBoundaryFailed"
    if route.get("mlp_exploration_pass"):
        return "R6-MLPWGOExplorationOpened"
    gate_means = [fval(r.get("gate_mean")) for r in candidate_pairwise]
    gate_stds = [fval(r.get("gate_std")) for r in candidate_pairwise]
    if gate_means and mean(gate_means) is not None and mean(gate_stds) is not None and (mean(gate_means) or 0.0) >= 0.95 and (mean(gate_stds) or 1.0) <= 0.03:
        return "R2-WGOIdentityCollapse_NoEffectiveFU"
    if any((fval(r.get("gate_class_count_correlation_abs"), 0.0) or 0.0) > 0.80 or (fval(r.get("gate_loss_correlation_abs"), 0.0) or 0.0) > 0.80 for r in candidate_pairwise):
        return "R3-ReweightingExplained_NoFU"
    if route["beats_strongest_rows"] == 0 and route["beats_POET_or_external_OET_rows"] == 0:
        return "R5-FUWeakOptimizerPatchOnly"
    if route["beats_best_control_rows"] < 6:
        return "R4-RandomGateOrAuxiliaryExplained_NoFU"
    if route["no_debt_rows"] < 7:
        return "SafetyDebtBlocksWGO"
    return "ExternalOETDominates_CurrentFUInsufficient"


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
            circles.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="#2563eb"><title>{label}: {x_key}={x:.6g}, {y_key}={y:.6g}</title></circle>')
        body = "\n".join(circles)
    else:
        body = '<text x="70" y="210" font-size="18" fill="#555">No completed numeric rows available.</text>'
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="40" y="38" font-size="22" font-family="Arial" fill="#111">{title}</text>
<line x1="70" y1="350" x2="660" y2="350" stroke="#444"/>
<line x1="70" y1="70" x2="70" y2="350" stroke="#444"/>
<text x="310" y="395" font-size="14" font-family="Arial" fill="#333">{x_key}</text>
<text x="12" y="220" font-size="14" font-family="Arial" fill="#333" transform="rotate(-90 12 220)">{y_key}</text>
{body}
</svg>
""",
        encoding="utf-8",
    )


def run_figures(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    pairwise = read_rows(OUT_ROOT / "v22_55_mlp_wgo_pairwise.csv")
    chunks = collect_chunk_rows()
    route = read_json(OUT_ROOT / "v22_55_final_route.json")
    write_svg(FIG_ROOT / "v22_55_wgo_gate_distribution.svg", "WGO Gate Distribution", chunks, "gate_mean", "gate_std")
    write_svg(FIG_ROOT / "v22_55_wgo_transfer_vs_noise.svg", "Transfer vs Noise", chunks, "source_witness_transfer_LCB", "noise_attenuation_rate")
    write_svg(FIG_ROOT / "v22_55_wgo_control_gap_by_dataset.svg", "Control Gap by Dataset", pairwise, "final_NLL", "best_control_final_NLL")
    write_svg(FIG_ROOT / "v22_55_poet_vs_wgo_pairwise.svg", "POET vs WGO", pairwise, "poet_final_NLL", "final_NLL")
    write_svg(FIG_ROOT / "v22_55_gate_correlation_heatmap.svg", "Gate Correlation Audit", chunks, "gate_class_count_correlation", "gate_loss_correlation")
    write_svg(FIG_ROOT / "v22_55_kan_vs_mlp_carrier_matrix.svg", f"KAN skipped: {route.get('final_route', 'unknown')}", [], "x", "y")
    files = sorted(str(p.relative_to(ROOT)) for p in FIG_ROOT.glob("v22_55_*.svg"))
    summary = {"figures": files, "figure_count": len(files)}
    write_json(OUT_ROOT / "v22_55_figure_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py --stage figures",
        task_id="v22_55_figures",
        status="pass",
        files="; ".join(files),
    )
    append_recap("Visualizations", "生成 SVG：\n\n" + "\n".join(f"- `{x}`" for x in files))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all", choices=["all", "o0", "reanalyse-v54", "synthetic", "collect", "reference", "wgo", "summarize", "figures"])
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="adamw")
    p.add_argument("--reference-method", default="poet_official")
    p.add_argument("--label", default="v22_55_collect")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default="")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label-prefix", default="v22_55_reference")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--steps", type=int, default=90)
    p.add_argument("--smoke-steps", type=int, default=4)
    p.add_argument("--smoke-device", default="cuda:0")
    p.add_argument("--synthetic-device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--cohorts", type=int, default=4)
    p.add_argument("--cohort-mode", default="random", choices=["random", "label_loss_stratified"])
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--wgo-lambda-var", type=float, default=0.25)
    p.add_argument("--wgo-lambda-debt", type=float, default=0.25)
    p.add_argument("--wgo-ema-beta", type=float, default=0.90)
    p.add_argument("--wgo-block-size", type=int, default=256)
    p.add_argument("--wgo-clip-min", type=float, default=0.0)
    p.add_argument("--wgo-clip-max", type=float, default=1.0)
    p.add_argument("--wgo-stats-interval", type=int, default=1)
    p.add_argument("--wgo-param-scope", default="all", choices=["all", "last_layer"])
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
        if args.stage == "reanalyse-v54":
            print(json.dumps(reanalyse_v22_54(args), ensure_ascii=False, sort_keys=True))
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
        if args.stage == "wgo":
            if args.label_prefix == "v22_55_reference":
                args.label_prefix = f"v22_55_wgo_over_{safe_fragment(args.reference_method)}"
            run_matrix(args, phase="wgo")
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
            reanalysis = reanalyse_v22_54(args)
            if not iflag(reanalysis.get("reanalysis_pass")):
                summarize(args)
                run_figures(args)
                return 3
            synth = run_synthetic(args)
            if not iflag(synth.get("all_synthetic_cases_pass")):
                summarize(args)
                run_figures(args)
                return 4
            args.label_prefix = "v22_55_reference"
            run_matrix(args, phase="reference")
            args.label_prefix = f"v22_55_wgo_over_{safe_fragment(args.reference_method)}"
            run_matrix(args, phase="wgo")
            summarize(args)
            run_figures(args)
            return 0
    except Exception:
        err_path = LOG_ROOT / f"v22_55_stage_{safe_fragment(args.stage)}_exception.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        append_exec(
            f"{PYTHON} experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py --stage {args.stage}",
            task_id=f"v22_55_stage_{args.stage}_exception",
            status="exception",
            files=str(err_path),
            exit_code="exception",
        )
        print(err_path.read_text(encoding="utf-8"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
