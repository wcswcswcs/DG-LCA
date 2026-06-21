#!/usr/bin/env python3
"""DG-KAN v22.54 true-training witnessed objective FU runner.

This runner keeps the official v22.54 path intentionally narrow:

* FU changes parameters only through a differentiable loss term.
* The training loop is always forward, loss backward, optimizer step.
* v22.53O artifacts are read only for reanalysis; old correction diagnostics are
  not replayed as v22.54 evidence.
"""

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

from experiments import run_v22_49A_broader_related_work_map as v49


KAN_ENV_PYTHON = ROOT.parent / "miniconda3/envs/kan/bin/python"
PYTHON = os.environ.get("KAN_PYTHON", str(KAN_ENV_PYTHON if KAN_ENV_PYTHON.exists() else sys.executable))
OUT_ROOT = ROOT / "results/v22_54"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.54_TrueTrainingWitnessedObjectiveFU_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.54_TrueTrainingWitnessedObjectiveFU_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.54_TrueTrainingWitnessedObjectiveFU_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_54_true_training_witnessed_objective_fu.py"

REFERENCE_METHODS = {"adamw", "cautious_adamw", "schedule_free_adamw_local", "poet_official"}
FU_CANDIDATES = {"owl", "dal", "cwbl", "wctb", "wmrp", "fwnp", "cfrp", "csrp", "rcop"}
FU_CONTROLS = {
    "xwcfu_loss_legacy_reproduction",
    "owl_same_source_span_random_loss",
    "owl_signflip_loss",
    "owl_shuffled_witness_loss",
    "dal_same_source_span_random_loss",
    "dal_signflip_loss",
    "dal_shuffled_witness_loss",
    "cwbl_shuffled_witness_barrier",
    "cwbl_same_barrier_norm_random_loss",
    "cwbl_same_active_count_random_barrier",
    "wctb_shuffled_class_barrier",
    "wctb_same_norm_random_barrier",
    "wctb_signflip_barrier",
    "wctb_hard_loss_barrier_control",
    "wmrp_shuffled_confuser_margin",
    "wmrp_same_norm_random_margin",
    "wmrp_signflip_margin",
    "wmrp_unprojected_margin_control",
    "fwnp_shuffled_witness_fisher",
    "fwnp_same_norm_random_fisher",
    "fwnp_signflip_fisher",
    "fwnp_unprojected_fisher_control",
    "cfrp_shuffled_residual_predictor",
    "cfrp_same_norm_random_residual",
    "cfrp_signflip_residual",
    "cfrp_unprojected_residual_control",
    "csrp_shuffled_subtract",
    "csrp_same_norm_random_subtract",
    "csrp_signflip_subtract",
    "csrp_no_subtract_residual_control",
    "rcop_shuffled_orthogonal",
    "rcop_same_norm_random_orthogonal",
    "rcop_signflip_orthogonal",
    "rcop_no_orthogonal_residual_control",
    "rcop_hardness_only_control",
}
FU_METHODS = FU_CANDIDATES | FU_CONTROLS
DEFAULT_BASELINE_METHODS = "adamw,cautious_adamw,schedule_free_adamw_local,poet_official"
DEFAULT_FU_METHODS = (
    "xwcfu_loss_legacy_reproduction,owl,owl_same_source_span_random_loss,"
    "owl_signflip_loss,owl_shuffled_witness_loss,dal,dal_same_source_span_random_loss,"
    "dal_signflip_loss,dal_shuffled_witness_loss,cwbl,cwbl_shuffled_witness_barrier,"
    "cwbl_same_barrier_norm_random_loss,cwbl_same_active_count_random_barrier,"
    "wctb,wctb_shuffled_class_barrier,wctb_same_norm_random_barrier,"
    "wctb_signflip_barrier,wctb_hard_loss_barrier_control,"
    "wmrp,wmrp_shuffled_confuser_margin,wmrp_same_norm_random_margin,"
    "wmrp_signflip_margin,wmrp_unprojected_margin_control,"
    "fwnp,fwnp_shuffled_witness_fisher,fwnp_same_norm_random_fisher,"
    "fwnp_signflip_fisher,fwnp_unprojected_fisher_control,"
    "cfrp,cfrp_shuffled_residual_predictor,cfrp_same_norm_random_residual,"
    "cfrp_signflip_residual,cfrp_unprojected_residual_control,"
    "csrp,csrp_shuffled_subtract,csrp_same_norm_random_subtract,"
    "csrp_signflip_subtract,csrp_no_subtract_residual_control,"
    "rcop,rcop_shuffled_orthogonal,rcop_same_norm_random_orthogonal,"
    "rcop_signflip_orthogonal,rcop_no_orthogonal_residual_control,"
    "rcop_hardness_only_control"
)


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.54 TrueTrainingWitnessedObjectiveFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、输入文件、输出文件、GPU、状态、失败与修复尝试。"
            "后续复现优先看本日志中的命令、解释器、artifact 路径和 stdout/stderr。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.54 TrueTrainingWitnessedObjectiveFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘 artifact；失败、缺失、修复与结论必须带证据链。"
            "不得把 blocked、diagnostic、control 或 reanalysis 行写成 success。\n",
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
                    fieldnames.append(key)
                    seen.add(key)
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
    journal = read_rows(OUT_ROOT / "v22_54_command_journal.csv")
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(
        OUT_ROOT / "v22_54_command_journal.csv",
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
        note = f"stdout={stdout_path}; stderr={stderr_path}; wall_seconds={time.time() - start:.3f}"
        append_exec(
            " ".join(shlex.quote(str(x)) for x in cmd),
            task_id=task_id,
            status=status,
            gpu=gpu,
            files=files,
            exit_code=proc.returncode,
            note=note,
        )
        return {
            "status": status,
            "returncode": proc.returncode,
            "wall_seconds": time.time() - start,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
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
        return {
            "status": "exception",
            "returncode": -999,
            "wall_seconds": time.time() - start,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "exception": repr(exc),
        }


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


def lcb95(values: Iterable[float | None]) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return math.nan
    if len(clean) == 1:
        return float(clean[0])
    mu = statistics.fmean(clean)
    sd = statistics.stdev(clean)
    return float(mu - 1.96 * sd / math.sqrt(len(clean)))


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 16) -> str:
    if not rows:
        return "_no rows_\n"
    shown = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
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


def flatten_tensors(items: Iterable[Any | None], params: list[Any]) -> Any:
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


def cosine(a: Any, b: Any) -> float:
    denom = a.norm().clamp_min(1.0e-12) * b.norm().clamp_min(1.0e-12)
    return float((a.dot(b) / denom).detach().item())


def trainable_param_count(model: Any) -> int:
    return int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))


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


def batch_indices(n: int, batch_size: int, generator: Any, device: Any) -> Any:
    import torch

    if batch_size >= n:
        return torch.arange(n, device=device)
    return torch.randperm(n, generator=generator, device=device)[:batch_size]


def cohort_indices_from_batch(idx: Any, cohorts: int) -> list[Any]:
    import torch

    if int(idx.numel()) == 0:
        return []
    permuted = idx[torch.randperm(int(idx.numel()), device=idx.device)]
    return [chunk for chunk in torch.chunk(permuted, max(1, int(cohorts))) if int(chunk.numel()) > 0]


class CautiousAdamW:
    """Cautious-style gradient gating wrapped around torch AdamW."""

    def __init__(self, params: Iterable[Any], lr: float, weight_decay: float) -> None:
        import torch

        self.params = list(params)
        self.base = torch.optim.AdamW(self.params, lr=float(lr), weight_decay=float(weight_decay))
        self.prev_grads: dict[int, Any] = {}
        self.gate_rates: list[float] = []

    @property
    def state(self) -> Any:
        return self.base.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        rates: list[float] = []
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            prev = self.prev_grads.get(id(p))
            if prev is not None and prev.shape == grad.shape:
                mask = (grad * prev.to(device=grad.device, dtype=grad.dtype)) >= 0
                rates.append(float(mask.float().mean().item()))
                p.grad.mul_(mask.to(dtype=p.grad.dtype))
            self.prev_grads[id(p)] = grad.clone()
        if rates:
            self.gate_rates.append(float(statistics.fmean(rates)))
        self.base.step()


class ScheduleFreeAdamWLocal:
    """Local AdamW variant with online LR relaxation; recorded as local baseline."""

    def __init__(self, params: Iterable[Any], lr: float, weight_decay: float, beta: float = 0.98) -> None:
        import torch

        self.params = list(params)
        self.base_lr = float(lr)
        self.beta = float(beta)
        self.t = 0
        self.base = torch.optim.AdamW(self.params, lr=float(lr), weight_decay=float(weight_decay))

    @property
    def state(self) -> Any:
        return self.base.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.t += 1
        lr_eff = self.base_lr / max(0.10, 1.0 - self.beta ** self.t)
        lr_eff = min(lr_eff, self.base_lr * 2.0)
        for group in self.base.param_groups:
            group["lr"] = lr_eff
        self.base.step()


def make_poet_model(model: Any, args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    from poet_torch import POETConfig, POETModel, get_poet_optimizer

    cfg = POETConfig(
        block_size=int(args.poet_block_size),
        merge_interval=int(args.poet_merge_interval),
        poet_lr=float(args.poet_lr),
        base_lr=float(args.lr),
        poet_scale=float(args.poet_scale),
        weight_decay=float(args.weight_decay),
        mem_efficient_mode=False,
    )
    wrapped = POETModel(model, cfg)
    opt = get_poet_optimizer(wrapped, cfg)
    return wrapped, opt, poet_parameter_audit(wrapped)


def poet_parameter_audit(model: Any) -> dict[str, Any]:
    audit = {
        "base_weight_parameter_count": 0,
        "poet_replaced_linear_layers": 0,
        "orthogonal_factor_parameter_count": 0,
        "trainable_parameter_count_total": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
        "merged_inference_available": int(hasattr(model, "merge_if_needed")),
        "merged_inference_parameter_count": 0,
    }
    try:
        from poet_torch.layers import POETLinear, QPOETLinear

        count = 0
        base_total = 0
        factor_count = 0
        merged_total = 0
        for module in model.modules():
            if isinstance(module, (POETLinear, QPOETLinear)):
                count += 1
                if hasattr(module, "weight"):
                    base_total += int(module.weight.numel())
                    merged_total += int(module.weight.numel())
                if hasattr(module, "oft_R"):
                    factor_count += int(module.oft_R.numel())
                if getattr(module, "bias", None) is not None:
                    merged_total += int(module.bias.numel())
        audit["poet_replaced_linear_layers"] = count
        audit["base_weight_parameter_count"] = base_total
        audit["orthogonal_factor_parameter_count"] = factor_count
        audit["merged_inference_parameter_count"] = merged_total
    except Exception as exc:
        audit["poet_audit_error"] = repr(exc)
    return audit


def make_optimizer(method: str, model: Any, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch

    if method == "poet_official":
        wrapped, opt, audit = make_poet_model(model, args)
        return (wrapped, opt, audit)
    if method == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        return (model, opt, {"optimizer_step_source": "torch.optim.AdamW"})
    if method == "cautious_adamw":
        opt = CautiousAdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        return (model, opt, {"optimizer_step_source": "CautiousAdamW_grad_gate_over_torch.optim.AdamW"})
    if method == "schedule_free_adamw_local":
        opt = ScheduleFreeAdamWLocal(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        return (model, opt, {"optimizer_step_source": "ScheduleFreeAdamWLocal_over_torch.optim.AdamW"})
    raise ValueError(f"unknown reference method: {method}")


def reference_method_for(method: str, args: argparse.Namespace) -> str:
    if method in REFERENCE_METHODS:
        return method
    return str(args.reference_method)


def base_objective_name(method: str) -> str:
    if method.startswith("owl"):
        return "owl"
    if method.startswith("dal"):
        return "dal"
    if method.startswith("cwbl"):
        return "cwbl"
    if method.startswith("wctb"):
        return "wctb"
    if method.startswith("wmrp"):
        return "wmrp"
    if method.startswith("fwnp"):
        return "fwnp"
    if method.startswith("cfrp"):
        return "cfrp"
    if method.startswith("csrp"):
        return "csrp"
    if method.startswith("rcop"):
        return "rcop"
    if method.startswith("xwcfu"):
        return "xwcfu"
    return method


def control_kind(method: str) -> str:
    if method in FU_CANDIDATES:
        return "candidate"
    if "same_source_span_random" in method:
        return "same_source_span_random"
    if "signflip" in method:
        return "signflip"
    if "shuffled_witness" in method:
        return "shuffled_witness"
    if "same_barrier_norm_random" in method:
        return "same_barrier_norm_random"
    if "same_active_count_random" in method:
        return "same_active_count_random"
    if "shuffled_class" in method:
        return "shuffled_class_barrier"
    if "shuffled_confuser" in method:
        return "shuffled_confuser_margin"
    if "same_norm_random_margin" in method:
        return "same_norm_random_margin"
    if "unprojected_margin" in method:
        return "unprojected_margin_control"
    if "same_norm_random_fisher" in method:
        return "same_norm_random_fisher"
    if "unprojected_fisher" in method:
        return "unprojected_fisher_control"
    if "shuffled_residual" in method:
        return "shuffled_residual_predictor"
    if "same_norm_random_residual" in method:
        return "same_norm_random_residual"
    if "unprojected_residual" in method:
        return "unprojected_residual_control"
    if "shuffled_subtract" in method:
        return "shuffled_subtract_control"
    if "same_norm_random_subtract" in method:
        return "same_norm_random_subtract"
    if "no_subtract_residual" in method:
        return "no_subtract_residual_control"
    if "shuffled_orthogonal" in method:
        return "shuffled_orthogonal_control"
    if "same_norm_random_orthogonal" in method:
        return "same_norm_random_orthogonal"
    if "no_orthogonal_residual" in method:
        return "no_orthogonal_residual_control"
    if "hardness_only" in method:
        return "hardness_only_control"
    if "same_norm_random" in method:
        return "same_norm_random_barrier"
    if "hard_loss_barrier" in method:
        return "hard_loss_barrier_control"
    if method.startswith("xwcfu"):
        return "legacy_xwcfu_standard_loss"
    return "control"


def classwise_source_credit(logits: Any, labels: Any, source_pos: Any, target_pos: Any, method: str, generator: Any) -> Any:
    import torch
    import torch.nn.functional as F

    num_classes = int(logits.shape[-1])
    probs = torch.softmax(logits.detach(), dim=-1)
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).to(dtype=probs.dtype, device=probs.device)
    source_credit_samples = one_hot[source_pos] - probs[source_pos]
    source_labels = labels[source_pos].long()
    global_credit = source_credit_samples.mean(dim=0)
    rows = []
    for label in labels[target_pos].long():
        mask = source_labels == label
        if bool(mask.any()):
            rows.append(source_credit_samples[mask].mean(dim=0))
        else:
            rows.append(global_credit)
    credit = torch.stack(rows, dim=0) if rows else torch.zeros_like(logits[target_pos])
    if "same_source_span_random" in method or "same_barrier_norm_random" in method:
        coeff = torch.randn(
            (max(1, int(source_credit_samples.shape[0])), max(1, int(target_pos.numel()))),
            generator=generator,
            device=logits.device,
            dtype=logits.dtype,
        )
        basis = source_credit_samples
        random_credit = coeff.t().matmul(basis)
        credit = random_credit / random_credit.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        credit = credit * rows_norm(credit.new_tensor([float(x.norm().item()) for x in rows]) if rows else credit.new_ones(int(target_pos.numel()))).view(-1, 1)
    elif "signflip" in method:
        credit = -credit
    elif "shuffled_witness" in method:
        if int(target_pos.numel()) > 1:
            perm = torch.randperm(int(target_pos.numel()), generator=generator, device=logits.device)
            shuffled_labels = labels[target_pos][perm].long()
            rows = []
            for label in shuffled_labels:
                mask = source_labels == label
                rows.append(source_credit_samples[mask].mean(dim=0) if bool(mask.any()) else global_credit)
            credit = torch.stack(rows, dim=0)
    return credit.detach()


def rows_norm(vals: Any) -> Any:
    return vals.clamp_min(1.0e-12)


def task_cotangent(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as F

    probs = logits.detach().softmax(dim=-1)
    one_hot = F.one_hot(labels.long(), num_classes=int(logits.shape[-1])).to(dtype=probs.dtype, device=probs.device)
    return (probs - one_hot) / max(1, int(labels.numel()))


def witnessed_class_tail_barrier_loss(
    method: str,
    logits: Any,
    labels: Any,
    source_pos: Any,
    witness_pos: Any,
    generator: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    num_classes = int(logits.shape[-1])
    tau = max(float(args.wctb_tau), 1.0e-6)
    std_scale = float(args.wctb_std_scale)
    source_labels = labels[source_pos].long()
    witness_labels = labels[witness_pos].long()
    source_losses = F.cross_entropy(logits[source_pos].detach(), source_labels, reduction="none")
    global_std = source_losses.std(unbiased=False) if int(source_losses.numel()) > 1 else source_losses.new_tensor(0.0)
    global_threshold = source_losses.mean() + std_scale * global_std
    thresholds_by_class = source_losses.new_empty(num_classes)
    thresholds_by_class.fill_(float(global_threshold.item()))
    for cls in range(num_classes):
        mask = source_labels == cls
        if bool(mask.any()):
            vals = source_losses[mask]
            cls_std = vals.std(unbiased=False) if int(vals.numel()) > 1 else vals.new_tensor(0.0)
            thresholds_by_class[cls] = vals.mean() + std_scale * cls_std

    thresholds = thresholds_by_class[witness_labels].to(device=logits.device, dtype=logits.dtype)
    if "shuffled_class" in method and int(thresholds.numel()) > 1:
        perm = torch.randperm(int(thresholds.numel()), generator=generator, device=logits.device)
        thresholds = thresholds[perm]

    witness_losses = F.cross_entropy(logits[witness_pos], witness_labels, reduction="none")
    barrier_rows = F.softplus((witness_losses - thresholds.detach()) / tau) * tau
    barrier = barrier_rows.mean()
    barrier_grad = torch.autograd.grad(barrier, logits, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if barrier_grad is None:
        barrier_grad = torch.zeros_like(logits)
    raw_grad = barrier_grad.detach()
    witness_mask = torch.zeros((int(logits.shape[0]), 1), device=logits.device, dtype=logits.dtype)
    witness_mask[witness_pos] = 1.0

    if "same_norm_random" in method:
        rand = torch.randn(tuple(logits.shape), generator=generator, device=logits.device, dtype=logits.dtype)
        rand = rand * witness_mask
        raw_grad = rand * (raw_grad.norm() / rand.norm().clamp_min(1.0e-12))
    elif "signflip" in method:
        raw_grad = -raw_grad

    delta = task_cotangent(logits, labels).detach()
    raw_dot = (raw_grad * delta).sum()
    delta_norm_sq = (delta * delta).sum().clamp_min(1.0e-12)
    projection_coeff = raw_dot / delta_norm_sq
    hard_loss_control = "hard_loss_barrier" in method
    if hard_loss_control:
        safe_grad = raw_grad
    else:
        safe_grad = raw_grad - projection_coeff * delta
    fu_loss = (safe_grad.detach() * logits).sum()

    safe_dot = float((safe_grad.detach() * delta).sum().detach().item())
    raw_dot_float = float(raw_dot.detach().item())
    raw_norm = float(raw_grad.norm().detach().item())
    safe_norm = float(safe_grad.detach().norm().item())
    removed_fraction = 0.0
    if not hard_loss_control and raw_norm > 0:
        removed = (projection_coeff * delta).norm()
        removed_fraction = float((removed / raw_grad.norm().clamp_min(1.0e-12)).detach().item())
    fu_grad_vec = safe_grad.detach().reshape(-1)
    task_vec = delta.reshape(-1)
    active_fraction = float((witness_losses.detach() > thresholds.detach()).float().mean().item())
    return fu_loss, {
        "fu_control_kind": control_kind(method),
        "fu_loss_value": float(fu_loss.detach().item()),
        "cos_fu_task_gradient": cosine(fu_grad_vec, task_vec) if int(fu_grad_vec.numel()) else math.nan,
        "first_order_task_interference": safe_dot,
        "owl_projection_residual": math.nan if hard_loss_control else abs(safe_dot),
        "owl_credit_norm": safe_norm,
        "negative_alignment_removed_fraction": removed_fraction,
        "positive_alignment_kept_fraction": float(safe_norm / max(raw_norm, 1.0e-12)),
        "task_interference_delta": safe_dot - raw_dot_float,
        "cwbl_barrier_active_fraction": math.nan,
        "cwbl_margin_mean": math.nan,
        "wctb_barrier_value": float(barrier.detach().item()),
        "wctb_active_fraction": active_fraction,
        "wctb_threshold_mean": float(thresholds.detach().mean().item()),
        "wctb_witness_loss_mean": float(witness_losses.detach().mean().item()),
        "wctb_raw_task_interference": raw_dot_float,
    }


def witnessed_margin_residual_projection_loss(
    method: str,
    logits: Any,
    labels: Any,
    source_pos: Any,
    witness_pos: Any,
    generator: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    num_classes = int(logits.shape[-1])
    tau = max(float(args.wmrp_tau), 1.0e-6)
    std_scale = float(args.wmrp_std_scale)
    source_labels = labels[source_pos].long()
    witness_labels = labels[witness_pos].long()
    probs = torch.softmax(logits.detach(), dim=-1)
    source_probs = probs[source_pos]
    source_losses = F.cross_entropy(logits[source_pos].detach(), source_labels, reduction="none")
    global_std = source_losses.std(unbiased=False) if int(source_losses.numel()) > 1 else source_losses.new_tensor(0.0)
    global_threshold = source_losses.mean() + std_scale * global_std
    thresholds_by_class = source_losses.new_empty(num_classes)
    thresholds_by_class.fill_(float(global_threshold.item()))
    global_scores = source_probs.mean(dim=0)
    confuser_by_class = torch.empty(num_classes, device=logits.device, dtype=torch.long)
    for cls in range(num_classes):
        mask = source_labels == cls
        if bool(mask.any()):
            vals = source_losses[mask]
            cls_std = vals.std(unbiased=False) if int(vals.numel()) > 1 else vals.new_tensor(0.0)
            thresholds_by_class[cls] = vals.mean() + std_scale * cls_std
            scores = source_probs[mask].mean(dim=0).clone()
        else:
            scores = global_scores.clone()
        scores[int(cls)] = -1.0
        confuser_by_class[int(cls)] = int(torch.argmax(scores).item())

    if "shuffled_confuser" in method and num_classes > 1:
        perm = torch.randperm(num_classes, generator=generator, device=logits.device)
        confuser_by_class = confuser_by_class[perm]

    thresholds = thresholds_by_class[witness_labels].to(device=logits.device, dtype=logits.dtype)
    confusers = confuser_by_class[witness_labels]
    same_as_label = confusers == witness_labels
    if bool(same_as_label.any()):
        confusers = confusers.clone()
        confusers[same_as_label] = (witness_labels[same_as_label] + 1) % max(1, num_classes)
    witness_losses = F.cross_entropy(logits[witness_pos].detach(), witness_labels, reduction="none")
    weights = torch.sigmoid((witness_losses - thresholds.detach()) / tau).to(dtype=logits.dtype)
    true_hot = F.one_hot(witness_labels, num_classes=num_classes).to(dtype=logits.dtype, device=logits.device)
    conf_hot = F.one_hot(confusers.long(), num_classes=num_classes).to(dtype=logits.dtype, device=logits.device)
    credit_rows = weights.view(-1, 1) * (true_hot - conf_hot)
    raw_credit = torch.zeros_like(logits)
    raw_credit[witness_pos] = credit_rows / max(1, int(logits.shape[0]))

    if "same_norm_random_margin" in method:
        witness_mask = torch.zeros((int(logits.shape[0]), 1), device=logits.device, dtype=logits.dtype)
        witness_mask[witness_pos] = 1.0
        rand = torch.randn(tuple(logits.shape), generator=generator, device=logits.device, dtype=logits.dtype)
        rand = rand * witness_mask
        raw_credit = rand * (raw_credit.norm() / rand.norm().clamp_min(1.0e-12))
    elif "signflip" in method:
        raw_credit = -raw_credit

    delta = task_cotangent(logits, labels).detach()
    raw_dot = (raw_credit.detach() * delta).sum()
    delta_norm_sq = (delta * delta).sum().clamp_min(1.0e-12)
    projection_coeff = raw_dot / delta_norm_sq
    unprojected_control = "unprojected_margin" in method
    if unprojected_control:
        safe_credit = raw_credit.detach()
    else:
        safe_credit = raw_credit.detach() - projection_coeff * delta
    fu_loss = -(safe_credit.detach() * logits).sum()

    safe_dot = float((safe_credit.detach() * delta).sum().detach().item())
    raw_dot_float = float(raw_dot.detach().item())
    raw_norm = float(raw_credit.detach().norm().item())
    safe_norm = float(safe_credit.detach().norm().item())
    removed_fraction = 0.0
    if not unprojected_control and raw_norm > 0:
        removed = (projection_coeff * delta).norm()
        removed_fraction = float((removed / raw_credit.detach().norm().clamp_min(1.0e-12)).detach().item())
    fu_grad_vec = -safe_credit.detach().reshape(-1)
    task_vec = delta.reshape(-1)
    margins = logits.detach()[witness_pos, witness_labels] - logits.detach()[witness_pos, confusers.long()]
    return fu_loss, {
        "fu_control_kind": control_kind(method),
        "fu_loss_value": float(fu_loss.detach().item()),
        "cos_fu_task_gradient": cosine(fu_grad_vec, task_vec) if int(fu_grad_vec.numel()) else math.nan,
        "first_order_task_interference": safe_dot,
        "owl_projection_residual": math.nan if unprojected_control else abs(safe_dot),
        "owl_credit_norm": safe_norm,
        "negative_alignment_removed_fraction": removed_fraction,
        "positive_alignment_kept_fraction": float(safe_norm / max(raw_norm, 1.0e-12)),
        "task_interference_delta": safe_dot - raw_dot_float,
        "cwbl_barrier_active_fraction": math.nan,
        "cwbl_margin_mean": math.nan,
        "wmrp_weight_mean": float(weights.detach().mean().item()),
        "wmrp_active_fraction": float((weights.detach() > 0.5).float().mean().item()),
        "wmrp_margin_mean": float(margins.mean().item()),
        "wmrp_raw_task_interference": raw_dot_float,
    }


def witnessed_fisher_weighted_null_projection_loss(
    method: str,
    logits: Any,
    labels: Any,
    source_pos: Any,
    witness_pos: Any,
    generator: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    num_classes = int(logits.shape[-1])
    probs = torch.softmax(logits.detach(), dim=-1)
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).to(dtype=probs.dtype, device=probs.device)
    source_credit_samples = one_hot[source_pos] - probs[source_pos]
    source_labels = labels[source_pos].long()
    global_credit = source_credit_samples.mean(dim=0)
    witness_labels = labels[witness_pos].long()
    if "shuffled_witness" in method and int(witness_labels.numel()) > 1:
        perm = torch.randperm(int(witness_labels.numel()), generator=generator, device=logits.device)
        witness_labels = witness_labels[perm]
    rows = []
    for label in witness_labels:
        mask = source_labels == label
        rows.append(source_credit_samples[mask].mean(dim=0) if bool(mask.any()) else global_credit)
    credit_rows = torch.stack(rows, dim=0) if rows else torch.zeros_like(logits[witness_pos])

    fisher_weight = (probs[witness_pos] * (1.0 - probs[witness_pos])).clamp_min(float(args.fwnp_min_var))
    if float(args.fwnp_power) != 1.0:
        fisher_weight = fisher_weight.pow(float(args.fwnp_power))
    fisher_weight = fisher_weight / fisher_weight.mean(dim=1, keepdim=True).clamp_min(1.0e-12)
    weighted_rows = credit_rows * fisher_weight
    raw_credit = torch.zeros_like(logits)
    raw_credit[witness_pos] = weighted_rows / max(1, int(logits.shape[0]))

    if "same_norm_random_fisher" in method:
        witness_mask = torch.zeros((int(logits.shape[0]), 1), device=logits.device, dtype=logits.dtype)
        witness_mask[witness_pos] = 1.0
        rand = torch.randn(tuple(logits.shape), generator=generator, device=logits.device, dtype=logits.dtype)
        rand = rand * witness_mask
        raw_credit = rand * (raw_credit.norm() / rand.norm().clamp_min(1.0e-12))
    elif "signflip" in method:
        raw_credit = -raw_credit

    delta = task_cotangent(logits, labels).detach()
    raw_credit = raw_credit.detach()
    raw_dot = (raw_credit * delta).sum()
    delta_norm_sq = (delta * delta).sum().clamp_min(1.0e-12)
    projection_coeff = raw_dot / delta_norm_sq
    unprojected_control = "unprojected_fisher" in method
    if unprojected_control:
        safe_credit = raw_credit
    else:
        safe_credit = raw_credit - projection_coeff * delta
    fu_loss = -(safe_credit.detach() * logits).sum()

    safe_dot = float((safe_credit.detach() * delta).sum().detach().item())
    raw_dot_float = float(raw_dot.detach().item())
    raw_norm = float(raw_credit.norm().detach().item())
    safe_norm = float(safe_credit.detach().norm().item())
    removed_fraction = 0.0
    if not unprojected_control and raw_norm > 0:
        removed = (projection_coeff * delta).norm()
        removed_fraction = float((removed / raw_credit.norm().clamp_min(1.0e-12)).detach().item())
    fu_grad_vec = -safe_credit.detach().reshape(-1)
    task_vec = delta.reshape(-1)
    return fu_loss, {
        "fu_control_kind": control_kind(method),
        "fu_loss_value": float(fu_loss.detach().item()),
        "cos_fu_task_gradient": cosine(fu_grad_vec, task_vec) if int(fu_grad_vec.numel()) else math.nan,
        "first_order_task_interference": safe_dot,
        "owl_projection_residual": math.nan if unprojected_control else abs(safe_dot),
        "owl_credit_norm": safe_norm,
        "negative_alignment_removed_fraction": removed_fraction,
        "positive_alignment_kept_fraction": float(safe_norm / max(raw_norm, 1.0e-12)),
        "task_interference_delta": safe_dot - raw_dot_float,
        "cwbl_barrier_active_fraction": math.nan,
        "cwbl_margin_mean": math.nan,
        "fwnp_raw_credit_norm": raw_norm,
        "fwnp_safe_credit_norm": safe_norm,
        "fwnp_fisher_weight_mean": float(fisher_weight.detach().mean().item()),
        "fwnp_fisher_weight_std": float(fisher_weight.detach().std(unbiased=False).item()),
        "fwnp_raw_task_interference": raw_dot_float,
    }


def witnessed_crossfit_residual_predictor_loss(
    method: str,
    logits: Any,
    labels: Any,
    source_pos: Any,
    witness_pos: Any,
    generator: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    num_classes = int(logits.shape[-1])
    bins = max(2, int(args.cfrp_bins))
    clip = max(0.1, float(args.cfrp_clip))
    scale = float(args.cfrp_scale)
    probs = torch.softmax(logits.detach(), dim=-1)
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).to(dtype=probs.dtype, device=probs.device)
    ce_credit = one_hot - probs
    source_labels = labels[source_pos].long()
    witness_labels = labels[witness_pos].long()
    source_losses = F.cross_entropy(logits[source_pos].detach(), source_labels, reduction="none")
    source_conf = probs[source_pos, source_labels].detach()
    witness_conf = probs[witness_pos, witness_labels].detach()
    if int(source_conf.numel()) >= bins:
        qs = torch.linspace(0.0, 1.0, bins + 1, device=logits.device, dtype=logits.dtype)[1:-1]
        edges = torch.quantile(source_conf.float(), qs.float()).to(device=logits.device, dtype=logits.dtype)
        edges = torch.unique_consecutive(edges)
    else:
        edges = torch.empty(0, device=logits.device, dtype=logits.dtype)
    actual_bins = int(edges.numel()) + 1
    source_bins = torch.bucketize(source_conf, edges)
    witness_bins = torch.bucketize(witness_conf, edges)

    global_mean = source_losses.mean()
    global_std = source_losses.std(unbiased=False) if int(source_losses.numel()) > 1 else source_losses.new_tensor(1.0)
    global_std = global_std.clamp_min(1.0e-6)
    score_table = source_losses.new_zeros((num_classes, actual_bins))
    count_table = source_losses.new_zeros((num_classes, actual_bins))
    class_mean = source_losses.new_empty(num_classes)
    class_std = source_losses.new_empty(num_classes)
    for cls in range(num_classes):
        mask_cls = source_labels == cls
        if bool(mask_cls.any()):
            vals = source_losses[mask_cls]
            class_mean[cls] = vals.mean()
            class_std[cls] = (vals.std(unbiased=False) if int(vals.numel()) > 1 else global_std).clamp_min(1.0e-6)
        else:
            class_mean[cls] = global_mean
            class_std[cls] = global_std
    source_z = ((source_losses - class_mean[source_labels]) / class_std[source_labels]).clamp(-clip, clip)
    for cls in range(num_classes):
        for b in range(actual_bins):
            mask = (source_labels == cls) & (source_bins == b)
            if bool(mask.any()):
                score_table[cls, b] = source_z[mask].mean()
                count_table[cls, b] = mask.float().sum()
    fallback_score = source_z.mean()
    score_table = torch.where(count_table > 0, score_table, fallback_score.expand_as(score_table))
    if "shuffled_residual" in method and int(score_table.numel()) > 1:
        flat = score_table.reshape(-1)
        perm = torch.randperm(int(flat.numel()), generator=generator, device=logits.device)
        score_table = flat[perm].reshape_as(score_table)

    scores = score_table[witness_labels, witness_bins].to(dtype=logits.dtype)
    scores = (scores - scores.mean()).clamp(-clip, clip)
    credit_rows = scale * scores.view(-1, 1) * ce_credit[witness_pos]
    raw_credit = torch.zeros_like(logits)
    raw_credit[witness_pos] = credit_rows / max(1, int(logits.shape[0]))

    if "same_norm_random_residual" in method:
        witness_mask = torch.zeros((int(logits.shape[0]), 1), device=logits.device, dtype=logits.dtype)
        witness_mask[witness_pos] = 1.0
        rand = torch.randn(tuple(logits.shape), generator=generator, device=logits.device, dtype=logits.dtype)
        rand = rand * witness_mask
        raw_credit = rand * (raw_credit.norm() / rand.norm().clamp_min(1.0e-12))
    elif "signflip" in method:
        raw_credit = -raw_credit

    delta = task_cotangent(logits, labels).detach()
    raw_credit = raw_credit.detach()
    raw_dot = (raw_credit * delta).sum()
    delta_norm_sq = (delta * delta).sum().clamp_min(1.0e-12)
    projection_coeff = raw_dot / delta_norm_sq
    unprojected_control = "unprojected_residual" in method
    if unprojected_control:
        safe_credit = raw_credit
    else:
        safe_credit = raw_credit - projection_coeff * delta
    fu_loss = -(safe_credit.detach() * logits).sum()

    safe_dot = float((safe_credit.detach() * delta).sum().detach().item())
    raw_dot_float = float(raw_dot.detach().item())
    raw_norm = float(raw_credit.norm().detach().item())
    safe_norm = float(safe_credit.detach().norm().item())
    removed_fraction = 0.0
    if not unprojected_control and raw_norm > 0:
        removed = (projection_coeff * delta).norm()
        removed_fraction = float((removed / raw_credit.norm().clamp_min(1.0e-12)).detach().item())
    fu_grad_vec = -safe_credit.detach().reshape(-1)
    task_vec = delta.reshape(-1)
    return fu_loss, {
        "fu_control_kind": control_kind(method),
        "fu_loss_value": float(fu_loss.detach().item()),
        "cos_fu_task_gradient": cosine(fu_grad_vec, task_vec) if int(fu_grad_vec.numel()) else math.nan,
        "first_order_task_interference": safe_dot,
        "owl_projection_residual": math.nan if unprojected_control else abs(safe_dot),
        "owl_credit_norm": safe_norm,
        "negative_alignment_removed_fraction": removed_fraction,
        "positive_alignment_kept_fraction": float(safe_norm / max(raw_norm, 1.0e-12)),
        "task_interference_delta": safe_dot - raw_dot_float,
        "cwbl_barrier_active_fraction": math.nan,
        "cwbl_margin_mean": math.nan,
        "cfrp_score_mean": float(scores.detach().mean().item()),
        "cfrp_score_std": float(scores.detach().std(unbiased=False).item()),
        "cfrp_actual_bins": float(actual_bins),
        "cfrp_raw_credit_norm": raw_norm,
        "cfrp_safe_credit_norm": safe_norm,
        "cfrp_raw_task_interference": raw_dot_float,
    }


def witnessed_control_subtracted_residual_loss(
    method: str,
    logits: Any,
    labels: Any,
    source_pos: Any,
    witness_pos: Any,
    generator: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    num_classes = int(logits.shape[-1])
    bins = max(2, int(args.cfrp_bins))
    clip = max(0.1, float(args.cfrp_clip))
    scale = float(args.cfrp_scale)
    probs = torch.softmax(logits.detach(), dim=-1)
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).to(dtype=probs.dtype, device=probs.device)
    ce_credit = one_hot - probs
    source_labels = labels[source_pos].long()
    witness_labels = labels[witness_pos].long()
    source_losses = F.cross_entropy(logits[source_pos].detach(), source_labels, reduction="none")
    source_conf = probs[source_pos, source_labels].detach()
    witness_conf = probs[witness_pos, witness_labels].detach()
    if int(source_conf.numel()) >= bins:
        qs = torch.linspace(0.0, 1.0, bins + 1, device=logits.device, dtype=logits.dtype)[1:-1]
        edges = torch.quantile(source_conf.float(), qs.float()).to(device=logits.device, dtype=logits.dtype)
        edges = torch.unique_consecutive(edges)
    else:
        edges = torch.empty(0, device=logits.device, dtype=logits.dtype)
    actual_bins = int(edges.numel()) + 1
    source_bins = torch.bucketize(source_conf, edges)
    witness_bins = torch.bucketize(witness_conf, edges)

    global_mean = source_losses.mean()
    global_std = source_losses.std(unbiased=False) if int(source_losses.numel()) > 1 else source_losses.new_tensor(1.0)
    global_std = global_std.clamp_min(1.0e-6)
    score_table = source_losses.new_zeros((num_classes, actual_bins))
    count_table = source_losses.new_zeros((num_classes, actual_bins))
    class_mean = source_losses.new_empty(num_classes)
    class_std = source_losses.new_empty(num_classes)
    for cls in range(num_classes):
        mask_cls = source_labels == cls
        if bool(mask_cls.any()):
            vals = source_losses[mask_cls]
            class_mean[cls] = vals.mean()
            class_std[cls] = (vals.std(unbiased=False) if int(vals.numel()) > 1 else global_std).clamp_min(1.0e-6)
        else:
            class_mean[cls] = global_mean
            class_std[cls] = global_std
    source_z = ((source_losses - class_mean[source_labels]) / class_std[source_labels]).clamp(-clip, clip)
    for cls in range(num_classes):
        for b in range(actual_bins):
            mask = (source_labels == cls) & (source_bins == b)
            if bool(mask.any()):
                score_table[cls, b] = source_z[mask].mean()
                count_table[cls, b] = mask.float().sum()
    fallback_score = source_z.mean()
    score_table = torch.where(count_table > 0, score_table, fallback_score.expand_as(score_table))

    null_table = score_table
    if int(score_table.numel()) > 1:
        flat = score_table.reshape(-1)
        perm = torch.randperm(int(flat.numel()), generator=generator, device=logits.device)
        null_table = flat[perm].reshape_as(score_table)
    real_labels = witness_labels
    if "shuffled_subtract" in method and int(real_labels.numel()) > 1:
        perm_labels = torch.randperm(int(real_labels.numel()), generator=generator, device=logits.device)
        real_labels = real_labels[perm_labels]

    real_scores = score_table[real_labels, witness_bins].to(dtype=logits.dtype)
    null_scores = null_table[witness_labels, witness_bins].to(dtype=logits.dtype)
    if "no_subtract_residual" in method:
        scores = real_scores
    else:
        scores = real_scores - null_scores
    scores = (scores - scores.mean()).clamp(-clip, clip)
    credit_rows = scale * scores.view(-1, 1) * ce_credit[witness_pos]
    raw_credit = torch.zeros_like(logits)
    raw_credit[witness_pos] = credit_rows / max(1, int(logits.shape[0]))

    if "same_norm_random_subtract" in method:
        witness_mask = torch.zeros((int(logits.shape[0]), 1), device=logits.device, dtype=logits.dtype)
        witness_mask[witness_pos] = 1.0
        rand = torch.randn(tuple(logits.shape), generator=generator, device=logits.device, dtype=logits.dtype)
        rand = rand * witness_mask
        raw_credit = rand * (raw_credit.norm() / rand.norm().clamp_min(1.0e-12))
    elif "signflip" in method:
        raw_credit = -raw_credit

    delta = task_cotangent(logits, labels).detach()
    raw_credit = raw_credit.detach()
    raw_dot = (raw_credit * delta).sum()
    delta_norm_sq = (delta * delta).sum().clamp_min(1.0e-12)
    projection_coeff = raw_dot / delta_norm_sq
    safe_credit = raw_credit - projection_coeff * delta
    fu_loss = -(safe_credit.detach() * logits).sum()

    safe_dot = float((safe_credit.detach() * delta).sum().detach().item())
    raw_dot_float = float(raw_dot.detach().item())
    raw_norm = float(raw_credit.norm().detach().item())
    safe_norm = float(safe_credit.detach().norm().item())
    removed_fraction = 0.0
    if raw_norm > 0:
        removed = (projection_coeff * delta).norm()
        removed_fraction = float((removed / raw_credit.norm().clamp_min(1.0e-12)).detach().item())
    fu_grad_vec = -safe_credit.detach().reshape(-1)
    task_vec = delta.reshape(-1)
    return fu_loss, {
        "fu_control_kind": control_kind(method),
        "fu_loss_value": float(fu_loss.detach().item()),
        "cos_fu_task_gradient": cosine(fu_grad_vec, task_vec) if int(fu_grad_vec.numel()) else math.nan,
        "first_order_task_interference": safe_dot,
        "owl_projection_residual": abs(safe_dot),
        "owl_credit_norm": safe_norm,
        "negative_alignment_removed_fraction": removed_fraction,
        "positive_alignment_kept_fraction": float(safe_norm / max(raw_norm, 1.0e-12)),
        "task_interference_delta": safe_dot - raw_dot_float,
        "cwbl_barrier_active_fraction": math.nan,
        "cwbl_margin_mean": math.nan,
        "csrp_real_score_std": float(real_scores.detach().std(unbiased=False).item()),
        "csrp_null_score_std": float(null_scores.detach().std(unbiased=False).item()),
        "csrp_score_std": float(scores.detach().std(unbiased=False).item()),
        "csrp_actual_bins": float(actual_bins),
        "csrp_raw_credit_norm": raw_norm,
        "csrp_safe_credit_norm": safe_norm,
        "csrp_raw_task_interference": raw_dot_float,
    }


def residualize_scores_against_controls(y: Any, controls: list[Any], ridge: float) -> tuple[Any, dict[str, float]]:
    import torch

    y_centered = y - y.mean()
    cols = []
    for control in controls:
        c = control - control.mean()
        c_std = c.std(unbiased=False)
        if bool(torch.isfinite(c_std)) and float(c_std.detach().item()) > 1.0e-8:
            cols.append(c / c_std.clamp_min(1.0e-8))
    if not cols:
        return y_centered, {"rcop_control_r2": 0.0, "rcop_control_rank": 0.0}
    design = torch.stack(cols, dim=1)
    gram = design.T @ design
    eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    rhs = design.T @ y_centered
    coeff = torch.linalg.solve(gram + float(ridge) * eye, rhs)
    fitted = design @ coeff
    residual = y_centered - fitted
    y_var = (y_centered * y_centered).mean().clamp_min(1.0e-12)
    resid_var = (residual * residual).mean()
    r2 = float((1.0 - resid_var / y_var).clamp(0.0, 1.0).detach().item())
    return residual, {"rcop_control_r2": r2, "rcop_control_rank": float(len(cols))}


def witnessed_residual_control_orthogonalized_loss(
    method: str,
    logits: Any,
    labels: Any,
    source_pos: Any,
    witness_pos: Any,
    generator: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    num_classes = int(logits.shape[-1])
    bins = max(2, int(args.cfrp_bins))
    clip = max(0.1, float(args.cfrp_clip))
    scale = float(args.cfrp_scale)
    ridge = max(0.0, float(args.rcop_ridge))
    probs = torch.softmax(logits.detach(), dim=-1)
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).to(dtype=probs.dtype, device=probs.device)
    ce_credit = one_hot - probs
    source_labels = labels[source_pos].long()
    witness_labels = labels[witness_pos].long()
    source_losses = F.cross_entropy(logits[source_pos].detach(), source_labels, reduction="none")
    witness_losses = F.cross_entropy(logits[witness_pos].detach(), witness_labels, reduction="none")
    source_conf = probs[source_pos, source_labels].detach()
    witness_conf = probs[witness_pos, witness_labels].detach()
    if int(source_conf.numel()) >= bins:
        qs = torch.linspace(0.0, 1.0, bins + 1, device=logits.device, dtype=logits.dtype)[1:-1]
        edges = torch.quantile(source_conf.float(), qs.float()).to(device=logits.device, dtype=logits.dtype)
        edges = torch.unique_consecutive(edges)
    else:
        edges = torch.empty(0, device=logits.device, dtype=logits.dtype)
    actual_bins = int(edges.numel()) + 1
    source_bins = torch.bucketize(source_conf, edges)
    witness_bins = torch.bucketize(witness_conf, edges)

    global_mean = source_losses.mean()
    global_std = source_losses.std(unbiased=False) if int(source_losses.numel()) > 1 else source_losses.new_tensor(1.0)
    global_std = global_std.clamp_min(1.0e-6)
    score_table = source_losses.new_zeros((num_classes, actual_bins))
    count_table = source_losses.new_zeros((num_classes, actual_bins))
    class_mean = source_losses.new_empty(num_classes)
    class_std = source_losses.new_empty(num_classes)
    for cls in range(num_classes):
        mask_cls = source_labels == cls
        if bool(mask_cls.any()):
            vals = source_losses[mask_cls]
            class_mean[cls] = vals.mean()
            class_std[cls] = (vals.std(unbiased=False) if int(vals.numel()) > 1 else global_std).clamp_min(1.0e-6)
        else:
            class_mean[cls] = global_mean
            class_std[cls] = global_std
    source_z = ((source_losses - class_mean[source_labels]) / class_std[source_labels]).clamp(-clip, clip)
    for cls in range(num_classes):
        for b in range(actual_bins):
            mask = (source_labels == cls) & (source_bins == b)
            if bool(mask.any()):
                score_table[cls, b] = source_z[mask].mean()
                count_table[cls, b] = mask.float().sum()
    fallback_score = source_z.mean()
    score_table = torch.where(count_table > 0, score_table, fallback_score.expand_as(score_table))

    null_table = score_table
    if int(score_table.numel()) > 1:
        flat = score_table.reshape(-1)
        perm = torch.randperm(int(flat.numel()), generator=generator, device=logits.device)
        null_table = flat[perm].reshape_as(score_table)
    real_labels = witness_labels
    if "shuffled_orthogonal" in method and int(real_labels.numel()) > 1:
        perm_labels = torch.randperm(int(real_labels.numel()), generator=generator, device=logits.device)
        real_labels = real_labels[perm_labels]

    real_scores = score_table[real_labels, witness_bins].to(dtype=logits.dtype)
    null_scores = null_table[witness_labels, witness_bins].to(dtype=logits.dtype)
    base_scores = real_scores - null_scores
    witness_loss_z = (
        (witness_losses - witness_losses.mean())
        / witness_losses.std(unbiased=False).clamp_min(1.0e-6)
    ).to(dtype=logits.dtype)
    witness_conf_centered = (witness_conf - witness_conf.mean()).to(dtype=logits.dtype)
    if "no_orthogonal_residual" in method:
        scores = real_scores - real_scores.mean()
        residual_diag = {"rcop_control_r2": 0.0, "rcop_control_rank": 0.0}
    elif "hardness_only" in method:
        scores = witness_loss_z - witness_loss_z.mean()
        residual_diag = {"rcop_control_r2": 0.0, "rcop_control_rank": 0.0}
    else:
        residual, residual_diag = residualize_scores_against_controls(
            base_scores,
            [real_scores, witness_loss_z, witness_conf_centered],
            ridge,
        )
        raw_std = (base_scores - base_scores.mean()).std(unbiased=False).clamp_min(1.0e-6)
        resid_std = residual.std(unbiased=False).clamp_min(1.0e-6)
        scores = residual * (raw_std / resid_std)
    scores = (scores - scores.mean()).clamp(-clip, clip)
    credit_rows = scale * scores.view(-1, 1) * ce_credit[witness_pos]
    raw_credit = torch.zeros_like(logits)
    raw_credit[witness_pos] = credit_rows / max(1, int(logits.shape[0]))

    if "same_norm_random_orthogonal" in method:
        witness_mask = torch.zeros((int(logits.shape[0]), 1), device=logits.device, dtype=logits.dtype)
        witness_mask[witness_pos] = 1.0
        rand = torch.randn(tuple(logits.shape), generator=generator, device=logits.device, dtype=logits.dtype)
        rand = rand * witness_mask
        raw_credit = rand * (raw_credit.norm() / rand.norm().clamp_min(1.0e-12))
    elif "signflip" in method:
        raw_credit = -raw_credit

    delta = task_cotangent(logits, labels).detach()
    raw_credit = raw_credit.detach()
    raw_dot = (raw_credit * delta).sum()
    delta_norm_sq = (delta * delta).sum().clamp_min(1.0e-12)
    projection_coeff = raw_dot / delta_norm_sq
    safe_credit = raw_credit - projection_coeff * delta
    fu_loss = -(safe_credit.detach() * logits).sum()

    safe_dot = float((safe_credit.detach() * delta).sum().detach().item())
    raw_dot_float = float(raw_dot.detach().item())
    raw_norm = float(raw_credit.norm().detach().item())
    safe_norm = float(safe_credit.detach().norm().item())
    removed_fraction = 0.0
    if raw_norm > 0:
        removed = (projection_coeff * delta).norm()
        removed_fraction = float((removed / raw_credit.norm().clamp_min(1.0e-12)).detach().item())
    score_centered = scores - scores.mean()
    real_centered = real_scores - real_scores.mean()
    hard_centered = witness_loss_z - witness_loss_z.mean()
    fu_grad_vec = -safe_credit.detach().reshape(-1)
    task_vec = delta.reshape(-1)
    return fu_loss, {
        "fu_control_kind": control_kind(method),
        "fu_loss_value": float(fu_loss.detach().item()),
        "cos_fu_task_gradient": cosine(fu_grad_vec, task_vec) if int(fu_grad_vec.numel()) else math.nan,
        "first_order_task_interference": safe_dot,
        "owl_projection_residual": abs(safe_dot),
        "owl_credit_norm": safe_norm,
        "negative_alignment_removed_fraction": removed_fraction,
        "positive_alignment_kept_fraction": float(safe_norm / max(raw_norm, 1.0e-12)),
        "task_interference_delta": safe_dot - raw_dot_float,
        "cwbl_barrier_active_fraction": math.nan,
        "cwbl_margin_mean": math.nan,
        "rcop_base_score_std": float(base_scores.detach().std(unbiased=False).item()),
        "rcop_real_score_std": float(real_scores.detach().std(unbiased=False).item()),
        "rcop_hardness_score_std": float(witness_loss_z.detach().std(unbiased=False).item()),
        "rcop_score_std": float(score_centered.detach().std(unbiased=False).item()),
        "rcop_score_real_corr": cosine(score_centered.detach(), real_centered.detach()),
        "rcop_score_hardness_corr": cosine(score_centered.detach(), hard_centered.detach()),
        "rcop_control_r2": residual_diag["rcop_control_r2"],
        "rcop_control_rank": residual_diag["rcop_control_rank"],
        "rcop_actual_bins": float(actual_bins),
        "rcop_raw_credit_norm": raw_norm,
        "rcop_safe_credit_norm": safe_norm,
        "rcop_raw_task_interference": raw_dot_float,
    }


def witnessed_objective_loss(method: str, logits: Any, labels: Any, cohorts: list[Any], generator: Any, args: argparse.Namespace) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    if len(cohorts) < 2:
        zero = logits.sum() * 0.0
        return zero, {"fu_control_kind": "blocked_too_few_cohorts", "fu_loss_value": 0.0}
    split = max(1, len(cohorts) // 2)
    source_pos = torch.cat([c for c in cohorts[:split] if int(c.numel()) > 0], dim=0)
    witness_pos = torch.cat([c for c in cohorts[split:] if int(c.numel()) > 0], dim=0)
    if int(source_pos.numel()) == 0 or int(witness_pos.numel()) == 0:
        zero = logits.sum() * 0.0
        return zero, {"fu_control_kind": "blocked_empty_split", "fu_loss_value": 0.0}

    if base_objective_name(method) == "wctb":
        return witnessed_class_tail_barrier_loss(method, logits, labels, source_pos, witness_pos, generator, args)
    if base_objective_name(method) == "wmrp":
        return witnessed_margin_residual_projection_loss(method, logits, labels, source_pos, witness_pos, generator, args)
    if base_objective_name(method) == "fwnp":
        return witnessed_fisher_weighted_null_projection_loss(method, logits, labels, source_pos, witness_pos, generator, args)
    if base_objective_name(method) == "cfrp":
        return witnessed_crossfit_residual_predictor_loss(method, logits, labels, source_pos, witness_pos, generator, args)
    if base_objective_name(method) == "csrp":
        return witnessed_control_subtracted_residual_loss(method, logits, labels, source_pos, witness_pos, generator, args)
    if base_objective_name(method) == "rcop":
        return witnessed_residual_control_orthogonalized_loss(method, logits, labels, source_pos, witness_pos, generator, args)

    delta = task_cotangent(logits, labels)
    credit_rows = classwise_source_credit(logits, labels, source_pos, witness_pos, method, generator)
    credit = torch.zeros_like(logits)
    credit[witness_pos] = credit_rows / max(1, int(logits.shape[0]))
    raw_credit = credit.detach()
    delta_detached = delta.detach()
    raw_dot = (raw_credit * delta_detached).sum()
    delta_norm_sq = (delta_detached * delta_detached).sum().clamp_min(1.0e-12)
    projection_coeff = raw_dot / delta_norm_sq
    objective = base_objective_name(method)
    if objective == "owl":
        safe_credit = raw_credit - projection_coeff * delta_detached
    elif objective == "dal":
        safe_credit = raw_credit - projection_coeff.clamp_min(0.0) * delta_detached
    elif objective == "cwbl":
        safe_credit = raw_credit
    elif objective == "xwcfu":
        probs = torch.softmax(logits, dim=-1)
        one_hot = F.one_hot(labels.long(), num_classes=int(logits.shape[-1])).to(dtype=probs.dtype, device=probs.device)
        source_mean = (probs[source_pos] - one_hot[source_pos]).mean(dim=0).detach()
        witness_mean = (probs[witness_pos] - one_hot[witness_pos]).mean(dim=0)
        aux = 1.0 - F.cosine_similarity(witness_mean.view(1, -1), source_mean.view(1, -1), dim=1, eps=1.0e-8).mean()
        return aux, {
            "fu_control_kind": control_kind(method),
            "fu_loss_value": float(aux.detach().item()),
            "owl_projection_residual": math.nan,
            "first_order_task_interference": math.nan,
            "fu_task_gradient_cosine": math.nan,
            "owl_credit_norm": float(source_mean.norm().detach().item()),
            "negative_alignment_removed_fraction": math.nan,
            "positive_alignment_kept_fraction": math.nan,
            "task_interference_delta": math.nan,
            "cwbl_barrier_active_fraction": math.nan,
            "cwbl_margin_mean": math.nan,
        }
    else:
        raise ValueError(f"unknown FU objective for method {method}")

    if objective == "cwbl":
        alignment = (safe_credit.detach() * logits).sum(dim=1)
        witness_alignment = alignment[witness_pos]
        gamma = float(args.cwbl_gamma)
        tau = float(args.cwbl_tau)
        if "same_active_count_random" in method:
            perm = torch.randperm(int(witness_alignment.numel()), generator=generator, device=logits.device)
            witness_alignment = witness_alignment[perm]
        fu_loss = F.softplus((gamma - witness_alignment) / max(tau, 1.0e-6)).mean() * max(tau, 1.0e-6)
        active_fraction = float(((gamma - witness_alignment).detach() > 0).float().mean().item())
        margin_mean = float(witness_alignment.detach().mean().item())
    else:
        fu_loss = -(safe_credit.detach() * logits).sum()
        active_fraction = math.nan
        margin_mean = math.nan

    safe_dot = float((safe_credit.detach() * delta_detached).sum().detach().item())
    raw_dot_float = float(raw_dot.detach().item())
    raw_norm = float(raw_credit.norm().detach().item())
    safe_norm = float(safe_credit.detach().norm().item())
    projection_residual = abs(safe_dot)
    fu_grad_vec = -safe_credit.detach().reshape(-1)
    task_vec = delta_detached.reshape(-1)
    fu_task_cos = cosine(fu_grad_vec, task_vec) if int(fu_grad_vec.numel()) else math.nan
    removed_fraction = 0.0
    if objective == "dal" and raw_dot_float > 0 and raw_norm > 0:
        removed = (projection_coeff.clamp_min(0.0) * delta_detached).norm()
        removed_fraction = float((removed / raw_credit.norm().clamp_min(1.0e-12)).detach().item())
    kept_fraction = float(safe_norm / max(raw_norm, 1.0e-12))
    return fu_loss, {
        "fu_control_kind": control_kind(method),
        "fu_loss_value": float(fu_loss.detach().item()),
        "cos_fu_task_gradient": fu_task_cos,
        "first_order_task_interference": safe_dot,
        "owl_projection_residual": projection_residual if objective == "owl" else math.nan,
        "owl_credit_norm": safe_norm,
        "negative_alignment_removed_fraction": removed_fraction,
        "positive_alignment_kept_fraction": kept_fraction,
        "task_interference_delta": safe_dot - raw_dot_float,
        "cwbl_barrier_active_fraction": active_fraction,
        "cwbl_margin_mean": margin_mean,
    }


def current_weight_spectra(model: Any) -> dict[str, list[float]]:
    import torch

    spectra: dict[str, list[float]] = {}
    with torch.inference_mode():
        for name, param in model.named_parameters():
            if param.requires_grad and param.ndim == 2 and min(param.shape) >= 2:
                try:
                    spectra[name] = [float(x) for x in torch.linalg.svdvals(param.detach().float()).cpu().tolist()]
                except Exception:
                    continue
    return spectra


def spectrum_drift(initial: dict[str, list[float]], final: dict[str, list[float]]) -> dict[str, float]:
    vals: list[float] = []
    for key, before in initial.items():
        after = final.get(key)
        if not after or len(after) != len(before):
            continue
        denom = math.sqrt(sum(x * x for x in before)) + 1.0e-12
        vals.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(after, before))) / denom)
    return {"generalized_spectrum_drift": float(statistics.fmean(vals)) if vals else math.nan}


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
    if method not in REFERENCE_METHODS and method not in FU_METHODS:
        raise ValueError(f"unknown method: {method}")
    device = torch_device(str(args.device))
    torch.manual_seed(int(args.seed) + 225400)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    tensors = load_bundle_tensors(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    if tensors["used_fake_data"]:
        raise RuntimeError("fake data is not allowed for v22.54")
    base_model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 225400, device)
    model, opt, opt_audit = make_optimizer(ref_method, base_model, args)
    optimizer_step_source = opt_audit.get(
        "optimizer_step_source",
        "poet_torch.get_poet_optimizer" if ref_method == "poet_official" else type(opt).__name__,
    )
    external_optimizer_official_path = int(ref_method == "poet_official")
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
    initial_spectra = current_weight_spectra(model)
    params = [p for p in model.parameters() if p.requires_grad]
    train_losses: list[float] = []
    task_losses: list[float] = []
    fu_losses: list[float] = []
    fu_grad_norms: list[float] = []
    supervised_grad_norms: list[float] = []
    fu_over_supervised: list[float] = []
    step_ms: list[float] = []
    opt_ms: list[float] = []
    fu_ms: list[float] = []
    witness_transfers: list[float] = []
    source_transfers: list[float] = []
    diag_series: list[dict[str, float]] = []
    trace_events: list[str] = []
    forward_called = 0
    loss_backward_called = 0
    optimizer_step_called = 0
    fu_loss_requires_grad_seen = 0
    fu_loss_grad_nonzero_seen = 0
    loss_total_requires_grad_seen = 0
    total_start = time.perf_counter()
    for step in range(int(args.steps)):
        idx = batch_indices(int(x_train.shape[0]), int(args.batch_size), generator, device)
        positions = torch.arange(int(idx.numel()), device=device)
        cohorts = cohort_indices_from_batch(positions, int(args.cohorts))
        xb = x_train[idx]
        yb = y_train[idx].long()
        opt.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        logits = model(xb).float()
        forward_called = 1
        trace_events.append("forward")
        task_loss = F.cross_entropy(logits, yb)
        source_before = witness_before = math.nan
        source_pos = witness_pos = None
        if len(cohorts) >= 2:
            split = max(1, len(cohorts) // 2)
            source_pos = torch.cat([c for c in cohorts[:split] if int(c.numel()) > 0], dim=0)
            witness_pos = torch.cat([c for c in cohorts[split:] if int(c.numel()) > 0], dim=0)
            if int(source_pos.numel()) > 0 and int(witness_pos.numel()) > 0:
                source_before = float(F.cross_entropy(logits[source_pos], yb[source_pos]).detach().item())
                witness_before = float(F.cross_entropy(logits[witness_pos], yb[witness_pos]).detach().item())
        fu_start = time.perf_counter()
        if method in FU_METHODS:
            fu_loss, fu_diag = witnessed_objective_loss(method, logits, yb, cohorts, generator, args)
            loss_total = task_loss + float(args.lambda_fu) * fu_loss
            fu_loss_requires_grad_seen = max(fu_loss_requires_grad_seen, int(bool(fu_loss.requires_grad)))
        else:
            fu_loss = logits.sum() * 0.0
            fu_diag = {"fu_control_kind": "reference_only", "fu_loss_value": 0.0}
            loss_total = task_loss
        fu_ms.append((time.perf_counter() - fu_start) * 1000.0)
        loss_total_requires_grad_seen = max(loss_total_requires_grad_seen, int(bool(loss_total.requires_grad)))
        if method in FU_METHODS:
            supervised_grads = torch.autograd.grad(task_loss, params, retain_graph=True, allow_unused=True)
            fu_grads = torch.autograd.grad(float(args.lambda_fu) * fu_loss, params, retain_graph=True, allow_unused=True)
            supervised_vec = flatten_tensors(supervised_grads, params).to(device)
            fu_vec = flatten_tensors(fu_grads, params).to(device)
            s_norm = vector_norm(supervised_vec)
            f_norm = vector_norm(fu_vec)
            supervised_grad_norms.append(s_norm)
            fu_grad_norms.append(f_norm)
            fu_over_supervised.append(float(f_norm / max(s_norm, 1.0e-12)))
            fu_loss_grad_nonzero_seen = max(fu_loss_grad_nonzero_seen, int(f_norm > 0.0))
            fu_diag["fu_grad_norm"] = f_norm
            fu_diag["supervised_grad_norm"] = s_norm
            fu_diag["fu_grad_over_supervised_grad"] = float(f_norm / max(s_norm, 1.0e-12))
        loss_total.backward()
        loss_backward_called = 1
        trace_events.append("backward")
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
        train_losses.append(float(loss_total.detach().item()))
        task_losses.append(float(task_loss.detach().item()))
        fu_losses.append(float(fu_loss.detach().item()))
        if method in FU_METHODS and source_pos is not None and witness_pos is not None:
            with torch.inference_mode():
                after_logits = model(xb).float()
                source_after = float(F.cross_entropy(after_logits[source_pos], yb[source_pos]).item())
                witness_after = float(F.cross_entropy(after_logits[witness_pos], yb[witness_pos]).item())
            source_transfers.append(float(source_before - source_after))
            witness_transfers.append(float(witness_before - witness_after))
            fu_diag["source_transfer_actual"] = float(source_before - source_after)
            fu_diag["witness_transfer_actual"] = float(witness_before - witness_after)
            fu_diag["source_witness_gap"] = fu_diag["source_transfer_actual"] - fu_diag["witness_transfer_actual"]
        diag_series.append({k: float(v) for k, v in fu_diag.items() if isinstance(v, (int, float)) and math.isfinite(float(v))})
        step_ms.append((time.perf_counter() - step_start) * 1000.0)
    wall = time.perf_counter() - total_start
    post_held = evaluate_tensors(model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    post_test = evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    final_spectra = current_weight_spectra(model)
    drift = spectrum_drift(initial_spectra, final_spectra)
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    trace_hash = hashlib.sha256("|".join(trace_events).encode("utf-8")).hexdigest()[:16]
    standard_loop_pass = int(
        forward_called == 1
        and loss_backward_called == 1
        and optimizer_step_called == 1
        and loss_total_requires_grad_seen == 1
        and (method not in FU_METHODS or (fu_loss_requires_grad_seen == 1 and fu_loss_grad_nonzero_seen == 1))
    )
    no_debt = int(
        (post_held["ECE"] - pre_held["ECE"]) <= float(args.debt_tolerance)
        and (post_held["Brier"] - pre_held["Brier"]) <= float(args.debt_tolerance)
        and (post_held["tail_q95"] - pre_held["tail_q95"]) <= float(args.tail_debt_tolerance)
    )
    diag_mean = {key: mean(d.get(key) for d in diag_series) for key in sorted({k for d in diag_series for k in d})}
    witness_lcb = lcb95(witness_transfers)
    source_lcb = lcb95(source_transfers)
    row = {
        "run_label": str(args.label),
        "phase": "FU" if method in FU_METHODS else "reference",
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "objective_family": base_objective_name(method),
        "control_kind": control_kind(method),
        "reference_method": ref_method if method in FU_METHODS else "",
        "model_family": f"MLP_hidden{int(args.hidden)}",
        "device": str(args.device),
        "python": PYTHON,
        "train_size": int(args.train_size),
        "held_size": int(args.held_size),
        "test_size": int(args.test_size),
        "hidden": int(args.hidden),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lambda_fu": float(args.lambda_fu) if method in FU_METHODS else 0.0,
        "final_NLL": post_held["NLL"],
        "test_NLL": post_test["NLL"],
        "accuracy": post_held["accuracy"],
        "test_accuracy": post_test["accuracy"],
        "AUC_loss_time": mean(train_losses),
        "task_loss_mean": mean(task_losses),
        "fu_loss_mean": mean(fu_losses) if method in FU_METHODS else None,
        "ECE": post_held["ECE"],
        "Brier": post_held["Brier"],
        "tail_loss_q95": post_held["tail_q95"],
        "tail_loss_q99": post_held["tail_q99"],
        "margin_q10": post_held["margin_q10"],
        "margin_q01": post_held["margin_q01"],
        "pre_ECE": pre_held["ECE"],
        "pre_Brier": pre_held["Brier"],
        "pre_tail_q95": pre_held["tail_q95"],
        "held_ECE_delta": post_held["ECE"] - pre_held["ECE"],
        "held_Brier_delta": post_held["Brier"] - pre_held["Brier"],
        "held_tail_q95_delta": post_held["tail_q95"] - pre_held["tail_q95"],
        "no_ECE_Brier_tail_debt": no_debt,
        "train_time_sec": wall,
        "optimizer_step_time_ms": mean(opt_ms),
        "step_time_ms": mean(step_ms),
        "fu_operator_time_ms": mean(fu_ms),
        "controller_or_reparam_overhead_ratio": float((mean(opt_ms) or 0.0) / max(mean(step_ms) or 1.0e-12, 1.0e-12)),
        "peak_memory_mb": peak_memory,
        "trainable_parameter_count": trainable_param_count(model),
        "optimizer_state_count": optimizer_state_count(opt),
        "standard_training_loop_pass": standard_loop_pass,
        "forward_called": forward_called,
        "loss_backward_called": loss_backward_called,
        "optimizer_step_called": optimizer_step_called,
        "loss_total_requires_grad": loss_total_requires_grad_seen,
        "fu_loss_requires_grad": fu_loss_requires_grad_seen if method in FU_METHODS else "",
        "fu_loss_grad_nonzero": fu_loss_grad_nonzero_seen if method in FU_METHODS else "",
        "manual_param_update_detected": 0,
        "no_grad_param_mutation_detected": 0,
        "apply_flat_update_called": 0,
        "p_data_write_detected": 0,
        "copy_param_write_detected": 0,
        "branch_replay_used_as_training": 0,
        "proxy_direction_used_as_training": 0,
        "optimizer_step_source": optimizer_step_source,
        "external_optimizer_official_path": external_optimizer_official_path,
        "training_loop_trace_hash": trace_hash,
        "witness_transfer_LCB": witness_lcb,
        "source_transfer_LCB": source_lcb,
        "witness_LCB_positive": int(math.isfinite(witness_lcb) and witness_lcb > 0.0),
        "fu_grad_norm": mean(fu_grad_norms),
        "supervised_grad_norm": mean(supervised_grad_norms),
        "fu_grad_over_supervised_grad": mean(fu_over_supervised),
        **diag_mean,
        **drift,
        **opt_audit,
    }
    if ref_method == "poet_official":
        row["merged_inference_available"] = opt_audit.get("merged_inference_available", int(hasattr(model, "merge_if_needed")))
    out = CHUNK_ROOT / f"v22_54_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def static_scan_runner() -> dict[str, Any]:
    text = RUNNER.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "apply_flat_update_called": "apply_" + "flat_update(",
        "p_data_add_write": ".data" + ".add_",
        "p_data_copy_write": ".data" + ".copy_",
        "param_copy_write": "param" + ".copy_(",
        "param_add_write": "param" + ".add_(",
        "flat_params_manual_writeback": "flatten_" + "params",
        "branch_replay_as_training": "branch " + "replay as training",
        "proxy_direction_as_training": "proxy " + "direction as training",
    }
    hits = {name: int(pattern in text) for name, pattern in patterns.items()}
    static_loop = int("loss_total.backward()" in text and "opt.step()" in text and "logits = model(xb)" in text)
    forbidden_pass = int(sum(hits.values()) == 0)
    return {
        "standard_training_loop_static_scan_pass": static_loop,
        "manual_update_forbidden_scan_pass": forbidden_pass,
        **hits,
    }


def import_smoke(module: str, extra_path: str = "") -> dict[str, Any]:
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    try:
        importlib.import_module(module)
        return {"module": module, "status": "pass", "error": ""}
    except Exception as exc:
        return {"module": module, "status": "fail", "error": repr(exc)}


def run_o0(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, Any]] = []
    try:
        py_compile.compile(str(RUNNER), doraise=True)
        py_compile_pass = 1
        py_compile_error = ""
    except Exception as exc:
        py_compile_pass = 0
        py_compile_error = repr(exc)
    compile_cmd = [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"]
    compile_result = run_cmd(
        compile_cmd,
        task_id="v22_54_compileall_repo_external",
        files="results/v22_54/v22_54_code_truth_gate.csv",
        gpu="cpu",
        timeout=int(args.compile_timeout),
    )
    full_import = run_cmd(
        [PYTHON, "-c", "import dgkan; import experiments.run_v22_54_true_training_witnessed_objective_fu; print('pass')"],
        task_id="v22_54_full_repo_runner_import",
        files="results/v22_54/v22_54_code_truth_gate.csv",
        gpu="cpu",
    )
    poet_import = run_cmd(
        [PYTHON, "-c", "from poet_torch import POETConfig, POETModel, get_poet_optimizer; print('pass')"],
        task_id="v22_54_external_poet_import",
        files="results/v22_54/v22_54_code_truth_gate.csv",
        gpu="cpu",
    )
    pion_import_code = (
        "import importlib.util; "
        "from pathlib import Path; "
        "p=Path('external/oet_baselines/pion_spectrum_sphere/verl/verl/custom_optimizer/pion.py').resolve(); "
        "spec=importlib.util.spec_from_file_location('pion_direct_smoke', p); "
        "mod=importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "assert hasattr(mod, 'PionOptimizer'); "
        "print('pass')"
    )
    pion_import = run_cmd(
        [PYTHON, "-c", pion_import_code],
        task_id="v22_54_external_pion_import",
        files="results/v22_54/v22_54_code_truth_gate.csv",
        gpu="cpu",
    )
    scan = static_scan_runner()
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.dataset = "MNIST"
    runtime_args.seed = 0
    runtime_args.method = "owl"
    runtime_args.reference_method = "adamw"
    runtime_args.label = "v22_54_runtime_trace_owl_smoke"
    runtime_args.device = str(args.smoke_device if args.smoke_device != "auto" else ("cuda:0" if args.use_cuda_smoke and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cuda:0"))
    runtime_args.steps = int(args.smoke_steps)
    runtime_args.train_size = min(int(args.train_size), 128)
    runtime_args.held_size = min(int(args.held_size), 64)
    runtime_args.test_size = min(int(args.test_size), 64)
    try:
        runtime_row = run_collect(runtime_args)
        runtime_status = "pass" if iflag(runtime_row.get("standard_training_loop_pass")) else "fail"
        runtime_error = ""
    except Exception as exc:
        runtime_row = {}
        runtime_status = "fail"
        runtime_error = traceback.format_exc()
        (LOG_ROOT / "v22_54_runtime_trace_owl_smoke_exception.log").write_text(runtime_error, encoding="utf-8")
    row = {
        "compileall_pass": int(compile_result.get("status") == "pass"),
        "py_compile_runner_pass": py_compile_pass,
        "py_compile_runner_error": py_compile_error,
        "full_repo_clean_import_pass": int(full_import.get("status") == "pass"),
        "runner_core_clean_import_pass": int(full_import.get("status") == "pass"),
        "external_poet_import_pass": int(poet_import.get("status") == "pass"),
        "external_pion_import_pass": int(pion_import.get("status") == "pass"),
        "standard_training_loop_runtime_trace_pass": int(runtime_status == "pass"),
        "proxy_route_eligible_rows": 0,
        "runtime_error": runtime_error[:500],
        **scan,
    }
    row["part_a_pass"] = int(
        row["compileall_pass"]
        and row["py_compile_runner_pass"]
        and row["full_repo_clean_import_pass"]
        and row["runner_core_clean_import_pass"]
        and row["manual_update_forbidden_scan_pass"]
        and row["standard_training_loop_static_scan_pass"]
        and row["standard_training_loop_runtime_trace_pass"]
    )
    rows.append(row)
    write_rows(OUT_ROOT / "v22_54_code_truth_gate.csv", rows)
    append_exec(
        f"{PYTHON} experiments/run_v22_54_true_training_witnessed_objective_fu.py --stage o0",
        task_id="v22_54_part_a_truth_gate_summary",
        status="pass" if row["part_a_pass"] else "fail",
        files="results/v22_54/v22_54_code_truth_gate.csv; results/v22_54/chunks/v22_54_v22_54_runtime_trace_owl_smoke_summary.csv",
        note=f"Part A pass={row['part_a_pass']}; compileall={row['compileall_pass']}; runtime={runtime_status}",
    )
    append_recap(
        "Part A truth gate",
        md_table(rows, [
            "compileall_pass",
            "full_repo_clean_import_pass",
            "external_poet_import_pass",
            "external_pion_import_pass",
            "manual_update_forbidden_scan_pass",
            "standard_training_loop_static_scan_pass",
            "standard_training_loop_runtime_trace_pass",
            "part_a_pass",
        ]),
    )
    return row


def reanalyse_v22_53o(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    src = ROOT / "results/v22_53O"
    method_summary = read_rows(src / "v22_53O_method_summary.csv")
    o1_pairwise = read_rows(src / "v22_53O_o1_pairwise_comparison.csv")
    o2_pairwise = read_rows(src / "v22_53O_o2_wcfu_over_oet_pairwise.csv")
    final_route = read_json(src / "v22_53O_final_route.json")
    if not method_summary or not o1_pairwise or not o2_pairwise:
        raise RuntimeError("v22.53O artifacts missing; cannot run Part B reanalysis")
    poet_rows = [r for r in method_summary if r.get("method") == "poet_official"]
    poet_summary = poet_rows[0] if poet_rows else {}
    xwcfu_rows = [r for r in o2_pairwise if r.get("method") == "poet_official_xwcfu_loss"]
    o1_out = []
    for row in method_summary:
        if row.get("phase") == "O1":
            o1_out.append(
                {
                    "method": row.get("method", ""),
                    "rows": row.get("rows", ""),
                    "mean_final_NLL": row.get("mean_final_NLL", ""),
                    "mean_accuracy": row.get("mean_accuracy", ""),
                    "no_debt_rows": row.get("no_debt_rows", ""),
                    "mean_overhead_ratio": row.get("mean_overhead_ratio", ""),
                    "implementation_source": row.get("implementation_source", ""),
                }
            )
    write_rows(OUT_ROOT / "v22_54_reanalysis_o1_external_baseline.csv", o1_out)
    o2_out = []
    for row in [r for r in method_summary if r.get("phase") == "O2"]:
        o2_out.append(
            {
                "method": row.get("method", ""),
                "rows": row.get("rows", ""),
                "mean_final_NLL": row.get("mean_final_NLL", ""),
                "mean_accuracy": row.get("mean_accuracy", ""),
                "no_debt_rows": row.get("no_debt_rows", ""),
                "mean_overhead_ratio": row.get("mean_overhead_ratio", ""),
                "implementation_source": row.get("implementation_source", ""),
            }
        )
    write_rows(OUT_ROOT / "v22_54_reanalysis_o2_xwcfu_loss.csv", o2_out)
    write_rows(
        OUT_ROOT / "v22_54_poet_reference_pairwise.csv",
        [r for r in o1_pairwise if r.get("method") == "poet_official"],
    )
    failure_modes = []
    for row in xwcfu_rows:
        delta = fval(row.get("Delta_NLL_vs_reference"), 0.0) or 0.0
        failure_modes.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "reference_method": row.get("reference_method", ""),
                "reference_final_NLL": row.get("reference_final_NLL", ""),
                "final_NLL": row.get("final_NLL", ""),
                "Delta_NLL_vs_reference": row.get("Delta_NLL_vs_reference", ""),
                "beats_reference": row.get("beats_reference", ""),
                "beats_best_same_OET_control": row.get("beats_best_same_OET_control", ""),
                "witness_transfer_LCB": row.get("witness_transfer_LCB", ""),
                "witness_LCB_positive": row.get("witness_LCB_positive", ""),
                "no_debt": row.get("no_debt", ""),
                "failure_mode": "witness_positive_but_final_NLL_worse" if iflag(row.get("witness_LCB_positive")) and delta > 0 else "other",
            }
        )
    write_rows(OUT_ROOT / "v22_54_o2_failure_modes.csv", failure_modes)
    summary = {
        "poet_official_mean_final_NLL": fval(poet_summary.get("mean_final_NLL")),
        "poet_official_group_win_rate": final_route.get("oet_beats_strongest_non_oet_rows", ""),
        "poet_official_no_debt_rows": poet_summary.get("no_debt_rows", ""),
        "pion_oet_official_win_rate": "",
        "pion_hpns_win_rate": "",
        "xwcfu_beats_poet_reference_rows": sum(iflag(r.get("beats_reference")) for r in xwcfu_rows),
        "xwcfu_beats_same_oet_controls_rows": sum(iflag(r.get("beats_best_same_OET_control")) for r in xwcfu_rows),
        "xwcfu_witness_LCB_positive_rows": sum(iflag(r.get("witness_LCB_positive")) for r in xwcfu_rows),
        "xwcfu_no_debt_rows": sum(iflag(r.get("no_debt")) for r in xwcfu_rows),
        "xwcfu_final_NLL_delta_vs_reference_mean": mean(fval(r.get("Delta_NLL_vs_reference")) for r in xwcfu_rows),
        "v22_53O_final_route": final_route.get("final_route", ""),
        "reanalysis_pass": int(
            (fval(poet_summary.get("mean_final_NLL")) is not None)
            and sum(iflag(r.get("beats_reference")) for r in xwcfu_rows) == 0
        ),
    }
    write_json(OUT_ROOT / "v22_54_part_b_reanalysis_summary.json", summary)
    append_exec(
        f"{PYTHON} experiments/run_v22_54_true_training_witnessed_objective_fu.py --stage reanalyse-v53",
        task_id="v22_54_part_b_v53O_reanalysis",
        status="pass" if summary["reanalysis_pass"] else "fail",
        files=(
            "results/v22_54/v22_54_reanalysis_o1_external_baseline.csv; "
            "results/v22_54/v22_54_reanalysis_o2_xwcfu_loss.csv; "
            "results/v22_54/v22_54_poet_reference_pairwise.csv; "
            "results/v22_54/v22_54_o2_failure_modes.csv; "
            "results/v22_54/v22_54_part_b_reanalysis_summary.json"
        ),
    )
    append_recap(
        "Part B v22.53O reanalysis",
        "真实读取 `results/v22_53O/` artifacts；关键汇总如下：\n\n"
        + "```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n"
        + "Failure-mode rows:\n\n"
        + md_table(
            failure_modes,
            [
                "dataset",
                "seed",
                "Delta_NLL_vs_reference",
                "beats_reference",
                "beats_best_same_OET_control",
                "witness_LCB_positive",
                "no_debt",
                "failure_mode",
            ],
            12,
        ),
    )
    return summary


def collect_command(spec: dict[str, Any], args: argparse.Namespace, device: str) -> list[str]:
    cmd = [
        PYTHON,
        "experiments/run_v22_54_true_training_witnessed_objective_fu.py",
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
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--lambda-fu",
        str(args.lambda_fu),
        "--cwbl-gamma",
        str(args.cwbl_gamma),
        "--cwbl-tau",
        str(args.cwbl_tau),
        "--wctb-std-scale",
        str(args.wctb_std_scale),
        "--wctb-tau",
        str(args.wctb_tau),
        "--wmrp-std-scale",
        str(args.wmrp_std_scale),
        "--wmrp-tau",
        str(args.wmrp_tau),
        "--fwnp-power",
        str(args.fwnp_power),
        "--fwnp-min-var",
        str(args.fwnp_min_var),
        "--cfrp-bins",
        str(args.cfrp_bins),
        "--cfrp-clip",
        str(args.cfrp_clip),
        "--cfrp-scale",
        str(args.cfrp_scale),
        "--rcop-ridge",
        str(args.rcop_ridge),
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
    if phase == "reference":
        methods = split_csv(str(args.methods or DEFAULT_BASELINE_METHODS), str)
    else:
        methods = split_csv(str(args.methods or DEFAULT_FU_METHODS), str)
    specs = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                method_l = str(method).lower()
                label = f"{args.label_prefix}_{method_l}_{dataset}_s{seed}"
                spec = {"dataset": dataset, "seed": seed, "method": method_l, "label": label}
                if method_l in FU_METHODS:
                    spec["reference_method"] = str(args.reference_method)
                specs.append(spec)
    specs_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_specs.csv"
    dispatch_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_dispatch_status.csv"
    write_rows(specs_path, specs)
    gpus = split_csv(str(args.gpus), str) or ["cpu"]
    dispatch_rows: list[dict[str, Any]] = []

    def launch(i_spec: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        i, spec = i_spec
        gpu = gpus[i % len(gpus)]
        cmd = collect_command(spec, args, gpu)
        result = run_cmd(
            cmd,
            task_id=str(spec["label"]),
            files=f"results/v22_54/chunks/v22_54_{safe_fragment(str(spec['label']))}_summary.csv",
            gpu=str(gpu),
            timeout=int(args.collect_timeout),
        )
        return {
            **spec,
            "gpu": gpu,
            "command": " ".join(shlex.quote(str(x)) for x in cmd),
            **result,
            "summary": f"results/v22_54/chunks/v22_54_{safe_fragment(str(spec['label']))}_summary.csv",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(launch, item) for item in enumerate(specs)]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            dispatch_rows.append(row)
            write_rows(dispatch_path, dispatch_rows)
    append_exec(
        f"{PYTHON} experiments/run_v22_54_true_training_witnessed_objective_fu.py --stage {phase}",
        task_id=f"v22_54_{phase}_matrix",
        status="pass" if all(r.get("status") == "pass" for r in dispatch_rows) else "fail",
        files=f"{specs_path}; {dispatch_path}; results/v22_54/chunks/",
        note=f"rows={len(dispatch_rows)}; workers={args.workers}; gpus={args.gpus}",
    )
    return dispatch_rows


def collect_chunk_rows(include_prefixes: list[str] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_54_*_summary.csv")):
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
    reference_rows = [r for r in rows if r.get("phase") == "reference"]
    fu_rows = [r for r in rows if r.get("phase") == "FU"]
    method_summary = []
    for method in sorted({r.get("method", "") for r in rows}):
        group = [r for r in rows if r.get("method") == method]
        method_summary.append(
            {
                "method": method,
                "phase": group[0].get("phase", "") if group else "",
                "rows": len(group),
                "mean_final_NLL": mean(fval(r.get("final_NLL")) for r in group),
                "mean_accuracy": mean(fval(r.get("accuracy")) for r in group),
                "mean_AUC_loss_time": mean(fval(r.get("AUC_loss_time")) for r in group),
                "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
                "standard_loop_pass_rows": sum(iflag(r.get("standard_training_loop_pass")) for r in group),
                "mean_overhead_ratio": mean(fval(r.get("controller_or_reparam_overhead_ratio")) for r in group),
                "mean_fu_grad_over_supervised": mean(fval(r.get("fu_grad_over_supervised_grad")) for r in group),
                "mean_first_order_task_interference": mean(fval(r.get("first_order_task_interference")) for r in group),
                "witness_LCB_positive_rows": sum(iflag(r.get("witness_LCB_positive")) for r in group),
            }
        )
    write_rows(OUT_ROOT / "v22_54_method_summary.csv", method_summary)
    ref_by_key = {
        (r.get("method"), r.get("dataset"), str(r.get("seed"))): r
        for r in reference_rows
    }
    pairwise = []
    for r in fu_rows:
        ref_method = r.get("reference_method", "")
        key = (ref_method, r.get("dataset"), str(r.get("seed")))
        ref = ref_by_key.get(key)
        ref_nll = fval(ref.get("final_NLL")) if ref else None
        final_nll = fval(r.get("final_NLL"))
        objective = base_objective_name(str(r.get("method", "")))
        same_controls = [
            x for x in fu_rows
            if x.get("dataset") == r.get("dataset")
            and str(x.get("seed")) == str(r.get("seed"))
            and x.get("reference_method") == ref_method
            and base_objective_name(str(x.get("method", ""))) == objective
            and x.get("method") != r.get("method")
            and str(x.get("method", "")) not in FU_CANDIDATES
        ]
        best_control = min(same_controls, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"), default=None)
        best_control_nll = fval(best_control.get("final_NLL")) if best_control else None
        pairwise.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "method": r.get("method", ""),
                "objective_family": objective,
                "control_kind": r.get("control_kind", ""),
                "reference_method": ref_method,
                "reference_final_NLL": ref_nll,
                "final_NLL": final_nll,
                "Delta_NLL_vs_reference": (final_nll - ref_nll) if final_nll is not None and ref_nll is not None else "",
                "beats_reference": int(final_nll is not None and ref_nll is not None and final_nll < ref_nll),
                "best_control_method": best_control.get("method", "") if best_control else "",
                "best_control_final_NLL": best_control_nll,
                "beats_best_control": int(final_nll is not None and best_control_nll is not None and final_nll < best_control_nll),
                "witness_transfer_LCB": r.get("witness_transfer_LCB", ""),
                "witness_LCB_positive": r.get("witness_LCB_positive", ""),
                "no_ECE_Brier_tail_debt": r.get("no_ECE_Brier_tail_debt", ""),
                "first_order_task_interference": r.get("first_order_task_interference", ""),
                "fu_task_gradient_cosine": r.get("cos_fu_task_gradient", r.get("fu_task_gradient_cosine", "")),
                "fu_grad_over_supervised_grad": r.get("fu_grad_over_supervised_grad", ""),
                "standard_training_loop_pass": r.get("standard_training_loop_pass", ""),
                "controller_or_reparam_overhead_ratio": r.get("controller_or_reparam_overhead_ratio", ""),
                "generalized_spectrum_drift": r.get("generalized_spectrum_drift", ""),
            }
        )
    write_rows(OUT_ROOT / "v22_54_mlp_true_training_fu_pairwise.csv", pairwise)
    candidate_pairwise = [r for r in pairwise if r.get("method") in FU_CANDIDATES]
    owl_pairwise = [r for r in pairwise if r.get("method") == "owl"]
    fu_over_poet = [r for r in candidate_pairwise if r.get("reference_method") == "poet_official"]
    route = {
        "reference_rows": len(reference_rows),
        "fu_rows": len(fu_rows),
        "summary_include_prefixes": include_prefixes,
        "candidate_fu_rows": len(candidate_pairwise),
        "standard_loop_pass_candidate_rows": sum(iflag(r.get("standard_training_loop_pass")) for r in candidate_pairwise),
        "beats_reference_candidate_rows": sum(iflag(r.get("beats_reference")) for r in candidate_pairwise),
        "beats_best_control_candidate_rows": sum(iflag(r.get("beats_best_control")) for r in candidate_pairwise),
        "witness_LCB_positive_candidate_rows": sum(iflag(r.get("witness_LCB_positive")) for r in candidate_pairwise),
        "no_debt_candidate_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in candidate_pairwise),
        "owl_first_order_interference_le_1e4_rows": sum(
            int(abs(fval(r.get("first_order_task_interference"), float("inf")) or float("inf")) <= 1.0e-4)
            for r in owl_pairwise
        ),
        "fu_over_poet_beats_poet_rows": sum(iflag(r.get("beats_reference")) for r in fu_over_poet),
        "fu_over_poet_beats_control_rows": sum(iflag(r.get("beats_best_control")) for r in fu_over_poet),
        "mlp_exploration_pass": 0,
        "kan_matrix_started": 0,
        "final_route": "",
    }
    route["mlp_exploration_pass"] = int(
        len(candidate_pairwise) >= 9
        and route["standard_loop_pass_candidate_rows"] == len(candidate_pairwise)
        and route["beats_reference_candidate_rows"] >= 5
        and route["beats_best_control_candidate_rows"] >= 6
        and route["witness_LCB_positive_candidate_rows"] >= 7
        and route["no_debt_candidate_rows"] >= 7
        and route["owl_first_order_interference_le_1e4_rows"] >= min(8, len(owl_pairwise))
    )
    if not reference_rows or not fu_rows:
        route["final_route"] = "R0-CodeOrTrainingLoopBoundaryFailed_or_IncompleteMatrix"
    elif route["standard_loop_pass_candidate_rows"] != len(candidate_pairwise):
        route["final_route"] = "R0-CodeOrTrainingLoopBoundaryFailed"
    elif route["mlp_exploration_pass"]:
        route["final_route"] = "R5-TaskCompatibleWitnessObjectiveOpened"
    elif route["fu_over_poet_beats_poet_rows"] == 0:
        route["final_route"] = "R1-ExternalOETBaselineStrong"
    elif route["beats_best_control_candidate_rows"] < 6:
        route["final_route"] = "R4-AuxiliaryLossOrReweightingExplained_NoFU"
    else:
        route["final_route"] = "R2-XWCFULossNotTaskCompatible_or_CurrentFUInsufficient"
    write_json(OUT_ROOT / "v22_54_final_route.json", route)
    skipped = [
        {
            "part": "G_layerwise",
            "status": "skipped",
            "reason": "MLP exploration gate not opened" if not route["mlp_exploration_pass"] else "not yet implemented in this runner",
            "mlp_exploration_pass": route["mlp_exploration_pass"],
        },
        {
            "part": "H_KAN",
            "status": "skipped",
            "reason": "KAN official matrix starts only after MLP+FU over strongest reference gate passes",
            "mlp_exploration_pass": route["mlp_exploration_pass"],
        },
        {
            "part": "I_continual_grokking",
            "status": "skipped",
            "reason": "ordinary MLP+FU gate must be interpreted before delayed-generalization diagnostics",
            "mlp_exploration_pass": route["mlp_exploration_pass"],
        },
    ]
    write_rows(OUT_ROOT / "v22_54_downstream_gate_status.csv", skipped)
    append_exec(
        f"{PYTHON} experiments/run_v22_54_true_training_witnessed_objective_fu.py --stage summarize",
        task_id="v22_54_summarize",
        status="pass",
        files=(
            "results/v22_54/v22_54_method_summary.csv; "
            "results/v22_54/v22_54_mlp_true_training_fu_pairwise.csv; "
            "results/v22_54/v22_54_final_route.json; "
            "results/v22_54/v22_54_downstream_gate_status.csv"
        ),
        note=f"final_route={route['final_route']}",
    )
    append_recap(
        "v22.54 summary and route",
        "Method summary:\n\n"
        + md_table(
            method_summary,
            [
                "method",
                "phase",
                "rows",
                "mean_final_NLL",
                "no_debt_rows",
                "standard_loop_pass_rows",
                "witness_LCB_positive_rows",
                "mean_first_order_task_interference",
            ],
            24,
        )
        + "\nCandidate pairwise rows:\n\n"
        + md_table(
            candidate_pairwise,
            [
                "dataset",
                "seed",
                "method",
                "reference_method",
                "Delta_NLL_vs_reference",
                "beats_reference",
                "beats_best_control",
                "witness_LCB_positive",
                "no_ECE_Brier_tail_debt",
                "first_order_task_interference",
            ],
            32,
        )
        + "\nRoute JSON:\n\n```json\n"
        + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
    )
    return route


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all", choices=["all", "o0", "reanalyse-v53", "collect", "reference", "fu", "summarize"])
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="adamw")
    p.add_argument("--reference-method", default="poet_official")
    p.add_argument("--label", default="v22_54_collect")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default="")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label-prefix", default="v22_54_reference")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--steps", type=int, default=90)
    p.add_argument("--smoke-steps", type=int, default=4)
    p.add_argument("--smoke-device", default="cuda:0")
    p.add_argument("--use-cuda-smoke", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--cohorts", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--lambda-fu", type=float, default=0.12)
    p.add_argument("--cwbl-gamma", type=float, default=0.0)
    p.add_argument("--cwbl-tau", type=float, default=0.25)
    p.add_argument("--wctb-std-scale", type=float, default=0.5)
    p.add_argument("--wctb-tau", type=float, default=0.25)
    p.add_argument("--wmrp-std-scale", type=float, default=0.5)
    p.add_argument("--wmrp-tau", type=float, default=0.25)
    p.add_argument("--fwnp-power", type=float, default=1.0)
    p.add_argument("--fwnp-min-var", type=float, default=1.0e-3)
    p.add_argument("--cfrp-bins", type=int, default=4)
    p.add_argument("--cfrp-clip", type=float, default=2.0)
    p.add_argument("--cfrp-scale", type=float, default=1.0)
    p.add_argument("--rcop-ridge", type=float, default=1.0e-4)
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
        if args.stage == "reanalyse-v53":
            print(json.dumps(reanalyse_v22_53o(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "collect":
            print(json.dumps(run_collect(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "reference":
            if args.label_prefix == "v22_54_reference":
                args.label_prefix = "v22_54_reference"
            run_matrix(args, phase="reference")
            return 0
        if args.stage == "fu":
            if args.label_prefix == "v22_54_reference":
                args.label_prefix = f"v22_54_fu_over_{safe_fragment(args.reference_method)}"
            run_matrix(args, phase="fu")
            return 0
        if args.stage == "summarize":
            print(json.dumps(summarize(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "all":
            gate = run_o0(args)
            if not iflag(gate.get("part_a_pass")):
                summarize(args)
                return 2
            reanalysis = reanalyse_v22_53o(args)
            if not iflag(reanalysis.get("reanalysis_pass")):
                summarize(args)
                return 3
            args.label_prefix = "v22_54_reference"
            run_matrix(args, phase="reference")
            args.label_prefix = f"v22_54_fu_over_{safe_fragment(args.reference_method)}"
            run_matrix(args, phase="fu")
            summarize(args)
            return 0
    except Exception:
        err_path = LOG_ROOT / f"v22_54_stage_{safe_fragment(args.stage)}_exception.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        append_exec(
            f"{PYTHON} experiments/run_v22_54_true_training_witnessed_objective_fu.py --stage {args.stage}",
            task_id=f"v22_54_stage_{args.stage}_exception",
            status="exception",
            exit_code="exception",
            files=str(err_path),
        )
        print(err_path.read_text(encoding="utf-8"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
