#!/usr/bin/env python3
"""DG-KAN v22.53O Pion / POET external OET baseline runner.

This runner is intentionally conservative.  It records repository provenance,
license/install/import smoke, then runs MLP-first OET baselines through normal
forward/backward/optimizer-step training loops.  Local approximations are
always labelled as local or diagnostic; official Pion/POET rows are only used
when their downloaded code path is actually exercised.
"""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import py_compile
import re
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
OUT_ROOT = ROOT / "results/v22_53O"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.53O_PionPOET_ExternalOETBaseline_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.53O_PionPOET_ExternalOETBaseline_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.53O_PionPOET_ExternalOETBaseline_补充计划.md"

OET_ROOT = ROOT / "external/oet_baselines"
REPOS = [
    {
        "repo_name": "poet_sphere",
        "repo_url": "https://github.com/Sphere-AI-Lab/poet",
        "path": OET_ROOT / "poet_sphere",
        "paper_url": "https://arxiv.org/abs/2506.08001; https://arxiv.org/abs/2603.05500",
        "claim_boundary": "official_POET_X_reparameterization_baseline_when_import_and_small_matrix_smoke_pass",
    },
    {
        "repo_name": "pion_spectrum_sphere",
        "repo_url": "https://github.com/Sphere-AI-Lab/pion",
        "path": OET_ROOT / "pion_spectrum_sphere",
        "paper_url": "not_declared_in_readme_snapshot",
        "claim_boundary": "official_Pion_OET_matrix_optimizer_if_megatron_PionOptimizer_step_smoke_passes",
    },
    {
        "repo_name": "pion_hpns_optml_optional",
        "repo_url": "https://github.com/OPTML-Group/Pion",
        "path": OET_ROOT / "pion_hpns_optml_optional",
        "paper_url": "https://arxiv.org/abs/2605.19282",
        "claim_boundary": "Pion_HPNS_optional_baseline_not_spectrum_preserving_Pion_OET",
    },
]

BASELINE_METHODS = [
    "adamw",
    "cautious_adamw",
    "schedule_free_adamw_local",
    "muon_like_local",
    "soap_shampoo_diag_diagnostic",
    "pion_oet_sphere_official",
    "pion_oet_local",
    "pion_hpns_optml_optional",
    "poet_official",
]

LEGACY_PROXY_O2_METHODS = [
    "pion_oet_local_wcfu",
    "pion_oet_local_same_source_span_random",
    "pion_oet_local_signflip",
    "pion_oet_local_shuffled_witness",
    "pion_oet_local_frozen_witness",
    "pion_oet_local_same_norm",
]
O2_LOSS_TRUE_TRAINING_METHODS = [
    "poet_official_xwcfu_loss",
    "poet_official_xwcfu_same_source_span_random_loss",
    "poet_official_xwcfu_signflip_loss",
    "poet_official_xwcfu_shuffled_witness_loss",
]
O2_MANUAL_CORRECTION_METHODS: list[str] = []
O2_METHODS = O2_LOSS_TRUE_TRAINING_METHODS + O2_MANUAL_CORRECTION_METHODS
O2_CANDIDATE_METHOD = "poet_official_xwcfu_loss"
REJECTED_DIAGNOSTIC_O2_PREFIXES = ("pion_oet_local_crpcf",)


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.53O PionPOET ExternalOETBaseline 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、输入文件、输出文件、GPU、状态、失败与修复尝试。"
            "本日志不写虚构数据；后续复现应优先看这里的命令、解释器、artifact 路径和 stdout/stderr。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.53O PionPOET ExternalOETBaseline 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘 artifact；失败、缺失、local reimplementation 与 official baseline "
            "必须分开写。结论必须带证据链，不允许把外部 OET 或 local diagnostic 冒充 FU/KAN success。\n",
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
    journal = read_rows(OUT_ROOT / "v22_53O_command_journal.csv")
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(
        OUT_ROOT / "v22_53O_command_journal.csv",
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
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=out, stderr=err, timeout=timeout)
        status = "pass" if proc.returncode == 0 else "fail"
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        status = "fail"
        returncode = "timeout"
        stdout_path.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8")
        stderr_path.write_text((exc.stderr or "") if isinstance(exc.stderr, str) else repr(exc), encoding="utf-8")
    command = " ".join(shlex.quote(part) for part in cmd)
    row = {
        "task_id": task_id,
        "command": command,
        "status": status,
        "returncode": returncode,
        "wall_seconds": time.time() - start,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
    }
    append_exec(
        command,
        task_id=task_id,
        status=status,
        gpu=gpu,
        exit_code=returncode,
        files=", ".join(x for x in [files, row["stdout"], row["stderr"]] if x),
    )
    return row


def fval(value: Any, default: float | None = None) -> float | None:
    if value in {"", None}:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def iflag(value: Any) -> int:
    if value in {1, "1", True, "true", "True", "yes", "pass"}:
        return 1
    return 0


def mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.fmean(clean) if clean else None


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 16) -> str:
    if not rows:
        return "_无行。_\n"
    shown = rows[:limit]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in shown:
        cells = []
        for col in columns:
            text = str(row.get(col, ""))
            if len(text) > 96:
                text = text[:93] + "..."
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    if len(rows) > limit:
        out.append(f"\n_只显示前 {limit} 行，共 {len(rows)} 行。_")
    return "\n".join(out) + "\n"


def torch_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def named_params(model: Any) -> list[tuple[str, Any]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def flatten_grads(grads: Iterable[Any | None], params: list[Any]) -> Any:
    import torch

    parts = []
    for grad, param in zip(grads, params):
        if grad is None:
            parts.append(torch.zeros_like(param).reshape(-1))
        else:
            parts.append(grad.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def assign_flat_grad(named: list[tuple[str, Any]], direction: Any) -> None:
    offset = 0
    for _name, p in named:
        n = int(p.numel())
        p.grad = direction[offset : offset + n].view_as(p).detach().clone()
        offset += n


def apply_flat_update(named: list[tuple[str, Any]], direction: Any, lr: float, scale: float) -> float:
    import torch

    offset = 0
    norm2 = 0.0
    with torch.no_grad():
        for _name, p in named:
            n = int(p.numel())
            chunk = direction[offset : offset + n].view_as(p).to(device=p.device, dtype=p.dtype)
            delta = -float(lr) * float(scale) * chunk
            p.add_(delta)
            norm2 += float(delta.norm().item()) ** 2
            offset += n
    return math.sqrt(norm2)


def flatten_params(named: list[tuple[str, Any]]) -> Any:
    import torch

    parts = [p.detach().float().reshape(-1).clone() for _name, p in named]
    return torch.cat(parts) if parts else torch.empty(0)


def lcb95(values: Iterable[float]) -> float:
    clean = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            clean.append(parsed)
    if not clean:
        return math.nan
    mu = statistics.fmean(clean)
    if len(clean) <= 1:
        return mu
    sd = statistics.pstdev(clean)
    return float(mu - 1.96 * sd / math.sqrt(len(clean)))


def vector_norm(vec: Any) -> float:
    if vec is None or int(vec.numel()) == 0:
        return 0.0
    val = float(vec.norm().detach().item())
    return val if math.isfinite(val) else 0.0


def cosine(a: Any, b: Any) -> float:
    denom = a.norm().clamp_min(1.0e-12) * b.norm().clamp_min(1.0e-12)
    return float((a.dot(b) / denom).detach().item())


def load_bundle_tensors(dataset: str, train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    return v49.load_bundle_tensors(dataset, train_size, held_size, test_size, seed)


def make_model(input_dim: int, num_classes: int, hidden: int, seed: int, device: Any) -> Any:
    return v49.make_model(input_dim, num_classes, hidden, seed, device)


def evaluate_tensors(model: Any, x: Any, y: Any, device: Any, num_classes: int, batch_size: int) -> dict[str, float]:
    return v49.evaluate_tensors(model, x, y, device, num_classes, batch_size)


class CautiousAdamW:
    """Small local CautiousAdamW-style baseline."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 1.0e-4,
    ) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.betas = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.state: dict[int, dict[str, Any]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        import torch

        beta1, beta2 = self.betas
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                sid = id(p)
                st = self.state.setdefault(
                    sid,
                    {"step": 0, "m": torch.zeros_like(p), "v": torch.zeros_like(p)},
                )
                st["step"] += 1
                if self.weight_decay:
                    p.mul_(1.0 - self.lr * self.weight_decay)
                st["m"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                st["v"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias1 = 1.0 - beta1 ** int(st["step"])
                bias2 = 1.0 - beta2 ** int(st["step"])
                update = (st["m"] / bias1) / ((st["v"] / bias2).sqrt() + self.eps)
                mask = (update * grad > 0.0).to(update.dtype)
                active = float(mask.float().mean().item())
                if active > 1.0e-8:
                    mask = mask / active
                p.add_(update * mask, alpha=-self.lr)


class ScheduleFreeAdamWLocal:
    """Compact local schedule-free-style AdamW averaging diagnostic."""

    def __init__(self, params: Iterable[Any], lr: float, weight_decay: float, beta: float = 0.98) -> None:
        import torch

        self.params = list(params)
        self.opt = torch.optim.AdamW(self.params, lr=float(lr), weight_decay=float(weight_decay))
        self.beta = float(beta)
        self.avg = [p.detach().clone() for p in self.params]
        self.steps = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.opt.step()
        self.steps += 1
        with __import__("torch").no_grad():
            for avg, p in zip(self.avg, self.params):
                avg.mul_(self.beta).add_(p.detach(), alpha=1.0 - self.beta)

    def swap_to_average(self) -> list[Any]:
        old = [p.detach().clone() for p in self.params]
        with __import__("torch").no_grad():
            for p, avg in zip(self.params, self.avg):
                p.copy_(avg)
        return old

    def restore(self, old: list[Any]) -> None:
        with __import__("torch").no_grad():
            for p, src in zip(self.params, old):
                p.copy_(src)


class MatrixGeometryOptimizer:
    """Mixed matrix-aware optimizer with AdamW fallback for non-2D params."""

    def __init__(self, method: str, model: Any, lr: float, weight_decay: float, device: Any) -> None:
        import torch

        self.method = method
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.matrix_params = [p for p in self.params if p.ndim == 2]
        self.fallback_params = [p for p in self.params if p.ndim != 2]
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = device
        self.state: dict[int, dict[str, Any]] = {}
        self.diags: list[dict[str, float]] = []
        self._official_pion = None
        self._hpns_fn = None
        if self.fallback_params:
            self.fallback = torch.optim.AdamW(self.fallback_params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.fallback = None
        if method == "pion_oet_sphere_official":
            sys.path.insert(0, str(ROOT / "external/oet_baselines/pion_spectrum_sphere/megatron-lm"))
            from megatron.core.optimizer.pion import PionOptimizer

            self._official_pion = PionOptimizer(
                self.matrix_params,
                lr=self.lr,
                weight_decay=0.0,
                degree=2,
                pion_scaling="rms",
                pion_rms=0.2,
                pion_momentum="lie_lie",
                pion_update_side="both",
            )
        if method == "pion_hpns_optml_optional":
            sys.path.insert(0, str(ROOT / "external/oet_baselines/pion_hpns_optml_optional/VLA/VLAAdapter"))
            mod = importlib.import_module("pion_optim.muon")
            self._hpns_fn = getattr(mod, "high_pass_ns")

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def _momentum(self, p: Any, grad: Any, beta: float = 0.9) -> Any:
        import torch

        st = self.state.setdefault(id(p), {})
        if "m" not in st:
            st["m"] = torch.zeros_like(grad)
        st["m"].mul_(beta).add_(grad, alpha=1.0 - beta)
        return st["m"]

    def _matrix_oet_step(self, p: Any) -> dict[str, float]:
        import torch

        grad = p.grad.detach().float()
        W = p.data.float()
        g = self._momentum(p, grad).float()
        rows, cols = W.shape
        before_sv = torch.linalg.svdvals(W)
        side = "out" if rows <= cols else "in"
        if side == "out":
            A = g @ W.t()
            A = A - A.t()
            induced = A @ W
        else:
            A = W.t() @ g
            A = A - A.t()
            induced = W @ A
        fro = induced.norm().clamp_min(1.0e-12)
        alpha = self.lr * 0.2 * math.sqrt(float(rows * cols)) / float(fro.item())
        A = (-alpha * A).to(device=W.device, dtype=torch.float32)
        Q = torch.linalg.matrix_exp(A)
        if side == "out":
            W_new = Q @ W
        else:
            W_new = W @ Q
        p.data.copy_(W_new.to(dtype=p.dtype))
        after_sv = torch.linalg.svdvals(p.data.float())
        drift = (after_sv - before_sv).abs()
        I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
        return {
            "singular_value_drift_mean": float(drift.mean().item()),
            "singular_value_drift_max": float(drift.max().item()),
            "orthogonality_error": float((Q.t() @ Q - I).norm().item()),
            "rotation_angle": float(A.norm().item()),
            "radial_channel_fraction": 0.0,
            "update_side_out": 1.0 if side == "out" else 0.0,
        }

    def _matrix_muon_like_step(self, p: Any, hpns: bool = False) -> dict[str, float]:
        import torch

        grad = p.grad.detach()
        g = self._momentum(p, grad).float()
        if g.ndim == 4:
            g2 = g.view(len(g), -1)
        else:
            g2 = g
        if hpns and self._hpns_fn is not None:
            update = self._hpns_fn(g2, promotion_steps=2, suppression_steps=3).float()
        else:
            U, _S, Vh = torch.linalg.svd(g2.float(), full_matrices=False)
            update = U @ Vh
        if update.shape != p.shape:
            update = update.view_as(p)
        with torch.no_grad():
            if self.weight_decay:
                p.mul_(1.0 - self.lr * self.weight_decay)
            scale = 0.2 * math.sqrt(float(max(p.shape[-2], p.shape[-1])))
            p.add_(update.to(device=p.device, dtype=p.dtype), alpha=-self.lr * scale)
        return {
            "orthogonality_error": 0.0,
            "rotation_angle": float(update.float().norm().item()),
            "radial_channel_fraction": 1.0,
        }

    def _matrix_shampoo_diag_step(self, p: Any) -> dict[str, float]:
        import torch

        grad = p.grad.detach().float()
        st = self.state.setdefault(id(p), {})
        if "row_v" not in st:
            st["row_v"] = torch.zeros(grad.shape[0], device=grad.device)
            st["col_v"] = torch.zeros(grad.shape[1], device=grad.device)
        st["row_v"].mul_(0.95).add_(grad.square().mean(dim=1), alpha=0.05)
        st["col_v"].mul_(0.95).add_(grad.square().mean(dim=0), alpha=0.05)
        update = grad / (st["row_v"].sqrt().add(1e-4)[:, None] * st["col_v"].sqrt().add(1e-4)[None, :]).sqrt()
        update = update / update.norm().clamp_min(1.0e-12) * grad.norm().clamp_min(1.0e-12)
        with torch.no_grad():
            if self.weight_decay:
                p.mul_(1.0 - self.lr * self.weight_decay)
            p.add_(update.to(dtype=p.dtype), alpha=-self.lr)
        return {"orthogonality_error": 0.0, "rotation_angle": float(update.norm().item()), "radial_channel_fraction": 1.0}

    def step(self) -> None:
        if self._official_pion is not None:
            self._official_pion.step()
            if self.fallback is not None:
                self.fallback.step()
            self.diags.append(
                {
                    "singular_value_drift_mean": math.nan,
                    "singular_value_drift_max": math.nan,
                    "orthogonality_error": math.nan,
                    "rotation_angle": math.nan,
                    "radial_channel_fraction": 0.0,
                }
            )
            return
        if self.fallback is not None:
            self.fallback.step()
        step_diags = []
        for p in self.matrix_params:
            if p.grad is None:
                continue
            if self.method == "pion_oet_local":
                step_diags.append(self._matrix_oet_step(p))
            elif self.method == "pion_hpns_optml_optional":
                step_diags.append(self._matrix_muon_like_step(p, hpns=True))
            elif self.method == "muon_like_local":
                step_diags.append(self._matrix_muon_like_step(p, hpns=False))
            elif self.method == "soap_shampoo_diag_diagnostic":
                step_diags.append(self._matrix_shampoo_diag_step(p))
        if step_diags:
            self.diags.append(
                {
                    key: statistics.fmean(float(d.get(key, 0.0)) for d in step_diags if math.isfinite(float(d.get(key, 0.0))))
                    if any(math.isfinite(float(d.get(key, 0.0))) for d in step_diags)
                    else math.nan
                    for key in {
                        "singular_value_drift_mean",
                        "singular_value_drift_max",
                        "orthogonality_error",
                        "rotation_angle",
                        "radial_channel_fraction",
                        "update_side_out",
                    }
                }
            )

    def state_numel(self) -> int:
        total = 0
        for st in self.state.values():
            for value in st.values():
                if hasattr(value, "numel"):
                    total += int(value.numel())
        if self.fallback is not None:
            for st in self.fallback.state.values():
                for value in st.values():
                    if hasattr(value, "numel"):
                        total += int(value.numel())
        if self._official_pion is not None:
            for st in self._official_pion.state.values():
                for value in st.values():
                    if hasattr(value, "numel"):
                        total += int(value.numel())
        return total


def make_optimizer(method: str, model: Any, args: argparse.Namespace, device: Any) -> Any:
    import torch

    if method == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if method == "cautious_adamw":
        return CautiousAdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if method == "schedule_free_adamw_local":
        return ScheduleFreeAdamWLocal(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if method in {"pion_oet_sphere_official", "pion_oet_local", "pion_hpns_optml_optional", "muon_like_local", "soap_shampoo_diag_diagnostic"}:
        return MatrixGeometryOptimizer(method, model, float(args.lr), float(args.weight_decay), device)
    raise ValueError(f"unknown optimizer method: {method}")


def maybe_wrap_poet(model: Any, method: str, args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    if method not in {"poet_official", "poetx_mem_official"}:
        return model, None, {}
    from poet_torch import POETConfig, POETModel, get_poet_optimizer

    cfg = POETConfig(
        block_size=int(args.poet_block_size),
        merge_interval=int(args.poet_merge_interval),
        poet_lr=float(args.poet_lr),
        base_lr=float(args.lr),
        poet_scale=float(args.poet_scale),
        weight_decay=float(args.weight_decay),
        mem_efficient_mode=(method == "poetx_mem_official"),
    )
    wrapped = POETModel(model, cfg)
    opt = get_poet_optimizer(wrapped, cfg)
    return wrapped, opt, poet_parameter_audit(wrapped)


def poet_parameter_audit(model: Any) -> dict[str, Any]:
    audit = {
        "base_weight_parameter_count": 0,
        "orthogonal_factor_parameter_count": 0,
        "trainable_parameter_count_total": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
        "merged_inference_available": 0,
        "merged_inference_parameter_count": 0,
        "poet_replaced_linear_layers": 0,
    }
    try:
        from poet_torch.layers import POETLinear, QPOETLinear

        base_total = 0
        orth_total = 0
        merged_total = 0
        count = 0
        for module in model.modules():
            if isinstance(module, (POETLinear, QPOETLinear)):
                count += 1
                if hasattr(module, "weight"):
                    base_total += int(module.weight.numel())
                    merged_total += int(module.weight.numel())
                if hasattr(module, "oft_R"):
                    orth_total += int(module.oft_R.numel())
                if getattr(module, "bias", None) is not None:
                    merged_total += int(module.bias.numel())
        audit.update(
            {
                "base_weight_parameter_count": base_total,
                "orthogonal_factor_parameter_count": orth_total,
                "merged_inference_available": int(count > 0),
                "merged_inference_parameter_count": merged_total,
                "poet_replaced_linear_layers": count,
            }
        )
    except Exception:
        pass
    return audit


def trainable_param_count(model: Any) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def optimizer_state_count(opt: Any) -> int:
    total = 0
    if hasattr(opt, "state_numel"):
        return int(opt.state_numel())
    if hasattr(opt, "state"):
        for st in opt.state.values():
            for value in st.values():
                if hasattr(value, "numel"):
                    total += int(value.numel())
                elif isinstance(value, (int, float)):
                    total += 1
    if hasattr(opt, "opt") and hasattr(opt.opt, "state"):
        for st in opt.opt.state.values():
            for value in st.values():
                if hasattr(value, "numel"):
                    total += int(value.numel())
    return int(total)


def current_weight_spectra(model: Any) -> dict[str, list[float]]:
    import torch

    spectra: dict[str, list[float]] = {}
    for name, module in model.named_modules():
        weight = None
        if hasattr(module, "get_effective_weight"):
            try:
                weight = module.get_effective_weight()
            except Exception:
                weight = None
        elif hasattr(module, "weight") and getattr(module, "weight") is not None and getattr(module.weight, "ndim", 0) == 2:
            weight = module.weight
        if weight is None or getattr(weight, "ndim", 0) != 2:
            continue
        with torch.no_grad():
            spectra[name or "root"] = [float(x) for x in torch.linalg.svdvals(weight.detach().float()).cpu()]
    return spectra


def spectrum_drift(initial: dict[str, list[float]], final: dict[str, list[float]]) -> dict[str, float]:
    vals = []
    for key, before in initial.items():
        after = final.get(key)
        if not after:
            continue
        n = min(len(before), len(after))
        if n <= 0:
            continue
        vals.extend(abs(float(after[i]) - float(before[i])) for i in range(n))
    if not vals:
        return {"singular_value_drift_mean": math.nan, "singular_value_drift_max": math.nan}
    return {"singular_value_drift_mean": statistics.fmean(vals), "singular_value_drift_max": max(vals)}


def functional_actuator_probe(model: Any, x: Any, device: Any, *, directions: int = 8, eps: float = 1.0e-3) -> dict[str, Any]:
    import torch

    if int(directions) <= 0:
        return {
            "functional_actuator_singular_values": "",
            "functional_actuator_effective_rank": "",
            "signal_reachable_energy": "",
            "output_logit_variance": "",
        }
    model.eval()
    xb = x[: min(32, int(x.shape[0]))].to(device)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    if not params:
        return {
            "functional_actuator_singular_values": "[]",
            "functional_actuator_effective_rank": math.nan,
            "signal_reachable_energy": math.nan,
            "output_logit_variance": math.nan,
        }
    with torch.no_grad():
        base = model(xb).float()
        cols = []
        gen = torch.Generator(device=device)
        gen.manual_seed(225300 + int(sum(p.numel() for _n, p in params) % 100000))
        for _ in range(int(directions)):
            noises = []
            total_norm2 = 0.0
            for _n, p in params:
                z = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
                noises.append(z)
                total_norm2 += float(z.float().norm().item()) ** 2
            total_norm = math.sqrt(max(total_norm2, 1.0e-12))
            for (_n, p), z in zip(params, noises):
                p.add_(z, alpha=float(eps) / total_norm)
            shifted = model(xb).float()
            for (_n, p), z in zip(params, noises):
                p.add_(z, alpha=-float(eps) / total_norm)
            cols.append(((shifted - base) / float(eps)).reshape(-1).detach())
        response = torch.stack(cols, dim=1).float()
        sv = torch.linalg.svdvals(response)
        energy = sv.square()
        total = energy.sum().clamp_min(1.0e-12)
        p = energy / total
        eff_rank = float(torch.exp(-(p * (p + 1.0e-12).log()).sum()).item())
        return {
            "functional_actuator_singular_values": json.dumps([float(v) for v in sv.detach().cpu()[:16]], ensure_ascii=False),
            "functional_actuator_effective_rank": eff_rank,
            "signal_reachable_energy": float(total.item()),
            "output_logit_variance": float(base.var().item()),
        }


def batch_indices(n: int, batch_size: int, generator: Any, device: Any) -> Any:
    import torch

    return torch.randint(0, int(n), (int(batch_size),), generator=generator, device=device)


def cohort_indices_from_batch(idx: Any, cohorts: int) -> list[Any]:
    chunks = []
    n = int(idx.numel())
    for c in range(int(cohorts)):
        start = int(round(c * n / max(1, int(cohorts))))
        end = int(round((c + 1) * n / max(1, int(cohorts))))
        if end > start:
            chunks.append(idx[start:end])
    return chunks


def cohort_gradients(model: Any, x: Any, y: Any, indices: list[Any], params: list[Any]) -> tuple[list[Any], list[float]]:
    import torch
    import torch.nn.functional as F

    grads = []
    losses = []
    model.train()
    for idx in indices:
        logits = model(x[idx]).float()
        loss = F.cross_entropy(logits, y[idx].long())
        grad_tensors = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        grads.append(flatten_grads(grad_tensors, params))
        losses.append(float(loss.detach().item()))
    return grads, losses


def cohort_gradients_train_labels(
    model: Any,
    x: Any,
    y: Any,
    indices: list[Any],
    params: list[Any],
    *,
    generator: Any | None = None,
    shuffle_labels: bool = False,
) -> tuple[list[Any], list[float]]:
    import torch
    import torch.nn.functional as F

    grads = []
    losses = []
    model.train()
    for idx in indices:
        labels = y[idx].long()
        if shuffle_labels and int(labels.numel()) > 1:
            perm = torch.randperm(int(labels.numel()), generator=generator, device=labels.device)
            labels = labels[perm]
        logits = model(x[idx]).float()
        loss = F.cross_entropy(logits, labels)
        grad_tensors = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        grads.append(flatten_grads(grad_tensors, params))
        losses.append(float(loss.detach().item()))
    return grads, losses


def cohort_losses_no_grad(model: Any, x: Any, y: Any, indices: list[Any]) -> float:
    import torch
    import torch.nn.functional as F

    vals = []
    model.eval()
    with torch.no_grad():
        for idx in indices:
            logits = model(x[idx]).float()
            vals.append(float(F.cross_entropy(logits, y[idx].long()).detach().item()))
    return mean(vals) or math.nan


def xwcfu_loss_from_logits(method: str, logits: Any, labels: Any, cohorts: list[Any], generator: Any) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    if len(cohorts) < 2:
        return logits.sum() * 0.0, {
            "xwcfu_aux_loss": 0.0,
            "xwcfu_cosine": math.nan,
            "xwcfu_target_norm": 0.0,
            "xwcfu_witness_norm": 0.0,
            "xwcfu_control_kind": "blocked_too_few_cohorts",
        }
    split = max(1, len(cohorts) // 2)
    source_idx = torch.cat([c for c in cohorts[:split] if int(c.numel()) > 0], dim=0)
    witness_idx = torch.cat([c for c in cohorts[split:] if int(c.numel()) > 0], dim=0)
    num_classes = int(logits.shape[-1])

    def residual_at(pos: Any, use_labels: Any) -> Any:
        probs = torch.softmax(logits[pos], dim=-1)
        one_hot = F.one_hot(use_labels.long(), num_classes=num_classes).to(dtype=probs.dtype)
        return probs - one_hot

    source_residuals = residual_at(source_idx, labels[source_idx])
    source_mean = source_residuals.mean(dim=0)
    control_kind = "xwcfu_true"
    target = source_mean
    witness_labels = labels[witness_idx]
    if method.endswith("same_source_span_random_loss"):
        coeff = torch.randn((int(source_residuals.shape[0]),), generator=generator, device=logits.device, dtype=logits.dtype)
        target = coeff @ source_residuals
        target = target / target.norm().clamp_min(1.0e-12) * source_mean.norm().detach().clamp_min(1.0e-12)
        control_kind = "same_source_span_random"
    elif method.endswith("signflip_loss"):
        target = -source_mean
        control_kind = "signflip"
    elif method.endswith("shuffled_witness_loss"):
        if int(witness_labels.numel()) > 1:
            perm = torch.randperm(int(witness_labels.numel()), generator=generator, device=witness_labels.device)
            witness_labels = witness_labels[perm]
        control_kind = "shuffled_witness_labels"

    witness_mean = residual_at(witness_idx, witness_labels).mean(dim=0)
    target_detached = target.detach()
    cosine_value = F.cosine_similarity(
        witness_mean.view(1, -1),
        target_detached.view(1, -1),
        dim=1,
        eps=1.0e-8,
    ).mean()
    aux = 1.0 - cosine_value
    return aux, {
        "xwcfu_aux_loss": float(aux.detach().item()),
        "xwcfu_cosine": float(cosine_value.detach().item()),
        "xwcfu_target_norm": float(target_detached.norm().detach().item()),
        "xwcfu_witness_norm": float(witness_mean.detach().norm().item()),
        "xwcfu_control_kind": control_kind,
    }


def o2_reference_method(method: str) -> str:
    if str(method).startswith("poet_official_xwcfu_"):
        return "poet_official"
    if str(method).startswith("pion_oet_local_"):
        return "pion_oet_local"
    raise ValueError(f"unknown O2 reference for method: {method}")


def crossfit_crpcf_correction(
    method: str,
    source_grads: list[Any],
    witness_grads: list[Any],
    reference_direction: Any,
    generator: Any,
    frozen_state: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from scipy.optimize import minimize
    import torch

    if not source_grads or not witness_grads:
        return torch.zeros_like(reference_direction), {
            "solver_status": "blocked_empty_source_or_witness",
            "witness_gradient_gain_LCB": math.nan,
            "correction_norm": 0.0,
        }
    ref = reference_direction.detach().float().reshape(-1)
    source_stack = torch.stack([g.detach().float().reshape(-1) for g in source_grads], dim=0)
    witness_stack = torch.stack([g.detach().float().reshape(-1) for g in witness_grads], dim=0)
    source_mean = source_stack.mean(dim=0)
    basis = source_stack - source_mean.view(1, -1)
    ref_norm = ref.norm().clamp_min(1.0e-12)
    if float(basis.norm().detach().item()) <= 1.0e-12:
        return torch.zeros_like(ref), {
            "solver_status": "zero_source_conflict_basis",
            "witness_gradient_gain_LCB": math.nan,
            "correction_norm": 0.0,
        }
    h = (basis @ basis.T).detach().double().cpu().numpy()
    c = (witness_stack @ basis.T).detach().double().cpu().numpy()
    ref_gain = (witness_stack @ ref).detach().double().cpu().numpy()
    k = int(basis.shape[0])
    j = int(witness_stack.shape[0])
    gamma_value = max(float(args.o2_gamma), float(args.o2_gamma_frac) * max(0.0, float(np.mean(ref_gain))))
    xi0 = np.maximum(gamma_value - ref_gain, 0.0)
    x0 = np.concatenate([np.zeros((k,), dtype=np.float64), xi0])
    rho = float(args.o2_rho)
    cost = float(args.o2_cost)

    def objective(z: np.ndarray) -> float:
        a = z[:k]
        xi = z[k:]
        return 0.5 * (1.0 + cost) * float(a @ h @ a) + 0.5 * rho * float(xi @ xi)

    def grad_obj(z: np.ndarray) -> np.ndarray:
        a = z[:k]
        xi = z[k:]
        return np.concatenate([(1.0 + cost) * (h @ a), rho * xi])

    constraints: list[dict[str, Any]] = []
    for i in range(j):
        xi_i = np.zeros((j,), dtype=np.float64)
        xi_i[i] = 1.0
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda z, i=i: float(ref_gain[i] + c[i] @ z[:k] + z[k + i] - gamma_value),
                "jac": lambda z, i=i, xi_i=xi_i: np.concatenate([c[i], xi_i]),
            }
        )
    start = time.perf_counter()
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=grad_obj,
        bounds=[(None, None) for _ in range(k)] + [(0.0, None) for _ in range(j)],
        constraints=constraints,
        options={"maxiter": int(args.o2_solver_maxiter), "ftol": 1.0e-10, "disp": False},
    )
    solver_ms = (time.perf_counter() - start) * 1000.0
    z = res.x if getattr(res, "x", None) is not None else x0
    coeff = torch.tensor(z[:k], device=ref.device, dtype=ref.dtype)
    delta_raw = coeff.view(1, -1) @ basis.to(device=ref.device, dtype=ref.dtype)
    delta_raw = delta_raw.reshape(-1)
    raw_norm = delta_raw.norm().clamp_min(1.0e-12)
    trust = float(args.o2_correction_trust_ratio) * ref_norm
    alpha = min(1.0, float(trust.detach().item()) / float(raw_norm.detach().item()))
    real = delta_raw * alpha
    real_norm = real.norm().clamp_min(1.0e-12)
    if method.endswith("same_source_span_random"):
        coeff_r = torch.randn((k,), generator=generator, device=ref.device, dtype=ref.dtype)
        corr = coeff_r.view(1, -1) @ basis.to(device=ref.device, dtype=ref.dtype)
        corr = corr.reshape(-1)
        corr = corr / corr.norm().clamp_min(1.0e-12) * real_norm
        control_kind = "same_source_span_random"
    elif method.endswith("signflip"):
        corr = -real
        control_kind = "signflip"
    elif method.endswith("frozen_witness"):
        if "frozen_crpcf_corr" not in frozen_state:
            frozen_state["frozen_crpcf_corr"] = real.detach().clone()
        corr = frozen_state["frozen_crpcf_corr"].to(device=ref.device, dtype=ref.dtype)
        corr = corr / corr.norm().clamp_min(1.0e-12) * real_norm
        control_kind = "frozen_witness"
    elif method.endswith("same_norm"):
        corr = torch.randn(ref.shape, generator=generator, device=ref.device, dtype=ref.dtype)
        corr = corr / corr.norm().clamp_min(1.0e-12) * real_norm
        control_kind = "same_norm"
    else:
        corr = real
        control_kind = "crpcf_true" if method.endswith("crpcf_true") else "shuffled_witness"
    pred_before = ref_gain
    pred_after = (witness_stack.to(device=ref.device, dtype=ref.dtype) @ (ref + corr)).detach().double().cpu().numpy()
    pred_gain_delta = pred_after - pred_before
    return corr.detach(), {
        "solver_status": "pass" if bool(getattr(res, "success", False)) else "warn",
        "solver_message": str(getattr(res, "message", "")),
        "solver_ms": solver_ms,
        "solver_iters": int(getattr(res, "nit", -1)),
        "o2_control_kind": control_kind,
        "o2_reference_direction_norm": float(ref_norm.detach().item()),
        "correction_norm": float(corr.norm().detach().item()),
        "real_correction_norm_before_control": float(real_norm.detach().item()),
        "correction_norm_over_reference_norm": float(corr.norm().detach().item() / ref_norm.detach().item()),
        "witness_gradient_gain_mean": float(np.mean(pred_gain_delta)) if len(pred_gain_delta) else math.nan,
        "witness_gradient_gain_LCB": lcb95([float(v) for v in pred_gain_delta]),
        "o2_gamma_value": float(gamma_value),
        "o2_trust_alpha": float(alpha),
    }


def wcfu_correction(
    method: str,
    grads: list[Any],
    generator: Any,
    frozen_state: dict[str, Any],
) -> tuple[Any, dict[str, float]]:
    import torch

    n = len(grads)
    split = max(1, n // 2)
    source = grads[:split]
    witness = grads[split:] or grads[:split]
    source_mean = torch.stack(source).mean(dim=0)
    all_mean = torch.stack(grads).mean(dim=0)
    proj = all_mean * (torch.dot(source_mean, all_mean) / all_mean.dot(all_mean).clamp_min(1.0e-12))
    real = source_mean - proj
    real_norm = real.norm().clamp_min(1.0e-12)
    if method.endswith("same_source_span_random"):
        coeff = torch.randn((len(source),), generator=generator, device=real.device)
        corr = coeff @ torch.stack(source)
        corr = corr / corr.norm().clamp_min(1.0e-12) * real_norm
    elif method.endswith("signflip"):
        corr = -real
    elif method.endswith("frozen_witness"):
        if "frozen_corr" not in frozen_state:
            frozen_state["frozen_corr"] = real.detach().clone()
        corr = frozen_state["frozen_corr"].to(real.device)
        corr = corr / corr.norm().clamp_min(1.0e-12) * real_norm
    elif method.endswith("same_norm"):
        corr = torch.randn(real.shape, generator=generator, device=real.device)
        corr = corr / corr.norm().clamp_min(1.0e-12) * real_norm
    else:
        corr = real
    witness_for_score = list(witness)
    if method.endswith("shuffled_witness") and len(witness_for_score) > 1:
        witness_for_score = witness_for_score[1:] + witness_for_score[:1]
    gains = torch.stack([-torch.dot(g, corr) for g in witness_for_score]).float()
    lcb = float((gains.mean() - 1.96 * gains.std(unbiased=False) / math.sqrt(max(1, int(gains.numel())))).detach().item())
    source_gain = float((-torch.dot(source_mean, corr)).detach().item())
    witness_gain = float(gains.mean().detach().item())
    if method.endswith("wcfu") and lcb <= 0.0:
        corr = torch.zeros_like(corr)
    return corr, {
        "witness_transfer_LCB": lcb,
        "source_gain_proxy": source_gain,
        "witness_gain_proxy": witness_gain,
        "source_witness_gap": source_gain - witness_gain,
        "correction_norm": float(corr.norm().detach().item()),
        "real_correction_norm_before_gate": float(real_norm.detach().item()),
    }


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    method = str(args.method)
    if method in LEGACY_PROXY_O2_METHODS:
        raise RuntimeError(
            "Legacy O2 proxy methods are invalidated for v22.53O. "
            "Use a true v22.53 WCFU training implementation only; this runner "
            "will not synthesize WCFU evidence from proxy directions."
        )
    if any(method.startswith(prefix) for prefix in REJECTED_DIAGNOSTIC_O2_PREFIXES):
        raise RuntimeError(
            "CR-PCF O2 diagnostic methods are disabled for v22.53O claim runs "
            "because the user rejected custom source/witness direction solving "
            "and manual correction updates as not true training."
        )
    device = torch_device(str(args.device))
    torch.manual_seed(int(args.seed) + 225300)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    tensors = load_bundle_tensors(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    if tensors["used_fake_data"]:
        raise RuntimeError("fake data is not allowed for v22.53O")
    model = make_model(int(tensors["input_dim"]), int(tensors["num_classes"]), int(args.hidden), int(args.seed) + 225300, device)
    optimizer_method = o2_reference_method(method) if method in O2_METHODS else method
    poet_audit: dict[str, Any] = {}
    if optimizer_method in {"poet_official", "poetx_mem_official"}:
        model, opt, poet_audit = maybe_wrap_poet(model, optimizer_method, args)
    else:
        opt = make_optimizer(optimizer_method, model, args, device)
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
    train_losses: list[float] = []
    step_ms: list[float] = []
    opt_ms: list[float] = []
    wcfu_diags: list[dict[str, float]] = []
    frozen_state: dict[str, Any] = {}
    total_start = time.perf_counter()
    for step in range(int(args.steps)):
        idx = batch_indices(int(x_train.shape[0]), int(args.batch_size), generator, device)
        step_start = time.perf_counter()
        if method in O2_MANUAL_CORRECTION_METHODS:
            params_named = named_params(model)
            params = [p for _n, p in params_named]
            cohorts = cohort_indices_from_batch(idx, int(args.cohorts))
            split = max(1, len(cohorts) // 2)
            source_cohorts = cohorts[:split]
            witness_cohorts = cohorts[split:] or cohorts[:split]
            source_grads, source_losses = cohort_gradients_train_labels(
                model, x_train, y_train, source_cohorts, params, generator=generator
            )
            witness_grads_true, witness_losses = cohort_gradients_train_labels(
                model, x_train, y_train, witness_cohorts, params, generator=generator
            )
            witness_grads_for_solver = witness_grads_true
            if method.endswith("shuffled_witness"):
                witness_grads_for_solver, _ = cohort_gradients_train_labels(
                    model,
                    x_train,
                    y_train,
                    witness_cohorts,
                    params,
                    generator=generator,
                    shuffle_labels=True,
                )
            if method.endswith("frozen_witness"):
                if "frozen_witness_grads" not in frozen_state:
                    frozen_state["frozen_witness_grads"] = [g.detach().clone() for g in witness_grads_true]
                witness_grads_for_solver = [g.to(device=device) for g in frozen_state["frozen_witness_grads"]]
            grads = source_grads + witness_grads_true
            mean_grad = torch.stack(grads).mean(dim=0)
            source_loss_before = float(statistics.fmean(source_losses))
            witness_loss_before = float(statistics.fmean(witness_losses))
            opt.zero_grad(set_to_none=True)
            assign_flat_grad(params_named, mean_grad)
            before_flat = flatten_params(params_named).to(device=device)
            opt_start = time.perf_counter()
            opt.step()
            after_reference_flat = flatten_params(params_named).to(device=device)
            reference_direction = (before_flat - after_reference_flat) / max(float(args.lr), 1.0e-12)
            corr, diag = crossfit_crpcf_correction(
                method,
                source_grads,
                witness_grads_for_solver,
                reference_direction,
                generator,
                frozen_state,
                args,
            )
            correction_update_norm = apply_flat_update(params_named, corr, float(args.lr), float(args.wcfu_scale))
            opt_ms.append((time.perf_counter() - opt_start) * 1000.0)
            source_loss_after = cohort_losses_no_grad(model, x_train, y_train, source_cohorts)
            witness_loss_after = cohort_losses_no_grad(model, x_train, y_train, witness_cohorts)
            actual_source_transfer = source_loss_before - float(source_loss_after)
            actual_witness_transfer = witness_loss_before - float(witness_loss_after)
            train_losses.append(float(statistics.fmean(source_losses + witness_losses)))
            diag["correction_update_norm"] = correction_update_norm
            diag["source_transfer_actual"] = actual_source_transfer
            diag["witness_transfer_actual"] = actual_witness_transfer
            diag["source_witness_gap"] = actual_source_transfer - actual_witness_transfer
            diag["witness_loss_before"] = witness_loss_before
            diag["witness_loss_after"] = float(witness_loss_after)
            wcfu_diags.append(diag)
            opt.zero_grad(set_to_none=True)
        else:
            opt.zero_grad(set_to_none=True)
            xb = x_train[idx]
            yb = y_train[idx].long()
            logits = model(xb).float()
            loss = F.cross_entropy(logits, yb)
            if method in O2_LOSS_TRUE_TRAINING_METHODS:
                positions = torch.arange(int(idx.numel()), device=device)
                cohorts = cohort_indices_from_batch(positions, int(args.cohorts))
                split = max(1, len(cohorts) // 2)
                source_pos = torch.cat([c for c in cohorts[:split] if int(c.numel()) > 0], dim=0)
                witness_pos = torch.cat([c for c in cohorts[split:] if int(c.numel()) > 0], dim=0)
                source_loss_before = float(F.cross_entropy(logits[source_pos], yb[source_pos]).detach().item())
                witness_loss_before = float(F.cross_entropy(logits[witness_pos], yb[witness_pos]).detach().item())
                aux_loss, diag = xwcfu_loss_from_logits(method, logits, yb, cohorts, generator)
                loss = loss + float(args.wcfu_scale) * aux_loss
            loss.backward()
            opt_start = time.perf_counter()
            opt.step()
            if optimizer_method in {"poet_official", "poetx_mem_official"}:
                try:
                    model.merge_if_needed(step + 1)
                except Exception:
                    pass
            opt_ms.append((time.perf_counter() - opt_start) * 1000.0)
            train_losses.append(float(loss.detach().item()))
            if method in O2_LOSS_TRUE_TRAINING_METHODS:
                with torch.no_grad():
                    after_logits = model(xb).float()
                    source_loss_after = float(F.cross_entropy(after_logits[source_pos], yb[source_pos]).detach().item())
                    witness_loss_after = float(F.cross_entropy(after_logits[witness_pos], yb[witness_pos]).detach().item())
                diag["source_transfer_actual"] = source_loss_before - source_loss_after
                diag["witness_transfer_actual"] = witness_loss_before - witness_loss_after
                diag["source_witness_gap"] = diag["source_transfer_actual"] - diag["witness_transfer_actual"]
                diag["loss_level_wcfu_scale"] = float(args.wcfu_scale)
                wcfu_diags.append(diag)
        step_ms.append((time.perf_counter() - step_start) * 1000.0)
    wall = time.perf_counter() - total_start
    if method == "schedule_free_adamw_local" and hasattr(opt, "swap_to_average"):
        old_params = opt.swap_to_average()
    else:
        old_params = None
    post_held = evaluate_tensors(model, x_held, y_held, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    post_test = evaluate_tensors(model, x_test, y_test, device, int(tensors["num_classes"]), int(args.eval_batch_size))
    actuator = functional_actuator_probe(model, x_held, device, directions=int(args.actuator_probe_directions))
    final_spectra = current_weight_spectra(model)
    if old_params is not None:
        opt.restore(old_params)
    drift = spectrum_drift(initial_spectra, final_spectra)
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    trainable_count = trainable_param_count(model)
    opt_state_count = optimizer_state_count(opt)
    matrix_numel = sum(int(p.numel()) for p in model.parameters() if p.requires_grad and p.ndim == 2)
    total_numel = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    non_oet_fraction = 1.0 - float(matrix_numel / max(1, total_numel))
    method_diags = getattr(opt, "diags", [])
    diag_mean = {
        key: mean(fval(d.get(key)) for d in method_diags) for key in [
            "orthogonality_error",
            "rotation_angle",
            "radial_channel_fraction",
            "update_side_out",
        ]
    }
    wcfu_mean = {
        key: mean(fval(d.get(key)) for d in wcfu_diags) for key in [
            "source_transfer_actual",
            "witness_transfer_actual",
            "source_witness_gap",
            "correction_norm",
            "real_correction_norm_before_control",
            "correction_update_norm",
            "correction_norm_over_reference_norm",
            "solver_ms",
            "witness_gradient_gain_mean",
            "witness_gradient_gain_LCB",
            "xwcfu_aux_loss",
            "xwcfu_cosine",
            "xwcfu_target_norm",
            "xwcfu_witness_norm",
            "loss_level_wcfu_scale",
        ]
    }
    if wcfu_diags:
        wcfu_mean["witness_transfer_LCB"] = lcb95(
            fval(d.get("witness_transfer_actual")) for d in wcfu_diags
        )
        wcfu_mean["source_transfer_LCB"] = lcb95(
            fval(d.get("source_transfer_actual")) for d in wcfu_diags
        )
    auc_loss_time = float(statistics.fmean(train_losses)) if train_losses else math.nan
    row = {
        "run_label": str(args.label),
        "phase": "O2" if method in O2_METHODS else "O1",
        "training_mode": "oet_reference_plus_loss_level_xwcfu_standard_backward_optimizer_step" if method in O2_LOSS_TRUE_TRAINING_METHODS else ("oet_reference_plus_crossfit_crpcf_full_training" if method in O2_MANUAL_CORRECTION_METHODS else "normal_forward_backward_optimizer_step"),
        "evidence_status": "valid_loss_level_true_training_o2" if method in O2_LOSS_TRUE_TRAINING_METHODS else ("valid_true_training_o2" if method in O2_MANUAL_CORRECTION_METHODS else "valid_true_training_baseline"),
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "method": method,
        "reference_method": o2_reference_method(method) if method in O2_METHODS else "",
        "implementation_source": implementation_source(method),
        "model_family": f"MLP_hidden{int(args.hidden)}",
        "device": str(args.device),
        "python": PYTHON,
        "train_size": int(args.train_size),
        "held_size": int(args.held_size),
        "test_size": int(args.test_size),
        "hidden": int(args.hidden),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "source_kind": tensors["source_kind"],
        "used_fake_data": tensors["used_fake_data"],
        "final_NLL": post_held["NLL"],
        "accuracy": post_held["accuracy"],
        "test_NLL": post_test["NLL"],
        "test_accuracy": post_test["accuracy"],
        "held_NLL_pre": pre_held["NLL"],
        "held_NLL_gain": float(pre_held["NLL"]) - float(post_held["NLL"]),
        "test_NLL_gain": float(pre_test["NLL"]) - float(post_test["NLL"]),
        "accuracy_delta": float(post_held["accuracy"]) - float(pre_held["accuracy"]),
        "AUC_loss_time": auc_loss_time,
        "wallclock_adjusted_AUC": auc_loss_time * wall if math.isfinite(auc_loss_time) else math.nan,
        "ECE": post_held["ECE"],
        "Brier": post_held["Brier"],
        "tail_loss_q95": post_held["tail_q95"],
        "tail_loss_q99": post_held["tail_q99"],
        "margin_q10": post_held["margin_q10"],
        "margin_q01": post_held["margin_q01"],
        "ECE_delta": float(post_held["ECE"]) - float(pre_held["ECE"]),
        "Brier_delta": float(post_held["Brier"]) - float(pre_held["Brier"]),
        "tail_q95_delta": float(post_held["tail_q95"]) - float(pre_held["tail_q95"]),
        "tail_q99_delta": float(post_held["tail_q99"]) - float(pre_held["tail_q99"]),
        "no_debt": int(
            float(post_held["ECE"]) <= float(pre_held["ECE"]) + float(args.debt_tolerance)
            and float(post_held["Brier"]) <= float(pre_held["Brier"]) + float(args.debt_tolerance)
            and float(post_held["tail_q95"]) <= float(pre_held["tail_q95"]) + float(args.tail_debt_tolerance)
        ),
        "peak_memory_mb": peak_memory,
        "wall_seconds": wall,
        "step_time_ms": mean(step_ms),
        "optimizer_step_time_ms": mean(opt_ms),
        "controller_or_reparam_overhead_ratio": float((sum(opt_ms) / max(1.0e-9, wall * 1000.0))) if opt_ms else 0.0,
        "trainable_parameter_count": trainable_count,
        "optimizer_state_parameter_count": opt_state_count,
        "flops_proxy": int(args.steps) * int(args.batch_size) * trainable_count,
        "non_oet_param_fraction": non_oet_fraction,
        "adamw_fallback_param_fraction": non_oet_fraction if method in {"pion_oet_sphere_official", "pion_oet_local", "pion_hpns_optml_optional", "muon_like_local", "soap_shampoo_diag_diagnostic"} else 0.0,
        "frozen_param_fraction": 0.0,
        "exp_approx_type": exp_approx_type(method),
        "alternate_or_bilateral_update": "official_POET_reference_plus_loss_level_xwcfu" if method in O2_LOSS_TRUE_TRAINING_METHODS else ("left_or_right_small_side" if method == "pion_oet_local" or method in O2_MANUAL_CORRECTION_METHODS else ("bilateral_truncated" if method == "pion_oet_sphere_official" else "")),
        "singular_value_drift_mean": drift["singular_value_drift_mean"],
        "singular_value_drift_max": drift["singular_value_drift_max"],
        "generalized_spectrum_drift": drift["singular_value_drift_mean"],
        "orthogonality_error_left": diag_mean.get("orthogonality_error"),
        "orthogonality_error_right": "",
        "left_rotation_angle_mean": diag_mean.get("rotation_angle"),
        "right_rotation_angle_mean": "",
        "left_right_rotation_imbalance": "",
        "radial_channel_fraction": diag_mean.get("radial_channel_fraction"),
        "OET_fidelity_after_FU": drift["singular_value_drift_max"],
        **actuator,
        **{key: ("" if value is None else value) for key, value in wcfu_mean.items()},
        **poet_audit,
    }
    out = CHUNK_ROOT / f"v22_53O_{safe_fragment(str(args.label))}_summary.csv"
    write_rows(out, [row])
    return row


def implementation_source(method: str) -> str:
    if method == "pion_oet_sphere_official":
        return "official_repo_Sphere-AI-Lab/pion_megatron_PionOptimizer_matrix_only_with_adamw_1d_fallback"
    if method == "poet_official":
        return "official_repo_Sphere-AI-Lab/poet_poet_torch_POETModel"
    if method == "poetx_mem_official":
        return "official_repo_Sphere-AI-Lab/poet_poet_torch_POETModel_mem_efficient_mode"
    if method == "pion_hpns_optml_optional":
        return "official_repo_OPTML-Group/Pion_high_pass_ns_wrapped_single_gpu_not_spectrum_OET"
    if method in O2_LOSS_TRUE_TRAINING_METHODS:
        return "local_loss_level_XWCFU_over_official_POET_standard_backward_optimizer_step"
    if method in O2_MANUAL_CORRECTION_METHODS:
        return "local_true_training_CR_PCF_over_local_Pion_OET_reference_not_official_v22_53_repo"
    if method == "pion_oet_local":
        return "local_reimplementation_from_paper_not_official_repo"
    if method in {"schedule_free_adamw_local", "muon_like_local", "soap_shampoo_diag_diagnostic"}:
        return "local_compact_diagnostic_not_official_optimizer_package"
    return "torch_or_local_baseline"


def exp_approx_type(method: str) -> str:
    if method in O2_LOSS_TRUE_TRAINING_METHODS:
        return "official_POET_reference_plus_loss_level_xwcfu"
    if method in O2_MANUAL_CORRECTION_METHODS:
        return "exact_matrix_exp_reference_plus_crossfit_crpcf_correction"
    if method == "pion_oet_local":
        return "exact_matrix_exp_small_side_left_or_right"
    if method == "pion_oet_sphere_official":
        return "official_truncated_taylor_degree2"
    if method == "poet_official":
        return "official_POET_Cayley_Neumann"
    if method == "poetx_mem_official":
        return "official_POET_X_mem_efficient_Cayley_Neumann"
    if method == "pion_hpns_optml_optional":
        return "HPNS_gradient_filter_not_OET"
    return ""


def git_value(path: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def detect_license(path: Path) -> str:
    matches = []
    for pat in ["LICENSE*", "COPYING*"]:
        matches.extend(path.glob(pat))
    if matches:
        text = matches[0].read_text(encoding="utf-8", errors="replace")[:400].lower()
        if "mit license" in text or "permission is hereby granted" in text:
            return "MIT"
        if "apache license" in text:
            return "Apache"
        return matches[0].name
    readme = path / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="replace")[:3000].lower()
        if "license-mit" in text or "license: mit" in text:
            return "MIT_badge_or_readme"
    return "not_detected"


def code_available(repo_name: str, path: Path) -> int:
    if repo_name == "poet_sphere":
        return int((path / "poet_torch/model_wrapper.py").exists())
    if repo_name == "pion_spectrum_sphere":
        return int((path / "megatron-lm/megatron/core/optimizer/pion.py").exists())
    if repo_name == "pion_hpns_optml_optional":
        return int((path / "VLA/VLAAdapter/pion_optim/muon.py").exists())
    return int(path.exists())


def run_o0(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    py_compile.compile(str(ROOT / "experiments/run_v22_53O_pion_poet_external_oet_baseline.py"), doraise=True)
    append_exec(
        f"{PYTHON} -m py_compile experiments/run_v22_53O_pion_poet_external_oet_baseline.py",
        task_id="v22_53O_code_truth_gate",
        status="pass",
        gpu="cpu",
        files="experiments/run_v22_53O_pion_poet_external_oet_baseline.py",
        note="runner compile gate",
    )
    install_rows = []
    import_rows = []
    smoke_rows = []
    provenance_rows = []
    license_rows = []
    OET_ROOT.mkdir(parents=True, exist_ok=True)
    for repo in REPOS:
        path = Path(repo["path"])
        clone_status = "present" if path.exists() else "missing"
        commit = git_value(path, ["rev-parse", "HEAD"]) if path.exists() else ""
        branch = git_value(path, ["branch", "--show-current"]) if path.exists() else ""
        license_detected = detect_license(path) if path.exists() else "missing"
        row = {
            "repo_name": repo["repo_name"],
            "repo_url": repo["repo_url"],
            "clone_status": clone_status,
            "commit_hash": commit,
            "branch": branch,
            "license_detected": license_detected,
            "paper_url": repo["paper_url"],
            "code_available": code_available(repo["repo_name"], path),
            "pip_install_pass": "",
            "import_pass": "",
            "minimal_train_smoke_pass": "",
            "optimizer_class_detected": "",
            "reparameterization_class_detected": "",
            "supports_2d_linear_weight": "",
            "supports_conv_weight": "",
            "supports_bias_or_1d_param": "",
            "requires_llm_stack": "",
            "requires_distributed_training": "",
            "claim_boundary": repo["claim_boundary"],
        }
        provenance_rows.append(row)
        license_rows.append(
            {
                "repo_name": repo["repo_name"],
                "license_detected": license_detected,
                "license_file_present": int(license_detected not in {"missing", "not_detected", "MIT_badge_or_readme"}),
                "audit_note": "POET top-level LICENSE not detected" if repo["repo_name"] == "poet_sphere" and license_detected == "not_detected" else "",
            }
        )
    write_rows(OUT_ROOT / "v22_53O_external_oet_repo_provenance.csv", provenance_rows)
    write_rows(OUT_ROOT / "v22_53O_external_oet_license_audit.csv", license_rows)
    install_cmds = [
        ([PYTHON, "-m", "pip", "install", "-e", "external/oet_baselines/poet_sphere"], "poet_sphere_pip_install_build_isolation"),
        ([PYTHON, "-m", "pip", "install", "-e", "external/oet_baselines/poet_sphere", "--no-build-isolation", "--no-deps"], "poet_sphere_pip_install_no_build_isolation_no_deps"),
    ]
    for cmd, task in install_cmds:
        result = run_cmd(cmd, task_id=f"v22_53O_{task}", files="results/v22_53O/v22_53O_external_oet_install_smoke.csv")
        install_rows.append({"repo_name": "poet_sphere", "install_task": task, **result})
    import_specs = [
        (
            "poet_sphere",
            [PYTHON, "-c", "from poet_torch import POETConfig, POETModel, get_poet_optimizer; print('pass')"],
            "POETConfig;POETModel;get_poet_optimizer",
            "POETModel",
        ),
        (
            "pion_spectrum_sphere",
            [
                PYTHON,
                "-c",
                "import sys; sys.path.insert(0, 'external/oet_baselines/pion_spectrum_sphere/megatron-lm'); "
                "from megatron.core.optimizer.pion import PionOptimizer; print('pass', PionOptimizer)",
            ],
            "PionOptimizer",
            "",
        ),
        (
            "pion_hpns_optml_optional",
            [
                PYTHON,
                "-c",
                "import sys; sys.path.insert(0, 'external/oet_baselines/pion_hpns_optml_optional/VLA/VLAAdapter'); "
                "from pion_optim.muon import DefaultPion, Muon, LowRankMuon; print('pass', DefaultPion, Muon, LowRankMuon)",
            ],
            "DefaultPion;Muon;LowRankMuon",
            "",
        ),
    ]
    for repo_name, cmd, optimizer_cls, reparam_cls in import_specs:
        result = run_cmd(cmd, task_id=f"v22_53O_{repo_name}_import_smoke", files="results/v22_53O/v22_53O_external_oet_import_matrix.csv")
        import_rows.append(
            {
                "repo_name": repo_name,
                "import_pass": int(result["status"] == "pass"),
                "optimizer_class_detected": optimizer_cls,
                "reparameterization_class_detected": reparam_cls,
                **result,
            }
        )
    smoke_specs = [
        ("poet_sphere", "poet_official"),
        ("pion_spectrum_sphere", "pion_oet_sphere_official"),
        ("pion_hpns_optml_optional", "pion_hpns_optml_optional"),
        ("local_pion_oet_reimplementation", "pion_oet_local"),
    ]
    for repo_name, method in smoke_specs:
        label = f"v22_53O_o0_smoke_{method}"
        cmd = collect_command(
            {
                "dataset": "MNIST",
                "seed": 0,
                "method": method,
                "label": label,
            },
            argparse.Namespace(**{**vars(args), "steps": int(args.smoke_steps), "train_size": 128, "held_size": 64, "test_size": 64}),
            "cuda:0" if args.smoke_device == "auto" else str(args.smoke_device),
        )
        result = run_cmd(cmd, task_id=label, files=f"results/v22_53O/chunks/{label}_summary.csv", gpu="0")
        smoke_rows.append({"repo_name": repo_name, "method": method, "minimal_train_smoke_pass": int(result["status"] == "pass"), **result})
    write_rows(OUT_ROOT / "v22_53O_external_oet_install_smoke.csv", install_rows)
    write_rows(OUT_ROOT / "v22_53O_external_oet_import_matrix.csv", import_rows)
    write_rows(OUT_ROOT / "v22_53O_external_oet_minimal_optimizer_smoke.csv", smoke_rows)
    # Update provenance with smoke statuses.
    install_pass = {r["repo_name"]: max([iflag(x.get("status")) for x in install_rows if x["repo_name"] == r["repo_name"]] or [0]) for r in provenance_rows}
    import_pass = {r["repo_name"]: max([int(x.get("import_pass") or 0) for x in import_rows if x["repo_name"] == r["repo_name"]] or [0]) for r in provenance_rows}
    smoke_pass = {r["repo_name"]: max([int(x.get("minimal_train_smoke_pass") or 0) for x in smoke_rows if x["repo_name"] == r["repo_name"]] or [0]) for r in provenance_rows}
    for row in provenance_rows:
        row["pip_install_pass"] = install_pass.get(row["repo_name"], "")
        row["import_pass"] = import_pass.get(row["repo_name"], "")
        row["minimal_train_smoke_pass"] = smoke_pass.get(row["repo_name"], "")
        if row["repo_name"] == "poet_sphere":
            row["reparameterization_class_detected"] = "POETModel"
            row["optimizer_class_detected"] = "POETAdamW"
            row["supports_2d_linear_weight"] = 1
            row["supports_conv_weight"] = 0
            row["supports_bias_or_1d_param"] = "bias_optional_or_adamw_fallback"
            row["requires_llm_stack"] = 0
            row["requires_distributed_training"] = 0
        elif row["repo_name"] == "pion_spectrum_sphere":
            row["optimizer_class_detected"] = "PionOptimizer"
            row["supports_2d_linear_weight"] = 1
            row["supports_conv_weight"] = 0
            row["supports_bias_or_1d_param"] = "requires_mixed_adamw_fallback"
            row["requires_llm_stack"] = "Megatron module import path, direct Optimizer usable for small 2D smoke"
            row["requires_distributed_training"] = 0
        elif row["repo_name"] == "pion_hpns_optml_optional":
            row["optimizer_class_detected"] = "DefaultPion;Muon;LowRankMuon"
            row["supports_2d_linear_weight"] = 1
            row["supports_conv_weight"] = 1
            row["supports_bias_or_1d_param"] = "requires AdamW fused variant or local fallback"
            row["requires_llm_stack"] = 0
            row["requires_distributed_training"] = "official base class uses dist all_gather; runner wraps high_pass_ns function for single GPU"
    write_rows(OUT_ROOT / "v22_53O_external_oet_repo_provenance.csv", provenance_rows)
    write_json(
        OUT_ROOT / "v22_53O_external_oet_claim_boundary.json",
        {
            "poet_sphere": "Official POET/POET-X external reparameterization baseline only when POETModel train smoke passes; not same-param DG-KAN proof.",
            "pion_spectrum_sphere": "Official Pion OET baseline when downloaded Megatron PionOptimizer is imported and stepped; 1D/bias params use explicit AdamW fallback.",
            "pion_hpns_optml_optional": "Pion-HPNS optional baseline, not spectrum-preserving Pion OET; kept separate from Sphere Pion.",
            "local_pion_oet_reimplementation": "Local reimplementation from paper definitions; strong geometry diagnostic but never official external repo result.",
        },
    )
    append_recap(
        "O0 provenance/install/import/smoke",
        "O0 已完成。关键 artifact：\n\n"
        "- `results/v22_53O/v22_53O_external_oet_repo_provenance.csv`\n"
        "- `results/v22_53O/v22_53O_external_oet_license_audit.csv`\n"
        "- `results/v22_53O/v22_53O_external_oet_install_smoke.csv`\n"
        "- `results/v22_53O/v22_53O_external_oet_import_matrix.csv`\n"
        "- `results/v22_53O/v22_53O_external_oet_minimal_optimizer_smoke.csv`\n"
        "- `results/v22_53O/v22_53O_external_oet_claim_boundary.json`\n\n"
        "边界：Sphere Pion 与 POET/POET-X 分开审计；OPTML Pion 仅作为 HPNS optional baseline；local Pion-OET 只作 paper-faithful diagnostic。",
    )
    return {"status": "pass", "repos": len(provenance_rows), "smoke_rows": len(smoke_rows)}


def collect_command(spec: dict[str, Any], args: argparse.Namespace, device: str) -> list[str]:
    return [
        PYTHON,
        "experiments/run_v22_53O_pion_poet_external_oet_baseline.py",
        "--stage",
        "collect",
        "--dataset",
        str(spec["dataset"]),
        "--seed",
        str(spec["seed"]),
        "--method",
        str(spec["method"]),
        "--label",
        str(spec["label"]),
        "--device",
        str(device),
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
        "--wcfu-scale",
        str(args.wcfu_scale),
        "--o2-gamma",
        str(args.o2_gamma),
        "--o2-gamma-frac",
        str(args.o2_gamma_frac),
        "--o2-rho",
        str(args.o2_rho),
        "--o2-cost",
        str(args.o2_cost),
        "--o2-correction-trust-ratio",
        str(args.o2_correction_trust_ratio),
        "--o2-solver-maxiter",
        str(args.o2_solver_maxiter),
        "--actuator-probe-directions",
        str(args.actuator_probe_directions),
    ]


def run_matrix(args: argparse.Namespace, *, phase: str) -> list[dict[str, Any]]:
    ensure_out()
    datasets = split_csv(str(args.datasets), str)
    seeds = split_csv(str(args.seeds), int)
    methods = split_csv(str(args.methods), str) if args.methods else (O2_METHODS if phase == "O2" else BASELINE_METHODS)
    if phase == "O2" and not methods:
        return run_o2_true_training_blocker(args)
    specs = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                specs.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "label": f"{safe_fragment(args.label_prefix)}_{method}_{dataset}_s{seed}",
                    }
                )
    specs_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_specs.csv"
    dispatch_path = OUT_ROOT / f"{safe_fragment(args.label_prefix)}_dispatch_status.csv"
    write_rows(specs_path, specs)
    gpus = split_csv(str(args.gpus), str) or ["cpu"]
    dispatch_rows: list[dict[str, Any]] = []

    def launch(i_spec: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        i, spec = i_spec
        gpu = gpus[i % len(gpus)]
        device = "cpu" if gpu == "cpu" else f"cuda:{gpu}"
        cmd = collect_command(spec, args, device)
        stdout_path = LOG_ROOT / f"{safe_fragment(spec['label'])}_stdout.log"
        stderr_path = LOG_ROOT / f"{safe_fragment(spec['label'])}_stderr.log"
        start = time.time()
        env = os.environ.copy()
        env["KAN_PYTHON"] = PYTHON
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=out, stderr=err)
        row = {
            **spec,
            "gpu": gpu,
            "command": " ".join(shlex.quote(part) for part in cmd),
            "status": "pass" if proc.returncode == 0 else "fail",
            "returncode": proc.returncode,
            "wall_seconds": time.time() - start,
            "stdout": str(stdout_path.relative_to(ROOT)),
            "stderr": str(stderr_path.relative_to(ROOT)),
            "summary": f"results/v22_53O/chunks/v22_53O_{safe_fragment(spec['label'])}_summary.csv",
        }
        append_exec(
            row["command"],
            task_id=str(spec["label"]),
            status=row["status"],
            gpu=str(gpu),
            exit_code=proc.returncode,
            files=f"{row['stdout']}, {row['stderr']}, {row['summary']}",
            note=f"phase={phase} dataset={spec['dataset']} seed={spec['seed']} method={spec['method']}",
        )
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        for row in ex.map(launch, enumerate(specs)):
            dispatch_rows.append(row)
            write_rows(dispatch_path, dispatch_rows)
    return dispatch_rows


def run_o2_true_training_blocker(args: argparse.Namespace) -> list[dict[str, Any]]:
    blocker = {
        "timestamp": now_sg(),
        "phase": "O2",
        "status": "blocked_true_training_implementation_missing",
        "blocked_component": "WCFU_or_CR_PCF_on_external_OET_reference",
        "invalidated_legacy_methods": ",".join(LEGACY_PROXY_O2_METHODS),
        "required_training_behavior": "single end-to-end training loop; no proxy direction rows; no manual candidate-step scoring",
        "attempted_repair_1": "searched repo for v22.53/Cross-Fit/WCFU training entry; no v22.53 main runner found",
        "attempted_repair_2": "inspected v22.52 CR-PCF runner; it is a full-loop FU runner but has no external Pion/POET reference optimizer path and still uses internal functional direction assignment",
        "attempted_repair_3": "kept O1 external OET/POET/Pion true optimizer training as valid baseline evidence",
        "claim_boundary": "O2 WCFU over OET not claimed in v22.53O until a true v22.53 WCFU training wrapper exists",
    }
    path = OUT_ROOT / "v22_53O_o2_true_training_blocker.csv"
    write_rows(path, [blocker])
    append_exec(
        f"{PYTHON} experiments/run_v22_53O_pion_poet_external_oet_baseline.py --stage o2",
        task_id="v22_53O_o2_true_training_blocker",
        status="blocked",
        gpu="cpu",
        files=str(path.relative_to(ROOT)),
        note=blocker["claim_boundary"],
        exit_code=0,
    )
    append_recap(
        "O2 true-training blocker",
        "\n".join(
            [
                "O2 旧代理方法已禁止继续作为证据。按用户要求，本轮不再用逐步候选方向、梯度内积代理或手工 correction 行来补 WCFU。",
                "",
                f"- blocker artifact: `{path.relative_to(ROOT)}`",
                f"- invalidated legacy methods: `{blocker['invalidated_legacy_methods']}`",
                "- repair audit: 已搜索 v22.53/Cross-Fit/WCFU 训练入口；未找到可直接接入 Pion/POET reference 的真实训练 wrapper。v22.52 CR-PCF 是 full-loop FU runner，但没有外部 OET/POET reference optimizer path，且实现方式不满足本次用户要求的边界，因此不迁移冒充 O2。",
                "- claim boundary: O2 WCFU over OET 不作成功 claim；最终结论只允许引用 O0/O1 external OET 真训练 baseline。",
            ]
        ),
    )
    return [blocker]


def invalidate_legacy_o2_proxy(args: argparse.Namespace) -> dict[str, Any]:
    invalid_rows: list[dict[str, Any]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_53O_*_summary.csv")):
        for row in read_rows(path):
            if row.get("phase") == "O2" or row.get("method") in LEGACY_PROXY_O2_METHODS:
                invalid_rows.append(
                    {
                        "invalidated_at": now_sg(),
                        "source_file": str(path.relative_to(ROOT)),
                        "run_label": row.get("run_label", ""),
                        "dataset": row.get("dataset", ""),
                        "seed": row.get("seed", ""),
                        "method": row.get("method", ""),
                        "old_phase": row.get("phase", ""),
                        "evidence_status": "invalid_for_claim",
                        "invalid_reason": "legacy O2 used proxy gradient/correction construction instead of a true v22.53 WCFU training wrapper",
                        "user_boundary": "no proxy; no manual step/candidate-row calculation; only true training behavior",
                    }
                )
    out = OUT_ROOT / "v22_53O_o2_proxy_invalidation.csv"
    write_rows(out, invalid_rows)
    append_exec(
        f"{PYTHON} experiments/run_v22_53O_pion_poet_external_oet_baseline.py --stage invalidate-o2",
        task_id="v22_53O_o2_proxy_invalidation",
        status="pass",
        gpu="cpu",
        files=str(out.relative_to(ROOT)),
        note=f"invalidated_rows={len(invalid_rows)}; excluded from summarize/final claims",
        exit_code=0,
    )
    append_recap(
        "O2 proxy invalidation",
        "\n".join(
            [
                f"- invalidated rows: `{len(invalid_rows)}`",
                f"- artifact: `{out.relative_to(ROOT)}`",
                "- reason: 旧 O2 rows 使用了梯度/修正方向代理，不满足用户要求的真正训练行为，也不能代表计划里的 v22.53 WCFU/CR-PCF over external OET。",
                "- action: `summarize` 不再读取这些 O2 rows；最终结论中禁止引用它们的 NLL、LCB、control win 或 route。",
            ]
        ),
    )
    return {"invalidated_rows": len(invalid_rows), "artifact": str(out.relative_to(ROOT))}


def collect_chunk_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_53O_*_summary.csv")):
        rows.extend(read_rows(path))
    return rows


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows = collect_chunk_rows()
    strict_o1_rows = [
        r for r in rows
        if r.get("phase") == "O1" and str(r.get("run_label", "")).startswith("v22_53O_o1_strict_true_train")
    ]
    formal_o1_rows = [
        r for r in rows
        if r.get("phase") == "O1" and str(r.get("run_label", "")).startswith("v22_53O_o1_true_train")
    ]
    exploratory_o1_rows = [
        r for r in rows
        if r.get("phase") == "O1" and str(r.get("run_label", "")).startswith("v22_53O_o1")
    ]
    if strict_o1_rows:
        o1_rows = strict_o1_rows
        o1_selection = "strict_true_train"
    elif formal_o1_rows:
        o1_rows = formal_o1_rows
        o1_selection = "formal_true_train"
    else:
        o1_rows = exploratory_o1_rows
        o1_selection = "exploratory_o1_available"
    invalidated_o2 = read_rows(OUT_ROOT / "v22_53O_o2_proxy_invalidation.csv")
    invalidated_crpcf_o2 = read_rows(OUT_ROOT / "v22_53O_o2_crpcf_true_training_boundary_invalidation.csv")
    o2_rows = [
        r for r in rows
        if r.get("phase") == "O2" and str(r.get("run_label", "")).startswith("v22_53O_o2_true_train")
        and "smoke" not in str(r.get("run_label", "")).lower()
        and str(r.get("method", "")) in O2_METHODS
        and not any(str(r.get("method", "")).startswith(prefix) for prefix in REJECTED_DIAGNOSTIC_O2_PREFIXES)
    ]
    method_summary = []
    for method in sorted({r.get("method", "") for r in o1_rows + o2_rows}):
        group = [r for r in o1_rows + o2_rows if r.get("method") == method]
        method_summary.append(
            {
                "method": method,
                "phase": group[0].get("phase", "") if group else "",
                "rows": len(group),
                "mean_final_NLL": mean(fval(r.get("final_NLL")) for r in group),
                "mean_accuracy": mean(fval(r.get("accuracy")) for r in group),
                "mean_AUC_loss_time": mean(fval(r.get("AUC_loss_time")) for r in group),
                "no_debt_rows": sum(iflag(r.get("no_debt")) for r in group),
                "mean_overhead_ratio": mean(fval(r.get("controller_or_reparam_overhead_ratio")) for r in group),
                "mean_peak_memory_mb": mean(fval(r.get("peak_memory_mb")) for r in group),
                "implementation_source": group[0].get("implementation_source", "") if group else "",
            }
        )
    write_rows(OUT_ROOT / "v22_53O_method_summary.csv", method_summary)
    adamw_by_key = {(r.get("dataset"), str(r.get("seed"))): r for r in o1_rows if r.get("method") == "adamw"}
    poet_fairness = []
    for r in o1_rows + o2_rows:
        method = str(r.get("method", ""))
        if "poet" not in method:
            continue
        key = (r.get("dataset"), str(r.get("seed")))
        adam = adamw_by_key.get(key)
        adam_peak = fval(adam.get("peak_memory_mb")) if adam else math.nan
        adam_trainable = fval(adam.get("trainable_parameter_count")) if adam else math.nan
        adam_state = fval(adam.get("optimizer_state_parameter_count")) if adam else math.nan
        poet_fairness.append(
            {
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "method": method,
                "phase": r.get("phase"),
                "reference_method": r.get("reference_method"),
                "final_NLL": r.get("final_NLL"),
                "base_weight_parameter_count": r.get("base_weight_parameter_count"),
                "orthogonal_factor_parameter_count": r.get("orthogonal_factor_parameter_count"),
                "trainable_parameter_count_total": r.get("trainable_parameter_count_total") or r.get("trainable_parameter_count"),
                "optimizer_state_count_total": r.get("optimizer_state_parameter_count"),
                "merged_inference_available": r.get("merged_inference_available"),
                "merged_inference_parameter_count": r.get("merged_inference_parameter_count"),
                "train_overhead_ratio": r.get("controller_or_reparam_overhead_ratio"),
                "inference_overhead_ratio": "not_measured_merged_inference_path_only",
                "memory_peak_mb": r.get("peak_memory_mb"),
                "memory_peak_ratio_vs_adamw": (fval(r.get("peak_memory_mb")) or math.nan) / adam_peak if adam_peak and math.isfinite(adam_peak) else math.nan,
                "trainable_parameter_ratio_vs_adamw": (fval(r.get("trainable_parameter_count")) or math.nan) / adam_trainable if adam_trainable and math.isfinite(adam_trainable) else math.nan,
                "optimizer_state_ratio_vs_adamw": (fval(r.get("optimizer_state_parameter_count")) or math.nan) / adam_state if adam_state and math.isfinite(adam_state) else math.nan,
                "claim_boundary": "External POET reparameterization/support baseline; not same-param DG-KAN/FU proof.",
            }
        )
    write_rows(OUT_ROOT / "v22_53O_poet_reparameterization_fairness_audit.csv", poet_fairness)
    pairwise = []
    key_to_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in o1_rows:
        key_to_rows.setdefault((str(r.get("dataset")), str(r.get("seed"))), []).append(r)
    for key, group in sorted(key_to_rows.items()):
        adam = next((r for r in group if r.get("method") == "adamw"), None)
        cautious = next((r for r in group if r.get("method") == "cautious_adamw"), None)
        strongest = min(group, key=lambda r: fval(r.get("final_NLL"), float("inf")) or float("inf")) if group else None
        for r in group:
            pairwise.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "method": r.get("method"),
                    "final_NLL": r.get("final_NLL"),
                    "delta_NLL_vs_adamw": (fval(r.get("final_NLL")) or math.nan) - (fval(adam.get("final_NLL")) if adam else math.nan),
                    "delta_NLL_vs_cautious": (fval(r.get("final_NLL")) or math.nan) - (fval(cautious.get("final_NLL")) if cautious else math.nan),
                    "strongest_method": strongest.get("method") if strongest else "",
                    "delta_NLL_vs_strongest": (fval(r.get("final_NLL")) or math.nan) - (fval(strongest.get("final_NLL")) if strongest else math.nan),
                    "beats_adamw": int(adam is not None and (fval(r.get("final_NLL"), float("inf")) or float("inf")) < (fval(adam.get("final_NLL"), -float("inf")) or -float("inf"))),
                    "beats_cautious": int(cautious is not None and (fval(r.get("final_NLL"), float("inf")) or float("inf")) < (fval(cautious.get("final_NLL"), -float("inf")) or -float("inf"))),
                    "no_debt": r.get("no_debt"),
                    "overhead": r.get("controller_or_reparam_overhead_ratio"),
                }
            )
    write_rows(OUT_ROOT / "v22_53O_o1_pairwise_comparison.csv", pairwise)
    o2_pairwise = []
    ref_by_key = {(r.get("method"), r.get("dataset"), str(r.get("seed"))): r for r in o1_rows}
    controls = [m for m in O2_METHODS if m != O2_CANDIDATE_METHOD]
    for r in o2_rows:
        key = (r.get("dataset"), r.get("seed"))
        reference_method = str(r.get("reference_method") or o2_reference_method(str(r.get("method", ""))))
        ref = ref_by_key.get((reference_method, key[0], str(key[1])))
        same_key_o2 = [x for x in o2_rows if (x.get("dataset"), x.get("seed")) == key]
        best_control = min(
            [x for x in same_key_o2 if x.get("method") in controls],
            key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"),
            default=None,
        )
        o2_pairwise.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "method": r.get("method"),
                "reference_method": reference_method,
                "reference_final_NLL": ref.get("final_NLL") if ref else "",
                "final_NLL": r.get("final_NLL"),
                "Delta_NLL_vs_reference": (fval(r.get("final_NLL")) or math.nan) - (fval(ref.get("final_NLL")) if ref else math.nan),
                "beats_reference": int(ref is not None and (fval(r.get("final_NLL"), float("inf")) or float("inf")) < (fval(ref.get("final_NLL"), -float("inf")) or -float("inf"))),
                "best_control_method": best_control.get("method") if best_control else "",
                "best_control_final_NLL": best_control.get("final_NLL") if best_control else "",
                "beats_best_same_OET_control": int(best_control is not None and (fval(r.get("final_NLL"), float("inf")) or float("inf")) < (fval(best_control.get("final_NLL"), -float("inf")) or -float("inf"))),
                "witness_transfer_LCB": r.get("witness_transfer_LCB"),
                "witness_LCB_positive": int((fval(r.get("witness_transfer_LCB"), -float("inf")) or -float("inf")) > 0.0),
                "no_debt": r.get("no_debt"),
                "OET_fidelity_after_FU": r.get("OET_fidelity_after_FU"),
                "overhead": r.get("controller_or_reparam_overhead_ratio"),
            }
        )
    write_rows(OUT_ROOT / "v22_53O_o2_wcfu_over_oet_pairwise.csv", o2_pairwise)
    oet_methods = {"pion_oet_sphere_official", "pion_oet_local", "pion_hpns_optml_optional", "poet_official", "poetx_mem_official"}
    oet_pairwise_rows = [r for r in pairwise if r.get("method") in oet_methods]
    beats_adamw = sum(iflag(r.get("beats_adamw")) for r in oet_pairwise_rows)
    beats_strong_non_oet = 0
    for r in oet_pairwise_rows:
        method = r.get("method")
        key = (r.get("dataset"), str(r.get("seed")))
        non_oet = [
            x for x in o1_rows
            if x.get("dataset") == key[0] and str(x.get("seed")) == key[1] and x.get("method") not in oet_methods
        ]
        if non_oet:
            best = min(non_oet, key=lambda x: fval(x.get("final_NLL"), float("inf")) or float("inf"))
            if (fval(next(x for x in o1_rows if x.get("dataset") == key[0] and str(x.get("seed")) == key[1] and x.get("method") == method).get("final_NLL"), float("inf")) or float("inf")) < (fval(best.get("final_NLL"), -float("inf")) or -float("inf")):
                beats_strong_non_oet += 1
    o2_real = [r for r in o2_pairwise if r.get("method") == O2_CANDIDATE_METHOD]
    wcfu_improves = sum(iflag(r.get("beats_reference")) for r in o2_real)
    wcfu_beats_controls = sum(iflag(r.get("beats_best_same_OET_control")) for r in o2_real)
    wcfu_lcb_pos = sum(iflag(r.get("witness_LCB_positive")) for r in o2_real)
    wcfu_no_debt = sum(iflag(r.get("no_debt")) for r in o2_real)
    external_strong = beats_strong_non_oet >= 7 and sum(iflag(r.get("no_debt")) for r in oet_pairwise_rows) >= 8
    final_route = "InsufficientRows"
    if o1_rows:
        final_route = "ExternalOETCodeUnavailable_LocalOnly"
        if o2_real and wcfu_improves >= 5 and wcfu_beats_controls >= 6 and wcfu_lcb_pos >= 5 and wcfu_no_debt >= 7:
            final_route = "WitnessedCreditFUOverOETOpened"
        elif o2_real:
            final_route = "ExternalOETDominates_CurrentFUInsufficient" if external_strong else "FUWeakOptimizerPatchOnly_or_OETSupportExplains"
        elif external_strong:
            final_route = "ExternalOETBaselineStrong"
        else:
            final_route = "ExternalOETBaselineDiagnostic_NoOfficialFUClaim"
    route = {
        "final_route": final_route,
        "o1_selection": o1_selection,
        "o1_rows": len(o1_rows),
        "o2_rows": len(o2_rows),
        "legacy_proxy_o2_rows_invalidated": len(invalidated_o2),
        "crpcf_diagnostic_o2_rows_invalidated": len(invalidated_crpcf_o2),
        "oet_pairwise_rows": len(oet_pairwise_rows),
        "oet_beats_adamw_rows": beats_adamw,
        "oet_beats_strongest_non_oet_rows": beats_strong_non_oet,
        "wcfu_rows": len(o2_real),
        "wcfu_improves_vs_oet_reference_rows": wcfu_improves,
        "wcfu_beats_same_oet_controls_rows": wcfu_beats_controls,
        "wcfu_witness_lcb_positive_rows": wcfu_lcb_pos,
        "wcfu_no_debt_rows": wcfu_no_debt,
        "claim_boundary": "No KAN claim in v22.53O runner; KAN phase requires MLP WCFU over OET gate first.",
    }
    write_json(OUT_ROOT / "v22_53O_final_route.json", route)
    append_exec(
        f"{PYTHON} experiments/run_v22_53O_pion_poet_external_oet_baseline.py --stage summarize",
        task_id="v22_53O_summarize",
        status="pass",
        gpu="cpu",
        files="results/v22_53O/v22_53O_method_summary.csv, results/v22_53O/v22_53O_o1_pairwise_comparison.csv, results/v22_53O/v22_53O_o2_wcfu_over_oet_pairwise.csv, results/v22_53O/v22_53O_poet_reparameterization_fairness_audit.csv, results/v22_53O/v22_53O_final_route.json",
        note=json.dumps(route, ensure_ascii=False),
    )
    recap = [
        "### Final route",
        "",
        f"- Final route: `{final_route}`。",
        f"- O1 rows: `{len(o1_rows)}`；O1 selection: `{o1_selection}`；O2 accepted rows: `{len(o2_rows)}`；legacy proxy O2 invalidated: `{len(invalidated_o2)}`；CR-PCF diagnostic O2 invalidated: `{len(invalidated_crpcf_o2)}`。",
        f"- OET beats AdamW rows: `{beats_adamw}/{len(oet_pairwise_rows)}`。",
        f"- OET beats strongest non-OET rows: `{beats_strong_non_oet}/{len(oet_pairwise_rows)}`。",
        f"- WCFU improves vs OET reference: `{wcfu_improves}/{len(o2_real)}`。",
        f"- WCFU beats best same-OET controls: `{wcfu_beats_controls}/{len(o2_real)}`。",
        f"- WCFU witness LCB positive: `{wcfu_lcb_pos}/{len(o2_real)}`。",
        f"- WCFU no-debt: `{wcfu_no_debt}/{len(o2_real)}`。",
        "",
        "### Method means",
        "",
        md_table(method_summary, ["method", "phase", "rows", "mean_final_NLL", "mean_accuracy", "no_debt_rows", "mean_overhead_ratio"], limit=32),
        "### Evidence chain",
        "",
        "- Provenance/install/import/smoke: `v22_53O_external_oet_*` artifacts。",
        "- Strict O1 training matrix, if present, is selected before older formal/exploratory O1 rows: `results/v22_53O/v22_53O_o1_strict_true_train_dispatch_status.csv`。",
        "- O1 pairwise: `results/v22_53O/v22_53O_o1_pairwise_comparison.csv`。",
        "- O2 WCFU/control pairwise: `results/v22_53O/v22_53O_o2_wcfu_over_oet_pairwise.csv`。",
        "- POET fairness audit: `results/v22_53O/v22_53O_poet_reparameterization_fairness_audit.csv`。",
        "- Formal O1 rerun disables finite-difference actuator probes; older exploratory probe fields, if present, are not used for route or claims。",
        "",
        "### Claim boundary",
        "",
        "本轮不写 KAN claim。POET/POET-X 若获胜只能是 external reparameterization/support baseline；Sphere Pion 若获胜只能是 external OET optimizer baseline；local Pion-OET 只能是 paper-faithful diagnostic。",
    ]
    append_recap("O1/O2 summary and route", "\n".join(recap))
    return route


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all", choices=["all", "o0", "collect", "o1", "o2", "invalidate-o2", "summarize"])
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="adamw")
    p.add_argument("--label", default="v22_53O_collect")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default="")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label-prefix", default="v22_53O_o1")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--steps", type=int, default=90)
    p.add_argument("--smoke-steps", type=int, default=8)
    p.add_argument("--smoke-device", default="auto")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--cohorts", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--debt-tolerance", type=float, default=1.0e-6)
    p.add_argument("--tail-debt-tolerance", type=float, default=1.0e-6)
    p.add_argument("--poet-block-size", type=int, default=16)
    p.add_argument("--poet-merge-interval", type=int, default=30)
    p.add_argument("--poet-lr", type=float, default=5.0e-4)
    p.add_argument("--poet-scale", type=float, default=0.5)
    p.add_argument("--wcfu-scale", type=float, default=0.12)
    p.add_argument("--o2-gamma", type=float, default=0.0)
    p.add_argument("--o2-gamma-frac", type=float, default=0.80)
    p.add_argument("--o2-rho", type=float, default=25.0)
    p.add_argument("--o2-cost", type=float, default=1.0e-4)
    p.add_argument("--o2-correction-trust-ratio", type=float, default=0.25)
    p.add_argument("--o2-solver-maxiter", type=int, default=80)
    p.add_argument("--actuator-probe-directions", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    try:
        if args.stage == "collect":
            row = run_collect(args)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "o0":
            print(json.dumps(run_o0(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "o1":
            args.label_prefix = args.label_prefix or "v22_53O_o1"
            rows = run_matrix(args, phase="O1")
            print(json.dumps({"rows": len(rows), "failed": sum(1 for r in rows if r["status"] != "pass")}, ensure_ascii=False))
            return 0 if all(r["status"] == "pass" for r in rows) else 1
        if args.stage == "o2":
            if args.label_prefix == "v22_53O_o1":
                args.label_prefix = "v22_53O_o2"
            rows = run_matrix(args, phase="O2")
            print(
                json.dumps(
                    {
                        "rows": len(rows),
                        "failed": sum(1 for r in rows if r["status"] not in {"pass", "blocked_true_training_implementation_missing", "blocked"}),
                        "blocked": sum(1 for r in rows if r["status"] in {"blocked_true_training_implementation_missing", "blocked"}),
                    },
                    ensure_ascii=False,
                )
            )
            return 0 if all(r["status"] in {"pass", "blocked_true_training_implementation_missing", "blocked"} for r in rows) else 1
        if args.stage == "invalidate-o2":
            print(json.dumps(invalidate_legacy_o2_proxy(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "summarize":
            print(json.dumps(summarize(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage == "all":
            run_o0(args)
            args.label_prefix = "v22_53O_o1"
            o1 = run_matrix(args, phase="O1")
            args.label_prefix = "v22_53O_o2"
            args.methods = ""
            o2 = run_matrix(args, phase="O2")
            route = summarize(args)
            failed = sum(1 for r in o1 + o2 if r["status"] not in {"pass", "blocked_true_training_implementation_missing", "blocked"})
            print(json.dumps({"failed": failed, **route}, ensure_ascii=False, sort_keys=True))
            return 0 if failed == 0 else 1
    except Exception as exc:
        ensure_out()
        err_path = LOG_ROOT / f"v22_53O_stage_{safe_fragment(args.stage)}_exception.txt"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        append_exec(
            f"{PYTHON} experiments/run_v22_53O_pion_poet_external_oet_baseline.py --stage {args.stage}",
            task_id=f"v22_53O_{args.stage}_exception",
            status="fail",
            gpu=str(args.device),
            exit_code="exception",
            files=str(err_path.relative_to(ROOT)),
            note=repr(exc),
        )
        append_recap("Exception / blocker", f"Stage `{args.stage}` failed with:\n\n```text\n{traceback.format_exc()}\n```")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
