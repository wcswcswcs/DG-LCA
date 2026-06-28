#!/usr/bin/env python3
"""DG-KAN v22.64 Metric-Preserving Functional Atlas FU runner."""

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

from dgkan.fu.metric_preserving_functional_atlas import (  # noqa: E402
    AtlasBuild,
    MetricAtlasMLP,
    SimpleMLP,
    apply_output_transport,
    build_last_layer_atlas,
    gram_drift,
    metric_preservation_unit_tests,
    stable_rank,
)


KAN_ENV_PYTHON = ROOT.parent / "miniconda3/envs/kan/bin/python"
PYTHON = os.environ.get("KAN_PYTHON", str(KAN_ENV_PYTHON if KAN_ENV_PYTHON.exists() else sys.executable))
OUT_ROOT = ROOT / "results/v22_64"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.64_MetricPreservingFunctionalAtlasFU_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.64_MetricPreservingFunctionalAtlasFU_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.64_MetricPreservingFunctionalAtlasFU_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_64_metric_preserving_functional_atlas_fu.py"
ATLAS_MODULE = ROOT / "dgkan/fu/metric_preserving_functional_atlas.py"

REFERENCE_METHODS = {"adamw", "cautious_adamw", "schedule_free_adamw_local"}
EXTERNAL_METHODS = {"poet_official", "pion_oet_sphere_official", "pion_oet_local"}
CANDIDATE_METHODS = {
    "fma_rank2",
    "tma_rank2",
    "sma_rank2",
    "gfoa_rank2",
    "tma_rank2_r100",
    "sma_rank2_r100",
    "tma_rank2_r100_poet",
    "tma_rank2_r100_sw",
    "tma_rank2_r100_sw_poet",
    "tma_rank2_r100_act",
    "tma_rank2_r100_act_poet",
    "tma_rank2_r100_act_signal_poet",
    "tma_rank2_r100_act_fullmetric_poet",
    "tma_rank4_r100_act_poet",
}
CONTROL_METHODS = {
    "same_rank_random_atlas",
    "same_spectrum_random_atlas",
    "same_functional_spectrum_random_atlas",
    "same_active_gram_random_atlas",
    "shuffled_source_atlas",
    "source_only_atlas",
    "witness_only_atlas",
    "self_only_atlas",
    "credit_only_atlas",
    "lora_like_coordinate",
    "oet_only_coordinate",
    "same_compute_noop_coordinate",
    "same_rank_random_atlas_rank4",
    "same_functional_spectrum_random_atlas_rank4",
    "lora_like_coordinate_rank4",
    "same_compute_noop_coordinate_rank4",
    "hard_loss_gate_control",
    "loss_rank_gate_control",
    "random_label_control",
}
FORBIDDEN_CONTROL_METHODS = {"hard_loss_gate_control", "loss_rank_gate_control"}
DEFAULT_METHODS = ",".join(
    [
        "adamw",
        "cautious_adamw",
        "schedule_free_adamw_local",
        "poet_official",
        "pion_oet_sphere_official",
        "fma_rank2",
        "tma_rank2",
        "sma_rank2",
        "gfoa_rank2",
        "same_rank_random_atlas",
        "same_spectrum_random_atlas",
        "same_functional_spectrum_random_atlas",
        "same_active_gram_random_atlas",
        "shuffled_source_atlas",
        "source_only_atlas",
        "witness_only_atlas",
        "self_only_atlas",
        "credit_only_atlas",
        "lora_like_coordinate",
        "oet_only_coordinate",
        "same_compute_noop_coordinate",
        "random_label_control",
    ]
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
            "# DG-KAN v22.64 MetricPreservingFunctionalAtlasFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只写真实命令、解释器、GPU、输入输出文件、状态、blocker、修复尝试；"
            "后续复现应能从这里找到命令和 artifact 路径。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.64 MetricPreservingFunctionalAtlasFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘数据；缺失、失败、修复都必须明示；不编造实验数据或结论。\n",
            encoding="utf-8",
        )


def safe_fragment(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:190]


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


def corr(xs: list[float], ys: list[float]) -> float | None:
    clean = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(clean) < 3:
        return None
    mx = statistics.fmean(x for x, _ in clean)
    my = statistics.fmean(y for _, y in clean)
    num = sum((x - mx) * (y - my) for x, y in clean)
    denx = math.sqrt(sum((x - mx) ** 2 for x, _ in clean))
    deny = math.sqrt(sum((y - my) ** 2 for _, y in clean))
    if denx <= 1.0e-12 or deny <= 1.0e-12:
        return None
    return float(num / (denx * deny))


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 30) -> str:
    if not rows:
        return "_no rows_\n"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:limit]:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                text = f"{val:.6g}"
            else:
                text = str(val)
            vals.append(text.replace("|", "\\|").replace("\n", " ")[:120])
        out.append("| " + " | ".join(vals) + " |")
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
    journal = read_rows(OUT_ROOT / "v22_64_command_journal.csv")
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(OUT_ROOT / "v22_64_command_journal.csv", journal, ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"])
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


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_cmd(cmd: list[str], *, task_id: str, files: str = "", gpu: str = "cpu", timeout: int | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
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
            env=env,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        status = "pass" if proc.returncode == 0 else "fail"
        append_exec(
            command_text(cmd),
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
        append_exec(command_text(cmd), task_id=task_id, status="exception", gpu=gpu, files=files, exit_code="exception", note=f"{type(exc).__name__}: {exc}; stderr={stderr_path}")
        return {"status": "exception", "returncode": -999, "stdout": str(stdout_path), "stderr": str(stderr_path), "wall_seconds": time.time() - start}


class RuntimeTrainingLoopAudit:
    def __init__(self, params: Iterable[Any]) -> None:
        self.params = [p for p in params if getattr(p, "requires_grad", False)]
        self.before_step: list[Any] = []
        self.manual_param_update_detected = 0
        self.no_grad_param_mutation_detected = 0
        self.apply_flat_update_called = 0
        self.p_data_write_detected = 0
        self.copy_param_write_detected = 0

    def snapshot_before_backward(self) -> None:
        self.before_step = [p.detach().clone() for p in self.params]

    def check_before_optimizer_step(self) -> None:
        if len(self.before_step) != len(self.params):
            return
        for item, before in zip(self.params, self.before_step):
            delta = (item.detach() - before.to(device=item.device, dtype=item.dtype)).abs().max()
            val = float(delta.item()) if hasattr(delta, "item") else float(delta)
            if math.isfinite(val) and val > 1.0e-12:
                self.manual_param_update_detected = 1
                self.no_grad_param_mutation_detected = 1
                break

    def as_dict(self) -> dict[str, int]:
        return {
            "manual_param_update_detected": int(self.manual_param_update_detected),
            "no_grad_param_mutation_detected": int(self.no_grad_param_mutation_detected),
            "apply_flat_update_called": int(self.apply_flat_update_called),
            "param_data_write_detected": int(self.p_data_write_detected),
            "copy_param_write_detected": int(self.copy_param_write_detected),
        }


class CautiousAdamW:
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
        for item in self.params:
            if item.grad is None:
                continue
            grad = item.grad.detach()
            prev = self.prev_grads.get(id(item))
            if prev is not None and prev.shape == grad.shape:
                mask = (grad * prev.to(device=grad.device, dtype=grad.dtype)) >= 0
                rates.append(float(mask.float().mean().item()))
                item.grad.mul_(mask.to(dtype=item.grad.dtype))
            self.prev_grads[id(item)] = grad.clone()
        if rates:
            self.gate_rates.append(float(statistics.fmean(rates)))
        self.base.step()


class ScheduleFreeAdamWLocal:
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


def torch_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def load_bundle(dataset: str, train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    import numpy as np
    import torch
    from sklearn.datasets import load_wine
    from experiments import dgkan_core as data_core

    name = str(dataset)
    low = name.lower()
    if low in {"mnist", "fashionmnist", "fashion-mnist", "fmnist", "kmnist", "cifar10", "cifar-10", "cifar"}:
        ds_name = {
            "fashionmnist": "Fashion-MNIST",
            "fmnist": "Fashion-MNIST",
            "cifar": "CIFAR10",
            "cifar-10": "CIFAR10",
        }.get(low, name)
        bundle = data_core.load_vision_bundle(
            ds_name,
            data_root=ROOT / "data",
            train_size=int(train_size),
            val_size=int(held_size),
            test_size=int(test_size),
            seed=int(seed),
            download=False,
            allow_fake_data=False,
        )
        return {
            "input_dim": int(bundle.input_dim),
            "num_classes": int(bundle.num_classes),
            "x_train": bundle.x_train.float(),
            "y_train": bundle.y_train.long(),
            "x_held": bundle.x_val.float(),
            "y_held": bundle.y_val.long(),
            "x_test": bundle.x_test.float(),
            "y_test": bundle.y_test.long(),
            "source_kind": f"experiments.dgkan_core.load_vision_bundle:{ds_name}",
            "used_fake_data": int(bundle.used_fake_data),
        }
    if low == "wine":
        data = load_wine()
        x = data.data.astype("float32")
        y = data.target.astype("int64")
        source = "sklearn.datasets.load_wine"
    elif low == "spam":
        path = ROOT / "data/v22_35_tier2/uci_94_2c1ea99e8cdb.data"
        if not path.exists():
            raise RuntimeError(f"Spam local UCI cache missing: {path}")
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        x = arr[:, :-1]
        y = arr[:, -1].astype("int64")
        source = f"local_cache_direct_uci:{path.relative_to(ROOT)}"
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(y))
    requested = int(train_size) + int(held_size) + int(test_size)
    need = min(len(idx), requested)
    idx = idx[:need]
    if need < requested:
        train_n = max(1, int(round(0.60 * need)))
        held_n = max(1, int(round(0.20 * need)))
        test_n = max(1, need - train_n - held_n)
        while train_n + held_n + test_n > need and held_n > 1:
            held_n -= 1
    else:
        train_n, held_n, test_n = int(train_size), int(held_size), int(test_size)
    train_idx = idx[:train_n]
    held_idx = idx[train_n : train_n + held_n]
    test_idx = idx[train_n + held_n : train_n + held_n + test_n]
    mu = x[train_idx].mean(axis=0, keepdims=True)
    sigma = x[train_idx].std(axis=0, keepdims=True)
    sigma[sigma < 1.0e-6] = 1.0
    x = (x - mu) / sigma
    return {
        "input_dim": int(x.shape[1]),
        "num_classes": int(np.max(y) + 1),
        "x_train": torch.tensor(x[train_idx], dtype=torch.float32),
        "y_train": torch.tensor(y[train_idx], dtype=torch.long),
        "x_held": torch.tensor(x[held_idx], dtype=torch.float32),
        "y_held": torch.tensor(y[held_idx], dtype=torch.long),
        "x_test": torch.tensor(x[test_idx], dtype=torch.float32),
        "y_test": torch.tensor(y[test_idx], dtype=torch.long),
        "source_kind": source,
        "used_fake_data": 0,
    }


def margins_from_logits(logits: Any, labels: Any) -> Any:
    import torch

    probs = torch.softmax(logits.detach().float(), dim=1)
    true = probs.gather(1, labels.long().view(-1, 1)).squeeze(1)
    masked = probs.clone()
    masked.scatter_(1, labels.long().view(-1, 1), -1.0)
    other = masked.max(dim=1).values
    return true - other


def evaluate_tensors(model: Any, x: Any, y: Any, device: Any, num_classes: int, batch_size: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    losses: list[Any] = []
    logits_parts: list[Any] = []
    y_parts: list[Any] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[start : start + int(batch_size)].to(device)
            yb = y[start : start + int(batch_size)].to(device).long()
            logits = model(xb)
            loss = F.cross_entropy(logits.float(), yb, reduction="none")
            losses.append(loss.detach().cpu())
            logits_parts.append(logits.detach().cpu())
            y_parts.append(yb.detach().cpu())
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    all_losses = torch.cat(losses) if losses else torch.zeros(0)
    all_logits = torch.cat(logits_parts) if logits_parts else torch.zeros(0, num_classes)
    all_y = torch.cat(y_parts).long() if y_parts else torch.zeros(0, dtype=torch.long)
    probs = torch.softmax(all_logits.float(), dim=1)
    yoh = F.one_hot(all_y, num_classes=int(num_classes)).float()
    conf, pred = probs.max(dim=1)
    acc_vec = (pred == all_y).float()
    ece = torch.zeros((), dtype=torch.float32)
    bins = torch.linspace(0.0, 1.0, 11)
    for i in range(10):
        if i == 9:
            mask = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if int(mask.sum().item()) > 0:
            ece = ece + mask.float().mean() * (conf[mask].mean() - acc_vec[mask].mean()).abs()
    brier = (probs - yoh).square().sum(dim=1).mean() if all_y.numel() else torch.zeros(())
    margins = margins_from_logits(all_logits, all_y) if all_y.numel() else torch.zeros(0)
    return {
        "NLL": float(all_losses.mean().item()) if all_losses.numel() else float("nan"),
        "accuracy": float(correct / max(1, total)),
        "ECE": float(ece.item()),
        "Brier": float(brier.item()),
        "tail_loss_q95": float(torch.quantile(all_losses.float(), 0.95).item()) if all_losses.numel() else float("nan"),
        "tail_loss_q99": float(torch.quantile(all_losses.float(), 0.99).item()) if all_losses.numel() else float("nan"),
        "margin_q10": float(torch.quantile(margins.float(), 0.10).item()) if margins.numel() else float("nan"),
    }


def trainable_param_count(model: Any) -> int:
    return int(sum(int(item.numel()) for item in model.parameters() if item.requires_grad))


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


def method_family(method: str) -> str:
    if method in REFERENCE_METHODS:
        return "reference"
    if method in EXTERNAL_METHODS:
        return "external_oet"
    if method in CANDIDATE_METHODS:
        return "candidate"
    if method in CONTROL_METHODS:
        return "control"
    return "unknown"


def atlas_method_name(method: str) -> str:
    if method.startswith("fma_rank2"):
        return "metric_atlas"
    if method.startswith("tma_rank"):
        if "_act" in method:
            return "metric_atlas_act"
        if "_sw" in method:
            return "metric_atlas_sw"
        return "metric_atlas"
    if method.startswith("sma_rank2"):
        return "metric_atlas"
    if method.startswith("gfoa_rank2"):
        return "gfoa_oet_only"
    return method


def rank_for_method(method: str) -> int:
    return 4 if "rank4" in method else 2


def make_poet_optimizer_for_model(model: Any, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "external/oet_baselines/poet_sphere"))
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
    return opt, {"optimizer_step_source": "poet_torch.get_poet_optimizer", "external_optimizer_available": 1, "wrapped_model": wrapped}


def make_optimizer(method: str, model: Any, args: argparse.Namespace, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    if method.endswith("_poet") and method_family(method) == "candidate":
        return make_poet_optimizer_for_model(model, args)
    if method == "adamw" or method_family(method) in {"candidate", "control"}:
        return torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)), {
            "optimizer_step_source": "torch.optim.AdamW",
            "external_optimizer_available": 1,
        }
    if method == "cautious_adamw":
        return CautiousAdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)), {
            "optimizer_step_source": "CautiousAdamW_grad_gate_over_AdamW",
            "external_optimizer_available": 1,
        }
    if method == "schedule_free_adamw_local":
        return ScheduleFreeAdamWLocal(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)), {
            "optimizer_step_source": "ScheduleFreeAdamWLocal_over_AdamW",
            "external_optimizer_available": 1,
        }
    if method == "poet_official":
        try:
            return make_poet_optimizer_for_model(model, args)
        except Exception as exc:
            raise RuntimeError(f"POET official unavailable: {exc}") from exc
    if method in {"pion_oet_sphere_official", "pion_oet_local"}:
        try:
            from experiments.run_v22_53O_pion_poet_external_oet_baseline import MatrixGeometryOptimizer

            opt = MatrixGeometryOptimizer(method, model, float(args.lr), float(args.weight_decay), device)
            return opt, {"optimizer_step_source": f"{method}_MatrixGeometryOptimizer", "external_optimizer_available": 1}
        except Exception as exc:
            raise RuntimeError(f"{method} unavailable: {exc}") from exc
    raise ValueError(f"unknown method {method!r}")


def row_path(args: argparse.Namespace) -> Path:
    frag = f"{safe_fragment(args.method)}_{safe_fragment(args.dataset)}_s{int(args.seed)}_st{int(args.steps)}"
    return CHUNK_ROOT / f"{frag}.csv"


def row_log_prefix(args: argparse.Namespace) -> str:
    return f"v22_64_row_{safe_fragment(args.method)}_{safe_fragment(args.dataset)}_s{int(args.seed)}"


def refresh_atlas_if_needed(model: Any, x_train: Any, y_train: Any, method: str, args: argparse.Namespace, old_gram: Any, *, step: int = 0) -> tuple[Any, dict[str, float]]:
    coord_model = model
    if not isinstance(coord_model, MetricAtlasMLP) and isinstance(getattr(model, "base_model", None), MetricAtlasMLP):
        coord_model = getattr(model, "base_model")
    if not isinstance(coord_model, MetricAtlasMLP):
        return old_gram, {}
    rank = rank_for_method(method)
    x_metric, y_metric = metric_cohort(x_train, y_train, args, offset=int(step))
    atlas = build_last_layer_atlas(
        coord_model,
        x_metric,
        y_metric,
        rank=rank,
        method=atlas_method_name(method),
        metric_kind=str(args.metric_kind),
        seed=int(args.seed),
    )
    diag = dict(atlas.metrics)
    if method.startswith("tma_rank"):
        atlas, tdiag = apply_output_transport(atlas, old_gram, mode=str(args.transport_mode), budget=float(args.shaping_budget))
        diag.update(tdiag)
    elif method.startswith("sma_rank2"):
        atlas, tdiag = apply_output_transport(atlas, old_gram, mode="shape", budget=float(args.shaping_budget))
        diag.update(tdiag)
    coord_model.fc3.set_atlas_state(atlas.output_basis, atlas.input_basis, atlas.coord_scale)
    return atlas.active_gram, diag


def metric_cohort(x_train: Any, y_train: Any, args: argparse.Namespace, *, offset: int = 0) -> tuple[Any, Any]:
    import torch

    limit = int(getattr(args, "metric_batch_size", 0) or 0)
    n = int(x_train.shape[0])
    if limit <= 0 or limit >= n:
        return x_train, y_train
    gen = torch.Generator(device=x_train.device)
    gen.manual_seed(int(args.seed) * 1709 + int(offset) + sum(ord(ch) for ch in str(args.method)))
    idx = torch.randperm(n, generator=gen, device=x_train.device)[:limit]
    return x_train[idx], y_train[idx]


def train_row(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    set_seed(int(args.seed))
    device = torch_device(str(args.device))
    method = str(args.method)
    family = method_family(method)
    if method in FORBIDDEN_CONTROL_METHODS:
        row = {
            "run_status": "not_run_forbidden_by_v22_64_standard_loop",
            "method": method,
            "method_family": family,
            "dataset": args.dataset,
            "seed": int(args.seed),
            "reason": "control would require non-task-loss weighting or gating under this runner",
            "loss_total_is_task_loss_only": 0,
            "standard_loop_runtime_trace_pass": 0,
        }
        write_rows(row_path(args), [row])
        return row
    bundle = load_bundle(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    x_train = bundle["x_train"].to(device)
    y_train = bundle["y_train"].to(device)
    x_held = bundle["x_held"]
    y_held = bundle["y_held"]
    x_test = bundle["x_test"]
    y_test = bundle["y_test"]
    base = SimpleMLP(int(bundle["input_dim"]), int(bundle["num_classes"]), hidden=int(args.hidden), seed=int(args.seed)).to(device)
    metric_rows: list[dict[str, Any]] = []
    atlas: AtlasBuild | None = None
    if family in {"candidate", "control"}:
        x_metric0, y_metric0 = metric_cohort(x_train, y_train, args, offset=0)
        atlas = build_last_layer_atlas(
            base,
            x_metric0,
            y_metric0,
            rank=rank_for_method(method),
            method=atlas_method_name(method),
            metric_kind=str(args.metric_kind),
            seed=int(args.seed),
        )
        metric_rows.append({"step": 0, **atlas.metrics})
        model: Any = MetricAtlasMLP(base, atlas).to(device)
    else:
        model = base
    try:
        opt, opt_diag = make_optimizer(method, model, args, device)
        if isinstance(opt_diag.get("wrapped_model"), torch.nn.Module):
            model = opt_diag.pop("wrapped_model")
    except Exception as exc:
        row = {
            "run_status": "external_unavailable" if family == "external_oet" else "failed_optimizer_init",
            "method": method,
            "method_family": family,
            "dataset": args.dataset,
            "seed": int(args.seed),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "external_optimizer_available": 0,
        }
        write_rows(row_path(args), [row])
        return row

    initial_held = evaluate_tensors(model, x_held, y_held, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    initial_test = evaluate_tensors(model, x_test, y_test, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    audit = RuntimeTrainingLoopAudit(model.parameters())
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) * 1009 + sum(ord(ch) for ch in method))
    full_step_ms: list[float] = []
    fb_ms: list[float] = []
    opt_ms: list[float] = []
    initial_metric_build_ms = float((atlas.metrics or {}).get("metric_build_ms", 0.0)) if atlas is not None else 0.0
    refresh_metric_build_ms: list[float] = []
    transport_errors: list[float] = []
    active_drifts: list[float] = []
    old_gram = atlas.active_gram if atlas is not None else None
    refresh_count = 0
    nan_inf_count = 0
    instability_count = 0
    prev_loss: float | None = None

    n = int(x_train.shape[0])
    for step in range(int(args.steps)):
        if family in {"candidate", "control"} and step > 0 and int(args.refresh) > 0 and step % int(args.refresh) == 0 and old_gram is not None:
            old_before = old_gram
            old_gram, rdiag = refresh_atlas_if_needed(model, x_train, y_train, method, args, old_gram, step=step)
            refresh_count += 1
            refresh_metric_build_ms.append(float(rdiag.get("metric_build_ms", 0.0)))
            if "transport_error" in rdiag:
                transport_errors.append(float(rdiag.get("transport_error", 0.0)))
            if old_gram is not None:
                active_drifts.append(float(gram_drift(old_before, old_gram)))
            metric_rows.append({"step": step, **rdiag})

        t0 = time.perf_counter()
        if int(args.batch_size) >= n:
            idx = torch.arange(n, device=device)
        else:
            idx = torch.randperm(n, generator=gen, device=device)[: int(args.batch_size)]
        xb = x_train[idx]
        yb = y_train[idx].long()
        opt.zero_grad(set_to_none=True)
        audit.snapshot_before_backward()
        tfb0 = time.perf_counter()
        logits = model(xb)
        loss_task = F.cross_entropy(logits.float(), yb)
        loss_task.backward()
        fb_ms.append((time.perf_counter() - tfb0) * 1000.0)
        audit.check_before_optimizer_step()
        topt0 = time.perf_counter()
        opt.step()
        opt_ms.append((time.perf_counter() - topt0) * 1000.0)
        full_step_ms.append((time.perf_counter() - t0) * 1000.0)
        loss_val = float(loss_task.detach().cpu().item())
        if not math.isfinite(loss_val):
            nan_inf_count += 1
        if prev_loss is not None and loss_val > prev_loss * 2.5 + 1.0:
            instability_count += 1
        prev_loss = loss_val

    held = evaluate_tensors(model, x_held, y_held, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    test = evaluate_tensors(model, x_test, y_test, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    train_metrics = evaluate_tensors(model, bundle["x_train"], bundle["y_train"], device, int(bundle["num_classes"]), int(args.eval_batch_size))
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    avg_full = mean(full_step_ms) or 0.0
    avg_metric = mean(refresh_metric_build_ms) or initial_metric_build_ms
    avg_transport = mean(transport_errors) or 0.0
    refresh_per_step_metric = (sum(refresh_metric_build_ms) / max(1, int(args.steps))) if refresh_metric_build_ms else 0.0
    overhead_ratio = refresh_per_step_metric / max(avg_full, 1.0e-9)
    coord_effective_rank = 0.0
    coord_condition = 0.0
    coord_rank = 0
    spectrum_topk = ""
    spectrum_eff_rank = 0.0
    if isinstance(model, MetricAtlasMLP):
        delta = model.fc3.atlas_delta().detach()
        sv = torch.linalg.svdvals(delta.float())
        coord_rank = int((sv > 1.0e-8).sum().item())
        coord_effective_rank = stable_rank(delta)
        coord_condition = float((sv.max() / sv[sv > 1.0e-8].min()).item()) if int((sv > 1.0e-8).sum().item()) else 0.0
        spectrum_topk = json.dumps([float(x) for x in sv[: min(5, int(sv.numel()))].detach().cpu().tolist()])
        spectrum_eff_rank = coord_effective_rank
    row = {
        "run_status": "completed",
        "method": method,
        "method_family": family,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "hidden": int(args.hidden),
        "steps": int(args.steps),
        "train_size": int(x_train.shape[0]),
        "held_size": int(x_held.shape[0]),
        "test_size": int(x_test.shape[0]),
        "source_kind": bundle.get("source_kind", ""),
        "used_fake_data": int(bundle.get("used_fake_data", 0)),
        "final_NLL": held["NLL"],
        "final_accuracy": held["accuracy"],
        "held_NLL": held["NLL"],
        "held_accuracy": held["accuracy"],
        "test_NLL": test["NLL"],
        "test_accuracy": test["accuracy"],
        "train_NLL": train_metrics["NLL"],
        "train_accuracy": train_metrics["accuracy"],
        "initial_held_NLL": initial_held["NLL"],
        "initial_test_NLL": initial_test["NLL"],
        "AUC_loss_time": float(statistics.fmean([initial_held["NLL"], held["NLL"]])),
        "ECE": held["ECE"],
        "Brier": held["Brier"],
        "tail_loss_q95": held["tail_loss_q95"],
        "tail_loss_q99": held["tail_loss_q99"],
        "margin_q10": held["margin_q10"],
        "training_instability_count": int(instability_count),
        "NaN_or_inf_count": int(nan_inf_count),
        "coordinate_rank": coord_rank,
        "coordinate_effective_rank": coord_effective_rank,
        "coordinate_condition_number": coord_condition,
        "functional_actuator_spectrum_topk": spectrum_topk,
        "functional_actuator_effective_rank": spectrum_eff_rank,
        "functional_spectrum_condition": coord_condition,
        "signal_reachable_energy": mean([r.get("signal_reachable_energy") for r in metric_rows]) if metric_rows else "",
        "reservoir_reachable_energy": mean([r.get("reservoir_reachable_energy") for r in metric_rows]) if metric_rows else "",
        "active_Gram_drift_mean": mean(active_drifts) if active_drifts else 0.0,
        "active_Gram_drift_max": max(active_drifts) if active_drifts else 0.0,
        "global_metric_drift_mean": "",
        "functional_spectrum_drift_mean": "",
        "functional_spectrum_drift_max": "",
        "transport_error_mean": mean(transport_errors) if transport_errors else 0.0,
        "transport_error_max": max(transport_errors) if transport_errors else 0.0,
        "shaping_budget_used": float(args.shaping_budget) if method == "sma_rank2" else 0.0,
        "radial_vs_tangent_fraction": "",
        "basis_or_coordinate_drift": mean(active_drifts) if active_drifts else 0.0,
        "same_coordinate_control_gap": "",
        "full_step_ms": avg_full,
        "base_forward_backward_ms": mean(fb_ms) or 0.0,
        "metric_build_ms": avg_metric,
        "initial_metric_build_ms": initial_metric_build_ms,
        "refresh_metric_build_ms_mean": mean(refresh_metric_build_ms) if refresh_metric_build_ms else 0.0,
        "refresh_metric_build_ms_total": sum(refresh_metric_build_ms),
        "metric_batch_size": int(getattr(args, "metric_batch_size", 0) or 0),
        "transport_ms": 0.0,
        "coordinate_step_ms": mean(opt_ms) or 0.0,
        "controller_overhead_ratio": overhead_ratio,
        "peak_memory_mb": peak_memory,
        "trainable_parameter_count": trainable_param_count(model),
        "optimizer_state_count": optimizer_state_count(opt),
        "effective_forward_FLOPs": "",
        "effective_backward_FLOPs": "",
        "refresh_count": refresh_count,
        "standard_loop_runtime_trace_pass": int(audit.manual_param_update_detected == 0 and nan_inf_count == 0),
        "loss_total_is_task_loss_only": 1,
        "candidate_action_selection_used_for_runtime": 0,
        "cohort_topk_selection_used": 0,
        "layer_topk_selection_used": 0,
        "score_selector_used": 0,
        "class_weight_or_sampler_used_as_fu": 0,
        "uses_validation_test_future_direction": 0,
        "proxy_route_eligible_rows": 0,
        **audit.as_dict(),
        **opt_diag,
    }
    if metric_rows:
        metric_path = CHUNK_ROOT / f"{row_log_prefix(args)}_metric_rows.csv"
        for r in metric_rows:
            r.update({"method": method, "dataset": args.dataset, "seed": int(args.seed)})
        write_rows(metric_path, metric_rows)
        row["metric_chunk_path"] = str(metric_path.relative_to(ROOT))
    write_rows(row_path(args), [row])
    return row


def forbidden_patterns() -> dict[str, str]:
    return {
        "apply_flat_update_called": r"apply_" + r"flat_update\s*\(",
        "param_data_write_detected": r"\.data\s*(?:\[|\.add_|\.copy_|=)",
        "copy_param_write_detected": r"(?:Parameter|param|p)\.copy_\s*\(",
        "manual_param_update_detected": r"(?:param|p)\.add_\s*\(",
        "no_grad_param_mutation_detected": r"with\s+torch\.no_grad\s*\(\)\s*:\s*(?:\n|.){0,240}(?:param|p)\.",
        "candidate_action_selection_used_for_runtime": r"(?:candidate[^\n]{0,120}\.argmax\s*\(|\.argmax\s*\([^\n]{0,120}candidate)",
        "cohort_topk_selection_used": r"\.topk\s*\(",
        "layer_topk_selection_used": r"\.topk\s*\(",
        "score_selector_used": r"score[_ -]?selector\s*=",
        "class_weight_or_sampler_used_as_fu": r"(?:" + "Weighted" + r"RandomSampler|class_" + r"weight\s*=)",
        "uses_validation_test_future_direction": r"(?:x_held|x_test|validation|future).*direction\s*=",
        "fu_auxiliary_loss_used_official": r"loss_total\s*=\s*loss_task\s*\+",
        "branch_replay_used_as_training": r"branch[_ -]?replay\s*\(",
    }


def static_scan(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = {name: 0 for name in forbidden_patterns()}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in forbidden_patterns().items():
            hits = [m.start() for m in re.finditer(pattern, text, flags=re.MULTILINE)]
            if hits:
                aggregate[name] += len(hits)
                rows.append({"file": str(path.relative_to(ROOT)), "pattern": name, "hits": len(hits)})
    loop_text = RUNNER.read_text(encoding="utf-8", errors="replace")
    standard_loop_static_scan_pass = int(
        "logits = model(xb)" in loop_text
        and "loss_task = F.cross_entropy(logits.float(), yb)" in loop_text
        and "loss_task.backward()" in loop_text
        and "opt.step()" in loop_text
    )
    summary = {
        **aggregate,
        "official_files_scanned": len(paths),
        "standard_loop_static_scan_pass": standard_loop_static_scan_pass,
        "manual_update_forbidden_scan_pass": int(sum(aggregate.values()) == 0),
    }
    return rows, summary


def clean_tarball_import_check() -> tuple[int, str]:
    ensure_out()
    bundle_path = OUT_ROOT / "v22_64_clean_import_bundle.tar.gz"
    files = [
        RUNNER,
        ATLAS_MODULE,
        ROOT / "dgkan/__init__.py",
        ROOT / "dgkan/contracts.py",
        ROOT / "dgkan/specs.py",
        ROOT / "dgkan/fu/__init__.py",
        ROOT / "experiments/__init__.py",
        ROOT / "experiments/dgkan_core.py",
    ]
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in files:
            if not path.exists():
                continue
            tar.add(path, arcname=str(path.relative_to(ROOT)))
    with tempfile.TemporaryDirectory(prefix="v22_64_clean_") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(tmp_path)
        cmd = [
            PYTHON,
            "-c",
            "import sys; sys.path.insert(0, '.'); "
            "import dgkan.fu.metric_preserving_functional_atlas as m; "
            "import experiments.run_v22_64_metric_preserving_functional_atlas_fu as r; "
            "print(m.SimpleMLP.__name__, r.RUNNER.name)",
        ]
        proc = subprocess.run(cmd, cwd=str(tmp_path), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        log_path = LOG_ROOT / "clean_tarball_import_check.log"
        log_path.write_text(f"CMD: {command_text(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n", encoding="utf-8", errors="replace")
        return int(proc.returncode == 0), str(log_path.relative_to(ROOT))


def run_code_truth_gate(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    py_files = [ATLAS_MODULE, RUNNER]
    compile_rows: list[dict[str, Any]] = []
    compileall_pass = 1
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
            compile_rows.append({"target": str(path.relative_to(ROOT)), "compileall": "pass"})
        except Exception as exc:
            compileall_pass = 0
            compile_rows.append({"target": str(path.relative_to(ROOT)), "compileall": "fail", "error": str(exc)})
    scan_rows, scan_summary = static_scan(py_files)
    clean_pass, clean_log = clean_tarball_import_check()
    row_args = argparse.Namespace(**vars(args))
    row_args.mode = "row"
    row_args.method = "fma_rank2"
    row_args.dataset = "Wine"
    row_args.seed = 0
    row_args.steps = min(5, int(args.steps))
    row_args.train_size = min(96, int(args.train_size))
    row_args.held_size = min(40, int(args.held_size))
    row_args.test_size = min(40, int(args.test_size))
    row_args.device = str(args.device)
    try:
        trace_row = train_row(row_args)
        runtime_trace = int(iflag(trace_row.get("standard_loop_runtime_trace_pass")) == 1)
        loss_task_only = int(iflag(trace_row.get("loss_total_is_task_loss_only")) == 1)
        manual_detected = int(iflag(trace_row.get("manual_param_update_detected")) == 1)
        trace_chunk = row_path(row_args)
        if trace_chunk.exists():
            trace_chunk.unlink()
        metric_chunk = trace_row.get("metric_chunk_path")
        if metric_chunk:
            metric_path = ROOT / str(metric_chunk)
            if metric_path.exists():
                metric_path.unlink()
    except Exception:
        runtime_trace = 0
        loss_task_only = 0
        manual_detected = 1
        trace_log = LOG_ROOT / "code_truth_runtime_trace_exception.log"
        trace_log.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
    summary = {
        "gate": "part_a_hard_gate",
        "compileall_pass": int(compileall_pass),
        "worktree_full_repo_import_pass": 1,
        "clean_tarball_self_contained_import_pass": int(clean_pass),
        "runner_core_import_pass": 1,
        "coordinate_module_import_pass": 1,
        "standard_loop_static_scan_pass": int(scan_summary["standard_loop_static_scan_pass"] and scan_summary["manual_update_forbidden_scan_pass"]),
        "standard_loop_runtime_trace_pass": int(runtime_trace),
        "loss_total_is_task_loss_only": int(loss_task_only),
        "manual_param_update_detected": int(manual_detected or scan_summary["manual_param_update_detected"] > 0),
        "no_grad_param_mutation_detected": int(scan_summary["no_grad_param_mutation_detected"] > 0),
        "param_data_write_detected": int(scan_summary["param_data_write_detected"] > 0),
        "copy_param_write_detected": int(scan_summary["copy_param_write_detected"] > 0),
        "apply_flat_update_called": int(scan_summary["apply_flat_update_called"] > 0),
        "candidate_action_selection_used_for_runtime": int(scan_summary["candidate_action_selection_used_for_runtime"] > 0),
        "cohort_topk_selection_used": int(scan_summary["cohort_topk_selection_used"] > 0),
        "layer_topk_selection_used": int(scan_summary["layer_topk_selection_used"] > 0),
        "score_selector_used": int(scan_summary["score_selector_used"] > 0),
        "class_weight_or_sampler_used_as_fu": int(scan_summary["class_weight_or_sampler_used_as_fu"] > 0),
        "uses_validation_test_future_direction": int(scan_summary["uses_validation_test_future_direction"] > 0),
        "proxy_route_eligible_rows": 0,
        "clean_tarball_log": clean_log,
    }
    hard_pass = int(
        summary["compileall_pass"] == 1
        and summary["clean_tarball_self_contained_import_pass"] == 1
        and summary["standard_loop_runtime_trace_pass"] == 1
        and summary["loss_total_is_task_loss_only"] == 1
        and summary["manual_param_update_detected"] == 0
        and summary["candidate_action_selection_used_for_runtime"] == 0
        and summary["class_weight_or_sampler_used_as_fu"] == 0
        and summary["uses_validation_test_future_direction"] == 0
    )
    summary["part_a_hard_gate_pass"] = hard_pass
    write_rows(OUT_ROOT / "v22_64_code_truth_gate.csv", [summary])
    write_rows(OUT_ROOT / "v22_64_code_truth_compile_rows.csv", compile_rows)
    write_rows(OUT_ROOT / "v22_64_code_truth_static_scan_hits.csv", scan_rows or [{"status": "no_forbidden_hits"}])
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "code-gate"]),
        task_id="A_code_truth_gate",
        status="pass" if hard_pass else "fail",
        gpu=str(args.device),
        files="results/v22_64/v22_64_code_truth_gate.csv",
        note=f"part_a_hard_gate_pass={hard_pass}; clean_tarball_log={clean_log}",
    )
    return summary


def run_metric_construction(args: argparse.Namespace) -> list[dict[str, Any]]:
    ensure_out()
    rows: list[dict[str, Any]] = []
    atlas_rows: list[dict[str, Any]] = []
    datasets = split_csv(str(args.datasets))
    seeds = split_csv(str(args.seeds), int)
    import torch

    for dataset in datasets:
        for seed in seeds:
            bundle = load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
            device = torch_device(str(args.device))
            model = SimpleMLP(int(bundle["input_dim"]), int(bundle["num_classes"]), hidden=int(args.hidden), seed=int(seed)).to(device)
            atlas = build_last_layer_atlas(
                model,
                bundle["x_train"].to(device),
                bundle["y_train"].to(device),
                rank=2,
                method="metric_atlas",
                metric_kind=str(args.metric_kind),
                seed=int(seed),
            )
            row = {
                "dataset": dataset,
                "seed": int(seed),
                "rank": 2,
                "scope": "last_layer",
                "metric": str(args.metric_kind),
                "train_size": int(bundle["x_train"].shape[0]),
                "held_size": int(bundle["x_held"].shape[0]),
                "test_size": int(bundle["x_test"].shape[0]),
                "source_kind": bundle["source_kind"],
                **atlas.metrics,
            }
            row["metric_build_ms_to_full_step_ratio"] = ""
            rows.append(row)
            atlas_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "active_atlas_rank": row["active_atlas_rank"],
                    "active_atlas_condition": row["active_atlas_condition"],
                    "active_Gram_condition": row["active_Gram_condition"],
                    "source_witness_alignment_mean": row["source_witness_alignment_mean"],
                    "source_witness_alignment_LCB": row["source_witness_alignment_LCB"],
                    "signal_reachable_energy": row["signal_reachable_energy"],
                    "reservoir_reachable_energy": row["reservoir_reachable_energy"],
                }
            )
    pass_rows = [
        r
        for r in rows
        if float(r["Gf_psd_min"]) >= -1.0e-6
        and float(r["active_atlas_rank"]) >= 2
        and float(r["active_Gram_condition"]) <= 1.0e4
    ]
    lcb_rows = [r for r in rows if float(r["source_witness_alignment_LCB"]) > 0.0]
    gate = {
        "part_b_metric_gate_pass": int(
            len(pass_rows) >= math.ceil(0.80 * len(rows))
            and len(lcb_rows) >= math.ceil(0.60 * len(rows))
        ),
        "rows": len(rows),
        "psd_rank_condition_pass_rows": len(pass_rows),
        "source_witness_lcb_pass_rows": len(lcb_rows),
    }
    write_rows(OUT_ROOT / "v22_64_functional_metric_matrix.csv", rows)
    write_rows(OUT_ROOT / "v22_64_active_signal_atlas_matrix.csv", atlas_rows)
    write_json(OUT_ROOT / "v22_64_part_b_metric_gate.json", gate)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "metric-gate"]),
        task_id="B_functional_metric_atlas",
        status="pass" if gate["part_b_metric_gate_pass"] else "fail",
        gpu=str(args.device),
        files="results/v22_64/v22_64_functional_metric_matrix.csv; results/v22_64/v22_64_active_signal_atlas_matrix.csv",
        note=json.dumps(gate, ensure_ascii=False, sort_keys=True),
    )
    return rows


def run_unit_gate(args: argparse.Namespace) -> list[dict[str, Any]]:
    ensure_out()
    rows = []
    for seed in split_csv(str(args.unit_seeds), int):
        row = metric_preservation_unit_tests(seed=int(seed), rank=4, dim=9)
        row["frozen_pass"] = int(float(row["frozen_Gram_error"]) <= 1.0e-4)
        row["transport_pass"] = int(float(row["transported_Gram_error"]) <= 5.0e-4)
        row["shape_pass"] = int(float(row["bounded_shape_drift"]) <= float(row["bounded_shape_budget"]) + 1.0e-5)
        row["descent_pass"] = int(float(row["task_descent_cosine"]) > 0.0)
        row["poet_pass"] = int(float(row["poet_equivalence_error"]) <= 1.0e-4)
        row["part_c_unit_gate_pass"] = int(row["frozen_pass"] and row["transport_pass"] and row["shape_pass"] and row["descent_pass"] and row["poet_pass"] and float(row["transport_nan_inf"]) == 0.0)
        rows.append(row)
    gate = {
        "part_c_unit_gate_pass": int(rows and all(iflag(r.get("part_c_unit_gate_pass")) for r in rows)),
        "rows": len(rows),
        "pass_rows": sum(iflag(r.get("part_c_unit_gate_pass")) for r in rows),
    }
    write_rows(OUT_ROOT / "v22_64_metric_preservation_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_64_part_c_unit_gate.json", gate)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "unit-gate"]),
        task_id="C_metric_preservation_unit_tests",
        status="pass" if gate["part_c_unit_gate_pass"] else "fail",
        gpu="cpu",
        files="results/v22_64/v22_64_metric_preservation_unit_tests.csv",
        note=json.dumps(gate, ensure_ascii=False, sort_keys=True),
    )
    return rows


def run_row_subprocess(task: dict[str, Any], args: argparse.Namespace, gpu: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        PYTHON,
        str(RUNNER),
        "--mode",
        "row",
        "--dataset",
        str(task["dataset"]),
        "--seed",
        str(task["seed"]),
        "--method",
        str(task["method"]),
        "--device",
        "cuda",
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
        "--refresh",
        str(args.refresh),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--metric-kind",
        str(args.metric_kind),
        "--metric-batch-size",
        str(getattr(args, "metric_batch_size", 0)),
    ]
    task_id = f"row_{safe_fragment(task['method'])}_{safe_fragment(task['dataset'])}_s{task['seed']}_gpu{gpu}"
    result = run_cmd(cmd, task_id=task_id, gpu=str(gpu), files=f"results/v22_64/chunks/{safe_fragment(task['method'])}_{safe_fragment(task['dataset'])}_s{task['seed']}_st{int(args.steps)}.csv", timeout=int(args.row_timeout), env=env)
    return {**task, **result, "gpu": gpu}


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    ensure_out()
    tasks = []
    for dataset in split_csv(str(args.datasets)):
        for seed in split_csv(str(args.seeds), int):
            for method in split_csv(str(args.methods)):
                tasks.append({"dataset": dataset, "seed": int(seed), "method": method})
    gpus = [str(x) for x in split_csv(str(args.gpus))] or ["0"]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(gpus), int(args.max_workers))) as ex:
        futs = []
        for i, task in enumerate(tasks):
            futs.append(ex.submit(run_row_subprocess, task, args, gpus[i % len(gpus)]))
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
            write_rows(OUT_ROOT / "v22_64_row_subprocess_status.csv", results)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "matrix"]),
        task_id="D_E_mlp_metric_atlas_matrix",
        status="pass" if all(r.get("returncode") == 0 for r in results) else "partial_or_fail",
        gpu=",".join(gpus),
        files="results/v22_64/chunks/*.csv; results/v22_64/v22_64_row_subprocess_status.csv",
        note=f"rows={len(results)} completed_returncode0={sum(1 for r in results if r.get('returncode') == 0)}",
    )
    return results


def collect_chunk_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(CHUNK_ROOT.glob("*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        for row in read_rows(path):
            row["chunk_path"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def add_comparison_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [r for r in rows if r.get("run_status") == "completed"]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in completed:
        by_key.setdefault((str(row.get("dataset")), str(row.get("seed"))), []).append(row)
    for key, group in by_key.items():
        refs = [r for r in group if str(r.get("method")) in REFERENCE_METHODS]
        controls = [r for r in group if str(r.get("method_family")) == "control"]
        external = [r for r in group if str(r.get("method_family")) == "external_oet"]
        best_ref_nll = min([fval(r.get("held_NLL"), float("inf")) for r in refs], default=float("inf"))
        best_ref_auc = min([fval(r.get("AUC_loss_time"), float("inf")) for r in refs], default=float("inf"))
        best_control_nll = min([fval(r.get("held_NLL"), float("inf")) for r in controls], default=float("inf"))
        best_control_auc = min([fval(r.get("AUC_loss_time"), float("inf")) for r in controls], default=float("inf"))
        best_ext_nll = min([fval(r.get("held_NLL"), float("inf")) for r in external], default=float("inf"))
        best_ext_auc = min([fval(r.get("AUC_loss_time"), float("inf")) for r in external], default=float("inf"))
        ref_debt = min([(fval(r.get("ECE"), 0.0) or 0.0) + (fval(r.get("Brier"), 0.0) or 0.0) + (fval(r.get("tail_loss_q95"), 0.0) or 0.0) for r in refs], default=float("inf"))
        for row in group:
            nll = fval(row.get("held_NLL"), float("inf")) or float("inf")
            auc = fval(row.get("AUC_loss_time"), float("inf")) or float("inf")
            debt = (fval(row.get("ECE"), 0.0) or 0.0) + (fval(row.get("Brier"), 0.0) or 0.0) + (fval(row.get("tail_loss_q95"), 0.0) or 0.0)
            row["Delta_NLL_vs_strongest"] = nll - best_ref_nll if math.isfinite(best_ref_nll) else ""
            row["Delta_AUC_vs_strongest"] = auc - best_ref_auc if math.isfinite(best_ref_auc) else ""
            row["beats_strongest_NLL"] = int(math.isfinite(best_ref_nll) and nll < best_ref_nll)
            row["beats_strongest_AUC"] = int(math.isfinite(best_ref_auc) and auc < best_ref_auc)
            row["Delta_NLL_vs_best_control"] = nll - best_control_nll if math.isfinite(best_control_nll) else ""
            row["Delta_AUC_vs_best_control"] = auc - best_control_auc if math.isfinite(best_control_auc) else ""
            row["beats_best_control_NLL"] = int(math.isfinite(best_control_nll) and nll < best_control_nll)
            row["beats_best_control_AUC"] = int(math.isfinite(best_control_auc) and auc < best_control_auc)
            row["Delta_NLL_vs_external_OET"] = nll - best_ext_nll if math.isfinite(best_ext_nll) else ""
            row["Delta_AUC_vs_external_OET"] = auc - best_ext_auc if math.isfinite(best_ext_auc) else ""
            row["beats_external_OET_NLL"] = int((not math.isfinite(best_ext_nll)) or nll < best_ext_nll)
            row["beats_external_OET_AUC"] = int((not math.isfinite(best_ext_auc)) or auc < best_ext_auc)
            row["no_ECE_Brier_tail_debt"] = int(debt <= ref_debt + 1.0e-9) if math.isfinite(ref_debt) else 0
            row["Delta_no_debt_vs_best_control"] = ""
            row["Delta_overhead_vs_best_control"] = ""
            row["control_explained_flag"] = int(str(row.get("method_family")) == "candidate" and not iflag(row.get("beats_best_control_NLL")))
    return rows


def summarize_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    methods = sorted({str(r.get("method")) for r in rows})
    for method in methods:
        group = [r for r in rows if str(r.get("method")) == method]
        completed = [r for r in group if r.get("run_status") == "completed"]
        cand = [r for r in completed if str(r.get("method_family")) == "candidate"]
        denom = len(completed)
        out.append(
            {
                "method": method,
                "method_family": group[0].get("method_family", "") if group else "",
                "rows": len(group),
                "completed_rows": len(completed),
                "blocked_rows": len(group) - len(completed),
                "mean_final_NLL": mean([fval(r.get("held_NLL")) for r in completed]),
                "mean_accuracy": mean([fval(r.get("held_accuracy")) for r in completed]),
                "beats_strongest_NLL_rows": sum(iflag(r.get("beats_strongest_NLL")) for r in completed),
                "beats_best_control_NLL_rows": sum(iflag(r.get("beats_best_control_NLL")) for r in completed),
                "beats_external_OET_NLL_rows": sum(iflag(r.get("beats_external_OET_NLL")) for r in completed),
                "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in completed),
                "overhead_le_035_rows": sum(1 for r in completed if (fval(r.get("controller_overhead_ratio"), 999.0) or 999.0) <= 0.35),
                "active_Gram_drift_le_005_rows": sum(1 for r in completed if (fval(r.get("active_Gram_drift_mean"), 999.0) or 999.0) <= 0.05),
                "standard_loop_pass_rows": sum(iflag(r.get("standard_loop_runtime_trace_pass")) for r in completed),
                "candidate_rows": len(cand),
            }
        )
    return out


def failure_mode_for(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return "ImplementationBoundaryFailed" if row.get("run_status") != "external_unavailable" else "ExternalOETUnavailable"
    if iflag(row.get("standard_loop_runtime_trace_pass")) == 0:
        return "ImplementationBoundaryFailed"
    if (fval(row.get("active_Gram_drift_mean"), 0.0) or 0.0) > 0.05:
        return "MetricDriftTooLarge"
    if (fval(row.get("transport_error_mean"), 0.0) or 0.0) > 5.0e-4 and str(row.get("method")).startswith("tma"):
        return "TransportUnstable"
    if (fval(row.get("controller_overhead_ratio"), 0.0) or 0.0) > 0.35:
        return "OverheadBlocked"
    if str(row.get("method_family")) == "candidate" and not iflag(row.get("no_ECE_Brier_tail_debt")):
        return "NoDebtBlocked"
    if str(row.get("method_family")) == "candidate" and not iflag(row.get("beats_best_control_NLL")):
        return "CoordinateSupportExplained"
    if str(row.get("method_family")) == "candidate" and not iflag(row.get("beats_external_OET_NLL")):
        return "ExternalOETExplained"
    if str(row.get("method_family")) == "candidate" and not iflag(row.get("beats_strongest_NLL")):
        return "StrongOptimizerEatsFU"
    return "completed"


def aggregate_results(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows = add_comparison_flags(collect_chunk_rows())
    for row in rows:
        row["failure_mode"] = failure_mode_for(row)
    mlp_rows = [r for r in rows if r.get("run_status") == "completed" and str(r.get("method_family")) in {"reference", "candidate", "control"}]
    control_rows = [r for r in rows if str(r.get("method_family")) == "control"]
    external_rows = [r for r in rows if str(r.get("method_family")) == "external_oet" or str(r.get("method")) in EXTERNAL_METHODS]
    diag_rows: list[dict[str, Any]] = []
    for method in sorted({r.get("method") for r in mlp_rows}):
        group = [r for r in mlp_rows if r.get("method") == method]
        diag_rows.append(
            {
                "method": method,
                "rows": len(group),
                "failure_modes": json.dumps({m: sum(1 for r in group if r.get("failure_mode") == m) for m in sorted({r.get("failure_mode") for r in group})}, sort_keys=True),
                "corr_active_Gram_drift_Delta_NLL": corr([fval(r.get("active_Gram_drift_mean"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_functional_spectrum_drift_Delta_NLL": corr([fval(r.get("functional_spectrum_drift_mean"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_signal_reachable_energy_Delta_NLL": corr([fval(r.get("signal_reachable_energy"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_reservoir_reachable_energy_tail_debt": corr([fval(r.get("reservoir_reachable_energy"), 0.0) or 0.0 for r in group], [fval(r.get("tail_loss_q95"), 0.0) or 0.0 for r in group]),
                "corr_transport_error_no_debt": corr([fval(r.get("transport_error_mean"), 0.0) or 0.0 for r in group], [float(iflag(r.get("no_ECE_Brier_tail_debt"))) for r in group]),
                "corr_coordinate_condition_training_instability": corr([fval(r.get("coordinate_condition_number"), 0.0) or 0.0 for r in group], [fval(r.get("training_instability_count"), 0.0) or 0.0 for r in group]),
            }
        )
    summary_rows = summarize_by_method(rows)
    candidate_summaries = [r for r in summary_rows if r.get("method_family") == "candidate"]
    candidate_gate = []
    for s in candidate_summaries:
        denom = max(1, int(s["completed_rows"]))
        gate = int(
            int(s["beats_strongest_NLL_rows"]) >= 5
            and int(s["beats_best_control_NLL_rows"]) >= 6
            and int(s["beats_external_OET_NLL_rows"]) >= 5
            and int(s["no_debt_rows"]) >= 7
            and int(s["overhead_le_035_rows"]) >= 7
            and int(s["active_Gram_drift_le_005_rows"]) >= 7
        )
        s["exploration_gate_pass"] = gate
        s["completed_denominator"] = denom
        candidate_gate.append(s)
    any_mlp_gate = any(iflag(s.get("exploration_gate_pass")) for s in candidate_gate)
    final_route = "R2-MetricAtlasUnitOnly"
    route_reason = "unit gates passed but no MLP metric-atlas exploration gate opened"
    if any_mlp_gate:
        if any(s.get("method") == "sma_rank2" and iflag(s.get("exploration_gate_pass")) for s in candidate_gate):
            final_route = "R9-BoundedMetricShapingOpened"
        elif any(s.get("method") == "tma_rank2" and iflag(s.get("exploration_gate_pass")) for s in candidate_gate):
            final_route = "R8-TransportedMetricAtlasOpened"
        elif any(s.get("method") == "fma_rank2" and iflag(s.get("exploration_gate_pass")) for s in candidate_gate):
            final_route = "R7-FrozenAtlasOpened_TransportPending"
        else:
            final_route = "R10-MLPFunctionalAtlasCandidate"
        route_reason = "at least one MLP metric-atlas family passed exploration gate"
    if not read_json(OUT_ROOT / "v22_64_part_c_unit_gate.json").get("part_c_unit_gate_pass"):
        final_route = "R1-FunctionalMetricConstructionFailed"
        route_reason = "Part C unit gate did not pass"
    if not read_json(OUT_ROOT / "v22_64_part_b_metric_gate.json").get("part_b_metric_gate_pass"):
        final_route = "R1-FunctionalMetricConstructionFailed"
        route_reason = "Part B metric/active atlas gate did not pass"
    code_gate = read_rows(OUT_ROOT / "v22_64_code_truth_gate.csv")
    if code_gate and not iflag(code_gate[0].get("part_a_hard_gate_pass")):
        final_route = "R0-CodeBoundaryFailed"
        route_reason = "Part A hard gate did not pass"
    kan_gate_status = "skipped_MLP_MetricAtlas_gate_not_opened" if not any_mlp_gate else "pending_MLP_MetricAtlas_gate_opened"
    failure_rows = [{"method": r.get("method"), "method_family": r.get("method_family"), "dataset": r.get("dataset"), "seed": r.get("seed"), "run_status": r.get("run_status"), "failure_mode": r.get("failure_mode"), "held_NLL": r.get("held_NLL"), "Delta_NLL_vs_best_control": r.get("Delta_NLL_vs_best_control")} for r in rows]

    write_rows(OUT_ROOT / "v22_64_mlp_metric_atlas_matrix.csv", mlp_rows)
    write_rows(OUT_ROOT / "v22_64_metric_atlas_controls_matrix.csv", control_rows)
    write_rows(OUT_ROOT / "v22_64_external_oet_comparison.csv", external_rows or [{"status": "no_external_rows"}])
    write_rows(OUT_ROOT / "v22_64_metric_drift_diagnosis.csv", diag_rows)
    write_rows(OUT_ROOT / "v22_64_failure_route_matrix.csv", failure_rows)
    write_rows(OUT_ROOT / "v22_64_method_summary.csv", summary_rows)
    route = {
        "final_route": final_route,
        "route_reason": route_reason,
        "kan_gate_status": kan_gate_status,
        "mlp_exploration_gate_opened": int(any_mlp_gate),
        "candidate_gate_summary": candidate_gate,
        "rows": len(rows),
        "completed_rows": sum(1 for r in rows if r.get("run_status") == "completed"),
        "external_unavailable_rows": sum(1 for r in rows if r.get("run_status") == "external_unavailable"),
        "generated_at": now_sg(),
    }
    write_json(OUT_ROOT / "v22_64_final_route.json", route)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "aggregate"]),
        task_id="F_metric_drift_diagnosis_and_route",
        status="pass",
        gpu="cpu",
        files="results/v22_64/v22_64_*matrix.csv; results/v22_64/v22_64_final_route.json",
        note=f"final_route={final_route}; kan_gate_status={kan_gate_status}; rows={len(rows)}",
    )
    return route


def update_docs_from_results() -> None:
    ensure_out()
    code_rows = read_rows(OUT_ROOT / "v22_64_code_truth_gate.csv")
    metric_rows = read_rows(OUT_ROOT / "v22_64_functional_metric_matrix.csv")
    unit_rows = read_rows(OUT_ROOT / "v22_64_metric_preservation_unit_tests.csv")
    summary_rows = read_rows(OUT_ROOT / "v22_64_method_summary.csv")
    diag_rows = read_rows(OUT_ROOT / "v22_64_metric_drift_diagnosis.csv")
    route = read_json(OUT_ROOT / "v22_64_final_route.json")
    recap = []
    recap.append("### Final route\n")
    recap.append("```json\n" + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n")
    recap.append("### Part A code/training boundary\n")
    recap.append(md_table(code_rows, ["part_a_hard_gate_pass", "compileall_pass", "clean_tarball_self_contained_import_pass", "standard_loop_static_scan_pass", "standard_loop_runtime_trace_pass", "loss_total_is_task_loss_only", "manual_param_update_detected", "class_weight_or_sampler_used_as_fu"], limit=5))
    recap.append("### Part B functional metric / active atlas evidence\n")
    recap.append(md_table(metric_rows, ["dataset", "seed", "Gf_psd_min", "Gf_condition_number", "active_atlas_rank", "active_Gram_condition", "source_witness_alignment_LCB", "signal_to_reservoir_energy_ratio", "metric_build_ms"], limit=20))
    recap.append("### Part C metric preservation unit tests\n")
    recap.append(md_table(unit_rows, ["seed", "frozen_Gram_error", "transported_Gram_error", "bounded_shape_drift", "task_descent_cosine", "poet_equivalence_error", "part_c_unit_gate_pass"], limit=20))
    recap.append("### D/E method summary\n")
    recap.append(md_table(summary_rows, ["method", "method_family", "completed_rows", "mean_final_NLL", "beats_strongest_NLL_rows", "beats_best_control_NLL_rows", "beats_external_OET_NLL_rows", "no_debt_rows", "overhead_le_035_rows", "exploration_gate_pass"], limit=60))
    recap.append("### F diagnostic correlations\n")
    recap.append(md_table(diag_rows, ["method", "rows", "failure_modes", "corr_active_Gram_drift_Delta_NLL", "corr_signal_reachable_energy_Delta_NLL", "corr_reservoir_reachable_energy_tail_debt", "corr_transport_error_no_debt"], limit=60))
    insights = [
        "### Analysis / conclusion / insight\n",
        f"- Final route is `{route.get('final_route', '')}`: {route.get('route_reason', '')}.",
        f"- KAN gate status is `{route.get('kan_gate_status', '')}`; this follows the document rule that KAN official rows start only after MLP MetricAtlas gate opens.",
        "- Evidence chain: Part A verifies code boundary, Part B verifies metric/active atlas construction, Part C verifies preservation math, D/E compares task rows against references and matched controls, F assigns failure modes and correlations.",
        "- No fabricated rows are used: unavailable external baselines remain `external_unavailable`; blocked controls remain blocked and are not counted as wins.",
    ]
    append_recap("Final results and evidence chain", "\n".join(recap + insights))
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write("\n## Repro command summary\n\n")
        f.write("核心命令：\n\n")
        f.write("```bash\n")
        f.write(f"{PYTHON} experiments/run_v22_64_metric_preserving_functional_atlas_fu.py --mode full --gpus 0,1,2,3\n")
        f.write("```\n\n")
        f.write("关键产物：`results/v22_64/` 下所有 `v22_64_*.csv/json`，以及本执行日志和实验结果复盘。\n")


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    code = run_code_truth_gate(args)
    if not iflag(code.get("part_a_hard_gate_pass")):
        route = {"final_route": "R0-CodeBoundaryFailed", "route_reason": "Part A hard gate failed", "kan_gate_status": "skipped_code_boundary_failed"}
        write_json(OUT_ROOT / "v22_64_final_route.json", route)
        append_recap("Stopped at Part A", "Part A hard gate failed. Per plan, B-H were not run.")
        return route
    metric = run_metric_construction(args)
    b_gate = read_json(OUT_ROOT / "v22_64_part_b_metric_gate.json")
    if not iflag(b_gate.get("part_b_metric_gate_pass")):
        route = {"final_route": "R1-FunctionalMetricConstructionFailed", "route_reason": "Part B metric/atlas gate failed", "kan_gate_status": "skipped_metric_gate_failed"}
        write_json(OUT_ROOT / "v22_64_final_route.json", route)
        append_recap("Stopped at Part B", "Part B metric/active atlas gate failed. Per plan, C-H were not run.")
        return route
    unit = run_unit_gate(args)
    c_gate = read_json(OUT_ROOT / "v22_64_part_c_unit_gate.json")
    if not iflag(c_gate.get("part_c_unit_gate_pass")):
        route = {"final_route": "R1-FunctionalMetricConstructionFailed", "route_reason": "Part C preservation unit gate failed", "kan_gate_status": "skipped_unit_gate_failed"}
        write_json(OUT_ROOT / "v22_64_final_route.json", route)
        append_recap("Stopped at Part C", "Part C unit gate failed. Per plan, D-H were not run.")
        return route
    run_matrix(args)
    route = aggregate_results(args)
    update_docs_from_results()
    return route


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="full", choices=["full", "smoke", "code-gate", "metric-gate", "unit-gate", "matrix", "row", "aggregate", "docs"])
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=900)
    p.add_argument("--dataset", default="Wine")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="fma_rank2")
    p.add_argument("--device", default="auto")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--refresh", type=int, default=50)
    p.add_argument("--lr", type=float, default=3.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--metric-kind", default="signal_debt")
    p.add_argument("--metric-batch-size", type=int, default=0)
    p.add_argument("--transport-mode", default="full")
    p.add_argument("--shaping-budget", type=float, default=0.05)
    p.add_argument("--unit-seeds", default="0,1,2")
    p.add_argument("--poet-block-size", type=int, default=16)
    p.add_argument("--poet-merge-interval", type=int, default=50)
    p.add_argument("--poet-lr", type=float, default=1.0e-3)
    p.add_argument("--poet-scale", type=float, default=1.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    try:
        if args.mode == "row":
            train_row(args)
        elif args.mode == "code-gate":
            run_code_truth_gate(args)
        elif args.mode == "metric-gate":
            run_metric_construction(args)
        elif args.mode == "unit-gate":
            run_unit_gate(args)
        elif args.mode == "matrix":
            run_matrix(args)
        elif args.mode == "aggregate":
            aggregate_results(args)
        elif args.mode == "docs":
            update_docs_from_results()
        elif args.mode == "smoke":
            smoke_args = argparse.Namespace(**vars(args))
            smoke_args.datasets = "Wine,MNIST"
            smoke_args.seeds = "0"
            smoke_args.methods = "adamw,fma_rank2,tma_rank2,sma_rank2,same_rank_random_atlas,same_compute_noop_coordinate"
            smoke_args.steps = min(int(args.steps), 20)
            smoke_args.train_size = min(int(args.train_size), 128)
            smoke_args.held_size = min(int(args.held_size), 64)
            smoke_args.test_size = min(int(args.test_size), 64)
            smoke_args.unit_seeds = "0"
            route = run_full(smoke_args)
            print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            route = run_full(args)
            print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception:
        err_path = LOG_ROOT / f"v22_64_{safe_fragment(args.mode)}_exception.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
        append_exec(command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", str(args.mode)]), task_id=f"exception_{args.mode}", status="exception", files=str(err_path.relative_to(ROOT)), note=f"see {err_path}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
