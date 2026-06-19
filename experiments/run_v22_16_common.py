#!/usr/bin/env python3
"""Shared helpers for v22.16 real-trajectory adaptive FU runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable
import zipfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dgkan.models.fc_purekan_primitives import MLPBaseline, PrimitiveKAN, PrimitiveSpec, count_parameters


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
V2216_ROOT = ROOT / "results/v22_16_real_trajectory_source_manifold_adaptive_fu"
V2216_OFFICIAL = V2216_ROOT / "official_v22_16"
V2216_PLAN_DOC = ROOT / "docs/DG-KAN_v22.16_RealTrajectorySourceManifoldAdaptiveFU_完整计划.md"
V2216_EXEC_DOC = ROOT / "docs/DG-KAN_v22.16_RealTrajectorySourceManifoldAdaptiveFU_执行日志.md"
V2216_RECAP_DOC = ROOT / "docs/DG-KAN_v22.16_RealTrajectorySourceManifoldAdaptiveFU_实验结果复盘.md"


REQUIRED_SOURCE_FILES = [
    "dgkan/fu/adaptive_controller.py",
    "dgkan/fu/source_guided_step.py",
    "dgkan/fu/source_state_transport.py",
    "dgkan/fu/loss_geometry.py",
    "dgkan/fu/source_manifold_basis.py",
    "dgkan/fu/source_manifold_controller.py",
    "dgkan/fu/basis_native_controller.py",
    "dgkan/fu/real_jacobian_commit.py",
    "dgkan/fu/mlp_adaptive_controller.py",
    "dgkan/fu/mlp_source_usefulness.py",
    "dgkan/fu/real_source_manifold.py",
    "dgkan/profiling/real_adaptive_fu_efficiency.py",
    "experiments/run_v22_16_common.py",
    "experiments/run_v22_16_s0_truth.py",
    "experiments/run_v22_16_real_trajectory_logger.py",
    "experiments/run_v22_16_real_mlp_adaptive_controller.py",
    "experiments/run_v22_16_real_source_manifold.py",
    "experiments/run_v22_16_real_loss_geometry.py",
    "experiments/run_v22_16_real_kan_basis_controller.py",
    "experiments/run_v22_16_real_efficiency_controller_loop.py",
    "experiments/run_v22_16_strict_task_proof.py",
    "experiments/run_v22_16_mlp_fu_mainline.py",
    "experiments/run_v22_16_mlp_fu_source_logger.py",
    "experiments/run_v22_16_mlp_fu_task_eval.py",
    "experiments/run_v22_16_mlp_fu_finalize_patch.py",
    "experiments/run_v22_16_finalize.py",
]

DATASETS = ["MNIST", "FashionMNIST", "KMNIST"]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else V2216_OFFICIAL
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def init_docs() -> None:
    V2216_EXEC_DOC.parent.mkdir(parents=True, exist_ok=True)
    if not V2216_EXEC_DOC.exists():
        V2216_EXEC_DOC.write_text(
            "# DG-KAN v22.16 Real-Trajectory Source-Manifold Adaptive FU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、文件、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。\n",
            encoding="utf-8",
        )
    if not V2216_RECAP_DOC.exists():
        V2216_RECAP_DOC.write_text(
            "# DG-KAN v22.16 Real-Trajectory Source-Manifold Adaptive FU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本文件只汇总落盘 artifact 与真实运行/读回结果；禁止编造数据。\n",
            encoding="utf-8",
        )


def append_exec(
    out_dir: Path,
    command: str,
    *,
    status: str = "",
    note: str = "",
    gpu: str = "",
    task_id: str = "",
    files: str = "",
) -> None:
    init_docs()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "gpu": gpu,
        "command": command,
        "status": status,
        "files": files,
        "note": note,
    }
    journal = out_dir / "v22_16_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "task_id", "gpu", "command", "status", "files", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    with V2216_EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n```bash\n{command}\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status or 'recorded'}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(r) for r in rows]
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if out == out and abs(out) != float("inf") else default


def md_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int = 30) -> str:
    if not rows:
        return "\n_无落盘 rows。_\n"
    shown = rows[:max_rows]
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown:
        out.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    if len(rows) > max_rows:
        out.append(f"\n_仅显示前 {max_rows} / {len(rows)} rows；完整 CSV 见 artifact。_")
    return "\n".join(out) + "\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def packet_source_paths() -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for rel in REQUIRED_SOURCE_FILES:
        path = ROOT / rel
        if path.exists() and path not in seen:
            seen.add(path)
            paths.append(path)
    for pattern in ["dgkan/models/**/*.py", "dgkan/kernels/**/*.py", "dgkan/profiling/**/*.py", "dgkan/fu/**/*.py", "experiments/run_v22_16*.py"]:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                paths.append(path)
    for path in [V2216_PLAN_DOC, V2216_EXEC_DOC, V2216_RECAP_DOC]:
        if path.exists() and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def build_code_review_packet(out_dir: Path) -> Path:
    packet = out_dir / "v22_16_code_review_packet.zip"
    if packet.exists():
        packet.unlink()
    included: list[dict[str, Any]] = []
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in packet_source_paths():
            arc = path.resolve().relative_to(ROOT)
            z.write(path, arc)
            included.append({"packet_path": str(arc), "source_path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        for path in sorted(out_dir.rglob("v22_16_*")):
            if path.is_file() and path.suffix.lower() != ".zip" and "_code_packet_unzip" not in path.parts:
                arc = Path("artifacts") / path.relative_to(out_dir)
                z.write(path, arc)
                included.append({"packet_path": str(arc), "source_path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(out_dir / "v22_16_code_review_packet_manifest.csv", included)
    return packet


def unpack_code_packet(out_dir: Path) -> Path:
    packet = out_dir / "v22_16_code_review_packet.zip"
    root = out_dir / "_code_packet_unzip"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet, "r") as z:
        z.extractall(root)
    return root


def build_results_bundle(out_dir: Path) -> Path:
    bundle = out_dir / "v22_16_results_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".zip" and "_code_packet_unzip" not in path.parts:
                z.write(path, path.relative_to(out_dir))
        for path in packet_source_paths():
            z.write(path, path.resolve().relative_to(ROOT))
    return bundle


def artifact_index(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and "_code_packet_unzip" not in path.parts:
            try:
                artifact = str(path.resolve().relative_to(ROOT))
            except ValueError:
                artifact = str(path)
            rows.append({"artifact": artifact, "exists": 1, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def write_execution_manifests(out_dir: Path) -> None:
    queue = [
        {"task_id": "S0", "gpu": "0", "script": "experiments/run_v22_16_s0_truth.py", "status": "runnable"},
        {"task_id": "B-real-trajectory", "gpu": "3", "script": "experiments/run_v22_16_real_trajectory_logger.py", "status": "runnable"},
        {"task_id": "C-D-risk-mlp", "gpu": "1", "script": "experiments/run_v22_16_real_mlp_adaptive_controller.py", "status": "runnable"},
        {"task_id": "E-source-manifold", "gpu": "3", "script": "experiments/run_v22_16_real_source_manifold.py", "status": "depends_on_trajectory"},
        {"task_id": "F-loss-geometry", "gpu": "3", "script": "experiments/run_v22_16_real_loss_geometry.py", "status": "runnable"},
        {"task_id": "G-kan-basis-dche", "gpu": "0", "script": "experiments/run_v22_16_real_kan_basis_controller.py --carriers D-CHE", "status": "runnable"},
        {"task_id": "G-kan-basis-dfou", "gpu": "2", "script": "experiments/run_v22_16_real_kan_basis_controller.py --carriers D-FOU", "status": "runnable"},
        {"task_id": "H-efficiency-dche", "gpu": "0", "script": "experiments/run_v22_16_real_efficiency_controller_loop.py --carriers D-CHE", "status": "runnable"},
        {"task_id": "H-efficiency-dfou", "gpu": "2", "script": "experiments/run_v22_16_real_efficiency_controller_loop.py --carriers D-FOU", "status": "runnable"},
        {"task_id": "I-task-proof", "gpu": "3", "script": "experiments/run_v22_16_strict_task_proof.py", "status": "gate_dependent"},
        {"task_id": "finalize", "gpu": "n/a", "script": "experiments/run_v22_16_finalize.py", "status": "gate_dependent"},
    ]
    write_rows(out_dir / "v22_16_runnable_queue.csv", queue)
    write_rows(out_dir / "v22_16_gpu_assignment_manifest.csv", queue)
    write_rows(out_dir / "v22_16_idle_violation.csv", [{"execution_contract_violation": 0, "reason": "initial manifest created; finalizer recomputes from command journal"}])
    write_json(out_dir / "v22_16_queue_drain_report.json", {"runnable_queue_nonempty_after_run": 0, "execution_contract_violation": 0})


def device_from_arg(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def dataset_obj(name: str, train: bool, *, download: bool = False) -> Any:
    root = ROOT / "data"
    tx = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.reshape(-1))])
    cls = {"MNIST": datasets.MNIST, "FashionMNIST": datasets.FashionMNIST, "KMNIST": datasets.KMNIST}[name]
    return cls(str(root), train=train, download=bool(download), transform=tx)


def loader_for(name: str, train: bool, size: int, batch: int, seed: int, *, download: bool = False, shuffle: bool | None = None) -> DataLoader:
    ds = dataset_obj(name, train, download=download)
    gen = torch.Generator()
    gen.manual_seed(int(seed) + (0 if train else 10000))
    perm = torch.randperm(len(ds), generator=gen)[: min(int(size), len(ds))].tolist()
    return DataLoader(Subset(ds, perm), batch_size=batch, shuffle=train if shuffle is None else shuffle, generator=gen, drop_last=False)


def make_kan(x_stats: torch.Tensor, hidden: int, seed: int, device: torch.device, *, carrier: str = "D-FOU", init_variant: str = "identity_residual_scale") -> PrimitiveKAN:
    input_dim = 784
    output_dim = 10
    k = 5
    mlp_budget = input_dim * hidden + hidden * hidden + hidden * output_dim
    basis_name = "fourier_lowfreq" if carrier == "D-FOU" else "chebyshev"
    basis_family = carrier
    extra_params = input_dim * output_dim if "linearres" in init_variant else 0
    effective_budget = max(4 * k * (input_dim + output_dim), mlp_budget - extra_params)
    kan_hidden = max(4, int(round(float(effective_budget) / float(k * (input_dim + output_dim)))))
    spec = PrimitiveSpec(
        candidate_id=f"v22.16-{carrier}",
        basis_family=basis_family,
        basis_name=basis_name,
        k=k,
        hidden_dim=kan_hidden,
        source="v22_16_real_training",
        local_support=0,
        global_support=1,
        uses_exp=0,
        uses_sin_cos=int(carrier == "D-FOU"),
        uses_division=0,
        uses_dense_basis_tensor=1,
        init_variant=init_variant,
    )
    return PrimitiveKAN(input_dim, output_dim, spec, x_stats.to(device), seed, device, param_budget=mlp_budget)


def make_model(variant: str, first_x: torch.Tensor, hidden: int, seed: int, device: torch.device, *, carrier: str = "D-FOU") -> torch.nn.Module:
    if variant.startswith("MLP"):
        return MLPBaseline(784, 10, hidden, seed, device).to(device)
    init_variant = "identity_residual_scale"
    if "readout" in variant or "coupled" in variant:
        init_variant = "identity_residual_scale_linearres050"
    return make_kan(first_x.to(device), hidden, seed, device, carrier=carrier, init_variant=init_variant).to(device)


def model_family(variant: str) -> str:
    return "MLP" if variant.startswith("MLP") else "KAN"


def selected_named_parameters(model: torch.nn.Module, selector: str = "all") -> list[tuple[str, torch.nn.Parameter]]:
    rows = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if selector == "basis":
        picked = [(n, p) for n, p in rows if n.endswith("w1") or ".w1" in n or n == "w1"]
        return picked or rows
    if selector == "readout":
        picked = [(n, p) for n, p in rows if n.endswith("w2") or ".w2" in n or "readout" in n]
        return picked or rows
    if selector == "coupled":
        return rows
    return rows


def flat_grads(named_params: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    parts = []
    for _name, p in named_params:
        if p.grad is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(p.grad.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def assign_flat_grads(named_params: list[tuple[str, torch.nn.Parameter]], flat: torch.Tensor) -> None:
    offset = 0
    for _name, p in named_params:
        n = int(p.numel())
        chunk = flat[offset : offset + n].reshape_as(p)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += n


def cosine_t(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().reshape(-1).cpu()
    bv = b.detach().float().reshape(-1).cpu()
    n = min(av.numel(), bv.numel())
    if n == 0:
        return 0.0
    av = av[:n]
    bv = bv[:n]
    denom = torch.linalg.vector_norm(av).clamp_min(1.0e-12) * torch.linalg.vector_norm(bv).clamp_min(1.0e-12)
    return float(torch.dot(av, bv).div(denom).item())


def ce_cotangent(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    target = F.one_hot(y.long(), num_classes=logits.shape[-1]).float()
    return (probs - target.to(probs.device)) / max(1, int(y.numel()))


def mse_cotangent(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    target = F.one_hot(y.long(), num_classes=logits.shape[-1]).float().to(logits.device)
    return 2.0 * (logits.float() - target) / max(1, int(logits.numel()))


def apply_gradient_guidance(
    model: torch.nn.Module,
    source_state: torch.Tensor | None,
    *,
    selector: str,
    mode: str,
    step: int,
    seed: int,
    lambda_override: float | None = None,
    risk_override: float | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    named = selected_named_parameters(model, selector)
    grad = flat_grads(named).float()
    if grad.numel() == 0:
        return source_state, {"risk_score": 0.0, "lambda_t": 0.0, "source_retention": 0.0, "controller_update_norm": 0.0, "ordinary_update_norm": 0.0}
    update = -grad
    update_norm = torch.linalg.vector_norm(update).clamp_min(1.0e-12)
    update_unit = update / update_norm
    if source_state is None or source_state.numel() != update.numel():
        if "random" in mode.lower():
            gen = torch.Generator(device=update.device)
            gen.manual_seed(int(seed) + 1000003)
            source_state = torch.randn(update.numel(), generator=gen, device=update.device)
        elif "signflip" in mode.lower():
            source_state = -update_unit.detach().clone()
        else:
            source_state = update_unit.detach().clone()
    source_unit = source_state.detach().float()
    source_unit = source_unit / torch.linalg.vector_norm(source_unit).clamp_min(1.0e-12)
    align = float(torch.dot(update_unit, source_unit).clamp(-1.0, 1.0).item())
    destructive = max(0.0, -align)
    weak = max(0.0, 0.25 - align)
    risk = max(0.0, min(1.0, 0.08 + 0.55 * destructive + 0.35 * weak + 0.10 * min(1.0, step / 1200.0)))
    if risk_override is not None:
        risk = max(0.0, min(1.0, float(risk_override)))
    conservative = "conservative" in mode.lower()
    if lambda_override is not None:
        lam = max(0.0, float(lambda_override))
    elif mode in {"none", "adamw", "sgd"}:
        lam = 0.0
    elif "fixed" in mode:
        lam = (0.25 if conservative else 0.45) if step % 200 == 0 else 0.0
    elif "reactive" in mode:
        lam = (0.40 if conservative else 0.85) if align < (0.0 if conservative else 0.20) else 0.0
    elif "release" in mode:
        if conservative:
            lam = 0.20 if align >= 0.10 and risk >= 0.45 else 0.0
        else:
            lam = 0.55 if align >= -0.05 else 0.05
    elif "manifold" in mode:
        lam = (0.15 + 0.35 * risk) if conservative else (0.35 + 0.65 * risk)
    elif "random" in mode.lower() or "signflip" in mode.lower():
        lam = 0.55 * risk
    else:
        if conservative:
            x = max(0.0, (risk - 0.45) / 0.55)
            lam = 0.45 * x * x
        else:
            lam = 0.20 + 0.80 * risk
    guided = (update + float(lam) * source_unit * update_norm) / (1.0 + float(lam))
    if "manifold" in mode and guided.numel() >= 16:
        blocks = min(8, max(1, guided.numel() // 4096))
        if blocks > 1:
            usable = blocks * (guided.numel() // blocks)
            block_mean = guided[:usable].reshape(blocks, -1).mean(dim=0).repeat(blocks)
            guided = guided.clone()
            guided[:usable] = 0.90 * guided[:usable] + 0.10 * block_mean
    assign_flat_grads(named, -guided.to(dtype=grad.dtype))
    next_state = (0.985 * source_unit + 0.015 * update_unit).detach()
    return next_state, {
        "risk_score": risk,
        "lambda_t": float(lam),
        "source_retention": align,
        "destructive_projection": destructive,
        "ordinary_update_norm": float(update_norm.item()),
        "controller_update_norm": float(torch.linalg.vector_norm(guided - update).item()),
        "controller_to_base_update_ratio": float(torch.linalg.vector_norm(guided - update).div(update_norm).item()),
        "source_loss_proxy": cosine_t(guided, update_unit),
    }


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    logits_all: list[torch.Tensor] = []
    y_all: list[torch.Tensor] = []
    losses_all: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            xb = x.to(device).float()
            yb = y.to(device).long()
            logits = model(xb).float()
            losses = F.cross_entropy(logits, yb, reduction="none")
            total_loss += float(losses.sum().item())
            total += int(yb.numel())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
            logits_all.append(logits.detach().cpu())
            y_all.append(yb.detach().cpu())
            losses_all.append(losses.detach().cpu())
    logits_cat = torch.cat(logits_all) if logits_all else torch.empty(0, 10)
    y_cat = torch.cat(y_all) if y_all else torch.empty(0, dtype=torch.long)
    losses_cat = torch.cat(losses_all) if losses_all else torch.empty(0)
    probs = torch.softmax(logits_cat.float(), dim=-1) if logits_cat.numel() else logits_cat
    target = F.one_hot(y_cat, num_classes=10).float() if y_cat.numel() else torch.empty_like(probs)
    ece = 0.0
    if y_cat.numel():
        conf, pred = probs.max(dim=-1)
        correct_vec = (pred == y_cat).float()
        for idx in range(10):
            lo = idx / 10.0
            hi = (idx + 1) / 10.0
            mask = (conf >= lo) & (conf <= hi if idx == 9 else conf < hi)
            if mask.any():
                ece += float(mask.float().mean().item()) * abs(float(conf[mask].mean().item()) - float(correct_vec[mask].mean().item()))
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "ECE": ece,
        "Brier": float((probs - target).square().sum(dim=-1).mean().item()) if y_cat.numel() else 0.0,
        "tail_loss_q95": float(torch.quantile(losses_cat.float(), 0.95).item()) if losses_cat.numel() else 0.0,
        "tail_loss_q99": float(torch.quantile(losses_cat.float(), 0.99).item()) if losses_cat.numel() else 0.0,
    }


def param_energy_by_channel(model: torch.nn.Module) -> tuple[float, float]:
    basis = 0.0
    readout = 0.0
    for name, p in model.named_parameters():
        val = float(p.detach().float().square().sum().item())
        if name.endswith("w1") or ".w1" in name or name == "w1":
            basis += val
        elif name.endswith("w2") or ".w2" in name or "readout" in name:
            readout += val
    total = max(1.0e-12, basis + readout)
    return basis / total, readout / total


def grad_energy_by_channel(model: torch.nn.Module) -> tuple[float, float]:
    basis = 0.0
    readout = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        val = float(p.grad.detach().float().square().sum().item())
        if name.endswith("w1") or ".w1" in name or name == "w1":
            basis += val
        elif name.endswith("w2") or ".w2" in name or "readout" in name:
            readout += val
    total = max(1.0e-12, basis + readout)
    return basis / total, readout / total


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


__all__ = [
    "DATASETS",
    "PYTHON",
    "REQUIRED_SOURCE_FILES",
    "ROOT",
    "V2216_EXEC_DOC",
    "V2216_OFFICIAL",
    "V2216_PLAN_DOC",
    "V2216_RECAP_DOC",
    "append_exec",
    "apply_gradient_guidance",
    "artifact_index",
    "build_code_review_packet",
    "build_results_bundle",
    "ce_cotangent",
    "cosine_t",
    "count_parameters",
    "device_from_arg",
    "ensure_out",
    "evaluate",
    "finite_float",
    "grad_energy_by_channel",
    "init_docs",
    "int_flag",
    "loader_for",
    "make_model",
    "md_table",
    "model_family",
    "mse_cotangent",
    "now_sg",
    "param_energy_by_channel",
    "read_json",
    "read_rows",
    "selected_named_parameters",
    "sha256_file",
    "tensor_sha256",
    "unpack_code_packet",
    "write_execution_manifests",
    "write_json",
    "write_rows",
]
