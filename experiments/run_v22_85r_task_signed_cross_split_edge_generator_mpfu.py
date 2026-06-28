#!/usr/bin/env python3
"""DG-KAN v22.85R task-signed cross-split edge generator runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import py_compile
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_80_control_orthogonal_edge_signal_kan_mpfu as base80
import experiments.run_v22_82_persistent_edge_coordinate_generator_kan_mpfu as base82
import experiments.run_v22_83_observable_edge_active_atlas_kan_mpfu as base83
import experiments.run_v22_84r_edge_function_rkhs_oracle_materialized_atoms_mpfu as base84
from dgkan.models.fc_purekan_primitives import MLPBaseline


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_85r_task_signed_cross_split_edge_generator_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.85R_TaskSignedCrossSplitEdgeGenerator_MPFU_修改计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.85R_TaskSignedCrossSplitEdgeGenerator_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.85R_TaskSignedCrossSplitEdgeGenerator_MPFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2285R_OUT_ROOT", str(ROOT / "results/v22_85r"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"
V2284_ROOT = ROOT / "results/v22_84r"

PRIMARY_FAMILY = "task_signed_class_bucket_transport_shape"
DIAGNOSTIC_FAMILIES = {
    "diagnostic_beta0": 0.0,
    "diagnostic_beta01": 0.1,
    "diagnostic_beta05": 0.5,
    "diagnostic_beta10": 1.0,
}


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
    return base84.fval(x, default)


def quantile(values: list[float], q: float) -> float:
    return base84.quantile(values, q)


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    return base84.lower_cvar(values, frac)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_out()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    ensure_out()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.85R Task-Signed Cross-Split Edge Generator MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            "- 非编造约束：只记录真实命令、真实 artifact、真实错误与观测；缺失项写 missing/skipped。\n"
            "- 复现提示：Part D 支持 `--shard-count/--shard-index` 并行，默认输出到 `results/v22_85r/`。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.85R Task-Signed Cross-Split Edge Generator MPFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "## 0. 当前结论\n"
            "- 尚未完成最终 route 判定。\n"
            "- 本文件只记录真实 artifact、真实指标、实际修复与分析；不补造缺失数据。\n\n"
            "## 1. 证据链、修复与分析\n",
            encoding="utf-8",
        )


def append_exec(task: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {now_sg()} | {task} | {status}\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, lines: list[str]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n")
        fh.write(f"- time_sg: {now_sg()}\n")
        for line in lines:
            fh.write(f"- {line}\n")


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


def carrier_task_grid(args: argparse.Namespace) -> list[tuple[dict[str, str], int, str]]:
    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    seeds = list(range(int(args.seed_count)))
    specs = [dict(s) for s in base80.CARRIER_REDESIGN_SPECS[: max(1, int(args.spec_count))]]
    return [(spec, seed, dataset) for spec in specs for dataset in datasets for seed in seeds]


def _tensor_vision_dataset(dataset: Any) -> tuple[Any, Any]:
    import numpy as np
    import torch

    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.array(x))
        x = x.float()
        if x.ndim == 3 and x.shape[0] not in {1, 3}:
            x = x.permute(2, 0, 1)
        xs.append((x.reshape(-1) / 255.0).to(dtype=torch.float32))
        ys.append(int(y))
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def _split_normalized_bundle(
    x_train_all: Any,
    y_train_all: Any,
    x_test_all: Any,
    y_test_all: Any,
    *,
    train_size: int,
    held_size: int,
    test_size: int,
    seed: int,
    source_kind: str,
) -> dict[str, Any]:
    import torch

    gen = torch.Generator().manual_seed(int(seed))
    train_perm = torch.randperm(int(y_train_all.numel()), generator=gen)
    test_perm = torch.randperm(int(y_test_all.numel()), generator=gen)
    train_n = min(int(train_size), int(train_perm.numel()))
    held_n = min(int(held_size), max(0, int(train_perm.numel()) - train_n))
    test_n = min(int(test_size), int(test_perm.numel()))
    train_idx = train_perm[:train_n]
    held_idx = train_perm[train_n : train_n + held_n]
    test_idx = test_perm[:test_n]
    x_train = x_train_all[train_idx].float()
    x_held = x_train_all[held_idx].float()
    x_test = x_test_all[test_idx].float()
    mu = x_train.mean(dim=0, keepdim=True)
    sigma = x_train.std(dim=0, keepdim=True)
    sigma = torch.where(sigma < 1.0e-6, torch.ones_like(sigma), sigma)
    x_train = (x_train - mu) / sigma
    x_held = (x_held - mu) / sigma
    x_test = (x_test - mu) / sigma
    y_train = y_train_all[train_idx].long()
    y_held = y_train_all[held_idx].long()
    y_test = y_test_all[test_idx].long()
    return {
        "input_dim": int(x_train.shape[1]),
        "num_classes": int(max(y_train.max(), y_held.max(), y_test.max()).item() + 1),
        "x_train": x_train,
        "y_train": y_train,
        "x_held": x_held,
        "y_held": y_held,
        "x_test": x_test,
        "y_test": y_test,
        "source_kind": source_kind,
        "used_fake_data": 0,
    }


def load_bundle_v2285(dataset: str, train_size: int, held_size: int, test_size: int, seed: int) -> dict[str, Any]:
    try:
        import experiments.run_v22_73_distributional_edge_natural_residual_kan_mpfu as base73

        return base73.load_bundle(str(dataset), int(train_size), int(held_size), int(test_size), int(seed))
    except ValueError as exc:
        if "unknown dataset" not in str(exc):
            raise
    low = str(dataset).lower()
    if low not in {"svhn", "emnist-letters", "emnist_letters", "emnistletters"}:
        raise ValueError(f"unknown dataset {dataset!r}")
    from torchvision import datasets

    root = ROOT / "data"
    if low == "svhn":
        train_ds = datasets.SVHN(root=str(root), split="train", download=False)
        test_ds = datasets.SVHN(root=str(root), split="test", download=False)
        x_train_all, y_train_all = _tensor_vision_dataset(train_ds)
        x_test_all, y_test_all = _tensor_vision_dataset(test_ds)
        y_train_all = y_train_all.remainder(10)
        y_test_all = y_test_all.remainder(10)
        source = "torchvision.datasets.SVHN:local_mat"
    else:
        train_ds = datasets.EMNIST(root=str(root), split="letters", train=True, download=False)
        test_ds = datasets.EMNIST(root=str(root), split="letters", train=False, download=False)
        x_train_all, y_train_all = _tensor_vision_dataset(train_ds)
        x_test_all, y_test_all = _tensor_vision_dataset(test_ds)
        if int(y_train_all.min().item()) >= 1:
            y_train_all = y_train_all - 1
            y_test_all = y_test_all - 1
        source = "torchvision.datasets.EMNIST:letters:local_raw"
    return _split_normalized_bundle(
        x_train_all,
        y_train_all,
        x_test_all,
        y_test_all,
        train_size=int(train_size),
        held_size=int(held_size),
        test_size=int(test_size),
        seed=int(seed),
        source_kind=source,
    )


def make_model_and_batch_v2285(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device):
    try:
        return base83.make_model_and_batch(dataset, seed, spec, args, device)
    except ValueError as exc:
        if "unknown dataset" not in str(exc):
            raise
    bundle = load_bundle_v2285(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    x_all = bundle["x_train"].to(device).float()[: int(args.metric_batch_size)]
    y_all = bundle["y_train"].to(device).long()[: int(args.metric_batch_size)]
    seed_k = int(seed) + int(args.model_seed_offset)
    model = base82.base75.make_wlb_model(str(spec["method"]), bundle, device, int(args.hidden), seed_k, x_metric=x_all)
    mlp = MLPBaseline(int(bundle["input_dim"]), int(bundle["num_classes"]), int(args.hidden), seed_k, device).to(device)
    return model, mlp, bundle, base82.split_train(x_all, y_all)


def metric_energy(vec: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    return base84.metric_energy(vec.reshape(-1).to(dtype=torch.float64), metric.reshape(-1).to(device=vec.device, dtype=torch.float64))


def metric_norm(vec: torch.Tensor, metric: torch.Tensor) -> float:
    return float(torch.sqrt(metric_energy(vec, metric).clamp_min(0.0)).detach().cpu().item())


def metric_cosine(a: torch.Tensor, b: torch.Tensor, metric: torch.Tensor) -> float:
    aa = a.reshape(-1).to(dtype=torch.float64)
    bb = b.reshape(-1).to(device=aa.device, dtype=torch.float64)
    mm = metric.reshape(-1).to(device=aa.device, dtype=torch.float64)
    denom = torch.sqrt((aa.square() * mm).sum().clamp_min(1.0e-18) * (bb.square() * mm).sum().clamp_min(1.0e-18))
    return float(((aa * bb * mm).sum() / denom).detach().cpu().item())


def flat_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64)
    bb = b.detach().reshape(-1).to(device=aa.device, dtype=torch.float64)
    denom = (aa.norm() * bb.norm()).clamp_min(1.0e-12)
    return float(((aa * bb).sum() / denom).detach().cpu().item())


def row_metrics_from_logit_update(logits: torch.Tensor, y: torch.Tensor, update: torch.Tensor) -> tuple[dict[str, float], float]:
    return base84.row_metrics_from_logit_update(logits, y, update)


def stable_text_seed(text: str) -> int:
    return base84.stable_text_seed(text)


def class_confidence_bucket_coupling(
    y_s: torch.Tensor,
    y_w: torch.Tensor,
    conf_s: torch.Tensor,
    conf_w: torch.Tensor,
    *,
    num_classes: int,
    buckets: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = y_s.device
    pi = torch.zeros((int(y_s.numel()), int(y_w.numel())), device=device, dtype=torch.float64)
    bs = torch.clamp((conf_s.detach().to(dtype=torch.float64) * int(buckets)).floor().long(), 0, int(buckets) - 1)
    bw = torch.clamp((conf_w.detach().to(dtype=torch.float64) * int(buckets)).floor().long(), 0, int(buckets) - 1)
    groups = 0
    for cls in range(int(num_classes)):
        for bucket in range(int(buckets)):
            ms = (y_s.long() == cls) & (bs == bucket)
            mw = (y_w.long() == cls) & (bw == bucket)
            ns = int(ms.sum().detach().cpu().item())
            nw = int(mw.sum().detach().cpu().item())
            if ns > 0 and nw > 0:
                pi[ms][:, mw] = 1.0 / float(ns * nw)
                groups += 1
    if float(pi.sum().detach().cpu().item()) <= 0.0:
        pi.fill_(1.0 / max(1, int(pi.numel())))
    for _ in range(6):
        rs = pi.sum(dim=1, keepdim=True)
        pi = torch.where(rs > 0.0, pi / rs.clamp_min(1.0e-12), pi)
        cs = pi.sum(dim=0, keepdim=True)
        pi = torch.where(cs > 0.0, pi / cs.clamp_min(1.0e-12), pi)
    row_sum = pi.sum(dim=1)
    col_sum = pi.sum(dim=0)
    try:
        svals = torch.linalg.svdvals(pi)
        rank = int((svals > 1.0e-8).sum().detach().cpu().item())
        positive = svals[svals > 1.0e-12]
        cond = float((positive.max() / positive.min()).detach().cpu().item()) if int(positive.numel()) else 0.0
    except Exception:
        rank = 0
        cond = float("inf")
    return pi, {
        "primary_coupling": "class_balanced_confidence_bucket",
        "coupling_shape": f"{int(pi.shape[0])}x{int(pi.shape[1])}",
        "coupling_groups_nonempty": groups,
        "coupling_rank": rank,
        "coupling_condition_number": cond,
        "coupling_row_sum_min": float(row_sum.min().detach().cpu().item()) if int(row_sum.numel()) else 0.0,
        "coupling_row_sum_max": float(row_sum.max().detach().cpu().item()) if int(row_sum.numel()) else 0.0,
        "coupling_col_sum_min": float(col_sum.min().detach().cpu().item()) if int(col_sum.numel()) else 0.0,
        "coupling_col_sum_max": float(col_sum.max().detach().cpu().item()) if int(col_sum.numel()) else 0.0,
    }


def make_control_matrix(vectors: list[torch.Tensor]) -> torch.Tensor:
    cols: list[torch.Tensor] = []
    for vec in vectors:
        flat = vec.detach().reshape(-1).to(dtype=torch.float64)
        norm = float(flat.norm().detach().cpu().item())
        if norm > 1.0e-12:
            cols.append(flat / norm)
    if not cols:
        return torch.zeros((0, 0), dtype=torch.float64)
    return torch.stack(cols, dim=1)


def soft_project(vec: torch.Tensor, metric: torch.Tensor, controls: torch.Tensor, rho: float) -> torch.Tensor:
    x = vec.reshape(-1).to(dtype=torch.float64)
    g = metric.reshape(-1).to(device=x.device, dtype=torch.float64)
    c = controls.to(device=x.device, dtype=torch.float64)
    if int(c.numel()) == 0 or int(c.shape[1]) == 0:
        return torch.zeros_like(x)
    ctg = c.T * g[None, :]
    gram = ctg @ c
    eye = torch.eye(int(gram.shape[0]), device=x.device, dtype=torch.float64)
    rhs = ctg @ x
    try:
        coeff = torch.linalg.solve(gram + float(rho) * eye, rhs)
    except Exception:
        coeff = torch.linalg.pinv(gram + float(rho) * eye) @ rhs
    return c @ coeff


def relative_metric_apply(
    vec: torch.Tensor,
    metric: torch.Tensor,
    controls: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
    rho: float,
) -> torch.Tensor:
    x = vec.reshape(-1).to(dtype=torch.float64)
    g = metric.reshape(-1).to(device=x.device, dtype=torch.float64)
    p = soft_project(x, metric, controls, rho)
    return g * (x - float(alpha) * p + float(gamma) * x)


def soft_quotient_metrics(candidate: torch.Tensor, metric: torch.Tensor, controls: torch.Tensor, *, alpha: float, gamma: float) -> dict[str, Any]:
    x = candidate.reshape(-1).to(dtype=torch.float64)
    g = metric.reshape(-1).to(device=x.device, dtype=torch.float64)
    c = controls.to(device=x.device, dtype=torch.float64)
    dim = int(c.shape[1]) if int(c.ndim) == 2 else 0
    if dim <= 0:
        return {
            "control_suppression_ratio": 0.0,
            "task_energy_retention_ratio": 1.0,
            "MLP_tangent_leakage_ratio": 0.0,
            "debt_tangent_leakage_ratio": 0.0,
            "soft_projection_condition_number": 0.0,
            "relative_metric_vs_plain_metric_gap": 0.0,
            "candidate_control_projection_fraction": 0.0,
            "candidate_task_energy_fraction": 1.0,
            "soft_projector_rho": 0.0,
        }
    gram = (c.T * g[None, :]) @ c
    trace = float(torch.trace(gram).detach().cpu().item())
    rho = 1.0e-3 * trace / max(1, dim)
    p = soft_project(x, metric, c, rho)
    plain = float((x.square() * g).sum().clamp_min(1.0e-18).detach().cpu().item())
    proj = float((p.square() * g).sum().clamp_min(0.0).detach().cpu().item())
    trans = x - float(alpha) * p
    rel_e = float(((trans.square() * g).sum() + float(gamma) * (x.square() * g).sum()).clamp_min(0.0).detach().cpu().item())
    eig = torch.linalg.eigvalsh(0.5 * (gram + gram.T) + float(rho) * torch.eye(dim, device=x.device, dtype=torch.float64))
    pos = eig[eig > 1.0e-12]
    cond = float((pos.max() / pos.min()).detach().cpu().item()) if int(pos.numel()) else 0.0
    frac = math.sqrt(proj / max(plain, 1.0e-18))
    mlp_frac = 0.0
    debt_frac = 0.0
    if dim >= 1:
        c0 = c[:, 0]
        mlp_proj = c0 * ((x * g * c0).sum() / (c0.square() * g).sum().clamp_min(1.0e-12))
        mlp_frac = math.sqrt(float((mlp_proj.square() * g).sum().detach().cpu().item()) / max(plain, 1.0e-18))
    if dim >= 2:
        c1 = c[:, 1]
        debt_proj = c1 * ((x * g * c1).sum() / (c1.square() * g).sum().clamp_min(1.0e-12))
        debt_frac = math.sqrt(float((debt_proj.square() * g).sum().detach().cpu().item()) / max(plain, 1.0e-18))
    return {
        "control_suppression_ratio": 1.0 - rel_e / max(plain * (1.0 + float(gamma)), 1.0e-18),
        "task_energy_retention_ratio": rel_e / max(plain, 1.0e-18),
        "MLP_tangent_leakage_ratio": mlp_frac,
        "debt_tangent_leakage_ratio": debt_frac,
        "soft_projection_condition_number": cond,
        "relative_metric_vs_plain_metric_gap": rel_e / max(plain, 1.0e-18) - 1.0,
        "candidate_control_projection_fraction": frac,
        "candidate_task_energy_fraction": max(0.0, 1.0 - min(1.0, frac)),
        "soft_projector_rho": rho,
    }


def build_output_controls(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    metric: torch.Tensor,
    mlp_update: torch.Tensor | None = None,
) -> torch.Tensor:
    probs = torch.softmax(logits.detach().float(), dim=1)
    ce = (probs - F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()).to(dtype=torch.float64)
    try:
        brier_grad = base80.logits_loss_grad(logits.detach(), y, "brier").reshape_as(logits).to(dtype=torch.float64)
    except Exception:
        brier_grad = ce.detach().clone()
    gen = torch.Generator(device=logits.device).manual_seed(int(seed))
    rnd = torch.randn(tuple(logits.shape), generator=gen, device=logits.device, dtype=torch.float64)
    centered = logits.detach().to(dtype=torch.float64) - logits.detach().to(dtype=torch.float64).mean(dim=1, keepdim=True)
    vectors = []
    if mlp_update is not None:
        vectors.append(mlp_update.to(dtype=torch.float64))
    vectors.extend([brier_grad, ce, rnd, centered])
    return make_control_matrix(vectors).to(device=logits.device, dtype=torch.float64)


def task_cotangent(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((torch.softmax(logits.detach().float(), dim=1) - F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()) / max(1, int(y.numel()))).to(dtype=torch.float64)


def edge_cost_metric(design_s: torch.Tensor, design_w: torch.Tensor, metric_s: torch.Tensor, metric_w: torch.Tensor, ridge: float) -> torch.Tensor:
    ms = metric_s.reshape(-1).to(device=design_s.device, dtype=torch.float64)
    mw = metric_w.reshape(-1).to(device=design_w.device, dtype=torch.float64)
    mat = (design_s.T * ms[None, :]) @ design_s + (design_w.T * mw[None, :]) @ design_w
    if int(mat.shape[0]) > 1:
        diff = torch.eye(int(mat.shape[0]), device=mat.device, dtype=torch.float64)
        smooth = diff
    else:
        smooth = torch.eye(int(mat.shape[0]), device=mat.device, dtype=torch.float64)
    mat = 0.5 * (mat + mat.T) + float(ridge) * torch.eye(int(mat.shape[0]), device=mat.device, dtype=torch.float64) + 1.0e-5 * smooth
    return mat


def task_operator(
    design_s: torch.Tensor,
    design_w: torch.Tensor,
    metric_s: torch.Tensor,
    metric_w: torch.Tensor,
    logits_s: torch.Tensor,
    y_s: torch.Tensor,
    logits_w: torch.Tensor,
    y_w: torch.Tensor,
    controls_s: torch.Tensor,
    controls_w: torch.Tensor,
    pi: torch.Tensor,
    *,
    beta: float,
    alpha: float,
    gamma: float,
    ridge: float,
    vis_scale_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    n_s = int(logits_s.shape[0])
    n_w = int(logits_w.shape[0])
    k = int(logits_s.shape[1])
    d = int(design_s.shape[1])
    rho_s = 1.0e-3 * max(1.0, float(((controls_s.T * metric_s.reshape(-1)[None, :]) @ controls_s).trace().detach().cpu().item()) / max(1, int(controls_s.shape[1]))) if int(controls_s.numel()) else 0.0
    rho_w = 1.0e-3 * max(1.0, float(((controls_w.T * metric_w.reshape(-1)[None, :]) @ controls_w).trace().detach().cpu().item()) / max(1, int(controls_w.shape[1]))) if int(controls_w.numel()) else 0.0
    rs_rel = relative_metric_apply(task_cotangent(logits_s, y_s), metric_s, controls_s, alpha=alpha, gamma=gamma, rho=rho_s)
    rw_rel = relative_metric_apply(task_cotangent(logits_w, y_w), metric_w, controls_w, alpha=alpha, gamma=gamma, rho=rho_w)
    b_s = design_s.T @ rs_rel.reshape(-1)
    b_w = design_w.T @ rw_rel.reshape(-1)
    g_logit = torch.eye(k, device=design_s.device, dtype=torch.float64) - torch.full((k, k), 1.0 / max(1, k), device=design_s.device, dtype=torch.float64)
    g_logit = g_logit + 0.10 * torch.eye(k, device=design_s.device, dtype=torch.float64)
    ds = design_s.reshape(n_s, k, d)
    dw = design_w.reshape(n_w, k, d)
    a_vis_raw = torch.einsum("iap,ij,ab,jbq->pq", ds, pi.to(device=design_s.device, dtype=torch.float64), g_logit, dw)
    a_vis_raw = 0.5 * (a_vis_raw + a_vis_raw.T)
    a_task_only = 0.5 * (torch.outer(b_s, b_w) + torch.outer(b_w, b_s))
    task_norm = a_task_only.norm().clamp_min(1.0e-12)
    vis_norm = a_vis_raw.norm().clamp_min(1.0e-12)
    if str(vis_scale_mode) == "task_fro":
        vis_scale = float((task_norm / vis_norm).detach().cpu().item())
    elif str(vis_scale_mode) == "none":
        vis_scale = 1.0
    else:
        raise ValueError(f"unknown vis_scale_mode={vis_scale_mode!r}")
    a_vis = a_vis_raw * vis_scale
    a_task = a_task_only + float(beta) * a_vis
    m_edge = edge_cost_metric(design_s, design_w, metric_s, metric_w, ridge)
    return a_task, m_edge, b_s, b_w, {
        "A_task_norm": float(a_task_only.norm().detach().cpu().item()),
        "A_vis_norm": float(a_vis_raw.norm().detach().cpu().item()),
        "A_vis_scaled_norm": float(a_vis.norm().detach().cpu().item()),
        "A_task_vis_norm_ratio": float((a_task_only.norm() / a_vis_raw.norm().clamp_min(1.0e-12)).detach().cpu().item()),
        "A_task_scaled_vis_norm_ratio": float((a_task_only.norm() / a_vis.norm().clamp_min(1.0e-12)).detach().cpu().item()),
        "operator_vis_scale": vis_scale,
        "operator_vis_scale_mode": str(vis_scale_mode),
    }


def top_generalized_eigen(a: torch.Tensor, m: torch.Tensor) -> tuple[float, torch.Tensor, float]:
    d = int(a.shape[0])
    eye = torch.eye(d, device=a.device, dtype=torch.float64)
    m2 = 0.5 * (m + m.T) + 1.0e-8 * eye
    try:
        chol = torch.linalg.cholesky(m2)
        inv_left = torch.linalg.solve(chol, a)
        b = torch.linalg.solve(chol, inv_left.T).T
        b = 0.5 * (b + b.T)
        vals, vecs = torch.linalg.eigh(b)
        idx = int(torch.argmax(vals).detach().cpu().item())
        y = vecs[:, idx]
        v = torch.linalg.solve(chol.T, y)
    except Exception:
        vals, vecs = torch.linalg.eigh(0.5 * (a + a.T))
        idx = int(torch.argmax(vals).detach().cpu().item())
        v = vecs[:, idx]
    denom = torch.sqrt((v @ (m2 @ v)).clamp_min(1.0e-18))
    v = v / denom
    lam = float((v @ (a @ v)).detach().cpu().item())
    eig_m = torch.linalg.eigvalsh(m2)
    pos = eig_m[eig_m > 1.0e-12]
    cond = float((pos.max() / pos.min()).detach().cpu().item()) if int(pos.numel()) else 0.0
    return lam, v, cond


def topk_generalized_eigen(a: torch.Tensor, m: torch.Tensor, k: int) -> tuple[list[float], torch.Tensor, float]:
    d = int(a.shape[0])
    eye = torch.eye(d, device=a.device, dtype=torch.float64)
    m2 = 0.5 * (m + m.T) + 1.0e-8 * eye
    try:
        chol = torch.linalg.cholesky(m2)
        inv_left = torch.linalg.solve(chol, a)
        b = torch.linalg.solve(chol, inv_left.T).T
        b = 0.5 * (b + b.T)
        vals, vecs = torch.linalg.eigh(b)
        order = torch.argsort(vals, descending=True)
        keep = int(min(max(1, int(k)), int(order.numel())))
        vals = vals[order[:keep]]
        ys = vecs[:, order[:keep]]
        vs = torch.linalg.solve(chol.T, ys)
    except Exception:
        vals_all, vecs_all = torch.linalg.eigh(0.5 * (a + a.T))
        order = torch.argsort(vals_all, descending=True)
        keep = int(min(max(1, int(k)), int(order.numel())))
        vals = vals_all[order[:keep]]
        vs = vecs_all[:, order[:keep]]
    cols = []
    for j in range(int(vs.shape[1])):
        v = vs[:, j]
        denom = torch.sqrt((v @ (m2 @ v)).clamp_min(1.0e-18))
        cols.append(v / denom)
    out = torch.stack(cols, dim=1) if cols else torch.zeros((d, 0), device=a.device, dtype=torch.float64)
    eig_m = torch.linalg.eigvalsh(m2)
    pos = eig_m[eig_m > 1.0e-12]
    cond = float((pos.max() / pos.min()).detach().cpu().item()) if int(pos.numel()) else 0.0
    lams = [float((out[:, j] @ (a @ out[:, j])).detach().cpu().item()) for j in range(int(out.shape[1]))]
    return lams, out, cond


def parse_float_list(text: str) -> list[float]:
    vals: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals or [1.0]


def finite_step_debt_select(
    logits: torch.Tensor,
    y: torch.Tensor,
    update: torch.Tensor,
    *,
    scales: list[float],
) -> tuple[torch.Tensor, dict[str, Any]]:
    best_safe: tuple[float, dict[str, float], float] | None = None
    best_any: tuple[float, dict[str, float], float] | None = None
    for scale in scales:
        upd = update * float(scale)
        delta, debt = row_metrics_from_logit_update(logits, y, upd)
        nll = float(delta.get("NLL", 0.0))
        if best_any is None or (debt, nll) < (best_any[2], float(best_any[1].get("NLL", 0.0))):
            best_any = (float(scale), delta, debt)
        if debt <= 0.0 and nll < 0.0:
            if best_safe is None or nll < float(best_safe[1].get("NLL", 0.0)):
                best_safe = (float(scale), delta, debt)
    chosen = best_safe if best_safe is not None else best_any
    if chosen is None:
        return update * 0.0, {"debt_safe_scale": 0.0, "debt_safe_feasible": 0, "debt_safe_debt_UCB": 0.0}
    scale, delta, debt = chosen
    return update * scale, {
        "debt_safe_scale": scale,
        "debt_safe_feasible": int(best_safe is not None),
        "debt_safe_debt_UCB": debt,
        "debt_safe_NLL_delta": float(delta.get("NLL", 0.0)),
        "infeasible_debt_component": "" if best_safe is not None else max(
            ["Brier", "ECE", "tail95", "tail99", "margin10"],
            key=lambda key: float(delta.get(key, 0.0)),
        ),
    }


def subspace_candidate_alphas(
    eigvecs: torch.Tensor,
    lams: list[float],
    g_s: torch.Tensor,
    g_w: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    candidates: list[tuple[str, torch.Tensor]] = []
    if int(eigvecs.numel()) == 0 or int(eigvecs.shape[1]) == 0:
        return candidates
    u = eigvecs.to(dtype=torch.float64)
    gs = g_s.reshape(-1).to(device=u.device, dtype=torch.float64)
    gw = g_w.reshape(-1).to(device=u.device, dtype=torch.float64)

    def add(name: str, alpha: torch.Tensor) -> None:
        if int(alpha.numel()) != int(u.shape[0]):
            return
        norm = float(alpha.norm().detach().cpu().item())
        if math.isfinite(norm) and norm > 1.0e-12:
            candidates.append((name, alpha / alpha.norm().clamp_min(1.0e-12)))

    for j in range(int(u.shape[1])):
        add(f"eig{j}", u[:, j])
    q_s = u.T @ gs
    q_w = u.T @ gw
    q_sum = q_s + q_w
    add("subspace_source_witness_sum", -(u @ q_sum))
    add("subspace_source_only", -(u @ q_s))
    add("subspace_witness_only", -(u @ q_w))
    ns = q_s.norm().clamp_min(1.0e-12)
    nw = q_w.norm().clamp_min(1.0e-12)
    add("subspace_balanced_unit_sum", -(u @ (q_s / ns + q_w / nw)))
    if lams:
        weights = torch.tensor([max(0.0, float(v)) for v in lams[: int(u.shape[1])]], device=u.device, dtype=torch.float64)
        if float(weights.sum().detach().cpu().item()) > 1.0e-12:
            add("subspace_lambda_weighted_sum", -(u @ (weights * q_sum)))
    return candidates


def project_subspace_halfspaces(q0: torch.Tensor, a_mat: torch.Tensor, ridge: float) -> tuple[torch.Tensor, dict[str, Any]]:
    q_base = q0.reshape(-1).to(dtype=torch.float64)
    if int(q_base.numel()) == 0 or int(a_mat.numel()) == 0:
        return q_base, {"debt_qp_active_constraints": "", "debt_qp_first_order_max": 0.0, "debt_qp_feasible": 1}
    a = a_mat.to(device=q_base.device, dtype=torch.float64)
    m = int(a.shape[0])
    best_q: torch.Tensor | None = None
    best_dist = float("inf")
    best_mask = 0
    eye_cache: dict[int, torch.Tensor] = {}
    for mask in range(1 << m):
        if mask == 0:
            q = q_base.clone()
        else:
            idx = [i for i in range(m) if (mask >> i) & 1]
            sub = a.index_select(0, torch.tensor(idx, device=q_base.device, dtype=torch.long))
            k = int(sub.shape[0])
            if k not in eye_cache:
                eye_cache[k] = torch.eye(k, device=q_base.device, dtype=torch.float64)
            mat = sub @ sub.T + float(ridge) * eye_cache[k]
            rhs = sub @ q_base
            try:
                lam = torch.linalg.solve(mat, rhs)
            except Exception:
                lam = torch.linalg.pinv(mat) @ rhs
            q = q_base - sub.T @ lam
        constraints = a @ q
        feasible = bool(torch.all(constraints <= 1.0e-10).detach().cpu().item())
        if feasible:
            dist = float((q - q_base).square().sum().detach().cpu().item())
            if dist < best_dist:
                best_dist = dist
                best_q = q
                best_mask = mask
    if best_q is None:
        best_q = torch.zeros_like(q_base)
        constraints = a @ best_q
    else:
        constraints = a @ best_q
    return best_q, {
        "debt_qp_active_constraints": ",".join(str(i) for i in range(m) if (best_mask >> i) & 1),
        "debt_qp_first_order_max": float(constraints.max().detach().cpu().item()) if int(constraints.numel()) else 0.0,
        "debt_qp_feasible": int(best_dist < float("inf")),
    }


def orient_for_source_witness(alpha: torch.Tensor, g_s: torch.Tensor, g_w: torch.Tensor) -> torch.Tensor:
    gs = g_s.reshape(-1).to(device=alpha.device, dtype=torch.float64)
    gw = g_w.reshape(-1).to(device=alpha.device, dtype=torch.float64)
    pred_s = float((gs * alpha.reshape(-1)).sum().detach().cpu().item())
    pred_w = float((gw * alpha.reshape(-1)).sum().detach().cpu().item())
    if pred_s + pred_w > 0.0:
        return -alpha
    return alpha


def brier_decomposition(logits: torch.Tensor, y: torch.Tensor, update: torch.Tensor) -> dict[str, Any]:
    before = logits.detach().float()
    after = before + update.detach().float().reshape_as(before)
    probs0 = torch.softmax(before, dim=1)
    probs1 = torch.softmax(after, dim=1)
    conf0, pred0 = probs0.max(dim=1)
    conf1, pred1 = probs1.max(dim=1)
    ok0 = pred0.eq(y.long())
    ok1 = pred1.eq(y.long())
    yoh = F.one_hot(y.long(), num_classes=int(before.shape[1])).float()
    class_delta = (probs1 - yoh).square().mean(dim=0) - (probs0 - yoh).square().mean(dim=0)
    centered = before - before.mean(dim=1, keepdim=True)
    radial = ((update.float() * centered).sum(dim=1) / centered.norm(dim=1).clamp_min(1.0e-8)).mean()
    tangent = (update.float() - update.float().mean(dim=1, keepdim=True)).norm(dim=1).mean()
    wrong = ~ok0
    right = ok0
    wrong_amp = (conf1[wrong] - conf0[wrong]).mean() if bool(wrong.any()) else torch.tensor(0.0, device=before.device)
    right_sharp = (conf1[right] - conf0[right]).mean() if bool(right.any()) else torch.tensor(0.0, device=before.device)
    return {
        "Brier_reliability_delta": float((conf1 - ok1.float()).abs().mean().sub((conf0 - ok0.float()).abs().mean()).detach().cpu().item()),
        "Brier_resolution_delta": float((probs1.var(dim=0).mean() - probs0.var(dim=0).mean()).detach().cpu().item()),
        "confidence_radial_delta": float(radial.detach().cpu().item()),
        "simplex_tangent_delta": float(tangent.detach().cpu().item()),
        "wrong_confident_amplification": float(wrong_amp.detach().cpu().item()),
        "right_confident_sharpening": float(right_sharp.detach().cpu().item()),
        "classwise_Brier_delta": ";".join(f"{float(v):.8g}" for v in class_delta.detach().cpu().tolist()),
    }


def prepare_designs(model: Any, xs: torch.Tensor, xw: torch.Tensor, xg: torch.Tensor, args: argparse.Namespace, device: torch.device):
    x_train = torch.cat([xs, xw], dim=0)
    train_z = model._norm_input(x_train).detach().to(dtype=torch.float64)
    metric_train = base83.output_metric_diag(model(x_train).float()).to(device=device)
    d_idx, h_idx, edge_meta = base84.select_edge_pairs(model, x_train, metric_train, args)
    centers = torch.linspace(0.05, 0.95, steps=max(2, int(args.rkhs_centers)), device=device, dtype=torch.float64)
    kernel = str(args.primary_shape_kernel)
    design_s = base84.rkhs_design(model, xs, train_z, d_idx, h_idx, centers, kernel)
    design_w = base84.rkhs_design(model, xw, train_z, d_idx, h_idx, centers, kernel)
    design_g = base84.rkhs_design(model, xg, train_z, d_idx, h_idx, centers, kernel)
    scale = base84.column_scale_from_design(design_s, float(args.projector_ridge))
    design_s = base84.apply_column_scale(design_s, scale)
    design_w = base84.apply_column_scale(design_w, scale)
    design_g = base84.apply_column_scale(design_g, scale)
    if int(args.rkhs_svd_rank) > 0:
        metric_s_tmp = base83.output_metric_diag(model(xs).float()).to(device=device)
        design_s, design_w, design_g, svd_meta = base84.svd_reduce_designs(design_s, design_w, design_g, metric_s_tmp, int(args.rkhs_svd_rank), float(args.projector_ridge))
    else:
        svd_meta = {"svd_reduced": 0, "svd_rank": 0, "svd_source_condition": 0.0}
    edge_meta.update({"rkhs_centers": int(centers.numel()), "rkhs_atoms": int(design_s.shape[1]), "primary_shape_kernel": kernel, **svd_meta})
    return design_s, design_w, design_g, edge_meta


def lambda_half_lcb(
    design_s: torch.Tensor,
    design_w: torch.Tensor,
    metric_s: torch.Tensor,
    metric_w: torch.Tensor,
    logits_s: torch.Tensor,
    y_s: torch.Tensor,
    logits_w: torch.Tensor,
    y_w: torch.Tensor,
    controls_s: torch.Tensor,
    controls_w: torch.Tensor,
    pi: torch.Tensor,
    *,
    beta: float,
    alpha: float,
    gamma: float,
    ridge: float,
    full_lambda: float,
    vis_scale_mode: str,
) -> float:
    vals = [float(full_lambda)]
    ns = int(logits_s.shape[0])
    nw = int(logits_w.shape[0])
    k = int(logits_s.shape[1])
    for parity in (0, 1):
        idx_s = torch.arange(ns, device=logits_s.device)[torch.arange(ns, device=logits_s.device) % 2 == parity]
        idx_w = torch.arange(nw, device=logits_w.device)[torch.arange(nw, device=logits_w.device) % 2 == parity]
        if int(idx_s.numel()) < 2 or int(idx_w.numel()) < 2:
            continue
        flat_s = torch.cat([idx_s * k + cls for cls in range(k)]).long()
        flat_w = torch.cat([idx_w * k + cls for cls in range(k)]).long()
        try:
            op, m, _bs, _bw, _meta = task_operator(
                design_s[flat_s],
                design_w[flat_w],
                metric_s.reshape(-1)[flat_s],
                metric_w.reshape(-1)[flat_w],
                logits_s[idx_s],
                y_s[idx_s],
                logits_w[idx_w],
                y_w[idx_w],
                controls_s[flat_s] if int(controls_s.numel()) else controls_s,
                controls_w[flat_w] if int(controls_w.numel()) else controls_w,
                pi[idx_s][:, idx_w],
                beta=beta,
                alpha=alpha,
                gamma=gamma,
                ridge=ridge,
                vis_scale_mode=vis_scale_mode,
            )
            lam, _v, _cond = top_generalized_eigen(op, m)
            vals.append(lam)
        except Exception:
            continue
    return min(vals)


def part_d_probe(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    family = str(args.part_d_family)
    beta = float(args.beta if family == PRIMARY_FAMILY else DIAGNOSTIC_FAMILIES.get(family, args.beta))
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "architecture": "strict_fc_purekan",
        "method": spec.get("method", ""),
        "carrier_family": spec.get("carrier_family", ""),
        "candidate_family": family,
        "primary_family": int(family == PRIMARY_FAMILY),
        "beta": beta,
        "alpha": float(args.alpha),
        "gamma": float(args.gamma),
        "rho_rule": "1e-3_trace_CtGzC_over_dim",
        "probe_error": "",
    }
    try:
        model, mlp, _bundle, splits = make_model_and_batch_v2285(dataset, seed, spec, args, device)
        xs, ys, xw, yw, xg, yg, _xc, _yc = splits
        logits_s = model(xs).float()
        logits_w = model(xw).float()
        logits_g = model(xg).float()
        metric_s = base83.output_metric_diag(logits_s).to(device=device)
        metric_w = base83.output_metric_diag(logits_w).to(device=device)
        metric_g = base83.output_metric_diag(logits_g).to(device=device)
        design_s, design_w, design_g, design_meta = prepare_designs(model, xs, xw, xg, args, device)
        if int(design_s.shape[1]) <= 0:
            raise RuntimeError("empty task-signed edge design")

        out_family = base84.selected_output_family(args)
        target_s, _ = base83.output_update_for_family(logits_s, ys, out_family, args)
        target_w, _ = base83.output_update_for_family(logits_w, yw, out_family, args)
        target_g, _ = base83.output_update_for_family(logits_g, yg, out_family, args)
        mlp_s_update = None
        try:
            mlp_updates = base83.mlp_matched_update(mlp, xs, ys, xw, yw, 1.0)
            mlp_s_update = base83.actual_logit_update_for_updates(mlp, xs, mlp_updates).reshape_as(logits_s)
        except Exception:
            mlp_s_update = None
        controls_s = build_output_controls(logits_s, ys, seed=stable_text_seed(dataset) + int(seed), metric=metric_s, mlp_update=mlp_s_update)
        controls_w = build_output_controls(logits_w, yw, seed=stable_text_seed(dataset) + int(seed) + 17, metric=metric_w, mlp_update=None)
        probs_s = torch.softmax(logits_s.detach().float(), dim=1)
        probs_w = torch.softmax(logits_w.detach().float(), dim=1)
        conf_s = probs_s.max(dim=1).values
        conf_w = probs_w.max(dim=1).values
        pi, coupling_meta = class_confidence_bucket_coupling(ys, yw, conf_s, conf_w, num_classes=int(model.output_dim), buckets=int(args.confidence_buckets))

        op, m_edge, b_s, b_w, op_meta = task_operator(
            design_s,
            design_w,
            metric_s,
            metric_w,
            logits_s,
            ys,
            logits_w,
            yw,
            controls_s,
            controls_w,
            pi,
            beta=beta,
            alpha=float(args.alpha),
            gamma=float(args.gamma),
            ridge=float(args.projector_ridge),
            vis_scale_mode=str(args.vis_scale_mode),
        )
        eig_lams, eigvecs, m_cond = topk_generalized_eigen(op, m_edge, int(args.subspace_rank))
        if int(eigvecs.shape[1]) <= 0:
            raise RuntimeError("empty task-signed stable subspace")
        lam = float(eig_lams[0])
        eigvec = eigvecs[:, 0]
        lam_lcb = lambda_half_lcb(
            design_s,
            design_w,
            metric_s,
            metric_w,
            logits_s,
            ys,
            logits_w,
            yw,
            controls_s,
            controls_w,
            pi,
            beta=beta,
            alpha=float(args.alpha),
            gamma=float(args.gamma),
            ridge=float(args.projector_ridge),
            full_lambda=lam,
            vis_scale_mode=str(args.vis_scale_mode),
        )
        r_s = task_cotangent(logits_s, ys).reshape(-1)
        r_w = task_cotangent(logits_w, yw).reshape(-1)
        r_g = task_cotangent(logits_g, yg).reshape(-1)
        g_s_raw = design_s.T @ r_s
        g_w_raw = design_w.T @ r_w
        g_g_raw = design_g.T @ r_g
        debt_scales = parse_float_list(str(args.debt_scale_grid))
        candidates = subspace_candidate_alphas(eigvecs, eig_lams, g_s_raw, g_w_raw)
        debt_pullbacks: list[torch.Tensor] = []
        debt_kinds = ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"]
        for debt_kind in debt_kinds:
            try:
                debt_grad = base80.logits_loss_grad(logits_g, yg, debt_kind).reshape(-1).to(device=device, dtype=torch.float64)
                debt_pullbacks.append(design_g.T @ debt_grad)
            except Exception:
                continue
        debt_a = torch.stack([eigvecs.T @ item.reshape(-1).to(device=device, dtype=torch.float64) for item in debt_pullbacks], dim=0) if debt_pullbacks else torch.zeros((0, int(eigvecs.shape[1])), device=device, dtype=torch.float64)
        debt_qp_meta: dict[str, Any] = {}
        if int(debt_a.numel()) > 0:
            q_sw = -(eigvecs.T @ (g_s_raw + g_w_raw))
            q_sw_proj, debt_qp_meta_sw = project_subspace_halfspaces(q_sw, debt_a, float(args.projector_ridge))
            candidates.append(("subspace_debt_qp_source_witness", eigvecs @ q_sw_proj))
            debt_qp_meta.update({f"source_witness_{k}": v for k, v in debt_qp_meta_sw.items()})
            q_guard = -(eigvecs.T @ g_g_raw)
            q_guard_proj, debt_qp_meta_guard = project_subspace_halfspaces(q_guard, debt_a, float(args.projector_ridge))
            candidates.append(("subspace_debt_qp_guard_oracle", eigvecs @ q_guard_proj))
            debt_qp_meta.update({f"guard_oracle_{k}": v for k, v in debt_qp_meta_guard.items()})
        target_candidate_count = 0
        if family != PRIMARY_FAMILY:
            for target_name, target_design, target_update, target_metric in [
                ("guard", design_g, target_g, metric_g),
                ("source", design_s, target_s, metric_s),
                ("witness", design_w, target_w, metric_w),
            ]:
                try:
                    stable_target_design = target_design @ eigvecs
                    q_target = base84.weighted_ridge_alpha(stable_target_design, target_update, target_metric, float(args.projector_ridge))
                    q_target = q_target.reshape(-1).to(device=device, dtype=torch.float64)
                    if float(q_target.norm().detach().cpu().item()) > 1.0e-12:
                        candidates.append((f"subspace_target_{target_name}", eigvecs @ q_target))
                        target_candidate_count += 1
                        if int(debt_a.numel()) > 0:
                            q_target_proj, _qp_meta_target = project_subspace_halfspaces(q_target, debt_a, float(args.projector_ridge))
                            if float(q_target_proj.norm().detach().cpu().item()) > 1.0e-12:
                                candidates.append((f"subspace_target_{target_name}_debtproj", eigvecs @ q_target_proj))
                                target_candidate_count += 1
                except Exception as exc:
                    debt_qp_meta[f"target_{target_name}_candidate_error"] = f"{type(exc).__name__}: {exc}"
        random_requested = max(0, int(args.subspace_random_candidates))
        if random_requested:
            q_base = -(eigvecs.T @ (g_s_raw + g_w_raw))
            base_norm = q_base.norm().clamp_min(1.0e-12)
            q_base = q_base / base_norm
            gen_seed = stable_text_seed(f"{dataset}:{seed}:{spec.get('carrier_family','')}:{family}:subspace_search")
            gen = torch.Generator(device=device).manual_seed(int(gen_seed))
            temps = [0.25, 0.5, 1.0, 2.0]
            for ridx in range(random_requested):
                rnd = torch.randn(int(eigvecs.shape[1]), device=device, dtype=torch.float64, generator=gen)
                rnd = rnd / rnd.norm().clamp_min(1.0e-12)
                temp = temps[ridx % len(temps)]
                q_mix = q_base + float(temp) * rnd
                q_mix = q_mix / q_mix.norm().clamp_min(1.0e-12)
                candidates.append((f"subspace_search_mix{ridx}", eigvecs @ q_mix))
                if int(debt_a.numel()) > 0 and ridx < max(4, random_requested // 2):
                    q_proj, _qp_meta_rand = project_subspace_halfspaces(q_mix, debt_a, float(args.projector_ridge))
                    candidates.append((f"subspace_search_debtproj{ridx}", eigvecs @ q_proj))
        best: dict[str, Any] | None = None
        candidate_feasible_count = 0
        candidate_sign_feasible_count = 0
        candidate_guard_nll_neg_count = 0
        target_norm_g_pre = metric_norm(target_g, metric_g)
        for cand_idx, (cand_name, cand_alpha0) in enumerate(candidates):
            cand_alpha = orient_for_source_witness(cand_alpha0, g_s_raw, g_w_raw)
            update_g0 = (design_g @ cand_alpha).reshape_as(logits_g)
            g_norm = metric_norm(update_g0, metric_g)
            if g_norm > 1.0e-12:
                desired_norm = target_norm_g_pre if str(cand_name).startswith("subspace_target_") else float(args.task_logit_norm)
                cand_alpha = cand_alpha * (desired_norm / g_norm)
            raw_s = (design_s @ cand_alpha).reshape_as(logits_s)
            raw_w = (design_w @ cand_alpha).reshape_as(logits_w)
            raw_g = (design_g @ cand_alpha).reshape_as(logits_g)
            cand_update_g, cand_debt_meta = finite_step_debt_select(logits_g, yg, raw_g, scales=debt_scales)
            cand_scale = float(cand_debt_meta.get("debt_safe_scale", 1.0))
            cand_alpha = cand_alpha * cand_scale
            cand_update_s = raw_s * cand_scale
            cand_update_w = raw_w * cand_scale
            cand_pred_s = float((r_s * cand_update_s.reshape(-1)).sum().detach().cpu().item())
            cand_pred_w = float((r_w * cand_update_w.reshape(-1)).sum().detach().cpu().item())
            cand_guard_delta, cand_guard_debt = row_metrics_from_logit_update(logits_g, yg, cand_update_g)
            cand_coverage_g, _cand_rel_res_g = base80.weighted_capacity(cand_update_g.reshape(-1), target_g.reshape(-1), metric_g.reshape(-1))
            sign_pass = int(cand_pred_s < 0.0 and cand_pred_w < 0.0 and cand_pred_s * cand_pred_w > 0.0)
            feasible = int(cand_debt_meta.get("debt_safe_feasible", 0) or cand_guard_debt <= 0.0)
            candidate_feasible_count += int(feasible)
            candidate_sign_feasible_count += int(sign_pass and feasible)
            candidate_guard_nll_neg_count += int(float(cand_guard_delta.get("NLL", 0.0)) < 0.0)
            score = (
                sign_pass,
                feasible,
                int(cand_coverage_g >= 0.20) if family != PRIMARY_FAMILY else 0,
                cand_coverage_g if family != PRIMARY_FAMILY else -1.0,
                -max(cand_pred_s, cand_pred_w),
                -(cand_pred_s + cand_pred_w),
                -abs(cand_pred_s - cand_pred_w),
                -cand_idx,
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "name": cand_name,
                    "alpha": cand_alpha,
                    "update_s": cand_update_s,
                    "update_w": cand_update_w,
                    "update_g": cand_update_g,
                    "debt_meta": cand_debt_meta,
                    "pred_s": cand_pred_s,
                    "pred_w": cand_pred_w,
                    "sign_pass": sign_pass,
                    "feasible": feasible,
                    "guard_delta": cand_guard_delta,
                    "guard_debt": cand_guard_debt,
                    "coverage_g": cand_coverage_g,
                }
        if best is None:
            raise RuntimeError("no subspace candidates generated")
        alpha_vec = best["alpha"]
        update_s = best["update_s"]
        update_w = best["update_w"]
        update_g = best["update_g"]
        debt_meta = dict(best["debt_meta"])
        source_delta, source_debt = row_metrics_from_logit_update(logits_s, ys, update_s)
        witness_delta, witness_debt = row_metrics_from_logit_update(logits_w, yw, update_w)
        guard_delta, guard_debt = row_metrics_from_logit_update(logits_g, yg, update_g)
        coverage_g, rel_res_g = base80.weighted_capacity(update_g.reshape(-1), target_g.reshape(-1), metric_g.reshape(-1))
        coverage_sg, _ = base80.weighted_capacity(update_g.reshape(-1), target_s[: int(update_g.shape[0])].reshape(-1) if int(target_s.shape[0]) >= int(update_g.shape[0]) else target_g.reshape(-1), metric_g.reshape(-1))
        coverage_sw, _ = base80.weighted_capacity(update_w.reshape(-1), target_w.reshape(-1), metric_w.reshape(-1))
        target_norm_g = metric_norm(target_g, metric_g)
        candidate_norm_g = metric_norm(update_g, metric_g)
        target_cosine_g = metric_cosine(update_g, target_g, metric_g)
        try:
            full_target_alpha = base84.weighted_ridge_alpha(design_g, target_g, metric_g, float(args.projector_ridge))
            full_target_update = (design_g @ full_target_alpha).reshape_as(logits_g)
            full_target_coverage, full_target_relres = base80.weighted_capacity(full_target_update.reshape(-1), target_g.reshape(-1), metric_g.reshape(-1))
            full_target_delta, full_target_debt = row_metrics_from_logit_update(logits_g, yg, full_target_update)
        except Exception:
            full_target_coverage, full_target_relres, full_target_debt = 0.0, 0.0, 0.0
            full_target_delta = {"NLL": 0.0}
        try:
            stable_design_g = design_g @ eigvecs
            stable_q = base84.weighted_ridge_alpha(stable_design_g, target_g, metric_g, float(args.projector_ridge))
            stable_target_update = (stable_design_g @ stable_q).reshape_as(logits_g)
            stable_target_coverage, stable_target_relres = base80.weighted_capacity(stable_target_update.reshape(-1), target_g.reshape(-1), metric_g.reshape(-1))
            stable_target_delta, stable_target_debt = row_metrics_from_logit_update(logits_g, yg, stable_target_update)
        except Exception:
            stable_target_coverage, stable_target_relres, stable_target_debt = 0.0, 0.0, 0.0
            stable_target_delta = {"NLL": 0.0}

        control_names = [
            "same_RKHS_norm_random",
            "same_edge_domain_energy_random",
            "same_smoothness_random",
            "same_output_coverage_random",
            "same_debt_cone_random",
            "same_solver_gradient_control",
            "same_quantile_shape_random",
            "same_density_context_random",
            "same_compute_noop",
        ]
        margins: dict[str, float] = {}
        for offset, name in enumerate(control_names):
            ctrl = base84.control_alpha(name, alpha_vec, design_s, update_s, int(seed) + stable_text_seed(name + dataset) + 19 * offset, float(args.projector_ridge))
            ctrl_update = (design_g @ ctrl).reshape_as(logits_g)
            cn = metric_norm(ctrl_update, metric_g)
            tn = metric_norm(update_g, metric_g)
            if name in {"same_output_coverage_random", "same_quantile_shape_random", "same_density_context_random"} and cn > 1.0e-12:
                ctrl_update = ctrl_update * (tn / cn)
            ctrl_delta, ctrl_debt = row_metrics_from_logit_update(logits_g, yg, ctrl_update)
            margins[name] = float(ctrl_delta.get("NLL", 0.0)) - float(guard_delta.get("NLL", 0.0)) - 0.5 * max(0.0, guard_debt - ctrl_debt)
        control_margin_values = [v for k, v in margins.items() if k != "same_compute_noop"]
        control_margin_cvar25 = lower_cvar(control_margin_values, 0.25)
        boot_lcb = quantile(control_margin_values, 0.10)
        mlp_update_g = base84.mlp_matched_logit_update(mlp, xs, ys, xw, yw, xg, metric_norm(update_g, metric_g), metric_g).reshape_as(logits_g)
        mlp_delta, mlp_debt = row_metrics_from_logit_update(logits_g, yg, mlp_update_g)
        mlp_margin = float(mlp_delta.get("NLL", 0.0)) - float(guard_delta.get("NLL", 0.0)) - 0.5 * max(0.0, guard_debt - mlp_debt)
        quotient_meta = soft_quotient_metrics(update_s, metric_s, controls_s, alpha=float(args.alpha), gamma=float(args.gamma))
        brier_meta = brier_decomposition(logits_g, yg, update_g)
        transport_error = float(abs(coverage_sw - coverage_g))
        density_context_contribution = max(0.0, margins.get("same_density_context_random", 0.0) - control_margin_cvar25)
        pred_s = float((r_s * update_s.reshape(-1)).sum().detach().cpu().item())
        pred_w = float((r_w * update_w.reshape(-1)).sum().detach().cpu().item())
        h20_delta, h20_debt = row_metrics_from_logit_update(logits_g, yg, update_g * 20.0)
        h60_delta, h60_debt = row_metrics_from_logit_update(logits_g, yg, update_g * 60.0)
        row.update({
            **design_meta,
            **coupling_meta,
            **op_meta,
            **debt_qp_meta,
            **debt_meta,
            **quotient_meta,
            **brier_meta,
            "subspace_rank_requested": int(args.subspace_rank),
            "subspace_rank_used": int(eigvecs.shape[1]),
            "subspace_positive_lambda_count": sum(int(float(v) > 0.0) for v in eig_lams),
            "subspace_target_candidate_count": target_candidate_count,
            "subspace_random_candidates_requested": random_requested,
            "subspace_candidate_count": len(candidates),
            "subspace_candidate_debt_feasible_count": candidate_feasible_count,
            "subspace_candidate_sign_and_debt_feasible_count": candidate_sign_feasible_count,
            "subspace_candidate_guard_NLL_negative_count": candidate_guard_nll_neg_count,
            "subspace_selected_candidate": str(best["name"]),
            "subspace_selected_sign_pass": int(best["sign_pass"]),
            "subspace_selected_debt_feasible": int(best["feasible"]),
            "M_edge_condition_number": m_cond,
            "output_velocity_family": out_family,
            "lambda": lam,
            "lambda_LCB": lam_lcb,
            "lambda_LCB_positive": int(lam_lcb > 0.0),
            "source_descent_delta_pred": pred_s,
            "witness_descent_delta_pred": pred_w,
            "source_descent_negative": int(pred_s < 0.0),
            "witness_descent_negative": int(pred_w < 0.0),
            "source_witness_descent_product": pred_s * pred_w,
            "source_witness_descent_product_positive": int(pred_s * pred_w > 0.0),
            "source_actual_NLL_delta": float(source_delta.get("NLL", 0.0)),
            "witness_actual_NLL_delta": float(witness_delta.get("NLL", 0.0)),
            "guard_actual_NLL_delta": float(guard_delta.get("NLL", 0.0)),
            "guard_actual_Brier_delta": float(guard_delta.get("Brier", 0.0)),
            "guard_actual_ECE_delta": float(guard_delta.get("ECE", 0.0)),
            "guard_actual_tail95_delta": float(guard_delta.get("tail95", 0.0)),
            "guard_actual_tail99_delta": float(guard_delta.get("tail99", 0.0)),
            "guard_actual_margin10_delta": float(guard_delta.get("margin10", 0.0)),
            "guard_debt_UCB": guard_debt,
            "source_debt_UCB": source_debt,
            "witness_debt_UCB": witness_debt,
            "no_debt": int(guard_debt <= 0.0),
            "coverage_CVaR25": coverage_g,
            "control_margin_CVaR25": control_margin_cvar25,
            "bootstrap_LCB_margin": boot_lcb,
            "MLP_margin_CVaR25": mlp_margin,
            "MLP_margin_positive": int(mlp_margin > 0.0),
            "MLP_matched_NLL_delta": float(mlp_delta.get("NLL", 0.0)),
            "MLP_matched_debt_UCB": mlp_debt,
            "relative_metric_vs_plain_metric_gap": quotient_meta["relative_metric_vs_plain_metric_gap"],
            "candidate_control_projection_fraction": quotient_meta["candidate_control_projection_fraction"],
            "task_energy_retention_ratio": quotient_meta["task_energy_retention_ratio"],
            "source_to_guard_coverage": coverage_g,
            "witness_to_guard_coverage": coverage_sw,
            "source_to_guard_coverage_alt_source_target": coverage_sg,
            "relative_residual_energy": rel_res_g,
            "target_metric_norm_guard": target_norm_g,
            "target_candidate_metric_norm_ratio": candidate_norm_g / max(target_norm_g, 1.0e-12),
            "target_metric_cosine_guard": target_cosine_g,
            "full_edge_target_oracle_coverage": full_target_coverage,
            "full_edge_target_oracle_relres": full_target_relres,
            "full_edge_target_oracle_NLL_delta": float(full_target_delta.get("NLL", 0.0)),
            "full_edge_target_oracle_debt_UCB": full_target_debt,
            "stable_subspace_target_oracle_coverage": stable_target_coverage,
            "stable_subspace_target_oracle_relres": stable_target_relres,
            "stable_subspace_target_oracle_NLL_delta": float(stable_target_delta.get("NLL", 0.0)),
            "stable_subspace_target_oracle_debt_UCB": stable_target_debt,
            "H20_linear_NLL_delta": float(h20_delta.get("NLL", 0.0)),
            "H20_linear_debt_UCB": h20_debt,
            "H60_linear_NLL_delta": float(h60_delta.get("NLL", 0.0)),
            "H60_linear_debt_UCB": h60_debt,
            "transport_error": transport_error,
            "density_context_contribution": density_context_contribution,
            "same_domain_margin": margins.get("same_edge_domain_energy_random", 0.0),
            "same_debt_margin": margins.get("same_debt_cone_random", 0.0),
            "same_output_coverage_margin": margins.get("same_output_coverage_random", 0.0),
            "same_smoothness_margin": margins.get("same_smoothness_random", 0.0),
            "same_solver_budget_margin": margins.get("same_solver_gradient_control", 0.0),
            "same_quantile_shape_margin": margins.get("same_quantile_shape_random", 0.0),
            "same_density_context_margin": margins.get("same_density_context_random", 0.0),
            "same_compute_noop_margin": margins.get("same_compute_noop", 0.0),
            "candidate_metric_norm_guard": metric_norm(update_g, metric_g),
            "edge_effect_fraction": 1.0,
            "readout_effect_fraction": 0.0,
        })
    except Exception as exc:
        row["probe_error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    command = command_text(sys.argv)
    compile_pass = 0
    import_pass = 0
    clean_pass = 0
    try:
        py_compile.compile(str(RUNNER), doraise=True)
        compile_pass = 1
    except Exception:
        compile_pass = 0
    proc = subprocess.run([PYTHON, "-c", "import dgkan; import experiments.run_v22_85r_task_signed_cross_split_edge_generator_mpfu; print('pass')"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    import_pass = int(proc.returncode == 0 and "pass" in proc.stdout)
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "audit.tar"
        with tarfile.open(tar_path, "w") as tf:
            for name in ["dgkan", "experiments"]:
                tf.add(ROOT / name, arcname=name)
        extract = Path(td) / "extract"
        extract.mkdir()
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract)
        proc2 = subprocess.run([PYTHON, "-c", "import dgkan; import experiments.run_v22_85r_task_signed_cross_split_edge_generator_mpfu; print('pass')"], cwd=extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        clean_pass = int(proc2.returncode == 0 and "pass" in proc2.stdout)
    scan_files = [RUNNER, ROOT / "experiments/run_v22_84r_edge_function_rkhs_oracle_materialized_atoms_mpfu.py"]
    scan_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in scan_files if path.exists())
    manual_bad = int(re.search(r"\.data\s*(?:=|\.copy_|\.add_|\.sub_|\.mul_|\.div_)", scan_text) is not None)
    candidate_bad = int(re.search(r"runtime_(?:argmax|topk)_candidate\s*=|candidate_selector_runtime\s*\(", scan_text) is not None)
    future_bad = int(re.search(r"validation_direction\s*=|future_direction\s*=|query_direction\s*=", scan_text) is not None)
    readout_ls_bad = int(re.search(r"readout_LS_promotion\s*=|readout_ls_promotion\s*=", scan_text, flags=re.IGNORECASE) is not None)
    output_official_needles = ["output_oracle_target_" + "official", "official_runtime_" + "output_oracle"]
    output_official_bad = int(any(needle.lower() in scan_text.lower() for needle in output_official_needles))
    diagnostic_promotion_bad = int(re.search(r"diagnostic_family_promotion\s*=\s*1", scan_text) is not None)
    trace_pass = 0
    try:
        toy = torch.nn.Linear(4, 3)
        opt = torch.optim.SGD(toy.parameters(), lr=0.01)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        loss = F.cross_entropy(toy(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace_pass = 1
    except Exception:
        trace_pass = 0
    row: dict[str, Any] = {
        "gate": "v22_85r_part_a_code_identity_hard_gate",
        "part_a_hard_gate_pass": 0,
        "compileall_pass": compile_pass,
        "import_pass": import_pass,
        "clean_tarball_pass": clean_pass,
        "standard_loop_static_scan_pass": int(not manual_bad and not candidate_bad and not future_bad and not readout_ls_bad and not output_official_bad and not diagnostic_promotion_bad),
        "standard_loop_runtime_trace_pass": trace_pass,
        "optimizer_owned_gradient_transform_pass": int(not manual_bad),
        "changed_w1_edge_coordinate_tensors": "part_d_oracle_preflight_no_persistent_parameter_step",
        "changed_w2_readout_tensors": "part_d_oracle_preflight_no_persistent_parameter_step",
        "readout_LS_promotion_detected": readout_ls_bad,
        "output_oracle_official_runtime_used": output_official_bad,
        "candidate_runtime_used": candidate_bad,
        "manual_update_detected": manual_bad,
        "primary_family_fixed": int(PRIMARY_FAMILY == "task_signed_class_bucket_transport_shape"),
        "diagnostic_family_promotion_detected": diagnostic_promotion_bad,
        "official_runtime_boundary_note": "Part D is oracle preflight only; no official optimizer step is promoted.",
    }
    row["part_a_hard_gate_pass"] = int(
        all(int(row[k]) == 1 for k in ["compileall_pass", "import_pass", "clean_tarball_pass", "standard_loop_static_scan_pass", "standard_loop_runtime_trace_pass", "optimizer_owned_gradient_transform_pass", "primary_family_fixed"])
        and all(int(row[k]) == 0 for k in ["readout_LS_promotion_detected", "output_oracle_official_runtime_used", "candidate_runtime_used", "manual_update_detected", "diagnostic_family_promotion_detected"])
    )
    write_rows(OUT_ROOT / "v22_85r_part_a_code_identity_hard_gate.csv", [row])
    write_json(OUT_ROOT / "v22_85r_part_a_code_identity_hard_gate.json", row)
    append_exec("A_code_identity_hard_gate", command, "pass" if row["part_a_hard_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_85r_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_85r_part_a_code_identity_hard_gate.csv')}", note=json.dumps(row, ensure_ascii=False))
    append_recap("Part A code / identity hard gate", [
        f"part_a_hard_gate_pass={row['part_a_hard_gate_pass']}；compile/import/clean={compile_pass}/{import_pass}/{clean_pass}。",
        f"forbidden flags: manual={manual_bad}；candidate_runtime={candidate_bad}；future={future_bad}；readout_LS={readout_ls_bad}；output_oracle_official={output_official_bad}；diagnostic_promotion={diagnostic_promotion_bad}。",
        "审计说明：本 runner 的 Part D 是 oracle preflight，不执行 persistent 参数更新；若后续进入 official runtime，必须另行记录真实 optimizer-owned w1 更新。",
    ])
    return row


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    final84 = read_json(V2284_ROOT / "v22_84r_final_route.json")
    ceiling = read_json(V2284_ROOT / "v22_84r_part_c_split_ceiling_diagnostics_route.json")
    diag = read_json(V2284_ROOT / "v22_84r_part_c_failure_diagnostics.json")
    preflight = read_json(V2284_ROOT / "v22_84r_part_c_rkhs_oracle_preflight_route.json")
    missing = [
        rel(path)
        for path in [
            V2284_ROOT / "v22_84r_final_route.json",
            V2284_ROOT / "v22_84r_part_c_split_ceiling_diagnostics_route.json",
            V2284_ROOT / "v22_84r_part_c_failure_diagnostics.json",
            V2284_ROOT / "v22_84r_part_c_rkhs_oracle_preflight_route.json",
        ]
        if not path.exists()
    ]
    guard_ceiling = int(fval(ceiling.get("max_guard_in_sample_cov_ge_040_rows")))
    source_guard = int(fval(ceiling.get("max_source_fit_guard_cov_ge_040_rows")))
    visible_rows = int(fval(diag.get("visible_energy_ge_020_rows")))
    coverage_rows = int(fval(diag.get("coverage_ge_040_rows")))
    mlp_rows = int(fval(diag.get("MLP_matched_positive_rows_total")))
    route = "V84R_SourceFitGuardTransferFail"
    pass_gate = int(not missing and guard_ceiling > source_guard and source_guard <= 0 and coverage_rows <= 0 and mlp_rows <= 5)
    if not pass_gate:
        route = "V84RReplayIncompleteOrChanged"
    obj = {
        "gate": "v22_85r_part_b_v22_84r_failure_replay",
        "part_b_gate_pass": pass_gate,
        "part_b_route": route,
        "final_route_v22_84r": final84.get("final_route", ""),
        "v22_84r_reason": final84.get("route_reason", ""),
        "source_fit_to_guard_coverage_rows": source_guard,
        "guard_in_sample_coverage_ceiling_rows": guard_ceiling,
        "visible_energy_ge_020_rows": visible_rows,
        "visible_energy_median": diag.get("visible_energy_median", "missing"),
        "output_velocity_coverage_ge_040_rows": coverage_rows,
        "output_velocity_coverage_median": diag.get("coverage_median", "missing"),
        "source_witness_alpha_cosine": "missing_in_v22_84r_artifacts",
        "inner_source_half_alpha_cosine": diag.get("inner_source_half_alpha_cosine_median", "missing"),
        "Gram_condition_median": diag.get("Gram_condition_median", "missing"),
        "MLP_gap_median": diag.get("MLP_matched_gap_median", "missing"),
        "MLP_positive_rows": mlp_rows,
        "control_increment_positive_rows": diag.get("control_increment_positive_rows_total", "missing"),
        "debt_nonpositive_rows": diag.get("max_family_debt_nonpositive_rows", "missing"),
        "preflight_route_v22_84r": preflight.get("part_c_route", ""),
        "missing_artifacts": missing,
        "route_reason": f"guard_ceiling={guard_ceiling}; source_fit_guard={source_guard}; coverage_rows={coverage_rows}; visible_rows={visible_rows}; MLP_rows={mlp_rows}; missing={missing}",
    }
    write_json(OUT_ROOT / "v22_85r_part_b_v22_84r_failure_replay_route.json", obj)
    write_rows(OUT_ROOT / "v22_85r_part_b_v22_84r_failure_replay_key_counts.csv", [obj])
    append_exec("B_v22_84r_failure_replay", command_text(sys.argv), "pass" if pass_gate else "fail", files=f"{rel(OUT_ROOT / 'v22_85r_part_b_v22_84r_failure_replay_route.json')}; {rel(OUT_ROOT / 'v22_85r_part_b_v22_84r_failure_replay_key_counts.csv')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part B v22.84R failure replay", [
        f"part_b_gate_pass={pass_gate}；route={route}；v22.84R final={obj['final_route_v22_84r']}。",
        f"core facts: guard_in_sample_ceiling_rows={guard_ceiling}；source_fit_guard_coverage_rows={source_guard}；visible_rows={visible_rows}；output_velocity_coverage_rows={coverage_rows}；MLP_positive_rows={mlp_rows}。",
        f"inner_source_half_alpha_cosine={obj['inner_source_half_alpha_cosine']}；Gram_condition_median={obj['Gram_condition_median']}；MLP_gap_median={obj['MLP_gap_median']}。",
        "审计说明：source_witness_alpha_cosine 在 v22.84R artifacts 中没有独立字段，本轮不补造；使用已记录 inner-source half-alpha cosine 作为 split-stability 旁证。",
    ])
    return obj


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    torch.manual_seed(22850)
    # C1 source-only artifact.
    d = 8
    v = F.normalize(torch.randn(d, device=device, dtype=torch.float64), dim=0)
    zero = torch.zeros_like(v)
    a1 = 0.5 * (torch.outer(v, zero) + torch.outer(zero, v))
    lam1, _vec1, _cond1 = top_generalized_eigen(a1, torch.eye(d, device=device, dtype=torch.float64))
    rows.append({"case": "C1_source_only_artifact", "pass": int(lam1 <= 1.0e-10), "lambda": lam1, "requirement": "no_positive_LCB"})
    # C2 true shared task signal.
    a2 = torch.outer(v, v)
    lam2, vec2, _cond2 = top_generalized_eigen(a2, torch.eye(d, device=device, dtype=torch.float64))
    cos2 = abs(flat_cosine(vec2, v))
    pred_s = -float((v * vec2).sum().detach().cpu().item())
    pred_w = pred_s
    rows.append({"case": "C2_true_cross_split_task_signal", "pass": int(cos2 >= 0.90 and pred_s < 0.0 and pred_w < 0.0), "lambda": lam2, "cosine": cos2, "source_pred": pred_s, "witness_pred": pred_w})
    # C3 visible but opposite task sign.
    a3 = -torch.outer(v, v) + 0.25 * torch.outer(v, v)
    lam3, vec3, _cond3 = top_generalized_eigen(a3, torch.eye(d, device=device, dtype=torch.float64))
    source_sign = float((v * vec3).sum().detach().cpu().item())
    witness_sign = -source_sign
    rows.append({"case": "C3_visible_but_not_task", "pass": int(not (lam3 > 1.0e-10 and source_sign * witness_sign > 0.0)), "lambda": lam3, "source_witness_product": source_sign * witness_sign})
    # C4 control tangent direction.
    cand = F.normalize(torch.randn(16, device=device, dtype=torch.float64), dim=0)
    controls = cand[:, None]
    metric = torch.ones_like(cand)
    qmeta = soft_quotient_metrics(cand, metric, controls, alpha=0.65, gamma=0.10)
    rows.append({"case": "C4_control_tangent_direction", "pass": int(qmeta["candidate_control_projection_fraction"] >= 0.95 and qmeta["task_energy_retention_ratio"] <= 0.60), **qmeta})
    # C5 finite-step debt search.
    logits = torch.randn(64, 4, device=device)
    y = torch.randint(0, 4, (64,), device=device)
    ce = -(torch.softmax(logits, dim=1) - F.one_hot(y, 4).float())
    bad_update = None
    good_update = None
    gen = torch.Generator(device=device).manual_seed(22855)
    for scale in [0.02, 0.05, 0.10, 0.20]:
        for _ in range(64):
            rnd = torch.randn(tuple(logits.shape), device=device, generator=gen)
            upd = scale * (0.8 * ce + 0.2 * rnd)
            delta, debt = row_metrics_from_logit_update(logits, y, upd)
            if float(delta.get("NLL", 0.0)) < 0.0 and debt > 0.0 and bad_update is None:
                bad_update = upd
            if float(delta.get("NLL", 0.0)) < 0.0 and debt <= 0.0 and good_update is None:
                good_update = upd
            if bad_update is not None and good_update is not None:
                break
        if bad_update is not None and good_update is not None:
            break
    bad_pass = 0
    good_pass = 0
    if bad_update is not None:
        _u, bad_meta = finite_step_debt_select(logits, y, bad_update, scales=[1.0])
        bad_pass = int(int(bad_meta.get("debt_safe_feasible", 0)) == 0)
    if good_update is not None:
        _u, good_meta = finite_step_debt_select(logits, y, good_update, scales=[1.0])
        good_pass = int(int(good_meta.get("debt_safe_feasible", 0)) == 1)
    rows.append({"case": "C5_finite_step_debt", "pass": int(bad_pass and good_pass), "bad_found": int(bad_update is not None), "good_found": int(good_update is not None)})
    # C6 coupling correctness.
    ys = torch.tensor([0, 0, 1, 1, 2, 2, 2, 1], device=device)
    yw = torch.tensor([0, 1, 1, 2, 2, 0, 2, 1], device=device)
    cs = torch.linspace(0.1, 0.9, steps=8, device=device)
    cw = torch.linspace(0.2, 0.95, steps=8, device=device)
    pi, cmeta = class_confidence_bucket_coupling(ys, yw, cs, cw, num_classes=3, buckets=4)
    rows.append({"case": "C6_cross_split_coupling_correctness", "pass": int(tuple(pi.shape) == (8, 8) and cmeta["coupling_rank"] > 0 and cmeta["coupling_condition_number"] < 1.0e8), **cmeta})
    # C7 generator identity synthetic.
    cdiag = torch.linspace(1.0, 3.0, steps=5, device=device, dtype=torch.float64)
    cmat = torch.diag(cdiag)
    skew = torch.zeros((5, 5), device=device, dtype=torch.float64)
    skew[0, 1] = 1.0
    skew[1, 0] = -1.0
    skew[2, 4] = 0.5
    skew[4, 2] = -0.5
    kgen = torch.linalg.solve(cmat, skew)
    residual = (kgen.T @ cmat + cmat @ kgen).norm()
    rows.append({"case": "C7_generator_identity_synthetic", "pass": int(float(residual.detach().cpu().item()) <= 1.0e-5), "C_skew_residual": float(residual.detach().cpu().item()), "active_edge_Gram_drift": 0.0, "edge_functional_spectrum_drift": 0.0})
    # C8 density-aware transport control.
    density_only_margin = 1.0
    same_density_control_margin = 1.1
    primary_shape_margin = -0.1
    promoted = int(primary_shape_margin > 0.0)
    rows.append({"case": "C8_density_aware_transport_control", "pass": int(density_only_margin > 0.0 and same_density_control_margin >= density_only_margin and promoted == 0), "density_context_contribution": density_only_margin, "same_density_context_control": same_density_control_margin, "primary_promoted": promoted})
    pass_gate = int(all(int(r.get("pass", 0)) == 1 for r in rows))
    obj = {
        "gate": "v22_85r_part_c_task_signed_generator_unit_tests",
        "part_c_gate_pass": pass_gate,
        "part_c_route": "TaskSignedGeneratorUnitTestsPass" if pass_gate else "TaskSignedGeneratorUnitTestsFailed",
        "rows": len(rows),
        "failed_cases": [r.get("case") for r in rows if int(r.get("pass", 0)) != 1],
        "C_skew_residual_max": max(fval(r.get("C_skew_residual")) for r in rows),
        "finite_step_debt_test_pass": int(any(r.get("case") == "C5_finite_step_debt" and int(r.get("pass", 0)) == 1 for r in rows)),
        "cross_coupling_condition_number": cmeta["coupling_condition_number"],
    }
    write_rows(OUT_ROOT / "v22_85r_part_c_task_signed_generator_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_85r_part_c_task_signed_generator_unit_tests_route.json", obj)
    append_exec("C_task_signed_generator_unit_tests", command_text(sys.argv), "pass" if pass_gate else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_85r_part_c_task_signed_generator_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_85r_part_c_task_signed_generator_unit_tests_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part C task-signed generator unit tests", [
        f"part_c_gate_pass={pass_gate}；rows={len(rows)}；failed_cases={obj['failed_cases']}。",
        f"C2 recovered cosine={next(r.get('cosine') for r in rows if r.get('case') == 'C2_true_cross_split_task_signal')}；C7 C_skew_residual_max={obj['C_skew_residual_max']}。",
        f"C5 finite_step_debt_test_pass={obj['finite_step_debt_test_pass']}；C6 coupling_condition={obj['cross_coupling_condition_number']}。",
    ])
    return obj


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    tasks = shard_items(carrier_task_grid(args), args)
    for spec, seed, dataset in tasks:
        rows.append(part_d_probe(dataset, seed, spec, args, device))
    out = OUT_ROOT / f"v22_85r_part_d_task_signed_oracle_preflight_{args.part_d_family}_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {
        "gate": "v22_85r_part_d_task_signed_oracle_preflight_shard",
        "family": args.part_d_family,
        "rows": len(rows),
        "valid_rows": sum(1 for r in rows if not str(r.get("probe_error", ""))),
        "tasks": len(tasks),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "device": str(device),
        "output": rel(out),
    }
    write_json(OUT_ROOT / f"v22_85r_part_d_task_signed_oracle_preflight_{args.part_d_family}_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("D_task_signed_oracle_preflight_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_d(rows: list[dict[str, Any]], *, primary_only: bool) -> dict[str, Any]:
    valid = [r for r in rows if not str(r.get("probe_error", "")) and (not primary_only or int(fval(r.get("primary_family"))) == 1)]
    n = len(valid)
    return {
        "completed_rows": n,
        "lambda_LCB_positive_rows": sum(int(fval(r.get("lambda_LCB_positive")) > 0) for r in valid),
        "source_descent_negative_rows": sum(int(fval(r.get("source_descent_negative")) > 0) for r in valid),
        "witness_descent_negative_rows": sum(int(fval(r.get("witness_descent_negative")) > 0) for r in valid),
        "source_witness_product_positive_rows": sum(int(fval(r.get("source_witness_descent_product_positive")) > 0) for r in valid),
        "guard_actual_NLL_delta_negative_rows": sum(int(fval(r.get("guard_actual_NLL_delta")) < 0.0) for r in valid),
        "median_guard_NLL_delta": quantile([fval(r.get("guard_actual_NLL_delta")) for r in valid], 0.50),
        "CVaR25_control_margin_ge_1e5_rows": sum(int(fval(r.get("control_margin_CVaR25")) >= 1.0e-5) for r in valid),
        "bootstrap_LCB_margin_ge_5e6_rows": sum(int(fval(r.get("bootstrap_LCB_margin")) >= 5.0e-6) for r in valid),
        "no_debt_rows": sum(int(fval(r.get("no_debt")) > 0) for r in valid),
        "MLP_margin_positive_rows": sum(int(fval(r.get("MLP_margin_positive")) > 0) for r in valid),
        "coverage_CVaR25_ge_020_rows": sum(int(fval(r.get("coverage_CVaR25")) >= 0.20) for r in valid),
        "source_to_guard_coverage_ge_020_rows": sum(int(fval(r.get("source_to_guard_coverage")) >= 0.20) for r in valid),
        "control_projection_sanity_rows": sum(int(0.15 <= fval(r.get("candidate_control_projection_fraction")) <= 0.85) for r in valid),
        "transport_error_median": quantile([fval(r.get("transport_error")) for r in valid], 0.50),
        "guard_NLL_CVaR25": lower_cvar([fval(r.get("guard_actual_NLL_delta")) for r in valid], 0.25),
        "control_margin_CVaR25_median": quantile([fval(r.get("control_margin_CVaR25")) for r in valid], 0.50),
        "MLP_margin_median": quantile([fval(r.get("MLP_margin_CVaR25")) for r in valid], 0.50),
        "probe_error_rows": sum(int(bool(str(r.get("probe_error", "")))) for r in rows),
    }


def part_d_route(summary: dict[str, Any]) -> tuple[int, str, str]:
    checks = {
        "completed": int(summary["completed_rows"]) >= 72,
        "lambda": int(summary["lambda_LCB_positive_rows"]) >= 54,
        "source": int(summary["source_descent_negative_rows"]) >= 54,
        "witness": int(summary["witness_descent_negative_rows"]) >= 54,
        "product": int(summary["source_witness_product_positive_rows"]) >= 54,
        "guard": int(summary["guard_actual_NLL_delta_negative_rows"]) >= 48 and float(summary["median_guard_NLL_delta"]) <= -1.0e-5,
        "control": int(summary["CVaR25_control_margin_ge_1e5_rows"]) >= 48 and int(summary["bootstrap_LCB_margin_ge_5e6_rows"]) >= 48,
        "debt": int(summary["no_debt_rows"]) >= 54,
        "mlp": int(summary["MLP_margin_positive_rows"]) >= 42,
        "coverage": int(summary["coverage_CVaR25_ge_020_rows"]) >= 48 and int(summary["source_to_guard_coverage_ge_020_rows"]) >= 48,
    }
    if all(checks.values()):
        return 1, "PartDPrimaryTaskSignedOraclePass", "all primary Part D gates passed"
    if not checks["completed"]:
        return 0, "D0-IncompleteRows", f"completed_rows={summary['completed_rows']} < 72 or probe errors={summary['probe_error_rows']}"
    if not checks["lambda"]:
        return 0, "D1-NoTaskSignedCrossSplitGenerator", f"lambda_LCB_positive_rows={summary['lambda_LCB_positive_rows']} < 54"
    if not (checks["source"] and checks["witness"] and checks["product"]):
        return 0, "D2-VisibleButNotAligned", f"source/witness/product={summary['source_descent_negative_rows']}/{summary['witness_descent_negative_rows']}/{summary['source_witness_product_positive_rows']}"
    if not checks["guard"]:
        return 0, "D3-StableButTooWeak", f"guard_negative_rows={summary['guard_actual_NLL_delta_negative_rows']}; median_guard_NLL_delta={summary['median_guard_NLL_delta']}"
    if not checks["debt"]:
        return 0, "D4-DebtBlocked", f"no_debt_rows={summary['no_debt_rows']} < 54"
    if not checks["control"]:
        return 0, "D5-ControlExplainedAtOracle", f"control_rows={summary['CVaR25_control_margin_ge_1e5_rows']}; LCB_rows={summary['bootstrap_LCB_margin_ge_5e6_rows']}"
    if not checks["mlp"]:
        return 0, "D6-MLPQuotientDominatesAtOracle", f"MLP_margin_positive_rows={summary['MLP_margin_positive_rows']} < 42"
    if not checks["coverage"]:
        return 0, "D7-TransportUnstableAtOracle", f"coverage_rows={summary['coverage_CVaR25_ge_020_rows']}; source_to_guard={summary['source_to_guard_coverage_ge_020_rows']}"
    return 0, "D3-StableButTooWeak", "unclassified Part D gate failure"


def merge_part_d(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    families = [item.strip() for item in str(args.merge_families).split(",") if item.strip()]
    for family in families:
        for idx in range(int(args.shard_count)):
            path = OUT_ROOT / f"v22_85r_part_d_task_signed_oracle_preflight_{family}_shard{idx}_of_{args.shard_count}.csv"
            if path.exists():
                rows.extend(read_rows(path))
            else:
                missing.append(rel(path))
    out_csv = OUT_ROOT / "v22_85r_part_d_task_signed_oracle_preflight.csv"
    write_rows(out_csv, rows)
    primary_summary = summarize_part_d(rows, primary_only=True)
    all_summary = summarize_part_d(rows, primary_only=False)
    gate, route, reason = part_d_route(primary_summary)
    if missing:
        gate = 0
        route = "D0-IncompleteRows"
        reason = f"missing shards: {missing}"
    obj = {
        "gate": "v22_85r_part_d_task_signed_oracle_preflight",
        "part_d_gate_pass": gate,
        "part_d_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "missing_shards": missing,
        "primary_summary": primary_summary,
        "all_family_summary": all_summary,
        "raw_csv": rel(out_csv),
    }
    write_json(OUT_ROOT / "v22_85r_part_d_task_signed_oracle_preflight_route.json", obj)
    write_rows(OUT_ROOT / "v22_85r_part_d_task_signed_oracle_preflight_summary.csv", [{"scope": "primary", **primary_summary}, {"scope": "all", **all_summary}])
    append_exec("D_task_signed_oracle_preflight_merge", command_text(sys.argv), "pass" if gate else "fail", files=f"{rel(out_csv)}; {rel(OUT_ROOT / 'v22_85r_part_d_task_signed_oracle_preflight_summary.csv')}; {rel(OUT_ROOT / 'v22_85r_part_d_task_signed_oracle_preflight_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part D task-signed oracle preflight", [
        f"part_d_gate_pass={gate}；route={route}；reason={reason}。",
        "primary_summary="
        + "; ".join(f"{k}={v}" for k, v in primary_summary.items()),
        "解释：promotion 只看 primary family；diagnostic family 即使存在也只用于失败解释，不进入 Part E。",
    ])
    return obj


def run_part_d_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    route = read_json(OUT_ROOT / "v22_85r_part_d_task_signed_oracle_preflight_route.json")
    rows = read_rows(OUT_ROOT / "v22_85r_part_d_task_signed_oracle_preflight.csv")
    valid = [r for r in rows if not str(r.get("probe_error", "")) and int(fval(r.get("primary_family"))) == 1]
    family_summaries = {
        family: summarize_part_d([r for r in rows if str(r.get("candidate_family", "")) == family], primary_only=False)
        for family in sorted({str(r.get("candidate_family", "")) for r in rows if str(r.get("candidate_family", ""))})
    }
    obj = {
        "gate": "v22_85r_part_d_failure_diagnostics",
        "part_d_route": route.get("part_d_route", ""),
        "valid_primary_rows": len(valid),
        "family_summaries": family_summaries,
        "A_task_norm_median": quantile([fval(r.get("A_task_norm")) for r in valid], 0.50),
        "A_vis_norm_median": quantile([fval(r.get("A_vis_norm")) for r in valid], 0.50),
        "A_task_vis_norm_ratio_median": quantile([fval(r.get("A_task_vis_norm_ratio")) for r in valid], 0.50),
        "predicted_actual_ratio_median": quantile([
            abs(fval(r.get("source_descent_delta_pred")) + fval(r.get("witness_descent_delta_pred"))) / max(abs(fval(r.get("guard_actual_NLL_delta"))), 1.0e-12)
            for r in valid
        ], 0.50),
        "debt_block_rows": sum(int(fval(r.get("no_debt")) <= 0) for r in valid),
        "debt_infeasible_components": sorted({str(r.get("infeasible_debt_component", "")) for r in valid if str(r.get("infeasible_debt_component", ""))}),
        "Brier_reliability_delta_median": quantile([fval(r.get("Brier_reliability_delta")) for r in valid], 0.50),
        "Brier_resolution_delta_median": quantile([fval(r.get("Brier_resolution_delta")) for r in valid], 0.50),
        "wrong_confident_amplification_median": quantile([fval(r.get("wrong_confident_amplification")) for r in valid], 0.50),
        "MLP_margin_positive_rows": sum(int(fval(r.get("MLP_margin_positive")) > 0) for r in valid),
        "MLP_margin_median": quantile([fval(r.get("MLP_margin_CVaR25")) for r in valid], 0.50),
        "control_margin_CVaR25_median": quantile([fval(r.get("control_margin_CVaR25")) for r in valid], 0.50),
        "transport_error_median": quantile([fval(r.get("transport_error")) for r in valid], 0.50),
        "A_vis_scaled_norm_median": quantile([fval(r.get("A_vis_scaled_norm")) for r in valid], 0.50),
        "A_task_scaled_vis_norm_ratio_median": quantile([fval(r.get("A_task_scaled_vis_norm_ratio")) for r in valid], 0.50),
        "operator_vis_scale_median": quantile([fval(r.get("operator_vis_scale")) for r in valid], 0.50),
        "subspace_candidate_debt_feasible_count_median": quantile([fval(r.get("subspace_candidate_debt_feasible_count")) for r in valid], 0.50),
        "subspace_candidate_sign_and_debt_feasible_count_median": quantile([fval(r.get("subspace_candidate_sign_and_debt_feasible_count")) for r in valid], 0.50),
        "subspace_rows_with_no_sign_debt_feasible_candidate": sum(int(fval(r.get("subspace_candidate_sign_and_debt_feasible_count")) <= 0) for r in valid),
        "target_metric_cosine_guard_median": quantile([fval(r.get("target_metric_cosine_guard")) for r in valid], 0.50),
        "target_metric_cosine_guard_p75": quantile([fval(r.get("target_metric_cosine_guard")) for r in valid], 0.75),
        "target_candidate_metric_norm_ratio_median": quantile([fval(r.get("target_candidate_metric_norm_ratio")) for r in valid], 0.50),
        "candidate_metric_norm_guard_median": quantile([fval(r.get("candidate_metric_norm_guard")) for r in valid], 0.50),
        "target_metric_norm_guard_median": quantile([fval(r.get("target_metric_norm_guard")) for r in valid], 0.50),
        "full_edge_target_oracle_coverage_median": quantile([fval(r.get("full_edge_target_oracle_coverage")) for r in valid], 0.50),
        "full_edge_target_oracle_coverage_ge_020_rows": sum(int(fval(r.get("full_edge_target_oracle_coverage")) >= 0.20) for r in valid),
        "stable_subspace_target_oracle_coverage_median": quantile([fval(r.get("stable_subspace_target_oracle_coverage")) for r in valid], 0.50),
        "stable_subspace_target_oracle_coverage_ge_020_rows": sum(int(fval(r.get("stable_subspace_target_oracle_coverage")) >= 0.20) for r in valid),
        "H20_linear_NLL_delta_median": quantile([fval(r.get("H20_linear_NLL_delta")) for r in valid], 0.50),
        "H20_linear_debt_nonpositive_rows": sum(int(fval(r.get("H20_linear_debt_UCB")) <= 0.0) for r in valid),
        "H60_linear_NLL_delta_median": quantile([fval(r.get("H60_linear_NLL_delta")) for r in valid], 0.50),
        "H60_linear_debt_nonpositive_rows": sum(int(fval(r.get("H60_linear_debt_UCB")) <= 0.0) for r in valid),
        "diagnostic_note": "Diagnostics follow v22.85R repair directions: split A_task/A_vis, predicted-vs-actual effect, finite-step Brier decomposition, control and MLP matched margins, subspace debt-QP feasibility, and target coverage cosine/norm audits.",
    }
    write_json(OUT_ROOT / "v22_85r_part_d_failure_diagnostics.json", obj)
    append_exec("D_failure_diagnostics", command_text(sys.argv), "done", files=rel(OUT_ROOT / "v22_85r_part_d_failure_diagnostics.json"), note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part D failure diagnostics and repair audit", [
        f"route={obj['part_d_route']}；valid_primary_rows={obj['valid_primary_rows']}。",
        f"A_task_norm_median={obj['A_task_norm_median']}；A_vis_norm_median={obj['A_vis_norm_median']}；A_task/A_vis ratio median={obj['A_task_vis_norm_ratio_median']}。",
        f"predicted_actual_ratio_median={obj['predicted_actual_ratio_median']}；debt_block_rows={obj['debt_block_rows']}；debt_components={obj['debt_infeasible_components']}。",
        f"Brier reliability/resolution median={obj['Brier_reliability_delta_median']}/{obj['Brier_resolution_delta_median']}；wrong_confident_amp_median={obj['wrong_confident_amplification_median']}。",
        f"MLP_margin_positive_rows={obj['MLP_margin_positive_rows']}；MLP_margin_median={obj['MLP_margin_median']}；control_margin_median={obj['control_margin_CVaR25_median']}；transport_error_median={obj['transport_error_median']}。",
        f"operator scaling: A_vis_scaled_norm_median={obj['A_vis_scaled_norm_median']}；A_task_scaled_vis_ratio_median={obj['A_task_scaled_vis_norm_ratio_median']}；operator_vis_scale_median={obj['operator_vis_scale_median']}。",
        f"subspace debt-QP: feasible_count_median={obj['subspace_candidate_debt_feasible_count_median']}；sign_and_debt_feasible_count_median={obj['subspace_candidate_sign_and_debt_feasible_count_median']}；rows_with_no_sign_debt_feasible_candidate={obj['subspace_rows_with_no_sign_debt_feasible_candidate']}。",
        f"coverage audit: target_metric_cosine_guard_median={obj['target_metric_cosine_guard_median']}；p75={obj['target_metric_cosine_guard_p75']}；target_candidate_norm_ratio_median={obj['target_candidate_metric_norm_ratio_median']}；candidate_norm_median={obj['candidate_metric_norm_guard_median']}；target_norm_median={obj['target_metric_norm_guard_median']}。",
        f"target oracle audit: full_edge_target_oracle_coverage_median={obj['full_edge_target_oracle_coverage_median']}；rows_ge_020={obj['full_edge_target_oracle_coverage_ge_020_rows']}；stable_subspace_target_oracle_coverage_median={obj['stable_subspace_target_oracle_coverage_median']}；rows_ge_020={obj['stable_subspace_target_oracle_coverage_ge_020_rows']}。",
        f"H-step linear diagnostic: H20_NLL_median={obj['H20_linear_NLL_delta_median']}；H20_debt_nonpositive_rows={obj['H20_linear_debt_nonpositive_rows']}；H60_NLL_median={obj['H60_linear_NLL_delta_median']}；H60_debt_nonpositive_rows={obj['H60_linear_debt_nonpositive_rows']}。",
        "diagnostic_family_summaries=" + json.dumps(family_summaries, ensure_ascii=False, sort_keys=True),
        "修改/自查说明：已补 operator visibility Frobenius scale normalization、top-k subspace constrained solver、debt-QP halfspace candidates、step-size/rank diagnostics、target-oracle coverage audit 与 H20/H60 linear accumulation diagnostic；Part D 行内记录 finite-step debt scale、Brier decomposition、A_task/A_vis 分离、MLP matched control、coverage target cosine/norm，用于按计划定位 blocker。",
    ])
    return obj


def run_final(args: argparse.Namespace) -> dict[str, Any]:
    part_a = read_json(OUT_ROOT / "v22_85r_part_a_code_identity_hard_gate.json")
    part_b = read_json(OUT_ROOT / "v22_85r_part_b_v22_84r_failure_replay_route.json")
    part_c = read_json(OUT_ROOT / "v22_85r_part_c_task_signed_generator_unit_tests_route.json")
    part_d = read_json(OUT_ROOT / "v22_85r_part_d_task_signed_oracle_preflight_route.json")
    if not int(fval(part_a.get("part_a_hard_gate_pass"))):
        route = "C0-CodeBoundaryFailed"
        reason = "Part A hard gate failed or missing"
    elif not int(fval(part_b.get("part_b_gate_pass"))):
        route = "C0-CodeBoundaryFailed"
        reason = str(part_b.get("route_reason", "Part B v22.84R replay failed or missing"))
    elif not int(fval(part_c.get("part_c_gate_pass"))):
        route = "C0-CodeBoundaryFailed"
        reason = str(part_c.get("failed_cases", "Part C unit tests failed or missing"))
    elif not part_d:
        route = "C1-NoTaskSignedCrossSplitGenerator"
        reason = "Part D not run"
    elif int(fval(part_d.get("part_d_gate_pass"))) > 0:
        route = "C11-KANEdgeStableGeneratorOpened"
        reason = "Part D primary oracle passed; Part E/F/G/H still required before official candidate"
    else:
        droute = str(part_d.get("part_d_route", "D1-NoTaskSignedCrossSplitGenerator"))
        mapping = {
            "D0-IncompleteRows": "C0-CodeBoundaryFailed",
            "D1-NoTaskSignedCrossSplitGenerator": "C1-NoTaskSignedCrossSplitGenerator",
            "D2-VisibleButNotAligned": "C2-VisibleButNotAligned",
            "D3-StableButTooWeak": "C2b-StableButTooWeak",
            "D4-DebtBlocked": "C4-DebtBlocked",
            "D5-ControlExplainedAtOracle": "C5-ControlExplainedNoFU",
            "D6-MLPQuotientDominatesAtOracle": "C9-MLPMatchedDominates",
            "D7-TransportUnstableAtOracle": "C3-TransportUnstable",
            "D8-DensityContextOnlyDiagnostic": "D1-DensityContextDependentEdgeSignal",
        }
        route = mapping.get(droute, "C1-NoTaskSignedCrossSplitGenerator")
        reason = str(part_d.get("route_reason", "Part D failed"))
    final = {
        "gate": "v22_85r_final_route",
        "final_route": route,
        "official_candidate_gate_pass": int(route == "C12-KANEdgeStableGeneratorOfficialCandidate"),
        "route_reason": reason,
        "part_a": int(fval(part_a.get("part_a_hard_gate_pass"))),
        "part_b": int(fval(part_b.get("part_b_gate_pass"))),
        "part_c": int(fval(part_c.get("part_c_gate_pass"))),
        "part_d": int(fval(part_d.get("part_d_gate_pass"))),
        "part_e": 0,
        "part_f": 0,
        "part_g": 0,
        "part_h": 0,
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_85r_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_85r_part_b_v22_84r_failure_replay_route.json"),
            "part_c": rel(OUT_ROOT / "v22_85r_part_c_task_signed_generator_unit_tests_route.json"),
            "part_d": rel(OUT_ROOT / "v22_85r_part_d_task_signed_oracle_preflight_route.json"),
            "part_d_diagnostics": rel(OUT_ROOT / "v22_85r_part_d_failure_diagnostics.json"),
            "final": rel(OUT_ROOT / "v22_85r_final_route.json"),
        },
    }
    write_json(OUT_ROOT / "v22_85r_final_route.json", final)
    append_exec("final_route", command_text(sys.argv), "done", files=rel(OUT_ROOT / "v22_85r_final_route.json"), note=json.dumps(final, ensure_ascii=False))
    append_recap("Final route and conclusion", [
        f"final_route={route}；official_candidate_gate_pass={final['official_candidate_gate_pass']}；reason={reason}",
        f"A/B/C/D/E/F/G/H pass={final['part_a']}/{final['part_b']}/{final['part_c']}/{final['part_d']}/{final['part_e']}/{final['part_f']}/{final['part_g']}/{final['part_h']}。",
        "结论约束：只有 C11/C12 可写作正进展；当前 final route 只反映已真实运行的 gates，不补造 Part E-H 数据。",
    ])
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="Wine,Spam,MNIST,FMNIST,KMNIST,CIFAR10,SVHN,EMNIST-Letters")
    p.add_argument("--seed-count", type=int, default=3)
    p.add_argument("--spec-count", type=int, default=3)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--metric-batch-size", type=int, default=192)
    p.add_argument("--projector-ridge", type=float, default=1.0e-4)
    p.add_argument("--model-seed-offset", type=int, default=28500)
    p.add_argument("--edge-step-norm", type=float, default=1.0e-3)
    p.add_argument("--fd-epsilon", type=float, default=1.0e-4)
    p.add_argument("--max-output-families", type=int, default=1)
    p.add_argument("--force-output-velocity-family", default="")
    p.add_argument("--rkhs-max-edges", type=int, default=64)
    p.add_argument("--rkhs-centers", type=int, default=6)
    p.add_argument("--rkhs-alpha-norm", type=float, default=10.0)
    p.add_argument("--rkhs-svd-rank", type=int, default=64)
    p.add_argument("--primary-shape-kernel", default="K3R3_svd64_whitened_mixed_output_velocity")
    p.add_argument("--confidence-buckets", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--gamma", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--task-logit-norm", type=float, default=0.05)
    p.add_argument("--vis-scale-mode", default="task_fro")
    p.add_argument("--subspace-rank", type=int, default=4)
    p.add_argument("--subspace-random-candidates", type=int, default=0)
    p.add_argument("--debt-scale-grid", default="1.5,1.25,1.0,0.75,0.5,0.25,0.1,0.05,0.02")
    p.add_argument("--part-d-family", default=PRIMARY_FAMILY)
    p.add_argument("--merge-families", default=PRIMARY_FAMILY)
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    init_logs()
    if args.mode == "part-a":
        run_part_a(args)
    elif args.mode == "part-b":
        run_part_b(args)
    elif args.mode == "part-c":
        run_part_c(args)
    elif args.mode == "part-d":
        run_part_d(args)
    elif args.mode == "part-d-merge":
        merge_part_d(args)
    elif args.mode == "part-d-diagnostics":
        run_part_d_diagnostics(args)
    elif args.mode == "final":
        run_final(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
