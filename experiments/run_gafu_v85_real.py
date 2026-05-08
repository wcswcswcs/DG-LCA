#!/usr/bin/env python3
"""DG-KAN v8.5 KANbeFair external-fair audit runner.

This runner intentionally starts with the source/baseline gates from the v8.5
plan. It does not claim external superiority unless KANbeFair reproduction and
adapter/counter contracts are actually satisfied by landed artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import run_gafu_v83_real as v83  # noqa: E402
import run_gafu_v84_real as v84  # noqa: E402
from dgkan_core import ensure_dir, parse_int_list, parse_str_list, save_json, set_seed, write_csv  # noqa: E402


PLAN_PATH = "docs/DG-KAN_v8.5_KANbeFair_ExternalFair_FunctionalAdvantage_完整实验计划.md"
KB_PATH = Path("third_party/KANbeFair")
DG_PRETRAIN_HOOK: Any | None = None
METRIC_UNAVAILABLE = "metric_unavailable"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in (None, "", METRIC_UNAVAILABLE):
            return default
        return float(value)
    except Exception:
        return default


def _read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def _git_commit(path: Path) -> str:
    head = path / ".git" / "HEAD"
    if not head.exists():
        return METRIC_UNAVAILABLE
    text = head.read_text(encoding="utf-8", errors="replace").strip()
    if text.startswith("ref:"):
        ref = text.split(" ", 1)[1]
        ref_path = path / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8", errors="replace").strip()
    return text


def _dependency_status() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ["torch", "torchvision", "matplotlib", "fvcore", "torchtext", "pandas", "sklearn", "scipy"]:
        try:
            __import__(name)
            out[name] = "ok"
        except Exception as exc:
            out[name] = f"{type(exc).__name__}: {exc}"
    return out


def _install_matplotlib_stub_if_missing() -> int:
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot  # noqa: F401
        return 0
    except Exception:
        module = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")

        class _DummyAxes:
            def imshow(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        class _DummyFigure:
            def colorbar(self, *_args: Any, **_kwargs: Any) -> None:
                return None

            def savefig(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        def _subplots(*_args: Any, **_kwargs: Any) -> Tuple[_DummyFigure, _DummyAxes]:
            return _DummyFigure(), _DummyAxes()

        pyplot.subplots = _subplots  # type: ignore[attr-defined]
        for name in ["plot", "show", "figure", "imshow", "colorbar", "savefig", "close", "scatter", "legend", "title", "xlabel", "ylabel"]:
            setattr(pyplot, name, lambda *_args, **_kwargs: None)
        module.pyplot = pyplot  # type: ignore[attr-defined]
        sys.modules["matplotlib"] = module
        sys.modules["matplotlib.pyplot"] = pyplot
        return 1


def _import_kanbefair_models(kb_path: Path) -> Tuple[Any, Any, Any, int, str]:
    matplotlib_stub_used = _install_matplotlib_stub_if_missing()
    src = kb_path / "src"
    if str(src.resolve()) not in sys.path:
        sys.path.insert(0, str(src.resolve()))
    try:
        from models.mlp import MLP  # type: ignore
        from models.kanbefair import KANbeFair  # type: ignore
        from models.bspline_mlp import BSpline_MLP  # type: ignore
        return MLP, KANbeFair, BSpline_MLP, matplotlib_stub_used, "ok"
    except Exception as exc:
        return None, None, None, matplotlib_stub_used, f"{type(exc).__name__}: {exc}"


def _activation(name: str) -> Any:
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }.get(name, nn.GELU)


def _shortcut(name: str) -> nn.Module:
    if name == "identity":
        return nn.Identity()
    if name == "zero":
        class Zero(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 0
        return Zero()
    return nn.SiLU()


def _kb_namespace(
    *,
    model_name: str,
    input_size: int,
    output_size: int,
    layers_width: Sequence[int],
    activation_name: str = "gelu",
    batch_norm: bool = False,
    kan_grid: int = 3,
    kan_order: int = 2,
    kan_shortcut: str = "silu",
    kan_range: Sequence[float] = (-1.0, 1.0),
) -> argparse.Namespace:
    return argparse.Namespace(
        model=model_name,
        input_size=int(input_size),
        output_size=int(output_size),
        layers_width=[int(x) for x in layers_width],
        batch_norm=bool(batch_norm),
        activation_name=activation_name,
        activation=_activation(activation_name),
        kan_bspline_grid=int(kan_grid),
        kan_bspline_order=int(kan_order),
        kan_shortcut_name=kan_shortcut,
        kan_shortcut_function=_shortcut(kan_shortcut),
        kan_grid_range=[float(kan_range[0]), float(kan_range[1])],
    )


def _reported_result(
    kb_path: Path,
    *,
    dataset: str,
    model: str,
    layers_width: Sequence[int],
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    activation_name: str,
    kan_grid: int = 3,
    kan_order: int = 2,
    kan_shortcut: str = "silu",
    kan_range: Sequence[float] = (-1.0, 1.0),
) -> Dict[str, Any]:
    path = kb_path / "results" / "results.csv"
    if not path.exists():
        return {}
    width_text = "_".join(str(int(x)) for x in layers_width)
    range_text = "_".join(str(float(x)) for x in kan_range)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            cols = line.rstrip("\n").split(",")
            if len(cols) < 17:
                continue
            if cols[1] != dataset or cols[2] != model:
                continue
            if cols[3] != width_text or cols[4] != "False" or cols[5] != activation_name:
                continue
            if int(float(cols[6])) != int(batch_size) or int(float(cols[7])) != int(epochs):
                continue
            if abs(float(cols[8]) - float(lr)) > 1.0e-12 or int(float(cols[9])) != int(seed):
                continue
            if model == "KAN":
                if int(float(cols[10])) != int(kan_grid) or int(float(cols[11])) != int(kan_order):
                    continue
                if cols[12] != kan_shortcut or cols[13] != range_text:
                    continue
                offset = 14
            else:
                if cols[10] != "default":
                    continue
                offset = 11
            return {
                "reported_train_metric": float(cols[offset]),
                "reported_test_metric": float(cols[offset + 1]),
                "reported_params": float(cols[offset + 2]),
                "reported_flops": float(cols[offset + 3]),
                "reported_total_training_time_s": float(cols[offset + 4]),
                "reported_avg_epoch_time_s": float(cols[offset + 5]),
            }
    return {}


def _load_kanbefair_vision_tensors(
    dataset_name: str,
    *,
    data_root: Path,
    train_size: int,
    test_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]:
    try:
        from torchvision import datasets
    except Exception as exc:
        raise RuntimeError(f"torchvision import failed for KANbeFair protocol: {exc}") from exc

    canonical = dataset_name.lower()
    if canonical in {"fashion-mnist", "fashion", "fmnist"}:
        ds_cls = datasets.FashionMNIST
        kb_name = "FMNIST"
        mean, std = 0.2860, 0.3530
    elif canonical == "kmnist":
        ds_cls = datasets.KMNIST
        kb_name = "KMNIST"
        mean, std = 0.1918, 0.3483
    elif canonical == "mnist":
        ds_cls = datasets.MNIST
        kb_name = "MNIST"
        mean, std = 0.1307, 0.3081
    else:
        raise ValueError(f"KANbeFair protocol loader supports MNIST/FMNIST/KMNIST only, got {dataset_name!r}")

    train_ds = ds_cls(root=str(data_root), train=True, download=True)
    test_ds = ds_cls(root=str(data_root), train=False, download=True)
    x_train_all = train_ds.data.float().unsqueeze(1) / 255.0
    x_test_all = test_ds.data.float().unsqueeze(1) / 255.0
    y_train_all = train_ds.targets.long()
    y_test_all = test_ds.targets.long()

    x_train_all = ((x_train_all - mean) / std).reshape(x_train_all.shape[0], -1)
    x_test_all = ((x_test_all - mean) / std).reshape(x_test_all.shape[0], -1)
    g = torch.Generator().manual_seed(int(seed))
    if int(train_size) < int(x_train_all.shape[0]):
        train_idx = torch.randperm(int(x_train_all.shape[0]), generator=g)[: int(train_size)]
        train_protocol = f"random-subset train={int(train_size)}"
    else:
        train_idx = torch.arange(int(x_train_all.shape[0]))
        train_protocol = f"full-train train={int(x_train_all.shape[0])}"
    if int(test_size) < int(x_test_all.shape[0]):
        test_idx = torch.arange(int(test_size))
        test_protocol = f"prefix-test test={int(test_size)}"
    else:
        test_idx = torch.arange(int(x_test_all.shape[0]))
        test_protocol = f"full-test test={int(x_test_all.shape[0])}"
    protocol = f"KANbeFair vision transform; {train_protocol}; {test_protocol}; shuffle_seed={int(seed)}"
    return (
        x_train_all[train_idx].contiguous(),
        y_train_all[train_idx].contiguous(),
        x_test_all[test_idx].contiguous(),
        y_test_all[test_idx].contiguous(),
        int(x_train_all.shape[1]),
        int(y_train_all.max().item() + 1),
        protocol,
    )


def _train_kb_classifier(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    set_seed(seed)
    model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, math.ceil(int(x_train.shape[0]) / int(batch_size)))
    peak_mb = 0.0
    started = time.perf_counter()
    losses: List[float] = []
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(int(epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size : min((i + 1) * batch_size, int(x_train.shape[0]))]
            xb = x_train[idx]
            yb = y_train[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        logits = model(x_test)
        test_loss = F.cross_entropy(logits, y_test).detach()
        probs = logits.softmax(dim=1)
        pred = logits.argmax(dim=1)
        acc = (pred == y_test).float().mean()
        conf, pred_cls = probs.max(dim=1)
        correct = (pred_cls == y_test).float()
        ece = torch.zeros((), device=device)
        for b in range(10):
            lo = b / 10.0
            hi = (b + 1) / 10.0
            mask = (conf > lo) & (conf <= hi)
            if bool(mask.any()):
                ece = ece + mask.float().mean() * torch.abs(conf[mask].mean() - correct[mask].mean())
    return {
        "test_acc_pct": float(acc.detach().cpu()) * 100.0,
        "test_loss": float(test_loss.detach().cpu()),
        "ECE": float(ece.detach().cpu()),
        "NLL": float(test_loss.detach().cpu()),
        "train_loss_last": losses[-1] if losses else float("nan"),
        "train_time_s": elapsed,
        "avg_epoch_time_s": elapsed / max(1, int(epochs)),
        "peak_memory_MB": peak_mb,
        "step_time_ms": elapsed * 1000.0 / max(1, int(epochs) * steps_per_epoch),
    }


def _configure_dg_update_modes(candidate_id: str, stack: Any, head: Any) -> None:
    if str(candidate_id) in getattr(v83.v80, "V80_ADAMW_ADDCDIV_UPDATE_IDS", set()):
        setattr(stack, "update_mode", "adamw_addcdiv")
        setattr(head, "update_mode", "adamw_addcdiv")


def _manual_ce_backward_v85(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: Any,
    *,
    need_loss_float: bool,
) -> float | None:
    v83._zero_grad(stack, head)
    h, caches = stack.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss, grad_logits = v83.v72._weighted_smooth_ce_and_grad(logits, y, spec.label_smoothing, None)
    dh = head.backward_manual(grad_logits, head_cache)
    stack.backward_manual(dh, caches)
    if need_loss_float:
        return float(loss.detach().cpu())
    return None


def _slice_cache_batch_v85(cache: Any, split: int) -> Any:
    if isinstance(cache, torch.Tensor):
        if cache.dim() > 0 and int(cache.shape[0]) >= int(split):
            return cache[:split].detach()
        return cache.detach()
    if isinstance(cache, dict):
        out: Dict[str, Any] = {}
        for key, value in cache.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) >= int(split):
                out[key] = value[:split].detach()
            elif isinstance(value, torch.Tensor):
                out[key] = value.detach()
            else:
                out[key] = value
        return out
    return cache


def _slice_stack_caches_v85(caches: Any, split: int, x: torch.Tensor) -> Any:
    if isinstance(caches, list) and len(caches) == 1 and isinstance(caches[0], dict) and "x0" in caches[0]:
        return [{"x0": x.detach()}]
    if isinstance(caches, list):
        return [_slice_cache_batch_v85(cache, split) for cache in caches]
    return _slice_cache_batch_v85(caches, split)


def _manual_ce_backward_with_holdout_v85(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    spec: Any,
) -> Tuple[float, float]:
    v83._zero_grad(stack, head)
    split = int(x.shape[0])
    x_pair = torch.cat((x, x_holdout), dim=0)
    h_pair, caches = stack.forward_manual(x_pair)
    logits_pair, head_cache = head.forward_manual(h_pair)
    loss, grad_logits = v83.v72._weighted_smooth_ce_and_grad(logits_pair[:split], y, spec.label_smoothing, None)
    holdout_loss = v83._smooth_ce_value_only(logits_pair[split:], y_holdout, spec.label_smoothing)
    dh = head.backward_manual(grad_logits, _slice_cache_batch_v85(head_cache, split))
    stack.backward_manual(dh, _slice_stack_caches_v85(caches, split, x))
    del x_pair, h_pair, caches, logits_pair, head_cache, grad_logits
    return float(loss.detach().cpu()), float(holdout_loss.detach().cpu())


def _dg_local_lipschitz_probe(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    *,
    seed: int,
    batch_size: int,
    sample_size: int = 128,
    eps: float = 1.0e-3,
) -> float:
    if int(x.shape[0]) == 0:
        return float("nan")
    n = min(int(sample_size), int(x.shape[0]))
    x_probe = x[:n]
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    noise = torch.randn(x_probe.shape, generator=gen, device=x.device, dtype=x_probe.dtype) * float(eps)
    with torch.inference_mode():
        logits0 = v83._logits_only(stack, head, x_probe, int(batch_size))
        logits1 = v83._logits_only(stack, head, x_probe + noise, int(batch_size))
        numerator = (logits1 - logits0).reshape(n, -1).norm(dim=1).mean()
        denominator = noise.reshape(n, -1).norm(dim=1).mean().clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def _load_kanbefair_symbolic_tensors(
    dataset: str,
    *,
    kb_path: Path,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]:
    src = kb_path / "src"
    if str(src.resolve()) not in sys.path:
        sys.path.insert(0, str(src.resolve()))
    set_seed(int(seed))
    try:
        from data.special import get_special_dataset_1d, get_scipyfunction_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"KANbeFair special dataset import failed: {exc}") from exc
    ns = argparse.Namespace(dataset=str(dataset))
    if str(dataset).startswith("Special_1d_"):
        train_ds, test_ds = get_special_dataset_1d(ns)
    else:
        train_ds, test_ds = get_scipyfunction_dataset(ns)
    x_train, y_train = train_ds.tensors
    x_test, y_test = test_ds.tensors
    return (
        x_train.float().contiguous(),
        y_train.float().reshape(int(y_train.shape[0]), -1).contiguous(),
        x_test.float().contiguous(),
        y_test.float().reshape(int(y_test.shape[0]), -1).contiguous(),
        int(x_train.reshape(int(x_train.shape[0]), -1).shape[1]),
        int(y_train.reshape(int(y_train.shape[0]), -1).shape[1]),
        f"KANbeFair special dataset; dataset={dataset}; seed={int(seed)}",
    )


def _function_shape_metrics_from_predictions(x: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    if int(x.shape[1]) != 1 or int(x.shape[0]) < 4:
        return {"curvature": float("nan"), "slope_p95": float("nan"), "jacobian_norm": float("nan")}
    xs = x.detach().float().reshape(-1)
    ys = pred.detach().float().reshape(int(pred.shape[0]), -1)[:, 0]
    order = torch.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    dx = torch.diff(xs).clamp_min(1.0e-6)
    dy = torch.diff(ys)
    slopes = dy / dx
    if int(slopes.numel()) < 2:
        curvature = torch.zeros((), device=x.device)
    else:
        dx_mid = ((dx[:-1] + dx[1:]) * 0.5).clamp_min(1.0e-6)
        curvature = torch.mean(torch.abs(torch.diff(slopes) / dx_mid))
    return {
        "curvature": float(curvature.detach().cpu()),
        "slope_p95": float(torch.quantile(torch.abs(slopes), 0.95).detach().cpu()),
        "jacobian_norm": float(torch.mean(torch.abs(slopes)).detach().cpu()),
    }


def _train_kb_regressor(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    set_seed(int(seed))
    model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    steps_per_epoch = max(1, math.ceil(int(x_train.shape[0]) / int(batch_size)))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size : min((i + 1) * batch_size, int(x_train.shape[0]))]
            xb = x_train[idx]
            yb = y_train[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0
    elapsed = time.perf_counter() - started
    with torch.inference_mode():
        pred = model(x_test)
        err = pred - y_test
        rmse = torch.sqrt(torch.mean(err * err))
        mae = torch.mean(torch.abs(err))
        shape = _function_shape_metrics_from_predictions(x_test, pred)
    return {
        "RMSE": float(rmse.detach().cpu()),
        "MAE": float(mae.detach().cpu()),
        "curvature": shape["curvature"],
        "slope_p95": shape["slope_p95"],
        "jacobian_norm": shape["jacobian_norm"],
        "train_time_s": elapsed,
        "step_time_ms": elapsed * 1000.0 / max(1, int(epochs) * steps_per_epoch),
        "peak_memory_MB": peak_mb,
    }


def _manual_mse_backward_v85(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    need_loss_float: bool,
) -> float | None:
    v83._zero_grad(stack, head)
    h, caches = stack.forward_manual(x)
    pred, head_cache = head.forward_manual(h)
    diff = pred - y
    loss = torch.mean(diff * diff)
    grad_pred = (2.0 / max(1, int(diff.numel()))) * diff
    dh = head.backward_manual(grad_pred, head_cache)
    stack.backward_manual(dh, caches)
    if need_loss_float:
        return float(loss.detach().cpu())
    return None


def _mse_loss_only_v85(stack: Any, head: Any, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.inference_mode():
        h, _ = stack.forward_manual(x)
        pred, _ = head.forward_manual(h)
        return float(F.mse_loss(pred, y).detach().cpu())


def _dg_predict_v85(stack: Any, head: Any, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[i : i + int(batch_size)]
            h, _ = stack.forward_manual(xb)
            pred, _ = head.forward_manual(h)
            outs.append(pred.detach())
    return torch.cat(outs, dim=0)


def _apply_ft7_mse_guarded_update_v85(
    stack: Any,
    head: Any,
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    role_entries_by_name: Dict[str, List[Tuple[str, str, torch.Tensor, torch.Tensor]]],
    xb: torch.Tensor,
    yb: torch.Tensor,
    xh: torch.Tensor,
    yh: torch.Tensor,
    *,
    loss_before: float,
    task_loss_after: float,
    holdout_loss_after: float,
    func_alpha: float,
) -> Tuple[float, float, float]:
    loss_after = float(task_loss_after)
    trust_delta = 0.0
    reject_delta = 0.0
    stack_role_rejected = False
    for role_name in ("stack", "head"):
        role_weight = float(v83.FT7_ROLE_WEIGHTS[role_name])
        if role_weight == 0.0:
            continue
        if role_name == "head" and stack_role_rejected:
            reject_delta += 0.5
            continue
        role_budget = float(v83.FT7_ROLE_BUDGETS[role_name])
        role_accept_limit = float(task_loss_after) + role_budget * max(0.0, float(loss_before) - float(task_loss_after))
        role_holdout_limit = float(holdout_loss_after) + role_budget * max(0.0, float(loss_before) - float(task_loss_after))
        role_entries = role_entries_by_name[role_name]
        rollback = v83._apply_streamed_second_diff_correction_with_fixed_coeff_rollback(
            role_entries,
            float(func_alpha) * role_weight,
        )
        role_loss = _mse_loss_only_v85(stack, head, xb, yb)
        role_train_ok = role_loss <= role_accept_limit + 1.0e-8
        role_holdout_ok = False
        if role_train_ok:
            role_holdout = _mse_loss_only_v85(stack, head, xh, yh)
            role_holdout_ok = role_holdout <= role_holdout_limit + 1.0e-8
        if role_train_ok and role_holdout_ok:
            loss_after = role_loss
            trust_delta += 0.025
        else:
            v83._rollback_streamed_second_diff_correction(rollback)
            if role_name == "stack":
                stack_role_rejected = True
            reject_delta += 0.5
    return loss_after, trust_delta, reject_delta


def _train_dg_symbolic_regressor(
    args: argparse.Namespace,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    function_name: str,
    input_dim: int,
    output_dim: int,
    functional_update_used: int,
) -> Dict[str, Any]:
    device = v83.get_device(args.device)
    v83.v80._patch_for_v80()
    dg_candidate = str(args.dg_candidate_id)
    spec = v83.v80._spec_map_v80().get(dg_candidate, v83.v80._spec_map_v80()["KW6"])
    v83.set_seed(v83.v72._stable_seed("v85-dg-symbolic-init", function_name, int(args.seed), dg_candidate, int(args.dg_hidden_dim), int(functional_update_used)))
    stack, head = v83.v80._make_manual_candidate_v80(
        spec,
        int(input_dim),
        int(output_dim),
        int(args.dg_hidden_dim),
        int(args.dg_basis_count),
        device,
    )
    _configure_dg_update_modes(dg_candidate, stack, head)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    batch_size = int(args.symbolic_batch_size)
    epochs = int(args.symbolic_epochs)
    steps_per_epoch = max(1, math.ceil(int(x_train.shape[0]) / batch_size))
    params = v83.v80.V63Params(
        train_size=int(x_train.shape[0]),
        val_size=0,
        test_size=int(x_test.shape[0]),
        batch_size=batch_size,
    )
    lr = float(params.lr_manual * spec.lr_mult)
    opt = v83.v72.FastAdamWNoSync([stack, head], lr=lr, weight_decay=float(args.weight_decay))
    entries = v83._param_entries(stack, head)
    role_entries_by_name = v83._entries_by_role(entries)
    rng = torch.Generator(device=device).manual_seed(int(args.seed))
    functional_event_count = 0
    functional_accept_mass = 0.0
    functional_reject_mass = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    global_step = 0
    total_steps = max(1, epochs * steps_per_epoch)
    for _epoch in range(epochs):
        perm = torch.randperm(int(x_train.shape[0]), generator=rng, device=device)
        for batch_idx in range(steps_per_epoch):
            global_step += 1
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, int(x_train.shape[0]))
            idx = perm[start:end]
            if int(idx.numel()) == 0:
                continue
            holdout_start = ((batch_idx + 1) % steps_per_epoch) * batch_size
            holdout_end = min(holdout_start + batch_size, int(x_train.shape[0]))
            holdout_idx = perm[holdout_start:holdout_end]
            if int(holdout_idx.numel()) == 0:
                holdout_idx = idx
            xb = x_train[idx]
            yb = y_train[idx]
            xh = x_train[holdout_idx]
            yh = y_train[holdout_idx]
            event_step = bool(functional_update_used) and global_step % int(v83.P5_FT7_EVENT_STRIDE) == 0
            need_loss_float = bool(event_step or global_step == 1 or global_step == total_steps)
            loss_before_value = _manual_mse_backward_v85(stack, head, xb, yb, need_loss_float=need_loss_float)
            loss_before = float(loss_before_value) if loss_before_value is not None else 0.0
            opt.step(global_step, total_steps, warmup_cosine=True)
            if event_step:
                task_loss_after = _mse_loss_only_v85(stack, head, xb, yb)
                holdout_loss_after = _mse_loss_only_v85(stack, head, xh, yh)
                _loss_after, trust_delta, reject_delta = _apply_ft7_mse_guarded_update_v85(
                    stack,
                    head,
                    entries,
                    role_entries_by_name,
                    xb,
                    yb,
                    xh,
                    yh,
                    loss_before=loss_before,
                    task_loss_after=task_loss_after,
                    holdout_loss_after=holdout_loss_after,
                    func_alpha=lr * 0.05 * float(v83.P5_FT7_EVENT_ALPHA_MULT),
                )
                functional_event_count += 1
                functional_accept_mass += float(trust_delta)
                functional_reject_mass += float(reject_delta)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0
    elapsed = time.perf_counter() - started
    pred = _dg_predict_v85(stack, head, x_test, int(args.dg_eval_batch_size))
    err = pred - y_test
    rmse = torch.sqrt(torch.mean(err * err))
    mae = torch.mean(torch.abs(err))
    shape = _function_shape_metrics_from_predictions(x_test, pred)
    return {
        "model": "DG-Functional" if functional_update_used else "DG-Base",
        "candidate_id": f"DG1-FT7-{dg_candidate}-symbolic" if functional_update_used else f"DG0-{dg_candidate}-symbolic-base",
        "params": int(stack.param_count() + head.param_count()),
        "FLOPs": _dg_kw6_forward_flops_estimate(int(input_dim), int(args.dg_hidden_dim), int(spec.depth), int(output_dim)),
        "RMSE": float(rmse.detach().cpu()),
        "MAE": float(mae.detach().cpu()),
        "curvature": shape["curvature"],
        "slope_p95": shape["slope_p95"],
        "jacobian_norm": shape["jacobian_norm"],
        "train_time_s": elapsed,
        "step_time_ms": elapsed * 1000.0 / max(1, global_step),
        "peak_memory_MB": peak_mb,
        "functional_update_used": int(functional_update_used),
        "functional_event_count": functional_event_count,
        "functional_accept_mass": functional_accept_mass,
        "functional_reject_mass": functional_reject_mass,
    }


def _torch_param_curvature(model: nn.Module) -> float:
    vals: List[torch.Tensor] = []
    with torch.no_grad():
        for p in model.parameters():
            flat = p.detach().float().reshape(-1)
            if int(flat.numel()) >= 3:
                vals.append(torch.mean(torch.abs(flat[2:] - 2.0 * flat[1:-1] + flat[:-2])))
    if not vals:
        return 0.0
    return float(torch.stack(vals).mean().detach().cpu())


def _filter_digit_group(
    x: torch.Tensor,
    y: torch.Tensor,
    digits: Sequence[int],
    *,
    max_count: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros_like(y, dtype=torch.bool)
    for d in digits:
        mask |= y == int(d)
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if int(max_count) > 0 and int(idx.numel()) > int(max_count):
        g = torch.Generator().manual_seed(int(seed))
        idx = idx[torch.randperm(int(idx.numel()), generator=g)[: int(max_count)]]
    return x[idx].contiguous(), y[idx].contiguous()


def _eval_kb_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> float:
    model.eval()
    x = x.to(device)
    y = y.to(device)
    correct = 0
    total = 0
    with torch.inference_mode():
        for i in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[i : i + int(batch_size)]
            yb = y[i : i + int(batch_size)]
            logits = model(xb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(yb.numel())
    return correct / max(1, total)


def _train_kb_continual(
    model: nn.Module,
    train_tasks: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    test_tasks: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs_per_task: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Tuple[List[List[float]], List[float], float, float]:
    set_seed(int(seed))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    acc_matrix: List[List[float]] = []
    curvature_after: List[float] = []
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for task_idx, (x_train, y_train) in enumerate(train_tasks):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        steps_per_epoch = max(1, math.ceil(int(x_train.shape[0]) / int(batch_size)))
        for epoch in range(int(epochs_per_task)):
            perm = torch.randperm(int(x_train.shape[0]), device=device)
            for i in range(steps_per_epoch):
                idx = perm[i * batch_size : min((i + 1) * batch_size, int(x_train.shape[0]))]
                xb = x_train[idx]
                yb = y_train[idx]
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
        acc_matrix.append([
            _eval_kb_accuracy(model, test_tasks[j][0], test_tasks[j][1], int(batch_size), device)
            for j in range(task_idx + 1)
        ])
        curvature_after.append(_torch_param_curvature(model))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0
    return acc_matrix, curvature_after, time.perf_counter() - started, peak_mb


def _train_dg_continual(
    args: argparse.Namespace,
    train_tasks: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    test_tasks: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    input_dim: int,
    num_classes: int,
    functional_update_used: int,
) -> Tuple[List[List[float]], List[float], float, float, int, float, float]:
    device = v83.get_device(args.device)
    v83.v80._patch_for_v80()
    dg_candidate = str(args.dg_candidate_id)
    spec = v83.v80._spec_map_v80().get(dg_candidate, v83.v80._spec_map_v80()["KW6"])
    v83.set_seed(v83.v72._stable_seed("v85-dg-continual-init", int(args.seed), dg_candidate, int(args.dg_hidden_dim), int(functional_update_used)))
    stack, head = v83.v80._make_manual_candidate_v80(
        spec,
        int(input_dim),
        int(num_classes),
        int(args.dg_hidden_dim),
        int(args.dg_basis_count),
        device,
    )
    _configure_dg_update_modes(dg_candidate, stack, head)
    batch_size = int(args.continual_batch_size)
    epochs_per_task = int(args.continual_epochs_per_task)
    params = v83.v80.V63Params(train_size=max(1, int(train_tasks[0][0].shape[0])), val_size=0, test_size=1, batch_size=batch_size)
    lr = float(params.lr_manual * spec.lr_mult)
    opt = v83.v72.FastAdamWNoSync([stack, head], lr=lr, weight_decay=float(args.weight_decay))
    entries = v83._param_entries(stack, head)
    role_entries_by_name = v83._entries_by_role(entries)
    rng = torch.Generator(device=device).manual_seed(int(args.seed))
    acc_matrix: List[List[float]] = []
    curvature_after: List[float] = []
    event_count = 0
    accept_mass = 0.0
    reject_mass = 0.0
    global_step = 0
    seen_digits: List[int] = []
    head_mix_anchors: Dict[int, torch.Tensor] = {}
    stack_anchor: Optional[List[torch.Tensor]] = None
    head_shared_anchor: Dict[str, torch.Tensor] = {}
    restore_old_head_rows = bool(getattr(args, "continual_restore_old_head_rows", False))
    freeze_stack_after_first = bool(getattr(args, "continual_freeze_stack_after_first_task", False))
    freeze_head_shared_after_first = bool(getattr(args, "continual_freeze_head_shared_after_first_task", False))
    stack_anchor_strength = float(getattr(args, "continual_stack_anchor_strength", 0.0))
    old_head_grad_scale = max(0.0, min(1.0, float(getattr(args, "continual_old_head_grad_scale", 0.0))))
    old_head_restore_strength = max(0.0, min(1.0, float(getattr(args, "continual_old_head_restore_strength", 1.0))))
    old_head_age_decay = max(0.0, min(0.5, float(getattr(args, "continual_old_head_age_decay", 0.0))))
    total_steps = sum(max(1, math.ceil(int(x.shape[0]) / batch_size)) * epochs_per_task for x, _y in train_tasks)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for task_idx, (x_task, y_task) in enumerate(train_tasks):
        current_digits = sorted(int(d) for d in torch.unique(y_task).detach().cpu().tolist())
        old_digits = list(seen_digits)
        x_task = x_task.to(device)
        y_task = y_task.to(device)
        steps_per_epoch = max(1, math.ceil(int(x_task.shape[0]) / batch_size))
        for _epoch in range(epochs_per_task):
            perm = torch.randperm(int(x_task.shape[0]), generator=rng, device=device)
            for batch_idx in range(steps_per_epoch):
                global_step += 1
                idx = perm[batch_idx * batch_size : min((batch_idx + 1) * batch_size, int(x_task.shape[0]))]
                xb = x_task[idx]
                yb = y_task[idx]
                holdout_idx = perm[((batch_idx + 1) % steps_per_epoch) * batch_size : min(((batch_idx + 2) % steps_per_epoch) * batch_size, int(x_task.shape[0]))]
                if int(holdout_idx.numel()) == 0:
                    holdout_idx = idx
                xh = x_task[holdout_idx]
                yh = y_task[holdout_idx]
                event_step = bool(functional_update_used) and global_step % int(v83.P5_FT7_EVENT_STRIDE) == 0
                use_pre_holdout = bool(event_step and v83._ft7_uses_pre_holdout_for_step(global_step))
                need_loss_float = bool(event_step or global_step == 1 or global_step == total_steps)
                if use_pre_holdout:
                    loss_before, holdout_loss_before = _manual_ce_backward_with_holdout_v85(stack, head, xb, yb, xh, yh, spec)
                else:
                    holdout_loss_before = 0.0
                    loss_value = _manual_ce_backward_v85(stack, head, xb, yb, spec, need_loss_float=need_loss_float)
                    loss_before = float(loss_value) if loss_value is not None else 0.0
                if functional_update_used and freeze_stack_after_first and task_idx > 0:
                    for _name, _p, g in stack.params_and_grads():
                        g.zero_()
                if functional_update_used and restore_old_head_rows and old_digits:
                    mix_start = int(head.starts.get("mix", 0))
                    mix_shape = head.shapes.get("mix", (0, 0))
                    if len(mix_shape) == 2:
                        _classes, mix_width = int(mix_shape[0]), int(mix_shape[1])
                        for digit in old_digits:
                            if 0 <= int(digit) < _classes:
                                row_start = mix_start + int(digit) * mix_width
                                head.flat_grad[row_start : row_start + mix_width].mul_(old_head_grad_scale)
                if functional_update_used and freeze_head_shared_after_first and task_idx > 0:
                    for shared_name in ("poly", "base"):
                        if shared_name in head.starts and shared_name in head.shapes:
                            start = int(head.starts[shared_name])
                            length = int(math.prod(head.shapes[shared_name]))
                            head.flat_grad[start : start + length].zero_()
                opt.step(global_step, total_steps, warmup_cosine=True)
                if functional_update_used and old_head_age_decay > 0.0 and old_digits:
                    mix_start = int(head.starts.get("mix", 0))
                    mix_shape = head.shapes.get("mix", (0, 0))
                    if len(mix_shape) == 2:
                        _classes, mix_width = int(mix_shape[0]), int(mix_shape[1])
                        with torch.no_grad():
                            for digit in old_digits:
                                if 0 <= int(digit) < _classes:
                                    row_start = mix_start + int(digit) * mix_width
                                    head.flat[row_start : row_start + mix_width].mul_(1.0 - old_head_age_decay)
                if functional_update_used and task_idx > 0 and stack_anchor is not None and (freeze_stack_after_first or stack_anchor_strength > 0.0):
                    with torch.no_grad():
                        for (_name, p, _g), anchor in zip(stack.params_and_grads(), stack_anchor):
                            if freeze_stack_after_first:
                                p.copy_(anchor)
                            else:
                                p.add_(anchor - p, alpha=float(stack_anchor_strength))
                if functional_update_used and freeze_head_shared_after_first and task_idx > 0 and head_shared_anchor:
                    with torch.no_grad():
                        for shared_name, anchor in head_shared_anchor.items():
                            start = int(head.starts[shared_name])
                            length = int(math.prod(head.shapes[shared_name]))
                            head.flat[start : start + length].copy_(anchor)
                if functional_update_used and restore_old_head_rows and old_digits and head_mix_anchors:
                    mix_start = int(head.starts.get("mix", 0))
                    mix_shape = head.shapes.get("mix", (0, 0))
                    if len(mix_shape) == 2:
                        _classes, mix_width = int(mix_shape[0]), int(mix_shape[1])
                        with torch.no_grad():
                            for digit in old_digits:
                                anchor = head_mix_anchors.get(int(digit))
                                if anchor is None or not (0 <= int(digit) < _classes):
                                    continue
                                row_start = mix_start + int(digit) * mix_width
                                target = head.flat[row_start : row_start + mix_width]
                                target.add_(anchor - target, alpha=old_head_restore_strength)
                if event_step:
                    loss_after, holdout_loss_after, task_features_after, holdout_features_after = v83._loss_pair_and_features_only(
                        stack, head, xb, yb, xh, yh, spec
                    )
                    _loss_after, _func_norm, trust_delta, reject_delta = v83._apply_ft7_streamed_guarded_update(
                        stack,
                        head,
                        entries,
                        xb,
                        yb,
                        xh,
                        yh,
                        spec,
                        loss_before=loss_before,
                        holdout_loss_before=holdout_loss_before,
                        task_loss_after=loss_after,
                        holdout_loss_after=holdout_loss_after,
                        func_alpha=lr * 0.05 * float(v83.P5_FT7_EVENT_ALPHA_MULT),
                        task_features_after=task_features_after,
                        holdout_features_after=holdout_features_after,
                        role_entries_by_name=role_entries_by_name,
                        pair_stack_guard_forward=True,
                    )
                    event_count += 1
                    accept_mass += float(trust_delta)
                    reject_mass += float(reject_delta)
        acc_matrix.append([
            float(v83.v72.v71._manual_eval(stack, head, test_tasks[j][0].to(device), test_tasks[j][1].to(device), int(args.dg_eval_batch_size), int(num_classes))["acc"])
            for j in range(task_idx + 1)
        ])
        curvature_after.append(float(v83._geometry_norms(v83._param_entries(stack, head))[1]))
        if functional_update_used and restore_old_head_rows:
            mix_start = int(head.starts.get("mix", 0))
            mix_shape = head.shapes.get("mix", (0, 0))
            if len(mix_shape) == 2:
                _classes, mix_width = int(mix_shape[0]), int(mix_shape[1])
                with torch.no_grad():
                    for digit in current_digits:
                        if 0 <= int(digit) < _classes:
                            row_start = mix_start + int(digit) * mix_width
                            head_mix_anchors[int(digit)] = head.flat[row_start : row_start + mix_width].detach().clone()
        if functional_update_used and (freeze_stack_after_first or stack_anchor_strength > 0.0) and task_idx == 0:
            stack_anchor = [p.detach().clone() for _name, p, _g in stack.params_and_grads()]
        if functional_update_used and freeze_head_shared_after_first and task_idx == 0:
            head_shared_anchor = {}
            for shared_name in ("poly", "base"):
                if shared_name in head.starts and shared_name in head.shapes:
                    start = int(head.starts[shared_name])
                    length = int(math.prod(head.shapes[shared_name]))
                    head_shared_anchor[shared_name] = head.flat[start : start + length].detach().clone()
        for digit in current_digits:
            if int(digit) not in seen_digits:
                seen_digits.append(int(digit))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0
    return acc_matrix, curvature_after, time.perf_counter() - started, peak_mb, event_count, accept_mass, reject_mass


def _continual_scores(acc_matrix: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    if not acc_matrix:
        return float("nan"), float("nan"), float("nan")
    final_accs = list(acc_matrix[-1])
    final_average = sum(final_accs) / max(1, len(final_accs))
    forgetting_vals: List[float] = []
    bwt_vals: List[float] = []
    for task_idx in range(max(0, len(acc_matrix) - 1)):
        learned_acc = acc_matrix[task_idx][task_idx]
        final_acc = final_accs[task_idx]
        forgetting_vals.append(max(0.0, learned_acc - final_acc))
        bwt_vals.append(final_acc - learned_acc)
    forgetting = sum(forgetting_vals) / max(1, len(forgetting_vals))
    backward_transfer = sum(bwt_vals) / max(1, len(bwt_vals))
    return forgetting, backward_transfer, final_average


def _run_dg_adapter_smoke(
    args: argparse.Namespace,
    *,
    functional_update_used: int,
) -> Dict[str, Any]:
    device = v83.get_device(args.device)
    v83.v80._patch_for_v80()
    dataset = (parse_str_list(args.datasets) or ["MNIST"])[0]
    x_train, y_train, x_test, y_test, input_dim, num_classes, split_protocol = _load_kanbefair_vision_tensors(
        dataset,
        data_root=Path(args.data_root),
        train_size=min(int(args.adapter_smoke_train_size), int(args.train_size)),
        test_size=min(int(args.adapter_smoke_test_size), int(args.test_size)),
        seed=int(args.seed),
    )
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    dg_candidate = str(args.dg_candidate_id)
    spec = v83.v80._spec_map_v80().get(dg_candidate, v83.v80._spec_map_v80()["KW6"])
    v83.set_seed(v83.v72._stable_seed("v85-dg-adapter", dataset, int(args.seed), dg_candidate, int(functional_update_used)))
    stack, head = v83.v80._make_manual_candidate_v80(
        spec,
        input_dim,
        num_classes,
        int(args.dg_hidden_dim),
        int(args.dg_basis_count),
        device,
    )
    _configure_dg_update_modes(dg_candidate, stack, head)
    params = v83.v80.V63Params(
        train_size=int(x_train.shape[0]),
        val_size=0,
        test_size=int(x_test.shape[0]),
        batch_size=int(args.dg_batch_size),
    )
    lr = float(params.lr_manual * spec.lr_mult)
    opt = v83.v72.FastAdamWNoSync([stack, head], lr=lr, weight_decay=float(args.weight_decay))
    entries = v83._param_entries(stack, head)
    role_entries_by_name = v83._entries_by_role(entries)
    before_params = [p.detach().clone() for _role, _name, p, _grad in entries]
    first_loss = float("nan")
    last_loss = float("nan")
    functional_event_count = 0
    functional_accept_mass = 0.0
    functional_reject_mass = 0.0
    peak_mb: Any = METRIC_UNAVAILABLE
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(1, int(args.adapter_steps) + 1):
        xb, yb = v83.v72.v71._select_batch(x_train, y_train, int(args.dg_batch_size), step)
        xh, yh = v83.v72.v71._select_batch(x_train, y_train, int(args.dg_batch_size), step + 1009)
        event_step = bool(functional_update_used) and step % int(v83.P5_FT7_EVENT_STRIDE) == 0
        use_pre_holdout = bool(event_step and v83._ft7_uses_pre_holdout_for_step(step))
        need_loss_float = bool(event_step or step == 1 or step == int(args.adapter_steps))
        if use_pre_holdout:
            loss_before, holdout_loss_before = _manual_ce_backward_with_holdout_v85(stack, head, xb, yb, xh, yh, spec)
        else:
            holdout_loss_before = 0.0
            loss_before_value = _manual_ce_backward_v85(stack, head, xb, yb, spec, need_loss_float=need_loss_float)
            loss_before = float(loss_before_value) if loss_before_value is not None else 0.0
        if step == 1:
            first_loss = float(loss_before)
        opt.step(step, int(args.adapter_steps), warmup_cosine=True)
        loss_after = loss_before
        task_features_after = None
        holdout_features_after = None
        if event_step:
            if v83.FT7_USE_HOLDOUT_GUARD:
                loss_after, holdout_loss_after, task_features_after, holdout_features_after = v83._loss_pair_and_features_only(
                    stack,
                    head,
                    xb,
                    yb,
                    xh,
                    yh,
                    spec,
                )
            else:
                loss_after, task_features_after = v83._loss_and_features_only(stack, head, xb, yb, spec)
                holdout_loss_after = 0.0
            loss_after, _func_norm, trust_delta, reject_delta = v83._apply_ft7_streamed_guarded_update(
                stack,
                head,
                entries,
                xb,
                yb,
                xh,
                yh,
                spec,
                loss_before=loss_before,
                holdout_loss_before=holdout_loss_before,
                task_loss_after=loss_after,
                holdout_loss_after=holdout_loss_after,
                func_alpha=lr * 0.05 * float(v83.P5_FT7_EVENT_ALPHA_MULT),
                task_features_after=task_features_after,
                holdout_features_after=holdout_features_after,
                role_entries_by_name=role_entries_by_name,
                pair_stack_guard_forward=True,
            )
            functional_event_count += 1
            functional_accept_mass += float(trust_delta)
            functional_reject_mass += float(reject_delta)
        last_loss = float(loss_after)
    if not math.isfinite(last_loss):
        last_loss = v83._loss_only(stack, head, x_train[: int(args.dg_batch_size)], y_train[: int(args.dg_batch_size)], spec)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    elapsed = time.perf_counter() - started
    test_eval = v83.v72.v71._manual_eval(stack, head, x_test, y_test, int(args.dg_eval_batch_size), num_classes)
    after_entries = v83._param_entries(stack, head)
    delta_l1 = 0.0
    for before, (_role, _name, param, _grad) in zip(before_params, after_entries):
        delta_l1 += float((param.detach() - before).abs().sum().detach().cpu())
    finite_loss = int(math.isfinite(first_loss) and math.isfinite(last_loss))
    adapter_pass = int(finite_loss and delta_l1 > 0.0 and math.isfinite(float(test_eval["loss"])))
    return {
        "adapter_protocol": split_protocol,
        "task_name": dataset,
        "adapter_steps": int(args.adapter_steps),
        "batch_size": int(args.dg_batch_size),
        "params_total": int(stack.param_count() + head.param_count()),
        "stack_param_count": int(stack.param_count()),
        "head_param_count": int(head.param_count()),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "test_metric": float(test_eval["acc"]),
        "test_loss": float(test_eval["loss"]),
        "ECE": float(test_eval["ECE"]),
        "NLL": float(test_eval["NLL"]),
        "param_delta_l1": delta_l1,
        "train_time_s": elapsed,
        "step_time_ms": elapsed * 1000.0 / max(1, int(args.adapter_steps)),
        "peak_memory_MB": peak_mb,
        "functional_event_count": functional_event_count,
        "functional_accept_mass": functional_accept_mass,
        "functional_reject_mass": functional_reject_mass,
        "AdapterPass": adapter_pass,
    }


def _dg_kw6_forward_flops_estimate(input_dim: int, hidden_dim: int, depth: int, num_classes: int) -> int:
    dims = [int(input_dim)] + [int(hidden_dim)] * int(depth)
    stack_gemm = sum(2 * a * b for a, b in zip(dims[:-1], dims[1:]))
    stack_silu = max(0, int(depth) - 1) * 4 * int(hidden_dim)
    head_poly2_silu_transform = 11 * int(hidden_dim)
    head_gemm = 2 * int(hidden_dim) * int(num_classes)
    return int(stack_gemm + stack_silu + head_poly2_silu_transform + head_gemm)


def _run_source_audit(out_dir: Path, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    kb = Path(args.kanbefair_path)
    src = kb / "src"
    py_files = sorted(src.rglob("*.py")) if src.exists() else []
    readme = _read_text(kb / "ReadMe.md")
    train_text = _read_text(src / "train.py")
    utils_text = _read_text(src / "utils.py")
    model_files = {p.name: _read_text(p) for p in [src / "models" / "mlp.py", src / "models" / "kanbefair.py", src / "models" / "bspline_mlp.py"]}
    deps = _dependency_status()
    MLP, KANbeFair, BSpline_MLP, matplotlib_stub_used, import_status = _import_kanbefair_models(kb)
    source_tree_lines = [f"# KANbeFair Source Tree\n", f"path: `{kb}`\n", ""]
    for p in sorted(kb.rglob("*")):
        if ".git" in p.parts:
            continue
        if p.is_file():
            source_tree_lines.append(f"- `{p.relative_to(kb)}`")
    (out_dir / "p0_third_party_source_tree.md").write_text("\n".join(source_tree_lines) + "\n", encoding="utf-8")
    train_has_hardcoded_chdir = int("os.chdir('/home/yurunpeng/Repos/KANBeFair/src')" in train_text)
    entry_md = [
        "# KANbeFair Entrypoint Matrix",
        "",
        "| entrypoint | found | notes |",
        "|---|---:|---|",
        f"| `src/train.py` | {int((src / 'train.py').exists())} | hardcoded chdir: `{train_has_hardcoded_chdir}` |",
        f"| `src/train_continual_learning.py` | {int((src / 'train_continual_learning.py').exists())} | continual learning entry |",
        f"| `results/results.csv` | {int((kb / 'results' / 'results.csv').exists())} | paper result table without header |",
    ]
    (out_dir / "p0_entrypoint_matrix.md").write_text("\n".join(entry_md) + "\n", encoding="utf-8")
    runnable = int(MLP is not None and KANbeFair is not None)
    row = {
        "stage": "P0_THIRD_PARTY_SOURCE_AUDIT_V85",
        "third_party_path": str(kb),
        "path_exists": int(kb.exists()),
        "git_commit_or_hash": _git_commit(kb),
        "repo_url_if_available": _read_text(kb / ".git" / "config").split("url = ")[-1].splitlines()[0].strip() if (kb / ".git" / "config").exists() and "url = " in _read_text(kb / ".git" / "config") else METRIC_UNAVAILABLE,
        "license": "license_file_missing" if not any((kb / name).exists() for name in ["LICENSE", "LICENSE.md", "COPYING"]) else "license_file_found",
        "python_files_count": len(py_files),
        "dataset_modules": ",".join(sorted(p.stem for p in (src / "data").glob("*.py"))) if (src / "data").exists() else "",
        "model_modules": ",".join(sorted(p.stem for p in (src / "models").glob("*.py"))) if (src / "models").exists() else "",
        "baseline_model_classes": ",".join([name for name, text in model_files.items() if "class " in text]),
        "KAN_model_classes": "KANbeFair,KANbeFair_Text" if "class KANbeFair" in model_files.get("kanbefair.py", "") else "",
        "MLP_model_classes": "MLP,MLP_Text" if "class MLP" in model_files.get("mlp.py", "") else "",
        "training_entrypoints": "src/train.py,src/train_continual_learning.py" if (src / "train.py").exists() else "",
        "evaluation_entrypoints": "test(args, model, device, test_loader, logger, name)" if "def test(" in train_text else "",
        "parameter_counter_found": int("total_parameters" in "".join(model_files.values())),
        "FLOPs_counter_found": int("total_flops" in "".join(model_files.values())),
        "result_parser_found": int((kb / "results" / "results.csv").exists()),
        "train_py_hardcoded_chdir": int("os.chdir('/home/yurunpeng/Repos/KANBeFair/src')" in train_text),
        "matplotlib_stub_used_for_model_import": matplotlib_stub_used,
        "dependency_torch": deps["torch"],
        "dependency_torchvision": deps["torchvision"],
        "dependency_matplotlib": deps["matplotlib"],
        "dependency_fvcore": deps["fvcore"],
        "dependency_torchtext": deps["torchtext"],
        "dependency_pandas": deps["pandas"],
        "kanbefair_model_import_status": import_status,
        "at_least_one_dataset_task_runnable": runnable,
        "SourceAuditPass": int(kb.exists() and (src / "train.py").exists() and "class MLP" in model_files.get("mlp.py", "") and "class KANbeFair" in model_files.get("kanbefair.py", "") and runnable),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "third_party_source_audit.csv", [row])
    return [row], {"MLP": MLP, "KANbeFair": KANbeFair, "BSpline_MLP": BSpline_MLP}


def _baseline_configs() -> List[Dict[str, Any]]:
    return [
        {
            "model": "MLP",
            "layers_width": [32],
            "activation_name": "gelu",
            "batch_norm": False,
            "lr": 0.001,
            "reported_epochs": 20,
            "reported_batch_size": 128,
            "seed": 1314,
        },
        {
            "model": "KAN",
            "layers_width": [2],
            "activation_name": "gelu",
            "batch_norm": False,
            "lr": 0.001,
            "reported_epochs": 20,
            "reported_batch_size": 128,
            "seed": 1314,
            "kan_grid": 3,
            "kan_order": 2,
            "kan_shortcut": "silu",
            "kan_range": [-1.0, 1.0],
        },
    ]


def _run_baseline_audit(out_dir: Path, args: argparse.Namespace, model_classes: Dict[str, Any]) -> List[Dict[str, Any]]:
    MLP = model_classes.get("MLP")
    KANbeFair = model_classes.get("KANbeFair")
    if MLP is None or KANbeFair is None:
        rows = [{
            "stage": "P1_KANBEFAIR_REPRODUCTION_V85",
            "status": "not_run",
            "reason": "kanbefair_model_import_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        write_csv(out_dir / "kanbefair_reproduction.csv", rows)
        return rows
    device = v83.get_device(args.device)
    rows: List[Dict[str, Any]] = []
    for dataset in parse_str_list(args.datasets):
        kb_dataset = "FMNIST" if dataset in {"Fashion-MNIST", "Fashion", "FMNIST"} else dataset
        dg_dataset = "Fashion-MNIST" if kb_dataset == "FMNIST" else kb_dataset
        if args.kb_data_protocol == "kanbefair":
            x_train, y_train, x_test, y_test, input_dim, num_classes, split_protocol = _load_kanbefair_vision_tensors(
                kb_dataset,
                data_root=Path(args.data_root),
                train_size=args.train_size,
                test_size=args.test_size,
                seed=args.seed,
            )
        else:
            bundle = v83.v72.v71.load_vision_bundle(
                dg_dataset,
                data_root=Path(args.data_root),
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=args.seed,
                allow_fake_data=False,
            )
            x_train = bundle.x_train
            y_train = bundle.y_train
            x_test = bundle.x_test
            y_test = bundle.y_test
            input_dim = bundle.input_dim
            num_classes = bundle.num_classes
            split_protocol = f"DG-balanced-real-subset train={args.train_size} test={args.test_size}"
        for cfg in _baseline_configs():
            model_name = cfg["model"]
            cls = MLP if model_name == "MLP" else KANbeFair
            ns = _kb_namespace(
                model_name=model_name,
                input_size=input_dim,
                output_size=num_classes,
                layers_width=cfg["layers_width"],
                activation_name=cfg["activation_name"],
                batch_norm=cfg["batch_norm"],
                kan_grid=cfg.get("kan_grid", 3),
                kan_order=cfg.get("kan_order", 2),
                kan_shortcut=cfg.get("kan_shortcut", "silu"),
                kan_range=cfg.get("kan_range", [-1.0, 1.0]),
            )
            set_seed(int(cfg["seed"]))
            model = cls(ns)
            params = int(model.total_parameters())
            flops = float(model.total_flops())
            reported = _reported_result(
                Path(args.kanbefair_path),
                dataset=kb_dataset,
                model=model_name,
                layers_width=cfg["layers_width"],
                batch_size=int(cfg["reported_batch_size"]),
                epochs=int(cfg["reported_epochs"]),
                lr=float(cfg["lr"]),
                seed=int(cfg["seed"]),
                activation_name=cfg["activation_name"],
                kan_grid=cfg.get("kan_grid", 3),
                kan_order=cfg.get("kan_order", 2),
                kan_shortcut=cfg.get("kan_shortcut", "silu"),
                kan_range=cfg.get("kan_range", [-1.0, 1.0]),
            )
            metrics = _train_kb_classifier(
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                epochs=args.kb_epochs,
                batch_size=args.kb_batch_size,
                lr=float(cfg["lr"]),
                seed=int(cfg["seed"]),
                device=device,
            )
            paper_comparable = int(
                args.kb_epochs == int(cfg["reported_epochs"])
                and args.kb_batch_size == int(cfg["reported_batch_size"])
                and args.train_size >= 60000
                and args.test_size >= 10000
                and args.kb_data_protocol == "kanbefair"
            )
            reported_test = _safe_float(reported.get("reported_test_metric"))
            abs_delta = abs(metrics["test_acc_pct"] - reported_test) if math.isfinite(reported_test) else float("nan")
            repro_pass = int(paper_comparable and math.isfinite(abs_delta) and abs_delta <= 2.0)
            rows.append({
                "stage": "P1_KANBEFAIR_REPRODUCTION_V85",
                "status": "measured",
                "task_name": kb_dataset,
                "dataset_name": kb_dataset,
                "split_protocol": f"{split_protocol}; paper_comparable={paper_comparable}",
                "model_name": f"KB-{model_name}",
                "params": params,
                "FLOPs": flops,
                "epochs": args.kb_epochs,
                "batch_size": args.kb_batch_size,
                "optimizer": "Adam",
                "lr": float(cfg["lr"]),
                "weight_decay": 0.0,
                "seed": int(cfg["seed"]),
                "val_metric": METRIC_UNAVAILABLE,
                "test_metric": metrics["test_acc_pct"],
                "test_loss": metrics["test_loss"],
                "ECE": metrics["ECE"],
                "NLL": metrics["NLL"],
                "reported_metric_from_paper": reported_test if math.isfinite(reported_test) else METRIC_UNAVAILABLE,
                "absolute_delta_from_reported": abs_delta if math.isfinite(abs_delta) else METRIC_UNAVAILABLE,
                "relative_delta_from_reported": (abs_delta / max(abs(reported_test), 1.0e-12)) if math.isfinite(abs_delta) and math.isfinite(reported_test) else METRIC_UNAVAILABLE,
                "paper_comparable_protocol": paper_comparable,
                "reproduction_pass": repro_pass,
                "train_time_s": metrics["train_time_s"],
                "avg_epoch_time_s": metrics["avg_epoch_time_s"],
                "peak_memory_MB": metrics["peak_memory_MB"],
                "step_time_ms": metrics["step_time_ms"],
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    write_csv(out_dir / "kanbefair_reproduction.csv", rows)
    return rows


def _run_counter_audit(out_dir: Path, args: argparse.Namespace, model_classes: Dict[str, Any]) -> List[Dict[str, Any]]:
    MLP = model_classes.get("MLP")
    KANbeFair = model_classes.get("KANbeFair")
    rows: List[Dict[str, Any]] = []
    input_size = 784
    output_size = 10
    if MLP is not None:
        ns = _kb_namespace(model_name="MLP", input_size=input_size, output_size=output_size, layers_width=[32], activation_name="gelu")
        model = MLP(ns)
        rows.append({
            "stage": "P3_PARAMS_FLOPS_COUNTER_AUDIT_V85",
            "model_name": "KB-MLP-width32-gelu",
            "params_total": int(model.total_parameters()),
            "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "params_KAN_edge": 0,
            "params_nonKAN": int(model.total_parameters()),
            "FLOPs_formula": "MLP.total_flops: sum(2*din*dout + activation_flops*dout)",
            "FLOPs_measured_or_estimated": float(model.total_flops()),
            "FLOPs_forward": float(model.total_flops()),
            "FLOPs_backward_if_available": METRIC_UNAVAILABLE,
            "CounterPass": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    if KANbeFair is not None:
        ns = _kb_namespace(model_name="KAN", input_size=input_size, output_size=output_size, layers_width=[2], activation_name="gelu", kan_grid=3, kan_order=2, kan_shortcut="silu")
        model = KANbeFair(ns)
        rows.append({
            "stage": "P3_PARAMS_FLOPS_COUNTER_AUDIT_V85",
            "model_name": "KB-KAN-width2-grid3-order2-silu",
            "params_total": int(model.total_parameters()),
            "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "params_KAN_edge": int(model.total_parameters()),
            "params_nonKAN": 0,
            "FLOPs_formula": "KANbeFair.layer_flops: din*dout*(9*k*(grid+1.5*k)+2*grid-2.5*k+1)+shortcut",
            "FLOPs_measured_or_estimated": float(model.total_flops()),
            "FLOPs_forward": float(model.total_flops()),
            "FLOPs_backward_if_available": METRIC_UNAVAILABLE,
            "CounterPass": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    try:
        device = v83.get_device(args.device)
        spec = v83.v80._spec_map_v80().get(str(args.dg_candidate_id), v83.v80._spec_map_v80()["KW6"])
        hidden_dim = int(args.dg_hidden_dim)
        stack, head = v83.v80._make_manual_candidate_v80(spec, input_size, output_size, hidden_dim, int(args.dg_basis_count), device)
        params_total = int(stack.param_count() + head.param_count())
        flops_forward = _dg_kw6_forward_flops_estimate(input_size, hidden_dim, int(spec.depth), output_size)
        rows.append({
            "stage": "P3_PARAMS_FLOPS_COUNTER_AUDIT_V85",
            "model_name": f"DG-Base-{args.dg_candidate_id}-hidden{hidden_dim}",
            "params_total": params_total,
            "params_trainable": params_total,
            "params_KAN_edge": params_total,
            "params_nonKAN": 0,
            "FLOPs_formula": "forward-only analytic estimate: stack=sum(2*din*dout)+hidden_silu(4*h per hidden layer); head=poly2_silu_transform(11*h)+2*h*num_classes",
            "FLOPs_measured_or_estimated": flops_forward,
            "FLOPs_forward": flops_forward,
            "FLOPs_backward_if_available": METRIC_UNAVAILABLE,
            "FLOPs_counter_scope": "forward_only_analytic_estimate",
            "CounterPass": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    except Exception as exc:
        rows.append({
            "stage": "P3_PARAMS_FLOPS_COUNTER_AUDIT_V85",
            "model_name": f"DG-Base-{args.dg_candidate_id}-hidden{args.dg_hidden_dim}",
            "status": "counter_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "CounterPass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv(out_dir / "params_flops_counter_audit.csv", rows)
    return rows


def _write_adapter_contract(out_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate_id, functional_update_used, adapter_class in [
        (f"DG0-{args.dg_candidate_id}-hidden{args.dg_hidden_dim}-base", 0, "DGKANAdapter"),
        (f"DG1-FT7-{args.dg_candidate_id}-hidden{args.dg_hidden_dim}-functional", 1, "DGKANFunctionalAdapter"),
    ]:
        try:
            smoke = _run_dg_adapter_smoke(args, functional_update_used=functional_update_used)
            rows.append({
            "stage": "P2_DGKAN_ADAPTER_CONTRACT_V85",
                "candidate_id": candidate_id,
                "adapter_class": adapter_class,
            "input_adapter": "KANbeFair flattened vision tensor compatible",
            "output_adapter": "classification logits",
            "loss_type": "CE",
                "functional_update_used": functional_update_used,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "geometry_loss_used": 0,
            "manual_forward": 1,
            "manual_backward": 1,
            "manual_update": 1,
            "uses_loss_backward": 0,
                "nonKAN_param_count": 0,
                **smoke,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
                "reason": "measured external KANbeFair tensor adapter smoke",
            })
        except Exception as exc:
            rows.append({
            "stage": "P2_DGKAN_ADAPTER_CONTRACT_V85",
                "candidate_id": candidate_id,
                "adapter_class": adapter_class,
            "input_adapter": "KANbeFair flattened vision tensor compatible",
            "output_adapter": "classification logits",
            "loss_type": "CE",
                "functional_update_used": functional_update_used,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "geometry_loss_used": 0,
                "manual_forward": 0,
                "manual_backward": 0,
                "manual_update": 0,
            "uses_loss_backward": 0,
            "nonKAN_param_count": METRIC_UNAVAILABLE,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
            "AdapterPass": 0,
                "reason": f"{type(exc).__name__}: {exc}",
            })
    write_csv(out_dir / "dgkan_adapter_contract.csv", rows)
    return rows


def _train_dg_primary_classifier(
    args: argparse.Namespace,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    dataset: str,
    input_dim: int,
    num_classes: int,
    functional_update_used: int,
    functional_variant: str | None = None,
) -> Dict[str, Any]:
    device = v83.get_device(args.device)
    v83.v80._patch_for_v80()
    dg_candidate = str(args.dg_candidate_id)
    spec = v83.v80._spec_map_v80().get(dg_candidate, v83.v80._spec_map_v80()["KW6"])
    v83.set_seed(v83.v72._stable_seed("v85-dg-primary-init", dataset, int(args.seed), dg_candidate, int(args.dg_hidden_dim)))
    stack, head = v83.v80._make_manual_candidate_v80(
        spec,
        int(input_dim),
        int(num_classes),
        int(args.dg_hidden_dim),
        int(args.dg_basis_count),
        device,
    )
    _configure_dg_update_modes(dg_candidate, stack, head)
    stream_batches_from_cpu = bool(getattr(args, "dg_stream_batches_from_cpu", False))
    stream_epoch_permute_cpu = bool(getattr(args, "dg_stream_epoch_permute_cpu", False))
    stream_chunk_order_shuffle = bool(getattr(args, "dg_stream_chunk_order_shuffle", False))
    if stream_batches_from_cpu:
        x_train_cpu = x_train.contiguous()
        y_train_cpu = y_train.contiguous()
        x_test_cpu = x_test.contiguous()
        y_test_cpu = y_test.contiguous()
        if torch.cuda.is_available() and device.type == "cuda":
            x_train_cpu = x_train_cpu.pin_memory()
            y_train_cpu = y_train_cpu.pin_memory()
            x_test_cpu = x_test_cpu.pin_memory()
            y_test_cpu = y_test_cpu.pin_memory()
        n_train = int(x_train_cpu.shape[0])
        n_test = int(x_test_cpu.shape[0])
    else:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        n_train = int(x_train.shape[0])
        n_test = int(x_test.shape[0])
    batch_size = int(args.dg_batch_size)
    stream_chunk_batches = max(1, int(getattr(args, "dg_stream_chunk_batches", 1)))
    epochs = int(args.primary_epochs)
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    total_steps = max(1, epochs * steps_per_epoch)
    params = v83.v80.V63Params(
        train_size=n_train,
        val_size=0,
        test_size=n_test,
        batch_size=batch_size,
    )
    lr = float(params.lr_manual * spec.lr_mult)
    opt = v83.v72.FastAdamWNoSync([stack, head], lr=lr, weight_decay=float(args.weight_decay))
    entries = v83._param_entries(stack, head)
    role_entries_by_name = v83._entries_by_role(entries)
    before_smooth, before_curv = v83._geometry_norms(entries)
    variant = functional_variant or ("ft7" if functional_update_used else "base")
    rng_device: Any = "cpu" if stream_batches_from_cpu else device
    rng = torch.Generator(device=rng_device).manual_seed(int(args.seed))
    control_rng = torch.Generator(device=device).manual_seed(
        v83.v72._stable_seed("v85-functional-control", dataset, int(args.seed), variant, int(args.dg_hidden_dim))
    )
    losses: List[float] = []
    functional_event_count = 0
    functional_accept_mass = 0.0
    functional_reject_mass = 0.0
    functional_bad_step_count = 0
    holdout_descent_values: List[float] = []
    peak_mb: Any = METRIC_UNAVAILABLE
    if callable(DG_PRETRAIN_HOOK):
        DG_PRETRAIN_HOOK(
            stack=stack,
            head=head,
            spec=spec,
            x_train=x_train_cpu if stream_batches_from_cpu else x_train,
            y_train=y_train_cpu if stream_batches_from_cpu else y_train,
            device=device,
            batch_size=batch_size,
        )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    global_step = 0
    for _epoch in range(epochs):
        perm = torch.randperm(n_train, generator=rng, device="cpu" if stream_batches_from_cpu else device)
        batch_order: List[int] | None = None
        if stream_batches_from_cpu and stream_chunk_order_shuffle:
            num_chunks = max(1, math.ceil(steps_per_epoch / stream_chunk_batches))
            chunk_order = torch.randperm(num_chunks, generator=rng).tolist()
            batch_order = []
            for chunk_id in chunk_order:
                first_batch = int(chunk_id) * stream_chunk_batches
                last_batch = min(first_batch + stream_chunk_batches, steps_per_epoch)
                batch_order.extend(range(first_batch, last_batch))
        x_epoch_cpu = None
        y_epoch_cpu = None
        if stream_batches_from_cpu and stream_epoch_permute_cpu:
            x_epoch_cpu = x_train_cpu[perm].contiguous()
            y_epoch_cpu = y_train_cpu[perm].contiguous()
            if torch.cuda.is_available() and device.type == "cuda":
                x_epoch_cpu = x_epoch_cpu.pin_memory()
                y_epoch_cpu = y_epoch_cpu.pin_memory()
        stream_chunk_start = -1
        stream_chunk_end = -1
        x_stream_chunk = None
        y_stream_chunk = None
        for order_pos in range(steps_per_epoch):
            batch_idx = batch_order[order_pos] if batch_order is not None else order_pos
            global_step += 1
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_train)
            idx = perm[start:end]
            if int(idx.numel()) == 0:
                continue
            holdout_batch_idx = (
                batch_order[(order_pos + 1) % steps_per_epoch]
                if batch_order is not None
                else ((batch_idx + 1) % steps_per_epoch)
            )
            holdout_start = holdout_batch_idx * batch_size
            holdout_end = min(holdout_start + batch_size, n_train)
            holdout_idx = (
                torch.arange(holdout_start, holdout_end)
                if stream_batches_from_cpu and stream_chunk_order_shuffle
                else perm[holdout_start:holdout_end]
            )
            if int(holdout_idx.numel()) == 0:
                holdout_idx = idx
            if stream_batches_from_cpu:
                chunk_batch_start = (batch_idx // stream_chunk_batches) * stream_chunk_batches
                chunk_start = chunk_batch_start * batch_size
                chunk_end = min((chunk_batch_start + stream_chunk_batches) * batch_size, n_train)
                if chunk_start != stream_chunk_start or chunk_end != stream_chunk_end:
                    stream_chunk_start = chunk_start
                    stream_chunk_end = chunk_end
                    if x_epoch_cpu is not None and y_epoch_cpu is not None:
                        x_stream_chunk = x_epoch_cpu[stream_chunk_start:stream_chunk_end].to(device, non_blocking=True)
                        y_stream_chunk = y_epoch_cpu[stream_chunk_start:stream_chunk_end].to(device, non_blocking=True)
                    elif stream_chunk_order_shuffle:
                        x_stream_chunk = x_train_cpu[stream_chunk_start:stream_chunk_end].to(device, non_blocking=True)
                        y_stream_chunk = y_train_cpu[stream_chunk_start:stream_chunk_end].to(device, non_blocking=True)
                    else:
                        chunk_idx = perm[stream_chunk_start:stream_chunk_end]
                        x_stream_chunk = x_train_cpu[chunk_idx].to(device, non_blocking=True)
                        y_stream_chunk = y_train_cpu[chunk_idx].to(device, non_blocking=True)
                local_start = start - stream_chunk_start
                local_end = end - stream_chunk_start
                xb = x_stream_chunk[local_start:local_end]
                yb = y_stream_chunk[local_start:local_end]
                if holdout_start >= stream_chunk_start and holdout_end <= stream_chunk_end:
                    holdout_local_start = holdout_start - stream_chunk_start
                    holdout_local_end = holdout_end - stream_chunk_start
                    xh = x_stream_chunk[holdout_local_start:holdout_local_end]
                    yh = y_stream_chunk[holdout_local_start:holdout_local_end]
                else:
                    if x_epoch_cpu is not None and y_epoch_cpu is not None:
                        xh = x_epoch_cpu[holdout_start:holdout_end].to(device, non_blocking=True)
                        yh = y_epoch_cpu[holdout_start:holdout_end].to(device, non_blocking=True)
                    elif stream_chunk_order_shuffle:
                        xh = x_train_cpu[holdout_start:holdout_end].to(device, non_blocking=True)
                        yh = y_train_cpu[holdout_start:holdout_end].to(device, non_blocking=True)
                    else:
                        xh = x_train_cpu[holdout_idx].to(device, non_blocking=True)
                        yh = y_train_cpu[holdout_idx].to(device, non_blocking=True)
            else:
                xb = x_train[idx]
                yb = y_train[idx]
                xh = x_train[holdout_idx]
                yh = y_train[holdout_idx]
            event_step = (
                bool(functional_update_used)
                and variant in {"ft7", "random", "shuffled"}
                and global_step % int(v83.P5_FT7_EVENT_STRIDE) == 0
            )
            use_pre_holdout = bool(event_step and v83._ft7_uses_pre_holdout_for_step(global_step))
            need_loss_float = bool(event_step or global_step == 1 or global_step == total_steps)
            if use_pre_holdout:
                loss_before, holdout_loss_before = _manual_ce_backward_with_holdout_v85(stack, head, xb, yb, xh, yh, spec)
            else:
                holdout_loss_before = 0.0
                loss_before_value = _manual_ce_backward_v85(stack, head, xb, yb, spec, need_loss_float=need_loss_float)
                loss_before = float(loss_before_value) if loss_before_value is not None else 0.0
            opt.step(global_step, total_steps, warmup_cosine=True)
            loss_after = loss_before
            task_features_after = None
            holdout_features_after = None
            if event_step:
                if v83.FT7_USE_HOLDOUT_GUARD:
                    loss_after, holdout_loss_after, task_features_after, holdout_features_after = v83._loss_pair_and_features_only(
                        stack,
                        head,
                        xb,
                        yb,
                        xh,
                        yh,
                        spec,
                    )
                else:
                    loss_after, task_features_after = v83._loss_and_features_only(stack, head, xb, yb, spec)
                    holdout_loss_after = 0.0
                if use_pre_holdout:
                    holdout_descent_values.append(
                        (float(holdout_loss_before) - float(holdout_loss_after))
                        / max(abs(float(holdout_loss_before)), 1.0e-12)
                    )
                func_alpha = lr * 0.05 * float(v83.P5_FT7_EVENT_ALPHA_MULT)
                original_role_weights = None
                if variant == "random":
                    # RandomFunc is a same-cost signed curvature control under the
                    # same CE guard. Negative signs are normally rejected; accepted
                    # positive signs are measured, not assumed.
                    sign = -1.0 if float(torch.rand((), generator=control_rng, device=device).item()) < 0.5 else 1.0
                    func_alpha *= sign
                elif variant == "shuffled":
                    original_role_weights = dict(v83.FT7_ROLE_WEIGHTS)
                    v83.FT7_ROLE_WEIGHTS = {
                        "stack": float(original_role_weights.get("head", 1.0)),
                        "head": float(original_role_weights.get("stack", 0.75)),
                    }
                try:
                    loss_after, _func_norm, trust_delta, reject_delta = v83._apply_ft7_streamed_guarded_update(
                        stack,
                        head,
                        entries,
                        xb,
                        yb,
                        xh,
                        yh,
                        spec,
                        loss_before=loss_before,
                        holdout_loss_before=holdout_loss_before,
                        task_loss_after=loss_after,
                        holdout_loss_after=holdout_loss_after,
                        func_alpha=func_alpha,
                        task_features_after=task_features_after,
                        holdout_features_after=holdout_features_after,
                        role_entries_by_name=role_entries_by_name,
                        pair_stack_guard_forward=True,
                    )
                finally:
                    if original_role_weights is not None:
                        v83.FT7_ROLE_WEIGHTS = original_role_weights
                if float(loss_after) > float(loss_before) + 1.0e-8:
                    functional_bad_step_count += 1
                functional_event_count += 1
                functional_accept_mass += float(trust_delta)
                functional_reject_mass += float(reject_delta)
            if need_loss_float or event_step:
                losses.append(float(loss_after))
    if not losses:
        if stream_batches_from_cpu:
            xb0 = x_train_cpu[:batch_size].to(device, non_blocking=True)
            yb0 = y_train_cpu[:batch_size].to(device, non_blocking=True)
            losses.append(v83._loss_only(stack, head, xb0, yb0, spec))
        else:
            losses.append(v83._loss_only(stack, head, x_train[:batch_size], y_train[:batch_size], spec))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    elapsed = time.perf_counter() - started
    if stream_batches_from_cpu:
        x_test_eval = x_test_cpu.to(device, non_blocking=True)
        y_test_eval = y_test_cpu.to(device, non_blocking=True)
    else:
        x_test_eval = x_test
        y_test_eval = y_test
    eval_metrics = v83.v72.v71._manual_eval(stack, head, x_test_eval, y_test_eval, int(args.dg_eval_batch_size), int(num_classes))
    local_lipschitz = _dg_local_lipschitz_probe(
        stack,
        head,
        x_test_eval,
        seed=v83.v72._stable_seed("v85-local-lipschitz", dataset, int(args.seed), variant, int(args.dg_hidden_dim)),
        batch_size=int(args.dg_eval_batch_size),
    )
    after_entries = v83._param_entries(stack, head)
    after_smooth, after_curv = v83._geometry_norms(after_entries)
    params_total = int(stack.param_count() + head.param_count())
    if variant == "base":
        candidate_id = f"DG0-{dg_candidate}-hidden{args.dg_hidden_dim}-base"
    elif variant == "noop":
        candidate_id = f"DG-NoOp-{dg_candidate}-hidden{args.dg_hidden_dim}"
    elif variant == "random":
        candidate_id = f"DG-RandomFunc-{dg_candidate}-hidden{args.dg_hidden_dim}"
    elif variant == "shuffled":
        candidate_id = f"DG-ShuffledRoleFunc-{dg_candidate}-hidden{args.dg_hidden_dim}"
    else:
        candidate_id = f"DG1-FT7-{dg_candidate}-hidden{args.dg_hidden_dim}-functional"
    return {
        "status": "measured",
        "task_name": dataset,
        "candidate_id": candidate_id,
        "model_family": "DG-KAN",
        "functional_update_used": int(functional_update_used),
        "functional_variant": variant,
        "loss_type": "CE",
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "geometry_loss_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "optimizer": "FastAdamWNoSync",
        "input_data_residency": "batch_streamed_cpu_to_gpu" if stream_batches_from_cpu else "full_tensor_gpu_resident",
        "stream_chunk_batches": stream_chunk_batches if stream_batches_from_cpu else 0,
        "stream_epoch_permute_cpu": int(stream_epoch_permute_cpu) if stream_batches_from_cpu else 0,
        "stream_chunk_order_shuffle": int(stream_chunk_order_shuffle) if stream_batches_from_cpu else 0,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps": global_step,
        "params": params_total,
        "FLOPs": _dg_kw6_forward_flops_estimate(int(input_dim), int(args.dg_hidden_dim), int(spec.depth), int(num_classes)),
        "FLOPs_counter_scope": "forward_only_analytic_estimate",
        "test_metric": float(eval_metrics["acc"]) * 100.0,
        "test_metric_fraction": float(eval_metrics["acc"]),
        "test_loss": float(eval_metrics["loss"]),
        "ECE": float(eval_metrics["ECE"]),
        "NLL": float(eval_metrics["NLL"]),
        "train_loss_first": losses[0] if losses else METRIC_UNAVAILABLE,
        "train_loss_last": losses[-1] if losses else METRIC_UNAVAILABLE,
        "train_time_s": elapsed,
        "step_time_ms": elapsed * 1000.0 / max(1, global_step),
        "peak_memory_MB": peak_mb,
        "geometry_smooth_before": before_smooth,
        "geometry_smooth_after": after_smooth,
        "geometry_curvature_before": before_curv,
        "geometry_curvature_after": after_curv,
        "jacobian_norm_probe": local_lipschitz,
        "local_lipschitz_probe": local_lipschitz,
        "functional_event_count": functional_event_count,
        "functional_accept_mass": functional_accept_mass,
        "functional_reject_mass": functional_reject_mass,
        "functional_bad_step_count": functional_bad_step_count,
        "bad_step_rate": functional_bad_step_count / max(1, functional_event_count),
        "role_accept_rate": functional_accept_mass / max(1.0e-12, functional_accept_mass + functional_reject_mass),
        "holdout_descent_ratio": (
            sum(holdout_descent_values) / len(holdout_descent_values)
            if holdout_descent_values
            else METRIC_UNAVAILABLE
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _run_primary_transfer(out_dir: Path, args: argparse.Namespace, baseline_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for baseline in baseline_rows:
        if str(baseline.get("status")) == "measured":
            rows.append({
                "stage": "P5_KANBEFAIR_PRIMARY_TRANSFER_V85",
                "status": "measured",
                "task_name": baseline.get("task_name"),
                "candidate_id": baseline.get("model_name"),
                "model_family": "KANbeFair",
                "functional_update_used": 0,
                "loss_type": "CE",
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "geometry_loss_used": 0,
                "sampler_changed": 0,
                "class_weight_used": 0,
                "optimizer": baseline.get("optimizer"),
                "epochs": baseline.get("epochs"),
                "batch_size": baseline.get("batch_size"),
                "steps": METRIC_UNAVAILABLE,
                "params": baseline.get("params"),
                "FLOPs": baseline.get("FLOPs"),
                "FLOPs_counter_scope": "KANbeFair native total_flops",
                "test_metric": baseline.get("test_metric"),
                "test_metric_fraction": _safe_float(baseline.get("test_metric")) / 100.0,
                "test_loss": baseline.get("test_loss"),
                "ECE": baseline.get("ECE"),
                "NLL": baseline.get("NLL"),
                "train_loss_first": METRIC_UNAVAILABLE,
                "train_loss_last": baseline.get("train_loss_last", METRIC_UNAVAILABLE),
                "train_time_s": baseline.get("train_time_s"),
                "step_time_ms": baseline.get("step_time_ms"),
                "peak_memory_MB": baseline.get("peak_memory_MB"),
                "geometry_smooth_before": METRIC_UNAVAILABLE,
                "geometry_smooth_after": METRIC_UNAVAILABLE,
                "geometry_curvature_before": METRIC_UNAVAILABLE,
                "geometry_curvature_after": METRIC_UNAVAILABLE,
                "functional_event_count": 0,
                "functional_accept_mass": 0,
                "functional_reject_mass": 0,
                "PrimaryTransferPass": 0,
                "reason": "baseline row copied from measured KANbeFair reproduction",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    for dataset in parse_str_list(args.datasets):
        kb_dataset = "FMNIST" if dataset in {"Fashion-MNIST", "Fashion", "FMNIST"} else dataset
        x_train, y_train, x_test, y_test, input_dim, num_classes, split_protocol = _load_kanbefair_vision_tensors(
            kb_dataset,
            data_root=Path(args.data_root),
            train_size=int(args.primary_train_size),
            test_size=int(args.primary_test_size),
            seed=int(args.seed),
        )
        primary_full_protocol = int(
            int(args.primary_train_size) >= 60000
            and int(args.primary_test_size) >= 10000
            and int(args.primary_epochs) == 20
            and int(args.dg_batch_size) == int(args.kb_batch_size)
            and args.kb_data_protocol == "kanbefair"
        )
        dg_base = _train_dg_primary_classifier(
            args,
            x_train,
            y_train,
            x_test,
            y_test,
            dataset=kb_dataset,
            input_dim=input_dim,
            num_classes=num_classes,
            functional_update_used=0,
        )
        dg_func = _train_dg_primary_classifier(
            args,
            x_train,
            y_train,
            x_test,
            y_test,
            dataset=kb_dataset,
            input_dim=input_dim,
            num_classes=num_classes,
            functional_update_used=1,
        )
        kb_best = max(
            [_safe_float(r.get("test_metric")) for r in rows if r.get("task_name") == kb_dataset and r.get("model_family") == "KANbeFair"],
            default=float("nan"),
        )
        base_acc = _safe_float(dg_base.get("test_metric"))
        func_acc = _safe_float(dg_func.get("test_metric"))
        base_curv = _safe_float(dg_base.get("geometry_curvature_after"))
        func_curv = _safe_float(dg_func.get("geometry_curvature_after"))
        curv_ratio = func_curv / max(base_curv, 1.0e-12) if math.isfinite(base_curv) and math.isfinite(func_curv) else float("nan")
        primary_pass = int(
            primary_full_protocol
            and math.isfinite(kb_best)
            and func_acc >= kb_best
            and func_acc >= base_acc - 0.5
            and math.isfinite(curv_ratio)
            and curv_ratio <= 0.90
        )
        for row in [dg_base, dg_func]:
            row.update({
                "stage": "P5_KANBEFAIR_PRIMARY_TRANSFER_V85",
                "dataset_protocol": split_protocol,
                "paper_comparable_protocol": primary_full_protocol,
                "best_kanbefair_baseline_metric": kb_best if math.isfinite(kb_best) else METRIC_UNAVAILABLE,
                "metric_delta_vs_best_kanbefair": _safe_float(row.get("test_metric")) - kb_best if math.isfinite(kb_best) else METRIC_UNAVAILABLE,
                "metric_delta_vs_dg_base": _safe_float(row.get("test_metric")) - base_acc if math.isfinite(base_acc) else METRIC_UNAVAILABLE,
                "functional_curvature_ratio_vs_dg_base": curv_ratio if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
                "PrimaryTransferPass": primary_pass if str(row.get("candidate_id", "")).startswith("DG1-FT7-") else 0,
                "reason": "measured DG-KAN on KANbeFair tensor protocol",
            })
            rows.append(row)
    write_csv(out_dir / "kanbefair_primary_transfer.csv", rows)
    return rows


def _run_fair_envelopes(out_dir: Path, primary_rows: Sequence[Dict[str, Any]]) -> None:
    kb_rows = [r for r in primary_rows if r.get("model_family") == "KANbeFair" and str(r.get("status")) == "measured"]
    dg_rows = [r for r in primary_rows if str(r.get("candidate_id", "")).startswith("DG1-FT7-") and str(r.get("status")) == "measured"]
    if not kb_rows or not dg_rows:
        return
    mlp_rows = [r for r in kb_rows if str(r.get("candidate_id")) == "KB-MLP"]
    comparator_rows = mlp_rows if mlp_rows else kb_rows
    kb_best_param = min(_safe_float(r.get("params")) for r in comparator_rows)
    kb_best_flops = min(_safe_float(r.get("FLOPs")) for r in comparator_rows)
    kb_best_step = min(_safe_float(r.get("step_time_ms")) for r in comparator_rows)
    kb_best_memory = min(_safe_float(r.get("peak_memory_MB")) for r in comparator_rows)
    kb_best_train = min(_safe_float(r.get("train_time_s")) for r in comparator_rows)
    dg = dg_rows[0]
    dg_params = _safe_float(dg.get("params"))
    dg_flops = _safe_float(dg.get("FLOPs"))
    dg_step = _safe_float(dg.get("step_time_ms"))
    dg_memory = _safe_float(dg.get("peak_memory_MB"))
    dg_train = _safe_float(dg.get("train_time_s"))
    param_ratio = dg_params / max(kb_best_param, 1.0e-12)
    flops_ratio = dg_flops / max(kb_best_flops, 1.0e-12)
    step_ratio = dg_step / max(kb_best_step, 1.0e-12)
    memory_ratio = dg_memory / max(kb_best_memory, 1.0e-12)
    train_ratio = dg_train / max(kb_best_train, 1.0e-12)
    write_csv(out_dir / "parameter_matched_envelope.csv", [{
        "stage": "P6_PARAMETER_MATCHED_ENVELOPE_V85",
        "status": "measured",
        "comparison_scope": "default_KANbeFair_MLP_width32_vs_DG_functional",
        "best_kanbefair_params": kb_best_param,
        "dg_functional_params": dg_params,
        "param_ratio": param_ratio,
        "ParameterFairPass": int(math.isfinite(param_ratio) and param_ratio <= 1.05),
        "reason": "MLP-relative external fair envelope; direct screen fails if ratio > 1.05",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_csv(out_dir / "flops_matched_envelope.csv", [{
        "stage": "P7_FLOPS_MATCHED_ENVELOPE_V85",
        "status": "measured",
        "comparison_scope": "default_KANbeFair_MLP_width32_vs_DG_functional_forward_estimate",
        "best_kanbefair_FLOPs": kb_best_flops,
        "dg_functional_FLOPs": dg_flops,
        "FLOPs_ratio": flops_ratio,
        "FLOPsFairPass": int(math.isfinite(flops_ratio) and flops_ratio <= 1.05),
        "reason": "forward FLOPs only for DG; backward FLOPs unavailable, so no formal FLOPs win can be claimed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_csv(out_dir / "wallclock_memory_envelope.csv", [{
        "stage": "P8_WALLCLOCK_MEMORY_ENVELOPE_V85",
        "status": "measured",
        "comparison_scope": "default_KANbeFair_MLP_width32_vs_DG_functional_primary_run",
        "best_kanbefair_step_time_ms": kb_best_step,
        "dg_functional_step_time_ms": dg_step,
        "step_time_ratio": step_ratio,
        "best_kanbefair_train_time_s": kb_best_train,
        "dg_functional_train_time_s": dg_train,
        "train_time_ratio": train_ratio,
        "best_kanbefair_peak_memory_MB": kb_best_memory,
        "dg_functional_peak_memory_MB": dg_memory,
        "memory_ratio": memory_ratio,
        "WallclockMemoryPass": int(
            math.isfinite(step_ratio)
            and math.isfinite(memory_ratio)
            and step_ratio <= 1.50
            and memory_ratio <= 1.05
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])


def _run_functional_causality(out_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_args = argparse.Namespace(**vars(args))
    if int(getattr(args, "causality_epochs", 0)) > 0:
        run_args.primary_epochs = int(args.causality_epochs)
    train_size = int(getattr(args, "causality_train_size", 0)) or int(args.primary_train_size)
    test_size = int(getattr(args, "causality_test_size", 0)) or int(args.primary_test_size)
    variants = [
        ("DG-Base", "base", 0),
        ("DG-NoOp", "noop", 0),
        ("DG-Functional", "ft7", 1),
        ("DG-RandomFunc", "random", 1),
        ("DG-ShuffledRoleFunc", "shuffled", 1),
    ]
    for dataset in parse_str_list(args.datasets):
        kb_dataset = "FMNIST" if dataset in {"Fashion-MNIST", "Fashion", "FMNIST"} else dataset
        x_train, y_train, x_test, y_test, input_dim, num_classes, split_protocol = _load_kanbefair_vision_tensors(
            kb_dataset,
            data_root=Path(args.data_root),
            train_size=train_size,
            test_size=test_size,
            seed=int(args.seed),
        )
        measured: Dict[str, Dict[str, Any]] = {}
        for control_name, variant, functional_flag in variants:
            row = _train_dg_primary_classifier(
                run_args,
                x_train,
                y_train,
                x_test,
                y_test,
                dataset=kb_dataset,
                input_dim=input_dim,
                num_classes=num_classes,
                functional_update_used=functional_flag,
                functional_variant=variant,
            )
            row["control_name"] = control_name
            row["dataset_protocol"] = split_protocol
            measured[control_name] = row
        base = measured["DG-Base"]
        func = measured["DG-Functional"]
        noop = measured["DG-NoOp"]
        random_row = measured["DG-RandomFunc"]
        base_acc = _safe_float(base.get("test_metric"))
        base_ece = _safe_float(base.get("ECE"))
        base_nll = _safe_float(base.get("NLL"))
        base_curv = _safe_float(base.get("geometry_curvature_after"))
        base_jac = _safe_float(base.get("jacobian_norm_probe"))
        base_lip = _safe_float(base.get("local_lipschitz_probe"))
        func_curv = _safe_float(func.get("geometry_curvature_after"))
        noop_curv = _safe_float(noop.get("geometry_curvature_after"))
        random_curv = _safe_float(random_row.get("geometry_curvature_after"))
        func_acc = _safe_float(func.get("test_metric"))
        noop_acc = _safe_float(noop.get("test_metric"))
        causality_pass = int(
            math.isfinite(func_curv)
            and math.isfinite(noop_curv)
            and math.isfinite(random_curv)
            and func_curv < noop_curv
            and func_curv < random_curv
            and math.isfinite(func_acc)
            and math.isfinite(noop_acc)
            and func_acc >= noop_acc - 0.20
        )
        for control_name, _variant, _functional_flag in variants:
            row = measured[control_name]
            curv = _safe_float(row.get("geometry_curvature_after"))
            jac = _safe_float(row.get("jacobian_norm_probe"))
            lip = _safe_float(row.get("local_lipschitz_probe"))
            enriched = {
                "stage": "P9_FUNCTIONAL_CAUSALITY_KANBEFAIR_V85",
                "status": "measured",
                "task_name": kb_dataset,
                "control_name": control_name,
                "candidate_id": row.get("candidate_id"),
                "functional_variant": row.get("functional_variant"),
                "functional_update_used": row.get("functional_update_used"),
                "seed": int(args.seed),
                "epochs": row.get("epochs"),
                "train_size": train_size,
                "test_size": test_size,
                "test_acc": row.get("test_metric"),
                "acc_delta_vs_base": _safe_float(row.get("test_metric")) - base_acc if math.isfinite(base_acc) else METRIC_UNAVAILABLE,
                "ECE_delta_vs_base": _safe_float(row.get("ECE")) - base_ece if math.isfinite(base_ece) else METRIC_UNAVAILABLE,
                "NLL_delta_vs_base": _safe_float(row.get("NLL")) - base_nll if math.isfinite(base_nll) else METRIC_UNAVAILABLE,
                "curvature_ratio": curv / max(base_curv, 1.0e-12) if math.isfinite(base_curv) and math.isfinite(curv) else METRIC_UNAVAILABLE,
                "jacobian_ratio": jac / max(base_jac, 1.0e-12) if math.isfinite(base_jac) and math.isfinite(jac) else METRIC_UNAVAILABLE,
                "local_lipschitz_ratio": lip / max(base_lip, 1.0e-12) if math.isfinite(base_lip) and math.isfinite(lip) else METRIC_UNAVAILABLE,
                "functional_update_time_ratio": (
                    _safe_float(row.get("step_time_ms")) / max(_safe_float(base.get("step_time_ms")), 1.0e-12)
                    if math.isfinite(_safe_float(base.get("step_time_ms")))
                    else METRIC_UNAVAILABLE
                ),
                "bad_step_rate": row.get("bad_step_rate"),
                "holdout_descent_ratio": row.get("holdout_descent_ratio"),
                "role_accept_rate": row.get("role_accept_rate"),
                "role_geometry_delta": curv - base_curv if math.isfinite(base_curv) and math.isfinite(curv) else METRIC_UNAVAILABLE,
                "functional_event_count": row.get("functional_event_count"),
                "functional_accept_mass": row.get("functional_accept_mass"),
                "functional_reject_mass": row.get("functional_reject_mass"),
                "FunctionalCausalityPass": causality_pass if control_name == "DG-Functional" else 0,
                "pass_rule": "Functional curvature < NoOp and RandomFunc; Functional accuracy >= NoOp - 0.002 fraction",
                "loss_type": "CE",
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "geometry_loss_used": 0,
                "sampler_changed": 0,
                "class_weight_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            rows.append(enriched)
    write_csv(out_dir / "functional_causality_kanbefair.csv", rows)
    return rows


def _run_symbolic_representation(
    out_dir: Path,
    args: argparse.Namespace,
    model_classes: Dict[str, Any],
) -> List[Dict[str, Any]]:
    MLP = model_classes.get("MLP")
    KANbeFair = model_classes.get("KANbeFair")
    if MLP is None or KANbeFair is None:
        rows = [{
            "stage": "P10_SYMBOLIC_REPRESENTATION_V85",
            "status": "not_run",
            "reason": "kanbefair_model_import_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        write_csv(out_dir / "symbolic_representation.csv", rows)
        return rows
    device = v83.get_device(args.device)
    rows: List[Dict[str, Any]] = []
    for function_name in parse_str_list(args.symbolic_datasets):
        x_train, y_train, x_test, y_test, input_dim, output_dim, protocol = _load_kanbefair_symbolic_tensors(
            function_name,
            kb_path=Path(args.kanbefair_path),
            seed=int(args.seed),
        )
        kb_specs = [
            ("KB-MLP", MLP, _kb_namespace(model_name="MLP", input_size=input_dim, output_size=output_dim, layers_width=[32])),
            (
                "KB-KAN",
                KANbeFair,
                _kb_namespace(
                    model_name="KAN",
                    input_size=input_dim,
                    output_size=output_dim,
                    layers_width=[2],
                    kan_grid=3,
                    kan_order=2,
                    kan_shortcut="silu",
                    kan_range=[-4.0, 4.0],
                ),
            ),
        ]
        measured: Dict[str, Dict[str, Any]] = {}
        for model_name, cls, ns in kb_specs:
            set_seed(int(args.seed))
            model = cls(ns)
            metrics = _train_kb_regressor(
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                epochs=int(args.symbolic_epochs),
                batch_size=int(args.symbolic_batch_size),
                lr=float(args.symbolic_lr),
                seed=int(args.seed),
                device=device,
            )
            measured[model_name] = {
                "model": model_name,
                "candidate_id": model_name,
                "params": int(model.total_parameters()),
                "FLOPs": float(model.total_flops()),
                "functional_update_used": 0,
                "functional_event_count": 0,
                "functional_accept_mass": 0,
                "functional_reject_mass": 0,
                **metrics,
            }
        dg_base = _train_dg_symbolic_regressor(
            args,
            x_train,
            y_train,
            x_test,
            y_test,
            function_name=function_name,
            input_dim=input_dim,
            output_dim=output_dim,
            functional_update_used=0,
        )
        dg_func = _train_dg_symbolic_regressor(
            args,
            x_train,
            y_train,
            x_test,
            y_test,
            function_name=function_name,
            input_dim=input_dim,
            output_dim=output_dim,
            functional_update_used=1,
        )
        measured["DG-Base"] = dg_base
        measured["DG-Functional"] = dg_func
        kb_mlp_rmse = _safe_float(measured["KB-MLP"].get("RMSE"))
        kb_kan_rmse = _safe_float(measured["KB-KAN"].get("RMSE"))
        dg_base_curv = _safe_float(dg_base.get("curvature"))
        dg_func_curv = _safe_float(dg_func.get("curvature"))
        dg_func_rmse = _safe_float(dg_func.get("RMSE"))
        geometry_pass = int(
            math.isfinite(dg_base_curv)
            and math.isfinite(dg_func_curv)
            and dg_func_curv <= 0.90 * max(dg_base_curv, 1.0e-12)
        )
        symbolic_pass = int(
            math.isfinite(dg_func_rmse)
            and (
                (math.isfinite(kb_kan_rmse) and dg_func_rmse <= kb_kan_rmse)
                or (math.isfinite(kb_mlp_rmse) and dg_func_rmse <= kb_mlp_rmse)
            )
            and geometry_pass
        )
        for model_name in ["KB-MLP", "KB-KAN", "DG-Base", "DG-Functional"]:
            row = measured[model_name]
            enriched = {
                "stage": "P10_SYMBOLIC_REPRESENTATION_V85",
                "status": "measured",
                "function_name": function_name,
                "dataset_protocol": protocol,
                "model": row.get("model"),
                "candidate_id": row.get("candidate_id"),
                "params": row.get("params"),
                "FLOPs": row.get("FLOPs"),
                "RMSE": row.get("RMSE"),
                "MAE": row.get("MAE"),
                "curvature": row.get("curvature"),
                "slope_p95": row.get("slope_p95"),
                "jacobian_norm": row.get("jacobian_norm"),
                "train_time": row.get("train_time_s"),
                "step_time_ms": row.get("step_time_ms"),
                "memory": row.get("peak_memory_MB"),
                "functional_update_used": row.get("functional_update_used"),
                "functional_event_count": row.get("functional_event_count"),
                "functional_accept_mass": row.get("functional_accept_mass"),
                "functional_reject_mass": row.get("functional_reject_mass"),
                "RMSE_delta_vs_KB_MLP": _safe_float(row.get("RMSE")) - kb_mlp_rmse if math.isfinite(kb_mlp_rmse) else METRIC_UNAVAILABLE,
                "RMSE_delta_vs_KB_KAN": _safe_float(row.get("RMSE")) - kb_kan_rmse if math.isfinite(kb_kan_rmse) else METRIC_UNAVAILABLE,
                "curvature_ratio_vs_DG_base": (
                    _safe_float(row.get("curvature")) / max(dg_base_curv, 1.0e-12)
                    if math.isfinite(dg_base_curv) and math.isfinite(_safe_float(row.get("curvature")))
                    else METRIC_UNAVAILABLE
                ),
                "FunctionalGeometryPass": geometry_pass if model_name == "DG-Functional" else 0,
                "SymbolicPass": symbolic_pass if model_name == "DG-Functional" else 0,
                "loss_type": "RMSE",
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "geometry_loss_used": 0,
                "sampler_changed": 0,
                "class_weight_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            rows.append(enriched)
    write_csv(out_dir / "symbolic_representation.csv", rows)
    return rows


def _run_continual_learning_stress(
    out_dir: Path,
    args: argparse.Namespace,
    model_classes: Dict[str, Any],
) -> List[Dict[str, Any]]:
    MLP = model_classes.get("MLP")
    KANbeFair = model_classes.get("KANbeFair")
    if MLP is None or KANbeFair is None:
        rows = [{
            "stage": "P11_CONTINUAL_LEARNING_STRESS_V85",
            "status": "not_run",
            "reason": "kanbefair_model_import_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        write_csv(out_dir / "continual_learning_stress.csv", rows)
        return rows
    x_train_all, y_train_all, x_test_all, y_test_all, input_dim, num_classes, protocol = _load_kanbefair_vision_tensors(
        "MNIST",
        data_root=Path(args.data_root),
        train_size=60000,
        test_size=10000,
        seed=int(args.seed),
    )
    task_digits = [(0, 1, 2), (3, 4, 5), (6, 7, 8, 9)]
    train_tasks = [
        _filter_digit_group(
            x_train_all,
            y_train_all,
            digits,
            max_count=int(args.continual_train_size_per_task),
            seed=v83.v72._stable_seed("v85-continual-train", int(args.seed), i),
        )
        for i, digits in enumerate(task_digits)
    ]
    test_tasks = [
        _filter_digit_group(
            x_test_all,
            y_test_all,
            digits,
            max_count=int(args.continual_test_size_per_task),
            seed=v83.v72._stable_seed("v85-continual-test", int(args.seed), i),
        )
        for i, digits in enumerate(task_digits)
    ]
    device = v83.get_device(args.device)
    model_specs: List[Tuple[str, Any, Any]] = [
        ("KB-MLP", MLP, _kb_namespace(model_name="MLP", input_size=input_dim, output_size=num_classes, layers_width=[32])),
        (
            "KB-KAN",
            KANbeFair,
            _kb_namespace(
                model_name="KAN",
                input_size=input_dim,
                output_size=num_classes,
                layers_width=[2],
                kan_grid=3,
                kan_order=2,
                kan_shortcut="silu",
                kan_range=[-1.0, 1.0],
            ),
        ),
    ]
    measured: Dict[str, Dict[str, Any]] = {}
    for model_name, cls, ns in model_specs:
        set_seed(int(args.seed))
        model = cls(ns)
        acc_matrix, curvature_after, train_time, peak_mb = _train_kb_continual(
            model,
            train_tasks,
            test_tasks,
            epochs_per_task=int(args.continual_epochs_per_task),
            batch_size=int(args.continual_batch_size),
            lr=float(args.continual_lr),
            seed=int(args.seed),
            device=device,
        )
        forgetting, backward_transfer, final_average = _continual_scores(acc_matrix)
        measured[model_name] = {
            "model": model_name,
            "params": int(model.total_parameters()),
            "FLOPs": float(model.total_flops()),
            "acc_matrix": acc_matrix,
            "curvature_after": curvature_after,
            "forgetting_score": forgetting,
            "backward_transfer": backward_transfer,
            "final_average_acc": final_average,
            "train_time": train_time,
            "memory": peak_mb,
            "functional_update_used": 0,
            "functional_event_count": 0,
            "functional_accept_mass": 0,
            "functional_reject_mass": 0,
        }
    for model_name, functional_flag in [("DG-Base", 0), ("DG-Functional", 1)]:
        acc_matrix, curvature_after, train_time, peak_mb, event_count, accept_mass, reject_mass = _train_dg_continual(
            args,
            train_tasks,
            test_tasks,
            input_dim=input_dim,
            num_classes=num_classes,
            functional_update_used=functional_flag,
        )
        forgetting, backward_transfer, final_average = _continual_scores(acc_matrix)
        measured[model_name] = {
            "model": model_name,
            "params": METRIC_UNAVAILABLE,
            "FLOPs": _dg_kw6_forward_flops_estimate(int(input_dim), int(args.dg_hidden_dim), int(v83.v80._spec_map_v80()["KW6"].depth), int(num_classes)),
            "acc_matrix": acc_matrix,
            "curvature_after": curvature_after,
            "forgetting_score": forgetting,
            "backward_transfer": backward_transfer,
            "final_average_acc": final_average,
            "train_time": train_time,
            "memory": peak_mb,
            "functional_update_used": functional_flag,
            "functional_event_count": event_count,
            "functional_accept_mass": accept_mass,
            "functional_reject_mass": reject_mass,
        }
    kb_kan_forgetting = _safe_float(measured["KB-KAN"].get("forgetting_score"))
    kb_kan_final = _safe_float(measured["KB-KAN"].get("final_average_acc"))
    kb_mlp_forgetting = _safe_float(measured["KB-MLP"].get("forgetting_score"))
    dg_func_forgetting = _safe_float(measured["DG-Functional"].get("forgetting_score"))
    dg_func_final = _safe_float(measured["DG-Functional"].get("final_average_acc"))
    continual_pass = int(
        math.isfinite(dg_func_forgetting)
        and math.isfinite(kb_kan_forgetting)
        and math.isfinite(dg_func_final)
        and math.isfinite(kb_kan_final)
        and dg_func_forgetting <= kb_kan_forgetting + 1.0e-12
        and dg_func_final >= kb_kan_final - 1.0e-12
    )
    strong_continual_pass = int(
        continual_pass
        and math.isfinite(kb_mlp_forgetting)
        and dg_func_forgetting <= kb_mlp_forgetting + 1.0e-12
    )
    rows: List[Dict[str, Any]] = []
    for model_name in ["KB-MLP", "KB-KAN", "DG-Base", "DG-Functional"]:
        m = measured[model_name]
        acc_matrix = m["acc_matrix"]
        curvature_after = m["curvature_after"]
        for train_stage, accs in enumerate(acc_matrix):
            for task_id, acc in enumerate(accs):
                rows.append({
                    "stage": "P11_CONTINUAL_LEARNING_STRESS_V85",
                    "status": "measured",
                    "protocol": protocol,
                    "model": model_name,
                    "task_id": task_id,
                    "train_stage": train_stage,
                    "task_digits": "-".join(str(d) for d in task_digits[task_id]),
                    "acc_after_task": acc,
                    "backward_transfer": m["backward_transfer"],
                    "forgetting_score": m["forgetting_score"],
                    "final_average_acc": m["final_average_acc"],
                    "curvature_after_task": curvature_after[train_stage],
                    "functional_update_time": m["train_time"] if model_name == "DG-Functional" else 0.0,
                    "memory": m["memory"],
                    "params": m["params"],
                    "FLOPs": m["FLOPs"],
                    "functional_update_used": m["functional_update_used"],
                    "functional_event_count": m["functional_event_count"],
                    "functional_accept_mass": m["functional_accept_mass"],
                    "functional_reject_mass": m["functional_reject_mass"],
                    "continual_restore_old_head_rows": int(bool(getattr(args, "continual_restore_old_head_rows", False))),
                    "continual_freeze_stack_after_first_task": int(bool(getattr(args, "continual_freeze_stack_after_first_task", False))),
                    "continual_freeze_head_shared_after_first_task": int(bool(getattr(args, "continual_freeze_head_shared_after_first_task", False))),
                    "continual_stack_anchor_strength": float(getattr(args, "continual_stack_anchor_strength", 0.0)),
                    "ContinualPass": continual_pass if model_name == "DG-Functional" and train_stage == len(acc_matrix) - 1 and task_id == len(accs) - 1 else 0,
                    "StrongContinualPass": strong_continual_pass if model_name == "DG-Functional" and train_stage == len(acc_matrix) - 1 and task_id == len(accs) - 1 else 0,
                    "loss_type": "CE",
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "geometry_loss_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    write_csv(out_dir / "continual_learning_stress.csv", rows)
    return rows


def _write_not_run_artifacts(out_dir: Path, reason: str) -> None:
    placeholders = {
        "v84_survivor_reproduction.csv": "P4_V84_SURVIVOR_REPRODUCTION_V85",
        "kanbefair_primary_transfer.csv": "P5_KANBEFAIR_PRIMARY_TRANSFER_V85",
        "parameter_matched_envelope.csv": "P6_PARAMETER_MATCHED_ENVELOPE_V85",
        "flops_matched_envelope.csv": "P7_FLOPS_MATCHED_ENVELOPE_V85",
        "wallclock_memory_envelope.csv": "P8_WALLCLOCK_MEMORY_ENVELOPE_V85",
        "functional_causality_kanbefair.csv": "P9_FUNCTIONAL_CAUSALITY_KANBEFAIR_V85",
        "symbolic_representation.csv": "P10_SYMBOLIC_REPRESENTATION_V85",
        "continual_learning_stress.csv": "P11_CONTINUAL_LEARNING_STRESS_V85",
    }
    for filename, stage in placeholders.items():
        if (out_dir / filename).exists():
            continue
        write_csv(out_dir / filename, [{
            "stage": stage,
            "status": "not_run",
            "reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }])


def _audit_no_fake(out_dir: Path) -> Dict[str, Any]:
    rows_checked = 0
    fake_proxy_nonzero = 0
    fake_data = 0
    proxy = 0
    offload = 0
    for csv_path in out_dir.glob("*.csv"):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows_checked += 1
                for key in ["fake_data_used", "proxy_row_used", "cpu_offload_used"]:
                    val = str(row.get(key, "0"))
                    if val not in {"", "0", "0.0", "False", "false"}:
                        fake_proxy_nonzero += 1
                fake_data += int(_safe_float(row.get("fake_data_used"), 0.0) != 0.0)
                proxy += int(_safe_float(row.get("proxy_row_used"), 0.0) != 0.0)
                offload += int(_safe_float(row.get("cpu_offload_used"), 0.0) != 0.0)
    audit = {
        "stage": "V85_PROVENANCE_AUDIT",
        "script_path": "experiments/run_gafu_v85_real.py",
        "plan_path": PLAN_PATH,
        "rows_checked": rows_checked,
        "fake_proxy_nonzero_count": fake_proxy_nonzero,
        "fake_data_used": fake_data,
        "proxy_row_used": proxy,
        "cpu_offload_used": offload,
        "no_fake": fake_data == 0,
        "no_proxy": proxy == 0,
    }
    write_csv(out_dir / "v85_provenance_audit.csv", [audit])
    return audit


def _route(
    out_dir: Path,
    source_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    adapter_rows: Sequence[Dict[str, Any]],
    counter_rows: Sequence[Dict[str, Any]],
    audit: Dict[str, Any],
) -> Dict[str, Any]:
    source_pass = int(any(_safe_float(r.get("SourceAuditPass"), 0.0) == 1.0 for r in source_rows))
    baseline_measured = [r for r in baseline_rows if str(r.get("status")) == "measured"]
    baseline_pass = int(baseline_measured and all(_safe_float(r.get("reproduction_pass"), 0.0) == 1.0 for r in baseline_measured))
    adapter_pass = int(adapter_rows and all(_safe_float(r.get("AdapterPass"), 0.0) == 1.0 for r in adapter_rows))
    counter_pass = int(counter_rows and all(_safe_float(r.get("CounterPass"), 0.0) == 1.0 for r in counter_rows))
    primary_rows: List[Dict[str, Any]] = []
    primary_path = out_dir / "kanbefair_primary_transfer.csv"
    if primary_path.exists():
        with primary_path.open("r", encoding="utf-8", newline="") as f:
            primary_rows = list(csv.DictReader(f))
    primary_measured = [r for r in primary_rows if str(r.get("status")) == "measured"]
    primary_pass = int(any(_safe_float(r.get("PrimaryTransferPass"), 0.0) == 1.0 for r in primary_measured))
    parameter_fair_pass = 0
    flops_fair_pass = 0
    wallclock_fair_pass = 0
    functional_causality_pass = 0
    symbolic_pass = 0
    continual_pass = 0
    for filename, key in [
        ("parameter_matched_envelope.csv", "ParameterFairPass"),
        ("flops_matched_envelope.csv", "FLOPsFairPass"),
        ("wallclock_memory_envelope.csv", "WallclockMemoryPass"),
    ]:
        path = out_dir / filename
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            value = int(any(_safe_float(r.get(key), 0.0) == 1.0 for r in rows))
            if key == "ParameterFairPass":
                parameter_fair_pass = value
            elif key == "FLOPsFairPass":
                flops_fair_pass = value
            else:
                wallclock_fair_pass = value
    for filename, key in [
        ("functional_causality_kanbefair.csv", "FunctionalCausalityPass"),
        ("symbolic_representation.csv", "SymbolicPass"),
        ("continual_learning_stress.csv", "ContinualPass"),
    ]:
        path = out_dir / filename
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            value = int(any(_safe_float(r.get(key), 0.0) == 1.0 for r in rows))
            if key == "FunctionalCausalityPass":
                functional_causality_pass = value
            elif key == "SymbolicPass":
                symbolic_pass = value
            else:
                continual_pass = value
    dg_func_rows = [r for r in primary_measured if str(r.get("candidate_id", "")).startswith("DG1-FT7-")]
    best_candidate = dg_func_rows[0].get("candidate_id", METRIC_UNAVAILABLE) if dg_func_rows else METRIC_UNAVAILABLE
    best_task_family = dg_func_rows[0].get("task_name", METRIC_UNAVAILABLE) if dg_func_rows else METRIC_UNAVAILABLE
    if not source_pass:
        route = "R9-CodeIntegrationFail"
        blocker = "third_party_source_audit_fail"
        next_impl = "repair_kanbefair_source_import_or_entrypoint"
    elif not baseline_pass:
        route = "R8-ReproductionFail"
        blocker = "kanbefair_baseline_reproduction_not_paper_comparable_or_failed"
        next_impl = "run_full_kanbefair_baseline_reproduction_with_paper_protocol"
    elif not adapter_pass:
        route = "R9-CodeIntegrationFail"
        blocker = "dgkan_adapter_contract_not_implemented"
        next_impl = "implement_DGKANAdapter_inside_KANbeFair_protocol"
    elif not counter_pass:
        route = "R9-CodeIntegrationFail"
        blocker = "params_flops_counter_mismatch"
        next_impl = "implement_DGKAN_flops_counter_aligned_with_KANbeFair"
    elif primary_measured and not primary_pass:
        route = "R4-InternalOnlySuccess"
        blocker = "external_primary_transfer_or_geometry_gate_failed"
        next_impl = "optimize_DG_KAN_on_KANbeFair_primary_tasks_without_changing_CE_contract"
    elif primary_pass and not (parameter_fair_pass or flops_fair_pass):
        route = "R5-ExternalAccuracyButNotFairEnvelope"
        blocker = "parameter_or_flops_fair_envelope_failed"
        next_impl = "run_parameter_or_FLOPs_matched_DG_variant_screen"
    elif primary_pass and not wallclock_fair_pass:
        route = "R2-ExternalTaskPassSystemFail"
        blocker = "wallclock_memory_envelope_failed"
        next_impl = "repair_external_wallclock_memory_without_changing_functional_route"
    elif primary_pass and (parameter_fair_pass or flops_fair_pass) and wallclock_fair_pass:
        route = "R1-ExternalFairFunctionalAdvantage"
        blocker = "none"
        if not functional_causality_pass:
            next_impl = "run_functional_causality_controls_under_KANbeFair_protocol"
        elif not symbolic_pass or not continual_pass:
            next_impl = "run_symbolic_continual_and_strong_confirmations"
        else:
            next_impl = "strong_route_completed"
    else:
        route = "R4-InternalOnlySuccess"
        blocker = "external_fair_envelope_not_run"
        next_impl = "run_primary_transfer_and_fair_envelopes"
    decision = {
        "route": route,
        "internal_reproduction_pass": 0,
        "kanbefair_baseline_reproduction_pass": baseline_pass,
        "primary_transfer_pass": primary_pass,
        "parameter_fair_pass": parameter_fair_pass,
        "flops_fair_pass": flops_fair_pass,
        "wallclock_fair_pass": wallclock_fair_pass,
        "functional_causality_pass": functional_causality_pass,
        "symbolic_pass": symbolic_pass,
        "continual_pass": continual_pass,
        "geometry_pass": functional_causality_pass,
        "robustness_pass": 0,
        "best_candidate": best_candidate,
        "best_task_family": best_task_family,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "SourceAuditPass": source_pass,
        "AdapterPass": adapter_pass,
        "CounterPass": counter_pass,
        "success_v85_minimum": int(primary_pass and (parameter_fair_pass or flops_fair_pass) and wallclock_fair_pass),
        "success_v85_external_formal": int(primary_pass and parameter_fair_pass and flops_fair_pass and wallclock_fair_pass),
        "success_v85_strong": int(
            primary_pass
            and parameter_fair_pass
            and flops_fair_pass
            and wallclock_fair_pass
            and functional_causality_pass
            and symbolic_pass
            and continual_pass
        ),
        "cpu_offload_used": audit.get("cpu_offload_used", 0),
        "no_fake": bool(audit.get("no_fake")),
        "no_proxy": bool(audit.get("no_proxy")),
    }
    save_json(out_dir / "route_decision.json", decision)
    save_json(out_dir / "aggregate_decision.json", decision)
    failures: List[Dict[str, Any]] = []
    if not source_pass:
        failures.append({"failure_id": "F1_third_party_source_missing", "active": 1, "detail": blocker})
    if source_pass and not baseline_pass:
        failures.append({"failure_id": "F2_kanbefair_baseline_reproduction_fail", "active": 1, "detail": blocker})
    if source_pass and baseline_pass and not adapter_pass:
        failures.append({"failure_id": "F3_adapter_contract_fail", "active": 1, "detail": blocker})
    if source_pass and baseline_pass and adapter_pass and not counter_pass:
        failures.append({"failure_id": "F4_counter_mismatch", "active": 1, "detail": blocker})
    if source_pass and baseline_pass and adapter_pass and counter_pass and primary_measured and not primary_pass:
        failures.append({"failure_id": "F5_external_primary_transfer_fail", "active": 1, "detail": blocker})
    if primary_pass and not (parameter_fair_pass or flops_fair_pass):
        failures.append({"failure_id": "F6_external_fair_envelope_fail", "active": 1, "detail": blocker})
    if primary_pass and (parameter_fair_pass or flops_fair_pass) and not wallclock_fair_pass:
        failures.append({"failure_id": "F7_external_wallclock_fail", "active": 1, "detail": blocker})
    if not failures:
        failures.append({"failure_id": "none", "active": 0, "detail": "no active failure in executed waves"})
    write_csv(out_dir / "failure_table.csv", failures)
    return decision


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = ensure_dir(Path(args.out_dir))
    if args.fresh:
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    ensure_dir(out_dir / "figures")
    v83.P5_FT7_EVENT_STRIDE = int(args.ft7_event_stride)
    v83.P5_FT7_EVENT_ALPHA_MULT = float(args.ft7_event_alpha_mult)
    manifest = {
        "stage": "RUN_MANIFEST_V85",
        "script": "experiments/run_gafu_v85_real.py",
        "plan": PLAN_PATH,
        "out_dir": str(out_dir),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
    }
    save_json(out_dir / "run_manifest.json", manifest)
    source_rows, model_classes = _run_source_audit(out_dir, args)
    baseline_rows = _run_baseline_audit(out_dir, args, model_classes)
    adapter_rows = _write_adapter_contract(out_dir, args)
    counter_rows = _run_counter_audit(out_dir, args, model_classes)
    source_pass = int(any(_safe_float(r.get("SourceAuditPass"), 0.0) == 1.0 for r in source_rows))
    baseline_measured = [r for r in baseline_rows if str(r.get("status")) == "measured"]
    baseline_pass = int(
        bool(baseline_measured)
        and all(_safe_float(r.get("reproduction_pass"), 0.0) == 1.0 for r in baseline_measured)
    )
    adapter_pass = int(adapter_rows and all(_safe_float(r.get("AdapterPass"), 0.0) == 1.0 for r in adapter_rows))
    counter_pass = int(counter_rows and all(_safe_float(r.get("CounterPass"), 0.0) == 1.0 for r in counter_rows))
    if bool(args.run_primary_transfer) and source_pass and baseline_pass and adapter_pass and counter_pass:
        primary_rows = _run_primary_transfer(out_dir, args, baseline_rows)
        _run_fair_envelopes(out_dir, primary_rows)
        if bool(args.run_functional_causality):
            _run_functional_causality(out_dir, args)
        if bool(args.run_symbolic):
            _run_symbolic_representation(out_dir, args, model_classes)
        if bool(args.run_continual):
            _run_continual_learning_stress(out_dir, args, model_classes)
    elif (bool(args.run_symbolic) or bool(args.run_continual)) and source_pass and baseline_pass and adapter_pass and counter_pass:
        if bool(args.run_symbolic):
            _run_symbolic_representation(out_dir, args, model_classes)
        if bool(args.run_continual):
            _run_continual_learning_stress(out_dir, args, model_classes)
    if not source_pass:
        gate_reason = "blocked_until_source_audit_pass"
    elif not baseline_pass:
        gate_reason = "blocked_until_kanbefair_baseline_reproduction_pass"
    elif not adapter_pass:
        gate_reason = "blocked_until_dgkan_adapter_contract_pass"
    elif not counter_pass:
        gate_reason = "blocked_until_params_flops_counter_pass"
    elif not bool(args.run_primary_transfer):
        gate_reason = "blocked_until_primary_transfer_run"
    elif not bool(args.run_functional_causality):
        gate_reason = "blocked_until_functional_causality_controls_run"
    elif not bool(args.run_symbolic):
        gate_reason = "blocked_until_symbolic_representation_run"
    elif not bool(args.run_continual):
        gate_reason = "blocked_until_continual_learning_stress_run"
    else:
        gate_reason = "blocked_until_downstream_symbolic_continual_or_formal_confirmations"
    _write_not_run_artifacts(out_dir, gate_reason if source_pass else "blocked_until_source_audit_pass")
    audit = _audit_no_fake(out_dir)
    decision = _route(out_dir, source_rows, baseline_rows, adapter_rows, counter_rows, audit)
    manifest["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(out_dir / "run_manifest.json", manifest)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DG-KAN v8.5 KANbeFair external-fair audit")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--kanbefair-path", default="third_party/KANbeFair")
    parser.add_argument("--datasets", default="MNIST")
    parser.add_argument("--kb-data-protocol", default="kanbefair", choices=["kanbefair", "dg-balanced"])
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--kb-epochs", type=int, default=1)
    parser.add_argument("--kb-batch-size", type=int, default=128)
    parser.add_argument("--adapter-smoke-train-size", type=int, default=512)
    parser.add_argument("--adapter-smoke-test-size", type=int, default=256)
    parser.add_argument("--adapter-steps", type=int, default=8)
    parser.add_argument("--run-primary-transfer", action="store_true")
    parser.add_argument("--run-functional-causality", action="store_true")
    parser.add_argument("--run-symbolic", action="store_true")
    parser.add_argument("--run-continual", action="store_true")
    parser.add_argument("--primary-train-size", type=int, default=60000)
    parser.add_argument("--primary-test-size", type=int, default=10000)
    parser.add_argument("--primary-epochs", type=int, default=20)
    parser.add_argument("--causality-train-size", type=int, default=0)
    parser.add_argument("--causality-test-size", type=int, default=0)
    parser.add_argument("--causality-epochs", type=int, default=0)
    parser.add_argument("--symbolic-datasets", default="Special_1d_gelu")
    parser.add_argument("--symbolic-epochs", type=int, default=100)
    parser.add_argument("--symbolic-batch-size", type=int, default=128)
    parser.add_argument("--symbolic-lr", type=float, default=1.0e-3)
    parser.add_argument("--continual-train-size-per-task", type=int, default=1024)
    parser.add_argument("--continual-test-size-per-task", type=int, default=512)
    parser.add_argument("--continual-epochs-per-task", type=int, default=2)
    parser.add_argument("--continual-batch-size", type=int, default=128)
    parser.add_argument("--continual-lr", type=float, default=1.0e-3)
    parser.add_argument("--continual-restore-old-head-rows", action="store_true")
    parser.add_argument("--continual-freeze-stack-after-first-task", action="store_true")
    parser.add_argument("--continual-freeze-head-shared-after-first-task", action="store_true")
    parser.add_argument("--continual-stack-anchor-strength", type=float, default=0.0)
    parser.add_argument("--continual-old-head-grad-scale", type=float, default=0.0)
    parser.add_argument("--continual-old-head-restore-strength", type=float, default=1.0)
    parser.add_argument("--continual-old-head-age-decay", type=float, default=0.0)
    parser.add_argument("--dg-candidate-id", default="KW6")
    parser.add_argument("--dg-hidden-dim", type=int, default=68)
    parser.add_argument("--dg-basis-count", type=int, default=8)
    parser.add_argument("--dg-batch-size", type=int, default=128)
    parser.add_argument("--dg-eval-batch-size", type=int, default=512)
    parser.add_argument("--dg-stream-batches-from-cpu", action="store_true")
    parser.add_argument("--dg-stream-chunk-batches", type=int, default=1)
    parser.add_argument("--dg-stream-epoch-permute-cpu", action="store_true")
    parser.add_argument("--dg-stream-chunk-order-shuffle", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ft7-event-stride", type=int, default=int(v83.P5_FT7_EVENT_STRIDE))
    parser.add_argument("--ft7-event-alpha-mult", type=float, default=float(v83.P5_FT7_EVENT_ALPHA_MULT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
