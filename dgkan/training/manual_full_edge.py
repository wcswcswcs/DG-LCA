"""Manual training helpers for v9 full-edge PureKAN candidates."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from dgkan.models.manual_full_edge import ManualFullEdgeClassifier
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig, adamw_update_


@dataclass(frozen=True)
class FullEdgeTrainConfig:
    candidate_id: str
    dims: Tuple[int, ...]
    edge_kind: str
    epochs: int = 20
    batch_size: int = 128
    eval_batch_size: int = 512
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    functional_update: str = "none"
    functional_interval: int = 128
    functional_strength: float = 0.05
    functional_guard_ratio: float = 1.003


def ce_loss_and_grad(logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logp = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(logp, y)
    grad = logp.exp()
    grad[torch.arange(int(y.numel()), device=logits.device), y] -= 1.0
    grad = grad / max(1, int(y.numel()))
    return loss, grad


def ece_from_logits(logits: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == y).float()
    ece = torch.zeros((), device=logits.device)
    for idx in range(int(bins)):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (conf > lo) & (conf <= hi)
        if bool(mask.any()):
            ece = ece + mask.float().mean() * torch.abs(conf[mask].mean() - correct[mask].mean())
    return float(ece.detach().cpu())


def evaluate_manual_full_edge(
    model: ManualFullEdgeClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int = 512,
) -> Dict[str, float]:
    logits_parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            logits, _ = model.forward(x[start : start + int(batch_size)])
            logits_parts.append(logits)
        logits_all = torch.cat(logits_parts, dim=0)
        loss = F.cross_entropy(logits_all, y)
        pred = logits_all.argmax(dim=1)
        acc = (pred == y).float().mean()
    return {
        "test_acc": float(acc.detach().cpu()),
        "test_acc_pct": float(acc.detach().cpu()) * 100.0,
        "NLL": float(loss.detach().cpu()),
        "ECE": ece_from_logits(logits_all, y),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def train_manual_full_edge(
    cfg: FullEdgeTrainConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> Tuple[ManualFullEdgeClassifier, Dict[str, Any], List[Dict[str, Any]]]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = ManualFullEdgeClassifier(cfg.dims, edge_kind=cfg.edge_kind, device=device)
    states = [AdamWState.zeros_like(layer.theta) for layer in model.layers]
    opt_cfg = ManualAdamWConfig(lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    n_train = int(x_train.shape[0])
    steps_per_epoch = max(1, math.ceil(n_train / int(cfg.batch_size)))
    losses: List[float] = []
    accepted = 0
    rejected = 0
    holdout_ratios: List[float] = []
    raw_rows: List[Dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    step = 0
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for epoch in range(int(cfg.epochs)):
        perm = torch.randperm(n_train, device=device, generator=generator)
        for batch_idx in range(steps_per_epoch):
            step += 1
            start = batch_idx * int(cfg.batch_size)
            end = min(start + int(cfg.batch_size), n_train)
            idx = perm[start:end]
            if int(idx.numel()) == 0:
                continue
            xb = x_train[idx]
            yb = y_train[idx]
            logits, caches = model.forward(xb)
            loss, grad_logits = ce_loss_and_grad(logits, yb)
            grads = model.backward(grad_logits, caches)
            for layer, grad, state in zip(model.layers, grads, states):
                adamw_update_(layer.theta, grad, state, opt_cfg)
            losses.append(float(loss.detach().cpu()))

            if cfg.functional_update != "none" and step % max(1, int(cfg.functional_interval)) == 0:
                holdout_start = ((batch_idx + 1) % steps_per_epoch) * int(cfg.batch_size)
                holdout_end = min(holdout_start + int(cfg.batch_size), n_train)
                holdout_idx = perm[holdout_start:holdout_end]
                if int(holdout_idx.numel()) == 0:
                    holdout_idx = idx
                xh = x_train[holdout_idx]
                yh = y_train[holdout_idx]
                with torch.no_grad():
                    before_logits, _ = model.forward(xh)
                    before_loss = F.cross_entropy(before_logits, yh)
                    before_curv = model.curvature()
                    snapshot = model.clone_parameters()
                    if cfg.functional_update == "random":
                        scale = float(cfg.functional_strength) * 0.001
                        for layer in model.layers:
                            layer.theta.add_(torch.randn_like(layer.theta), alpha=scale)
                    elif cfg.functional_update == "shuffle":
                        for layer in model.layers:
                            layer.theta[..., 0].mul_(max(0.0, 1.0 - float(cfg.functional_strength)))
                    elif cfg.functional_update == "functional":
                        model.apply_functional_smoothing_(float(cfg.functional_strength))
                    after_logits, _ = model.forward(xh)
                    after_loss = F.cross_entropy(after_logits, yh)
                    after_curv = model.curvature()
                    ratio = float((after_loss / before_loss.clamp_min(1.0e-12)).detach().cpu())
                    holdout_ratios.append(ratio)
                    ok = ratio <= float(cfg.functional_guard_ratio)
                    if ok:
                        accepted += 1
                    else:
                        rejected += 1
                        model.restore_parameters_(snapshot)
                    raw_rows.append(
                        {
                            "step": step,
                            "epoch": epoch,
                            "functional_update": cfg.functional_update,
                            "event_triggered": 1,
                            "event_accepted": int(ok),
                            "holdout_descent_ratio": ratio,
                            "curvature_before": float(before_curv.detach().cpu()),
                            "curvature_after": float(after_curv.detach().cpu()),
                            "bad_step_flag": int(not ok),
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        }
                    )
    _sync(device)
    elapsed = time.perf_counter() - started
    eval_metrics = evaluate_manual_full_edge(model, x_test, y_test, batch_size=int(cfg.eval_batch_size))
    summary: Dict[str, Any] = {
        **eval_metrics,
        "candidate_id": cfg.candidate_id,
        "model_level": "full_edge",
        "edge_basis": cfg.edge_kind,
        "dims": "x".join(str(v) for v in cfg.dims),
        "loss_type": "CE",
        "label_smoothing": 0.0,
        "uses_loss_backward": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "geometry_loss_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "functional_update": cfg.functional_update,
        "functional_events": accepted + rejected,
        "functional_accept_count": accepted,
        "functional_reject_count": rejected,
        "functional_bad_step_rate": rejected / max(1, accepted + rejected),
        "functional_holdout_ratio_mean": sum(holdout_ratios) / max(1, len(holdout_ratios)),
        "curvature": float(model.curvature().detach().cpu()),
        "params": model.parameter_count(),
        "FLOPs_forward": model.flops_count(),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "train_loss_last": losses[-1] if losses else float("nan"),
        "train_time_s": elapsed,
        "step_time_ms": elapsed * 1000.0 / max(1, int(cfg.epochs) * steps_per_epoch),
        "peak_memory_MB": _peak_memory_mb(device),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return model, summary, raw_rows


def time_manual_full_edge_step(
    cfg: FullEdgeTrainConfig,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    warmup: int,
    reps: int,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    model = ManualFullEdgeClassifier(cfg.dims, edge_kind=cfg.edge_kind, device=device)
    states = [AdamWState.zeros_like(layer.theta) for layer in model.layers]
    opt_cfg = ManualAdamWConfig(lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    x = x.to(device)
    y = y.to(device)
    bs = min(int(cfg.batch_size), int(x.shape[0]))

    def one_step(pos: int) -> None:
        start = (pos * bs) % int(x.shape[0])
        idx = torch.arange(start, min(start + bs, int(x.shape[0])), device=device)
        if int(idx.numel()) < bs:
            idx = torch.arange(0, bs, device=device) % int(x.shape[0])
        logits, caches = model.forward(x[idx])
        _loss, grad = ce_loss_and_grad(logits, y[idx])
        grads = model.backward(grad, caches)
        for layer, grad_tensor, state in zip(model.layers, grads, states):
            adamw_update_(layer.theta, grad_tensor, state, opt_cfg)

    for i in range(int(warmup)):
        one_step(i)
    _sync(device)
    started = time.perf_counter()
    for i in range(int(reps)):
        one_step(i + int(warmup))
    _sync(device)
    elapsed = time.perf_counter() - started
    return {
        "candidate_id": cfg.candidate_id,
        "step_time_ms": elapsed * 1000.0 / max(1, int(reps)),
        "warmup": int(warmup),
        "reps": int(reps),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }

