#!/usr/bin/env python3
"""Minimal DG-LCA v0.3 experiments.

This script implements small, auditable gates from docs/DG-LCA_v0.3_*:

Gate 0: credit subspace diagnostics on a toy classification MLP.
Gate 1: single-block local VJP distillation with a conditional linear router.
Gate 2: KAN-like 1D edge-function regression with Sobolev preconditioning.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits, make_circles, make_moons
from tqdm import trange


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_float(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu())
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=to_float)


def parse_float_list(text: str) -> List[float]:
    if not text:
        return []
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def make_toy_classification(
    dataset: str,
    n_samples: int,
    noise: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dataset == "moons":
        x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif dataset == "circles":
        x, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.45,
            random_state=seed,
        )
    elif dataset == "digits":
        digits = load_digits()
        x = digits.data
        y = digits.target
        if n_samples < len(x):
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(x), size=n_samples, replace=False)
            x = x[keep]
            y = y[keep]
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    x = x.astype("float32")
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    y = y.astype("int64")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    split = int(0.8 * n_samples)
    train_idx, test_idx = perm[:split], perm[split:]
    return (
        torch.from_numpy(x[train_idx]),
        torch.from_numpy(y[train_idx]),
        torch.from_numpy(x[test_idx]),
        torch.from_numpy(y[test_idx]),
    )


def batch_indices(n: int, batch_size: int, seed: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


class InstrumentedMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, depth: int = 4, out_dim: int = 2):
        super().__init__()
        blocks: List[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * depth
        for i in range(depth):
            blocks.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]),
                    nn.SiLU(),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, retain: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = x
        activations: List[torch.Tensor] = []
        for block in self.blocks:
            h = block(h)
            if retain:
                h.retain_grad()
                activations.append(h)
        return self.head(h), activations


def accuracy(model: InstrumentedMLP, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        logits, _ = model(x.to(device), retain=False)
        pred = logits.argmax(dim=-1)
        return float((pred == y.to(device)).float().mean().cpu())


def singular_metrics(credits: torch.Tensor, top_rs: Tuple[int, ...]) -> Dict[str, Any]:
    credits = credits.detach().float().cpu()
    _, s, vh = torch.linalg.svd(credits, full_matrices=False)
    energy = s.square()
    denom = energy.sum().clamp_min(1e-12)
    p = energy / denom
    out: Dict[str, Any] = {
        "num_samples": int(credits.shape[0]),
        "dim": int(credits.shape[1]),
        "rank_upper_bound": int(min(credits.shape)),
        "participation_rank": float(1.0 / p.square().sum().clamp_min(1e-12)),
        "entropy_rank": float(torch.exp(-(p * torch.log(p + 1e-12)).sum())),
        "top_singular_values": [float(v) for v in s[: min(8, s.numel())]],
        "variance_energy": {},
        "nuclear_energy": {},
    }
    s_sum = s.sum().clamp_min(1e-12)
    for r in top_rs:
        rr = min(r, s.numel())
        out["variance_energy"][f"E{r}"] = float(energy[:rr].sum() / denom)
        out["nuclear_energy"][f"E{r}"] = float(s[:rr].sum() / s_sum)
    out["_vh"] = vh
    return out


def subspace_overlap(vh_a: torch.Tensor, vh_b: torch.Tensor, r: int) -> float:
    rr = min(r, vh_a.shape[0], vh_b.shape[0])
    if rr <= 0:
        return float("nan")
    ua = vh_a[:rr].T
    ub = vh_b[:rr].T
    return float((ua.T @ ub).square().sum() / rr)


def audit_credit_rank(
    model: InstrumentedMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    audit_batches: int,
    device: torch.device,
    seed: int,
    top_rs: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
    overlap_r: int = 16,
) -> Dict[str, Any]:
    model.eval()
    all_layer_credits: Optional[List[List[torch.Tensor]]] = None
    per_batch_vh: Optional[List[List[torch.Tensor]]] = None

    batches = list(batch_indices(len(x), batch_size, seed))[:audit_batches]
    for idx in batches:
        xb = x[idx].to(device)
        yb = y[idx].to(device)
        model.zero_grad(set_to_none=True)
        logits, activations = model(xb, retain=True)
        loss = F.cross_entropy(logits, yb)
        loss.backward()

        if all_layer_credits is None:
            all_layer_credits = [[] for _ in activations]
            per_batch_vh = [[] for _ in activations]

        for layer_id, act in enumerate(activations):
            grad = act.grad.detach().cpu()
            all_layer_credits[layer_id].append(grad)
            batch_metrics = singular_metrics(grad, top_rs)
            per_batch_vh[layer_id].append(batch_metrics["_vh"])

    assert all_layer_credits is not None
    assert per_batch_vh is not None

    layers: Dict[str, Any] = {}
    for layer_id, parts in enumerate(all_layer_credits):
        credits = torch.cat(parts, dim=0)
        metrics = singular_metrics(credits, top_rs)
        batch_overlaps = []
        for a, b in zip(per_batch_vh[layer_id], per_batch_vh[layer_id][1:]):
            batch_overlaps.append(subspace_overlap(a, b, overlap_r))
        metrics["batch_subspace_overlap_top16_mean"] = (
            float(np.mean(batch_overlaps)) if batch_overlaps else float("nan")
        )
        metrics["batch_subspace_overlap_top16_values"] = batch_overlaps
        metrics.pop("_vh")
        layers[f"layer_{layer_id}"] = metrics

    return {
        "batch_size": batch_size,
        "audit_batches": len(batches),
        "layers": layers,
    }


def run_gate0(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    set_seed(args.seed)
    device = get_device(args.device)
    x_train, y_train, x_test, y_test = make_toy_classification(
        args.dataset,
        args.samples,
        args.noise,
        args.seed,
    )
    in_dim = int(x_train.shape[1])
    out_dim = int(torch.cat([y_train, y_test]).max().item() + 1)
    model = InstrumentedMLP(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        out_dim=out_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    audits: List[Dict[str, Any]] = []
    audit_epochs = sorted(set([0, max(1, args.epochs // 2), args.epochs]))

    def do_audit(epoch: int) -> Dict[str, Any]:
        audit = audit_credit_rank(
            model,
            x_train,
            y_train,
            batch_size=args.batch_size,
            audit_batches=args.audit_batches,
            device=device,
            seed=args.seed + 1000 + epoch,
        )
        audit["epoch"] = epoch
        audit["train_acc"] = accuracy(model, x_train, y_train, device)
        audit["test_acc"] = accuracy(model, x_test, y_test, device)
        return audit

    if 0 in audit_epochs:
        audits.append(do_audit(0))

    for epoch in trange(1, args.epochs + 1, desc="gate0 train", leave=False):
        model.train()
        for idx in batch_indices(len(x_train), args.batch_size, args.seed + epoch):
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(xb, retain=False)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
        if epoch in audit_epochs:
            audits.append(do_audit(epoch))

    # Cross-epoch overlap is recomputed from fresh audits on a fixed batch seed.
    fixed_vhs: List[List[torch.Tensor]] = []
    for audit_epoch in audit_epochs:
        # The stored JSON-friendly audits do not keep singular vectors. To keep this
        # script simple and deterministic, report batch-overlap in the main audit
        # and leave full cross-epoch vector persistence to larger experiments.
        fixed_vhs.append([])

    summary = {
        "gate": "gate0_credit_rank",
        "dataset": args.dataset,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "audits": audits,
        "interpretation": interpret_gate0(audits[-1]),
    }
    save_json(out_dir / "gate0_credit_rank.json", summary)
    return summary


def interpret_gate0(final_audit: Dict[str, Any]) -> Dict[str, Any]:
    layers = final_audit["layers"]
    e64_values = []
    pr_values = []
    overlap_values = []
    for metrics in layers.values():
        e64_values.append(metrics["variance_energy"].get("E64", float("nan")))
        pr_values.append(metrics["participation_rank"])
        overlap_values.append(metrics["batch_subspace_overlap_top16_mean"])
    mean_e64 = float(np.nanmean(e64_values))
    mean_pr = float(np.nanmean(pr_values))
    mean_overlap = float(np.nanmean(overlap_values))
    if mean_e64 > 0.9 and mean_overlap > 0.5:
        level = "medium"
    elif mean_e64 > 0.8:
        level = "weak"
    else:
        level = "not_passed"
    return {
        "mean_E64_variance_energy": mean_e64,
        "mean_participation_rank": mean_pr,
        "mean_batch_subspace_overlap_top16": mean_overlap,
        "gate0_status": level,
    }


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, scale: float = 0.7):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.scale = scale

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.scale * self.lin2(torch.tanh(self.lin1(h)))


class PlainMLPBlock(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim, bias=False)
        self.lin2 = nn.Linear(dim, dim, bias=False)
        self.scale = scale
        nn.init.orthogonal_(self.lin1.weight)
        nn.init.orthogonal_(self.lin2.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scale * self.lin2(torch.tanh(self.lin1(h)))


class BottleneckResidualBlock(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 16, scale: float = 1.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.scale = scale
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.scale * self.up(torch.tanh(self.down(h)))


def build_gate1_block(block_type: str, dim: int, scale: float, bottleneck: int) -> nn.Module:
    if block_type == "residual":
        return ResidualBlock(dim, scale=scale)
    if block_type == "mlp":
        return PlainMLPBlock(dim, scale=scale)
    if block_type == "bottleneck_residual":
        return BottleneckResidualBlock(dim, bottleneck=bottleneck, scale=scale)
    raise ValueError(f"unknown gate1 block type: {block_type}")


class DiagLowRankRouter(nn.Module):
    def __init__(self, dim: int, rank: int, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.rank = rank
        out_dim = dim + 2 * dim * rank
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[:dim].fill_(1.0)
            if rank > 0:
                last.bias[dim:].normal_(0.0, 1e-2)

    def forward(self, h: torch.Tensor, h_next: torch.Tensor, g_next: torch.Tensor) -> torch.Tensor:
        params = self.net(torch.cat([h, h_next], dim=-1))
        diag = params[:, : self.dim]
        out = diag * g_next
        if self.rank > 0:
            rest = params[:, self.dim :]
            u_raw, v_raw = rest.chunk(2, dim=-1)
            u = u_raw.view(-1, self.dim, self.rank)
            v = v_raw.view(-1, self.dim, self.rank)
            vg = torch.einsum("bdr,bd->br", v, g_next)
            out = out + torch.einsum("bdr,br->bd", u, vg)
        return out


def exact_vjp(block: nn.Module, h: torch.Tensor, g_next: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    h = h.detach().clone().requires_grad_(True)
    h_next = block(h)
    teacher = torch.autograd.grad(h_next, h, grad_outputs=g_next, retain_graph=False)[0]
    return h_next.detach(), teacher.detach()


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=-1, eps=1e-8).mean().detach().cpu())


def relerr_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-8)).mean().detach().cpu())


def norm_ratio_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-8)).mean().detach().cpu())


def evaluate_router(
    block: nn.Module,
    router: DiagLowRankRouter,
    *,
    dim: int,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> Dict[str, Any]:
    router.eval()
    cos_values = []
    rel_values = []
    norm_values = []
    id_cos = []
    id_rel = []
    id_norm = []
    lin_res = []

    with torch.no_grad():
        for _ in range(eval_batches):
            h0 = torch.randn(batch_size, dim, device=device)
            g1 = torch.randn(batch_size, dim, device=device)
            g2 = torch.randn(batch_size, dim, device=device)
            alpha = 0.7
            beta = -1.3

            # Leave no_grad temporarily for exact teacher construction.
            with torch.enable_grad():
                h_next, teacher = exact_vjp(block, h0, g1)
                h_next2, _ = exact_vjp(block, h0, g2)
            assert torch.allclose(h_next, h_next2, atol=1e-5)

            pred = router(h0, h_next, g1)
            cos_values.append(cosine_mean(pred, teacher))
            rel_values.append(relerr_mean(pred, teacher))
            norm_values.append(norm_ratio_mean(pred, teacher))
            id_cos.append(cosine_mean(g1, teacher))
            id_rel.append(relerr_mean(g1, teacher))
            id_norm.append(norm_ratio_mean(g1, teacher))

            pred_mix = router(h0, h_next, alpha * g1 + beta * g2)
            pred_sum = alpha * router(h0, h_next, g1) + beta * router(h0, h_next, g2)
            denom = alpha * router(h0, h_next, g1).norm(dim=-1) + abs(beta) * router(h0, h_next, g2).norm(dim=-1)
            lin_res.append(float(((pred_mix - pred_sum).norm(dim=-1) / denom.clamp_min(1e-8)).mean().cpu()))

    return {
        "router_cosine": float(np.mean(cos_values)),
        "router_relerr": float(np.mean(rel_values)),
        "router_norm_ratio": float(np.mean(norm_values)),
        "identity_cosine": float(np.mean(id_cos)),
        "identity_relerr": float(np.mean(id_rel)),
        "identity_norm_ratio": float(np.mean(id_norm)),
        "linearity_residual": float(np.mean(lin_res)),
    }


def perturb_block(block: nn.Module, noise_scale: float) -> nn.Module:
    stale_block = copy.deepcopy(block)
    with torch.no_grad():
        for p in stale_block.parameters():
            if p.numel() == 0:
                continue
            ref_scale = p.detach().float().std().clamp_min(1e-6)
            p.add_(noise_scale * ref_scale * torch.randn_like(p))
    return stale_block


def evaluate_router_staleness(
    block: nn.Module,
    router: DiagLowRankRouter,
    *,
    dim: int,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
    noise_levels: List[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for noise in noise_levels:
        stale_block = block if noise == 0 else perturb_block(block, noise).to(device)
        metrics = evaluate_router(
            stale_block,
            router,
            dim=dim,
            batch_size=batch_size,
            eval_batches=eval_batches,
            device=device,
        )
        rows.append({"noise_scale": noise, **metrics})
    return rows


def run_gate1(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    set_seed(args.seed)
    device = get_device(args.device)
    block = build_gate1_block(
        args.gate1_block,
        args.router_dim,
        args.block_scale,
        args.gate1_bottleneck,
    ).to(device)
    for p in block.parameters():
        p.requires_grad_(False)
    router = DiagLowRankRouter(args.router_dim, args.router_rank, hidden=args.router_hidden).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=args.router_lr, weight_decay=1e-5)

    losses: List[float] = []
    for _ in trange(args.router_steps, desc="gate1 router", leave=False):
        h = torch.randn(args.router_batch_size, args.router_dim, device=device)
        g_next = torch.randn(args.router_batch_size, args.router_dim, device=device)
        h_next, teacher = exact_vjp(block, h, g_next)
        pred = router(h, h_next, g_next)
        rel_mse = (pred - teacher).square().sum(dim=-1) / teacher.square().sum(dim=-1).clamp_min(1e-8)
        cos_loss = 1.0 - F.cosine_similarity(pred, teacher, dim=-1, eps=1e-8)
        norm_loss = (pred.norm(dim=-1) / teacher.norm(dim=-1).clamp_min(1e-8) - 1.0).square()
        loss = rel_mse.mean() + 0.2 * cos_loss.mean() + 0.1 * norm_loss.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    metrics = evaluate_router(
        block,
        router,
        dim=args.router_dim,
        batch_size=args.router_batch_size,
        eval_batches=args.router_eval_batches,
        device=device,
    )
    stale_noise_levels = parse_float_list(args.stale_noise_levels)
    stale_metrics = evaluate_router_staleness(
        block,
        router,
        dim=args.router_dim,
        batch_size=args.router_batch_size,
        eval_batches=args.router_eval_batches,
        device=device,
        noise_levels=stale_noise_levels,
    )
    status = "not_passed"
    if (
        metrics["router_cosine"] > 0.95
        and metrics["router_relerr"] < 0.2
        and 0.8 < metrics["router_norm_ratio"] < 1.2
        and metrics["linearity_residual"] < 0.05
    ):
        status = "medium"
    elif (
        metrics["router_cosine"] > 0.9
        and metrics["router_relerr"] < 0.35
        and 0.7 < metrics["router_norm_ratio"] < 1.3
    ):
        status = "weak"

    summary = {
        "gate": "gate1_single_block_vjp_router",
        "device": str(device),
        "seed": args.seed,
        "block_type": args.gate1_block,
        "block_scale": args.block_scale,
        "block_bottleneck": args.gate1_bottleneck if args.gate1_block == "bottleneck_residual" else None,
        "dim": args.router_dim,
        "rank": args.router_rank,
        "steps": args.router_steps,
        "final_train_loss": losses[-1],
        "train_loss_first": losses[0],
        "train_loss_last_20_mean": float(np.mean(losses[-20:])),
        "metrics": metrics,
        "stale_metrics": stale_metrics,
        "gate1_status": status,
    }
    save_json(out_dir / "gate1_vjp_router.json", summary)
    return summary


class RBFEdge1D(nn.Module):
    def __init__(self, n_basis: int, x_min: float, x_max: float, width_scale: float = 1.2):
        super().__init__()
        centers = torch.linspace(x_min, x_max, n_basis)
        spacing = float((x_max - x_min) / max(1, n_basis - 1))
        self.register_buffer("centers", centers)
        self.width = spacing * width_scale
        self.coeff = nn.Parameter(0.01 * torch.randn(n_basis))
        self.bias = nn.Parameter(torch.zeros(()))

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        z = (x - self.centers.view(1, -1)) / self.width
        b = torch.exp(-0.5 * z.square())
        return b / b.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.basis(x) @ self.coeff + self.bias

    def edge_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def target_function(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "sin":
        return torch.sin(x)
    if kind == "sin5":
        return torch.sin(x) + 0.3 * torch.sin(5.0 * x)
    if kind == "smooth_step":
        return torch.sin(x) + torch.sigmoid(10.0 * x)
    raise ValueError(f"unknown regression target: {kind}")


def sobolev_metric(
    model: RBFEdge1D,
    *,
    n_grid: int,
    alpha: float,
    beta: float,
    rho: float,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    grid = torch.linspace(-math.pi, math.pi, n_grid, device=device).requires_grad_(True)
    basis = model.basis(grid)
    cols = []
    dcols = []
    ddcols = []
    for j in range(basis.shape[1]):
        bj = basis[:, j]
        dbj = torch.autograd.grad(bj.sum(), grid, create_graph=True)[0]
        ddbj = torch.autograd.grad(dbj.sum(), grid, create_graph=True)[0]
        cols.append(bj.detach())
        dcols.append(dbj.detach())
        ddcols.append(ddbj.detach())
    b = torch.stack(cols, dim=1)
    db = torch.stack(dcols, dim=1)
    ddb = torch.stack(ddcols, dim=1)
    dx = float((2.0 * math.pi) / (n_grid - 1))
    metric = dx * (b.T @ b + alpha * (db.T @ db) + beta * (ddb.T @ ddb))
    metric = metric + rho * torch.eye(metric.shape[0], device=device)
    cond = float(torch.linalg.cond(metric).detach().cpu())
    return metric.detach(), cond


def function_shape_metrics(model: RBFEdge1D, device: torch.device, n_grid: int = 512) -> Dict[str, float]:
    grid = torch.linspace(-math.pi, math.pi, n_grid, device=device)
    with torch.no_grad():
        y = model.edge_values(grid)
        dx = float((2.0 * math.pi) / (n_grid - 1))
        dy = torch.gradient(y, spacing=dx)[0]
        ddy = torch.gradient(dy, spacing=dx)[0]
        return {
            "max_abs_slope": float(dy.abs().max().cpu()),
            "curvature_energy": float((ddy.square().sum() * dx).cpu()),
        }


def regression_data(seed: int, n_train: int, n_val: int, target: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x_train = (2.0 * math.pi) * torch.rand(n_train, generator=gen) - math.pi
    x_val = torch.linspace(-math.pi, math.pi, n_val)
    y_train = target_function(x_train, target)
    y_val = target_function(x_val, target)
    return x_train.to(device), y_train.to(device), x_val.to(device), y_val.to(device)


@dataclass
class KANTrainResult:
    method: str
    train_loss: float
    val_loss: float
    first_train_loss: float
    steps: int
    max_abs_slope: float
    curvature_energy: float
    descent_agreement_rate: Optional[float] = None
    metric_condition_number: Optional[float] = None


def train_kan_like(
    method: str,
    *,
    seed: int,
    n_basis: int,
    steps: int,
    target: str,
    device: torch.device,
    lr: float,
    alpha: float,
    beta: float,
    rho: float,
) -> KANTrainResult:
    set_seed(seed)
    x_train, y_train, x_val, y_val = regression_data(seed, 512, 512, target, device)
    model = RBFEdge1D(n_basis, -math.pi, math.pi).to(device)
    metric = None
    metric_cond = None
    opt: Optional[torch.optim.Optimizer] = None
    if method == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif method == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif method == "sobolev_precond":
        metric, metric_cond = sobolev_metric(
            model,
            n_grid=512,
            alpha=alpha,
            beta=beta,
            rho=rho,
            device=device,
        )
    else:
        raise ValueError(f"unknown KAN method: {method}")

    losses: List[float] = []
    descent_ok = 0
    descent_total = 0
    for _ in trange(steps, desc=f"gate2 {method}", leave=False):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        losses.append(float(loss.detach().cpu()))

        if method in {"adam", "sgd"}:
            assert opt is not None
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        else:
            assert metric is not None
            model.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                before = float(loss.detach().cpu())
                coeff_grad = model.coeff.grad.detach()
                bias_grad = model.bias.grad.detach()
                delta_coeff = -lr * torch.linalg.solve(metric, coeff_grad.view(-1, 1)).view(-1)
                delta_bias = -min(lr, 0.05) * bias_grad
                pred_descent = float((coeff_grad * delta_coeff).sum().detach().cpu() + (bias_grad * delta_bias).detach().cpu())
                model.coeff.add_(delta_coeff)
                model.bias.add_(delta_bias)
                after_loss = F.mse_loss(model(x_train), y_train)
                actual_descent = float((after_loss - before).detach().cpu())
                if pred_descent < 0:
                    descent_total += 1
                    if actual_descent < 0:
                        descent_ok += 1

    with torch.no_grad():
        train_loss = float(F.mse_loss(model(x_train), y_train).cpu())
        val_loss = float(F.mse_loss(model(x_val), y_val).cpu())
    shape = function_shape_metrics(model, device)
    return KANTrainResult(
        method=method,
        train_loss=train_loss,
        val_loss=val_loss,
        first_train_loss=losses[0],
        steps=steps,
        max_abs_slope=shape["max_abs_slope"],
        curvature_energy=shape["curvature_energy"],
        descent_agreement_rate=(descent_ok / descent_total if descent_total else None),
        metric_condition_number=metric_cond,
    )


def run_gate2(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    device = get_device(args.device)
    results = [
        train_kan_like(
            "sgd",
            seed=args.seed,
            n_basis=args.kan_basis,
            steps=args.kan_steps,
            target=args.kan_target,
            device=device,
            lr=args.kan_sgd_lr,
            alpha=args.sobolev_alpha,
            beta=args.sobolev_beta,
            rho=args.sobolev_rho,
        ),
        train_kan_like(
            "adam",
            seed=args.seed,
            n_basis=args.kan_basis,
            steps=args.kan_steps,
            target=args.kan_target,
            device=device,
            lr=args.kan_adam_lr,
            alpha=args.sobolev_alpha,
            beta=args.sobolev_beta,
            rho=args.sobolev_rho,
        ),
        train_kan_like(
            "sobolev_precond",
            seed=args.seed,
            n_basis=args.kan_basis,
            steps=args.kan_steps,
            target=args.kan_target,
            device=device,
            lr=args.kan_precond_lr,
            alpha=args.sobolev_alpha,
            beta=args.sobolev_beta,
            rho=args.sobolev_rho,
        ),
    ]
    rows = [asdict(r) for r in results]
    by_method = {r["method"]: r for r in rows}
    adam = by_method["adam"]
    pre = by_method["sobolev_precond"]
    status = "not_passed"
    if (
        pre["val_loss"] <= 1.05 * adam["val_loss"]
        and pre["max_abs_slope"] < adam["max_abs_slope"]
        and pre["curvature_energy"] < adam["curvature_energy"]
    ):
        status = "weak"
    if (
        pre["val_loss"] <= 0.95 * adam["val_loss"]
        or pre["descent_agreement_rate"] is not None
        and pre["descent_agreement_rate"] >= 0.7
        and pre["curvature_energy"] < adam["curvature_energy"]
    ):
        status = "medium_candidate"
    summary = {
        "gate": "gate2_kan_functional_update",
        "device": str(device),
        "seed": args.seed,
        "target": args.kan_target,
        "basis": args.kan_basis,
        "steps": args.kan_steps,
        "methods": rows,
        "gate2_status": status,
    }
    save_json(out_dir / "gate2_kan_functional_update.json", summary)
    return summary


def compact_report(results: Dict[str, Any]) -> str:
    lines = []
    if "gate0" in results:
        interp = results["gate0"]["interpretation"]
        lines.append(
            "Gate0 mean_E64={:.4f}, mean_PR={:.2f}, overlap16={:.4f}, status={}".format(
                interp["mean_E64_variance_energy"],
                interp["mean_participation_rank"],
                interp["mean_batch_subspace_overlap_top16"],
                interp["gate0_status"],
            )
        )
    if "gate1" in results:
        m = results["gate1"]["metrics"]
        lines.append(
            "Gate1 router cos={:.4f}, relerr={:.4f}, norm={:.4f}, lin={:.2e}, status={}".format(
                m["router_cosine"],
                m["router_relerr"],
                m["router_norm_ratio"],
                m["linearity_residual"],
                results["gate1"]["gate1_status"],
            )
        )
    if "gate2" in results:
        methods = {m["method"]: m for m in results["gate2"]["methods"]}
        adam = methods["adam"]
        pre = methods["sobolev_precond"]
        lines.append(
            "Gate2 Adam val={:.6f}, slope={:.3f}, curv={:.3f}; precond val={:.6f}, slope={:.3f}, curv={:.3f}, descent_agree={}, status={}".format(
                adam["val_loss"],
                adam["max_abs_slope"],
                adam["curvature_energy"],
                pre["val_loss"],
                pre["max_abs_slope"],
                pre["curvature_energy"],
                pre["descent_agreement_rate"],
                results["gate2"]["gate2_status"],
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", choices=["gate0", "gate1", "gate2", "all"], default="all")
    p.add_argument("--out-dir", type=Path, default=Path("results/dg_lca_toy"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--dataset", choices=["moons", "circles", "digits"], default="moons")
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--noise", type=float, default=0.12)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--audit-batches", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)

    p.add_argument("--router-dim", type=int, default=64)
    p.add_argument("--router-rank", type=int, default=16)
    p.add_argument("--router-hidden", type=int, default=128)
    p.add_argument("--router-steps", type=int, default=800)
    p.add_argument("--router-batch-size", type=int, default=256)
    p.add_argument("--router-eval-batches", type=int, default=8)
    p.add_argument("--router-lr", type=float, default=2e-3)
    p.add_argument("--block-scale", type=float, default=0.7)
    p.add_argument(
        "--gate1-block",
        choices=["residual", "mlp", "bottleneck_residual"],
        default="residual",
    )
    p.add_argument("--gate1-bottleneck", type=int, default=16)
    p.add_argument("--stale-noise-levels", default="0,0.01,0.03,0.1")

    p.add_argument("--kan-target", choices=["sin", "sin5", "smooth_step"], default="sin5")
    p.add_argument("--kan-basis", type=int, default=32)
    p.add_argument("--kan-steps", type=int, default=500)
    p.add_argument("--kan-sgd-lr", type=float, default=0.08)
    p.add_argument("--kan-adam-lr", type=float, default=0.03)
    p.add_argument("--kan-precond-lr", type=float, default=0.25)
    p.add_argument("--sobolev-alpha", type=float, default=1e-2)
    p.add_argument("--sobolev-beta", type=float, default=5e-4)
    p.add_argument("--sobolev-rho", type=float, default=1e-2)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_dir = ensure_dir(args.out_dir / now_tag())
    results: Dict[str, Any] = {
        "command": " ".join([os.path.basename(__file__)] + os.sys.argv[1:]),
        "run_dir": str(run_dir),
    }
    save_json(run_dir / "config.json", vars(args))

    if args.gate in {"gate0", "all"}:
        results["gate0"] = run_gate0(args, run_dir)
    if args.gate in {"gate1", "all"}:
        results["gate1"] = run_gate1(args, run_dir)
    if args.gate in {"gate2", "all"}:
        results["gate2"] = run_gate2(args, run_dir)

    save_json(run_dir / "summary.json", results)
    print(compact_report(results))
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
