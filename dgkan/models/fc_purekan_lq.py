"""FC-PureKAN LinearLiftQuadraticEdgeBasis primitive.

This module owns the reusable model-side implementation for the v9.2.6+
FC-PureKAN LQ primitive.  Experiment runners should orchestrate protocols and
artifact writing; they should not carry the primitive math inline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from dgkan.training.manual_full_edge import ce_loss_and_grad


@dataclass(frozen=True)
class LQSpec:
    candidate_id: str
    basis: str
    hidden_dim: int
    init_variant: str = "default"
    output_scale: float = 1.0
    epochs: int = 20
    official: bool = True
    repair_hypothesis: str = "reference"


BASIS_CHANNELS = {"t2": 2, "t2t3": 3, "legendre23": 3}


def basis_from_lift(
    h: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    clip: float,
    out_div: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    raw = (h - mu) / std.clamp_min(1.0e-6)
    z = raw.clamp(-clip, clip) / out_div
    dzdh = (((raw >= -clip) & (raw <= clip)).to(h.dtype)) / (std.clamp_min(1.0e-6) * out_div)
    vals: List[torch.Tensor] = [h]
    ders: List[torch.Tensor] = [torch.ones_like(h)]
    if basis in {"t2", "t2t3"}:
        vals.append(2.0 * z.square() - 1.0)
        ders.append(4.0 * z * dzdh)
    if basis == "t2t3":
        vals.append(4.0 * z.pow(3) - 3.0 * z)
        ders.append((12.0 * z.square() - 3.0) * dzdh)
    if basis == "legendre23":
        vals.append(0.5 * (3.0 * z.square() - 1.0))
        ders.append(3.0 * z * dzdh)
        vals.append(0.5 * (5.0 * z.pow(3) - 3.0 * z))
        ders.append(0.5 * (15.0 * z.square() - 3.0) * dzdh)
    if basis not in BASIS_CHANNELS:
        raise ValueError(f"unknown LQ basis {basis}")
    return vals, ders


def lift_basis_forward(
    x: torch.Tensor,
    A: torch.Tensor,
    weights: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    h = x @ A
    vals, _ders = basis_from_lift(h, mu, std, basis, clip, out_div)
    y = vals[0] @ weights[0]
    for v, w in zip(vals[1:], weights[1:]):
        y = y + v @ w
    return y


def lift_basis_fwd_bwd(
    x: torch.Tensor,
    labels: torch.Tensor,
    A: torch.Tensor,
    *rest: torch.Tensor,
    basis: str,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, ...]:
    mu = rest[-2]
    std = rest[-1]
    weights = list(rest[:-2])
    h = x @ A
    vals, ders = basis_from_lift(h, mu, std, basis, clip, out_div)
    logits = vals[0] @ weights[0]
    for v, w in zip(vals[1:], weights[1:]):
        logits = logits + v @ w
    loss, dy = ce_loss_and_grad(logits, labels)
    dweights = [v.T @ dy for v in vals]
    dh = torch.zeros_like(h)
    for der, w in zip(ders, weights):
        dh = dh + (dy @ w.T) * der
    dA = x.T @ dh
    return (loss, dA, *dweights)


def lift_basis_fwd_bwd_t2(
    x: torch.Tensor,
    labels: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return lift_basis_fwd_bwd(x, labels, A, W0, W2, mu, std, basis="t2", clip=clip, out_div=out_div)  # type: ignore[return-value]


def lift_basis_forward_t2(
    x: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    return lift_basis_forward(x, A, [W0, W2], mu, std, "t2", clip, out_div)


def lift_basis_fwd_bwd_t2t3(
    x: torch.Tensor,
    labels: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return lift_basis_fwd_bwd(x, labels, A, W0, W2, W3, mu, std, basis="t2t3", clip=clip, out_div=out_div)  # type: ignore[return-value]


def lift_basis_forward_t2t3(
    x: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    return lift_basis_forward(x, A, [W0, W2, W3], mu, std, "t2t3", clip, out_div)


def lift_basis_fwd_bwd_legendre23(
    x: torch.Tensor,
    labels: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return lift_basis_fwd_bwd(x, labels, A, W0, W2, W3, mu, std, basis="legendre23", clip=clip, out_div=out_div)  # type: ignore[return-value]


def lift_basis_forward_legendre23(
    x: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    return lift_basis_forward(x, A, [W0, W2, W3], mu, std, "legendre23", clip, out_div)


def functions_for_basis(basis: str):
    if basis == "t2":
        return lift_basis_forward_t2, lift_basis_fwd_bwd_t2
    if basis == "t2t3":
        return lift_basis_forward_t2t3, lift_basis_fwd_bwd_t2t3
    if basis == "legendre23":
        return lift_basis_forward_legendre23, lift_basis_fwd_bwd_legendre23
    raise ValueError(f"unknown LQ basis {basis}")


def init_lq_params(
    input_dim: int,
    output_dim: int,
    spec: LQSpec,
    x_for_stats: torch.Tensor,
    device: torch.device,
    seed: int,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(int(seed))
    n_basis = BASIS_CHANNELS[spec.basis]
    if spec.init_variant == "orthogonal_lift":
        raw = torch.randn(input_dim, spec.hidden_dim, device=device, generator=gen)
        q, _ = torch.linalg.qr(raw, mode="reduced")
        A = q[:, : spec.hidden_dim].contiguous()
    else:
        A = torch.randn(input_dim, spec.hidden_dim, device=device, generator=gen) / math.sqrt(input_dim)
    weights = [
        torch.randn(spec.hidden_dim, output_dim, device=device, generator=gen) * (float(spec.output_scale) / math.sqrt(spec.hidden_dim))
        for _ in range(n_basis)
    ]
    h_stats = x_for_stats[: min(2048, int(x_for_stats.shape[0]))] @ A
    mu = h_stats.mean(dim=0)
    std = h_stats.std(dim=0).clamp_min(1.0e-3)
    return [A, *weights], mu, std


def predict_lq_logits(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    x: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    fwd_core, _ = functions_for_basis(basis)
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            parts.append(fwd_core(x[start : start + int(batch_size)], *params, mu, std, 2.0, 2.0))
    return torch.cat(parts, dim=0)


def basis_condition_metrics(h: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, basis: str) -> Dict[str, float]:
    vals, _ = basis_from_lift(h, mu, std, basis, 2.0, 2.0)
    feat = torch.stack([v.reshape(-1) for v in vals], dim=1)
    feat = feat - feat.mean(dim=0, keepdim=True)
    cov = feat.T @ feat / max(1, feat.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov.float()).clamp_min(1.0e-12)
    energy = eig / eig.sum().clamp_min(1.0e-12)
    entropy = float((-(energy * energy.log()).sum() / math.log(max(2, int(eig.numel())))).detach().cpu())
    return {
        "basis_condition_number": float((eig.max() / eig.min()).detach().cpu()),
        "basis_usage_entropy": entropy,
        "dominant_basis_fraction": float(energy.max().detach().cpu()),
    }


def lift_feature_metrics(h: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, basis: str) -> Dict[str, float]:
    hc = h - h.mean(dim=0, keepdim=True)
    cov = hc.T @ hc / max(1, h.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov.float()).clamp_min(1.0e-12)
    energy = eig / eig.sum().clamp_min(1.0e-12)
    entropy = -(energy * energy.log()).sum()
    effective_rank = float(torch.exp(entropy).detach().cpu())
    dominant_dim_fraction = float(energy.max().detach().cpu())
    dead_dim_fraction = float((h.std(dim=0) < 1.0e-6).float().mean().detach().cpu())
    out = {
        "lift_feature_mean": float(h.mean().detach().cpu()),
        "lift_feature_std": float(h.std().detach().cpu()),
        "lift_condition_number": float((eig.max() / eig.min()).detach().cpu()),
        "lift_effective_rank": effective_rank,
        "dead_lift_dim_fraction": dead_dim_fraction,
        "dominant_lift_dim_fraction": dominant_dim_fraction,
    }
    out.update(basis_condition_metrics(h[: min(512, int(h.shape[0]))], mu, std, basis))
    return out
