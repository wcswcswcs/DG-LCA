"""Population-risk gates for Edge-Sobolev coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch


EPS = 1.0e-12


@dataclass
class GateResult:
    gate: torch.Tensor
    snr: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    summary: dict[str, float | int | str]


def _f(value: torch.Tensor | float, default: float = 0.0) -> float:
    try:
        out = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64)
    bb = b.detach().reshape(-1).to(dtype=torch.float64)
    n = min(int(aa.numel()), int(bb.numel()))
    if n <= 0:
        return 0.0
    aa = aa[:n]
    bb = bb[:n]
    denom = aa.norm().clamp_min(EPS) * bb.norm().clamp_min(EPS)
    return _f((aa @ bb) / denom)


def _quantile(x: torch.Tensor, q: float) -> float:
    if int(x.numel()) == 0:
        return 0.0
    return _f(torch.quantile(x.detach().reshape(-1).to(dtype=torch.float64), float(q)))


def gate_summary(gate: torch.Tensor, snr: torch.Tensor, prefix: str = "") -> dict[str, float | int | str]:
    gg = gate.detach().reshape(-1).to(dtype=torch.float64)
    ss = snr.detach().reshape(-1).to(dtype=torch.float64)
    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}gate_density_mean": _f(gg.mean()) if int(gg.numel()) else 0.0,
        f"{key}gate_density_median": _quantile(gg, 0.5),
        f"{key}gate_density_p10": _quantile(gg, 0.1),
        f"{key}gate_density_p90": _quantile(gg, 0.9),
        f"{key}snr_mean": _f(ss.mean()) if int(ss.numel()) else 0.0,
        f"{key}snr_top_decile": _quantile(ss, 0.9),
        f"{key}snr_max": _f(ss.max()) if int(ss.numel()) else 0.0,
    }


def diagonal_snr_gate(
    grads: torch.Tensor,
    *,
    beta: float = 2.0,
    threshold: float = 1.0,
    eps: float = 1.0e-8,
    use_log_snr: bool = True,
    variance_floor: float = 0.0,
) -> GateResult:
    """Compute the coordinate-wise population-risk SNR gate.

    ``grads`` is ``[examples, coordinates]`` and must already be in
    Edge-Sobolev whitened coordinates.
    """

    gg = grads.detach().to(dtype=torch.float64)
    if gg.ndim != 2 or int(gg.shape[0]) == 0:
        z = torch.zeros(0, dtype=torch.float64)
        return GateResult(z, z, z, z, {"gate_family": "diagonal", "per_example_count": 0})
    b = int(gg.shape[0])
    mu = gg.mean(dim=0)
    var = (gg - mu.unsqueeze(0)).square().mean(dim=0).clamp_min(float(variance_floor))
    snr = mu.square() / (var / max(1, b - 1) + float(eps))
    if use_log_snr:
        logits = float(beta) * (torch.log(snr + float(eps)) - math.log(float(threshold)))
    else:
        logits = float(beta) * (snr - float(threshold))
    gate = torch.sigmoid(logits)
    summary = {
        "gate_family": "diagonal",
        "snr_beta": float(beta),
        "snr_threshold": float(threshold),
        "use_log_snr": int(use_log_snr),
        "per_example_count": b,
        **gate_summary(gate, snr),
    }
    return GateResult(gate, snr, mu, var, summary)


def block_snr_gate(
    grads: torch.Tensor,
    blocks: list[list[int]],
    *,
    beta: float = 2.0,
    threshold: float = 1.0,
    eps: float = 1.0e-8,
) -> GateResult:
    gg = grads.detach().to(dtype=torch.float64)
    if gg.ndim != 2 or int(gg.shape[0]) == 0:
        z = torch.zeros(0, dtype=torch.float64)
        return GateResult(z, z, z, z, {"gate_family": "block", "block_count": 0})
    b, n = int(gg.shape[0]), int(gg.shape[1])
    gate = torch.zeros(n, dtype=torch.float64)
    snr = torch.zeros(n, dtype=torch.float64)
    mu = gg.mean(dim=0)
    var = (gg - mu.unsqueeze(0)).square().mean(dim=0)
    block_vals: list[float] = []
    clean_blocks = [sorted({int(i) for i in block if 0 <= int(i) < n}) for block in blocks]
    clean_blocks = [block for block in clean_blocks if block]
    for block in clean_blocks:
        idx = torch.tensor(block, dtype=torch.long)
        mu_b = mu[idx]
        var_b = var[idx]
        denom = var_b.sum() / max(1, b - 1) + float(eps)
        snr_b = mu_b.square().sum() / denom
        q_b = torch.sigmoid(float(beta) * (torch.log(snr_b + float(eps)) - math.log(float(threshold))))
        gate[idx] = q_b
        snr[idx] = snr_b
        block_vals.append(_f(snr_b))
    summary = {
        "gate_family": "block",
        "snr_beta": float(beta),
        "snr_threshold": float(threshold),
        "block_count": len(clean_blocks),
        "block_snr_top_decile": float(sorted(block_vals)[int(0.9 * (len(block_vals) - 1))]) if block_vals else 0.0,
        **gate_summary(gate, snr, prefix="block"),
    }
    return GateResult(gate, snr, mu, var, summary)


def lowrank_ab_gate(
    grads: torch.Tensor,
    *,
    rank: int = 4,
    tau: float = 1.0e-3,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor | dict[str, float | int | str]]:
    gg = grads.detach().to(dtype=torch.float64)
    if gg.ndim != 2 or int(gg.shape[0]) < 2:
        return {"operator": torch.zeros((0, 0), dtype=torch.float64), "summary": {"gate_family": "lowrank_ab", "positive_eigenvalue_count": 0}}
    b, n = int(gg.shape[0]), int(gg.shape[1])
    mu = gg.mean(dim=0)
    centered = gg - mu.unsqueeze(0)
    cov = centered.T @ centered / float(max(1, b))
    a_b = mu[:, None] @ mu[None, :] - cov / float(max(1, b - 1))
    a_b = 0.5 * (a_b + a_b.T)
    vals, vecs = torch.linalg.eigh(a_b)
    order = torch.argsort(vals, descending=True)
    vals = vals[order]
    vecs = vecs[:, order]
    pos = vals > float(eps)
    pos_count = int(pos.sum().item())
    r = min(max(1, int(rank)), pos_count, n)
    if r <= 0:
        op = torch.zeros((n, n), dtype=torch.float64)
    else:
        lam = vals[:r].clamp_min(0.0)
        scale = lam / (lam + float(tau))
        u = vecs[:, :r]
        op = (u * scale.reshape(1, -1)) @ u.T
    neg_mass = _f(vals[vals < 0.0].abs().sum()) if int(vals.numel()) else 0.0
    pos_mass = _f(vals[vals > 0.0].sum()) if int(vals.numel()) else 0.0
    summary = {
        "gate_family": "lowrank_ab",
        "lowrank_rank": int(r),
        "positive_eigenvalue_count": int(pos_count),
        "positive_eigenvalue_mass": pos_mass,
        "negative_eigenvalue_mass": neg_mass,
        "lowrank_top_eigenvalue": _f(vals[0]) if int(vals.numel()) else 0.0,
    }
    return {"operator": op, "summary": summary, "mean": mu, "variance": centered.square().mean(dim=0)}


def cohort_stability_audit(gates: Iterable[torch.Tensor]) -> dict[str, float | int]:
    vecs = [g.detach().reshape(-1).to(dtype=torch.float64) for g in gates if int(g.numel()) > 0]
    vals: list[float] = []
    for i, a in enumerate(vecs):
        for b in vecs[i + 1 :]:
            vals.append(_safe_cosine(a, b))
    densities = [_f(v.mean()) for v in vecs]
    return {
        "cohort_count": len(vecs),
        "cohort_stability": float(sum(vals) / len(vals)) if vals else 0.0,
        "cohort_gate_density_mean": float(sum(densities) / len(densities)) if densities else 0.0,
        "cohort_gate_density_std": float(torch.tensor(densities, dtype=torch.float64).std(unbiased=False).item()) if densities else 0.0,
    }


def population_risk_gate_smoke_test() -> dict[str, float | int]:
    same = torch.ones((8, 6), dtype=torch.float64) + 0.01 * torch.randn((8, 6), dtype=torch.float64)
    rnd = torch.randn((8, 6), dtype=torch.float64)
    g_same = diagonal_snr_gate(same, beta=2.0)
    g_rnd = diagonal_snr_gate(rnd, beta=2.0)
    blocks = [list(range(0, 3)), list(range(3, 6))]
    b_same = block_snr_gate(same, blocks, beta=2.0)
    return {
        "same_direction_gate_mean": _f(g_same.gate.mean()),
        "random_sign_gate_mean": _f(g_rnd.gate.mean()),
        "same_gt_random": int(_f(g_same.gate.mean()) > _f(g_rnd.gate.mean())),
        "block_gate_mean": _f(b_same.gate.mean()),
    }
