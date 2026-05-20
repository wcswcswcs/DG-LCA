"""Good Geometry Battery metrics for the v12 LQ mainline.

The helpers in this file are measurement-only.  They do not own protocol
decisions and they do not mutate models.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from dgkan.models import fc_purekan_lq as lq


EPS = 1.0e-12


def safe_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


def compute_effective_rank(z: torch.Tensor, eps: float = EPS) -> float:
    """Entropy effective rank from singular values.

    Constant nonzero tensors return rank near 1; empty tensors return 0.
    """

    if z.numel() == 0:
        return 0.0
    m = z.detach().float()
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    else:
        m = m.reshape(int(m.shape[0]), -1)
    if int(m.shape[0]) <= 1:
        return 1.0
    m = m - m.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(m)
    except RuntimeError:
        cov = m.T @ m / max(1, int(m.shape[0]) - 1)
        s = torch.linalg.eigvalsh(cov).clamp_min(0.0).sqrt()
    mass = s / s.sum().clamp_min(eps)
    entropy = -(mass * (mass + eps).log()).sum()
    return safe_float(torch.exp(entropy))


def coefficient_total_variation(coeff: torch.Tensor) -> torch.Tensor:
    if coeff.shape[-1] < 2:
        return torch.zeros((), device=coeff.device, dtype=coeff.dtype)
    return torch.diff(coeff, dim=-1).abs().mean()


def coefficient_second_difference_energy(coeff: torch.Tensor) -> torch.Tensor:
    if coeff.shape[-1] < 3:
        return torch.zeros((), device=coeff.device, dtype=coeff.dtype)
    diff2 = coeff[..., 2:] - 2.0 * coeff[..., 1:-1] + coeff[..., :-2]
    return diff2.square().mean()


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    ece = torch.zeros((), device=logits.device)
    for idx in range(int(bins)):
        lo = idx / int(bins)
        hi = (idx + 1) / int(bins)
        mask = (conf > lo) & (conf <= hi)
        if bool(mask.any()):
            ece = ece + mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return safe_float(ece)


def compute_tail_metrics(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> Dict[str, Any]:
    log_probs = logits.log_softmax(dim=1)
    ce = -log_probs[torch.arange(labels.numel(), device=labels.device), labels]
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    true_logits = logits[torch.arange(labels.numel(), device=labels.device), labels]
    masked = logits.clone()
    masked[torch.arange(labels.numel(), device=labels.device), labels] = -torch.inf
    top_wrong = masked.max(dim=1).values
    margin = true_logits - top_wrong
    onehot = F.one_hot(labels, num_classes=logits.shape[1]).to(probs)
    brier = (probs - onehot).square().sum(dim=1).mean()
    wrong_conf = conf[pred != labels]
    classwise: List[Dict[str, Any]] = []
    for cls in range(int(logits.shape[1])):
        mask = labels == cls
        if bool(mask.any()):
            classwise.append(
                {
                    "class_id": cls,
                    "count": int(mask.sum().detach().cpu()),
                    "CE_p99": safe_float(torch.quantile(ce[mask], 0.99)),
                    "acc": safe_float((pred[mask] == labels[mask]).float().mean()),
                }
            )
    return {
        "CE_mean": safe_float(ce.mean()),
        "CE_p90": safe_float(torch.quantile(ce, 0.90)),
        "CE_p95": safe_float(torch.quantile(ce, 0.95)),
        "CE_p99": safe_float(torch.quantile(ce, 0.99)),
        "margin_mean": safe_float(margin.mean()),
        "margin_p10": safe_float(torch.quantile(margin, 0.10)),
        "wrong_confidence_p95": safe_float(torch.quantile(wrong_conf, 0.95)) if bool(wrong_conf.numel()) else 0.0,
        "ECE": compute_ece(logits, labels, bins=bins),
        "NLL": safe_float(F.cross_entropy(logits, labels)),
        "Brier": safe_float(brier),
        "hard_tail_class_distribution": classwise,
    }


def compute_input_perturbation_drift(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    eps_list: Sequence[float] = (1.0e-3, 3.0e-3),
    seed: int = 0,
) -> Dict[str, float]:
    if int(x.shape[0]) == 0:
        return {
            "perturb_logit_drift_mean": 0.0,
            "perturb_logit_drift_p95": 0.0,
            "perturb_prediction_flip_rate": 0.0,
            "local_jacobian_norm_median": 0.0,
            "local_jacobian_norm_p95": 0.0,
        }
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    with torch.no_grad():
        base = forward_fn(x)
        base_pred = base.argmax(dim=1)
        drifts: List[torch.Tensor] = []
        local_norms: List[torch.Tensor] = []
        flips: List[torch.Tensor] = []
        for eps in eps_list:
            noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
            x2 = x + noise * float(eps)
            out = forward_fn(x2)
            diff = (out - base).reshape(int(x.shape[0]), -1).norm(dim=1)
            denom = (noise.reshape(int(x.shape[0]), -1).norm(dim=1) * float(eps)).clamp_min(EPS)
            drifts.append(diff)
            local_norms.append(diff / denom)
            flips.append((out.argmax(dim=1) != base_pred).float())
        drift = torch.cat(drifts)
        local = torch.cat(local_norms)
        flip = torch.cat(flips)
    return {
        "perturb_logit_drift_mean": safe_float(drift.mean()),
        "perturb_logit_drift_p95": safe_float(torch.quantile(drift, 0.95)),
        "perturb_prediction_flip_rate": safe_float(flip.mean()),
        "local_jacobian_norm_median": safe_float(torch.quantile(local, 0.50)),
        "local_jacobian_norm_p95": safe_float(torch.quantile(local, 0.95)),
    }


def compute_lq_channel_stats(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    x: torch.Tensor,
) -> Dict[str, float]:
    with torch.no_grad():
        h = x @ params[0]
        vals, _ = lq.basis_from_lift(h, mu, std, basis, 2.0, 2.0)
        logits = vals[0] @ params[1]
        for v, w in zip(vals[1:], params[2:]):
            logits = logits + v @ w
        lift_stats = lq.lift_feature_metrics(h, mu, std, basis)
        basis_energy = torch.stack([v.float().square().mean().sqrt() for v in vals])
        p = basis_energy / basis_energy.sum().clamp_min(EPS)
        occ_entropy = -(p * (p + EPS).log()).sum() / math.log(max(2, int(p.numel())))
        channel_norms = torch.tensor([safe_float(w.norm()) for w in params[1:]], device=x.device)
        total_channel = channel_norms.sum().clamp_min(EPS)
    out = {
        "effective_rank_input": compute_effective_rank(x),
        "effective_rank_lift": compute_effective_rank(h),
        "effective_rank_hidden": compute_effective_rank(h),
        "effective_rank_logits": compute_effective_rank(logits),
        "basis_usage_entropy": safe_float(occ_entropy),
        "dead_basis_fraction": safe_float((basis_energy < 1.0e-8).float().mean()),
        "quadratic_channel_norm": safe_float(params[2].norm()) if len(params) > 2 else 0.0,
        "linear_channel_norm": safe_float(params[1].norm()) if len(params) > 1 else 0.0,
        "lift_norm": safe_float(params[0].norm()),
        "output_linear_norm": safe_float(params[1].norm()) if len(params) > 1 else 0.0,
        "quadratic_coeff_norm": safe_float(params[2].norm()) if len(params) > 2 else 0.0,
        "channel_norm_ratio": safe_float(channel_norms.max() / total_channel) if bool(channel_norms.numel()) else 0.0,
    }
    out.update({k: float(v) for k, v in lift_stats.items()})
    out["lift_condition_proxy"] = out.get("lift_condition_number", 0.0)
    return out


def compute_generic_representation_stats(
    x: torch.Tensor,
    hidden: torch.Tensor,
    logits: torch.Tensor,
) -> Dict[str, float]:
    return {
        "effective_rank_input": compute_effective_rank(x),
        "effective_rank_lift": 0.0,
        "effective_rank_hidden": compute_effective_rank(hidden),
        "effective_rank_logits": compute_effective_rank(logits),
        "lift_condition_proxy": 0.0,
        "basis_usage_entropy": 0.0,
        "dead_basis_fraction": 0.0,
        "quadratic_channel_norm": 0.0,
        "linear_channel_norm": 0.0,
        "lift_norm": 0.0,
        "output_linear_norm": 0.0,
        "quadratic_coeff_norm": 0.0,
        "channel_norm_ratio": 0.0,
        "lift_feature_mean": 0.0,
        "lift_feature_std": 0.0,
        "lift_condition_number": 0.0,
        "lift_effective_rank": 0.0,
        "dead_lift_dim_fraction": 0.0,
        "dominant_lift_dim_fraction": 0.0,
    }


def geometry_debt_lq(params: Sequence[torch.Tensor], basis: str) -> Dict[str, float]:
    debt = torch.zeros((), device=params[0].device, dtype=params[0].dtype)
    tv = torch.zeros_like(debt)
    for p in params[1:]:
        if p.ndim >= 2:
            debt = debt + coefficient_second_difference_energy(p.T)
            tv = tv + coefficient_total_variation(p.T)
    if len(params) > 2:
        quad_norm = params[2].float().norm()
    else:
        quad_norm = torch.zeros((), device=params[0].device)
    return {
        "curvature_debt": safe_float(debt),
        "coefficient_total_variation": safe_float(tv),
        "rolewise_quadratic_coeff_norm": safe_float(quad_norm),
        "basis_type_has_spline_grid": 0.0,
    }


def signal_consistency_from_grads(grads_by_microbatch: Sequence[Sequence[torch.Tensor]]) -> float:
    if not grads_by_microbatch:
        return 0.0
    flats = [torch.cat([g.detach().float().reshape(-1) for g in pack]) for pack in grads_by_microbatch]
    stack = torch.stack(flats, dim=0)
    mean = stack.mean(dim=0)
    var = (stack - mean).square().sum(dim=1).mean()
    return safe_float(mean.square().sum() / var.clamp_min(EPS))


def rolewise_snr_from_micrograds(
    grads_by_microbatch: Sequence[Sequence[torch.Tensor]],
    roles: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    positives = []
    means = []
    for idx, role in enumerate(roles):
        stack = torch.stack([pack[idx].detach().float().reshape(-1) for pack in grads_by_microbatch], dim=0)
        mu = stack.mean(dim=0)
        var = stack.var(dim=0, unbiased=stack.shape[0] > 1)
        snr_margin = mu.square() - var / max(1, int(stack.shape[0]) - 1)
        pos = (snr_margin > 0).float()
        out[f"snr_positive_fraction_{role}"] = safe_float(pos.mean())
        out[f"snr_mean_{role}"] = safe_float(snr_margin.mean())
        out[f"snr_p10_{role}"] = safe_float(torch.quantile(snr_margin, 0.10))
        out[f"snr_p90_{role}"] = safe_float(torch.quantile(snr_margin, 0.90))
        positives.append(pos.mean())
        means.append(snr_margin.mean())
    out["snr_positive_fraction_global"] = safe_float(torch.stack(positives).mean()) if positives else 0.0
    out["snr_mean_global"] = safe_float(torch.stack(means).mean()) if means else 0.0
    return out


def kernel_condition_from_output_responses(responses: torch.Tensor) -> Dict[str, float]:
    if responses.numel() == 0:
        return {
            "kernel_condition_proxy": 0.0,
            "kernel_top_eigen_share_proxy": 0.0,
            "kernel_effective_rank_proxy": 0.0,
            "update_to_output_ratio_median": 0.0,
            "update_to_output_ratio_p95": 0.0,
            "gradient_spike_p95": 0.0,
        }
    r = responses.detach().float().reshape(int(responses.shape[0]), -1)
    k = r @ r.T / max(1, int(r.shape[1]))
    eig = torch.linalg.eigvalsh(k).clamp_min(EPS)
    energy = eig / eig.sum().clamp_min(EPS)
    ratios = r.norm(dim=1)
    return {
        "kernel_condition_proxy": safe_float(torch.quantile(eig, 0.90) / torch.quantile(eig, 0.10).clamp_min(EPS)),
        "kernel_top_eigen_share_proxy": safe_float(eig.max() / eig.sum().clamp_min(EPS)),
        "kernel_effective_rank_proxy": safe_float(torch.exp(-(energy * (energy + EPS).log()).sum())),
        "update_to_output_ratio_median": safe_float(torch.quantile(ratios, 0.50)),
        "update_to_output_ratio_p95": safe_float(torch.quantile(ratios, 0.95)),
        "gradient_spike_p95": safe_float(torch.quantile(ratios, 0.95) / torch.quantile(ratios, 0.50).clamp_min(EPS)),
    }


def finite_difference_output_responses(
    params: Sequence[torch.Tensor],
    forward_from_params: Callable[[Sequence[torch.Tensor], torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    directions: int = 6,
    eps: float = 1.0e-4,
    seed: int = 0,
) -> torch.Tensor:
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    with torch.no_grad():
        base = forward_from_params(params, x)
        rows: List[torch.Tensor] = []
        for _ in range(int(directions)):
            delta = [torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) for p in params]
            norm = torch.sqrt(sum(d.float().square().sum() for d in delta)).clamp_min(EPS)
            scaled = [d * (float(eps) / norm) for d in delta]
            moved = [p + d for p, d in zip(params, scaled)]
            out = forward_from_params(moved, x)
            rows.append(((out - base) / float(eps)).reshape(-1))
    return torch.stack(rows, dim=0) if rows else torch.empty(0, device=x.device)
