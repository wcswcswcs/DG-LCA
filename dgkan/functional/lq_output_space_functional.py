"""Output-space functional directions for FC-PureKAN LQ.

Experiment runners own protocols, gates, and artifact writing.  This module
owns reusable output-target construction and parameter-subspace VJP helpers for
v9.2.12 style functional direction reconstruction.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from dgkan.functional import snr_gated_lq as snr_lq
from dgkan.models import fc_purekan_lq as lq


OUTPUT_TARGET_IDS = [
    "O1-HardTailLogitCorrection",
    "O2-MarginTailExpansion",
    "O3-CalibrationTailCompression",
    "O4-CurvatureOutputFlattening",
    "O6-KMNISTHardModeOutputTarget",
]

SUBSPACE_IDS = [
    "S1-QuadraticCoeffSubspace",
    "S2-LiftSubspace",
    "S3-OutputLinearSubspace",
    "S4-LiftPlusQuadraticSubspace",
    "S5-RecentSignalSubspace",
    "S6-OrthogonalToAdamWSubspace",
]

SOLVER_IDS = [
    "SOL0-ProjectedGradient",
    "SOL2-ConstrainedTaskSafe",
    "SOL3-TrustRegionTaskSafe",
]


def candidate_id(target_id: str, subspace_id: str, solver_id: str) -> str:
    return f"{target_id}|{subspace_id}|{solver_id}"


def _true_wrong_logits(logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(y.numel())
    row = torch.arange(n, device=logits.device)
    true = logits[row, y]
    masked = logits.clone()
    masked[row, y] = -torch.inf
    wrong, wrong_idx = masked.max(dim=1)
    pred = logits.argmax(dim=1)
    return true, wrong, wrong_idx, pred


def build_output_target(
    target_id: str,
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    dataset: str,
    strength: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Construct a desired logit displacement target.

    The targets are diagnostic output-space directions.  They do not alter the
    CE loss, sampler, labels, or optimization objective; runners only use them
    to build a one-step functional update candidate.
    """

    log_probs = logits.log_softmax(dim=1)
    probs = logits.softmax(dim=1)
    n, c = logits.shape
    row = torch.arange(n, device=logits.device)
    true, wrong, wrong_idx, pred = _true_wrong_logits(logits, y)
    per_ce = -log_probs[row, y]
    margin = true - wrong
    conf, _ = probs.max(dim=1)
    wrong_conf = torch.where(pred != y, conf, torch.zeros_like(conf))
    centered = logits - logits.mean(dim=1, keepdim=True)

    target = torch.zeros_like(logits)
    selected = torch.zeros(n, device=logits.device, dtype=torch.bool)
    tid = str(target_id)
    if tid == "O1-HardTailLogitCorrection":
        tau = torch.quantile(per_ce.float(), 0.90)
        selected = per_ce >= tau
        amp = (per_ce / per_ce.detach().mean().clamp_min(1.0e-6)).clamp(0.5, 4.0)
        target[row, y] = amp
        target[row, wrong_idx] = target[row, wrong_idx] - amp
    elif tid == "O2-MarginTailExpansion":
        tau = torch.quantile(margin.float(), 0.20)
        selected = margin <= tau
        amp = ((tau - margin).relu() / (margin.detach().abs().mean().clamp_min(1.0e-6))).clamp(0.25, 4.0)
        target[row, y] = amp
        target[row, wrong_idx] = target[row, wrong_idx] - amp
    elif tid == "O3-CalibrationTailCompression":
        tau = torch.quantile(wrong_conf.float(), 0.90)
        selected = (pred != y) & (wrong_conf >= tau)
        amp = wrong_conf.clamp(0.1, 1.0)
        target[row, pred] = target[row, pred] - amp
        target[row, y] = target[row, y] + 0.25 * amp
    elif tid == "O4-CurvatureOutputFlattening":
        norm = centered.norm(dim=1)
        tau = torch.quantile(norm.float(), 0.90)
        selected = norm >= tau
        target = -centered / centered.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    elif tid == "O6-KMNISTHardModeOutputTarget":
        if str(dataset) == "KMNIST":
            tau = torch.quantile(margin.float(), 0.30)
            selected = margin <= tau
            amp = ((tau - margin).relu() / margin.detach().abs().mean().clamp_min(1.0e-6)).clamp(0.25, 5.0)
            target[row, y] = amp
            target[row, wrong_idx] = target[row, wrong_idx] - amp
        else:
            selected = torch.zeros(n, device=logits.device, dtype=torch.bool)
    else:
        raise ValueError(f"unknown output target {target_id}")

    target = target * selected.float().unsqueeze(1) * float(strength)
    target = target - target.mean(dim=1, keepdim=True)
    target_norm = target.float().norm()
    if bool(target_norm.detach().cpu() > 0):
        target = target / target_norm * (float(selected.float().mean().detach().cpu()) + 1.0e-6)
    return target.detach(), {
        "target_selected_fraction": float(selected.float().mean().detach().cpu()),
        "target_norm_raw": float(target_norm.detach().cpu()),
        "target_mean_ce": float(per_ce[selected].mean().detach().cpu()) if bool(selected.any()) else 0.0,
        "target_mean_margin": float(margin[selected].mean().detach().cpu()) if bool(selected.any()) else 0.0,
        "target_wrong_confidence_mean": float(wrong_conf[selected].mean().detach().cpu()) if bool(selected.any()) else 0.0,
    }


def output_vjp_direction(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    x: torch.Tensor,
    target_delta_logits: torch.Tensor,
) -> List[torch.Tensor]:
    """Return J^T target_delta_logits for the LQ logits map."""

    h = x @ params[0]
    vals, ders = lq.basis_from_lift(h, mu, std, basis, 2.0, 2.0)
    scale = 1.0 / max(1, int(x.shape[0]))
    dy = target_delta_logits.detach() * scale
    dweights = [v.T @ dy for v in vals]
    dh = torch.zeros_like(h)
    for der, weight in zip(ders, params[1:]):
        dh = dh + (dy @ weight.T) * der
    d_a = x.T @ dh
    return [d_a, *dweights]


def apply_subspace(
    direction: Sequence[torch.Tensor],
    params: Sequence[torch.Tensor],
    task_grads: Sequence[torch.Tensor],
    subspace_id: str,
) -> List[torch.Tensor]:
    out = snr_lq.zero_like_params(params)
    sid = str(subspace_id)
    if sid == "S1-QuadraticCoeffSubspace":
        if len(out) >= 3:
            out[2] = direction[2].detach()
    elif sid == "S2-LiftSubspace":
        out[0] = direction[0].detach()
    elif sid == "S3-OutputLinearSubspace":
        if len(out) >= 2:
            out[1] = direction[1].detach()
    elif sid == "S4-LiftPlusQuadraticSubspace":
        out[0] = direction[0].detach()
        if len(out) >= 3:
            out[2] = direction[2].detach()
    elif sid == "S5-RecentSignalSubspace":
        denom = snr_lq.step_dot(task_grads, task_grads).clamp_min(1.0e-12)
        coeff = snr_lq.step_dot(direction, task_grads) / denom
        out = snr_lq.scale_step(task_grads, coeff)
    elif sid == "S6-OrthogonalToAdamWSubspace":
        denom = snr_lq.step_dot(task_grads, task_grads).clamp_min(1.0e-12)
        coeff = snr_lq.step_dot(direction, task_grads) / denom
        parallel = snr_lq.scale_step(task_grads, coeff)
        out = [d.detach() - p.detach() for d, p in zip(direction, parallel)]
    else:
        raise ValueError(f"unknown subspace_id {subspace_id}")
    return out


def solve_functional_step(
    *,
    params: Sequence[torch.Tensor],
    task_grads: Sequence[torch.Tensor],
    task_step: Sequence[torch.Tensor],
    raw_direction: Sequence[torch.Tensor],
    solver_id: str,
    step_fraction: float,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    raw_norm = snr_lq.step_norm(raw_direction)
    scaled = snr_lq.scale_direction_to_fraction_of_task_step(raw_direction, task_step, float(step_fraction))
    if str(solver_id) in {"SOL0-ProjectedGradient", "SOL2-ConstrainedTaskSafe", "SOL3-TrustRegionTaskSafe"}:
        projected, removed_norm = snr_lq.project_step_to_task_safe(scaled, task_grads)
    else:
        raise ValueError(f"unknown solver_id {solver_id}")
    if str(solver_id) == "SOL3-TrustRegionTaskSafe":
        trust = snr_lq.step_norm(task_step) * float(step_fraction)
        pnorm = snr_lq.step_norm(projected)
        if bool((pnorm > trust).detach().cpu()):
            projected = snr_lq.scale_step(projected, trust / pnorm.clamp_min(1.0e-12))
    solved_norm = snr_lq.step_norm(projected)
    gdot = snr_lq.step_dot(task_grads, projected)
    cos_task = gdot / (snr_lq.step_norm(task_grads).clamp_min(1.0e-12) * solved_norm.clamp_min(1.0e-12))
    return projected, {
        "raw_direction_norm": float(raw_norm.detach().cpu()),
        "solved_delta_norm": float(solved_norm.detach().cpu()),
        "projection_removed_norm": float(removed_norm.detach().cpu()),
        "norm_after_projection_ratio": float((solved_norm / snr_lq.step_norm(scaled).clamp_min(1.0e-12)).detach().cpu()),
        "cos_with_task_gradient": float(cos_task.detach().cpu()) if bool((solved_norm > 0).detach().cpu()) else 0.0,
    }


def output_fit_metrics(
    *,
    params: Sequence[torch.Tensor],
    delta: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    x: torch.Tensor,
    target_delta_logits: torch.Tensor,
) -> Dict[str, float]:
    fwd, _bwd = lq.functions_for_basis(basis)
    with torch.no_grad():
        before = fwd(x, *params, mu, std, 2.0, 2.0)
        after_params = [p.detach() + d.detach() for p, d in zip(params, delta)]
        after = fwd(x, *after_params, mu, std, 2.0, 2.0)
        actual = after - before
        target = target_delta_logits.detach()
        sse = (actual.float() - target.float()).square().sum()
        centered = target.float() - target.float().mean()
        sst = centered.square().sum().clamp_min(1.0e-12)
        r2 = 1.0 - sse / sst
        ratio = actual.float().norm() / target.float().norm().clamp_min(1.0e-12)
    return {
        "output_target_fit_r2": float(r2.detach().cpu()),
        "output_displacement_norm": float(actual.float().norm().detach().cpu()),
        "target_norm": float(target.float().norm().detach().cpu()),
        "output_displacement_to_target_ratio": float(ratio.detach().cpu()),
    }

