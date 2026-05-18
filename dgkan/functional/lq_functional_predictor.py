"""Functional predictor/direction helpers for FC-PureKAN LQ.

Experiment runners own protocol decisions and artifact writing.  This module
owns reusable direction construction and lightweight mechanism metrics for
functional-update audits.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from dgkan.functional import snr_gated_lq as snr_lq
from dgkan.models import fc_purekan_lq as lq


DIRECTION_IDS = [
    "D1-QuadraticCoeffDamping",
    "D2-QuadraticCoeffSignFlip",
    "D3-LiftDamping",
    "D4-OutputLinearDamping",
    "D5-AllOutputCoeffDamping",
    "D6-AllParamNormDamping",
    "D7-AdamWResidualDiagnostic",
    "D8-FisherDiagQuadraticDamping",
    "D9-SignalChannelProjection",
    "D10-OrthogonalSignalGeometry",
    "D11-SignalSubspaceCurvature",
    "D12-SignalSubspaceMarginTail",
    "D13-LiftConditionCorrection",
    "D14-QuadraticBasisEntropyCorrection",
    "D15-OutputScaleTailCorrection",
    "D16-KMNISTHardModeGeometry",
]


def _project_onto_grads(direction: Sequence[torch.Tensor], grads: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    denom = snr_lq.step_dot(grads, grads).clamp_min(1.0e-12)
    coeff = snr_lq.step_dot(direction, grads) / denom
    return snr_lq.scale_step(grads, coeff)


def _remove_grad_parallel(direction: Sequence[torch.Tensor], grads: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    parallel = _project_onto_grads(direction, grads)
    return [d.detach() - p.detach() for d, p in zip(direction, parallel)]


def direction_for_id(
    direction_id: str,
    params: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor],
    basis: str,
) -> List[torch.Tensor]:
    out = snr_lq.zero_like_params(params)
    if direction_id == "D1-QuadraticCoeffDamping":
        return snr_lq.quadratic_coeff_direction(params, basis)
    if direction_id == "D2-QuadraticCoeffSignFlip":
        base = snr_lq.quadratic_coeff_direction(params, basis)
        return [-x for x in base]
    if direction_id == "D3-LiftDamping":
        out[0] = -params[0].detach()
        return out
    if direction_id == "D4-OutputLinearDamping":
        if len(out) >= 2:
            out[1] = -params[1].detach()
        return out
    if direction_id == "D5-AllOutputCoeffDamping":
        for idx in range(1, len(out)):
            out[idx] = -params[idx].detach()
        return out
    if direction_id == "D6-AllParamNormDamping":
        return [-p.detach() for p in params]
    if direction_id == "D7-AdamWResidualDiagnostic":
        return [-g.detach() for g in grads]
    if direction_id == "D8-FisherDiagQuadraticDamping":
        if len(out) >= 3:
            denom = grads[2].detach().abs().mean().clamp_min(1.0e-8) + grads[2].detach().abs()
            out[2] = -params[2].detach() / denom
        return out
    if direction_id == "D9-SignalChannelProjection":
        base = snr_lq.quadratic_coeff_direction(params, basis)
        return _project_onto_grads(base, grads)
    if direction_id == "D10-OrthogonalSignalGeometry":
        base = snr_lq.quadratic_coeff_direction(params, basis)
        return _remove_grad_parallel(base, grads)
    if direction_id == "D11-SignalSubspaceCurvature":
        base = direction_for_id("D8-FisherDiagQuadraticDamping", params, grads, basis)
        return _project_onto_grads(base, grads)
    if direction_id == "D12-SignalSubspaceMarginTail":
        base = direction_for_id("D4-OutputLinearDamping", params, grads, basis)
        return _project_onto_grads(base, grads)
    if direction_id == "D13-LiftConditionCorrection":
        if params:
            col_norm = params[0].detach().float().norm(dim=0, keepdim=True)
            scale = (col_norm / col_norm.mean().clamp_min(1.0e-8)).clamp(0.25, 4.0).to(params[0].dtype)
            out[0] = -params[0].detach() * scale
        return out
    if direction_id == "D14-QuadraticBasisEntropyCorrection":
        base = snr_lq.quadratic_coeff_direction(params, basis)
        if len(base) >= 3:
            out[2] = -base[2]
        return out
    if direction_id == "D15-OutputScaleTailCorrection":
        return direction_for_id("D5-AllOutputCoeffDamping", params, grads, basis)
    if direction_id == "D16-KMNISTHardModeGeometry":
        d10 = direction_for_id("D10-OrthogonalSignalGeometry", params, grads, basis)
        d13 = direction_for_id("D13-LiftConditionCorrection", params, grads, basis)
        return snr_lq.add_steps(d10, d13, alpha=0.25)
    raise ValueError(f"unknown direction_id {direction_id}")


def random_matched_direction(
    params: Sequence[torch.Tensor],
    reference_step: Sequence[torch.Tensor],
    *,
    seed: int,
    device: torch.device,
) -> List[torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(int(seed))
    out = snr_lq.zero_like_params(params)
    out[-1] = torch.randn(params[-1].shape, device=device, generator=gen, dtype=params[-1].dtype)
    return snr_lq.scale_direction_to_fraction_of_task_step(out, reference_step, 1.0)


def mechanism_metrics(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    x: torch.Tensor,
) -> Dict[str, float]:
    with torch.no_grad():
        h = x @ params[0]
        lift = lq.lift_feature_metrics(h, mu, std, basis)
        vals, ders = lq.basis_from_lift(h, mu, std, basis, 2.0, 2.0)
        # Local output sensitivity to lifted coordinates.
        dh_terms = []
        for der, weight in zip(ders, params[1:]):
            dh_terms.append(der.unsqueeze(-1) * weight.unsqueeze(0))
        dh = torch.stack(dh_terms, dim=0).sum(dim=0)
        local_lipschitz = float(dh.float().norm(dim=2).mean().detach().cpu())
        # Nonlinear curvature proxy: magnitude of nonlinear coefficients scaled
        # by basis normalization.  It is a proxy, not a Hessian claim.
        curvature = torch.zeros((), device=x.device, dtype=torch.float32)
        if basis in {"t2", "t2t3", "legendre23"} and len(params) >= 3:
            scale = (std.float().clamp_min(1.0e-6) * 2.0).square().reciprocal()
            curvature = curvature + (params[2].float().norm(dim=1) * scale).mean()
        if basis in {"t2t3", "legendre23"} and len(params) >= 4:
            scale = (std.float().clamp_min(1.0e-6) * 2.0).pow(3).reciprocal()
            curvature = curvature + (params[3].float().norm(dim=1) * scale).mean()
        out = {
            "curvature_proxy": float(curvature.detach().cpu()),
            "local_lipschitz_proxy": local_lipschitz,
            "basis_usage_entropy": float(lift.get("basis_usage_entropy", 0.0)),
            "lift_condition_number": float(lift.get("lift_condition_number", 0.0)),
            "dominant_basis_fraction": float(lift.get("dominant_basis_fraction", 0.0)),
        }
    return out


def is_route_eligible_direction(direction_id: str) -> bool:
    return direction_id != "D7-AdamWResidualDiagnostic"


def event_accepts(
    event_type: str,
    *,
    metrics: Dict[str, float],
    mechanism: Dict[str, float],
    quadratic_snr: float,
    step: int,
    stride: int = 8,
    cep99_tau: float = 4.0,
    margin_tau: float = 0.0,
    curvature_tau: float = 0.08,
    entropy_tau: float = 0.68,
    snr_tau: float = 20.0,
) -> bool:
    if event_type == "E0-uniform-stride":
        return int(step) % max(1, int(stride)) == 0
    if event_type == "E1-CEp99-tail":
        return float(metrics.get("CE_p99", metrics.get("CEp99", 0.0))) >= float(cep99_tau)
    if event_type == "E2-margin-tail":
        return float(metrics.get("correct_margin_p10", metrics.get("margin_p10", 0.0))) <= float(margin_tau)
    if event_type == "E3-curvature-spike":
        return float(mechanism.get("curvature_proxy", 0.0)) >= float(curvature_tau)
    if event_type == "E4-basis-entropy-collapse":
        return float(mechanism.get("basis_usage_entropy", 1.0)) <= float(entropy_tau)
    if event_type == "E5-high-confidence-SNR":
        return float(quadratic_snr) >= float(snr_tau)
    raise ValueError(f"unknown event_type {event_type}")
