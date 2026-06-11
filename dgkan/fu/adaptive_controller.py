"""Adaptive function-space controller primitives for v22.15.

The functions in this module consume train-stream tensors and past source
state only.  They do not inspect task names, held-out metrics, or labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


EPS = 1.0e-12


@dataclass
class RiskFeatures:
    source_retention: float
    source_retention_delta: float
    predicted_step_source_projection: float
    optimizer_destructive_projection: float
    control_projection_fraction: float
    source_loss_linear_gain: float
    source_age: float
    pairwise_antisymmetry_error: float = 0.0
    basis_channel_energy_fraction: float = 0.0


@dataclass
class RiskDecision:
    washout_risk: float
    lambda_t: float
    release_source: int
    refresh_source: int
    reason: str

    def to_row(self) -> dict[str, Any]:
        return {
            "washout_risk": self.washout_risk,
            "controller_lambda_t": self.lambda_t,
            "release_source": self.release_source,
            "refresh_source": self.refresh_source,
            "controller_reason": self.reason,
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().reshape(-1)
    bv = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(av).clamp_min(EPS) * torch.linalg.vector_norm(bv).clamp_min(EPS)
    return float(torch.dot(av, bv).div(denom).item())


def retention_score(effect: torch.Tensor, source: torch.Tensor) -> float:
    return cosine(effect, source)


def linearized_source_loss(cotangent: torch.Tensor, effect: torch.Tensor) -> float:
    c = cotangent.detach().float().reshape(-1)
    e = effect.detach().float().reshape(-1)
    n = max(1, int(c.numel()))
    return float((-(c * e).sum() / n).item())


def destructive_projection(base_effect: torch.Tensor, source: torch.Tensor) -> float:
    align = retention_score(base_effect, source)
    return max(0.0, -align)


def risk_from_features(features: RiskFeatures) -> float:
    loss_boundary = max(0.0, -float(features.source_loss_linear_gain))
    decay = max(0.0, -float(features.source_retention_delta))
    destructive = max(0.0, float(features.optimizer_destructive_projection))
    weak_current = max(0.0, 0.30 - float(features.source_retention))
    pair_error = max(0.0, float(features.pairwise_antisymmetry_error) - 0.05)
    basis_underuse = max(0.0, 0.30 - float(features.basis_channel_energy_fraction))
    age = min(1.0, max(0.0, float(features.source_age) / 2000.0))
    support = max(0.0, float(features.predicted_step_source_projection))
    risk = (
        0.06
        + 0.30 * destructive
        + 0.24 * min(1.0, 8.0 * loss_boundary)
        + 0.20 * min(1.0, 4.0 * decay)
        + 0.14 * min(1.0, 2.0 * weak_current)
        + 0.08 * age
        + 0.06 * min(1.0, 5.0 * pair_error)
        + 0.06 * min(1.0, basis_underuse)
        + 0.04 * _clamp01(features.control_projection_fraction)
        - 0.18 * min(1.0, support)
    )
    return _clamp01(risk)


def lambda_from_risk(
    washout_risk: float,
    *,
    threshold: float = 0.35,
    max_lambda: float = 8.0,
    floor_lambda: float = 0.0,
) -> float:
    r = _clamp01(washout_risk)
    if r <= threshold:
        return float(floor_lambda)
    x = (r - threshold) / max(EPS, 1.0 - threshold)
    return float(floor_lambda + max_lambda * x * x)


def decide_controller(
    features: RiskFeatures,
    *,
    threshold: float = 0.35,
    max_lambda: float = 8.0,
    release_loss_threshold: float = -1.0e-6,
    refresh_age_threshold: float = 1200.0,
) -> RiskDecision:
    risk = risk_from_features(features)
    lam = lambda_from_risk(risk, threshold=threshold, max_lambda=max_lambda)
    release = int(float(features.source_loss_linear_gain) < float(release_loss_threshold))
    refresh = int(float(features.source_age) >= float(refresh_age_threshold) or risk >= 0.70)
    reasons: list[str] = []
    if release:
        reasons.append("source_loss_boundary")
    if refresh:
        reasons.append("risk_or_age_refresh")
    if lam > 0.0:
        reasons.append("predictive_guidance")
    if not reasons:
        reasons.append("base_step_supported")
    return RiskDecision(risk, lam, release, refresh, "+".join(reasons))


def source_guided_update(
    jacobian: torch.Tensor,
    base_update: torch.Tensor,
    source: torch.Tensor,
    lambda_t: float,
    *,
    damping: float = 1.0e-5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve a low-rank prox step in parameter coordinates."""
    j = jacobian.detach().float()
    u = base_update.detach().float().reshape(-1)
    z = source.detach().float().reshape(-1)
    if j.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    if j.shape[1] != u.numel():
        raise ValueError("jacobian columns must match base_update")
    if j.shape[0] != z.numel():
        raise ValueError("jacobian rows must match source")

    base_effect = j @ u
    before = torch.linalg.vector_norm(base_effect - z).item()
    base_align = retention_score(base_effect, z)
    if float(lambda_t) <= 0.0:
        return u.reshape_as(base_update).clone(), {
            "prox_residual_before": float(before),
            "prox_residual_after": float(before),
            "source_alignment_before": float(base_align),
            "source_alignment_after": float(base_align),
            "source_guided_solve_status": "lambda_zero",
        }

    eye = torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
    lam = float(lambda_t)
    lhs = eye + lam * (j.T @ j) + float(damping) * eye
    rhs = u + lam * (j.T @ z)
    status = "solve"
    try:
        corrected = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        corrected = torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution.squeeze(1)
        status = "lstsq"
    effect = j @ corrected
    after = torch.linalg.vector_norm(effect - z).item()
    return corrected.reshape_as(base_update), {
        "prox_residual_before": float(before),
        "prox_residual_after": float(after),
        "source_alignment_before": float(base_align),
        "source_alignment_after": retention_score(effect, z),
        "source_guided_solve_status": status,
    }


def make_features_from_step(
    jacobian: torch.Tensor,
    base_update: torch.Tensor,
    source: torch.Tensor,
    cotangent: torch.Tensor,
    *,
    previous_retention: float,
    source_age: float,
    control_projection_fraction: float = 0.0,
    pairwise_antisymmetry_error: float = 0.0,
    basis_channel_energy_fraction: float = 0.0,
) -> RiskFeatures:
    effect = jacobian.detach().float() @ base_update.detach().float().reshape(-1)
    retention = retention_score(effect, source)
    return RiskFeatures(
        source_retention=retention,
        source_retention_delta=retention - float(previous_retention),
        predicted_step_source_projection=retention,
        optimizer_destructive_projection=destructive_projection(effect, source),
        control_projection_fraction=float(control_projection_fraction),
        source_loss_linear_gain=linearized_source_loss(cotangent, effect),
        source_age=float(source_age),
        pairwise_antisymmetry_error=float(pairwise_antisymmetry_error),
        basis_channel_energy_fraction=float(basis_channel_energy_fraction),
    )


__all__ = [
    "RiskDecision",
    "RiskFeatures",
    "cosine",
    "decide_controller",
    "destructive_projection",
    "lambda_from_risk",
    "linearized_source_loss",
    "make_features_from_step",
    "retention_score",
    "risk_from_features",
    "source_guided_update",
]
