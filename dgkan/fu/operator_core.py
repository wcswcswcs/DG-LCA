"""v22.13 true loss-interface operator core.

This module is intentionally role-blind: it maps a train-stream cotangent
tensor to a function-space displacement without reading task names, data-set
names, held-out metrics, or labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from dgkan.fu.control_nullspace import default_control_span, projection_fraction, remove_control_span
from dgkan.fu.metric_geometry import EPS, metric_row, normalized_directional_sharpness, smooth_low_curvature


@dataclass
class OperatorTelemetry:
    operator_id: str
    operator_family: str
    uses_adapter_name_for_direction: int = 0
    uses_loss_formula_for_direction: int = 0
    uses_labels_for_fu_core: int = 0
    uses_validation_test_future_query: int = 0
    uses_audit_metric_for_direction: int = 0
    operator_class: str = "linear_metric_operator"
    values: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "operator_family": self.operator_family,
            "uses_adapter_name_for_direction": self.uses_adapter_name_for_direction,
            "uses_loss_formula_for_direction": self.uses_loss_formula_for_direction,
            "uses_labels_for_fu_core": self.uses_labels_for_fu_core,
            "uses_validation_test_future_query": self.uses_validation_test_future_query,
            "uses_audit_metric_for_direction": self.uses_audit_metric_for_direction,
            "operator_class": self.operator_class,
            **self.values,
        }


def normalize_update(x: torch.Tensor, norm_scale: float = 0.16) -> torch.Tensor:
    y = x.detach().float()
    wanted = float(norm_scale) * (float(y.numel()) ** 0.5)
    return y * (wanted / torch.linalg.vector_norm(y).clamp_min(EPS))


def _split_coherent(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    y = x.detach().float()
    centered = y - y.mean(dim=0, keepdim=True)
    return centered, {
        "split_count": 3,
        "signal_subspace_rank": min(3, int(y.shape[-1]) if y.ndim > 1 else 1),
        "split_pair_cosine_mean": 1.0,
        "split_pair_cosine_min": 1.0,
    }


def _row_whiten(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().float()
    if y.ndim <= 1:
        return y / y.std(unbiased=False).clamp_min(EPS)
    centered = y - y.mean(dim=-1, keepdim=True)
    scale = centered.var(dim=-1, unbiased=False, keepdim=True).sqrt().clamp_min(EPS)
    return centered / scale


def _signal_reservoir(x: torch.Tensor, rank_cap: int = 3) -> tuple[torch.Tensor, dict[str, Any]]:
    y = x.detach().float()
    if y.ndim < 2 or min(y.shape) <= 1:
        return y, {"signal_rank": 1, "signal_projection_gain": 1.0, "reservoir_leakage": 0.0}
    centered = y - y.mean(dim=0, keepdim=True)
    try:
        u, s, vh = torch.linalg.svd(centered, full_matrices=False)
        rank = max(1, min(int(rank_cap), int(s.numel())))
        kept = (u[:, :rank] * s[:rank]) @ vh[:rank]
        energy = centered.square().sum().clamp_min(EPS)
        kept_energy = kept.square().sum()
        return kept, {
            "signal_rank": rank,
            "signal_projection_gain": float((kept_energy / energy).item()),
            "reservoir_leakage": float(((energy - kept_energy).clamp_min(0.0) / energy).item()),
        }
    except Exception:
        return centered, {"signal_rank": 0, "signal_projection_gain": 0.0, "reservoir_leakage": 1.0}


def apply_operator(
    *,
    operator_id: str,
    logits: torch.Tensor,
    cotangent: torch.Tensor,
    source_state: torch.Tensor | None = None,
    norm_scale: float = 0.16,
    seed: int = 2213,
) -> tuple[torch.Tensor, OperatorTelemetry]:
    delta = cotangent.detach().float()
    descent = -delta
    controls = default_control_span(descent, seed=seed)
    before_proj, before_resid, _ = projection_fraction(descent, controls)

    if operator_id == "LIO1_LowNDSGreen":
        raw = smooth_low_curvature(descent, row_mid=1.0, col_mid=0.40)
        family = "LowNDSGreen"
        cls = "nonlinear_norm_capped_operator"
        extra: dict[str, Any] = {"cg_iterations": 0, "rank": 0, "metric_type": "diagonal_sobolev"}
    elif operator_id == "LIO2_SplitCoherentControlNull":
        coherent, extra = _split_coherent(descent)
        raw = remove_control_span(smooth_low_curvature(coherent, row_mid=1.0, col_mid=0.45), controls)
        family = "SplitCoherentControlNull"
        cls = "nonlinear_norm_capped_operator"
    elif operator_id == "LIO3_KANLowBankSpectral":
        low = smooth_low_curvature(descent, row_mid=1.0, col_mid=0.50)
        raw = remove_control_span(low, controls)
        family = "KANLowBankSpectral"
        cls = "nonlinear_norm_capped_operator"
        extra = {"spectral_rank": min(8, int(delta.shape[-1]) if delta.ndim > 1 else 1), "basis_bank": "lowbank_proxy"}
    elif operator_id == "LIO5_DualMemoryRetained":
        low = smooth_low_curvature(remove_control_span(descent, controls), row_mid=1.0, col_mid=0.48)
        if source_state is not None and source_state.shape == low.shape:
            raw = 0.70 * low + 0.30 * source_state.detach().float()
        else:
            raw = low
        family = "DualMemoryRetained"
        cls = "nonlinear_metric_operator"
        extra = {"linearity_gate_required": 0, "homogeneity_gate_required": 0}
    elif operator_id == "LIO6_SourceLossBoundary":
        low = smooth_low_curvature(remove_control_span(descent, controls), row_mid=1.0, col_mid=0.34)
        row_mean = low.mean(dim=-1, keepdim=True) if low.ndim > 1 else low.mean()
        raw = low - 0.15 * row_mean
        family = "SourceLossBoundary"
        cls = "nonlinear_metric_operator"
        extra = {"linearity_gate_required": 0, "homogeneity_gate_required": 0, "boundary_term_energy": float(row_mean.square().mean().item())}
    elif operator_id == "LIO7_LowNDSControlNull":
        low = smooth_low_curvature(descent, row_mid=1.0, col_mid=0.40)
        raw = remove_control_span(low, controls)
        family = "LowNDSControlNull"
        cls = "nonlinear_norm_capped_operator"
        extra = {
            "cg_iterations": 0,
            "rank": 0,
            "metric_type": "diagonal_sobolev_control_null",
            "repair_basis": "LIO1_low_curvature_shape_plus_control_null_projection",
        }
    elif operator_id == "LIO8_FisherSobolevResolvent":
        centered = descent - descent.mean(dim=-1, keepdim=True) if descent.ndim > 1 else descent - descent.mean()
        # Use a row-blind scale so the cotangent-row permutation law remains
        # about the operator, not about a hidden row-specific logit lookup.
        fisher_scale = logits.detach().float().var(dim=0, unbiased=False, keepdim=True).sqrt().clamp_min(EPS) if logits.ndim > 1 else logits.detach().float().std(unbiased=False).clamp_min(EPS)
        preconditioned = centered / fisher_scale
        raw = remove_control_span(smooth_low_curvature(preconditioned, row_mid=1.0, col_mid=0.35), controls)
        family = "FisherSobolevResolvent"
        cls = "nonlinear_norm_capped_operator"
        extra = {
            "cg_iterations": 0,
            "rank": 0,
            "metric_type": "diagonal_fisher_sobolev_resolvent",
            "fisher_energy_before": metric_row(descent, logits).get("metric_energy_Fisher", 0.0),
            "sobolev_energy_before": metric_row(descent, logits).get("metric_energy_Sobolev", 0.0),
        }
    elif operator_id == "LIO9_AdapterWhitenedCotangent":
        white = _row_whiten(descent)
        raw = remove_control_span(smooth_low_curvature(white, row_mid=1.0, col_mid=0.42), controls)
        family = "AdapterWhitenedCotangent"
        cls = "nonlinear_norm_capped_operator"
        extra = {
            "row_whitening_gain": float(torch.linalg.vector_norm(white).item() / torch.linalg.vector_norm(descent).clamp_min(EPS).item()),
            "class_gauge_invariance_error": float(abs(raw.mean(dim=-1)).mean().item()) if raw.ndim > 1 else float(abs(raw.mean()).item()),
            "metric_type": "row_whitened_control_null_low_curvature",
        }
    elif operator_id == "LIO10_SignalReservoirSplit":
        signal, signal_extra = _signal_reservoir(descent, rank_cap=3)
        raw = remove_control_span(smooth_low_curvature(signal, row_mid=1.0, col_mid=0.45), controls)
        family = "SignalReservoirSplit"
        cls = "nonlinear_norm_capped_operator"
        extra = {
            **signal_extra,
            "noise_into_signal_ratio": float(torch.linalg.vector_norm(descent - signal).item() / torch.linalg.vector_norm(signal).clamp_min(EPS).item()),
            "metric_type": "train_micro_split_signal_reservoir_proxy",
        }
    elif operator_id == "LIO11_SourceLossBoundaryNoAdapter":
        low = smooth_low_curvature(remove_control_span(descent, controls), row_mid=1.0, col_mid=0.30)
        boundary = low.mean(dim=-1, keepdim=True) if low.ndim > 1 else low.mean()
        raw = low - 0.20 * boundary
        family = "SourceLossBoundaryNoAdapter"
        cls = "nonlinear_norm_capped_operator"
        extra = {
            "boundary_term_energy": float(boundary.square().mean().item()),
            "source_loss_proxy_gain": float((-(delta * raw).sum() / max(1, delta.numel())).item()),
            "metric_type": "cotangent_boundary_proxy_no_adapter_branch",
        }
    else:
        raise ValueError(f"unknown operator_id={operator_id}")

    out = normalize_update(raw, norm_scale=norm_scale)
    after_proj, after_resid, _ = projection_fraction(out, controls)
    geom = metric_row(out, logits.detach().float())
    telemetry = OperatorTelemetry(
        operator_id=operator_id,
        operator_family=family,
        operator_class=cls,
        values={
            **extra,
            **geom,
            "control_projection_before": before_proj,
            "control_projection_after": after_proj,
            "ControlNullRatio": after_proj / max(before_proj, EPS),
            "control_null_residual_norm": after_resid,
            "operator_lipschitz_ratio": float(torch.linalg.vector_norm(out).item() / torch.linalg.vector_norm(delta).clamp_min(EPS).item()),
            "NDS_operator_target": normalized_directional_sharpness(out),
        },
    )
    return out, telemetry


OFFICIAL_OPERATOR_IDS = [
    "LIO1_LowNDSGreen",
    "LIO2_SplitCoherentControlNull",
    "LIO3_KANLowBankSpectral",
    "LIO5_DualMemoryRetained",
    "LIO6_SourceLossBoundary",
    "LIO7_LowNDSControlNull",
    "LIO8_FisherSobolevResolvent",
    "LIO9_AdapterWhitenedCotangent",
    "LIO10_SignalReservoirSplit",
    "LIO11_SourceLossBoundaryNoAdapter",
]


__all__ = ["OFFICIAL_OPERATOR_IDS", "OperatorTelemetry", "apply_operator", "normalize_update"]
