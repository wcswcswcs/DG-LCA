"""Shared trust-feature extraction helpers for DG-KAN v23.02."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

import torch

from dgkan.fu.predictive_trust_radius import DEBT_COMPONENTS


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def flatten_params(params: Iterable[torch.Tensor]) -> torch.Tensor:
    rows = [p.detach().reshape(-1).to(dtype=torch.float64).cpu() for p in params]
    return torch.cat(rows) if rows else torch.zeros(0, dtype=torch.float64)


def debt_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    return {
        "Brier": finite(after.get("brier", after.get("Brier", 0.0))) - finite(before.get("brier", before.get("Brier", 0.0))),
        "ECE": finite(after.get("ece", after.get("ECE", 0.0))) - finite(before.get("ece", before.get("ECE", 0.0))),
        "tail95": finite(after.get("tail95", 0.0)) - finite(before.get("tail95", 0.0)),
        "tail99": finite(after.get("tail99", 0.0)) - finite(before.get("tail99", 0.0)),
        "margin10": finite(after.get("margin10", 0.0)) - finite(before.get("margin10", 0.0)),
    }


def quadratic_coefficients(
    baseline: dict[str, Any],
    probe_metrics: dict[str, Any],
    *,
    probe_scale: float,
    linear_terms: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fit componentwise quadratic coefficients from one probe point."""
    out: dict[str, float] = {}
    deltas = debt_delta(probe_metrics, baseline)
    denom = float(probe_scale) * float(probe_scale) + 1.0e-12
    for component in DEBT_COMPONENTS:
        a = finite((linear_terms or {}).get(component, 0.0))
        b = (finite(deltas.get(component, 0.0)) - a * float(probe_scale)) / denom
        out[f"predicted_debt_delta_linear_{component}"] = a
        out[f"predicted_debt_delta_quadratic_{component}"] = b
        out[f"observed_probe_debt_delta_{component}"] = finite(deltas.get(component, 0.0))
    return out


def mode_energy_low_mid_high(vec: torch.Tensor) -> str:
    flat = vec.detach().reshape(-1).to(dtype=torch.float64).cpu()
    if int(flat.numel()) == 0:
        return json.dumps({"low": 0.0, "mid": 0.0, "high": 0.0}, sort_keys=True)
    chunks = torch.chunk(flat.square(), 3)
    total = flat.square().sum().clamp_min(1.0e-12)
    vals = [float(c.sum().div(total).item()) for c in chunks]
    while len(vals) < 3:
        vals.append(0.0)
    return json.dumps({"low": vals[0], "mid": vals[1], "high": vals[2]}, sort_keys=True)


def summarize_recent(recent_accepts: Iterable[float], recent_scales: Iterable[float]) -> dict[str, float]:
    acc = [finite(x) for x in recent_accepts]
    scales = [finite(x) for x in recent_scales]
    return {
        "recent_accept_rate_ema": sum(acc[-20:]) / max(1, len(acc[-20:])),
        "recent_scale_mean_ema": sum(scales[-20:]) / max(1, len(scales[-20:])),
    }


def build_trust_feature_row(
    *,
    step: int,
    scheme: str,
    task: str,
    seed: int,
    current_metrics: dict[str, Any],
    update_vector: torch.Tensor,
    gate_density: float,
    gate_entropy: float = 0.0,
    block_snr_mean: float = 0.0,
    block_snr_top_decile: float = 0.0,
    functional_gram_condition: float = 1.0,
    recent_accepts: Iterable[float] = (),
    recent_scales: Iterable[float] = (),
    quadratic_terms: dict[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": int(step),
        "scheme": str(scheme),
        "task": str(task),
        "seed": int(seed),
        "current_NLL": finite(current_metrics.get("nll", current_metrics.get("NLL", 0.0))),
        "current_accuracy": finite(current_metrics.get("accuracy", 0.0)),
        "current_Brier": finite(current_metrics.get("brier", current_metrics.get("Brier", 0.0))),
        "current_ECE": finite(current_metrics.get("ece", current_metrics.get("ECE", 0.0))),
        "current_tail95": finite(current_metrics.get("tail95", 0.0)),
        "current_tail99": finite(current_metrics.get("tail99", 0.0)),
        "current_margin10": finite(current_metrics.get("margin10", 0.0)),
        "update_norm_L2": float(update_vector.detach().reshape(-1).to(dtype=torch.float64).norm().item()) if int(update_vector.numel()) else 0.0,
        "update_norm_Gedge": float(update_vector.detach().reshape(-1).to(dtype=torch.float64).norm().item()) if int(update_vector.numel()) else 0.0,
        "qpop_gate_density": float(gate_density),
        "qpop_gate_entropy": float(gate_entropy),
        "block_snr_mean": float(block_snr_mean),
        "block_snr_top_decile": float(block_snr_top_decile),
        "functional_gram_condition": float(functional_gram_condition),
        "mode_energy_low_mid_high": mode_energy_low_mid_high(update_vector),
    }
    row.update(summarize_recent(recent_accepts, recent_scales))
    if quadratic_terms:
        row.update(quadratic_terms)
    row["risk_score"] = max(finite(row.get(f"predicted_debt_delta_linear_{c}", 0.0)) + finite(row.get(f"predicted_debt_delta_quadratic_{c}", 0.0)) for c in DEBT_COMPONENTS)
    return row


def trust_feature_smoke_test() -> dict[str, int | float]:
    base = {"brier": 0.2, "ece": 0.1, "tail95": 0.4, "tail99": 0.5, "margin10": 0.0}
    probe = {"brier": 0.19, "ece": 0.11, "tail95": 0.39, "tail99": 0.49, "margin10": 0.0}
    quad = quadratic_coefficients(base, probe, probe_scale=0.125)
    row = build_trust_feature_row(step=1, scheme="smoke", task="smoke", seed=0, current_metrics=base, update_vector=torch.ones(4), gate_density=0.5, quadratic_terms=quad)
    return {
        "trust_feature_extractor_available": 1,
        "trust_feature_has_risk_score": int("risk_score" in row),
        "trust_feature_update_norm_L2": finite(row.get("update_norm_L2")),
    }


__all__ = [
    "finite",
    "flatten_params",
    "debt_delta",
    "quadratic_coefficients",
    "mode_energy_low_mid_high",
    "summarize_recent",
    "build_trust_feature_row",
    "trust_feature_smoke_test",
]
