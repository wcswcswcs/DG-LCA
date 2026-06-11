"""Train-flow algebra certificate primitives for v22.09."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.core import cosine


EPS = 1.0e-12


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def commutator_metrics(
    b1_then_b2: torch.Tensor,
    b2_then_b1: torch.Tensor,
    *,
    source_direction: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Measure finite train-flow non-commutativity for two micro flows."""

    a = b1_then_b2.detach().float().reshape(-1)
    b = b2_then_b1.detach().float().reshape(-1)
    k = a - b
    denom = torch.linalg.vector_norm(a).item() + torch.linalg.vector_norm(b).item() + EPS
    out = {
        "commutator_norm": float(torch.linalg.vector_norm(k).item()),
        "commutator_norm_ratio": float(torch.linalg.vector_norm(k).item() / denom),
        "flow_order_gap": float(torch.mean(torch.abs(a - b)).item()) if a.numel() else 0.0,
    }
    if source_direction is not None:
        out["commutator_function_cosine"] = cosine(k, source_direction.detach().float().reshape(-1))
    else:
        out["commutator_function_cosine"] = ""
    return out


def train_flow_algebra_score(row: dict[str, Any]) -> float:
    """Score low-order-gap, repeatable train-flow source evidence."""

    b12 = finite_float(row.get("B1_then_B2_source_gain"), 0.0)
    b21 = finite_float(row.get("B2_then_B1_source_gain"), 0.0)
    reproducibility = finite_float(row.get("source_direction_reproducibility"), 0.0)
    micro = finite_float(row.get("micro_horizon_stability_h1_h2_h4_h8"), 0.0)
    comm = finite_float(row.get("commutator_norm_ratio"), 999.0)
    order_gap = finite_float(row.get("flow_order_gap"), 999.0)
    associativity = finite_float(row.get("multi_split_associativity_error"), 999.0)
    control_gap = finite_float(row.get("random_sign_corrupt_gap"), 0.0)
    return 0.35 * (b12 + b21) + 0.30 * reproducibility + 0.20 * micro + 0.20 * control_gap - 0.55 * comm - 0.35 * order_gap - 0.20 * associativity


def train_flow_commutator_unit_tests() -> list[dict[str, Any]]:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.1, 2.9])
    metrics = commutator_metrics(a, b, source_direction=torch.tensor([0.0, 1.0, -1.0]))
    score = train_flow_algebra_score(
        {
            "B1_then_B2_source_gain": 0.1,
            "B2_then_B1_source_gain": 0.1,
            "source_direction_reproducibility": 1.0,
            "micro_horizon_stability_h1_h2_h4_h8": 1.0,
            "random_sign_corrupt_gap": 0.2,
            "commutator_norm_ratio": 0.01,
            "flow_order_gap": 0.01,
            "multi_split_associativity_error": 0.01,
        }
    )
    return [
        {"test": "commutator_norm_nonnegative", "pass": int(float(metrics["commutator_norm"]) >= 0.0)},
        {"test": "commutator_ratio_bounded", "pass": int(0.0 <= float(metrics["commutator_norm_ratio"]) <= 1.0)},
        {"test": "score_finite", "pass": int(score == score)},
    ]
