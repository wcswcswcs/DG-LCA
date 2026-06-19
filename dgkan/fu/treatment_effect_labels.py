"""Treatment-effect label helpers for v22.18 benefit-conditioned audits."""

from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in {float("inf"), float("-inf")}:
        return default
    return out


def treatment_effect_label(
    row: dict[str, Any],
    base: dict[str, Any],
    *,
    horizon: int,
    w_n: float = 1.0,
    w_a: float = 0.20,
    w_s: float = 0.05,
    w_t: float = 0.10,
    w_e: float = 0.10,
    w_cost: float = 0.05,
) -> dict[str, Any]:
    nll_delta = safe_float(row.get("final_test_loss_NLL")) - safe_float(base.get("final_test_loss_NLL"))
    auc_delta = safe_float(row.get("AUC_loss_time")) - safe_float(base.get("AUC_loss_time"))
    source_loss_delta = safe_float(row.get("source_loss_h_final"), safe_float(row.get("KAN_source_loss_horizon"))) - safe_float(
        base.get("source_loss_h_final"), safe_float(base.get("KAN_base_source_loss_horizon"))
    )
    tail_delta = safe_float(row.get("tail_loss_q99")) - safe_float(base.get("tail_loss_q99"))
    ece_delta = safe_float(row.get("ECE")) - safe_float(base.get("ECE"))
    cost = safe_float(row.get("controller_overhead_ratio"))
    utility = -w_n * nll_delta - w_a * auc_delta + w_s * source_loss_delta - w_t * tail_delta - w_e * ece_delta - w_cost * cost
    return {
        "horizon": int(horizon),
        "delta_NLL": nll_delta,
        "delta_AUC_loss_time": auc_delta,
        "delta_source_loss": source_loss_delta,
        "delta_tail_q99": tail_delta,
        "delta_ECE": ece_delta,
        "controller_cost": cost,
        "benefit_utility": utility,
        "benefit_positive": int(nll_delta < -0.01 or (nll_delta <= 0.0 and auc_delta < 0.0)),
        "true_improvement_positive": int(nll_delta < -0.01 and auc_delta < 0.0),
        "noharm_positive": int(nll_delta <= 0.02 and tail_delta <= 0.05 and ece_delta <= 0.02),
        "label_source": "final_run_delta_proxy",
        "true_horizon_counterfactual": 0,
    }
