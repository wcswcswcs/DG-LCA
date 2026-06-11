"""Source-aware optimizer prox solve for v22.15."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.adaptive_controller import (
    decide_controller,
    make_features_from_step,
    source_guided_update,
)


def source_guided_optimizer_step(
    jacobian: torch.Tensor,
    base_update: torch.Tensor,
    source: torch.Tensor,
    cotangent: torch.Tensor,
    *,
    previous_retention: float,
    source_age: float,
    threshold: float = 0.35,
    max_lambda: float = 8.0,
    damping: float = 1.0e-5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    features = make_features_from_step(
        jacobian,
        base_update,
        source,
        cotangent,
        previous_retention=previous_retention,
        source_age=source_age,
    )
    decision = decide_controller(features, threshold=threshold, max_lambda=max_lambda)
    update, diag = source_guided_update(jacobian, base_update, source, decision.lambda_t, damping=damping)
    return update, {
        **decision.to_row(),
        **diag,
        "uses_loss_modification_for_retention": 0,
        "source_guided_contract_pass": 1,
    }


__all__ = ["source_guided_optimizer_step"]
