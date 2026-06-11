"""v17.0.1 update type aliases and canonical metadata mapping."""

from __future__ import annotations

from typing import Any

from dgkan.fu.core import UpdateTensor


KIND_TO_DIRECTION_TYPE = {
    "gradient": "gradient_like",
    "step": "step_like",
    "cotangent": "cotangent_like",
    "metric_state": "metric_state",
    "function_displacement": "function_displacement",
}

SIGN_TO_CONVENTION = {
    "subtract": "descent_positive",
    "add": "ascent_positive",
}


def canonical_update_row(update: UpdateTensor) -> dict[str, Any]:
    direction_type = KIND_TO_DIRECTION_TYPE.get(update.kind, "unknown")
    sign_convention = SIGN_TO_CONVENTION.get(update.sign_rule, "unknown")
    return {
        "update_id": update.mechanism,
        "direction_type": direction_type,
        "commit_semantics": "add_to_param" if update.kind in {"step", "gradient", "cotangent"} else "readback_only",
        "sign_convention": sign_convention,
        "uses_optimizer": "none",
        "uses_weight_decay": 0,
        "uses_momentum_state": int(update.mechanism == "M2-SGDMomentumPrimaryFU"),
        "uses_second_moment_state": int(update.mechanism == "M1-AdamWPrimaryFUResidual"),
        "role_mask": update.role,
        "param_count_touched": int(update.tensor.numel()),
        "space": update.space,
        "source": update.source,
    }
