"""AdamW coupling audit helpers for v17.1."""

from __future__ import annotations

from typing import Any


def adamw_coupling_row(method: str) -> dict[str, Any]:
    uses_adamw = method in {"CTRL-AdamW", "CTRL-RecoveryOnly", "M1-AdamWPrimaryFUResidual"}
    return {
        "method": method,
        "uses_adamw_step_for_commit": int(uses_adamw),
        "writes_to_grad_then_adamw_step": 0,
        "manual_step_commit": int(not uses_adamw),
        "fu_primary_commit": int(method in {"M3-FUPrimary", "M4-FUOnlyKeyParams", "M6-SlowStateFU", "M7-MatrixBlockFU", "M8-PopRiskSNRFU", "M9-FunctionSpaceOperatorFU", "M10-RolePartitionOptimizer"}),
        "fu_only_commit": int(method == "M4-FUOnlyKeyParams"),
        "alternating_commit": int(method == "M5-AlternatingFUGradient"),
        "role_partition_commit": int(method == "M10-RolePartitionOptimizer"),
    }
