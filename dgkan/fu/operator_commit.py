"""v22.13 metric commit wrapper."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.constructive_commit import solve_external_target_commit


def commit_operator_target(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    solver_level: str,
    block_role: str = "readout_only",
    damping: float = 1.0e-3,
    fit_scope: str = "all_train_stream",
    seed: int = 2213,
) -> tuple[torch.Tensor, dict[str, Any]]:
    update, diag = solve_external_target_commit(
        model,
        x,
        None,
        target_delta,
        solver_level=solver_level,
        block_role=block_role,
        damping=damping,
        optimizer_state_write_fraction=0.0,
        fit_scope=fit_scope,
        seed=seed,
    )
    diag.update(
        {
            "uses_adapter_name_for_direction": 0,
            "uses_loss_formula_for_direction": 0,
            "uses_labels_for_fu_core": 0,
            "uses_validation_test_future_query": 0,
            "uses_audit_metric_for_direction": 0,
            "source_state_write_fraction": 1.0,
        }
    )
    return update, diag


__all__ = ["commit_operator_target"]

