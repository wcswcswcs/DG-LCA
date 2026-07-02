"""Finite-step trust-region utilities for DG-KAN v23.01.

The experiment runner owns the task-specific debt metrics, while this module
provides auditable snapshot/restore primitives used by the trust-region smoke
tests and implementation identity checks.
"""

from __future__ import annotations

import copy
from typing import Any

import torch


def clone_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_state_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_state_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_state_value(v) for v in value)
    return copy.deepcopy(value)


def snapshot_model_params(model: torch.nn.Module) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    return [(p, p.detach().clone()) for p in model.parameters()]


@torch.no_grad()
def restore_model_params(snapshot: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    for param, value in snapshot:
        param.copy_(value)


def max_snapshot_error(snapshot: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> float:
    err = 0.0
    for param, value in snapshot:
        err = max(err, float((param.detach() - value).abs().max().cpu().item()))
    return err


def snapshot_optimizer(opt: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "state": {p: {k: clone_state_value(v) for k, v in state.items()} for p, state in opt.state.items()},
        "group_scalars": [
            {k: clone_state_value(v) for k, v in group.items() if k != "params" and not isinstance(v, torch.Tensor)}
            for group in opt.param_groups
        ],
    }


def restore_optimizer(opt: torch.optim.Optimizer, snapshot: dict[str, Any]) -> None:
    opt.state.clear()
    for param, state in snapshot.get("state", {}).items():
        opt.state[param] = {k: clone_state_value(v) for k, v in state.items()}
    for group, scalars in zip(opt.param_groups, snapshot.get("group_scalars", [])):
        for key, value in scalars.items():
            group[key] = clone_state_value(value)


def optimizer_state_max_abs_error(opt: torch.optim.Optimizer, snapshot: dict[str, Any]) -> float:
    err = 0.0
    for param, expected in snapshot.get("state", {}).items():
        actual = opt.state.get(param, {})
        for key, exp_value in expected.items():
            act_value = actual.get(key)
            if isinstance(exp_value, torch.Tensor) and isinstance(act_value, torch.Tensor):
                err = max(err, float((act_value.detach() - exp_value).abs().max().cpu().item()))
            elif exp_value != act_value:
                err = max(err, 1.0)
    return err


__all__ = [
    "clone_state_value",
    "snapshot_model_params",
    "restore_model_params",
    "max_snapshot_error",
    "snapshot_optimizer",
    "restore_optimizer",
    "optimizer_state_max_abs_error",
]
