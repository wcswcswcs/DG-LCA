"""Optimizer-owned witnessed gradient transform wrapper."""

from __future__ import annotations

from typing import Any


class WitnessedOptimizerWrapper:
    """Wrap a PyTorch-style optimizer and transform gradients inside step().

    This class deliberately owns only gradient tensors. It delegates parameter
    updates to the wrapped optimizer.
    """

    def __init__(self, base_optimizer: Any, witnessed_operator: Any) -> None:
        self.base = base_optimizer
        self.witnessed_operator = witnessed_operator
        self.step_called = 0
        self.gradient_transform_inside_step_called = 0

    @property
    def param_groups(self) -> Any:
        groups = getattr(self.base, "param_groups", None)
        if groups is not None:
            return groups
        params = getattr(self.base, "params", None)
        return [{"params": list(params)}] if params is not None else []

    @property
    def state(self) -> Any:
        return getattr(self.base, "state", {})

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any | None = None) -> Any:
        self.step_called += 1
        self.witnessed_operator.set_inside_optimizer_step(True)
        try:
            for group in self.param_groups:
                for param in group.get("params", []):
                    grad = getattr(param, "grad", None)
                    if grad is None:
                        continue
                    if hasattr(self.witnessed_operator, "owns_param") and not self.witnessed_operator.owns_param(param):
                        continue
                    new_grad = self.witnessed_operator.transform(grad, param, group)
                    if new_grad is not grad:
                        param.grad = new_grad
                    self.gradient_transform_inside_step_called += 1
            if closure is None:
                return self.base.step()
            return self.base.step(closure=closure)
        finally:
            self.witnessed_operator.set_inside_optimizer_step(False)

    def diagnostics(self) -> dict[str, Any]:
        out = {
            "optimizer_step_called": int(self.step_called > 0),
            "optimizer_owned_gradient_transform_pass": int(self.gradient_transform_inside_step_called > 0),
            "wgo_wrapper_step_calls": self.step_called,
            "wgo_wrapper_gradient_transform_calls": self.gradient_transform_inside_step_called,
            "wrapped_optimizer_class": type(self.base).__name__,
        }
        if hasattr(self.witnessed_operator, "diagnostics"):
            out.update(self.witnessed_operator.diagnostics())
        return out


__all__ = ["WitnessedOptimizerWrapper"]
