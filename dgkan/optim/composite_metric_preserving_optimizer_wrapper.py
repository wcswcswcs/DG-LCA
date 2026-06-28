"""Optimizer-owned wrapper for layer composite metric-preserving flow."""

from __future__ import annotations

from typing import Any


class CompositeMetricPreservingOptimizerWrapper:
    """Wrap a PyTorch optimizer and transform/retract edge parameters in step()."""

    def __init__(self, base_optimizer: Any, operator: Any) -> None:
        self.base = base_optimizer
        self.operator = operator
        self.step_called = 0
        self.gradient_transform_inside_step_called = 0
        self.retraction_inside_step_called = 0

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
        self.operator.set_inside_optimizer_step(True)
        try:
            for group in self.param_groups:
                for param in group.get("params", []):
                    grad = getattr(param, "grad", None)
                    if grad is None:
                        continue
                    if hasattr(self.operator, "owns_param") and not self.operator.owns_param(param):
                        continue
                    new_grad = self.operator.transform(grad, param, group)
                    if new_grad is not grad:
                        param.grad = new_grad
                    self.gradient_transform_inside_step_called += 1
            if closure is None:
                result = self.base.step()
            else:
                result = self.base.step(closure=closure)
            if hasattr(self.operator, "after_base_step"):
                self.operator.after_base_step()
                self.retraction_inside_step_called += 1
            return result
        finally:
            self.operator.set_inside_optimizer_step(False)

    def diagnostics(self) -> dict[str, Any]:
        out = {
            "optimizer_step_called": int(self.step_called > 0),
            "optimizer_owned_gradient_transform_pass": int(self.gradient_transform_inside_step_called > 0),
            "optimizer_owned_retraction_pass": int(self.retraction_inside_step_called > 0),
            "cmp_wrapper_step_calls": self.step_called,
            "cmp_wrapper_gradient_transform_calls": self.gradient_transform_inside_step_called,
            "cmp_wrapper_retraction_calls": self.retraction_inside_step_called,
            "wrapped_optimizer_class": type(self.base).__name__,
        }
        if hasattr(self.operator, "diagnostics"):
            out.update(self.operator.diagnostics())
        return out


__all__ = ["CompositeMetricPreservingOptimizerWrapper"]
