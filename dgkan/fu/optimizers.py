"""Optimizer-pluggable update backends for v17."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from dgkan.fu.core import UpdateTensor, apply_update


@dataclass
class OptimizerStepResult:
    optimizer_name: str
    update_kind: str
    applied_lr: float
    step_count: int


class ManualOptimizer:
    def __init__(self, name: str, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float = 0.0, momentum: float = 0.0) -> None:
        self.name = str(name)
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.step_count = 0
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step_gradient(self) -> OptimizerStepResult:
        self.step_count += 1
        with torch.no_grad():
            for idx, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if self.weight_decay:
                    g = g + self.weight_decay * p.detach()
                if self.momentum:
                    self.velocity[idx].mul_(self.momentum).add_(g)
                    g = self.velocity[idx]
                p.add_(g, alpha=-self.lr)
        return OptimizerStepResult(self.name, "gradient", self.lr, self.step_count)

    def step_update(self, model: torch.nn.Module, update: UpdateTensor, lr: float | None = None) -> OptimizerStepResult:
        self.step_count += 1
        apply_update(model, update, self.lr if lr is None else float(lr))
        return OptimizerStepResult(self.name, update.kind, self.lr if lr is None else float(lr), self.step_count)


def make_manual_optimizer(name: str, model: torch.nn.Module, lr: float, weight_decay: float = 0.0) -> ManualOptimizer:
    low = name.lower()
    if low in {"sgd", "fu-only"}:
        return ManualOptimizer(name, model.parameters(), lr, weight_decay=weight_decay, momentum=0.0)
    if low in {"momentum", "nesterov", "sgd-momentum"}:
        return ManualOptimizer(name, model.parameters(), lr, weight_decay=weight_decay, momentum=0.9)
    if low in {"schedule-free", "slow-state"}:
        return ManualOptimizer(name, model.parameters(), lr, weight_decay=0.0, momentum=0.95)
    if low in {"lookahead"}:
        return ManualOptimizer(name, model.parameters(), lr * 0.5, weight_decay=weight_decay, momentum=0.5)
    return ManualOptimizer(name, model.parameters(), lr, weight_decay=weight_decay, momentum=0.0)

