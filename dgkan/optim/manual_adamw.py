"""AdamW-equivalent manual task optimizer.

Functional updates intentionally do not live here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ManualAdamWConfig:
    lr: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    weight_decay: float = 0.0


@dataclass
class AdamWState:
    step: int
    m: torch.Tensor
    v: torch.Tensor

    @classmethod
    def zeros_like(cls, param: torch.Tensor) -> "AdamWState":
        return cls(step=0, m=torch.zeros_like(param), v=torch.zeros_like(param))


def adamw_update_(param: torch.Tensor, grad: torch.Tensor, state: AdamWState, cfg: ManualAdamWConfig) -> None:
    state.step += 1
    state.m.mul_(cfg.beta1).add_(grad, alpha=1.0 - cfg.beta1)
    state.v.mul_(cfg.beta2).addcmul_(grad, grad, value=1.0 - cfg.beta2)
    bias1 = 1.0 - cfg.beta1**state.step
    bias2 = 1.0 - cfg.beta2**state.step
    denom = state.v.sqrt() / (bias2**0.5)
    update = (state.m / bias1) / (denom + cfg.eps)
    if cfg.weight_decay:
        update = update + cfg.weight_decay * param
    param.add_(update, alpha=-cfg.lr)
