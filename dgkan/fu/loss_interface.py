"""Train-stream loss-interface utilities for functional updates.

v22.11 adds an explicit arbitrary-loss interface.  Core FU code can consume a
logit-space cotangent from this module without knowing whether it came from CE,
MSE, ranking, a policy-preference smoke loss, or a caller-supplied upstream
gradient.  Historical helper functions at the bottom are kept for older
experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class LossInterfaceOutput:
    loss_name: str
    loss_value: float
    logits: torch.Tensor
    output_cotangent: torch.Tensor
    direction_source: str = "train_stream_loss_interface"
    uses_validation_test_future_query: int = 0
    audit_metrics_used_for_direction: int = 0

    def to_row(self) -> dict[str, Any]:
        return {
            "loss_name": self.loss_name,
            "loss_value": self.loss_value,
            "direction_source": self.direction_source,
            "uses_validation_test_future_query": self.uses_validation_test_future_query,
            "audit_metrics_used_for_direction": self.audit_metrics_used_for_direction,
            "cotangent_norm": float(torch.linalg.vector_norm(self.output_cotangent.detach()).item()),
            "logit_shape": "x".join(str(x) for x in self.logits.shape),
        }


class LossInterface(ABC):
    """Minimal contract for arbitrary differentiable task losses."""

    @abstractmethod
    def value(self, logits: torch.Tensor, target_or_task_data: Any = None) -> torch.Tensor:
        raise NotImplementedError

    def cotangent(self, logits: torch.Tensor, target_or_task_data: Any = None) -> torch.Tensor:
        work = logits.detach().clone().requires_grad_(True)
        loss = self.value(work, target_or_task_data)
        grad = torch.autograd.grad(loss, work, retain_graph=False, create_graph=False)[0]
        return grad.detach()

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


def _as_tensor(data: Any, *, like: torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(data, device=like.device, dtype=like.dtype)


@dataclass
class GenericUpstreamCotangent(LossInterface):
    """Direct upstream cotangent adapter.

    ``value`` is the linear functional whose derivative equals the supplied
    upstream cotangent.  This is the cleanest way to benchmark VJP/update cost
    without making any claim about a task loss.
    """

    default_delta: torch.Tensor | None = None
    adapter_name: str = "Delta-Gaussian"

    def _delta(self, logits: torch.Tensor, target_or_task_data: Any = None) -> torch.Tensor:
        data = target_or_task_data
        if isinstance(data, dict):
            data = data.get("delta", data.get("cotangent", None))
        if data is None:
            data = self.default_delta
        if data is None:
            raise ValueError("GenericUpstreamCotangent requires a delta tensor")
        return _as_tensor(data, like=logits)

    def value(self, logits: torch.Tensor, target_or_task_data: Any = None) -> torch.Tensor:
        delta = self._delta(logits, target_or_task_data)
        return (logits.float() * delta.float()).sum()

    def cotangent(self, logits: torch.Tensor, target_or_task_data: Any = None) -> torch.Tensor:
        return self._delta(logits, target_or_task_data).detach().to(device=logits.device, dtype=logits.dtype)

    def name(self) -> str:
        return self.adapter_name


class ClassificationCEAdapter(LossInterface):
    def value(self, logits: torch.Tensor, target_or_task_data: Any = None) -> torch.Tensor:
        if target_or_task_data is None:
            raise ValueError("ClassificationCEAdapter requires integer labels")
        labels = torch.as_tensor(target_or_task_data, device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits.float(), labels)

    def name(self) -> str:
        return "Delta-LossCEAdapter"


class RegressionMSEAdapter(LossInterface):
    def value(self, logits: torch.Tensor, target_or_task_data: Any = None) -> torch.Tensor:
        if target_or_task_data is None:
            target = torch.zeros_like(logits)
        else:
            target = _as_tensor(target_or_task_data, like=logits)
        return F.mse_loss(logits.float(), target.float())

    def name(self) -> str:
        return "Delta-MSEAdapter"


class PairwiseRankingAdapter(LossInterface):
    def value(self, logits: torch.Tensor, target_or_task_data: Any = None) -> torch.Tensor:
        scores = logits.float().mean(dim=-1)
        n = int(scores.numel())
        if isinstance(target_or_task_data, dict) and "pairs" in target_or_task_data:
            pairs = torch.as_tensor(target_or_task_data["pairs"], device=logits.device, dtype=torch.long)
            sign = torch.as_tensor(target_or_task_data.get("sign", torch.ones(len(pairs))), device=logits.device, dtype=scores.dtype)
        else:
            half = max(1, n // 2)
            right = torch.arange(half, min(n, 2 * half), device=logits.device)
            left = torch.arange(0, int(right.numel()), device=logits.device)
            pairs = torch.stack([left, right], dim=1) if right.numel() else torch.zeros((0, 2), device=logits.device, dtype=torch.long)
            sign = torch.ones(int(pairs.shape[0]), device=logits.device, dtype=scores.dtype)
        if int(pairs.numel()) == 0:
            return scores.sum() * 0.0
        margin = sign * (scores[pairs[:, 0]] - scores[pairs[:, 1]])
        return F.softplus(-margin).mean()

    def name(self) -> str:
        return "Delta-RankingAdapter"


class PolicyPreferenceAdapter(LossInterface):
    def __init__(self, beta: float = 0.1) -> None:
        self.beta = float(beta)

    def value(self, logits: torch.Tensor, target_or_task_data: Any = None) -> torch.Tensor:
        scores = logits.float().mean(dim=-1)
        n = int(scores.numel())
        half = max(1, n // 2)
        chosen = torch.arange(0, min(half, n), device=logits.device)
        rejected = torch.arange(half, min(n, half + int(chosen.numel())), device=logits.device)
        chosen = chosen[: int(rejected.numel())]
        if isinstance(target_or_task_data, dict):
            chosen = torch.as_tensor(target_or_task_data.get("chosen", chosen), device=logits.device, dtype=torch.long)
            rejected = torch.as_tensor(target_or_task_data.get("rejected", rejected), device=logits.device, dtype=torch.long)
        if int(chosen.numel()) == 0 or int(rejected.numel()) == 0:
            return scores.sum() * 0.0
        diff = scores[chosen] - scores[rejected]
        return -F.logsigmoid(self.beta * diff).mean()

    def name(self) -> str:
        return "Delta-PolicyPreferenceAdapter"


def stable_random_delta_like(logits: torch.Tensor, *, seed: int = 2211, kind: str = "gaussian") -> torch.Tensor:
    if kind == "stable":
        raw = torch.sin(torch.arange(logits.numel(), device=logits.device, dtype=logits.dtype)).reshape_as(logits)
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        raw = torch.randn(logits.shape, generator=gen, dtype=logits.detach().cpu().dtype).to(device=logits.device, dtype=logits.dtype)
    return raw / torch.linalg.vector_norm(raw).clamp_min(1.0e-8) * (float(logits.numel()) ** 0.5)


def loss_interface_unit_tests() -> list[dict[str, Any]]:
    torch.manual_seed(2211)
    logits = torch.randn(18, 5)
    labels = torch.arange(18) % 5
    mse_target = torch.tanh(logits.detach())
    random_delta = stable_random_delta_like(logits, seed=2211, kind="gaussian")
    stable_delta = stable_random_delta_like(logits, seed=2211, kind="stable")
    adapters: list[tuple[LossInterface, Any]] = [
        (GenericUpstreamCotangent(random_delta, "Delta-Gaussian"), None),
        (GenericUpstreamCotangent(stable_delta, "Delta-StableRandom"), None),
        (ClassificationCEAdapter(), labels),
        (RegressionMSEAdapter(), mse_target),
        (PairwiseRankingAdapter(), None),
        (PolicyPreferenceAdapter(), None),
    ]
    rows: list[dict[str, Any]] = []
    for adapter, data in adapters:
        value = adapter.value(logits, data)
        delta = adapter.cotangent(logits, data)
        finite = bool(torch.isfinite(value).all().item()) and bool(torch.isfinite(delta).all().item())
        rows.append(
            {
                "adapter": adapter.name(),
                "value": float(value.detach().item()),
                "cotangent_shape": "x".join(str(x) for x in delta.shape),
                "cotangent_norm": float(torch.linalg.vector_norm(delta).item()),
                "upstream_cotangent_contract_pass": int(delta.shape == logits.shape and finite),
                "uses_validation_test_future_query": 0,
                "audit_metrics_used_for_direction": 0,
                "pass": int(delta.shape == logits.shape and finite),
            }
        )
    return rows


def cross_entropy_cotangent(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> LossInterfaceOutput:
    logits = model(x).float()
    loss = F.cross_entropy(logits, y)
    probs = torch.softmax(logits.detach(), dim=1)
    target = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=probs.dtype)
    cotangent = (probs - target) / max(1, int(y.numel()))
    return LossInterfaceOutput("cross_entropy", float(loss.detach().item()), logits, cotangent)


def brier_cotangent(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> LossInterfaceOutput:
    logits = model(x).float()
    probs = torch.softmax(logits, dim=1)
    target = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=probs.dtype)
    loss = (probs - target).square().sum(dim=1).mean()
    grad_probs = 2.0 * (probs.detach() - target) / max(1, int(y.numel()))
    # Convert probability-space cotangent into a stable logit-space proxy.
    cotangent = probs.detach() * (grad_probs - (grad_probs * probs.detach()).sum(dim=1, keepdim=True))
    return LossInterfaceOutput("brier", float(loss.detach().item()), logits, cotangent)


__all__ = [
    "ClassificationCEAdapter",
    "GenericUpstreamCotangent",
    "LossInterface",
    "LossInterfaceOutput",
    "PairwiseRankingAdapter",
    "PolicyPreferenceAdapter",
    "RegressionMSEAdapter",
    "brier_cotangent",
    "cross_entropy_cotangent",
    "loss_interface_unit_tests",
    "stable_random_delta_like",
]
