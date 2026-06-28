"""Constrained Meta-FU operator-mixture gradient transform.

The controller output is a continuous mixture over a small analytic operator
library.  It never emits parameter deltas and is intended to be called only
from an optimizer-owned ``step()`` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import torch


META_FU_OPERATOR_NAMES = (
    "A0_identity",
    "A1_credit_only",
    "A2_self_geometry",
    "A3_static_credit_self",
    "A4_state_transition",
    "A5_oet_tangent",
    "A6_tail_debt_safe",
)


@dataclass
class OperatorMixtureConfig:
    mixture: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    trust: float = 0.12
    credit_cosine: float = 0.0
    tail_safety: float = 0.35
    state_scale: float = 0.25
    official_frozen: int = 1
    train_only_controller_inputs: int = 1
    no_dataset_or_seed_input: int = 1
    controller_mode: str = "fixed_continuous_mixture"
    normalized_step: float = 0.0
    total_steps: int = 1
    learning_rate: float = 0.0


@dataclass
class OperatorMixtureTrace:
    rows: list[dict[str, Any]] = field(default_factory=list)
    transform_calls: int = 0
    trust_clip_events: int = 0
    trust_clip_checks: int = 0
    controller_parameter_count: int = 0


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def normalize_mixture(weights: Iterable[float]) -> tuple[float, ...]:
    vals = [max(0.0, _finite(v)) for v in list(weights)[: len(META_FU_OPERATOR_NAMES)]]
    if len(vals) < len(META_FU_OPERATOR_NAMES):
        vals.extend([0.0] * (len(META_FU_OPERATOR_NAMES) - len(vals)))
    total = sum(vals)
    if not math.isfinite(total) or total <= 0.0:
        vals = [1.0] + [0.0] * (len(META_FU_OPERATOR_NAMES) - 1)
        total = 1.0
    return tuple(float(v / total) for v in vals)


def mixture_entropy(weights: Iterable[float]) -> float:
    vals = normalize_mixture(weights)
    ent = 0.0
    for val in vals:
        if val > 0.0:
            ent -= val * math.log(max(val, 1.0e-12))
    return float(ent / math.log(len(META_FU_OPERATOR_NAMES)))


def _safe_norm(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().norm().clamp_min(1.0e-12)


def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(int(tensor.shape[0]), -1)


def analytic_operator_bank(
    grad: torch.Tensor,
    param: torch.Tensor,
    *,
    credit_cosine: float = 0.0,
    prev_grad: torch.Tensor | None = None,
    tail_safety: float = 0.35,
    state_scale: float = 0.25,
) -> list[torch.Tensor]:
    """Return A0-A6 analytic transforms for one gradient tensor."""

    g = grad
    g_float = g.detach().float()
    raw_norm = _safe_norm(g)
    credit = max(-1.0, min(1.0, _finite(credit_cosine)))
    credit_gain = 1.0 + 0.20 * max(credit, 0.0) - 0.25 * max(-credit, 0.0)
    a0 = g
    a1 = g * float(max(0.15, min(1.25, credit_gain)))

    # Self-geometry: damp radial movement and scale by weight/gradient ratio.
    p_float = param.detach().float()
    p_norm = p_float.norm().clamp_min(1.0e-12)
    radial_coeff = (g_float * p_float).sum() / (p_norm.square())
    radial = (radial_coeff.to(dtype=g.dtype, device=g.device) * param).to(dtype=g.dtype)
    tangent = g - radial
    ratio = float((p_norm / raw_norm).clamp(0.25, 4.0).item())
    a2 = tangent + 0.30 * radial
    a2 = a2 * math.sqrt(ratio)

    a3 = 0.50 * a1 + 0.50 * a2

    if prev_grad is not None and tuple(prev_grad.shape) == tuple(g.shape):
        drift = g - prev_grad.to(device=g.device, dtype=g.dtype)
        a4 = g + float(state_scale) * drift
    else:
        a4 = a3

    # OET/Pion-like tangent: remove radial component.
    a5 = tangent

    # Tail-debt-safe: shrink very large coordinates while keeping cone direction.
    mean_abs = g_float.abs().mean().clamp_min(1.0e-12)
    clip = float((mean_abs * (2.0 + 4.0 * max(0.0, min(1.0, tail_safety)))).item())
    a6 = g.clamp(min=-clip, max=clip)
    safe_norm = _safe_norm(a6)
    if float(safe_norm.item()) > 0.0:
        a6 = a6 * (raw_norm.to(device=g.device, dtype=g.dtype) / safe_norm.to(device=g.device, dtype=g.dtype)).clamp(0.25, 1.0)

    return [a0, a1, a2, a3, a4, a5, a6]


def mix_operator_bank(bank: list[torch.Tensor], weights: Iterable[float]) -> torch.Tensor:
    vals = normalize_mixture(weights)
    out = torch.zeros_like(bank[0])
    for w, item in zip(vals, bank):
        if w:
            out = out + float(w) * item
    return out


class FixedOperatorMixtureController(torch.nn.Module):
    """Frozen zero-parameter controller for official smoke/runtime rows."""

    def __init__(self, mixture: Iterable[float]) -> None:
        super().__init__()
        self.register_buffer("mixture", torch.tensor(normalize_mixture(mixture), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.mixture


class MetaFUOperatorMixtureOperator:
    """Continuous A0-A6 operator mixture for optimizer-owned gradient transforms."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        config: OperatorMixtureConfig | None = None,
        *,
        param_names: dict[int, str] | None = None,
    ) -> None:
        self.params = [p for p in params if getattr(p, "requires_grad", False)]
        self.param_ids = {id(p) for p in self.params}
        self.param_names = dict(param_names or {})
        self.config = config or OperatorMixtureConfig()
        self.controller = FixedOperatorMixtureController(self.config.mixture)
        self.controller.eval()
        for param in self.controller.parameters():
            param.requires_grad_(False)
        self.trace = OperatorMixtureTrace()
        self._inside_optimizer_step = False
        self._prev_grad: dict[int, torch.Tensor] = {}

    def owns_param(self, param: torch.nn.Parameter) -> bool:
        return id(param) in self.param_ids

    def set_inside_optimizer_step(self, value: bool) -> None:
        self._inside_optimizer_step = bool(value)

    def set_progress(self, step: int, total_steps: int, learning_rate: float, weight_decay: float = 0.0) -> None:
        self.config.normalized_step = float(step) / max(1.0, float(total_steps))
        self.config.total_steps = int(max(1, total_steps))
        self.config.learning_rate = float(learning_rate)

    def set_train_metrics(self, loss_value: float) -> None:
        del loss_value

    def transform(self, grad: torch.Tensor, param: torch.nn.Parameter, group: dict[str, Any] | None = None) -> torch.Tensor:
        del group
        if not self._inside_optimizer_step:
            return grad
        weights = normalize_mixture(self.controller().detach().cpu().tolist())
        bank = analytic_operator_bank(
            grad,
            param,
            credit_cosine=float(self.config.credit_cosine),
            prev_grad=self._prev_grad.get(id(param)),
            tail_safety=float(self.config.tail_safety),
            state_scale=float(self.config.state_scale),
        )
        mixed = mix_operator_bank(bank, weights)
        self._prev_grad[id(param)] = grad.detach().clone()
        raw_norm = float(grad.detach().float().norm().item())
        delta = mixed - grad
        delta_norm = float(delta.detach().float().norm().item())
        max_delta = max(1.0e-12, float(self.config.trust) * max(raw_norm, 1.0e-12))
        self.trace.trust_clip_checks += 1
        if math.isfinite(delta_norm) and delta_norm > max_delta:
            mixed = grad + delta * (max_delta / max(delta_norm, 1.0e-12))
            self.trace.trust_clip_events += 1
        self.trace.transform_calls += 1
        row = {
            "refresh_id": self.trace.transform_calls,
            "param_id": id(param),
            "param_name": self.param_names.get(id(param), ""),
            "param_numel": int(param.numel()),
            "normalized_step": float(self.config.normalized_step),
            "operator_mixture_entropy": mixture_entropy(weights),
            "meta_controller_frozen_on_meta_test": int(self.config.official_frozen),
            "meta_controller_train_only_inputs": int(self.config.train_only_controller_inputs),
            "meta_controller_no_dataset_seed_input": int(self.config.no_dataset_or_seed_input),
        }
        for name, weight in zip(META_FU_OPERATOR_NAMES, weights):
            row[f"weight_{name}"] = float(weight)
        self.trace.rows.append(row)
        return mixed

    def controller_trace_rows(self) -> list[dict[str, Any]]:
        return list(self.trace.rows)

    def diagnostics(self) -> dict[str, Any]:
        weights = normalize_mixture(self.controller().detach().cpu().tolist())
        return {
            "metafu_operator_mixture_transform_calls": int(self.trace.transform_calls),
            "meta_controller_mode": str(self.config.controller_mode),
            "meta_controller_parameter_count": int(self.trace.controller_parameter_count),
            "meta_controller_frozen_on_meta_test": int(self.config.official_frozen),
            "meta_controller_train_only_inputs": int(self.config.train_only_controller_inputs),
            "meta_controller_no_dataset_seed_input": int(self.config.no_dataset_or_seed_input),
            "operator_mixture_entropy": mixture_entropy(weights),
            "operator_mixture_TV_norm": 0.0,
            "controller_trust_clip_rate": float(self.trace.trust_clip_events / max(1, self.trace.trust_clip_checks)),
            **{f"operator_weight_{name}": float(weight) for name, weight in zip(META_FU_OPERATOR_NAMES, weights)},
        }


__all__ = [
    "META_FU_OPERATOR_NAMES",
    "OperatorMixtureConfig",
    "FixedOperatorMixtureController",
    "MetaFUOperatorMixtureOperator",
    "analytic_operator_bank",
    "mix_operator_bank",
    "mixture_entropy",
    "normalize_mixture",
]
