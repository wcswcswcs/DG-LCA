"""Train-only witnessed gradient operator for v22.55.

The operator consumes supervised cohort gradients and transforms the already
computed task gradient inside an optimizer wrapper. It does not create an
auxiliary loss and it never writes parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterable

import torch


@dataclass
class WGOConfig:
    variant: str = "diag_snr"
    control: str = "none"
    lambda_var: float = 0.25
    lambda_debt: float = 0.25
    ema_beta: float = 0.90
    eps: float = 1.0e-8
    block_size: int = 256
    lowrank_rank: int = 4
    random_seed: int = 0
    safety_margin_q: float = 0.10
    clip_min: float = 0.0
    clip_max: float = 1.0


@dataclass
class WGOTrace:
    observe_calls: int = 0
    transform_calls: int = 0
    gradients_transformed: int = 0
    optimizer_owned_transform_pass: int = 0
    observe_time_ms: float = 0.0
    transform_time_ms: float = 0.0
    gate_mean: float = 1.0
    gate_std: float = 0.0
    gate_entropy: float = 0.0
    gate_active_fraction: float = 1.0
    gate_zero_fraction: float = 0.0
    gate_one_fraction: float = 1.0
    source_witness_transfer_mean: float = 0.0
    source_witness_transfer_lcb: float = 0.0
    transfer_variance: float = 0.0
    noise_attenuation_rate: float = 0.0
    witness_gain_cvar25: float = 0.0
    held_train_witness_gain: float = 0.0
    gate_class_count_correlation: float = 0.0
    gate_loss_correlation: float = 0.0
    gate_margin_correlation: float = 0.0
    gate_label_correlation: float = 0.0
    deleak_projection_fraction: float = 0.0
    deleak_nuisance_std: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "wgo_observe_calls": self.observe_calls,
            "wgo_transform_calls": self.transform_calls,
            "wgo_gradients_transformed": self.gradients_transformed,
            "optimizer_owned_gradient_transform_pass": self.optimizer_owned_transform_pass,
            "wgo_stats_ms": self.observe_time_ms,
            "wgo_transform_ms": self.transform_time_ms,
            "gate_mean": self.gate_mean,
            "gate_std": self.gate_std,
            "gate_entropy": self.gate_entropy,
            "gate_active_fraction": self.gate_active_fraction,
            "gate_zero_fraction": self.gate_zero_fraction,
            "gate_one_fraction": self.gate_one_fraction,
            "source_witness_transfer_mean": self.source_witness_transfer_mean,
            "source_witness_transfer_LCB": self.source_witness_transfer_lcb,
            "transfer_variance": self.transfer_variance,
            "noise_attenuation_rate": self.noise_attenuation_rate,
            "witness_gain_CVaR25": self.witness_gain_cvar25,
            "held_train_witness_gain": self.held_train_witness_gain,
            "gate_class_count_correlation": self.gate_class_count_correlation,
            "gate_loss_correlation": self.gate_loss_correlation,
            "gate_margin_correlation": self.gate_margin_correlation,
            "gate_label_correlation": self.gate_label_correlation,
            "deleak_projection_fraction": self.deleak_projection_fraction,
            "deleak_nuisance_std": self.deleak_nuisance_std,
        }


def _flatten_grads(grads: Iterable[torch.Tensor | None], params: list[torch.nn.Parameter]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            parts.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
        else:
            parts.append(grad.detach().reshape(-1))
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2 or b.numel() < 2:
        return 0.0
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.numel() != b.numel():
        n = min(int(a.numel()), int(b.numel()))
        a = a[:n]
        b = b[:n]
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm().clamp_min(1.0e-12) * b.norm().clamp_min(1.0e-12)
    out = float((a.dot(b) / denom).item())
    return out if math.isfinite(out) else 0.0


def _lcb95(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    tensor = torch.tensor(clean, dtype=torch.float64)
    return float(tensor.mean().item() - 1.96 * tensor.std(unbiased=True).item() / math.sqrt(len(clean)))


def _entropy01(gates: torch.Tensor, eps: float) -> float:
    if gates.numel() == 0:
        return 0.0
    clamp_eps = max(float(eps), 1.0e-6)
    p = gates.detach().float().clamp(clamp_eps, 1.0 - clamp_eps)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return float(ent.mean().item())


def _standardized(values: list[float], device: torch.device, dtype: torch.dtype, eps: float) -> torch.Tensor | None:
    if not values:
        return None
    tensor = torch.tensor(values, device=device, dtype=dtype)
    if tensor.numel() < 2:
        return None
    centered = tensor - tensor.mean()
    std = centered.std(unbiased=False)
    if float(std.item()) <= float(eps):
        return None
    return centered / std.clamp_min(float(eps))


def _class_balance_metric(cohort_labels: list[torch.Tensor] | None, device: torch.device) -> list[float]:
    if not cohort_labels:
        return []
    labels_all = [labels.detach().reshape(-1).to(device=device) for labels in cohort_labels if labels.numel() > 0]
    if not labels_all:
        return [0.0 for _ in cohort_labels]
    all_labels = torch.cat(labels_all)
    values, counts = torch.unique(all_labels.long(), return_counts=True)
    count_map = {int(v.item()): float(c.item()) for v, c in zip(values, counts)}
    out: list[float] = []
    for labels in cohort_labels:
        labels = labels.detach().reshape(-1).to(device=device)
        if labels.numel() == 0:
            out.append(0.0)
            continue
        inv = [1.0 / max(1.0, count_map.get(int(v.item()), 1.0)) for v in labels.long()]
        out.append(float(sum(inv) / len(inv)))
    return out


def _tensor_means(items: list[torch.Tensor] | None) -> list[float]:
    if not items:
        return []
    return [float(item.detach().float().mean().item()) if item.numel() > 0 else 0.0 for item in items]


class WitnessedGradientOperator:
    """Block/diagonal witnessed operator over supervised gradients."""

    def __init__(self, params: Iterable[torch.nn.Parameter], config: WGOConfig | None = None) -> None:
        self.params = [p for p in params if p.requires_grad]
        self._param_ids = {id(p) for p in self.params}
        self.config = config or WGOConfig()
        self._gates: dict[int, torch.Tensor] = {}
        self._ema_gates: dict[int, torch.Tensor] = {}
        self._flat_gate: torch.Tensor | None = None
        self.trace = WGOTrace()
        self._inside_optimizer_step = False
        self._generator_by_device: dict[str, torch.Generator] = {}

    def set_inside_optimizer_step(self, value: bool) -> None:
        self._inside_optimizer_step = bool(value)

    def owns_param(self, param: torch.nn.Parameter) -> bool:
        return id(param) in self._param_ids

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        gen = self._generator_by_device.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self.config.random_seed))
            self._generator_by_device[key] = gen
        return gen

    def _apply_block_structure(self, gate: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        cfg = self.config
        if gate.numel() == 0:
            return gate
        if cfg.variant == "block_lowrank":
            flat = gate.reshape(-1)
            block = max(1, int(cfg.block_size))
            padded = int(math.ceil(flat.numel() / block) * block)
            if padded != flat.numel():
                flat = torch.cat([flat, flat.new_ones(padded - flat.numel())])
            view = flat.view(-1, block)
            block_gate = view.mean(dim=1, keepdim=True).expand_as(view).reshape(-1)[: gate.numel()]
            return block_gate.reshape(shape)
        return gate

    def _control_gate(self, gate: torch.Tensor, param: torch.nn.Parameter, loss_scalar: float, margin_scalar: float) -> torch.Tensor:
        cfg = self.config
        control = cfg.control
        if control in {"", "none"}:
            return gate
        if control == "same_compute_noop_gate":
            return torch.ones_like(gate)
        gen = self._generator(gate.device)
        flat = gate.reshape(-1)
        if control in {"same_gate_distribution_random", "shuffled_witness_gate"}:
            if flat.numel() <= 1:
                return gate
            return flat[torch.randperm(flat.numel(), device=flat.device, generator=gen)].reshape_as(gate)
        if control == "same_gate_entropy_random":
            mean = float(flat.mean().item()) if flat.numel() else 1.0
            rand = torch.rand(flat.shape, device=flat.device, generator=gen, dtype=flat.dtype)
            return (rand < mean).to(flat.dtype).reshape_as(gate)
        if control == "same_block_norm_random_gate":
            rand = torch.rand(flat.shape, device=flat.device, generator=gen, dtype=flat.dtype)
            rand = (rand - rand.mean()) / rand.std(unbiased=False).clamp_min(cfg.eps)
            out = rand * flat.std(unbiased=False) + flat.mean()
            return out.clamp(cfg.clip_min, cfg.clip_max).reshape_as(gate)
        if control == "frozen_gate_from_previous_seed":
            rand = torch.rand(flat.shape, device=flat.device, generator=gen, dtype=flat.dtype)
            return (rand < float(flat.mean().clamp(0.0, 1.0).item())).to(flat.dtype).reshape_as(gate)
        if control == "inverse_class_count_gate":
            scalar = 0.5 + 0.5 * math.tanh(abs(loss_scalar))
            return torch.full_like(gate, float(max(cfg.clip_min, min(cfg.clip_max, scalar))))
        if control == "hard_loss_gate":
            scalar = 1.0 / (1.0 + math.exp(-loss_scalar))
            return torch.full_like(gate, float(max(cfg.clip_min, min(cfg.clip_max, scalar))))
        if control == "loss_rank_gate":
            scalar = 0.5 + 0.5 * math.tanh(loss_scalar - 1.0)
            return torch.full_like(gate, float(max(cfg.clip_min, min(cfg.clip_max, scalar))))
        if control == "margin_shuffle_gate":
            scalar = 1.0 / (1.0 + math.exp(margin_scalar))
            return torch.full_like(gate, float(max(cfg.clip_min, min(cfg.clip_max, scalar))))
        if control == "source_only_gate" or control == "witness_only_gate":
            return gate
        if control == "random_label_gate":
            return gate
        return gate

    def _residualize_nuisance(
        self,
        flat_stack: torch.Tensor,
        cohort_labels: list[torch.Tensor] | None,
        cohort_losses: list[torch.Tensor] | None,
        cohort_margins: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        cfg = self.config
        metrics: list[torch.Tensor] = []
        device = flat_stack.device
        dtype = flat_stack.dtype
        for values in (
            _class_balance_metric(cohort_labels, device),
            _tensor_means(cohort_losses),
            _tensor_means(cohort_margins),
        ):
            metric = _standardized(values, device, dtype, float(cfg.eps))
            if metric is not None and metric.numel() == flat_stack.shape[0]:
                metrics.append(metric)
        if not metrics:
            self.trace.deleak_projection_fraction = 0.0
            self.trace.deleak_nuisance_std = 0.0
            return flat_stack
        nuisance = torch.stack(metrics, dim=1)
        nuisance = nuisance - nuisance.mean(dim=0, keepdim=True)
        gram = nuisance.transpose(0, 1).matmul(nuisance)
        scale = torch.diagonal(gram).mean().clamp_min(float(cfg.eps))
        ridge = scale * 1.0e-3
        gram = gram + torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype) * ridge
        rhs = nuisance.transpose(0, 1).matmul(flat_stack)
        try:
            coeff = torch.linalg.solve(gram, rhs)
        except RuntimeError:
            self.trace.deleak_projection_fraction = 0.0
            self.trace.deleak_nuisance_std = 0.0
            return flat_stack
        projection = nuisance.matmul(coeff)
        if projection.shape != flat_stack.shape:
            self.trace.deleak_projection_fraction = 0.0
            self.trace.deleak_nuisance_std = 0.0
            return flat_stack
        residual = flat_stack - projection
        total_energy = flat_stack.pow(2).sum().clamp_min(float(cfg.eps))
        self.trace.deleak_projection_fraction = float((projection.pow(2).sum() / total_energy).clamp(0.0, 1.0).item())
        self.trace.deleak_nuisance_std = float(nuisance.std(unbiased=False).item())
        return residual

    def observe(
        self,
        cohort_grads: list[list[torch.Tensor | None]],
        *,
        cohort_labels: list[torch.Tensor] | None = None,
        cohort_losses: list[torch.Tensor] | None = None,
        cohort_margins: list[torch.Tensor] | None = None,
        witness_gains: list[float] | None = None,
    ) -> dict[str, float | int]:
        """Update gates from train-batch supervised cohort gradients."""

        start = time.perf_counter()
        cfg = self.config
        self.trace.observe_calls += 1
        if not cohort_grads:
            self._gates = {id(p): torch.ones_like(p) for p in self.params}
            self._flat_gate = torch.cat([g.reshape(-1) for g in self._gates.values()]) if self._gates else None
            return self.trace.as_dict()

        k = len(cohort_grads)
        flat_by_cohort = [_flatten_grads(grads, self.params) for grads in cohort_grads]
        flat_stack = torch.stack(flat_by_cohort, dim=0)
        if cfg.variant in {"residual_deleak_diag", "residual_safety_diag"}:
            flat_stack = self._residualize_nuisance(flat_stack, cohort_labels, cohort_losses, cohort_margins)
        if cfg.control == "shuffled_witness_gate" and k > 1:
            perm = torch.randperm(k, device=flat_stack.device, generator=self._generator(flat_stack.device))
            flat_stack = flat_stack[perm]

        split = max(1, k // 2)
        source = flat_stack[:split]
        witness = flat_stack[split:] if split < k else flat_stack[:split]
        if cfg.control == "source_only_gate":
            source = flat_stack[:split]
            witness = source
        elif cfg.control == "witness_only_gate":
            witness = flat_stack[split:] if split < k else flat_stack[:split]
            source = witness

        transfer_samples: list[torch.Tensor] = []
        pairs = min(int(source.shape[0]), int(witness.shape[0]))
        for i in range(max(1, pairs)):
            s = source[i % int(source.shape[0])]
            w = witness[i % int(witness.shape[0])]
            transfer_samples.append(s * w)
        transfer = torch.stack(transfer_samples, dim=0)
        mu = transfer.mean(dim=0)
        sigma = transfer.std(dim=0, unbiased=False) if transfer.shape[0] > 1 else torch.zeros_like(mu)
        debt = torch.zeros_like(mu)
        if cfg.variant in {"safety_diag", "residual_safety_diag"}:
            all_mean = flat_stack.mean(dim=0)
            conflict = (flat_stack * all_mean.unsqueeze(0) < 0.0).float().mean(dim=0)
            debt = conflict * all_mean.abs()
        raw_gate = (mu - float(cfg.lambda_var) * sigma - float(cfg.lambda_debt) * debt) / (mu.abs() + float(cfg.eps))
        flat_gate = raw_gate.clamp(float(cfg.clip_min), float(cfg.clip_max))

        gates: dict[int, torch.Tensor] = {}
        offset = 0
        loss_scalar = _mean_scalar(cohort_losses)
        margin_scalar = _mean_scalar(cohort_margins)
        for param in self.params:
            n = int(param.numel())
            gate = flat_gate[offset : offset + n].reshape_as(param).to(device=param.device, dtype=param.dtype)
            gate = self._apply_block_structure(gate, param.shape)
            gate = self._control_gate(gate, param, loss_scalar, margin_scalar)
            previous = self._ema_gates.get(id(param))
            if previous is not None and previous.shape == gate.shape:
                gate = previous.to(device=gate.device, dtype=gate.dtype).mul(float(cfg.ema_beta)).add(gate, alpha=1.0 - float(cfg.ema_beta))
            self._ema_gates[id(param)] = gate.detach().clone()
            gates[id(param)] = gate
            offset += n
        self._gates = gates
        self._flat_gate = torch.cat([g.detach().float().reshape(-1) for g in gates.values()]) if gates else None

        transfers = [float(t.mean().item()) for t in transfer_samples]
        gate_flat = self._flat_gate if self._flat_gate is not None else torch.ones(1)
        self.trace.gate_mean = float(gate_flat.mean().item())
        self.trace.gate_std = float(gate_flat.std(unbiased=False).item()) if gate_flat.numel() > 1 else 0.0
        self.trace.gate_entropy = _entropy01(gate_flat, cfg.eps)
        self.trace.gate_active_fraction = float((gate_flat > 1.0e-6).float().mean().item())
        self.trace.gate_zero_fraction = float((gate_flat <= 1.0e-6).float().mean().item())
        self.trace.gate_one_fraction = float((gate_flat >= 1.0 - 1.0e-6).float().mean().item())
        self.trace.source_witness_transfer_mean = float(torch.stack(transfer_samples).mean().item())
        self.trace.source_witness_transfer_lcb = _lcb95(transfers)
        self.trace.transfer_variance = float(torch.tensor(transfers).var(unbiased=False).item()) if len(transfers) > 1 else 0.0
        self.trace.noise_attenuation_rate = float((1.0 - gate_flat).mean().item())
        if witness_gains:
            clean = sorted(float(x) for x in witness_gains if math.isfinite(float(x)))
            if clean:
                cut = max(1, int(math.ceil(0.25 * len(clean))))
                self.trace.witness_gain_cvar25 = float(sum(clean[:cut]) / cut)
                self.trace.held_train_witness_gain = float(sum(clean) / len(clean))
        corr = self._cohort_correlations(flat_stack, cohort_labels, cohort_losses, cohort_margins)
        self.trace.gate_class_count_correlation = corr["gate_class_count_correlation"]
        self.trace.gate_loss_correlation = corr["gate_loss_correlation"]
        self.trace.gate_margin_correlation = corr["gate_margin_correlation"]
        self.trace.gate_label_correlation = corr["gate_label_correlation"]
        self.trace.observe_time_ms += (time.perf_counter() - start) * 1000.0
        return self.trace.as_dict()

    def _cohort_correlations(
        self,
        flat_stack: torch.Tensor,
        cohort_labels: list[torch.Tensor] | None,
        cohort_losses: list[torch.Tensor] | None,
        cohort_margins: list[torch.Tensor] | None,
    ) -> dict[str, float]:
        if flat_stack.numel() == 0:
            return {
                "gate_class_count_correlation": 0.0,
                "gate_loss_correlation": 0.0,
                "gate_margin_correlation": 0.0,
                "gate_label_correlation": 0.0,
            }
        mean_grad = flat_stack.mean(dim=0)
        denom = mean_grad.norm().clamp_min(1.0e-12)
        scores = (flat_stack @ mean_grad) / (flat_stack.norm(dim=1).clamp_min(1.0e-12) * denom)
        device = flat_stack.device
        class_metrics = []
        label_metrics = []
        if cohort_labels:
            all_labels = torch.cat([x.detach().reshape(-1).to(device=device) for x in cohort_labels if x.numel() > 0])
            if all_labels.numel() > 0:
                values, counts = torch.unique(all_labels.long(), return_counts=True)
                count_map = {int(v.item()): float(c.item()) for v, c in zip(values, counts)}
            else:
                count_map = {}
            for labels in cohort_labels:
                labels = labels.detach().reshape(-1).to(device=device)
                if labels.numel() == 0:
                    class_metrics.append(0.0)
                    label_metrics.append(0.0)
                else:
                    inv = [1.0 / max(1.0, count_map.get(int(v.item()), 1.0)) for v in labels.long()]
                    class_metrics.append(float(sum(inv) / len(inv)))
                    label_metrics.append(float(labels.float().mean().item()))
        losses = [float(x.detach().float().mean().item()) for x in cohort_losses] if cohort_losses else []
        margins = [float(x.detach().float().mean().item()) for x in cohort_margins] if cohort_margins else []
        def tensor_or_zero(values: list[float]) -> torch.Tensor:
            if not values:
                return torch.zeros_like(scores)
            t = torch.tensor(values, device=device, dtype=scores.dtype)
            if t.numel() != scores.numel():
                n = min(int(t.numel()), int(scores.numel()))
                t = t[:n]
            return t
        return {
            "gate_class_count_correlation": _safe_corr(scores, tensor_or_zero(class_metrics)),
            "gate_loss_correlation": _safe_corr(scores, tensor_or_zero(losses)),
            "gate_margin_correlation": _safe_corr(scores, tensor_or_zero(margins)),
            "gate_label_correlation": _safe_corr(scores, tensor_or_zero(label_metrics)),
        }

    def transform(self, grad: torch.Tensor | None, param: torch.nn.Parameter, group: dict[str, Any] | None = None) -> torch.Tensor | None:
        """Return transformed supervised gradient.

        The wrapper calls this method only from optimizer.step().
        """

        if grad is None:
            return None
        start = time.perf_counter()
        self.trace.transform_calls += 1
        if self._inside_optimizer_step:
            self.trace.optimizer_owned_transform_pass = 1
        gate = self._gates.get(id(param))
        if gate is None or gate.shape != grad.shape:
            return grad
        grad.mul_(gate.to(device=grad.device, dtype=grad.dtype))
        self.trace.gradients_transformed += 1
        self.trace.transform_time_ms += (time.perf_counter() - start) * 1000.0
        return grad

    def diagnostics(self) -> dict[str, float | int]:
        return self.trace.as_dict()


def _mean_scalar(items: list[torch.Tensor] | None) -> float:
    if not items:
        return 0.0
    vals = [float(x.detach().float().mean().item()) for x in items if x.numel() > 0]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


__all__ = ["WGOConfig", "WGOTrace", "WitnessedGradientOperator"]
