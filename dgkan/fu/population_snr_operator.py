"""Train-only population-SNR functional gradient operator.

The operator observes micro-cohort task gradients and builds a bounded PSD
gradient preconditioner. It is intended to be used behind an optimizer wrapper:
it transforms ``param.grad`` inside ``optimizer.step()`` and never writes model
parameters directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterable

import torch


@dataclass
class PopulationSNRConfig:
    variant: str = "psnr"
    control: str = "none"
    rank: int = 4
    alpha: float = 0.18
    rho: float = 0.25
    eta: float = 1.0e-3
    eps: float = 1.0e-8
    ema_beta: float = 0.90
    random_seed: int = 0
    max_norm_ratio: float = 1.35
    scalar_gate_floor: float = 0.50
    scalar_gate_cap: float = 1.20


@dataclass
class _SNRParamState:
    basis: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    positive_rank: int = 0
    snr: float = 0.0
    signal_energy_fraction: float = 0.0
    reservoir_energy_fraction: float = 1.0
    source_witness_gap: float = 0.0
    scalar_gate: float = 1.0
    psd_min: float = 1.0
    condition_number: float = 1.0


@dataclass
class PopulationSNRTrace:
    observe_calls: int = 0
    transform_calls: int = 0
    gradients_transformed: int = 0
    optimizer_owned_transform_pass: int = 0
    observe_time_ms: float = 0.0
    transform_time_ms: float = 0.0
    positive_rank_values: list[float] = field(default_factory=list)
    snr_values: list[float] = field(default_factory=list)
    signal_fraction_values: list[float] = field(default_factory=list)
    reservoir_fraction_values: list[float] = field(default_factory=list)
    source_witness_gap_values: list[float] = field(default_factory=list)
    cos_values: list[float] = field(default_factory=list)
    norm_ratio_values: list[float] = field(default_factory=list)
    psd_min_values: list[float] = field(default_factory=list)
    condition_values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "psnr_observe_calls": self.observe_calls,
            "psnr_transform_calls": self.transform_calls,
            "psnr_gradients_transformed": self.gradients_transformed,
            "optimizer_owned_gradient_transform_pass": self.optimizer_owned_transform_pass,
            "psnr_stats_ms": self.observe_time_ms,
            "psnr_transform_ms": self.transform_time_ms,
            "population_SNR_mean": _mean(self.snr_values),
            "population_SNR_positive_rank": _mean(self.positive_rank_values),
            "signal_energy_fraction": _mean(self.signal_fraction_values),
            "reservoir_energy_fraction": _mean(self.reservoir_fraction_values, 1.0),
            "source_witness_gap": _mean(self.source_witness_gap_values),
            "cos_g_Pg": _mean(self.cos_values, 1.0),
            "norm_Pg_over_norm_g": _mean(self.norm_ratio_values, 1.0),
            "operator_PSD_min": min(self.psd_min_values) if self.psd_min_values else 1.0,
            "operator_condition_number": max(self.condition_values) if self.condition_values else 1.0,
        }


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else float(default)


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    n = min(int(aa.numel()), int(bb.numel()))
    if n < 2:
        return 0.0
    aa = aa[:n] - aa[:n].mean()
    bb = bb[:n] - bb[:n].mean()
    denom = aa.norm().clamp_min(1.0e-12) * bb.norm().clamp_min(1.0e-12)
    out = float((aa.dot(bb) / denom).item())
    return out if math.isfinite(out) else 0.0


def _random_orthogonal(n: int, k: int, device: torch.device, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    mat = torch.randn(n, max(1, k), device=device, dtype=dtype, generator=gen)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q[:, :k]


class PopulationSNRPreconditioner:
    """Population-SNR preconditioner for supervised task gradients."""

    def __init__(self, params: Iterable[torch.nn.Parameter], config: PopulationSNRConfig | None = None) -> None:
        self.params = [p for p in params if getattr(p, "requires_grad", False)]
        self.config = config or PopulationSNRConfig()
        self.trace = PopulationSNRTrace()
        self._state: dict[int, _SNRParamState] = {}
        self._inside_optimizer_step = False
        self._gen_by_device: dict[str, torch.Generator] = {}

    def owns_param(self, param: torch.nn.Parameter) -> bool:
        return any(param is p for p in self.params)

    def set_inside_optimizer_step(self, value: bool) -> None:
        self._inside_optimizer_step = bool(value)

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        gen = self._gen_by_device.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self.config.random_seed) + (0 if device.type == "cpu" else 1009 * max(0, int(device.index or 0))))
            self._gen_by_device[key] = gen
        return gen

    def observe(
        self,
        cohort_grads: list[list[torch.Tensor | None]],
        *,
        cohort_labels: list[torch.Tensor] | None = None,
        cohort_losses: list[torch.Tensor] | None = None,
        cohort_margins: list[torch.Tensor] | None = None,
        optimizer_state: Any | None = None,
    ) -> dict[str, float | int]:
        del cohort_margins, optimizer_state
        start = time.perf_counter()
        self.trace.observe_calls += 1
        if not cohort_grads:
            return self.diagnostics()
        for p_idx, param in enumerate(self.params):
            grads = []
            for row in cohort_grads:
                grad = row[p_idx] if p_idx < len(row) else None
                grads.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.detach().to(device=param.device, dtype=param.dtype).reshape(-1))
            if len(grads) < 2:
                continue
            stack = torch.stack(grads, dim=0).float()
            state = self._build_state(stack, param, cohort_labels, cohort_losses)
            self._state[id(param)] = state
            self.trace.positive_rank_values.append(float(state.positive_rank))
            self.trace.snr_values.append(float(state.snr))
            self.trace.signal_fraction_values.append(float(state.signal_energy_fraction))
            self.trace.reservoir_fraction_values.append(float(state.reservoir_energy_fraction))
            self.trace.source_witness_gap_values.append(float(state.source_witness_gap))
            self.trace.psd_min_values.append(float(state.psd_min))
            self.trace.condition_values.append(float(state.condition_number))
        self.trace.observe_time_ms += (time.perf_counter() - start) * 1000.0
        return self.diagnostics()

    def _build_state(
        self,
        stack: torch.Tensor,
        param: torch.nn.Parameter,
        cohort_labels: list[torch.Tensor] | None,
        cohort_losses: list[torch.Tensor] | None,
    ) -> _SNRParamState:
        cfg = self.config
        m = int(stack.shape[0])
        n = int(stack.shape[1])
        control = str(cfg.control).lower()
        gen = self._generator(param.device)
        work = stack
        if control == "source_only_snr":
            work = stack[: max(2, m // 2)]
        elif control == "witness_only_snr":
            work = stack[max(0, m // 2) :]
            if int(work.shape[0]) < 2:
                work = stack
        elif control == "shuffled_cohort_snr":
            signs = torch.randint(0, 2, (m, 1), device=stack.device, generator=gen).float().mul_(2.0).sub_(1.0)
            work = stack * signs

        drift = work.mean(dim=0)
        centered = work - drift.unsqueeze(0)
        signal_energy = float(drift.dot(drift).item())
        diffusion_energy = float(centered.square().sum(dim=1).mean().item()) if int(centered.numel()) else 0.0
        snr = signal_energy / max(float(cfg.eps), diffusion_energy)
        total_energy = signal_energy + diffusion_energy + float(cfg.eps)
        signal_fraction = signal_energy / total_energy
        reservoir_fraction = diffusion_energy / total_energy
        first = stack[: max(1, m // 2)].mean(dim=0)
        second = stack[max(1, m // 2) :].mean(dim=0) if m > 1 else first
        source_witness_gap = max(0.0, 1.0 - _safe_corr(first, second))

        cov = centered.transpose(0, 1).matmul(centered) / max(1, int(work.shape[0]) - 1)
        a_mat = drift[:, None].matmul(drift[None, :]) - float(cfg.rho) * cov
        a_mat = 0.5 * (a_mat + a_mat.transpose(0, 1))
        try:
            vals, vecs = torch.linalg.eigh(a_mat)
        except Exception:
            return _SNRParamState(snr=snr, signal_energy_fraction=signal_fraction, reservoir_energy_fraction=reservoir_fraction, source_witness_gap=source_witness_gap)
        order = torch.argsort(vals, descending=True)
        vals = vals[order]
        vecs = vecs[:, order]
        positive = vals > float(cfg.eps)
        pos_count = int(positive.sum().item())
        k = max(0, min(int(cfg.rank), pos_count, n))
        if control == "same_compute_noop":
            k = 0
        if control == "same_rank_random_psd" and pos_count > 0:
            k = max(1, min(int(cfg.rank), pos_count, n))
        if k <= 0:
            scalar = self._scalar_gate(control, cohort_labels, cohort_losses)
            return _SNRParamState(
                positive_rank=0,
                snr=snr,
                signal_energy_fraction=signal_fraction,
                reservoir_energy_fraction=reservoir_fraction,
                source_witness_gap=source_witness_gap,
                scalar_gate=scalar,
            )
        pos_vals = vals[:k].clamp_min(0.0)
        basis = vecs[:, :k].to(device=param.device, dtype=param.dtype)
        if control in {"same_spectrum_random_psd", "same_rank_random_psd"}:
            basis = _random_orthogonal(n, k, param.device, param.dtype, gen)
        if control == "same_rank_random_psd":
            pos_vals = torch.full_like(pos_vals, float(pos_vals.mean().item()) if int(pos_vals.numel()) else 1.0)
        scale = (pos_vals / (pos_vals + float(cfg.eta))).to(device=param.device, dtype=param.dtype)
        cond = float(((1.0 + float(cfg.alpha) * scale.max()).detach() / (1.0 + float(cfg.alpha) * scale.min()).detach().clamp_min(float(cfg.eps))).item()) if int(scale.numel()) else 1.0
        return _SNRParamState(
            basis=basis,
            scale=scale,
            positive_rank=k,
            snr=snr,
            signal_energy_fraction=signal_fraction,
            reservoir_energy_fraction=reservoir_fraction,
            source_witness_gap=source_witness_gap,
            scalar_gate=self._scalar_gate(control, cohort_labels, cohort_losses),
            psd_min=1.0,
            condition_number=cond,
        )

    def _scalar_gate(
        self,
        control: str,
        cohort_labels: list[torch.Tensor] | None,
        cohort_losses: list[torch.Tensor] | None,
    ) -> float:
        cfg = self.config
        gate = 1.0
        if control in {"hard_loss_gate", "loss_rank_gate"} and cohort_losses:
            means = [float(x.detach().float().mean().item()) for x in cohort_losses if int(x.numel()) > 0]
            if means:
                spread = max(means) - min(means)
                base = sum(means) / len(means)
                gate = 1.0 + 0.10 * spread / max(abs(base), float(cfg.eps))
        elif control == "inverse_class_count_gate" and cohort_labels:
            labels = torch.cat([x.detach().long().reshape(-1).cpu() for x in cohort_labels if int(x.numel()) > 0], dim=0) if cohort_labels else torch.empty(0, dtype=torch.long)
            if int(labels.numel()) > 0:
                counts = torch.bincount(labels)
                nz = counts[counts > 0].float()
                if int(nz.numel()) > 1:
                    gate = 1.0 + 0.05 * float((nz.max() / nz.min().clamp_min(1.0) - 1.0).item())
        return float(max(float(cfg.scalar_gate_floor), min(float(cfg.scalar_gate_cap), gate)))

    def transform(self, grad: torch.Tensor, param: torch.nn.Parameter, group: dict[str, Any] | None = None) -> torch.Tensor:
        del group
        start = time.perf_counter()
        self.trace.transform_calls += 1
        if self._inside_optimizer_step:
            self.trace.optimizer_owned_transform_pass = 1
        state = self._state.get(id(param))
        if state is None or state.basis is None or state.scale is None or str(self.config.control).lower() == "same_compute_noop":
            self.trace.transform_time_ms += (time.perf_counter() - start) * 1000.0
            return grad
        flat = grad.detach().reshape(-1)
        basis = state.basis.to(device=grad.device, dtype=grad.dtype)
        scale = state.scale.to(device=grad.device, dtype=grad.dtype)
        proj = basis.transpose(0, 1).matmul(flat)
        delta = basis.matmul(scale * proj)
        out_flat = flat + float(self.config.alpha) * float(state.scalar_gate) * delta
        norm_g = flat.norm().clamp_min(float(self.config.eps))
        ratio = float((out_flat.norm() / norm_g).item())
        if ratio > float(self.config.max_norm_ratio):
            out_flat = flat + (out_flat - flat) * (float(self.config.max_norm_ratio) - 1.0) / max(float(self.config.eps), ratio - 1.0)
            ratio = float((out_flat.norm() / norm_g).item())
        cos = float((flat.dot(out_flat) / (norm_g * out_flat.norm().clamp_min(float(self.config.eps)))).item())
        self.trace.cos_values.append(cos if math.isfinite(cos) else 0.0)
        self.trace.norm_ratio_values.append(ratio if math.isfinite(ratio) else 1.0)
        self.trace.gradients_transformed += 1
        self.trace.transform_time_ms += (time.perf_counter() - start) * 1000.0
        return out_flat.reshape_as(grad)

    def diagnostics(self) -> dict[str, Any]:
        return self.trace.as_dict()


__all__ = ["PopulationSNRConfig", "PopulationSNRPreconditioner", "PopulationSNRTrace"]
