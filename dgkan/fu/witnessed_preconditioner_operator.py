"""Task-compatible witnessed preconditioner for v22.56.

The operator consumes train-only source/witness cohort gradients and transforms
the supervised task gradient inside an optimizer wrapper. It never constructs an
auxiliary loss and it never mutates model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterable

import torch


@dataclass
class TCWPConfig:
    variant: str = "lowrank"
    control: str = "none"
    rank: int = 4
    alpha: float = 0.25
    beta: float = 0.25
    eta: float = 1.0e-3
    ema_beta: float = 0.90
    eps: float = 1.0e-8
    c_min: float = 0.20
    source_fraction: float = 0.50
    random_seed: int = 0
    scalar_gate_floor: float = 0.25
    scalar_gate_cap: float = 1.25
    nuisance_residualize: bool = False
    crossfit_folds: bool = False
    temporal_ema: bool = False
    fast_apply: bool = False
    diagnostic_interval: int = 1
    cpu_build: bool = False
    safety_attenuate: bool = False
    safety_strength: float = 1.0
    safety_floor: float = 0.25
    radial_safety_attenuate: bool = False
    radial_beta: float = 0.60


@dataclass
class _BlockState:
    kind: str
    basis: torch.Tensor | None
    scales: torch.Tensor | None
    psd_min_eigenvalue: float
    condition_number: float
    lowrank_energy_fraction: float
    source_witness_transfer_gain: float
    noise_attenuation_rate: float
    scalar_gate: float = 1.0
    fold_transfers: tuple[float, ...] = ()


@dataclass
class TCWPTrace:
    observe_calls: int = 0
    transform_calls: int = 0
    gradients_transformed: int = 0
    optimizer_owned_transform_pass: int = 0
    observe_time_ms: float = 0.0
    transform_time_ms: float = 0.0
    operator_PSD_min_eigenvalue: float = 1.0
    operator_condition_number: float = 1.0
    cos_g_Pg: float = 1.0
    norm_Pg_over_norm_g: float = 1.0
    witness_transfer_LCB: float = 0.0
    source_witness_gap: float = 0.0
    noise_attenuation_rate: float = 0.0
    lowrank_energy_fraction: float = 0.0
    trust_projection_rate: float = 0.0
    safety_projection_rate: float = 0.0
    operator_build_ms: float = 0.0
    operator_apply_ms: float = 0.0
    operator_spectrum_entropy: float = 0.0
    operator_basis_label_correlation: float = 0.0
    operator_basis_loss_correlation: float = 0.0
    operator_basis_margin_correlation: float = 0.0
    gate_or_operator_class_count_correlation_abs: float = 0.0
    gate_or_operator_loss_correlation_abs: float = 0.0
    gate_or_operator_margin_correlation_abs: float = 0.0
    _cos_values: list[float] = field(default_factory=list)
    _norm_ratios: list[float] = field(default_factory=list)
    _transfer_values: list[float] = field(default_factory=list)
    _projection_events: int = 0
    _safety_events: int = 0
    _spectrum_values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        total = max(1, self.transform_calls)
        return {
            "tcwp_observe_calls": self.observe_calls,
            "tcwp_transform_calls": self.transform_calls,
            "tcwp_gradients_transformed": self.gradients_transformed,
            "optimizer_owned_gradient_transform_pass": self.optimizer_owned_transform_pass,
            "tcwp_stats_ms": self.observe_time_ms,
            "tcwp_transform_ms": self.transform_time_ms,
            "operator_PSD_min_eigenvalue": self.operator_PSD_min_eigenvalue,
            "operator_condition_number": self.operator_condition_number,
            "cos_g_Pg": _mean(self._cos_values, self.cos_g_Pg),
            "norm_Pg_over_norm_g": _mean(self._norm_ratios, self.norm_Pg_over_norm_g),
            "witness_transfer_LCB": _lcb95(self._transfer_values),
            "source_witness_gap": self.source_witness_gap,
            "noise_attenuation_rate": self.noise_attenuation_rate,
            "lowrank_energy_fraction": self.lowrank_energy_fraction,
            "trust_projection_rate": float(self._projection_events / total),
            "safety_projection_rate": float(self._safety_events / total),
            "operator_build_ms": self.operator_build_ms,
            "operator_apply_ms": self.operator_apply_ms,
            "operator_spectrum_entropy": self.operator_spectrum_entropy,
            "operator_basis_label_correlation": self.operator_basis_label_correlation,
            "operator_basis_loss_correlation": self.operator_basis_loss_correlation,
            "operator_basis_margin_correlation": self.operator_basis_margin_correlation,
            "gate_or_operator_class_count_correlation_abs": self.gate_or_operator_class_count_correlation_abs,
            "gate_or_operator_loss_correlation_abs": self.gate_or_operator_loss_correlation_abs,
            "gate_or_operator_margin_correlation_abs": self.gate_or_operator_margin_correlation_abs,
        }


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else float(default)


def _lcb95(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    tensor = torch.tensor(clean, dtype=torch.float64)
    return float(tensor.mean().item() - 1.96 * tensor.std(unbiased=True).item() / math.sqrt(len(clean)))


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2 or b.numel() < 2:
        return 0.0
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    n = min(int(aa.numel()), int(bb.numel()))
    aa = aa[:n] - aa[:n].mean()
    bb = bb[:n] - bb[:n].mean()
    denom = aa.norm().clamp_min(1.0e-12) * bb.norm().clamp_min(1.0e-12)
    out = float((aa.dot(bb) / denom).item())
    return out if math.isfinite(out) else 0.0


def _flatten_grads(grads: Iterable[torch.Tensor | None], params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            out.append(torch.zeros_like(param, memory_format=torch.preserve_format).detach().reshape(-1))
        else:
            out.append(grad.detach().reshape(-1))
    return out


def _class_balance_metric(cohort_labels: list[torch.Tensor] | None, device: torch.device) -> list[float]:
    if not cohort_labels:
        return []
    flat = [x.detach().reshape(-1).to(device=device) for x in cohort_labels if int(x.numel()) > 0]
    if not flat:
        return [0.0 for _ in cohort_labels]
    all_labels = torch.cat(flat).long()
    values, counts = torch.unique(all_labels, return_counts=True)
    count_map = {int(v.item()): float(c.item()) for v, c in zip(values, counts)}
    out: list[float] = []
    for labels in cohort_labels:
        labels = labels.detach().reshape(-1).to(device=device).long()
        if int(labels.numel()) == 0:
            out.append(0.0)
        else:
            vals = [1.0 / max(1.0, count_map.get(int(v.item()), 1.0)) for v in labels]
            out.append(float(sum(vals) / len(vals)))
    return out


def _tensor_means(items: list[torch.Tensor] | None) -> list[float]:
    if not items:
        return []
    return [float(x.detach().float().mean().item()) if int(x.numel()) > 0 else 0.0 for x in items]


class TaskCompatibleWitnessedPreconditioner:
    """Low-rank PSD or reservoir-removal gradient operator."""

    def __init__(self, params: Iterable[torch.nn.Parameter], config: TCWPConfig | None = None) -> None:
        self.params = [p for p in params if p.requires_grad]
        self._param_ids = {id(p) for p in self.params}
        self.config = config or TCWPConfig()
        self.trace = TCWPTrace()
        self._state: dict[int, _BlockState] = {}
        self._ema_basis_by_param: dict[int, torch.Tensor] = {}
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

    def observe(
        self,
        cohort_grads: list[list[torch.Tensor | None]],
        *,
        cohort_labels: list[torch.Tensor] | None = None,
        cohort_losses: list[torch.Tensor] | None = None,
        cohort_margins: list[torch.Tensor] | None = None,
    ) -> dict[str, float | int]:
        start = time.perf_counter()
        cfg = self.config
        self.trace.observe_calls += 1
        if not cohort_grads or not self.params:
            return self.diagnostics()
        flat_by_cohort = [_flatten_grads(grads, self.params) for grads in cohort_grads]
        n_cohorts = len(flat_by_cohort)
        split = min(max(1, int(round(n_cohorts * float(cfg.source_fraction)))), max(1, n_cohorts - 1))
        source_ids = list(range(split))
        witness_ids = list(range(split, n_cohorts)) or list(range(split))
        if cfg.control == "shuffled_source_witness_pairing":
            witness_ids = list(reversed(witness_ids))
        if cfg.control == "source_only_operator":
            witness_ids = source_ids
        if cfg.control == "witness_only_operator":
            source_ids = witness_ids

        block_states: dict[int, _BlockState] = {}
        transfers: list[float] = []
        gaps: list[float] = []
        min_eigs: list[float] = []
        conds: list[float] = []
        noise_rates: list[float] = []
        energies: list[float] = []
        spectra: list[torch.Tensor] = []
        fold_transfer_accum: list[tuple[float, ...]] = []
        cohort_metric_tensor = self._cohort_metric_tensor(cohort_labels, cohort_losses, cohort_margins)

        for block_idx, param in enumerate(self.params):
            vectors = [items[block_idx].to(device=param.device, dtype=param.dtype) for items in flat_by_cohort]
            all_vectors = torch.stack(vectors, dim=0)
            source = torch.stack([vectors[i] for i in source_ids], dim=0)
            witness = torch.stack([vectors[i] for i in witness_ids], dim=0)
            if cfg.crossfit_folds:
                safety_weights = self._cohort_safety_weights(cohort_losses, cohort_margins, param.device, param.dtype)
                state = self._crossfit_lowrank_state(param, all_vectors, cohort_metric_tensor, safety_weights)
            else:
                state = self._build_block_state(param, source, witness)
            block_states[id(param)] = state
            transfers.append(state.source_witness_transfer_gain)
            if state.fold_transfers:
                fold_transfer_accum.append(state.fold_transfers)
            gaps.append(float((source.mean(dim=0) - witness.mean(dim=0)).norm().item()))
            min_eigs.append(state.psd_min_eigenvalue)
            conds.append(state.condition_number)
            noise_rates.append(state.noise_attenuation_rate)
            energies.append(state.lowrank_energy_fraction)
            if state.scales is not None and int(state.scales.numel()) > 0:
                spectra.append(state.scales.detach().abs().float())

        self._state = block_states
        self.trace.operator_PSD_min_eigenvalue = min(min_eigs) if min_eigs else 1.0
        self.trace.operator_condition_number = max(conds) if conds else 1.0
        self.trace.source_witness_gap = _mean(gaps, 0.0)
        self.trace.noise_attenuation_rate = _mean(noise_rates, 0.0)
        self.trace.lowrank_energy_fraction = _mean(energies, 0.0)
        self.trace._transfer_values.extend(transfers)
        self.trace.witness_transfer_LCB = _lcb95(self.trace._transfer_values)
        if spectra:
            spectrum = torch.cat(spectra).clamp_min(float(cfg.eps))
            probs = spectrum / spectrum.sum().clamp_min(float(cfg.eps))
            self.trace.operator_spectrum_entropy = float((-(probs * probs.log()).sum()).item())
            self.trace._spectrum_values.extend(float(x.item()) for x in spectrum)
        self._record_nuisance_correlations(cohort_labels, cohort_losses, cohort_margins, transfers, fold_transfer_accum)
        self.trace.operator_build_ms = (time.perf_counter() - start) * 1000.0
        self.trace.observe_time_ms += self.trace.operator_build_ms
        return self.diagnostics()

    def _build_block_state(self, param: torch.nn.Parameter, source: torch.Tensor, witness: torch.Tensor) -> _BlockState:
        cfg = self.config
        control = cfg.control.lower()
        if control == "same_compute_noop":
            _ = self._lowrank_state(param, source, witness, random_basis=False)
            return _BlockState("noop", None, None, 1.0, 1.0, 0.0, 0.0, 0.0)
        if control in {"inverse_class_count_gate", "hard_loss_gate", "loss_rank_gate", "margin_shuffle_gate", "random_label_gate"}:
            source_norm = float(source.norm(dim=1).mean().item()) if int(source.numel()) > 0 else 0.0
            witness_norm = float(witness.norm(dim=1).mean().item()) if int(witness.numel()) > 0 else 0.0
            raw = 1.0 + 0.05 * math.tanh(source_norm - witness_norm)
            gate = max(float(cfg.scalar_gate_floor), min(float(cfg.scalar_gate_cap), raw))
            return _BlockState("scalar_gate", None, None, gate, gate, 0.0, source_norm - witness_norm, 0.0, gate)
        if cfg.variant in {"noise_projection", "safety_aware"} or control == "same_noise_projection_random":
            return self._noise_state(param, source, witness, random_basis=(control == "same_noise_projection_random"))
        random_basis = control in {
            "same_operator_rank_random_psd",
            "same_operator_spectrum_random_basis",
            "same_operator_trace_random",
            "same_operator_condition_random",
            "shuffled_source_witness_pairing",
        }
        return self._lowrank_state(param, source, witness, random_basis=random_basis)

    def _normalised_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(dim=1, keepdim=True).clamp_min(float(self.config.eps))

    def _orthonormal_columns(self, raw_basis: torch.Tensor) -> torch.Tensor:
        if int(raw_basis.numel()) == 0:
            return raw_basis
        gram = raw_basis.transpose(0, 1).matmul(raw_basis)
        evals, evecs = torch.linalg.eigh(gram.float())
        keep = evals > float(self.config.eps)
        if not bool(keep.any()):
            return raw_basis[:, :0]
        vals = evals[keep].to(device=raw_basis.device, dtype=raw_basis.dtype)
        vecs = evecs[:, keep].to(device=raw_basis.device, dtype=raw_basis.dtype)
        order = torch.argsort(vals, descending=True)
        vals = vals[order]
        vecs = vecs[:, order]
        return raw_basis.matmul(vecs / vals.sqrt().clamp_min(float(self.config.eps)))

    def _cohort_metric_tensor(
        self,
        cohort_labels: list[torch.Tensor] | None,
        cohort_losses: list[torch.Tensor] | None,
        cohort_margins: list[torch.Tensor] | None,
    ) -> torch.Tensor | None:
        if not self.params:
            return None
        device = self.params[0].device
        cols: list[torch.Tensor] = []
        for values in (
            _class_balance_metric(cohort_labels, device),
            _tensor_means(cohort_losses),
            _tensor_means(cohort_margins),
        ):
            if len(values) >= 2:
                col = torch.tensor(values, device=device, dtype=torch.float32)
                centered = col - col.mean()
                std = centered.std(unbiased=False)
                if float(std.item()) > float(self.config.eps):
                    cols.append(centered / std.clamp_min(float(self.config.eps)))
        if not cols:
            return None
        return torch.stack(cols, dim=1)

    def _residualize_against_metrics(self, rows: torch.Tensor, metrics: torch.Tensor | None) -> torch.Tensor:
        if metrics is None or not self.config.nuisance_residualize or int(rows.shape[0]) != int(metrics.shape[0]):
            return rows
        m = metrics.to(device=rows.device, dtype=rows.dtype)
        if int(m.numel()) == 0:
            return rows
        # Remove cohort-level class/loss/margin linear nuisance components while
        # retaining the common task-gradient component by omitting an intercept.
        coeff = torch.linalg.pinv(m.float()).to(device=rows.device, dtype=rows.dtype).matmul(rows)
        return rows - m.matmul(coeff)

    def _cohort_safety_weights(
        self,
        cohort_losses: list[torch.Tensor] | None,
        cohort_margins: list[torch.Tensor] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.config.safety_attenuate:
            return None
        loss_values = _tensor_means(cohort_losses)
        margin_values = _tensor_means(cohort_margins)
        n = max(len(loss_values), len(margin_values))
        if n < 2:
            return None
        if len(loss_values) != n:
            loss_values = [0.0 for _ in range(n)]
        if len(margin_values) != n:
            margin_values = [0.0 for _ in range(n)]
        loss = torch.tensor(loss_values, device=device, dtype=dtype)
        margin = torch.tensor(margin_values, device=device, dtype=dtype)
        loss_z = (loss - loss.mean()) / loss.std(unbiased=False).clamp_min(float(self.config.eps))
        margin_z = (margin - margin.mean()) / margin.std(unbiased=False).clamp_min(float(self.config.eps))
        risk = loss_z - margin_z
        excess = torch.relu(risk - risk.mean())
        weights = torch.exp(-float(self.config.safety_strength) * excess)
        return weights.clamp_min(float(self.config.safety_floor)).clamp_max(1.0)

    def _crossfit_lowrank_state(
        self,
        param: torch.nn.Parameter,
        vectors: torch.Tensor,
        metrics: torch.Tensor | None,
        safety_weights: torch.Tensor | None = None,
    ) -> _BlockState:
        cfg = self.config
        dtype = torch.float32 if cfg.cpu_build else param.dtype
        device = torch.device("cpu") if cfg.cpu_build else param.device
        rows = vectors.detach().to(device=device, dtype=dtype)
        rows = self._normalised_rows(rows)
        rows = self._residualize_against_metrics(rows, metrics)
        rows = self._normalised_rows(rows)
        k = int(rows.shape[0])
        if k < 2 or float(rows.norm().item()) <= float(cfg.eps):
            return _BlockState("identity", None, None, 1.0, 1.0, 0.0, 0.0, 0.0)
        weights = None
        if safety_weights is not None and int(safety_weights.numel()) == k:
            weights = safety_weights.detach().to(device=device, dtype=dtype).reshape(-1).clamp_min(float(cfg.safety_floor)).clamp_max(1.0)
        source_means: list[torch.Tensor] = []
        witness_means: list[torch.Tensor] = []
        fold_transfers: list[float] = []
        for i in range(k):
            mask = torch.ones(k, device=device, dtype=torch.bool)
            mask[i] = False
            if weights is None:
                source = rows[i]
                witness = rows[mask].mean(dim=0)
            else:
                source = rows[i] * weights[i]
                witness_w = weights[mask]
                witness = (rows[mask] * witness_w[:, None]).sum(dim=0) / witness_w.sum().clamp_min(float(cfg.eps))
            source_means.append(source)
            witness_means.append(witness)
            fold_transfers.append(float(torch.dot(source, witness).item()))
        if cfg.variant == "safety_projected_crossfit":
            risk_weights = (1.0 - weights) if weights is not None else torch.ones(k, device=device, dtype=dtype)
            debt_rows = rows * risk_weights.reshape(-1, 1)
            if float(debt_rows.norm().item()) <= float(cfg.eps):
                return _BlockState("identity", None, None, 1.0, 1.0, 0.0, _mean(fold_transfers, 0.0), 0.0, fold_transfers=tuple(fold_transfers))
            basis = self._orthonormal_columns(debt_rows.transpose(0, 1))
            take = min(max(1, int(cfg.rank)), int(basis.shape[1]))
            basis = basis[:, :take]
            if int(basis.numel()) == 0:
                return _BlockState("identity", None, None, 1.0, 1.0, 0.0, _mean(fold_transfers, 0.0), 0.0, fold_transfers=tuple(fold_transfers))
            control = cfg.control.lower()
            if control == "same_compute_noop":
                return _BlockState("noop", None, None, 1.0, 1.0, 0.0, _mean(fold_transfers, 0.0), 0.0, fold_transfers=tuple(fold_transfers))
            if control in {"same_operator_rank_random_psd", "same_operator_spectrum_random_basis", "same_operator_trace_random", "same_operator_condition_random"}:
                gen = self._generator(device)
                rand = torch.randn((int(param.numel()), take), device=device, dtype=dtype, generator=gen)
                basis = self._orthonormal_columns(rand)[:, :take]
            scales = torch.full((take,), float(cfg.beta), device=device, dtype=dtype)
            min_eig = max(0.0, 1.0 - float(cfg.beta))
            cond = 1.0 / max(min_eig, float(cfg.eps))
            energy = float((debt_rows.norm() / rows.norm().clamp_min(float(cfg.eps))).item())
            return _BlockState(
                "noise_projection",
                basis.to(device=param.device, dtype=param.dtype).detach(),
                scales.to(device=param.device, dtype=param.dtype).detach(),
                min_eig,
                cond,
                energy,
                _mean(fold_transfers, 0.0),
                energy,
                fold_transfers=tuple(fold_transfers),
            )
        raw_basis = torch.stack(source_means + witness_means, dim=1)
        if cfg.temporal_ema:
            prev = self._ema_basis_by_param.get(id(param))
            if prev is not None and tuple(prev.shape) == tuple(raw_basis.shape):
                raw_basis = float(cfg.ema_beta) * prev.to(device=device, dtype=dtype) + (1.0 - float(cfg.ema_beta)) * raw_basis
            self._ema_basis_by_param[id(param)] = raw_basis.detach()
        if float(raw_basis.norm().item()) <= float(cfg.eps):
            return _BlockState("identity", None, None, 1.0, 1.0, 0.0, _mean(fold_transfers, 0.0), 0.0, fold_transfers=tuple(fold_transfers))
        q = self._orthonormal_columns(raw_basis)
        if int(q.numel()) == 0:
            return _BlockState("identity", None, None, 1.0, 1.0, 0.0, _mean(fold_transfers, 0.0), 0.0, fold_transfers=tuple(fold_transfers))
        small_a = torch.zeros((int(q.shape[1]), int(q.shape[1])), device=device, dtype=dtype)
        for source, witness in zip(source_means, witness_means):
            qs = q.transpose(0, 1).matmul(source)
            qw = q.transpose(0, 1).matmul(witness)
            small_a = small_a + 0.5 * (qs[:, None].matmul(qw[None, :]) + qw[:, None].matmul(qs[None, :]))
        small_a = small_a / float(max(1, k))
        eigvals, eigvecs = torch.linalg.eigh(small_a.float())
        eigvals = eigvals.to(device=device, dtype=dtype)
        eigvecs = eigvecs.to(device=device, dtype=dtype)
        pos = eigvals > 0
        if not bool(pos.any()):
            return _BlockState("identity", None, None, 1.0, 1.0, 0.0, _mean(fold_transfers, 0.0), 0.0, fold_transfers=tuple(fold_transfers))
        vals = eigvals[pos]
        vecs = eigvecs[:, pos]
        order = torch.argsort(vals, descending=True)
        take = min(max(1, int(cfg.rank)), int(order.numel()))
        vals = vals[order[:take]]
        vecs = vecs[:, order[:take]]
        basis = q.matmul(vecs)
        control = cfg.control.lower()
        if control == "same_compute_noop":
            return _BlockState("noop", None, None, 1.0, 1.0, 0.0, _mean(fold_transfers, 0.0), 0.0, fold_transfers=tuple(fold_transfers))
        if control in {"same_operator_rank_random_psd", "same_operator_spectrum_random_basis", "same_operator_trace_random", "same_operator_condition_random"}:
            gen = self._generator(device)
            rand = torch.randn((int(param.numel()), take), device=device, dtype=dtype, generator=gen)
            basis = self._orthonormal_columns(rand)
        scales = float(cfg.alpha) * vals / (vals + float(cfg.eta))
        if control == "same_operator_trace_random":
            scales = torch.full_like(scales, float(scales.mean().item()) if int(scales.numel()) else 0.0)
        if control == "same_operator_condition_random" and int(scales.numel()) > 1:
            scales = torch.linspace(float(scales.max().item()), float(scales.min().item()), int(scales.numel()), device=device, dtype=dtype)
        min_eig = 1.0
        max_eig = 1.0 + float(scales.max().item()) if int(scales.numel()) else 1.0
        cond = max_eig / max(min_eig, float(cfg.eps))
        energy = float((vals.sum() / eigvals.abs().sum().clamp_min(float(cfg.eps))).item())
        return _BlockState(
            "lowrank",
            basis.to(device=param.device, dtype=param.dtype).detach(),
            scales.to(device=param.device, dtype=param.dtype).detach(),
            min_eig,
            cond,
            energy,
            _mean(fold_transfers, 0.0),
            0.0,
            fold_transfers=tuple(fold_transfers),
        )

    def _lowrank_state(self, param: torch.nn.Parameter, source: torch.Tensor, witness: torch.Tensor, *, random_basis: bool) -> _BlockState:
        cfg = self.config
        dtype = param.dtype
        device = param.device
        s = self._normalised_rows(source.detach().to(device=device, dtype=dtype))
        w = self._normalised_rows(witness.detach().to(device=device, dtype=dtype))
        s_bar = s.mean(dim=0)
        w_bar = w.mean(dim=0)
        transfer = float(torch.dot(s_bar, w_bar).item())
        raw_basis = torch.stack([s_bar, w_bar], dim=1)
        if int(raw_basis.numel()) == 0 or float(raw_basis.norm().item()) <= float(cfg.eps):
            return _BlockState("identity", None, None, 1.0, 1.0, 0.0, transfer, 0.0)
        q = self._orthonormal_columns(raw_basis)
        if int(q.numel()) == 0:
            return _BlockState("identity", None, None, 1.0, 1.0, 0.0, transfer, 0.0)
        q_s = q.transpose(0, 1).matmul(s_bar)
        q_w = q.transpose(0, 1).matmul(w_bar)
        small_a = 0.5 * (q_s[:, None].matmul(q_w[None, :]) + q_w[:, None].matmul(q_s[None, :]))
        eigvals, eigvecs = torch.linalg.eigh(small_a.float())
        eigvals = eigvals.to(device=device, dtype=dtype)
        eigvecs = eigvecs.to(device=device, dtype=dtype)
        pos = eigvals > 0
        if not bool(pos.any()):
            return _BlockState("identity", None, None, 1.0, 1.0, 0.0, transfer, 0.0)
        vals = eigvals[pos]
        vecs = eigvecs[:, pos]
        order = torch.argsort(vals, descending=True)
        take = min(max(1, int(cfg.rank)), int(order.numel()))
        vals = vals[order[:take]]
        vecs = vecs[:, order[:take]]
        basis = q.matmul(vecs)
        if random_basis:
            gen = self._generator(device)
            rand = torch.randn((int(param.numel()), take), device=device, dtype=dtype, generator=gen)
            basis = self._orthonormal_columns(rand)
        scales = float(cfg.alpha) * vals / (vals + float(cfg.eta))
        control = cfg.control.lower()
        if control == "same_operator_trace_random":
            scales = torch.full_like(scales, float(scales.mean().item()) if int(scales.numel()) else 0.0)
        if control == "same_operator_condition_random" and int(scales.numel()) > 1:
            scales = torch.linspace(float(scales.max().item()), float(scales.min().item()), int(scales.numel()), device=device, dtype=dtype)
        min_eig = 1.0
        max_eig = 1.0 + float(scales.max().item()) if int(scales.numel()) else 1.0
        cond = max_eig / max(min_eig, float(cfg.eps))
        energy = float((vals.sum() / eigvals.abs().sum().clamp_min(float(cfg.eps))).item())
        return _BlockState("lowrank", basis.detach(), scales.detach(), min_eig, cond, energy, transfer, 0.0)

    def _noise_state(self, param: torch.nn.Parameter, source: torch.Tensor, witness: torch.Tensor, *, random_basis: bool) -> _BlockState:
        cfg = self.config
        dtype = param.dtype
        device = param.device
        s = self._normalised_rows(source.detach().to(device=device, dtype=dtype))
        w = self._normalised_rows(witness.detach().to(device=device, dtype=dtype))
        transfer = float(torch.dot(s.mean(dim=0), w.mean(dim=0)).item())
        q_w = self._orthonormal_columns(w.transpose(0, 1))
        if int(q_w.numel()) == 0:
            return _BlockState("identity", None, None, 1.0, 1.0, 0.0, transfer, 0.0)
        residual = s - s.matmul(q_w).matmul(q_w.transpose(0, 1))
        if random_basis:
            gen = self._generator(device)
            residual = torch.randn(residual.shape, device=device, dtype=dtype, generator=gen)
        if float(residual.norm().item()) <= float(cfg.eps):
            return _BlockState("identity", None, None, 1.0, 1.0, 0.0, transfer, 0.0)
        _u, singular, vh = torch.linalg.svd(residual.float(), full_matrices=False)
        take = min(max(1, int(cfg.rank)), int(vh.shape[0]))
        basis = vh[:take].transpose(0, 1).to(device=device, dtype=dtype)
        scales = torch.full((take,), float(cfg.beta), device=device, dtype=dtype)
        source_energy = float((s.matmul(basis) ** 2).sum().item())
        residual_energy = float((residual ** 2).sum().item())
        noise_rate = float(source_energy / max(float((s ** 2).sum().item()), float(cfg.eps)))
        energy = float((singular[:take].sum() / singular.sum().clamp_min(float(cfg.eps))).item()) if int(singular.numel()) else 0.0
        min_eig = max(0.0, 1.0 - float(cfg.beta))
        cond = 1.0 / max(min_eig, float(cfg.eps))
        state = _BlockState("noise_projection", basis.detach(), scales.detach(), min_eig, cond, energy, transfer, noise_rate)
        if residual_energy <= float(cfg.eps):
            state.noise_attenuation_rate = 0.0
        return state

    def _record_nuisance_correlations(
        self,
        cohort_labels: list[torch.Tensor] | None,
        cohort_losses: list[torch.Tensor] | None,
        cohort_margins: list[torch.Tensor] | None,
        transfers: list[float],
        fold_transfer_accum: list[tuple[float, ...]] | None = None,
    ) -> None:
        if not self.params:
            return
        device = self.params[0].device
        if fold_transfer_accum:
            width = min(len(x) for x in fold_transfer_accum if x)
            if width >= 2:
                vals = []
                for i in range(width):
                    vals.append(_mean([x[i] for x in fold_transfer_accum], 0.0))
                t = torch.tensor(vals, device=device, dtype=torch.float32)
            else:
                t = torch.tensor(transfers, device=device, dtype=torch.float32)
        else:
            t = torch.tensor(transfers, device=device, dtype=torch.float32)
        labels = torch.tensor(_class_balance_metric(cohort_labels, device), device=device, dtype=torch.float32)
        losses = torch.tensor(_tensor_means(cohort_losses), device=device, dtype=torch.float32)
        margins = torch.tensor(_tensor_means(cohort_margins), device=device, dtype=torch.float32)
        c_label = _safe_corr(t, labels)
        c_loss = _safe_corr(t, losses)
        c_margin = _safe_corr(t, margins)
        self.trace.operator_basis_label_correlation = c_label
        self.trace.operator_basis_loss_correlation = c_loss
        self.trace.operator_basis_margin_correlation = c_margin
        self.trace.gate_or_operator_class_count_correlation_abs = abs(c_label)
        self.trace.gate_or_operator_loss_correlation_abs = abs(c_loss)
        self.trace.gate_or_operator_margin_correlation_abs = abs(c_margin)

    def transform(self, grad: torch.Tensor, param: torch.nn.Parameter, group: dict[str, Any] | None = None) -> torch.Tensor:
        start = time.perf_counter()
        self.trace.transform_calls += 1
        if not self._inside_optimizer_step:
            raise RuntimeError("TaskCompatibleWitnessedPreconditioner.transform must run inside optimizer.step()")
        state = self._state.get(id(param))
        if state is None:
            return grad
        sample_diag = int(self.config.diagnostic_interval) <= 1 or self.trace.transform_calls % int(self.config.diagnostic_interval) == 0
        if state.kind in {"identity", "noop"} and not self.config.radial_safety_attenuate:
            self.trace.gradients_transformed += 1
            self.trace.optimizer_owned_transform_pass = 1
            if sample_diag:
                self.trace._cos_values.append(1.0)
                self.trace.cos_g_Pg = 1.0
                self.trace._norm_ratios.append(1.0)
                self.trace.norm_Pg_over_norm_g = 1.0
            self.trace.operator_apply_ms = (time.perf_counter() - start) * 1000.0
            self.trace.transform_time_ms += self.trace.operator_apply_ms
            return grad
        raw = grad.detach()
        flat = grad.reshape(-1)
        if state.kind in {"identity", "noop"}:
            out = flat
        elif state.kind == "scalar_gate":
            out = flat * float(state.scalar_gate)
        elif state.kind == "lowrank" and state.basis is not None and state.scales is not None:
            basis = state.basis.to(device=flat.device, dtype=flat.dtype)
            scales = state.scales.to(device=flat.device, dtype=flat.dtype)
            coeff = basis.transpose(0, 1).matmul(flat)
            out = flat + basis.matmul(scales * coeff)
        elif state.kind == "noise_projection" and state.basis is not None and state.scales is not None:
            basis = state.basis.to(device=flat.device, dtype=flat.dtype)
            beta = float(state.scales.max().item()) if int(state.scales.numel()) else float(self.config.beta)
            coeff = basis.transpose(0, 1).matmul(flat)
            out = flat - beta * basis.matmul(coeff)
            self.trace._safety_events += int(self.config.variant == "safety_aware")
        else:
            out = flat
        out, radial_projected = self._radial_safety_project(out, param)
        self.trace._safety_events += int(radial_projected)
        if self.config.fast_apply and state.kind == "lowrank" and not radial_projected:
            projected = False
        else:
            out, projected = self._trust_project(flat, out)
        self.trace._projection_events += int(projected)
        if sample_diag:
            cos = float(torch.nn.functional.cosine_similarity(flat.reshape(1, -1), out.reshape(1, -1), dim=1, eps=float(self.config.eps)).item())
            ratio = float(out.norm().div(flat.norm().clamp_min(float(self.config.eps))).item())
            if math.isfinite(cos):
                self.trace._cos_values.append(cos)
                self.trace.cos_g_Pg = cos
            if math.isfinite(ratio):
                self.trace._norm_ratios.append(ratio)
                self.trace.norm_Pg_over_norm_g = ratio
        self.trace.gradients_transformed += 1
        self.trace.optimizer_owned_transform_pass = 1
        self.trace.operator_apply_ms = (time.perf_counter() - start) * 1000.0
        self.trace.transform_time_ms += self.trace.operator_apply_ms
        return out.reshape_as(grad).to(dtype=raw.dtype)

    def _radial_safety_project(self, proposed: torch.Tensor, param: torch.nn.Parameter) -> tuple[torch.Tensor, bool]:
        if not self.config.radial_safety_attenuate:
            return proposed, False
        p = param.detach().reshape(-1).to(device=proposed.device, dtype=proposed.dtype)
        p_norm = p.norm()
        if float(p_norm.item()) <= float(self.config.eps):
            return proposed, False
        unit = p / p_norm.clamp_min(float(self.config.eps))
        coeff = torch.dot(proposed, unit)
        beta = max(0.0, min(0.95, float(self.config.radial_beta)))
        return proposed - beta * coeff * unit, True

    def _trust_project(self, raw: torch.Tensor, proposed: torch.Tensor) -> tuple[torch.Tensor, bool]:
        cfg = self.config
        raw_norm = raw.norm().clamp_min(float(cfg.eps))
        prop_norm = proposed.norm().clamp_min(float(cfg.eps))
        cos = float(raw.dot(proposed).div(raw_norm * prop_norm).item())
        if math.isfinite(cos) and cos >= float(cfg.c_min):
            return proposed, False
        lo, hi = 0.0, 1.0
        best = raw
        for _ in range(24):
            rho = 0.5 * (lo + hi)
            mixed = (1.0 - rho) * raw + rho * proposed
            mixed_norm = mixed.norm().clamp_min(float(cfg.eps))
            mixed_cos = float(raw.dot(mixed).div(raw_norm * mixed_norm).item())
            if math.isfinite(mixed_cos) and mixed_cos >= float(cfg.c_min):
                best = mixed
                lo = rho
            else:
                hi = rho
        return best, True

    def diagnostics(self) -> dict[str, float | int]:
        return self.trace.as_dict()


__all__ = ["TCWPConfig", "TCWPTrace", "TaskCompatibleWitnessedPreconditioner"]
