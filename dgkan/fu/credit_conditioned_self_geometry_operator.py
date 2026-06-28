"""Credit-conditioned self-geometry functional update operator.

The operator transforms supervised gradients inside an optimizer wrapper.  It
does not build an auxiliary loss and it never mutates parameters directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterable

import torch


@dataclass
class CCSGConfig:
    variant: str = "full"
    control: str = "none"
    rank: int = 4
    alpha: float = 0.18
    beta: float = 0.18
    eta: float = 1.0e-3
    eps: float = 1.0e-5
    ema_beta: float = 0.90
    source_fraction: float = 0.50
    random_seed: int = 0
    c_min: float = 0.20
    use_out: bool = True
    use_in: bool = True
    self_weight: float = 1.0
    weight_spectrum_weight: float = 0.25
    gradient_cov_weight: float = 1.0
    momentum_weight: float = 0.10
    sharpness_weight: float = 0.0
    max_condition: float = 1.0e4
    scalar_gate_floor: float = 0.25
    scalar_gate_cap: float = 1.25
    diagnostic_interval: int = 1


@dataclass
class _MatrixBlockState:
    p_out: torch.Tensor | None = None
    p_in: torch.Tensor | None = None
    credit_out: torch.Tensor | None = None
    credit_in: torch.Tensor | None = None
    metric_out: torch.Tensor | None = None
    metric_in: torch.Tensor | None = None
    positive_rank_out: int = 0
    positive_rank_in: int = 0
    residual_out: float = 0.0
    residual_in: float = 0.0
    psd_min: float = 1.0
    condition_number: float = 1.0
    credit_self_alignment: float = 0.0
    credit_only_alignment: float = 0.0
    self_only_alignment: float = 0.0
    self_condition_out: float = 1.0
    self_condition_in: float = 1.0
    spectrum_entropy: float = 0.0
    update_spectrum_drift: float = 0.0
    scalar_gate: float = 1.0


@dataclass
class CCSGTrace:
    observe_calls: int = 0
    transform_calls: int = 0
    gradients_transformed: int = 0
    optimizer_owned_transform_pass: int = 0
    observe_time_ms: float = 0.0
    transform_time_ms: float = 0.0
    operator_build_ms: float = 0.0
    operator_apply_ms: float = 0.0
    credit_positive_rank_out: float = 0.0
    credit_positive_rank_in: float = 0.0
    self_condition_out: float = 1.0
    self_condition_in: float = 1.0
    generalized_eigen_residual: float = 0.0
    operator_PSD_min: float = 1.0
    operator_condition_number: float = 1.0
    credit_self_alignment: float = 0.0
    credit_only_alignment: float = 0.0
    self_only_alignment: float = 0.0
    source_witness_transfer_LCB: float = 0.0
    source_witness_gap: float = 0.0
    self_spectrum_entropy: float = 0.0
    update_spectrum_drift: float = 0.0
    momentum_alignment: float = 0.0
    OET_tangent_alignment: float = 0.0
    sharpness_proxy_delta: float = 0.0
    operator_class_corr_abs: float = 0.0
    operator_loss_corr_abs: float = 0.0
    operator_margin_corr_abs: float = 0.0
    trust_projection_rate: float = 0.0
    _cos_values: list[float] = field(default_factory=list)
    _norm_ratios: list[float] = field(default_factory=list)
    _residuals: list[float] = field(default_factory=list)
    _psd_values: list[float] = field(default_factory=list)
    _condition_values: list[float] = field(default_factory=list)
    _rank_out_values: list[float] = field(default_factory=list)
    _rank_in_values: list[float] = field(default_factory=list)
    _self_out_values: list[float] = field(default_factory=list)
    _self_in_values: list[float] = field(default_factory=list)
    _credit_self_values: list[float] = field(default_factory=list)
    _credit_only_values: list[float] = field(default_factory=list)
    _self_only_values: list[float] = field(default_factory=list)
    _transfer_values: list[float] = field(default_factory=list)
    _projection_events: int = 0

    def as_dict(self) -> dict[str, float | int]:
        total = max(1, self.transform_calls)
        return {
            "ccsg_observe_calls": self.observe_calls,
            "ccsg_transform_calls": self.transform_calls,
            "ccsg_gradients_transformed": self.gradients_transformed,
            "optimizer_owned_gradient_transform_pass": self.optimizer_owned_transform_pass,
            "ccsg_stats_ms": self.observe_time_ms,
            "ccsg_transform_ms": self.transform_time_ms,
            "operator_build_ms": self.operator_build_ms,
            "operator_apply_ms": self.operator_apply_ms,
            "credit_positive_rank_out": _mean(self._rank_out_values, self.credit_positive_rank_out),
            "credit_positive_rank_in": _mean(self._rank_in_values, self.credit_positive_rank_in),
            "self_condition_out": _mean(self._self_out_values, self.self_condition_out),
            "self_condition_in": _mean(self._self_in_values, self.self_condition_in),
            "generalized_eigen_residual": _mean(self._residuals, self.generalized_eigen_residual),
            "operator_PSD_min": min(self._psd_values) if self._psd_values else self.operator_PSD_min,
            "operator_condition_number": max(self._condition_values) if self._condition_values else self.operator_condition_number,
            "cos_g_Pg": _mean(self._cos_values, 1.0),
            "norm_Pg_over_norm_g": _mean(self._norm_ratios, 1.0),
            "credit_self_alignment": _mean(self._credit_self_values, self.credit_self_alignment),
            "credit_only_alignment": _mean(self._credit_only_values, self.credit_only_alignment),
            "self_only_alignment": _mean(self._self_only_values, self.self_only_alignment),
            "source_witness_transfer_LCB": _lcb95(self._transfer_values),
            "source_witness_gap": self.source_witness_gap,
            "self_spectrum_entropy": self.self_spectrum_entropy,
            "update_spectrum_drift": self.update_spectrum_drift,
            "momentum_alignment": self.momentum_alignment,
            "OET_tangent_alignment": self.OET_tangent_alignment,
            "sharpness_proxy_delta": self.sharpness_proxy_delta,
            "operator_class_corr_abs": self.operator_class_corr_abs,
            "operator_loss_corr_abs": self.operator_loss_corr_abs,
            "operator_margin_corr_abs": self.operator_margin_corr_abs,
            "gate_or_operator_class_count_correlation_abs": self.operator_class_corr_abs,
            "gate_or_operator_loss_correlation_abs": self.operator_loss_corr_abs,
            "gate_or_operator_margin_correlation_abs": self.operator_margin_corr_abs,
            "trust_projection_rate": float(self._projection_events / total),
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
            out.append(torch.zeros_like(param, memory_format=torch.preserve_format).detach())
        else:
            out.append(grad.detach())
    return out


def _sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def _eye(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(n, device=device, dtype=dtype)


def _normalize_symmetric(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = _sym(x)
    scale = x.detach().float().norm().clamp_min(float(eps))
    return x / scale.to(device=x.device, dtype=x.dtype)


def _condition_number_psd(x: torch.Tensor, eps: float) -> float:
    vals = torch.linalg.eigvalsh(_sym(x).float())
    vals = torch.clamp(vals, min=float(eps))
    return float((vals.max() / vals.min().clamp_min(float(eps))).item())


def _spectral_entropy(vals: torch.Tensor, eps: float = 1.0e-12) -> float:
    clean = torch.clamp(vals.detach().float().abs(), min=0.0)
    total = clean.sum()
    if float(total.item()) <= eps:
        return 0.0
    p = clean / total.clamp_min(eps)
    entropy = -(p * torch.log(p.clamp_min(eps))).sum()
    denom = math.log(max(2, int(p.numel())))
    return float((entropy / denom).item()) if denom > 0 else 0.0


def _random_orthogonal(n: int, device: torch.device, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    mat = torch.randn(n, n, device=device, dtype=dtype, generator=gen)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q


def _metric_from_weight(weight: torch.Tensor, side: str, eps: float) -> torch.Tensor:
    if weight.ndim != 2:
        n = int(weight.numel())
        return _eye(n, weight.device, weight.dtype)
    w = weight.detach()
    if side == "out":
        gram = w.matmul(w.transpose(0, 1))
    else:
        gram = w.transpose(0, 1).matmul(w)
    gram = _normalize_symmetric(gram, eps)
    return gram


def _low_cost_credit(metric: torch.Tensor, eps: float) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(_sym(metric).float())
    inv = 1.0 / vals.clamp_min(float(eps))
    inv = inv / inv.max().clamp_min(float(eps))
    return (vecs * inv.unsqueeze(0)).matmul(vecs.transpose(0, 1)).to(device=metric.device, dtype=metric.dtype)


def _generalized_projector(
    credit: torch.Tensor,
    metric: torch.Tensor,
    *,
    rank: int,
    strength: float,
    eta: float,
    eps: float,
    randomize_basis: bool,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    n = int(credit.shape[0])
    device = credit.device
    dtype = credit.dtype
    ident = _eye(n, device, dtype)
    credit = _normalize_symmetric(credit, eps)
    metric = _sym(metric)
    metric = metric + float(eps) * ident
    m_vals, m_vecs = torch.linalg.eigh(metric.float())
    m_vals = torch.clamp(m_vals, min=float(eps))
    max_val = m_vals.max().clamp_min(float(eps))
    if float(max_val.item() / m_vals.min().item()) > 1.0e8:
        m_vals = torch.clamp(m_vals, min=float(max_val.item() / 1.0e8))
    inv_sqrt = (m_vecs * torch.rsqrt(m_vals).unsqueeze(0)).matmul(m_vecs.transpose(0, 1))
    whitened = _sym(inv_sqrt.matmul(credit.float()).matmul(inv_sqrt))
    evals, evecs = torch.linalg.eigh(whitened)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    pos = evals > 0
    positive_rank = int(pos.sum().item())
    k = min(max(0, int(rank)), positive_rank, n)
    if k == 0 or abs(float(strength)) <= 0.0:
        return ident, {
            "positive_rank": positive_rank,
            "residual": 0.0,
            "psd_min": 1.0,
            "condition_number": 1.0,
            "self_condition": _condition_number_psd(metric, eps),
            "spectrum_entropy": _spectral_entropy(m_vals),
        }
    lambdas = evals[:k].to(device=device, dtype=dtype)
    q = inv_sqrt.to(device=device, dtype=dtype).matmul(evecs[:, :k].to(device=device, dtype=dtype))
    q = torch.nn.functional.normalize(q, dim=0, eps=float(eps))
    if randomize_basis and generator is not None:
        rand = _random_orthogonal(n, device, dtype, generator)[:, :k]
        q = rand
    phi = torch.clamp(lambdas, min=0.0) / (torch.clamp(lambdas, min=0.0) + float(eta))
    projector = (q * phi.unsqueeze(0)).matmul(q.transpose(0, 1))
    p_mat = _sym(ident + float(strength) * projector)
    p_vals = torch.linalg.eigvalsh(p_mat.float())
    psd_min = float(p_vals.min().item())
    condition_number = float((p_vals.max() / p_vals.min().clamp_min(float(eps))).item())
    residual = 0.0
    if k > 0:
        metric_q = metric.float().matmul(q.float())
        credit_q = credit.float().matmul(q.float())
        lam = lambdas.float().to(device=credit_q.device)
        denom = credit_q.norm().clamp_min(float(eps))
        residual = float((credit_q - metric_q * lam.unsqueeze(0)).norm().div(denom).item())
    return p_mat, {
        "positive_rank": positive_rank,
        "residual": residual,
        "psd_min": psd_min,
        "condition_number": condition_number,
        "self_condition": _condition_number_psd(metric, eps),
        "spectrum_entropy": _spectral_entropy(m_vals),
    }


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


class CreditConditionedSelfGeometryOperator:
    """Bounded matrix-factor gradient transform for CCSG experiments."""

    def __init__(self, params: Iterable[torch.nn.Parameter], config: CCSGConfig | None = None) -> None:
        self.params = [p for p in params if p.requires_grad]
        self._param_ids = {id(p) for p in self.params}
        self.config = config or CCSGConfig()
        self.trace = CCSGTrace()
        self._state: dict[int, _MatrixBlockState] = {}
        self._ema_cov_out: dict[int, torch.Tensor] = {}
        self._ema_cov_in: dict[int, torch.Tensor] = {}
        self._prev_update_basis: dict[int, torch.Tensor] = {}
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
        optimizer_state: Any | None = None,
    ) -> dict[str, float | int]:
        start = time.perf_counter()
        self.trace.observe_calls += 1
        if not cohort_grads or not self.params:
            return self.diagnostics()
        cfg = self.config
        flat_by_cohort = [_flatten_grads(grads, self.params) for grads in cohort_grads]
        n_cohorts = len(flat_by_cohort)
        split = min(max(1, int(round(n_cohorts * float(cfg.source_fraction)))), max(1, n_cohorts - 1))
        source_ids = list(range(split))
        witness_ids = list(range(split, n_cohorts)) or list(range(split))
        control = str(cfg.control).lower()
        if control == "shuffled_source_witness_pairing":
            witness_ids = list(reversed(witness_ids))
        if control in {"source_only_crossfit_factor", "source_only_operator"}:
            witness_ids = source_ids
        if control in {"witness_only_crossfit_factor", "witness_only_operator"}:
            source_ids = witness_ids
        build_start = time.perf_counter()
        transfers: list[float] = []
        for p_idx, param in enumerate(self.params):
            device = param.device
            dtype = param.dtype if param.dtype.is_floating_point else torch.float32
            source = torch.stack([flat_by_cohort[i][p_idx].to(device=device, dtype=dtype) for i in source_ids], dim=0).mean(dim=0)
            witness = torch.stack([flat_by_cohort[i][p_idx].to(device=device, dtype=dtype) for i in witness_ids], dim=0).mean(dim=0)
            if control == "signflip_cotangent_factor":
                witness = -witness
            if control == "random_activation_factor":
                witness = torch.randn(witness.shape, device=device, dtype=dtype, generator=self._generator(device))
            state = self._build_param_state(param, source, witness, optimizer_state)
            self._state[id(param)] = state
            transfers.append(_safe_corr(source, witness))
            self._record_state_metrics(state)
        self.trace.operator_build_ms = (time.perf_counter() - build_start) * 1000.0
        self.trace.source_witness_transfer_LCB = _lcb95(transfers)
        self.trace.source_witness_gap = float(1.0 - self.trace.source_witness_transfer_LCB)
        self._record_nuisance_correlations(cohort_labels, cohort_losses, cohort_margins)
        self.trace.observe_time_ms += (time.perf_counter() - start) * 1000.0
        return self.diagnostics()

    def _record_state_metrics(self, state: _MatrixBlockState) -> None:
        self.trace._rank_out_values.append(float(state.positive_rank_out))
        self.trace._rank_in_values.append(float(state.positive_rank_in))
        self.trace._self_out_values.append(float(state.self_condition_out))
        self.trace._self_in_values.append(float(state.self_condition_in))
        self.trace._residuals.append(max(float(state.residual_out), float(state.residual_in)))
        self.trace._psd_values.append(float(state.psd_min))
        self.trace._condition_values.append(float(state.condition_number))
        self.trace._credit_self_values.append(float(state.credit_self_alignment))
        self.trace._credit_only_values.append(float(state.credit_only_alignment))
        self.trace._self_only_values.append(float(state.self_only_alignment))
        self.trace.self_spectrum_entropy = state.spectrum_entropy
        self.trace.update_spectrum_drift = state.update_spectrum_drift

    def _record_nuisance_correlations(
        self,
        cohort_labels: list[torch.Tensor] | None,
        cohort_losses: list[torch.Tensor] | None,
        cohort_margins: list[torch.Tensor] | None,
    ) -> None:
        first_device = self.params[0].device if self.params else torch.device("cpu")
        state_scores = torch.tensor([s.scalar_gate for s in self._state.values()], device=first_device, dtype=torch.float32)
        if int(state_scores.numel()) == 0:
            return
        # Repeat or truncate cohort statistics to match the number of transformed blocks.
        class_vals = _class_balance_metric(cohort_labels, first_device)
        loss_vals = _tensor_means(cohort_losses)
        margin_vals = _tensor_means(cohort_margins)
        for name, values in (
            ("operator_class_corr_abs", class_vals),
            ("operator_loss_corr_abs", loss_vals),
            ("operator_margin_corr_abs", margin_vals),
        ):
            if not values:
                setattr(self.trace, name, 0.0)
                continue
            reps = int(math.ceil(max(1, int(state_scores.numel())) / max(1, len(values))))
            vec = torch.tensor((values * reps)[: int(state_scores.numel())], device=first_device, dtype=torch.float32)
            setattr(self.trace, name, abs(_safe_corr(state_scores, vec)))

    def _build_param_state(
        self,
        param: torch.nn.Parameter,
        source: torch.Tensor,
        witness: torch.Tensor,
        optimizer_state: Any | None,
    ) -> _MatrixBlockState:
        cfg = self.config
        control = str(cfg.control).lower()
        state = _MatrixBlockState()
        if control == "same_compute_noop":
            state.scalar_gate = 1.0
            return state
        if source.ndim == 1:
            source_m = source.reshape(-1, 1)
            witness_m = witness.reshape(-1, 1)
            weight_m = param.detach().reshape(-1, 1).to(dtype=source_m.dtype)
        elif source.ndim == 2:
            source_m = source
            witness_m = witness
            weight_m = param.detach().to(dtype=source_m.dtype)
        else:
            source_m = source.reshape(int(source.shape[0]), -1)
            witness_m = witness.reshape_as(source_m)
            weight_m = param.detach().reshape_as(source_m).to(dtype=source_m.dtype)
        out_dim, in_dim = int(source_m.shape[0]), int(source_m.shape[1])
        eps = float(cfg.eps)
        c_out = _normalize_symmetric(source_m.matmul(witness_m.transpose(0, 1)), eps)
        c_in = _normalize_symmetric(source_m.transpose(0, 1).matmul(witness_m), eps)
        m_out = self._self_metric(param, source_m, witness_m, weight_m, "out", out_dim, optimizer_state)
        m_in = self._self_metric(param, source_m, witness_m, weight_m, "in", in_dim, optimizer_state)
        randomize = control in {
            "same_generalized_spectrum_random_basis",
            "same_rank_random_out_in_factor",
            "same_spectrum_random_out_in_factor",
        }
        if control == "credit_only_operator":
            m_out = _eye(out_dim, source_m.device, source_m.dtype)
            m_in = _eye(in_dim, source_m.device, source_m.dtype)
        elif control in {
            "self_only_operator",
            "debt_self_only_control",
            "same_debt_metric_self_only",
            "oet_self_geometry_only",
            "soap_like_eigenbasis_control",
            "kfac_style_train_covariance_only",
        }:
            c_out = _low_cost_credit(m_out, eps)
            c_in = _low_cost_credit(m_in, eps)
        elif control == "same_credit_random_self_metric":
            m_out = self._randomize_metric(m_out)
            m_in = self._randomize_metric(m_in)
        elif control in {"same_self_random_credit", "same_debt_metric_random_credit"}:
            c_out = self._randomize_metric(c_out)
            c_in = self._randomize_metric(c_in)
        alpha = float(cfg.alpha) if cfg.use_out else 0.0
        beta = float(cfg.beta) if cfg.use_in else 0.0
        gen = self._generator(source_m.device)
        p_out, info_out = _generalized_projector(
            c_out,
            m_out,
            rank=int(cfg.rank),
            strength=alpha,
            eta=float(cfg.eta),
            eps=eps,
            randomize_basis=randomize,
            generator=gen,
        )
        p_in, info_in = _generalized_projector(
            c_in,
            m_in,
            rank=int(cfg.rank),
            strength=beta,
            eta=float(cfg.eta),
            eps=eps,
            randomize_basis=randomize,
            generator=gen,
        )
        state.p_out = p_out
        state.p_in = p_in
        state.credit_out = c_out
        state.credit_in = c_in
        state.metric_out = m_out
        state.metric_in = m_in
        state.positive_rank_out = int(info_out["positive_rank"])
        state.positive_rank_in = int(info_in["positive_rank"])
        state.residual_out = float(info_out["residual"])
        state.residual_in = float(info_in["residual"])
        state.psd_min = min(float(info_out["psd_min"]), float(info_in["psd_min"]))
        state.condition_number = max(float(info_out["condition_number"]), float(info_in["condition_number"]))
        state.self_condition_out = float(info_out["self_condition"])
        state.self_condition_in = float(info_in["self_condition"])
        state.spectrum_entropy = 0.5 * (float(info_out["spectrum_entropy"]) + float(info_in["spectrum_entropy"]))
        state.credit_self_alignment = 0.5 * (_safe_corr(c_out, _low_cost_credit(m_out, eps)) + _safe_corr(c_in, _low_cost_credit(m_in, eps)))
        state.credit_only_alignment = 0.5 * (_safe_corr(c_out, source_m.matmul(source_m.transpose(0, 1))) + _safe_corr(c_in, source_m.transpose(0, 1).matmul(source_m)))
        state.self_only_alignment = 0.5 * (_safe_corr(_low_cost_credit(m_out, eps), source_m.matmul(source_m.transpose(0, 1))) + _safe_corr(_low_cost_credit(m_in, eps), source_m.transpose(0, 1).matmul(source_m)))
        state.update_spectrum_drift = self._update_spectrum_drift(param, source_m)
        state.scalar_gate = self._scalar_control_gate(control, state, source, witness)
        return state

    def _randomize_metric(self, metric: torch.Tensor) -> torch.Tensor:
        vals = torch.linalg.eigvalsh(_sym(metric).float()).to(device=metric.device, dtype=metric.dtype)
        q = _random_orthogonal(int(metric.shape[0]), metric.device, metric.dtype, self._generator(metric.device))
        return _sym((q * vals.unsqueeze(0)).matmul(q.transpose(0, 1)))

    def _self_metric(
        self,
        param: torch.nn.Parameter,
        source: torch.Tensor,
        witness: torch.Tensor,
        weight: torch.Tensor,
        side: str,
        n: int,
        optimizer_state: Any | None,
    ) -> torch.Tensor:
        cfg = self.config
        ident = _eye(n, source.device, source.dtype)
        if side == "out":
            cov = _normalize_symmetric(source.matmul(source.transpose(0, 1)) + witness.matmul(witness.transpose(0, 1)), cfg.eps)
        else:
            cov = _normalize_symmetric(source.transpose(0, 1).matmul(source) + witness.transpose(0, 1).matmul(witness), cfg.eps)
        key = id(param)
        store = self._ema_cov_out if side == "out" else self._ema_cov_in
        prev = store.get(key)
        if prev is None or tuple(prev.shape) != tuple(cov.shape):
            ema = cov.detach()
        else:
            ema = float(cfg.ema_beta) * prev.to(device=cov.device, dtype=cov.dtype) + (1.0 - float(cfg.ema_beta)) * cov.detach()
        store[key] = ema.detach()
        weight_metric = _metric_from_weight(weight, side, cfg.eps) if weight.ndim == 2 else ident
        metric = ident + float(cfg.gradient_cov_weight) * ema + float(cfg.weight_spectrum_weight) * weight_metric
        if optimizer_state is not None:
            metric = metric + float(cfg.momentum_weight) * self._momentum_metric(param, side, n, source.dtype, optimizer_state)
        if float(cfg.sharpness_weight) > 0.0:
            metric = metric + float(cfg.sharpness_weight) * cov.matmul(cov.transpose(0, 1) if cov.ndim == 2 else cov)
        return _sym(metric + float(cfg.eps) * ident)

    def _momentum_metric(
        self,
        param: torch.nn.Parameter,
        side: str,
        n: int,
        dtype: torch.dtype,
        optimizer_state: Any,
    ) -> torch.Tensor:
        device = param.device
        ident = _eye(n, device, dtype)
        try:
            state = optimizer_state.get(param, {}) if isinstance(optimizer_state, dict) else {}
            moment = state.get("exp_avg") if isinstance(state, dict) else None
            if moment is None:
                return ident * 0.0
            m = moment.detach().to(device=device, dtype=dtype)
            if m.ndim == 1:
                mat = m.reshape(-1, 1)
            elif m.ndim == 2:
                mat = m
            else:
                mat = m.reshape(int(m.shape[0]), -1)
            if side == "out":
                gram = mat.matmul(mat.transpose(0, 1))
            else:
                gram = mat.transpose(0, 1).matmul(mat)
            if int(gram.shape[0]) != n:
                return ident * 0.0
            self.trace.momentum_alignment = max(self.trace.momentum_alignment, abs(_safe_corr(gram, mat.matmul(mat.transpose(0, 1)) if side == "out" else mat.transpose(0, 1).matmul(mat))))
            return _normalize_symmetric(gram, self.config.eps)
        except Exception:
            return ident * 0.0

    def _update_spectrum_drift(self, param: torch.nn.Parameter, source: torch.Tensor) -> float:
        if source.ndim != 2:
            return 0.0
        try:
            _u, _s, vh = torch.linalg.svd(source.float(), full_matrices=False)
            top = vh[: min(4, int(vh.shape[0]))].detach()
            prev = self._prev_update_basis.get(id(param))
            self._prev_update_basis[id(param)] = top
            if prev is None or tuple(prev.shape) != tuple(top.shape):
                return 0.0
            sim = torch.linalg.svdvals(prev.to(device=top.device).matmul(top.transpose(0, 1))).mean().clamp(0.0, 1.0)
            return float((1.0 - sim).item())
        except Exception:
            return 0.0

    def _scalar_control_gate(self, control: str, state: _MatrixBlockState, source: torch.Tensor, witness: torch.Tensor) -> float:
        if control in {
            "inverse_class_count_gate",
            "hard_loss_gate",
            "loss_rank_gate",
            "same_debt_metric_hard_loss_control",
            "same_debt_metric_loss_rank_control",
            "margin_shuffle_gate",
            "random_label_gate",
        }:
            raw = 1.0 + 0.10 * float(state.credit_self_alignment)
            return float(max(self.config.scalar_gate_floor, min(self.config.scalar_gate_cap, raw)))
        return 1.0

    def transform(self, grad: torch.Tensor, param: torch.nn.Parameter, group: dict[str, Any] | None = None) -> torch.Tensor:
        start = time.perf_counter()
        self.trace.transform_calls += 1
        if not self._inside_optimizer_step:
            return grad
        state = self._state.get(id(param))
        if state is None:
            return grad
        apply_start = time.perf_counter()
        raw = grad.detach()
        if state.p_out is None and state.p_in is None:
            transformed = grad * float(state.scalar_gate)
        else:
            if grad.ndim == 1:
                mat = grad.reshape(-1, 1)
                out = state.p_out.matmul(mat) if state.p_out is not None else mat
                transformed = out.reshape_as(grad)
            elif grad.ndim == 2:
                out = grad
                if state.p_out is not None:
                    out = state.p_out.to(device=grad.device, dtype=grad.dtype).matmul(out)
                if state.p_in is not None:
                    out = out.matmul(state.p_in.to(device=grad.device, dtype=grad.dtype))
                transformed = out
            else:
                mat = grad.reshape(int(grad.shape[0]), -1)
                out = mat
                if state.p_out is not None and int(state.p_out.shape[0]) == int(out.shape[0]):
                    out = state.p_out.to(device=grad.device, dtype=grad.dtype).matmul(out)
                if state.p_in is not None and int(state.p_in.shape[0]) == int(out.shape[1]):
                    out = out.matmul(state.p_in.to(device=grad.device, dtype=grad.dtype))
                transformed = out.reshape_as(grad)
            transformed = transformed * float(state.scalar_gate)
        dot = float((raw.reshape(-1).float().dot(transformed.detach().reshape(-1).float())).item())
        raw_norm = float(raw.reshape(-1).float().norm().item())
        new_norm = float(transformed.detach().reshape(-1).float().norm().item())
        cos = dot / max(raw_norm * new_norm, 1.0e-12)
        if not math.isfinite(cos) or cos < float(self.config.c_min):
            transformed = grad
            self.trace._projection_events += 1
            cos = 1.0
            new_norm = raw_norm
        self.trace.optimizer_owned_transform_pass = 1
        self.trace.gradients_transformed += 1
        self.trace._cos_values.append(float(cos))
        self.trace._norm_ratios.append(float(new_norm / max(raw_norm, 1.0e-12)))
        self.trace.operator_apply_ms += (time.perf_counter() - apply_start) * 1000.0
        self.trace.transform_time_ms += (time.perf_counter() - start) * 1000.0
        return transformed

    def diagnostics(self) -> dict[str, float | int]:
        return self.trace.as_dict()


__all__ = ["CCSGConfig", "CreditConditionedSelfGeometryOperator"]
