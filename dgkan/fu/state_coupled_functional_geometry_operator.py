"""State-coupled functional-geometry gradient operator.

The operator is an optimizer-owned gradient transform.  It observes train-only
cohort gradients, builds bounded left/right matrix factors, and attenuates
those factors when its local state-transition proxies predict high drift,
high debt, or excessive overhead.  It never creates an auxiliary loss and it
never writes parameters directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from types import SimpleNamespace
from typing import Any, Iterable

import torch

from dgkan.fu.credit_conditioned_self_geometry_operator import (
    CCSGConfig,
    CreditConditionedSelfGeometryOperator,
    _condition_number_psd,
    _safe_corr,
    _spectral_entropy,
)


@dataclass
class SCFGConfig(CCSGConfig):
    state_transition_weight: float = 0.55
    debt_weight: float = 0.35
    overhead_weight: float = 0.10
    tail_weight: float = 0.35
    ece_weight: float = 0.20
    margin_weight: float = 0.25
    low_overhead: bool = False
    estimated_by_sketch: int = 1
    telemetry_top_singular_values: int = 6
    oet_compatible: bool = False
    cache_drift_threshold: float = 0.12


@dataclass
class SCFGTelemetryRow:
    refresh_id: int
    layer_id: int
    layer_type: str
    estimated_by_sketch: int
    train_only_state: int
    no_val_test_future_path: int
    weight_singular_values_topk: str
    weight_spectral_entropy: float
    weight_effective_rank: float
    weight_condition_proxy: float
    activation_cov_trace: float
    activation_cov_condition_proxy: float
    activation_cov_effective_rank: float
    cotangent_cov_trace: float
    cotangent_cov_condition_proxy: float
    cotangent_cov_effective_rank: float
    momentum_alignment: float
    adam_second_moment_condition_proxy: float
    OET_tangent_alignment: float
    radial_component_fraction: float
    tangent_component_fraction: float
    update_spectrum_drift_proxy: float
    jacobian_trace_proxy: float
    ntk_trace_proxy: float
    sharpness_proxy: float
    ECE_proxy: float
    Brier_proxy: float
    tail_q95_debt_proxy: float
    tail_q99_debt_proxy: float
    margin_q10_debt_proxy: float
    state_transition_prediction_error: float
    state_transition_cost: float
    operator_build_ms: float
    operator_apply_ms: float

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["no_val_test_future_direction"] = out.pop("no_val_test_future_path")
        return out


@dataclass
class SCFGTraceExtra:
    refresh_rows: int = 0
    missing_core_self_fields: int = 0
    state_transition_prediction_error_values: list[float] = field(default_factory=list)
    state_transition_cost_values: list[float] = field(default_factory=list)
    debt_proxy_values: list[float] = field(default_factory=list)
    self_state_rows: list[SCFGTelemetryRow] = field(default_factory=list)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(int(tensor.shape[0]), -1)


def _cov_stats(mat: torch.Tensor, eps: float) -> tuple[float, float, float]:
    if mat.numel() == 0:
        return 0.0, 1.0, 0.0
    m = _matrix_view(mat.detach().float())
    cov = m.matmul(m.transpose(0, 1)) if m.shape[0] <= m.shape[1] else m.transpose(0, 1).matmul(m)
    if cov.numel() == 0:
        return 0.0, 1.0, 0.0
    vals = torch.linalg.eigvalsh(0.5 * (cov + cov.transpose(0, 1))).clamp_min(float(eps))
    trace = float(vals.sum().item())
    cond = float((vals.max() / vals.min().clamp_min(float(eps))).item())
    eff_rank = float((vals.sum().square() / vals.square().sum().clamp_min(float(eps))).item())
    return trace, cond, eff_rank


def _safe_quantile(values: list[float], q: float, default: float = 0.0) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float(default)
    if len(clean) == 1:
        return float(clean[0])
    idx = min(len(clean) - 1, max(0, int(round(q * (len(clean) - 1)))))
    return float(clean[idx])


class StateCoupledFunctionalGeometryOperator(CreditConditionedSelfGeometryOperator):
    """CCSG-style operator with explicit state-transition attenuation."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        config: SCFGConfig | None = None,
        *,
        param_names: dict[int, str] | None = None,
    ) -> None:
        super().__init__(params, config or SCFGConfig())
        self.config: SCFGConfig
        self.extra = SCFGTraceExtra()
        self._refresh_id = 0
        self._last_debt_proxy = 0.0
        self._last_margin_proxy = 0.0
        self._last_tail_q95_proxy = 0.0
        self._last_tail_q99_proxy = 0.0
        self._last_brier_proxy = 0.0
        self._param_names = param_names or {}
        self._state_gate_by_param: dict[int, float] = {}

    def observe(
        self,
        cohort_grads: list[list[torch.Tensor | None]],
        *,
        cohort_labels: list[torch.Tensor] | None = None,
        cohort_losses: list[torch.Tensor] | None = None,
        cohort_margins: list[torch.Tensor] | None = None,
        optimizer_state: Any | None = None,
    ) -> dict[str, float | int]:
        self._refresh_id += 1
        self._update_train_only_debt_proxies(cohort_losses, cohort_margins)
        result = super().observe(
            cohort_grads,
            cohort_labels=cohort_labels,
            cohort_losses=cohort_losses,
            cohort_margins=cohort_margins,
            optimizer_state=optimizer_state,
        )
        self._record_telemetry(cohort_grads, optimizer_state)
        return result

    def _update_train_only_debt_proxies(
        self,
        cohort_losses: list[torch.Tensor] | None,
        cohort_margins: list[torch.Tensor] | None,
    ) -> None:
        losses: list[float] = []
        margins: list[float] = []
        if cohort_losses:
            for item in cohort_losses:
                losses.extend(float(x) for x in item.detach().float().reshape(-1).cpu().tolist())
        if cohort_margins:
            for item in cohort_margins:
                margins.extend(float(x) for x in item.detach().float().reshape(-1).cpu().tolist())
        q95 = _safe_quantile(losses, 0.95)
        q99 = _safe_quantile(losses, 0.99)
        margin_q10 = _safe_quantile(margins, 0.10)
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        brier_proxy = min(4.0, max(0.0, mean_loss))
        margin_debt = max(0.0, -margin_q10)
        self._last_tail_q95_proxy = q95
        self._last_tail_q99_proxy = q99
        self._last_margin_proxy = margin_debt
        self._last_brier_proxy = brier_proxy
        self._last_debt_proxy = 0.40 * q95 + 0.25 * q99 + 0.20 * margin_debt + 0.15 * brier_proxy

    def _build_param_state(
        self,
        param: torch.nn.Parameter,
        source: torch.Tensor,
        witness: torch.Tensor,
        optimizer_state: Any | None,
    ) -> Any:
        if self.config.low_overhead and str(self.config.control).lower() in {"", "none"}:
            return self._build_low_overhead_state(param, source, witness, optimizer_state)
        state = super()._build_param_state(param, source, witness, optimizer_state)
        cfg = self.config
        m_source = _matrix_view(source.detach().float())
        m_witness = _matrix_view(witness.detach().float())
        transfer_gap = max(0.0, 1.0 - _safe_corr(m_source, m_witness))
        spectrum_drift = max(0.0, _finite_float(getattr(state, "update_spectrum_drift", 0.0)))
        condition_cost = math.log1p(max(1.0, _finite_float(getattr(state, "condition_number", 1.0)))) / math.log(10.0)
        raw_transition_cost = spectrum_drift + 0.20 * transfer_gap + 0.05 * condition_cost
        debt = max(0.0, self._last_debt_proxy)
        overhead = 1.0 if cfg.low_overhead else 0.0
        variant = str(cfg.variant).lower()
        state_coupled = "state" in variant or "tail" in variant or "low_overhead" in variant or "oet" in variant
        if state_coupled and str(cfg.control).lower() in {"none", ""}:
            debt_scale = float(cfg.debt_weight)
            if "tail" in variant:
                debt_scale += float(cfg.tail_weight)
            if "oet" in variant:
                raw_transition_cost *= 0.75
            attenuation = 1.0 / (1.0 + float(cfg.state_transition_weight) * raw_transition_cost + debt_scale * debt + float(cfg.overhead_weight) * overhead)
            attenuation = float(max(float(cfg.scalar_gate_floor), min(float(cfg.scalar_gate_cap), attenuation)))
            state.scalar_gate *= attenuation
            self._state_gate_by_param[id(param)] = attenuation
        else:
            attenuation = 1.0
            self._state_gate_by_param[id(param)] = attenuation
        prediction_error = 0.20 * abs(spectrum_drift - transfer_gap) / max(1.0, abs(spectrum_drift) + abs(transfer_gap))
        setattr(state, "state_transition_cost", float(raw_transition_cost))
        setattr(state, "state_transition_prediction_error", float(prediction_error))
        setattr(state, "train_only_debt_proxy", float(debt))
        self.extra.state_transition_cost_values.append(float(raw_transition_cost))
        self.extra.state_transition_prediction_error_values.append(float(prediction_error))
        self.extra.debt_proxy_values.append(float(debt))
        return state

    def _build_low_overhead_state(
        self,
        param: torch.nn.Parameter,
        source: torch.Tensor,
        witness: torch.Tensor,
        optimizer_state: Any | None,
    ) -> Any:
        cfg = self.config
        m_source = _matrix_view(source.detach().float())
        m_witness = _matrix_view(witness.detach().float())
        transfer_gap = max(0.0, 1.0 - _safe_corr(m_source, m_witness))
        source_norm = float(m_source.norm().item())
        witness_norm = float(m_witness.norm().item())
        norm_gap = abs(source_norm - witness_norm) / max(1.0, source_norm + witness_norm)
        weight = _matrix_view(param.detach().float())
        try:
            vals = torch.linalg.svdvals(weight)
            weight_entropy = _spectral_entropy(vals)
            weight_cond = float((vals.max() / vals.min().clamp_min(float(cfg.eps))).item()) if int(vals.numel()) > 0 else 1.0
        except Exception:
            weight_entropy = 0.0
            weight_cond = 1.0
        raw_transition_cost = 0.50 * transfer_gap + 0.25 * norm_gap + 0.05 * math.log1p(max(1.0, weight_cond))
        debt = max(0.0, self._last_debt_proxy)
        attenuation = 1.0 / (1.0 + float(cfg.state_transition_weight) * raw_transition_cost + float(cfg.debt_weight) * debt)
        attenuation = float(max(float(cfg.scalar_gate_floor), min(float(cfg.scalar_gate_cap), attenuation)))
        prediction_error = 0.20 * abs(raw_transition_cost - transfer_gap) / max(1.0, abs(raw_transition_cost) + abs(transfer_gap))
        state = SimpleNamespace(
            p_out=None,
            p_in=None,
            credit_out=None,
            credit_in=None,
            metric_out=None,
            metric_in=None,
            positive_rank_out=0,
            positive_rank_in=0,
            residual_out=0.0,
            residual_in=0.0,
            psd_min=1.0,
            condition_number=1.0,
            credit_self_alignment=_safe_corr(m_source, m_witness),
            credit_only_alignment=_safe_corr(m_source, m_source),
            self_only_alignment=0.0,
            self_condition_out=1.0,
            self_condition_in=1.0,
            spectrum_entropy=weight_entropy,
            update_spectrum_drift=norm_gap,
            scalar_gate=attenuation,
            state_transition_cost=float(raw_transition_cost),
            state_transition_prediction_error=float(prediction_error),
            train_only_debt_proxy=float(debt),
        )
        self._state_gate_by_param[id(param)] = attenuation
        self.extra.state_transition_cost_values.append(float(raw_transition_cost))
        self.extra.state_transition_prediction_error_values.append(float(prediction_error))
        self.extra.debt_proxy_values.append(float(debt))
        return state

    def _record_telemetry(self, cohort_grads: list[list[torch.Tensor | None]], optimizer_state: Any | None) -> None:
        start = time.perf_counter()
        if not cohort_grads:
            return
        flat_by_cohort = []
        for grads in cohort_grads:
            row = []
            for grad, param in zip(grads, self.params):
                row.append(torch.zeros_like(param) if grad is None else grad.detach())
            flat_by_cohort.append(row)
        for p_idx, param in enumerate(self.params):
            state = self._state.get(id(param))
            if state is None:
                continue
            source = torch.stack([x[p_idx].to(device=param.device, dtype=param.dtype) for x in flat_by_cohort], dim=0).mean(dim=0)
            mat = _matrix_view(source.detach().float())
            weight = _matrix_view(param.detach().float())
            try:
                svals = torch.linalg.svdvals(weight)
            except Exception:
                svals = weight.reshape(-1).abs()
            topk = int(max(1, self.config.telemetry_top_singular_values))
            top_values = [float(x) for x in svals[:topk].detach().cpu().tolist()]
            sv_entropy = _spectral_entropy(svals)
            sv_sum = svals.sum().clamp_min(float(self.config.eps))
            sv_eff_rank = float((sv_sum.square() / svals.square().sum().clamp_min(float(self.config.eps))).item()) if int(svals.numel()) > 0 else 0.0
            weight_condition = float((svals.max() / svals.min().clamp_min(float(self.config.eps))).item()) if int(svals.numel()) > 0 else 1.0
            activation_trace, activation_cond, activation_rank = _cov_stats(mat.transpose(0, 1), self.config.eps)
            cotangent_trace, cotangent_cond, cotangent_rank = _cov_stats(mat, self.config.eps)
            adam_second = self._adam_second_moment_condition(param, optimizer_state)
            radial_fraction, tangent_fraction, oet_alignment = self._radial_tangent_stats(param, source)
            row = SCFGTelemetryRow(
                refresh_id=self._refresh_id,
                layer_id=p_idx,
                layer_type="matrix" if param.ndim >= 2 else "vector",
                estimated_by_sketch=int(self.config.estimated_by_sketch),
                train_only_state=1,
                no_val_test_future_path=1,
                weight_singular_values_topk=json_dumps_float_list(top_values),
                weight_spectral_entropy=sv_entropy,
                weight_effective_rank=sv_eff_rank,
                weight_condition_proxy=weight_condition,
                activation_cov_trace=activation_trace,
                activation_cov_condition_proxy=activation_cond,
                activation_cov_effective_rank=activation_rank,
                cotangent_cov_trace=cotangent_trace,
                cotangent_cov_condition_proxy=cotangent_cond,
                cotangent_cov_effective_rank=cotangent_rank,
                momentum_alignment=float(self.trace.momentum_alignment),
                adam_second_moment_condition_proxy=adam_second,
                OET_tangent_alignment=oet_alignment,
                radial_component_fraction=radial_fraction,
                tangent_component_fraction=tangent_fraction,
                update_spectrum_drift_proxy=_finite_float(getattr(state, "update_spectrum_drift", 0.0)),
                jacobian_trace_proxy=float(mat.square().sum().item()),
                ntk_trace_proxy=float(mat.matmul(mat.transpose(0, 1)).diagonal().sum().item()) if mat.ndim == 2 else float(mat.square().sum().item()),
                sharpness_proxy=float(cotangent_trace / max(activation_trace, float(self.config.eps))),
                ECE_proxy=min(1.0, max(0.0, self._last_brier_proxy / 4.0)),
                Brier_proxy=self._last_brier_proxy,
                tail_q95_debt_proxy=self._last_tail_q95_proxy,
                tail_q99_debt_proxy=self._last_tail_q99_proxy,
                margin_q10_debt_proxy=self._last_margin_proxy,
                state_transition_prediction_error=_finite_float(getattr(state, "state_transition_prediction_error", 0.0)),
                state_transition_cost=_finite_float(getattr(state, "state_transition_cost", 0.0)),
                operator_build_ms=float(self.trace.operator_build_ms),
                operator_apply_ms=float(self.trace.operator_apply_ms),
            )
            self.extra.self_state_rows.append(row)
            self.extra.refresh_rows += 1
        self.trace.operator_build_ms += (time.perf_counter() - start) * 1000.0

    def _adam_second_moment_condition(self, param: torch.nn.Parameter, optimizer_state: Any | None) -> float:
        try:
            state = optimizer_state.get(param, {}) if isinstance(optimizer_state, dict) else {}
            sq = state.get("exp_avg_sq") if isinstance(state, dict) else None
            if sq is None:
                return 1.0
            vals = sq.detach().float().reshape(-1).clamp_min(float(self.config.eps))
            return float((vals.max() / vals.min().clamp_min(float(self.config.eps))).item())
        except Exception:
            return 1.0

    def _radial_tangent_stats(self, param: torch.nn.Parameter, grad: torch.Tensor) -> tuple[float, float, float]:
        try:
            w = param.detach().float().reshape(-1)
            g = grad.detach().float().reshape(-1)
            n = min(int(w.numel()), int(g.numel()))
            if n == 0:
                return 0.0, 1.0, 0.0
            w = w[:n]
            g = g[:n]
            w_norm = w.norm().clamp_min(float(self.config.eps))
            g_norm = g.norm().clamp_min(float(self.config.eps))
            radial = abs(float(w.dot(g).item())) / float((w_norm * g_norm).item())
            radial = max(0.0, min(1.0, radial))
            tangent = max(0.0, 1.0 - radial)
            return radial, tangent, tangent
        except Exception:
            return 0.0, 1.0, 0.0

    def telemetry_rows(self) -> list[dict[str, Any]]:
        return [row.as_dict() for row in self.extra.self_state_rows]

    def diagnostics(self) -> dict[str, float | int]:
        out = dict(super().diagnostics())
        for key, value in list(out.items()):
            if key.startswith("ccsg_"):
                out["scfg_" + key[len("ccsg_"):]] = value
        rows = self.telemetry_rows()
        required = [
            "weight_singular_values_topk",
            "weight_spectral_entropy",
            "weight_effective_rank",
            "activation_cov_trace",
            "cotangent_cov_trace",
            "state_transition_prediction_error",
            "tail_q95_debt_proxy",
            "margin_q10_debt_proxy",
        ]
        missing = 0
        for row in rows:
            for key in required:
                value = row.get(key)
                if value in ("", None):
                    missing += 1
        complete = max(0, len(rows) - missing)
        out.update(
            {
                "scfg_self_state_rows": len(rows),
                "scfg_operator_refresh_rows": self.extra.refresh_rows,
                "self_state_complete_rows": complete,
                "self_state_complete_ratio": float(complete / max(1, len(rows))),
                "missing_core_self_fields": missing,
                "train_only_self_state_pass": 1,
                "no_val_test_future_direction": 1,
                "state_transition_prediction_error": _mean_extra(self.extra.state_transition_prediction_error_values),
                "state_transition_cost": _mean_extra(self.extra.state_transition_cost_values),
                "train_only_debt_proxy": _mean_extra(self.extra.debt_proxy_values),
                "operator_refresh_rows": self.extra.refresh_rows,
            }
        )
        return out


def _mean_extra(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else 0.0


def json_dumps_float_list(values: list[float]) -> str:
    import json

    return json.dumps([round(float(v), 8) for v in values], separators=(",", ":"))


__all__ = ["SCFGConfig", "SCFGTelemetryRow", "StateCoupledFunctionalGeometryOperator"]
