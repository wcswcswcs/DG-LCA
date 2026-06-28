"""Constrained Meta-FU scalar control law over analytic FU operators.

Meta-FU controls low-dimensional parameters of the existing analytic SCFG
operator.  It does not emit parameter deltas, it does not build an auxiliary
loss, and it only runs as an optimizer-owned gradient transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from dgkan.fu.credit_conditioned_self_geometry_operator import _safe_corr, _spectral_entropy
from dgkan.fu.state_coupled_functional_geometry_operator import (
    SCFGConfig,
    StateCoupledFunctionalGeometryOperator,
)


META_FU_FEATURE_DIM = 48
META_FU_OUTPUT_NAMES = ("alpha", "beta", "lambda_debt", "lambda_state", "tau", "rho")


@dataclass
class MetaFUConfig(SCFGConfig):
    controller_mode: str = "constant_scalar_scheduler"
    controller_path: str = ""
    input_dim: int = META_FU_FEATURE_DIM
    hidden_dim: int = 8
    alpha_max: float = 0.18
    beta_max: float = 0.18
    lambda_debt_max: float = 1.25
    lambda_state_max: float = 1.25
    tau_min: float = 0.02
    tau_max: float = 0.35
    default_rho: float = 0.50
    normalized_step: float = 0.0
    total_steps: int = 1
    learning_rate: float = 0.0
    weight_decay_value: float = 0.0
    controller_random_seed: int = 0
    official_frozen: int = 1
    train_only_controller_inputs: int = 1
    no_dataset_or_seed_input: int = 1
    feature_mode: str = "default"


@dataclass
class MetaFUTrace:
    controller_rows: list[dict[str, Any]] = field(default_factory=list)
    feature_rows: list[dict[str, Any]] = field(default_factory=list)
    trust_clip_events: int = 0
    trust_clip_checks: int = 0
    controller_eval_ms: float = 0.0
    controller_parameter_count: int = 0
    controller_checkpoint_loaded: int = 0
    controller_frozen_on_eval: int = 1
    controller_mode: str = ""
    input_dim: int = META_FU_FEATURE_DIM


def _finite(value: Any, default: float = 0.0) -> float:
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


def _safe_norm(tensor: torch.Tensor) -> float:
    try:
        return float(tensor.detach().float().norm().item())
    except Exception:
        return 0.0


def _weight_spectral_stats(param: torch.Tensor) -> tuple[float, float, float, float]:
    try:
        mat = _matrix_view(param.detach().float())
        if min(mat.shape) <= 0:
            return 0.0, 0.0, 0.0, 1.0
        vals = torch.linalg.svdvals(mat)
        if int(vals.numel()) == 0:
            return 0.0, 0.0, 0.0, 1.0
        top = float(vals.max().item())
        minv = float(vals.min().clamp_min(1.0e-8).item())
        eff = float(vals.sum().square().div(vals.square().sum().clamp_min(1.0e-12)).item())
        return _spectral_entropy(vals), eff, top, top / max(minv, 1.0e-8)
    except Exception:
        return 0.0, 0.0, 0.0, 1.0


class BoundedScalarController(torch.nn.Module):
    """Tiny bounded controller for M0 scalar scheduler outputs."""

    def __init__(
        self,
        input_dim: int = META_FU_FEATURE_DIM,
        hidden_dim: int = 8,
        kind: str = "mlp",
        *,
        feature_mean: Iterable[float] | None = None,
        feature_std: Iterable[float] | None = None,
        constant_raw: Iterable[float] | None = None,
        bounds: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.kind = str(kind)
        self.bounds = dict(bounds or {})
        if feature_mean is None:
            feature_mean = [0.0] * self.input_dim
        if feature_std is None:
            feature_std = [1.0] * self.input_dim
        self.register_buffer("feature_mean", torch.tensor(list(feature_mean), dtype=torch.float32).reshape(1, -1))
        self.register_buffer("feature_std", torch.tensor(list(feature_std), dtype=torch.float32).reshape(1, -1).clamp_min(1.0e-6))
        if self.kind == "mlp":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hidden_dim, len(META_FU_OUTPUT_NAMES)),
            )
        elif self.kind == "linear":
            self.net = torch.nn.Linear(self.input_dim, len(META_FU_OUTPUT_NAMES))
        elif self.kind == "gru":
            self.gru = torch.nn.GRUCell(self.input_dim, self.hidden_dim)
            self.head = torch.nn.Linear(self.hidden_dim, len(META_FU_OUTPUT_NAMES))
            self.register_buffer("_eval_hidden", torch.zeros(1, self.hidden_dim, dtype=torch.float32))
            self.net = torch.nn.Identity()
        elif self.kind == "constant":
            raw = torch.zeros(len(META_FU_OUTPUT_NAMES), dtype=torch.float32)
            if constant_raw is not None:
                raw = torch.tensor(list(constant_raw), dtype=torch.float32)
            self.register_buffer("constant_raw", raw.reshape(1, -1))
            self.net = torch.nn.Identity()
        else:
            raise ValueError(f"unknown controller kind: {kind}")

    def raw(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z = z.reshape(1, -1)
        z = z.float()
        if int(z.shape[-1]) < self.input_dim:
            z = torch.nn.functional.pad(z, (0, self.input_dim - int(z.shape[-1])))
        if int(z.shape[-1]) > self.input_dim:
            z = z[..., : self.input_dim]
        if self.kind == "constant":
            return self.constant_raw.to(device=z.device, dtype=z.dtype).expand(int(z.shape[0]), -1)
        x = (z - self.feature_mean.to(device=z.device, dtype=z.dtype)) / self.feature_std.to(device=z.device, dtype=z.dtype)
        if self.kind == "gru":
            if self.training or int(x.shape[0]) != 1:
                h0 = torch.zeros(int(x.shape[0]), self.hidden_dim, dtype=x.dtype, device=x.device)
                h = self.gru(x, h0)
            else:
                h0 = self._eval_hidden.to(device=x.device, dtype=x.dtype)
                h = self.gru(x, h0)
                self._eval_hidden = h.detach().to(device=self._eval_hidden.device, dtype=self._eval_hidden.dtype)
            return self.head(h)
        return self.net(x)

    def reset_state(self) -> None:
        if hasattr(self, "_eval_hidden"):
            self._eval_hidden.zero_()

    def bounded(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.raw(z)
        alpha_max = float(self.bounds.get("alpha_max", 0.18))
        beta_max = float(self.bounds.get("beta_max", 0.18))
        lambda_debt_max = float(self.bounds.get("lambda_debt_max", 1.25))
        lambda_state_max = float(self.bounds.get("lambda_state_max", 1.25))
        tau_min = float(self.bounds.get("tau_min", 0.02))
        tau_max = float(self.bounds.get("tau_max", 0.35))
        alpha = alpha_max * torch.sigmoid(raw[:, 0])
        beta = beta_max * torch.sigmoid(raw[:, 1])
        lambda_debt = lambda_debt_max * torch.sigmoid(raw[:, 2])
        lambda_state = lambda_state_max * torch.sigmoid(raw[:, 3])
        tau = tau_min + (tau_max - tau_min) * torch.sigmoid(raw[:, 4])
        rho = torch.sigmoid(raw[:, 5])
        return raw, {
            "alpha": alpha,
            "beta": beta,
            "lambda_debt": lambda_debt,
            "lambda_state": lambda_state,
            "tau": tau,
            "rho": rho,
        }


def controller_parameter_count(module: torch.nn.Module) -> int:
    return int(sum(int(p.numel()) for p in module.parameters()))


def load_controller_artifact(path: str | Path, mode: str, config: MetaFUConfig) -> tuple[BoundedScalarController, dict[str, Any], int]:
    artifact: dict[str, Any] = {}
    loaded = 0
    if path and Path(path).exists():
        artifact = torch.load(str(path), map_location="cpu")
        loaded = 1
    kind = "mlp" if "mlp" in str(mode) else ("linear" if "linear" in str(mode) else ("gru" if "gru" in str(mode) else "constant"))
    if str(mode).startswith("constant"):
        kind = "constant"
    bounds = {
        "alpha_max": float(config.alpha_max),
        "beta_max": float(config.beta_max),
        "lambda_debt_max": float(config.lambda_debt_max),
        "lambda_state_max": float(config.lambda_state_max),
        "tau_min": float(config.tau_min),
        "tau_max": float(config.tau_max),
    }
    feature_mean = artifact.get("feature_mean", [0.0] * int(config.input_dim))
    feature_std = artifact.get("feature_std", [1.0] * int(config.input_dim))
    constant_raw = artifact.get("constant_raw", [0.0] * len(META_FU_OUTPUT_NAMES))
    hidden_dim = int(artifact.get("hidden_dim", config.hidden_dim))
    input_dim = int(artifact.get("input_dim", config.input_dim))
    module = BoundedScalarController(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kind=kind,
        feature_mean=feature_mean,
        feature_std=feature_std,
        constant_raw=constant_raw,
        bounds=bounds,
    )
    state_key = "state_dict"
    if loaded and state_key in artifact and kind in {"mlp", "linear", "gru"}:
        module.load_state_dict(artifact[state_key], strict=True)
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module, artifact, loaded


class MetaFUScalarOperator(StateCoupledFunctionalGeometryOperator):
    """M0 controller over SCFG scalar strengths and safety coefficients."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        config: MetaFUConfig | None = None,
        *,
        param_names: dict[int, str] | None = None,
    ) -> None:
        super().__init__(params, config or MetaFUConfig(), param_names=param_names)
        self.config: MetaFUConfig
        self.meta_trace = MetaFUTrace(controller_mode=str(self.config.controller_mode), input_dim=int(self.config.input_dim))
        self._controller, self._artifact, loaded = load_controller_artifact(self.config.controller_path, self.config.controller_mode, self.config)
        self.meta_trace.controller_checkpoint_loaded = int(loaded)
        self.meta_trace.controller_parameter_count = controller_parameter_count(self._controller)
        self.meta_trace.controller_frozen_on_eval = int(all(not p.requires_grad for p in self._controller.parameters()))
        self._raw_samples = self._artifact.get("target_raw_samples", [])
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.config.controller_random_seed))
        self._loss_ema = 0.0
        self._loss_slope_ema = 0.0
        self._prev_loss = None
        self._last_train_loss = 0.0
        self._param_meta: dict[int, dict[str, float]] = {}

    def set_progress(self, step: int, total_steps: int, learning_rate: float, weight_decay: float) -> None:
        self.config.normalized_step = float(step) / max(1.0, float(total_steps))
        self.config.total_steps = int(max(1, total_steps))
        self.config.learning_rate = float(learning_rate)
        self.config.weight_decay_value = float(weight_decay)

    def set_train_metrics(self, loss_value: float) -> None:
        loss = _finite(loss_value)
        if self._prev_loss is None:
            slope = 0.0
            self._loss_ema = loss
        else:
            slope = loss - float(self._prev_loss)
            self._loss_ema = 0.95 * self._loss_ema + 0.05 * loss
        self._loss_slope_ema = 0.95 * self._loss_slope_ema + 0.05 * slope
        self._prev_loss = loss
        self._last_train_loss = loss

    def _feature_vector(
        self,
        param: torch.nn.Parameter,
        source: torch.Tensor,
        witness: torch.Tensor,
        optimizer_state: Any | None,
    ) -> list[float]:
        src = source.detach().float()
        wit = witness.detach().float()
        w_entropy, w_eff_rank, w_top_sv, w_cond = _weight_spectral_stats(param)
        src_norm = _safe_norm(src)
        wit_norm = _safe_norm(wit)
        param_norm = _safe_norm(param)
        cos_sw = _safe_corr(src, wit)
        state = optimizer_state.get(param, {}) if isinstance(optimizer_state, dict) else {}
        exp_avg = state.get("exp_avg") if isinstance(state, dict) else None
        exp_avg_sq = state.get("exp_avg_sq") if isinstance(state, dict) else None
        mom_norm = _safe_norm(exp_avg) if exp_avg is not None else 0.0
        mom_grad_cos = _safe_corr(exp_avg, src) if exp_avg is not None else 0.0
        second_norm = _safe_norm(exp_avg_sq) if exp_avg_sq is not None else 0.0
        m_src = _matrix_view(src)
        m_wit = _matrix_view(wit)
        try:
            act_trace = float(m_src.matmul(m_src.transpose(0, 1)).trace().item()) if m_src.ndim == 2 else 0.0
            cot_trace = float(m_src.transpose(0, 1).matmul(m_src).trace().item()) if m_src.ndim == 2 else 0.0
        except Exception:
            act_trace = 0.0
            cot_trace = 0.0
        transfer_gap = max(0.0, 1.0 - cos_sw)
        debt = _finite(getattr(self, "_last_debt_proxy", 0.0))
        tail_q95 = _finite(getattr(self, "_last_tail_q95_proxy", 0.0))
        tail_q99 = _finite(getattr(self, "_last_tail_q99_proxy", 0.0))
        margin_debt = _finite(getattr(self, "_last_margin_proxy", 0.0))
        brier = _finite(getattr(self, "_last_brier_proxy", 0.0))
        trend_values = [
            _finite(self.config.normalized_step),
            math.tanh(_finite(self._loss_slope_ema)),
            cos_sw,
            transfer_gap,
            abs(src_norm - wit_norm) / max(src_norm + wit_norm, 1.0e-8),
            _safe_corr(src.abs(), wit.abs()),
            _safe_corr(src.sign(), wit.sign()),
            _finite(self.trace.source_witness_transfer_LCB),
            _finite(self.trace.source_witness_gap),
            math.tanh(_finite(self.trace.update_spectrum_drift)),
            _finite(self.trace.momentum_alignment),
            _finite(self.trace.OET_tangent_alignment),
            math.tanh(_finite(self.trace.sharpness_proxy_delta)),
            math.tanh(debt),
            math.tanh(tail_q95),
            math.tanh(tail_q99),
            math.tanh(margin_debt),
            math.tanh(brier),
            mom_grad_cos,
            math.tanh(second_norm / max(1.0, src_norm + wit_norm)),
            w_entropy,
            math.tanh(w_eff_rank / max(1.0, min(float(param.numel()), 1024.0))),
            math.tanh(math.log1p(max(1.0, w_cond)) / 10.0),
            _finite(self.config.learning_rate),
        ]
        feature_mode = str(self.config.feature_mode).lower()
        if feature_mode == "time_only":
            step = _finite(self.config.normalized_step)
            values = [step, math.sin(math.pi * step), math.cos(math.pi * step), _finite(self.config.learning_rate)]
        elif feature_mode in {"anti_leak", "anti_leak_v1"}:
            values = trend_values
        else:
            values = [
                1.0,
                _finite(self.config.normalized_step),
                _finite(self._loss_ema),
                _finite(self._loss_slope_ema),
                src_norm,
                wit_norm,
                math.log1p(src_norm),
                math.log1p(wit_norm),
                cos_sw,
                transfer_gap,
                abs(src_norm - wit_norm) / max(src_norm + wit_norm, 1.0e-8),
                param_norm,
                math.log1p(param_norm),
                w_entropy,
                w_eff_rank,
                math.log1p(w_top_sv),
                math.log1p(max(1.0, w_cond)),
                act_trace,
                cot_trace,
                math.log1p(abs(act_trace)),
                math.log1p(abs(cot_trace)),
                mom_norm,
                math.log1p(mom_norm),
                mom_grad_cos,
                second_norm,
                math.log1p(second_norm),
                debt,
                tail_q95,
                tail_q99,
                margin_debt,
                brier,
                math.log1p(max(0.0, debt)),
                _finite(self.config.learning_rate),
                math.log1p(max(0.0, _finite(self.config.learning_rate))),
                _finite(self.config.weight_decay_value),
                float(param.ndim),
                float(param.numel()),
                math.log1p(float(param.numel())),
                float(src.numel()),
                math.log1p(float(src.numel())),
                _safe_corr(src.abs(), wit.abs()),
                _safe_corr(src.sign(), wit.sign()),
                _finite(self.trace.source_witness_transfer_LCB),
                _finite(self.trace.source_witness_gap),
                _finite(self.trace.update_spectrum_drift),
                _finite(self.trace.momentum_alignment),
                _finite(self.trace.OET_tangent_alignment),
                _finite(self.trace.sharpness_proxy_delta),
            ]
        if len(values) < int(self.config.input_dim):
            values.extend([0.0] * (int(self.config.input_dim) - len(values)))
        return [_finite(v) for v in values[: int(self.config.input_dim)]]

    def _sample_random_raw(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._raw_samples:
            idx = int(torch.randint(0, len(self._raw_samples), (1,), generator=self._rng).item())
            raw = torch.tensor(self._raw_samples[idx], dtype=torch.float32).reshape(1, -1)
        else:
            raw = torch.zeros(1, len(META_FU_OUTPUT_NAMES), dtype=torch.float32)
        return raw.to(device=device, dtype=dtype)

    def _controller_output(self, z_values: list[float], device: torch.device) -> dict[str, float]:
        start = time.perf_counter()
        self._controller.to(device)
        z = torch.tensor(z_values, dtype=torch.float32, device=device).reshape(1, -1)
        mode = str(self.config.controller_mode)
        if mode.startswith("shuffled_state_feature_controller"):
            perm = torch.randperm(int(z.shape[-1]), generator=self._rng).to(device=z.device)
            z = z[:, perm]
        elif mode.startswith("random_state_feature_controller"):
            mu = float(z.detach().mean().cpu().item())
            sigma = float(z.detach().std(unbiased=False).clamp_min(1.0e-6).cpu().item())
            z = (torch.randn(tuple(z.shape), generator=self._rng, dtype=torch.float32) * sigma + mu).to(device=device)
        if mode.startswith("random") and not mode.startswith("random_state_feature_controller"):
            raw = self._sample_random_raw(device, z.dtype)
            _, bounded = self._controller.bounded(z)
            raw_for_bounds, _ = self._controller.bounded(z)
            del raw_for_bounds
            alpha_max = float(self.config.alpha_max)
            beta_max = float(self.config.beta_max)
            lambda_debt_max = float(self.config.lambda_debt_max)
            lambda_state_max = float(self.config.lambda_state_max)
            tau_min = float(self.config.tau_min)
            tau_max = float(self.config.tau_max)
            output = {
                "alpha": float((alpha_max * torch.sigmoid(raw[:, 0])).item()),
                "beta": float((beta_max * torch.sigmoid(raw[:, 1])).item()),
                "lambda_debt": float((lambda_debt_max * torch.sigmoid(raw[:, 2])).item()),
                "lambda_state": float((lambda_state_max * torch.sigmoid(raw[:, 3])).item()),
                "tau": float((tau_min + (tau_max - tau_min) * torch.sigmoid(raw[:, 4])).item()),
                "rho": float(torch.sigmoid(raw[:, 5]).item()),
                "raw_json": json.dumps([float(x) for x in raw.reshape(-1).detach().cpu().tolist()]),
            }
        else:
            with torch.no_grad():
                raw, bounded = self._controller.bounded(z)
            output = {name: float(tensor[0].detach().cpu().item()) for name, tensor in bounded.items()}
            output["raw_json"] = json.dumps([float(x) for x in raw.reshape(-1).detach().cpu().tolist()])
        self.meta_trace.controller_eval_ms += (time.perf_counter() - start) * 1000.0
        return output

    def _build_param_state(
        self,
        param: torch.nn.Parameter,
        source: torch.Tensor,
        witness: torch.Tensor,
        optimizer_state: Any | None,
    ) -> Any:
        z_values = self._feature_vector(param, source, witness, optimizer_state)
        output = self._controller_output(z_values, param.device)
        saved = (
            self.config.alpha,
            self.config.beta,
            self.config.debt_weight,
            self.config.state_transition_weight,
            self.config.tail_weight,
        )
        self.config.alpha = float(output["alpha"])
        self.config.beta = float(output["beta"])
        self.config.debt_weight = float(output["lambda_debt"])
        self.config.state_transition_weight = float(output["lambda_state"])
        self.config.tail_weight = max(float(self.config.tail_weight), 0.5 * float(output["lambda_debt"]))
        try:
            state = super()._build_param_state(param, source, witness, optimizer_state)
        finally:
            (
                self.config.alpha,
                self.config.beta,
                self.config.debt_weight,
                self.config.state_transition_weight,
                self.config.tail_weight,
            ) = saved
        rho = float(output["rho"])
        state.scalar_gate = 1.0 + rho * (float(state.scalar_gate) - 1.0)
        setattr(state, "meta_tau", float(output["tau"]))
        setattr(state, "meta_rho", rho)
        self._param_meta[id(param)] = {"tau": float(output["tau"]), "rho": rho}
        row = {
            "refresh_id": int(getattr(self, "_refresh_id", self.trace.observe_calls)),
            "param_id": id(param),
            "param_numel": int(param.numel()),
            **{name: output[name] for name in META_FU_OUTPUT_NAMES},
            "controller_mode": str(self.config.controller_mode),
            "controller_frozen": int(self.meta_trace.controller_frozen_on_eval),
            "train_only_controller_inputs": int(self.config.train_only_controller_inputs),
            "dataset_name_not_available_to_controller": int(self.config.no_dataset_or_seed_input),
            "seed_not_available_to_controller": int(self.config.no_dataset_or_seed_input),
            "raw_output_json": output["raw_json"],
        }
        self.meta_trace.controller_rows.append(row)
        feature_row = {"refresh_id": row["refresh_id"], "param_id": id(param)}
        for idx, value in enumerate(z_values):
            feature_row[f"z{idx:02d}"] = value
        self.meta_trace.feature_rows.append(feature_row)
        return state

    def transform(self, grad: torch.Tensor, param: torch.nn.Parameter, group: dict[str, Any] | None = None) -> torch.Tensor:
        if not self._inside_optimizer_step:
            return grad
        transformed = super().transform(grad, param, group)
        meta = self._param_meta.get(id(param), {})
        tau = float(meta.get("tau", self.config.tau_max))
        self.meta_trace.trust_clip_checks += 1
        raw_norm = float(grad.detach().float().norm().item())
        delta = transformed - grad
        delta_norm = float(delta.detach().float().norm().item())
        max_delta = max(1.0e-12, tau * max(raw_norm, 1.0e-12))
        if math.isfinite(delta_norm) and delta_norm > max_delta:
            transformed = grad + delta * (max_delta / max(delta_norm, 1.0e-12))
            self.meta_trace.trust_clip_events += 1
        return transformed

    def controller_trace_rows(self) -> list[dict[str, Any]]:
        return list(self.meta_trace.controller_rows)

    def feature_trace_rows(self) -> list[dict[str, Any]]:
        return list(self.meta_trace.feature_rows)

    def diagnostics(self) -> dict[str, float | int]:
        out = dict(super().diagnostics())
        rows = self.meta_trace.controller_rows
        for name in META_FU_OUTPUT_NAMES:
            vals = [float(r[name]) for r in rows if name in r]
            out[f"controller_output_{name}_mean"] = float(sum(vals) / len(vals)) if vals else 0.0
            out[f"controller_output_{name}_std"] = float(torch.tensor(vals).std(unbiased=False).item()) if len(vals) > 1 else 0.0
        rho_vals = [float(r["rho"]) for r in rows if "rho" in r]
        if rho_vals:
            hist = torch.histc(torch.tensor(rho_vals, dtype=torch.float32), bins=8, min=0.0, max=1.0)
            prob = hist / hist.sum().clamp_min(1.0)
            entropy = float((-(prob * torch.log(prob.clamp_min(1.0e-12))).sum() / math.log(8.0)).item())
        else:
            entropy = 0.0
        raw_vectors: list[list[float]] = []
        for row in rows:
            try:
                raw_vectors.append([float(x) for x in json.loads(str(row.get("raw_output_json", "[]")))])
            except Exception:
                pass
        if len(raw_vectors) > 1:
            raw_tensor = torch.tensor(raw_vectors, dtype=torch.float32)
            tv_norm = float((raw_tensor[1:] - raw_tensor[:-1]).norm(dim=1).mean().item())
            output_std_mean = float(raw_tensor.std(dim=0, unbiased=False).mean().item())
        elif raw_vectors:
            tv_norm = 0.0
            output_std_mean = 0.0
        else:
            tv_norm = 0.0
            output_std_mean = 0.0
        out.update(
            {
                "meta_controller_mode": str(self.config.controller_mode),
                "meta_controller_input_dim": int(self.config.input_dim),
                "meta_controller_parameter_count": int(self.meta_trace.controller_parameter_count),
                "meta_controller_checkpoint_loaded": int(self.meta_trace.controller_checkpoint_loaded),
                "meta_controller_frozen_on_meta_test": int(self.meta_trace.controller_frozen_on_eval),
                "meta_controller_train_only_inputs": int(self.config.train_only_controller_inputs),
                "meta_controller_no_dataset_seed_input": int(self.config.no_dataset_or_seed_input),
                "controller_output_entropy": entropy,
                "controller_TV_norm": tv_norm,
                "controller_output_std_mean": output_std_mean,
                "controller_trust_clip_rate": float(self.meta_trace.trust_clip_events / max(1, self.meta_trace.trust_clip_checks)),
                "controller_eval_ms": float(self.meta_trace.controller_eval_ms),
            }
        )
        return out


__all__ = [
    "META_FU_FEATURE_DIM",
    "META_FU_OUTPUT_NAMES",
    "BoundedScalarController",
    "MetaFUConfig",
    "MetaFUScalarOperator",
    "controller_parameter_count",
]
