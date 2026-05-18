"""SNR instrumentation helpers for FC-PureKAN LQ functional updates.

The experiment runner owns protocol decisions and artifact writing.  This
module owns reusable role-aware SNR measurement primitives for the LQ model.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

from dgkan.models import fc_purekan_lq as lq
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig


@dataclass(frozen=True)
class SNRConfig:
    microbatch_count: int = 4
    tau1: float = 1.0
    tau2: float = 2.0
    eps: float = 1.0e-12
    temperature: float = 0.0


@dataclass
class EMARoleSNRState:
    beta: float
    step: int
    mean: List[torch.Tensor]
    second: List[torch.Tensor]

    @classmethod
    def zeros_like(cls, params: Sequence[torch.Tensor], beta: float) -> "EMARoleSNRState":
        return cls(
            beta=float(beta),
            step=0,
            mean=[torch.zeros_like(p, dtype=torch.float32) for p in params],
            second=[torch.zeros_like(p, dtype=torch.float32) for p in params],
        )


@dataclass
class ScalarRoleSNRState:
    beta: float
    step: int
    mean: List[torch.Tensor]
    second: List[torch.Tensor]

    @classmethod
    def zeros(cls, role_count: int, beta: float, device: torch.device) -> "ScalarRoleSNRState":
        return cls(
            beta=float(beta),
            step=0,
            mean=[torch.zeros((), device=device, dtype=torch.float32) for _ in range(role_count)],
            second=[torch.zeros((), device=device, dtype=torch.float32) for _ in range(role_count)],
        )


def role_names_for_basis(basis: str) -> List[str]:
    if basis == "t2":
        return ["lift_identity", "output_linear", "quadratic_coeff"]
    if basis == "t2t3":
        return ["lift_identity", "output_linear", "quadratic_coeff", "cubic_coeff"]
    if basis == "legendre23":
        return ["lift_identity", "output_linear", "legendre_p2_coeff", "legendre_p3_coeff"]
    raise ValueError(f"unknown LQ basis {basis}")


def clone_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def zero_like_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.zeros_like(p) for p in params]


def step_norm(step: Sequence[torch.Tensor]) -> torch.Tensor:
    total = None
    for part in step:
        value = part.float().square().sum()
        total = value if total is None else total + value
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total.sqrt()


def step_dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> torch.Tensor:
    total = None
    for a, b in zip(left, right):
        value = (a.float() * b.float()).sum()
        total = value if total is None else total + value
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total


def scale_step(step: Sequence[torch.Tensor], scale: torch.Tensor | float) -> List[torch.Tensor]:
    return [part * scale for part in step]


def add_steps(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor], alpha: torch.Tensor | float = 1.0) -> List[torch.Tensor]:
    return [a + b * alpha for a, b in zip(left, right)]


def apply_step(params: Sequence[torch.Tensor], step: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach() + delta.detach() for p, delta in zip(params, step)]


def gradient_descent_task_step(grads: Sequence[torch.Tensor], lr: float) -> List[torch.Tensor]:
    return [(-float(lr)) * g.detach() for g in grads]


def quadratic_coeff_direction(params: Sequence[torch.Tensor], basis: str) -> List[torch.Tensor]:
    """Return a simple geometry direction that damps quadratic-like channels.

    For the current v9.2.9 FC-PureKAN LQ mainline this is intentionally small:
    only the first nonlinear edge-basis coefficient is touched.  That keeps the
    direction interpretable for one-step safety audits and avoids hiding task
    loss inside a large arbitrary parameter update.
    """

    direction = zero_like_params(params)
    if basis in {"t2", "t2t3", "legendre23"} and len(direction) >= 3:
        direction[2] = -params[2].detach()
    return direction


def scale_direction_to_fraction_of_task_step(
    direction: Sequence[torch.Tensor],
    task_step: Sequence[torch.Tensor],
    fraction: float,
    eps: float = 1.0e-12,
) -> List[torch.Tensor]:
    raw_norm = step_norm(direction).clamp_min(eps)
    target_norm = step_norm(task_step) * float(fraction)
    return scale_step(direction, target_norm / raw_norm)


def project_step_to_task_safe(
    functional_step: Sequence[torch.Tensor],
    task_grads: Sequence[torch.Tensor],
    eps: float = 1.0e-12,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Remove the component that first-order task CE predicts harmful.

    If g^T delta <= 0, the step already predicts a non-increase in task CE and
    no projection is applied.  Otherwise subtract the projection onto g.
    """

    g_dot_delta = step_dot(task_grads, functional_step)
    if bool((g_dot_delta <= 0).detach().cpu()):
        return [d.detach().clone() for d in functional_step], torch.zeros((), device=functional_step[0].device, dtype=torch.float32)
    g_norm_sq = step_dot(task_grads, task_grads).clamp_min(eps)
    removed = scale_step(task_grads, g_dot_delta / g_norm_sq)
    projected = [d - r for d, r in zip(functional_step, removed)]
    return projected, step_norm(removed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(device: torch.device, fn) -> Tuple[float, Any]:
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0, out


def _safe_float(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu())


def _quantiles_json(values: torch.Tensor, max_items: int = 0) -> str:
    v = values.detach().flatten().float()
    if v.numel() == 0:
        return "{}"
    if max_items > 0 and v.numel() > max_items:
        # Deterministic uniform subsample for telemetry.  Exact active fraction
        # and means are computed separately; this only reduces histogram cost.
        idx = torch.linspace(0, v.numel() - 1, steps=max_items, device=v.device).long()
        v = v[idx]
    qs = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=v.device)
    qv = torch.quantile(v, qs)
    return json.dumps(
        {
            "min": _safe_float(qv[0]),
            "p10": _safe_float(qv[1]),
            "p25": _safe_float(qv[2]),
            "p50": _safe_float(qv[3]),
            "p75": _safe_float(qv[4]),
            "p90": _safe_float(qv[5]),
            "max": _safe_float(qv[6]),
        },
        sort_keys=True,
    )


def _snr_rows_for_role(
    *,
    role: str,
    grad_stack: torch.Tensor,
    cfg: SNRConfig,
) -> Dict[str, Any]:
    mu = grad_stack.mean(dim=0)
    centered = grad_stack - mu
    if grad_stack.shape[0] > 1:
        var = centered.square().sum(dim=0) / float(grad_stack.shape[0] - 1)
        var_norm = centered.square().sum() / float(grad_stack.shape[0] - 1)
    else:
        var = torch.zeros_like(mu)
        var_norm = torch.zeros((), device=grad_stack.device, dtype=grad_stack.dtype)
    snr = mu.square() / (var / max(1, int(grad_stack.shape[0]) - 1) + cfg.eps)
    snr_role = mu.square().sum() / (var_norm / max(1, int(grad_stack.shape[0]) - 1) + cfg.eps)
    return {
        "role": role,
        "grad_mean_norm": _safe_float(mu.norm()),
        "grad_var_norm": _safe_float(var_norm),
        "SNR_mean": _safe_float(snr.mean()),
        "SNR_role": _safe_float(snr_role),
        "SNR_p10": _safe_float(torch.quantile(snr.float(), 0.10)),
        "SNR_p50": _safe_float(torch.quantile(snr.float(), 0.50)),
        "SNR_p90": _safe_float(torch.quantile(snr.float(), 0.90)),
        "SNR_active_fraction_tau1": _safe_float((snr > cfg.tau1).float().mean()),
        "SNR_active_fraction_tau2": _safe_float((snr > cfg.tau2).float().mean()),
        "per_param_SNR_histogram": _quantiles_json(snr),
    }


def compute_microbatch_snr(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    cfg: SNRConfig,
    device: torch.device,
) -> Tuple[float, List[Dict[str, Any]]]:
    roles = role_names_for_basis(basis)
    _fwd, bwd = lq.functions_for_basis(basis)
    n = int(x.shape[0])
    if n % int(cfg.microbatch_count) != 0:
        raise ValueError("batch size must be divisible by microbatch_count for v9.2.9 SNR instrumentation")

    def _compute() -> List[Tuple[torch.Tensor, ...]]:
        chunks: List[Tuple[torch.Tensor, ...]] = []
        chunk_size = n // int(cfg.microbatch_count)
        for idx in range(int(cfg.microbatch_count)):
            lo = idx * chunk_size
            hi = lo + chunk_size
            pack = bwd(x[lo:hi], y[lo:hi], *params, mu, std, 2.0, 2.0)
            chunks.append(tuple(g.detach() for g in pack[1:]))
        return chunks

    elapsed_ms, micro_grads = _time_ms(device, _compute)
    rows: List[Dict[str, Any]] = []
    for param_idx, role in enumerate(roles):
        grad_stack = torch.stack([g[param_idx].reshape(-1).float() for g in micro_grads], dim=0)
        rows.append(_snr_rows_for_role(role=role, grad_stack=grad_stack, cfg=cfg))
    return elapsed_ms, rows


def _snr_row_from_ema(
    *,
    role: str,
    grad: torch.Tensor,
    mean: torch.Tensor,
    second: torch.Tensor,
    step: int,
    beta: float,
    cfg: SNRConfig,
    histogram_max_items: int,
) -> Dict[str, Any]:
    bias = max(1.0e-12, 1.0 - float(beta) ** max(1, int(step)))
    mu = mean / bias
    sec = second / bias
    var = (sec - mu.square()).clamp_min(cfg.eps)
    role_snr = mu.square().sum() / (var.sum() + cfg.eps)
    mu_flat = mu.detach().flatten()
    var_flat = var.detach().flatten()
    sampled = int(histogram_max_items > 0 and mu_flat.numel() > histogram_max_items)
    if sampled:
        idx = torch.linspace(0, mu_flat.numel() - 1, steps=histogram_max_items, device=mu_flat.device).long()
        mu_metric = mu_flat[idx]
        var_metric = var_flat[idx]
    else:
        mu_metric = mu_flat
        var_metric = var_flat
    snr_metric = mu_metric.square() / (var_metric + cfg.eps)
    return {
        "role": role,
        "grad_mean_norm": _safe_float(mu.norm()),
        "grad_var_norm": _safe_float(var.sum()),
        "SNR_mean": _safe_float(snr_metric.mean()),
        "SNR_role": _safe_float(role_snr),
        "SNR_p10": _safe_float(torch.quantile(snr_metric.float(), 0.10)),
        "SNR_p50": _safe_float(torch.quantile(snr_metric.float(), 0.50)),
        "SNR_p90": _safe_float(torch.quantile(snr_metric.float(), 0.90)),
        "SNR_active_fraction_tau1": _safe_float((snr_metric > cfg.tau1).float().mean()),
        "SNR_active_fraction_tau2": _safe_float((snr_metric > cfg.tau2).float().mean()),
        "per_param_SNR_histogram": _quantiles_json(snr_metric, max_items=0),
        "per_param_snr_sampled": sampled,
        "per_param_snr_sample_size": int(snr_metric.numel()),
        "instant_grad_norm": _safe_float(grad.norm()),
        "ema_step": int(step),
        "ema_beta": float(beta),
        "histogram_max_items": int(histogram_max_items),
    }


def measure_ema_role_snr_step(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    params: Sequence[torch.Tensor],
    states: Sequence[AdamWState],
    ema_state: EMARoleSNRState,
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    snr_cfg: SNRConfig,
    opt_cfg: ManualAdamWConfig,
    device: torch.device,
    update_fn,
    histogram_max_items: int = 2048,
) -> Tuple[float, float, float, List[Dict[str, Any]]]:
    """Run one normal manual step and update role-level EMA SNR telemetry.

    The backward pass is the normal task backward used for AdamW.  The measured
    SNR cost is only the extra EMA/statistics work, so this estimator can be
    compared against the P1 20% overhead gate without charging an extra
    ghost-batch backward.
    """

    roles = role_names_for_basis(basis)
    _fwd, bwd = lq.functions_for_basis(basis)
    _sync(device)
    total_start = time.perf_counter()
    pack = bwd(x, y, *params, mu, std, 2.0, 2.0)
    grads = [g.detach() for g in pack[1:]]

    _sync(device)
    snr_start = time.perf_counter()
    ema_state.step += 1
    one_minus_beta = 1.0 - float(ema_state.beta)
    rows: List[Dict[str, Any]] = []
    for idx, (role, grad) in enumerate(zip(roles, grads)):
        gf = grad.float()
        ema_state.mean[idx].mul_(ema_state.beta).add_(gf, alpha=one_minus_beta)
        ema_state.second[idx].mul_(ema_state.beta).addcmul_(gf, gf, value=one_minus_beta)
        rows.append(
            _snr_row_from_ema(
                role=role,
                grad=gf,
                mean=ema_state.mean[idx],
                second=ema_state.second[idx],
                step=ema_state.step,
                beta=ema_state.beta,
                cfg=snr_cfg,
                histogram_max_items=histogram_max_items,
            )
        )
    _sync(device)
    snr_ms = (time.perf_counter() - snr_start) * 1000.0

    update_fn(params, grads, states, opt_cfg)
    _sync(device)
    total_ms = (time.perf_counter() - total_start) * 1000.0
    baseline_step_ms = max(1.0e-12, total_ms - snr_ms)
    return snr_ms, baseline_step_ms, float(pack[0].detach().cpu()), rows


def measure_scalar_ema_role_snr_step(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    params: Sequence[torch.Tensor],
    states: Sequence[AdamWState],
    scalar_state: ScalarRoleSNRState,
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    snr_cfg: SNRConfig,
    opt_cfg: ManualAdamWConfig,
    device: torch.device,
    update_fn,
) -> Tuple[float, float, float, List[Dict[str, Any]]]:
    """Run one step and update scalar role-level EMA SNR telemetry.

    This is a deliberately lightweight gate: it does not estimate per-parameter
    SNR, and the emitted histogram field says so.  It is meant for role-level
    online event gating when ghost-batch or per-parameter EMA is too expensive.
    """

    roles = role_names_for_basis(basis)
    _fwd, bwd = lq.functions_for_basis(basis)
    _sync(device)
    total_start = time.perf_counter()
    pack = bwd(x, y, *params, mu, std, 2.0, 2.0)
    grads = [g.detach() for g in pack[1:]]

    _sync(device)
    snr_start = time.perf_counter()
    scalar_state.step += 1
    one_minus_beta = 1.0 - float(scalar_state.beta)
    role_snrs: List[torch.Tensor] = []
    grad_norms: List[torch.Tensor] = []
    for idx, grad in enumerate(grads):
        grad_norm = grad.float().norm()
        grad_norms.append(grad_norm)
        scalar_state.mean[idx].mul_(scalar_state.beta).add_(grad_norm, alpha=one_minus_beta)
        scalar_state.second[idx].mul_(scalar_state.beta).addcmul_(grad_norm, grad_norm, value=one_minus_beta)
        bias = max(1.0e-12, 1.0 - float(scalar_state.beta) ** max(1, int(scalar_state.step)))
        mean_hat = scalar_state.mean[idx] / bias
        second_hat = scalar_state.second[idx] / bias
        var_hat = (second_hat - mean_hat.square()).clamp_min(snr_cfg.eps)
        role_snrs.append(mean_hat.square() / (var_hat + snr_cfg.eps))
    if float(snr_cfg.temperature) > 0:
        temp = max(float(snr_cfg.temperature), 1.0e-6)
        active_fraction_tau1 = torch.stack(
            [torch.sigmoid((torch.log(s + snr_cfg.eps) - math.log(max(float(snr_cfg.tau1), snr_cfg.eps))) / temp) for s in role_snrs]
        ).mean()
        active_fraction_tau2 = torch.stack(
            [torch.sigmoid((torch.log(s + snr_cfg.eps) - math.log(max(float(snr_cfg.tau2), snr_cfg.eps))) / temp) for s in role_snrs]
        ).mean()
        active_fraction_scope = "smooth_role_fraction"
    else:
        active_fraction_tau1 = torch.stack([(s > snr_cfg.tau1).float() for s in role_snrs]).mean()
        active_fraction_tau2 = torch.stack([(s > snr_cfg.tau2).float() for s in role_snrs]).mean()
        active_fraction_scope = "role_fraction"
    rows: List[Dict[str, Any]] = []
    for role, snr, grad_norm in zip(roles, role_snrs, grad_norms):
        rows.append(
            {
                "role": role,
                "grad_mean_norm": _safe_float(grad_norm),
                "grad_var_norm": "",
                "SNR_mean": _safe_float(snr),
                "SNR_role": _safe_float(snr),
                "SNR_p10": _safe_float(snr),
                "SNR_p50": _safe_float(snr),
                "SNR_p90": _safe_float(snr),
                "SNR_active_fraction_tau1": _safe_float(active_fraction_tau1),
                "SNR_active_fraction_tau2": _safe_float(active_fraction_tau2),
                "per_param_SNR_histogram": "not_measured_role_scalar_ema",
                "per_param_snr_sampled": 0,
                "per_param_snr_sample_size": 0,
                "instant_grad_norm": _safe_float(grad_norm),
                "ema_step": int(scalar_state.step),
                "ema_beta": float(scalar_state.beta),
                "histogram_max_items": 0,
                "active_fraction_scope": active_fraction_scope,
                "snr_smooth_temperature": float(snr_cfg.temperature),
            }
        )
    _sync(device)
    snr_ms = (time.perf_counter() - snr_start) * 1000.0

    update_fn(params, grads, states, opt_cfg)
    _sync(device)
    total_ms = (time.perf_counter() - total_start) * 1000.0
    baseline_step_ms = max(1.0e-12, total_ms - snr_ms)
    return snr_ms, baseline_step_ms, float(pack[0].detach().cpu()), rows


def scalar_ema_snr_from_grads(
    *,
    grads: Sequence[torch.Tensor],
    scalar_state: ScalarRoleSNRState,
    basis: str,
    snr_cfg: SNRConfig,
) -> List[Dict[str, Any]]:
    """Update scalar role EMA SNR from already-computed task gradients.

    This helper is for one-step functional audits where the runner needs the
    task gradients for prediction/projection and should not pay for another
    backward just to measure role-level SNR.
    """

    roles = role_names_for_basis(basis)
    scalar_state.step += 1
    one_minus_beta = 1.0 - float(scalar_state.beta)
    role_snrs: List[torch.Tensor] = []
    grad_norms: List[torch.Tensor] = []
    for idx, grad in enumerate(grads):
        grad_norm = grad.detach().float().norm()
        grad_norms.append(grad_norm)
        scalar_state.mean[idx].mul_(scalar_state.beta).add_(grad_norm, alpha=one_minus_beta)
        scalar_state.second[idx].mul_(scalar_state.beta).addcmul_(grad_norm, grad_norm, value=one_minus_beta)
        bias = max(1.0e-12, 1.0 - float(scalar_state.beta) ** max(1, int(scalar_state.step)))
        mean_hat = scalar_state.mean[idx] / bias
        second_hat = scalar_state.second[idx] / bias
        var_hat = (second_hat - mean_hat.square()).clamp_min(snr_cfg.eps)
        role_snrs.append(mean_hat.square() / (var_hat + snr_cfg.eps))
    if float(snr_cfg.temperature) > 0:
        temp = max(float(snr_cfg.temperature), 1.0e-6)
        active_fraction_tau1 = torch.stack(
            [torch.sigmoid((torch.log(s + snr_cfg.eps) - math.log(max(float(snr_cfg.tau1), snr_cfg.eps))) / temp) for s in role_snrs]
        ).mean()
        active_fraction_tau2 = torch.stack(
            [torch.sigmoid((torch.log(s + snr_cfg.eps) - math.log(max(float(snr_cfg.tau2), snr_cfg.eps))) / temp) for s in role_snrs]
        ).mean()
        active_fraction_scope = "smooth_role_fraction"
    else:
        active_fraction_tau1 = torch.stack([(s > snr_cfg.tau1).float() for s in role_snrs]).mean()
        active_fraction_tau2 = torch.stack([(s > snr_cfg.tau2).float() for s in role_snrs]).mean()
        active_fraction_scope = "role_fraction"
    rows: List[Dict[str, Any]] = []
    for role, snr, grad_norm in zip(roles, role_snrs, grad_norms):
        rows.append(
            {
                "role": role,
                "SNR_mean": _safe_float(snr),
                "SNR_role": _safe_float(snr),
                "SNR_p10": _safe_float(snr),
                "SNR_p50": _safe_float(snr),
                "SNR_p90": _safe_float(snr),
                "SNR_active_fraction_tau1": _safe_float(active_fraction_tau1),
                "SNR_active_fraction_tau2": _safe_float(active_fraction_tau2),
                "instant_grad_norm": _safe_float(grad_norm),
                "ema_step": int(scalar_state.step),
                "ema_beta": float(scalar_state.beta),
                "active_fraction_scope": active_fraction_scope,
                "snr_smooth_temperature": float(snr_cfg.temperature),
            }
        )
    return rows


def measure_manual_step(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    params: Sequence[torch.Tensor],
    states: Sequence[AdamWState],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    cfg: ManualAdamWConfig,
    device: torch.device,
    update_fn,
) -> Tuple[float, float]:
    _fwd, bwd = lq.functions_for_basis(basis)

    def _step() -> torch.Tensor:
        pack = bwd(x, y, *params, mu, std, 2.0, 2.0)
        update_fn(params, pack[1:], states, cfg)
        return pack[0].detach()

    elapsed_ms, loss = _time_ms(device, _step)
    return elapsed_ms, float(loss.detach().cpu())


def summarize_stability(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, int, int, str], float]:
    groups: Dict[Tuple[str, str, int, int, str], List[float]] = {}
    for row in rows:
        key = (
            str(row.get("candidate_id", "")),
            str(row.get("dataset", "")),
            int(row.get("seed", 0)),
            int(row.get("microbatch_count", 0)),
            str(row.get("role", "")),
        )
        groups.setdefault(key, []).append(float(row.get("SNR_role", 0.0)))
    out: Dict[Tuple[str, str, int, int, str], float] = {}
    for key, values in groups.items():
        if len(values) <= 1:
            out[key] = 0.0
            continue
        t = torch.tensor(values, dtype=torch.float64)
        mean = float(t.mean().item())
        std = float(t.std(unbiased=True).item())
        out[key] = std / max(abs(mean), 1.0e-12)
    return out


def not_run_row(stage: str, artifact: str, reason: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "artifact": artifact,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
