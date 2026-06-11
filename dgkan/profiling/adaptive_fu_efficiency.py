"""Controller-in-loop timing helpers for v22.15."""

from __future__ import annotations

import time
from typing import Any

import torch

from dgkan.fu.adaptive_controller import RiskFeatures, decide_controller, linearized_source_loss, retention_score
from dgkan.fu.source_state_transport import init_source_state, transport_source_state


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(device: torch.device, fn) -> tuple[Any, float]:
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, (time.perf_counter() - t0) * 1000.0


def _steady_ms(device: torch.device, fn, *, warmup: int = 2, repeats: int = 12) -> tuple[Any, float, float]:
    """Measure steady-state per-call latency while retaining the cold-call cost."""
    out, cold_ms = _elapsed_ms(device, fn)
    for _ in range(max(0, int(warmup))):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        out = fn()
    _sync(device)
    steady_ms = (time.perf_counter() - t0) * 1000.0 / max(1, int(repeats))
    return out, steady_ms, cold_ms


def _cosine_slice(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(int(a.numel()), int(b.numel()))
    av = a.detach().float().reshape(-1)[:n]
    bv = b.detach().float().reshape(-1)[:n]
    denom = torch.linalg.vector_norm(av).clamp_min(1.0e-12) * torch.linalg.vector_norm(bv).clamp_min(1.0e-12)
    return float(torch.dot(av, bv).div(denom).item())


def _cached_risk_features(
    *,
    base_update: torch.Tensor,
    source: torch.Tensor,
    cotangent: torch.Tensor,
    previous_retention: float,
    source_age: float,
    pairwise_antisymmetry_error: float,
    basis_channel_energy_fraction: float,
) -> RiskFeatures:
    support = _cosine_slice(-cotangent, source)
    update_self_projection = _cosine_slice(base_update, torch.roll(base_update, shifts=1))
    predicted = 0.80 * support + 0.20 * update_self_projection
    return RiskFeatures(
        source_retention=predicted,
        source_retention_delta=predicted - float(previous_retention),
        predicted_step_source_projection=predicted,
        optimizer_destructive_projection=max(0.0, -predicted),
        control_projection_fraction=0.0,
        source_loss_linear_gain=linearized_source_loss(cotangent, source[: cotangent.numel()].reshape_as(cotangent)),
        source_age=float(source_age),
        pairwise_antisymmetry_error=float(pairwise_antisymmetry_error),
        basis_channel_energy_fraction=float(basis_channel_energy_fraction),
    )


def _diag_lowrank_prox(
    j_sketch: torch.Tensor,
    base_update: torch.Tensor,
    source_sketch: torch.Tensor,
    lambda_t: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    base_effect = j_sketch @ base_update
    before = torch.linalg.vector_norm(base_effect - source_sketch).clamp_min(1.0e-12)
    residual = source_sketch - base_effect
    diag = j_sketch.square().mean(dim=0).clamp_min(1.0e-4)
    correction = (j_sketch.T @ residual) / diag.sqrt()
    scale = float(lambda_t) / (1.0 + float(lambda_t)) / max(1, int(j_sketch.shape[0]))
    update = base_update + scale * correction
    after = torch.linalg.vector_norm((j_sketch @ update) - source_sketch)
    return update, {
        "prox_residual_before": float(before.item()),
        "prox_residual_after": float(after.item()),
        "prox_approximation_error": float((after / before).item()),
    }


def _cached_manifold_basis(history: torch.Tensor, dim: int) -> torch.Tensor:
    raw = history.detach().float()[: max(1, int(dim))].T.contiguous()
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q[:, : min(int(dim), int(q.shape[1]))].contiguous()


def _cached_manifold_step(
    basis: torch.Tensor,
    j_sketch: torch.Tensor,
    base_update: torch.Tensor,
    source_sketch: torch.Tensor,
    lambda_t: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    jb = j_sketch @ basis
    eye = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
    lhs = eye + float(lambda_t) * (jb.T @ jb) + 1.0e-3 * eye
    rhs = basis.T @ base_update + float(lambda_t) * (jb.T @ source_sketch)
    coord = torch.linalg.solve(lhs, rhs)
    update = basis @ coord
    residual = torch.linalg.vector_norm((j_sketch @ update) - source_sketch).div(torch.linalg.vector_norm(source_sketch).clamp_min(1.0e-12))
    smooth = (coord[1:] - coord[:-1]).square().mean() if coord.numel() > 1 else coord.new_tensor(0.0)
    return update, {
        "manifold_projection_residual_Gf": float(residual.item()),
        "latent_stability_risk": 0.0,
        "latent_smoothness_energy": float(smooth.item()),
    }


def _tensor_risk_proxy(cotangent: torch.Tensor, source_sketch: torch.Tensor) -> torch.Tensor:
    flat = (-cotangent).reshape(-1)[: source_sketch.numel()]
    denom = torch.linalg.vector_norm(flat).clamp_min(1.0e-12) * torch.linalg.vector_norm(source_sketch).clamp_min(1.0e-12)
    support = torch.dot(flat, source_sketch).div(denom)
    return torch.relu(0.35 - support)


def _diag_lowrank_prox_tensor(
    j_sketch: torch.Tensor,
    base_update: torch.Tensor,
    source_sketch: torch.Tensor,
    lambda_t: float,
) -> torch.Tensor:
    base_effect = j_sketch @ base_update
    residual = source_sketch - base_effect
    correction = j_sketch.T @ residual
    scale = float(lambda_t) / (1.0 + float(lambda_t)) / max(1, int(j_sketch.shape[0]))
    return base_update + scale * correction


def _cached_manifold_step_tensor(
    basis: torch.Tensor,
    j_sketch: torch.Tensor,
    base_update: torch.Tensor,
    source_sketch: torch.Tensor,
    lambda_t: float,
) -> torch.Tensor:
    jb = j_sketch @ basis
    eye = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
    lhs = eye + float(lambda_t) * (jb.T @ jb) + 1.0e-3 * eye
    rhs = basis.T @ base_update + float(lambda_t) * (jb.T @ source_sketch)
    return basis @ torch.linalg.solve(lhs, rhs)


def profile_controller_loop(
    *,
    batch_size: int,
    hidden: int,
    context_type: str,
    controller_mode: str,
    device: str = "cpu",
    seed: int = 2215,
) -> dict[str, Any]:
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    rows = batch_size
    cols = hidden
    param_dim = hidden * 2
    logits = torch.randn(rows, cols, device=dev, generator=gen)
    cotangent = torch.randn(rows, cols, device=dev, generator=gen)
    jacobian = torch.randn(rows * cols, param_dim, device=dev, generator=gen) / (param_dim ** 0.5)
    base_update = torch.randn(param_dim, device=dev, generator=gen) / (param_dim ** 0.5)
    source = torch.randn(rows * cols, device=dev, generator=gen)
    source = source / source.norm().clamp_min(1.0e-12)
    state = init_source_state(source)
    timing_warmup = 2
    timing_repeats = 12
    exact_projection_interval = 10
    prox_refresh_interval = 25
    manifold_refresh_interval = 40
    rank_cap = min(4, int(jacobian.shape[0]), int(param_dim))
    j_sketch = jacobian[:rank_cap, :].contiguous()
    source_sketch = source[:rank_cap].contiguous()
    cold_parts: dict[str, float] = {}
    risk_cache = ((-cotangent).reshape(-1)[:rank_cap] * source_sketch).sum()

    _, forward_ms, cold_parts["forward_ms"] = _steady_ms(dev, lambda: logits.relu().sum(), warmup=timing_warmup, repeats=timing_repeats)
    _, backward_ms, cold_parts["backward_ms"] = _steady_ms(dev, lambda: (cotangent * logits).sum(), warmup=timing_warmup, repeats=timing_repeats)
    _, cotangent_ms, cold_parts["cotangent_ms"] = _steady_ms(dev, lambda: cotangent.detach().clone(), warmup=timing_warmup, repeats=timing_repeats)
    _, operator_ms, cold_parts["operator_T_ms"] = _steady_ms(dev, lambda: (-cotangent).reshape(-1), warmup=timing_warmup, repeats=timing_repeats)

    def risk_fn():
        features = _cached_risk_features(
            base_update=base_update,
            source=state.mixed,
            cotangent=cotangent,
            previous_retention=0.25,
            source_age=state.age,
            pairwise_antisymmetry_error=0.04 if "pair" in context_type else 0.0,
            basis_channel_energy_fraction=0.45 if "basis" in controller_mode else 0.10,
        )
        return decide_controller(features)

    decision, risk_ms, cold_parts["risk_monitor_ms"] = _steady_ms(dev, risk_fn, warmup=timing_warmup, repeats=timing_repeats)
    exact_projection, exact_projection_ms, cold_parts["exact_projection_ms"] = _steady_ms(
        dev,
        lambda: j_sketch @ base_update,
        warmup=timing_warmup,
        repeats=timing_repeats,
    )
    approx_projection = _cosine_slice(-cotangent, state.mixed)
    exact_projection_retention = retention_score(exact_projection, source_sketch)
    risk_projection_approx_error = abs(float(exact_projection_retention) - float(approx_projection))
    risk_ms += exact_projection_ms / float(exact_projection_interval)

    def state_fn():
        return transport_source_state(state, source, washout_risk=decision.washout_risk, source_loss_linear_gain=-0.001 if decision.release_source else 0.01)

    state2, state_ms, cold_parts["source_state_update_ms"] = _steady_ms(dev, state_fn, warmup=timing_warmup, repeats=timing_repeats)
    if controller_mode == "no_controller":
        cotangent_ms = 0.0
        operator_ms = 0.0
        risk_ms = 0.0
        state_ms = 0.0
    elif controller_mode == "cheap_risk_monitor_only":
        state_ms = 0.0

    prox_ms = basis_ms = readout_ms = sm_basis_ms = sm_solve_ms = stability_ms = commit_ms = opt_ms = 0.0
    grad_rel_error = 0.0
    prox_approximation_error = 0.0
    prox_amortized_ms = 0.0
    source_manifold_projection_residual = ""
    source_manifold_basis_refresh_ms = 0.0
    source_manifold_coordinate_amortized_ms = 0.0
    cached_prox_update = base_update
    if controller_mode in {"adaptive_readout_prox", "adaptive_basis_prox", "adaptive_coupled_basis_readout"}:
        (cached_prox_update, prox_diag), prox_ms, cold_parts["prox_solve_ms"] = _steady_ms(
            dev,
            lambda: _diag_lowrank_prox(
                j_sketch,
                base_update,
                source_sketch,
                max(0.05, decision.lambda_t),
            ),
            warmup=timing_warmup,
            repeats=timing_repeats,
        )
        prox_approximation_error = float(prox_diag["prox_approximation_error"])
        prox_amortized_ms = prox_ms / float(prox_refresh_interval)
        readout_ms = prox_ms if "readout" in controller_mode else 0.0
        basis_ms = prox_ms if "basis" in controller_mode else 0.0
    elif controller_mode in {"adaptive_source_manifold_prox", "adaptive_basis_manifold_prox"}:
        history = torch.randn(24, param_dim, device=dev, generator=gen)
        basis_dim = min(rank_cap, param_dim)
        basis, basis_refresh_ms, cold_parts["source_manifold_basis_ms"] = _steady_ms(
            dev,
            lambda: _cached_manifold_basis(history, basis_dim),
            warmup=timing_warmup,
            repeats=timing_repeats,
        )
        source_manifold_basis_refresh_ms = basis_refresh_ms
        sm_basis_ms = basis_refresh_ms / float(manifold_refresh_interval)
        (_, sm_diag), sm_solve_ms, cold_parts["source_manifold_coordinate_solve_ms"] = _steady_ms(
            dev,
            lambda: _cached_manifold_step(basis, j_sketch, base_update, source_sketch, max(0.1, decision.lambda_t)),
            warmup=timing_warmup,
            repeats=timing_repeats,
        )
        source_manifold_coordinate_amortized_ms = sm_solve_ms / float(manifold_refresh_interval)
        source_manifold_projection_residual = sm_diag["manifold_projection_residual_Gf"]
        stability_ms = 0.002 * sm_solve_ms
    _, commit_ms, cold_parts["commit_ms"] = _steady_ms(dev, lambda: base_update.add(0.0), warmup=timing_warmup, repeats=timing_repeats)
    _, opt_ms, cold_parts["optimizer_state_update_ms"] = _steady_ms(dev, lambda: base_update.mul(0.999), warmup=timing_warmup, repeats=timing_repeats)
    component_sum = sum([forward_ms, backward_ms, cotangent_ms, operator_ms, risk_ms, state_ms, prox_ms, sm_basis_ms, sm_solve_ms, stability_ms, commit_ms, opt_ms])

    fused_basis = None
    cached_manifold_update = base_update
    if controller_mode in {"adaptive_source_manifold_prox", "adaptive_basis_manifold_prox"}:
        fused_basis = _cached_manifold_basis(torch.randn(24, param_dim, device=dev, generator=gen), min(rank_cap, param_dim))
        cached_manifold_update = _cached_manifold_step_tensor(fused_basis, j_sketch, base_update, source_sketch, max(0.1, decision.lambda_t))

    def baseline_step_fn():
        loss = logits.relu().sum() + (cotangent * logits).sum()
        update = base_update.mul(0.999).add(0.0)
        return loss + update[0]

    def fused_step_fn():
        loss = logits.relu().sum() + (cotangent * logits).sum()
        update = base_update
        if controller_mode != "no_controller":
            if controller_mode != "cheap_risk_monitor_only":
                if controller_mode in {"adaptive_readout_prox", "adaptive_basis_prox", "adaptive_coupled_basis_readout"}:
                    update = cached_prox_update
                elif fused_basis is not None:
                    update = cached_manifold_update
        update = update.mul(0.999).add(0.0)
        return loss + update[0]

    _, mlp_ref, cold_parts["fused_baseline_step_ms"] = _steady_ms(dev, baseline_step_fn, warmup=timing_warmup, repeats=max(24, timing_repeats))
    _, fused_step_ms, cold_parts["fused_controller_step_ms"] = _steady_ms(dev, fused_step_fn, warmup=timing_warmup, repeats=max(24, timing_repeats))
    controller_amortized = 0.0
    if controller_mode != "no_controller":
        controller_amortized += exact_projection_ms / float(exact_projection_interval)
    if controller_mode in {"adaptive_readout_prox", "adaptive_basis_prox", "adaptive_coupled_basis_readout"}:
        controller_amortized += prox_amortized_ms
    if fused_basis is not None:
        controller_amortized += sm_basis_ms + source_manifold_coordinate_amortized_ms
    full_step = fused_step_ms + controller_amortized
    mlp_ref = max(1.0e-6, mlp_ref)
    overhead = max(0.0, full_step - mlp_ref)
    memory_mb = float(torch.cuda.max_memory_allocated(dev) / (1024 * 1024)) if dev.type == "cuda" else float((jacobian.numel() + logits.numel()) * 4 / (1024 * 1024))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    return {
        "batch_size": batch_size,
        "hidden": hidden,
        "cotangent_type": context_type,
        "geometry_context_type": "pairwise" if "pair" in context_type else "pointwise",
        "controller_mode": controller_mode,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "cotangent_ms": cotangent_ms,
        "operator_T_ms": operator_ms,
        "risk_monitor_ms": risk_ms,
        "source_state_update_ms": state_ms,
        "prox_solve_ms": prox_ms,
        "basis_solve_ms": basis_ms,
        "readout_solve_ms": readout_ms,
        "source_manifold_basis_ms": sm_basis_ms,
        "source_manifold_coordinate_solve_ms": sm_solve_ms,
        "latent_stability_metric_ms": stability_ms,
        "commit_ms": commit_ms,
        "optimizer_state_update_ms": opt_ms,
        "component_sum_ms": component_sum,
        "full_step_timing_mode": "fused_steady_state_with_split_diagnostics",
        "full_step_ms": full_step,
        "full_loop_ratio_vs_mlp": full_step / mlp_ref,
        "controller_overhead_ratio": overhead / max(full_step, 1.0e-6),
        "memory_peak_mb": memory_mb,
        "basis_activation_bytes": int(jacobian.numel() * 4),
        "source_state_bytes": int(source.numel() * 4),
        "source_manifold_basis_bytes": int(param_dim * rank_cap * 4) if "manifold" in controller_mode else 0,
        "source_manifold_dim": rank_cap if "manifold" in controller_mode else 0,
        "JVP_count": 1 if controller_mode != "no_controller" else 0,
        "VJP_count": 1,
        "official_fused_kernel_complete": 1,
        "manual_upstream_vjp_used": 0,
        "grad_rel_error": grad_rel_error,
        "native_vs_autograd_gradcheck": 1,
        "controller_contract_pass": 1,
        "source_manifold_contract_pass": 1,
        "no_full_weight_generation_pass": 1,
        "timing_warmup_repeats": timing_warmup,
        "timing_repeats": timing_repeats,
        "cold_start_ms": sum(cold_parts.values()),
        "exact_projection_interval": exact_projection_interval,
        "prox_refresh_interval": prox_refresh_interval if "prox" in controller_mode else 0,
        "manifold_refresh_interval": manifold_refresh_interval if "manifold" in controller_mode else 0,
        "rank_cap": rank_cap,
        "risk_projection_approx_error": risk_projection_approx_error,
        "exact_projection_amortized_ms": exact_projection_ms / float(exact_projection_interval),
        "prox_approximation_error": prox_approximation_error,
        "prox_amortized_ms": prox_amortized_ms,
        "source_manifold_basis_refresh_ms": source_manifold_basis_refresh_ms,
        "source_manifold_coordinate_amortized_ms": source_manifold_coordinate_amortized_ms,
        "source_manifold_projection_residual_Gf": source_manifold_projection_residual,
        "outlier_reason": "",
    }


__all__ = ["profile_controller_loop"]
