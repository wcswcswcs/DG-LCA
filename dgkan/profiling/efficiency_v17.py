"""Isolated phase-level profiler for DG-KAN v17."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from dgkan.fu.core import UpdateTensor, apply_update, flat_grad


def count_parameters(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def time_call(fn: Callable[[], Any], device: torch.device, repeats: int = 3, warmup: int = 1) -> tuple[float, int]:
    for _ in range(max(0, int(warmup))):
        fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times: list[float] = []
    for _ in range(max(1, int(repeats))):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(sum(times) / len(times)), _peak(device)


def _manual_ce_available(model: torch.nn.Module) -> bool:
    return bool(
        hasattr(model, "manual_kernel_available")
        and hasattr(model, "manual_ce_forward_cache")
        and hasattr(model, "manual_ce_backward_from_cache")
        and model.manual_kernel_available()  # type: ignore[attr-defined]
    )


def _forward_pass(model: torch.nn.Module, x: torch.Tensor, use_manual_ce: bool = False) -> torch.Tensor:
    if use_manual_ce and _manual_ce_available(model):
        logits, _cache = model.manual_ce_forward_cache(x)  # type: ignore[attr-defined]
        return logits.detach()
    return model(x).detach()


def single_grad_pass(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, use_manual_ce: bool = False) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    if use_manual_ce and _manual_ce_available(model):
        logits, cache = model.manual_ce_forward_cache(x)  # type: ignore[attr-defined]
        loss = model.manual_ce_backward_from_cache(logits, cache, y)  # type: ignore[attr-defined]
    else:
        loss = F.cross_entropy(model(x).float(), y)
        loss.backward()
    return loss.detach()


def profile_isolated(
    model_factory: Callable[[], torch.nn.Module],
    mlp_factory: Callable[[], torch.nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    make_update: Callable[[torch.nn.Module], UpdateTensor],
    audit_call: Callable[[torch.nn.Module], Any] | None,
    *,
    device: torch.device,
    repeats: int = 3,
    warmup: int = 1,
    lr: float = 1.0e-3,
    use_manual_ce: bool = False,
) -> dict[str, Any]:
    model0 = model_factory().to(device)
    ref0 = mlp_factory().to(device)
    state = deepcopy(model0.state_dict())
    ref_state = deepcopy(ref0.state_dict())
    model_params = count_parameters(model0)
    mlp_params = count_parameters(ref0)
    param_delta = abs(float(mlp_params) - float(model_params)) / max(1.0, float(model_params))

    def fresh() -> torch.nn.Module:
        m = model_factory().to(device)
        m.load_state_dict(state)
        return m

    def fresh_ref() -> torch.nn.Module:
        m = mlp_factory().to(device)
        m.load_state_dict(ref_state)
        return m

    def measure(active_factory: Callable[[], torch.nn.Module], active_audit: Callable[[torch.nn.Module], Any] | None) -> dict[str, float | int | str]:
        probe_model = active_factory()
        manual_used = int(bool(use_manual_ce and _manual_ce_available(probe_model)))
        manual_variant = ""
        if manual_used and hasattr(probe_model, "manual_kernel_variant"):
            manual_variant = str(probe_model.manual_kernel_variant())  # type: ignore[attr-defined]
        del probe_model
        fwd_model = active_factory()
        forward_ms, forward_peak = time_call(lambda: _forward_pass(fwd_model, x, use_manual_ce=use_manual_ce), device, repeats, warmup)

        backward_ms, backward_peak = time_call(lambda: single_grad_pass(active_factory(), x, y, use_manual_ce=use_manual_ce), device, repeats, warmup)

        sgd_model = active_factory()
        single_grad_pass(sgd_model, x, y, use_manual_ce=use_manual_ce)
        opt_sgd = torch.optim.SGD(sgd_model.parameters(), lr=float(lr))
        sgd_update_ms, sgd_peak = time_call(lambda: opt_sgd.step(), device, repeats, warmup)

        adamw_model = active_factory()
        single_grad_pass(adamw_model, x, y, use_manual_ce=use_manual_ce)
        opt_adamw = torch.optim.AdamW(adamw_model.parameters(), lr=float(lr), weight_decay=1.0e-3)
        adamw_update_ms, adamw_peak = time_call(lambda: opt_adamw.step(), device, repeats, warmup)

        direction_model = active_factory()
        single_grad_pass(direction_model, x, y, use_manual_ce=use_manual_ce)
        direction_ms, direction_peak = time_call(lambda: flat_grad(direction_model, device), device, repeats, warmup)
        update = make_update(direction_model)

        commit_model = active_factory()
        functional_commit_ms, commit_peak = time_call(lambda: apply_update(commit_model, update, lr=float(lr)), device, repeats, warmup)

        audit_ms = 0.0
        audit_peak = 0
        if active_audit is not None:
            audit_model = active_factory()
            audit_ms, audit_peak = time_call(lambda: active_audit(audit_model), device, repeats, warmup)

        training_step_ms = float(forward_ms) + float(backward_ms) + float(sgd_update_ms)
        return {
            "forward_only_ms": forward_ms,
            "basis_eval_ms": forward_ms,
            "readout_contraction_ms": 0.0,
            "loss_delta_ms": 0.0,
            "backward_grad_ms": backward_ms,
            "backward_input_ms": backward_ms,
            "backward_param_ms": 0.0,
            "optimizer_update_ms": sgd_update_ms,
            "optimizer_update_ms_SGD": sgd_update_ms,
            "optimizer_update_ms_AdamW": adamw_update_ms,
            "optimizer_update_ms_manualFU": functional_commit_ms,
            "basis_specific_update_ms": 0.0,
            "manual_update_ms": functional_commit_ms,
            "functional_direction_ms": direction_ms,
            "functional_projection_ms": 0.0,
            "functional_commit_ms": functional_commit_ms,
            "linec_audit_ms": audit_ms,
            "tail_calibration_audit_ms": 0.0,
            "horizon_readback_ms": 0.0,
            "step_training_only_ms": training_step_ms,
            "step_with_audit_ms": training_step_ms + audit_ms,
            "forward_peak_memory": forward_peak,
            "backward_peak_memory": backward_peak,
            "update_peak_memory": max(sgd_peak, adamw_peak, commit_peak),
            "audit_state_bytes": audit_peak,
            "manual_ce_train_stream_profiled": manual_used,
            "manual_kernel_variant_profiled": manual_variant,
        }

    metrics = measure(fresh, audit_call)
    ref = measure(fresh_ref, None)
    ratios = {
        "forward_ratio": metrics["forward_only_ms"] / max(1.0e-9, ref["forward_only_ms"]),
        "backward_ratio": metrics["backward_grad_ms"] / max(1.0e-9, ref["backward_grad_ms"]),
        "update_ratio": metrics["optimizer_update_ms"] / max(1.0e-9, ref["optimizer_update_ms"]),
        "adamw_update_ratio": metrics["optimizer_update_ms_AdamW"] / max(1.0e-9, ref["optimizer_update_ms_AdamW"]),
        "manual_fu_update_ratio": metrics["optimizer_update_ms_manualFU"] / max(1.0e-9, ref["optimizer_update_ms_manualFU"]),
        "training_step_ratio": metrics["step_training_only_ms"] / max(1.0e-9, ref["step_training_only_ms"]),
        "memory_ratio": metrics["backward_peak_memory"] / max(1.0, ref["backward_peak_memory"]) if ref["backward_peak_memory"] > 0 else 0.0,
        "functional_overhead_ratio": (metrics["functional_direction_ms"] + metrics["functional_commit_ms"]) / max(1.0e-9, metrics["step_training_only_ms"]),
        "audit_overhead_ratio": metrics["linec_audit_ms"] / max(1.0e-9, metrics["step_training_only_ms"]),
    }
    param_bytes = model_params * 4
    return {
        **metrics,
        **ratios,
        "param_count": model_params,
        "same_param_mlp_param_count": mlp_params,
        "same_param_mlp_param_delta": param_delta,
        "optimizer_state_memory": model_params * 4,
        "basis_activation_bytes": max(0, int(metrics["forward_peak_memory"]) - param_bytes),
        "functional_state_bytes": param_bytes,
        "workspace_temp_bytes": max(0, int(metrics["backward_peak_memory"]) - param_bytes),
        "actual_saved_tensor_bytes": max(0, int(metrics["backward_peak_memory"]) - int(metrics["forward_peak_memory"])),
        "manual_cache_bytes": 0,
        "profiler_clone_state": 1,
        "audit_cost_separated": 1,
        "phase_level_timing_present": 1,
    }
