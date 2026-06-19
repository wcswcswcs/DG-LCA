"""MLP-specific adaptive functional update controller for v22.16-M."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from dgkan.fu.mlp_source_usefulness import decide_source_usefulness
from dgkan.fu.real_source_manifold import SourceManifold, build_source_manifold, project_to_manifold


EPS = 1.0e-12


@dataclass
class MLPFUState:
    source_state: torch.Tensor | None = None
    source_age: int = 0
    release_count: int = 0
    refresh_count: int = 0
    history: list[torch.Tensor] = field(default_factory=list)
    history_loss_gains: list[float] = field(default_factory=list)
    manifold_cache: SourceManifold | None = None
    manifold_cache_step: int = -1


def _norm(v: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(v.detach().float()).clamp_min(EPS)


def _unit(v: torch.Tensor) -> torch.Tensor:
    vf = v.detach().float()
    return vf / _norm(vf)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    au = _unit(a).reshape(-1)
    bu = _unit(b).reshape(-1)
    n = min(au.numel(), bu.numel())
    if n == 0:
        return 0.0
    return float(torch.dot(au[:n], bu[:n]).clamp(-1.0, 1.0).item())


def _variant_mode(variant: str) -> str:
    if "MonitorOnly" in variant:
        return "monitor"
    if "StableRandomSourceControl" in variant:
        return "stable_random"
    if "RandomSourceControl" in variant:
        return "random"
    if "SignFlipSourceControl" in variant:
        return "signflip"
    if "CorruptSourceControl" in variant:
        return "corrupt"
    if "RealSourceManifold-k16" in variant:
        return "manifold16"
    if "RealSourceManifold-k8" in variant:
        return "manifold8"
    if "LowRankDirectProx-r16" in variant:
        return "lowrank16"
    if "LowRankDirectProx-r8" in variant:
        return "lowrank8"
    if "SourceRelease" in variant:
        return "release"
    if "SourceLossGated" in variant:
        return "loss_gated"
    if "WeakPredictive" in variant:
        return "weak"
    if "current" in variant or "replay" in variant:
        return "current"
    return "none"


def _init_control_source(mode: str, update_unit: torch.Tensor, seed: int, device: torch.device) -> torch.Tensor:
    if mode in {"random", "stable_random", "corrupt"}:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + (910003 if mode != "corrupt" else 920003))
        src = torch.randn(update_unit.numel(), generator=gen, device=device)
        if mode == "corrupt":
            src = torch.roll(src, shifts=max(1, src.numel() // 17))
        return _unit(src).to(device)
    if mode == "signflip":
        return (-update_unit).detach().clone()
    return update_unit.detach().clone()


def _lambda_for_mode(mode: str, risk: float, release: int) -> float:
    if mode in {"none", "monitor"}:
        return 0.0
    if mode == "current":
        return 0.55 + 0.35 * float(risk)
    if mode == "weak":
        return min(0.18, 0.04 + 0.20 * float(risk))
    if mode == "loss_gated":
        return 0.0 if release else min(0.22, 0.05 + 0.25 * float(risk))
    if mode == "release":
        return 0.0 if release else min(0.16, 0.04 + 0.18 * float(risk))
    if mode.startswith("manifold"):
        return 0.0 if release else min(0.20, 0.05 + 0.22 * float(risk))
    if mode.startswith("lowrank"):
        return 0.0 if release else min(0.16, 0.04 + 0.18 * float(risk))
    if mode in {"random", "stable_random", "signflip", "corrupt"}:
        return min(0.14, 0.05 + 0.16 * float(risk))
    return 0.0


def apply_mlp_fu_update(
    grad_flat: torch.Tensor,
    state: MLPFUState,
    *,
    variant: str,
    source_signal: torch.Tensor,
    source_loss_gain: float,
    step: int,
    seed: int,
    ratio_cap: float = 0.30,
) -> tuple[torch.Tensor, MLPFUState, dict[str, Any]]:
    grad = grad_flat.detach().float().reshape(-1)
    update = -grad
    update_norm = _norm(update)
    update_unit = update / update_norm
    mode = _variant_mode(variant)
    source_vec = source_signal.detach().float().reshape(-1).to(update.device)
    if source_vec.numel() != update.numel():
        repeat = (update.numel() + source_vec.numel() - 1) // max(1, source_vec.numel())
        source_vec = source_vec.repeat(repeat)[: update.numel()]
    source_unit_from_loss = _unit(source_vec).to(update.device)
    if state.source_state is None or state.source_state.numel() != update.numel():
        state.source_state = _init_control_source(mode, update_unit, seed, update.device)
        state.source_age = 0

    source_state_unit = _unit(state.source_state).to(update.device)
    align = _cosine(update_unit, source_state_unit)
    source_func = _cosine(source_unit_from_loss, source_state_unit)
    destructive = max(0.0, -align)
    nds = float(source_unit_from_loss.abs().mean().item())
    useful = decide_source_usefulness(
        source_loss_gain=source_loss_gain,
        source_func=source_func,
        destructive_projection=destructive,
        source_age=state.source_age,
        nds=nds,
        control_projection_fraction=0.0,
    )
    negative_source_loss = int(float(useful.boundary_distance) < 0.0)
    stale_refresh = int(useful.stale_source and not negative_source_loss)
    release = negative_source_loss if mode in {"loss_gated", "release", "manifold8", "manifold16", "lowrank8", "lowrank16"} else 0
    lam = _lambda_for_mode(mode, useful.risk_score, release)
    target = source_state_unit
    manifold: SourceManifold | None = None
    manifold_diag: dict[str, Any] = {}
    if mode.startswith("manifold"):
        k = 16 if mode.endswith("16") else 8
        if len(state.history) >= max(8, k) and (state.manifold_cache is None or int(step) - int(state.manifold_cache_step) >= 100):
            state.manifold_cache = build_source_manifold(state.history[-64:], state.history_loss_gains[-64:], dim=k)
            state.manifold_cache_step = int(step)
        manifold = state.manifold_cache
        target, manifold_diag = project_to_manifold(target, manifold)
        target = _unit(target).to(update.device)
    elif mode.startswith("lowrank"):
        chunks = 16 if mode.endswith("16") else 8
        if update.numel() >= chunks:
            usable = chunks * (target.numel() // chunks)
            if usable > 0:
                pooled = target[:usable].reshape(chunks, -1).mean(dim=1).repeat_interleave(target.numel() // chunks)
                target = target.clone()
                target[:usable] = pooled[:usable]
                target = _unit(target).to(update.device)

    guided = (update + float(lam) * target * update_norm) / (1.0 + float(lam))
    delta = guided - update
    ratio = float(_norm(delta).div(update_norm).item())
    if ratio > float(ratio_cap) and ratio > 0.0:
        delta = delta * (float(ratio_cap) / ratio)
        guided = update + delta
        ratio = float(_norm(delta).div(update_norm).item())

    if release:
        state.release_count += 1
        state.source_state = source_unit_from_loss.detach().clone()
        state.source_age = 0
    elif stale_refresh and mode in {"release", "manifold8", "manifold16", "lowrank8", "lowrank16"}:
        state.refresh_count += 1
        state.source_state = source_unit_from_loss.detach().clone()
        state.source_age = 0
    else:
        alpha = 0.010 if mode in {"weak", "loss_gated", "release"} else 0.005
        if mode.startswith("manifold"):
            alpha = 0.015
        state.source_state = _unit((1.0 - alpha) * source_state_unit + alpha * source_unit_from_loss).detach().clone()
        state.source_age += 1
    if useful.risk_score > 0.80 and not stale_refresh and mode not in {"random", "stable_random", "signflip", "corrupt"}:
        state.refresh_count += 1
    if len(state.history) < 512:
        state.history.append(source_unit_from_loss.detach().cpu())
        state.history_loss_gains.append(float(source_loss_gain))
    else:
        state.history.pop(0)
        state.history_loss_gains.pop(0)
        state.history.append(source_unit_from_loss.detach().cpu())
        state.history_loss_gains.append(float(source_loss_gain))

    diag = {
        "controller_mode_internal": mode,
        "risk_score": useful.risk_score,
        "lambda_t": float(lam),
        "intervention_flag": int(float(lam) > 0.0),
        "source_func": float(source_func),
        "source_loss_gain": float(source_loss_gain),
        "source_loss_boundary_distance": useful.boundary_distance,
        "source_age": int(state.source_age),
        "source_release_count": int(state.release_count),
        "source_refresh_count": int(state.refresh_count),
        "stale_refresh": stale_refresh,
        "negative_source_loss_release": release,
        "source_state_alignment": align,
        "source_state_decay_rate": 0.985,
        "destructive_projection": destructive,
        "NDS": nds,
        "control_projection_fraction": 0.0,
        "ordinary_update_norm": float(update_norm.item()),
        "controller_update_norm": float(_norm(guided - update).item()),
        "controller_to_base_update_ratio": ratio,
        "prox_residual_before": float(_norm(update_unit - source_state_unit).item()),
        "prox_residual_after": float(_norm(_unit(guided) - source_state_unit).item()),
        **manifold_diag,
    }
    return guided.to(dtype=grad_flat.dtype).reshape_as(grad_flat), state, diag


__all__ = ["MLPFUState", "apply_mlp_fu_update"]
