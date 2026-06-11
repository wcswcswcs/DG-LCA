"""Metric-as-dynamics commit for v22.10 constructive source targets.

The official v22.10 commit path is loss agnostic: a pre-built function-space
target is committed by readout/block projection and scored by target readback
geometry against matched controls.  A supervised loss may be supplied by older
callers for historical hidden-gradient repairs, but v22.10 passes ``y=None``.
"""

from __future__ import annotations

from math import isfinite
import time
from typing import Any

import torch
import torch.nn.functional as F

from dgkan.fu.basis_channel_metric import basis_channel_energy, parameter_channel_slices
from dgkan.fu.core import flat_params, load_flat_params
from dgkan.fu.jacobian_sketch import function_displacement_cosine


EPS = 1.0e-8


def _readout_parameter(model: torch.nn.Module) -> tuple[str, torch.nn.Parameter]:
    for name, p in model.named_parameters():
        if p.requires_grad and ("w2" in name.lower() or "readout" in name.lower() or "classifier" in name.lower()):
            if p.ndim == 2:
                return name, p
    for name, p in model.named_parameters():
        if p.requires_grad and p.ndim == 2:
            return name, p
    raise ValueError("no 2D readout-like trainable parameter found")


def _split2(n: int) -> tuple[tuple[int, int], tuple[int, int]]:
    mid = max(1, n // 2)
    return (0, mid), (mid, n)


def _readout_update_vector(model: torch.nn.Module, delta_w: torch.Tensor, readout_name: str) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == readout_name:
            chunks.append((-delta_w).to(device=p.device, dtype=p.dtype).reshape(-1))
        else:
            chunks.append(torch.zeros_like(p).reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0)


def _solve_readout_delta(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    damping: float,
) -> tuple[str, torch.Tensor, torch.Tensor, float]:
    readout_name, _ = _readout_parameter(model)
    feats = model.frozen_readout_features(x).detach().float()
    gram = feats.T @ feats + float(damping) * torch.eye(int(feats.shape[1]), device=feats.device, dtype=feats.dtype)
    rhs = feats.T @ target.detach().float()
    try:
        delta_w = torch.linalg.solve(gram, rhs)
    except Exception:
        delta_w = torch.linalg.lstsq(gram, rhs).solution
    cond = float(torch.linalg.cond(gram.detach().float()).item()) if gram.numel() else 0.0
    return readout_name, delta_w, gram, cond


def _apply_block_mask(model: torch.nn.Module, update_vec: torch.Tensor, block_role: str) -> torch.Tensor:
    role = str(block_role or "all")
    if role == "all":
        return update_vec
    allowed = {
        "readout_only": {"readout"},
        "hidden_only": {"hidden"},
        "hidden_readout": {"hidden", "readout"},
        "optimizer_state_only": set(),
    }.get(role, {"readout"})
    mask = torch.zeros_like(update_vec)
    for row in parameter_channel_slices(model):
        if str(row.get("channel")) in allowed:
            mask[int(row["start"]) : int(row["end"])] = 1.0
    return update_vec * mask.to(device=update_vec.device, dtype=update_vec.dtype)


def solve_external_target_commit(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor | None,
    target_delta: torch.Tensor,
    *,
    solver_level: str = "S4.1-readout-exact",
    block_role: str = "readout_only",
    damping: float = 1.0e-3,
    optimizer_state_write_fraction: float = 0.0,
    fit_scope: str = "split_A",
    seed: int = 2210,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve a readout/block commit for an externally constructed target."""

    start = time.perf_counter()
    before = flat_params(model).detach()
    (a0, a1), (b0, b1) = _split2(int(x.shape[0]))
    scope = str(fit_scope or "split_A")
    if scope == "all_train_stream":
        xb = x
        target_train = target_delta.detach().float()
    else:
        xb = x[a0:a1]
        target_train = target_delta[a0:a1].detach().float()
    x2 = x[b0:b1]
    target_readback = target_delta.detach().float()
    with torch.no_grad():
        base_all = model(x).detach().float()
        base2 = model(x2).detach().float()
    readout_name, delta_w, gram, cond = _solve_readout_delta(model, xb, target_train, damping=damping)
    update_vec = _readout_update_vector(model, delta_w, readout_name)
    if block_role in {"hidden_only", "hidden_readout"} and y is not None:
        yb = y if scope == "all_train_stream" else y[a0:a1]
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb.to(dtype=torch.long)).backward()
        grad_chunks = []
        for p in model.parameters():
            if p.requires_grad:
                grad_chunks.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1))
        hidden_grad = torch.cat(grad_chunks) if grad_chunks else torch.zeros_like(update_vec)
        hidden_mask = torch.zeros_like(update_vec)
        for row in parameter_channel_slices(model):
            if str(row.get("channel")) == "hidden":
                hidden_mask[int(row["start"]) : int(row["end"])] = 1.0
        hidden_grad = hidden_grad * hidden_mask.to(device=hidden_grad.device, dtype=hidden_grad.dtype)
        if float(torch.linalg.vector_norm(hidden_grad).item()) > EPS:
            update_vec = update_vec + hidden_grad / torch.linalg.vector_norm(hidden_grad).clamp_min(EPS) * (0.10 * torch.linalg.vector_norm(update_vec).clamp_min(EPS))
    update_vec = _apply_block_mask(model, update_vec, block_role)
    if block_role == "optimizer_state_only":
        update_vec = torch.zeros_like(update_vec)

    with torch.no_grad():
        load_flat_params(model, before - update_vec.to(device=before.device, dtype=before.dtype))
        moved_all = model(x).detach().float()
        moved2 = model(x2).detach().float()
        load_flat_params(model, before)
    actual = moved_all - base_all
    actual2 = moved2 - base2
    residual = target_readback.to(device=actual.device, dtype=actual.dtype) - actual
    target_norm = torch.linalg.vector_norm(target_readback).clamp_min(EPS)
    centered = target_readback.reshape(-1) - target_readback.reshape(-1).mean()
    r2_denom = torch.sum(centered.square()).clamp_min(EPS)
    target2 = target_delta[b0:b1].detach().float()
    target2_norm = torch.linalg.vector_norm(target2).clamp_min(EPS)
    b2_residual = torch.linalg.vector_norm(target2.to(device=actual2.device, dtype=actual2.dtype) - actual2)
    b2_transfer = function_displacement_cosine(actual2, target2) - float((b2_residual / target2_norm).item())

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 311)
    random_target = torch.randn(target_delta.shape, generator=gen, dtype=target_delta.detach().cpu().dtype).to(device=target_delta.device, dtype=target_delta.dtype)
    random_target = random_target * (torch.linalg.vector_norm(target_delta).clamp_min(EPS) / torch.linalg.vector_norm(random_target).clamp_min(EPS))
    random_target_train = random_target if scope == "all_train_stream" else random_target[a0:a1]
    _, random_delta_w, _, _ = _solve_readout_delta(model, xb, random_target_train, damping=damping)
    random_update = _apply_block_mask(model, _readout_update_vector(model, random_delta_w, readout_name), block_role)
    with torch.no_grad():
        load_flat_params(model, before - random_update.to(device=before.device, dtype=before.dtype))
        random2 = model(x2).detach().float()
        load_flat_params(model, before)
    random_actual2 = random2 - base2
    random_residual = torch.linalg.vector_norm(target2.to(device=random_actual2.device, dtype=random_actual2.dtype) - random_actual2)
    random_gain = function_displacement_cosine(random_actual2, target2) - float((random_residual / target2_norm).item())
    energies = basis_channel_energy(model, update_vec)
    diagnostics: dict[str, Any] = {
        "solver_level": solver_level,
        "block_role": block_role,
        "readout_parameter": readout_name,
        "JVP_count": 1,
        "VJP_count": 0,
        "CG_iterations": 0,
        "projection_residual_Gf": float(torch.linalg.vector_norm(residual).item() / target_norm.item()),
        "ActuationR2": float((1.0 - torch.sum(residual.reshape(-1).square()) / r2_denom).item()),
        "B2_transfer_gain": float(b2_transfer),
        "random_matched_B2_transfer_gain": float(random_gain),
        "function_displacement_cos_with_target": function_displacement_cosine(actual, target_readback),
        "solve_time_ms": (time.perf_counter() - start) * 1000.0,
        "update_norm": float(torch.linalg.vector_norm(update_vec).item()),
        "function_displacement_norm": float(torch.linalg.vector_norm(actual).item()),
        "target_function_displacement_norm": float(torch.linalg.vector_norm(target_readback).item()),
        "condition_estimate": cond,
        "optimizer_state_write_fraction": float(optimizer_state_write_fraction if block_role != "optimizer_state_only" else 1.0),
        "fit_scope": scope,
        "loss_agnostic_contract_pass": int(y is None),
        "uses_labels_for_direction": int(y is not None),
        "uses_loss_for_direction": int(y is not None and block_role in {"hidden_only", "hidden_readout"}),
        "uses_future_or_validation": 0,
        "uses_audit_metric_for_direction": 0,
        **energies,
    }
    pass_flag = int(
        diagnostics["projection_residual_Gf"] <= 0.40
        and diagnostics["ActuationR2"] >= 0.20
        and diagnostics["B2_transfer_gain"] > diagnostics["random_matched_B2_transfer_gain"]
        and diagnostics["function_displacement_cos_with_target"] >= 0.30
    )
    blockers = []
    if diagnostics["projection_residual_Gf"] > 0.40:
        blockers.append("projection_residual_Gf_gate")
    if diagnostics["ActuationR2"] < 0.20:
        blockers.append("ActuationR2_gate")
    if diagnostics["B2_transfer_gain"] <= diagnostics["random_matched_B2_transfer_gain"]:
        blockers.append("B2_transfer_gain_vs_random_gate")
    if diagnostics["function_displacement_cos_with_target"] < 0.30:
        blockers.append("function_displacement_cos_gate")
    if y is not None:
        blockers.append("loss_agnostic_contract_gate")
    final_pass = int(pass_flag and y is None)
    diagnostics["S4_metric_dynamics_commit_pass"] = final_pass
    diagnostics["blocker"] = "" if final_pass else ";".join(blockers)
    return update_vec.detach(), diagnostics


def constructive_commit_unit_tests() -> list[dict[str, Any]]:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(6, 24)
            self.w2 = torch.nn.Parameter(torch.randn(24, 4) * 0.02)

        def frozen_readout_features(self, xb: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.fc1(xb))

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            return self.frozen_readout_features(xb) @ self.w2

    torch.manual_seed(2210)
    model = Tiny()
    x = torch.randn(32, 6)
    with torch.no_grad():
        logits = model(x).detach().float()
    target = logits - logits.mean(dim=-1, keepdim=True)
    update, diag = solve_external_target_commit(model, x, None, target * 0.05, block_role="readout_only", damping=1.0e-3)
    finite = all(isfinite(float(v)) for v in diag.values() if isinstance(v, (float, int)))
    return [
        {
            "case": "external_target_readout_commit",
            "update_nonzero": int(float(torch.linalg.vector_norm(update).item()) > 0.0),
            "projection_residual_Gf": diag.get("projection_residual_Gf", ""),
            "ActuationR2": diag.get("ActuationR2", ""),
            "diagnostics_finite": int(finite),
            "loss_agnostic_contract_pass": diag.get("loss_agnostic_contract_pass", 0),
            "pass": int(float(torch.linalg.vector_norm(update).item()) > 0.0 and finite and int(diag.get("loss_agnostic_contract_pass", 0))),
        }
    ]


__all__ = ["constructive_commit_unit_tests", "solve_external_target_commit"]
