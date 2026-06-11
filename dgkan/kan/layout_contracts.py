"""Corrected KAN readout layout truth tests for v22.13."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.core import flat_params, load_flat_params
from dgkan.kan.corrected_readout_solve import solve_corrected_readout_update
from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec


def make_kan(carrier: str, x: torch.Tensor, seed: int = 2213) -> PrimitiveKAN:
    if carrier == "D-CHE":
        spec = PrimitiveSpec(
            candidate_id="v22.13-D-CHE-layout",
            basis_family="D-CHE",
            basis_name="chebyshev",
            k=3,
            hidden_dim=64,
            source="v22_13_layout_contract",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=0,
            uses_division=0,
            uses_dense_basis_tensor=1,
        )
    else:
        spec = PrimitiveSpec(
            candidate_id="v22.13-D-FOU-layout",
            basis_family="D-FOU",
            basis_name="fourier_lowfreq",
            k=3,
            hidden_dim=64,
            source="v22_13_layout_contract",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=1,
            uses_division=0,
            uses_dense_basis_tensor=1,
        )
    return PrimitiveKAN(8, 5, spec, x, int(seed), torch.device("cpu"), param_budget=4096)


def _fit_from_update(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor, update: torch.Tensor) -> tuple[float, float]:
    before = flat_params(model).detach().float()
    with torch.no_grad():
        base = model(x).detach().float()
        load_flat_params(model, before + update.to(dtype=before.dtype))
        moved = model(x).detach().float()
        load_flat_params(model, before)
    actual = (moved - base).reshape(-1)
    tgt = target.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(actual).clamp_min(1.0e-8) * torch.linalg.vector_norm(tgt).clamp_min(1.0e-8)
    fit_cos = float((actual @ tgt / denom).clamp(-1.0, 1.0).item())
    residual = float((torch.linalg.vector_norm(actual - tgt) / torch.linalg.vector_norm(tgt).clamp_min(1.0e-8)).item())
    return fit_cos, residual


def _legacy_update(model: PrimitiveKAN, x: torch.Tensor, target: torch.Tensor, damping: float = 1.0e-3) -> torch.Tensor:
    before = flat_params(model).detach().float()
    with torch.no_grad():
        h = model.hidden(x).detach().float()
        b2 = model.layer2_basis(h).detach().float()
        design = b2.reshape(int(x.shape[0]), -1) / (float(max(1, int(model.hidden_dim))) ** 0.5)
    gram = design.T @ design + float(damping) * torch.eye(int(design.shape[1]), dtype=design.dtype)
    rhs = design.T @ target.detach().float()
    try:
        coeff = torch.linalg.solve(gram, rhs)
    except Exception:
        coeff = torch.linalg.lstsq(gram, rhs).solution
    hidden, classes, basis = int(model.w2.shape[0]), int(model.w2.shape[1]), int(model.w2.shape[2])
    wrong = coeff.reshape(hidden, classes, basis).contiguous()
    update = torch.zeros_like(before)
    offset = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        if name == "w2":
            update[offset : offset + n] = wrong.reshape(-1).to(dtype=update.dtype)
        offset += n
    return update


def _layout_row(carrier: str, seed: int) -> dict[str, Any]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    x = torch.randn(64, 8, generator=gen)
    model = make_kan(carrier, x, seed)
    with torch.no_grad():
        h = model.hidden(x).detach().float()
        b2 = model.layer2_basis(h).detach().float()
        design = b2.reshape(int(x.shape[0]), -1) / (float(max(1, int(model.hidden_dim))) ** 0.5)
        coeff = torch.randn(int(design.shape[1]), 5, generator=gen) * 0.02
        target = design @ coeff
    legacy = _legacy_update(model, x, target)
    legacy_cos, legacy_resid = _fit_from_update(model, x, target, legacy)
    corrected, diag = solve_corrected_readout_update(model, x, target)
    corrected_cos, corrected_resid = _fit_from_update(model, x, target, corrected)
    gap = int((corrected_cos - legacy_cos) >= 0.05 or (legacy_resid - corrected_resid) >= 0.05)
    return {
        "carrier": carrier,
        "w2_layout_contract": diag.get("w2_layout_contract", ""),
        "frozen_readout_feature_layout": diag.get("frozen_readout_feature_layout", ""),
        "legacy_layout_fit_cosine": legacy_cos,
        "legacy_layout_projection_residual": legacy_resid,
        "corrected_layout_fit_cosine": corrected_cos,
        "corrected_layout_projection_residual": corrected_resid,
        "legacy_vs_corrected_gap_detected": gap,
        "layout_unit_test_pass": int(corrected_cos >= 0.99 and corrected_resid <= 0.10 and gap),
    }


def layout_truth_rows(seed: int = 2213) -> list[dict[str, Any]]:
    return [_layout_row("D-CHE", seed), _layout_row("D-FOU", seed + 1)]


__all__ = ["layout_truth_rows", "make_kan"]

