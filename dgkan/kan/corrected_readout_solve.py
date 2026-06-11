"""Corrected frozen-readout solve for FC-PureKAN basis/readout layout."""

from __future__ import annotations

import math
from typing import Any

import torch

from dgkan.fu.core import flat_params, load_flat_params


def _channel_energy(model: torch.nn.Module, update: torch.Tensor) -> tuple[float, float]:
    basis = 0.0
    readout = 0.0
    offset = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        value = float(torch.linalg.vector_norm(update[offset : offset + n].detach().float()).item())
        low = name.lower()
        if "w1" in low:
            basis += value
        if "w2" in low or "readout" in low:
            readout += value
        offset += n
    total = basis + readout
    return (basis / total, readout / total) if total > 1.0e-12 else (0.0, 0.0)


def solve_corrected_readout_update(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor, damping: float = 1.0e-3) -> tuple[torch.Tensor, dict[str, Any]]:
    base_params = flat_params(model).detach().float()
    with torch.no_grad():
        base_logits = model(x).detach().float()
        h = model.hidden(x).detach().float() if hasattr(model, "hidden") else model.frozen_readout_features(x).detach().float()
        b2 = model.layer2_basis(h).detach().float() if hasattr(model, "layer2_basis") else model.frozen_readout_features(x).detach().float()
        if hasattr(model, "w2") and getattr(model, "w2").ndim == 3:
            design = b2.reshape(int(x.shape[0]), -1) / math.sqrt(max(1, int(getattr(model, "hidden_dim", b2.shape[1]))))
            shape = model.w2.shape
            param_name = "w2"
        else:
            design = model.frozen_readout_features(x).detach().float()
            shape = next(p.shape for n, p in model.named_parameters() if "w2" in n.lower() or "readout" in n.lower())
            param_name = next(n for n, _p in model.named_parameters() if "w2" in n.lower() or "readout" in n.lower())
    gram = design.T @ design + float(damping) * torch.eye(int(design.shape[1]), dtype=design.dtype, device=design.device)
    rhs = design.T @ target.detach().float().to(device=design.device)
    try:
        coeff = torch.linalg.solve(gram, rhs)
    except Exception:
        coeff = torch.linalg.lstsq(gram, rhs).solution
    if len(shape) == 3:
        hidden, classes, basis = int(shape[0]), int(shape[1]), int(shape[2])
        delta_param = coeff[: hidden * basis].reshape(hidden, basis, classes).permute(0, 2, 1).contiguous()
    else:
        delta_param = coeff.reshape(shape).contiguous()
    update = torch.zeros_like(base_params)
    offset = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        if name == param_name:
            update[offset : offset + n] = delta_param.reshape(-1).to(dtype=update.dtype)
        offset += n
    with torch.no_grad():
        load_flat_params(model, base_params + update.to(dtype=base_params.dtype))
        actual = (model(x).detach().float() - base_logits).reshape(-1)
        load_flat_params(model, base_params)
    target_flat = target.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(actual).clamp_min(1.0e-8) * torch.linalg.vector_norm(target_flat).clamp_min(1.0e-8)
    fit_cos = float((actual @ target_flat / denom).clamp(-1.0, 1.0).item())
    residual = float((torch.linalg.vector_norm(actual - target_flat) / torch.linalg.vector_norm(target_flat).clamp_min(1.0e-8)).item())
    basis_energy, readout_energy = _channel_energy(model, update)
    return update.detach().float(), {
        "w2_layout_contract": "w2(hidden,class,basis)" if len(shape) == 3 else "readout_2d",
        "frozen_readout_feature_layout": "features(hidden,basis)_flattened",
        "corrected_layout_fit_cosine": fit_cos,
        "corrected_layout_projection_residual": residual,
        "basis_channel_energy": basis_energy,
        "readout_channel_energy": readout_energy,
        "layout_unit_test_pass": int(fit_cos >= 0.99 and residual <= 0.10),
    }


__all__ = ["solve_corrected_readout_update"]

