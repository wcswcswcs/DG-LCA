"""Small JVP/VJP-style sketches for v22.06 metric-solver audits.

These helpers are intentionally local and train-stream only.  They provide a
finite-difference function-displacement readback for update vectors and a tiny
unit-test surface used by the S0.13 gate.  They do not create directions from
validation/test/future information.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import torch

from dgkan.fu.core import flat_params, load_flat_params


EPS = 1.0e-8


def finite_difference_jvp(
    model: torch.nn.Module,
    x: torch.Tensor,
    update: torch.Tensor,
    *,
    eps: float = 1.0e-3,
) -> torch.Tensor:
    """Return ``(f(theta + eps * update) - f(theta)) / eps``.

    The model parameters are restored before returning.  ``update`` is assumed
    to use the same flat trainable-parameter order as ``dgkan.fu.core``.
    """

    before = flat_params(model).detach().to(device=x.device)
    direction = update.detach().to(device=x.device, dtype=before.dtype)
    if direction.numel() != before.numel() or direction.numel() == 0:
        return torch.zeros((int(x.shape[0]), 0), device=x.device)
    with torch.no_grad():
        base = model(x).detach().float()
        load_flat_params(model, before + float(eps) * direction)
        moved = model(x).detach().float()
        load_flat_params(model, before)
    return (moved - base) / float(eps)


def function_displacement_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= EPS:
        return 0.0
    return float((x @ y / denom).clamp(-1.0, 1.0).item())


def jacobian_sketch_diagnostics(
    model: torch.nn.Module,
    x: torch.Tensor,
    update: torch.Tensor,
    target_delta: torch.Tensor | None = None,
) -> dict[str, Any]:
    jvp = finite_difference_jvp(model, x, update)
    out: dict[str, Any] = {
        "JVP_count": 1,
        "VJP_count": 0,
        "function_displacement_norm": float(torch.linalg.vector_norm(jvp.detach().float()).item()) if jvp.numel() else 0.0,
    }
    if target_delta is not None and target_delta.numel() and jvp.numel():
        target = target_delta.detach().float().reshape_as(jvp)
        residual = target - jvp.detach().float()
        denom = torch.linalg.vector_norm(target).clamp_min(EPS)
        centered = target.reshape(-1) - target.reshape(-1).mean()
        r2_denom = torch.sum(centered.square()).clamp_min(EPS)
        out.update(
            {
                "projection_residual_Gf": float(torch.linalg.vector_norm(residual).item() / denom.item()),
                "ActuationR2": float((1.0 - torch.sum(residual.reshape(-1).square()) / r2_denom).item()),
                "ActuationCosine": function_displacement_cosine(jvp, target),
            }
        )
    return out


def jacobian_sketch_unit_tests() -> list[dict[str, Any]]:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(3, 2)

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            return self.fc(xb)

    torch.manual_seed(2206)
    model = Tiny()
    x = torch.randn(4, 3)
    update = torch.zeros_like(flat_params(model))
    update[-2:] = 1.0
    jvp = finite_difference_jvp(model, x, update)
    diag = jacobian_sketch_diagnostics(model, x, update, jvp)
    finite = all(isfinite(float(v)) for v in diag.values() if isinstance(v, (float, int)))
    return [
        {
            "case": "finite_difference_jvp_bias_direction",
            "jvp_shape": "x".join(str(i) for i in jvp.shape),
            "jvp_nonzero": int(float(torch.linalg.vector_norm(jvp).item()) > 0.0),
            "diagnostics_finite": int(finite),
            "pass": int(jvp.shape == (4, 2) and float(torch.linalg.vector_norm(jvp).item()) > 0.0 and finite),
        }
    ]


__all__ = [
    "finite_difference_jvp",
    "function_displacement_cosine",
    "jacobian_sketch_diagnostics",
    "jacobian_sketch_unit_tests",
]
