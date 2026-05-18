"""Manual gradient smoke checks for full-edge primitives."""

from __future__ import annotations

from typing import Dict

import torch

from dgkan.models.edge_layers import EdgeFunctionLayer


def edge_layer_manual_shape_smoke(seed: int = 0) -> Dict[str, object]:
    torch.manual_seed(seed)
    layer = EdgeFunctionLayer(3, 2)
    x = torch.randn(5, 3)
    y, cache = layer.forward(x)
    dy = torch.randn_like(y)
    dx, grad = layer.backward(dy, cache)
    return {
        "stage": "B1_CORE_SMOKE_EDGE_LAYER",
        "status": "measured",
        "input_shape": str(tuple(x.shape)),
        "output_shape": str(tuple(y.shape)),
        "dx_shape": str(tuple(dx.shape)),
        "grad_shape": str(tuple(grad.shape)),
        "manual_backward_shape_pass": int(dx.shape == x.shape and grad.shape == layer.theta.shape),
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def edge_layer_manual_autograd_gradcheck(seed: int = 1, *, atol: float = 1.0e-5) -> Dict[str, object]:
    """Compare manual full-edge backward to an autograd reference.

    The reference is used only for verification. It calls
    ``torch.autograd.grad`` rather than ``loss.backward`` and is not an
    official training path.
    """

    torch.manual_seed(seed)
    layer = EdgeFunctionLayer(3, 2)
    x = torch.randn(7, 3)
    dy = torch.randn(7, 2)
    manual_y, cache = layer.forward(x)
    manual_dx, manual_grad = layer.backward(dy, cache)

    x_ref = x.detach().clone().requires_grad_(True)
    theta_ref = layer.theta.detach().clone().requires_grad_(True)
    edge_input = x_ref.unsqueeze(2).expand(-1, layer.in_features, layer.out_features)
    a0, a1, a2, scale, b1, b2 = theta_ref.unsqueeze(0).unbind(dim=-1)
    silu_x = edge_input * torch.sigmoid(edge_input)
    edge_values = a0 + a1 * edge_input + a2 * silu_x + scale * (b1 * edge_input + b2 * edge_input.square())
    ref_y = edge_values.sum(dim=1)
    ref_scalar = (ref_y * dy).sum()
    ref_dx, ref_grad = torch.autograd.grad(ref_scalar, (x_ref, theta_ref))

    output_max_abs_diff = float((manual_y - ref_y.detach()).abs().max().item())
    dx_max_abs_diff = float((manual_dx - ref_dx.detach()).abs().max().item())
    grad_max_abs_diff = float((manual_grad - ref_grad.detach()).abs().max().item())
    grad_pass = int(max(output_max_abs_diff, dx_max_abs_diff, grad_max_abs_diff) <= float(atol))
    return {
        "stage": "D1_FULL_EDGE_POLY2SILU_EDGE_LAYER_GRADCHECK",
        "status": "measured",
        "candidate_id": "DG-FullEdge-Poly2Silu-edge-layer-smoke",
        "model_level": "full_edge",
        "edge_basis": "poly2_silu",
        "autograd_reference_used_for_verification": 1,
        "uses_loss_backward": 0,
        "output_max_abs_diff": output_max_abs_diff,
        "dx_max_abs_diff": dx_max_abs_diff,
        "grad_max_abs_diff": grad_max_abs_diff,
        "GradPass": grad_pass,
        "GradRelErr_max": "not_computed_for_zero_safe_smoke",
        "GradCos_min": "not_computed_for_zero_safe_smoke",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
