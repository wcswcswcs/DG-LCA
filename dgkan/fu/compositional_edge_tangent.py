"""Matrix-free compositional edge tangent operators for DG-KAN v23.16.

The operators here are intentionally narrow: they act on existing PureKAN edge
coefficient tensors and expose the first-order chain-rule split

    xi_{l+1} = B_l delta A_l + D_l xi_l

without adding model parameters or changing the forward chart.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch

from dgkan.models.fc_purekan_primitives import _basis_derivative


EPS = 1.0e-12


@dataclass
class CompositionalTangentCache:
    activations: list[torch.Tensor]
    bases: list[torch.Tensor]
    basis_derivatives: list[torch.Tensor]
    edge_derivatives: list[torch.Tensor]
    logits: torch.Tensor
    horizontal_mode: str = "normal"
    horizontal_permutations: list[torch.Tensor] | None = None
    horizontal_signs: list[torch.Tensor] | None = None

    @property
    def depth(self) -> int:
        return len(self.bases)


def _scaled_basis_and_derivative(model: Any, h: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scaled basis values and d(scaled basis)/dh for TrueDeepPureKAN."""
    del layer_idx
    gain = float(getattr(model, "basis_input_gain", 1.0))
    z = torch.tanh(h * gain)
    basis = model.basis(h) / math.sqrt(max(1, int(h.shape[1])))
    raw_deriv = _basis_derivative(
        z,
        str(getattr(model, "basis_name")),
        int(getattr(model, "k")),
        model.centers.to(device=h.device, dtype=h.dtype),
        model.scales.to(device=h.device, dtype=h.dtype),
    )
    deriv = raw_deriv * (gain * (1.0 - z.square())).unsqueeze(-1)
    deriv = deriv / math.sqrt(max(1, int(h.shape[1])))
    return basis, deriv


def build_tangent_cache(
    model: Any,
    x: torch.Tensor,
    *,
    horizontal_mode: str = "normal",
    seed: int = 0,
) -> CompositionalTangentCache:
    h = x
    activations = [h]
    bases: list[torch.Tensor] = []
    basis_derivatives: list[torch.Tensor] = []
    edge_derivatives: list[torch.Tensor] = []
    perms: list[torch.Tensor] = []
    signs: list[torch.Tensor] = []
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    for layer_idx, coeff in enumerate(model.coeffs):
        basis, dbasis = _scaled_basis_and_derivative(model, h, int(layer_idx))
        coeff_work = coeff.to(device=h.device, dtype=h.dtype)
        edge_deriv = torch.einsum("bik,iok->bio", dbasis, coeff_work)
        if str(horizontal_mode) == "suppressed":
            edge_deriv = torch.zeros_like(edge_deriv)
        elif str(horizontal_mode) == "shuffled":
            perm = torch.randperm(int(edge_deriv.shape[0]), generator=gen, device=edge_deriv.device)
            perms.append(perm)
            edge_deriv = edge_deriv[perm]
        elif str(horizontal_mode) == "random_signs":
            sign = torch.randint(0, 2, edge_deriv.shape, generator=gen, device=edge_deriv.device, dtype=torch.int64)
            sign = sign.to(dtype=edge_deriv.dtype).mul_(2.0).sub_(1.0)
            signs.append(sign)
            edge_deriv = edge_deriv * sign
        bases.append(basis)
        basis_derivatives.append(dbasis)
        edge_derivatives.append(edge_deriv)
        h = torch.einsum("bik,iok->bo", basis, coeff_work)
        activations.append(h)
    return CompositionalTangentCache(
        activations=activations,
        bases=bases,
        basis_derivatives=basis_derivatives,
        edge_derivatives=edge_derivatives,
        logits=h,
        horizontal_mode=str(horizontal_mode),
        horizontal_permutations=perms or None,
        horizontal_signs=signs or None,
    )


def zeros_like_coeffs(model: Any, *, dtype: torch.dtype | None = None) -> list[torch.Tensor]:
    return [torch.zeros_like(c, dtype=dtype or c.dtype) for c in model.coeffs]


def flatten_coeffs(coeffs: Iterable[torch.Tensor]) -> torch.Tensor:
    parts = [c.detach().reshape(-1).to(dtype=torch.float64) for c in coeffs]
    if not parts:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(parts, dim=0)


def unflatten_coeffs(flat: torch.Tensor, templates: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    pos = 0
    work = flat.to(dtype=torch.float64)
    for tmpl in templates:
        n = int(tmpl.numel())
        out.append(work[pos : pos + n].reshape_as(tmpl).to(device=tmpl.device, dtype=torch.float64))
        pos += n
    if pos != int(work.numel()):
        raise ValueError(f"flat vector has {int(work.numel())} entries, consumed {pos}")
    return out


def coeff_layer_sizes(model: Any) -> list[int]:
    return [int(c.numel()) for c in model.coeffs]


def coeff_layer_slices(model: Any) -> list[slice]:
    out: list[slice] = []
    pos = 0
    for size in coeff_layer_sizes(model):
        out.append(slice(pos, pos + size))
        pos += size
    return out


def apply_j(
    model: Any,
    cache: CompositionalTangentCache,
    delta_coeffs: list[torch.Tensor],
    *,
    return_parts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    xi = torch.zeros_like(cache.activations[0], dtype=torch.float64)
    verticals: list[torch.Tensor] = []
    horizontals: list[torch.Tensor] = []
    for layer_idx, (basis, edge_deriv, delta) in enumerate(zip(cache.bases, cache.edge_derivatives, delta_coeffs)):
        del layer_idx
        b = basis.to(dtype=torch.float64)
        d = delta.to(device=b.device, dtype=torch.float64)
        vertical = torch.einsum("bik,iok->bo", b, d)
        horizontal = torch.einsum("bio,bi->bo", edge_deriv.to(dtype=torch.float64), xi)
        xi = vertical + horizontal
        verticals.append(vertical)
        horizontals.append(horizontal)
    if return_parts:
        return xi, verticals, horizontals
    return xi


def apply_jt(model: Any, cache: CompositionalTangentCache, output_cotangent: torch.Tensor) -> list[torch.Tensor]:
    del model
    q = output_cotangent.to(dtype=torch.float64)
    grads: list[torch.Tensor] = [torch.empty(0)] * cache.depth
    for layer_idx in reversed(range(cache.depth)):
        basis = cache.bases[layer_idx].to(device=q.device, dtype=torch.float64)
        edge_deriv = cache.edge_derivatives[layer_idx].to(device=q.device, dtype=torch.float64)
        grads[layer_idx] = torch.einsum("bik,bo->iok", basis, q)
        q = torch.einsum("bio,bo->bi", edge_deriv, q)
    return grads


def coeff_inner(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    vals = [
        (aa.to(dtype=torch.float64) * bb.to(device=aa.device, dtype=torch.float64)).sum()
        for aa, bb in zip(a, b)
    ]
    if not vals:
        return torch.tensor(0.0, dtype=torch.float64)
    return torch.stack(vals).sum()


def output_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.to(dtype=torch.float64) * b.to(device=a.device, dtype=torch.float64)).sum()


def explicit_jacobian(model: Any, cache: CompositionalTangentCache) -> torch.Tensor:
    """Build a dense output-by-coeff Jacobian for small audits."""
    templates = [c.detach().to(dtype=torch.float64) for c in model.coeffs]
    total = sum(int(c.numel()) for c in templates)
    out_dim = int(cache.logits.numel())
    jac = torch.empty((out_dim, total), device=cache.logits.device, dtype=torch.float64)
    pos = 0
    for layer_idx, tmpl in enumerate(templates):
        flat_count = int(tmpl.numel())
        for local in range(flat_count):
            deltas = [torch.zeros_like(t) for t in templates]
            deltas[layer_idx].reshape(-1)[local] = 1.0
            jac[:, pos + local] = apply_j(model, cache, deltas).reshape(-1)
        pos += flat_count
    return jac


def autograd_jvp(model: Any, x: torch.Tensor, delta_coeffs: list[torch.Tensor]) -> torch.Tensor:
    """Reference JVP using torch.func; finite difference fallback if unavailable."""
    try:
        from torch.func import functional_call, jvp

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        coeff_names = [f"coeffs.{i}" for i in range(len(delta_coeffs))]
        base = tuple(params[name].detach().clone().requires_grad_(True) for name in coeff_names)
        tangents = tuple(d.to(device=base[i].device, dtype=base[i].dtype) for i, d in enumerate(delta_coeffs))

        def fn(*coeff_tuple: torch.Tensor) -> torch.Tensor:
            local_params = dict(params)
            for name, coeff in zip(coeff_names, coeff_tuple):
                local_params[name] = coeff
            return functional_call(model, (local_params, buffers), (x,))

        _y, tangent = jvp(fn, base, tangents)
        return tangent.detach().to(dtype=torch.float64)
    except Exception:
        eps = 1.0e-6 if x.dtype == torch.float64 else 1.0e-3
        before = [p.detach().clone() for p in model.coeffs]
        with torch.no_grad():
            y0 = model(x).detach().to(dtype=torch.float64)
            for param, delta in zip(model.coeffs, delta_coeffs):
                param.add_(float(eps) * delta.to(device=param.device, dtype=param.dtype))
            y1 = model(x).detach().to(dtype=torch.float64)
            for param, value in zip(model.coeffs, before):
                param.copy_(value)
        return (y1 - y0) / float(eps)


def derivative_health(cache: CompositionalTangentCache) -> dict[str, float]:
    out: dict[str, float] = {}
    for layer_idx, (act, edge_deriv) in enumerate(zip(cache.activations[:-1], cache.edge_derivatives)):
        a = act.detach().to(dtype=torch.float64)
        d = edge_deriv.detach().to(dtype=torch.float64)
        finite = torch.isfinite(d)
        out[f"activation_min_l{layer_idx}"] = float(a.min().detach().cpu().item())
        out[f"activation_max_l{layer_idx}"] = float(a.max().detach().cpu().item())
        out[f"activation_out_of_nominal_domain_fraction_l{layer_idx}"] = float((a.abs() > 1.0).to(dtype=torch.float64).mean().detach().cpu().item())
        out[f"edge_derivative_rms_l{layer_idx}"] = float(d.square().mean().sqrt().detach().cpu().item())
        out[f"edge_derivative_p95_l{layer_idx}"] = float(torch.quantile(d.abs().reshape(-1), 0.95).detach().cpu().item())
        out[f"edge_derivative_max_l{layer_idx}"] = float(d.abs().max().detach().cpu().item())
        out[f"horizontal_operator_norm_estimate_l{layer_idx}"] = float(d.square().sum(dim=2).sqrt().amax().detach().cpu().item())
        out[f"nan_inf_count_l{layer_idx}"] = float((~finite).sum().detach().cpu().item())
    return out
