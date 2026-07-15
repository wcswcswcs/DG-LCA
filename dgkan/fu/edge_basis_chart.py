"""Basis chart utilities for DG-KAN v23.15.

The helpers in this module do not change model structure.  They build audit
representations of the same edge functions under invertible basis transforms.
For a basis row vector ``psi`` and coefficient vector ``a`` we use

    psi' = psi @ S,    a' = S^{-1} a,

so ``psi' @ a' == psi @ a`` up to numerical error.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


EPS = 1.0e-12


@dataclass(frozen=True)
class Chart:
    name: str
    transform_type: str
    S: torch.Tensor

    @property
    def condition_number(self) -> float:
        vals = torch.linalg.svdvals(self.S.detach().to(dtype=torch.float64))
        return float((vals.max().clamp_min(EPS) / vals.min().clamp_min(EPS)).cpu().item())


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.T)


def orient_columns(q: torch.Tensor) -> torch.Tensor:
    out = q.detach().clone()
    idx = out.abs().argmax(dim=0)
    cols = torch.arange(int(out.shape[1]), device=out.device)
    signs = torch.sign(out[idx, cols])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return out * signs.reshape(1, -1)


def source_basis_stats(phi_layer: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat = phi_layer.detach().to(dtype=torch.float64).reshape(-1, int(k))
    rms = flat.square().mean(dim=0).sqrt().clamp_min(EPS)
    gram = sym(flat.T @ flat / float(max(1, int(flat.shape[0]))))
    return rms, gram


def make_chart(
    name: str,
    k: int,
    *,
    seed: int = 0,
    source_phi: torch.Tensor | None = None,
    dyadic_exp_cap: int = 3,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Chart:
    kk = int(k)
    dev = device if device is not None else (source_phi.device if source_phi is not None else torch.device("cpu"))
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    key = str(name)
    if key in {"A0_identity", "B0_exact_identity_chart", "identity"}:
        S = torch.eye(kk, device=dev, dtype=torch.float64)
        return Chart(key, "identity", S.to(dtype=dtype))
    if source_phi is not None:
        rms, gram = source_basis_stats(source_phi, kk)
        rms = rms.to(device=dev)
        gram = gram.to(device=dev)
    else:
        rms = torch.ones(kk, device=dev, dtype=torch.float64)
        gram = torch.eye(kk, device=dev, dtype=torch.float64)
    if "source_rms" in key or "dyadic" in key and "random" not in key:
        target = torch.exp(torch.log(rms).mean())
        exp = torch.log2(target / rms).round().clamp(-int(dyadic_exp_cap), int(dyadic_exp_cap))
        S = torch.diag(torch.pow(torch.tensor(2.0, device=dev, dtype=torch.float64), exp))
        return Chart(key, "source_rms_dyadic", S.to(dtype=dtype))
    if "random_dyadic" in key:
        exp = torch.randint(-int(dyadic_exp_cap), int(dyadic_exp_cap) + 1, (kk,), generator=gen, device=dev).to(dtype=torch.float64)
        S = torch.diag(torch.pow(torch.tensor(2.0, device=dev, dtype=torch.float64), exp))
        return Chart(key, "random_dyadic", S.to(dtype=dtype))
    if "random_positive_diagonal" in key or "random_diagonal" in key:
        vals = torch.exp(torch.empty(kk, device=dev, dtype=torch.float64).uniform_(-math.log(3.0), math.log(3.0), generator=gen))
        vals = vals / vals.median().clamp_min(EPS)
        S = torch.diag(vals)
        return Chart(key, "random_positive_diagonal", S.to(dtype=dtype))
    if "source_gram_orthogonal" in key or "source_orthogonal" in key:
        vals, vecs = torch.linalg.eigh(gram)
        order = torch.argsort(vals, descending=True)
        S = orient_columns(vecs[:, order])
        return Chart(key, "source_gram_orthogonal", S.to(dtype=dtype))
    if "random_orthogonal" in key:
        raw = torch.randn((kk, kk), device=dev, dtype=torch.float64, generator=gen)
        q, _ = torch.linalg.qr(raw)
        S = orient_columns(q)
        return Chart(key, "random_orthogonal", S.to(dtype=dtype))
    if "dense_nonorthogonal" in key or "well_conditioned" in key:
        q1, _ = torch.linalg.qr(torch.randn((kk, kk), device=dev, dtype=torch.float64, generator=gen))
        q2, _ = torch.linalg.qr(torch.randn((kk, kk), device=dev, dtype=torch.float64, generator=gen))
        vals = torch.linspace(1.0, 10.0, kk, device=dev, dtype=torch.float64)
        S = orient_columns(q1) @ torch.diag(vals) @ orient_columns(q2).T
        return Chart(key, "well_conditioned_dense_nonorthogonal", S.to(dtype=dtype))
    if "ill_conditioned" in key:
        vals = torch.logspace(0.0, 5.0, kk, device=dev, dtype=torch.float64)
        S = torch.diag(vals)
        return Chart(key, "ill_conditioned_diagnostic", S.to(dtype=dtype))
    raise ValueError(f"unknown chart {name}")


def block_transform(S: torch.Tensor, in_dim: int) -> torch.Tensor:
    work = S.contiguous()
    eye = torch.eye(int(in_dim), device=work.device, dtype=work.dtype)
    return torch.kron(eye, work)


def transform_phi(phi: torch.Tensor, S: torch.Tensor, in_dim: int) -> torch.Tensor:
    T = block_transform(S.to(device=phi.device, dtype=phi.dtype), int(in_dim))
    return phi @ T


def expanded_metric(G: torch.Tensor, in_dim: int) -> torch.Tensor:
    eye = torch.eye(int(in_dim), device=G.device, dtype=G.dtype)
    return torch.kron(eye, G)


def transform_metric(G_expanded: torch.Tensor, S: torch.Tensor, in_dim: int) -> torch.Tensor:
    T = block_transform(S.to(device=G_expanded.device, dtype=G_expanded.dtype), int(in_dim))
    return sym(T.T @ G_expanded @ T)


def coeff_to_chart(coeff: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    work = coeff.detach().to(dtype=torch.float64)
    flat = work.reshape(-1, int(work.shape[-1]))
    Sinv_flat = torch.linalg.solve(S.to(device=work.device, dtype=torch.float64), flat.T).T
    return Sinv_flat.reshape_as(work).to(device=coeff.device, dtype=coeff.dtype)


def coeff_from_chart(coeff_chart: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    work = coeff_chart.detach().to(dtype=torch.float64)
    flat = work.reshape(-1, int(work.shape[-1]))
    out = flat @ S.to(device=work.device, dtype=torch.float64).T
    return out.reshape_as(work).to(device=coeff_chart.device, dtype=coeff_chart.dtype)


def flat_delta_to_chart(delta: torch.Tensor, S: torch.Tensor, in_dim: int) -> torch.Tensor:
    T = block_transform(S.to(device=delta.device, dtype=torch.float64), int(in_dim))
    return torch.linalg.solve(T, delta.to(dtype=torch.float64))


def flat_delta_from_chart(delta_chart: torch.Tensor, S: torch.Tensor, in_dim: int) -> torch.Tensor:
    T = block_transform(S.to(device=delta_chart.device, dtype=torch.float64), int(in_dim))
    return T @ delta_chart.to(dtype=torch.float64)


def charted_forward_with_activations(model: Any, x: torch.Tensor, charts: list[Chart]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    h = x
    activations = [h]
    for layer_idx, coeff in enumerate(model.coeffs):
        chart = charts[int(layer_idx)]
        S = chart.S.to(device=h.device, dtype=h.dtype)
        basis = model.basis(h) / math.sqrt(max(1, int(model.dims[int(layer_idx)])))
        basis_chart = torch.einsum("bik,kl->bil", basis, S)
        coeff_chart = coeff_to_chart(coeff, chart.S).to(device=h.device, dtype=h.dtype)
        h = torch.einsum("bik,iok->bo", basis_chart, coeff_chart)
        activations.append(h)
    return h, activations


def max_relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().to(dtype=torch.float64)
    bb = b.detach().to(device=aa.device, dtype=torch.float64)
    return float((aa - bb).norm().div(bb.norm().clamp_min(EPS)).detach().cpu().item())


def coefficient_roundtrip_error(coeff: torch.Tensor, S: torch.Tensor) -> float:
    c_chart = coeff_to_chart(coeff, S)
    c_back = coeff_from_chart(c_chart, S)
    return max_relative_error(c_back, coeff.detach())
