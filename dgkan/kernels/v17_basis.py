"""Basis kernel reference and v17 torch-vectorized paths."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.models.fc_purekan_primitives import _basis_eval


FAMILY_TO_BASIS = {
    "D-CHE": ("chebyshev", 4, 1),
    "D-FOU": ("fourier_lowfreq", 4, 1),
    "D-RBF": ("compact_rbf", 4, 0),
    "D-WAV": ("hat_wavelet", 4, 0),
    "D-RAT": ("rational_kat_lite", 4, 1),
    "LQ": ("legendre", 4, 1),
}


def basis_spec(family: str) -> tuple[str, int, int]:
    if family not in FAMILY_TO_BASIS:
        raise ValueError(f"unknown kernel family {family}")
    return FAMILY_TO_BASIS[family]


def reference_eval(z: torch.Tensor, family: str) -> torch.Tensor:
    name, k, _dense = basis_spec(family)
    centers = torch.linspace(-1.0, 1.0, k, device=z.device, dtype=z.dtype)
    scales = torch.tensor([max(0.2, 2.0 / max(1, k - 1))], device=z.device, dtype=z.dtype)
    return _basis_eval(z, name, k, centers, scales)


def _chebyshev_recurrence(z: torch.Tensor, k: int) -> torch.Tensor:
    vals = [torch.ones_like(z)]
    if k > 1:
        vals.append(z)
    for _idx in range(2, k):
        vals.append(2.0 * z * vals[-1] - vals[-2])
    return torch.stack(vals[:k], dim=-1)


def _fourier_recurrence(z: torch.Tensor, k: int) -> torch.Tensor:
    vals = [z]
    freq = 1
    while len(vals) < k:
        angle = torch.pi * float(freq) * z
        vals.append(torch.sin(angle))
        if len(vals) < k:
            vals.append(torch.cos(angle))
        freq += 1
    return torch.stack(vals[:k], dim=-1) / torch.sqrt(torch.tensor(float(max(1, k)), device=z.device, dtype=z.dtype))


def _rbf_local_k(z: torch.Tensor, k: int) -> torch.Tensor:
    centers = torch.linspace(-1.0, 1.0, k, device=z.device, dtype=z.dtype)
    width = torch.tensor([max(0.2, 2.0 / max(1, k - 1))], device=z.device, dtype=z.dtype)[0].clamp_min(1.0e-3)
    r = (z.unsqueeze(-1) - centers) / width
    return torch.exp(-0.5 * r.square())


def _wavelet_support(z: torch.Tensor, k: int) -> torch.Tensor:
    centers = torch.linspace(-1.0, 1.0, k, device=z.device, dtype=z.dtype)
    width = torch.tensor([max(0.2, 2.0 / max(1, k - 1))], device=z.device, dtype=z.dtype)[0].clamp_min(1.0e-3)
    r = (z.unsqueeze(-1) - centers).abs() / width
    inner = torch.relu(1.0 - r)
    outer = torch.relu(1.0 - 0.5 * r)
    return inner - 0.5 * outer


def _rational_branchless(z: torch.Tensor, k: int) -> torch.Tensor:
    denom = 1.0 + 0.5 * z.abs() + 0.125 * z.square()
    vals = []
    cur = z
    for idx in range(k):
        if idx > 0:
            cur = cur * z
        vals.append(cur / (denom + 0.05 * idx))
    return torch.stack(vals[:k], dim=-1)


def _legendre_recurrence(z: torch.Tensor, k: int) -> torch.Tensor:
    vals = [torch.ones_like(z)]
    if k > 1:
        vals.append(z)
    for n in range(2, k):
        vals.append(((2 * n - 1) * z * vals[-1] - (n - 1) * vals[-2]) / n)
    return torch.stack(vals[:k], dim=-1)


def repaired_eval(z: torch.Tensor, family: str) -> torch.Tensor:
    _name, k, _dense = basis_spec(family)
    if family == "D-CHE":
        return _chebyshev_recurrence(z, k)
    if family == "D-FOU":
        return _fourier_recurrence(z, k)
    if family == "D-RBF":
        return _rbf_local_k(z, k)
    if family == "D-WAV":
        return _wavelet_support(z, k)
    if family == "D-RAT":
        return _rational_branchless(z, k)
    if family == "LQ":
        return _legendre_recurrence(z, k)
    return reference_eval(z, family)


def kernel_correctness_row(family: str, *, seed: int = 1700, device: torch.device | None = None) -> dict[str, Any]:
    dev = device or torch.device("cpu")
    gen = torch.Generator(device=dev).manual_seed(int(seed) + sum(ord(c) for c in family))
    z_ref = torch.randn(32, 8, generator=gen, device=dev, dtype=torch.float64, requires_grad=True)
    z_rep = z_ref.detach().clone().requires_grad_(True)
    ref = reference_eval(z_ref, family)
    rep = repaired_eval(z_rep, family)
    f_relerr = float(torch.linalg.vector_norm((rep - ref).detach()) / torch.linalg.vector_norm(ref.detach()).clamp_min(1.0e-12))
    grad_seed = torch.randn(ref.shape, generator=gen, device=dev, dtype=torch.float64)
    (ref * grad_seed).sum().backward()
    (rep * grad_seed).sum().backward()
    g_ref = z_ref.grad.detach()
    g_rep = z_rep.grad.detach()
    g_relerr = float(torch.linalg.vector_norm(g_rep - g_ref) / torch.linalg.vector_norm(g_ref).clamp_min(1.0e-12))
    denom = torch.linalg.vector_norm(g_rep) * torch.linalg.vector_norm(g_ref)
    grad_cosine = float((g_rep.reshape(-1) @ g_ref.reshape(-1) / denom.clamp_min(1.0e-12)).clamp(-1.0, 1.0))
    _name, _k, dense = basis_spec(family)
    return {
        "family": family,
        "kernel_forward_relerr": f_relerr,
        "kernel_grad_relerr": g_relerr,
        "kernel_grad_cosine": grad_cosine,
        "dense_basis_materialized": int(dense),
        "kernel_correctness_exploration_pass": int(f_relerr <= 1.0e-4 or grad_cosine >= 0.999),
        "kernel_correctness_official_pass": int(f_relerr <= 1.0e-5 and grad_cosine >= 0.9999),
        "no_nan_inf": int(bool(torch.isfinite(rep).all() and torch.isfinite(g_rep).all())),
        "workspace_no_dense_materialization_audit": int(not dense),
        "implementation_path": "dgkan.kernels.v17_basis.family_specific_torch_repair",
        "repair_scope": "torch_family_specific_not_fused_cuda",
        "official_fused_kernel_complete": 0,
    }
