"""Edge-Sobolev geometry helpers for DG-KAN v23.00R.

These utilities operate on existing D-CHE / D-FOUR coefficient tensors only.
They intentionally do not create new edge basis functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from dgkan.models.fc_purekan_primitives import _basis_derivative, _basis_eval


EPS = 1.0e-12


@dataclass(frozen=True)
class ModeWeightSummary:
    mode_weight_min: float
    mode_weight_max: float
    mode_weight_median: float
    mode_weight_condition: float
    basis_key: str
    exponent: float
    normalization: str


def _safe_float(value: torch.Tensor | float) -> float:
    out = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
    return out if math.isfinite(out) else 0.0


def fourier_omega_vector(k: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return D-FOUR frequencies with sin/cos pairs sharing a frequency."""

    vals: list[float] = [0.0]
    freq = 1
    while len(vals) < int(k):
        vals.append(float(freq))
        if len(vals) < int(k):
            vals.append(float(freq))
        freq += 1
    return torch.tensor(vals[: int(k)], device=device, dtype=dtype)


def make_mode_weights(
    basis_key: str,
    k: int,
    *,
    exponent: float = 1.0,
    normalization: str = "median",
    ridge: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Create normalized mode weights for D-CHE or D-FOUR.

    ``basis_key`` may be a plan key (``dche_k5`` / ``dfour_default``) or the
    local model basis name (``chebyshev`` / ``fourier_lowfreq``).
    """

    key = str(basis_key).lower()
    kk = int(k)
    if kk <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if "four" in key or "fou" in key:
        omega = fourier_omega_vector(kk, device=device, dtype=dtype)
        weights = (1.0 + omega.square()).pow(float(exponent))
    else:
        degree = torch.arange(kk, device=device, dtype=dtype)
        weights = (1.0 + degree.square()).pow(float(exponent))
    if float(ridge) > 0.0:
        weights = weights + float(ridge)
    if str(normalization) == "mean":
        scale = weights.mean().clamp_min(EPS)
        weights = weights / scale
    elif str(normalization) == "trace":
        scale = weights.sum().div(float(max(1, kk))).clamp_min(EPS)
        weights = weights / scale
    else:
        scale = weights.median().clamp_min(EPS)
        weights = weights / scale
    return weights.clamp_min(EPS).to(dtype=dtype)


def mode_weight_summary(weights: torch.Tensor, basis_key: str, exponent: float, normalization: str) -> ModeWeightSummary:
    ww = weights.detach().to(dtype=torch.float64).reshape(-1)
    if int(ww.numel()) == 0:
        return ModeWeightSummary(0.0, 0.0, 0.0, 0.0, str(basis_key), float(exponent), str(normalization))
    mn = ww.min().clamp_min(EPS)
    mx = ww.max().clamp_min(EPS)
    med = ww.median().clamp_min(EPS)
    return ModeWeightSummary(
        mode_weight_min=_safe_float(mn),
        mode_weight_max=_safe_float(mx),
        mode_weight_median=_safe_float(med),
        mode_weight_condition=_safe_float(mx / mn),
        basis_key=str(basis_key),
        exponent=float(exponent),
        normalization=str(normalization),
    )


def flatten_param_mode_weights(param: torch.Tensor, mode_weights: torch.Tensor, *, mode_axis: int = -1) -> torch.Tensor:
    """Expand mode weights to match ``param.reshape(-1)`` order."""

    pp = param.detach()
    axis = int(mode_axis)
    if axis < 0:
        axis += pp.ndim
    shape = list(pp.shape)
    if not shape:
        return torch.ones(1, device=pp.device, dtype=torch.float64)
    k = int(shape[axis])
    ww = mode_weights.to(device=pp.device, dtype=torch.float64).reshape(-1)
    if int(ww.numel()) != k:
        raise ValueError(f"mode weight length {int(ww.numel())} does not match axis length {k}")
    view_shape = [1] * pp.ndim
    view_shape[axis] = k
    expanded = ww.reshape(view_shape).expand(shape)
    return expanded.reshape(-1).contiguous()


def matrix_mode_weights(matrix_shape: Iterable[int], mode_weights: torch.Tensor, *, mode_axis_period: int) -> torch.Tensor:
    """Expand weights for v22.94 ``w1_to_matrix`` layout: rows are edge-mode rows."""

    rows, cols = [int(x) for x in matrix_shape]
    k = int(mode_axis_period)
    if k <= 0:
        raise ValueError("mode_axis_period must be positive")
    ww = mode_weights.detach().to(dtype=torch.float64).reshape(-1)
    if int(ww.numel()) != k:
        raise ValueError(f"mode weight length {int(ww.numel())} does not match period {k}")
    row_weights = ww.repeat(int(math.ceil(rows / k)))[:rows]
    return row_weights.reshape(rows, 1).expand(rows, cols).reshape(-1).contiguous()


def basis_name_for_key(basis_key: str) -> str:
    key = str(basis_key).lower()
    return "fourier_lowfreq" if "four" in key or "fou" in key else "chebyshev"


def functional_edge_gram(
    basis_key: str,
    k: int,
    *,
    sobolev_order: float = 1.0,
    quadrature_points: int = 257,
    normalization: str = "trace",
    ridge: float = 1.0e-6,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Numerically estimate the edge-function Hilbert/Sobolev Gram.

    The Gram is built from the same fixed primitive basis used by PureKAN:

    ``G = int psi psi^T du + sobolev_order * int psi' psi'^T du``.

    This is a dense KxK Gram in function space, not a diagonal mode proxy.
    """

    kk = int(k)
    if kk <= 0:
        return torch.empty((0, 0), device=device, dtype=dtype)
    n = max(8, int(quadrature_points))
    z = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
    centers = torch.linspace(-1.0, 1.0, kk, device=device, dtype=dtype)
    scales = torch.tensor([max(0.2, 2.0 / max(1, kk - 1))], device=device, dtype=dtype)
    basis_name = basis_name_for_key(basis_key)
    phi = _basis_eval(z, basis_name, kk, centers, scales).reshape(n, kk).to(dtype=dtype)
    dphi = _basis_derivative(z, basis_name, kk, centers, scales).reshape(n, kk).to(dtype=dtype)
    dz = 2.0 / float(max(1, n - 1))
    weights = torch.ones(n, device=device, dtype=dtype) * dz
    weights[0] *= 0.5
    weights[-1] *= 0.5
    gram = phi.T @ (phi * weights.reshape(-1, 1))
    if float(sobolev_order) > 0.0:
        gram = gram + float(sobolev_order) * (dphi.T @ (dphi * weights.reshape(-1, 1)))
    gram = 0.5 * (gram + gram.T)
    if str(normalization) == "trace":
        gram = gram / (torch.trace(gram).div(float(max(1, kk))).clamp_min(EPS))
    elif str(normalization) == "mean":
        gram = gram / gram.mean().abs().clamp_min(EPS)
    else:
        gram = gram / torch.diag(gram).median().clamp_min(EPS)
    if float(ridge) > 0.0:
        gram = gram + float(ridge) * torch.eye(kk, device=device, dtype=dtype)
    return 0.5 * (gram + gram.T)


def functional_gram_condition(gram: torch.Tensor, eps: float = EPS) -> float:
    if int(gram.numel()) == 0:
        return 0.0
    vals = torch.linalg.eigvalsh(0.5 * (gram + gram.T)).to(dtype=torch.float64)
    return _safe_float(vals.max().clamp_min(eps) / vals.min().clamp_min(eps))


def _matrix_to_edge_vectors(matrix: torch.Tensor, k: int) -> torch.Tensor:
    mat = matrix.detach().to(dtype=torch.float64)
    rows, cols = int(mat.shape[0]), int(mat.shape[1])
    kk = int(k)
    edge_count = int(math.ceil(rows / kk))
    padded_rows = edge_count * kk
    if padded_rows != rows:
        pad = torch.zeros((padded_rows - rows, cols), dtype=mat.dtype, device=mat.device)
        mat = torch.cat([mat, pad], dim=0)
    return mat.reshape(edge_count, kk, cols).permute(0, 2, 1).reshape(edge_count * cols, kk)


def functional_gram_cost(matrix: torch.Tensor, gram: torch.Tensor, *, mode_axis_period: int) -> float:
    vecs = _matrix_to_edge_vectors(matrix, int(mode_axis_period))
    if int(vecs.numel()) == 0:
        return 0.0
    g = gram.to(device=vecs.device, dtype=vecs.dtype)
    return _safe_float((vecs @ g * vecs).sum())


def functional_gram_whitened_vector(matrix: torch.Tensor, gram: torch.Tensor, *, mode_axis_period: int, eps: float = EPS) -> torch.Tensor:
    vecs = _matrix_to_edge_vectors(matrix, int(mode_axis_period))
    if int(vecs.numel()) == 0:
        return torch.zeros(0, dtype=torch.float64)
    g = gram.to(device=vecs.device, dtype=vecs.dtype)
    vals, vec_basis = torch.linalg.eigh(0.5 * (g + g.T))
    inv_sqrt = (vec_basis * vals.clamp_min(float(eps)).rsqrt().reshape(1, -1)) @ vec_basis.T
    white = vecs @ inv_sqrt.T
    return white.reshape(-1).contiguous()


def functional_gram_inverse_retention(matrix: torch.Tensor, gram: torch.Tensor, *, mode_axis_period: int, eps: float = EPS) -> float:
    vecs = _matrix_to_edge_vectors(matrix, int(mode_axis_period))
    if int(vecs.numel()) == 0:
        return 0.0
    g = gram.to(device=vecs.device, dtype=vecs.dtype)
    vals, vec_basis = torch.linalg.eigh(0.5 * (g + g.T))
    inv = (vec_basis * vals.clamp_min(float(eps)).reciprocal().reshape(1, -1)) @ vec_basis.T
    energy = vecs.square().sum().clamp_min(float(eps))
    low_cost_energy = (vecs @ inv * vecs).sum()
    normalizer = torch.trace(inv).div(float(max(1, int(g.shape[0])))).clamp_min(float(eps))
    return _safe_float(low_cost_energy / (energy * normalizer))


def edge_sobolev_whiten(x: torch.Tensor, weights: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    ww = weights.to(device=x.device, dtype=x.dtype).reshape(-1)
    xx = x.reshape(-1)
    if int(ww.numel()) != int(xx.numel()):
        raise ValueError(f"weight length {int(ww.numel())} != tensor length {int(xx.numel())}")
    return (xx / ww.clamp_min(float(eps)).sqrt()).reshape_as(x)


def edge_sobolev_unwhiten(x: torch.Tensor, weights: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    ww = weights.to(device=x.device, dtype=x.dtype).reshape(-1)
    xx = x.reshape(-1)
    if int(ww.numel()) != int(xx.numel()):
        raise ValueError(f"weight length {int(ww.numel())} != tensor length {int(xx.numel())}")
    return (xx * ww.clamp_min(float(eps)).sqrt()).reshape_as(x)


def edge_sobolev_norm(x: torch.Tensor, weights: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    ww = weights.to(device=x.device, dtype=x.dtype).reshape(-1).clamp_min(float(eps))
    xx = x.reshape(-1)
    if int(ww.numel()) != int(xx.numel()):
        raise ValueError(f"weight length {int(ww.numel())} != tensor length {int(xx.numel())}")
    return torch.sqrt((ww * xx.square()).sum().clamp_min(float(eps)))


def inverse_weight_retention(x: torch.Tensor, weights: torch.Tensor, eps: float = EPS) -> float:
    """Continuous low-cost retention score.

    A score above 1 means the vector is more concentrated in low-cost modes than
    a uniform-energy vector under this metric; below 1 means high-cost
    concentration. It is not used as a probability.
    """

    xx = x.detach().reshape(-1).to(dtype=torch.float64)
    ww = weights.detach().reshape(-1).to(dtype=torch.float64).clamp_min(float(eps))
    if int(xx.numel()) == 0 or int(xx.numel()) != int(ww.numel()):
        return 0.0
    energy = xx.square()
    denom = energy.sum().clamp_min(float(eps)) * (1.0 / ww).mean().clamp_min(float(eps))
    return _safe_float((energy / ww).sum() / denom)


def mode_band_slices(k: int, basis_key: str) -> dict[str, list[int]]:
    kk = int(k)
    key = str(basis_key).lower()
    if "four" in key or "fou" in key:
        return {
            "constant": list(range(0, min(1, kk))),
            "low_harmonic": list(range(1, min(3, kk))),
            "mid_harmonic": list(range(3, min(5, kk))),
            "high_harmonic": list(range(5, kk)),
        }
    return {
        "degree_0_1": list(range(0, min(2, kk))),
        "degree_2_3": list(range(2, min(4, kk))),
        "degree_4_5": list(range(4, min(6, kk))),
        "degree_6_plus": list(range(6, kk)),
    }


def edge_sobolev_smoke_test() -> dict[str, float | int]:
    x = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    w = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    z = edge_sobolev_whiten(x, w)
    xr = edge_sobolev_unwhiten(z, w)
    step = edge_sobolev_whiten(torch.ones_like(x), w).abs()
    return {
        "whiten_unwhiten_identity_error": _safe_float((xr - x).abs().max()),
        "high_cost_step_smaller": int(_safe_float(step[-1]) < _safe_float(step[0])),
        "mode_weight_condition": mode_weight_summary(w, "toy", 1.0, "none").mode_weight_condition,
    }
