"""Low-rank source manifold basis construction for v22.15."""

from __future__ import annotations

from typing import Any

import torch


EPS = 1.0e-12


FIREWALL_FIELDS = {
    "uses_full_weight_generator": 0,
    "uses_trainable_hypernetwork": 0,
    "uses_parameter_compression_objective": 0,
    "uses_mapping_loss_in_task_objective": 0,
    "trains_only_latent_instead_of_model": 0,
    "reports_compression_as_success": 0,
    "source_manifold_basis_train_only": 1,
    "controls_have_own_random_or_shuffled_basis": 1,
}


def _orthonormal_columns(matrix: torch.Tensor, dim: int) -> torch.Tensor:
    x = matrix.detach().float()
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if x.shape[1] == 0:
        raise ValueError("cannot build a basis from zero columns")
    try:
        q, _ = torch.linalg.qr(x, mode="reduced")
    except RuntimeError:
        u, _, _ = torch.linalg.svd(x, full_matrices=False)
        q = u
    k = min(int(dim), int(q.shape[1]))
    return q[:, :k].contiguous()


def source_history_basis(history: torch.Tensor, dim: int) -> tuple[torch.Tensor, dict[str, Any]]:
    h = history.detach().float()
    if h.ndim == 1:
        h = h.unsqueeze(0)
    centered = h - h.mean(dim=0, keepdim=True)
    try:
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        k = max(1, min(int(dim), int(vh.shape[0])))
        basis = vh[:k].T.contiguous()
        energy = centered.square().sum().clamp_min(EPS)
        kept = s[:k].square().sum()
        coverage = float((kept / energy).item())
    except RuntimeError:
        basis = _orthonormal_columns(centered.T, dim)
        coverage = 0.0
    return basis, {"basis_source": "source_history", "manifold_coverage_ratio": coverage}


def split_coherent_basis(updates: torch.Tensor, dim: int) -> tuple[torch.Tensor, dict[str, Any]]:
    u = updates.detach().float()
    if u.ndim == 1:
        u = u.unsqueeze(0)
    signs = torch.sign((u * u.mean(dim=0, keepdim=True)).sum(dim=1, keepdim=True)).clamp_min(0.0) * 2.0 - 1.0
    coherent = u * signs
    return source_history_basis(coherent, dim)


def random_basis(width: int, dim: int, *, seed: int, device: torch.device | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    gen = torch.Generator(device=device or "cpu")
    gen.manual_seed(int(seed))
    raw = torch.randn(width, dim, generator=gen, device=device)
    return _orthonormal_columns(raw, dim), {"basis_source": "random_control", "manifold_coverage_ratio": ""}


def shuffled_basis(history: torch.Tensor, dim: int, *, seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    h = history.detach().float()
    gen = torch.Generator(device=h.device)
    gen.manual_seed(int(seed))
    perm = torch.randperm(h.shape[-1], generator=gen, device=h.device)
    basis, row = source_history_basis(h[..., perm], dim)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=h.device)
    basis = basis[inverse]
    row["basis_source"] = "shuffled_control"
    return basis, row


def signflip_basis(history: torch.Tensor, dim: int) -> tuple[torch.Tensor, dict[str, Any]]:
    basis, row = source_history_basis(-history.detach().float(), dim)
    row["basis_source"] = "signflip_control"
    return basis, row


def build_source_manifold_basis(
    family: str,
    history: torch.Tensor,
    *,
    dim: int,
    seed: int = 2215,
) -> tuple[torch.Tensor, dict[str, Any]]:
    width = int(history.shape[-1]) if history.ndim > 1 else int(history.numel())
    if family in {"ReadoutSourcePCA", "HiddenReadoutTrajectoryBasis", "D-CHELowDegreeBasisManifold", "D-FOULowFrequencyBasisManifold"}:
        basis, row = source_history_basis(history, dim)
    elif family == "SplitCoherentSourceBasis":
        basis, row = split_coherent_basis(history, dim)
    elif family == "ControlNullSourceBasis":
        h = history.detach().float()
        centered = h - h.mean(dim=0, keepdim=True)
        basis, row = source_history_basis(centered, dim)
        row["basis_source"] = "optimizer_state"
    elif family == "RandomManifold":
        basis, row = random_basis(width, dim, seed=seed, device=history.device)
    elif family == "ShuffledSourceManifold":
        basis, row = shuffled_basis(history, dim, seed=seed)
    elif family == "SignFlipManifold":
        basis, row = signflip_basis(history, dim)
    else:
        raise ValueError(f"unknown source manifold family: {family}")
    return basis, {
        "source_manifold_family": family,
        "source_manifold_dim": int(basis.shape[1]),
        "uses_validation_test_future": 0,
        **FIREWALL_FIELDS,
        **row,
    }


__all__ = [
    "FIREWALL_FIELDS",
    "build_source_manifold_basis",
    "random_basis",
    "shuffled_basis",
    "signflip_basis",
    "source_history_basis",
    "split_coherent_basis",
]
