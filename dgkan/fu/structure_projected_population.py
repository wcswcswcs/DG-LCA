"""Population-risk gates computed after structure projection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from dgkan.fu.population_risk_gate import block_snr_gate
from dgkan.fu.structure_coherence import clean_blocks, permute_param_examples


EPS = 1.0e-12


@dataclass
class StructureProjectedGate:
    gate: torch.Tensor
    snr: torch.Tensor
    projected_grads: torch.Tensor
    projection_energy: float
    projection_retention: float
    snr_mean: float
    orbit_residual_snr: float


def _f(value: torch.Tensor | float, default: float = 0.0) -> float:
    try:
        out = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def structure_project_grads(
    source_grads: torch.Tensor,
    param: torch.nn.Parameter,
    transformed: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    aligned = [source_grads]
    for transformed_grads, perm in transformed:
        aligned.append(permute_param_examples(transformed_grads, param, perm, inverse=True))
    n = min(int(x.shape[0]) for x in aligned) if aligned else 0
    if n <= 0:
        return source_grads[:0]
    return torch.stack([x[:n] for x in aligned], dim=0).mean(dim=0)


def blockwise_orbit_residual_gate(
    snr: torch.Tensor,
    blocks: list[list[int]],
    *,
    beta: float,
    seed: int,
    samples: int = 5,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, float]:
    flat = snr.detach().reshape(-1).cpu().to(dtype=torch.float64)
    out = torch.zeros_like(flat)
    blocks = clean_blocks(blocks, int(flat.numel()))
    if not blocks:
        return out.to(device=snr.device, dtype=snr.dtype), 0.0
    block_vals = torch.tensor([float(flat[block].mean().item()) for block in blocks], dtype=torch.float64)
    gen = torch.Generator().manual_seed(int(seed))
    residuals = []
    for i, block in enumerate(blocks):
        draws = []
        same_size = [j for j, other in enumerate(blocks) if len(other) == len(block) and j != i]
        if not same_size:
            same_size = [j for j in range(len(blocks)) if j != i] or [i]
        for _ in range(max(1, int(samples))):
            j = same_size[int(torch.randint(0, len(same_size), (1,), generator=gen).item())]
            draws.append(float(block_vals[j].item()))
        baseline = float(torch.tensor(draws, dtype=torch.float64).median().item())
        residual = math.log(float(block_vals[i].item()) + float(eps)) - math.log(baseline + float(eps))
        residuals.append(residual)
        q = 1.0 / (1.0 + math.exp(-float(beta) * residual))
        out[torch.tensor(block, dtype=torch.long)] = q
    return out.to(device=snr.device, dtype=snr.dtype), float(torch.tensor(residuals, dtype=torch.float64).mean().item())


def structure_projected_block_snr_gate(
    opt: object,
    param: torch.nn.Parameter,
    group: dict,
    source_grads: torch.Tensor,
    transformed: list[tuple[torch.Tensor, torch.Tensor]],
    blocks: list[list[int]],
    *,
    beta: float,
    residualize_orbit: bool = False,
    residual_seed: int = 0,
) -> StructureProjectedGate:
    projected = structure_project_grads(source_grads, param, transformed)
    if int(projected.numel()) <= 0:
        z = torch.zeros(int(param.numel()), device=param.device, dtype=param.dtype)
        return StructureProjectedGate(z, z, projected, 0.0, 0.0, 0.0, 0.0)
    white_source = opt._apply_metric_inv_sqrt(source_grads[: int(projected.shape[0])], param, group).reshape(int(projected.shape[0]), -1)
    white_projected = opt._apply_metric_inv_sqrt(projected, param, group).reshape(int(projected.shape[0]), -1)
    if not blocks:
        gate = torch.ones(int(param.numel()), device=param.device, dtype=param.dtype)
        snr = gate.detach().clone()
    else:
        res = block_snr_gate(white_projected.detach().cpu(), blocks, beta=float(beta))
        gate = res.gate.to(device=param.device, dtype=param.dtype).reshape(-1)
        snr = res.snr.to(device=param.device, dtype=param.dtype).reshape(-1)
    orbit_residual = 0.0
    if residualize_orbit:
        gate, orbit_residual = blockwise_orbit_residual_gate(snr, blocks, beta=float(beta), seed=int(residual_seed))
    proj_energy = _f(white_projected.square().mean()) if int(white_projected.numel()) else 0.0
    src_energy = _f(white_source.square().mean()) if int(white_source.numel()) else 0.0
    retention = proj_energy / max(src_energy, EPS)
    return StructureProjectedGate(
        gate=gate,
        snr=snr,
        projected_grads=projected,
        projection_energy=proj_energy,
        projection_retention=retention,
        snr_mean=_f(snr.mean()) if int(snr.numel()) else 0.0,
        orbit_residual_snr=orbit_residual,
    )
