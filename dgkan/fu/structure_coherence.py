"""Structure-transform coherence helpers for DG-KAN FU diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


EPS = 1.0e-12


@dataclass(frozen=True)
class TransformSpec:
    name: str
    align_with_transform: bool = True


def _f(value: torch.Tensor | float, default: float = 0.0) -> float:
    try:
        out = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def transform_perm(side: int, name: str, *, seed: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(int(side) * int(side), device=device).reshape(int(side), int(side))

    def swap_patch(grid: torch.Tensor, a: tuple[int, int], b: tuple[int, int], size: int = 2) -> torch.Tensor:
        out_grid = grid.clone()
        ar, ac = a
        br, bc = b
        if min(ar, ac, br, bc) < 0 or max(ar + size, br + size) > int(side) or max(ac + size, bc + size) > int(side):
            return out_grid
        first = grid[ar : ar + size, ac : ac + size].clone()
        second = grid[br : br + size, bc : bc + size].clone()
        out_grid[ar : ar + size, ac : ac + size] = second
        out_grid[br : br + size, bc : bc + size] = first
        return out_grid

    if name == "shift_down":
        out = torch.roll(idx, shifts=1, dims=0)
    elif name == "shift_right":
        out = torch.roll(idx, shifts=1, dims=1)
    elif name == "patch_shift_diag":
        out = torch.roll(idx, shifts=(2, 2), dims=(0, 1))
    elif name == "corner_preserving_jitter":
        out = idx.clone()
        if int(side) >= 4:
            out[1:-1, 1:-1] = torch.roll(out[1:-1, 1:-1], shifts=1, dims=0)
    elif name == "rot90":
        out = torch.rot90(idx, 1, dims=(0, 1))
    elif name == "hflip":
        out = torch.flip(idx, dims=[1])
    elif name == "vflip":
        out = torch.flip(idx, dims=[0])
    elif name == "rot180":
        out = torch.rot90(idx, 2, dims=(0, 1))
    elif name == "swap_top_patch_pair":
        out = swap_patch(idx, (1, 1), (1, int(side) - 3))
    elif name == "swap_bottom_patch_pair":
        out = swap_patch(idx, (int(side) - 3, 1), (int(side) - 3, int(side) - 3))
    elif name == "swap_local_patch_pairs":
        out = swap_patch(idx, (1, 1), (1, int(side) - 3))
        out = swap_patch(out, (int(side) - 3, 1), (int(side) - 3, int(side) - 3))
    elif name == "swap_rotation_checkers":
        out = swap_patch(idx, (1, int(side) // 2 - 1), (int(side) - 3, int(side) // 2 - 1))
    elif name.startswith("destructive"):
        gen = torch.Generator(device=device).manual_seed(int(seed))
        out = torch.randperm(int(side) * int(side), generator=gen, device=device).reshape(int(side), int(side))
    else:
        out = idx
    return out.reshape(-1).long()


def transform_specs(family: str, profile: str = "default") -> list[TransformSpec]:
    if family == "local":
        if profile == "task_orbit":
            return [TransformSpec(x) for x in ["swap_top_patch_pair", "swap_bottom_patch_pair", "swap_local_patch_pairs"]]
        base = ["shift_down", "shift_right", "patch_shift_diag", "corner_preserving_jitter"]
        return [TransformSpec(x) for x in (base + (["shift_down", "patch_shift_diag"] if profile == "expanded" else []))]
    if family == "rotation":
        if profile == "task_orbit":
            return [TransformSpec(x) for x in ["hflip", "vflip", "rot180", "swap_rotation_checkers"]]
        base = ["rot90", "hflip", "vflip"]
        return [TransformSpec(x) for x in (base + (["rot90", "hflip"] if profile == "expanded" else []))]
    if family.startswith("destructive"):
        return [TransformSpec(family, align_with_transform=False)]
    return []


def apply_perm(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return x[:, perm.to(device=x.device)]


def permute_param_examples(grads: torch.Tensor, param: torch.nn.Parameter, perm: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    if int(getattr(param, "_kan_layer_id", -1)) != 0 or int(param.ndim) < 3:
        return grads
    if int(param.shape[0]) != int(perm.numel()):
        return grads
    idx = perm.to(device=grads.device)
    if inverse:
        inv = torch.empty_like(idx)
        inv[idx] = torch.arange(int(idx.numel()), device=idx.device)
        idx = inv
    return grads[:, idx, :, :]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64).cpu()
    bb = b.detach().reshape(-1).to(dtype=torch.float64).cpu()
    n = min(int(aa.numel()), int(bb.numel()))
    if n <= 0:
        return 0.0
    aa = aa[:n]
    bb = bb[:n]
    return _f((aa @ bb) / (aa.norm().clamp_min(EPS) * bb.norm().clamp_min(EPS)))


def clean_blocks(blocks: list[list[int]], n: int) -> list[list[int]]:
    out = [sorted({int(i) for i in block if 0 <= int(i) < int(n)}) for block in blocks]
    return [block for block in out if block]


def block_shuffle(values: torch.Tensor, blocks: list[list[int]], seed: int) -> torch.Tensor:
    out = values.detach().clone().reshape(-1).cpu().to(dtype=torch.float64)
    blocks = clean_blocks(blocks, int(out.numel()))
    gen = torch.Generator().manual_seed(int(seed))
    by_size: dict[int, list[list[int]]] = {}
    for block in blocks:
        by_size.setdefault(len(block), []).append(block)
    for group_blocks in by_size.values():
        if len(group_blocks) <= 1:
            continue
        vals = [out[torch.tensor(block, dtype=torch.long)].clone() for block in group_blocks]
        order = torch.randperm(len(vals), generator=gen).tolist()
        for block, src_idx in zip(group_blocks, order):
            src = vals[int(src_idx)]
            if int(src.numel()) != len(block):
                src = src[:1].repeat(len(block))
            out[torch.tensor(block, dtype=torch.long)] = src
    return out.to(device=values.device, dtype=values.dtype)


def phase_shuffle(values: torch.Tensor, blocks: list[list[int]], seed: int) -> torch.Tensor:
    shuffled = block_shuffle(values, blocks, seed)
    out = shuffled.detach().clone().reshape(-1).cpu().to(dtype=torch.float64)
    blocks = clean_blocks(blocks, int(out.numel()))
    gen = torch.Generator().manual_seed(int(seed))
    for block in blocks:
        if len(block) <= 1:
            continue
        idx = torch.tensor(block, dtype=torch.long)
        order = torch.randperm(len(block), generator=gen)
        out[idx] = out[idx][order]
    return out.to(device=values.device, dtype=values.dtype)


def density_random_like(values: torch.Tensor, seed: int) -> torch.Tensor:
    flat = values.detach().reshape(-1)
    if int(flat.numel()) <= 0:
        return flat.clone()
    gen = torch.Generator(device=flat.device).manual_seed(int(seed))
    prob = float(flat.mean().clamp(0.0, 1.0).item())
    return (torch.rand(flat.shape, generator=gen, device=flat.device, dtype=flat.dtype) < prob).to(dtype=flat.dtype)


def hist_random_like(values: torch.Tensor, seed: int) -> torch.Tensor:
    flat = values.detach().reshape(-1)
    if int(flat.numel()) <= 0:
        return flat.clone()
    gen = torch.Generator(device=flat.device).manual_seed(int(seed))
    return flat[torch.randperm(int(flat.numel()), generator=gen, device=flat.device)]


def quantile_normalize(values: torch.Tensor, target_density: float) -> torch.Tensor:
    flat = values.detach().reshape(-1).to(dtype=torch.float64)
    if int(flat.numel()) <= 0:
        return values.detach().clone()
    target = float(max(0.01, min(0.99, target_density)))
    rank = torch.argsort(torch.argsort(flat))
    scaled = (rank.to(dtype=torch.float64) + 0.5) / float(flat.numel())
    threshold = 1.0 - target
    out = torch.clamp((scaled - threshold) / max(target, EPS), 0.0, 1.0)
    return out.reshape_as(values).to(device=values.device, dtype=values.dtype)


def weighted_structure_score(gate: torch.Tensor, struct: torch.Tensor) -> float:
    g = gate.detach().reshape(-1).to(dtype=torch.float64).cpu().clamp_min(0.0)
    s = struct.detach().reshape(-1).to(dtype=torch.float64).cpu().clamp_min(0.0)
    n = min(int(g.numel()), int(s.numel()))
    if n <= 0:
        return 0.0
    return _f((g[:n] * s[:n]).sum() / g[:n].sum().clamp_min(EPS))
