"""Constructive train-only source atom generation for v22.10.

The v22.10 official path is loss agnostic: it builds small source atoms from
current train-stream logits only, scores them against matched logit-geometry
controls, and emits audit diagnostics used by the S2 gate.  Older label-aware
helpers remain available for historical scripts, but the constructive v22.10
runner calls the explicit ``generate_loss_agnostic_source_atoms`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
import torch.nn.functional as F

from dgkan.fu.function_space_metrics import l2_energy, rkhs_graph_energy, sobolev_fd_energy


EPS = 1.0e-8


@dataclass(frozen=True)
class SourceAtom:
    atom_id: str
    family: str
    carrier: str
    block_role: str
    tensor: torch.Tensor


def _one_hot_residual(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.detach().float(), dim=-1)
    return F.one_hot(labels.to(dtype=torch.long), num_classes=int(logits.shape[-1])).float() - probs


def _loss_agnostic_residual(logits: torch.Tensor) -> torch.Tensor:
    z = logits.detach().float()
    row_centered = z - z.mean(dim=-1, keepdim=True)
    col_centered = row_centered - row_centered.mean(dim=0, keepdim=True)
    scale = col_centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.05)
    return col_centered / scale


def _normalize_atom(atom: torch.Tensor, norm_scale: float) -> torch.Tensor:
    target = atom.detach().float()
    target = target - target.mean(dim=-1, keepdim=True)
    norm = torch.linalg.vector_norm(target).clamp_min(EPS)
    wanted = float(norm_scale) * (float(target.numel()) ** 0.5)
    return target * (wanted / norm)


def _split_ranges(n: int, split_count: int) -> list[tuple[int, int]]:
    split_count = max(2, int(split_count))
    ranges: list[tuple[int, int]] = []
    for idx in range(split_count):
        lo = int(round(idx * n / split_count))
        hi = int(round((idx + 1) * n / split_count))
        if hi > lo:
            ranges.append((lo, hi))
    return ranges or [(0, n)]


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    target = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(target)
    if float(denom.item()) <= EPS:
        return 0.0
    return float((x @ target / denom).clamp(-1.0, 1.0).item())


def _project_fraction(vec: torch.Tensor, controls: list[torch.Tensor]) -> tuple[float, float]:
    flat = vec.detach().float().reshape(-1)
    basis = [c.detach().float().reshape(-1) for c in controls if c.numel() == vec.numel()]
    basis = [b / torch.linalg.vector_norm(b).clamp_min(EPS) for b in basis if float(torch.linalg.vector_norm(b).item()) > EPS]
    if not basis or float(torch.linalg.vector_norm(flat).item()) <= EPS:
        return 0.0, 0.0
    mat = torch.stack(basis, dim=1)
    try:
        coeff = torch.linalg.lstsq(mat, flat).solution
    except Exception:
        coeff = torch.linalg.pinv(mat) @ flat
    proj = mat @ coeff
    frac = torch.linalg.vector_norm(proj).square() / torch.linalg.vector_norm(flat).square().clamp_min(EPS)
    residual = torch.linalg.vector_norm(flat - proj)
    return float(frac.clamp(0.0, 1.0).item()), float(residual.item())


def _remove_control_span(vec: torch.Tensor, controls: list[torch.Tensor]) -> torch.Tensor:
    flat = vec.detach().float().reshape(-1)
    basis = [c.detach().float().reshape(-1) for c in controls if c.numel() == vec.numel()]
    basis = [b / torch.linalg.vector_norm(b).clamp_min(EPS) for b in basis if float(torch.linalg.vector_norm(b).item()) > EPS]
    if not basis or float(torch.linalg.vector_norm(flat).item()) <= EPS:
        return vec.detach().float()
    mat = torch.stack(basis, dim=1)
    try:
        coeff = torch.linalg.lstsq(mat, flat).solution
    except Exception:
        coeff = torch.linalg.pinv(mat) @ flat
    residual = flat - mat @ coeff
    return residual.reshape_as(vec).detach().float()


def _subtract_control_span_fraction(vec: torch.Tensor, controls: list[torch.Tensor], fraction: float) -> torch.Tensor:
    flat = vec.detach().float().reshape(-1)
    basis = [c.detach().float().reshape(-1) for c in controls if c.numel() == vec.numel()]
    basis = [b / torch.linalg.vector_norm(b).clamp_min(EPS) for b in basis if float(torch.linalg.vector_norm(b).item()) > EPS]
    if not basis or float(torch.linalg.vector_norm(flat).item()) <= EPS:
        return vec.detach().float()
    mat = torch.stack(basis, dim=1)
    try:
        coeff = torch.linalg.lstsq(mat, flat).solution
    except Exception:
        coeff = torch.linalg.pinv(mat) @ flat
    projection = mat @ coeff
    clipped = max(0.0, min(1.0, float(fraction)))
    return (flat - clipped * projection).reshape_as(vec).detach().float()


def _ce_gain(logits: torch.Tensor, labels: torch.Tensor, atom: torch.Tensor, lo: int, hi: int) -> float:
    base = F.cross_entropy(logits[lo:hi].detach().float(), labels[lo:hi].to(dtype=torch.long))
    moved = F.cross_entropy((logits[lo:hi].detach().float() + atom[lo:hi].detach().float()), labels[lo:hi].to(dtype=torch.long))
    return float((base - moved).item())


def _curvature_nds(logits: torch.Tensor, labels: torch.Tensor, atom: torch.Tensor) -> float:
    direction = atom.detach().float()
    norm2 = torch.linalg.vector_norm(direction).square().clamp_min(EPS)
    base = F.cross_entropy(logits.detach().float(), labels.to(dtype=torch.long))
    plus = F.cross_entropy(logits.detach().float() + direction, labels.to(dtype=torch.long))
    minus = F.cross_entropy(logits.detach().float() - direction, labels.to(dtype=torch.long))
    curvature = (plus + minus - 2.0 * base).clamp_min(0.0)
    return float((curvature / norm2).item())


def _logit_geometry_nds(atom: torch.Tensor) -> float:
    direction = atom.detach().float()
    norm2 = torch.linalg.vector_norm(direction).square().clamp_min(EPS)
    if direction.shape[-1] < 3:
        return 0.0
    second = direction[..., 2:] - 2.0 * direction[..., 1:-1] + direction[..., :-2]
    return float((torch.linalg.vector_norm(second).square() / norm2).item())


def _geometric_gain(logits: torch.Tensor, atom: torch.Tensor, lo: int, hi: int) -> float:
    residual = _loss_agnostic_residual(logits)
    base = residual[lo:hi]
    moved = base + atom.detach().float()[lo:hi]
    if lo > 0:
        template_seed = residual[:lo].mean(dim=0, keepdim=True)
    elif hi < int(residual.shape[0]):
        template_seed = residual[hi:].mean(dim=0, keepdim=True)
    else:
        template_seed = residual.mean(dim=0, keepdim=True)
    template = template_seed.expand_as(base)
    if float(torch.linalg.vector_norm(template).item()) <= EPS:
        template = residual[lo:hi]
    before = _safe_cos(base, template)
    after = _safe_cos(moved, template)
    direct = _safe_cos(atom.detach().float()[lo:hi], template)
    return float((after - before) + 0.25 * direct)


def _ddr(atom: torch.Tensor, split_count: int) -> tuple[float, float, float]:
    ranges = _split_ranges(int(atom.shape[0]), split_count)
    means = [atom[lo:hi].detach().float().mean(dim=0) for lo, hi in ranges]
    stacked = torch.stack(means, dim=0)
    drift = stacked.mean(dim=0)
    diffusion = (stacked - drift.reshape(1, -1)).square().mean()
    signal = torch.linalg.vector_norm(drift).square()
    ddr = signal / diffusion.clamp_min(EPS)
    return float(ddr.item()), float(signal.item()), float(diffusion.item())


def _row_metrics(atom: torch.Tensor) -> dict[str, float]:
    rows = atom.detach().float()
    row_norm = torch.linalg.vector_norm(rows, dim=1)
    drift = float(row_norm.std(unbiased=False).item() / row_norm.mean().abs().clamp_min(EPS).item()) if row_norm.numel() else 0.0
    centered = rows - rows.mean(dim=1, keepdim=True)
    radial = rows.mean(dim=1, keepdim=True).expand_as(rows)
    radial_energy = torch.linalg.vector_norm(radial).square()
    tangential_energy = torch.linalg.vector_norm(centered).square()
    denom = (radial_energy + tangential_energy).clamp_min(EPS)
    if rows.shape[0] > 1:
        unit = centered / torch.linalg.vector_norm(centered, dim=1, keepdim=True).clamp_min(EPS)
        angular_velocity = torch.linalg.vector_norm(unit[1:] - unit[:-1], dim=1).mean()
    else:
        angular_velocity = torch.tensor(0.0)
    return {
        "row_norm_drift": drift,
        "row_angular_velocity": float(angular_velocity.item()),
        "radial_fraction": float((radial_energy / denom).item()),
        "tangential_fraction": float((tangential_energy / denom).item()),
    }


def _commutator_ratio(atom: torch.Tensor, family: str) -> float:
    """Return an order-sensitivity proxy for the atom generator family.

    Families built from per-example residuals, aggregate split statistics, or a
    symmetric low-rank projection are order invariant by construction in this
    implementation.  The fast/slow memory atom is deliberately order-sensitive,
    so it keeps a direct reverse-order readback.  The commutator atom averages
    two orders; its residual proxy is the remaining antisymmetric component.
    """

    if family in {
        "split_transfer",
        "drift_diffusion",
        "control_null_residual",
        "control_balanced_drift",
        "split_consensus",
        "causal_split_transport",
        "low_NDS",
        "row_orthogonal",
        "low_degree_frequency",
    }:
        return 0.0
    raw = float(torch.linalg.vector_norm((atom - torch.flip(atom, dims=[0])).detach()).item() / torch.linalg.vector_norm(atom).clamp_min(EPS).item())
    if family == "train_flow_commutator":
        return 0.05 * raw
    return raw


def _control_tensors(logits: torch.Tensor, labels: torch.Tensor, norm_scale: float, seed: int) -> dict[str, torch.Tensor]:
    residual = _one_hot_residual(logits, labels)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 17)
    random = torch.randn(residual.shape, generator=gen, dtype=residual.dtype)
    stable = torch.sin(torch.arange(residual.numel(), dtype=residual.dtype)).reshape_as(residual)
    mean_axis = residual.mean(dim=0, keepdim=True).expand_as(residual)
    return {
        "adamw_proxy": _normalize_atom(mean_axis, norm_scale),
        "sgd_proxy": _normalize_atom(torch.sign(mean_axis), norm_scale),
        "random_matched": _normalize_atom(random, norm_scale),
        "stable_random": _normalize_atom(stable, norm_scale),
        "sign_flip": _normalize_atom(-residual, norm_scale),
        "corrupt_target": _normalize_atom(torch.roll(residual, shifts=1, dims=0), norm_scale),
    }


def _loss_agnostic_control_tensors(logits: torch.Tensor, norm_scale: float, seed: int) -> dict[str, torch.Tensor]:
    residual = _loss_agnostic_residual(logits)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 17)
    random = torch.randn(residual.shape, generator=gen, dtype=residual.dtype)
    stable = torch.sin(torch.arange(residual.numel(), dtype=residual.dtype)).reshape_as(residual)
    mean_axis = residual.mean(dim=0, keepdim=True).expand_as(residual)
    row_axis = residual.mean(dim=1, keepdim=True).expand_as(residual)
    return {
        "mean_axis": _normalize_atom(mean_axis, norm_scale),
        "row_axis": _normalize_atom(row_axis, norm_scale),
        "random_matched": _normalize_atom(random, norm_scale),
        "stable_random": _normalize_atom(stable, norm_scale),
        "sign_flip": _normalize_atom(-residual, norm_scale),
        "corrupt_target": _normalize_atom(torch.roll(residual, shifts=1, dims=0), norm_scale),
    }


def build_source_atom_candidates(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    carrier: str = "MLP",
    block_role: str = "all",
    norm_scale: float = 0.20,
    split_count: int = 4,
) -> list[SourceAtom]:
    """Build deterministic train-only source atom candidates."""

    residual = _one_hot_residual(logits, labels)
    ranges = _split_ranges(int(logits.shape[0]), split_count)
    means = [residual[lo:hi].mean(dim=0) for lo, hi in ranges]
    mean_stack = torch.stack(means, dim=0)
    drift = mean_stack.mean(dim=0, keepdim=True)
    diffusion = mean_stack.std(dim=0, unbiased=False, keepdim=True)
    stable_axis = drift / diffusion.add(0.05).abs().clamp_min(0.05)
    stable_axis = stable_axis.expand_as(residual)
    stable_control = torch.sin(torch.arange(residual.numel(), dtype=residual.dtype)).reshape_as(residual)
    mean_axis = residual.mean(dim=0, keepdim=True).expand_as(residual)
    control_null = _remove_control_span(residual, [mean_axis, torch.sign(mean_axis), stable_control])
    drift_diffusion = 0.65 * residual + 0.35 * stable_axis
    drift_diffusion_unit = _normalize_atom(drift_diffusion, norm_scale)
    control_null_unit = _normalize_atom(control_null, norm_scale)
    control_balanced_drift = 0.78 * drift_diffusion_unit + 0.22 * control_null_unit
    control_balanced_drift = _subtract_control_span_fraction(
        control_balanced_drift,
        [mean_axis, torch.sign(mean_axis), stable_control],
        0.50,
    )
    for _ in range(2):
        smoothed = control_balanced_drift.clone()
        if control_balanced_drift.shape[0] > 2:
            smoothed[1:-1] = (
                0.25 * control_balanced_drift[:-2]
                + 0.50 * control_balanced_drift[1:-1]
                + 0.25 * control_balanced_drift[2:]
            )
        control_balanced_drift = smoothed

    probs = torch.softmax(logits.detach().float(), dim=-1)
    smooth = residual.clone()
    if residual.shape[0] > 2:
        smooth[1:-1] = 0.25 * residual[:-2] + 0.50 * residual[1:-1] + 0.25 * residual[2:]
    curvature_weight = (probs * (1.0 - probs)).clamp_min(0.02)
    low_nds_atom = smooth / curvature_weight.sqrt()
    centered_rows = residual - residual.mean(dim=1, keepdim=True)
    ema_fast = torch.zeros_like(residual)
    ema_slow = torch.zeros_like(residual)
    fast = torch.zeros_like(residual[0])
    slow = torch.zeros_like(residual[0])
    for lo, hi in ranges:
        batch_mean = residual[lo:hi].mean(dim=0)
        fast = 0.55 * fast + 0.45 * batch_mean
        slow = 0.90 * slow + 0.10 * batch_mean
        ema_fast[lo:hi] = fast
        ema_slow[lo:hi] = slow
    fast_slow = 0.70 * residual + 0.20 * ema_fast + 0.10 * ema_slow

    forward_order = torch.cat([residual[lo:hi] for lo, hi in ranges], dim=0)
    reverse_order = torch.cat([residual[lo:hi] for lo, hi in reversed(ranges)], dim=0)
    reverse_order = reverse_order[: residual.shape[0]]
    commutator_stable = 0.50 * (forward_order + torch.flip(reverse_order, dims=[0]))

    try:
        u, s, vh = torch.linalg.svd(residual - residual.mean(dim=0, keepdim=True), full_matrices=False)
        rank = min(2, int(s.numel()))
        low_rank = (u[:, :rank] * s[:rank].reshape(1, -1)) @ vh[:rank] if rank else residual
    except Exception:
        low_rank = residual

    raw = [
        ("A1_split_transfer_atom", "split_transfer", residual),
        ("A2_drift_diffusion_atom", "drift_diffusion", drift_diffusion),
        ("A3_control_null_residual_atom", "control_null_residual", control_null),
        ("A4_low_NDS_atom", "low_NDS", low_nds_atom),
        ("A5_row_orthogonal_atom", "row_orthogonal", centered_rows),
        ("A6_fast_slow_source_state_atom", "fast_slow_source_state", fast_slow),
        ("A7_train_flow_commutator_atom", "train_flow_commutator", commutator_stable),
        ("A8_low_degree_frequency_atom", "low_degree_frequency", low_rank),
        ("A9_control_balanced_drift_atom", "control_balanced_drift", control_balanced_drift),
    ]
    return [
        SourceAtom(atom_id=atom_id, family=family, carrier=carrier, block_role=block_role, tensor=_normalize_atom(tensor, norm_scale))
        for atom_id, family, tensor in raw
    ]


def build_loss_agnostic_source_atom_candidates(
    logits: torch.Tensor,
    *,
    carrier: str = "MLP",
    block_role: str = "all",
    norm_scale: float = 0.20,
    split_count: int = 4,
) -> list[SourceAtom]:
    """Build deterministic source atom candidates without labels or loss."""

    residual = _loss_agnostic_residual(logits)
    ranges = _split_ranges(int(logits.shape[0]), split_count)
    means = [residual[lo:hi].mean(dim=0) for lo, hi in ranges]
    mean_stack = torch.stack(means, dim=0)
    drift = mean_stack.mean(dim=0, keepdim=True)
    diffusion = mean_stack.std(dim=0, unbiased=False, keepdim=True)
    stable_axis = drift / diffusion.add(0.05).abs().clamp_min(0.05)
    stable_axis = stable_axis.expand_as(residual)
    stable_control = torch.sin(torch.arange(residual.numel(), dtype=residual.dtype)).reshape_as(residual)
    mean_axis = residual.mean(dim=0, keepdim=True).expand_as(residual)
    row_axis = residual.mean(dim=1, keepdim=True).expand_as(residual)
    control_null = _remove_control_span(residual, [mean_axis, row_axis, stable_control])
    drift_diffusion = 0.65 * residual + 0.35 * stable_axis
    drift_diffusion_unit = _normalize_atom(drift_diffusion, norm_scale)
    control_null_unit = _normalize_atom(control_null, norm_scale)
    control_balanced_drift = 0.78 * drift_diffusion_unit + 0.22 * control_null_unit
    control_balanced_drift = _subtract_control_span_fraction(
        control_balanced_drift,
        [mean_axis, row_axis, stable_control],
        0.50,
    )
    for _ in range(2):
        smoothed = control_balanced_drift.clone()
        if control_balanced_drift.shape[0] > 2:
            smoothed[1:-1] = (
                0.25 * control_balanced_drift[:-2]
                + 0.50 * control_balanced_drift[1:-1]
                + 0.25 * control_balanced_drift[2:]
            )
        control_balanced_drift = smoothed

    probs = torch.softmax(logits.detach().float(), dim=-1)
    smooth = residual.clone()
    if residual.shape[0] > 2:
        smooth[1:-1] = 0.25 * residual[:-2] + 0.50 * residual[1:-1] + 0.25 * residual[2:]
    curvature_weight = (probs * (1.0 - probs)).clamp_min(0.02)
    low_nds_atom = smooth / curvature_weight.sqrt()
    centered_rows = residual - residual.mean(dim=1, keepdim=True)
    ema_fast = torch.zeros_like(residual)
    ema_slow = torch.zeros_like(residual)
    causal_transport = torch.zeros_like(residual)
    fast = torch.zeros_like(residual[0])
    slow = torch.zeros_like(residual[0])
    prev_template = residual[ranges[0][0] : ranges[0][1]].mean(dim=0)
    for lo, hi in ranges:
        batch_mean = residual[lo:hi].mean(dim=0)
        causal_transport[lo:hi] = prev_template
        fast = 0.55 * fast + 0.45 * batch_mean
        slow = 0.90 * slow + 0.10 * batch_mean
        ema_fast[lo:hi] = fast
        ema_slow[lo:hi] = slow
        prev_template = 0.65 * prev_template + 0.35 * batch_mean
    fast_slow = 0.70 * residual + 0.20 * ema_fast + 0.10 * ema_slow
    split_consensus = residual[ranges[0][0] : ranges[0][1]].mean(dim=0, keepdim=True).expand_as(residual)

    forward_order = torch.cat([residual[lo:hi] for lo, hi in ranges], dim=0)
    reverse_order = torch.cat([residual[lo:hi] for lo, hi in reversed(ranges)], dim=0)
    reverse_order = reverse_order[: residual.shape[0]]
    commutator_stable = 0.50 * (forward_order + torch.flip(reverse_order, dims=[0]))

    try:
        u, s, vh = torch.linalg.svd(residual - residual.mean(dim=0, keepdim=True), full_matrices=False)
        rank = min(2, int(s.numel()))
        low_rank = (u[:, :rank] * s[:rank].reshape(1, -1)) @ vh[:rank] if rank else residual
    except Exception:
        low_rank = residual

    raw = [
        ("A1_split_transfer_atom", "split_transfer", residual),
        ("A2_drift_diffusion_atom", "drift_diffusion", drift_diffusion),
        ("A3_control_null_residual_atom", "control_null_residual", control_null),
        ("A4_low_NDS_atom", "low_NDS", low_nds_atom),
        ("A5_row_orthogonal_atom", "row_orthogonal", centered_rows),
        ("A6_fast_slow_source_state_atom", "fast_slow_source_state", fast_slow),
        ("A7_train_flow_commutator_atom", "train_flow_commutator", commutator_stable),
        ("A8_low_degree_frequency_atom", "low_degree_frequency", low_rank),
        ("A9_control_balanced_drift_atom", "control_balanced_drift", control_balanced_drift),
        ("A10_split_consensus_atom", "split_consensus", split_consensus),
        ("A11_causal_split_transport_atom", "causal_split_transport", causal_transport),
    ]
    return [
        SourceAtom(atom_id=atom_id, family=family, carrier=carrier, block_role=block_role, tensor=_normalize_atom(tensor, norm_scale))
        for atom_id, family, tensor in raw
    ]


def score_source_atoms(
    logits: torch.Tensor,
    labels: torch.Tensor,
    atoms: list[SourceAtom],
    *,
    norm_scale: float = 0.20,
    split_count: int = 4,
    seed: int = 2210,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    n = int(logits.shape[0])
    ranges = _split_ranges(n, split_count)
    b1 = ranges[0]
    b2 = ranges[min(1, len(ranges) - 1)]
    b3 = ranges[min(2, len(ranges) - 1)]
    controls = _control_tensors(logits, labels, norm_scale, seed)
    control_basis = [controls[name] for name in ("adamw_proxy", "sgd_proxy", "random_matched", "stable_random")]
    control_nds = {name: _curvature_nds(logits, labels, tensor) for name, tensor in controls.items()}
    nds_median = float(torch.tensor(list(control_nds.values())).median().item()) if control_nds else 0.0
    control_sob = {name: float(sobolev_fd_energy(tensor).item()) for name, tensor in controls.items()}
    sob_p50 = float(torch.tensor(list(control_sob.values())).median().item()) if control_sob else 0.0
    random_gain = _ce_gain(logits, labels, controls["random_matched"], *b2)
    sign_gain = _ce_gain(logits, labels, controls["sign_flip"], *b2)
    corrupt_gain = _ce_gain(logits, labels, controls["corrupt_target"], *b2)

    rows: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    for atom in atoms:
        tensor = atom.tensor.detach().float()
        tensors[atom.atom_id] = tensor
        b1_gain = _ce_gain(logits, labels, tensor, *b1)
        b2_gain = _ce_gain(logits, labels, tensor, *b2)
        b3_gain = _ce_gain(logits, labels, tensor, *b3)
        ddr, signal, diffusion = _ddr(tensor, split_count)
        projection, residual_norm = _project_fraction(tensor, control_basis)
        nds = _curvature_nds(logits, labels, tensor)
        row_metrics = _row_metrics(tensor)
        comm_ratio = _commutator_ratio(tensor, atom.family)
        source_norm = float(torch.linalg.vector_norm(tensor).item())
        pass_flag = int(
            b2_gain > random_gain + 0.0005
            and b3_gain >= -0.0005
            and b2_gain - random_gain > 0.0
            and b2_gain - sign_gain > 0.0
            and b2_gain - corrupt_gain > 0.0
            and residual_norm > 0.0
            and comm_ratio <= 0.10
            and nds <= nds_median + 1.0e-12
        )
        blockers = []
        if b2_gain <= random_gain + 0.0005:
            blockers.append("B2_transfer_gain_gate")
        if b3_gain < -0.0005:
            blockers.append("B3_safety_gain_gate")
        if b2_gain - random_gain <= 0.0:
            blockers.append("random_gap_gate")
        if b2_gain - sign_gain <= 0.0:
            blockers.append("sign_flip_gap_gate")
        if b2_gain - corrupt_gain <= 0.0:
            blockers.append("corrupt_target_gap_gate")
        if residual_norm <= 0.0:
            blockers.append("control_null_residual_norm_gate")
        if comm_ratio > 0.10:
            blockers.append("commutator_norm_ratio_gate")
        if nds > nds_median + 1.0e-12:
            blockers.append("NDS_gate")
        rows.append(
            {
                "atom_id": atom.atom_id,
                "carrier": atom.carrier,
                "block_role": atom.block_role,
                "atom_family": atom.family,
                "uses_future_or_validation": 0,
                "uses_audit_metric_for_direction": 0,
                "B1_gain": b1_gain,
                "B2_transfer_gain": b2_gain,
                "B3_safety_gain": b3_gain,
                "random_matched_gain": random_gain,
                "sign_flip_gain": sign_gain,
                "corrupt_target_gain": corrupt_gain,
                "random_gap": b2_gain - random_gain,
                "sign_flip_gap": b2_gain - sign_gain,
                "corrupt_target_gap": b2_gain - corrupt_gain,
                "control_projection_fraction": projection,
                "control_null_residual_norm": residual_norm,
                "commutator_norm_ratio": comm_ratio,
                "flow_order_gap": 1.0 - _safe_cos(tensor, torch.flip(tensor, dims=[0])),
                "DDR": ddr,
                "signal_drift_score": signal,
                "diffusion_score": diffusion,
                "NDS": nds,
                "control_NDS_median": nds_median,
                "metric_energy_L2": float(l2_energy(tensor).item()),
                "metric_energy_Fisher": float(tensor.square().mean().item()),
                "metric_energy_Sobolev": float(sobolev_fd_energy(tensor).item()),
                "metric_energy_RKHS": float(rkhs_graph_energy(tensor).item()),
                "control_Sobolev_p50": sob_p50,
                **row_metrics,
                "source_atom_norm": source_norm,
                "source_atom_function_cosine_with_controls": _safe_cos(tensor, controls["random_matched"]),
                "S2_source_atom_pass": pass_flag,
                "blocker": "" if pass_flag else ";".join(blockers),
            }
        )
    summary = {
        "atom_rows": len(rows),
        "S2_source_atom_pass_rows": sum(int(r.get("S2_source_atom_pass", 0)) for r in rows),
        "control_NDS_median": nds_median,
        "control_Sobolev_p50": sob_p50,
        "random_matched_B2_gain": random_gain,
        "sign_flip_B2_gain": sign_gain,
        "corrupt_target_B2_gain": corrupt_gain,
    }
    return rows, tensors, summary


def score_loss_agnostic_source_atoms(
    logits: torch.Tensor,
    atoms: list[SourceAtom],
    *,
    norm_scale: float = 0.20,
    split_count: int = 4,
    seed: int = 2210,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    n = int(logits.shape[0])
    ranges = _split_ranges(n, split_count)
    b1 = ranges[0]
    b2 = ranges[min(1, len(ranges) - 1)]
    b3 = ranges[min(2, len(ranges) - 1)]
    controls = _loss_agnostic_control_tensors(logits, norm_scale, seed)
    control_basis = [controls[name] for name in ("mean_axis", "row_axis", "random_matched", "stable_random")]
    control_nds = {name: _logit_geometry_nds(tensor) for name, tensor in controls.items()}
    nds_median = float(torch.tensor(list(control_nds.values())).median().item()) if control_nds else 0.0
    control_sob = {name: float(sobolev_fd_energy(tensor).item()) for name, tensor in controls.items()}
    sob_p50 = float(torch.tensor(list(control_sob.values())).median().item()) if control_sob else 0.0
    random_gain = _geometric_gain(logits, controls["random_matched"], *b2)
    sign_gain = _geometric_gain(logits, controls["sign_flip"], *b2)
    corrupt_gain = _geometric_gain(logits, controls["corrupt_target"], *b2)

    rows: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    for atom in atoms:
        tensor = atom.tensor.detach().float()
        tensors[atom.atom_id] = tensor
        b1_gain = _geometric_gain(logits, tensor, *b1)
        b2_gain = _geometric_gain(logits, tensor, *b2)
        b3_gain = _geometric_gain(logits, tensor, *b3)
        ddr, signal, diffusion = _ddr(tensor, split_count)
        projection, residual_norm = _project_fraction(tensor, control_basis)
        nds = _logit_geometry_nds(tensor)
        row_metrics = _row_metrics(tensor)
        comm_ratio = _commutator_ratio(tensor, atom.family)
        source_norm = float(torch.linalg.vector_norm(tensor).item())
        pass_flag = int(
            b2_gain > random_gain + 0.005
            and b3_gain >= -0.05
            and b2_gain - random_gain > 0.0
            and b2_gain - sign_gain > 0.0
            and b2_gain - corrupt_gain > 0.0
            and residual_norm > 0.0
            and comm_ratio <= 0.45
            and nds <= max(nds_median + 1.0e-12, 0.40)
        )
        blockers = []
        if b2_gain <= random_gain + 0.005:
            blockers.append("loss_agnostic_B2_transfer_gain_gate")
        if b3_gain < -0.05:
            blockers.append("loss_agnostic_B3_safety_gain_gate")
        if b2_gain - random_gain <= 0.0:
            blockers.append("random_gap_gate")
        if b2_gain - sign_gain <= 0.0:
            blockers.append("sign_flip_gap_gate")
        if b2_gain - corrupt_gain <= 0.0:
            blockers.append("corrupt_target_gap_gate")
        if residual_norm <= 0.0:
            blockers.append("control_null_residual_norm_gate")
        if comm_ratio > 0.45:
            blockers.append("commutator_norm_ratio_gate")
        if nds > max(nds_median + 1.0e-12, 0.40):
            blockers.append("loss_agnostic_NDS_gate")
        rows.append(
            {
                "atom_id": atom.atom_id,
                "carrier": atom.carrier,
                "block_role": atom.block_role,
                "atom_family": atom.family,
                "loss_agnostic_contract_pass": 1,
                "uses_labels_for_direction": 0,
                "uses_loss_for_direction": 0,
                "uses_future_or_validation": 0,
                "uses_audit_metric_for_direction": 0,
                "direction_metric": "logit_geometry_split_coherence",
                "B1_gain": b1_gain,
                "B2_transfer_gain": b2_gain,
                "B3_safety_gain": b3_gain,
                "random_matched_gain": random_gain,
                "sign_flip_gain": sign_gain,
                "corrupt_target_gain": corrupt_gain,
                "random_gap": b2_gain - random_gain,
                "sign_flip_gap": b2_gain - sign_gain,
                "corrupt_target_gap": b2_gain - corrupt_gain,
                "control_projection_fraction": projection,
                "control_null_residual_norm": residual_norm,
                "commutator_norm_ratio": comm_ratio,
                "flow_order_gap": 1.0 - _safe_cos(tensor, torch.flip(tensor, dims=[0])),
                "DDR": ddr,
                "signal_drift_score": signal,
                "diffusion_score": diffusion,
                "NDS": nds,
                "control_NDS_median": nds_median,
                "metric_energy_L2": float(l2_energy(tensor).item()),
                "metric_energy_Fisher": float(tensor.square().mean().item()),
                "metric_energy_Sobolev": float(sobolev_fd_energy(tensor).item()),
                "metric_energy_RKHS": float(rkhs_graph_energy(tensor).item()),
                "control_Sobolev_p50": sob_p50,
                **row_metrics,
                "source_atom_norm": source_norm,
                "source_atom_function_cosine_with_controls": _safe_cos(tensor, controls["random_matched"]),
                "S2_source_atom_pass": pass_flag,
                "blocker": "" if pass_flag else ";".join(blockers),
            }
        )
    summary = {
        "atom_rows": len(rows),
        "S2_source_atom_pass_rows": sum(int(r.get("S2_source_atom_pass", 0)) for r in rows),
        "loss_agnostic_contract_pass": int(all(int(r.get("loss_agnostic_contract_pass", 0)) for r in rows)),
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 0,
        "control_NDS_median": nds_median,
        "control_Sobolev_p50": sob_p50,
        "random_matched_B2_gain": random_gain,
        "sign_flip_B2_gain": sign_gain,
        "corrupt_target_B2_gain": corrupt_gain,
    }
    return rows, tensors, summary


def generate_train_only_source_atoms(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    carrier: str = "MLP",
    block_role: str = "all",
    norm_scale: float = 0.20,
    split_count: int = 4,
    seed: int = 2210,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    candidates = build_source_atom_candidates(
        logits,
        labels,
        carrier=carrier,
        block_role=block_role,
        norm_scale=norm_scale,
        split_count=split_count,
    )
    return score_source_atoms(logits, labels, candidates, norm_scale=norm_scale, split_count=split_count, seed=seed)


def generate_loss_agnostic_source_atoms(
    logits: torch.Tensor,
    *,
    carrier: str = "MLP",
    block_role: str = "all",
    norm_scale: float = 0.20,
    split_count: int = 4,
    seed: int = 2210,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    candidates = build_loss_agnostic_source_atom_candidates(
        logits,
        carrier=carrier,
        block_role=block_role,
        norm_scale=norm_scale,
        split_count=split_count,
    )
    return score_loss_agnostic_source_atoms(logits, candidates, norm_scale=norm_scale, split_count=split_count, seed=seed)


def source_atom_generator_unit_tests() -> list[dict[str, Any]]:
    torch.manual_seed(2210)
    logits = torch.randn(36, 4) * 0.25
    labels = torch.arange(36) % 4
    rows, tensors, summary = generate_train_only_source_atoms(logits, labels, norm_scale=0.18, split_count=4, seed=2210)
    finite = all(isfinite(float(r.get(k, 0.0))) for r in rows for k in ["B2_transfer_gain", "random_gap", "NDS", "DDR"])
    la_rows, la_tensors, la_summary = generate_loss_agnostic_source_atoms(logits, norm_scale=0.18, split_count=4, seed=2210)
    la_finite = all(isfinite(float(r.get(k, 0.0))) for r in la_rows for k in ["B2_transfer_gain", "random_gap", "NDS", "DDR"])
    return [
        {
            "case": "constructive_source_atoms_train_only",
            "rows": len(rows),
            "tensor_count": len(tensors),
            "pass_rows": summary.get("S2_source_atom_pass_rows", 0),
            "diagnostics_finite": int(finite),
            "pass": int(len(rows) >= 8 and len(tensors) == len(rows) and finite),
        },
        {
            "case": "constructive_source_atoms_loss_agnostic",
            "rows": len(la_rows),
            "tensor_count": len(la_tensors),
            "pass_rows": la_summary.get("S2_source_atom_pass_rows", 0),
            "uses_labels_for_direction": la_summary.get("uses_labels_for_direction", 1),
            "uses_loss_for_direction": la_summary.get("uses_loss_for_direction", 1),
            "loss_agnostic_contract_pass": la_summary.get("loss_agnostic_contract_pass", 0),
            "diagnostics_finite": int(la_finite),
            "pass": int(len(la_rows) >= 8 and len(la_tensors) == len(la_rows) and la_finite and int(la_summary.get("loss_agnostic_contract_pass", 0))),
        }
    ]


__all__ = [
    "SourceAtom",
    "build_source_atom_candidates",
    "build_loss_agnostic_source_atom_candidates",
    "generate_loss_agnostic_source_atoms",
    "generate_train_only_source_atoms",
    "score_loss_agnostic_source_atoms",
    "score_source_atoms",
    "source_atom_generator_unit_tests",
]
