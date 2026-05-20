"""Strict FC-PureKAN primitive basis modules for v12.4 screens.

The module intentionally keeps the architecture simple: every learnable tensor
belongs to an edge function.  Fixed normalizers, centers, and scales are stored
as buffers and never branch on dataset names or labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F


EPS = 1.0e-12


@dataclass(frozen=True)
class PrimitiveSpec:
    candidate_id: str
    basis_family: str
    basis_name: str
    k: int
    hidden_dim: int
    source: str
    local_support: int
    global_support: int
    uses_exp: int
    uses_sin_cos: int
    uses_division: int
    uses_gather_scatter: int = 0
    uses_dense_basis_tensor: int = 1
    diagnostic_only: int = 0
    basis_order: int = 1
    init_variant: str = "default"
    model_kind: str = "edge_kan"


def matched_hidden(param_budget: int, input_dim: int, output_dim: int, k: int) -> int:
    denom = max(1, int(k) * (int(input_dim) + int(output_dim)))
    return max(4, int(round(float(param_budget) / float(denom))))


def _basis_eval(z: torch.Tensor, basis_name: str, k: int, centers: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    k = int(k)
    if basis_name in {"relu_hinge", "rswaf_hinge"}:
        vals = [z]
        for idx in range(k - 1):
            vals.append(F.relu(z - centers[idx]) / math.sqrt(max(1, k - 1)))
        return torch.stack(vals, dim=-1)
    if basis_name in {"compact_rbf", "fastkan_rbf"}:
        width = scales[0].clamp_min(1.0e-3)
        r = (z.unsqueeze(-1) - centers[:k]) / width
        return torch.exp(-0.5 * r.square())
    if basis_name == "legendre":
        vals = [torch.ones_like(z)]
        if k > 1:
            vals.append(z)
        for n in range(2, k):
            vals.append(((2 * n - 1) * z * vals[-1] - (n - 1) * vals[-2]) / n)
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "chebyshev":
        vals = [torch.ones_like(z)]
        if k > 1:
            vals.append(z)
        for _ in range(2, k):
            vals.append(2.0 * z * vals[-1] - vals[-2])
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "fourier_lowfreq":
        vals = [z]
        freq = 1
        while len(vals) < k:
            vals.append(torch.sin(math.pi * freq * z))
            if len(vals) < k:
                vals.append(torch.cos(math.pi * freq * z))
            freq += 1
        return torch.stack(vals[:k], dim=-1) / math.sqrt(max(1, k))
    if basis_name == "ricker_wavelet":
        vals = []
        width = scales[0].clamp_min(1.0e-3)
        for idx in range(k):
            r = (z - centers[idx]) / width
            vals.append((1.0 - r.square()) * torch.exp(-0.5 * r.square()))
        return torch.stack(vals, dim=-1)
    if basis_name == "bspline_order1":
        width = scales[0].clamp_min(1.0e-3)
        r = (z.unsqueeze(-1) - centers[:k]).abs() / width
        return F.relu(1.0 - r)
    if basis_name == "bspline_relu_combo":
        vals = []
        width = scales[0].clamp_min(1.0e-3)
        local_count = max(1, k - 2)
        for idx in range(local_count):
            vals.append(F.relu(1.0 - (z - centers[idx]).abs() / width))
        vals.append(z)
        vals.append(F.relu(z))
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "poly2_relu_combo":
        vals = [z, z.square() - (1.0 / 3.0), F.relu(z)]
        while len(vals) < k:
            vals.append(vals[-1] * z)
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "rational_kat_lite":
        denom = 1.0 + 0.5 * z.abs() + 0.125 * z.square()
        vals = [z / denom]
        cur = z
        for idx in range(1, k):
            cur = cur * z
            vals.append(cur / (denom + 0.05 * idx))
        return torch.stack(vals[:k], dim=-1)
    raise ValueError(f"unknown primitive basis {basis_name}")


def _basis_channel(z: torch.Tensor, basis_name: str, k: int, centers: torch.Tensor, scales: torch.Tensor, idx: int) -> torch.Tensor:
    if basis_name in {"relu_hinge", "rswaf_hinge"}:
        if idx == 0:
            return z
        return F.relu(z - centers[idx - 1]) / math.sqrt(max(1, k - 1))
    if basis_name in {"compact_rbf", "fastkan_rbf"}:
        width = scales[0].clamp_min(1.0e-3)
        return torch.exp(-0.5 * ((z - centers[idx]) / width).square())
    if basis_name == "bspline_order1":
        width = scales[0].clamp_min(1.0e-3)
        return F.relu(1.0 - (z - centers[idx]).abs() / width)
    if basis_name == "bspline_relu_combo":
        local_count = max(1, k - 2)
        if idx < local_count:
            width = scales[0].clamp_min(1.0e-3)
            return F.relu(1.0 - (z - centers[idx]).abs() / width)
        if idx == local_count:
            return z
        return F.relu(z)
    if basis_name == "poly2_relu_combo":
        if idx == 0:
            return z
        if idx == 1:
            return z.square() - (1.0 / 3.0)
        if idx == 2:
            return F.relu(z)
        return z.pow(idx + 1)
    return _basis_eval(z, basis_name, k, centers, scales)[..., idx]


def _stream_mix(z: torch.Tensor, weight: torch.Tensor, basis_name: str, k: int, centers: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    out = None
    for idx in range(int(k)):
        part = _basis_channel(z, basis_name, k, centers, scales, idx) @ weight[..., idx]
        out = part if out is None else out + part
    assert out is not None
    return out


class PrimitiveKAN(nn.Module):
    """Two-layer strict edge-basis KAN with fixed basis centers/scales."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
        param_budget: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        hidden_dim = int(spec.hidden_dim) if int(spec.hidden_dim) > 0 else matched_hidden(param_budget, input_dim, output_dim, spec.k)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.k = int(spec.k)
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        mu = xs.mean(dim=0)
        std = xs.std(dim=0).clamp_min(1.0e-3)
        self.register_buffer("mu", mu)
        self.register_buffer("std", std)
        if spec.basis_name in {"relu_hinge", "rswaf_hinge"}:
            centers = torch.linspace(-1.0, 1.0, max(1, self.k - 1), device=device)
        else:
            centers = torch.linspace(-1.0, 1.0, self.k, device=device)
        self.register_buffer("centers", centers)
        self.register_buffer("scales", torch.tensor([max(0.2, 2.0 / max(1, self.k - 1))], device=device))
        gen = torch.Generator(device=device).manual_seed(int(seed))
        fan1 = math.sqrt(max(1, self.input_dim * self.k))
        fan2 = math.sqrt(max(1, self.hidden_dim * self.k))
        self.w1 = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim, self.k, device=device, generator=gen) / fan1)
        self.w2 = nn.Parameter(torch.randn(self.hidden_dim, self.output_dim, self.k, device=device, generator=gen) / fan2)
        if spec.init_variant == "fan_scale_repair":
            with torch.no_grad():
                self.w1.mul_(math.sqrt(max(1, self.input_dim)))
                self.w2.mul_(math.sqrt(max(1, self.hidden_dim)))
        if spec.init_variant == "identity_residual_scale":
            with torch.no_grad():
                self.w1.mul_(0.05)
                identity_channel = 1 if self.k > 1 else 0
                diag_cols = min(self.input_dim, self.hidden_dim)
                for idx in range(diag_cols):
                    self.w1[idx, idx, identity_channel] = math.sqrt(float(self.input_dim))
                self.w2.mul_(0.25)
        if spec.init_variant in {"signed_pair_linear", "signed_pair_random_linear"}:
            with torch.no_grad():
                self.w1.mul_(0.05)
                col = 0
                pair_limit = self.hidden_dim if spec.init_variant == "signed_pair_linear" else max(2, self.hidden_dim // 2)
                for start in range(0, self.input_dim - 1, 2):
                    if col + 1 >= pair_limit:
                        break
                    self.w1[start, col, 0] = math.sqrt(self.input_dim / 2.0)
                    self.w1[start + 1, col, 0] = math.sqrt(self.input_dim / 2.0)
                    col += 1
                    self.w1[start, col, 0] = math.sqrt(self.input_dim / 2.0)
                    self.w1[start + 1, col, 0] = -math.sqrt(self.input_dim / 2.0)
                    col += 1
                if spec.init_variant == "signed_pair_random_linear" and col < self.hidden_dim:
                    for idx in range(col, self.hidden_dim):
                        vec = torch.randn(self.input_dim, device=device, generator=gen)
                        vec = vec / vec.norm().clamp_min(1.0e-6)
                        self.w1[:, idx, 0] = vec * math.sqrt(float(self.input_dim))
                elif col < self.hidden_dim:
                    eye_cols = min(self.input_dim, self.hidden_dim - col)
                    for idx in range(eye_cols):
                        self.w1[idx, col + idx, 0] = math.sqrt(float(self.input_dim))

    @property
    def edge_param_count(self) -> int:
        return int(self.w1.numel() + self.w2.numel())

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh((x - self.mu) / self.std)

    def layer1_basis(self, x: torch.Tensor) -> torch.Tensor:
        return _basis_eval(self._norm_input(x), self.spec.basis_name, self.k, self.centers, self.scales)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        z = self._norm_input(x)
        if int(self.spec.uses_dense_basis_tensor) == 0:
            h = _stream_mix(z, self.w1, self.spec.basis_name, self.k, self.centers, self.scales) / math.sqrt(max(1, self.input_dim))
        else:
            b1 = _basis_eval(z, self.spec.basis_name, self.k, self.centers, self.scales)
            h = torch.einsum("bdk,dhk->bh", b1, self.w1) / math.sqrt(max(1, self.input_dim))
        return torch.tanh(h)

    def layer2_basis(self, h: torch.Tensor) -> torch.Tensor:
        return _basis_eval(h, self.spec.basis_name, self.k, self.centers, self.scales)

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        b2 = self.layer2_basis(h)
        return b2.reshape(int(x.shape[0]), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        if int(self.spec.uses_dense_basis_tensor) == 0:
            return _stream_mix(h, self.w2, self.spec.basis_name, self.k, self.centers, self.scales) / math.sqrt(max(1, self.hidden_dim))
        b2 = self.layer2_basis(h)
        return torch.einsum("bhk,hck->bc", b2, self.w2) / math.sqrt(max(1, self.hidden_dim))

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            return {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "basis_output_norm_p95": float(torch.quantile(feats.norm(dim=1), 0.95).item()),
            }


class QuadraticSketchKAN(nn.Module):
    """Quadratic sketch primitive with learnable basis readout.

    Fixed-projection variants are intentionally marked diagnostic-only in the
    v12.4 manifest. Trainable-projection variants are official-capable
    candidates and must pass the same gates as every other base candidate.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(spec.hidden_dim)
        self.k = int(spec.k)
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        self.register_buffer("mu", xs.mean(dim=0))
        self.register_buffer("std", xs.std(dim=0).clamp_min(1.0e-3))
        gen = torch.Generator(device=device).manual_seed(int(seed) + 7919)
        proj = torch.randn(self.input_dim, self.hidden_dim, device=device, generator=gen) / math.sqrt(max(1, self.input_dim))
        if spec.model_kind == "trainable_quadratic_sketch":
            self.proj = nn.Parameter(proj)
        else:
            self.register_buffer("proj", proj)
        self.readout = nn.Parameter(torch.randn(self.hidden_dim, self.k, self.output_dim, device=device, generator=gen) / math.sqrt(max(1, self.hidden_dim * self.k)))
        self.bias = nn.Parameter(torch.zeros(self.output_dim, device=device))

    @property
    def edge_param_count(self) -> int:
        proj_count = int(self.proj.numel()) if isinstance(self.proj, nn.Parameter) else 0
        return int(proj_count + self.readout.numel() + self.bias.numel())

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh((x - self.mu) / self.std)

    def _features_3d(self, x: torch.Tensor) -> torch.Tensor:
        z = self._norm_input(x) @ self.proj
        vals = [z]
        if self.k > 1:
            vals.append(z.square() - z.square().mean(dim=0, keepdim=True).detach())
        while len(vals) < self.k:
            vals.append(vals[-1] * z)
        return torch.stack(vals[: self.k], dim=2)

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._features_3d(x)
        return feats.reshape(int(x.shape[0]), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._features_3d(x)
        return torch.einsum("bhk,hkc->bc", feats, self.readout) + self.bias

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            return {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "basis_output_norm_p95": float(torch.quantile(feats.norm(dim=1), 0.95).item()),
            }

    def basis_functional_direction(self, mode: str) -> List[torch.Tensor]:
        with torch.no_grad():
            direction = -self.readout.detach().clone()
            if mode in {"basis_aware_snr_projected", "basis_aware_orthogonal"}:
                mask = torch.ones_like(direction)
                mask[:, 0, :] = 0.25
                direction = direction * mask
            if isinstance(self.proj, nn.Parameter):
                return [torch.zeros_like(self.proj), direction, -self.bias.detach().clone()]
            return [direction, -self.bias.detach().clone()]


class LiteGatedLegendreQuadraticKAN(nn.Module):
    """Low-cost Legendre direct readout plus quadratic sketch primitive.

    B30 is a cheaper follow-up to the B21/B23/B27 GatedHybrid family.  It keeps
    the basis-specific input geometry idea, but removes the second Legendre KAN
    layer: the Legendre branch reads out directly from normalized input basis
    features while the quadratic branch keeps a trainable sketch for interaction
    coverage.  Both input normalizers are fixed from unlabeled train-stream
    statistics and never branch on dataset names or labels.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(spec.hidden_dim)
        self.k = 4
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        self.register_buffer("mu", xs.mean(dim=0))
        self.register_buffer("std", xs.std(dim=0).clamp_min(1.0e-3))
        self.register_buffer("input_target_rms", torch.tensor([0.85], device=device))
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, self.k, device=device))
        self.register_buffer("scales", torch.tensor([max(0.2, 2.0 / max(1, self.k - 1))], device=device))
        gen = torch.Generator(device=device).manual_seed(int(seed) + 30477)
        self.leg_readout = nn.Parameter(
            torch.randn(self.input_dim, self.output_dim, self.k, device=device, generator=gen)
            / math.sqrt(max(1, self.input_dim * self.k))
        )
        self.quad_proj = nn.Parameter(
            torch.randn(self.input_dim, self.hidden_dim, device=device, generator=gen)
            / math.sqrt(max(1, self.input_dim))
        )
        self.quad_readout = nn.Parameter(
            torch.randn(self.hidden_dim, 2, self.output_dim, device=device, generator=gen)
            / math.sqrt(max(1, self.hidden_dim * 2))
        )
        branch_init = [1.0, 0.35]
        if "quadboost" in str(spec.init_variant):
            branch_init = [0.75, 0.75]
        if "midboost" in str(spec.init_variant):
            branch_init = [1.0, 0.55]
        self.branch_scale = nn.Parameter(torch.tensor(branch_init, device=device))
        gain_init = 1.0
        if "temp050" in str(spec.init_variant):
            gain_init = 0.50
        elif "temp075" in str(spec.init_variant):
            gain_init = 0.75
        self.logit_gain = nn.Parameter(torch.tensor([gain_init], device=device))
        self.bias = nn.Parameter(torch.zeros(self.output_dim, device=device))
        self.register_buffer("quad_feature_std", torch.ones(self.hidden_dim, device=device))
        self._calibrate_quadratic_feature_norm(xs)

    @property
    def edge_param_count(self) -> int:
        return int(
            self.leg_readout.numel()
            + self.quad_proj.numel()
            + self.quad_readout.numel()
            + self.branch_scale.numel()
            + self.logit_gain.numel()
            + self.bias.numel()
        )

    def _rms_normalized_input(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mu) / self.std
        rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
        return z / rms * self.input_target_rms

    def _legendre_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self._rms_normalized_input(x))

    def _quadratic_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._rms_normalized_input(x).clamp(-3.0, 3.0)

    def _calibrate_quadratic_feature_norm(self, xs: torch.Tensor) -> None:
        with torch.no_grad():
            sample = xs[: min(2048, int(xs.shape[0]))]
            z = self._quadratic_input(sample) @ self.quad_proj
            self.quad_feature_std.copy_(z.float().std(dim=0).clamp_min(1.0e-4))

    def _legendre_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = _basis_eval(self._legendre_input(x), "legendre", self.k, self.centers, self.scales)
        logits = torch.einsum("bdk,dck->bc", b, self.leg_readout) / math.sqrt(max(1, self.input_dim))
        return logits, b.reshape(int(x.shape[0]), -1)

    def _quadratic_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = (self._quadratic_input(x) @ self.quad_proj) / self.quad_feature_std.clamp_min(1.0e-4)
        centered_square = z.square() - z.square().mean(dim=0, keepdim=True).detach()
        feats = torch.stack([z, centered_square], dim=2)
        logits = torch.einsum("bhk,hkc->bc", feats, self.quad_readout)
        return logits, feats.reshape(int(x.shape[0]), -1)

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        _, leg_feats = self._legendre_logits_and_features(x)
        _, quad_feats = self._quadratic_logits_and_features(x)
        return torch.cat([leg_feats, quad_feats], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leg_logits, _ = self._legendre_logits_and_features(x)
        quad_logits, _ = self._quadratic_logits_and_features(x)
        logits = self.branch_scale[0] * leg_logits + self.branch_scale[1] * quad_logits + self.bias
        return self.logit_gain.clamp(0.25, 4.0) * logits

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            return {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "basis_output_norm_p95": float(torch.quantile(feats.norm(dim=1), 0.95).item()),
            }

    def basis_functional_direction(self, mode: str) -> List[torch.Tensor]:
        with torch.no_grad():
            leg = -self.leg_readout.detach().clone()
            quad_proj = torch.zeros_like(self.quad_proj)
            quad_readout = -self.quad_readout.detach().clone()
            if mode in {"basis_aware_snr_projected", "basis_aware_orthogonal"}:
                leg[..., 0] *= 0.25
                quad_readout[:, 0, :] *= 0.25
            return [
                leg,
                quad_proj,
                quad_readout,
                torch.zeros_like(self.branch_scale),
                torch.zeros_like(self.logit_gain),
                -self.bias.detach().clone(),
            ]


class SimpleFastTaskGeometryKAN(nn.Module):
    """Compact hinge/direct basis plus minimal quadratic interaction.

    This is the v12.5.2 R1/R6 fallback when full GatedLQ fusion remains too
    slow.  It keeps every learnable tensor in an edge-basis or interaction
    readout role, uses only fixed train-stream normalization, and avoids the
    two-layer Legendre path that dominated the B47 forward bottleneck.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(spec.hidden_dim)
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        self.register_buffer("mu", xs.mean(dim=0))
        self.register_buffer("std", xs.std(dim=0).clamp_min(1.0e-3))
        variant = str(spec.init_variant)
        if "twohinge" in variant:
            hinge_centers = [0.25, 0.75]
        elif "hinge050" in variant:
            hinge_centers = [0.50]
        else:
            hinge_centers = [0.25]
        self.hinge_center0 = float(hinge_centers[0])
        self.register_buffer("hinge_centers", torch.tensor(hinge_centers, device=device))
        z_stats = ((xs - self.mu) / self.std).clamp(-3.0, 3.0)
        self.register_buffer("z2_mean", z_stats.square().mean(dim=0))
        self.register_buffer("z3_mean", z_stats.pow(3).mean(dim=0))
        self.register_buffer("zabs_mean", z_stats.abs().mean(dim=0))
        self.register_buffer("zabs_scalar_mean", z_stats.abs().mean(dim=1, keepdim=True).mean(dim=0))
        self.register_buffer("zmean_scalar_mean", z_stats.mean(dim=1, keepdim=True).mean(dim=0))
        self.register_buffer("quad_feature_std", torch.ones(self.hidden_dim, device=device))
        gen = torch.Generator(device=device).manual_seed(int(seed) + 48127)
        self.direct_absmix_square_enabled = "absmixsq" in variant
        self.direct_absmix_square_scale = 0.25
        if "absmixsq010" in variant:
            self.direct_absmix_square_scale = 0.10
        if "absmixsq050" in variant:
            self.direct_absmix_square_scale = 0.50
        self.direct_abs_square_enabled = ("absquad" in variant) and not self.direct_absmix_square_enabled
        self.direct_cubic_enabled = ("cubicdiag" in variant) and not (self.direct_abs_square_enabled or self.direct_absmix_square_enabled)
        self.direct_square_enabled = ("sqdiag" in variant) and not (self.direct_abs_square_enabled or self.direct_absmix_square_enabled or self.direct_cubic_enabled)
        self.direct_abs_enabled = ("absdiag" in variant) and not (self.direct_abs_square_enabled or self.direct_absmix_square_enabled or self.direct_cubic_enabled)
        self.direct_normabs_enabled = "normabs" in variant
        self.direct_meanstat_enabled = "meanstat" in variant
        self.direct_groupabs_count = 8 if "groupabs8" in variant else (4 if "groupabs4" in variant else 0)
        if self.direct_groupabs_count:
            group_size = max(1, self.input_dim // self.direct_groupabs_count)
            group_means = []
            for group_idx in range(self.direct_groupabs_count):
                start = group_idx * group_size
                end = self.input_dim if group_idx == self.direct_groupabs_count - 1 else min(self.input_dim, (group_idx + 1) * group_size)
                group_means.append(z_stats[:, start:end].abs().mean(dim=1, keepdim=True).mean(dim=0))
            self.register_buffer("zgroupabs_mean", torch.cat(group_means, dim=0))
        else:
            self.register_buffer("zgroupabs_mean", torch.empty(0, device=device))
        self.quad_batch_rms_enabled = "rmsq" in variant
        self.quad_tanh_enabled = "boundq" in variant
        direct_feature_multiplier = 1 + 2 * int(self.hinge_centers.numel())
        if self.direct_abs_square_enabled:
            direct_feature_multiplier += 2
        elif self.direct_square_enabled or self.direct_abs_enabled or self.direct_absmix_square_enabled or self.direct_cubic_enabled:
            direct_feature_multiplier += 1
        direct_feature_count = direct_feature_multiplier * self.input_dim
        if self.direct_normabs_enabled:
            direct_feature_count += 1
        if self.direct_meanstat_enabled:
            direct_feature_count += 1
        direct_feature_count += int(self.direct_groupabs_count)
        scale_direct = math.sqrt(max(1, self.input_dim))
        direct_readout = torch.randn(direct_feature_count, self.output_dim, device=device, generator=gen) / scale_direct
        identity_init_scale = 1.0
        if "identityamp150" in variant:
            identity_init_scale = 1.50
        if identity_init_scale != 1.0:
            direct_readout[: self.input_dim, :].mul_(identity_init_scale)
        hinge_init_scale = 1.0
        if "hingeamp075" in variant:
            hinge_init_scale = 0.75
        if "hingeamp050" in variant:
            hinge_init_scale = 0.50
        if "hingeamp025" in variant:
            hinge_init_scale = 0.25
        if hinge_init_scale != 1.0:
            hinge_start = self.input_dim
            hinge_end = self.input_dim + 2 * int(self.hinge_centers.numel()) * self.input_dim
            direct_readout[hinge_start:hinge_end, :].mul_(hinge_init_scale)
        if self.direct_abs_square_enabled:
            absquad_abs_scale = 0.50
            absquad_square_scale = 0.25
            if "absquad050050" in variant:
                absquad_abs_scale = 0.50
                absquad_square_scale = 0.50
            if "absquad075025" in variant:
                absquad_abs_scale = 0.75
                absquad_square_scale = 0.25
            if "absquad050010" in variant:
                absquad_abs_scale = 0.50
                absquad_square_scale = 0.10
            diag_start = (1 + 2 * int(self.hinge_centers.numel())) * self.input_dim
            direct_readout[diag_start : diag_start + self.input_dim, :].mul_(absquad_abs_scale)
            direct_readout[diag_start + self.input_dim : diag_start + 2 * self.input_dim, :].mul_(absquad_square_scale)
        elif self.direct_absmix_square_enabled:
            absmix_init_scale = 0.50
            if "absmixsq025init025" in variant:
                absmix_init_scale = 0.25
            if "absmixsq025init075" in variant:
                absmix_init_scale = 0.75
            diag_start = (1 + 2 * int(self.hinge_centers.numel())) * self.input_dim
            direct_readout[diag_start : diag_start + self.input_dim, :].mul_(absmix_init_scale)
        elif self.direct_square_enabled or self.direct_abs_enabled or self.direct_cubic_enabled:
            square_init_scale = 0.25 if self.direct_abs_enabled else 1.0
            if "sqdiag025" in variant:
                square_init_scale = 0.25
            if "sqdiag010" in variant:
                square_init_scale = 0.10
            if "cubicdiag025" in variant:
                square_init_scale = 0.25
            if "cubicdiag010" in variant:
                square_init_scale = 0.10
            if "cubicdiag050" in variant:
                square_init_scale = 0.50
            if "absdiag050" in variant:
                square_init_scale = 0.50
            if "absdiag075" in variant:
                square_init_scale = 0.75
            if "absdiag100" in variant:
                square_init_scale = 1.00
            if "absdiag025" in variant:
                square_init_scale = 0.25
            if "absdiag010" in variant:
                square_init_scale = 0.10
            diag_start = (1 + 2 * int(self.hinge_centers.numel())) * self.input_dim
            direct_readout[diag_start : diag_start + self.input_dim, :].mul_(square_init_scale)
        scalar_start = direct_feature_multiplier * self.input_dim
        scalar_row = scalar_start
        if self.direct_normabs_enabled:
            normabs_init_scale = 0.25
            if "normabs010" in variant:
                normabs_init_scale = 0.10
            if "normabs025" in variant:
                normabs_init_scale = 0.25
            if "normabs050" in variant:
                normabs_init_scale = 0.50
            if "normabs075" in variant:
                normabs_init_scale = 0.75
            if "normabs100" in variant:
                normabs_init_scale = 1.00
            direct_readout[scalar_row : scalar_row + 1, :].mul_(normabs_init_scale)
            scalar_row += 1
        if self.direct_meanstat_enabled:
            meanstat_init_scale = 0.25
            if "meanstat010" in variant:
                meanstat_init_scale = 0.10
            if "meanstat025" in variant:
                meanstat_init_scale = 0.25
            if "meanstat050" in variant:
                meanstat_init_scale = 0.50
            if "meanstat075" in variant:
                meanstat_init_scale = 0.75
            if "meanstat100" in variant:
                meanstat_init_scale = 1.00
            direct_readout[scalar_row : scalar_row + 1, :].mul_(meanstat_init_scale)
            scalar_row += 1
        if self.direct_groupabs_count:
            groupabs_init_scale = 0.25
            if "groupabs010" in variant:
                groupabs_init_scale = 0.10
            if "groupabs025" in variant:
                groupabs_init_scale = 0.25
            if "groupabs050" in variant:
                groupabs_init_scale = 0.50
            if "groupabs075" in variant:
                groupabs_init_scale = 0.75
            if "groupabs100" in variant:
                groupabs_init_scale = 1.00
            direct_readout[scalar_row : scalar_row + int(self.direct_groupabs_count), :].mul_(groupabs_init_scale)
        self.direct_readout = nn.Parameter(direct_readout)
        quad_proj = torch.randn(self.input_dim, self.hidden_dim, device=device, generator=gen) / math.sqrt(max(1, self.input_dim))
        if "orthop" in variant:
            if self.input_dim >= self.hidden_dim:
                frame = torch.randn(self.input_dim, self.hidden_dim, device=device, generator=gen)
                q_frame, r_frame = torch.linalg.qr(frame, mode="reduced")
                signs = torch.sign(torch.diag(r_frame))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                quad_proj = (q_frame * signs.view(1, -1)).contiguous()
            else:
                frame = torch.randn(self.hidden_dim, self.input_dim, device=device, generator=gen)
                q_frame, r_frame = torch.linalg.qr(frame, mode="reduced")
                signs = torch.sign(torch.diag(r_frame))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                quad_proj = (q_frame * signs.view(1, -1)).transpose(0, 1).contiguous()
        if "signedpairtail" in variant:
            quad_proj.zero_()
            norm = 1.0 / math.sqrt(2.0)
            pair_cols = max(2, self.hidden_dim // 2)
            col = 0
            for start in range(0, max(1, self.input_dim - 1), 2):
                if col >= min(pair_cols, self.hidden_dim):
                    break
                a = start % self.input_dim
                b = (start + 1) % self.input_dim
                quad_proj[a, col] = norm
                quad_proj[b, col] = norm
                col += 1
                if col >= min(pair_cols, self.hidden_dim):
                    break
                quad_proj[a, col] = norm
                quad_proj[b, col] = -norm
                col += 1
            while col < self.hidden_dim:
                vec = torch.randn(self.input_dim, device=device, generator=gen)
                vec = vec / vec.norm().clamp_min(1.0e-6)
                quad_proj[:, col] = vec
                col += 1
        elif "signedpairlite" in variant:
            quad_proj.zero_()
            norm = 1.0 / math.sqrt(2.0)
            col = 0
            for start in range(0, max(1, self.input_dim - 1), 2):
                if col >= self.hidden_dim:
                    break
                a = start % self.input_dim
                b = (start + 1) % self.input_dim
                quad_proj[a, col] = norm
                quad_proj[b, col] = norm
                col += 1
                if col >= self.hidden_dim:
                    break
                quad_proj[a, col] = norm
                quad_proj[b, col] = -norm
                col += 1
            while col < self.hidden_dim:
                idx = col % self.input_dim
                quad_proj[idx, col] = 1.0
                col += 1
        if "fixedp" in variant:
            self.register_buffer("quad_proj", quad_proj)
        else:
            self.quad_proj = nn.Parameter(quad_proj)
        self.quad_readout = nn.Parameter(torch.randn(2 * self.hidden_dim, self.output_dim, device=device, generator=gen) / math.sqrt(max(1, self.hidden_dim * 2)))
        branch_init = [1.0, 0.30]
        if "quad050" in variant:
            branch_init = [1.0, 0.50]
        if "quad040" in variant:
            branch_init = [1.0, 0.40]
        if "quad020" in variant:
            branch_init = [1.0, 0.20]
        if "quad010" in variant:
            branch_init = [1.0, 0.10]
        if "identitytailquad060" in variant:
            branch_init = [1.35, 0.60]
        elif "identitytail130quad040" in variant:
            branch_init = [1.30, 0.40]
        elif "identitytail125quad040" in variant:
            branch_init = [1.25, 0.40]
        elif "identitytail145quad040" in variant:
            branch_init = [1.45, 0.40]
        elif "identitytail140quad040" in variant:
            branch_init = [1.40, 0.40]
        elif "identitytail150quad040" in variant:
            branch_init = [1.50, 0.40]
        elif "identitytailquad040" in variant:
            branch_init = [1.35, 0.40]
        elif "identitytailquad030" in variant:
            branch_init = [1.35, 0.30]
        elif "identitytailquad020" in variant:
            branch_init = [1.35, 0.20]
        elif "identitytailquad050" in variant:
            branch_init = [1.35, 0.50]
        elif "identitytail" in variant:
            branch_init = [1.35, 0.10]
        self.class_branch_scale_enabled = "classbranch" in variant
        if self.class_branch_scale_enabled:
            branch_tensor = torch.stack(
                [
                    torch.full((self.output_dim,), float(branch_init[0]), device=device),
                    torch.full((self.output_dim,), float(branch_init[1]), device=device),
                ]
            )
        else:
            branch_tensor = torch.tensor(branch_init, device=device)
        if "fixedbranch" in variant:
            self.register_buffer("branch_scale", branch_tensor)
        else:
            self.branch_scale = nn.Parameter(branch_tensor)
        gain_init = 0.75
        if "temp050" in variant:
            gain_init = 0.50
        if "temp065" in variant:
            gain_init = 0.65
        if "temp100" in variant:
            gain_init = 1.00
        self.class_logit_gain_enabled = "classgain" in variant
        self.logit_norm_enabled = "logitnorm" in variant
        norm_target = 1.50
        if "logitnorm100" in variant:
            norm_target = 1.00
        if "logitnorm150" in variant:
            norm_target = 1.50
        if "logitnorm200" in variant:
            norm_target = 2.00
        self.register_buffer("logit_norm_target", torch.tensor([norm_target], device=device))
        if self.class_logit_gain_enabled:
            gain_tensor = torch.full((self.output_dim,), float(gain_init), device=device)
        else:
            gain_tensor = torch.tensor([gain_init], device=device)
        if "fixedgain" in variant:
            self.register_buffer("logit_gain", gain_tensor)
        else:
            self.logit_gain = nn.Parameter(gain_tensor)
        self.bias = nn.Parameter(torch.zeros(self.output_dim, device=device))
        self._calibrate_quadratic_feature_norm(xs)

    @property
    def edge_param_count(self) -> int:
        return int(
            self.direct_readout.numel()
            + (self.quad_proj.numel() if isinstance(self.quad_proj, nn.Parameter) and self.quad_proj.requires_grad else 0)
            + self.quad_readout.numel()
            + (self.branch_scale.numel() if isinstance(self.branch_scale, nn.Parameter) and self.branch_scale.requires_grad else 0)
            + (self.logit_gain.numel() if isinstance(self.logit_gain, nn.Parameter) and self.logit_gain.requires_grad else 0)
            + self.bias.numel()
        )

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mu) / self.std).clamp(-3.0, 3.0)

    def _hinge_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self._input(x)
        centers = self.hinge_centers.view(1, 1, -1)
        zu = z.unsqueeze(-1)
        pos = F.relu(zu - centers).flatten(1)
        neg = F.relu(-zu - centers).flatten(1)
        return z, pos, neg

    def _calibrate_quadratic_feature_norm(self, xs: torch.Tensor) -> None:
        with torch.no_grad():
            z = self._input(xs[: min(2048, int(xs.shape[0]))]) @ self.quad_proj
            self.quad_feature_std.copy_(z.float().std(dim=0).clamp_min(1.0e-4))

    def _direct_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, pos, neg = self._hinge_features(x)
        parts = [z, pos, neg]
        if self.direct_abs_square_enabled:
            parts.append(z.abs() - self.zabs_mean)
            parts.append(z.square() - self.z2_mean)
        elif self.direct_absmix_square_enabled:
            parts.append((z.abs() - self.zabs_mean) + float(self.direct_absmix_square_scale) * (z.square() - self.z2_mean))
        elif self.direct_square_enabled:
            parts.append(z.square() - self.z2_mean)
        elif self.direct_cubic_enabled:
            parts.append(z.pow(3) - self.z3_mean)
        elif self.direct_abs_enabled:
            parts.append(z.abs() - self.zabs_mean)
        if self.direct_normabs_enabled:
            normabs = (z.abs().mean(dim=1, keepdim=True) - self.zabs_scalar_mean) * math.sqrt(max(1, self.input_dim))
            parts.append(normabs)
        if self.direct_meanstat_enabled:
            meanstat = (z.mean(dim=1, keepdim=True) - self.zmean_scalar_mean) * math.sqrt(max(1, self.input_dim))
            parts.append(meanstat)
        if self.direct_groupabs_count:
            group_size = max(1, self.input_dim // int(self.direct_groupabs_count))
            group_feats = []
            for group_idx in range(int(self.direct_groupabs_count)):
                start = group_idx * group_size
                end = self.input_dim if group_idx == int(self.direct_groupabs_count) - 1 else min(self.input_dim, (group_idx + 1) * group_size)
                group_abs = (z[:, start:end].abs().mean(dim=1, keepdim=True) - self.zgroupabs_mean[group_idx : group_idx + 1]) * math.sqrt(max(1, end - start))
                group_feats.append(group_abs)
            parts.append(torch.cat(group_feats, dim=1))
        feats = torch.cat(parts, dim=1)
        logits = feats @ self.direct_readout / math.sqrt(max(1, self.input_dim))
        return logits, feats

    def _quadratic_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = (self._input(x) @ self.quad_proj) / self.quad_feature_std.clamp_min(1.0e-4)
        if self.quad_batch_rms_enabled:
            q = q / q.square().mean(dim=0, keepdim=True).sqrt().detach().clamp_min(1.0e-4)
        if self.quad_tanh_enabled:
            q = torch.tanh(q)
        q2 = q.square() - q.square().mean(dim=0, keepdim=True).detach()
        feats = torch.cat([q, q2], dim=1)
        logits = feats @ self.quad_readout
        return logits, feats

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        _, direct = self._direct_logits_and_features(x)
        _, quad = self._quadratic_logits_and_features(x)
        return torch.cat([direct, quad], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direct_logits, _ = self._direct_logits_and_features(x)
        quad_logits, _ = self._quadratic_logits_and_features(x)
        if self.class_branch_scale_enabled:
            logits = self.branch_scale[0].view(1, -1) * direct_logits + self.branch_scale[1].view(1, -1) * quad_logits + self.bias
        else:
            logits = self.branch_scale[0] * direct_logits + self.branch_scale[1] * quad_logits + self.bias
        if self.logit_norm_enabled:
            rms = logits.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-4)
            logits = self.logit_norm_target * logits / rms
        return self.logit_gain.clamp(0.25, 4.0) * logits

    def manual_kernel_available(self) -> bool:
        return True

    def manual_ce_forward_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        with torch.no_grad():
            z = self._input(x)
            centers = self.hinge_centers.view(1, 1, -1)
            zu = z.unsqueeze(-1)
            pos = F.relu(zu - centers).flatten(1)
            neg = F.relu(-zu - centers).flatten(1)
            direct_parts = [z, pos, neg]
            if self.direct_abs_square_enabled:
                direct_parts.append(z.abs() - self.zabs_mean)
                direct_parts.append(z.square() - self.z2_mean)
            elif self.direct_absmix_square_enabled:
                direct_parts.append((z.abs() - self.zabs_mean) + float(self.direct_absmix_square_scale) * (z.square() - self.z2_mean))
            elif self.direct_square_enabled:
                direct_parts.append(z.square() - self.z2_mean)
            elif self.direct_cubic_enabled:
                direct_parts.append(z.pow(3) - self.z3_mean)
            elif self.direct_abs_enabled:
                direct_parts.append(z.abs() - self.zabs_mean)
            if self.direct_normabs_enabled:
                normabs = (z.abs().mean(dim=1, keepdim=True) - self.zabs_scalar_mean) * math.sqrt(max(1, self.input_dim))
                direct_parts.append(normabs)
            if self.direct_meanstat_enabled:
                meanstat = (z.mean(dim=1, keepdim=True) - self.zmean_scalar_mean) * math.sqrt(max(1, self.input_dim))
                direct_parts.append(meanstat)
            if self.direct_groupabs_count:
                group_size = max(1, self.input_dim // int(self.direct_groupabs_count))
                group_feats = []
                for group_idx in range(int(self.direct_groupabs_count)):
                    start = group_idx * group_size
                    end = self.input_dim if group_idx == int(self.direct_groupabs_count) - 1 else min(self.input_dim, (group_idx + 1) * group_size)
                    group_abs = (z[:, start:end].abs().mean(dim=1, keepdim=True) - self.zgroupabs_mean[group_idx : group_idx + 1]) * math.sqrt(max(1, end - start))
                    group_feats.append(group_abs)
                direct_parts.append(torch.cat(group_feats, dim=1))
            direct_feats = torch.cat(direct_parts, dim=1)
            direct_logits = direct_feats @ self.direct_readout / math.sqrt(max(1, self.input_dim))
            q = (z @ self.quad_proj) / self.quad_feature_std.clamp_min(1.0e-4)
            if self.quad_batch_rms_enabled:
                q = q / q.square().mean(dim=0, keepdim=True).sqrt().detach().clamp_min(1.0e-4)
            if self.quad_tanh_enabled:
                q = torch.tanh(q)
            q2 = q.square() - q.square().mean(dim=0, keepdim=True)
            quad_feats = torch.cat([q, q2], dim=1)
            quad_logits = quad_feats @ self.quad_readout
            if self.class_branch_scale_enabled:
                logits_unscaled = self.branch_scale[0].view(1, -1) * direct_logits + self.branch_scale[1].view(1, -1) * quad_logits + self.bias
            else:
                logits_unscaled = self.branch_scale[0] * direct_logits + self.branch_scale[1] * quad_logits + self.bias
            logits_pre_gain = logits_unscaled
            if self.logit_norm_enabled:
                rms = logits_unscaled.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-4)
                logits_pre_gain = self.logit_norm_target * logits_unscaled / rms
            logits = self.logit_gain.clamp(0.25, 4.0) * logits_pre_gain
            return logits, (z, direct_feats, q, quad_feats, direct_logits, quad_logits, logits_unscaled, logits_pre_gain)

    def manual_ce_backward_from_cache(self, logits: torch.Tensor, cache: tuple[torch.Tensor, ...], y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z, direct_feats, q, quad_feats, direct_logits, quad_logits, logits_unscaled, logits_pre_gain = cache
            probs = torch.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y)
            grad_output = probs
            grad_output[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
            grad_output = grad_output / float(max(1, int(y.numel())))
            gain = self.logit_gain.clamp(0.25, 4.0)
            g_pre_gain = grad_output * gain
            if self.logit_norm_enabled:
                rms = logits_unscaled.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-4)
                dot = (g_pre_gain * logits_unscaled).sum(dim=1, keepdim=True)
                denom = float(max(1, self.output_dim)) * rms.square()
                g_logits = self.logit_norm_target * (g_pre_gain - logits_unscaled * dot / denom) / rms
            else:
                g_logits = g_pre_gain
            gain_mask = ((self.logit_gain >= 0.25) & (self.logit_gain <= 4.0)).to(dtype=grad_output.dtype)
            if isinstance(self.logit_gain, nn.Parameter) and self.logit_gain.requires_grad:
                gain_grad = grad_output * logits_pre_gain
                if self.logit_gain.numel() == self.output_dim:
                    self.logit_gain.grad = gain_grad.sum(dim=0).view_as(self.logit_gain) * gain_mask
                else:
                    self.logit_gain.grad = gain_grad.sum().view_as(self.logit_gain) * gain_mask
            if isinstance(self.branch_scale, nn.Parameter) and self.branch_scale.requires_grad:
                if self.class_branch_scale_enabled:
                    self.branch_scale.grad = torch.stack(
                        [
                            (g_logits * direct_logits).sum(dim=0),
                            (g_logits * quad_logits).sum(dim=0),
                        ]
                    ).view_as(self.branch_scale)
                else:
                    self.branch_scale.grad = torch.stack(
                        [
                            (g_logits * direct_logits).sum(),
                            (g_logits * quad_logits).sum(),
                        ]
                    ).view_as(self.branch_scale)
            self.bias.grad = g_logits.sum(dim=0)

            if self.class_branch_scale_enabled:
                grad_direct_logits = g_logits * self.branch_scale[0].view(1, -1)
            else:
                grad_direct_logits = g_logits * self.branch_scale[0]
            self.direct_readout.grad = direct_feats.transpose(0, 1) @ grad_direct_logits / math.sqrt(max(1, self.input_dim))

            if self.class_branch_scale_enabled:
                grad_quad_logits = g_logits * self.branch_scale[1].view(1, -1)
            else:
                grad_quad_logits = g_logits * self.branch_scale[1]
            self.quad_readout.grad = quad_feats.transpose(0, 1) @ grad_quad_logits
            grad_quad_feats = grad_quad_logits @ self.quad_readout.transpose(0, 1)
            grad_q = grad_quad_feats[:, : self.hidden_dim] + 2.0 * q * grad_quad_feats[:, self.hidden_dim :]
            if self.quad_tanh_enabled:
                grad_q = grad_q * (1.0 - q.square())
            if self.quad_batch_rms_enabled:
                q_pre = (z @ self.quad_proj) / self.quad_feature_std.clamp_min(1.0e-4)
                q_rms = q_pre.square().mean(dim=0, keepdim=True).sqrt().detach().clamp_min(1.0e-4)
                grad_q = grad_q / q_rms
            grad_q = grad_q / self.quad_feature_std.clamp_min(1.0e-4)
            if isinstance(self.quad_proj, nn.Parameter) and self.quad_proj.requires_grad:
                self.quad_proj.grad = z.transpose(0, 1) @ grad_q
            return loss.detach()

    def manual_gradient_audit(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        params = [p for p in self.parameters() if p.requires_grad]
        self.zero_grad(set_to_none=True)
        logits, cache = self.manual_ce_forward_cache(x)
        self.manual_ce_backward_from_cache(logits, cache, y)
        manual_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        ref_logits = self(x)
        F.cross_entropy(ref_logits, y).backward()
        ref_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        relerrs: List[float] = []
        coses: List[float] = []
        for gm, gr in zip(manual_grads, ref_grads):
            denom = gr.norm().clamp_min(1.0e-8)
            relerrs.append(float((gm - gr).norm().div(denom).detach().item()))
            if gm.norm().item() > 0.0 and gr.norm().item() > 0.0:
                coses.append(float(F.cosine_similarity(gm.flatten(), gr.flatten(), dim=0).detach().item()))
        return {
            "manual_forward_available": 1.0,
            "manual_backward_available": 1.0,
            "grad_relerr_max": max(relerrs) if relerrs else float("inf"),
            "grad_cos_min": min(coses) if coses else -1.0,
            "output_max_abs_error": float((logits - ref_logits).abs().max().detach().item()),
        }

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            return {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "basis_output_norm_p95": float(torch.quantile(feats.norm(dim=1), 0.95).item()),
            }

    def basis_functional_direction(self, mode: str) -> List[torch.Tensor]:
        with torch.no_grad():
            direct = -self.direct_readout.detach().clone()
            quad_readout = -self.quad_readout.detach().clone()
            if mode in {"basis_aware_snr_projected", "basis_aware_orthogonal"}:
                direct[self.input_dim :, :].mul_(0.5)
                quad_readout[: self.hidden_dim, :].mul_(0.25)
            out = [direct]
            if isinstance(self.quad_proj, nn.Parameter) and self.quad_proj.requires_grad:
                out.append(torch.zeros_like(self.quad_proj))
            out.append(quad_readout)
            if isinstance(self.branch_scale, nn.Parameter) and self.branch_scale.requires_grad:
                out.append(torch.zeros_like(self.branch_scale))
            out.append(-self.bias.detach().clone())
            if isinstance(self.logit_gain, nn.Parameter) and self.logit_gain.requires_grad:
                out.insert(-1, torch.zeros_like(self.logit_gain))
            return out


def _gated_branch_dims(hidden_dim: int) -> tuple[int, int]:
    legendre_hidden = max(8, int(hidden_dim) // 4)
    quadratic_hidden = max(16, int(hidden_dim) // 2)
    return legendre_hidden, quadratic_hidden


def _legendre4(z: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.ones_like(z),
            z,
            0.5 * (3.0 * z.square() - 1.0),
            0.5 * (5.0 * z * z.square() - 3.0 * z),
        ],
        dim=-1,
    )


def _legendre4_derivative(z: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.zeros_like(z),
            torch.ones_like(z),
            3.0 * z,
            0.5 * (15.0 * z.square() - 3.0),
        ],
        dim=-1,
    )


class _GatedLQFastReuseManualFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
        block_mu: torch.Tensor,
        block_whiten: torch.Tensor,
        block_norm_target_rms: torch.Tensor,
        residual_geom_mix: torch.Tensor,
        leg_w1: torch.Tensor,
        leg_w2: torch.Tensor,
        quad_proj: torch.Tensor,
        quad_readout: torch.Tensor,
        branch_scale: torch.Tensor,
        logit_gain: torch.Tensor,
        bias: torch.Tensor,
        direct_readout: torch.Tensor,
        quad_feature_std: torch.Tensor,
        leg_logit_std: torch.Tensor,
        quad_logit_std: torch.Tensor,
        direct_skip_scale: torch.Tensor,
        input_dim: int,
        input_norm_padded_dim: int,
        input_norm_block_size: int,
        legendre_hidden: int,
        quad_feature_norm_enabled: bool,
        branch_output_norm_enabled: bool,
    ) -> torch.Tensor:
        del input_dim, legendre_hidden
        batch = int(x.shape[0])
        base = (x - mu) / std
        pad = int(input_norm_padded_dim) - int(x.shape[1])
        xpad = F.pad(x, (0, pad)) if pad > 0 else x
        blocks = xpad.reshape(batch, -1, int(input_norm_block_size))
        z = torch.einsum("bni,nij->bnj", blocks - block_mu.unsqueeze(0), block_whiten)
        z = z.reshape(batch, int(input_norm_padded_dim))[:, : int(x.shape[1])]
        rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
        z = z / rms * block_norm_target_rms
        mixed = (1.0 - residual_geom_mix) * base + residual_geom_mix * z
        leg_z = torch.tanh(mixed)
        quad_x = mixed.clamp(-3.0, 3.0)

        sqrt_input = math.sqrt(max(1, int(x.shape[1])))
        sqrt_leg = math.sqrt(max(1, int(leg_w1.shape[1])))
        b1 = _legendre4(leg_z)
        h_pre = torch.einsum("bdk,dhk->bh", b1, leg_w1) / sqrt_input
        h = torch.tanh(h_pre)
        b2 = _legendre4(h)
        leg_logits = torch.einsum("bhk,hck->bc", b2, leg_w2) / sqrt_leg

        q_raw = quad_x @ quad_proj
        q = q_raw / quad_feature_std.clamp_min(1.0e-4) if bool(quad_feature_norm_enabled) else q_raw
        centered_square = q.square() - q.square().mean(dim=0, keepdim=True).detach()
        quad_feats = torch.stack([q, centered_square], dim=2)
        quad_logits = torch.einsum("bhk,hkc->bc", quad_feats, quad_readout)

        if bool(branch_output_norm_enabled):
            leg_logits_norm = leg_logits / leg_logit_std.clamp_min(1.0e-4)
            quad_logits_norm = quad_logits / quad_logit_std.clamp_min(1.0e-4)
        else:
            leg_logits_norm = leg_logits
            quad_logits_norm = quad_logits

        direct_logits = torch.einsum("bdk,dck->bc", b1, direct_readout) / sqrt_input
        logits_unscaled = (
            branch_scale[0] * leg_logits_norm
            + branch_scale[1] * quad_logits_norm
            + bias
            + direct_skip_scale * direct_logits
        )
        gain = logit_gain.clamp(0.25, 4.0)
        out = gain * logits_unscaled
        ctx.save_for_backward(
            b1,
            h,
            b2,
            quad_x,
            q,
            quad_feats,
            leg_logits_norm,
            quad_logits_norm,
            direct_logits,
            logits_unscaled,
            leg_w2,
            quad_proj,
            quad_readout,
            branch_scale,
            logit_gain,
            quad_feature_std,
            leg_logit_std,
            quad_logit_std,
            direct_skip_scale,
        )
        ctx.sqrt_input = sqrt_input
        ctx.sqrt_leg = sqrt_leg
        ctx.quad_feature_norm_enabled = bool(quad_feature_norm_enabled)
        ctx.branch_output_norm_enabled = bool(branch_output_norm_enabled)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (
            b1,
            h,
            b2,
            quad_x,
            q,
            quad_feats,
            leg_logits_norm,
            quad_logits_norm,
            direct_logits,
            logits_unscaled,
            leg_w2,
            _quad_proj,
            quad_readout,
            branch_scale,
            logit_gain,
            quad_feature_std,
            leg_logit_std,
            quad_logit_std,
            direct_skip_scale,
        ) = ctx.saved_tensors
        del _quad_proj
        gain = logit_gain.clamp(0.25, 4.0)
        g_logits = grad_output * gain
        gain_mask = ((logit_gain >= 0.25) & (logit_gain <= 4.0)).to(dtype=grad_output.dtype)
        grad_logit_gain = (grad_output * logits_unscaled).sum().view_as(logit_gain) * gain_mask
        grad_branch_scale = torch.stack(
            [
                (g_logits * leg_logits_norm).sum(),
                (g_logits * quad_logits_norm).sum(),
            ]
        ).view_as(branch_scale)
        grad_bias = g_logits.sum(dim=0)

        grad_direct_logits = g_logits * direct_skip_scale
        grad_direct_readout = torch.einsum("bdk,bc->dck", b1, grad_direct_logits) / ctx.sqrt_input

        grad_leg_logits = g_logits * branch_scale[0]
        if ctx.branch_output_norm_enabled:
            grad_leg_logits = grad_leg_logits / leg_logit_std.clamp_min(1.0e-4)
        grad_leg_w2 = torch.einsum("bhk,bc->hck", b2, grad_leg_logits) / ctx.sqrt_leg
        grad_b2 = torch.einsum("bc,hck->bhk", grad_leg_logits, leg_w2) / ctx.sqrt_leg
        grad_h = (grad_b2 * _legendre4_derivative(h)).sum(dim=2)
        grad_h_pre = grad_h * (1.0 - h.square())
        grad_leg_w1 = torch.einsum("bdk,bh->dhk", b1, grad_h_pre) / ctx.sqrt_input

        grad_quad_logits = g_logits * branch_scale[1]
        if ctx.branch_output_norm_enabled:
            grad_quad_logits = grad_quad_logits / quad_logit_std.clamp_min(1.0e-4)
        grad_quad_readout = torch.einsum("bhk,bc->hkc", quad_feats, grad_quad_logits)
        grad_quad_feats = torch.einsum("bc,hkc->bhk", grad_quad_logits, quad_readout)
        grad_q = grad_quad_feats[:, :, 0] + 2.0 * q * grad_quad_feats[:, :, 1]
        if ctx.quad_feature_norm_enabled:
            grad_q = grad_q / quad_feature_std.clamp_min(1.0e-4)
        grad_quad_proj = quad_x.transpose(0, 1) @ grad_q

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_leg_w1,
            grad_leg_w2,
            grad_quad_proj,
            grad_quad_readout,
            grad_branch_scale,
            grad_logit_gain,
            grad_bias,
            grad_direct_readout,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class GatedLegendreQuadraticKAN(nn.Module):
    """Official-capable coupling of Legendre edge basis and quadratic sketch.

    This primitive is the v12.4 follow-up to the B10/B11 and B3b split:
    B10/B11 have global quadratic coverage, while B3b has synthetic
    trainability but task failure.  The branch coupling keeps all learnable
    tensors as basis weights/projections and does not add an MLP shortcut.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(spec.hidden_dim)
        self.k = 4
        self.legendre_hidden, self.quadratic_hidden = _gated_branch_dims(self.hidden_dim)
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        self.register_buffer("mu", xs.mean(dim=0))
        self.register_buffer("std", xs.std(dim=0).clamp_min(1.0e-3))
        geom_norm_variants = {
            "geom_input_norm",
            "quad_boost_geom_input_norm",
            "geom_input_basis_norm",
            "quadboost_geom_input_basis_norm",
        }
        residual_geom_norm_variants = {
            "residual_geom_input_norm",
            "quad_boost_residual_geom_input_norm",
            "residual_geom_input_basis_norm",
            "quadboost_residual_geom_input_basis_norm",
            "residual_geom_mix15",
            "residual_geom_mix25",
            "quad_boost_residual_geom_mix15",
            "quad_boost_residual_geom_mix25",
            "basis_specific_residual_quad_norm",
            "quad_boost_basis_specific_residual_quad_norm",
            "basis_specific_residual_quad_midboost",
            "basis_specific_residual_quad_lowboost",
            "basis_specific_residual_quad_lowboost_temp065",
            "basis_specific_residual_quad_lowboost_temp075",
            "basis_specific_residual_quad_lowboost_temp050",
            "basis_specific_residual_quad_lowboost_temp050_directskip",
            "basis_specific_residual_quad_lowboost_temp065_directskip",
            "basis_specific_residual_quad_lowboost_temp075_directskip",
            "basis_specific_residual_quad_lowboost_temp050_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw",
            "basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse",
        }
        basis_specific_quad_norm_variants = {
            "basis_specific_quad_norm",
            "quad_boost_basis_specific_quad_norm",
            "basis_specific_quad_norm_temp075",
            "basis_specific_quad_norm_temp050",
            "basis_specific_residual_quad_norm",
            "quad_boost_basis_specific_residual_quad_norm",
            "basis_specific_residual_quad_midboost",
            "basis_specific_residual_quad_lowboost",
            "basis_specific_residual_quad_lowboost_temp065",
            "basis_specific_residual_quad_lowboost_temp075",
            "basis_specific_residual_quad_lowboost_temp050",
            "basis_specific_residual_quad_lowboost_temp050_directskip",
            "basis_specific_residual_quad_lowboost_temp065_directskip",
            "basis_specific_residual_quad_lowboost_temp075_directskip",
            "basis_specific_residual_quad_lowboost_temp050_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw",
            "basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse",
        }
        group_rms_norm_variants = {
            "group_rms_input_norm",
            "quad_boost_group_rms_input_norm",
            "group_rms_input_basis_norm",
            "quadboost_group_rms_input_basis_norm",
        }
        self.input_block_norm_enabled = spec.init_variant in geom_norm_variants
        self.input_residual_block_norm_enabled = spec.init_variant in residual_geom_norm_variants
        self.input_group_rms_norm_enabled = spec.init_variant in group_rms_norm_variants
        self.basis_specific_quad_input_norm_enabled = spec.init_variant in basis_specific_quad_norm_variants
        self.input_norm_block_size = 32
        self.input_norm_padded_dim = int(math.ceil(float(self.input_dim) / float(self.input_norm_block_size)) * self.input_norm_block_size)
        self.register_buffer("geom_std", self.std)
        self.register_buffer("group_norm_target_rms", torch.tensor([0.75], device=device))
        residual_mix = 0.35
        if spec.init_variant in {
            "residual_geom_mix15",
            "quad_boost_residual_geom_mix15",
            "basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse",
        }:
            residual_mix = 0.15
        if spec.init_variant in {"residual_geom_mix25", "quad_boost_residual_geom_mix25"}:
            residual_mix = 0.25
        if spec.init_variant in {
            "basis_specific_residual_quad_norm",
            "quad_boost_basis_specific_residual_quad_norm",
            "basis_specific_residual_quad_midboost",
            "basis_specific_residual_quad_lowboost",
            "basis_specific_residual_quad_lowboost_temp065",
            "basis_specific_residual_quad_lowboost_temp075",
            "basis_specific_residual_quad_lowboost_temp050",
            "basis_specific_residual_quad_lowboost_temp050_directskip",
            "basis_specific_residual_quad_lowboost_temp065_directskip",
            "basis_specific_residual_quad_lowboost_temp075_directskip",
            "basis_specific_residual_quad_lowboost_temp050_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw",
            "basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse",
        }:
            residual_mix = 0.25
        self.register_buffer("residual_geom_mix", torch.tensor([residual_mix], device=device))
        if self.input_group_rms_norm_enabled:
            var = xs.var(dim=0, unbiased=False)
            global_var = var.mean().clamp_min(1.0e-6)
            shrink_std = (0.50 * var + 0.50 * global_var).sqrt().clamp_min(1.0e-3)
            self.geom_std.copy_(shrink_std)
        if self.input_block_norm_enabled or self.input_residual_block_norm_enabled:
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(xs, (0, pad)) if pad > 0 else xs
            blocks = xpad.reshape(int(xpad.shape[0]), -1, self.input_norm_block_size)
            block_mu = blocks.mean(dim=0)
            centered = blocks - block_mu.unsqueeze(0)
            denom = max(1, int(centered.shape[0]) - 1)
            cov = torch.einsum("nbi,nbj->bij", centered, centered) / float(denom)
            eye = torch.eye(self.input_norm_block_size, device=device).unsqueeze(0)
            trace = cov.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0e-6)
            cov = 0.90 * cov + 0.10 * trace[:, None, None] * eye
            evals, evecs = torch.linalg.eigh(cov.float())
            inv_sqrt = torch.diag_embed(evals.clamp_min(1.0e-4).rsqrt())
            block_whiten = evecs @ inv_sqrt @ evecs.transpose(-1, -2)
            self.register_buffer("block_mu", block_mu)
            self.register_buffer("block_whiten", block_whiten)
            self.register_buffer("block_norm_target_rms", torch.tensor([0.75], device=device))
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, self.k, device=device))
        self.register_buffer("scales", torch.tensor([max(0.2, 2.0 / max(1, self.k - 1))], device=device))
        gen = torch.Generator(device=device).manual_seed(int(seed) + 12414)
        self.leg_w1 = nn.Parameter(
            torch.randn(self.input_dim, self.legendre_hidden, self.k, device=device, generator=gen)
            / math.sqrt(max(1, self.input_dim * self.k))
        )
        self.leg_w2 = nn.Parameter(
            torch.randn(self.legendre_hidden, self.output_dim, self.k, device=device, generator=gen)
            / math.sqrt(max(1, self.legendre_hidden * self.k))
        )
        self.quad_proj = nn.Parameter(
            torch.randn(self.input_dim, self.quadratic_hidden, device=device, generator=gen)
            / math.sqrt(max(1, self.input_dim))
        )
        self.quad_readout = nn.Parameter(
            torch.randn(self.quadratic_hidden, 2, self.output_dim, device=device, generator=gen)
            / math.sqrt(max(1, self.quadratic_hidden * 2))
        )
        self.quad_feature_norm_enabled = spec.init_variant in basis_specific_quad_norm_variants
        self.register_buffer("quad_feature_std", torch.ones(self.quadratic_hidden, device=device))
        if self.quad_feature_norm_enabled:
            self._calibrate_quadratic_feature_norm(xs)
        branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "loss_gain":
            branch_init = [1.0, 0.75]
        if spec.init_variant == "quad_boost_loss_gain":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "geom_input_norm":
            branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost_geom_input_norm":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "basis_norm":
            branch_init = [0.20, 0.20]
        if spec.init_variant == "quad_boost_basis_norm":
            branch_init = [0.15, 0.25]
        if spec.init_variant == "geom_input_basis_norm":
            branch_init = [0.20, 0.20]
        if spec.init_variant == "quadboost_geom_input_basis_norm":
            branch_init = [0.15, 0.25]
        if spec.init_variant == "group_rms_input_norm":
            branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost_group_rms_input_norm":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "group_rms_input_basis_norm":
            branch_init = [0.20, 0.20]
        if spec.init_variant == "quadboost_group_rms_input_basis_norm":
            branch_init = [0.15, 0.25]
        if spec.init_variant == "residual_geom_input_norm":
            branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost_residual_geom_input_norm":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "residual_geom_input_basis_norm":
            branch_init = [0.20, 0.20]
        if spec.init_variant == "quadboost_residual_geom_input_basis_norm":
            branch_init = [0.15, 0.25]
        if spec.init_variant in {"residual_geom_mix15", "residual_geom_mix25"}:
            branch_init = [1.0, 0.25]
        if spec.init_variant in {"quad_boost_residual_geom_mix15", "quad_boost_residual_geom_mix25"}:
            branch_init = [0.75, 0.75]
        if spec.init_variant == "basis_specific_quad_norm":
            branch_init = [1.0, 0.25]
        if spec.init_variant in {"basis_specific_quad_norm_temp075", "basis_specific_quad_norm_temp050"}:
            branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost_basis_specific_quad_norm":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "basis_specific_residual_quad_norm":
            branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost_basis_specific_residual_quad_norm":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "basis_specific_residual_quad_midboost":
            branch_init = [1.0, 0.50]
        if spec.init_variant == "basis_specific_residual_quad_lowboost":
            branch_init = [1.0, 0.35]
        if spec.init_variant in {
            "basis_specific_residual_quad_lowboost_temp065",
            "basis_specific_residual_quad_lowboost_temp075",
            "basis_specific_residual_quad_lowboost_temp050",
            "basis_specific_residual_quad_lowboost_temp050_directskip",
            "basis_specific_residual_quad_lowboost_temp065_directskip",
            "basis_specific_residual_quad_lowboost_temp075_directskip",
            "basis_specific_residual_quad_lowboost_temp050_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw",
            "basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse",
        }:
            branch_init = [1.0, 0.35]
        self.branch_scale = nn.Parameter(torch.tensor(branch_init, device=device))
        gain_init = 2.0 if spec.init_variant in {"loss_gain", "quad_boost_loss_gain"} else 1.0
        if spec.init_variant in {"basis_specific_residual_quad_lowboost_temp075", "basis_specific_residual_quad_lowboost_temp075_directskip", "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse", "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw", "basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse", "basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse"}:
            gain_init = 0.75
        if spec.init_variant in {"basis_specific_residual_quad_lowboost_temp065", "basis_specific_residual_quad_lowboost_temp065_directskip", "basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse", "basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse"}:
            gain_init = 0.65
        if spec.init_variant in {"basis_specific_residual_quad_lowboost_temp050", "basis_specific_residual_quad_lowboost_temp050_directskip", "basis_specific_residual_quad_lowboost_temp050_directskip_fastreuse"}:
            gain_init = 0.50
        if spec.init_variant == "basis_specific_quad_norm_temp075":
            gain_init = 0.75
        if spec.init_variant == "basis_specific_quad_norm_temp050":
            gain_init = 0.50
        self.logit_gain = nn.Parameter(torch.tensor([gain_init], device=device))
        self.bias = nn.Parameter(torch.zeros(self.output_dim, device=device))
        self.direct_skip_enabled = "directskip" in str(spec.init_variant)
        self.fast_reuse_forward_enabled = "fastreuse" in str(spec.init_variant)
        self.manual_backward_enabled = "manualbw" in str(spec.init_variant)
        if self.direct_skip_enabled:
            self.direct_readout = nn.Parameter(
                torch.randn(self.input_dim, self.output_dim, self.k, device=device, generator=gen)
                / math.sqrt(max(1, self.input_dim * self.k))
            )
        direct_skip_scale = 0.50 if "directskip050" in str(spec.init_variant) else 0.25
        self.register_buffer("direct_skip_scale", torch.tensor([direct_skip_scale], device=device))
        self.register_buffer("leg_logit_std", torch.ones(1, device=device))
        self.register_buffer("quad_logit_std", torch.ones(1, device=device))
        self.branch_output_norm_enabled = spec.init_variant in {
            "basis_norm",
            "quad_boost_basis_norm",
            "geom_input_basis_norm",
            "quadboost_geom_input_basis_norm",
            "group_rms_input_basis_norm",
            "quadboost_group_rms_input_basis_norm",
            "residual_geom_input_basis_norm",
            "quadboost_residual_geom_input_basis_norm",
        }
        if self.branch_output_norm_enabled:
            self._calibrate_branch_output_norm(xs)

    def _calibrate_quadratic_feature_norm(self, xs: torch.Tensor) -> None:
        """Calibrate quadratic projection coordinates without labels."""
        with torch.no_grad():
            sample = xs[: min(2048, int(xs.shape[0]))]
            z = self._quadratic_input(sample) @ self.quad_proj
            self.quad_feature_std.copy_(z.float().std(dim=0).clamp_min(1.0e-4))

    def _calibrate_branch_output_norm(self, xs: torch.Tensor) -> None:
        """Calibrate branch output scale from unlabeled train-stream samples."""
        with torch.no_grad():
            sample = xs[: min(1024, int(xs.shape[0]))]
            leg_logits, _ = self._legendre_logits_and_features(sample)
            quad_logits, _ = self._quadratic_logits_and_features(sample)
            self.leg_logit_std.copy_(leg_logits.float().std().clamp_min(1.0e-4).view(1))
            self.quad_logit_std.copy_(quad_logits.float().std().clamp_min(1.0e-4).view(1))

    @property
    def edge_param_count(self) -> int:
        return int(
            self.leg_w1.numel()
            + self.leg_w2.numel()
            + self.quad_proj.numel()
            + self.quad_readout.numel()
            + self.branch_scale.numel()
            + self.logit_gain.numel()
            + self.bias.numel()
            + (self.direct_readout.numel() if self.direct_skip_enabled else 0)
        )

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_residual_block_norm_enabled:
            base = (x - self.mu) / self.std
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(x, (0, pad)) if pad > 0 else x
            blocks = xpad.reshape(int(x.shape[0]), -1, self.input_norm_block_size)
            z = torch.einsum("bni,nij->bnj", blocks - self.block_mu.unsqueeze(0), self.block_whiten)
            z = z.reshape(int(x.shape[0]), self.input_norm_padded_dim)[:, : self.input_dim]
            rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
            z = z / rms * self.block_norm_target_rms
            mixed = (1.0 - self.residual_geom_mix) * base + self.residual_geom_mix * z
            return torch.tanh(mixed)
        if self.input_group_rms_norm_enabled:
            z = (x - self.mu) / self.geom_std.clamp_min(1.0e-3)
            pad = self.input_norm_padded_dim - self.input_dim
            zpad = F.pad(z, (0, pad)) if pad > 0 else z
            blocks = zpad.reshape(int(x.shape[0]), -1, self.input_norm_block_size)
            block_rms = blocks.float().square().mean(dim=2, keepdim=True).sqrt().clamp_min(1.0e-3)
            blocks = blocks / block_rms * self.group_norm_target_rms
            z = blocks.reshape(int(x.shape[0]), self.input_norm_padded_dim)[:, : self.input_dim]
            return torch.tanh(z)
        if self.input_block_norm_enabled:
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(x, (0, pad)) if pad > 0 else x
            blocks = xpad.reshape(int(x.shape[0]), -1, self.input_norm_block_size)
            z = torch.einsum("bni,nij->bnj", blocks - self.block_mu.unsqueeze(0), self.block_whiten)
            z = z.reshape(int(x.shape[0]), self.input_norm_padded_dim)[:, : self.input_dim]
            rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
            z = z / rms * self.block_norm_target_rms
            return torch.tanh(z)
        return torch.tanh((x - self.mu) / self.std)

    def _quadratic_input(self, x: torch.Tensor) -> torch.Tensor:
        """Basis-specific input normalization for the quadratic branch.

        Legendre features need bounded coordinates.  Quadratic features need
        the linear coordinate system to survive the input geometry layer, so
        these variants avoid the final tanh while still using fixed,
        unlabeled train-stream normalization and a conservative clamp.
        """
        if not self.basis_specific_quad_input_norm_enabled:
            return self._norm_input(x)
        base = (x - self.mu) / self.std
        if self.input_residual_block_norm_enabled:
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(x, (0, pad)) if pad > 0 else x
            blocks = xpad.reshape(int(x.shape[0]), -1, self.input_norm_block_size)
            z = torch.einsum("bni,nij->bnj", blocks - self.block_mu.unsqueeze(0), self.block_whiten)
            z = z.reshape(int(x.shape[0]), self.input_norm_padded_dim)[:, : self.input_dim]
            rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
            z = z / rms * self.block_norm_target_rms
            base = (1.0 - self.residual_geom_mix) * base + self.residual_geom_mix * z
        return base.clamp(-3.0, 3.0)

    def _paired_legendre_quadratic_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return equivalent Legendre and quadratic inputs while sharing norm work."""
        if self.basis_specific_quad_input_norm_enabled and self.input_residual_block_norm_enabled:
            base = (x - self.mu) / self.std
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(x, (0, pad)) if pad > 0 else x
            blocks = xpad.reshape(int(x.shape[0]), -1, self.input_norm_block_size)
            z = torch.einsum("bni,nij->bnj", blocks - self.block_mu.unsqueeze(0), self.block_whiten)
            z = z.reshape(int(x.shape[0]), self.input_norm_padded_dim)[:, : self.input_dim]
            rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
            z = z / rms * self.block_norm_target_rms
            mixed = (1.0 - self.residual_geom_mix) * base + self.residual_geom_mix * z
            return torch.tanh(mixed), mixed.clamp(-3.0, 3.0)
        if self.basis_specific_quad_input_norm_enabled:
            base = (x - self.mu) / self.std
            return self._norm_input(x), base.clamp(-3.0, 3.0)
        z = self._norm_input(x)
        return z, z

    def _forward_reuse_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path for directskip variants; mathematically matches helper calls."""
        leg_z, quad_x = self._paired_legendre_quadratic_inputs(x)
        b1 = _basis_eval(leg_z, "legendre", self.k, self.centers, self.scales)
        h = torch.einsum("bdk,dhk->bh", b1, self.leg_w1) / math.sqrt(max(1, self.input_dim))
        h = torch.tanh(h)
        b2 = _basis_eval(h, "legendre", self.k, self.centers, self.scales)
        leg_logits = torch.einsum("bhk,hck->bc", b2, self.leg_w2) / math.sqrt(max(1, self.legendre_hidden))

        q = quad_x @ self.quad_proj
        if self.quad_feature_norm_enabled:
            q = q / self.quad_feature_std.clamp_min(1.0e-4)
        centered_square = q.square() - q.square().mean(dim=0, keepdim=True).detach()
        quad_logits = torch.einsum("bhk,hkc->bc", torch.stack([q, centered_square], dim=2), self.quad_readout)

        if self.branch_output_norm_enabled:
            leg_logits = leg_logits / self.leg_logit_std.clamp_min(1.0e-4)
            quad_logits = quad_logits / self.quad_logit_std.clamp_min(1.0e-4)
        logits = self.branch_scale[0] * leg_logits + self.branch_scale[1] * quad_logits + self.bias
        direct_logits = torch.einsum("bdk,dck->bc", b1, self.direct_readout) / math.sqrt(max(1, self.input_dim))
        logits = logits + self.direct_skip_scale * direct_logits
        return self.logit_gain.clamp(0.25, 4.0) * logits

    def _manual_kernel_ready(self) -> bool:
        return bool(
            self.manual_backward_enabled
            and self.fast_reuse_forward_enabled
            and self.direct_skip_enabled
            and self.input_residual_block_norm_enabled
            and self.basis_specific_quad_input_norm_enabled
            and self.k == 4
            and hasattr(self, "block_mu")
            and hasattr(self, "block_whiten")
            and hasattr(self, "direct_readout")
        )

    def _forward_reuse_inputs_manual(self, x: torch.Tensor) -> torch.Tensor:
        return _GatedLQFastReuseManualFunction.apply(
            x,
            self.mu,
            self.std,
            self.block_mu,
            self.block_whiten,
            self.block_norm_target_rms,
            self.residual_geom_mix,
            self.leg_w1,
            self.leg_w2,
            self.quad_proj,
            self.quad_readout,
            self.branch_scale,
            self.logit_gain,
            self.bias,
            self.direct_readout,
            self.quad_feature_std,
            self.leg_logit_std,
            self.quad_logit_std,
            self.direct_skip_scale,
            self.input_dim,
            self.input_norm_padded_dim,
            self.input_norm_block_size,
            self.legendre_hidden,
            self.quad_feature_norm_enabled,
            self.branch_output_norm_enabled,
        )

    def manual_kernel_available(self) -> bool:
        return self._manual_kernel_ready()

    def manual_gradient_audit(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        if not self._manual_kernel_ready():
            return {
                "manual_forward_available": 0.0,
                "manual_backward_available": 0.0,
                "grad_relerr_max": float("inf"),
                "grad_cos_min": -1.0,
                "output_max_abs_error": float("inf"),
            }
        params = [p for p in self.parameters() if p.requires_grad]
        self.zero_grad(set_to_none=True)
        manual_logits = self._forward_reuse_inputs_manual(x)
        F.cross_entropy(manual_logits, y).backward()
        manual_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        ref_logits = self._forward_reuse_inputs(x)
        F.cross_entropy(ref_logits, y).backward()
        ref_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        relerrs: List[float] = []
        coses: List[float] = []
        for gm, gr in zip(manual_grads, ref_grads):
            denom = gr.norm().clamp_min(1.0e-8)
            relerrs.append(float((gm - gr).norm().div(denom).detach().item()))
            if gm.norm().item() > 0.0 and gr.norm().item() > 0.0:
                coses.append(float(F.cosine_similarity(gm.flatten(), gr.flatten(), dim=0).detach().item()))
        return {
            "manual_forward_available": 1.0,
            "manual_backward_available": 1.0,
            "grad_relerr_max": max(relerrs) if relerrs else float("inf"),
            "grad_cos_min": min(coses) if coses else -1.0,
            "output_max_abs_error": float((manual_logits - ref_logits).abs().max().detach().item()),
        }

    def manual_ce_forward_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self._manual_kernel_ready():
            raise RuntimeError("manual CE path requested for unsupported GatedLegendreQuadraticKAN variant")
        with torch.no_grad():
            batch = int(x.shape[0])
            base = (x - self.mu) / self.std
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(x, (0, pad)) if pad > 0 else x
            blocks = xpad.reshape(batch, -1, self.input_norm_block_size)
            z = torch.einsum("bni,nij->bnj", blocks - self.block_mu.unsqueeze(0), self.block_whiten)
            z = z.reshape(batch, self.input_norm_padded_dim)[:, : self.input_dim]
            rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
            z = z / rms * self.block_norm_target_rms
            mixed = (1.0 - self.residual_geom_mix) * base + self.residual_geom_mix * z
            leg_z = torch.tanh(mixed)
            quad_x = mixed.clamp(-3.0, 3.0)
            b1 = _legendre4(leg_z)
            h_pre = torch.einsum("bdk,dhk->bh", b1, self.leg_w1) / math.sqrt(max(1, self.input_dim))
            h = torch.tanh(h_pre)
            b2 = _legendre4(h)
            leg_logits = torch.einsum("bhk,hck->bc", b2, self.leg_w2) / math.sqrt(max(1, self.legendre_hidden))
            q_raw = quad_x @ self.quad_proj
            q = q_raw / self.quad_feature_std.clamp_min(1.0e-4) if self.quad_feature_norm_enabled else q_raw
            centered_square = q.square() - q.square().mean(dim=0, keepdim=True)
            quad_feats = torch.stack([q, centered_square], dim=2)
            quad_logits = torch.einsum("bhk,hkc->bc", quad_feats, self.quad_readout)
            leg_logits_norm = leg_logits / self.leg_logit_std.clamp_min(1.0e-4) if self.branch_output_norm_enabled else leg_logits
            quad_logits_norm = quad_logits / self.quad_logit_std.clamp_min(1.0e-4) if self.branch_output_norm_enabled else quad_logits
            direct_logits = torch.einsum("bdk,dck->bc", b1, self.direct_readout) / math.sqrt(max(1, self.input_dim))
            logits_unscaled = (
                self.branch_scale[0] * leg_logits_norm
                + self.branch_scale[1] * quad_logits_norm
                + self.bias
                + self.direct_skip_scale * direct_logits
            )
            logits = self.logit_gain.clamp(0.25, 4.0) * logits_unscaled
            cache = (
                b1,
                h,
                b2,
                quad_x,
                q,
                quad_feats,
                leg_logits_norm,
                quad_logits_norm,
                direct_logits,
                logits_unscaled,
            )
            return logits, cache

    def manual_ce_backward_from_cache(self, logits: torch.Tensor, cache: tuple[torch.Tensor, ...], y: torch.Tensor) -> torch.Tensor:
        if not self._manual_kernel_ready():
            raise RuntimeError("manual CE path requested for unsupported GatedLegendreQuadraticKAN variant")
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y)
            grad_output = probs
            grad_output[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
            grad_output = grad_output / float(max(1, int(y.numel())))
            (
                b1,
                h,
                b2,
                quad_x,
                q,
                quad_feats,
                leg_logits_norm,
                quad_logits_norm,
                _direct_logits,
                logits_unscaled,
            ) = cache
            del _direct_logits
            gain = self.logit_gain.clamp(0.25, 4.0)
            g_logits = grad_output * gain
            gain_mask = ((self.logit_gain >= 0.25) & (self.logit_gain <= 4.0)).to(dtype=grad_output.dtype)
            self.logit_gain.grad = (grad_output * logits_unscaled).sum().view_as(self.logit_gain) * gain_mask
            self.branch_scale.grad = torch.stack(
                [
                    (g_logits * leg_logits_norm).sum(),
                    (g_logits * quad_logits_norm).sum(),
                ]
            ).view_as(self.branch_scale)
            self.bias.grad = g_logits.sum(dim=0)
            grad_direct_logits = g_logits * self.direct_skip_scale
            self.direct_readout.grad = torch.einsum("bdk,bc->dck", b1, grad_direct_logits) / math.sqrt(max(1, self.input_dim))

            grad_leg_logits = g_logits * self.branch_scale[0]
            if self.branch_output_norm_enabled:
                grad_leg_logits = grad_leg_logits / self.leg_logit_std.clamp_min(1.0e-4)
            self.leg_w2.grad = torch.einsum("bhk,bc->hck", b2, grad_leg_logits) / math.sqrt(max(1, self.legendre_hidden))
            grad_b2 = torch.einsum("bc,hck->bhk", grad_leg_logits, self.leg_w2) / math.sqrt(max(1, self.legendre_hidden))
            grad_h = (grad_b2 * _legendre4_derivative(h)).sum(dim=2)
            grad_h_pre = grad_h * (1.0 - h.square())
            self.leg_w1.grad = torch.einsum("bdk,bh->dhk", b1, grad_h_pre) / math.sqrt(max(1, self.input_dim))

            grad_quad_logits = g_logits * self.branch_scale[1]
            if self.branch_output_norm_enabled:
                grad_quad_logits = grad_quad_logits / self.quad_logit_std.clamp_min(1.0e-4)
            self.quad_readout.grad = torch.einsum("bhk,bc->hkc", quad_feats, grad_quad_logits)
            grad_quad_feats = torch.einsum("bc,hkc->bhk", grad_quad_logits, self.quad_readout)
            grad_q = grad_quad_feats[:, :, 0] + 2.0 * q * grad_quad_feats[:, :, 1]
            if self.quad_feature_norm_enabled:
                grad_q = grad_q / self.quad_feature_std.clamp_min(1.0e-4)
            self.quad_proj.grad = quad_x.transpose(0, 1) @ grad_q
            return loss.detach()

    def _legendre_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._norm_input(x)
        b1 = _basis_eval(z, "legendre", self.k, self.centers, self.scales)
        h = torch.einsum("bdk,dhk->bh", b1, self.leg_w1) / math.sqrt(max(1, self.input_dim))
        h = torch.tanh(h)
        b2 = _basis_eval(h, "legendre", self.k, self.centers, self.scales)
        logits = torch.einsum("bhk,hck->bc", b2, self.leg_w2) / math.sqrt(max(1, self.legendre_hidden))
        return logits, b2.reshape(int(x.shape[0]), -1)

    def _quadratic_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._quadratic_input(x) @ self.quad_proj
        if self.quad_feature_norm_enabled:
            z = z / self.quad_feature_std.clamp_min(1.0e-4)
        centered_square = z.square() - z.square().mean(dim=0, keepdim=True).detach()
        feats = torch.stack([z, centered_square], dim=2)
        logits = torch.einsum("bhk,hkc->bc", feats, self.quad_readout)
        return logits, feats.reshape(int(x.shape[0]), -1)

    def _direct_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = _basis_eval(self._norm_input(x), "legendre", self.k, self.centers, self.scales)
        logits = torch.einsum("bdk,dck->bc", b, self.direct_readout) / math.sqrt(max(1, self.input_dim))
        return logits, b.reshape(int(x.shape[0]), -1)

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        _, leg_feats = self._legendre_logits_and_features(x)
        _, quad_feats = self._quadratic_logits_and_features(x)
        if self.direct_skip_enabled:
            _, direct_feats = self._direct_logits_and_features(x)
            return torch.cat([leg_feats, quad_feats, direct_feats], dim=1)
        return torch.cat([leg_feats, quad_feats], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._manual_kernel_ready():
            return self._forward_reuse_inputs_manual(x)
        if self.fast_reuse_forward_enabled and self.direct_skip_enabled:
            return self._forward_reuse_inputs(x)
        leg_logits, _ = self._legendre_logits_and_features(x)
        quad_logits, _ = self._quadratic_logits_and_features(x)
        if self.branch_output_norm_enabled:
            leg_logits = leg_logits / self.leg_logit_std.clamp_min(1.0e-4)
            quad_logits = quad_logits / self.quad_logit_std.clamp_min(1.0e-4)
        logits = self.branch_scale[0] * leg_logits + self.branch_scale[1] * quad_logits + self.bias
        if self.direct_skip_enabled:
            direct_logits, _ = self._direct_logits_and_features(x)
            logits = logits + self.direct_skip_scale * direct_logits
        return self.logit_gain.clamp(0.25, 4.0) * logits

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            return {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "basis_output_norm_p95": float(torch.quantile(feats.norm(dim=1), 0.95).item()),
            }

    def basis_functional_direction(self, mode: str) -> List[torch.Tensor]:
        with torch.no_grad():
            leg1 = -self.leg_w1.detach().clone()
            leg2 = -self.leg_w2.detach().clone()
            quad_proj = torch.zeros_like(self.quad_proj)
            quad_readout = -self.quad_readout.detach().clone()
            if mode in {"basis_aware_snr_projected", "basis_aware_orthogonal"}:
                leg1[..., 0] *= 0.25
                leg2[..., 0] *= 0.25
                quad_readout[:, 0, :] *= 0.25
            out = [
                leg1,
                leg2,
                quad_proj,
                quad_readout,
                torch.zeros_like(self.branch_scale),
                torch.zeros_like(self.logit_gain),
                -self.bias.detach().clone(),
            ]
            if self.direct_skip_enabled:
                out.append(-self.direct_readout.detach().clone())
            return out


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, seed: int, device: torch.device) -> None:
        super().__init__()
        gen = torch.Generator(device=device).manual_seed(int(seed))
        self.w0 = nn.Parameter(torch.randn(input_dim, hidden_dim, device=device, generator=gen) / math.sqrt(input_dim))
        self.w1 = nn.Parameter(torch.randn(hidden_dim, hidden_dim, device=device, generator=gen) / math.sqrt(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, output_dim, device=device, generator=gen) / math.sqrt(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(x @ self.w0)
        h = F.silu(h @ self.w1)
        return h @ self.w2

    @property
    def hidden_dim(self) -> int:
        return int(self.w1.shape[0])


def count_parameters(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def primitive_specs(param_budget: int, input_dim: int, output_dim: int) -> List[PrimitiveSpec]:
    def h(k: int) -> int:
        return matched_hidden(param_budget, input_dim, output_dim, k)

    return [
        PrimitiveSpec("B1r-ReLU-KAN-stream-K2-repair", "Activation", "relu_hinge", 2, h(2), "R1_repair_stream_basis_mix", 1, 0, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=1),
        PrimitiveSpec("B1a-ReLU-KAN-local-hinge-K4", "Activation", "relu_hinge", 4, h(4), "native+plan", 1, 0, 0, 0, 0, basis_order=1),
        PrimitiveSpec("B1b-RSWAF-hinge-K8", "Activation", "rswaf_hinge", 8, h(8), "native+plan", 1, 0, 0, 0, 0, basis_order=1),
        PrimitiveSpec("B2r-FastKAN-RBF-stream-K2-repair", "RBF", "fastkan_rbf", 2, h(2), "R1_repair_stream_basis_mix+third_party/MJKAN", 1, 0, 1, 0, 0, uses_dense_basis_tensor=0, basis_order=1),
        PrimitiveSpec("B2a-GaussianRBF-K4-compact", "RBF", "compact_rbf", 4, h(4), "third_party/MJKAN", 1, 0, 1, 0, 0, basis_order=1),
        PrimitiveSpec("B2b-FastKAN-RBF-K8", "RBF", "fastkan_rbf", 8, h(8), "third_party/MJKAN", 1, 0, 1, 0, 0, basis_order=1),
        PrimitiveSpec("B3a-ChebyKAN-K4", "OrthogonalPolynomial", "chebyshev", 4, h(4), "native+awesome-kan-family", 0, 1, 0, 0, 0, basis_order=4),
        PrimitiveSpec("B3b-LegendreKAN-K4", "OrthogonalPolynomial", "legendre", 4, h(4), "native+awesome-kan-family", 0, 1, 0, 0, 0, basis_order=4),
        PrimitiveSpec("B12a-LegendreKAN-K4-fan-scale-repair", "OrthogonalPolynomial", "legendre", 4, h(4), "R3_repair_global_fan_scale_init", 0, 1, 0, 0, 0, basis_order=4, init_variant="fan_scale_repair"),
        PrimitiveSpec("B12b-ChebyKAN-K4-fan-scale-repair", "OrthogonalPolynomial", "chebyshev", 4, h(4), "R3_repair_global_fan_scale_init", 0, 1, 0, 0, 0, basis_order=4, init_variant="fan_scale_repair"),
        PrimitiveSpec("B13a-LegendreKAN-K4-identity-residual-repair", "OrthogonalPolynomial", "legendre", 4, h(4), "R3_repair_identity_residual_scale_init", 0, 1, 0, 0, 0, basis_order=4, init_variant="identity_residual_scale"),
        PrimitiveSpec("B13b-ChebyKAN-K4-identity-residual-repair", "OrthogonalPolynomial", "chebyshev", 4, h(4), "R3_repair_identity_residual_scale_init", 0, 1, 0, 0, 0, basis_order=4, init_variant="identity_residual_scale"),
        PrimitiveSpec("B14a-GatedLegendreQuadratic-h160", "GatedHybrid", "legendre_quadratic", 4, 160, "R5_repair_gated_legendre_quadratic_coupling", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B14b-GatedLegendreQuadratic-h224", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_gated_legendre_quadratic_coupling", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B14c-GatedLegendreQuadratic-h224-quad-boost", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_gated_legendre_quadratic_quad_boost", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B14d-GatedLegendreQuadratic-h228-quad-boost", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_gated_legendre_quadratic_quad_boost_near_param", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B15a-GatedLegendreQuadratic-h224-loss-gain", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_gated_legendre_quadratic_loss_gain", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="loss_gain", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_gated_legendre_quadratic_quadboost_loss_gain", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_loss_gain", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B16a-GatedLegendreQuadratic-h224-basis-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_basis_specific_output_norm_init", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_basis_specific_output_norm_init", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B17a-GatedLegendreQuadratic-h224-input-geom-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_fixed_block_zca_rms_input_geometry_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="geom_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_fixed_block_zca_rms_input_geometry_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_geom_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_fixed_input_geometry_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="geom_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_input_geometry_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quadboost_geom_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B18a-GatedLegendreQuadratic-h224-group-rms-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_light_diagonal_shrink_group_rms_input_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="group_rms_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_light_diagonal_shrink_group_rms_input_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_group_rms_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_light_group_rms_input_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="group_rms_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_group_rms_input_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quadboost_group_rms_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B19a-GatedLegendreQuadratic-h224-residual-geom-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_residual_block_zca_input_geometry_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="residual_geom_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_residual_block_zca_input_geometry_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_residual_geom_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_residual_input_geometry_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="residual_geom_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_residual_input_geometry_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quadboost_residual_geom_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B20a-GatedLegendreQuadratic-h224-residual-mix15", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_input_norm_objective_residual_mix_grid_015", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="residual_geom_mix15", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B20b-GatedLegendreQuadratic-h224-residual-mix25", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_input_norm_objective_residual_mix_grid_025", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="residual_geom_mix25", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_quadboost_input_norm_objective_residual_mix_grid_015", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_residual_geom_mix15", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_quadboost_input_norm_objective_residual_mix_grid_025", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_residual_geom_mix25", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_basis_specific_input_norm_legendre_bounded_quadratic_linear", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_quadboost_basis_specific_input_norm_legendre_bounded_quadratic_linear", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_basis_specific_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_basis_specific_input_norm_legendre_residual_quadratic_linear", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_quadboost_basis_specific_input_norm_legendre_residual_quadratic_linear", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_basis_specific_residual_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_task_repair_basis_specific_norm_mid_quadratic_branch_scale", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_midboost", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_task_repair_basis_specific_norm_low_quadratic_branch_scale", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_auc_ece_repair_lowboost_temperature_075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_auc_ece_repair_lowboost_temperature_050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B26a-GatedLegendreQuadratic-h224-basis-specific-temp075", "GatedHybrid", "legendre_quadratic", 4, 224, "R3_auc_ece_repair_plain_basis_specific_temperature_075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_quad_norm_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B26b-GatedLegendreQuadratic-h224-basis-specific-temp050", "GatedHybrid", "legendre_quadratic", 4, 224, "R3_auc_ece_repair_plain_basis_specific_temperature_050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_quad_norm_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr", "GatedHybrid", "legendre_quadratic", 4, 224, "R3_task_repair_basis_specific_quad_norm_warmup_cosine_lr", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr", "GatedHybrid", "legendre_quadratic", 4, 228, "R3_task_repair_basis_specific_residual_quad_norm_warmup_cosine_lr", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_basis_specific_residual_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr", "GatedHybrid", "legendre_quadratic", 4, 228, "R3_task_repair_lowboost_temp050_warmup_cosine_lr", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B28a-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 228, "R3_task_repair_lowboost_temp050_warmup10_cosine_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B28b-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 228, "R3_task_repair_lowboost_temp050_warmup10_cosine_final075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B29a-GatedLegendreQuadratic-h224-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 224, "R1_R3_repair_h224_lowboost_temp050_warmup10_cosine_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B29b-GatedLegendreQuadratic-h224-lowboost-temp050-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 224, "R1_R3_repair_h224_lowboost_temp050_warmup10_cosine_final075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B30a-LiteGatedLegendreQuadratic-h192-temp050-cosine-lr-final050", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 192, "R1_R3_repair_lite_direct_legendre_quadratic_basis_specific_input_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_temp050", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 256, "R1_R3_repair_lite_direct_legendre_quadratic_basis_specific_input_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_temp050", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B30c-LiteGatedLegendreQuadratic-h296-temp075-cosine-lr-final075", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 296, "R1_R3_repair_near_param_lite_direct_legendre_quadratic_basis_specific_input_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_temp075", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B31a-LiteGatedLegendreQuadratic-h224-midboost-temp100-cosine-lr-final050", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 224, "R2_repair_lite_quadratic_branch_midboost_basis_specific_init", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_midboost_temp100", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B31b-LiteGatedLegendreQuadratic-h256-quadboost-temp075-cosine-lr-final075", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 256, "R2_repair_lite_quadratic_branch_quadboost_basis_specific_init", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_quadboost_temp075", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B31c-LiteGatedLegendreQuadratic-h256-quadboost-temp100-cosine-lr-final050", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 256, "R2_repair_lite_quadratic_branch_quadboost_basis_specific_init", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_quadboost_temp100", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 160, "R1_R2_R3_repair_smaller_gated_basis_specific_lowboost_temp050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B32b-GatedLegendreQuadratic-h192-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 192, "R1_R2_R3_repair_smaller_gated_basis_specific_lowboost_temp050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 192, "R1_R2_R3_repair_smaller_gated_basis_specific_lowboost_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 160, "R3_repair_smaller_gated_temperature_hidden_tradeoff", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_smaller_gated_temperature_hidden_tradeoff", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_smaller_gated_temperature_hidden_tradeoff", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B34a-GatedLegendreQuadratic-h168-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_auc_task_tradeoff_hidden_168", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_auc_task_tradeoff_hidden_168_temp075_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 160, "R3_repair_auc_task_tradeoff_h160_temp075_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_auc_task_tradeoff_intermediate_temperature_065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_auc_task_tradeoff_intermediate_temperature_065_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_auc_task_tradeoff_h168_intermediate_temperature_065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 160, "R3_repair_direct_legendre_skip_for_task_stability", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_direct_legendre_skip_for_task_stability_temp065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_direct_legendre_skip_for_task_stability_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B37a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-branchslow-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_branchwise_optimizer_scale_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_branchwise_optimizer_scale_temp065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B37c-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-branchslow-final050", "GatedHybrid", "legendre_quadratic", 4, 160, "R3_repair_branchwise_optimizer_scale_temp050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B38a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-legfast-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_branchwise_legendre_direct_fast_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B38b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-legfast-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_branchwise_legendre_direct_fast_temp065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B38c-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-legfast-final050", "GatedHybrid", "legendre_quadratic", 4, 160, "R3_repair_branchwise_legendre_direct_fast_temp050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B39a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_b36c_temp075_directskip_final050_singlepoint", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B41a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R1_R3_repair_b36c_light_h168_temp075_directskip", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B41b-GatedLegendreQuadratic-h168-lowboost-temp065-directskip-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R1_R3_repair_b36b_light_h168_temp065_directskip", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B42a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-fastreuse-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R1_repair_manual_reuse_input_norm_and_legendre_basis_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R1_repair_manual_reuse_input_norm_and_legendre_basis_h168_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B42c-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-fastreuse-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R1_R3_repair_manual_reuse_input_norm_and_legendre_basis_temp065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B43a-GatedLegendreQuadratic-h172-lowboost-temp075-directskip-fastreuse-final075", "GatedHybrid", "legendre_quadratic", 4, 172, "R1_R3_repair_h172_fastreuse_capacity_speed_tradeoff_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050", "GatedHybrid", "legendre_quadratic", 4, 172, "R1_R3_repair_h172_fastreuse_capacity_speed_tradeoff_temp065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B44a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-mix15-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_input_norm_residual_mix15_temp075_fastreuse", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B44b-GatedLegendreQuadratic-h168-lowboost-temp065-directskip-fastreuse-mix15-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_input_norm_residual_mix15_temp065_fastreuse", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_b42b_temp075_fastreuse_cosine_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B46a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip050-fastreuse-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_directskip_scale050_task_geometry_temp075_final075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B46b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip050-fastreuse-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_directskip_scale050_task_geometry_temp075_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B47a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R1_repair_true_manual_backward_kernel_temp075_final075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R1_R3_repair_true_manual_backward_kernel_temp075_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B48a-SimpleFastTaskGeometry-h128-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 128, "R6_simple_fast_task_geometry_hinge_minimal_quadratic", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad030_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B48b-SimpleFastTaskGeometry-h192-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 192, "R6_simple_fast_task_geometry_hinge_mid_quadratic", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B48c-SimpleFastTaskGeometry-h64-quad020-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 64, "R6_simple_fast_task_geometry_lowrank_minimal_quadratic_efficiency_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad020_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B48d-SimpleFastTaskGeometry-h128-temp050", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 128, "R3_simple_fast_task_geometry_low_temperature_task_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad030_temp050", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B48e-SimpleFastTaskGeometry-h128-temp065", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 128, "R3_simple_fast_task_geometry_mid_temperature_auc_accuracy_tradeoff", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad030_temp065", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B49a-SimpleFastTaskGeometry-h96-sqdiag-temp075", "SimpleFastTaskGeometry", "hinge_direct_diag_quadratic", 2, 96, "R6_simple_fast_task_geometry_diag_quadratic_direct_channel", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_quad030_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B50a-SimpleFastTaskGeometry-h128-temp100", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 128, "R3_auc_nll_repair_high_logit_gain", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad030_temp100", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B50b-SimpleFastTaskGeometry-h96-twohinge-temp075", "SimpleFastTaskGeometry", "twohinge_direct_quadratic", 2, 96, "R6_low_cost_twohinge_task_geometry_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="twohinge_quad020_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B50c-SimpleFastTaskGeometry-h128-twohinge-temp075", "SimpleFastTaskGeometry", "twohinge_direct_quadratic", 2, 128, "R2_R6_twohinge_expression_repair_with_efficiency_slack", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="twohinge_quad030_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B50d-SimpleFastTaskGeometry-h112-twohinge-temp075", "SimpleFastTaskGeometry", "twohinge_direct_quadratic", 2, 112, "R1_R2_twohinge_midpoint_expression_efficiency_tradeoff", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="twohinge_quad020_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B51a-SimpleFastTaskGeometry-h128-fixedP-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic_fixedp", 2, 128, "FHQ0_fixed_projection_backward_cost_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad030_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B52a-SimpleFastTaskGeometry-h128-fixedP-identitytail-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic_fixedp", 2, 128, "FHQ_identity_tail_direct_path_task_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad010_identitytail_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B54a-SimpleFastTaskGeometry-h128-learnableP-identitytail-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 128, "LineP_learnable_projection_identitytail_task_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad010_identitytail_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B55a-SimpleFastTaskGeometry-h128-learnableP-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 128, "LineP_learnable_projection_quadboost_task_auc_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B56a-SimpleFastTaskGeometry-h96-learnableP-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 96, "LineP_quadboost_h96_timing_stability_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B56b-SimpleFastTaskGeometry-h112-learnableP-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 112, "LineP_quadboost_h112_expression_timing_midpoint", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B56c-SimpleFastTaskGeometry-h120-learnableP-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 120, "LineP_quadboost_h120_expression_timing_midpoint", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B57a-SimpleFastTaskGeometry-h112-learnableP-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 112, "LineP_identitytail_quad050_h112_expression_repair_with_timing_slack", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B57b-SimpleFastTaskGeometry-h116-learnableP-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 116, "LineP_identitytail_quad050_h116_A4_timing_midpoint", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B57c-SimpleFastTaskGeometry-h118-learnableP-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 118, "LineP_identitytail_quad050_h118_A4_timing_midpoint", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B58a-SimpleFastTaskGeometry-h128-fixedP-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic_fixedp", 2, 128, "LineP_fixedP_identitytail_quad050_task_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B59a-SimpleFastTaskGeometry-h116-learnableP-identitytailquad060-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 116, "LineP_identitytail_quad060_h116_early_interaction_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad060_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B59b-SimpleFastTaskGeometry-h116-learnableP-identitytailquad040-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 116, "LineP_identitytail_quad040_h116_early_interaction_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad040_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B60a-SimpleFastTaskGeometry-h116-learnableP-hinge050-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge050_direct_identitytail_quadratic", 2, 116, "LineP_hinge050_identitytail_quad050_h116_basis_shape_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge050_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B60b-SimpleFastTaskGeometry-h112-learnableP-hinge050-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge050_direct_identitytail_quadratic", 2, 112, "LineP_hinge050_identitytail_quad050_h112_timing_midpoint", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge050_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B61a-SimpleFastTaskGeometry-h112-twohinge-identitytailquad050-temp075", "SimpleFastTaskGeometry", "twohinge_direct_identitytail_quadratic", 2, 112, "LineP_twohinge_identitytail_quad050_h112_shape_and_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="twohinge_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B62a-SimpleFastTaskGeometry-h128-fixedP-signedpairlitequad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairlite_quadratic_fixedp", 2, 128, "LineP_signedpairlite_fixed_projection_frame_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairlite_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B62b-SimpleFastTaskGeometry-h128-learnableP-signedpairlitequad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairlite_quadratic", 2, 128, "LineP_signedpairlite_learnable_projection_init_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairlite_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B63a-SimpleFastTaskGeometry-h128-fixedP-signedpairtailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairtail_quadratic_fixedp", 2, 128, "LineP_signedpairlite_random_tail_projection_repair_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairtail_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B63b-SimpleFastTaskGeometry-h128-learnableP-signedpairtailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairtail_quadratic", 2, 128, "LineP_signedpairlite_random_tail_projection_repair_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairtail_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B64a-SimpleFastTaskGeometry-h128-fixedP-signedpairtail-boundq-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairtail_bounded_quadratic_fixedp", 2, 128, "LineP_signedpairtail_bounded_q_explosion_repair_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairtail_quad050_boundq_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B64b-SimpleFastTaskGeometry-h128-learnableP-signedpairtail-boundq-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairtail_bounded_quadratic", 2, 128, "LineP_signedpairtail_bounded_q_explosion_repair_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairtail_quad050_boundq_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B65a-SimpleFastTaskGeometry-h128-fixedP-rmsq-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_rmsq_quadratic_fixedp", 2, 128, "LineP_stopgrad_batch_rms_quadratic_trajectory_repair_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_rmsq_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B65b-SimpleFastTaskGeometry-h128-learnableP-rmsq-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_rmsq_quadratic", 2, 128, "LineP_stopgrad_batch_rms_quadratic_trajectory_repair_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_rmsq_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B66a-SimpleFastTaskGeometry-h128-fixedP-orthop-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_orthogonal_projection_quadratic_fixedp", 2, 128, "LineP_low_coherence_orthogonal_projection_frame_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_orthop_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B66b-SimpleFastTaskGeometry-h128-learnableP-orthop-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_orthogonal_projection_quadratic", 2, 128, "LineP_low_coherence_orthogonal_projection_frame_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_orthop_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B67a-SimpleFastTaskGeometry-h112-fixedP-orthop-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_orthogonal_projection_quadratic_fixedp", 2, 112, "LineP_B66_orthogonal_projection_efficiency_repair_fixedP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_orthop_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B67b-SimpleFastTaskGeometry-h112-learnableP-orthop-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_orthogonal_projection_quadratic", 2, 112, "LineP_B66_orthogonal_projection_efficiency_repair_learnableP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_orthop_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B68a-SimpleFastTaskGeometry-h128-fixedP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 128, "LineP_classwise_direct_quad_branch_scale_fixedP_tail_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B68b-SimpleFastTaskGeometry-h128-learnableP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 128, "LineP_classwise_direct_quad_branch_scale_learnableP_tail_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B69a-SimpleFastTaskGeometry-h120-fixedP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 120, "LineP_B68_classbranch_timing_repair_fixedP_h120", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B69b-SimpleFastTaskGeometry-h120-learnableP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 120, "LineP_B68_classbranch_timing_repair_learnableP_h120", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B69c-SimpleFastTaskGeometry-h112-fixedP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 112, "LineP_B68_classbranch_timing_repair_fixedP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B69d-SimpleFastTaskGeometry-h112-learnableP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 112, "LineP_B68_classbranch_timing_repair_learnableP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B70a-SimpleFastTaskGeometry-h124-fixedP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 124, "LineP_B69_classbranch_h124_accuracy_timing_midpoint_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B70b-SimpleFastTaskGeometry-h124-learnableP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 124, "LineP_B69_classbranch_h124_accuracy_timing_midpoint_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B71a-SimpleFastTaskGeometry-h126-fixedP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 126, "LineP_B70_classbranch_h126_accuracy_timing_midpoint_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B71b-SimpleFastTaskGeometry-h126-learnableP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 126, "LineP_B70_classbranch_h126_accuracy_timing_midpoint_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B72a-SimpleFastTaskGeometry-h126-fixedP-classbranch-quad040-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 126, "LineP_B71_classbranch_quad040_auc_tail_repair_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad040_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B72b-SimpleFastTaskGeometry-h126-learnableP-classbranch-quad040-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 126, "LineP_B71_classbranch_quad040_auc_tail_repair_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad040_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B73a-SimpleFastTaskGeometry-h112-fixedP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic_fixedp", 2, 112, "LineP_B71_sqdiag_direct_energy_classbranch_fixedP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B73b-SimpleFastTaskGeometry-h112-learnableP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic", 2, 112, "LineP_B71_sqdiag_direct_energy_classbranch_learnableP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B73c-SimpleFastTaskGeometry-h96-fixedP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic_fixedp", 2, 96, "LineP_B73_sqdiag_direct_energy_timing_repair_fixedP_h96", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B73d-SimpleFastTaskGeometry-h96-learnableP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic", 2, 96, "LineP_B73_sqdiag_direct_energy_timing_repair_learnableP_h96", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B74a-SimpleFastTaskGeometry-h116-fixedP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic_fixedp", 2, 116, "LineP_B73_sqdiag_expression_capacity_repair_fixedP_h116", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B74b-SimpleFastTaskGeometry-h116-learnableP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic", 2, 116, "LineP_B73_sqdiag_expression_capacity_repair_learnableP_h116", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B74c-SimpleFastTaskGeometry-h120-fixedP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic_fixedp", 2, 120, "LineP_B74_sqdiag_expression_capacity_repair_fixedP_h120", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B74d-SimpleFastTaskGeometry-h120-learnableP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic", 2, 120, "LineP_B74_sqdiag_expression_capacity_repair_learnableP_h120", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B75a-SimpleFastTaskGeometry-h112-fixedP-sqdiag025-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag025_classwise_branch_quadratic_fixedp", 2, 112, "LineP_B73_low_init_sqdiag_energy_fixedP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag025_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B75b-SimpleFastTaskGeometry-h112-learnableP-sqdiag025-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag025_classwise_branch_quadratic", 2, 112, "LineP_B73_low_init_sqdiag_energy_learnableP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag025_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B75c-SimpleFastTaskGeometry-h112-fixedP-sqdiag010-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag010_classwise_branch_quadratic_fixedp", 2, 112, "LineP_B75_lower_init_sqdiag_energy_fixedP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag010_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B75d-SimpleFastTaskGeometry-h112-learnableP-sqdiag010-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag010_classwise_branch_quadratic", 2, 112, "LineP_B75_lower_init_sqdiag_energy_learnableP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag010_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B76a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad040-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B71_classbranch_identitytail_direct_boost_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B76b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad040-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B71_classbranch_identitytail_direct_boost_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B76c-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B76_classbranch_identitytail_quad050_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B76d-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B76_classbranch_identitytail_quad050_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B77a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad030-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B76_classbranch_identitytail_quad030_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad030_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B77b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad030-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B76_classbranch_identitytail_quad030_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad030_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B77c-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad020-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B77_classbranch_identitytail_quad020_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad020_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B77d-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad020-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B77_classbranch_identitytail_quad020_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad020_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B78a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad040-hingeamp075-temp075", "SimpleFastTaskGeometry", "hingeamp075_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B76_hinge_tail_amplitude_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp075_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B78b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad040-hingeamp075-temp075", "SimpleFastTaskGeometry", "hingeamp075_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B76_hinge_tail_amplitude_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp075_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B78c-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad040-hingeamp050-temp075", "SimpleFastTaskGeometry", "hingeamp050_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B78_stronger_hinge_tail_amplitude_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B78d-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad040-hingeamp050-temp075", "SimpleFastTaskGeometry", "hingeamp050_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B78_stronger_hinge_tail_amplitude_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B79a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B78_aggressive_hinge_tail_amplitude_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B79b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B78_aggressive_hinge_tail_amplitude_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B80a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad040-hingeamp025-boundq-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_classwise_identitytail_bounded_quadratic_fixedp", 2, 126, "LineP_B79_bounded_quadratic_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp025_boundq_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B80b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad040-hingeamp025-boundq-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_classwise_identitytail_bounded_quadratic", 2, 126, "LineP_B79_bounded_quadratic_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp025_boundq_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B81a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytail150quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "direct150_hingeamp025_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B79_direct_branch_trajectory_boost_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytail150quad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B81b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytail150quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "direct150_hingeamp025_classwise_identitytail_quadratic", 2, 126, "LineP_B79_direct_branch_trajectory_boost_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytail150quad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B82a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identityamp150-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "raw_identity150_hingeamp025_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B79_raw_identity_only_trajectory_boost_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_identityamp150_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B82b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identityamp150-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "raw_identity150_hingeamp025_classwise_identitytail_quadratic", 2, 126, "LineP_B79_raw_identity_only_trajectory_boost_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_identityamp150_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B83a-SimpleFastTaskGeometry-h126-fixedP-absdiag025-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag025_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B79_centered_abs_magnitude_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag025_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B83b-SimpleFastTaskGeometry-h126-learnableP-absdiag025-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag025_classwise_identitytail_quadratic", 2, 126, "LineP_B79_centered_abs_magnitude_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag025_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B84a-SimpleFastTaskGeometry-h126-fixedP-absdiag010-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag010_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B83_lower_abs_magnitude_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag010_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B84b-SimpleFastTaskGeometry-h126-learnableP-absdiag010-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag010_classwise_identitytail_quadratic", 2, 126, "LineP_B83_lower_abs_magnitude_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag010_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B85a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B83_stronger_abs_magnitude_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B85b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic", 2, 126, "LineP_B83_stronger_abs_magnitude_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B86a-SimpleFastTaskGeometry-h126-fixedP-absdiag100-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_full_abs_magnitude_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B86b-SimpleFastTaskGeometry-h126-learnableP-absdiag100-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quadratic", 2, 126, "LineP_B85_full_abs_magnitude_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B87a-SimpleFastTaskGeometry-h126-fixedP-absdiag075-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag075_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_B86_abs_magnitude_midpoint_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag075_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B87b-SimpleFastTaskGeometry-h126-learnableP-absdiag075-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag075_classwise_identitytail_quadratic", 2, 126, "LineP_B85_B86_abs_magnitude_midpoint_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag075_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B88a-SimpleFastTaskGeometry-h126-fixedP-absdiag100-classbranch-identitytailquad050-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quad050_fixedp", 2, 126, "LineP_B86_abs_magnitude_with_stronger_quad_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad050_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B88b-SimpleFastTaskGeometry-h126-learnableP-absdiag100-classbranch-identitytailquad050-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quad050", 2, 126, "LineP_B86_abs_magnitude_with_stronger_quad_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad050_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B89a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad050-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quad050_fixedp", 2, 126, "LineP_B85_abs_magnitude_quad050_midpoint_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad050_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B89b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad050-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quad050", 2, 126, "LineP_B85_abs_magnitude_quad050_midpoint_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad050_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B90a-SimpleFastTaskGeometry-h126-fixedP-absdiag100-classbranch-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quad030_fixedp", 2, 126, "LineP_B86_full_abs_magnitude_lower_quad_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B90b-SimpleFastTaskGeometry-h126-learnableP-absdiag100-classbranch-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quad030", 2, 126, "LineP_B86_full_abs_magnitude_lower_quad_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B91a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quad030_fixedp", 2, 126, "LineP_B85_mid_abs_magnitude_lower_quad_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B91b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quad030", 2, 126, "LineP_B85_mid_abs_magnitude_lower_quad_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B92a-SimpleFastTaskGeometry-h126-fixedP-normabs050-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_normabs050_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_pooled_abs_magnitude_scalar_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_normabs050_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B92b-SimpleFastTaskGeometry-h126-learnableP-normabs050-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_normabs050_classwise_identitytail_quadratic", 2, 126, "LineP_B85_pooled_abs_magnitude_scalar_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_normabs050_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B93a-SimpleFastTaskGeometry-h126-fixedP-normabs100-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_normabs100_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B92_stronger_pooled_abs_scalar_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_normabs100_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B93b-SimpleFastTaskGeometry-h126-learnableP-normabs100-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_normabs100_classwise_identitytail_quadratic", 2, 126, "LineP_B92_stronger_pooled_abs_scalar_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_normabs100_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B94a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-normabs010-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs010_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_weak_pooled_abs_scalar_on_absdiag_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs010_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B94b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-normabs010-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs010_classwise_identitytail_quadratic", 2, 126, "LineP_B85_weak_pooled_abs_scalar_on_absdiag_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs010_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B95a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-normabs025-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs025_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B94_mid_pooled_abs_scalar_on_absdiag_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs025_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B95b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-normabs025-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs025_classwise_identitytail_quadratic", 2, 126, "LineP_B94_mid_pooled_abs_scalar_on_absdiag_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs025_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B96a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytail140quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail140_quadratic_fixedp", 2, 126, "LineP_B85_mild_direct_branch_boost_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail140quad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B96b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytail140quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail140_quadratic", 2, 126, "LineP_B85_mild_direct_branch_boost_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail140quad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B97a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytail145quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail145_quadratic_fixedp", 2, 126, "LineP_B96_mid_direct_branch_boost_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail145quad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B97b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytail145quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail145_quadratic", 2, 126, "LineP_B96_mid_direct_branch_boost_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail145quad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B98a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytail130quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail130_quadratic_fixedp", 2, 126, "LineP_B85_lower_direct_branch_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail130quad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B98b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytail130quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail130_quadratic", 2, 126, "LineP_B85_lower_direct_branch_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail130quad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B99a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytail125quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail125_quadratic_fixedp", 2, 126, "LineP_B98_lower_direct_branch_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail125quad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B99b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytail125quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail125_quadratic", 2, 126, "LineP_B98_lower_direct_branch_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail125quad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B100a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad040-hingeamp050-temp075", "SimpleFastTaskGeometry", "hingeamp050_direct_absdiag050_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_stronger_hinge_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B100b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad040-hingeamp050-temp075", "SimpleFastTaskGeometry", "hingeamp050_direct_absdiag050_classwise_identitytail_quadratic", 2, 126, "LineP_B85_stronger_hinge_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B101a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-signedpairlite-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairlite_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_structured_projection_trajectory_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairlite_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B101b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-signedpairlite-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairlite_classwise_identitytail_quadratic", 2, 126, "LineP_B85_structured_projection_trajectory_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairlite_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B102a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-signedpairtail-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairtail_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B101_structured_projection_random_tail_A4_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairtail_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B102b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-signedpairtail-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairtail_classwise_identitytail_quadratic", 2, 126, "LineP_B101_structured_projection_random_tail_A4_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairtail_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B103a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-signedpairtail-boundq-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairtail_boundq_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B102_bounded_projection_trajectory_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairtail_boundq_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B103b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-signedpairtail-boundq-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairtail_boundq_classwise_identitytail_quadratic", 2, 126, "LineP_B102_bounded_projection_trajectory_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairtail_boundq_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B104a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad040-hingeamp025-fixedgain-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic_fixedgain_fixedp", 2, 126, "LineP_B85_fixed_global_gain_trajectory_stabilizer_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_fixedgain_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B104b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad040-hingeamp025-fixedgain-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic_fixedgain", 2, 126, "LineP_B85_fixed_global_gain_trajectory_stabilizer_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_fixedgain_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B105a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad040-hingeamp025-temp100", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic_temp100_fixedp", 2, 126, "LineP_B85_higher_initial_gain_nll_trajectory_probe_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_temp100_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B105b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad040-hingeamp025-temp100", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic_temp100", 2, 126, "LineP_B85_higher_initial_gain_nll_trajectory_probe_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_temp100", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B106a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_classwise_gain_low_cost_auc_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B106b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic", 2, 126, "LineP_B85_classwise_gain_low_cost_auc_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B107a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic_fixedp_h160", 2, 160, "LineP_B106_kernel_headroom_capacity_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B107b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic_h160", 2, 160, "LineP_B106_kernel_headroom_capacity_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B108a-SimpleFastTaskGeometry-h144-fixedP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic_fixedp_h144", 2, 144, "LineP_B106_B107_capacity_midpoint_fixedP_h144", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B108b-SimpleFastTaskGeometry-h144-learnableP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic_h144", 2, 144, "LineP_B106_B107_capacity_midpoint_learnableP_h144", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B109a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B107_lower_quadratic_branch_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B109b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B107_lower_quadratic_branch_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B110a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quad020_fixedp_h160", 2, 160, "LineP_B109_lower_quadratic_branch_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad020_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B110b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quad020_h160", 2, 160, "LineP_B109_lower_quadratic_branch_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad020_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B111a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-fixedgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_fixed_classgain_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B111b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_fixed_classgain_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B112a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_global_contrast_auc_trajectory_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B112b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_global_contrast_auc_trajectory_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B113a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-normabs025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs025_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B112_lower_global_contrast_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs025_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B113b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-normabs025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs025_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B112_lower_global_contrast_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B114a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-normabs100-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs100_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B112_stronger_global_contrast_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs100_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B114b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-normabs100-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs100_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B112_stronger_global_contrast_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs100_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B115a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-normabs050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad040_fixedp_h126", 2, 126, "LineP_B106_global_contrast_auc_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B115b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-normabs050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad040_h126", 2, 126, "LineP_B106_global_contrast_auc_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B116a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_fixedp_h126", 2, 126, "LineP_B115_lower_quad_global_contrast_auc_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B116b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_h126", 2, 126, "LineP_B115_lower_quad_global_contrast_auc_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B117a-SimpleFastTaskGeometry-h128-fixedP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_fixedp_h128", 2, 128, "LineP_B116_block128_capacity_balance_auc_repair_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B117b-SimpleFastTaskGeometry-h128-learnableP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_h128", 2, 128, "LineP_B116_block128_capacity_balance_auc_repair_learnableP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B118a-SimpleFastTaskGeometry-h128-fixedP-absdiag050-normabs050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad040_fixedp_h128", 2, 128, "LineP_B117_quad040_seed2_auc_repair_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B118b-SimpleFastTaskGeometry-h128-learnableP-absdiag050-normabs050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad040_h128", 2, 128, "LineP_B117_quad040_seed2_auc_repair_learnableP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B119a-SimpleFastTaskGeometry-h128-fixedP-absdiag050-normabs050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_meanstat050_classwise_gain_identitytail_quad030_fixedp_h128", 2, 128, "LineP_B117_global_contrast_signed_mean_auc_repair_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B119b-SimpleFastTaskGeometry-h128-learnableP-absdiag050-normabs050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_meanstat050_classwise_gain_identitytail_quad030_h128", 2, 128, "LineP_B117_global_contrast_signed_mean_auc_repair_learnableP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B120a-SimpleFastTaskGeometry-h128-fixedP-absdiag050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_meanstat050_classwise_gain_identitytail_quad030_fixedp_h128", 2, 128, "LineP_B117_signed_mean_only_auc_repair_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B120b-SimpleFastTaskGeometry-h128-learnableP-absdiag050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_meanstat050_classwise_gain_identitytail_quad030_h128", 2, 128, "LineP_B117_signed_mean_only_auc_repair_learnableP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B121a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_meanstat050_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_signed_mean_bestline_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B121b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_meanstat050_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_signed_mean_bestline_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B122a-SimpleFastTaskGeometry-h128-fixedP-absdiag050-groupabs4-050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_050_classwise_gain_identitytail_quad030_fixedp_h128", 2, 128, "LineP_B117_grouped_abs_low_variance_auc_repair_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B122b-SimpleFastTaskGeometry-h128-learnableP-absdiag050-groupabs4-050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_050_classwise_gain_identitytail_quad030_h128", 2, 128, "LineP_B117_grouped_abs_low_variance_auc_repair_learnableP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B123a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-groupabs4-050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_050_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_grouped_abs_bestline_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B123b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-groupabs4-050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_050_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_grouped_abs_bestline_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B124a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_branch_gain_identitytail_quad020_fixedp_h160", 2, 160, "LineP_B110_fixed_branch_gain_CEp99_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_branch_gain_identitytail_quad020_h160", 2, 160, "LineP_B110_fixed_branch_gain_CEp99_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B125a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp050", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_branch_low_gain_identitytail_quad020_fixedp_h160", 2, 160, "LineP_B124_fixed_low_gain_CEp99_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp050_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B125b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp050", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_branch_low_gain_identitytail_quad020_h160", 2, 160, "LineP_B124_fixed_low_gain_CEp99_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp050", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B126a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-logitnorm150-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_logitnorm_fixed_branch_gain_identitytail_quad020_fixedp_h160", 2, 160, "LineP_B125_samplewise_logit_norm_CEp99_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_logitnorm150_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B126b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-logitnorm150-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_logitnorm_fixed_branch_gain_identitytail_quad020_h160", 2, 160, "LineP_B125_samplewise_logit_norm_CEp99_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_logitnorm150_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B127a-SimpleFastTaskGeometry-h160-fixedP-absquad050025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absquad050025_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_abs_plus_square_tail_single_kernel_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absquad050025_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B127b-SimpleFastTaskGeometry-h160-learnableP-absquad050025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absquad050025_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_abs_plus_square_tail_single_kernel_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absquad050025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B128a-SimpleFastTaskGeometry-h160-fixedP-absmixsq025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absmixsq025_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B127_single_tail_abs_square_mix_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absmixsq025_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B128b-SimpleFastTaskGeometry-h160-learnableP-absmixsq025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absmixsq025_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B127_single_tail_abs_square_mix_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absmixsq025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B129a-SimpleFastTaskGeometry-h160-fixedP-sqdiag025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_sqdiag025_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_centered_energy_tail_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag025_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B129b-SimpleFastTaskGeometry-h160-learnableP-sqdiag025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_sqdiag025_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_centered_energy_tail_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B130a-SimpleFastTaskGeometry-h160-fixedP-cubicdiag025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_cubicdiag025_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_signed_cubic_tail_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_cubicdiag025_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B130b-SimpleFastTaskGeometry-h160-learnableP-cubicdiag025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_cubicdiag025_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_signed_cubic_tail_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_cubicdiag025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B4a-FourierKAN-lowfreq-K4", "Fourier", "fourier_lowfreq", 4, h(4), "native+awesome-kan-family", 0, 1, 0, 1, 0, basis_order=4),
        PrimitiveSpec("B5c-RickerWaveletKAN-lite-K4", "Wavelet", "ricker_wavelet", 4, h(4), "native+wavelet-family", 1, 0, 1, 0, 0, basis_order=1),
        PrimitiveSpec("B6r-BSpline-order1-stream-K2-repair", "BSpline", "bspline_order1", 2, h(2), "R1_repair_stream_basis_mix+third_party/KANbeFair", 1, 0, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=1),
        PrimitiveSpec("B6a-BSpline-order1-local-K4", "BSpline", "bspline_order1", 4, h(4), "third_party/KANbeFair", 1, 0, 0, 0, 0, basis_order=1),
        PrimitiveSpec("B6b-BSpline-order1-local-K8-expression-repair", "BSpline", "bspline_order1", 8, h(8), "R2_repair_increase_K+third_party/KANbeFair", 1, 0, 0, 0, 0, basis_order=1),
        PrimitiveSpec("B8d-BSpline-ReLU-lite-combo-K4-repair", "Hybrid", "bspline_relu_combo", 4, h(4), "R2_repair_lite_hybrid", 1, 1, 0, 0, 0, basis_order=1),
        PrimitiveSpec("B9a-Poly2SignedPair-stream-K3-repair", "HybridPoly", "poly2_relu_combo", 3, h(3), "R5_base_repair_signed_pair_poly2", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="signed_pair_linear"),
        PrimitiveSpec("B9b-Poly2SignedPair-h64-diagnostic-repair", "HybridPoly", "poly2_relu_combo", 3, 64, "R2_repair_rank_limited_signed_pair_poly2_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, init_variant="signed_pair_linear"),
        PrimitiveSpec("B9c-Poly2SignedPair-h64-K2-diagnostic-repair", "HybridPoly", "poly2_relu_combo", 2, 64, "R2_repair_minimal_signed_pair_poly2_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, init_variant="signed_pair_linear"),
        PrimitiveSpec("B9d-Poly2PairRandom-h64-K2-diagnostic-repair", "HybridPoly", "poly2_relu_combo", 2, 64, "R2_repair_global_pair_random_poly2_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, init_variant="signed_pair_random_linear"),
        PrimitiveSpec("B10a-QuadraticSketch-h64-diagnostic", "QuadraticSketch", "quadratic_sketch", 2, 64, "R2_repair_fused_global_quadratic_sketch_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, model_kind="quadratic_sketch"),
        PrimitiveSpec("B10b-QuadraticSketch-h128-diagnostic", "QuadraticSketch", "quadratic_sketch", 2, 128, "R2_repair_fused_global_quadratic_sketch_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, model_kind="quadratic_sketch"),
        PrimitiveSpec("B10c-QuadraticSketch-h256-diagnostic", "QuadraticSketch", "quadratic_sketch", 2, 256, "R2_repair_high_rank_global_quadratic_sketch_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, model_kind="quadratic_sketch"),
        PrimitiveSpec("B10d-QuadraticSketch-h512-diagnostic", "QuadraticSketch", "quadratic_sketch", 2, 512, "R2_repair_high_rank_global_quadratic_sketch_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, model_kind="quadratic_sketch"),
        PrimitiveSpec("B11a-TrainableQuadraticSketch-h128", "QuadraticSketch", "quadratic_sketch", 2, 128, "R2_repair_trainable_global_quadratic_sketch", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=0, basis_order=2, model_kind="trainable_quadratic_sketch"),
        PrimitiveSpec("B11b-TrainableQuadraticSketch-h256", "QuadraticSketch", "quadratic_sketch", 2, 256, "R2_repair_trainable_global_quadratic_sketch", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=0, basis_order=2, model_kind="trainable_quadratic_sketch"),
        PrimitiveSpec("B7a-RationalKAT-lite-safe-den-K4", "Rational", "rational_kat_lite", 4, h(4), "third_party/rational_kat_cu", 0, 1, 0, 0, 1, basis_order=4),
    ]


def basis_functional_direction(module: nn.Module, mode: str) -> List[torch.Tensor]:
    if hasattr(module, "basis_functional_direction") and not isinstance(module, PrimitiveKAN):
        return module.basis_functional_direction(mode)  # type: ignore[no-any-return]
    grads: List[torch.Tensor] = []
    with torch.no_grad():
        for p in [module.w1, module.w2]:
            direction = -p.detach().clone()
            if mode in {"basis_aware_snr_projected", "basis_aware_orthogonal"}:
                # Damp only the non-linear channels and keep the first channel
                # closer to task descent; this is a basis-level diagnostic, not
                # an official controller.
                mask = torch.ones_like(direction)
                mask[..., 0] = 0.25
                direction = direction * mask
            grads.append(direction)
    return grads
