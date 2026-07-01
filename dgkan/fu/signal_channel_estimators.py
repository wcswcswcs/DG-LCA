"""Train-only signal-channel estimators for v22.96."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from dgkan.fu.layer_composite_metric import EPS, sym


def _f(value: torch.Tensor | float) -> float:
    out = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
    return out if math.isfinite(out) else 0.0


def flatten_params_or_grads(tensors: Iterable[torch.Tensor | None], params: list[torch.nn.Parameter]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for tensor, param in zip(tensors, params):
        if tensor is None:
            parts.append(torch.zeros_like(param.detach()).reshape(-1).to(dtype=torch.float64))
        else:
            parts.append(tensor.detach().reshape(-1).to(dtype=torch.float64))
    return torch.cat(parts).to(dtype=torch.float64) if parts else torch.empty(0, dtype=torch.float64)


def output_values(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor | None = None, mode: str = "margin") -> torch.Tensor:
    logits = model(x)
    if str(mode) == "margin":
        if y is None:
            raise ValueError("margin output mode requires labels")
        yy = y.long().reshape(-1, 1)
        true_logit = logits.gather(1, yy).reshape(-1)
        if int(logits.shape[1]) <= 1:
            return true_logit
        others = logits.float().clone()
        others.scatter_(1, yy, float("-inf"))
        return true_logit - others.max(dim=1).values
    return logits.reshape(-1)


def _sketch_values(values: torch.Tensor, max_outputs: int | None, *, sketch_mode: str = "prefix", sketch_seed: int = 0) -> torch.Tensor:
    flat = values.reshape(-1)
    if max_outputs is None or int(flat.numel()) <= int(max_outputs):
        return flat
    k = int(max_outputs)
    if str(sketch_mode) == "rademacher":
        gen = torch.Generator(device="cpu").manual_seed(int(sketch_seed))
        signs = torch.randint(0, 2, (k, int(flat.numel())), generator=gen, device="cpu")
        probes = signs.to(device=flat.device, dtype=flat.dtype).mul(2.0).sub(1.0)
        probes = probes / math.sqrt(float(flat.numel()))
        return probes @ flat
    return flat[:k]


def full_output_jacobian(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    max_outputs: int | None = None,
    mode: str = "margin",
    sketch_mode: str = "prefix",
    sketch_seed: int = 0,
) -> torch.Tensor:
    params = list(getattr(model, "coeffs", list(model.parameters())))
    values = _sketch_values(output_values(model, x, y, mode=mode), max_outputs, sketch_mode=sketch_mode, sketch_seed=int(sketch_seed))
    rows: list[torch.Tensor] = []
    for idx in range(int(values.numel())):
        grads = torch.autograd.grad(values[idx], params, retain_graph=True, allow_unused=True)
        rows.append(flatten_params_or_grads(grads, params).detach().cpu())
    if not rows:
        return torch.zeros((0, 0), dtype=torch.float64)
    return torch.stack(rows, dim=0).to(dtype=torch.float64)


def uniform_label_loss(logits: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits.float(), dim=1).mean(dim=1).mean()


def loss_gradient_vector(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, label_prior_correction: bool = False) -> torch.Tensor:
    params = list(getattr(model, "coeffs", list(model.parameters())))
    logits = model(x)
    loss = F.cross_entropy(logits.float(), y.long())
    grads = torch.autograd.grad(loss, params, retain_graph=bool(label_prior_correction), allow_unused=True)
    g = flatten_params_or_grads(grads, params)
    if label_prior_correction:
        null_logits = model(x)
        null_loss = uniform_label_loss(null_logits)
        null_grads = torch.autograd.grad(null_loss, params, retain_graph=False, allow_unused=True)
        g = g - flatten_params_or_grads(null_grads, params)
    return g.to(dtype=torch.float64)


def per_example_gradient_matrix(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, max_examples: int, label_prior_correction: bool = False) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    n = min(int(max_examples), int(x.shape[0]))
    for idx in range(n):
        rows.append(loss_gradient_vector(model, x[idx : idx + 1], y[idx : idx + 1], label_prior_correction=label_prior_correction).detach().cpu())
    return torch.stack(rows, dim=0).to(dtype=torch.float64) if rows else torch.zeros((0, 0), dtype=torch.float64)


def projector_overlap(u: torch.Tensor, v: torch.Tensor) -> float:
    if int(u.numel()) == 0 or int(v.numel()) == 0:
        return 0.0
    r = min(int(u.shape[1]), int(v.shape[1]))
    if r <= 0:
        return 0.0
    val = torch.linalg.norm(u[:, :r].T @ v[:, :r], ord="fro").square() / float(r)
    return _f(val.clamp(0.0, 1.0))


def vector_channel_fraction(u: torch.Tensor, vec: torch.Tensor) -> float:
    if int(u.numel()) == 0 or int(vec.numel()) == 0:
        return 0.0
    vv = vec.detach().reshape(-1).to(dtype=torch.float64).cpu()
    n = min(int(vv.numel()), int(u.shape[0]))
    denom = vv[:n].square().sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return 0.0
    return _f((u[:n].T @ vv[:n]).square().sum().div(denom).clamp(0.0, 1.0))


@dataclass
class ChannelProjector:
    u: torch.Tensor
    eigenvalues: torch.Tensor
    rank: int
    summary: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def projector(self) -> torch.Tensor:
        if int(self.u.numel()) == 0:
            return torch.zeros((0, 0), dtype=torch.float64)
        return self.u @ self.u.T


def projector_from_psd(w: torch.Tensor, rank: int, *, estimator_type: str) -> ChannelProjector:
    ww = sym(w).to(dtype=torch.float64).cpu()
    if int(ww.numel()) == 0:
        return ChannelProjector(torch.zeros((0, 0), dtype=torch.float64), torch.zeros(0, dtype=torch.float64), 0, {"estimator_type": estimator_type})
    vals, vecs = torch.linalg.eigh(ww)
    order = torch.argsort(vals.abs(), descending=True)
    vals = vals[order]
    vecs = vecs[:, order]
    r = min(max(1, int(rank)), int(vecs.shape[1]))
    pos = vals.clamp_min(0.0)
    trace = float(pos.sum().item())
    top_mass = float(pos[:r].sum().item()) / max(trace, EPS)
    probs = pos / max(trace, EPS)
    eff = _f(torch.exp(-(probs * probs.clamp_min(EPS).log()).sum())) if trace > 0.0 else 0.0
    summary = {
        "estimator_type": estimator_type,
        "output_dim": int(ww.shape[0]),
        "W_trace_estimate": trace,
        "W_top1": float(vals[0].item()) if int(vals.numel()) else 0.0,
        "W_top_mass_ratio": top_mass,
        "W_effective_rank": eff,
        "PSD_violation_min_eig": float(vals.min().item()) if int(vals.numel()) else 0.0,
        "signal_channel_projector_rank": int(r),
    }
    return ChannelProjector(vecs[:, :r].contiguous(), vals.contiguous(), int(r), summary)


class VelocityProxyChannelEstimator:
    def estimate(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        metric_diag: torch.Tensor,
        *,
        rank: int = 2,
        max_outputs: int = 96,
        cohorts: int = 8,
        output_mode: str = "margin",
        label_prior_correction: bool = False,
        output_sketch_mode: str = "prefix",
        output_sketch_seed: int = 0,
    ) -> tuple[ChannelProjector, torch.Tensor, torch.Tensor]:
        j = full_output_jacobian(
            model,
            x,
            y,
            max_outputs=int(max_outputs),
            mode=output_mode,
            sketch_mode=output_sketch_mode,
            sketch_seed=int(output_sketch_seed),
        )
        m = metric_diag.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(EPS)
        n_dim = min(int(j.shape[1]), int(m.numel()))
        chunks = torch.chunk(torch.arange(int(x.shape[0]), device=x.device), max(1, min(int(cohorts), int(x.shape[0]))))
        cols: list[torch.Tensor] = []
        for chunk in chunks:
            if int(chunk.numel()) == 0:
                continue
            g = loss_gradient_vector(model, x[chunk], y[chunk], label_prior_correction=label_prior_correction).detach().cpu()
            cols.append(-(j[:, :n_dim] @ (m[:n_dim] * g[:n_dim])))
        v = torch.stack(cols, dim=1).to(dtype=torch.float64) if cols else torch.zeros((int(j.shape[0]), 1), dtype=torch.float64)
        w = sym(v @ v.T / float(max(1, int(v.shape[1]))))
        proj = projector_from_psd(w, rank, estimator_type="velocity_proxy")
        proj.summary["velocity_cohort_count"] = int(v.shape[1])
        proj.summary["velocity_norm_median"] = _median([float(v[:, idx].norm().item()) for idx in range(int(v.shape[1]))])
        return proj, j, v


class WindowedDissipationEstimator:
    def estimate_from_jacobians(
        self,
        jacobians: list[torch.Tensor],
        metric_diags: list[torch.Tensor],
        *,
        rank: int = 2,
        weights: list[float] | None = None,
        estimator_type: str = "window_no_pg",
        center_mode: str = "none",
        output_center_mode: str = "none",
    ) -> ChannelProjector:
        if not jacobians:
            return projector_from_psd(torch.zeros((1, 1), dtype=torch.float64), rank, estimator_type=estimator_type)
        out_dim = int(jacobians[0].shape[0])
        w = torch.zeros((out_dim, out_dim), dtype=torch.float64)
        ws = weights or [1.0] * len(jacobians)
        base_j = jacobians[0].detach().to(dtype=torch.float64).cpu()
        for j, m, weight in zip(jacobians, metric_diags, ws):
            jj = j.detach().to(dtype=torch.float64).cpu()
            if str(center_mode) == "delta_j0":
                jj = jj - base_j[: int(jj.shape[0]), : int(jj.shape[1])]
            if str(output_center_mode) == "row_mean" and int(jj.shape[0]) > 1:
                jj = jj - jj.mean(dim=0, keepdim=True)
            mm = m.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(EPS)
            n = min(int(jj.shape[1]), int(mm.numel()))
            w = w + float(weight) * sym((jj[:, :n] * mm[:n].reshape(1, -1)) @ jj[:, :n].T)
        proj = projector_from_psd(w, rank, estimator_type=estimator_type)
        proj.summary["checkpoint_count"] = int(len(jacobians))
        proj.summary["window_weight_sum"] = float(sum(ws))
        proj.summary["window_center_mode"] = str(center_mode)
        proj.summary["window_output_center_mode"] = str(output_center_mode)
        return proj

    def estimate_pg1_reduced(
        self,
        jacobians: list[torch.Tensor],
        metric_diags: list[torch.Tensor],
        *,
        rank: int = 2,
        gamma: float = 0.15,
        center_mode: str = "none",
        output_center_mode: str = "none",
    ) -> ChannelProjector:
        base = self.estimate_from_jacobians(
            jacobians,
            metric_diags,
            rank=rank,
            estimator_type="window_pg1",
            center_mode=center_mode,
            output_center_mode=output_center_mode,
        )
        if int(base.u.numel()) == 0:
            return base
        p = base.projector
        w = p + float(gamma) * sym(p @ p)
        proj = projector_from_psd(w, rank, estimator_type="window_pg1")
        proj.summary["checkpoint_count"] = int(len(jacobians))
        proj.summary["pg1_gamma"] = float(gamma)
        proj.summary["window_output_center_mode"] = str(output_center_mode)
        return proj


def _median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def diagonal_snr_gate(g: torch.Tensor, *, beta: float = 4.0, tau: float = 1.0, normalize_rows: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
    if int(g.numel()) == 0 or int(g.shape[0]) <= 1:
        return torch.zeros((int(g.shape[1]) if int(g.ndim) == 2 else 0,), dtype=torch.float64), {"diagonal_snr_median": 0.0}
    gg = g.detach().to(dtype=torch.float64)
    if normalize_rows:
        gg = gg / gg.norm(dim=1, keepdim=True).clamp_min(EPS)
    mu = gg.mean(dim=0)
    var = gg.var(dim=0, unbiased=False)
    snr = mu.square() / (var / float(max(1, int(gg.shape[0]) - 1)) + EPS)
    log_snr = snr.clamp_min(EPS).log()
    gate = torch.sigmoid(float(beta) * (log_snr - math.log(float(tau))))
    sorted_snr = torch.sort(snr).values
    k = max(1, int(math.ceil(0.10 * int(sorted_snr.numel()))))
    signal = float(mu.square().sum().item())
    noise = float((var / float(max(1, int(gg.shape[0]) - 1))).sum().item())
    return gate.to(dtype=torch.float64), {
        "diagonal_snr_median": float(torch.median(snr).item()),
        "diagonal_snr_top_decile_mean": float(sorted_snr[-k:].mean().item()),
        "AB_positive_trace_fraction": signal / max(signal + noise, EPS),
        "AB_negative_trace_fraction": noise / max(signal + noise, EPS),
        "snr_gate_density": float((gate > 0.5).to(dtype=torch.float64).mean().item()),
        "snr_gate_mean": float(gate.mean().item()),
    }


class OffDiagonalAgreementEstimator:
    def estimate(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        metric_diag: torch.Tensor,
        *,
        rank: int = 2,
        max_examples: int = 24,
        max_outputs: int = 96,
        output_mode: str = "margin",
        label_prior_correction: bool = False,
        layer_spans: list[tuple[int, int, int]] | None = None,
        output_sketch_mode: str = "prefix",
        output_sketch_seed: int = 0,
    ) -> tuple[ChannelProjector, torch.Tensor, torch.Tensor, dict[str, float]]:
        j = full_output_jacobian(
            model,
            x,
            y,
            max_outputs=int(max_outputs),
            mode=output_mode,
            sketch_mode=output_sketch_mode,
            sketch_seed=int(output_sketch_seed),
        )
        g = per_example_gradient_matrix(model, x, y, max_examples=int(max_examples), label_prior_correction=label_prior_correction)
        m = metric_diag.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(EPS)
        n = min(int(g.shape[1]), int(m.numel()), int(j.shape[1]))
        gw = g[:, :n] * m[:n].sqrt().reshape(1, -1)
        gate, info = diagonal_snr_gate(gw, beta=4.0, tau=1.0, normalize_rows=True)
        mu = gw.mean(dim=0)
        centered = gw - mu.reshape(1, -1)
        cov = centered.T @ centered / float(max(1, int(gw.shape[0]) - 1))
        ab = sym(mu.reshape(-1, 1) @ mu.reshape(1, -1) - cov / float(max(1, int(gw.shape[0]) - 1)))
        vals, vecs = torch.linalg.eigh(ab)
        order = torch.argsort(vals, descending=True)
        vals = vals[order]
        vecs = vecs[:, order]
        pos = vals.clamp_min(0.0)
        ab_pos = sym((vecs * pos.reshape(1, -1)) @ vecs.T)
        diag_w = sym((j[:, :n] * gate[:n].reshape(1, -1)) @ j[:, :n].T)
        lowrank_w = sym(j[:, :n] @ ab_pos @ j[:, :n].T)
        proj = projector_from_psd(diag_w + lowrank_w, rank, estimator_type="offdiag_population_risk")
        info.update(self.block_snr(gw, layer_spans or []))
        info["offdiag_lowrank_positive_eigs"] = int((vals > 0.0).sum().item())
        info["offdiag_ab_top_eig"] = float(vals[0].item()) if int(vals.numel()) else 0.0
        proj.summary.update(info)
        return proj, j, g, info

    def block_snr(self, g: torch.Tensor, layer_spans: list[tuple[int, int, int]]) -> dict[str, float]:
        out: dict[str, float] = {}
        vals: list[float] = []
        for layer_idx, start, end in layer_spans:
            block = g[:, int(start) : int(end)]
            if int(block.numel()) == 0:
                continue
            mu = block.mean(dim=0)
            var = block.var(dim=0, unbiased=False)
            snr = float(mu.square().sum().div(var.sum() / float(max(1, int(block.shape[0]) - 1)) + EPS).item())
            out[f"layer{int(layer_idx)}_block_snr"] = snr
            vals.append(snr)
        out["layer_block_snr"] = _median(vals)
        return out


def windowed_linear_smoke_test() -> dict[str, float]:
    j = torch.tensor([[1.0, 2.0], [0.5, -1.0]], dtype=torch.float64)
    m = torch.tensor([2.0, 3.0], dtype=torch.float64)
    h = 4
    est = WindowedDissipationEstimator()
    proj = est.estimate_from_jacobians([j] * h, [m] * h, rank=1)
    expected = h * sym((j * m.reshape(1, -1)) @ j.T)
    got_trace = float(proj.summary.get("W_trace_estimate", 0.0))
    return {
        "windowed_linear_trace_error": abs(got_trace - float(torch.trace(expected).item())),
        "windowed_linear_expected_trace": float(torch.trace(expected).item()),
    }


def offdiag_synthetic_smoke_test() -> dict[str, float]:
    base = torch.ones((16, 8), dtype=torch.float64)
    same_gate, same = diagonal_snr_gate(base, normalize_rows=False)
    signs = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(16)], dtype=torch.float64).reshape(-1, 1)
    flip_gate, flip = diagonal_snr_gate(base * signs, normalize_rows=False)
    return {
        "offdiag_same_direction_snr": float(same.get("diagonal_snr_median", 0.0)),
        "offdiag_random_sign_snr": float(flip.get("diagonal_snr_median", 0.0)),
        "offdiag_same_gt_random": float(float(same.get("diagonal_snr_median", 0.0)) > float(flip.get("diagonal_snr_median", 0.0))),
        "offdiag_same_gate_mean": float(same_gate.mean().item()) if int(same_gate.numel()) else 0.0,
        "offdiag_flip_gate_mean": float(flip_gate.mean().item()) if int(flip_gate.numel()) else 0.0,
    }


__all__ = [
    "ChannelProjector",
    "OffDiagonalAgreementEstimator",
    "VelocityProxyChannelEstimator",
    "WindowedDissipationEstimator",
    "diagonal_snr_gate",
    "full_output_jacobian",
    "loss_gradient_vector",
    "offdiag_synthetic_smoke_test",
    "per_example_gradient_matrix",
    "projector_from_psd",
    "projector_overlap",
    "vector_channel_fraction",
    "windowed_linear_smoke_test",
]
