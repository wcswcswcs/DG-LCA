"""Train-only one-dimensional domain transport for KAN edge functions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dgkan.fu.kan_edge_function_metric import wasserstein1_1d


EPS = 1.0e-12


def _as_1d(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1).to(dtype=torch.float64)


def _dedup_knots(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(x)
    xs = x.index_select(0, order)
    ys = y.index_select(0, order)
    keep_x = []
    keep_y = []
    i = 0
    while i < int(xs.numel()):
        j = i + 1
        while j < int(xs.numel()) and abs(float(xs[j] - xs[i])) < 1.0e-10:
            j += 1
        keep_x.append(xs[i:j].mean())
        keep_y.append(ys[i:j].mean())
        i = j
    return torch.stack(keep_x), torch.stack(keep_y)


def _linear_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    xx = x.to(dtype=torch.float64)
    idx = torch.searchsorted(xp, xx.clamp(float(xp[0]), float(xp[-1])), right=True) - 1
    idx = idx.clamp(0, int(xp.numel()) - 2)
    x0 = xp.index_select(0, idx)
    x1 = xp.index_select(0, idx + 1)
    y0 = fp.index_select(0, idx)
    y1 = fp.index_select(0, idx + 1)
    t = (xx - x0) / (x1 - x0).clamp_min(EPS)
    return y0 + t * (y1 - y0)


def _pchip_slopes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    h = (x[1:] - x[:-1]).clamp_min(EPS)
    delta = (y[1:] - y[:-1]) / h
    m = torch.zeros_like(y)
    if int(y.numel()) == 2:
        m[:] = delta[0]
        return m
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, int(y.numel()) - 1):
        if float(delta[i - 1] * delta[i]) <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    return m


def _pchip_eval(xq: torch.Tensor, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    xx = xq.to(dtype=torch.float64).clamp(float(x[0]), float(x[-1]))
    idx = torch.searchsorted(x, xx, right=True) - 1
    idx = idx.clamp(0, int(x.numel()) - 2)
    x0 = x.index_select(0, idx)
    x1 = x.index_select(0, idx + 1)
    y0 = y.index_select(0, idx)
    y1 = y.index_select(0, idx + 1)
    m0 = m.index_select(0, idx)
    m1 = m.index_select(0, idx + 1)
    h = (x1 - x0).clamp_min(EPS)
    t = (xx - x0) / h
    h00 = 2 * t.pow(3) - 3 * t.pow(2) + 1
    h10 = t.pow(3) - 2 * t.pow(2) + t
    h01 = -2 * t.pow(3) + 3 * t.pow(2)
    h11 = t.pow(3) - t.pow(2)
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


@dataclass
class EdgeDomainTransport:
    kind: str
    old_knots: torch.Tensor
    new_knots: torch.Tensor
    old_mean: float = 0.0
    old_std: float = 1.0
    new_mean: float = 0.0
    new_std: float = 1.0
    forward_slopes: torch.Tensor | None = None
    inverse_slopes: torch.Tensor | None = None

    def forward(self, old_u: torch.Tensor) -> torch.Tensor:
        x = _as_1d(old_u)
        if self.kind == "affine":
            return (x - self.old_mean) / max(self.old_std, EPS) * max(self.new_std, EPS) + self.new_mean
        if self.kind == "monotone_spline" and self.forward_slopes is not None:
            return _pchip_eval(x, self.old_knots, self.new_knots, self.forward_slopes)
        return _linear_interp(x, self.old_knots, self.new_knots)

    def inverse(self, new_u: torch.Tensor) -> torch.Tensor:
        x = _as_1d(new_u)
        if self.kind == "affine":
            return (x - self.new_mean) / max(self.new_std, EPS) * max(self.old_std, EPS) + self.old_mean
        if self.kind == "monotone_spline" and self.inverse_slopes is not None:
            return _pchip_eval(x, self.new_knots, self.old_knots, self.inverse_slopes)
        return _linear_interp(x, self.new_knots, self.old_knots)

    def monotonicity_violation(self) -> int:
        return int(((self.new_knots[1:] - self.new_knots[:-1]) < -1.0e-10).sum().detach().cpu().item())

    def inverse_error(self, samples: torch.Tensor) -> float:
        old = _as_1d(samples)
        recovered = self.inverse(self.forward(old))
        return float((old - recovered).abs().max().detach().cpu().item())


def fit_transport(old_u: torch.Tensor, new_u: torch.Tensor, kind: str = "quantile", num_knots: int = 513) -> EdgeDomainTransport:
    old = _as_1d(old_u)
    new = _as_1d(new_u)
    if int(old.numel()) < 2 or int(new.numel()) < 2:
        old = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        new = old.clone()
    kind_s = str(kind)
    old_mean = float(old.mean().detach().cpu().item())
    new_mean = float(new.mean().detach().cpu().item())
    old_std = float(old.std(unbiased=False).clamp_min(1.0e-6).detach().cpu().item())
    new_std = float(new.std(unbiased=False).clamp_min(1.0e-6).detach().cpu().item())
    if kind_s == "affine":
        return EdgeDomainTransport("affine", torch.stack([old.min(), old.max()]), torch.stack([new.min(), new.max()]), old_mean, old_std, new_mean, new_std)
    probs = torch.linspace(0.0, 1.0, max(3, int(num_knots)), dtype=torch.float64, device=old.device)
    old_q = torch.quantile(old, probs)
    new_q = torch.quantile(new, probs)
    old_q, new_q = _dedup_knots(old_q, new_q)
    if int(old_q.numel()) < 2:
        old_q = torch.tensor([float(old.min()), float(old.max()) + 1.0e-6], dtype=torch.float64, device=old.device)
        new_q = torch.tensor([float(new.min()), float(new.max()) + 1.0e-6], dtype=torch.float64, device=old.device)
    fwd_slopes = inv_slopes = None
    if kind_s == "monotone_spline":
        fwd_slopes = _pchip_slopes(old_q, new_q)
        inv_slopes = _pchip_slopes(new_q, old_q)
    return EdgeDomainTransport(kind_s, old_q, new_q, old_mean, old_std, new_mean, new_std, fwd_slopes, inv_slopes)


def transport_diagnostics(old_u: torch.Tensor, new_u: torch.Tensor, kind: str) -> dict[str, float | str]:
    transport = fit_transport(old_u, new_u, kind)
    mapped = transport.forward(old_u)
    before = wasserstein1_1d(old_u, new_u)
    after = wasserstein1_1d(mapped, new_u)
    return {
        "transport_type": kind,
        "transport_monotonicity_violation": float(transport.monotonicity_violation()),
        "transport_inverse_error": transport.inverse_error(old_u),
        "domain_wasserstein_before": before,
        "domain_wasserstein_after": after,
    }
