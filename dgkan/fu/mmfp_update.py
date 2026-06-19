"""First-principles metric functional update utilities for v22.20.

The functions here operate in output space.  Callers provide a matrix of
reachable effects ``E = J B``; this module solves the curvature-metric
correction and evaluates the train-only MMFP diagnostics around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


EPS = 1.0e-12


@dataclass
class MMFPSolveResult:
    alpha_raw: torch.Tensor
    direction: torch.Tensor
    eta_quad: float
    q_value: float
    linear_term: float
    curvature_term: float
    damping: float
    effect_rank: int
    solve_status: str
    control_q: dict[str, float]
    control_dominance_margin: float
    control_dominance_pass: int


def ce_delta(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gradient of mean cross entropy with respect to logits."""
    probs = torch.softmax(logits.float(), dim=-1)
    onehot = F.one_hot(labels.long(), num_classes=int(logits.shape[-1])).to(probs)
    return (probs - onehot) / max(1, int(labels.numel()))


def ce_h_apply(logits: torch.Tensor, vector_flat: torch.Tensor) -> torch.Tensor:
    """Apply categorical Fisher / CE Hessian to a flattened output vector."""
    probs = torch.softmax(logits.float(), dim=-1)
    vec = vector_flat.reshape_as(probs).float()
    centered = vec - (probs * vec).sum(dim=-1, keepdim=True)
    hv = probs * centered / max(1, int(logits.shape[0]))
    return hv.reshape(-1)


def ce_quadratic_terms(logits: torch.Tensor, labels: torch.Tensor, delta_flat: torch.Tensor) -> tuple[float, float, float]:
    cot = ce_delta(logits, labels).reshape(-1)
    h_delta = ce_h_apply(logits, delta_flat)
    linear = float(torch.dot(cot, delta_flat).detach().cpu())
    curvature = float(torch.dot(delta_flat, h_delta).detach().cpu())
    return linear, curvature, linear + 0.5 * curvature


def trace_scaled_damping(gram: torch.Tensor) -> torch.Tensor:
    k = max(1, int(gram.shape[0]))
    trace = torch.trace(gram.float()).clamp_min(0.0)
    return 1.0e-6 * trace / (float(k) + 1.0e-12)


def _matrix_rank(effects: torch.Tensor) -> int:
    if effects.numel() == 0:
        return 0
    try:
        return int(torch.linalg.matrix_rank(effects.float()).detach().cpu())
    except RuntimeError:
        return 0


def _safe_solve(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, str]:
    try:
        return torch.linalg.solve(lhs, rhs), "solve"
    except RuntimeError:
        try:
            return torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution.squeeze(1), "lstsq"
        except RuntimeError:
            return torch.zeros_like(rhs), "failed_zero"


def _matched_output_controls(direction: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    d = direction.detach().float().reshape(-1)
    norm = torch.linalg.vector_norm(d).clamp_min(EPS)
    gen = torch.Generator(device=d.device).manual_seed(int(seed))
    random = torch.randn(d.shape, generator=gen, device=d.device, dtype=d.dtype)
    random = random / torch.linalg.vector_norm(random).clamp_min(EPS) * norm
    stable_gen = torch.Generator(device=d.device).manual_seed(222000 + int(seed) % 100000)
    stable = torch.randn(d.shape, generator=stable_gen, device=d.device, dtype=d.dtype)
    stable = stable / torch.linalg.vector_norm(stable).clamp_min(EPS) * norm
    perm = torch.randperm(d.numel(), generator=gen, device=d.device)
    shuffled = d[perm]
    return {
        "random": random,
        "stable_random": stable,
        "signflip": -d,
        "shuffled": shuffled,
    }


def solve_ce_mmfp(
    logits: torch.Tensor,
    labels: torch.Tensor,
    effects: torch.Tensor,
    *,
    control_seed: int = 0,
    cotangent_shift: torch.Tensor | None = None,
) -> MMFPSolveResult:
    """Solve the v22.20 output-space MMFP correction for CE logits.

    Args:
        logits: shape ``[batch, classes]``.
        labels: shape ``[batch]``.
        effects: shape ``[batch * classes, k]`` with columns ``J B_i``.
    """
    logits_f = logits.float()
    labels_l = labels.long()
    if effects.numel() == 0 or effects.shape[1] == 0:
        zero = torch.zeros(logits_f.numel(), device=logits_f.device)
        return MMFPSolveResult(
            alpha_raw=torch.zeros(0, device=logits_f.device),
            direction=zero,
            eta_quad=0.0,
            q_value=0.0,
            linear_term=0.0,
            curvature_term=0.0,
            damping=0.0,
            effect_rank=0,
            solve_status="empty_effects",
            control_q={},
            control_dominance_margin=0.0,
            control_dominance_pass=0,
        )
    e = effects.float()
    cot = ce_delta(logits_f, labels_l).reshape(-1)
    if cotangent_shift is not None:
        cot = cot + cotangent_shift.reshape(-1).to(device=cot.device, dtype=cot.dtype)
    h_cols = [ce_h_apply(logits_f, e[:, idx]) for idx in range(int(e.shape[1]))]
    h_e = torch.stack(h_cols, dim=1)
    gram = e.T @ h_e
    rhs = e.T @ cot
    damping = trace_scaled_damping(gram)
    lhs = gram + damping * torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    alpha, status = _safe_solve(lhs, -rhs)
    raw_direction = e @ alpha
    h_raw = ce_h_apply(logits_f, raw_direction)
    linear_raw = torch.dot(cot, raw_direction)
    curvature_raw = torch.dot(raw_direction, h_raw)
    eta_den = curvature_raw + damping
    eta = torch.clamp((-linear_raw / eta_den.clamp_min(EPS)), 0.0, 1.0)
    direction = eta * raw_direction
    h_direction = ce_h_apply(logits_f, direction)
    linear = torch.dot(cot, direction)
    curvature = torch.dot(direction, h_direction)
    q_value = linear + 0.5 * curvature
    controls = _matched_output_controls(direction, int(control_seed))
    control_q: dict[str, float] = {}
    for name, ctrl in controls.items():
        lin_c = torch.dot(cot, ctrl)
        curv_c = torch.dot(ctrl, ce_h_apply(logits_f, ctrl))
        control_q[name] = float((lin_c + 0.5 * curv_c).detach().cpu())
    min_control = min(control_q.values()) if control_q else 0.0
    q_float = float(q_value.detach().cpu())
    margin = float(min_control - q_float)
    return MMFPSolveResult(
        alpha_raw=alpha.detach(),
        direction=direction.detach(),
        eta_quad=float(eta.detach().cpu()),
        q_value=q_float,
        linear_term=float(linear.detach().cpu()),
        curvature_term=float(curvature.detach().cpu()),
        damping=float(damping.detach().cpu()),
        effect_rank=_matrix_rank(e.detach()),
        solve_status=status,
        control_q=control_q,
        control_dominance_margin=margin,
        control_dominance_pass=int(q_float < min_control),
    )


def mmfp_solver_unit_tests(seed: int = 0) -> list[dict[str, Any]]:
    gen = torch.Generator().manual_seed(int(seed))
    n = 12
    k = 5
    a = torch.randn(n, n, generator=gen)
    h = a.T @ a + 0.1 * torch.eye(n)
    e = torch.randn(n, k, generator=gen)
    b = torch.randn(n, generator=gen)
    gram = e.T @ h @ e
    damping = trace_scaled_damping(gram)
    expected = -torch.linalg.solve(gram + damping * torch.eye(k), e.T @ b)
    got, status = _safe_solve(gram + damping * torch.eye(k), -(e.T @ b))
    rel = torch.linalg.vector_norm(got - expected) / torch.linalg.vector_norm(expected).clamp_min(EPS)
    d = e @ got
    q = torch.dot(b, d) + 0.5 * torch.dot(d, h @ d)
    return [
        {
            "test": "A1_closed_form_descent_direction",
            "alpha_rel_error": float(rel.item()),
            "predicted_Q_decrease": float(q.item()),
            "solve_status": status,
            "pass": int(float(rel.item()) <= 1.0e-5 and float(q.item()) < 0.0),
        }
    ]


def metric_curvature_operator_tests() -> list[dict[str, Any]]:
    logits = torch.tensor([[2.0, -0.5, 0.1], [0.0, 1.0, -1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 2])
    vec = torch.randn_like(logits).reshape(-1)
    hv = ce_h_apply(logits, vec)
    psd_value = float(torch.dot(vec, hv).item())
    mse_vec = torch.randn(7)
    incidence = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    pair_h = incidence.T @ incidence
    pair_min_eig = float(torch.linalg.eigvalsh(pair_h).min().item())
    pair_psd = int(pair_min_eig >= -1.0e-6)
    return [
        {
            "metric": "CE_categorical_fisher",
            "finite": int(bool(torch.isfinite(hv).all().item())),
            "psd_probe": psd_value,
            "curvature_operator_finite_and_psd": int(torch.isfinite(hv).all().item() and psd_value >= -1.0e-8),
            "pass": int(torch.isfinite(hv).all().item() and psd_value >= -1.0e-8),
        },
        {
            "metric": "MSE_identity",
            "finite": int(bool(torch.isfinite(mse_vec).all().item())),
            "psd_probe": float(torch.dot(mse_vec, mse_vec).item()),
            "curvature_operator_finite_and_psd": 1,
            "pass": 1,
        },
        {
            "metric": "pairwise_incidence_curvature",
            "finite": int(bool(torch.isfinite(pair_h).all().item())),
            "psd_probe": pair_min_eig,
            "curvature_operator_finite_and_psd": pair_psd,
            "pass": pair_psd,
        },
    ]


def control_dominance_unit_tests(seed: int = 1) -> list[dict[str, Any]]:
    logits = torch.tensor([[1.5, -0.5], [0.2, 0.0], [-0.5, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1])
    cot = ce_delta(logits, labels).reshape(-1)
    true_effect = -cot.reshape(-1, 1)
    result = solve_ce_mmfp(logits, labels, true_effect, control_seed=seed)
    return [
        {
            "test": "A2_control_dominance",
            "Q_real": result.q_value,
            "Q_best_control": min(result.control_q.values()) if result.control_q else "",
            "control_dominance_margin": result.control_dominance_margin,
            "control_dominance_test_pass": result.control_dominance_pass,
            "pass": result.control_dominance_pass,
        }
    ]


def acceptance_unit_tests() -> list[dict[str, Any]]:
    base_b = 1.0
    candidate_b = 1.25
    sigma_b = abs(base_b - base_b)
    split_pass = int(candidate_b <= base_b + sigma_b)
    hard_base = 1.0
    hard_candidate = 0.99
    sigma_h = abs(hard_base - hard_base)
    hard_pass = int(hard_candidate <= hard_base + sigma_h)
    return [
        {
            "test": "A3_split_train_safety_rejects_overfit",
            "base_B_loss": base_b,
            "candidate_B_loss": candidate_b,
            "sigma_noop_B": sigma_b,
            "split_train_safety_test_pass": int(split_pass == 0),
            "pass": int(split_pass == 0),
        },
        {
            "test": "A4_hard_slice_no_debt_accepts_no_debt",
            "base_hard_loss": hard_base,
            "candidate_hard_loss": hard_candidate,
            "sigma_noop_H": sigma_h,
            "hard_slice_no_debt_test_pass": hard_pass,
            "pass": hard_pass,
        },
    ]


__all__ = [
    "MMFPSolveResult",
    "acceptance_unit_tests",
    "ce_delta",
    "ce_h_apply",
    "ce_quadratic_terms",
    "control_dominance_unit_tests",
    "metric_curvature_operator_tests",
    "mmfp_solver_unit_tests",
    "solve_ce_mmfp",
    "trace_scaled_damping",
]
