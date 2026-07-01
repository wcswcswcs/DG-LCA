"""Train-only output-space safe velocity helpers for v22.90."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class OutputSafeVelocityResult:
    velocity: torch.Tensor
    kkt_residual: float
    debt_violation_after: float
    task_preserved_fraction: float
    active_constraints: int


def project_velocity_halfspaces(
    task_velocity: torch.Tensor,
    constraint_grads: Iterable[torch.Tensor],
    constraint_rhs: Iterable[float | torch.Tensor],
    *,
    passes: int = 8,
    eps: float = 1.0e-12,
) -> OutputSafeVelocityResult:
    """Project ``task_velocity`` onto linear halfspaces ``g_i dot v <= rhs_i``.

    The cyclic projection is deterministic and sufficient for the small QP
    smoke tests and compact train-batch diagnostics used by the runner.
    """

    v_task = task_velocity.to(dtype=torch.float64)
    grads = [g.detach().to(device=v_task.device, dtype=torch.float64) for g in constraint_grads]
    rhs_vals = [torch.as_tensor(r, device=v_task.device, dtype=torch.float64) for r in constraint_rhs]
    if grads:
        flat_task = v_task.reshape(-1)
        flat_grads = torch.stack([g.reshape(-1) for g in grads], dim=0)
        rhs_vec = torch.stack([r.reshape(()) for r in rhs_vals], dim=0)
        m = int(flat_grads.shape[0])
        best_v: torch.Tensor | None = None
        best_obj = float("inf")
        # Exact active-set enumeration is cheap for the five v22.90 output
        # constraints and avoids treating an under-iterated cyclic projection
        # as a debt-safe certificate.
        if m <= 12:
            for mask in range(1 << m):
                active = [i for i in range(m) if mask & (1 << i)]
                if not active:
                    cand = flat_task
                    lamb = torch.zeros(0, device=flat_task.device, dtype=torch.float64)
                else:
                    ga = flat_grads[active]
                    ba = ga @ flat_task - rhs_vec[active]
                    gram = ga @ ga.T
                    ridge = float(eps) * torch.eye(len(active), device=flat_task.device, dtype=torch.float64)
                    lamb = torch.linalg.pinv(gram + ridge) @ ba
                    cand = flat_task - ga.T @ lamb
                residual = flat_grads @ cand - rhs_vec
                if active and bool((lamb < -1.0e-8).any().detach().cpu().item()):
                    continue
                if bool((residual > 1.0e-8).any().detach().cpu().item()):
                    continue
                obj = float((cand - flat_task).square().sum().detach().cpu().item())
                if obj < best_obj:
                    best_obj = obj
                    best_v = cand
            if best_v is not None:
                v = best_v.reshape_as(v_task)
                residuals = [torch.relu(torch.sum(g * v) - rhs).detach() for g, rhs in zip(grads, rhs_vals)]
                max_res = max([float(r.cpu().item()) for r in residuals] or [0.0])
                denom = (v_task.norm() * v.norm()).clamp_min(float(eps))
                preserved = float((torch.sum(v_task * v) / denom).detach().cpu().item()) if float(v.norm().detach().cpu().item()) > 0.0 else 0.0
                active_count = sum(float(r.cpu().item()) > 1.0e-8 for r in residuals)
                return OutputSafeVelocityResult(
                    velocity=v.to(dtype=task_velocity.dtype),
                    kkt_residual=float(max_res),
                    debt_violation_after=float(max_res),
                    task_preserved_fraction=preserved,
                    active_constraints=int(active_count),
                )
    v = v_task.clone()
    active = 0
    for _ in range(max(1, int(passes))):
        active = 0
        for g, rhs in zip(grads, rhs_vals):
            lhs = torch.sum(g * v)
            violation = lhs - rhs
            if float(violation.detach().cpu().item()) > 0.0:
                v = v - violation * g / torch.sum(g * g).clamp_min(float(eps))
                active += 1
    residuals = [torch.relu(torch.sum(g * v) - rhs).detach() for g, rhs in zip(grads, rhs_vals)]
    max_res = max([float(r.cpu().item()) for r in residuals] or [0.0])
    denom = (v_task.norm() * v.norm()).clamp_min(float(eps))
    preserved = float((torch.sum(v_task * v) / denom).detach().cpu().item()) if float(v.norm().detach().cpu().item()) > 0.0 else 0.0
    return OutputSafeVelocityResult(
        velocity=v.to(dtype=task_velocity.dtype),
        kkt_residual=float(max_res),
        debt_violation_after=float(max_res),
        task_preserved_fraction=preserved,
        active_constraints=int(active),
    )


def output_debt_surrogates(logits: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    prob = F.softmax(logits.float(), dim=1)
    onehot = F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()
    brier = (prob - onehot).square().sum(dim=1).mean()
    conf = prob.max(dim=1).values
    pred = prob.argmax(dim=1)
    correct = (pred == y.long()).float().detach()
    ece_proxy = (conf - correct).abs().mean()
    true_prob = prob.gather(1, y.long().reshape(-1, 1)).reshape(-1)
    tail95 = torch.quantile(1.0 - true_prob, 0.95)
    if int(logits.shape[1]) > 1:
        true_logit = logits.float().gather(1, y.long().reshape(-1, 1)).reshape(-1)
        other_logits = logits.float().clone()
        other_logits.scatter_(1, y.long().reshape(-1, 1), float("-inf"))
        margin = true_logit - other_logits.max(dim=1).values
    else:
        margin = logits.float().reshape(-1)
    margin10_debt = -torch.quantile(margin, 0.10)
    coverage_debt = -torch.quantile(true_prob, 0.25)
    return {
        "brier": brier,
        "ece": ece_proxy,
        "tail95": tail95,
        "margin10": margin10_debt,
        "coverage_cvar25": coverage_debt,
    }


def output_safe_velocity_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    task_velocity: torch.Tensor,
    *,
    alpha: float = 0.10,
    passes: int = 64,
    eps: float = 1.0e-12,
) -> OutputSafeVelocityResult:
    """Build a train-only safe velocity from current logits and labels.

    Constraints are local linearizations of debt surrogates.  The right-hand
    side asks the velocity to reduce positive debt at rate ``alpha``.
    """

    z = logits.detach().clone().requires_grad_(True)
    debts = output_debt_surrogates(z, y)
    grads: list[torch.Tensor] = []
    rhs: list[torch.Tensor] = []
    for value in debts.values():
        grad = torch.autograd.grad(value, z, retain_graph=True, create_graph=False, allow_unused=False)[0]
        debt_now = torch.relu(value.detach())
        grads.append(grad)
        rhs.append(-float(alpha) * debt_now)
    return project_velocity_halfspaces(task_velocity, grads, rhs, passes=int(passes), eps=eps)


__all__ = [
    "OutputSafeVelocityResult",
    "output_debt_surrogates",
    "output_safe_velocity_from_logits",
    "project_velocity_halfspaces",
]
