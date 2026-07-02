"""Debt-conflict veto utilities for v23.00R population flow."""

from __future__ import annotations

import math
from typing import Mapping

import torch


EPS = 1.0e-12


def _f(value: torch.Tensor | float, default: float = 0.0) -> float:
    try:
        out = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def positive_cosine(task_mu: torch.Tensor, debt_mu: torch.Tensor) -> torch.Tensor:
    a = task_mu.detach().reshape(-1).to(dtype=torch.float64)
    b = debt_mu.detach().reshape(-1).to(dtype=torch.float64)
    n = min(int(a.numel()), int(b.numel()))
    if n <= 0:
        return torch.tensor(0.0, dtype=torch.float64)
    val = (a[:n] @ b[:n]) / (a[:n].norm().clamp_min(EPS) * b[:n].norm().clamp_min(EPS))
    return val.clamp_min(0.0)


def task_debt_cosine(task_mu: torch.Tensor, debt_mu: torch.Tensor) -> float:
    a = task_mu.detach().reshape(-1).to(dtype=torch.float64)
    b = debt_mu.detach().reshape(-1).to(dtype=torch.float64)
    n = min(int(a.numel()), int(b.numel()))
    if n <= 0:
        return 0.0
    return _f((a[:n] @ b[:n]) / (a[:n].norm().clamp_min(EPS) * b[:n].norm().clamp_min(EPS)))


def conflict_veto_gate(
    task_gate: torch.Tensor,
    task_mu: torch.Tensor,
    debt_gates: Mapping[str, torch.Tensor],
    debt_mus: Mapping[str, torch.Tensor],
    *,
    conflict_mode: str = "legacy_positive",
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Apply component-wise conflict veto.

    For scalar/diagonal gates, component conflict is global cosine_+^2 times
    each component debt gate. The maximum component veto is used.
    """

    tg = task_gate.detach().reshape(-1).to(dtype=torch.float64)
    if int(tg.numel()) == 0:
        return tg, {"veto_density_mean": 0.0, "component_count": 0}
    max_veto = torch.zeros_like(tg)
    component_rows: dict[str, float] = {}
    for name, dg in debt_gates.items():
        if name not in debt_mus:
            continue
        debt_gate = dg.detach().reshape(-1).to(dtype=torch.float64)
        if int(debt_gate.numel()) != int(tg.numel()):
            debt_gate = debt_gate[: int(tg.numel())]
            if int(debt_gate.numel()) < int(tg.numel()):
                debt_gate = torch.nn.functional.pad(debt_gate, (0, int(tg.numel()) - int(debt_gate.numel())))
        cosine = task_debt_cosine(task_mu, debt_mus[name])
        if str(conflict_mode) in {"descent_negative", "corrected_descent"}:
            cos_pos = torch.tensor(max(0.0, -float(cosine)), dtype=torch.float64)
        else:
            cos_pos = torch.tensor(max(0.0, float(cosine)), dtype=torch.float64)
        veto = debt_gate.clamp(0.0, 1.0) * cos_pos.square()
        max_veto = torch.maximum(max_veto, veto)
        component_rows[f"veto_density_{name}"] = _f(veto.mean())
        component_rows[f"task_debt_cosine_{name}"] = float(cosine)
    out = tg * (1.0 - max_veto.clamp(0.0, 1.0))
    removed = (tg - out).clamp_min(0.0)
    summary: dict[str, float | int | str] = {
        "component_count": len(component_rows) // 2,
        "conflict_mode": str(conflict_mode),
        "veto_density_mean": _f(max_veto.mean()),
        "vetoed_task_energy_fraction": _f(removed.sum() / tg.sum().clamp_min(EPS)),
        "preserved_task_energy_fraction": _f(out.sum() / tg.sum().clamp_min(EPS)),
        **component_rows,
    }
    return out, summary


def component_debt_report(component_deltas: Mapping[str, float]) -> dict[str, float | int | str]:
    clean = {str(k): float(v) for k, v in component_deltas.items() if math.isfinite(float(v))}
    worsened = {k: v for k, v in clean.items() if v > 0.0}
    return {
        **{f"{k}_delta": v for k, v in clean.items()},
        "debt_component_count": len(clean),
        "debt_component_worsened_count": len(worsened),
        "all_components_non_positive": int(not worsened and bool(clean)),
        "worst_debt_component_delta": max(worsened.values()) if worsened else 0.0,
    }


def debt_conflict_veto_smoke_test() -> dict[str, float | int]:
    task_gate = torch.ones(4, dtype=torch.float64)
    task_mu = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    same_mu = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    orth_mu = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    same, same_summary = conflict_veto_gate(task_gate, task_mu, {"same": task_gate}, {"same": same_mu})
    orth, orth_summary = conflict_veto_gate(task_gate, task_mu, {"orth": task_gate}, {"orth": orth_mu})
    opposite_mu = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    corrected, corrected_summary = conflict_veto_gate(
        task_gate,
        task_mu,
        {"opposite": task_gate},
        {"opposite": opposite_mu},
        conflict_mode="descent_negative",
    )
    return {
        "same_direction_gate_mean": _f(same.mean()),
        "orthogonal_gate_mean": _f(orth.mean()),
        "same_reduces_gate": int(_f(same.mean()) < _f(orth.mean())),
        "same_veto_density": float(same_summary.get("veto_density_mean", 0.0)),
        "orth_veto_density": float(orth_summary.get("veto_density_mean", 0.0)),
        "corrected_opposite_gate_mean": _f(corrected.mean()),
        "corrected_opposite_reduces_gate": int(_f(corrected.mean()) < 1.0),
        "corrected_opposite_veto_density": float(corrected_summary.get("veto_density_mean", 0.0)),
    }
