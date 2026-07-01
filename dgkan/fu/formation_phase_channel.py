"""Formation-phase channel diagnostics for v22.98.

These helpers keep early/late channel comparisons explicit.  They do not pick
an optimizer at runtime; they only summarize pre-registered checkpoint grids.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from dgkan.fu.layer_composite_metric import EPS


def retention_decay_summary(retentions: Iterable[float], steps: Iterable[int]) -> dict[str, float | int]:
    vals = [float(v) for v in retentions if math.isfinite(float(v))]
    ss = [int(s) for s in steps]
    if not vals:
        return {
            "early_quotient_retention": 0.0,
            "late_quotient_retention": 0.0,
            "retention_decay_rate": 0.0,
            "early_gt_late": 0,
        }
    early = vals[0]
    late = vals[-1]
    span = max(1, int(ss[-1] - ss[0])) if ss else max(1, len(vals) - 1)
    decay = (early - late) / float(span)
    return {
        "early_quotient_retention": early,
        "late_quotient_retention": late,
        "retention_decay_rate": decay,
        "early_gt_late": int(early > late),
    }


def fixed_schedule_steps(total_steps: int, formation_fraction: float = 0.20) -> dict[str, int]:
    total = max(1, int(total_steps))
    form = max(1, min(total, int(round(float(formation_fraction) * total))))
    return {"total_steps": total, "formation_steps": form, "preservation_steps": max(0, total - form)}


def source_guard_retention_table(
    task_retention: torch.Tensor,
    bad_retention: torch.Tensor,
    *,
    threshold: float = 1.25,
) -> dict[str, float | int]:
    task = task_retention.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(0.0)
    bad = bad_retention.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(0.0)
    n = min(int(task.numel()), int(bad.numel()))
    if n <= 0:
        return {"safe_block_count": 0, "safe_task_mass": 0.0, "safe_bad_mass": 0.0}
    task = task[:n]
    bad = bad[:n]
    ratio = task / bad.clamp_min(EPS)
    mask = ratio >= float(threshold)
    return {
        "safe_block_count": int(mask.sum().item()),
        "safe_task_mass": float(task[mask].sum().item()) / max(float(task.sum().item()), EPS),
        "safe_bad_mass": float(bad[mask].sum().item()) / max(float(bad.sum().item()), EPS),
    }


def formation_phase_smoke_test() -> dict[str, float | int]:
    summary = retention_decay_summary([0.8, 0.6, 0.2], [0, 5, 10])
    sched = fixed_schedule_steps(10, 0.2)
    table = source_guard_retention_table(torch.tensor([2.0, 0.1]), torch.tensor([0.5, 1.0]))
    return {
        "formation_decay_smoke": int(summary["early_gt_late"] == 1 and summary["retention_decay_rate"] > 0.0),
        "formation_schedule_smoke": int(sched["formation_steps"] == 2 and sched["preservation_steps"] == 8),
        "formation_safe_block_smoke": int(table["safe_block_count"] == 1),
    }

