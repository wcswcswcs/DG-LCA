"""Diagonal Fisher metric wrapper for v22.05."""

from __future__ import annotations

import torch

from dgkan.fu.function_space_metrics import fisher_diag_from_logits, fisher_energy


def fisher_metric_unit_tests() -> list[dict[str, int | str]]:
    logits = torch.tensor([[1.0, 0.0], [0.25, -0.25]])
    delta = torch.ones_like(logits) * 0.1
    energy = fisher_energy(delta, logits)
    diag = fisher_diag_from_logits(logits)
    return [
        {
            "case": "diag_positive",
            "pass": int(bool(torch.isfinite(energy).item()) and float(energy.item()) > 0.0 and bool((diag > 0).all().item())),
        }
    ]


__all__ = ["fisher_diag_from_logits", "fisher_energy", "fisher_metric_unit_tests"]
