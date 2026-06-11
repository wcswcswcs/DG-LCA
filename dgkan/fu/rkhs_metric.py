"""RKHS / graph metric wrapper for v22.05."""

from __future__ import annotations

import torch

from dgkan.fu.function_space_metrics import rkhs_graph_energy


def rkhs_metric_unit_tests() -> list[dict[str, int | str]]:
    values = torch.eye(5)
    energy = rkhs_graph_energy(values, k=2)
    return [{"case": "knn_energy_finite", "pass": int(bool(torch.isfinite(energy).item()) and float(energy.item()) >= 0.0)}]


__all__ = ["rkhs_graph_energy", "rkhs_metric_unit_tests"]
