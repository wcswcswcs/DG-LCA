"""Sobolev-H1 finite-difference metric wrapper for v22.05."""

from __future__ import annotations

import torch

from dgkan.fu.function_space_metrics import sobolev_fd_energy


def sobolev_metric_unit_tests() -> list[dict[str, int | str]]:
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    energy = sobolev_fd_energy(values)
    return [{"case": "fd_energy_finite", "pass": int(bool(torch.isfinite(energy).item()) and float(energy.item()) > 0.0)}]


__all__ = ["sobolev_fd_energy", "sobolev_metric_unit_tests"]
