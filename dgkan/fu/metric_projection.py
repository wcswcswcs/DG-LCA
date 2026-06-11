"""Parameter pullback projection wrappers for v22.05 metrics."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.function_space_metrics import METRIC_NAMES, metric_project_vector


def project_metric_update(gradient: torch.Tensor, metric_name: str) -> tuple[torch.Tensor, dict[str, Any]]:
    return metric_project_vector(gradient, metric_name)


def metric_projection_unit_tests() -> list[dict[str, Any]]:
    g = torch.randn(32, generator=torch.Generator().manual_seed(2205))
    rows = []
    for metric in METRIC_NAMES:
        projected, diag = project_metric_update(g, metric)
        rows.append(
            {
                "case": metric,
                "pass": int(projected.shape == g.shape and torch.isfinite(projected).all().item() and diag.get("metric_name") == metric),
            }
        )
    return rows


__all__ = ["METRIC_NAMES", "metric_projection_unit_tests", "project_metric_update"]
