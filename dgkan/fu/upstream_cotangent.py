"""Arbitrary upstream-cotangent helpers for v22.11."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dgkan.fu.loss_interface import (
    ClassificationCEAdapter,
    GenericUpstreamCotangent,
    LossInterface,
    PairwiseRankingAdapter,
    PolicyPreferenceAdapter,
    RegressionMSEAdapter,
    stable_random_delta_like,
)


@dataclass(frozen=True)
class CotangentSpec:
    cotangent_type: str
    adapter: LossInterface
    task_data: Any = None
    uses_labels_for_adapter: int = 0
    smoke_only: int = 0


def source_target_delta_like(logits: torch.Tensor) -> torch.Tensor:
    z = logits.detach().float()
    row_centered = z - z.mean(dim=-1, keepdim=True)
    col_centered = row_centered - row_centered.mean(dim=0, keepdim=True)
    return col_centered / torch.linalg.vector_norm(col_centered).clamp_min(1.0e-8) * (float(z.numel()) ** 0.5)


def build_cotangent_suite(logits: torch.Tensor, *, seed: int = 2211, labels: torch.Tensor | None = None) -> list[CotangentSpec]:
    labels = labels if labels is not None else (torch.arange(int(logits.shape[0]), device=logits.device) % int(logits.shape[-1]))
    mse_target = torch.tanh(logits.detach().float())
    return [
        CotangentSpec(
            "Delta-Gaussian",
            GenericUpstreamCotangent(stable_random_delta_like(logits, seed=seed, kind="gaussian"), "Delta-Gaussian"),
        ),
        CotangentSpec(
            "Delta-StableRandom",
            GenericUpstreamCotangent(stable_random_delta_like(logits, seed=seed, kind="stable"), "Delta-StableRandom"),
        ),
        CotangentSpec(
            "Delta-SourceTarget",
            GenericUpstreamCotangent(source_target_delta_like(logits), "Delta-SourceTarget"),
        ),
        CotangentSpec("Delta-LossCEAdapter", ClassificationCEAdapter(), labels, uses_labels_for_adapter=1),
        CotangentSpec("Delta-MSEAdapter", RegressionMSEAdapter(), mse_target),
        CotangentSpec("Delta-RankingAdapter", PairwiseRankingAdapter()),
        CotangentSpec("Delta-PolicyPreferenceAdapter", PolicyPreferenceAdapter(), smoke_only=1),
    ]


def validate_cotangent_suite(logits: torch.Tensor, specs: list[CotangentSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        delta = spec.adapter.cotangent(logits, spec.task_data)
        finite = bool(torch.isfinite(delta).all().item())
        rows.append(
            {
                "cotangent_type": spec.cotangent_type,
                "adapter_name": spec.adapter.name(),
                "shape": "x".join(str(x) for x in delta.shape),
                "cotangent_norm": float(torch.linalg.vector_norm(delta.detach().float()).item()),
                "uses_labels_for_adapter": int(spec.uses_labels_for_adapter),
                "uses_loss_formula_specific_direction": 0,
                "smoke_only": int(spec.smoke_only),
                "upstream_cotangent_contract_pass": int(delta.shape == logits.shape and finite),
            }
        )
    return rows


def upstream_cotangent_unit_tests() -> list[dict[str, Any]]:
    torch.manual_seed(2211)
    logits = torch.randn(20, 6)
    specs = build_cotangent_suite(logits, seed=2211)
    rows = validate_cotangent_suite(logits, specs)
    return [
        {
            "case": "build_and_validate_cotangent_suite",
            "cotangent_types": len(rows),
            "contract_pass_rows": sum(int(r.get("upstream_cotangent_contract_pass", 0)) for r in rows),
            "uses_loss_formula_specific_direction": 0,
            "pass": int(len(rows) >= 6 and all(int(r.get("upstream_cotangent_contract_pass", 0)) for r in rows)),
        }
    ]


__all__ = [
    "CotangentSpec",
    "build_cotangent_suite",
    "source_target_delta_like",
    "upstream_cotangent_unit_tests",
    "validate_cotangent_suite",
]
