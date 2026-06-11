"""Basis/readout channel metric diagnostics for v22.06.

The functions here split a flat update into coarse trainable-parameter
channels.  They are audit helpers and optional metric weights; they do not use
validation/test/future outcomes.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import torch


EPS = 1.0e-8


def parameter_channel_slices(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        low = name.lower()
        if any(tok in low for tok in ["w2", "readout", "classifier"]):
            channel = "readout"
        elif any(tok in low for tok in ["w1", "fc1", "hidden"]):
            channel = "hidden"
        elif any(tok in low for tok in ["cent", "scale", "basis", "den", "num"]):
            channel = "basis"
        elif "bias" in low or low.endswith(".b2") or low.endswith(".bias"):
            channel = "bias"
        else:
            channel = "other"
        rows.append({"name": name, "channel": channel, "start": offset, "end": offset + n, "numel": n})
        offset += n
    return rows


def basis_channel_energy(model: torch.nn.Module, update: torch.Tensor) -> dict[str, Any]:
    vec = update.detach().float().reshape(-1)
    total = torch.linalg.vector_norm(vec).square().clamp_min(EPS)
    sums: dict[str, float] = {}
    for row in parameter_channel_slices(model):
        part = vec[int(row["start"]) : int(row["end"])]
        sums[str(row["channel"])] = sums.get(str(row["channel"]), 0.0) + float(torch.linalg.vector_norm(part).square().item())
    source = sums.get("readout", 0.0) + sums.get("hidden", 0.0)
    reservoir = sums.get("basis", 0.0) + sums.get("other", 0.0)
    denom = max(source + reservoir, EPS)
    out: dict[str, Any] = {
        "hidden_source_energy": sums.get("hidden", 0.0),
        "readout_source_energy": sums.get("readout", 0.0),
        "bias_source_energy": sums.get("bias", 0.0),
        "basis_channel_energy": sums.get("basis", 0.0),
        "other_source_energy": sums.get("other", 0.0),
        "source_channel_projection": source / denom,
        "reservoir_projection": reservoir / denom,
        "source_to_reservoir_leakage": reservoir / denom,
        "hidden_source_fraction": sums.get("hidden", 0.0) / float(total.item()),
        "readout_source_fraction": sums.get("readout", 0.0) / float(total.item()),
    }
    return out


def basis_channel_metric_unit_tests() -> list[dict[str, Any]]:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(3, 4)
            self.w2 = torch.nn.Parameter(torch.randn(4, 2) * 0.01)

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.fc1(xb)) @ self.w2

    torch.manual_seed(2206)
    model = Tiny()
    vec = torch.zeros(sum(p.numel() for p in model.parameters() if p.requires_grad))
    for row in parameter_channel_slices(model):
        if row["name"] == "w2":
            vec[int(row["start"]) : int(row["end"])] = 1.0
    diag = basis_channel_energy(model, vec)
    finite = all(isfinite(float(v)) for v in diag.values() if isinstance(v, (float, int)))
    return [
        {
            "case": "readout_channel_energy",
            "readout_source_energy": diag.get("readout_source_energy", ""),
            "source_channel_projection": diag.get("source_channel_projection", ""),
            "diagnostics_finite": int(finite),
            "pass": int(finite and float(diag.get("readout_source_energy", 0.0)) > 0.0),
        }
    ]


__all__ = ["basis_channel_energy", "basis_channel_metric_unit_tests", "parameter_channel_slices"]
