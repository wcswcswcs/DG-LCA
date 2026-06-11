#!/usr/bin/env python3
"""v22.10 S2 train-only source atom generation with planned fallbacks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.source_atoms import generate_loss_agnostic_source_atoms  # noqa: E402
from experiments.run_v22_10_common import PYTHON, append_exec, ensure_out, int_flag, simple_svg, write_json, write_rows  # noqa: E402


class TinyConstructiveMLP(torch.nn.Module):
    def __init__(self, input_dim: int = 8, hidden: int = 64, classes: int = 5) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden)
        self.w2 = torch.nn.Parameter(torch.randn(hidden, classes) * 0.02)

    def frozen_readout_features(self, xb: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc1(xb))

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.frozen_readout_features(xb) @ self.w2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=2210)
    return p


def _make_probe(seed: int) -> tuple[TinyConstructiveMLP, torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(seed))
    model = TinyConstructiveMLP()
    x = torch.randn(48, 8)
    with torch.no_grad():
        logits = model(x).detach().float()
    return model, x, logits


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    model, x, logits = _make_probe(args.seed)
    attempts = [
        {"attempt": "initial_norm_0p08_split4_all", "norm_scale": 0.08, "split_count": 4, "block_role": "all", "fallback_step": "initial"},
        {"attempt": "fallback_lower_norm_0p04_split4_all", "norm_scale": 0.04, "split_count": 4, "block_role": "all", "fallback_step": "lower_source_atom_norm"},
        {"attempt": "fallback_lower_norm_0p02_split4_all", "norm_scale": 0.02, "split_count": 4, "block_role": "all", "fallback_step": "lower_source_atom_norm"},
        {"attempt": "fallback_more_splits_0p02_split6_all", "norm_scale": 0.02, "split_count": 6, "block_role": "all", "fallback_step": "increase_micro_batch_split_count"},
        {"attempt": "fallback_block_restricted_0p02_split6_readout", "norm_scale": 0.02, "split_count": 6, "block_role": "readout_only", "fallback_step": "block_restricted_atom"},
    ]
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected: tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]] | None = None
    for attempt in attempts:
        rows, tensors, summary = generate_loss_agnostic_source_atoms(
            logits,
            carrier="MLP",
            block_role=str(attempt["block_role"]),
            norm_scale=float(attempt["norm_scale"]),
            split_count=int(attempt["split_count"]),
            seed=int(args.seed),
        )
        for row in rows:
            row.update(attempt)
        pass_rows = sum(int_flag(r.get("S2_source_atom_pass")) for r in rows)
        summary_row = {**attempt, **summary, "S2_source_atom_pass_rows": pass_rows, "route": "S2-SourceAtomPass" if pass_rows else "S2-SourceAtomAttemptBlocked"}
        summary_rows.append(summary_row)
        all_rows.extend(rows)
        if pass_rows > 0 and selected is None:
            selected = (attempt, tensors, summary_row)

    if selected is None:
        route = {
            "route": "S2-SourceAtomGenerationNoGo",
            "S2_source_atom_pass_rows": 0,
            "loss_agnostic_contract_pass": int(all(int_flag(r.get("loss_agnostic_contract_pass")) for r in all_rows)) if all_rows else 0,
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "attempt_rows": len(summary_rows),
            "selected_attempt": "",
            "promotion_allowed": 0,
            "blocker": ";".join(dict.fromkeys(part for row in all_rows for part in str(row.get("blocker", "")).split(";") if part)),
        }
        selected_tensors: dict[str, torch.Tensor] = {}
        selected_attempt = attempts[-1]
    else:
        selected_attempt, selected_tensors, selected_summary = selected
        route = {
            "route": "S2-SourceAtomProgress",
            "S2_source_atom_pass_rows": int(selected_summary.get("S2_source_atom_pass_rows", 0)),
            "loss_agnostic_contract_pass": int_flag(selected_summary.get("loss_agnostic_contract_pass")),
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "attempt_rows": len(summary_rows),
            "selected_attempt": selected_attempt.get("attempt", ""),
            "selected_norm_scale": selected_attempt.get("norm_scale", ""),
            "selected_split_count": selected_attempt.get("split_count", ""),
            "selected_block_role": selected_attempt.get("block_role", ""),
            "promotion_allowed": 0,
            "blocker": "",
        }
    payload = {
        "seed": int(args.seed),
        "model_config": {"input_dim": 8, "hidden": 64, "classes": 5},
        "model_state": model.state_dict(),
        "x": x,
        "logits": logits,
        "loss_agnostic_contract_pass": int_flag(route.get("loss_agnostic_contract_pass")),
        "selected_attempt": dict(selected_attempt),
        "source_atom_tensors": selected_tensors,
    }
    torch.save(payload, out_dir / "v22_10_source_atom_payload.pt")
    write_rows(out_dir / "v22_10_source_atom_attempt_summary.csv", summary_rows)
    write_rows(out_dir / "v22_10_source_atom_rows.csv", all_rows)
    write_json(out_dir / "v22_10_source_atom_route.json", route)
    simple_svg(out_dir / "figures/v22_10_source_atom_DDR_vs_NDS.svg", "v22.10 source atom DDR/NDS", all_rows, "DDR")
    simple_svg(out_dir / "figures/v22_10_source_atom_B2_gain_vs_random_gap.svg", "v22.10 source atom B2/random gap", all_rows, "B2_transfer_gain")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_10_source_atom_generation.py --seed {int(args.seed)} --out-dir {out_dir}",
        status="completed",
        note=f"route={route['route']} pass_rows={route['S2_source_atom_pass_rows']} selected_attempt={route.get('selected_attempt')} payload={out_dir / 'v22_10_source_atom_payload.pt'}",
    )


if __name__ == "__main__":
    main()
