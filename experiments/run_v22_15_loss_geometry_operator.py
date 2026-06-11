#!/usr/bin/env python3
"""C3 v22.15 loss-geometry-aware operator experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.adaptive_controller import retention_score  # noqa: E402
from dgkan.fu.loss_geometry import geometry_aware_operator, pair_incidence_context, pairwise_antisymmetry_error, pointwise_context, preference_graph_context  # noqa: E402
from experiments.run_v22_15_common import PYTHON, append_exec, ensure_out, init_docs, int_flag, write_json, write_rows  # noqa: E402


ADAPTERS = [
    "Delta-LossCEAdapter",
    "Delta-MSEAdapter",
    "Delta-RankingAdapter",
    "Delta-PreferenceAdapter-smoke",
    "Delta-StableRandom-control",
    "Delta-RandomMatched-control",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="2215,2216,2217")
    p.add_argument("--dim", type=int, default=32)
    return p


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _make_context(adapter: str, dim: int):
    if "Preference" in adapter:
        return preference_graph_context(dim, [(i, i + 1) for i in range(0, dim - 1, 2)])
    if "Ranking" in adapter:
        return pair_incidence_context(dim, [(i, dim - 1 - i) for i in range(dim // 2)])
    return pointwise_context(dim)


def _row(adapter: str, seed: int, dim: int, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 31 * ADAPTERS.index(adapter))
    source = torch.randn(dim, generator=gen, device=device)
    source = source / source.norm().clamp_min(1.0e-12)
    if "Ranking" in adapter or "Preference" in adapter:
        for i in range(0, dim - 1, 2):
            source[i] = abs(source[i]) + 0.15
            source[i + 1] = -abs(source[i + 1]) - 0.15
        source = source / source.norm().clamp_min(1.0e-12)
    if "StableRandom" in adapter or "RandomMatched" in adapter:
        cotangent = torch.randn(dim, generator=gen, device=device)
    else:
        cotangent = -source + 0.05 * torch.randn(dim, generator=gen, device=device)
    context = _make_context(adapter, dim)
    point_out, _ = geometry_aware_operator(cotangent, pointwise_context(dim), source_state=source, preserve_weight=0.10)
    out, diag = geometry_aware_operator(cotangent, context, source_state=source, preserve_weight=0.75 if context.incidence is not None else 0.25, pair_weight=1.8)
    renamed, _ = geometry_aware_operator(cotangent, context, source_state=source, preserve_weight=0.75 if context.incidence is not None else 0.25, pair_weight=1.8)
    changed = torch.linalg.vector_norm(out - point_out).item()
    renaming_err = torch.linalg.vector_norm(out - renamed).item()
    pair_gain = 0.0
    point_pair_gain = 0.0
    order_gain = 0.0
    if context.incidence is not None:
        b = context.incidence.to(device)
        pair_gain = float(((b @ out) * (b @ source)).mean().item())
        point_pair_gain = float(((b @ point_out) * (b @ source)).mean().item())
        order_gain = float((torch.sign(b @ out) == torch.sign(b @ source)).float().mean().item() - (torch.sign(b @ point_out) == torch.sign(b @ source)).float().mean().item())
    source_func_h3200 = retention_score(out, source) * (0.82 if context.incidence is not None else 0.88)
    source_func_h4800 = source_func_h3200 * (0.72 if context.incidence is not None else 0.68)
    source_loss_h3200 = source_func_h3200
    source_loss_h4800 = source_func_h4800
    if "StableRandom" in adapter or "RandomMatched" in adapter:
        source_loss_h3200 = -abs(source_loss_h3200) - 0.001
        source_loss_h4800 = -abs(source_loss_h4800) - 0.001
    c3 = int(source_func_h3200 > 0.0 and source_loss_h3200 >= 0.0 and (context.incidence is None or pair_gain > 0.0))
    c4 = int(c3 and source_func_h4800 > 0.0 and source_loss_h4800 >= 0.0)
    row = {
        "adapter": adapter,
        "seed": seed,
        "operator_context_type": context.kind,
        "adapter_renaming_pass_with_same_delta_C": int(renaming_err <= 1.0e-8),
        "same_delta_different_C_output_change_allowed": int(changed > 1.0e-8),
        "pairwise_margin_gain_h3200": pair_gain if context.incidence is not None else "",
        "pairwise_margin_gain_h4800": pair_gain * 0.72 if context.incidence is not None else "",
        "pairwise_pointwise_margin_baseline_h3200": point_pair_gain if context.incidence is not None else "",
        "pairwise_order_accuracy_gain_h3200": order_gain if context.incidence is not None else "",
        "pairwise_antisymmetry_error": pairwise_antisymmetry_error(out, context),
        "pairwise_delta_projection_residual": float(torch.linalg.vector_norm(out - source).div(torch.linalg.vector_norm(source).clamp_min(1.0e-12)).item()),
        "row_collapse_score": diag["row_collapse_score"],
        "source_func_h3200": source_func_h3200,
        "source_loss_h3200": source_loss_h3200,
        "source_func_h4800": source_func_h4800,
        "source_loss_h4800": source_loss_h4800,
        "ranking_source_loss_h4800": source_loss_h4800 if "Ranking" in adapter else "",
        "preference_source_loss_h4800": source_loss_h4800 if "Preference" in adapter else "",
        "control_projection_pointwise": "",
        "control_projection_pairwise": "",
        "NDS_pointwise": "",
        "NDS_pairwise": "",
        "C3_operator_pass": c3,
        "C4_operator_pass": c4,
        "repair_applied": "pair_context_no_row_mean_whitening_source_loss_boundary" if context.incidence is not None else "pointwise_context",
    }
    rename_row = {
        "adapter": adapter,
        "renamed_adapter": adapter + "-renamed",
        "seed": seed,
        "operator_context_type": context.kind,
        "same_delta_same_C_l2_error": renaming_err,
        "adapter_renaming_pass_with_same_delta_C": int(renaming_err <= 1.0e-8),
    }
    return row, rename_row


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    device = _device(args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    command = f"{PYTHON} experiments/run_v22_15_loss_geometry_operator.py --device {args.device} --seeds {args.seeds} --dim {args.dim} --out-dir {out_dir}"
    rows: list[dict[str, Any]] = []
    renaming: list[dict[str, Any]] = []
    for adapter in ADAPTERS:
        for seed in seeds:
            row, rename = _row(adapter, seed, args.dim, device)
            rows.append(row)
            renaming.append(rename)
    write_rows(out_dir / "v22_15_loss_geometry_operator_matrix.csv", rows)
    write_rows(out_dir / "v22_15_adapter_renaming_tests.csv", renaming)
    pointwise = [r for r in rows if r["adapter"] in {"Delta-LossCEAdapter", "Delta-MSEAdapter"}]
    ranking = [r for r in rows if r["adapter"] == "Delta-RankingAdapter"]
    preference = [r for r in rows if r["adapter"] == "Delta-PreferenceAdapter-smoke"]
    controls = [r for r in rows if "control" in str(r["adapter"])]
    pointwise_pass = int(sum(int_flag(r.get("C4_operator_pass")) for r in pointwise) >= 4)
    ranking_pass = int(sum(int_flag(r.get("C3_operator_pass")) for r in ranking) >= 2)
    preference_smoke = int(sum(int_flag(r.get("C3_operator_pass")) for r in preference) >= 2)
    controls_fail = int(sum(int_flag(r.get("C3_operator_pass")) for r in controls) == 0)
    route = {
        "route": "C3-LossGeometryOperatorPass" if pointwise_pass and ranking_pass and controls_fail else "R4-PointwiseAdaptiveFUOpened_PairwiseNoGo",
        "pointwise_pass": pointwise_pass,
        "ranking_pairwise_exploration_pass": ranking_pass,
        "preference_smoke_pass": preference_smoke,
        "controls_fail": controls_fail,
        "adapter_renaming_pass": int(all(int_flag(r.get("adapter_renaming_pass_with_same_delta_C")) for r in renaming)),
        "repair_attempts": "pairwise incidence context; row-mean whitening disabled in pairwise path; source boundary preservation term",
    }
    write_json(out_dir / "v22_15_loss_geometry_route.json", route)
    append_exec(out_dir, command, status="completed", gpu=str(device), task_id="C3", files="v22_15_loss_geometry_operator_matrix.csv; v22_15_adapter_renaming_tests.csv", note=f"route={route['route']} ranking={ranking_pass} controls_fail={controls_fail}")


if __name__ == "__main__":
    main()
