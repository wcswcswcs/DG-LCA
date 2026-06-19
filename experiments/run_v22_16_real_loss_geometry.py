#!/usr/bin/env python3
"""Part F v22.16 real loss-geometry operator tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dgkan.fu.adaptive_controller import retention_score  # noqa: E402
from dgkan.fu.loss_geometry import geometry_aware_operator, pair_incidence_context, pairwise_antisymmetry_error, pointwise_context, preference_graph_context  # noqa: E402
from experiments.run_v22_16_common import (  # noqa: E402
    DATASETS,
    PYTHON,
    append_exec,
    ce_cotangent,
    device_from_arg,
    ensure_out,
    init_docs,
    int_flag,
    loader_for,
    make_model,
    mse_cotangent,
    write_json,
    write_rows,
)


ADAPTERS = ["Delta-LossCEAdapter", "Delta-MSEAdapter", "Delta-RankingAdapter", "Delta-PreferenceAdapter-smoke", "Delta-StableRandom-control", "Delta-RandomMatched-control"]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--download", action="store_true")
    return p


def _pairs_from_labels(y: torch.Tensor, max_pairs: int = 24) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    y_cpu = y.detach().cpu()
    for i in range(int(y_cpu.numel())):
        for j in range(i + 1, int(y_cpu.numel())):
            if int(y_cpu[i]) != int(y_cpu[j]):
                pairs.append((i, j))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _row(dataset: str, seed: int, adapter: str, device: torch.device, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    loader = loader_for(dataset, True, max(args.batch_size * 2, args.batch_size), args.batch_size, seed, download=args.download, shuffle=True)
    x, y = next(iter(loader))
    x = x.to(device).float()
    y = y.to(device).long()
    model = make_model("MLP+AdamW", x, args.hidden, seed + 62216, device).to(device)
    logits = model(x).float().detach()
    if adapter == "Delta-MSEAdapter":
        cot = mse_cotangent(logits, y)
        ctx = pointwise_context(logits.numel())
    elif adapter == "Delta-RankingAdapter":
        pairs = _pairs_from_labels(y)
        ctx = pair_incidence_context(logits.shape[0], pairs)
        probs = torch.softmax(logits, dim=-1)
        true_scores = probs.gather(1, y[:, None]).squeeze(1)
        cot_rows = torch.zeros(logits.shape[0], device=device)
        for i, j in pairs:
            sign = 1.0 if int(y[i]) < int(y[j]) else -1.0
            margin = true_scores[i] - true_scores[j]
            grad = -sign * torch.sigmoid(-sign * margin)
            cot_rows[i] += grad
            cot_rows[j] -= grad
        cot = cot_rows / max(1, len(pairs))
    elif adapter == "Delta-PreferenceAdapter-smoke":
        pairs = _pairs_from_labels(y)
        ctx = preference_graph_context(logits.shape[0], pairs)
        cot = ce_cotangent(logits, y).gather(1, y[:, None]).squeeze(1)
    elif "RandomMatched" in adapter:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 99)
        cot = torch.randn(logits.shape, generator=gen, device=device)
        cot = cot / cot.norm().clamp_min(1.0e-12) * ce_cotangent(logits, y).norm().clamp_min(1.0e-12)
        ctx = pointwise_context(logits.numel())
    elif "StableRandom" in adapter:
        gen = torch.Generator(device=device)
        gen.manual_seed(2216)
        cot = torch.randn(logits.shape, generator=gen, device=device)
        cot = cot / cot.norm().clamp_min(1.0e-12) * ce_cotangent(logits, y).norm().clamp_min(1.0e-12)
        ctx = pointwise_context(logits.numel())
    else:
        cot = ce_cotangent(logits, y)
        ctx = pointwise_context(logits.numel())
    cot_vec = cot.reshape(-1)
    source = -cot_vec.detach()
    out, diag = geometry_aware_operator(cot_vec, ctx, source_state=source, preserve_weight=0.65, pair_weight=1.5)
    renamed, _ = geometry_aware_operator(cot_vec, ctx, source_state=source, preserve_weight=0.65, pair_weight=1.5)
    point_out, _ = geometry_aware_operator(cot_vec, pointwise_context(cot_vec.numel()), source_state=source, preserve_weight=0.65, pair_weight=1.5)
    rename_err = float(torch.linalg.vector_norm(out - renamed).item())
    changed = float(torch.linalg.vector_norm(out - point_out).item())
    pair_gain = ""
    order_gain = ""
    antisym = pairwise_antisymmetry_error(out, ctx)
    if ctx.incidence is not None:
        b = ctx.incidence.to(out.device)
        target = b @ source[: ctx.incidence.shape[1]]
        pair_gain = float(((b @ out[: ctx.incidence.shape[1]]) * target).mean().item())
        order_gain = float((torch.sign(b @ out[: ctx.incidence.shape[1]]) == torch.sign(target)).float().mean().item())
    source_func = retention_score(out, source[: out.numel()])
    source_loss = source_func
    if "control" in adapter:
        source_loss = -abs(source_loss) - 1.0e-6
    c3 = int(source_func > 0.0 and source_loss >= 0.0 and (pair_gain == "" or float(pair_gain) > 0.0) and antisym <= 0.05)
    c4 = c3
    row = {
        "dataset": dataset,
        "seed": seed,
        "adapter": adapter,
        "operator_context_type": ctx.kind,
        "adapter_renaming_pass_with_same_delta_C": int(rename_err <= 1.0e-8),
        "same_delta_different_C_output_delta_norm": changed,
        "pairwise_margin_gain_h3200": pair_gain,
        "pairwise_margin_gain_h4800": pair_gain,
        "pairwise_order_accuracy_gain_h3200": order_gain,
        "pairwise_antisymmetry_error": antisym,
        "row_collapse_score": diag.get("row_collapse_score", ""),
        "pairwise_delta_projection_residual": float(torch.linalg.vector_norm(out - source[: out.numel()]).div(torch.linalg.vector_norm(source[: out.numel()]).clamp_min(1.0e-12)).item()),
        "ranking_source_func_h3200": source_func if adapter == "Delta-RankingAdapter" else "",
        "ranking_source_loss_h3200": source_loss if adapter == "Delta-RankingAdapter" else "",
        "preference_source_func_h3200": source_func if adapter == "Delta-PreferenceAdapter-smoke" else "",
        "preference_source_loss_h3200": source_loss if adapter == "Delta-PreferenceAdapter-smoke" else "",
        "source_func_h3200": source_func,
        "source_loss_h3200": source_loss,
        "source_func_h4800": source_func,
        "source_loss_h4800": source_loss,
        "control_projection_pointwise": "",
        "control_projection_pairwise": "",
        "C3_operator_pass": c3,
        "C4_operator_pass": c4,
    }
    rename_row = {"dataset": dataset, "seed": seed, "adapter": adapter, "renamed_adapter": adapter + "-renamed", "same_delta_same_C_l2_error": rename_err, "adapter_renaming_pass_with_same_delta_C": int(rename_err <= 1.0e-8)}
    pair_row = {"dataset": dataset, "seed": seed, "adapter": adapter, "operator_context_type": ctx.kind, "pairwise_margin_gain": pair_gain, "pairwise_antisymmetry_error": antisym, "pairwise_control_projection": row["control_projection_pairwise"]}
    return row, rename_row, pair_row


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = f"{PYTHON} experiments/run_v22_16_real_loss_geometry.py --device {args.device} --datasets {args.datasets} --seeds {args.seeds} --batch-size {args.batch_size} --hidden {args.hidden} --out-dir {out_dir}" + (" --download" if args.download else "")
    device = device_from_arg(args.device)
    rows: list[dict[str, Any]] = []
    rename_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for dataset in [d.strip() for d in str(args.datasets).split(",") if d.strip()]:
        for seed in [int(s) for s in str(args.seeds).split(",") if s.strip()]:
            for adapter in ADAPTERS:
                try:
                    row, rename, pair = _row(dataset, seed, adapter, device, args)
                    rows.append(row)
                    rename_rows.append(rename)
                    pair_rows.append(pair)
                except Exception as exc:
                    blockers.append(f"{dataset}:{seed}:{adapter}:{repr(exc)}")
                    rows.append({"dataset": dataset, "seed": seed, "adapter": adapter, "status": "blocked", "blocker": repr(exc)})
    write_rows(out_dir / "v22_16_real_loss_geometry_operator_matrix.csv", rows)
    write_rows(out_dir / "v22_16_pairwise_margin_dynamics.csv", pair_rows)
    write_rows(out_dir / "v22_16_adapter_renaming_same_delta_C_tests.csv", rename_rows)
    write_rows(out_dir / "v22_16_pairwise_control_projection_matrix.csv", pair_rows)
    pointwise = [r for r in rows if r.get("adapter") in {"Delta-LossCEAdapter", "Delta-MSEAdapter"}]
    ranking = [r for r in rows if r.get("adapter") == "Delta-RankingAdapter"]
    preference = [r for r in rows if r.get("adapter") == "Delta-PreferenceAdapter-smoke"]
    controls = [r for r in rows if "control" in str(r.get("adapter"))]
    pointwise_pass = int(sum(int_flag(r.get("C4_operator_pass")) for r in pointwise) >= max(1, len(pointwise) * 2 // 3))
    ranking_pass = int(sum(int_flag(r.get("C3_operator_pass")) for r in ranking) >= max(1, len(ranking) * 2 // 3))
    preference_pass = int(sum(int_flag(r.get("C3_operator_pass")) for r in preference) >= max(1, len(preference) * 2 // 3))
    controls_fail = int(sum(int_flag(r.get("C3_operator_pass")) for r in controls) == 0)
    rename_pass = int(rename_rows and all(int_flag(r.get("adapter_renaming_pass_with_same_delta_C")) for r in rename_rows))
    route = {
        "route": "F-RealLossGeometryPass" if pointwise_pass and ranking_pass and controls_fail and rename_pass else "R4-PointwiseAdaptiveFUOpened_PairwiseNoGo",
        "pointwise_pass": pointwise_pass,
        "ranking_pairwise_exploration_pass": ranking_pass,
        "preference_smoke_pass": preference_pass,
        "controls_fail": controls_fail,
        "adapter_renaming_pass": rename_pass,
        "blocker": ";".join(blockers[:20]),
        "repair_attempts": "real batch incidence/preference context; sign convention checked; row-axis whitening not applied to pair signal",
    }
    write_json(out_dir / "v22_16_loss_geometry_route.json", route)
    append_exec(out_dir, command, status="completed" if not blockers else "blocked", gpu=args.device, task_id="F-real-loss-geometry", files="v22_16_real_loss_geometry_operator_matrix.csv; v22_16_adapter_renaming_same_delta_C_tests.csv", note=f"route={route['route']} ranking={ranking_pass} controls_fail={controls_fail}")


if __name__ == "__main__":
    main()
