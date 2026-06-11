#!/usr/bin/env python3
"""v22.07 C3 optimizer-state preservation / anti-washout smoke.

This stage runs only after semantic target repair opens h100/h400/h800 but
washes out by h1600/h3200.  It uses the already-observed train-only semantic
target families as source axes and tests optimizer-state preservation variants.
It is a smoke repair: it does not claim official C2 solver closure.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dgkan.fu.core import flat_grad, normalized_like, trainable_parameters  # noqa: E402
from experiments.run_v17_common import carrier_model, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_07_c3_semantic_target_repair import (  # noqa: E402
    _assign_flat_gradient,
    _cap_norm,
    _eval,
    _orthogonal_component,
    _semantic_target,
)
from experiments.run_v22_07_common import PYTHON, V2206_COMBINED_SOURCE, append_exec, ensure_out, finite_float, int_flag, read_rows, write_json, write_rows  # noqa: E402
from experiments.run_v22_07_metric_dynamics_fu import _filtered_matrix  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)
PRESERVATION_RUNS = (
    "PRES-project-opposing-grad-cap000",
    "PRES-source-floor-cap005",
    "PRES-dual-ema-blend-cap010",
    "PRES-reflect-opposing-grad-cap000",
    "PRES-row-orth-source-cap005",
    "PRES-row-orth-reflect-cap005",
    "PRES-source-aligned-adamw-cap002",
    "PRES-source-aligned-adamw-cap005",
    "PRES-row-orth-transport-cap002",
    "PRES-row-orth-transport-cap005",
)
CONTROL_RUNS = ("CTRL-AdamW", "CTRL-SGD", "CTRL-NoOp", "CTRL-RandomMatched")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2206_COMBINED_SOURCE))
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--data-root", default="data")
    p.add_argument("--train-size", type=int, default=64)
    p.add_argument("--val-size", type=int, default=48)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--target-refresh", type=int, default=25)
    p.add_argument("--preserve-start", type=int, default=800)
    p.add_argument("--preserve-start-grid", default="")
    p.add_argument("--top-k", type=int, default=3)
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _source_axis(target: torch.Tensor) -> torch.Tensor:
    # Semantic targets are gradient-like; the parameter-space source step is
    # descent along -target.
    return -target


def _remove_opposing_grad(grad: torch.Tensor, source_axis: torch.Tensor) -> tuple[torch.Tensor, float]:
    if grad.numel() == 0 or source_axis.numel() == 0:
        return grad, 0.0
    step = -grad
    dot = torch.dot(step.float(), source_axis.float())
    denom = torch.dot(source_axis.float(), source_axis.float()).clamp_min(1.0e-12)
    if float(dot.item()) >= 0.0:
        return grad, 0.0
    opposing_step = (dot / denom).to(grad.dtype) * source_axis
    new_step = step - opposing_step
    removed = float(torch.linalg.vector_norm(opposing_step.float()).item())
    return -new_step, removed


def _reflect_opposing_grad(grad: torch.Tensor, source_axis: torch.Tensor) -> tuple[torch.Tensor, float]:
    if grad.numel() == 0 or source_axis.numel() == 0:
        return grad, 0.0
    step = -grad
    dot = torch.dot(step.float(), source_axis.float())
    denom = torch.dot(source_axis.float(), source_axis.float()).clamp_min(1.0e-12)
    if float(dot.item()) >= 0.0:
        return grad, 0.0
    opposing_step = (dot / denom).to(grad.dtype) * source_axis
    new_step = step - 2.0 * opposing_step
    removed = float(torch.linalg.vector_norm(opposing_step.float()).item())
    return -new_step, removed


def _cap_and_blend(run_kind: str, default_cap: float = 0.05, default_blend: float = 0.10) -> tuple[float, float]:
    if "cap002" in run_kind:
        return 0.02, 0.02
    if "cap005" in run_kind:
        return 0.05, default_blend
    return default_cap, default_blend


def _row_orthogonal_step(model: torch.nn.Module, flat_step: torch.Tensor) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    offset = 0
    for param in trainable_parameters(model):
        n = int(param.numel())
        step = flat_step[offset : offset + n].view_as(param).to(device=param.device, dtype=param.dtype)
        if param.ndim >= 2:
            w = param.detach().reshape(param.shape[0], -1)
            s = step.reshape(param.shape[0], -1)
            denom = (w.float() * w.float()).sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            coeff = (s.float() * w.float()).sum(dim=1, keepdim=True) / denom
            orth = s - coeff.to(s.dtype) * w
            chunks.append(orth.reshape_as(param).reshape(-1))
        else:
            chunks.append(torch.zeros_like(step).reshape(-1))
        offset += n
    return torch.cat(chunks) if chunks else torch.zeros_like(flat_step)


def _transport_adam_state(model: torch.nn.Module, opt: torch.optim.Optimizer, source_grad: torch.Tensor, blend: float) -> float:
    offset = 0
    moved = 0.0
    for param in trainable_parameters(model):
        n = int(param.numel())
        chunk = source_grad[offset : offset + n].view_as(param).to(device=param.device, dtype=param.dtype)
        offset += n
        state = opt.state.get(param)
        if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
            continue
        old = state["exp_avg"].detach().clone()
        state["exp_avg"].mul_(1.0 - blend).add_(chunk, alpha=blend)
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].mul_(1.0 - blend).addcmul_(chunk, chunk, value=blend)
        moved += float(torch.linalg.vector_norm((state["exp_avg"] - old).float()).item())
    return moved


def _selected_pairs(out_dir: Path, top_k: int) -> list[dict[str, Any]]:
    rows = read_rows(out_dir / "v22_07_c3_semantic_target_repair_matrix.csv")
    candidates = [
        r
        for r in rows
        if str(r.get("run_kind", "")).startswith("SEM-")
        and all(_f(r.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800])
    ]
    candidates = sorted(candidates, key=lambda r: _f(r.get("source_vs_best_control_h800"), -999.0), reverse=True)
    return candidates[: int(top_k)]


def _match_source_row(source_rows: list[dict[str, str]], pair: dict[str, Any]) -> dict[str, str]:
    for row in source_rows:
        if (
            str(row.get("v21_id", "")) == str(pair.get("v21_id", ""))
            and str(row.get("dataset", "")) == str(pair.get("dataset", ""))
            and str(row.get("seed", "")) == str(pair.get("seed", ""))
            and str(row.get("train_seed", "")) == str(pair.get("train_seed", ""))
        ):
            return row
    for row in source_rows:
        if str(row.get("v21_id", "")) == str(pair.get("v21_id", "")) and str(row.get("dataset", "")) == str(pair.get("dataset", "")):
            return row
    return {}


def _train_run(args: argparse.Namespace, source_row: dict[str, str], semantic_kind: str, run_kind: str) -> dict[str, Any]:
    device = resolve_device(args.device)
    dataset = str(source_row.get("dataset", "MNIST"))
    seed = int(float(source_row.get("seed", 0) or 0))
    local = deepcopy(args)
    local.basis_repair_variant = str(source_row.get("basis_repair_variant", "R0-current") or "R0-current")
    x_train, y_train, x_val, y_val = load_dataset(dataset, Path(args.data_root), int(args.train_size), int(args.val_size), seed, device, int(args.input_size))
    train_seed = int(float(source_row.get("train_seed") or 0)) if str(source_row.get("train_seed", "")).strip() else stable_train_seed(source_row)
    model = carrier_model("MLP", x_train, train_seed, local, device)
    opt_adam = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    opt_sgd = torch.optim.SGD(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(191_021 + train_seed + sum(ord(c) for c in run_kind + semantic_kind))
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, Any]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    cached_target = torch.zeros(0, device=device)
    cached_diag: dict[str, Any] = {}
    slow_axis = torch.zeros(0, device=device)
    injection_norm_sum = 0.0
    antiwashout_removed_norm_sum = 0.0
    row_orthogonal_norm_sum = 0.0
    optimizer_transport_norm_sum = 0.0
    negative_projection_count = 0
    refresh_count = 0
    for step in range(1, int(args.steps) + 1):
        idx = torch.randint(0, int(x_train.shape[0]), (batch_size,), generator=gen, device=device)
        xb = x_train[idx]
        yb = y_train[idx]
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        grad = flat_grad(model, device).detach().clone()
        if run_kind == "CTRL-AdamW":
            opt_adam.step()
        elif run_kind == "CTRL-SGD":
            opt_sgd.step()
        elif run_kind == "CTRL-NoOp":
            pass
        elif run_kind == "CTRL-RandomMatched":
            noise = torch.randn(grad.shape, device=device, generator=gen)
            injected = _cap_norm(normalized_like(noise, grad), grad, 0.15)
            _assign_flat_gradient(model, grad + injected)
            opt_adam.step()
            injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        else:
            if cached_target.numel() == 0 or (step - 1) % max(1, int(args.target_refresh)) == 0:
                cached_target, cached_diag = _semantic_target(model, x_train, y_train, semantic_kind, target_scale=1.0)
                axis = _source_axis(cached_target)
                slow_axis = axis if slow_axis.numel() == 0 else 0.995 * slow_axis + 0.005 * axis
                refresh_count += 1
            target = cached_target if "SEM-direct-" in semantic_kind else _orthogonal_component(cached_target, grad)
            cap = _f(cached_diag.get("target_cap_ratio"), 0.10)
            if step <= int(args.preserve_start):
                injected = _cap_norm(target, grad, cap)
                final_grad = grad + injected
            else:
                source_axis = slow_axis if slow_axis.numel() else _source_axis(target)
                step_alignment = torch.dot((-grad).float(), source_axis.float()) if source_axis.numel() else torch.tensor(0.0, device=device)
                if float(step_alignment.item()) < 0.0:
                    negative_projection_count += 1
                if run_kind == "PRES-project-opposing-grad-cap000":
                    final_grad, removed = _remove_opposing_grad(grad, source_axis)
                    injected = final_grad - grad
                elif run_kind == "PRES-reflect-opposing-grad-cap000":
                    final_grad, removed = _reflect_opposing_grad(grad, source_axis)
                    injected = final_grad - grad
                elif run_kind == "PRES-source-floor-cap005":
                    final_grad, removed = _remove_opposing_grad(grad, source_axis)
                    floor = _cap_norm(-source_axis, grad, 0.05)
                    final_grad = final_grad + floor
                    injected = final_grad - grad
                elif run_kind == "PRES-dual-ema-blend-cap010":
                    final_grad, removed = _remove_opposing_grad(grad, source_axis)
                    floor = _cap_norm(-source_axis, grad, 0.10)
                    fresh = _cap_norm(target, grad, max(0.02, 0.50 * cap))
                    final_grad = final_grad + 0.65 * floor + 0.35 * fresh
                    injected = final_grad - grad
                elif run_kind == "PRES-row-orth-source-cap005":
                    final_grad, removed = _remove_opposing_grad(grad, source_axis)
                    row_step = _cap_norm(_row_orthogonal_step(model, source_axis), grad, 0.05)
                    row_grad = -row_step
                    final_grad = final_grad + row_grad
                    injected = final_grad - grad
                    row_orthogonal_norm_sum += float(torch.linalg.vector_norm(row_step.float()).item())
                elif run_kind == "PRES-row-orth-reflect-cap005":
                    final_grad, removed = _reflect_opposing_grad(grad, source_axis)
                    row_step = _cap_norm(_row_orthogonal_step(model, source_axis), grad, 0.05)
                    row_grad = -row_step
                    final_grad = final_grad + row_grad
                    injected = final_grad - grad
                    row_orthogonal_norm_sum += float(torch.linalg.vector_norm(row_step.float()).item())
                elif run_kind in {"PRES-source-aligned-adamw-cap002", "PRES-source-aligned-adamw-cap005"}:
                    floor_cap, state_blend = _cap_and_blend(run_kind)
                    final_grad, removed = _remove_opposing_grad(grad, source_axis)
                    floor = _cap_norm(-source_axis, grad, floor_cap)
                    final_grad = final_grad + floor
                    injected = final_grad - grad
                    optimizer_transport_norm_sum += _transport_adam_state(model, opt_adam, floor, state_blend)
                elif run_kind in {"PRES-row-orth-transport-cap002", "PRES-row-orth-transport-cap005"}:
                    floor_cap, state_blend = _cap_and_blend(run_kind)
                    final_grad, removed = _remove_opposing_grad(grad, source_axis)
                    row_step = _cap_norm(_row_orthogonal_step(model, source_axis), grad, floor_cap)
                    row_grad = -row_step
                    final_grad = final_grad + row_grad
                    injected = final_grad - grad
                    row_orthogonal_norm_sum += float(torch.linalg.vector_norm(row_step.float()).item())
                    optimizer_transport_norm_sum += _transport_adam_state(model, opt_adam, row_grad, state_blend)
                else:
                    final_grad = grad
                    injected = torch.zeros_like(grad)
                    removed = 0.0
                antiwashout_removed_norm_sum += float(removed)
            _assign_flat_gradient(model, final_grad)
            opt_adam.step()
            injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    row: dict[str, Any] = {
        "run_kind": run_kind,
        "semantic_kind": semantic_kind,
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": source_row.get("mechanism", ""),
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "preserve_start": args.preserve_start,
        "target_refresh": args.target_refresh,
        "target_injection_norm_sum": injection_norm_sum,
        "antiwashout_removed_norm_sum": antiwashout_removed_norm_sum,
        "row_orthogonal_norm_sum": row_orthogonal_norm_sum,
        "optimizer_transport_norm_sum": optimizer_transport_norm_sum,
        "negative_projection_count": negative_projection_count,
        "target_refresh_count": refresh_count,
        "official_C2_solver_claimed": 0,
        "execution_status": "measured",
    }
    for h in HORIZONS:
        metrics = traces.get(h, {})
        for key, value in metrics.items():
            row[f"{key}_h{h}"] = value
    return row


def _summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    controls: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        if str(row.get("run_kind", "")).startswith("PRES-"):
            continue
        key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("train_seed")))
        controls.setdefault(key, {})
        for h in HORIZONS:
            val = _f(row.get(f"val_loss_h{h}"))
            if math.isfinite(val):
                controls[key][h] = min(controls[key].get(h, float("inf")), val)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("train_seed")))
        for h in HORIZONS:
            val = _f(row.get(f"val_loss_h{h}"))
            best = controls.get(key, {}).get(h, float("nan"))
            item[f"best_control_val_loss_h{h}"] = best if math.isfinite(best) else ""
            item[f"source_vs_best_control_h{h}"] = best - val if math.isfinite(best) and math.isfinite(val) else ""
        enriched.append(item)
    summary: list[dict[str, Any]] = []
    for row in enriched:
        if not str(row.get("run_kind", "")).startswith("PRES-"):
            continue
        early_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800]))
        full_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        summary.append(
            {
                "run_kind": row.get("run_kind", ""),
                "semantic_kind": row.get("semantic_kind", ""),
                "v21_id": row.get("v21_id", ""),
                "preserve_start": row.get("preserve_start", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "target_injection_norm_sum": row.get("target_injection_norm_sum", ""),
                "antiwashout_removed_norm_sum": row.get("antiwashout_removed_norm_sum", ""),
                "row_orthogonal_norm_sum": row.get("row_orthogonal_norm_sum", ""),
                "optimizer_transport_norm_sum": row.get("optimizer_transport_norm_sum", ""),
                "negative_projection_count": row.get("negative_projection_count", ""),
                "C3_preservation_early_pass": early_pass,
                "C3_preservation_full_pass": full_pass,
                "official_C2_solver_claimed": 0,
                "official_C3_pass": 0,
                "blocker": "" if full_pass else "h3200_retention_or_official_C2_gate_failed",
            }
        )
    route = {
        "C3_preservation_rows": len(summary),
        "C3_preservation_early_pass_rows": sum(int_flag(r.get("C3_preservation_early_pass")) for r in summary),
        "C3_preservation_full_pass_rows": sum(int_flag(r.get("C3_preservation_full_pass")) for r in summary),
        "official_C2_solver_claimed": 0,
        "official_C3_pass_rows": 0,
        "row_positive_count_h100": sum(1 for r in summary if _f(r.get("source_vs_best_control_h100"), -999.0) >= 0.005),
        "row_positive_count_h400": sum(1 for r in summary if _f(r.get("source_vs_best_control_h400"), -999.0) >= 0.005),
        "row_positive_count_h800": sum(1 for r in summary if _f(r.get("source_vs_best_control_h800"), -999.0) >= 0.005),
        "row_positive_count_h1600": sum(1 for r in summary if _f(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005),
        "row_positive_count_h3200": sum(1 for r in summary if _f(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005),
        "route": "C3-PreservationRepairOpened" if any(int_flag(r.get("C3_preservation_full_pass")) for r in summary) else "C3-PreservationRepairBlocked",
        "promotion_allowed": 0,
        "blocker": "official_C2_and_independent_full_C3_not_completed" if any(int_flag(r.get("C3_preservation_full_pass")) for r in summary) else "preservation_h3200_gate_failed",
    }
    return enriched, summary, route


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_rows = _filtered_matrix(Path(args.source_dir), include_controls=1)
    pairs = _selected_pairs(out_dir, int(args.top_k))
    preserve_starts = [int(args.preserve_start)]
    for token in str(args.preserve_start_grid).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(float(token))
        if value not in preserve_starts:
            preserve_starts.append(value)
    raw_rows: list[dict[str, Any]] = []
    seen_controls: set[tuple[str, str, str]] = set()
    for pair in pairs:
        source_row = _match_source_row(source_rows, pair)
        if not source_row:
            continue
        semantic_kind = str(pair.get("run_kind", ""))
        for start in preserve_starts:
            local_args = deepcopy(args)
            local_args.preserve_start = start
            for kind in PRESERVATION_RUNS:
                raw_rows.append(_train_run(local_args, source_row, semantic_kind, kind))
        control_key = (str(source_row.get("dataset", "")), str(source_row.get("seed", "")), str(source_row.get("train_seed", "")))
        if control_key not in seen_controls:
            seen_controls.add(control_key)
            for kind in CONTROL_RUNS:
                raw_rows.append(_train_run(args, source_row, semantic_kind, kind))
    enriched, summary, route = _summarize(raw_rows)
    write_rows(out_dir / "v22_07_c3_preservation_repair_raw_runs.csv", raw_rows)
    write_rows(out_dir / "v22_07_c3_preservation_repair_matrix.csv", enriched)
    write_rows(out_dir / "v22_07_c3_preservation_repair_summary.csv", summary)
    write_json(out_dir / "v22_07_c3_preservation_repair_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_c3_preservation_repair.py --source-dir {args.source_dir} --device {args.device} --out-dir {out_dir}"
        + f" --top-k {args.top_k} --steps {args.steps} --preserve-start {args.preserve_start} --preserve-start-grid {args.preserve_start_grid} --target-refresh {args.target_refresh}",
        status="completed",
        note=(
            f"rows={route['C3_preservation_rows']} early_pass={route['C3_preservation_early_pass_rows']} "
            f"h1600_pos={route['row_positive_count_h1600']} h3200_pos={route['row_positive_count_h3200']} route={route['route']}"
        ),
    )


if __name__ == "__main__":
    main()
