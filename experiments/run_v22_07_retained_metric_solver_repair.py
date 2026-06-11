#!/usr/bin/env python3
"""v22.07 retained-source metric-solver repair smoke.

This stage is entered after semantic targets open h100/h400/h800 but
optimizer-state preservation still washes out at h1600/h3200.  It stops
same-family semantic preservation tweaks and reuses the audited v22.06
readout metric solver for retained-source target families.

The source-theory top-candidate file is used only as a train-only ranking
artifact.  Future source columns in that file remain audit labels and are not
used to build directions.
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

from dgkan.fu.core import UpdateTensor, apply_update, flat_grad, normalized_like  # noqa: E402
from dgkan.fu.metric_solver import V2206_SOLVER_CONFIGS, solve_metric_readout_update  # noqa: E402
from experiments.run_v17_common import carrier_model, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_07_c3_semantic_target_repair import _eval  # noqa: E402
from experiments.run_v22_07_common import PYTHON, V2206_COMBINED_SOURCE, append_exec, ensure_out, finite_float, int_flag, read_rows, write_json, write_rows  # noqa: E402
from experiments.run_v22_07_metric_dynamics_fu import _filtered_matrix  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)
DEFAULT_SOLVERS = (
    "M263-V2206MetricSolverT5G6LowNDSFU",
    "M264-V2206MetricSolverT6G0DualMemoryFU",
    "M269-V2206MetricSolverT11G0EarlyObservableFU",
    "M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU",
)
COMMIT_VARIANTS = ("direct", "slowmix025")
CONTROL_RUNS = ("CTRL-AdamW", "CTRL-SGD", "CTRL-NoOp", "CTRL-RandomMetricMatched")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2206_COMBINED_SOURCE))
    p.add_argument("--selection-file", default="")
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
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--solver-interval", type=int, default=100)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--solver-grid", default=",".join(DEFAULT_SOLVERS))
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _selected_source_rows(args: argparse.Namespace, source_rows: list[dict[str, str]], out_dir: Path) -> list[dict[str, str]]:
    selection_file = Path(args.selection_file) if str(args.selection_file).strip() else out_dir / "v22_07_source_theory_top_candidates.csv"
    top = read_rows(selection_file)
    by_order = {idx: row for idx, row in enumerate(source_rows)}
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in top:
        row = by_order.get(int(float(item.get("job_order", -1))))
        if not row:
            continue
        key = (str(row.get("v21_id", "")), str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("train_seed", "")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= int(args.top_k):
            break
    return selected


def _solver_names(args: argparse.Namespace, source_row: dict[str, str]) -> list[str]:
    names = [x.strip() for x in str(args.solver_grid).replace(";", ",").split(",") if x.strip()]
    own = str(source_row.get("mechanism", ""))
    if own in V2206_SOLVER_CONFIGS and own not in names:
        names.insert(0, own)
    out: list[str] = []
    for name in names:
        if name in V2206_SOLVER_CONFIGS and name not in out:
            out.append(name)
    return out


def _quality_pass(diag: dict[str, Any]) -> int:
    return int(
        str(diag.get("solver_status", "")) == "metric_readout_exact_solve"
        and not int_flag(diag.get("is_proxy"))
        and _f(diag.get("projection_residual_Gf"), 999.0) <= 0.35
        and _f(diag.get("ActuationR2"), -999.0) >= 0.60
        and _f(diag.get("B2_transfer_gain"), -999.0) >= 0.005
        and _f(diag.get("B3_safety_gain"), -999.0) >= -0.005
    )


def _blend_update(update: UpdateTensor, slow: torch.Tensor, variant: str) -> tuple[UpdateTensor, torch.Tensor, float]:
    vec = update.tensor.detach().clone()
    if slow.numel() == 0:
        next_slow = vec.detach().clone()
    else:
        next_slow = 0.995 * slow + 0.005 * vec
    if variant == "slowmix025" and next_slow.numel() == vec.numel():
        mixed = 0.75 * vec + 0.25 * next_slow.to(device=vec.device, dtype=vec.dtype)
    else:
        mixed = vec
    mixed_update = UpdateTensor(
        mixed.detach().clone(),
        update.kind,
        update.sign_rule,
        update.space,
        f"v22_07_retained_metric_solver_{variant}",
        update.mechanism,
        role=f"retained_metric_solver_{variant}",
        one_step_descent_claim=update.one_step_descent_claim,
        diagnostics=dict(update.diagnostics or {}),
    )
    carry_norm = float(torch.linalg.vector_norm(next_slow.float()).item()) if next_slow.numel() else 0.0
    return mixed_update, next_slow.detach().clone(), carry_norm


def _run_train(args: argparse.Namespace, source_row: dict[str, str], run_kind: str, solver_name: str = "", commit_variant: str = "") -> dict[str, Any]:
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
    gen = torch.Generator(device=device).manual_seed(229_071 + train_seed + sum(ord(c) for c in run_kind + solver_name + commit_variant))
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, Any]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    solver_count = 0
    solver_proxy_count = 0
    solver_quality_pass_count = 0
    solver_projection_residual_sum = 0.0
    solver_actuation_r2_sum = 0.0
    solver_b2_sum = 0.0
    solver_b3_min = float("inf")
    solver_update_norm_sum = 0.0
    random_update_norm_sum = 0.0
    slow_memory_norm_last = 0.0
    slow_state = torch.zeros(0, device=device)
    cfg = V2206_SOLVER_CONFIGS.get(solver_name, {})
    for step in range(1, int(args.steps) + 1):
        idx = torch.randint(0, int(x_train.shape[0]), (batch_size,), generator=gen, device=device)
        xb = x_train[idx]
        yb = y_train[idx]
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        if run_kind == "CTRL-SGD":
            opt_sgd.step()
        elif run_kind == "CTRL-NoOp":
            pass
        else:
            opt_adam.step()
        if run_kind == "CTRL-RandomMetricMatched" and step % max(1, int(args.solver_interval)) == 0:
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb).float(), yb).backward()
            grad = flat_grad(model, device).detach()
            noise = torch.randn(grad.shape, device=device, generator=gen)
            vec = normalized_like(noise, grad)
            update = UpdateTensor(vec, "matched_random_metric_solver_control", "subtract", "random_parameter_control", "v22_07_retained_metric_solver_random_control", run_kind, one_step_descent_claim=0)
            apply_update(model, update, lr=float(args.fu_lr))
            random_update_norm_sum += float(torch.linalg.vector_norm(vec.float()).item())
        elif run_kind.startswith("RET-") and step % max(1, int(args.solver_interval)) == 0:
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb).float(), yb).backward()
            update = solve_metric_readout_update(
                model,
                xb,
                yb,
                target_family=str(cfg.get("target_family", "T5-LowNDS")),
                metric_family=str(cfg.get("metric_family", "G6-LowNDS")),
                mechanism=solver_name,
                hidden_residual_scale=float(cfg.get("hidden_residual_scale", 0.35)),
                hidden_function_fraction=float(cfg.get("hidden_function_fraction", 0.25)),
                seed=train_seed + step,
            )
            mixed, slow_state, slow_memory_norm_last = _blend_update(update, slow_state, commit_variant)
            apply_update(model, mixed, lr=float(args.fu_lr))
            diag = dict(update.diagnostics or {})
            solver_count += 1
            solver_proxy_count += int_flag(diag.get("is_proxy"))
            solver_quality_pass_count += _quality_pass(diag)
            solver_projection_residual_sum += _f(diag.get("projection_residual_Gf"), 999.0)
            solver_actuation_r2_sum += _f(diag.get("ActuationR2"), -999.0)
            solver_b2_sum += _f(diag.get("B2_transfer_gain"), 0.0)
            solver_b3_min = min(solver_b3_min, _f(diag.get("B3_safety_gain"), 999.0))
            solver_update_norm_sum += float(torch.linalg.vector_norm(update.tensor.detach().float()).item()) if update.tensor.numel() else 0.0
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    row: dict[str, Any] = {
        "run_kind": run_kind,
        "solver_name": solver_name,
        "commit_variant": commit_variant,
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": source_row.get("mechanism", ""),
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "solver_interval": args.solver_interval,
        "fu_lr": args.fu_lr,
        "solver_count": solver_count,
        "solver_proxy_count": solver_proxy_count,
        "solver_quality_pass_count": solver_quality_pass_count,
        "solver_projection_residual_mean": solver_projection_residual_sum / solver_count if solver_count else "",
        "solver_ActuationR2_mean": solver_actuation_r2_sum / solver_count if solver_count else "",
        "solver_B2_transfer_gain_mean": solver_b2_sum / solver_count if solver_count else "",
        "solver_B3_safety_gain_min": solver_b3_min if math.isfinite(solver_b3_min) else "",
        "solver_update_norm_sum": solver_update_norm_sum,
        "slow_memory_norm_last": slow_memory_norm_last,
        "random_update_norm_sum": random_update_norm_sum,
        "official_C2_solver_reused": int(run_kind.startswith("RET-")),
        "official_C2_solver_claimed": int(run_kind.startswith("RET-") and solver_count > 0 and solver_proxy_count == 0 and solver_quality_pass_count == solver_count and commit_variant == "direct"),
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
        if str(row.get("run_kind", "")).startswith("RET-"):
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
        if not str(row.get("run_kind", "")).startswith("RET-"):
            continue
        early_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800]))
        full_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        official_c2 = int_flag(row.get("official_C2_solver_claimed"))
        summary.append(
            {
                "run_kind": row.get("run_kind", ""),
                "solver_name": row.get("solver_name", ""),
                "commit_variant": row.get("commit_variant", ""),
                "v21_id": row.get("v21_id", ""),
                "dataset": row.get("dataset", ""),
                "train_seed": row.get("train_seed", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "solver_count": row.get("solver_count", ""),
                "solver_proxy_count": row.get("solver_proxy_count", ""),
                "solver_quality_pass_count": row.get("solver_quality_pass_count", ""),
                "solver_projection_residual_mean": row.get("solver_projection_residual_mean", ""),
                "solver_ActuationR2_mean": row.get("solver_ActuationR2_mean", ""),
                "solver_B2_transfer_gain_mean": row.get("solver_B2_transfer_gain_mean", ""),
                "solver_B3_safety_gain_min": row.get("solver_B3_safety_gain_min", ""),
                "solver_update_norm_sum": row.get("solver_update_norm_sum", ""),
                "slow_memory_norm_last": row.get("slow_memory_norm_last", ""),
                "C3_retained_solver_early_pass": early_pass,
                "C3_retained_solver_full_pass": full_pass,
                "official_C2_solver_claimed": official_c2,
                "official_C3_pass": int(full_pass and official_c2),
                "blocker": "" if full_pass and official_c2 else "retained_h3200_or_official_C2_gate_failed",
            }
        )
    route = {
        "C3_retained_solver_rows": len(summary),
        "C3_retained_solver_early_pass_rows": sum(int_flag(r.get("C3_retained_solver_early_pass")) for r in summary),
        "C3_retained_solver_full_pass_rows": sum(int_flag(r.get("C3_retained_solver_full_pass")) for r in summary),
        "official_C2_solver_claimed_rows": sum(int_flag(r.get("official_C2_solver_claimed")) for r in summary),
        "official_C3_pass_rows": sum(int_flag(r.get("official_C3_pass")) for r in summary),
        "row_positive_count_h100": sum(1 for r in summary if _f(r.get("source_vs_best_control_h100"), -999.0) >= 0.005),
        "row_positive_count_h400": sum(1 for r in summary if _f(r.get("source_vs_best_control_h400"), -999.0) >= 0.005),
        "row_positive_count_h800": sum(1 for r in summary if _f(r.get("source_vs_best_control_h800"), -999.0) >= 0.005),
        "row_positive_count_h1600": sum(1 for r in summary if _f(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005),
        "row_positive_count_h3200": sum(1 for r in summary if _f(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005),
        "route": "C3-RetainedMetricSolverOpened" if any(int_flag(r.get("official_C3_pass")) for r in summary) else "C3-RetainedMetricSolverBlocked",
        "promotion_allowed": 0,
        "blocker": "none" if any(int_flag(r.get("official_C3_pass")) for r in summary) else "retained_metric_solver_h3200_or_C2_gate_failed",
    }
    return enriched, summary, route


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_rows = _filtered_matrix(Path(args.source_dir), include_controls=1)
    selected = _selected_source_rows(args, source_rows, out_dir)
    raw_rows: list[dict[str, Any]] = []
    seen_controls: set[tuple[str, str, str]] = set()
    for source_row in selected:
        for solver_name in _solver_names(args, source_row):
            for variant in COMMIT_VARIANTS:
                raw_rows.append(_run_train(args, source_row, f"RET-{solver_name}-{variant}", solver_name, variant))
        control_key = (str(source_row.get("dataset", "")), str(source_row.get("seed", "")), str(source_row.get("train_seed", "")))
        if control_key not in seen_controls:
            seen_controls.add(control_key)
            for kind in CONTROL_RUNS:
                raw_rows.append(_run_train(args, source_row, kind))
    enriched, summary, route = _summarize(raw_rows)
    write_rows(out_dir / "v22_07_retained_metric_solver_repair_raw_runs.csv", raw_rows)
    write_rows(out_dir / "v22_07_retained_metric_solver_repair_matrix.csv", enriched)
    write_rows(out_dir / "v22_07_retained_metric_solver_repair_summary.csv", summary)
    write_json(out_dir / "v22_07_retained_metric_solver_repair_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_retained_metric_solver_repair.py --source-dir {args.source_dir} --device {args.device} --out-dir {out_dir}"
        + f" --top-k {args.top_k} --steps {args.steps} --solver-interval {args.solver_interval} --fu-lr {args.fu_lr} --solver-grid {args.solver_grid}",
        status="completed",
        note=(
            f"rows={route['C3_retained_solver_rows']} early={route['C3_retained_solver_early_pass_rows']} "
            f"full={route['C3_retained_solver_full_pass_rows']} official_C2_rows={route['official_C2_solver_claimed_rows']} "
            f"h1600_pos={route['row_positive_count_h1600']} h3200_pos={route['row_positive_count_h3200']} route={route['route']}"
        ),
    )


if __name__ == "__main__":
    main()
