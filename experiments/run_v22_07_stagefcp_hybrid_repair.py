#!/usr/bin/env python3
"""v22.07 Stage F/C/P hybrid source repair smoke.

Semantic target repair opened transient h100/h400/h800 source, while retained
metric-solver and preservation variants failed h1600/h3200.  This runner tests
the plan's staged interpretation directly:

  Stage F: train-only semantic target for early formation.
  Stage C/P: audited v22.06 metric readout solver for consolidation/preservation.

The runner remains a smoke repair.  It reports whether the metric-solver phase
meets repeated C2 quality; it does not promote any row unless both official C2
and full C3 gates are satisfied.
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

from dgkan.fu.core import UpdateTensor, apply_update, flat_grad  # noqa: E402
from dgkan.fu.metric_solver import V2206_SOLVER_CONFIGS, solve_metric_readout_update  # noqa: E402
from experiments.run_v17_common import carrier_model, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_07_c3_semantic_target_repair import _assign_flat_gradient, _cap_norm, _eval, _orthogonal_component, _semantic_target  # noqa: E402
from experiments.run_v22_07_common import PYTHON, V2206_COMBINED_SOURCE, append_exec, ensure_out, finite_float, int_flag, read_rows, write_json, write_rows  # noqa: E402
from experiments.run_v22_07_metric_dynamics_fu import _filtered_matrix  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)
SOLVERS = (
    "M264-V2206MetricSolverT6G0DualMemoryFU",
    "M269-V2206MetricSolverT11G0EarlyObservableFU",
    "M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU",
)
SWITCH_STEPS = (400, 800)
CONTROL_RUNS = ("CTRL-AdamW", "CTRL-SGD", "CTRL-NoOp", "CTRL-RandomHybridMatched")


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
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--semantic-refresh", type=int, default=25)
    p.add_argument("--solver-interval", type=int, default=100)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--solver-grid", default=",".join(SOLVERS))
    p.add_argument("--switch-grid", default=",".join(str(x) for x in SWITCH_STEPS))
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _semantic_pairs(out_dir: Path, top_k: int) -> list[dict[str, Any]]:
    rows = read_rows(out_dir / "v22_07_c3_semantic_target_repair_matrix.csv")
    candidates = [
        r
        for r in rows
        if str(r.get("run_kind", "")).startswith("SEM-")
        and all(_f(r.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800])
    ]
    return sorted(candidates, key=lambda r: _f(r.get("source_vs_best_control_h800"), -999.0), reverse=True)[: int(top_k)]


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


def _quality_pass(diag: dict[str, Any]) -> int:
    return int(
        str(diag.get("solver_status", "")) == "metric_readout_exact_solve"
        and not int_flag(diag.get("is_proxy"))
        and _f(diag.get("projection_residual_Gf"), 999.0) <= 0.35
        and _f(diag.get("ActuationR2"), -999.0) >= 0.60
        and _f(diag.get("B2_transfer_gain"), -999.0) >= 0.005
        and _f(diag.get("B3_safety_gain"), -999.0) >= -0.005
    )


def _run_train(args: argparse.Namespace, source_row: dict[str, str], semantic_kind: str, run_kind: str, solver_name: str = "", switch_step: int = 0) -> dict[str, Any]:
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
    gen = torch.Generator(device=device).manual_seed(227_071 + train_seed + sum(ord(c) for c in run_kind + solver_name + semantic_kind))
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, Any]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    cached_target = torch.zeros(0, device=device)
    cached_diag: dict[str, Any] = {}
    semantic_injection_norm_sum = 0.0
    solver_update_norm_sum = 0.0
    random_update_norm_sum = 0.0
    solver_count = 0
    solver_proxy_count = 0
    solver_quality_pass_count = 0
    solver_projection_residual_sum = 0.0
    solver_actuation_sum = 0.0
    solver_b2_sum = 0.0
    solver_b3_min = float("inf")
    cfg = V2206_SOLVER_CONFIGS.get(solver_name, {})
    for step in range(1, int(args.steps) + 1):
        idx = torch.randint(0, int(x_train.shape[0]), (batch_size,), generator=gen, device=device)
        xb = x_train[idx]
        yb = y_train[idx]
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        grad = flat_grad(model, device).detach().clone()
        if run_kind == "CTRL-SGD":
            opt_sgd.step()
        elif run_kind == "CTRL-NoOp":
            pass
        elif run_kind == "CTRL-RandomHybridMatched":
            opt_adam.step()
            if step % max(1, int(args.solver_interval)) == 0:
                noise = torch.randn(grad.shape, device=device, generator=gen)
                vec = _cap_norm(noise, grad, 0.10)
                update = UpdateTensor(vec, "matched_random_hybrid_control", "subtract", "random_parameter_control", "v22_07_stagefcp_random_control", run_kind, one_step_descent_claim=0)
                apply_update(model, update, lr=float(args.fu_lr))
                random_update_norm_sum += float(torch.linalg.vector_norm(vec.float()).item())
        elif run_kind.startswith("HYB-") and step <= int(switch_step):
            if cached_target.numel() == 0 or (step - 1) % max(1, int(args.semantic_refresh)) == 0:
                cached_target, cached_diag = _semantic_target(model, x_train, y_train, semantic_kind, target_scale=1.0)
            target = cached_target if "SEM-direct-" in semantic_kind else _orthogonal_component(cached_target, grad)
            injected = _cap_norm(target, grad, _f(cached_diag.get("target_cap_ratio"), 0.10))
            _assign_flat_gradient(model, grad + injected)
            opt_adam.step()
            semantic_injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        elif run_kind.startswith("HYB-"):
            opt_adam.step()
            if step % max(1, int(args.solver_interval)) == 0:
                model.zero_grad(set_to_none=True)
                F.cross_entropy(model(xb).float(), yb).backward()
                update = solve_metric_readout_update(
                    model,
                    xb,
                    yb,
                    target_family=str(cfg.get("target_family", "T6-DualMemorySource")),
                    metric_family=str(cfg.get("metric_family", "G0-L2")),
                    mechanism=solver_name,
                    hidden_residual_scale=float(cfg.get("hidden_residual_scale", 0.35)),
                    hidden_function_fraction=float(cfg.get("hidden_function_fraction", 0.25)),
                    seed=train_seed + step,
                )
                apply_update(model, update, lr=float(args.fu_lr))
                diag = dict(update.diagnostics or {})
                solver_count += 1
                solver_proxy_count += int_flag(diag.get("is_proxy"))
                solver_quality_pass_count += _quality_pass(diag)
                solver_projection_residual_sum += _f(diag.get("projection_residual_Gf"), 999.0)
                solver_actuation_sum += _f(diag.get("ActuationR2"), -999.0)
                solver_b2_sum += _f(diag.get("B2_transfer_gain"), 0.0)
                solver_b3_min = min(solver_b3_min, _f(diag.get("B3_safety_gain"), 999.0))
                solver_update_norm_sum += float(torch.linalg.vector_norm(update.tensor.detach().float()).item()) if update.tensor.numel() else 0.0
        else:
            opt_adam.step()
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    row: dict[str, Any] = {
        "run_kind": run_kind,
        "semantic_kind": semantic_kind,
        "solver_name": solver_name,
        "switch_step": switch_step,
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": source_row.get("mechanism", ""),
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "semantic_injection_norm_sum": semantic_injection_norm_sum,
        "solver_update_norm_sum": solver_update_norm_sum,
        "random_update_norm_sum": random_update_norm_sum,
        "solver_count": solver_count,
        "solver_proxy_count": solver_proxy_count,
        "solver_quality_pass_count": solver_quality_pass_count,
        "solver_projection_residual_mean": solver_projection_residual_sum / solver_count if solver_count else "",
        "solver_ActuationR2_mean": solver_actuation_sum / solver_count if solver_count else "",
        "solver_B2_transfer_gain_mean": solver_b2_sum / solver_count if solver_count else "",
        "solver_B3_safety_gain_min": solver_b3_min if math.isfinite(solver_b3_min) else "",
        "official_C2_solver_claimed": int(run_kind.startswith("HYB-") and solver_count > 0 and solver_proxy_count == 0 and solver_quality_pass_count == solver_count),
        "execution_status": "measured",
    }
    for h in HORIZONS:
        for key, value in traces.get(h, {}).items():
            row[f"{key}_h{h}"] = value
    return row


def _summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    controls: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        if str(row.get("run_kind", "")).startswith("HYB-"):
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
        if not str(row.get("run_kind", "")).startswith("HYB-"):
            continue
        early_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800]))
        full_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        official_c2 = int_flag(row.get("official_C2_solver_claimed"))
        summary.append(
            {
                "run_kind": row.get("run_kind", ""),
                "semantic_kind": row.get("semantic_kind", ""),
                "solver_name": row.get("solver_name", ""),
                "switch_step": row.get("switch_step", ""),
                "v21_id": row.get("v21_id", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "semantic_injection_norm_sum": row.get("semantic_injection_norm_sum", ""),
                "solver_update_norm_sum": row.get("solver_update_norm_sum", ""),
                "solver_count": row.get("solver_count", ""),
                "solver_quality_pass_count": row.get("solver_quality_pass_count", ""),
                "solver_projection_residual_mean": row.get("solver_projection_residual_mean", ""),
                "solver_ActuationR2_mean": row.get("solver_ActuationR2_mean", ""),
                "solver_B2_transfer_gain_mean": row.get("solver_B2_transfer_gain_mean", ""),
                "C3_stagefcp_early_pass": early_pass,
                "C3_stagefcp_full_pass": full_pass,
                "official_C2_solver_claimed": official_c2,
                "official_C3_pass": int(full_pass and official_c2),
                "blocker": "" if full_pass and official_c2 else "stagefcp_h3200_or_official_C2_gate_failed",
            }
        )
    route = {
        "C3_stagefcp_rows": len(summary),
        "C3_stagefcp_early_pass_rows": sum(int_flag(r.get("C3_stagefcp_early_pass")) for r in summary),
        "C3_stagefcp_full_pass_rows": sum(int_flag(r.get("C3_stagefcp_full_pass")) for r in summary),
        "official_C2_solver_claimed_rows": sum(int_flag(r.get("official_C2_solver_claimed")) for r in summary),
        "official_C3_pass_rows": sum(int_flag(r.get("official_C3_pass")) for r in summary),
        "row_positive_count_h100": sum(1 for r in summary if _f(r.get("source_vs_best_control_h100"), -999.0) >= 0.005),
        "row_positive_count_h400": sum(1 for r in summary if _f(r.get("source_vs_best_control_h400"), -999.0) >= 0.005),
        "row_positive_count_h800": sum(1 for r in summary if _f(r.get("source_vs_best_control_h800"), -999.0) >= 0.005),
        "row_positive_count_h1600": sum(1 for r in summary if _f(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005),
        "row_positive_count_h3200": sum(1 for r in summary if _f(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005),
        "route": "C3-StageFCPHybridOpened" if any(int_flag(r.get("official_C3_pass")) for r in summary) else "C3-StageFCPHybridBlocked",
        "promotion_allowed": 0,
        "blocker": "none" if any(int_flag(r.get("official_C3_pass")) for r in summary) else "stagefcp_h3200_or_C2_gate_failed",
    }
    return enriched, summary, route


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_rows = _filtered_matrix(Path(args.source_dir), include_controls=1)
    pairs = _semantic_pairs(out_dir, int(args.top_k))
    solvers = [x.strip() for x in str(args.solver_grid).replace(";", ",").split(",") if x.strip() and x.strip() in V2206_SOLVER_CONFIGS]
    switches = [int(float(x)) for x in str(args.switch_grid).replace(";", ",").split(",") if x.strip()]
    raw_rows: list[dict[str, Any]] = []
    seen_controls: set[tuple[str, str, str]] = set()
    for pair in pairs:
        source_row = _match_source_row(source_rows, pair)
        if not source_row:
            continue
        semantic_kind = str(pair.get("run_kind", ""))
        for switch in switches:
            for solver in solvers:
                raw_rows.append(_run_train(args, source_row, semantic_kind, f"HYB-semantic{switch}-then-{solver}", solver, switch))
        control_key = (str(source_row.get("dataset", "")), str(source_row.get("seed", "")), str(source_row.get("train_seed", "")))
        if control_key not in seen_controls:
            seen_controls.add(control_key)
            for kind in CONTROL_RUNS:
                raw_rows.append(_run_train(args, source_row, semantic_kind, kind))
    enriched, summary, route = _summarize(raw_rows)
    write_rows(out_dir / "v22_07_stagefcp_hybrid_repair_raw_runs.csv", raw_rows)
    write_rows(out_dir / "v22_07_stagefcp_hybrid_repair_matrix.csv", enriched)
    write_rows(out_dir / "v22_07_stagefcp_hybrid_repair_summary.csv", summary)
    write_json(out_dir / "v22_07_stagefcp_hybrid_repair_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_stagefcp_hybrid_repair.py --source-dir {args.source_dir} --device {args.device} --out-dir {out_dir}"
        + f" --top-k {args.top_k} --steps {args.steps} --semantic-refresh {args.semantic_refresh} --solver-interval {args.solver_interval}"
        + f" --fu-lr {args.fu_lr} --solver-grid {args.solver_grid} --switch-grid {args.switch_grid}",
        status="completed",
        note=(
            f"rows={route['C3_stagefcp_rows']} early={route['C3_stagefcp_early_pass_rows']} "
            f"full={route['C3_stagefcp_full_pass_rows']} official_C2_rows={route['official_C2_solver_claimed_rows']} "
            f"h1600_pos={route['row_positive_count_h1600']} h3200_pos={route['row_positive_count_h3200']} route={route['route']}"
        ),
    )


if __name__ == "__main__":
    main()
