#!/usr/bin/env python3
"""v22.07 Observer6 train-only curvature-guarded source target smoke.

Observer5's activation-manifold target opened only a transient early chain.
Observer6 tests the plan's Muon/NDS hypothesis directly: early source may be
washed out because the target lives in high-curvature directions.  It builds a
train-only activation target, estimates its Hessian-vector product on training
data, removes the high-curvature component, and optionally carries a slow EMA
target after the early formation phase.

No validation/test/future source is used to build directions.  This is a smoke
path and does not claim official C2 because it is not a metric-defined solver.
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

from dgkan.fu.core import UpdateTensor, flat_grad, normalized_like, trainable_parameters  # noqa: E402
from experiments.run_v17_common import carrier_model, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_07_c3_semantic_target_repair import _assign_flat_gradient, _cap_norm, _eval, _orthogonal_component, _remove_positive_component  # noqa: E402
from experiments.run_v22_07_common import PYTHON, V2206_COMBINED_SOURCE, append_exec, ensure_out, finite_float, int_flag, read_rows, write_json, write_rows  # noqa: E402
from experiments.run_v22_07_metric_dynamics_fu import _filtered_matrix, _simulate_gains  # noqa: E402
from experiments.run_v22_07_observer4_transfer_meta import _selected_rows  # noqa: E402
from experiments.run_v22_07_observer5_activation_manifold import _manifold_target  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)
CURV_RUNS = (
    "OBS6-neighbor-lownds-cap006",
    "OBS6-neighbor-lownds-longema-cap004",
    "OBS6-centroid-lownds-cap006",
    "OBS6-neighbor-lownds-preserve800-cap004",
)
CONTROL_RUNS = ("CTRL-AdamW", "CTRL-SGD", "CTRL-NoOp", "CTRL-RandomCurvMatched")


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
    p.add_argument("--target-refresh", type=int, default=100)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--target-scale", type=float, default=16.0)
    p.add_argument("--ema-beta", type=float, default=0.90)
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _base_kind(run_kind: str) -> str:
    if "centroid" in run_kind:
        return "OBS5-centroid-info-consensus-cap010"
    return "OBS5-neighbor-preserve-cap010"


def _hvp(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    params = trainable_parameters(model)
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x).float(), y)
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    flat = torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])
    dot = torch.dot(flat.float(), vec.detach().float())
    hvps = torch.autograd.grad(dot, params, allow_unused=True)
    out = torch.cat([
        (h if h is not None else torch.zeros_like(p)).detach().reshape(-1)
        for h, p in zip(hvps, params)
    ])
    model.zero_grad(set_to_none=True)
    return out.to(device=vec.device, dtype=vec.dtype)


def _nds(vec: torch.Tensor, hv: torch.Tensor) -> float:
    denom = torch.dot(vec.float(), vec.float()).clamp_min(1.0e-12)
    return float((torch.dot(vec.float(), hv.float()) / denom).item())


def _curvature_guard(vec: torch.Tensor, hv: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    raw_nds = _nds(vec, hv)
    denom = torch.dot(hv.float(), hv.float()).clamp_min(1.0e-12)
    coeff = torch.dot(vec.float(), hv.float()) / denom
    clipped = torch.clamp(coeff, min=0.0, max=0.75)
    removed = clipped.to(dtype=vec.dtype) * hv
    guarded = vec - removed
    return guarded, {
        "raw_NDS": raw_nds,
        "guarded_NDS": _nds(guarded, hv),
        "curvature_component_coeff": float(clipped.item()),
        "curvature_removed_norm": float(torch.linalg.vector_norm(removed.float()).item()),
        "hvp_norm": float(torch.linalg.vector_norm(hv.float()).item()),
    }


def _curvature_target(
    args: argparse.Namespace,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    run_kind: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target, diag = _manifold_target(args, model, x, y, _base_kind(run_kind))
    hv = _hvp(model, x, y, target)
    guarded, curv = _curvature_guard(target, hv)
    corrupt = _hvp(model, x, (y + 1) % max(2, int(y.max().item()) + 1), guarded)
    guarded, removed_corrupt = _remove_positive_component(guarded, corrupt)
    ce = flat_grad(model, x.device).detach().clone()
    if ce.numel() != guarded.numel() or float(torch.linalg.vector_norm(ce.float()).item()) <= 1.0e-12:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).float(), y).backward()
        ce = flat_grad(model, x.device).detach().clone()
    guarded = normalized_like(guarded, ce) * float(args.target_scale)
    cap_ratio = 0.04 if "cap004" in run_kind else 0.06
    diag = {
        **diag,
        **curv,
        "observer_family": run_kind,
        "base_target_family": _base_kind(run_kind),
        "target_role": "train_only_curvature_guarded_activation_target",
        "target_cap_ratio": cap_ratio,
        "corrupt_hvp_removed_norm": removed_corrupt,
        "uses_future_source_for_direction": 0,
        "official_C2_solver_claimed": 0,
    }
    return guarded.detach().clone(), diag


def _c1_rows(args: argparse.Namespace, selected: list[dict[str, str]]) -> list[dict[str, Any]]:
    device = resolve_device(args.device)
    rows: list[dict[str, Any]] = []
    for source_row in selected:
        dataset = str(source_row.get("dataset", "MNIST"))
        seed = int(float(source_row.get("seed", 0) or 0))
        local = deepcopy(args)
        local.basis_repair_variant = str(source_row.get("basis_repair_variant", "R0-current") or "R0-current")
        x_train, y_train, _x_val, _y_val = load_dataset(dataset, Path(args.data_root), int(args.train_size), int(args.val_size), seed, device, int(args.input_size))
        train_seed = int(float(source_row.get("train_seed") or 0)) if str(source_row.get("train_seed", "")).strip() else stable_train_seed(source_row)
        model = carrier_model("MLP", x_train, train_seed, local, device)
        xb = x_train[: min(int(args.batch_size), int(x_train.shape[0]))]
        yb = y_train[: int(xb.shape[0])]
        for run_kind in CURV_RUNS:
            target, diag = _curvature_target(args, model, xb, yb, run_kind)
            update = UpdateTensor(target, run_kind, "subtract", "parameter", "v22_07_observer6_curvature_guard", str(source_row.get("mechanism", "")), one_step_descent_claim=1, diagnostics=diag)
            gen = torch.Generator(device=device).manual_seed(train_seed + sum(ord(c) for c in run_kind) + 26_071)
            noise = torch.randn(target.shape, device=device, generator=gen)
            random_update = UpdateTensor(normalized_like(noise, target), "matched_random_curv", "subtract", "parameter", "v22_07_observer6_random", "CTRL-RandomCurvMatched", one_step_descent_claim=0)
            sign_update = UpdateTensor(target, "sign_flip_curv", "add", "parameter", "v22_07_observer6_sign_flip", "CTRL-SignFlipCurv", one_step_descent_claim=0)
            corrupt_target, _cdiag = _curvature_target(args, model, xb, (yb + 1) % max(2, int(args.classes)), run_kind)
            corrupt_update = UpdateTensor(corrupt_target, "corrupt_curv", "subtract", "parameter", "v22_07_observer6_corrupt", "CTRL-CorruptCurv", one_step_descent_claim=0)
            true_gains = _simulate_gains(model, update, float(args.fu_lr), xb, yb, int(args.classes))
            random_gains = _simulate_gains(model, random_update, float(args.fu_lr), xb, yb, int(args.classes))
            sign_gains = _simulate_gains(model, sign_update, float(args.fu_lr), xb, yb, int(args.classes))
            corrupt_gains = _simulate_gains(model, corrupt_update, float(args.fu_lr), xb, yb, int(args.classes))
            c1_pass = int(
                true_gains["B2_transfer_gain"] >= random_gains["B2_transfer_gain"] + 0.005
                and true_gains["B2_transfer_gain"] >= sign_gains["B2_transfer_gain"] + 0.005
                and true_gains["B2_transfer_gain"] >= corrupt_gains["B2_transfer_gain"] + 0.005
                and true_gains["B3_safety_gain"] >= -0.005
            )
            rows.append(
                {
                    "run_kind": run_kind,
                    "v21_id": source_row.get("v21_id", ""),
                    "mechanism": source_row.get("mechanism", ""),
                    "dataset": dataset,
                    "seed": seed,
                    "train_seed": train_seed,
                    "target_norm": float(torch.linalg.vector_norm(target.float()).item()),
                    "raw_NDS": diag.get("raw_NDS", ""),
                    "guarded_NDS": diag.get("guarded_NDS", ""),
                    "curvature_removed_norm": diag.get("curvature_removed_norm", ""),
                    "hvp_norm": diag.get("hvp_norm", ""),
                    "B2_transfer_gain": true_gains["B2_transfer_gain"],
                    "B3_safety_gain": true_gains["B3_safety_gain"],
                    "random_target_gap": true_gains["B2_transfer_gain"] - random_gains["B2_transfer_gain"],
                    "sign_flip_gap": true_gains["B2_transfer_gain"] - sign_gains["B2_transfer_gain"],
                    "corrupt_gap": true_gains["B2_transfer_gain"] - corrupt_gains["B2_transfer_gain"],
                    "C1_observer6_pass": c1_pass,
                    "official_C2_solver_claimed": 0,
                    "blocker": "" if c1_pass else "observer6_not_above_random_sign_corrupt_or_B3_safety",
                }
            )
    return rows


def _train_run(args: argparse.Namespace, source_row: dict[str, str], run_kind: str) -> dict[str, Any]:
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
    gen = torch.Generator(device=device).manual_seed(262_071 + train_seed + sum(ord(c) for c in run_kind))
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, Any]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    cached_target = torch.zeros(0, device=device)
    cached_diag: dict[str, Any] = {}
    ema_target = torch.zeros(0, device=device)
    target_norm_sum = 0.0
    injection_norm_sum = 0.0
    raw_nds_sum = 0.0
    guarded_nds_sum = 0.0
    removed_sum = 0.0
    refresh_count = 0
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
        elif run_kind == "CTRL-RandomCurvMatched":
            noise = torch.randn(grad.shape, device=device, generator=gen)
            injected = _cap_norm(normalized_like(noise, grad), grad, 0.06)
            _assign_flat_gradient(model, grad + injected)
            opt_adam.step()
            injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        elif run_kind.startswith("OBS6-"):
            if cached_target.numel() == 0 or (step - 1) % max(1, int(args.target_refresh)) == 0:
                cached_target, cached_diag = _curvature_target(args, model, x_train, y_train, run_kind)
                if ema_target.numel() == 0:
                    ema_target = cached_target.detach().clone()
                else:
                    ema_target = float(args.ema_beta) * ema_target + (1.0 - float(args.ema_beta)) * cached_target
                refresh_count += 1
                target_norm_sum += float(torch.linalg.vector_norm(cached_target.float()).item())
                raw_nds_sum += _f(cached_diag.get("raw_NDS"), 0.0)
                guarded_nds_sum += _f(cached_diag.get("guarded_NDS"), 0.0)
                removed_sum += _f(cached_diag.get("curvature_removed_norm"), 0.0)
            target = cached_target
            if "longema" in run_kind or ("preserve800" in run_kind and step >= 800):
                target = ema_target
            target = _orthogonal_component(target, grad)
            cap = _f(cached_diag.get("target_cap_ratio"), 0.06)
            if "preserve800" in run_kind and step >= 800:
                cap = min(cap, 0.04)
            injected = _cap_norm(target, grad, cap)
            _assign_flat_gradient(model, grad + injected)
            opt_adam.step()
            injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        else:
            opt_adam.step()
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    row: dict[str, Any] = {
        "run_kind": run_kind,
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": source_row.get("mechanism", ""),
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "target_refresh": args.target_refresh,
        "target_norm_mean": target_norm_sum / refresh_count if refresh_count else "",
        "injection_norm_sum": injection_norm_sum,
        "raw_NDS_mean": raw_nds_sum / refresh_count if refresh_count else "",
        "guarded_NDS_mean": guarded_nds_sum / refresh_count if refresh_count else "",
        "curvature_removed_norm_mean": removed_sum / refresh_count if refresh_count else "",
        "official_C2_solver_claimed": 0,
        "execution_status": "measured",
    }
    for h in HORIZONS:
        metrics = traces.get(h, {})
        for key, value in metrics.items():
            row[f"{key}_h{h}"] = value
    return row


def _summarize(rows: list[dict[str, Any]], c1_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    controls: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        if str(row.get("run_kind", "")).startswith("OBS6-"):
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
    c1_by_key = {
        (str(r.get("v21_id")), str(r.get("dataset")), str(r.get("train_seed")), str(r.get("run_kind"))): int_flag(r.get("C1_observer6_pass"))
        for r in c1_rows
    }
    summary: list[dict[str, Any]] = []
    for row in enriched:
        if not str(row.get("run_kind", "")).startswith("OBS6-"):
            continue
        c1_pass = c1_by_key.get((str(row.get("v21_id")), str(row.get("dataset")), str(row.get("train_seed")), str(row.get("run_kind"))), 0)
        early_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800]))
        full_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        summary.append(
            {
                "run_kind": row.get("run_kind", ""),
                "v21_id": row.get("v21_id", ""),
                "dataset": row.get("dataset", ""),
                "train_seed": row.get("train_seed", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "target_norm_mean": row.get("target_norm_mean", ""),
                "injection_norm_sum": row.get("injection_norm_sum", ""),
                "raw_NDS_mean": row.get("raw_NDS_mean", ""),
                "guarded_NDS_mean": row.get("guarded_NDS_mean", ""),
                "curvature_removed_norm_mean": row.get("curvature_removed_norm_mean", ""),
                "C1_observer6_pass": c1_pass,
                "C3_observer6_early_pass": early_pass,
                "C3_observer6_full_pass": full_pass,
                "official_C2_solver_claimed": 0,
                "official_C3_pass": 0,
                "blocker": "" if full_pass and c1_pass else "observer6_full_C3_or_C1_gate_failed",
            }
        )
    route = {
        "C1_observer6_rows": len(c1_rows),
        "C1_observer6_pass_rows": sum(int_flag(r.get("C1_observer6_pass")) for r in c1_rows),
        "C3_observer6_rows": len(summary),
        "C3_observer6_early_pass_rows": sum(int_flag(r.get("C3_observer6_early_pass")) for r in summary),
        "C3_observer6_full_pass_rows": sum(int_flag(r.get("C3_observer6_full_pass")) for r in summary),
        "official_C2_solver_claimed_rows": 0,
        "official_C3_pass_rows": 0,
        "row_positive_count_h100": sum(1 for r in summary if _f(r.get("source_vs_best_control_h100"), -999.0) >= 0.005),
        "row_positive_count_h400": sum(1 for r in summary if _f(r.get("source_vs_best_control_h400"), -999.0) >= 0.005),
        "row_positive_count_h800": sum(1 for r in summary if _f(r.get("source_vs_best_control_h800"), -999.0) >= 0.005),
        "row_positive_count_h1600": sum(1 for r in summary if _f(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005),
        "row_positive_count_h3200": sum(1 for r in summary if _f(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005),
        "route": "C3-Observer6CurvatureGuardOpened" if any(int_flag(r.get("C3_observer6_full_pass")) and int_flag(r.get("C1_observer6_pass")) for r in summary) else "C3-Observer6CurvatureGuardBlocked",
        "promotion_allowed": 0,
        "future_source_used_for_direction": 0,
        "blocker": "none" if any(int_flag(r.get("C3_observer6_full_pass")) for r in summary) else "observer6_h3200_or_C1_gate_failed",
    }
    return enriched, summary, route


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_rows = _filtered_matrix(Path(args.source_dir), include_controls=1)
    selected = _selected_rows(args, source_rows, out_dir)
    c1_rows = _c1_rows(args, selected)
    raw_rows: list[dict[str, Any]] = []
    seen_controls: set[tuple[str, str, str]] = set()
    for source_row in selected:
        for kind in CURV_RUNS:
            raw_rows.append(_train_run(args, source_row, kind))
        key = (str(source_row.get("dataset", "")), str(source_row.get("seed", "")), str(source_row.get("train_seed", "")))
        if key not in seen_controls:
            seen_controls.add(key)
            for kind in CONTROL_RUNS:
                raw_rows.append(_train_run(args, source_row, kind))
    enriched, summary, route = _summarize(raw_rows, c1_rows)
    write_rows(out_dir / "v22_07_observer6_curvature_guard_c1.csv", c1_rows)
    write_rows(out_dir / "v22_07_observer6_curvature_guard_raw_runs.csv", raw_rows)
    write_rows(out_dir / "v22_07_observer6_curvature_guard_matrix.csv", enriched)
    write_rows(out_dir / "v22_07_observer6_curvature_guard_summary.csv", summary)
    write_json(out_dir / "v22_07_observer6_curvature_guard_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_observer6_curvature_guard.py --source-dir {args.source_dir} --device {args.device} --out-dir {out_dir}"
        + f" --top-k {args.top_k} --steps {args.steps} --target-refresh {args.target_refresh} --target-scale {args.target_scale} --fu-lr {args.fu_lr} --ema-beta {args.ema_beta}",
        status="completed",
        note=(
            f"C1_pass={route['C1_observer6_pass_rows']} rows={route['C3_observer6_rows']} "
            f"early={route['C3_observer6_early_pass_rows']} full={route['C3_observer6_full_pass_rows']} "
            f"h1600_pos={route['row_positive_count_h1600']} h3200_pos={route['row_positive_count_h3200']} route={route['route']}"
        ),
    )


if __name__ == "__main__":
    main()
