#!/usr/bin/env python3
"""v22.07 Observer7 train-only control-nullspace source target smoke.

Observer6 showed that lowering NDS alone does not preserve source to h1600/
h3200.  Observer7 tests a different principle from the plan's control
equivalence gates: useful FU source may need to live outside the ordinary
AdamW/SGD train-flow subspace.  It builds a train-only activation or CE target,
projects out current CE, short AdamW flow, short SGD flow, and corrupt-label
components, then tests the residual against matched controls.

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

from dgkan.fu.core import UpdateTensor, flat_grad, flat_params, normalized_like  # noqa: E402
from experiments.run_v17_common import carrier_model, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_07_c3_semantic_target_repair import _assign_flat_gradient, _cap_norm, _eval, _remove_positive_component  # noqa: E402
from experiments.run_v22_07_common import PYTHON, V2206_COMBINED_SOURCE, append_exec, ensure_out, finite_float, int_flag, read_rows, write_json, write_rows  # noqa: E402
from experiments.run_v22_07_metric_dynamics_fu import _filtered_matrix, _simulate_gains  # noqa: E402
from experiments.run_v22_07_observer3_flow_theory import _flow_displacement  # noqa: E402
from experiments.run_v22_07_observer4_transfer_meta import _selected_rows  # noqa: E402
from experiments.run_v22_07_observer5_activation_manifold import _manifold_target  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)
NULL_RUNS = (
    "OBS7-neighbor-controlnull-cap006",
    "OBS7-neighbor-controlnull-longema-cap004",
    "OBS7-centroid-controlnull-cap006",
    "OBS7-ce-controlnull-corruptreject-cap006",
)
CONTROL_RUNS = ("CTRL-AdamW", "CTRL-SGD", "CTRL-NoOp", "CTRL-RandomNullMatched")


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
    p.add_argument("--flow-steps", type=int, default=4)
    p.add_argument("--ema-beta", type=float, default=0.90)
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _grad_vec(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, class_balanced: bool = False) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(x).float()
    if class_balanced:
        losses = F.cross_entropy(logits, y, reduction="none")
        parts = []
        for cls in torch.unique(y):
            mask = y == cls
            if bool(mask.any().item()):
                parts.append(losses[mask].mean())
        loss = torch.stack(parts).mean() if parts else losses.mean()
    else:
        loss = F.cross_entropy(logits, y)
    loss.backward()
    return flat_grad(model, x.device).detach().clone()


def _sgd_displacement(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    flow_steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    clone = deepcopy(model)
    clone.train()
    before = flat_params(clone).detach().clone().to(device=x.device)
    opt = torch.optim.SGD(clone.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    n = int(x.shape[0])
    bsz = min(int(batch_size), n)
    for _step in range(int(flow_steps)):
        idx = torch.randint(0, n, (bsz,), generator=gen, device=x.device)
        clone.zero_grad(set_to_none=True)
        F.cross_entropy(clone(x[idx]).float(), y[idx]).backward()
        opt.step()
    after = flat_params(clone).detach().clone().to(device=x.device)
    return after - before


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a.float()) * torch.linalg.vector_norm(b.float())
    if float(denom.item()) <= 1.0e-12:
        return 0.0
    return float((torch.dot(a.float(), b.float()) / denom).item())


def _orthogonal_basis(basis: list[torch.Tensor]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for vec in basis:
        q = vec.detach().clone()
        for prev in out:
            denom = torch.dot(prev.float(), prev.float()).clamp_min(1.0e-12)
            q = q - (torch.dot(q.float(), prev.float()) / denom).to(q.dtype) * prev
        if float(torch.linalg.vector_norm(q.float()).item()) > 1.0e-10:
            out.append(q)
    return out


def _project_out(vec: torch.Tensor, basis: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    orth = _orthogonal_basis([b for b in basis if b.numel() == vec.numel()])
    residual = vec.detach().clone()
    removed_norm = 0.0
    max_control_cos = 0.0
    for q in orth:
        max_control_cos = max(max_control_cos, abs(_cosine(vec, q)))
        denom = torch.dot(q.float(), q.float()).clamp_min(1.0e-12)
        removed = (torch.dot(residual.float(), q.float()) / denom).to(vec.dtype) * q
        residual = residual - removed
        removed_norm += float(torch.linalg.vector_norm(removed.float()).item())
    base_norm = float(torch.linalg.vector_norm(vec.float()).item())
    resid_norm = float(torch.linalg.vector_norm(residual.float()).item())
    return residual, {
        "control_basis_rank": float(len(orth)),
        "control_projection_removed_norm": removed_norm,
        "control_null_residual_norm": resid_norm,
        "control_null_residual_fraction": resid_norm / max(base_norm, 1.0e-12),
        "max_control_cosine": max_control_cos,
    }


def _base_target(args: argparse.Namespace, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, run_kind: str) -> tuple[torch.Tensor, dict[str, Any]]:
    if "ce-controlnull" in run_kind:
        grad = _grad_vec(model, x, y, class_balanced=True)
        diag = {"base_target_family": "class_balanced_CE_gradient"}
        return grad, diag
    if "centroid" in run_kind:
        target, diag = _manifold_target(args, model, x, y, "OBS5-centroid-info-consensus-cap010")
        diag["base_target_family"] = "activation_centroid_info"
        return target, diag
    target, diag = _manifold_target(args, model, x, y, "OBS5-neighbor-preserve-cap010")
    diag["base_target_family"] = "activation_neighbor_preserve"
    return target, diag


def _control_null_target(
    args: argparse.Namespace,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    run_kind: str,
    train_seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base, diag = _base_target(args, model, x, y, run_kind)
    ce = _grad_vec(model, x, y)
    seed_base = int(train_seed) + 72_071 + sum(ord(c) for c in run_kind)
    adam = -_flow_displacement(
        model,
        x,
        y,
        flow_steps=int(args.flow_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        seed=seed_base,
    )
    sgd = -_sgd_displacement(
        model,
        x,
        y,
        flow_steps=int(args.flow_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        seed=seed_base + 1009,
    )
    corrupt_grad = _grad_vec(model, x, (y + 1) % max(2, int(y.max().item()) + 1))
    residual, proj = _project_out(base, [ce, adam, sgd, corrupt_grad])
    if "corruptreject" in run_kind:
        residual, removed = _remove_positive_component(residual, corrupt_grad)
    else:
        removed = 0.0
    if float(torch.linalg.vector_norm(residual.float()).item()) <= 1.0e-10:
        residual = base - ce * 0.0
    residual = normalized_like(residual, ce) * float(args.target_scale)
    cap_ratio = 0.04 if "cap004" in run_kind else 0.06
    diag = {
        **diag,
        **proj,
        "observer_family": run_kind,
        "target_role": "train_only_control_nullspace_residual",
        "flow_steps": int(args.flow_steps),
        "ce_cosine": _cosine(base, ce),
        "adam_flow_cosine": _cosine(base, adam),
        "sgd_flow_cosine": _cosine(base, sgd),
        "corrupt_gradient_cosine": _cosine(base, corrupt_grad),
        "corrupt_removed_norm": removed,
        "target_cap_ratio": cap_ratio,
        "uses_future_source_for_direction": 0,
        "official_C2_solver_claimed": 0,
    }
    return residual.detach().clone(), diag


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
        for run_kind in NULL_RUNS:
            target, diag = _control_null_target(args, model, xb, yb, run_kind, train_seed)
            update = UpdateTensor(target, run_kind, "subtract", "parameter", "v22_07_observer7_control_nullspace", str(source_row.get("mechanism", "")), one_step_descent_claim=1, diagnostics=diag)
            gen = torch.Generator(device=device).manual_seed(train_seed + sum(ord(c) for c in run_kind) + 27_071)
            noise = torch.randn(target.shape, device=device, generator=gen)
            random_update = UpdateTensor(normalized_like(noise, target), "matched_random_null", "subtract", "parameter", "v22_07_observer7_random", "CTRL-RandomNullMatched", one_step_descent_claim=0)
            sign_update = UpdateTensor(target, "sign_flip_null", "add", "parameter", "v22_07_observer7_sign_flip", "CTRL-SignFlipNull", one_step_descent_claim=0)
            corrupt_target, _cdiag = _control_null_target(args, model, xb, (yb + 1) % max(2, int(args.classes)), run_kind, train_seed)
            corrupt_update = UpdateTensor(corrupt_target, "corrupt_null", "subtract", "parameter", "v22_07_observer7_corrupt", "CTRL-CorruptNull", one_step_descent_claim=0)
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
                    "control_basis_rank": diag.get("control_basis_rank", ""),
                    "control_projection_removed_norm": diag.get("control_projection_removed_norm", ""),
                    "control_null_residual_fraction": diag.get("control_null_residual_fraction", ""),
                    "max_control_cosine": diag.get("max_control_cosine", ""),
                    "ce_cosine": diag.get("ce_cosine", ""),
                    "adam_flow_cosine": diag.get("adam_flow_cosine", ""),
                    "sgd_flow_cosine": diag.get("sgd_flow_cosine", ""),
                    "B2_transfer_gain": true_gains["B2_transfer_gain"],
                    "B3_safety_gain": true_gains["B3_safety_gain"],
                    "random_target_gap": true_gains["B2_transfer_gain"] - random_gains["B2_transfer_gain"],
                    "sign_flip_gap": true_gains["B2_transfer_gain"] - sign_gains["B2_transfer_gain"],
                    "corrupt_gap": true_gains["B2_transfer_gain"] - corrupt_gains["B2_transfer_gain"],
                    "C1_observer7_pass": c1_pass,
                    "official_C2_solver_claimed": 0,
                    "blocker": "" if c1_pass else "observer7_not_above_random_sign_corrupt_or_B3_safety",
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
    gen = torch.Generator(device=device).manual_seed(272_071 + train_seed + sum(ord(c) for c in run_kind))
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, Any]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    cached_target = torch.zeros(0, device=device)
    cached_diag: dict[str, Any] = {}
    ema_target = torch.zeros(0, device=device)
    target_norm_sum = 0.0
    injection_norm_sum = 0.0
    residual_fraction_sum = 0.0
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
        elif run_kind == "CTRL-RandomNullMatched":
            noise = torch.randn(grad.shape, device=device, generator=gen)
            injected = _cap_norm(normalized_like(noise, grad), grad, 0.06)
            _assign_flat_gradient(model, grad + injected)
            opt_adam.step()
            injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        elif run_kind.startswith("OBS7-"):
            if cached_target.numel() == 0 or (step - 1) % max(1, int(args.target_refresh)) == 0:
                cached_target, cached_diag = _control_null_target(args, model, x_train, y_train, run_kind, train_seed)
                if ema_target.numel() == 0:
                    ema_target = cached_target.detach().clone()
                else:
                    ema_target = float(args.ema_beta) * ema_target + (1.0 - float(args.ema_beta)) * cached_target
                refresh_count += 1
                target_norm_sum += float(torch.linalg.vector_norm(cached_target.float()).item())
                residual_fraction_sum += _f(cached_diag.get("control_null_residual_fraction"), 0.0)
                removed_sum += _f(cached_diag.get("control_projection_removed_norm"), 0.0)
            target = ema_target if "longema" in run_kind else cached_target
            injected = _cap_norm(target, grad, _f(cached_diag.get("target_cap_ratio"), 0.06))
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
        "control_null_residual_fraction_mean": residual_fraction_sum / refresh_count if refresh_count else "",
        "control_projection_removed_norm_mean": removed_sum / refresh_count if refresh_count else "",
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
        if str(row.get("run_kind", "")).startswith("OBS7-"):
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
        (str(r.get("v21_id")), str(r.get("dataset")), str(r.get("train_seed")), str(r.get("run_kind"))): int_flag(r.get("C1_observer7_pass"))
        for r in c1_rows
    }
    summary: list[dict[str, Any]] = []
    for row in enriched:
        if not str(row.get("run_kind", "")).startswith("OBS7-"):
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
                "control_null_residual_fraction_mean": row.get("control_null_residual_fraction_mean", ""),
                "control_projection_removed_norm_mean": row.get("control_projection_removed_norm_mean", ""),
                "C1_observer7_pass": c1_pass,
                "C3_observer7_early_pass": early_pass,
                "C3_observer7_full_pass": full_pass,
                "official_C2_solver_claimed": 0,
                "official_C3_pass": 0,
                "blocker": "" if full_pass and c1_pass else "observer7_full_C3_or_C1_gate_failed",
            }
        )
    route = {
        "C1_observer7_rows": len(c1_rows),
        "C1_observer7_pass_rows": sum(int_flag(r.get("C1_observer7_pass")) for r in c1_rows),
        "C3_observer7_rows": len(summary),
        "C3_observer7_early_pass_rows": sum(int_flag(r.get("C3_observer7_early_pass")) for r in summary),
        "C3_observer7_full_pass_rows": sum(int_flag(r.get("C3_observer7_full_pass")) for r in summary),
        "official_C2_solver_claimed_rows": 0,
        "official_C3_pass_rows": 0,
        "row_positive_count_h100": sum(1 for r in summary if _f(r.get("source_vs_best_control_h100"), -999.0) >= 0.005),
        "row_positive_count_h400": sum(1 for r in summary if _f(r.get("source_vs_best_control_h400"), -999.0) >= 0.005),
        "row_positive_count_h800": sum(1 for r in summary if _f(r.get("source_vs_best_control_h800"), -999.0) >= 0.005),
        "row_positive_count_h1600": sum(1 for r in summary if _f(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005),
        "row_positive_count_h3200": sum(1 for r in summary if _f(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005),
        "route": "C3-Observer7ControlNullspaceOpened" if any(int_flag(r.get("C3_observer7_full_pass")) and int_flag(r.get("C1_observer7_pass")) for r in summary) else "C3-Observer7ControlNullspaceBlocked",
        "promotion_allowed": 0,
        "future_source_used_for_direction": 0,
        "blocker": "none" if any(int_flag(r.get("C3_observer7_full_pass")) for r in summary) else "observer7_h3200_or_C1_gate_failed",
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
        for kind in NULL_RUNS:
            raw_rows.append(_train_run(args, source_row, kind))
        key = (str(source_row.get("dataset", "")), str(source_row.get("seed", "")), str(source_row.get("train_seed", "")))
        if key not in seen_controls:
            seen_controls.add(key)
            for kind in CONTROL_RUNS:
                raw_rows.append(_train_run(args, source_row, kind))
    enriched, summary, route = _summarize(raw_rows, c1_rows)
    write_rows(out_dir / "v22_07_observer7_control_nullspace_c1.csv", c1_rows)
    write_rows(out_dir / "v22_07_observer7_control_nullspace_raw_runs.csv", raw_rows)
    write_rows(out_dir / "v22_07_observer7_control_nullspace_matrix.csv", enriched)
    write_rows(out_dir / "v22_07_observer7_control_nullspace_summary.csv", summary)
    write_json(out_dir / "v22_07_observer7_control_nullspace_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_observer7_control_nullspace.py --source-dir {args.source_dir} --device {args.device} --out-dir {out_dir}"
        + f" --top-k {args.top_k} --steps {args.steps} --target-refresh {args.target_refresh} --target-scale {args.target_scale} --fu-lr {args.fu_lr} --flow-steps {args.flow_steps} --ema-beta {args.ema_beta}",
        status="completed",
        note=(
            f"C1_pass={route['C1_observer7_pass_rows']} rows={route['C3_observer7_rows']} "
            f"early={route['C3_observer7_early_pass_rows']} full={route['C3_observer7_full_pass_rows']} "
            f"h1600_pos={route['row_positive_count_h1600']} h3200_pos={route['row_positive_count_h3200']} route={route['route']}"
        ),
    )


if __name__ == "__main__":
    main()
