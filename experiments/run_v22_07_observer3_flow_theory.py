#!/usr/bin/env python3
"""v22.07 Observer3 optimizer-flow source-observability smoke.

Observer2 showed that split-SNR and margin-band targets were too weak and did
not pass C1/full C3.  Observer3 tests a different train-only theory:

  short optimizer-flow forecast:
    clone the current model, unroll a few train-only AdamW steps, and convert
    the resulting parameter displacement into a gradient-like target.

  corrupt-flow rejection:
    subtract components shared with a corrupt-label forecast.

  split-flow consensus:
    build forecasts on train splits and keep stable agreement components.

No validation/test/future source is used to build directions.  This remains a
smoke path and does not claim official C2 because it is not a metric-defined
solver.
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

from dgkan.fu.core import UpdateTensor, flat_grad, flat_params, normalized_like, trainable_parameters  # noqa: E402
from experiments.run_v17_common import carrier_model, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_07_c3_semantic_target_repair import _assign_flat_gradient, _cap_norm, _eval, _orthogonal_component, _remove_positive_component  # noqa: E402
from experiments.run_v22_07_common import PYTHON, V2206_COMBINED_SOURCE, append_exec, ensure_out, finite_float, int_flag, read_rows, write_json, write_rows  # noqa: E402
from experiments.run_v22_07_metric_dynamics_fu import _filtered_matrix, _simulate_gains  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)
FLOW_RUNS = (
    "OBS3-flow8-direct-cap010",
    "OBS3-flow8-corruptreject-cap010",
    "OBS3-flow8-splitconsensus-cap010",
    "OBS3-flow16-corruptreject-cap010",
    "OBS3-flow16-rowbalanced-cap012",
)
CONTROL_RUNS = ("CTRL-AdamW", "CTRL-SGD", "CTRL-NoOp", "CTRL-RandomFlowMatched")


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
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _chunks(x: torch.Tensor, y: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    n = int(x.shape[0])
    a = max(2, n // 3)
    b = max(a + 2, 2 * n // 3)
    return [(x[:a], y[:a]), (x[a:b], y[a:b]), (x[b:], y[b:])]


def _flow_steps(run_kind: str) -> int:
    return 16 if "flow16" in run_kind else 8


def _grad_vec(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x).float(), y).backward()
    return flat_grad(model, x.device).detach().clone()


def _flow_displacement(
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
    opt = torch.optim.AdamW(clone.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    n = int(x.shape[0])
    bsz = min(int(batch_size), n)
    for _step in range(int(flow_steps)):
        idx = torch.randint(0, n, (bsz,), generator=gen, device=x.device)
        xb = x[idx]
        yb = y[idx]
        clone.zero_grad(set_to_none=True)
        F.cross_entropy(clone(xb).float(), yb).backward()
        opt.step()
    after = flat_params(clone).detach().clone().to(device=x.device)
    return after - before


def _row_balance(model: torch.nn.Module, vec: torch.Tensor) -> tuple[torch.Tensor, float]:
    chunks = []
    offset = 0
    row_norms = []
    for param in trainable_parameters(model):
        n = int(param.numel())
        part = vec[offset : offset + n].view_as(param).detach().clone()
        offset += n
        if param.ndim >= 2:
            flat = part.float().view(int(param.shape[0]), -1)
            norms = torch.linalg.vector_norm(flat, dim=1).clamp_min(1.0e-12)
            nz = norms[norms > 1.0e-10]
            target = torch.median(nz.detach()) if nz.numel() else torch.median(norms.detach())
            scale = (target / norms).to(dtype=part.dtype).view(-1, *([1] * (part.ndim - 1)))
            part = part * torch.clamp(scale, 0.25, 4.0)
            row_norms.extend(float(v.item()) for v in nz.detach().cpu())
        elif param.ndim == 1:
            part = 0.5 * part
        chunks.append(part.reshape(-1))
    if not chunks:
        return vec, 0.0
    imbalance = max(row_norms) / max(min(row_norms), 1.0e-12) if row_norms else 0.0
    return torch.cat(chunks).to(device=vec.device, dtype=vec.dtype), float(imbalance)


def _split_consensus(vecs: list[torch.Tensor]) -> tuple[torch.Tensor, float]:
    if not vecs:
        return torch.zeros(0), 0.0
    stacked = torch.stack([v.detach() for v in vecs], dim=0)
    signs = torch.sign(stacked.float())
    agree = (torch.abs(signs.sum(dim=0)) == float(len(vecs))) & (torch.abs(signs).sum(dim=0) == float(len(vecs)))
    avg = stacked.mean(dim=0)
    masked = torch.where(agree.to(device=avg.device), avg, torch.zeros_like(avg))
    density = float(agree.float().mean().item()) if agree.numel() else 0.0
    return masked.to(dtype=vecs[0].dtype), density


def _flow_target(
    args: argparse.Namespace,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    run_kind: str,
    train_seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    flow_steps = _flow_steps(run_kind)
    g_all = _grad_vec(model, x, y)
    seed_base = int(train_seed) + 31_071 + sum(ord(c) for c in run_kind)
    if "splitconsensus" in run_kind:
        parts = []
        for idx, (xi, yi) in enumerate(_chunks(x, y)):
            if int(xi.shape[0]) <= 0:
                continue
            disp = _flow_displacement(
                model,
                xi,
                yi,
                flow_steps=flow_steps,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                batch_size=min(int(args.batch_size), int(xi.shape[0])),
                seed=seed_base + idx * 1009,
            )
            parts.append(-disp)
        target, density = _split_consensus(parts)
    else:
        disp = _flow_displacement(
            model,
            x,
            y,
            flow_steps=flow_steps,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.batch_size),
            seed=seed_base,
        )
        target = -disp
        density = 1.0
    corrupt_removed = 0.0
    if "corruptreject" in run_kind:
        corrupt_y = (y + 1) % max(2, int(y.max().item()) + 1)
        corrupt_disp = _flow_displacement(
            model,
            x,
            corrupt_y,
            flow_steps=flow_steps,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.batch_size),
            seed=seed_base + 77_777,
        )
        target, corrupt_removed = _remove_positive_component(target, -corrupt_disp)
    row_imbalance = 0.0
    if "rowbalanced" in run_kind:
        target, row_imbalance = _row_balance(model, target)
    target = normalized_like(target, g_all) * float(args.target_scale)
    diag = {
        "observer_family": run_kind,
        "flow_steps": flow_steps,
        "flow_consensus_density": density,
        "corrupt_removed_norm": corrupt_removed,
        "row_balance_applied": int("rowbalanced" in run_kind),
        "row_norm_imbalance_before": row_imbalance,
        "target_cap_ratio": 0.12 if "cap012" in run_kind else 0.10,
        "uses_future_source_for_direction": 0,
        "official_C2_solver_claimed": 0,
    }
    return target.detach().clone(), diag


def _selected_rows(args: argparse.Namespace, source_rows: list[dict[str, str]], out_dir: Path) -> list[dict[str, str]]:
    selection = Path(args.selection_file) if str(args.selection_file).strip() else out_dir / "v22_07_source_theory_top_candidates.csv"
    top = read_rows(selection)
    by_order = {idx: row for idx, row in enumerate(source_rows)}
    selected = []
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
        for run_kind in FLOW_RUNS:
            target, diag = _flow_target(args, model, xb, yb, run_kind, train_seed)
            update = UpdateTensor(target, run_kind, "subtract", "parameter", "v22_07_observer3_flow_theory", str(source_row.get("mechanism", "")), one_step_descent_claim=1, diagnostics=diag)
            gen = torch.Generator(device=device).manual_seed(train_seed + sum(ord(c) for c in run_kind) + 23_071)
            noise = torch.randn(target.shape, device=device, generator=gen)
            random_update = UpdateTensor(normalized_like(noise, target), "matched_random_flow", "subtract", "parameter", "v22_07_observer3_random", "CTRL-RandomFlowMatched", one_step_descent_claim=0)
            sign_update = UpdateTensor(target, "sign_flip_flow", "add", "parameter", "v22_07_observer3_sign_flip", "CTRL-SignFlipFlow", one_step_descent_claim=0)
            corrupt_target, _cdiag = _flow_target(args, model, xb, (yb + 1) % max(2, int(args.classes)), run_kind, train_seed + 37)
            corrupt_update = UpdateTensor(corrupt_target, "corrupt_flow", "subtract", "parameter", "v22_07_observer3_corrupt", "CTRL-CorruptFlow", one_step_descent_claim=0)
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
                    "flow_steps": diag.get("flow_steps", ""),
                    "flow_consensus_density": diag.get("flow_consensus_density", ""),
                    "corrupt_removed_norm": diag.get("corrupt_removed_norm", ""),
                    "row_balance_applied": diag.get("row_balance_applied", ""),
                    "row_norm_imbalance_before": diag.get("row_norm_imbalance_before", ""),
                    "B2_transfer_gain": true_gains["B2_transfer_gain"],
                    "B3_safety_gain": true_gains["B3_safety_gain"],
                    "random_target_gap": true_gains["B2_transfer_gain"] - random_gains["B2_transfer_gain"],
                    "sign_flip_gap": true_gains["B2_transfer_gain"] - sign_gains["B2_transfer_gain"],
                    "corrupt_gap": true_gains["B2_transfer_gain"] - corrupt_gains["B2_transfer_gain"],
                    "C1_observer3_pass": c1_pass,
                    "official_C2_solver_claimed": 0,
                    "blocker": "" if c1_pass else "observer3_not_above_random_sign_corrupt_or_B3_safety",
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
    gen = torch.Generator(device=device).manual_seed(231_071 + train_seed + sum(ord(c) for c in run_kind))
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, Any]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    cached_target = torch.zeros(0, device=device)
    cached_diag: dict[str, Any] = {}
    target_norm_sum = 0.0
    injection_norm_sum = 0.0
    consensus_density_sum = 0.0
    corrupt_removed_norm_sum = 0.0
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
        elif run_kind == "CTRL-RandomFlowMatched":
            noise = torch.randn(grad.shape, device=device, generator=gen)
            injected = _cap_norm(normalized_like(noise, grad), grad, 0.12)
            _assign_flat_gradient(model, grad + injected)
            opt_adam.step()
            injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        elif run_kind.startswith("OBS3-"):
            if cached_target.numel() == 0 or (step - 1) % max(1, int(args.target_refresh)) == 0:
                cached_target, cached_diag = _flow_target(args, model, x_train, y_train, run_kind, train_seed + step)
                refresh_count += 1
                target_norm_sum += float(torch.linalg.vector_norm(cached_target.float()).item())
                consensus_density_sum += _f(cached_diag.get("flow_consensus_density"), 0.0)
                corrupt_removed_norm_sum += _f(cached_diag.get("corrupt_removed_norm"), 0.0)
            target = _orthogonal_component(cached_target, grad)
            injected = _cap_norm(target, grad, _f(cached_diag.get("target_cap_ratio"), 0.10))
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
        "flow_consensus_density_mean": consensus_density_sum / refresh_count if refresh_count else "",
        "corrupt_removed_norm_mean": corrupt_removed_norm_sum / refresh_count if refresh_count else "",
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
        if str(row.get("run_kind", "")).startswith("OBS3-"):
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
        (str(r.get("v21_id")), str(r.get("dataset")), str(r.get("train_seed")), str(r.get("run_kind"))): int_flag(r.get("C1_observer3_pass"))
        for r in c1_rows
    }
    summary: list[dict[str, Any]] = []
    for row in enriched:
        if not str(row.get("run_kind", "")).startswith("OBS3-"):
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
                "flow_consensus_density_mean": row.get("flow_consensus_density_mean", ""),
                "corrupt_removed_norm_mean": row.get("corrupt_removed_norm_mean", ""),
                "C1_observer3_pass": c1_pass,
                "C3_observer3_early_pass": early_pass,
                "C3_observer3_full_pass": full_pass,
                "official_C2_solver_claimed": 0,
                "official_C3_pass": 0,
                "blocker": "" if full_pass and c1_pass else "observer3_full_C3_or_C1_gate_failed",
            }
        )
    route = {
        "C1_observer3_rows": len(c1_rows),
        "C1_observer3_pass_rows": sum(int_flag(r.get("C1_observer3_pass")) for r in c1_rows),
        "C3_observer3_rows": len(summary),
        "C3_observer3_early_pass_rows": sum(int_flag(r.get("C3_observer3_early_pass")) for r in summary),
        "C3_observer3_full_pass_rows": sum(int_flag(r.get("C3_observer3_full_pass")) for r in summary),
        "official_C2_solver_claimed_rows": 0,
        "official_C3_pass_rows": 0,
        "row_positive_count_h100": sum(1 for r in summary if _f(r.get("source_vs_best_control_h100"), -999.0) >= 0.005),
        "row_positive_count_h400": sum(1 for r in summary if _f(r.get("source_vs_best_control_h400"), -999.0) >= 0.005),
        "row_positive_count_h800": sum(1 for r in summary if _f(r.get("source_vs_best_control_h800"), -999.0) >= 0.005),
        "row_positive_count_h1600": sum(1 for r in summary if _f(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005),
        "row_positive_count_h3200": sum(1 for r in summary if _f(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005),
        "route": "C3-Observer3FlowTheoryOpened" if any(int_flag(r.get("C3_observer3_full_pass")) and int_flag(r.get("C1_observer3_pass")) for r in summary) else "C3-Observer3FlowTheoryBlocked",
        "promotion_allowed": 0,
        "future_source_used_for_direction": 0,
        "blocker": "none" if any(int_flag(r.get("C3_observer3_full_pass")) for r in summary) else "observer3_h3200_or_C1_gate_failed",
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
        for kind in FLOW_RUNS:
            raw_rows.append(_train_run(args, source_row, kind))
        key = (str(source_row.get("dataset", "")), str(source_row.get("seed", "")), str(source_row.get("train_seed", "")))
        if key not in seen_controls:
            seen_controls.add(key)
            for kind in CONTROL_RUNS:
                raw_rows.append(_train_run(args, source_row, kind))
    enriched, summary, route = _summarize(raw_rows, c1_rows)
    write_rows(out_dir / "v22_07_observer3_flow_theory_c1.csv", c1_rows)
    write_rows(out_dir / "v22_07_observer3_flow_theory_raw_runs.csv", raw_rows)
    write_rows(out_dir / "v22_07_observer3_flow_theory_matrix.csv", enriched)
    write_rows(out_dir / "v22_07_observer3_flow_theory_summary.csv", summary)
    write_json(out_dir / "v22_07_observer3_flow_theory_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_observer3_flow_theory.py --source-dir {args.source_dir} --device {args.device} --out-dir {out_dir}"
        + f" --top-k {args.top_k} --steps {args.steps} --target-refresh {args.target_refresh} --target-scale {args.target_scale} --fu-lr {args.fu_lr}",
        status="completed",
        note=(
            f"C1_pass={route['C1_observer3_pass_rows']} rows={route['C3_observer3_rows']} "
            f"early={route['C3_observer3_early_pass_rows']} full={route['C3_observer3_full_pass_rows']} "
            f"h1600_pos={route['row_positive_count_h1600']} h3200_pos={route['row_positive_count_h3200']} route={route['route']}"
        ),
    )


if __name__ == "__main__":
    main()
