#!/usr/bin/env python3
"""v22.07 Observer5 train-only activation-manifold smoke.

Observer4 showed a transient h100/h400/h800 chain but still failed C1 and
h1600/h3200 retention.  Observer5 switches observability principle instead of
adding another metric-solver name: it builds train-only targets from hidden
activation geometry, signal/reservoir separation, neighbor consistency, and
information-volume floor diagnostics.

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
from experiments.run_v22_07_observer4_transfer_meta import _agreement, _row_balance, _selected_rows  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)
MANIFOLD_RUNS = (
    "OBS5-centroid-margin-cap010",
    "OBS5-info-volume-floor-cap010",
    "OBS5-neighbor-preserve-cap010",
    "OBS5-centroid-info-consensus-cap010",
    "OBS5-unfold-corruptreject-rowbalanced-cap012",
    "OBS5-neighbor-hidden-channel-cap008",
    "OBS5-neighbor-readout-channel-cap012",
    "OBS5-centroid-info-hidden-channel-cap010",
)
CONTROL_RUNS = ("CTRL-AdamW", "CTRL-SGD", "CTRL-NoOp", "CTRL-RandomManifoldMatched")


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
    p.add_argument("--target-refresh", type=int, default=50)
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


def _hidden(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "frozen_readout_features"):
        return model.frozen_readout_features(x).float()
    return model(x).float()


def _ce_grad(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x).float(), y).backward()
    return flat_grad(model, x.device).detach().clone()


def _normed_hidden(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    h = _hidden(model, x)
    h = h - h.mean(dim=0, keepdim=True)
    scale = torch.sqrt(h.square().mean()).clamp_min(1.0e-6)
    return h / scale


def _centroid_terms(h: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = h.sum() * 0.0
    within_parts = []
    centroids = []
    for cls in torch.unique(y):
        mask = y == cls
        if bool(mask.any().item()):
            hc = h[mask]
            center = hc.mean(dim=0)
            centroids.append(center)
            within_parts.append((hc - center).square().sum(dim=1).mean())
    within = torch.stack(within_parts).mean() if within_parts else zeros
    if len(centroids) < 2:
        return within, zeros
    c = torch.stack(centroids, dim=0)
    dist = torch.cdist(c, c, p=2).square()
    mask = ~torch.eye(int(c.shape[0]), dtype=torch.bool, device=h.device)
    between = dist[mask].mean() if bool(mask.any().item()) else zeros
    return within, between


def _logdet_floor(h: torch.Tensor) -> torch.Tensor:
    n = int(h.shape[0])
    if n < 3:
        return h.sum() * 0.0
    cov = h.T @ h / float(max(1, n - 1))
    eye = torch.eye(int(cov.shape[0]), device=h.device, dtype=cov.dtype)
    evals = torch.linalg.eigvalsh(cov + 1.0e-4 * eye).clamp_min(1.0e-8)
    return torch.log(evals).mean()


def _neighbor_loss(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = int(h.shape[0])
    if n < 3:
        return h.sum() * 0.0
    dist = torch.cdist(h, h, p=2)
    eye = torch.eye(n, dtype=torch.bool, device=h.device)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = (y[:, None] != y[None, :]) & ~eye
    zeros = h.sum() * 0.0
    same_loss = dist[same].square().mean() if bool(same.any().item()) else zeros
    diff_loss = F.relu(1.25 - dist[diff]).square().mean() if bool(diff.any().item()) else zeros
    return same_loss + diff_loss


def _manifold_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, run_kind: str) -> torch.Tensor:
    h = _normed_hidden(model, x)
    within, between = _centroid_terms(h, y)
    log_volume = _logdet_floor(h)
    neigh = _neighbor_loss(h, y)
    if "info-volume-floor" in run_kind:
        return -log_volume + 0.08 * within
    if "neighbor-preserve" in run_kind:
        return neigh + 0.05 * within - 0.02 * between
    if "centroid-info" in run_kind:
        return within - 0.18 * between - 0.04 * log_volume
    if "unfold" in run_kind:
        return 0.45 * neigh + 0.35 * within - 0.10 * between - 0.08 * log_volume
    return within - 0.20 * between


def _manifold_grad(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, run_kind: str) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    _manifold_loss(model, x, y, run_kind).backward()
    return flat_grad(model, x.device).detach().clone()


def _alignment(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a.float()) * torch.linalg.vector_norm(b.float())
    if float(denom.item()) <= 1.0e-12:
        return 0.0
    return float((torch.dot(a.float(), b.float()) / denom).item())


def _role_mask(model: torch.nn.Module, role: str, device: torch.device) -> torch.Tensor:
    params = trainable_parameters(model)
    chunks = []
    last = max(0, len(params) - 1)
    for idx, param in enumerate(params):
        keep = 1.0
        if role == "hidden":
            keep = 1.0 if idx < last and param.ndim >= 2 else 0.0
        elif role == "readout":
            keep = 1.0 if idx == last else 0.0
        chunks.append(torch.full_like(param.detach(), keep).reshape(-1))
    return torch.cat(chunks).to(device=device) if chunks else torch.zeros(0, device=device)


def _manifold_stats(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        h = _normed_hidden(model, x)
        n = int(h.shape[0])
        within, between = _centroid_terms(h, y)
        cov = h.T @ h / float(max(1, n - 1))
        eye = torch.eye(int(cov.shape[0]), device=h.device, dtype=cov.dtype)
        evals = torch.linalg.eigvalsh(cov + 1.0e-4 * eye).clamp_min(1.0e-8)
        total = float(evals.sum().item())
        intrinsic = float((evals.sum().square() / evals.square().sum().clamp_min(1.0e-12)).item())
        condition = float((evals.max() / evals.min().clamp_min(1.0e-12)).item())
        info_volume = float(torch.log(evals).mean().item())
        dist_h = torch.cdist(h, h, p=2)
        dist_x = torch.cdist(x.float(), x.float(), p=2)
        if n > 1:
            inf = torch.full((n,), float("inf"), device=h.device)
            dist_h = dist_h + torch.diag(inf)
            dist_x = dist_x + torch.diag(inf)
            nearest_h = dist_h.argmin(dim=1)
            same_neighbor = float((y[nearest_h] == y).float().mean().item())
            nearest_x = dist_x.argmin(dim=1)
            stretch = dist_h[torch.arange(n, device=h.device), nearest_x] / dist_x[torch.arange(n, device=h.device), nearest_x].clamp_min(1.0e-6)
            local_stretch_p95 = float(torch.quantile(stretch.detach().float(), 0.95).item())
            fold_proxy = float((stretch > torch.quantile(stretch.detach().float(), 0.90)).float().mean().item())
        else:
            same_neighbor = 0.0
            local_stretch_p95 = 0.0
            fold_proxy = 0.0
        energy = float((within + between).abs().item())
        signal = float((between / (within + between).abs().clamp_min(1.0e-12)).item()) if energy > 0 else 0.0
        reservoir = float((within / (within + between).abs().clamp_min(1.0e-12)).item()) if energy > 0 else 0.0
    return {
        "intrinsic_dimension": intrinsic,
        "info_volume": info_volume,
        "condition_number_of_source_subspace": condition,
        "neighbor_preservation": same_neighbor,
        "fold_proxy": fold_proxy,
        "local_stretch_p95": local_stretch_p95,
        "signal_channel_projection": signal,
        "reservoir_projection": reservoir,
        "within_class_energy": float(within.item()),
        "between_class_energy": float(between.item()),
        "eigenvalue_sum": total,
    }


def _manifold_target(
    args: argparse.Namespace,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    run_kind: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if "consensus" in run_kind:
        grads = [_manifold_grad(model, xb, yb, run_kind) for xb, yb in _chunks(x, y) if int(xb.shape[0])]
        target, density = _agreement(grads)
    else:
        target = _manifold_grad(model, x, y, run_kind)
        density = 1.0
    corrupt_removed = 0.0
    corrupt = _manifold_grad(model, x, (y + 1) % max(2, int(y.max().item()) + 1), run_kind)
    if "corruptreject" in run_kind:
        target, corrupt_removed = _remove_positive_component(target, corrupt)
    channel_role = "all"
    if "hidden-channel" in run_kind:
        channel_role = "hidden"
        target = target * _role_mask(model, "hidden", x.device)
    elif "readout-channel" in run_kind:
        channel_role = "readout"
        target = target * _role_mask(model, "readout", x.device)
    row_imbalance = 0.0
    if "rowbalanced" in run_kind:
        target, row_imbalance = _row_balance(model, target)
    ce = _ce_grad(model, x, y)
    stats = _manifold_stats(model, x, y)
    raw_norm = float(torch.linalg.vector_norm(target.float()).item())
    target = normalized_like(target, ce) * float(args.target_scale)
    cap_ratio = 0.12 if "cap012" in run_kind else (0.08 if "cap008" in run_kind else 0.10)
    diag = {
        "observer_family": run_kind,
        "target_role": "activation_manifold_geometry",
        "channel_role": channel_role,
        "manifold_consensus_density": density,
        "corrupt_removed_norm": corrupt_removed,
        "row_balance_applied": int("rowbalanced" in run_kind),
        "row_norm_imbalance_before": row_imbalance,
        "target_cap_ratio": cap_ratio,
        "raw_manifold_grad_norm": raw_norm,
        "ce_alignment": _alignment(target, ce),
        "corrupt_alignment": _alignment(target, corrupt),
        "uses_future_source_for_direction": 0,
        "official_C2_solver_claimed": 0,
        **stats,
    }
    return target.detach().clone(), diag


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
        for run_kind in MANIFOLD_RUNS:
            target, diag = _manifold_target(args, model, xb, yb, run_kind)
            update = UpdateTensor(target, run_kind, "subtract", "parameter", "v22_07_observer5_activation_manifold", str(source_row.get("mechanism", "")), one_step_descent_claim=1, diagnostics=diag)
            gen = torch.Generator(device=device).manual_seed(train_seed + sum(ord(c) for c in run_kind) + 25_071)
            noise = torch.randn(target.shape, device=device, generator=gen)
            random_update = UpdateTensor(normalized_like(noise, target), "matched_random_manifold", "subtract", "parameter", "v22_07_observer5_random", "CTRL-RandomManifoldMatched", one_step_descent_claim=0)
            sign_update = UpdateTensor(target, "sign_flip_manifold", "add", "parameter", "v22_07_observer5_sign_flip", "CTRL-SignFlipManifold", one_step_descent_claim=0)
            corrupt_target, _cdiag = _manifold_target(args, model, xb, (yb + 1) % max(2, int(args.classes)), run_kind)
            corrupt_update = UpdateTensor(corrupt_target, "corrupt_manifold", "subtract", "parameter", "v22_07_observer5_corrupt", "CTRL-CorruptManifold", one_step_descent_claim=0)
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
                    "manifold_consensus_density": diag.get("manifold_consensus_density", ""),
                    "intrinsic_dimension": diag.get("intrinsic_dimension", ""),
                    "info_volume": diag.get("info_volume", ""),
                    "condition_number_of_source_subspace": diag.get("condition_number_of_source_subspace", ""),
                    "neighbor_preservation": diag.get("neighbor_preservation", ""),
                    "fold_proxy": diag.get("fold_proxy", ""),
                    "local_stretch_p95": diag.get("local_stretch_p95", ""),
                    "signal_channel_projection": diag.get("signal_channel_projection", ""),
                    "reservoir_projection": diag.get("reservoir_projection", ""),
                    "ce_alignment": diag.get("ce_alignment", ""),
                    "B2_transfer_gain": true_gains["B2_transfer_gain"],
                    "B3_safety_gain": true_gains["B3_safety_gain"],
                    "random_target_gap": true_gains["B2_transfer_gain"] - random_gains["B2_transfer_gain"],
                    "sign_flip_gap": true_gains["B2_transfer_gain"] - sign_gains["B2_transfer_gain"],
                    "corrupt_gap": true_gains["B2_transfer_gain"] - corrupt_gains["B2_transfer_gain"],
                    "C1_observer5_pass": c1_pass,
                    "official_C2_solver_claimed": 0,
                    "blocker": "" if c1_pass else "observer5_not_above_random_sign_corrupt_or_B3_safety",
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
    gen = torch.Generator(device=device).manual_seed(252_071 + train_seed + sum(ord(c) for c in run_kind))
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, Any]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    cached_target = torch.zeros(0, device=device)
    cached_diag: dict[str, Any] = {}
    target_norm_sum = 0.0
    injection_norm_sum = 0.0
    consensus_density_sum = 0.0
    intrinsic_sum = 0.0
    info_volume_sum = 0.0
    neighbor_sum = 0.0
    signal_sum = 0.0
    reservoir_sum = 0.0
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
        elif run_kind == "CTRL-RandomManifoldMatched":
            noise = torch.randn(grad.shape, device=device, generator=gen)
            injected = _cap_norm(normalized_like(noise, grad), grad, 0.12)
            _assign_flat_gradient(model, grad + injected)
            opt_adam.step()
            injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        elif run_kind.startswith("OBS5-"):
            if cached_target.numel() == 0 or (step - 1) % max(1, int(args.target_refresh)) == 0:
                cached_target, cached_diag = _manifold_target(args, model, x_train, y_train, run_kind)
                refresh_count += 1
                target_norm_sum += float(torch.linalg.vector_norm(cached_target.float()).item())
                consensus_density_sum += _f(cached_diag.get("manifold_consensus_density"), 0.0)
                intrinsic_sum += _f(cached_diag.get("intrinsic_dimension"), 0.0)
                info_volume_sum += _f(cached_diag.get("info_volume"), 0.0)
                neighbor_sum += _f(cached_diag.get("neighbor_preservation"), 0.0)
                signal_sum += _f(cached_diag.get("signal_channel_projection"), 0.0)
                reservoir_sum += _f(cached_diag.get("reservoir_projection"), 0.0)
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
        "manifold_consensus_density_mean": consensus_density_sum / refresh_count if refresh_count else "",
        "intrinsic_dimension_mean": intrinsic_sum / refresh_count if refresh_count else "",
        "info_volume_mean": info_volume_sum / refresh_count if refresh_count else "",
        "neighbor_preservation_mean": neighbor_sum / refresh_count if refresh_count else "",
        "signal_channel_projection_mean": signal_sum / refresh_count if refresh_count else "",
        "reservoir_projection_mean": reservoir_sum / refresh_count if refresh_count else "",
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
        if str(row.get("run_kind", "")).startswith("OBS5-"):
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
        (str(r.get("v21_id")), str(r.get("dataset")), str(r.get("train_seed")), str(r.get("run_kind"))): int_flag(r.get("C1_observer5_pass"))
        for r in c1_rows
    }
    summary: list[dict[str, Any]] = []
    for row in enriched:
        if not str(row.get("run_kind", "")).startswith("OBS5-"):
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
                "manifold_consensus_density_mean": row.get("manifold_consensus_density_mean", ""),
                "intrinsic_dimension_mean": row.get("intrinsic_dimension_mean", ""),
                "info_volume_mean": row.get("info_volume_mean", ""),
                "neighbor_preservation_mean": row.get("neighbor_preservation_mean", ""),
                "signal_channel_projection_mean": row.get("signal_channel_projection_mean", ""),
                "reservoir_projection_mean": row.get("reservoir_projection_mean", ""),
                "C1_observer5_pass": c1_pass,
                "C3_observer5_early_pass": early_pass,
                "C3_observer5_full_pass": full_pass,
                "official_C2_solver_claimed": 0,
                "official_C3_pass": 0,
                "blocker": "" if full_pass and c1_pass else "observer5_full_C3_or_C1_gate_failed",
            }
        )
    route = {
        "C1_observer5_rows": len(c1_rows),
        "C1_observer5_pass_rows": sum(int_flag(r.get("C1_observer5_pass")) for r in c1_rows),
        "C3_observer5_rows": len(summary),
        "C3_observer5_early_pass_rows": sum(int_flag(r.get("C3_observer5_early_pass")) for r in summary),
        "C3_observer5_full_pass_rows": sum(int_flag(r.get("C3_observer5_full_pass")) for r in summary),
        "official_C2_solver_claimed_rows": 0,
        "official_C3_pass_rows": 0,
        "row_positive_count_h100": sum(1 for r in summary if _f(r.get("source_vs_best_control_h100"), -999.0) >= 0.005),
        "row_positive_count_h400": sum(1 for r in summary if _f(r.get("source_vs_best_control_h400"), -999.0) >= 0.005),
        "row_positive_count_h800": sum(1 for r in summary if _f(r.get("source_vs_best_control_h800"), -999.0) >= 0.005),
        "row_positive_count_h1600": sum(1 for r in summary if _f(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005),
        "row_positive_count_h3200": sum(1 for r in summary if _f(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005),
        "route": "C3-Observer5ActivationManifoldOpened" if any(int_flag(r.get("C3_observer5_full_pass")) and int_flag(r.get("C1_observer5_pass")) for r in summary) else "C3-Observer5ActivationManifoldBlocked",
        "promotion_allowed": 0,
        "future_source_used_for_direction": 0,
        "blocker": "none" if any(int_flag(r.get("C3_observer5_full_pass")) for r in summary) else "observer5_h3200_or_C1_gate_failed",
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
        for kind in MANIFOLD_RUNS:
            raw_rows.append(_train_run(args, source_row, kind))
        key = (str(source_row.get("dataset", "")), str(source_row.get("seed", "")), str(source_row.get("train_seed", "")))
        if key not in seen_controls:
            seen_controls.add(key)
            for kind in CONTROL_RUNS:
                raw_rows.append(_train_run(args, source_row, kind))
    enriched, summary, route = _summarize(raw_rows, c1_rows)
    write_rows(out_dir / "v22_07_observer5_activation_manifold_c1.csv", c1_rows)
    write_rows(out_dir / "v22_07_observer5_activation_manifold_raw_runs.csv", raw_rows)
    write_rows(out_dir / "v22_07_observer5_activation_manifold_matrix.csv", enriched)
    write_rows(out_dir / "v22_07_observer5_activation_manifold_summary.csv", summary)
    write_json(out_dir / "v22_07_observer5_activation_manifold_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_observer5_activation_manifold.py --source-dir {args.source_dir} --device {args.device} --out-dir {out_dir}"
        + f" --top-k {args.top_k} --steps {args.steps} --target-refresh {args.target_refresh} --target-scale {args.target_scale} --fu-lr {args.fu_lr}",
        status="completed",
        note=(
            f"C1_pass={route['C1_observer5_pass_rows']} rows={route['C3_observer5_rows']} "
            f"early={route['C3_observer5_early_pass_rows']} full={route['C3_observer5_full_pass_rows']} "
            f"h1600_pos={route['row_positive_count_h1600']} h3200_pos={route['row_positive_count_h3200']} route={route['route']}"
        ),
    )


if __name__ == "__main__":
    main()
