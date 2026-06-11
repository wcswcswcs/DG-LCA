#!/usr/bin/env python3
"""v22.07 C3 train-only semantic target repair smoke.

This runner tests target/source semantics after target-rescale and
optimizer-integration repairs failed.  It builds train-only split-consensus,
corrupt-reject, class-balanced, and readout-channel target gradients, then
checks whether any early source survives to h800 against AdamW/SGD/NoOp/random
controls.  It does not claim an official C2 metric-solver pass.
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

from dgkan.fu.core import UpdateTensor, apply_update, flat_grad, normalized_like, trainable_parameters  # noqa: E402
from experiments.run_v17_common import carrier_model, classification_brier, classification_ece, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_07_common import PYTHON, V2206_COMBINED_SOURCE, append_exec, ensure_out, finite_float, int_flag, read_rows, write_json, write_rows  # noqa: E402
from experiments.run_v22_07_metric_dynamics_fu import _filtered_matrix, _simulate_gains  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)
SEMANTIC_RUNS = (
    "SEM-split-consensus-corrupt-reject-cap010",
    "SEM-class-balanced-corrupt-reject-cap015",
    "SEM-b3-stable-residual-cap010",
    "SEM-readout-channel-consensus-cap020",
    "SEM-contrastive-minus-corrupt-cap010",
    "SEM-direct-split-consensus-cap005",
    "SEM-direct-class-balanced-cap008",
    "SEM-direct-readout-consensus-cap010",
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
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--target-scale", type=float, default=16.0)
    p.add_argument("--target-refresh", type=int, default=25)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--selection-file", default="")
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _eval(model: torch.nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        train_loss = float(F.cross_entropy(model(x_train).float(), y_train).item())
        logits = model(x_val).float()
        val_loss = float(F.cross_entropy(logits, y_val).item())
        ce = F.cross_entropy(logits, y_val, reduction="none")
        pred = logits.argmax(dim=1)
    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": float((pred == y_val).float().mean().item()),
        "CEp99": float(torch.quantile(ce.detach(), 0.99).item()),
        "ECE": classification_ece(logits, y_val),
        "Brier": classification_brier(logits, y_val),
    }


def _assign_flat_gradient(model: torch.nn.Module, vec: torch.Tensor) -> None:
    offset = 0
    for param in trainable_parameters(model):
        n = int(param.numel())
        param.grad = vec[offset : offset + n].view_as(param).to(device=param.device, dtype=param.dtype).detach().clone()
        offset += n


def _chunks(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    a = max(2, n // 3)
    b = max(a + 2, 2 * n // 3)
    return x[:a], y[:a], x[a:b], y[a:b], x[b:], y[b:]


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


def _remove_positive_component(vec: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, float]:
    if vec.numel() == 0 or ref.numel() == 0:
        return vec, 0.0
    dot = torch.dot(vec.float(), ref.float())
    denom = torch.dot(ref.float(), ref.float()).clamp_min(1.0e-12)
    if float(dot.item()) <= 0.0:
        return vec, 0.0
    removed = (dot / denom).to(vec.dtype) * ref
    return vec - removed, float(torch.linalg.vector_norm(removed.float()).item())


def _orthogonal_component(vec: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if vec.numel() == 0 or ref.numel() == 0:
        return vec
    dot = torch.dot(vec.float(), ref.float())
    denom = torch.dot(ref.float(), ref.float()).clamp_min(1.0e-12)
    return vec - (dot / denom).to(vec.dtype) * ref


def _cap_norm(vec: torch.Tensor, reference: torch.Tensor, ratio: float) -> torch.Tensor:
    vnorm = torch.linalg.vector_norm(vec.float())
    rnorm = torch.linalg.vector_norm(reference.float())
    cap = float(ratio) * rnorm
    if float(vnorm.item()) <= 1.0e-12 or float(cap.item()) <= 1.0e-12:
        return torch.zeros_like(vec)
    return vec * min(1.0, float((cap / vnorm).item()))


def _role_mask(model: torch.nn.Module, role: str, device: torch.device) -> torch.Tensor:
    params = trainable_parameters(model)
    chunks = []
    last_two = {max(0, len(params) - 1), max(0, len(params) - 2)}
    for idx, param in enumerate(params):
        keep = 1.0
        if role == "readout":
            keep = 1.0 if idx in last_two else 0.0
        elif role == "matrix":
            keep = 1.0 if param.ndim >= 2 else 0.0
        chunks.append(torch.full_like(param.detach(), keep).reshape(-1))
    return torch.cat(chunks).to(device=device) if chunks else torch.zeros(0, device=device)


def _agreement_vector(vecs: list[torch.Tensor]) -> tuple[torch.Tensor, float]:
    if not vecs:
        return torch.zeros(0), 0.0
    stacked = torch.stack([v.detach() for v in vecs], dim=0)
    signs = torch.sign(stacked.float())
    agree = (torch.abs(signs.sum(dim=0)) == float(len(vecs))) & (torch.abs(signs).sum(dim=0) == float(len(vecs)))
    avg = stacked.mean(dim=0)
    masked = torch.where(agree.to(device=avg.device), avg, torch.zeros_like(avg))
    density = float(agree.float().mean().item()) if agree.numel() else 0.0
    if float(torch.linalg.vector_norm(masked.float()).item()) <= 1.0e-12:
        return avg, density
    return masked, density


def _semantic_target(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    run_kind: str,
    *,
    target_scale: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    xa, ya, xb, yb, xc, yc = _chunks(x, y)
    g_all = _grad_vec(model, x, y)
    g_a = _grad_vec(model, xa, ya)
    g_b = _grad_vec(model, xb, yb)
    g_c = _grad_vec(model, xc, yc)
    g_bal = _grad_vec(model, x, y, class_balanced=True)
    g_corrupt = _grad_vec(model, x, (y + 1) % max(2, int(y.max().item()) + 1))
    consensus, density = _agreement_vector([g_a, g_b, g_c])
    removed_corrupt = 0.0
    role = "all"
    cap = 0.10
    if run_kind == "SEM-split-consensus-corrupt-reject-cap010":
        target, removed_corrupt = _remove_positive_component(consensus, g_corrupt)
        target = 0.75 * target + 0.25 * _orthogonal_component(g_all, g_corrupt)
        cap = 0.10
    elif run_kind == "SEM-class-balanced-corrupt-reject-cap015":
        target, removed_corrupt = _remove_positive_component(g_bal, g_corrupt)
        target = 0.65 * target + 0.35 * consensus
        cap = 0.15
    elif run_kind == "SEM-b3-stable-residual-cap010":
        ab, _removed_ab = _remove_positive_component(0.50 * (g_a + g_b), g_corrupt)
        target = 0.70 * ab + 0.30 * g_c
        target = _orthogonal_component(target, g_corrupt)
        removed_corrupt = _removed_ab
        cap = 0.10
    elif run_kind == "SEM-readout-channel-consensus-cap020":
        target, removed_corrupt = _remove_positive_component(consensus, g_corrupt)
        role = "readout"
        target = target * _role_mask(model, role, x.device)
        cap = 0.20
    elif run_kind == "SEM-contrastive-minus-corrupt-cap010":
        target = 0.50 * consensus + 0.50 * g_bal - 0.35 * normalized_like(g_corrupt, g_all)
        target, removed_corrupt = _remove_positive_component(target, g_corrupt)
        cap = 0.10
    elif run_kind == "SEM-direct-split-consensus-cap005":
        target, removed_corrupt = _remove_positive_component(consensus, g_corrupt)
        cap = 0.05
    elif run_kind == "SEM-direct-class-balanced-cap008":
        target, removed_corrupt = _remove_positive_component(g_bal, g_corrupt)
        cap = 0.08
    elif run_kind == "SEM-direct-readout-consensus-cap010":
        target, removed_corrupt = _remove_positive_component(consensus, g_corrupt)
        role = "readout"
        target = target * _role_mask(model, role, x.device)
        cap = 0.10
    else:
        target = consensus
    target = normalized_like(target, g_all) * float(target_scale)
    diag = {
        "target_semantic_family": run_kind,
        "agreement_density": density,
        "corrupt_removed_norm": removed_corrupt,
        "target_role": role,
        "target_cap_ratio": cap,
        "official_C2_solver_claimed": 0,
    }
    return target.detach().clone(), diag


def _selected_candidates(out_dir: Path, top_k: int, selection_file: str = "") -> list[dict[str, Any]]:
    rows = read_rows(Path(selection_file)) if str(selection_file).strip() else read_rows(out_dir / "v22_07_c0_estimator_repair_top20.csv")
    selected = []
    for row in rows:
        if str(row.get("v21_id", "")).startswith("CTRL"):
            continue
        selected.append(row)
        if len(selected) >= int(top_k):
            break
    return selected


def _c1_semantic_rows(args: argparse.Namespace, source_rows: list[dict[str, str]], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    device = resolve_device(args.device)
    by_order = {idx: row for idx, row in enumerate(source_rows)}
    rows: list[dict[str, Any]] = []
    for cand in selected:
        source_row = by_order.get(int(float(cand.get("job_order", -1))))
        if not source_row:
            continue
        dataset = str(source_row.get("dataset", "MNIST"))
        seed = int(float(source_row.get("seed", 0) or 0))
        local = deepcopy(args)
        local.basis_repair_variant = str(source_row.get("basis_repair_variant", "R0-current") or "R0-current")
        x_train, y_train, _x_val, _y_val = load_dataset(dataset, Path(args.data_root), int(args.train_size), int(args.val_size), seed, device, int(args.input_size))
        train_seed = int(float(source_row.get("train_seed") or 0)) if str(source_row.get("train_seed", "")).strip() else stable_train_seed(source_row)
        model = carrier_model("MLP", x_train, train_seed, local, device)
        xb = x_train[: min(int(args.batch_size), int(x_train.shape[0]))]
        yb = y_train[: int(xb.shape[0])]
        for run_kind in SEMANTIC_RUNS:
            target, diag = _semantic_target(model, xb, yb, run_kind, target_scale=float(args.target_scale))
            update = UpdateTensor(target, run_kind, "subtract", "parameter", "v22_07_semantic_target_repair", str(source_row.get("mechanism", "")), one_step_descent_claim=1, diagnostics=diag)
            gen = torch.Generator(device=device).manual_seed(train_seed + sum(ord(c) for c in run_kind) + 21_011)
            noise = torch.randn(target.shape, device=device, generator=gen)
            random_update = UpdateTensor(normalized_like(noise, target), "matched_random_target", "subtract", "parameter", "v22_07_semantic_random_matched", "CTRL-RandomMatchedTarget", one_step_descent_claim=0)
            sign_update = UpdateTensor(target, "sign_flip_target", "add", "parameter", "v22_07_semantic_sign_flip", "CTRL-SignFlipTarget", one_step_descent_claim=0)
            corrupt_target, _corrupt_diag = _semantic_target(model, xb, (yb + 1) % max(2, int(args.classes)), run_kind, target_scale=float(args.target_scale))
            corrupt_update = UpdateTensor(corrupt_target, "corrupt_semantic_target", "subtract", "parameter", "v22_07_semantic_corrupt_target", "CTRL-CorruptSemanticTarget", one_step_descent_claim=0)
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
                    "repair_scope": "semantic_target_repair_c1_no_future_direction",
                    "job_order": cand.get("job_order", ""),
                    "v21_id": source_row.get("v21_id", ""),
                    "mechanism": source_row.get("mechanism", ""),
                    "dataset": dataset,
                    "seed": seed,
                    "train_seed": train_seed,
                    "run_kind": run_kind,
                    "target_scale": args.target_scale,
                    "target_norm": float(torch.linalg.vector_norm(target.float()).item()),
                    "agreement_density": diag.get("agreement_density", ""),
                    "corrupt_removed_norm": diag.get("corrupt_removed_norm", ""),
                    "target_role": diag.get("target_role", ""),
                    "B1_gain": true_gains["B1_gain"],
                    "B2_transfer_gain": true_gains["B2_transfer_gain"],
                    "B3_safety_gain": true_gains["B3_safety_gain"],
                    "random_target_B2": random_gains["B2_transfer_gain"],
                    "sign_flip_B2": sign_gains["B2_transfer_gain"],
                    "corrupt_target_B2": corrupt_gains["B2_transfer_gain"],
                    "random_target_gap": true_gains["B2_transfer_gain"] - random_gains["B2_transfer_gain"],
                    "sign_flip_gap": true_gains["B2_transfer_gain"] - sign_gains["B2_transfer_gain"],
                    "corrupt_gap": true_gains["B2_transfer_gain"] - corrupt_gains["B2_transfer_gain"],
                    "C1_semantic_pass": c1_pass,
                    "official_C2_solver_claimed": 0,
                    "blocker": "" if c1_pass else "semantic_target_not_above_random_sign_corrupt_or_B3_safety",
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
    gen = torch.Generator(device=device).manual_seed(91_027 + train_seed + sum(ord(c) for c in run_kind))
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, Any]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    cached_target = torch.zeros(0, device=device)
    cached_diag: dict[str, Any] = {}
    injection_norm_sum = 0.0
    corrupt_removed_norm_sum = 0.0
    agreement_density_sum = 0.0
    refresh_count = 0
    for step in range(1, int(args.steps) + 1):
        idx = torch.randint(0, int(x_train.shape[0]), (batch_size,), generator=gen, device=device)
        xb = x_train[idx]
        yb = y_train[idx]
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
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
                cached_target, cached_diag = _semantic_target(model, x_train, y_train, run_kind, target_scale=1.0)
                refresh_count += 1
                corrupt_removed_norm_sum += _f(cached_diag.get("corrupt_removed_norm"), 0.0)
                agreement_density_sum += _f(cached_diag.get("agreement_density"), 0.0)
            target = cached_target if "SEM-direct-" in run_kind else _orthogonal_component(cached_target, grad)
            cap = _f(cached_diag.get("target_cap_ratio"), 0.10)
            injected = _cap_norm(target, grad, cap)
            _assign_flat_gradient(model, grad + injected)
            opt_adam.step()
            injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    row: dict[str, Any] = {
        "run_kind": run_kind,
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": source_row.get("mechanism", ""),
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "target_to_training_repair": int(run_kind.startswith("SEM-")),
        "target_injection_norm_sum": injection_norm_sum,
        "target_refresh_count": refresh_count,
        "agreement_density_mean": agreement_density_sum / refresh_count if refresh_count else "",
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
        if str(row.get("run_kind", "")).startswith("SEM-"):
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
    c1_pass_by_key = {
        (str(r.get("v21_id")), str(r.get("run_kind"))): int_flag(r.get("C1_semantic_pass"))
        for r in c1_rows
    }
    summary: list[dict[str, Any]] = []
    for row in enriched:
        if not str(row.get("run_kind", "")).startswith("SEM-"):
            continue
        early_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800]))
        full_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        c1_pass = c1_pass_by_key.get((str(row.get("v21_id")), str(row.get("run_kind"))), 0)
        summary.append(
            {
                "run_kind": row.get("run_kind", ""),
                "v21_id": row.get("v21_id", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "target_injection_norm_sum": row.get("target_injection_norm_sum", ""),
                "target_refresh_count": row.get("target_refresh_count", ""),
                "agreement_density_mean": row.get("agreement_density_mean", ""),
                "corrupt_removed_norm_mean": row.get("corrupt_removed_norm_mean", ""),
                "C1_semantic_pass": c1_pass,
                "C3_semantic_early_pass": early_pass,
                "C3_semantic_full_pass": full_pass,
                "official_C2_solver_claimed": 0,
                "official_C3_pass": 0,
                "blocker": "" if early_pass and c1_pass else "semantic_target_or_early_source_gate_failed",
            }
        )
    route = {
        "C1_semantic_rows": len(c1_rows),
        "C1_semantic_pass_rows": sum(int_flag(r.get("C1_semantic_pass")) for r in c1_rows),
        "C3_semantic_rows": len(summary),
        "C3_semantic_early_pass_rows": sum(int_flag(r.get("C3_semantic_early_pass")) for r in summary),
        "C3_semantic_full_pass_rows": sum(int_flag(r.get("C3_semantic_full_pass")) for r in summary),
        "official_C2_solver_claimed": 0,
        "official_C3_pass_rows": 0,
        "row_positive_count_h100": sum(1 for r in summary if _f(r.get("source_vs_best_control_h100"), -999.0) >= 0.005),
        "row_positive_count_h400": sum(1 for r in summary if _f(r.get("source_vs_best_control_h400"), -999.0) >= 0.005),
        "row_positive_count_h800": sum(1 for r in summary if _f(r.get("source_vs_best_control_h800"), -999.0) >= 0.005),
        "route": "C3-SemanticTargetRepairOpened" if any(int_flag(r.get("C3_semantic_early_pass")) and int_flag(r.get("C1_semantic_pass")) for r in summary) else "C3-SemanticTargetRepairBlocked",
        "promotion_allowed": 0,
        "blocker": "official_C2_and_full_C3_not_completed" if any(int_flag(r.get("C3_semantic_early_pass")) for r in summary) else "semantic_train_only_target_source_gate_failed",
    }
    return enriched, summary, route


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_rows = _filtered_matrix(Path(args.source_dir), include_controls=1)
    source_by_order = {idx: row for idx, row in enumerate(source_rows)}
    selected = _selected_candidates(out_dir, int(args.top_k), str(args.selection_file))
    c1_rows = _c1_semantic_rows(args, source_rows, selected)
    raw_rows: list[dict[str, Any]] = []
    for cand in selected:
        source_row = source_by_order.get(int(float(cand.get("job_order", -1))))
        if not source_row:
            continue
        for kind in [*SEMANTIC_RUNS, *CONTROL_RUNS]:
            raw_rows.append(_train_run(args, source_row, kind))
    enriched, summary, route = _summarize(raw_rows, c1_rows)
    write_rows(out_dir / "v22_07_c3_semantic_target_repair_c1.csv", c1_rows)
    write_rows(out_dir / "v22_07_c3_semantic_target_repair_raw_runs.csv", raw_rows)
    write_rows(out_dir / "v22_07_c3_semantic_target_repair_matrix.csv", enriched)
    write_rows(out_dir / "v22_07_c3_semantic_target_repair_summary.csv", summary)
    write_json(out_dir / "v22_07_c3_semantic_target_repair_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_c3_semantic_target_repair.py --source-dir {args.source_dir} --device {args.device} --out-dir {out_dir}"
        + f" --top-k {args.top_k} --steps {args.steps} --target-scale {args.target_scale} --target-refresh {args.target_refresh}"
        + (f" --selection-file {args.selection_file}" if str(args.selection_file).strip() else ""),
        status="completed",
        note=(
            f"C1_pass={route['C1_semantic_pass_rows']} rows={route['C3_semantic_rows']} "
            f"early_pass={route['C3_semantic_early_pass_rows']} h800_pos={route['row_positive_count_h800']} route={route['route']}"
        ),
    )


if __name__ == "__main__":
    main()
