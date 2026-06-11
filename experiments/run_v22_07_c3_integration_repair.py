#!/usr/bin/env python3
"""v22.07 C3 target-to-training-dynamics integration repair.

This runner is intentionally narrower than a full C3 matrix.  It takes C2
target-repair candidates that opened one-step actuation, converts the repaired
target into optimizer-integrated directions, and checks whether early source
survives to h800 against AdamW/SGD/NoOp/random controls.
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
from experiments.run_v22_07_metric_dynamics_fu import _filtered_matrix, _make_update_with_grad  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)


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
    p.add_argument("--top-k", type=int, default=2)
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


def _selected_candidates(out_dir: Path, top_k: int) -> list[dict[str, Any]]:
    rows = read_rows(out_dir / "v22_07_c2_solver_repair_matrix.csv")
    rows = [
        r
        for r in rows
        if int_flag(r.get("C2_repair_pass"))
        and str(r.get("block_role")) == "all"
        and str(r.get("target_repair_variant")) == "all_x64"
    ]
    rows = sorted(rows, key=lambda r: _f(r.get("B2_transfer_gain"), -999.0), reverse=True)
    return rows[: int(top_k)]


def _project_nonconflicting(target_grad: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    if target_grad.numel() == 0 or grad.numel() == 0:
        return target_grad
    dot = torch.dot(target_grad.float(), grad.float())
    denom = torch.dot(grad.float(), grad.float()).clamp_min(1.0e-12)
    if float(dot.item()) < 0.0:
        target_grad = target_grad - (dot / denom).to(target_grad.dtype) * grad
    return target_grad


def _orthogonal_component(target_grad: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    if target_grad.numel() == 0 or grad.numel() == 0:
        return target_grad
    denom = torch.dot(grad.float(), grad.float()).clamp_min(1.0e-12)
    dot = torch.dot(target_grad.float(), grad.float())
    return target_grad - (dot / denom).to(target_grad.dtype) * grad


def _cap_norm(vec: torch.Tensor, reference: torch.Tensor, ratio: float) -> torch.Tensor:
    vnorm = torch.linalg.vector_norm(vec.float())
    rnorm = torch.linalg.vector_norm(reference.float())
    cap = float(ratio) * rnorm
    if float(vnorm.item()) <= 1.0e-12 or float(cap.item()) <= 1.0e-12:
        return torch.zeros_like(vec)
    scale = min(1.0, float((cap / vnorm).item()))
    return vec * scale


def _target_as_gradient(update: UpdateTensor, scale: float) -> torch.Tensor:
    sign = 1.0 if update.sign_rule == "subtract" else -1.0
    return update.tensor.detach().clone() * float(scale) * sign


def _train_run(args: argparse.Namespace, source_row: dict[str, str], run_kind: str, repair_row: dict[str, Any] | None) -> dict[str, Any]:
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
    gen = torch.Generator(device=device).manual_seed(83_307 + train_seed + sum(ord(c) for c in run_kind))
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    xb0 = x_train[:batch_size]
    yb0 = y_train[:batch_size]
    mechanism = str(source_row.get("mechanism", ""))
    base_update = _make_update_with_grad(model, mechanism, xb0, yb0, seed=train_seed + 1)
    target_scale = _f(repair_row.get("target_scale"), 64.0) if repair_row else 64.0
    base_target_grad = _target_as_gradient(base_update, target_scale)
    long_state = torch.zeros_like(base_target_grad)
    traces: dict[int, dict[str, Any]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    injection_norm_sum = 0.0
    conflict_projection_count = 0
    for step in range(1, int(args.steps) + 1):
        idx = torch.randint(0, int(x_train.shape[0]), (batch_size,), generator=gen, device=device)
        xb = x_train[idx]
        yb = y_train[idx]
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        grad = flat_grad(model, device).detach()
        if run_kind == "CTRL-AdamW":
            opt_adam.step()
        elif run_kind == "CTRL-SGD":
            opt_sgd.step()
        elif run_kind == "CTRL-NoOp":
            pass
        elif run_kind == "CTRL-RandomMatched":
            noise = torch.randn(base_target_grad.shape, device=device, generator=gen)
            random_vec = normalized_like(noise, base_target_grad)
            injected = _cap_norm(random_vec, grad, 0.15)
            _assign_flat_gradient(model, grad + injected)
            opt_adam.step()
            injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        else:
            projected = _project_nonconflicting(base_target_grad, grad)
            if float(torch.dot(base_target_grad.float(), grad.float()).item()) < 0.0:
                conflict_projection_count += 1
            if run_kind == "INT-grad-transport-cap025":
                injected = _cap_norm(projected, grad, 0.25)
                _assign_flat_gradient(model, grad + injected)
                opt_adam.step()
            elif run_kind == "INT-short-long-ema-cap015":
                long_state = 0.98 * long_state + 0.02 * projected
                mix = projected if step <= 100 else long_state
                injected = _cap_norm(mix, grad, 0.15)
                _assign_flat_gradient(model, grad + injected)
                opt_adam.step()
            elif run_kind == "INT-low-nds-orthogonal-cap010":
                orth = _orthogonal_component(projected, grad)
                injected = _cap_norm(orth, grad, 0.10)
                _assign_flat_gradient(model, grad + injected)
                opt_adam.step()
            elif run_kind == "INT-post-adamw-slow-anchor-cap010":
                injected = _cap_norm(projected, grad, 0.10)
                opt_adam.step()
                update = UpdateTensor(
                    injected,
                    "optimizer_integrated_slow_anchor",
                    "subtract",
                    "parameter",
                    "v22_07_c3_integration_post_adamw_slow_anchor",
                    mechanism,
                    one_step_descent_claim=0,
                )
                apply_update(model, update, lr=float(args.fu_lr))
            else:
                injected = torch.zeros_like(grad)
                opt_adam.step()
            injection_norm_sum += float(torch.linalg.vector_norm(injected.float()).item())
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    row: dict[str, Any] = {
        "run_kind": run_kind,
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "target_repair_variant": repair_row.get("target_repair_variant", "") if repair_row else "",
        "target_scale": target_scale,
        "C2_repair_B2_transfer_gain": repair_row.get("B2_transfer_gain", "") if repair_row else "",
        "C2_repair_projection_residual_Gf": repair_row.get("projection_residual_Gf", "") if repair_row else "",
        "C2_repair_ActuationR2": repair_row.get("ActuationR2", "") if repair_row else "",
        "target_to_training_repair": int(run_kind.startswith("INT-")),
        "target_injection_norm_sum": injection_norm_sum,
        "conflict_projection_count": conflict_projection_count,
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
        if str(row.get("run_kind", "")).startswith("INT-"):
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
    candidates = [r for r in enriched if str(r.get("run_kind", "")).startswith("INT-")]
    for row in candidates:
        early_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800]))
        full_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        summary.append(
            {
                "run_kind": row.get("run_kind", ""),
                "v21_id": row.get("v21_id", ""),
                "target_repair_variant": row.get("target_repair_variant", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "target_injection_norm_sum": row.get("target_injection_norm_sum", ""),
                "conflict_projection_count": row.get("conflict_projection_count", ""),
                "C3_integration_early_pass": early_pass,
                "C3_integration_full_pass": full_pass,
                "official_C3_pass": 0,
                "blocker": "" if early_pass else "early_source_horizon_gate_failed",
            }
        )
    route = {
        "C3_integration_rows": len(summary),
        "C3_integration_early_pass_rows": sum(int_flag(r.get("C3_integration_early_pass")) for r in summary),
        "C3_integration_full_pass_rows": sum(int_flag(r.get("C3_integration_full_pass")) for r in summary),
        "official_C3_pass_rows": 0,
        "row_positive_count_h100": sum(1 for r in summary if _f(r.get("source_vs_best_control_h100"), -999.0) >= 0.005),
        "row_positive_count_h400": sum(1 for r in summary if _f(r.get("source_vs_best_control_h400"), -999.0) >= 0.005),
        "row_positive_count_h800": sum(1 for r in summary if _f(r.get("source_vs_best_control_h800"), -999.0) >= 0.005),
        "route": "C3-IntegrationEarlySourceOpened" if any(int_flag(r.get("C3_integration_early_pass")) for r in summary) else "C3-IntegrationRepairBlocked",
        "promotion_allowed": 0,
        "blocker": "official_full_C3_h3200_not_completed" if any(int_flag(r.get("C3_integration_early_pass")) for r in summary) else "target_to_training_dynamics_source_gate_failed",
    }
    return enriched, summary, route


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    candidates = _selected_candidates(out_dir, int(args.top_k))
    source_rows = _filtered_matrix(Path(args.source_dir), include_controls=1)
    by_order = {idx: row for idx, row in enumerate(source_rows)}
    raw_rows: list[dict[str, Any]] = []
    for cand in candidates:
        source_row = by_order.get(int(float(cand.get("job_order", -1))))
        if not source_row:
            continue
        for kind in [
            "INT-grad-transport-cap025",
            "INT-short-long-ema-cap015",
            "INT-low-nds-orthogonal-cap010",
            "INT-post-adamw-slow-anchor-cap010",
            "CTRL-AdamW",
            "CTRL-SGD",
            "CTRL-NoOp",
            "CTRL-RandomMatched",
        ]:
            raw_rows.append(_train_run(args, source_row, kind, cand))
    enriched, summary, route = _summarize(raw_rows)
    write_rows(out_dir / "v22_07_c3_integration_repair_raw_runs.csv", raw_rows)
    write_rows(out_dir / "v22_07_c3_integration_repair_matrix.csv", enriched)
    write_rows(out_dir / "v22_07_c3_integration_repair_summary.csv", summary)
    write_json(out_dir / "v22_07_c3_integration_repair_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_c3_integration_repair.py --source-dir {args.source_dir} --device {args.device} --out-dir {out_dir}",
        status="completed",
        note=f"rows={len(summary)} early_pass={route['C3_integration_early_pass_rows']} h800_pos={route['row_positive_count_h800']} route={route['route']}",
    )


if __name__ == "__main__":
    main()
