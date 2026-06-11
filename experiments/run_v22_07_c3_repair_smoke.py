#!/usr/bin/env python3
"""v22.07 C3 source-formation smoke for repaired C0/C1/C2 candidates."""

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

from dgkan.fu.core import UpdateTensor, apply_update, normalized_like  # noqa: E402
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
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--top-k", type=int, default=9)
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


def _selected_candidates(out_dir: Path, top_k: int) -> list[dict[str, Any]]:
    rows = read_rows(out_dir / "v22_07_c2_solver_repair_matrix.csv")
    rows = [
        r
        for r in rows
        if int_flag(r.get("C2_repair_pass"))
        and str(r.get("block_role")) == "all"
        and str(r.get("target_repair_variant", "")).startswith("all_x")
    ]
    rows = sorted(rows, key=lambda r: _f(r.get("B2_transfer_gain"), -999.0), reverse=True)
    return rows[: int(top_k)]


def _source_row_by_order(source_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {idx: row for idx, row in enumerate(source_rows)}


def _train_run(args: argparse.Namespace, source_row: dict[str, str], run_kind: str, target_scale: float, repair_row: dict[str, Any] | None) -> dict[str, Any]:
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
    gen = torch.Generator(device=device).manual_seed(72_207 + train_seed + sum(ord(c) for c in run_kind))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, Any]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    mechanism = str(source_row.get("mechanism", ""))
    for step in range(1, int(args.steps) + 1):
        idx = torch.randint(0, int(x_train.shape[0]), (batch_size,), generator=gen, device=device)
        xb = x_train[idx]
        yb = y_train[idx]
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        if run_kind == "CTRL-AdamW":
            opt_adam.step()
        elif run_kind == "CTRL-SGD":
            opt_sgd.step()
        elif run_kind == "CTRL-NoOp":
            pass
        elif run_kind == "CTRL-RandomMatched":
            base_update = _make_update_with_grad(model, mechanism, xb, yb, seed=train_seed + step)
            tensor = base_update.tensor.detach() * float(target_scale)
            noise = torch.randn(tensor.shape, device=device, generator=gen)
            random_update = UpdateTensor(
                normalized_like(noise, tensor),
                "matched_random_target",
                base_update.sign_rule,
                base_update.space,
                "v22_07_c3_random_matched_target_scale",
                "CTRL-RandomMatchedTarget",
                one_step_descent_claim=0,
            )
            opt_adam.step()
            apply_update(model, random_update, lr=float(args.fu_lr))
        else:
            update = _make_update_with_grad(model, mechanism, xb, yb, seed=train_seed + step)
            scaled = UpdateTensor(
                update.tensor.detach() * float(target_scale),
                update.kind,
                update.sign_rule,
                update.space,
                f"v22_07_c3_repair_target_scale_x{target_scale:g}",
                update.mechanism,
                role=update.role,
                one_step_descent_claim=update.one_step_descent_claim,
                diagnostics=dict(update.diagnostics or {}),
            )
            opt_adam.step()
            apply_update(model, scaled, lr=float(args.fu_lr))
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    row: dict[str, Any] = {
        "run_kind": run_kind,
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "target_scale": target_scale,
        "target_repair_variant": repair_row.get("target_repair_variant", "") if repair_row else "",
        "C2_repair_B2_transfer_gain": repair_row.get("B2_transfer_gain", "") if repair_row else "",
        "C2_repair_projection_residual_Gf": repair_row.get("projection_residual_Gf", "") if repair_row else "",
        "C2_repair_ActuationR2": repair_row.get("ActuationR2", "") if repair_row else "",
        "execution_status": "measured",
    }
    for h in HORIZONS:
        metrics = traces.get(h, {})
        for key, value in metrics.items():
            row[f"{key}_h{h}"] = value
    return row


def _summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controls: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        if not str(row.get("run_kind", "")).startswith("CTRL-"):
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

    candidate_rows = [r for r in enriched if not str(r.get("run_kind", "")).startswith("CTRL-")]
    summary: list[dict[str, Any]] = []
    for row in candidate_rows:
        pass_flag = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        debt_ok = int(
            _f(row.get("CEp99_h3200"), 0.0) <= 20.0
            and _f(row.get("ECE_h3200"), 0.0) <= 1.0
            and _f(row.get("Brier_h3200"), 0.0) <= 1.0
        )
        summary.append(
            {
                "v21_id": row.get("v21_id", ""),
                "target_repair_variant": row.get("target_repair_variant", ""),
                "target_scale": row.get("target_scale", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h2400": row.get("source_vs_best_control_h2400", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "CEp99_h3200": row.get("CEp99_h3200", ""),
                "ECE_h3200": row.get("ECE_h3200", ""),
                "Brier_h3200": row.get("Brier_h3200", ""),
                "debt_not_exploded": debt_ok,
                "C3_repair_smoke_pass": int(pass_flag and debt_ok),
                "official_C3_pass": 0,
                "blocker": "" if pass_flag and debt_ok else "source_horizon_or_debt_gate_failed",
            }
        )
    positives = {
        h: sum(1 for r in summary if _f(r.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005)
        for h in HORIZONS
    }
    route = {
        "C3_repair_smoke_rows": len(summary),
        "C3_repair_smoke_pass_rows": sum(int_flag(r.get("C3_repair_smoke_pass")) for r in summary),
        "official_C3_pass_rows": 0,
        "row_positive_count_h100": positives[100],
        "row_positive_count_h400": positives[400],
        "row_positive_count_h800": positives[800],
        "row_positive_count_h1600": positives[1600],
        "row_positive_count_h3200": positives[3200],
        "route": "C3-RepairSmokeOpened" if any(int_flag(r.get("C3_repair_smoke_pass")) for r in summary) else "C3-RepairSmokeBlocked",
        "promotion_allowed": 0,
        "blocker": "official_independent_9row_C3_not_completed" if any(int_flag(r.get("C3_repair_smoke_pass")) for r in summary) else "source_horizon_or_debt_gate_failed",
    }
    return enriched, {"summary": summary, "route": route}


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    candidates = _selected_candidates(out_dir, int(args.top_k))
    source_rows = _filtered_matrix(Path(args.source_dir), include_controls=1)
    by_order = _source_row_by_order(source_rows)
    raw_rows: list[dict[str, Any]] = []
    control_keys: set[tuple[int, str]] = set()
    for cand in candidates:
        job_order = int(float(cand.get("job_order", -1)))
        source_row = by_order.get(job_order)
        if not source_row:
            continue
        scale = _f(cand.get("target_scale"), 1.0)
        raw_rows.append(_train_run(args, source_row, "C3Repair-AdamWPlusScaledFU", scale, cand))
        key = (job_order, str(cand.get("target_repair_variant", "")))
        control_keys.add(key)
        for control in ["CTRL-AdamW", "CTRL-SGD", "CTRL-NoOp", "CTRL-RandomMatched"]:
            raw_rows.append(_train_run(args, source_row, control, scale, cand))
    enriched, summary_pack = _summary(raw_rows)
    summary = summary_pack["summary"]
    route = summary_pack["route"]
    write_rows(out_dir / "v22_07_c3_repair_smoke_raw_runs.csv", raw_rows)
    write_rows(out_dir / "v22_07_c3_repair_smoke_matrix.csv", enriched)
    write_rows(out_dir / "v22_07_c3_repair_smoke_summary.csv", summary)
    write_json(out_dir / "v22_07_c3_repair_smoke_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_c3_repair_smoke.py --source-dir {args.source_dir} --device {args.device} --out-dir {out_dir}",
        status="completed",
        note=f"rows={len(summary)} C3_smoke_pass={route['C3_repair_smoke_pass_rows']} h800_pos={route['row_positive_count_h800']} h3200_pos={route['row_positive_count_h3200']} route={route['route']}",
    )


if __name__ == "__main__":
    main()
