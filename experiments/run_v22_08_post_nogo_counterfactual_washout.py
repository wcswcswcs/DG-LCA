#!/usr/bin/env python3
"""v22.08 post-no-go counterfactual washout certificate.

C-O7 is a new train-only retained-source observability probe after C-O6 fails.
It asks whether a candidate functional direction leaves a persistent
train-stream functional displacement after a short AdamW washout, compared
with sign-flip, corrupt-label, and random matched controls.  Future source
labels remain audit columns only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dgkan.fu.core import UpdateTensor, apply_update, cosine, flat_params, normalized_like  # noqa: E402
from dgkan.fu.mechanisms import CONTROL_MECHANISMS, make_update  # noqa: E402
from experiments.run_v17_common import carrier_model, classification_brier, classification_ece, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_07_metric_dynamics_fu import _filtered_matrix  # noqa: E402
from experiments.run_v22_08_common import (  # noqa: E402
    PYTHON,
    V2206_COMBINED_SOURCE,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_json,
    simple_svg,
    write_json,
    write_rows,
)
from experiments.run_v22_08_post_nogo_source_state_certificate import _auc, _precision_topk, _rank, _zscore  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2206_COMBINED_SOURCE))
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--data-root", default="data")
    p.add_argument("--train-size", type=int, default=64)
    p.add_argument("--val-size", type=int, default=48)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--washout-steps", type=int, default=12)
    p.add_argument("--max-probe-jobs", type=int, default=0)
    p.add_argument("--fresh-top-k", type=int, default=2)
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--refresh-interval", type=int, default=8)
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _flip(update: UpdateTensor) -> UpdateTensor:
    sign = "add" if update.sign_rule == "subtract" else "subtract"
    return UpdateTensor(
        update.tensor.detach(),
        update.kind,
        sign,
        update.space,
        "v22_08_counterfactual_sign_flip_control",
        update.mechanism,
        role=update.role,
        one_step_descent_claim=0,
        diagnostics=dict(update.diagnostics or {}),
    )


def _random_like(update: UpdateTensor, gen: torch.Generator) -> UpdateTensor:
    noise = torch.randn(update.tensor.shape, device=update.tensor.device, generator=gen)
    return UpdateTensor(
        normalized_like(noise, update.tensor.detach()),
        "counterfactual_random_matched",
        update.sign_rule,
        update.space,
        "v22_08_counterfactual_random_matched_control",
        update.mechanism,
        role="random_matched_washout_control",
        one_step_descent_claim=0,
    )


def _eval(model: torch.nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor) -> dict[str, float]:
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


def _candidate_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = _filtered_matrix(Path(args.source_dir), include_controls=1)
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        v21_id = str(row.get("v21_id", ""))
        mechanism = str(row.get("mechanism", ""))
        if not v21_id.startswith("MLP-V2206"):
            continue
        if mechanism in CONTROL_MECHANISMS:
            continue
        key = (v21_id, str(row.get("dataset", "")), str(row.get("seed", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    max_jobs = int(args.max_probe_jobs)
    return out[:max_jobs] if max_jobs > 0 else out


def _loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(model(x).float(), y).item())


def _logit_delta_norm(model_a: torch.nn.Module, model_b: torch.nn.Module, x: torch.Tensor) -> float:
    with torch.no_grad():
        delta = model_a(x).float() - model_b(x).float()
    return float(torch.linalg.vector_norm(delta.flatten()).item())


def _short_train(model: torch.nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]], lr: float, weight_decay: float) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for xb, yb in batches:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        opt.step()


def _probe_one(args: argparse.Namespace, row: dict[str, str], job_order: int) -> dict[str, Any]:
    start = time.perf_counter()
    device = resolve_device(args.device)
    dataset = str(row.get("dataset", "MNIST"))
    seed = int(float(row.get("seed", 0) or 0))
    local = deepcopy(args)
    local.basis_repair_variant = str(row.get("basis_repair_variant", "R0-current") or "R0-current")
    x_train, y_train, _x_val, _y_val = load_dataset(dataset, Path(args.data_root), int(args.train_size), int(args.val_size), seed, device, int(args.input_size))
    train_seed = int(float(row.get("train_seed") or 0)) if str(row.get("train_seed", "")).strip() else stable_train_seed(row)
    model = carrier_model("MLP", x_train, train_seed, local, device)
    batch = min(int(args.batch_size), int(x_train.shape[0]))
    xb = x_train[:batch]
    yb = y_train[:batch]
    x_b2 = x_train[batch : min(2 * batch, int(x_train.shape[0]))]
    y_b2 = y_train[batch : min(2 * batch, int(y_train.shape[0]))]
    if int(x_b2.shape[0]) == 0:
        x_b2, y_b2 = xb, yb
    x_b3 = x_train[::2][:batch]
    y_b3 = y_train[::2][:batch]
    mechanism = str(row.get("mechanism", ""))
    gen = torch.Generator(device=device).manual_seed(228_807 + train_seed + job_order)
    raw = make_update(model, mechanism, xb, yb, seed=train_seed + 71)
    corrupt = make_update(model, mechanism, xb, (yb + 1) % max(2, int(args.classes)), seed=train_seed + 72)
    variants = {
        "source": raw,
        "random": _random_like(raw, gen),
        "sign_flip": _flip(raw),
        "corrupt": corrupt,
    }
    wash_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(max(1, int(args.washout_steps))):
        idx = torch.randint(0, int(x_train.shape[0]), (batch,), generator=gen, device=device)
        wash_batches.append((x_train[idx], y_train[idx]))
    baseline = deepcopy(model)
    _short_train(baseline, wash_batches, float(args.lr), float(args.weight_decay))
    base_b2 = _loss(baseline, x_b2, y_b2)
    base_b3 = _loss(baseline, x_b3, y_b3)
    metrics: dict[str, float] = {}
    for name, update in variants.items():
        clone0 = deepcopy(model)
        apply_update(clone0, update, lr=float(args.fu_lr))
        initial_delta = _logit_delta_norm(clone0, model, x_train)
        clone = deepcopy(clone0)
        _short_train(clone, wash_batches, float(args.lr), float(args.weight_decay))
        final_delta = _logit_delta_norm(clone, baseline, x_train)
        metrics[f"{name}_initial_delta_norm"] = initial_delta
        metrics[f"{name}_final_delta_norm"] = final_delta
        metrics[f"{name}_washout_retention_ratio"] = final_delta / max(initial_delta, 1.0e-12)
        metrics[f"{name}_B2_loss_gain_after_washout"] = base_b2 - _loss(clone, x_b2, y_b2)
        metrics[f"{name}_B3_loss_gain_after_washout"] = base_b3 - _loss(clone, x_b3, y_b3)
    source = {h: _f(row.get(f"source_h{h}"), -999.0) for h in [100, 400, 800, 1600, 3200]}
    retention_controls = max(metrics["random_washout_retention_ratio"], metrics["sign_flip_washout_retention_ratio"], metrics["corrupt_washout_retention_ratio"])
    b2_controls = max(metrics["random_B2_loss_gain_after_washout"], metrics["sign_flip_B2_loss_gain_after_washout"], metrics["corrupt_B2_loss_gain_after_washout"])
    b3_controls = max(metrics["random_B3_loss_gain_after_washout"], metrics["sign_flip_B3_loss_gain_after_washout"], metrics["corrupt_B3_loss_gain_after_washout"])
    param_cos = abs(cosine(flat_params(model).detach().float(), raw.tensor.detach().float()))
    return {
        "job_order": job_order,
        "v21_id": row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "washout_steps": int(args.washout_steps),
        **metrics,
        "source_retention_control_gap": metrics["source_washout_retention_ratio"] - retention_controls,
        "source_B2_control_gap_after_washout": metrics["source_B2_loss_gain_after_washout"] - b2_controls,
        "source_B3_control_gap_after_washout": metrics["source_B3_loss_gain_after_washout"] - b3_controls,
        "source_param_radial_cosine": param_cos,
        "future_audit_source_h100": row.get("source_h100", ""),
        "future_audit_source_h400": row.get("source_h400", ""),
        "future_audit_source_h800": row.get("source_h800", ""),
        "future_audit_source_h1600": row.get("source_h1600", ""),
        "future_audit_source_h3200": row.get("source_h3200", ""),
        "official_early_chain_h100_h400_h800_positive": int(source[100] >= 0.005 and source[400] >= 0.005 and source[800] >= 0.005),
        "retained_h800_h3200_positive": int(source[800] >= 0.005 and source[3200] >= 0.005),
        "h3200_positive": int(source[3200] >= 0.005),
        "uses_future_for_direction": 0,
        "future_source_used_as_audit_label_only": 1,
        "execution_status": "measured",
        "probe_wall_ms": (time.perf_counter() - start) * 1000.0,
        "blocker": "",
    }


def _score_probe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [(str(r.get("dataset", "")), str(r.get("seed", ""))) for r in rows]
    feature_keys = [
        "source_washout_retention_ratio",
        "source_retention_control_gap",
        "source_B2_control_gap_after_washout",
        "source_B3_control_gap_after_washout",
        "source_param_radial_cosine",
    ]
    z = {key: _zscore([_f(r.get(key)) for r in rows], groups) for key in feature_keys}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        score = (
            0.40 * z["source_washout_retention_ratio"][i]
            + 0.35 * z["source_retention_control_gap"][i]
            + 0.30 * z["source_B2_control_gap_after_washout"][i]
            + 0.20 * z["source_B3_control_gap_after_washout"][i]
            - 0.15 * z["source_param_radial_cosine"][i]
        )
        train_gate = int(
            _f(row.get("source_retention_control_gap"), -999.0) >= -0.05
            and _f(row.get("source_B2_control_gap_after_washout"), -999.0) >= -0.001
            and _f(row.get("source_B3_control_gap_after_washout"), -999.0) >= -0.002
            and _f(row.get("source_washout_retention_ratio"), -999.0) >= 0.02
        )
        item = dict(row)
        item["COW_score"] = score
        item["COW_train_only_certificate_pass"] = train_gate
        item["certificate_family"] = "C-O7_counterfactual_washout_persistence"
        out.append(item)
    return sorted(out, key=lambda r: _f(r.get("COW_score"), -999.0), reverse=True)


def _summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    scores = [_f(r.get("COW_score")) for r in rows]
    labels = [int_flag(r.get("official_early_chain_h100_h400_h800_positive")) for r in rows]
    retained = [int_flag(r.get("retained_h800_h3200_positive")) for r in rows]
    h3200 = [int_flag(r.get("h3200_positive")) for r in rows]
    controls = [0 for _r in rows]
    p20, r20, control_frac = _precision_topk(scores, labels, controls, top_k)
    blockers: list[str] = []
    if sum(labels) == 0:
        blockers.append("no_positive_official_early_chain_label_rows")
    if _f(_auc(scores, labels), -1.0) < 0.75:
        blockers.append("auc_official_early_chain_gate")
    if _f(p20, -1.0) < 0.50:
        blockers.append("precision_top20_gate")
    return {
        "certificate_family": "C-O7_counterfactual_washout_persistence",
        "rows": len(rows),
        "positive_official_early_chain_rows": sum(labels),
        "positive_h3200_rows": sum(h3200),
        "positive_retained_h800_h3200_rows": sum(retained),
        "AUC_predict_official_early_chain": _auc(scores, labels),
        "AUC_predict_h3200_positive": _auc(scores, h3200),
        "AUC_predict_retained_h800_h3200_positive": _auc(scores, retained),
        "precision_at_top20": p20,
        "recall_at_top20": r20,
        "control_equivalent_fraction": control_frac,
        "COW_train_only_certificate_pass_rows": sum(int_flag(r.get("COW_train_only_certificate_pass")) for r in rows),
        "COW_official_observer_pass": int(not blockers),
        "blocker": ";".join(dict.fromkeys(blockers)),
    }


def _train_fresh(args: argparse.Namespace, source_row: dict[str, Any], run_kind: str) -> dict[str, Any]:
    device = resolve_device(args.device)
    dataset = str(source_row.get("dataset", "MNIST"))
    seed = int(float(source_row.get("seed", 0) or 0))
    local = deepcopy(args)
    local.basis_repair_variant = "R0-current"
    x_train, y_train, x_val, y_val = load_dataset(dataset, Path(args.data_root), int(args.train_size), int(args.val_size), seed, device, int(args.input_size))
    train_seed = int(float(source_row.get("train_seed") or 0))
    model = carrier_model("MLP", x_train, train_seed, local, device)
    opt_adam = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    opt_sgd = torch.optim.SGD(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(228_817 + train_seed + sum(ord(c) for c in run_kind))
    batch = min(int(args.batch_size), int(x_train.shape[0]))
    mechanism = str(source_row.get("mechanism", ""))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, float]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    injection_count = 0
    injection_norm_sum = 0.0
    for step in range(1, int(args.steps) + 1):
        idx = torch.randint(0, int(x_train.shape[0]), (batch,), generator=gen, device=device)
        xb = x_train[idx]
        yb = y_train[idx]
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        if run_kind == "CTRL-AdamW":
            opt_adam.step()
        elif run_kind == "CTRL-SGD":
            opt_sgd.step()
        elif run_kind == "CTRL-NoOp":
            pass
        else:
            opt_adam.step()
            if step % max(1, int(args.refresh_interval)) == 0:
                raw = make_update(model, mechanism, xb, yb, seed=train_seed + step)
                update = raw if run_kind == "COW-AdamWPlusPeriodicPersistentFU" else _random_like(raw, gen)
                injection_count += 1
                injection_norm_sum += float(torch.linalg.vector_norm(update.tensor.detach().float()).item())
                apply_update(model, update, lr=float(args.fu_lr))
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    out: dict[str, Any] = {
        "run_kind": run_kind,
        "certificate_family": "C-O7_counterfactual_washout_persistence",
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "COW_score": source_row.get("COW_score", ""),
        "refresh_interval": int(args.refresh_interval),
        "injection_count": injection_count,
        "injection_norm_sum": injection_norm_sum,
        "execution_status": "measured",
    }
    for h in HORIZONS:
        metrics = traces.get(h, {})
        for key, value in metrics.items():
            out[f"{key}_h{h}"] = value
    return out


def _fresh_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
            val = _f(item.get(f"val_loss_h{h}"))
            best = controls.get(key, {}).get(h, float("nan"))
            item[f"best_control_val_loss_h{h}"] = best if math.isfinite(best) else ""
            item[f"source_vs_best_control_h{h}"] = best - val if math.isfinite(best) and math.isfinite(val) else ""
        enriched.append(item)
    candidates = [r for r in enriched if str(r.get("run_kind")) == "COW-AdamWPlusPeriodicPersistentFU"]
    summary: list[dict[str, Any]] = []
    for row in candidates:
        debt_ok = int(_f(row.get("CEp99_h3200"), 0.0) <= 20.0 and _f(row.get("ECE_h3200"), 0.0) <= 1.0 and _f(row.get("Brier_h3200"), 0.0) <= 1.0)
        source_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        summary.append(
            {
                "v21_id": row.get("v21_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "COW_score": row.get("COW_score", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h2400": row.get("source_vs_best_control_h2400", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "injection_count": row.get("injection_count", ""),
                "injection_norm_sum": row.get("injection_norm_sum", ""),
                "debt_not_exploded": debt_ok,
                "COW_fresh_C3_source_washout_pass": int(source_pass and debt_ok),
                "official_C3_pass": 0,
                "blocker": "" if source_pass and debt_ok else "source_horizon_or_debt_gate_failed",
            }
        )
    positives = {h: sum(1 for r in summary if _f(r.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005) for h in HORIZONS}
    route = {
        "fresh_rows": len(summary),
        "fresh_C3_source_washout_pass_rows": sum(int_flag(r.get("COW_fresh_C3_source_washout_pass")) for r in summary),
        "official_C3_pass_rows": 0,
        "row_positive_count_h100": positives[100],
        "row_positive_count_h400": positives[400],
        "row_positive_count_h800": positives[800],
        "row_positive_count_h1600": positives[1600],
        "row_positive_count_h3200": positives[3200],
    }
    return enriched, {"summary": summary, "route": route}


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    prev_route = read_json(out_dir / "v22_08_post_nogo_source_state_route.json")
    rows = _candidate_rows(args)
    raw: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        try:
            raw.append(_probe_one(args, row, idx))
        except Exception as exc:
            raw.append(
                {
                    "job_order": idx,
                    "v21_id": row.get("v21_id", ""),
                    "mechanism": row.get("mechanism", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "execution_status": f"blocked:{type(exc).__name__}",
                    "blocker": str(exc)[:500],
                }
            )
    measured = [r for r in raw if str(r.get("execution_status")) == "measured"]
    scored = _score_probe(measured)
    blocked = [r for r in raw if str(r.get("execution_status")) != "measured"]
    matrix = scored + blocked
    summary = _summary(scored, 20)
    selected = scored[: max(0, int(args.fresh_top_k))]
    fresh_rows: list[dict[str, Any]] = []
    for row in selected:
        for run_kind in [
            "COW-AdamWPlusPeriodicPersistentFU",
            "CTRL-AdamW",
            "CTRL-SGD",
            "CTRL-NoOp",
            "CTRL-RandomMatchedPeriodicFU",
        ]:
            fresh_rows.append(_train_fresh(args, row, run_kind))
    fresh_enriched, fresh_pack = _fresh_summary(fresh_rows)
    fresh_summary = fresh_pack["summary"]
    fresh_route = fresh_pack["route"]
    route = {
        "route": "PostNoGoCounterfactualWashoutFreshC3Opened" if int(fresh_route["fresh_C3_source_washout_pass_rows"]) else "PostNoGoCounterfactualWashoutBlocked",
        "previous_post_nogo_route": prev_route.get("route", ""),
        "certificate_family": "C-O7_counterfactual_washout_persistence",
        "probe_rows": len(matrix),
        "probe_measured_rows": len(scored),
        "COW_train_only_certificate_pass_rows": summary.get("COW_train_only_certificate_pass_rows", 0),
        "COW_official_observer_pass": summary.get("COW_official_observer_pass", 0),
        "fresh_C3_source_washout_pass_rows": fresh_route["fresh_C3_source_washout_pass_rows"],
        "official_C3_pass_rows": 0,
        "promotion_allowed": 0,
        "blocker": "official_observer_gate_or_fresh_C3_source_horizon_failed",
        "next_codex_action": "C-O7 failed under counterfactual washout; require a source observability principle outside C-O1..C-O7 or external theoretical input",
    }
    write_rows(out_dir / "v22_08_post_nogo_counterfactual_washout_matrix.csv", matrix)
    write_rows(out_dir / "v22_08_post_nogo_counterfactual_washout_summary.csv", [summary])
    write_rows(out_dir / "v22_08_post_nogo_counterfactual_washout_selected_candidates.csv", selected)
    write_rows(out_dir / "v22_08_post_nogo_counterfactual_washout_fresh_c3_matrix.csv", fresh_enriched)
    write_rows(out_dir / "v22_08_post_nogo_counterfactual_washout_fresh_c3_summary.csv", fresh_summary)
    write_json(out_dir / "v22_08_post_nogo_counterfactual_washout_route.json", route)
    simple_svg(out_dir / "figures/v22_08_post_nogo_cow_score.svg", "v22.08 post-no-go COW score", scored, "COW_score")
    simple_svg(out_dir / "figures/v22_08_post_nogo_cow_h3200.svg", "v22.08 post-no-go COW h3200", fresh_summary, "source_vs_best_control_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_post_nogo_counterfactual_washout.py --source-dir {args.source_dir} --device {args.device} --fresh-top-k {int(args.fresh_top_k)} --steps {int(args.steps)} --washout-steps {int(args.washout_steps)} --refresh-interval {int(args.refresh_interval)} --out-dir {out_dir}",
        status="completed",
        note=f"probe_rows={len(matrix)} train_only_pass={summary.get('COW_train_only_certificate_pass_rows', 0)} fresh_C3_pass={fresh_route['fresh_C3_source_washout_pass_rows']} route={route['route']}",
    )


if __name__ == "__main__":
    main()
