#!/usr/bin/env python3
"""v22.09 post-state row-orthogonal source transport certificate.

This is a Nora-inspired row-wise angular stability attempt. It projects a
train-only source update onto the tangent space of each weight row, suppressing
row-norm jitter while preserving angular transport. Fresh runs compare the
row-orthogonal source update against AdamW/SGD/NoOp/random matched controls.
Future horizon columns are audit labels only.
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

from dgkan.fu.core import UpdateTensor, apply_update, cosine, normalized_like  # noqa: E402
from dgkan.fu.mechanisms import make_update  # noqa: E402
from experiments.run_v17_common import carrier_model, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_08_post_nogo_crossfit_influence_certificate import (  # noqa: E402
    HORIZONS,
    _candidate_rows,
    _eval,
    _f,
    _loss_value,
    _margin_value,
    _random_like,
    _split_indices,
)
from experiments.run_v22_08_post_nogo_source_state_certificate import _auc, _precision_topk, _zscore  # noqa: E402
from experiments.run_v22_09_common import (  # noqa: E402
    PYTHON,
    V2206_COMBINED_SOURCE,
    append_exec,
    ensure_out,
    int_flag,
    read_json,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2206_COMBINED_SOURCE))
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--data-root", default="data")
    p.add_argument("--train-size", type=int, default=96)
    p.add_argument("--val-size", type=int, default=48)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-probe-jobs", type=int, default=144)
    p.add_argument("--fresh-top-k", type=int, default=2)
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--probe-lr", type=float, default=0.0001)
    p.add_argument("--refresh-interval", type=int, default=8)
    return p


def _signed_displacement(update: UpdateTensor) -> torch.Tensor:
    sign = -1.0 if update.sign_rule == "subtract" else 1.0
    return update.tensor.detach().float() * sign


def _row_tangent_vector(model: torch.nn.Module, vector: torch.Tensor) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    offset = 0
    for param in model.parameters():
        n = param.numel()
        u = vector[offset : offset + n].detach().float().reshape_as(param.detach())
        w = param.detach().float()
        offset += n
        if w.ndim < 2:
            chunks.append(torch.zeros_like(u).reshape(-1))
            continue
        rows = w.reshape(w.shape[0], -1)
        upd = u.reshape(w.shape[0], -1)
        denom = (rows * rows).sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        radial = ((upd * rows).sum(dim=1, keepdim=True) / denom) * rows
        tangent = upd - radial
        chunks.append(tangent.reshape_as(w).reshape(-1))
    return torch.cat(chunks).to(vector.device) if chunks else torch.zeros_like(vector)


def _row_norm_drift(model: torch.nn.Module, vector: torch.Tensor, reference: torch.Tensor, lr: float) -> float:
    clone = deepcopy(model)
    before: list[torch.Tensor] = []
    for p in model.parameters():
        if p.ndim >= 2:
            before.append(torch.linalg.vector_norm(p.detach().float().reshape(p.shape[0], -1), dim=1))
    apply_update(clone, _vector_update(vector, reference, "rost_row_norm_probe"), lr=lr)
    drifts: list[torch.Tensor] = []
    idx = 0
    for p in clone.parameters():
        if p.ndim >= 2:
            after = torch.linalg.vector_norm(p.detach().float().reshape(p.shape[0], -1), dim=1)
            base = before[idx].to(after.device)
            drifts.append(torch.abs(after - base) / base.clamp_min(1.0e-12))
            idx += 1
    if not drifts:
        return 0.0
    return float(torch.cat(drifts).mean().item())


def _vector_update(vector: torch.Tensor, reference: torch.Tensor, kind: str, diagnostics: dict[str, float] | None = None) -> UpdateTensor:
    return UpdateTensor(
        tensor=normalized_like(vector.detach().float(), reference.detach().float()),
        kind=kind,
        sign_rule="add",
        space="row_orthogonal_source_transport",
        source="train_only_row_orthogonal_source_transport_certificate",
        mechanism="ROST-RowOrthogonalSourceTransport",
        role="all",
        one_step_descent_claim=0,
        diagnostics=diagnostics or {},
    )


def _tiny_effects(
    model: torch.nn.Module,
    vector: torch.Tensor,
    reference: torch.Tensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    x_c: torch.Tensor,
    y_c: torch.Tensor,
    lr: float,
) -> dict[str, float]:
    base_b = _loss_value(model, x_b, y_b)
    base_c = _loss_value(model, x_c, y_c)
    base_b_margin = _margin_value(model, x_b, y_b)
    clone = deepcopy(model)
    apply_update(clone, _vector_update(vector, reference, "rost_tiny_effect"), lr=lr)
    return {
        "B_loss_gain": base_b - _loss_value(clone, x_b, y_b),
        "C_safety_loss_gain": base_c - _loss_value(clone, x_c, y_c),
        "B_margin_gain": _margin_value(clone, x_b, y_b) - base_b_margin,
        "row_norm_drift": _row_norm_drift(model, vector, reference, lr),
    }


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
    batch = min(int(args.batch_size), int(x_train.shape[0]) // 3 if int(x_train.shape[0]) >= 3 else int(x_train.shape[0]))
    x_a, y_a = x_train[:batch], y_train[:batch]
    x_b, y_b = x_train[batch : 2 * batch], y_train[batch : 2 * batch]
    x_c, y_c = x_train[2 * batch : 3 * batch], y_train[2 * batch : 3 * batch]
    if int(x_b.shape[0]) == 0:
        x_b, y_b = x_a, y_a
    if int(x_c.shape[0]) == 0:
        x_c, y_c = x_b, y_b
    gen = torch.Generator(device=device).manual_seed(441_251 + train_seed + job_order)
    mechanism = str(row.get("mechanism", ""))
    raw = make_update(model, mechanism, x_a, y_a, seed=train_seed + 251)
    corrupt = make_update(model, mechanism, x_a, (y_a + 1) % max(2, int(args.classes)), seed=train_seed + 252)
    source_vec = _signed_displacement(raw)
    source_tangent = _row_tangent_vector(model, source_vec)
    random_tangent = _row_tangent_vector(model, _random_like(source_vec, gen))
    sign_tangent = _row_tangent_vector(model, -source_vec)
    corrupt_tangent = _row_tangent_vector(model, _signed_displacement(corrupt))
    source_eff = _tiny_effects(model, source_tangent, source_vec, x_b, y_b, x_c, y_c, float(args.probe_lr))
    random_eff = _tiny_effects(model, random_tangent, source_vec, x_b, y_b, x_c, y_c, float(args.probe_lr))
    sign_eff = _tiny_effects(model, sign_tangent, source_vec, x_b, y_b, x_c, y_c, float(args.probe_lr))
    corrupt_eff = _tiny_effects(model, corrupt_tangent, source_vec, x_b, y_b, x_c, y_c, float(args.probe_lr))
    control_b = max(random_eff["B_loss_gain"], sign_eff["B_loss_gain"], corrupt_eff["B_loss_gain"])
    control_c = max(random_eff["C_safety_loss_gain"], sign_eff["C_safety_loss_gain"], corrupt_eff["C_safety_loss_gain"])
    control_margin = max(random_eff["B_margin_gain"], sign_eff["B_margin_gain"], corrupt_eff["B_margin_gain"])
    control_drift = min(random_eff["row_norm_drift"], sign_eff["row_norm_drift"], corrupt_eff["row_norm_drift"])
    source_norm = float(torch.linalg.vector_norm(source_vec).item())
    tangent_norm = float(torch.linalg.vector_norm(source_tangent).item())
    fut = {h: _f(row.get(f"source_h{h}"), -999.0) for h in [100, 400, 800, 1600, 3200]}
    return {
        "job_order": job_order,
        "v21_id": row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "source_row_orthogonal_B_loss_gain": source_eff["B_loss_gain"],
        "source_row_orthogonal_C_safety_gain": source_eff["C_safety_loss_gain"],
        "B_row_orthogonal_control_gap": source_eff["B_loss_gain"] - control_b,
        "C_row_orthogonal_control_gap": source_eff["C_safety_loss_gain"] - control_c,
        "source_row_orthogonal_B_margin_gain": source_eff["B_margin_gain"],
        "B_margin_control_gap": source_eff["B_margin_gain"] - control_margin,
        "row_norm_drift": source_eff["row_norm_drift"],
        "row_norm_drift_control_gap": control_drift - source_eff["row_norm_drift"],
        "row_tangential_energy_fraction": tangent_norm / max(1.0e-12, source_norm),
        "row_tangent_source_cosine": cosine(source_vec, source_tangent),
        "future_audit_source_h100": row.get("source_h100", ""),
        "future_audit_source_h400": row.get("source_h400", ""),
        "future_audit_source_h800": row.get("source_h800", ""),
        "future_audit_source_h1600": row.get("source_h1600", ""),
        "future_audit_source_h3200": row.get("source_h3200", ""),
        "official_early_chain_h100_h400_h800_positive": int(fut[100] >= 0.005 and fut[400] >= 0.005 and fut[800] >= 0.005),
        "retained_h800_h3200_positive": int(fut[800] >= 0.005 and fut[3200] >= 0.005),
        "h3200_positive": int(fut[3200] >= 0.005),
        "uses_future_for_direction": 0,
        "future_source_used_as_audit_label_only": 1,
        "execution_status": "measured",
        "probe_wall_ms": (time.perf_counter() - start) * 1000.0,
        "blocker": "",
    }


def _score_probe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [(str(r.get("dataset", "")), str(r.get("seed", ""))) for r in rows]
    feature_keys = [
        "source_row_orthogonal_B_loss_gain",
        "source_row_orthogonal_C_safety_gain",
        "B_row_orthogonal_control_gap",
        "C_row_orthogonal_control_gap",
        "source_row_orthogonal_B_margin_gain",
        "B_margin_control_gap",
        "row_norm_drift_control_gap",
        "row_tangential_energy_fraction",
        "row_tangent_source_cosine",
    ]
    z = {key: _zscore([_f(r.get(key)) for r in rows], groups) for key in feature_keys}
    out = []
    for i, row in enumerate(rows):
        score = (
            0.30 * z["source_row_orthogonal_B_loss_gain"][i]
            + 0.25 * z["source_row_orthogonal_C_safety_gain"][i]
            + 0.30 * z["B_row_orthogonal_control_gap"][i]
            + 0.20 * z["C_row_orthogonal_control_gap"][i]
            + 0.15 * z["source_row_orthogonal_B_margin_gain"][i]
            + 0.15 * z["B_margin_control_gap"][i]
            + 0.20 * z["row_norm_drift_control_gap"][i]
            + 0.25 * z["row_tangential_energy_fraction"][i]
            + 0.20 * z["row_tangent_source_cosine"][i]
        )
        train_gate = int(
            _f(row.get("source_row_orthogonal_B_loss_gain"), -999.0) > 0.0
            and _f(row.get("B_row_orthogonal_control_gap"), -999.0) >= -1.0e-3
            and _f(row.get("C_row_orthogonal_control_gap"), -999.0) >= -1.0e-3
            and _f(row.get("row_norm_drift"), 999.0) <= 1.0e-3
            and _f(row.get("row_tangential_energy_fraction"), 0.0) >= 0.05
        )
        item = dict(row)
        item["ROST_score"] = score
        item["ROST_train_only_certificate_pass"] = train_gate
        item["certificate_family"] = "ROST_Row_Orthogonal_Source_Transport_Certificate"
        out.append(item)
    return sorted(out, key=lambda r: _f(r.get("ROST_score"), -999.0), reverse=True)


def _summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    scores = [_f(r.get("ROST_score")) for r in rows]
    labels = [int_flag(r.get("official_early_chain_h100_h400_h800_positive")) for r in rows]
    retained = [int_flag(r.get("retained_h800_h3200_positive")) for r in rows]
    h3200 = [int_flag(r.get("h3200_positive")) for r in rows]
    p20, r20, control_frac = _precision_topk(scores, labels, [0 for _ in rows], top_k)
    blockers = []
    if sum(labels) == 0:
        blockers.append("no_positive_official_early_chain_label_rows")
    if _f(_auc(scores, labels), -1.0) < 0.75:
        blockers.append("auc_official_early_chain_gate")
    if _f(p20, -1.0) < 0.50:
        blockers.append("precision_top20_gate")
    return {
        "certificate_family": "ROST_Row_Orthogonal_Source_Transport_Certificate",
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
        "ROST_train_only_certificate_pass_rows": sum(int_flag(r.get("ROST_train_only_certificate_pass")) for r in rows),
        "ROST_official_observer_pass": int(not blockers),
        "blocker": ";".join(dict.fromkeys(blockers)),
    }


def _rost_update(model: torch.nn.Module, raw: UpdateTensor, gen: torch.Generator | None = None) -> UpdateTensor:
    source_vec = _signed_displacement(raw)
    base = _random_like(source_vec, gen) if gen is not None else source_vec
    tangent = _row_tangent_vector(model, base)
    source_norm = max(1.0e-12, float(torch.linalg.vector_norm(source_vec).item()))
    diagnostics = {
        "row_tangential_energy_fraction": float(torch.linalg.vector_norm(tangent).item()) / source_norm,
        "row_tangent_source_cosine": 0.0 if gen is not None else cosine(source_vec, tangent),
    }
    return _vector_update(tangent, raw.tensor.detach().float(), "row_orthogonal_source_transport", diagnostics)


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
    gen = torch.Generator(device=device).manual_seed(441_907 + train_seed + sum(ord(c) for c in run_kind))
    batch = min(int(args.batch_size), max(1, int(x_train.shape[0]) // 2))
    mechanism = str(source_row.get("mechanism", ""))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, float]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    injection_count = 0
    injection_norm_sum = 0.0
    tangent_frac_sum = 0.0
    tangent_cos_sum = 0.0
    row_loss_gain_sum = 0.0
    for step in range(1, int(args.steps) + 1):
        idx_a, idx_b = _split_indices(int(x_train.shape[0]), batch, gen, device)
        x_a, y_a = x_train[idx_a], y_train[idx_a]
        x_b, y_b = x_train[idx_b], y_train[idx_b]
        opt_adam.zero_grad(set_to_none=True)
        opt_sgd.zero_grad(set_to_none=True)
        F.cross_entropy(model(x_a).float(), y_a).backward()
        if run_kind == "CTRL-AdamW":
            opt_adam.step()
        elif run_kind == "CTRL-SGD":
            opt_sgd.step()
        elif run_kind == "CTRL-NoOp":
            pass
        else:
            opt_adam.step()
            if step % max(1, int(args.refresh_interval)) == 0:
                before = _loss_value(model, x_b, y_b)
                raw = make_update(model, mechanism, x_a, y_a, seed=train_seed + step)
                update = _rost_update(model, raw, gen if run_kind == "CTRL-RandomMatchedRowOrthogonalFU" else None)
                injection_count += 1
                injection_norm_sum += float(torch.linalg.vector_norm(update.tensor.detach().float()).item())
                tangent_frac_sum += float(update.diagnostics.get("row_tangential_energy_fraction", 0.0) if update.diagnostics else 0.0)
                tangent_cos_sum += float(update.diagnostics.get("row_tangent_source_cosine", 0.0) if update.diagnostics else 0.0)
                apply_update(model, update, lr=float(args.fu_lr))
                row_loss_gain_sum += before - _loss_value(model, x_b, y_b)
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    out: dict[str, Any] = {
        "run_kind": run_kind,
        "certificate_family": "ROST_Row_Orthogonal_Source_Transport_Certificate",
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "ROST_score": source_row.get("ROST_score", ""),
        "refresh_interval": int(args.refresh_interval),
        "injection_count": injection_count,
        "injection_norm_sum": injection_norm_sum,
        "row_tangential_energy_fraction_mean": tangent_frac_sum / max(1, injection_count),
        "row_tangent_source_cosine_mean": tangent_cos_sum / max(1, injection_count),
        "row_orthogonal_loss_gain_mean": row_loss_gain_sum / max(1, injection_count),
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
    enriched = []
    for row in rows:
        item = dict(row)
        key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("train_seed")))
        for h in HORIZONS:
            val = _f(item.get(f"val_loss_h{h}"))
            best = controls.get(key, {}).get(h, float("nan"))
            item[f"best_control_val_loss_h{h}"] = best if math.isfinite(best) else ""
            item[f"source_vs_best_control_h{h}"] = best - val if math.isfinite(best) and math.isfinite(val) else ""
        enriched.append(item)
    candidates = [r for r in enriched if str(r.get("run_kind")) == "ROST-AdamWPlusRowOrthogonalFU"]
    summary = []
    for row in candidates:
        debt_ok = int(_f(row.get("CEp99_h3200"), 0.0) <= 20.0 and _f(row.get("ECE_h3200"), 0.0) <= 1.0 and _f(row.get("Brier_h3200"), 0.0) <= 1.0)
        source_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        summary.append(
            {
                "v21_id": row.get("v21_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "ROST_score": row.get("ROST_score", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h2400": row.get("source_vs_best_control_h2400", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "row_tangential_energy_fraction_mean": row.get("row_tangential_energy_fraction_mean", ""),
                "row_tangent_source_cosine_mean": row.get("row_tangent_source_cosine_mean", ""),
                "row_orthogonal_loss_gain_mean": row.get("row_orthogonal_loss_gain_mean", ""),
                "injection_count": row.get("injection_count", ""),
                "injection_norm_sum": row.get("injection_norm_sum", ""),
                "debt_not_exploded": debt_ok,
                "ROST_fresh_C3_row_orthogonal_pass": int(source_pass and debt_ok),
                "official_C3_pass": 0,
                "blocker": "" if source_pass and debt_ok else "source_horizon_or_debt_gate_failed",
            }
        )
    return enriched, {"summary": summary, "route": {"fresh_rows": len(summary), "fresh_C3_row_orthogonal_pass_rows": sum(int_flag(r.get("ROST_fresh_C3_row_orthogonal_pass")) for r in summary), "official_C3_pass_rows": 0}}


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    prev_route = read_json(out_dir / "v22_09_final_route.json")
    rows = _candidate_rows(args)
    raw = []
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
    summary = _summary(scored, 20)
    eligible = [r for r in scored if int_flag(r.get("ROST_train_only_certificate_pass"))]
    selected = (eligible if eligible else scored)[: max(0, int(args.fresh_top_k))]
    fresh_rows = []
    for row in selected:
        for run_kind in [
            "ROST-AdamWPlusRowOrthogonalFU",
            "CTRL-AdamW",
            "CTRL-SGD",
            "CTRL-NoOp",
            "CTRL-RandomMatchedRowOrthogonalFU",
        ]:
            fresh_rows.append(_train_fresh(args, row, run_kind))
    fresh_enriched, fresh_pack = _fresh_summary(fresh_rows)
    fresh_summary = fresh_pack["summary"]
    fresh_route = fresh_pack["route"]
    blockers = []
    if not int_flag(summary.get("ROST_official_observer_pass")):
        blockers.append("official_observer_gate_failed")
    if not int(fresh_route["fresh_C3_row_orthogonal_pass_rows"]):
        blockers.append("fresh_C3_source_horizon_failed")
    blockers.append("official_C3_gate_not_claimed")
    route = {
        "route": "PostRowOrthogonalFreshC3Opened" if int(fresh_route["fresh_C3_row_orthogonal_pass_rows"]) else "PostRowOrthogonalBlocked",
        "previous_route": prev_route.get("route", ""),
        "certificate_family": "ROST_Row_Orthogonal_Source_Transport_Certificate",
        "probe_rows": len(scored) + len(blocked),
        "probe_measured_rows": len(scored),
        "ROST_train_only_certificate_pass_rows": summary.get("ROST_train_only_certificate_pass_rows", 0),
        "ROST_official_observer_pass": summary.get("ROST_official_observer_pass", 0),
        "fresh_C3_row_orthogonal_pass_rows": fresh_route["fresh_C3_row_orthogonal_pass_rows"],
        "official_C3_pass_rows": 0,
        "promotion_allowed": 0,
        "blocker": ";".join(dict.fromkeys(blockers)),
        "next_codex_action": "ROST failed fresh/official gate; record v22.09 local boundary or change principle again",
    }
    write_rows(out_dir / "v22_09_row_orthogonal_source_transport_certificate.csv", scored + blocked)
    write_rows(out_dir / "v22_09_row_orthogonal_source_transport_summary.csv", [summary])
    write_rows(out_dir / "v22_09_row_orthogonal_source_transport_selected_candidates.csv", selected)
    write_rows(out_dir / "v22_09_row_orthogonal_source_transport_fresh_c3_matrix.csv", fresh_enriched)
    write_rows(out_dir / "v22_09_row_orthogonal_source_transport_fresh_c3_summary.csv", fresh_summary)
    write_json(out_dir / "v22_09_row_orthogonal_source_transport_route.json", route)
    simple_svg(out_dir / "figures/v22_09_row_orthogonal_source_transport_certificate.svg", "v22.09 ROST certificate", scored, "ROST_score")
    simple_svg(out_dir / "figures/v22_09_row_orthogonal_source_transport_fresh_h3200.svg", "v22.09 ROST fresh h3200", fresh_summary, "source_vs_best_control_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_post_row_orthogonal_source_transport_certificate.py --source-dir {args.source_dir} --device {args.device} --max-probe-jobs {int(args.max_probe_jobs)} --fresh-top-k {int(args.fresh_top_k)} --steps {int(args.steps)} --refresh-interval {int(args.refresh_interval)} --out-dir {out_dir}",
        status="completed",
        note=f"probe_rows={len(scored)+len(blocked)} measured={len(scored)} train_only_pass={summary.get('ROST_train_only_certificate_pass_rows', 0)} fresh_pass={fresh_route['fresh_C3_row_orthogonal_pass_rows']} route={route['route']}",
    )


if __name__ == "__main__":
    main()
