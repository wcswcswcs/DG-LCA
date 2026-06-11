#!/usr/bin/env python3
"""v22.09 post-boundary optimizer-state holonomy certificate.

This follow-up is not a C-O13 threshold tweak. It tests whether a train-only
source boundary remains coherent when embedded in short AdamW stateful flow
cycles over disjoint train splits. Fresh runs inject a source/momentum mixed
direction and compare against AdamW/SGD/NoOp/random matched controls.
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
    _unit,
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
    p.add_argument("--boundary-lr", type=float, default=0.0001)
    p.add_argument("--micro-horizon", type=int, default=6)
    p.add_argument("--refresh-interval", type=int, default=8)
    return p


def _signed_displacement(update: UpdateTensor) -> torch.Tensor:
    sign = -1.0 if update.sign_rule == "subtract" else 1.0
    return update.tensor.detach().float() * sign


def _vector_update(vector: torch.Tensor, reference: torch.Tensor, kind: str, diagnostics: dict[str, float] | None = None) -> UpdateTensor:
    return UpdateTensor(
        tensor=normalized_like(vector.detach().float(), reference.detach().float()),
        kind=kind,
        sign_rule="add",
        space="optimizer_state_holonomy",
        source="train_only_optimizer_state_holonomy_certificate",
        mechanism="OSH-OptimizerStateHolonomy",
        role="all",
        one_step_descent_claim=0,
        diagnostics=diagnostics or {},
    )


def _adam_state_vector(model: torch.nn.Module, opt: torch.optim.Optimizer) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for p in model.parameters():
        state = opt.state.get(p, {})
        exp_avg = state.get("exp_avg")
        if exp_avg is None:
            pieces.append(torch.zeros_like(p.detach()).reshape(-1).float())
        else:
            pieces.append(exp_avg.detach().reshape(-1).float())
    return torch.cat(pieces) if pieces else torch.empty(0)


def _run_stateful_sequence(
    model: torch.nn.Module,
    seq: list[tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
    *,
    vector: torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, torch.Tensor]:
    clone = deepcopy(model)
    if vector is not None and reference is not None:
        apply_update(clone, _vector_update(vector, reference, "osh_boundary_seed"), lr=float(args.boundary_lr))
    opt = torch.optim.AdamW(clone.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    horizon = max(1, int(args.micro_horizon))
    for i in range(horizon):
        x, y = seq[i % len(seq)]
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(clone(x).float(), y).backward()
        opt.step()
    clone.zero_grad(set_to_none=True)
    return clone, _adam_state_vector(clone, opt)


def _cycle_metrics(
    model: torch.nn.Module,
    vector: torch.Tensor,
    reference: torch.Tensor,
    x_a: torch.Tensor,
    y_a: torch.Tensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    x_c: torch.Tensor,
    y_c: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    seq_ab = [(x_a, y_a), (x_b, y_b)]
    seq_ba = [(x_b, y_b), (x_a, y_a)]
    base_ab, base_mom_ab = _run_stateful_sequence(model, seq_ab, args)
    base_ba, base_mom_ba = _run_stateful_sequence(model, seq_ba, args)
    src_ab, src_mom_ab = _run_stateful_sequence(model, seq_ab, args, vector=vector, reference=reference)
    src_ba, src_mom_ba = _run_stateful_sequence(model, seq_ba, args, vector=vector, reference=reference)
    base_b_loss = 0.5 * (_loss_value(base_ab, x_b, y_b) + _loss_value(base_ba, x_b, y_b))
    base_c_loss = 0.5 * (_loss_value(base_ab, x_c, y_c) + _loss_value(base_ba, x_c, y_c))
    base_b_margin = 0.5 * (_margin_value(base_ab, x_b, y_b) + _margin_value(base_ba, x_b, y_b))
    src_b_loss = 0.5 * (_loss_value(src_ab, x_b, y_b) + _loss_value(src_ba, x_b, y_b))
    src_c_loss = 0.5 * (_loss_value(src_ab, x_c, y_c) + _loss_value(src_ba, x_c, y_c))
    src_b_margin = 0.5 * (_margin_value(src_ab, x_b, y_b) + _margin_value(src_ba, x_b, y_b))
    holonomy = torch.linalg.vector_norm(flat_params(src_ab) - flat_params(src_ba)).item()
    base_holonomy = torch.linalg.vector_norm(flat_params(base_ab) - flat_params(base_ba)).item()
    momentum = -0.5 * (src_mom_ab + src_mom_ba)
    base_momentum = -0.5 * (base_mom_ab + base_mom_ba)
    ref_norm = max(1.0e-12, float(torch.linalg.vector_norm(reference).item()))
    return {
        "B_cycle_loss_gain": base_b_loss - src_b_loss,
        "C_cycle_loss_gain": base_c_loss - src_c_loss,
        "B_cycle_margin_gain": src_b_margin - base_b_margin,
        "stateful_holonomy_norm": holonomy,
        "base_holonomy_norm": base_holonomy,
        "stateful_holonomy_ratio": holonomy / ref_norm,
        "source_momentum_cosine": cosine(vector, momentum),
        "momentum_control_cosine": cosine(vector, base_momentum),
        "momentum_norm_ratio": float(torch.linalg.vector_norm(momentum).item()) / ref_norm,
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
    gen = torch.Generator(device=device).manual_seed(331_337 + train_seed + job_order)
    mechanism = str(row.get("mechanism", ""))
    raw = make_update(model, mechanism, x_a, y_a, seed=train_seed + 337)
    corrupt = make_update(model, mechanism, x_a, (y_a + 1) % max(2, int(args.classes)), seed=train_seed + 338)
    source_vec = _signed_displacement(raw)
    corrupt_vec = _signed_displacement(corrupt)
    random_vec = _random_like(source_vec, gen)
    sign_vec = -source_vec
    source = _cycle_metrics(model, source_vec, source_vec, x_a, y_a, x_b, y_b, x_c, y_c, args)
    random = _cycle_metrics(model, random_vec, source_vec, x_a, y_a, x_b, y_b, x_c, y_c, args)
    sign = _cycle_metrics(model, sign_vec, source_vec, x_a, y_a, x_b, y_b, x_c, y_c, args)
    corrupt_m = _cycle_metrics(model, corrupt_vec, source_vec, x_a, y_a, x_b, y_b, x_c, y_c, args)
    control_b = max(random["B_cycle_loss_gain"], sign["B_cycle_loss_gain"], corrupt_m["B_cycle_loss_gain"])
    control_c = max(random["C_cycle_loss_gain"], sign["C_cycle_loss_gain"], corrupt_m["C_cycle_loss_gain"])
    control_margin = max(random["B_cycle_margin_gain"], sign["B_cycle_margin_gain"], corrupt_m["B_cycle_margin_gain"])
    control_holonomy = min(random["stateful_holonomy_ratio"], sign["stateful_holonomy_ratio"], corrupt_m["stateful_holonomy_ratio"])
    control_mom = max(random["source_momentum_cosine"], sign["source_momentum_cosine"], corrupt_m["source_momentum_cosine"])
    fut = {h: _f(row.get(f"source_h{h}"), -999.0) for h in [100, 400, 800, 1600, 3200]}
    return {
        "job_order": job_order,
        "v21_id": row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "source_stateful_B_loss_gain": source["B_cycle_loss_gain"],
        "source_stateful_C_safety_gain": source["C_cycle_loss_gain"],
        "B_stateful_control_gap": source["B_cycle_loss_gain"] - control_b,
        "C_stateful_control_gap": source["C_cycle_loss_gain"] - control_c,
        "source_stateful_B_margin_gain": source["B_cycle_margin_gain"],
        "B_margin_control_gap": source["B_cycle_margin_gain"] - control_margin,
        "stateful_holonomy_ratio": source["stateful_holonomy_ratio"],
        "stateful_holonomy_control_gap": control_holonomy - source["stateful_holonomy_ratio"],
        "base_holonomy_norm": source["base_holonomy_norm"],
        "source_momentum_cosine": source["source_momentum_cosine"],
        "momentum_control_gap": source["source_momentum_cosine"] - control_mom,
        "momentum_norm_ratio": source["momentum_norm_ratio"],
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
        "source_stateful_B_loss_gain",
        "source_stateful_C_safety_gain",
        "B_stateful_control_gap",
        "C_stateful_control_gap",
        "source_stateful_B_margin_gain",
        "B_margin_control_gap",
        "stateful_holonomy_control_gap",
        "source_momentum_cosine",
        "momentum_control_gap",
    ]
    z = {key: _zscore([_f(r.get(key)) for r in rows], groups) for key in feature_keys}
    out = []
    for i, row in enumerate(rows):
        score = (
            0.30 * z["source_stateful_B_loss_gain"][i]
            + 0.25 * z["source_stateful_C_safety_gain"][i]
            + 0.30 * z["B_stateful_control_gap"][i]
            + 0.20 * z["C_stateful_control_gap"][i]
            + 0.15 * z["source_stateful_B_margin_gain"][i]
            + 0.15 * z["B_margin_control_gap"][i]
            + 0.30 * z["stateful_holonomy_control_gap"][i]
            + 0.20 * z["source_momentum_cosine"][i]
            + 0.25 * z["momentum_control_gap"][i]
        )
        train_gate = int(
            _f(row.get("source_stateful_B_loss_gain"), -999.0) > 0.0
            and _f(row.get("B_stateful_control_gap"), -999.0) >= -1.0e-3
            and _f(row.get("C_stateful_control_gap"), -999.0) >= -1.0e-3
            and _f(row.get("stateful_holonomy_ratio"), 999.0) <= 0.01
            and _f(row.get("source_momentum_cosine"), -999.0) > -0.20
        )
        item = dict(row)
        item["OSH_score"] = score
        item["OSH_train_only_certificate_pass"] = train_gate
        item["certificate_family"] = "OSH_Optimizer_State_Holonomy_Certificate"
        out.append(item)
    return sorted(out, key=lambda r: _f(r.get("OSH_score"), -999.0), reverse=True)


def _summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    scores = [_f(r.get("OSH_score")) for r in rows]
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
        "certificate_family": "OSH_Optimizer_State_Holonomy_Certificate",
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
        "OSH_train_only_certificate_pass_rows": sum(int_flag(r.get("OSH_train_only_certificate_pass")) for r in rows),
        "OSH_official_observer_pass": int(not blockers),
        "blocker": ";".join(dict.fromkeys(blockers)),
    }


def _live_momentum_update(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    raw: UpdateTensor,
    gen: torch.Generator | None,
) -> UpdateTensor:
    source_vec = _signed_displacement(raw)
    if gen is not None:
        mixed = _random_like(source_vec, gen)
        mom_cos = 0.0
        mom_ratio = 0.0
    else:
        momentum = -_adam_state_vector(model, opt)
        mixed = normalized_like(0.60 * _unit(source_vec) + 0.40 * _unit(momentum), source_vec)
        mom_cos = cosine(source_vec, momentum)
        mom_ratio = float(torch.linalg.vector_norm(momentum).item()) / max(1.0e-12, float(torch.linalg.vector_norm(source_vec).item()))
    return _vector_update(mixed, raw.tensor.detach().float(), "optimizer_state_holonomy", {"source_momentum_cosine": mom_cos, "momentum_norm_ratio": mom_ratio})


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
    gen = torch.Generator(device=device).manual_seed(331_901 + train_seed + sum(ord(c) for c in run_kind))
    batch = min(int(args.batch_size), max(1, int(x_train.shape[0]) // 2))
    mechanism = str(source_row.get("mechanism", ""))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, float]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    injection_count = 0
    injection_norm_sum = 0.0
    momentum_cos_sum = 0.0
    momentum_ratio_sum = 0.0
    state_loss_gain_sum = 0.0
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
                update = _live_momentum_update(model, opt_adam, raw, gen if run_kind == "CTRL-RandomMatchedStateFU" else None)
                injection_count += 1
                injection_norm_sum += float(torch.linalg.vector_norm(update.tensor.detach().float()).item())
                momentum_cos_sum += float(update.diagnostics.get("source_momentum_cosine", 0.0) if update.diagnostics else 0.0)
                momentum_ratio_sum += float(update.diagnostics.get("momentum_norm_ratio", 0.0) if update.diagnostics else 0.0)
                apply_update(model, update, lr=float(args.fu_lr))
                state_loss_gain_sum += before - _loss_value(model, x_b, y_b)
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    out: dict[str, Any] = {
        "run_kind": run_kind,
        "certificate_family": "OSH_Optimizer_State_Holonomy_Certificate",
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "OSH_score": source_row.get("OSH_score", ""),
        "refresh_interval": int(args.refresh_interval),
        "micro_horizon": int(args.micro_horizon),
        "injection_count": injection_count,
        "injection_norm_sum": injection_norm_sum,
        "source_momentum_cosine_mean": momentum_cos_sum / max(1, injection_count),
        "momentum_norm_ratio_mean": momentum_ratio_sum / max(1, injection_count),
        "state_loss_gain_mean": state_loss_gain_sum / max(1, injection_count),
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
    candidates = [r for r in enriched if str(r.get("run_kind")) == "OSH-AdamWPlusStateHolonomyFU"]
    summary = []
    for row in candidates:
        debt_ok = int(_f(row.get("CEp99_h3200"), 0.0) <= 20.0 and _f(row.get("ECE_h3200"), 0.0) <= 1.0 and _f(row.get("Brier_h3200"), 0.0) <= 1.0)
        source_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        summary.append(
            {
                "v21_id": row.get("v21_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "OSH_score": row.get("OSH_score", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h2400": row.get("source_vs_best_control_h2400", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "source_momentum_cosine_mean": row.get("source_momentum_cosine_mean", ""),
                "momentum_norm_ratio_mean": row.get("momentum_norm_ratio_mean", ""),
                "state_loss_gain_mean": row.get("state_loss_gain_mean", ""),
                "injection_count": row.get("injection_count", ""),
                "injection_norm_sum": row.get("injection_norm_sum", ""),
                "debt_not_exploded": debt_ok,
                "OSH_fresh_C3_state_holonomy_pass": int(source_pass and debt_ok),
                "official_C3_pass": 0,
                "blocker": "" if source_pass and debt_ok else "source_horizon_or_debt_gate_failed",
            }
        )
    return enriched, {"summary": summary, "route": {"fresh_rows": len(summary), "fresh_C3_state_holonomy_pass_rows": sum(int_flag(r.get("OSH_fresh_C3_state_holonomy_pass")) for r in summary), "official_C3_pass_rows": 0}}


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
    eligible = [r for r in scored if int_flag(r.get("OSH_train_only_certificate_pass"))]
    selected = (eligible if eligible else scored)[: max(0, int(args.fresh_top_k))]
    fresh_rows = []
    for row in selected:
        for run_kind in [
            "OSH-AdamWPlusStateHolonomyFU",
            "CTRL-AdamW",
            "CTRL-SGD",
            "CTRL-NoOp",
            "CTRL-RandomMatchedStateFU",
        ]:
            fresh_rows.append(_train_fresh(args, row, run_kind))
    fresh_enriched, fresh_pack = _fresh_summary(fresh_rows)
    fresh_summary = fresh_pack["summary"]
    fresh_route = fresh_pack["route"]
    blockers = []
    if not int_flag(summary.get("OSH_official_observer_pass")):
        blockers.append("official_observer_gate_failed")
    if not int(fresh_route["fresh_C3_state_holonomy_pass_rows"]):
        blockers.append("fresh_C3_source_horizon_failed")
    blockers.append("official_C3_gate_not_claimed")
    route = {
        "route": "PostStateHolonomyFreshC3Opened" if int(fresh_route["fresh_C3_state_holonomy_pass_rows"]) else "PostStateHolonomyBlocked",
        "previous_route": prev_route.get("route", ""),
        "certificate_family": "OSH_Optimizer_State_Holonomy_Certificate",
        "probe_rows": len(scored) + len(blocked),
        "probe_measured_rows": len(scored),
        "OSH_train_only_certificate_pass_rows": summary.get("OSH_train_only_certificate_pass_rows", 0),
        "OSH_official_observer_pass": summary.get("OSH_official_observer_pass", 0),
        "fresh_C3_state_holonomy_pass_rows": fresh_route["fresh_C3_state_holonomy_pass_rows"],
        "official_C3_pass_rows": 0,
        "promotion_allowed": 0,
        "blocker": ";".join(dict.fromkeys(blockers)),
        "next_codex_action": "OSH failed fresh/official gate; next attempt must change principle or record boundary",
    }
    write_rows(out_dir / "v22_09_optimizer_state_holonomy_certificate.csv", scored + blocked)
    write_rows(out_dir / "v22_09_optimizer_state_holonomy_summary.csv", [summary])
    write_rows(out_dir / "v22_09_optimizer_state_holonomy_selected_candidates.csv", selected)
    write_rows(out_dir / "v22_09_optimizer_state_holonomy_fresh_c3_matrix.csv", fresh_enriched)
    write_rows(out_dir / "v22_09_optimizer_state_holonomy_fresh_c3_summary.csv", fresh_summary)
    write_json(out_dir / "v22_09_optimizer_state_holonomy_route.json", route)
    simple_svg(out_dir / "figures/v22_09_optimizer_state_holonomy_certificate.svg", "v22.09 OSH certificate", scored, "OSH_score")
    simple_svg(out_dir / "figures/v22_09_optimizer_state_holonomy_fresh_h3200.svg", "v22.09 OSH fresh h3200", fresh_summary, "source_vs_best_control_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_post_optimizer_state_holonomy_certificate.py --source-dir {args.source_dir} --device {args.device} --max-probe-jobs {int(args.max_probe_jobs)} --fresh-top-k {int(args.fresh_top_k)} --steps {int(args.steps)} --micro-horizon {int(args.micro_horizon)} --refresh-interval {int(args.refresh_interval)} --out-dir {out_dir}",
        status="completed",
        note=f"probe_rows={len(scored)+len(blocked)} measured={len(scored)} train_only_pass={summary.get('OSH_train_only_certificate_pass_rows', 0)} fresh_pass={fresh_route['fresh_C3_state_holonomy_pass_rows']} route={route['route']}",
    )


if __name__ == "__main__":
    main()
