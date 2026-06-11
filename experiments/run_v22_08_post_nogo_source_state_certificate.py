#!/usr/bin/env python3
"""v22.08 post-no-go source-state persistence certificate.

This is deliberately not a C-O1..C-O5 scale/cap/floor variant.  It tests a
new train-only principle: a retained source should appear as a persistent
slow source-state across legal train-stream micro-batches, and a fresh
source-state integration run should beat matched controls across horizons.
Future source labels are used only for audit columns.
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

from dgkan.fu.core import UpdateTensor, apply_update, cosine, flat_grad, flat_params, normalized_like  # noqa: E402
from dgkan.fu.mechanisms import CONTROL_MECHANISMS, make_update  # noqa: E402
from experiments.run_v17_common import carrier_model, classification_brier, classification_ece, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_07_metric_dynamics_fu import _filtered_matrix, _simulate_gains  # noqa: E402
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
    p.add_argument("--probe-batches", type=int, default=4)
    p.add_argument("--max-probe-jobs", type=int, default=0)
    p.add_argument("--fresh-top-k", type=int, default=2)
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--short-beta", type=float, default=0.50)
    p.add_argument("--long-beta", type=float, default=0.97)
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _unit(x: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x.detach().float())
    if float(norm.item()) <= 1.0e-12:
        return torch.zeros_like(x)
    return x / norm.clamp_min(1.0e-12)


def _signed_displacement(update: UpdateTensor) -> torch.Tensor:
    sign = -1.0 if update.sign_rule == "subtract" else 1.0
    return update.tensor.detach().float() * sign


def _rank(values: list[float]) -> list[float]:
    pairs = sorted((v, i) for i, v in enumerate(values))
    out = [0.0] * len(values)
    pos = 0
    while pos < len(pairs):
        end = pos + 1
        while end < len(pairs) and pairs[end][0] == pairs[pos][0]:
            end += 1
        avg = 0.5 * (pos + end - 1) + 1.0
        for _v, idx in pairs[pos:end]:
            out[idx] = avg
        pos = end
    return out


def _auc(scores: list[float], labels: list[int]) -> float | str:
    valid = [(s, y) for s, y in zip(scores, labels) if math.isfinite(s)]
    if not valid:
        return ""
    scores = [s for s, _y in valid]
    labels = [y for _s, y in valid]
    pos = sum(labels)
    neg = len(labels) - pos
    if pos <= 0 or neg <= 0:
        return ""
    ranks = _rank(scores)
    pos_rank = sum(r for r, y in zip(ranks, labels) if y)
    return (pos_rank - pos * (pos + 1) / 2.0) / float(pos * neg)


def _precision_topk(scores: list[float], labels: list[int], controls: list[int], k: int) -> tuple[float | str, float | str, float | str]:
    valid = [(i, s) for i, s in enumerate(scores) if math.isfinite(s)]
    if not valid:
        return "", "", ""
    order = [i for i, _s in sorted(valid, key=lambda item: item[1], reverse=True)[: min(k, len(valid))]]
    if not order:
        return "", "", ""
    hits = sum(labels[i] for i in order)
    total_pos = sum(labels)
    return hits / float(len(order)), hits / float(total_pos) if total_pos else "", sum(controls[i] for i in order) / float(len(order))


def _zscore(values: list[float], groups: list[tuple[str, str]]) -> list[float]:
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    for group in sorted(set(groups)):
        vals = [v for v, g in zip(values, groups) if g == group and math.isfinite(v)]
        if not vals:
            stats[group] = (0.0, 1.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        stats[group] = (mean, math.sqrt(var) if var > 1.0e-12 else 1.0)
    return [((_f(v, 0.0) - stats[g][0]) / stats[g][1]) for v, g in zip(values, groups)]


def _micro_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, count: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    n = int(x.shape[0])
    bs = min(int(batch_size), n)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(max(2, int(count))):
        if i == 0:
            idx = torch.arange(0, bs, device=x.device)
        elif i == 1:
            idx = torch.arange(max(0, n - bs), n, device=x.device)
        else:
            idx = torch.randint(0, n, (bs,), generator=gen, device=x.device)
        batches.append((x[idx], y[idx]))
    return batches


def _grad_displacement(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(xb).float(), yb).backward()
    return -flat_grad(model, xb.device).detach().float()


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
    out = []
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
    mechanism = str(row.get("mechanism", ""))
    batches = _micro_batches(x_train, y_train, int(args.batch_size), int(args.probe_batches), train_seed + 82_208)
    vectors: list[torch.Tensor] = []
    controls: list[torch.Tensor] = []
    ref_update: UpdateTensor | None = None
    for i, (xb, yb) in enumerate(batches):
        update = make_update(model, mechanism, xb, yb, seed=train_seed + 13 + i)
        if ref_update is None:
            ref_update = update
        vectors.append(_signed_displacement(update))
        controls.append(_grad_displacement(model, xb, yb))
        corrupt_y = (yb + 1) % max(2, int(args.classes))
        controls.append(_grad_displacement(model, xb, corrupt_y))
    if ref_update is None or not vectors:
        raise RuntimeError("no probe vectors generated")
    units = [_unit(v) for v in vectors]
    pair_cos: list[float] = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            pair_cos.append(cosine(units[i], units[j]))
    source_state = torch.stack(units).mean(dim=0)
    state_norm = float(torch.linalg.vector_norm(source_state).item())
    state_unit = _unit(source_state)
    control_cos = [abs(cosine(state_unit, _unit(c))) for c in controls if c.numel()]
    params = flat_params(model).detach().float()
    radial_fraction = abs(cosine(state_unit, _unit(params))) if params.numel() else 0.0
    state_update = UpdateTensor(
        tensor=normalized_like(state_unit, ref_update.tensor.detach()),
        kind="source_state_certificate",
        sign_rule="add",
        space="optimizer_source_state",
        source="train_stream_pathwise_source_state_persistence",
        mechanism="C-O6-PathwiseSourceStatePersistence",
        role="slow_source_state",
        one_step_descent_claim=0,
        diagnostics={"source_state_norm": state_norm},
    )
    gains = _simulate_gains(model, state_update, float(args.fu_lr), x_train[: min(int(args.batch_size), int(x_train.shape[0]))], y_train[: min(int(args.batch_size), int(y_train.shape[0]))], int(args.classes))
    source = {h: _f(row.get(f"source_h{h}"), -999.0) for h in [100, 400, 800, 1600, 3200]}
    return {
        "job_order": job_order,
        "v21_id": row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "probe_batches": len(batches),
        "source_state_pair_cosine_mean": sum(pair_cos) / len(pair_cos) if pair_cos else "",
        "source_state_pair_cosine_min": min(pair_cos) if pair_cos else "",
        "source_state_norm": state_norm,
        "source_state_coherence": state_norm,
        "max_control_cosine": max(control_cos) if control_cos else "",
        "mean_control_cosine": sum(control_cos) / len(control_cos) if control_cos else "",
        "radial_fraction": radial_fraction,
        "B1_gain": gains.get("B1_gain", ""),
        "B2_transfer_gain": gains.get("B2_transfer_gain", ""),
        "B3_safety_gain": gains.get("B3_safety_gain", ""),
        "corrupt_gain": gains.get("corrupt_gain", ""),
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
        "source_state_coherence",
        "source_state_pair_cosine_mean",
        "source_state_pair_cosine_min",
        "B2_transfer_gain",
        "B3_safety_gain",
        "max_control_cosine",
        "radial_fraction",
    ]
    z = {key: _zscore([_f(r.get(key)) for r in rows], groups) for key in feature_keys}
    out = []
    for i, row in enumerate(rows):
        score = (
            0.35 * z["source_state_coherence"][i]
            + 0.30 * z["source_state_pair_cosine_mean"][i]
            + 0.20 * z["source_state_pair_cosine_min"][i]
            + 0.25 * z["B2_transfer_gain"][i]
            + 0.15 * z["B3_safety_gain"][i]
            - 0.25 * z["max_control_cosine"][i]
            - 0.10 * z["radial_fraction"][i]
        )
        train_only_gate = int(
            _f(row.get("source_state_coherence"), -999.0) >= 0.20
            and _f(row.get("source_state_pair_cosine_mean"), -999.0) >= -0.10
            and _f(row.get("max_control_cosine"), 999.0) <= 0.995
            and _f(row.get("B3_safety_gain"), -999.0) >= -0.005
        )
        item = dict(row)
        item["PSSC_score"] = score
        item["PSSC_train_only_certificate_pass"] = train_only_gate
        item["certificate_family"] = "C-O6_pathwise_source_state_persistence"
        out.append(item)
    return sorted(out, key=lambda r: _f(r.get("PSSC_score"), -999.0), reverse=True)


def _summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    scores = [_f(r.get("PSSC_score")) for r in rows]
    labels = [int_flag(r.get("official_early_chain_h100_h400_h800_positive")) for r in rows]
    retained = [int_flag(r.get("retained_h800_h3200_positive")) for r in rows]
    h3200 = [int_flag(r.get("h3200_positive")) for r in rows]
    controls = [0 for _r in rows]
    p20, r20, control_frac = _precision_topk(scores, labels, controls, top_k)
    blockers = []
    if sum(labels) == 0:
        blockers.append("no_positive_official_early_chain_label_rows")
    if _f(_auc(scores, labels), -1.0) < 0.75:
        blockers.append("auc_official_early_chain_gate")
    if _f(p20, -1.0) < 0.50:
        blockers.append("precision_top20_gate")
    if _f(control_frac, 1.0) > 0.10:
        blockers.append("control_equivalent_fraction_gate")
    return {
        "certificate_family": "C-O6_pathwise_source_state_persistence",
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
        "PSSC_train_only_certificate_pass_rows": sum(int_flag(r.get("PSSC_train_only_certificate_pass")) for r in rows),
        "PSSC_official_observer_pass": int(not blockers),
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
    gen = torch.Generator(device=device).manual_seed(208_806 + train_seed + sum(ord(c) for c in run_kind))
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    mechanism = str(source_row.get("mechanism", ""))
    short_state: torch.Tensor | None = None
    long_state: torch.Tensor | None = None
    source_state_norm_sum = 0.0
    source_state_alignment_sum = 0.0
    source_state_count = 0
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, float]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
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
        else:
            raw = make_update(model, mechanism, xb, yb, seed=train_seed + step)
            raw_vec = _signed_displacement(raw)
            if run_kind == "PSSC-AdamWPlusDirectFU":
                opt_adam.step()
                direct = UpdateTensor(raw.tensor.detach(), raw.kind, raw.sign_rule, raw.space, "v22_08_post_nogo_direct_fu_control", raw.mechanism, role=raw.role, one_step_descent_claim=0)
                apply_update(model, direct, lr=float(args.fu_lr))
            elif run_kind == "CTRL-RandomMatchedState":
                opt_adam.step()
                noise = torch.randn(raw_vec.shape, device=device, generator=gen)
                rnd = UpdateTensor(normalized_like(noise, raw.tensor.detach()), "source_state_random_control", raw.sign_rule, "optimizer_source_state", "v22_08_post_nogo_random_matched_state", raw.mechanism, role="slow_source_state_control", one_step_descent_claim=0)
                apply_update(model, rnd, lr=float(args.fu_lr))
            else:
                short_state = raw_vec.clone() if short_state is None else float(args.short_beta) * short_state + (1.0 - float(args.short_beta)) * raw_vec
                long_state = raw_vec.clone() if long_state is None else float(args.long_beta) * long_state + (1.0 - float(args.long_beta)) * raw_vec
                stable_vec = 0.25 * short_state + 0.75 * long_state
                source_state_norm_sum += float(torch.linalg.vector_norm(stable_vec.float()).item())
                source_state_alignment_sum += cosine(_unit(stable_vec), _unit(raw_vec))
                source_state_count += 1
                opt_adam.step()
                stable_update = UpdateTensor(
                    tensor=normalized_like(stable_vec, raw.tensor.detach()),
                    kind="source_state_slow_momentum",
                    sign_rule="add",
                    space="optimizer_source_state",
                    source="v22_08_post_nogo_pathwise_slow_source_state",
                    mechanism="C-O6-PathwiseSourceStatePersistence",
                    role="slow_source_state",
                    one_step_descent_claim=0,
                )
                apply_update(model, stable_update, lr=float(args.fu_lr))
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    out: dict[str, Any] = {
        "run_kind": run_kind,
        "certificate_family": "C-O6_pathwise_source_state_persistence",
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "PSSC_score": source_row.get("PSSC_score", ""),
        "source_state_norm_mean": source_state_norm_sum / source_state_count if source_state_count else "",
        "source_state_alignment_mean": source_state_alignment_sum / source_state_count if source_state_count else "",
        "short_beta": args.short_beta,
        "long_beta": args.long_beta,
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
    candidate_rows = [r for r in enriched if str(r.get("run_kind")) == "PSSC-AdamWPlusSlowSourceState"]
    summary = []
    for row in candidate_rows:
        debt_ok = int(_f(row.get("CEp99_h3200"), 0.0) <= 20.0 and _f(row.get("ECE_h3200"), 0.0) <= 1.0 and _f(row.get("Brier_h3200"), 0.0) <= 1.0)
        source_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        summary.append(
            {
                "v21_id": row.get("v21_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "PSSC_score": row.get("PSSC_score", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h2400": row.get("source_vs_best_control_h2400", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "source_state_norm_mean": row.get("source_state_norm_mean", ""),
                "source_state_alignment_mean": row.get("source_state_alignment_mean", ""),
                "debt_not_exploded": debt_ok,
                "PSSC_fresh_C3_source_state_pass": int(source_pass and debt_ok),
                "official_C3_pass": 0,
                "blocker": "" if source_pass and debt_ok else "source_horizon_or_debt_gate_failed",
            }
        )
    positives = {h: sum(1 for r in summary if _f(r.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005) for h in HORIZONS}
    route = {
        "fresh_rows": len(summary),
        "fresh_C3_source_state_pass_rows": sum(int_flag(r.get("PSSC_fresh_C3_source_state_pass")) for r in summary),
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
    observer_route = read_json(out_dir / "v22_08_retained_source_observer_route.json")
    candidates = _candidate_rows(args)
    probe_raw: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        try:
            probe_raw.append(_probe_one(args, row, idx))
        except Exception as exc:
            probe_raw.append(
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
    measured = [r for r in probe_raw if str(r.get("execution_status")) == "measured"]
    scored = _score_probe(measured)
    blocked = [r for r in probe_raw if str(r.get("execution_status")) != "measured"]
    matrix = scored + blocked
    summary = _summary(scored, 20)
    selected = scored[: max(0, int(args.fresh_top_k))]
    fresh_rows: list[dict[str, Any]] = []
    for row in selected:
        for run_kind in [
            "PSSC-AdamWPlusSlowSourceState",
            "PSSC-AdamWPlusDirectFU",
            "CTRL-AdamW",
            "CTRL-SGD",
            "CTRL-NoOp",
            "CTRL-RandomMatchedState",
        ]:
            fresh_rows.append(_train_fresh(args, row, run_kind))
    fresh_enriched, fresh_pack = _fresh_summary(fresh_rows)
    fresh_summary = fresh_pack["summary"]
    fresh_route = fresh_pack["route"]
    route = {
        "route": "PostNoGoSourceStateFreshC3Opened" if int(fresh_route["fresh_C3_source_state_pass_rows"]) else "PostNoGoSourceStateCertificateBlocked",
        "previous_observer_route": observer_route.get("route", ""),
        "certificate_family": "C-O6_pathwise_source_state_persistence",
        "probe_rows": len(matrix),
        "probe_measured_rows": len(scored),
        "PSSC_train_only_certificate_pass_rows": summary.get("PSSC_train_only_certificate_pass_rows", 0),
        "PSSC_official_observer_pass": summary.get("PSSC_official_observer_pass", 0),
        "fresh_C3_source_state_pass_rows": fresh_route["fresh_C3_source_state_pass_rows"],
        "official_C3_pass_rows": 0,
        "promotion_allowed": 0,
        "blocker": "official_observer_gate_or_fresh_C3_source_horizon_failed",
        "next_codex_action": "C-O6 failed under fresh source-state integration; require a new certificate outside C-O1..C-O6 or external theoretical input",
    }
    write_rows(out_dir / "v22_08_post_nogo_source_state_certificate_matrix.csv", matrix)
    write_rows(out_dir / "v22_08_post_nogo_source_state_certificate_summary.csv", [summary])
    write_rows(out_dir / "v22_08_post_nogo_source_state_selected_candidates.csv", selected)
    write_rows(out_dir / "v22_08_post_nogo_source_state_fresh_c3_matrix.csv", fresh_enriched)
    write_rows(out_dir / "v22_08_post_nogo_source_state_fresh_c3_summary.csv", fresh_summary)
    write_json(out_dir / "v22_08_post_nogo_source_state_route.json", route)
    simple_svg(out_dir / "figures/v22_08_post_nogo_pssc_score.svg", "v22.08 post-no-go PSSC score", scored, "PSSC_score")
    simple_svg(out_dir / "figures/v22_08_post_nogo_source_state_h3200.svg", "v22.08 post-no-go source-state h3200", fresh_summary, "source_vs_best_control_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_post_nogo_source_state_certificate.py --source-dir {args.source_dir} --device {args.device} --fresh-top-k {int(args.fresh_top_k)} --steps {int(args.steps)} --out-dir {out_dir}",
        status="completed",
        note=f"probe_rows={len(matrix)} train_only_pass={summary.get('PSSC_train_only_certificate_pass_rows', 0)} fresh_C3_pass={fresh_route['fresh_C3_source_state_pass_rows']} route={route['route']}",
    )


if __name__ == "__main__":
    main()
