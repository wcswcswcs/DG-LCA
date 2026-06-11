#!/usr/bin/env python3
"""v22.08 post-no-go cross-fit influence transport certificate.

C-O11 is a train-only retained-source observability probe after C-O1..C-O10
fail.  It tests whether a candidate FU direction generated on one train fold
can be transported by an influence direction from a disjoint train fold, while
remaining distinct from random, sign-flip, and corrupt-label controls.  Future
source labels are audit columns only and never generate the direction.
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

from dgkan.fu.core import UpdateTensor, apply_update, cosine, flat_grad, normalized_like  # noqa: E402
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
from experiments.run_v22_08_post_nogo_source_state_certificate import _auc, _precision_topk, _zscore  # noqa: E402


HORIZONS = (100, 400, 800, 1600, 2400, 3200)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2206_COMBINED_SOURCE))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--train-size", type=int, default=96)
    p.add_argument("--val-size", type=int, default=48)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
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


def _signed_displacement(update: UpdateTensor) -> torch.Tensor:
    sign = -1.0 if update.sign_rule == "subtract" else 1.0
    return update.tensor.detach().float() * sign


def _unit(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector.detach().float())
    if float(norm.item()) <= 1.0e-12:
        return torch.zeros_like(vector)
    return vector.detach().float() / norm.clamp_min(1.0e-12)


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


def _loss_value(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(model(x).float(), y).item())


def _margin_value(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(x).float()
        masked = logits.clone()
        masked[torch.arange(int(x.shape[0]), device=x.device), y] = -float("inf")
        competitor = masked.argmax(dim=1)
        margin = logits[torch.arange(int(x.shape[0]), device=x.device), y] - logits[torch.arange(int(x.shape[0]), device=x.device), competitor]
    return float(margin.mean().item())


def _grad_descent_displacement(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x).float(), y).backward()
    direction = -flat_grad(model, x.device).detach().float()
    model.zero_grad(set_to_none=True)
    return direction


def _random_like(reference: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    noise = torch.randn(reference.shape, device=reference.device, generator=gen)
    return normalized_like(noise, reference.detach())


def _tiny_effects(
    model: torch.nn.Module,
    vector: torch.Tensor,
    reference: torch.Tensor,
    x_a: torch.Tensor,
    y_a: torch.Tensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    x_c: torch.Tensor,
    y_c: torch.Tensor,
    fu_lr: float,
) -> dict[str, float]:
    base_a = _loss_value(model, x_a, y_a)
    base_b = _loss_value(model, x_b, y_b)
    base_c = _loss_value(model, x_c, y_c)
    base_b_margin = _margin_value(model, x_b, y_b)
    base_c_margin = _margin_value(model, x_c, y_c)
    clone = deepcopy(model)
    update = UpdateTensor(
        tensor=normalized_like(vector.detach().float(), reference.detach().float()),
        kind="crossfit_influence_probe",
        sign_rule="add",
        space="train_crossfit_influence_transport",
        source="train_only_crossfit_influence_certificate",
        mechanism="C-O11-CrossFitInfluenceProbe",
        role="all",
        one_step_descent_claim=0,
    )
    apply_update(clone, update, lr=float(fu_lr))
    return {
        "A_train_loss_gain": base_a - _loss_value(clone, x_a, y_a),
        "B_crossfit_loss_gain": base_b - _loss_value(clone, x_b, y_b),
        "C_safety_loss_gain": base_c - _loss_value(clone, x_c, y_c),
        "B_margin_gain": _margin_value(clone, x_b, y_b) - base_b_margin,
        "C_margin_gain": _margin_value(clone, x_c, y_c) - base_c_margin,
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
    if batch <= 0:
        raise RuntimeError("empty train split")
    x_a = x_train[:batch]
    y_a = y_train[:batch]
    x_b = x_train[batch : 2 * batch]
    y_b = y_train[batch : 2 * batch]
    x_c = x_train[2 * batch : 3 * batch]
    y_c = y_train[2 * batch : 3 * batch]
    if int(x_b.shape[0]) == 0:
        x_b, y_b = x_a, y_a
    if int(x_c.shape[0]) == 0:
        x_c, y_c = x_b, y_b
    gen = torch.Generator(device=device).manual_seed(229_210 + train_seed + job_order)
    mechanism = str(row.get("mechanism", ""))
    raw = make_update(model, mechanism, x_a, y_a, seed=train_seed + 101)
    corrupt = make_update(model, mechanism, x_a, (y_a + 1) % max(2, int(args.classes)), seed=train_seed + 102)
    source_vec = _signed_displacement(raw)
    corrupt_vec = _signed_displacement(corrupt)
    influence_vec = _grad_descent_displacement(model, x_b, y_b)
    mixed_vec = normalized_like(_unit(source_vec) + _unit(influence_vec), source_vec)
    random_vec = _random_like(source_vec, gen)
    sign_vec = -source_vec
    source_eff = _tiny_effects(model, mixed_vec, source_vec, x_a, y_a, x_b, y_b, x_c, y_c, float(args.fu_lr))
    raw_eff = _tiny_effects(model, source_vec, source_vec, x_a, y_a, x_b, y_b, x_c, y_c, float(args.fu_lr))
    random_eff = _tiny_effects(model, random_vec, source_vec, x_a, y_a, x_b, y_b, x_c, y_c, float(args.fu_lr))
    sign_eff = _tiny_effects(model, sign_vec, source_vec, x_a, y_a, x_b, y_b, x_c, y_c, float(args.fu_lr))
    corrupt_eff = _tiny_effects(model, corrupt_vec, source_vec, x_a, y_a, x_b, y_b, x_c, y_c, float(args.fu_lr))
    a_control = max(random_eff["A_train_loss_gain"], sign_eff["A_train_loss_gain"], corrupt_eff["A_train_loss_gain"])
    b_control = max(random_eff["B_crossfit_loss_gain"], sign_eff["B_crossfit_loss_gain"], corrupt_eff["B_crossfit_loss_gain"])
    c_control = max(random_eff["C_safety_loss_gain"], sign_eff["C_safety_loss_gain"], corrupt_eff["C_safety_loss_gain"])
    b_margin_control = max(random_eff["B_margin_gain"], sign_eff["B_margin_gain"], corrupt_eff["B_margin_gain"])
    c_margin_control = max(random_eff["C_margin_gain"], sign_eff["C_margin_gain"], corrupt_eff["C_margin_gain"])
    source = {h: _f(row.get(f"source_h{h}"), -999.0) for h in [100, 400, 800, 1600, 3200]}
    source_influence_cosine = cosine(source_vec, influence_vec)
    random_influence_cosine = cosine(random_vec, influence_vec)
    corrupt_influence_cosine = cosine(corrupt_vec, influence_vec)
    sign_influence_cosine = cosine(sign_vec, influence_vec)
    return {
        "job_order": job_order,
        "v21_id": row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "fold_A_size": int(x_a.shape[0]),
        "fold_B_size": int(x_b.shape[0]),
        "fold_C_size": int(x_c.shape[0]),
        "source_raw_A_train_loss_gain": raw_eff["A_train_loss_gain"],
        "source_crossfit_A_train_loss_gain": source_eff["A_train_loss_gain"],
        "source_crossfit_B_loss_gain": source_eff["B_crossfit_loss_gain"],
        "source_crossfit_C_safety_loss_gain": source_eff["C_safety_loss_gain"],
        "random_B_loss_gain": random_eff["B_crossfit_loss_gain"],
        "sign_flip_B_loss_gain": sign_eff["B_crossfit_loss_gain"],
        "corrupt_B_loss_gain": corrupt_eff["B_crossfit_loss_gain"],
        "A_train_control_gap": source_eff["A_train_loss_gain"] - a_control,
        "B_crossfit_control_gap": source_eff["B_crossfit_loss_gain"] - b_control,
        "C_safety_control_gap": source_eff["C_safety_loss_gain"] - c_control,
        "source_crossfit_B_margin_gain": source_eff["B_margin_gain"],
        "source_crossfit_C_margin_gain": source_eff["C_margin_gain"],
        "B_margin_control_gap": source_eff["B_margin_gain"] - b_margin_control,
        "C_margin_control_gap": source_eff["C_margin_gain"] - c_margin_control,
        "source_influence_cosine": source_influence_cosine,
        "random_influence_cosine": random_influence_cosine,
        "sign_flip_influence_cosine": sign_influence_cosine,
        "corrupt_influence_cosine": corrupt_influence_cosine,
        "crossfit_alignment_control_gap": source_influence_cosine - max(random_influence_cosine, sign_influence_cosine, corrupt_influence_cosine),
        "influence_vector_norm": float(torch.linalg.vector_norm(influence_vec.detach().float()).item()),
        "source_update_norm": float(torch.linalg.vector_norm(source_vec.detach().float()).item()),
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
        "source_crossfit_B_loss_gain",
        "B_crossfit_control_gap",
        "source_crossfit_C_safety_loss_gain",
        "C_safety_control_gap",
        "source_crossfit_B_margin_gain",
        "B_margin_control_gap",
        "source_influence_cosine",
        "crossfit_alignment_control_gap",
        "A_train_control_gap",
        "random_influence_cosine",
        "corrupt_influence_cosine",
    ]
    z = {key: _zscore([_f(r.get(key)) for r in rows], groups) for key in feature_keys}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        score = (
            0.30 * z["source_crossfit_B_loss_gain"][i]
            + 0.35 * z["B_crossfit_control_gap"][i]
            + 0.15 * z["source_crossfit_C_safety_loss_gain"][i]
            + 0.15 * z["C_safety_control_gap"][i]
            + 0.15 * z["source_crossfit_B_margin_gain"][i]
            + 0.15 * z["B_margin_control_gap"][i]
            + 0.15 * z["source_influence_cosine"][i]
            + 0.25 * z["crossfit_alignment_control_gap"][i]
            + 0.10 * z["A_train_control_gap"][i]
            - 0.10 * z["random_influence_cosine"][i]
            - 0.10 * z["corrupt_influence_cosine"][i]
        )
        train_gate = int(
            _f(row.get("source_crossfit_B_loss_gain"), -999.0) > 0.0
            and _f(row.get("B_crossfit_control_gap"), -999.0) >= -1.0e-3
            and _f(row.get("C_safety_control_gap"), -999.0) >= -1.0e-3
            and _f(row.get("B_margin_control_gap"), -999.0) >= -1.0e-3
            and _f(row.get("source_influence_cosine"), -999.0) > -0.05
            and _f(row.get("crossfit_alignment_control_gap"), -999.0) >= -0.05
        )
        item = dict(row)
        item["CFI_score"] = score
        item["CFI_train_only_certificate_pass"] = train_gate
        item["certificate_family"] = "C-O11_CrossFit_Influence_Transport_Certificate"
        out.append(item)
    return sorted(out, key=lambda r: _f(r.get("CFI_score"), -999.0), reverse=True)


def _summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    scores = [_f(r.get("CFI_score")) for r in rows]
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
        "certificate_family": "C-O11_CrossFit_Influence_Transport_Certificate",
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
        "CFI_train_only_certificate_pass_rows": sum(int_flag(r.get("CFI_train_only_certificate_pass")) for r in rows),
        "CFI_official_observer_pass": int(not blockers),
        "blocker": ";".join(dict.fromkeys(blockers)),
    }


def _crossfit_influence_update(
    model: torch.nn.Module,
    raw: UpdateTensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    *,
    random_gen: torch.Generator | None = None,
) -> UpdateTensor:
    source_vec = _signed_displacement(raw)
    if random_gen is not None:
        mixed = _random_like(source_vec, random_gen)
        raw_crossfit_cosine = 0.0
    else:
        influence_vec = _grad_descent_displacement(model, x_b, y_b)
        mixed = _unit(source_vec) + _unit(influence_vec)
        mixed = normalized_like(mixed, source_vec)
        raw_crossfit_cosine = cosine(source_vec, influence_vec)
    return UpdateTensor(
        tensor=normalized_like(mixed.detach().float(), raw.tensor.detach().float()),
        kind="crossfit_influence_transport",
        sign_rule="add",
        space="train_crossfit_influence_transport",
        source="train_only_crossfit_influence_certificate",
        mechanism="C-O11-CrossFitInfluenceTransport",
        role="all",
        one_step_descent_claim=0,
        diagnostics={"raw_crossfit_cosine": raw_crossfit_cosine},
    )


def _split_indices(n: int, batch: int, gen: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if n >= 2 * batch:
        perm = torch.randperm(n, generator=gen, device=device)
        return perm[:batch], perm[batch : 2 * batch]
    idx_a = torch.randint(0, n, (batch,), generator=gen, device=device)
    idx_b = torch.randint(0, n, (batch,), generator=gen, device=device)
    return idx_a, idx_b


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
    gen = torch.Generator(device=device).manual_seed(229_318 + train_seed + sum(ord(c) for c in run_kind))
    batch = min(int(args.batch_size), max(1, int(x_train.shape[0]) // 2))
    mechanism = str(source_row.get("mechanism", ""))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, float]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    injection_count = 0
    injection_norm_sum = 0.0
    raw_crossfit_cosine_sum = 0.0
    crossfit_loss_gain_sum = 0.0
    for step in range(1, int(args.steps) + 1):
        idx_a, idx_b = _split_indices(int(x_train.shape[0]), batch, gen, device)
        x_a = x_train[idx_a]
        y_a = y_train[idx_a]
        x_b = x_train[idx_b]
        y_b = y_train[idx_b]
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
                update = _crossfit_influence_update(model, raw, x_b, y_b, random_gen=gen if run_kind == "CTRL-RandomMatchedCrossFitInfluenceFU" else None)
                raw_crossfit_cosine_sum += float(update.diagnostics.get("raw_crossfit_cosine", 0.0) if update.diagnostics else 0.0)
                injection_count += 1
                injection_norm_sum += float(torch.linalg.vector_norm(update.tensor.detach().float()).item())
                apply_update(model, update, lr=float(args.fu_lr))
                crossfit_loss_gain_sum += before - _loss_value(model, x_b, y_b)
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    out: dict[str, Any] = {
        "run_kind": run_kind,
        "certificate_family": "C-O11_CrossFit_Influence_Transport_Certificate",
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "CFI_score": source_row.get("CFI_score", ""),
        "refresh_interval": int(args.refresh_interval),
        "injection_count": injection_count,
        "injection_norm_sum": injection_norm_sum,
        "raw_crossfit_cosine_mean": raw_crossfit_cosine_sum / max(1, injection_count),
        "crossfit_loss_gain_mean": crossfit_loss_gain_sum / max(1, injection_count),
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
    candidates = [r for r in enriched if str(r.get("run_kind")) == "CFI-AdamWPlusCrossFitInfluenceFU"]
    summary: list[dict[str, Any]] = []
    for row in candidates:
        debt_ok = int(_f(row.get("CEp99_h3200"), 0.0) <= 20.0 and _f(row.get("ECE_h3200"), 0.0) <= 1.0 and _f(row.get("Brier_h3200"), 0.0) <= 1.0)
        source_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        summary.append(
            {
                "v21_id": row.get("v21_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "CFI_score": row.get("CFI_score", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h2400": row.get("source_vs_best_control_h2400", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "raw_crossfit_cosine_mean": row.get("raw_crossfit_cosine_mean", ""),
                "crossfit_loss_gain_mean": row.get("crossfit_loss_gain_mean", ""),
                "injection_count": row.get("injection_count", ""),
                "injection_norm_sum": row.get("injection_norm_sum", ""),
                "debt_not_exploded": debt_ok,
                "CFI_fresh_C3_crossfit_influence_pass": int(source_pass and debt_ok),
                "official_C3_pass": 0,
                "blocker": "" if source_pass and debt_ok else "source_horizon_or_debt_gate_failed",
            }
        )
    positives = {h: sum(1 for r in summary if _f(r.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005) for h in HORIZONS}
    route = {
        "fresh_rows": len(summary),
        "fresh_C3_crossfit_influence_pass_rows": sum(int_flag(r.get("CFI_fresh_C3_crossfit_influence_pass")) for r in summary),
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
    prev_route = read_json(out_dir / "v22_08_post_nogo_bootstrap_basin_route.json")
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
            "CFI-AdamWPlusCrossFitInfluenceFU",
            "CTRL-AdamW",
            "CTRL-SGD",
            "CTRL-NoOp",
            "CTRL-RandomMatchedCrossFitInfluenceFU",
        ]:
            fresh_rows.append(_train_fresh(args, row, run_kind))
    fresh_enriched, fresh_pack = _fresh_summary(fresh_rows)
    fresh_summary = fresh_pack["summary"]
    fresh_route = fresh_pack["route"]
    route = {
        "route": "PostNoGoCrossFitInfluenceFreshC3Opened" if int(fresh_route["fresh_C3_crossfit_influence_pass_rows"]) else "PostNoGoCrossFitInfluenceBlocked",
        "previous_post_nogo_route": prev_route.get("route", ""),
        "certificate_family": "C-O11_CrossFit_Influence_Transport_Certificate",
        "probe_rows": len(matrix),
        "probe_measured_rows": len(scored),
        "CFI_train_only_certificate_pass_rows": summary.get("CFI_train_only_certificate_pass_rows", 0),
        "CFI_official_observer_pass": summary.get("CFI_official_observer_pass", 0),
        "fresh_C3_crossfit_influence_pass_rows": fresh_route["fresh_C3_crossfit_influence_pass_rows"],
        "official_C3_pass_rows": 0,
        "promotion_allowed": 0,
        "blocker": "official_observer_gate_or_fresh_C3_source_horizon_failed",
        "next_codex_action": "C-O11 failed under train-only cross-fit influence transport; require external theory or mark C-O1..C-O11 local boundary",
    }
    write_rows(out_dir / "v22_08_post_nogo_crossfit_influence_matrix.csv", matrix)
    write_rows(out_dir / "v22_08_post_nogo_crossfit_influence_summary.csv", [summary])
    write_rows(out_dir / "v22_08_post_nogo_crossfit_influence_selected_candidates.csv", selected)
    write_rows(out_dir / "v22_08_post_nogo_crossfit_influence_fresh_c3_matrix.csv", fresh_enriched)
    write_rows(out_dir / "v22_08_post_nogo_crossfit_influence_fresh_c3_summary.csv", fresh_summary)
    write_json(out_dir / "v22_08_post_nogo_crossfit_influence_route.json", route)
    simple_svg(out_dir / "figures/v22_08_post_nogo_cfi_score.svg", "v22.08 post-no-go CFI score", scored, "CFI_score")
    simple_svg(out_dir / "figures/v22_08_post_nogo_cfi_h3200.svg", "v22.08 post-no-go CFI h3200", fresh_summary, "source_vs_best_control_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_post_nogo_crossfit_influence_certificate.py --source-dir {args.source_dir} --device {args.device} --fresh-top-k {int(args.fresh_top_k)} --steps {int(args.steps)} --refresh-interval {int(args.refresh_interval)} --out-dir {out_dir}",
        status="completed",
        note=f"probe_rows={len(matrix)} train_only_pass={summary.get('CFI_train_only_certificate_pass_rows', 0)} fresh_C3_pass={fresh_route['fresh_C3_crossfit_influence_pass_rows']} route={route['route']}",
    )


if __name__ == "__main__":
    main()
