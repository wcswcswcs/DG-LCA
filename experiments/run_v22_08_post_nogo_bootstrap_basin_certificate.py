#!/usr/bin/env python3
"""v22.08 post-no-go bootstrap basin-consensus certificate.

C-O10 is a train-only retained-source observability probe after C-O1..C-O9
fail.  It tests whether a candidate FU direction is aligned with a short-run
bootstrap basin direction formed by independent train-only sub-training
replicas, while remaining distinct from random, sign-flip, and corrupt-label
controls.  Future source labels are audit columns only and never generate the
direction.
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

from dgkan.fu.core import UpdateTensor, apply_update, cosine, flat_params, load_flat_params, normalized_like  # noqa: E402
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
    p.add_argument("--train-size", type=int, default=64)
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
    p.add_argument("--bootstrap-steps", type=int, default=16)
    p.add_argument("--bootstrap-replicas", type=int, default=3)
    p.add_argument("--bootstrap-lr", type=float, default=0.003)
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


def _random_like(reference: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    noise = torch.randn(reference.shape, device=reference.device, generator=gen)
    return normalized_like(noise, reference.detach())


def _bootstrap_basin_direction(
    args: argparse.Namespace,
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_seed: int,
    cache_key: tuple[str, int, int],
    cache: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    if cache_key in cache:
        return cache[cache_key]
    device = x_train.device
    base = flat_params(model).detach().float().clone()
    batch = min(int(args.batch_size), int(x_train.shape[0]))
    displacements: list[torch.Tensor] = []
    loss_gains: list[float] = []
    for replica in range(max(1, int(args.bootstrap_replicas))):
        clone = deepcopy(model)
        opt = torch.optim.AdamW(clone.parameters(), lr=float(args.bootstrap_lr), weight_decay=float(args.weight_decay))
        gen = torch.Generator(device=device).manual_seed(229_010 + train_seed * 17 + replica * 101)
        before = _loss_value(clone, x_train[:batch], y_train[:batch])
        for _step in range(max(1, int(args.bootstrap_steps))):
            idx = torch.randint(0, int(x_train.shape[0]), (batch,), generator=gen, device=device)
            xb = x_train[idx]
            yb = y_train[idx]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(clone(xb).float(), yb).backward()
            opt.step()
        after = _loss_value(clone, x_train[:batch], y_train[:batch])
        disp = flat_params(clone).detach().float() - base
        if float(torch.linalg.vector_norm(disp).item()) > 1.0e-12:
            displacements.append(disp)
        loss_gains.append(before - after)
    if displacements:
        stack = torch.stack(displacements, dim=0)
        consensus = stack.mean(dim=0)
        norms = torch.linalg.vector_norm(stack, dim=1)
        ratio = float(torch.linalg.vector_norm(consensus).item() / max(float(norms.mean().item()), 1.0e-12))
        pair_cosines: list[float] = []
        for i in range(len(displacements)):
            for j in range(i + 1, len(displacements)):
                pair_cosines.append(cosine(displacements[i], displacements[j]))
        pair_mean = sum(pair_cosines) / len(pair_cosines) if pair_cosines else 1.0
        pair_min = min(pair_cosines) if pair_cosines else 1.0
    else:
        consensus = torch.zeros_like(base)
        ratio = 0.0
        pair_mean = 0.0
        pair_min = 0.0
    item = {
        "basin_vector": consensus.detach().float(),
        "bootstrap_pair_cosine_mean": pair_mean,
        "bootstrap_pair_cosine_min": pair_min,
        "bootstrap_consensus_norm_ratio": ratio,
        "bootstrap_loss_gain_mean": sum(loss_gains) / max(1, len(loss_gains)),
        "bootstrap_replicas": int(args.bootstrap_replicas),
        "bootstrap_steps": int(args.bootstrap_steps),
    }
    cache[cache_key] = item
    return item


def _tiny_effects(
    model: torch.nn.Module,
    vector: torch.Tensor,
    reference: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    x_b2: torch.Tensor,
    y_b2: torch.Tensor,
    x_b3: torch.Tensor,
    y_b3: torch.Tensor,
    fu_lr: float,
) -> dict[str, float]:
    base_train = _loss_value(model, xb, yb)
    base_b2 = _loss_value(model, x_b2, y_b2)
    base_b3 = _loss_value(model, x_b3, y_b3)
    clone = deepcopy(model)
    update = UpdateTensor(
        tensor=normalized_like(vector.detach().float(), reference.detach().float()),
        kind="bootstrap_basin_probe",
        sign_rule="add",
        space="train_bootstrap_basin_consensus",
        source="train_only_bootstrap_basin_certificate",
        mechanism="C-O10-BootstrapBasinProbe",
        role="all",
        one_step_descent_claim=0,
    )
    apply_update(clone, update, lr=float(fu_lr))
    return {
        "train_loss_gain": base_train - _loss_value(clone, xb, yb),
        "B2_transfer_gain": base_b2 - _loss_value(clone, x_b2, y_b2),
        "B3_safety_gain": base_b3 - _loss_value(clone, x_b3, y_b3),
    }


def _probe_one(
    args: argparse.Namespace,
    row: dict[str, str],
    job_order: int,
    cache: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
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
    y_b2 = y_train[batch : min(2 * batch, int(x_train.shape[0]))]
    if int(x_b2.shape[0]) == 0:
        x_b2, y_b2 = xb, yb
    x_b3 = x_train[-batch:]
    y_b3 = y_train[-batch:]
    gen = torch.Generator(device=device).manual_seed(229_080 + train_seed + job_order)
    mechanism = str(row.get("mechanism", ""))
    raw = make_update(model, mechanism, xb, yb, seed=train_seed + 101)
    corrupt = make_update(model, mechanism, xb, (yb + 1) % max(2, int(args.classes)), seed=train_seed + 102)
    source_vec = _signed_displacement(raw)
    corrupt_vec = _signed_displacement(corrupt)
    basin = _bootstrap_basin_direction(args, model, x_train, y_train, train_seed, (dataset, seed, train_seed), cache)
    basin_vec = basin["basin_vector"].to(device=source_vec.device)
    mixed_vec = normalized_like(_unit(source_vec) + _unit(basin_vec), source_vec)
    random_vec = _random_like(source_vec, gen)
    sign_vec = -source_vec
    source_eff = _tiny_effects(model, mixed_vec, source_vec, xb, yb, x_b2, y_b2, x_b3, y_b3, float(args.fu_lr))
    raw_eff = _tiny_effects(model, source_vec, source_vec, xb, yb, x_b2, y_b2, x_b3, y_b3, float(args.fu_lr))
    random_eff = _tiny_effects(model, random_vec, source_vec, xb, yb, x_b2, y_b2, x_b3, y_b3, float(args.fu_lr))
    sign_eff = _tiny_effects(model, sign_vec, source_vec, xb, yb, x_b2, y_b2, x_b3, y_b3, float(args.fu_lr))
    corrupt_eff = _tiny_effects(model, corrupt_vec, source_vec, xb, yb, x_b2, y_b2, x_b3, y_b3, float(args.fu_lr))
    train_control = max(random_eff["train_loss_gain"], sign_eff["train_loss_gain"], corrupt_eff["train_loss_gain"])
    b2_control = max(random_eff["B2_transfer_gain"], sign_eff["B2_transfer_gain"], corrupt_eff["B2_transfer_gain"])
    b3_control = max(random_eff["B3_safety_gain"], sign_eff["B3_safety_gain"], corrupt_eff["B3_safety_gain"])
    source = {h: _f(row.get(f"source_h{h}"), -999.0) for h in [100, 400, 800, 1600, 3200]}
    source_basin_cosine = cosine(source_vec, basin_vec)
    random_basin_cosine = cosine(random_vec, basin_vec)
    corrupt_basin_cosine = cosine(corrupt_vec, basin_vec)
    sign_basin_cosine = cosine(sign_vec, basin_vec)
    return {
        "job_order": job_order,
        "v21_id": row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "bootstrap_steps": int(args.bootstrap_steps),
        "bootstrap_replicas": int(args.bootstrap_replicas),
        "source_raw_train_loss_gain": raw_eff["train_loss_gain"],
        "source_basin_train_loss_gain": source_eff["train_loss_gain"],
        "random_train_loss_gain": random_eff["train_loss_gain"],
        "sign_flip_train_loss_gain": sign_eff["train_loss_gain"],
        "corrupt_train_loss_gain": corrupt_eff["train_loss_gain"],
        "train_loss_control_gap": source_eff["train_loss_gain"] - train_control,
        "B2_transfer_gain": source_eff["B2_transfer_gain"],
        "B3_safety_gain": source_eff["B3_safety_gain"],
        "B2_control_gap": source_eff["B2_transfer_gain"] - b2_control,
        "B3_control_gap": source_eff["B3_safety_gain"] - b3_control,
        "source_basin_cosine": source_basin_cosine,
        "random_basin_cosine": random_basin_cosine,
        "sign_flip_basin_cosine": sign_basin_cosine,
        "corrupt_basin_cosine": corrupt_basin_cosine,
        "basin_alignment_control_gap": source_basin_cosine - max(random_basin_cosine, sign_basin_cosine, corrupt_basin_cosine),
        "bootstrap_pair_cosine_mean": basin["bootstrap_pair_cosine_mean"],
        "bootstrap_pair_cosine_min": basin["bootstrap_pair_cosine_min"],
        "bootstrap_consensus_norm_ratio": basin["bootstrap_consensus_norm_ratio"],
        "bootstrap_loss_gain_mean": basin["bootstrap_loss_gain_mean"],
        "basin_vector_norm": float(torch.linalg.vector_norm(basin_vec.detach().float()).item()),
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
        "source_basin_train_loss_gain",
        "train_loss_control_gap",
        "source_basin_cosine",
        "basin_alignment_control_gap",
        "bootstrap_pair_cosine_mean",
        "bootstrap_consensus_norm_ratio",
        "bootstrap_loss_gain_mean",
        "B2_control_gap",
        "B3_control_gap",
        "random_basin_cosine",
        "corrupt_basin_cosine",
    ]
    z = {key: _zscore([_f(r.get(key)) for r in rows], groups) for key in feature_keys}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        score = (
            0.25 * z["source_basin_train_loss_gain"][i]
            + 0.25 * z["train_loss_control_gap"][i]
            + 0.25 * z["source_basin_cosine"][i]
            + 0.30 * z["basin_alignment_control_gap"][i]
            + 0.15 * z["bootstrap_pair_cosine_mean"][i]
            + 0.15 * z["bootstrap_consensus_norm_ratio"][i]
            + 0.10 * z["bootstrap_loss_gain_mean"][i]
            + 0.10 * z["B2_control_gap"][i]
            + 0.10 * z["B3_control_gap"][i]
            - 0.10 * z["random_basin_cosine"][i]
            - 0.10 * z["corrupt_basin_cosine"][i]
        )
        train_gate = int(
            _f(row.get("source_basin_train_loss_gain"), -999.0) > 0.0
            and _f(row.get("train_loss_control_gap"), -999.0) >= -1.0e-3
            and _f(row.get("source_basin_cosine"), -999.0) > -0.05
            and _f(row.get("basin_alignment_control_gap"), -999.0) >= -0.05
            and _f(row.get("bootstrap_consensus_norm_ratio"), -999.0) > 0.10
            and _f(row.get("B2_control_gap"), -999.0) >= -1.0e-3
            and _f(row.get("B3_control_gap"), -999.0) >= -1.0e-3
        )
        item = dict(row)
        item["BBC_score"] = score
        item["BBC_train_only_certificate_pass"] = train_gate
        item["certificate_family"] = "C-O10_Bootstrap_Basin_Consensus_Certificate"
        out.append(item)
    return sorted(out, key=lambda r: _f(r.get("BBC_score"), -999.0), reverse=True)


def _summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    scores = [_f(r.get("BBC_score")) for r in rows]
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
        "certificate_family": "C-O10_Bootstrap_Basin_Consensus_Certificate",
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
        "BBC_train_only_certificate_pass_rows": sum(int_flag(r.get("BBC_train_only_certificate_pass")) for r in rows),
        "BBC_official_observer_pass": int(not blockers),
        "blocker": ";".join(dict.fromkeys(blockers)),
    }


def _bootstrap_basin_update(
    raw: UpdateTensor,
    basin_vec: torch.Tensor,
    *,
    random_gen: torch.Generator | None = None,
) -> UpdateTensor:
    source_vec = _signed_displacement(raw)
    if random_gen is not None:
        mixed = _random_like(source_vec, random_gen)
    else:
        mixed = _unit(source_vec) + _unit(basin_vec.to(device=source_vec.device))
        mixed = normalized_like(mixed, source_vec)
    return UpdateTensor(
        tensor=normalized_like(mixed.detach().float(), raw.tensor.detach().float()),
        kind="bootstrap_basin_consensus",
        sign_rule="add",
        space="train_bootstrap_basin_consensus",
        source="train_only_bootstrap_basin_certificate",
        mechanism="C-O10-BootstrapBasinConsensus",
        role="all",
        one_step_descent_claim=0,
        diagnostics={"raw_basin_cosine": cosine(source_vec, basin_vec.to(device=source_vec.device))},
    )


def _train_fresh(args: argparse.Namespace, source_row: dict[str, Any], run_kind: str) -> dict[str, Any]:
    device = resolve_device(args.device)
    dataset = str(source_row.get("dataset", "MNIST"))
    seed = int(float(source_row.get("seed", 0) or 0))
    local = deepcopy(args)
    local.basis_repair_variant = "R0-current"
    x_train, y_train, x_val, y_val = load_dataset(dataset, Path(args.data_root), int(args.train_size), int(args.val_size), seed, device, int(args.input_size))
    train_seed = int(float(source_row.get("train_seed") or 0))
    model = carrier_model("MLP", x_train, train_seed, local, device)
    base_params = flat_params(model).detach().float().clone()
    basin_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    basin = _bootstrap_basin_direction(args, model, x_train, y_train, train_seed, (dataset, seed, train_seed), basin_cache)
    basin_vec = basin["basin_vector"].to(device=device)
    load_flat_params(model, base_params)
    opt_adam = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    opt_sgd = torch.optim.SGD(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(229_118 + train_seed + sum(ord(c) for c in run_kind))
    batch = min(int(args.batch_size), int(x_train.shape[0]))
    mechanism = str(source_row.get("mechanism", ""))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, float]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    injection_count = 0
    injection_norm_sum = 0.0
    raw_basin_cosine_sum = 0.0
    train_loss_gain_sum = 0.0
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
                before = _loss_value(model, xb, yb)
                raw = make_update(model, mechanism, xb, yb, seed=train_seed + step)
                update = _bootstrap_basin_update(raw, basin_vec, random_gen=gen if run_kind == "CTRL-RandomMatchedBootstrapBasinFU" else None)
                raw_basin_cosine_sum += float(update.diagnostics.get("raw_basin_cosine", 0.0) if update.diagnostics else 0.0)
                injection_count += 1
                injection_norm_sum += float(torch.linalg.vector_norm(update.tensor.detach().float()).item())
                apply_update(model, update, lr=float(args.fu_lr))
                train_loss_gain_sum += before - _loss_value(model, xb, yb)
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    out: dict[str, Any] = {
        "run_kind": run_kind,
        "certificate_family": "C-O10_Bootstrap_Basin_Consensus_Certificate",
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "BBC_score": source_row.get("BBC_score", ""),
        "bootstrap_steps": int(args.bootstrap_steps),
        "bootstrap_replicas": int(args.bootstrap_replicas),
        "bootstrap_pair_cosine_mean": basin["bootstrap_pair_cosine_mean"],
        "bootstrap_consensus_norm_ratio": basin["bootstrap_consensus_norm_ratio"],
        "refresh_interval": int(args.refresh_interval),
        "injection_count": injection_count,
        "injection_norm_sum": injection_norm_sum,
        "raw_basin_cosine_mean": raw_basin_cosine_sum / max(1, injection_count),
        "train_loss_gain_mean": train_loss_gain_sum / max(1, injection_count),
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
    candidates = [r for r in enriched if str(r.get("run_kind")) == "BBC-AdamWPlusBootstrapBasinFU"]
    summary: list[dict[str, Any]] = []
    for row in candidates:
        debt_ok = int(_f(row.get("CEp99_h3200"), 0.0) <= 20.0 and _f(row.get("ECE_h3200"), 0.0) <= 1.0 and _f(row.get("Brier_h3200"), 0.0) <= 1.0)
        source_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        summary.append(
            {
                "v21_id": row.get("v21_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "BBC_score": row.get("BBC_score", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h2400": row.get("source_vs_best_control_h2400", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "bootstrap_pair_cosine_mean": row.get("bootstrap_pair_cosine_mean", ""),
                "bootstrap_consensus_norm_ratio": row.get("bootstrap_consensus_norm_ratio", ""),
                "raw_basin_cosine_mean": row.get("raw_basin_cosine_mean", ""),
                "train_loss_gain_mean": row.get("train_loss_gain_mean", ""),
                "injection_count": row.get("injection_count", ""),
                "injection_norm_sum": row.get("injection_norm_sum", ""),
                "debt_not_exploded": debt_ok,
                "BBC_fresh_C3_bootstrap_basin_pass": int(source_pass and debt_ok),
                "official_C3_pass": 0,
                "blocker": "" if source_pass and debt_ok else "source_horizon_or_debt_gate_failed",
            }
        )
    positives = {h: sum(1 for r in summary if _f(r.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005) for h in HORIZONS}
    route = {
        "fresh_rows": len(summary),
        "fresh_C3_bootstrap_basin_pass_rows": sum(int_flag(r.get("BBC_fresh_C3_bootstrap_basin_pass")) for r in summary),
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
    prev_route = read_json(out_dir / "v22_08_post_nogo_aug_tangent_route.json")
    rows = _candidate_rows(args)
    raw: list[dict[str, Any]] = []
    basin_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        try:
            raw.append(_probe_one(args, row, idx, basin_cache))
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
            "BBC-AdamWPlusBootstrapBasinFU",
            "CTRL-AdamW",
            "CTRL-SGD",
            "CTRL-NoOp",
            "CTRL-RandomMatchedBootstrapBasinFU",
        ]:
            fresh_rows.append(_train_fresh(args, row, run_kind))
    fresh_enriched, fresh_pack = _fresh_summary(fresh_rows)
    fresh_summary = fresh_pack["summary"]
    fresh_route = fresh_pack["route"]
    route = {
        "route": "PostNoGoBootstrapBasinFreshC3Opened" if int(fresh_route["fresh_C3_bootstrap_basin_pass_rows"]) else "PostNoGoBootstrapBasinBlocked",
        "previous_post_nogo_route": prev_route.get("route", ""),
        "certificate_family": "C-O10_Bootstrap_Basin_Consensus_Certificate",
        "probe_rows": len(matrix),
        "probe_measured_rows": len(scored),
        "BBC_train_only_certificate_pass_rows": summary.get("BBC_train_only_certificate_pass_rows", 0),
        "BBC_official_observer_pass": summary.get("BBC_official_observer_pass", 0),
        "fresh_C3_bootstrap_basin_pass_rows": fresh_route["fresh_C3_bootstrap_basin_pass_rows"],
        "official_C3_pass_rows": 0,
        "promotion_allowed": 0,
        "blocker": "official_observer_gate_or_fresh_C3_source_horizon_failed",
        "next_codex_action": "C-O10 failed under train-only bootstrap basin consensus; require external theory or mark C-O1..C-O10 local boundary",
    }
    write_rows(out_dir / "v22_08_post_nogo_bootstrap_basin_matrix.csv", matrix)
    write_rows(out_dir / "v22_08_post_nogo_bootstrap_basin_summary.csv", [summary])
    write_rows(out_dir / "v22_08_post_nogo_bootstrap_basin_selected_candidates.csv", selected)
    write_rows(out_dir / "v22_08_post_nogo_bootstrap_basin_fresh_c3_matrix.csv", fresh_enriched)
    write_rows(out_dir / "v22_08_post_nogo_bootstrap_basin_fresh_c3_summary.csv", fresh_summary)
    write_json(out_dir / "v22_08_post_nogo_bootstrap_basin_route.json", route)
    simple_svg(out_dir / "figures/v22_08_post_nogo_bbc_score.svg", "v22.08 post-no-go BBC score", scored, "BBC_score")
    simple_svg(out_dir / "figures/v22_08_post_nogo_bbc_h3200.svg", "v22.08 post-no-go BBC h3200", fresh_summary, "source_vs_best_control_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_post_nogo_bootstrap_basin_certificate.py --source-dir {args.source_dir} --device {args.device} --fresh-top-k {int(args.fresh_top_k)} --steps {int(args.steps)} --bootstrap-steps {int(args.bootstrap_steps)} --bootstrap-replicas {int(args.bootstrap_replicas)} --refresh-interval {int(args.refresh_interval)} --out-dir {out_dir}",
        status="completed",
        note=f"probe_rows={len(matrix)} train_only_pass={summary.get('BBC_train_only_certificate_pass_rows', 0)} fresh_C3_pass={fresh_route['fresh_C3_bootstrap_basin_pass_rows']} route={route['route']}",
    )


if __name__ == "__main__":
    main()
