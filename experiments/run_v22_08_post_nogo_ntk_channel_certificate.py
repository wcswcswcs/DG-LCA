#!/usr/bin/env python3
"""v22.08 post-no-go NTK/Jacobian eigen-channel certificate.

C-O8 is a new train-only retained-source observability probe after C-O1..C-O7
fail.  It tests whether a candidate FU direction is controllable through a
stable train-margin Jacobian channel rather than through a near-null or
control-equivalent direction.  Future source labels are audit columns only.
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

from dgkan.fu.core import UpdateTensor, apply_update, cosine, normalized_like, trainable_parameters  # noqa: E402
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
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--data-root", default="data")
    p.add_argument("--train-size", type=int, default=64)
    p.add_argument("--val-size", type=int, default=48)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--jacobian-samples", type=int, default=16)
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


def _unit(x: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x.detach().float())
    if float(norm.item()) <= 1.0e-12:
        return torch.zeros_like(x)
    return x.detach().float() / norm.clamp_min(1.0e-12)


def _signed_displacement(update: UpdateTensor) -> torch.Tensor:
    sign = -1.0 if update.sign_rule == "subtract" else 1.0
    return update.tensor.detach().float() * sign


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


def _margin_jacobian(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    params = trainable_parameters(model)
    logits = model(x).float()
    with torch.no_grad():
        masked = logits.detach().clone()
        masked[torch.arange(int(x.shape[0]), device=x.device), y] = -float("inf")
        competitor = masked.argmax(dim=1)
    rows: list[torch.Tensor] = []
    for i in range(int(x.shape[0])):
        scalar = logits[i, y[i]] - logits[i, competitor[i]]
        grads = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
        chunks: list[torch.Tensor] = []
        for p, g in zip(params, grads):
            if g is None:
                chunks.append(torch.zeros_like(p).reshape(-1))
            else:
                chunks.append(g.detach().reshape(-1))
        rows.append(torch.cat(chunks).float())
    raw = torch.stack(rows, dim=0) if rows else torch.zeros(0, 0, device=x.device)
    norms = torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1.0e-12)
    unit = raw / norms
    return raw, unit


def _basis_from_jacobian(unit_jacobian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if unit_jacobian.numel() == 0:
        device = unit_jacobian.device
        return torch.zeros(0, 0, device=device), torch.zeros(0, device=device), torch.zeros(0, 0, device=device)
    centered = unit_jacobian - unit_jacobian.mean(dim=0, keepdim=True)
    if float(torch.linalg.vector_norm(centered).item()) <= 1.0e-12:
        centered = unit_jacobian
    u, s, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    return u, s, vh


def _channel_metrics(vector: torch.Tensor, raw_j: torch.Tensor, s: torch.Tensor, vh: torch.Tensor) -> dict[str, float]:
    vec = vector.detach().float().reshape(-1)
    total = float(torch.sum(vec * vec).item())
    if vh.numel() == 0 or total <= 1.0e-18:
        return {
            "jacobian_span_energy_fraction": 0.0,
            "top_eigen_channel_fraction": 0.0,
            "mid_eigen_channel_fraction": 0.0,
            "low_eigen_channel_fraction": 0.0,
            "jacobian_null_energy_fraction": 1.0,
            "jacobian_output_norm": 0.0,
            "train_margin_gain_mean": 0.0,
        }
    coeff = vh @ vec
    energy = coeff * coeff
    rank = int(energy.numel())
    low = max(1, int(round(0.20 * rank)))
    top = max(1, int(round(0.20 * rank)))
    if rank <= 2:
        mid_slice = slice(0, rank)
        top_energy = float(energy[:top].sum().item())
        low_energy = float(energy[-low:].sum().item())
        mid_energy = float(energy[mid_slice].sum().item())
    else:
        lo = min(top, rank)
        hi = max(lo, rank - low)
        top_energy = float(energy[:lo].sum().item())
        low_energy = float(energy[hi:].sum().item())
        mid_energy = float(energy[lo:hi].sum().item())
    span = float(energy.sum().item())
    projected = raw_j @ vec
    return {
        "jacobian_span_energy_fraction": span / total,
        "top_eigen_channel_fraction": top_energy / total,
        "mid_eigen_channel_fraction": mid_energy / total,
        "low_eigen_channel_fraction": low_energy / total,
        "jacobian_null_energy_fraction": max(0.0, 1.0 - span / total),
        "jacobian_output_norm": float(torch.linalg.vector_norm(projected).item()),
        "train_margin_gain_mean": float(projected.mean().item()),
    }


def _project_mid(vector: torch.Tensor, vh: torch.Tensor) -> torch.Tensor:
    if vh.numel() == 0:
        return torch.zeros_like(vector)
    rank = int(vh.shape[0])
    low = max(1, int(round(0.20 * rank)))
    top = max(1, int(round(0.20 * rank)))
    if rank <= 2:
        basis = vh
    else:
        lo = min(top, rank)
        hi = max(lo, rank - low)
        basis = vh[lo:hi] if hi > lo else vh
    coeff = basis @ vector.detach().float().reshape(-1)
    return basis.transpose(0, 1) @ coeff


def _random_like(reference: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    noise = torch.randn(reference.shape, device=reference.device, generator=gen)
    return normalized_like(noise, reference.detach())


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
    jn = min(int(args.jacobian_samples), int(x_train.shape[0]))
    xb = x_train[:batch]
    yb = y_train[:batch]
    xj = x_train[:jn]
    yj = y_train[:jn]
    mechanism = str(row.get("mechanism", ""))
    raw_j, unit_j = _margin_jacobian(model, xj, yj)
    _u, s, vh = _basis_from_jacobian(unit_j)
    raw = make_update(model, mechanism, xb, yb, seed=train_seed + 91)
    corrupt = make_update(model, mechanism, xb, (yb + 1) % max(2, int(args.classes)), seed=train_seed + 92)
    source_vec = _signed_displacement(raw)
    corrupt_vec = _signed_displacement(corrupt)
    gen = torch.Generator(device=device).manual_seed(228_808 + train_seed + job_order)
    random_vec = _random_like(raw.tensor.detach().float(), gen)
    sign_vec = -source_vec
    src = _channel_metrics(source_vec, raw_j, s, vh)
    rnd = _channel_metrics(random_vec, raw_j, s, vh)
    sgn = _channel_metrics(sign_vec, raw_j, s, vh)
    crp = _channel_metrics(corrupt_vec, raw_j, s, vh)
    channel_control = max(rnd["mid_eigen_channel_fraction"], sgn["mid_eigen_channel_fraction"], crp["mid_eigen_channel_fraction"])
    margin_control = max(rnd["train_margin_gain_mean"], sgn["train_margin_gain_mean"], crp["train_margin_gain_mean"])
    output_control = max(rnd["jacobian_output_norm"], sgn["jacobian_output_norm"], crp["jacobian_output_norm"])
    source = {h: _f(row.get(f"source_h{h}"), -999.0) for h in [100, 400, 800, 1600, 3200]}
    random_cos = abs(cosine(source_vec, random_vec))
    corrupt_cos = abs(cosine(source_vec, corrupt_vec))
    return {
        "job_order": job_order,
        "v21_id": row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "jacobian_samples": jn,
        "jacobian_rank": int(s.numel()),
        "jacobian_spectrum_top": float(s[0].item()) if s.numel() else 0.0,
        "jacobian_spectrum_tail": float(s[-1].item()) if s.numel() else 0.0,
        "source_mid_eigen_channel_fraction": src["mid_eigen_channel_fraction"],
        "source_top_eigen_channel_fraction": src["top_eigen_channel_fraction"],
        "source_low_eigen_channel_fraction": src["low_eigen_channel_fraction"],
        "source_jacobian_null_energy_fraction": src["jacobian_null_energy_fraction"],
        "source_jacobian_span_energy_fraction": src["jacobian_span_energy_fraction"],
        "source_jacobian_output_norm": src["jacobian_output_norm"],
        "source_train_margin_gain_mean": src["train_margin_gain_mean"],
        "random_mid_eigen_channel_fraction": rnd["mid_eigen_channel_fraction"],
        "sign_flip_mid_eigen_channel_fraction": sgn["mid_eigen_channel_fraction"],
        "corrupt_mid_eigen_channel_fraction": crp["mid_eigen_channel_fraction"],
        "random_train_margin_gain_mean": rnd["train_margin_gain_mean"],
        "sign_flip_train_margin_gain_mean": sgn["train_margin_gain_mean"],
        "corrupt_train_margin_gain_mean": crp["train_margin_gain_mean"],
        "mid_channel_control_gap": src["mid_eigen_channel_fraction"] - channel_control,
        "margin_control_gap": src["train_margin_gain_mean"] - margin_control,
        "output_norm_control_gap": src["jacobian_output_norm"] - output_control,
        "random_control_cosine": random_cos,
        "corrupt_control_cosine": corrupt_cos,
        "max_nonflip_control_cosine": max(random_cos, corrupt_cos),
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
        "source_mid_eigen_channel_fraction",
        "source_train_margin_gain_mean",
        "mid_channel_control_gap",
        "margin_control_gap",
        "output_norm_control_gap",
        "source_jacobian_null_energy_fraction",
        "max_nonflip_control_cosine",
    ]
    z = {key: _zscore([_f(r.get(key)) for r in rows], groups) for key in feature_keys}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        score = (
            0.35 * z["source_mid_eigen_channel_fraction"][i]
            + 0.25 * z["source_train_margin_gain_mean"][i]
            + 0.25 * z["mid_channel_control_gap"][i]
            + 0.25 * z["margin_control_gap"][i]
            + 0.15 * z["output_norm_control_gap"][i]
            - 0.20 * z["source_jacobian_null_energy_fraction"][i]
            - 0.10 * z["max_nonflip_control_cosine"][i]
        )
        train_gate = int(
            _f(row.get("source_mid_eigen_channel_fraction"), -999.0) >= 0.02
            and _f(row.get("source_jacobian_span_energy_fraction"), -999.0) >= 0.04
            and _f(row.get("source_jacobian_null_energy_fraction"), 999.0) <= 0.97
            and _f(row.get("margin_control_gap"), -999.0) >= -1.0e-3
            and _f(row.get("mid_channel_control_gap"), -999.0) >= -0.20
        )
        item = dict(row)
        item["NTKC_score"] = score
        item["NTKC_train_only_certificate_pass"] = train_gate
        item["certificate_family"] = "C-O8_NTK_Jacobian_eigen_channel_certificate"
        out.append(item)
    return sorted(out, key=lambda r: _f(r.get("NTKC_score"), -999.0), reverse=True)


def _summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    scores = [_f(r.get("NTKC_score")) for r in rows]
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
        "certificate_family": "C-O8_NTK_Jacobian_eigen_channel_certificate",
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
        "NTKC_train_only_certificate_pass_rows": sum(int_flag(r.get("NTKC_train_only_certificate_pass")) for r in rows),
        "NTKC_official_observer_pass": int(not blockers),
        "blocker": ";".join(dict.fromkeys(blockers)),
    }


def _projected_update(raw: UpdateTensor, vh: torch.Tensor, *, random_gen: torch.Generator | None = None) -> UpdateTensor:
    base = _signed_displacement(raw)
    if random_gen is not None:
        base = _random_like(base, random_gen)
    projected = _project_mid(base, vh)
    return UpdateTensor(
        tensor=normalized_like(projected, raw.tensor.detach()),
        kind="ntk_eigen_channel_projected",
        sign_rule="add",
        space="train_margin_jacobian_mid_spectrum_channel",
        source="train_only_ntk_jacobian_eigen_channel_certificate",
        mechanism="C-O8-NTKJacobianEigenChannel",
        role="mid_spectrum_margin_channel",
        one_step_descent_claim=0,
        diagnostics={"projected_norm": float(torch.linalg.vector_norm(projected.detach().float()).item())},
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
    jn = min(int(args.jacobian_samples), int(x_train.shape[0]))
    raw_j, unit_j = _margin_jacobian(model, x_train[:jn], y_train[:jn])
    _u, s, vh = _basis_from_jacobian(unit_j)
    opt_adam = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    opt_sgd = torch.optim.SGD(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(228_818 + train_seed + sum(ord(c) for c in run_kind))
    batch = min(int(args.batch_size), int(x_train.shape[0]))
    mechanism = str(source_row.get("mechanism", ""))
    horizons = {h for h in HORIZONS if h <= int(args.steps)}
    traces: dict[int, dict[str, float]] = {0: _eval(model, x_train, y_train, x_val, y_val)}
    injection_count = 0
    injection_norm_sum = 0.0
    projected_channel_fraction_sum = 0.0
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
                update = _projected_update(raw, vh, random_gen=gen if run_kind == "CTRL-RandomMatchedEigenChannelFU" else None)
                injection_count += 1
                injection_norm_sum += float(torch.linalg.vector_norm(update.tensor.detach().float()).item())
                projected_channel_fraction_sum += _channel_metrics(update.tensor.detach().float(), raw_j, s, vh)["mid_eigen_channel_fraction"]
                apply_update(model, update, lr=float(args.fu_lr))
        if step in horizons:
            traces[step] = _eval(model, x_train, y_train, x_val, y_val)
    out: dict[str, Any] = {
        "run_kind": run_kind,
        "certificate_family": "C-O8_NTK_Jacobian_eigen_channel_certificate",
        "v21_id": source_row.get("v21_id", ""),
        "mechanism": mechanism,
        "dataset": dataset,
        "seed": seed,
        "train_seed": train_seed,
        "NTKC_score": source_row.get("NTKC_score", ""),
        "jacobian_samples": jn,
        "jacobian_rank": int(s.numel()),
        "refresh_interval": int(args.refresh_interval),
        "injection_count": injection_count,
        "injection_norm_sum": injection_norm_sum,
        "projected_mid_channel_fraction_mean": projected_channel_fraction_sum / max(1, injection_count),
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
    candidates = [r for r in enriched if str(r.get("run_kind")) == "NTKC-AdamWPlusEigenChannelFU"]
    summary: list[dict[str, Any]] = []
    for row in candidates:
        debt_ok = int(_f(row.get("CEp99_h3200"), 0.0) <= 20.0 and _f(row.get("ECE_h3200"), 0.0) <= 1.0 and _f(row.get("Brier_h3200"), 0.0) <= 1.0)
        source_pass = int(all(_f(row.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200]))
        summary.append(
            {
                "v21_id": row.get("v21_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "NTKC_score": row.get("NTKC_score", ""),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100", ""),
                "source_vs_best_control_h400": row.get("source_vs_best_control_h400", ""),
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_vs_best_control_h1600": row.get("source_vs_best_control_h1600", ""),
                "source_vs_best_control_h2400": row.get("source_vs_best_control_h2400", ""),
                "source_vs_best_control_h3200": row.get("source_vs_best_control_h3200", ""),
                "projected_mid_channel_fraction_mean": row.get("projected_mid_channel_fraction_mean", ""),
                "injection_count": row.get("injection_count", ""),
                "injection_norm_sum": row.get("injection_norm_sum", ""),
                "debt_not_exploded": debt_ok,
                "NTKC_fresh_C3_eigen_channel_pass": int(source_pass and debt_ok),
                "official_C3_pass": 0,
                "blocker": "" if source_pass and debt_ok else "source_horizon_or_debt_gate_failed",
            }
        )
    positives = {h: sum(1 for r in summary if _f(r.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005) for h in HORIZONS}
    route = {
        "fresh_rows": len(summary),
        "fresh_C3_eigen_channel_pass_rows": sum(int_flag(r.get("NTKC_fresh_C3_eigen_channel_pass")) for r in summary),
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
    prev_route = read_json(out_dir / "v22_08_post_nogo_counterfactual_washout_route.json")
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
            "NTKC-AdamWPlusEigenChannelFU",
            "CTRL-AdamW",
            "CTRL-SGD",
            "CTRL-NoOp",
            "CTRL-RandomMatchedEigenChannelFU",
        ]:
            fresh_rows.append(_train_fresh(args, row, run_kind))
    fresh_enriched, fresh_pack = _fresh_summary(fresh_rows)
    fresh_summary = fresh_pack["summary"]
    fresh_route = fresh_pack["route"]
    route = {
        "route": "PostNoGoNTKChannelFreshC3Opened" if int(fresh_route["fresh_C3_eigen_channel_pass_rows"]) else "PostNoGoNTKChannelBlocked",
        "previous_post_nogo_route": prev_route.get("route", ""),
        "certificate_family": "C-O8_NTK_Jacobian_eigen_channel_certificate",
        "probe_rows": len(matrix),
        "probe_measured_rows": len(scored),
        "NTKC_train_only_certificate_pass_rows": summary.get("NTKC_train_only_certificate_pass_rows", 0),
        "NTKC_official_observer_pass": summary.get("NTKC_official_observer_pass", 0),
        "fresh_C3_eigen_channel_pass_rows": fresh_route["fresh_C3_eigen_channel_pass_rows"],
        "official_C3_pass_rows": 0,
        "promotion_allowed": 0,
        "blocker": "official_observer_gate_or_fresh_C3_source_horizon_failed",
        "next_codex_action": "C-O8 failed under train-margin Jacobian eigen-channel controllability; require external theory or mark C-O1..C-O8 local boundary",
    }
    write_rows(out_dir / "v22_08_post_nogo_ntk_channel_matrix.csv", matrix)
    write_rows(out_dir / "v22_08_post_nogo_ntk_channel_summary.csv", [summary])
    write_rows(out_dir / "v22_08_post_nogo_ntk_channel_selected_candidates.csv", selected)
    write_rows(out_dir / "v22_08_post_nogo_ntk_channel_fresh_c3_matrix.csv", fresh_enriched)
    write_rows(out_dir / "v22_08_post_nogo_ntk_channel_fresh_c3_summary.csv", fresh_summary)
    write_json(out_dir / "v22_08_post_nogo_ntk_channel_route.json", route)
    simple_svg(out_dir / "figures/v22_08_post_nogo_ntkc_score.svg", "v22.08 post-no-go NTKC score", scored, "NTKC_score")
    simple_svg(out_dir / "figures/v22_08_post_nogo_ntkc_h3200.svg", "v22.08 post-no-go NTKC h3200", fresh_summary, "source_vs_best_control_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_post_nogo_ntk_channel_certificate.py --source-dir {args.source_dir} --device {args.device} --fresh-top-k {int(args.fresh_top_k)} --steps {int(args.steps)} --jacobian-samples {int(args.jacobian_samples)} --refresh-interval {int(args.refresh_interval)} --out-dir {out_dir}",
        status="completed",
        note=f"probe_rows={len(matrix)} train_only_pass={summary.get('NTKC_train_only_certificate_pass_rows', 0)} fresh_C3_pass={fresh_route['fresh_C3_eigen_channel_pass_rows']} route={route['route']}",
    )


if __name__ == "__main__":
    main()
