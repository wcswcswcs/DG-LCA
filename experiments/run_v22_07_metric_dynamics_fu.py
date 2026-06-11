#!/usr/bin/env python3
"""v22.07 metric-as-dynamics FU C0/C1/C2/C3 audits.

C0/C1 reconstruct the initial train-stream state and call make_update without
committing it.  Future source labels are read only after the no-commit rows are
materialized, as audit labels for estimator/target quality.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from dgkan.fu.core import UpdateTensor, apply_update, cosine, flat_grad, flat_params, load_flat_params, normalized_like  # noqa: E402
from dgkan.fu.mechanisms import CONTROL_MECHANISMS, make_update, poprisk_snr_update_with_diagnostics  # noqa: E402
from experiments.run_v17_common import carrier_model, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_01_source_retention import stable_train_seed  # noqa: E402
from experiments.run_v22_07_common import (  # noqa: E402
    PYTHON,
    V2206_COMBINED_SOURCE,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


HORIZONS = (100, 400, 800, 1600, 2400, 3200)
ESTIMATORS = {
    "E0_loss_cotangent_magnitude": "E0 current loss/update magnitude",
    "E1_train_split_B2_transfer": "E1 train split B1/B2 transfer gain",
    "E2_gradient_drift_snr": "E2 per-example gradient drift/diffusion SNR",
    "E3_signal_reservoir_projection": "E3 signal-channel/reservoir projection",
    "E4_class_balanced_density": "E4 class-balanced early source density",
    "E5_hidden_readout_decomposition": "E5 hidden/readout source decomposition",
    "E6_optimizer_conflict_score": "E6 optimizer-state conflict score",
    "E7_low_NDS_score": "E7 low-NDS score",
    "E8_info_volume_no_fold": "E8 information-volume/no-fold score",
    "E9_KAN_basis_channel_score": "E9 KAN low-degree/low-frequency channel source score",
}


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
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--include-controls", type=int, default=1)
    return p


def _filtered_matrix(source_dir: Path, include_controls: int) -> list[dict[str, str]]:
    matrix = read_rows(source_dir / "v21_01_source_retention_matrix.csv")
    if not matrix:
        matrix = read_rows(source_dir / "v22_06_metric_solver_raw_matrix.csv")
    rows = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in matrix:
        v21_id = str(row.get("v21_id", ""))
        is_control = v21_id.startswith("CTRL") or str(row.get("mechanism", "")) in CONTROL_MECHANISMS
        if not (v21_id.startswith("MLP-V2206") or (include_controls and is_control)):
            continue
        if str(row.get("carrier", "MLP")) != "MLP":
            continue
        key = (
            v21_id,
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("init_seed_offset", "0")),
            str(row.get("source_artifact", row.get("run_label", ""))),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _chunks(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    a = max(2, n // 3)
    b = max(a + 2, 2 * n // 3)
    return x[:a], y[:a], x[a:b], y[a:b], x[b:], y[b:]


def _losses(model: torch.nn.Module, xa: torch.Tensor, ya: torch.Tensor, xb: torch.Tensor, yb: torch.Tensor, xc: torch.Tensor, yc: torch.Tensor, ycorr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        la = F.cross_entropy(model(xa).float(), ya, reduction="none")
        lb = F.cross_entropy(model(xb).float(), yb, reduction="none") if int(xb.shape[0]) else la[:0]
        lc = F.cross_entropy(model(xc).float(), yc, reduction="none") if int(xc.shape[0]) else la[:0]
        lcorrupt = F.cross_entropy(model(xc).float(), ycorr, reduction="none") if int(xc.shape[0]) else la[:0]
    return la, lb, lc, lcorrupt


def _simulate_gains(model: torch.nn.Module, update: UpdateTensor, lr: float, x: torch.Tensor, y: torch.Tensor, classes: int) -> dict[str, float]:
    xa, ya, xb, yb, xc, yc = _chunks(x, y)
    corrupt = (yc + 1) % max(2, classes)
    before = flat_params(model).detach().clone()
    ba, bb, bc, bcorrupt = _losses(model, xa, ya, xb, yb, xc, yc, corrupt)
    apply_update(model, update, lr=lr)
    aa, ab, ac, acorrupt = _losses(model, xa, ya, xb, yb, xc, yc, corrupt)
    load_flat_params(model, before)

    def gain(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
        if not int(lhs.numel()) or not int(rhs.numel()):
            return 0.0
        return float((lhs - rhs).mean().item())

    return {
        "B1_gain": gain(ba, aa),
        "B2_transfer_gain": gain(bb, ab),
        "B3_safety_gain": gain(bc, ac),
        "corrupt_gain": gain(bcorrupt, acorrupt),
    }


def _make_update_with_grad(model: torch.nn.Module, mechanism: str, x: torch.Tensor, y: torch.Tensor, seed: int) -> UpdateTensor:
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x).float(), y).backward()
    return make_update(model, mechanism, x, y, seed=seed)


def _safe_diag(update: UpdateTensor) -> dict[str, Any]:
    diag = dict(update.diagnostics or {})
    diag.update(update.to_metadata())
    return diag


def _rank(values: list[float]) -> list[float]:
    pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(pairs):
        end = pos + 1
        while end < len(pairs) and pairs[end][0] == pairs[pos][0]:
            end += 1
        avg = 0.5 * (pos + end - 1) + 1.0
        for _v, idx in pairs[pos:end]:
            ranks[idx] = avg
        pos = end
    return ranks


def _auc(scores: list[float], labels: list[int]) -> float | str:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos <= 0 or neg <= 0:
        return ""
    ranks = _rank(scores)
    pos_rank = sum(r for r, y in zip(ranks, labels) if y)
    return (pos_rank - pos * (pos + 1) / 2.0) / float(pos * neg)


def _spearman(scores: list[float], values: list[float]) -> float | str:
    if len(scores) < 3:
        return ""
    rx = _rank(scores)
    ry = _rank(values)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return ""
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(vx * vy)


def _precision_recall_topk(scores: list[float], labels: list[int], k: int = 20) -> tuple[float | str, float | str]:
    if not scores:
        return "", ""
    kk = min(k, len(scores))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:kk]
    hits = sum(labels[i] for i in order)
    total_pos = sum(labels)
    return hits / kk if kk else "", hits / total_pos if total_pos else ""


def _source_value(row: dict[str, Any], horizon: int) -> float:
    return finite_float(row.get(f"source_h{horizon}"))


def _no_commit_rows(args: argparse.Namespace, matrix_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    device = resolve_device(args.device)
    out_c0: list[dict[str, Any]] = []
    out_c1: list[dict[str, Any]] = []
    out_c2: list[dict[str, Any]] = []
    max_jobs = int(args.max_jobs)
    selected = matrix_rows[:max_jobs] if max_jobs > 0 else matrix_rows
    for idx, row in enumerate(selected):
        start = time.perf_counter()
        try:
            dataset = str(row.get("dataset", "MNIST"))
            seed = int(float(row.get("seed", 0) or 0))
            local = deepcopy(args)
            local.classes = int(args.classes)
            local.hidden = int(args.hidden)
            local.param_budget = int(args.param_budget)
            local.input_size = int(args.input_size)
            local.basis_repair_variant = str(row.get("basis_repair_variant", "R0-current") or "R0-current")
            x_train, y_train, _x_val, _y_val = load_dataset(dataset, Path(args.data_root), int(args.train_size), int(args.val_size), seed, device, int(args.input_size))
            train_seed = int(float(row.get("train_seed") or 0)) if str(row.get("train_seed", "")).strip() else stable_train_seed(row)
            model = carrier_model("MLP", x_train, train_seed, local, device)
            xb = x_train[: min(int(args.batch_size), int(x_train.shape[0]))]
            yb = y_train[: int(xb.shape[0])]
            mechanism = str(row.get("mechanism", ""))
            update = _make_update_with_grad(model, mechanism, xb, yb, seed=train_seed + 1)
            diag = _safe_diag(update)
            grad = flat_grad(model, device).detach()
            snr_vec, snr_diag = poprisk_snr_update_with_diagnostics(model, xb, yb, max_examples=min(8, int(xb.shape[0])))
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb).float(), yb).backward()
            true_gains = _simulate_gains(model, update, float(args.fu_lr), xb, yb, int(args.classes))

            gen = torch.Generator(device=device).manual_seed(train_seed + 77_007)
            noise = torch.randn(update.tensor.shape, device=device, generator=gen)
            random_update = UpdateTensor(
                normalized_like(noise, update.tensor.detach()),
                "matched_random_target",
                "subtract",
                "function_target_control",
                "v22_07_random_matched_no_commit",
                "CTRL-RandomMatchedTarget",
                one_step_descent_claim=0,
            )
            sign_update = UpdateTensor(
                update.tensor.detach().clone(),
                "sign_flip_target",
                "add" if update.sign_rule == "subtract" else "subtract",
                update.space,
                "v22_07_sign_flip_no_commit",
                "CTRL-SignFlipTarget",
                one_step_descent_claim=0,
            )
            corrupt_y = (yb + 1) % max(2, int(args.classes))
            corrupt_update = _make_update_with_grad(model, mechanism, xb, corrupt_y, seed=train_seed + 99)
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb).float(), yb).backward()
            random_gains = _simulate_gains(model, random_update, float(args.fu_lr), xb, yb, int(args.classes))
            sign_gains = _simulate_gains(model, sign_update, float(args.fu_lr), xb, yb, int(args.classes))
            corrupt_gains = _simulate_gains(model, corrupt_update, float(args.fu_lr), xb, yb, int(args.classes))

            update_norm = finite_float(diag.get("update_norm"), 0.0)
            grad_norm = float(torch.linalg.vector_norm(grad.float()).item()) if grad.numel() else 0.0
            nds = finite_float(diag.get("NDS"), finite_float(row.get("NDS_h100"), 0.0))
            source_channel = finite_float(diag.get("source_channel_projection"), finite_float(row.get("source_channel_projection_h100"), 0.0))
            reservoir = finite_float(diag.get("reservoir_projection"), finite_float(row.get("reservoir_projection_h100"), 0.0))
            leakage = finite_float(diag.get("source_to_reservoir_leakage"), finite_float(row.get("source_to_reservoir_leakage_h100"), 1.0))
            hidden_fraction = finite_float(diag.get("hidden_source_fraction"), finite_float(row.get("hidden_source_fraction_h100"), 0.0))
            readout_fraction = finite_float(diag.get("readout_source_fraction"), finite_float(row.get("readout_source_fraction_h100"), 0.0))
            condition = finite_float(diag.get("condition_estimate"), finite_float(row.get("condition_estimate_h100"), 0.0))
            rank = finite_float(diag.get("source_subspace_rank"), finite_float(row.get("source_subspace_rank_h100"), 0.0))
            fold = finite_float(diag.get("fold_rate"), finite_float(row.get("fold_rate"), 0.0))
            basis_score = (
                finite_float(diag.get("low_degree_source_energy"), finite_float(row.get("low_degree_source_energy_h100"), 0.0))
                + finite_float(diag.get("low_frequency_source_energy"), finite_float(row.get("low_frequency_source_energy_h100"), 0.0))
            )
            grad_cos = cosine(update.tensor.detach(), grad)
            scores = {
                "E0_loss_cotangent_magnitude": update_norm * max(0.0, grad_norm),
                "E1_train_split_B2_transfer": true_gains["B2_transfer_gain"],
                "E2_gradient_drift_snr": finite_float(snr_diag.get("PopRisk_SNR_score"), 0.0),
                "E3_signal_reservoir_projection": source_channel - reservoir - leakage,
                "E4_class_balanced_density": finite_float(diag.get("target_early_observable_class_density"), finite_float(row.get("target_early_observable_class_density_h100"), 0.0)),
                "E5_hidden_readout_decomposition": hidden_fraction - readout_fraction,
                "E6_optimizer_conflict_score": -grad_cos,
                "E7_low_NDS_score": -nds,
                "E8_info_volume_no_fold": (rank / (1.0 + max(0.0, condition))) - fold,
                "E9_KAN_basis_channel_score": basis_score,
            }
            precommit_score = scores["E1_train_split_B2_transfer"] + scores["E3_signal_reservoir_projection"] + scores["E7_low_NDS_score"]
            base = {
                "audit_source": "v22.07 no-commit reconstruction + v22.06 future source readback",
                "job_order": idx,
                "v21_id": row.get("v21_id", ""),
                "mechanism": mechanism,
                "dataset": dataset,
                "seed": seed,
                "train_seed": train_seed,
                "target_family": diag.get("target_family", row.get("target_family_h100", "")),
                "metric_family": diag.get("metric_family", row.get("metric_family_h100", "")),
                "solver_level": diag.get("solver_level", row.get("solver_level_h100", "")),
                "is_control": int(str(row.get("v21_id", "")).startswith("CTRL") or mechanism in CONTROL_MECHANISMS),
                "precommit_score": precommit_score,
                "make_update_wall_ms": (time.perf_counter() - start) * 1000.0,
                "future_audit_source_h100": row.get("source_h100", ""),
                "future_audit_source_h400": row.get("source_h400", ""),
                "future_audit_source_h800": row.get("source_h800", ""),
                "future_audit_source_h1600": row.get("source_h1600", ""),
                "future_audit_source_h2400": row.get("source_h2400", ""),
                "future_audit_source_h3200": row.get("source_h3200", ""),
                "source_h800_positive": int(_source_value(row, 800) >= 0.005),
                "source_h3200_positive": int(_source_value(row, 3200) >= 0.005),
                "execution_status": "measured",
                "blocker": "",
            }
            for key, value in scores.items():
                base[key] = value
            out_c0.append(base)

            c1 = {
                **base,
                "target_norm": update_norm,
                "target_L2_energy": diag.get("metric_energy_L2", row.get("metric_energy_L2_h100", "")),
                "target_Fisher_energy": diag.get("metric_energy_Fisher", row.get("metric_energy_Fisher_h100", "")),
                "target_Sobolev_energy": diag.get("metric_energy_Sobolev", row.get("metric_energy_Sobolev_h100", "")),
                "target_RKHS_energy": diag.get("metric_energy_RKHS", row.get("metric_energy_RKHS_h100", "")),
                "target_NDS": nds,
                "target_info_volume": rank,
                "target_neighbor_preservation": 1.0 - fold,
                "target_signal_projection": source_channel,
                "target_reservoir_projection": reservoir,
                "B1_gain": true_gains["B1_gain"],
                "B2_transfer_gain": true_gains["B2_transfer_gain"],
                "B3_safety_gain": true_gains["B3_safety_gain"],
                "random_target_B2": random_gains["B2_transfer_gain"],
                "sign_flip_B2": sign_gains["B2_transfer_gain"],
                "corrupt_target_B2": corrupt_gains["B2_transfer_gain"],
                "random_target_gap": true_gains["B2_transfer_gain"] - random_gains["B2_transfer_gain"],
                "sign_flip_gap": true_gains["B2_transfer_gain"] - sign_gains["B2_transfer_gain"],
                "corrupt_gap": true_gains["B2_transfer_gain"] - corrupt_gains["B2_transfer_gain"],
                "C1_pass": int(
                    true_gains["B2_transfer_gain"] >= random_gains["B2_transfer_gain"] + 0.005
                    and true_gains["B2_transfer_gain"] >= sign_gains["B2_transfer_gain"] + 0.005
                    and true_gains["B2_transfer_gain"] >= corrupt_gains["B2_transfer_gain"] + 0.005
                    and true_gains["B3_safety_gain"] >= -0.005
                ),
            }
            if not int(c1["C1_pass"]):
                c1["C1_blocker"] = "target_not_above_random_sign_corrupt_or_B3_safety"
            out_c1.append(c1)

            c2_quality = int(
                finite_float(diag.get("projection_residual_Gf"), 999.0) <= 0.35
                and finite_float(diag.get("ActuationR2"), -999.0) >= 0.60
                and true_gains["B2_transfer_gain"] >= 0.005
                and true_gains["B3_safety_gain"] >= -0.005
                and int(c1["C1_pass"])
            )
            c2 = {
                **base,
                "projection_residual_Gf": diag.get("projection_residual_Gf", row.get("projection_residual_Gf_h100", "")),
                "ActuationR2": diag.get("ActuationR2", row.get("ActuationR2_h100", "")),
                "ActuationCosine": diag.get("ActuationCosine", row.get("ActuationCosine_h100", "")),
                "B1_gain": true_gains["B1_gain"],
                "B2_transfer_gain": true_gains["B2_transfer_gain"],
                "B3_safety_gain": true_gains["B3_safety_gain"],
                "solve_time_ms": diag.get("solve_time_ms", row.get("solve_time_ms_h100", "")),
                "make_update_wall_ms": base["make_update_wall_ms"],
                "JVP_count": diag.get("JVP_count", row.get("JVP_count_h100", "")),
                "VJP_count": diag.get("VJP_count", row.get("VJP_count_h100", "")),
                "CG_iterations": diag.get("CG_iterations", row.get("CG_iterations_h100", "")),
                "solver_rank": diag.get("solver_rank", row.get("solver_rank_h100", "")),
                "condition_estimate": condition,
                "metric_energy_L2": c1["target_L2_energy"],
                "metric_energy_Fisher": c1["target_Fisher_energy"],
                "metric_energy_Sobolev": c1["target_Sobolev_energy"],
                "metric_energy_RKHS": c1["target_RKHS_energy"],
                "NDS": nds,
                "info_volume": rank,
                "fold_proxy": fold,
                "function_displacement_cosine": diag.get("function_displacement_cosine", ""),
                "parameter_update_cosine": grad_cos,
                "C2_solver_quality_pass": c2_quality,
                "C2_runtime_pass": "",
                "C2_pass": c2_quality,
                "C2_blocker": "" if c2_quality else "projection_or_actuation_or_C1_or_B2_B3_gate_failed",
            }
            out_c2.append(c2)
        except Exception as exc:
            base = {
                "job_order": idx,
                "v21_id": row.get("v21_id", ""),
                "mechanism": row.get("mechanism", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "execution_status": f"blocked:{type(exc).__name__}",
                "blocker": str(exc)[:500],
            }
            out_c0.append(base)
            out_c1.append(dict(base))
            out_c2.append(dict(base))
    return out_c0, out_c1, out_c2


def _c0_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    measured = [r for r in rows if str(r.get("execution_status")) == "measured"]
    for key, desc in ESTIMATORS.items():
        valid = [(finite_float(r.get(key)), int_flag(r.get("source_h800_positive")), int_flag(r.get("source_h3200_positive")), finite_float(r.get("future_audit_source_h800")), finite_float(r.get("future_audit_source_h3200")), int_flag(r.get("is_control"))) for r in measured]
        valid = [v for v in valid if math.isfinite(v[0])]
        if not valid:
            out.append({"estimator": key, "description": desc, "rows": 0, "C0_pass": 0, "blocker": "no_finite_scores"})
            continue
        scores = [v[0] for v in valid]
        h800_labels = [v[1] for v in valid]
        h3200_labels = [v[2] for v in valid]
        h800_values = [v[3] for v in valid if math.isfinite(v[3])]
        h3200_values = [v[4] for v in valid if math.isfinite(v[4])]
        h800_scores = [v[0] for v in valid if math.isfinite(v[3])]
        h3200_scores = [v[0] for v in valid if math.isfinite(v[4])]
        p20, r20 = _precision_recall_topk(scores, h800_labels, 20)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: min(20, len(scores))]
        control_equiv = sum(valid[i][5] for i in top) / len(top) if top else ""
        pass_flag = int(finite_float(_auc(scores, h800_labels), -1.0) >= 0.70 and finite_float(p20, -1.0) >= 0.50 and finite_float(control_equiv, 1.0) <= 0.50)
        out.append(
            {
                "estimator": key,
                "description": desc,
                "rows": len(valid),
                "AUC_predict_h800_positive": _auc(scores, h800_labels),
                "AUC_predict_h3200_positive": _auc(scores, h3200_labels),
                "Spearman_score_vs_source_h800": _spearman(h800_scores, h800_values),
                "Spearman_score_vs_source_h3200": _spearman(h3200_scores, h3200_values),
                "precision_at_top20": p20,
                "recall_at_top20": r20,
                "control_equivalent_fraction": control_equiv,
                "C0_pass": pass_flag,
                "blocker": "" if pass_flag else "auc_or_precision_or_control_equivalent_gate_failed",
            }
        )
    return out


def _group_mean(rows: list[dict[str, Any]], key: str) -> float | str:
    vals = [finite_float(r.get(key)) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else ""


def _target_summary(c1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in c1_rows:
        if str(row.get("execution_status")) == "measured":
            buckets[(str(row.get("target_family", "")), str(row.get("metric_family", "")))].append(row)
    out = []
    for (target, metric), group in sorted(buckets.items()):
        c1_pass_rows = sum(int_flag(r.get("C1_pass")) for r in group)
        out.append(
            {
                "target_family": target,
                "metric_family": metric,
                "rows": len(group),
                "C1_pass_rows": c1_pass_rows,
                "C1_family_pass": int(c1_pass_rows > 0),
                "B2_transfer_gain_mean": _group_mean(group, "B2_transfer_gain"),
                "B3_safety_gain_mean": _group_mean(group, "B3_safety_gain"),
                "random_target_gap_mean": _group_mean(group, "random_target_gap"),
                "sign_flip_gap_mean": _group_mean(group, "sign_flip_gap"),
                "corrupt_gap_mean": _group_mean(group, "corrupt_gap"),
            }
        )
    return out


def _c2_summary(c2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in c2_rows:
        if str(row.get("execution_status")) == "measured":
            buckets[(str(row.get("solver_level", "")), str(row.get("target_family", "")), str(row.get("metric_family", "")))].append(row)
    out = []
    for (solver, target, metric), group in sorted(buckets.items()):
        pass_rows = sum(int_flag(r.get("C2_pass")) for r in group)
        out.append(
            {
                "solver_level": solver,
                "target_family": target,
                "metric_family": metric,
                "rows": len(group),
                "C2_pass_rows": pass_rows,
                "C2_family_pass": int(pass_rows > 0),
                "ActuationR2_mean": _group_mean(group, "ActuationR2"),
                "projection_residual_Gf_mean": _group_mean(group, "projection_residual_Gf"),
                "B2_transfer_gain_mean": _group_mean(group, "B2_transfer_gain"),
                "solve_time_ms_mean": _group_mean(group, "solve_time_ms"),
                "make_update_wall_ms_mean": _group_mean(group, "make_update_wall_ms"),
            }
        )
    return out


def _source_formation(matrix_rows: list[dict[str, str]], c1_rows: list[dict[str, Any]], c2_rows: list[dict[str, Any]], c0_pass: int) -> list[dict[str, Any]]:
    c1_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    c2_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in c1_rows:
        c1_by_id[str(r.get("v21_id", ""))].append(r)
    for r in c2_rows:
        c2_by_id[str(r.get("v21_id", ""))].append(r)
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in matrix_rows:
        if str(row.get("v21_id", "")).startswith("MLP-V2206"):
            buckets[str(row.get("v21_id", ""))].append(row)
    out = []
    for v21_id, group in sorted(buckets.items()):
        item: dict[str, Any] = {"v21_id": v21_id, "rows": len(group), "mechanism": group[0].get("mechanism", "") if group else ""}
        for h in HORIZONS:
            vals = [finite_float(r.get(f"source_h{h}")) for r in group]
            vals = [v for v in vals if math.isfinite(v)]
            item[f"source_vs_best_control_h{h}"] = sum(vals) / len(vals) if vals else ""
            item[f"row_positive_count_h{h}"] = sum(1 for v in vals if v >= 0.005)
        s800 = finite_float(item.get("source_vs_best_control_h800"), -999.0)
        s1600 = finite_float(item.get("source_vs_best_control_h1600"), -999.0)
        s3200 = finite_float(item.get("source_vs_best_control_h3200"), -999.0)
        item["retention_h1600_over_h800"] = max(0.0, s1600) / max(1.0e-12, s800) if s800 > 0 else ""
        item["retention_h3200_over_h1600"] = max(0.0, s3200) / max(1.0e-12, s1600) if s1600 > 0 else ""
        item["C1_pass_rows"] = sum(int_flag(r.get("C1_pass")) for r in c1_by_id.get(v21_id, []))
        item["C2_pass_rows"] = sum(int_flag(r.get("C2_pass")) for r in c2_by_id.get(v21_id, []))
        item["entered_after_C0C1C2"] = int(c0_pass and int(item["C1_pass_rows"]) > 0 and int(item["C2_pass_rows"]) > 0)
        item["LineC_debt_final_h3200"] = _group_mean(group, "LineC_fast_loss_h3200")
        item["CEp99_debt_final_h3200"] = _group_mean(group, "CEp99_h3200")
        item["ECE_debt_final_h3200"] = _group_mean(group, "ECE_h3200")
        item["Brier_debt_final_h3200"] = _group_mean(group, "Brier_h3200")
        item["AUCtime_ratio"] = ""
        debt_ok = int(
            finite_float(item.get("CEp99_debt_final_h3200"), 0.0) <= 20.0
            and finite_float(item.get("ECE_debt_final_h3200"), 0.0) <= 1.0
            and finite_float(item.get("Brier_debt_final_h3200"), 0.0) <= 1.0
        )
        c3_pass = int(
            item["entered_after_C0C1C2"]
            and all(finite_float(item.get(f"source_vs_best_control_h{h}"), -999.0) >= 0.005 for h in [100, 400, 800, 1600, 3200])
            and int(item.get("row_positive_count_h3200", 0)) >= 6
            and debt_ok
        )
        item["debt_not_exploded"] = debt_ok
        item["C3_source_formation_pass"] = c3_pass
        if not c3_pass:
            blockers = []
            if not item["entered_after_C0C1C2"]:
                blockers.append("C0_C1_C2_gate_not_all_open")
            for h in [100, 400, 800, 1600, 3200]:
                if finite_float(item.get(f"source_vs_best_control_h{h}"), -999.0) < 0.005:
                    blockers.append(f"source_h{h}")
            if int(item.get("row_positive_count_h3200", 0)) < 6:
                blockers.append("row_h3200_positive_count")
            if not debt_ok:
                blockers.append("debt_exploded")
            item["C3_blocker"] = ";".join(blockers)
        out.append(item)
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    matrix_rows = _filtered_matrix(source_dir, int(args.include_controls))
    c0_rows, c1_rows, c2_rows = _no_commit_rows(args, matrix_rows)
    c0_summary = _c0_summary(c0_rows)
    c1_summary = _target_summary(c1_rows)
    c2_summary = _c2_summary(c2_rows)
    c0_pass = int(any(int_flag(r.get("C0_pass")) for r in c0_summary))
    c3_rows = _source_formation(matrix_rows, c1_rows, c2_rows, c0_pass)
    route = {
        "source_rows": len(matrix_rows),
        "C0_pass": c0_pass,
        "C1_pass_rows": sum(int_flag(r.get("C1_pass")) for r in c1_rows),
        "C2_pass_rows": sum(int_flag(r.get("C2_pass")) for r in c2_rows),
        "C3_pass_rows": sum(int_flag(r.get("C3_source_formation_pass")) for r in c3_rows),
        "best_C0_estimator": max(c0_summary, key=lambda r: finite_float(r.get("AUC_predict_h800_positive"), -1.0)).get("estimator", "") if c0_summary else "",
        "functional_route": "C3-SourceFormationOpened" if any(int_flag(r.get("C3_source_formation_pass")) for r in c3_rows) else ("C2-TrueSolverOnly" if any(int_flag(r.get("C2_pass")) for r in c2_rows) else ("C1-TargetOnly" if any(int_flag(r.get("C1_pass")) for r in c1_rows) else "C0-EstimatorBlocked")),
        "promotion_allowed": 0,
    }
    blockers = []
    if not c0_pass:
        blockers.append("C0_source_estimator_predictor_gate")
    if not route["C1_pass_rows"]:
        blockers.append("C1_target_contrast_gate")
    if not route["C2_pass_rows"]:
        blockers.append("C2_metric_solver_gate")
    if not route["C3_pass_rows"]:
        blockers.append("C3_source_formation_gate")
    route["blocker"] = ";".join(blockers)
    write_rows(out_dir / "v22_07_c0_no_commit_source_estimator_matrix.csv", c0_rows)
    write_rows(out_dir / "v22_07_c0_no_commit_source_estimator_summary.csv", c0_summary)
    write_rows(out_dir / "v22_07_c1_target_contrast_matrix.csv", c1_rows)
    write_rows(out_dir / "v22_07_c1_target_contrast_summary.csv", c1_summary)
    write_rows(out_dir / "v22_07_c2_metric_solver_matrix.csv", c2_rows)
    write_rows(out_dir / "v22_07_c2_metric_solver_summary.csv", c2_summary)
    write_rows(out_dir / "v22_07_c3_source_formation_matrix.csv", c3_rows)
    write_json(out_dir / "v22_07_metric_dynamics_route.json", route)
    simple_svg(out_dir / "figures/source_observability_predictor_auc.svg", "v22.07 C0 source estimator AUC", c0_summary, "AUC_predict_h800_positive")
    simple_svg(out_dir / "figures/metric_operator_energy_dashboard.svg", "v22.07 metric operator energy", c1_rows, "target_L2_energy")
    simple_svg(out_dir / "figures/projection_residual_vs_B2_transfer.svg", "v22.07 projection residual vs B2", c2_rows, "projection_residual_Gf")
    simple_svg(out_dir / "figures/source_trajectory_h100_to_h6400.svg", "v22.07 source trajectory", c3_rows, "source_vs_best_control_h3200")
    simple_svg(out_dir / "figures/NDS_vs_retention.svg", "v22.07 NDS vs retention", c2_rows, "NDS")
    simple_svg(out_dir / "figures/Sobolev_RKHS_energy_vs_source.svg", "v22.07 Sobolev/RKHS vs source", c1_rows, "target_Sobolev_energy")
    simple_svg(out_dir / "figures/info_volume_and_fold_proxy_trace.svg", "v22.07 info volume/fold proxy", c2_rows, "info_volume")
    simple_svg(out_dir / "figures/signal_reservoir_projection.svg", "v22.07 signal/reservoir projection", c1_rows, "target_signal_projection")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_metric_dynamics_fu.py --source-dir {source_dir} --device {args.device} --out-dir {out_dir}",
        status="completed",
        note=f"C0={route['C0_pass']} C1_rows={route['C1_pass_rows']} C2_rows={route['C2_pass_rows']} C3_rows={route['C3_pass_rows']} route={route['functional_route']}",
    )


if __name__ == "__main__":
    main()

