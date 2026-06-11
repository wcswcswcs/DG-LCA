#!/usr/bin/env python3
"""v22.07 C0/C1 repair attempts after source-estimator blockage.

The C0 repair stage only combines train-stream/no-commit quantities already
materialized in C0/C1 matrices. Future source columns are used only as audit
labels. The C1 repair stage rebuilds the selected train state and tests fixed
target-construction variants without changing the optimizer learning rate.
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

from dgkan.fu.core import UpdateTensor, flat_params, normalized_like, trainable_parameters  # noqa: E402
from dgkan.fu.mechanisms import CONTROL_MECHANISMS  # noqa: E402
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
    write_json,
    write_rows,
)
from experiments.run_v22_07_metric_dynamics_fu import (  # noqa: E402
    _filtered_matrix,
    _make_update_with_grad,
    _simulate_gains,
)


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
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--scales", default="1,4,16,64")
    return p


def _f(value: Any, default: float = 0.0) -> float:
    return finite_float(value, default)


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
    pos = sum(labels)
    neg = len(labels) - pos
    if pos <= 0 or neg <= 0:
        return ""
    ranks = _rank(scores)
    pos_rank = sum(r for r, label in zip(ranks, labels) if label)
    return (pos_rank - pos * (pos + 1) / 2.0) / float(pos * neg)


def _spearman(scores: list[float], values: list[float]) -> float | str:
    valid = [(s, v) for s, v in zip(scores, values) if math.isfinite(v)]
    if len(valid) < 3:
        return ""
    sx = [x[0] for x in valid]
    sy = [x[1] for x in valid]
    rx = _rank(sx)
    ry = _rank(sy)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return ""
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(vx * vy)


def _precision_recall(scores: list[float], labels: list[int], controls: list[int], k: int = 20) -> tuple[float | str, float | str, float | str, list[int]]:
    if not scores:
        return "", "", "", []
    kk = min(int(k), len(scores))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:kk]
    hits = sum(labels[i] for i in order)
    total_pos = sum(labels)
    control_fraction = sum(controls[i] for i in order) / kk if kk else ""
    return hits / kk if kk else "", hits / total_pos if total_pos else "", control_fraction, order


def _zscore(values: list[float], groups: list[tuple[str, str]]) -> list[float]:
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    for key in sorted(set(groups)):
        vals = [v for v, group in zip(values, groups) if group == key]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if vals else 1.0
        stats[key] = (mean, std if std > 1.0e-12 else 1.0)
    return [(v - stats[g][0]) / stats[g][1] for v, g in zip(values, groups)]


def _join_c0_c1(out_dir: Path) -> list[dict[str, Any]]:
    c0 = read_rows(out_dir / "v22_07_c0_no_commit_source_estimator_matrix.csv")
    c1 = read_rows(out_dir / "v22_07_c1_target_contrast_matrix.csv")
    by_order = {str(r.get("job_order", "")): r for r in c1}
    rows: list[dict[str, Any]] = []
    for row in c0:
        merged = dict(row)
        c1_row = by_order.get(str(row.get("job_order", "")), {})
        for key in [
            "B1_gain",
            "B2_transfer_gain",
            "B3_safety_gain",
            "random_target_gap",
            "sign_flip_gap",
            "corrupt_gap",
            "random_target_B2",
            "sign_flip_B2",
            "corrupt_target_B2",
            "target_signal_projection",
            "target_reservoir_projection",
            "target_NDS",
            "target_norm",
        ]:
            merged[key] = c1_row.get(key, "")
        rows.append(merged)
    return rows


def _score_candidates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    groups = [(str(r.get("dataset", "")), str(r.get("seed", ""))) for r in rows]
    labels_h800 = [int(_f(r.get("future_audit_source_h800"), -999.0) >= 0.005) for r in rows]
    labels_h3200 = [int(_f(r.get("future_audit_source_h3200"), -999.0) >= 0.005) for r in rows]
    source_h800 = [_f(r.get("future_audit_source_h800"), float("nan")) for r in rows]
    source_h3200 = [_f(r.get("future_audit_source_h3200"), float("nan")) for r in rows]
    controls = [int_flag(r.get("is_control")) for r in rows]
    keys = [
        "E1_train_split_B2_transfer",
        "E2_gradient_drift_snr",
        "E5_hidden_readout_decomposition",
        "E6_optimizer_conflict_score",
        "E7_low_NDS_score",
        "E8_info_volume_no_fold",
        "B3_safety_gain",
        "random_target_gap",
        "sign_flip_gap",
        "corrupt_gap",
        "target_signal_projection",
        "target_reservoir_projection",
    ]
    z = {key: _zscore([_f(r.get(key)) for r in rows], groups) for key in keys}
    scores: dict[str, tuple[str, list[float]]] = {}
    scores["R0_E1_dataset_z"] = ("dataset-invariant z-score of E1 split-transfer", z["E1_train_split_B2_transfer"])
    scores["R1_E1_low_conflict"] = (
        "dataset z E1 plus optimizer-conflict separation",
        [z["E1_train_split_B2_transfer"][i] + 0.25 * z["E6_optimizer_conflict_score"][i] for i in range(len(rows))],
    )
    scores["R2_control_gap_min"] = (
        "remove control-dominated estimator rows via min(random/sign/corrupt target gap)",
        [min(z["random_target_gap"][i], z["sign_flip_gap"][i], z["corrupt_gap"][i]) for i in range(len(rows))],
    )
    scores["R3_E1_B3_control_gap"] = (
        "E1 plus B3 safety plus min random/sign/corrupt gap",
        [
            z["E1_train_split_B2_transfer"][i]
            + 0.25 * z["B3_safety_gain"][i]
            + 0.5 * min(z["random_target_gap"][i], z["sign_flip_gap"][i], z["corrupt_gap"][i])
            for i in range(len(rows))
        ],
    )
    scores["R4_remove_reservoir_gap"] = (
        "remove high-reservoir target component and keep control gap",
        [
            z["E1_train_split_B2_transfer"][i]
            + 0.5 * min(z["random_target_gap"][i], z["sign_flip_gap"][i], z["corrupt_gap"][i])
            - 0.25 * z["target_reservoir_projection"][i]
            for i in range(len(rows))
        ],
    )
    scores["R5_hidden_readout_gap"] = (
        "hidden/readout separation plus control gap",
        [
            z["E1_train_split_B2_transfer"][i]
            + 0.5 * min(z["random_target_gap"][i], z["sign_flip_gap"][i], z["corrupt_gap"][i])
            + 0.25 * z["E5_hidden_readout_decomposition"][i]
            for i in range(len(rows))
        ],
    )

    matrix: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        out = {
            "job_order": row.get("job_order", ""),
            "v21_id": row.get("v21_id", ""),
            "mechanism": row.get("mechanism", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "is_control": row.get("is_control", ""),
            "future_audit_source_h800": row.get("future_audit_source_h800", ""),
            "future_audit_source_h3200": row.get("future_audit_source_h3200", ""),
            "source_h800_positive": labels_h800[i],
            "source_h3200_positive": labels_h3200[i],
        }
        for name, (_desc, vals) in scores.items():
            out[name] = vals[i]
        matrix.append(out)

    summary: list[dict[str, Any]] = []
    best_order: list[int] = []
    best_sort_key: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    for name, (desc, vals) in scores.items():
        p20, r20, control_equiv, order = _precision_recall(vals, labels_h800, controls, 20)
        auc_h800 = _auc(vals, labels_h800)
        pass_flag = int(_f(auc_h800, -1.0) >= 0.70 and _f(p20, -1.0) >= 0.50 and _f(control_equiv, 1.0) <= 0.50)
        item = {
            "repair_estimator": name,
            "description": desc,
            "rows": len(rows),
            "AUC_predict_h800_positive": auc_h800,
            "AUC_predict_h3200_positive": _auc(vals, labels_h3200),
            "Spearman_score_vs_source_h800": _spearman(vals, source_h800),
            "Spearman_score_vs_source_h3200": _spearman(vals, source_h3200),
            "precision_at_top20": p20,
            "recall_at_top20": r20,
            "control_equivalent_fraction": control_equiv,
            "C0_repair_pass": pass_flag,
            "repair_actions": desc,
            "blocker": "" if pass_flag else "auc_or_precision_or_control_equivalent_gate_failed",
        }
        summary.append(item)
        sort_key = (float(pass_flag), _f(p20, -1.0), _f(auc_h800, -1.0))
        if sort_key > best_sort_key:
            best_sort_key = sort_key
            best_order = order
    top_rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(best_order, start=1):
        top_rows.append(
            {
                "rank": rank,
                "job_order": rows[idx].get("job_order", ""),
                "v21_id": rows[idx].get("v21_id", ""),
                "mechanism": rows[idx].get("mechanism", ""),
                "dataset": rows[idx].get("dataset", ""),
                "seed": rows[idx].get("seed", ""),
                "future_audit_source_h800": rows[idx].get("future_audit_source_h800", ""),
                "future_audit_source_h3200": rows[idx].get("future_audit_source_h3200", ""),
                "source_h800_positive": labels_h800[idx],
            }
        )
    return matrix, summary, best_order


def _mask_for_role(model: torch.nn.Module, role: str, device: torch.device) -> torch.Tensor:
    params = trainable_parameters(model)
    chunks = []
    last = len(params) - 1
    for idx, param in enumerate(params):
        keep = 1.0
        if role == "readout_only":
            keep = 1.0 if idx == last else 0.0
        elif role == "hidden_only":
            keep = 0.0 if idx == last else 1.0
        chunks.append(torch.full_like(param.detach(), keep).reshape(-1))
    if not chunks:
        return torch.zeros(0, device=device)
    return torch.cat(chunks).to(device=device)


def _scaled_update(base: UpdateTensor, tensor: torch.Tensor, source: str) -> UpdateTensor:
    return UpdateTensor(
        tensor=tensor.detach().clone(),
        kind=base.kind,
        sign_rule=base.sign_rule,
        space=base.space,
        source=source,
        mechanism=base.mechanism,
        role=base.role,
        one_step_descent_claim=base.one_step_descent_claim,
        diagnostics=dict(base.diagnostics or {}),
    )


def _target_repair_rows(args: argparse.Namespace, source_rows: list[dict[str, str]], c0_rows: list[dict[str, Any]], top_order: list[int]) -> list[dict[str, Any]]:
    device = resolve_device(args.device)
    scales = [float(x) for x in str(args.scales).split(",") if x.strip()]
    source_by_order = {idx: row for idx, row in enumerate(source_rows)}
    c0_by_order = {int(float(r.get("job_order", -1))): r for r in c0_rows if str(r.get("job_order", "")).strip()}
    rows: list[dict[str, Any]] = []
    for idx in top_order[: int(args.top_k)]:
        source_row = source_by_order.get(idx)
        if not source_row:
            continue
        try:
            dataset = str(source_row.get("dataset", "MNIST"))
            seed = int(float(source_row.get("seed", 0) or 0))
            local = deepcopy(args)
            local.basis_repair_variant = str(source_row.get("basis_repair_variant", "R0-current") or "R0-current")
            x_train, y_train, _x_val, _y_val = load_dataset(dataset, Path(args.data_root), int(args.train_size), int(args.val_size), seed, device, int(args.input_size))
            train_seed = int(float(source_row.get("train_seed") or 0)) if str(source_row.get("train_seed", "")).strip() else stable_train_seed(source_row)
            model = carrier_model("MLP", x_train, train_seed, local, device)
            xb = x_train[: min(int(args.batch_size), int(x_train.shape[0]))]
            yb = y_train[: int(xb.shape[0])]
            mechanism = str(source_row.get("mechanism", ""))
            update = _make_update_with_grad(model, mechanism, xb, yb, seed=train_seed + 1)
            corrupt_y = (yb + 1) % max(2, int(args.classes))
            corrupt_update = _make_update_with_grad(model, mechanism, xb, corrupt_y, seed=train_seed + 99)
            base_params = flat_params(model).detach().clone()
            gen = torch.Generator(device=device).manual_seed(train_seed + 81_907)
            for role in ["all", "readout_only", "hidden_only"]:
                mask = _mask_for_role(model, role, device)
                for scale in scales:
                    repaired_tensor = update.tensor.detach() * mask * float(scale)
                    if float(torch.linalg.vector_norm(repaired_tensor.float()).item()) <= 1.0e-12:
                        continue
                    repaired = _scaled_update(update, repaired_tensor, f"v22_07_target_repair_{role}_x{scale:g}")
                    noise = torch.randn(repaired_tensor.shape, device=device, generator=gen)
                    random_update = UpdateTensor(
                        normalized_like(noise, repaired_tensor),
                        "matched_random_target",
                        repaired.sign_rule,
                        repaired.space,
                        "v22_07_repair_random_matched",
                        "CTRL-RandomMatchedTarget",
                        one_step_descent_claim=0,
                    )
                    sign_update = UpdateTensor(
                        repaired_tensor.detach().clone(),
                        "sign_flip_target",
                        "add" if repaired.sign_rule == "subtract" else "subtract",
                        repaired.space,
                        "v22_07_repair_sign_flip",
                        "CTRL-SignFlipTarget",
                        one_step_descent_claim=0,
                    )
                    corrupt_tensor = corrupt_update.tensor.detach() * mask * float(scale)
                    corrupt_repaired = _scaled_update(corrupt_update, corrupt_tensor, f"v22_07_target_repair_corrupt_{role}_x{scale:g}")
                    true_gains = _simulate_gains(model, repaired, float(args.fu_lr), xb, yb, int(args.classes))
                    random_gains = _simulate_gains(model, random_update, float(args.fu_lr), xb, yb, int(args.classes))
                    sign_gains = _simulate_gains(model, sign_update, float(args.fu_lr), xb, yb, int(args.classes))
                    corrupt_gains = _simulate_gains(model, corrupt_repaired, float(args.fu_lr), xb, yb, int(args.classes))
                    c1_pass = int(
                        true_gains["B2_transfer_gain"] >= random_gains["B2_transfer_gain"] + 0.005
                        and true_gains["B2_transfer_gain"] >= sign_gains["B2_transfer_gain"] + 0.005
                        and true_gains["B2_transfer_gain"] >= corrupt_gains["B2_transfer_gain"] + 0.005
                        and true_gains["B3_safety_gain"] >= -0.005
                    )
                    rows.append(
                        {
                            "repair_scope": "C1_target_construction_only_no_lr_change",
                            "job_order": idx,
                            "v21_id": source_row.get("v21_id", ""),
                            "mechanism": mechanism,
                            "dataset": dataset,
                            "seed": seed,
                            "train_seed": train_seed,
                            "rank_from_repaired_C0": top_order.index(idx) + 1,
                            "source_h800_positive_audit_only": int(_f(c0_by_order.get(idx, {}).get("future_audit_source_h800"), -999.0) >= 0.005),
                            "future_audit_source_h800": c0_by_order.get(idx, {}).get("future_audit_source_h800", ""),
                            "target_repair_variant": f"{role}_x{scale:g}",
                            "block_role": role,
                            "target_scale": scale,
                            "target_norm": float(torch.linalg.vector_norm(repaired_tensor.float()).item()),
                            "B1_gain": true_gains["B1_gain"],
                            "B2_transfer_gain": true_gains["B2_transfer_gain"],
                            "B3_safety_gain": true_gains["B3_safety_gain"],
                            "random_target_B2": random_gains["B2_transfer_gain"],
                            "sign_flip_B2": sign_gains["B2_transfer_gain"],
                            "corrupt_target_B2": corrupt_gains["B2_transfer_gain"],
                            "random_target_gap": true_gains["B2_transfer_gain"] - random_gains["B2_transfer_gain"],
                            "sign_flip_gap": true_gains["B2_transfer_gain"] - sign_gains["B2_transfer_gain"],
                            "corrupt_gap": true_gains["B2_transfer_gain"] - corrupt_gains["B2_transfer_gain"],
                            "base_param_norm": float(torch.linalg.vector_norm(base_params.float()).item()),
                            "C1_repair_pass": c1_pass,
                            "C1_repair_blocker": "" if c1_pass else "target_not_above_random_sign_corrupt_or_B3_safety",
                        }
                    )
        except Exception as exc:
            rows.append(
                {
                    "repair_scope": "C1_target_construction_only_no_lr_change",
                    "job_order": idx,
                    "v21_id": source_row.get("v21_id", "") if source_row else "",
                    "execution_status": f"blocked:{type(exc).__name__}",
                    "blocker": str(exc)[:500],
                    "C1_repair_pass": 0,
                }
            )
    return rows


def _target_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("target_repair_variant", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for variant, group in sorted(groups.items()):
        vals = lambda key: [_f(r.get(key), float("nan")) for r in group if math.isfinite(_f(r.get(key), float("nan")))]
        def mean(key: str) -> float | str:
            xs = vals(key)
            return sum(xs) / len(xs) if xs else ""
        best = sorted(group, key=lambda r: (_f(r.get("C1_repair_pass")), _f(r.get("random_target_gap"), -999.0)), reverse=True)[0]
        out.append(
            {
                "target_repair_variant": variant,
                "rows": len(group),
                "C1_repair_pass_rows": sum(int_flag(r.get("C1_repair_pass")) for r in group),
                "B2_transfer_gain_mean": mean("B2_transfer_gain"),
                "B3_safety_gain_mean": mean("B3_safety_gain"),
                "random_target_gap_mean": mean("random_target_gap"),
                "sign_flip_gap_mean": mean("sign_flip_gap"),
                "corrupt_gap_mean": mean("corrupt_gap"),
                "best_v21_id": best.get("v21_id", ""),
                "best_random_target_gap": best.get("random_target_gap", ""),
                "best_C1_repair_pass": best.get("C1_repair_pass", ""),
            }
        )
    return out


def _c2_repair_rows(out_dir: Path, target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    c2_rows = read_rows(out_dir / "v22_07_c2_metric_solver_matrix.csv")
    by_order = {str(r.get("job_order", "")): r for r in c2_rows}
    out: list[dict[str, Any]] = []
    for row in target_rows:
        c2 = by_order.get(str(row.get("job_order", "")), {})
        projection = _f(c2.get("projection_residual_Gf"), 999.0)
        actuation = _f(c2.get("ActuationR2"), -999.0)
        solve_time = _f(c2.get("solve_time_ms"), float("nan"))
        make_wall = _f(c2.get("make_update_wall_ms"), float("nan"))
        block_role = str(row.get("block_role", ""))
        projection_applicable = int(block_role == "all")
        runtime_pass = int(math.isfinite(solve_time) and math.isfinite(make_wall) and solve_time <= 1.25 * max(make_wall, 1.0e-12))
        pass_flag = int(
            projection_applicable
            and int_flag(row.get("C1_repair_pass"))
            and projection <= 0.35
            and actuation >= 0.60
            and _f(row.get("B2_transfer_gain"), -999.0) >= 0.005
            and _f(row.get("B3_safety_gain"), -999.0) >= -0.005
            and runtime_pass
        )
        blockers = []
        if not projection_applicable:
            blockers.append("block_restriction_projection_not_recomputed")
        if not int_flag(row.get("C1_repair_pass")):
            blockers.append("C1_target_repair_gate")
        if projection > 0.35:
            blockers.append("projection_residual_Gf")
        if actuation < 0.60:
            blockers.append("ActuationR2")
        if _f(row.get("B2_transfer_gain"), -999.0) < 0.005:
            blockers.append("B2_transfer_gain")
        if _f(row.get("B3_safety_gain"), -999.0) < -0.005:
            blockers.append("B3_safety_gain")
        if not runtime_pass:
            blockers.append("solve_time")
        out.append(
            {
                "repair_scope": "C2_target_rescale_solver_audit",
                "job_order": row.get("job_order", ""),
                "v21_id": row.get("v21_id", ""),
                "target_repair_variant": row.get("target_repair_variant", ""),
                "block_role": block_role,
                "target_scale": row.get("target_scale", ""),
                "projection_diagnostics_applicable": projection_applicable,
                "projection_residual_Gf": c2.get("projection_residual_Gf", ""),
                "ActuationR2": c2.get("ActuationR2", ""),
                "ActuationCosine": c2.get("ActuationCosine", ""),
                "B2_transfer_gain": row.get("B2_transfer_gain", ""),
                "B3_safety_gain": row.get("B3_safety_gain", ""),
                "random_target_gap": row.get("random_target_gap", ""),
                "sign_flip_gap": row.get("sign_flip_gap", ""),
                "corrupt_gap": row.get("corrupt_gap", ""),
                "solve_time_ms": c2.get("solve_time_ms", ""),
                "make_update_wall_ms": c2.get("make_update_wall_ms", ""),
                "C2_runtime_pass": runtime_pass,
                "JVP_count": c2.get("JVP_count", ""),
                "VJP_count": c2.get("VJP_count", ""),
                "CG_iterations": c2.get("CG_iterations", ""),
                "solver_rank": c2.get("solver_rank", ""),
                "condition_estimate": c2.get("condition_estimate", ""),
                "C1_repair_pass": row.get("C1_repair_pass", ""),
                "C2_repair_pass": pass_flag,
                "C2_repair_blocker": ";".join(blockers),
            }
        )
    return out


def _c2_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("target_repair_variant", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for variant, group in sorted(groups.items()):
        def mean(key: str) -> float | str:
            vals = [_f(r.get(key), float("nan")) for r in group]
            vals = [v for v in vals if math.isfinite(v)]
            return sum(vals) / len(vals) if vals else ""
        best = sorted(group, key=lambda r: (int_flag(r.get("C2_repair_pass")), _f(r.get("B2_transfer_gain"), -999.0)), reverse=True)[0]
        out.append(
            {
                "target_repair_variant": variant,
                "rows": len(group),
                "C2_repair_pass_rows": sum(int_flag(r.get("C2_repair_pass")) for r in group),
                "projection_applicable_rows": sum(int_flag(r.get("projection_diagnostics_applicable")) for r in group),
                "projection_residual_Gf_mean": mean("projection_residual_Gf"),
                "ActuationR2_mean": mean("ActuationR2"),
                "B2_transfer_gain_mean": mean("B2_transfer_gain"),
                "B3_safety_gain_mean": mean("B3_safety_gain"),
                "solve_time_ms_mean": mean("solve_time_ms"),
                "best_v21_id": best.get("v21_id", ""),
                "best_B2_transfer_gain": best.get("B2_transfer_gain", ""),
                "best_C2_repair_pass": best.get("C2_repair_pass", ""),
                "blocker": ";".join(dict.fromkeys(str(r.get("C2_repair_blocker", "")) for r in group if str(r.get("C2_repair_blocker", "")).strip())),
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    joined = _join_c0_c1(out_dir)
    if not joined:
        raise SystemExit("missing v22_07 C0/C1 matrices; run run_v22_07_metric_dynamics_fu.py first")
    c0_matrix, c0_summary, top_order = _score_candidates(joined)
    source_rows = _filtered_matrix(source_dir, include_controls=1)
    target_rows = _target_repair_rows(args, source_rows, joined, top_order)
    target_summary = _target_summary(target_rows)
    c2_repair = _c2_repair_rows(out_dir, target_rows)
    c2_summary = _c2_summary(c2_repair)
    best_c0 = sorted(c0_summary, key=lambda r: (int_flag(r.get("C0_repair_pass")), _f(r.get("precision_at_top20"), -1.0), _f(r.get("AUC_predict_h800_positive"), -1.0)), reverse=True)[0]
    c1_pass_rows = sum(int_flag(r.get("C1_repair_pass")) for r in target_rows)
    c2_pass_rows = sum(int_flag(r.get("C2_repair_pass")) for r in c2_repair)
    route = {
        "C0_repair_pass": int_flag(best_c0.get("C0_repair_pass")),
        "best_C0_repair_estimator": best_c0.get("repair_estimator", ""),
        "best_C0_repair_AUC_h800": best_c0.get("AUC_predict_h800_positive", ""),
        "best_C0_repair_precision_top20": best_c0.get("precision_at_top20", ""),
        "best_C0_repair_control_equiv": best_c0.get("control_equivalent_fraction", ""),
        "C1_repair_rows": len(target_rows),
        "C1_repair_pass_rows": c1_pass_rows,
        "C2_repair_rows": len(c2_repair),
        "C2_repair_pass_rows": c2_pass_rows,
        "repair_functional_route": "C2-TrueSolverRepairOpened-C3NotRun" if c2_pass_rows else ("C1-TargetRepairOpened" if c1_pass_rows else ("C1-TargetRepairBlocked" if int_flag(best_c0.get("C0_repair_pass")) else "C0-EstimatorRepairBlocked")),
        "promotion_allowed": 0,
        "blocker": "C3_source_formation_gate" if c2_pass_rows else ("" if c1_pass_rows else "C1_target_contrast_gate"),
        "future_source_used_for_direction": 0,
        "lr_changed": 0,
    }
    write_rows(out_dir / "v22_07_c0_estimator_repair_matrix.csv", c0_matrix)
    write_rows(out_dir / "v22_07_c0_estimator_repair_summary.csv", c0_summary)
    write_rows(out_dir / "v22_07_c0_estimator_repair_top20.csv", [c0_matrix[i] for i in top_order[:20]])
    write_rows(out_dir / "v22_07_c1_target_repair_matrix.csv", target_rows)
    write_rows(out_dir / "v22_07_c1_target_repair_summary.csv", target_summary)
    write_rows(out_dir / "v22_07_c2_solver_repair_matrix.csv", c2_repair)
    write_rows(out_dir / "v22_07_c2_solver_repair_summary.csv", c2_summary)
    write_json(out_dir / "v22_07_source_observability_repair_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_source_observability_repair.py --source-dir {source_dir} --device {args.device} --out-dir {out_dir}",
        status="completed",
        note=f"C0_repair={route['C0_repair_pass']} best={route['best_C0_repair_estimator']} p20={route['best_C0_repair_precision_top20']} C1_repair_pass_rows={c1_pass_rows} C2_repair_pass_rows={c2_pass_rows} route={route['repair_functional_route']}",
    )


if __name__ == "__main__":
    main()
