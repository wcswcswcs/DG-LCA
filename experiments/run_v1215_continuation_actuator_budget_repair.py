#!/usr/bin/env python3
"""v12.15 continuation: safety-capped actuator budget repair.

This is a v12.15 continuation experiment, not a new version. It follows the
v12.15 post-route recommendation after B15 and Line D both failed:

* test whether stronger role-specific primitive actuators can stop being weak;
* keep the functional direction generator loss-agnostic;
* cap update size with unlabeled train-batch logit drift, not CE labels;
* audit higher-level estimator features from the real measured responses.

The script does not open P4 and does not promote a P3 survivor by itself.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation as v15


DEFAULT_ROOT = REPO_ROOT / "results" / "v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation"
DEFAULT_OFFICIAL_DIR = DEFAULT_ROOT / "official_3x3_b32_w5"
EPS = 1.0e-12


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_floats(text: Any) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def logit_drift_on_batch(v1252: Any, torch_mod: Any, model: Any, deltas: Sequence[Any], x: Any) -> float:
    trial = copy.deepcopy(model)
    v1252._apply_delta(trial, deltas, 1.0)
    with torch_mod.no_grad():
        before = model(x)
        after = trial(x)
    return float((after - before).abs().max().item())


def safety_cap_delta(
    v1252: Any,
    torch_mod: Any,
    model: Any,
    deltas: Sequence[Any],
    x_safe: Any,
    target_logit: float,
) -> tuple[list[Any], float, float, float]:
    requested_drift = logit_drift_on_batch(v1252, torch_mod, model, deltas, x_safe)
    if requested_drift <= 0.0:
        return list(deltas), requested_drift, 1.0, requested_drift
    scale = min(1.0, float(target_logit) / max(EPS, requested_drift))
    capped = v15.scale_delta(deltas, scale)
    return capped, requested_drift, scale, requested_drift * scale


def combine_roles(model: Any, raw: Sequence[Any], task_delta: Sequence[Any], roles: Sequence[tuple[str, float]]) -> tuple[list[Any], dict[str, Any]]:
    parts: list[tuple[float, Sequence[Any]]] = []
    role_names = []
    for role, weight in roles:
        role_delta, _stats = v15.filter_delta_by_role(model, raw, task_delta, role)
        parts.append((float(weight), role_delta))
        role_names.append(f"{weight:g}*{role}")
    return v15.add_deltas(parts, task_delta), {"role_recipe": "+".join(role_names)}


def random_role_param_delta(
    torch_mod: Any,
    model: Any,
    task_delta: Sequence[Any],
    role: str,
    seed: int,
    mode: str,
) -> tuple[list[Any], dict[str, Any]]:
    tokens = v15.ROLE_TOKENS.get(role, v15.ROLE_TOKENS.get("all", ()))
    gen = torch_mod.Generator(device=next(model.parameters()).device).manual_seed(int(seed))
    deltas = []
    active_tensors = 0
    active_params = 0
    for (name, param), ref in zip(model.named_parameters(), task_delta):
        if any(token in name for token in tokens):
            noise = torch_mod.randn(param.shape, device=param.device, dtype=param.dtype, generator=gen)
            if mode == "balanced":
                noise = noise / noise.norm().clamp_min(EPS) * param.detach().float().norm().clamp_min(EPS).to(dtype=param.dtype)
            elif mode == "sign":
                noise = noise.sign() * param.detach().abs().mean().clamp_min(EPS)
            deltas.append(noise)
            active_tensors += 1
            active_params += int(param.numel())
        else:
            deltas.append(torch_mod.zeros_like(ref))
    deltas = v15.normalize_like(deltas, task_delta)
    return deltas, {
        "role_recipe": f"primitive_param_{mode}:{role}",
        "primitive_param_mode": mode,
        "primitive_param_role": role,
        "primitive_param_active_tensors": active_tensors,
        "primitive_param_active_params": active_params,
    }


def make_repair_base_updates(v14: Any, torch_mod: Any, model: Any, xb: Any, task_delta: Sequence[Any], args: argparse.Namespace, seed: int) -> dict[str, dict[str, Any]]:
    raw_specs = {
        # v12.14 cotangent construction assumes count fits the local logit
        # cotangent space. Keep the continuation count aligned with v12.15.
        "low_lift_mixed": (32, "mixed", "low_lift", ""),
        "top_damp_orthogonal": (32, "orthogonal_rademacher", "top_damp", ""),
        "isotropic_whitened": (32, "whitened", "isotropic", ""),
    }
    raws: dict[str, tuple[Sequence[Any], Mapping[str, Any]]] = {}
    for raw_id, (count, kind, objective, role) in raw_specs.items():
        raw, stats, _sketch = v14.cotangent_spectral_direction(
            torch_mod,
            model,
            xb,
            task_delta,
            float(args.probe_lr),
            int(args.sketch_dim),
            int(seed) + v15.stable_seed(raw_id),
            int(count),
            str(kind),
            str(objective),
            str(role),
            False,
        )
        raws[raw_id] = (raw, stats)

    recipes: list[tuple[str, str, Sequence[tuple[str, float]], str]] = [
        ("J1-QuadBudgetCapLowLift", "low_lift_mixed", [("quad", 1.0)], "quad_budget_repair"),
        ("J2-ProjectionBudgetCapLowLift", "low_lift_mixed", [("projection", 1.0)], "projection_budget_repair"),
        ("J3-ProjectionQuadBudgetCapLowLift", "low_lift_mixed", [("projection", 1.0), ("quad", 1.0)], "projection_quad_budget_repair"),
        ("J4-QuadBranchBudgetCapLowLift", "low_lift_mixed", [("quad", 1.0), ("branch", 1.0)], "quad_branch_budget_repair"),
        ("J5-AllRoleBudgetCapLowLift", "low_lift_mixed", [("all", 1.0)], "all_role_budget_repair"),
        ("J6-QuadTopDampBudgetCap", "top_damp_orthogonal", [("quad", 1.0)], "quad_top_damp_budget_repair"),
        ("J7-ProjectionQuadTopDampBudgetCap", "top_damp_orthogonal", [("projection", 1.0), ("quad", 1.0)], "projection_quad_top_damp_budget_repair"),
        ("J8-QuadIsotropicWhitenedBudgetCap", "isotropic_whitened", [("quad", 1.0)], "quad_isotropic_budget_repair"),
    ]
    out: dict[str, dict[str, Any]] = {}
    for repair_id, raw_id, roles, family in recipes:
        raw, raw_stats = raws[raw_id]
        deltas, recipe_stats = combine_roles(model, raw, task_delta, roles)
        out[repair_id] = {
            "deltas": deltas,
            "repair_family": family,
            "source_raw_direction": raw_id,
            "loss_agnostic_direction": 1,
            "ce_vector_used_for_direction": 0,
            "label_used_for_direction": 0,
            "permuted_label_used_for_direction": 0,
            "validation_used_for_commit": 0,
            "dataset_name_used_for_commit": 0,
            **dict(raw_stats),
            **recipe_stats,
        }
    fused_specs = [
        ("K1-FusedQuadParamNoiseBudgetCap", "quad", "noise", "fused_quad_param_budget_repair"),
        ("K2-FusedProjectionParamNoiseBudgetCap", "projection", "noise", "fused_projection_param_budget_repair"),
        ("K3-FusedProjectionQuadParamNoiseBudgetCap", "direct_quad", "noise", "fused_projection_quad_param_budget_repair"),
        ("K4-FusedQuadBranchParamNoiseBudgetCap", "quad_branch", "noise", "fused_quad_branch_param_budget_repair"),
        ("K5-FusedAllActiveBalancedBudgetCap", "all", "balanced", "fused_all_active_balanced_budget_repair"),
        ("K6-FusedQuadSignBudgetCap", "quad", "sign", "fused_quad_sign_budget_repair"),
    ]
    for repair_id, role, mode, family in fused_specs:
        deltas, stats = random_role_param_delta(torch_mod, model, task_delta, role, int(seed) + v15.stable_seed(repair_id), mode)
        out[repair_id] = {
            "deltas": deltas,
            "repair_family": family,
            "source_raw_direction": "primitive_param_space",
            "loss_agnostic_direction": 1,
            "ce_vector_used_for_direction": 0,
            "label_used_for_direction": 0,
            "permuted_label_used_for_direction": 0,
            "validation_used_for_commit": 0,
            "dataset_name_used_for_commit": 0,
            **stats,
        }
    return out


def classify_actuator(metrics: Mapping[str, Any]) -> tuple[int, str]:
    sketch = v15.safe_float(metrics.get("sketch_delta_fro"))
    angle = max(v15.safe_float(metrics.get("signal_projector_angle")), v15.safe_float(metrics.get("reservoir_projector_angle")))
    logit = v15.safe_float(metrics.get("logit_max_abs_drift"))
    cep99 = v15.safe_float(metrics.get("CEp99_delta_audit"))
    noise = v15.safe_float(metrics.get("NoiseSignalLeak_delta"))
    reservoir = v15.safe_float(metrics.get("RealSignalReservoirRatio_delta"))
    gate = int(sketch >= 0.01 and angle >= 1.0 and logit <= 0.05 and cep99 <= 0.05)
    if logit > 0.05 or cep99 > 0.05:
        return gate, "SafetyRejected"
    if sketch < 0.01 and angle < 1.0:
        return gate, "ActuatorWeak"
    if gate and noise > -0.01 and reservoir > -0.01:
        return gate, "ActuatorGateOpenButObjectiveMisaligned"
    if sketch >= 0.01 or angle >= 1.0:
        return gate, "ActuatorPartiallyImproved"
    return gate, "ActuatorWeak"


def summarize_repair_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("repair_id", ""))].append(row)
    summary: list[dict[str, Any]] = []
    for repair_id, items in sorted(groups.items()):
        values = lambda key: [v15.safe_float(item.get(key)) for item in items]
        modes: dict[str, int] = defaultdict(int)
        for item in items:
            modes[str(item.get("classification_failure_mode", ""))] += 1
        summary.append(
            {
                "stage": "V1215_CONTINUATION_ACTUATOR_BUDGET_SUMMARY",
                "repair_id": repair_id,
                "repair_family": items[0].get("repair_family", ""),
                "rows": len(items),
                "actuator_gate_pass_rows": sum(v15.safe_int(item.get("actuator_gate_pass")) for item in items),
                "mean_safety_cap_scale": float(np.mean(values("safety_cap_scale"))) if items else 0.0,
                "mean_logit_max_abs_drift": float(np.mean(values("logit_max_abs_drift"))) if items else 0.0,
                "max_logit_max_abs_drift": max(values("logit_max_abs_drift") or [0.0]),
                "mean_sketch_delta_fro": float(np.mean(values("sketch_delta_fro"))) if items else 0.0,
                "max_sketch_delta_fro": max(values("sketch_delta_fro") or [0.0]),
                "mean_projector_angle_deg": float(np.mean([max(v15.safe_float(item.get("signal_projector_angle_deg")), v15.safe_float(item.get("reservoir_projector_angle_deg"))) for item in items])) if items else 0.0,
                "max_projector_angle_deg": max([max(v15.safe_float(item.get("signal_projector_angle_deg")), v15.safe_float(item.get("reservoir_projector_angle_deg"))) for item in items] or [0.0]),
                "best_noise_delta": min(values("NoiseSignalLeak_delta") or [0.0]),
                "best_reservoir_delta": min(values("RealSignalReservoirRatio_delta") or [0.0]),
                "mean_CouplingR2_delta": float(np.mean(values("CouplingR2_delta"))) if items else 0.0,
                "failure_mode_counts": json.dumps(dict(sorted(modes.items())), sort_keys=True),
            }
        )
    return summary


def common_response_row(row: Mapping[str, Any], source: str) -> dict[str, Any]:
    method_id = row.get("probe_update_id") or row.get("repair_id") or row.get("actuator_id") or ""
    family = row.get("probe_update_family") or row.get("repair_family") or row.get("role_recipe") or ""
    out = {
        "source": source,
        "method_id": method_id,
        "method_family": family,
        "dataset": row.get("dataset", ""),
        "seed": row.get("seed", ""),
        "loss_agnostic_direction": row.get("loss_agnostic_direction", 1),
        "label_used_for_direction": row.get("label_used_for_direction", 0),
        "ce_vector_used_for_direction": row.get("ce_vector_used_for_direction", 0),
        "validation_used_for_commit": row.get("validation_used_for_commit", 0),
        "dataset_name_used_for_commit": row.get("dataset_name_used_for_commit", 0),
    }
    numeric_cols = [
        "functional_norm_ratio",
        "role_direct_norm",
        "role_quad_norm",
        "role_branch_norm",
        "role_projection_norm",
        "role_readout_norm",
        "logit_drift_l2",
        "logit_max_abs_drift",
        "sketch_delta_fro",
        "sketch_delta_op",
        "sketch_top_eigen_delta",
        "sketch_eff_rank_delta",
        "signal_projector_angle",
        "reservoir_projector_angle",
        "signal_projector_angle_deg",
        "reservoir_projector_angle_deg",
        "signal_eigen_mass_delta",
        "reservoir_fraction_delta",
        "budget_multiplier",
        "requested_logit_drift_train",
        "safety_cap_scale",
        "capped_logit_drift_train",
        "CouplingR2_delta",
        "NoiseSignalLeak_delta",
        "RealSignalReservoirRatio_delta",
    ]
    for col in numeric_cols:
        out[col] = v15.safe_float(row.get(col))
    if out["signal_projector_angle"] == 0.0 and out["signal_projector_angle_deg"] != 0.0:
        out["signal_projector_angle"] = out["signal_projector_angle_deg"]
    if out["reservoir_projector_angle"] == 0.0 and out["reservoir_projector_angle_deg"] != 0.0:
        out["reservoir_projector_angle"] = out["reservoir_projector_angle_deg"]
    return out


def add_engineered_features(row: Mapping[str, Any]) -> dict[str, float]:
    sketch = v15.safe_float(row.get("sketch_delta_fro"))
    op = v15.safe_float(row.get("sketch_delta_op"))
    signal_angle = v15.safe_float(row.get("signal_projector_angle"))
    reservoir_angle = v15.safe_float(row.get("reservoir_projector_angle"))
    direct = v15.safe_float(row.get("role_direct_norm"))
    quad = v15.safe_float(row.get("role_quad_norm"))
    branch = v15.safe_float(row.get("role_branch_norm"))
    projection = v15.safe_float(row.get("role_projection_norm"))
    readout = v15.safe_float(row.get("role_readout_norm"))
    roles = np.array([max(0.0, direct), max(0.0, quad), max(0.0, branch), max(0.0, projection), max(0.0, readout)], dtype=float)
    total = float(np.sum(roles))
    probs = roles / max(EPS, total)
    entropy = float(-np.sum([p * math.log(max(EPS, p)) for p in probs]))
    angle_max = max(signal_angle, reservoir_angle)
    angle_sum = signal_angle + reservoir_angle
    return {
        "log1p_sketch_delta": math.log1p(max(0.0, sketch)),
        "sketch_op_ratio": op / max(EPS, sketch),
        "angle_max": angle_max,
        "angle_sum": angle_sum,
        "sketch_angle_product": sketch * angle_max,
        "quad_minus_direct": quad - direct,
        "projection_plus_quad": projection + quad,
        "branch_plus_readout": branch + readout,
        "role_entropy": entropy,
        "drift_per_norm": v15.safe_float(row.get("logit_max_abs_drift")) / max(EPS, v15.safe_float(row.get("functional_norm_ratio"))),
        "budget_after_cap": v15.safe_float(row.get("budget_multiplier")) * v15.safe_float(row.get("safety_cap_scale"), 1.0),
    }


def rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def corr_np(a: Sequence[float], b: Sequence[float], kind: str) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if kind == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    if float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float) -> np.ndarray:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma < EPS] = 1.0
    x0 = (train_x - mu) / sigma
    xt = (test_x - mu) / sigma
    x0 = np.concatenate([np.ones((x0.shape[0], 1)), x0], axis=1)
    xt = np.concatenate([np.ones((xt.shape[0], 1)), xt], axis=1)
    reg = np.eye(x0.shape[1]) * float(ridge)
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(x0.T @ x0 + reg) @ x0.T @ train_y
    return xt @ beta


def estimator_cv_rows(source_rows: Sequence[Mapping[str, Any]], ridge: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        if v15.safe_int(source_row.get("loss_agnostic_direction"), 1) != 1:
            continue
        if v15.safe_int(source_row.get("label_used_for_direction"), 0) != 0:
            continue
        method_id = str(source_row.get("method_id", ""))
        if method_id.startswith("U0-NoOp"):
            continue
        enriched = dict(source_row)
        enriched.update(add_engineered_features(source_row))
        feature_rows.append(enriched)

    base_features = [
        "functional_norm_ratio",
        "role_direct_norm",
        "role_quad_norm",
        "role_branch_norm",
        "role_projection_norm",
        "logit_drift_l2",
        "logit_max_abs_drift",
        "sketch_delta_fro",
        "sketch_top_eigen_delta",
        "signal_projector_angle",
        "reservoir_projector_angle",
    ]
    higher_features = base_features + [
        "sketch_delta_op",
        "signal_eigen_mass_delta",
        "reservoir_fraction_delta",
        "log1p_sketch_delta",
        "sketch_op_ratio",
        "angle_max",
        "angle_sum",
        "sketch_angle_product",
        "quad_minus_direct",
        "projection_plus_quad",
        "branch_plus_readout",
        "role_entropy",
        "drift_per_norm",
    ]
    safety_features = higher_features + [
        "budget_multiplier",
        "requested_logit_drift_train",
        "safety_cap_scale",
        "capped_logit_drift_train",
        "budget_after_cap",
    ]
    feature_sets = {
        "base_v1215_univariate_style_features": base_features,
        "higher_level_sketch_role_features": higher_features,
        "safety_budget_augmented_features": safety_features,
    }
    targets = ["CouplingR2_delta", "NoiseSignalLeak_delta", "RealSignalReservoirRatio_delta"]
    fold_specs = {
        "leave_dataset_out": sorted({str(row.get("dataset")) for row in feature_rows if row.get("dataset") != ""}),
        "leave_method_family_out": sorted({str(row.get("method_family")) for row in feature_rows if row.get("method_family") != ""}),
    }
    for target in targets:
        for feature_set_id, features in feature_sets.items():
            for fold_type, fold_values in fold_specs.items():
                y_all: list[float] = []
                pred_all: list[float] = []
                fold_count = 0
                for fold_value in fold_values:
                    if fold_type == "leave_dataset_out":
                        test_mask = [str(row.get("dataset")) == fold_value for row in feature_rows]
                    else:
                        test_mask = [str(row.get("method_family")) == fold_value for row in feature_rows]
                    train = [row for row, is_test in zip(feature_rows, test_mask) if not is_test]
                    test = [row for row, is_test in zip(feature_rows, test_mask) if is_test]
                    if len(train) < 8 or len(test) < 2:
                        continue
                    train_x = np.asarray([[v15.safe_float(row.get(feature)) for feature in features] for row in train], dtype=float)
                    train_y = np.asarray([v15.safe_float(row.get(target)) for row in train], dtype=float)
                    test_x = np.asarray([[v15.safe_float(row.get(feature)) for feature in features] for row in test], dtype=float)
                    pred = ridge_predict(train_x, train_y, test_x, ridge)
                    y_all.extend([v15.safe_float(row.get(target)) for row in test])
                    pred_all.extend([float(x) for x in pred])
                    fold_count += 1
                if not y_all:
                    continue
                y = np.asarray(y_all, dtype=float)
                pred = np.asarray(pred_all, dtype=float)
                ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
                ss_res = float(np.sum((y - pred) ** 2))
                hard_actual = y <= -0.01
                hard_pred = pred <= -0.01
                tp = int(np.sum(hard_actual & hard_pred))
                fp = int(np.sum(~hard_actual & hard_pred))
                fn = int(np.sum(hard_actual & ~hard_pred))
                near_threshold = float(np.quantile(y, 0.20))
                near_actual = y <= near_threshold
                near_pred = pred <= float(np.quantile(pred, 0.20))
                ntp = int(np.sum(near_actual & near_pred))
                nfp = int(np.sum(~near_actual & near_pred))
                nfn = int(np.sum(near_actual & ~near_pred))
                rows.append(
                    {
                        "stage": "V1215_CONTINUATION_ESTIMATOR_CV",
                        "model_id": "ridge_standardized",
                        "feature_set": feature_set_id,
                        "feature_count": len(features),
                        "metric_target": target,
                        "fold_type": fold_type,
                        "fold_count": fold_count,
                        "rows": len(y_all),
                        "dataset_name_used_as_feature": 0,
                        "label_or_ce_vector_used_as_feature": 0,
                        "spearman_corr_oos": corr_np(y, pred, "spearman"),
                        "pearson_corr_oos": corr_np(y, pred, "pearson"),
                        "r2_oos": 1.0 - ss_res / max(EPS, ss_tot),
                        "sign_accuracy_oos": float(np.mean((pred <= 0.0) == (y <= 0.0))),
                        "hard_release_threshold": -0.01,
                        "hard_release_actual_support": int(np.sum(hard_actual)),
                        "hard_release_pred_support": int(np.sum(hard_pred)),
                        "hard_release_precision": tp / max(1, tp + fp),
                        "hard_release_recall": tp / max(1, tp + fn),
                        "diagnostic_bottom20_threshold": near_threshold,
                        "diagnostic_bottom20_precision": ntp / max(1, ntp + nfp),
                        "diagnostic_bottom20_recall": ntp / max(1, ntp + nfn),
                    }
                )
    return rows, feature_rows


def route_payload(out_dir: Path, repair_rows: Sequence[Mapping[str, Any]], estimator_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    gate_pass = sum(v15.safe_int(row.get("actuator_gate_pass")) for row in repair_rows)
    max_sketch = max([v15.safe_float(row.get("sketch_delta_fro")) for row in repair_rows] or [0.0])
    max_angle = max([max(v15.safe_float(row.get("signal_projector_angle_deg")), v15.safe_float(row.get("reservoir_projector_angle_deg"))) for row in repair_rows] or [0.0])
    min_noise = min([v15.safe_float(row.get("NoiseSignalLeak_delta")) for row in repair_rows] or [0.0])
    min_res = min([v15.safe_float(row.get("RealSignalReservoirRatio_delta")) for row in repair_rows] or [0.0])
    max_noise_spearman = max([abs(v15.safe_float(row.get("spearman_corr_oos"))) for row in estimator_rows if row.get("metric_target") == "NoiseSignalLeak_delta"] or [0.0])
    max_res_spearman = max([abs(v15.safe_float(row.get("spearman_corr_oos"))) for row in estimator_rows if row.get("metric_target") == "RealSignalReservoirRatio_delta"] or [0.0])
    hard_support_noise = max([v15.safe_int(row.get("hard_release_actual_support")) for row in estimator_rows if row.get("metric_target") == "NoiseSignalLeak_delta"] or [0])
    hard_support_res = max([v15.safe_int(row.get("hard_release_actual_support")) for row in estimator_rows if row.get("metric_target") == "RealSignalReservoirRatio_delta"] or [0])
    if gate_pass > 0:
        route = "R4-ContinuationActuatorGateOpenedButP3NotClaimed"
        next_action = "build a new B15 continuation candidate only from gate-open actuators, then run P3"
    else:
        route = "R4-ContinuationActuatorBudgetRepairStillWeak"
        next_action = "do not repeat budget scaling; redesign Line C sketch construction or implement deeper primitive fused actuator"
    return {
        "stage": "V1215_CONTINUATION_ROUTE",
        "generated_at": now_iso(),
        "run_id": out_dir.name,
        "route": route,
        "p4_open": 0,
        "official_p3_claim": 0,
        "repair_rows": len(repair_rows),
        "actuator_gate_pass_rows": gate_pass,
        "max_repair_sketch_delta_fro": max_sketch,
        "max_repair_projector_angle_deg": max_angle,
        "best_repair_noise_delta": min_noise,
        "best_repair_reservoir_delta": min_res,
        "max_noise_estimator_abs_spearman_oos": max_noise_spearman,
        "max_reservoir_estimator_abs_spearman_oos": max_res_spearman,
        "hard_noise_release_actual_support": hard_support_noise,
        "hard_reservoir_release_actual_support": hard_support_res,
        "target_logit_drift_train": float(args.target_logit_drift),
        "budget_multipliers": str(args.budget_multipliers),
        "source_official_dir": str(args.source_official_dir),
        "no_fake_proxy_cpu": 1,
        "next_recommended_action": next_action,
    }


def run_main(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    base_mod = v15.v1213()
    v14 = v15.v1214()
    prev = base_mod.load_v1211_runner()
    torch_mod, F_mod, v120, v124, v1252, v1283 = prev._lazy_probe_modules()
    device = v1283._device_from_arg(str(args.probe_device))
    if str(device).startswith("cuda"):
        torch_mod.cuda.set_device(device.index if device.index is not None else 0)

    datasets = v15.parse_list(args.probe_datasets)
    seeds = v15.parse_ints(args.probe_seeds)
    windows = [v15.safe_int(item) for item in v15.parse_list(args.windows)]
    budgets = parse_floats(args.budget_multipliers)
    batch_size = int(args.functional_batch_size)
    repair_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        canon = v120._canonical_dataset(dataset)
        load_args = argparse.Namespace(
            data_root=args.data_root,
            no_download=bool(args.no_download),
            seed=seeds[0] if seeds else 0,
            train_size=int(args.probe_train_size),
            val_size=int(args.probe_val_size),
            test_size=int(args.probe_test_size),
            datasets=canon,
        )
        data = v120._load_vision_split(load_args, canon, train_size=int(args.probe_train_size), val_size=int(args.probe_val_size), test_size=int(args.probe_test_size))
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = data[0], data[1], data[2], data[3]
        input_dim = int(data[6])
        output_dim = int(data[7])
        _, budget = v124._param_budget(input_dim, output_dim)
        specs = {spec.candidate_id: spec for spec in v1283.prim.primitive_specs(budget, input_dim, output_dim)}
        for seed in seeds:
            x_train = x_train_cpu.to(device=device, dtype=torch_mod.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch_mod.float32)
            y_val = y_val_cpu.to(device=device)
            base = v1283._make_model(v15.B320_ID, input_dim, output_dim, x_train, device, int(seed) + 1215500, specs, y_train)
            xq = x_val[:batch_size]
            yq = y_val[:batch_size]
            x_safe = x_train[:batch_size]
            for split_id in range(int(args.probe_splits)):
                start = (split_id * batch_size) % max(1, int(x_train.shape[0]) - batch_size + 1)
                xb = x_train[start : start + batch_size]
                yb = y_train[start : start + batch_size]
                task_delta = v1252._grad_delta(base, xb, yb, float(args.probe_lr))
                seed_base = v15.stable_seed(canon, seed, split_id, "v1215_continuation")
                base_updates = make_repair_base_updates(v14, torch_mod, base, xb, task_delta, args, seed_base)
                for window in windows:
                    for repair_id, update in base_updates.items():
                        for budget_multiplier in budgets:
                            requested = v15.scale_delta(update["deltas"], float(window) * float(budget_multiplier))
                            capped, requested_drift, cap_scale, capped_drift = safety_cap_delta(
                                v1252,
                                torch_mod,
                                base,
                                requested,
                                x_safe,
                                float(args.target_logit_drift),
                            )
                            metrics = v15.evaluate_direction_full(
                                prev,
                                torch_mod,
                                F_mod,
                                v1252,
                                v1283,
                                base,
                                capped,
                                task_delta,
                                xb,
                                yb,
                                xq,
                                yq,
                                seed_base + v15.stable_seed(repair_id, window, budget_multiplier),
                                args,
                            )
                            gate, mode = classify_actuator(metrics)
                            repair_rows.append(
                                {
                                    "stage": "V1215_CONTINUATION_ACTUATOR_BUDGET_REPAIR",
                                    "run_id": out_dir.name,
                                    "repair_id": repair_id,
                                    "repair_family": update.get("repair_family", ""),
                                    "source_raw_direction": update.get("source_raw_direction", ""),
                                    "role_recipe": update.get("role_recipe", ""),
                                    "primitive_param_mode": update.get("primitive_param_mode", ""),
                                    "primitive_param_role": update.get("primitive_param_role", ""),
                                    "primitive_param_active_tensors": update.get("primitive_param_active_tensors", ""),
                                    "primitive_param_active_params": update.get("primitive_param_active_params", ""),
                                    "dataset": canon,
                                    "seed": seed,
                                    "split_id": split_id,
                                    "window": window,
                                    "budget_multiplier": budget_multiplier,
                                    "batch_size": batch_size,
                                    "target_logit_drift_train": float(args.target_logit_drift),
                                    "requested_logit_drift_train": requested_drift,
                                    "safety_cap_scale": cap_scale,
                                    "capped_logit_drift_train": capped_drift,
                                    "loss_agnostic_direction": update.get("loss_agnostic_direction", 1),
                                    "ce_vector_used_for_direction": update.get("ce_vector_used_for_direction", 0),
                                    "label_used_for_direction": update.get("label_used_for_direction", 0),
                                    "permuted_label_used_for_direction": update.get("permuted_label_used_for_direction", 0),
                                    "validation_used_for_commit": update.get("validation_used_for_commit", 0),
                                    "dataset_name_used_for_commit": update.get("dataset_name_used_for_commit", 0),
                                    "functional_norm_ratio": metrics.get("functional_norm_ratio", ""),
                                    "role_direct_norm": metrics.get("role_direct_norm", ""),
                                    "role_quad_norm": metrics.get("role_quad_norm", ""),
                                    "role_branch_norm": metrics.get("role_branch_norm", ""),
                                    "role_projection_norm": metrics.get("role_projection_norm", ""),
                                    "role_readout_norm": metrics.get("role_readout_norm", ""),
                                    "logit_drift_l2": metrics.get("logit_drift_l2", ""),
                                    "logit_max_abs_drift": metrics.get("logit_max_abs_drift", ""),
                                    "sketch_delta_fro": metrics.get("sketch_delta_fro", ""),
                                    "sketch_delta_op": metrics.get("sketch_delta_op", ""),
                                    "sketch_top_eigen_delta": metrics.get("sketch_top_eigen_delta", ""),
                                    "sketch_eff_rank_delta": metrics.get("sketch_eff_rank_delta", ""),
                                    "signal_projector_angle_deg": metrics.get("signal_projector_angle", ""),
                                    "reservoir_projector_angle_deg": metrics.get("reservoir_projector_angle", ""),
                                    "signal_eigen_mass_delta": metrics.get("signal_eigen_mass_delta", ""),
                                    "reservoir_fraction_delta": metrics.get("reservoir_fraction_delta", ""),
                                    "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                                    "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                                    "CouplingR2_delta": metrics.get("CouplingR2_delta", ""),
                                    "CEp99_delta_audit": metrics.get("CEp99_delta_audit", ""),
                                    "ECE_delta_audit": metrics.get("ECE_delta_audit", ""),
                                    "holdout_loss_ratio_audit": metrics.get("holdout_loss_ratio_audit", ""),
                                    "actuator_gate_pass": gate,
                                    "classification_failure_mode": mode,
                                }
                            )

    summary_rows = summarize_repair_rows(repair_rows)
    source_linec_rows = [
        common_response_row(row, "v1215_official_linec")
        for row in read_csv_rows(Path(args.source_official_dir) / "v1215_linec_response_identifiability.csv")
    ]
    continuation_response_rows = [common_response_row(row, "v1215_continuation_actuator_budget_repair") for row in repair_rows]
    estimator_rows, estimator_feature_rows = estimator_cv_rows(source_linec_rows + continuation_response_rows, float(args.ridge_lambda_estimator))
    route = route_payload(out_dir, repair_rows, estimator_rows, args)

    write_csv_rows(out_dir / "v1215_continuation_actuator_budget_repair.csv", repair_rows)
    write_csv_rows(out_dir / "v1215_continuation_actuator_budget_summary.csv", summary_rows)
    write_csv_rows(out_dir / "v1215_continuation_estimator_feature_table.csv", estimator_feature_rows)
    write_csv_rows(out_dir / "v1215_continuation_estimator_cv.csv", estimator_rows)
    with (out_dir / "v1215_continuation_route_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(route, handle, indent=2, sort_keys=True)

    manifest = {
        name: v15.sha256_file(out_dir / name)
        for name in [
            "v1215_continuation_actuator_budget_repair.csv",
            "v1215_continuation_actuator_budget_summary.csv",
            "v1215_continuation_estimator_feature_table.csv",
            "v1215_continuation_estimator_cv.csv",
            "v1215_continuation_route_decision.json",
        ]
        if (out_dir / name).exists()
    }
    with (out_dir / "v1215_continuation_hash_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return route


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-device", default="auto")
    parser.add_argument("--probe-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--probe-seeds", default="0,1,2")
    parser.add_argument("--probe-splits", type=int, default=1)
    parser.add_argument("--windows", default="5")
    parser.add_argument("--functional-batch-size", type=int, default=32)
    parser.add_argument("--probe-train-size", type=int, default=512)
    parser.add_argument("--probe-val-size", type=int, default=256)
    parser.add_argument("--probe-test-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=0.01)
    parser.add_argument("--sketch-dim", type=int, default=12)
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    parser.add_argument("--ridge-lambda-estimator", type=float, default=1.0e-3)
    parser.add_argument("--budget-multipliers", default="2,4,8")
    parser.add_argument("--target-logit-drift", type=float, default=0.045)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--source-official-dir", default=str(DEFAULT_OFFICIAL_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_ROOT / "continuation_actuator_budget_repair_3x3_b32_w5"))
    parser.add_argument("--fresh", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route = run_main(args, out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
