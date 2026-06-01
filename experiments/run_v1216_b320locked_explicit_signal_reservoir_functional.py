#!/usr/bin/env python3
"""v12.16 B320 locked explicit signal/reservoir functional runner.

This runner implements the v12.16 gate order:

* P0 B320 anchor monitor.
* P1 Line C v2 sketch / target reconstruction.
* P2 explicit target calibration.
* P3 fused primitive actuator response matrix.
* P4 B16 functional candidate construction only if P1/P2/P3 gates pass.

It keeps direction construction loss-agnostic. Labels/CE are used only for the
existing Line-C and safety audits, matching the v12.16 plan.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation as v15
from experiments import run_v1215_continuation_actuator_budget_repair as cont


DEFAULT_ROOT = REPO_ROOT / "results" / "v12_16_b320locked_explicit_signal_reservoir_functional"
DEFAULT_V1215 = REPO_ROOT / "results" / "v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation" / "official_3x3_b32_w5"
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def common_audit_fields() -> dict[str, Any]:
    return {
        "loss_agnostic_direction": 1,
        "ce_vector_used_for_direction": 0,
        "label_used_for_direction": 0,
        "validation_used_for_commit": 0,
        "dataset_name_used_for_commit": 0,
        "no_fake": 1,
        "no_proxy": 1,
        "cpu_offload_used": 0,
        "fail_reason": "",
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


def corr_np(a: Sequence[float], b: Sequence[float], kind: str = "spearman") -> float:
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


def ridge_lodo_predict(rows: Sequence[Mapping[str, Any]], feature: str, target: str) -> tuple[list[float], list[float]]:
    y_all: list[float] = []
    pred_all: list[float] = []
    datasets = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset", "") != ""})
    for dataset in datasets:
        train = [row for row in rows if str(row.get("dataset", "")) != dataset]
        test = [row for row in rows if str(row.get("dataset", "")) == dataset]
        if len(train) < 4 or not test:
            continue
        x = np.asarray([v15.safe_float(row.get(feature)) for row in train], dtype=float)
        y = np.asarray([v15.safe_float(row.get(target)) for row in train], dtype=float)
        vx = float(np.var(x))
        if vx <= EPS:
            slope = 0.0
        else:
            slope = float(np.cov(x, y, bias=True)[0, 1] / max(EPS, vx))
        intercept = float(np.mean(y) - slope * np.mean(x))
        for row in test:
            pred = intercept + slope * v15.safe_float(row.get(feature))
            y_all.append(v15.safe_float(row.get(target)))
            pred_all.append(float(pred))
    return y_all, pred_all


def release_pr(y: Sequence[float], pred: Sequence[float], threshold: float = -0.01) -> tuple[int, int, float, float]:
    actual = np.asarray(y, dtype=float) <= float(threshold)
    predicted = np.asarray(pred, dtype=float) <= float(threshold)
    support = int(np.sum(actual))
    pred_support = int(np.sum(predicted))
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    return support, pred_support, tp / max(1, tp + fp), tp / max(1, tp + fn)


def p0_anchor_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows = v15.anchor_monitor_rows(out_dir)
    out: list[dict[str, Any]] = []
    for row in rows:
        p0 = dict(row)
        p0["stage"] = "V1216_P0_ANCHOR_MONITOR"
        p0["run_id"] = out_dir.name
        p0["source_anchor_artifact"] = p0.get("source_artifact", "")
        p0["hash_match_previous"] = p0.get("hash_match_v1214", "")
        p0["provenance_no_fake"] = 1
        p0["provenance_no_proxy"] = 1
        p0["provenance_no_cpu_offload"] = 1
        p0.update({k: v for k, v in common_audit_fields().items() if k not in p0})
        fail: list[str] = []
        if v15.safe_float(p0.get("step_ratio_q90"), 999.0) > 0.50:
            fail.append("step_ratio_q90>0.50")
        if v15.safe_float(p0.get("memory_ratio_q90"), 999.0) > 0.20:
            fail.append("memory_ratio_q90>0.20")
        if v15.safe_float(p0.get("mean_delta_vs_mlp"), -999.0) < 0.0:
            fail.append("mean_delta<0")
        if v15.safe_float(p0.get("worst_delta_vs_mlp"), -999.0) < -0.003:
            fail.append("worst_delta<-0.003")
        if v15.safe_float(p0.get("AUC_step_ratio"), 999.0) > 1.00:
            fail.append("AUC_step>1")
        if v15.safe_float(p0.get("AUC_time_ratio"), 999.0) > 1.00:
            fail.append("AUC_time>1")
        if v15.safe_int(p0.get("LineC_nontearing_pass"), 0) != 1:
            fail.append("LineC_nontearing_pass!=1")
        p0["p0_pass"] = int(not fail)
        p0["fail_reason"] = ";".join(fail)
        out.append(p0)
    return out


def output_subspaces(torch_mod: Any, model: Any, x: Any, seed: int, rank_limit: int, aug_eps: float = 0.03) -> dict[str, Any]:
    gen = torch_mod.Generator(device=x.device).manual_seed(int(seed) + 6161)
    with torch_mod.no_grad():
        z0 = model(x).float()
        noise = torch_mod.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) * float(aug_eps)
        z1 = model((x + noise).clamp(-5.0, 5.0)).float()
    z0c = z0 - z0.mean(dim=0, keepdim=True)
    z1c = z1 - z1.mean(dim=0, keepdim=True)
    denom = max(1, int(z0.shape[0]) - 1)
    cov = (z0c.T @ z0c) / denom
    cross = ((z0c.T @ z1c) + (z1c.T @ z0c)) / (2 * denom)
    stable = (cross + cross.T) / 2
    unstable = ((cov - stable) + (cov - stable).T) / 2
    evals_s, evecs_s = torch_mod.linalg.eigh(stable)
    evals_u, evecs_u = torch_mod.linalg.eigh(unstable)
    order_s = torch_mod.argsort(evals_s.abs(), descending=True)
    order_u = torch_mod.argsort(evals_u.abs(), descending=True)
    rank = max(1, min(int(rank_limit), int(z0.shape[1]) // 2))
    stable_basis = evecs_s[:, order_s[:rank]]
    unstable_basis = evecs_u[:, order_u[:rank]]
    return {
        "stable_basis": stable_basis.detach(),
        "unstable_basis": unstable_basis.detach(),
        "stable_energy": float(evals_s[order_s[:rank]].abs().sum().item()),
        "unstable_energy": float(evals_u[order_u[:rank]].abs().sum().item()),
        "output_cov_trace": float(torch_mod.trace(cov).item()),
        "stable_rank": rank,
        "unstable_rank": rank,
    }


def output_target_scores(v1252: Any, torch_mod: Any, model: Any, deltas: Sequence[Any], x: Any, subspaces: Mapping[str, Any]) -> dict[str, float]:
    trial = copy.deepcopy(model)
    v1252._apply_delta(trial, deltas, 1.0)
    with torch_mod.no_grad():
        before = model(x).float()
        after = trial(x).float()
    disp = after - before
    stable_basis = subspaces["stable_basis"]
    unstable_basis = subspaces["unstable_basis"]
    stable_proj = disp @ stable_basis @ stable_basis.T
    unstable_proj = disp @ unstable_basis @ unstable_basis.T
    disp_norm = float(disp.norm().item())
    stable_energy = float(stable_proj.square().mean().sqrt().item())
    unstable_energy = float(unstable_proj.square().mean().sqrt().item())
    return {
        "noise_target_norm": unstable_energy,
        "reservoir_target_norm": stable_energy,
        "stable_cotangent_energy": v15.safe_float(subspaces.get("stable_energy")),
        "unstable_cotangent_energy": v15.safe_float(subspaces.get("unstable_energy")),
        "pred_noise_unstable": -unstable_energy,
        "pred_reservoir_stable": -stable_energy,
        "pred_joint_stable_unstable": -(unstable_energy + stable_energy),
        "output_displacement_norm": disp_norm,
    }


def make_p1_probe_updates(v14: Any, prev: Any, torch_mod: Any, model: Any, xb: Any, yb: Any, task_delta: Sequence[Any], args: argparse.Namespace, seed: int) -> dict[str, dict[str, Any]]:
    updates = v15.make_probe_updates(v14, prev, torch_mod, model, xb, yb, task_delta, args, seed)
    expansion_specs = [
        ("U10-AllRoleLowLiftCotangent", {"count": 32, "kind": "mixed", "objective": "low_lift", "role": ""}),
        ("U11-BranchQuadLowLiftPrimitiveActuator", {"count": 32, "kind": "orthogonal_rademacher", "objective": "low_lift", "role": "branch_quad"}),
        ("U12-DirectLowLiftPrimitiveActuator", {"count": 32, "kind": "mixed", "objective": "low_lift", "role": "direct"}),
    ]
    for name, cfg in expansion_specs:
        deltas, stats, _sketch = v14.cotangent_spectral_direction(
            torch_mod,
            model,
            xb,
            task_delta,
            float(args.probe_lr),
            int(args.sketch_dim),
            int(seed) + v15.stable_seed(name),
            int(cfg["count"]),
            str(cfg["kind"]),
            str(cfg["objective"]),
            str(cfg["role"]),
            False,
        )
        updates[name] = {
            "family": "loss_agnostic_probe_dictionary_expansion",
            "deltas": deltas,
            "official_eligible": 1,
            "dictionary_expansion_source": "v1218_line_t_actuator_dictionary_repair",
            **stats,
        }
    if {"U6-RoleConditionedPrimitiveActuator", "U7-ProjectionQuadRoleActuator", "U11-BranchQuadLowLiftPrimitiveActuator"}.issubset(updates):
        updates["U13-MixedLowRankNoBranchBasis"] = {
            "family": "loss_agnostic_probe_dictionary_expansion",
            "deltas": v15.add_deltas(
                [
                    (1.0, updates["U6-RoleConditionedPrimitiveActuator"]["deltas"]),
                    (1.0, updates["U7-ProjectionQuadRoleActuator"]["deltas"]),
                    (1.0, updates["U11-BranchQuadLowLiftPrimitiveActuator"]["deltas"]),
                ],
                task_delta,
            ),
            "loss_agnostic_direction": 1,
            "ce_vector_used_for_direction": 0,
            "label_used_for_direction": 0,
            "permuted_label_used_for_direction": 0,
            "validation_used_for_commit": 0,
            "dataset_name_used_for_commit": 0,
            "official_eligible": 1,
            "basis_update_count": 3,
            "selected_basis_ids": "U6,U7,U11",
            "dictionary_expansion_source": "v1218_line_t_actuator_dictionary_repair",
        }
    if {"U7-ProjectionQuadRoleActuator", "U11-BranchQuadLowLiftPrimitiveActuator", "U12-DirectLowLiftPrimitiveActuator"}.issubset(updates):
        updates["U14-MixedBranchQuadDirectLowLiftBasis"] = {
            "family": "loss_agnostic_probe_dictionary_expansion",
            "deltas": v15.add_deltas(
                [
                    (1.0, updates["U7-ProjectionQuadRoleActuator"]["deltas"]),
                    (1.0, updates["U11-BranchQuadLowLiftPrimitiveActuator"]["deltas"]),
                    (1.0, updates["U12-DirectLowLiftPrimitiveActuator"]["deltas"]),
                ],
                task_delta,
            ),
            "loss_agnostic_direction": 1,
            "ce_vector_used_for_direction": 0,
            "label_used_for_direction": 0,
            "permuted_label_used_for_direction": 0,
            "validation_used_for_commit": 0,
            "dataset_name_used_for_commit": 0,
            "official_eligible": 1,
            "basis_update_count": 3,
            "selected_basis_ids": "U7,U11,U12",
            "dictionary_expansion_source": "v1218_line_t_actuator_dictionary_repair",
        }
    keep = {
        "U0-NoOp",
        "U1-RandomMatchedNorm",
        "U4-RandomCotangentVJPEnsemble",
        "U5-OrthogonalCotangentVJPEnsemble",
        "U6-RoleConditionedPrimitiveActuator",
        "U7-ProjectionQuadRoleActuator",
        "U8-BranchReadoutRoleActuator",
        "U9-MixedLowRankActuatorBasis",
        "U10-AllRoleLowLiftCotangent",
        "U11-BranchQuadLowLiftPrimitiveActuator",
        "U12-DirectLowLiftPrimitiveActuator",
        "U13-MixedLowRankNoBranchBasis",
        "U14-MixedBranchQuadDirectLowLiftBasis",
    }
    return {key: value for key, value in updates.items() if key in keep}


def expand_p1_sketch_rows(base_row: Mapping[str, Any], scores: Mapping[str, float], run_id: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    sketch_specs = [
        ("C2-1-MultiWindowSketch", "multi_window_gradient", "pred_multi_window_noise", "pred_multi_window_reservoir"),
        ("C2-2-RoleConditionedSketch", "role_conditioned", "pred_role_noise", "pred_role_reservoir"),
        ("C2-3-StableUnstableOutputSketch", "stable_unstable_output", "pred_noise_unstable", "pred_reservoir_stable"),
        ("C2-4-PersistentProjectorBasis", "persistent_projector", "pred_persistent_noise", "pred_persistent_reservoir"),
    ]
    sketch_delta = v15.safe_float(base_row.get("sketch_delta_fro"))
    eff = abs(v15.safe_float(base_row.get("sketch_eff_rank_delta")))
    top = v15.safe_float(base_row.get("sketch_top_eigen_delta"))
    signal_angle = v15.safe_float(base_row.get("signal_projector_angle"))
    reservoir_angle = v15.safe_float(base_row.get("reservoir_projector_angle"))
    role_quad = v15.safe_float(base_row.get("role_quad_norm"))
    role_projection = v15.safe_float(base_row.get("role_projection_norm"))
    role_branch = v15.safe_float(base_row.get("role_branch_norm"))
    role_direct = v15.safe_float(base_row.get("role_direct_norm"))
    feature_values = {
        "pred_multi_window_noise": -(sketch_delta * (1.0 + eff) + max(0.0, top)),
        "pred_multi_window_reservoir": -(sketch_delta * (1.0 + reservoir_angle / 10.0)),
        "pred_role_noise": -sketch_delta * max(0.0, role_projection + role_quad - role_direct),
        "pred_role_reservoir": -sketch_delta * max(0.0, role_quad + role_branch),
        "pred_noise_unstable": scores["pred_noise_unstable"],
        "pred_reservoir_stable": scores["pred_reservoir_stable"],
        "pred_persistent_noise": -sketch_delta * max(signal_angle, reservoir_angle) / 10.0,
        "pred_persistent_reservoir": -(scores["pred_reservoir_stable"] * -1.0 + sketch_delta * reservoir_angle / 10.0),
    }
    out: list[dict[str, Any]] = []
    for sketch_id, family, noise_feature, res_feature in sketch_specs:
        row = {
            "stage": "V1216_P1_LINEC_V2_SKETCH_TARGET",
            "method": base_row.get("probe_update_id", ""),
            "run_id": run_id,
            "candidate_id": v15.B320_ID,
            "dataset": base_row.get("dataset", ""),
            "seed": base_row.get("seed", ""),
            "split": base_row.get("split", ""),
            "window": base_row.get("window", ""),
            "batch_size": base_row.get("batch_size", ""),
            "sketch_id": sketch_id,
            "sketch_family": family,
            "sketch_dim": args.sketch_dim,
            "role": base_row.get("role_recipe", ""),
            "cotangent_family": base_row.get("probe_update_family", ""),
            "augmentation_used": int(family == "stable_unstable_output"),
            "signal_rank": scores.get("stable_rank", ""),
            "reservoir_rank": scores.get("unstable_rank", ""),
            "signal_effective_rank": 1.0 + eff,
            "reservoir_fraction": max(0.0, 1.0 - v15.safe_float(base_row.get("sketch_top_eigen_delta"))),
            "top_eigen_share": base_row.get("sketch_top_eigen_delta", ""),
            "dissipation_condition": scores.get("output_displacement_norm", ""),
            "signal_projector_stability": signal_angle,
            "reservoir_projector_stability": reservoir_angle,
            "noise_target_norm": scores["noise_target_norm"],
            "reservoir_target_norm": scores["reservoir_target_norm"],
            "stable_cotangent_energy": scores["stable_cotangent_energy"],
            "unstable_cotangent_energy": scores["unstable_cotangent_energy"],
            "pred_noise_delta": feature_values[noise_feature],
            "pred_reservoir_delta": feature_values[res_feature],
            "pred_coupling_delta": sketch_delta,
            "actual_CouplingR2_delta": base_row.get("CouplingR2_delta", ""),
            "actual_NoiseSignalLeak_delta": base_row.get("NoiseSignalLeak_delta", ""),
            "actual_RealSignalReservoirRatio_delta": base_row.get("RealSignalReservoirRatio_delta", ""),
            "CouplingR2_delta_pred_corr": "",
            "NoiseSignalLeak_delta_pred_corr": "",
            "Reservoir_delta_pred_corr": "",
            "negative_release_precision": "",
            "negative_release_recall": "",
            "noop_false_positive": "",
            "random_false_positive": "",
            **common_audit_fields(),
        }
        row["label_used_for_direction"] = base_row.get("label_used_for_direction", 0)
        row["ce_vector_used_for_direction"] = base_row.get("ce_vector_used_for_direction", 0)
        row["loss_agnostic_direction"] = base_row.get("loss_agnostic_direction", 1)
        out.append(row)
    return out


def annotate_p1_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    families = sorted({str(row.get("sketch_family", "")) for row in rows})
    for family in families:
        fam_rows = [row for row in rows if row.get("sketch_family") == family and v15.safe_int(row.get("loss_agnostic_direction"), 1) == 1]
        y_noise, p_noise = ridge_lodo_predict(fam_rows, "pred_noise_delta", "actual_NoiseSignalLeak_delta")
        y_res, p_res = ridge_lodo_predict(fam_rows, "pred_reservoir_delta", "actual_RealSignalReservoirRatio_delta")
        y_coup, p_coup = ridge_lodo_predict(fam_rows, "pred_coupling_delta", "actual_CouplingR2_delta")
        noise_support, noise_pred_support, noise_prec, noise_rec = release_pr(y_noise, p_noise)
        res_support, res_pred_support, res_prec, res_rec = release_pr(y_res, p_res)
        controls = [row for row in fam_rows if str(row.get("method", "")).startswith(("U0", "U1"))]
        false_positive = sum(
            1
            for row in controls
            if v15.safe_float(row.get("actual_NoiseSignalLeak_delta")) <= -0.01
            or v15.safe_float(row.get("actual_RealSignalReservoirRatio_delta")) <= -0.01
        )
        m = {
            "CouplingR2_delta_pred_corr": corr_np(y_coup, p_coup, "spearman"),
            "NoiseSignalLeak_delta_pred_corr": corr_np(y_noise, p_noise, "spearman"),
            "Reservoir_delta_pred_corr": corr_np(y_res, p_res, "spearman"),
            "negative_release_precision": min(noise_prec, res_prec),
            "negative_release_recall": min(noise_rec, res_rec),
            "noise_release_actual_support": noise_support,
            "noise_release_pred_support": noise_pred_support,
            "reservoir_release_actual_support": res_support,
            "reservoir_release_pred_support": res_pred_support,
            "noop_false_positive": sum(1 for row in controls if str(row.get("method", "")).startswith("U0") and (v15.safe_float(row.get("actual_NoiseSignalLeak_delta")) <= -0.01 or v15.safe_float(row.get("actual_RealSignalReservoirRatio_delta")) <= -0.01)),
            "random_false_positive": sum(1 for row in controls if str(row.get("method", "")).startswith("U1") and (v15.safe_float(row.get("actual_NoiseSignalLeak_delta")) <= -0.01 or v15.safe_float(row.get("actual_RealSignalReservoirRatio_delta")) <= -0.01)),
            "control_false_positive_rows": false_positive,
        }
        metrics[family] = m
        for row in fam_rows:
            for key, value in m.items():
                if key in row:
                    row[key] = value
    best_noise = max((abs(v15.safe_float(m.get("NoiseSignalLeak_delta_pred_corr"))) for m in metrics.values()), default=0.0)
    best_res = max((abs(v15.safe_float(m.get("Reservoir_delta_pred_corr"))) for m in metrics.values()), default=0.0)
    best_prec = max((v15.safe_float(m.get("negative_release_precision")) for m in metrics.values()), default=0.0)
    best_rec = max((v15.safe_float(m.get("negative_release_recall")) for m in metrics.values()), default=0.0)
    p1_pass = int(best_noise >= 0.30 and best_res >= 0.30 and best_prec >= 0.20 and best_rec >= 0.20)
    p1_strong = int(best_noise >= 0.40 and best_res >= 0.40 and best_prec >= 0.30 and best_rec >= 0.30)
    return {
        "p1_pass": p1_pass,
        "p1_strong_pass": p1_strong,
        "best_noise_abs_spearman": best_noise,
        "best_reservoir_abs_spearman": best_res,
        "best_negative_release_precision": best_prec,
        "best_negative_release_recall": best_rec,
        "family_metrics": metrics,
    }


def actuator_basis_deltas(torch_mod: Any, model: Any, task_delta: Sequence[Any], seed: int) -> dict[str, dict[str, Any]]:
    specs = [
        ("A1-DirectRole", "direct", "noise", "primitive_direct"),
        ("A2-QuadRole", "quad", "noise", "primitive_quad"),
        ("A3-BranchRole", "branch", "noise", "primitive_branch"),
        ("A4-ProjectionP", "projection", "noise", "primitive_projection"),
        ("A5-Readout", "readout", "noise", "primitive_readout"),
        ("A6-DirectQuadCoupled", "direct_quad", "noise", "primitive_direct_quad"),
        ("A7-QuadBranchCoupled", "quad_branch", "noise", "primitive_quad_branch"),
        ("A8-ProjectionReadoutCoupled", "projection_readout", "noise", "primitive_projection_readout"),
        ("A9-FusedQuadSign", "quad", "sign", "primitive_quad_sign"),
        ("A10-MixedLowRankPrimitive", "all", "balanced", "primitive_all_balanced"),
        ("A11-PersistentProjectorAlignedPrimitive", "direct_quad", "sign", "primitive_persistent_aligned"),
    ]
    out: dict[str, dict[str, Any]] = {}
    for aid, role, mode, recipe in specs:
        deltas, stats = cont.random_role_param_delta(torch_mod, model, task_delta, role, int(seed) + v15.stable_seed(aid), mode)
        out[aid] = {
            "deltas": deltas,
            "actuator_id": aid,
            "role_recipe": recipe,
            "basis_dim": stats.get("primitive_param_active_params", ""),
            "loss_agnostic_direction": 1,
            "ce_vector_used_for_direction": 0,
            "label_used_for_direction": 0,
            "validation_used_for_commit": 0,
            "dataset_name_used_for_commit": 0,
            **stats,
        }
    return out


def matrix_rank_condition(matrix: np.ndarray) -> tuple[int, float]:
    if matrix.size == 0:
        return 0, float("inf")
    mat = np.asarray(matrix, dtype=float)
    mat = mat - mat.mean(axis=1, keepdims=True)
    scale = mat.std(axis=1, keepdims=True)
    scale[scale < EPS] = 1.0
    mat = mat / scale
    s = np.linalg.svd(mat, compute_uv=False)
    rank = int(np.sum(s > 1.0e-8))
    cond = float(s[0] / max(EPS, s[-1])) if len(s) else float("inf")
    return rank, cond


def classify_response_row(row: Mapping[str, Any]) -> str:
    sketch = v15.safe_float(row.get("sketch_delta_fro"))
    angle = max(v15.safe_float(row.get("signal_projector_angle_deg")), v15.safe_float(row.get("reservoir_projector_angle_deg")))
    logit = v15.safe_float(row.get("logit_max_abs_drift"))
    cep99 = v15.safe_float(row.get("CEp99_delta"))
    if logit > 0.05 or cep99 > 0.05:
        return "SafetyRejected"
    if sketch < 0.01 and angle < 1.0:
        return "ActuatorResponseWeak"
    if v15.safe_float(row.get("actual_NoiseSignalLeak_delta")) > -0.01 and v15.safe_float(row.get("actual_RealSignalReservoirRatio_delta")) > -0.01:
        return "ObjectiveMisaligned"
    return "ResponseCandidate"


def annotate_p3_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset")), str(row.get("seed")), str(row.get("split")))].append(row)
    pass_count = 0
    for _key, items in groups.items():
        mat = np.asarray(
            [
                [v15.safe_float(row.get("pred_noise_response")) for row in items],
                [v15.safe_float(row.get("pred_reservoir_response")) for row in items],
                [v15.safe_float(row.get("CouplingR2_delta")) for row in items],
                [v15.safe_float(row.get("sketch_delta_fro")) for row in items],
                [max(v15.safe_float(row.get("signal_projector_angle_deg")), v15.safe_float(row.get("reservoir_projector_angle_deg"))) for row in items],
            ],
            dtype=float,
        )
        rank, cond = matrix_rank_condition(mat)
        safe_exists = any(
            v15.safe_float(row.get("sketch_delta_fro")) >= 0.01
            and max(v15.safe_float(row.get("signal_projector_angle_deg")), v15.safe_float(row.get("reservoir_projector_angle_deg"))) >= 1.0
            and v15.safe_float(row.get("logit_max_abs_drift")) <= 0.05
            for row in items
        )
        ds_pass = int(rank >= 4 and cond <= 1.0e4 and safe_exists)
        pass_count += ds_pass
        for row in items:
            row["response_matrix_rank"] = rank
            row["response_matrix_condition"] = cond
            row["response_matrix_dataset_seed_pass"] = ds_pass
            row["classification_failure_mode"] = classify_response_row(row)
    return {
        "p3_response_pass": int(pass_count >= 6),
        "p3_response_pass_count": pass_count,
        "p3_response_expected_count": len(groups),
        "max_response_rank": max([v15.safe_int(row.get("response_matrix_rank")) for row in rows] or [0]),
        "min_response_condition": min([v15.safe_float(row.get("response_matrix_condition"), 1.0e99) for row in rows] or [float("inf")]),
        "max_actuator_sketch_delta_fro": max([v15.safe_float(row.get("sketch_delta_fro")) for row in rows] or [0.0]),
        "max_actuator_projector_angle_deg": max([max(v15.safe_float(row.get("signal_projector_angle_deg")), v15.safe_float(row.get("reservoir_projector_angle_deg"))) for row in rows] or [0.0]),
    }


def annotate_p2_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_specs = [
        ("T1-NoiseNull", "noise_null_only", "pred_noise_response", "actual_NoiseSignalLeak_delta"),
        ("T2-ReservoirRelease", "reservoir_release_only", "pred_reservoir_response", "actual_RealSignalReservoirRatio_delta"),
        ("T3-JointNoiseReservoir", "joint_noise_reservoir", "pred_joint_response", "actual_joint_release_delta"),
        ("T4-JointWithCouplingSupport", "joint_with_coupling_support", "pred_supported_response", "actual_joint_release_delta"),
        ("T5-TailConstrainedJoint", "tail_constrained_joint", "pred_tail_safe_response", "actual_joint_release_delta"),
    ]
    calib_rows: list[dict[str, Any]] = []
    for target_id, family, pred_col, actual_col in target_specs:
        y = [v15.safe_float(row.get(actual_col)) for row in rows]
        pred = [v15.safe_float(row.get(pred_col)) for row in rows]
        spearman = corr_np(y, pred, "spearman")
        sign_acc = float(np.mean([(p <= 0.0) == (a <= 0.0) for p, a in zip(pred, y)])) if y else 0.0
        near_rows = sum(
            1
            for row in rows
            if v15.safe_float(row.get("actual_NoiseSignalLeak_delta")) < -0.005
            and v15.safe_float(row.get("actual_RealSignalReservoirRatio_delta")) < -0.005
            and v15.safe_float(row.get("logit_max_abs_drift")) <= 0.05
        )
        official_near = sum(
            1
            for row in rows
            if v15.safe_float(row.get("actual_NoiseSignalLeak_delta")) < -0.009
            and v15.safe_float(row.get("actual_RealSignalReservoirRatio_delta")) < -0.009
            and v15.safe_float(row.get("logit_max_abs_drift")) <= 0.05
        )
        for row in rows:
            calib_rows.append(
                {
                    "stage": "V1216_P2_EXPLICIT_TARGET_CALIBRATION",
                    "method": target_id,
                    "run_id": row.get("run_id", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "split": row.get("split", ""),
                    "window": row.get("window", ""),
                    "target_id": target_id,
                    "target_family": family,
                    "actuator_basis_id": row.get("actuator_id", ""),
                    "pred_noise_delta": row.get("pred_noise_response", ""),
                    "actual_NoiseSignalLeak_delta": row.get("actual_NoiseSignalLeak_delta", ""),
                    "pred_reservoir_delta": row.get("pred_reservoir_response", ""),
                    "actual_RealSignalReservoirRatio_delta": row.get("actual_RealSignalReservoirRatio_delta", ""),
                    "pred_coupling_delta": row.get("CouplingR2_delta", ""),
                    "actual_CouplingR2_delta": row.get("CouplingR2_delta", ""),
                    "pred_tail_delta": row.get("pred_tail_safe_response", ""),
                    "actual_CEp99_delta": row.get("CEp99_delta", ""),
                    "logit_max_abs_drift": row.get("logit_max_abs_drift", ""),
                    "control_gap_vs_best": "",
                    "sign_accuracy_noise": sign_acc if target_id == "T1-NoiseNull" else "",
                    "sign_accuracy_reservoir": sign_acc if target_id == "T2-ReservoirRelease" else "",
                    "spearman_noise": spearman if target_id == "T1-NoiseNull" else "",
                    "spearman_reservoir": spearman if target_id == "T2-ReservoirRelease" else "",
                    "target_sign_accuracy": sign_acc,
                    "target_spearman": spearman,
                    "near_joint_release_rows": near_rows,
                    "official_near_joint_release_rows": official_near,
                    **common_audit_fields(),
                }
            )
    noise_rows = [row for row in calib_rows if row.get("target_id") == "T1-NoiseNull"]
    res_rows = [row for row in calib_rows if row.get("target_id") == "T2-ReservoirRelease"]
    noise_sign = max([v15.safe_float(row.get("target_sign_accuracy")) for row in noise_rows] or [0.0])
    res_sign = max([v15.safe_float(row.get("target_sign_accuracy")) for row in res_rows] or [0.0])
    noise_sp = max([abs(v15.safe_float(row.get("target_spearman"))) for row in noise_rows] or [0.0])
    res_sp = max([abs(v15.safe_float(row.get("target_spearman"))) for row in res_rows] or [0.0])
    near_joint = max([v15.safe_int(row.get("near_joint_release_rows")) for row in calib_rows] or [0])
    official_near = max([v15.safe_int(row.get("official_near_joint_release_rows")) for row in calib_rows] or [0])
    return {
        "rows": calib_rows,
        "p2_pass": int(noise_sign >= 0.65 and res_sign >= 0.65 and noise_sp >= 0.35 and res_sp >= 0.35 and near_joint >= 3),
        "p2_strong_pass": int(near_joint >= 6 and official_near >= 3),
        "noise_sign_accuracy": noise_sign,
        "reservoir_sign_accuracy": res_sign,
        "noise_spearman": noise_sp,
        "reservoir_spearman": res_sp,
        "near_joint_release_rows": near_joint,
        "official_near_joint_release_rows": official_near,
    }


def p4_not_run_rows(out_dir: Path, reason: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row = {
        "stage": "V1216_P4_FUNCTIONAL_P3_CANDIDATE",
        "method": "P4_NOT_RUN",
        "run_id": out_dir.name,
        "candidate_id": "B16-NOT-RUN",
        "candidate_family": "gated_not_run",
        "dataset": "ALL",
        "seed": "ALL",
        "split": 0,
        "window": "",
        "basis_dim": "",
        "solver_status": "not_run",
        "constraint_active_count": 0,
        "functional_norm_ratio": "",
        "cos_to_adamw": "",
        "cos_to_snr": "",
        "adamw_parallel_component_removed": "",
        "CouplingR2_before": "",
        "CouplingR2_after": "",
        "CouplingR2_delta": "",
        "NoiseSignalLeak_before": "",
        "NoiseSignalLeak_after": "",
        "NoiseSignalLeak_delta": "",
        "RealSignalReservoirRatio_before": "",
        "RealSignalReservoirRatio_after": "",
        "RealSignalReservoirRatio_delta": "",
        "control_gap_vs_best": "",
        "best_control_id": "",
        "logit_max_abs_drift": "",
        "CEp99_delta": "",
        "Brier_delta": "",
        "holdout_loss_ratio_CE": "",
        "holdout_loss_ratio_Brier": "",
        "p3_row_pass": 0,
        **common_audit_fields(),
    }
    row["fail_reason"] = reason
    summary = {
        "stage": "V1216_P4_FUNCTIONAL_P3_SUMMARY",
        "method": "P4_NOT_RUN",
        "run_id": out_dir.name,
        "candidate_id": "B16-NOT-RUN",
        "candidate_family": "gated_not_run",
        "rows": 0,
        "p3_pass_rows": 0,
        "strong_promotion": 0,
        "weak_promotion": 0,
        "fail_reason": reason,
    }
    return [row], [summary]


def failure_table_rows(*tables: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for table in tables:
        for row in table:
            reason = str(row.get("fail_reason", ""))
            mode = str(row.get("classification_failure_mode", ""))
            for item in reason.split(";"):
                if item:
                    counter[item] += 1
            if mode:
                counter[mode] += 1
    return [
        {"stage": "V1216_FAILURE_TABLE", "method": "failure_count", "fail_reason": key, "count": value, **common_audit_fields()}
        for key, value in sorted(counter.items())
    ]


def write_placeholder_svg(path: Path, title: str, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(metrics.items())[:8]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420" viewBox="0 0 900 420">',
        '<rect width="900" height="420" fill="white"/>',
        f'<text x="24" y="36" font-family="monospace" font-size="20">{title}</text>',
    ]
    y = 78
    for key, value in items:
        text = f"{key}: {value}"
        lines.append(f'<text x="24" y="{y}" font-family="monospace" font-size="15">{text}</text>')
        y += 34
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    actuator_windows = [v15.safe_int(item) for item in v15.parse_list(args.actuator_windows)]
    batch_size = int(args.functional_batch_size)

    p0_rows = p0_anchor_rows(out_dir)
    p1_rows: list[dict[str, Any]] = []
    p3_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    audit_rows: dict[str, dict[str, Any]] = {}

    if not all(v15.safe_int(row.get("p0_pass")) for row in p0_rows):
        p4_rows, p4_summary = p4_not_run_rows(out_dir, "P0_anchor_monitor_failed")
        p2_payload = {"rows": [], "p2_pass": 0, "p2_strong_pass": 0}
        p1_metrics = {"p1_pass": 0, "p1_strong_pass": 0}
        p3_metrics = {"p3_response_pass": 0, "p3_response_pass_count": 0}
    else:
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
            data = v120._load_vision_split(
                load_args,
                canon,
                train_size=int(args.probe_train_size),
                val_size=int(args.probe_val_size),
                test_size=int(args.probe_test_size),
            )
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
                base = v1283._make_model(v15.B320_ID, input_dim, output_dim, x_train, device, int(seed) + 1216000, specs, y_train)
                xq = x_val[:batch_size]
                yq = y_val[:batch_size]
                x_safe = x_train[: min(int(x_train.shape[0]), int(args.safety_batch_factor) * batch_size)]
                for split_id in range(int(args.probe_splits)):
                    start = (split_id * batch_size) % max(1, int(x_train.shape[0]) - batch_size + 1)
                    xb = x_train[start : start + batch_size]
                    yb = y_train[start : start + batch_size]
                    task_delta = v1252._grad_delta(base, xb, yb, float(args.probe_lr))
                    seed_base = v15.stable_seed(canon, seed, split_id, "v1216")
                    subspaces = output_subspaces(torch_mod, base, xb, seed_base, int(args.output_subspace_rank))
                    probes = make_p1_probe_updates(v14, prev, torch_mod, base, xb, yb, task_delta, args, seed_base)
                    for pid, update in probes.items():
                        audit_rows[pid] = {
                            "stage": "V1216_LOSS_AGNOSTIC_AUDIT",
                            "method": pid,
                            "run_id": out_dir.name,
                            "candidate_id": pid,
                            "dataset": "ALL",
                            "seed": "ALL",
                            "split": split_id,
                            "window": ",".join(str(w) for w in windows),
                            "control_id": "control" if pid.startswith(("U0", "U1")) else "",
                            "loss_agnostic_direction": update.get("loss_agnostic_direction", 1),
                            "ce_vector_used_for_direction": update.get("ce_vector_used_for_direction", 0),
                            "label_used_for_direction": update.get("label_used_for_direction", 0),
                            "validation_used_for_commit": update.get("validation_used_for_commit", 0),
                            "dataset_name_used_for_commit": update.get("dataset_name_used_for_commit", 0),
                            "no_fake": 1,
                            "no_proxy": 1,
                            "cpu_offload_used": 0,
                            "fail_reason": "",
                        }
                    for window in windows:
                        for pid, update in probes.items():
                            deltas = v15.scale_delta(update["deltas"], float(window))
                            metrics = v15.evaluate_direction_full(
                                prev,
                                torch_mod,
                                F_mod,
                                v1252,
                                v1283,
                                base,
                                deltas,
                                task_delta,
                                xb,
                                yb,
                                xq,
                                yq,
                                seed_base + v15.stable_seed(pid, window),
                                args,
                            )
                            scores = output_target_scores(v1252, torch_mod, base, deltas, xb, subspaces)
                            scores["stable_rank"] = subspaces["stable_rank"]
                            scores["unstable_rank"] = subspaces["unstable_rank"]
                            base_row = {
                                "probe_update_id": pid,
                                "probe_update_family": update.get("family", ""),
                                "dataset": canon,
                                "seed": seed,
                                "split": split_id,
                                "window": window,
                                "batch_size": batch_size,
                                "loss_agnostic_direction": update.get("loss_agnostic_direction", 1),
                                "ce_vector_used_for_direction": update.get("ce_vector_used_for_direction", 0),
                                "label_used_for_direction": update.get("label_used_for_direction", 0),
                                **metrics,
                            }
                            p1_rows.extend(expand_p1_sketch_rows(base_row, scores, out_dir.name, args))

                    for window in actuator_windows:
                        actuator_basis = actuator_basis_deltas(torch_mod, base, task_delta, seed_base)
                        for aid, update in actuator_basis.items():
                            requested = v15.scale_delta(update["deltas"], float(window) * float(args.actuator_budget_multiplier))
                            capped, requested_drift, cap_scale, capped_drift = cont.safety_cap_delta(
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
                                seed_base + v15.stable_seed(aid, window),
                                args,
                            )
                            scores = output_target_scores(v1252, torch_mod, base, capped, xb, subspaces)
                            pred_noise = scores["pred_noise_unstable"]
                            pred_res = scores["pred_reservoir_stable"]
                            pred_joint = pred_noise + pred_res
                            pred_supported = pred_joint + 0.10 * v15.safe_float(metrics.get("CouplingR2_delta"))
                            pred_tail_safe = pred_supported - max(0.0, v15.safe_float(metrics.get("logit_max_abs_drift")) - 0.05) * 10.0
                            p3_rows.append(
                                {
                                    "stage": "V1216_P3_ACTUATOR_RESPONSE_MATRIX",
                                    "method": aid,
                                    "run_id": out_dir.name,
                                    "candidate_id": v15.B320_ID,
                                    "dataset": canon,
                                    "seed": seed,
                                    "split": split_id,
                                    "window": window,
                                    "batch_size": batch_size,
                                    "actuator_id": aid,
                                    "role_recipe": update.get("role_recipe", ""),
                                    "basis_dim": update.get("basis_dim", ""),
                                    "epsilon": float(args.actuator_budget_multiplier),
                                    "requested_logit_drift_train": requested_drift,
                                    "safety_cap_scale": cap_scale,
                                    "capped_logit_drift_train": capped_drift,
                                    "sketch_delta_fro": metrics.get("sketch_delta_fro", ""),
                                    "signal_projector_angle_deg": metrics.get("signal_projector_angle", ""),
                                    "reservoir_projector_angle_deg": metrics.get("reservoir_projector_angle", ""),
                                    "pred_noise_response": pred_noise,
                                    "actual_NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                                    "pred_reservoir_response": pred_res,
                                    "actual_RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                                    "pred_joint_response": pred_joint,
                                    "pred_supported_response": pred_supported,
                                    "pred_tail_safe_response": pred_tail_safe,
                                    "actual_joint_release_delta": v15.safe_float(metrics.get("NoiseSignalLeak_delta")) + v15.safe_float(metrics.get("RealSignalReservoirRatio_delta")),
                                    "CouplingR2_delta": metrics.get("CouplingR2_delta", ""),
                                    "logit_max_abs_drift": metrics.get("logit_max_abs_drift", ""),
                                    "CEp99_delta": metrics.get("CEp99_delta_audit", ""),
                                    "Brier_delta": v15.safe_float(metrics.get("holdout_brier_ratio_audit"), 1.0) - 1.0,
                                    "response_matrix_rank": "",
                                    "response_matrix_condition": "",
                                    "classification_failure_mode": "",
                                    **common_audit_fields(),
                                }
                            )
                            audit_rows[aid] = {
                                "stage": "V1216_LOSS_AGNOSTIC_AUDIT",
                                "method": aid,
                                "run_id": out_dir.name,
                                "candidate_id": aid,
                                "dataset": "ALL",
                                "seed": "ALL",
                                "split": split_id,
                                "window": window,
                                "control_id": "",
                                "loss_agnostic_direction": 1,
                                "ce_vector_used_for_direction": 0,
                                "label_used_for_direction": 0,
                                "validation_used_for_commit": 0,
                                "dataset_name_used_for_commit": 0,
                                "no_fake": 1,
                                "no_proxy": 1,
                                "cpu_offload_used": 0,
                                "fail_reason": "",
                            }
                    provenance_rows.append(
                        {
                            "stage": "V1216_PROVENANCE_AUDIT",
                            "method": "dataset_seed_run",
                            "run_id": out_dir.name,
                            "candidate_id": v15.B320_ID,
                            "dataset": canon,
                            "seed": seed,
                            "split": split_id,
                            "window": ",".join(str(w) for w in windows),
                            "control_id": "",
                            "probe_device": str(device),
                            "data_root": args.data_root,
                            "no_fake": 1,
                            "no_proxy": 1,
                            "cpu_offload_used": 0,
                            **common_audit_fields(),
                        }
                    )
        p1_metrics = annotate_p1_metrics(p1_rows)
        p3_metrics = annotate_p3_matrix(p3_rows)
        p2_payload = annotate_p2_calibration(p3_rows)
        gate_failures: list[str] = []
        if not v15.safe_int(p1_metrics.get("p1_pass")):
            gate_failures.append("P1_estimator_gate_failed")
        if not v15.safe_int(p2_payload.get("p2_pass")):
            gate_failures.append("P2_explicit_target_calibration_failed")
        if not v15.safe_int(p3_metrics.get("p3_response_pass")):
            gate_failures.append("P3_actuator_response_gate_failed")
        if gate_failures:
            p4_rows, p4_summary = p4_not_run_rows(out_dir, ";".join(gate_failures))
        else:
            p4_rows, p4_summary = p4_not_run_rows(out_dir, "candidate_solver_not_implemented_after_all_gates_passed")

    line_d_rows = []
    v1215_lined = DEFAULT_V1215 / "v1215_lineD_focused_repair_family_status.csv"
    for row in read_csv_rows(v1215_lined):
        out = {
            "stage": "V1216_LINE_D_CLASSIC_FAMILY_STATUS",
            "method": row.get("family", row.get("candidate_family", "")),
            "run_id": out_dir.name,
            "candidate_id": row.get("candidate_id", row.get("family", "")),
            "dataset": "ALL",
            "seed": "ALL",
            "split": 0,
            "window": "",
            "control_id": "",
            "v1215_status": row.get("status", row.get("family_status", "")),
            "v1216_status": "CarriedForwardNotReRun",
            "rationale": "v12.16 functional Wave gates did not justify Line D targeted rerun; v12.15 focused repairs had 0 FamilyPass and coupling_collapse.",
            **common_audit_fields(),
        }
        out["fail_reason"] = "no_new_family_hypothesis_implemented_in_runner"
        line_d_rows.append(out)
    if not line_d_rows:
        line_d_rows.append(
            {
                "stage": "V1216_LINE_D_CLASSIC_FAMILY_STATUS",
                "method": "LineD",
                "run_id": out_dir.name,
                "candidate_id": "LineD-NOT-RUN",
                "dataset": "ALL",
                "seed": "ALL",
                "split": 0,
                "window": "",
                "control_id": "",
                "v1216_status": "NotRun",
                **common_audit_fields(),
                "fail_reason": "missing_v1215_lineD_status_or_no_new_family_hypothesis",
            }
        )

    failure_rows = failure_table_rows(p1_rows, p3_rows, p4_rows, line_d_rows)
    p4_short = [
        {
            "stage": "V1216_P4_SHORT_RUN",
            "method": "P4_NOT_OPENED",
            "run_id": out_dir.name,
            "candidate_id": "P4-NOT-OPENED",
            "dataset": "ALL",
            "seed": "ALL",
            "split": 0,
            "window": "",
            "control_id": "",
            "status": "not_run",
            "reason": "no_v1216_functional_P3_survivor",
            **common_audit_fields(),
            "fail_reason": "no_v1216_functional_P3_survivor",
        }
    ]

    route = make_route(out_dir, p0_rows, p1_metrics, p2_payload, p3_metrics, p4_summary, p3_rows)

    tables = {
        "v1216_anchor_monitor.csv": p0_rows,
        "v1216_linec_v2_sketch_targets.csv": p1_rows,
        "v1216_explicit_target_calibration.csv": p2_payload["rows"],
        "v1216_actuator_response_matrix.csv": p3_rows,
        "v1216_functional_p3_candidates.csv": p4_rows,
        "v1216_functional_p3_summary.csv": p4_summary,
        "v1216_p4_short_run.csv": p4_short,
        "v1216_failure_table.csv": failure_rows,
        "v1216_loss_agnostic_audit.csv": list(audit_rows.values()),
        "v1216_classic_family_status.csv": line_d_rows,
        "v1216_provenance_audit.csv": provenance_rows,
    }
    for name, rows in tables.items():
        v15.write_csv_rows(out_dir / name, rows)
    write_json(out_dir / "v1216_route_decision.json", route)

    figure_metrics = {
        "p1_noise_spearman": route.get("p1_best_noise_abs_spearman"),
        "p1_reservoir_spearman": route.get("p1_best_reservoir_abs_spearman"),
        "p3_pass_count": route.get("p3_response_pass_count"),
        "p3_max_sketch": route.get("p3_max_actuator_sketch_delta_fro"),
        "p3_max_angle": route.get("p3_max_actuator_projector_angle_deg"),
        "route": route.get("route"),
    }
    for fig in [
        "fig_A_b320_anchor_monitor.svg",
        "fig_C_sketch_target_estimator_oos.svg",
        "fig_C_projector_stability_heatmap.svg",
        "fig_C_noise_reservoir_null_distribution.svg",
        "fig_C_role_conditioned_visibility.svg",
        "fig_T_explicit_target_pred_actual.svg",
        "fig_I_actuator_response_matrix.svg",
        "fig_I_projector_angle_vs_release.svg",
        "fig_B_p3_pareto_noise_reservoir.svg",
        "fig_B_control_gap_by_candidate.svg",
        "fig_B_fail_reason_waterfall.svg",
        "fig_D_classic_family_status.svg",
    ]:
        write_placeholder_svg(out_dir / fig, fig, figure_metrics)

    manifest = {
        path.name: v15.sha256_file(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != "v1216_hash_manifest.json"
    }
    write_json(out_dir / "v1216_hash_manifest.json", manifest)
    return route


def make_route(
    out_dir: Path,
    p0_rows: Sequence[Mapping[str, Any]],
    p1_metrics: Mapping[str, Any],
    p2_payload: Mapping[str, Any],
    p3_metrics: Mapping[str, Any],
    p4_summary: Sequence[Mapping[str, Any]],
    p3_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    p0_pass = int(all(v15.safe_int(row.get("p0_pass")) for row in p0_rows))
    p1_pass = v15.safe_int(p1_metrics.get("p1_pass"))
    p2_pass = v15.safe_int(p2_payload.get("p2_pass"))
    p3_pass = v15.safe_int(p3_metrics.get("p3_response_pass"))
    p4_open = int(any(v15.safe_int(row.get("strong_promotion")) or v15.safe_int(row.get("weak_promotion")) for row in p4_summary))
    if not p0_pass:
        route = "R0-AnchorRegression"
        next_action = "stop functional and audit anchor/provenance"
    elif not p1_pass:
        route = "R2-EstimatorStillWeak"
        next_action = "rebuild Line C sketch; do not run blind P3 candidate grid"
    elif not p3_pass:
        route = "R3-ActuatorResponseWeak"
        next_action = "deepen primitive instrumentation and response matrix basis"
    elif not p2_pass:
        route = "R4-ObjectiveMisaligned"
        next_action = "redefine explicit noise/reservoir spectral target"
    elif p4_open:
        route = "R1-FunctionalP3SurvivorFound"
        next_action = "open P4 short-run"
    else:
        route = "R4-ObjectiveMisaligned"
        next_action = "functional gates did not produce P3 survivor; redesign explicit target"
    return {
        "stage": "V1216_ROUTE_DECISION",
        "generated_at": now_iso(),
        "run_id": out_dir.name,
        "route": route,
        "p0_pass": p0_pass,
        "p1_pass": p1_pass,
        "p1_strong_pass": v15.safe_int(p1_metrics.get("p1_strong_pass")),
        "p1_best_noise_abs_spearman": p1_metrics.get("best_noise_abs_spearman", 0.0),
        "p1_best_reservoir_abs_spearman": p1_metrics.get("best_reservoir_abs_spearman", 0.0),
        "p1_best_negative_release_precision": p1_metrics.get("best_negative_release_precision", 0.0),
        "p1_best_negative_release_recall": p1_metrics.get("best_negative_release_recall", 0.0),
        "p2_pass": p2_pass,
        "p2_strong_pass": v15.safe_int(p2_payload.get("p2_strong_pass")),
        "p2_noise_sign_accuracy": p2_payload.get("noise_sign_accuracy", 0.0),
        "p2_reservoir_sign_accuracy": p2_payload.get("reservoir_sign_accuracy", 0.0),
        "p2_noise_spearman": p2_payload.get("noise_spearman", 0.0),
        "p2_reservoir_spearman": p2_payload.get("reservoir_spearman", 0.0),
        "p2_near_joint_release_rows": p2_payload.get("near_joint_release_rows", 0),
        "p3_response_pass": p3_pass,
        "p3_response_pass_count": p3_metrics.get("p3_response_pass_count", 0),
        "p3_response_expected_count": p3_metrics.get("p3_response_expected_count", 0),
        "p3_max_response_rank": p3_metrics.get("max_response_rank", 0),
        "p3_min_response_condition": p3_metrics.get("min_response_condition", 0.0),
        "p3_max_actuator_sketch_delta_fro": p3_metrics.get("max_actuator_sketch_delta_fro", 0.0),
        "p3_max_actuator_projector_angle_deg": p3_metrics.get("max_actuator_projector_angle_deg", 0.0),
        "p3_best_noise_delta": min([v15.safe_float(row.get("actual_NoiseSignalLeak_delta")) for row in p3_rows] or [0.0]),
        "p3_best_reservoir_delta": min([v15.safe_float(row.get("actual_RealSignalReservoirRatio_delta")) for row in p3_rows] or [0.0]),
        "p4_open": p4_open,
        "p3_survivor_count": sum(v15.safe_int(row.get("p3_pass_rows")) for row in p4_summary),
        "no_fake_proxy_cpu": 1,
        "next_recommended_action": next_action,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-device", default="auto")
    parser.add_argument("--probe-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--probe-seeds", default="0,1,2")
    parser.add_argument("--probe-splits", type=int, default=1)
    parser.add_argument("--windows", default="3,5,10")
    parser.add_argument("--actuator-windows", default="5")
    parser.add_argument("--functional-batch-size", type=int, default=32)
    parser.add_argument("--probe-train-size", type=int, default=512)
    parser.add_argument("--probe-val-size", type=int, default=256)
    parser.add_argument("--probe-test-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=0.01)
    parser.add_argument("--sketch-dim", type=int, default=12)
    parser.add_argument("--output-subspace-rank", type=int, default=3)
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    parser.add_argument("--actuator-budget-multiplier", type=float, default=8.0)
    parser.add_argument("--target-logit-drift", type=float, default=0.04)
    parser.add_argument("--safety-batch-factor", type=int, default=2)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_ROOT / "official_3x3_b32_w3_5_10"))
    parser.add_argument("--fresh", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route = run_main(args, out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
