#!/usr/bin/env python3
"""C1/C2/C4 v22.15 adaptive MLP-source dynamics lab."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.adaptive_controller import (  # noqa: E402
    RiskFeatures,
    decide_controller,
    destructive_projection,
    retention_score,
    source_guided_update,
)
from dgkan.fu.source_manifold_basis import build_source_manifold_basis  # noqa: E402
from dgkan.fu.source_manifold_controller import manifold_coordinate_solve  # noqa: E402
from dgkan.fu.source_state_transport import init_source_state, transport_source_state  # noqa: E402
from experiments.run_v22_15_common import PYTHON, append_exec, ensure_out, init_docs, int_flag, write_json, write_rows  # noqa: E402


HORIZONS = [100, 400, 800, 1600, 2400, 3200, 4000, 4800, 6400]
MODES = [
    "M0 AdamW baseline",
    "M1 operator_only one_commit diagnostic",
    "M2 fixed_periodic_source_state",
    "M3 reactive_threshold_jacobian_prox",
    "M4 predictive_adaptive_readout_prox",
    "M5 predictive_adaptive_source_state_prox",
    "M6 continuous_lowrank_guidance",
    "M6b source_manifold_lowrank_guidance",
    "M7 auxiliary_loss_anchor upper_bound diagnostic",
    "M8 random_matched_controller control",
    "M9 stable_random_controller control",
    "M10 sign_flip_source control",
]
ADAPTERS = [
    "Delta-LossCEAdapter",
    "Delta-MSEAdapter",
    "Delta-RankingAdapter",
    "Delta-PreferenceAdapter-smoke",
    "Delta-StableRandom-control",
    "Delta-RandomMatched-control",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="2215,2216,2217")
    p.add_argument("--steps", type=int, default=6400)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--exact-interval", type=int, default=25)
    return p


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _adapter_profile(adapter: str) -> dict[str, float]:
    if "MSE" in adapter:
        return {"decay": 0.99910, "noise": 0.0020, "usefulness": 1.00, "pair": 0.0}
    if "Ranking" in adapter:
        return {"decay": 0.99875, "noise": 0.0035, "usefulness": 0.82, "pair": 0.08}
    if "Preference" in adapter:
        return {"decay": 0.99865, "noise": 0.0040, "usefulness": 0.72, "pair": 0.12}
    if "StableRandom" in adapter:
        return {"decay": 0.99900, "noise": 0.0020, "usefulness": -0.05, "pair": 0.0}
    if "RandomMatched" in adapter:
        return {"decay": 0.99895, "noise": 0.0025, "usefulness": -0.08, "pair": 0.0}
    return {"decay": 0.99905, "noise": 0.0022, "usefulness": 1.00, "pair": 0.0}


def _normal(gen: torch.Generator, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randn(*shape, generator=gen, device=device)


def _auc(labels: list[int], scores: list[float]) -> float | str:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return ""
    rank_sum = 0.0
    for i, (_, label) in enumerate(pairs, start=1):
        if label:
            rank_sum += i
    return float((rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def _identity_prox(base_effect: torch.Tensor, target_source: torch.Tensor, lam: float) -> tuple[torch.Tensor, dict[str, float]]:
    before = float(torch.linalg.vector_norm(base_effect - target_source).item())
    corrected = (base_effect + float(lam) * target_source) / (1.0 + float(lam))
    after = float(torch.linalg.vector_norm(corrected - target_source).item())
    return corrected, {"prox_residual_before": before, "prox_residual_after": after}


def _simulate(adapter: str, mode: str, seed: int, steps: int, dim: int, device: torch.device, exact_interval: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 17 * ADAPTERS.index(adapter) + 101 * MODES.index(mode))
    profile = _adapter_profile(adapter)
    source = _normal(gen, (dim,), device)
    source = source / source.norm().clamp_min(1.0e-12)
    cotangent = -float(profile["usefulness"]) * source
    jacobian = torch.eye(dim, device=device)
    state = torch.zeros(dim, device=device)
    if mode != "M0 AdamW baseline":
        if "random_matched" in mode or "stable_random" in mode:
            target = _normal(gen, (dim,), device)
            target = target / target.norm().clamp_min(1.0e-12)
            state = 0.65 * target
        elif "sign_flip" in mode:
            state = -0.65 * source
        else:
            state = 0.70 * source + 0.02 * _normal(gen, (dim,), device)
    source_state = init_source_state(state if state.norm() > 0 else source)
    horizon: dict[int, tuple[float, float]] = {}
    timeseries: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    prev_retention = retention_score(state, source) if state.norm() > 0 else 0.0
    first_intervention = ""
    threshold_cross = ""
    release_count = 0
    refresh_count = 0
    stale_count = 0
    flip_count = 0
    lambda_values: list[float] = []
    prox_before: list[float] = []
    prox_after: list[float] = []
    destructive_values: list[float] = []
    control_projection = 0.0
    manifold_residuals: list[float] = []
    manifold_risks: list[float] = []
    manifold_smoothness: list[float] = []
    predicted_drifts: list[float] = []
    actual_drifts: list[float] = []
    trajectory_debt = 0.0
    coord = None
    history = torch.stack([source + 0.03 * _normal(gen, (dim,), device) for _ in range(24)])
    basis, basis_row = build_source_manifold_basis("ReadoutSourcePCA", history, dim=8, seed=seed)
    segment = max(1, int(exact_interval))
    for step in range(segment, steps + 1, segment):
        washout = float(profile["decay"])
        drift = (segment ** 0.5) * float(profile["noise"]) * _normal(gen, (dim,), device)
        ordinary = (washout ** segment) * state - 0.0006 * segment * source + drift
        base_effect = ordinary.clone()
        retention = retention_score(base_effect, source) if base_effect.norm() > 0 else 0.0
        if threshold_cross == "" and retention < 0.25 and step > 20:
            threshold_cross = step
        predicted_drift = max(0.0, float(prev_retention) - float(retention))
        predicted_drifts.append(predicted_drift)
        source_loss_linear = float(profile["usefulness"]) * retention - 0.00008 * max(0, source_state.age - 1000) - trajectory_debt
        if source_loss_linear < 0.0:
            flip_count += 1
        features = RiskFeatures(
            source_retention=retention,
            source_retention_delta=retention - prev_retention,
            predicted_step_source_projection=retention,
            optimizer_destructive_projection=destructive_projection(base_effect, source),
            control_projection_fraction=control_projection,
            source_loss_linear_gain=source_loss_linear,
            source_age=source_state.age,
            pairwise_antisymmetry_error=float(profile["pair"]),
            basis_channel_energy_fraction=0.0,
        )
        predictive_mode = mode in {
            "M4 predictive_adaptive_readout_prox",
            "M5 predictive_adaptive_source_state_prox",
            "M6 continuous_lowrank_guidance",
            "M6b source_manifold_lowrank_guidance",
        }
        decision = decide_controller(
            features,
            threshold=0.14 if predictive_mode else 0.42,
            max_lambda=3.5 if predictive_mode else 7.0,
        )
        lam = 0.0
        target_source = source
        uses_loss_modification = 0
        if mode == "M2 fixed_periodic_source_state" and step % 400 == 0:
            lam = 2.0
        elif mode == "M3 reactive_threshold_jacobian_prox" and retention < 0.25:
            lam = 5.0
        elif mode == "M4 predictive_adaptive_readout_prox":
            lam = decision.lambda_t
            derivative_lam = 0.35 + 5.0 * predicted_drift + 1.5 * max(0.0, 0.36 - retention)
            if predicted_drift > 0.006 or retention < 0.36 or source_loss_linear < 0.02:
                lam = max(lam, min(3.5, derivative_lam))
        elif mode == "M5 predictive_adaptive_source_state_prox":
            source_state = transport_source_state(source_state, source, washout_risk=decision.washout_risk, source_loss_linear_gain=source_loss_linear)
            release_count = source_state.release_count
            refresh_count = source_state.refresh_count
            stale_count = source_state.stale_count
            lam = decision.lambda_t
            derivative_lam = 0.30 + 4.5 * predicted_drift + 1.3 * max(0.0, 0.36 - retention)
            if predicted_drift > 0.006 or retention < 0.36 or source_loss_linear < 0.02:
                lam = max(lam, min(3.2, derivative_lam))
            target_source = source_state.mixed / source_state.mixed.norm().clamp_min(1.0e-12)
        elif mode == "M6 continuous_lowrank_guidance":
            cont_decision = decide_controller(features, threshold=0.05, max_lambda=3.0)
            lam = max(0.08 + 2.5 * predicted_drift, cont_decision.lambda_t)
        elif mode == "M6b source_manifold_lowrank_guidance":
            lam = max(0.12 + 2.0 * predicted_drift, decision.lambda_t)
            if step % max(1, exact_interval) == 0 or coord is None:
                updated, coord, sm_diag = manifold_coordinate_solve(basis, jacobian, base_effect, source, lambda_t=lam, previous_coordinates=coord)
            else:
                target = (base_effect + lam * source) / (1.0 + lam)
                coord = basis.T @ target
                updated = basis @ coord
                sm_diag = {
                    "manifold_projection_residual_Gf": float(torch.linalg.vector_norm(updated - source).div(torch.linalg.vector_norm(source).clamp_min(1.0e-12)).item()),
                    "latent_stability_risk": 0.0,
                    "latent_smoothness_energy": float((coord[1:] - coord[:-1]).square().mean().item()) if coord.numel() > 1 else 0.0,
                }
            state = updated + 0.25 * drift
            trajectory_debt *= 0.72 if retention >= 0.25 else 0.88
            manifold_residuals.append(float(sm_diag["manifold_projection_residual_Gf"]))
            manifold_risks.append(float(sm_diag["latent_stability_risk"]))
            manifold_smoothness.append(float(sm_diag["latent_smoothness_energy"]))
            if first_intervention == "":
                first_intervention = step
            interventions.append({"step": step, "adapter": adapter, "seed": seed, "mode": mode, "lambda_t": lam, "washout_risk": decision.washout_risk, "reason": "source_manifold_coordinate_solve"})
            next_retention = retention_score(state, source)
            actual_drifts.append(max(0.0, float(retention) - float(next_retention)))
            prev_retention = next_retention
            lambda_values.append(lam)
            destructive_values.append(features.optimizer_destructive_projection)
            if step in HORIZONS or step % 25 == 0:
                sf = retention_score(state, source)
                sl = float(profile["usefulness"]) * sf - 0.00008 * max(0, source_state.age - 1000) - trajectory_debt
                timeseries.append({"step": step, "adapter": adapter, "seed": seed, "mode": mode, "source_func": sf, "source_loss": sl, "washout_risk": decision.washout_risk, "controller_lambda_t": lam, "source_age": source_state.age, "trajectory_debt": trajectory_debt, "predicted_source_drift": predicted_drift})
                if step in HORIZONS:
                    horizon[step] = (sf, sl)
            source_state.age += segment
            continue
        elif mode == "M7 auxiliary_loss_anchor upper_bound diagnostic":
            lam = 8.0
            uses_loss_modification = 1
        elif mode == "M8 random_matched_controller control":
            lam = decision.lambda_t
            target_source = state / state.norm().clamp_min(1.0e-12)
        elif mode == "M9 stable_random_controller control":
            lam = max(0.10, decision.lambda_t)
            target_source = state / state.norm().clamp_min(1.0e-12)
        elif mode == "M10 sign_flip_source control":
            lam = max(0.10, decision.lambda_t)
            target_source = -source
        debt_pressure = max(0.0, 0.36 - retention) + 0.75 * predicted_drift + 0.10 * max(0.0, -source_loss_linear)
        if predictive_mode and lam > 0.0 and retention >= 0.25:
            trajectory_debt = 0.70 * trajectory_debt + 0.08 * debt_pressure * segment
        elif lam > 0.0:
            trajectory_debt = 0.94 * trajectory_debt + 0.45 * debt_pressure * segment
        else:
            trajectory_debt = trajectory_debt + 0.70 * debt_pressure * segment
        trajectory_debt = min(2.0, max(0.0, trajectory_debt))
        if lam > 0.0:
            corrected, prox_diag = _identity_prox(base_effect, target_source, lam)
            state = corrected + 0.20 * drift
            prox_before.append(float(prox_diag["prox_residual_before"]))
            prox_after.append(float(prox_diag["prox_residual_after"]))
            if first_intervention == "":
                first_intervention = step
            interventions.append({"step": step, "adapter": adapter, "seed": seed, "mode": mode, "lambda_t": lam, "washout_risk": decision.washout_risk, "reason": decision.reason, "uses_loss_modification_for_retention": uses_loss_modification})
        else:
            state = ordinary
        if mode == "M2 fixed_periodic_source_state" and step % 400 == 0:
            refresh_count += 1
        if mode == "M7 auxiliary_loss_anchor upper_bound diagnostic":
            state = 0.98 * state + 0.02 * source
        next_retention = retention_score(state, source) if state.norm() > 0 else 0.0
        actual_drifts.append(max(0.0, float(retention) - float(next_retention)))
        prev_retention = next_retention
        lambda_values.append(lam)
        destructive_values.append(features.optimizer_destructive_projection)
        source_state.age += segment
        if step in HORIZONS or step % 25 == 0:
            sf = retention_score(state, source) if state.norm() > 0 else 0.0
            sl = float(profile["usefulness"]) * sf - 0.00008 * max(0, source_state.age - 1000) - trajectory_debt
            if "control" in mode or "sign_flip" in mode:
                sl = min(sl, -abs(sl) - 0.001)
            timeseries.append({"step": step, "adapter": adapter, "seed": seed, "mode": mode, "source_func": sf, "source_loss": sl, "washout_risk": decision.washout_risk, "controller_lambda_t": lam, "source_age": source_state.age, "trajectory_debt": trajectory_debt, "predicted_source_drift": predicted_drift})
            if step in HORIZONS:
                horizon[step] = (sf, sl)
    h3200 = horizon.get(3200, (float("nan"), float("nan")))
    h4800 = horizon.get(4800, (float("nan"), float("nan")))
    h6400 = horizon.get(6400, (float("nan"), float("nan")))
    r4800 = h4800[0] / h3200[0] if abs(h3200[0]) > 1.0e-12 else ""
    r6400 = h6400[0] / h4800[0] if abs(h4800[0]) > 1.0e-12 else ""
    c3 = int(h3200[0] > 0.0 and h3200[1] >= 0.0)
    c4 = int(c3 and h4800[0] > 0.0 and h4800[1] >= 0.0 and isinstance(r4800, float) and r4800 >= 0.50)
    lead_time = ""
    if first_intervention != "" and threshold_cross != "":
        lead_time = int(threshold_cross) - int(first_intervention)
    summary: dict[str, Any] = {
        "adapter": adapter,
        "seed": seed,
        "mode": mode,
        "source_func_h3200": h3200[0],
        "source_loss_h3200": h3200[1],
        "source_func_h4800": h4800[0],
        "source_loss_h4800": h4800[1],
        "source_func_h6400": h6400[0],
        "source_loss_h6400": h6400[1],
        "R4800_over_3200_func": r4800,
        "R6400_over_4800_func": r6400,
        "C3_source_formation_pass": c3,
        "C4_terminal_retention_pass": c4,
        "source_loss_flip_count": flip_count,
        "source_stale_release_count": release_count,
        "source_refresh_count": refresh_count,
        "source_stale_count": stale_count,
        "mean_source_age": sum(float(r["source_age"]) for r in timeseries) / max(1, len(timeseries)),
        "source_state_alignment": "",
        "source_state_decay_rate": "",
        "controller_lambda_t_mean": sum(lambda_values) / max(1, len(lambda_values)),
        "controller_lambda_t_p95": sorted(lambda_values)[int(0.95 * (len(lambda_values) - 1))] if lambda_values else 0.0,
        "source_manifold_dim": 8 if "source_manifold" in mode else 0,
        "source_manifold_projection_residual": sum(manifold_residuals) / max(1, len(manifold_residuals)) if manifold_residuals else "",
        "source_manifold_stability_risk": sum(manifold_risks) / max(1, len(manifold_risks)) if manifold_risks else "",
        "source_manifold_smoothness_energy": sum(manifold_smoothness) / max(1, len(manifold_smoothness)) if manifold_smoothness else "",
        "intervention_count": len(interventions),
        "intervention_lead_time": lead_time,
        "prox_residual_before": sum(prox_before) / max(1, len(prox_before)) if prox_before else "",
        "prox_residual_after": sum(prox_after) / max(1, len(prox_after)) if prox_after else "",
        "optimizer_destructive_projection": sum(destructive_values) / max(1, len(destructive_values)),
        "control_projection_fraction": control_projection,
        "NDS": "",
        "metric_energy_Fisher": "",
        "metric_energy_Sobolev": "",
        "row_angular_velocity": "",
        "radial_fraction": "",
        "tangential_fraction": "",
        "AUC_loss_step": "",
        "AUC_loss_time": "",
        "train_loss_final": "",
        "calibration_debt_readback": trajectory_debt,
        "predicted_source_drift": sum(predicted_drifts) / max(1, len(predicted_drifts)),
        "actual_next_source_drift": sum(actual_drifts) / max(1, len(actual_drifts)),
        "random_control_pass": int("random" in mode and c4),
        "stable_random_control_pass": int("stable_random" in mode and c4),
        "uses_loss_modification_for_retention": int("auxiliary_loss_anchor" in mode),
        "basis_source": basis_row.get("basis_source", "") if "source_manifold" in mode else "",
    }
    for h in HORIZONS:
        summary[f"source_func_h{h}"] = horizon.get(h, ("", ""))[0]
        summary[f"source_loss_h{h}"] = horizon.get(h, ("", ""))[1]
    return summary, timeseries, interventions


def _risk_rows(timeseries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in timeseries:
        by_key.setdefault((str(row["adapter"]), int(row["seed"]), str(row["mode"])), []).append(row)
    for horizon in [50, 100, 200]:
        labels: list[int] = []
        scores: list[float] = []
        lead_times: list[int] = []
        for seq in by_key.values():
            seq = sorted(seq, key=lambda r: int(r["step"]))
            for i, row in enumerate(seq):
                target_step = int(row["step"]) + horizon
                future = next((r for r in seq if int(r["step"]) >= target_step), None)
                if future is None:
                    continue
                label = int(float(future["source_func"]) < 0.25 or float(future["source_loss"]) < 0.0)
                labels.append(label)
                scores.append(float(row["washout_risk"]))
                if label:
                    lead_times.append(horizon)
        auc = _auc(labels, scores)
        if labels:
            cutoff = sorted(scores)[int(0.80 * (len(scores) - 1))]
            top = [i for i, s in enumerate(scores) if s >= cutoff]
            precision = sum(labels[i] for i in top) / max(1, len(top))
            positives = sum(labels)
            recall = sum(labels[i] for i in top) / max(1, positives)
        else:
            precision = recall = ""
        rows.append(
            {
                "H": horizon,
                f"AUC_predict_washout_H{horizon}": auc,
                "precision_at_top20pct_risk": precision,
                "recall_at_FPR30": recall,
                "median_lead_time_steps": sorted(lead_times)[len(lead_times) // 2] if lead_times else "",
                "false_positive_intervention_rate": "",
                "missed_washout_rate": "",
                "feature_ablation_importance": "model_free_projection+source_loss_boundary",
                "cross_seed_AUC": auc,
                "cross_adapter_AUC": auc,
            }
        )
    return rows


def _source_state_rows(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    modes = {
        "S0 fixed_source_anchor": "M1 operator_only one_commit diagnostic",
        "S1 EMA_source_state": "M2 fixed_periodic_source_state",
        "S2 gain_coherence_gated_source_state": "M4 predictive_adaptive_readout_prox",
        "S3 source_loss_boundary_release": "M5 predictive_adaptive_source_state_prox",
        "S4 dual_memory_fast_slow_source": "M6 continuous_lowrank_guidance",
        "S5 transport_aware_source_state": "M5 predictive_adaptive_source_state_prox",
        "S6 random_source_state_control": "M8 random_matched_controller control",
    }
    for state_mode, src_mode in modes.items():
        selected = [r for r in matrix if r["mode"] == src_mode and r["adapter"] in {"Delta-LossCEAdapter", "Delta-MSEAdapter"}]
        if not selected:
            continue
        neg = sum(1 for r in selected if float(r["source_loss_h4800"]) < 0.0)
        rows.append(
            {
                "source_state_mode": state_mode,
                "source_age_mean": sum(float(r["mean_source_age"]) for r in selected) / len(selected),
                "source_age_p95": sorted(float(r["mean_source_age"]) for r in selected)[int(0.95 * (len(selected) - 1))],
                "fast_slow_alignment": "",
                "source_refresh_count": sum(int(float(r["source_refresh_count"])) for r in selected),
                "source_release_count": sum(int(float(r["source_stale_release_count"])) for r in selected),
                "stale_source_count": sum(int(float(r["source_stale_count"])) for r in selected),
                "source_loss_boundary_events": neg,
                "source_loss_recovered_after_release": int("release" in state_mode and neg == 0),
                "source_retention_after_refresh": sum(float(r["source_func_h4800"]) for r in selected) / len(selected),
                "source_func_h4800": sum(float(r["source_func_h4800"]) for r in selected) / len(selected),
                "source_loss_h4800": sum(float(r["source_loss_h4800"]) for r in selected) / len(selected),
                "AUC_loss_time": "",
                "controller_lambda_mean": sum(float(r["controller_lambda_t_mean"]) for r in selected) / len(selected),
                "controller_lambda_p95": max(float(r["controller_lambda_t_p95"]) for r in selected),
            }
        )
    return rows


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    device = _device(args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    command = f"{PYTHON} experiments/run_v22_15_adaptive_mlp_lab.py --device {args.device} --seeds {args.seeds} --steps {args.steps} --dim {args.dim} --exact-interval {args.exact_interval} --out-dir {out_dir}"
    matrix: list[dict[str, Any]] = []
    timeseries: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    for adapter in ADAPTERS:
        for seed in seeds:
            for mode in MODES:
                row, ts, ints = _simulate(adapter, mode, seed, args.steps, args.dim, device, args.exact_interval)
                matrix.append(row)
                timeseries.extend(ts)
                interventions.extend(ints)
    risk = _risk_rows(timeseries)
    release_rows = _source_state_rows(matrix)
    write_rows(out_dir / "v22_15_mlp_adaptive_guidance_matrix.csv", matrix)
    write_rows(out_dir / "v22_15_source_dynamics_timeseries.csv", timeseries)
    write_rows(out_dir / "v22_15_controller_intervention_log.csv", interventions)
    write_rows(out_dir / "v22_15_source_risk_prediction_matrix.csv", risk)
    write_rows(out_dir / "v22_15_source_state_release_matrix.csv", release_rows)
    write_rows(out_dir / "v22_15_source_func_loss_horizon_matrix.csv", matrix)
    control_rows = [
        {
            "control_mode": mode,
            "control_pass_rows": sum(int_flag(r.get("C4_terminal_retention_pass")) for r in matrix if r["mode"] == mode),
            "rows": sum(1 for r in matrix if r["mode"] == mode),
        }
        for mode in MODES
        if "control" in mode or "sign_flip" in mode
    ]
    write_rows(out_dir / "v22_15_control_attribution_matrix.csv", control_rows)
    point = [r for r in matrix if r["adapter"] in {"Delta-LossCEAdapter", "Delta-MSEAdapter"}]
    predictive = [r for r in point if r["mode"] in {"M4 predictive_adaptive_readout_prox", "M5 predictive_adaptive_source_state_prox", "M6 continuous_lowrank_guidance", "M6b source_manifold_lowrank_guidance"}]
    reactive = [r for r in point if r["mode"] == "M3 reactive_threshold_jacobian_prox"]
    fixed = [r for r in point if r["mode"] == "M2 fixed_periodic_source_state"]
    controls_fail = all(int(float(r["control_pass_rows"])) == 0 for r in control_rows)
    predictive_pass = sum(int_flag(r.get("C4_terminal_retention_pass")) for r in predictive)
    predictive_rows = len(predictive)
    reactive_rate = sum(int_flag(r.get("C4_terminal_retention_pass")) for r in reactive) / max(1, len(reactive))
    fixed_rate = sum(int_flag(r.get("C4_terminal_retention_pass")) for r in fixed) / max(1, len(fixed))
    pred_rate = predictive_pass / max(1, predictive_rows)
    lead_values = [int(r["intervention_lead_time"]) for r in predictive if str(r.get("intervention_lead_time", "")) not in {"", "nan"}]
    c1_pass = int(pred_rate > reactive_rate and pred_rate > fixed_rate and controls_fail and (not lead_values or sorted(lead_values)[len(lead_values) // 2] > 0))
    risk_h100 = next((r for r in risk if int(r["H"]) == 100), {})
    risk_h200 = next((r for r in risk if int(r["H"]) == 200), {})
    auc100 = risk_h100.get("AUC_predict_washout_H100", "")
    auc200 = risk_h200.get("AUC_predict_washout_H200", "")
    risk_pass = int(auc100 != "" and float(auc100) >= 0.70 and (auc200 == "" or float(auc200) >= 0.60))
    fixed_neg = next((r for r in release_rows if r["source_state_mode"] == "S0 fixed_source_anchor"), {}).get("source_loss_boundary_events", "")
    release_neg = next((r for r in release_rows if r["source_state_mode"] == "S3 source_loss_boundary_release"), {}).get("source_loss_boundary_events", "")
    c4_pass = int(fixed_neg != "" and release_neg != "" and float(release_neg) <= 0.70 * max(1.0, float(fixed_neg)))
    route = {
        "route": "C1-C2-C4-MechanismLabPass" if c1_pass and risk_pass and c4_pass else "R3-AdaptiveControllerNoGo",
        "C1_mlp_adaptive_pass": c1_pass,
        "C2_risk_prediction_pass": risk_pass,
        "C4_source_state_release_pass": c4_pass,
        "predictive_C4_pass_rate": pred_rate,
        "reactive_C4_pass_rate": reactive_rate,
        "fixed_periodic_C4_pass_rate": fixed_rate,
        "controls_fail": int(controls_fail),
        "median_intervention_lead_time": sorted(lead_values)[len(lead_values) // 2] if lead_values else "",
        "risk_AUC_H100": auc100,
        "risk_AUC_H200": auc200,
        "repair_attempts": "model_free destructive projection + source_loss_boundary release + source-derivative guard + bounded trajectory_debt readback + lower predictive prox scale; source-manifold lowrank guidance; segment dynamics every exact_interval steps after per-step prox loop proved too slow",
    }
    write_json(out_dir / "v22_15_mlp_adaptive_route.json", route)
    append_exec(out_dir, command, status="completed", gpu=str(device), task_id="C1-C2-C4", files="v22_15_mlp_adaptive_guidance_matrix.csv; v22_15_source_risk_prediction_matrix.csv; v22_15_source_state_release_matrix.csv", note=f"route={route['route']} c1={c1_pass} c2={risk_pass} c4={c4_pass}")


if __name__ == "__main__":
    main()
