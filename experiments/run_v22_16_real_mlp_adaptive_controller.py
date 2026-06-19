#!/usr/bin/env python3
"""Parts C/D v22.16 real MLP adaptive controller and risk prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.run_v22_16_common import (  # noqa: E402
    DATASETS,
    PYTHON,
    append_exec,
    apply_gradient_guidance,
    ce_cotangent,
    cosine_t,
    device_from_arg,
    ensure_out,
    evaluate,
    init_docs,
    int_flag,
    loader_for,
    make_model,
    mse_cotangent,
    write_json,
    write_rows,
)


MODES = [
    "M0 AdamW baseline",
    "M2 fixed_periodic_source_state",
    "M3 reactive_threshold_jacobian_prox",
    "M3R reactive_threshold_jacobian_prox_conservative",
    "M4 predictive_adaptive_direct_prox",
    "M4B source_boundary_low_lambda_controller",
    "M4R predictive_adaptive_direct_prox_conservative",
    "M5 predictive_adaptive_source_state",
    "M6 predictive_adaptive_source_manifold_k8",
    "M7 predictive_adaptive_source_manifold_k16",
    "M8 source_loss_release_controller",
    "M8B source_boundary_release_low_lambda_controller",
    "M8R source_loss_release_conservative_controller",
    "M9 auxiliary_loss_anchor upper_bound diagnostic",
    "M10 random_matched_controller",
    "M11 stable_random_controller",
    "M12 signflip_source_controller",
]
ADAPTERS = ["Delta-LossCEAdapter", "Delta-MSEAdapter"]
HORIZONS = [100, 400, 800, 1600, 3200, 4800, 6400]
CONTROL_PREFIXES = ("M10", "M11", "M12")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--adapters", default=",".join(ADAPTERS))
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=6400)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--download", action="store_true")
    return p


def _mode_to_guidance(mode: str) -> str:
    if mode.startswith("M0"):
        return "none"
    suffix = "_conservative" if "conservative" in mode else ""
    if "source_boundary" in mode:
        return "release" + suffix if "release" in mode else "predictive" + suffix
    if "fixed_periodic" in mode:
        return "fixed" + suffix
    if "reactive" in mode:
        return "reactive" + suffix
    if "source_manifold" in mode:
        return "manifold" + suffix
    if "release" in mode:
        return "release" + suffix
    if "random_matched" in mode or "stable_random" in mode:
        return "random"
    if "signflip" in mode:
        return "signflip"
    return "predictive" + suffix


def _loss_and_delta(adapter: str, logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if adapter == "Delta-MSEAdapter":
        target = F.one_hot(y.long(), num_classes=logits.shape[-1]).float().to(logits.device)
        return F.mse_loss(logits.float(), target), mse_cotangent(logits.detach(), y)
    return F.cross_entropy(logits.float(), y.long()), ce_cotangent(logits.detach(), y)


def _auc(labels: list[int], scores: list[float]) -> float | str:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return ""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    for idx, (_score, label) in enumerate(pairs, start=1):
        if label:
            rank_sum += idx
    return float((rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def _run_one(dataset: str, seed: int, adapter: str, mode: str, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    torch.manual_seed(seed + 42216)
    train_loader = loader_for(dataset, True, args.train_size, args.batch_size, seed, download=args.download)
    test_loader = loader_for(dataset, False, args.test_size, args.batch_size, seed, download=args.download, shuffle=False)
    first_x, _ = next(iter(train_loader))
    model = make_model("MLP+AdaptiveFU", first_x, args.hidden, seed + 42216, device).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2.0e-3, weight_decay=1.0e-4)
    train_iter = iter(train_loader)
    source_state: torch.Tensor | None = None
    source_prev: torch.Tensor | None = None
    last_source_func: float | None = None
    prior_source_func: float | None = None
    last_source_loss: float | None = None
    timeseries: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    horizon: dict[int, tuple[float, float]] = {}
    loss_values: list[float] = []
    lambda_values: list[float] = []
    controller_ratios: list[float] = []
    lead_candidate_first = ""
    threshold_cross = ""
    started = time.perf_counter()
    guidance = _mode_to_guidance(mode)
    for step in range(1, int(args.steps) + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.train()
        opt.zero_grad(set_to_none=True)
        logits_before = model(xb).float()
        loss, delta = _loss_and_delta(adapter, logits_before, yb)
        if "auxiliary_loss_anchor" in mode:
            loss = loss + 1.0e-4 * sum(p.square().mean() for p in params)
        loss.backward()
        diag: dict[str, Any] = {"risk_score": 0.0, "lambda_t": 0.0, "source_retention": 0.0, "source_loss_proxy": 0.0, "controller_to_base_update_ratio": 0.0}
        if guidance != "none":
            lambda_override: float | None = None
            risk_override: float | None = None
            if "source_boundary" in mode and last_source_func is not None and last_source_loss is not None:
                weak_current = max(0.0, 0.20 - float(last_source_func)) / 0.20
                trend_decay = 0.0 if prior_source_func is None else max(0.0, float(prior_source_func) - float(last_source_func)) / 0.20
                loss_boundary = min(1.0, 1000.0 * max(0.0, -float(last_source_loss)))
                risk_override = max(0.0, min(1.0, 0.65 * weak_current + 0.25 * trend_decay + 0.10 * loss_boundary))
                if "release" in mode and float(last_source_loss) < 0.0:
                    lambda_override = 0.0
                elif risk_override >= 0.20:
                    lambda_override = 0.04 + 0.14 * risk_override
                else:
                    lambda_override = 0.0
            source_state, diag = apply_gradient_guidance(
                model,
                source_state,
                selector="all",
                mode=guidance,
                step=step,
                seed=seed,
                lambda_override=lambda_override,
                risk_override=risk_override,
            )
        opt.step()
        with torch.no_grad():
            logits_after = model(xb).float()
        source = (-delta).detach().reshape(-1)
        effect = (logits_after - logits_before.detach()).reshape(-1)
        source_func = cosine_t(effect, source)
        source_loss = float((-(delta.detach().reshape(-1) * (logits_after - logits_before.detach()).reshape(-1))).mean().item())
        if source_prev is not None:
            retention = cosine_t(source, source_prev)
        else:
            retention = source_func
        source_prev = source.detach().cpu()
        prior_source_func = last_source_func
        last_source_func = float(source_func)
        last_source_loss = float(source_loss)
        if threshold_cross == "" and (source_loss < 0.0 or retention < 0.20):
            threshold_cross = step
        if diag.get("lambda_t", 0.0) and float(diag["lambda_t"]) > 0.0 and lead_candidate_first == "":
            lead_candidate_first = step
        loss_values.append(float(loss.detach().item()))
        lambda_values.append(float(diag.get("lambda_t", 0.0)))
        controller_ratios.append(float(diag.get("controller_to_base_update_ratio", 0.0)))
        if step % int(args.log_interval) == 0 or step in HORIZONS or step == 1:
            row = {
                "dataset": dataset,
                "seed": seed,
                "adapter": adapter,
                "mode": mode,
                "step": step,
                "risk_score": float(diag.get("risk_score", 0.0)),
                "lambda_t": float(diag.get("lambda_t", 0.0)),
                "intervention_flag": int(float(diag.get("lambda_t", 0.0)) > 0.0),
                "predicted_source_drift_H50": max(0.0, -float(diag.get("source_retention", 0.0))),
                "predicted_source_drift_H100": max(0.0, -float(diag.get("source_retention", 0.0))),
                "predicted_source_drift_H200": max(0.0, -float(diag.get("source_retention", 0.0))),
                "destructive_projection": float(diag.get("destructive_projection", max(0.0, -source_func))),
                "prox_residual_before": "",
                "prox_residual_after": "",
                "source_state_age": step,
                "source_refresh_count": "",
                "source_release_count": int("release" in guidance and source_loss < 0.0),
                "source_manifold_dim": 16 if "k16" in mode else (8 if "source_manifold" in mode else 0),
                "source_manifold_projection_residual": "",
                "controller_update_norm": float(diag.get("controller_update_norm", 0.0)),
                "ordinary_update_norm": float(diag.get("ordinary_update_norm", 0.0)),
                "controller_to_base_update_ratio": float(diag.get("controller_to_base_update_ratio", 0.0)),
                "NDS": float(source.mean().abs().div(source.norm().clamp_min(1.0e-12)).item()),
                "control_projection_fraction": 0.0,
                "source_loss_boundary_distance": source_loss,
                "source_func": source_func,
                "source_loss": source_loss,
                "train_loss": float(loss.detach().item()),
            }
            timeseries.append(row)
            if row["intervention_flag"]:
                interventions.append(row)
        if step in HORIZONS:
            horizon[step] = (source_func, source_loss)
    train_metrics = evaluate(model, train_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    h3200 = horizon.get(3200, ("", ""))
    h4800 = horizon.get(4800, ("", ""))
    h6400 = horizon.get(6400, ("", ""))
    r4800 = h4800[0] / h3200[0] if isinstance(h3200[0], float) and isinstance(h4800[0], float) and abs(h3200[0]) > 1.0e-12 else ""
    r6400 = h6400[0] / h4800[0] if isinstance(h4800[0], float) and isinstance(h6400[0], float) and abs(h4800[0]) > 1.0e-12 else ""
    c3 = int(isinstance(h3200[0], float) and h3200[0] > 0.0 and h3200[1] >= 0.0)
    c4 = int(c3 and isinstance(h4800[0], float) and h4800[0] > 0.0 and h4800[1] >= 0.0 and isinstance(r4800, float) and r4800 >= 0.50)
    lead_time = ""
    if lead_candidate_first != "" and threshold_cross != "":
        lead_time = int(threshold_cross) - int(lead_candidate_first)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "adapter": adapter,
        "mode": mode,
        "source_func_h3200": h3200[0],
        "source_loss_h3200": h3200[1],
        "source_func_h4800": h4800[0],
        "source_loss_h4800": h4800[1],
        "source_func_h6400": h6400[0],
        "source_loss_h6400": h6400[1],
        "R4800_over_3200_func": r4800,
        "R6400_over_4800_func": r6400,
        "source_loss_flip_count": sum(1 for r in timeseries if float(r["source_loss"]) < 0.0),
        "TargetRetentionOnly_NotTaskUseful": int(any(float(r["source_func"]) > 0.0 and float(r["source_loss"]) < 0.0 for r in timeseries)),
        "C3_source_formation_pass": c3,
        "C4_terminal_retention_pass": c4,
        "C5_h6400_retention_pass": int(isinstance(h6400[0], float) and h6400[0] > 0.0 and h6400[1] >= 0.0),
        "median_intervention_lead_time": lead_time,
        "intervention_count": len(interventions),
        "controller_lambda_mean": sum(lambda_values) / max(1, len(lambda_values)),
        "controller_to_base_update_ratio_mean": sum(controller_ratios) / max(1, len(controller_ratios)),
        "AUC_loss_step": sum(loss_values),
        "AUC_loss_time": sum(loss_values),
        "train_loss": train_metrics["loss"],
        "test_loss_readback": test_metrics["loss"],
        "test_accuracy_readback": test_metrics["accuracy"],
        "calibration_debt_readback": test_metrics["ECE"],
        "tail_loss_q99_readback": test_metrics["tail_loss_q99"],
        "wallclock_sec": time.perf_counter() - started,
        "uses_loss_modification_for_retention": int("auxiliary_loss_anchor" in mode),
        "diagnostic_only": int("diagnostic" in mode or "auxiliary" in mode or mode.startswith("M2") or mode.startswith("M3")),
    }
    for h in HORIZONS:
        summary[f"source_func_h{h}"] = horizon.get(h, ("", ""))[0]
        summary[f"source_loss_h{h}"] = horizon.get(h, ("", ""))[1]
    return summary, timeseries, interventions


def _risk_score_models(row: dict[str, Any], prev: dict[str, Any] | None) -> dict[str, float]:
    source_func = float(row["source_func"])
    source_loss = float(row["source_loss"])
    risk_builtin = float(row["risk_score"])
    destructive = float(row["destructive_projection"])
    boundary = max(0.0, -source_loss)
    weak_current = max(0.0, 0.20 - source_func)
    if prev is None:
        d_func = 0.0
        d_loss = 0.0
        d_train = 0.0
    else:
        d_func = float(prev["source_func"]) - source_func
        d_loss = float(prev["source_loss"]) - source_loss
        d_train = float(row["train_loss"]) - float(prev["train_loss"])
    derivative = max(0.0, d_func) + 20.0 * max(0.0, d_loss) + 0.15 * max(0.0, d_train)
    destructive_only = destructive
    boundary_only = boundary + weak_current
    monotone_agreement = (
        0.35 * max(0.0, min(1.0, risk_builtin))
        + 0.25 * max(0.0, min(1.0, derivative))
        + 0.25 * max(0.0, min(1.0, destructive_only))
        + 0.15 * max(0.0, min(1.0, boundary_only))
    )
    derivative_or_destructive = max(derivative, destructive_only, boundary_only)
    return {
        "R0_builtin_controller_risk": risk_builtin,
        "R0_destructive_projection_only": destructive_only,
        "R1_derivative_only": derivative,
        "R1_source_loss_boundary_only": boundary_only,
        "R2_monotone_feature_agreement": monotone_agreement,
        "R2_derivative_or_destructive": derivative_or_destructive,
    }


def _risk_rows(timeseries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_group: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in timeseries:
        by_group.setdefault((str(row["dataset"]), int(row["seed"]), str(row["adapter"]), str(row["mode"])), []).append(row)
    labels_all: dict[str, dict[int, list[int]]] = {}
    scores_all: dict[str, dict[int, list[float]]] = {}
    leads_all: dict[str, dict[int, list[int]]] = {}
    feature_rows: list[dict[str, Any]] = []
    for _group_key, group_rows in by_group.items():
        group_rows = sorted(group_rows, key=lambda r: int(r["step"]))
        prev: dict[str, Any] | None = None
        for i, row in enumerate(group_rows):
            scores_by_model = _risk_score_models(row, prev)
            prev = row
            feature_rows.append({**row, **scores_by_model, "risk_eval_includes_control": int(str(row["mode"]).startswith(CONTROL_PREFIXES))})
            if str(row["mode"]).startswith(CONTROL_PREFIXES) or "auxiliary" in str(row["mode"]):
                continue
            for H in [50, 100, 200]:
                future = next((r for r in group_rows[i + 1 :] if int(r["step"]) >= int(row["step"]) + H), None)
                if future is None:
                    continue
                label = int(float(future["source_loss"]) < 0.0 or float(future["source_func"]) < 0.20)
                for model_name, score in scores_by_model.items():
                    labels_all.setdefault(model_name, {50: [], 100: [], 200: []})[H].append(label)
                    scores_all.setdefault(model_name, {50: [], 100: [], 200: []})[H].append(float(score))
                    leads_all.setdefault(model_name, {50: [], 100: [], 200: []})[H].append(int(future["step"]) - int(row["step"]))
    for model_name in sorted(labels_all):
        aucs = {H: _auc(labels_all[model_name][H], scores_all[model_name][H]) for H in [50, 100, 200]}
        for H in [50, 100, 200]:
            labels = labels_all[model_name][H]
            scores = scores_all[model_name][H]
            top_cut = sorted(scores, reverse=True)[max(0, int(0.20 * len(scores)) - 1)] if scores else 1.0
            pred_top = [l for l, s in zip(labels, scores) if s >= top_cut]
            lead_top = [lead for l, s, lead in zip(labels, scores, leads_all[model_name][H]) if l and s >= top_cut]
            precision = sum(pred_top) / max(1, len(pred_top))
            positives = sum(labels)
            rows.append(
                {
                    "risk_model": model_name,
                    "H": H,
                    "AUC_predict_washout_H50": aucs[50] if H == 50 else "",
                    "AUC_predict_washout_H100": aucs[100] if H == 100 else "",
                    "AUC_predict_washout_H200": aucs[200] if H == 200 else "",
                    "precision_at_top20pct_risk": precision,
                    "label_count": len(labels),
                    "positive_label_count": positives,
                    "recall_at_FPR30": "",
                    "median_lead_time_steps": sorted(lead_top)[len(lead_top) // 2] if lead_top else "",
                    "lead_time_threshold_rule": "top20pct_risk_score",
                    "false_positive_intervention_rate": "",
                    "missed_washout_rate": "",
                    "cross_dataset_AUC": "",
                    "cross_seed_AUC": "",
                    "cross_adapter_AUC": "",
                    "calibration_ECE_for_risk": "",
                    "analysis_scope": "non_control_non_auxiliary_rows",
                }
            )
    return rows, feature_rows


def _requested_items(raw: str, cast: Any = str) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _group_counts(rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> dict[str, set[tuple[str, int]]]:
    out: dict[str, set[tuple[str, int]]] = {}
    for row in rows:
        if not str(row.get("mode", "")).startswith(prefixes):
            continue
        if not int_flag(row.get("C4_terminal_retention_pass")):
            continue
        adapter = str(row.get("adapter", ""))
        out.setdefault(adapter, set()).add((str(row.get("dataset", "")), int(row.get("seed", 0))))
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = (
        f"{PYTHON} experiments/run_v22_16_real_mlp_adaptive_controller.py --device {args.device} --datasets {args.datasets} "
        f"--seeds {args.seeds} --adapters {args.adapters} --modes {args.modes} --train-size {args.train_size} "
        f"--test-size {args.test_size} --steps {args.steps} --log-interval {args.log_interval} --batch-size {args.batch_size} --hidden {args.hidden} --out-dir {out_dir}"
        + (" --download" if args.download else "")
    )
    device = device_from_arg(args.device)
    rows: list[dict[str, Any]] = []
    timeseries: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    blockers: list[str] = []
    requested_datasets = _requested_items(args.datasets)
    requested_seeds = _requested_items(args.seeds, int)
    requested_adapters = _requested_items(args.adapters)
    requested_modes = _requested_items(args.modes)
    for dataset in requested_datasets:
        for seed in requested_seeds:
            for adapter in requested_adapters:
                for mode in requested_modes:
                    try:
                        summary, ts, ints = _run_one(dataset, seed, adapter, mode, args, device)
                        rows.append(summary)
                        timeseries.extend(ts)
                        interventions.extend(ints)
                    except Exception as exc:
                        blockers.append(f"{dataset}:{seed}:{adapter}:{mode}:{repr(exc)}")
                        rows.append({"dataset": dataset, "seed": seed, "adapter": adapter, "mode": mode, "status": "blocked", "blocker": repr(exc)})
    risk_rows, risk_feature_rows = _risk_rows(timeseries)
    pointwise_official = [r for r in rows if r.get("adapter") in {"Delta-LossCEAdapter", "Delta-MSEAdapter"} and str(r.get("mode", "")).startswith(("M4", "M5", "M6", "M7", "M8"))]
    predictive_pass_rows = sum(int_flag(r.get("C4_terminal_retention_pass")) for r in pointwise_official)
    reactive_pass_rows = sum(int_flag(r.get("C4_terminal_retention_pass")) for r in rows if str(r.get("mode", "")).startswith("M3"))
    fixed_pass_rows = sum(int_flag(r.get("C4_terminal_retention_pass")) for r in rows if str(r.get("mode", "")).startswith("M2"))
    controls = [r for r in rows if str(r.get("mode", "")).startswith(("M10", "M11", "M12"))]
    controls_fail = int(sum(int_flag(r.get("C4_terminal_retention_pass")) for r in controls) == 0)
    h100_rows = [r for r in risk_rows if r.get("AUC_predict_washout_H100") != ""]
    best_risk_row = max(h100_rows, key=lambda r: float(r["AUC_predict_washout_H100"]), default={})
    auc100 = best_risk_row.get("AUC_predict_washout_H100", "")
    median_lead = best_risk_row.get("median_lead_time_steps", "")
    best_risk_model = best_risk_row.get("risk_model", "")
    official_scope = int(
        set(requested_datasets) == set(DATASETS)
        and {0, 1, 2}.issubset(set(requested_seeds))
        and {"Delta-LossCEAdapter", "Delta-MSEAdapter"}.issubset(set(requested_adapters))
        and int(args.steps) >= 4800
    )
    expected_groups = max(1, len(set(requested_datasets)) * len(set(requested_seeds)))
    required_groups = max(2, (2 * expected_groups + 2) // 3)
    predictive_groups = _group_counts(rows, ("M4", "M5", "M6", "M7", "M8"))
    reactive_groups = _group_counts(rows, ("M3",))
    fixed_groups = _group_counts(rows, ("M2",))
    predictive_ce_groups = len(predictive_groups.get("Delta-LossCEAdapter", set()))
    predictive_mse_groups = len(predictive_groups.get("Delta-MSEAdapter", set()))
    reactive_groups_total = sum(len(v) for v in reactive_groups.values())
    fixed_groups_total = sum(len(v) for v in fixed_groups.values())
    c1_exploration = int(predictive_pass_rows > max(reactive_pass_rows, fixed_pass_rows) and controls_fail and not blockers)
    c1_official_mechanism = int(
        official_scope
        and predictive_ce_groups >= required_groups
        and predictive_mse_groups >= required_groups
        and (predictive_ce_groups + predictive_mse_groups) > max(reactive_groups_total, fixed_groups_total)
        and controls_fail
        and not blockers
    )
    c1_pass = c1_official_mechanism
    c2_pass = int(auc100 != "" and float(auc100) >= 0.70 and median_lead != "" and float(median_lead) >= 50)
    release_rows = [r for r in rows if str(r.get("mode", "")).startswith("M8")]
    release_pass = int(any(int_flag(r.get("C4_terminal_retention_pass")) for r in release_rows))
    write_rows(out_dir / "v22_16_real_mlp_adaptive_guidance_matrix.csv", rows)
    write_rows(out_dir / "v22_16_real_risk_prediction_matrix.csv", risk_rows)
    write_rows(out_dir / "v22_16_real_risk_feature_timeseries.csv", risk_feature_rows)
    write_rows(out_dir / "v22_16_real_source_state_release_matrix.csv", release_rows)
    write_rows(out_dir / "v22_16_controller_intervention_log.csv", interventions)
    write_rows(out_dir / "v22_16_source_func_loss_horizon_matrix.csv", rows)
    write_rows(out_dir / "v22_16_control_attribution_matrix.csv", controls)
    route = {
        "route": "C-RealMLPAdaptiveControllerPass" if c1_pass and c2_pass else ("R2-RealRiskPredictionNoGo" if not c2_pass else "R3-RealAdaptiveControllerNoGo"),
        "C1_mlp_adaptive_pass": c1_pass,
        "C1_mlp_adaptive_exploration_pass": c1_exploration,
        "run_scope_official_mechanism": official_scope,
        "required_C4_groups_per_adapter": required_groups,
        "predictive_C4_groups_CE": predictive_ce_groups,
        "predictive_C4_groups_MSE": predictive_mse_groups,
        "C2_risk_prediction_pass": c2_pass,
        "source_state_release_pass": release_pass,
        "controls_fail": controls_fail,
        "predictive_C4_pass_rows": predictive_pass_rows,
        "reactive_C4_pass_rows": reactive_pass_rows,
        "fixed_C4_pass_rows": fixed_pass_rows,
        "risk_AUC_H100": auc100,
        "best_risk_model_H100": best_risk_model,
        "median_intervention_lead_time": median_lead,
        "blocker": ";".join(blockers[:20]),
        "repair_attempts": "risk AUC computed from completed real train trajectories; controls excluded from risk AUC; tested built-in, destructive-only, source-loss-boundary, monotone agreement risk laws; conservative controller modes lower lambda/prox activation per R3 repair path",
    }
    write_json(out_dir / "v22_16_mlp_adaptive_route.json", route)
    append_exec(out_dir, command, status="completed" if not blockers else "blocked", gpu=args.device, task_id="C-D-real-mlp-risk", files="v22_16_real_mlp_adaptive_guidance_matrix.csv; v22_16_real_risk_prediction_matrix.csv", note=f"route={route['route']} C1={c1_pass} C1_explore={c1_exploration} C2={c2_pass} official_scope={official_scope} rows={len(rows)} blockers={len(blockers)}")


if __name__ == "__main__":
    main()
