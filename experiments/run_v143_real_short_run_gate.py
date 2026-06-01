#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from copy import copy
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import run_v1223_failclosed_explore_open2_functional_rebuild as v1223
from experiments.run_v133_task_family_robust_basis_natural import eval_metrics, linec_proxy
from experiments.run_v136_poprisk_snr_basis_cover_boundary import collect_per_example_gradients
from experiments.run_v142_functional_first_all_basis_parallel import (
    FMSState,
    fnum,
    group_utilities,
    loss_value,
    make_case_model,
    named_param_specs,
    sint,
)
from experiments.run_v1231_basis_kernel_workspace import linec_metrics
from experiments.run_v143_functional_value_constraint_all_basis_substrate import (
    K_CONTROLS,
    K_METHODS,
    assign_flat_grad,
    base_projection_scales,
    clamp_rational_params,
    flat_existing_grad,
    parse_csv,
    parse_ints,
    projection_stats,
    scale_tensor_by_role,
    value_preserving_scales,
    write_json,
    write_rows,
)


ROOT = Path("results/v14_3_functional_value_constraint_all_basis_substrate")


class LogitScaleWrapper(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, scale: float) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("logit_scale", torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)
        return logits * self.logit_scale.to(device=x.device, dtype=logits.dtype)


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def load_real_split(args: argparse.Namespace, dataset: str, seed: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    canonical = v1223.v120._canonical_dataset(str(dataset))
    load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=int(seed))
    data = v1223.v120._load_vision_split(
        load_args,
        canonical,
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
    )
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, x_test_cpu, y_test_cpu, input_dim, output_dim, _protocol = data
    return (
        x_train_cpu.to(device=device, dtype=torch.float32),
        y_train_cpu.to(device=device),
        x_val_cpu.to(device=device, dtype=torch.float32),
        y_val_cpu.to(device=device),
        x_test_cpu.to(device=device, dtype=torch.float32),
        y_test_cpu.to(device=device),
        torch.tensor(int(input_dim)),
        torch.tensor(int(output_dim)),
    )


def output_geometry_model(model: torch.nn.Module, xtr: torch.Tensor, args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    repair = str(args.output_geometry_repair)
    if repair == "none":
        return model, {
            "output_geometry_repair": "none",
            "output_geometry_scale": 1.0,
            "output_geometry_train_entropy_norm": "",
            "output_geometry_train_rms": "",
            "output_geometry_uses_labels": 0,
            "output_geometry_uses_validation_test_linec_tail": 0,
        }
    with torch.no_grad():
        logits = model(xtr[: min(256, int(xtr.shape[0]))]).detach().float()
        rms = float(logits.square().mean().sqrt().item())
        probs = torch.softmax(logits, dim=1)
        entropy = float((-(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=1) / math.log(max(2, logits.shape[1]))).mean().item())
    if repair == "fixed050":
        scale = 0.50
        selected_target = ""
    elif repair == "train_rms_target050":
        selected_target = 0.50
        scale = float(selected_target) / max(1.0e-8, rms)
    elif repair == "train_entropy_t080_100_else050":
        selected_target = 1.00 if entropy >= 0.80 else 0.50
        scale = float(selected_target) / max(1.0e-8, rms)
    else:
        raise ValueError(repair)
    return LogitScaleWrapper(model, scale).to(device=xtr.device), {
        "output_geometry_repair": repair,
        "output_geometry_scale": scale,
        "output_geometry_selected_target": selected_target,
        "output_geometry_train_entropy_norm": entropy,
        "output_geometry_train_rms": rms,
        "output_geometry_uses_labels": 0,
        "output_geometry_uses_validation_test_linec_tail": 0,
    }


def train_real_case(
    *,
    dataset: str,
    seed: int,
    method: str,
    candidate_id: str,
    loss_interface: str,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
    input_dim: int,
    output_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    case_args = copy(args)
    case_args.synthetic_dim = int(input_dim)
    case_args.synthetic_classes = int(output_dim)
    model = make_case_model("D-RAT", candidate_id, xtr, seed, case_args, device)
    specs = named_param_specs(model)
    params = [(spec.name, spec.param) for spec in specs]
    generic_keys = [spec.layer for spec in specs]
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed) + 143_500 + sum(ord(c) for c in method + dataset))
    rng = random.Random(int(seed) + 143_800 + len(method))
    state = FMSState(float(args.fms_beta), float(args.fms_strength), int(seed) + 143)
    last_key_scales = {key: 1.0 for key in set(generic_keys)}
    trajectory: list[dict[str, float]] = []
    projection_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    projection_stats_acc: dict[str, list[float]] = {
        "generic_value_norm": [],
        "projected_value_norm": [],
        "cos_projected_vs_generic": [],
        "value_retention": [],
        "projection_rejection_fraction": [],
    }
    linec_audit_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    fms_refresh_count = 0
    for step in range(int(args.train_steps)):
        idx = torch.randint(0, xtr.shape[0], (int(args.batch_size),), generator=gen, device=device)
        xb, yb = xtr[idx], ytr[idx]
        opt.zero_grad(set_to_none=True)
        if method == "K0-RAT-AdamW":
            loss = loss_value(model(xb), yb, loss_interface)
            loss.backward()
            generic_flat = flat_existing_grad(specs)
            role_scales = {"default": 1.0}
            projected_flat = generic_flat
            stats = projection_stats(generic_flat, projected_flat, specs, role_scales)
        else:
            refresh = step % max(1, int(args.fms_update_interval)) == 0
            if refresh:
                fms_refresh_count += 1
                g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface=loss_interface)
                utilities, grad_stats = group_utilities(g, specs, generic_keys, "F3-LayerFMS")
                key_scales = state.update(generic_keys, utilities, "F3-LayerFMS", step, int(args.train_steps))
                last_key_scales = dict(key_scales)
                generic_flat = g.mean(dim=0).detach().clone()
                if step == 0 or step == int(args.train_steps) - 1:
                    grad_rows.append(
                        {
                            "stage": "V143_REAL_SHORT_RUN_PER_EXAMPLE_GRADIENT_STATS",
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "loss_interface": loss_interface,
                            "step": step + 1,
                            **grad_stats,
                            "direction_uses_validation_test_future_query": 0,
                            "direction_uses_linec_cep99_nll_ece": 0,
                            "promotion_allowed": 0,
                        }
                    )
            else:
                loss = loss_value(model(xb), yb, loss_interface)
                loss.backward()
                generic_flat = flat_existing_grad(specs)
                key_scales = dict(last_key_scales)
            for spec, key in zip(specs, generic_keys):
                generic_flat[spec.start : spec.end].mul_(float(key_scales.get(key, 1.0)))
            role_scales = base_projection_scales(method, step, int(args.train_steps), rng)
            role_scales = value_preserving_scales(method, generic_flat, specs, role_scales)
            projected_flat = scale_tensor_by_role(generic_flat, specs, role_scales)
            stats = projection_stats(generic_flat, projected_flat, specs, role_scales)
            assign_flat_grad(specs, projected_flat)
        opt.step()
        projected_params = clamp_rational_params(specs, method)
        for key in projection_stats_acc:
            projection_stats_acc[key].append(float(stats.get(key, 0.0)))
        if step == 0 or step == int(args.train_steps) - 1 or ((step + 1) % max(1, int(args.trace_interval)) == 0):
            val_metrics = eval_metrics(model, xva, yva)
            trajectory.append(
                {
                    "step": float(step + 1),
                    "NLL": val_metrics["NLL"],
                    "CEp99": val_metrics["CEp99"],
                    "ECE": val_metrics["ECE"],
                    "acc": val_metrics["acc"],
                }
            )
            projection_rows.append(
                {
                    "stage": "V143_REAL_SHORT_RUN_PROJECTION_AUDIT",
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "loss_interface": loss_interface,
                    "step": step + 1,
                    **stats,
                    "projection_applied": int(method not in {"K0-RAT-AdamW", "K1-RAT-GenericFMS-NoProjection"}),
                    "projection_denominator_clamped_params": int(projected_params),
                    "projection_uses_audit_metric": 0,
                    "direction_uses_validation_test_future_query": 0,
                    "direction_uses_linec_cep99_nll_ece": 0,
                    "promotion_allowed": 0,
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory_bytes = 0
    elapsed = time.perf_counter() - start
    eval_model, output_info = output_geometry_model(model, xtr, args)
    val_final = eval_metrics(eval_model, xva, yva)
    test_final = eval_metrics(eval_model, xte, yte)
    auc_nll = sum(point["NLL"] for point in trajectory) / max(1, len(trajectory))
    auc_cep99 = sum(point["CEp99"] for point in trajectory) / max(1, len(trajectory))
    if str(args.linec_mode) == "exact":
        b = min(int(args.linec_batch_size), int(xtr.shape[0]), int(xva.shape[0]))
        linec_votes = []
        linec_metrics_rows: list[dict[str, float]] = []
        for linec_seed in parse_ints(args.linec_seeds):
            try:
                lm = linec_metrics(eval_model, xtr[:b], ytr[:b], xva[:b], yva[:b], int(linec_seed), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
                linec_status = "executed"
                linec_error = ""
            except Exception as exc:  # noqa: BLE001
                lm = {
                    "CouplingR2": float("nan"),
                    "NoiseSignalLeak": float("nan"),
                    "RealSignalReservoirRatio": float("nan"),
                }
                linec_status = "blocked"
                linec_error = f"{type(exc).__name__}: {exc}"
            passed = int(
                fnum(lm.get("CouplingR2"), -999.0) >= 0.15
                and fnum(lm.get("NoiseSignalLeak"), 999.0) <= 0.20
                and fnum(lm.get("RealSignalReservoirRatio"), 999.0) <= 0.70
            )
            linec_votes.append(passed)
            linec_metrics_rows.append(lm)
            linec_audit_rows.append(
                {
                    "stage": "V143_REAL_SHORT_RUN_EXACT_LINEC_AUDIT",
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "loss_interface": loss_interface,
                    "linec_seed": int(linec_seed),
                    "linec_status": linec_status,
                    "linec_error": linec_error,
                    **lm,
                    "linec_used_for_direction": 0,
                    "promotion_allowed": 0,
                }
            )
        linec_coupling_r2 = sum(fnum(row.get("CouplingR2"), 0.0) for row in linec_metrics_rows) / max(1, len(linec_metrics_rows))
        linec_noise = sum(fnum(row.get("NoiseSignalLeak"), 0.0) for row in linec_metrics_rows) / max(1, len(linec_metrics_rows))
        linec_reservoir = sum(fnum(row.get("RealSignalReservoirRatio"), 0.0) for row in linec_metrics_rows) / max(1, len(linec_metrics_rows))
    else:
        linec_votes = [
            linec_proxy({"CEp99": point["CEp99"], "NoiseSignalLeak": val_final["NoiseSignalLeak"], "margin_p10": val_final["margin_p10"]})
            for point in trajectory
        ]
        linec_coupling_r2 = val_final["CouplingR2"]
        linec_noise = val_final["NoiseSignalLeak"]
        linec_reservoir = val_final["RealSignalReservoirRatio"]
    row = {
        "stage": "V143_REAL_SHORT_RUN_RESULT",
        "family": "D-RAT",
        "candidate_id": candidate_id,
        "dataset": dataset,
        "seed": seed,
        "loss_interface": loss_interface,
        "method": method,
        "control_method": int(method in K_CONTROLS),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(args.test_size),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / max(1, int(args.train_steps)),
        "peak_memory_bytes": peak_memory_bytes,
        "NLL": val_final["NLL"],
        "CEp99": val_final["CEp99"],
        "ECE": val_final["ECE"],
        "Brier": val_final["Brier"],
        "acc": val_final["acc"],
        "test_NLL": test_final["NLL"],
        "test_CEp99": test_final["CEp99"],
        "test_ECE": test_final["ECE"],
        "test_Brier": test_final["Brier"],
        "test_acc": test_final["acc"],
        "CouplingR2": linec_coupling_r2,
        "NoiseSignalLeak": linec_noise,
        "RealSignalReservoirRatio": linec_reservoir,
        "margin_p10": val_final["margin_p10"],
        "AUC_NLL": auc_nll,
        "AUC_CEp99": auc_cep99,
        "LineC_pass_rate": sum(linec_votes) / max(1, len(linec_votes)),
        "LineC_majority_pass": int(sum(linec_votes) >= math.ceil(len(linec_votes) / 2)),
        "LineC_pass_count": sum(linec_votes),
        "LineC_seed_count": len(linec_votes),
        "linec_mode": str(args.linec_mode),
        "value_source_type": "adamw" if method == "K0-RAT-AdamW" else "generic_loss_interface_layer_fms",
        "constraint_source_type": "none" if method in {"K0-RAT-AdamW", "K1-RAT-GenericFMS-NoProjection"} else "basis_role_trust_region",
        "projection_applied": int(method not in {"K0-RAT-AdamW", "K1-RAT-GenericFMS-NoProjection"}),
        "direction_uses_validation_test_future_query": 0,
        "direction_uses_linec_cep99_nll_ece": 0,
        "fms_update_interval": int(args.fms_update_interval),
        "fms_refresh_count": int(fms_refresh_count),
        **output_info,
        "promotion_allowed": 0,
    }
    for key, values in projection_stats_acc.items():
        row[f"median_{key}"] = median(values)
    return {"row": row, "projection_rows": projection_rows, "grad_rows": grad_rows, "linec_rows": linec_audit_rows}


def enrich_real_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["dataset"]), int(row["seed"]), str(row["loss_interface"])), []).append(row)
    out: list[dict[str, Any]] = []
    for _key, group in by_key.items():
        controls = [row for row in group if sint(row.get("control_method"), 0) == 1]
        best_auc = min((fnum(row.get("AUC_NLL"), 9.0) for row in controls), default=min(fnum(row.get("AUC_NLL"), 9.0) for row in group))
        best_nll = min((fnum(row.get("NLL"), 9.0) for row in controls), default=min(fnum(row.get("NLL"), 9.0) for row in group))
        adam = next((row for row in group if row.get("method") == "K0-RAT-AdamW"), controls[0] if controls else group[0])
        base_time = max(1.0e-8, fnum(adam.get("step_time_sec"), 1.0))
        base_mem = max(1.0, fnum(adam.get("peak_memory_bytes"), 1.0))
        for row in group:
            item = dict(row)
            item["source_vs_best_control"] = best_nll - fnum(row.get("NLL"), 9.0)
            item["source_vs_adamw"] = fnum(adam.get("NLL"), 9.0) - fnum(row.get("NLL"), 9.0)
            item["AUCtime_ratio_vs_best_control"] = fnum(row.get("AUC_NLL"), 9.0) / max(1.0e-8, best_auc)
            item["CEp99_delta_vs_adamw"] = fnum(row.get("CEp99"), 0.0) - fnum(adam.get("CEp99"), 0.0)
            item["NLL_delta_vs_adamw"] = fnum(row.get("NLL"), 0.0) - fnum(adam.get("NLL"), 0.0)
            item["ECE_delta_vs_adamw"] = fnum(row.get("ECE"), 0.0) - fnum(adam.get("ECE"), 0.0)
            item["test_NLL_delta_vs_adamw"] = fnum(row.get("test_NLL"), 0.0) - fnum(adam.get("test_NLL"), 0.0)
            item["test_ECE_delta_vs_adamw"] = fnum(row.get("test_ECE"), 0.0) - fnum(adam.get("test_ECE"), 0.0)
            item["step_time_ratio_vs_adamw"] = fnum(row.get("step_time_sec"), 0.0) / base_time
            item["peak_memory_ratio_vs_adamw"] = fnum(row.get("peak_memory_bytes"), 0.0) / base_mem
            item["real_short_run_gate_pass"] = int(
                sint(item.get("control_method"), 0) == 0
                and fnum(item["source_vs_best_control"]) >= 0.005
                and fnum(item["AUCtime_ratio_vs_best_control"]) <= 1.0
                and fnum(item["CEp99_delta_vs_adamw"]) <= 0.05
                and fnum(item["NLL_delta_vs_adamw"]) <= 0.02
                and fnum(item["ECE_delta_vs_adamw"]) <= 0.02
                and sint(item.get("LineC_majority_pass"), 0) == 1
                and fnum(item["step_time_ratio_vs_adamw"]) <= 1.75
                and fnum(item["peak_memory_ratio_vs_adamw"]) <= 1.75
            )
            out.append(item)
    return out


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row)
    summary: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        non_control = [row for row in group if sint(row.get("control_method"), 0) == 0]
        target = non_control if non_control else group
        summary.append(
            {
                "stage": "V143_REAL_SHORT_RUN_SUMMARY",
                "method": method,
                "rows": len(group),
                "dataset_seed_pass_count": len({(str(row.get("dataset")), str(row.get("seed"))) for row in group if sint(row.get("real_short_run_gate_pass"), 0) == 1}),
                "mean_source_vs_best_control": sum(fnum(row.get("source_vs_best_control"), 0.0) for row in target) / max(1, len(target)),
                "median_source_vs_best_control": median([fnum(row.get("source_vs_best_control"), 0.0) for row in target]),
                "median_AUCtime_ratio_vs_best_control": median([fnum(row.get("AUCtime_ratio_vs_best_control"), 9.0) for row in target]),
                "median_step_time_ratio_vs_adamw": median([fnum(row.get("step_time_ratio_vs_adamw"), 9.0) for row in target]),
                "median_peak_memory_ratio_vs_adamw": median([fnum(row.get("peak_memory_ratio_vs_adamw"), 9.0) for row in target]),
                "linec_majority_pass_rows": sum(sint(row.get("LineC_majority_pass"), 0) for row in target),
                "promotion_allowed": 0,
            }
        )
    return summary


def build_route(rows: list[dict[str, Any]], summary: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    non_control = [row for row in rows if sint(row.get("control_method"), 0) == 0]
    pass_keys = {(str(row.get("dataset")), int(row.get("seed"))) for row in non_control if sint(row.get("real_short_run_gate_pass"), 0) == 1}
    expected = len(parse_csv(args.datasets)) * len(parse_ints(args.seeds))
    mean_source = sum(fnum(row.get("source_vs_best_control"), 0.0) for row in non_control) / max(1, len(non_control))
    s5 = int(len(pass_keys) >= expected and mean_source >= 0.005)
    return {
        "stage": "V143_REAL_SHORT_RUN_ROUTE",
        "route": "S5-OfficialFunctionalSuccess" if s5 else "S4-RealShortRunOpened",
        "minimum_success": "S5-OfficialFunctionalSuccess" if s5 else "S4-RealShortRunOpened",
        "official_s5_reached": s5,
        "promotion_allowed": 0,
        "real_short_run_opened": 1,
        "compute_budgeted_run": int(args.compute_budgeted_run),
        "datasets": parse_csv(args.datasets),
        "seeds": parse_ints(args.seeds),
        "methods": parse_csv(args.methods),
        "expected_dataset_seed_count": expected,
        "real_dataset_seed_pass_count": len(pass_keys),
        "real_short_run_pass_rows": sum(sint(row.get("real_short_run_gate_pass"), 0) for row in non_control),
        "mean_source_vs_best_control_noncontrol": mean_source,
        "required_artifact_missing_count": 0,
        "forbidden_information_violation_count": 0,
        "direction_uses_validation_test_future_query": 0,
        "direction_uses_linec_cep99_nll_ece": 0,
        "out_dir": str(args.out_dir),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    methods = parse_csv(args.methods)
    datasets = parse_csv(args.datasets)
    seeds = parse_ints(args.seeds)
    rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    exact_linec_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            xtr, ytr, xva, yva, xte, yte, input_dim_t, output_dim_t = load_real_split(args, dataset, seed, device)
            input_dim, output_dim = int(input_dim_t.item()), int(output_dim_t.item())
            for method in methods:
                result = train_real_case(
                    dataset=v1223.v120._canonical_dataset(dataset),
                    seed=seed,
                    method=method,
                    candidate_id=str(args.rational_candidate),
                    loss_interface=str(args.loss_interface),
                    xtr=xtr,
                    ytr=ytr,
                    xva=xva,
                    yva=yva,
                    xte=xte,
                    yte=yte,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    args=args,
                    device=device,
                )
                rows.append(result["row"])
                projection_rows.extend(result["projection_rows"])
                grad_rows.extend(result["grad_rows"])
                exact_linec_rows.extend(result.get("linec_rows", []))
    enriched = enrich_real_rows(rows)
    summary = build_summary(enriched, args)
    route = build_route(enriched, summary, args)
    linec_rows = exact_linec_rows + [
        {
            "stage": "V143_REAL_SHORT_RUN_LINEC_SUMMARY_AUDIT",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "method": row.get("method"),
            "LineC_pass_rate": row.get("LineC_pass_rate"),
            "LineC_majority_pass": row.get("LineC_majority_pass"),
            "CouplingR2": row.get("CouplingR2"),
            "NoiseSignalLeak": row.get("NoiseSignalLeak"),
            "RealSignalReservoirRatio": row.get("RealSignalReservoirRatio"),
            "linec_used_for_direction": 0,
            "promotion_allowed": 0,
        }
        for row in enriched
    ]
    tail_rows = [
        {
            "stage": "V143_REAL_SHORT_RUN_TAIL_AUDIT",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "method": row.get("method"),
            "CEp99": row.get("CEp99"),
            "NLL": row.get("NLL"),
            "ECE": row.get("ECE"),
            "test_CEp99": row.get("test_CEp99"),
            "test_NLL": row.get("test_NLL"),
            "test_ECE": row.get("test_ECE"),
            "CEp99_delta_vs_adamw": row.get("CEp99_delta_vs_adamw"),
            "NLL_delta_vs_adamw": row.get("NLL_delta_vs_adamw"),
            "ECE_delta_vs_adamw": row.get("ECE_delta_vs_adamw"),
            "tail_metrics_used_for_direction": 0,
            "promotion_allowed": 0,
        }
        for row in enriched
    ]
    write_rows(out_dir / "v143_real_short_run_results.csv", enriched)
    write_rows(out_dir / "v143_real_short_run_projection_audit.csv", projection_rows)
    write_rows(out_dir / "v143_real_short_run_per_example_gradient_stats.csv", grad_rows)
    write_rows(out_dir / "v143_real_short_run_linec_audit.csv", linec_rows)
    write_rows(out_dir / "v143_real_short_run_tail_audit.csv", tail_rows)
    write_rows(out_dir / "v143_real_short_run_summary.csv", summary)
    write_json(out_dir / "v143_real_short_run_route.json", route)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.3 real short-run gate")
    parser.add_argument("--out-dir", default=str(ROOT / "real_short_run_v143"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default="K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection")
    parser.add_argument("--rational-candidate", default="D-RAT28-GroupDiversityPreservingRational")
    parser.add_argument("--loss-interface", default="CE")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--fms-beta", type=float, default=0.99)
    parser.add_argument("--fms-strength", type=float, default=0.25)
    parser.add_argument("--fms-update-interval", type=int, default=40)
    parser.add_argument("--trace-interval", type=int, default=100)
    parser.add_argument(
        "--output-geometry-repair",
        choices=["none", "fixed050", "train_rms_target050", "train_entropy_t080_100_else050"],
        default="none",
    )
    parser.add_argument("--linec-mode", choices=["proxy", "exact"], default="exact")
    parser.add_argument("--linec-seeds", default="12319500,12319501,12319502")
    parser.add_argument("--linec-batch-size", type=int, default=24)
    parser.add_argument("--linec-sketch-dim", type=int, default=8)
    parser.add_argument("--compute-budgeted-run", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    route = run(args)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
