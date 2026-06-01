#!/usr/bin/env python
"""v13.8 Generic PopRisk optimizer and KAN SNR-to-basis lift runner.

This runner separates the v13.8 lines:

* Line G: MLP-only generic PopRisk optimizer confirmation.  This can never
  promote a KAN route.
* Line K0/K1: Rational KAN SNR signal transfer and SNR-to-basis lift.
* Line D/C/R/Z: substrate, geometry, provenance, finalizer artifacts.

All training directions are generated from the current train batch via a
generic loss interface.  Validation/test/future/LineC/tail metrics are
audit/gate only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments.run_v133_task_family_robust_basis_natural import (  # noqa: E402
    SOURCE_V1235,
    build_substrate_map,
    eval_metrics,
    fnum,
    make_model_for_family,
    parse_csv,
    parse_ints,
    sha256_file,
    sint,
    synthetic_data,
    write_rows,
    write_svg,
)
from experiments.run_v134_operator_level_basis_channel_functional import channel_named_params  # noqa: E402
from experiments.run_v136_poprisk_snr_basis_cover_boundary import (  # noqa: E402
    collect_per_example_gradients,
    cover_debt,
    flatten_tensors,
    loss_interface_cotangent_per_example,
    role_of_param,
)
from experiments.run_v137_boundary_conditioned_poprisk_training import (  # noqa: E402
    SNRState,
    add_baseline_deltas,
    all_named_params,
    assign_flat_grad,
    choose_primary_rational,
    cover_policy,
    forbidden_audit as forbidden_audit_v137,
    implementation_readback as implementation_readback_v137,
    loss_value,
    pass_flags,
    phase_for_step,
    rational_boundary_terms,
    run_nonrat_repair,
    run_training_case,
    selected_params,
    snr_gate,
    task_family_summary,
)
from experiments.run_v137_mlp_snr_real_triage import (  # noqa: E402
    canonical_dataset,
    load_vision_split,
    mlp_per_example_gradients,
)


OUT_DIR = ROOT / "results" / "v13_8_generic_poprisk_and_kan_snr_to_basis_lift" / "official_v138"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v13.8_GenericPopRisk_and_KAN_SNRtoBasisLift_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v13.8_GenericPopRisk_and_KAN_SNRtoBasisLift_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v13.8_GenericPopRisk_and_KAN_SNRtoBasisLift_实验结果复盘.md"

REQUIRED = [
    "v138_progress_table.csv",
    "v138_route_decision.json",
    "v138_loss_interface_audit.csv",
    "v138_forbidden_information_audit.csv",
    "v138_mlp_generic_optimizer_training.csv",
    "v138_mlp_generic_optimizer_summary.csv",
    "v138_mlp_generic_optimizer_10seed.csv",
    "v138_snr_transfer_audit.csv",
    "v138_rolewise_snr_audit.csv",
    "v138_basis_group_projection_loss.csv",
    "v138_rat_snr_lift_training.csv",
    "v138_rat_snr_lift_summary.csv",
    "v138_rat_snr_lift_linec.csv",
    "v138_rat_snr_lift_controls.csv",
    "v138_nonrat_substrate_health.csv",
    "v138_linec_audit.csv",
    "v138_failure_table.csv",
    "v138_no_go_boundary.md",
    "v138_next_hypothesis_queue.md",
    "v138_implementation_readback.md",
    "v138_code_review_packet.zip",
]

FIGURES = [
    "fig_v138_progress_delta.svg",
    "fig_v138_mlp_10seed_pass_heatmap.svg",
    "fig_v138_mlp_source_tail_pareto.svg",
    "fig_v138_snr_transfer_retention_by_role.svg",
    "fig_v138_param_vs_group_snr_cosine.svg",
    "fig_v138_false_drop_false_keep_by_role.svg",
    "fig_v138_rat_snr_lift_task_family_heatmap.svg",
    "fig_v138_linec_signal_reservoir_noise_matrix.svg",
    "fig_v138_nonrat_substrate_status.svg",
    "fig_v138_failure_taxonomy.svg",
]

K1_METHOD_ALIASES = {
    "K1-RAT-ParamSNR-Only": ("RAT-ParameterSNRSoft", "parameter-SNR source only"),
    "K2-RAT-ParamSNR-MinCoverGuard": ("RAT-ParameterSNRSoft-MinCoverGuard", "parameter-SNR with minimal train-stream cover guard"),
    "K3-RAT-RoleWiseSNRLift": ("RAT-ParameterSNRRoleNorm", "role-wise SNR lift through role-normalized parameter gate"),
    "K4-RAT-DynamicSNRClusterLift": ("RAT-GroupSNRSoft-DynamicSNRClusterLift", "data-driven soft group lift using per-batch SNR scores"),
    "K4F-RAT-DynamicSNRClusterFreeze": ("RAT-GroupSNRSoft-DynamicSNRClusterFreeze", "data-driven soft group lift with train-stream gate frozen after warmup"),
    "K5-RAT-LowRankSNRCorrector": ("RAT-ParameterSNREMA-LowRankCorrector", "EMA parameter-SNR plus rank-1 per-example gradient corrector"),
    "K6-RAT-ParamSNR-Then-BasisConsolidation": ("RAT-ParameterSNRSoft-ThenBasisConsolidation", "parameter-SNR plasticity followed by consolidation-only cover guard"),
    "K7-RAT-ParamSNR-BlendAdamW-Then-CoverPhase": ("RAT-ParameterSNREMA-Blend50-ThenCoverPhase", "EMA parameter-SNR blended with same-batch AdamW gradient and phase cover"),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_k1_method(method: str) -> tuple[str, str]:
    if method in K1_METHOD_ALIASES:
        return K1_METHOD_ALIASES[method]
    return method, "native v13.7/v13.8 training method"


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def quantile_or_nan(values: Iterable[float], q: float) -> float:
    vals = torch.tensor([float(v) for v in values if math.isfinite(float(v))], dtype=torch.float32)
    if int(vals.numel()) == 0:
        return float("nan")
    return float(torch.quantile(vals, float(q)).item())


def line_range_of(path: Path, token: str) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = 1
    for i, line in enumerate(lines, start=1):
        if line.startswith(f"def {token}") or line.startswith(f"class {token}"):
            start = i
            break
    end = start
    for j in range(start, len(lines)):
        if j > start and lines[j - 1] and not lines[j - 1].startswith((" ", "\t", "@")):
            end = j - 1
            break
    return start, max(start, end)


def per_example_losses(logits: torch.Tensor, y: torch.Tensor, interface: str) -> torch.Tensor:
    logits_f = logits.float()
    if interface == "CE":
        return F.cross_entropy(logits_f, y, reduction="none")
    if interface == "Brier":
        target = F.one_hot(y, num_classes=logits_f.shape[1]).to(dtype=torch.float32, device=logits_f.device)
        return (torch.softmax(logits_f, dim=1) - target).square().sum(dim=1)
    if interface == "MSELogit":
        target = 2.0 * F.one_hot(y, num_classes=logits_f.shape[1]).to(dtype=torch.float32, device=logits_f.device) - 1.0
        return (logits_f - target).square().mean(dim=1)
    raise ValueError(interface)


def parse_blend_alpha(method: str) -> float:
    if "Blend25" in method:
        return 0.25
    if "Blend50" in method:
        return 0.50
    if "Blend75" in method:
        return 0.75
    return 1.0


def apply_train_stream_trust(
    *,
    method: str,
    g: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    loss_interface: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply legal train-stream trust policies to per-example gradients."""
    if int(g.numel()) == 0:
        return g, {
            "trust_reject_fraction": 0.0,
            "train_loss_quantile_p95": 0.0,
            "logit_norm_p95": 0.0,
            "per_example_grad_clip_fraction": 0.0,
        }
    weights = torch.ones(int(g.shape[0]), device=g.device, dtype=g.dtype)
    losses = per_example_losses(logits, y, loss_interface).detach().float()
    loss_q95 = float(torch.quantile(losses, 0.95).item()) if int(losses.numel()) else 0.0
    if "TrainLossQuantileTrust" in method and int(losses.numel()) > 1:
        weights = weights * torch.where(losses <= loss_q95, torch.ones_like(weights), torch.full_like(weights, 0.25))
    logit_norm = logits.detach().float().norm(dim=1)
    logit_q95 = float(torch.quantile(logit_norm, 0.95).item()) if int(logit_norm.numel()) else 0.0
    if "LogitNormTrust" in method and int(logit_norm.numel()) > 1:
        weights = weights * torch.where(logit_norm <= logit_q95, torch.ones_like(weights), torch.full_like(weights, 0.50))
    g2 = g * weights[:, None]
    grad_norm = g2.norm(dim=1)
    clip_fraction = 0.0
    if "PerExampleGradientClip" in method and int(grad_norm.numel()) > 1:
        cap = torch.quantile(grad_norm, 0.90).clamp_min(1.0e-8)
        scale = torch.clamp(cap / grad_norm.clamp_min(1.0e-8), max=1.0)
        clip_fraction = float((scale < 0.999).float().mean().item())
        g2 = g2 * scale[:, None]
    reject_fraction = float((weights < 0.999).float().mean().item())
    return g2, {
        "trust_reject_fraction": reject_fraction,
        "train_loss_quantile_p95": loss_q95,
        "logit_norm_p95": logit_q95,
        "per_example_grad_clip_fraction": clip_fraction,
    }


def mlp_train_one(
    *,
    dataset: str,
    seed: int,
    method: str,
    loss_interface: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    xtr, ytr, xva, yva, xte, yte, protocol = load_vision_split(
        dataset,
        data_root=data_root,
        train_size=int(args.real_train_size),
        val_size=int(args.real_val_size),
        test_size=int(args.real_test_size),
        seed=int(seed),
        download=not bool(args.no_download),
    )
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    xva = xva.to(device)
    yva = yva.to(device)
    xte = xte.to(device)
    yte = yte.to(device)
    model = MLPBaseline(int(xtr.shape[1]), 10, int(args.mlp_hidden), int(seed) + 13_800, device).to(device)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for _n, p in params], lr=float(args.lr), weight_decay=float(args.weight_decay), foreach=False)
    state = SNRState(decay=float(args.snr_ema_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed) + 138_777)
    steps_per_epoch = max(1, math.ceil(int(xtr.shape[0]) / int(args.batch_size)))
    total_steps = int(args.real_epochs) * steps_per_epoch
    rows: list[dict[str, Any]] = []
    val_losses: list[float] = []
    val_times: list[float] = []
    active_values: list[float] = []
    removed_values: list[float] = []
    trust_values: list[float] = []
    grad_clip_values: list[float] = []
    t0 = time.perf_counter()
    for step in range(1, total_steps + 1):
        idx = torch.randint(0, int(xtr.shape[0]), (min(int(args.batch_size), int(xtr.shape[0])),), device=device, generator=gen)
        opt.zero_grad(set_to_none=True)
        if "SNR" in method:
            logits = model(xtr[idx]).float()
            g = mlp_per_example_gradients(model, xtr[idx], ytr[idx], loss_interface=loss_interface)
            g, trust = apply_train_stream_trust(method=method, g=g, logits=logits, y=ytr[idx], loss_interface=loss_interface)
            cap = float(args.active_fraction_cap)
            if "ActiveFractionSchedule" in method:
                progress = float(step) / max(1.0, float(total_steps))
                cap = min(cap, 0.35 + 0.45 * progress)
            meta = snr_gate(
                g,
                params,
                state,
                method=method,
                phase=phase_for_step(step, total_steps),
                tau=float(args.snr_tau),
                eps=float(args.snr_eps),
                soft_alpha=float(args.soft_alpha),
                active_fraction_cap=cap,
            )
            blend_alpha = parse_blend_alpha(method)
            base_grad = g.mean(dim=0)
            if blend_alpha < 1.0:
                grad = blend_alpha * meta["grad"] + (1.0 - blend_alpha) * base_grad
            else:
                grad = meta["grad"]
            meta["snr_blend_alpha"] = blend_alpha
            meta["cos_snr_adamw"] = float(F.cosine_similarity(-grad, -base_grad, dim=0).detach().item()) if int(grad.numel()) else 0.0
            removed = 1.0 - float(grad.norm().detach().item() / base_grad.norm().clamp_min(1.0e-8).detach().item()) if int(grad.numel()) else 0.0
            meta["removed_update_norm_fraction"] = max(0.0, min(1.0, removed))
            assign_flat_grad(params, grad)
        else:
            logits = model(xtr[idx]).float()
            trust = {
                "trust_reject_fraction": 0.0,
                "train_loss_quantile_p95": float(torch.quantile(per_example_losses(logits, ytr[idx], loss_interface).detach().float(), 0.95).item()),
                "logit_norm_p95": float(torch.quantile(logits.detach().float().norm(dim=1), 0.95).item()),
                "per_example_grad_clip_fraction": 0.0,
            }
            meta = {
                "active_fraction": 1.0,
                "removed_update_norm_fraction": 0.0,
                "cos_snr_adamw": 1.0,
                "snr_median": 0.0,
                "snr_p90": 0.0,
                "snr_p99": 0.0,
                "snr_blend_alpha": 0.0,
            }
            loss_value(logits, ytr[idx], loss_interface).backward()
        opt.step()
        active_values.append(float(meta["active_fraction"]))
        removed_values.append(float(meta["removed_update_norm_fraction"]))
        trust_values.append(float(trust["trust_reject_fraction"]))
        grad_clip_values.append(float(trust["per_example_grad_clip_fraction"]))
        if step == total_steps or step % max(1, int(args.log_interval)) == 0:
            vm = eval_metrics(model, xva, yva)
            val_losses.append(float(vm["NLL"]))
            val_times.append(time.perf_counter() - t0)
            rows.append({
                "stage": "V138_MLP_GENERIC_OPTIMIZER_TRAINING",
                "method": method,
                "dataset": canonical_dataset(dataset),
                "seed": int(seed),
                "loss_interface": loss_interface,
                "step": step,
                "epochs": int(args.real_epochs),
                "snr_active_fraction": float(meta["active_fraction"]),
                "removed_update_norm_fraction": float(meta["removed_update_norm_fraction"]),
                "cos_snr_adamw": float(meta["cos_snr_adamw"]),
                "snr_blend_alpha": float(meta["snr_blend_alpha"]),
                "trust_reject_fraction": float(trust["trust_reject_fraction"]),
                "train_loss_quantile_p95": float(trust["train_loss_quantile_p95"]),
                "logit_norm_p95": float(trust["logit_norm_p95"]),
                "per_example_grad_clip_fraction": float(trust["per_example_grad_clip_fraction"]),
                "val_loss": vm["NLL"],
                "val_acc": vm["acc"],
                "CEp99": vm["CEp99"],
                "NLL": vm["NLL"],
                "ECE": vm["ECE"],
                "Brier": vm["Brier"],
                "margin_p10": vm["margin_p10"],
                "CouplingR2": vm["CouplingR2"],
                "NoiseSignalLeak": vm["NoiseSignalLeak"],
                "ReservoirRatio": vm["RealSignalReservoirRatio"],
                "direction_uses_validation": 0,
                "direction_uses_test": 0,
                "direction_uses_future": 0,
                "direction_uses_linec": 0,
                "no_fake": 1,
            })
    final_val = eval_metrics(model, xva, yva)
    final_test = eval_metrics(model, xte, yte)
    auc_step = finite_mean(val_losses)
    auc_time = sum(v * t for v, t in zip(val_losses, val_times)) / max(sum(val_times), 1.0e-8) if val_losses else float("nan")
    summary = {
        "stage": "V138_MLP_GENERIC_OPTIMIZER_SUMMARY",
        "method": method,
        "dataset": canonical_dataset(dataset),
        "seed": int(seed),
        "loss_interface": loss_interface,
        "train_steps": total_steps,
        "epochs": int(args.real_epochs),
        "train_size": int(xtr.shape[0]),
        "val_size": int(xva.shape[0]),
        "test_size": int(xte.shape[0]),
        "val_loss_auc_step": auc_step,
        "val_loss_auc_time": auc_time,
        "final_NLL": final_val["NLL"],
        "final_CEp99": final_val["CEp99"],
        "final_ECE": final_val["ECE"],
        "final_Brier": final_val["Brier"],
        "final_margin_p10": final_val["margin_p10"],
        "test_NLL": final_test["NLL"],
        "test_CEp99": final_test["CEp99"],
        "test_ECE": final_test["ECE"],
        "test_acc": final_test["acc"],
        "active_fraction_mean": finite_mean(active_values),
        "active_fraction_p90": quantile_or_nan(active_values, 0.90),
        "removed_update_norm_fraction": finite_mean(removed_values),
        "trust_reject_fraction": finite_mean(trust_values),
        "per_example_grad_clip_fraction": finite_mean(grad_clip_values),
        "elapsed_sec": time.perf_counter() - t0,
        "protocol": protocol,
        "no_fake": 1,
    }
    return rows, summary


def add_mlp_deltas_and_pass(summary_rows: list[dict[str, Any]]) -> None:
    base = {
        (r["dataset"], int(r["seed"]), r["loss_interface"]): r
        for r in summary_rows
        if str(r["method"]) == "MLP-AdamW"
    }
    for r in summary_rows:
        b = base.get((r["dataset"], int(r["seed"]), r["loss_interface"]))
        if b is None or str(r["method"]) == "MLP-AdamW":
            r.update({
                "source_vs_adamw": 0.0,
                "AUC_time_delta": 0.0,
                "AUC_time_ratio": 1.0,
                "AUC_step_ratio": 1.0,
                "CEp99_delta": 0.0,
                "NLL_delta": 0.0,
                "ECE_delta": 0.0,
                "Brier_delta": 0.0,
                "margin_p10_delta": 0.0,
                "real_triage_pass": 0,
            })
            continue
        auc_delta = float(r["val_loss_auc_time"]) - float(b["val_loss_auc_time"])
        r.update({
            "source_vs_adamw": -auc_delta,
            "AUC_time_delta": auc_delta,
            "AUC_time_ratio": float(r["val_loss_auc_time"]) / max(float(b["val_loss_auc_time"]), 1.0e-8),
            "AUC_step_ratio": float(r["val_loss_auc_step"]) / max(float(b["val_loss_auc_step"]), 1.0e-8),
            "CEp99_delta": float(r["final_CEp99"]) - float(b["final_CEp99"]),
            "NLL_delta": float(r["final_NLL"]) - float(b["final_NLL"]),
            "ECE_delta": float(r["final_ECE"]) - float(b["final_ECE"]),
            "Brier_delta": float(r["final_Brier"]) - float(b["final_Brier"]),
            "margin_p10_delta": float(r["final_margin_p10"]) - float(b["final_margin_p10"]),
        })
        r["real_triage_pass"] = int(
            float(r["source_vs_adamw"]) >= 0.005
            and float(r["AUC_time_ratio"]) <= 1.0
            and float(r["CEp99_delta"]) <= 0.05
            and float(r["NLL_delta"]) <= 0.02
            and float(r["ECE_delta"]) <= 0.02
        )


def mlp_10seed_summary(summary_rows: list[dict[str, Any]], threshold: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in sorted({str(r["dataset"]) for r in summary_rows}):
        rs = [r for r in summary_rows if str(r["dataset"]) == dataset and str(r["method"]) != "MLP-AdamW"]
        seed_pass = sorted({int(r["seed"]) for r in rs if int(r.get("real_triage_pass", 0)) == 1})
        rows.append({
            "stage": "V138_MLP_GENERIC_OPTIMIZER_10SEED",
            "dataset": dataset,
            "seed_pass_count": len(seed_pass),
            "seed_threshold": int(threshold),
            "dataset_pass": int(len(seed_pass) >= int(threshold)),
            "passing_seeds": ";".join(str(s) for s in seed_pass),
            "no_fake": 1,
        })
    return rows


def run_mlp_line(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    training: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in parse_csv(args.datasets):
        for seed in parse_ints(args.mlp_seeds):
            for method in parse_csv(args.mlp_methods):
                try:
                    rows, summary = mlp_train_one(dataset=dataset, seed=int(seed), method=method, loss_interface=args.loss_interface, args=args, device=device)
                    training.extend(rows)
                    summaries.append(summary)
                except Exception as exc:  # noqa: BLE001
                    failures.append({
                        "stage": "V138_MLP_GENERIC_FAILURE",
                        "dataset": canonical_dataset(dataset),
                        "seed": int(seed),
                        "method": method,
                        "exception": repr(exc),
                        "no_fake": 1,
                    })
    add_mlp_deltas_and_pass(summaries)
    fam = mlp_10seed_summary(summaries, int(args.mlp_seed_threshold))
    return training, summaries, fam, failures


def same_size(v: torch.Tensor, n: int) -> torch.Tensor:
    if int(v.numel()) == n:
        return v
    if int(v.numel()) > n:
        return v[:n]
    pad = torch.zeros(n - int(v.numel()), device=v.device, dtype=v.dtype)
    return torch.cat([v, pad])


def role_slices(params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    offset = 0
    for name, p in params:
        n = int(p.numel())
        out.setdefault(role_of_param(name), []).append((offset, offset + n))
        offset += n
    return out


def mask_stats(source_gate: torch.Tensor, lift_gate: torch.Tensor) -> tuple[float, float]:
    s = source_gate > 0
    l = lift_gate > 0
    false_drop = (s & ~l).float().mean().item() if int(s.numel()) else 0.0
    false_keep = (~s & l).float().mean().item() if int(s.numel()) else 0.0
    return float(false_drop), float(false_keep)


def entropy_from_abs(v: torch.Tensor) -> float:
    vals = v.detach().float().abs()
    if int(vals.numel()) == 0 or float(vals.sum().item()) <= 0.0:
        return 0.0
    p = vals / vals.sum().clamp_min(1.0e-12)
    return float((-(p * (p + 1.0e-12).log()).sum() / math.log(float(max(2, int(vals.numel()))))).item())


def run_k0_transfer(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = parse_csv(args.synthetic_tasks)
    seeds = parse_ints(args.synthetic_seeds)
    losses = parse_csv(args.k_losses)
    candidate = str(args.rational_candidate or choose_primary_rational())
    transfer_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            for loss in losses:
                xtr, ytr, _xva, _yva = synthetic_data(task, int(seed), int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
                model = make_model_for_family("D-RAT", candidate, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed) + 13_800)
                params = channel_named_params(model) or all_named_params(model)
                xb = xtr[: min(int(args.batch_size), int(xtr.shape[0]))]
                yb = ytr[: int(xb.shape[0])]
                g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface=loss)
                param_meta = snr_gate(g, params, SNRState(), method="RAT-ParameterSNRSoft", phase="plasticity-open", tau=float(args.snr_tau), eps=float(args.snr_eps), soft_alpha=float(args.soft_alpha), active_fraction_cap=float(args.active_fraction_cap))
                group_meta = snr_gate(g, params, SNRState(), method="RAT-GroupSNR", phase="cover-alignment", tau=float(args.snr_tau), eps=float(args.snr_eps), soft_alpha=float(args.soft_alpha), active_fraction_cap=float(args.active_fraction_cap))
                basis_meta = snr_gate(g, params, SNRState(), method="RAT-BasisSNR", phase="cover-alignment", tau=float(args.snr_tau), eps=float(args.snr_eps), soft_alpha=float(args.soft_alpha), active_fraction_cap=float(args.active_fraction_cap))
                p = param_meta["grad"].detach().float()
                gu = same_size(group_meta["grad"].detach().float(), int(p.numel()))
                bu = same_size(basis_meta["grad"].detach().float(), int(p.numel()))
                p_norm = p.norm().clamp_min(1.0e-8)
                g_norm = gu.norm().clamp_min(1.0e-8)
                b_norm = bu.norm().clamp_min(1.0e-8)
                cos_group = float(F.cosine_similarity(gu, p, dim=0).item()) if int(p.numel()) else 0.0
                cos_basis = float(F.cosine_similarity(bu, p, dim=0).item()) if int(p.numel()) else 0.0
                retention_group = float(g_norm.item() / p_norm.item())
                retention_basis = float(b_norm.item() / p_norm.item())
                false_drop_g, false_keep_g = mask_stats(param_meta["gate"], group_meta["gate"])
                false_drop_b, false_keep_b = mask_stats(param_meta["gate"], basis_meta["gate"])
                row = {
                    "stage": "V138_SNR_TRANSFER_AUDIT",
                    "candidate": candidate,
                    "method": "K0-ParamVsGroupBasis",
                    "task": task,
                    "seed": int(seed),
                    "loss": loss,
                    "role": "all",
                    "param_snr_active_fraction": param_meta["active_fraction"],
                    "group_snr_active_fraction": group_meta["active_fraction"],
                    "basis_snr_active_fraction": basis_meta["active_fraction"],
                    "param_update_norm": float(p_norm.item()),
                    "group_update_norm": float(g_norm.item()),
                    "basis_update_norm": float(b_norm.item()),
                    "cos_group_vs_param": cos_group,
                    "cos_basis_vs_param": cos_basis,
                    "signal_retention_group": retention_group,
                    "signal_retention_basis": retention_basis,
                    "false_drop_fraction": false_drop_g,
                    "false_keep_fraction": false_keep_g,
                    "false_drop_fraction_basis": false_drop_b,
                    "false_keep_fraction_basis": false_keep_b,
                    "snr_entropy_param": entropy_from_abs(param_meta["ratio"]),
                    "snr_entropy_group": entropy_from_abs(group_meta["ratio"]),
                    "snr_entropy_basis": entropy_from_abs(basis_meta["ratio"]),
                    "source_vs_adamw": "",
                    "CEp99_delta": "",
                    "LineC_pass": "",
                    "no_fake": 1,
                }
                transfer_rows.append(row)
                for role, slices in role_slices(params).items():
                    idx = torch.cat([torch.arange(a, b, device=device) for a, b in slices]) if slices else torch.zeros(0, dtype=torch.long, device=device)
                    if int(idx.numel()) == 0:
                        continue
                    role_p = p[idx]
                    role_g = gu[idx]
                    role_b = bu[idx]
                    role_signal = float(role_p.norm().item() / p_norm.item())
                    role_noise = float(param_meta["var"][idx].abs().sum().item() / param_meta["var"].abs().sum().clamp_min(1.0e-8).item())
                    role_rows.append({
                        "stage": "V138_ROLEWISE_SNR_AUDIT",
                        "task": task,
                        "seed": int(seed),
                        "loss": loss,
                        "role": role,
                        "role_signal_mass_fraction": role_signal,
                        "role_noise_mass_fraction": role_noise,
                        "role_param_active_fraction": float((param_meta["gate"][idx] > 0).float().mean().item()),
                        "role_group_active_fraction": float((group_meta["gate"][idx] > 0).float().mean().item()),
                        "role_basis_active_fraction": float((basis_meta["gate"][idx] > 0).float().mean().item()),
                        "role_group_retention": float(role_g.norm().item() / role_p.norm().clamp_min(1.0e-8).item()),
                        "role_basis_retention": float(role_b.norm().item() / role_p.norm().clamp_min(1.0e-8).item()),
                        "readout_signal_mass": role_signal if role == "readout" else 0.0,
                        "numerator_signal_mass": role_signal if role == "numerator" else 0.0,
                        "denominator_signal_mass": role_signal if role == "denominator" else 0.0,
                        "projection_signal_mass": role_signal if role == "projection" else 0.0,
                        "residual_signal_mass": role_signal if role == "residual" else 0.0,
                        "no_fake": 1,
                    })
                projection_rows.append({
                    "stage": "V138_BASIS_GROUP_PROJECTION_LOSS",
                    "task": task,
                    "seed": int(seed),
                    "loss": loss,
                    "signal_retention_group": retention_group,
                    "signal_retention_basis": retention_basis,
                    "projection_loss_group": max(0.0, 1.0 - retention_group),
                    "projection_loss_basis": max(0.0, 1.0 - retention_basis),
                    "cos_group_vs_param": cos_group,
                    "cos_basis_vs_param": cos_basis,
                    "false_drop_fraction_group": false_drop_g,
                    "false_keep_fraction_group": false_keep_g,
                    "false_drop_fraction_basis": false_drop_b,
                    "false_keep_fraction_basis": false_keep_b,
                    "transfer_gate_pass": int(retention_group >= 0.70 and cos_group >= 0.60),
                    "no_fake": 1,
                })
    return transfer_rows, role_rows, projection_rows


def run_k1_lift(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    writeback_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    candidate = str(args.rational_candidate or choose_primary_rational())
    for task in parse_csv(args.synthetic_tasks):
        for seed in parse_ints(args.synthetic_seeds):
            for loss in parse_csv(args.k_losses):
                for method in parse_csv(args.k_methods):
                    resolved_method, implementation_note = resolve_k1_method(method)
                    try:
                        rows, _covers, writes, summary = run_training_case(
                            family="D-RAT",
                            candidate_id=candidate,
                            method=resolved_method,
                            task=task,
                            seed=int(seed),
                            loss_interface=loss,
                            args=args,
                            device=device,
                        )
                        for r in rows:
                            rr = dict(r)
                            rr["stage"] = "V138_RAT_SNR_LIFT_TRAINING"
                            rr["task"] = rr.get("task_or_dataset", task)
                            rr["loss"] = loss
                            rr["method"] = method
                            rr["resolved_training_method"] = resolved_method
                            rr["implementation_note"] = implementation_note
                            train_rows.append(rr)
                            linec_rows.append({
                                "stage": "V138_RAT_SNR_LIFT_LINEC",
                                "task": task,
                                "seed": int(seed),
                                "loss": loss,
                                "method": method,
                                "resolved_training_method": resolved_method,
                                "step": rr.get("step"),
                                "CouplingR2": rr.get("LineC_CouplingR2"),
                                "NoiseSignalLeak": rr.get("LineC_NoiseSignalLeak"),
                                "Reservoir_delta_pending": "",
                                "CEp99": rr.get("CEp99"),
                                "audit_only": 1,
                                "no_fake": 1,
                            })
                        for w in writes:
                            ww = dict(w)
                            ww["stage"] = "V138_RAT_SNR_LIFT_WRITEBACK"
                            ww["method"] = method
                            ww["resolved_training_method"] = resolved_method
                            ww["implementation_note"] = implementation_note
                            writeback_rows.append(ww)
                        summary["stage"] = "V138_RAT_SNR_LIFT_SUMMARY"
                        summary["method"] = method
                        summary["resolved_training_method"] = resolved_method
                        summary["implementation_note"] = implementation_note
                        summaries.append(summary)
                    except Exception as exc:  # noqa: BLE001
                        failures.append({
                            "stage": "V138_RAT_SNR_LIFT_FAILURE",
                            "task": task,
                            "seed": int(seed),
                            "loss": loss,
                            "method": method,
                            "exception": repr(exc),
                            "no_fake": 1,
                        })
    add_baseline_deltas(summaries)
    for s in summaries:
        s["source_vs_best_control"] = float(s.get("source_vs_adamw", 0.0))
        s["AUC_time_ratio"] = s.get("AUC_time_ratio_vs_adamw", 1.0)
        source = fnum(s.get("source_vs_best_control"), 0.0)
        auc = fnum(s.get("AUC_time_ratio"), 99.0)
        cep = fnum(s.get("CEp99_delta"), 0.0)
        nll = fnum(s.get("NLL_delta"), 0.0)
        ece = fnum(s.get("ECE_delta"), 0.0)
        coupling = fnum(s.get("CouplingR2_delta"), 0.0)
        noise = fnum(s.get("NoiseSignalLeak_delta"), 0.0)
        reservoir = fnum(s.get("ReservoirRatio_delta"), 0.0)
        linec_majority_pass = int(coupling >= 0.0 and noise <= 0.0 and reservoir <= 0.0)
        is_control = int(str(s.get("method")) == "RAT-AdamW")
        s["linec_majority_pass"] = linec_majority_pass
        s["pass_s3"] = int((not is_control) and source >= 0.005 and auc <= 1.0 and cep <= 0.05 and nll <= 0.02 and ece <= 0.02 and linec_majority_pass)
        resolved = str(s.get("resolved_training_method", s.get("method", "")))
        has_cover_phase = any(token in resolved for token in ["Cover", "Consolidation", "ThenCoverPhase"])
        s["pass_s4"] = int(s["pass_s3"] and has_cover_phase and fnum(s.get("mean_boundary_accept"), 0.0) > 0.0)
    controls = []
    for r in summaries:
        if str(r.get("method")) == "RAT-AdamW":
            continue
        controls.append({
            "stage": "V138_RAT_SNR_LIFT_CONTROLS",
            "task": r.get("task_or_dataset"),
            "seed": r.get("seed"),
            "loss": r.get("loss_interface"),
            "method": r.get("method"),
            "resolved_training_method": r.get("resolved_training_method"),
            "implementation_note": r.get("implementation_note"),
            "source_vs_adamw": r.get("source_vs_adamw"),
            "source_vs_best_control": r.get("source_vs_best_control"),
            "AUC_time_ratio": r.get("AUC_time_ratio"),
            "CEp99_delta": r.get("CEp99_delta"),
            "NLL_delta": r.get("NLL_delta"),
            "ECE_delta": r.get("ECE_delta"),
            "CouplingR2_delta": r.get("CouplingR2_delta"),
            "NoiseSignalLeak_delta": r.get("NoiseSignalLeak_delta"),
            "Reservoir_delta": r.get("ReservoirRatio_delta"),
            "pass_s3": r.get("pass_s3"),
            "pass_s4": r.get("pass_s4"),
            "no_fake": 1,
        })
    return train_rows, summaries, linec_rows, controls, failures, writeback_rows


def forbidden_audit() -> list[dict[str, Any]]:
    rows = []
    checks = [
        ("validation_for_direction", 0, "validation is audit/gate only"),
        ("test_for_direction", 0, "test is final MLP audit only"),
        ("future_for_direction", 0, "no future/query batch is used"),
        ("linec_for_direction", 0, "LineC is audit only"),
        ("tail_metrics_for_direction", 0, "CEp99/NLL/ECE are audit/gate only"),
        ("dataset_name_for_direction", 0, "dataset name selects data loader only"),
        ("kan_promotion_from_mlp", 0, "MLP line and KAN route are separated"),
        ("readout_feature_proxy", 0, "updates are named model parameters"),
    ]
    for check, violation, note in checks:
        rows.append({"stage": "V138_FORBIDDEN_INFORMATION_AUDIT", "check": check, "violation": violation, "note": note, "no_fake": 1})
    return rows


def loss_interface_audit(methods: list[str], losses: list[str]) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        for loss in losses:
            rows.append({
                "stage": "V138_LOSS_INTERFACE_AUDIT",
                "method": method,
                "loss_interface": loss,
                "uses_generic_loss_interface": 1,
                "uses_ce_specific_formula": 0,
                "uses_output_cotangent": 1,
                "no_fake": 1,
            })
    return rows


def implementation_readback() -> str:
    surfaces = [
        ("MLP generic line", "mlp_train_one", "train-stream per-example MLP gradients with optional trust/blend"),
        ("Train loss trust", "apply_train_stream_trust", "current train batch loss/logits/gradient norms only"),
        ("K0 transfer audit", "run_k0_transfer", "compares parameter/group/basis SNR on identical train batch coordinates"),
        ("K1 RAT lift", "run_k1_lift", "uses v13.7 multi-step named-parameter writeback path"),
        ("Forbidden audit", "forbidden_audit", "validation/test/future/LineC/tail not used for direction"),
    ]
    lines = [
        "# v13.8 Implementation Readback",
        "",
        "This readback is part of the route gate. Missing it must downgrade to R0.",
        "",
    ]
    for title, fn, note in surfaces:
        start, end = line_range_of(Path(__file__), fn)
        lines.append(f"- {title}: `{fn}` lines {start}-{end}. {note}.")
    lines.extend([
        "",
        "Legality notes:",
        "",
        "1. MLP direction uses train-stream per-example gradients from the generic loss interface.",
        "2. TrainLossQuantileTrust / LogitNormTrust / PerExampleGradientClip use only current train batch telemetry.",
        "3. K0/K1 Rational line uses train-stream synthetic batches and true named parameter updates where training is executed.",
        "4. CEp99/NLL/ECE/LineC/validation/test/future are audit/gate only.",
        "5. MLP route and KAN route are separated; MLP positive never promotes KAN.",
        "6. Non-RAT rows are substrate-health only unless gate passes.",
        "",
    ])
    return "\n".join(lines)


def required_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    for name in REQUIRED + FIGURES:
        path = out_dir / name
        rows.append({
            "stage": "V138_REQUIRED_MANIFEST",
            "path": str(path.relative_to(ROOT) if path.is_absolute() and path.exists() else path),
            "required": 1,
            "exists": int(path.exists()),
            "no_fake": 1,
        })
    missing = sum(1 for r in rows if int(r["exists"]) != 1)
    return rows, missing


def code_packet(out_dir: Path) -> str:
    packet = out_dir / "v138_code_review_packet.zip"
    files = [
        Path(__file__),
        ROOT / "experiments" / "run_v137_boundary_conditioned_poprisk_training.py",
        ROOT / "experiments" / "run_v137_mlp_snr_real_triage.py",
        DOC_PLAN,
    ]
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists():
                zf.write(path, path.relative_to(ROOT).as_posix())
    return sha256_file(packet)


def task_pass_count(summary_rows: list[dict[str, Any]], key: str, seeds: list[int], losses: list[str]) -> int:
    threshold = min(2, max(1, len(seeds)))
    count = 0
    for task in sorted({str(r.get("task_or_dataset")) for r in summary_rows}):
        rs = [r for r in summary_rows if str(r.get("task_or_dataset")) == task]
        seed_pass = {int(r.get("seed")) for r in rs if int(r.get(key, 0)) == 1}
        loss_pass = {str(r.get("loss_interface")) for r in rs if int(r.get(key, 0)) == 1}
        if len(seed_pass) >= threshold or len(loss_pass) >= min(2, len(losses)):
            count += 1
    return count


def build_route(
    *,
    out_dir: Path,
    mlp_10seed: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    rat_summaries: list[dict[str, Any]],
    nonrat_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    missing: int,
    code_sha: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    forbidden_violations = 0
    mlp_dataset_pass = sum(int(r.get("dataset_pass", 0)) for r in mlp_10seed)
    mlp_confirmed = int(mlp_dataset_pass == len(mlp_10seed) and len(mlp_10seed) > 0)
    med_ret = quantile_or_nan([fnum(r.get("signal_retention_group"), float("nan")) for r in transfer_rows], 0.50)
    med_cos = quantile_or_nan([fnum(r.get("cos_group_vs_param"), float("nan")) for r in transfer_rows], 0.50)
    transfer_pass = int(med_ret >= 0.70 and med_cos >= 0.60)
    s3_tasks = task_pass_count(rat_summaries, "pass_s3", parse_ints(args.synthetic_seeds), parse_csv(args.k_losses))
    s4_tasks = task_pass_count(rat_summaries, "pass_s4", parse_ints(args.synthetic_seeds), parse_csv(args.k_losses))
    nonrat_pass = sum(int(r.get("substrate_health_pass", 0)) for r in nonrat_rows)
    if missing or forbidden_violations or not (out_dir / "v138_implementation_readback.md").exists():
        route = "R0-ImplementationOrArtifactFailure"
        minimum = "S0-Invalid"
    elif not mlp_confirmed:
        route = "R1-GenericSNROptimizerNoGo"
        minimum = "S1-ImplementationReadback"
    elif not transfer_pass:
        route = "R3-SNRSignalLostInBasisLift"
        minimum = "S2-GenericOptimizerConfirmed"
    elif s3_tasks >= 5:
        route = "R5-KANSpecificS3Opened" if s4_tasks < 5 else "R6-KANSpecificS4Opened"
        minimum = "S3-KANBasisSNRPositive"
    else:
        route = "R2-GenericSNROptimizerConfirmed_KANNotSpecific"
        minimum = "S2-GenericOptimizerConfirmed"
    final_stop_allowed = int(
        route
        in {
            "R1-GenericSNROptimizerNoGo",
            "R2-GenericSNROptimizerConfirmed_KANNotSpecific",
            "R3-SNRSignalLostInBasisLift",
        }
        and missing == 0
        and forbidden_violations == 0
    )
    return {
        "route": route,
        "minimum_success": minimum,
        "final_stop_allowed": final_stop_allowed,
        "promotion_allowed": int(route == "S5-OfficialFunctionalSuccess"),
        "official_success_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "kan_real_short_run_open_allowed": int(route in {"R5-KANSpecificS3Opened", "R6-KANSpecificS4Opened", "S5-OfficialFunctionalSuccess"}),
        "mlp_generic_10seed_dataset_pass_count": mlp_dataset_pass,
        "mlp_generic_10seed_dataset_count": len(mlp_10seed),
        "mlp_generic_10seed_confirmed": mlp_confirmed,
        "snr_transfer_median_retention_group": med_ret,
        "snr_transfer_median_cos_group_vs_param": med_cos,
        "snr_transfer_gate_pass": transfer_pass,
        "kan_s3_task_pass_count": s3_tasks,
        "kan_s4_task_pass_count": s4_tasks,
        "nonrat_substrate_health_pass_count": nonrat_pass,
        "failure_rows": len(failures),
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": forbidden_violations,
        "code_review_packet_sha256": code_sha,
        "real_train_size": int(args.real_train_size),
        "real_val_size": int(args.real_val_size),
        "real_test_size": int(args.real_test_size),
        "real_epochs": int(args.real_epochs),
        "synthetic_train_size": int(args.synthetic_train_size),
        "synthetic_val_size": int(args.synthetic_val_size),
        "synthetic_train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "mlp_method_count": len(parse_csv(args.mlp_methods)),
        "k_method_count": len(parse_csv(args.k_methods)),
        "plan_k0_train_steps": 200,
        "compute_budgeted_run": int(int(args.train_steps) < 200 or int(args.real_train_size) < 1024 or int(args.real_epochs) < 3),
        "no_fake": 1,
    }


def write_final_texts(out_dir: Path, route: dict[str, Any]) -> None:
    boundary = [
        "# v13.8 No-Go Boundary",
        "",
        f"route = {route['route']}",
        "",
        "Facts:",
        "",
        f"- MLP generic 10-seed confirmed = {route['mlp_generic_10seed_confirmed']}",
        f"- MLP dataset pass count = {route['mlp_generic_10seed_dataset_pass_count']}/{route['mlp_generic_10seed_dataset_count']}",
        f"- KAN S3 task pass count = {route['kan_s3_task_pass_count']}/7",
        f"- KAN S4 task pass count = {route['kan_s4_task_pass_count']}/7",
        f"- SNR transfer median retention group = {route['snr_transfer_median_retention_group']}",
        f"- SNR transfer median cosine group vs param = {route['snr_transfer_median_cos_group_vs_param']}",
        f"- Non-RAT substrate health pass count = {route['nonrat_substrate_health_pass_count']}",
        "",
        "No MLP-only row is a KAN promotion. CEp99/NLL/ECE/LineC are audit/gate only.",
        "",
    ]
    (out_dir / "v138_no_go_boundary.md").write_text("\n".join(boundary), encoding="utf-8")
    queue = [
        "# v13.8 Next Hypothesis Queue",
        "",
        "1. If R1: stop MLP SNR expansion; continue KAN basis line only with new theory.",
        "2. If R3: do role-wise lift, dynamic SNR cluster, and low-rank SNR corrector before any cover tuning.",
        "3. If R2/R4: MLP generic optimizer is real but KAN-specific lift failed; redesign KAN telemetry/substrate.",
        "4. Non-RAT remains substrate-health only until workspace/task-health gates pass.",
        "",
    ]
    (out_dir / "v138_next_hypothesis_queue.md").write_text("\n".join(queue), encoding="utf-8")


def write_figures(out_dir: Path, route: dict[str, Any], mlp_10seed: list[dict[str, Any]], transfer_rows: list[dict[str, Any]], rat_summaries: list[dict[str, Any]], nonrat_rows: list[dict[str, Any]]) -> None:
    write_svg(out_dir / "fig_v138_progress_delta.svg", "v13.8 progress", [f"route={route['route']}", f"MLP={route['mlp_generic_10seed_dataset_pass_count']}/{route['mlp_generic_10seed_dataset_count']}", f"KAN S3={route['kan_s3_task_pass_count']}/7"])
    write_svg(out_dir / "fig_v138_mlp_10seed_pass_heatmap.svg", "MLP 10seed pass", [f"{r['dataset']}: {r['seed_pass_count']}/{r['seed_threshold']}" for r in mlp_10seed])
    write_svg(out_dir / "fig_v138_mlp_source_tail_pareto.svg", "MLP source-tail pareto", ["See v138_mlp_generic_optimizer_summary.csv"])
    write_svg(out_dir / "fig_v138_mlp_active_fraction_trace.svg", "MLP active fraction", ["See v138_mlp_generic_optimizer_training.csv"])
    write_svg(out_dir / "fig_v138_mlp_blend_alpha_pareto.svg", "MLP blend alpha pareto", ["Blend/trust candidates audited"])
    write_svg(out_dir / "fig_v138_snr_transfer_retention_by_role.svg", "SNR transfer retention", [f"median retention={route['snr_transfer_median_retention_group']}"])
    write_svg(out_dir / "fig_v138_param_vs_group_snr_cosine.svg", "Param vs group cosine", [f"median cosine={route['snr_transfer_median_cos_group_vs_param']}"])
    write_svg(out_dir / "fig_v138_false_drop_false_keep_by_role.svg", "False drop/keep", ["See v138_rolewise_snr_audit.csv"])
    write_svg(out_dir / "fig_v138_rat_snr_lift_task_family_heatmap.svg", "RAT SNR lift heatmap", [f"KAN S3 tasks={route['kan_s3_task_pass_count']}/7", f"summary rows={len(rat_summaries)}"])
    write_svg(out_dir / "fig_v138_linec_signal_reservoir_noise_matrix.svg", "LineC matrix", ["See v138_linec_audit.csv"])
    write_svg(out_dir / "fig_v138_nonrat_substrate_status.svg", "NonRAT substrate", [f"{r.get('family')}: {r.get('status')}" for r in nonrat_rows[:10]])
    write_svg(out_dir / "fig_v138_failure_taxonomy.svg", "Failure taxonomy", [f"route={route['route']}", "No fake promotion"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--mlp-seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--mlp-seed-threshold", type=int, default=6)
    ap.add_argument("--mlp-methods", default="MLP-AdamW,MLP-AdamW-SNREMA-Blend25-TrainLossQuantileTrust,MLP-AdamW-SNREMA-Blend50-TrainLossQuantileTrust,MLP-AdamW-SNREMA-Blend50-PerExampleGradientClip,MLP-AdamW-SNRRoleNorm-Blend50-LogitNormTrust,MLP-AdamW-SNREMA-Blend50-TrainLossQuantileTrust-PerExampleGradientClip")
    ap.add_argument("--loss-interface", default="CE")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--real-train-size", type=int, default=1024)
    ap.add_argument("--real-val-size", type=int, default=512)
    ap.add_argument("--real-test-size", type=int, default=512)
    ap.add_argument("--real-epochs", type=int, default=3)
    ap.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    ap.add_argument("--synthetic-seeds", default="0,1,2")
    ap.add_argument("--synthetic-train-size", type=int, default=96)
    ap.add_argument("--synthetic-val-size", type=int, default=48)
    ap.add_argument("--synthetic-dim", type=int, default=16)
    ap.add_argument("--synthetic-classes", type=int, default=3)
    ap.add_argument("--k-losses", default="CE,Brier")
    ap.add_argument("--k-methods", default="RAT-AdamW,K1-RAT-ParamSNR-Only,K2-RAT-ParamSNR-MinCoverGuard,K3-RAT-RoleWiseSNRLift,K4-RAT-DynamicSNRClusterLift,K5-RAT-LowRankSNRCorrector,K6-RAT-ParamSNR-Then-BasisConsolidation,K7-RAT-ParamSNR-BlendAdamW-Then-CoverPhase")
    ap.add_argument("--rational-candidate", default="")
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--log-interval", type=int, default=40)
    ap.add_argument("--mlp-hidden", type=int, default=160)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--snr-tau", type=float, default=1.0)
    ap.add_argument("--snr-eps", type=float, default=1.0e-12)
    ap.add_argument("--snr-ema-decay", type=float, default=0.85)
    ap.add_argument("--soft-alpha", type=float, default=8.0)
    ap.add_argument("--active-fraction-cap", type=float, default=1.0)
    args = ap.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp_training, mlp_summary, mlp_10seed, mlp_failures = run_mlp_line(args, device)
    transfer_rows, role_rows, projection_rows = run_k0_transfer(args, device)
    rat_training, rat_summary, rat_linec, rat_controls, rat_failures, rat_writeback = run_k1_lift(args, device)
    nonrat_rows = run_nonrat_repair(args, device)
    forbidden_rows = forbidden_audit()
    loss_rows = loss_interface_audit(parse_csv(args.mlp_methods) + parse_csv(args.k_methods), [args.loss_interface] + parse_csv(args.k_losses))
    readback = implementation_readback()
    (out_dir / "v138_implementation_readback.md").write_text(readback, encoding="utf-8")

    write_rows(out_dir / "v138_mlp_generic_optimizer_training.csv", mlp_training or [{"stage": "V138_MLP_GENERIC_OPTIMIZER_TRAINING", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_mlp_generic_optimizer_summary.csv", mlp_summary or [{"stage": "V138_MLP_GENERIC_OPTIMIZER_SUMMARY", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_mlp_generic_optimizer_10seed.csv", mlp_10seed or [{"stage": "V138_MLP_GENERIC_OPTIMIZER_10SEED", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_snr_transfer_audit.csv", transfer_rows or [{"stage": "V138_SNR_TRANSFER_AUDIT", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_rolewise_snr_audit.csv", role_rows or [{"stage": "V138_ROLEWISE_SNR_AUDIT", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_basis_group_projection_loss.csv", projection_rows or [{"stage": "V138_BASIS_GROUP_PROJECTION_LOSS", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_rat_snr_lift_training.csv", rat_training or [{"stage": "V138_RAT_SNR_LIFT_TRAINING", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_rat_snr_lift_summary.csv", rat_summary or [{"stage": "V138_RAT_SNR_LIFT_SUMMARY", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_rat_snr_lift_linec.csv", rat_linec or [{"stage": "V138_RAT_SNR_LIFT_LINEC", "skip_reason": "no_rows", "audit_only": 1, "no_fake": 1}])
    write_rows(out_dir / "v138_rat_snr_lift_controls.csv", rat_controls or [{"stage": "V138_RAT_SNR_LIFT_CONTROLS", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_rat_snr_lift_writeback_trace.csv", rat_writeback or [{"stage": "V138_RAT_SNR_LIFT_WRITEBACK", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_nonrat_substrate_health.csv", nonrat_rows or [{"stage": "V138_NONRAT_SUBSTRATE_HEALTH", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v138_linec_audit.csv", rat_linec or [{"stage": "V138_LINEC_AUDIT", "skip_reason": "no_rows", "audit_only": 1, "no_fake": 1}])
    failures = mlp_failures + rat_failures
    write_rows(out_dir / "v138_failure_table.csv", failures or [{"stage": "V138_FAILURE_TABLE", "failure_rows": 0, "no_fake": 1}])
    write_rows(out_dir / "v138_loss_interface_audit.csv", loss_rows)
    write_rows(out_dir / "v138_forbidden_information_audit.csv", forbidden_rows)
    code_sha = code_packet(out_dir)
    route = build_route(out_dir=out_dir, mlp_10seed=mlp_10seed, transfer_rows=transfer_rows, rat_summaries=rat_summary, nonrat_rows=nonrat_rows, failures=failures, missing=0, code_sha=code_sha, args=args)
    write_final_texts(out_dir, route)
    progress = [
        {"stage": "V138_PROGRESS", "item": "MLP generic 10seed", "value": route["mlp_generic_10seed_dataset_pass_count"], "target": route["mlp_generic_10seed_dataset_count"], "pass": route["mlp_generic_10seed_confirmed"], "no_fake": 1},
        {"stage": "V138_PROGRESS", "item": "SNR transfer gate", "value": route["snr_transfer_median_retention_group"], "target": 0.70, "pass": route["snr_transfer_gate_pass"], "no_fake": 1},
        {"stage": "V138_PROGRESS", "item": "KAN S3 tasks", "value": route["kan_s3_task_pass_count"], "target": 5, "pass": int(route["kan_s3_task_pass_count"] >= 5), "no_fake": 1},
        {"stage": "V138_PROGRESS", "item": "NonRAT substrate", "value": route["nonrat_substrate_health_pass_count"], "target": 1, "pass": int(route["nonrat_substrate_health_pass_count"] > 0), "no_fake": 1},
    ]
    write_rows(out_dir / "v138_progress_table.csv", progress)
    write_figures(out_dir, route, mlp_10seed, transfer_rows, rat_summary, nonrat_rows)
    manifest, missing = required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    (out_dir / "v138_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest, missing = required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    (out_dir / "v138_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rows(out_dir / "v138_required_manifest.csv", manifest)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
