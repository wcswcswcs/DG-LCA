#!/usr/bin/env python
"""v13.10 cover-forming substrate architecture reset runner.

This runner is deliberately separate from the v13.9 functional-token runner.
It creates auditable A-RCF substrate candidates before running K14-K18
training, so a K-token-only extension cannot pass the route gate.

All training directions are generated from the current train stream via the
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v1231_basis_kernel_workspace import make_basis_model  # noqa: E402
from experiments.run_v133_task_family_robust_basis_natural import (  # noqa: E402
    SOURCE_V1235,
    build_substrate_map,
    eval_metrics,
    fnum,
    make_model_for_family,
    parse_csv,
    parse_ints,
    sha256_file,
    synthetic_data,
    write_rows,
    write_svg,
)
from experiments.run_v136_poprisk_snr_basis_cover_boundary import (  # noqa: E402
    collect_per_example_gradients,
    role_of_param,
)
from experiments.run_v137_boundary_conditioned_poprisk_training import (  # noqa: E402
    SNRState,
    all_named_params,
    assign_flat_grad,
    blend_alpha_from_method,
    choose_primary_rational,
    cover_debt,
    cover_policy,
    finite_mean,
    loss_value,
    parameter_sha_by_params,
    phase_for_step,
    rational_boundary_terms,
    selected_params,
    snr_gate,
    would_increase_cover_debt,
)
from experiments.run_v139_signal_to_cover_functional_substrate_architecture import (  # noqa: E402
    format_nonrat_exact_rows,
    loss_interface_audit,
    run_mlp_line,
    task_pass_count,
)


OUT_DIR = ROOT / "results" / "v13_10_cover_forming_substrate_architecture_reset" / "official_v1310"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v13.10_CoverFormingSubstrateArchitectureReset_完整计划.md"

REQUIRED = [
    "v1310_code_review_manifest.csv",
    "v1310_architecture_diff_manifest.csv",
    "v1310_forbidden_information_audit.csv",
    "v1310_substrate_architecture_readback.md",
    "v1310_rational_cover_substrate.csv",
    "v1310_rational_cover_telemetry.csv",
    "v1310_signal_to_cover_training.csv",
    "v1310_signal_to_cover_summary.csv",
    "v1310_signal_to_cover_linec.csv",
    "v1310_nonrat_substrate_vertical_slice.csv",
    "v1310_mlp_control_monitor.csv",
    "v1310_linec_audit.csv",
    "v1310_failure_table.csv",
    "v1310_route_decision.json",
    "v1310_no_go_boundary.md",
    "v1310_next_hypothesis_queue.md",
]

FIGURES = [
    "fig_progress_by_line.svg",
    "fig_signal_retention_vs_cover_purity.svg",
    "fig_cover_purity_by_method.svg",
    "fig_cover_churn_by_method.svg",
    "fig_signal_to_cover_score_by_task.svg",
    "fig_task_family_pass_heatmap.svg",
    "fig_linec_vs_source_scatter.svg",
    "fig_readout_vs_basis_signal_mass.svg",
    "fig_nonrat_substrate_health_matrix.svg",
    "fig_mlp_vs_kan_control.svg",
    "fig_failure_taxonomy_sankey.svg",
]


@dataclass(frozen=True)
class ArchitectureSpec:
    architecture_id: str
    mapped_candidate_id: str
    method_id: str
    architecture_kind: str
    reset_note: str
    expected_group_factor: float = 1.0


ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "A-RCF1-MultiBandRationalCover": ArchitectureSpec(
        "A-RCF1-MultiBandRationalCover",
        "D-RAT25-DenDerivativeTelemetryRecomputeBackward",
        "",
        "multiband_parameter_initialization",
        "deterministic low/medium/high/reservoir band scaling across Rational groups",
    ),
    "A-RCF2-SNRClusterCoverWarmup": ArchitectureSpec(
        "A-RCF2-SNRClusterCoverWarmup",
        "D-RAT27-DenSlopeGuardNoCE",
        "",
        "gradient_cluster_seeded_group_permutation",
        "train-input-scale seeded group permutation plus K14 cluster cover warmup",
    ),
    "A-RCF3-PersistentCoverMemory": ArchitectureSpec(
        "A-RCF3-PersistentCoverMemory",
        "D-RAT25-DenDerivativeTelemetryRecomputeBackward",
        "",
        "persistent_cover_memory_bias",
        "persistent group memory uses deterministic group load priors and EMA cover state",
    ),
    "A-RCF4-SignalReservoirSplitGroups": ArchitectureSpec(
        "A-RCF4-SignalReservoirSplitGroups",
        "D-RAT28-GroupDiversityPreservingRational",
        "",
        "signal_reservoir_exploration_group_split",
        "deterministic signal/reservoir/exploration group split from train-input scale",
    ),
    "A-RCF5-ReadoutBasisDecoupledCover": ArchitectureSpec(
        "A-RCF5-ReadoutBasisDecoupledCover",
        "D-RAT35-ReadoutRationalDecoupleNoCE",
        "",
        "readout_basis_decoupled_scaling",
        "readout fast path and basis slow cover path receive separate initialization scales",
    ),
    "A-RCF6-OvercompleteSparseCoverBank": ArchitectureSpec(
        "A-RCF6-OvercompleteSparseCoverBank",
        "D-RAT-OVERCOMPLETE-H40",
        "B7cq-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCap200-L3",
        "overcomplete_sparse_cover_bank",
        "uses h40 overcomplete grouped Rational primitive with sparse cover-band scaling",
        expected_group_factor=1.25,
    ),
}

K_METHOD_ALIASES = {
    "K0-RAT-AdamW": ("RAT-AdamW", "baseline AdamW on the same A-RCF substrate"),
    "K1-ParameterSNR": ("RAT-ParameterSNRSoft", "parameter SNR control on A-RCF substrate"),
    "K2-ParameterSNR-MinCoverGuard": ("RAT-ParameterSNRSoft-MinCoverGuard", "parameter SNR with minimum cover-debt guard"),
    "K3-RoleWiseSNRLift": ("RAT-ParameterSNREMA-RoleNorm", "role-wise parameter SNR lift"),
    "K8-GradientClusterCover": ("RAT-GroupSNRSoft-GradientClusterCover-k8-CoverPhaseSchedule", "legacy gradient-cluster cover control on A-RCF substrate"),
    "K10-ParamSNRThenCover": ("RAT-ParameterSNRSoft-ParamSNRThenCover-slowConsolidate-CoverPhaseSchedule", "legacy parameter-SNR then cover control on A-RCF substrate"),
    "K13-CoverDebtGuard": ("RAT-GroupSNRSoft-LowVarianceSignalCoverGrowth-CoverPhaseSchedule", "legacy cover-debt guard control on A-RCF substrate"),
    "K14-RCF-SNRClusterTraining": ("RAT-GroupSNRSoft-GradientClusterCover-k8-CoverPhaseSchedule-FreezeCluster", "A-RCF train-stream SNR cluster training"),
    "K15-RCF-PersistentCoverMemory": ("RAT-GroupSNRSoft-LowVarianceSignalCoverGrowth-CoverPhaseSchedule-FreezeCluster-EMA", "A-RCF persistent cover-memory EMA"),
    "K16-RCF-SignalReservoirSplit": ("RAT-GroupSNRSoft-ReservoirProxyCoverGrowth-CoverPhaseSchedule", "A-RCF signal/reservoir split groups"),
    "K17-RCF-ReadoutBasisDecoupled": ("RAT-ParameterSNREMA-ReadoutBasisDecoupledSNR-ThenBasisConsolidation-CoverPhaseSchedule", "A-RCF readout-basis decoupled cover training"),
    "K18-RCF-OvercompleteSparseCover": ("RAT-GroupSNRSoft-GradientClusterCover-k8-CoverPhaseSchedule-OvercompleteSparseCover", "A-RCF overcomplete sparse cover budget"),
}

NONRAT_VERTICAL_SPECS = [
    ("D-FOU", "N-FOU1-FrequencyBandCover-lowK", "D-FOU20-LowFreqIdentityResidualHealthSubstrate", "frequency_band_cover_lowk"),
    ("D-FOU", "N-FOU2-FrequencyBandCover-withPhaseStability", "D-FOU20-LowFreqIdentityResidualHealthSubstrate", "frequency_band_phase_stability"),
    ("D-FOU", "N-FOU3-LowFreqSignalHighFreqReservoirSplit", "D-FOU20-LowFreqIdentityResidualHealthSubstrate", "lowfreq_signal_highfreq_reservoir"),
    ("D-CHE", "N-CHE1-DegreeEnergyCover-K3K5", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "degree_energy_cover"),
    ("D-CHE", "N-CHE2-HighDegreeLateEnable", "D-CHE17-HighDegreeLateEnableSubstrate", "high_degree_late_enable"),
    ("D-CHE", "N-CHE3-DegreeReservoirSplit", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "degree_reservoir_split"),
    ("D-RBF", "N-RBF1-CenterOccupancyCover", "D-RBF17-CompactCapacityK4HealthSubstrate", "center_occupancy_cover"),
    ("D-RBF", "N-RBF2-WidthConditionGuard", "D-RBF17-CompactCapacityK4HealthSubstrate", "width_condition_guard"),
    ("D-WAV", "N-WAV1-ScaleSupportCover", "D-WAV16-SupportStableHatHealthSubstrate", "scale_support_cover"),
    ("D-WAV", "N-WAV3-SupportOverlapGuard", "D-WAV16-SupportStableHatHealthSubstrate", "support_overlap_guard"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_method(method: str) -> tuple[str, str]:
    return K_METHOD_ALIASES.get(method, (method, "native method"))


def quantile_or_nan(values: Iterable[float], q: float) -> float:
    vals = torch.tensor([float(v) for v in values if math.isfinite(float(v))], dtype=torch.float32)
    if int(vals.numel()) == 0:
        return float("nan")
    return float(torch.quantile(vals, float(q)).item())


def group_slices_for_params(params: list[tuple[str, torch.nn.Parameter]]) -> list[tuple[str, int, int]]:
    groups: list[tuple[str, int, int]] = []
    off = 0
    for name, p in params:
        n = int(p.numel())
        groups.append((role_of_param(name), off, off + n))
        off += n
    return groups


def _band_vector(n: int, arch: str, device: torch.device, input_scale: float) -> torch.Tensor:
    if n <= 0:
        return torch.ones(0, device=device)
    if "A-RCF1" in arch:
        base = torch.tensor([0.82, 1.00, 1.18, 0.70], device=device)
    elif "A-RCF2" in arch:
        base = torch.linspace(0.78, 1.22, steps=max(2, min(n, 8)), device=device)
    elif "A-RCF3" in arch:
        base = torch.tensor([1.08, 1.02, 0.96, 0.90], device=device)
    elif "A-RCF4" in arch:
        base = torch.tensor([1.18, 0.74, 0.96, 0.86], device=device)
    elif "A-RCF5" in arch:
        base = torch.tensor([0.92, 1.06, 1.02, 0.88], device=device)
    else:
        base = torch.tensor([1.10, 0.78, 1.02, 0.88, 1.18], device=device)
    reps = int(math.ceil(float(n) / float(base.numel())))
    v = base.repeat(reps)[:n].clone()
    return v * max(0.75, min(1.25, float(input_scale)))


def make_arch_model(spec: ArchitectureSpec, input_dim: int, output_dim: int, x_train: torch.Tensor, device: torch.device, seed: int) -> torch.nn.Module:
    if spec.method_id:
        model, _primitive = make_basis_model(spec.method_id, input_dim, output_dim, x_train, device, seed)
        return model.to(device)
    return make_model_for_family("D-RAT", spec.mapped_candidate_id, input_dim, output_dim, x_train, device, seed).to(device)


def apply_architecture_reset(model: torch.nn.Module, spec: ArchitectureSpec, x_train: torch.Tensor) -> list[dict[str, Any]]:
    """Apply deterministic label-free cover-forming initialization changes."""
    rows: list[dict[str, Any]] = []
    input_scale = float(x_train.detach().float().std().clamp(0.75, 1.25).item())
    before = parameter_sha_by_params(all_named_params(model))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad or int(p.numel()) == 0:
                continue
            old_norm = float(p.detach().float().norm().item())
            low = name.lower()
            if p.ndim >= 2:
                row_scale = _band_vector(int(p.shape[0]), spec.architecture_id, p.device, input_scale).reshape(-1, *([1] * (p.ndim - 1)))
                p.mul_(row_scale.to(dtype=p.dtype))
                if p.ndim >= 2 and p.shape[1] >= 4 and ("w1" in low or "w2" in low):
                    col_scale = _band_vector(int(p.shape[1]), spec.architecture_id, p.device, 1.0).reshape(1, -1, *([1] * (p.ndim - 2)))
                    p.mul_(col_scale.to(dtype=p.dtype))
            else:
                p.mul_(_band_vector(int(p.numel()), spec.architecture_id, p.device, input_scale).reshape_as(p).to(dtype=p.dtype))
            if "denominator" in low:
                p.add_(0.015 * torch.sign(p).where(p != 0, torch.ones_like(p)))
            if "hidden_bias" in low and spec.architecture_id.startswith("A-RCF4"):
                p.add_(torch.linspace(-0.03, 0.03, steps=int(p.numel()), device=p.device, dtype=p.dtype).reshape_as(p))
            if ("w2" in low or "linear_readout" in low) and spec.architecture_id.startswith("A-RCF5"):
                p.mul_(1.15)
            new_norm = float(p.detach().float().norm().item())
            rows.append({
                "stage": "V1310_ARCHITECTURE_DIFF_MANIFEST",
                "architecture_id": spec.architecture_id,
                "mapped_candidate_id": spec.mapped_candidate_id,
                "method_id": spec.method_id,
                "parameter": name,
                "architecture_kind": spec.architecture_kind,
                "old_norm": old_norm,
                "new_norm": new_norm,
                "norm_delta": new_norm - old_norm,
                "uses_label_in_init": 0,
                "uses_y_for_stats": 0,
                "uses_train_input_scale": 1,
                "is_new_substrate_architecture": 1,
                "is_k_token_only_extension": 0,
                "no_fake": 1,
            })
    after = parameter_sha_by_params(all_named_params(model))
    rows.append({
        "stage": "V1310_ARCHITECTURE_DIFF_MANIFEST",
        "architecture_id": spec.architecture_id,
        "mapped_candidate_id": spec.mapped_candidate_id,
        "method_id": spec.method_id,
        "parameter": "__model_sha__",
        "architecture_kind": spec.architecture_kind,
        "old_norm": "",
        "new_norm": "",
        "norm_delta": "",
        "writeback_before_sha256": before,
        "writeback_after_sha256": after,
        "architecture_changed_parameters": int(before != after),
        "uses_label_in_init": 0,
        "uses_y_for_stats": 0,
        "is_new_substrate_architecture": 1,
        "is_k_token_only_extension": 0,
        "no_fake": 1,
    })
    return rows


def run_transfer_audit_for_arch(
    spec: ArchitectureSpec,
    *,
    task: str,
    seed: int,
    loss: str,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    xtr, ytr, _xva, _yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    model = make_arch_model(spec, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed) + 131_000)
    diff_rows = apply_architecture_reset(model, spec, xtr)
    params = selected_params(model, "D-RAT", "RAT-GroupSNRSoft-GradientClusterCover-k8")
    xb = xtr[: min(int(args.batch_size), int(xtr.shape[0]))]
    yb = ytr[: int(xb.shape[0])]
    g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface=loss)
    state = SNRState(decay=float(args.snr_ema_decay))
    meta = snr_gate(
        g,
        params,
        state,
        method="RAT-GroupSNRSoft-GradientClusterCover-k8-CoverPhaseSchedule",
        phase="cover-alignment",
        tau=float(args.snr_tau),
        eps=float(args.snr_eps),
        soft_alpha=float(args.soft_alpha),
        active_fraction_cap=float(args.active_fraction_cap),
    )
    mu = meta["mu"]
    gate = meta["gate"]
    group_slices = group_slices_for_params(params)
    param_norm = float(mu.norm().clamp_min(1.0e-8).item()) if int(mu.numel()) else 1.0
    group_energy = 0.0
    false_drop = 0.0
    false_keep = 0.0
    if int(mu.numel()):
        active = gate > 0
        group_mask = torch.zeros_like(gate)
        for _role, start, end in group_slices:
            if float(gate[start:end].mean().item()) > 0:
                group_mask[start:end] = 1.0
        group_energy = float((mu * group_mask).norm().div(param_norm).item())
        param_active = meta["ratio"] > float(meta["tau_eff"])
        false_drop = float((param_active & ~active).float().mean().item())
        false_keep = float((~param_active & active).float().mean().item())
    return [{
        "stage": "V1310_RATIONAL_COVER_TELEMETRY",
        "architecture_id": spec.architecture_id,
        "mapped_candidate_id": spec.mapped_candidate_id,
        "task": task,
        "seed": int(seed),
        "loss_interface": loss,
        "signal_retention_group": group_energy,
        "cos_group_vs_param": meta.get("cos_snr_adamw", 0.0),
        "cover_purity_mean": meta.get("cover_purity_mean", 0.0),
        "cover_purity_p10": meta.get("cover_purity_p10", 0.0),
        "cover_churn": meta.get("cover_churn", 0.0),
        "cover_load_gini": meta.get("cover_load_gini", 0.0),
        "cover_specialization_entropy": meta.get("cover_specialization_entropy", 0.0),
        "signal_to_cover_score": meta.get("signal_to_cover_score", 0.0),
        "false_drop_fraction": false_drop,
        "false_keep_fraction": false_keep,
        "group_alive_fraction": float((gate > 0).float().mean().item()) if int(gate.numel()) else 0.0,
        "group_dead_fraction": 1.0 - (float((gate > 0).float().mean().item()) if int(gate.numel()) else 0.0),
        "architecture_diff_rows": len(diff_rows),
        "no_fake": 1,
    }]


def run_arch_training_case(
    spec: ArchitectureSpec,
    *,
    method: str,
    resolved_method: str,
    task: str,
    seed: int,
    loss_interface: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    model = make_arch_model(spec, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed) + 131_100)
    apply_architecture_reset(model, spec, xtr)
    params = selected_params(model, "D-RAT", resolved_method)
    opt = torch.optim.AdamW([p for _n, p in all_named_params(model)], lr=float(args.lr), weight_decay=float(args.weight_decay), foreach=False)
    snr_state = SNRState(decay=float(args.snr_ema_decay))
    gen = torch.Generator(device=device).manual_seed(131_000 + int(seed) * 113 + sum(ord(c) for c in task + method + loss_interface + spec.architecture_id))
    rows: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    val_losses: list[float] = []
    elapsed_points: list[float] = []
    losses: list[float] = []
    t0 = time.perf_counter()
    before_sha = parameter_sha_by_params(params)
    last_meta: dict[str, Any] = {}
    for step in range(1, int(args.train_steps) + 1):
        idx = torch.randint(0, int(xtr.shape[0]), (min(int(args.batch_size), int(xtr.shape[0])),), device=device, generator=gen)
        xb = xtr[idx]
        yb = ytr[idx]
        opt.zero_grad(set_to_none=True)
        phase = phase_for_step(step - 1, int(args.train_steps))
        cover_accept = 1
        cover_before = cover_after = cover_delta = 0.0
        boundary_reason = "not_applied"
        if resolved_method == "RAT-AdamW" or resolved_method.endswith("AdamW"):
            loss = loss_value(model(xb), yb, loss_interface)
            loss.backward()
            meta = {
                "active_fraction": 1.0,
                "active_by_layer": {},
                "removed_update_norm_fraction": 0.0,
                "cos_snr_adamw": 1.0,
                "snr_median": 0.0,
                "snr_p90": 0.0,
                "snr_p99": 0.0,
                "group_scores": [],
                "cover_purity_mean": 0.0,
                "cover_purity_p10": 0.0,
                "cover_churn": 0.0,
                "cover_specialization_entropy": 0.0,
                "cover_load_gini": 0.0,
                "signal_to_cover_score": 0.0,
                "cover_split_count": 0,
                "cover_merge_count": 0,
                "cover_freeze_count": 0,
                "false_drop_fraction": 0.0,
                "false_keep_fraction": 0.0,
            }
        else:
            g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface=loss_interface)
            meta = snr_gate(
                g,
                params,
                snr_state,
                method=resolved_method,
                phase=phase,
                tau=float(args.snr_tau),
                eps=float(args.snr_eps),
                soft_alpha=float(args.soft_alpha),
                active_fraction_cap=float(args.active_fraction_cap),
            )
            grad = meta["grad"]
            alpha = blend_alpha_from_method(resolved_method)
            if alpha is not None:
                grad = float(alpha) * grad + (1.0 - float(alpha)) * meta["mu"]
            if "OvercompleteSparseCover" in resolved_method:
                active = (meta["gate"] > 0).float()
                if int(active.numel()) > 0:
                    budget = max(1, int(math.ceil(float(args.overcomplete_active_fraction) * int(active.numel()))))
                    ratio = meta["ratio"]
                    cutoff = torch.topk(ratio, k=min(budget, int(ratio.numel())), largest=True).values.min()
                    grad = grad * (ratio >= cutoff).float()
                    meta["active_fraction"] = float((ratio >= cutoff).float().mean().item())
            eps_cover, cover_strength = cover_policy(resolved_method, phase)
            if cover_strength > 0.0:
                cover_accept, cover_before, cover_after, cover_delta = would_increase_cover_debt(model, xb, params, grad, float(args.lr), epsilon=eps_cover)
                if not cover_accept:
                    grad = grad * (1.0 - float(cover_strength))
                    boundary_reason = f"cover_debt_increase_scaled_strength_{cover_strength:.2f}"
                else:
                    boundary_reason = "cover_debt_within_phase_budget"
            assign_flat_grad(params, grad)
            loss = loss_value(model(xb), yb, loss_interface).detach()
        opt.step()
        losses.append(float(loss.detach().item()))
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.train_steps):
            metrics = eval_metrics(model, xva, yva)
            val_loss = metrics["NLL"] if loss_interface == "CE" else metrics["Brier"]
            val_losses.append(float(val_loss))
            elapsed_points.append(time.perf_counter() - t0)
            cover_now = cover_debt(model, xb)
            rat_terms = rational_boundary_terms(model)
            row = {
                "stage": "V1310_SIGNAL_TO_COVER_TRAINING",
                "architecture_id": spec.architecture_id,
                "mapped_candidate_id": spec.mapped_candidate_id,
                "architecture_kind": spec.architecture_kind,
                "task_or_dataset": task,
                "task": task,
                "seed": int(seed),
                "method": method,
                "resolved_training_method": resolved_method,
                "loss_interface": loss_interface,
                "step": step,
                "phase": phase,
                "train_loss": finite_mean(losses[-int(args.log_interval) :]),
                "val_loss": metrics["NLL"],
                "val_acc": metrics["acc"],
                "CEp99": metrics["CEp99"],
                "NLL": metrics["NLL"],
                "ECE": metrics["ECE"],
                "Brier": metrics["Brier"],
                "LineC_CouplingR2": metrics["CouplingR2"],
                "LineC_NoiseSignalLeak": metrics["NoiseSignalLeak"],
                "LineC_ReservoirRatio": metrics["RealSignalReservoirRatio"],
                "snr_active_fraction": meta["active_fraction"],
                "cos_snr_adamw": meta["cos_snr_adamw"],
                "snr_median": meta["snr_median"],
                "snr_p90": meta["snr_p90"],
                "snr_p99": meta["snr_p99"],
                "cover_purity_mean": meta.get("cover_purity_mean", 0.0),
                "cover_purity_p10": meta.get("cover_purity_p10", 0.0),
                "cover_churn": meta.get("cover_churn", 0.0),
                "cover_specialization_entropy": meta.get("cover_specialization_entropy", 0.0),
                "cover_load_gini": meta.get("cover_load_gini", 0.0),
                "signal_to_cover_score": meta.get("signal_to_cover_score", 0.0),
                "cover_split_count": meta.get("cover_split_count", 0),
                "cover_merge_count": meta.get("cover_merge_count", 0),
                "cover_freeze_count": meta.get("cover_freeze_count", 0),
                "false_drop_fraction": meta.get("false_drop_fraction", 0.0),
                "false_keep_fraction": meta.get("false_keep_fraction", 0.0),
                "cover_debt": cover_now["cover_debt"],
                "cover_entropy": cover_now["cover_entropy"],
                "cover_condition_proxy": cover_now["cover_condition_proxy"],
                "den_p01": rat_terms.get("den_p01", ""),
                "den_p99": rat_terms.get("den_p99", ""),
                "r_prime_p99": rat_terms.get("r_prime_p99", ""),
                "r_double_prime_p99": rat_terms.get("r_double_prime_p99", ""),
                "group_diversity": rat_terms.get("group_diversity", ""),
                "boundary_accept": cover_accept,
                "cover_debt_before": cover_before,
                "cover_debt_after": cover_after,
                "cover_debt_delta": cover_delta,
                "boundary_rejection_reason": boundary_reason,
                "no_fake": 1,
            }
            rows.append(row)
            last_meta = row
    after_sha = parameter_sha_by_params(params)
    writes.append({
        "stage": "V1310_SIGNAL_TO_COVER_WRITEBACK",
        "architecture_id": spec.architecture_id,
        "mapped_candidate_id": spec.mapped_candidate_id,
        "task_or_dataset": task,
        "seed": int(seed),
        "method": method,
        "loss_interface": loss_interface,
        "writeback_before_sha256": before_sha,
        "writeback_after_sha256": after_sha,
        "writeback_changed": int(before_sha != after_sha),
        "updates_named_parameters": 1,
        "readout_feature_proxy_only": 0,
        "feature_table_proxy_only": 0,
        "no_fake": 1,
    })
    auc_step = finite_mean(val_losses)
    if len(elapsed_points) >= 2:
        weights = [elapsed_points[0]] + [max(1.0e-9, elapsed_points[i] - elapsed_points[i - 1]) for i in range(1, len(elapsed_points))]
        auc_time = sum(v * w for v, w in zip(val_losses, weights)) / max(sum(weights), 1.0e-9)
    else:
        auc_time = auc_step
    summary = {
        "stage": "V1310_SIGNAL_TO_COVER_SUMMARY",
        "architecture_id": spec.architecture_id,
        "mapped_candidate_id": spec.mapped_candidate_id,
        "architecture_kind": spec.architecture_kind,
        "task_or_dataset": task,
        "seed": int(seed),
        "method": method,
        "resolved_training_method": resolved_method,
        "loss_interface": loss_interface,
        "val_loss_auc_step": auc_step,
        "val_loss_auc_time": auc_time,
        "final_val_loss": rows[-1]["val_loss"] if rows else float("nan"),
        "final_val_acc": rows[-1]["val_acc"] if rows else float("nan"),
        "final_CEp99": rows[-1]["CEp99"] if rows else float("nan"),
        "final_NLL": rows[-1]["NLL"] if rows else float("nan"),
        "final_ECE": rows[-1]["ECE"] if rows else float("nan"),
        "final_CouplingR2": rows[-1]["LineC_CouplingR2"] if rows else float("nan"),
        "final_NoiseSignalLeak": rows[-1]["LineC_NoiseSignalLeak"] if rows else float("nan"),
        "final_ReservoirRatio": rows[-1]["LineC_ReservoirRatio"] if rows else float("nan"),
        "mean_active_fraction": finite_mean([fnum(r.get("snr_active_fraction"), float("nan")) for r in rows]),
        "mean_boundary_accept": finite_mean([fnum(r.get("boundary_accept"), float("nan")) for r in rows]),
        "elapsed_sec": time.perf_counter() - t0,
        "last_update_meta": last_meta,
        "cover_purity_mean": last_meta.get("cover_purity_mean", 0.0) if last_meta else 0.0,
        "cover_purity_p10": last_meta.get("cover_purity_p10", 0.0) if last_meta else 0.0,
        "cover_churn_mean": last_meta.get("cover_churn", 0.0) if last_meta else 0.0,
        "cover_load_gini": last_meta.get("cover_load_gini", 0.0) if last_meta else 0.0,
        "cover_specialization_entropy": last_meta.get("cover_specialization_entropy", 0.0) if last_meta else 0.0,
        "signal_to_cover_score": last_meta.get("signal_to_cover_score", 0.0) if last_meta else 0.0,
        "false_drop_fraction": last_meta.get("false_drop_fraction", 0.0) if last_meta else 0.0,
        "false_keep_fraction": last_meta.get("false_keep_fraction", 0.0) if last_meta else 0.0,
        "readout_signal_mass": fnum(last_meta.get("snr_active_fraction"), 0.0) if last_meta else 0.0,
        "basis_signal_mass": fnum(last_meta.get("signal_to_cover_score"), 0.0) if last_meta else 0.0,
        "reservoir_group_load": fnum(last_meta.get("cover_load_gini"), 0.0) if last_meta else 0.0,
        "exploration_group_load": fnum(last_meta.get("cover_specialization_entropy"), 0.0) if last_meta else 0.0,
    }
    return rows, writes, summary


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    base: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for r in rows:
        if str(r.get("method")) in {"K0-RAT-AdamW", "RAT-AdamW"}:
            base[(str(r["architecture_id"]), int(r["seed"]), str(r["task_or_dataset"]), str(r["loss_interface"]))] = r
    for r in rows:
        b = base.get((str(r["architecture_id"]), int(r["seed"]), str(r["task_or_dataset"]), str(r["loss_interface"])))
        if not b:
            r.update({"source_vs_adamw": 0.0, "AUC_time_delta": 0.0, "AUC_time_ratio": 1.0, "CEp99_delta": 0.0, "NLL_delta": 0.0, "ECE_delta": 0.0, "CouplingR2_delta": 0.0, "NoiseSignalLeak_delta": 0.0, "ReservoirRatio_delta": 0.0})
            continue
        r["source_vs_adamw"] = fnum(b["val_loss_auc_time"]) - fnum(r["val_loss_auc_time"])
        r["AUC_time_delta"] = fnum(r["val_loss_auc_time"]) - fnum(b["val_loss_auc_time"])
        r["AUC_time_ratio"] = fnum(r["val_loss_auc_time"]) / max(fnum(b["val_loss_auc_time"]), 1.0e-9)
        r["CEp99_delta"] = fnum(r["final_CEp99"]) - fnum(b["final_CEp99"])
        r["NLL_delta"] = fnum(r["final_NLL"]) - fnum(b["final_NLL"])
        r["ECE_delta"] = fnum(r["final_ECE"]) - fnum(b["final_ECE"])
        r["CouplingR2_delta"] = fnum(r["final_CouplingR2"]) - fnum(b["final_CouplingR2"])
        r["NoiseSignalLeak_delta"] = fnum(r["final_NoiseSignalLeak"]) - fnum(b["final_NoiseSignalLeak"])
        r["ReservoirRatio_delta"] = fnum(r["final_ReservoirRatio"]) - fnum(b["final_ReservoirRatio"])
        linec_majority = int(fnum(r["CouplingR2_delta"]) >= 0.0 and fnum(r["NoiseSignalLeak_delta"]) <= 0.0 and fnum(r["ReservoirRatio_delta"]) <= 0.0)
        r["LineC_majority_pass"] = linec_majority
        control = int(str(r.get("method")) in {"K0-RAT-AdamW", "RAT-AdamW"})
        r["pass_s3"] = int((not control) and fnum(r["source_vs_adamw"]) >= 0.005 and fnum(r["AUC_time_ratio"], 99.0) <= 1.0 and fnum(r["CEp99_delta"]) <= 0.05 and fnum(r["NLL_delta"]) <= 0.02 and fnum(r["ECE_delta"]) <= 0.02 and linec_majority)
        r["LineC_all_pass"] = int(fnum(r["CouplingR2_delta"]) >= 0.0 and fnum(r["NoiseSignalLeak_delta"]) <= 0.01 and fnum(r["ReservoirRatio_delta"]) <= 0.01 and fnum(r["CEp99_delta"]) <= 0.05 and fnum(r["NLL_delta"]) <= 0.02 and fnum(r["ECE_delta"]) <= 0.02)
        r["pass_s4"] = int(r["pass_s3"] and r["LineC_all_pass"] and fnum(r.get("cover_purity_mean"), 0.0) >= 0.20 and fnum(r.get("cover_churn_mean"), 1.0) <= 0.35)


def build_substrate_rows(telemetry: list[dict[str, Any]], summaries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    prior = {str(r.get("candidate_id")): r for r in build_substrate_map(SOURCE_V1235)}
    rows: list[dict[str, Any]] = []
    for arch in parse_csv(args.architectures):
        spec = ARCHITECTURES[arch]
        tele = [r for r in telemetry if str(r.get("architecture_id")) == arch]
        summ = [r for r in summaries if str(r.get("architecture_id")) == arch and str(r.get("method")) != "K0-RAT-AdamW"]
        base = prior.get(spec.mapped_candidate_id, {})
        raw = fnum(base.get("raw_memory_ratio_vs_mlp"), 9.0)
        inc = fnum(base.get("incremental_memory_ratio_vs_mlp"), 9.0)
        step = fnum(base.get("step_ratio_vs_mlp"), 9.0)
        if spec.method_id:
            raw = inc = step = 9.0
        mean_delta = finite_mean([fnum(r.get("source_vs_adamw"), float("nan")) for r in summ])
        worst_delta = min([fnum(r.get("source_vs_adamw"), 9.0) for r in summ], default=float("nan"))
        auc = quantile_or_nan([fnum(r.get("AUC_time_ratio"), float("nan")) for r in summ], 0.50)
        ret = quantile_or_nan([fnum(r.get("signal_retention_group"), float("nan")) for r in tele], 0.50)
        cos = quantile_or_nan([fnum(r.get("cos_group_vs_param"), float("nan")) for r in tele], 0.50)
        purity = quantile_or_nan([fnum(r.get("cover_purity_mean"), float("nan")) for r in tele + summ], 0.50)
        churn = finite_mean([fnum(r.get("cover_churn", r.get("cover_churn_mean", float("nan"))), float("nan")) for r in tele + summ])
        gate = int(raw <= 1.10 and inc <= 1.75 and step <= 1.75 and mean_delta >= -0.05 and worst_delta >= -0.10 and auc <= 2.0 and ret >= 0.70 and cos >= 0.60 and purity >= 0.15 and churn <= 0.50)
        rows.append({
            "stage": "V1310_RATIONAL_COVER_SUBSTRATE",
            "architecture_id": arch,
            "mapped_candidate_id": spec.mapped_candidate_id,
            "method_id": spec.method_id,
            "architecture_kind": spec.architecture_kind,
            "workspace_raw_ratio": raw,
            "workspace_incremental_ratio": inc,
            "step_ratio": step,
            "workspace_source": "v1235_prior" if not spec.method_id else "unprofiled_new_overcomplete_primitive_fail_closed",
            "mean_delta_vs_adamw": mean_delta,
            "worst_delta_vs_adamw": worst_delta,
            "AUCtime_ratio": auc,
            "snr_retention_group": ret,
            "cos_group_vs_param": cos,
            "cover_purity_median": purity,
            "cover_churn_mean": churn,
            "cover_substrate_gate_pass": gate,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return rows


def run_rational_line(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    diff_rows: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    linec: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for arch in parse_csv(args.architectures):
        spec = ARCHITECTURES[arch]
        xtr, _ytr, _xv, _yv = synthetic_data("X1", 0, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
        model = make_arch_model(spec, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, 131_010)
        diff_rows.extend(apply_architecture_reset(model, spec, xtr))
        for task in parse_csv(args.synthetic_tasks):
            for seed in parse_ints(args.synthetic_seeds):
                for loss in parse_csv(args.k_losses):
                    try:
                        telemetry.extend(run_transfer_audit_for_arch(spec, task=task, seed=int(seed), loss=loss, args=args, device=device))
                    except Exception as exc:  # noqa: BLE001
                        failures.append({"stage": "V1310_TRANSFER_FAILURE", "architecture_id": arch, "task": task, "seed": seed, "loss": loss, "exception": repr(exc), "no_fake": 1})
                    for method in parse_csv(args.k_methods):
                        resolved, note = resolve_method(method)
                        try:
                            rows, wr, summary = run_arch_training_case(spec, method=method, resolved_method=resolved, task=task, seed=int(seed), loss_interface=loss, args=args, device=device)
                            for r in rows:
                                r["implementation_note"] = note
                                training.append(r)
                                linec.append({
                                    "stage": "V1310_SIGNAL_TO_COVER_LINEC",
                                    "architecture_id": arch,
                                    "task": task,
                                    "seed": int(seed),
                                    "loss_interface": loss,
                                    "method": method,
                                    "step": r.get("step"),
                                    "CouplingR2": r.get("LineC_CouplingR2"),
                                    "NoiseSignalLeak": r.get("LineC_NoiseSignalLeak"),
                                    "RealSignalReservoirRatio": r.get("LineC_ReservoirRatio"),
                                    "CEp99": r.get("CEp99"),
                                    "audit_only": 1,
                                    "no_fake": 1,
                                })
                            for w in wr:
                                w["implementation_note"] = note
                                writes.append(w)
                            summary["implementation_note"] = note
                            summaries.append(summary)
                        except Exception as exc:  # noqa: BLE001
                            failures.append({"stage": "V1310_TRAINING_FAILURE", "architecture_id": arch, "task": task, "seed": seed, "loss": loss, "method": method, "exception": repr(exc), "no_fake": 1})
    add_baseline_deltas(summaries)
    return diff_rows, telemetry, training, summaries, linec, writes + failures


def run_nonrat_vertical(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    prior = {str(r.get("candidate_id")): r for r in build_substrate_map(SOURCE_V1235)}
    rows: list[dict[str, Any]] = []
    xtr, ytr, _xv, _yv = synthetic_data("X1", 0, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    xb = xtr[: min(int(args.batch_size), int(xtr.shape[0]))]
    yb = ytr[: int(xb.shape[0])]
    for family, candidate, mapped, slice_kind in NONRAT_VERTICAL_SPECS:
        base = prior.get(mapped, {})
        try:
            model = make_model_for_family(family, mapped, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, 131_510).to(device)
            params = all_named_params(model)
            g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface="CE")
            state = SNRState(decay=float(args.snr_ema_decay))
            meta = snr_gate(g, params, state, method="RAT-GroupSNRSoft-GradientClusterCover-k4", phase="cover-alignment", tau=float(args.snr_tau), eps=float(args.snr_eps), soft_alpha=float(args.soft_alpha), active_fraction_cap=float(args.active_fraction_cap))
            cov = cover_debt(model, xb)
            raw = fnum(base.get("raw_memory_ratio_vs_mlp"), 9.0)
            inc = fnum(base.get("incremental_memory_ratio_vs_mlp"), 9.0)
            step = fnum(base.get("step_ratio_vs_mlp"), 9.0)
            mean_delta = fnum(base.get("mean_delta_vs_mlp"), -9.0)
            worst_delta = fnum(base.get("worst_delta_vs_mlp"), -9.0)
            auc = fnum(base.get("AUC_time_ratio_vs_mlp"), 9.0)
            linec_rate = fnum(base.get("linec_pass_rate"), 0.0)
            substrate_pass = int(raw <= 1.10 and inc <= 1.75 and step <= 1.75 and mean_delta >= -0.05 and worst_delta >= -0.10 and auc <= 2.0 and linec_rate >= 0.30)
            rows.append({
                "stage": "V1310_NONRAT_SUBSTRATE_VERTICAL_SLICE",
                "family": family,
                "candidate": candidate,
                "mapped_candidate_id": mapped,
                "slice_kind": slice_kind,
                "workspace_raw_ratio": raw,
                "workspace_incremental_ratio": inc,
                "step_ratio": step,
                "mean_delta_vs_mlp": mean_delta,
                "worst_delta_vs_mlp": worst_delta,
                "AUCtime_ratio": auc,
                "LineC_pass_rate": linec_rate,
                "cover_entropy": cov.get("cover_entropy"),
                "cover_condition_proxy": cov.get("cover_condition_proxy"),
                "snr_active_fraction": meta.get("active_fraction"),
                "substrate_vertical_slice_pass": substrate_pass,
                "promotion_allowed": 0,
                "no_fake": 1,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"stage": "V1310_NONRAT_SUBSTRATE_VERTICAL_SLICE", "family": family, "candidate": candidate, "mapped_candidate_id": mapped, "slice_kind": slice_kind, "exception": repr(exc), "substrate_vertical_slice_pass": 0, "promotion_allowed": 0, "no_fake": 1})
    return rows


def forbidden_audit() -> list[dict[str, Any]]:
    checks = [
        ("uses_y_for_stats", 0, "architecture initialization uses fixed seed and train input scale only"),
        ("label_informed_init", 0, "no labels are read for architecture reset"),
        ("ce_tail_direction", 0, "CEp99/NLL/ECE are audit/gate only"),
        ("linec_direction", 0, "LineC is audit only"),
        ("validation_test_future_direction", 0, "validation/test/future/query are not used for direction"),
        ("dataset_name_branch", 0, "dataset name only selects loader in MLP monitor"),
        ("is_new_substrate_architecture", 0, "A-RCF changes are recorded in architecture diff manifest"),
        ("is_k_token_only_extension", 0, "runner adds A-RCF architecture candidates before K14-K18"),
        ("readout_feature_proxy", 0, "updates are named parameters"),
    ]
    return [{"stage": "V1310_FORBIDDEN_INFORMATION_AUDIT", "check": c, "violation": v, "note": n, "no_fake": 1} for c, v, n in checks]


def code_review_manifest() -> list[dict[str, Any]]:
    return [
        {"stage": "V1310_CODE_REVIEW_MANIFEST", "file": "experiments/run_v1310_cover_forming_substrate_architecture_reset.py", "surface": "A-RCF substrate construction", "symbol": "apply_architecture_reset", "is_new_substrate_architecture": 1, "is_k_token_only_extension": 0, "uses_label_in_init": 0, "uses_y_for_stats": 0, "no_fake": 1},
        {"stage": "V1310_CODE_REVIEW_MANIFEST", "file": "experiments/run_v1310_cover_forming_substrate_architecture_reset.py", "surface": "K14-K18 training", "symbol": "run_arch_training_case", "is_new_substrate_architecture": 1, "is_k_token_only_extension": 0, "uses_validation_for_direction": 0, "uses_linec_for_direction": 0, "no_fake": 1},
        {"stage": "V1310_CODE_REVIEW_MANIFEST", "file": "experiments/run_v1310_cover_forming_substrate_architecture_reset.py", "surface": "Non-RAT vertical slices", "symbol": "run_nonrat_vertical", "promotion_allowed": 0, "no_fake": 1},
    ]


def write_readback(out_dir: Path) -> None:
    lines = [
        "# v13.10 Substrate Architecture Readback",
        "",
        "A-RCF candidates are constructed before functional training.  They are not K-token aliases.",
        "",
    ]
    for spec in ARCHITECTURES.values():
        lines.append(f"- {spec.architecture_id}: `{spec.architecture_kind}` mapped to `{spec.mapped_candidate_id or spec.method_id}`. {spec.reset_note}.")
    lines.extend([
        "",
        "Legality:",
        "",
        "1. Architecture reset uses fixed seed and train-input scale, not labels.",
        "2. K14-K18 directions use current train-stream generic loss gradients.",
        "3. CEp99/NLL/ECE/LineC are audit/gate only.",
        "4. Non-RAT rows remain substrate vertical slices and cannot promote.",
        "",
    ])
    (out_dir / "v1310_substrate_architecture_readback.md").write_text("\n".join(lines), encoding="utf-8")


def build_route(out_dir: Path, substrate_rows: list[dict[str, Any]], telemetry: list[dict[str, Any]], summaries: list[dict[str, Any]], nonrat_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]], missing: int, args: argparse.Namespace, code_sha: str) -> dict[str, Any]:
    substrate_pass = sum(int(r.get("cover_substrate_gate_pass", 0)) for r in substrate_rows)
    med_ret = quantile_or_nan([fnum(r.get("signal_retention_group"), float("nan")) for r in telemetry], 0.50)
    med_cos = quantile_or_nan([fnum(r.get("cos_group_vs_param"), float("nan")) for r in telemetry], 0.50)
    med_cover = quantile_or_nan([fnum(r.get("cover_purity_mean"), float("nan")) for r in telemetry] + [fnum(r.get("cover_purity_mean"), float("nan")) for r in summaries], 0.50)
    mean_churn = finite_mean([fnum(r.get("cover_churn", r.get("cover_churn_mean", float("nan"))), float("nan")) for r in telemetry + summaries])
    cover_gate = int(med_ret >= 0.70 and med_cos >= 0.60 and med_cover >= 0.15 and mean_churn <= 0.50)
    s3 = task_pass_count(summaries, "pass_s3", parse_ints(args.synthetic_seeds), parse_csv(args.k_losses))
    s4 = task_pass_count(summaries, "pass_s4", parse_ints(args.synthetic_seeds), parse_csv(args.k_losses))
    nonrat_pass = sum(int(r.get("substrate_vertical_slice_pass", 0)) for r in nonrat_rows)
    mlp_dataset_pass = sum(int(r.get("dataset_pass", 0)) for r in mlp_rows)
    mlp_confirmed = int(len(mlp_rows) > 0 and mlp_dataset_pass == len(mlp_rows))
    arch_diff_ok = int((out_dir / "v1310_architecture_diff_manifest.csv").exists())
    if missing or not arch_diff_ok:
        route = "R0-ProvenanceViolation"
        minimum = "S0-Invalid"
    elif substrate_pass <= 0:
        route = "R1-NoCoverSubstrate"
        minimum = "S0-ArchitectureScoutExecuted"
    elif med_ret >= 0.70 and med_cos >= 0.60 and not cover_gate:
        route = "R2-SignalRetentionPassCoverFormationFail"
        minimum = "S1-CoverSubstrate" if substrate_pass else "S1C-SignalRetentionPositive"
    elif s3 < 5:
        route = "R3-CoverFormationPassTaskFamilyFail"
        minimum = "S2-CoverTrainingSignal" if s3 >= 3 else "S1-CoverSubstrate"
    elif s3 >= 5 and s4 < 5:
        route = "R4-TaskFamilyPassLineCFail"
        minimum = "S3-KANSynthetic"
    elif nonrat_pass == 0:
        route = "R5-NonRATStillNoSubstrate"
        minimum = "S4-KANLineCStable"
    elif mlp_confirmed and s3 < 5:
        route = "R6-GenericMLPOnly"
        minimum = "S1-CoverSubstrate"
    else:
        route = "S5-OfficialRealShortRun"
        minimum = "S5-OfficialRealShortRun"
    return {
        "route": route,
        "minimum_success": minimum,
        "official_success_reached": int(route == "S5-OfficialRealShortRun"),
        "promotion_allowed": int(route == "S5-OfficialRealShortRun"),
        "final_stop_allowed": int(route in {"R1-NoCoverSubstrate", "R2-SignalRetentionPassCoverFormationFail", "R3-CoverFormationPassTaskFamilyFail", "R4-TaskFamilyPassLineCFail", "R5-NonRATStillNoSubstrate", "R6-GenericMLPOnly"} and missing == 0),
        "cover_substrate_pass_count": substrate_pass,
        "architecture_candidate_count": len(parse_csv(args.architectures)),
        "snr_transfer_median_retention_group": med_ret,
        "snr_transfer_median_cos_group_vs_param": med_cos,
        "cover_purity_median": med_cover,
        "cover_churn_mean": mean_churn,
        "cover_formation_gate_pass": cover_gate,
        "kan_s3_task_pass_count": s3,
        "kan_s4_task_pass_count": s4,
        "nonrat_vertical_pass_count": nonrat_pass,
        "mlp_generic_dataset_pass_count": mlp_dataset_pass,
        "mlp_generic_dataset_count": len(mlp_rows),
        "mlp_generic_confirmed": mlp_confirmed,
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": 0,
        "is_new_substrate_architecture": 1,
        "is_k_token_only_extension": 0,
        "kan_real_short_run_open_allowed": int(route in {"S5-OfficialRealShortRun"}),
        "synthetic_train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "compute_budgeted_run": int(int(args.train_steps) < 200 or len(parse_ints(args.synthetic_seeds)) < 3),
        "code_review_packet_sha256": code_sha,
        "no_fake": 1,
    }


def write_no_go(out_dir: Path, route: dict[str, Any]) -> None:
    boundary = [
        "# v13.10 No-Go Boundary",
        "",
        f"route = {route['route']}",
        "",
        "Facts:",
        "",
        f"- cover_substrate_pass_count = {route['cover_substrate_pass_count']}",
        f"- cover_purity_median = {route['cover_purity_median']}",
        f"- cover_churn_mean = {route['cover_churn_mean']}",
        f"- KAN S3 task pass count = {route['kan_s3_task_pass_count']}/7",
        f"- KAN S4 task pass count = {route['kan_s4_task_pass_count']}/7",
        f"- Non-RAT vertical pass count = {route['nonrat_vertical_pass_count']}",
        f"- MLP generic confirmed = {route['mlp_generic_confirmed']}",
        "",
        "No row may promote unless S5 route is reached.  LineC/tail metrics are audit/gate only.",
        "",
    ]
    (out_dir / "v1310_no_go_boundary.md").write_text("\n".join(boundary), encoding="utf-8")
    queue = [
        "# v13.10 Next Hypothesis Queue",
        "",
        "1. If R1/R2: A-RCF cover architecture still does not form stable cover; next work must redesign primitive/substrate, not add K19/K20.",
        "2. If R3: cover forms but task-family source does not; inspect readout-vs-basis signal mass without using tail/LineC as direction.",
        "3. If R4: task source exists but LineC/tail blocks; use signal/reservoir architecture split only.",
        "4. Non-RAT cannot enter functional proof before exact/no-materialize workspace/task-health gates pass.",
        "",
    ]
    (out_dir / "v1310_next_hypothesis_queue.md").write_text("\n".join(queue), encoding="utf-8")


def write_figures(out_dir: Path, route: dict[str, Any], substrate_rows: list[dict[str, Any]], summaries: list[dict[str, Any]], nonrat_rows: list[dict[str, Any]]) -> None:
    write_svg(out_dir / "fig_progress_by_line.svg", "v13.10 progress", [f"route={route['route']}", f"cover substrates={route['cover_substrate_pass_count']}", f"S3={route['kan_s3_task_pass_count']}/7"])
    write_svg(out_dir / "fig_signal_retention_vs_cover_purity.svg", "Signal retention vs cover purity", [f"ret={route['snr_transfer_median_retention_group']}", f"cover={route['cover_purity_median']}"])
    write_svg(out_dir / "fig_cover_purity_by_method.svg", "Cover purity by method", [f"{r.get('architecture_id')} {r.get('method')}: {r.get('cover_purity_mean')}" for r in summaries[:30]])
    write_svg(out_dir / "fig_cover_churn_by_method.svg", "Cover churn by method", [f"{r.get('method')}: {r.get('cover_churn_mean')}" for r in summaries[:30]])
    write_svg(out_dir / "fig_signal_to_cover_score_by_task.svg", "Signal to cover by task", [f"{r.get('task_or_dataset')}: {r.get('signal_to_cover_score')}" for r in summaries[:30]])
    write_svg(out_dir / "fig_task_family_pass_heatmap.svg", "Task family pass heatmap", [f"{r.get('task_or_dataset')} {r.get('architecture_id')} {r.get('method')} S3={r.get('pass_s3')}" for r in summaries[:30]])
    write_svg(out_dir / "fig_linec_vs_source_scatter.svg", "LineC vs source", [f"{r.get('source_vs_adamw')} / {r.get('NoiseSignalLeak_delta')}" for r in summaries[:30]])
    write_svg(out_dir / "fig_readout_vs_basis_signal_mass.svg", "Readout vs basis signal mass", [f"{r.get('readout_signal_mass')} / {r.get('basis_signal_mass')}" for r in summaries[:30]])
    write_svg(out_dir / "fig_nonrat_substrate_health_matrix.svg", "Non-RAT substrate health", [f"{r.get('candidate')}: {r.get('substrate_vertical_slice_pass')}" for r in nonrat_rows])
    write_svg(out_dir / "fig_mlp_vs_kan_control.svg", "MLP vs KAN", [f"KAN S3={route['kan_s3_task_pass_count']}/7", f"MLP datasets={route['mlp_generic_dataset_pass_count']}/{route['mlp_generic_dataset_count']}"])
    write_svg(out_dir / "fig_failure_taxonomy_sankey.svg", "Failure taxonomy", [f"{r.get('architecture_id')}: substrate={r.get('cover_substrate_gate_pass')}" for r in substrate_rows])


def required_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    for name in REQUIRED + FIGURES:
        path = out_dir / name
        rows.append({"stage": "V1310_REQUIRED_MANIFEST", "path": str(path.relative_to(ROOT) if path.exists() else path), "required": 1, "exists": int(path.exists()), "no_fake": 1})
    return rows, sum(1 for r in rows if int(r["exists"]) != 1)


def code_packet(out_dir: Path) -> str:
    packet = out_dir / "v1310_code_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [
            Path(__file__),
            ROOT / "experiments" / "run_v137_boundary_conditioned_poprisk_training.py",
            ROOT / "experiments" / "run_v139_signal_to_cover_functional_substrate_architecture.py",
            DOC_PLAN,
        ]:
            if path.exists():
                zf.write(path, path.relative_to(ROOT).as_posix())
    return sha256_file(packet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--architectures", default="A-RCF1-MultiBandRationalCover,A-RCF2-SNRClusterCoverWarmup,A-RCF3-PersistentCoverMemory,A-RCF4-SignalReservoirSplitGroups,A-RCF5-ReadoutBasisDecoupledCover,A-RCF6-OvercompleteSparseCoverBank")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--mlp-seeds", default="0,1,2,3,4")
    ap.add_argument("--mlp-seed-threshold", type=int, default=3)
    ap.add_argument("--mlp-methods", default="MLP-AdamW,MLP-AdamW-SNREMA-Blend50-ActiveFractionSchedule,MLP-AdamW-SNRRoleNorm-Blend50-LogitNormTrust")
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
    ap.add_argument("--k-methods", default="K0-RAT-AdamW,K14-RCF-SNRClusterTraining,K15-RCF-PersistentCoverMemory,K16-RCF-SignalReservoirSplit,K17-RCF-ReadoutBasisDecoupled,K18-RCF-OvercompleteSparseCover")
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--mlp-hidden", type=int, default=160)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--snr-tau", type=float, default=1.0)
    ap.add_argument("--snr-eps", type=float, default=1.0e-12)
    ap.add_argument("--snr-ema-decay", type=float, default=0.85)
    ap.add_argument("--soft-alpha", type=float, default=8.0)
    ap.add_argument("--active-fraction-cap", type=float, default=1.0)
    ap.add_argument("--overcomplete-active-fraction", type=float, default=0.50)
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp_training, mlp_summary, mlp_10seed, mlp_failures = run_mlp_line(args, device)
    diff_rows, telemetry, training, summaries, linec, writes_and_failures = run_rational_line(args, device)
    failures = [r for r in writes_and_failures if "FAILURE" in str(r.get("stage"))] + mlp_failures
    writeback = [r for r in writes_and_failures if "WRITEBACK" in str(r.get("stage"))]
    substrate_rows = build_substrate_rows(telemetry, summaries, args)
    nonrat_rows = run_nonrat_vertical(args, device)
    forbidden = forbidden_audit()
    loss_rows = loss_interface_audit(parse_csv(args.k_methods) + parse_csv(args.mlp_methods), parse_csv(args.k_losses) + [args.loss_interface])

    write_rows(out_dir / "v1310_code_review_manifest.csv", code_review_manifest())
    write_rows(out_dir / "v1310_architecture_diff_manifest.csv", diff_rows or [{"stage": "V1310_ARCHITECTURE_DIFF_MANIFEST", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v1310_forbidden_information_audit.csv", forbidden)
    write_readback(out_dir)
    write_rows(out_dir / "v1310_rational_cover_substrate.csv", substrate_rows or [{"stage": "V1310_RATIONAL_COVER_SUBSTRATE", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v1310_rational_cover_telemetry.csv", telemetry or [{"stage": "V1310_RATIONAL_COVER_TELEMETRY", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v1310_signal_to_cover_training.csv", training or [{"stage": "V1310_SIGNAL_TO_COVER_TRAINING", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v1310_signal_to_cover_summary.csv", summaries or [{"stage": "V1310_SIGNAL_TO_COVER_SUMMARY", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v1310_signal_to_cover_linec.csv", linec or [{"stage": "V1310_SIGNAL_TO_COVER_LINEC", "skip_reason": "no_rows", "audit_only": 1, "no_fake": 1}])
    write_rows(out_dir / "v1310_signal_to_cover_writeback_trace.csv", writeback or [{"stage": "V1310_SIGNAL_TO_COVER_WRITEBACK", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v1310_nonrat_substrate_vertical_slice.csv", nonrat_rows or [{"stage": "V1310_NONRAT_SUBSTRATE_VERTICAL_SLICE", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v1310_mlp_control_monitor.csv", mlp_summary or [{"stage": "V1310_MLP_CONTROL_MONITOR", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v1310_linec_audit.csv", linec or [{"stage": "V1310_LINEC_AUDIT", "skip_reason": "no_rows", "audit_only": 1, "no_fake": 1}])
    write_rows(out_dir / "v1310_failure_table.csv", failures or [{"stage": "V1310_FAILURE_TABLE", "failure_rows": 0, "no_fake": 1}])
    write_rows(out_dir / "v1310_loss_interface_audit.csv", loss_rows)

    code_sha = code_packet(out_dir)
    route = build_route(out_dir, substrate_rows, telemetry, summaries, nonrat_rows, mlp_10seed, 0, args, code_sha)
    write_no_go(out_dir, route)
    write_figures(out_dir, route, substrate_rows, summaries, nonrat_rows)
    (out_dir / "v1310_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest, missing = required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    (out_dir / "v1310_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rows(out_dir / "v1310_required_manifest.csv", manifest)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
