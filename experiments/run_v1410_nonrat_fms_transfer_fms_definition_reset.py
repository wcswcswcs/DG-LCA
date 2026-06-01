#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
import time
import zipfile
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v133_task_family_robust_basis_natural import eval_metrics, linec_proxy, synthetic_data
from experiments.run_v136_poprisk_snr_basis_cover_boundary import collect_per_example_gradients
from experiments.run_v142_functional_first_all_basis_parallel import (
    FMSState,
    ParamSpec,
    fnum,
    loss_value,
    make_case_model,
    named_param_specs,
    param_role,
    read_rows,
    sha256_file,
    sint,
    write_json,
    write_rows,
    write_svg,
)
from experiments import run_v144_real_transfer_fms_all_basis_substrate as v144
from experiments.run_v1231_basis_kernel_workspace import linec_metrics


ROOT = Path("results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel")
PLAN_PATH = Path("docs/DG-KAN_v14.10_NonRAT_FMS_Transfer_FMSDefinitionReset_完整计划.md")
V149_SUBSTRATE_DIR = Path("results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix_linec3")

REQUIRED = [
    "v1410_route_decision.json",
    "v1410_progress_table.csv",
    "v1410_forbidden_information_audit.csv",
    "v1410_no_action_search_audit.csv",
    "v1410_dche_fms_synthetic_results.csv",
    "v1410_dche_fms_synthetic_summary.csv",
    "v1410_dche_fms_real_results.csv",
    "v1410_dche_fms_real_summary.csv",
    "v1410_dche_fms_controls.csv",
    "v1410_dche_degree_telemetry.csv",
    "v1410_dche_projection_retention.csv",
    "v1410_mlp_generic_controls.csv",
    "v1410_all_basis_substrate_status.csv",
    "v1410_fourier_substrate_hardening.csv",
    "v1410_rbf_substrate_hardening.csv",
    "v1410_wavelet_substrate_hardening.csv",
    "v1410_rational_monitor.csv",
    "v1410_linec_tail_audit.csv",
    "v1410_failure_taxonomy.csv",
    "v1410_required_artifact_manifest.csv",
    "v1410_code_review_packet.zip",
    "v1410_no_go_boundary.md",
    "v1410_next_hypothesis_queue.md",
]

FIGURES = [
    "fig_v1410_progress_by_line.svg",
    "fig_dche_synthetic_5of7_heatmap.svg",
    "fig_dche_real_3x3_pass_matrix.svg",
    "fig_dche_source_auc_tail_scatter.svg",
    "fig_dche_degree_energy_before_after.svg",
    "fig_dche_projection_value_retention.svg",
    "fig_dche_fms_vs_controls_diffindiff.svg",
    "fig_mlp_vs_dche_fms_gain.svg",
    "fig_all_basis_substrate_status.svg",
    "fig_fourier_rbf_wavelet_substrate_heatmap.svg",
    "fig_linec_tail_audit_by_method.svg",
    "fig_failure_taxonomy.svg",
]

DEFAULT_D_CHE_CANDIDATE = "D-CHE17-HighDegreeLateEnableSubstrate"

CONTROL_METHODS = {
    "C0-D-CHE-AdamW",
    "C1-D-CHE-AdamW-NoOpMatchedOverhead",
    "C2-D-CHE-AdamW-RandomMatchedNorm",
    "C3-D-CHE-AdamW-AdamWParallelDirectionControl",
    "C4-D-CHE-AdamW-GenericOptimizerStateControl",
    "C5-D-CHE-AdamW-SameActiveFractionControl",
}

F_CHE_METHODS = {
    "F-CHE1-GenericParameterFMS-NoBasisProjection",
    "F-CHE2-DegreeWiseFMS",
    "F-CHE3-DegreeEnergyTrustRegionFMS",
    "F-CHE4-LowDegreeAnchorHighDegreeLateEnableFMS",
    "F-CHE5-ReadoutDegreeDecoupledFMS",
    "F-CHE6-PhaseScheduleDegreeFMS",
    "F-CHE7-ValuePreservingDegreeProjectionFMS",
}

FALLBACK_METHODS = {
    "F-CHE-FB1-ValuePathOnly",
    "F-CHE-FB2-DegreeConstraintOnly",
    "F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection",
}

REAL_TRANSFER_METHODS = {
    "F-CHE-RT1-TrainSplitAgreement",
    "F-CHE-RT2-ValueRetentionTrust",
    "F-CHE-RT3-DegreeEnergySafetyProjection",
    "F-CHE-RT4-CompositeTransferTrust",
}

MLP_METHODS = {
    "M0-MLP-AdamW",
    "M1-MLP-GenericFMS",
    "M2-MLP-DegreeAnalogFMS",
    "M3-MLP-RandomMatchedNorm",
    "M4-MLP-GenericOptimizerStateControl",
    "M5-MLP-NoOpMatchedOverhead",
    "M6-MLP-SameActiveFractionControl",
    "M7-MLP-BoundaryOnly",
    "M8-MLP-MetricState-MS1",
    "M9-MLP-MetricState-MS2",
    "M10-MLP-SameMetricScaleRandomPermutation",
    "M11-MLP-AdamWParallelDirectionControl",
}

METRIC_STATE_METHODS = {
    "MS1-ParameterMetricState",
    "MS2-DegreeRoleMetricState",
    "MS3-BasisGroupMetricState",
    "M8-MLP-MetricState-MS1",
    "M9-MLP-MetricState-MS2",
}

METRIC_STATE_CONTROL_METHODS = {
    "C6-D-CHE-SameTCRandomDirection",
    "C7-D-CHE-SameMetricScaleRandomPermutation",
    "M10-MLP-SameMetricScaleRandomPermutation",
    "M11-MLP-AdamWParallelDirectionControl",
}

LINE_A_CONTROL_METHODS = {
    "A-D4-SameProjectionRejectionRandom",
    "A-D5-SameValueRetentionRandom",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def median(values: list[float]) -> float:
    vals = sorted(x for x in values if not math.isnan(x) and not math.isinf(x))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def flat_existing_grad(specs: list[ParamSpec]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for spec in specs:
        if spec.param.grad is None:
            chunks.append(torch.zeros(spec.end - spec.start, device=spec.param.device))
        else:
            chunks.append(spec.param.grad.detach().flatten())
    return torch.cat(chunks) if chunks else torch.zeros(0)


def assign_flat_grad(specs: list[ParamSpec], flat_grad: torch.Tensor) -> None:
    for spec in specs:
        spec.param.grad = flat_grad[spec.start : spec.end].view_as(spec.param).detach().clone()


def train_batch_probe_metrics(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor, loss_interface: str) -> dict[str, float]:
    with torch.no_grad():
        logits = model(xb)
        losses = []
        for i in range(int(xb.shape[0])):
            losses.append(float(loss_value(model(xb[i : i + 1]), yb[i : i + 1], loss_interface).item()))
        probs = torch.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[-1] > 1 else 0.0)
        entropy = -(probs * torch.log(probs.clamp_min(1.0e-12))).sum(dim=-1) / math.log(max(2, probs.shape[-1]))
        return {
            "loss_mean": sum(losses) / max(1, len(losses)),
            "loss_q95": float(torch.tensor(losses, device=xb.device).quantile(0.95).item()) if losses else 0.0,
            "margin_p10": float(margin.quantile(0.10).item()),
            "logit_rms": float(logits.float().pow(2).mean().sqrt().item()),
            "entropy_norm": float(entropy.mean().item()),
        }


def stateless_probe_update(model: torch.nn.Module, flat_grad: torch.Tensor, lr: float, weight_decay: float) -> None:
    specs = named_param_specs(model)
    assign_flat_grad(specs, flat_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt.step()
    opt.zero_grad(set_to_none=True)


def fms_key(spec: ParamSpec, method: str, family: str) -> str:
    if method in LINE_A_CONTROL_METHODS:
        return spec.layer
    if method in {"MS1-ParameterMetricState", "C6-D-CHE-SameTCRandomDirection"}:
        return spec.name
    if method in {"MS2-DegreeRoleMetricState", "M9-MLP-MetricState-MS2"}:
        return f"{family}:{spec.role}:{degree_index(spec.name, spec.param)}"
    if method in {"MS3-BasisGroupMetricState"}:
        return f"{family}:basis-group" if spec.role == "basis" else f"{family}:{spec.role}"
    if method in {"C7-D-CHE-SameMetricScaleRandomPermutation", "M8-MLP-MetricState-MS1", "M10-MLP-SameMetricScaleRandomPermutation"}:
        return spec.layer
    if method in {"F-CHE1-GenericParameterFMS-NoBasisProjection", "F-CHE-FB1-ValuePathOnly", "F-CHE-RT1-TrainSplitAgreement"}:
        return spec.name
    if method in {"F-CHE2-DegreeWiseFMS", "F-CHE3-DegreeEnergyTrustRegionFMS", "F-CHE4-LowDegreeAnchorHighDegreeLateEnableFMS", "F-CHE6-PhaseScheduleDegreeFMS", "F-CHE7-ValuePreservingDegreeProjectionFMS", "F-CHE-FB2-DegreeConstraintOnly", "F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection", "F-CHE-RT2-ValueRetentionTrust", "F-CHE-RT3-DegreeEnergySafetyProjection", "F-CHE-RT4-CompositeTransferTrust"}:
        return f"{family}:{spec.role}:{degree_index(spec.name, spec.param)}"
    if method == "F-CHE5-ReadoutDegreeDecoupledFMS":
        return f"{family}:{spec.role}:{degree_index(spec.name, spec.param)}:{spec.layer}"
    if method == "M1-MLP-GenericFMS":
        return spec.layer
    if method == "M2-MLP-DegreeAnalogFMS":
        return spec.role
    if method in {
        "M3-MLP-RandomMatchedNorm",
        "M4-MLP-GenericOptimizerStateControl",
        "M5-MLP-NoOpMatchedOverhead",
        "M6-MLP-SameActiveFractionControl",
        "M7-MLP-BoundaryOnly",
        "C4-D-CHE-AdamW-GenericOptimizerStateControl",
        "C5-D-CHE-AdamW-SameActiveFractionControl",
    }:
        return spec.layer
    return spec.layer


def degree_index(name: str, param: torch.nn.Parameter) -> str:
    if param.ndim >= 3 and param.shape[-1] <= 8:
        return "degree_all"
    if "cheby" in name.lower() or "degree" in name.lower():
        return "degree_aux"
    return "degree_na"


def degree_energy(model: torch.nn.Module) -> dict[str, float]:
    energies: dict[str, float] = {}
    total = 0.0
    high = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            role = param_role(name)
            val = param.detach().float()
            if val.ndim >= 3 and val.shape[-1] <= 8:
                per_degree = val.square().mean(dim=tuple(range(val.ndim - 1)))
                for i, e in enumerate(per_degree):
                    key = f"{role}:degree{i}"
                    energies[key] = energies.get(key, 0.0) + float(e.item())
                    total += float(e.item())
                    if i >= max(1, int(val.shape[-1]) // 2):
                        high += float(e.item())
            else:
                e = float(val.square().mean().item())
                key = f"{role}:nondegree"
                energies[key] = energies.get(key, 0.0) + e
                total += e
    probs = [v / max(1.0e-12, total) for v in energies.values()]
    entropy = -sum(p * math.log(max(1.0e-12, p)) for p in probs) / math.log(max(2, len(probs)))
    energies["degree_energy_total"] = total
    energies["high_degree_energy_fraction"] = high / max(1.0e-12, total)
    energies["degree_entropy"] = entropy
    return energies


def projection_stats(generic_flat: torch.Tensor, projected_flat: torch.Tensor) -> dict[str, float]:
    gnorm = float(generic_flat.norm().item())
    pnorm = float(projected_flat.norm().item())
    dot = float(torch.dot(generic_flat, projected_flat).item()) if generic_flat.numel() else 0.0
    cos = dot / max(1.0e-8, gnorm * pnorm)
    retention = dot / max(1.0e-8, gnorm * gnorm)
    rejection = float((generic_flat - projected_flat).norm().item()) / max(1.0e-8, gnorm)
    return {
        "generic_value_norm": gnorm,
        "projected_value_norm": pnorm,
        "cos_projected_vs_generic": cos,
        "value_retention_after_degree_projection": retention,
        "degree_projection_rejection_fraction": rejection,
    }


def collect_streaming_per_example_summary(
    model: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    params: list[tuple[str, torch.nn.Parameter]],
    specs: list[ParamSpec],
    keys: list[str],
    loss_interface: str,
    utility_mode: str = "snr_energy",
) -> tuple[torch.Tensor, dict[str, float]]:
    total_numel = sum(int(p.numel()) for _name, p in params)
    if total_numel == 0 or xb.shape[0] == 0:
        return torch.zeros(0, device=xb.device), {}
    sum_flat = torch.zeros(total_numel, device=xb.device)
    sumsq_flat = torch.zeros(total_numel, device=xb.device)
    for i in range(int(xb.shape[0])):
        model.zero_grad(set_to_none=True)
        loss = loss_value(model(xb[i : i + 1]), yb[i : i + 1], loss_interface)
        loss.backward()
        chunks: list[torch.Tensor] = []
        for _name, param in params:
            if param.grad is None:
                chunks.append(torch.zeros(param.numel(), device=xb.device))
            else:
                chunks.append(param.grad.detach().flatten())
        flat = torch.cat(chunks) if chunks else torch.zeros(0, device=xb.device)
        sum_flat.add_(flat)
        sumsq_flat.add_(flat.square())
    model.zero_grad(set_to_none=True)
    n = float(xb.shape[0])
    mu = sum_flat / max(1.0, n)
    var = (sumsq_flat / max(1.0, n) - mu.square()).clamp_min(0.0)
    utilities: dict[str, float] = {}
    counts: dict[str, int] = {}
    for spec, key in zip(specs, keys):
        m = mu[spec.start : spec.end]
        v = var[spec.start : spec.end]
        snr = float(m.norm().item() / (v.mean().sqrt().item() + 1.0e-8))
        energy = float(m.square().mean().item())
        diffusion = float(v.mean().item())
        if utility_mode == "population_metric_state":
            # Population-risk style state: reward drift and penalize diffusion.
            # This is train-stream only and never reads validation/test/audit metrics.
            denom = diffusion + 1.0e-8
            value = energy / denom - diffusion / max(1.0, float(xb.shape[0] - 1))
        else:
            value = math.log1p(max(0.0, snr)) + math.log1p(max(0.0, energy) * 1.0e3)
        utilities[key] = utilities.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1
    for key, count in counts.items():
        utilities[key] /= max(1, count)
    return mu.detach().clone(), utilities


def apply_scales_to_flat(
    method: str,
    specs: list[ParamSpec],
    keys: list[str],
    flat: torch.Tensor,
    key_scales: dict[str, float],
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, float]:
    out = flat.clone()
    phase = float(step + 1) / max(1.0, float(total_steps))
    active = 0
    total = 0
    for spec, key in zip(specs, keys):
        scale = float(key_scales.get(key, 1.0))
        role = spec.role
        is_degree_tensor = spec.param.ndim >= 3 and spec.param.shape[-1] <= 8
        if method in {"MS1-ParameterMetricState", "M8-MLP-MetricState-MS1", "C7-D-CHE-SameMetricScaleRandomPermutation", "M10-MLP-SameMetricScaleRandomPermutation"}:
            scale = max(0.75, min(1.25, scale))
        elif method in {"MS2-DegreeRoleMetricState", "M9-MLP-MetricState-MS2"}:
            if role == "basis" or is_degree_tensor:
                scale = max(0.85, min(1.15, scale))
            else:
                scale = max(0.90, min(1.10, scale))
        elif method == "MS3-BasisGroupMetricState":
            if role == "basis":
                scale = max(0.70, min(1.05, scale))
            else:
                scale = max(0.90, min(1.10, scale))
        elif method in {"F-CHE3-DegreeEnergyTrustRegionFMS", "F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection"}:
            if role == "basis":
                scale = max(0.70, min(1.15, scale))
            else:
                scale = max(0.80, min(1.20, scale))
        elif method == "F-CHE4-LowDegreeAnchorHighDegreeLateEnableFMS":
            if is_degree_tensor and phase < 0.50:
                scale = 0.85 + 0.15 * scale
            elif is_degree_tensor:
                scale = 0.95 + 0.05 * scale
        elif method == "F-CHE5-ReadoutDegreeDecoupledFMS":
            if role == "readout":
                scale = max(0.75, min(1.10, scale))
        elif method == "F-CHE6-PhaseScheduleDegreeFMS":
            scale = 1.0 + phase * (scale - 1.0)
        elif method == "F-CHE7-ValuePreservingDegreeProjectionFMS":
            scale = 1.0 + 0.50 * (scale - 1.0)
        elif method == "F-CHE-FB2-DegreeConstraintOnly":
            scale = 0.95 if is_degree_tensor else 1.0
        elif method in {"F-CHE-RT3-DegreeEnergySafetyProjection", "F-CHE-RT4-CompositeTransferTrust"}:
            if role == "basis" or is_degree_tensor:
                scale = max(0.92, min(1.05, scale))
            else:
                scale = max(0.90, min(1.10, scale))
        start, end = spec.start, spec.end
        out[start:end].mul_(scale)
        active += int(abs(scale - 1.0) > 0.01) * (end - start)
        total += end - start
    return out, active / max(1, total)


def random_direction_with_scales(
    specs: list[ParamSpec],
    keys: list[str],
    flat: torch.Tensor,
    key_scales: dict[str, float],
    gen: torch.Generator,
) -> tuple[torch.Tensor, float]:
    out = torch.zeros_like(flat)
    active = 0
    total = 0
    for spec, key in zip(specs, keys):
        start, end = spec.start, spec.end
        segment = flat[start:end]
        noise = torch.randn(segment.shape, generator=gen, device=segment.device, dtype=segment.dtype)
        noise_norm = noise.norm().clamp_min(1.0e-8)
        segment_norm = segment.norm()
        scale = max(0.75, min(1.25, float(key_scales.get(key, 1.0))))
        out[start:end] = noise / noise_norm * segment_norm * scale
        active += int(abs(scale - 1.0) > 0.01) * (end - start)
        total += end - start
    return out, active / max(1, total)


def random_unit_orthogonal(flat: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    if flat.numel() == 0:
        return flat.clone()
    noise = torch.randn(flat.shape, generator=gen, device=flat.device, dtype=flat.dtype)
    norm = flat.norm()
    if float(norm.item()) > 1.0e-8:
        noise = noise - torch.dot(noise, flat) / norm.square().clamp_min(1.0e-8) * flat
    return noise / noise.norm().clamp_min(1.0e-8)


def same_projection_rejection_random(generic_flat: torch.Tensor, target_rejection: float, gen: torch.Generator) -> torch.Tensor:
    gnorm = generic_flat.norm()
    if float(gnorm.item()) <= 1.0e-8:
        return generic_flat.clone()
    target = max(0.0, min(2.0, float(target_rejection)))
    return generic_flat + target * gnorm * random_unit_orthogonal(generic_flat, gen)


def same_value_retention_random(generic_flat: torch.Tensor, target_retention: float, gen: torch.Generator) -> torch.Tensor:
    gnorm = generic_flat.norm()
    if float(gnorm.item()) <= 1.0e-8:
        return generic_flat.clone()
    target = max(0.0, min(1.25, float(target_retention)))
    parallel = target * generic_flat
    perp_scale = math.sqrt(max(0.0, 1.0 - min(1.0, target) ** 2))
    return parallel + perp_scale * gnorm * random_unit_orthogonal(generic_flat, gen)


def permute_metric_scales(keys: list[str], key_scales: dict[str, float], rng: random.Random) -> dict[str, float]:
    unique = sorted(set(keys))
    values = [float(key_scales.get(k, 1.0)) for k in unique]
    rng.shuffle(values)
    return dict(zip(unique, values))


def split_agreement_scales(
    specs: list[ParamSpec],
    keys: list[str],
    flat_a: torch.Tensor,
    flat_b: torch.Tensor,
) -> dict[str, float]:
    numer: dict[str, float] = {}
    denom_a: dict[str, float] = {}
    denom_b: dict[str, float] = {}
    for spec, key in zip(specs, keys):
        a = flat_a[spec.start : spec.end]
        b = flat_b[spec.start : spec.end]
        numer[key] = numer.get(key, 0.0) + float(torch.dot(a, b).item())
        denom_a[key] = denom_a.get(key, 0.0) + float(a.square().sum().item())
        denom_b[key] = denom_b.get(key, 0.0) + float(b.square().sum().item())
    out: dict[str, float] = {}
    for key in set(keys):
        cos = numer.get(key, 0.0) / max(1.0e-8, math.sqrt(denom_a.get(key, 0.0)) * math.sqrt(denom_b.get(key, 0.0)))
        out[key] = max(0.0, min(1.0, cos))
    return out


def update_controls_for_method(method: str) -> dict[str, int]:
    return {
        "uses_label_in_init": 0,
        "uses_y_for_stats": 0,
        "uses_validation_for_direction": 0,
        "uses_test_for_direction": 0,
        "uses_future_for_direction": 0,
        "uses_query_for_direction": 0,
        "uses_linec_for_direction": 0,
        "uses_cep99_for_direction": 0,
        "uses_nll_for_direction": 0,
        "uses_ece_for_direction": 0,
        "uses_auctime_for_direction": 0,
        "uses_dataset_name_branch": 0,
        "uses_seed_specific_scale": 0,
        "is_action_token_extension": 0,
        "controller_executed": 0,
        "promotion_allowed": 0,
    }


def train_model_case(
    *,
    family: str,
    candidate_id: str,
    method: str,
    dataset: str,
    task: str,
    seed: int,
    loss_interface: str,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    xte: torch.Tensor | None,
    yte: torch.Tensor | None,
    input_dim: int,
    output_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    real_linec: bool,
) -> dict[str, Any]:
    case_args = copy(args)
    case_args.synthetic_dim = int(input_dim)
    case_args.synthetic_classes = int(output_dim)
    if family == "MLP":
        model = make_case_model("MLP", "MLP-v1410", xtr, seed, case_args, device)
    else:
        model = make_case_model("D-CHE", candidate_id, xtr, seed, case_args, device)
    specs = named_param_specs(model)
    params = [(spec.name, spec.param) for spec in specs]
    keys = [fms_key(spec, method, family) for spec in specs]
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed) + 1410_000 + sum(ord(c) for c in method + dataset + task + family + loss_interface))
    rng = random.Random(int(seed) + 1410_100 + len(method))
    state = FMSState(float(args.fms_beta), float(args.fms_strength), int(seed) + 1410)
    last_key_scales = {key: 1.0 for key in set(keys)}
    trajectory: list[dict[str, float]] = []
    degree_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    selector_active_values: list[float] = []
    projection_acc: dict[str, list[float]] = {
        "cos_projected_vs_generic": [],
        "value_retention_after_degree_projection": [],
        "degree_projection_rejection_fraction": [],
        "generic_value_norm": [],
        "projected_value_norm": [],
    }
    before = degree_energy(model) if family == "D-CHE" else {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    fms_refresh_count = 0
    for step in range(int(args.train_steps)):
        skip_optimizer_step = False
        utilities: dict[str, float] = {}
        idx = torch.randint(0, xtr.shape[0], (int(args.batch_size),), generator=gen, device=device)
        xb, yb = xtr[idx], ytr[idx]
        opt.zero_grad(set_to_none=True)
        regular_adamw = method in {
            "C0-D-CHE-AdamW",
            "C1-D-CHE-AdamW-NoOpMatchedOverhead",
            "C3-D-CHE-AdamW-AdamWParallelDirectionControl",
            "M0-MLP-AdamW",
            "M5-MLP-NoOpMatchedOverhead",
            "M11-MLP-AdamWParallelDirectionControl",
        }
        if regular_adamw:
            loss = loss_value(model(xb), yb, loss_interface)
            loss.backward()
            generic_flat = flat_existing_grad(specs)
            projected_flat = generic_flat
            degree_gate_active_fraction = 0.0
        else:
            refresh = step % max(1, int(args.fms_update_interval)) == 0
            if refresh:
                fms_refresh_count += 1
                agreement_scales: dict[str, float] = {}
                if method in {"F-CHE-RT1-TrainSplitAgreement", "F-CHE-RT4-CompositeTransferTrust"}:
                    split = max(1, int(xb.shape[0]) // 2)
                    flat_a, util_a = collect_streaming_per_example_summary(model, xb[:split], yb[:split], params, specs, keys, loss_interface)
                    flat_b, util_b = collect_streaming_per_example_summary(model, xb[split:], yb[split:], params, specs, keys, loss_interface)
                    generic_flat = 0.5 * (flat_a + flat_b)
                    utilities = {key: 0.5 * (float(util_a.get(key, 0.0)) + float(util_b.get(key, 0.0))) for key in set(keys)}
                    agreement_scales = split_agreement_scales(specs, keys, flat_a, flat_b)
                elif int(getattr(args, "streaming_per_example_gradients", 0)) == 1:
                    generic_flat, utilities = collect_streaming_per_example_summary(
                        model,
                        xb,
                        yb,
                        params,
                        specs,
                        keys,
                        loss_interface,
                        "population_metric_state" if method in METRIC_STATE_METHODS or method in METRIC_STATE_CONTROL_METHODS else "snr_energy",
                    )
                else:
                    g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface=loss_interface)
                    utilities = {}
                    for spec, key in zip(specs, keys):
                        sl = g[:, spec.start : spec.end]
                        mu = sl.mean(dim=0)
                        var = sl.var(dim=0, unbiased=False)
                        snr = float(mu.norm().item() / (var.mean().sqrt().item() + 1.0e-8))
                        energy = float(mu.square().mean().item())
                        diffusion = float(var.mean().item())
                        if method in METRIC_STATE_METHODS or method in METRIC_STATE_CONTROL_METHODS:
                            value = energy / (diffusion + 1.0e-8) - diffusion / max(1.0, float(g.shape[0] - 1))
                        else:
                            value = math.log1p(max(0.0, snr)) + math.log1p(max(0.0, energy) * 1.0e3)
                        utilities[key] = utilities.get(key, 0.0) + float(value)
                    generic_flat = g.mean(dim=0).detach().clone()
                if method in {"C2-D-CHE-AdamW-RandomMatchedNorm", "M3-MLP-RandomMatchedNorm"}:
                    key_scales = state.random_scales(keys)
                elif method in {"C5-D-CHE-AdamW-SameActiveFractionControl", "M6-MLP-SameActiveFractionControl"}:
                    reference_scales = state.update(keys, utilities, "F3-LayerFMS", step, int(args.train_steps))
                    active_keys = [key for key, scale in reference_scales.items() if abs(float(scale) - 1.0) > 0.01]
                    unique_keys = sorted(set(keys))
                    target_count = min(len(unique_keys), max(1, len(active_keys)))
                    shuffled = list(unique_keys)
                    rng.shuffle(shuffled)
                    selected = set(shuffled[:target_count])
                    random_scales = state.random_scales(keys)
                    key_scales = {key: (float(random_scales.get(key, 1.0)) if key in selected else 1.0) for key in unique_keys}
                elif method == "M7-MLP-BoundaryOnly":
                    boundary_scales = state.update(keys, utilities, "F3-LayerFMS", step, int(args.train_steps))
                    key_scales = {key: min(1.0, float(boundary_scales.get(key, 1.0))) for key in set(keys)}
                elif method == "C6-D-CHE-SameTCRandomDirection":
                    key_scales = state.update(keys, utilities, "MetricStateFMS", step, int(args.train_steps))
                    projected_flat, degree_gate_active_fraction = random_direction_with_scales(specs, keys, generic_flat, key_scales, gen)
                    last_key_scales = dict(key_scales)
                    assign_flat_grad(specs, projected_flat)
                    opt.step()
                    stats = projection_stats(generic_flat, projected_flat)
                    for key, value in stats.items():
                        projection_acc[key].append(float(value))
                    if step == 0 or step == int(args.train_steps) - 1 or ((step + 1) % max(1, int(args.trace_interval)) == 0):
                        metrics = eval_metrics(model, xva, yva)
                        trajectory.append({"step": float(step + 1), "NLL": metrics["NLL"], "CEp99": metrics["CEp99"], "ECE": metrics["ECE"], "acc": metrics["acc"]})
                        projection_rows.append(
                            {
                                "stage": "V1410_DCHE_PROJECTION_RETENTION",
                                "dataset": dataset,
                                "task": task,
                                "seed": seed,
                                "loss_interface": loss_interface,
                                "family": family,
                                "method": method,
                                "step": step + 1,
                                "degree_gate_active_fraction": degree_gate_active_fraction,
                        **stats,
                        "line_a_selector_id": str(getattr(args, "line_a_selector_id", "")),
                        "line_a_direction_id": str(getattr(args, "line_a_direction_id", "")),
                        "line_a_boundary_id": str(getattr(args, "line_a_boundary_id", "")),
                        "metric_used_as_direction": 0,
                        "metric_used_as_gate": 1,
                        "promotion_allowed": 0,
                            }
                        )
                    continue
                elif method in {"C7-D-CHE-SameMetricScaleRandomPermutation", "M10-MLP-SameMetricScaleRandomPermutation"}:
                    reference_scales = state.update(keys, utilities, "MetricStateFMS", step, int(args.train_steps))
                    key_scales = permute_metric_scales(keys, reference_scales, rng)
                elif method in LINE_A_CONTROL_METHODS:
                    reference_scales = state.update(keys, utilities, "F3-LayerFMS", step, int(args.train_steps))
                    reference_flat, _reference_active = apply_scales_to_flat(
                        "F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection",
                        specs,
                        keys,
                        generic_flat,
                        reference_scales,
                        step,
                        int(args.train_steps),
                    )
                    ref_stats = projection_stats(generic_flat, reference_flat)
                    key_scales = {key: 1.0 for key in set(keys)}
                    if method == "A-D4-SameProjectionRejectionRandom":
                        projected_flat = same_projection_rejection_random(generic_flat, ref_stats["degree_projection_rejection_fraction"], gen)
                    else:
                        projected_flat = same_value_retention_random(generic_flat, ref_stats["value_retention_after_degree_projection"], gen)
                    degree_gate_active_fraction = 1.0
                    last_key_scales = dict(key_scales)
                    assign_flat_grad(specs, projected_flat)
                elif method in {"C4-D-CHE-AdamW-GenericOptimizerStateControl", "M4-MLP-GenericOptimizerStateControl"}:
                    key_scales = {key: 1.0 for key in set(keys)}
                    state.update(keys, utilities, "F3-LayerFMS", step, int(args.train_steps))
                elif method == "F-CHE-FB2-DegreeConstraintOnly":
                    key_scales = {key: 1.0 for key in set(keys)}
                elif method in METRIC_STATE_METHODS:
                    key_scales = state.update(keys, utilities, "MetricStateFMS", step, int(args.train_steps))
                else:
                    fms_method = "F7-PhaseScheduleFMS" if method in {"F-CHE6-PhaseScheduleDegreeFMS"} else "F3-LayerFMS"
                    key_scales = state.update(keys, utilities, fms_method, step, int(args.train_steps))
                if method in {"F-CHE-RT1-TrainSplitAgreement", "F-CHE-RT4-CompositeTransferTrust"}:
                    key_scales = {key: 1.0 + agreement_scales.get(key, 0.0) * (float(key_scales.get(key, 1.0)) - 1.0) for key in set(keys)}
                if method not in LINE_A_CONTROL_METHODS:
                    projected_flat, degree_gate_active_fraction = apply_scales_to_flat(method, specs, keys, generic_flat, key_scales, step, int(args.train_steps))
                if method in {"F-CHE-RT2-ValueRetentionTrust", "F-CHE-RT4-CompositeTransferTrust"}:
                    stats_now = projection_stats(generic_flat, projected_flat)
                    if stats_now["cos_projected_vs_generic"] < 0.85 or stats_now["value_retention_after_degree_projection"] < 0.85:
                        projected_flat = 0.5 * projected_flat + 0.5 * generic_flat
                        degree_gate_active_fraction *= 0.5
                if method == "F-CHE-RT4-CompositeTransferTrust" and int(getattr(args, "actual_micro_horizon_gating", 0)) == 1:
                    horizon = max(1, int(getattr(args, "actual_micro_horizon_steps", 1)))
                    probe_split = max(1, int(xb.shape[0]) // 2)
                    xb2_probe, yb2_probe = xb[probe_split:], yb[probe_split:]
                    if int(xb2_probe.shape[0]) == 0:
                        xb2_probe, yb2_probe = xb[:probe_split], yb[:probe_split]
                    before_probe = train_batch_probe_metrics(model, xb2_probe, yb2_probe, loss_interface)
                    adamw_probe = deepcopy(model)
                    fms_probe = deepcopy(model)
                    adamw_metrics = []
                    fms_metrics = []
                    for _probe_step in range(horizon):
                        stateless_probe_update(adamw_probe, generic_flat, float(args.lr), float(args.weight_decay))
                        stateless_probe_update(fms_probe, projected_flat, float(args.lr), float(args.weight_decay))
                        adamw_metrics.append(train_batch_probe_metrics(adamw_probe, xb2_probe, yb2_probe, loss_interface))
                        fms_metrics.append(train_batch_probe_metrics(fms_probe, xb2_probe, yb2_probe, loss_interface))
                    adamw_final = adamw_metrics[-1]
                    fms_final = fms_metrics[-1]
                    delta_integral = sum(m["loss_mean"] for m in adamw_metrics) - sum(m["loss_mean"] for m in fms_metrics)
                    delta_q95 = fms_final["loss_q95"] - adamw_final["loss_q95"]
                    delta_margin = fms_final["margin_p10"] - adamw_final["margin_p10"]
                    rms_drift = abs(fms_final["logit_rms"] - before_probe["logit_rms"]) - abs(adamw_final["logit_rms"] - before_probe["logit_rms"])
                    entropy_collapse = max(0.0, before_probe["entropy_norm"] - fms_final["entropy_norm"]) - max(0.0, before_probe["entropy_norm"] - adamw_final["entropy_norm"])
                    recovery_lag = sum(max(0.0, m["loss_mean"] - before_probe["loss_mean"]) for m in fms_metrics)
                    micro_score = delta_integral - 0.25 * max(0.0, delta_q95) + 0.10 * delta_margin - 0.10 * max(0.0, rms_drift) - 0.10 * max(0.0, entropy_collapse) - 0.05 * recovery_lag
                    micro_gate = max(0.25, min(1.0, 0.5 + 0.5 * math.tanh(5.0 * micro_score)))
                    projected_flat = generic_flat + micro_gate * (projected_flat - generic_flat)
                    degree_gate_active_fraction *= micro_gate
                if int(getattr(args, "boundary_abstention_gating", 0)) == 1:
                    stats_now = projection_stats(generic_flat, projected_flat)
                    unsafe = (
                        max(0.0, 0.85 - float(stats_now["cos_projected_vs_generic"]))
                        + max(0.0, 0.85 - float(stats_now["value_retention_after_degree_projection"]))
                        + max(0.0, float(stats_now["degree_projection_rejection_fraction"]) - 0.20)
                    )
                    trust = max(0.10, min(1.0, 1.0 - 2.0 * unsafe))
                    projected_flat = trust * projected_flat
                    degree_gate_active_fraction *= trust
                last_key_scales = dict(key_scales)
                assign_flat_grad(specs, projected_flat)
            else:
                loss = loss_value(model(xb), yb, loss_interface)
                loss.backward()
                generic_flat = flat_existing_grad(specs)
                projected_flat, degree_gate_active_fraction = apply_scales_to_flat(method, specs, keys, generic_flat, last_key_scales, step, int(args.train_steps))
                assign_flat_grad(specs, projected_flat)
        line_a_boundary = str(getattr(args, "line_a_boundary_id", "") or "")
        if line_a_boundary:
            stats_now = projection_stats(generic_flat, projected_flat)
            if line_a_boundary == "B1-degree-projection-safety":
                if stats_now["degree_projection_rejection_fraction"] > 0.30 or stats_now["value_retention_after_degree_projection"] < 0.75:
                    projected_flat = 0.50 * projected_flat + 0.50 * generic_flat
            elif line_a_boundary == "B4-value-retention-floor":
                if stats_now["value_retention_after_degree_projection"] < 0.85:
                    projected_flat = generic_flat + 0.25 * (projected_flat - generic_flat)
            assign_flat_grad(specs, projected_flat)
        line_a_selector = str(getattr(args, "line_a_selector_id", "") or "")
        if line_a_selector:
            selector_active = 1.0
            if line_a_selector == "S1-predictor-high-score":
                util_values = list(utilities.values()) if "utilities" in locals() else [0.0]
                selector_score = sum(float(v) for v in util_values) / max(1, len(util_values))
                selector_active = float(selector_score > 0.0)
            elif line_a_selector == "S4-NoOp-safe-abstention":
                stats_now = projection_stats(generic_flat, projected_flat)
                selector_active = float(
                    stats_now["value_retention_after_degree_projection"] >= 0.85
                    and stats_now["degree_projection_rejection_fraction"] <= 0.50
                )
            if selector_active <= 0.0:
                projected_flat = torch.zeros_like(generic_flat)
                for spec in specs:
                    spec.param.grad = None
                skip_optimizer_step = True
            else:
                assign_flat_grad(specs, projected_flat)
            selector_active_values.append(selector_active)
        if method in {"C1-D-CHE-AdamW-NoOpMatchedOverhead", "M5-MLP-NoOpMatchedOverhead"}:
            _ = sum(float(v) for v in degree_energy(model).values())
        if method in {"C3-D-CHE-AdamW-AdamWParallelDirectionControl", "M11-MLP-AdamWParallelDirectionControl"}:
            with torch.no_grad():
                _ = flat_existing_grad(specs).norm().item()
        if not skip_optimizer_step:
            opt.step()
        stats = projection_stats(generic_flat, projected_flat)
        for key, value in stats.items():
            projection_acc[key].append(float(value))
        if step == 0 or step == int(args.train_steps) - 1 or ((step + 1) % max(1, int(args.trace_interval)) == 0):
            metrics = eval_metrics(model, xva, yva)
            trajectory.append({"step": float(step + 1), "NLL": metrics["NLL"], "CEp99": metrics["CEp99"], "ECE": metrics["ECE"], "acc": metrics["acc"]})
            if family == "D-CHE":
                de = degree_energy(model)
                degree_rows.append(
                    {
                        "stage": "V1410_DCHE_DEGREE_TELEMETRY",
                        "dataset": dataset,
                        "task": task,
                        "seed": seed,
                        "loss_interface": loss_interface,
                        "method": method,
                        "step": step + 1,
                        "degree_energy_total": de.get("degree_energy_total", 0.0),
                        "high_degree_energy_fraction": de.get("high_degree_energy_fraction", 0.0),
                        "degree_entropy": de.get("degree_entropy", 0.0),
                        "degree_gate_active_fraction": degree_gate_active_fraction,
                        **update_controls_for_method(method),
                    }
                )
            projection_rows.append(
                {
                    "stage": "V1410_DCHE_PROJECTION_RETENTION",
                    "dataset": dataset,
                    "task": task,
                    "seed": seed,
                    "loss_interface": loss_interface,
                    "family": family,
                    "method": method,
                    "step": step + 1,
                    "degree_gate_active_fraction": degree_gate_active_fraction,
                    **stats,
                    "line_a_selector_id": str(getattr(args, "line_a_selector_id", "")),
                    "line_a_direction_id": str(getattr(args, "line_a_direction_id", "")),
                    "line_a_boundary_id": str(getattr(args, "line_a_boundary_id", "")),
                    "metric_used_as_direction": 0,
                    "metric_used_as_gate": 1,
                    "promotion_allowed": 0,
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory_bytes = 0
    elapsed = time.perf_counter() - start
    final = eval_metrics(model, xva, yva)
    test_final = eval_metrics(model, xte, yte) if xte is not None and yte is not None else {}
    auc_nll = sum(item["NLL"] for item in trajectory) / max(1, len(trajectory))
    auc_cep99 = sum(item["CEp99"] for item in trajectory) / max(1, len(trajectory))
    if family == "MLP":
        linec_votes = [1]
        linec_rows: list[dict[str, Any]] = []
        linec_mean = {"CouplingR2": final["CouplingR2"], "NoiseSignalLeak": final["NoiseSignalLeak"], "RealSignalReservoirRatio": final["RealSignalReservoirRatio"]}
    elif real_linec:
        b = min(int(args.linec_batch_size), int(xtr.shape[0]), int(xva.shape[0]))
        linec_votes = []
        linec_rows = []
        for ls in parse_ints(args.linec_seeds):
            try:
                lm = linec_metrics(model, xtr[:b], ytr[:b], xva[:b], yva[:b], int(ls), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
                status = "executed"
                error = ""
            except Exception as exc:  # noqa: BLE001
                lm = {"CouplingR2": float("nan"), "NoiseSignalLeak": float("nan"), "RealSignalReservoirRatio": float("nan")}
                status = "blocked"
                error = f"{type(exc).__name__}: {exc}"
            passed = int(fnum(lm.get("CouplingR2"), -999.0) >= 0.15 and fnum(lm.get("NoiseSignalLeak"), 999.0) <= 0.20 and fnum(lm.get("RealSignalReservoirRatio"), 999.0) <= 0.70)
            linec_votes.append(passed)
            linec_rows.append({"linec_seed": int(ls), "linec_status": status, "linec_error": error, "LineC_pass": passed, **lm})
        linec_mean = {
            "CouplingR2": sum(fnum(r.get("CouplingR2"), 0.0) for r in linec_rows) / max(1, len(linec_rows)),
            "NoiseSignalLeak": sum(fnum(r.get("NoiseSignalLeak"), 0.0) for r in linec_rows) / max(1, len(linec_rows)),
            "RealSignalReservoirRatio": sum(fnum(r.get("RealSignalReservoirRatio"), 0.0) for r in linec_rows) / max(1, len(linec_rows)),
        }
    else:
        linec_votes = [linec_proxy({"CEp99": item["CEp99"], "NoiseSignalLeak": final["NoiseSignalLeak"], "margin_p10": final["margin_p10"]}) for item in trajectory]
        linec_rows = []
        linec_mean = {"CouplingR2": final["CouplingR2"], "NoiseSignalLeak": final["NoiseSignalLeak"], "RealSignalReservoirRatio": final["RealSignalReservoirRatio"]}
    after = degree_energy(model) if family == "D-CHE" else {}
    row = {
        "stage": "V1410_DCHE_CASE_RESULT",
        "family": family,
        "candidate_id": candidate_id if family == "D-CHE" else "MLP-v1410-control",
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "loss_interface": loss_interface,
        "method": method,
        "control_method": int(
            method in CONTROL_METHODS
            or method in METRIC_STATE_CONTROL_METHODS
            or method in LINE_A_CONTROL_METHODS
            or (family == "MLP" and method in METRIC_STATE_METHODS)
            or method
            in {
                "M0-MLP-AdamW",
                "M3-MLP-RandomMatchedNorm",
                "M4-MLP-GenericOptimizerStateControl",
                "M5-MLP-NoOpMatchedOverhead",
                "M6-MLP-SameActiveFractionControl",
                "M7-MLP-BoundaryOnly",
                "M10-MLP-SameMetricScaleRandomPermutation",
                "M11-MLP-AdamWParallelDirectionControl",
            }
        ),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / max(1, int(args.train_steps)),
        "peak_memory_bytes": peak_memory_bytes,
        "NLL": final["NLL"],
        "CEp99": final["CEp99"],
        "ECE": final["ECE"],
        "Brier": final["Brier"],
        "acc": final["acc"],
        "test_NLL": test_final.get("NLL", ""),
        "test_CEp99": test_final.get("CEp99", ""),
        "test_ECE": test_final.get("ECE", ""),
        "test_acc": test_final.get("acc", ""),
        "CouplingR2": linec_mean["CouplingR2"],
        "NoiseSignalLeak": linec_mean["NoiseSignalLeak"],
        "RealSignalReservoirRatio": linec_mean["RealSignalReservoirRatio"],
        "margin_p10": final["margin_p10"],
        "AUC_NLL": auc_nll,
        "AUC_CEp99": auc_cep99,
        "LineC_pass_rate": sum(linec_votes) / max(1, len(linec_votes)),
        "LineC_majority_pass": int(sum(linec_votes) >= math.ceil(len(linec_votes) / 2)),
        "fms_state_norm": float(sum(abs(v) for v in state.state.values())),
        "parameter_update_norm": median(projection_acc["projected_value_norm"]),
        "degree_energy_before": before.get("degree_energy_total", ""),
        "degree_energy_after": after.get("degree_energy_total", ""),
        "high_degree_energy_fraction": after.get("high_degree_energy_fraction", ""),
        "degree_entropy": after.get("degree_entropy", ""),
        "degree_gate_active_fraction": median([fnum(r.get("degree_gate_active_fraction"), 0.0) for r in projection_rows]),
        "degree_projection_rejection_fraction": median(projection_acc["degree_projection_rejection_fraction"]),
        "value_retention_after_degree_projection": median(projection_acc["value_retention_after_degree_projection"]),
        "cos_projected_vs_generic": median(projection_acc["cos_projected_vs_generic"]),
        "control_equivalent_flag": int(
            method in CONTROL_METHODS
            or method in METRIC_STATE_CONTROL_METHODS
            or method in LINE_A_CONTROL_METHODS
            or (family == "MLP" and method in METRIC_STATE_METHODS)
            or method
            in {
                "M3-MLP-RandomMatchedNorm",
                "M4-MLP-GenericOptimizerStateControl",
                "M5-MLP-NoOpMatchedOverhead",
                "M6-MLP-SameActiveFractionControl",
                "M7-MLP-BoundaryOnly",
                "M10-MLP-SameMetricScaleRandomPermutation",
                "M11-MLP-AdamWParallelDirectionControl",
            }
        ),
        "metric_state_variant": int(method in METRIC_STATE_METHODS),
        "population_risk_utility_mode": int(method in METRIC_STATE_METHODS or method in METRIC_STATE_CONTROL_METHODS),
        "line_a_selector_id": str(getattr(args, "line_a_selector_id", "")),
        "line_a_direction_id": str(getattr(args, "line_a_direction_id", "")),
        "line_a_boundary_id": str(getattr(args, "line_a_boundary_id", "")),
        "line_a_selector_active_fraction": sum(selector_active_values) / max(1, len(selector_active_values)),
        "fms_refresh_count": int(fms_refresh_count),
        "streaming_per_example_gradients": int(getattr(args, "streaming_per_example_gradients", 0)),
        "actual_micro_horizon_gating": int(getattr(args, "actual_micro_horizon_gating", 0)),
        "actual_micro_horizon_steps": int(getattr(args, "actual_micro_horizon_steps", 0)),
        "boundary_abstention_gating": int(getattr(args, "boundary_abstention_gating", 0)),
        "direction_uses_train_stream_only": 1,
        **update_controls_for_method(method),
    }
    lc_full_rows = []
    for item in linec_rows:
        lc_full_rows.append(
            {
                "stage": "V1410_LINEC_TAIL_AUDIT",
                "family": family,
                "dataset": dataset,
                "task": task,
                "seed": seed,
                "loss_interface": loss_interface,
                "method": method,
                **item,
                "CEp99": final["CEp99"],
                "NLL": final["NLL"],
                "ECE": final["ECE"],
                "Brier": final["Brier"],
                "metric_used_as_direction": 0,
                "metric_used_as_gate": 1,
                "promotion_allowed": 0,
            }
        )
    if not lc_full_rows:
        lc_full_rows.append(
            {
                "stage": "V1410_LINEC_TAIL_AUDIT",
                "family": family,
                "dataset": dataset,
                "task": task,
                "seed": seed,
                "loss_interface": loss_interface,
                "method": method,
                "LineC_pass": row["LineC_majority_pass"],
                "CouplingR2": row["CouplingR2"],
                "NoiseSignalLeak": row["NoiseSignalLeak"],
                "RealSignalReservoirRatio": row["RealSignalReservoirRatio"],
                "CEp99": final["CEp99"],
                "NLL": final["NLL"],
                "ECE": final["ECE"],
                "Brier": final["Brier"],
                "metric_used_as_direction": 0,
                "metric_used_as_gate": 1,
                "promotion_allowed": 0,
            }
        )
    return {"row": row, "degree_rows": degree_rows, "projection_rows": projection_rows, "linec_rows": lc_full_rows}


def enrich_rows(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("family")), str(row.get("dataset")), int(row.get("seed", 0)), str(row.get("task")), str(row.get("loss_interface")))
        by_key.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for _key, group in by_key.items():
        controls = [r for r in group if sint(r.get("control_method"), 0) == 1]
        if not controls:
            controls = group
        adam = next((r for r in group if str(r.get("method")) in {"C0-D-CHE-AdamW", "M0-MLP-AdamW"}), controls[0])
        best_nll = min(fnum(r.get("NLL"), 9.0) for r in controls)
        best_auc = min(fnum(r.get("AUC_NLL"), 9.0) for r in controls)
        base_time = max(1.0e-8, fnum(adam.get("step_time_sec"), 1.0))
        base_mem = max(1.0, fnum(adam.get("peak_memory_bytes"), 1.0))
        for row in group:
            item = dict(row)
            item["stage"] = stage
            item["source_vs_adamw"] = fnum(adam.get("NLL"), 9.0) - fnum(row.get("NLL"), 9.0)
            item["source_vs_best_control"] = best_nll - fnum(row.get("NLL"), 9.0)
            item["AUCtime_ratio"] = fnum(row.get("AUC_NLL"), 9.0) / max(1.0e-8, best_auc)
            item["CEp99_delta"] = fnum(row.get("CEp99"), 0.0) - fnum(adam.get("CEp99"), 0.0)
            item["NLL_delta"] = fnum(row.get("NLL"), 0.0) - fnum(adam.get("NLL"), 0.0)
            item["ECE_delta"] = fnum(row.get("ECE"), 0.0) - fnum(adam.get("ECE"), 0.0)
            item["step_time_ratio"] = fnum(row.get("step_time_sec"), 0.0) / base_time
            item["memory_ratio"] = fnum(row.get("peak_memory_bytes"), 0.0) / base_mem if base_mem > 1.0 else 1.0
            item["strict_gate_pass"] = int(
                sint(item.get("control_method"), 0) == 0
                and fnum(item.get("source_vs_best_control"), -999.0) >= 0.005
                and fnum(item.get("AUCtime_ratio"), 9.0) <= 1.0
                and fnum(item.get("CEp99_delta"), 999.0) <= 0.05
                and fnum(item.get("NLL_delta"), 999.0) <= 0.02
                and fnum(item.get("ECE_delta"), 999.0) <= 0.02
                and sint(item.get("LineC_majority_pass"), 0) == 1
                and fnum(item.get("step_time_ratio"), 999.0) <= 1.25
                and fnum(item.get("memory_ratio"), 999.0) <= 1.25
            )
            out.append(item)
    return out


def family_task_pass(rows: list[dict[str, Any]]) -> tuple[int, dict[str, int]]:
    task_to_seeds: dict[str, set[int]] = {}
    task_to_losses: dict[str, set[str]] = {}
    for row in rows:
        if sint(row.get("strict_gate_pass"), 0) != 1:
            continue
        task = str(row.get("task"))
        task_to_seeds.setdefault(task, set()).add(int(row.get("seed", 0)))
        task_to_losses.setdefault(task, set()).add(str(row.get("loss_interface")))
    task_pass: dict[str, int] = {}
    for task in sorted(set(task_to_seeds) | set(task_to_losses)):
        task_pass[task] = int(len(task_to_seeds.get(task, set())) >= 2 or len(task_to_losses.get(task, set())) >= 2)
    return sum(task_pass.values()), task_pass


def summarize_methods(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in sorted({str(r.get("method")) for r in rows}):
        group = [r for r in rows if str(r.get("method")) == method]
        pass_keys = {(str(r.get("dataset")), str(r.get("task")), int(r.get("seed", 0)), str(r.get("loss_interface"))) for r in group if sint(r.get("strict_gate_pass"), 0) == 1}
        out.append(
            {
                "stage": stage,
                "method": method,
                "rows": len(group),
                "strict_pass_rows": len([r for r in group if sint(r.get("strict_gate_pass"), 0) == 1]),
                "strict_unique_pass_count": len(pass_keys),
                "task_family_pass_count": family_task_pass(group)[0] if any(str(r.get("task")).startswith("X") for r in group) else "",
                "dataset_seed_pass_count": len({(str(r.get("dataset")), int(r.get("seed", 0))) for r in group if sint(r.get("strict_gate_pass"), 0) == 1}),
                "mean_source_vs_best_control": sum(fnum(r.get("source_vs_best_control"), 0.0) for r in group) / max(1, len(group)),
                "median_AUCtime_ratio": median([fnum(r.get("AUCtime_ratio"), 9.0) for r in group]),
                "median_step_time_ratio": median([fnum(r.get("step_time_ratio"), 9.0) for r in group]),
                "median_memory_ratio": median([fnum(r.get("memory_ratio"), 9.0) for r in group]),
                "control_method": int(all(sint(r.get("control_method"), 0) == 1 for r in group)),
                "promotion_allowed": 0,
            }
        )
    return out


def failure_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if fnum(row.get("source_vs_best_control"), -999.0) < 0.005:
        reasons.append("source")
    if fnum(row.get("AUCtime_ratio"), 9.0) > 1.0:
        reasons.append("AUCtime")
    if fnum(row.get("CEp99_delta"), 999.0) > 0.05:
        reasons.append("CEp99")
    if fnum(row.get("NLL_delta"), 999.0) > 0.02:
        reasons.append("NLL")
    if fnum(row.get("ECE_delta"), 999.0) > 0.02:
        reasons.append("ECE")
    if sint(row.get("LineC_majority_pass"), 0) != 1:
        reasons.append("LineC")
    if fnum(row.get("step_time_ratio"), 9.0) > 1.25:
        reasons.append("step")
    if fnum(row.get("memory_ratio"), 9.0) > 1.25:
        reasons.append("memory")
    if sint(row.get("control_equivalent_flag"), 0) == 1:
        reasons.append("control_equivalent")
    return reasons


def failure_taxonomy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if sint(row.get("control_method"), 0) == 1:
            continue
        reasons = failure_reasons(row)
        out.append(
            {
                "stage": "V1410_FAILURE_TAXONOMY",
                "family": row.get("family"),
                "dataset": row.get("dataset"),
                "task": row.get("task"),
                "seed": row.get("seed"),
                "loss_interface": row.get("loss_interface"),
                "method": row.get("method"),
                "strict_gate_pass": row.get("strict_gate_pass"),
                "failure_reason": "pass" if not reasons else ";".join(reasons),
                "source_fail": int("source" in reasons),
                "AUC_fail": int("AUCtime" in reasons),
                "CEp99_fail": int("CEp99" in reasons),
                "NLL_fail": int("NLL" in reasons),
                "ECE_fail": int("ECE" in reasons),
                "LineC_fail": int("LineC" in reasons),
                "step_fail": int("step" in reasons),
                "memory_fail": int("memory" in reasons),
                "control_equivalent": int("control_equivalent" in reasons),
                "promotion_allowed": 0,
            }
        )
    return out


def substrate_status() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = read_rows(V149_SUBSTRATE_DIR / "v149_line_d_substrate_repair_summary.csv")
    result = read_rows(V149_SUBSTRATE_DIR / "v149_line_d_substrate_repair_results.csv")
    status_rows: list[dict[str, Any]] = []
    for row in summary:
        status_rows.append(
            {
                "stage": "V1410_ALL_BASIS_SUBSTRATE_STATUS",
                "family": row.get("family"),
                "source_artifact": str(V149_SUBSTRATE_DIR / "v149_line_d_substrate_repair_summary.csv"),
                "dataset_seed_pass_count": row.get("family_dataset_seed_pass_count", 0),
                "official_fms_eligibility": row.get("family_official_fms_eligibility", 0),
                "exploration_gate_pass": row.get("family_exploration_gate_pass", 0),
                "best_candidate": row.get("best_candidate", ""),
                "best_candidate_dataset_seed_pass_count": row.get("best_candidate_dataset_seed_pass_count", ""),
                "best_mean_delta_vs_MLP": row.get("best_mean_delta_vs_MLP", ""),
                "best_LineC_pass_rate": row.get("best_LineC_pass_rate", ""),
                "min_NLL_ratio_vs_MLP": row.get("min_NLL_ratio_vs_MLP", ""),
                "median_train_step_ratio_vs_MLP": row.get("median_train_step_ratio_vs_MLP", ""),
                "official_fms_proof_executed": 0,
                "promotion_allowed": 0,
            }
        )
    def fam_rows(prefix: str) -> list[dict[str, Any]]:
        return [
            {
                "stage": f"V1410_{prefix}_SUBSTRATE_HARDENING",
                **row,
                "source_artifact": str(V149_SUBSTRATE_DIR / "v149_line_d_substrate_repair_results.csv"),
                "official_fms_proof_executed": 0,
                "promotion_allowed": 0,
            }
            for row in result
            if str(row.get("family")) == prefix
        ]
    return status_rows, fam_rows("D-FOU"), fam_rows("D-RBF"), fam_rows("D-WAV")


def write_audits(out_dir: Path, methods: list[str]) -> int:
    rows = []
    for method in methods:
        row = {"stage": "V1410_FORBIDDEN_INFORMATION_AUDIT", "method": method, **update_controls_for_method(method)}
        row["violation"] = int(any(sint(row.get(k), 0) for k in row if k.startswith("uses_")) or sint(row.get("is_action_token_extension"), 0) or sint(row.get("controller_executed"), 0))
        rows.append(row)
    write_rows(out_dir / "v1410_forbidden_information_audit.csv", rows)
    no_action = []
    for method in methods:
        no_action.append(
            {
                "stage": "V1410_NO_ACTION_SEARCH_AUDIT",
                "method": method,
                "pre_registered_method_set": int(method in CONTROL_METHODS or method in F_CHE_METHODS or method in FALLBACK_METHODS or method in REAL_TRANSFER_METHODS or method in MLP_METHODS),
                "is_action_token_extension": 0,
                "controller_executed": 0,
                "reset_route_executed": 0,
                "uses_task_dataset_seed_branch": 0,
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v1410_no_action_search_audit.csv", no_action)
    return sum(sint(row.get("violation"), 0) for row in rows)


def write_required_manifest(out_dir: Path) -> int:
    rows = []
    for name in REQUIRED + FIGURES:
        path = out_dir / name
        rows.append({"artifact": name, "exists": int(path.exists()), "bytes": path.stat().st_size if path.exists() else 0})
    write_rows(out_dir / "v1410_required_artifact_manifest.csv", rows)
    return sum(1 for row in rows if sint(row.get("exists"), 0) == 0)


def write_code_packet(out_dir: Path) -> None:
    files = [
        Path(__file__),
        PLAN_PATH,
        Path("experiments/run_v142_functional_first_all_basis_parallel.py"),
        Path("experiments/run_v149_line_d_all_basis_substrate_repair.py"),
        Path("experiments/run_v133_task_family_robust_basis_natural.py"),
        Path("experiments/run_v1231_basis_kernel_workspace.py"),
    ]
    manifest = []
    with zipfile.ZipFile(out_dir / "v1410_code_review_packet.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if not path.exists():
                manifest.append({"path": str(path), "exists": 0, "sha256": ""})
                continue
            zf.write(path, arcname=str(path))
            manifest.append({"path": str(path), "exists": 1, "sha256": sha256_file(path)})
        zf.writestr("v1410_code_review_manifest.csv", rows_to_csv(manifest))


def build_route(
    synthetic_rows: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    substrate_rows: list[dict[str, Any]],
    forbidden_count: int,
    missing: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dche_substrate = next((r for r in substrate_rows if str(r.get("family")) == "D-CHE"), {})
    dche_eligible = sint(dche_substrate.get("official_fms_eligibility"), 0) == 1
    synthetic_pass_count, task_pass = family_task_pass(synthetic_rows)
    real_pass_count = len({(str(r.get("dataset")), int(r.get("seed", 0))) for r in real_rows if sint(r.get("strict_gate_pass"), 0) == 1 and sint(r.get("control_method"), 0) == 0})
    mlp_best = max((fnum(r.get("source_vs_adamw"), -999.0) for r in mlp_rows if sint(r.get("control_method"), 0) == 0), default=-999.0)
    dche_best = max((fnum(r.get("source_vs_adamw"), -999.0) for r in synthetic_rows + real_rows if sint(r.get("control_method"), 0) == 0), default=-999.0)
    generic_confound = int(dche_best <= mlp_best and dche_best > -998.0)
    if forbidden_count > 0:
        route = "R0-ProvenanceOrActionSearchViolation"
        minimum = "S0-ExecutionCompleteNoPromotion"
    elif not dche_eligible:
        route = "R4-DCHESubstrateRegression"
        minimum = "S0-ExecutionCompleteNoPromotion"
    elif synthetic_pass_count < 5:
        route = "R1-DCHESyntheticFMSFail"
        minimum = "S0-ExecutionCompleteNoPromotion"
    elif generic_confound and real_pass_count >= 9:
        route = "R3-GenericFMSConfound"
        minimum = "S4-DCHRealTransferExplorationPositive"
    elif real_rows and real_pass_count >= 9:
        route = "S5-OfficialFunctionalSuccess"
        minimum = "S5-OfficialFunctionalSuccess"
    elif real_rows and real_pass_count >= 6:
        route = "S4-DCHRealTransferExplorationPositive"
        minimum = "S4-DCHRealTransferExplorationPositive"
    elif real_rows:
        route = "R2-DCHESyntheticDoesNotTransfer"
        minimum = "S3-DCHESyntheticFMSPass"
    elif synthetic_pass_count >= 5:
        route = "S3-DCHESyntheticFMSPass"
        minimum = "S3-DCHESyntheticFMSPass"
    else:
        route = "S0-ExecutionCompleteNoPromotion"
        minimum = "S0-ExecutionCompleteNoPromotion"
    return {
        "stage": "V1410_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "official_s5_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "synthetic_task_family_pass_count": synthetic_pass_count,
        "synthetic_task_pass": task_pass,
        "real_dataset_seed_pass_count": real_pass_count,
        "dche_substrate_official_fms_eligibility": int(dche_eligible),
        "dche_best_source_vs_adamw": dche_best,
        "mlp_best_source_vs_adamw": mlp_best,
        "generic_fms_confound": generic_confound,
        "forbidden_information_violation_count": forbidden_count,
        "required_artifact_missing_count": missing,
        "promotion_allowed": int(route == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0 and missing == 0 and forbidden_count == 0),
        "compute_budgeted_run": int(args.compute_budgeted_run),
    }


def write_boundaries(out_dir: Path, route: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# v14.10 no-go boundary",
        "",
        f"- route = {route.get('route')}",
        f"- minimum_success = {route.get('minimum_success')}",
        f"- promotion_allowed = {route.get('promotion_allowed')}",
        "- D-CHE substrate eligibility is not functional success by itself.",
        "- LineC / CEp99 / NLL / ECE / AUCtime / Brier are audit/gate fields only.",
        "- No reset/action/controller route was executed.",
    ]
    if route.get("route") == "R1-DCHESyntheticFMSFail":
        lines += [
            "",
            "## Case A decomposition",
            "",
            "- D-CHE synthetic S3 did not pass, so real 3x3 is not opened.",
            "- Allowed fallback set is limited to FB1/FB2/FB3; no F-CHE8 was added.",
        ]
        counts: dict[str, int] = {}
        for row in failures:
            for reason in str(row.get("failure_reason", "")).split(";"):
                if reason and reason != "pass":
                    counts[reason] = counts.get(reason, 0) + 1
        for key, value in sorted(counts.items()):
            lines.append(f"- {key}: {value}")
    (out_dir / "v1410_no_go_boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    next_lines = [
        "# v14.10 next hypothesis queue",
        "",
        "- If D-CHE synthetic fails after FB1/FB2/FB3, reset the FMS definition rather than adding action/search tokens.",
        "- If D-CHE synthetic passes but real <6/9, design the next version around train-stream-visible transfer decomposition.",
        "- If D-FOU/D-RBF/D-WAV reach 9/9 substrate, pre-register v14.11 FMS proof before executing it.",
    ]
    (out_dir / "v1410_next_hypothesis_queue.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def write_figures(out_dir: Path, route: dict[str, Any], progress: list[dict[str, Any]], substrate: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    write_svg(out_dir / "fig_v1410_progress_by_line.svg", "v14.10 progress", [f"route={route.get('route')}", f"synthetic={route.get('synthetic_task_family_pass_count')}/7", f"real={route.get('real_dataset_seed_pass_count')}/9"])
    write_svg(out_dir / "fig_dche_synthetic_5of7_heatmap.svg", "D-CHE synthetic", [f"{r.get('method')}: task_pass={r.get('task_family_pass_count')}" for r in progress if str(r.get("method")).startswith("F-CHE")][:16])
    write_svg(out_dir / "fig_dche_real_3x3_pass_matrix.svg", "D-CHE real 3x3", [f"{r.get('method')}: ds_seed={r.get('dataset_seed_pass_count')}" for r in progress if str(r.get("stage")) == "V1410_REAL_SUMMARY"][:16])
    write_svg(out_dir / "fig_dche_source_auc_tail_scatter.svg", "D-CHE source/AUC/tail", [f"{r.get('method')}: src={fnum(r.get('mean_source_vs_best_control')):.4f} auc={fnum(r.get('median_AUCtime_ratio')):.3f}" for r in progress[:16]])
    write_svg(out_dir / "fig_dche_degree_energy_before_after.svg", "D-CHE degree energy", [f"route={route.get('route')}", f"best_source={fnum(route.get('dche_best_source_vs_adamw')):.4f}"])
    write_svg(out_dir / "fig_dche_projection_value_retention.svg", "D-CHE projection retention", [f"{r.get('method')}: step={fnum(r.get('median_step_time_ratio')):.3f}" for r in progress[:16]])
    write_svg(out_dir / "fig_dche_fms_vs_controls_diffindiff.svg", "D-CHE FMS vs controls", [f"{r.get('method')}: pass={r.get('strict_unique_pass_count')}" for r in progress[:16]])
    write_svg(out_dir / "fig_mlp_vs_dche_fms_gain.svg", "MLP vs D-CHE FMS", [f"dche_best={fnum(route.get('dche_best_source_vs_adamw')):.4f}", f"mlp_best={fnum(route.get('mlp_best_source_vs_adamw')):.4f}"])
    write_svg(out_dir / "fig_all_basis_substrate_status.svg", "All-basis substrate status", [f"{r.get('family')}: {r.get('dataset_seed_pass_count')}/9 eligible={r.get('official_fms_eligibility')}" for r in substrate])
    write_svg(out_dir / "fig_fourier_rbf_wavelet_substrate_heatmap.svg", "FOU/RBF/WAV substrate", [f"{r.get('family')}: best={r.get('best_candidate')}" for r in substrate if r.get("family") in {"D-FOU", "D-RBF", "D-WAV"}])
    write_svg(out_dir / "fig_linec_tail_audit_by_method.svg", "LineC/tail audit", [f"{r.get('method')}: {r.get('failure_reason')}" for r in failures[:16]])
    write_svg(out_dir / "fig_failure_taxonomy.svg", "Failure taxonomy", [f"{r.get('method')}: {r.get('failure_reason')}" for r in failures[:16]])


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    methods = parse_csv(args.methods)
    mlp_methods = [] if int(args.skip_mlp_controls) else parse_csv(args.mlp_methods)
    all_methods_for_audit = methods + mlp_methods + (parse_csv(args.fallback_methods) if int(args.run_fallbacks) else [])
    forbidden_count = write_audits(out_dir, all_methods_for_audit)

    substrate_rows, fou_rows, rbf_rows, wav_rows = substrate_status()
    write_rows(out_dir / "v1410_all_basis_substrate_status.csv", substrate_rows)
    write_rows(out_dir / "v1410_fourier_substrate_hardening.csv", fou_rows)
    write_rows(out_dir / "v1410_rbf_substrate_hardening.csv", rbf_rows)
    write_rows(out_dir / "v1410_wavelet_substrate_hardening.csv", wav_rows)
    write_rows(out_dir / "v1410_rational_monitor.csv", [])

    synthetic_raw: list[dict[str, Any]] = []
    degree_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    linec_tail_rows: list[dict[str, Any]] = []
    synthetic_source_dir = Path(str(args.synthetic_source_dir)) if str(args.synthetic_source_dir).strip() else None
    if synthetic_source_dir is not None:
        dche_synth = read_rows(synthetic_source_dir / "v1410_dche_fms_synthetic_results.csv")
        mlp_synth_all = read_rows(synthetic_source_dir / "v1410_mlp_generic_controls.csv")
        mlp_synth = [r for r in mlp_synth_all if str(r.get("family")) == "MLP" and str(r.get("dataset")) == "synthetic"]
        for row in dche_synth + mlp_synth:
            row["synthetic_reused_from_s3_artifact"] = 1
            row["synthetic_source_artifact_dir"] = str(synthetic_source_dir)
        degree_rows.extend(read_rows(synthetic_source_dir / "v1410_dche_degree_telemetry.csv"))
        projection_rows.extend(read_rows(synthetic_source_dir / "v1410_dche_projection_retention.csv"))
        linec_tail_rows.extend(read_rows(synthetic_source_dir / "v1410_linec_tail_audit.csv"))
        synth_summary = summarize_methods(dche_synth, "V1410_SYNTHETIC_SUMMARY")
        mlp_summary = summarize_methods(mlp_synth, "V1410_MLP_SYNTHETIC_SUMMARY")
    else:
        for task in parse_csv(args.synthetic_tasks):
            for seed in parse_ints(args.synthetic_seeds):
                for loss_interface in parse_csv(args.loss_interfaces):
                    xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
                    for method in methods + mlp_methods:
                        family = "MLP" if method.startswith("M") else "D-CHE"
                        result = train_model_case(
                            family=family,
                            candidate_id=str(args.dche_candidate),
                            method=method,
                            dataset="synthetic",
                            task=task,
                            seed=seed,
                            loss_interface=loss_interface,
                            xtr=xtr,
                            ytr=ytr,
                            xva=xva,
                            yva=yva,
                            xte=None,
                            yte=None,
                            input_dim=int(args.synthetic_dim),
                            output_dim=int(args.synthetic_classes),
                            args=args,
                            device=device,
                            real_linec=False,
                        )
                        synthetic_raw.append(result["row"])
                        degree_rows.extend(result["degree_rows"])
                        projection_rows.extend(result["projection_rows"])
                        linec_tail_rows.extend(result["linec_rows"])
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
        synthetic_enriched = enrich_rows(synthetic_raw, "V1410_DCHE_SYNTHETIC_RESULT")
        dche_synth = [r for r in synthetic_enriched if r.get("family") == "D-CHE"]
        mlp_synth = [r for r in synthetic_enriched if r.get("family") == "MLP"]
        synth_summary = summarize_methods(dche_synth, "V1410_SYNTHETIC_SUMMARY")
        mlp_summary = summarize_methods(mlp_synth, "V1410_MLP_SYNTHETIC_SUMMARY")
    synth_pass, _task_pass = family_task_pass(dche_synth)

    fallback_rows: list[dict[str, Any]] = []
    if synthetic_source_dir is None and synth_pass < 5 and int(args.run_fallbacks):
        for task in parse_csv(args.synthetic_tasks):
            for seed in parse_ints(args.synthetic_seeds):
                for loss_interface in parse_csv(args.loss_interfaces):
                    xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
                    for method in parse_csv(args.fallback_methods):
                        result = train_model_case(
                            family="D-CHE",
                            candidate_id=str(args.dche_candidate),
                            method=method,
                            dataset="synthetic",
                            task=task,
                            seed=seed,
                            loss_interface=loss_interface,
                            xtr=xtr,
                            ytr=ytr,
                            xva=xva,
                            yva=yva,
                            xte=None,
                            yte=None,
                            input_dim=int(args.synthetic_dim),
                            output_dim=int(args.synthetic_classes),
                            args=args,
                            device=device,
                            real_linec=False,
                        )
                        fallback_rows.append(result["row"])
                        degree_rows.extend(result["degree_rows"])
                        projection_rows.extend(result["projection_rows"])
                        linec_tail_rows.extend(result["linec_rows"])
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
        fb_enriched = enrich_rows(fallback_rows + [r for r in synthetic_raw if r.get("method") in CONTROL_METHODS], "V1410_DCHE_SYNTHETIC_FALLBACK_RESULT")
        dche_synth.extend([r for r in fb_enriched if str(r.get("method")).startswith("F-CHE-FB")])
        synth_summary.extend(summarize_methods([r for r in fb_enriched if str(r.get("method")).startswith("F-CHE-FB")], "V1410_SYNTHETIC_FALLBACK_SUMMARY"))
        synth_pass, _task_pass = family_task_pass(dche_synth)

    real_enriched: list[dict[str, Any]] = []
    real_summary: list[dict[str, Any]] = []
    if synth_pass >= 5 and int(args.skip_real) == 0:
        real_raw: list[dict[str, Any]] = []
        for dataset in parse_csv(args.datasets):
            for seed in parse_ints(args.seeds):
                xtr, ytr, xva, yva, xte, yte, input_dim_t, output_dim_t = v144.load_real_split(args, dataset, seed, device)
                input_dim = int(input_dim_t.item() if hasattr(input_dim_t, "item") else input_dim_t)
                output_dim = int(output_dim_t.item() if hasattr(output_dim_t, "item") else output_dim_t)
                for method in methods:
                    result = train_model_case(
                        family="D-CHE",
                        candidate_id=str(args.dche_candidate),
                        method=method,
                        dataset=dataset,
                        task="real",
                        seed=seed,
                        loss_interface="CE",
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
                        real_linec=True,
                    )
                    real_raw.append(result["row"])
                    degree_rows.extend(result["degree_rows"])
                    projection_rows.extend(result["projection_rows"])
                    linec_tail_rows.extend(result["linec_rows"])
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        real_enriched = enrich_rows(real_raw, "V1410_DCHE_REAL_RESULT")
        real_summary = summarize_methods(real_enriched, "V1410_REAL_SUMMARY")

    failures = failure_taxonomy(dche_synth + real_enriched)
    progress = synth_summary + real_summary + mlp_summary
    write_rows(out_dir / "v1410_dche_fms_synthetic_results.csv", dche_synth)
    write_rows(out_dir / "v1410_dche_fms_synthetic_summary.csv", synth_summary)
    write_rows(out_dir / "v1410_dche_fms_real_results.csv", real_enriched)
    write_rows(out_dir / "v1410_dche_fms_real_summary.csv", real_summary)
    write_rows(out_dir / "v1410_dche_fms_controls.csv", [r for r in dche_synth + real_enriched if sint(r.get("control_method"), 0) == 1])
    write_rows(out_dir / "v1410_mlp_generic_controls.csv", mlp_synth + mlp_summary)
    write_rows(out_dir / "v1410_dche_degree_telemetry.csv", degree_rows)
    write_rows(out_dir / "v1410_dche_projection_retention.csv", projection_rows)
    write_rows(out_dir / "v1410_linec_tail_audit.csv", linec_tail_rows)
    write_rows(out_dir / "v1410_failure_taxonomy.csv", failures)
    write_rows(out_dir / "v1410_progress_table.csv", progress)
    write_code_packet(out_dir)
    route = build_route(dche_synth, real_enriched, mlp_synth, substrate_rows, forbidden_count, 999, args)
    write_boundaries(out_dir, route, failures)
    write_figures(out_dir, route, progress, substrate_rows, failures)
    missing = write_required_manifest(out_dir)
    route = build_route(dche_synth, real_enriched, mlp_synth, substrate_rows, forbidden_count, missing, args)
    write_json(out_dir / "v1410_route_decision.json", route)
    write_boundaries(out_dir, route, failures)
    write_figures(out_dir, route, progress, substrate_rows, failures)
    missing = write_required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    route["promotion_allowed"] = int(route["route"] == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0 and missing == 0 and forbidden_count == 0)
    write_json(out_dir / "v1410_route_decision.json", route)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.10 Non-RAT FMS Transfer + FMS definition reset")
    parser.add_argument("--out-dir", default=str(ROOT / "official_v1410"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    parser.add_argument("--synthetic-seeds", default="0,1,2")
    parser.add_argument("--loss-interfaces", default="CE,Brier")
    parser.add_argument("--synthetic-train-size", type=int, default=96)
    parser.add_argument("--synthetic-val-size", type=int, default=64)
    parser.add_argument("--synthetic-dim", type=int, default=16)
    parser.add_argument("--synthetic-classes", type=int, default=3)
    parser.add_argument("--synthetic-source-dir", default="", help="Reuse an already-passed v14.10 S3 synthetic artifact directory before opening real 3x3.")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--methods", default=",".join(list(CONTROL_METHODS) + list(F_CHE_METHODS)))
    parser.add_argument("--dche-candidate", default=DEFAULT_D_CHE_CANDIDATE)
    parser.add_argument("--fallback-methods", default=",".join(FALLBACK_METHODS))
    parser.add_argument("--mlp-methods", default=",".join(MLP_METHODS))
    parser.add_argument("--skip-mlp-controls", action="store_true")
    parser.add_argument("--skip-real", type=int, default=0)
    parser.add_argument("--run-fallbacks", type=int, default=1)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--fms-beta", type=float, default=0.99)
    parser.add_argument("--fms-strength", type=float, default=0.05)
    parser.add_argument("--fms-update-interval", type=int, default=80)
    parser.add_argument("--streaming-per-example-gradients", type=int, default=0)
    parser.add_argument("--trace-interval", type=int, default=100)
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
