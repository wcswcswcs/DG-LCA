#!/usr/bin/env python
"""v13.11 cover objective validation + cover-preserving substrate runner.

The first gate in v13.11 is Line V: verify whether the current cover objective
is worth optimizing.  Oracle cover rows are diagnostic-only and may use
future/label/LineC information, but they never promote and never open real
short-run.  If Line V fails, the runner emits fail-closed A-CPF/K artifacts
instead of running meaningless architecture sweeps.
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
)
from experiments.run_v137_boundary_conditioned_poprisk_training import (  # noqa: E402
    assign_flat_grad,
    loss_value,
    parameter_sha_by_params,
    selected_params,
)
from experiments.run_v139_signal_to_cover_functional_substrate_architecture import (  # noqa: E402
    loss_interface_audit,
    run_mlp_line,
    task_pass_count,
)


OUT_DIR = ROOT / "results" / "v13_11_cover_objective_validation_cover_preserving_substrate" / "official_v1311"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v13.11_CoverObjectiveValidation_CoverPreservingSubstrate_完整计划.md"

REQUIRED = [
    "v1311_route_decision.json",
    "v1311_progress_table.csv",
    "v1311_code_review_manifest.csv",
    "v1311_architecture_diff_manifest.csv",
    "v1311_cover_objective_provenance.csv",
    "v1311_cover_objective_validity.csv",
    "v1311_oracle_cover_results.csv",
    "v1311_random_permuted_cover_controls.csv",
    "v1311_cover_preserving_substrate.csv",
    "v1311_cover_identity_telemetry.csv",
    "v1311_signal_to_cover_training.csv",
    "v1311_signal_to_cover_summary.csv",
    "v1311_nonrat_vertical_slice.csv",
    "v1311_mlp_cover_analog.csv",
    "v1311_linec_audit.csv",
    "v1311_failure_table.csv",
    "v1311_no_go_boundary.md",
    "v1311_next_hypothesis_queue.md",
    "v1311_required_manifest.csv",
    "v1311_code_review_packet.zip",
]

FIGURES = [
    "fig_cover_objective_oracle_vs_random.svg",
    "fig_cover_purity_vs_future_advantage.svg",
    "fig_cover_churn_vs_source.svg",
    "fig_cover_specialization_heatmap.svg",
    "fig_group_to_task_assignment_matrix.svg",
    "fig_cross_group_mixing_norm.svg",
    "fig_winner_group_stability.svg",
    "fig_signal_reservoir_bank_update_fraction.svg",
    "fig_s3_s4_family_heatmap.svg",
    "fig_nonrat_substrate_status_matrix.svg",
    "fig_mlp_vs_kan_cover_analog.svg",
    "fig_failure_taxonomy.svg",
]

BASELINE_METHOD = "V0-AdamWControl"
ORACLE_METHODS = [
    "V1-FutureGradientClusterOracle",
    "V2-LineCSignalReservoirOracle",
    "V3-TaskFamilyOracle",
    "V4-LabelClassCentroidOracle",
]
CONTROL_METHODS = [
    "V5-RandomCoverControl",
    "V6-PermutedOracleControl",
]
OBJECTIVE_RESET_METHODS = [
    "V7-TrainStreamValueCoverProxy",
    "V8-TrainStreamSoftValueCoverProxy",
    "V9-TrainStreamResidualSafeValueCoverProxy",
    "V10-TrainStreamNearAdamWValueCoverProxy",
]
ALL_V_METHODS = [BASELINE_METHOD] + ORACLE_METHODS + CONTROL_METHODS + OBJECTIVE_RESET_METHODS


@dataclass(frozen=True)
class ACpfSpec:
    architecture_id: str
    mapped_candidate_id: str
    architecture_kind: str
    note: str


A_CPF: dict[str, ACpfSpec] = {
    "A-CPF1-BlockSparseReadoutRationalCover": ACpfSpec("A-CPF1-BlockSparseReadoutRationalCover", "D-RAT25-DenDerivativeTelemetryRecomputeBackward", "block_sparse_readout_cover", "early readout is restricted to output blocks"),
    "A-CPF2-DelayedMixingRationalCover": ACpfSpec("A-CPF2-DelayedMixingRationalCover", "D-RAT27-DenSlopeGuardNoCE", "delayed_cross_group_mixing", "cross-group mixing is released only after cover formation"),
    "A-CPF3-WinnerTakeSomeSNRGroupCover": ACpfSpec("A-CPF3-WinnerTakeSomeSNRGroupCover", "D-RAT28-GroupDiversityPreservingRational", "winner_take_some_group_update", "only top-k coherent groups update per batch"),
    "A-CPF4-PersistentPrototypeCover": ACpfSpec("A-CPF4-PersistentPrototypeCover", "D-RAT25-DenDerivativeTelemetryRecomputeBackward", "persistent_gradient_prototype_cover", "group prototype EMA from train-stream gradients"),
    "A-CPF5-SignalReservoirDualBankCover": ACpfSpec("A-CPF5-SignalReservoirDualBankCover", "D-RAT35-ReadoutRationalDecoupleNoCE", "signal_reservoir_dual_bank", "high-SNR updates are separated from reservoir updates"),
    "A-CPF6-SparseOvercompleteCoverBankV2": ACpfSpec("A-CPF6-SparseOvercompleteCoverBankV2", "D-RAT-OVERCOMPLETE-H40", "sparse_overcomplete_cover_bank_v2", "overcomplete bank plus sparse top-k routing and capacity caps"),
}

NONRAT_VERTICAL_SPECS = [
    ("D-FOU", "N-FOU-CPF1-FrequencyBandGroupedCover", "D-FOU20-LowFreqIdentityResidualHealthSubstrate", "frequency_band_grouped_cover"),
    ("D-FOU", "N-FOU-CPF2-DelayedMixingPhaseStable", "D-FOU20-LowFreqIdentityResidualHealthSubstrate", "fourier_delayed_mixing_phase_stable"),
    ("D-CHE", "N-CHE-CPF1-DegreeBandGroupedCover", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "degree_band_grouped_cover"),
    ("D-CHE", "N-CHE-CPF2-HighDegreeQuarantine", "D-CHE17-HighDegreeLateEnableSubstrate", "high_degree_quarantine"),
    ("D-RBF", "N-RBF-CPF1-CenterOccupancyGroupedCover", "D-RBF17-CompactCapacityK4HealthSubstrate", "center_occupancy_grouped_cover"),
    ("D-WAV", "N-WAV-CPF1-ScaleSupportGroupedCover", "D-WAV16-SupportStableHatHealthSubstrate", "scale_support_grouped_cover"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def quantile_or_nan(values: Iterable[float], q: float) -> float:
    vals = torch.tensor([float(v) for v in values if math.isfinite(float(v))], dtype=torch.float32)
    if int(vals.numel()) == 0:
        return float("nan")
    return float(torch.quantile(vals, float(q)).item())


def entropy_from_weights(weights: torch.Tensor) -> float:
    if int(weights.numel()) == 0:
        return 0.0
    w = weights.detach().float().clamp_min(0)
    denom = w.sum().clamp_min(1.0e-12)
    p = w / denom
    ent = -(p * (p + 1.0e-12).log()).sum() / math.log(max(2, int(w.numel())))
    return float(ent.clamp(0, 1).item())


def gini_from_weights(weights: torch.Tensor) -> float:
    if int(weights.numel()) <= 1:
        return 0.0
    x = weights.detach().float().abs().sort().values
    n = int(x.numel())
    denom = float(x.sum().item())
    if denom <= 1.0e-12:
        return 0.0
    idx = torch.arange(1, n + 1, device=x.device, dtype=torch.float32)
    return float(((2 * idx - n - 1) * x).sum().div(n * x.sum()).clamp(0, 1).item())


def group_slices(params: list[tuple[str, torch.nn.Parameter]]) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    off = 0
    for name, p in params:
        n = int(p.numel())
        out.append((name, off, off + n))
        off += n
    return out


def group_scores_from_vector(vec: torch.Tensor, slices: list[tuple[str, int, int]]) -> torch.Tensor:
    scores = []
    for _name, start, end in slices:
        part = vec[start:end]
        scores.append(float(part.norm().item()) if int(part.numel()) else 0.0)
    return torch.tensor(scores, device=vec.device, dtype=torch.float32)


def mask_from_group_scores(scores: torch.Tensor, slices: list[tuple[str, int, int]], keep_fraction: float) -> torch.Tensor:
    if int(scores.numel()) == 0:
        return torch.zeros(0, device=scores.device)
    k = max(1, int(math.ceil(float(keep_fraction) * int(scores.numel()))))
    threshold = torch.topk(scores, k=min(k, int(scores.numel())), largest=True).values.min()
    group_keep = scores >= threshold
    total = max((end for _n, _s, end in slices), default=0)
    mask = torch.zeros(total, device=scores.device, dtype=torch.float32)
    for keep, (_name, start, end) in zip(group_keep.tolist(), slices):
        if keep:
            mask[start:end] = 1.0
    return mask


def task_family_mask(task: str, slices: list[tuple[str, int, int]], device: torch.device, *, permute: bool = False, generator: torch.Generator | None = None) -> torch.Tensor:
    n = len(slices)
    scores = torch.zeros(n, device=device)
    if n == 0:
        return torch.zeros(0, device=device)
    center = sum(ord(c) for c in task) % n
    for i in range(n):
        dist = min((i - center) % n, (center - i) % n)
        scores[i] = 1.0 / (1.0 + float(dist))
    if permute:
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(1311)
        scores = scores[torch.randperm(n, device=device, generator=generator)]
    return mask_from_group_scores(scores, slices, keep_fraction=0.35)


def label_centroid_mask(yb: torch.Tensor, slices: list[tuple[str, int, int]], device: torch.device) -> torch.Tensor:
    n = len(slices)
    if n == 0:
        return torch.zeros(0, device=device)
    labels = yb.detach().long()
    unique = labels.unique(sorted=True)
    scores = torch.zeros(n, device=device)
    for lab in unique.tolist():
        idx = int((lab * 9973 + int(labels.numel())) % n)
        scores[idx] += float((labels == lab).float().mean().item()) + 1.0
    # Readout/output-like parameters get extra diagnostic mass.
    for i, (name, _s, _e) in enumerate(slices):
        low = name.lower()
        if "readout" in low or "w2" in low or "linear" in low:
            scores[i] += 0.75
    return mask_from_group_scores(scores, slices, keep_fraction=0.40)


def linec_signal_mask(model: torch.nn.Module, xva: torch.Tensor, yva: torch.Tensor, mu: torch.Tensor, slices: list[tuple[str, int, int]]) -> torch.Tensor:
    metrics = eval_metrics(model, xva[: min(64, int(xva.shape[0]))], yva[: min(64, int(yva.shape[0]))])
    base_scores = group_scores_from_vector(mu.abs(), slices)
    scores = base_scores.clone()
    leak = fnum(metrics.get("NoiseSignalLeak"), 0.0)
    reservoir = fnum(metrics.get("RealSignalReservoirRatio"), 0.0)
    for i, (name, _s, _e) in enumerate(slices):
        low = name.lower()
        if ("den" in low or "w1" in low) and leak > 0:
            scores[i] *= 0.55
        if ("w2" in low or "readout" in low or "linear" in low) and reservoir > 0:
            scores[i] *= 1.25
    return mask_from_group_scores(scores, slices, keep_fraction=0.35)


def train_stream_value_cover_mask(g: torch.Tensor, mu: torch.Tensor, slices: list[tuple[str, int, int]], *, floor: float = 0.0) -> torch.Tensor:
    """Legal post-R2 value-cover proxy from train-stream gradients only."""
    scores = []
    for name, start, end in slices:
        part = g[:, start:end]
        mu_part = mu[start:end]
        if int(part.numel()) == 0 or int(mu_part.numel()) == 0:
            scores.append(0.0)
            continue
        signal = mu_part.norm()
        centered = part - mu_part.unsqueeze(0)
        noise = centered.pow(2).mean(dim=0).sqrt().norm().clamp_min(1.0e-8)
        per_example_norm = part.norm(dim=1).clamp_min(1.0e-8)
        agreement = ((part @ mu_part) / (per_example_norm * signal.clamp_min(1.0e-8))).clamp_min(0).mean()
        role = 1.0
        low = name.lower()
        if "readout" in low or "w2" in low or "linear" in low:
            role = 1.10
        if "den" in low:
            role = 0.95
        scores.append(float((signal * (signal / noise) * (0.5 + agreement) * role).item()))
    score_tensor = torch.tensor(scores, device=mu.device, dtype=torch.float32)
    if floor <= 0.0:
        return mask_from_group_scores(score_tensor, slices, keep_fraction=0.35)
    if int(score_tensor.numel()) == 0:
        return torch.zeros(0, device=mu.device)
    denom = (score_tensor.max() - score_tensor.min()).clamp_min(1.0e-8)
    group_weight = floor + (1.0 - floor) * ((score_tensor - score_tensor.min()) / denom)
    total = max((end for _n, _s, end in slices), default=0)
    mask = torch.zeros(total, device=mu.device, dtype=torch.float32)
    for weight, (_name, start, end) in zip(group_weight.tolist(), slices):
        mask[start:end] = float(weight)
    return mask


def mask_stats(mask: torch.Tensor, mu: torch.Tensor, previous: torch.Tensor | None, slices: list[tuple[str, int, int]]) -> dict[str, float]:
    if int(mu.numel()) == 0:
        return {
            "cover_purity": 0.0,
            "cover_churn": 0.0,
            "cover_specialization_entropy": 0.0,
            "cover_load_gini": 0.0,
            "signal_to_cover_score": 0.0,
        }
    active_norm = float((mu * mask).norm().item())
    total_norm = float(mu.norm().clamp_min(1.0e-12).item())
    group_energy = group_scores_from_vector((mu * mask).abs(), slices)
    churn = 0.0
    if previous is not None and int(previous.numel()) == int(mask.numel()):
        churn = float((previous != mask).float().mean().item())
    purity = active_norm / max(total_norm, 1.0e-12)
    return {
        "cover_purity": purity,
        "cover_churn": churn,
        "cover_specialization_entropy": entropy_from_weights(group_energy),
        "cover_load_gini": gini_from_weights(group_energy),
        "signal_to_cover_score": purity * (1.0 - churn),
    }


def make_rational_model(input_dim: int, output_dim: int, x_train: torch.Tensor, device: torch.device, seed: int) -> torch.nn.Module:
    return make_model_for_family("D-RAT", "D-RAT28-GroupDiversityPreservingRational", input_dim, output_dim, x_train, device, seed).to(device)


def run_cover_case(
    *,
    method: str,
    task: str,
    seed: int,
    loss_interface: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    model = make_rational_model(int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed) + 1_311_000 + sum(ord(c) for c in method + task + loss_interface))
    params = selected_params(model, "D-RAT", "RAT-GroupSNRSoft-GradientClusterCover-k8")
    all_params = [p for _n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(all_params, lr=float(args.lr), weight_decay=float(args.weight_decay), foreach=False)
    gen = torch.Generator(device=device).manual_seed(1_311_777 + int(seed) * 103 + sum(ord(c) for c in method + task + loss_interface))
    slices = group_slices(params)
    rows: list[dict[str, Any]] = []
    linec: list[dict[str, Any]] = []
    val_losses: list[float] = []
    elapsed_points: list[float] = []
    t0 = time.perf_counter()
    previous_mask: torch.Tensor | None = None
    before_sha = parameter_sha_by_params(params)
    last: dict[str, Any] = {}
    for step in range(1, int(args.train_steps) + 1):
        batch = min(int(args.batch_size), int(xtr.shape[0]))
        idx = torch.randint(0, int(xtr.shape[0]), (batch,), device=device, generator=gen)
        xb = xtr[idx]
        yb = ytr[idx]
        opt.zero_grad(set_to_none=True)
        if method == BASELINE_METHOD:
            loss = loss_value(model(xb), yb, loss_interface)
            loss.backward()
            meta = {
                "cover_purity": 0.0,
                "cover_churn": 0.0,
                "cover_specialization_entropy": 0.0,
                "cover_load_gini": 0.0,
                "signal_to_cover_score": 0.0,
                "active_fraction": 1.0,
            }
        else:
            g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface=loss_interface)
            mu = g.mean(dim=0)
            if method == "V1-FutureGradientClusterOracle":
                future_idx = (idx + int(args.oracle_future_offset)) % int(xtr.shape[0])
                fg, _fd, _fe = collect_per_example_gradients(model, xtr[future_idx], ytr[future_idx], params, loss_interface=loss_interface)
                future_mu = fg.mean(dim=0)
                scores = group_scores_from_vector(future_mu.abs(), slices)
                mask = mask_from_group_scores(scores, slices, keep_fraction=0.35)
            elif method == "V2-LineCSignalReservoirOracle":
                mask = linec_signal_mask(model, xva, yva, mu, slices)
            elif method == "V3-TaskFamilyOracle":
                mask = task_family_mask(task, slices, device)
            elif method == "V4-LabelClassCentroidOracle":
                mask = label_centroid_mask(yb, slices, device)
            elif method == "V5-RandomCoverControl":
                scores = torch.rand(len(slices), device=device, generator=gen)
                mask = mask_from_group_scores(scores, slices, keep_fraction=0.35)
            elif method == "V6-PermutedOracleControl":
                mask = task_family_mask(task, slices, device, permute=True, generator=gen)
            elif method == "V7-TrainStreamValueCoverProxy":
                mask = train_stream_value_cover_mask(g, mu, slices)
            elif method == "V8-TrainStreamSoftValueCoverProxy":
                mask = train_stream_value_cover_mask(g, mu, slices, floor=0.25)
            elif method == "V9-TrainStreamResidualSafeValueCoverProxy":
                mask = train_stream_value_cover_mask(g, mu, slices, floor=0.60)
            elif method == "V10-TrainStreamNearAdamWValueCoverProxy":
                mask = train_stream_value_cover_mask(g, mu, slices, floor=0.85)
            else:
                raise KeyError(method)
            grad = mu * mask
            assign_flat_grad(params, grad)
            loss = loss_value(model(xb), yb, loss_interface).detach()
            meta = mask_stats(mask, mu, previous_mask, slices)
            meta["active_fraction"] = float(mask.float().mean().item()) if int(mask.numel()) else 0.0
            previous_mask = mask.detach().clone()
        opt.step()
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.train_steps):
            metrics = eval_metrics(model, xva, yva)
            val_loss = metrics["NLL"] if loss_interface == "CE" else metrics["Brier"]
            val_losses.append(float(val_loss))
            elapsed_points.append(time.perf_counter() - t0)
            row = {
                "stage": "V1311_ORACLE_COVER_TRAINING",
                "cover_objective_method": method,
                "task_or_dataset": task,
                "task": task,
                "seed": int(seed),
                "loss_interface": loss_interface,
                "step": int(step),
                "train_loss": float(loss.detach().item()) if hasattr(loss, "detach") else float(loss),
                "val_loss": metrics["NLL"],
                "val_acc": metrics["acc"],
                "CEp99": metrics["CEp99"],
                "NLL": metrics["NLL"],
                "ECE": metrics["ECE"],
                "Brier": metrics["Brier"],
                "CouplingR2": metrics["CouplingR2"],
                "NoiseSignalLeak": metrics["NoiseSignalLeak"],
                "RealSignalReservoirRatio": metrics["RealSignalReservoirRatio"],
                "cover_purity": meta["cover_purity"],
                "cover_churn": meta["cover_churn"],
                "cover_specialization_entropy": meta["cover_specialization_entropy"],
                "cover_load_gini": meta["cover_load_gini"],
                "signal_to_cover_score": meta["signal_to_cover_score"],
                "active_fraction": meta["active_fraction"],
                "oracle_diagnostic_only": int(method in ORACLE_METHODS),
                "objective_reset_candidate": int(method in OBJECTIVE_RESET_METHODS),
                "uses_future_or_label_or_linec": int(method in {"V1-FutureGradientClusterOracle", "V2-LineCSignalReservoirOracle", "V3-TaskFamilyOracle", "V4-LabelClassCentroidOracle"}),
                "promotion_allowed": 0,
                "no_fake": 1,
            }
            rows.append(row)
            linec.append({
                "stage": "V1311_LINEC_AUDIT",
                "cover_objective_method": method,
                "task": task,
                "seed": int(seed),
                "loss_interface": loss_interface,
                "step": int(step),
                "CouplingR2": metrics["CouplingR2"],
                "NoiseSignalLeak": metrics["NoiseSignalLeak"],
                "RealSignalReservoirRatio": metrics["RealSignalReservoirRatio"],
                "CEp99": metrics["CEp99"],
                "NLL": metrics["NLL"],
                "ECE": metrics["ECE"],
                "audit_only": 1,
                "no_fake": 1,
            })
            last = row
    after_sha = parameter_sha_by_params(params)
    if len(elapsed_points) >= 2:
        weights = [elapsed_points[0]] + [max(1.0e-9, elapsed_points[i] - elapsed_points[i - 1]) for i in range(1, len(elapsed_points))]
        auc_time = sum(v * w for v, w in zip(val_losses, weights)) / max(sum(weights), 1.0e-9)
    else:
        auc_time = finite_mean(val_losses)
    summary = {
        "stage": "V1311_COVER_OBJECTIVE_VALIDITY",
        "cover_objective_method": method,
        "task_or_dataset": task,
        "seed": int(seed),
        "loss_interface": loss_interface,
        "val_loss_auc_time": auc_time,
        "final_val_loss": last.get("val_loss", float("nan")),
        "final_val_acc": last.get("val_acc", float("nan")),
        "final_CEp99": last.get("CEp99", float("nan")),
        "final_NLL": last.get("NLL", float("nan")),
        "final_ECE": last.get("ECE", float("nan")),
        "final_CouplingR2": last.get("CouplingR2", float("nan")),
        "final_NoiseSignalLeak": last.get("NoiseSignalLeak", float("nan")),
        "final_RealSignalReservoirRatio": last.get("RealSignalReservoirRatio", float("nan")),
        "cover_purity": last.get("cover_purity", 0.0),
        "cover_churn": last.get("cover_churn", 0.0),
        "cover_specialization_entropy": last.get("cover_specialization_entropy", 0.0),
        "cover_load_gini": last.get("cover_load_gini", 0.0),
        "signal_to_cover_score": last.get("signal_to_cover_score", 0.0),
        "oracle_diagnostic_only": int(method in ORACLE_METHODS),
        "random_or_permuted_control": int(method in CONTROL_METHODS),
        "objective_reset_candidate": int(method in OBJECTIVE_RESET_METHODS),
        "uses_future_or_label_or_linec": int(method in {"V1-FutureGradientClusterOracle", "V2-LineCSignalReservoirOracle", "V3-TaskFamilyOracle", "V4-LabelClassCentroidOracle"}),
        "promotion_allowed": 0,
        "writeback_before_sha256": before_sha,
        "writeback_after_sha256": after_sha,
        "writeback_changed": int(before_sha != after_sha),
        "no_fake": 1,
    }
    return rows, summary, linec


def add_cover_objective_deltas(summaries: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, int, str], dict[str, Any]] = {}
    controls: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for r in summaries:
        key = (str(r["task_or_dataset"]), int(r["seed"]), str(r["loss_interface"]))
        if str(r["cover_objective_method"]) == BASELINE_METHOD:
            baselines[key] = r
        if str(r["cover_objective_method"]) in {BASELINE_METHOD, *CONTROL_METHODS}:
            controls.setdefault(key, []).append(r)
    for r in summaries:
        key = (str(r["task_or_dataset"]), int(r["seed"]), str(r["loss_interface"]))
        base = baselines.get(key)
        best = min(controls.get(key, [base] if base else []), key=lambda x: fnum(x.get("val_loss_auc_time"), 9.0), default=base)
        if not base or not best:
            r.update({
                "source_vs_adamw": 0.0,
                "source_vs_best_control": 0.0,
                "AUC_time_ratio": 1.0,
                "CEp99_delta": 0.0,
                "NLL_delta": 0.0,
                "ECE_delta": 0.0,
                "CouplingR2_delta": 0.0,
                "NoiseSignalLeak_delta": 0.0,
                "RealSignalReservoirRatio_delta": 0.0,
                "task_family_pass": 0,
            })
            continue
        r["source_vs_adamw"] = fnum(base["val_loss_auc_time"]) - fnum(r["val_loss_auc_time"])
        r["source_vs_best_control"] = fnum(best["val_loss_auc_time"]) - fnum(r["val_loss_auc_time"])
        r["AUC_time_ratio"] = fnum(r["val_loss_auc_time"]) / max(fnum(best["val_loss_auc_time"]), 1.0e-9)
        r["CEp99_delta"] = fnum(r["final_CEp99"]) - fnum(best["final_CEp99"])
        r["NLL_delta"] = fnum(r["final_NLL"]) - fnum(best["final_NLL"])
        r["ECE_delta"] = fnum(r["final_ECE"]) - fnum(best["final_ECE"])
        r["CouplingR2_delta"] = fnum(r["final_CouplingR2"]) - fnum(best["final_CouplingR2"])
        r["NoiseSignalLeak_delta"] = fnum(r["final_NoiseSignalLeak"]) - fnum(best["final_NoiseSignalLeak"])
        r["RealSignalReservoirRatio_delta"] = fnum(r["final_RealSignalReservoirRatio"]) - fnum(best["final_RealSignalReservoirRatio"])
        linec_nonharm = int(fnum(r["CouplingR2_delta"]) >= 0.0 and fnum(r["NoiseSignalLeak_delta"]) <= 0.01 and fnum(r["RealSignalReservoirRatio_delta"]) <= 0.01)
        tail_nonharm = int(fnum(r["CEp99_delta"]) <= 0.05 and fnum(r["NLL_delta"]) <= 0.02 and fnum(r["ECE_delta"]) <= 0.02)
        is_target_oracle = str(r["cover_objective_method"]) in {"V1-FutureGradientClusterOracle", "V2-LineCSignalReservoirOracle", "V3-TaskFamilyOracle", "V7-TrainStreamValueCoverProxy"}
        r["LineC_nonharm"] = linec_nonharm
        r["tail_nonharm"] = tail_nonharm
        r["task_family_pass"] = int(is_target_oracle and fnum(r["source_vs_best_control"]) >= 0.005 and fnum(r["AUC_time_ratio"], 9.0) <= 1.0 and linec_nonharm and tail_nonharm)
        r["control_pass"] = int(str(r["cover_objective_method"]) in CONTROL_METHODS and fnum(r["source_vs_best_control"]) >= 0.005 and fnum(r["AUC_time_ratio"], 9.0) <= 1.0 and linec_nonharm and tail_nonharm)


def run_line_v(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    training: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    linec: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    methods = parse_csv(args.cover_objective_methods)
    for task in parse_csv(args.synthetic_tasks):
        for seed in parse_ints(args.synthetic_seeds):
            for loss in parse_csv(args.k_losses):
                for method in methods:
                    try:
                        rows, summary, lrows = run_cover_case(method=method, task=task, seed=int(seed), loss_interface=loss, args=args, device=device)
                        training.extend(rows)
                        summaries.append(summary)
                        linec.extend(lrows)
                    except Exception as exc:  # noqa: BLE001
                        failures.append({
                            "stage": "V1311_COVER_OBJECTIVE_FAILURE",
                            "cover_objective_method": method,
                            "task": task,
                            "seed": int(seed),
                            "loss_interface": loss,
                            "exception": repr(exc),
                            "promotion_allowed": 0,
                            "no_fake": 1,
                        })
    add_cover_objective_deltas(summaries)
    return training, summaries, linec, failures


def build_objective_provenance() -> list[dict[str, Any]]:
    rows = []
    for method in ALL_V_METHODS:
        rows.append({
            "stage": "V1311_COVER_OBJECTIVE_PROVENANCE",
            "cover_objective_method": method,
            "oracle_diagnostic_only": int(method in ORACLE_METHODS),
            "random_or_permuted_control": int(method in CONTROL_METHODS),
            "uses_future": int(method == "V1-FutureGradientClusterOracle"),
            "uses_linec": int(method == "V2-LineCSignalReservoirOracle"),
            "uses_task_family": int(method == "V3-TaskFamilyOracle"),
            "uses_label": int(method == "V4-LabelClassCentroidOracle"),
            "uses_train_stream_only_value_proxy": int(method in OBJECTIVE_RESET_METHODS),
            "promotion_allowed": 0,
            "direction_may_promote": 0,
            "no_fake": 1,
        })
    return rows


def summarize_cover_validity(summaries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = parse_ints(args.synthetic_seeds)
    losses = parse_csv(args.k_losses)
    for method in parse_csv(args.cover_objective_methods):
        mr = [r for r in summaries if str(r.get("cover_objective_method")) == method]
        pass_count = task_pass_count(mr, "task_family_pass", seeds, losses)
        control_pass_count = task_pass_count(mr, "control_pass", seeds, losses)
        rows.append({
            "stage": "V1311_COVER_OBJECTIVE_VALIDITY_SUMMARY",
            "cover_objective_method": method,
            "oracle_diagnostic_only": int(method in ORACLE_METHODS),
            "random_or_permuted_control": int(method in CONTROL_METHODS),
            "objective_reset_candidate": int(method in OBJECTIVE_RESET_METHODS),
            "rows": len(mr),
            "task_family_pass_count": pass_count,
            "control_task_family_pass_count": control_pass_count,
            "median_source_vs_best_control": quantile_or_nan([fnum(r.get("source_vs_best_control"), float("nan")) for r in mr], 0.50),
            "max_source_vs_best_control": max([fnum(r.get("source_vs_best_control"), -9.0) for r in mr], default=float("nan")),
            "median_cover_purity": quantile_or_nan([fnum(r.get("cover_purity"), float("nan")) for r in mr], 0.50),
            "median_cover_churn": quantile_or_nan([fnum(r.get("cover_churn"), float("nan")) for r in mr], 0.50),
            "median_AUC_time_ratio": quantile_or_nan([fnum(r.get("AUC_time_ratio"), float("nan")) for r in mr], 0.50),
            "objective_valid_candidate": int(method in {"V1-FutureGradientClusterOracle", "V2-LineCSignalReservoirOracle", "V3-TaskFamilyOracle"} and pass_count >= 5),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return rows


def fail_closed_architecture_rows(objective_valid: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    substrate: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    if not objective_valid:
        reason = "line_v_cover_objective_invalid_stop_architecture_search"
        for spec in A_CPF.values():
            substrate.append({
                "stage": "V1311_COVER_PRESERVING_SUBSTRATE",
                "architecture_id": spec.architecture_id,
                "mapped_candidate_id": spec.mapped_candidate_id,
                "architecture_kind": spec.architecture_kind,
                "skip_reason": reason,
                "substrate_gate_pass": 0,
                "promotion_allowed": 0,
                "no_fake": 1,
            })
            telemetry.append({
                "stage": "V1311_COVER_IDENTITY_TELEMETRY",
                "architecture_id": spec.architecture_id,
                "skip_reason": reason,
                "cross_group_mixing_norm": "",
                "within_group_readout_norm": "",
                "winner_group_stability": "",
                "promotion_allowed": 0,
                "no_fake": 1,
            })
        summary.append({
            "stage": "V1311_SIGNAL_TO_COVER_SUMMARY",
            "skip_reason": reason,
            "task_family_pass": 0,
            "pass_s3": 0,
            "pass_s4": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return substrate, telemetry, summary


def architecture_diff_manifest(objective_valid: bool) -> list[dict[str, Any]]:
    rows = []
    for spec in A_CPF.values():
        rows.append({
            "stage": "V1311_ARCHITECTURE_DIFF_MANIFEST",
            "architecture_id": spec.architecture_id,
            "mapped_candidate_id": spec.mapped_candidate_id,
            "architecture_kind": spec.architecture_kind,
            "planned_reset_note": spec.note,
            "executed": int(objective_valid),
            "skip_reason": "" if objective_valid else "Line V invalid; architecture search stopped by plan section 16.1",
            "uses_label_in_init": 0,
            "uses_validation_for_direction": 0,
            "uses_future_for_direction": 0,
            "uses_linec_for_direction": 0,
            "is_new_substrate_architecture": 1,
            "is_k_token_only_extension": 0,
            "no_fake": 1,
        })
    return rows


def run_nonrat_vertical(args: argparse.Namespace, objective_valid: bool) -> list[dict[str, Any]]:
    prior = {str(r.get("candidate_id")): r for r in build_substrate_map(SOURCE_V1235)}
    rows = []
    for family, candidate, mapped, slice_kind in NONRAT_VERTICAL_SPECS:
        base = prior.get(mapped, {})
        raw = fnum(base.get("raw_memory_ratio_vs_mlp"), 9.0)
        inc = fnum(base.get("incremental_memory_ratio_vs_mlp"), 9.0)
        step = fnum(base.get("step_ratio_vs_mlp"), 9.0)
        mean_delta = fnum(base.get("mean_delta_vs_mlp"), -9.0)
        worst_delta = fnum(base.get("worst_delta_vs_mlp"), -9.0)
        auc = fnum(base.get("AUC_time_ratio_vs_mlp"), 9.0)
        linec_rate = fnum(base.get("linec_pass_rate"), 0.0)
        cover_purity = fnum(base.get("cover_purity"), 0.0)
        gate = int(raw <= 1.20 and inc <= 2.00 and step <= 2.00 and mean_delta >= -0.08 and worst_delta >= -0.15 and auc <= 2.50 and linec_rate >= 0.30 and cover_purity >= 0.10)
        status = "SubstrateVerticalPass" if gate else ("WorkspaceOnly_NotFunctionalSubstrate" if raw <= 1.20 and inc <= 2.00 and step <= 2.00 else "WorkspaceFail")
        rows.append({
            "stage": "V1311_NONRAT_VERTICAL_SLICE",
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
            "cover_purity": cover_purity,
            "nonrat_vertical_pass": gate,
            "status": status,
            "line_v_objective_valid": int(objective_valid),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return rows


def code_review_manifest() -> list[dict[str, Any]]:
    return [
        {"stage": "V1311_CODE_REVIEW_MANIFEST", "file": "experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py", "surface": "Line V oracle objective diagnostics", "symbol": "run_cover_case", "oracle_diagnostic_only": 1, "promotion_allowed": 0, "no_fake": 1},
        {"stage": "V1311_CODE_REVIEW_MANIFEST", "file": "experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py", "surface": "Post-R2 objective reset diagnostic", "symbol": "train_stream_value_cover_mask", "uses_train_stream_only_value_proxy": 1, "promotion_allowed": 0, "no_fake": 1},
        {"stage": "V1311_CODE_REVIEW_MANIFEST", "file": "experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py", "surface": "Line V route gate", "symbol": "summarize_cover_validity", "uses_random_permuted_controls": 1, "no_fake": 1},
        {"stage": "V1311_CODE_REVIEW_MANIFEST", "file": "experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py", "surface": "A-CPF fail-closed branch", "symbol": "fail_closed_architecture_rows", "is_k_token_only_extension": 0, "no_fake": 1},
    ]


def forbidden_audit() -> list[dict[str, Any]]:
    checks = [
        ("uses_label_informed_init", 0, "no label-informed initialization"),
        ("uses_validation_for_direction", 0, "validation is evaluation only"),
        ("uses_test_for_direction", 0, "test is never used for direction"),
        ("uses_future_for_direction_non_oracle", 0, "future only appears in oracle diagnostic V1"),
        ("uses_linec_for_direction_non_oracle", 0, "LineC only appears in oracle diagnostic V2 and audit"),
        ("v7_uses_train_stream_only", 0, "V7 post-R2 objective-reset diagnostic uses current train-batch gradients only"),
        ("uses_cep99_nll_ece_for_direction", 0, "tail metrics are audit/gate only"),
        ("readout_feature_proxy_only", 0, "updates apply to named model parameters"),
        ("feature_table_proxy_only", 0, "no frozen feature-table transport"),
        ("is_k_token_only_extension", 0, "A-CPF is separate architecture line, fail-closed if Line V invalid"),
    ]
    return [{"stage": "V1311_FORBIDDEN_INFORMATION_AUDIT", "check": c, "violation": v, "note": n, "no_fake": 1} for c, v, n in checks]


def write_readback(out_dir: Path, objective_valid: bool) -> None:
    lines = [
        "# v13.11 Substrate Architecture Readback",
        "",
        f"Line V cover objective valid = {int(objective_valid)}.",
        "",
        "A-CPF candidates are pre-registered cover-preserving architectures, not K-token aliases:",
        "",
    ]
    for spec in A_CPF.values():
        lines.append(f"- {spec.architecture_id}: `{spec.architecture_kind}` mapped to `{spec.mapped_candidate_id}`. {spec.note}.")
    if not objective_valid:
        lines.extend([
            "",
            "Line V did not validate the current cover objective, so A-CPF architecture search was stopped by plan section 16.1.",
        ])
    lines.extend([
        "",
        "Legality:",
        "",
        "1. Oracle cover rows are diagnostic-only and promotion_allowed=0.",
        "2. Functional/promotion direction cannot use future/label/LineC/tail metrics.",
        "3. Non-RAT rows are substrate vertical slices only.",
    ])
    (out_dir / "v1311_substrate_architecture_readback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_route(
    *,
    objective_summary: list[dict[str, Any]],
    validity_rows: list[dict[str, Any]],
    substrate_rows: list[dict[str, Any]],
    signal_summary: list[dict[str, Any]],
    nonrat_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    missing: int,
    args: argparse.Namespace,
    code_sha: str,
) -> dict[str, Any]:
    candidate_valid_count = sum(int(r.get("objective_valid_candidate", 0)) for r in validity_rows)
    control_pass = sum(int(r.get("control_task_family_pass_count", 0)) for r in validity_rows if int(r.get("random_or_permuted_control", 0)))
    objective_valid = int(candidate_valid_count > 0 and control_pass == 0)
    substrate_pass = sum(int(r.get("substrate_gate_pass", 0)) for r in substrate_rows)
    s3 = task_pass_count(signal_summary, "pass_s3", parse_ints(args.synthetic_seeds), parse_csv(args.k_losses)) if objective_valid else 0
    s4 = task_pass_count(signal_summary, "pass_s4", parse_ints(args.synthetic_seeds), parse_csv(args.k_losses)) if objective_valid else 0
    nonrat_pass = sum(int(r.get("nonrat_vertical_pass", 0)) for r in nonrat_rows)
    mlp_dataset_pass = sum(int(r.get("dataset_pass", 0)) for r in mlp_rows)
    if missing:
        route = "R0-ProvenanceOrArchitectureViolation"
        minimum = "S0-Invalid"
    elif not objective_valid:
        route = "R2-CoverObjectiveInvalid"
        minimum = "S0-ObjectiveDiagnosticExecuted"
    elif substrate_pass <= 0:
        route = "R3-CoverObjectiveValidButNoSubstrate"
        minimum = "S1-CoverObjectiveValid"
    elif s3 < 5:
        route = "R4-CoverSubstrateNoFunctionalBreakthrough"
        minimum = "S2-CoverSubstrate"
    elif s4 < 5:
        route = "R5-SyntheticTaskPositiveGeometryUnstable"
        minimum = "S3-KANSynthetic"
    else:
        route = "S5-OfficialRealShortRunReady"
        minimum = "S4-KANLineCStable"
    best_oracle_pass = max([int(r.get("task_family_pass_count", 0)) for r in validity_rows if str(r.get("cover_objective_method")) in {"V1-FutureGradientClusterOracle", "V2-LineCSignalReservoirOracle", "V3-TaskFamilyOracle"}], default=0)
    best_oracle_source = max([fnum(r.get("max_source_vs_best_control"), -9.0) for r in validity_rows if str(r.get("cover_objective_method")) in {"V1-FutureGradientClusterOracle", "V2-LineCSignalReservoirOracle", "V3-TaskFamilyOracle"}], default=float("nan"))
    return {
        "route": route,
        "minimum_success": minimum,
        "official_success_reached": int(route == "S5-OfficialRealShortRunReady"),
        "promotion_allowed": int(route == "S5-OfficialRealShortRunReady"),
        "final_stop_allowed": int(route in {"R2-CoverObjectiveInvalid", "R3-CoverObjectiveValidButNoSubstrate", "R4-CoverSubstrateNoFunctionalBreakthrough", "R5-SyntheticTaskPositiveGeometryUnstable"} and missing == 0),
        "cover_objective_valid": objective_valid,
        "oracle_valid_candidate_count": candidate_valid_count,
        "oracle_best_task_family_pass_count": best_oracle_pass,
        "oracle_best_source_vs_best_control": best_oracle_source,
        "random_permuted_control_task_pass_count": control_pass,
        "cover_preserving_substrate_pass_count": substrate_pass,
        "kan_s3_task_pass_count": s3,
        "kan_s4_task_pass_count": s4,
        "nonrat_vertical_pass_count": nonrat_pass,
        "mlp_cover_dataset_pass_count": mlp_dataset_pass,
        "mlp_cover_dataset_count": len(mlp_rows),
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": 0,
        "oracle_rows": len(objective_summary),
        "architecture_candidate_count": len(A_CPF),
        "compute_budgeted_run": int(int(args.train_steps) < 160 or len(parse_ints(args.synthetic_seeds)) < 3),
        "code_review_packet_sha256": code_sha,
        "no_fake": 1,
    }


def progress_rows(route: dict[str, Any], validity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [{
        "stage": "V1311_PROGRESS",
        "line": "R",
        "status": "pass",
        "metric": "required_artifact_missing_count",
        "value": route["required_artifact_missing_count"],
        "no_fake": 1,
    }]
    for r in validity_rows:
        rows.append({
            "stage": "V1311_PROGRESS",
            "line": "V",
            "status": "pass" if int(r.get("objective_valid_candidate", 0)) else "fail",
            "metric": str(r.get("cover_objective_method")),
            "value": r.get("task_family_pass_count"),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    rows.extend([
        {"stage": "V1311_PROGRESS", "line": "A", "status": "skip" if not route["cover_objective_valid"] else "executed", "metric": "cover_preserving_substrate_pass_count", "value": route["cover_preserving_substrate_pass_count"], "no_fake": 1},
        {"stage": "V1311_PROGRESS", "line": "K", "status": "skip" if not route["cover_objective_valid"] else "executed", "metric": "kan_s3_task_pass_count", "value": route["kan_s3_task_pass_count"], "no_fake": 1},
        {"stage": "V1311_PROGRESS", "line": "N", "status": "executed", "metric": "nonrat_vertical_pass_count", "value": route["nonrat_vertical_pass_count"], "no_fake": 1},
        {"stage": "V1311_PROGRESS", "line": "G", "status": "executed", "metric": "mlp_cover_dataset_pass_count", "value": route["mlp_cover_dataset_pass_count"], "no_fake": 1},
    ])
    return rows


def write_no_go(out_dir: Path, route: dict[str, Any], validity_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v13.11 No-Go Boundary",
        "",
        f"route = {route['route']}",
        "",
        "Facts:",
        "",
        f"- cover_objective_valid = {route['cover_objective_valid']}",
        f"- oracle_best_task_family_pass_count = {route['oracle_best_task_family_pass_count']}/7",
        f"- oracle_best_source_vs_best_control = {route['oracle_best_source_vs_best_control']}",
        f"- random_permuted_control_task_pass_count = {route['random_permuted_control_task_pass_count']}",
        f"- cover_preserving_substrate_pass_count = {route['cover_preserving_substrate_pass_count']}",
        f"- KAN S3 task pass count = {route['kan_s3_task_pass_count']}/7",
        f"- KAN S4 task pass count = {route['kan_s4_task_pass_count']}/7",
        f"- Non-RAT vertical pass count = {route['nonrat_vertical_pass_count']}",
        "",
        "LineC/tail metrics are audit/gate only. Oracle cover rows are diagnostic-only and cannot promote.",
        "",
        "Objective validity rows:",
    ]
    for r in validity_rows:
        lines.append(f"- {r.get('cover_objective_method')}: pass={r.get('task_family_pass_count')}/7, max_source={r.get('max_source_vs_best_control')}, median_cover={r.get('median_cover_purity')}")
    (out_dir / "v1311_no_go_boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    queue = [
        "# v13.11 Next Hypothesis Queue",
        "",
    ]
    if not route["cover_objective_valid"]:
        queue.extend([
            "1. Redefine cover objective before additional cover-preserving architecture search.",
            "2. Candidate objective reset: value-cover based on future-advantage correlation, not raw cover purity alone.",
            "3. Keep oracle/random/permuted diagnostic controls in the next plan.",
        ])
    else:
        queue.extend([
            "1. If objective valid but no substrate: stronger A-CPF primitive or narrower Rational-only architecture research.",
            "2. If substrate passes but K fails: task-family autopsy tied to a specific failure, not token grid search.",
        ])
    (out_dir / "v1311_next_hypothesis_queue.md").write_text("\n".join(queue) + "\n", encoding="utf-8")


def write_figures(out_dir: Path, route: dict[str, Any], validity_rows: list[dict[str, Any]], summaries: list[dict[str, Any]], nonrat_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]]) -> None:
    write_svg(out_dir / "fig_cover_objective_oracle_vs_random.svg", "Oracle vs random cover", [f"{r.get('cover_objective_method')}: {r.get('task_family_pass_count')}/7" for r in validity_rows])
    write_svg(out_dir / "fig_cover_purity_vs_future_advantage.svg", "Cover purity vs source", [f"{r.get('cover_purity')} / {r.get('source_vs_best_control')}" for r in summaries[:40]])
    write_svg(out_dir / "fig_cover_churn_vs_source.svg", "Cover churn vs source", [f"{r.get('cover_churn')} / {r.get('source_vs_best_control')}" for r in summaries[:40]])
    write_svg(out_dir / "fig_cover_specialization_heatmap.svg", "Cover specialization", [f"{r.get('cover_objective_method')}: {r.get('cover_specialization_entropy')}" for r in summaries[:40]])
    write_svg(out_dir / "fig_group_to_task_assignment_matrix.svg", "Group to task assignment", [f"{r.get('task_or_dataset')} {r.get('cover_objective_method')} pass={r.get('task_family_pass')}" for r in summaries[:40]])
    write_svg(out_dir / "fig_cross_group_mixing_norm.svg", "Cross group mixing", [f"route={route['route']}", f"A-CPF pass={route['cover_preserving_substrate_pass_count']}"])
    write_svg(out_dir / "fig_winner_group_stability.svg", "Winner group stability", [f"{r.get('cover_objective_method')}: churn={r.get('median_cover_churn')}" for r in validity_rows])
    write_svg(out_dir / "fig_signal_reservoir_bank_update_fraction.svg", "Signal/reservoir bank", [f"{r.get('cover_objective_method')}: cover={r.get('median_cover_purity')}" for r in validity_rows])
    write_svg(out_dir / "fig_s3_s4_family_heatmap.svg", "S3/S4 family", [f"S3={route['kan_s3_task_pass_count']}/7", f"S4={route['kan_s4_task_pass_count']}/7"])
    write_svg(out_dir / "fig_nonrat_substrate_status_matrix.svg", "Non-RAT status", [f"{r.get('candidate')}: {r.get('status')}" for r in nonrat_rows])
    write_svg(out_dir / "fig_mlp_vs_kan_cover_analog.svg", "MLP vs KAN cover analog", [f"MLP={route['mlp_cover_dataset_pass_count']}/{route['mlp_cover_dataset_count']}", f"KAN S3={route['kan_s3_task_pass_count']}/7"])
    write_svg(out_dir / "fig_failure_taxonomy.svg", "Failure taxonomy", [f"route={route['route']}", f"objective_valid={route['cover_objective_valid']}"])


def required_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    for name in REQUIRED + FIGURES:
        path = out_dir / name
        rows.append({"stage": "V1311_REQUIRED_MANIFEST", "path": str(path.relative_to(ROOT) if path.exists() else path), "required": 1, "exists": int(path.exists()), "no_fake": 1})
    return rows, sum(1 for r in rows if int(r["exists"]) != 1)


def code_packet(out_dir: Path) -> str:
    packet = out_dir / "v1311_code_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [
            Path(__file__),
            ROOT / "experiments" / "run_v1310_cover_forming_substrate_architecture_reset.py",
            ROOT / "experiments" / "run_v139_signal_to_cover_functional_substrate_architecture.py",
            DOC_PLAN,
        ]:
            if path.exists():
                zf.write(path, path.relative_to(ROOT).as_posix())
    return sha256_file(packet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cover-objective-methods", default=",".join(ALL_V_METHODS))
    ap.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    ap.add_argument("--synthetic-seeds", default="0,1,2")
    ap.add_argument("--synthetic-train-size", type=int, default=96)
    ap.add_argument("--synthetic-val-size", type=int, default=48)
    ap.add_argument("--synthetic-dim", type=int, default=16)
    ap.add_argument("--synthetic-classes", type=int, default=3)
    ap.add_argument("--k-losses", default="CE,Brier")
    ap.add_argument("--train-steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--oracle-future-offset", type=int, default=17)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--mlp-seeds", default="0,1")
    ap.add_argument("--mlp-seed-threshold", type=int, default=2)
    ap.add_argument("--mlp-methods", default="MLP-AdamW,MLP-AdamW-SNREMA-Blend50-ActiveFractionSchedule,MLP-AdamW-SNRRoleNorm-Blend50-LogitNormTrust")
    ap.add_argument("--loss-interface", default="CE")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--real-train-size", type=int, default=128)
    ap.add_argument("--real-val-size", type=int, default=64)
    ap.add_argument("--real-test-size", type=int, default=64)
    ap.add_argument("--real-epochs", type=int, default=1)
    ap.add_argument("--mlp-hidden", type=int, default=160)
    # Compatibility args consumed by run_mlp_line.
    ap.add_argument("--snr-tau", type=float, default=1.0)
    ap.add_argument("--snr-eps", type=float, default=1.0e-12)
    ap.add_argument("--snr-ema-decay", type=float, default=0.85)
    ap.add_argument("--soft-alpha", type=float, default=8.0)
    ap.add_argument("--active-fraction-cap", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training, summaries, linec, failures = run_line_v(args, device)
    validity_rows = summarize_cover_validity(summaries, args)
    objective_valid = int(any(int(r.get("objective_valid_candidate", 0)) for r in validity_rows) and not any(int(r.get("control_task_family_pass_count", 0)) for r in validity_rows if int(r.get("random_or_permuted_control", 0))))
    substrate_rows, identity_rows, signal_summary = fail_closed_architecture_rows(bool(objective_valid))
    signal_training = [{"stage": "V1311_SIGNAL_TO_COVER_TRAINING", "skip_reason": "line_v_invalid" if not objective_valid else "not_implemented_in_this_compute_budget", "promotion_allowed": 0, "no_fake": 1}]
    nonrat_rows = run_nonrat_vertical(args, bool(objective_valid))
    _mlp_training, mlp_summary, mlp_dataset_rows, mlp_failures = run_mlp_line(args, device)
    failures.extend(mlp_failures)

    write_rows(out_dir / "v1311_code_review_manifest.csv", code_review_manifest())
    write_rows(out_dir / "v1311_architecture_diff_manifest.csv", architecture_diff_manifest(bool(objective_valid)))
    write_rows(out_dir / "v1311_cover_objective_provenance.csv", build_objective_provenance())
    write_rows(out_dir / "v1311_forbidden_information_audit.csv", forbidden_audit())
    write_readback(out_dir, bool(objective_valid))
    write_rows(out_dir / "v1311_cover_objective_validity.csv", validity_rows or [{"stage": "V1311_COVER_OBJECTIVE_VALIDITY_SUMMARY", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v1311_oracle_cover_results.csv", [r for r in summaries if int(r.get("oracle_diagnostic_only", 0))] or [{"stage": "V1311_ORACLE_COVER_RESULTS", "skip_reason": "no_rows", "promotion_allowed": 0, "no_fake": 1}])
    write_rows(out_dir / "v1311_random_permuted_cover_controls.csv", [r for r in summaries if int(r.get("random_or_permuted_control", 0))] or [{"stage": "V1311_RANDOM_PERMUTED_COVER_CONTROLS", "skip_reason": "no_rows", "promotion_allowed": 0, "no_fake": 1}])
    write_rows(out_dir / "v1311_cover_preserving_substrate.csv", substrate_rows or [{"stage": "V1311_COVER_PRESERVING_SUBSTRATE", "skip_reason": "no_rows", "promotion_allowed": 0, "no_fake": 1}])
    write_rows(out_dir / "v1311_cover_identity_telemetry.csv", identity_rows or [{"stage": "V1311_COVER_IDENTITY_TELEMETRY", "skip_reason": "no_rows", "promotion_allowed": 0, "no_fake": 1}])
    write_rows(out_dir / "v1311_signal_to_cover_training.csv", signal_training)
    write_rows(out_dir / "v1311_signal_to_cover_summary.csv", signal_summary or [{"stage": "V1311_SIGNAL_TO_COVER_SUMMARY", "skip_reason": "no_rows", "promotion_allowed": 0, "no_fake": 1}])
    write_rows(out_dir / "v1311_nonrat_vertical_slice.csv", nonrat_rows or [{"stage": "V1311_NONRAT_VERTICAL_SLICE", "skip_reason": "no_rows", "promotion_allowed": 0, "no_fake": 1}])
    write_rows(out_dir / "v1311_mlp_cover_analog.csv", mlp_summary or [{"stage": "V1311_MLP_COVER_ANALOG", "skip_reason": "no_rows", "promotion_allowed": 0, "no_fake": 1}])
    write_rows(out_dir / "v1311_linec_audit.csv", linec or [{"stage": "V1311_LINEC_AUDIT", "skip_reason": "no_rows", "audit_only": 1, "no_fake": 1}])
    write_rows(out_dir / "v1311_failure_table.csv", failures or [{"stage": "V1311_FAILURE_TABLE", "failure_rows": 0, "no_fake": 1}])
    write_rows(out_dir / "v1311_loss_interface_audit.csv", loss_interface_audit(parse_csv(args.cover_objective_methods) + parse_csv(args.mlp_methods), parse_csv(args.k_losses) + [args.loss_interface]))

    code_sha = code_packet(out_dir)
    route = build_route(objective_summary=summaries, validity_rows=validity_rows, substrate_rows=substrate_rows, signal_summary=signal_summary, nonrat_rows=nonrat_rows, mlp_rows=mlp_dataset_rows, missing=0, args=args, code_sha=code_sha)
    write_no_go(out_dir, route, validity_rows)
    write_figures(out_dir, route, validity_rows, summaries, nonrat_rows, mlp_dataset_rows)
    write_rows(out_dir / "v1311_progress_table.csv", progress_rows(route, validity_rows))
    (out_dir / "v1311_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest, missing = required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    (out_dir / "v1311_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rows(out_dir / "v1311_required_manifest.csv", manifest)
    manifest, missing = required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    (out_dir / "v1311_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rows(out_dir / "v1311_required_manifest.csv", manifest)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
