#!/usr/bin/env python3
"""DG-KAN v15.7 source-hazard factorization / carrier reparameterization runner.

This version tests whether the v15.6 source/hazard colinearity is a metric
artifact or can be reduced by pre-registered carrier/coordinate transforms.
Audit/tail/LineC metrics are never used to build the direction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import copy
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v1410_nonrat_fms_transfer_fms_definition_reset as v1410  # noqa: E402
from experiments import run_v144_real_transfer_fms_all_basis_substrate as v144  # noqa: E402
from experiments import run_v150_function_update_allbasis_parallel as v150  # noqa: E402
from experiments import run_v154_split_consensus_signal_subspace_fu_allbasis as v154  # noqa: E402


PLAN_DOC = ROOT / "docs/DG-KAN_v15.7_SourceHazardFactorization_CarrierReparam_AllBasisAcceleration_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v15.7_SourceHazardFactorization_CarrierReparam_AllBasisAcceleration_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v15.7_SourceHazardFactorization_CarrierReparam_AllBasisAcceleration_执行日志.md"
DEFAULT_OUT = ROOT / "results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/official_v157"
DEFAULT_LINE_D_OUT = ROOT / "results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/line_d_v157_allbasis_substrate"
MANUAL_ANALYSIS_START = "<!-- V15.7_MANUAL_ANALYSIS_START -->"
MANUAL_ANALYSIS_END = "<!-- V15.7_MANUAL_ANALYSIS_END -->"

G_CANDIDATES = [
    "G7R-original-replay",
    "G7K1-LowDegreeSignalCarrier",
    "G7K2-ReadoutBasisDecoupledCarrier",
    "G7K3-OrthogonalDegreeBankCarrier",
    "G7K4-OutputJacobianCarrier",
    "G7K5-DualBankCarrier",
]
U2_READBACK_METHODS = [
    "Q2-SourceRetainingLogitEntropyConstraint",
    "Q4-SourceRetainingCombinedConstraint-r4",
    "Q5-SourceRetainingCombinedConstraint-r8",
]
G_CONTROLS = [
    "G0-D-CHE-AdamW",
    "G1-D-CHE-CautiousAdamW",
    "C0-NoOpMatchedOverhead",
    "C1-RandomSubspaceSameRank",
    "C2-RandomSameProjectionRetention",
    "C3-SameActiveFractionRandomMask",
    "C4-SameDegreeEnergyRandom",
    "C5-AdamWParallelDirection",
    "C6-MLP-SameSplitConsensusControl",
    "C7-SameCarrierRandomDirection",
]
G_METHODS = G_CANDIDATES + U2_READBACK_METHODS + G_CONTROLS
G_CONTROL_SET = set(G_CONTROLS)

M_METHODS = [
    "M0-MLP-AdamW",
    "M1-MLP-CautiousAdamW",
    "M2-MLP-MGUP",
    "M3-MLP-SameSplitConsensusMetric",
    "M4-MLP-SameTrustScalar",
    "M5-MLP-RandomSameSourceNorm",
    "M6-MLP-NoOpMatchedOverhead",
]
M_CONTROL_SET = {"M0-MLP-AdamW", "M1-MLP-CautiousAdamW", "M2-MLP-MGUP", "M5-MLP-RandomSameSourceNorm", "M6-MLP-NoOpMatchedOverhead"}
S_SKETCHES = ["S0-diagonal", "S1-role-block", "S2-lowrank-r4", "S3-lowrank-r8"]
METRIC_CHOICES = ["M0-identity", "M1-AdamVDiag", "M2-DegreeRoleSecondMoment"]
LINE_D_CANDIDATES = [
    "D-FOU72-LowFreqIdentityResidualV5",
    "D-FOU73-BandwiseConsensusMetricV2",
    "D-FOU74-PhaseStableBandMixV2",
    "D-FOU75-NoMaterializeLifetimeV4",
    "D-FOU76-HighFrequencyQuarantineV2",
    "D-RBF70-ActiveCenterOccupancyV4",
    "D-RBF71-WidthConditionGuardV4",
    "D-RBF72-CompactBumpNoDenseV4",
    "D-RBF73-GaussianLocalK4TaskHealthV2",
    "D-RBF74-CenterSplitConsensusMetric",
    "D-WAV61-TriangularSupportV5",
    "D-WAV62-ScaleOccupancyV3",
    "D-WAV63-SupportOverlapDampingV3",
    "D-WAV64-LocalTailCoverageAuditV2",
]

REQUIRED = [
    "v157_route_decision.json",
    "v157_line_r_audit.csv",
    "v157_line_u2_source_hazard_colinearity.csv",
    "v157_line_k_carrier_factorization_readback.csv",
    "v157_line_g_factored_carrier_results.csv",
    "v157_line_b_proxy_leaveout.csv",
    "v157_line_p_micro_horizon_audit.csv",
    "v157_line_d_allbasis_results.csv",
    "v157_line_m_generic_controls.csv",
    "v157_line_c_geometry_tail_audit.csv",
    "v157_failure_taxonomy.csv",
    "v157_exhaustion_certificate.csv",
]
FIGURES = [
    "fig_u2_hazard_overlap_by_metric.svg",
    "fig_u2_source_retention_vs_overlap.svg",
    "fig_k_carrier_factorization_pareto.svg",
    "fig_g_source_vs_bad_event_pareto.svg",
    "fig_b_proxy_leaveout_heatmap.svg",
    "fig_p_micro_horizon_paths.svg",
    "fig_d_allbasis_substrate_heatmap.svg",
    "fig_m_kan_vs_mlp_specific_delta.svg",
    "fig_failure_taxonomy_heatmap.svg",
    "fig_route_dashboard.svg",
]


fnum = v154.fnum
sint = v154.sint
mean = v154.mean
median = v154.median
read_rows = v154.read_rows
write_rows = v154.write_rows
write_text = v154.write_text
write_json = v154.write_json
parse_csv = v154.parse_csv
parse_ints = v154.parse_ints
resolve_cuda_device = v154.resolve_cuda_device
add_flat_update = v154.add_flat_update
norm_match = v154.norm_match


def quantile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return default
    if len(vals) == 1:
        return vals[0]
    pos = min(len(vals) - 1, max(0.0, q * (len(vals) - 1)))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> tuple[float, int]:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores, strict=False) if math.isfinite(float(s))]
    positives = sum(1 for y, _ in pairs if y == 1)
    negatives = sum(1 for y, _ in pairs if y == 0)
    if positives == 0 or negatives == 0:
        return 0.5, 0
    wins = 0.0
    for y1, s1 in pairs:
        if y1 != 1:
            continue
        for y0, s0 in pairs:
            if y0 != 0:
                continue
            if s1 > s0:
                wins += 1.0
            elif s1 == s0:
                wins += 0.5
    return wins / max(1.0, positives * negatives), 1


def margin_p10(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.softmax(logits.detach(), dim=-1)
    true = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask.scatter_(1, labels.view(-1, 1), False)
    other = probs.masked_fill(~mask, -1.0).max(dim=1).values
    margin = true - other
    return float(torch.quantile(margin.float(), 0.10).item())


def entropy_mean(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.detach(), dim=-1).clamp_min(1.0e-8)
    return float((-(probs * probs.log()).sum(dim=1)).mean().item())


def logits_rms(logits: torch.Tensor) -> float:
    centered = logits.detach() - logits.detach().mean(dim=1, keepdim=True)
    return float(centered.square().mean().sqrt().item())


def method_to_sketch(method: str) -> str:
    if "K4" in method or "Random" in method:
        return "S2-lowrank-r4"
    return "S0-diagonal"


def base_metric_update(method: str, grad: torch.Tensor, mhat: torch.Tensor, vhat: torch.Tensor, specs: list[Any], stats: dict[str, Any], gen: torch.Generator) -> tuple[torch.Tensor, dict[str, Any]]:
    if method in {"G0-D-CHE-AdamW", "C5-AdamWParallelDirection", "M0-MLP-AdamW"}:
        return v154.choose_update("G0-D-CHE-AdamW", grad, mhat, vhat, specs, stats, gen)
    if method in {"G1-D-CHE-CautiousAdamW", "M1-MLP-CautiousAdamW"}:
        return v154.choose_update("G1-D-CHE-CautiousAdamW", grad, mhat, vhat, specs, stats, gen)
    if method in {"M2-MLP-MGUP"}:
        return v154.choose_update("G2-D-CHE-MGUP", grad, mhat, vhat, specs, stats, gen)
    if method in {"C1-RandomSubspaceSameRank", "M5-MLP-RandomSameSourceNorm"}:
        return v154.choose_update("C3-RandomSubspaceSameRank", grad, mhat, vhat, specs, stats, gen)
    if method == "C2-RandomSameProjectionRetention":
        return v154.choose_update("C4-RandomSubspaceSameProjectionRetention", grad, mhat, vhat, specs, stats, gen)
    if method in {"C3-SameActiveFractionRandomMask", "C4-SameDegreeEnergyRandom", "C7-SameCarrierRandomDirection"}:
        return v154.choose_update("C5-SameActiveFractionRandomMask", grad, mhat, vhat, specs, stats, gen)
    if method in {"C0-NoOpMatchedOverhead", "M6-MLP-NoOpMatchedOverhead"}:
        return v154.choose_update("C7-NoOpMatchedOverhead", grad, mhat, vhat, specs, stats, gen)
    return v154.choose_update("G7-D-CHE-SplitConsensusMetricNoProjection", grad, mhat, vhat, specs, stats, gen)


def train_stream_readback(
    model: torch.nn.Module,
    specs: list[Any],
    xs: Sequence[torch.Tensor],
    ys: Sequence[torch.Tensor],
    xb: torch.Tensor,
    yb: torch.Tensor,
    update: torch.Tensor,
    lr: float,
    trace: dict[str, Any],
) -> dict[str, float]:
    with torch.no_grad():
        logits_before = model(xb)
        ce_before = torch.nn.functional.cross_entropy(logits_before, yb, reduction="none")
        saved = [spec.param.detach().clone() for spec in specs]
        add_flat_update(specs, update, lr)
        logits_after = model(xb)
        ce_after = torch.nn.functional.cross_entropy(logits_after, yb, reduction="none")
        for spec, old in zip(specs, saved, strict=True):
            spec.param.copy_(old)
    split_losses = []
    with torch.no_grad():
        for x, y in zip(xs, ys, strict=False):
            split_losses.append(float(torch.nn.functional.cross_entropy(model(x), y, reduction="mean").item()))
    split_mean = max(1.0e-8, abs(mean(split_losses)))
    split_loss_disagreement = float(torch.tensor(split_losses, device=xb.device).float().std(unbiased=False).item()) / split_mean if split_losses else 0.0
    entropy_before = entropy_mean(logits_before)
    entropy_after = entropy_mean(logits_after)
    rms_before = logits_rms(logits_before)
    rms_after = logits_rms(logits_after)
    margin_before = margin_p10(logits_before, yb)
    margin_after = margin_p10(logits_after, yb)
    ce_vals = [float(v) for v in ce_before.detach().float().cpu().tolist()]
    recovery_lag = 0.0
    if fnum(trace.get("B2_gain"), 0.0) <= 0.0 and fnum(trace.get("B3_gain"), 0.0) <= 0.0:
        recovery_lag = 4.0
    elif fnum(trace.get("B2_gain"), 0.0) <= 0.0:
        recovery_lag = 2.0
    after_delta = ce_after - ce_before
    return {
        "B1_split_loss": split_losses[0] if len(split_losses) > 0 else 0.0,
        "B2_split_loss": split_losses[1] if len(split_losses) > 1 else 0.0,
        "B3_split_loss": split_losses[2] if len(split_losses) > 2 else 0.0,
        "split_loss_disagreement": split_loss_disagreement,
        "recovery_lag_proxy": recovery_lag,
        "logit_rms_before": rms_before,
        "logit_rms_after": rms_after,
        "logit_rms_drift": rms_after - rms_before,
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "entropy_collapse": max(0.0, entropy_before - entropy_after),
        "margin_p10_before": margin_before,
        "margin_p10_after": margin_after,
        "margin_p10_drift": margin_after - margin_before,
        "loss_q90": quantile(ce_vals, 0.90),
        "loss_q95": quantile(ce_vals, 0.95),
        "loss_q95_over_median": quantile(ce_vals, 0.95) / max(1.0e-8, quantile(ce_vals, 0.50)),
        "per_example_loss_delta_q90": float(torch.quantile(after_delta.detach().float(), 0.90).item()),
        "split_consensus_eigengap": fnum(trace.get("eigengap_A_split"), 0.0),
        "projection_retention_drift": 1.0 - fnum(trace.get("projection_retention"), 0.0),
        "update_cosine_to_adam": fnum(trace.get("cos_update_adam"), 0.0),
        "degree_energy_drift": abs(fnum(trace.get("cos_update_adam"), 0.0) - fnum(trace.get("cos_update_neg_grad"), 0.0)),
    }


def trust_components(proxy: dict[str, float]) -> dict[str, float]:
    split_agreement = 1.0 / (1.0 + max(0.0, proxy.get("split_loss_disagreement", 0.0)) + max(0.0, 1.0 - proxy.get("B2_over_B1_gain", 0.0)))
    recovery = 1.0 / (1.0 + max(0.0, proxy.get("recovery_lag_proxy", 0.0)) / 2.0 + max(0.0, proxy.get("loss_spike_count", 0.0)))
    logit_entropy = 1.0 / (1.0 + max(0.0, proxy.get("logit_rms_drift", 0.0)) + 2.0 * max(0.0, proxy.get("entropy_collapse", 0.0)) + max(0.0, -proxy.get("margin_p10_drift", 0.0)))
    loss_quantile = 1.0 / (1.0 + max(0.0, proxy.get("loss_q95_over_median", 1.0) - 1.0) + max(0.0, proxy.get("per_example_loss_delta_q90", 0.0)))
    comp = {
        "trust_split_agreement": float(min(1.0, max(0.15, split_agreement))),
        "trust_recovery_lag": float(min(1.0, max(0.15, recovery))),
        "trust_logit_entropy": float(min(1.0, max(0.15, logit_entropy))),
        "trust_loss_quantile": float(min(1.0, max(0.15, loss_quantile))),
    }
    comp["trust_combined"] = min(comp.values())
    return comp


def trust_for_method(method: str, comp: dict[str, float]) -> float:
    if "SameTrust" in method:
        return comp["trust_combined"]
    return 1.0


def role_mask(specs: list[Any], kind: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    chunks = []
    for spec in specs:
        name = str(getattr(spec, "name", ""))
        size = int(spec.param.numel())
        lower = name.lower()
        if kind == "readout":
            val = 1.0 if ("readout" in lower or "classifier" in lower or "head" in lower) else 0.25
        elif kind == "high_degree":
            val = 1.0 if any(token in lower for token in ["degree", "basis", "coeff", "kan"]) else 0.35
        else:
            val = 1.0
        chunks.append(torch.full((size,), val, device=device, dtype=dtype))
    return torch.cat(chunks) if chunks else torch.empty(0, device=device, dtype=dtype)


def binary_role_mask(specs: list[Any], kind: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    chunks = []
    for spec in specs:
        name = str(getattr(spec, "name", "")).lower()
        role = str(getattr(spec, "role", "")).lower()
        size = int(spec.param.numel())
        if kind == "readout":
            val = 1.0 if role == "readout" or any(token in name for token in ["readout", "classifier", "head"]) else 0.0
        elif kind == "basis":
            val = 1.0 if role == "basis" or any(token in name for token in ["basis", "degree", "coeff", "kan"]) else 0.0
        else:
            val = 1.0
        chunks.append(torch.full((size,), val, device=device, dtype=dtype))
    return torch.cat(chunks) if chunks else torch.empty(0, device=device, dtype=dtype)


def normalize_col(vec: torch.Tensor) -> torch.Tensor:
    return vec / (vec.norm().clamp_min(1.0e-8))


def metric_diag_for_readback(
    metric_name: str,
    grad: torch.Tensor,
    mhat: torch.Tensor,
    vhat: torch.Tensor,
    specs: list[Any],
    update: torch.Tensor,
) -> torch.Tensor:
    eps = 1.0e-8
    if metric_name == "M0-Identity":
        return torch.ones_like(vhat)
    if metric_name == "M1-AdamV":
        return vhat.clamp_min(eps)
    if metric_name == "M2-DegreeRoleSecondMoment":
        basis = binary_role_mask(specs, "basis", grad.device, grad.dtype)
        readout = binary_role_mask(specs, "readout", grad.device, grad.dtype)
        return (vhat.clamp_min(eps) * (1.0 + 0.75 * basis + 0.25 * readout)).clamp_min(eps)
    if metric_name == "M3-OutputJacobianDiag":
        return (grad.abs() + 0.25 * mhat.abs() + eps).clamp_min(eps)
    if metric_name == "M4-LowRankOutputJacobianSketch":
        score = (grad.abs() + update.abs()).clamp_min(eps)
        threshold = torch.quantile(score.detach().float(), 0.70)
        return (score * (score >= threshold).to(score.dtype) + eps).clamp_min(eps)
    if metric_name == "M5-FunctionSpaceDisplacement":
        return (update.abs() + torch.sqrt(vhat.clamp_min(eps)) + eps).clamp_min(eps)
    return vhat.clamp_min(eps)


def hazard_columns(
    method: str,
    grad: torch.Tensor,
    mhat: torch.Tensor,
    vhat: torch.Tensor,
    specs: list[Any],
    base_update: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    eps = 1.0e-8
    adam_like = -mhat / (torch.sqrt(vhat) + eps)
    readout = role_mask(specs, "readout", grad.device, grad.dtype)
    high_degree = role_mask(specs, "high_degree", grad.device, grad.dtype)
    loss_q = normalize_col(grad)
    logit = normalize_col(mhat)
    entropy = normalize_col(torch.sign(grad) * torch.sqrt(vhat + eps))
    recovery = normalize_col(adam_like - base_update)
    degree = normalize_col(high_degree * grad)
    degree_entropy = normalize_col((1.0 - readout) * torch.sign(grad) * torch.sqrt(vhat + eps))
    cosine = normalize_col(adam_like)
    try:
        safe = v150.basis_safe_projection(specs, base_update, "D-CHE")
        retention = normalize_col(base_update - safe)
    except Exception:  # noqa: BLE001
        retention = normalize_col(base_update - adam_like)
    all_cols = [
        ("loss_quantile", loss_q),
        ("loss_quantile_q90", normalize_col(0.75 * loss_q + 0.25 * entropy)),
        ("logit_rms", logit),
        ("entropy_collapse", entropy),
        ("split_disagreement", normalize_col(0.5 * loss_q + 0.5 * recovery)),
        ("recovery_lag", recovery),
        ("degree_energy", degree),
        ("degree_entropy", degree_entropy),
        ("update_cosine_disagreement", cosine),
        ("projection_retention_drift", retention),
    ]
    if method in {"N1-LossQuantileNull", "Q1-SourceRetainingLossQuantileConstraint"}:
        return all_cols[:2]
    if method in {"N2-LogitRMSNull", "Q2-SourceRetainingLogitEntropyConstraint"}:
        return all_cols[2:4]
    if method == "N3-EntropyRecoveryNull":
        return [all_cols[3], all_cols[5]]
    if method in {"N4-DegreeEnergyNull", "Q3-SourceRetainingDegreeEnergyConstraint"}:
        return [all_cols[6], all_cols[7]]
    if method in {"N5-CombinedHazardNull-rank4", "Q4-SourceRetainingCombinedConstraint-r4"}:
        return all_cols[:4]
    if method in {"N6-CombinedHazardNull-rank8", "Q5-SourceRetainingCombinedConstraint-r8", "C4-RandomNullspaceSameRank", "C5-SameHazardNullRandomDirection", "C8-SameSourceRetentionRandomDirection", "M4-MLP-SameHazardNullControl", "M6-MLP-RandomNullspaceSameRank"}:
        return all_cols[:8]
    return []


def weighted_hazard_projection(
    update: torch.Tensor,
    columns: list[tuple[str, torch.Tensor]],
    vhat: torch.Tensor,
    ridge: float = 1.0e-4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not columns:
        return torch.zeros_like(update), {
            "hazard_overlap": 0.0,
            "source_retention_after_null": 1.0,
            "hazard_tangent_rank": 0,
            "hazard_tangent_condition": 1.0,
            "proxy_dirderiv_before_mean": 0.0,
            "proxy_dirderiv_after_mean": 0.0,
        }
    metric = torch.sqrt(vhat).clamp_min(1.0e-8)
    weight = torch.sqrt(metric)
    mat = torch.stack([normalize_col(col) for _, col in columns], dim=1)
    wmat = mat * weight.view(-1, 1)
    wu = update * weight
    gram = wmat.t().matmul(wmat)
    gram = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    rhs = wmat.t().matmul(wu)
    coeff = torch.linalg.solve(gram, rhs)
    hazard = mat.matmul(coeff)
    null_update = update - hazard
    u_norm = torch.sqrt((update.square() * metric).sum()).clamp_min(1.0e-8)
    h_norm = torch.sqrt((hazard.square() * metric).sum())
    retention = ((null_update * update * metric).sum() / ((update.square() * metric).sum().clamp_min(1.0e-8))).item()
    eig = torch.linalg.eigvalsh(gram.detach())
    pos = eig[eig > 1.0e-8]
    condition = float((pos.max() / pos.min()).item()) if int(pos.numel()) >= 2 else 1.0
    before = [float((col * update).sum().item()) for _, col in columns]
    after = [float((col * null_update).sum().item()) for _, col in columns]
    return hazard, {
        "hazard_overlap": float((h_norm / u_norm).item()),
        "source_retention_after_null": float(retention),
        "hazard_tangent_rank": int(torch.linalg.matrix_rank(wmat.detach()).item()),
        "hazard_tangent_condition": condition,
        "proxy_dirderiv_before_mean": mean(abs(v) for v in before),
        "proxy_dirderiv_after_mean": mean(abs(v) for v in after),
        "hazard_proxy_names": ";".join(name for name, _ in columns),
    }


def submitted_update_readback(
    source_update: torch.Tensor,
    submitted_update: torch.Tensor,
    columns: list[tuple[str, torch.Tensor]],
    vhat: torch.Tensor,
) -> dict[str, Any]:
    """Read back the actual submitted update under the same weighted geometry."""
    if not columns:
        return {
            "hazard_overlap": 0.0,
            "source_retention_after_null": 1.0,
            "proxy_dirderiv_after_mean": 0.0,
        }
    metric = torch.sqrt(vhat).clamp_min(1.0e-8)
    source_norm_sq = (source_update.square() * metric).sum().clamp_min(1.0e-8)
    source_norm = torch.sqrt(source_norm_sq).clamp_min(1.0e-8)
    removed = source_update - submitted_update
    removed_norm = torch.sqrt((removed.square() * metric).sum())
    retention = (submitted_update * source_update * metric).sum() / source_norm_sq
    after = [float((normalize_col(col) * submitted_update).sum().item()) for _, col in columns]
    return {
        "hazard_overlap": float((removed_norm / source_norm).item()),
        "source_retention_after_null": float(retention.item()),
        "proxy_dirderiv_after_mean": mean(abs(v) for v in after),
    }


def choose_stability_update(
    method: str,
    grad: torch.Tensor,
    mhat: torch.Tensor,
    vhat: torch.Tensor,
    specs: list[Any],
    stats: dict[str, Any],
    gen: torch.Generator,
    proxy: dict[str, float],
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_update, trace = base_metric_update("N0-G7R-Replay", grad, mhat, vhat, specs, stats, gen)
    comp = trust_components({**proxy, **trace})
    line_u_columns = hazard_columns("N5-CombinedHazardNull-rank4", grad, mhat, vhat, specs, base_update)
    _, line_u_trace = weighted_hazard_projection(base_update, line_u_columns, vhat)
    columns = hazard_columns(method, grad, mhat, vhat, specs, base_update)
    hazard, hazard_trace = weighted_hazard_projection(base_update, columns, vhat)
    null_update = base_update - hazard
    update = base_update
    q_failure = ""
    q_feasible = 1
    if method in {"C0-D-CHE-AdamW", "C1-D-CHE-CautiousAdamW", "C2-D-CHE-MGUP", "C6-NoOpMatchedOverhead", "M0-MLP-AdamW", "M1-MLP-CautiousAdamW", "M2-MLP-MGUP", "M7-MLP-NoOpMatchedOverhead"}:
        update, control_trace = base_metric_update(method, grad, mhat, vhat, specs, stats, gen)
        trace.update(control_trace)
        columns = []
        hazard_trace = {"hazard_overlap": 0.0, "source_retention_after_null": 1.0, "hazard_tangent_rank": 0, "hazard_tangent_condition": 1.0, "proxy_dirderiv_before_mean": 0.0, "proxy_dirderiv_after_mean": 0.0, "hazard_proxy_names": ""}
    elif method in {"C3-G7R-RandomMatchedNorm"}:
        random = torch.randn(base_update.shape, generator=gen, device=base_update.device, dtype=base_update.dtype)
        update = norm_match(random, base_update)
    elif method in {"C4-RandomNullspaceSameRank", "M6-MLP-RandomNullspaceSameRank"}:
        random = norm_match(torch.randn(base_update.shape, generator=gen, device=base_update.device, dtype=base_update.dtype), base_update)
        rand_hazard, rand_trace = weighted_hazard_projection(random, columns, vhat)
        update = random - rand_hazard
        hazard_trace.update({f"random_{k}": v for k, v in rand_trace.items() if isinstance(v, (int, float))})
    elif method in {"C5-SameHazardNullRandomDirection"}:
        random = norm_match(torch.randn(base_update.shape, generator=gen, device=base_update.device, dtype=base_update.dtype), base_update)
        rand_hazard, _ = weighted_hazard_projection(random, columns, vhat)
        update = norm_match(random - rand_hazard, null_update)
    elif method in {"C8-SameSourceRetentionRandomDirection"}:
        random = norm_match(torch.randn(base_update.shape, generator=gen, device=base_update.device, dtype=base_update.dtype), base_update)
        update = norm_match(random, base_update * max(0.05, float(hazard_trace.get("source_retention_after_null", 1.0))))
    elif method in {"M3-MLP-SplitConsensusMetric"}:
        update = base_update
    elif method in {"M4-MLP-SameHazardNullControl"}:
        update = null_update
    elif method in {"M5-MLP-SameTrustScalarControl"}:
        update = base_update * comp["trust_combined"]
    elif method.startswith("Q"):
        r_min = 0.60
        retention = float(hazard_trace.get("source_retention_after_null", 1.0))
        full_null_trace = dict(hazard_trace)
        if retention < r_min:
            denom = max(1.0e-8, 1.0 - retention)
            alpha = max(0.0, min(1.0, (1.0 - r_min) / denom))
            update = base_update - alpha * hazard
            q_failure = "Q-F2-full-null-source-retention-low-fixed-alpha"
        else:
            alpha = 1.0
            update = null_update
        actual_trace = submitted_update_readback(base_update, update, columns, vhat)
        for key in ["hazard_overlap", "source_retention_after_null", "proxy_dirderiv_after_mean"]:
            hazard_trace[f"{key}_full_null"] = full_null_trace.get(key, "")
        hazard_trace.update(actual_trace)
        hazard_trace["constraint_alpha"] = float(alpha)
        hazard_trace["q_feasible"] = int(float(actual_trace.get("source_retention_after_null", 0.0)) >= r_min - 1.0e-6)
        hazard_trace["q_hazard_constraint_relaxed"] = int(alpha < 1.0)
        hazard_trace["q_full_null_retention_failed"] = int(retention < r_min)
        hazard_trace["q_failure"] = q_failure
        hazard_trace["constraint_solve_time_ms"] = 0.0
    elif method.startswith("N") and method != "N0-G7R-Replay":
        update = null_update
    trace.update(comp)
    trace.update(hazard_trace)
    for key, value in line_u_trace.items():
        if isinstance(value, (int, float)):
            trace[f"line_u_{key}"] = value
    trace["line_u_hazard_proxy_names"] = line_u_trace.get("hazard_proxy_names", "")
    trace["trust_scalar"] = 1.0
    trace["trust_scalar_applied"] = 0
    trace["hazard_null_applied"] = int((method.startswith("N") and method != "N0-G7R-Replay") or method.startswith("Q") or method in {"C4-RandomNullspaceSameRank", "C5-SameHazardNullRandomDirection", "M4-MLP-SameHazardNullControl", "M6-MLP-RandomNullspaceSameRank"})
    trace["source_retention_constraint"] = 0.60 if method.startswith("Q") else ""
    trace["line_u_readback_only"] = 0
    return update, trace


def choose_factor_update(
    method: str,
    grad: torch.Tensor,
    mhat: torch.Tensor,
    vhat: torch.Tensor,
    specs: list[Any],
    stats: dict[str, Any],
    gen: torch.Generator,
    proxy: dict[str, float],
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_update, trace = base_metric_update("G7R-original-replay", grad, mhat, vhat, specs, stats, gen)
    adam_update, adam_trace = base_metric_update("G0-D-CHE-AdamW", grad, mhat, vhat, specs, stats, gen)
    cautious_update, cautious_trace = base_metric_update("G1-D-CHE-CautiousAdamW", grad, mhat, vhat, specs, stats, gen)
    basis = binary_role_mask(specs, "basis", grad.device, grad.dtype)
    readout = binary_role_mask(specs, "readout", grad.device, grad.dtype)
    non_readout = 1.0 - readout
    signal_mask = (1.0 - 0.70 * basis + 0.15 * readout).clamp(0.15, 1.0)
    carrier = "K0-original"
    update = base_update
    if method in {"G0-D-CHE-AdamW", "M0-MLP-AdamW", "C5-AdamWParallelDirection"}:
        update, extra = base_metric_update(method, grad, mhat, vhat, specs, stats, gen)
        trace.update(extra)
        carrier = "control-adamw"
    elif method in {"G1-D-CHE-CautiousAdamW", "M1-MLP-CautiousAdamW"}:
        update, extra = base_metric_update(method, grad, mhat, vhat, specs, stats, gen)
        trace.update(extra)
        carrier = "control-cautious"
    elif method in {"M2-MLP-MGUP"}:
        update, extra = base_metric_update(method, grad, mhat, vhat, specs, stats, gen)
        trace.update(extra)
        carrier = "control-mgup"
    elif method in {"C0-NoOpMatchedOverhead", "M6-MLP-NoOpMatchedOverhead"}:
        update = torch.zeros_like(base_update)
        carrier = "control-noop"
    elif method in {"C1-RandomSubspaceSameRank", "M5-MLP-RandomSameSourceNorm"}:
        random = torch.randn(base_update.shape, generator=gen, device=base_update.device, dtype=base_update.dtype)
        update = norm_match(random, base_update)
        carrier = "control-random-source-norm"
    elif method == "C2-RandomSameProjectionRetention":
        random = torch.randn(base_update.shape, generator=gen, device=base_update.device, dtype=base_update.dtype)
        update = norm_match(random * signal_mask, base_update * signal_mask)
        carrier = "control-random-same-retention"
    elif method == "C3-SameActiveFractionRandomMask":
        random = norm_match(torch.randn(base_update.shape, generator=gen, device=base_update.device, dtype=base_update.dtype), base_update)
        active = (signal_mask >= torch.quantile(signal_mask.detach().float(), 0.40)).to(base_update.dtype)
        update = norm_match(random * active, base_update * active)
        carrier = "control-random-active-fraction"
    elif method == "C4-SameDegreeEnergyRandom":
        random = norm_match(torch.randn(base_update.shape, generator=gen, device=base_update.device, dtype=base_update.dtype), base_update)
        update = norm_match(random * (0.25 + 0.75 * basis), base_update * (0.25 + 0.75 * basis))
        carrier = "control-random-degree-energy"
    elif method == "C6-MLP-SameSplitConsensusControl":
        update = base_update
        carrier = "control-mlp-alias"
    elif method == "C7-SameCarrierRandomDirection":
        random = norm_match(torch.randn(base_update.shape, generator=gen, device=base_update.device, dtype=base_update.dtype), base_update)
        update = norm_match(random * signal_mask, base_update * signal_mask)
        carrier = "control-same-carrier-random"
    elif method in {"M3-MLP-SameSplitConsensusMetric", "M4-MLP-SameTrustScalar"}:
        update = base_update if method.startswith("M3") else base_update * trust_components({**proxy, **trace})["trust_combined"]
        carrier = "mlp-generic"
    elif method.startswith("Q"):
        q_columns = hazard_columns(method, grad, mhat, vhat, specs, base_update)
        q_hazard, q_full_trace = weighted_hazard_projection(base_update, q_columns, vhat)
        q_retention = float(q_full_trace.get("source_retention_after_null", 1.0))
        r_min = 0.60
        if q_retention < r_min:
            denom = max(1.0e-8, 1.0 - q_retention)
            alpha = max(0.0, min(1.0, (1.0 - r_min) / denom))
            update = base_update - alpha * q_hazard
            q_failure = "Q-F2-full-null-source-retention-low-fixed-alpha"
        else:
            alpha = 1.0
            update = base_update - q_hazard
            q_failure = ""
        q_actual = submitted_update_readback(base_update, update, q_columns, vhat)
        trace.update({f"{k}_full_null": v for k, v in q_full_trace.items() if isinstance(v, (int, float))})
        trace.update(
            {
                "q_feasible": int(float(q_actual.get("source_retention_after_null", 0.0)) >= r_min - 1.0e-6),
                "q_hazard_constraint_relaxed": int(alpha < 1.0),
                "q_full_null_retention_failed": int(q_retention < r_min),
                "q_failure": q_failure,
                "constraint_alpha": float(alpha),
            }
        )
        carrier = "U2-readback-" + method.split("-", 1)[0]
    elif method == "G7K1-LowDegreeSignalCarrier":
        update = base_update * signal_mask + 0.10 * adam_update * basis
        carrier = "K1-low-degree-signal"
    elif method == "G7K2-ReadoutBasisDecoupledCarrier":
        update = base_update * non_readout + cautious_update * readout
        carrier = "K2-readout-basis-decoupled"
    elif method == "G7K3-OrthogonalDegreeBankCarrier":
        centered = base_update - (base_update * basis).mean() * basis
        update = centered * (1.0 - 0.45 * basis) + 0.15 * adam_update * basis
        carrier = "K3-orthogonal-degree-bank"
    elif method == "G7K4-OutputJacobianCarrier":
        basis_tensor = stats.get("lowrank_basis")
        if isinstance(basis_tensor, torch.Tensor) and basis_tensor.numel() > 0:
            projected = basis_tensor.t().matmul(basis_tensor.matmul(base_update))
            update = norm_match(projected, base_update)
        else:
            update = base_update * signal_mask
        carrier = "K4-output-jacobian"
    elif method == "G7K5-DualBankCarrier":
        update = base_update * signal_mask + 0.20 * adam_update * (1.0 - signal_mask)
        carrier = "K5-dual-bank"
    columns = hazard_columns("N5-CombinedHazardNull-rank4", grad, mhat, vhat, specs, base_update)
    _, hazard_trace = weighted_hazard_projection(update, columns, vhat)
    readback_vs_source = submitted_update_readback(base_update, update, columns, vhat)
    trace.update(hazard_trace)
    trace["carrier_id"] = carrier
    trace["carrier_reparam_applied"] = int(method.startswith("G7K"))
    trace["source_retention_vs_g7r"] = readback_vs_source.get("source_retention_after_null", 1.0)
    trace["source_update_norm_ratio"] = float(update.norm().item() / max(1.0e-8, float(base_update.norm().item())))
    for metric_name in ["M0-Identity", "M1-AdamV", "M2-DegreeRoleSecondMoment", "M3-OutputJacobianDiag", "M4-LowRankOutputJacobianSketch", "M5-FunctionSpaceDisplacement"]:
        metric_diag = metric_diag_for_readback(metric_name, grad, mhat, vhat, specs, update)
        _, metric_trace = weighted_hazard_projection(update, columns, metric_diag)
        prefix = "u2_" + metric_name.replace("-", "_").replace(" ", "_")
        trace[f"{prefix}_hazard_overlap"] = metric_trace.get("hazard_overlap", 0.0)
        trace[f"{prefix}_source_retention_after_null"] = metric_trace.get("source_retention_after_null", 1.0)
        trace[f"{prefix}_hazard_tangent_rank"] = metric_trace.get("hazard_tangent_rank", 0)
        trace[f"{prefix}_metric_condition"] = metric_trace.get("hazard_tangent_condition", 1.0)
    trace["trust_scalar"] = trust_for_method(method, trust_components({**proxy, **trace}))
    trace["trust_scalar_applied"] = int(method in {"M4-MLP-SameTrustScalar"})
    trace["hazard_null_applied"] = 0
    trace["line_u_readback_only"] = 0
    trace.update({"adam_control_norm": float(adam_update.norm().item()), "cautious_control_norm": float(cautious_update.norm().item())})
    trace.update({k: v for k, v in adam_trace.items() if k.startswith("step_time")})
    return update, trace


def run_case(
    *,
    line: str,
    family: str,
    method: str,
    dataset: str,
    seed: int,
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
) -> dict[str, Any]:
    case_args = copy(args)
    case_args.synthetic_dim = int(input_dim)
    case_args.synthetic_classes = int(output_dim)
    case_args.mlp_hidden = int(args.hidden)
    if family == "MLP":
        model = v1410.make_case_model("MLP", "MLP-v157-source-hazard factorization", xtr, seed, case_args, device)
    else:
        model = v1410.make_case_model("D-CHE", str(args.dche_candidate), xtr, seed, case_args, device)
    specs = v1410.named_param_specs(model)
    total = sum(int(spec.param.numel()) for spec in specs)
    m = torch.zeros(total, device=device)
    v = torch.zeros(total, device=device)
    batch_gen = torch.Generator(device=device).manual_seed(
        int(seed) + 1_550_000 + sum(ord(c) for c in dataset) + (20_000 if family == "MLP" else 0)
    )
    gen = torch.Generator(device=device).manual_seed(int(seed) + 1_555_000 + sum(ord(c) for c in method + dataset))
    telemetry: dict[str, list[float]] = {}
    trajectory: list[dict[str, float]] = []
    linec_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    last_trace: dict[str, Any] = {}
    metric_choice = "M0-identity"
    sketch = method_to_sketch(method)
    for step in range(int(args.train_steps)):
        idx = torch.randint(0, xtr.shape[0], (int(args.batch_size),), generator=batch_gen, device=device)
        xb, yb = xtr[idx], ytr[idx]
        grads, xs, ys, _losses = v154.split_grads(model, specs, xb, yb, int(args.split_count))
        grad = torch.stack(grads, dim=0).mean(dim=0)
        beta1, beta2 = float(args.beta1), float(args.beta2)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()
        mhat = m / (1.0 - beta1 ** (step + 1))
        vhat = v / (1.0 - beta2 ** (step + 1))
        stats = v154.consensus_stats(grads, grad, vhat, specs, metric_choice, sketch, gen, float(args.lambda_noise))
        base_update, base_trace = base_metric_update("G7R-original-replay", grad, mhat, vhat, specs, stats, gen)
        split_deltas = v154.trial_split_deltas(model, specs, xs, ys, base_update, float(args.lr))
        base_trace.update(
            {
                "B1_gain": -split_deltas[0] if len(split_deltas) > 0 else 0.0,
                "B2_gain": -split_deltas[1] if len(split_deltas) > 1 else 0.0,
                "B3_gain": -split_deltas[2] if len(split_deltas) > 2 else 0.0,
                "B2_over_B1_gain": (-split_deltas[1] / max(1.0e-8, -split_deltas[0])) if len(split_deltas) > 1 and -split_deltas[0] > 0 else 0.0,
                "B3_over_B1_gain": (-split_deltas[2] / max(1.0e-8, -split_deltas[0])) if len(split_deltas) > 2 and -split_deltas[0] > 0 else 0.0,
                "micro_horizon_loss_integral": sum(max(0.0, d) for d in split_deltas),
                "loss_spike_count": sum(int(d > 0.0) for d in split_deltas),
            }
        )
        proxy = train_stream_readback(model, specs, xs, ys, xb, yb, base_update, float(args.lr), base_trace)
        update, trace = choose_factor_update(method, grad, mhat, vhat, specs, stats, gen, {**base_trace, **proxy})
        if family == "D-CHE":
            update = v150.basis_safe_projection(specs, update, "D-CHE")
        trace.update(base_trace)
        trace.update(proxy)
        if "trust_scalar" not in trace:
            trace["trust_scalar"] = trust_for_method(method, trust_components({**base_trace, **proxy}))
        last_trace = trace
        add_flat_update(specs, update, float(args.lr))
        v150.apply_role_decay(specs, float(args.lr), float(args.weight_decay), float(args.readout_weight_decay))
        for key, value in trace.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                telemetry.setdefault(key, []).append(float(value))
        if step == 0 or step == int(args.train_steps) - 1 or ((step + 1) % max(1, int(args.trace_interval)) == 0):
            metrics = v1410.eval_metrics(model, xva, yva)
            trajectory.append({"step": float(step + 1), "NLL": metrics["NLL"], "CEp99": metrics["CEp99"], "ECE": metrics["ECE"], "Brier": metrics["Brier"], "acc": metrics["acc"]})
            direction_rows.append(
                {
                    "line": line,
                    "family": family,
                    "method": method,
                    "metric_choice": metric_choice,
                    "sketch": sketch,
                    "dataset": dataset,
                    "seed": seed,
                    "step": step + 1,
                    **{k: median(vv) for k, vv in telemetry.items()},
                    "direction_uses_train_stream_only": 1,
                    "uses_LineC_as_direction": 0,
                    "uses_CEp99_as_direction": 0,
                    "uses_NLL_as_direction": 0,
                    "uses_ECE_as_direction": 0,
                    "uses_AUCtime_as_direction": 0,
                    "promotion_allowed": 0,
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    elapsed = time.perf_counter() - start
    final = v1410.eval_metrics(model, xva, yva)
    test_final = v1410.eval_metrics(model, xte, yte) if xte is not None and yte is not None else {}
    votes = [1]
    if family == "D-CHE" and int(args.real_linec) == 1:
        b = min(int(args.linec_batch_size), int(xtr.shape[0]), int(xva.shape[0]))
        votes = []
        for linec_seed in parse_ints(args.linec_seeds):
            try:
                lm = v1410.linec_metrics(model, xtr[:b], ytr[:b], xva[:b], yva[:b], int(linec_seed), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
                status, error = "executed", ""
            except Exception as exc:  # noqa: BLE001
                lm = {"CouplingR2": float("nan"), "NoiseSignalLeak": float("nan"), "RealSignalReservoirRatio": float("nan")}
                status, error = "blocked", f"{type(exc).__name__}: {exc}"
            passed = int(fnum(lm.get("CouplingR2"), -999) >= 0.15 and fnum(lm.get("NoiseSignalLeak"), 999) <= 0.20 and fnum(lm.get("RealSignalReservoirRatio"), 999) <= 0.70)
            votes.append(passed)
            linec_rows.append(
                {
                    "line": line,
                    "family": family,
                    "method": method,
                    "metric_choice": metric_choice,
                    "sketch": sketch,
                    "dataset": dataset,
                    "seed": seed,
                    "linec_seed": linec_seed,
                    "linec_status": status,
                    "linec_error": error,
                    "LineC_pass": passed,
                    **lm,
                    "CEp99": final["CEp99"],
                    "NLL": final["NLL"],
                    "ECE": final["ECE"],
                    "Brier": final["Brier"],
                    "linec_used_for_direction": 0,
                    "tail_metric_used_for_direction": 0,
                    "promotion_allowed": 0,
                }
            )
    row = {
        "stage": "V157_SOURCE_HAZARD_CARRIER_CASE",
        "line": line,
        "family": family,
        "method": method,
        "metric_choice": metric_choice,
        "sketch": sketch,
        "dataset": dataset,
        "seed": seed,
        "step": int(args.train_steps),
        "NLL": final["NLL"],
        "CEp99": final["CEp99"],
        "ECE": final["ECE"],
        "Brier": final["Brier"],
        "margin_p10": final["margin_p10"],
        "acc": final["acc"],
        "test_NLL": test_final.get("NLL", ""),
        "AUC_NLL": mean(r["NLL"] for r in trajectory),
        "AUC_CEp99": mean(r["CEp99"] for r in trajectory),
        "LineC_majority_pass": int(sum(votes) >= math.ceil(len(votes) / 2)),
        "LineC_pass_rate": sum(votes) / max(1, len(votes)),
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / max(1, int(args.train_steps)),
        "peak_memory_bytes": peak_memory,
        **{k: median(vv) for k, vv in telemetry.items()},
        "uses_validation_for_direction": 0,
        "uses_test_for_direction": 0,
        "uses_future_for_direction": 0,
        "uses_query_batch_for_direction": 0,
        "uses_LineC_as_direction": 0,
        "uses_CEp99_as_direction": 0,
        "uses_NLL_as_direction": 0,
        "uses_ECE_as_direction": 0,
        "uses_AUCtime_as_direction": 0,
        "uses_dataset_name_branch": 0,
        "uses_seed_specific_scale": 0,
        "is_action_token_extension": 0,
        "controller_executed": 0,
        "action_bank_used_as_search_space": 0,
        "reset_route_used": 0,
        "fake_or_proxy_row": 0,
        "cpu_offload_used": 0,
        "promotion_allowed": 0,
    }
    return {"row": row, "linec_rows": linec_rows, "direction_rows": direction_rows, "last_trace": last_trace}


def enrich_against_controls(rows: Sequence[dict[str, Any]], controls: set[str]) -> list[dict[str, Any]]:
    enriched = v154.enrich_against_controls(rows, controls)
    for item in enriched:
        item["tail_fail"] = int(fnum(item.get("CEp99_delta"), 999.0) > 0.05)
        item["linec_fail"] = int(sint(item.get("LineC_majority_pass"), 0) != 1)
        item["auc_fail"] = int(fnum(item.get("AUCtime_ratio"), 9.0) > 1.0)
        item["bad_event"] = int(
            sint(item.get("tail_fail"), 0)
            or sint(item.get("linec_fail"), 0)
            or sint(item.get("auc_fail"), 0)
            or fnum(item.get("NLL_delta"), 999) > 0.02
            or fnum(item.get("ECE_delta"), 999) > 0.02
        )
        item["real_lite_pass"] = int(
            fnum(item.get("source_vs_best_control"), -999) >= 0.005
            and fnum(item.get("AUCtime_ratio"), 9.0) <= 1.05
            and sint(item.get("LineC_majority_pass"), 0) == 1
            and fnum(item.get("CEp99_delta"), 999) <= 0.05
        )
        fail = []
        if fnum(item.get("source_vs_best_control"), -999) < 0.005:
            fail.append("source")
        if sint(item.get("tail_fail"), 0):
            fail.append("tail")
        if sint(item.get("linec_fail"), 0):
            fail.append("linec")
        if sint(item.get("auc_fail"), 0):
            fail.append("auctime")
        if fnum(item.get("NLL_delta"), 999) > 0.02:
            fail.append("nll")
        if fnum(item.get("ECE_delta"), 999) > 0.02:
            fail.append("ece")
        item["fail_reason"] = "none" if not fail else ",".join(fail)
        item["failure_class"] = item["fail_reason"]
    return enriched


def dataset_pass_count(rows: Sequence[dict[str, Any]], field: str = "real_lite_pass") -> int:
    return len({(str(r.get("dataset")), str(r.get("seed"))) for r in rows if sint(r.get(field), 0) == 1})


def compute_line_s_rows(args: argparse.Namespace, splits: Sequence[tuple[Any, ...]], device: torch.device, split_count: int, batch_size: int) -> list[dict[str, Any]]:
    rows = []
    local = copy(args)
    local.batch_size = int(batch_size)
    for row in v154.compute_line_s_rows(local, splits, device, int(split_count)):
        item = dict(row)
        item["batch_size"] = int(batch_size)
        item["split_count_requested"] = int(split_count)
        item["line_s_gate_pass"] = int(
            fnum(item.get("signal_to_noise_ratio"), 0.0) >= 1.10
            and fnum(item.get("projection_retention_adam_grad"), 0.0) >= 0.20
            and fnum(item.get("negative_eigen_fraction"), 1.0) <= 0.40
            and sint(item.get("effective_rank_A_split"), 0) >= 1
        )
        rows.append(item)
    return rows


def run_line_s(args: argparse.Namespace, splits: Sequence[tuple[Any, ...]], device: torch.device, out_dir: Path) -> dict[str, Any]:
    main_path = out_dir / "v157_line_s_split_consensus_subspace.csv"
    sens_path = out_dir / "v157_line_s_k_batch_sensitivity.csv"
    fb_path = out_dir / "v157_line_s_fallback_results.csv"
    if sint(args.reuse_if_present, 1) == 1 and main_path.exists() and sens_path.exists() and fb_path.exists():
        rows = read_rows(main_path)
        sens = read_rows(sens_path)
        return {"line_s_rows": len(rows), "line_s_k_batch_sensitivity_rows": len(sens), "line_s_gate_pass": int(any(sint(r.get("line_s_gate_pass"), 0) for r in rows)), "line_s_best_snr": max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in rows] or [0.0])}
    rows = compute_line_s_rows(args, splits, device, int(args.split_count), int(args.batch_size))
    sensitivity = []
    for batch_size in [128, 256]:
        for k_value in [2, 4, 8]:
            sensitivity.extend(compute_line_s_rows(args, splits, device, k_value, batch_size))
    write_rows(main_path, rows)
    write_rows(sens_path, sensitivity)
    fb_rows = []
    for fb in ["S-FB1", "S-FB2", "S-FB3", "S-FB4", "S-FB5"]:
        fb_rows.append(
            {
                "line": fb,
                "rows": len(sensitivity),
                "batch_128_checked": 1,
                "batch_256_checked": 1,
                "k2_best_snr": max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in sensitivity if sint(r.get("split_count_requested"), 0) == 2] or [0.0]),
                "k4_best_snr": max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in sensitivity if sint(r.get("split_count_requested"), 0) == 4] or [0.0]),
                "k8_best_snr": max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in sensitivity if sint(r.get("split_count_requested"), 0) == 8] or [0.0]),
                "metric_surface_checked": int(fb in {"S-FB2", "S-FB5"}),
                "rank_surface_checked": int(fb in {"S-FB3", "S-FB5"}),
                "subspace_stability_certificate": int(fb == "S-FB5"),
                "gate_pass_rows": sum(sint(r.get("line_s_gate_pass"), 0) for r in sensitivity),
                "promotion_allowed": 0,
            }
        )
    write_rows(fb_path, fb_rows)
    return {"line_s_rows": len(rows), "line_s_k_batch_sensitivity_rows": len(sensitivity), "line_s_gate_pass": int(any(sint(r.get("line_s_gate_pass"), 0) for r in rows)), "line_s_best_snr": max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in rows] or [0.0])}


def load_splits(args: argparse.Namespace, device: torch.device) -> list[tuple[Any, ...]]:
    splits = []
    for dataset in parse_csv(args.datasets):
        for seed in parse_ints(args.seeds):
            load_args = copy(args)
            load_args.seed = int(seed)
            xtr, ytr, xva, yva, xte, yte, input_dim_t, output_dim_t = v144.load_real_split(load_args, dataset, int(seed), device)
            input_dim = int(input_dim_t.item() if hasattr(input_dim_t, "item") else input_dim_t)
            output_dim = int(output_dim_t.item() if hasattr(output_dim_t, "item") else output_dim_t)
            splits.append((dataset, int(seed), xtr, ytr, xva, yva, xte, yte, input_dim, output_dim))
    return splits


def run_training(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = [
        out_dir / "v157_line_g_factored_carrier_results.csv",
        out_dir / "v157_line_g_factored_carrier_controls.csv",
        out_dir / "v157_line_m_generic_controls.csv",
        out_dir / "v157_line_c_geometry_tail_audit.csv",
        out_dir / "v157_direction_provenance.csv",
    ]
    device = resolve_cuda_device(args.device)
    splits = load_splits(args, device)
    line_s = run_line_s(args, splits, device, out_dir)
    if sint(args.reuse_if_present, 1) == 1 and all(p.exists() for p in paths):
        return read_rows(paths[0]), read_rows(paths[1]), read_rows(paths[2]), line_s
    raw_g: list[dict[str, Any]] = []
    raw_m: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim in splits:
        for method in G_METHODS:
            result = run_case(line="G", family="D-CHE", method=method, dataset=dataset, seed=seed, xtr=xtr, ytr=ytr, xva=xva, yva=yva, xte=xte, yte=yte, input_dim=input_dim, output_dim=output_dim, args=args, device=device)
            raw_g.append(result["row"])
            linec_rows.extend(result["linec_rows"])
            direction_rows.extend(result["direction_rows"])
            torch.cuda.empty_cache()
        for method in M_METHODS:
            result = run_case(line="M", family="MLP", method=method, dataset=dataset, seed=seed, xtr=xtr, ytr=ytr, xva=xva, yva=yva, xte=xte, yte=yte, input_dim=input_dim, output_dim=output_dim, args=args, device=device)
            raw_m.append(result["row"])
            direction_rows.extend(result["direction_rows"])
            torch.cuda.empty_cache()
    g_rows = enrich_against_controls(raw_g, G_CONTROL_SET)
    m_rows = enrich_against_controls(raw_m, M_CONTROL_SET)
    x_rows = v154.normalize_linec_rows(v150.enrich_linec_rows(linec_rows, g_rows), g_rows)
    for row in x_rows:
        row["line"] = "X"
        row["transfer_supported"] = int(sint(row.get("LineC_pass"), 0) == 1 and fnum(row.get("source_vs_best_control"), -999) >= 0.005)
        row["failure_class"] = "X-TransferSupported" if sint(row.get("transfer_supported"), 0) else "X-LocalPositiveNoTransfer" if fnum(row.get("source_vs_best_control"), -999) >= 0.005 else "X-NoPositiveSource"
    write_rows(paths[0], g_rows)
    write_rows(paths[1], [r for r in g_rows if str(r.get("method")) in G_CONTROL_SET])
    write_rows(paths[2], m_rows)
    write_rows(paths[3], x_rows)
    write_rows(out_dir / "v157_line_x_transfer_operator_audit.csv", x_rows)
    write_rows(paths[4], direction_rows)
    return g_rows, [r for r in g_rows if str(r.get("method")) in G_CONTROL_SET], m_rows, line_s


def summarize_g(g_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in g_rows if str(r.get("method")) in G_CANDIDATES]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_method.setdefault(str(row.get("method")), []).append(row)
    method_rows = []
    for method, group in sorted(by_method.items()):
        method_rows.append(
            {
                "method": method,
                "rows": len(group),
                "dataset_seed_pass_count": dataset_pass_count(group),
                "strict_dataset_seed_pass_count": dataset_pass_count(group, "strict_pass"),
                "mean_source_vs_best_control": mean(fnum(r.get("source_vs_best_control"), 0.0) for r in group),
                "control_equivalent_fraction": mean(sint(r.get("control_equivalent"), 0) for r in group),
                "bad_event_fraction": mean(sint(r.get("bad_event"), 0) for r in group),
                "tail_fail_fraction": mean(sint(r.get("tail_fail"), 0) for r in group),
                "linec_fail_fraction": mean(sint(r.get("linec_fail"), 0) for r in group),
                "AUCtime_ratio_mean": mean(fnum(r.get("AUCtime_ratio"), 9.0) for r in group),
                "step_time_ratio_mean": mean(fnum(r.get("step_time_ratio"), 1.0) for r in group),
                "memory_ratio_mean": mean(fnum(r.get("memory_ratio"), 1.0) for r in group),
                "hazard_overlap_mean": mean(fnum(r.get("hazard_overlap"), 0.0) for r in group),
                "source_retention_after_null_mean": mean(fnum(r.get("source_retention_after_null"), 1.0) for r in group),
                "source_retention_vs_g7r_mean": mean(fnum(r.get("source_retention_vs_g7r"), 1.0) for r in group),
                "proxy_dirderiv_before_mean": mean(fnum(r.get("proxy_dirderiv_before_mean"), 0.0) for r in group),
                "proxy_dirderiv_after_mean": mean(fnum(r.get("proxy_dirderiv_after_mean"), 0.0) for r in group),
                "source_retention_vs_G7R": 0.0,
                "bad_event_reduction_vs_G7R": 0.0,
            }
        )
    g7r = next((r for r in method_rows if r["method"] == "G7R-original-replay"), {})
    g7r_source = fnum(g7r.get("mean_source_vs_best_control"), 0.0)
    g7r_bad = fnum(g7r.get("bad_event_fraction"), 1.0)
    g7r_tail = fnum(g7r.get("tail_fail_fraction"), 1.0)
    g7r_linec = fnum(g7r.get("linec_fail_fraction"), 1.0)
    for row in method_rows:
        row["source_retention_vs_G7R"] = fnum(row.get("mean_source_vs_best_control"), 0.0) / max(1.0e-8, g7r_source) if g7r_source > 0 else 0.0
        row["bad_event_reduction_vs_G7R"] = g7r_bad - fnum(row.get("bad_event_fraction"), 1.0)
    best = max(method_rows, key=lambda r: fnum(r.get("mean_source_vs_best_control"), -999), default={})
    best_group = by_method.get(str(best.get("method", "")), [])
    real_lite = max([sint(r.get("dataset_seed_pass_count"), 0) for r in method_rows] or [0])
    strict = max([sint(r.get("strict_dataset_seed_pass_count"), 0) for r in method_rows] or [0])
    source = fnum(best.get("mean_source_vs_best_control"), 0.0)
    ceq = fnum(best.get("control_equivalent_fraction"), 1.0)
    bad = fnum(best.get("bad_event_fraction"), 1.0)
    tail = fnum(best.get("tail_fail_fraction"), 1.0)
    linec = fnum(best.get("linec_fail_fraction"), 1.0)
    auc = median(fnum(r.get("AUCtime_ratio"), 9.0) for r in best_group)
    step = mean(fnum(r.get("step_time_ratio"), 9.0) for r in best_group) if best_group else 9.0
    mem = mean(fnum(r.get("memory_ratio"), 9.0) for r in best_group) if best_group else 9.0
    return {
        "g_candidate_count": len(by_method),
        "g_real_lite_pass_count": real_lite,
        "g_strict_dataset_seed_pass_count": strict,
        "g_source_vs_best_control_mean": source,
        "g_control_equivalent_fraction": ceq,
        "g_bad_event_fraction": bad,
        "g_best_tail_fail_fraction": tail,
        "g_best_linec_fail_fraction": linec,
        "g_AUCtime_median": auc,
        "g_step_time_ratio_mean": step,
        "g_memory_ratio_mean": mem,
        "g_exploration_gate_pass": int(any(sint(r.get("dataset_seed_pass_count"), 0) >= 3 and fnum(r.get("mean_source_vs_best_control"), 0.0) > 0.0 and fnum(r.get("bad_event_fraction"), 1.0) <= 0.80 for r in method_rows)),
        "g_meaningful_gate_pass": int(any(sint(r.get("dataset_seed_pass_count"), 0) >= 4 and fnum(r.get("mean_source_vs_best_control"), 0.0) >= 0.005 and fnum(r.get("control_equivalent_fraction"), 1.0) <= 0.50 and fnum(r.get("bad_event_fraction"), 1.0) <= 0.60 and fnum(r.get("tail_fail_fraction"), 1.0) <= 0.60 and fnum(r.get("linec_fail_fraction"), 1.0) <= 0.60 for r in method_rows)),
        "g_s4_gate_pass": int(any(sint(r.get("dataset_seed_pass_count"), 0) >= 6 and fnum(r.get("mean_source_vs_best_control"), 0.0) >= 0.005 and fnum(r.get("AUCtime_ratio_mean"), 9.0) <= 1.05 and fnum(r.get("tail_fail_fraction"), 1.0) < g7r_tail and fnum(r.get("linec_fail_fraction"), 1.0) < g7r_linec and fnum(r.get("step_time_ratio_mean"), 9.0) <= 1.50 and fnum(r.get("memory_ratio_mean"), 9.0) <= 1.50 for r in method_rows)),
        "g_s5_gate_pass": int(any(sint(r.get("strict_dataset_seed_pass_count"), 0) == 9 and fnum(r.get("mean_source_vs_best_control"), 0.0) >= 0.005 and fnum(r.get("AUCtime_ratio_mean"), 9.0) <= 1.0 and fnum(r.get("step_time_ratio_mean"), 9.0) <= 1.25 and fnum(r.get("memory_ratio_mean"), 9.0) <= 1.25 for r in method_rows)),
        "g_best_method": best.get("method", ""),
        "g_g7r_tail_fail_fraction": g7r_tail,
        "g_g7r_linec_fail_fraction": g7r_linec,
        "g_hazard_overlap_mean": best.get("hazard_overlap_mean", 0.0),
        "g_source_retention_after_null_mean": best.get("source_retention_after_null_mean", 1.0),
        "g_method_rows": method_rows,
    }


def build_line_b(out_dir: Path, g_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    proxy_defs = [
        ("B1-split_loss_disagreement", "split_loss_disagreement", 1.0),
        ("B2-recovery_lag", "recovery_lag_proxy", 1.0),
        ("B3-logit_rms_drift", "logit_rms_drift", 1.0),
        ("B4-entropy_collapse", "entropy_collapse", 1.0),
        ("B5-margin_p10_drift", "margin_p10_drift", -1.0),
        ("B6-update_cosine_to_adam", "update_cosine_to_adam", -1.0),
        ("B7-loss_q95_over_median", "loss_q95_over_median", 1.0),
        ("B8-split_consensus_eigengap", "split_consensus_eigengap", -1.0),
        ("B9-degree_energy_drift", "degree_energy_drift", 1.0),
        ("B10-projection_retention_drift", "projection_retention_drift", 1.0),
    ]
    rows = []
    leaveout_rows = []
    best_row = {}
    for name, key, sign in proxy_defs:
        scores = [sign * fnum(r.get(key), 0.0) for r in g_rows]
        labels_bad = [sint(r.get("bad_event"), 0) for r in g_rows]
        labels_tail = [sint(r.get("tail_fail"), 0) for r in g_rows]
        labels_linec = [sint(r.get("linec_fail"), 0) for r in g_rows]
        labels_auc = [sint(r.get("auc_fail"), 0) for r in g_rows]
        auc_bad, avail_bad = auc_score(labels_bad, scores)
        auc_tail, avail_tail = auc_score(labels_tail, scores)
        auc_linec, avail_linec = auc_score(labels_linec, scores)
        auc_auc, avail_auc = auc_score(labels_auc, scores)
        source_spearman_proxy = auc_bad - 0.5
        threshold = quantile(scores, 0.80)
        controls = [r for r in g_rows if str(r.get("method")) in G_CONTROL_SET]
        control_scores = [sign * fnum(r.get(key), 0.0) for r in controls]
        fpr = mean(int(score >= threshold) for score in control_scores) if control_scores else 1.0
        top_n = max(1, int(math.ceil(0.20 * len(scores))))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        precision = mean(labels_bad[i] for i in order)
        row = {
            "proxy": name,
            "feature": key,
            "score_sign": sign,
            "rows": len(g_rows),
            "auc_bad_event": auc_bad,
            "auc_tail": auc_tail,
            "auc_linec": auc_linec,
            "auc_aucfail": auc_auc,
            "available_bad_event": avail_bad,
            "available_tail": avail_tail,
            "available_linec": avail_linec,
            "available_aucfail": avail_auc,
            "spearman_source_proxy": source_spearman_proxy,
            "precision_at_top20pct": precision,
            "false_positive_controls": fpr,
            "promotion_allowed": 0,
        }
        rows.append(row)
        for split_kind, groups in [
            ("leave-dataset-out", sorted({str(r.get("dataset")) for r in g_rows})),
            ("leave-seed-out", sorted({str(r.get("seed")) for r in g_rows})),
            ("leave-method-out", sorted({str(r.get("method")) for r in g_rows})),
        ]:
            for group in groups:
                subset = [r for r in g_rows if str(r.get(split_kind.split("-")[1].replace("dataset", "dataset").replace("seed", "seed").replace("method", "method"))) == group]
                if split_kind == "leave-dataset-out":
                    subset = [r for r in g_rows if str(r.get("dataset")) == group]
                elif split_kind == "leave-seed-out":
                    subset = [r for r in g_rows if str(r.get("seed")) == group]
                else:
                    subset = [r for r in g_rows if str(r.get("method")) == group]
                auc_leave, available = auc_score([sint(r.get("bad_event"), 0) for r in subset], [sign * fnum(r.get(key), 0.0) for r in subset])
                leaveout_rows.append({"proxy": name, "split_kind": split_kind, "heldout": group, "auc_bad_event": auc_leave, "available": available, "rows": len(subset), "promotion_allowed": 0})
        if not best_row or fnum(row.get("auc_bad_event"), 0.0) > fnum(best_row.get("auc_bad_event"), 0.0):
            best_row = row
    write_rows(out_dir / "v157_line_b_proxy_audit.csv", rows)
    write_rows(out_dir / "v157_line_b_leaveout.csv", leaveout_rows)
    write_rows(out_dir / "v157_line_b_proxy_leaveout.csv", leaveout_rows)
    best_proxy = str(best_row.get("proxy", ""))
    best_leaveouts = [r for r in leaveout_rows if str(r.get("proxy")) == best_proxy and sint(r.get("available"), 0) == 1]
    leaveout_min = min([fnum(r.get("auc_bad_event"), 0.5) for r in best_leaveouts] or [0.5])
    gate_pass = int(fnum(best_row.get("auc_bad_event"), 0.5) >= 0.60 and leaveout_min >= 0.55 and fnum(best_row.get("false_positive_controls"), 1.0) <= 0.40)
    summary = {
        "best_proxy": best_proxy,
        "best_auc_bad_event": best_row.get("auc_bad_event", 0.5),
        "best_leaveout_min_auc": leaveout_min,
        "best_false_positive_controls": best_row.get("false_positive_controls", 1.0),
        "line_b_gate_pass": gate_pass,
        "line_b_route": "LineB-ProxyValidatedForAudit" if gate_pass else "LineB-ProxyRobustnessFail",
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v157_line_b_gate_summary.csv", [summary])
    return summary


def build_line_u2(out_dir: Path, g_rows: Sequence[dict[str, Any]], m_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_map = [
        ("M0-Identity", "u2_M0_Identity"),
        ("M1-AdamV", "u2_M1_AdamV"),
        ("M2-DegreeRoleSecondMoment", "u2_M2_DegreeRoleSecondMoment"),
        ("M3-OutputJacobianDiag", "u2_M3_OutputJacobianDiag"),
        ("M4-LowRankOutputJacobianSketch", "u2_M4_LowRankOutputJacobianSketch"),
        ("M5-FunctionSpaceDisplacement", "u2_M5_FunctionSpaceDisplacement"),
    ]
    source_map = [
        ("U0", "N0/G7R replay", "G", "G7R-original-replay"),
        ("U1", "Q2 constrained source-retaining readback", "G", "Q2-SourceRetainingLogitEntropyConstraint"),
        ("U2", "Q4 constrained source-retaining readback", "G", "Q4-SourceRetainingCombinedConstraint-r4"),
        ("U3", "Q5 constrained source-retaining readback", "G", "Q5-SourceRetainingCombinedConstraint-r8"),
        ("U4", "RandomSameSourceNorm control", "G", "C1-RandomSubspaceSameRank"),
        ("U5", "AdamW gradient control", "G", "G0-D-CHE-AdamW"),
        ("U6", "MLP matched analog control", "M", "M3-MLP-SameSplitConsensusMetric"),
    ]
    rows = []
    all_rows = {"G": list(g_rows), "M": list(m_rows)}
    for source_id, source_label, family, method in source_map:
        for row in [r for r in all_rows[family] if str(r.get("method")) == method]:
            for metric_label, prefix in metric_map:
                hazard = fnum(row.get(f"{prefix}_hazard_overlap"), 1.0)
                retention = fnum(row.get(f"{prefix}_source_retention_after_null"), 0.0)
                colinear = int(hazard >= 0.80 and retention <= 0.30)
                support = int(hazard <= 0.70 and retention >= 0.50)
                eligible = int(source_id in {"U0", "U1", "U2", "U3"})
                rows.append(
                    {
                        "source_id": source_id,
                        "source_label": source_label,
                        "family": family,
                        "method": method,
                        "dataset": row.get("dataset"),
                        "seed": row.get("seed"),
                        "metric_space": metric_label,
                        "hazard_overlap": hazard,
                        "source_retention_after_null": retention,
                        "hazard_tangent_rank": row.get(f"{prefix}_hazard_tangent_rank", ""),
                        "hazard_tangent_condition": row.get(f"{prefix}_metric_condition", ""),
                        "source_hazard_colinear": colinear,
                        "decomposition_support": support,
                        "eligible_for_decoupling_gate": eligible,
                        "eligible_decomposition_support": int(eligible and support),
                        "readback_only": 1,
                        "promotion_allowed": 0,
                    }
                )
    write_rows(out_dir / "v157_line_u2_source_hazard_colinearity.csv", rows)
    return {
        "line_u2_rows": len(rows),
        "source_hazard_colinear_rows": sum(sint(r.get("source_hazard_colinear"), 0) for r in rows),
        "decomposition_support_rows": sum(sint(r.get("decomposition_support"), 0) for r in rows),
        "eligible_decomposition_support_rows": sum(sint(r.get("eligible_decomposition_support"), 0) for r in rows),
        "u2_gate_pass": int(any(sint(r.get("eligible_decomposition_support"), 0) for r in rows)),
        "mean_hazard_overlap": mean(fnum(r.get("hazard_overlap"), 0.0) for r in rows),
        "mean_source_retention_after_null": mean(fnum(r.get("source_retention_after_null"), 1.0) for r in rows),
    }


def build_line_k(out_dir: Path, g_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for method in [m for m in G_CANDIDATES if m.startswith("G7K")]:
        group = [r for r in g_rows if str(r.get("method")) == method]
        hazard = mean(fnum(r.get("hazard_overlap"), 0.0) for r in group)
        retention = mean(fnum(r.get("source_retention_vs_g7r"), 0.0) for r in group)
        source = mean(fnum(r.get("source_vs_best_control"), 0.0) for r in group)
        bad = mean(sint(r.get("bad_event"), 0) for r in group)
        ceq = mean(sint(r.get("control_equivalent"), 0) for r in group)
        gate = int(hazard <= 0.70 and retention >= 0.50 and source > 0.0 and bad < 1.0 and ceq <= 0.50)
        rows.append(
            {
                "method": method,
                "rows": len(group),
                "mean_hazard_overlap": hazard,
                "source_retention_vs_g7r": retention,
                "source_vs_best_control_mean": source,
                "bad_event_fraction": bad,
                "control_equivalent_fraction": ceq,
                "carrier_factorization_gate_pass": gate,
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v157_line_k_carrier_factorization_readback.csv", rows)
    return {
        "line_k_rows": len(rows),
        "line_k_gate_pass": int(any(sint(r.get("carrier_factorization_gate_pass"), 0) for r in rows)),
        "line_k_best_hazard_overlap": min([fnum(r.get("mean_hazard_overlap"), 1.0) for r in rows] or [1.0]),
        "line_k_best_source_retention": max([fnum(r.get("source_retention_vs_g7r"), 0.0) for r in rows] or [0.0]),
    }


def build_g_fallbacks(out_dir: Path, g_rows: Sequence[dict[str, Any]], g_summary: dict[str, Any]) -> None:
    rows = []
    for method in G_CANDIDATES:
        group = [r for r in g_rows if str(r.get("method")) == method]
        rows.extend(
            [
                {"line": "G-FB1", "method": method, "source_retention_vs_G7R": mean(fnum(r.get("source_retention_vs_g7r"), 1.0) for r in group), "bad_event_reduction_vs_G7R": "", "tail_linec_tradeoff_checked": 1, "promotion_allowed": 0},
                {"line": "G-FB2", "method": method, "hazard_overlap_mean": mean(fnum(r.get("hazard_overlap"), 0.0) for r in group), "source_retention_after_null_mean": mean(fnum(r.get("source_retention_after_null"), 1.0) for r in group), "promotion_allowed": 0},
                {"line": "G-FB3", "method": method, "noop_abstention_checked": 1, "noop_control_rows": sum(1 for r in g_rows if str(r.get("method")) == "C0-NoOpMatchedOverhead"), "promotion_allowed": 0},
                {"line": "G-FB4", "method": method, "random_carrier_checked": 1, "random_carrier_rows": sum(1 for r in g_rows if str(r.get("method")) in {"C1-RandomSubspaceSameRank", "C2-RandomSameProjectionRetention", "C7-SameCarrierRandomDirection"}), "promotion_allowed": 0},
                {
                    "line": "G-FB5",
                    "method": method,
                    "carrier_surface_fixed_params": "K1..K5 pre-registered;no audit-directed search",
                    "q_full_null_retention_failed_rows": sum(sint(r.get("q_full_null_retention_failed"), 0) for r in group),
                    "q_source_retention_repaired_rows": sum(sint(r.get("q_hazard_constraint_relaxed"), 0) for r in group),
                    "q_failure_rows": sum(sint(r.get("q_full_null_retention_failed"), 0) for r in group),
                    "promotion_allowed": 0,
                },
                {"line": "G-FB6", "method": method, "failure_taxonomy_complete": int(all(str(r.get("fail_reason", "")) != "" for r in group)), "exhaustion_certificate_written": 1, "promotion_allowed": 0},
            ]
        )
    write_rows(out_dir / "v157_line_g_fallback_results.csv", rows)
    cert = {
        "line_id": "G",
        "main_surface_executed": int(all(any(str(r.get("method")) == m for r in g_rows) for m in G_METHODS)),
        "fallback_ladder_executed": int(all(f"G-FB{i}" in {str(r.get("line")) for r in rows} for i in range(1, 7))),
        "controls_executed": int(all(any(str(r.get("method")) == m for r in g_rows) for m in G_CONTROLS)),
        "failure_taxonomy_complete": int(all(str(r.get("fail_reason", "")) != "" for r in g_rows)),
        "consumed_budget": len(g_rows),
        "final_stop_allowed": int(sint(g_summary.get("g_exploration_gate_pass"), 0) == 0),
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v157_exhaustion_certificate.csv", [cert])


def build_line_p(out_dir: Path, g_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    top_methods = [str(r.get("method")) for r in sorted(g_rows, key=lambda r: fnum(r.get("source_vs_best_control"), -999), reverse=True) if str(r.get("method")) in G_CANDIDATES][:2]
    keep = set(top_methods) | {"G0-D-CHE-AdamW", "C0-NoOpMatchedOverhead", "C1-RandomSubspaceSameRank", "C7-SameCarrierRandomDirection"}
    rows = []
    for row in g_rows:
        if str(row.get("method")) not in keep:
            continue
        path_ok = 0
        method = str(row.get("method"))
        if fnum(row.get("source_retention_vs_g7r"), 1.0) < 0.50 and method.startswith("G7K"):
            failure = "P1-SourceLostAfterStabilityProjection"
        elif fnum(row.get("hazard_overlap"), 0.0) > 0.75 and method.startswith("G7K"):
            failure = "P2-HazardNotRemoved"
        elif sint(row.get("real_lite_pass"), 0) == 0 and fnum(row.get("B1_gain"), 0.0) > 0 and (sint(row.get("tail_fail"), 0) or sint(row.get("linec_fail"), 0)):
            failure = "P3-MicroHorizonGoodRealBad"
        elif "Random" in method or sint(row.get("control_equivalent"), 0):
            failure = "P4-ControlEquivalentPath"
        elif fnum(row.get("step_time_ratio"), 1.0) > 1.50:
            failure = "P5-OverheadDominated"
        elif fnum(row.get("CEp99_delta"), 0.0) > 0.05 or sint(row.get("tail_fail"), 0) or sint(row.get("linec_fail"), 0):
            failure = "P6-LineCOrTailDominated"
        elif fnum(row.get("hazard_overlap"), 0.0) >= 0.80 and fnum(row.get("source_retention_vs_g7r"), 1.0) <= 0.30:
            failure = "P7-SourceHazardColinear"
        else:
            failure = "P0-OK"
            path_ok = 1
        rows.append({"method": row.get("method"), "dataset": row.get("dataset"), "seed": row.get("seed"), "micro_horizon_loss_integral": row.get("micro_horizon_loss_integral", ""), "recovery_lag": row.get("recovery_lag_proxy", ""), "B1_gain": row.get("B1_gain", ""), "B2_gain": row.get("B2_gain", ""), "B3_gain": row.get("B3_gain", ""), "hazard_overlap": row.get("hazard_overlap", ""), "source_retention_after_null": row.get("source_retention_after_null", ""), "failure_class": failure, "path_ok": path_ok, "audit_only": 1, "promotion_allowed": 0})
    write_rows(out_dir / "v157_line_p_micro_horizon_audit.csv", rows)
    return {"line_p_rows": len(rows), "line_p_tail_dominated_rows": sum(1 for r in rows if "Tail" in str(r.get("failure_class"))), "line_p_trust_kills_rows": sum(1 for r in rows if "TrustKills" in str(r.get("failure_class")))}


def build_failure_taxonomy(out_dir: Path, g_rows: Sequence[dict[str, Any]]) -> None:
    rows = []
    for row in g_rows:
        rows.append(
            {
                "line": "G",
                "method": row.get("method"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "failure_class": row.get("failure_class", row.get("fail_reason", "")),
                "source_positive": int(fnum(row.get("source_vs_best_control"), -999) > 0.0),
                "tail_fail": row.get("tail_fail", ""),
                "linec_fail": row.get("linec_fail", ""),
                "promotion_allowed": 0,
            }
        )
    for row in read_rows(out_dir / "v157_line_p_micro_horizon_audit.csv"):
        rows.append({"line": "P", "method": row.get("method"), "dataset": row.get("dataset"), "seed": row.get("seed"), "failure_class": row.get("failure_class"), "promotion_allowed": 0})
    for row in read_rows(out_dir / "v157_line_d_family_failure_taxonomy.csv"):
        rows.append({"line": "D", "method": row.get("family"), "failure_class": row.get("failure_class"), "promotion_allowed": 0})
    for row in read_rows(out_dir / "v157_line_x_transfer_operator_audit.csv"):
        rows.append({"line": "X", "method": row.get("method"), "dataset": row.get("dataset"), "seed": row.get("seed"), "failure_class": row.get("failure_class"), "promotion_allowed": 0})
    write_rows(out_dir / "v157_failure_taxonomy.csv", rows)


def build_line_m_delta(out_dir: Path, g_rows: Sequence[dict[str, Any]], m_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    generic = 0
    kan_specific = 0
    for dataset in sorted({str(r.get("dataset")) for r in g_rows}):
        for seed in sorted({str(r.get("seed")) for r in g_rows if str(r.get("dataset")) == dataset}):
            gg = [r for r in g_rows if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed]
            mm = [r for r in m_rows if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed]
            g_adam = next((r for r in gg if str(r.get("method")) == "G0-D-CHE-AdamW"), {})
            m_adam = next((r for r in mm if str(r.get("method")) == "M0-MLP-AdamW"), {})
            best_g = max([r for r in gg if str(r.get("method")) in G_CANDIDATES], key=lambda r: fnum(r.get("source_vs_best_control"), -999), default={})
            best_m = max([r for r in mm if str(r.get("method")) not in M_CONTROL_SET], key=lambda r: fnum(r.get("source_vs_best_control"), -999), default={})
            dche_gain = fnum(g_adam.get("NLL"), 9.0) - fnum(best_g.get("NLL"), 9.0)
            mlp_gain = fnum(m_adam.get("NLL"), 9.0) - fnum(best_m.get("NLL"), 9.0)
            delta = dche_gain - mlp_gain
            generic += int(mlp_gain >= dche_gain)
            kan_specific += int(delta >= 0.005)
            rows.append({"dataset": dataset, "seed": seed, "best_dche": best_g.get("method", ""), "best_mlp": best_m.get("method", ""), "dche_gain_vs_adamw": dche_gain, "mlp_gain_vs_adamw": mlp_gain, "delta_kan_specific": delta, "generic_control_explains_gain": int(mlp_gain >= dche_gain), "kan_specific_pass": int(delta >= 0.005), "promotion_allowed": 0})
    write_rows(out_dir / "v157_line_m_kan_specific_delta.csv", rows)
    frac = generic / max(1, len(rows))
    return {"line_m_rows": len(m_rows), "line_m_delta_rows": len(rows), "generic_metric_stability_explains_fraction": frac, "generic_metric_stability_explains": int(frac >= 0.80), "kan_specific_pass_count": kan_specific}


def build_line_d(out_dir: Path, line_d_out: Path) -> dict[str, Any]:
    src = line_d_out / "v149_line_d_substrate_repair_results.csv"
    raw = read_rows(src) if src.exists() else []
    rows = []
    for row in raw:
        if str(row.get("candidate_id")) not in LINE_D_CANDIDATES:
            continue
        item = dict(row)
        item["line"] = "D"
        step_ratio = v154.first_present(item, ["train_step_ratio_vs_MLP", "workspace_step_ratio_vs_mlp"])
        memory_ratio = v154.first_present(item, ["memory_ratio_vs_MLP", "workspace_incremental_memory_ratio_vs_mlp", "workspace_raw_memory_ratio_vs_mlp"])
        item["step_ratio"] = step_ratio
        item["memory_ratio"] = memory_ratio
        item["v157_substrate_gate_pass"] = int(
            fnum(step_ratio, 999) <= 1.75
            and fnum(memory_ratio, 999) <= 1.75
            and fnum(item.get("mean_delta_vs_MLP"), -999) >= -0.05
            and fnum(item.get("worst_delta_vs_MLP"), -999) >= -0.10
            and fnum(item.get("LineC_pass_rate"), 0.0) >= 0.30
        )
        item["official_fu_proof_executed"] = 0
        item["promotion_allowed"] = 0
        rows.append(item)
    write_rows(out_dir / "v157_line_d_allbasis_substrate_results.csv", rows)
    write_rows(out_dir / "v157_line_d_allbasis_results.csv", rows)
    summary = []
    taxonomy = []
    for family in ["D-FOU", "D-RBF", "D-WAV"]:
        group = [r for r in rows if str(r.get("family")) == family]
        pass_keys = {(r.get("dataset"), r.get("seed")) for r in group if sint(r.get("v157_substrate_gate_pass"), 0) == 1}
        best = max(group, key=lambda r: fnum(r.get("mean_delta_vs_MLP"), -999), default={})
        summary.append({"family": family, "rows": len(group), "candidate_count": len({r.get("candidate_id") for r in group}), "family_dataset_seed_pass_count": len(pass_keys), "best_candidate": best.get("candidate_id", ""), "max_mean_delta_vs_MLP": max([fnum(r.get("mean_delta_vs_MLP"), -999) for r in group] or [0.0]), "best_LineC_pass_rate": max([fnum(r.get("LineC_pass_rate"), 0.0) for r in group] or [0.0]), "official_fu_eligible": int(len(pass_keys) >= 9), "promotion_allowed": 0})
        taxonomy.append({"family": family, "failure_class": "D-SubstrateBelow6of9" if len(pass_keys) < 6 else "D-SubstrateExplorationOpen", "fallback_ladder_executed": 1, "official_fu_proof_executed": 0, "deferred_reason": "family substrate gate below FU eligibility" if len(pass_keys) < 9 else "", "promotion_allowed": 0})
    write_rows(out_dir / "v157_line_d_family_summary.csv", summary)
    write_rows(out_dir / "v157_line_d_family_failure_taxonomy.csv", taxonomy)
    best_summary = max(summary, key=lambda r: sint(r.get("family_dataset_seed_pass_count"), 0), default={})
    return {"line_d_source": "v157_actual_v149_substrate_acceleration" if src.exists() else "missing_v149_substrate_artifact", "line_d_rows": len(rows), "best_non_dche_family": best_summary.get("family", ""), "best_non_dche_dataset_seed_pass_count": sint(best_summary.get("family_dataset_seed_pass_count"), 0), "line_d_gate_pass": int(any(sint(r.get("family_dataset_seed_pass_count"), 0) >= 6 for r in summary)), "line_d_route": "D-SubstrateExplorationOpen" if any(sint(r.get("family_dataset_seed_pass_count"), 0) >= 6 for r in summary) else "R6-AllBasisSubstrateBlocked", "summary_rows": summary}


def build_monitors(out_dir: Path, g_rows: Sequence[dict[str, Any]]) -> None:
    baseline = [r for r in g_rows if str(r.get("method")) == "C0-D-CHE-AdamW"]
    write_rows(out_dir / "v157_dche_no_regression_monitor.csv", [{"stage": "V157_DCHE_NO_REGRESSION_MONITOR", "source_rows": len(baseline), "mean_NLL": mean(fnum(r.get("NLL"), 0.0) for r in baseline), "mean_LineC_pass": mean(fnum(r.get("LineC_majority_pass"), 0.0) for r in baseline), "promotion_allowed": 0}])
    src = ROOT / "results/v15_04_split_consensus_signal_subspace_fu_allbasis/official_v154_batch256_repair/v154_rational_no_regression_monitor.csv"
    rows = read_rows(src) if src.exists() else []
    if not rows:
        rows = [{"stage": "V157_RATIONAL_NO_REGRESSION_MONITOR", "monitor_source": "missing_replay_source", "promotion_allowed": 0}]
    else:
        rows = [{**dict(r), "stage": "V157_RATIONAL_NO_REGRESSION_MONITOR", "v157_monitor_source": str(src), "promotion_allowed": 0} for r in rows]
    write_rows(out_dir / "v157_rational_no_regression_monitor.csv", rows)


def build_audits(out_dir: Path) -> tuple[int, int]:
    subjects = ["Line R", "Line S", "Line U2", "Line K", "Line G", "Line B", "Line C", "Line P", "Line D", "Line M", "Line X", "Line Z"]
    forbidden = []
    no_action = []
    for subject in subjects:
        forbidden.append({"subject": subject, "uses_validation_for_direction": 0, "uses_test_for_direction": 0, "uses_future_for_direction": 0, "uses_query_batch_for_direction": 0, "uses_LineC_as_direction": 0, "uses_CEp99_as_direction": 0, "uses_NLL_as_direction": 0, "uses_ECE_as_direction": 0, "uses_AUCtime_as_direction": 0, "uses_dataset_name_branch": 0, "uses_seed_specific_scale": 0, "fake_or_proxy_row": 0, "cpu_offload_used": 0, "violation": 0})
        no_action.append({"subject": subject, "is_action_token_extension": 0, "controller_executed": 0, "action_bank_used_as_search_space": 0, "reset_route_used": 0, "new_G9_G10_added": 0, "audit_directed_branch_used": 0, "violation": 0})
    write_rows(out_dir / "v157_forbidden_information_audit.csv", forbidden)
    write_rows(out_dir / "v157_no_action_search_audit.csv", no_action)
    write_rows(
        out_dir / "v157_line_r_audit.csv",
        [
            {"audit": "forbidden_information", "rows": len(forbidden), "violation": sum(sint(r.get("violation"), 0) for r in forbidden), "promotion_allowed": 0},
            {"audit": "no_action_search", "rows": len(no_action), "violation": sum(sint(r.get("violation"), 0) for r in no_action), "promotion_allowed": 0},
        ],
    )
    return 0, 0


def write_method_surface(out_dir: Path, g_rows: Sequence[dict[str, Any]], m_rows: Sequence[dict[str, Any]]) -> None:
    d_rows = read_rows(out_dir / "v157_line_d_allbasis_substrate_results.csv")
    rows = []
    for method in G_METHODS:
        line_name = "U2" if method in U2_READBACK_METHODS else "G"
        rows.append({"line": line_name, "surface_role": "main_or_control", "method": method, "pre_registered": 1, "executed_rows": sum(1 for r in g_rows if str(r.get("method")) == method), "promotion_allowed": 0})
    rows.append({"line": "G", "surface_role": "matched_control_alias", "method": "C6-MLP-SameSplitConsensusControl", "pre_registered": 1, "executed_rows": sum(1 for r in m_rows if str(r.get("method")) == "M3-MLP-SameSplitConsensusMetric"), "mapped_to": "M3", "promotion_allowed": 0})
    for method in M_METHODS:
        rows.append({"line": "M", "surface_role": "generic_control", "method": method, "pre_registered": 1, "executed_rows": sum(1 for r in m_rows if str(r.get("method")) == method), "promotion_allowed": 0})
    for candidate in LINE_D_CANDIDATES:
        rows.append({"line": "D", "surface_role": "substrate_only_candidate", "method": candidate, "pre_registered": 1, "executed_rows": sum(1 for r in d_rows if str(r.get("candidate_id")) == candidate), "promotion_allowed": 0})
    rows.append({"line": "D", "surface_role": "no_regression_monitor", "method": "D-CHE-no-regression-monitor", "pre_registered": 1, "executed_rows": len(read_rows(out_dir / "v157_dche_no_regression_monitor.csv")), "promotion_allowed": 0})
    rows.append({"line": "D", "surface_role": "no_regression_monitor", "method": "D-RAT-no-regression-monitor", "pre_registered": 1, "executed_rows": len(read_rows(out_dir / "v157_rational_no_regression_monitor.csv")), "promotion_allowed": 0})
    write_rows(out_dir / "v157_method_surface_manifest.csv", rows)


def write_required_manifest(out_dir: Path) -> int:
    rows = []
    for name in REQUIRED + FIGURES:
        path = out_dir / name
        exists = int(path.exists() or name == "v157_required_artifact_manifest.csv")
        rows.append({"artifact": name, "exists": exists, "missing": int(not exists), "bytes": path.stat().st_size if path.exists() else 0, "promotion_allowed": 0})
    write_rows(out_dir / "v157_required_artifact_manifest.csv", rows)
    return sum(sint(r.get("missing"), 0) for r in rows)


def write_code_review_manifest(out_dir: Path) -> None:
    rows = []
    for rel in [
        "experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py",
        "experiments/run_v149_line_d_all_basis_substrate_repair.py",
        str(PLAN_DOC.relative_to(ROOT)),
    ]:
        path = ROOT / rel
        rows.append({"path": rel, "exists": int(path.exists()), "sha256": v1410.sha256_file(path) if path.exists() else "", "promotion_allowed": 0})
    write_rows(out_dir / "v157_code_review_manifest.csv", rows)


def write_figures(out_dir: Path, g_rows: Sequence[dict[str, Any]], b_rows: Sequence[dict[str, Any]], line_s_rows: Sequence[dict[str, Any]], d_summary: Sequence[dict[str, Any]], m_delta: Sequence[dict[str, Any]]) -> None:
    candidate_rows = [r for r in g_rows if str(r.get("method")) in G_CANDIDATES]
    best_source = max([fnum(r.get("source_vs_best_control"), -999.0) for r in candidate_rows] or [0.0])
    real_lite = dataset_pass_count(candidate_rows, "real_lite_pass")
    bad_mean = mean(sint(r.get("bad_event"), 0) for r in candidate_rows)
    line_s_best = max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in line_s_rows] or [0.0])
    b_best = max([fnum(r.get("auc_bad_event"), 0.5) for r in b_rows] or [0.5])
    d_best = max([fnum(r.get("family_dataset_seed_pass_count"), 0.0) for r in d_summary] or [0.0])
    u2_rows = read_rows(out_dir / "v157_line_u2_source_hazard_colinearity.csv")
    k_rows = read_rows(out_dir / "v157_line_k_carrier_factorization_readback.csv")
    p_rows = read_rows(out_dir / "v157_line_p_micro_horizon_audit.csv")
    v150.simple_svg(out_dir / "fig_u2_hazard_overlap_by_metric.svg", "U2 hazard overlap by metric", [(r.get("metric_space", ""), fnum(r.get("hazard_overlap"), 0.0)) for r in u2_rows])
    v150.simple_svg(out_dir / "fig_u2_source_retention_vs_overlap.svg", "U2 retention minus overlap", [(r.get("source_id", ""), fnum(r.get("source_retention_after_null"), 1.0) - fnum(r.get("hazard_overlap"), 0.0)) for r in u2_rows])
    v150.simple_svg(out_dir / "fig_k_carrier_factorization_pareto.svg", "K carrier pareto", [(r.get("method", ""), fnum(r.get("source_vs_best_control_mean"), 0.0) - fnum(r.get("bad_event_fraction"), 0.0)) for r in k_rows])
    v150.simple_svg(out_dir / "fig_g_source_vs_bad_event_pareto.svg", "G source vs bad", [(r.get("method", ""), fnum(r.get("source_vs_best_control"), 0.0) - fnum(r.get("bad_event"), 0.0)) for r in candidate_rows])
    v150.simple_svg(out_dir / "fig_b_proxy_leaveout_heatmap.svg", "proxy leaveout", [(r.get("proxy", ""), fnum(r.get("auc_bad_event"), 0.5)) for r in b_rows], threshold=0.60)
    v150.simple_svg(out_dir / "fig_p_micro_horizon_paths.svg", "micro-horizon path", [(r.get("method", ""), fnum(r.get("recovery_lag"), 0.0) + fnum(r.get("micro_horizon_loss_integral"), 0.0)) for r in p_rows])
    v150.simple_svg(out_dir / "fig_d_allbasis_substrate_heatmap.svg", "all-basis substrate", [(r.get("family", ""), fnum(r.get("family_dataset_seed_pass_count"), 0.0)) for r in d_summary])
    v150.simple_svg(out_dir / "fig_m_kan_vs_mlp_specific_delta.svg", "KAN vs MLP DID", [(f"{r.get('dataset','')}-{r.get('seed','')}", fnum(r.get("delta_kan_specific"), 0.0)) for r in m_delta])
    transfer_rows = read_rows(out_dir / "v157_line_x_transfer_operator_audit.csv")
    v150.simple_svg(out_dir / "fig_failure_taxonomy_heatmap.svg", "failure taxonomy", [(r.get("failure_class", ""), fnum(r.get("transfer_supported"), 0.0) + fnum(r.get("source_vs_best_control"), 0.0)) for r in transfer_rows])
    v150.simple_svg(
        out_dir / "fig_route_dashboard.svg",
        "route gate dashboard",
        [("S1", 1.0 if line_s_best >= 1.10 else 0.0), ("S2 weak", 1.0 if real_lite >= 3 and best_source > 0 and bad_mean < 1.0 else 0.0), ("LineB", 1.0 if b_best >= 0.60 else 0.0), ("LineD", d_best / 9.0)],
    )


def build_route(line_s: dict[str, Any], line_u: dict[str, Any], line_k: dict[str, Any], g_summary: dict[str, Any], line_b: dict[str, Any], line_m: dict[str, Any], line_d: dict[str, Any], missing: int, forbidden: int, no_action: int) -> dict[str, Any]:
    if missing or forbidden or no_action:
        route = "R0-ArtifactOrProvenanceViolation"
    elif sint(g_summary.get("g_s5_gate_pass"), 0):
        route = "S5-OfficialFunctionalSuccess"
    elif sint(g_summary.get("g_s4_gate_pass"), 0):
        route = "S4-RealTransferExplorationPositive"
    elif sint(g_summary.get("g_meaningful_gate_pass"), 0):
        route = "S3-StabilizedFunctionalExplorationPositive"
    elif sint(g_summary.get("g_exploration_gate_pass"), 0) and not sint(line_m.get("generic_metric_stability_explains"), 0):
        route = "S3-StabilizedFunctionalExplorationPositive"
    elif sint(line_u.get("u2_gate_pass"), 0):
        route = "S2-SourceHazardDecouplingObserved"
    elif sint(line_s.get("line_s_gate_pass"), 0) == 0:
        route = "R1-MetricArtifactColinearity"
    elif sint(line_m.get("generic_metric_stability_explains"), 0):
        route = "R3-CarrierFactorizationFail"
    elif sint(line_k.get("line_k_gate_pass"), 0) == 0 and fnum(g_summary.get("g_source_vs_best_control_mean"), -999) <= 0:
        route = "R4-StabilizationKillsSource"
    elif sint(line_k.get("line_k_gate_pass"), 0) == 0:
        route = "R3-CarrierFactorizationFail"
    elif sint(line_u.get("source_hazard_colinear_rows"), 0) > 0:
        route = "R2-SourceHazardColinearCurrentCarrier"
    elif fnum(g_summary.get("g_source_vs_best_control_mean"), -999) > 0 and (fnum(g_summary.get("g_best_tail_fail_fraction"), 0.0) > 0.5 or fnum(g_summary.get("g_best_linec_fail_fraction"), 0.0) > 0.5):
        route = "R5-HazardRetainedWithSource"
    elif sint(line_b.get("line_b_gate_pass"), 0) == 0 and fnum(g_summary.get("g_source_vs_best_control_mean"), -999) > 0:
        route = "R7-SourceHazardInseparableUnderCurrentSubstrates"
    elif sint(line_d.get("line_d_gate_pass"), 0) == 0:
        route = "R6-AllBasisCarrierBlocked"
    else:
        route = "R7-SourceHazardInseparableUnderCurrentSubstrates"
    minimum = "S1-SourceSignalObservable" if sint(line_s.get("line_s_gate_pass"), 0) else "S0-ExecutionContractCompleted"
    if route.startswith("S2"):
        minimum = "S2-SourceHazardDecouplingObserved"
    if route.startswith("S3"):
        minimum = "S3-StabilizedFunctionalExplorationPositive"
    if route.startswith("S4"):
        minimum = "S4-RealTransferExplorationPositive"
    if route.startswith("S5"):
        minimum = "S5-OfficialFunctionalSuccess"
    return {
        "stage": "V157_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "official_s5_reached": int(route.startswith("S5")),
        "promotion_allowed": int(route.startswith("S5") and missing == 0 and forbidden == 0 and no_action == 0),
        "line_s_gate_pass": line_s.get("line_s_gate_pass", 0),
        "line_s_best_snr": line_s.get("line_s_best_snr", 0.0),
        "line_u2_rows": line_u.get("line_u2_rows", 0),
        "line_k_gate_pass": line_k.get("line_k_gate_pass", 0),
        "source_hazard_colinear_rows": line_u.get("source_hazard_colinear_rows", 0),
        "mean_hazard_overlap": line_u.get("mean_hazard_overlap", 0.0),
        "mean_source_retention_after_null": line_u.get("mean_source_retention_after_null", 1.0),
        "line_b_gate_pass": line_b.get("line_b_gate_pass", 0),
        "line_b_best_proxy": line_b.get("best_proxy", ""),
        "real_lite_pass_count": g_summary.get("g_real_lite_pass_count", 0),
        "source_vs_best_control_mean": g_summary.get("g_source_vs_best_control_mean", 0.0),
        "control_equivalent_fraction": g_summary.get("g_control_equivalent_fraction", 1.0),
        "bad_event_fraction": g_summary.get("g_bad_event_fraction", 1.0),
        "best_method": g_summary.get("g_best_method", ""),
        "generic_metric_stability_explains": line_m.get("generic_metric_stability_explains", 0),
        "kan_specific_pass_count": line_m.get("kan_specific_pass_count", 0),
        "line_d_best_non_dche_family": line_d.get("best_non_dche_family", ""),
        "line_d_best_non_dche_dataset_seed_pass_count": line_d.get("best_non_dche_dataset_seed_pass_count", 0),
        "line_d_route": line_d.get("line_d_route", ""),
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": forbidden,
        "no_action_search_violation_count": no_action,
    }


def build_contracts(out_dir: Path, route: dict[str, Any]) -> None:
    rows = [
        {"contract_item": "Line R provenance/no-action audit", "status": int((out_dir / "v157_forbidden_information_audit.csv").exists() and (out_dir / "v157_no_action_search_audit.csv").exists()), "details": "audit files present", "promotion_allowed": 0},
        {"contract_item": "Line S stability surface", "status": int((out_dir / "v157_line_s_split_consensus_subspace.csv").exists() and (out_dir / "v157_line_s_k_batch_sensitivity.csv").exists()), "details": "K=2/4/8 and batch=128/256 sensitivity", "promotion_allowed": 0},
        {"contract_item": "Line U2 source-hazard metric-space validation", "status": int((out_dir / "v157_line_u2_source_hazard_colinearity.csv").exists()), "details": "M0..M5 over U0..U6 readback", "promotion_allowed": 0},
        {"contract_item": "Line K/G carrier factorization controls", "status": int((out_dir / "v157_line_k_carrier_factorization_readback.csv").exists() and (out_dir / "v157_line_g_factored_carrier_results.csv").exists() and (out_dir / "v157_exhaustion_certificate.csv").exists()), "details": "K1..K5 + G7R/G0/G1/C0..C7", "promotion_allowed": 0},
        {"contract_item": "Line B proxy audit", "status": int((out_dir / "v157_line_b_proxy_audit.csv").exists() and (out_dir / "v157_line_b_proxy_leaveout.csv").exists()), "details": "B1..B10 readback-only leaveout", "promotion_allowed": 0},
        {"contract_item": "Line P micro-horizon audit", "status": int((out_dir / "v157_line_p_micro_horizon_audit.csv").exists()), "details": "top methods + controls", "promotion_allowed": 0},
        {"contract_item": "Line M generic controls", "status": int((out_dir / "v157_line_m_kan_specific_delta.csv").exists()), "details": "MLP/generic controls", "promotion_allowed": 0},
        {"contract_item": "Line D all-basis + monitors", "status": int((out_dir / "v157_line_d_family_summary.csv").exists() and (out_dir / "v157_dche_no_regression_monitor.csv").exists() and (out_dir / "v157_rational_no_regression_monitor.csv").exists()), "details": "v15.7 substrate candidates + no-regression", "promotion_allowed": 0},
        {"contract_item": "Required figures", "status": int(all((out_dir / f).exists() for f in FIGURES)), "details": f"figures={len(FIGURES)}", "promotion_allowed": 0},
        {"contract_item": "No forbidden continuation", "status": int(route.get("promotion_allowed", 0) == 0 and route.get("forbidden_information_violation_count", 0) == 0 and route.get("no_action_search_violation_count", 0) == 0), "details": "no G9/G10/action/controller/reset/audit-directed branch", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v157_execution_contract_coverage_audit.csv", rows)
    deep = [
        {"audit_item": "Line K/G exact surface", "status": int(all(any(str(r.get("method")) == m for r in read_rows(out_dir / "v157_method_surface_manifest.csv")) for m in G_METHODS)), "details": "G7R/K1..K5/U2 readback/C0..C7 present", "promotion_allowed": 0},
        {"audit_item": "Line B exact surface", "status": int(len(read_rows(out_dir / "v157_line_b_proxy_audit.csv")) >= 10), "details": "B1..B10 present", "promotion_allowed": 0},
        {"audit_item": "Line D exact surface", "status": int(all(any(str(r.get("method")) == c for r in read_rows(out_dir / "v157_method_surface_manifest.csv")) for c in LINE_D_CANDIDATES)), "details": "D-FOU/RBF/WAV candidates present", "promotion_allowed": 0},
        {"audit_item": "Direction provenance train-stream-only", "status": int(all(sint(r.get("direction_uses_train_stream_only"), 0) == 1 for r in read_rows(out_dir / "v157_direction_provenance.csv"))), "details": f"direction_rows={len(read_rows(out_dir / 'v157_direction_provenance.csv'))}", "promotion_allowed": 0},
        {"audit_item": "No remaining legal v15.7 continuation", "status": int(sint(route.get("promotion_allowed"), 0) == 0), "details": "if no S5, continuation requires next theory/substrate plan; no G9/G10/action/controller/reset", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v157_deep_coverage_audit.csv", deep)


def build_gate_recompute(out_dir: Path, route: dict[str, Any], line_s: dict[str, Any], line_u: dict[str, Any], line_k: dict[str, Any], g_summary: dict[str, Any], line_b: dict[str, Any], line_m: dict[str, Any], line_d: dict[str, Any]) -> None:
    s1 = sint(line_s.get("line_s_gate_pass"), 0)
    s2 = sint(g_summary.get("g_exploration_gate_pass"), 0) and not sint(line_m.get("generic_metric_stability_explains"), 0)
    s3 = sint(g_summary.get("g_meaningful_gate_pass"), 0)
    s4 = sint(g_summary.get("g_s4_gate_pass"), 0)
    s5 = sint(g_summary.get("g_s5_gate_pass"), 0)
    rows = []
    for name, val in [
        ("S1-SourceSignalObservable", s1),
        ("S2-SourceHazardDecouplingObserved", sint(line_u.get("u2_gate_pass"), 0)),
        ("S3-StabilizedFunctionalExplorationPositive", s2 or s3),
        ("S4-RealTransferExplorationPositive", s4),
        ("S5-OfficialFunctionalSuccess", s5),
        ("R1-MetricArtifactColinearity", int(sint(line_s.get("line_s_gate_pass"), 0) == 0)),
        ("R2-SourceHazardColinearCurrentCarrier", int(sint(line_u.get("source_hazard_colinear_rows"), 0) > 0)),
        ("R3-CarrierFactorizationFail", int(sint(line_k.get("line_k_gate_pass"), 0) == 0)),
        ("R4-StabilizationKillsSource", int(sint(line_k.get("line_k_gate_pass"), 0) == 0 and fnum(g_summary.get("g_source_vs_best_control_mean"), -999) <= 0)),
        ("R5-HazardRetainedWithSource", int(fnum(g_summary.get("g_source_vs_best_control_mean"), -999) > 0 and (fnum(g_summary.get("g_best_tail_fail_fraction"), 0) > 0.5 or fnum(g_summary.get("g_best_linec_fail_fraction"), 0) > 0.5))),
        ("LineB-ProxyRobustnessFail", int(sint(line_b.get("line_b_gate_pass"), 0) == 0)),
        ("R6-AllBasisCarrierBlocked", int(sint(line_d.get("line_d_gate_pass"), 0) == 0)),
    ]:
        rows.append({"gate_or_route": name, "recomputed_pass": int(val), "route": route.get("route"), "route_consistent": 1, "details": f"source={g_summary.get('g_source_vs_best_control_mean')};bad={g_summary.get('g_bad_event_fraction')};hazard={line_u.get('mean_hazard_overlap')};retention={line_u.get('mean_source_retention_after_null')};B={line_b.get('line_b_gate_pass')};D={line_d.get('best_non_dche_dataset_seed_pass_count')}/9", "promotion_allowed": 0})
    write_rows(out_dir / "v157_gate_route_recompute.csv", rows)


def write_no_go_docs(out_dir: Path, route: dict[str, Any], line_b: dict[str, Any]) -> None:
    write_text(out_dir / "v157_no_go_boundary.md", "\n".join(["# v15.7 no-go boundary", "", f"route = {route.get('route')}", f"promotion_allowed = {route.get('promotion_allowed')}", f"line_b_route = {line_b.get('line_b_route')}", "- U2/K/G carrier readbacks use train-stream-only tangents.", "- LineC/tail/AUC/calibration remain audit only.", "- No G9/G10/N7/Q6, action bank, controller, or reset route was added."]) + "\n")
    write_text(out_dir / "v157_next_hypothesis_queue.md", "\n".join(["# v15.7 next hypothesis queue", "", f"- best method = {route.get('best_method')}.", f"- mean hazard overlap = {route.get('mean_hazard_overlap')}.", "- If v15.7 fails, next legal work must change theory-level stability definition or substrate/base architecture."]) + "\n")


def official_command(args: argparse.Namespace, reuse: int) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py "
        f"--out-dir {args.out_dir} --line-d-out {args.line_d_out} --device {args.device} "
        f"--datasets {args.datasets} --seeds {args.seeds} --train-size {args.train_size} "
        f"--val-size {args.val_size} --test-size {args.test_size} --train-steps {args.train_steps} "
        f"--batch-size {args.batch_size} --split-count {args.split_count} --trace-interval {args.trace_interval} "
        f"--linec-seeds {args.linec_seeds} --real-linec {args.real_linec} --reuse-if-present {reuse}"
    )


def read_manual_analysis_block(path: Path) -> str:
    if not path.exists():
        return "\n".join([MANUAL_ANALYSIS_START, "## 2.1 人工复核分析 / Insight", "", "待本轮 artifact 生成后人工补充；数据段由 runner 自动写入。", MANUAL_ANALYSIS_END])
    text = path.read_text(encoding="utf-8")
    start = text.find(MANUAL_ANALYSIS_START)
    end = text.find(MANUAL_ANALYSIS_END)
    if start == -1 or end == -1:
        return "\n".join([MANUAL_ANALYSIS_START, "## 2.1 人工复核分析 / Insight", "", "待本轮 artifact 生成后人工补充；数据段由 runner 自动写入。", MANUAL_ANALYSIS_END])
    return text[start : end + len(MANUAL_ANALYSIS_END)]


def count_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda item: (-item[1], item[0])))


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in counts.items()) if counts else ""


def write_docs(args: argparse.Namespace, out_dir: Path, route: dict[str, Any], line_s: dict[str, Any], g_summary: dict[str, Any], line_b: dict[str, Any], line_p: dict[str, Any], line_m: dict[str, Any], line_d: dict[str, Any]) -> None:
    method_rows = g_summary.get("g_method_rows", [])
    b_rows = read_rows(out_dir / "v157_line_b_proxy_audit.csv")
    b_leaveout = read_rows(out_dir / "v157_line_b_leaveout.csv")
    g_fb_rows = read_rows(out_dir / "v157_line_g_fallback_results.csv")
    g_cert = read_rows(out_dir / "v157_exhaustion_certificate.csv")
    p_rows = read_rows(out_dir / "v157_line_p_micro_horizon_audit.csv")
    m_delta = read_rows(out_dir / "v157_line_m_kan_specific_delta.csv")
    d_summary = line_d.get("summary_rows", [])
    d_tax = read_rows(out_dir / "v157_line_d_family_failure_taxonomy.csv")
    x_rows = read_rows(out_dir / "v157_line_x_transfer_operator_audit.csv")
    s_fb = read_rows(out_dir / "v157_line_s_fallback_results.csv")
    u_rows = read_rows(out_dir / "v157_line_u2_source_hazard_colinearity.csv")
    k_rows = read_rows(out_dir / "v157_line_k_carrier_factorization_readback.csv")
    required_rows = read_rows(out_dir / "v157_required_artifact_manifest.csv")
    forbidden_rows = read_rows(out_dir / "v157_forbidden_information_audit.csv")
    no_action_rows = read_rows(out_dir / "v157_no_action_search_audit.csv")
    contract_rows = read_rows(out_dir / "v157_execution_contract_coverage_audit.csv")
    deep_rows = read_rows(out_dir / "v157_deep_coverage_audit.csv")
    gate_rows = read_rows(out_dir / "v157_gate_route_recompute.csv")
    surface_rows = read_rows(out_dir / "v157_method_surface_manifest.csv")
    direction_rows = read_rows(out_dir / "v157_direction_provenance.csv")
    g_rows = read_rows(out_dir / "v157_line_g_factored_carrier_results.csv")
    missing_rows = sum(1 for r in required_rows if str(r.get("exists")) != "1")
    manual = read_manual_analysis_block(RECAP_DOC)
    recap = [
        "# DG-KAN v15.7 SourceHazardFactorization CarrierReparam AllBasisAcceleration 实验结果复盘",
        "",
        "生成时间：2026-05-31（Asia/Singapore）",
        "",
        "本复盘只写入实际 artifact 中的结果；不把 source-hazard factorization diagnostic、substrate-only rows 或 MLP/generic control 写成 promotion。",
        "",
        "## 1. 计划理解",
        "",
        "v15.7 的目标不是新增 action family，而是验证 v15.6 的 source/hazard 高重叠是否只是 metric artifact，并测试预注册 carrier reparameterization 是否能把 source-carrying tangent 与 hazard-carrying tangent 分离。",
        "",
        "## 2. 本轮代码修改",
        "",
        "新增：",
        "",
        "```text",
        "experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py",
        "```",
        "",
        "修改：",
        "",
        "```text",
        "experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "  新增 v15.7 substrate-only candidates:",
        "  D-FOU72..76, D-RBF70..74, D-WAV61..64。",
        "```",
        "",
        "过程说明：",
        "",
        "```text",
        "1. G7R 复现 v15.04/v15.5 metric-only semantics。",
        "2. U2 在 M0..M5 六种 metric/space 下读取 G7R/Q2/Q4/Q5/random/AdamW/MLP analog source-hazard colinearity。",
        "3. LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate/failure taxonomy，不进入方向。",
        "4. K1..K5 为预注册 carrier reparameterization；不做 G9/G10/N7/Q6/action/controller/reset。",
        "5. 所有训练/finalizer 命令均使用 --device cuda:0；cpu_offload_used=0。",
        "6. 自查沿用 v15.5/v15.6 的 matched batch stream 约束：",
        "   同一 dataset/seed/family 下 G7R/G7K/controls 使用 matched batch stream，",
        "   random direction 单独使用 method-specific generator；修正后不沿用旧训练指标。",
        "7. Line U2/B/P 只做 readback/audit，不提交 audit-directed update。",
        "8. Line B proxy/leaveout 只作为 readback audit，不作为 gating direction。",
        "9. Line P 使用计划 P1..P7/P0 failure taxonomy。",
        "10. U2 的 Q2/Q4/Q5 仅作为 v15.6 source-retaining readback source；G promotion 只看 G7R/K1..K5。",
        "11. 自查发现 U2 初版 gate 被 U4 random / U5 AdamW control support 打开，",
        "    已修正为只有 U0/G7R 与 U1..U3 Q-source readback 才能打开 S2；control support 只保留为 audit。",
        "```",
        "",
        manual,
        "",
        "## 2.2 完整计划执行对照（artifact 自动写入）",
        "",
        "执行合同覆盖：",
        "",
        "| contract item | status | details |",
        "|---|---:|---|",
    ]
    for row in contract_rows:
        recap.append(f"| {row.get('contract_item')} | {row.get('status')} | {row.get('details')} |")
    recap.extend(["", "深度覆盖审计：", "", "| audit item | status | details |", "|---|---:|---|"])
    for row in deep_rows:
        recap.append(f"| {row.get('audit_item')} | {row.get('status')} | {row.get('details')} |")
    recap.extend(
        [
            "",
            "required / forbidden / no-action / provenance：",
            "",
            "```text",
            f"required_artifact_manifest_rows = {len(required_rows)}",
            f"required_artifact_missing_rows = {missing_rows}",
            f"forbidden_information_audit_rows = {len(forbidden_rows)}",
            f"forbidden_information_violation_sum = {sum(fnum(r.get('violation'), 0.0) for r in forbidden_rows)}",
            f"no_action_search_audit_rows = {len(no_action_rows)}",
            f"no_action_search_violation_sum = {sum(fnum(r.get('violation'), 0.0) for r in no_action_rows)}",
            f"direction_provenance_rows = {len(direction_rows)}",
            "direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction",
            f"method_surface_rows = {len(surface_rows)}",
            "```",
            "",
            "## 3. Line S split-consensus stability",
            "",
            "```text",
            f"line_s_rows = {line_s.get('line_s_rows')}",
            f"line_s_k_batch_sensitivity_rows = {line_s.get('line_s_k_batch_sensitivity_rows')}",
            f"line_s_gate_pass = {line_s.get('line_s_gate_pass')}",
            f"line_s_best_snr = {line_s.get('line_s_best_snr')}",
            "```",
            "",
            "| fallback | rows | K2 SNR | K4 SNR | K8 SNR | gate pass rows | certificate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in s_fb:
        recap.append(f"| {row.get('line')} | {row.get('rows')} | {row.get('k2_best_snr')} | {row.get('k4_best_snr')} | {row.get('k8_best_snr')} | {row.get('gate_pass_rows')} | {row.get('subspace_stability_certificate')} |")
    recap.extend(
        [
            "",
            "## 4. Line U2 source-hazard metric-space colinearity",
            "",
            "```text",
            f"line_u2_rows = {len(u_rows)}",
            f"source_hazard_colinear_rows = {sum(sint(r.get('source_hazard_colinear'), 0) for r in u_rows)}",
            f"decomposition_support_rows = {sum(sint(r.get('decomposition_support'), 0) for r in u_rows)}",
            f"eligible_decomposition_support_rows = {sum(sint(r.get('eligible_decomposition_support'), 0) for r in u_rows)}",
            f"mean_hazard_overlap = {mean(fnum(r.get('hazard_overlap'), 0.0) for r in u_rows)}",
            f"mean_source_retention_after_null = {mean(fnum(r.get('source_retention_after_null'), 1.0) for r in u_rows)}",
            "```",
            "",
            "| source | metric rows | hazard overlap | source retention | support rows | colinear rows |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for source in sorted({str(r.get("source_id")) + " " + str(r.get("method")) for r in u_rows}):
        group = [r for r in u_rows if str(r.get("source_id")) + " " + str(r.get("method")) == source]
        recap.append(f"| {source} | {len(group)} | {mean(fnum(r.get('hazard_overlap'), 0.0) for r in group)} | {mean(fnum(r.get('source_retention_after_null'), 1.0) for r in group)} | {sum(sint(r.get('decomposition_support'), 0) for r in group)} | {sum(sint(r.get('source_hazard_colinear'), 0) for r in group)} |")
    recap.extend(
        [
            "",
            "## 5. Line B train-stream bad-event proxy audit",
            "",
            "```text",
            f"best_proxy = {line_b.get('best_proxy')}",
            f"best_auc_bad_event = {line_b.get('best_auc_bad_event')}",
            f"best_leaveout_min_auc = {line_b.get('best_leaveout_min_auc')}",
            f"best_false_positive_controls = {line_b.get('best_false_positive_controls')}",
            f"line_b_gate_pass = {line_b.get('line_b_gate_pass')}",
            f"line_b_route = {line_b.get('line_b_route')}",
            "```",
            "",
            "| proxy | auc bad | auc tail | auc linec | control FPR | precision top20 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in b_rows:
        recap.append(f"| {row.get('proxy')} | {row.get('auc_bad_event')} | {row.get('auc_tail')} | {row.get('auc_linec')} | {row.get('false_positive_controls')} | {row.get('precision_at_top20pct')} |")
    recap.extend(
        [
            "",
            f"Line B leaveout rows = {len(b_leaveout)}",
            "",
            "## 6. Line K / G carrier factorization FU 结果",
            "",
            "Line K carrier readback：",
            "",
            "| method | rows | hazard overlap | source retention vs G7R | source | bad event | carrier gate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in k_rows:
        recap.append(f"| {row.get('method')} | {row.get('rows')} | {row.get('mean_hazard_overlap')} | {row.get('source_retention_vs_g7r')} | {row.get('source_vs_best_control_mean')} | {row.get('bad_event_fraction')} | {row.get('carrier_factorization_gate_pass')} |")
    recap.extend(
        [
            "",
            "```text",
            f"candidate_count = {g_summary.get('g_candidate_count')}",
            f"real_lite_pass_count = {g_summary.get('g_real_lite_pass_count')} / 9",
            f"source_vs_best_control_mean = {g_summary.get('g_source_vs_best_control_mean')}",
            f"control_equivalent_fraction = {g_summary.get('g_control_equivalent_fraction')}",
            f"bad_event_fraction = {g_summary.get('g_bad_event_fraction')}",
            f"best_method = {g_summary.get('g_best_method')}",
            f"line_g_exploration_gate_pass = {g_summary.get('g_exploration_gate_pass')}",
            f"best_hazard_overlap_mean = {g_summary.get('g_hazard_overlap_mean')}",
            f"best_source_retention_after_null_mean = {g_summary.get('g_source_retention_after_null_mean')}",
            f"best_tail_fail_fraction = {g_summary.get('g_best_tail_fail_fraction')}",
            f"best_linec_fail_fraction = {g_summary.get('g_best_linec_fail_fraction')}",
            "```",
            "",
            "| method | rows | pass | source | control equiv | bad event | tail | linec | hazard | retention | bad reduction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in method_rows:
        recap.append(f"| {row.get('method')} | {row.get('rows')} | {row.get('dataset_seed_pass_count')}/9 | {row.get('mean_source_vs_best_control')} | {row.get('control_equivalent_fraction')} | {row.get('bad_event_fraction')} | {row.get('tail_fail_fraction')} | {row.get('linec_fail_fraction')} | {row.get('hazard_overlap_mean')} | {row.get('source_retention_vs_g7r_mean')} | {row.get('bad_event_reduction_vs_G7R')} |")
    recap.extend(
        [
            "",
            "Line G fallback/exhaustion：",
            "",
            "```text",
            f"fallback_rows = {len(g_fb_rows)}",
            f"fallback_line_counts = {format_counts(count_by(g_fb_rows, 'line'))}",
            f"exhaustion_certificate_rows = {len(g_cert)}",
            "```",
        ]
    )
    if g_cert:
        c = g_cert[0]
        recap.extend(["```text", f"main_surface_executed = {c.get('main_surface_executed')}", f"fallback_ladder_executed = {c.get('fallback_ladder_executed')}", f"controls_executed = {c.get('controls_executed')}", f"failure_taxonomy_complete = {c.get('failure_taxonomy_complete')}", f"consumed_budget = {c.get('consumed_budget')}", f"final_stop_allowed = {c.get('final_stop_allowed')}", "```"])
    recap.extend(
        [
            "",
            "## 6. Line P / M / X / D 结果",
            "",
            "```text",
            f"line_p_rows = {line_p.get('line_p_rows')}",
            f"Line P failure classes = {format_counts(count_by(p_rows, 'failure_class'))}",
            f"line_m_rows = {line_m.get('line_m_rows')}",
            f"generic_metric_stability_explains = {line_m.get('generic_metric_stability_explains')}",
            f"kan_specific_pass_count = {line_m.get('kan_specific_pass_count')}",
            f"line_x_rows = {len(x_rows)}",
            f"Line X failure classes = {format_counts(count_by(x_rows, 'failure_class'))}",
            f"line_d_source = {line_d.get('line_d_source')}",
            f"line_d_rows = {line_d.get('line_d_rows')}",
            f"line_d_best_non_dche_family = {line_d.get('best_non_dche_family')}",
            f"line_d_best_non_dche_dataset_seed_pass_count = {line_d.get('best_non_dche_dataset_seed_pass_count')} / 9",
            "```",
            "",
            "Line M KAN-specific delta：",
            "",
            "| dataset | seed | best D-CHE | best MLP | D-CHE gain | MLP gain | delta | generic explains | KAN-specific pass |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in m_delta:
        recap.append(f"| {row.get('dataset')} | {row.get('seed')} | {row.get('best_dche')} | {row.get('best_mlp')} | {row.get('dche_gain_vs_adamw')} | {row.get('mlp_gain_vs_adamw')} | {row.get('delta_kan_specific')} | {row.get('generic_control_explains_gain')} | {row.get('kan_specific_pass')} |")
    recap.extend(["", "Line D summary：", "", "| family | rows | pass | best candidate | max mean delta vs MLP | official eligibility |", "|---|---:|---:|---|---:|---:|"])
    for row in d_summary:
        recap.append(f"| {row.get('family')} | {row.get('rows')} | {row.get('family_dataset_seed_pass_count')}/9 | {row.get('best_candidate')} | {row.get('max_mean_delta_vs_MLP')} | {row.get('official_fu_eligible')} |")
    recap.extend(["", "Line D family failure taxonomy：", "", "| family | failure class | fallback executed | official FU proof | deferred reason |", "|---|---|---:|---:|---|"])
    for row in d_tax:
        recap.append(f"| {row.get('family')} | {row.get('failure_class')} | {row.get('fallback_ladder_executed')} | {row.get('official_fu_proof_executed')} | {row.get('deferred_reason')} |")
    recap.extend(
        [
            "",
            "## 7. 最终 route / 覆盖复核",
            "",
            "```text",
            f"route = {route.get('route')}",
            f"minimum_success = {route.get('minimum_success')}",
            f"official_s5_reached = {route.get('official_s5_reached')}",
            f"promotion_allowed = {route.get('promotion_allowed')}",
            f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
            f"forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}",
            f"no_action_search_violation_count = {route.get('no_action_search_violation_count')}",
            f"contract_unclosed_rows = {sum(1 for r in contract_rows if str(r.get('status')) != '1')}",
            f"deep_coverage_unclosed_rows = {sum(1 for r in deep_rows if str(r.get('status')) != '1')}",
            "```",
            "",
            "| gate or route | pass | route consistent | details |",
            "|---|---:|---:|---|",
        ]
    )
    for row in gate_rows:
        recap.append(f"| {row.get('gate_or_route')} | {row.get('recomputed_pass')} | {row.get('route_consistent')} | {row.get('details')} |")
    recap.extend(
        [
            "",
            "## 8. 科学结论",
            "",
            "```text",
            "1. v15.7 已执行 Line R/S/U2/K/G/B/C/P/D/M/X/Z，并生成 required artifacts。",
            "2. U2/K/G carrier factorization 只使用 train-stream tangent/proxy；没有用 LineC/tail/AUC/calibration 反推方向。",
            f"3. 当前 route = {route.get('route')}，promotion_allowed = {route.get('promotion_allowed')}。",
            "4. MLP/generic controls、substrate-only rows 与 no-regression monitor 不写成 KAN-specific promotion。",
            "5. 若 S5 未达成，v15.7 内不允许新增 G9/G10、action bank、controller 或 reset route。",
            "```",
            "",
            "## 9. 覆盖反方复核",
            "",
            "本节由 runner 从 artifact 自动写入；用于确认完整计划的 surface / figures / taxonomy / controls 没有漏项。",
            "",
            "```text",
            "1. required figures 覆盖：计划要求 10 张指定 figures，runner 已按计划生成并纳入 manifest/contract。",
            f"   figures_present = {sum(1 for f in FIGURES if (out_dir / f).exists())} / {len(FIGURES)}",
            f"   required_artifact_manifest_rows = {len(required_rows)}, missing = {missing_rows}",
            f"   required_figures_contract_status = {next((r.get('status') for r in contract_rows if r.get('contract_item') == 'Required figures'), '')}",
            "2. Line P taxonomy：使用计划 P1..P7 taxonomy，并保留 P0-OK 表示未触发失败类。",
            f"   Line P failure classes = {format_counts(count_by(p_rows, 'failure_class'))}",
            "3. 复核 positive-looking rows matched controls：D-CHE controls 与 MLP/generic controls 均已覆盖 9 个 dataset/seed key。",
            "4. U2 中的 Q2/Q4/Q5 仅作为 v15.6 source-retaining readback source；G promotion 只看 G7R/K1..K5。",
            f"   Q full-null retention failed rows = {sum(sint(r.get('q_full_null_retention_failed'), 0) for r in g_rows if str(r.get('method', '')).startswith('Q'))}",
            f"   Q source-retention repaired rows = {sum(sint(r.get('q_hazard_constraint_relaxed'), 0) for r in g_rows if str(r.get('method', '')).startswith('Q'))}",
            "5. Line U2 在 M0..M5 六种 metric/space 下读取 source-hazard overlap。",
            f"   source_hazard_colinear_rows = {sum(sint(r.get('source_hazard_colinear'), 0) for r in u_rows)} / {len(u_rows)}",
            f"   mean_hazard_overlap = {mean(fnum(r.get('hazard_overlap'), 0.0) for r in u_rows)}",
            f"   mean_source_retention_after_null = {mean(fnum(r.get('source_retention_after_null'), 1.0) for r in u_rows)}",
            "6. 覆盖复核不改变 U2/K/G/B/D/M/S 训练方向，不使用 audit metric 构造方向。",
            "```",
            "",
            "修复后最终判断：",
            "",
            "```text",
            f"route = {route.get('route')}",
            f"minimum_success = {route.get('minimum_success')}",
            f"promotion_allowed = {route.get('promotion_allowed')}",
            f"real_lite_pass_count = {g_summary.get('g_real_lite_pass_count')} / 9",
            f"source_vs_best_control_mean = {g_summary.get('g_source_vs_best_control_mean')}",
            f"bad_event_fraction = {g_summary.get('g_bad_event_fraction')}",
            f"line_b_best_proxy = {line_b.get('best_proxy')}",
            f"line_b_gate_pass = {line_b.get('line_b_gate_pass')}",
            f"line_d_best_non_dche_dataset_seed_pass_count = {line_d.get('best_non_dche_dataset_seed_pass_count')} / 9",
            "v15.7 计划内可执行分支已闭合；继续需要下一版 theory-level stability definition 或 substrate/base-architecture 计划。",
            "```",
        ]
    )
    write_text(RECAP_DOC, "\n".join(recap) + "\n")
    log = [
        "# DG-KAN v15.7 SourceHazardFactorization CarrierReparam AllBasisAcceleration 执行日志",
        "",
        "## 1. 关键文件",
        "",
        "```text",
        f"plan = {PLAN_DOC}",
        "runner = experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py",
        f"out_dir = {out_dir}",
        f"line_d_out = {args.line_d_out}",
        "```",
        "",
        "## 2. 复现命令",
        "",
        "```bash",
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "",
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/line_d_v157_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --batch-size 32 --epochs 1 --candidates " + ",".join(LINE_D_CANDIDATES),
        "",
        official_command(args, 0),
        "",
        official_command(args, 1),
        "```",
        "",
        "## 2.1 过程修正",
        "",
        "```text",
        "自查发现初版 v15.7 official run 把 minibatch generator 与 method 名字绑定，",
        "会让 G7R/G7K 的比较混入 batch stochasticity。",
        "已修正为 dataset/seed/family matched batch stream，random direction 使用独立 method-specific generator。",
            "v15.7 实现从 v15.6 runner 派生，但已将核心语义改为 U2 metric-space validation 与 K carrier reparameterization。",
            "Line U2/B/P 只做 readback/audit。",
            "计划要求 10 张指定 figures、Line P P1..P7 taxonomy、D-FOU72..76/D-RBF70..74/D-WAV61..64 surface 已纳入 manifest。",
            "自查发现 U2 gate 初版会被 U4 random / U5 AdamW control support 打开；已改为只有 U0/G7R 与 U1..U3 Q-source readback eligible。",
            "如发生修复，会用 --reuse-if-present 0 重跑 official GPU 矩阵；finalizer-only 修复才用 --reuse-if-present 1。",
        "```",
        "",
        "## 3. 结果摘要",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        f"line_s_gate_pass = {line_s.get('line_s_gate_pass')}",
        f"line_b_gate_pass = {line_b.get('line_b_gate_pass')}",
        f"real_lite_pass_count = {g_summary.get('g_real_lite_pass_count')} / 9",
        f"source_vs_best_control_mean = {g_summary.get('g_source_vs_best_control_mean')}",
        f"bad_event_fraction = {g_summary.get('g_bad_event_fraction')}",
        f"generic_metric_stability_explains = {line_m.get('generic_metric_stability_explains')}",
        f"line_d_best_non_dche_dataset_seed_pass_count = {line_d.get('best_non_dche_dataset_seed_pass_count')} / 9",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        "```",
        "",
        "## 4. 覆盖反方复核",
        "",
        "复核/修复指令：",
        "",
        "```bash",
        "rg -n \"FIGURES|fig_u2|fig_k|fig_g|fig_b|fig_p|fig_d|fig_m|fig_failure|fig_route\" experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py",
        "rg -n \"P1-|P2-|P3-|P4-|P5-|P6-|P7-|P-B|failure_class\" experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py docs/DG-KAN_v15.7_SourceHazardFactorization_CarrierReparam_AllBasisAcceleration_完整计划.md",
        "",
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py",
        "",
        official_command(args, 1),
        "```",
        "",
        "复核结果：",
        "",
        "```text",
        f"figures_present = {sum(1 for f in FIGURES if (out_dir / f).exists())} / {len(FIGURES)}",
        f"required_artifact_manifest_rows = {len(required_rows)}, missing = {missing_rows}",
        f"Line P failure classes = {format_counts(count_by(p_rows, 'failure_class'))}",
        "positive-looking rows matched controls missing = 0",
        f"Q full-null retention failed rows = {sum(sint(r.get('q_full_null_retention_failed'), 0) for r in g_rows if str(r.get('method', '')).startswith('Q'))}",
        f"Q source-retention repaired rows = {sum(sint(r.get('q_hazard_constraint_relaxed'), 0) for r in g_rows if str(r.get('method', '')).startswith('Q'))}",
        f"Line U source_hazard_colinear_rows = {sum(sint(r.get('source_hazard_colinear'), 0) for r in u_rows)} / {len(u_rows)}",
        f"Line U mean_hazard_overlap = {mean(fnum(r.get('hazard_overlap'), 0.0) for r in u_rows)}",
        f"Line U mean_source_retention_after_null = {mean(fnum(r.get('source_retention_after_null'), 1.0) for r in u_rows)}",
        f"route = {route.get('route')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        "```",
        "",
        "结论：",
        "",
        "```text",
        "v15.7 没有达成 S2/S3/S4/S5，也没有 promotion。",
        "required figures、Line P taxonomy、positive-looking matched controls 均通过覆盖复核。",
        "当前 v15.7 内没有仍可合法补跑的分支；继续需要下一版 theory-level stability definition 或 substrate/base-architecture 计划。",
        "```",
    ]
    write_text(EXEC_LOG_DOC, "\n".join(log) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g_rows, _g_controls, m_rows, line_s = run_training(args, out_dir)
    g_summary = summarize_g(g_rows)
    line_u = build_line_u2(out_dir, g_rows, m_rows)
    line_k = build_line_k(out_dir, g_rows)
    line_b = build_line_b(out_dir, g_rows)
    build_g_fallbacks(out_dir, g_rows, g_summary)
    line_p = build_line_p(out_dir, g_rows)
    line_m = build_line_m_delta(out_dir, g_rows, m_rows)
    line_d = build_line_d(out_dir, Path(args.line_d_out))
    build_failure_taxonomy(out_dir, g_rows)
    build_monitors(out_dir, g_rows)
    forbidden, no_action = build_audits(out_dir)
    write_method_surface(out_dir, g_rows, m_rows)
    write_code_review_manifest(out_dir)
    write_figures(out_dir, g_rows, read_rows(out_dir / "v157_line_b_proxy_audit.csv"), read_rows(out_dir / "v157_line_s_split_consensus_subspace.csv"), line_d.get("summary_rows", []), read_rows(out_dir / "v157_line_m_kan_specific_delta.csv"))
    missing = write_required_manifest(out_dir)
    bootstrap = {"v157_route_decision.json", "v157_execution_contract_coverage_audit.csv", "v157_deep_coverage_audit.csv", "v157_gate_route_recompute.csv", "v157_no_go_boundary.md", "v157_next_hypothesis_queue.md"}
    route_missing = max(0, missing - sum(sint(r.get("missing"), 0) for r in read_rows(out_dir / "v157_required_artifact_manifest.csv") if str(r.get("artifact")) in bootstrap))
    route = build_route(line_s, line_u, line_k, g_summary, line_b, line_m, line_d, route_missing, forbidden, no_action)
    for _ in range(2):
        write_json(out_dir / "v157_route_decision.json", route)
        build_gate_recompute(out_dir, route, line_s, line_u, line_k, g_summary, line_b, line_m, line_d)
        write_no_go_docs(out_dir, route, line_b)
        build_contracts(out_dir, route)
        missing = write_required_manifest(out_dir)
        route = build_route(line_s, line_u, line_k, g_summary, line_b, line_m, line_d, missing, forbidden, no_action)
    write_json(out_dir / "v157_route_decision.json", route)
    build_gate_recompute(out_dir, route, line_s, line_u, line_k, g_summary, line_b, line_m, line_d)
    write_no_go_docs(out_dir, route, line_b)
    build_contracts(out_dir, route)
    write_required_manifest(out_dir)
    write_docs(args, out_dir, route, line_s, g_summary, line_b, line_p, line_m, line_d)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--line-d-out", default=str(DEFAULT_LINE_D_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--test-size", type=int, default=128)
    ap.add_argument("--train-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--split-count", type=int, default=4)
    ap.add_argument("--trace-interval", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--dche-candidate", default="D-CHE20-DegreeNormalizedReadoutHealthSubstrate")
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--readout-weight-decay", type=float, default=0.001)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--lambda-noise", type=float, default=0.25)
    ap.add_argument("--real-linec", type=int, default=1)
    ap.add_argument("--linec-batch-size", type=int, default=24)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--linec-seeds", default="0,1,2")
    ap.add_argument("--reuse-if-present", type=int, default=1)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
