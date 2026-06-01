"""Loss-agnostic MLP functional source primitives.

The objects here are architecture-level mechanics shared by runners: source
candidate ids, matched controls, unlabeled functional objectives, and delta
application helpers. Versioned runners should decide datasets/artifacts, not own
these primitives.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from dgkan.models.fc_purekan_primitives import MLPBaseline

SOURCE_CANDIDATES = {
    "M-F1-ActivationCovTransport": "activation covariance transport",
    "M-F2-LogitCovStabilization": "logit covariance stabilization",
    "M-F3-RandomCotangentJacobianSketch": "random-cotangent Jacobian sketch shaping",
    "M-F4-HiddenSpectralBalance": "hidden spectral balance",
    "M-F5-GeometrySafeWeightAnchor": "geometry-safe weight anchor",
    "M-F6-InputJitterConsistency": "input-jitter logit/hidden consistency",
    "M-G1-UnlabeledCloneResponseCov": "unlabeled clone-response covariance maintenance",
    "M-G2-RandomCotangentResponseStability": "random-cotangent response stability under deterministic jitter",
    "M-G3-InputPerturbationConsistencyTransport": "input perturbation response transport with covariance guard",
    "M-G4-HiddenCovarianceTransportWithControlResidual": "hidden covariance transport with control-residual shaping",
    "M-G5-ActivationSubspaceGuardThenTaskNeutral": "activation subspace guard with task-neutral logit damping",
    "M-H1-UnlabeledMicroProbeResponsePredictor": "unlabeled micro-probe response predictor",
    "M-H2-RandomCotangentLowRankResponseController": "random-cotangent low-rank response controller",
    "M-H3-ActivationSpectrumGuardWithMatchedControls": "activation spectrum guard with matched controls",
    "M-I1-CloneProbeCovarianceUpperBound": "clone-probe covariance upper-bound closure",
    "M-I2-PrecommitUnlabeledResponseStability": "precommit unlabeled response stability closure",
    "M-I3-ControlResidualizedMicroProbe": "control-residualized micro-probe closure",
    "M-I4-ArchitectureNeutralSNRTransport": "architecture-neutral SNR transport closure",
    "M-J1-CloneProbeCovarianceTransportV2": "clone-probe covariance transport v2 closure",
    "M-J2-UnlabeledOptimizerObservableTransportV2": "unlabeled optimizer-observable transport v2 closure",
    "M-J3-ArchitectureNeutralSNRTransportV2": "architecture-neutral SNR transport v2 closure",
}

CONTROL_IDS = [
    "C0-TaskOnlyAdamW",
    "C1-NoOpMatchedOverhead",
    "C2-RandomMatchedNorm",
    "C3-AdamWParallelDirection",
    "C4-SNROnlyAudit",
]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def hidden_forward(model: MLPBaseline, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h1 = F.silu(x @ model.w0)
    h2 = F.silu(h1 @ model.w1)
    logits = h2 @ model.w2
    return logits, h1, h2


def centered_cov(x: torch.Tensor) -> torch.Tensor:
    z = x.float() - x.float().mean(dim=0, keepdim=True)
    return (z.T @ z) / max(1, int(z.shape[0]) - 1)


def offdiag_energy(cov: torch.Tensor) -> torch.Tensor:
    return (cov - torch.diag(torch.diagonal(cov))).square().mean()


def functional_objective(
    model: MLPBaseline,
    x: torch.Tensor,
    candidate_id: str,
    anchor: dict[str, torch.Tensor] | None,
    seed: int,
) -> torch.Tensor:
    logits, h1, h2 = hidden_forward(model, x)
    if candidate_id == "M-F1-ActivationCovTransport":
        cov = centered_cov(h2)
        diag = torch.diagonal(cov).clamp_min(1.0e-8)
        return offdiag_energy(cov) + 0.01 * ((diag / diag.mean().clamp_min(1.0e-8) - 1.0).square().mean())
    if candidate_id == "M-F2-LogitCovStabilization":
        cov = centered_cov(logits)
        diag = torch.diagonal(cov).clamp_min(1.0e-8)
        return offdiag_energy(cov) + 0.02 * ((diag / diag.mean().clamp_min(1.0e-8) - 1.0).square().mean())
    if candidate_id == "M-F3-RandomCotangentJacobianSketch":
        gen = torch.Generator(device=x.device).manual_seed(int(seed))
        cot = torch.randn(logits.shape, device=x.device, generator=gen)
        cot = cot / cot.norm().clamp_min(1.0e-8)
        return (logits.float() * cot).sum().square() / max(1, int(logits.shape[0]))
    if candidate_id == "M-F4-HiddenSpectralBalance":
        cov1 = centered_cov(h1)
        cov2 = centered_cov(h2)
        d1 = torch.diagonal(cov1).clamp_min(1.0e-8)
        d2 = torch.diagonal(cov2).clamp_min(1.0e-8)
        return ((d1 / d1.mean().clamp_min(1.0e-8) - 1.0).square().mean() + (d2 / d2.mean().clamp_min(1.0e-8) - 1.0).square().mean())
    if candidate_id == "M-F5-GeometrySafeWeightAnchor":
        if anchor is None:
            return logits.float().square().mean() * 0.0
        loss = torch.zeros((), device=x.device)
        for name, param in model.named_parameters():
            loss = loss + (param.float() - anchor[name].float()).square().mean()
        return loss
    if candidate_id == "M-F6-InputJitterConsistency":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 31337)
        noise = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.035
        x_jitter = (x + noise).clamp(0.0, 1.0)
        logits_jitter, _h1_jitter, h2_jitter = hidden_forward(model, x_jitter)
        return (logits_jitter.float() - logits.detach().float()).square().mean() + 0.10 * (h2_jitter.float() - h2.detach().float()).square().mean()
    if candidate_id == "M-G1-UnlabeledCloneResponseCov":
        if anchor is None:
            target_h2 = h2.detach()
            target_logits = logits.detach()
        else:
            with torch.no_grad():
                clone = MLPBaseline(int(model.w0.shape[0]), int(model.w2.shape[1]), int(model.w1.shape[1]), int(seed), x.device).to(x.device)
                for name, param in clone.named_parameters():
                    param.copy_(anchor[name])
                target_logits, _target_h1, target_h2 = hidden_forward(clone, x)
        cov_loss = (centered_cov(h2) - centered_cov(target_h2).detach()).square().mean()
        logit_cov_loss = (centered_cov(logits) - centered_cov(target_logits).detach()).square().mean()
        return cov_loss + 0.15 * logit_cov_loss + 0.001 * logits.float().square().mean()
    if candidate_id == "M-G2-RandomCotangentResponseStability":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 43117)
        noise = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.025
        x_jitter = (x + noise).clamp(0.0, 1.0)
        logits_jitter, _h1_jitter, _h2_jitter = hidden_forward(model, x_jitter)
        cot = torch.randn(logits.shape, device=x.device, generator=gen, dtype=logits.dtype)
        cot = cot / cot.float().norm().clamp_min(1.0e-8)
        response = ((logits_jitter - logits.detach()).float() * cot.float()).sum(dim=1)
        return response.square().mean() + 0.01 * offdiag_energy(centered_cov(logits))
    if candidate_id == "M-G3-InputPerturbationConsistencyTransport":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 313370)
        noise = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.04
        x_jitter = (x + noise).clamp(0.0, 1.0)
        logits_jitter, _h1_jitter, h2_jitter = hidden_forward(model, x_jitter)
        consistency = (logits_jitter.float() - logits.detach().float()).square().mean()
        hidden_consistency = (h2_jitter.float() - h2.detach().float()).square().mean()
        cov_transport = (centered_cov(h2_jitter) - centered_cov(h2).detach()).square().mean()
        return consistency + 0.20 * hidden_consistency + 0.05 * cov_transport
    if candidate_id == "M-G4-HiddenCovarianceTransportWithControlResidual":
        cov1 = centered_cov(h1)
        cov2 = centered_cov(h2)
        d2 = torch.diagonal(cov2).clamp_min(1.0e-8)
        residual = offdiag_energy(cov2) - 0.25 * offdiag_energy(cov1).detach()
        balance = ((d2 / d2.mean().clamp_min(1.0e-8) - 1.0).square().mean())
        return residual.square() + 0.05 * balance + 0.001 * logits.float().square().mean()
    if candidate_id == "M-G5-ActivationSubspaceGuardThenTaskNeutral":
        cov2 = centered_cov(h2)
        trace = torch.diagonal(cov2).sum().clamp_min(1.0e-8)
        try:
            evals = torch.linalg.eigvalsh(cov2.float()).clamp_min(0.0)
            top_share = evals[-1] / trace
        except Exception:
            top_share = torch.diagonal(cov2).max() / trace
        probs = logits.softmax(dim=1)
        entropy = -(probs.clamp_min(1.0e-8).log() * probs).sum(dim=1).mean()
        uniform_entropy = math.log(max(1, int(logits.shape[1])))
        return top_share.square() + 0.02 * offdiag_energy(cov2) + 0.002 * (entropy - uniform_entropy).square()
    if candidate_id == "M-H1-UnlabeledMicroProbeResponsePredictor":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 58123)
        probe = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.020
        x_probe = (x + probe).clamp(0.0, 1.0)
        logits_probe, _h1_probe, h2_probe = hidden_forward(model, x_probe)
        if anchor is not None:
            with torch.no_grad():
                clone = MLPBaseline(int(model.w0.shape[0]), int(model.w2.shape[1]), int(model.w1.shape[1]), int(seed), x.device).to(x.device)
                for name, param in clone.named_parameters():
                    param.copy_(anchor[name])
                anchor_logits, _anchor_h1, anchor_h2 = hidden_forward(clone, x)
                anchor_probe_logits, _anchor_probe_h1, anchor_probe_h2 = hidden_forward(clone, x_probe)
                target_logit_response = anchor_probe_logits - anchor_logits
                target_h2_response = anchor_probe_h2 - anchor_h2
        else:
            target_logit_response = torch.zeros_like(logits_probe)
            target_h2_response = torch.zeros_like(h2_probe)
        logit_response = logits_probe - logits.detach()
        h2_response = h2_probe - h2.detach()
        return (logit_response.float() - target_logit_response.float()).square().mean() + 0.10 * (h2_response.float() - target_h2_response.float()).square().mean()
    if candidate_id == "M-H2-RandomCotangentLowRankResponseController":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 61057)
        rank = max(2, min(8, int(logits.shape[1])))
        cot_u = torch.randn((int(logits.shape[1]), rank), device=x.device, generator=gen, dtype=logits.dtype)
        cot_u = cot_u / cot_u.float().norm(dim=0, keepdim=True).clamp_min(1.0e-8).to(dtype=logits.dtype)
        h_dir = torch.randn((int(h2.shape[1]), rank), device=x.device, generator=gen, dtype=h2.dtype)
        h_dir = h_dir / h_dir.float().norm(dim=0, keepdim=True).clamp_min(1.0e-8).to(dtype=h2.dtype)
        projected_logits = logits @ cot_u
        projected_h2 = h2 @ h_dir
        logit_cov = centered_cov(projected_logits)
        hidden_cov = centered_cov(projected_h2)
        diag_l = torch.diagonal(logit_cov).clamp_min(1.0e-8)
        diag_h = torch.diagonal(hidden_cov).clamp_min(1.0e-8)
        return offdiag_energy(logit_cov) + 0.25 * offdiag_energy(hidden_cov) + 0.01 * ((diag_l / diag_l.mean().clamp_min(1.0e-8) - 1.0).square().mean() + (diag_h / diag_h.mean().clamp_min(1.0e-8) - 1.0).square().mean())
    if candidate_id == "M-H3-ActivationSpectrumGuardWithMatchedControls":
        cov1 = centered_cov(h1)
        cov2 = centered_cov(h2)
        trace1 = torch.diagonal(cov1).sum().clamp_min(1.0e-8)
        trace2 = torch.diagonal(cov2).sum().clamp_min(1.0e-8)
        try:
            evals1 = torch.linalg.eigvalsh(cov1.float()).clamp_min(0.0)
            evals2 = torch.linalg.eigvalsh(cov2.float()).clamp_min(0.0)
            top1 = evals1[-1] / trace1
            top2 = evals2[-1] / trace2
        except Exception:
            top1 = torch.diagonal(cov1).max() / trace1
            top2 = torch.diagonal(cov2).max() / trace2
        probs = logits.softmax(dim=1)
        confidence = probs.max(dim=1).values
        logit_norm = logits.float().norm(dim=1)
        return (top1 - top2.detach()).square() + 0.50 * top2.square() + 0.01 * offdiag_energy(cov2) + 0.001 * confidence.var() + 0.0005 * logit_norm.var()
    if candidate_id == "M-I1-CloneProbeCovarianceUpperBound":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 73001)
        probe = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.018
        x_probe = (x + probe).clamp(0.0, 1.0)
        logits_probe, _h1_probe, h2_probe = hidden_forward(model, x_probe)
        response = logits_probe - logits.detach()
        h_response = h2_probe - h2.detach()
        if anchor is not None:
            with torch.no_grad():
                clone = MLPBaseline(int(model.w0.shape[0]), int(model.w2.shape[1]), int(model.w1.shape[1]), int(seed), x.device).to(x.device)
                for name, param in clone.named_parameters():
                    param.copy_(anchor[name])
                clone_logits, _clone_h1, clone_h2 = hidden_forward(clone, x)
                clone_probe_logits, _clone_probe_h1, clone_probe_h2 = hidden_forward(clone, x_probe)
                target_response = clone_probe_logits - clone_logits
                target_h_response = clone_probe_h2 - clone_h2
        else:
            target_response = torch.zeros_like(response)
            target_h_response = torch.zeros_like(h_response)
        cov_gap = (centered_cov(response) - centered_cov(target_response).detach()).square().mean()
        h_cov_gap = (centered_cov(h_response) - centered_cov(target_h_response).detach()).square().mean()
        response_bound = F.relu(response.float().norm(dim=1) - target_response.detach().float().norm(dim=1).median().clamp_min(1.0e-6)).square().mean()
        return cov_gap + 0.20 * h_cov_gap + 0.02 * response_bound
    if candidate_id == "M-I2-PrecommitUnlabeledResponseStability":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 74117)
        noise_a = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.015
        noise_b = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.030
        logits_a, _h1_a, h2_a = hidden_forward(model, (x + noise_a).clamp(0.0, 1.0))
        logits_b, _h1_b, h2_b = hidden_forward(model, (x + noise_b).clamp(0.0, 1.0))
        response_a = logits_a - logits.detach()
        response_b = logits_b - logits.detach()
        scale_match = (response_a.float().norm(dim=1) * 2.0 - response_b.float().norm(dim=1)).square().mean()
        direction_match = 1.0 - F.cosine_similarity(response_a.float(), response_b.float(), dim=1, eps=1.0e-8).mean()
        hidden_stability = (h2_a.float() - h2.detach().float()).square().mean() + 0.50 * (h2_b.float() - h2.detach().float()).square().mean()
        return 0.02 * scale_match + direction_match.square() + 0.05 * hidden_stability
    if candidate_id == "M-I3-ControlResidualizedMicroProbe":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 75223)
        probe = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.020
        control = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.020
        logits_probe, _h1_probe, h2_probe = hidden_forward(model, (x + probe).clamp(0.0, 1.0))
        logits_control, _h1_control, h2_control = hidden_forward(model, (x + control).clamp(0.0, 1.0))
        resp = logits_probe - logits.detach()
        ctrl = logits_control.detach() - logits.detach()
        coef = (resp.float() * ctrl.float()).sum(dim=1, keepdim=True) / ctrl.float().square().sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        residual = resp.float() - coef * ctrl.float()
        hidden_residual = (h2_probe.float() - h2.detach().float()) - (h2_control.detach().float() - h2.detach().float())
        return residual.square().mean() + 0.05 * hidden_residual.square().mean() + 0.01 * offdiag_energy(centered_cov(residual))
    if candidate_id == "M-I4-ArchitectureNeutralSNRTransport":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 76367)
        rank = max(4, min(16, int(h2.shape[1]), int(logits.shape[0])))
        proj_h = torch.randn((int(h2.shape[1]), rank), device=x.device, generator=gen, dtype=h2.dtype)
        proj_h = proj_h / proj_h.float().norm(dim=0, keepdim=True).clamp_min(1.0e-8).to(dtype=h2.dtype)
        proj_l = torch.randn((int(logits.shape[1]), min(rank, int(logits.shape[1]))), device=x.device, generator=gen, dtype=logits.dtype)
        proj_l = proj_l / proj_l.float().norm(dim=0, keepdim=True).clamp_min(1.0e-8).to(dtype=logits.dtype)
        hp = h2 @ proj_h
        lp = logits @ proj_l
        h_signal = hp.float().mean(dim=0).square()
        h_noise = hp.float().var(dim=0, unbiased=False).clamp_min(1.0e-8)
        l_signal = lp.float().mean(dim=0).square()
        l_noise = lp.float().var(dim=0, unbiased=False).clamp_min(1.0e-8)
        h_snr = h_signal / h_noise
        l_snr = l_signal / l_noise
        snr_balance = (h_snr / h_snr.mean().clamp_min(1.0e-8) - 1.0).square().mean() + (l_snr / l_snr.mean().clamp_min(1.0e-8) - 1.0).square().mean()
        cov_guard = offdiag_energy(centered_cov(hp)) + offdiag_energy(centered_cov(lp))
        return 0.05 * snr_balance + 0.02 * cov_guard + 0.0005 * logits.float().square().mean()
    if candidate_id == "M-J1-CloneProbeCovarianceTransportV2":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 78101)
        probe = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.015
        x_probe = (x + probe).clamp(0.0, 1.0)
        logits_probe, _h1_probe, h2_probe = hidden_forward(model, x_probe)
        response = logits_probe - logits.detach()
        h_response = h2_probe - h2.detach()
        if anchor is not None:
            with torch.no_grad():
                clone = MLPBaseline(int(model.w0.shape[0]), int(model.w2.shape[1]), int(model.w1.shape[1]), int(seed), x.device).to(x.device)
                for name, param in clone.named_parameters():
                    param.copy_(anchor[name])
                clone_logits, _clone_h1, clone_h2 = hidden_forward(clone, x)
                clone_probe_logits, _clone_probe_h1, clone_probe_h2 = hidden_forward(clone, x_probe)
                target_response = clone_probe_logits - clone_logits
                target_h_response = clone_probe_h2 - clone_h2
        else:
            target_response = torch.zeros_like(response)
            target_h_response = torch.zeros_like(h_response)
        cov_gap = (centered_cov(response) - centered_cov(target_response).detach()).square().mean()
        h_cov_gap = (centered_cov(h_response) - centered_cov(target_h_response).detach()).square().mean()
        norm_gap = (response.float().norm(dim=1) - target_response.detach().float().norm(dim=1)).square().mean()
        return cov_gap + 0.25 * h_cov_gap + 0.01 * norm_gap + 0.00025 * logits.float().square().mean()
    if candidate_id == "M-J2-UnlabeledOptimizerObservableTransportV2":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 79217)
        jitter = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.020
        logits_jitter, _h1_jitter, h2_jitter = hidden_forward(model, (x + jitter).clamp(0.0, 1.0))
        response = logits_jitter - logits.detach()
        h_response = h2_jitter - h2.detach()
        response_energy = response.float().square().mean(dim=0)
        hidden_energy = h_response.float().square().mean(dim=0)
        response_balance = (response_energy / response_energy.mean().clamp_min(1.0e-8) - 1.0).square().mean()
        hidden_balance = (hidden_energy / hidden_energy.mean().clamp_min(1.0e-8) - 1.0).square().mean()
        probs = logits.softmax(dim=1)
        entropy = -(probs.clamp_min(1.0e-8).log() * probs).sum(dim=1)
        return 0.05 * response_balance + 0.02 * hidden_balance + 0.002 * entropy.var(unbiased=False) + 0.00025 * logits.float().square().mean()
    if candidate_id == "M-J3-ArchitectureNeutralSNRTransportV2":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 80341)
        rank = max(4, min(16, int(h2.shape[1]), int(logits.shape[0])))
        proj_h = torch.randn((int(h2.shape[1]), rank), device=x.device, generator=gen, dtype=h2.dtype)
        proj_h = proj_h / proj_h.float().norm(dim=0, keepdim=True).clamp_min(1.0e-8).to(dtype=h2.dtype)
        proj_l = torch.randn((int(logits.shape[1]), min(rank, int(logits.shape[1]))), device=x.device, generator=gen, dtype=logits.dtype)
        proj_l = proj_l / proj_l.float().norm(dim=0, keepdim=True).clamp_min(1.0e-8).to(dtype=logits.dtype)
        hp = h2 @ proj_h
        lp = logits @ proj_l
        h_snr = hp.float().mean(dim=0).square() / hp.float().var(dim=0, unbiased=False).clamp_min(1.0e-8)
        l_snr = lp.float().mean(dim=0).square() / lp.float().var(dim=0, unbiased=False).clamp_min(1.0e-8)
        snr_gap = (h_snr / h_snr.mean().clamp_min(1.0e-8) - l_snr.mean().detach() / l_snr.mean().detach().clamp_min(1.0e-8)).square().mean()
        cov_guard = offdiag_energy(centered_cov(hp)) + 0.50 * offdiag_energy(centered_cov(lp))
        return 0.04 * snr_gap + 0.02 * cov_guard + 0.00025 * logits.float().square().mean()
    raise KeyError(candidate_id)


def grad_delta_from_objective(model: MLPBaseline, objective: torch.Tensor) -> list[torch.Tensor]:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(objective, params, retain_graph=False, create_graph=False, allow_unused=True)
    out: list[torch.Tensor] = []
    for p, g in zip(params, grads):
        out.append(torch.zeros_like(p) if g is None else -g.detach())
    return out


def delta_norm(delta: list[torch.Tensor]) -> float:
    total = torch.zeros((), device=delta[0].device if delta else "cpu")
    for d in delta:
        total = total + d.float().square().sum()
    return float(total.sqrt().item())


def scale_delta(delta: list[torch.Tensor], target_norm: float) -> list[torch.Tensor]:
    norm = delta_norm(delta)
    if not math.isfinite(norm) or norm <= 0.0:
        return [torch.zeros_like(d) for d in delta]
    scale = float(target_norm) / max(norm, 1.0e-12)
    return [d * scale for d in delta]


def random_delta_like(delta: list[torch.Tensor], target_norm: float, seed: int) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for idx, d in enumerate(delta):
        gen = torch.Generator(device=d.device).manual_seed(int(seed) + 7919 * idx)
        out.append(torch.randn(d.shape, device=d.device, generator=gen, dtype=d.dtype))
    return scale_delta(out, target_norm)


def apply_delta(model: MLPBaseline, delta: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, d in zip([p for p in model.parameters() if p.requires_grad], delta):
            p.add_(d)


def param_norm(model: MLPBaseline) -> float:
    total = torch.zeros((), device=model.w0.device)
    for p in model.parameters():
        total = total + p.detach().float().square().sum()
    return float(total.sqrt().item())
