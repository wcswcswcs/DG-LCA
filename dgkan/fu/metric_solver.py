"""Metric-as-geometry readout solver for v22.06.

This module implements the smallest audit-safe solver required for the v22.06
route: a readout exact solve for

    min_u ||J_theta u - delta_f_target||_G^2 + rho ||u||^2.

It is deliberately readout/block limited (solver level S1), train-stream only,
and emits enough diagnostics to distinguish it from flat-gradient metric
proxies.
"""

from __future__ import annotations

from math import isfinite
import time
from typing import Any

import torch
import torch.nn.functional as F

from dgkan.fu.basis_channel_metric import basis_channel_energy, parameter_channel_slices
from dgkan.fu.core import UpdateTensor, flat_grad, flat_params, load_flat_params
from dgkan.fu.function_space_metrics import (
    fisher_diag_from_logits,
    l2_energy,
    output_metric_energies,
    rkhs_graph_energy,
    sobolev_fd_energy,
)
from dgkan.fu.jacobian_sketch import finite_difference_jvp, function_displacement_cosine


EPS = 1.0e-8


V2206_SOLVER_CONFIGS: dict[str, dict[str, str]] = {
    "M259-V2206MetricSolverT0G0ReadoutFU": {
        "target_family": "T0-LossCotangent",
        "metric_family": "G0-L2",
        "solver_level": "S1-ReadoutExactSolve",
    },
    "M260-V2206MetricSolverT1G0SplitTransferFU": {
        "target_family": "T1-SplitTransfer",
        "metric_family": "G0-L2",
        "solver_level": "S1-ReadoutExactSolve",
    },
    "M261-V2206MetricSolverT3G3SignalSobolevFU": {
        "target_family": "T3-SignalChannel",
        "metric_family": "G3-SobolevH1",
        "solver_level": "S1-ReadoutExactSolve",
    },
    "M262-V2206MetricSolverT4G3SmoothManifoldFU": {
        "target_family": "T4-SmoothManifold",
        "metric_family": "G3-SobolevH1",
        "solver_level": "S1-ReadoutExactSolve",
    },
    "M263-V2206MetricSolverT5G6LowNDSFU": {
        "target_family": "T5-LowNDS",
        "metric_family": "G6-LowNDS",
        "solver_level": "S1-ReadoutExactSolve",
    },
    "M264-V2206MetricSolverT6G0DualMemoryFU": {
        "target_family": "T6-DualMemorySource",
        "metric_family": "G0-L2",
        "solver_level": "S1-ReadoutExactSolve",
    },
    "M265-V2206MetricSolverT7G0HiddenBlockFU": {
        "target_family": "T7-BlockSourceChannel",
        "metric_family": "G0-L2",
        "solver_level": "S2-ReadoutExactHiddenResidualBlock",
        "hidden_residual_scale": "0.35",
    },
    "M266-V2206MetricSolverT8G0AdaptiveHiddenBlockFU": {
        "target_family": "T8-AdaptiveBlockSourceChannel",
        "metric_family": "G0-L2",
        "solver_level": "S2-ReadoutExactAdaptiveHiddenResidualBlock",
        "hidden_residual_scale": "0.35",
        "hidden_function_fraction": "0.25",
    },
    "M267-V2206MetricSolverT9G0C3GatedHiddenBlockFU": {
        "target_family": "T9-C3GatedBlockSourceChannel",
        "metric_family": "G0-L2",
        "solver_level": "S2-ReadoutExactC3GatedHiddenResidualBlock",
        "hidden_residual_scale": "1.00",
    },
    "M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU": {
        "target_family": "T10-CompensatedBlockSourceChannel",
        "metric_family": "G0-L2",
        "solver_level": "S2-ReadoutExactCompensatedHiddenResidualBlock",
        "hidden_residual_scale": "0.35",
    },
    "M269-V2206MetricSolverT11G0EarlyObservableFU": {
        "target_family": "T11-EarlyObservableSourceChannel",
        "metric_family": "G0-L2",
        "solver_level": "S2-ReadoutExactCompensatedHiddenResidualBlock",
        "hidden_residual_scale": "0.28",
    },
    "M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU": {
        "target_family": "T12-SoftCompensatedBlockSourceChannel",
        "metric_family": "G0-L2",
        "solver_level": "S2-ReadoutExactSoftCompensatedHiddenResidualBlock",
        "hidden_residual_scale": "0.35",
    },
}


def _one_hot_like(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.one_hot(labels.to(dtype=torch.long), num_classes=int(logits.shape[-1])).to(device=logits.device, dtype=logits.dtype)


def _split_batch(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    split1 = max(1, n // 3)
    split2 = max(split1 + 1, (2 * n) // 3)
    xb1, yb1 = x[:split1], y[:split1]
    xb2, yb2 = x[split1:split2], y[split1:split2]
    xb3, yb3 = x[split2:], y[split2:]
    if int(xb3.shape[0]) == 0:
        xb3, yb3 = xb2, yb2
    return xb1, yb1, xb2, yb2, xb3, yb3


def _target_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_family: str,
    *,
    reference_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    probs = torch.softmax(logits.detach().float(), dim=-1)
    ref = _one_hot_like(logits, labels).float() - probs
    diag: dict[str, Any] = {"target_family": target_family}
    if target_family == "T0-LossCotangent":
        return ref, diag
    if target_family == "T1-SplitTransfer":
        if reference_target is None:
            return ref, diag
        ref_mean = reference_target.detach().float().mean(dim=0)
        reference = torch.sign(ref_mean)
        current = torch.sign(ref.mean(dim=0))
        mask = ((reference == current) & (current != 0.0)).to(device=logits.device, dtype=ref.dtype)
        strength = (ref_mean.abs() / ref_mean.abs().max().clamp_min(EPS)).to(device=logits.device, dtype=ref.dtype)
        weight = torch.where(mask > 0.0, strength.clamp(0.15, 1.0), torch.zeros_like(strength))
        diag["target_consensus_density"] = float(mask.mean().item()) if mask.numel() else 0.0
        diag["target_transfer_weight_mean"] = float(weight.mean().item()) if weight.numel() else 0.0
        return ref * weight.reshape(1, -1), diag
    if target_family == "T6-DualMemorySource":
        confidence = probs.gather(1, labels.reshape(-1, 1)).clamp(0.0, 1.0)
        short_weight = (1.0 - confidence).clamp(0.15, 1.0)
        short_target = ref * short_weight
        if reference_target is None:
            diag["target_dual_memory_agreement"] = 1.0
            diag["target_dual_memory_weight_mean"] = float(short_weight.mean().item()) if short_weight.numel() else 0.0
            return short_target, diag
        ref_mean = reference_target.detach().float().mean(dim=0).to(device=logits.device, dtype=ref.dtype)
        cur_mean = ref.mean(dim=0)
        ref_sign = torch.sign(ref_mean)
        cur_sign = torch.sign(cur_mean)
        agree = (ref_sign == cur_sign) & (cur_sign != 0.0)
        ref_scale = ref_mean.abs() / ref_mean.abs().max().clamp_min(EPS)
        cur_scale = cur_mean.abs() / cur_mean.abs().max().clamp_min(EPS)
        long_axis = cur_sign * torch.minimum(ref_scale, cur_scale)
        long_target = long_axis.reshape(1, -1).expand_as(ref)
        gate = torch.where(agree, torch.ones_like(cur_mean), torch.full_like(cur_mean, 0.25))
        dual_target = (0.72 * short_target + 0.28 * long_target) * gate.reshape(1, -1)
        diag["target_dual_memory_agreement"] = float(agree.float().mean().item()) if agree.numel() else 0.0
        diag["target_dual_memory_weight_mean"] = float(gate.mean().item()) if gate.numel() else 0.0
        diag["target_dual_memory_short_weight_mean"] = float(short_weight.mean().item()) if short_weight.numel() else 0.0
        return dual_target, diag
    if target_family == "T7-BlockSourceChannel":
        target, inner = _target_from_logits(logits, labels, "T6-DualMemorySource", reference_target=reference_target)
        inner["target_family"] = target_family
        inner["target_block_structured_hidden_readout"] = 1
        return target, inner
    if target_family == "T8-AdaptiveBlockSourceChannel":
        target, inner = _target_from_logits(logits, labels, "T6-DualMemorySource", reference_target=reference_target)
        inner["target_family"] = target_family
        inner["target_adaptive_hidden_block"] = 1
        return target, inner
    if target_family == "T9-C3GatedBlockSourceChannel":
        target, inner = _target_from_logits(logits, labels, "T6-DualMemorySource", reference_target=reference_target)
        inner["target_family"] = target_family
        inner["target_c3_gated_hidden_block"] = 1
        return target, inner
    if target_family == "T10-CompensatedBlockSourceChannel":
        target, inner = _target_from_logits(logits, labels, "T6-DualMemorySource", reference_target=reference_target)
        inner["target_family"] = target_family
        inner["target_compensated_hidden_block"] = 1
        return target, inner
    if target_family == "T12-SoftCompensatedBlockSourceChannel":
        target, inner = _target_from_logits(logits, labels, "T6-DualMemorySource", reference_target=reference_target)
        inner["target_family"] = target_family
        inner["target_soft_compensated_hidden_block"] = 1
        return target, inner
    if target_family == "T11-EarlyObservableSourceChannel":
        # Build a train-only class axis from two disjoint class-conditional
        # halves. This targets C1 source observability before optimizer
        # transport, without using audit metrics or future horizons.
        n = int(ref.shape[0])
        split = max(1, n // 2)
        labels_long = labels.to(dtype=torch.long)
        raw = ref.detach().float()
        target = torch.zeros_like(raw)
        classes = int(raw.shape[-1])
        active = 0
        cos_values: list[float] = []
        true_prob = probs.gather(1, labels_long.reshape(-1, 1)).clamp(0.0, 1.0)
        mid_debt = (4.0 * true_prob * (1.0 - true_prob)).clamp(0.10, 1.0)
        for cls in range(classes):
            mask_a = labels_long[:split] == cls
            mask_b = labels_long[split:] == cls
            if bool(mask_a.any().item()) and bool(mask_b.any().item()):
                axis_a = raw[:split][mask_a].mean(dim=0)
                axis_b = raw[split:][mask_b].mean(dim=0)
                denom = torch.linalg.vector_norm(axis_a).clamp_min(EPS) * torch.linalg.vector_norm(axis_b).clamp_min(EPS)
                cos = torch.dot(axis_a, axis_b) / denom
                cos_values.append(float(cos.item()))
                if float(cos.item()) > 0.0:
                    axis = 0.50 * (axis_a + axis_b)
                    axis = axis - axis.mean()
                    axis_norm = torch.linalg.vector_norm(axis).clamp_min(EPS)
                    row_mask = labels_long == cls
                    row_norm = torch.linalg.vector_norm(raw[row_mask], dim=1, keepdim=True).clamp_min(EPS)
                    target[row_mask] = axis.reshape(1, -1) / axis_norm * row_norm
                    active += int(row_mask.sum().item())
        if active == 0:
            target = raw.clone()
        density = float(active) / float(max(1, n))
        target = 0.72 * target + 0.28 * raw
        target = target * mid_debt.to(device=target.device, dtype=target.dtype)
        if reference_target is not None:
            ref_mean = reference_target.detach().float().mean(dim=0).to(device=target.device, dtype=target.dtype)
            cur_mean = target.mean(dim=0)
            agree = (torch.sign(ref_mean) == torch.sign(cur_mean)) & (torch.sign(cur_mean) != 0.0)
            gate = torch.where(agree, torch.ones_like(cur_mean), torch.full_like(cur_mean, 0.35))
            target = target * gate.reshape(1, -1)
            diag["target_early_observable_reference_agreement"] = float(agree.float().mean().item()) if agree.numel() else 0.0
        diag["target_early_observable_class_density"] = density
        diag["target_early_observable_class_cos_mean"] = float(sum(cos_values) / len(cos_values)) if cos_values else 0.0
        diag["target_early_observable_mid_debt_mean"] = float(mid_debt.mean().item()) if mid_debt.numel() else 0.0
        diag["target_compensated_hidden_block"] = 1
        return target, diag
    if target_family == "T3-SignalChannel":
        confidence = probs.gather(1, labels.reshape(-1, 1)).clamp(0.0, 1.0)
        weight = (1.0 - confidence).clamp(0.10, 1.0)
        centered = ref - ref.mean(dim=0, keepdim=True)
        signal = 0.65 * ref + 0.35 * centered
        diag["target_signal_weight_mean"] = float(weight.mean().item()) if weight.numel() else 0.0
        return signal * weight, diag
    if target_family == "T4-SmoothManifold":
        smooth = ref.clone()
        if smooth.shape[0] > 2:
            smooth[1:-1] = 0.25 * ref[:-2] + 0.50 * ref[1:-1] + 0.25 * ref[2:]
        diag["target_smooth_manifold"] = 1
        return smooth, diag
    if target_family == "T5-LowNDS":
        smooth = ref.clone()
        if smooth.shape[0] > 2:
            smooth[1:-1] = 0.20 * ref[:-2] + 0.60 * ref[1:-1] + 0.20 * ref[2:]
        smooth = smooth - 0.50 * smooth.mean(dim=0, keepdim=True)
        confidence = probs.gather(1, labels.reshape(-1, 1)).clamp(0.0, 1.0)
        weight = (1.0 - confidence).clamp(0.05, 0.75)
        low_nds = smooth * weight
        diag["target_low_nds"] = 1
        diag["target_low_nds_weight_mean"] = float(weight.mean().item()) if weight.numel() else 0.0
        return low_nds, diag
    return ref, diag


def _metric_weights(logits: torch.Tensor, metric_family: str) -> torch.Tensor:
    if metric_family == "G1-DiagFisher":
        return fisher_diag_from_logits(logits.detach().float()).clamp_min(EPS)
    if metric_family == "G3-SobolevH1":
        # S1 readout solve uses a diagonal surrogate but records true Sobolev
        # audit energy separately.
        return torch.ones_like(logits.detach().float())
    if metric_family == "G6-LowNDS":
        lf = logits.detach().float()
        if lf.shape[0] > 2:
            curvature = torch.zeros_like(lf)
            curvature[1:-1] = (lf[:-2] - 2.0 * lf[1:-1] + lf[2:]).abs()
            return (1.0 / (1.0 + curvature)).clamp(0.10, 1.0)
        return torch.ones_like(lf)
    return torch.ones_like(logits.detach().float())


def _find_readout(model: torch.nn.Module) -> tuple[str, torch.nn.Parameter | None]:
    for name, p in model.named_parameters():
        low = name.lower()
        if p.requires_grad and p.ndim == 2 and ("w2" in low or "readout" in low or "classifier" in low):
            return name, p
    return "", None


def solve_metric_readout_update(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    target_family: str = "T0-LossCotangent",
    metric_family: str = "G0-L2",
    mechanism: str = "M259-V2206MetricSolverT0G0ReadoutFU",
    damping: float = 1.0e-3,
    hidden_residual_scale: float = 0.35,
    hidden_function_fraction: float = 0.25,
    target_scale: float = 1.0,
    block_role: str = "all",
    seed: int = 0,
) -> UpdateTensor:
    start = time.perf_counter()
    device = x.device
    g_ref = flat_grad(model, device)
    readout_name, readout = _find_readout(model)
    if readout is None or not hasattr(model, "frozen_readout_features") or int(x.shape[0]) < 6:
        return UpdateTensor(
            g_ref.clone(),
            "metric_solver_fallback_gradient",
            "subtract",
            "MetricProxy",
            "v22_06_metric_solver_unavailable",
            mechanism,
            role="fallback",
            one_step_descent_claim=0,
            diagnostics={"solver_status": "readout_exact_unavailable", "is_proxy": 1},
        )

    before = flat_params(model).detach().to(device=device)
    saved_grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
    xb1, yb1, xb2, yb2, xb3, yb3 = _split_batch(x, y)
    with torch.no_grad():
        base1 = model(xb1).detach().float()
        base2 = model(xb2).detach().float()
        base3 = model(xb3).detach().float()
        feats1 = model.frozen_readout_features(xb1).detach().float()
        feats2 = model.frozen_readout_features(xb2).detach().float()
        target1, d1 = _target_from_logits(base1, yb1, target_family)
        target2, d2 = _target_from_logits(base2, yb2, target_family, reference_target=target1)
        feats = torch.cat([feats1, feats2], dim=0)
        target = torch.cat([target1, target2], dim=0) * float(target_scale)
        weights = _metric_weights(torch.cat([base1, base2], dim=0), metric_family).to(device=device, dtype=feats.dtype)
        row_weights = weights.mean(dim=1, keepdim=True).clamp_min(EPS)
        wfeats = feats * row_weights.sqrt()
        wtarget = target.to(device=device, dtype=feats.dtype) * row_weights.sqrt()
        gram = wfeats.T @ wfeats + float(damping) * torch.eye(int(wfeats.shape[1]), device=device, dtype=wfeats.dtype)
        rhs = wfeats.T @ wtarget
        try:
            delta_w = torch.linalg.solve(gram, rhs)
        except Exception:
            delta_w = torch.linalg.lstsq(gram, rhs).solution

        chunks: list[torch.Tensor] = []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if pname == readout_name:
                chunks.append((-delta_w).to(device=device, dtype=p.dtype).reshape(-1))
            else:
                chunks.append(torch.zeros_like(p, device=device).reshape(-1))
        update_vec = torch.cat(chunks) if chunks else torch.zeros_like(g_ref)
        hidden_target_families = {
            "T7-BlockSourceChannel",
            "T8-AdaptiveBlockSourceChannel",
            "T9-C3GatedBlockSourceChannel",
            "T10-CompensatedBlockSourceChannel",
            "T11-EarlyObservableSourceChannel",
            "T12-SoftCompensatedBlockSourceChannel",
        }
        compensated_target_families = {
            "T10-CompensatedBlockSourceChannel",
            "T11-EarlyObservableSourceChannel",
            "T12-SoftCompensatedBlockSourceChannel",
        }
        soft_compensated = target_family == "T12-SoftCompensatedBlockSourceChannel"
        hidden_residual_vec = torch.zeros_like(update_vec)
        base_readout_update_vec = update_vec.detach().clone()
        hidden_residual_norm = 0.0
        hidden_residual_fraction = 0.0
        hidden_function_norm = 0.0
        hidden_function_cap_ratio = ""
        hidden_compensation_norm = 0.0
        hidden_compensation_scale = ""
        hidden_compensation_mix = ""
        hidden_compensation_residual_ratio = ""
        hidden_compensation_b2_gain = ""
        if target_family in hidden_target_families:
            hidden_mask = torch.zeros_like(update_vec, device=device)
            for row in parameter_channel_slices(model):
                if str(row.get("channel")) == "hidden":
                    hidden_mask[int(row["start"]) : int(row["end"])] = 1.0
            hidden_grad = g_ref.to(device=device, dtype=update_vec.dtype) * hidden_mask.to(device=device, dtype=update_vec.dtype)
            hidden_norm = torch.linalg.vector_norm(hidden_grad.detach()).clamp_min(EPS)
            readout_norm = torch.linalg.vector_norm(update_vec.detach()).clamp_min(EPS)
            if float(hidden_norm.item()) > EPS and float(readout_norm.item()) > EPS:
                hidden_residual_vec = hidden_grad / hidden_norm * (float(hidden_residual_scale) * readout_norm)
                if target_family == "T8-AdaptiveBlockSourceChannel":
                    load_flat_params(model, before - hidden_residual_vec.to(device=device, dtype=before.dtype))
                    hidden_after1 = model(xb1).detach().float()
                    hidden_after2 = model(xb2).detach().float()
                    load_flat_params(model, before)
                    hidden_disp = torch.cat([hidden_after1 - base1, hidden_after2 - base2], dim=0)
                    hidden_function_norm_tensor = torch.linalg.vector_norm(hidden_disp.detach().float()).clamp_min(EPS)
                    target_norm_for_cap = torch.linalg.vector_norm(target.detach().float()).clamp_min(EPS)
                    cap = float(hidden_function_fraction) * target_norm_for_cap
                    if float(hidden_function_norm_tensor.item()) > float(cap.item()):
                        scale = cap / hidden_function_norm_tensor
                        hidden_residual_vec = hidden_residual_vec * scale.to(device=device, dtype=hidden_residual_vec.dtype)
                        hidden_function_cap_ratio = float(scale.item())
                    else:
                        hidden_function_cap_ratio = 1.0
                    hidden_function_norm = float(min(hidden_function_norm_tensor.item(), cap.item()))
                elif target_family == "T9-C3GatedBlockSourceChannel":
                    proposed_hidden = hidden_residual_vec.detach().clone()
                    chosen_hidden = torch.zeros_like(proposed_hidden)
                    chosen_scale = 0.0
                    chosen_function_norm = 0.0
                    for scale_value in (1.0, 0.50, 0.25, 0.125, 0.0625, 0.0):
                        cand_hidden = proposed_hidden * scale_value
                        cand_update = update_vec + cand_hidden
                        load_flat_params(model, before - cand_update.to(device=device, dtype=before.dtype))
                        cand_after1 = model(xb1).detach().float()
                        cand_after2 = model(xb2).detach().float()
                        load_flat_params(model, before)
                        cand_actual = torch.cat([cand_after1 - base1, cand_after2 - base2], dim=0)
                        cand_residual = target.to(device=device, dtype=cand_actual.dtype) - cand_actual
                        cand_centered = target.reshape(-1).to(device=device, dtype=cand_actual.dtype)
                        cand_centered = cand_centered - cand_centered.mean()
                        cand_r2 = 1.0 - torch.sum(cand_residual.reshape(-1).square()) / torch.sum(cand_centered.square()).clamp_min(EPS)
                        cand_ratio = torch.linalg.vector_norm(cand_residual.detach().float()) / torch.linalg.vector_norm(target.detach().float()).clamp_min(EPS)
                        if float(cand_ratio.item()) <= 0.80 and float(cand_r2.item()) >= 0.50:
                            chosen_hidden = cand_hidden
                            chosen_scale = scale_value
                            hidden_disp = cand_actual - (feats @ delta_w).to(device=device, dtype=cand_actual.dtype)
                            chosen_function_norm = float(torch.linalg.vector_norm(hidden_disp.detach().float()).item())
                            break
                    hidden_residual_vec = chosen_hidden
                    hidden_function_cap_ratio = chosen_scale
                    hidden_function_norm = chosen_function_norm
                elif target_family in compensated_target_families:
                    proposed_hidden = hidden_residual_vec.detach().clone()
                    best_score = -1.0e9
                    chosen_hidden = torch.zeros_like(proposed_hidden)
                    chosen_update = update_vec.detach().clone()
                    chosen_delta_w = delta_w.detach().clone()
                    chosen_scale = 0.0
                    chosen_function_norm = 0.0
                    chosen_comp_norm = 0.0
                    chosen_mix = 1.0
                    chosen_ratio = ""
                    chosen_b2 = ""
                    chosen_gram = gram.detach().clone()
                    for scale_value in (1.0, 0.75, 0.50, 0.25, 0.125, 0.0):
                        cand_hidden = proposed_hidden * scale_value
                        load_flat_params(model, before - cand_hidden.to(device=device, dtype=before.dtype))
                        hidden_base1 = model(xb1).detach().float()
                        hidden_base2 = model(xb2).detach().float()
                        hidden_feats1 = model.frozen_readout_features(xb1).detach().float()
                        hidden_feats2 = model.frozen_readout_features(xb2).detach().float()
                        load_flat_params(model, before)
                        hidden_actual = torch.cat([hidden_base1 - base1, hidden_base2 - base2], dim=0)
                        comp_target = target.to(device=device, dtype=hidden_actual.dtype) - hidden_actual
                        comp_feats = torch.cat([hidden_feats1, hidden_feats2], dim=0).to(device=device, dtype=feats.dtype)
                        comp_wfeats = comp_feats * row_weights.sqrt()
                        comp_wtarget = comp_target.to(device=device, dtype=feats.dtype) * row_weights.sqrt()
                        comp_gram = comp_wfeats.T @ comp_wfeats + float(damping) * torch.eye(int(comp_wfeats.shape[1]), device=device, dtype=comp_wfeats.dtype)
                        comp_rhs = comp_wfeats.T @ comp_wtarget
                        try:
                            comp_delta_w = torch.linalg.solve(comp_gram, comp_rhs)
                        except Exception:
                            comp_delta_w = torch.linalg.lstsq(comp_gram, comp_rhs).solution
                        comp_chunks: list[torch.Tensor] = []
                        for pname, p in model.named_parameters():
                            if not p.requires_grad:
                                continue
                            if pname == readout_name:
                                comp_chunks.append((-comp_delta_w).to(device=device, dtype=p.dtype).reshape(-1))
                            else:
                                comp_chunks.append(torch.zeros_like(p, device=device).reshape(-1))
                        comp_update_vec = torch.cat(comp_chunks) if comp_chunks else torch.zeros_like(g_ref)
                        mix_values = (0.75, 0.50, 0.25, 0.0, 1.0) if soft_compensated else (1.0,)
                        for mix_value in mix_values:
                            cand_readout = float(mix_value) * comp_update_vec + (1.0 - float(mix_value)) * base_readout_update_vec
                            cand_update = cand_readout + cand_hidden
                            load_flat_params(model, before - cand_update.to(device=device, dtype=before.dtype))
                            cand_after1 = model(xb1).detach().float()
                            cand_after2 = model(xb2).detach().float()
                            load_flat_params(model, before)
                            cand_actual = torch.cat([cand_after1 - base1, cand_after2 - base2], dim=0)
                            cand_residual = target.to(device=device, dtype=cand_actual.dtype) - cand_actual
                            cand_centered = target.reshape(-1).to(device=device, dtype=cand_actual.dtype)
                            cand_centered = cand_centered - cand_centered.mean()
                            cand_r2 = 1.0 - torch.sum(cand_residual.reshape(-1).square()) / torch.sum(cand_centered.square()).clamp_min(EPS)
                            cand_ratio = torch.linalg.vector_norm(cand_residual.detach().float()) / torch.linalg.vector_norm(target.detach().float()).clamp_min(EPS)
                            cand_b2 = F.cross_entropy(base2, yb2) - F.cross_entropy(cand_after2, yb2)
                            c3_ok = int(float(cand_ratio.item()) <= 0.80 and float(cand_r2.item()) >= 0.50 and float(cand_b2.item()) > 0.005)
                            if soft_compensated:
                                score = (
                                    (1000.0 * c3_ok)
                                    + float(cand_r2.item())
                                    - float(cand_ratio.item())
                                    + (0.25 * float(scale_value))
                                    + (0.15 * (1.0 - float(mix_value)))
                                    + (0.01 * float(cand_b2.item()))
                                )
                            else:
                                score = (1000.0 * c3_ok) + float(cand_r2.item()) - float(cand_ratio.item()) + (0.10 * float(scale_value)) + (0.01 * float(cand_b2.item()))
                            if score > best_score:
                                best_score = score
                                chosen_hidden = cand_hidden
                                chosen_update = cand_update
                                chosen_delta_w = comp_delta_w
                                chosen_scale = scale_value
                                chosen_mix = float(mix_value)
                                chosen_function_norm = float(torch.linalg.vector_norm(hidden_actual.detach().float()).item())
                                chosen_comp_norm = float(torch.linalg.vector_norm(cand_readout.detach().float()).item())
                                chosen_ratio = float(cand_ratio.item())
                                chosen_b2 = float(cand_b2.item())
                                chosen_gram = comp_gram.detach().clone()
                    hidden_residual_vec = chosen_hidden
                    update_vec = chosen_update
                    delta_w = chosen_delta_w
                    gram = chosen_gram
                    hidden_function_cap_ratio = chosen_scale
                    hidden_function_norm = chosen_function_norm
                    hidden_compensation_norm = chosen_comp_norm
                    hidden_compensation_scale = chosen_scale
                    hidden_compensation_mix = chosen_mix
                    hidden_compensation_residual_ratio = chosen_ratio
                    hidden_compensation_b2_gain = chosen_b2
                if target_family not in compensated_target_families:
                    update_vec = update_vec + hidden_residual_vec
                hidden_residual_norm = float(torch.linalg.vector_norm(hidden_residual_vec.detach()).item())
                hidden_residual_fraction = hidden_residual_norm / float(torch.linalg.vector_norm(update_vec.detach()).clamp_min(EPS).item())

        block_role_normalized = str(block_role or "all")
        if block_role_normalized != "all":
            allowed_channels = {
                "hidden_only": {"hidden"},
                "readout_only": {"readout"},
                "basis_only": {"basis"},
            }.get(block_role_normalized, set())
            block_mask = torch.zeros_like(update_vec, device=device)
            if allowed_channels:
                for row in parameter_channel_slices(model):
                    if str(row.get("channel")) in allowed_channels:
                        block_mask[int(row["start"]) : int(row["end"])] = 1.0
            update_vec = update_vec * block_mask.to(device=device, dtype=update_vec.dtype)

        # Read back actual function displacement under the sign convention used
        # by apply_update(sign_rule="subtract").
        load_flat_params(model, before - update_vec.to(device=device, dtype=before.dtype))
        after1 = model(xb1).detach().float()
        after2 = model(xb2).detach().float()
        after3 = model(xb3).detach().float()
        load_flat_params(model, before)
        actual = torch.cat([after1 - base1, after2 - base2], dim=0)
        residual = target.to(device=device, dtype=actual.dtype) - actual
        centered = target.reshape(-1).to(device=device, dtype=actual.dtype)
        centered = centered - centered.mean()
        r2 = 1.0 - torch.sum(residual.reshape(-1).square()) / torch.sum(centered.square()).clamp_min(EPS)
        b1_gain = F.cross_entropy(base1, yb1) - F.cross_entropy(after1, yb1)
        b2_gain = F.cross_entropy(base2, yb2) - F.cross_entropy(after2, yb2)
        b3_gain = F.cross_entropy(base3, yb3) - F.cross_entropy(after3, yb3)

    # Restore gradients exactly enough for caller-side training.
    model.zero_grad(set_to_none=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p, grad in zip(trainable, saved_grads):
        p.grad = None if grad is None else grad.to(device=p.device, dtype=p.dtype)

    jvp = finite_difference_jvp(model, torch.cat([xb1, xb2], dim=0), -update_vec.detach(), eps=1.0e-3)
    energies = output_metric_energies(model, x, update_vec)
    channel = basis_channel_energy(model, update_vec)
    target_energy = {
        "target_norm_L2": float(torch.linalg.vector_norm(target.detach().float()).item()),
        "target_norm_Fisher": float((fisher_diag_from_logits(torch.cat([base1, base2], dim=0)) * target.detach().float().square()).mean().item()),
        "target_Sobolev_H1_energy": float(sobolev_fd_energy(target).item()),
        "target_RKHS_energy": float(rkhs_graph_energy(target).item()),
        "target_NDS_proxy": float(sobolev_fd_energy(target).item() / l2_energy(target).clamp_min(EPS).item()),
    }
    residual_norm = torch.linalg.vector_norm(residual.detach().float())
    target_norm = torch.linalg.vector_norm(target.detach().float()).clamp_min(EPS)
    diagnostics: dict[str, Any] = {
        "solver_status": "metric_readout_exact_solve",
        "is_proxy": 0,
        "mechanism": mechanism,
        "target_family": target_family,
        "metric_family": metric_family,
        "solver_level": (
            "S2-ReadoutExactC3GatedHiddenResidualBlock"
            if target_family == "T9-C3GatedBlockSourceChannel"
            else
            "S2-ReadoutExactSoftCompensatedHiddenResidualBlock"
            if target_family == "T12-SoftCompensatedBlockSourceChannel"
            else
            "S2-ReadoutExactCompensatedHiddenResidualBlock"
            if target_family in {"T10-CompensatedBlockSourceChannel", "T11-EarlyObservableSourceChannel"}
            else
            "S2-ReadoutExactAdaptiveHiddenResidualBlock"
            if target_family == "T8-AdaptiveBlockSourceChannel"
            else
            "S2-ReadoutExactHiddenResidualBlock"
            if target_family == "T7-BlockSourceChannel"
            else "S1-ReadoutExactSolve"
        ),
        "solver_type": (
            "readout_exact_plus_c3_gated_train_gradient_hidden_block"
            if target_family == "T9-C3GatedBlockSourceChannel"
            else
            "readout_exact_plus_soft_compensated_train_gradient_hidden_block"
            if target_family == "T12-SoftCompensatedBlockSourceChannel"
            else
            "readout_exact_plus_compensated_train_gradient_hidden_block"
            if target_family in {"T10-CompensatedBlockSourceChannel", "T11-EarlyObservableSourceChannel"}
            else
            "readout_exact_plus_adaptive_train_gradient_hidden_block"
            if target_family == "T8-AdaptiveBlockSourceChannel"
            else
            "readout_exact_plus_train_gradient_hidden_block"
            if target_family == "T7-BlockSourceChannel"
            else "readout_exact_solve"
        ),
        "commit_type": "direct_parameter_commit",
        "target_scale": float(target_scale),
        "block_role": block_role_normalized,
        "block_restricted_solver": int(block_role_normalized != "all"),
        "block_restricted_solver_status": "block_mask_applied" if block_role_normalized != "all" else "not_restricted",
        "hidden_residual_is_exact_solve": 0 if target_family in hidden_target_families else "",
        "block_source_hidden_residual_norm": hidden_residual_norm,
        "block_source_hidden_residual_fraction": hidden_residual_fraction,
        "block_source_hidden_function_norm": hidden_function_norm,
        "block_source_hidden_function_cap_ratio": hidden_function_cap_ratio,
        "block_source_hidden_compensation_norm": hidden_compensation_norm,
        "block_source_hidden_compensation_scale": hidden_compensation_scale,
        "block_source_hidden_compensation_mix": hidden_compensation_mix,
        "block_source_hidden_compensation_residual_ratio": hidden_compensation_residual_ratio,
        "block_source_hidden_compensation_b2_gain": hidden_compensation_b2_gain,
        "projection_residual_Gf": float(residual_norm.item() / target_norm.item()),
        "ActuationR2": float(r2.item()),
        "ActuationCosine": function_displacement_cosine(actual, target),
        "function_displacement_norm": float(torch.linalg.vector_norm(actual.detach().float()).item()),
        "parameter_update_norm": float(torch.linalg.vector_norm(update_vec.detach().float()).item()),
        "solve_time_ms": (time.perf_counter() - start) * 1000.0,
        "JVP_count": 1,
        "VJP_count": 0,
        "CG_iterations": 0,
        "solver_rank": int(delta_w.numel()),
        "condition_estimate": float(torch.linalg.cond(gram.detach().float()).item()) if gram.numel() else "",
        "B1_gain": float(b1_gain.item()),
        "B2_transfer_gain": float(b2_gain.item()),
        "B3_safety_gain": float(b3_gain.item()),
        "B2_transfer_after_actuation": float(b2_gain.item()),
        "random_target_actuation_gap": "",
        "function_displacement_fd_cosine": function_displacement_cosine(jvp, actual),
        "uses_future_or_validation": 0,
        "uses_audit_metric_for_direction": 0,
        **d1,
        **{f"b2_{k}": v for k, v in d2.items() if k != "target_family"},
        **energies,
        **channel,
        **target_energy,
    }
    return UpdateTensor(
        update_vec,
        "metric_solver_readout_exact",
        "subtract",
        "function_metric_solver",
        f"v22_06_metric_solver_{target_family}_{metric_family}",
        mechanism,
        role="metric_as_geometry_readout_solver",
        one_step_descent_claim=0,
        diagnostics=diagnostics,
    )


def metric_solver_unit_tests() -> list[dict[str, Any]]:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(5, 6)
            self.w2 = torch.nn.Parameter(torch.randn(6, 3) * 0.02)

        def frozen_readout_features(self, xb: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.fc1(xb))

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            return self.frozen_readout_features(xb) @ self.w2

    torch.manual_seed(2206)
    model = Tiny()
    x = torch.randn(12, 5)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    rows: list[dict[str, Any]] = []
    for mechanism, cfg in V2206_SOLVER_CONFIGS.items():
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).float(), y).backward()
        update = solve_metric_readout_update(
            model,
            x,
            y,
            mechanism=mechanism,
            target_family=cfg["target_family"],
            metric_family=cfg["metric_family"],
            seed=2206,
        )
        diag = update.diagnostics or {}
        finite = all(isfinite(float(v)) for v in diag.values() if isinstance(v, (float, int)))
        rows.append(
            {
                "case": mechanism,
                "solver_status": diag.get("solver_status", ""),
                "ActuationR2": diag.get("ActuationR2", ""),
                "projection_residual_Gf": diag.get("projection_residual_Gf", ""),
                "solve_time_ms": diag.get("solve_time_ms", ""),
                "diagnostics_finite": int(finite),
                "pass": int(str(diag.get("solver_status")) == "metric_readout_exact_solve" and finite and float(diag.get("parameter_update_norm", 0.0)) > 0.0),
            }
        )
    return rows


__all__ = ["V2206_SOLVER_CONFIGS", "metric_solver_unit_tests", "solve_metric_readout_update"]
