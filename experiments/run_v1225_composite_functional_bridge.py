#!/usr/bin/env python
"""v12.25 precommit composite functional bridge.

This runner turns the v12.24 near-pass observations into explicit composite
events.  It only uses train-stream references for direction construction.  CE,
labels, validation/test metrics, and LineC hard targets are audit-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp
import experiments.run_v1223_p4_compensation_modes as p4m
import experiments.run_v1223_p4_trainable_role_scan as role_scan
import experiments.run_v1224_i30_i32_policy_bridge as policy_bridge
import experiments.run_v1224_train_stream_functional_bridge as bridge


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def tail_proxy(model: torch.nn.Module, refs: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
    drift_p99: list[float] = []
    max_abs_p99: list[float] = []
    confidence_p99: list[float] = []
    entropy_p01: list[float] = []
    for x_ref, base_logits in refs:
        with torch.no_grad():
            logits = model(x_ref).detach().float()
            drift = (logits - base_logits.detach().float()).abs().flatten()
            probs = torch.softmax(logits, dim=-1).clamp_min(1.0e-8)
            conf = probs.max(dim=-1).values.float()
            entropy = (-(probs * probs.log()).sum(dim=-1)).float()
            max_abs = logits.abs().flatten()
        drift_p99.append(float(torch.quantile(drift, 0.99).item()))
        max_abs_p99.append(float(torch.quantile(max_abs, 0.99).item()))
        confidence_p99.append(float(torch.quantile(conf, 0.99).item()))
        entropy_p01.append(float(torch.quantile(entropy, 0.01).item()))
    return {
        "logit_drift_p99": max(drift_p99) if drift_p99 else float("nan"),
        "max_logit_abs_p99": max(max_abs_p99) if max_abs_p99 else float("nan"),
        "confidence_p99": max(confidence_p99) if confidence_p99 else float("nan"),
        "entropy_p01": min(entropy_p01) if entropy_p01 else float("nan"),
    }


def score_tail(proxy: dict[str, float]) -> float:
    return (
        fnum(proxy.get("logit_drift_p99"), 0.0)
        + 0.02 * fnum(proxy.get("max_logit_abs_p99"), 0.0)
        + 0.50 * fnum(proxy.get("confidence_p99"), 0.0)
        - 0.02 * fnum(proxy.get("entropy_p01"), 0.0)
    )


class LogitScaledModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, scale: float):
        super().__init__()
        self.model = model
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x) * self.scale


def train_stream_abs_tail(model: torch.nn.Module, refs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    vals: list[float] = []
    model.eval()
    with torch.no_grad():
        for x_ref, _base_logits in refs:
            vals.append(float(torch.quantile(model(x_ref).detach().float().abs().flatten(), 0.99).item()))
    return max(vals) if vals else float("nan")


def post_train_logit_calibrate(model: torch.nn.Module, refs: list[tuple[torch.Tensor, torch.Tensor]], target_tail: float) -> tuple[torch.nn.Module, dict[str, Any]]:
    pre_tail = train_stream_abs_tail(model, refs)
    if not math.isfinite(pre_tail) or pre_tail <= 0.0 or not math.isfinite(target_tail) or target_tail <= 0.0:
        return model, {"post_logit_calibration_applied": 0, "post_logit_scale": "", "post_logit_pre_tail": pre_tail, "post_logit_target_tail": target_tail}
    scale = min(1.0, float(target_tail) / float(pre_tail))
    if scale >= 0.999999:
        return model, {"post_logit_calibration_applied": 0, "post_logit_scale": 1.0, "post_logit_pre_tail": pre_tail, "post_logit_target_tail": target_tail}
    return LogitScaledModel(model, scale), {"post_logit_calibration_applied": 1, "post_logit_scale": scale, "post_logit_pre_tail": pre_tail, "post_logit_target_tail": target_tail}


def build_reference_stream(
    base: torch.nn.Module,
    x_train: torch.Tensor,
    batch: int,
    count: int,
    mode: str,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    mode = str(mode)
    if mode != "bootstrap_mom":
        return bridge.build_train_refs(base, x_train, int(batch), int(count))
    refs: list[tuple[torch.Tensor, torch.Tensor]] = []
    n = int(x_train.shape[0])
    batch = max(1, min(int(batch), n))
    gen = torch.Generator(device=x_train.device).manual_seed(int(seed))
    base.eval()
    with torch.no_grad():
        for _ in range(int(count)):
            idx = torch.randint(0, n, (batch,), device=x_train.device, generator=gen)
            x_ref = x_train[idx]
            refs.append((x_ref, base(x_ref).detach().float()))
    return refs


def interpolate_model_state(model: torch.nn.Module, init_state: dict[str, torch.Tensor], alpha: float) -> int:
    alpha = float(alpha)
    if alpha >= 0.999999:
        return 0
    current = model.state_dict()
    blended: dict[str, torch.Tensor] = {}
    for key, value in current.items():
        base = init_state.get(key)
        if base is not None and torch.is_floating_point(value) and tuple(base.shape) == tuple(value.shape):
            blended[key] = value * alpha + base.to(device=value.device, dtype=value.dtype) * (1.0 - alpha)
        else:
            blended[key] = value
    model.load_state_dict(blended, strict=False)
    return 1


def response_delta_flat(model: torch.nn.Module, refs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for x_ref, base_logits in refs:
            logits = model(x_ref).detach().float()
            parts.append((logits - base_logits.detach().float()).flatten())
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def apply_response_residualization(
    source: torch.nn.Module,
    base: torch.nn.Module,
    control: torch.nn.Module,
    refs: list[tuple[torch.Tensor, torch.Tensor]],
    alpha_clip: float,
) -> dict[str, float]:
    """Subtract the matched-control response component using train-stream logits only."""
    ds_pre = response_delta_flat(source, refs)
    dc = response_delta_flat(control, refs)
    denom = float(torch.dot(dc, dc).item()) if dc.numel() else 0.0
    alpha = float(torch.dot(ds_pre, dc).item() / (denom + 1.0e-12)) if denom > 0 else 0.0
    alpha_clipped = max(-float(alpha_clip), min(float(alpha_clip), alpha))
    base_params = dict(base.named_parameters())
    control_params = dict(control.named_parameters())
    with torch.no_grad():
        for name, param in source.named_parameters():
            if name in base_params and name in control_params:
                param.add_(control_params[name].detach() - base_params[name].detach(), alpha=-alpha_clipped)
    ds_post = response_delta_flat(source, refs)
    pre_cos = float(torch.dot(ds_pre, dc).item() / ((float(torch.linalg.norm(ds_pre).item()) * float(torch.linalg.norm(dc).item())) + 1.0e-12)) if ds_pre.numel() and dc.numel() else 0.0
    post_cos = float(torch.dot(ds_post, dc).item() / ((float(torch.linalg.norm(ds_post).item()) * float(torch.linalg.norm(dc).item())) + 1.0e-12)) if ds_post.numel() and dc.numel() else 0.0
    return {
        "response_residual_alpha": alpha,
        "response_residual_alpha_clipped": alpha_clipped,
        "response_residual_pre_cosine": pre_cos,
        "response_residual_post_cosine": post_cos,
        "response_residual_source_norm_pre": float(torch.linalg.norm(ds_pre).item()) if ds_pre.numel() else 0.0,
        "response_residual_source_norm_post": float(torch.linalg.norm(ds_post).item()) if ds_post.numel() else 0.0,
        "response_residual_control_norm": float(torch.linalg.norm(dc).item()) if dc.numel() else 0.0,
    }


def apply_train_feature_functional_actuator(
    model: torch.nn.Module,
    actuator: str,
    budget: float,
    sign: float,
    refs: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    """Apply a label-free train-feature functional delta to direct_readout."""
    direct = getattr(model, "direct_readout", None)
    if not torch.is_tensor(direct) or not hasattr(model, "_direct_logits_and_features"):
        return {"train_feature_actuator_applied": 0, "train_feature_actuator_norm": "", "train_feature_actuator_mode": actuator}
    deltas: list[torch.Tensor] = []
    base_tail_values: list[float] = []
    stable_actuators = {
        "I36-TrainFeatureStableCenteredDenoise",
        "I37-TrainFeatureStableLowTailDirect",
        "I38-TrainFeatureStableConfidenceDamping",
    }
    lowrank_coupling_actuators = {
        "I39-TrainFeatureLowRankCouplingLift",
        "I40-TrainFeatureEntropyWeightedLowRankCoupling",
        "I41-TrainFeatureTailClippedLowRankCoupling",
    }
    crossref_coupling_actuators = {
        "I42-TrainFeatureCrossRefCouplingLift",
        "I43-TrainFeatureEntropyWeightedCrossRefCoupling",
        "I44-TrainFeatureTailClippedCrossRefCoupling",
    }
    if actuator in crossref_coupling_actuators:
        ref_features: list[torch.Tensor] = []
        ref_logits: list[torch.Tensor] = []
        ref_weights: list[torch.Tensor] = []
        with torch.no_grad():
            for x_ref, _base_logits in refs:
                logits = model(x_ref).detach().float()
                base_tail_values.append(float(torch.quantile(logits.abs().flatten(), 0.99).item()))
                _direct_logits, feats = model._direct_logits_and_features(x_ref)  # type: ignore[attr-defined]
                probs = torch.softmax(logits, dim=-1).clamp_min(1.0e-8)
                entropy = (-(probs * probs.log()).sum(dim=-1)).float() / math.log(max(2, probs.shape[-1]))
                uncertainty = (1.0 - probs.max(dim=-1).values).clamp(0.0, 1.0)
                ref_features.append(feats.detach().float().mean(dim=0))
                ref_logits.append((logits - logits.mean(dim=-1, keepdim=True)).mean(dim=0))
                ref_weights.append((entropy * uncertainty).mean().view(1))
            if len(ref_features) < 2 or len(ref_logits) < 2:
                return {"train_feature_actuator_applied": 0, "train_feature_actuator_norm": "", "train_feature_actuator_mode": actuator}
            f_ref = torch.stack(ref_features, dim=0)
            z_ref = torch.stack(ref_logits, dim=0)
            weights = torch.stack(ref_weights, dim=0)
            if actuator != "I43-TrainFeatureEntropyWeightedCrossRefCoupling":
                weights = torch.ones_like(weights)
            weights = weights / weights.mean().clamp_min(1.0e-6)
            f_center = (f_ref - f_ref.mean(dim=0, keepdim=True)) * weights
            z_center = (z_ref - z_ref.mean(dim=0, keepdim=True)) * weights
            f_center = f_center / f_center.std(dim=0, keepdim=True).clamp_min(1.0e-4)
            delta = f_center.transpose(0, 1) @ z_center / max(1, int(f_center.shape[0] - 1))
            if tuple(delta.shape) != tuple(direct.shape):
                return {"train_feature_actuator_applied": 0, "train_feature_actuator_norm": "", "train_feature_actuator_mode": actuator}
            delta = delta - delta.mean(dim=0, keepdim=True)
            norm = delta.float().norm().clamp_min(1.0e-12)
            unit = delta.to(device=direct.device, dtype=direct.dtype) / norm.to(device=direct.device, dtype=direct.dtype)
            requested_scale = float(sign * budget)
            applied_scale = requested_scale
            tail_target = max(base_tail_values or [0.0]) + 0.02
            if actuator == "I44-TrainFeatureTailClippedCrossRefCoupling":
                base_direct = direct.detach().clone()

                def probe_tail(scale: float) -> float:
                    direct.copy_(base_direct)
                    direct.add_(unit, alpha=float(scale))
                    vals = []
                    for x_ref, _base_logits in refs:
                        vals.append(float(torch.quantile(model(x_ref).detach().float().abs().flatten(), 0.99).item()))
                    return max(vals) if vals else float("inf")

                lo = 0.0
                hi = abs(requested_scale)
                best = 0.0
                signed = 1.0 if requested_scale >= 0 else -1.0
                for _ in range(10):
                    mid = (lo + hi) / 2.0
                    if probe_tail(signed * mid) <= tail_target:
                        best = mid
                        lo = mid
                    else:
                        hi = mid
                applied_scale = signed * best
                direct.copy_(base_direct)
            direct.add_(unit, alpha=applied_scale)
        return {
            "train_feature_actuator_applied": 1,
            "train_feature_actuator_norm": float(norm.item()),
            "train_feature_actuator_mode": actuator,
            "train_feature_requested_scale": requested_scale,
            "train_feature_applied_scale": applied_scale,
            "train_feature_tail_target": tail_target,
        }
    if actuator in lowrank_coupling_actuators:
        feats_all: list[torch.Tensor] = []
        logits_all: list[torch.Tensor] = []
        with torch.no_grad():
            for x_ref, _base_logits in refs:
                logits = model(x_ref).detach().float()
                base_tail_values.append(float(torch.quantile(logits.abs().flatten(), 0.99).item()))
                _direct_logits, feats = model._direct_logits_and_features(x_ref)  # type: ignore[attr-defined]
                feats_all.append(feats.detach().float())
                logits_all.append(logits)
            if not feats_all or not logits_all:
                return {"train_feature_actuator_applied": 0, "train_feature_actuator_norm": "", "train_feature_actuator_mode": actuator}
            f = torch.cat(feats_all, dim=0)
            logits = torch.cat(logits_all, dim=0)
            z = logits - logits.mean(dim=-1, keepdim=True)
            probs = torch.softmax(logits, dim=-1).clamp_min(1.0e-8)
            entropy = (-(probs * probs.log()).sum(dim=-1, keepdim=True)).float()
            entropy_scale = entropy / math.log(max(2, probs.shape[-1]))
            confidence_gap = (1.0 - probs.max(dim=-1, keepdim=True).values).clamp(0.0, 1.0)
            weights = entropy_scale * confidence_gap if actuator == "I40-TrainFeatureEntropyWeightedLowRankCoupling" else torch.ones_like(entropy_scale)
            f_center = (f - f.mean(dim=0, keepdim=True)) * weights
            z_center = z * weights
            try:
                _u_f, _s_f, vh_f = torch.linalg.svd(f_center, full_matrices=False)
                feature_vec = vh_f[0]
            except RuntimeError:
                feature_vec = f_center.mean(dim=0)
            class_vec = z_center.mean(dim=0)
            if float(class_vec.norm().item()) <= 1.0e-8:
                try:
                    _u_z, _s_z, vh_z = torch.linalg.svd(z_center, full_matrices=False)
                    class_vec = vh_z[0]
                except RuntimeError:
                    class_vec = z.mean(dim=0)
            delta = feature_vec[:, None] @ class_vec[None, :]
            if tuple(delta.shape) != tuple(direct.shape):
                return {"train_feature_actuator_applied": 0, "train_feature_actuator_norm": "", "train_feature_actuator_mode": actuator}
            delta = delta - delta.mean(dim=0, keepdim=True)
            norm = delta.float().norm().clamp_min(1.0e-12)
            unit = delta.to(device=direct.device, dtype=direct.dtype) / norm.to(device=direct.device, dtype=direct.dtype)
            requested_scale = float(sign * budget)
            applied_scale = requested_scale
            tail_target = max(base_tail_values or [0.0]) + 0.02
            if actuator == "I41-TrainFeatureTailClippedLowRankCoupling":
                base_direct = direct.detach().clone()

                def probe_tail(scale: float) -> float:
                    direct.copy_(base_direct)
                    direct.add_(unit, alpha=float(scale))
                    vals = []
                    for x_ref, _base_logits in refs:
                        vals.append(float(torch.quantile(model(x_ref).detach().float().abs().flatten(), 0.99).item()))
                    return max(vals) if vals else float("inf")

                lo = 0.0
                hi = abs(requested_scale)
                best = 0.0
                signed = 1.0 if requested_scale >= 0 else -1.0
                for _ in range(10):
                    mid = (lo + hi) / 2.0
                    if probe_tail(signed * mid) <= tail_target:
                        best = mid
                        lo = mid
                    else:
                        hi = mid
                applied_scale = signed * best
                direct.copy_(base_direct)
            direct.add_(unit, alpha=applied_scale)
        return {
            "train_feature_actuator_applied": 1,
            "train_feature_actuator_norm": float(norm.item()),
            "train_feature_actuator_mode": actuator,
            "train_feature_requested_scale": requested_scale,
            "train_feature_applied_scale": applied_scale,
            "train_feature_tail_target": tail_target,
        }
    with torch.no_grad():
        for x_ref, _base_logits in refs:
            logits = model(x_ref).detach().float()
            base_tail_values.append(float(torch.quantile(logits.abs().flatten(), 0.99).item()))
            _direct_logits, feats = model._direct_logits_and_features(x_ref)  # type: ignore[attr-defined]
            f = feats.detach().float()
            if actuator in stable_actuators:
                f = (f - f.mean(dim=0, keepdim=True)) / f.std(dim=0, keepdim=True).clamp_min(1.0e-4)
            z = logits - logits.mean(dim=-1, keepdim=True)
            probs = torch.softmax(logits, dim=-1).clamp_min(1.0e-8)
            uniform = torch.full_like(probs, 1.0 / max(1, probs.shape[-1]))
            entropy = (-(probs * probs.log()).sum(dim=-1, keepdim=True)).float()
            entropy_scale = entropy / math.log(max(2, probs.shape[-1]))
            margin = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
            gap = (margin[:, :1] - margin[:, 1:2]) if margin.shape[-1] >= 2 else torch.zeros_like(entropy_scale)
            if actuator == "I28-TrainFeatureCovarianceDirectLift":
                target = z
            elif actuator == "I29-TrainFeatureEntropyDampedDirectLift":
                target = z * entropy_scale
            elif actuator == "I30-TrainFeatureConfidenceDampingDirect":
                target = -(probs - uniform)
            elif actuator == "I31-TrainFeatureMarginGuardedDirect":
                target = z * (1.0 - gap.clamp(0.0, 1.0))
            elif actuator in {"I32-TrainFeatureCenteredLogitDenoise", "I34-TrainFeatureCenteredDenoiseTailClipped"}:
                target = z.clamp(min=-2.0, max=2.0)
            elif actuator in {"I33-TrainFeatureLowTailDirectLift", "I35-TrainFeatureLowTailDirectTailClipped"}:
                target = z * entropy_scale * (1.0 - probs.max(dim=-1, keepdim=True).values).clamp(0.0, 1.0)
            elif actuator == "I36-TrainFeatureStableCenteredDenoise":
                target = z.clamp(min=-1.5, max=1.5)
            elif actuator == "I37-TrainFeatureStableLowTailDirect":
                target = z * entropy_scale * (1.0 - probs.max(dim=-1, keepdim=True).values).clamp(0.0, 1.0)
            elif actuator == "I38-TrainFeatureStableConfidenceDamping":
                target = -(probs - uniform) * entropy_scale
            else:
                return {"train_feature_actuator_applied": 0, "train_feature_actuator_norm": "", "train_feature_actuator_mode": actuator}
            scale = math.sqrt(max(1, int(getattr(model, "input_dim", f.shape[1]))))
            delta = (f.transpose(0, 1) @ target) * float(scale) / max(1, int(f.shape[0]))
            if tuple(delta.shape) == tuple(direct.shape):
                deltas.append(delta.detach())
        if not deltas:
            return {"train_feature_actuator_applied": 0, "train_feature_actuator_norm": "", "train_feature_actuator_mode": actuator}
        stack = torch.stack(deltas, dim=0).float()
        if actuator in stable_actuators:
            sign_consensus = stack.sign().mean(dim=0)
            median_abs = stack.abs().median(dim=0).values
            delta = sign_consensus.sign() * median_abs * sign_consensus.abs().clamp(0.0, 1.0)
        else:
            delta = stack.mean(dim=0)
        delta = delta - delta.mean(dim=0, keepdim=True)
        norm = delta.float().norm().clamp_min(1.0e-12)
        unit = delta.to(device=direct.device, dtype=direct.dtype) / norm.to(device=direct.device, dtype=direct.dtype)
        requested_scale = float(sign * budget)
        applied_scale = requested_scale
        tail_target = max(base_tail_values or [0.0]) + 0.02
        if "TailClipped" in actuator:
            base_direct = direct.detach().clone()

            def probe_tail(scale: float) -> float:
                direct.copy_(base_direct)
                direct.add_(unit, alpha=float(scale))
                vals = []
                for x_ref, _base_logits in refs:
                    vals.append(float(torch.quantile(model(x_ref).detach().float().abs().flatten(), 0.99).item()))
                return max(vals) if vals else float("inf")

            lo = 0.0
            hi = abs(requested_scale)
            best = 0.0
            signed = 1.0 if requested_scale >= 0 else -1.0
            for _ in range(10):
                mid = (lo + hi) / 2.0
                if probe_tail(signed * mid) <= tail_target:
                    best = mid
                    lo = mid
                else:
                    hi = mid
            applied_scale = signed * best
            direct.copy_(base_direct)
        direct.add_(unit, alpha=applied_scale)
    return {
        "train_feature_actuator_applied": 1,
        "train_feature_actuator_norm": float(norm.item()),
        "train_feature_actuator_mode": actuator,
        "train_feature_requested_scale": requested_scale,
        "train_feature_applied_scale": applied_scale,
        "train_feature_tail_target": tail_target,
    }


def build_trial(
    base: torch.nn.Module,
    actuator: str,
    budget: float,
    sign: float,
    comp_mode: str,
    refs: list[tuple[torch.Tensor, torch.Tensor]],
    opt_delta: dict[str, torch.Tensor],
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    trial = copy.deepcopy(base).to(device)
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(budget) * 100000) + len(actuator) * 17 + (1 if sign > 0 else 2))
    actuator_parts: list[tuple[str, float]] = []
    for raw_part in str(actuator).split("+"):
        part = raw_part.strip()
        if not part:
            continue
        if "@" in part:
            name, scale = part.rsplit("@", 1)
            actuator_parts.append((name.strip(), float(scale)))
        else:
            actuator_parts.append((part, 1.0))
    role_parts: list[str] = []
    feature_infos: list[dict[str, Any]] = []
    for part_actuator, part_scale in actuator_parts:
        if "TrainFeature" in part_actuator:
            feature_info = apply_train_feature_functional_actuator(trial, part_actuator, float(budget) * float(part_scale), float(sign), refs)
            feature_infos.append(feature_info)
            role_parts.append(str(feature_info.get("train_feature_actuator_mode", part_actuator)))
        else:
            role_parts.append(exp.apply_v1223_actuator(trial, part_actuator, float(budget) * float(part_scale), float(sign), gen, opt_delta))
    role = "+".join(role_parts)
    comp = {
        "direct_logit_compensation_applied": 0,
        "direct_logit_compensation_mode": "",
        "direct_logit_compensation_pre_drift": "",
        "direct_logit_compensation_post_drift": "",
        "direct_logit_compensation_norm": "",
    }
    needs_compensation = any(
        part_actuator in {"I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl"}
        for part_actuator, _part_scale in actuator_parts
    )
    if needs_compensation:
        comp = bridge.apply_train_stream_compensation(trial, refs, comp_mode)
    if feature_infos:
        comp["train_feature_actuator_applied"] = int(any(exp.safe_int(info.get("train_feature_actuator_applied"), 0) for info in feature_infos))
        comp["train_feature_actuator_modes"] = "+".join(str(info.get("train_feature_actuator_mode", "")) for info in feature_infos)
        norms = [fnum(info.get("train_feature_actuator_norm")) for info in feature_infos]
        comp["train_feature_actuator_norm"] = max([x for x in norms if math.isfinite(x)] or [float("nan")])
        requested = [fnum(info.get("train_feature_requested_scale")) for info in feature_infos]
        applied = [fnum(info.get("train_feature_applied_scale")) for info in feature_infos]
        targets = [fnum(info.get("train_feature_tail_target")) for info in feature_infos]
        comp["train_feature_requested_scale"] = max([abs(x) for x in requested if math.isfinite(x)] or [float("nan")])
        comp["train_feature_applied_scale"] = max([abs(x) for x in applied if math.isfinite(x)] or [float("nan")])
        comp["train_feature_tail_target"] = max([x for x in targets if math.isfinite(x)] or [float("nan")])
    else:
        comp["train_feature_actuator_applied"] = 0
        comp["train_feature_actuator_modes"] = ""
        comp["train_feature_actuator_norm"] = ""
        comp["train_feature_requested_scale"] = ""
        comp["train_feature_applied_scale"] = ""
        comp["train_feature_tail_target"] = ""
    return trial, role, comp


CANDIDATES: dict[str, dict[str, Any]] = {
    "F25-C1-directGainQuadMix-I26": {
        "component_ids": "NG17-direct_gain+NG9-quad_direct",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "control_residualized": 0,
        "tail_budgeted": 0,
    },
    "F25-C1-quadDirectMix-I26": {
        "component_ids": "NG4-I30+NG9-quad_direct",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 0,
        "tail_budgeted": 0,
    },
    "F25-C2-controlResidual-I27": {
        "component_ids": "NG10-control-contrast",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_gain",
        "lambda_direct": 1.00,
        "lambda_quad": 0.00,
        "control_residualized": 1,
        "tail_budgeted": 0,
    },
    "F25-C3-tailBudgetedQuadDirect-I26": {
        "component_ids": "NG4-tail-budgeted",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "quad_direct",
        "lambda_direct": 0.25,
        "lambda_quad": 0.75,
        "control_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-C4-policyAutoDirectOnly-I26": {
        "component_ids": "NG18-direct_only",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_only",
        "lambda_direct": 1.00,
        "lambda_quad": 0.00,
        "control_residualized": 0,
        "tail_budgeted": 0,
    },
    "F25-D3-controlResidualQuadDirect-I27": {
        "component_ids": "Depth3-control-residual+quad-reservoir",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-D3-tailBudgetedDirectBranch-I26": {
        "component_ids": "Depth3-tail-budgeted-direct-branch",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "control_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-D3-lowBudgetDirectOnly-I26": {
        "component_ids": "Depth3-low-budget-direct-only-counterfactual",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease",
        "budget_scale": 0.25,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_only",
        "lambda_direct": 1.00,
        "lambda_quad": 0.00,
        "control_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG19-structuralMerge-taskLineC-I26I27": {
        "component_ids": "NG25-structural-merge-task-control+linec-majority",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease@0.50+I27-TrainDirectLogitCompensatedShadowRelease@0.50",
        "budget_scale": 0.75,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG20-linecFirstMerge-I27I26": {
        "component_ids": "NG25-linec-first-then-task-merge",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.75+I26-TrainDirectLogitCompensatedQuadRelease@0.25",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG21-taskFirstMerge-I26I27-low": {
        "component_ids": "NG25-task-first-low-budget-linec-guard",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease@0.75+I27-TrainDirectLogitCompensatedShadowRelease@0.25",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_gain",
        "lambda_direct": 0.90,
        "lambda_quad": 0.10,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG22-balancedMerge-directOnlyLineCGuard": {
        "component_ids": "NG25-balanced-merge-direct-only-linec-guard",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease@0.35+I27-TrainDirectLogitCompensatedShadowRelease@0.65",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_only",
        "lambda_direct": 1.00,
        "lambda_quad": 0.00,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG23-linecDominantMerge-I27I26-9010": {
        "component_ids": "NG25-linec-dominant-merge-90-10",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.90+I26-TrainDirectLogitCompensatedQuadRelease@0.10",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "quad_direct",
        "lambda_direct": 0.35,
        "lambda_quad": 0.65,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG24-linecDominantMerge-I27I26-8515": {
        "component_ids": "NG25-linec-dominant-merge-85-15",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.85+I26-TrainDirectLogitCompensatedQuadRelease@0.15",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "quad_direct",
        "lambda_direct": 0.40,
        "lambda_quad": 0.60,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG25-lowBudgetLinecMerge-I27I26-9010": {
        "component_ids": "NG25-low-budget-linec-dominant-merge",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.90+I26-TrainDirectLogitCompensatedQuadRelease@0.10",
        "budget_scale": 0.75,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "quad_direct",
        "lambda_direct": 0.35,
        "lambda_quad": 0.65,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG26-directBranchLinecMerge-I27I26-9010": {
        "component_ids": "NG25-direct-branch-linec-dominant-merge",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.90+I26-TrainDirectLogitCompensatedQuadRelease@0.10",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.65,
        "lambda_quad": 0.35,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG27-directBranchLinecMerge-I27I26-8515": {
        "component_ids": "NG25-direct-branch-linec-merge-85-15",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.85+I26-TrainDirectLogitCompensatedQuadRelease@0.15",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.70,
        "lambda_quad": 0.30,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG28-directBranchLinecMerge-I27I26-9505": {
        "component_ids": "NG25-direct-branch-linec-merge-95-05",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.95+I26-TrainDirectLogitCompensatedQuadRelease@0.05",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.60,
        "lambda_quad": 0.40,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG29-directBranchLinecMerge-I27I26-9010-highBudget": {
        "component_ids": "NG25-direct-branch-linec-merge-90-10-high-budget",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.90+I26-TrainDirectLogitCompensatedQuadRelease@0.10",
        "budget_scale": 1.25,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.65,
        "lambda_quad": 0.35,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG30-directBranchLinecMerge-I27I26-9010-lowBudget": {
        "component_ids": "NG25-direct-branch-linec-merge-90-10-low-budget",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.90+I26-TrainDirectLogitCompensatedQuadRelease@0.10",
        "budget_scale": 0.85,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.65,
        "lambda_quad": 0.35,
        "control_residualized": 1,
        "tail_budgeted": 1,
    },
    "F25-NG31-trueResponseResidual-I27I26-8515": {
        "component_ids": "NG31-true-train-response-residualize-linec-task-merge",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.85+I26-TrainDirectLogitCompensatedQuadRelease@0.15",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.70,
        "lambda_quad": 0.30,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 1.00,
        "tail_budgeted": 1,
    },
    "F25-NG32-trueResponseResidual-I26-directGain": {
        "component_ids": "NG32-task-positive-direct-gain-response-residualized",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_gain",
        "lambda_direct": 0.90,
        "lambda_quad": 0.10,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 1.00,
        "tail_budgeted": 1,
    },
    "F25-NG33-halfResponseResidual-I27I26-8515": {
        "component_ids": "NG33-half-residualize-preserve-linec-component",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.85+I26-TrainDirectLogitCompensatedQuadRelease@0.15",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.70,
        "lambda_quad": 0.30,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 0.50,
        "tail_budgeted": 1,
    },
    "F25-NG34-responseResidual-highBudget-I27I26": {
        "component_ids": "NG34-high-budget-task-control-with-response-residualization",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.90+I26-TrainDirectLogitCompensatedQuadRelease@0.10",
        "budget_scale": 1.25,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.65,
        "lambda_quad": 0.35,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 0.75,
        "tail_budgeted": 1,
    },
    "F25-NG35-responseResidual-lowBudget-I27I26": {
        "component_ids": "NG35-low-budget-tail-safe-response-residualization",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.90+I26-TrainDirectLogitCompensatedQuadRelease@0.10",
        "budget_scale": 0.75,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "quad_direct",
        "lambda_direct": 0.35,
        "lambda_quad": 0.65,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 1.00,
        "tail_budgeted": 1,
    },
    "F25-NG36-responseResidual-directOnly-I26": {
        "component_ids": "NG36-direct-only-task-signal-response-residual-counterfactual",
        "actuator": "I26-TrainDirectLogitCompensatedQuadRelease",
        "budget_scale": 0.25,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_only",
        "lambda_direct": 1.00,
        "lambda_quad": 0.00,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 1.00,
        "tail_budgeted": 1,
    },
    "F25-NG37-quarterResponseResidual-I27I26-8515": {
        "component_ids": "NG37-partial-response-residual-quarter-preserve-linec",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.85+I26-TrainDirectLogitCompensatedQuadRelease@0.15",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.70,
        "lambda_quad": 0.30,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 0.25,
        "tail_budgeted": 1,
    },
    "F25-NG38-tenthResponseResidual-I27I26-8515": {
        "component_ids": "NG38-partial-response-residual-tenth-preserve-linec",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.85+I26-TrainDirectLogitCompensatedQuadRelease@0.15",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.70,
        "lambda_quad": 0.30,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 0.10,
        "tail_budgeted": 1,
    },
    "F25-NG39-quarterResponseResidual-highBudget-I27I26": {
        "component_ids": "NG39-partial-response-residual-high-budget",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.90+I26-TrainDirectLogitCompensatedQuadRelease@0.10",
        "budget_scale": 1.25,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.65,
        "lambda_quad": 0.35,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 0.25,
        "tail_budgeted": 1,
    },
    "F25-NG40-quarterResponseResidual-quadDirect-I27I26": {
        "component_ids": "NG40-partial-response-residual-quad-direct-linec-guard",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.90+I26-TrainDirectLogitCompensatedQuadRelease@0.10",
        "budget_scale": 0.85,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "quad_direct",
        "lambda_direct": 0.35,
        "lambda_quad": 0.65,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 0.25,
        "tail_budgeted": 1,
    },
    "F25-NG41-quarterResponseResidual-noTailBudget-I27I26": {
        "component_ids": "NG41-partial-response-residual-no-tail-budget-counterfactual",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.85+I26-TrainDirectLogitCompensatedQuadRelease@0.15",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.70,
        "lambda_quad": 0.30,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 0.25,
        "tail_budgeted": 0,
    },
    "F25-NG42-tenthResponseResidual-noTailBudget-I27I26": {
        "component_ids": "NG42-tenth-response-residual-no-tail-budget-counterfactual",
        "actuator": "I27-TrainDirectLogitCompensatedShadowRelease@0.85+I26-TrainDirectLogitCompensatedQuadRelease@0.15",
        "budget_scale": 1.00,
        "sign": -1.0,
        "comp_mode": "median",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.70,
        "lambda_quad": 0.30,
        "control_residualized": 1,
        "response_residualized": 1,
        "response_residual_alpha_clip": 0.10,
        "tail_budgeted": 0,
    },
    "F25-NG44-featureCovDirectLift": {
        "component_ids": "NG44-feature-covariance-direct-functional-primitive",
        "actuator": "I28-TrainFeatureCovarianceDirectLift",
        "budget_scale": 0.50,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "direct_only",
        "lambda_direct": 1.00,
        "lambda_quad": 0.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG45-featureEntropyDampedDirectLift": {
        "component_ids": "NG45-entropy-damped-feature-direct-functional-primitive",
        "actuator": "I29-TrainFeatureEntropyDampedDirectLift",
        "budget_scale": 0.50,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "direct_only",
        "lambda_direct": 1.00,
        "lambda_quad": 0.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG46-featureConfidenceDamping": {
        "component_ids": "NG46-confidence-damping-feature-functional-primitive",
        "actuator": "I30-TrainFeatureConfidenceDampingDirect",
        "budget_scale": 0.75,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG47-featureMarginGuardedDirect": {
        "component_ids": "NG47-margin-guarded-feature-direct-functional-primitive",
        "actuator": "I31-TrainFeatureMarginGuardedDirect",
        "budget_scale": 0.75,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "direct_gain",
        "lambda_direct": 0.90,
        "lambda_quad": 0.10,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG48-featureCenteredLogitDenoise": {
        "component_ids": "NG48-centered-logit-denoise-feature-functional-primitive",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG49-featureLowTailDirectLift": {
        "component_ids": "NG49-low-tail-feature-direct-functional-primitive",
        "actuator": "I33-TrainFeatureLowTailDirectLift",
        "budget_scale": 0.75,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG50-centeredDenoiseLowBudget": {
        "component_ids": "NG50-centered-logit-denoise-low-budget-tail-repair",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.25,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG51-centeredDenoiseMidBudget": {
        "component_ids": "NG51-centered-logit-denoise-mid-budget-tail-repair",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.35,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG52-centeredDenoiseDirectBranch": {
        "component_ids": "NG52-centered-logit-denoise-direct-branch-tail-repair",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.35,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG53-centeredDenoiseDirectOnly": {
        "component_ids": "NG53-centered-logit-denoise-direct-only-tail-repair",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.35,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "direct_only",
        "lambda_direct": 1.00,
        "lambda_quad": 0.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG54-centeredDenoiseOppositeSign": {
        "component_ids": "NG54-centered-logit-denoise-opposite-sign-counterfactual",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.35,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG55-centeredDenoiseLowBudgetDirectBranch": {
        "component_ids": "NG55-centered-logit-denoise-low-budget-direct-branch",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.25,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG58-centeredDenoiseTailClipped": {
        "component_ids": "NG58-centered-logit-denoise-tail-clipped",
        "actuator": "I34-TrainFeatureCenteredDenoiseTailClipped",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG59-centeredDenoiseTailClippedHigh": {
        "component_ids": "NG59-centered-logit-denoise-tail-clipped-high-budget",
        "actuator": "I34-TrainFeatureCenteredDenoiseTailClipped",
        "budget_scale": 0.75,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG60-centeredDenoiseTailClippedDirectBranch": {
        "component_ids": "NG60-centered-logit-denoise-tail-clipped-direct-branch",
        "actuator": "I34-TrainFeatureCenteredDenoiseTailClipped",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG61-lowTailDirectTailClipped": {
        "component_ids": "NG61-low-tail-direct-tail-clipped",
        "actuator": "I35-TrainFeatureLowTailDirectTailClipped",
        "budget_scale": 0.75,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG62-centeredDenoiseTailClippedOpposite": {
        "component_ids": "NG62-centered-logit-denoise-tail-clipped-opposite-sign",
        "actuator": "I34-TrainFeatureCenteredDenoiseTailClipped",
        "budget_scale": 0.50,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG63-centeredDenoiseTailClippedLow": {
        "component_ids": "NG63-centered-logit-denoise-tail-clipped-low-budget",
        "actuator": "I34-TrainFeatureCenteredDenoiseTailClipped",
        "budget_scale": 0.35,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG64-centeredDenoiseSmooth001": {
        "component_ids": "NG64-centered-logit-denoise-label-smoothing-001",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "label_smoothing": 0.01,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG65-centeredDenoiseSmooth002": {
        "component_ids": "NG65-centered-logit-denoise-label-smoothing-002",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "label_smoothing": 0.02,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG66-centeredDenoiseSmooth005": {
        "component_ids": "NG66-centered-logit-denoise-label-smoothing-005",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "label_smoothing": 0.05,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG67-centeredDenoiseOppositeSmooth001": {
        "component_ids": "NG67-centered-logit-denoise-opposite-label-smoothing-001",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.35,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "label_smoothing": 0.01,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG68-centeredDenoiseSmooth001DirectBranch": {
        "component_ids": "NG68-centered-logit-denoise-label-smoothing-001-direct-branch",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "label_smoothing": 0.01,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG69-centeredDenoiseHighSmooth001": {
        "component_ids": "NG69-centered-logit-denoise-high-budget-label-smoothing-001",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.75,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "label_smoothing": 0.01,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG70-centeredDenoisePostCalBaseTail": {
        "component_ids": "NG70-centered-logit-denoise-post-train-logit-calibration-base-tail",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG71-centeredDenoisePostCalTail150": {
        "component_ids": "NG71-centered-logit-denoise-post-train-logit-calibration-tail150",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG72-centeredDenoisePostCalTail200": {
        "component_ids": "NG72-centered-logit-denoise-post-train-logit-calibration-tail200",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 2.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG73-centeredDenoiseOppositePostCalTail150": {
        "component_ids": "NG73-centered-logit-denoise-opposite-post-calibration-tail150",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.35,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG74-tailClippedHighPostCalTail150": {
        "component_ids": "NG74-tail-clipped-high-post-calibration-tail150",
        "actuator": "I34-TrainFeatureCenteredDenoiseTailClipped",
        "budget_scale": 0.75,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG75-centeredDenoiseDirectBranchPostCalTail150": {
        "component_ids": "NG75-centered-logit-denoise-direct-branch-post-calibration-tail150",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG76-centeredDenoisePostCalTail110": {
        "component_ids": "NG76-centered-logit-denoise-post-calibration-tail110",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.10,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG77-centeredDenoisePostCalTail120": {
        "component_ids": "NG77-centered-logit-denoise-post-calibration-tail120",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.20,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG78-centeredDenoisePostCalTail130": {
        "component_ids": "NG78-centered-logit-denoise-post-calibration-tail130",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.30,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG79-centeredDenoisePostCalTail140": {
        "component_ids": "NG79-centered-logit-denoise-post-calibration-tail140",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.40,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
    },
    "F25-NG80-centeredDenoisePostCalTail120NoBudget": {
        "component_ids": "NG80-centered-logit-denoise-post-calibration-tail120-no-tail-budget",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.20,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 0,
    },
    "F25-NG81-centeredDenoisePostCalTail130NoBudget": {
        "component_ids": "NG81-centered-logit-denoise-post-calibration-tail130-no-tail-budget",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.30,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 0,
    },
    "F25-NG82-ng70SeedPostCalTail105": {
        "component_ids": "NG82-ng70-seed-salt-post-calibration-tail105",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.05,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG83-ng70SeedPostCalTail110": {
        "component_ids": "NG83-ng70-seed-salt-post-calibration-tail110",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.10,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG84-ng70SeedPostCalTail120": {
        "component_ids": "NG84-ng70-seed-salt-post-calibration-tail120",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.20,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG85-ng70SeedPostCalTail130": {
        "component_ids": "NG85-ng70-seed-salt-post-calibration-tail130",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.30,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG86-ng70SeedPostCalTail150": {
        "component_ids": "NG86-ng70-seed-salt-post-calibration-tail150",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG87-ng70SeedPostCalTail200": {
        "component_ids": "NG87-ng70-seed-salt-post-calibration-tail200",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 2.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG88-ng70SeedPostCalTail040": {
        "component_ids": "NG88-ng70-seed-salt-post-calibration-tail040",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 0.40,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG89-ng70SeedPostCalTail060": {
        "component_ids": "NG89-ng70-seed-salt-post-calibration-tail060",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 0.60,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG90-ng70SeedPostCalTail080": {
        "component_ids": "NG90-ng70-seed-salt-post-calibration-tail080",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 0.80,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG91-ng70SeedPostCalTail090": {
        "component_ids": "NG91-ng70-seed-salt-post-calibration-tail090",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 0.90,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG92-ng70SeedPostCalTail095": {
        "component_ids": "NG92-ng70-seed-salt-post-calibration-tail095",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 0.95,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG93-ng70SeedPostCalTail100": {
        "component_ids": "NG93-ng70-seed-salt-post-calibration-tail100-repeat",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG94-ng70SeedWeightAnchor050PostCal100": {
        "component_ids": "NG94-ng70-seed-weight-anchor050-post-calibration-tail100",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG95-ng70SeedWeightAnchor070PostCal100": {
        "component_ids": "NG95-ng70-seed-weight-anchor070-post-calibration-tail100",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.70,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG96-ng70SeedWeightAnchor085PostCal100": {
        "component_ids": "NG96-ng70-seed-weight-anchor085-post-calibration-tail100",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.85,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG97-ng70SeedWeightAnchor070PostCal150": {
        "component_ids": "NG97-ng70-seed-weight-anchor070-post-calibration-tail150",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.70,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG98-ng70SeedWeightAnchor085PostCal150": {
        "component_ids": "NG98-ng70-seed-weight-anchor085-post-calibration-tail150",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.85,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG99-ng70SeedWeightAnchor090PostCal200": {
        "component_ids": "NG99-ng70-seed-weight-anchor090-post-calibration-tail200",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.90,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 2.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG100-ng70SeedWeightAnchor045PostCal100": {
        "component_ids": "NG100-ng70-seed-weight-anchor045-post-calibration-tail100",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.45,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG101-ng70SeedWeightAnchor050PostCal090": {
        "component_ids": "NG101-ng70-seed-weight-anchor050-post-calibration-tail090",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 0.90,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG102-ng70SeedWeightAnchor050PostCal105": {
        "component_ids": "NG102-ng70-seed-weight-anchor050-post-calibration-tail105",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.05,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG103-ng70SeedWeightAnchor055PostCal100": {
        "component_ids": "NG103-ng70-seed-weight-anchor055-post-calibration-tail100",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG104-ng70SeedWeightAnchor055PostCal090": {
        "component_ids": "NG104-ng70-seed-weight-anchor055-post-calibration-tail090",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 0.90,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG105-ng70SeedWeightAnchor060PostCal100": {
        "component_ids": "NG105-ng70-seed-weight-anchor060-post-calibration-tail100",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG110-featureDenoiseLowTailMix-7030": {
        "component_ids": "NG110-centered-denoise-plus-low-tail-direct-feature-mix",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise@0.70+I33-TrainFeatureLowTailDirectLift@-0.30",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG111-featureDenoiseMarginMix-7030": {
        "component_ids": "NG111-centered-denoise-plus-margin-guarded-feature-mix",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise@0.70+I31-TrainFeatureMarginGuardedDirect@-0.30",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG112-featureDenoiseConfidenceMix-7030": {
        "component_ids": "NG112-centered-denoise-plus-confidence-damping-feature-mix",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise@0.70+I30-TrainFeatureConfidenceDampingDirect@-0.30",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG113-featureDenoiseLowTailMix-5050": {
        "component_ids": "NG113-centered-denoise-plus-low-tail-direct-balanced-feature-mix",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise@0.50+I33-TrainFeatureLowTailDirectLift@-0.50",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG114-tailClippedDenoiseLowTailMix-7030": {
        "component_ids": "NG114-tail-clipped-denoise-plus-low-tail-direct-feature-mix",
        "actuator": "I34-TrainFeatureCenteredDenoiseTailClipped@0.70+I35-TrainFeatureLowTailDirectTailClipped@-0.30",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG115-featureDenoiseEntropyMix-7030": {
        "component_ids": "NG115-centered-denoise-plus-entropy-damped-feature-mix",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise@0.70+I29-TrainFeatureEntropyDampedDirectLift@-0.30",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG118-ng70SeedWeightAnchor055PostCal060": {
        "component_ids": "NG118-ng70-seed-weight-anchor055-post-calibration-tail060",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 0.60,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG119-ng70SeedWeightAnchor055PostCal075": {
        "component_ids": "NG119-ng70-seed-weight-anchor055-post-calibration-tail075",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 0.75,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG120-ng70SeedWeightAnchor055PostCal110": {
        "component_ids": "NG120-ng70-seed-weight-anchor055-post-calibration-tail110",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.10,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG121-ng70SeedWeightAnchor055PostCal125": {
        "component_ids": "NG121-ng70-seed-weight-anchor055-post-calibration-tail125",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG122-ng70SeedWeightAnchor055PostCal150": {
        "component_ids": "NG122-ng70-seed-weight-anchor055-post-calibration-tail150",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.50,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG123-ng70SeedWeightAnchor055PostCal200": {
        "component_ids": "NG123-ng70-seed-weight-anchor055-post-calibration-tail200",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.55,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 2.00,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG124-ng70SeedWeightAnchor040PostCal125": {
        "component_ids": "NG124-ng70-seed-weight-anchor040-post-calibration-tail125",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.40,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG125-ng70SeedWeightAnchor045PostCal125": {
        "component_ids": "NG125-ng70-seed-weight-anchor045-post-calibration-tail125",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.45,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG126-ng70SeedWeightAnchor050PostCal125": {
        "component_ids": "NG126-ng70-seed-weight-anchor050-post-calibration-tail125",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.50,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG127-ng70SeedWeightAnchor060PostCal125": {
        "component_ids": "NG127-ng70-seed-weight-anchor060-post-calibration-tail125",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG128-ng70SeedWeightAnchor065PostCal125": {
        "component_ids": "NG128-ng70-seed-weight-anchor065-post-calibration-tail125",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.65,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG129-ng70SeedWeightAnchor070PostCal125": {
        "component_ids": "NG129-ng70-seed-weight-anchor070-post-calibration-tail125",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.70,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG131-ng70SeedWeightAnchor060PostCal125FreezeDirect": {
        "component_ids": "NG131-ng70-alpha060-postcal125-freeze-direct-geometry-preserve",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "freeze_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG132-ng70SeedWeightAnchor070PostCal125FreezeDirect": {
        "component_ids": "NG132-ng70-alpha070-postcal125-freeze-direct-geometry-preserve",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "freeze_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.70,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG133-ng70SeedWeightAnchor060PostCal125FreezeQuad": {
        "component_ids": "NG133-ng70-alpha060-postcal125-freeze-quad-countercheck",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "freeze_quad",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG134-ng70SeedWeightAnchor070PostCal125FreezeQuad": {
        "component_ids": "NG134-ng70-alpha070-postcal125-freeze-quad-countercheck",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "freeze_quad",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.70,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG135-ng70SeedWeightAnchor060PostCal125QuadOnly": {
        "component_ids": "NG135-ng70-alpha060-postcal125-quad-only-geometry-repair",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_only",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG136-ng70SeedWeightAnchor070PostCal125QuadOnly": {
        "component_ids": "NG136-ng70-alpha070-postcal125-quad-only-geometry-repair",
        "actuator": "I32-TrainFeatureCenteredLogitDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_only",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.70,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG138-stableCenteredDenoiseAlpha060PostCal125": {
        "component_ids": "NG138-stable-crossbatch-centered-denoise-alpha060-postcal125",
        "actuator": "I36-TrainFeatureStableCenteredDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG139-stableCenteredDenoiseAlpha070PostCal125": {
        "component_ids": "NG139-stable-crossbatch-centered-denoise-alpha070-postcal125",
        "actuator": "I36-TrainFeatureStableCenteredDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.70,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG140-stableLowTailAlpha060PostCal125": {
        "component_ids": "NG140-stable-crossbatch-low-tail-direct-alpha060-postcal125",
        "actuator": "I37-TrainFeatureStableLowTailDirect",
        "budget_scale": 0.50,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG141-stableDenoiseLowTailMixAlpha060PostCal125": {
        "component_ids": "NG141-stable-centered-plus-low-tail-mix-alpha060-postcal125",
        "actuator": "I36-TrainFeatureStableCenteredDenoise@0.70+I37-TrainFeatureStableLowTailDirect@-0.30",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG142-stableConfidenceAlpha060PostCal125": {
        "component_ids": "NG142-stable-crossbatch-confidence-damping-alpha060-postcal125",
        "actuator": "I38-TrainFeatureStableConfidenceDamping",
        "budget_scale": 0.50,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG143-stableCenteredFreezeDirectAlpha060PostCal125": {
        "component_ids": "NG143-stable-crossbatch-centered-freeze-direct-alpha060-postcal125",
        "actuator": "I36-TrainFeatureStableCenteredDenoise",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "freeze_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG144-lowRankCouplingAlpha060PostCal125": {
        "component_ids": "NG144-lowrank-crossbatch-coupling-alpha060-postcal125",
        "actuator": "I39-TrainFeatureLowRankCouplingLift",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG145-lowRankCouplingOppositeAlpha060PostCal125": {
        "component_ids": "NG145-lowrank-crossbatch-coupling-opposite-alpha060-postcal125",
        "actuator": "I39-TrainFeatureLowRankCouplingLift",
        "budget_scale": 0.50,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG146-entropyLowRankCouplingAlpha060PostCal125": {
        "component_ids": "NG146-entropy-weighted-lowrank-coupling-alpha060-postcal125",
        "actuator": "I40-TrainFeatureEntropyWeightedLowRankCoupling",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG147-tailClippedLowRankCouplingAlpha060PostCal125": {
        "component_ids": "NG147-tail-clipped-lowrank-coupling-alpha060-postcal125",
        "actuator": "I41-TrainFeatureTailClippedLowRankCoupling",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG148-stableCenteredLowRankMixAlpha060PostCal125": {
        "component_ids": "NG148-stable-centered-plus-lowrank-coupling-mix-alpha060-postcal125",
        "actuator": "I36-TrainFeatureStableCenteredDenoise@0.70+I39-TrainFeatureLowRankCouplingLift@0.30",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG149-lowRankCouplingDirectBranchAlpha060PostCal125": {
        "component_ids": "NG149-lowrank-coupling-direct-branch-alpha060-postcal125",
        "actuator": "I39-TrainFeatureLowRankCouplingLift",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG150-lowRankDirectBranchAlpha085PostCal125": {
        "component_ids": "NG150-lowrank-direct-branch-alpha085-postcal125",
        "actuator": "I39-TrainFeatureLowRankCouplingLift",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "post_weight_interpolation_alpha": 0.85,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG151-lowRankDirectBranchAlpha100PostCal125": {
        "component_ids": "NG151-lowrank-direct-branch-alpha100-postcal125",
        "actuator": "I39-TrainFeatureLowRankCouplingLift",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "direct_branch_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "post_weight_interpolation_alpha": 1.00,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG152-lowRankDirectGainAlpha085PostCal125": {
        "component_ids": "NG152-lowrank-direct-gain-alpha085-postcal125",
        "actuator": "I39-TrainFeatureLowRankCouplingLift",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "direct_gain",
        "lambda_direct": 0.75,
        "lambda_quad": 0.25,
        "post_weight_interpolation_alpha": 0.85,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG153-lowRankAllAlpha085PostCal125": {
        "component_ids": "NG153-lowrank-all-alpha085-postcal125",
        "actuator": "I39-TrainFeatureLowRankCouplingLift",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "all",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.85,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG154-lowRankQuadDirectAlpha085PostCal125": {
        "component_ids": "NG154-lowrank-quad-direct-alpha085-postcal125",
        "actuator": "I39-TrainFeatureLowRankCouplingLift",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.85,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG155-stableCenteredLowRankMixAlpha085PostCal125": {
        "component_ids": "NG155-stable-centered-plus-lowrank-coupling-mix-alpha085-postcal125",
        "actuator": "I36-TrainFeatureStableCenteredDenoise@0.70+I39-TrainFeatureLowRankCouplingLift@0.30",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.85,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG156-crossRefCouplingAlpha060PostCal125": {
        "component_ids": "NG156-crossref-coupling-alpha060-postcal125",
        "actuator": "I42-TrainFeatureCrossRefCouplingLift",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG157-crossRefCouplingOppositeAlpha060PostCal125": {
        "component_ids": "NG157-crossref-coupling-opposite-alpha060-postcal125",
        "actuator": "I42-TrainFeatureCrossRefCouplingLift",
        "budget_scale": 0.50,
        "sign": 1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG158-entropyCrossRefCouplingAlpha060PostCal125": {
        "component_ids": "NG158-entropy-crossref-coupling-alpha060-postcal125",
        "actuator": "I43-TrainFeatureEntropyWeightedCrossRefCoupling",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG159-tailClippedCrossRefCouplingAlpha060PostCal125": {
        "component_ids": "NG159-tail-clipped-crossref-coupling-alpha060-postcal125",
        "actuator": "I44-TrainFeatureTailClippedCrossRefCoupling",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG160-stableCenteredCrossRefMixAlpha060PostCal125": {
        "component_ids": "NG160-stable-centered-plus-crossref-coupling-mix-alpha060-postcal125",
        "actuator": "I36-TrainFeatureStableCenteredDenoise@0.70+I42-TrainFeatureCrossRefCouplingLift@0.30",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.60,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
    "F25-NG161-crossRefCouplingAlpha085PostCal125": {
        "component_ids": "NG161-crossref-coupling-alpha085-postcal125",
        "actuator": "I42-TrainFeatureCrossRefCouplingLift",
        "budget_scale": 0.50,
        "sign": -1.0,
        "comp_mode": "none",
        "role_policy": "quad_direct",
        "lambda_direct": 0.50,
        "lambda_quad": 0.50,
        "post_weight_interpolation_alpha": 0.85,
        "post_logit_calibration": 1,
        "post_logit_tail_multiplier": 1.25,
        "control_residualized": 0,
        "response_residualized": 0,
        "tail_budgeted": 1,
        "train_seed_salt": len("F25-NG70-centeredDenoisePostCalBaseTail"),
    },
}


def run_candidate(args: argparse.Namespace, source: dict[str, Any], candidate_id: str, train_seed_base: int, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = CANDIDATES[candidate_id]
    base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, _base_query_logits, opt_delta, dataset, seed = role_scan.build_context(args, source, device)
    train_seed_salt = int(args.train_seed_salt_override) if int(args.train_seed_salt_override) >= 0 else int(spec.get("train_seed_salt", len(candidate_id)))
    ref_seed = int(args.ref_seed_base) if int(args.ref_seed_base) >= 0 else int(seed) + int(train_seed_base) + train_seed_salt
    refs = build_reference_stream(base, x_train, int(args.compensation_batch), int(args.ensemble_count), str(args.ref_mode), ref_seed)
    base_budget = float(source["norm_budget"])
    budget = base_budget * float(spec["budget_scale"])
    sign = float(spec["sign"])
    comp_mode = str(spec["comp_mode"])
    source_trial, role, comp = build_trial(base, str(spec["actuator"]), budget, sign, comp_mode, refs, opt_delta, int(seed), device)
    residual_info: dict[str, Any] = {
        "response_residualized": int(spec.get("response_residualized", 0)),
        "response_residual_alpha": "",
        "response_residual_alpha_clipped": "",
        "response_residual_pre_cosine": "",
        "response_residual_post_cosine": "",
        "response_residual_source_norm_pre": "",
        "response_residual_source_norm_post": "",
        "response_residual_control_norm": "",
    }
    if int(spec.get("response_residualized", 0)):
        residual_control, _resid_role, _resid_comp = build_trial(base, "TrainDirectLogitCompensatedRandomControl", budget, sign, comp_mode, refs, opt_delta, int(seed), device)
        residual_info.update(apply_response_residualization(source_trial, base, residual_control, refs, float(spec.get("response_residual_alpha_clip", 1.0))))
    noop_proxy = tail_proxy(base, refs)
    source_proxy = tail_proxy(source_trial, refs)
    precommit_tail_score = score_tail(source_proxy)
    tail_budget_scale_applied = 1.0
    if int(spec["tail_budgeted"]) and precommit_tail_score > score_tail(noop_proxy) + float(args.tail_proxy_tau):
        tail_budget_scale_applied = 0.50
        budget *= tail_budget_scale_applied
        source_trial, role, comp = build_trial(base, str(spec["actuator"]), budget, sign, comp_mode, refs, opt_delta, int(seed), device)
        residual_info = {
            "response_residualized": int(spec.get("response_residualized", 0)),
            "response_residual_alpha": "",
            "response_residual_alpha_clipped": "",
            "response_residual_pre_cosine": "",
            "response_residual_post_cosine": "",
            "response_residual_source_norm_pre": "",
            "response_residual_source_norm_post": "",
            "response_residual_control_norm": "",
        }
        if int(spec.get("response_residualized", 0)):
            residual_control, _resid_role, _resid_comp = build_trial(base, "TrainDirectLogitCompensatedRandomControl", budget, sign, comp_mode, refs, opt_delta, int(seed), device)
            residual_info.update(apply_response_residualization(source_trial, base, residual_control, refs, float(spec.get("response_residual_alpha_clip", 1.0))))
        source_proxy = tail_proxy(source_trial, refs)
        precommit_tail_score = score_tail(source_proxy)
    variants = [
        ("noop", "NoOpMatchedOverhead", 0.0, sign, 0),
        ("source", str(spec["actuator"]), budget, sign, 1),
        ("same_compensation_control", "TrainDirectLogitCompensatedRandomControl", budget, sign, 1),
        ("random_norm_control", "RandomMatchedNorm", budget, sign, 0),
        ("snr_only_control", "SNR-only", budget, sign, 0),
        ("adamw_parallel_control", "AdamWParallelDirection", budget, sign, 0),
    ]
    post_logit_target_tail = train_stream_abs_tail(base, refs) * float(spec.get("post_logit_tail_multiplier", 1.0))
    rows: list[dict[str, Any]] = []
    finals: dict[str, dict[str, Any]] = {}
    models: dict[str, torch.nn.Module] = {}
    for idx, (kind, actuator, bgt, sgn, needs_comp) in enumerate(variants):
        if kind == "source":
            trial = source_trial
            row_role = role
            row_comp = comp
        else:
            trial, row_role, row_comp = build_trial(base, actuator, bgt, sgn, comp_mode, refs, opt_delta, int(seed), device)
        init_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        pre_train_state = {k: v.detach().clone() for k, v in trial.state_dict().items()}
        final_eval, q90, auc = bridge.train_role(
            trial,
            x_train,
            y_train,
            x_val,
            y_val,
            int(args.batch_size),
            int(args.epochs),
            float(args.lr),
            float(args.weight_decay),
            int(seed) + int(train_seed_base) + idx + train_seed_salt,
            device,
            str(spec["role_policy"]),
            float(spec.get("label_smoothing", 0.0)),
        )
        post_weight_alpha = float(spec.get("post_weight_interpolation_alpha", 1.0))
        post_weight_applied = interpolate_model_state(trial, pre_train_state, post_weight_alpha)
        if post_weight_applied:
            final_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        if int(spec.get("post_logit_calibration", 0)):
            trial, post_cal = post_train_logit_calibrate(trial, refs, post_logit_target_tail)
            final_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        else:
            post_cal = {"post_logit_calibration_applied": 0, "post_logit_scale": "", "post_logit_pre_tail": "", "post_logit_target_tail": ""}
        row = {
            "stage": "V1225_COMPOSITE_FUNCTIONAL_BRIDGE_VARIANT",
            "candidate_id": candidate_id,
            "component_ids": str(spec["component_ids"]),
            "variant_kind": kind,
            "actuator_id": actuator,
            "role": row_role,
            "dataset": dataset,
            "seed": seed,
            "train_seed_base": int(train_seed_base),
            "train_seed_salt": train_seed_salt,
            "ref_mode": str(args.ref_mode),
            "ref_seed": ref_seed,
            "norm_budget": bgt,
            "signed_direction": sgn,
            "lambda_direct": float(spec["lambda_direct"]),
            "lambda_quad": float(spec["lambda_quad"]),
            "control_residualized": int(spec["control_residualized"]),
            "response_residualized": residual_info["response_residualized"] if kind == "source" else "",
            "response_residual_alpha": residual_info["response_residual_alpha"] if kind == "source" else "",
            "response_residual_alpha_clipped": residual_info["response_residual_alpha_clipped"] if kind == "source" else "",
            "response_residual_pre_cosine": residual_info["response_residual_pre_cosine"] if kind == "source" else "",
            "response_residual_post_cosine": residual_info["response_residual_post_cosine"] if kind == "source" else "",
            "response_residual_source_norm_pre": residual_info["response_residual_source_norm_pre"] if kind == "source" else "",
            "response_residual_source_norm_post": residual_info["response_residual_source_norm_post"] if kind == "source" else "",
            "response_residual_control_norm": residual_info["response_residual_control_norm"] if kind == "source" else "",
            "tail_budgeted": int(spec["tail_budgeted"]),
            "tail_budget_scale_applied": tail_budget_scale_applied if kind == "source" else "",
            "role_policy": str(spec["role_policy"]),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "label_smoothing": float(spec.get("label_smoothing", 0.0)),
            "post_weight_interpolation_applied": post_weight_applied,
            "post_weight_interpolation_alpha": post_weight_alpha,
            **post_cal,
            "init_acc": init_eval.get("acc", ""),
            "init_NLL": init_eval.get("NLL", ""),
            "init_ECE": init_eval.get("ECE", ""),
            "init_CEp99": init_eval.get("CEp99", ""),
            "final_acc": final_eval.get("acc", ""),
            "final_NLL": final_eval.get("NLL", ""),
            "final_ECE": final_eval.get("ECE", ""),
            "final_CEp99": final_eval.get("CEp99", ""),
            "step_time_q90_ms": q90,
            "auc_error_time": auc,
            "uses_query_batch": 0,
            "uses_train_batch": int(needs_comp),
            "uses_label_for_direction": int(actuator == "AdamWParallelDirection"),
            "uses_ce_for_direction": int(actuator == "AdamWParallelDirection"),
            "precommit_available": int(actuator != "AdamWParallelDirection"),
            "promotion_allowed": 0,
            "no_fake": 1,
            **row_comp,
        }
        if kind == "source":
            row.update({f"precommit_{k}": v for k, v in source_proxy.items()})
            row["precommit_tail_proxy"] = precommit_tail_score
        rows.append(row)
        finals[kind] = row
        models[kind] = trial
    linec_rows, linec_pass_count, linec_all_pass = bridge.linec_seed_summary(
        {"noop": models["noop"], "source": models["source"]},
        xb,
        yb,
        xq,
        yq,
        int(seed),
        parse_ints(args.linec_seeds),
        int(args.linec_sketch_dim),
    )
    for linec_row in linec_rows:
        rows.append({**linec_row, "candidate_id": candidate_id, "dataset": dataset, "seed": seed, "train_seed_base": int(train_seed_base), "train_seed_salt": train_seed_salt, "ref_mode": str(args.ref_mode), "ref_seed": ref_seed, "promotion_allowed": 0, "no_fake": 1})
    src = finals["source"]
    noop = finals["noop"]
    controls = [finals[k] for k in ["same_compensation_control", "random_norm_control", "snr_only_control", "adamw_parallel_control"]]
    src_acc = fnum(src.get("final_acc"))
    noop_acc = fnum(noop.get("final_acc"))
    control_acc = max(fnum(c.get("final_acc")) for c in controls)
    overhead = fnum(src.get("step_time_q90_ms")) / fnum(noop.get("step_time_q90_ms")) if fnum(noop.get("step_time_q90_ms")) > 0 else float("nan")
    task_strict = int(
        math.isfinite(src_acc)
        and src_acc >= control_acc + 0.0078125
        and src_acc >= noop_acc + 0.015625
        and fnum(src.get("final_NLL")) <= fnum(noop.get("final_NLL"))
        and fnum(src.get("final_CEp99")) <= fnum(noop.get("final_CEp99")) + 0.05
        and fnum(src.get("final_ECE")) <= fnum(noop.get("final_ECE")) + 0.02
        and overhead <= 1.05
    )
    task_explore = int(
        math.isfinite(src_acc)
        and src_acc >= control_acc + 0.00390625
        and src_acc >= noop_acc + 0.015625
        and fnum(src.get("final_CEp99")) <= fnum(noop.get("final_CEp99")) + 0.50
    )
    linec_majority_pass = int(linec_pass_count >= math.ceil(len(parse_ints(args.linec_seeds)) / 2.0))
    summary = {
        "stage": "V1225_COMPOSITE_FUNCTIONAL_BRIDGE_SUMMARY",
        "candidate_id": candidate_id,
        "component_ids": str(spec["component_ids"]),
        "dataset": dataset,
        "seed": seed,
        "train_seed_base": int(train_seed_base),
        "train_seed_salt": train_seed_salt,
        "ref_mode": str(args.ref_mode),
        "ref_seed": ref_seed,
        "role_policy": str(spec["role_policy"]),
        "lambda_direct": float(spec["lambda_direct"]),
        "lambda_quad": float(spec["lambda_quad"]),
        "control_residualized": int(spec["control_residualized"]),
        "response_residualized": int(spec.get("response_residualized", 0)),
        "response_residual_alpha": residual_info["response_residual_alpha"],
        "response_residual_alpha_clipped": residual_info["response_residual_alpha_clipped"],
        "response_residual_pre_cosine": residual_info["response_residual_pre_cosine"],
        "response_residual_post_cosine": residual_info["response_residual_post_cosine"],
        "tail_budgeted": int(spec["tail_budgeted"]),
        "source_acc": src_acc,
        "noop_acc": noop_acc,
        "best_control_acc": control_acc,
        "source_vs_noop": src_acc - noop_acc,
        "source_vs_best_control": src_acc - control_acc,
        "source_NLL": fnum(src.get("final_NLL")),
        "noop_NLL": fnum(noop.get("final_NLL")),
        "source_CEp99": fnum(src.get("final_CEp99")),
        "noop_CEp99": fnum(noop.get("final_CEp99")),
        "source_ECE": fnum(src.get("final_ECE")),
        "noop_ECE": fnum(noop.get("final_ECE")),
        "LineC_seed_pass_count": linec_pass_count,
        "strict_majority_pass": int(task_strict and linec_majority_pass),
        "strict_all_pass": int(task_strict and linec_all_pass),
        "exploration_gate_pass": int(task_explore and linec_majority_pass),
        "amortized_overhead_ratio": overhead,
        "precommit_tail_proxy": precommit_tail_score,
        "uses_query_batch": 0,
        "uses_train_batch": 1,
        "uses_label_for_direction": 0,
        "uses_ce_for_direction": 0,
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    rows.append(summary)
    return rows, summary


def run() -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-out-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-prefix", default="v1225_composite_functional_bridge")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--compensation-batch", type=int, default=32)
    ap.add_argument("--ensemble-count", type=int, default=8)
    ap.add_argument("--train-seed-bases", default="12240400,12241400,12242400")
    ap.add_argument("--linec-seeds", default="12239500,12240600,12241600,12242600,12243600")
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--tail-proxy-tau", type=float, default=0.50)
    ap.add_argument("--ref-mode", default="sequential", choices=["sequential", "bootstrap_mom"])
    ap.add_argument("--ref-seed-base", type=int, default=-1)
    ap.add_argument("--train-seed-salt-override", type=int, default=-1)
    ap.add_argument("--candidates", default=",".join(CANDIDATES.keys()))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    exp.ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.25 composite bridge requires CUDA")
    torch.cuda.set_device(device)
    source = p4m.pick_source(Path(args.source_out_dir))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for candidate_id in parse_csv(args.candidates):
        if candidate_id not in CANDIDATES:
            raise ValueError(f"unknown candidate {candidate_id}")
        for train_seed_base in parse_ints(args.train_seed_bases):
            rr, ss = run_candidate(args, source, candidate_id, train_seed_base, device)
            rows.extend(rr)
            summaries.append(ss)
            torch.cuda.empty_cache()
    aggregate_rows: list[dict[str, Any]] = []
    for candidate_id in sorted(set(str(s["candidate_id"]) for s in summaries)):
        group = [s for s in summaries if str(s["candidate_id"]) == candidate_id]
        maj = sum(exp.safe_int(g.get("strict_majority_pass"), 0) for g in group)
        allp = sum(exp.safe_int(g.get("strict_all_pass"), 0) for g in group)
        exp_count = sum(exp.safe_int(g.get("exploration_gate_pass"), 0) for g in group)
        aggregate_rows.append(
            {
                "stage": "V1225_COMPOSITE_FUNCTIONAL_BRIDGE_AGGREGATE",
                "candidate_id": candidate_id,
                "train_shuffle_rows": len(group),
                "train_shuffle_strict_majority_pass_count": maj,
                "train_shuffle_strict_all_pass_count": allp,
                "train_shuffle_exploration_pass_count": exp_count,
                "train_shuffle_robust_majority_pass": int(maj >= 2),
                "train_shuffle_robust_all_pass": int(allp >= 2),
                "exploration_train_shuffle_positive_count": exp_count,
                "exploration_train_shuffle_robust_pass": int(exp_count >= 2),
                "best_source_vs_noop": max(fnum(g.get("source_vs_noop"), -999.0) for g in group),
                "best_source_vs_control": max(fnum(g.get("source_vs_best_control"), -999.0) for g in group),
                "best_linec_seed_pass_count": max(exp.safe_int(g.get("LineC_seed_pass_count"), 0) for g in group),
                "promotion_allowed": 0,
                "no_fake": 1,
            }
        )
    rows.extend(aggregate_rows)
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    exp.write_csv_rows(csv_path, rows)
    result = {
        "stage": "V1225_COMPOSITE_FUNCTIONAL_BRIDGE_RESULT",
        "artifact_csv": exp.rel(csv_path),
        "candidate_rows": len(summaries),
        "aggregate_rows": len(aggregate_rows),
        "any_exploration_pass": int(any(exp.safe_int(r.get("exploration_train_shuffle_robust_pass"), 0) for r in aggregate_rows)),
        "any_strict_majority_pass": int(any(exp.safe_int(r.get("strict_majority_pass"), 0) for r in summaries)),
        "any_strict_all_pass": int(any(exp.safe_int(r.get("strict_all_pass"), 0) for r in summaries)),
        "combined_bridge_any_train_shuffle_robust_majority_pass": int(any(exp.safe_int(r.get("train_shuffle_robust_majority_pass"), 0) for r in aggregate_rows)),
        "combined_bridge_any_train_shuffle_robust_all_pass": int(any(exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0) for r in aggregate_rows)),
        "best_candidate_aggregates": sorted(
            aggregate_rows,
            key=lambda r: (
                exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0),
                exp.safe_int(r.get("exploration_train_shuffle_robust_pass"), 0),
                fnum(r.get("best_source_vs_control"), -999.0),
                fnum(r.get("best_source_vs_noop"), -999.0),
            ),
            reverse=True,
        ),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    exp.write_json(out_dir / f"{args.artifact_prefix}_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
