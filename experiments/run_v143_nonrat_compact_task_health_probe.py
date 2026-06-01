#!/usr/bin/env python
"""v14.3 compact Non-RAT task-health probe.

This diagnostic follows the R4-NonRATSubstrateMissing branch after a compact
manual-kernel workspace probe finds memory/step-feasible rows.  It does not run
FMS proof and does not use LineC/tail metrics as direction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.diagnostics.basis_workspace import V1235_BASIS_CANDIDATES, fnum, parse_csv, parse_ints  # noqa: E402
from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments import run_v1223_failclosed_explore_open2_functional_rebuild as v1223  # noqa: E402
from experiments.run_v1231_basis_kernel_workspace import classification_basic, linec_metrics, make_adamw, train_mlp_for_reference  # noqa: E402
from experiments.run_v143_nonrat_manual_kernel_substrate_probe import make_probe_model  # noqa: E402


DEFAULT_CANDIDATES = (
    "D-RBF14-OOGBoundaryRepairSubstrate,"
    "D-CHE13-FusedReadoutGradNoMaterialize-K3,"
    "D-FOU13-FusedReadoutGradNoMaterialize-K2,"
    "D-WAV10-HatWaveletLifetimeRepair-Monitor"
)


class LogitScaleWrapper(torch.nn.Module):
    """Label-free output geometry wrapper used only for compact substrate audit."""

    def __init__(self, base: torch.nn.Module, scale: float) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("logit_scale", torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)
        return logits * self.logit_scale.to(device=x.device, dtype=logits.dtype)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_workspace_rows(path: str) -> dict[tuple[str, str], dict[str, str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[tuple[str, str], dict[str, str]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[(str(row.get("candidate_id", "")), str(row.get("probe_mode", "")))] = row
    return out


def apply_wavelet_role_constraint(model: torch.nn.Module, args: argparse.Namespace) -> int:
    constraint = str(getattr(args, "wavelet_role_constraint", "none"))
    if constraint == "none" or getattr(getattr(model, "spec", None), "basis_name", "") != "hat_wavelet":
        return 0
    grad = getattr(getattr(model, "linear_readout", None), "grad", None)
    if grad is None:
        return 0
    if constraint == "freeze_linear_readout":
        grad.zero_()
        return 1
    if constraint == "linear_readout_grad050":
        grad.mul_(0.50)
        return 1
    if constraint == "linear_readout_grad025":
        grad.mul_(0.25)
        return 1
    if constraint == "linear_readout_grad010":
        grad.mul_(0.10)
        return 1
    if constraint == "linear_readout_grad005":
        grad.mul_(0.05)
        return 1
    raise ValueError(f"unknown wavelet role constraint: {constraint}")


def manual_train_step(model: torch.nn.Module, opt: torch.optim.Optimizer, xb: torch.Tensor, yb: torch.Tensor, device: torch.device, args: argparse.Namespace | None = None) -> tuple[float, int]:
    opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    if hasattr(model, "manual_ce_forward_cache") and hasattr(model, "manual_ce_backward_from_cache"):
        logits, cache = model.manual_ce_forward_cache(xb)
        _loss = model.manual_ce_backward_from_cache(logits, cache, yb)
        del logits, cache, _loss
    else:
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
    constraint_applied = apply_wavelet_role_constraint(model, args) if args is not None else 0
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0, int(constraint_applied)


def wavelet_support_audit(model: torch.nn.Module, x_train: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {
        "wavelet_scale": "",
        "wavelet_center_min": "",
        "wavelet_center_max": "",
        "wavelet_layer1_active_fraction": "",
        "wavelet_layer1_channel_occ_std": "",
    }
    if getattr(getattr(model, "spec", None), "basis_name", "") != "hat_wavelet":
        return out
    with torch.no_grad():
        x = x_train[: min(256, int(x_train.shape[0]))]
        b1 = model.layer1_basis(x)
        active = b1.abs() > 1.0e-6
        channel_occ = active.float().mean(dim=tuple(range(active.ndim - 1)))
        out.update(
            {
                "wavelet_scale": float(model.scales[0].detach().float().item()),
                "wavelet_center_min": float(model.centers.detach().float().min().item()),
                "wavelet_center_max": float(model.centers.detach().float().max().item()),
                "wavelet_layer1_active_fraction": float(active.float().mean().item()),
                "wavelet_layer1_channel_occ_std": float(channel_occ.std(unbiased=False).item()) if channel_occ.numel() > 1 else 0.0,
            }
        )
    return out


def apply_wavelet_support_repair(model: torch.nn.Module, args: argparse.Namespace, x_train: torch.Tensor) -> dict[str, Any]:
    repair = str(getattr(args, "wavelet_support_repair", "none"))
    out: dict[str, Any] = {
        "wavelet_support_repair": repair,
        "wavelet_support_repair_applied": 0,
        "wavelet_support_uses_train_stream_features": 0,
        "wavelet_support_uses_labels": 0,
        "wavelet_support_uses_linec_tail_direction": 0,
        "wavelet_scale_before": "",
        "wavelet_scale_after": "",
        "wavelet_center_min_before": "",
        "wavelet_center_max_before": "",
        "wavelet_center_min_after": "",
        "wavelet_center_max_after": "",
        "wavelet_layer1_active_fraction_after": "",
        "wavelet_layer1_channel_occ_std_after": "",
    }
    if repair == "none" or getattr(getattr(model, "spec", None), "basis_name", "") != "hat_wavelet":
        out.update(wavelet_support_audit(model, x_train))
        return out
    with torch.no_grad():
        out["wavelet_scale_before"] = float(model.scales[0].detach().float().item())
        out["wavelet_center_min_before"] = float(model.centers.detach().float().min().item())
        out["wavelet_center_max_before"] = float(model.centers.detach().float().max().item())
        factor = 1.0
        if repair.startswith("scale"):
            factor = float(repair.replace("scale", "")) / 100.0
        elif repair == "quantile":
            factor = 1.0
        elif repair == "quantile_scale075":
            factor = 0.75
        elif repair == "quantile_scale050":
            factor = 0.50
        else:
            raise ValueError(f"unknown wavelet support repair: {repair}")
        if repair.startswith("quantile"):
            z = model._norm_input(x_train[: min(4096, int(x_train.shape[0]))]).detach().flatten().float()
            probs = torch.linspace(0.10, 0.90, int(model.k), device=z.device)
            centers = torch.quantile(z, probs).clamp(-1.5, 1.5)
            model.centers.copy_(centers.to(device=model.centers.device, dtype=model.centers.dtype))
        model.scales.mul_(factor).clamp_(min=0.05, max=4.0)
        out["wavelet_support_repair_applied"] = 1
        out["wavelet_support_uses_train_stream_features"] = int(repair.startswith("quantile"))
        out["wavelet_scale_after"] = float(model.scales[0].detach().float().item())
        out["wavelet_center_min_after"] = float(model.centers.detach().float().min().item())
        out["wavelet_center_max_after"] = float(model.centers.detach().float().max().item())
    audit = wavelet_support_audit(model, x_train)
    out["wavelet_layer1_active_fraction_after"] = audit["wavelet_layer1_active_fraction"]
    out["wavelet_layer1_channel_occ_std_after"] = audit["wavelet_layer1_channel_occ_std"]
    out.update(audit)
    return out


def rbf_center_audit(model: torch.nn.Module, x_train: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rbf_scale": "",
        "rbf_center_min": "",
        "rbf_center_max": "",
        "rbf_layer1_active_fraction": "",
        "rbf_layer1_channel_occ_std": "",
        "center_occupancy_entropy": "",
        "empty_center_fraction": "",
        "width_condition": "",
        "out_of_grid_fraction": "",
        "residual_over_base": "",
        "residual_over_base_available": 0,
        "residual_over_base_source": "",
        "residual_over_base_uses_labels": 0,
        "residual_over_base_uses_linec_tail_direction": 0,
    }
    if getattr(getattr(model, "spec", None), "basis_name", "") not in {"compact_rbf", "fastkan_rbf"}:
        return out
    with torch.no_grad():
        x = x_train[: min(256, int(x_train.shape[0]))]
        b1 = model.layer1_basis(x)
        active = b1 > 1.0e-3
        channel_occ = active.float().mean(dim=tuple(range(active.ndim - 1)))
        occ = channel_occ.float().clamp_min(0.0)
        occ_dist = occ / occ.sum().clamp_min(1.0e-8)
        center_entropy = -(occ_dist * occ_dist.clamp_min(1.0e-8).log()).sum() / math.log(max(2, int(occ_dist.numel())))
        centers = model.centers.detach().float()
        scales = model.scales.detach().float().abs().clamp_min(1.0e-8)
        residual_over_base = 0.0
        residual_source = "no_linear_residual_present"
        if bool(getattr(model, "linear_residual_enabled", False)) and hasattr(model, "linear_readout"):
            z = model._norm_input(x)
            logits = model(x).detach().float()
            residual = (z @ model.linear_readout).detach().float() / float(model._linear_residual_denominator())
            base = logits - residual
            residual_over_base = float(residual.square().mean().sqrt().div(base.square().mean().sqrt().clamp_min(1.0e-8)).item())
            residual_source = "train_stream_logit_residual_norm_ratio"
        out.update(
            {
                "rbf_scale": float(model.scales[0].detach().float().item()),
                "rbf_center_min": float(model.centers.detach().float().min().item()),
                "rbf_center_max": float(model.centers.detach().float().max().item()),
                "rbf_layer1_active_fraction": float(active.float().mean().item()),
                "rbf_layer1_channel_occ_std": float(channel_occ.std(unbiased=False).item()) if channel_occ.numel() > 1 else 0.0,
                "center_occupancy_entropy": float(center_entropy.item()),
                "empty_center_fraction": float((channel_occ <= 1.0e-6).float().mean().item()),
                "width_condition": float((scales.max() / scales.min()).item()),
                "out_of_grid_fraction": float(((centers < -1.5) | (centers > 1.5)).float().mean().item()),
                "residual_over_base": residual_over_base,
                "residual_over_base_available": 1,
                "residual_over_base_source": residual_source,
            }
        )
    return out


def fourier_band_audit(model: torch.nn.Module, x_train: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {
        "fourier_telemetry_available": 0,
        "fourier_telemetry_source": "",
        "fourier_telemetry_uses_labels": 0,
        "fourier_telemetry_uses_linec_tail_direction": 0,
        "band_energy_low": "",
        "band_energy_mid": "",
        "band_energy_high": "",
        "phase_drift": "",
        "high_freq_ratio": "",
        "bandwise_snr": "",
    }
    if getattr(getattr(model, "spec", None), "basis_name", "") != "fourier_lowfreq":
        return out
    with torch.no_grad():
        x = x_train[: min(256, int(x_train.shape[0]))]
        basis = model.layer1_basis(x).detach().float()
        channel_energy = basis.square().mean(dim=tuple(range(basis.ndim - 1)))
        bands = torch.tensor_split(channel_energy, 3)
        band_vals = [
            float(b.mean().item()) if b.numel() else 0.0
            for b in bands
        ]
        total_energy = float(channel_energy.sum().clamp_min(1.0e-8).item())
        high_energy_sum = float(bands[2].sum().item()) if len(bands) >= 3 and bands[2].numel() else 0.0
        z = model._norm_input(x).detach().float()
        phase = math.pi * z
        resultant = torch.sqrt(torch.cos(phase).mean(dim=0).square() + torch.sin(phase).mean(dim=0).square())
        phase_drift = 1.0 - float(resultant.mean().clamp(0.0, 1.0).item())
        out.update(
            {
                "fourier_telemetry_available": 1,
                "fourier_telemetry_source": "train_stream_layer1_basis_channel_split",
                "band_energy_low": band_vals[0],
                "band_energy_mid": band_vals[1],
                "band_energy_high": band_vals[2],
                "phase_drift": phase_drift,
                "high_freq_ratio": high_energy_sum / total_energy,
                "bandwise_snr": band_vals[0] / max(1.0e-8, band_vals[1] + band_vals[2]),
            }
        )
    return out


def apply_rbf_center_repair(model: torch.nn.Module, args: argparse.Namespace, x_train: torch.Tensor) -> dict[str, Any]:
    repair = str(getattr(args, "rbf_center_repair", "none"))
    out: dict[str, Any] = {
        "rbf_center_repair": repair,
        "rbf_center_repair_applied": 0,
        "rbf_center_repair_uses_train_stream_features": 0,
        "rbf_center_repair_uses_labels": 0,
        "rbf_center_repair_uses_linec_tail_direction": 0,
        "rbf_scale_before": "",
        "rbf_scale_after": "",
        "rbf_center_min_before": "",
        "rbf_center_max_before": "",
        "rbf_center_min_after": "",
        "rbf_center_max_after": "",
        "rbf_layer1_active_fraction_after": "",
        "rbf_layer1_channel_occ_std_after": "",
    }
    if repair == "none" or getattr(getattr(model, "spec", None), "basis_name", "") not in {"compact_rbf", "fastkan_rbf"}:
        out.update(rbf_center_audit(model, x_train))
        return out
    with torch.no_grad():
        out["rbf_scale_before"] = float(model.scales[0].detach().float().item())
        out["rbf_center_min_before"] = float(model.centers.detach().float().min().item())
        out["rbf_center_max_before"] = float(model.centers.detach().float().max().item())
        z = model._norm_input(x_train[: min(4096, int(x_train.shape[0]))]).detach().flatten().float()
        probs = torch.linspace(0.10, 0.90, int(model.k), device=z.device)
        centers = torch.quantile(z, probs).clamp(-1.5, 1.5)
        model.centers.copy_(centers.to(device=model.centers.device, dtype=model.centers.dtype))
        factor = 1.0
        if repair == "quantile_width075":
            factor = 0.75
        elif repair == "quantile_width100":
            factor = 1.00
        elif repair == "quantile_width125":
            factor = 1.25
        elif repair == "quantile_width175":
            factor = 1.75
        elif repair != "quantile":
            raise ValueError(f"unknown RBF center repair: {repair}")
        model.scales.mul_(factor).clamp_(min=0.05, max=4.0)
        out["rbf_center_repair_applied"] = 1
        out["rbf_center_repair_uses_train_stream_features"] = 1
        out["rbf_scale_after"] = float(model.scales[0].detach().float().item())
        out["rbf_center_min_after"] = float(model.centers.detach().float().min().item())
        out["rbf_center_max_after"] = float(model.centers.detach().float().max().item())
    audit = rbf_center_audit(model, x_train)
    out["rbf_layer1_active_fraction_after"] = audit["rbf_layer1_active_fraction"]
    out["rbf_layer1_channel_occ_std_after"] = audit["rbf_layer1_channel_occ_std"]
    out.update(audit)
    return out


def apply_output_geometry_repair(model: torch.nn.Module, args: argparse.Namespace, x_train: torch.Tensor) -> tuple[torch.nn.Module, dict[str, Any]]:
    repair = str(getattr(args, "output_geometry_repair", "none"))
    out: dict[str, Any] = {
        "output_geometry_repair": repair,
        "output_geometry_repair_applied": 0,
        "output_geometry_uses_train_stream_logits": 0,
        "output_geometry_uses_labels": 0,
        "output_geometry_uses_linec_tail_direction": 0,
        "output_geometry_logit_rms_before": "",
        "output_geometry_logit_scale": 1.0,
        "output_geometry_logit_rms_after": "",
        "output_geometry_entropy_norm": "",
        "output_geometry_top_prob_mean": "",
        "output_geometry_margin_mean": "",
        "output_geometry_soft_class_entropy_norm": "",
        "output_geometry_selected_target": "",
    }
    with torch.no_grad():
        logits = model(x_train[: min(512, int(x_train.shape[0]))])
        rms = float(logits.detach().float().square().mean().sqrt().clamp_min(1.0e-8).item())
        probs = logits.detach().float().softmax(dim=1)
        entropy = -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=1)
        entropy_norm = float((entropy / math.log(max(2, int(probs.shape[1])))).mean().item())
        top2 = torch.topk(probs, k=min(2, int(probs.shape[1])), dim=1).values
        top_prob_mean = float(top2[:, 0].mean().item())
        margin_mean = float((top2[:, 0] - top2[:, 1]).mean().item()) if top2.shape[1] > 1 else 0.0
        class_mass = probs.mean(dim=0)
        class_entropy = -(class_mass * class_mass.clamp_min(1.0e-8).log()).sum()
        soft_class_entropy_norm = float((class_entropy / math.log(max(2, int(probs.shape[1])))).item())
    out["output_geometry_logit_rms_before"] = rms
    out["output_geometry_entropy_norm"] = entropy_norm
    out["output_geometry_top_prob_mean"] = top_prob_mean
    out["output_geometry_margin_mean"] = margin_mean
    out["output_geometry_soft_class_entropy_norm"] = soft_class_entropy_norm
    if repair == "none":
        return model, out
    scale = 1.0
    target = 1.0
    if repair == "fixed050":
        scale = 0.50
        target = rms * scale
    elif repair == "fixed025":
        scale = 0.25
        target = rms * scale
    elif repair.startswith("train_rms_target"):
        target = float(repair.replace("train_rms_target", "")) / 100.0
        scale = max(0.05, min(2.0, target / max(1.0e-8, rms)))
        out["output_geometry_uses_train_stream_logits"] = 1
    elif repair.startswith("train_entropy_t") and repair.endswith("_100_else050"):
        thresh = float(repair.split("_")[2].replace("t", "")) / 100.0
        target = 1.0 if entropy_norm >= thresh else 0.50
        scale = max(0.05, min(2.0, target / max(1.0e-8, rms)))
        out["output_geometry_uses_train_stream_logits"] = 1
    elif repair.startswith("train_margin_t") and repair.endswith("_050_else100"):
        thresh = float(repair.split("_")[2].replace("t", "")) / 100.0
        target = 0.50 if margin_mean >= thresh else 1.0
        scale = max(0.05, min(2.0, target / max(1.0e-8, rms)))
        out["output_geometry_uses_train_stream_logits"] = 1
    elif repair.startswith("train_topprob_t") and repair.endswith("_050_else100"):
        thresh = float(repair.split("_")[2].replace("t", "")) / 100.0
        target = 0.50 if top_prob_mean >= thresh else 1.0
        scale = max(0.05, min(2.0, target / max(1.0e-8, rms)))
        out["output_geometry_uses_train_stream_logits"] = 1
    else:
        raise ValueError(f"unknown output geometry repair: {repair}")
    out["output_geometry_repair_applied"] = 1
    out["output_geometry_selected_target"] = float(target)
    out["output_geometry_logit_scale"] = float(scale)
    out["output_geometry_logit_rms_after"] = float(rms * scale)
    return LogitScaleWrapper(model, float(scale)).to(next(model.parameters()).device), out


def train_compact_candidate(args: argparse.Namespace, candidate_id: str, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, input_dim: int, output_dim: int, seed: int, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cand = V1235_BASIS_CANDIDATES[candidate_id]
    model, spec = make_probe_model(cand.candidate_id, input_dim, output_dim, x_train, device, int(seed) + 14_360, int(args.hidden_override))
    support_info = apply_wavelet_support_repair(model, args, x_train)
    rbf_info = apply_rbf_center_repair(model, args, x_train)
    fourier_info = fourier_band_audit(model, x_train)
    opt = make_adamw(args, [p for p in model.parameters() if p.requires_grad])
    gen = torch.Generator(device=device).manual_seed(int(seed) + 14_370)
    times: list[float] = []
    role_constraint_steps = 0
    for _epoch in range(int(args.epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[off:off + int(args.batch_size)]
            dt_ms, applied = manual_train_step(model, opt, x_train[idx], y_train[idx], device, args)
            times.append(dt_ms)
            role_constraint_steps += int(applied)
    eval_model, output_info = apply_output_geometry_repair(model, args, x_train)
    ev = classification_basic(eval_model, x_val, y_val)

    linec_rows: list[dict[str, Any]] = []
    b = min(int(args.linec_batch_size), int(x_train.shape[0]), int(x_val.shape[0]))
    for ls in parse_ints(args.linec_seeds):
        try:
            lm = linec_metrics(eval_model, x_train[:b], y_train[:b], x_val[:b], y_val[:b], int(ls), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
            status = "executed"
            error = ""
        except Exception as exc:  # noqa: BLE001
            lm = {"CouplingR2": float("nan"), "NoiseSignalLeak": float("nan"), "RealSignalReservoirRatio": float("nan")}
            status = "blocked"
            error = f"{type(exc).__name__}: {exc}"
        linec_rows.append({
            "stage": "V143_COMPACT_NONRAT_LINEC_AUDIT",
            "family": cand.family,
            "candidate_id": cand.candidate_id,
            "mapped_method_id": cand.method_id,
            "linec_seed": int(ls),
            "linec_status": status,
            "linec_error": error,
            **lm,
            "linec_used_for_direction": 0,
            "wavelet_support_repair": support_info.get("wavelet_support_repair", "none"),
            "wavelet_role_constraint": str(getattr(args, "wavelet_role_constraint", "none")),
            "wavelet_support_uses_linec_tail_direction": 0,
            "wavelet_role_constraint_uses_linec_tail_direction": 0,
            "rbf_center_repair": rbf_info.get("rbf_center_repair", "none"),
            "rbf_center_repair_uses_linec_tail_direction": 0,
            "fourier_telemetry_source": fourier_info.get("fourier_telemetry_source", ""),
            "fourier_telemetry_uses_linec_tail_direction": 0,
            "output_geometry_repair": output_info.get("output_geometry_repair", "none"),
            "output_geometry_uses_linec_tail_direction": 0,
            "promotion_allowed": 0,
        })
    step_q90 = float(torch.tensor(times, device=device).quantile(0.90).item()) if times else float("nan")
    row = {
        "stage": "V143_COMPACT_NONRAT_TASK_HEALTH",
        "family": cand.family,
        "candidate_id": cand.candidate_id,
        "mapped_method_id": cand.method_id,
        "basis_name": getattr(spec, "basis_name", ""),
        "init_variant": getattr(spec, "init_variant", ""),
        "spec_hidden_dim": int(getattr(spec, "hidden_dim", 0)),
        "hidden_override": int(args.hidden_override),
        "capacity_reduced_substrate_probe": int(int(args.hidden_override) > 0),
        "manual_train_stream_ce_path": int(hasattr(model, "manual_ce_forward_cache") and hasattr(model, "manual_ce_backward_from_cache")),
        "epochs": int(args.epochs),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "val_acc": ev["acc"],
        "NLL": ev["NLL"],
        "ECE": ev["ECE"],
        "CEp99": ev["CEp99"],
        "step_time_q90_ms": step_q90,
        "LineC_pass_count": sum(
            int(fnum(r.get("CouplingR2"), -999.0) >= 0.15 and fnum(r.get("NoiseSignalLeak"), 999.0) <= 0.20 and fnum(r.get("RealSignalReservoirRatio"), 999.0) <= 0.70)
            for r in linec_rows
        ),
        "LineC_seed_count": len(linec_rows),
        "LineC_pass_rate": sum(
            int(fnum(r.get("CouplingR2"), -999.0) >= 0.15 and fnum(r.get("NoiseSignalLeak"), 999.0) <= 0.20 and fnum(r.get("RealSignalReservoirRatio"), 999.0) <= 0.70)
            for r in linec_rows
        ) / max(1, len(linec_rows)),
        "LineC_CouplingR2_mean": sum(fnum(r.get("CouplingR2"), 0.0) for r in linec_rows) / max(1, len(linec_rows)),
        "official_fms_proof_executed": 0,
        "promotion_allowed": 0,
        "wavelet_role_constraint": str(getattr(args, "wavelet_role_constraint", "none")),
        "wavelet_role_constraint_steps": int(role_constraint_steps),
        "wavelet_role_constraint_uses_labels": 0,
        "wavelet_role_constraint_uses_linec_tail_direction": 0,
        **support_info,
        **rbf_info,
        **fourier_info,
        **output_info,
    }
    return row, linec_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workspace-csv", default="")
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--dataset", default="MNIST")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-size", type=int, default=128)
    ap.add_argument("--val-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--mlp-hidden", type=int, default=160)
    ap.add_argument("--hidden-override", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--adamw-foreach", choices=["auto", "true", "false"], default="false")
    ap.add_argument("--workspace-warmup-steps", type=int, default=1)
    ap.add_argument("--workspace-profile-steps", type=int, default=1)
    ap.add_argument("--hardening-epochs", type=int, default=1)
    ap.add_argument("--linec-batch-size", type=int, default=24)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--linec-seeds", default="12319500")
    ap.add_argument(
        "--wavelet-support-repair",
        choices=["none", "scale090", "scale075", "scale050", "scale125", "quantile", "quantile_scale075", "quantile_scale050"],
        default="none",
    )
    ap.add_argument(
        "--wavelet-role-constraint",
        choices=["none", "freeze_linear_readout", "linear_readout_grad050", "linear_readout_grad025", "linear_readout_grad010", "linear_readout_grad005"],
        default="none",
    )
    ap.add_argument(
        "--rbf-center-repair",
        choices=["none", "quantile", "quantile_width075", "quantile_width100", "quantile_width125", "quantile_width175"],
        default="none",
    )
    ap.add_argument(
        "--output-geometry-repair",
        choices=[
            "none",
            "fixed050",
            "fixed025",
            "train_rms_target100",
            "train_rms_target075",
            "train_rms_target050",
            "train_rms_target025",
            "train_entropy_t080_100_else050",
            "train_entropy_t085_100_else050",
            "train_entropy_t090_100_else050",
            "train_entropy_t095_100_else050",
            "train_margin_t015_050_else100",
            "train_margin_t020_050_else100",
            "train_topprob_t020_050_else100",
            "train_topprob_t025_050_else100",
        ],
        default="none",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("compact task-health probe requires CUDA")
    torch.cuda.set_device(device)

    dataset = v1223.v120._canonical_dataset(str(args.dataset))
    load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=int(args.seed))
    data = v1223.v120._load_vision_split(load_args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.val_size))
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)

    mlp = train_mlp_for_reference(args, int(input_dim), int(output_dim), x_train, y_train, x_val, y_val, int(args.seed), device)
    workspace = load_workspace_rows(str(args.workspace_csv))
    task_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    for cid in parse_csv(args.candidates):
        row, lc = train_compact_candidate(args, cid, x_train, y_train, x_val, y_val, int(input_dim), int(output_dim), int(args.seed), device)
        ws = workspace.get((cid, "manual_no_materialize"), {})
        row.update({
            "dataset": dataset,
            "seed": int(args.seed),
            "mlp_val_acc": mlp.get("mlp_val_acc", ""),
            "mean_delta_vs_MLP": fnum(row.get("val_acc")) - fnum(mlp.get("mlp_val_acc")) if math.isfinite(fnum(mlp.get("mlp_val_acc"))) else float("nan"),
            "worst_delta_vs_MLP": fnum(row.get("val_acc")) - fnum(mlp.get("mlp_val_acc")) if math.isfinite(fnum(mlp.get("mlp_val_acc"))) else float("nan"),
            "mlp_NLL": mlp.get("mlp_NLL", ""),
            "NLL_ratio_vs_MLP": fnum(row.get("NLL"), 9.0) / max(1.0e-8, fnum(mlp.get("mlp_NLL"), 9.0)),
            "mlp_step_time_q90_ms": mlp.get("mlp_step_time_q90_ms", ""),
            "train_step_ratio_vs_MLP": fnum(row.get("step_time_q90_ms"), 9.0) / max(1.0e-8, fnum(mlp.get("mlp_step_time_q90_ms"), 9.0)),
            "workspace_manual_gate_pass": int(ws.get("manual_workspace_gate_pass", 0) or 0),
            "workspace_raw_memory_ratio_vs_mlp": ws.get("raw_memory_ratio_vs_mlp", ""),
            "workspace_incremental_memory_ratio_vs_mlp": ws.get("incremental_memory_ratio_vs_mlp", ""),
            "workspace_step_ratio_vs_mlp": ws.get("step_ratio_vs_mlp", ""),
        })
        row["compact_task_health_gate_pass"] = int(
            int(row["workspace_manual_gate_pass"]) == 1
            and fnum(row["mean_delta_vs_MLP"], -999.0) >= -0.05
            and fnum(row["worst_delta_vs_MLP"], -999.0) >= -0.10
            and fnum(row["NLL_ratio_vs_MLP"], 9.0) <= 2.0
            and fnum(row["LineC_pass_rate"], 0.0) >= 0.30
        )
        task_rows.append(row)
        for lrow in lc:
            lrow["dataset"] = dataset
            lrow["seed"] = int(args.seed)
        linec_rows.extend(lc)
        torch.cuda.empty_cache()

    write_rows(out_dir / "v143_nonrat_compact_task_health.csv", task_rows)
    write_rows(out_dir / "v143_nonrat_compact_task_linec.csv", linec_rows)
    summary_rows: list[dict[str, Any]] = []
    for family in sorted({str(r["family"]) for r in task_rows}):
        fam = [r for r in task_rows if str(r["family"]) == family]
        summary_rows.append({
            "stage": "V143_COMPACT_NONRAT_TASK_HEALTH_SUMMARY",
            "family": family,
            "rows": len(fam),
            "workspace_manual_gate_pass_rows": sum(int(r.get("workspace_manual_gate_pass", 0)) for r in fam),
            "compact_task_health_gate_pass_rows": sum(int(r.get("compact_task_health_gate_pass", 0)) for r in fam),
            "best_mean_delta_vs_MLP": max(fnum(r.get("mean_delta_vs_MLP"), -999.0) for r in fam),
            "best_NLL_ratio_vs_MLP": min(fnum(r.get("NLL_ratio_vs_MLP"), 999.0) for r in fam),
            "best_LineC_pass_rate": max(fnum(r.get("LineC_pass_rate"), 0.0) for r in fam),
            "promotion_allowed": 0,
        })
    write_rows(out_dir / "v143_nonrat_compact_task_health_summary.csv", summary_rows)
    route = {
        "stage": "V143_COMPACT_NONRAT_TASK_HEALTH_ROUTE",
        "diagnostic_route": "D6-CompactNonRATTaskHealthProbe",
        "official_route_unchanged": "R4-NonRATSubstrateMissing",
        "candidate_rows": len(task_rows),
        "workspace_manual_gate_pass_rows": sum(int(r.get("workspace_manual_gate_pass", 0)) for r in task_rows),
        "compact_task_health_gate_pass_rows": sum(int(r.get("compact_task_health_gate_pass", 0)) for r in task_rows),
        "official_fms_proof_executed": 0,
        "promotion_allowed": 0,
        "real_short_run_open_allowed": 0,
    }
    (out_dir / "v143_nonrat_compact_task_health_route.json").write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
