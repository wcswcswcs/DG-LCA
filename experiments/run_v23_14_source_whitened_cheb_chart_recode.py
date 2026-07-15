#!/usr/bin/env python3
"""DG-KAN v23.14 source-whitened Chebyshev chart recode runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_08_downstream_coupled_edge_residual_inverse_flow as v2308
import experiments.run_v23_12_total_audit_feature_learning_tangent_escape as v2312
import experiments.run_v23_13_cheb_domain_transport_representation_geometry as v2313
from dgkan.models.fc_purekan_primitives import _basis_eval


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.14_SourceWhitenedChebChartRecode_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.14_SourceWhitenedChebChartRecode_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.14_SourceWhitenedChebChartRecode_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2314_OUT_ROOT", str(ROOT / "results/v23_14_source_whitened_cheb_chart_recode"))).resolve()
FORMAL_ROOT = ROOT / "results/v23_09_part_b_repair_tol001_formal_15seed80"

V2312_STAGE2R = ROOT / "results/v23_12_total_audit_feature_learning_tangent_escape_stage2r_reduced_s5_80/stage2_total_audit_summary.json"
V2313_STAGE1F = ROOT / "results/v23_13_function_preserving_gain_recode_reduced_s5/stage1f_gain_recode_summary.json"
V2313_NEXT = FORMAL_ROOT / "next_actions_for_v23_13.json"

CHART_SCHEMES = [
    "C39_Candidate_source_whitened_cheb_chart_recode_total_audit",
    "C40_Control_identity_chart_total_audit",
    "C41_Control_permuted_source_whitened_chart_total_audit",
]

FRAME_SCHEMES = [
    "C42_Candidate_source_gram_pca_cheb_frame_recode_total_audit",
    "C43_Control_identity_frame_total_audit",
    "C44_Control_random_orthogonal_cheb_frame_recode_total_audit",
]

DYADIC_SCHEMES = [
    "C45_Candidate_source_rms_dyadic_cheb_frame_recode_total_audit",
    "C43_Control_identity_frame_total_audit",
    "C47_Control_random_dyadic_cheb_frame_recode_total_audit",
]

STAGE2_SCHEMES = [
    "C48_Candidate_C45_dyadic_frame_then_terminal_C15_total_audit",
    "C49_Control_C43_identity_frame_then_terminal_C15_total_audit",
    "C50_Control_C47_random_dyadic_frame_then_terminal_C15_total_audit",
    "C51_Control_C45_dyadic_frame_no_inverse_total_audit",
]

AUDIT_DEFAULTS = {
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "metric_winner_selection_used": 0,
    "candidate_update_selection_used": 0,
    "new_edge_function_added": 0,
    "external_product_feature_used": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
    "structure_transform_in_forward": 0,
    "basis_chart_transform_used": 0,
    "basis_frame_transform_used": 0,
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def command_text() -> str:
    cmd = f"{sys.executable} {rel(RUNNER)} {' '.join(sys.argv[1:])}"
    env = os.environ.get("V2314_OUT_ROOT")
    return f"V2314_OUT_ROOT={env} {cmd}" if env else cmd


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    try:
        return int(round(fval(value, float(default))))
    except Exception:
        return int(default)


def median(values: list[Any]) -> float:
    vals = sorted(fval(v, float("nan")) for v in values if v not in (None, ""))
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float(0.5 * (vals[mid - 1] + vals[mid]))


def float_items(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_logs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text("# DG-KAN v23.14 执行日志\n", encoding="utf-8")
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text("# DG-KAN v23.14 实验结果复盘\n", encoding="utf-8")


def append_exec(title: str, command: str, *, files: str, gpu: str, note: str) -> None:
    ensure_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title} done\n\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        fh.write(f"- files: `{files}`\n")
        fh.write(f"- note: {note}\n")


def append_recap(title: str, data: dict[str, Any]) -> None:
    ensure_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
        fh.write("\n```\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys: list[str] = []
    for row in enriched:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(enriched)
    return path


def strict_total_audit_pass(before_guard: dict[str, float], after_guard: dict[str, float], budget: float) -> tuple[int, dict[str, float]]:
    debt = v2308.debt_deltas(before_guard, after_guard)
    component_ok = int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0)
    return int(v2308.no_debt_ok(debt, float(budget)) and component_ok), debt


def model_seed_for(task: str, seed: int, args: argparse.Namespace) -> int:
    return 23140000 + int(seed) * 1009 + sum(ord(ch) for ch in str(task)) + int(args.depth) * 17


def install_chart_methods(model: Any) -> None:
    device = model.coeffs[0].device
    dtype = model.coeffs[0].dtype
    dims = [int(d) for d in model.dims[:-1]]
    model.chart_centers = [torch.zeros(dim, device=device, dtype=dtype) for dim in dims]
    model.chart_inv_scales = [torch.ones(dim, device=device, dtype=dtype) for dim in dims]
    model.chart_frames = [torch.eye(int(model.k), device=device, dtype=dtype) for _ in dims]

    def chart_basis(self: Any, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        idx = int(layer_idx)
        center = self.chart_centers[idx].to(device=h.device, dtype=h.dtype)
        inv_scale = self.chart_inv_scales[idx].to(device=h.device, dtype=h.dtype)
        z = torch.tanh((h - center.reshape(1, -1)) * float(self.basis_input_gain) * inv_scale.reshape(1, -1))
        base = _basis_eval(z, self.basis_name, self.k, self.centers, self.scales)
        frame = self.chart_frames[idx].to(device=h.device, dtype=h.dtype)
        return torch.einsum("bik,kl->bil", base, frame)

    def layer_phi(self: Any, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        b = self.chart_basis(h, int(layer_idx)) / math.sqrt(max(1, int(self.dims[int(layer_idx)])))
        return b.reshape(int(h.shape[0]), -1).to(dtype=torch.float64)

    def forward_with_activations(self: Any, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = x
        activations = [h]
        for layer_idx, coeff in enumerate(self.coeffs):
            b = self.chart_basis(h, int(layer_idx)) / math.sqrt(max(1, int(self.dims[int(layer_idx)])))
            h = torch.einsum("bik,iok->bo", b, coeff)
            activations.append(h)
        return h, activations

    def forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        logits, _acts = self.forward_with_activations(x)
        return logits

    model.chart_basis = MethodType(chart_basis, model)
    model.layer_phi = MethodType(layer_phi, model)
    model.forward_with_activations = MethodType(forward_with_activations, model)
    model.forward = MethodType(forward, model)


def chart_state(model: Any) -> dict[str, Any]:
    return {
        "state": v2308.model_state(model),
        "basis_input_gain": float(model.basis_input_gain),
        "chart_centers": [c.detach().clone() for c in model.chart_centers],
        "chart_inv_scales": [s.detach().clone() for s in model.chart_inv_scales],
        "chart_frames": [q.detach().clone() for q in model.chart_frames],
    }


def load_chart_state(model: Any, state: dict[str, Any]) -> None:
    model.basis_input_gain = float(state["basis_input_gain"])
    v2308.load_model_state(model, state["state"])
    device = model.coeffs[0].device
    dtype = model.coeffs[0].dtype
    model.chart_centers = [c.detach().clone().to(device=device, dtype=dtype) for c in state["chart_centers"]]
    model.chart_inv_scales = [s.detach().clone().to(device=device, dtype=dtype) for s in state["chart_inv_scales"]]
    model.chart_frames = [q.detach().clone().to(device=device, dtype=dtype) for q in state.get("chart_frames", [torch.eye(int(model.k)) for _ in model.chart_centers])]


def interpolated_chart_state(before: dict[str, Any], after: dict[str, Any], alpha: float) -> dict[str, Any]:
    a = float(alpha)
    if abs(a) <= 1.0e-12:
        return {
            "state": {name: tensor.detach().clone() for name, tensor in before["state"].items()},
            "basis_input_gain": float(before["basis_input_gain"]),
            "chart_centers": [c.detach().clone() for c in before["chart_centers"]],
            "chart_inv_scales": [s.detach().clone() for s in before["chart_inv_scales"]],
            "chart_frames": [q.detach().clone() for q in before["chart_frames"]],
        }
    if abs(a - 1.0) <= 1.0e-12:
        return {
            "state": {name: tensor.detach().clone() for name, tensor in after["state"].items()},
            "basis_input_gain": float(after["basis_input_gain"]),
            "chart_centers": [c.detach().clone() for c in after["chart_centers"]],
            "chart_inv_scales": [s.detach().clone() for s in after["chart_inv_scales"]],
            "chart_frames": [q.detach().clone() for q in after["chart_frames"]],
        }
    return {
        "state": v2312.interpolated_state(before["state"], after["state"], a),
        "basis_input_gain": float(before["basis_input_gain"]) + a * (float(after["basis_input_gain"]) - float(before["basis_input_gain"])),
        "chart_centers": [b + a * (c - b) for b, c in zip(before["chart_centers"], after["chart_centers"])],
        "chart_inv_scales": [b + a * (s - b) for b, s in zip(before["chart_inv_scales"], after["chart_inv_scales"])],
        "chart_frames": [b + a * (q - b) for b, q in zip(before["chart_frames"], after["chart_frames"])],
    }


def source_chart_stats(model: Any, acts_train: list[torch.Tensor], scheme: str, seed: int, task: str, args: argparse.Namespace) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    centers: list[torch.Tensor] = []
    inv_scales: list[torch.Tensor] = []
    center_abs: list[float] = []
    inv_vals: list[float] = []
    gen = torch.Generator(device=acts_train[0].device).manual_seed(23141000 + int(seed) * 1009 + sum(ord(c) for c in str(task)))
    for layer_idx, act in enumerate(acts_train[:-1]):
        aa = act.detach().to(dtype=torch.float64)
        center = aa.mean(dim=0)
        std = aa.std(dim=0, unbiased=False).clamp_min(float(args.chart_scale_floor))
        inv = (1.0 / std).clamp(max=float(args.chart_inv_scale_cap))
        if scheme.startswith("C41"):
            perm = torch.randperm(int(center.numel()), device=center.device, generator=gen)
            center = center[perm]
            inv = inv[perm]
        centers.append(center.to(device=model.coeffs[0].device, dtype=model.coeffs[0].dtype))
        inv_scales.append(inv.to(device=model.coeffs[0].device, dtype=model.coeffs[0].dtype))
        center_abs.extend([float(v) for v in center.abs().detach().cpu().reshape(-1)])
        inv_vals.extend([float(v) for v in inv.detach().cpu().reshape(-1)])
    return centers, inv_scales, {
        "chart_center_abs_median": median(center_abs),
        "chart_center_abs_max": max(center_abs or [0.0]),
        "chart_inv_scale_median": median(inv_vals),
        "chart_inv_scale_max": max(inv_vals or [0.0]),
    }


def base_cheb_basis(model: Any, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
    idx = int(layer_idx)
    center = model.chart_centers[idx].to(device=h.device, dtype=h.dtype)
    inv_scale = model.chart_inv_scales[idx].to(device=h.device, dtype=h.dtype)
    z = torch.tanh((h - center.reshape(1, -1)) * float(model.basis_input_gain) * inv_scale.reshape(1, -1))
    return _basis_eval(z, model.basis_name, model.k, model.centers, model.scales)


def orient_orthogonal_columns(q: torch.Tensor) -> torch.Tensor:
    out = q.detach().clone()
    with torch.no_grad():
        idx = out.abs().argmax(dim=0)
        cols = torch.arange(int(out.shape[1]), device=out.device)
        signs = torch.sign(out[idx, cols])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        out = out * signs.reshape(1, -1)
    return out


def source_gram_pca_frames(model: Any, acts_train: list[torch.Tensor], scheme: str, seed: int, task: str) -> tuple[list[torch.Tensor], dict[str, float]]:
    frames: list[torch.Tensor] = []
    offdiag_before: list[float] = []
    offdiag_after: list[float] = []
    gen = torch.Generator(device=acts_train[0].device).manual_seed(23142000 + int(seed) * 1009 + sum(ord(c) for c in str(task)))
    for layer_idx, act in enumerate(acts_train[:-1]):
        if scheme.startswith("C44"):
            raw = torch.randn((int(model.k), int(model.k)), device=act.device, dtype=torch.float64, generator=gen)
            q, _r = torch.linalg.qr(raw)
            q = orient_orthogonal_columns(q)
            frames.append(q.to(device=model.coeffs[0].device, dtype=model.coeffs[0].dtype))
            offdiag_before.append(0.0)
            offdiag_after.append(0.0)
            continue
        b = base_cheb_basis(model, act.detach(), int(layer_idx)).detach().to(dtype=torch.float64)
        flat = b.reshape(-1, int(model.k))
        gram = flat.T @ flat / max(1, int(flat.shape[0]))
        eigvals, eigvecs = torch.linalg.eigh(gram)
        order = torch.argsort(eigvals, descending=True)
        q = orient_orthogonal_columns(eigvecs[:, order])
        rotated = q.T @ gram @ q
        diag_norm = torch.diag(gram).norm().clamp_min(1.0e-12)
        offdiag_before.append(float((gram - torch.diag(torch.diag(gram))).norm().div(diag_norm).detach().cpu().item()))
        offdiag_after.append(float((rotated - torch.diag(torch.diag(rotated))).norm().div(torch.diag(rotated).norm().clamp_min(1.0e-12)).detach().cpu().item()))
        frames.append(q.to(device=model.coeffs[0].device, dtype=model.coeffs[0].dtype))
    return frames, {
        "frame_source_gram_offdiag_before_median": median(offdiag_before),
        "frame_source_gram_offdiag_after_median": median(offdiag_after),
    }


def source_rms_dyadic_frames(model: Any, acts_train: list[torch.Tensor], scheme: str, seed: int, task: str, args: argparse.Namespace) -> tuple[list[torch.Tensor], dict[str, float]]:
    frames: list[torch.Tensor] = []
    exp_abs: list[float] = []
    rms_spread_before: list[float] = []
    rms_spread_after: list[float] = []
    gen = torch.Generator(device=acts_train[0].device).manual_seed(23145000 + int(seed) * 1009 + sum(ord(c) for c in str(task)))
    for layer_idx, act in enumerate(acts_train[:-1]):
        b = base_cheb_basis(model, act.detach(), int(layer_idx)).detach().to(dtype=torch.float64)
        flat = b.reshape(-1, int(model.k))
        rms = flat.square().mean(dim=0).sqrt().clamp_min(1.0e-12)
        spread_before = float((rms.max() / rms.min().clamp_min(1.0e-12)).detach().cpu().item())
        if scheme.startswith("C47"):
            cap = int(args.dyadic_exp_cap)
            exp = torch.randint(low=-cap, high=cap + 1, size=(int(model.k),), device=act.device, generator=gen).to(dtype=torch.float64)
        else:
            target = torch.exp(torch.log(rms).mean())
            raw = torch.log2(target / rms)
            exp = raw.round().clamp(-int(args.dyadic_exp_cap), int(args.dyadic_exp_cap))
        scale = torch.pow(torch.tensor(2.0, device=act.device, dtype=torch.float64), exp)
        frame = torch.diag(scale)
        frames.append(frame.to(device=model.coeffs[0].device, dtype=model.coeffs[0].dtype))
        rms_after = (rms * scale.abs()).clamp_min(1.0e-12)
        rms_spread_before.append(spread_before)
        rms_spread_after.append(float((rms_after.max() / rms_after.min()).detach().cpu().item()))
        exp_abs.extend([float(v) for v in exp.abs().detach().cpu().reshape(-1)])
    return frames, {
        "frame_source_gram_offdiag_before_median": 0.0,
        "frame_source_gram_offdiag_after_median": 0.0,
        "frame_dyadic_exp_abs_median": median(exp_abs),
        "frame_dyadic_exp_abs_max": max(exp_abs or [0.0]),
        "frame_rms_spread_before_median": median(rms_spread_before),
        "frame_rms_spread_after_median": median(rms_spread_after),
    }


def apply_orthogonal_frame_recode(model: Any, x: torch.Tensor, xg: torch.Tensor, scheme: str, seed: int, task: str, args: argparse.Namespace) -> dict[str, Any]:
    with torch.no_grad():
        original_logits_train, original_acts_train = model.forward_with_activations(x)
        original_logits_guard, original_acts_guard = model.forward_with_activations(xg)
        if scheme.startswith("C45") or scheme.startswith("C47"):
            frames, frame_diag = source_rms_dyadic_frames(model, original_acts_train, scheme, int(seed), task, args)
        else:
            frames, frame_diag = source_gram_pca_frames(model, original_acts_train, scheme, int(seed), task)
        old_coeffs = [coeff.detach().clone().to(dtype=torch.float64) for coeff in model.coeffs]
        model.chart_frames = frames
        for layer_idx, coeff in enumerate(model.coeffs):
            q = frames[int(layer_idx)].to(device=coeff.device, dtype=torch.float64)
            new_coeff = torch.empty_like(old_coeffs[int(layer_idx)])
            diag = torch.diag(q)
            is_diagonal = bool(torch.allclose(q, torch.diag(diag), atol=0.0, rtol=0.0))
            if is_diagonal:
                new_coeff = old_coeffs[int(layer_idx)] / diag.reshape(1, 1, -1).clamp_min(1.0e-12)
            else:
                for in_idx in range(int(coeff.shape[0])):
                    c = old_coeffs[int(layer_idx)][int(in_idx)].permute(1, 0).contiguous()
                    cp = torch.linalg.solve(q, c)
                    new_coeff[int(in_idx)] = cp.permute(1, 0).contiguous()
            coeff.copy_(new_coeff.to(dtype=coeff.dtype))
        recoded_logits_train, recoded_acts_train = model.forward_with_activations(x)
        recoded_logits_guard, recoded_acts_guard = model.forward_with_activations(xg)
        layer_fit_rel: list[float] = []
        for layer_idx in range(len(model.coeffs)):
            target = original_acts_train[int(layer_idx) + 1].detach().to(dtype=torch.float64)
            got = recoded_acts_train[int(layer_idx) + 1].detach().to(dtype=torch.float64)
            layer_fit_rel.append(float((got - target).norm().div(target.norm().clamp_min(1.0e-12)).detach().cpu().item()))
        train_logit_rel = float(
            (recoded_logits_train.detach().to(dtype=torch.float64) - original_logits_train.detach().to(dtype=torch.float64))
            .norm()
            .div(original_logits_train.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12))
            .detach()
            .cpu()
            .item()
        )
        guard_logit_rel = float(
            (recoded_logits_guard.detach().to(dtype=torch.float64) - original_logits_guard.detach().to(dtype=torch.float64))
            .norm()
            .div(original_logits_guard.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12))
            .detach()
            .cpu()
            .item()
        )
        activation_drift = v2312.activation_drift_stats(original_acts_guard, recoded_acts_guard)
        identity_frames = [torch.eye(int(model.k), device=model.coeffs[0].device, dtype=model.coeffs[0].dtype) for _ in model.chart_frames]
        saved_frames = [q.detach().clone() for q in model.chart_frames]
        design_vals: list[float] = []
        for layer_idx in range(len(model.coeffs)):
            model.chart_frames = identity_frames
            phi_before = model.layer_phi(original_acts_guard[int(layer_idx)].detach(), int(layer_idx))
            model.chart_frames = saved_frames
            phi_after = model.layer_phi(recoded_acts_guard[int(layer_idx)].detach(), int(layer_idx))
            design_vals.append(float((phi_after - phi_before).norm().div(phi_before.norm().clamp_min(1.0e-12)).detach().cpu().item()))
        model.chart_frames = saved_frames
    return {
        **frame_diag,
        "chart_center_abs_median": 0.0,
        "chart_center_abs_max": 0.0,
        "chart_inv_scale_median": 1.0,
        "chart_inv_scale_max": 1.0,
        "chart_layer_fit_rel_median": median(layer_fit_rel),
        "chart_layer_fit_rel_max": max(layer_fit_rel or [0.0]),
        "chart_layer_fit_rel_curve": json.dumps(layer_fit_rel),
        "chart_phi_condition_median": 0.0,
        "chart_phi_condition_max": 0.0,
        "chart_train_logit_rel_error": train_logit_rel,
        "chart_guard_logit_rel_error": guard_logit_rel,
        "chart_raw_design_drift_median": median(design_vals),
        "chart_raw_design_drift_max": max(design_vals or [0.0]),
        "activation_drift_median": activation_drift["activation_drift_median"],
        "activation_drift_max": activation_drift["activation_drift_max"],
        "activation_drift_layer_count": activation_drift["activation_drift_layer_count"],
    }


def chart_drift_stats(model: Any, xg: torch.Tensor, before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, float]:
    with torch.no_grad():
        load_chart_state(model, before_state)
        _lb, acts_before = model.forward_with_activations(xg)
        phis_before = [model.layer_phi(acts_before[int(layer_idx)].detach(), int(layer_idx)) for layer_idx in range(len(model.coeffs))]
        load_chart_state(model, after_state)
        _la, acts_after = model.forward_with_activations(xg)
        phis_after = [model.layer_phi(acts_after[int(layer_idx)].detach(), int(layer_idx)) for layer_idx in range(len(model.coeffs))]
    act = v2312.activation_drift_stats(acts_before, acts_after)
    design_vals = [
        float((pa - pb).norm().div(pb.norm().clamp_min(1.0e-12)).detach().cpu().item())
        for pb, pa in zip(phis_before, phis_after)
    ]
    return {
        "activation_drift_median": act["activation_drift_median"],
        "activation_drift_max": act["activation_drift_max"],
        "activation_drift_layer_count": act["activation_drift_layer_count"],
        "chart_design_drift_median": median(design_vals),
        "chart_design_drift_max": max(design_vals or [0.0]),
    }


def apply_source_whiten_chart_recode(model: Any, x: torch.Tensor, xg: torch.Tensor, scheme: str, seed: int, task: str, args: argparse.Namespace) -> dict[str, Any]:
    with torch.no_grad():
        original_logits_train, original_acts_train = model.forward_with_activations(x)
        original_logits_guard, original_acts_guard = model.forward_with_activations(xg)
        centers, inv_scales, stat_diag = source_chart_stats(model, original_acts_train, scheme, int(seed), task, args)
        model.chart_centers = centers
        model.chart_inv_scales = inv_scales
        layer_fit_rel: list[float] = []
        layer_phi_cond: list[float] = []
        for layer_idx, coeff in enumerate(model.coeffs):
            _cur_logits, cur_acts = model.forward_with_activations(x)
            h_current = cur_acts[int(layer_idx)].detach()
            target = original_acts_train[int(layer_idx) + 1].detach()
            phi = model.layer_phi(h_current, int(layer_idx))
            new_coeff = v2313.solve_layer_coefficients(phi, target, int(coeff.shape[0]), int(coeff.shape[1]), int(coeff.shape[2]), float(args.chart_ridge))
            coeff.copy_(new_coeff.to(dtype=coeff.dtype))
            _after_logits, after_acts = model.forward_with_activations(x)
            fit_err = (after_acts[int(layer_idx) + 1].detach().to(dtype=torch.float64) - target.to(dtype=torch.float64)).norm()
            fit_den = target.to(dtype=torch.float64).norm().clamp_min(1.0e-12)
            layer_fit_rel.append(float((fit_err / fit_den).detach().cpu().item()))
            try:
                vals = torch.linalg.svdvals(phi.to(dtype=torch.float64))
                cond = float((vals.max().clamp_min(1.0e-12) / vals.min().clamp_min(1.0e-12)).detach().cpu().item())
            except Exception:
                cond = 0.0
            layer_phi_cond.append(cond)
        recoded_logits_train, recoded_acts_train = model.forward_with_activations(x)
        recoded_logits_guard, recoded_acts_guard = model.forward_with_activations(xg)
        train_logit_rel = float(
            (recoded_logits_train.detach().to(dtype=torch.float64) - original_logits_train.detach().to(dtype=torch.float64))
            .norm()
            .div(original_logits_train.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12))
            .detach()
            .cpu()
            .item()
        )
        guard_logit_rel = float(
            (recoded_logits_guard.detach().to(dtype=torch.float64) - original_logits_guard.detach().to(dtype=torch.float64))
            .norm()
            .div(original_logits_guard.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12))
            .detach()
            .cpu()
            .item()
        )
        activation_drift = v2312.activation_drift_stats(original_acts_guard, recoded_acts_guard)
        design_vals: list[float] = []
        saved_centers = [c.detach().clone() for c in model.chart_centers]
        saved_inv = [s.detach().clone() for s in model.chart_inv_scales]
        identity_centers = [torch.zeros_like(c) for c in saved_centers]
        identity_inv = [torch.ones_like(s) for s in saved_inv]
        for layer_idx in range(len(model.coeffs)):
            model.chart_centers = identity_centers
            model.chart_inv_scales = identity_inv
            phi_before = model.layer_phi(original_acts_guard[int(layer_idx)].detach(), int(layer_idx))
            model.chart_centers = saved_centers
            model.chart_inv_scales = saved_inv
            phi_after = model.layer_phi(recoded_acts_guard[int(layer_idx)].detach(), int(layer_idx))
            design_vals.append(float((phi_after - phi_before).norm().div(phi_before.norm().clamp_min(1.0e-12)).detach().cpu().item()))
        model.chart_centers = saved_centers
        model.chart_inv_scales = saved_inv
    return {
        **stat_diag,
        "chart_layer_fit_rel_median": median(layer_fit_rel),
        "chart_layer_fit_rel_max": max(layer_fit_rel or [0.0]),
        "chart_layer_fit_rel_curve": json.dumps(layer_fit_rel),
        "chart_phi_condition_median": median(layer_phi_cond),
        "chart_phi_condition_max": max(layer_phi_cond or [0.0]),
        "chart_train_logit_rel_error": train_logit_rel,
        "chart_guard_logit_rel_error": guard_logit_rel,
        "chart_raw_design_drift_median": median(design_vals),
        "chart_raw_design_drift_max": max(design_vals or [0.0]),
        "activation_drift_median": activation_drift["activation_drift_median"],
        "activation_drift_max": activation_drift["activation_drift_max"],
        "activation_drift_layer_count": activation_drift["activation_drift_layer_count"],
    }


def project_chart_state(
    model: Any,
    original_state: dict[str, Any],
    raw_state: dict[str, Any],
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    guard_original: dict[str, float],
    args: argparse.Namespace,
) -> tuple[float, str, dict[str, Any], dict[str, float], dict[str, float]]:
    selected_alpha = 0.0
    selected_reason = "alpha_zero_fallback"
    selected_state = original_state
    load_chart_state(model, original_state)
    selected_train = v2308.actual_metrics(model, x, y)
    selected_guard = guard_original
    for alpha in float_items(str(args.chart_alphas)):
        cand_state = interpolated_chart_state(original_state, raw_state, float(alpha))
        load_chart_state(model, cand_state)
        train_cand = v2308.actual_metrics(model, x, y)
        guard_cand = v2308.actual_metrics(model, xg, yg)
        cand_pass, _debt = strict_total_audit_pass(guard_original, guard_cand, float(args.no_debt_budget))
        if cand_pass:
            selected_alpha = float(alpha)
            selected_reason = "alpha_zero_fallback" if abs(float(alpha)) <= 1.0e-12 else "first_total_audit_alpha"
            selected_state = cand_state
            selected_train = train_cand
            selected_guard = guard_cand
            break
    load_chart_state(model, selected_state)
    return selected_alpha, selected_reason, selected_state, selected_train, selected_guard


def stage2_f_args(args: argparse.Namespace, *, alphas: str | None = None) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.part_f_steps = int(args.stage2_steps)
    out.part_f_residual_target = str(args.stage2_residual_target)
    out.part_f_lambda = float(args.stage2_lambda)
    out.part_f_alphas = str(alphas if alphas is not None else args.stage2_alphas)
    out.part_f_delta_scale = float(args.stage2_delta_scale)
    out.part_f_delta_sign = float(args.stage2_delta_sign)
    out.part_f_trust_mode = str(args.stage2_trust_mode)
    out.part_f_trust_tolerance = float(args.stage2_trust_tolerance)
    out.part_f_local_reference_target = str(args.stage2_local_reference_target)
    out.part_f_functionalgram_lr = float(args.stage2_functionalgram_lr)
    out.part_f_adam_lr = float(args.stage2_adam_lr)
    out.part_f_solver = str(args.stage2_solver)
    out.part_f_sketch_rank = int(args.stage2_sketch_rank)
    out.part_f_activation_drift_cap = float(args.stage2_activation_drift_cap)
    out.part_f_alignment_min = -2.0
    out.part_f_alignment_soft_floor = -2.0
    out.part_f_alignment_soft_ceiling = 1.0
    out.part_f_alignment_soft_min_scale = 0.25
    out.part_f_alignment_soft_power = 0.5
    out.part_f_tail99_step_budget = -1.0
    out.part_f_tail99_cumulative_budget = -1.0
    out.part_f_tail_correction_scales = "0.25,0.125,0.0625,0"
    out.part_f_tail_correction_target = "tail_weighted_norm"
    out.part_f_tail_correction_layer_group = "topdown"
    out.part_f_tail_correction_transport_min = 0.90
    out.part_f_checkpoint = str(args.checkpoint)
    return out


def frame_scheme_for_stage2(scheme: str) -> str:
    if scheme.startswith("C48") or scheme.startswith("C51"):
        return "C45_Candidate_source_rms_dyadic_cheb_frame_recode_total_audit"
    if scheme.startswith("C49"):
        return "C43_Control_identity_frame_total_audit"
    if scheme.startswith("C50"):
        return "C47_Control_random_dyadic_cheb_frame_recode_total_audit"
    raise ValueError(f"unknown stage2 scheme {scheme}")


def prepare_stage2_frame_context(scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    x, y, xg, yg = v2308.v2307.visual_data(task, int(seed), args, device)
    classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
    model_seed = model_seed_for(task, int(seed), args)
    model = v2308.v2307.make_model(str(args.basis_key), int(args.depth), int(x.shape[1]), classes, model_seed, args, device, basis_input_gain=float(args.part_d_basis_input_gain))
    install_chart_methods(model)
    ckpt = v2308.v2307.train_checkpoint(model, x, y, str(args.checkpoint), int(seed), args)
    original_state = chart_state(model)
    train_original = v2308.actual_metrics(model, x, y)
    guard_original = v2308.actual_metrics(model, xg, yg)
    frame_scheme = frame_scheme_for_stage2(scheme)
    if not frame_scheme.startswith("C43"):
        _diag = apply_orthogonal_frame_recode(model, x, xg, frame_scheme, int(seed), task, args)
    else:
        _diag = {
            "chart_layer_fit_rel_median": 0.0,
            "chart_layer_fit_rel_max": 0.0,
            "chart_raw_design_drift_median": 0.0,
            "frame_rms_spread_before_median": 0.0,
            "frame_rms_spread_after_median": 0.0,
            "frame_dyadic_exp_abs_median": 0.0,
        }
    frame_state = chart_state(model)
    train_frame = v2308.actual_metrics(model, x, y)
    guard_frame = v2308.actual_metrics(model, xg, yg)
    frame_drift = chart_drift_stats(model, xg, original_state, frame_state)
    frame_pass, frame_debt = strict_total_audit_pass(guard_original, guard_frame, float(args.no_debt_budget))
    return {
        "model": model,
        "x": x,
        "y": y,
        "xg": xg,
        "yg": yg,
        "ckpt": ckpt,
        "classes": classes,
        "model_seed": model_seed,
        "frame_scheme": frame_scheme,
        "original_state": original_state,
        "frame_state": frame_state,
        "train_original": train_original,
        "guard_original": guard_original,
        "train_frame": train_frame,
        "guard_frame": guard_frame,
        "frame_diag": _diag,
        "frame_drift": frame_drift,
        "frame_total_audit_pass": frame_pass,
        "frame_debt": frame_debt,
    }


def stage0_lock(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    required = {
        "plan": PLAN,
        "runner": RUNNER,
        "v23_12_stage2r": V2312_STAGE2R,
        "v23_13_stage1f": V2313_STAGE1F,
        "v23_13_next": V2313_NEXT,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    stage2r = read_json(V2312_STAGE2R)
    stage1f = read_json(V2313_STAGE1F)
    nxt13 = read_json(V2313_NEXT)
    checks = {
        "no_missing_required_artifacts": not missing,
        "v23_12_stage2r_failed": ival(stage2r.get("gate_pass"), -1) == 0,
        "v23_12_stage2r_no_fake": bool(stage2r.get("checks", {}).get("no_fake_data")),
        "v23_12_stage2r_no_held": bool(stage2r.get("checks", {}).get("no_held_test_usage")),
        "v23_13_stage1f_failed": ival(stage1f.get("gate_pass"), -1) == 0,
        "v23_13_stage1f_no_fake": bool(stage1f.get("checks", {}).get("no_fake_data")),
        "v23_13_stage1f_no_held": bool(stage1f.get("checks", {}).get("no_held_test_usage")),
        "v23_13_stage1f_blocker_locked": str(stage1f.get("dominant_blocker")) == "c36_no_basis_design_drift",
        "v23_13_next_forbids_promotion": ival(nxt13.get("promotion_allowed"), -1) == 0,
    }
    gate = int(all(checks.values()))
    summary = {
        "part": "0",
        "route": "V2314EvidenceLockPass" if gate else "V2314EvidenceLockFailed",
        "gate_pass": gate,
        "dominant_blocker": "none" if gate else "missing_or_inconsistent_prior_evidence",
        "checks": checks,
        "missing": missing,
        "locked_metrics": {
            "v23_12_c28_coverage": stage2r.get("metrics", {}).get("c28_coverage"),
            "v23_12_c28_minus_c30": stage2r.get("metrics", {}).get("c28_minus_c30"),
            "v23_12_c28_minus_c31": stage2r.get("metrics", {}).get("c28_minus_c31"),
            "v23_13_c36_abs_coverage": stage1f.get("metrics", {}).get("c36_abs_coverage"),
            "v23_13_c36_design_drift": stage1f.get("metrics", {}).get("c36_design_drift"),
            "v23_13_c36_alpha_selected_median": stage1f.get("metrics", {}).get("c36_alpha_selected_median"),
        },
        "hashes": {name: sha256_file(path) for name, path in required.items() if path.exists()},
        "next_action": "run_stage1g_source_whitened_chart_recode" if gate else "repair_evidence_lock",
    }
    out = write_json(OUT_ROOT / "stage0_evidence_lock_summary.json", summary)
    append_exec("Stage 0 evidence lock", command_text(), files=rel(out), gpu=str(args.device), note=f"gate={gate}; blocker={summary['dominant_blocker']}")
    append_recap("Stage 0 evidence lock", summary)
    return summary


def run_chart_row(scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    start = time.time()
    try:
        x, y, xg, yg = v2308.v2307.visual_data(task, int(seed), args, device)
        classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = model_seed_for(task, int(seed), args)
        model = v2308.v2307.make_model(str(args.basis_key), int(args.depth), int(x.shape[1]), classes, model_seed, args, device, basis_input_gain=float(args.part_d_basis_input_gain))
        install_chart_methods(model)
        ckpt = v2308.v2307.train_checkpoint(model, x, y, str(args.checkpoint), int(seed), args)
        if int(args.frame_eval_float64):
            model = model.to(dtype=torch.float64)
            x = x.to(dtype=torch.float64)
            xg = xg.to(dtype=torch.float64)
            install_chart_methods(model)
        original_state = chart_state(model)
        train_original = v2308.actual_metrics(model, x, y)
        guard_original = v2308.actual_metrics(model, xg, yg)
        if scheme.startswith("C40"):
            raw_state = original_state
            raw_train = train_original
            raw_guard = guard_original
            diag = {
                "chart_center_abs_median": 0.0,
                "chart_center_abs_max": 0.0,
                "chart_inv_scale_median": 1.0,
                "chart_inv_scale_max": 1.0,
                "chart_layer_fit_rel_median": 0.0,
                "chart_layer_fit_rel_max": 0.0,
                "chart_layer_fit_rel_curve": "[]",
                "chart_phi_condition_median": 0.0,
                "chart_phi_condition_max": 0.0,
                "chart_train_logit_rel_error": 0.0,
                "chart_guard_logit_rel_error": 0.0,
                "chart_raw_design_drift_median": 0.0,
                "chart_raw_design_drift_max": 0.0,
                "activation_drift_median": 0.0,
                "activation_drift_max": 0.0,
                "activation_drift_layer_count": len(model.coeffs),
            }
        else:
            diag = apply_source_whiten_chart_recode(model, x, xg, scheme, int(seed), task, args)
            raw_state = chart_state(model)
            raw_train = v2308.actual_metrics(model, x, y)
            raw_guard = v2308.actual_metrics(model, xg, yg)
        raw_pass, raw_debt = strict_total_audit_pass(guard_original, raw_guard, float(args.no_debt_budget))
        alpha, reason, selected_state, final_train, final_guard = project_chart_state(model, original_state, raw_state, x, y, xg, yg, guard_original, args)
        selected_drift = chart_drift_stats(model, xg, original_state, selected_state)
        load_chart_state(model, selected_state)
        debt = v2308.debt_deltas(guard_original, final_guard)
        no_debt = v2308.no_debt_ok(debt, float(args.no_debt_budget))
        return {
            **AUDIT_DEFAULTS,
            "part": "1G",
            "status": "ok",
            "scheme": scheme,
            "task": task,
            "seed": int(seed),
            "basis_key": str(args.basis_key),
            "basis_family": str(model.basis_name),
            "depth": int(args.depth),
            "train_size": int(args.train_size),
            "guard_size": int(args.guard_size),
            "checkpoint_name": str(args.checkpoint),
            "checkpoint_optimizer": ckpt.get("checkpoint_optimizer", ""),
            "model_seed": model_seed,
            "basis_chart_transform_used": int(not scheme.startswith("C40")),
            "chart_alpha_candidates": str(args.chart_alphas),
            "chart_alpha_selected": alpha,
            "chart_alpha_selection_reason": reason,
            **diag,
            "selected_activation_drift_median": selected_drift["activation_drift_median"],
            "selected_activation_drift_max": selected_drift["activation_drift_max"],
            "selected_chart_design_drift_median": selected_drift["chart_design_drift_median"],
            "selected_chart_design_drift_max": selected_drift["chart_design_drift_max"],
            "original_guard_coverage": guard_original["coverage"],
            "raw_chart_guard_coverage": raw_guard["coverage"],
            "final_guard_coverage": final_guard["coverage"],
            "total_C2_coverage_improvement": final_guard["coverage"] - guard_original["coverage"],
            "raw_chart_total_C2_coverage_improvement": raw_guard["coverage"] - guard_original["coverage"],
            "total_C2_accuracy_improvement": final_guard["accuracy"] - guard_original["accuracy"],
            "total_guard_nll_delta": final_guard["loss"] - guard_original["loss"],
            "total_train_nll_delta": final_train["loss"] - train_original["loss"],
            "raw_train_logit_rel_error": diag["chart_train_logit_rel_error"],
            "raw_guard_logit_rel_error": diag["chart_guard_logit_rel_error"],
            "F5_no_debt_pass": no_debt,
            "raw_chart_total_audit_pass": raw_pass,
            "raw_chart_total_audit_reject": int(not raw_pass),
            "Brier_delta": debt["Brier_delta"],
            "ECE_delta": debt["ECE_delta"],
            "tail95_delta": debt["tail95_delta"],
            "tail99_delta": debt["tail99_delta"],
            "margin10_delta": debt["margin10_delta"],
            "debt_delta": debt["debt_delta"],
            "raw_Brier_delta": raw_debt["Brier_delta"],
            "raw_ECE_delta": raw_debt["ECE_delta"],
            "raw_tail95_delta": raw_debt["tail95_delta"],
            "raw_tail99_delta": raw_debt["tail99_delta"],
            "raw_margin10_delta": raw_debt["margin10_delta"],
            "margin10_delta_correct_sign": int(debt["margin10_delta"] >= -float(args.no_debt_budget)),
            "component_non_positive": int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0),
            "original_to_final_total_audit": 1,
            "post_chart_incremental_only": 0,
            "wall_time_s": time.time() - start,
        }
    except Exception as exc:
        return {**AUDIT_DEFAULTS, "part": "1G", "status": "error", "scheme": scheme, "task": task, "seed": int(seed), "error_message": repr(exc), "wall_time_s": time.time() - start}


def run_frame_row(scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    start = time.time()
    try:
        x, y, xg, yg = v2308.v2307.visual_data(task, int(seed), args, device)
        classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = model_seed_for(task, int(seed), args)
        model = v2308.v2307.make_model(str(args.basis_key), int(args.depth), int(x.shape[1]), classes, model_seed, args, device, basis_input_gain=float(args.part_d_basis_input_gain))
        install_chart_methods(model)
        ckpt = v2308.v2307.train_checkpoint(model, x, y, str(args.checkpoint), int(seed), args)
        original_state = chart_state(model)
        train_original = v2308.actual_metrics(model, x, y)
        guard_original = v2308.actual_metrics(model, xg, yg)
        if scheme.startswith("C43"):
            raw_state = original_state
            raw_train = train_original
            raw_guard = guard_original
            diag = {
                "frame_source_gram_offdiag_before_median": 0.0,
                "frame_source_gram_offdiag_after_median": 0.0,
                "chart_center_abs_median": 0.0,
                "chart_center_abs_max": 0.0,
                "chart_inv_scale_median": 1.0,
                "chart_inv_scale_max": 1.0,
                "chart_layer_fit_rel_median": 0.0,
                "chart_layer_fit_rel_max": 0.0,
                "chart_layer_fit_rel_curve": "[]",
                "chart_phi_condition_median": 0.0,
                "chart_phi_condition_max": 0.0,
                "chart_train_logit_rel_error": 0.0,
                "chart_guard_logit_rel_error": 0.0,
                "chart_raw_design_drift_median": 0.0,
                "chart_raw_design_drift_max": 0.0,
                "activation_drift_median": 0.0,
                "activation_drift_max": 0.0,
                "activation_drift_layer_count": len(model.coeffs),
            }
        else:
            diag = apply_orthogonal_frame_recode(model, x, xg, scheme, int(seed), task, args)
            raw_state = chart_state(model)
            raw_train = v2308.actual_metrics(model, x, y)
            raw_guard = v2308.actual_metrics(model, xg, yg)
        raw_pass, raw_debt = strict_total_audit_pass(guard_original, raw_guard, float(args.no_debt_budget))
        alpha, reason, selected_state, final_train, final_guard = project_chart_state(model, original_state, raw_state, x, y, xg, yg, guard_original, args)
        selected_drift = chart_drift_stats(model, xg, original_state, selected_state)
        load_chart_state(model, selected_state)
        debt = v2308.debt_deltas(guard_original, final_guard)
        no_debt = v2308.no_debt_ok(debt, float(args.no_debt_budget))
        return {
            **AUDIT_DEFAULTS,
            "part": "1H",
            "status": "ok",
            "scheme": scheme,
            "task": task,
            "seed": int(seed),
            "basis_key": str(args.basis_key),
            "basis_family": str(model.basis_name),
            "depth": int(args.depth),
            "train_size": int(args.train_size),
            "guard_size": int(args.guard_size),
            "checkpoint_name": str(args.checkpoint),
            "checkpoint_optimizer": ckpt.get("checkpoint_optimizer", ""),
            "model_seed": model_seed,
            "basis_frame_transform_used": int(not scheme.startswith("C43")),
            "frame_eval_float64": int(args.frame_eval_float64),
            "chart_alpha_candidates": str(args.chart_alphas),
            "chart_alpha_selected": alpha,
            "chart_alpha_selection_reason": reason,
            **diag,
            "selected_activation_drift_median": selected_drift["activation_drift_median"],
            "selected_activation_drift_max": selected_drift["activation_drift_max"],
            "selected_chart_design_drift_median": selected_drift["chart_design_drift_median"],
            "selected_chart_design_drift_max": selected_drift["chart_design_drift_max"],
            "original_guard_coverage": guard_original["coverage"],
            "raw_chart_guard_coverage": raw_guard["coverage"],
            "final_guard_coverage": final_guard["coverage"],
            "total_C2_coverage_improvement": final_guard["coverage"] - guard_original["coverage"],
            "raw_chart_total_C2_coverage_improvement": raw_guard["coverage"] - guard_original["coverage"],
            "total_C2_accuracy_improvement": final_guard["accuracy"] - guard_original["accuracy"],
            "total_guard_nll_delta": final_guard["loss"] - guard_original["loss"],
            "total_train_nll_delta": final_train["loss"] - train_original["loss"],
            "raw_train_logit_rel_error": diag["chart_train_logit_rel_error"],
            "raw_guard_logit_rel_error": diag["chart_guard_logit_rel_error"],
            "F5_no_debt_pass": no_debt,
            "raw_chart_total_audit_pass": raw_pass,
            "raw_chart_total_audit_reject": int(not raw_pass),
            "Brier_delta": debt["Brier_delta"],
            "ECE_delta": debt["ECE_delta"],
            "tail95_delta": debt["tail95_delta"],
            "tail99_delta": debt["tail99_delta"],
            "margin10_delta": debt["margin10_delta"],
            "debt_delta": debt["debt_delta"],
            "raw_Brier_delta": raw_debt["Brier_delta"],
            "raw_ECE_delta": raw_debt["ECE_delta"],
            "raw_tail95_delta": raw_debt["tail95_delta"],
            "raw_tail99_delta": raw_debt["tail99_delta"],
            "raw_margin10_delta": raw_debt["margin10_delta"],
            "margin10_delta_correct_sign": int(debt["margin10_delta"] >= -float(args.no_debt_budget)),
            "component_non_positive": int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0),
            "original_to_final_total_audit": 1,
            "post_chart_incremental_only": 0,
            "wall_time_s": time.time() - start,
        }
    except Exception as exc:
        return {**AUDIT_DEFAULTS, "part": "1H", "status": "error", "scheme": scheme, "task": task, "seed": int(seed), "error_message": repr(exc), "wall_time_s": time.time() - start}


def chart_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    schemes = [item.strip() for item in str(args.chart_schemes).split(",") if item.strip()]
    tasks = [item.strip() for item in str(args.chart_tasks).split(",") if item.strip()]
    return [(scheme, task, seed) for scheme in schemes for task in tasks for seed in range(int(args.chart_seed_count))]


def frame_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    schemes = [item.strip() for item in str(args.frame_schemes).split(",") if item.strip()]
    tasks = [item.strip() for item in str(args.frame_tasks).split(",") if item.strip()]
    return [(scheme, task, seed) for scheme in schemes for task in tasks for seed in range(int(args.frame_seed_count))]


def dyadic_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    schemes = [item.strip() for item in str(args.dyadic_schemes).split(",") if item.strip()]
    tasks = [item.strip() for item in str(args.dyadic_tasks).split(",") if item.strip()]
    return [(scheme, task, seed) for scheme in schemes for task in tasks for seed in range(int(args.dyadic_seed_count))]


def chart_run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    lock_path = OUT_ROOT / "stage0_evidence_lock_summary.json"
    if not lock_path.exists() or ival(read_json(lock_path).get("gate_pass")) != 1:
        summary = {"part": "1G", "route": "Stage1GBlockedByStage0", "gate_pass": 0, "dominant_blocker": "stage0_missing_or_failed"}
        out = write_json(OUT_ROOT / "stage1g_chart_recode_summary.json", summary)
        append_exec("Stage 1G chart recode blocked", command_text(), files=rel(out), gpu=str(args.device), note=summary["dominant_blocker"])
        return summary
    device = torch.device(str(args.device))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"stage1g_chart_recode_matrix{suffix}.csv"
    existing = read_rows(matrix) if int(args.chart_resume) else []
    done = {(r.get("scheme"), r.get("task"), str(r.get("seed"))) for r in existing if r.get("status") == "ok"}
    jobs = [job for idx, job in enumerate(chart_jobs(args)) if idx % int(args.shard_count) == int(args.shard_index)]
    rows = list(existing)
    for scheme, task, seed in jobs:
        if (scheme, task, str(seed)) in done:
            continue
        rows.append(run_chart_row(scheme, task, int(seed), args, device))
        if int(args.chart_flush_every) > 0:
            write_rows(matrix, rows)
            print(json.dumps({"part": "1G", "rows_written": len(rows), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
    write_rows(matrix, rows)
    summary = {"part": "1G", "route": "Stage1GShardDone", "gate_pass": 0, "matrix": rel(matrix), "rows_written": len(rows), "ok_rows": sum(1 for row in rows if row.get("status") == "ok")}
    out = write_json(OUT_ROOT / f"stage1g_chart_recode_summary{suffix}.json", summary)
    append_exec("Stage 1G chart recode shard", command_text(), files=f"{rel(matrix)}; {rel(out)}", gpu=str(args.device), note=f"rows={len(rows)}; ok={summary['ok_rows']}")
    return summary


def frame_run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    lock_path = OUT_ROOT / "stage0_evidence_lock_summary.json"
    if not lock_path.exists() or ival(read_json(lock_path).get("gate_pass")) != 1:
        summary = {"part": "1H", "route": "Stage1HBlockedByStage0", "gate_pass": 0, "dominant_blocker": "stage0_missing_or_failed"}
        out = write_json(OUT_ROOT / "stage1h_frame_recode_summary.json", summary)
        append_exec("Stage 1H frame recode blocked", command_text(), files=rel(out), gpu=str(args.device), note=summary["dominant_blocker"])
        return summary
    device = torch.device(str(args.device))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"stage1h_frame_recode_matrix{suffix}.csv"
    existing = read_rows(matrix) if int(args.frame_resume) else []
    done = {(r.get("scheme"), r.get("task"), str(r.get("seed"))) for r in existing if r.get("status") == "ok"}
    jobs = [job for idx, job in enumerate(frame_jobs(args)) if idx % int(args.shard_count) == int(args.shard_index)]
    rows = list(existing)
    for scheme, task, seed in jobs:
        if (scheme, task, str(seed)) in done:
            continue
        rows.append(run_frame_row(scheme, task, int(seed), args, device))
        if int(args.frame_flush_every) > 0:
            write_rows(matrix, rows)
            print(json.dumps({"part": "1H", "rows_written": len(rows), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
    write_rows(matrix, rows)
    summary = {"part": "1H", "route": "Stage1HShardDone", "gate_pass": 0, "matrix": rel(matrix), "rows_written": len(rows), "ok_rows": sum(1 for row in rows if row.get("status") == "ok")}
    out = write_json(OUT_ROOT / f"stage1h_frame_recode_summary{suffix}.json", summary)
    append_exec("Stage 1H frame recode shard", command_text(), files=f"{rel(matrix)}; {rel(out)}", gpu=str(args.device), note=f"rows={len(rows)}; ok={summary['ok_rows']}")
    return summary


def dyadic_run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    lock_path = OUT_ROOT / "stage0_evidence_lock_summary.json"
    if not lock_path.exists() or ival(read_json(lock_path).get("gate_pass")) != 1:
        summary = {"part": "1I", "route": "Stage1IBlockedByStage0", "gate_pass": 0, "dominant_blocker": "stage0_missing_or_failed"}
        out = write_json(OUT_ROOT / "stage1i_dyadic_frame_summary.json", summary)
        append_exec("Stage 1I dyadic frame blocked", command_text(), files=rel(out), gpu=str(args.device), note=summary["dominant_blocker"])
        return summary
    device = torch.device(str(args.device))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"stage1i_dyadic_frame_matrix{suffix}.csv"
    existing = read_rows(matrix) if int(args.dyadic_resume) else []
    done = {(r.get("scheme"), r.get("task"), str(r.get("seed"))) for r in existing if r.get("status") == "ok"}
    jobs = [job for idx, job in enumerate(dyadic_jobs(args)) if idx % int(args.shard_count) == int(args.shard_index)]
    rows = list(existing)
    for scheme, task, seed in jobs:
        if (scheme, task, str(seed)) in done:
            continue
        row = run_frame_row(scheme, task, int(seed), args, device)
        row["part"] = "1I"
        rows.append(row)
        if int(args.dyadic_flush_every) > 0:
            write_rows(matrix, rows)
            print(json.dumps({"part": "1I", "rows_written": len(rows), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
    write_rows(matrix, rows)
    summary = {"part": "1I", "route": "Stage1IShardDone", "gate_pass": 0, "matrix": rel(matrix), "rows_written": len(rows), "ok_rows": sum(1 for row in rows if row.get("status") == "ok")}
    out = write_json(OUT_ROOT / f"stage1i_dyadic_frame_summary{suffix}.json", summary)
    append_exec("Stage 1I dyadic frame shard", command_text(), files=f"{rel(matrix)}; {rel(out)}", gpu=str(args.device), note=f"rows={len(rows)}; ok={summary['ok_rows']}")
    return summary


def collect_chart_rows() -> list[dict[str, str]]:
    shard_files = sorted(OUT_ROOT.glob("stage1g_chart_recode_matrix_shard*_of_*.csv"))
    files = shard_files if shard_files else [OUT_ROOT / "stage1g_chart_recode_matrix.csv"]
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in files:
        for row in read_rows(path):
            key = (row.get("scheme", ""), row.get("task", ""), str(row.get("seed", "")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def collect_frame_rows() -> list[dict[str, str]]:
    shard_files = sorted(OUT_ROOT.glob("stage1h_frame_recode_matrix_shard*_of_*.csv"))
    files = shard_files if shard_files else [OUT_ROOT / "stage1h_frame_recode_matrix.csv"]
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in files:
        for row in read_rows(path):
            key = (row.get("scheme", ""), row.get("task", ""), str(row.get("seed", "")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def collect_dyadic_rows() -> list[dict[str, str]]:
    shard_files = sorted(OUT_ROOT.glob("stage1i_dyadic_frame_matrix_shard*_of_*.csv"))
    files = shard_files if shard_files else [OUT_ROOT / "stage1i_dyadic_frame_matrix.csv"]
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in files:
        for row in read_rows(path):
            key = (row.get("scheme", ""), row.get("task", ""), str(row.get("seed", "")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def chart_merge(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    rows = collect_chart_rows()
    matrix = write_rows(OUT_ROOT / "stage1g_chart_recode_matrix.csv", rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for scheme in [item.strip() for item in str(args.chart_schemes).split(",") if item.strip()]:
        group_rows = [row for row in ok if row.get("scheme") == scheme]
        groups.append({
            "scheme": scheme,
            "rows": len(group_rows),
            "total_C2_coverage_improvement_median": median([r.get("total_C2_coverage_improvement") for r in group_rows]),
            "total_C2_accuracy_improvement_median": median([r.get("total_C2_accuracy_improvement") for r in group_rows]),
            "total_guard_nll_delta_median": median([r.get("total_guard_nll_delta") for r in group_rows]),
            "F5_no_debt_count": sum(ival(r.get("F5_no_debt_pass")) for r in group_rows),
            "component_non_positive_rows": sum(ival(r.get("component_non_positive")) for r in group_rows),
            "raw_chart_reject_count": sum(ival(r.get("raw_chart_total_audit_reject")) for r in group_rows),
            "chart_alpha_selected_median": median([r.get("chart_alpha_selected") for r in group_rows]),
            "chart_layer_fit_rel_median": median([r.get("chart_layer_fit_rel_median") for r in group_rows]),
            "chart_layer_fit_rel_max_median": median([r.get("chart_layer_fit_rel_max") for r in group_rows]),
            "raw_train_logit_rel_error_median": median([r.get("raw_train_logit_rel_error") for r in group_rows]),
            "raw_guard_logit_rel_error_median": median([r.get("raw_guard_logit_rel_error") for r in group_rows]),
            "chart_center_abs_median": median([r.get("chart_center_abs_median") for r in group_rows]),
            "chart_inv_scale_median": median([r.get("chart_inv_scale_median") for r in group_rows]),
            "selected_chart_design_drift_median": median([r.get("selected_chart_design_drift_median") for r in group_rows]),
            "selected_activation_drift_median": median([r.get("selected_activation_drift_median") for r in group_rows]),
            "wall_time_s_median": median([r.get("wall_time_s") for r in group_rows]),
        })
    group_csv = write_rows(OUT_ROOT / "stage1g_chart_recode_group_summary.csv", groups)
    by_scheme = {g["scheme"]: g for g in groups}
    c39 = by_scheme.get("C39_Candidate_source_whitened_cheb_chart_recode_total_audit", {})
    c40 = by_scheme.get("C40_Control_identity_chart_total_audit", {})
    expected_rows = len(chart_jobs(args))
    c39_expected = sum(1 for scheme, _task, _seed in chart_jobs(args) if scheme == "C39_Candidate_source_whitened_cheb_chart_recode_total_audit")
    c39_cov = fval(c39.get("total_C2_coverage_improvement_median"))
    c39_design = fval(c39.get("selected_chart_design_drift_median"))
    c40_design = fval(c40.get("selected_chart_design_drift_median"))
    c39_fit = fval(c39.get("chart_layer_fit_rel_median"))
    checks = {
        "row_count_expected_ok": len(rows) == expected_rows and len(ok) == expected_rows,
        "no_fake_data": sum(ival(r.get("used_fake_data_rows")) for r in ok) == 0,
        "no_held_test_usage": sum(ival(r.get("held_test_usage")) for r in ok) == 0,
        "original_to_final_total_audit_all_rows": sum(ival(r.get("original_to_final_total_audit")) for r in ok) == len(ok),
        "post_chart_incremental_only_no_rows": sum(ival(r.get("post_chart_incremental_only")) for r in ok) == 0,
        "c39_total_no_debt_clean": c39_expected > 0 and fval(c39.get("F5_no_debt_count")) == c39_expected and fval(c39.get("component_non_positive_rows")) == c39_expected,
        "c39_abs_coverage_le_0p01": abs(c39_cov) <= 0.01,
        "c39_design_drift_ge_0p001": c39_design >= 0.001,
        "c39_fit_rel_le_0p05": c39_fit <= 0.05,
        "c39_design_minus_identity_ge_0p001": c39_design - c40_design >= 0.001,
    }
    gate = int(all(checks.values()))
    if not checks["row_count_expected_ok"]:
        blocker = "unexpected_stage1g_row_count_or_errors"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif not checks["c39_total_no_debt_clean"]:
        blocker = "c39_total_audit_debt_or_component_positive"
    elif not checks["c39_abs_coverage_le_0p01"]:
        blocker = "c39_not_function_preserving_on_total_coverage"
    elif not checks["c39_design_drift_ge_0p001"]:
        blocker = "c39_no_selected_chart_design_drift"
    elif not checks["c39_fit_rel_le_0p05"]:
        blocker = "c39_source_reconstruction_error_too_high"
    elif not checks["c39_design_minus_identity_ge_0p001"]:
        blocker = "c39_design_drift_not_above_identity"
    else:
        blocker = "none"
    summary = {
        "part": "1G",
        "route": "Stage1GSourceWhitenedChartRecodePass" if gate else "Stage1GSourceWhitenedChartRecodeFailed",
        "gate_pass": gate,
        "dominant_blocker": blocker,
        "checks": checks,
        "metrics": {
            "c39_coverage": c39_cov,
            "c39_abs_coverage": abs(c39_cov),
            "c39_design_drift": c39_design,
            "c40_design_drift": c40_design,
            "c39_design_minus_identity": c39_design - c40_design,
            "c39_fit_rel": c39_fit,
            "c39_fit_rel_max": fval(c39.get("chart_layer_fit_rel_max_median")),
            "c39_raw_train_logit_rel_error": fval(c39.get("raw_train_logit_rel_error_median")),
            "c39_raw_guard_logit_rel_error": fval(c39.get("raw_guard_logit_rel_error_median")),
            "c39_alpha_selected_median": fval(c39.get("chart_alpha_selected_median")),
            "c39_chart_center_abs_median": fval(c39.get("chart_center_abs_median")),
            "c39_chart_inv_scale_median": fval(c39.get("chart_inv_scale_median")),
            "c39_wall_time_s_median": fval(c39.get("wall_time_s_median")),
        },
        "artifacts": {"matrix": rel(matrix), "group_summary": rel(group_csv)},
        "hashes": {"matrix": sha256_file(matrix), "group_summary": sha256_file(group_csv)},
        "next_action": "run_stage2_inverse_from_C39_chart" if gate else "stop_or_repair_source_whitened_chart_before_inverse",
    }
    out = write_json(OUT_ROOT / "stage1g_chart_recode_summary.json", summary)
    nxt = write_json(
        FORMAL_ROOT / "next_actions_for_v23_14.json",
        {
            "version": "v23.14",
            "last_completed_stage": "stage1g_source_whitened_chart_recode",
            "stage1g_gate_pass": gate,
            "stage1g_route": summary["route"],
            "stage1g_blocker": blocker,
            "promotion_allowed": 0,
            "completed_artifacts": {"stage1g_summary": rel(out), "matrix": rel(matrix), "group_summary": rel(group_csv)},
            "allowed_next_actions": ["run_stage2_inverse_from_C39_chart"] if gate else ["stop_or_repair_source_whitened_chart_before_inverse"],
            "forbidden_next_actions": [
                "promote_C39_without_inverse_and_full_controls",
                "fabricate_data",
                "held_test_induction",
                "runtime_winner_selection",
                "add_mlp_stem_or_readout",
                "add_new_edge_function_family",
            ],
        },
    )
    append_exec("Stage 1G source-whitened chart recode merge", command_text(), files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Stage 1G source-whitened chart recode", summary)
    return summary


def frame_merge(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    rows = collect_frame_rows()
    matrix = write_rows(OUT_ROOT / "stage1h_frame_recode_matrix.csv", rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for scheme in [item.strip() for item in str(args.frame_schemes).split(",") if item.strip()]:
        group_rows = [row for row in ok if row.get("scheme") == scheme]
        groups.append({
            "scheme": scheme,
            "rows": len(group_rows),
            "total_C2_coverage_improvement_median": median([r.get("total_C2_coverage_improvement") for r in group_rows]),
            "total_C2_accuracy_improvement_median": median([r.get("total_C2_accuracy_improvement") for r in group_rows]),
            "total_guard_nll_delta_median": median([r.get("total_guard_nll_delta") for r in group_rows]),
            "F5_no_debt_count": sum(ival(r.get("F5_no_debt_pass")) for r in group_rows),
            "component_non_positive_rows": sum(ival(r.get("component_non_positive")) for r in group_rows),
            "raw_chart_reject_count": sum(ival(r.get("raw_chart_total_audit_reject")) for r in group_rows),
            "chart_alpha_selected_median": median([r.get("chart_alpha_selected") for r in group_rows]),
            "chart_layer_fit_rel_median": median([r.get("chart_layer_fit_rel_median") for r in group_rows]),
            "chart_layer_fit_rel_max_median": median([r.get("chart_layer_fit_rel_max") for r in group_rows]),
            "raw_train_logit_rel_error_median": median([r.get("raw_train_logit_rel_error") for r in group_rows]),
            "raw_guard_logit_rel_error_median": median([r.get("raw_guard_logit_rel_error") for r in group_rows]),
            "frame_source_gram_offdiag_before_median": median([r.get("frame_source_gram_offdiag_before_median") for r in group_rows]),
            "frame_source_gram_offdiag_after_median": median([r.get("frame_source_gram_offdiag_after_median") for r in group_rows]),
            "selected_chart_design_drift_median": median([r.get("selected_chart_design_drift_median") for r in group_rows]),
            "selected_activation_drift_median": median([r.get("selected_activation_drift_median") for r in group_rows]),
            "wall_time_s_median": median([r.get("wall_time_s") for r in group_rows]),
        })
    group_csv = write_rows(OUT_ROOT / "stage1h_frame_recode_group_summary.csv", groups)
    by_scheme = {g["scheme"]: g for g in groups}
    c42 = by_scheme.get("C42_Candidate_source_gram_pca_cheb_frame_recode_total_audit", {})
    c43 = by_scheme.get("C43_Control_identity_frame_total_audit", {})
    expected_rows = len(frame_jobs(args))
    c42_expected = sum(1 for scheme, _task, _seed in frame_jobs(args) if scheme == "C42_Candidate_source_gram_pca_cheb_frame_recode_total_audit")
    c42_cov = fval(c42.get("total_C2_coverage_improvement_median"))
    c42_design = fval(c42.get("selected_chart_design_drift_median"))
    c43_design = fval(c43.get("selected_chart_design_drift_median"))
    c42_fit = fval(c42.get("chart_layer_fit_rel_median"))
    checks = {
        "row_count_expected_ok": len(rows) == expected_rows and len(ok) == expected_rows,
        "no_fake_data": sum(ival(r.get("used_fake_data_rows")) for r in ok) == 0,
        "no_held_test_usage": sum(ival(r.get("held_test_usage")) for r in ok) == 0,
        "original_to_final_total_audit_all_rows": sum(ival(r.get("original_to_final_total_audit")) for r in ok) == len(ok),
        "post_chart_incremental_only_no_rows": sum(ival(r.get("post_chart_incremental_only")) for r in ok) == 0,
        "c42_total_no_debt_clean": c42_expected > 0 and fval(c42.get("F5_no_debt_count")) == c42_expected and fval(c42.get("component_non_positive_rows")) == c42_expected,
        "c42_abs_coverage_le_0p01": abs(c42_cov) <= 0.01,
        "c42_design_drift_ge_0p001": c42_design >= 0.001,
        "c42_fit_rel_le_0p001": c42_fit <= 0.001,
        "c42_design_minus_identity_ge_0p001": c42_design - c43_design >= 0.001,
    }
    gate = int(all(checks.values()))
    if not checks["row_count_expected_ok"]:
        blocker = "unexpected_stage1h_row_count_or_errors"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif not checks["c42_total_no_debt_clean"]:
        blocker = "c42_total_audit_debt_or_component_positive"
    elif not checks["c42_abs_coverage_le_0p01"]:
        blocker = "c42_not_function_preserving_on_total_coverage"
    elif not checks["c42_design_drift_ge_0p001"]:
        blocker = "c42_no_selected_frame_design_drift"
    elif not checks["c42_fit_rel_le_0p001"]:
        blocker = "c42_source_reconstruction_error_too_high"
    elif not checks["c42_design_minus_identity_ge_0p001"]:
        blocker = "c42_design_drift_not_above_identity"
    else:
        blocker = "none"
    summary = {
        "part": "1H",
        "route": "Stage1HSourceGramFrameRecodePass" if gate else "Stage1HSourceGramFrameRecodeFailed",
        "gate_pass": gate,
        "dominant_blocker": blocker,
        "checks": checks,
        "metrics": {
            "c42_coverage": c42_cov,
            "c42_abs_coverage": abs(c42_cov),
            "c42_design_drift": c42_design,
            "c43_design_drift": c43_design,
            "c42_design_minus_identity": c42_design - c43_design,
            "c42_fit_rel": c42_fit,
            "c42_fit_rel_max": fval(c42.get("chart_layer_fit_rel_max_median")),
            "c42_raw_train_logit_rel_error": fval(c42.get("raw_train_logit_rel_error_median")),
            "c42_raw_guard_logit_rel_error": fval(c42.get("raw_guard_logit_rel_error_median")),
            "c42_alpha_selected_median": fval(c42.get("chart_alpha_selected_median")),
            "c42_source_gram_offdiag_before": fval(c42.get("frame_source_gram_offdiag_before_median")),
            "c42_source_gram_offdiag_after": fval(c42.get("frame_source_gram_offdiag_after_median")),
            "c42_wall_time_s_median": fval(c42.get("wall_time_s_median")),
        },
        "artifacts": {"matrix": rel(matrix), "group_summary": rel(group_csv)},
        "hashes": {"matrix": sha256_file(matrix), "group_summary": sha256_file(group_csv)},
        "next_action": "run_stage2_inverse_from_C42_frame" if gate else "stop_or_repair_source_gram_frame_before_inverse",
    }
    out = write_json(OUT_ROOT / "stage1h_frame_recode_summary.json", summary)
    nxt = write_json(
        FORMAL_ROOT / "next_actions_for_v23_14.json",
        {
            "version": "v23.14",
            "last_completed_stage": "stage1h_source_gram_frame_recode",
            "stage1h_gate_pass": gate,
            "stage1h_route": summary["route"],
            "stage1h_blocker": blocker,
            "promotion_allowed": 0,
            "completed_artifacts": {"stage1h_summary": rel(out), "matrix": rel(matrix), "group_summary": rel(group_csv)},
            "allowed_next_actions": ["run_stage2_inverse_from_C42_frame"] if gate else ["stop_or_repair_source_gram_frame_before_inverse"],
            "forbidden_next_actions": [
                "promote_C42_without_inverse_and_full_controls",
                "fabricate_data",
                "held_test_induction",
                "runtime_winner_selection",
                "add_mlp_stem_or_readout",
                "add_new_edge_function_family",
            ],
        },
    )
    append_exec("Stage 1H source-gram frame recode merge", command_text(), files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Stage 1H source-gram frame recode", summary)
    return summary


def dyadic_merge(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    rows = collect_dyadic_rows()
    matrix = write_rows(OUT_ROOT / "stage1i_dyadic_frame_matrix.csv", rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for scheme in [item.strip() for item in str(args.dyadic_schemes).split(",") if item.strip()]:
        group_rows = [row for row in ok if row.get("scheme") == scheme]
        groups.append({
            "scheme": scheme,
            "rows": len(group_rows),
            "total_C2_coverage_improvement_median": median([r.get("total_C2_coverage_improvement") for r in group_rows]),
            "total_guard_nll_delta_median": median([r.get("total_guard_nll_delta") for r in group_rows]),
            "F5_no_debt_count": sum(ival(r.get("F5_no_debt_pass")) for r in group_rows),
            "component_non_positive_rows": sum(ival(r.get("component_non_positive")) for r in group_rows),
            "raw_chart_reject_count": sum(ival(r.get("raw_chart_total_audit_reject")) for r in group_rows),
            "chart_alpha_selected_median": median([r.get("chart_alpha_selected") for r in group_rows]),
            "chart_layer_fit_rel_median": median([r.get("chart_layer_fit_rel_median") for r in group_rows]),
            "chart_layer_fit_rel_max_median": median([r.get("chart_layer_fit_rel_max") for r in group_rows]),
            "raw_guard_logit_rel_error_median": median([r.get("raw_guard_logit_rel_error") for r in group_rows]),
            "frame_dyadic_exp_abs_median": median([r.get("frame_dyadic_exp_abs_median") for r in group_rows]),
            "frame_rms_spread_before_median": median([r.get("frame_rms_spread_before_median") for r in group_rows]),
            "frame_rms_spread_after_median": median([r.get("frame_rms_spread_after_median") for r in group_rows]),
            "selected_chart_design_drift_median": median([r.get("selected_chart_design_drift_median") for r in group_rows]),
            "selected_activation_drift_median": median([r.get("selected_activation_drift_median") for r in group_rows]),
            "wall_time_s_median": median([r.get("wall_time_s") for r in group_rows]),
        })
    group_csv = write_rows(OUT_ROOT / "stage1i_dyadic_frame_group_summary.csv", groups)
    by_scheme = {g["scheme"]: g for g in groups}
    c45 = by_scheme.get("C45_Candidate_source_rms_dyadic_cheb_frame_recode_total_audit", {})
    c43 = by_scheme.get("C43_Control_identity_frame_total_audit", {})
    expected_rows = len(dyadic_jobs(args))
    c45_expected = sum(1 for scheme, _task, _seed in dyadic_jobs(args) if scheme == "C45_Candidate_source_rms_dyadic_cheb_frame_recode_total_audit")
    c45_cov = fval(c45.get("total_C2_coverage_improvement_median"))
    c45_design = fval(c45.get("selected_chart_design_drift_median"))
    c43_design = fval(c43.get("selected_chart_design_drift_median"))
    c45_fit = fval(c45.get("chart_layer_fit_rel_median"))
    checks = {
        "row_count_expected_ok": len(rows) == expected_rows and len(ok) == expected_rows,
        "no_fake_data": sum(ival(r.get("used_fake_data_rows")) for r in ok) == 0,
        "no_held_test_usage": sum(ival(r.get("held_test_usage")) for r in ok) == 0,
        "original_to_final_total_audit_all_rows": sum(ival(r.get("original_to_final_total_audit")) for r in ok) == len(ok),
        "post_chart_incremental_only_no_rows": sum(ival(r.get("post_chart_incremental_only")) for r in ok) == 0,
        "c45_total_no_debt_clean": c45_expected > 0 and fval(c45.get("F5_no_debt_count")) == c45_expected and fval(c45.get("component_non_positive_rows")) == c45_expected,
        "c45_abs_coverage_le_0p01": abs(c45_cov) <= 0.01,
        "c45_design_drift_ge_0p001": c45_design >= 0.001,
        "c45_fit_rel_le_0p001": c45_fit <= 0.001,
        "c45_design_minus_identity_ge_0p001": c45_design - c43_design >= 0.001,
    }
    gate = int(all(checks.values()))
    if not checks["row_count_expected_ok"]:
        blocker = "unexpected_stage1i_row_count_or_errors"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif not checks["c45_total_no_debt_clean"]:
        blocker = "c45_total_audit_debt_or_component_positive"
    elif not checks["c45_abs_coverage_le_0p01"]:
        blocker = "c45_not_function_preserving_on_total_coverage"
    elif not checks["c45_design_drift_ge_0p001"]:
        blocker = "c45_no_selected_dyadic_design_drift"
    elif not checks["c45_fit_rel_le_0p001"]:
        blocker = "c45_source_reconstruction_error_too_high"
    elif not checks["c45_design_minus_identity_ge_0p001"]:
        blocker = "c45_design_drift_not_above_identity"
    else:
        blocker = "none"
    summary = {
        "part": "1I",
        "route": "Stage1ISourceRmsDyadicFramePass" if gate else "Stage1ISourceRmsDyadicFrameFailed",
        "gate_pass": gate,
        "dominant_blocker": blocker,
        "checks": checks,
        "metrics": {
            "c45_coverage": c45_cov,
            "c45_abs_coverage": abs(c45_cov),
            "c45_design_drift": c45_design,
            "c43_design_drift": c43_design,
            "c45_design_minus_identity": c45_design - c43_design,
            "c45_fit_rel": c45_fit,
            "c45_fit_rel_max": fval(c45.get("chart_layer_fit_rel_max_median")),
            "c45_raw_guard_logit_rel_error": fval(c45.get("raw_guard_logit_rel_error_median")),
            "c45_alpha_selected_median": fval(c45.get("chart_alpha_selected_median")),
            "c45_dyadic_exp_abs_median": fval(c45.get("frame_dyadic_exp_abs_median")),
            "c45_rms_spread_before": fval(c45.get("frame_rms_spread_before_median")),
            "c45_rms_spread_after": fval(c45.get("frame_rms_spread_after_median")),
            "c45_wall_time_s_median": fval(c45.get("wall_time_s_median")),
        },
        "artifacts": {"matrix": rel(matrix), "group_summary": rel(group_csv)},
        "hashes": {"matrix": sha256_file(matrix), "group_summary": sha256_file(group_csv)},
        "next_action": "run_stage2_inverse_from_C45_dyadic_frame" if gate else "stop_or_repair_source_rms_dyadic_frame_before_inverse",
    }
    out = write_json(OUT_ROOT / "stage1i_dyadic_frame_summary.json", summary)
    nxt = write_json(
        FORMAL_ROOT / "next_actions_for_v23_14.json",
        {
            "version": "v23.14",
            "last_completed_stage": "stage1i_source_rms_dyadic_frame",
            "stage1i_gate_pass": gate,
            "stage1i_route": summary["route"],
            "stage1i_blocker": blocker,
            "promotion_allowed": 0,
            "completed_artifacts": {"stage1i_summary": rel(out), "matrix": rel(matrix), "group_summary": rel(group_csv)},
            "allowed_next_actions": ["run_stage2_inverse_from_C45_dyadic_frame"] if gate else ["stop_or_repair_source_rms_dyadic_frame_before_inverse"],
            "forbidden_next_actions": ["promote_C45_without_inverse_and_full_controls", "fabricate_data", "held_test_induction", "runtime_winner_selection", "add_mlp_stem_or_readout", "add_new_edge_function_family"],
        },
    )
    append_exec("Stage 1I source-rms dyadic frame merge", command_text(), files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Stage 1I source-rms dyadic frame", summary)
    return summary


def stage2_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    schemes = [item.strip() for item in str(args.stage2_schemes).split(",") if item.strip()]
    tasks = [item.strip() for item in str(args.stage2_tasks).split(",") if item.strip()]
    return [(scheme, task, seed) for scheme in schemes for task in tasks for seed in range(int(args.stage2_seed_count))]


def run_stage2_row(scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    start = time.time()
    try:
        ctx = prepare_stage2_frame_context(scheme, task, int(seed), args, device)
        model = ctx["model"]
        x, y, xg, yg = ctx["x"], ctx["y"], ctx["xg"], ctx["yg"]
        source_scheme = "frame_only_no_inverse"
        raw_final_train = ctx["train_frame"]
        raw_final_guard = ctx["guard_frame"]
        raw_final_state = ctx["frame_state"]
        inverse_diag: dict[str, Any] = {
            "finite_step_accept_count": 0,
            "finite_step_accept_rate": 0.0,
            "finite_step_scale_mean": 0.0,
            "finite_step_skip_count": 0,
            "finite_step_reject_reasons": json.dumps({"frame_only_no_inverse": 1}, sort_keys=True),
            "solve_residual_max": 0.0,
            "condition_number_median": 0.0,
            "cg_iteration_median": 0.0,
            "activation_drift_max": 0.0,
            "activation_drift_median": 0.0,
        }
        if not scheme.startswith("C51"):
            source_scheme = str(args.stage2_c15_source_scheme)
            fargs = stage2_f_args(args, alphas=str(args.stage2_alphas))
            inverse_diag = v2312.run_multistep_from_current(model, x, y, xg, yg, source_scheme, fargs, device)
            raw_final_state = chart_state(model)
            raw_final_train = inverse_diag["final_train"]
            raw_final_guard = inverse_diag["final_guard"]
        raw_pass, raw_debt = strict_total_audit_pass(ctx["guard_original"], raw_final_guard, float(args.no_debt_budget))
        selected_alpha = 0.0
        selected_reason = "alpha_zero_fallback"
        selected_state = ctx["frame_state"]
        final_train = ctx["train_frame"]
        final_guard = ctx["guard_frame"]
        terminal_pass = 0
        terminal_reject = 1
        for alpha in float_items(str(args.stage2_terminal_alphas)):
            cand_state = interpolated_chart_state(ctx["frame_state"], raw_final_state, float(alpha))
            load_chart_state(model, cand_state)
            train_cand = v2308.actual_metrics(model, x, y)
            guard_cand = v2308.actual_metrics(model, xg, yg)
            cand_pass, _cand_debt = strict_total_audit_pass(ctx["guard_original"], guard_cand, float(args.no_debt_budget))
            if cand_pass:
                selected_alpha = float(alpha)
                selected_reason = "alpha_zero_fallback" if abs(float(alpha)) <= 1.0e-12 else "first_total_audit_alpha"
                selected_state = cand_state
                final_train = train_cand
                final_guard = guard_cand
                terminal_pass = 1
                terminal_reject = 0
                break
        selected_drift = chart_drift_stats(model, xg, ctx["original_state"], selected_state)
        debt = v2308.debt_deltas(ctx["guard_original"], final_guard)
        return {
            **AUDIT_DEFAULTS,
            "part": "2",
            "status": "ok",
            "scheme": scheme,
            "source_scheme": source_scheme,
            "frame_scheme": ctx["frame_scheme"],
            "task": task,
            "seed": int(seed),
            "basis_key": str(args.basis_key),
            "basis_family": str(model.basis_name),
            "depth": int(args.depth),
            "train_size": int(args.train_size),
            "guard_size": int(args.guard_size),
            "checkpoint_name": str(args.checkpoint),
            "checkpoint_optimizer": ctx["ckpt"].get("checkpoint_optimizer", ""),
            "model_seed": ctx["model_seed"],
            "stage2_steps": int(args.stage2_steps),
            "stage2_residual_target": str(args.stage2_residual_target),
            "stage2_lambda": float(args.stage2_lambda),
            "stage2_alphas": str(args.stage2_alphas),
            "stage2_terminal_alphas": str(args.stage2_terminal_alphas),
            "frame_total_audit_pass": ctx["frame_total_audit_pass"],
            "frame_design_drift_median": ctx["frame_drift"]["chart_design_drift_median"],
            "frame_activation_drift_median": ctx["frame_drift"]["activation_drift_median"],
            "frame_fit_rel": fval(ctx["frame_diag"].get("chart_layer_fit_rel_median")),
            "frame_raw_guard_logit_rel_error": fval(ctx["frame_diag"].get("chart_guard_logit_rel_error")),
            "frame_rms_spread_before": fval(ctx["frame_diag"].get("frame_rms_spread_before_median")),
            "frame_rms_spread_after": fval(ctx["frame_diag"].get("frame_rms_spread_after_median")),
            "original_guard_coverage": ctx["guard_original"]["coverage"],
            "frame_guard_coverage": ctx["guard_frame"]["coverage"],
            "raw_inverse_guard_coverage": raw_final_guard["coverage"],
            "final_guard_coverage": final_guard["coverage"],
            "total_C2_coverage_improvement": final_guard["coverage"] - ctx["guard_original"]["coverage"],
            "frame_to_final_C2_coverage_improvement": final_guard["coverage"] - ctx["guard_frame"]["coverage"],
            "raw_inverse_total_C2_coverage_improvement": raw_final_guard["coverage"] - ctx["guard_original"]["coverage"],
            "total_C2_accuracy_improvement": final_guard["accuracy"] - ctx["guard_original"]["accuracy"],
            "total_guard_nll_delta": final_guard["loss"] - ctx["guard_original"]["loss"],
            "total_train_nll_delta": final_train["loss"] - ctx["train_original"]["loss"],
            "source_guard_c2_gap": (final_train["coverage"] - ctx["train_original"]["coverage"]) - (final_guard["coverage"] - ctx["guard_original"]["coverage"]),
            "F5_no_debt_pass": v2308.no_debt_ok(debt, float(args.no_debt_budget)),
            "Brier_delta": debt["Brier_delta"],
            "ECE_delta": debt["ECE_delta"],
            "tail95_delta": debt["tail95_delta"],
            "tail99_delta": debt["tail99_delta"],
            "margin10_delta": debt["margin10_delta"],
            "debt_delta": debt["debt_delta"],
            "margin10_delta_correct_sign": int(debt["margin10_delta"] >= -float(args.no_debt_budget)),
            "component_non_positive": int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0),
            "terminal_total_audit_pass": terminal_pass,
            "terminal_total_audit_reject": terminal_reject,
            "inverse_terminal_alpha_selected": selected_alpha,
            "inverse_terminal_alpha_selection_reason": selected_reason,
            "raw_inverse_terminal_total_audit_pass": raw_pass,
            "raw_inverse_terminal_total_audit_reject": int(not raw_pass),
            "raw_inverse_Brier_delta": raw_debt["Brier_delta"],
            "raw_inverse_ECE_delta": raw_debt["ECE_delta"],
            "raw_inverse_tail95_delta": raw_debt["tail95_delta"],
            "raw_inverse_tail99_delta": raw_debt["tail99_delta"],
            "selected_chart_design_drift_median": selected_drift["chart_design_drift_median"],
            "selected_activation_drift_median": selected_drift["activation_drift_median"],
            "original_to_final_total_audit": 1,
            "post_frame_incremental_only": 0,
            "wall_time_s": time.time() - start,
            **{k: v for k, v in inverse_diag.items() if k not in {"start_train", "start_guard", "final_train", "final_guard"}},
        }
    except Exception as exc:
        return {**AUDIT_DEFAULTS, "part": "2", "status": "error", "scheme": scheme, "task": task, "seed": int(seed), "error_message": repr(exc), "wall_time_s": time.time() - start}


def stage2_run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    stage1i_path = Path(str(args.stage2_stage1i_summary))
    if not stage1i_path.is_absolute():
        stage1i_path = ROOT / stage1i_path
    if not stage1i_path.exists() or ival(read_json(stage1i_path).get("gate_pass")) != 1:
        summary = {"part": "2", "route": "Stage2BlockedByStage1I", "gate_pass": 0, "dominant_blocker": "stage1i_missing_or_failed"}
        out = write_json(OUT_ROOT / "stage2_dyadic_inverse_summary.json", summary)
        append_exec("Stage 2 dyadic inverse blocked", command_text(), files=rel(out), gpu=str(args.device), note=summary["dominant_blocker"])
        return summary
    device = torch.device(str(args.device))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"stage2_dyadic_inverse_matrix{suffix}.csv"
    existing = read_rows(matrix) if int(args.stage2_resume) else []
    done = {(r.get("scheme"), r.get("task"), str(r.get("seed"))) for r in existing if r.get("status") == "ok"}
    jobs = [job for idx, job in enumerate(stage2_jobs(args)) if idx % int(args.shard_count) == int(args.shard_index)]
    rows = list(existing)
    for scheme, task, seed in jobs:
        if (scheme, task, str(seed)) in done:
            continue
        rows.append(run_stage2_row(scheme, task, int(seed), args, device))
        if int(args.stage2_flush_every) > 0:
            write_rows(matrix, rows)
            print(json.dumps({"part": "2", "rows_written": len(rows), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
    write_rows(matrix, rows)
    summary = {"part": "2", "route": "Stage2ShardDone", "gate_pass": 0, "matrix": rel(matrix), "rows_written": len(rows), "ok_rows": sum(1 for row in rows if row.get("status") == "ok")}
    out = write_json(OUT_ROOT / f"stage2_dyadic_inverse_summary{suffix}.json", summary)
    append_exec("Stage 2 dyadic inverse shard", command_text(), files=f"{rel(matrix)}; {rel(out)}", gpu=str(args.device), note=f"rows={len(rows)}; ok={summary['ok_rows']}")
    return summary


def collect_stage2_rows() -> list[dict[str, str]]:
    shard_files = sorted(OUT_ROOT.glob("stage2_dyadic_inverse_matrix_shard*_of_*.csv"))
    files = shard_files if shard_files else [OUT_ROOT / "stage2_dyadic_inverse_matrix.csv"]
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in files:
        for row in read_rows(path):
            key = (row.get("scheme", ""), row.get("task", ""), str(row.get("seed", "")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def stage2_merge(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    rows = collect_stage2_rows()
    matrix = write_rows(OUT_ROOT / "stage2_dyadic_inverse_matrix.csv", rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for scheme in [item.strip() for item in str(args.stage2_schemes).split(",") if item.strip()]:
        group_rows = [row for row in ok if row.get("scheme") == scheme]
        groups.append({
            "scheme": scheme,
            "rows": len(group_rows),
            "total_C2_coverage_improvement_median": median([r.get("total_C2_coverage_improvement") for r in group_rows]),
            "total_C2_accuracy_improvement_median": median([r.get("total_C2_accuracy_improvement") for r in group_rows]),
            "total_guard_nll_delta_median": median([r.get("total_guard_nll_delta") for r in group_rows]),
            "F5_no_debt_count": sum(ival(r.get("F5_no_debt_pass")) for r in group_rows),
            "component_non_positive_rows": sum(ival(r.get("component_non_positive")) for r in group_rows),
            "terminal_total_audit_reject_count": sum(ival(r.get("terminal_total_audit_reject")) for r in group_rows),
            "inverse_terminal_alpha_selected_median": median([r.get("inverse_terminal_alpha_selected") for r in group_rows]),
            "raw_inverse_terminal_reject_count": sum(ival(r.get("raw_inverse_terminal_total_audit_reject")) for r in group_rows),
            "frame_design_drift_median": median([r.get("frame_design_drift_median") for r in group_rows]),
            "selected_chart_design_drift_median": median([r.get("selected_chart_design_drift_median") for r in group_rows]),
            "finite_step_accept_rate_median": median([r.get("finite_step_accept_rate") for r in group_rows]),
            "finite_step_skip_count_median": median([r.get("finite_step_skip_count") for r in group_rows]),
            "wall_time_s_median": median([r.get("wall_time_s") for r in group_rows]),
        })
    group_csv = write_rows(OUT_ROOT / "stage2_dyadic_inverse_group_summary.csv", groups)
    by_scheme = {g["scheme"]: g for g in groups}
    c48 = by_scheme.get("C48_Candidate_C45_dyadic_frame_then_terminal_C15_total_audit", {})
    c49 = by_scheme.get("C49_Control_C43_identity_frame_then_terminal_C15_total_audit", {})
    c50 = by_scheme.get("C50_Control_C47_random_dyadic_frame_then_terminal_C15_total_audit", {})
    c51 = by_scheme.get("C51_Control_C45_dyadic_frame_no_inverse_total_audit", {})
    expected_rows = len(stage2_jobs(args))
    c48_expected = sum(1 for scheme, _task, _seed in stage2_jobs(args) if scheme == "C48_Candidate_C45_dyadic_frame_then_terminal_C15_total_audit")
    c48_cov = fval(c48.get("total_C2_coverage_improvement_median"))
    checks = {
        "row_count_expected_ok": len(rows) == expected_rows and len(ok) == expected_rows,
        "no_fake_data": sum(ival(r.get("used_fake_data_rows")) for r in ok) == 0,
        "no_held_test_usage": sum(ival(r.get("held_test_usage")) for r in ok) == 0,
        "original_to_final_total_audit_all_rows": sum(ival(r.get("original_to_final_total_audit")) for r in ok) == len(ok),
        "post_frame_incremental_only_no_rows": sum(ival(r.get("post_frame_incremental_only")) for r in ok) == 0,
        "required_stage2_controls_present": all(by_scheme.get(s, {}).get("rows", 0) for s in STAGE2_SCHEMES),
        "c48_total_no_debt_clean": c48_expected > 0 and fval(c48.get("F5_no_debt_count")) == c48_expected and fval(c48.get("component_non_positive_rows")) == c48_expected and fval(c48.get("terminal_total_audit_reject_count")) == 0,
        "c48_minus_c49_ge_0p005": c48_cov - fval(c49.get("total_C2_coverage_improvement_median")) >= 0.005,
        "c48_minus_c50_ge_0p005": c48_cov - fval(c50.get("total_C2_coverage_improvement_median")) >= 0.005,
        "c48_minus_c51_ge_0p005": c48_cov - fval(c51.get("total_C2_coverage_improvement_median")) >= 0.005,
    }
    gate = int(all(checks.values()))
    if not checks["row_count_expected_ok"]:
        blocker = "unexpected_stage2_row_count_or_errors"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif not checks["required_stage2_controls_present"]:
        blocker = "missing_required_stage2_controls"
    elif not checks["c48_total_no_debt_clean"]:
        blocker = "c48_terminal_total_audit_reject_or_debt"
    elif not checks["c48_minus_c49_ge_0p005"]:
        blocker = "c48_does_not_beat_identity_frame_inverse"
    elif not checks["c48_minus_c50_ge_0p005"]:
        blocker = "c48_does_not_beat_random_dyadic_frame_inverse"
    elif not checks["c48_minus_c51_ge_0p005"]:
        blocker = "c48_does_not_beat_frame_only_no_inverse"
    else:
        blocker = "none"
    summary = {
        "part": "2",
        "route": "Stage2DyadicFrameInversePass" if gate else "Stage2DyadicFrameInverseFailed",
        "gate_pass": gate,
        "dominant_blocker": blocker,
        "checks": checks,
        "metrics": {
            "c48_coverage": c48_cov,
            "c49_coverage": fval(c49.get("total_C2_coverage_improvement_median")),
            "c50_coverage": fval(c50.get("total_C2_coverage_improvement_median")),
            "c51_coverage": fval(c51.get("total_C2_coverage_improvement_median")),
            "c48_minus_c49": c48_cov - fval(c49.get("total_C2_coverage_improvement_median")),
            "c48_minus_c50": c48_cov - fval(c50.get("total_C2_coverage_improvement_median")),
            "c48_minus_c51": c48_cov - fval(c51.get("total_C2_coverage_improvement_median")),
            "c48_alpha_selected_median": fval(c48.get("inverse_terminal_alpha_selected_median")),
            "c48_terminal_reject_count": fval(c48.get("terminal_total_audit_reject_count")),
            "c48_finite_step_accept_rate": fval(c48.get("finite_step_accept_rate_median")),
            "c48_wall_time_s_median": fval(c48.get("wall_time_s_median")),
        },
        "artifacts": {"matrix": rel(matrix), "group_summary": rel(group_csv)},
        "hashes": {"matrix": sha256_file(matrix), "group_summary": sha256_file(group_csv)},
        "next_action": "consider_full_controls_for_C48" if gate else "analyze_stage2_dyadic_frame_blocker",
    }
    out = write_json(OUT_ROOT / "stage2_dyadic_inverse_summary.json", summary)
    nxt = write_json(
        FORMAL_ROOT / "next_actions_for_v23_14.json",
        {
            "version": "v23.14",
            "last_completed_stage": "stage2_dyadic_frame_inverse",
            "stage2_gate_pass": gate,
            "stage2_route": summary["route"],
            "stage2_blocker": blocker,
            "promotion_allowed": 0,
            "completed_artifacts": {"stage2_summary": rel(out), "matrix": rel(matrix), "group_summary": rel(group_csv)},
            "allowed_next_actions": ["consider_full_controls_for_C48"] if gate else ["analyze_stage2_dyadic_frame_blocker"],
            "forbidden_next_actions": ["promote_C48_without_full_controls", "fabricate_data", "held_test_induction", "runtime_winner_selection", "add_mlp_stem_or_readout", "add_new_edge_function_family"],
        },
    )
    append_exec("Stage 2 dyadic-frame inverse merge", command_text(), files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Stage 2 dyadic-frame inverse", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--basis-key", default="dche_k9")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--train-size", type=int, default=768)
    p.add_argument("--guard-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--checkpoint", default="h10_isomorphic_partial")
    p.add_argument("--checkpoint-steps", type=int, default=20)
    p.add_argument("--checkpoint-lr", type=float, default=0.02)
    p.add_argument("--quadrature-points", type=int, default=257)
    p.add_argument("--functional-gram-ridge", type=float, default=1.0e-6)
    p.add_argument("--cg-tol", type=float, default=1.0e-6)
    p.add_argument("--cg-max-iter", type=int, default=512)
    p.add_argument("--part-d-lambda", type=float, default=1.0)
    p.add_argument("--part-d-solver", default="exact")
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--part-d-basis-input-gain", type=float, default=0.25)
    p.add_argument("--no-debt-budget", type=float, default=1.0e-8)
    p.add_argument("--chart-schemes", default=",".join(CHART_SCHEMES))
    p.add_argument("--chart-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--chart-seed-count", type=int, default=5)
    p.add_argument("--chart-resume", type=int, default=1)
    p.add_argument("--chart-flush-every", type=int, default=1)
    p.add_argument("--chart-scale-floor", type=float, default=0.5)
    p.add_argument("--chart-inv-scale-cap", type=float, default=4.0)
    p.add_argument("--chart-ridge", type=float, default=1.0e-6)
    p.add_argument("--chart-alphas", default="1,0.5,0.25,0.125,0.0625,0")
    p.add_argument("--frame-schemes", default=",".join(FRAME_SCHEMES))
    p.add_argument("--frame-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--frame-seed-count", type=int, default=5)
    p.add_argument("--frame-resume", type=int, default=1)
    p.add_argument("--frame-flush-every", type=int, default=1)
    p.add_argument("--frame-eval-float64", type=int, default=1)
    p.add_argument("--dyadic-schemes", default=",".join(DYADIC_SCHEMES))
    p.add_argument("--dyadic-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--dyadic-seed-count", type=int, default=5)
    p.add_argument("--dyadic-resume", type=int, default=1)
    p.add_argument("--dyadic-flush-every", type=int, default=1)
    p.add_argument("--dyadic-exp-cap", type=int, default=3)
    p.add_argument("--stage2-schemes", default=",".join(STAGE2_SCHEMES))
    p.add_argument("--stage2-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--stage2-seed-count", type=int, default=5)
    p.add_argument("--stage2-stage1i-summary", default="results/v23_14_source_rms_dyadic_frame_reduced_s5/stage1i_dyadic_frame_summary.json")
    p.add_argument("--stage2-c15-source-scheme", default="F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between")
    p.add_argument("--stage2-steps", type=int, default=80)
    p.add_argument("--stage2-residual-target", default="norm")
    p.add_argument("--stage2-lambda", type=float, default=10.0)
    p.add_argument("--stage2-alphas", default="1,0.5,0.25,0.125")
    p.add_argument("--stage2-terminal-alphas", default="1,0.5,0.25,0.125,0.0625,0")
    p.add_argument("--stage2-delta-scale", type=float, default=1.0)
    p.add_argument("--stage2-delta-sign", type=float, default=1.0)
    p.add_argument("--stage2-trust-mode", default="pareto")
    p.add_argument("--stage2-trust-tolerance", type=float, default=0.002)
    p.add_argument("--stage2-local-reference-target", default="norm")
    p.add_argument("--stage2-functionalgram-lr", type=float, default=0.0075)
    p.add_argument("--stage2-adam-lr", type=float, default=0.02)
    p.add_argument("--stage2-solver", default="exact")
    p.add_argument("--stage2-sketch-rank", type=int, default=16)
    p.add_argument("--stage2-activation-drift-cap", type=float, default=-1.0)
    p.add_argument("--stage2-resume", type=int, default=1)
    p.add_argument("--stage2-flush-every", type=int, default=1)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode).lower()
    if mode in {"stage0", "stage0-lock", "evidence-lock"}:
        return stage0_lock(args)
    if mode in {"stage1g-run", "chart-run"}:
        return chart_run(args)
    if mode in {"stage1g-merge", "stage1g", "chart-merge"}:
        return chart_merge(args)
    if mode in {"stage1h-run", "frame-run"}:
        return frame_run(args)
    if mode in {"stage1h-merge", "stage1h", "frame-merge"}:
        return frame_merge(args)
    if mode in {"stage1i-run", "dyadic-run"}:
        return dyadic_run(args)
    if mode in {"stage1i-merge", "stage1i", "dyadic-merge"}:
        return dyadic_merge(args)
    if mode in {"stage2-run", "dyadic-inverse-run"}:
        return stage2_run(args)
    if mode in {"stage2-merge", "stage2", "dyadic-inverse-merge"}:
        return stage2_merge(args)
    raise SystemExit(f"unknown v23.14 mode: {args.mode}")


if __name__ == "__main__":
    main()
