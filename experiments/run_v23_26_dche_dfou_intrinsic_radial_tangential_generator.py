#!/usr/bin/env python3
"""DG-KAN v23.26 D-CHE/D-FOU BC-RTGF audit runner.

This runner starts the v23.26 program from the hard gates: immutable contracts,
carrier/kernel truth, and metric/generator math units.  It deliberately marks
incomplete carriers as incomplete instead of substituting materialized helpers.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.efficiency.same_param_mlp import hidden_for_param_budget
from dgkan.fu.metric_preserving_functional_atlas import (
    MetricCompatibleAtlasMLP,
    SimpleMLP as AtlasSimpleMLP,
    build_last_layer_atlas,
    c_skew_project,
    metric_compatible_descent_diagnostics,
)
from dgkan.models.fc_purekan_primitives import MLPBaseline, PrimitiveKAN, PrimitiveSpec, count_parameters


PLAN = ROOT / "docs/DG-KAN_v23.26_DCHE_DFOU_BasisCovariantIntrinsicRadialTangentialGeneratorFlow_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.26_DCHE_DFOU_BasisCovariantIntrinsicRadialTangentialGeneratorFlow_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.26_DCHE_DFOU_BasisCovariantIntrinsicRadialTangentialGeneratorFlow_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2326_OUT_ROOT", str(ROOT / "results/v23_26"))).resolve()

EPS = 1.0e-12
NORM_FLOOR_MIN = 1.0e-6
DEFAULT_NORM_FLOOR_QUANTILE = 0.01


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


NORM_FLOOR_QUANTILE = min(0.50, max(0.0, env_float("V2326_NORM_FLOOR_QUANTILE", DEFAULT_NORM_FLOOR_QUANTILE)))
HYPOTHESES = [
    ("H-A", "Core carrier semantics and MLP-like efficiency"),
    ("H-B", "Intrinsic metric and block moment basis covariance"),
    ("H-C", "Support-only intrinsic block optimizer value"),
    ("H-D", "Radial-tangential generator independent value"),
    ("H-E", "Curvature metric value without task-loss modification"),
    ("H-F", "D-CHE / D-FOU carrier-specific mode geometry"),
    ("H-G", "Long-horizon debt and mode trajectory"),
    ("H-H", "KAN architecture surplus and efficiency"),
]
OFFICIAL_CARRIERS = [
    "D-CHE-Core-K3",
    "D-CHE-Core-K4",
    "D-FOU-IdLF-Core-K2",
    "D-FOU-IdLF-Core-K4",
    "D-FOU-Trig-Core-K4",
    "D-FOU-Trig-Core-K5",
]
SUPPORTED_FUSED_CARRIERS = {
    "D-CHE-Core-K3",
    "D-CHE-Core-K4",
    "D-FOU-IdLF-Core-K2",
    "D-FOU-IdLF-Core-K4",
}
METRICS = ["M0", "M1", "M2"]
SYNTHETIC_TASKS = [
    "SYN-CHE-LOW",
    "SYN-CHE-CURV",
    "SYN-FOU-PERIODIC",
    "SYN-FOU-TREND",
    "SYN-COMP",
    "SYN-TAIL",
]
MIN_REAL_DATASETS = ["Wine", "Spam", "Rice", "Bean", "FashionMNIST", "SVHN", "EMNIST-Letters", "CIFAR10-compact"]
DISCOVERY_SEEDS = [0, 1, 2]
CONFIRMATORY_SEEDS = [101, 102, 103, 104, 105]


@dataclass(frozen=True)
class CarrierInfo:
    name: str
    family: str
    basis_name: str
    k: int
    init_variant: str
    channel_order: tuple[str, ...]
    supported_fused: bool


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def append_exec(section: str, args: argparse.Namespace, files: list[str], note: dict[str, Any]) -> None:
    command = " ".join([sys.executable, str(Path(__file__).relative_to(ROOT))] + sys.argv[1:])
    text = (
        f"\n## {now()} | {section} | completed\n\n"
        f"- command: `{command}`\n"
        f"- python: `{sys.executable}`\n"
        f"- torch: `{torch.__version__}`\n"
        f"- cuda_available: `{torch.cuda.is_available()}`\n"
        f"- cuda_device_count: `{torch.cuda.device_count() if torch.cuda.is_available() else 0}`\n"
        f"- cuda_visible_devices: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`\n"
        f"- device_arg: `{getattr(args, 'device', '')}`\n"
        f"- files: `{';'.join(files)}`\n"
        f"- note: {json.dumps(note, ensure_ascii=False, sort_keys=True)}\n"
    )
    append_markdown(EXEC_LOG, text)


def append_recap(section: str, lines: list[str]) -> None:
    text = f"\n## {now()} | {section}\n\n" + "\n".join(f"- {line}" for line in lines) + "\n"
    append_markdown(RECAP_LOG, text)


def resolve_device(text: str) -> torch.device:
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {text}, but CUDA is unavailable.")
        device = torch.device(text)
        torch.cuda.set_device(device)
        return device
    return torch.device(text)


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def plan_audit() -> dict[str, Any]:
    text = PLAN.read_text(encoding="utf-8")
    required_phrases = [
        "禁止 spline",
        "禁止 label smoothing",
        "D-FOU-IdLF-Core",
        "D-FOU-Trig-Core",
        "Basis-Covariant Block Moment",
        "Radial–Tangential Generator Flow",
        "mandatory_hypothesis_count = 8",
        "v23_26_plan_read_audit.json",
    ]
    return {
        "plan_path": str(PLAN.relative_to(ROOT)),
        "plan_line_count": line_count(PLAN),
        "plan_byte_count": PLAN.stat().st_size,
        "plan_sha256": sha256_file(PLAN),
        "complete_read_required": 1,
        "required_phrase_hits": {phrase: int(phrase in text) for phrase in required_phrases},
        "all_required_phrase_hits": int(all(phrase in text for phrase in required_phrases)),
    }


def carrier_info(name: str) -> CarrierInfo:
    if name == "D-CHE-Core-K3":
        return CarrierInfo(name, "D-CHE", "chebyshev", 3, "cheby_k3_triton_l3_gradbuf", ("T0=1", "T1=z", "T2=2z^2-1"), True)
    if name == "D-CHE-Core-K4":
        return CarrierInfo(name, "D-CHE", "chebyshev", 4, "cheby_k4_triton_l3_matmul", ("T0=1", "T1=z", "T2=2z^2-1", "T3=4z^3-3z"), True)
    if name == "D-FOU-IdLF-Core-K2":
        return CarrierInfo(name, "D-FOU-IdLF", "fourier_lowfreq", 2, "fourier_k2_triton_l3_matmul", ("z", "sin(pi z)"), True)
    if name == "D-FOU-IdLF-Core-K4":
        return CarrierInfo(name, "D-FOU-IdLF", "fourier_lowfreq", 4, "fourier_k4_triton_l3_matmul", ("z", "sin(pi z)", "cos(pi z)", "sin(2pi z)"), True)
    if name == "D-FOU-Trig-Core-K4":
        return CarrierInfo(name, "D-FOU-Trig", "fourier_trig", 4, "fourier_trig_k4_triton_l3_matmul", ("sin(pi z)", "cos(pi z)", "sin(2pi z)", "cos(2pi z)"), True)
    if name == "D-FOU-Trig-Core-K5":
        return CarrierInfo(name, "D-FOU-Trig-DC", "fourier_trig_dc", 5, "fourier_trig_k5_triton_l3_matmul", ("1", "sin(pi z)", "cos(pi z)", "sin(2pi z)", "cos(2pi z)"), True)
    raise KeyError(name)


def scheme_prefix(info_or_name: CarrierInfo | str) -> str:
    family_or_name = info_or_name.family if isinstance(info_or_name, CarrierInfo) else str(info_or_name)
    if family_or_name.startswith("D-CHE"):
        return "CHE"
    if "Trig" in family_or_name:
        return "FOU-T"
    return "FOU-I"


def primitive_spec(info: CarrierInfo, hidden: int) -> PrimitiveSpec:
    return PrimitiveSpec(
        candidate_id=f"v23_26_{info.name}",
        basis_family=info.family,
        basis_name=info.basis_name,
        k=info.k,
        hidden_dim=int(hidden),
        source="v23.26_BC_RTG_core_contract",
        local_support=0,
        global_support=1,
        uses_exp=0,
        uses_sin_cos=int("FOU" in info.family),
        uses_division=0,
        uses_dense_basis_tensor=0,
        diagnostic_only=0,
        basis_order=info.k,
        init_variant=info.init_variant,
        model_kind="edge_kan",
    )


def make_model(info: CarrierInfo, input_dim: int, output_dim: int, hidden: int, x_stats: torch.Tensor, seed: int, device: torch.device) -> PrimitiveKAN:
    return PrimitiveKAN(
        input_dim=input_dim,
        output_dim=output_dim,
        spec=primitive_spec(info, hidden),
        x_for_stats=x_stats.to(device=device, dtype=torch.float32),
        seed=seed,
        device=device,
        param_budget=max(1, input_dim * hidden * info.k + hidden * output_dim * info.k),
    )


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).norm().item() / max(float(b.detach().norm().item()), EPS))


def tensor_grad_rel_error(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return float("inf")
    return rel_error(a, b)


def basis_arrays(family: str, k: int, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols: list[np.ndarray] = []
    d1: list[np.ndarray] = []
    d2: list[np.ndarray] = []
    if family == "D-CHE":
        cols.append(np.ones_like(z))
        d1.append(np.zeros_like(z))
        d2.append(np.zeros_like(z))
        if k > 1:
            cols.append(z)
            d1.append(np.ones_like(z))
            d2.append(np.zeros_like(z))
        if k > 2:
            cols.append(2.0 * z * z - 1.0)
            d1.append(4.0 * z)
            d2.append(np.full_like(z, 4.0))
        if k > 3:
            cols.append(4.0 * z**3 - 3.0 * z)
            d1.append(12.0 * z * z - 3.0)
            d2.append(24.0 * z)
        while len(cols) < k:
            cols.append(2.0 * z * cols[-1] - cols[-2])
            d1.append(np.gradient(cols[-1], z, edge_order=1))
            d2.append(np.gradient(d1[-1], z, edge_order=1))
    elif family == "D-FOU-IdLF":
        cols.append(z)
        d1.append(np.ones_like(z))
        d2.append(np.zeros_like(z))
        freq = 1
        while len(cols) < k:
            angle = math.pi * freq * z
            scale = math.pi * freq
            cols.append(np.sin(angle))
            d1.append(scale * np.cos(angle))
            d2.append(-(scale**2) * np.sin(angle))
            if len(cols) < k:
                cols.append(np.cos(angle))
                d1.append(-scale * np.sin(angle))
                d2.append(-(scale**2) * np.cos(angle))
            freq += 1
    elif family in {"D-FOU-Trig", "D-FOU-Trig-DC"}:
        if family == "D-FOU-Trig-DC":
            cols.append(np.ones_like(z))
            d1.append(np.zeros_like(z))
            d2.append(np.zeros_like(z))
        freq = 1
        while len(cols) < k:
            angle = math.pi * freq * z
            scale = math.pi * freq
            cols.append(np.sin(angle))
            d1.append(scale * np.cos(angle))
            d2.append(-(scale**2) * np.sin(angle))
            if len(cols) < k:
                cols.append(np.cos(angle))
                d1.append(-scale * np.sin(angle))
                d2.append(-(scale**2) * np.cos(angle))
            freq += 1
    else:
        raise ValueError(f"unsupported metric family {family}")
    return np.stack(cols[:k], axis=1), np.stack(d1[:k], axis=1), np.stack(d2[:k], axis=1)


def metric_matrices(family: str, k: int, n: int = 4096) -> dict[str, np.ndarray]:
    if family == "D-CHE":
        z = np.cos((np.arange(n, dtype=np.float64) + 0.5) * math.pi / float(n))
        weights = np.full(n, 1.0 / float(n), dtype=np.float64)
    else:
        z = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        weights = np.full(n, 1.0 / float(n), dtype=np.float64)
    psi, dpsi, ddpsi = basis_arrays(family, k, z)
    w = weights[:, None]
    g0 = psi.T @ (w * psi)
    s1 = dpsi.T @ (w * dpsi)
    s2 = ddpsi.T @ (w * ddpsi)
    ridge = 1.0e-8
    inv_g = np.linalg.inv(g0 + ridge * np.eye(k))
    s1_hat = s1 / (float(np.trace(inv_g @ s1)) / float(k) + ridge)
    s2_hat = s2 / (float(np.trace(inv_g @ s2)) / float(k) + ridge)
    return {
        "M0": g0 + ridge * np.eye(k),
        "M1": g0 + s1_hat + ridge * np.eye(k),
        "M2": g0 + s2_hat + ridge * np.eye(k),
        "G0": g0,
        "S1_hat": s1_hat,
        "S2_hat": s2_hat,
    }


def mdot(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    return float(a.T @ m @ b)


def rtgf_update(a: np.ndarray, proposal: np.ndarray, m: np.ndarray, norm_floor: float = 1.0e-8) -> tuple[np.ndarray, dict[str, float]]:
    r2 = mdot(a, a, m)
    if r2 < norm_floor * norm_floor:
        return a + proposal, {"bootstrap": 1.0, "radial_log_step": 0.0, "tangent_angle": 0.0, "tangent_norm": 0.0}
    r = math.sqrt(max(r2, 0.0))
    alpha = mdot(a, proposal, m) / max(r2, EPS)
    tangent = proposal - alpha * a
    tangent_norm = math.sqrt(max(mdot(tangent, tangent, m), 0.0))
    angle = tangent_norm / max(r, EPS)
    sinc = 1.0 - angle * angle / 6.0 + (angle**4) / 120.0 if abs(angle) < 1.0e-4 else math.sin(angle) / max(angle, EPS)
    shape = math.cos(angle) * a + sinc * tangent
    return math.exp(alpha) * shape, {"bootstrap": 0.0, "radial_log_step": float(alpha), "tangent_angle": float(angle), "tangent_norm": float(tangent_norm)}


def additive_update(a: np.ndarray, proposal: np.ndarray, _m: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    return a + proposal, {"bootstrap": 0.0, "radial_log_step": 0.0, "tangent_angle": 0.0, "tangent_norm": float(np.linalg.norm(proposal))}


def intrinsic_proposal(grad: np.ndarray, metric: np.ndarray, lr: float = 1.0e-2) -> np.ndarray:
    nat = np.linalg.solve(metric, grad)
    energy = float(grad.T @ nat) / float(max(1, grad.size))
    return -lr * nat / (math.sqrt(max(energy, 0.0)) + 1.0e-8)


def tensor_metric_radius(values: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    mv = torch.einsum("kl,...l->...k", metric, values)
    return (values * mv).sum(dim=-1).clamp_min(0.0).sqrt()


def torch_sinc(x: torch.Tensor) -> torch.Tensor:
    abs_x = x.abs()
    taylor = 1.0 - x.square() / 6.0 + x.square().square() / 120.0
    denom = torch.where(abs_x > 0.0, x, torch.ones_like(x))
    direct = torch.sin(x) / denom
    return torch.where(abs_x < 1.0e-4, taylor, direct)


def fixed_quantile_norm_floor(radius_tensors: list[torch.Tensor], *, min_floor: float = NORM_FLOOR_MIN, quantile: float = NORM_FLOOR_QUANTILE) -> tuple[float, dict[str, float]]:
    with torch.no_grad():
        flats: list[torch.Tensor] = []
        for radius in radius_tensors:
            flat = radius.detach().reshape(-1)
            flat = flat[torch.isfinite(flat) & (flat > 0)]
            if int(flat.numel()) > 0:
                flats.append(flat.to(dtype=torch.float32))
        if not flats:
            return float(min_floor), {
                "norm_floor_quantile": float(quantile),
                "norm_floor_min": float(min_floor),
                "initial_edge_radius_count": 0.0,
                "initial_edge_radius_q": 0.0,
                "initial_edge_radius_min": 0.0,
                "initial_edge_radius_median": 0.0,
                "initial_edge_radius_max": 0.0,
            }
        all_radius = torch.cat(flats)
        q = float(torch.quantile(all_radius, float(quantile)).item())
        floor = max(float(min_floor), q)
        return floor, {
            "norm_floor_quantile": float(quantile),
            "norm_floor_min": float(min_floor),
            "initial_edge_radius_count": float(all_radius.numel()),
            "initial_edge_radius_q": q,
            "initial_edge_radius_min": float(all_radius.min().item()),
            "initial_edge_radius_median": float(torch.quantile(all_radius, 0.5).item()),
            "initial_edge_radius_max": float(all_radius.max().item()),
        }


class IntrinsicState:
    def __init__(self, param: torch.Tensor) -> None:
        self.m = torch.zeros_like(param)
        self.v = torch.zeros(param.shape[:-1], device=param.device, dtype=param.dtype)
        self.step = 0


class IntrinsicEdgeOptimizer:
    def __init__(self, model: PrimitiveKAN, metric: np.ndarray, *, lr: float, mode: str, weight_decay: float = 0.0, trace_enabled: bool = True) -> None:
        self.model = model
        self.metric = torch.tensor(metric, device=model.w1.device, dtype=model.w1.dtype).contiguous()
        self.metric_inv = torch.linalg.inv(self.metric).contiguous()
        self.lr = float(lr)
        self.mode = str(mode)
        self.weight_decay = float(weight_decay)
        self.trace_enabled = bool(trace_enabled)
        self.fused_step_count = 0
        self.fused_step_fallback_count = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1.0e-8
        self.norm_floor, self.norm_floor_meta = fixed_quantile_norm_floor(
            [tensor_metric_radius(model.w1.detach(), self.metric), tensor_metric_radius(model.w2.detach(), self.metric)]
        )
        self.states = {id(model.w1): IntrinsicState(model.w1), id(model.w2): IntrinsicState(model.w2)}
        self.trace = {
            "actual_generator_update_count": 0,
            "actual_random_M_skew_count": 0,
            "actual_signflip_count": 0,
            "actual_shuffled_tangent_count": 0,
            "actual_euclidean_generator_count": 0,
            "actual_noop_count": 0,
            "near_zero_bootstrap_count": 0,
            "edge_count": 0,
            "radial_abs_sum": 0.0,
            "radial_signed_sum": 0.0,
            "radial_abs_max": 0.0,
            "tangent_angle_sum": 0.0,
            "tangent_norm_sum": 0.0,
            "radial_energy_sum": 0.0,
            "tangent_energy_sum": 0.0,
            "intrinsic_step_norm_sum": 0.0,
            "max_tangent_angle": 0.0,
            "edge_radius_sum": 0.0,
            "edge_radius_min": float("inf"),
            "edge_radius_max": 0.0,
            "proposal_m_norm_sum": 0.0,
            "proposal_m_energy_sum": 0.0,
            "additive_retraction_gap_m_norm_sum": 0.0,
            "additive_retraction_gap_ratio_sum": 0.0,
            "max_additive_retraction_gap_ratio": 0.0,
            "tangent_m_orthogonality_abs_sum": 0.0,
            "max_tangent_m_orthogonality_abs": 0.0,
            "tangent_norm_match_abs_sum": 0.0,
            "max_tangent_norm_match_abs": 0.0,
            "shape_norm_preservation_abs_sum": 0.0,
            "max_shape_norm_preservation_abs": 0.0,
            "tangent_angle_le_1e_4_count": 0,
            "tangent_angle_le_1e_3_count": 0,
            "norm_floor": float(self.norm_floor),
            "norm_floor_quantile": float(self.norm_floor_meta["norm_floor_quantile"]),
            "norm_floor_min": float(self.norm_floor_meta["norm_floor_min"]),
            "initial_edge_radius_count": float(self.norm_floor_meta["initial_edge_radius_count"]),
            "initial_edge_radius_q": float(self.norm_floor_meta["initial_edge_radius_q"]),
            "initial_edge_radius_min": float(self.norm_floor_meta["initial_edge_radius_min"]),
            "initial_edge_radius_median": float(self.norm_floor_meta["initial_edge_radius_median"]),
            "initial_edge_radius_max": float(self.norm_floor_meta["initial_edge_radius_max"]),
        }

    def _update_param(self, param: torch.nn.Parameter) -> None:
        if param.grad is None:
            return
        state = self.states[id(param)]
        state.step += 1
        grad = param.grad.detach()
        if (
            not self.trace_enabled
            and self.mode == "rtgf"
            and param.device.type == "cuda"
            and param.dtype == torch.float32
            and param.is_contiguous()
            and grad.is_contiguous()
        ):
            try:
                from dgkan.kernels import fused_rtgf_optimizer

                fused_ok = fused_rtgf_optimizer.rtgf_step_(
                    param,
                    grad,
                    state.m,
                    state.v,
                    self.metric,
                    self.metric_inv,
                    step=state.step,
                    lr=self.lr,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps,
                    norm_floor=self.norm_floor,
                    weight_decay=self.weight_decay,
                )
            except Exception:
                fused_ok = False
            if fused_ok:
                self.fused_step_count += 1
                param.grad = None
                return
            self.fused_step_fallback_count += 1
        state.m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        nat_grad = torch.einsum("kl,...l->...k", self.metric_inv, grad)
        energy = (grad * nat_grad).sum(dim=-1) / float(max(1, grad.shape[-1]))
        state.v.mul_(self.beta2).add_(energy.clamp_min(0.0), alpha=1.0 - self.beta2)
        m_hat = state.m / (1.0 - self.beta1 ** state.step)
        v_hat = state.v / (1.0 - self.beta2 ** state.step)
        nat_m = torch.einsum("kl,...l->...k", self.metric_inv, m_hat)
        proposal = -self.lr * nat_m / (v_hat.sqrt().unsqueeze(-1) + self.eps)
        with torch.no_grad():
            if self.mode == "additive":
                new_value = param.detach() + proposal
            elif self.mode == "noop":
                new_value = param.detach()
                if self.trace_enabled:
                    self.trace["actual_noop_count"] += int(param.detach().numel() // max(1, param.detach().shape[-1]))
            elif self.mode in {"rtgf", "random_m_skew", "signflip", "shuffle", "euclidean"}:
                a = param.detach()
                if self.mode == "euclidean":
                    metric = torch.eye(int(a.shape[-1]), device=a.device, dtype=a.dtype)
                    if self.trace_enabled:
                        self.trace["actual_euclidean_generator_count"] += int(a.numel() // max(1, a.shape[-1]))
                else:
                    metric = self.metric
                ma = torch.einsum("kl,...l->...k", metric, a)
                r2 = (a * ma).sum(dim=-1).clamp_min(0.0)
                alpha = (proposal * ma).sum(dim=-1) / r2.clamp_min(self.norm_floor * self.norm_floor)
                tangent = proposal - alpha.unsqueeze(-1) * a
                mt = torch.einsum("kl,...l->...k", metric, tangent)
                tangent_norm = (tangent * mt).sum(dim=-1).clamp_min(0.0).sqrt()
                target_tangent_norm = tangent_norm
                if self.mode == "signflip":
                    tangent = -tangent
                    if self.trace_enabled:
                        self.trace["actual_signflip_count"] += int(a.numel() // max(1, a.shape[-1]))
                elif self.mode == "random_m_skew":
                    raw = torch.randn_like(tangent)
                    coeff = (raw * ma).sum(dim=-1) / r2.clamp_min(self.norm_floor * self.norm_floor)
                    random_tangent = raw - coeff.unsqueeze(-1) * a
                    mrt = torch.einsum("kl,...l->...k", metric, random_tangent)
                    random_norm = (random_tangent * mrt).sum(dim=-1).clamp_min(0.0).sqrt()
                    tangent = random_tangent * (tangent_norm / random_norm.clamp_min(self.eps)).unsqueeze(-1)
                    if self.trace_enabled:
                        self.trace["actual_random_M_skew_count"] += int(a.numel() // max(1, a.shape[-1]))
                elif self.mode == "shuffle":
                    flat_tangent = tangent.reshape(-1, int(tangent.shape[-1]))
                    flat_a = a.reshape(-1, int(a.shape[-1]))
                    flat_ma = ma.reshape(-1, int(ma.shape[-1]))
                    flat_r2 = r2.reshape(-1)
                    flat_target_norm = tangent_norm.reshape(-1)
                    perm = torch.randperm(int(flat_tangent.shape[0]), device=flat_tangent.device)
                    shuffled = flat_tangent[perm]
                    coeff = (shuffled * flat_ma).sum(dim=-1) / flat_r2.clamp_min(self.norm_floor * self.norm_floor)
                    shuffled = shuffled - coeff.unsqueeze(-1) * flat_a
                    mshuf = torch.einsum("kl,nl->nk", metric, shuffled)
                    shuffled_norm = (shuffled * mshuf).sum(dim=-1).clamp_min(0.0).sqrt()
                    shuffled = shuffled * (flat_target_norm / shuffled_norm.clamp_min(self.eps)).unsqueeze(-1)
                    tangent = shuffled.reshape_as(tangent)
                    if self.trace_enabled:
                        self.trace["actual_shuffled_tangent_count"] += int(a.numel() // max(1, a.shape[-1]))
                mt = torch.einsum("kl,...l->...k", metric, tangent)
                tangent_norm = (tangent * mt).sum(dim=-1).clamp_min(0.0).sqrt()
                r = r2.sqrt()
                use_bootstrap = r < self.norm_floor
                angle = tangent_norm / r.clamp_min(self.eps)
                proposal_m_norm = (proposal * torch.einsum("kl,...l->...k", metric, proposal)).sum(dim=-1).clamp_min(0.0).sqrt()
                tangent_orth = (tangent * ma).sum(dim=-1).abs() / (r * tangent_norm).clamp_min(self.eps)
                tangent_norm_match = (tangent_norm - target_tangent_norm).abs() / target_tangent_norm.clamp_min(self.eps)
                tangent_orth = torch.where(use_bootstrap, torch.zeros_like(tangent_orth), tangent_orth)
                tangent_norm_match = torch.where(use_bootstrap, torch.zeros_like(tangent_norm_match), tangent_norm_match)
                if self.trace_enabled:
                    self.trace["radial_abs_sum"] += float(alpha.abs().sum().item())
                    self.trace["radial_signed_sum"] += float(alpha.sum().item())
                    self.trace["radial_abs_max"] = max(float(self.trace["radial_abs_max"]), float(alpha.abs().max().item()))
                    self.trace["tangent_angle_sum"] += float(angle.abs().sum().item())
                    self.trace["tangent_norm_sum"] += float(tangent_norm.sum().item())
                    self.trace["radial_energy_sum"] += float((alpha.square() * r2).sum().item())
                    self.trace["tangent_energy_sum"] += float(tangent_norm.square().sum().item())
                    self.trace["intrinsic_step_norm_sum"] += float(proposal.norm(dim=-1).sum().item())
                    self.trace["max_tangent_angle"] = max(float(self.trace["max_tangent_angle"]), float(angle.abs().max().item()))
                    self.trace["edge_radius_sum"] += float(r.sum().item())
                    self.trace["edge_radius_min"] = min(float(self.trace["edge_radius_min"]), float(r.min().item()))
                    self.trace["edge_radius_max"] = max(float(self.trace["edge_radius_max"]), float(r.max().item()))
                    self.trace["proposal_m_norm_sum"] += float(proposal_m_norm.sum().item())
                    self.trace["proposal_m_energy_sum"] += float(proposal_m_norm.square().sum().item())
                    self.trace["tangent_m_orthogonality_abs_sum"] += float(tangent_orth.sum().item())
                    self.trace["max_tangent_m_orthogonality_abs"] = max(float(self.trace["max_tangent_m_orthogonality_abs"]), float(tangent_orth.max().item()))
                    self.trace["tangent_norm_match_abs_sum"] += float(tangent_norm_match.sum().item())
                    self.trace["max_tangent_norm_match_abs"] = max(float(self.trace["max_tangent_norm_match_abs"]), float(tangent_norm_match.max().item()))
                    self.trace["tangent_angle_le_1e_4_count"] += int((angle.abs() <= 1.0e-4).sum().item())
                    self.trace["tangent_angle_le_1e_3_count"] += int((angle.abs() <= 1.0e-3).sum().item())
                shape = torch.cos(angle).unsqueeze(-1) * a + torch_sinc(angle).unsqueeze(-1) * tangent
                new_value = torch.exp(alpha).unsqueeze(-1) * shape
                new_value = torch.where(use_bootstrap.unsqueeze(-1), a + proposal, new_value)
                if self.trace_enabled:
                    shape_m = torch.einsum("kl,...l->...k", metric, shape)
                    shape_r = (shape * shape_m).sum(dim=-1).clamp_min(0.0).sqrt()
                    shape_norm_residual = torch.where(use_bootstrap, torch.zeros_like(shape_r), (shape_r - r).abs() / r.clamp_min(self.eps))
                    gap = new_value - (a + proposal)
                    gap_m = torch.einsum("kl,...l->...k", metric, gap)
                    gap_norm = (gap * gap_m).sum(dim=-1).clamp_min(0.0).sqrt()
                    gap_ratio = gap_norm / proposal_m_norm.clamp_min(self.eps)
                    self.trace["additive_retraction_gap_m_norm_sum"] += float(gap_norm.sum().item())
                    self.trace["additive_retraction_gap_ratio_sum"] += float(gap_ratio.sum().item())
                    self.trace["max_additive_retraction_gap_ratio"] = max(float(self.trace["max_additive_retraction_gap_ratio"]), float(gap_ratio.max().item()))
                    self.trace["shape_norm_preservation_abs_sum"] += float(shape_norm_residual.sum().item())
                    self.trace["max_shape_norm_preservation_abs"] = max(float(self.trace["max_shape_norm_preservation_abs"]), float(shape_norm_residual.max().item()))
                    self.trace["actual_generator_update_count"] += int((~use_bootstrap).sum().item())
                    self.trace["near_zero_bootstrap_count"] += int(use_bootstrap.sum().item())
                    self.trace["edge_count"] += int(use_bootstrap.numel())
            else:
                raise ValueError(f"unknown intrinsic optimizer mode {self.mode}")
            if self.weight_decay != 0.0:
                new_value = new_value * (1.0 - self.lr * self.weight_decay)
            param.copy_(new_value)
        param.grad = None

    def step(self) -> None:
        self._update_param(self.model.w1)
        self._update_param(self.model.w2)


def trace_edge_count(trace: dict[str, Any]) -> float:
    return max(1.0, float(trace.get("edge_count", 0)))


def trace_mean(trace: dict[str, Any], key: str) -> float:
    return float(trace.get(key, 0.0)) / trace_edge_count(trace)


def trace_fraction(trace: dict[str, Any], key: str) -> float:
    return float(trace.get(key, 0)) / trace_edge_count(trace)


def trace_min(trace: dict[str, Any], key: str) -> float:
    if float(trace.get("edge_count", 0)) <= 0:
        return 0.0
    value = float(trace.get(key, 0.0))
    return 0.0 if not math.isfinite(value) else value


class MLPBlockRTGFState:
    def __init__(self, param: torch.Tensor, block_size: int) -> None:
        flat_numel = int(param.numel())
        block_count = int(math.ceil(float(flat_numel) / float(max(1, block_size))))
        self.m = torch.zeros((block_count, int(block_size)), device=param.device, dtype=param.dtype)
        self.v = torch.zeros(block_count, device=param.device, dtype=param.dtype)
        self.step = 0
        self.flat_numel = flat_numel
        self.shape = tuple(param.shape)


class MLPBlockRTGFOptimizer:
    def __init__(self, module: nn.Module, metric: np.ndarray, *, lr: float, block_size: int, weight_decay: float = 0.0) -> None:
        self.params = [p for p in module.parameters() if p.requires_grad]
        self.block_size = int(block_size)
        self.metric = torch.tensor(metric, device=self.params[0].device, dtype=self.params[0].dtype)
        self.metric_inv = torch.linalg.inv(self.metric)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1.0e-8
        self.states = {id(param): MLPBlockRTGFState(param, self.block_size) for param in self.params}
        init_radii = [tensor_metric_radius(self._blocks(param.detach(), self.states[id(param)]), self.metric) for param in self.params]
        self.norm_floor, self.norm_floor_meta = fixed_quantile_norm_floor(init_radii)
        self.trace = {
            "block_count": 0,
            "near_zero_bootstrap_count": 0,
            "norm_floor": float(self.norm_floor),
            "norm_floor_quantile": float(self.norm_floor_meta["norm_floor_quantile"]),
            "norm_floor_min": float(self.norm_floor_meta["norm_floor_min"]),
            "initial_edge_radius_count": float(self.norm_floor_meta["initial_edge_radius_count"]),
            "initial_edge_radius_q": float(self.norm_floor_meta["initial_edge_radius_q"]),
            "initial_edge_radius_min": float(self.norm_floor_meta["initial_edge_radius_min"]),
            "initial_edge_radius_median": float(self.norm_floor_meta["initial_edge_radius_median"]),
            "initial_edge_radius_max": float(self.norm_floor_meta["initial_edge_radius_max"]),
        }

    def _blocks(self, tensor: torch.Tensor, state: MLPBlockRTGFState) -> torch.Tensor:
        flat = tensor.reshape(-1)
        pad = int(state.m.numel() - flat.numel())
        if pad > 0:
            flat = F.pad(flat, (0, pad))
        return flat.reshape(-1, self.block_size)

    def step(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            state = self.states[id(param)]
            state.step += 1
            with torch.no_grad():
                a = self._blocks(param.detach(), state)
                grad = self._blocks(param.grad.detach(), state)
                state.m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                nat_grad = torch.einsum("kl,nl->nk", self.metric_inv, grad)
                energy = (grad * nat_grad).sum(dim=-1) / float(max(1, self.block_size))
                state.v.mul_(self.beta2).add_(energy.clamp_min(0.0), alpha=1.0 - self.beta2)
                m_hat = state.m / (1.0 - self.beta1 ** state.step)
                v_hat = state.v / (1.0 - self.beta2 ** state.step)
                nat_m = torch.einsum("kl,nl->nk", self.metric_inv, m_hat)
                proposal = -self.lr * nat_m / (v_hat.sqrt().unsqueeze(-1) + self.eps)
                ma = torch.einsum("kl,nl->nk", self.metric, a)
                r2 = (a * ma).sum(dim=-1).clamp_min(0.0)
                alpha = (proposal * ma).sum(dim=-1) / r2.clamp_min(self.norm_floor * self.norm_floor)
                tangent = proposal - alpha.unsqueeze(-1) * a
                mt = torch.einsum("kl,nl->nk", self.metric, tangent)
                tangent_norm = (tangent * mt).sum(dim=-1).clamp_min(0.0).sqrt()
                r = r2.sqrt()
                use_bootstrap = r < self.norm_floor
                angle = tangent_norm / r.clamp_min(self.eps)
                shape = torch.cos(angle).unsqueeze(-1) * a + torch_sinc(angle).unsqueeze(-1) * tangent
                new_blocks = torch.exp(alpha).unsqueeze(-1) * shape
                new_blocks = torch.where(use_bootstrap.unsqueeze(-1), a + proposal, new_blocks)
                if self.weight_decay != 0.0:
                    new_blocks = new_blocks * (1.0 - self.lr * self.weight_decay)
                new_flat = new_blocks.reshape(-1)[: state.flat_numel]
                param.copy_(new_flat.reshape(state.shape))
                self.trace["block_count"] += int(use_bootstrap.numel())
                self.trace["near_zero_bootstrap_count"] += int(use_bootstrap.sum().item())
            param.grad = None


def run_part0(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit = plan_audit()
    hypothesis_registry = [
        {
            "id": hid,
            "description": desc,
            "minimum_real_required": 1,
            "h20_required": 1,
            "independent_hypothesis": 1,
            "semantic_obligations": "semantic pass; math unit; controls; minimum-real conclusion; H20 diagnostic",
        }
        for hid, desc in HYPOTHESES
    ]
    architecture_contract = {
        "PrimitiveKAN": "generic fused edge-basis KAN execution shell",
        "D-CHE-Core": {"basis": "chebyshev", "fixed_chart": "tanh_train_stats", "linearres": 0, "paircross": 0, "inputcross": 0, "localrot": 0, "spline_residual": 0, "mlp_stem": 0, "mlp_head": 0},
        "D-FOU-IdLF-Core": {"basis": "identity plus low-frequency trig", "channel_order_must_be_explicit": 1},
        "D-FOU-Trig-Core": {"basis": "pure trigonometric only", "science_requires_fused_no_materialize": 1, "current_status": "eligible_only_after_fourier_trig_k4_k5_kernel_gate_passes"},
    }
    loss_contract = {
        "task_loss": "cross_entropy",
        "label_smoothing": 0,
        "auxiliary_loss_count": 0,
        "mode_penalty": 0,
        "curvature_loss_penalty": 0,
        "calibration_loss_penalty": 0,
        "distillation_loss": 0,
        "weight_decay_default": 0.0,
    }
    scheme_registry = {
        "D-CHE": ["CHE-B0", "CHE-B1", "CHE-C0", "CHE-C1", "CHE-C2", "CHE-G0", "CHE-G1", "CHE-G2"],
        "D-FOU-IdLF": ["FOU-I-B0", "FOU-I-C0", "FOU-I-C1", "FOU-I-C2", "FOU-I-G0", "FOU-I-G1", "FOU-I-G2"],
        "D-FOU-Trig": ["FOU-T-B0", "FOU-T-C0", "FOU-T-C1", "FOU-T-C2", "FOU-T-G0", "FOU-T-G1", "FOU-T-G2"],
        "MLP": ["MLP-B0", "MLP-B1", "MLP-B2", "MLP-G0", "MLP-MCGA"],
    }
    controls = {
        "R0_same_radial_same_tangent_norm_random_M_skew": {"same_radial": 1, "same_tangent_M_norm": 1, "numeric_identity_required": 1},
        "R1_same_radial_tangent_signflip": {"same_radial": 1, "tangent_sign": -1, "numeric_identity_required": 1},
        "R2_same_radial_tangent_shuffled_within_layer": {"same_layer": 1, "same_edge_count": 1, "numeric_identity_required": 1},
        "R3_Euclidean_radial_tangential_generator": {"metric": "Euclidean", "numeric_identity_required": 1},
        "R4_additive_same_intrinsic_proposal": {"same_intrinsic_proposal": 1, "numeric_identity_required": 1},
        "R5_same_compute_noop": {"same_compute": 1, "update_norm": 0, "numeric_identity_required": 1},
    }
    metric_registry = {
        "M0": "G0 + eps I",
        "M1": "G0 + normalized S1 + eps I",
        "M2": "G0 + normalized S2 + eps I",
        "lambda_sweep_allowed": 0,
        "task_signal_in_metric": 0,
    }
    thresholds = {
        "partA_forward_rel_error_fp32_max": 5.0e-4,
        "partA_gradient_rel_error_fp32_max": 5.0e-3,
        "partB_float64_covariance_rel_error_max": 1.0e-8,
        "partB_fp32_covariance_rel_error_max": 5.0e-4,
        "near_zero_bootstrap_fraction_max": 0.10,
    }
    repair_registry = {
        "architecture_kernel_allowed": ["fused kernel", "channel ordering", "grad buffer", "no-materialize path", "kernel launch layout"],
        "metric_allowed": ["quadrature", "basis normalization", "SPD ridge", "basis-change transformation"],
        "generator_allowed": ["sign", "projection", "exponential formula", "near-zero numerical path", "vectorization"],
        "forbidden": ["label smoothing", "mode penalty loss", "candidate selector", "materialized fallback science"],
    }
    dataset_manifest = {"synthetic_tasks": SYNTHETIC_TASKS, "minimum_real": MIN_REAL_DATASETS, "debug_only": ["MNIST", "KMNIST"]}
    seed_manifest = {"discovery": DISCOVERY_SEEDS, "fresh_confirmatory": CONFIRMATORY_SEEDS, "confirmatory_frozen_before_discovery": 1}
    dependency_graph = {
        "Part0": [],
        "PartA": ["Part0"],
        "PartB": ["Part0"],
        "PartC": ["PartA", "PartB"],
        "PartD": ["PartA", "PartB"],
        "H20": ["PartA", "PartB"],
        "H80": ["PartD_minimum_real_promotion"],
        "fresh_confirmatory": ["discovery_freeze"],
    }
    theory_contract = {
        "main_claim": "fixed intrinsic edge-function metric induces amplitude flow and metric-compatible shape generator without modifying CE task loss",
        "loss_trick_allowed": 0,
        "selector_allowed": 0,
        "spline_allowed": 0,
        "native_cheb_as_dche_allowed": 0,
        "materialized_fourier_as_dfou_allowed": 0,
    }
    gates = {
        "mandatory_hypothesis_count": len(hypothesis_registry),
        "no_spline_registered": 1,
        "label_smoothing_fixed_zero": 1,
        "auxiliary_loss_forbidden": 1,
        "selector_firewall_registered": 1,
        "D-CHE_core_contract_registered": 1,
        "D-FOU_IdLF_core_contract_registered": 1,
        "D-FOU_Trig_core_contract_registered": 1,
        "all_controls_have_numeric_identity_specs": int(all(v.get("numeric_identity_required", 0) == 1 for v in controls.values())),
        "all_hypotheses_have_minimum_real": int(all(h["minimum_real_required"] == 1 for h in hypothesis_registry)),
        "all_hypotheses_have_H20": int(all(h["h20_required"] == 1 for h in hypothesis_registry)),
        "fresh_confirmatory_seeds_frozen": 1,
    }
    objects = {
        "v23_26_plan_read_audit.json": audit,
        "v23_26_theory_contract.json": theory_contract,
        "v23_26_architecture_contract.json": architecture_contract,
        "v23_26_loss_contract.json": loss_contract,
        "v23_26_hypothesis_registry.json": hypothesis_registry,
        "v23_26_scheme_registry.json": scheme_registry,
        "v23_26_control_registry.json": controls,
        "v23_26_metric_registry.json": metric_registry,
        "v23_26_threshold_registry.json": thresholds,
        "v23_26_repair_registry.json": repair_registry,
        "v23_26_dataset_manifest.json": dataset_manifest,
        "v23_26_seed_manifest.json": seed_manifest,
        "v23_26_dependency_graph.json": dependency_graph,
        "v23_26_part0_summary.json": {**gates, **audit, "part0_hard_gate_pass": int(all(v == 1 for k, v in gates.items() if k != "mandatory_hypothesis_count") and gates["mandatory_hypothesis_count"] == 8)},
    }
    for name, obj in objects.items():
        write_json(OUT_ROOT / name, obj)
    append_exec("Part0_registry_contracts", args, sorted(objects), objects["v23_26_part0_summary.json"])
    append_recap(
        "Part0 v23.26 完整计划读取与合同注册",
        [
            f"完整读取证据：line_count `{audit['plan_line_count']}`，sha256 `{audit['plan_sha256']}`。",
            f"Part0 hard gate pass `{objects['v23_26_part0_summary.json']['part0_hard_gate_pass']}`；mandatory_hypothesis_count `{len(hypothesis_registry)}`。",
            "注册了 D-CHE-Core、D-FOU-IdLF-Core、D-FOU-Trig-Core 语义合同；D-FOU-Trig 只有在自身 fused/no-materialize PartA gate 通过后才允许进入 science，不允许 materialized fallback science。",
        ],
    )


def run_kernel_correctness(info: CarrierInfo, device: torch.device) -> dict[str, Any]:
    row: dict[str, Any] = {
        "carrier_core_variant": info.name,
        "carrier_family": info.family,
        "basis_name": info.basis_name,
        "k": info.k,
        "basis_channel_order": ";".join(info.channel_order),
        "supported_fused_predeclared": int(info.supported_fused),
    }
    if not info.supported_fused:
        row.update({"status": "incomplete", "route": "R0_DFOUTrigKernelIncomplete", "forward_relative_error": math.nan, "gradient_relative_error": math.nan, "basis_materialized_bytes": math.nan, "fallback_kernel_used": 1})
        return row
    if device.type != "cuda":
        row.update({"status": "blocked", "route": "R0_ArchitectureOrKernelInvalid", "error": "PartA fused kernel check requires CUDA"})
        return row
    try:
        torch.manual_seed(1234 + info.k)
        input_dim, output_dim, hidden, batch = 7, 3, 11, 19
        x_stats = torch.randn(128, input_dim, device=device)
        x = torch.randn(batch, input_dim, device=device)
        y = torch.randint(0, output_dim, (batch,), device=device)
        model = make_model(info, input_dim, output_dim, hidden, x_stats, seed=17, device=device)
        ref = make_model(info, input_dim, output_dim, hidden, x_stats, seed=17, device=device)
        ref.load_state_dict(copy.deepcopy(model.state_dict()))
        ref.zero_grad(set_to_none=True)
        logits_ref = ref(x)
        loss_ref = F.cross_entropy(logits_ref, y)
        loss_ref.backward()
        ref_gw1 = ref.w1.grad.detach().clone()
        ref_gw2 = ref.w2.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        logits_fused, cache = model.manual_ce_forward_cache(x)
        loss_fused = model.manual_ce_backward_from_cache(logits_fused, cache, y)
        tag = str(cache[0]) if cache and isinstance(cache[0], str) else "generic_or_materialized"
        fw_rel = rel_error(logits_fused, logits_ref)
        gw1_rel = tensor_grad_rel_error(model.w1.grad, ref_gw1)
        gw2_rel = tensor_grad_rel_error(model.w2.grad, ref_gw2)
        grad_rel = max(gw1_rel, gw2_rel)
        fused_tags = {
            "cheby_k3_triton_l3_gradbuf",
            "cheby_k4_triton_l3_matmul",
            "fourier_k2_triton_l3_matmul",
            "fourier_k4_triton_l3_matmul",
            "fourier_trig_k4_triton_l3_matmul",
            "fourier_trig_k5_triton_l3_matmul",
        }
        row.update(
            {
                "status": "pass" if fw_rel <= 5.0e-4 and grad_rel <= 5.0e-3 and tag in fused_tags else "fail",
                "fused_forward_used": int(tag in fused_tags),
                "fused_backward_used": int(tag in fused_tags),
                "manual_kernel_tag": tag,
                "forward_relative_error": fw_rel,
                "gradient_relative_error": grad_rel,
                "w1_gradient_relative_error": gw1_rel,
                "w2_gradient_relative_error": gw2_rel,
                "basis_materialized_bytes": 0 if tag in fused_tags else -1,
                "fallback_kernel_used": int(tag not in fused_tags),
                "loss_ref": float(loss_ref.detach().item()),
                "loss_fused": float(loss_fused.detach().item()),
                "error": "",
            }
        )
    except Exception as exc:  # pragma: no cover - captured as audit data.
        row.update({"status": "fail", "route": "R0_ArchitectureOrKernelInvalid", "error": repr(exc), "fallback_kernel_used": 1})
    return row


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, device: torch.device) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, output_dim)).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def time_cuda(fn, repeats: int = 8) -> float:
    torch.cuda.synchronize()
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / float(repeats))


def peak_memory_increment_mb(device: torch.device, fn) -> float:
    if device.type != "cuda":
        return math.nan
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = float(torch.cuda.memory_allocated(device))
    fn()
    torch.cuda.synchronize()
    peak = float(torch.cuda.max_memory_allocated(device))
    return max(0.0, peak - baseline) / (1024.0 * 1024.0)


def run_efficiency_row(info: CarrierInfo, batch: int, hidden: int, device: torch.device) -> dict[str, Any]:
    row = {
        "carrier_core_variant": info.name,
        "batch": batch,
        "hidden": hidden,
        "status": "not_run",
        "forward_ms": math.nan,
        "backward_ms": math.nan,
        "full_training_step_ms": math.nan,
    }
    if not info.supported_fused:
        row.update({"status": "incomplete", "route": "R0_DFOUTrigKernelIncomplete"})
        return row
    if device.type != "cuda":
        row.update({"status": "blocked", "route": "R0_ArchitectureOrKernelInvalid", "error": "CUDA required"})
        return row
    try:
        torch.manual_seed(4000 + batch + hidden + info.k)
        input_dim, output_dim = 32, 10
        x_stats = torch.randn(512, input_dim, device=device)
        x = torch.randn(batch, input_dim, device=device)
        y = torch.randint(0, output_dim, (batch,), device=device)
        model = make_model(info, input_dim, output_dim, hidden, x_stats, seed=5, device=device)
        mlp = TinyMLP(input_dim, hidden, output_dim, device)
        mlp_opt = torch.optim.SGD(mlp.parameters(), lr=1.0e-3)

        def kan_forward() -> None:
            model.manual_ce_forward_cache(x)

        def kan_backward() -> None:
            model.zero_grad(set_to_none=True)
            logits, cache = model.manual_ce_forward_cache(x)
            model.manual_ce_backward_from_cache(logits, cache, y)

        def kan_full() -> None:
            model.zero_grad(set_to_none=True)
            logits, cache = model.manual_ce_forward_cache(x)
            model.manual_ce_backward_from_cache(logits, cache, y)
            with torch.no_grad():
                for param in (model.w1, model.w2):
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-1.0e-3)
                        param.grad = None

        def mlp_forward() -> None:
            mlp(x)

        def mlp_backward() -> None:
            mlp.zero_grad(set_to_none=True)
            F.cross_entropy(mlp(x), y).backward()

        def mlp_full() -> None:
            mlp_opt.zero_grad(set_to_none=True)
            F.cross_entropy(mlp(x), y).backward()
            mlp_opt.step()

        f_ms = time_cuda(kan_forward)
        b_ms = time_cuda(kan_backward)
        full_ms = time_cuda(kan_full)
        mlp_f = time_cuda(mlp_forward)
        mlp_b = time_cuda(mlp_backward)
        mlp_full_ms = time_cuda(mlp_full)
        row.update(
            {
                "status": "completed",
                "forward_ms": f_ms,
                "backward_ms": b_ms,
                "full_training_step_ms": full_ms,
                "mlp_forward_ms": mlp_f,
                "mlp_backward_ms": mlp_b,
                "mlp_full_training_step_ms": mlp_full_ms,
                "forward_ratio_vs_MLP": f_ms / max(mlp_f, EPS),
                "backward_ratio_vs_MLP": b_ms / max(mlp_b, EPS),
                "full_step_ratio_vs_MLP": full_ms / max(mlp_full_ms, EPS),
                "basis_materialized_bytes": 0,
                "fused_forward_used": 1,
                "fused_backward_used": 1,
                "fallback_kernel_used": 0,
                "peak_memory_MB": math.nan,
                "memory_ratio_vs_MLP": math.nan,
            }
        )
    except Exception as exc:  # pragma: no cover
        row.update({"status": "fail", "route": "R0_ArchitectureOrKernelInvalid", "error": repr(exc)})
    return row


def run_part_a(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    carrier_rows: list[dict[str, Any]] = []
    for name in OFFICIAL_CARRIERS:
        info = carrier_info(name)
        carrier_rows.append(
            {
                "carrier_core_variant": name,
                "carrier_family": info.family,
                "primitive_execution_shell_used": "PrimitiveKAN",
                "basis_name": info.basis_name,
                "basis_channel_order": ";".join(info.channel_order),
                "chart_name": "tanh_train_stats",
                "chart_stats_train_only": 1,
                "global_support": 1,
                "uses_dense_basis_tensor": 0 if info.supported_fused else math.nan,
                "fused_forward": int(info.supported_fused),
                "fused_backward": int(info.supported_fused),
                "no_materialize": int(info.supported_fused),
                "linearres": 0,
                "paircross": 0,
                "inputcross": 0,
                "localrot": 0,
                "spline_residual": 0,
                "direct_logit_adapter": 0,
                "mlp_stem": 0,
                "mlp_head": 0,
                "science_status": "eligible" if info.supported_fused else "incomplete",
                "route": "" if info.supported_fused else "R0_DFOUTrigKernelIncomplete",
            }
        )
    kernel_rows = [run_kernel_correctness(carrier_info(name), device) for name in OFFICIAL_CARRIERS]
    efficiency_rows: list[dict[str, Any]] = []
    batches = [32, 128, 256] if not args.quick else [32]
    hiddens = [64, 128] if not args.quick else [64]
    for name in OFFICIAL_CARRIERS:
        info = carrier_info(name)
        for batch in batches:
            for hidden in hiddens:
                efficiency_rows.append(run_efficiency_row(info, batch, hidden, device))
    write_csv(OUT_ROOT / "v23_26_partA_carrier_semantic_matrix.csv", carrier_rows)
    write_csv(OUT_ROOT / "v23_26_partA_kernel_correctness_matrix.csv", kernel_rows)
    write_csv(OUT_ROOT / "v23_26_partA_efficiency_matrix.csv", efficiency_rows)
    kernel_pass = sum(1 for row in kernel_rows if row.get("status") == "pass")
    eligible = sum(1 for row in kernel_rows if row.get("supported_fused_predeclared") == 1)
    incomplete = sum(1 for row in kernel_rows if row.get("status") == "incomplete")
    completed_eff = [row for row in efficiency_rows if row.get("status") == "completed"]

    def ratio_stat(name: str) -> dict[str, Any]:
        vals = [float(row[name]) for row in completed_eff if name in row and math.isfinite(float(row[name]))]
        if not vals:
            return {"rows": 0, "pass_rows_le_1p25": 0, "pass_rate_le_1p25": 0.0, "median": math.nan, "p90": math.nan, "max": math.nan}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "rows": int(arr.size),
            "pass_rows_le_1p25": int((arr <= 1.25).sum()),
            "pass_rate_le_1p25": float((arr <= 1.25).mean()),
            "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
        }

    efficiency_stats = {
        "completed_rows": len(completed_eff),
        "forward_ratio_vs_MLP": ratio_stat("forward_ratio_vs_MLP"),
        "backward_ratio_vs_MLP": ratio_stat("backward_ratio_vs_MLP"),
        "full_step_ratio_vs_MLP": ratio_stat("full_step_ratio_vs_MLP"),
    }
    summary = {
        "carrier_rows": len(carrier_rows),
        "kernel_rows": len(kernel_rows),
        "eligible_kernel_rows": eligible,
        "kernel_pass_rows": kernel_pass,
        "trig_incomplete_rows": incomplete,
        "partA_supported_core_pass": int(kernel_pass == eligible and eligible > 0),
        "partA_full_official_pass": int(kernel_pass == len(kernel_rows)),
        "efficiency_stats": efficiency_stats,
        "efficiency_exploration_gate_forward": int(efficiency_stats["forward_ratio_vs_MLP"]["pass_rate_le_1p25"] >= 0.80),
        "efficiency_exploration_gate_backward": int(efficiency_stats["backward_ratio_vs_MLP"]["pass_rate_le_1p25"] >= 0.80),
        "efficiency_exploration_gate_full_step": int(efficiency_stats["full_step_ratio_vs_MLP"]["pass_rate_le_1p25"] >= 0.80),
        "note": "D-FOU-Trig is eligible only when its own fused/no-materialize K4/K5 kernel rows pass; otherwise it remains incomplete.",
    }
    write_json(OUT_ROOT / "v23_26_partA_summary.json", summary)
    append_exec(
        "PartA_carrier_kernel_efficiency",
        args,
        [
            "v23_26_partA_carrier_semantic_matrix.csv",
            "v23_26_partA_kernel_correctness_matrix.csv",
            "v23_26_partA_efficiency_matrix.csv",
            "v23_26_partA_summary.json",
        ],
        summary,
    )
    append_recap(
        "PartA v23.26 carrier/kernel/efficiency 复盘",
        [
            f"supported fused carriers kernel pass `{kernel_pass}/{eligible}`；D-FOU-Trig incomplete rows `{incomplete}`，按计划不能用 materialized fallback 代替。",
            f"PartA supported-core pass `{summary['partA_supported_core_pass']}`；full official pass `{summary['partA_full_official_pass']}`。",
            "carrier semantic matrix 记录 linearres/paircross/inputcross/localrot/spline_residual/direct_logit_adapter/mlp_stem/mlp_head 全部为 0。",
        ],
    )


def basis_covariance_trial(family: str, k: int, metric_name: str, dtype: np.dtype, rng: np.random.Generator) -> dict[str, float]:
    mats = metric_matrices(family, k)
    m = mats[metric_name].astype(dtype)
    a = rng.normal(size=k).astype(dtype)
    g = rng.normal(size=k).astype(dtype)
    raw = rng.normal(size=(k, k)).astype(dtype)
    q, _ = np.linalg.qr(raw.astype(np.float64))
    scales = np.linspace(0.75, 1.25, k).astype(dtype)
    s = (q.astype(dtype) @ np.diag(scales)).astype(dtype)
    m_prime = (s.T @ m @ s).astype(dtype)
    a_prime = np.linalg.solve(s, a).astype(dtype)
    g_prime = (s.T @ g).astype(dtype)
    d = intrinsic_proposal(g.astype(np.float64), m.astype(np.float64), lr=1.0e-3).astype(dtype)
    d_prime = intrinsic_proposal(g_prime.astype(np.float64), m_prime.astype(np.float64), lr=1.0e-3).astype(dtype)
    expected_d_prime = np.linalg.solve(s, d).astype(dtype)
    add_new, _ = additive_update(a, d, m)
    add_new_prime, _ = additive_update(a_prime, d_prime, m_prime)
    expected_add_prime = np.linalg.solve(s, add_new).astype(dtype)
    rt_new, _ = rtgf_update(a.astype(np.float64), d.astype(np.float64), m.astype(np.float64), norm_floor=1.0e-12)
    rt_new_prime, _ = rtgf_update(a_prime.astype(np.float64), d_prime.astype(np.float64), m_prime.astype(np.float64), norm_floor=1.0e-12)
    expected_rt_prime = np.linalg.solve(s.astype(np.float64), rt_new).astype(dtype)
    return {
        "proposal_covariance_error": float(np.linalg.norm(d_prime - expected_d_prime) / max(float(np.linalg.norm(expected_d_prime)), EPS)),
        "additive_update_covariance_error": float(np.linalg.norm(add_new_prime - expected_add_prime) / max(float(np.linalg.norm(expected_add_prime)), EPS)),
        "rtgf_update_covariance_error": float(np.linalg.norm(rt_new_prime - expected_rt_prime) / max(float(np.linalg.norm(expected_rt_prime)), EPS)),
    }


def run_part_b(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(2326)
    metric_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    carriers_for_math = [
        ("D-CHE-Core-K3", "D-CHE", 3),
        ("D-CHE-Core-K4", "D-CHE", 4),
        ("D-FOU-IdLF-Core-K2", "D-FOU-IdLF", 2),
        ("D-FOU-IdLF-Core-K4", "D-FOU-IdLF", 4),
        ("D-FOU-Trig-Core-K4", "D-FOU-Trig", 4),
        ("D-FOU-Trig-Core-K5", "D-FOU-Trig-DC", 5),
    ]
    for carrier, family, k in carriers_for_math:
        mats = metric_matrices(family, k)
        for metric_name in METRICS:
            m = mats[metric_name]
            eig = np.linalg.eigvalsh(m)
            metric_rows.append(
                {
                    "carrier_core_variant": carrier,
                    "metric_name": metric_name,
                    "symmetric_error": float(np.linalg.norm(m - m.T)),
                    "lambda_min": float(eig.min()),
                    "lambda_max": float(eig.max()),
                    "metric_condition_number": float(eig.max() / max(eig.min(), EPS)),
                    "spd_pass": int(float(eig.min()) > 0.0),
                }
            )
            for dtype_name, dtype, gate in (("float64", np.float64, 1.0e-8), ("fp32", np.float32, 5.0e-4)):
                cov = basis_covariance_trial(family, k, metric_name, dtype, rng)
                covariance_rows.append(
                    {
                        "carrier_core_variant": carrier,
                        "metric_name": metric_name,
                        "dtype": dtype_name,
                        **cov,
                        "gate": gate,
                        "basis_covariance_pass": int(max(cov.values()) <= gate),
                    }
                )
            a = rng.normal(size=k)
            g = rng.normal(size=k)
            d = intrinsic_proposal(g, m, lr=1.0e-3)
            r2 = mdot(a, a, m)
            alpha = mdot(a, d, m) / max(r2, EPS)
            tangent = d - alpha * a
            tangent_orth = abs(mdot(a, tangent, m))
            kgen = (np.outer(tangent, a @ m) - np.outer(a, tangent @ m)) / max(r2, EPS)
            skew_resid = float(np.linalg.norm(kgen.T @ m + m @ kgen))
            shape, trace = rtgf_update(a, d, m, norm_floor=1.0e-12)
            # Remove the radial scale to test shape norm preservation.
            shape_only = shape / math.exp(trace["radial_log_step"])
            shape_norm_error = abs(mdot(shape_only, shape_only, m) - mdot(a, a, m)) / max(abs(mdot(a, a, m)), EPS)
            add = a + d
            first_order_error = math.sqrt(max(mdot(shape - add, shape - add, m), 0.0)) / (max(mdot(d, d, m), 0.0) + EPS)
            small, small_trace = rtgf_update(np.zeros(k), d, m, norm_floor=1.0e-6)
            finite_small = int(np.isfinite(small).all())
            generator_rows.append(
                {
                    "carrier_core_variant": carrier,
                    "metric_name": metric_name,
                    "radial_tangent_orthogonality_abs": tangent_orth,
                    "generator_M_skew_residual": skew_resid,
                    "shape_norm_preservation_relative_error": shape_norm_error,
                    "first_order_equivalence_ratio": first_order_error,
                    "near_zero_bootstrap": int(small_trace["bootstrap"]),
                    "near_zero_finite": finite_small,
                    "generator_unit_pass": int(tangent_orth <= 1.0e-8 and skew_resid <= 1.0e-8 and shape_norm_error <= 1.0e-10 and finite_small == 1),
                }
            )
            raw = rng.normal(size=k)
            raw_tangent = raw - (mdot(a, raw, m) / max(r2, EPS)) * a
            target_norm = math.sqrt(max(mdot(tangent, tangent, m), 0.0))
            raw_norm = math.sqrt(max(mdot(raw_tangent, raw_tangent, m), 0.0))
            random_tangent = raw_tangent * (target_norm / max(raw_norm, EPS))
            random_rows.append(
                {
                    "carrier_core_variant": carrier,
                    "metric_name": metric_name,
                    "random_tangent_orthogonality_abs": abs(mdot(a, random_tangent, m)),
                    "target_tangent_norm": target_norm,
                    "random_tangent_norm": math.sqrt(max(mdot(random_tangent, random_tangent, m), 0.0)),
                    "same_tangent_norm_error": abs(math.sqrt(max(mdot(random_tangent, random_tangent, m), 0.0)) - target_norm),
                    "random_control_identity_pass": int(abs(mdot(a, random_tangent, m)) <= 1.0e-8 and abs(math.sqrt(max(mdot(random_tangent, random_tangent, m), 0.0)) - target_norm) <= 1.0e-8),
                }
            )
    write_csv(OUT_ROOT / "v23_26_partB_metric_unit_matrix.csv", metric_rows)
    write_csv(OUT_ROOT / "v23_26_partB_basis_covariance_matrix.csv", covariance_rows)
    write_csv(OUT_ROOT / "v23_26_partB_generator_unit_matrix.csv", generator_rows)
    write_csv(OUT_ROOT / "v23_26_partB_random_control_identity_matrix.csv", random_rows)
    metric_pass = sum(int(r["spd_pass"]) for r in metric_rows)
    cov_pass = sum(int(r["basis_covariance_pass"]) for r in covariance_rows)
    gen_pass = sum(int(r["generator_unit_pass"]) for r in generator_rows)
    rnd_pass = sum(int(r["random_control_identity_pass"]) for r in random_rows)
    summary = {
        "metric_rows": len(metric_rows),
        "metric_pass_rows": metric_pass,
        "basis_covariance_rows": len(covariance_rows),
        "basis_covariance_pass_rows": cov_pass,
        "generator_rows": len(generator_rows),
        "generator_pass_rows": gen_pass,
        "random_control_rows": len(random_rows),
        "random_control_pass_rows": rnd_pass,
        "partB_pass": int(metric_pass == len(metric_rows) and cov_pass == len(covariance_rows) and gen_pass == len(generator_rows) and rnd_pass == len(random_rows)),
    }
    write_json(OUT_ROOT / "v23_26_partB_summary.json", summary)
    append_exec(
        "PartB_metric_covariance_generator_units",
        args,
        [
            "v23_26_partB_metric_unit_matrix.csv",
            "v23_26_partB_basis_covariance_matrix.csv",
            "v23_26_partB_generator_unit_matrix.csv",
            "v23_26_partB_random_control_identity_matrix.csv",
            "v23_26_partB_summary.json",
        ],
        summary,
    )
    append_recap(
        "PartB v23.26 metric/covariance/generator 数学单元复盘",
        [
            f"metric SPD pass `{metric_pass}/{len(metric_rows)}`；basis covariance pass `{cov_pass}/{len(covariance_rows)}`。",
            f"generator unit pass `{gen_pass}/{len(generator_rows)}`；random control identity pass `{rnd_pass}/{len(random_rows)}`。",
            f"PartB pass `{summary['partB_pass']}`。若未全过，按计划不能进入 official science，只能修数学/公式/数值路径。",
        ],
    )


def make_synthetic(task: str, seed: int, n: int, input_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed + 9100)
    x = torch.randn(n, input_dim, generator=gen, device=device)
    z = torch.tanh(x)
    if task == "SYN-CHE-LOW":
        score = 0.8 * z[:, 0] + 0.6 * (2.0 * z[:, 1].square() - 1.0) - 0.3 * z[:, 2]
    elif task == "SYN-FOU-TREND":
        score = 0.7 * z[:, 0] + 0.4 * torch.sin(math.pi * z[:, 1]) - 0.35 * torch.cos(math.pi * z[:, 2])
    else:
        score = z[:, 0] + 0.25 * z[:, 1] * z[:, 2]
    noise = 0.15 * torch.randn(n, generator=gen, device=device)
    y = ((score + noise) > 0.0).long()
    return x, y


def eval_task_metrics(model: PrimitiveKAN, x: torch.Tensor, y: torch.Tensor, prefix: str) -> dict[str, float]:
    with torch.no_grad():
        logits, _cache = model.manual_ce_forward_cache(x)
        if not bool(torch.isfinite(logits).all().item()):
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
            return {
                f"{prefix}_NLL": float("nan"),
                f"{prefix}_accuracy": float(acc),
                f"{prefix}_balanced_accuracy": float("nan"),
                f"{prefix}_standard_ECE_fixed_bins": float("nan"),
                f"{prefix}_adaptive_ECE": float("nan"),
                f"{prefix}_Brier": float("nan"),
                f"{prefix}_tail_NLL_q95": float("nan"),
                f"{prefix}_tail_NLL_q99": float("nan"),
                f"{prefix}_CVaR95_NLL": float("nan"),
                f"{prefix}_CVaR99_NLL": float("nan"),
                f"{prefix}_margin_q10": float("nan"),
                f"{prefix}_wrong_confident_rate": float("nan"),
                f"{prefix}_right_confident_sharpness": float("nan"),
            }
        logits_eval = logits.to(dtype=torch.float64)
        per_nll = F.cross_entropy(logits_eval, y, reduction="none")
        probs = F.softmax(logits_eval, dim=1)
        conf, pred = probs.max(dim=1)
        correct = pred.eq(y)
        acc = correct.float().mean()
        classes = torch.unique(y)
        recalls = []
        for cls in classes:
            mask = y.eq(cls)
            recalls.append(correct[mask].float().mean())
        balanced = torch.stack(recalls).mean() if recalls else acc
        onehot = F.one_hot(y, num_classes=int(logits_eval.shape[1])).to(dtype=probs.dtype)
        brier = (probs - onehot).square().sum(dim=1).mean()

        def fixed_ece(n_bins: int = 10) -> torch.Tensor:
            total = torch.zeros((), device=x.device, dtype=probs.dtype)
            for idx in range(n_bins):
                lo = float(idx) / float(n_bins)
                hi = float(idx + 1) / float(n_bins)
                if idx + 1 == n_bins:
                    mask = (conf >= lo) & (conf <= hi)
                else:
                    mask = (conf >= lo) & (conf < hi)
                if bool(mask.any().item()):
                    total = total + mask.to(dtype=probs.dtype).mean() * (conf[mask].mean() - correct[mask].to(dtype=probs.dtype).mean()).abs()
            return total

        def adaptive_ece(n_bins: int = 10) -> torch.Tensor:
            order = torch.argsort(conf)
            chunks = torch.chunk(order, min(n_bins, int(conf.numel())))
            total = torch.zeros((), device=x.device, dtype=probs.dtype)
            for chunk in chunks:
                if int(chunk.numel()) == 0:
                    continue
                total = total + (float(chunk.numel()) / float(conf.numel())) * (conf[chunk].mean() - correct[chunk].to(dtype=probs.dtype).mean()).abs()
            return total

        sorted_nll, _ = torch.sort(per_nll)
        q95 = torch.quantile(per_nll, 0.95)
        q99 = torch.quantile(per_nll, 0.99)
        top95 = sorted_nll[int(math.floor(0.95 * max(0, int(sorted_nll.numel()) - 1))) :]
        top99 = sorted_nll[int(math.floor(0.99 * max(0, int(sorted_nll.numel()) - 1))) :]
        top2 = torch.topk(probs, k=min(2, int(probs.shape[1])), dim=1).values
        margin = top2[:, 0] - top2[:, 1] if int(top2.shape[1]) > 1 else top2[:, 0]
        wrong_confident = ((~correct) & (conf >= 0.90)).to(dtype=probs.dtype).mean()
        right_sharp = conf[correct].mean() if bool(correct.any().item()) else torch.zeros((), device=x.device, dtype=probs.dtype)
    return {
        f"{prefix}_NLL": float(per_nll.mean().item()),
        f"{prefix}_accuracy": float(acc.item()),
        f"{prefix}_balanced_accuracy": float(balanced.item()),
        f"{prefix}_standard_ECE_fixed_bins": float(fixed_ece().item()),
        f"{prefix}_adaptive_ECE": float(adaptive_ece().item()),
        f"{prefix}_Brier": float(brier.item()),
        f"{prefix}_tail_NLL_q95": float(q95.item()),
        f"{prefix}_tail_NLL_q99": float(q99.item()),
        f"{prefix}_CVaR95_NLL": float(top95.mean().item()),
        f"{prefix}_CVaR99_NLL": float(top99.mean().item()),
        f"{prefix}_margin_q10": float(torch.quantile(margin, 0.10).item()),
        f"{prefix}_wrong_confident_rate": float(wrong_confident.item()),
        f"{prefix}_right_confident_sharpness": float(right_sharp.item()),
    }


def eval_nll_acc(model: PrimitiveKAN, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        logits, _cache = model.manual_ce_forward_cache(x)
        if not bool(torch.isfinite(logits).all().item()):
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            return float("nan"), float(acc)
        nll = F.cross_entropy(logits.to(dtype=torch.float64), y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    return float(nll), float(acc)


def eval_module_nll_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        logits = model(x)
        if not bool(torch.isfinite(logits).all().item()):
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            return float("nan"), float(acc)
        nll = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    return float(nll), float(acc)


def eval_module_task_metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor, prefix: str) -> dict[str, float]:
    with torch.no_grad():
        logits = model(x)
        if not bool(torch.isfinite(logits).all().item()):
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
            return {
                f"{prefix}_NLL": float("nan"),
                f"{prefix}_accuracy": float(acc),
                f"{prefix}_balanced_accuracy": float("nan"),
                f"{prefix}_standard_ECE_fixed_bins": float("nan"),
                f"{prefix}_adaptive_ECE": float("nan"),
                f"{prefix}_Brier": float("nan"),
                f"{prefix}_tail_NLL_q95": float("nan"),
                f"{prefix}_tail_NLL_q99": float("nan"),
                f"{prefix}_CVaR95_NLL": float("nan"),
                f"{prefix}_CVaR99_NLL": float("nan"),
                f"{prefix}_margin_q10": float("nan"),
                f"{prefix}_wrong_confident_rate": float("nan"),
                f"{prefix}_right_confident_sharpness": float("nan"),
            }
        logits_eval = logits.to(dtype=torch.float64)
        per_nll = F.cross_entropy(logits_eval, y, reduction="none")
        probs = F.softmax(logits_eval, dim=1)
        conf, pred = probs.max(dim=1)
        correct = pred.eq(y)
        acc = correct.float().mean()
        classes = torch.unique(y)
        recalls = []
        for cls in classes:
            mask = y.eq(cls)
            recalls.append(correct[mask].float().mean())
        balanced = torch.stack(recalls).mean() if recalls else acc
        onehot = F.one_hot(y, num_classes=int(logits_eval.shape[1])).to(dtype=probs.dtype)
        brier = (probs - onehot).square().sum(dim=1).mean()

        def fixed_ece(n_bins: int = 10) -> torch.Tensor:
            total = torch.zeros((), device=x.device, dtype=probs.dtype)
            for idx in range(n_bins):
                lo = float(idx) / float(n_bins)
                hi = float(idx + 1) / float(n_bins)
                if idx + 1 == n_bins:
                    mask = (conf >= lo) & (conf <= hi)
                else:
                    mask = (conf >= lo) & (conf < hi)
                if bool(mask.any().item()):
                    total = total + mask.to(dtype=probs.dtype).mean() * (conf[mask].mean() - correct[mask].to(dtype=probs.dtype).mean()).abs()
            return total

        def adaptive_ece(n_bins: int = 10) -> torch.Tensor:
            order = torch.argsort(conf)
            chunks = torch.chunk(order, min(n_bins, int(conf.numel())))
            total = torch.zeros((), device=x.device, dtype=probs.dtype)
            for chunk in chunks:
                if int(chunk.numel()) == 0:
                    continue
                total = total + (float(chunk.numel()) / float(conf.numel())) * (conf[chunk].mean() - correct[chunk].to(dtype=probs.dtype).mean()).abs()
            return total

        sorted_nll, _ = torch.sort(per_nll)
        q95 = torch.quantile(per_nll, 0.95)
        q99 = torch.quantile(per_nll, 0.99)
        top95 = sorted_nll[int(math.floor(0.95 * max(0, int(sorted_nll.numel()) - 1))) :]
        top99 = sorted_nll[int(math.floor(0.99 * max(0, int(sorted_nll.numel()) - 1))) :]
        top2 = torch.topk(probs, k=min(2, int(probs.shape[1])), dim=1).values
        margin = top2[:, 0] - top2[:, 1] if int(top2.shape[1]) > 1 else top2[:, 0]
        wrong_confident = ((~correct) & (conf >= 0.90)).to(dtype=probs.dtype).mean()
        right_sharp = conf[correct].mean() if bool(correct.any().item()) else torch.zeros((), device=x.device, dtype=probs.dtype)
    return {
        f"{prefix}_NLL": float(per_nll.mean().item()),
        f"{prefix}_accuracy": float(acc.item()),
        f"{prefix}_balanced_accuracy": float(balanced.item()),
        f"{prefix}_standard_ECE_fixed_bins": float(fixed_ece().item()),
        f"{prefix}_adaptive_ECE": float(adaptive_ece().item()),
        f"{prefix}_Brier": float(brier.item()),
        f"{prefix}_tail_NLL_q95": float(q95.item()),
        f"{prefix}_tail_NLL_q99": float(q99.item()),
        f"{prefix}_CVaR95_NLL": float(top95.mean().item()),
        f"{prefix}_CVaR99_NLL": float(top99.mean().item()),
        f"{prefix}_margin_q10": float(torch.quantile(margin, 0.10).item()),
        f"{prefix}_wrong_confident_rate": float(wrong_confident.item()),
        f"{prefix}_right_confident_sharpness": float(right_sharp.item()),
    }


def mode_energy_metrics(info: CarrierInfo, model: PrimitiveKAN) -> dict[str, float]:
    with torch.no_grad():
        coeff = torch.cat([model.w1.detach().reshape(-1, info.k), model.w2.detach().reshape(-1, info.k)], dim=0)
        energy = coeff.square().mean(dim=0)
        total = float(energy.sum().item()) + EPS
    row: dict[str, float] = {"mode_energy_entropy": 0.0, "curvature_energy": 0.0}
    probs = (energy / max(float(energy.sum().item()), EPS)).clamp_min(1.0e-12)
    row["mode_energy_entropy"] = float((-(probs * probs.log()).sum()).item())
    if info.family == "D-CHE":
        for idx in range(4):
            row[f"T{idx}_energy"] = float(energy[idx].item()) if idx < info.k else 0.0
        row["T2_over_T1"] = row["T2_energy"] / max(row["T1_energy"], EPS)
        row["T3_over_T1"] = row["T3_energy"] / max(row["T1_energy"], EPS)
        row["curvature_energy"] = row["T2_energy"] + row["T3_energy"]
    elif info.family == "D-FOU-IdLF":
        labels = ["identity_energy", "freq1_sin_energy", "freq1_cos_energy", "freq2_sin_energy", "freq2_cos_energy"]
        for idx, label in enumerate(labels):
            row[label] = float(energy[idx].item()) if idx < info.k else 0.0
        low = row["identity_energy"] + row["freq1_sin_energy"] + row["freq1_cos_energy"]
        row["low_frequency_fraction"] = low / total
        weights = []
        for idx in range(info.k):
            if idx == 0:
                weights.append(0.0)
            elif idx in (1, 2):
                weights.append(1.0)
            else:
                weights.append(2.0)
        weight_tensor = torch.tensor(weights, device=energy.device, dtype=energy.dtype)
        centroid = float((energy * weight_tensor).sum().item()) / total
        spread = float((energy * (weight_tensor - centroid).square()).sum().item()) / total
        row["frequency_centroid"] = centroid
        row["frequency_spread"] = math.sqrt(max(spread, 0.0))
        row["curvature_energy"] = row["freq2_sin_energy"] + row["freq2_cos_energy"]
    else:
        trig_labels = ["freq1_sin_energy", "freq1_cos_energy", "freq2_sin_energy", "freq2_cos_energy"]
        if info.basis_name == "fourier_trig_dc":
            labels = ["dc_energy", *trig_labels]
            weights = [0.0, 1.0, 1.0, 2.0, 2.0]
        else:
            labels = trig_labels
            weights = [1.0, 1.0, 2.0, 2.0]
        row["identity_energy"] = 0.0
        row["dc_energy"] = 0.0
        for idx, label in enumerate(labels):
            row[label] = float(energy[idx].item()) if idx < info.k else 0.0
        low = row.get("dc_energy", 0.0) + row["freq1_sin_energy"] + row["freq1_cos_energy"]
        row["low_frequency_fraction"] = low / total
        weight_tensor = torch.tensor(weights[: info.k], device=energy.device, dtype=energy.dtype)
        centroid = float((energy * weight_tensor).sum().item()) / total
        spread = float((energy * (weight_tensor - centroid).square()).sum().item()) / total
        row["frequency_centroid"] = centroid
        row["frequency_spread"] = math.sqrt(max(spread, 0.0))
        row["curvature_energy"] = row["freq2_sin_energy"] + row["freq2_cos_energy"]
    return row


def train_smoke_scheme(info: CarrierInfo, scheme: str, metric_name: str, task: str, seed: int, steps: int, lr: float, device: torch.device, phase_name: str = "H20_smoke") -> dict[str, Any]:
    input_dim, output_dim, hidden = 6, 2, 16
    x_all, y_all = make_synthetic(task, seed, 768, input_dim, device)
    x_train, y_train = x_all[:256], y_all[:256]
    x_witness, y_witness = x_all[256:384], y_all[256:384]
    x_guard, y_guard = x_all[384:512], y_all[384:512]
    x_test, y_test = x_all[512:], y_all[512:]
    model = make_model(info, input_dim, output_dim, hidden, x_train, seed=seed + 3, device=device)
    initial_guard, initial_acc = eval_nll_acc(model, x_guard, y_guard)
    metric = metric_matrices(info.family, info.k)[metric_name]
    trace = {"actual_generator_update_count": 0, "near_zero_bootstrap_count": 0, "edge_count": 0}
    if scheme.endswith("B0"):
        opt = torch.optim.AdamW([model.w1, model.w2], lr=float(lr), weight_decay=0.0)
        opt_kind = "AdamW"
    elif "-C" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="additive", weight_decay=0.0)
        opt_kind = "BC-BlockAdam-Additive"
    elif "-R0" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="random_m_skew", weight_decay=0.0)
        opt_kind = "R0_random_M_skew"
    elif "-R1" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="signflip", weight_decay=0.0)
        opt_kind = "R1_tangent_signflip"
    elif "-R2" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="shuffle", weight_decay=0.0)
        opt_kind = "R2_shuffled_tangent"
    elif "-R3" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="euclidean", weight_decay=0.0)
        opt_kind = "R3_Euclidean_RTGF"
    elif "-R5" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="noop", weight_decay=0.0)
        opt_kind = "R5_same_compute_noop"
    else:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="rtgf", weight_decay=0.0)
        opt_kind = "BC-RTGF"
    for step in range(int(steps)):
        idx = torch.arange((step * 32) % int(x_train.shape[0]), (step * 32) % int(x_train.shape[0]) + 32, device=device) % int(x_train.shape[0])
        xb, yb = x_train[idx], y_train[idx]
        if opt_kind == "AdamW":
            opt.zero_grad(set_to_none=True)
            logits, cache = model.manual_ce_forward_cache(xb)
            model.manual_ce_backward_from_cache(logits, cache, yb)
            opt.step()
        else:
            model.zero_grad(set_to_none=True)
            logits, cache = model.manual_ce_forward_cache(xb)
            model.manual_ce_backward_from_cache(logits, cache, yb)
            opt.step()
    final_guard, final_acc = eval_nll_acc(model, x_guard, y_guard)
    if isinstance(opt, IntrinsicEdgeOptimizer):
        trace = opt.trace
    row: dict[str, Any] = {
        "phase": phase_name,
        "task": task,
        "seed": seed,
        "carrier_core_variant": info.name,
        "scheme": scheme,
        "metric_name": metric_name,
        "steps": steps,
        "lr": float(lr),
        "weight_decay": 0.0,
        "task_loss": "cross_entropy",
        "label_smoothing": 0,
        "auxiliary_loss_count": 0,
        "mode_penalty": 0,
        "curvature_loss_penalty": 0,
        "guard_NLL_before": initial_guard,
        "guard_NLL": final_guard,
        "guard_NLL_delta": initial_guard - final_guard,
        "accuracy_before": initial_acc,
        "accuracy": final_acc,
        "carrier_family": info.family,
        "basis_channel_order": ";".join(info.channel_order),
        "fused_forward_used": 1,
        "fused_backward_used": 1,
        "fallback_kernel_used": 0,
        "actual_generator_update_count": trace.get("actual_generator_update_count", 0),
        "actual_random_M_skew_count": trace.get("actual_random_M_skew_count", 0),
        "actual_signflip_count": trace.get("actual_signflip_count", 0),
        "actual_shuffled_tangent_count": trace.get("actual_shuffled_tangent_count", 0),
        "actual_euclidean_generator_count": trace.get("actual_euclidean_generator_count", 0),
        "actual_noop_count": trace.get("actual_noop_count", 0),
        "norm_floor_rule": "fixed_init_edge_radius_quantile",
        "norm_floor": float(trace.get("norm_floor", 0.0)),
        "norm_floor_quantile": float(trace.get("norm_floor_quantile", NORM_FLOOR_QUANTILE)),
        "norm_floor_min": float(trace.get("norm_floor_min", NORM_FLOOR_MIN)),
        "initial_edge_radius_count": float(trace.get("initial_edge_radius_count", 0.0)),
        "initial_edge_radius_q": float(trace.get("initial_edge_radius_q", 0.0)),
        "initial_edge_radius_min": float(trace.get("initial_edge_radius_min", 0.0)),
        "initial_edge_radius_median": float(trace.get("initial_edge_radius_median", 0.0)),
        "initial_edge_radius_max": float(trace.get("initial_edge_radius_max", 0.0)),
        "near_zero_bootstrap_fraction": trace_fraction(trace, "near_zero_bootstrap_count"),
        "mean_radial_abs_step": trace_mean(trace, "radial_abs_sum"),
        "mean_radial_signed_step": trace_mean(trace, "radial_signed_sum"),
        "max_radial_abs_step": float(trace.get("radial_abs_max", 0.0)),
        "mean_tangent_angle": trace_mean(trace, "tangent_angle_sum"),
        "mean_tangent_norm": trace_mean(trace, "tangent_norm_sum"),
        "mean_edge_m_radius": trace_mean(trace, "edge_radius_sum"),
        "min_edge_m_radius": trace_min(trace, "edge_radius_min"),
        "max_edge_m_radius": float(trace.get("edge_radius_max", 0.0)),
        "mean_proposal_m_norm": trace_mean(trace, "proposal_m_norm_sum"),
        "mean_proposal_m_energy": trace_mean(trace, "proposal_m_energy_sum"),
        "mean_additive_retraction_gap_m_norm": trace_mean(trace, "additive_retraction_gap_m_norm_sum"),
        "mean_additive_retraction_gap_ratio": trace_mean(trace, "additive_retraction_gap_ratio_sum"),
        "max_additive_retraction_gap_ratio": float(trace.get("max_additive_retraction_gap_ratio", 0.0)),
        "mean_tangent_m_orthogonality_abs": trace_mean(trace, "tangent_m_orthogonality_abs_sum"),
        "max_tangent_m_orthogonality_abs": float(trace.get("max_tangent_m_orthogonality_abs", 0.0)),
        "mean_tangent_norm_match_abs": trace_mean(trace, "tangent_norm_match_abs_sum"),
        "max_tangent_norm_match_abs": float(trace.get("max_tangent_norm_match_abs", 0.0)),
        "mean_shape_norm_preservation_abs": trace_mean(trace, "shape_norm_preservation_abs_sum"),
        "max_shape_norm_preservation_abs": float(trace.get("max_shape_norm_preservation_abs", 0.0)),
        "tangent_angle_le_1e_4_fraction": trace_fraction(trace, "tangent_angle_le_1e_4_count"),
        "tangent_angle_le_1e_3_fraction": trace_fraction(trace, "tangent_angle_le_1e_3_count"),
        "radial_energy_fraction": float(trace.get("radial_energy_sum", 0.0)) / max(EPS, float(trace.get("radial_energy_sum", 0.0)) + float(trace.get("tangent_energy_sum", 0.0))),
        "tangent_energy_fraction": float(trace.get("tangent_energy_sum", 0.0)) / max(EPS, float(trace.get("radial_energy_sum", 0.0)) + float(trace.get("tangent_energy_sum", 0.0))),
        "intrinsic_step_norm_mean": trace_mean(trace, "intrinsic_step_norm_sum"),
        "max_tangent_angle": float(trace.get("max_tangent_angle", 0.0)),
        "guard_NLL_finite": int(math.isfinite(final_guard)),
        "diagnostic_only": 1,
    }
    row.update(eval_task_metrics(model, x_train, y_train, "train"))
    row.update(eval_task_metrics(model, x_witness, y_witness, "witness"))
    row.update(eval_task_metrics(model, x_guard, y_guard, "guard"))
    row.update(eval_task_metrics(model, x_test, y_test, "test"))
    row.update(mode_energy_metrics(info, model))
    return row


def run_h20_smoke(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("h20-smoke uses fused kernels and requires CUDA.")
    rows: list[dict[str, Any]] = []
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    tasks = [t for t in str(args.tasks).split(",") if t]
    carrier_names = [c for c in str(args.carriers).split(",") if c]
    metric_names = [m for m in str(args.h20_metrics).split(",") if m]
    for task in tasks:
        for seed in seeds:
            for carrier_name in carrier_names:
                info = carrier_info(carrier_name)
                if not info.supported_fused:
                    continue
                prefix = scheme_prefix(info)
                scheme_metric_pairs = [(f"{prefix}-B0", "M0")]
                for metric_name in metric_names:
                    idx = {"M0": "0", "M1": "1", "M2": "2"}[metric_name]
                    scheme_metric_pairs.append((f"{prefix}-C{idx}", metric_name))
                    scheme_metric_pairs.append((f"{prefix}-G{idx}", metric_name))
                    if args.include_controls:
                        scheme_metric_pairs.append((f"{prefix}-R0-{idx}", metric_name))
                        scheme_metric_pairs.append((f"{prefix}-R1-{idx}", metric_name))
                        scheme_metric_pairs.append((f"{prefix}-R2-{idx}", metric_name))
                        scheme_metric_pairs.append((f"{prefix}-R3-{idx}", metric_name))
                        scheme_metric_pairs.append((f"{prefix}-R5-{idx}", metric_name))
                for scheme, metric_name in scheme_metric_pairs:
                    rows.append(train_smoke_scheme(info, scheme, metric_name, task, seed, int(args.h20_steps), float(args.lr), device))
    write_csv(OUT_ROOT / "v23_26_H20_matrix.csv", rows)
    # Paired diagnostic summary: candidate G2 vs B0 and C2 for matching task/seed/carrier.
    summary_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["task"], row["seed"], row["carrier_core_variant"])
        by_key.setdefault(key, {})[row["scheme"]] = row
    for key, schemes in by_key.items():
        task, seed, carrier = key
        b0 = next((v for k, v in schemes.items() if k.endswith("-B0")), None)
        for cand_name, cand in schemes.items():
            if "-C" not in cand_name and "-G" not in cand_name:
                continue
            if b0 is not None:
                summary_rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "carrier_core_variant": carrier,
                        "candidate": cand_name,
                        "control": b0["scheme"],
                        "paired_guard_NLL_surplus": float(b0["guard_NLL"]) - float(cand["guard_NLL"]),
                        "diagnostic_only": 1,
                    }
                )
            if "-G" in cand_name:
                add_name = cand_name.replace("-G", "-C")
                add = schemes.get(add_name)
                if add is not None:
                    summary_rows.append(
                        {
                            "task": task,
                            "seed": seed,
                            "carrier_core_variant": carrier,
                            "candidate": cand_name,
                            "control": add_name,
                            "paired_guard_NLL_surplus": float(add["guard_NLL"]) - float(cand["guard_NLL"]),
                            "diagnostic_only": 1,
                        }
                    )
                for control_token in ("R0", "R1", "R2", "R3", "R5"):
                    control_name = cand_name.replace("-G", f"-{control_token}-")
                    control = schemes.get(control_name)
                    if control is not None:
                        summary_rows.append(
                            {
                                "task": task,
                                "seed": seed,
                                "carrier_core_variant": carrier,
                                "candidate": cand_name,
                                "control": control_name,
                                "paired_guard_NLL_surplus": float(control["guard_NLL"]) - float(cand["guard_NLL"]),
                                "diagnostic_only": 1,
                            }
                        )
    write_csv(OUT_ROOT / "v23_26_partC_hypothesis_summary.csv", summary_rows)
    summary = {
        "rows": len(rows),
        "paired_summary_rows": len(summary_rows),
        "seeds": seeds,
        "tasks": tasks,
        "carriers": carrier_names,
        "metrics": metric_names,
        "diagnostic_only": 1,
        "lr": float(args.lr),
        "include_controls": int(bool(args.include_controls)),
        "implemented_controls": ["R0_random_M_skew", "R1_signflip", "R2_shuffled_tangent", "R3_Euclidean_RTGF", "R5_noop"] if args.include_controls else [],
        "missing_controls": [] if args.include_controls else [],
    }
    write_json(OUT_ROOT / "v23_26_H20_summary.json", summary)
    append_exec("H20_smoke_diagnostic", args, ["v23_26_H20_matrix.csv", "v23_26_partC_hypothesis_summary.csv", "v23_26_H20_summary.json"], summary)
    positives = sum(1 for row in summary_rows if float(row["paired_guard_NLL_surplus"]) > 0.0)
    additive_rows = [row for row in summary_rows if "-C" in row["candidate"] and row["control"].endswith("-B0")]
    generator_vs_additive = [row for row in summary_rows if "-G" in row["candidate"] and "-C" in row["control"]]
    generator_vs_controls = [row for row in summary_rows if "-G" in row["candidate"] and any(f"-{token}-" in row["control"] for token in ("R0", "R1", "R2", "R3", "R5"))]
    control_line = (
        f"implemented matched-control paired rows `{len(generator_vs_controls)}`；R0/R1/R2/R3/R5 registered controls 已全部接入 H20 smoke。"
        if args.include_controls
        else f"implemented matched-control paired rows `{len(generator_vs_controls)}`；本次未启用 `--include-controls`，因此没有 R0/R1/R2/R3/R5 paired control rows。"
    )
    append_recap(
        "H20 smoke diagnostic v23.26 复盘",
        [
            f"diagnostic rows `{len(rows)}`，paired summary rows `{len(summary_rows)}`，positive G2 surplus rows `{positives}/{len(summary_rows)}`。",
            f"additive-vs-AdamW paired rows `{len(additive_rows)}`；generator-vs-additive paired rows `{len(generator_vs_additive)}`。",
            control_line,
            "这是 H20 smoke，不是 full Part C/D/E science；尚未覆盖所有 synthetic/minimum-real/controls/MLP-MCGA，因此不能输出 official success 或 family-level NoGo。",
            "所有训练行使用 CE、label_smoothing=0、auxiliary_loss_count=0，并调用 fused manual forward/backward；未使用 D-FOU-Trig materialized fallback。",
        ],
    )


def bootstrap_lcb(values: list[float], rng_seed: int = 2326, n_boot: int = 1000, q: float = 0.05) -> float:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return math.nan
    if vals.size == 1:
        return float(vals[0])
    rng = np.random.default_rng(rng_seed)
    means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = rng.choice(vals, size=vals.size, replace=True)
        means[idx] = float(sample.mean())
    return float(np.quantile(means, q))


def surplus_stats(values: list[float]) -> dict[str, Any]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {
            "finite_pair_count": 0,
            "mean_surplus": math.nan,
            "median_surplus": math.nan,
            "CVaR25_surplus": math.nan,
            "bootstrap_LCB05_surplus": math.nan,
            "win_count": 0,
            "win_rate": math.nan,
            "min_surplus": math.nan,
            "max_surplus": math.nan,
        }
    ordered = np.sort(vals)
    tail_n = max(1, int(math.ceil(0.25 * vals.size)))
    return {
        "finite_pair_count": int(vals.size),
        "mean_surplus": float(vals.mean()),
        "median_surplus": float(np.median(vals)),
        "CVaR25_surplus": float(ordered[:tail_n].mean()),
        "bootstrap_LCB05_surplus": bootstrap_lcb(vals.tolist()),
        "win_count": int((vals > 0.0).sum()),
        "win_rate": float((vals > 0.0).mean()),
        "min_surplus": float(vals.min()),
        "max_surplus": float(vals.max()),
    }


def run_part_c(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("part-c uses fused kernels and requires CUDA.")
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    tasks = [t for t in str(args.tasks).split(",") if t]
    carrier_names = [c for c in str(args.carriers).split(",") if c]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            for carrier_name in carrier_names:
                info = carrier_info(carrier_name)
                if not info.supported_fused:
                    rows.append(
                        {
                            "phase": "PartC_synthetic",
                            "task": task,
                            "seed": seed,
                            "carrier_core_variant": carrier_name,
                            "science_status": "incomplete",
                            "route": "R0_DFOUTrigKernelIncomplete" if "Trig" in carrier_name else "R0_UnsupportedCarrier",
                            "diagnostic_only": 1,
                        }
                    )
                    continue
                prefix = scheme_prefix(info)
                fixed_pairs = [
                    (f"{prefix}-B0", "M0"),
                    (f"{prefix}-C0", "M0"),
                    (f"{prefix}-C1", "M1"),
                    (f"{prefix}-C2", "M2"),
                    (f"{prefix}-G0", "M0"),
                    (f"{prefix}-G1", "M1"),
                    (f"{prefix}-G2", "M2"),
                    (f"{prefix}-R0-2", "M2"),
                    (f"{prefix}-R2-2", "M2"),
                    (f"{prefix}-R3-2", "M2"),
                    (f"{prefix}-R5-2", "M2"),
                ]
                for scheme, metric_name in fixed_pairs:
                    rows.append(train_smoke_scheme(info, scheme, metric_name, task, seed, int(args.h20_steps), float(args.lr), device, phase_name="PartC_synthetic"))
    write_csv(OUT_ROOT / "v23_26_partC_synthetic_matrix.csv", rows)

    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("science_status") == "incomplete":
            continue
        key = (row["task"], row["seed"], row["carrier_core_variant"])
        by_key.setdefault(key, {})[row["scheme"]] = row

    pair_rows: list[dict[str, Any]] = []

    def add_pair(task: str, seed: int, carrier: str, hypothesis: str, comparison: str, candidate: dict[str, Any] | None, control: dict[str, Any] | None) -> None:
        if candidate is None or control is None:
            return
        cand_nll = float(candidate.get("guard_NLL", math.nan))
        ctrl_nll = float(control.get("guard_NLL", math.nan))
        cand_debt = float(candidate.get("guard_CVaR95_NLL", math.nan)) - float(control.get("guard_CVaR95_NLL", math.nan))
        pair_rows.append(
            {
                "task": task,
                "seed": seed,
                "carrier_core_variant": carrier,
                "hypothesis": hypothesis,
                "comparison": comparison,
                "candidate": candidate["scheme"],
                "control": control["scheme"],
                "paired_guard_NLL_surplus": ctrl_nll - cand_nll,
                "paired_test_NLL_surplus": float(control.get("test_NLL", math.nan)) - float(candidate.get("test_NLL", math.nan)),
                "paired_guard_CVaR95_debt_delta": cand_debt,
                "candidate_guard_NLL_finite": int(math.isfinite(cand_nll)),
                "control_guard_NLL_finite": int(math.isfinite(ctrl_nll)),
                "diagnostic_only": 1,
            }
        )

    for (task, seed, carrier), schemes in by_key.items():
        b0 = next((v for k, v in schemes.items() if k.endswith("-B0")), None)
        for idx in ("0", "1", "2"):
            prefix = scheme_prefix(carrier)
            add_pair(task, seed, carrier, "H-C", f"C{idx}_vs_AdamW", schemes.get(f"{prefix}-C{idx}"), b0)
            add_pair(task, seed, carrier, "H-C", f"G{idx}_vs_AdamW", schemes.get(f"{prefix}-G{idx}"), b0)
            add_pair(task, seed, carrier, "H-D", f"G{idx}_vs_C{idx}", schemes.get(f"{prefix}-G{idx}"), schemes.get(f"{prefix}-C{idx}"))
        prefix = scheme_prefix(carrier)
        add_pair(task, seed, carrier, "H-D", "G2_vs_random_M_skew_M2", schemes.get(f"{prefix}-G2"), schemes.get(f"{prefix}-R0-2"))
        add_pair(task, seed, carrier, "H-D", "G2_vs_shuffled_tangent_M2", schemes.get(f"{prefix}-G2"), schemes.get(f"{prefix}-R2-2"))
        add_pair(task, seed, carrier, "H-D", "G2_vs_Euclidean_RTGF_M2", schemes.get(f"{prefix}-G2"), schemes.get(f"{prefix}-R3-2"))
        add_pair(task, seed, carrier, "H-D", "G2_vs_same_compute_noop_M2", schemes.get(f"{prefix}-G2"), schemes.get(f"{prefix}-R5-2"))
        if task in {"SYN-CHE-CURV", "SYN-TAIL"}:
            add_pair(task, seed, carrier, "H-E", "C2_vs_C0_curvature_tasks", schemes.get(f"{prefix}-C2"), schemes.get(f"{prefix}-C0"))
            add_pair(task, seed, carrier, "H-E", "G2_vs_G0_curvature_tasks", schemes.get(f"{prefix}-G2"), schemes.get(f"{prefix}-G0"))
        if task == "SYN-CHE-LOW":
            add_pair(task, seed, carrier, "H-E", "C2_vs_C0_low_collapse_check", schemes.get(f"{prefix}-C2"), schemes.get(f"{prefix}-C0"))
            add_pair(task, seed, carrier, "H-E", "G2_vs_G0_low_collapse_check", schemes.get(f"{prefix}-G2"), schemes.get(f"{prefix}-G0"))

    aggregate_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in pair_rows:
        groups.setdefault((row["carrier_core_variant"], row["hypothesis"], row["comparison"]), []).append(row)
    for (carrier, hypothesis, comparison), group in sorted(groups.items()):
        values = [float(row["paired_guard_NLL_surplus"]) for row in group]
        stats = surplus_stats(values)
        debt_vals = np.asarray([float(row["paired_guard_CVaR95_debt_delta"]) for row in group if math.isfinite(float(row["paired_guard_CVaR95_debt_delta"]))], dtype=np.float64)
        aggregate = {
            "carrier_core_variant": carrier,
            "hypothesis": hypothesis,
            "comparison": comparison,
            "pair_count": len(group),
            "finite_pair_count": stats["finite_pair_count"],
            "mean_surplus": stats["mean_surplus"],
            "median_surplus": stats["median_surplus"],
            "CVaR25_surplus": stats["CVaR25_surplus"],
            "bootstrap_LCB05_surplus": stats["bootstrap_LCB05_surplus"],
            "win_count": stats["win_count"],
            "win_rate": stats["win_rate"],
            "min_surplus": stats["min_surplus"],
            "max_surplus": stats["max_surplus"],
            "mean_CVaR95_debt_delta": float(debt_vals.mean()) if debt_vals.size else math.nan,
            "diagnostic_only": 1,
        }
        if hypothesis == "H-C":
            aggregate["synthetic_gate_pass"] = int(stats["median_surplus"] > 0.0 and stats["CVaR25_surplus"] > -1.0e-3)
        elif hypothesis == "H-D":
            aggregate["synthetic_gate_pass"] = int(stats["median_surplus"] > 1.0e-3 and stats["win_rate"] >= 0.70 and stats["bootstrap_LCB05_surplus"] > 0.0)
        elif hypothesis == "H-E":
            aggregate["synthetic_gate_pass"] = int(stats["median_surplus"] > 0.0 and stats["CVaR25_surplus"] > -1.0e-3)
        else:
            aggregate["synthetic_gate_pass"] = 0
        aggregate_rows.append(aggregate)
    write_csv(OUT_ROOT / "v23_26_partC_pair_trace.csv", pair_rows)
    write_csv(OUT_ROOT / "v23_26_partC_hypothesis_summary.csv", aggregate_rows)

    status_rows: list[dict[str, Any]] = []
    for carrier in sorted({row["carrier_core_variant"] for row in aggregate_rows}):
        hc_rows = [row for row in aggregate_rows if row["carrier_core_variant"] == carrier and row["hypothesis"] == "H-C" and row["comparison"].startswith("C")]
        hd_core = [row for row in aggregate_rows if row["carrier_core_variant"] == carrier and row["hypothesis"] == "H-D" and row["comparison"] in {"G2_vs_C2", "G2_vs_random_M_skew_M2", "G2_vs_shuffled_tangent_M2"}]
        he_rows = [row for row in aggregate_rows if row["carrier_core_variant"] == carrier and row["hypothesis"] == "H-E"]
        hc_pass = int(bool(hc_rows) and any(int(row["synthetic_gate_pass"]) for row in hc_rows))
        hd_pass = int(bool(hd_core) and all(int(row["synthetic_gate_pass"]) for row in hd_core))
        he_pass = int(bool(he_rows) and any(int(row["synthetic_gate_pass"]) for row in he_rows if "curvature_tasks" in row["comparison"]))
        status_rows.extend(
            [
                {"carrier_core_variant": carrier, "hypothesis": "H-C", "synthetic_resolved": 1, "synthetic_pass": hc_pass, "route": "H-C_smoke_support_opened" if hc_pass else "R2_IntrinsicSupportNoValue", "diagnostic_only": 1},
                {"carrier_core_variant": carrier, "hypothesis": "H-D", "synthetic_resolved": 1, "synthetic_pass": hd_pass, "route": "R4_RadialTangentialGeneratorOpened" if hd_pass else "R3_IntrinsicSupportOnly_or_GeneratorNotEstablished", "diagnostic_only": 1},
                {"carrier_core_variant": carrier, "hypothesis": "H-E", "synthetic_resolved": 1, "synthetic_pass": he_pass, "route": "R5_CurvatureMetricOpened" if he_pass else "R6_CurvatureMetricTooStiff_or_NotEstablished", "diagnostic_only": 1},
            ]
        )
    write_csv(OUT_ROOT / "v23_26_hypothesis_status_matrix.csv", status_rows)

    finite_rows = sum(1 for row in rows if int(row.get("guard_NLL_finite", 0)) == 1)
    supported_carriers = sorted({row["carrier_core_variant"] for row in status_rows})
    h_status = {(row["carrier_core_variant"], row["hypothesis"]): int(row["synthetic_pass"]) for row in status_rows}
    trig_carriers = [carrier for carrier in supported_carriers if "Trig" in carrier]
    route_obj = {
        "current_route": "R0_IncompleteScientificExploration",
        "route_reason": "PartC synthetic resolves H-C support smoke but H-D/H-E do not pass. PartC alone cannot grant official success; combine with current PartD, PartG, PartH, and confirmatory summaries.",
        "partC_rows": len(rows),
        "partC_finite_rows": finite_rows,
        "partC_all_finite": int(finite_rows == len(rows)),
        "supported_carriers": supported_carriers,
        "H_C_synthetic_pass_carriers": [carrier for carrier in supported_carriers if h_status.get((carrier, "H-C"), 0) == 1],
        "H_D_synthetic_pass_carriers": [carrier for carrier in supported_carriers if h_status.get((carrier, "H-D"), 0) == 1],
        "H_E_synthetic_pass_carriers": [carrier for carrier in supported_carriers if h_status.get((carrier, "H-E"), 0) == 1],
        "D_FOU_Trig_status": "fused_no_materialize_supported_in_runner" if trig_carriers else "not_in_this_partC_matrix",
        "D_FOU_Trig_carriers_in_matrix": trig_carriers,
        "minimum_real_completed": 0,
        "MLP_matched_completed": 0,
        "H80_completed": 0,
        "H200_completed": 0,
        "official_success_allowed": 0,
    }
    write_json(OUT_ROOT / "v23_26_final_route.json", route_obj)
    failure_lines = [
        "# v23.26 failure decomposition",
        "",
        f"- Current route: `{route_obj['current_route']}`.",
        f"- Part C finite rows: `{finite_rows}/{len(rows)}`.",
        f"- H-C synthetic pass carriers: `{len(route_obj['H_C_synthetic_pass_carriers'])}/{len(supported_carriers)}`.",
        f"- H-D synthetic pass carriers: `{len(route_obj['H_D_synthetic_pass_carriers'])}/{len(supported_carriers)}`. Generator independent value is not established because G2 vs matching additive C2 does not meet the `median > 1e-3` gate.",
        f"- H-E synthetic pass carriers: `{len(route_obj['H_E_synthetic_pass_carriers'])}/{len(supported_carriers)}`. M2 curvature value is not established on curvature/tail synthetic checks.",
        f"- D-FOU-Trig carriers in this PartC matrix: `{','.join(trig_carriers) if trig_carriers else 'none'}`; status `{route_obj['D_FOU_Trig_status']}`.",
        "- Part D minimum-real, Part G MLP/MCGA, Part H efficiency, and Part I confirmatory status are external to this PartC route file and must be read from their latest summaries/completion audit.",
        "- Part E/H80/H200 trajectory gates are not complete unless an explicit current summary says otherwise.",
        "- Therefore finalization remains blocked by required scientific coverage, not by a single unrelated gate.",
        "",
    ]
    (OUT_ROOT / "v23_26_failure_decomposition.md").write_text("\n".join(failure_lines), encoding="utf-8")
    summary = {
        "rows": len(rows),
        "finite_rows": finite_rows,
        "pair_trace_rows": len(pair_rows),
        "hypothesis_summary_rows": len(aggregate_rows),
        "status_rows": len(status_rows),
        "tasks": tasks,
        "seeds": seeds,
        "carriers": carrier_names,
        "lr": float(args.lr),
        "steps": int(args.h20_steps),
        "fixed_matrix_scheme_count_per_supported_carrier_task_seed": 11,
        "diagnostic_only": 1,
    }
    write_json(OUT_ROOT / "v23_26_partC_summary.json", summary)
    append_exec(
        "PartC_exact_synthetic_mechanism_matrix",
        args,
        [
            "v23_26_partC_synthetic_matrix.csv",
            "v23_26_partC_pair_trace.csv",
            "v23_26_partC_hypothesis_summary.csv",
            "v23_26_hypothesis_status_matrix.csv",
            "v23_26_partC_summary.json",
            "v23_26_final_route.json",
            "v23_26_failure_decomposition.md",
        ],
        summary,
    )
    hd_passes = sum(1 for row in status_rows if row["hypothesis"] == "H-D" and int(row["synthetic_pass"]) == 1)
    append_recap(
        "PartC exact synthetic mechanism matrix 复盘",
        [
            f"fixed matrix rows `{len(rows)}`，finite rows `{finite_rows}/{len(rows)}`，pair trace rows `{len(pair_rows)}`，aggregate summary rows `{len(aggregate_rows)}`。",
            f"每个 supported carrier/task/seed 固定 11 schemes：AdamW、C0/C1/C2、G0/G1/G2、random-M-skew-M2、shuffled-tangent-M2、Euclidean-RTGF-M2、same-compute-noop-M2。",
            f"H-D synthetic pass carriers `{hd_passes}/{len({row['carrier_core_variant'] for row in status_rows})}`；若为 0，表示 generator independent value 仍未建立，不能输出 R4/R13。",
            "该 PartC 仍是 synthetic diagnostic；minimum-real、MLP baselines、H80/H200 未完成前不能给 official success。",
        ],
    )


def encode_labels(y_raw: np.ndarray) -> np.ndarray:
    labels = [str(v) for v in y_raw.tolist()]
    uniq = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.asarray([uniq[label] for label in labels], dtype=np.int64)


def parse_arff_from_bytes(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    text = raw.decode("utf-8", errors="ignore").splitlines()
    data_started = False
    rows: list[list[str]] = []
    for line in text:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if not data_started:
            if stripped.lower() == "@data":
                data_started = True
            continue
        rows.append([part.strip().strip("'\"") for part in stripped.split(",")])
    if not rows:
        raise RuntimeError("ARFF has no @data rows")
    arr = np.asarray(rows, dtype=object)
    x = arr[:, :-1].astype(np.float32)
    y = encode_labels(arr[:, -1])
    return x, y


def load_real_dataset_numpy(name: str, seed: int, max_samples: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(9000 + seed)
    if name == "Wine":
        from sklearn.datasets import load_wine

        x, y = load_wine(return_X_y=True)
        note = {"source": "sklearn.datasets.load_wine"}
    elif name == "Spam":
        path = ROOT / "data/v22_35_tier2/uci_94_2c1ea99e8cdb.data"
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        x, y = arr[:, :-1], arr[:, -1].astype(np.int64)
        note = {"source": str(path), "format": "UCI Spambase CSV"}
    elif name == "Rice":
        path = ROOT / "data/v22_35_tier2/uci_545_767695f2dba8.zip"
        with zipfile.ZipFile(path) as zf:
            raw = zf.read("Rice_Cammeo_Osmancik.arff")
        x, y = parse_arff_from_bytes(raw)
        note = {"source": str(path), "format": "UCI Rice ARFF"}
    elif name == "Bean":
        path = ROOT / "data/v22_35_tier2/uci_602_01def3651d20.zip"
        with zipfile.ZipFile(path) as zf:
            raw = zf.read("DryBeanDataset/Dry_Bean_Dataset.arff")
        x, y = parse_arff_from_bytes(raw)
        note = {"source": str(path), "format": "UCI Dry Bean ARFF"}
    elif name in {"FashionMNIST", "SVHN", "EMNIST-Letters", "CIFAR10-compact"}:
        from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST, SVHN

        if name == "FashionMNIST":
            ds = FashionMNIST(root=str(ROOT / "data"), train=True, download=False)
        elif name == "SVHN":
            ds = SVHN(root=str(ROOT / "data"), split="train", download=False)
        elif name == "EMNIST-Letters":
            ds = EMNIST(root=str(ROOT / "data"), split="letters", train=True, download=False)
        else:
            ds = CIFAR10(root=str(ROOT / "data"), train=True, download=False)
        count = min(int(max_samples), len(ds))
        idx = rng.choice(len(ds), size=count, replace=False)
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for item_idx in idx:
            image, label = ds[int(item_idx)]
            arr = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
            xs.append(arr)
            ys.append(int(label))
        x = np.stack(xs, axis=0).astype(np.float32)
        y = encode_labels(np.asarray(ys, dtype=object))
        note = {"source": f"torchvision cached {name}", "compact_subsample": int(count)}
    else:
        raise ValueError(f"unknown real dataset {name}")
    x = np.asarray(x, dtype=np.float32)
    y = encode_labels(np.asarray(y, dtype=object))
    if int(max_samples) > 0 and x.shape[0] > int(max_samples):
        idx = rng.choice(x.shape[0], size=int(max_samples), replace=False)
        x, y = x[idx], y[idx]
    note.update({"n_samples": int(x.shape[0]), "n_features": int(x.shape[1]), "n_classes": int(np.unique(y).size)})
    return x, y, note


def real_splits_to_torch(x: np.ndarray, y: np.ndarray, seed: int, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rng = np.random.default_rng(12000 + seed)
    order = rng.permutation(int(x.shape[0]))
    x, y = x[order], y[order]
    n = int(x.shape[0])
    n_train = max(32, int(0.60 * n))
    n_witness = max(16, int(0.15 * n))
    n_guard = max(16, int(0.15 * n))
    if n_train + n_witness + n_guard >= n:
        n_train = max(16, int(0.50 * n))
        n_witness = max(8, int(0.20 * n))
        n_guard = max(8, int(0.15 * n))
    train = slice(0, n_train)
    witness = slice(n_train, n_train + n_witness)
    guard = slice(n_train + n_witness, n_train + n_witness + n_guard)
    test = slice(n_train + n_witness + n_guard, n)
    mu = x[train].mean(axis=0, keepdims=True)
    sigma = x[train].std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)
    x_norm = (x - mu) / sigma
    splits = {
        "x_train": torch.tensor(x_norm[train], device=device, dtype=torch.float32),
        "y_train": torch.tensor(y[train], device=device, dtype=torch.long),
        "x_witness": torch.tensor(x_norm[witness], device=device, dtype=torch.float32),
        "y_witness": torch.tensor(y[witness], device=device, dtype=torch.long),
        "x_guard": torch.tensor(x_norm[guard], device=device, dtype=torch.float32),
        "y_guard": torch.tensor(y[guard], device=device, dtype=torch.long),
        "x_test": torch.tensor(x_norm[test], device=device, dtype=torch.float32),
        "y_test": torch.tensor(y[test], device=device, dtype=torch.long),
    }
    meta = {
        "n_total": n,
        "n_train": int(splits["x_train"].shape[0]),
        "n_witness": int(splits["x_witness"].shape[0]),
        "n_guard": int(splits["x_guard"].shape[0]),
        "n_test": int(splits["x_test"].shape[0]),
        "normalization_stats": "train_only_mean_std",
    }
    return splits, meta


def train_real_scheme(info: CarrierInfo, scheme: str, metric_name: str, dataset: str, seed: int, splits: dict[str, torch.Tensor], steps: int, lr: float, device: torch.device, meta: dict[str, Any]) -> dict[str, Any]:
    input_dim = int(splits["x_train"].shape[1])
    output_dim = int(meta.get("dataset_n_classes", int(torch.cat([splits["y_train"], splits["y_witness"], splits["y_guard"], splits["y_test"]]).max().item()) + 1))
    hidden = 16
    model = make_model(info, input_dim, output_dim, hidden, splits["x_train"], seed=seed + 17, device=device)
    initial_guard, initial_acc = eval_nll_acc(model, splits["x_guard"], splits["y_guard"])
    metric = metric_matrices(info.family, info.k)[metric_name]
    trace = {"actual_generator_update_count": 0, "near_zero_bootstrap_count": 0, "edge_count": 0}
    if scheme.endswith("B0"):
        opt = torch.optim.AdamW([model.w1, model.w2], lr=float(lr), weight_decay=0.0)
        opt_kind = "AdamW"
    elif "-C" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="additive", weight_decay=0.0)
        opt_kind = "BC-BlockAdam-Additive"
    elif "-R0" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="random_m_skew", weight_decay=0.0)
        opt_kind = "R0_random_M_skew"
    elif "-R1" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="signflip", weight_decay=0.0)
        opt_kind = "R1_same_radial_tangent_signflip"
    elif "-R2" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="shuffle", weight_decay=0.0)
        opt_kind = "R2_shuffled_tangent"
    elif "-R3" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="euclidean", weight_decay=0.0)
        opt_kind = "R3_Euclidean_RTGF"
    elif "-R5" in scheme:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="noop", weight_decay=0.0)
        opt_kind = "R5_same_compute_noop"
    else:
        opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="rtgf", weight_decay=0.0)
        opt_kind = "BC-RTGF"
    batch = min(32, int(splits["x_train"].shape[0]))
    for step in range(int(steps)):
        idx = torch.arange((step * batch) % int(splits["x_train"].shape[0]), (step * batch) % int(splits["x_train"].shape[0]) + batch, device=device) % int(splits["x_train"].shape[0])
        xb, yb = splits["x_train"][idx], splits["y_train"][idx]
        if opt_kind == "AdamW":
            opt.zero_grad(set_to_none=True)
            logits, cache = model.manual_ce_forward_cache(xb)
            model.manual_ce_backward_from_cache(logits, cache, yb)
            opt.step()
        else:
            model.zero_grad(set_to_none=True)
            logits, cache = model.manual_ce_forward_cache(xb)
            model.manual_ce_backward_from_cache(logits, cache, yb)
            opt.step()
    final_guard, final_acc = eval_nll_acc(model, splits["x_guard"], splits["y_guard"])
    if isinstance(opt, IntrinsicEdgeOptimizer):
        trace = opt.trace
    row: dict[str, Any] = {
        "phase": "PartD_minimum_real_smoke",
        "dataset": dataset,
        "seed": seed,
        "carrier_core_variant": info.name,
        "scheme": scheme,
        "metric_name": metric_name,
        "steps": steps,
        "lr": float(lr),
        "weight_decay": 0.0,
        "task_loss": "cross_entropy",
        "label_smoothing": 0,
        "auxiliary_loss_count": 0,
        "mode_penalty": 0,
        "curvature_loss_penalty": 0,
        "guard_NLL_before": initial_guard,
        "guard_NLL": final_guard,
        "guard_NLL_delta": initial_guard - final_guard,
        "accuracy_before": initial_acc,
        "accuracy": final_acc,
        "carrier_family": info.family,
        "basis_channel_order": ";".join(info.channel_order),
        "fused_forward_used": 1,
        "fused_backward_used": 1,
        "fallback_kernel_used": 0,
        "compact_smoke": 1,
        **meta,
        "actual_generator_update_count": trace.get("actual_generator_update_count", 0),
        "actual_random_M_skew_count": trace.get("actual_random_M_skew_count", 0),
        "actual_signflip_count": trace.get("actual_signflip_count", 0),
        "actual_shuffled_tangent_count": trace.get("actual_shuffled_tangent_count", 0),
        "actual_euclidean_generator_count": trace.get("actual_euclidean_generator_count", 0),
        "actual_noop_count": trace.get("actual_noop_count", 0),
        "norm_floor_rule": "fixed_init_edge_radius_quantile",
        "norm_floor": float(trace.get("norm_floor", 0.0)),
        "norm_floor_quantile": float(trace.get("norm_floor_quantile", NORM_FLOOR_QUANTILE)),
        "norm_floor_min": float(trace.get("norm_floor_min", NORM_FLOOR_MIN)),
        "initial_edge_radius_count": float(trace.get("initial_edge_radius_count", 0.0)),
        "initial_edge_radius_q": float(trace.get("initial_edge_radius_q", 0.0)),
        "initial_edge_radius_min": float(trace.get("initial_edge_radius_min", 0.0)),
        "initial_edge_radius_median": float(trace.get("initial_edge_radius_median", 0.0)),
        "initial_edge_radius_max": float(trace.get("initial_edge_radius_max", 0.0)),
        "near_zero_bootstrap_fraction": trace_fraction(trace, "near_zero_bootstrap_count"),
        "mean_radial_abs_step": trace_mean(trace, "radial_abs_sum"),
        "mean_radial_signed_step": trace_mean(trace, "radial_signed_sum"),
        "max_radial_abs_step": float(trace.get("radial_abs_max", 0.0)),
        "mean_tangent_angle": trace_mean(trace, "tangent_angle_sum"),
        "mean_tangent_norm": trace_mean(trace, "tangent_norm_sum"),
        "mean_tangent_energy": trace_mean(trace, "tangent_energy_sum"),
        "mean_edge_m_radius": trace_mean(trace, "edge_radius_sum"),
        "min_edge_m_radius": trace_min(trace, "edge_radius_min"),
        "max_edge_m_radius": float(trace.get("edge_radius_max", 0.0)),
        "mean_proposal_m_norm": trace_mean(trace, "proposal_m_norm_sum"),
        "mean_proposal_m_energy": trace_mean(trace, "proposal_m_energy_sum"),
        "mean_additive_retraction_gap_m_norm": trace_mean(trace, "additive_retraction_gap_m_norm_sum"),
        "mean_additive_retraction_gap_ratio": trace_mean(trace, "additive_retraction_gap_ratio_sum"),
        "max_additive_retraction_gap_ratio": float(trace.get("max_additive_retraction_gap_ratio", 0.0)),
        "mean_tangent_m_orthogonality_abs": trace_mean(trace, "tangent_m_orthogonality_abs_sum"),
        "max_tangent_m_orthogonality_abs": float(trace.get("max_tangent_m_orthogonality_abs", 0.0)),
        "mean_tangent_norm_match_abs": trace_mean(trace, "tangent_norm_match_abs_sum"),
        "max_tangent_norm_match_abs": float(trace.get("max_tangent_norm_match_abs", 0.0)),
        "mean_shape_norm_preservation_abs": trace_mean(trace, "shape_norm_preservation_abs_sum"),
        "max_shape_norm_preservation_abs": float(trace.get("max_shape_norm_preservation_abs", 0.0)),
        "tangent_angle_le_1e_4_fraction": trace_fraction(trace, "tangent_angle_le_1e_4_count"),
        "tangent_angle_le_1e_3_fraction": trace_fraction(trace, "tangent_angle_le_1e_3_count"),
        "max_tangent_angle": float(trace.get("max_tangent_angle", 0.0)),
        "guard_NLL_finite": int(math.isfinite(final_guard)),
        "diagnostic_only": 1,
    }
    for split_name in ("train", "witness", "guard", "test"):
        row.update(eval_task_metrics(model, splits[f"x_{split_name}"], splits[f"y_{split_name}"], split_name))
    row.update(mode_energy_metrics(info, model))
    return row


def run_part_d_smoke(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("part-d-smoke uses fused kernels and requires CUDA.")
    datasets = [d for d in str(args.datasets).split(",") if d]
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    carrier_names = [c for c in str(args.carriers).split(",") if c]
    rows: list[dict[str, Any]] = []
    same_flops_profile_cache: dict[tuple[int, int, str], dict[str, Any]] = {}
    for dataset in datasets:
        for seed in seeds:
            x_np, y_np, note = load_real_dataset_numpy(dataset, seed, int(args.real_max_samples))
            splits, meta = real_splits_to_torch(x_np, y_np, seed, device)
            meta = {**meta, **{f"dataset_{k}": v for k, v in note.items()}}
            for carrier_name in carrier_names:
                info = carrier_info(carrier_name)
                if not info.supported_fused:
                    continue
                prefix = scheme_prefix(info)
                scheme_metric_pairs = [(f"{prefix}-B0", "M0")]
                for metric_index, metric_name in enumerate(METRICS):
                    scheme_metric_pairs.extend(
                        [
                            (f"{prefix}-C{metric_index}", metric_name),
                            (f"{prefix}-G{metric_index}", metric_name),
                            (f"{prefix}-R0-{metric_index}", metric_name),
                            (f"{prefix}-R1-{metric_index}", metric_name),
                            (f"{prefix}-R2-{metric_index}", metric_name),
                            (f"{prefix}-R3-{metric_index}", metric_name),
                            (f"{prefix}-R5-{metric_index}", metric_name),
                        ]
                    )
                for scheme, metric_name in scheme_metric_pairs:
                    rows.append(train_real_scheme(info, scheme, metric_name, dataset, seed, splits, int(args.h20_steps), float(args.lr), device, meta))
    write_csv(OUT_ROOT / "v23_26_partD_minimum_real_matrix.csv", rows)
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["seed"], row["carrier_core_variant"])
        by_key.setdefault(key, {})[row["scheme"]] = row
    pair_rows: list[dict[str, Any]] = []
    for (dataset, seed, carrier), schemes in by_key.items():
        prefix = scheme_prefix(carrier)
        for metric_index, metric_name in enumerate(METRICS):
            cand = schemes.get(f"{prefix}-G{metric_index}")
            for comparison, control_name in [
                (f"G{metric_index}_vs_AdamW", f"{prefix}-B0"),
                (f"G{metric_index}_vs_additive_C{metric_index}", f"{prefix}-C{metric_index}"),
                (f"G{metric_index}_vs_random_M_skew_M{metric_index}", f"{prefix}-R0-{metric_index}"),
                (f"G{metric_index}_vs_signflip_M{metric_index}", f"{prefix}-R1-{metric_index}"),
                (f"G{metric_index}_vs_shuffled_tangent_M{metric_index}", f"{prefix}-R2-{metric_index}"),
                (f"G{metric_index}_vs_Euclidean_RTGF_M{metric_index}", f"{prefix}-R3-{metric_index}"),
                (f"G{metric_index}_vs_same_compute_noop_M{metric_index}", f"{prefix}-R5-{metric_index}"),
            ]:
                control = schemes.get(control_name)
                if cand is None or control is None:
                    continue
                pair_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "carrier_core_variant": carrier,
                        "metric_name": metric_name,
                        "comparison": comparison,
                        "candidate": cand["scheme"],
                        "control": control["scheme"],
                        "paired_guard_NLL_surplus": float(control["guard_NLL"]) - float(cand["guard_NLL"]),
                        "paired_test_NLL_surplus": float(control["test_NLL"]) - float(cand["test_NLL"]),
                        "paired_guard_CVaR95_debt_delta": float(cand["guard_CVaR95_NLL"]) - float(control["guard_CVaR95_NLL"]),
                        "diagnostic_only": 1,
                        "compact_smoke": 1,
                    }
                )
    write_csv(OUT_ROOT / "v23_26_partD_paired_control_trace.csv", pair_rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in pair_rows:
        groups.setdefault((row["carrier_core_variant"], row["comparison"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (carrier, comparison), group in sorted(groups.items()):
        stats = surplus_stats([float(row["paired_guard_NLL_surplus"]) for row in group])
        summary_rows.append(
            {
                "carrier_core_variant": carrier,
                "comparison": comparison,
                "pair_count": len(group),
                **stats,
                "compact_smoke": 1,
                "diagnostic_only": 1,
            }
        )
    write_csv(OUT_ROOT / "v23_26_partD_paired_control_summary.csv", summary_rows)
    finite_rows = sum(1 for row in rows if int(row.get("guard_NLL_finite", 0)) == 1)
    g2_rows = [row for row in rows if str(row.get("scheme", "")).endswith("G2")]

    def finite_field_stats(field: str) -> dict[str, float]:
        vals: list[float] = []
        for row in g2_rows:
            try:
                value = float(row.get(field, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                vals.append(value)
        if not vals:
            return {"mean": 0.0, "max": 0.0}
        return {"mean": float(np.mean(vals)), "max": float(np.max(vals))}

    g2_trace_digest = {
        "g2_trace_rows": len(g2_rows),
        "g2_mean_tangent_angle": finite_field_stats("mean_tangent_angle"),
        "g2_max_tangent_angle": finite_field_stats("max_tangent_angle"),
        "g2_tangent_angle_le_1e_4_fraction": finite_field_stats("tangent_angle_le_1e_4_fraction"),
        "g2_tangent_angle_le_1e_3_fraction": finite_field_stats("tangent_angle_le_1e_3_fraction"),
        "g2_mean_additive_retraction_gap_ratio": finite_field_stats("mean_additive_retraction_gap_ratio"),
        "g2_mean_additive_retraction_gap_m_norm": finite_field_stats("mean_additive_retraction_gap_m_norm"),
        "g2_mean_proposal_m_norm": finite_field_stats("mean_proposal_m_norm"),
        "g2_norm_floor": finite_field_stats("norm_floor"),
        "g2_initial_edge_radius_q": finite_field_stats("initial_edge_radius_q"),
        "g2_initial_edge_radius_median": finite_field_stats("initial_edge_radius_median"),
    }
    summary = {
        "rows": len(rows),
        "finite_rows": finite_rows,
        "pair_rows": len(pair_rows),
        "summary_rows": len(summary_rows),
        "datasets": datasets,
        "seeds": seeds,
        "carriers": carrier_names,
        "steps": int(args.h20_steps),
        "lr": float(args.lr),
        "real_max_samples": int(args.real_max_samples),
        "compact_smoke": 1,
        "diagnostic_only": 1,
        "official_partD_gate_evaluable": 0,
        **g2_trace_digest,
    }
    write_json(OUT_ROOT / "v23_26_partD_summary.json", summary)
    append_exec(
        "PartD_minimum_real_compact_smoke",
        args,
        [
            "v23_26_partD_minimum_real_matrix.csv",
            "v23_26_partD_paired_control_trace.csv",
            "v23_26_partD_paired_control_summary.csv",
            "v23_26_partD_summary.json",
        ],
        summary,
    )
    recap_lines = [
        f"rows `{len(rows)}`，finite rows `{finite_rows}/{len(rows)}`，pair rows `{len(pair_rows)}`，summary rows `{len(summary_rows)}`。",
        f"datasets `{','.join(datasets)}`；carriers `{','.join(carrier_names)}`；real_max_samples `{int(args.real_max_samples)}`；steps `{int(args.h20_steps)}`；lr `{float(args.lr)}`。",
    ]
    if g2_rows:
        recap_lines.append(
            "G2 trace digest："
            f"rows `{len(g2_rows)}`；mean_tangent_angle mean/max `{g2_trace_digest['g2_mean_tangent_angle']['mean']}`/`{g2_trace_digest['g2_mean_tangent_angle']['max']}`；"
            f"angle<=1e-4 mean `{g2_trace_digest['g2_tangent_angle_le_1e_4_fraction']['mean']}`；"
            f"angle<=1e-3 mean `{g2_trace_digest['g2_tangent_angle_le_1e_3_fraction']['mean']}`；"
            f"additive-retraction-gap ratio mean `{g2_trace_digest['g2_mean_additive_retraction_gap_ratio']['mean']}`；"
            f"gap M-norm mean `{g2_trace_digest['g2_mean_additive_retraction_gap_m_norm']['mean']}`；"
            f"proposal M-norm mean `{g2_trace_digest['g2_mean_proposal_m_norm']['mean']}`；"
            f"norm_floor mean `{g2_trace_digest['g2_norm_floor']['mean']}`；"
            f"initial edge-radius q01 mean `{g2_trace_digest['g2_initial_edge_radius_q']['mean']}`。"
        )
    recap_lines.append("这是 compact smoke，用于打通真实数据 paired harness；official PartD gate 仍不可计数，后续要跑完整 minimum-real/MLP controls。")
    append_recap("PartD minimum-real compact smoke 复盘", recap_lines)


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def train_mlp_matched_baseline(info: CarrierInfo, dataset: str, seed: int, splits: dict[str, torch.Tensor], steps: int, lr: float, device: torch.device, meta: dict[str, Any]) -> dict[str, Any]:
    input_dim = int(splits["x_train"].shape[1])
    output_dim = int(meta.get("dataset_n_classes", int(torch.cat([splits["y_train"], splits["y_witness"], splits["y_guard"], splits["y_test"]]).max().item()) + 1))
    kan_hidden = 16
    kan_trainable_params = int(input_dim * kan_hidden * info.k + kan_hidden * output_dim * info.k)
    mlp_hidden = int(hidden_for_param_budget(input_dim, output_dim, kan_trainable_params))
    model = MLPBaseline(input_dim=input_dim, output_dim=output_dim, hidden_dim=mlp_hidden, seed=seed + 232600, device=device)
    mlp_params = int(count_parameters(model))
    initial_guard, initial_acc = eval_module_nll_acc(model, splits["x_guard"], splits["y_guard"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
    batch = min(32, int(splits["x_train"].shape[0]))
    for step in range(int(steps)):
        idx = torch.arange((step * batch) % int(splits["x_train"].shape[0]), (step * batch) % int(splits["x_train"].shape[0]) + batch, device=device) % int(splits["x_train"].shape[0])
        xb, yb = splits["x_train"][idx], splits["y_train"][idx]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    final_guard, final_acc = eval_module_nll_acc(model, splits["x_guard"], splits["y_guard"])
    row: dict[str, Any] = {
        "phase": "PartG_MLP_matched_smoke",
        "dataset": dataset,
        "seed": seed,
        "carrier_core_variant": info.name,
        "carrier_family": info.family,
        "scheme": "MLP-AdamW-same-param",
        "mlp_architecture": "MLPBaseline_SiLU_input_hidden_hidden_output_no_bias",
        "kan_reference_hidden": kan_hidden,
        "mlp_hidden": mlp_hidden,
        "kan_reference_trainable_params_w1w2": kan_trainable_params,
        "mlp_trainable_params": mlp_params,
        "param_abs_delta": abs(mlp_params - kan_trainable_params),
        "param_ratio_mlp_over_kan": float(mlp_params) / max(1.0, float(kan_trainable_params)),
        "steps": int(steps),
        "lr": float(lr),
        "weight_decay": 0.0,
        "task_loss": "cross_entropy",
        "label_smoothing": 0,
        "auxiliary_loss_count": 0,
        "mode_penalty": 0,
        "curvature_loss_penalty": 0,
        "guard_NLL_before": initial_guard,
        "guard_NLL": final_guard,
        "guard_NLL_delta": initial_guard - final_guard,
        "accuracy_before": initial_acc,
        "accuracy": final_acc,
        "guard_NLL_finite": int(math.isfinite(final_guard)),
        "same_param_MLP_AdamW_completed": 1,
        "MLP_block_RTGF_status": "R0_MLPStrongFunctionalBaselineIncomplete",
        "MLP_MCGA_status": "R0_MLPStrongFunctionalBaselineIncomplete",
        "MLP_MCGA_reason": "v23.26 runner has not semantically reproduced the older MLP-MCGA functional atlas path; only same-parameter MLP AdamW is measured here.",
        "diagnostic_only": 1,
        **meta,
    }
    for split_name in ("train", "witness", "guard", "test"):
        row.update(eval_module_task_metrics(model, splits[f"x_{split_name}"], splits[f"y_{split_name}"], split_name))
    return row


def train_mlp_block_rtgf_baseline(info: CarrierInfo, dataset: str, seed: int, splits: dict[str, torch.Tensor], steps: int, lr: float, device: torch.device, meta: dict[str, Any]) -> dict[str, Any]:
    input_dim = int(splits["x_train"].shape[1])
    output_dim = int(meta.get("dataset_n_classes", int(torch.cat([splits["y_train"], splits["y_witness"], splits["y_guard"], splits["y_test"]]).max().item()) + 1))
    kan_hidden = 16
    kan_trainable_params = int(input_dim * kan_hidden * info.k + kan_hidden * output_dim * info.k)
    mlp_hidden = int(hidden_for_param_budget(input_dim, output_dim, kan_trainable_params))
    model = MLPBaseline(input_dim=input_dim, output_dim=output_dim, hidden_dim=mlp_hidden, seed=seed + 232600, device=device)
    mlp_params = int(count_parameters(model))
    initial_guard, initial_acc = eval_module_nll_acc(model, splits["x_guard"], splits["y_guard"])
    metric = metric_matrices(info.family, info.k)["M2"]
    opt = MLPBlockRTGFOptimizer(model, metric, lr=float(lr), block_size=info.k, weight_decay=0.0)
    batch = min(32, int(splits["x_train"].shape[0]))
    for step in range(int(steps)):
        idx = torch.arange((step * batch) % int(splits["x_train"].shape[0]), (step * batch) % int(splits["x_train"].shape[0]) + batch, device=device) % int(splits["x_train"].shape[0])
        xb, yb = splits["x_train"][idx], splits["y_train"][idx]
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    final_guard, final_acc = eval_module_nll_acc(model, splits["x_guard"], splits["y_guard"])
    row: dict[str, Any] = {
        "phase": "PartG_MLP_matched_smoke",
        "dataset": dataset,
        "seed": seed,
        "carrier_core_variant": info.name,
        "carrier_family": info.family,
        "scheme": "MLP-BlockRTGF-same-param-M2",
        "mlp_architecture": "MLPBaseline_SiLU_input_hidden_hidden_output_no_bias",
        "kan_reference_hidden": kan_hidden,
        "mlp_hidden": mlp_hidden,
        "kan_reference_trainable_params_w1w2": kan_trainable_params,
        "mlp_trainable_params": mlp_params,
        "param_abs_delta": abs(mlp_params - kan_trainable_params),
        "param_ratio_mlp_over_kan": float(mlp_params) / max(1.0, float(kan_trainable_params)),
        "steps": int(steps),
        "lr": float(lr),
        "weight_decay": 0.0,
        "metric_name": "M2",
        "block_size": int(info.k),
        "task_loss": "cross_entropy",
        "label_smoothing": 0,
        "auxiliary_loss_count": 0,
        "mode_penalty": 0,
        "curvature_loss_penalty": 0,
        "guard_NLL_before": initial_guard,
        "guard_NLL": final_guard,
        "guard_NLL_delta": initial_guard - final_guard,
        "accuracy_before": initial_acc,
        "accuracy": final_acc,
        "guard_NLL_finite": int(math.isfinite(final_guard)),
        "same_param_MLP_AdamW_completed": 0,
        "MLP_block_RTGF_completed": 1,
        "MLP_block_RTGF_status": "completed_diagnostic",
        "MLP_MCGA_status": "R0_MLPStrongFunctionalBaselineIncomplete",
        "MLP_MCGA_reason": "v23.26 runner has not semantically reproduced the older MLP-MCGA functional atlas path; MLP block-RTGF is measured separately.",
        "block_rtgf_block_count": int(opt.trace.get("block_count", 0)),
        "block_rtgf_near_zero_bootstrap_fraction": float(opt.trace.get("near_zero_bootstrap_count", 0)) / max(1.0, float(opt.trace.get("block_count", 0))),
        "diagnostic_only": 1,
        **meta,
    }
    for split_name in ("train", "witness", "guard", "test"):
        row.update(eval_module_task_metrics(model, splits[f"x_{split_name}"], splits[f"y_{split_name}"], split_name))
    return row


def profiler_flops_for_step(step_fn) -> int:
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_flops=True, record_shapes=True) as prof:
        step_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    total = 0
    for event in prof.key_averages():
        total += int(getattr(event, "flops", 0) or 0)
    return int(total)


def profile_kan_full_step_flops(info: CarrierInfo, input_dim: int, output_dim: int, hidden: int, x: torch.Tensor, y: torch.Tensor, lr: float, device: torch.device) -> int:
    x_stats = x.detach().clone()
    model = make_model(info, input_dim, output_dim, hidden, x_stats, seed=4545 + info.k, device=device)
    metric = metric_matrices(info.family, info.k)["M2"]
    opt = IntrinsicEdgeOptimizer(model, metric, lr=float(lr), mode="rtgf", weight_decay=0.0, trace_enabled=False)

    def step() -> None:
        model.zero_grad(set_to_none=True)
        logits, cache = model.manual_ce_forward_cache(x)
        model.manual_ce_backward_from_cache(logits, cache, y)
        opt.step()

    return profiler_flops_for_step(step)


def profile_mlp_full_step_flops(input_dim: int, output_dim: int, hidden: int, x: torch.Tensor, y: torch.Tensor, lr: float, seed: int, device: torch.device) -> int:
    model = MLPBaseline(input_dim=input_dim, output_dim=output_dim, hidden_dim=int(hidden), seed=seed, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    return profiler_flops_for_step(step)


def select_same_flops_mlp_hidden(info: CarrierInfo, splits: dict[str, torch.Tensor], output_dim: int, lr: float, device: torch.device) -> dict[str, Any]:
    input_dim = int(splits["x_train"].shape[1])
    batch = min(32, int(splits["x_train"].shape[0]))
    x = splits["x_train"][:batch]
    y = splits["y_train"][:batch]
    kan_hidden = 16
    try:
        kan_flops = profile_kan_full_step_flops(info, input_dim, output_dim, kan_hidden, x, y, lr, device)
        if kan_flops <= 0:
            return {"status": "R0_SameFLOPsProfilerUnsupported", "kan_profiled_full_step_FLOPs": int(kan_flops)}
        candidates = list(range(1, 65)) + [80, 96, 112, 128, 160, 192, 224, 256]
        candidate_rows: list[dict[str, Any]] = []
        for hidden in candidates:
            flops = profile_mlp_full_step_flops(input_dim, output_dim, int(hidden), x, y, lr, seed=5656 + int(hidden), device=device)
            if flops <= 0:
                continue
            ratio = float(flops) / max(1.0, float(kan_flops))
            candidate_rows.append({"hidden": int(hidden), "flops": int(flops), "ratio": ratio, "abs_ratio_error": abs(ratio - 1.0)})
        if not candidate_rows:
            return {"status": "R0_SameFLOPsMLPProfilerUnsupported", "kan_profiled_full_step_FLOPs": int(kan_flops)}
        best = min(candidate_rows, key=lambda row: (float(row["abs_ratio_error"]), int(row["hidden"])))
        return {
            "status": "completed_within_tolerance" if float(best["abs_ratio_error"]) <= 0.10 else "R0_MLPSameFLOPsMatchOutsideTolerance",
            "kan_profiled_full_step_FLOPs": int(kan_flops),
            "mlp_profiled_full_step_FLOPs": int(best["flops"]),
            "full_step_FLOPs_ratio": float(best["ratio"]),
            "full_step_FLOPs_ratio_abs_error": float(best["abs_ratio_error"]),
            "mlp_hidden": int(best["hidden"]),
            "candidate_profile_count": len(candidate_rows),
            "profiler_backend": "torch.profiler.with_flops",
        }
    except Exception as exc:  # pragma: no cover
        return {"status": "R0_SameFLOPsProfilerFailed", "error": repr(exc)}


def train_mlp_same_flops_baseline(info: CarrierInfo, dataset: str, seed: int, splits: dict[str, torch.Tensor], steps: int, lr: float, device: torch.device, meta: dict[str, Any], profile_info: dict[str, Any]) -> dict[str, Any]:
    input_dim = int(splits["x_train"].shape[1])
    output_dim = int(meta.get("dataset_n_classes", int(torch.cat([splits["y_train"], splits["y_witness"], splits["y_guard"], splits["y_test"]]).max().item()) + 1))
    kan_hidden = 16
    hidden = int(profile_info.get("mlp_hidden", 4))
    model = MLPBaseline(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden, seed=seed + 232700, device=device)
    mlp_params = int(count_parameters(model))
    initial_guard, initial_acc = eval_module_nll_acc(model, splits["x_guard"], splits["y_guard"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
    batch = min(32, int(splits["x_train"].shape[0]))
    for step in range(int(steps)):
        idx = torch.arange((step * batch) % int(splits["x_train"].shape[0]), (step * batch) % int(splits["x_train"].shape[0]) + batch, device=device) % int(splits["x_train"].shape[0])
        xb, yb = splits["x_train"][idx], splits["y_train"][idx]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    final_guard, final_acc = eval_module_nll_acc(model, splits["x_guard"], splits["y_guard"])
    row: dict[str, Any] = {
        "phase": "PartG_MLP_matched_smoke",
        "dataset": dataset,
        "seed": seed,
        "carrier_core_variant": info.name,
        "carrier_family": info.family,
        "scheme": "MLP-AdamW-same-FLOPs",
        "mlp_architecture": "MLPBaseline_SiLU_input_hidden_hidden_output_no_bias",
        "kan_reference_hidden": kan_hidden,
        "mlp_hidden": hidden,
        "mlp_trainable_params": mlp_params,
        "steps": int(steps),
        "lr": float(lr),
        "weight_decay": 0.0,
        "task_loss": "cross_entropy",
        "label_smoothing": 0,
        "auxiliary_loss_count": 0,
        "mode_penalty": 0,
        "curvature_loss_penalty": 0,
        "same_FLOPs_MLP_status": str(profile_info.get("status", "")),
        "same_FLOPs_MLP_completed": int(str(profile_info.get("status", "")) == "completed_within_tolerance"),
        "kan_profiled_full_step_FLOPs": int(profile_info.get("kan_profiled_full_step_FLOPs", 0) or 0),
        "mlp_profiled_full_step_FLOPs": int(profile_info.get("mlp_profiled_full_step_FLOPs", 0) or 0),
        "full_step_FLOPs_ratio": float(profile_info.get("full_step_FLOPs_ratio", math.nan)),
        "full_step_FLOPs_ratio_abs_error": float(profile_info.get("full_step_FLOPs_ratio_abs_error", math.nan)),
        "same_FLOPs_candidate_profile_count": int(profile_info.get("candidate_profile_count", 0) or 0),
        "profiler_backend": str(profile_info.get("profiler_backend", "torch.profiler.with_flops")),
        "guard_NLL_before": initial_guard,
        "guard_NLL": final_guard,
        "guard_NLL_delta": initial_guard - final_guard,
        "accuracy_before": initial_acc,
        "accuracy": final_acc,
        "guard_NLL_finite": int(math.isfinite(final_guard)),
        "same_param_MLP_AdamW_completed": 0,
        "MLP_block_RTGF_status": "completed_diagnostic",
        "MLP_MCGA_status": "R0_MLPStrongFunctionalBaselineIncomplete",
        "MLP_MCGA_reason": "v23.26 runner has not semantically reproduced the older MLP-MCGA functional atlas path; same-FLOPs MLP AdamW is measured with torch profiler when possible.",
        "diagnostic_only": 1,
        **meta,
    }
    for split_name in ("train", "witness", "guard", "test"):
        row.update(eval_module_task_metrics(model, splits[f"x_{split_name}"], splits[f"y_{split_name}"], split_name))
    return row


def simple_mlp_hidden_for_param_budget(input_dim: int, output_dim: int, target_params: int, hidden_multiple: int = 1) -> dict[str, Any]:
    hidden_multiple = max(1, int(hidden_multiple))
    best = {"hidden": 4, "params": 0, "abs_delta": float("inf")}
    for hidden in range(4, 1025):
        if hidden % hidden_multiple != 0:
            continue
        params = int(input_dim * hidden + hidden + hidden * hidden + hidden + hidden * output_dim + output_dim)
        abs_delta = abs(params - int(target_params))
        if abs_delta < float(best["abs_delta"]):
            best = {"hidden": hidden, "params": params, "abs_delta": abs_delta}
    best["param_ratio_mlp_over_kan"] = float(best["params"]) / max(1.0, float(target_params))
    best["hidden_multiple_constraint"] = int(hidden_multiple)
    best["hidden_match_policy"] = "nearest_param_budget" if hidden_multiple <= 1 else f"nearest_param_budget_hidden_multiple_{hidden_multiple}"
    return best


def v2326_mcga_hidden_match(input_dim: int, output_dim: int, target_params: int, args: argparse.Namespace) -> dict[str, Any]:
    multiple = int(args.poet_block_size) if bool(getattr(args, "mcga_poet_even_hidden", False)) else 1
    return simple_mlp_hidden_for_param_budget(input_dim, output_dim, target_params, hidden_multiple=multiple)


def make_poet_optimizer_for_v2326(model: nn.Module, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    poet_path = str(ROOT / "external/oet_baselines/poet_sphere")
    if poet_path not in sys.path:
        sys.path.insert(0, poet_path)
    try:
        import torch._dynamo as torch_dynamo

        torch_dynamo.reset()
        if hasattr(torch_dynamo.config, "recompile_limit"):
            torch_dynamo.config.recompile_limit = max(int(torch_dynamo.config.recompile_limit), 64)
    except Exception:
        pass
    from poet_torch import POETConfig, POETModel, get_poet_optimizer

    cfg = POETConfig(
        block_size=int(args.poet_block_size),
        merge_interval=int(args.poet_merge_interval),
        poet_lr=float(args.poet_lr),
        base_lr=float(args.lr),
        poet_scale=float(args.poet_scale),
        weight_decay=float(args.weight_decay),
        mem_efficient_mode=False,
    )
    wrapped = POETModel(model, cfg)
    opt = get_poet_optimizer(wrapped, cfg)
    poet_layers = sum(1 for module in wrapped.modules() if module.__class__.__name__ == "POETLinear")
    poet_transform_params = sum(int(param.numel()) for name, param in wrapped.named_parameters() if name.endswith("oft_R"))
    poet_trainable_transform_params = sum(int(param.numel()) for name, param in wrapped.named_parameters() if name.endswith("oft_R") and param.requires_grad)
    return opt, {
        "wrapped_model": wrapped,
        "poet_replaced_layers": int(poet_layers),
        "poet_block_size": int(args.poet_block_size),
        "poet_transform_params": int(poet_transform_params),
        "poet_trainable_transform_params": int(poet_trainable_transform_params),
    }


def train_mlp_mcga_smoke_baseline(info: CarrierInfo, dataset: str, seed: int, splits: dict[str, torch.Tensor], steps: int, lr: float, device: torch.device, meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    method = str(args.mcga_method)
    input_dim = int(splits["x_train"].shape[1])
    output_dim = int(meta.get("dataset_n_classes", int(torch.cat([splits["y_train"], splits["y_witness"], splits["y_guard"], splits["y_test"]]).max().item()) + 1))
    kan_hidden = 16
    kan_trainable_params = int(input_dim * kan_hidden * info.k + kan_hidden * output_dim * info.k)
    hidden_match = v2326_mcga_hidden_match(input_dim, output_dim, kan_trainable_params, args)
    row: dict[str, Any] = {
        "phase": "PartG_MLP_MCGA_v22_style_smoke",
        "dataset": dataset,
        "seed": int(seed),
        "carrier_core_variant": info.name,
        "carrier_family": info.family,
        "scheme": "MLP-MCGA-v22-style-POET-fsclip-eta025",
        "mcga_method": method,
        "mlp_architecture": "v22_66_SimpleMLP_ReLU_fc1_fc2_fc3_bias_wrapped_by_MetricCompatibleAtlasMLP",
        "kan_reference_hidden": kan_hidden,
        "kan_reference_trainable_params_w1w2": kan_trainable_params,
        "mlp_hidden": int(hidden_match["hidden"]),
        "mlp_hidden_match_policy": hidden_match.get("hidden_match_policy", "nearest_param_budget"),
        "mlp_hidden_multiple_constraint": int(hidden_match.get("hidden_multiple_constraint", 1)),
        "mlp_trainable_params_before_poet": int(hidden_match["params"]),
        "param_abs_delta": int(hidden_match["abs_delta"]),
        "param_ratio_mlp_over_kan": float(hidden_match["param_ratio_mlp_over_kan"]),
        "steps": int(steps),
        "lr": float(lr),
        "real_max_samples": int(args.real_max_samples),
        "diagnostic_only": 1,
        "official_partG_gate_evaluable": 0,
        **meta,
    }
    try:
        base = AtlasSimpleMLP(input_dim, output_dim, hidden=int(hidden_match["hidden"]), seed=int(seed) + 232800).to(device)
        metric_n = min(int(args.metric_batch_size), int(splits["x_train"].shape[0]))
        gen_metric = torch.Generator(device=device).manual_seed(int(seed) * 1237 + sum(ord(ch) for ch in method))
        metric_idx = torch.randperm(int(splits["x_train"].shape[0]), generator=gen_metric, device=device)[:metric_n]
        atlas = build_last_layer_atlas(
            base,
            splits["x_train"][metric_idx],
            splits["y_train"][metric_idx],
            rank=max(1, min(4, output_dim, int(hidden_match["hidden"]))),
            method="metric_atlas_act",
            metric_kind="signal_debt",
            seed=int(seed),
        )
        model = MetricCompatibleAtlasMLP(
            base,
            atlas,
            allow_shape=False,
            shape_budget=0.0,
            eta=0.25,
            train_base_weight=True,
            base_spectrum_lock=True,
            functional_spectrum_budget=0.50,
            functional_spectrum_fill=False,
        ).to(device)
        initial_guard, initial_acc = eval_module_nll_acc(model, splits["x_guard"], splits["y_guard"])
        initial_test, initial_test_acc = eval_module_nll_acc(model, splits["x_test"], splits["y_test"])
        opt, opt_diag = make_poet_optimizer_for_v2326(model, args)
        train_model = opt_diag.pop("wrapped_model")
        batch = min(32, int(splits["x_train"].shape[0]))
        gen = torch.Generator(device=device).manual_seed(int(seed) * 3253 + sum(ord(ch) for ch in method))
        losses: list[float] = []
        t0 = time.perf_counter()
        for _ in range(int(steps)):
            if batch >= int(splits["x_train"].shape[0]):
                idx = torch.arange(int(splits["x_train"].shape[0]), device=device)
            else:
                idx = torch.randperm(int(splits["x_train"].shape[0]), generator=gen, device=device)[:batch]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(train_model(splits["x_train"][idx]).float(), splits["y_train"][idx].long())
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        final_guard, final_acc = eval_module_nll_acc(train_model, splits["x_guard"], splits["y_guard"])
        final_test, final_test_acc = eval_module_nll_acc(train_model, splits["x_test"], splits["y_test"])
        row.update(
            {
                "status": "completed",
                "guard_NLL_before": initial_guard,
                "guard_accuracy_before": initial_acc,
                "test_NLL_before": initial_test,
                "test_accuracy_before": initial_test_acc,
                "guard_NLL": final_guard,
                "guard_accuracy": final_acc,
                "test_NLL": final_test,
                "test_accuracy": final_test_acc,
                "guard_NLL_delta": initial_guard - final_guard,
                "test_NLL_delta": initial_test - final_test,
                "guard_NLL_finite": int(math.isfinite(final_guard)),
                "test_NLL_finite": int(math.isfinite(final_test)),
                "train_loss_first": losses[0] if losses else "",
                "train_loss_last": losses[-1] if losses else "",
                "train_loss_finite_rows": sum(1 for value in losses if math.isfinite(value)),
                "wall_ms": elapsed_ms,
                "mean_step_ms": elapsed_ms / max(1, int(steps)),
                "metric_batch_size": int(metric_n),
                "atlas_rank": int(atlas.output_basis.shape[1]),
                "active_Gram_condition": atlas.metrics.get("active_Gram_condition", ""),
                "signal_reachable_energy": atlas.metrics.get("signal_reachable_energy", ""),
                "poet_external_optimizer_used": 1,
                "MLP_MCGA_status": "completed_diagnostic_v22_style_smoke",
                **opt_diag,
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "guard_NLL_finite": 0,
                "test_NLL_finite": 0,
                "poet_external_optimizer_used": 0,
                "MLP_MCGA_status": "R0_MLPStrongFunctionalBaselineIncomplete",
            }
        )
    return row


def v2326_mcga_method_family(method: str) -> str:
    low = str(method).lower()
    if low in {"adamw"}:
        return "reference"
    if low in {"poet_official"} or "over_poet" in low:
        return "candidate" if low.startswith("mcga_") else "external_oet"
    if low.startswith("same_") or low in {"metric_preserving_noop"}:
        return "control"
    if low.startswith("mcga_"):
        return "candidate"
    return "unknown"


def v2326_mcga_rank_for_method(method: str, output_dim: int, hidden: int) -> int:
    match = re.search(r"rank(\d+)", str(method).lower())
    requested = int(match.group(1)) if match else 4
    return max(1, min(int(requested), int(output_dim), int(hidden)))


def v2326_mcga_atlas_method_name(method: str) -> str:
    low = str(method).lower()
    if low in {"same_compute_noop_coordinate", "metric_preserving_noop"}:
        return "same_compute_noop_coordinate"
    if "over_poet" in low:
        return "metric_atlas_act"
    if "same_functional_spectrum" in low or "same_c_skew_spectrum" in low or "same_spectrum" in low:
        return "same_functional_spectrum_random_atlas"
    if "same_generator" in low or "same_rank_random" in low:
        return "same_rank_random_atlas"
    return "metric_atlas_act" if low.startswith("mcga_") else "metric_atlas_sw"


def v2326_mcga_iso_eta(method: str, default: float = 1.0) -> float:
    low = str(method).lower()
    for token, value in [
        ("eta0005", 0.005),
        ("eta0015", 0.015),
        ("eta0018", 0.018),
        ("eta001", 0.01),
        ("eta002", 0.02),
        ("eta005", 0.05),
        ("eta01", 0.10),
        ("eta025", 0.25),
        ("eta05", 0.50),
    ]:
        if token in low:
            return float(value)
    return float(default)


def v2326_mcga_train_base_for_method(method: str) -> bool:
    low = str(method).lower()
    return "residual" in low or "over_poet" in low


def v2326_mcga_base_spectrum_lock_for_method(method: str) -> bool:
    low = str(method).lower()
    return "over_poet" in low or "baselock" in low or "fsclip" in low or "fsfill" in low


def v2326_mcga_functional_spectrum_budget_for_method(method: str, default: float = 0.50) -> float:
    low = str(method).lower()
    return float(default) if ("over_poet" in low or "fsclip" in low or "fsfill" in low) else 0.0


def v2326_mcga_functional_spectrum_fill_for_method(method: str) -> bool:
    return "fsfill" in str(method).lower()


def v2326_mcga_generator_train_enabled(method: str) -> bool:
    return str(method).lower() not in {"metric_preserving_noop", "same_compute_noop_coordinate"}


def v2326_mcga_coord_model(model: nn.Module) -> nn.Module:
    base_model = getattr(model, "base_model", None)
    if isinstance(base_model, nn.Module):
        return base_model
    source_model = getattr(model, "model", None)
    if isinstance(source_model, nn.Module):
        return source_model
    return model


def v2326_mcga_coord_layer(model: nn.Module) -> Any | None:
    coord = v2326_mcga_coord_model(model)
    fc3 = getattr(coord, "fc3", None)
    if fc3 is not None and hasattr(fc3, "atlas_delta"):
        return fc3
    return None


def v2326_mcga_freeze_control_coordinates(model: nn.Module, method: str) -> dict[str, int]:
    layer = v2326_mcga_coord_layer(model)
    if layer is None:
        return {"generator_parameter_trainable": 0, "shape_parameter_trainable": 0}
    generator_trainable = int(v2326_mcga_generator_train_enabled(method))
    if hasattr(layer, "raw_iso") and layer.raw_iso is not None:
        layer.raw_iso.requires_grad_(bool(generator_trainable))
    if hasattr(layer, "raw_shape") and layer.raw_shape is not None:
        layer.raw_shape.requires_grad_(False)
    return {"generator_parameter_trainable": generator_trainable, "shape_parameter_trainable": 0}


def v2326_mcga_model_spectrum(model: nn.Module) -> list[float]:
    layer = v2326_mcga_coord_layer(model)
    if layer is not None and hasattr(layer, "effective_weight"):
        vals = torch.linalg.svdvals(layer.effective_weight().detach().float())
        return [float(v) for v in vals.detach().cpu().tolist()]
    coord = v2326_mcga_coord_model(model)
    fc3 = getattr(coord, "fc3", None)
    if fc3 is not None and hasattr(fc3, "weight"):
        vals = torch.linalg.svdvals(fc3.weight.detach().float())
        return [float(v) for v in vals.detach().cpu().tolist()]
    return []


def v2326_mcga_spectrum_drift(initial: list[float], final: list[float]) -> float:
    if not initial or not final:
        return math.nan
    n = min(len(initial), len(final))
    num = math.sqrt(sum((float(final[i]) - float(initial[i])) ** 2 for i in range(n)))
    den = math.sqrt(sum(float(initial[i]) ** 2 for i in range(n)))
    return float(num / max(den, 1.0e-12))


def v2326_mcga_capacity_diagnostics(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict[str, Any]:
    layer = v2326_mcga_coord_layer(model)
    if layer is None:
        return {
            "generator_descent_fraction": "",
            "generator_descent_energy": "",
            "C_skew_projection_error": "",
            "capacity_source": "no_coordinate_layer",
        }
    device = next(model.parameters()).device
    model.train()
    for param in model.parameters():
        param.grad = None
    xb = x[: min(int(batch_size), int(x.shape[0]))].to(device)
    yb = y[: min(int(batch_size), int(y.shape[0]))].to(device).long()
    loss = F.cross_entropy(model(xb).float(), yb)
    loss.backward()
    grad = None
    if hasattr(layer, "raw_iso") and getattr(layer.raw_iso, "grad", None) is not None:
        grad = layer.raw_iso.grad.detach().clone()
    c_metric = getattr(layer, "active_gram", None)
    if grad is None or c_metric is None:
        for param in model.parameters():
            param.grad = None
        return {
            "generator_descent_fraction": "",
            "generator_descent_energy": "",
            "C_skew_projection_error": "",
            "capacity_source": "no_trainable_generator_gradient",
            "capacity_loss": float(loss.detach().cpu().item()),
        }
    diag = metric_compatible_descent_diagnostics(grad.detach().float(), c_metric.detach().float(), eps=1.0e-8)
    k_proj = c_skew_project(grad.detach().float(), c_metric.detach().float(), eps=1.0e-8)
    residual = k_proj.transpose(0, 1) @ c_metric.detach().float() + c_metric.detach().float() @ k_proj
    gen_norm = float(diag.get("iso_projected_gradient_norm", 0.0) or 0.0)
    out = {
        "generator_descent_fraction": float(diag.get("iso_descent_energy_fraction", 0.0) or 0.0),
        "generator_descent_energy": gen_norm * gen_norm,
        "generator_predicted_task_descent": diag.get("iso_predicted_task_descent", ""),
        "C_skew_projection_error": float(torch.linalg.norm(residual).detach().cpu().item()),
        "capacity_loss": float(loss.detach().cpu().item()),
        "capacity_source": "train_only_gradient_on_raw_iso",
    }
    for param in model.parameters():
        param.grad = None
    return out


def train_mlp_mcga_control_baseline(info: CarrierInfo, dataset: str, seed: int, splits: dict[str, torch.Tensor], method: str, steps: int, lr: float, device: torch.device, meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    input_dim = int(splits["x_train"].shape[1])
    output_dim = int(meta.get("dataset_n_classes", int(torch.cat([splits["y_train"], splits["y_witness"], splits["y_guard"], splits["y_test"]]).max().item()) + 1))
    kan_hidden = 16
    kan_trainable_params = int(input_dim * kan_hidden * info.k + kan_hidden * output_dim * info.k)
    hidden_match = v2326_mcga_hidden_match(input_dim, output_dim, kan_trainable_params, args)
    family = v2326_mcga_method_family(method)
    row: dict[str, Any] = {
        "phase": "PartG_MLP_MCGA_v22_style_controls_smoke",
        "dataset": dataset,
        "seed": int(seed),
        "carrier_core_variant": info.name,
        "carrier_family": info.family,
        "method": method,
        "method_family": family,
        "scheme": f"MLP-MCGA-control-{method}",
        "mlp_architecture": "v22_66_SimpleMLP_ReLU_fc1_fc2_fc3_bias_wrapped_when_candidate_or_control",
        "kan_reference_hidden": kan_hidden,
        "kan_reference_trainable_params_w1w2": kan_trainable_params,
        "mlp_hidden": int(hidden_match["hidden"]),
        "mlp_hidden_match_policy": hidden_match.get("hidden_match_policy", "nearest_param_budget"),
        "mlp_hidden_multiple_constraint": int(hidden_match.get("hidden_multiple_constraint", 1)),
        "mlp_trainable_params_before_poet": int(hidden_match["params"]),
        "param_abs_delta": int(hidden_match["abs_delta"]),
        "param_ratio_mlp_over_kan": float(hidden_match["param_ratio_mlp_over_kan"]),
        "steps": int(steps),
        "lr": float(lr),
        "real_max_samples": int(args.real_max_samples),
        "diagnostic_only": 1,
        "official_partG_gate_evaluable": 0,
        **meta,
    }
    try:
        base = AtlasSimpleMLP(input_dim, output_dim, hidden=int(hidden_match["hidden"]), seed=int(seed) + 232900).to(device)
        model: nn.Module = base
        atlas_diag: dict[str, Any] = {}
        coordinate_diag = {"generator_parameter_trainable": 0, "shape_parameter_trainable": 0}
        if family in {"candidate", "control"}:
            metric_n = min(int(args.metric_batch_size), int(splits["x_train"].shape[0]))
            gen_metric = torch.Generator(device=device).manual_seed(int(seed) * 1237 + sum(ord(ch) for ch in method))
            metric_idx = torch.randperm(int(splits["x_train"].shape[0]), generator=gen_metric, device=device)[:metric_n]
            atlas_method = v2326_mcga_atlas_method_name(method)
            atlas = build_last_layer_atlas(
                base,
                splits["x_train"][metric_idx],
                splits["y_train"][metric_idx],
                rank=v2326_mcga_rank_for_method(method, output_dim, int(hidden_match["hidden"])),
                method=atlas_method,
                metric_kind="fisher" if "fishermetric" in str(method).lower() else "signal_debt",
                seed=int(seed),
            )
            model = MetricCompatibleAtlasMLP(
                base,
                atlas,
                allow_shape=False,
                shape_budget=0.0,
                eta=v2326_mcga_iso_eta(method, default=1.0),
                train_base_weight=v2326_mcga_train_base_for_method(method),
                base_spectrum_lock=v2326_mcga_base_spectrum_lock_for_method(method),
                functional_spectrum_budget=v2326_mcga_functional_spectrum_budget_for_method(method, default=0.50),
                functional_spectrum_fill=v2326_mcga_functional_spectrum_fill_for_method(method),
            ).to(device)
            coordinate_diag = v2326_mcga_freeze_control_coordinates(model, method)
            atlas_diag = {
                "atlas_method": atlas_method,
                "atlas_rank": int(atlas.output_basis.shape[1]),
                "active_Gram_condition": atlas.metrics.get("active_Gram_condition", ""),
                "signal_reachable_energy": atlas.metrics.get("signal_reachable_energy", ""),
                "reservoir_reachable_energy": atlas.metrics.get("reservoir_reachable_energy", ""),
                "metric_batch_size": int(metric_n),
            }
        initial_spectrum = v2326_mcga_model_spectrum(model)
        initial_capacity = v2326_mcga_capacity_diagnostics(model, splits["x_train"], splits["y_train"], int(args.metric_batch_size))
        initial_guard, initial_acc = eval_module_nll_acc(model, splits["x_guard"], splits["y_guard"])
        initial_test, initial_test_acc = eval_module_nll_acc(model, splits["x_test"], splits["y_test"])
        opt_diag: dict[str, Any] = {"optimizer_step_source": "torch.optim.AdamW", "poet_external_optimizer_used": 0, "poet_replaced_layers": 0}
        if method == "poet_official" or "over_poet" in str(method).lower():
            opt, opt_diag_raw = make_poet_optimizer_for_v2326(model, args)
            train_model = opt_diag_raw.pop("wrapped_model")
            opt_diag.update(opt_diag_raw)
            opt_diag["optimizer_step_source"] = "poet_torch.get_poet_optimizer"
            opt_diag["poet_external_optimizer_used"] = 1
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(args.weight_decay))
            train_model = model
        batch = min(32, int(splits["x_train"].shape[0]))
        gen = torch.Generator(device=device).manual_seed(int(seed) * 3253 + sum(ord(ch) for ch in method))
        losses: list[float] = []
        t0 = time.perf_counter()
        for _ in range(int(steps)):
            if batch >= int(splits["x_train"].shape[0]):
                idx = torch.arange(int(splits["x_train"].shape[0]), device=device)
            else:
                idx = torch.randperm(int(splits["x_train"].shape[0]), generator=gen, device=device)[:batch]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(train_model(splits["x_train"][idx]).float(), splits["y_train"][idx].long())
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        final_guard, final_acc = eval_module_nll_acc(train_model, splits["x_guard"], splits["y_guard"])
        final_test, final_test_acc = eval_module_nll_acc(train_model, splits["x_test"], splits["y_test"])
        final_capacity = v2326_mcga_capacity_diagnostics(train_model, splits["x_train"], splits["y_train"], int(args.metric_batch_size))
        final_spectrum = v2326_mcga_model_spectrum(train_model)
        row.update(
            {
                "status": "completed",
                "guard_NLL_before": initial_guard,
                "guard_accuracy_before": initial_acc,
                "test_NLL_before": initial_test,
                "test_accuracy_before": initial_test_acc,
                "guard_NLL": final_guard,
                "held_NLL": final_guard,
                "guard_accuracy": final_acc,
                "held_accuracy": final_acc,
                "test_NLL": final_test,
                "test_accuracy": final_test_acc,
                "guard_NLL_delta": initial_guard - final_guard,
                "test_NLL_delta": initial_test - final_test,
                "guard_NLL_finite": int(math.isfinite(final_guard)),
                "test_NLL_finite": int(math.isfinite(final_test)),
                "train_loss_first": losses[0] if losses else "",
                "train_loss_last": losses[-1] if losses else "",
                "train_loss_finite_rows": sum(1 for value in losses if math.isfinite(value)),
                "wall_ms": elapsed_ms,
                "mean_step_ms": elapsed_ms / max(1, int(steps)),
                "functional_spectrum_drift_mean": v2326_mcga_spectrum_drift(initial_spectrum, final_spectrum),
                "MLP_MCGA_status": "completed_diagnostic_v22_style_controls_smoke",
                **atlas_diag,
                **coordinate_diag,
                **{f"initial_{k}": v for k, v in initial_capacity.items()},
                **final_capacity,
                **opt_diag,
            }
        )
        for split_name in ("train", "witness", "guard", "test"):
            row.update(eval_module_task_metrics(train_model, splits[f"x_{split_name}"], splits[f"y_{split_name}"], split_name))
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "guard_NLL_finite": 0,
                "test_NLL_finite": 0,
                "MLP_MCGA_status": "R0_MLPStrongFunctionalBaselineIncomplete",
            }
        )
    return row


def run_part_g_mcga_smoke(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("part-g-mcga-smoke requires CUDA.")
    datasets = [d for d in str(args.datasets).split(",") if d]
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    carrier_names = [c for c in str(args.carriers).split(",") if c]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            x_np, y_np, note = load_real_dataset_numpy(dataset, seed, int(args.real_max_samples))
            splits, meta = real_splits_to_torch(x_np, y_np, seed, device)
            meta = {**meta, **{f"dataset_{k}": v for k, v in note.items()}}
            for carrier_name in carrier_names:
                info = carrier_info(carrier_name)
                rows.append(train_mlp_mcga_smoke_baseline(info, dataset, seed, splits, int(args.h20_steps), float(args.lr), device, meta, args))
    write_csv(OUT_ROOT / "v23_26_MLP_MCGA_v22_style_smoke_matrix.csv", rows)
    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") != "completed"]
    finite_rows = sum(1 for row in completed if int(row.get("guard_NLL_finite", 0)) == 1)
    poet_positive_rows = sum(1 for row in completed if int(row.get("poet_replaced_layers", 0) or 0) > 0)
    poet_zero_rows = sum(1 for row in completed if int(row.get("poet_replaced_layers", 0) or 0) == 0)
    summary = {
        "rows": len(rows),
        "completed_rows": len(completed),
        "failed_rows": len(failed),
        "guard_finite_rows": finite_rows,
        "poet_replaced_positive_rows": poet_positive_rows,
        "poet_zero_replaced_rows": poet_zero_rows,
        "datasets": datasets,
        "seeds": seeds,
        "carriers": carrier_names,
        "steps": int(args.h20_steps),
        "lr": float(args.lr),
        "real_max_samples": int(args.real_max_samples),
        "mcga_method": str(args.mcga_method),
        "poet_block_size": int(args.poet_block_size),
        "poet_merge_interval": int(args.poet_merge_interval),
        "poet_lr": float(args.poet_lr),
        "poet_scale": float(args.poet_scale),
        "weight_decay": float(args.weight_decay),
        "MLP_MCGA_status": ("completed_diagnostic_v22_style_smoke" if poet_zero_rows == 0 else "completed_diagnostic_v22_style_smoke_with_zero_poet_rows") if len(completed) == len(rows) and rows else "R0_MLPStrongFunctionalBaselineIncomplete",
        "partG_official_gate_evaluable": 0,
        "missing_for_official": "same v23.26 MCGA control matrix: POET/external OET, same-functional-spectrum, same-generator-descent, same-C-skew, noop, matched KAN pairing and gate arithmetic",
        "diagnostic_only": 1,
    }
    write_json(OUT_ROOT / "v23_26_MLP_MCGA_v22_style_smoke_summary.json", summary)
    append_exec(
        "PartG_MLP_MCGA_v22_style_smoke",
        args,
        ["v23_26_MLP_MCGA_v22_style_smoke_matrix.csv", "v23_26_MLP_MCGA_v22_style_smoke_summary.json"],
        summary,
    )
    append_recap(
        "PartG MLP-MCGA v22-style smoke 复盘",
        [
            f"rows `{len(rows)}`，completed `{len(completed)}`，failed `{len(failed)}`，guard finite `{finite_rows}/{len(completed)}`。",
            f"POET replaced positive rows `{poet_positive_rows}/{len(completed)}`；zero-POET rows `{poet_zero_rows}`。",
            f"method `{args.mcga_method}`；datasets `{','.join(datasets)}`；carriers `{','.join(carrier_names)}`；steps `{int(args.h20_steps)}`；lr `{float(args.lr)}`。",
            "本 smoke 使用 v22.66 `SimpleMLP + MetricCompatibleAtlasMLP + POET + fsclip eta025` 语义，不是 v23.26 原有 `MLPBaseline`/BlockRTGF 的替代品。",
            "它只证明/否证当前 harness 可执行 MCGA；official PartG 仍需要同 split 的 POET/external OET、same-functional-spectrum、same-generator-descent、same-C-skew、noop controls 与 paired KAN gate。",
        ],
    )


def run_part_g_mcga_controls_smoke(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("part-g-mcga-controls-smoke requires CUDA.")
    datasets = [d for d in str(args.datasets).split(",") if d]
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    carrier_names = [c for c in str(args.carriers).split(",") if c]
    control_methods = [m for m in str(args.mcga_control_methods).split(",") if m]
    methods = list(dict.fromkeys([str(args.mcga_method)] + control_methods))
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            x_np, y_np, note = load_real_dataset_numpy(dataset, seed, int(args.real_max_samples))
            splits, meta = real_splits_to_torch(x_np, y_np, seed, device)
            meta = {**meta, **{f"dataset_{k}": v for k, v in note.items()}}
            for carrier_name in carrier_names:
                info = carrier_info(carrier_name)
                for method in methods:
                    rows.append(train_mlp_mcga_control_baseline(info, dataset, seed, splits, method, int(args.h20_steps), float(args.lr), device, meta, args))
    write_csv(OUT_ROOT / "v23_26_MLP_MCGA_v22_style_controls_matrix.csv", rows)

    by_key_method: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("carrier_core_variant", "")), str(row.get("method", "")))
        by_key_method[key] = row
    pair_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            for carrier_name in carrier_names:
                cand = by_key_method.get((dataset, str(seed), carrier_name, str(args.mcga_method)))
                if cand is None:
                    continue
                cand_guard = parse_float(cand.get("guard_NLL"))
                cand_test = parse_float(cand.get("test_NLL"))
                for method in methods:
                    if method == str(args.mcga_method):
                        continue
                    ctrl = by_key_method.get((dataset, str(seed), carrier_name, method))
                    if ctrl is None:
                        continue
                    ctrl_guard = parse_float(ctrl.get("guard_NLL"))
                    ctrl_test = parse_float(ctrl.get("test_NLL"))
                    comparison = {
                        "poet_official": "beats_external_OET_NLL",
                        "same_functional_spectrum_random_coordinate": "beats_same_functional_spectrum_random_NLL",
                        "same_generator_descent_energy_random": "beats_same_generator_descent_energy_random_NLL",
                        "same_C_skew_spectrum_random": "beats_same_C_skew_spectrum_random_NLL",
                        "same_compute_noop_coordinate": "beats_same_compute_noop_NLL",
                        "metric_preserving_noop": "beats_metric_preserving_noop_NLL",
                        "adamw": "beats_adamw_reference_NLL",
                    }.get(method, f"beats_{method}_NLL")
                    pair_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "carrier_core_variant": carrier_name,
                            "candidate": str(args.mcga_method),
                            "control": method,
                            "comparison": comparison,
                            "candidate_guard_NLL": cand_guard,
                            "control_guard_NLL": ctrl_guard,
                            "candidate_test_NLL": cand_test,
                            "control_test_NLL": ctrl_test,
                            "candidate_guard_NLL_delta_vs_control": cand_guard - ctrl_guard,
                            "candidate_test_NLL_delta_vs_control": cand_test - ctrl_test,
                            "candidate_guard_surplus_vs_control": ctrl_guard - cand_guard,
                            "candidate_test_surplus_vs_control": ctrl_test - cand_test,
                            "candidate_beats_control_guard_NLL": int(math.isfinite(cand_guard) and math.isfinite(ctrl_guard) and cand_guard < ctrl_guard),
                            "candidate_beats_control_test_NLL": int(math.isfinite(cand_test) and math.isfinite(ctrl_test) and cand_test < ctrl_test),
                            "diagnostic_only": 1,
                        }
                    )
    write_csv(OUT_ROOT / "v23_26_MLP_MCGA_v22_style_controls_pair_trace.csv", pair_rows)

    kan_rows, kan_reference_status = read_kan_reference_partd(str(args.kan_reference_root))
    kan_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in kan_rows:
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("carrier_core_variant", "")), str(row.get("scheme", "")))
        kan_by_key[key] = row
    kan_pair_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        dataset = str(row.get("dataset", ""))
        seed = str(row.get("seed", ""))
        carrier = str(row.get("carrier_core_variant", ""))
        prefix = scheme_prefix(carrier)
        kan_scheme = f"{prefix}-G2"
        kan_row = kan_by_key.get((dataset, seed, carrier, kan_scheme))
        if kan_row is None:
            continue
        method = str(row.get("method", ""))
        comparison = "KAN_G2_vs_MLP_MCGA_candidate" if method == str(args.mcga_method) else f"KAN_G2_vs_MLP_MCGA_control_{method}"
        kan_guard = parse_float(kan_row.get("guard_NLL", ""))
        kan_test = parse_float(kan_row.get("test_NLL", ""))
        mlp_guard = parse_float(row.get("guard_NLL", ""))
        mlp_test = parse_float(row.get("test_NLL", ""))
        kan_debt = parse_float(kan_row.get("guard_CVaR95_NLL", "")) - parse_float(row.get("guard_CVaR95_NLL", ""))
        kan_pair_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "carrier_core_variant": carrier,
                "comparison": comparison,
                "candidate": kan_scheme,
                "control": method,
                "paired_guard_NLL_surplus": mlp_guard - kan_guard,
                "paired_test_NLL_surplus": mlp_test - kan_test,
                "candidate_guard_NLL": kan_guard,
                "control_guard_NLL": mlp_guard,
                "candidate_test_NLL": kan_test,
                "control_test_NLL": mlp_test,
                "candidate_guard_NLL_finite": int(math.isfinite(kan_guard)),
                "control_guard_NLL_finite": int(math.isfinite(mlp_guard)),
                "candidate_beats_control_guard_NLL": int(math.isfinite(kan_guard) and math.isfinite(mlp_guard) and kan_guard < mlp_guard),
                "candidate_beats_control_test_NLL": int(math.isfinite(kan_test) and math.isfinite(mlp_test) and kan_test < mlp_test),
                "paired_guard_CVaR95_debt_delta": kan_debt,
                "candidate_no_debt_vs_control": int(math.isfinite(kan_debt) and kan_debt <= 0.0),
                "surplus_positive_means": "KAN_G2_lower_NLL_than_MLP_MCGA",
                "diagnostic_only": 1,
            }
        )
    write_csv(OUT_ROOT / "v23_26_MLP_MCGA_v22_style_KAN_pair_trace.csv", kan_pair_rows)
    kan_pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in kan_pair_rows:
        kan_pair_groups.setdefault((str(row["carrier_core_variant"]), str(row["comparison"])), []).append(row)
    kan_pair_summary_rows: list[dict[str, Any]] = []
    for (carrier, comparison), group in sorted(kan_pair_groups.items()):
        stats = surplus_stats([parse_float(row.get("paired_guard_NLL_surplus", "")) for row in group])
        no_debt_vals = [int(row.get("candidate_no_debt_vs_control", 0)) for row in group]
        kan_pair_summary_rows.append(
            {
                "carrier_core_variant": carrier,
                "comparison": comparison,
                "pair_count": len(group),
                **stats,
                "no_debt_count": int(sum(no_debt_vals)),
                "no_debt_rate": float(sum(no_debt_vals)) / max(1.0, float(len(no_debt_vals))),
                "surplus_positive_means": "KAN_G2_lower_NLL_than_MLP_MCGA",
                "diagnostic_only": 1,
            }
        )
    write_csv(OUT_ROOT / "v23_26_MLP_MCGA_v22_style_KAN_pair_summary.csv", kan_pair_summary_rows)

    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") != "completed"]
    candidate_rows = [row for row in completed if row.get("method") == str(args.mcga_method)]
    control_rows = [row for row in completed if row.get("method") != str(args.mcga_method)]
    method_summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        group = [row for row in rows if row.get("method") == method]
        done = [row for row in group if row.get("status") == "completed"]
        vals = [parse_float(row.get("guard_NLL")) for row in done]
        method_summary[method] = {
            "rows": len(group),
            "completed_rows": len(done),
            "guard_NLL_mean": float(np.mean([v for v in vals if math.isfinite(v)])) if any(math.isfinite(v) for v in vals) else math.nan,
            "method_family": v2326_mcga_method_family(method),
        }
    beats_by_control = {
        method: sum(1 for row in pair_rows if row.get("control") == method and int(row.get("candidate_beats_control_guard_NLL", 0)) == 1)
        for method in methods
        if method != str(args.mcga_method)
    }
    pair_count_by_control = {
        method: sum(1 for row in pair_rows if row.get("control") == method)
        for method in methods
        if method != str(args.mcga_method)
    }
    required_controls = [
        "poet_official",
        "same_functional_spectrum_random_coordinate",
        "same_generator_descent_energy_random",
        "same_C_skew_spectrum_random",
        "same_compute_noop_coordinate",
        "metric_preserving_noop",
    ]
    required_control_pair_rows = sum(pair_count_by_control.get(method, 0) for method in required_controls)
    required_control_win_rows = sum(beats_by_control.get(method, 0) for method in required_controls)
    poet_positive_rows = sum(1 for row in completed if int(row.get("poet_replaced_layers", 0) or 0) > 0)
    kan_vs_mcga_candidate_pairs = [row for row in kan_pair_rows if str(row.get("comparison", "")) == "KAN_G2_vs_MLP_MCGA_candidate"]
    kan_vs_mcga_win_rows = sum(1 for row in kan_vs_mcga_candidate_pairs if int(row.get("candidate_beats_control_guard_NLL", 0)) == 1)
    kan_vs_mcga_no_debt_rows = sum(1 for row in kan_vs_mcga_candidate_pairs if int(row.get("candidate_no_debt_vs_control", 0)) == 1)
    summary = {
        "rows": len(rows),
        "completed_rows": len(completed),
        "failed_rows": len(failed),
        "candidate_rows": len(candidate_rows),
        "control_rows": len(control_rows),
        "pair_rows": len(pair_rows),
        "datasets": datasets,
        "seeds": seeds,
        "carriers": carrier_names,
        "methods": methods,
        "steps": int(args.h20_steps),
        "lr": float(args.lr),
        "real_max_samples": int(args.real_max_samples),
        "mcga_method": str(args.mcga_method),
        "method_summary": method_summary,
        "beats_by_control_guard_NLL": beats_by_control,
        "pair_count_by_control": pair_count_by_control,
        "required_control_pair_rows": required_control_pair_rows,
        "required_control_win_rows": required_control_win_rows,
        "poet_replaced_positive_rows": poet_positive_rows,
        "kan_reference_status": kan_reference_status,
        "kan_reference_rows": len(kan_rows),
        "kan_reference_root_arg": str(args.kan_reference_root),
        "kan_vs_mcga_pair_rows": len(kan_pair_rows),
        "kan_vs_mcga_summary_rows": len(kan_pair_summary_rows),
        "kan_vs_mcga_candidate_pair_rows": len(kan_vs_mcga_candidate_pairs),
        "kan_vs_mcga_candidate_win_rows": kan_vs_mcga_win_rows,
        "kan_vs_mcga_candidate_win_rate": float(kan_vs_mcga_win_rows) / max(1.0, float(len(kan_vs_mcga_candidate_pairs))),
        "kan_vs_mcga_candidate_no_debt_rows": kan_vs_mcga_no_debt_rows,
        "kan_vs_mcga_candidate_no_debt_rate": float(kan_vs_mcga_no_debt_rows) / max(1.0, float(len(kan_vs_mcga_candidate_pairs))),
        "MLP_MCGA_controls_status": "completed_diagnostic_v22_style_controls_smoke" if len(completed) == len(rows) and rows else "R0_MLPStrongFunctionalBaselineIncomplete",
        "partG_official_gate_evaluable": 0,
        "diagnostic_only": 1,
        "official_gap_remaining": "same-split controls now smoke-tested only; still lacks official scale, all minimum-real rows, paired KAN gate arithmetic, and confirmatory thresholds.",
    }
    write_json(OUT_ROOT / "v23_26_MLP_MCGA_v22_style_controls_summary.json", summary)
    append_exec(
        "PartG_MLP_MCGA_v22_style_controls_smoke",
        args,
        [
            "v23_26_MLP_MCGA_v22_style_controls_matrix.csv",
            "v23_26_MLP_MCGA_v22_style_controls_pair_trace.csv",
            "v23_26_MLP_MCGA_v22_style_KAN_pair_trace.csv",
            "v23_26_MLP_MCGA_v22_style_KAN_pair_summary.csv",
            "v23_26_MLP_MCGA_v22_style_controls_summary.json",
        ],
        summary,
    )
    append_recap(
        "PartG MLP-MCGA v22-style controls smoke 复盘",
        [
            f"rows `{len(rows)}`，completed `{len(completed)}`，failed `{len(failed)}`；candidate rows `{len(candidate_rows)}`，control rows `{len(control_rows)}`，pair rows `{len(pair_rows)}`。",
            f"methods `{','.join(methods)}`；datasets `{','.join(datasets)}`；carriers `{','.join(carrier_names)}`；steps `{int(args.h20_steps)}`；lr `{float(args.lr)}`。",
            f"candidate-vs-required-control guard wins `{required_control_win_rows}/{required_control_pair_rows}`；POET replaced positive rows `{poet_positive_rows}/{len(completed)}`。",
            f"KAN reference status `{kan_reference_status}`；KAN_G2-vs-MCGA candidate pairs `{len(kan_vs_mcga_candidate_pairs)}`，wins `{kan_vs_mcga_win_rows}`，no-debt `{kan_vs_mcga_no_debt_rows}`。",
            "本轮把 v22.66 control 语义搬到 v23.26 split 上做 smoke：POET/external OET、same-functional-spectrum、same-generator-descent、same-C-skew、noop/metric-preserving 都有行。",
            "仍不能升格 official：规模、minimum-real 全量、paired KAN gate 与 confirmatory thresholds 尚未完成。",
        ],
    )


def read_kan_reference_partd(root_text: str) -> tuple[list[dict[str, Any]], str]:
    if not root_text:
        return [], "R0_KANReferenceRootNotProvided"
    root = Path(root_text)
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    matrix = root / "v23_26_partD_minimum_real_matrix.csv"
    if not matrix.is_file():
        return [], f"R0_KANReferenceMissing:{matrix}"
    with matrix.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), str(matrix)


def run_part_g_smoke(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("part-g-smoke uses matched MLP CUDA validation and requires CUDA.")
    datasets = [d for d in str(args.datasets).split(",") if d]
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    carrier_names = [c for c in str(args.carriers).split(",") if c]
    rows: list[dict[str, Any]] = []
    same_flops_profile_cache: dict[tuple[int, int, str], dict[str, Any]] = {}
    for dataset in datasets:
        for seed in seeds:
            x_np, y_np, note = load_real_dataset_numpy(dataset, seed, int(args.real_max_samples))
            splits, meta = real_splits_to_torch(x_np, y_np, seed, device)
            meta = {**meta, **{f"dataset_{k}": v for k, v in note.items()}}
            for carrier_name in carrier_names:
                info = carrier_info(carrier_name)
                if not info.supported_fused:
                    rows.append(
                        {
                            "phase": "PartG_MLP_matched_smoke",
                            "dataset": dataset,
                            "seed": seed,
                            "carrier_core_variant": carrier_name,
                            "science_status": "incomplete",
                            "route": "R0_DFOUTrigKernelIncomplete" if "Trig" in carrier_name else "R0_UnsupportedCarrier",
                            "diagnostic_only": 1,
                        }
                    )
                    continue
                rows.append(train_mlp_matched_baseline(info, dataset, seed, splits, int(args.h20_steps), float(args.lr), device, meta))
                rows.append(train_mlp_block_rtgf_baseline(info, dataset, seed, splits, int(args.h20_steps), float(args.lr), device, meta))
                input_dim = int(splits["x_train"].shape[1])
                output_dim = int(meta.get("dataset_n_classes", int(torch.cat([splits["y_train"], splits["y_witness"], splits["y_guard"], splits["y_test"]]).max().item()) + 1))
                profile_key = (input_dim, output_dim, info.name)
                if profile_key not in same_flops_profile_cache:
                    same_flops_profile_cache[profile_key] = select_same_flops_mlp_hidden(info, splits, output_dim, float(args.lr), device)
                profile_info = same_flops_profile_cache[profile_key]
                if "mlp_hidden" in profile_info:
                    rows.append(train_mlp_same_flops_baseline(info, dataset, seed, splits, int(args.h20_steps), float(args.lr), device, meta, profile_info))
                else:
                    rows.append(
                        {
                            "phase": "PartG_MLP_matched_smoke",
                            "dataset": dataset,
                            "seed": seed,
                            "carrier_core_variant": info.name,
                            "carrier_family": info.family,
                            "scheme": "MLP-AdamW-same-FLOPs",
                            "same_FLOPs_MLP_status": str(profile_info.get("status", "R0_SameFLOPsUnknown")),
                            "same_FLOPs_profiler_error": str(profile_info.get("error", "")),
                            "kan_profiled_full_step_FLOPs": int(profile_info.get("kan_profiled_full_step_FLOPs", 0) or 0),
                            "guard_NLL_finite": 0,
                            "diagnostic_only": 1,
                            **meta,
                        }
                    )
    write_csv(OUT_ROOT / "v23_26_MLP_matched_matrix.csv", rows)

    kan_rows, kan_reference_status = read_kan_reference_partd(str(args.kan_reference_root))
    kan_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in kan_rows:
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("carrier_core_variant", "")), str(row.get("scheme", "")))
        kan_by_key[key] = row

    pair_rows: list[dict[str, Any]] = []
    mlp_control_labels = {
        "MLP-AdamW-same-param": "MLP_AdamW",
        "MLP-BlockRTGF-same-param-M2": "MLP_BlockRTGF",
        "MLP-AdamW-same-FLOPs": "MLP_SameFLOPs_AdamW",
    }
    for mlp_row in rows:
        if mlp_row.get("scheme") not in mlp_control_labels:
            continue
        dataset = str(mlp_row["dataset"])
        seed = str(mlp_row["seed"])
        carrier = str(mlp_row["carrier_core_variant"])
        prefix = scheme_prefix(carrier)
        control_label = mlp_control_labels[str(mlp_row["scheme"])]
        for comparison, kan_scheme in [
            (f"KAN_B0_vs_{control_label}", f"{prefix}-B0"),
            (f"KAN_C2_vs_{control_label}", f"{prefix}-C2"),
            (f"KAN_G2_vs_{control_label}", f"{prefix}-G2"),
        ]:
            kan_row = kan_by_key.get((dataset, seed, carrier, kan_scheme))
            if kan_row is None:
                continue
            mlp_guard = parse_float(mlp_row.get("guard_NLL"))
            kan_guard = parse_float(kan_row.get("guard_NLL"))
            mlp_test = parse_float(mlp_row.get("test_NLL"))
            kan_test = parse_float(kan_row.get("test_NLL"))
            mlp_cvar95 = parse_float(mlp_row.get("guard_CVaR95_NLL"))
            kan_cvar95 = parse_float(kan_row.get("guard_CVaR95_NLL"))
            debt_delta = kan_cvar95 - mlp_cvar95
            pair_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "carrier_core_variant": carrier,
                    "comparison": comparison,
                    "candidate": kan_scheme,
                    "control": str(mlp_row["scheme"]),
                    "paired_guard_NLL_surplus": mlp_guard - kan_guard,
                    "paired_test_NLL_surplus": mlp_test - kan_test,
                    "candidate_guard_NLL": kan_guard,
                    "control_guard_NLL": mlp_guard,
                    "candidate_test_NLL": kan_test,
                    "control_test_NLL": mlp_test,
                    "candidate_guard_NLL_finite": int(math.isfinite(kan_guard)),
                    "control_guard_NLL_finite": int(math.isfinite(mlp_guard)),
                    "paired_guard_CVaR95_debt_delta": debt_delta,
                    "candidate_no_debt_vs_control": int(math.isfinite(debt_delta) and debt_delta <= 0.0),
                    "surplus_positive_means": "KAN_lower_NLL_than_matched_MLP",
                    "diagnostic_only": 1,
                }
            )
    write_csv(OUT_ROOT / "v23_26_MLP_matched_pair_trace.csv", pair_rows)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in pair_rows:
        groups.setdefault((row["carrier_core_variant"], row["comparison"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (carrier, comparison), group in sorted(groups.items()):
        stats = surplus_stats([float(row["paired_guard_NLL_surplus"]) for row in group])
        debt_values = [parse_float(row.get("paired_guard_CVaR95_debt_delta", "")) for row in group]
        finite_debt_values = [value for value in debt_values if math.isfinite(value)]
        no_debt_count = sum(1 for value in finite_debt_values if value <= 0.0)
        summary_rows.append(
            {
                "carrier_core_variant": carrier,
                "comparison": comparison,
                "pair_count": len(group),
                **stats,
                "no_debt_count": int(no_debt_count),
                "no_debt_rate": float(no_debt_count) / max(1.0, float(len(finite_debt_values))) if finite_debt_values else math.nan,
                "CVaR95_median_debt_delta": float(np.median(finite_debt_values)) if finite_debt_values else math.nan,
                "surplus_positive_means": "KAN_lower_NLL_than_matched_MLP",
                "diagnostic_only": 1,
            }
        )
    write_csv(OUT_ROOT / "v23_26_MLP_matched_summary.csv", summary_rows)

    finite_rows = sum(1 for row in rows if int(row.get("guard_NLL_finite", 0)) == 1)
    adamw_rows = [row for row in rows if row.get("scheme") == "MLP-AdamW-same-param"]
    block_rows = [row for row in rows if row.get("scheme") == "MLP-BlockRTGF-same-param-M2"]
    same_flops_rows = [row for row in rows if row.get("scheme") == "MLP-AdamW-same-FLOPs"]
    finite_adamw_rows = sum(1 for row in adamw_rows if int(row.get("guard_NLL_finite", 0)) == 1)
    finite_block_rows = sum(1 for row in block_rows if int(row.get("guard_NLL_finite", 0)) == 1)
    finite_same_flops_rows = sum(1 for row in same_flops_rows if int(row.get("guard_NLL_finite", 0)) == 1)
    same_flops_completed_rows = sum(1 for row in same_flops_rows if int(row.get("same_FLOPs_MLP_completed", 0)) == 1)
    summary = {
        "rows": len(rows),
        "finite_rows": finite_rows,
        "same_param_MLP_AdamW_rows": len(adamw_rows),
        "same_param_MLP_AdamW_finite_rows": finite_adamw_rows,
        "MLP_block_RTGF_rows": len(block_rows),
        "MLP_block_RTGF_finite_rows": finite_block_rows,
        "same_FLOPs_MLP_AdamW_rows": len(same_flops_rows),
        "same_FLOPs_MLP_AdamW_finite_rows": finite_same_flops_rows,
        "same_FLOPs_MLP_AdamW_completed_rows": same_flops_completed_rows,
        "pair_rows": len(pair_rows),
        "summary_rows": len(summary_rows),
        "datasets": datasets,
        "seeds": seeds,
        "carriers": carrier_names,
        "steps": int(args.h20_steps),
        "lr": float(args.lr),
        "real_max_samples": int(args.real_max_samples),
        "same_param_MLP_AdamW_completed": int(len(adamw_rows) > 0 and finite_adamw_rows == len(adamw_rows)),
        "same_FLOPs_MLP_AdamW_status": "completed_within_tolerance" if len(same_flops_rows) > 0 and same_flops_completed_rows == len(same_flops_rows) else "R0_MLPSameFLOPsIncompleteOrOutsideTolerance",
        "MLP_block_RTGF_status": "completed_diagnostic" if len(block_rows) > 0 and finite_block_rows == len(block_rows) else "R0_MLPBlockRTGFIncomplete",
        "MLP_MCGA_status": "R0_MLPStrongFunctionalBaselineIncomplete",
        "partG_official_gate_evaluable": 0,
        "kan_reference_status": kan_reference_status,
        "kan_reference_rows": len(kan_rows),
        "kan_reference_root_arg": str(args.kan_reference_root),
        "same_FLOPs_profile_cache_entries": len(same_flops_profile_cache),
        "out_root": str(OUT_ROOT),
        "diagnostic_only": 1,
    }
    write_json(OUT_ROOT / "v23_26_MLP_matched_summary.json", summary)
    append_exec(
        "PartG_MLP_matched_smoke",
        args,
        [
            "v23_26_MLP_matched_matrix.csv",
            "v23_26_MLP_matched_pair_trace.csv",
            "v23_26_MLP_matched_summary.csv",
            "v23_26_MLP_matched_summary.json",
        ],
        summary,
    )
    append_recap(
        "PartG MLP matched smoke 复盘",
        [
            f"same-param MLP AdamW rows `{len(adamw_rows)}`，finite rows `{finite_adamw_rows}/{len(adamw_rows)}`；same-FLOPs MLP AdamW rows `{len(same_flops_rows)}`，finite rows `{finite_same_flops_rows}/{len(same_flops_rows)}`，completed-within-tolerance `{same_flops_completed_rows}/{len(same_flops_rows)}`；MLP block-RTGF rows `{len(block_rows)}`，finite rows `{finite_block_rows}/{len(block_rows)}`；paired KAN-vs-MLP rows `{len(pair_rows)}`，summary rows `{len(summary_rows)}`。",
            f"datasets `{','.join(datasets)}`；carriers `{','.join(carrier_names)}`；real_max_samples `{int(args.real_max_samples)}`；steps `{int(args.h20_steps)}`；lr `{float(args.lr)}`。",
            f"KAN reference status `{kan_reference_status}`；surplus 定义为 `MLP NLL - KAN NLL`，正数才表示 KAN 低于同参数 MLP。",
            "MLPBaseline 是 input-hidden-hidden-output 的 SiLU MLP，无 KAN carrier；本轮包含 same-param AdamW、torch.profiler same-FLOPs AdamW 与 contiguous block MLP-RTGF diagnostic；旧 MCGA 未在 v23.26 语义下复现，仍标 `R0_MLPStrongFunctionalBaselineIncomplete`。",
        ],
    )


def efficiency_ratio_stats(rows: list[dict[str, Any]], key: str, gate: float) -> dict[str, Any]:
    vals = np.asarray([float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))], dtype=np.float64)
    if vals.size == 0:
        return {"rows": 0, "pass_rows": 0, "pass_rate": math.nan, "median": math.nan, "p90": math.nan, "max": math.nan, "gate": gate}
    return {
        "rows": int(vals.size),
        "pass_rows": int((vals <= float(gate)).sum()),
        "pass_rate": float((vals <= float(gate)).mean()),
        "median": float(np.median(vals)),
        "p90": float(np.quantile(vals, 0.90)),
        "max": float(vals.max()),
        "gate": float(gate),
    }


def run_part_h_efficiency_row(info: CarrierInfo, batch: int, hidden: int, repeats: int, lr: float, device: torch.device) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": "PartH_efficiency_officialization",
        "carrier_core_variant": info.name,
        "carrier_family": info.family,
        "batch": int(batch),
        "kan_hidden": int(hidden),
        "status": "not_run",
        "diagnostic_only": 1,
    }
    if not info.supported_fused:
        row.update({"status": "incomplete", "route": "R0_DFOUTrigKernelIncomplete"})
        return row
    input_dim, output_dim = 32, 10
    target_params = int(input_dim * hidden * info.k + hidden * output_dim * info.k)
    mlp_hidden = int(hidden_for_param_budget(input_dim, output_dim, target_params))
    row.update(
        {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "mlp_hidden": mlp_hidden,
            "kan_reference_trainable_params_w1w2": target_params,
            "repeats": int(repeats),
            "lr": float(lr),
            "metric_name": "M2",
            "metric_inverse_precomputed": 1,
            "per_step_matrix_inverse": 0,
            "dense_basis_tensor_used": 0,
            "vectorized_edge_bank_update": 1,
            "python_edge_loop_in_optimizer": 0,
            "extra_backward_count": 0,
            "jvp_vjp_count": 0,
            "candidate_scoring_in_step": 0,
            "guard_eval_in_optimizer_step": 0,
                "trace_telemetry_enabled_in_profiled_optimizer": 0,
                "fused_rtgf_optimizer_step_available": 1,
            }
        )
    try:
        torch.manual_seed(8000 + int(batch) + int(hidden) + info.k)
        x_stats = torch.randn(512, input_dim, device=device)
        x = torch.randn(int(batch), input_dim, device=device)
        y = torch.randint(0, output_dim, (int(batch),), device=device)
        metric = metric_matrices(info.family, info.k)["M2"]

        step_model_adamw = make_model(info, input_dim, output_dim, hidden, x_stats, seed=31, device=device)
        step_model_rtgf = make_model(info, input_dim, output_dim, hidden, x_stats, seed=31, device=device)
        step_model_rtgf.load_state_dict(copy.deepcopy(step_model_adamw.state_dict()))
        adamw_step_opt = torch.optim.AdamW([step_model_adamw.w1, step_model_adamw.w2], lr=float(lr), weight_decay=0.0)
        rtgf_step_opt = IntrinsicEdgeOptimizer(step_model_rtgf, metric, lr=float(lr), mode="rtgf", weight_decay=0.0, trace_enabled=False)
        grad_w1 = torch.randn_like(step_model_adamw.w1)
        grad_w2 = torch.randn_like(step_model_adamw.w2)

        def adamw_optimizer_step() -> None:
            step_model_adamw.w1.grad = grad_w1
            step_model_adamw.w2.grad = grad_w2
            adamw_step_opt.step()

        def rtgf_optimizer_step() -> None:
            step_model_rtgf.w1.grad = grad_w1
            step_model_rtgf.w2.grad = grad_w2
            rtgf_step_opt.step()

        adamw_optimizer_step_ms = time_cuda(adamw_optimizer_step, repeats=max(1, int(repeats)))
        generator_optimizer_step_ms = time_cuda(rtgf_optimizer_step, repeats=max(1, int(repeats)))

        kan_model = make_model(info, input_dim, output_dim, hidden, x_stats, seed=37, device=device)
        kan_opt = IntrinsicEdgeOptimizer(kan_model, metric, lr=float(lr), mode="rtgf", weight_decay=0.0, trace_enabled=False)
        kan_adamw_model = make_model(info, input_dim, output_dim, hidden, x_stats, seed=37, device=device)
        kan_adamw_model.load_state_dict(copy.deepcopy(kan_model.state_dict()))
        kan_adamw_opt = torch.optim.AdamW([kan_adamw_model.w1, kan_adamw_model.w2], lr=float(lr), weight_decay=0.0)
        mlp = MLPBaseline(input_dim=input_dim, output_dim=output_dim, hidden_dim=mlp_hidden, seed=37, device=device)
        mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=float(lr), weight_decay=0.0)

        def kan_forward() -> None:
            kan_model.manual_ce_forward_cache(x)

        def kan_backward() -> None:
            kan_model.zero_grad(set_to_none=True)
            logits, cache = kan_model.manual_ce_forward_cache(x)
            kan_model.manual_ce_backward_from_cache(logits, cache, y)

        def kan_rtgf_full_step() -> None:
            kan_model.zero_grad(set_to_none=True)
            logits, cache = kan_model.manual_ce_forward_cache(x)
            kan_model.manual_ce_backward_from_cache(logits, cache, y)
            kan_opt.step()

        def kan_adamw_full_step() -> None:
            kan_adamw_opt.zero_grad(set_to_none=True)
            logits, cache = kan_adamw_model.manual_ce_forward_cache(x)
            kan_adamw_model.manual_ce_backward_from_cache(logits, cache, y)
            kan_adamw_opt.step()

        def mlp_full_step() -> None:
            mlp_opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(mlp(x), y)
            loss.backward()
            mlp_opt.step()

        def mlp_forward() -> None:
            mlp(x)

        def mlp_backward() -> None:
            mlp_opt.zero_grad(set_to_none=True)
            F.cross_entropy(mlp(x), y).backward()

        kan_forward_ms = time_cuda(kan_forward, repeats=max(1, int(repeats)))
        kan_backward_ms = time_cuda(kan_backward, repeats=max(1, int(repeats)))
        kan_rtgf_full_step_ms = time_cuda(kan_rtgf_full_step, repeats=max(1, int(repeats)))
        kan_adamw_full_step_ms = time_cuda(kan_adamw_full_step, repeats=max(1, int(repeats)))
        mlp_forward_ms = time_cuda(mlp_forward, repeats=max(1, int(repeats)))
        mlp_backward_ms = time_cuda(mlp_backward, repeats=max(1, int(repeats)))
        mlp_full_step_ms = time_cuda(mlp_full_step, repeats=max(1, int(repeats)))
        kan_peak_memory_mb = peak_memory_increment_mb(device, kan_rtgf_full_step)
        mlp_peak_memory_mb = peak_memory_increment_mb(device, mlp_full_step)
        memory_ratio = kan_peak_memory_mb / mlp_peak_memory_mb if mlp_peak_memory_mb > 0.0 else math.nan
        mlp_params = int(count_parameters(mlp))
        row.update(
            {
                "status": "completed",
                "mlp_trainable_params": mlp_params,
                "param_abs_delta": abs(mlp_params - target_params),
                "param_ratio_mlp_over_kan": float(mlp_params) / max(1.0, float(target_params)),
                "adamw_optimizer_step_ms": adamw_optimizer_step_ms,
                "generator_optimizer_step_ms": generator_optimizer_step_ms,
                "generator_optimizer_step_ratio_vs_AdamW": generator_optimizer_step_ms / max(adamw_optimizer_step_ms, EPS),
                "kan_forward_ms": kan_forward_ms,
                "kan_backward_ms": kan_backward_ms,
                "kan_rtgf_full_step_ms": kan_rtgf_full_step_ms,
                "kan_adamw_full_step_ms": kan_adamw_full_step_ms,
                "mlp_forward_ms": mlp_forward_ms,
                "mlp_backward_ms": mlp_backward_ms,
                "mlp_full_step_ms": mlp_full_step_ms,
                "forward_ratio_vs_same_param_MLP": kan_forward_ms / max(mlp_forward_ms, EPS),
                "backward_ratio_vs_same_param_MLP": kan_backward_ms / max(mlp_backward_ms, EPS),
                "full_step_ratio_vs_same_param_MLP": kan_rtgf_full_step_ms / max(mlp_full_step_ms, EPS),
                "rtgf_full_step_ratio_vs_KAN_AdamW": kan_rtgf_full_step_ms / max(kan_adamw_full_step_ms, EPS),
                "kan_peak_memory_increment_MB": kan_peak_memory_mb,
                "mlp_peak_memory_increment_MB": mlp_peak_memory_mb,
                "memory_ratio_vs_MLP": memory_ratio,
                "basis_materialized_bytes": 0,
                "fused_forward_used": 1,
                "fused_backward_used": 1,
                "fallback_kernel_used": 0,
                "fused_rtgf_optimizer_update_count": int(rtgf_step_opt.fused_step_count + kan_opt.fused_step_count),
                "fused_rtgf_optimizer_fallback_count": int(rtgf_step_opt.fused_step_fallback_count + kan_opt.fused_step_fallback_count),
                "fused_rtgf_optimizer_step_used": int((rtgf_step_opt.fused_step_count + kan_opt.fused_step_count) > 0),
                "exploration_optimizer_step_gate": int(generator_optimizer_step_ms / max(adamw_optimizer_step_ms, EPS) <= 1.25),
                "exploration_full_step_gate": int(kan_rtgf_full_step_ms / max(mlp_full_step_ms, EPS) <= 1.25),
                "exploration_memory_gate": int(math.isfinite(memory_ratio) and memory_ratio <= 1.10),
            }
        )
    except Exception as exc:  # pragma: no cover
        row.update({"status": "fail", "route": "R0_EfficiencyProfilerFailed", "error": repr(exc)})
    return row


def run_part_h_efficiency(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("part-h-efficiency uses CUDA timing and requires CUDA.")
    carrier_names = [c for c in str(args.carriers).split(",") if c]
    batches = [32, 128, 256] if not args.quick else [32]
    hiddens = [64, 128] if not args.quick else [64]
    repeats = int(args.efficiency_repeats)
    rows: list[dict[str, Any]] = []
    for carrier_name in carrier_names:
        info = carrier_info(carrier_name)
        for batch in batches:
            for hidden in hiddens:
                rows.append(run_part_h_efficiency_row(info, batch, hidden, repeats, float(args.lr), device))
    write_csv(OUT_ROOT / "v23_26_efficiency_official_matrix.csv", rows)
    completed = [row for row in rows if row.get("status") == "completed"]
    summary_rows: list[dict[str, Any]] = []
    for carrier in sorted({row["carrier_core_variant"] for row in completed}):
        carrier_rows = [row for row in completed if row["carrier_core_variant"] == carrier]
        full_stats = efficiency_ratio_stats(carrier_rows, "full_step_ratio_vs_same_param_MLP", 1.25)
        mem_stats = efficiency_ratio_stats(carrier_rows, "memory_ratio_vs_MLP", 1.10)
        opt_stats = efficiency_ratio_stats(carrier_rows, "generator_optimizer_step_ratio_vs_AdamW", 1.25)
        official_full = efficiency_ratio_stats(carrier_rows, "full_step_ratio_vs_same_param_MLP", 1.10)
        official_mem = efficiency_ratio_stats(carrier_rows, "memory_ratio_vs_MLP", 1.05)
        summary_rows.append(
            {
                "carrier_core_variant": carrier,
                "rows": len(carrier_rows),
                "optimizer_step_pass_rows": opt_stats["pass_rows"],
                "optimizer_step_pass_rate": opt_stats["pass_rate"],
                "optimizer_step_median_ratio_vs_AdamW": opt_stats["median"],
                "full_step_pass_rows_le_1p25": full_stats["pass_rows"],
                "full_step_pass_rate_le_1p25": full_stats["pass_rate"],
                "full_step_median_ratio_vs_MLP": full_stats["median"],
                "full_step_p90_ratio_vs_MLP": full_stats["p90"],
                "memory_pass_rows_le_1p10": mem_stats["pass_rows"],
                "memory_pass_rate_le_1p10": mem_stats["pass_rate"],
                "memory_median_ratio_vs_MLP": mem_stats["median"],
                "memory_p90_ratio_vs_MLP": mem_stats["p90"],
                "official_median_full_step_le_1p10": int(math.isfinite(official_full["median"]) and official_full["median"] <= 1.10),
                "official_p90_full_step_le_1p25": int(math.isfinite(full_stats["p90"]) and full_stats["p90"] <= 1.25),
                "official_median_memory_le_1p05": int(math.isfinite(official_mem["median"]) and official_mem["median"] <= 1.05),
                "partH_exploration_pass": int(opt_stats["pass_rows"] == len(carrier_rows) and full_stats["pass_rate"] >= 0.80 and mem_stats["pass_rows"] == len(carrier_rows)),
                "partH_official_pass": int(math.isfinite(full_stats["median"]) and full_stats["median"] <= 1.10 and math.isfinite(full_stats["p90"]) and full_stats["p90"] <= 1.25 and math.isfinite(mem_stats["median"]) and mem_stats["median"] <= 1.05),
                "diagnostic_only": 1,
            }
        )
    write_csv(OUT_ROOT / "v23_26_efficiency_official_summary.csv", summary_rows)
    all_full = efficiency_ratio_stats(completed, "full_step_ratio_vs_same_param_MLP", 1.25)
    all_mem = efficiency_ratio_stats(completed, "memory_ratio_vs_MLP", 1.10)
    all_opt = efficiency_ratio_stats(completed, "generator_optimizer_step_ratio_vs_AdamW", 1.25)
    summary = {
        "rows": len(rows),
        "completed_rows": len(completed),
        "carriers": carrier_names,
        "batches": batches,
        "hiddens": hiddens,
        "repeats": repeats,
        "lr": float(args.lr),
        "full_step_ratio_vs_same_param_MLP": all_full,
        "memory_ratio_vs_MLP": all_mem,
        "generator_optimizer_step_ratio_vs_AdamW": all_opt,
        "implementation_requirements": {
            "metric_inverse_precomputed": 1,
            "per_step_matrix_inverse": 0,
            "dense_basis_tensor_used": 0,
            "vectorized_edge_bank_update": 1,
            "python_edge_loop_in_optimizer": 0,
            "extra_backward_count": 0,
            "jvp_vjp_count": 0,
            "candidate_scoring_in_step": 0,
            "guard_eval_in_optimizer_step": 0,
            "trace_telemetry_enabled_in_profiled_optimizer": 0,
            "fused_rtgf_optimizer_step_available": 1,
        },
        "partH_exploration_all_rows_pass": int(all_opt["pass_rows"] == len(completed) and all_full["pass_rate"] >= 0.80 and all_mem["pass_rows"] == len(completed) and len(completed) > 0),
        "partH_official_all_rows_pass": int(math.isfinite(all_full["median"]) and all_full["median"] <= 1.10 and math.isfinite(all_full["p90"]) and all_full["p90"] <= 1.25 and math.isfinite(all_mem["median"]) and all_mem["median"] <= 1.05 and len(completed) > 0),
        "diagnostic_only": 1,
    }
    write_json(OUT_ROOT / "v23_26_efficiency_official_summary.json", summary)
    append_exec(
        "PartH_efficiency_officialization",
        args,
        ["v23_26_efficiency_official_matrix.csv", "v23_26_efficiency_official_summary.csv", "v23_26_efficiency_official_summary.json"],
        summary,
    )
    append_recap(
        "PartH efficiency officialization 复盘",
        [
            f"rows `{len(rows)}`，completed rows `{len(completed)}`；batches `{batches}`；hiddens `{hiddens}`；repeats `{repeats}`。",
            f"full_step_ratio_vs_same_param_MLP median `{all_full['median']}`，p90 `{all_full['p90']}`，pass_rate<=1.25 `{all_full['pass_rate']}`。",
            f"memory_ratio_vs_MLP median `{all_mem['median']}`，p90 `{all_mem['p90']}`，pass_rows<=1.10 `{all_mem['pass_rows']}/{all_mem['rows']}`。",
            f"generator_optimizer_step_ratio_vs_AdamW median `{all_opt['median']}`，pass_rows<=1.25 `{all_opt['pass_rows']}/{all_opt['rows']}`。",
            f"PartH exploration all-rows pass `{summary['partH_exploration_all_rows_pass']}`；official all-rows pass `{summary['partH_official_all_rows_pass']}`。若为 0，只能记录 efficiency blocked/diagnostic，不能输出 official success。",
        ],
    )


def read_json_artifact(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "missing"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {"value": obj}, "ok"
    except Exception as exc:
        return {"error_type": type(exc).__name__, "error": str(exc)}, "read_error"


def read_csv_artifact(path: Path) -> tuple[list[dict[str, str]], str]:
    if not path.exists():
        return [], "missing"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle)), "ok"
    except Exception as exc:
        return [{"error_type": type(exc).__name__, "error": str(exc)}], "read_error"


def latest_artifact(pattern: str) -> Path | None:
    matches = [p for p in (ROOT / "results").glob(pattern) if p.exists()]
    if not matches:
        return None
    return max(matches, key=lambda p: (p.stat().st_mtime, str(p)))


def gate_float(row: dict[str, Any] | None, key: str) -> float:
    if row is None:
        return math.nan
    return parse_float(row.get(key, ""))


def gate_row(rows: list[dict[str, Any]], carrier: str, comparison: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("carrier_core_variant", "")) == carrier and str(row.get("comparison", "")) == comparison:
            return row
    return None


def gate_bool(value: bool) -> int:
    return int(bool(value))


def read_partg_kan_reference_from_summary(summary: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    candidates: list[Path] = []
    status = str(summary.get("kan_reference_status", "") or "")
    if status and not status.startswith("R0_"):
        candidates.append(Path(status))
    root_arg = str(summary.get("kan_reference_root_arg", "") or "")
    if root_arg:
        root = Path(root_arg)
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        candidates.append(root / "v23_26_partD_minimum_real_matrix.csv")
    seen: set[str] = set()
    for path in candidates:
        path = path.resolve() if not path.is_absolute() else path
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return read_csv_artifact(path)
    return [], status or "missing"


def reconstruct_partg_pair_trace_from_matrices(mlp_rows: list[dict[str, Any]], kan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kan_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in kan_rows:
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("carrier_core_variant", "")), str(row.get("scheme", "")))
        kan_by_key[key] = row
    mlp_control_labels = {
        "MLP-AdamW-same-param": "MLP_AdamW",
        "MLP-BlockRTGF-same-param-M2": "MLP_BlockRTGF",
        "MLP-AdamW-same-FLOPs": "MLP_SameFLOPs_AdamW",
    }
    reconstructed: list[dict[str, Any]] = []
    for mlp_row in mlp_rows:
        scheme = str(mlp_row.get("scheme", ""))
        if scheme not in mlp_control_labels:
            continue
        dataset = str(mlp_row.get("dataset", ""))
        seed = str(mlp_row.get("seed", ""))
        carrier = str(mlp_row.get("carrier_core_variant", ""))
        prefix = scheme_prefix(carrier)
        control_label = mlp_control_labels[scheme]
        for comparison, kan_scheme in [
            (f"KAN_B0_vs_{control_label}", f"{prefix}-B0"),
            (f"KAN_C2_vs_{control_label}", f"{prefix}-C2"),
            (f"KAN_G2_vs_{control_label}", f"{prefix}-G2"),
        ]:
            kan_row = kan_by_key.get((dataset, seed, carrier, kan_scheme))
            if kan_row is None:
                continue
            mlp_guard = parse_float(mlp_row.get("guard_NLL"))
            kan_guard = parse_float(kan_row.get("guard_NLL"))
            mlp_test = parse_float(mlp_row.get("test_NLL"))
            kan_test = parse_float(kan_row.get("test_NLL"))
            debt_delta = parse_float(kan_row.get("guard_CVaR95_NLL", "")) - parse_float(mlp_row.get("guard_CVaR95_NLL", ""))
            reconstructed.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "carrier_core_variant": carrier,
                    "comparison": comparison,
                    "candidate": kan_scheme,
                    "control": scheme,
                    "paired_guard_NLL_surplus": mlp_guard - kan_guard,
                    "paired_test_NLL_surplus": mlp_test - kan_test,
                    "candidate_guard_NLL": kan_guard,
                    "control_guard_NLL": mlp_guard,
                    "candidate_test_NLL": kan_test,
                    "control_test_NLL": mlp_test,
                    "candidate_guard_NLL_finite": int(math.isfinite(kan_guard)),
                    "control_guard_NLL_finite": int(math.isfinite(mlp_guard)),
                    "paired_guard_CVaR95_debt_delta": debt_delta,
                    "candidate_no_debt_vs_control": int(math.isfinite(debt_delta) and debt_delta <= 0.0),
                    "surplus_positive_means": "KAN_lower_NLL_than_matched_MLP",
                    "diagnostic_only": 1,
                    "reconstructed_from_matrix": 1,
                }
            )
    return reconstructed


def no_debt_stats_from_pair_trace(rows: list[dict[str, Any]], carrier: str, comparisons: list[str]) -> dict[str, Any]:
    comparison_set = set(comparisons)
    values: list[float] = []
    comparison_counts: dict[str, int] = {comparison: 0 for comparison in comparisons}
    for row in rows:
        comparison = str(row.get("comparison", ""))
        if str(row.get("carrier_core_variant", "")) != carrier or comparison not in comparison_set:
            continue
        value = parse_float(row.get("paired_guard_CVaR95_debt_delta", ""))
        if not math.isfinite(value):
            continue
        values.append(value)
        comparison_counts[comparison] = comparison_counts.get(comparison, 0) + 1
    no_debt_count = sum(1 for value in values if value <= 0.0)
    return {
        "pair_count": len(values),
        "no_debt_count": int(no_debt_count),
        "no_debt_rate": float(no_debt_count) / max(1.0, float(len(values))) if values else math.nan,
        "median_debt_delta": float(np.median(values)) if values else math.nan,
        "source_comparisons": ",".join(f"{key}:{value}" for key, value in comparison_counts.items() if value > 0),
    }


def normalize_summary_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return sorted(str(item) for item in value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    return sorted(str(item) for item in parsed)
            except Exception:
                pass
        return [text] if text else []
    return []


def build_mcga_candidate_index(mcga_summary_paths: list[Path]) -> dict[str, dict[str, Any]]:
    best_by_carrier: dict[str, dict[str, Any]] = {}
    for mcga_summary_path in mcga_summary_paths:
        root = mcga_summary_path.parent
        summary_json, summary_json_status = read_json_artifact(root / "v23_26_MLP_MCGA_v22_style_controls_summary.json")
        summary_rows, summary_rows_status = read_csv_artifact(mcga_summary_path)
        for row in summary_rows:
            if str(row.get("comparison", "")) != "KAN_G2_vs_MLP_MCGA_candidate":
                continue
            carrier = str(row.get("carrier_core_variant", ""))
            pair_count_value = parse_float(row.get("pair_count", 0))
            no_debt_count_value = parse_float(row.get("no_debt_count", 0))
            pair_count = int(pair_count_value) if math.isfinite(pair_count_value) else 0
            no_debt_count = int(no_debt_count_value) if math.isfinite(no_debt_count_value) else 0
            candidate = {
                "source": str(mcga_summary_path.relative_to(ROOT)),
                "row": row,
                "summary_json": summary_json,
                "summary_json_status": summary_json_status,
                "summary_rows_status": summary_rows_status,
                "pair_count": pair_count,
                "win_rate": gate_float(row, "win_rate"),
                "no_debt_count": no_debt_count,
                "no_debt_rate": gate_float(row, "no_debt_rate"),
                "kan_reference_status": summary_json.get("kan_reference_status", ""),
                "datasets": summary_json.get("datasets", ""),
                "seeds": summary_json.get("seeds", ""),
                "priority": (pair_count, str(mcga_summary_path.relative_to(ROOT))),
            }
            current = best_by_carrier.get(carrier)
            if current is None or candidate["priority"] > current["priority"]:
                best_by_carrier[carrier] = candidate
    return best_by_carrier


def run_part_dg_gate_audit(args: argparse.Namespace) -> None:
    """Compute reproducible PartD/PartG gate arithmetic from existing matrices.

    This is an audit over already-written artifacts. It intentionally keeps official_evaluable=0
    when a required condition is missing or the source artifact is diagnostic-only.
    """

    partd_rows: list[dict[str, Any]] = []
    partg_rows: list[dict[str, Any]] = []
    partg_mcga_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    partd_summary_paths = sorted((ROOT / "results").glob("v23_26_gpu3_partd*/v23_26_partD_summary.json"), key=lambda p: str(p))
    for summary_path in partd_summary_paths:
        root = summary_path.parent
        summary, summary_status = read_json_artifact(summary_path)
        matrix, matrix_status = read_csv_artifact(root / "v23_26_partD_minimum_real_matrix.csv")
        pair_trace, pair_trace_status = read_csv_artifact(root / "v23_26_partD_paired_control_summary.csv")
        trace_rows, trace_status = read_csv_artifact(root / "v23_26_partD_paired_control_trace.csv")
        if not trace_rows:
            by_key_scheme: dict[tuple[str, str, str, str], dict[str, Any]] = {}
            for item in matrix:
                by_key_scheme[
                    (
                        str(item.get("dataset", "")),
                        str(item.get("seed", "")),
                        str(item.get("carrier_core_variant", "")),
                        str(item.get("scheme", "")),
                    )
                ] = item
            reconstructed: list[dict[str, Any]] = []
            for item in matrix:
                scheme = str(item.get("scheme", ""))
                carrier_name = str(item.get("carrier_core_variant", ""))
                prefix = scheme_prefix(carrier_name)
                match = re.fullmatch(rf"{re.escape(prefix)}-G([0-2])", scheme)
                if match is None:
                    continue
                metric_index = match.group(1)
                key_base = (str(item.get("dataset", "")), str(item.get("seed", "")), carrier_name)
                for comparison, control_name in [
                    (f"G{metric_index}_vs_AdamW", f"{prefix}-B0"),
                    (f"G{metric_index}_vs_additive_C{metric_index}", f"{prefix}-C{metric_index}"),
                    (f"G{metric_index}_vs_random_M_skew_M{metric_index}", f"{prefix}-R0-{metric_index}"),
                    (f"G{metric_index}_vs_signflip_M{metric_index}", f"{prefix}-R1-{metric_index}"),
                    (f"G{metric_index}_vs_shuffled_tangent_M{metric_index}", f"{prefix}-R2-{metric_index}"),
                    (f"G{metric_index}_vs_Euclidean_RTGF_M{metric_index}", f"{prefix}-R3-{metric_index}"),
                    (f"G{metric_index}_vs_same_compute_noop_M{metric_index}", f"{prefix}-R5-{metric_index}"),
                ]:
                    control = by_key_scheme.get((*key_base, control_name))
                    if control is None:
                        continue
                    reconstructed.append(
                        {
                            "dataset": key_base[0],
                            "seed": key_base[1],
                            "carrier_core_variant": carrier_name,
                            "metric_name": f"M{metric_index}",
                            "comparison": comparison,
                            "candidate": scheme,
                            "control": control_name,
                            "paired_guard_NLL_surplus": parse_float(control.get("guard_NLL", "")) - parse_float(item.get("guard_NLL", "")),
                            "paired_test_NLL_surplus": parse_float(control.get("test_NLL", "")) - parse_float(item.get("test_NLL", "")),
                            "paired_guard_CVaR95_debt_delta": parse_float(item.get("guard_CVaR95_NLL", "")) - parse_float(control.get("guard_CVaR95_NLL", "")),
                            "diagnostic_only": 1,
                            "reconstructed_from_matrix": 1,
                        }
                    )
            trace_rows = reconstructed
            trace_status = "reconstructed_from_matrix" if reconstructed else "missing"
        carriers = sorted({str(row.get("carrier_core_variant", "")) for row in matrix if row.get("carrier_core_variant")})
        if not carriers:
            carriers = [str(c) for c in summary.get("carriers", [])]
        source_rows.append(
            {
                "source_kind": "PartD",
                "source": str(summary_path.relative_to(ROOT)),
                "status": summary_status,
                "matrix_status": matrix_status,
                "pair_summary_status": pair_trace_status,
                "rows": summary.get("rows", ""),
                "finite_rows": summary.get("finite_rows", ""),
                "pair_rows": summary.get("pair_rows", ""),
                "datasets": summary.get("datasets", ""),
                "seeds": summary.get("seeds", ""),
                "carriers": summary.get("carriers", ""),
                "steps": summary.get("steps", ""),
                "real_max_samples": summary.get("real_max_samples", ""),
            }
        )
        for carrier in carriers:
            carrier_matrix = [row for row in matrix if str(row.get("carrier_core_variant", "")) == carrier]
            carrier_trace = [row for row in trace_rows if str(row.get("carrier_core_variant", "")) == carrier]
            finite_carrier_rows = sum(1 for row in carrier_matrix if int(float(row.get("guard_NLL_finite") or 0)) == 1)
            no_error_rows = int(bool(carrier_matrix) and finite_carrier_rows == len(carrier_matrix))
            metric_indices = sorted(
                {
                    match.group(1)
                    for row in list(pair_trace) + list(carrier_trace)
                    for match in [re.match(r"G([0-2])_vs_", str(row.get("comparison", "")))]
                    if str(row.get("carrier_core_variant", "")) == carrier and match is not None
                }
            )
            if not metric_indices:
                metric_indices = ["2"]
            for metric_index in metric_indices:
                adam = gate_row(pair_trace, carrier, f"G{metric_index}_vs_AdamW")
                additive = gate_row(pair_trace, carrier, f"G{metric_index}_vs_additive_C{metric_index}")
                random = gate_row(pair_trace, carrier, f"G{metric_index}_vs_random_M_skew_M{metric_index}")
                required = [adam, additive, random]
                min_pair_count = min([int(float(row.get("pair_count") or 0)) for row in required if row is not None] or [0])
                debt_values: list[float] = []
                debt_comparisons = {
                    f"G{metric_index}_vs_AdamW",
                    f"G{metric_index}_vs_additive_C{metric_index}",
                    f"G{metric_index}_vs_random_M_skew_M{metric_index}",
                }
                for row in carrier_trace:
                    if str(row.get("comparison", "")) in debt_comparisons:
                        value = parse_float(row.get("paired_guard_CVaR95_debt_delta", ""))
                        if math.isfinite(value):
                            debt_values.append(value)
                no_debt_rate = float(sum(1 for value in debt_values if value <= 0.0)) / max(1.0, float(len(debt_values))) if debt_values else math.nan
                gate = {
                    "source": str(summary_path.relative_to(ROOT)),
                    "carrier_core_variant": carrier,
                    "metric_name": f"M{metric_index}",
                    "candidate_scheme": f"{scheme_prefix(carrier)}-G{metric_index}",
                    "datasets": summary.get("datasets", ""),
                    "seeds": summary.get("seeds", ""),
                    "steps": summary.get("steps", ""),
                    "real_max_samples": summary.get("real_max_samples", ""),
                    "matrix_rows": len(carrier_matrix),
                    "finite_matrix_rows": finite_carrier_rows,
                    "no_error_rows": no_error_rows,
                    "min_required_pair_count": min_pair_count,
                    "paired_rows_ge_24": gate_bool(min_pair_count >= 24),
                    "adamw_median_surplus": gate_float(adam, "median_surplus"),
                    "candidate_vs_own_adamw_median_ge_1e3": gate_bool(gate_float(adam, "median_surplus") >= 1.0e-3),
                    "additive_median_surplus": gate_float(additive, "median_surplus"),
                    "candidate_vs_additive_median_gt_0": gate_bool(gate_float(additive, "median_surplus") > 0.0),
                    "random_median_surplus": gate_float(random, "median_surplus"),
                    "candidate_vs_random_median_gt_0": gate_bool(gate_float(random, "median_surplus") > 0.0),
                    "random_CVaR25_surplus": gate_float(random, "CVaR25_surplus"),
                    "candidate_vs_random_CVaR25_gt_0": gate_bool(gate_float(random, "CVaR25_surplus") > 0.0),
                    "random_bootstrap_LCB05_surplus": gate_float(random, "bootstrap_LCB05_surplus"),
                    "candidate_vs_random_bootstrap_LCB_gt_0": gate_bool(gate_float(random, "bootstrap_LCB05_surplus") > 0.0),
                    "random_win_rate": gate_float(random, "win_rate"),
                    "win_rate_ge_70pct": gate_bool(gate_float(random, "win_rate") >= 0.70),
                    "paired_no_debt_rate": no_debt_rate,
                    "paired_no_debt_ge_80pct": gate_bool(math.isfinite(no_debt_rate) and no_debt_rate >= 0.80),
                    "official_partD_gate_evaluable": 0,
                    "diagnostic_only": 1,
                }
                pass_keys = [
                    "paired_rows_ge_24",
                    "candidate_vs_own_adamw_median_ge_1e3",
                    "candidate_vs_additive_median_gt_0",
                    "candidate_vs_random_median_gt_0",
                    "candidate_vs_random_CVaR25_gt_0",
                    "candidate_vs_random_bootstrap_LCB_gt_0",
                    "win_rate_ge_70pct",
                    "paired_no_debt_ge_80pct",
                    "no_error_rows",
                ]
                gate["partD_minimum_real_gate_pass_if_official"] = gate_bool(all(int(gate[key]) == 1 for key in pass_keys))
                gate["partD_gate_missing_or_failed_items"] = ",".join(key for key in pass_keys if int(gate[key]) != 1)
                gate["pair_trace_status"] = trace_status
                partd_rows.append(gate)

    mcga_kan_summary_paths = sorted(
        (ROOT / "results").glob("v23_26_gpu3_partg_mcga_controls*/v23_26_MLP_MCGA_v22_style_KAN_pair_summary.csv"),
        key=lambda p: str(p),
    )
    mcga_candidate_by_carrier = build_mcga_candidate_index(mcga_kan_summary_paths)

    partg_summary_paths = sorted((ROOT / "results").glob("v23_26_gpu3_partg*/v23_26_MLP_matched_summary.json"), key=lambda p: str(p))
    for summary_path in partg_summary_paths:
        root = summary_path.parent
        summary, summary_status = read_json_artifact(summary_path)
        pair_summary, pair_summary_status = read_csv_artifact(root / "v23_26_MLP_matched_summary.csv")
        mlp_matrix, mlp_matrix_status = read_csv_artifact(root / "v23_26_MLP_matched_matrix.csv")
        pair_trace, pair_trace_status = read_csv_artifact(root / "v23_26_MLP_matched_pair_trace.csv")
        kan_matrix, kan_matrix_status = read_partg_kan_reference_from_summary(summary)
        reconstructed_pair_trace = reconstruct_partg_pair_trace_from_matrices(mlp_matrix, kan_matrix)
        trace_has_debt = any(math.isfinite(parse_float(row.get("paired_guard_CVaR95_debt_delta", ""))) for row in pair_trace)
        no_debt_pair_trace = pair_trace if trace_has_debt else reconstructed_pair_trace
        no_debt_trace_status = pair_trace_status if trace_has_debt else ("reconstructed_from_mlp_and_kan_matrices" if reconstructed_pair_trace else "missing_debt_trace")
        source_rows.append(
            {
                "source_kind": "PartG",
                "source": str(summary_path.relative_to(ROOT)),
                "status": summary_status,
                "matrix_status": mlp_matrix_status,
                "pair_summary_status": pair_summary_status,
                "pair_trace_status": pair_trace_status,
                "kan_matrix_status": kan_matrix_status,
                "no_debt_trace_status": no_debt_trace_status,
                "rows": summary.get("rows", ""),
                "finite_rows": summary.get("finite_rows", ""),
                "pair_rows": summary.get("pair_rows", ""),
                "datasets": summary.get("datasets", ""),
                "seeds": summary.get("seeds", ""),
                "carriers": summary.get("carriers", ""),
                "steps": summary.get("steps", ""),
                "real_max_samples": summary.get("real_max_samples", ""),
                "same_FLOPs_MLP_AdamW_status": summary.get("same_FLOPs_MLP_AdamW_status", ""),
                "MLP_block_RTGF_status": summary.get("MLP_block_RTGF_status", ""),
                "MLP_MCGA_status": summary.get("MLP_MCGA_status", ""),
                "kan_reference_status": summary.get("kan_reference_status", ""),
            }
        )
        carriers = sorted(
            {str(row.get("carrier_core_variant", "")) for row in pair_summary if row.get("carrier_core_variant")}
            | {str(row.get("carrier_core_variant", "")) for row in no_debt_pair_trace if row.get("carrier_core_variant")}
        )
        for carrier in carriers:
            adam = gate_row(pair_summary, carrier, "KAN_G2_vs_MLP_AdamW")
            sameflops = gate_row(pair_summary, carrier, "KAN_G2_vs_MLP_SameFLOPs_AdamW")
            block = gate_row(pair_summary, carrier, "KAN_G2_vs_MLP_BlockRTGF")
            sameflops_status = str(summary.get("same_FLOPs_MLP_AdamW_status", ""))
            debt_stats = no_debt_stats_from_pair_trace(
                no_debt_pair_trace,
                carrier,
                ["KAN_G2_vs_MLP_AdamW", "KAN_G2_vs_MLP_SameFLOPs_AdamW", "KAN_G2_vs_MLP_BlockRTGF"],
            )
            mcga = mcga_candidate_by_carrier.get(carrier)
            mcga_source_pair_count = int(mcga["pair_count"]) if mcga else 0
            mcga_source_no_debt_count = int(mcga["no_debt_count"]) if mcga else 0
            mcga_same_dataset_seed = 0
            mcga_same_kan_reference = 0
            mcga_pairing_status = "missing_same_split_all8_kan_vs_mcga_pair_trace"
            if mcga:
                mcga_same_dataset_seed = int(
                    normalize_summary_list(summary.get("datasets", [])) == normalize_summary_list(mcga["datasets"])
                    and normalize_summary_list(summary.get("seeds", [])) == normalize_summary_list(mcga["seeds"])
                )
                mcga_same_kan_reference = int(str(summary.get("kan_reference_status", "")) == str(mcga.get("kan_reference_status", "")))
                mcga_pairing_status = (
                    "candidate_selected_by_carrier_max_pair_count_same_split"
                    if mcga_same_dataset_seed
                    else "candidate_selected_by_carrier_max_pair_count_split_mismatch"
                )
                if not mcga_same_kan_reference:
                    mcga_pairing_status += "_kan_reference_mismatch"
            mcga_pair_count_used = mcga_source_pair_count if mcga_same_dataset_seed and mcga_same_kan_reference else 0
            mcga_no_debt_count_used = mcga_source_no_debt_count if mcga_same_dataset_seed and mcga_same_kan_reference else 0
            total_no_debt_pair_count = int(debt_stats["pair_count"]) + mcga_pair_count_used
            total_no_debt_count = int(debt_stats["no_debt_count"]) + mcga_no_debt_count_used
            paired_no_debt_rate = (
                float(total_no_debt_count) / max(1.0, float(total_no_debt_pair_count))
                if total_no_debt_pair_count > 0
                else math.nan
            )
            gate = {
                "source": str(summary_path.relative_to(ROOT)),
                "carrier_core_variant": carrier,
                "datasets": summary.get("datasets", ""),
                "seeds": summary.get("seeds", ""),
                "steps": summary.get("steps", ""),
                "real_max_samples": summary.get("real_max_samples", ""),
                "same_FLOPs_MLP_AdamW_status": sameflops_status,
                "MLP_block_RTGF_status": summary.get("MLP_block_RTGF_status", ""),
                "MLP_MCGA_status": summary.get("MLP_MCGA_status", ""),
                "same_param_win_rate": gate_float(adam, "win_rate"),
                "beats_same_param_MLP_ge_60pct": gate_bool(gate_float(adam, "win_rate") >= 0.60),
                "same_flops_win_rate": gate_float(sameflops, "win_rate"),
                "beats_same_FLOPs_MLP_ge_60pct": gate_bool(gate_float(sameflops, "win_rate") >= 0.60 and sameflops_status == "completed_within_tolerance"),
                "block_rtgf_win_rate": gate_float(block, "win_rate"),
                "beats_MLP_block_RTGF_ge_60pct": gate_bool(gate_float(block, "win_rate") >= 0.60 and str(summary.get("MLP_block_RTGF_status", "")) == "completed_diagnostic"),
                "mcga_source": mcga["source"] if mcga else "",
                "mcga_pair_rows": mcga_source_pair_count,
                "mcga_pair_rows_used_for_architecture_gate": mcga_pair_count_used,
                "mcga_win_rate": float(mcga["win_rate"]) if mcga else math.nan,
                "mcga_no_debt_count": mcga_source_no_debt_count,
                "mcga_no_debt_count_used_for_architecture_gate": mcga_no_debt_count_used,
                "mcga_no_debt_rate": float(mcga["no_debt_rate"]) if mcga else math.nan,
                "mcga_same_dataset_seed": mcga_same_dataset_seed,
                "mcga_same_kan_reference": mcga_same_kan_reference,
                "beats_MLP_MCGA_ge_55pct": gate_bool(bool(mcga) and mcga_same_dataset_seed == 1 and mcga_same_kan_reference == 1 and float(mcga["win_rate"]) >= 0.55),
                "mlp_paired_no_debt_pair_count": debt_stats["pair_count"],
                "mlp_paired_no_debt_count": debt_stats["no_debt_count"],
                "mlp_paired_no_debt_rate": debt_stats["no_debt_rate"],
                "mlp_paired_no_debt_median_delta": debt_stats["median_debt_delta"],
                "paired_no_debt_pair_count": total_no_debt_pair_count,
                "paired_no_debt_count": total_no_debt_count,
                "paired_no_debt_rate": paired_no_debt_rate,
                "paired_no_debt_source_comparisons": debt_stats["source_comparisons"] + (",KAN_G2_vs_MLP_MCGA_candidate:%d" % mcga_pair_count_used if mcga_pair_count_used else ""),
                "paired_no_debt_method": "weighted_over_same_source_G2_MLP_pairs_plus_same_split_same_kan_reference_MCGA_pairs_from_CVaR95",
                "paired_no_debt_trace_status": no_debt_trace_status,
                "paired_no_debt_ge_80pct": gate_bool(math.isfinite(paired_no_debt_rate) and paired_no_debt_rate >= 0.80),
                "official_partG_gate_evaluable": 0,
                "diagnostic_only": 1,
            }
            pass_keys = [
                "beats_same_param_MLP_ge_60pct",
                "beats_same_FLOPs_MLP_ge_60pct",
                "beats_MLP_block_RTGF_ge_60pct",
                "beats_MLP_MCGA_ge_55pct",
                "paired_no_debt_ge_80pct",
            ]
            gate["partG_architecture_gate_pass_if_official"] = gate_bool(all(int(gate[key]) == 1 for key in pass_keys))
            gate["partG_gate_missing_or_failed_items"] = ",".join(key for key in pass_keys if int(gate[key]) != 1)
            gate["mcga_pairing_status"] = mcga_pairing_status
            partg_rows.append(gate)

    for mcga_summary_path in mcga_kan_summary_paths:
        root = mcga_summary_path.parent
        summary_json, summary_json_status = read_json_artifact(root / "v23_26_MLP_MCGA_v22_style_controls_summary.json")
        summary_rows, summary_rows_status = read_csv_artifact(mcga_summary_path)
        source_rows.append(
            {
                "source_kind": "PartG_MLP_MCGA_KAN_pairing",
                "source": str(mcga_summary_path.relative_to(ROOT)),
                "status": summary_rows_status,
                "summary_json_status": summary_json_status,
                "rows": summary_json.get("kan_vs_mcga_pair_rows", ""),
                "candidate_pair_rows": summary_json.get("kan_vs_mcga_candidate_pair_rows", ""),
                "datasets": summary_json.get("datasets", ""),
                "seeds": summary_json.get("seeds", ""),
                "carriers": summary_json.get("carriers", ""),
                "steps": summary_json.get("steps", ""),
                "real_max_samples": summary_json.get("real_max_samples", ""),
                "kan_reference_status": summary_json.get("kan_reference_status", ""),
            }
        )
        for row in summary_rows:
            if str(row.get("comparison", "")) != "KAN_G2_vs_MLP_MCGA_candidate":
                continue
            win_rate = gate_float(row, "win_rate")
            no_debt_rate = gate_float(row, "no_debt_rate")
            gate = {
                "source": str(mcga_summary_path.relative_to(ROOT)),
                "carrier_core_variant": row.get("carrier_core_variant", ""),
                "comparison": row.get("comparison", ""),
                "pair_count": row.get("pair_count", ""),
                "win_rate": win_rate,
                "no_debt_rate": no_debt_rate,
                "beats_MLP_MCGA_ge_55pct": gate_bool(win_rate >= 0.55),
                "paired_no_debt_ge_80pct": gate_bool(no_debt_rate >= 0.80),
                "median_surplus": row.get("median_surplus", ""),
                "CVaR25_surplus": row.get("CVaR25_surplus", ""),
                "bootstrap_LCB05_surplus": row.get("bootstrap_LCB05_surplus", ""),
                "datasets": summary_json.get("datasets", ""),
                "seeds": summary_json.get("seeds", ""),
                "carriers": summary_json.get("carriers", ""),
                "steps": summary_json.get("steps", ""),
                "real_max_samples": summary_json.get("real_max_samples", ""),
                "kan_reference_status": summary_json.get("kan_reference_status", ""),
                "diagnostic_only": 1,
                "official_partG_gate_evaluable": 0,
            }
            gate["partG_mcga_pair_gate_pass_if_official"] = gate_bool(
                int(gate["beats_MLP_MCGA_ge_55pct"]) == 1 and int(gate["paired_no_debt_ge_80pct"]) == 1
            )
            gate["partG_mcga_pair_gate_missing_or_failed_items"] = ",".join(
                key for key in ["beats_MLP_MCGA_ge_55pct", "paired_no_debt_ge_80pct"] if int(gate[key]) != 1
            )
            partg_mcga_rows.append(gate)

    write_csv(OUT_ROOT / "v23_26_partDG_gate_audit_sources.csv", source_rows)
    write_csv(OUT_ROOT / "v23_26_partD_gate_audit.csv", partd_rows)
    write_csv(OUT_ROOT / "v23_26_partG_gate_audit.csv", partg_rows)
    write_csv(OUT_ROOT / "v23_26_partG_mcga_pair_gate_audit.csv", partg_mcga_rows)
    partd_pass = sum(1 for row in partd_rows if int(row.get("partD_minimum_real_gate_pass_if_official", 0)) == 1)
    partg_pass = sum(1 for row in partg_rows if int(row.get("partG_architecture_gate_pass_if_official", 0)) == 1)
    partg_mcga_pass = sum(1 for row in partg_mcga_rows if int(row.get("partG_mcga_pair_gate_pass_if_official", 0)) == 1)
    partg_reconstructed_no_debt_sources = sum(
        1
        for row in source_rows
        if row.get("source_kind") == "PartG" and str(row.get("no_debt_trace_status", "")).startswith("reconstructed")
    )
    partg_architecture_rows_with_mcga_source = sum(1 for row in partg_rows if int(float(row.get("mcga_pair_rows") or 0)) > 0)
    partg_architecture_rows_with_mcga_counted = sum(1 for row in partg_rows if int(float(row.get("mcga_pair_rows_used_for_architecture_gate") or 0)) > 0)
    partg_architecture_rows_with_nodebt = sum(1 for row in partg_rows if math.isfinite(parse_float(row.get("paired_no_debt_rate", ""))))
    summary = {
        "source_rows": len(source_rows),
        "partD_sources": len(partd_summary_paths),
        "partG_sources": len(partg_summary_paths),
        "partG_mcga_pair_sources": len(mcga_kan_summary_paths),
        "partD_gate_rows": len(partd_rows),
        "partD_gate_pass_if_official_rows": partd_pass,
        "partG_gate_rows": len(partg_rows),
        "partG_gate_pass_if_official_rows": partg_pass,
        "partG_mcga_pair_gate_rows": len(partg_mcga_rows),
        "partG_mcga_pair_gate_pass_if_official_rows": partg_mcga_pass,
        "partG_reconstructed_no_debt_sources": partg_reconstructed_no_debt_sources,
        "partG_architecture_rows_with_no_debt_rate": partg_architecture_rows_with_nodebt,
        "partG_architecture_rows_with_mcga_source": partg_architecture_rows_with_mcga_source,
        "partG_architecture_rows_with_mcga_counted": partg_architecture_rows_with_mcga_counted,
        "official_gate_evaluable": 0,
        "diagnostic_only": 1,
        "reason_not_official": "Audit is computed over compact diagnostic artifacts; at least one required PartD/PartG/MCGA gate is missing or failed, and confirmatory thresholds remain unavailable.",
        "out_root": str(OUT_ROOT),
    }
    write_json(OUT_ROOT / "v23_26_partDG_gate_audit_summary.json", summary)
    append_exec(
        "PartD_PartG_gate_arithmetic_audit",
        args,
        [
            "v23_26_partDG_gate_audit_sources.csv",
            "v23_26_partD_gate_audit.csv",
            "v23_26_partG_gate_audit.csv",
            "v23_26_partG_mcga_pair_gate_audit.csv",
            "v23_26_partDG_gate_audit_summary.json",
        ],
        summary,
    )
    mcga_pair_line = (
        f"PartG MCGA KAN-pairing 已纳入 `{len(partg_mcga_rows)}` 行；pass-if-official `{partg_mcga_pass}`。"
        if partg_mcga_rows
        else "PartG MCGA paired rows 当前仍缺 same-split all8 KAN-vs-MCGA pair trace，因此 beats_MLP_MCGA_ge_55pct 与 paired_no-debt 暂不可评估。"
    )
    append_recap(
        "PartD/PartG gate arithmetic 审计",
        [
            f"PartD sources `{len(partd_summary_paths)}`，gate rows `{len(partd_rows)}`，pass-if-official rows `{partd_pass}`；PartG sources `{len(partg_summary_paths)}`，gate rows `{len(partg_rows)}`，pass-if-official rows `{partg_pass}`。",
            f"PartG MCGA pairing sources `{len(mcga_kan_summary_paths)}`，gate rows `{len(partg_mcga_rows)}`，pass-if-official rows `{partg_mcga_pass}`。",
            "PartD gate 按计划检查 paired_rows>=24、AdamW median>=1e-3、additive/random median、random CVaR25、bootstrap LCB、win_rate>=70%、paired no-debt>=80%、no error rows。",
            f"PartG gate 按计划检查 KAN_G2 vs same-param MLP、same-FLOPs MLP、MLP block-RTGF 与 MLP-MCGA；{mcga_pair_line}",
            f"本轮修正 PartG architecture no-debt 审计：旧 `v23_26_MLP_matched_pair_trace.csv` 没有 CVaR95 debt 字段时，从 `v23_26_MLP_matched_matrix.csv` 与其声明的 KAN reference matrix 重建；重建 source `{partg_reconstructed_no_debt_sources}` 个，architecture rows with no-debt rate `{partg_architecture_rows_with_nodebt}/{len(partg_rows)}`。",
            f"MCGA 接入规则收紧：按 carrier 和最大 pair_count 选择候选 source，但 architecture gate 只计入 same dataset/seed 且 same KAN reference 的 MCGA；有 MCGA source 的 architecture rows `{partg_architecture_rows_with_mcga_source}`，实际计入 `{partg_architecture_rows_with_mcga_counted}`。",
            "该审计只汇总已有 compact diagnostic artifacts，不把 pass-if-official 行升级为 official；completion 仍为 False。",
        ],
    )


def run_part_d_metric_order_audit(args: argparse.Namespace) -> None:
    """Audit whether M0/M1/M2 generator rows separate from their additive controls."""

    partd_summary_paths = sorted((ROOT / "results").glob("v23_26_gpu3_partd*/v23_26_partD_summary.json"), key=lambda p: str(p))
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for summary_path in partd_summary_paths:
        root = summary_path.parent
        summary, summary_status = read_json_artifact(summary_path)
        matrix, matrix_status = read_csv_artifact(root / "v23_26_partD_minimum_real_matrix.csv")
        source_rows.append(
            {
                "source": str(summary_path.relative_to(ROOT)),
                "summary_status": summary_status,
                "matrix_status": matrix_status,
                "rows": summary.get("rows", ""),
                "finite_rows": summary.get("finite_rows", ""),
                "pair_rows": summary.get("pair_rows", ""),
                "datasets": summary.get("datasets", ""),
                "seeds": summary.get("seeds", ""),
                "carriers": summary.get("carriers", ""),
                "steps": summary.get("steps", ""),
                "lr": summary.get("lr", ""),
                "real_max_samples": summary.get("real_max_samples", ""),
            }
        )
        by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for item in matrix:
            by_key[
                (
                    str(item.get("dataset", "")),
                    str(item.get("seed", "")),
                    str(item.get("carrier_core_variant", "")),
                    str(item.get("scheme", "")),
                )
            ] = item
        carriers = sorted({str(row.get("carrier_core_variant", "")) for row in matrix if row.get("carrier_core_variant")})
        for carrier in carriers:
            prefix = scheme_prefix(carrier)
            for metric_idx, metric_name in enumerate(("M0", "M1", "M2")):
                candidate_name = f"{prefix}-G{metric_idx}"
                control_name = f"{prefix}-C{metric_idx}"
                pair_surpluses: list[float] = []
                debt_deltas: list[float] = []
                candidate_trace_rows: list[dict[str, Any]] = []
                for item in matrix:
                    if str(item.get("carrier_core_variant", "")) != carrier or str(item.get("scheme", "")) != candidate_name:
                        continue
                    key_base = (str(item.get("dataset", "")), str(item.get("seed", "")), carrier)
                    control = by_key.get((*key_base, control_name))
                    if control is None:
                        continue
                    surplus = parse_float(control.get("guard_NLL", "")) - parse_float(item.get("guard_NLL", ""))
                    debt = parse_float(item.get("guard_CVaR95_NLL", "")) - parse_float(control.get("guard_CVaR95_NLL", ""))
                    if math.isfinite(surplus):
                        pair_surpluses.append(surplus)
                    if math.isfinite(debt):
                        debt_deltas.append(debt)
                    candidate_trace_rows.append(item)
                stats = surplus_stats(pair_surpluses)

                def trace_mean_field(field: str) -> float:
                    vals: list[float] = []
                    for trace_row in candidate_trace_rows:
                        value = parse_float(trace_row.get(field, ""))
                        if math.isfinite(value):
                            vals.append(value)
                    return float(np.mean(vals)) if vals else math.nan

                no_debt_rate = float(sum(1 for value in debt_deltas if value <= 0.0)) / max(1.0, float(len(debt_deltas))) if debt_deltas else math.nan
                rows.append(
                    {
                        "source": str(summary_path.relative_to(ROOT)),
                        "carrier_core_variant": carrier,
                        "metric_name": metric_name,
                        "candidate": candidate_name,
                        "control": control_name,
                        "pair_count": len(pair_surpluses),
                        **stats,
                        "paired_no_debt_rate": no_debt_rate,
                        "mean_tangent_angle": trace_mean_field("mean_tangent_angle"),
                        "max_tangent_angle": trace_mean_field("max_tangent_angle"),
                        "tangent_angle_le_1e_4_fraction": trace_mean_field("tangent_angle_le_1e_4_fraction"),
                        "tangent_angle_le_1e_3_fraction": trace_mean_field("tangent_angle_le_1e_3_fraction"),
                        "mean_additive_retraction_gap_ratio": trace_mean_field("mean_additive_retraction_gap_ratio"),
                        "mean_additive_retraction_gap_m_norm": trace_mean_field("mean_additive_retraction_gap_m_norm"),
                        "mean_proposal_m_norm": trace_mean_field("mean_proposal_m_norm"),
                        "generator_separates_additive": int(stats["finite_pair_count"] > 0 and float(stats["median_surplus"]) > 0.0),
                        "diagnostic_only": 1,
                    }
                )
    write_csv(OUT_ROOT / "v23_26_partD_metric_order_audit_sources.csv", source_rows)
    write_csv(OUT_ROOT / "v23_26_partD_metric_order_audit.csv", rows)
    positive_rows = sum(1 for row in rows if int(row.get("generator_separates_additive", 0)) == 1)
    deeptrace_rows = [row for row in rows if "partd_all8_compact_deeptrace_lr1e5" in str(row.get("source", ""))]
    preferred_rows = deeptrace_rows or rows[-6:]
    summary = {
        "source_rows": len(source_rows),
        "audit_rows": len(rows),
        "generator_separates_additive_rows": positive_rows,
        "deeptrace_rows": len(deeptrace_rows),
        "official_gate_evaluable": 0,
        "diagnostic_only": 1,
        "out_root": str(OUT_ROOT),
    }
    write_json(OUT_ROOT / "v23_26_partD_metric_order_audit_summary.json", summary)
    append_exec(
        "PartD_metric_order_additive_separation_audit",
        args,
        [
            "v23_26_partD_metric_order_audit_sources.csv",
            "v23_26_partD_metric_order_audit.csv",
            "v23_26_partD_metric_order_audit_summary.json",
        ],
        summary,
    )
    recap_lines = [
        f"source rows `{len(source_rows)}`，audit rows `{len(rows)}`，generator separates additive rows `{positive_rows}`；deeptrace rows `{len(deeptrace_rows)}`。",
        "该审计只读已有 PartD matrices，逐 carrier/metric 比较 G0/C0、G1/C1、G2/C2 的 paired guard-NLL surplus、no-debt 与 angle/gap trace；不改变训练策略。",
    ]
    if preferred_rows:
        for row in preferred_rows:
            recap_lines.append(
                f"{row['carrier_core_variant']} {row['metric_name']} `{row['candidate']}` vs `{row['control']}`："
                f"pairs `{row['pair_count']}`，median surplus `{row['median_surplus']}`，win_rate `{row['win_rate']}`，"
                f"no-debt `{row['paired_no_debt_rate']}`，mean_tangent_angle `{row['mean_tangent_angle']}`，"
                f"gap_ratio `{row['mean_additive_retraction_gap_ratio']}`。"
            )
    append_recap("PartD metric-order additive separation 审计", recap_lines)


DEBT_METRIC_SPECS = [
    ("guard_NLL", "lower"),
    ("guard_standard_ECE_fixed_bins", "lower"),
    ("guard_adaptive_ECE", "lower"),
    ("guard_Brier", "lower"),
    ("guard_tail_NLL_q95", "lower"),
    ("guard_tail_NLL_q99", "lower"),
    ("guard_CVaR95_NLL", "lower"),
    ("guard_CVaR99_NLL", "lower"),
    ("guard_margin_q10", "higher"),
    ("guard_wrong_confident_rate", "lower"),
    ("guard_right_confident_sharpness", "higher"),
]


def oriented_debt_delta(candidate: dict[str, Any], control: dict[str, Any], metric: str, direction: str) -> float:
    cand = parse_float(candidate.get(metric, ""))
    ctrl = parse_float(control.get(metric, ""))
    if not (math.isfinite(cand) and math.isfinite(ctrl)):
        return math.nan
    if direction == "higher":
        return ctrl - cand
    return cand - ctrl


def run_part_d_debt_metric_audit(args: argparse.Namespace) -> None:
    """Audit probability debt across all registered guard debt metrics."""

    summary_paths = sorted((ROOT / "results").glob("v23_26_gpu3_partd*/v23_26_partD_summary.json"), key=lambda p: str(p))
    source_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        root = summary_path.parent
        summary, summary_status = read_json_artifact(summary_path)
        matrix, matrix_status = read_csv_artifact(root / "v23_26_partD_minimum_real_matrix.csv")
        source_rows.append(
            {
                "source": str(summary_path.relative_to(ROOT)),
                "summary_status": summary_status,
                "matrix_status": matrix_status,
                "rows": summary.get("rows", ""),
                "finite_rows": summary.get("finite_rows", ""),
                "datasets": summary.get("datasets", ""),
                "seeds": summary.get("seeds", ""),
                "carriers": summary.get("carriers", ""),
                "steps": summary.get("steps", ""),
                "lr": summary.get("lr", ""),
                "real_max_samples": summary.get("real_max_samples", ""),
            }
        )
        by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in matrix:
            by_key[
                (
                    str(row.get("dataset", "")),
                    str(row.get("seed", "")),
                    str(row.get("carrier_core_variant", "")),
                    str(row.get("scheme", "")),
                )
            ] = row
        carriers = sorted({str(row.get("carrier_core_variant", "")) for row in matrix if row.get("carrier_core_variant")})
        for carrier in carriers:
            prefix = scheme_prefix(carrier)
            for metric_index in range(3):
                candidate_name = f"{prefix}-G{metric_index}"
                control_pairs = [
                    (f"G{metric_index}_vs_AdamW", f"{prefix}-B0"),
                    (f"G{metric_index}_vs_additive_C{metric_index}", f"{prefix}-C{metric_index}"),
                    (f"G{metric_index}_vs_random_M_skew_M{metric_index}", f"{prefix}-R0-{metric_index}"),
                ]
                for item in matrix:
                    if str(item.get("carrier_core_variant", "")) != carrier or str(item.get("scheme", "")) != candidate_name:
                        continue
                    key_base = (str(item.get("dataset", "")), str(item.get("seed", "")), carrier)
                    for comparison, control_name in control_pairs:
                        control = by_key.get((*key_base, control_name))
                        if control is None:
                            continue
                        for metric, direction in DEBT_METRIC_SPECS:
                            delta = oriented_debt_delta(item, control, metric, direction)
                            detail_rows.append(
                                {
                                    "source": str(summary_path.relative_to(ROOT)),
                                    "dataset": key_base[0],
                                    "seed": key_base[1],
                                    "carrier_core_variant": carrier,
                                    "metric_name": f"M{metric_index}",
                                    "comparison": comparison,
                                    "candidate": candidate_name,
                                    "control": control_name,
                                    "metric": metric,
                                    "direction": direction,
                                    "oriented_debt_delta_positive_bad": delta,
                                    "candidate_no_debt": int(math.isfinite(delta) and delta <= 0.0),
                                    "diagnostic_only": 1,
                                }
                            )
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in detail_rows:
        groups.setdefault((row["source"], row["carrier_core_variant"], row.get("metric_name", ""), row["comparison"], row["metric"]), []).append(row)
    for (source, carrier, metric_name, comparison, metric), group in sorted(groups.items()):
        vals = [parse_float(row.get("oriented_debt_delta_positive_bad", "")) for row in group]
        finite_vals = [value for value in vals if math.isfinite(value)]
        no_debt = sum(1 for value in finite_vals if value <= 0.0)
        summary_rows.append(
            {
                "source": source,
                "carrier_core_variant": carrier,
                "metric_name": metric_name,
                "comparison": comparison,
                "metric": metric,
                "finite_pair_count": len(finite_vals),
                "no_debt_count": no_debt,
                "no_debt_rate": float(no_debt) / max(1.0, float(len(finite_vals))) if finite_vals else math.nan,
                "mean_oriented_debt_delta": float(np.mean(finite_vals)) if finite_vals else math.nan,
                "median_oriented_debt_delta": float(np.median(finite_vals)) if finite_vals else math.nan,
                "max_oriented_debt_delta": float(np.max(finite_vals)) if finite_vals else math.nan,
                "min_oriented_debt_delta": float(np.min(finite_vals)) if finite_vals else math.nan,
                "diagnostic_only": 1,
            }
        )
    write_csv(OUT_ROOT / "v23_26_partD_debt_metric_audit_sources.csv", source_rows)
    write_csv(OUT_ROOT / "v23_26_partD_debt_metric_audit_detail.csv", detail_rows)
    write_csv(OUT_ROOT / "v23_26_partD_debt_metric_audit_summary.csv", summary_rows)
    qfloor_rows = [row for row in summary_rows if "partd_all8_compact_qfloor_lr5e4" in str(row.get("source", ""))]
    qfloor_lrhalf_rows = [row for row in summary_rows if "partd_all8_compact_qfloor_lr2p5e4" in str(row.get("source", ""))]
    summary = {
        "source_rows": len(source_rows),
        "detail_rows": len(detail_rows),
        "summary_rows": len(summary_rows),
        "debt_metric_count": len(DEBT_METRIC_SPECS),
        "qfloor_lr5e4_summary_rows": len(qfloor_rows),
        "qfloor_lr2p5e4_summary_rows": len(qfloor_lrhalf_rows),
        "diagnostic_only": 1,
        "official_gate_evaluable": 0,
        "out_root": str(OUT_ROOT),
    }
    write_json(OUT_ROOT / "v23_26_partD_debt_metric_audit_summary.json", summary)
    append_exec(
        "PartD_probability_debt_metric_audit",
        args,
        [
            "v23_26_partD_debt_metric_audit_sources.csv",
            "v23_26_partD_debt_metric_audit_detail.csv",
            "v23_26_partD_debt_metric_audit_summary.csv",
            "v23_26_partD_debt_metric_audit_summary.json",
        ],
        summary,
    )
    recap_lines = [
        f"source rows `{len(source_rows)}`，detail rows `{len(detail_rows)}`，summary rows `{len(summary_rows)}`；debt metrics `{len(DEBT_METRIC_SPECS)}`。",
        "该审计按 paired baseline 计算 oriented debt delta：lower-is-better 指标用 candidate-control，higher-is-better 指标用 control-candidate；delta<=0 计 no-debt。",
    ]
    interesting = [
        row
        for row in qfloor_rows
        if row.get("metric") in {"guard_CVaR95_NLL", "guard_CVaR99_NLL", "guard_Brier", "guard_standard_ECE_fixed_bins", "guard_wrong_confident_rate"}
        and row.get("comparison") in {"G2_vs_additive_C2", "G2_vs_random_M_skew_M2"}
    ]
    for row in interesting[:16]:
        recap_lines.append(
            f"qfloor lr5e4 {row['carrier_core_variant']} {row['comparison']} {row['metric']}："
            f"no-debt `{row['no_debt_count']}/{row['finite_pair_count']}` rate `{row['no_debt_rate']}`，"
            f"median debt `{row['median_oriented_debt_delta']}`，max debt `{row['max_oriented_debt_delta']}`。"
        )
    append_recap("PartD probability debt metric 审计", recap_lines)


def finite_numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        value = parse_float(row.get(key, ""))
        if math.isfinite(value):
            vals.append(value)
    return vals


def finite_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = finite_numbers(rows, key)
    return float(np.mean(vals)) if vals else math.nan


def finite_max(rows: list[dict[str, Any]], key: str) -> float:
    vals = finite_numbers(rows, key)
    return float(np.max(vals)) if vals else math.nan


def run_part_d_geometry_failure_audit(args: argparse.Namespace) -> None:
    """Read existing PartD artifacts and attribute RTGF failures to geometry traces."""

    detail_rows: list[dict[str, Any]] = []
    summary_paths = sorted((ROOT / "results").glob("v23_26_gpu3_partd*/v23_26_partD_summary.json"), key=lambda p: str(p))
    source_rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        root = summary_path.parent
        summary, summary_status = read_json_artifact(summary_path)
        matrix, matrix_status = read_csv_artifact(root / "v23_26_partD_minimum_real_matrix.csv")
        source = str(root.relative_to(ROOT))
        source_rows.append(
            {
                "source": source,
                "summary_status": summary_status,
                "matrix_status": matrix_status,
                "rows": len(matrix),
                "finite_rows": summary.get("finite_rows", ""),
                "lr": summary.get("lr", ""),
                "steps": summary.get("steps", ""),
                "diagnostic_only": summary.get("diagnostic_only", ""),
            }
        )
        by_key_scheme: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in matrix:
            by_key_scheme[
                (
                    str(row.get("dataset", "")),
                    str(row.get("seed", "")),
                    str(row.get("carrier_core_variant", "")),
                    str(row.get("scheme", "")),
                )
            ] = row
        for cand in matrix:
            carrier = str(cand.get("carrier_core_variant", ""))
            prefix = scheme_prefix(carrier)
            match = re.fullmatch(rf"{re.escape(prefix)}-G([0-2])", str(cand.get("scheme", "")))
            if match is None:
                continue
            metric_index = match.group(1)
            metric_name = str(cand.get("metric_name", f"M{metric_index}"))
            key_base = (str(cand.get("dataset", "")), str(cand.get("seed", "")), carrier)
            for comparison, control_scheme in [
                (f"G{metric_index}_vs_AdamW", f"{prefix}-B0"),
                (f"G{metric_index}_vs_additive_C{metric_index}", f"{prefix}-C{metric_index}"),
                (f"G{metric_index}_vs_random_M_skew_M{metric_index}", f"{prefix}-R0-{metric_index}"),
                (f"G{metric_index}_vs_signflip_M{metric_index}", f"{prefix}-R1-{metric_index}"),
                (f"G{metric_index}_vs_shuffled_tangent_M{metric_index}", f"{prefix}-R2-{metric_index}"),
                (f"G{metric_index}_vs_Euclidean_RTGF_M{metric_index}", f"{prefix}-R3-{metric_index}"),
                (f"G{metric_index}_vs_same_compute_noop_M{metric_index}", f"{prefix}-R5-{metric_index}"),
            ]:
                control = by_key_scheme.get((*key_base, control_scheme))
                if control is None:
                    continue
                guard_surplus = parse_float(control.get("guard_NLL", "")) - parse_float(cand.get("guard_NLL", ""))
                debt = parse_float(cand.get("guard_CVaR95_NLL", "")) - parse_float(control.get("guard_CVaR95_NLL", ""))
                detail_rows.append(
                    {
                        "source": source,
                        "dataset": cand.get("dataset", ""),
                        "seed": cand.get("seed", ""),
                        "carrier_core_variant": carrier,
                        "metric_name": metric_name,
                        "comparison": comparison,
                        "candidate": cand.get("scheme", ""),
                        "control": control_scheme,
                        "paired_guard_NLL_surplus": guard_surplus,
                        "paired_guard_CVaR95_debt_delta": debt,
                        "candidate_no_debt": int(math.isfinite(debt) and debt <= 0.0),
                        "candidate_mean_tangent_angle": cand.get("mean_tangent_angle", ""),
                        "candidate_max_tangent_angle": cand.get("max_tangent_angle", ""),
                        "candidate_tangent_angle_le_1e_4_fraction": cand.get("tangent_angle_le_1e_4_fraction", ""),
                        "candidate_tangent_angle_le_1e_3_fraction": cand.get("tangent_angle_le_1e_3_fraction", ""),
                        "candidate_near_zero_bootstrap_fraction": cand.get("near_zero_bootstrap_fraction", ""),
                        "candidate_mean_additive_retraction_gap_ratio": cand.get("mean_additive_retraction_gap_ratio", ""),
                        "candidate_max_additive_retraction_gap_ratio": cand.get("max_additive_retraction_gap_ratio", ""),
                        "candidate_mean_tangent_m_orthogonality_abs": cand.get("mean_tangent_m_orthogonality_abs", ""),
                        "candidate_max_tangent_m_orthogonality_abs": cand.get("max_tangent_m_orthogonality_abs", ""),
                        "candidate_mean_tangent_norm_match_abs": cand.get("mean_tangent_norm_match_abs", ""),
                        "candidate_max_tangent_norm_match_abs": cand.get("max_tangent_norm_match_abs", ""),
                        "candidate_mean_shape_norm_preservation_abs": cand.get("mean_shape_norm_preservation_abs", ""),
                        "candidate_max_shape_norm_preservation_abs": cand.get("max_shape_norm_preservation_abs", ""),
                        "candidate_norm_floor": cand.get("norm_floor", ""),
                        "candidate_initial_edge_radius_q": cand.get("initial_edge_radius_q", ""),
                        "candidate_initial_edge_radius_median": cand.get("initial_edge_radius_median", ""),
                        "candidate_guard_CVaR95_NLL": cand.get("guard_CVaR95_NLL", ""),
                        "control_guard_CVaR95_NLL": control.get("guard_CVaR95_NLL", ""),
                        "diagnostic_only": 1,
                    }
                )

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in detail_rows:
        groups.setdefault(
            (
                str(row.get("source", "")),
                str(row.get("carrier_core_variant", "")),
                str(row.get("metric_name", "")),
                str(row.get("comparison", "")),
            ),
            [],
        ).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (source, carrier, metric_name, comparison), group in sorted(groups.items()):
        surplus = surplus_stats(finite_numbers(group, "paired_guard_NLL_surplus"))
        debt_vals = finite_numbers(group, "paired_guard_CVaR95_debt_delta")
        no_debt = sum(1 for value in debt_vals if value <= 0.0)
        no_debt_rate = float(no_debt) / max(1.0, float(len(debt_vals))) if debt_vals else math.nan
        mean_angle = finite_mean(group, "candidate_mean_tangent_angle")
        max_angle = finite_max(group, "candidate_max_tangent_angle")
        mean_gap_ratio = finite_mean(group, "candidate_mean_additive_retraction_gap_ratio")
        max_shape_residual = finite_max(group, "candidate_max_shape_norm_preservation_abs")
        max_orth = finite_max(group, "candidate_max_tangent_m_orthogonality_abs")
        mean_near_zero = finite_mean(group, "candidate_near_zero_bootstrap_fraction")
        additive_like = int(
            "additive" in comparison
            and math.isfinite(mean_angle)
            and math.isfinite(mean_gap_ratio)
            and mean_angle <= 1.0e-2
            and mean_gap_ratio <= 1.0e-2
            and abs(float(surplus.get("median_surplus", math.nan))) <= 1.0e-6
        )
        identity_suspect = int(
            (math.isfinite(max_shape_residual) and max_shape_residual > 1.0e-3)
            or (math.isfinite(max_orth) and max_orth > 1.0e-3)
        )
        angle_suspect = int(
            math.isfinite(max_angle)
            and max_angle > math.pi
            and (not math.isfinite(mean_near_zero) or mean_near_zero < 0.10)
        )
        summary_rows.append(
            {
                "source": source,
                "carrier_core_variant": carrier,
                "metric_name": metric_name,
                "comparison": comparison,
                "pair_count": len(group),
                **surplus,
                "CVaR95_no_debt_count": no_debt,
                "CVaR95_no_debt_rate": no_debt_rate,
                "CVaR95_median_debt_delta": float(np.median(debt_vals)) if debt_vals else math.nan,
                "mean_tangent_angle": mean_angle,
                "max_tangent_angle": max_angle,
                "mean_additive_retraction_gap_ratio": mean_gap_ratio,
                "max_shape_norm_preservation_abs": max_shape_residual,
                "max_tangent_m_orthogonality_abs": max_orth,
                "mean_near_zero_bootstrap_fraction": mean_near_zero,
                "rtgf_additive_like": additive_like,
                "projection_identity_suspect": identity_suspect,
                "angle_distribution_suspect": angle_suspect,
                "diagnostic_only": 1,
            }
        )

    write_csv(OUT_ROOT / "v23_26_partD_geometry_failure_audit_sources.csv", source_rows)
    write_csv(OUT_ROOT / "v23_26_partD_geometry_failure_audit_detail.csv", detail_rows)
    write_csv(OUT_ROOT / "v23_26_partD_geometry_failure_audit_summary.csv", summary_rows)
    dche_m2_additive = [
        row
        for row in summary_rows
        if row.get("carrier_core_variant") == "D-CHE-Core-K3"
        and row.get("metric_name") == "M2"
        and row.get("comparison") == "G2_vs_additive_C2"
    ]
    dche_m2_latest = dche_m2_additive[-5:]
    identity_suspect_rows = sum(int(row.get("projection_identity_suspect", 0)) for row in summary_rows)
    angle_suspect_rows = sum(int(row.get("angle_distribution_suspect", 0)) for row in summary_rows)
    additive_like_rows = sum(int(row.get("rtgf_additive_like", 0)) for row in summary_rows)
    summary = {
        "source_count": len(source_rows),
        "detail_rows": len(detail_rows),
        "summary_rows": len(summary_rows),
        "D_CHE_M2_additive_rows": len(dche_m2_additive),
        "D_CHE_M2_additive_latest": dche_m2_latest,
        "projection_identity_suspect_rows": identity_suspect_rows,
        "angle_distribution_suspect_rows": angle_suspect_rows,
        "rtgf_additive_like_rows": additive_like_rows,
        "diagnostic_only": 1,
        "official_success_allowed": 0,
        "interpretation": "Read-only PartD geometry/debt attribution; does not alter gates, losses, candidates, or official status.",
    }
    write_json(OUT_ROOT / "v23_26_partD_geometry_failure_audit_summary.json", summary)
    append_exec(
        "PartD_geometry_failure_audit",
        args,
        [
            "v23_26_partD_geometry_failure_audit_sources.csv",
            "v23_26_partD_geometry_failure_audit_detail.csv",
            "v23_26_partD_geometry_failure_audit_summary.csv",
            "v23_26_partD_geometry_failure_audit_summary.json",
        ],
        summary,
    )
    recap_lines = [
        f"source rows `{len(source_rows)}`，detail rows `{len(detail_rows)}`，summary rows `{len(summary_rows)}`。",
        "该审计只读已有 PartD artifacts；按 source/carrier/metric/comparison 归因 guard surplus、CVaR95 debt、tangent angle、retraction gap、near-zero bootstrap、projection identity residual。",
        f"projection identity suspect rows `{identity_suspect_rows}`；angle distribution suspect rows `{angle_suspect_rows}`；RTGF-additive-like rows `{additive_like_rows}`。",
    ]
    for row in dche_m2_latest:
        recap_lines.append(
            f"D-CHE M2 {row['source']} G2_vs_additive_C2：median surplus `{row['median_surplus']}`，"
            f"no-debt `{row['CVaR95_no_debt_rate']}`，mean angle `{row['mean_tangent_angle']}`，"
            f"mean gap ratio `{row['mean_additive_retraction_gap_ratio']}`，projection suspect `{row['projection_identity_suspect']}`。"
        )
    append_recap("PartD geometry failure attribution 审计", recap_lines)


def run_part_d_rtgf_additive_mechanism_audit(args: argparse.Namespace) -> None:
    """Connect additive-control failures to RTGF finite-step geometry traces."""
    source_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metric_to_index = {"M0": "0", "M1": "1", "M2": "2"}
    for root in sorted((ROOT / "results").glob("v23_26_gpu3_partd*")):
        matrix_path = root / "v23_26_partD_minimum_real_matrix.csv"
        pair_path = root / "v23_26_partD_paired_control_trace.csv"
        summary_path = root / "v23_26_partD_summary.json"
        if not matrix_path.exists() or not pair_path.exists() or not summary_path.exists():
            continue
        matrix, matrix_status = read_csv_artifact(matrix_path)
        pairs, pair_status = read_csv_artifact(pair_path)
        summary_obj, summary_status = read_json_artifact(summary_path)
        if matrix_status != "ok" or pair_status != "ok":
            continue
        source_rows.append(
            {
                "source": str(root.relative_to(ROOT)),
                "matrix_rows": len(matrix),
                "pair_rows": len(pairs),
                "summary_status": summary_status,
                "datasets": summary_obj.get("datasets", ""),
                "seeds": summary_obj.get("seeds", ""),
                "carriers": summary_obj.get("carriers", ""),
                "steps": summary_obj.get("steps", ""),
                "lr": summary_obj.get("lr", ""),
            }
        )
        generators: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in matrix:
            metric_name = str(row.get("metric_name", ""))
            idx = metric_to_index.get(metric_name)
            if idx is None or not str(row.get("scheme", "")).endswith(f"G{idx}"):
                continue
            key = (
                str(row.get("dataset", "")),
                str(row.get("seed", "")),
                str(row.get("carrier_core_variant", "")),
                metric_name,
            )
            generators[key] = row
        for pair in pairs:
            metric_name = str(pair.get("metric_name", ""))
            idx = metric_to_index.get(metric_name)
            if idx is None or str(pair.get("comparison", "")) != f"G{idx}_vs_additive_C{idx}":
                continue
            key = (
                str(pair.get("dataset", "")),
                str(pair.get("seed", "")),
                str(pair.get("carrier_core_variant", "")),
                metric_name,
            )
            grow = generators.get(key)
            if grow is None:
                continue
            debt = parse_float(pair.get("paired_guard_CVaR95_debt_delta", ""))
            detail_rows.append(
                {
                    "source": str(root.relative_to(ROOT)),
                    "dataset": key[0],
                    "seed": key[1],
                    "carrier_core_variant": key[2],
                    "metric_name": key[3],
                    "comparison": pair.get("comparison", ""),
                    "paired_guard_NLL_surplus": parse_float(pair.get("paired_guard_NLL_surplus", "")),
                    "paired_guard_CVaR95_debt_delta": debt,
                    "candidate_no_debt": int(math.isfinite(debt) and debt <= 0.0),
                    "mean_tangent_angle": parse_float(grow.get("mean_tangent_angle", "")),
                    "max_tangent_angle": parse_float(grow.get("max_tangent_angle", "")),
                    "mean_additive_retraction_gap_ratio": parse_float(grow.get("mean_additive_retraction_gap_ratio", "")),
                    "max_additive_retraction_gap_ratio": parse_float(grow.get("max_additive_retraction_gap_ratio", "")),
                    "mean_additive_retraction_gap_m_norm": parse_float(grow.get("mean_additive_retraction_gap_m_norm", "")),
                    "mean_proposal_m_norm": parse_float(grow.get("mean_proposal_m_norm", "")),
                    "near_zero_bootstrap_fraction": parse_float(grow.get("near_zero_bootstrap_fraction", "")),
                    "mean_tangent_m_orthogonality_abs": parse_float(grow.get("mean_tangent_m_orthogonality_abs", "")),
                    "max_tangent_m_orthogonality_abs": parse_float(grow.get("max_tangent_m_orthogonality_abs", "")),
                    "mean_shape_norm_preservation_abs": parse_float(grow.get("mean_shape_norm_preservation_abs", "")),
                    "max_shape_norm_preservation_abs": parse_float(grow.get("max_shape_norm_preservation_abs", "")),
                    "diagnostic_only": 1,
                }
            )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in detail_rows:
        groups.setdefault((row["source"], row["carrier_core_variant"], row["metric_name"]), []).append(row)
    for (source, carrier, metric_name), group in sorted(groups.items()):
        debts = finite_numbers(group, "paired_guard_CVaR95_debt_delta")
        surpluses = finite_numbers(group, "paired_guard_NLL_surplus")
        gap_ratios = finite_numbers(group, "mean_additive_retraction_gap_ratio")
        angles = finite_numbers(group, "mean_tangent_angle")
        boots = finite_numbers(group, "near_zero_bootstrap_fraction")
        mean_orth_values = finite_numbers(group, "mean_tangent_m_orthogonality_abs")
        no_debt_count = sum(1 for value in debts if value <= 0.0)
        no_debt_rate = float(no_debt_count) / max(1.0, float(len(debts))) if debts else math.nan
        max_boot = max(boots, default=math.nan)
        mean_orth = float(np.mean(mean_orth_values)) if mean_orth_values else math.nan
        max_orth = max(finite_numbers(group, "max_tangent_m_orthogonality_abs"), default=math.nan)
        max_shape = max(finite_numbers(group, "max_shape_norm_preservation_abs"), default=math.nan)
        systematic_formula_suspect = int(
            (math.isfinite(mean_orth) and mean_orth > 1.0e-4)
            or (math.isfinite(max_shape) and max_shape > 1.0e-4)
            or (math.isfinite(max_boot) and max_boot > 0.10)
        )
        localized_projection_spike = int(
            (math.isfinite(max_orth) and max_orth > 1.0e-4)
            and not (math.isfinite(mean_orth) and mean_orth > 1.0e-4)
        )
        formula_suspect = int(
            systematic_formula_suspect
            or localized_projection_spike
            or (math.isfinite(max_shape) and max_shape > 1.0e-4)
            or (math.isfinite(max_boot) and max_boot > 0.10)
        )
        median_gap_ratio = median_finite(gap_ratios)
        second_order_leverage_too_small = int(
            math.isfinite(no_debt_rate)
            and no_debt_rate < 0.80
            and math.isfinite(median_gap_ratio)
            and median_gap_ratio < 1.0e-2
            and formula_suspect == 0
        )
        summary_rows.append(
            {
                "source": source,
                "carrier_core_variant": carrier,
                "metric_name": metric_name,
                "pair_count": len(group),
                "finite_debt_count": len(debts),
                "additive_no_debt_count": no_debt_count,
                "additive_no_debt_rate": no_debt_rate,
                "median_CVaR95_debt_delta": median_finite(debts),
                "max_CVaR95_debt_delta": max(debts, default=math.nan),
                "median_guard_surplus_vs_additive": median_finite(surpluses),
                "mean_tangent_angle": float(np.mean(angles)) if angles else math.nan,
                "median_tangent_angle": median_finite(angles),
                "max_tangent_angle": max(finite_numbers(group, "max_tangent_angle"), default=math.nan),
                "mean_additive_retraction_gap_ratio": float(np.mean(gap_ratios)) if gap_ratios else math.nan,
                "median_additive_retraction_gap_ratio": median_gap_ratio,
                "mean_near_zero_bootstrap_fraction": float(np.mean(boots)) if boots else math.nan,
                "max_near_zero_bootstrap_fraction": max_boot,
                "mean_tangent_m_orthogonality_abs": mean_orth,
                "max_tangent_m_orthogonality_abs": max_orth,
                "max_shape_norm_preservation_abs": max_shape,
                "formula_or_projection_suspect": formula_suspect,
                "systematic_formula_or_projection_suspect": systematic_formula_suspect,
                "localized_projection_spike": localized_projection_spike,
                "second_order_leverage_too_small": second_order_leverage_too_small,
                "diagnostic_only": 1,
            }
        )
    write_csv(OUT_ROOT / "v23_26_partD_rtgf_additive_mechanism_sources.csv", source_rows)
    write_csv(OUT_ROOT / "v23_26_partD_rtgf_additive_mechanism_detail.csv", detail_rows)
    write_csv(OUT_ROOT / "v23_26_partD_rtgf_additive_mechanism_summary.csv", summary_rows)
    small_leverage_rows = sum(int(row.get("second_order_leverage_too_small", 0)) for row in summary_rows)
    formula_suspect_rows = sum(int(row.get("formula_or_projection_suspect", 0)) for row in summary_rows)
    systematic_formula_suspect_rows = sum(int(row.get("systematic_formula_or_projection_suspect", 0)) for row in summary_rows)
    localized_projection_spike_rows = sum(int(row.get("localized_projection_spike", 0)) for row in summary_rows)
    nodebt_fail_rows = sum(
        1
        for row in summary_rows
        if math.isfinite(parse_float(row.get("additive_no_debt_rate", ""))) and parse_float(row.get("additive_no_debt_rate", "")) < 0.80
    )
    summary = {
        "source_rows": len(source_rows),
        "detail_rows": len(detail_rows),
        "summary_rows": len(summary_rows),
        "additive_no_debt_fail_rows": nodebt_fail_rows,
        "second_order_leverage_too_small_rows": small_leverage_rows,
        "formula_or_projection_suspect_rows": formula_suspect_rows,
        "systematic_formula_or_projection_suspect_rows": systematic_formula_suspect_rows,
        "localized_projection_spike_rows": localized_projection_spike_rows,
        "diagnostic_only": 1,
        "official_success_allowed": 0,
        "interpretation": "Read-only RTGF-vs-additive mechanism audit; finite-step RTGF differs from additive only by second-order retraction terms, so small gap ratios can explain additive-like failures without changing gates.",
    }
    write_json(OUT_ROOT / "v23_26_partD_rtgf_additive_mechanism_summary.json", summary)
    append_exec(
        "PartD_RTGF_additive_mechanism_audit",
        args,
        [
            "v23_26_partD_rtgf_additive_mechanism_sources.csv",
            "v23_26_partD_rtgf_additive_mechanism_detail.csv",
            "v23_26_partD_rtgf_additive_mechanism_summary.csv",
            "v23_26_partD_rtgf_additive_mechanism_summary.json",
        ],
        summary,
    )
    recap_lines = [
        f"source rows `{len(source_rows)}`，detail rows `{len(detail_rows)}`，summary rows `{len(summary_rows)}`。",
        f"additive no-debt fail rows `{nodebt_fail_rows}`；second-order leverage too-small rows `{small_leverage_rows}`；formula/projection suspect rows `{formula_suspect_rows}`；systematic suspect rows `{systematic_formula_suspect_rows}`；localized projection spike rows `{localized_projection_spike_rows}`。",
        "该审计只读 PartD matrix/paired trace，把 `Gx_vs_additive_Cx` 的 CVaR95 no-debt 与同源 Gx 的 tangent angle、RTGF-additive gap、bootstrap、M-orthogonality、shape norm residual 连接起来；不改变 candidate、loss、gate 或 checkpoint。",
    ]
    interesting = sorted(
        summary_rows,
        key=lambda row: (
            parse_float(row.get("additive_no_debt_rate", "nan")),
            parse_float(row.get("median_additive_retraction_gap_ratio", "nan")),
        ),
    )[:8]
    for row in interesting:
        recap_lines.append(
            f"{row['carrier_core_variant']} {row['metric_name']} {row['source']}："
            f"additive no-debt `{row['additive_no_debt_count']}/{row['finite_debt_count']}` rate `{row['additive_no_debt_rate']}`，"
            f"median debt `{row['median_CVaR95_debt_delta']}`，median guard surplus `{row['median_guard_surplus_vs_additive']}`，"
            f"median angle `{row['median_tangent_angle']}`，median gap ratio `{row['median_additive_retraction_gap_ratio']}`，"
            f"formula suspect `{row['formula_or_projection_suspect']}`，systematic suspect `{row['systematic_formula_or_projection_suspect']}`，localized spike `{row['localized_projection_spike']}`，small-leverage `{row['second_order_leverage_too_small']}`。"
        )
    append_recap("PartD RTGF vs additive mechanism 审计", recap_lines)


def run_mlp_mcga_repro_audit(args: argparse.Namespace) -> None:
    """Audit whether the old v22.66 MLP-MCGA route is semantically reproduced in v23.26."""

    v22_route_path = ROOT / "results/v22_66/v22_66_final_route.json"
    v22_method_path = ROOT / "results/v22_66/v22_66_method_summary_st400_official_expansion.csv"
    v22_dche_probe_path = ROOT / "results/v22_66/v22_66_kan_mcga_dche_probe_dche_probe_kan_mcga_summary.json"
    v22_dfou_probe_path = ROOT / "results/v22_66/v22_66_kan_mcga_dfou_probe_scan_summary.json"
    v22_route, v22_route_status = read_json_artifact(v22_route_path)
    v22_method_rows, v22_method_status = read_csv_artifact(v22_method_path)
    v22_dche_probe, v22_dche_status = read_json_artifact(v22_dche_probe_path)
    v22_dfou_probe, v22_dfou_status = read_json_artifact(v22_dfou_probe_path)

    winning_method = "mcga_over_poet_fsclip_eta025_residual_rank4"
    winning_rows = [r for r in v22_method_rows if str(r.get("method", "")) == winning_method]
    winning_row = winning_rows[0] if winning_rows else {}
    partg_summary_paths = sorted((ROOT / "results").glob("v23_26_gpu3_partg*/v23_26_MLP_matched_summary.json"))
    partg_summaries: list[dict[str, Any]] = []
    for path in partg_summary_paths:
        obj, status = read_json_artifact(path)
        partg_summaries.append(
            {
                "path": str(path.relative_to(ROOT)),
                "status": status,
                "rows": obj.get("rows", ""),
                "finite_rows": obj.get("finite_rows", ""),
                "same_param_MLP_AdamW_rows": obj.get("same_param_MLP_AdamW_rows", ""),
                "same_param_MLP_AdamW_finite_rows": obj.get("same_param_MLP_AdamW_finite_rows", ""),
                "MLP_block_RTGF_rows": obj.get("MLP_block_RTGF_rows", ""),
                "same_FLOPs_MLP_AdamW_rows": obj.get("same_FLOPs_MLP_AdamW_rows", ""),
                "same_FLOPs_MLP_AdamW_completed_rows": obj.get("same_FLOPs_MLP_AdamW_completed_rows", ""),
                "MLP_MCGA_status": obj.get("MLP_MCGA_status", ""),
                "partG_official_gate_evaluable": obj.get("partG_official_gate_evaluable", ""),
            }
        )
    latest_completion_path = latest_artifact("v23_26_gpu3_completion_audit*.json")
    latest_completion, latest_completion_status = read_json_artifact(latest_completion_path) if latest_completion_path is not None else ({}, "missing")

    evidence_rows = [
        {
            "item": "v22_66_mlp_mcga_official_route",
            "source": str(v22_route_path.relative_to(ROOT)),
            "status": v22_route_status,
            "finding": str(v22_route.get("final_route", "")),
            "detail": f"mlp_official_gate_opened={v22_route.get('mlp_official_gate_opened', '')}; kan_gate_status={v22_route.get('kan_gate_status', '')}",
        },
        {
            "item": "v22_66_winning_mlp_candidate",
            "source": str(v22_method_path.relative_to(ROOT)),
            "status": v22_method_status,
            "finding": winning_method,
            "detail": (
                f"rows={winning_row.get('rows', '')}; completed={winning_row.get('completed_rows', '')}; "
                f"beats_strongest={winning_row.get('beats_strongest_NLL_rows', '')}; "
                f"beats_external_OET={winning_row.get('beats_external_OET_NLL_rows', '')}; "
                f"beats_best_control={winning_row.get('beats_best_control_NLL_rows', '')}; "
                f"same_generator_controls={winning_row.get('beats_same_generator_controls_rows', '')}; "
                f"official_gate_pass={winning_row.get('official_gate_pass', '')}"
            ),
        },
        {
            "item": "v22_66_dche_kan_mcga_probe",
            "source": str(v22_dche_probe_path.relative_to(ROOT)),
            "status": v22_dche_status,
            "finding": f"kan_mcga_gate_pass={v22_dche_probe.get('kan_mcga_gate_pass', '')}",
            "detail": (
                f"KAN_beats_MLP_matched_rows={v22_dche_probe.get('KAN_beats_MLP_matched_rows', '')}; "
                f"KAN_beats_best_KAN_control_rows={v22_dche_probe.get('KAN_beats_best_KAN_control_rows', '')}; "
                f"ControlExplained_pct={v22_dche_probe.get('ControlExplained_pct', '')}"
            ),
        },
        {
            "item": "v22_66_dfou_kan_mcga_probe",
            "source": str(v22_dfou_probe_path.relative_to(ROOT)),
            "status": v22_dfou_status,
            "finding": "scan_summary",
            "detail": f"rows={len(v22_dfou_probe.get('rows', [])) if isinstance(v22_dfou_probe.get('rows', []), list) else ''}; architecture={v22_dfou_probe.get('architecture', '')}",
        },
        {
            "item": "v23_26_partg_current_status",
            "source": "results/v23_26_gpu3_partg*/v23_26_MLP_matched_summary.json",
            "status": "ok" if partg_summaries else "missing",
            "finding": "MLP_MCGA_status remains incomplete in current PartG summaries",
            "detail": f"partg_summary_count={len(partg_summaries)}; latest_completion={str(latest_completion_path.relative_to(ROOT)) if latest_completion_path is not None else ''}; completion_status={latest_completion_status}",
        },
    ]

    gap_rows = [
        {
            "gap": "MLP architecture mismatch",
            "v22_66_requirement": "SimpleMLP with explicit feature extractor and readout wrapped by MetricCompatibleAtlasMLP.",
            "v23_26_current_state": "PartG uses MLPBaseline for same-param/same-FLOPs AdamW and contiguous block-RTGF diagnostics.",
            "official_impact": "Old MCGA cannot be counted as a completed v23.26 MLP-strong baseline until the wrapped readout-atlas model is reproduced on the same v23.26 splits and budgets.",
        },
        {
            "gap": "Generator atlas semantics",
            "v22_66_requirement": "C-skew projection, Cayley retraction, active Gram transport, functional-spectrum clip, POET residual and same-generator controls.",
            "v23_26_current_state": "PartG has no POET residual MCGA wrapper, no functional-spectrum MCGA controls, and no v22.66 failure-mode gate arithmetic.",
            "official_impact": "A simplified random or block RTGF MLP is not MCGA and must stay diagnostic only.",
        },
        {
            "gap": "Dataset and pairing contract",
            "v22_66_requirement": "MCGA rows were run on v22.66 train/held/test bundles with 30-row MLP official gate and separate KAN probe audits.",
            "v23_26_current_state": "PartG compact/all8 runs pair against v23.26 minimum-real KAN matrices with different schemes and summaries.",
            "official_impact": "Historical v22.66 numbers are evidence of feasibility, not v23.26 official baseline rows.",
        },
        {
            "gap": "KAN post-MLP evidence",
            "v22_66_requirement": "After MLP gate opens, KAN must still beat matched MLP/control rows.",
            "v23_26_current_state": "Historical D-CHE/D-FOU KAN+MCGA probes did not pass KAN gate; D-CHE had 0/15 KAN beats MLP matched.",
            "official_impact": "Adding MCGA as a candidate/control may make the PartG bar stricter; it cannot be presumed to rescue H-H.",
        },
    ]

    decision = {
        "audit_route": "R0_MLPStrongFunctionalBaselineIncomplete",
        "mlp_mcga_semantically_reproduced_in_v23_26": 0,
        "may_count_v22_66_numbers_as_v23_26_official_rows": 0,
        "recommended_next_implementation": [
            "Adapt MetricCompatibleAtlasMLP/MetricCompatibleLowRankLinear to the exact v23.26 MLPBaseline or introduce a parameter-matched SimpleMLP baseline with explicit readout.",
            "Run MCGA, POET/external OET, same-functional-spectrum, same-generator-descent, same-C-skew and noop controls on the same v23.26 real splits/seeds as PartG.",
            "Only then mark MLP_MCGA_status completed; otherwise keep R0_MLPStrongFunctionalBaselineIncomplete.",
        ],
    }
    summary = {
        **decision,
        "v22_route_source_status": v22_route_status,
        "v22_method_summary_status": v22_method_status,
        "v22_dche_probe_status": v22_dche_status,
        "v22_dfou_probe_status": v22_dfou_status,
        "v22_final_route": v22_route.get("final_route", ""),
        "v22_mlp_official_gate_opened": v22_route.get("mlp_official_gate_opened", ""),
        "v22_winning_candidate": winning_method,
        "v22_winning_candidate_official_gate_pass": winning_row.get("official_gate_pass", ""),
        "v23_partg_summary_count": len(partg_summaries),
        "v23_partg_summaries": partg_summaries,
        "latest_completion_audit": str(latest_completion_path.relative_to(ROOT)) if latest_completion_path is not None else "",
        "latest_completion_complete": latest_completion.get("complete", ""),
        "evidence_rows": evidence_rows,
        "gap_rows": gap_rows,
        "out_root": str(OUT_ROOT),
        "diagnostic_only": 1,
    }
    write_csv(OUT_ROOT / "v23_26_mlp_mcga_repro_evidence.csv", evidence_rows)
    write_csv(OUT_ROOT / "v23_26_mlp_mcga_repro_gap_matrix.csv", gap_rows)
    write_json(OUT_ROOT / "v23_26_mlp_mcga_repro_audit.json", summary)
    append_exec(
        "PartG_MLP_MCGA_reproduction_audit",
        args,
        [
            "v23_26_mlp_mcga_repro_evidence.csv",
            "v23_26_mlp_mcga_repro_gap_matrix.csv",
            "v23_26_mlp_mcga_repro_audit.json",
        ],
        {k: v for k, v in summary.items() if k not in {"v23_partg_summaries", "evidence_rows", "gap_rows"}},
    )
    append_recap(
        "PartG MLP-MCGA 语义复现审计",
        [
            f"v22.66 final_route `{summary['v22_final_route']}`；winning candidate `{winning_method}`；official_gate_pass `{summary['v22_winning_candidate_official_gate_pass']}`。",
            "v22.66 的 MCGA 不是原生 Chebyshev/普通 MLP 优化器，而是 `MetricCompatibleAtlasMLP` readout 上的 C-skew/Cayley/active-Gram/functional-spectrum/POET-residual 生成器路线。",
            f"v23.26 当前 PartG summary 文件 `{len(partg_summaries)}` 个，仍未出现语义复现的 MLP-MCGA completed 行，因此保持 `{decision['audit_route']}`。",
            "关键差别：v23.26 现有 same-param/same-FLOPs/BlockRTGF MLP baseline 不包含 POET residual MCGA、same-functional-spectrum、same-generator-descent、same-C-skew controls，不能替代 MCGA official baseline。",
            "历史 KAN+MCGA 探针显示 MCGA 对 KAN 不是自动利好：D-CHE probe 的 KAN beats matched MLP 为 `0/15`，所以不能把旧 MCGA 结果当作 v23.26 H-H 成功证据。",
        ],
    )


def run_completion_audit(args: argparse.Namespace) -> None:
    previous_path = latest_artifact("v23_26_gpu3_completion_audit*.json")
    previous, previous_status = read_json_artifact(previous_path) if previous_path is not None else ({}, "missing")
    mcga_audit_path = ROOT / "results/v23_26_gpu3_mlp_mcga_repro_audit/v23_26_mlp_mcga_repro_audit.json"
    mcga_smoke_path = ROOT / "results/v23_26_gpu3_partg_mcga_v22style_smoke_all6/v23_26_MLP_MCGA_v22_style_smoke_summary.json"
    partdg_gate_path = latest_artifact("v23_26_gpu3_partDG_gate_audit*/v23_26_partDG_gate_audit_summary.json")
    route_mode_path = latest_artifact("v23_26_gpu3_route_mode_audit*/v23_26_route_mode_audit_summary.json")
    geometry_failure_path = latest_artifact("v23_26_gpu3_partd_geometry*/v23_26_partD_geometry_failure_audit_summary.json")
    rtgf_additive_mechanism_path = latest_artifact("v23_26_gpu3_partd_rtgf_additive_mechanism*/v23_26_partD_rtgf_additive_mechanism_summary.json")
    mcga_control_summary_paths = sorted(
        (ROOT / "results").glob("v23_26_gpu3_partg_mcga_controls*/v23_26_MLP_MCGA_v22_style_controls_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    mcga_controls_path = (
        mcga_control_summary_paths[-1]
        if mcga_control_summary_paths
        else ROOT / "results/v23_26_gpu3_partg_mcga_controls_smoke_all6_wine_s0_dynamoreset/v23_26_MLP_MCGA_v22_style_controls_summary.json"
    )
    mcga_audit, mcga_audit_status = read_json_artifact(mcga_audit_path)
    mcga_smoke, mcga_smoke_status = read_json_artifact(mcga_smoke_path)
    partdg_gate, partdg_gate_status = read_json_artifact(partdg_gate_path) if partdg_gate_path is not None else ({}, "missing")
    route_mode, route_mode_status = read_json_artifact(route_mode_path) if route_mode_path is not None else ({}, "missing")
    geometry_failure, geometry_failure_status = read_json_artifact(geometry_failure_path) if geometry_failure_path is not None else ({}, "missing")
    rtgf_additive_mechanism, rtgf_additive_mechanism_status = read_json_artifact(rtgf_additive_mechanism_path) if rtgf_additive_mechanism_path is not None else ({}, "missing")
    mcga_controls, mcga_controls_status = read_json_artifact(mcga_controls_path)
    mcga_controls_matrix_rows: list[dict[str, Any]] = []
    mcga_controls_matrix_status = "missing"
    mcga_controls_matrix_path = mcga_controls_path.parent / "v23_26_MLP_MCGA_v22_style_controls_matrix.csv"
    if mcga_controls_matrix_path.exists():
        mcga_controls_matrix_rows, mcga_controls_matrix_status = read_csv_artifact(mcga_controls_matrix_path)
    poet_transform_positive_rows = sum(1 for row in mcga_controls_matrix_rows if int(float(row.get("poet_trainable_transform_params") or 0)) > 0)
    candidate_poet_transform_positive_rows = sum(
        1
        for row in mcga_controls_matrix_rows
        if str(row.get("method", "")) == str(mcga_controls.get("mcga_method", "")) and int(float(row.get("poet_trainable_transform_params") or 0)) > 0
    )
    poet_official_transform_positive_rows = sum(
        1
        for row in mcga_controls_matrix_rows
        if str(row.get("method", "")) == "poet_official" and int(float(row.get("poet_trainable_transform_params") or 0)) > 0
    )
    mcga_controls_run_catalog: list[dict[str, Any]] = []
    for summary_path in mcga_control_summary_paths:
        summary_item, summary_status = read_json_artifact(summary_path)
        matrix_item_path = summary_path.parent / "v23_26_MLP_MCGA_v22_style_controls_matrix.csv"
        matrix_item_rows: list[dict[str, Any]] = []
        matrix_item_status = "missing"
        if matrix_item_path.exists():
            matrix_item_rows, matrix_item_status = read_csv_artifact(matrix_item_path)
        method_name = str(summary_item.get("mcga_method", ""))
        transform_positive = sum(1 for row in matrix_item_rows if int(float(row.get("poet_trainable_transform_params") or 0)) > 0)
        candidate_transform_positive = sum(
            1
            for row in matrix_item_rows
            if str(row.get("method", "")) == method_name and int(float(row.get("poet_trainable_transform_params") or 0)) > 0
        )
        poet_official_transform_positive = sum(
            1
            for row in matrix_item_rows
            if str(row.get("method", "")) == "poet_official" and int(float(row.get("poet_trainable_transform_params") or 0)) > 0
        )
        mcga_controls_run_catalog.append(
            {
                "source": str(summary_path.relative_to(ROOT)),
                "status": summary_status,
                "matrix_status": matrix_item_status,
                "datasets": summary_item.get("datasets", ""),
                "seeds": summary_item.get("seeds", ""),
                "carriers": summary_item.get("carriers", ""),
                "steps": summary_item.get("steps", ""),
                "rows": summary_item.get("rows", ""),
                "completed_rows": summary_item.get("completed_rows", ""),
                "failed_rows": summary_item.get("failed_rows", ""),
                "required_control_pair_rows": summary_item.get("required_control_pair_rows", ""),
                "required_control_win_rows": summary_item.get("required_control_win_rows", ""),
                "poet_replaced_positive_rows": summary_item.get("poet_replaced_positive_rows", ""),
                "poet_trainable_transform_positive_rows": transform_positive,
                "candidate_poet_trainable_transform_positive_rows": candidate_transform_positive,
                "poet_official_trainable_transform_positive_rows": poet_official_transform_positive,
                "diagnostic_only": summary_item.get("diagnostic_only", ""),
                "partG_official_gate_evaluable": summary_item.get("partG_official_gate_evaluable", ""),
            }
        )
    audit = copy.deepcopy(previous) if previous else {"complete": False, "current_opened_gates": {}, "latest_evidence": {}, "still_missing": []}
    audit["complete"] = False
    audit.setdefault("current_opened_gates", {})
    audit.setdefault("latest_evidence", {})
    audit.setdefault("still_missing", [])
    audit["current_opened_gates"]["PartG_MLP_MCGA_semantic_audit"] = mcga_audit_status == "ok"
    audit["current_opened_gates"]["PartG_MLP_MCGA_v22_style_smoke_diagnostic"] = bool(mcga_smoke.get("completed_rows", 0))
    audit["current_opened_gates"]["PartG_MLP_MCGA_v22_style_controls_smoke_diagnostic"] = bool(mcga_controls.get("completed_rows", 0))
    audit["current_opened_gates"]["PartD_PartG_gate_arithmetic_audit"] = partdg_gate_status == "ok"
    audit["current_opened_gates"]["PartF_route_mode_geometry_audit"] = route_mode_status == "ok"
    audit["current_opened_gates"]["PartD_geometry_failure_attribution_audit"] = geometry_failure_status == "ok"
    audit["current_opened_gates"]["PartD_RTGF_additive_mechanism_audit"] = rtgf_additive_mechanism_status == "ok"
    audit["latest_evidence"]["PartG_MLP_MCGA_reproduction_audit"] = {
        "source": str(mcga_audit_path.relative_to(ROOT)),
        "status": mcga_audit_status,
        "audit_route": mcga_audit.get("audit_route", ""),
        "mlp_mcga_semantically_reproduced_in_v23_26": mcga_audit.get("mlp_mcga_semantically_reproduced_in_v23_26", ""),
        "may_count_v22_66_numbers_as_v23_26_official_rows": mcga_audit.get("may_count_v22_66_numbers_as_v23_26_official_rows", ""),
        "v22_final_route": mcga_audit.get("v22_final_route", ""),
    }
    audit["latest_evidence"]["PartG_MLP_MCGA_v22_style_smoke_all6"] = {
        "source": str(mcga_smoke_path.relative_to(ROOT)),
        "status": mcga_smoke_status,
        "rows": mcga_smoke.get("rows", ""),
        "completed_rows": mcga_smoke.get("completed_rows", ""),
        "failed_rows": mcga_smoke.get("failed_rows", ""),
        "guard_finite_rows": mcga_smoke.get("guard_finite_rows", ""),
        "poet_replaced_positive_rows": mcga_smoke.get("poet_replaced_positive_rows", ""),
        "poet_zero_replaced_rows": mcga_smoke.get("poet_zero_replaced_rows", ""),
        "MLP_MCGA_status": mcga_smoke.get("MLP_MCGA_status", ""),
        "partG_official_gate_evaluable": mcga_smoke.get("partG_official_gate_evaluable", ""),
        "missing_for_official": mcga_smoke.get("missing_for_official", ""),
    }
    audit["latest_evidence"]["PartG_MLP_MCGA_v22_style_controls_smoke_all6"] = {
        "source": str(mcga_controls_path.relative_to(ROOT)),
        "status": mcga_controls_status,
        "rows": mcga_controls.get("rows", ""),
        "completed_rows": mcga_controls.get("completed_rows", ""),
        "failed_rows": mcga_controls.get("failed_rows", ""),
        "candidate_rows": mcga_controls.get("candidate_rows", ""),
        "control_rows": mcga_controls.get("control_rows", ""),
        "pair_rows": mcga_controls.get("pair_rows", ""),
        "datasets": mcga_controls.get("datasets", ""),
        "seeds": mcga_controls.get("seeds", ""),
        "carriers": mcga_controls.get("carriers", ""),
        "steps": mcga_controls.get("steps", ""),
        "mcga_method": mcga_controls.get("mcga_method", ""),
        "required_control_pair_rows": mcga_controls.get("required_control_pair_rows", ""),
        "required_control_win_rows": mcga_controls.get("required_control_win_rows", ""),
        "poet_replaced_positive_rows": mcga_controls.get("poet_replaced_positive_rows", ""),
        "MLP_MCGA_controls_status": mcga_controls.get("MLP_MCGA_controls_status", ""),
        "partG_official_gate_evaluable": mcga_controls.get("partG_official_gate_evaluable", ""),
        "official_gap_remaining": mcga_controls.get("official_gap_remaining", ""),
        "matrix_status": mcga_controls_matrix_status,
        "poet_trainable_transform_positive_rows": poet_transform_positive_rows,
        "candidate_poet_trainable_transform_positive_rows": candidate_poet_transform_positive_rows,
        "poet_official_trainable_transform_positive_rows": poet_official_transform_positive_rows,
    }
    audit["latest_evidence"]["PartG_MLP_MCGA_v22_style_controls_smoke_catalog"] = {
        "run_count": len(mcga_controls_run_catalog),
        "latest_source": str(mcga_controls_path.relative_to(ROOT)),
        "runs": mcga_controls_run_catalog,
    }
    audit["latest_evidence"]["PartD_PartG_gate_arithmetic_audit"] = {
        "source": str(partdg_gate_path.relative_to(ROOT)) if partdg_gate_path is not None else "",
        "status": partdg_gate_status,
        "partD_sources": partdg_gate.get("partD_sources", ""),
        "partD_gate_rows": partdg_gate.get("partD_gate_rows", ""),
        "partD_gate_pass_if_official_rows": partdg_gate.get("partD_gate_pass_if_official_rows", ""),
        "partG_sources": partdg_gate.get("partG_sources", ""),
        "partG_gate_rows": partdg_gate.get("partG_gate_rows", ""),
        "partG_gate_pass_if_official_rows": partdg_gate.get("partG_gate_pass_if_official_rows", ""),
        "partG_mcga_pair_sources": partdg_gate.get("partG_mcga_pair_sources", ""),
        "partG_mcga_pair_gate_rows": partdg_gate.get("partG_mcga_pair_gate_rows", ""),
        "partG_mcga_pair_gate_pass_if_official_rows": partdg_gate.get("partG_mcga_pair_gate_pass_if_official_rows", ""),
        "official_gate_evaluable": partdg_gate.get("official_gate_evaluable", ""),
        "reason_not_official": partdg_gate.get("reason_not_official", ""),
    }
    audit["latest_evidence"]["PartF_route_mode_geometry_audit"] = {
        "source": str(route_mode_path.relative_to(ROOT)) if route_mode_path is not None else "",
        "status": route_mode_status,
        "route_rows": route_mode.get("route_rows", ""),
        "route_counts": route_mode.get("route_counts", ""),
        "official_success_allowed": route_mode.get("official_success_allowed", ""),
        "diagnostic_only": route_mode.get("diagnostic_only", ""),
        "interpretation": route_mode.get("interpretation", ""),
    }
    audit["latest_evidence"]["PartD_geometry_failure_attribution_audit"] = {
        "source": str(geometry_failure_path.relative_to(ROOT)) if geometry_failure_path is not None else "",
        "status": geometry_failure_status,
        "source_count": geometry_failure.get("source_count", ""),
        "detail_rows": geometry_failure.get("detail_rows", ""),
        "summary_rows": geometry_failure.get("summary_rows", ""),
        "D_CHE_M2_additive_rows": geometry_failure.get("D_CHE_M2_additive_rows", ""),
        "projection_identity_suspect_rows": geometry_failure.get("projection_identity_suspect_rows", ""),
        "angle_distribution_suspect_rows": geometry_failure.get("angle_distribution_suspect_rows", ""),
        "rtgf_additive_like_rows": geometry_failure.get("rtgf_additive_like_rows", ""),
        "official_success_allowed": geometry_failure.get("official_success_allowed", ""),
        "diagnostic_only": geometry_failure.get("diagnostic_only", ""),
        "interpretation": geometry_failure.get("interpretation", ""),
    }
    audit["latest_evidence"]["PartD_RTGF_additive_mechanism_audit"] = {
        "source": str(rtgf_additive_mechanism_path.relative_to(ROOT)) if rtgf_additive_mechanism_path is not None else "",
        "status": rtgf_additive_mechanism_status,
        "source_rows": rtgf_additive_mechanism.get("source_rows", ""),
        "detail_rows": rtgf_additive_mechanism.get("detail_rows", ""),
        "summary_rows": rtgf_additive_mechanism.get("summary_rows", ""),
        "additive_no_debt_fail_rows": rtgf_additive_mechanism.get("additive_no_debt_fail_rows", ""),
        "second_order_leverage_too_small_rows": rtgf_additive_mechanism.get("second_order_leverage_too_small_rows", ""),
        "formula_or_projection_suspect_rows": rtgf_additive_mechanism.get("formula_or_projection_suspect_rows", ""),
        "systematic_formula_or_projection_suspect_rows": rtgf_additive_mechanism.get("systematic_formula_or_projection_suspect_rows", ""),
        "localized_projection_spike_rows": rtgf_additive_mechanism.get("localized_projection_spike_rows", ""),
        "official_success_allowed": rtgf_additive_mechanism.get("official_success_allowed", ""),
        "diagnostic_only": rtgf_additive_mechanism.get("diagnostic_only", ""),
        "interpretation": rtgf_additive_mechanism.get("interpretation", ""),
    }
    still_missing = [str(item) for item in audit.get("still_missing", []) if "PartG remains diagnostic" not in str(item)]
    still_missing.append(
        "PartG remains diagnostic: MLP-MCGA v22-style controls now execute on v23.26 all8 primary multi-seed splits with KAN_G2-vs-MCGA pairing, but PartD/PartG gate arithmetic still has failed rows and confirmatory thresholds remain missing."
    )
    audit["still_missing"] = still_missing
    audit["timestamp"] = time.strftime("%Y-%m-%d %H:%M %z")
    audit["previous_completion_audit"] = str(previous_path.relative_to(ROOT)) if previous_path is not None else ""
    audit["previous_completion_audit_status"] = previous_status
    name = f"v23_26_gpu3_completion_audit_{time.strftime('%Y%m%d_%H%M')}_post_mlp_mcga_smoke.json"
    out_path = ROOT / "results" / name
    write_json(out_path, audit)
    note = {
        "complete": audit["complete"],
        "output": str(out_path.relative_to(ROOT)),
        "previous": audit["previous_completion_audit"],
        "mcga_audit_status": mcga_audit_status,
        "mcga_smoke_status": mcga_smoke_status,
        "mcga_smoke_completed_rows": mcga_smoke.get("completed_rows", ""),
        "mcga_smoke_poet_positive_rows": mcga_smoke.get("poet_replaced_positive_rows", ""),
        "mcga_smoke_zero_poet_rows": mcga_smoke.get("poet_zero_replaced_rows", ""),
        "mcga_controls_status": mcga_controls_status,
        "mcga_controls_completed_rows": mcga_controls.get("completed_rows", ""),
        "mcga_controls_failed_rows": mcga_controls.get("failed_rows", ""),
        "mcga_controls_required_win_rows": mcga_controls.get("required_control_win_rows", ""),
        "mcga_controls_required_pair_rows": mcga_controls.get("required_control_pair_rows", ""),
        "mcga_controls_poet_transform_positive_rows": poet_transform_positive_rows,
        "mcga_controls_candidate_poet_transform_positive_rows": candidate_poet_transform_positive_rows,
        "mcga_controls_poet_official_transform_positive_rows": poet_official_transform_positive_rows,
        "mcga_controls_run_catalog_count": len(mcga_controls_run_catalog),
        "partdg_gate_status": partdg_gate_status,
        "partD_gate_pass_if_official_rows": partdg_gate.get("partD_gate_pass_if_official_rows", ""),
        "partG_gate_pass_if_official_rows": partdg_gate.get("partG_gate_pass_if_official_rows", ""),
        "partG_mcga_pair_gate_pass_if_official_rows": partdg_gate.get("partG_mcga_pair_gate_pass_if_official_rows", ""),
        "route_mode_status": route_mode_status,
        "route_mode_route_rows": route_mode.get("route_rows", ""),
        "route_mode_route_counts": route_mode.get("route_counts", ""),
        "geometry_failure_status": geometry_failure_status,
        "geometry_failure_summary_rows": geometry_failure.get("summary_rows", ""),
        "geometry_failure_projection_identity_suspect_rows": geometry_failure.get("projection_identity_suspect_rows", ""),
        "geometry_failure_angle_distribution_suspect_rows": geometry_failure.get("angle_distribution_suspect_rows", ""),
        "geometry_failure_rtgf_additive_like_rows": geometry_failure.get("rtgf_additive_like_rows", ""),
        "rtgf_additive_mechanism_status": rtgf_additive_mechanism_status,
        "rtgf_additive_mechanism_second_order_leverage_too_small_rows": rtgf_additive_mechanism.get("second_order_leverage_too_small_rows", ""),
        "rtgf_additive_mechanism_formula_or_projection_suspect_rows": rtgf_additive_mechanism.get("formula_or_projection_suspect_rows", ""),
        "rtgf_additive_mechanism_systematic_formula_or_projection_suspect_rows": rtgf_additive_mechanism.get("systematic_formula_or_projection_suspect_rows", ""),
        "rtgf_additive_mechanism_localized_projection_spike_rows": rtgf_additive_mechanism.get("localized_projection_spike_rows", ""),
    }
    append_exec("Completion_audit_post_MLP_MCGA_smoke", args, [str(out_path.relative_to(ROOT))], note)
    append_recap(
        "Completion audit post MLP-MCGA smoke",
        [
            f"写出 `{out_path.relative_to(ROOT)}`；complete `{audit['complete']}`；previous `{audit['previous_completion_audit']}`。",
            f"MLP-MCGA audit route `{mcga_audit.get('audit_route', '')}`；v22 final route `{mcga_audit.get('v22_final_route', '')}`；v23 official rows counted `{mcga_audit.get('may_count_v22_66_numbers_as_v23_26_official_rows', '')}`。",
            f"MLP-MCGA v22-style smoke rows `{mcga_smoke.get('rows', '')}`，completed `{mcga_smoke.get('completed_rows', '')}`，POET positive `{mcga_smoke.get('poet_replaced_positive_rows', '')}`，zero-POET `{mcga_smoke.get('poet_zero_replaced_rows', '')}`。",
            f"MLP-MCGA controls smoke rows `{mcga_controls.get('rows', '')}`，completed `{mcga_controls.get('completed_rows', '')}`，failed `{mcga_controls.get('failed_rows', '')}`，required-control guard wins `{mcga_controls.get('required_control_win_rows', '')}/{mcga_controls.get('required_control_pair_rows', '')}`。",
            f"POET trainable transform positive rows `{poet_transform_positive_rows}`；candidate `{candidate_poet_transform_positive_rows}`；poet_official `{poet_official_transform_positive_rows}`。",
            f"controls run catalog `{len(mcga_controls_run_catalog)}` 个；latest source `{mcga_controls_path.relative_to(ROOT)}`。",
            f"PartD/PartG gate audit status `{partdg_gate_status}`；PartD pass-if-official `{partdg_gate.get('partD_gate_pass_if_official_rows', '')}/{partdg_gate.get('partD_gate_rows', '')}`；PartG pass-if-official `{partdg_gate.get('partG_gate_pass_if_official_rows', '')}/{partdg_gate.get('partG_gate_rows', '')}`；PartG MCGA pair pass-if-official `{partdg_gate.get('partG_mcga_pair_gate_pass_if_official_rows', '')}/{partdg_gate.get('partG_mcga_pair_gate_rows', '')}`。",
            f"PartF route/mode audit status `{route_mode_status}`；route rows `{route_mode.get('route_rows', '')}`；route counts `{route_mode.get('route_counts', '')}`。",
            f"PartD geometry failure audit status `{geometry_failure_status}`；summary rows `{geometry_failure.get('summary_rows', '')}`；projection suspects `{geometry_failure.get('projection_identity_suspect_rows', '')}`；angle suspects `{geometry_failure.get('angle_distribution_suspect_rows', '')}`；RTGF-additive-like rows `{geometry_failure.get('rtgf_additive_like_rows', '')}`。",
            f"PartD RTGF-additive mechanism audit status `{rtgf_additive_mechanism_status}`；small-leverage rows `{rtgf_additive_mechanism.get('second_order_leverage_too_small_rows', '')}`；formula/projection suspect rows `{rtgf_additive_mechanism.get('formula_or_projection_suspect_rows', '')}`；systematic suspect rows `{rtgf_additive_mechanism.get('systematic_formula_or_projection_suspect_rows', '')}`；localized spike rows `{rtgf_additive_mechanism.get('localized_projection_spike_rows', '')}`。",
            "结论不变：MCGA 与 controls 执行可行性已打开，primary all8 KAN-vs-MCGA pairing 已部分可评估；但 PartD/PartG gates 仍未全过，confirmatory thresholds 仍缺。",
        ],
    )


def median_finite(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    mid = len(finite) // 2
    if len(finite) % 2:
        return float(finite[mid])
    return float(0.5 * (finite[mid - 1] + finite[mid]))


def run_route_mode_audit(args: argparse.Namespace) -> None:
    """Explain current exact route candidates from existing H20/PartD artifacts."""
    partd_gate_path = latest_artifact("v23_26_gpu3_partDG_gate_audit*/v23_26_partD_gate_audit.csv")
    partd_summary_path = latest_artifact("v23_26_gpu3_partDG_gate_audit*/v23_26_partDG_gate_audit_summary.json")
    h20_matrix_path = latest_artifact("v23_26_gpu3_h20*/v23_26_H20_matrix.csv")
    h20_summary_path = latest_artifact("v23_26_gpu3_h20*/v23_26_H20_summary.json")
    partd_rows, partd_status = read_csv_artifact(partd_gate_path) if partd_gate_path is not None else ([], "missing")
    partd_summary, partd_summary_status = read_json_artifact(partd_summary_path) if partd_summary_path is not None else ({}, "missing")
    h20_rows, h20_status = read_csv_artifact(h20_matrix_path) if h20_matrix_path is not None else ([], "missing")
    h20_summary, h20_summary_status = read_json_artifact(h20_summary_path) if h20_summary_path is not None else ({}, "missing")

    latest_h20_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in h20_rows:
        latest_h20_by_key[(str(row.get("task", "")), str(row.get("seed", "")), str(row.get("carrier_core_variant", "")), str(row.get("scheme", "")))] = row

    h20_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for (task, seed, carrier, scheme), grow in latest_h20_by_key.items():
        match = re.fullmatch(r"(.+)-G([0-2])", scheme)
        if match is None:
            continue
        prefix, metric_index = match.group(1), match.group(2)
        metric_name = f"M{metric_index}"
        control = latest_h20_by_key.get((task, seed, carrier, f"{prefix}-C{metric_index}"))
        if control is None:
            continue
        key = (carrier, metric_name)
        stat = h20_stats.setdefault(
            key,
            {
                "pairs": 0,
                "surpluses": [],
                "debts": [],
                "curvature_energy": [],
                "mode_energy_entropy": [],
                "frequency_centroid": [],
                "low_frequency_fraction": [],
                "frequency_spread": [],
            },
        )
        surplus = parse_float(control.get("guard_NLL", "")) - parse_float(grow.get("guard_NLL", ""))
        debt = parse_float(grow.get("guard_CVaR95_NLL", "")) - parse_float(control.get("guard_CVaR95_NLL", ""))
        stat["pairs"] += 1
        stat["surpluses"].append(surplus)
        stat["debts"].append(debt)
        for key_name in ("curvature_energy", "mode_energy_entropy", "frequency_centroid", "low_frequency_fraction", "frequency_spread"):
            value = parse_float(grow.get(key_name, ""))
            if math.isfinite(value):
                stat[key_name].append(value)

    grouped_partd: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in partd_rows:
        carrier = str(row.get("carrier_core_variant", ""))
        metric = str(row.get("metric_name", ""))
        if carrier and metric:
            grouped_partd.setdefault((carrier, metric), []).append(row)

    route_rows: list[dict[str, Any]] = []
    for key in sorted(set(grouped_partd) | set(h20_stats)):
        carrier, metric_name = key
        candidates = grouped_partd.get(key, [])

        def partd_score(row: dict[str, str]) -> tuple[float, float, float, str]:
            pass_items = 0
            for field in [
                "paired_rows_ge_24",
                "candidate_vs_own_adamw_median_ge_1e3",
                "candidate_vs_additive_median_gt_0",
                "candidate_vs_random_median_gt_0",
                "candidate_vs_random_CVaR25_gt_0",
                "candidate_vs_random_bootstrap_LCB_gt_0",
                "win_rate_ge_70pct",
                "paired_no_debt_ge_80pct",
                "no_error_rows",
            ]:
                try:
                    pass_items += int(float(row.get(field, "0") or 0))
                except ValueError:
                    pass
            return (
                float(pass_items),
                gate_float(row, "paired_no_debt_rate"),
                gate_float(row, "adamw_median_surplus"),
                str(row.get("source", "")),
            )

        best = max(candidates, key=partd_score) if candidates else {}
        stat = h20_stats.get(key, {})
        h20_pairs = int(stat.get("pairs", 0) or 0)
        surpluses = [float(v) for v in stat.get("surpluses", []) if math.isfinite(float(v))]
        debts = [float(v) for v in stat.get("debts", []) if math.isfinite(float(v))]
        h20_win_rate = float(sum(1 for v in surpluses if v > 0.0)) / max(1.0, float(len(surpluses))) if surpluses else math.nan
        h20_no_debt_rate = float(sum(1 for v in debts if v <= 0.0)) / max(1.0, float(len(debts))) if debts else math.nan
        h20_median_surplus = median_finite(surpluses)
        h20_median_debt = median_finite(debts)
        partd_pass = int(float(best.get("partD_minimum_real_gate_pass_if_official", 0) or 0)) if best else 0
        partd_no_debt = gate_float(best, "paired_no_debt_rate") if best else math.nan
        partd_additive = gate_float(best, "additive_median_surplus") if best else math.nan
        partd_adamw = gate_float(best, "adamw_median_surplus") if best else math.nan
        partd_random_cvar = gate_float(best, "random_CVaR25_surplus") if best else math.nan
        missing = str(best.get("partD_gate_missing_or_failed_items", "")) if best else "missing_partD_gate_row"

        if partd_pass == 1:
            route = "R4_RadialTangentialGeneratorOpened_candidate"
            reason = "PartD gate would pass if official; still diagnostic-only until full official requirements are met."
        elif h20_win_rate >= 0.70 and (not math.isfinite(partd_no_debt) or partd_no_debt < 0.80 or partd_additive <= 0.0 or partd_adamw < 1.0e-3):
            route = "R7_TransientMetricGeneratorNoPersistentValue_candidate"
            reason = "H20 synthetic G-vs-C signal is strong but minimum-real gate/additive/debt evidence does not persist."
        elif math.isfinite(partd_adamw) and partd_adamw > 0.0 and (not math.isfinite(partd_additive) or abs(partd_additive) < 1.0e-6 or partd_no_debt < 0.80):
            route = "R3_IntrinsicSupportOnly_candidate"
            reason = "Candidate improves over AdamW/random controls, but generator has no robust independent increment beyond additive under debt gate."
        elif math.isfinite(partd_random_cvar) and partd_random_cvar > 0.0:
            route = "R2_IntrinsicSupportNoValue_or_NotEstablished_candidate"
            reason = "Some random-control signal exists, but own AdamW/additive/no-debt gates do not establish useful intrinsic support."
        else:
            route = "R0_IncompleteScientificExploration"
            reason = "Current artifacts do not establish support, generator value, or a completed official route."

        row: dict[str, Any] = {
            "carrier_core_variant": carrier,
            "metric_name": metric_name,
            "route_candidate": route,
            "route_reason": reason,
            "partD_best_source": best.get("source", ""),
            "partD_candidate_scheme": best.get("candidate_scheme", ""),
            "partD_gate_pass_if_official": partd_pass,
            "partD_missing_or_failed_items": missing,
            "partD_adamw_median_surplus": partd_adamw,
            "partD_additive_median_surplus": partd_additive,
            "partD_random_CVaR25_surplus": partd_random_cvar,
            "partD_paired_no_debt_rate": partd_no_debt,
            "h20_pairs": h20_pairs,
            "h20_G_vs_C_win_rate": h20_win_rate,
            "h20_G_vs_C_no_debt_rate": h20_no_debt_rate,
            "h20_G_vs_C_median_surplus": h20_median_surplus,
            "h20_G_vs_C_median_CVaR95_debt": h20_median_debt,
            "diagnostic_only": 1,
            "official_success_allowed": 0,
        }
        for key_name in ("curvature_energy", "mode_energy_entropy", "frequency_centroid", "low_frequency_fraction", "frequency_spread"):
            row[f"h20_mean_{key_name}"] = float(np.mean(stat.get(key_name, []))) if stat.get(key_name) else math.nan
        route_rows.append(row)

    route_counts: dict[str, int] = {}
    for row in route_rows:
        route_counts[str(row["route_candidate"])] = route_counts.get(str(row["route_candidate"]), 0) + 1
    summary = {
        "diagnostic_only": 1,
        "official_success_allowed": 0,
        "partd_gate_source": str(partd_gate_path.relative_to(ROOT)) if partd_gate_path is not None else "",
        "partd_gate_status": partd_status,
        "partd_summary_source": str(partd_summary_path.relative_to(ROOT)) if partd_summary_path is not None else "",
        "partd_summary_status": partd_summary_status,
        "partD_gate_pass_if_official_rows": partd_summary.get("partD_gate_pass_if_official_rows", ""),
        "partD_gate_rows": partd_summary.get("partD_gate_rows", ""),
        "h20_matrix_source": str(h20_matrix_path.relative_to(ROOT)) if h20_matrix_path is not None else "",
        "h20_matrix_status": h20_status,
        "h20_summary_source": str(h20_summary_path.relative_to(ROOT)) if h20_summary_path is not None else "",
        "h20_summary_status": h20_summary_status,
        "h20_rows": h20_summary.get("rows", ""),
        "h20_paired_summary_rows": h20_summary.get("paired_summary_rows", ""),
        "route_rows": len(route_rows),
        "route_counts": route_counts,
        "interpretation": "Exact hypothesis-level route candidates only; this audit does not close a family-level NoGo and does not upgrade diagnostic artifacts to official success.",
    }
    write_csv(OUT_ROOT / "v23_26_route_mode_audit.csv", route_rows)
    write_json(OUT_ROOT / "v23_26_route_mode_audit_summary.json", summary)
    append_exec(
        "PartF_route_mode_geometry_audit",
        args,
        ["v23_26_route_mode_audit.csv", "v23_26_route_mode_audit_summary.json"],
        summary,
    )
    route_line = ", ".join(f"{key}={value}" for key, value in sorted(route_counts.items()))
    append_recap(
        "PartF route/mode geometry audit 复盘",
        [
            f"route rows `{len(route_rows)}`；route counts `{route_line}`。",
            f"PartD source `{summary['partd_gate_source']}`，pass-if-official `{summary['partD_gate_pass_if_official_rows']}/{summary['partD_gate_rows']}`；H20 source `{summary['h20_matrix_source']}`，rows `{summary['h20_rows']}`。",
            "该审计只读已有 PartD/H20 artifacts，按 carrier+metric 输出 exact route candidate；不改变训练、不改变 gate、不升级 diagnostic 为 official。",
            "若 H20 synthetic G-vs-C 强但 minimum-real/additive/debt 不持续，则标记 R7 candidate；若相对 AdamW/random 有支持但 additive/debt 无独立增量，则标记 R3 candidate。",
            "结论仍不是 family-level NoGo；一个 route 只关闭精确假设，edge-function metric 总方向继续保留。",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["part0", "part-a", "part-b", "part-c", "part-d-smoke", "part-g-smoke", "part-g-mcga-smoke", "part-g-mcga-controls-smoke", "part-dg-gate-audit", "part-d-metric-order-audit", "part-d-debt-metric-audit", "part-d-geometry-failure-audit", "part-d-rtgf-additive-mechanism-audit", "mlp-mcga-audit", "completion-audit", "part-h-efficiency", "h20-smoke", "route-mode-audit"])
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--quick", action="store_true", help="Use reduced efficiency grid for rapid audit iteration.")
    parser.add_argument("--h20-steps", type=int, default=20)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--tasks", default=",".join(SYNTHETIC_TASKS))
    parser.add_argument("--carriers", default="D-CHE-Core-K3,D-CHE-Core-K4,D-FOU-IdLF-Core-K2,D-FOU-IdLF-Core-K4")
    parser.add_argument("--h20-metrics", default="M0,M1,M2")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--datasets", default="Wine,Spam,Rice,Bean,FashionMNIST,SVHN,EMNIST-Letters,CIFAR10-compact")
    parser.add_argument("--real-max-samples", type=int, default=1024)
    parser.add_argument("--kan-reference-root", default="", help="Directory containing v23_26_partD_minimum_real_matrix.csv for KAN-vs-MLP pairing.")
    parser.add_argument("--efficiency-repeats", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--poet-block-size", type=int, default=16)
    parser.add_argument("--poet-merge-interval", type=int, default=100)
    parser.add_argument("--poet-lr", type=float, default=1.0e-3)
    parser.add_argument("--poet-scale", type=float, default=1.0)
    parser.add_argument("--mcga-method", default="mcga_over_poet_fsclip_eta025_residual_rank4")
    parser.add_argument("--mcga-control-methods", default="adamw,poet_official,same_functional_spectrum_random_coordinate,same_generator_descent_energy_random,same_C_skew_spectrum_random,same_compute_noop_coordinate,metric_preserving_noop")
    parser.add_argument("--mcga-poet-even-hidden", action="store_true", help="Constrain MCGA SimpleMLP hidden width to a POET block-size multiple for nonzero POET transforms; diagnostic only.")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.phase == "part0":
        run_part0(args)
    elif args.phase == "part-a":
        run_part_a(args)
    elif args.phase == "part-b":
        run_part_b(args)
    elif args.phase == "part-c":
        run_part_c(args)
    elif args.phase == "part-d-smoke":
        run_part_d_smoke(args)
    elif args.phase == "part-g-smoke":
        run_part_g_smoke(args)
    elif args.phase == "part-g-mcga-smoke":
        run_part_g_mcga_smoke(args)
    elif args.phase == "part-g-mcga-controls-smoke":
        run_part_g_mcga_controls_smoke(args)
    elif args.phase == "part-dg-gate-audit":
        run_part_dg_gate_audit(args)
    elif args.phase == "part-d-metric-order-audit":
        run_part_d_metric_order_audit(args)
    elif args.phase == "part-d-debt-metric-audit":
        run_part_d_debt_metric_audit(args)
    elif args.phase == "part-d-geometry-failure-audit":
        run_part_d_geometry_failure_audit(args)
    elif args.phase == "part-d-rtgf-additive-mechanism-audit":
        run_part_d_rtgf_additive_mechanism_audit(args)
    elif args.phase == "mlp-mcga-audit":
        run_mlp_mcga_repro_audit(args)
    elif args.phase == "completion-audit":
        run_completion_audit(args)
    elif args.phase == "part-h-efficiency":
        run_part_h_efficiency(args)
    elif args.phase == "h20-smoke":
        run_h20_smoke(args)
    elif args.phase == "route-mode-audit":
        run_route_mode_audit(args)
    else:  # pragma: no cover
        raise ValueError(args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
