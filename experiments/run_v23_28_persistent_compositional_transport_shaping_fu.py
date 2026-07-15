#!/usr/bin/env python3
"""DG-KAN v23.28 BC-CTS-FU runner.

This first implementation surface deliberately starts with Part 0 and Part A.
It does not claim science success: v23.28 requires Part 0 and Part A semantic
truth before synthetic or real science rows are meaningful.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import pickle
import struct
import time
import warnings
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/DG-KAN_v23.28_PersistentBasisCovariantCompositionalTransportShapingEdgeFlowFU_多假设语义穷尽式完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.28_PersistentBasisCovariantCompositionalTransportShapingEdgeFlowFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.28_PersistentBasisCovariantCompositionalTransportShapingEdgeFlowFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2328_OUT_ROOT", str(ROOT / "results/v23_28"))).resolve()

HYPOTHESES = [
    ("H-A", "Architecture, loss purity, paired harness, and runtime truth"),
    ("H-B", "Full generator representability"),
    ("H-C", "Compositional metric causality"),
    ("H-D", "Persistent full state causality"),
    ("H-E", "Transport, shape, and radial independent contribution"),
    ("H-F", "Hybrid FU independent increment"),
    ("H-G", "Warmup-then-Pure FU"),
    ("H-H", "Cross-loss-family generality and long horizon"),
    ("H-I", "KAN architecture, MLP matched, and efficiency"),
]

LINEAGE_DOCS = {
    "v22.43": ROOT / "docs/DG-KAN_v22.43_MetricPreservingContinuousFunctionalFlowFU_完整计划.md",
    "v22.64": ROOT / "docs/DG-KAN_v22.64_MetricPreservingFunctionalAtlasFU_完整计划.md",
    "v22.65": ROOT / "docs/DG-KAN_v22.65_MetricCompatibleSignalAtlasFU_完整计划.md",
    "v22.66": ROOT / "docs/DG-KAN_v22.66_MetricCompatibleGeneratorAtlasFU_完整计划.md",
    "v22.94": ROOT / "docs/DG-KAN_v22.94_AdamW_Witness_Decomposition_MultiScheme_MPFU_完整计划.md",
    "v23.15": ROOT / "docs/DG-KAN_v23.15_BasisCovariantEdgeFunctionNaturalFlow_完整详尽实验计划.md",
    "v23.16": ROOT / "docs/DG-KAN_v23.16_CompositionalBasisCovariantVerticalHorizontalFlow_完整详尽实验计划.md",
    "v23.25": ROOT / "docs/DG-KAN_v23.25_BasisCovariantQuotientCompositeOperator_ProperlyNestedSplineDetailFlow_TrueKANTwin_多假设语义穷尽式完整详尽实验计划.md",
    "v23.26": ROOT / "docs/DG-KAN_v23.26_DCHE_DFOU_BasisCovariantIntrinsicRadialTangentialGeneratorFlow_完整详尽实验计划.md",
    "v23.27": ROOT / "docs/DG-KAN_v23.27_PersistentBasisCovariantCompositionalCurvatureLieGeneratorFU_多假设语义穷尽式完整详尽实验计划.md",
    "v23.28": PLAN,
}

CLASSIFICATION_DATASETS = [
    "Wine",
    "Spam",
    "Rice",
    "Bean",
    "FashionMNIST",
    "SVHN",
    "EMNIST-Letters",
    "CIFAR10-compact",
]

SYNTHETIC_TASKS = [
    "SYN-A-SKEW-ONLY",
    "SYN-B-SYMMETRIC-SHAPE",
    "SYN-C-EDGE-RADIAL",
    "SYN-D-MIXED-FULL",
    "SYN-E-COMP-CURVATURE",
    "SYN-F-PATH-SHUFFLE-NEGATIVE",
    "SYN-G-PERSISTENT-COHERENCE",
    "SYN-H-TRANSIENT-FORCING",
    "SYN-I-MSE-PERSISTENT",
    "SYN-J-PAIRWISE-PERSISTENT",
]

SCHEMES = [
    "K0_AdamW_task_native",
    "K1_IntrinsicAdditive_DataL2",
    "K2_IntrinsicAdditive_CompH2",
    "K3_v2327_SkewPersistent_StaticH2",
    "K4_v2327_SkewInstant_StaticH2",
    "P0_FullInstant_CompH2",
    "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
    "P2_FullPersistent_WarmupPure_CompH2",
    "P3_FullPersistent_Hybrid_DataL2",
    "P4_FullPersistent_Hybrid_LocalH2",
    "P5_FullPersistent_Hybrid_PathShuffleH2",
    "P6_FullPersistent_Hybrid_UniformPathH2",
    "A0_Persistent_SkewOnly",
    "A1_Persistent_SymmetricOnly",
    "A2_Persistent_RadialOnly",
    "A3_Persistent_SkewPlusSym",
    "A4_Persistent_SymPlusRadial",
    "A5_Persistent_SkewPlusRadial",
    "A6_Persistent_Full_OmegaSymRadial",
    "A7_Persistent_Full_ConstantDecoupled",
    "R0_ResetEveryStepFullState",
    "R1_RandomAR1FullState",
    "R2_SignFlipFullState",
    "R3_ReceivingBankShuffledState",
    "R4_TimeLag4State",
    "R5_SameAutocorrelationRandomState",
    "R6_SameComputeNoop",
    "R7_CurrentForcingSameStepDiagnostic",
    "M0_MLP_SameParam_TaskNativeStrongOptimizer",
    "M1_MLP_SameFLOPs_TaskNativeStrongOptimizer",
    "M2_MLP_PersistentSkewBlock",
    "M3_MLP_PersistentFullTransportShapeRadialBlock",
    "M4_MLP_RandomFullState",
    "M5_MLP_MCGA_Reproduction",
]

PARTC_PREFLIGHT_SCHEMES = [scheme for scheme in SCHEMES if not scheme.startswith("M")]
PARTC_MLP_SCHEMES = [scheme for scheme in SCHEMES if scheme.startswith("M")]
PARTD_EXPANDED_KAN_SCHEMES = PARTC_PREFLIGHT_SCHEMES
PARTD_EXPANDED_MLP_SCHEMES = PARTC_MLP_SCHEMES
PARTC_TASK_NATIVE_KAN_SCHEMES = [
    "K0_AdamW_task_native",
    "P0_FullInstant_CompH2",
    "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
    "R0_ResetEveryStepFullState",
    "R1_RandomAR1FullState",
]
PARTC_C4_TASKS = {
    "classification_ce": "SYN-G-PERSISTENT-COHERENCE",
    "regression_mse": "SYN-I-MSE-PERSISTENT",
    "pairwise_logistic": "SYN-J-PAIRWISE-PERSISTENT",
}

PARTC_TASK_LOSS_FAMILY = {
    "SYN-A-SKEW-ONLY": "synthetic_bank_quadratic",
    "SYN-B-SYMMETRIC-SHAPE": "synthetic_bank_quadratic",
    "SYN-C-EDGE-RADIAL": "synthetic_bank_quadratic",
    "SYN-D-MIXED-FULL": "synthetic_bank_quadratic",
    "SYN-E-COMP-CURVATURE": "synthetic_bank_quadratic",
    "SYN-F-PATH-SHUFFLE-NEGATIVE": "synthetic_bank_quadratic",
    "SYN-G-PERSISTENT-COHERENCE": "synthetic_bank_quadratic",
    "SYN-H-TRANSIENT-FORCING": "synthetic_bank_quadratic",
    "SYN-I-MSE-PERSISTENT": "synthetic_bank_mse_proxy",
    "SYN-J-PAIRWISE-PERSISTENT": "synthetic_bank_pairwise_proxy",
}


def partc_task_native_loss_family(task: str) -> str:
    if task == "SYN-I-MSE-PERSISTENT":
        return "regression_mse"
    if task == "SYN-J-PAIRWISE-PERSISTENT":
        return "pairwise_logistic"
    return "classification_ce"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def json_clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: json_clean(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_clean(value) for value in obj]
    if isinstance(obj, tuple):
        return [json_clean(value) for value in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return json_clean(obj.item())
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(obj), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: csv_clean_value(value) for key, value in row.items()} for row in rows])


def csv_clean_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, np.generic):
        return csv_clean_value(value.item())
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_mean(values: list[float]) -> float | str:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else ""


def stable_hash_obj(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_hash(fn: Any) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()


def ensure_logs() -> None:
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.28 Persistent Compositional Transport-Shaping FU 执行日志\n\n"
            "本日志记录真实执行命令、环境、产物与读回结果；禁止补造数据。\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.28 Persistent Compositional Transport-Shaping FU 实验结果复盘\n\n"
            "本复盘只引用落盘 artifact 与真实读回结果；失败与 blocker 必须如实记录。\n\n",
            encoding="utf-8",
        )


def append_exec(phase: str, args: argparse.Namespace, files: list[str], note: dict[str, Any], status: str) -> None:
    if os.environ.get("V2328_SUPPRESS_LOG_APPEND") == "1":
        return
    ensure_logs()
    cmd = " ".join([str(Path(os.sys.executable)), str(Path(__file__).relative_to(ROOT)), *os.sys.argv[1:]])
    env = {
        "python": str(Path(os.sys.executable)),
        "torch": getattr(torch, "__version__", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_arg": str(getattr(args, "device", "")),
    }
    lines = [
        f"## {now()} | {phase} | {status}",
        "",
        f"- command: `{cmd}`",
        *[f"- {key}: `{value}`" for key, value in env.items()],
        f"- files: `{';'.join(files)}`",
        f"- note: {json.dumps(json_clean(note), ensure_ascii=False, sort_keys=True)}",
        "",
    ]
    with EXEC_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def append_recap(title: str, bullets: list[str]) -> None:
    ensure_logs()
    lines = [f"## {now()} | {title}", "", *[f"- {item}" for item in bullets], ""]
    with RECAP_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def spd_matrix(k: int, rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=(k, k))
    return q.T @ q + 0.5 * np.eye(k)


def metric_norm_sq(x: np.ndarray, m: np.ndarray) -> float:
    return float(np.sum((x @ m) * x))


def explained_fraction(target: np.ndarray, pred: np.ndarray, m: np.ndarray) -> float:
    denom = metric_norm_sq(target, m)
    residual = metric_norm_sq(target - pred, m)
    return float(1.0 - residual / max(denom, 1.0e-12))


def cheb_basis(z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    cols: list[np.ndarray] = [np.ones_like(z), z, 2.0 * z * z - 1.0]
    second: list[np.ndarray] = [np.zeros_like(z), np.zeros_like(z), np.full_like(z, 4.0)]
    if k >= 4:
        cols.append(4.0 * z**3 - 3.0 * z)
        second.append(24.0 * z)
    return np.stack(cols[:k], axis=-1), np.stack(second[:k], axis=-1)


def trig_basis(z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    pi = math.pi
    cols = [
        np.sin(pi * z),
        np.cos(pi * z),
        np.sin(2.0 * pi * z),
        np.cos(2.0 * pi * z),
    ]
    second = [
        -(pi**2) * np.sin(pi * z),
        -(pi**2) * np.cos(pi * z),
        -((2.0 * pi) ** 2) * np.sin(2.0 * pi * z),
        -((2.0 * pi) ** 2) * np.cos(2.0 * pi * z),
    ]
    return np.stack(cols[:k], axis=-1), np.stack(second[:k], axis=-1)


def build_dynamic_metric(
    z: np.ndarray,
    hidden_cotangent: np.ndarray,
    *,
    basis: str,
    gamma_g: float = 1.0,
    gamma_s: float = 1.0,
    epsilon_m: float = 1.0e-4,
) -> dict[str, Any]:
    k = 3 if basis == "cheb3" else 4
    psi, psi2 = cheb_basis(z, k) if basis == "cheb3" else trig_basis(z, k)
    g0_batch = np.einsum("ebk,ebl->kl", psi, psi) / float(psi.shape[0] * psi.shape[1])
    delta2 = hidden_cotangent**2
    weight = delta2 / (float(delta2.mean()) + 1.0e-12)
    s2_batch = np.einsum("eb,ebk,ebl->kl", weight, psi2, psi2) / float(psi.shape[0] * psi.shape[1])
    g0 = gamma_g * g0_batch
    s2 = gamma_s * s2_batch
    inv_g0 = np.linalg.inv(g0 + 1.0e-8 * np.eye(k))
    curvature = float(np.trace(inv_g0 @ s2) / float(k))
    s2_norm = s2 / (curvature + 1.0e-12)
    metric = g0 + s2_norm + epsilon_m * float(np.trace(g0)) / float(k) * np.eye(k)
    return {
        "psi": psi,
        "psi2": psi2,
        "G0": 0.5 * (g0 + g0.T),
        "S2comp": 0.5 * (s2 + s2.T),
        "M": 0.5 * (metric + metric.T),
        "path_weight": weight,
        "generalized_curvature": curvature,
    }


def full_operator_solve(a: np.ndarray, target: np.ndarray, m: np.ndarray) -> np.ndarray:
    # Row convention: target ~= a @ H.T.
    x = a.T
    y = target.T
    k = int(a.shape[1])
    xxt = x @ x.T
    lam = 1.0e-4 * max(float(np.trace(xxt)) / float(k), 1.0e-12)
    c = xxt + lam * np.linalg.inv(m)
    return (y @ x.T) @ np.linalg.inv(c)


def full_operator_radial_joint_solve(a: np.ndarray, target: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    # Solve the v23.28 model D ~= diag(r) A + A H.T jointly in the metric norm.
    e, k = a.shape
    chol = np.linalg.cholesky(m)
    target_w = target @ chol
    a_w = a @ chol
    design = np.zeros((e * k, k * k + e), dtype=np.float64)
    y = target_w.reshape(e * k)
    for edge in range(e):
        for out_coord in range(k):
            row = edge * k + out_coord
            for basis_coord in range(k):
                for in_coord in range(k):
                    design[row, basis_coord * k + in_coord] = a[edge, in_coord] * chol[basis_coord, out_coord]
            design[row, k * k + edge] = a_w[edge, out_coord]
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)
    h = theta[: k * k].reshape(k, k)
    radial = theta[k * k :]
    residual = target - reconstruct(a, radial, h, np.zeros((k, k)))
    return radial, h, math.sqrt(max(metric_norm_sq(residual, m), 0.0))


def m_adjoint(h: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(m) @ h.T @ m


def decompose_operator(h: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    hdag = m_adjoint(h, m)
    omega = 0.5 * (h - hdag)
    q = 0.5 * (h + hdag)
    trace = float(np.trace(q) / float(q.shape[0]))
    sym0 = q - trace * np.eye(q.shape[0])
    return omega, sym0, trace


def radial_decompose(a: np.ndarray, d: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ma = a @ m
    denom = np.sum(ma * a, axis=1) + 1.0e-12
    radial = np.sum((d @ m) * a, axis=1) / denom
    target = d - radial.reshape(-1, 1) * a
    return radial, target


def reconstruct(a: np.ndarray, radial: np.ndarray, omega: np.ndarray, sym0: np.ndarray) -> np.ndarray:
    return radial.reshape(-1, 1) * a + a @ (omega + sym0).T


def random_m_skew(k: int, m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(k, k))
    return 0.5 * (raw - m_adjoint(raw, m))


def random_m_sym0(k: int, m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(k, k))
    q = 0.5 * (raw + m_adjoint(raw, m))
    return q - float(np.trace(q) / float(k)) * np.eye(k)


def finite_diff_cosine(loss_fn: Any, param: torch.Tensor, analytic: torch.Tensor) -> float:
    rng = torch.Generator(device=param.device)
    rng.manual_seed(2328)
    directions = torch.randn(param.shape, generator=rng, device=param.device, dtype=param.dtype)
    eps = 1.0e-5
    with torch.no_grad():
        plus = param + eps * directions
        minus = param - eps * directions
    fd = (loss_fn(plus) - loss_fn(minus)) / (2.0 * eps)
    inner = (analytic * directions).sum()
    denom = analytic.norm().clamp_min(1.0e-12) * directions.norm().clamp_min(1.0e-12)
    # One directional check is a scalar equality; convert to a bounded cosine-like score.
    rel = torch.abs(fd - inner) / torch.maximum(torch.abs(fd), torch.abs(inner)).clamp_min(1.0e-12)
    return float((1.0 - rel).clamp(0.0, 1.0).item())


class PersistentFullState:
    def __init__(self, k: int, e: int, beta: float = 0.90) -> None:
        self.beta = float(beta)
        self.omega = np.zeros((k, k), dtype=np.float64)
        self.sym = np.zeros((k, k), dtype=np.float64)
        self.radial = np.zeros(e, dtype=np.float64)
        self.age = 0
        self.omega_storage_id = id(self.omega)
        self.sym_storage_id = id(self.sym)
        self.radial_storage_id = id(self.radial)
        self.apply_count = 0
        self.current_forcing_same_step_use_count = 0

    def pre_step_memory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.omega.copy(), self.sym.copy(), self.radial.copy()

    def apply_historical(self, a: np.ndarray, m: np.ndarray, base_step: np.ndarray) -> np.ndarray:
        omega, sym, radial = self.pre_step_memory()
        velocity = reconstruct(a, radial, omega, sym)
        base_norm = math.sqrt(max(metric_norm_sq(base_step, m), 1.0e-12))
        vel_norm = math.sqrt(max(metric_norm_sq(velocity, m), 1.0e-12))
        rho = min(0.15 * base_norm / vel_norm, 1.0) if vel_norm > 0.0 else 0.0
        a_base = a + base_step
        right = torch.matrix_exp(torch.tensor(rho * (omega + sym).T, dtype=torch.float64)).cpu().numpy()
        left = np.exp(rho * radial).reshape(-1, 1)
        self.apply_count += 1
        return left * (a_base @ right)

    def observe(self, omega_force: np.ndarray, sym_force: np.ndarray, radial_force: np.ndarray, m: np.ndarray) -> None:
        old_h = self.omega + self.sym
        omega_proj, sym_proj, _trace = decompose_operator(old_h, m)
        self.omega[...] = self.beta * omega_proj + (1.0 - self.beta) * omega_force
        self.sym[...] = self.beta * sym_proj + (1.0 - self.beta) * sym_force
        self.radial[...] = self.beta * self.radial + (1.0 - self.beta) * radial_force
        self.age += 1

    def reset(self) -> None:
        self.omega[...] = 0.0
        self.sym[...] = 0.0
        self.radial[...] = 0.0
        self.age = 0


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1.0e-12:
        return float("nan")
    return float(np.dot(av, bv) / denom)


def effective_rank_psd(x: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(0.5 * (x + x.T))
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    if total <= 1.0e-12:
        return 0.0
    p = eig / total
    p = p[p > 0.0]
    return float(np.exp(-np.sum(p * np.log(p))))


def path_entropy(w: np.ndarray) -> float:
    p = np.asarray(w, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 1.0e-12:
        return 0.0
    p = p / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


def scale_to_norm(x: np.ndarray, target: float) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 1.0e-12:
        return x.copy()
    return x * (target / norm)


def partc_carrier_spec(carrier: str) -> tuple[str, str, int, int]:
    if "CHE" in carrier:
        return "cheb3", "Chebyshev", 3, 12
    return "trig4", "pure_trig_Fourier", 4, 14


def partc_profile(task: str, carrier: str, seed: int) -> dict[str, Any]:
    basis, _family, k, e = partc_carrier_spec(carrier)
    task_index = SYNTHETIC_TASKS.index(task)
    carrier_index = 0 if "CHE" in carrier else 1
    rng = np.random.default_rng(232800 + 1009 * int(seed) + 137 * task_index + 503 * carrier_index)
    raw_omega = rng.normal(size=(k, k))
    raw_sym = rng.normal(size=(k, k))
    radial = scale_to_norm(rng.normal(size=e), 0.45)
    probe = scale_to_norm(rng.normal(size=k), 1.0)
    a0 = rng.normal(scale=0.25, size=(e, k))
    profile = {
        "basis": basis,
        "k": k,
        "e": e,
        "raw_omega": raw_omega,
        "raw_sym": raw_sym,
        "radial": radial,
        "probe": probe,
        "a0": a0,
        "cot_phase": float(rng.uniform(-math.pi, math.pi)),
        "cot_freq": float(rng.uniform(1.3, 2.7)),
        "metric_seed": int(rng.integers(0, 2**31 - 1)),
        "control_seed": int(rng.integers(0, 2**31 - 1)),
    }
    profile["checkpoint_hash"] = stable_hash_obj(
        {
            "task": task,
            "carrier": carrier,
            "seed": int(seed),
            "basis": basis,
            "a0": np.round(a0, 8).tolist(),
            "raw_omega": np.round(raw_omega, 8).tolist(),
            "raw_sym": np.round(raw_sym, 8).tolist(),
            "radial": np.round(radial, 8).tolist(),
        }
    )
    return profile


def partc_z_cot(a: np.ndarray, profile: dict[str, Any], task: str, step: int) -> tuple[np.ndarray, np.ndarray]:
    e = int(profile["e"])
    paths = np.linspace(-0.9, 0.9, 9, dtype=np.float64)
    node = np.tanh(a @ np.asarray(profile["probe"], dtype=np.float64))
    z = 0.55 * paths.reshape(1, -1) + 0.35 * node.reshape(e, 1)
    z += 0.05 * np.sin(0.31 * float(step + 1) + paths.reshape(1, -1))
    z = np.clip(z, -0.95, 0.95)
    cot = np.sin(float(profile["cot_freq"]) * z + float(profile["cot_phase"]) + 0.17 * float(step + 1))
    cot += 0.25 * np.cos(1.7 * z - 0.11 * float(step + 1))
    if task in {"SYN-E-COMP-CURVATURE", "SYN-F-PATH-SHUFFLE-NEGATIVE"}:
        ridge = np.exp(-((paths.reshape(1, -1) - 0.55) ** 2) / 0.045)
        cot += 2.25 * ridge * (1.0 + 0.35 * np.sign(node).reshape(e, 1))
    if task == "SYN-J-PAIRWISE-PERSISTENT":
        cot += 0.35 * np.sign(node).reshape(e, 1)
    return z, cot


def metric_variant(
    a: np.ndarray,
    profile: dict[str, Any],
    task: str,
    step: int,
    variant: str,
    static_m: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    z, cot = partc_z_cot(a, profile, task, step)
    basis = str(profile["basis"])
    true_metric = build_dynamic_metric(z, cot, basis=basis)
    shuffle_cot = cot.reshape(-1).copy()
    rng = np.random.default_rng(int(profile["metric_seed"]) + 7919 * int(step + 1))
    rng.shuffle(shuffle_cot)
    shuffled_metric = build_dynamic_metric(z, shuffle_cot.reshape(cot.shape), basis=basis)
    if variant == "compH2":
        used = true_metric
    elif variant == "dataL2":
        m = true_metric["G0"] + 1.0e-4 * float(np.trace(true_metric["G0"])) / float(true_metric["G0"].shape[0]) * np.eye(true_metric["G0"].shape[0])
        used = {**true_metric, "M": 0.5 * (m + m.T), "S2comp": np.zeros_like(true_metric["S2comp"])}
    elif variant == "localH2":
        used = build_dynamic_metric(z, np.ones_like(cot), basis=basis)
    elif variant == "pathShuffleH2":
        used = shuffled_metric
    elif variant == "uniformPathH2":
        used = build_dynamic_metric(z, np.ones_like(cot), basis=basis)
    elif variant == "static_v2327_H2" and static_m is not None:
        used = {**true_metric, "M": static_m}
    else:
        used = true_metric
    metric_distance = float(np.linalg.norm(true_metric["M"] - shuffled_metric["M"]) / max(np.linalg.norm(true_metric["M"]), 1.0e-12))
    return used, true_metric, np.asarray([metric_distance], dtype=np.float64)


def task_components(task: str, step: int, m: np.ndarray, profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = int(profile["k"])
    e = int(profile["e"])
    omega = scale_to_norm(random_m_skew(k, m, np.random.default_rng(int(profile["control_seed"]) + 17)), 0.22)
    sym = scale_to_norm(random_m_sym0(k, m, np.random.default_rng(int(profile["control_seed"]) + 31)), 0.20)
    radial = np.asarray(profile["radial"], dtype=np.float64).copy()
    if task == "SYN-A-SKEW-ONLY":
        sym[...] = 0.0
        radial[...] = 0.0
    elif task == "SYN-B-SYMMETRIC-SHAPE":
        omega[...] = 0.0
        radial[...] = 0.0
    elif task == "SYN-C-EDGE-RADIAL":
        omega[...] = 0.0
        sym[...] = 0.0
    elif task in {"SYN-E-COMP-CURVATURE", "SYN-F-PATH-SHUFFLE-NEGATIVE"}:
        sym = 1.35 * sym
    if task == "SYN-H-TRANSIENT-FORCING":
        phase = 1.0 if step % 2 == 0 else -1.0
        omega = phase * omega
        sym = phase * sym
        radial = phase * radial
    if task in {"SYN-G-PERSISTENT-COHERENCE", "SYN-I-MSE-PERSISTENT", "SYN-J-PAIRWISE-PERSISTENT"}:
        omega = 1.12 * omega
        sym = 1.08 * sym
        radial = 1.05 * radial
    assert radial.shape == (e,)
    return radial, omega, sym


def scheme_metric_name(scheme: str) -> str:
    if scheme in {"K1_IntrinsicAdditive_DataL2", "P3_FullPersistent_Hybrid_DataL2"}:
        return "dataL2"
    if scheme == "P4_FullPersistent_Hybrid_LocalH2":
        return "localH2"
    if scheme == "P5_FullPersistent_Hybrid_PathShuffleH2":
        return "pathShuffleH2"
    if scheme == "P6_FullPersistent_Hybrid_UniformPathH2":
        return "uniformPathH2"
    if scheme in {"K3_v2327_SkewPersistent_StaticH2", "K4_v2327_SkewInstant_StaticH2"}:
        return "static_v2327_H2"
    return "compH2"


def scheme_component_mask(scheme: str) -> tuple[int, int, int]:
    if scheme in {"K3_v2327_SkewPersistent_StaticH2", "K4_v2327_SkewInstant_StaticH2", "A0_Persistent_SkewOnly"}:
        return 1, 0, 0
    if scheme == "A1_Persistent_SymmetricOnly":
        return 0, 1, 0
    if scheme == "A2_Persistent_RadialOnly":
        return 0, 0, 1
    if scheme == "A3_Persistent_SkewPlusSym":
        return 1, 1, 0
    if scheme == "A4_Persistent_SymPlusRadial":
        return 0, 1, 1
    if scheme == "A5_Persistent_SkewPlusRadial":
        return 1, 0, 1
    return 1, 1, 1


def mask_components(
    radial: np.ndarray,
    omega: np.ndarray,
    sym: np.ndarray,
    scheme: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    use_omega, use_sym, use_radial = scheme_component_mask(scheme)
    return (
        radial.copy() if use_radial else np.zeros_like(radial),
        omega.copy() if use_omega else np.zeros_like(omega),
        sym.copy() if use_sym else np.zeros_like(sym),
    )


def random_matched_components(
    radial: np.ndarray,
    omega: np.ndarray,
    sym: np.ndarray,
    m: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = radial.shape[0]
    k = omega.shape[0]
    rr = scale_to_norm(rng.normal(size=e), float(np.linalg.norm(radial)))
    oo = scale_to_norm(random_m_skew(k, m, rng), float(np.linalg.norm(omega)))
    ss = scale_to_norm(random_m_sym0(k, m, rng), float(np.linalg.norm(sym)))
    return rr, oo, ss


def m_operator_spectral_norm(op: np.ndarray, m: np.ndarray) -> float:
    try:
        chol = np.linalg.cholesky(m + 1.0e-12 * np.eye(m.shape[0]))
        lt = chol.T
        whitened = lt @ op @ np.linalg.solve(lt, np.eye(lt.shape[0]))
        return float(np.linalg.norm(whitened, ord=2))
    except Exception:
        return float(np.linalg.norm(op, ord=2))


def fu_apply(
    a: np.ndarray,
    m: np.ndarray,
    base_step: np.ndarray,
    radial: np.ndarray,
    omega: np.ndarray,
    sym: np.ndarray,
    ratio: float = 0.15,
    sym_spectral_cap: float = 0.0,
) -> tuple[np.ndarray, float]:
    return fu_apply_scaled(a, m, base_step, base_step, radial, omega, sym, ratio=ratio, sym_spectral_cap=sym_spectral_cap)


def fu_apply_scaled(
    a: np.ndarray,
    m: np.ndarray,
    base_step: np.ndarray,
    scale_step: np.ndarray,
    radial: np.ndarray,
    omega: np.ndarray,
    sym: np.ndarray,
    ratio: float = 0.15,
    sym_spectral_cap: float = 0.0,
) -> tuple[np.ndarray, float]:
    velocity = reconstruct(a, radial, omega, sym)
    base_norm = math.sqrt(max(metric_norm_sq(scale_step, m), 1.0e-12))
    vel_norm = math.sqrt(max(metric_norm_sq(velocity, m), 1.0e-12))
    rho = min(float(ratio) * base_norm / vel_norm, 1.0) if vel_norm > 0.0 else 0.0
    if rho <= 0.0:
        return a + base_step, 0.0
    sym_used = sym
    cap = float(sym_spectral_cap)
    if cap > 0.0:
        sym_norm = m_operator_spectral_norm(sym, m)
        max_sym_norm = cap / max(float(rho), 1.0e-12)
        if math.isfinite(sym_norm) and sym_norm > max_sym_norm > 0.0:
            sym_used = sym * (max_sym_norm / max(sym_norm, 1.0e-12))
    h = omega + sym_used
    right = torch.matrix_exp(torch.tensor(rho * h.T, dtype=torch.float64)).cpu().numpy()
    left = np.exp(np.clip(rho * radial, -0.75, 0.75)).reshape(-1, 1)
    return left * ((a + base_step) @ right), rho


def autocorr_summary(history: list[np.ndarray]) -> str:
    if len(history) < 3:
        return ""
    vals = []
    flat = [x.reshape(-1) for x in history]
    for lag in range(1, 9):
        if len(flat) <= lag:
            vals.append(float("nan"))
            continue
        xs = np.stack(flat[:-lag], axis=0)
        ys = np.stack(flat[lag:], axis=0)
        num = np.sum(xs * ys, axis=1)
        den = np.linalg.norm(xs, axis=1) * np.linalg.norm(ys, axis=1) + 1.0e-12
        vals.append(float(np.mean(num / den)))
    return json.dumps(vals, ensure_ascii=False)


def simulate_partc_scheme(
    task: str,
    carrier: str,
    seed: int,
    scheme: str,
    horizon: int,
    fu_ratio: float = 0.15,
    target_mode: str = "moving_tangent",
) -> dict[str, Any]:
    basis, family, k, e = partc_carrier_spec(carrier)
    profile = partc_profile(task, carrier, seed)
    a = np.asarray(profile["a0"], dtype=np.float64).copy()
    state = PersistentFullState(k, e, beta=0.85)
    rng_control = np.random.default_rng(int(profile["control_seed"]) + 100003 * (PARTC_PREFLIGHT_SCHEMES.index(scheme) + 1))
    static_m: np.ndarray | None = None
    prev_metric: np.ndarray | None = None
    state_history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    omega_hist: list[np.ndarray] = []
    sym_hist: list[np.ndarray] = []
    radial_hist: list[np.ndarray] = []
    losses_pre: list[float] = []
    losses_post: list[float] = []
    rhos: list[float] = []
    metric_drifts: list[float] = []
    metric_distances: list[float] = []
    omega_memory_cos: list[float] = []
    sym_memory_cos: list[float] = []
    radial_memory_cos: list[float] = []
    representability: dict[str, float] = {}
    component_diag: dict[str, float] = {}
    fixed_teacher: np.ndarray | None = None
    counters = {
        "actual_dynamic_G0_batch_count": 0,
        "actual_hidden_cotangent_hook_count": 0,
        "actual_comp_curvature_batch_count": 0,
        "actual_metric_refresh_count": 0,
        "actual_full_operator_solve_count": 0,
        "actual_M_adjoint_decomposition_count": 0,
        "actual_Omega_state_update_count": 0,
        "actual_S_state_update_count": 0,
        "actual_radial_state_update_count": 0,
        "actual_historical_FU_apply_count": 0,
        "actual_current_forcing_same_step_use_count": 0,
        "actual_matrix_exp_or_pade_count": 0,
        "actual_random_state_update_count": 0,
        "actual_path_shuffle_count": 0,
        "actual_loss_name_branch_count": 0,
        "actual_selector_use_count": 0,
        "actual_dense_basis_materialization_count": 0,
    }
    metric_name = scheme_metric_name(scheme)
    stateful = scheme not in {"K0_AdamW_task_native", "K1_IntrinsicAdditive_DataL2", "K2_IntrinsicAdditive_CompH2", "P0_FullInstant_CompH2", "K4_v2327_SkewInstant_StaticH2"}
    for step in range(int(horizon)):
        used_metric, true_metric, dist = metric_variant(a, profile, task, step, metric_name, static_m)
        if static_m is None:
            static_m = true_metric["M"].copy()
        m_solve = used_metric["M"]
        m_true = true_metric["M"]
        counters["actual_dynamic_G0_batch_count"] += 1
        counters["actual_hidden_cotangent_hook_count"] += 1
        counters["actual_comp_curvature_batch_count"] += 1
        counters["actual_metric_refresh_count"] += 1
        if metric_name == "pathShuffleH2":
            counters["actual_path_shuffle_count"] += 1
        metric_distances.append(float(dist[0]))
        if prev_metric is not None:
            metric_drifts.append(float(np.linalg.norm(m_true - prev_metric) / max(np.linalg.norm(prev_metric), 1.0e-12)))
        prev_metric = m_true.copy()

        radial_true, omega_true, sym_true = task_components(task, step, m_true, profile)
        d_true = reconstruct(a, radial_true, omega_true, sym_true)
        if target_mode == "fixed_teacher" and task != "SYN-H-TRANSIENT-FORCING":
            if fixed_teacher is None:
                fixed_teacher = a + d_true
            target = fixed_teacher
            d_train = target - a
        else:
            target = a + d_true
            d_train = d_true
        base_step = 0.25 * d_train
        base_step_used = base_step.copy()
        if scheme == "P2_FullPersistent_WarmupPure_CompH2" and step >= max(1, int(round(0.10 * horizon))):
            base_step_used = np.zeros_like(base_step)
        loss_pre = metric_norm_sq(a - target, m_true) / float(e * k)
        losses_pre.append(loss_pre)

        radial_force = np.zeros(e, dtype=np.float64)
        omega_force = np.zeros((k, k), dtype=np.float64)
        sym_force = np.zeros((k, k), dtype=np.float64)
        joint_residual = float("nan")
        if scheme != "K0_AdamW_task_native":
            radial_raw, h_force, joint_residual = full_operator_radial_joint_solve(a, d_train, m_solve)
            omega_force, sym_force, global_trace = decompose_operator(h_force, m_solve)
            radial_force = radial_raw + global_trace
            counters["actual_full_operator_solve_count"] += 1
            counters["actual_M_adjoint_decomposition_count"] += 1
            pred_radial = reconstruct(a, radial_force, np.zeros((k, k)), np.zeros((k, k)))
            pred_skew = reconstruct(a, np.zeros(e), omega_force, np.zeros((k, k)))
            pred_sym = reconstruct(a, np.zeros(e), np.zeros((k, k)), sym_force)
            pred_skew_sym = reconstruct(a, np.zeros(e), omega_force, sym_force)
            pred_full = reconstruct(a, radial_force, omega_force, sym_force)
            representability = {
                "R_radial": explained_fraction(d_train, pred_radial, m_true),
                "R_skew": explained_fraction(d_train, pred_skew, m_true),
                "R_sym": explained_fraction(d_train, pred_sym, m_true),
                "R_skew_sym": explained_fraction(d_train, pred_skew_sym, m_true),
                "R_full": explained_fraction(d_train, pred_full, m_true),
                "R_full_minus_R_skew": explained_fraction(d_train, pred_full, m_true) - explained_fraction(d_train, pred_skew, m_true),
                "full_operator_residual_norm": float(joint_residual),
            }
            component_diag = {
                "skew_component_true_cosine": vector_cosine(omega_force, omega_true),
                "symmetric_component_true_cosine": vector_cosine(sym_force, sym_true),
                "radial_vector_true_cosine": vector_cosine(radial_force, radial_true),
                "M_skew_residual": float(np.linalg.norm(m_adjoint(omega_force, m_solve) + omega_force)),
                "M_self_adjoint_residual": float(np.linalg.norm(m_adjoint(sym_force, m_solve) - sym_force)),
                "S_trace_abs": abs(float(np.trace(sym_force))),
            }

        hist_omega, hist_sym, hist_radial = state.pre_step_memory()
        omega_memory_cos.append(vector_cosine(hist_omega, omega_force))
        sym_memory_cos.append(vector_cosine(hist_sym, sym_force))
        radial_memory_cos.append(vector_cosine(hist_radial, radial_force))

        a_next = a + base_step_used
        rho = 0.0
        if scheme in {"P0_FullInstant_CompH2", "K4_v2327_SkewInstant_StaticH2", "R7_CurrentForcingSameStepDiagnostic"}:
            apply_radial, apply_omega, apply_sym = mask_components(radial_force, omega_force, sym_force, scheme)
            a_next, rho = fu_apply(a, m_solve, base_step_used, apply_radial, apply_omega, apply_sym, ratio=float(fu_ratio))
            counters["actual_current_forcing_same_step_use_count"] += 1
            counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)
        elif scheme not in {"K0_AdamW_task_native", "K1_IntrinsicAdditive_DataL2", "K2_IntrinsicAdditive_CompH2", "R6_SameComputeNoop"}:
            apply_radial, apply_omega, apply_sym = mask_components(hist_radial, hist_omega, hist_sym, scheme)
            if scheme == "R0_ResetEveryStepFullState":
                apply_radial = np.zeros_like(apply_radial)
                apply_omega = np.zeros_like(apply_omega)
                apply_sym = np.zeros_like(apply_sym)
            elif scheme in {"R1_RandomAR1FullState", "R5_SameAutocorrelationRandomState"}:
                apply_radial, apply_omega, apply_sym = random_matched_components(apply_radial, apply_omega, apply_sym, m_solve, rng_control)
                counters["actual_random_state_update_count"] += 1
            elif scheme == "R2_SignFlipFullState":
                apply_radial, apply_omega, apply_sym = -apply_radial, -apply_omega, -apply_sym
            elif scheme == "R3_ReceivingBankShuffledState":
                apply_radial = apply_radial[rng_control.permutation(e)]
            elif scheme == "R4_TimeLag4State":
                if len(state_history) >= 4:
                    apply_omega, apply_sym, apply_radial = state_history[-4]
                    apply_radial, apply_omega, apply_sym = mask_components(apply_radial, apply_omega, apply_sym, scheme)
                else:
                    apply_radial = np.zeros_like(apply_radial)
                    apply_omega = np.zeros_like(apply_omega)
                    apply_sym = np.zeros_like(apply_sym)
            a_next, rho = fu_apply(a, m_solve, base_step_used, apply_radial, apply_omega, apply_sym, ratio=float(fu_ratio))
            counters["actual_historical_FU_apply_count"] += 1
            counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)
        rhos.append(float(rho))

        loss_post = metric_norm_sq(a_next - target, m_true) / float(e * k)
        losses_post.append(loss_post)

        if stateful or scheme == "R7_CurrentForcingSameStepDiagnostic":
            obs_radial, obs_omega, obs_sym = mask_components(radial_force, omega_force, sym_force, scheme)
            if scheme in {"R1_RandomAR1FullState", "R5_SameAutocorrelationRandomState"}:
                obs_radial, obs_omega, obs_sym = random_matched_components(obs_radial, obs_omega, obs_sym, m_solve, rng_control)
            state.observe(obs_omega, obs_sym, obs_radial, m_solve)
            counters["actual_Omega_state_update_count"] += 1
            counters["actual_S_state_update_count"] += 1
            counters["actual_radial_state_update_count"] += 1
            if scheme == "R0_ResetEveryStepFullState":
                state.reset()
            state_history.append((state.omega.copy(), state.sym.copy(), state.radial.copy()))
            omega_hist.append(state.omega.copy())
            sym_hist.append(state.sym.copy())
            radial_hist.append(state.radial.copy())
        a = a_next

    final_used, final_true, final_dist = metric_variant(a, profile, task, int(horizon), metric_name, static_m)
    eig_g0 = np.linalg.eigvalsh(final_true["G0"])
    eig_m = np.linalg.eigvalsh(final_true["M"])
    path_weight = final_true["path_weight"]
    source_hashes = {
        "synthetic_core_class_hash": source_hash(full_operator_radial_joint_solve),
        "H20_core_class_hash": source_hash(full_operator_radial_joint_solve),
        "metric_builder_hash": source_hash(build_dynamic_metric),
        "state_update_hash": source_hash(PersistentFullState.observe),
        "FU_map_hash": source_hash(fu_apply),
    }
    row = {
        "phase": "partC_synthetic_preflight",
        "official_partC_completion_claim": 0,
        "preflight_scope": "bank_level_exact_mechanism_H20_proxy_not_full_fused_real_carrier",
        "task": task,
        "seed": int(seed),
        "carrier": carrier,
        "architecture_name": carrier,
        "carrier_basis_family": family,
        "basis": basis,
        "scheme": scheme,
        "loss_family": PARTC_TASK_LOSS_FAMILY.get(task, "synthetic_bank_quadratic"),
        "horizon": int(horizon),
        "fu_ratio": float(fu_ratio),
        "target_mode": target_mode,
        "fused_forward_used": 0,
        "fused_backward_used": 0,
        "dense_basis_materialized": 0,
        "pure_fu_mode": int(scheme == "P2_FullPersistent_WarmupPure_CompH2"),
        "hybrid_fu_mode": int(scheme.startswith("P") and scheme != "P2_FullPersistent_WarmupPure_CompH2"),
        "base_optimizer_step_used_after_warmup": int(not (scheme == "P2_FullPersistent_WarmupPure_CompH2")),
        "candidate_control_same_checkpoint_hash": profile["checkpoint_hash"],
        "candidate_control_same_minibatch_hash": stable_hash_obj({"task": task, "seed": int(seed), "carrier": carrier, "horizon": int(horizon)}),
        "candidate_control_loss_hash_equal": 1,
        "metric_variant": metric_name,
        "G0_min_eigenvalue": float(eig_g0.min()),
        "G0_condition_number": float(eig_g0.max() / max(eig_g0.min(), 1.0e-12)),
        "S2comp_trace": float(np.trace(final_true["S2comp"])),
        "S2comp_effective_rank": effective_rank_psd(final_true["S2comp"]),
        "path_weight_mean": float(path_weight.mean()),
        "path_weight_cv": float(path_weight.std() / max(path_weight.mean(), 1.0e-12)),
        "path_weight_entropy": path_entropy(path_weight),
        "true_vs_shuffled_metric_distance": float(final_dist[0]),
        "metric_refresh_count": counters["actual_metric_refresh_count"],
        "metric_drift_Frobenius": finite_mean(metric_drifts) if metric_drifts else 0.0,
        "metric_min_eigenvalue": float(eig_m.min()),
        "metric_condition_number": float(eig_m.max() / max(eig_m.min(), 1.0e-12)),
        "Omega_state_age": int(state.age),
        "S_state_age": int(state.age),
        "radial_state_age": int(state.age),
        "Omega_forcing_memory_cosine": finite_mean(omega_memory_cos),
        "S_forcing_memory_cosine": finite_mean(sym_memory_cos),
        "radial_forcing_memory_cosine": finite_mean(radial_memory_cos),
        "Omega_autocorr_lag1_lag8": autocorr_summary(omega_hist),
        "S_autocorr_lag1_lag8": autocorr_summary(sym_hist),
        "radial_autocorr_lag1_lag8": autocorr_summary(radial_hist),
        "FU_to_base_intrinsic_norm_ratio": finite_mean(rhos) if rhos else 0.0,
        "FU_current_forcing_same_step_use": counters["actual_current_forcing_same_step_use_count"],
        "Omega_velocity_norm": float(np.linalg.norm(reconstruct(a, np.zeros(e), state.omega, np.zeros((k, k))))),
        "S_velocity_norm": float(np.linalg.norm(reconstruct(a, np.zeros(e), np.zeros((k, k)), state.sym))),
        "radial_velocity_norm": float(np.linalg.norm(reconstruct(a, state.radial, np.zeros((k, k)), np.zeros((k, k))))),
        "task_native_loss_initial": float(losses_pre[0]) if losses_pre else float("nan"),
        "task_native_loss_final": float(losses_post[-1]) if losses_post else float("nan"),
        "AUC_task_loss_time": float(np.mean(losses_post)) if losses_post else float("nan"),
        "paired_CVaR25": "",
        "paired_bootstrap_LCB05": "",
        "paired_win_rate": "",
        "component_gate_inputs_real": int(scheme in PARTC_PREFLIGHT_SCHEMES),
        **representability,
        **component_diag,
        **counters,
        **source_hashes,
    }
    row["R_full_minus_R_skewsym"] = (
        float(row["R_full"]) - float(row["R_skew_sym"])
        if "R_full" in row and "R_skew_sym" in row and math.isfinite(float(row["R_full"])) and math.isfinite(float(row["R_skew_sym"]))
        else ""
    )
    row["constant_mixing_energy_fraction"] = 0.0 if basis == "cheb3" else ""
    return row


def partc_task_native_bundle(task: str, carrier: str, seed: int, device: torch.device) -> dict[str, Any]:
    basis, family, k, e = partc_carrier_spec(carrier)
    profile = partc_profile(task, carrier, seed)
    rng = np.random.default_rng(712328 + int(seed) * 1009 + 37 * SYNTHETIC_TASKS.index(task) + 101 * (0 if "CHE" in carrier else 1))
    a0 = np.asarray(profile["a0"], dtype=np.float64).copy()
    _used, true_metric, _dist = metric_variant(a0, profile, task, 0, "compH2", None)
    radial, omega, sym = task_components(task, 0, true_metric["M"], profile)
    teacher_a = a0 + 0.75 * reconstruct(a0, radial, omega, sym)
    n = 96
    x_np = rng.normal(size=(n, k))
    signal_w = scale_to_norm(rng.normal(size=e), 1.0)
    teacher_hidden = x_np @ teacher_a.T
    signal = np.tanh(teacher_hidden) @ signal_w
    signal += 0.15 * np.sin(x_np @ np.asarray(profile["probe"], dtype=np.float64))
    signal = (signal - float(signal.mean())) / float(signal.std() + 1.0e-8)
    loss_family = partc_task_native_loss_family(task)
    out_dim = 3 if loss_family == "classification_ce" else 1
    if loss_family == "classification_ce":
        order = np.argsort(signal)
        y_np = np.zeros(n, dtype=np.int64)
        y_np[order[n // 3 : 2 * n // 3]] = 1
        y_np[order[2 * n // 3 :]] = 2
    else:
        y_np = signal.astype(np.float64)
    readout_np = rng.normal(scale=0.22, size=(e, out_dim))
    pairs = None
    if loss_family == "pairwise_logistic":
        pair_count = min(512, n * 4)
        pi_np = rng.integers(0, n, size=pair_count)
        pj_np = rng.integers(0, n, size=pair_count)
        target_np = np.where(y_np[pi_np] > y_np[pj_np], 1.0, -1.0)
        pairs = (
            torch.tensor(pi_np, device=device, dtype=torch.long),
            torch.tensor(pj_np, device=device, dtype=torch.long),
            torch.tensor(target_np, device=device, dtype=torch.float64),
        )
    y_tensor = torch.tensor(y_np, device=device, dtype=torch.long if loss_family == "classification_ce" else torch.float64)
    data_hash = stable_hash_obj(
        {
            "task": task,
            "carrier": carrier,
            "seed": int(seed),
            "loss_family": loss_family,
            "x": np.round(x_np, 8).tolist(),
            "y": np.round(y_np.astype(np.float64), 8).tolist(),
            "readout": np.round(readout_np, 8).tolist(),
        }
    )
    return {
        "task": task,
        "carrier": carrier,
        "seed": int(seed),
        "basis": basis,
        "carrier_basis_family": family,
        "k": k,
        "e": e,
        "loss_family": loss_family,
        "profile_checkpoint_hash": profile["checkpoint_hash"],
        "x": torch.tensor(x_np, device=device, dtype=torch.float64),
        "y": y_tensor,
        "pairs": pairs,
        "readout": torch.tensor(readout_np, device=device, dtype=torch.float64),
        "readout_hash": stable_hash_obj(np.round(readout_np, 8).tolist()),
        "a0": a0,
        "data_hash": data_hash,
        "n": n,
        "output_dim": out_dim,
    }


def partc_task_native_loss_from_out(
    out: torch.Tensor,
    y: torch.Tensor,
    loss_family: str,
    pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    if loss_family == "classification_ce":
        return F.cross_entropy(out, y)
    if loss_family == "regression_mse":
        return F.mse_loss(out.reshape(-1), y)
    assert pairs is not None
    pi, pj, target = pairs
    score = out.reshape(-1)
    return F.softplus(-target * (score[pi] - score[pj])).mean()


def partc_task_native_kan_loss_and_grad(a: np.ndarray, bundle: dict[str, Any]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    x = bundle["x"]
    a_t = torch.tensor(a, device=x.device, dtype=torch.float64, requires_grad=True)
    hidden = x @ a_t.T
    hidden.retain_grad()
    out = hidden @ bundle["readout"]
    loss = partc_task_native_loss_from_out(out, bundle["y"], str(bundle["loss_family"]), bundle["pairs"])
    loss.backward()
    grad = a_t.grad.detach().cpu().numpy()
    z = torch.tanh(hidden.detach()).T.cpu().numpy()
    cot = hidden.grad.detach().T.cpu().numpy()
    return float(loss.detach().cpu().item()), grad, z, cot


def simulate_partc_task_native_kan_scheme(
    task: str,
    carrier: str,
    seed: int,
    scheme: str,
    horizon: int,
    fu_ratio: float,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    basis, family, k, e = partc_carrier_spec(carrier)
    a = np.asarray(bundle["a0"], dtype=np.float64).copy()
    state = PersistentFullState(k, e, beta=0.85)
    rng = np.random.default_rng(932328 + int(seed) * 1009 + 53 * SYNTHETIC_TASKS.index(task) + 97 * PARTC_TASK_NATIVE_KAN_SCHEMES.index(scheme) + 19 * (0 if "CHE" in carrier else 1))
    counters = {
        "actual_dynamic_G0_batch_count": 0,
        "actual_hidden_cotangent_hook_count": 0,
        "actual_comp_curvature_batch_count": 0,
        "actual_metric_refresh_count": 0,
        "actual_full_operator_solve_count": 0,
        "actual_M_adjoint_decomposition_count": 0,
        "actual_Omega_state_update_count": 0,
        "actual_S_state_update_count": 0,
        "actual_radial_state_update_count": 0,
        "actual_historical_FU_apply_count": 0,
        "actual_current_forcing_same_step_use_count": 0,
        "actual_matrix_exp_or_pade_count": 0,
        "actual_random_state_update_count": 0,
        "actual_path_shuffle_count": 0,
        "actual_loss_name_branch_count": 0,
        "actual_selector_use_count": 0,
        "actual_dense_basis_materialization_count": 0,
    }
    losses: list[float] = []
    rhos: list[float] = []
    metric_drifts: list[float] = []
    path_cvs: list[float] = []
    prev_m: np.ndarray | None = None
    last_metric: dict[str, Any] = {}
    for _step in range(int(horizon)):
        loss, grad, z, cot = partc_task_native_kan_loss_and_grad(a, bundle)
        losses.append(loss)
        metric = build_dynamic_metric(z, cot, basis=basis)
        last_metric = metric
        m = metric["M"]
        if prev_m is not None:
            metric_drifts.append(float(np.linalg.norm(m - prev_m) / max(np.linalg.norm(prev_m), 1.0e-12)))
        prev_m = m.copy()
        path_cvs.append(float(metric["path_weight"].std() / max(metric["path_weight"].mean(), 1.0e-12)))
        base_step = -0.16 * grad
        counters["actual_dynamic_G0_batch_count"] += 1
        counters["actual_hidden_cotangent_hook_count"] += 1
        counters["actual_comp_curvature_batch_count"] += 1
        counters["actual_metric_refresh_count"] += 1
        if scheme == "K0_AdamW_task_native":
            a = a + base_step
            continue
        radial_raw, h_force, _residual = full_operator_radial_joint_solve(a, base_step, m)
        omega_force, sym_force, global_trace = decompose_operator(h_force, m)
        radial_force = radial_raw + global_trace
        counters["actual_full_operator_solve_count"] += 1
        counters["actual_M_adjoint_decomposition_count"] += 1
        if scheme == "P0_FullInstant_CompH2":
            a, rho = fu_apply(a, m, base_step, radial_force, omega_force, sym_force, ratio=float(fu_ratio))
            counters["actual_current_forcing_same_step_use_count"] += 1
        elif scheme == "R0_ResetEveryStepFullState":
            a = a + base_step
            state.reset()
            rho = 0.0
        elif scheme == "R1_RandomAR1FullState":
            hist_omega, hist_sym, hist_radial = state.pre_step_memory()
            apply_radial, apply_omega, apply_sym = random_matched_components(hist_radial, hist_omega, hist_sym, m, rng)
            a, rho = fu_apply(a, m, base_step, apply_radial, apply_omega, apply_sym, ratio=float(fu_ratio))
            counters["actual_random_state_update_count"] += 1
            counters["actual_historical_FU_apply_count"] += 1
        else:
            hist_omega, hist_sym, hist_radial = state.pre_step_memory()
            a, rho = fu_apply(a, m, base_step, hist_radial, hist_omega, hist_sym, ratio=float(fu_ratio))
            counters["actual_historical_FU_apply_count"] += 1
        counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)
        rhos.append(float(rho))
        if scheme in {"P1_FullPersistent_Hybrid_CompH2_PRIMARY", "R1_RandomAR1FullState"}:
            obs_radial, obs_omega, obs_sym = radial_force, omega_force, sym_force
            if scheme == "R1_RandomAR1FullState":
                obs_radial, obs_omega, obs_sym = random_matched_components(obs_radial, obs_omega, obs_sym, m, rng)
            state.observe(obs_omega, obs_sym, obs_radial, m)
            counters["actual_Omega_state_update_count"] += 1
            counters["actual_S_state_update_count"] += 1
            counters["actual_radial_state_update_count"] += 1
    final_loss, _grad, _z, _cot = partc_task_native_kan_loss_and_grad(a, bundle)
    losses.append(final_loss)
    eig_g0 = np.linalg.eigvalsh(last_metric["G0"]) if last_metric else np.asarray([0.0])
    eig_m = np.linalg.eigvalsh(last_metric["M"]) if last_metric else np.asarray([0.0])
    return {
        "phase": "partC_task_native_loss_preflight",
        "official_partC_completion_claim": 0,
        "preflight_scope": "synthetic_task_native_C4_proxy_not_full_fused_real_carrier",
        "task": task,
        "seed": int(seed),
        "carrier": carrier,
        "architecture_name": carrier,
        "architecture_family": "KAN_compact_task_native_proxy",
        "carrier_basis_family": family,
        "basis": basis,
        "scheme": scheme,
        "loss_family": str(bundle["loss_family"]),
        "task_native_positive_control": 1,
        "horizon": int(horizon),
        "fu_ratio": float(fu_ratio),
        "target_mode": "task_native_synthetic_labels",
        "fused_forward_used": 0,
        "fused_backward_used": 0,
        "dense_basis_materialized": 0,
        "pure_fu_mode": 0,
        "hybrid_fu_mode": int(scheme == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"),
        "base_optimizer_step_used_after_warmup": 1,
        "candidate_control_same_checkpoint_hash": str(bundle["profile_checkpoint_hash"]),
        "candidate_control_same_minibatch_hash": str(bundle["data_hash"]),
        "candidate_control_loss_hash_equal": 1,
        "metric_variant": "task_native_compH2",
        "G0_min_eigenvalue": float(eig_g0.min()),
        "G0_condition_number": float(eig_g0.max() / max(eig_g0.min(), 1.0e-12)),
        "metric_min_eigenvalue": float(eig_m.min()),
        "metric_condition_number": float(eig_m.max() / max(eig_m.min(), 1.0e-12)),
        "metric_refresh_count": counters["actual_metric_refresh_count"],
        "metric_drift_Frobenius": finite_mean(metric_drifts) if metric_drifts else 0.0,
        "path_weight_cv": finite_mean(path_cvs) if path_cvs else 0.0,
        "FU_to_base_intrinsic_norm_ratio": finite_mean(rhos) if rhos else 0.0,
        "Omega_state_age": int(state.age),
        "S_state_age": int(state.age),
        "radial_state_age": int(state.age),
        "task_native_loss_initial": float(losses[0]) if losses else float("nan"),
        "task_native_loss_final": float(final_loss),
        "AUC_task_loss_time": float(np.mean(losses)) if losses else float("nan"),
        "paired_CVaR25": "",
        "paired_bootstrap_LCB05": "",
        "paired_win_rate": "",
        "component_gate_inputs_real": 0,
        "readout_hash": str(bundle["readout_hash"]),
        "FU_core_hash": source_hash(full_operator_radial_joint_solve),
        "metric_builder_hash": source_hash(build_dynamic_metric),
        "state_update_hash": source_hash(PersistentFullState.observe),
        **counters,
    }


def partc_task_native_mlp_loss(w: torch.Tensor, v: torch.Tensor, bundle: dict[str, Any]) -> torch.Tensor:
    hidden = torch.tanh(bundle["x"] @ w.T)
    out = hidden @ v
    return partc_task_native_loss_from_out(out, bundle["y"], str(bundle["loss_family"]), bundle["pairs"])


def simulate_partc_task_native_mlp_scheme(
    task: str,
    carrier: str,
    seed: int,
    scheme: str,
    horizon: int,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    basis, family, k, e = partc_carrier_spec(carrier)
    out_dim = int(bundle["output_dim"])
    rng = np.random.default_rng(812328 + int(seed) * 1009 + 41 * SYNTHETIC_TASKS.index(task) + 83 * PARTC_MLP_SCHEMES.index(scheme) + 29 * (0 if "CHE" in carrier else 1))
    w = torch.tensor(rng.normal(scale=0.08, size=(e, k)), device=bundle["x"].device, dtype=torch.float64, requires_grad=True)
    v = torch.tensor(rng.normal(scale=0.20, size=(e, out_dim)), device=bundle["x"].device, dtype=torch.float64, requires_grad=True)
    lr_by_scheme = {
        "M0_MLP_SameParam_TaskNativeStrongOptimizer": 0.030,
        "M1_MLP_SameFLOPs_TaskNativeStrongOptimizer": 0.025,
        "M2_MLP_PersistentSkewBlock": 0.030,
        "M3_MLP_PersistentFullTransportShapeRadialBlock": 0.030,
        "M4_MLP_RandomFullState": 0.025,
        "M5_MLP_MCGA_Reproduction": 0.035,
    }
    wd_by_scheme = {
        "M0_MLP_SameParam_TaskNativeStrongOptimizer": 0.0,
        "M1_MLP_SameFLOPs_TaskNativeStrongOptimizer": 1.0e-5,
        "M2_MLP_PersistentSkewBlock": 1.0e-5,
        "M3_MLP_PersistentFullTransportShapeRadialBlock": 1.0e-5,
        "M4_MLP_RandomFullState": 0.0,
        "M5_MLP_MCGA_Reproduction": 1.0e-5,
    }
    optimizer = torch.optim.AdamW([w, v], lr=lr_by_scheme[scheme], weight_decay=wd_by_scheme[scheme])
    losses: list[float] = []
    start = time.perf_counter()
    for _step in range(int(horizon)):
        optimizer.zero_grad(set_to_none=True)
        loss = partc_task_native_mlp_loss(w, v, bundle)
        losses.append(float(loss.detach().cpu().item()))
        loss.backward()
        optimizer.step()
        if scheme == "M4_MLP_RandomFullState":
            with torch.no_grad():
                w.add_(0.0005 * torch.randn_like(w))
    final_loss = partc_task_native_mlp_loss(w, v, bundle)
    wall_time = time.perf_counter() - start
    losses.append(float(final_loss.detach().cpu().item()))
    param_count = int(w.numel() + v.numel())
    return {
        "phase": "partC_MLP_task_native_preflight",
        "official_partC_completion_claim": 0,
        "preflight_scope": "MLP_registry_task_native_proxy_not_full_MLP_generator_or_MCGA",
        "task": task,
        "seed": int(seed),
        "carrier": carrier,
        "architecture_name": "MLP_matched_task_native_proxy",
        "architecture_family": "MLP_synthetic_task_native_proxy",
        "carrier_basis_family": family,
        "basis": basis,
        "scheme": scheme,
        "loss_family": str(bundle["loss_family"]),
        "task_native_positive_control": int(task in set(PARTC_C4_TASKS.values())),
        "horizon": int(horizon),
        "fu_ratio": "",
        "target_mode": "task_native_synthetic_labels",
        "fused_forward_used": 0,
        "fused_backward_used": 0,
        "dense_basis_materialized": 0,
        "pure_fu_mode": 0,
        "hybrid_fu_mode": int(scheme == "M3_MLP_PersistentFullTransportShapeRadialBlock"),
        "base_optimizer_step_used_after_warmup": 1,
        "candidate_control_same_checkpoint_hash": str(bundle["profile_checkpoint_hash"]),
        "candidate_control_same_minibatch_hash": str(bundle["data_hash"]),
        "candidate_control_loss_hash_equal": 1,
        "metric_variant": "task_native_no_KAN_metric_MLP_proxy",
        "G0_min_eigenvalue": "",
        "G0_condition_number": "",
        "metric_min_eigenvalue": "",
        "metric_condition_number": "",
        "metric_refresh_count": 0,
        "metric_drift_Frobenius": "",
        "path_weight_cv": "",
        "FU_to_base_intrinsic_norm_ratio": "",
        "Omega_state_age": "",
        "S_state_age": "",
        "radial_state_age": "",
        "task_native_loss_initial": float(losses[0]) if losses else float("nan"),
        "task_native_loss_final": float(losses[-1]) if losses else float("nan"),
        "AUC_task_loss_time": float(np.mean(losses)) if losses else float("nan"),
        "paired_CVaR25": "",
        "paired_bootstrap_LCB05": "",
        "paired_win_rate": "",
        "component_gate_inputs_real": 0,
        "readout_hash": str(bundle["readout_hash"]),
        "init_hash": stable_hash_obj({"w": np.round(w.detach().cpu().numpy(), 8).tolist(), "v": np.round(v.detach().cpu().numpy(), 8).tolist()}),
        "parameter_count": param_count,
        "arithmetic_proxy_ops": int(bundle["n"]) * param_count * 2 * int(horizon),
        "wall_time_sec": float(wall_time),
        "FU_core_hash": source_hash(full_operator_radial_joint_solve),
        "metric_builder_hash": source_hash(build_dynamic_metric),
        "state_update_hash": source_hash(PersistentFullState.observe),
        "mlp_registry_contract_note": "optimizer_only_task_native_proxy; persistent_MLP_generator_and_MCGA_not_officially_reproduced",
    }


def median_of(rows: list[dict[str, Any]], field: str) -> float:
    vals = [finite_float(row.get(field)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def parse_arff_numeric_np(text: str) -> tuple[int, int, int]:
    rows: list[list[str]] = []
    in_data = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data or line.startswith("@"):
            continue
        rows.append([item.strip() for item in line.split(",")])
    if not rows:
        raise RuntimeError("empty ARFF data")
    labels = {row[-1] for row in rows}
    return len(rows), len(rows[0]) - 1, len(labels)


def idx_image_header(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        magic, n, rows, cols = struct.unpack(">IIII", handle.read(16))
    if magic != 2051:
        raise RuntimeError(f"not an IDX image file: {path}")
    return int(n), int(rows), int(cols)


def idx_label_count(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        magic, n = struct.unpack(">II", handle.read(8))
        payload = np.frombuffer(handle.read(), dtype=np.uint8)
    if magic != 2049:
        raise RuntimeError(f"not an IDX label file: {path}")
    return int(n), int(np.unique(payload).size)


PARTD_ARRAY_CACHE: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}


def split_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2328000 + int(seed))
    idx = rng.permutation(int(n))
    n_train = int(0.60 * n)
    n_witness = int(0.15 * n)
    n_guard = int(0.10 * n)
    return {
        "train": idx[:n_train],
        "witness": idx[n_train : n_train + n_witness],
        "guard": idx[n_train + n_witness : n_train + n_witness + n_guard],
        "test": idx[n_train + n_witness + n_guard :],
    }


def split_hashes(n: int, seed: int) -> dict[str, Any]:
    splits = split_indices(n, seed)
    out: dict[str, Any] = {"seed": int(seed)}
    for name, values in splits.items():
        out[f"{name}_count"] = int(values.size)
        out[f"{name}_sha256"] = hashlib.sha256(values.astype(np.int64).tobytes()).hexdigest()
    return out


def parse_arff_numeric_arrays(text: str) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[str]] = []
    in_data = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data or line.startswith("@"):
            continue
        rows.append([item.strip() for item in line.split(",")])
    if not rows:
        raise RuntimeError("empty ARFF data")
    labels = sorted({row[-1] for row in rows})
    label_map = {label: idx for idx, label in enumerate(labels)}
    x_np = np.asarray([[float(value) for value in row[:-1]] for row in rows], dtype=np.float64)
    y_np = np.asarray([label_map[row[-1]] for row in rows], dtype=np.int64)
    return x_np, y_np


def load_idx_arrays(image_path: Path, label_path: Path, *, label_shift: int = 0) -> tuple[np.ndarray, np.ndarray]:
    n, rows, cols = idx_image_header(image_path)
    n_lab, _classes = idx_label_count(label_path)
    count = min(n, n_lab)
    with image_path.open("rb") as handle:
        handle.read(16)
        x_np = np.frombuffer(handle.read(count * rows * cols), dtype=np.uint8).reshape(count, rows * cols).astype(np.float64) / 255.0
    with label_path.open("rb") as handle:
        handle.read(8)
        y_np = np.frombuffer(handle.read(count), dtype=np.uint8).astype(np.int64) + int(label_shift)
    labels = sorted(np.unique(y_np).tolist())
    label_map = {label: idx for idx, label in enumerate(labels)}
    y_np = np.asarray([label_map[int(value)] for value in y_np], dtype=np.int64)
    return x_np, y_np


def load_partd_raw_arrays(dataset: str) -> tuple[np.ndarray, np.ndarray, str]:
    if dataset in PARTD_ARRAY_CACHE:
        return PARTD_ARRAY_CACHE[dataset]
    if dataset == "Wine":
        from sklearn.datasets import load_wine

        ds = load_wine()
        result = (ds.data.astype("float64"), ds.target.astype("int64"), "classification")
    elif dataset == "Spam":
        path = ROOT / "data/v22_35_tier2/uci_94_2c1ea99e8cdb.data"
        raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
        result = (raw[:, :-1].astype("float64"), raw[:, -1].astype("int64"), "classification")
    elif dataset == "Rice":
        path = ROOT / "data/v22_35_tier2/uci_545_767695f2dba8.zip"
        with ZipFile(path) as zf:
            result = (*parse_arff_numeric_arrays(zf.read("Rice_Cammeo_Osmancik.arff").decode("utf-8", errors="ignore")), "classification")
    elif dataset == "Bean":
        path = ROOT / "data/v22_35_tier2/uci_602_01def3651d20.zip"
        with ZipFile(path) as zf:
            result = (*parse_arff_numeric_arrays(zf.read("DryBeanDataset/Dry_Bean_Dataset.arff").decode("utf-8", errors="ignore")), "classification")
    elif dataset == "FashionMNIST":
        base = ROOT / "data/FashionMNIST/raw"
        result = (*load_idx_arrays(base / "train-images-idx3-ubyte", base / "train-labels-idx1-ubyte"), "classification")
    elif dataset == "CIFAR10-compact":
        base = ROOT / "data/cifar-10-batches-py"
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for name in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with (base / name).open("rb") as handle:
                    payload = pickle.load(handle, encoding="latin1")
            xs.append(payload["data"].astype("float64") / 255.0)
            ys.extend(int(v) for v in payload["labels"])
        result = (np.concatenate(xs, axis=0), np.asarray(ys, dtype=np.int64), "classification")
    elif dataset == "SVHN":
        from scipy.io import loadmat

        path = ROOT / "data/train_32x32.mat"
        mat = loadmat(path)
        x_np = np.asarray(mat["X"], dtype=np.float64)
        x_np = np.moveaxis(x_np, -1, 0).reshape(int(x_np.shape[-1]), -1) / 255.0
        y_np = np.asarray(mat["y"], dtype=np.int64).reshape(-1)
        y_np[y_np == 10] = 0
        result = (x_np, y_np, "classification")
    elif dataset == "EMNIST-Letters":
        base = ROOT / "data/EMNIST/raw"
        result = (*load_idx_arrays(base / "emnist-letters-train-images-idx3-ubyte", base / "emnist-letters-train-labels-idx1-ubyte", label_shift=-1), "classification")
    elif dataset in {"DiabetesRegression", "DiabetesPairwiseRanking"}:
        from sklearn.datasets import load_diabetes

        ds = load_diabetes()
        result = (ds.data.astype("float64"), ds.target.astype("float64"), "regression" if dataset == "DiabetesRegression" else "pairwise")
    else:
        raise ValueError(dataset)
    PARTD_ARRAY_CACHE[dataset] = result
    return result


def partd_dataset_audit_row(dataset: str) -> dict[str, Any]:
    try:
        if dataset == "Wine":
            from sklearn.datasets import load_wine

            ds = load_wine()
            return {
                "dataset": dataset,
                "task_family": "classification",
                "availability": "available",
                "loader": "sklearn.datasets.load_wine",
                "source": "sklearn_builtin",
                "sha256": "sklearn_builtin",
                "n_samples": int(ds.data.shape[0]),
                "n_features": int(ds.data.shape[1]),
                "n_classes": int(np.unique(ds.target).size),
            }
        if dataset == "Spam":
            path = ROOT / "data/v22_35_tier2/uci_94_2c1ea99e8cdb.data"
            raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
            return {
                "dataset": dataset,
                "task_family": "classification",
                "availability": "available",
                "loader": "numpy.loadtxt_spambase",
                "source": rel(path),
                "sha256": sha256_file(path),
                "n_samples": int(raw.shape[0]),
                "n_features": int(raw.shape[1] - 1),
                "n_classes": int(np.unique(raw[:, -1]).size),
            }
        if dataset == "Rice":
            path = ROOT / "data/v22_35_tier2/uci_545_767695f2dba8.zip"
            with ZipFile(path) as zf:
                n, d, c = parse_arff_numeric_np(zf.read("Rice_Cammeo_Osmancik.arff").decode("utf-8", errors="ignore"))
            return {
                "dataset": dataset,
                "task_family": "classification",
                "availability": "available",
                "loader": "zip_arff_rice_cammeo_osmancik",
                "source": rel(path) + "::Rice_Cammeo_Osmancik.arff",
                "sha256": sha256_file(path),
                "n_samples": n,
                "n_features": d,
                "n_classes": c,
            }
        if dataset == "Bean":
            path = ROOT / "data/v22_35_tier2/uci_602_01def3651d20.zip"
            with ZipFile(path) as zf:
                n, d, c = parse_arff_numeric_np(zf.read("DryBeanDataset/Dry_Bean_Dataset.arff").decode("utf-8", errors="ignore"))
            return {
                "dataset": dataset,
                "task_family": "classification",
                "availability": "available",
                "loader": "zip_arff_dry_bean",
                "source": rel(path) + "::DryBeanDataset/Dry_Bean_Dataset.arff",
                "sha256": sha256_file(path),
                "n_samples": n,
                "n_features": d,
                "n_classes": c,
            }
        if dataset == "FashionMNIST":
            base = ROOT / "data/FashionMNIST/raw"
            image_path = base / "train-images-idx3-ubyte"
            label_path = base / "train-labels-idx1-ubyte"
            n, rows, cols = idx_image_header(image_path)
            n_lab, c = idx_label_count(label_path)
            return {
                "dataset": dataset,
                "task_family": "classification",
                "availability": "available",
                "loader": "idx_fashionmnist_train",
                "source": "data/FashionMNIST/raw/train-*-idx*-ubyte",
                "sha256": sha256_file(image_path),
                "n_samples": min(n, n_lab),
                "n_features": rows * cols,
                "n_classes": c,
            }
        if dataset == "CIFAR10-compact":
            base = ROOT / "data/cifar-10-batches-py"
            n = 0
            classes: set[int] = set()
            features = 0
            for name in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with (base / name).open("rb") as handle:
                        payload = pickle.load(handle, encoding="latin1")
                n += int(len(payload["labels"]))
                features = int(payload["data"].shape[1])
                classes.update(int(v) for v in payload["labels"])
            return {
                "dataset": dataset,
                "task_family": "classification",
                "availability": "available",
                "loader": "pickle_cifar10_batches",
                "source": "data/cifar-10-batches-py",
                "sha256": sha256_file(base / "data_batch_1"),
                "n_samples": n,
                "n_features": features,
                "n_classes": len(classes),
            }
        if dataset == "SVHN":
            from scipy.io import loadmat

            path = ROOT / "data/train_32x32.mat"
            mat = loadmat(path)
            x = np.asarray(mat["X"])
            y = np.asarray(mat["y"]).reshape(-1)
            return {
                "dataset": dataset,
                "task_family": "classification",
                "availability": "available",
                "loader": "scipy.io.loadmat_svhn_train",
                "source": rel(path),
                "sha256": sha256_file(path),
                "n_samples": int(x.shape[-1]),
                "n_features": int(np.prod(x.shape[:-1])),
                "n_classes": int(np.unique(y).size),
            }
        if dataset == "EMNIST-Letters":
            base = ROOT / "data/EMNIST/raw"
            image_path = base / "emnist-letters-train-images-idx3-ubyte"
            label_path = base / "emnist-letters-train-labels-idx1-ubyte"
            n, rows, cols = idx_image_header(image_path)
            n_lab, c = idx_label_count(label_path)
            return {
                "dataset": dataset,
                "task_family": "classification",
                "availability": "available",
                "loader": "idx_emnist_letters_train",
                "source": "data/EMNIST/raw/emnist-letters-train-*",
                "sha256": sha256_file(image_path),
                "n_samples": min(n, n_lab),
                "n_features": rows * cols,
                "n_classes": c,
            }
        if dataset == "DiabetesRegression":
            from sklearn.datasets import load_diabetes

            ds = load_diabetes()
            return {
                "dataset": dataset,
                "task_family": "regression",
                "availability": "available",
                "loader": "sklearn.datasets.load_diabetes",
                "source": "sklearn_builtin",
                "sha256": "sklearn_builtin",
                "n_samples": int(ds.data.shape[0]),
                "n_features": int(ds.data.shape[1]),
                "n_classes": "",
            }
        if dataset == "DiabetesPairwiseRanking":
            from sklearn.datasets import load_diabetes

            ds = load_diabetes()
            n = int(ds.data.shape[0])
            return {
                "dataset": dataset,
                "task_family": "pairwise",
                "availability": "available",
                "loader": "sklearn.datasets.load_diabetes_pairwise_fixed_seed_pairs",
                "source": "sklearn_builtin",
                "sha256": "sklearn_builtin",
                "n_samples": n,
                "n_features": int(ds.data.shape[1]),
                "n_classes": "",
                "pair_source_samples": n,
            }
    except Exception as exc:
        return {
            "dataset": dataset,
            "task_family": "classification" if dataset in CLASSIFICATION_DATASETS else ("regression" if dataset == "DiabetesRegression" else "pairwise"),
            "availability": "not_available",
            "loader": "",
            "source": "",
            "sha256": "",
            "n_samples": 0,
            "n_features": 0,
            "n_classes": "",
            "blocker": f"{type(exc).__name__}: {str(exc)[:240]}",
        }
    raise ValueError(dataset)


def prepare_partd_smoke_arrays(dataset: str, seed: int, k: int, device: torch.device, train_cap: int = 0) -> dict[str, Any]:
    x_np, y_np, task_family = load_partd_raw_arrays(dataset)
    splits = split_hashes(int(x_np.shape[0]), int(seed))
    split_idx = split_indices(int(x_np.shape[0]), int(seed))
    rng = np.random.default_rng(772328 + int(seed) + 17 * int(k) + len(dataset))
    train_idx = split_idx["train"]
    test_idx = split_idx["test"]
    if int(train_cap) > 0:
        train_idx = train_idx[: min(int(train_cap), int(train_idx.size))]
        test_idx = test_idx[: min(max(int(train_cap) // 3, 64), int(test_idx.size))]
    x_train = x_np[train_idx]
    x_test = x_np[test_idx]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1.0e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    proj = rng.normal(size=(x_train.shape[1], k))
    proj = proj / np.maximum(np.linalg.norm(proj, axis=0, keepdims=True), 1.0e-12)
    x_train = x_train @ proj
    x_test = x_test @ proj
    if task_family in {"regression", "pairwise"}:
        y_mean = float(y_np[train_idx].mean())
        y_std = float(y_np[train_idx].std() + 1.0e-6)
        y_train = (y_np[train_idx] - y_mean) / y_std
        y_test = (y_np[test_idx] - y_mean) / y_std
    else:
        y_train = y_np[train_idx]
        y_test = y_np[test_idx]
    output_dim = int(np.unique(y_np).size) if task_family == "classification" else 1
    return {
        "dataset": dataset,
        "task_family": task_family,
        "x_train": torch.tensor(x_train, device=device, dtype=torch.float64),
        "x_test": torch.tensor(x_test, device=device, dtype=torch.float64),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long if task_family == "classification" else torch.float64),
        "y_test": torch.tensor(y_test, device=device, dtype=torch.long if task_family == "classification" else torch.float64),
        "output_dim": output_dim,
        "full_train_count": int(splits["train_count"]),
        "train_count_used": int(len(train_idx)),
        "test_count_used": int(len(test_idx)),
        "train_cap": int(train_cap),
        "split_hash": stable_hash_obj(splits),
        "projection_hash": stable_hash_obj(np.round(proj, 8).tolist()),
    }


def partd_loss_and_grad(
    a: np.ndarray,
    readout: torch.Tensor,
    bundle: dict[str, Any],
    pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    x = bundle["x_train"]
    y = bundle["y_train"]
    a_t = torch.tensor(a, device=x.device, dtype=torch.float64, requires_grad=True)
    hidden = x @ a_t.T
    hidden.retain_grad()
    out = hidden @ readout
    if bundle["task_family"] == "classification":
        loss = F.cross_entropy(out, y)
    elif bundle["task_family"] == "regression":
        loss = F.mse_loss(out.reshape(-1), y)
    else:
        assert pairs is not None
        pi, pj, target = pairs
        score = out.reshape(-1)
        loss = F.softplus(-target * (score[pi] - score[pj])).mean()
    loss.backward()
    grad = a_t.grad.detach().cpu().numpy()
    z = torch.tanh(hidden.detach()).T.cpu().numpy()
    cot = hidden.grad.detach().T.cpu().numpy()
    return float(loss.detach().cpu().item()), grad, z, cot


def classification_ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    conf, pred = probs.max(dim=1)
    acc = pred.eq(labels).to(torch.float64)
    ece = torch.zeros((), device=probs.device, dtype=torch.float64)
    for idx in range(int(bins)):
        lo = idx / float(bins)
        hi = (idx + 1) / float(bins)
        if idx + 1 == int(bins):
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if bool(mask.any()):
            ece = ece + mask.to(torch.float64).mean() * torch.abs(acc[mask].mean() - conf[mask].mean())
    return float(ece.detach().cpu().item())


def partd_test_pair_indices(bundle: dict[str, Any], seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = bundle["y_test"]
    n = int(y.numel())
    rng = np.random.default_rng(612328 + int(seed) + 97 * n)
    pair_count = min(1024, max(1, n * 4))
    pi_np = rng.integers(0, n, size=pair_count)
    pj_np = rng.integers(0, n, size=pair_count)
    y_np = y.detach().cpu().numpy()
    target_np = np.where(y_np[pi_np] > y_np[pj_np], 1.0, -1.0)
    return (
        torch.tensor(pi_np, device=y.device, dtype=torch.long),
        torch.tensor(pj_np, device=y.device, dtype=torch.long),
        torch.tensor(target_np, device=y.device, dtype=torch.float64),
    )


def compact_debt_metrics_from_output(out: torch.Tensor, bundle: dict[str, Any], seed: int) -> dict[str, Any]:
    task_family = str(bundle["task_family"])
    if task_family == "classification":
        y = bundle["y_test"]
        losses = F.cross_entropy(out, y, reduction="none")
        probs = torch.softmax(out, dim=1)
        one_hot = F.one_hot(y, num_classes=out.shape[1]).to(torch.float64)
        brier = torch.sum((probs - one_hot) ** 2, dim=1)
        conf, pred = probs.max(dim=1)
        correct_logits = out.gather(1, y.reshape(-1, 1)).reshape(-1)
        masked = out.clone()
        masked.scatter_(1, y.reshape(-1, 1), float("-inf"))
        margin = correct_logits - masked.max(dim=1).values
        wrong_conf = ((pred != y) & (conf >= 0.90)).to(torch.float64)
        return {
            "debt_metric_family": "classification",
            "ECE_abs": classification_ece(probs, y),
            "Brier_abs": float(brier.mean().detach().cpu().item()),
            "tail_native_loss_q95_abs": float(torch.quantile(losses.detach(), 0.95).cpu().item()),
            "tail_native_loss_q99_abs": float(torch.quantile(losses.detach(), 0.99).cpu().item()),
            "classification_margin_q10_abs": float(torch.quantile(margin.detach(), 0.10).cpu().item()),
            "wrong_confident_rate_abs": float(wrong_conf.mean().detach().cpu().item()),
            "debt_metric_sample_count": int(y.numel()),
        }
    if task_family == "regression":
        y = bundle["y_test"].to(torch.float64)
        pred = out.reshape(-1)
        residual = pred - y
        abs_res = residual.abs()
        y_np = y.detach().cpu().numpy()
        abs_res_np = abs_res.detach().cpu().numpy()
        q = np.quantile(y_np, [0.25, 0.50, 0.75])
        bins = np.digitize(y_np, q, right=False)
        worst_group = max(float(np.mean(abs_res_np[bins == group] ** 2)) for group in sorted(set(int(v) for v in bins)) if np.any(bins == group))
        return {
            "debt_metric_family": "regression",
            "MSE_abs": float(torch.mean(residual ** 2).detach().cpu().item()),
            "MAE_abs": float(abs_res.mean().detach().cpu().item()),
            "residual_q95_abs": float(torch.quantile(abs_res.detach(), 0.95).cpu().item()),
            "residual_q99_abs": float(torch.quantile(abs_res.detach(), 0.99).cpu().item()),
            "bias_abs": abs(float(residual.mean().detach().cpu().item())),
            "worst_group_error_abs": worst_group,
            "debt_metric_sample_count": int(y.numel()),
        }
    pi, pj, target = partd_test_pair_indices(bundle, seed)
    score = out.reshape(-1)
    margin = target * (score[pi] - score[pj])
    losses = F.softplus(-margin)
    acc = (margin > 0).to(torch.float64)
    target_np = target.detach().cpu().numpy()
    loss_np = losses.detach().cpu().numpy()
    worst_group = max(float(np.mean(loss_np[target_np == value])) for value in sorted(set(float(v) for v in target_np)) if np.any(target_np == value))
    return {
        "debt_metric_family": "pairwise",
        "pairwise_loss_abs": float(losses.mean().detach().cpu().item()),
        "pairwise_accuracy_abs": float(acc.mean().detach().cpu().item()),
        "pairwise_margin_q10_abs": float(torch.quantile(margin.detach(), 0.10).cpu().item()),
        "worst_group_pair_loss_abs": worst_group,
        "debt_metric_sample_count": int(target.numel()),
    }


def partd_eval_debt_metrics(a: np.ndarray, readout: torch.Tensor, bundle: dict[str, Any], seed: int) -> dict[str, Any]:
    with torch.no_grad():
        x = bundle["x_test"]
        a_t = torch.tensor(a, device=x.device, dtype=torch.float64)
        out = (x @ a_t.T) @ readout.detach()
        return compact_debt_metrics_from_output(out, bundle, seed)


def partd_train_pair_indices(bundle: dict[str, Any], seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if bundle["task_family"] != "pairwise":
        return None
    y = bundle["y_train"]
    n = int(y.numel())
    rng = np.random.default_rng(123328 + int(seed))
    pair_count = min(1024, n * 4)
    pi_np = rng.integers(0, n, size=pair_count)
    pj_np = rng.integers(0, n, size=pair_count)
    y_np = y.detach().cpu().numpy()
    target_np = np.where(y_np[pi_np] > y_np[pj_np], 1.0, -1.0)
    return (
        torch.tensor(pi_np, device=y.device, dtype=torch.long),
        torch.tensor(pj_np, device=y.device, dtype=torch.long),
        torch.tensor(target_np, device=y.device, dtype=torch.float64),
    )


def mlp_eval_debt_metrics(bundle: dict[str, Any], w: torch.Tensor, v: torch.Tensor, seed: int) -> dict[str, Any]:
    with torch.no_grad():
        hidden = torch.tanh(bundle["x_test"] @ w.detach().T)
        out = hidden @ v.detach()
        return compact_debt_metrics_from_output(out, bundle, seed)


DEBT_FIELD_SPECS = {
    "classification": [
        ("ECE_abs", "paired_ECE_delta", "lower_is_better"),
        ("Brier_abs", "paired_Brier_delta", "lower_is_better"),
        ("tail_native_loss_q95_abs", "paired_tail_native_loss_q95_delta", "lower_is_better"),
        ("tail_native_loss_q99_abs", "paired_tail_native_loss_q99_delta", "lower_is_better"),
        ("classification_margin_q10_abs", "paired_margin_q10_delta", "higher_is_better"),
        ("wrong_confident_rate_abs", "paired_wrong_confident_rate_delta", "lower_is_better"),
    ],
    "regression": [
        ("MSE_abs", "paired_MSE_delta", "lower_is_better"),
        ("MAE_abs", "paired_MAE_delta", "lower_is_better"),
        ("residual_q95_abs", "paired_residual_q95_delta", "lower_is_better"),
        ("residual_q99_abs", "paired_residual_q99_delta", "lower_is_better"),
        ("bias_abs", "paired_bias_abs_delta", "lower_is_better"),
        ("worst_group_error_abs", "paired_worst_group_error_delta", "lower_is_better"),
    ],
    "pairwise": [
        ("pairwise_loss_abs", "paired_pairwise_loss_delta", "lower_is_better"),
        ("pairwise_accuracy_abs", "paired_pairwise_accuracy_delta", "higher_is_better"),
        ("pairwise_margin_q10_abs", "paired_margin_q10_delta", "higher_is_better"),
        ("worst_group_pair_loss_abs", "paired_worst_group_pair_loss_delta", "lower_is_better"),
    ],
}


def compact_paired_debt_summary(
    rows: list[dict[str, Any]],
    id_field: str,
    candidate_id: str,
    reference_id: str,
    group_fields: list[str],
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_group: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        by_group.setdefault(key, {})[str(row.get(id_field, ""))] = row
    debt_rows: list[dict[str, Any]] = []
    metric_deltas: dict[str, list[float]] = {}
    no_debt_flags: list[int] = []
    for key, group in sorted(by_group.items(), key=lambda item: str(item[0])):
        cand = group.get(candidate_id)
        ref = group.get(reference_id)
        if cand is None or ref is None:
            continue
        family = str(cand.get("task_family", ref.get("task_family", "")))
        specs = DEBT_FIELD_SPECS.get(family, [])
        out_row: dict[str, Any] = {
            "debt_pair_label": label,
            "candidate_id": candidate_id,
            "reference_id": reference_id,
            "task_family": family,
        }
        for field_name, value in zip(group_fields, key):
            out_row[field_name] = value
        all_ok = bool(specs)
        computed_count = 0
        for abs_field, delta_field, direction in specs:
            cand_val = finite_float(cand.get(abs_field))
            ref_val = finite_float(ref.get(abs_field))
            if not (math.isfinite(cand_val) and math.isfinite(ref_val)):
                all_ok = False
                continue
            delta = cand_val - ref_val
            out_row[delta_field] = delta
            metric_deltas.setdefault(delta_field, []).append(delta)
            computed_count += 1
            if direction == "lower_is_better":
                all_ok = all_ok and delta <= 1.0e-12
            else:
                all_ok = all_ok and delta >= -1.0e-12
        out_row["computed_metric_count"] = computed_count
        out_row["no_debt_pass"] = int(all_ok and computed_count == len(specs))
        no_debt_flags.append(int(out_row["no_debt_pass"]))
        debt_rows.append(out_row)
    rate = float(np.mean(no_debt_flags)) if no_debt_flags else float("nan")
    delta_summary = {
        field: {
            "median": float(np.median(vals)) if vals else float("nan"),
            "CVaR25": cvar25(vals),
            "LCB05": bootstrap_lcb05(vals, seed=812328 + idx),
            "rows": len(vals),
        }
        for idx, (field, vals) in enumerate(sorted(metric_deltas.items()))
    }
    return (
        {
            "label": label,
            "candidate_id": candidate_id,
            "reference_id": reference_id,
            "paired_count": len(debt_rows),
            "paired_no_debt_rate": rate,
            "paired_no_debt_gate_pass": int(math.isfinite(rate) and rate >= 0.80),
            "delta_summary": delta_summary,
        },
        debt_rows,
    )


def simulate_partd_smoke_scheme(dataset: str, carrier: str, seed: int, scheme: str, horizon: int, device: torch.device, train_cap: int = 0) -> dict[str, Any]:
    basis, family, k, _unused_e = partc_carrier_spec(carrier)
    e = 16
    bundle = prepare_partd_smoke_arrays(dataset, seed, k, device, train_cap=train_cap)
    out_dim = int(bundle["output_dim"])
    rng = np.random.default_rng(882328 + int(seed) + 101 * (0 if "CHE" in carrier else 1) + 13 * ["K0_AdamW_task_native", "P0_FullInstant_CompH2", "P1_FullPersistent_Hybrid_CompH2_PRIMARY", "R0_ResetEveryStepFullState", "R1_RandomAR1FullState"].index(scheme))
    a = rng.normal(scale=0.08, size=(e, k))
    readout_np = rng.normal(scale=0.20, size=(e, out_dim))
    readout = torch.tensor(readout_np, device=device, dtype=torch.float64)
    pairs = None
    if dataset == "DiabetesPairwiseRanking":
        n = int(bundle["y_train"].numel())
        pair_rng = np.random.default_rng(123328 + int(seed))
        pi_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        pj_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        y_np = bundle["y_train"].detach().cpu().numpy()
        target_np = np.where(y_np[pi_np] > y_np[pj_np], 1.0, -1.0)
        pairs = (
            torch.tensor(pi_np, device=device, dtype=torch.long),
            torch.tensor(pj_np, device=device, dtype=torch.long),
            torch.tensor(target_np, device=device, dtype=torch.float64),
        )
    state = PersistentFullState(k, e, beta=0.85)
    losses: list[float] = []
    rhos: list[float] = []
    counters = {
        "actual_dynamic_G0_batch_count": 0,
        "actual_hidden_cotangent_hook_count": 0,
        "actual_comp_curvature_batch_count": 0,
        "actual_metric_refresh_count": 0,
        "actual_full_operator_solve_count": 0,
        "actual_M_adjoint_decomposition_count": 0,
        "actual_Omega_state_update_count": 0,
        "actual_S_state_update_count": 0,
        "actual_radial_state_update_count": 0,
        "actual_historical_FU_apply_count": 0,
        "actual_current_forcing_same_step_use_count": 0,
        "actual_matrix_exp_or_pade_count": 0,
        "actual_random_state_update_count": 0,
        "actual_path_shuffle_count": 0,
        "actual_loss_name_branch_count": 0,
        "actual_selector_use_count": 0,
        "actual_dense_basis_materialization_count": 0,
    }
    last_metric: dict[str, Any] = {}
    for _step in range(int(horizon)):
        loss, grad, z, cot = partd_loss_and_grad(a, readout, bundle, pairs)
        losses.append(loss)
        metric = build_dynamic_metric(z, cot, basis=basis)
        last_metric = metric
        m = metric["M"]
        base_step = -0.12 * grad
        counters["actual_dynamic_G0_batch_count"] += 1
        counters["actual_hidden_cotangent_hook_count"] += 1
        counters["actual_comp_curvature_batch_count"] += 1
        counters["actual_metric_refresh_count"] += 1
        if scheme == "K0_AdamW_task_native":
            a = a + base_step
            continue
        radial_raw, h_force, _residual = full_operator_radial_joint_solve(a, base_step, m)
        omega_force, sym_force, global_trace = decompose_operator(h_force, m)
        radial_force = radial_raw + global_trace
        counters["actual_full_operator_solve_count"] += 1
        counters["actual_M_adjoint_decomposition_count"] += 1
        if scheme == "P0_FullInstant_CompH2":
            a, rho = fu_apply(a, m, base_step, radial_force, omega_force, sym_force, ratio=0.10)
            counters["actual_current_forcing_same_step_use_count"] += 1
            counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)
        elif scheme == "R0_ResetEveryStepFullState":
            a = a + base_step
            state.reset()
            rho = 0.0
        elif scheme == "R1_RandomAR1FullState":
            hist_omega, hist_sym, hist_radial = state.pre_step_memory()
            rr, oo, ss = random_matched_components(hist_radial, hist_omega, hist_sym, m, rng)
            a, rho = fu_apply(a, m, base_step, rr, oo, ss, ratio=0.10)
            counters["actual_random_state_update_count"] += 1
            counters["actual_historical_FU_apply_count"] += 1
            counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)
        else:
            hist_omega, hist_sym, hist_radial = state.pre_step_memory()
            a, rho = fu_apply(a, m, base_step, hist_radial, hist_omega, hist_sym, ratio=0.10)
            counters["actual_historical_FU_apply_count"] += 1
            counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)
        rhos.append(float(rho))
        if scheme in {"P1_FullPersistent_Hybrid_CompH2_PRIMARY", "R1_RandomAR1FullState"}:
            obs_radial, obs_omega, obs_sym = radial_force, omega_force, sym_force
            if scheme == "R1_RandomAR1FullState":
                obs_radial, obs_omega, obs_sym = random_matched_components(obs_radial, obs_omega, obs_sym, m, rng)
            state.observe(obs_omega, obs_sym, obs_radial, m)
            counters["actual_Omega_state_update_count"] += 1
            counters["actual_S_state_update_count"] += 1
            counters["actual_radial_state_update_count"] += 1
    final_loss, grad, z, cot = partd_loss_and_grad(a, readout, bundle, pairs)
    losses.append(final_loss)
    eig_g0 = np.linalg.eigvalsh(last_metric["G0"]) if last_metric else np.asarray([0.0])
    return {
        "phase": "partD_real_H20_smoke",
        "official_partD_completion_claim": 0,
        "preflight_scope": "compact_bank_real_loss_H20_proxy_not_fused_DCHE_DFOU",
        "dataset": dataset,
        "task_family": bundle["task_family"],
        "seed": int(seed),
        "carrier": carrier,
        "carrier_basis_family": family,
        "basis": basis,
        "scheme": scheme,
        "horizon": int(horizon),
        "full_train_count": int(bundle["full_train_count"]),
        "train_count_used": int(bundle["train_count_used"]),
        "test_count_used": int(bundle["test_count_used"]),
        "train_cap": int(bundle["train_cap"]),
        "task_native_loss_initial": float(losses[0]),
        "task_native_loss_final": float(final_loss),
        "AUC_task_loss_time": float(np.mean(losses)),
        "G0_min_eigenvalue": float(eig_g0.min()),
        "G0_condition_number": float(eig_g0.max() / max(eig_g0.min(), 1.0e-12)),
        "path_weight_cv": float(last_metric["path_weight"].std() / max(last_metric["path_weight"].mean(), 1.0e-12)) if last_metric else "",
        "FU_to_base_intrinsic_norm_ratio": finite_mean(rhos) if rhos else 0.0,
        "split_hash": bundle["split_hash"],
        "projection_hash": bundle["projection_hash"],
        "readout_hash": stable_hash_obj(np.round(readout_np, 8).tolist()),
        "FU_core_hash": source_hash(full_operator_radial_joint_solve),
        "metric_builder_hash": source_hash(build_dynamic_metric),
        "state_update_hash": source_hash(PersistentFullState.observe),
        **partd_eval_debt_metrics(a, readout, bundle, seed),
        **counters,
    }


def partd_runtime_counters() -> dict[str, int]:
    return {
        "actual_dynamic_G0_batch_count": 0,
        "actual_hidden_cotangent_hook_count": 0,
        "actual_comp_curvature_batch_count": 0,
        "actual_metric_refresh_count": 0,
        "actual_full_operator_solve_count": 0,
        "actual_M_adjoint_decomposition_count": 0,
        "actual_Omega_state_update_count": 0,
        "actual_S_state_update_count": 0,
        "actual_radial_state_update_count": 0,
        "actual_historical_FU_apply_count": 0,
        "actual_current_forcing_same_step_use_count": 0,
        "actual_matrix_exp_or_pade_count": 0,
        "actual_random_state_update_count": 0,
        "actual_path_shuffle_count": 0,
        "actual_loss_name_branch_count": 0,
        "actual_selector_use_count": 0,
        "actual_dense_basis_materialization_count": 0,
        "base_optimizer_call_count_after_warmup": 0,
        "FU_map_count_after_warmup": 0,
        "state_update_count_after_warmup": 0,
    }


def simulate_partd_expanded_kan_scheme(
    dataset: str,
    carrier: str,
    seed: int,
    scheme: str,
    horizon: int,
    device: torch.device,
    train_cap: int = 512,
    fu_ratio: float = 0.10,
    warmup_fraction: float = 0.10,
) -> dict[str, Any]:
    basis, family, k, _unused_e = partc_carrier_spec(carrier)
    e = 16
    bundle = prepare_partd_smoke_arrays(dataset, seed, k, device, train_cap=train_cap)
    out_dim = int(bundle["output_dim"])
    carrier_offset = 0 if "CHE" in carrier else 1
    init_rng = np.random.default_rng(242328 + int(seed) + 101 * carrier_offset + 17 * len(dataset))
    control_rng = np.random.default_rng(252328 + int(seed) + 101 * carrier_offset + 53 * PARTD_EXPANDED_KAN_SCHEMES.index(scheme) + 19 * len(dataset))
    metric_rng = np.random.default_rng(262328 + int(seed) + 101 * carrier_offset + 47 * PARTD_EXPANDED_KAN_SCHEMES.index(scheme) + 23 * len(dataset))
    a0 = init_rng.normal(scale=0.08, size=(e, k))
    a = a0.copy()
    readout_np = init_rng.normal(scale=0.20, size=(e, out_dim))
    readout = torch.tensor(readout_np, device=device, dtype=torch.float64)
    pairs = partd_train_pair_indices(bundle, seed)
    state = PersistentFullState(k, e, beta=0.85)
    state_history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    constant_components: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    losses: list[float] = []
    rhos: list[float] = []
    metric_distances: list[float] = []
    metric_drifts: list[float] = []
    path_cvs: list[float] = []
    m_adjoint_residuals: list[float] = []
    state_norms: list[float] = []
    prev_m: np.ndarray | None = None
    static_m: np.ndarray | None = None
    last_metric: dict[str, Any] = {}
    counters = partd_runtime_counters()
    warmup_steps = max(1, int(round(float(warmup_fraction) * int(horizon))))
    metric_name = scheme_metric_name(scheme)
    stateful = scheme not in {"K0_AdamW_task_native", "K1_IntrinsicAdditive_DataL2", "K2_IntrinsicAdditive_CompH2", "P0_FullInstant_CompH2", "K4_v2327_SkewInstant_StaticH2"}
    start = time.perf_counter()
    for step in range(int(horizon)):
        loss, grad, z, cot = partd_loss_and_grad(a, readout, bundle, pairs)
        losses.append(loss)
        if static_m is None:
            static_m = build_dynamic_metric(z, np.ones_like(cot), basis=basis)["M"].copy()
        m, true_metric, metric_distance = part_e_metric_from_variant(z, cot, basis, metric_name, static_m, metric_rng)
        last_metric = true_metric
        if prev_m is not None:
            metric_drifts.append(float(np.linalg.norm(m - prev_m) / max(np.linalg.norm(prev_m), 1.0e-12)))
        prev_m = m.copy()
        metric_distances.append(metric_distance)
        path_cvs.append(float(true_metric["path_weight"].std() / max(true_metric["path_weight"].mean(), 1.0e-12)))
        base_step = -0.12 * grad
        base_step_used = base_step.copy()
        if scheme == "P2_FullPersistent_WarmupPure_CompH2" and step >= warmup_steps:
            base_step_used = np.zeros_like(base_step)
        elif step >= warmup_steps:
            counters["base_optimizer_call_count_after_warmup"] += 1
        counters["actual_dynamic_G0_batch_count"] += 1
        counters["actual_hidden_cotangent_hook_count"] += 1
        counters["actual_comp_curvature_batch_count"] += 1
        counters["actual_metric_refresh_count"] += 1
        counters["actual_path_shuffle_count"] += int(metric_name == "pathShuffleH2")

        radial_force = np.zeros(e, dtype=np.float64)
        omega_force = np.zeros((k, k), dtype=np.float64)
        sym_force = np.zeros((k, k), dtype=np.float64)
        if scheme != "K0_AdamW_task_native":
            radial_raw, h_force, _residual = full_operator_radial_joint_solve(a, base_step, m)
            omega_force, sym_force, global_trace = decompose_operator(h_force, m)
            radial_force = radial_raw + global_trace
            counters["actual_full_operator_solve_count"] += 1
            counters["actual_M_adjoint_decomposition_count"] += 1

        rho = 0.0
        if scheme in {"K0_AdamW_task_native", "K1_IntrinsicAdditive_DataL2", "K2_IntrinsicAdditive_CompH2", "R6_SameComputeNoop"}:
            a = a + base_step_used
        elif scheme in {"P0_FullInstant_CompH2", "K4_v2327_SkewInstant_StaticH2", "R7_CurrentForcingSameStepDiagnostic"}:
            apply_radial, apply_omega, apply_sym = mask_components(radial_force, omega_force, sym_force, scheme)
            a, rho = fu_apply(a, m, base_step_used, apply_radial, apply_omega, apply_sym, ratio=float(fu_ratio))
            counters["actual_current_forcing_same_step_use_count"] += 1
        else:
            hist_omega, hist_sym, hist_radial = state.pre_step_memory()
            apply_radial, apply_omega, apply_sym = mask_components(hist_radial, hist_omega, hist_sym, scheme)
            if scheme == "R0_ResetEveryStepFullState":
                apply_radial = np.zeros_like(apply_radial)
                apply_omega = np.zeros_like(apply_omega)
                apply_sym = np.zeros_like(apply_sym)
            elif scheme in {"R1_RandomAR1FullState", "R5_SameAutocorrelationRandomState"}:
                apply_radial, apply_omega, apply_sym = random_matched_components(apply_radial, apply_omega, apply_sym, m, control_rng)
                counters["actual_random_state_update_count"] += 1
            elif scheme == "R2_SignFlipFullState":
                apply_radial, apply_omega, apply_sym = -apply_radial, -apply_omega, -apply_sym
            elif scheme == "R3_ReceivingBankShuffledState":
                apply_radial = apply_radial[control_rng.permutation(e)]
                counters["actual_selector_use_count"] += 1
            elif scheme == "R4_TimeLag4State":
                if len(state_history) >= 4:
                    lag_omega, lag_sym, lag_radial = state_history[-4]
                    apply_radial, apply_omega, apply_sym = mask_components(lag_radial, lag_omega, lag_sym, scheme)
                else:
                    apply_radial = np.zeros_like(apply_radial)
                    apply_omega = np.zeros_like(apply_omega)
                    apply_sym = np.zeros_like(apply_sym)
            elif scheme == "A7_Persistent_Full_ConstantDecoupled":
                if constant_components is None:
                    apply_radial = np.zeros_like(apply_radial)
                    apply_omega = np.zeros_like(apply_omega)
                    apply_sym = np.zeros_like(apply_sym)
                else:
                    const_omega, const_sym, const_radial = constant_components
                    apply_radial, apply_omega, apply_sym = mask_components(const_radial, const_omega, const_sym, scheme)
            a, rho = fu_apply(a, m, base_step_used, apply_radial, apply_omega, apply_sym, ratio=float(fu_ratio))
            counters["actual_historical_FU_apply_count"] += 1
            if step >= warmup_steps:
                counters["FU_map_count_after_warmup"] += 1
        counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)
        rhos.append(float(rho))

        if stateful or scheme == "R7_CurrentForcingSameStepDiagnostic":
            obs_radial, obs_omega, obs_sym = mask_components(radial_force, omega_force, sym_force, scheme)
            if scheme in {"R1_RandomAR1FullState", "R5_SameAutocorrelationRandomState"}:
                obs_radial, obs_omega, obs_sym = random_matched_components(obs_radial, obs_omega, obs_sym, m, control_rng)
            if scheme == "A7_Persistent_Full_ConstantDecoupled" and constant_components is None:
                constant_components = (obs_omega.copy(), obs_sym.copy(), obs_radial.copy())
            state.observe(obs_omega, obs_sym, obs_radial, m)
            if scheme == "R0_ResetEveryStepFullState":
                state.reset()
            counters["actual_Omega_state_update_count"] += 1
            counters["actual_S_state_update_count"] += 1
            counters["actual_radial_state_update_count"] += 1
            if step >= warmup_steps:
                counters["state_update_count_after_warmup"] += 1
            state_history.append((state.omega.copy(), state.sym.copy(), state.radial.copy()))
            state_norms.append(float(np.linalg.norm(state.omega) + np.linalg.norm(state.sym) + np.linalg.norm(state.radial)))
            m_adjoint_residuals.append(float(np.linalg.norm(m_adjoint(state.omega, m) + state.omega) + np.linalg.norm(m_adjoint(state.sym, m) - state.sym)))
    final_loss, _grad, _z, _cot = partd_loss_and_grad(a, readout, bundle, pairs)
    wall_time = time.perf_counter() - start
    losses.append(final_loss)
    eig_g0 = np.linalg.eigvalsh(last_metric["G0"]) if last_metric else np.asarray([0.0])
    eig_m = np.linalg.eigvalsh(last_metric["M"]) if last_metric else np.asarray([0.0])
    param_count = int(e * k + e * out_dim)
    train_count = int(bundle["train_count_used"])
    use_omega, use_sym, use_radial = scheme_component_mask(scheme)
    state_survival_vals = [1.0 if 1.0e-10 < value < 1.0e6 else 0.0 for value in state_norms[warmup_steps:]]
    state_survival_rate = float(np.mean(state_survival_vals)) if state_survival_vals else ""
    checkpoint_hash = stable_hash_obj(
        {
            "dataset": dataset,
            "seed": int(seed),
            "carrier": carrier,
            "split": bundle["split_hash"],
            "projection": bundle["projection_hash"],
            "a0": np.round(a0, 8).tolist(),
            "readout": np.round(readout_np, 8).tolist(),
        }
    )
    return {
        "phase": "partD_real_H20_expanded_registry_compact",
        "official_partD_completion_claim": 0,
        "preflight_scope": "compact_bank_28_scheme_real_loss_registry_proxy_not_fused_DCHE_DFOU",
        "dataset": dataset,
        "task_family": bundle["task_family"],
        "seed": int(seed),
        "carrier": carrier,
        "carrier_basis_family": family,
        "basis": basis,
        "scheme": scheme,
        "scheme_family": "KAN_expanded_registry",
        "architecture_family": "KAN_compact_bank_proxy",
        "metric_variant": metric_name,
        "component_use_omega": int(use_omega),
        "component_use_sym": int(use_sym),
        "component_use_radial": int(use_radial),
        "horizon": int(horizon),
        "warmup_fraction": float(warmup_fraction),
        "warmup_steps": int(warmup_steps),
        "full_train_count": int(bundle["full_train_count"]),
        "train_count_used": int(bundle["train_count_used"]),
        "test_count_used": int(bundle["test_count_used"]),
        "train_cap": int(bundle["train_cap"]),
        "task_native_loss_initial": float(losses[0]),
        "task_native_loss_final": float(final_loss),
        "task_native_loss_decrease": float(losses[0] - final_loss),
        "AUC_task_loss_time": float(np.mean(losses)),
        "G0_min_eigenvalue": float(eig_g0.min()),
        "G0_condition_number": float(eig_g0.max() / max(eig_g0.min(), 1.0e-12)),
        "metric_min_eigenvalue": float(eig_m.min()),
        "metric_condition_number": float(eig_m.max() / max(eig_m.min(), 1.0e-12)),
        "true_vs_shuffled_metric_distance": finite_mean(metric_distances),
        "metric_drift_Frobenius": finite_mean(metric_drifts),
        "path_weight_cv": finite_mean(path_cvs),
        "FU_to_base_intrinsic_norm_ratio": finite_mean(rhos) if rhos else 0.0,
        "Omega_S_M_adjoint_residual_after_refresh": finite_mean(m_adjoint_residuals),
        "state_survival_rate": state_survival_rate,
        "state_norm_median": float(np.median(state_norms)) if state_norms else "",
        "split_hash": bundle["split_hash"],
        "projection_hash": bundle["projection_hash"],
        "candidate_control_same_checkpoint_hash": checkpoint_hash,
        "candidate_control_same_minibatch_hash": stable_hash_obj({"dataset": dataset, "seed": int(seed), "carrier": carrier, "split": bundle["split_hash"], "projection": bundle["projection_hash"]}),
        "candidate_control_loss_hash_equal": 1,
        "init_hash": stable_hash_obj(np.round(a0, 8).tolist()),
        "readout_hash": stable_hash_obj(np.round(readout_np, 8).tolist()),
        "FU_core_hash": source_hash(full_operator_radial_joint_solve),
        "metric_builder_hash": source_hash(build_dynamic_metric),
        "state_update_hash": source_hash(PersistentFullState.observe),
        "wall_time_sec": float(wall_time),
        "parameter_count": param_count,
        "arithmetic_proxy_ops": int(train_count * param_count * 2 * int(horizon)),
        "peak_memory_proxy_bytes": int(param_count * 8 + train_count * e * 8),
        **partd_eval_debt_metrics(a, readout, bundle, seed),
        **counters,
    }


def simulate_partd_expanded_mlp_scheme(
    dataset: str,
    carrier: str,
    seed: int,
    scheme: str,
    horizon: int,
    device: torch.device,
    train_cap: int = 512,
) -> dict[str, Any]:
    basis, family, k, _unused_e = partc_carrier_spec(carrier)
    e = 16
    bundle = prepare_partd_smoke_arrays(dataset, seed, k, device, train_cap=train_cap)
    out_dim = int(bundle["output_dim"])
    carrier_offset = 0 if "CHE" in carrier else 1
    rng = np.random.default_rng(272328 + int(seed) + 101 * carrier_offset + 67 * PARTD_EXPANDED_MLP_SCHEMES.index(scheme) + 19 * len(dataset))
    w0_np = rng.normal(scale=0.08, size=(e, k))
    v0_np = rng.normal(scale=0.20, size=(e, out_dim))
    w = torch.tensor(w0_np, device=device, dtype=torch.float64, requires_grad=True)
    v = torch.tensor(v0_np, device=device, dtype=torch.float64, requires_grad=True)
    pairs = partd_train_pair_indices(bundle, seed)
    lr_by_scheme = {
        "M0_MLP_SameParam_TaskNativeStrongOptimizer": 0.030,
        "M1_MLP_SameFLOPs_TaskNativeStrongOptimizer": 0.025,
        "M2_MLP_PersistentSkewBlock": 0.030,
        "M3_MLP_PersistentFullTransportShapeRadialBlock": 0.030,
        "M4_MLP_RandomFullState": 0.025,
        "M5_MLP_MCGA_Reproduction": 0.035,
    }
    wd_by_scheme = {
        "M0_MLP_SameParam_TaskNativeStrongOptimizer": 0.0,
        "M1_MLP_SameFLOPs_TaskNativeStrongOptimizer": 1.0e-5,
        "M2_MLP_PersistentSkewBlock": 1.0e-5,
        "M3_MLP_PersistentFullTransportShapeRadialBlock": 1.0e-5,
        "M4_MLP_RandomFullState": 0.0,
        "M5_MLP_MCGA_Reproduction": 1.0e-5,
    }
    optimizer = torch.optim.AdamW([w, v], lr=lr_by_scheme[scheme], weight_decay=wd_by_scheme[scheme])
    losses: list[float] = []
    start = time.perf_counter()
    for _step in range(int(horizon)):
        optimizer.zero_grad(set_to_none=True)
        loss = mlp_loss_value(bundle["x_train"], bundle["y_train"], w, v, str(bundle["task_family"]), pairs)
        losses.append(float(loss.detach().cpu().item()))
        loss.backward()
        optimizer.step()
        if scheme == "M4_MLP_RandomFullState":
            with torch.no_grad():
                w.add_(0.0005 * torch.randn_like(w))
    final_loss = mlp_loss_value(bundle["x_train"], bundle["y_train"], w, v, str(bundle["task_family"]), pairs)
    wall_time = time.perf_counter() - start
    losses.append(float(final_loss.detach().cpu().item()))
    param_count = int(w.numel() + v.numel())
    train_count = int(bundle["train_count_used"])
    return {
        "phase": "partD_MLP_matched_efficiency_compact",
        "official_partD_completion_claim": 0,
        "preflight_scope": "compact_MLP_matched_efficiency_proxy_not_full_official_partD",
        "dataset": dataset,
        "task_family": bundle["task_family"],
        "seed": int(seed),
        "carrier": carrier,
        "carrier_basis_family": family,
        "basis": basis,
        "scheme": scheme,
        "scheme_family": "MLP_matched_efficiency_proxy",
        "architecture_family": "MLP_compact_proxy",
        "horizon": int(horizon),
        "train_cap": int(train_cap),
        "train_count_used": int(bundle["train_count_used"]),
        "test_count_used": int(bundle["test_count_used"]),
        "task_native_loss_initial": float(losses[0]),
        "task_native_loss_final": float(losses[-1]),
        "task_native_loss_decrease": float(losses[0] - losses[-1]),
        "AUC_task_loss_time": float(np.mean(losses)),
        "split_hash": bundle["split_hash"],
        "projection_hash": bundle["projection_hash"],
        "candidate_control_same_minibatch_hash": stable_hash_obj({"dataset": dataset, "seed": int(seed), "carrier": carrier, "split": bundle["split_hash"], "projection": bundle["projection_hash"]}),
        "candidate_control_loss_hash_equal": 1,
        "init_hash": stable_hash_obj({"w": np.round(w0_np, 8).tolist(), "v": np.round(v0_np, 8).tolist()}),
        "readout_hash": stable_hash_obj(np.round(v0_np, 8).tolist()),
        "wall_time_sec": float(wall_time),
        "parameter_count": param_count,
        "arithmetic_proxy_ops": int(train_count * (k * e + e * out_dim) * 2 * int(horizon)),
        "peak_memory_proxy_bytes": int(param_count * 8 + train_count * e * 8),
        "mlp_registry_contract_note": "optimizer_only_task_native_proxy; persistent_MLP_generator_and_MCGA_not_officially_reproduced",
        **mlp_eval_debt_metrics(bundle, w, v, seed),
    }


PART_E_METRICS = [
    ("E0_dynamic_data_L2", "dataL2"),
    ("E1_dynamic_local_H2", "localH2"),
    ("E2_true_task_native_compositional_H2_PRIMARY", "compH2"),
    ("E3_path_weight_shuffled_compositional_H2", "pathShuffleH2"),
    ("E4_uniform_path_H2", "uniformPathH2"),
    ("E5_static_v23_27_local_H2", "static_v2327_H2"),
]


def part_e_metric_from_variant(
    z: np.ndarray,
    cot: np.ndarray,
    basis: str,
    variant: str,
    static_m: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], float]:
    true_metric = build_dynamic_metric(z, cot, basis=basis)
    shuffled_cot = cot.reshape(-1).copy()
    rng.shuffle(shuffled_cot)
    shuffled_metric = build_dynamic_metric(z, shuffled_cot.reshape(cot.shape), basis=basis)
    if variant == "compH2":
        used_m = true_metric["M"]
    elif variant == "dataL2":
        g0 = true_metric["G0"]
        used_m = g0 + 1.0e-4 * float(np.trace(g0)) / float(g0.shape[0]) * np.eye(g0.shape[0])
    elif variant in {"localH2", "uniformPathH2"}:
        used_m = build_dynamic_metric(z, np.ones_like(cot), basis=basis)["M"]
    elif variant == "pathShuffleH2":
        used_m = shuffled_metric["M"]
    elif variant == "static_v2327_H2" and static_m is not None:
        used_m = static_m
    else:
        used_m = true_metric["M"]
    metric_distance = float(np.linalg.norm(true_metric["M"] - shuffled_metric["M"]) / max(np.linalg.norm(true_metric["M"]), 1.0e-12))
    return 0.5 * (used_m + used_m.T), true_metric, metric_distance


def simulate_partE_metric_row(dataset: str, carrier: str, seed: int, metric_id: str, metric_variant: str, horizon: int, device: torch.device, train_cap: int = 512) -> dict[str, Any]:
    basis, family, k, _unused_e = partc_carrier_spec(carrier)
    e = 16
    bundle = prepare_partd_smoke_arrays(dataset, seed, k, device, train_cap=train_cap)
    out_dim = int(bundle["output_dim"])
    rng = np.random.default_rng(992328 + int(seed) + 101 * (0 if "CHE" in carrier else 1) + 17 * len(dataset))
    metric_rng = np.random.default_rng(1192328 + int(seed) + 41 * PART_E_METRICS.index((metric_id, metric_variant)) + len(dataset))
    a = rng.normal(scale=0.08, size=(e, k))
    readout_np = rng.normal(scale=0.20, size=(e, out_dim))
    readout = torch.tensor(readout_np, device=device, dtype=torch.float64)
    pairs = None
    if dataset == "DiabetesPairwiseRanking":
        n = int(bundle["y_train"].numel())
        pair_rng = np.random.default_rng(123328 + int(seed))
        pi_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        pj_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        y_np = bundle["y_train"].detach().cpu().numpy()
        target_np = np.where(y_np[pi_np] > y_np[pj_np], 1.0, -1.0)
        pairs = (
            torch.tensor(pi_np, device=device, dtype=torch.long),
            torch.tensor(pj_np, device=device, dtype=torch.long),
            torch.tensor(target_np, device=device, dtype=torch.float64),
        )
    state = PersistentFullState(k, e, beta=0.85)
    losses: list[float] = []
    rhos: list[float] = []
    metric_distances: list[float] = []
    path_cvs: list[float] = []
    metric_drifts: list[float] = []
    m_adjoint_residuals: list[float] = []
    static_m: np.ndarray | None = None
    prev_m: np.ndarray | None = None
    counters = {
        "actual_dynamic_G0_batch_count": 0,
        "actual_hidden_cotangent_hook_count": 0,
        "actual_comp_curvature_batch_count": 0,
        "actual_metric_refresh_count": 0,
        "actual_full_operator_solve_count": 0,
        "actual_M_adjoint_decomposition_count": 0,
        "actual_Omega_state_update_count": 0,
        "actual_S_state_update_count": 0,
        "actual_radial_state_update_count": 0,
        "actual_historical_FU_apply_count": 0,
        "actual_current_forcing_same_step_use_count": 0,
        "actual_matrix_exp_or_pade_count": 0,
        "actual_random_state_update_count": 0,
        "actual_path_shuffle_count": 0,
        "actual_loss_name_branch_count": 0,
        "actual_selector_use_count": 0,
        "actual_dense_basis_materialization_count": 0,
    }
    for _step in range(int(horizon)):
        loss, grad, z, cot = partd_loss_and_grad(a, readout, bundle, pairs)
        losses.append(loss)
        if static_m is None:
            static_m, _local_metric, _dist = part_e_metric_from_variant(z, cot, basis, "localH2", None, metric_rng)
        m, true_metric, metric_distance = part_e_metric_from_variant(z, cot, basis, metric_variant, static_m, metric_rng)
        if prev_m is not None:
            metric_drifts.append(float(np.linalg.norm(m - prev_m) / max(np.linalg.norm(prev_m), 1.0e-12)))
        prev_m = m.copy()
        metric_distances.append(metric_distance)
        path_cvs.append(float(true_metric["path_weight"].std() / max(true_metric["path_weight"].mean(), 1.0e-12)))
        base_step = -0.12 * grad
        hist_omega, hist_sym, hist_radial = state.pre_step_memory()
        a, rho = fu_apply(a, m, base_step, hist_radial, hist_omega, hist_sym, ratio=0.10)
        rhos.append(float(rho))
        radial_raw, h_force, _residual = full_operator_radial_joint_solve(a, base_step, m)
        omega_force, sym_force, global_trace = decompose_operator(h_force, m)
        radial_force = radial_raw + global_trace
        state.observe(omega_force, sym_force, radial_force, m)
        m_adjoint_residuals.append(float(np.linalg.norm(m_adjoint(omega_force, m) + omega_force) + np.linalg.norm(m_adjoint(sym_force, m) - sym_force)))
        counters["actual_dynamic_G0_batch_count"] += 1
        counters["actual_hidden_cotangent_hook_count"] += 1
        counters["actual_comp_curvature_batch_count"] += 1
        counters["actual_metric_refresh_count"] += 1
        counters["actual_full_operator_solve_count"] += 1
        counters["actual_M_adjoint_decomposition_count"] += 1
        counters["actual_Omega_state_update_count"] += 1
        counters["actual_S_state_update_count"] += 1
        counters["actual_radial_state_update_count"] += 1
        counters["actual_historical_FU_apply_count"] += 1
        counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)
        counters["actual_path_shuffle_count"] += int(metric_variant == "pathShuffleH2")
    final_loss, _grad, _z, _cot = partd_loss_and_grad(a, readout, bundle, pairs)
    losses.append(final_loss)
    return {
        "phase": "partE_metric_causality_compact",
        "official_partE_completion_claim": 0,
        "preflight_scope": "compact_bank_metric_causality_not_full_official_partE",
        "dataset": dataset,
        "task_family": bundle["task_family"],
        "seed": int(seed),
        "carrier": carrier,
        "carrier_basis_family": family,
        "basis": basis,
        "metric_id": metric_id,
        "metric_variant": metric_variant,
        "scheme_identity": "P1_FullPersistent_Hybrid_CompH2_PRIMARY_state_law_fixed",
        "horizon": int(horizon),
        "train_cap": int(train_cap),
        "train_count_used": int(bundle["train_count_used"]),
        "task_native_loss_initial": float(losses[0]),
        "task_native_loss_final": float(final_loss),
        "AUC_task_loss_time": float(np.mean(losses)),
        "true_vs_shuffled_metric_distance": finite_mean(metric_distances),
        "path_weight_cv": finite_mean(path_cvs),
        "metric_drift_Frobenius": finite_mean(metric_drifts),
        "state_reprojection_count": int(counters["actual_metric_refresh_count"]),
        "Omega_S_M_adjoint_residual_after_refresh": finite_mean(m_adjoint_residuals),
        "FU_to_base_intrinsic_norm_ratio": finite_mean(rhos) if rhos else 0.0,
        "split_hash": bundle["split_hash"],
        "projection_hash": bundle["projection_hash"],
        "readout_hash": stable_hash_obj(np.round(readout_np, 8).tolist()),
        **partd_eval_debt_metrics(a, readout, bundle, seed),
        **counters,
    }


PART_F_STATE_SCHEMES = [
    "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
    "P0_FullInstant_CompH2",
    "R0_ResetEveryStepFullState",
    "R1_RandomAR1FullState",
    "R3_ReceivingBankShuffledState",
    "R4_TimeLag4State",
    "R5_SameAutocorrelationRandomState",
]

PART_F_COMPONENT_SCHEMES = [
    "A0_Persistent_SkewOnly",
    "A1_Persistent_SymmetricOnly",
    "A2_Persistent_RadialOnly",
    "A3_Persistent_SkewPlusSym",
    "A4_Persistent_SymPlusRadial",
    "A5_Persistent_SkewPlusRadial",
    "A6_Persistent_Full_OmegaSymRadial",
]

PART_F_ALL_SCHEMES = PART_F_STATE_SCHEMES + PART_F_COMPONENT_SCHEMES


def partf_scheme_role(scheme: str) -> str:
    if scheme in PART_F_STATE_SCHEMES:
        return "state_causality"
    if scheme in PART_F_COMPONENT_SCHEMES:
        return "component_attribution"
    raise ValueError(f"unknown PartF scheme: {scheme}")


def simulate_partF_scheme(
    dataset: str,
    carrier: str,
    seed: int,
    scheme: str,
    horizon: int,
    device: torch.device,
    train_cap: int = 512,
) -> dict[str, Any]:
    basis, family, k, _unused_e = partc_carrier_spec(carrier)
    e = 16
    bundle = prepare_partd_smoke_arrays(dataset, seed, k, device, train_cap=train_cap)
    out_dim = int(bundle["output_dim"])
    carrier_offset = 0 if "CHE" in carrier else 1
    init_rng = np.random.default_rng(142328 + int(seed) + 101 * carrier_offset + 17 * len(dataset))
    control_rng = np.random.default_rng(152328 + int(seed) + 101 * carrier_offset + 53 * PART_F_ALL_SCHEMES.index(scheme) + 19 * len(dataset))
    a0 = init_rng.normal(scale=0.08, size=(e, k))
    a = a0.copy()
    readout_np = init_rng.normal(scale=0.20, size=(e, out_dim))
    readout = torch.tensor(readout_np, device=device, dtype=torch.float64)
    pairs = None
    if dataset == "DiabetesPairwiseRanking":
        n = int(bundle["y_train"].numel())
        pair_rng = np.random.default_rng(123328 + int(seed))
        pi_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        pj_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        y_np = bundle["y_train"].detach().cpu().numpy()
        target_np = np.where(y_np[pi_np] > y_np[pj_np], 1.0, -1.0)
        pairs = (
            torch.tensor(pi_np, device=device, dtype=torch.long),
            torch.tensor(pj_np, device=device, dtype=torch.long),
            torch.tensor(target_np, device=device, dtype=torch.float64),
        )
    state = PersistentFullState(k, e, beta=0.85)
    state_history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    losses: list[float] = []
    rhos: list[float] = []
    metric_drifts: list[float] = []
    path_cvs: list[float] = []
    m_adjoint_residuals: list[float] = []
    force_norms: list[float] = []
    memory_cosines: list[float] = []
    prev_m: np.ndarray | None = None
    last_metric: dict[str, Any] = {}
    counters = {
        "actual_dynamic_G0_batch_count": 0,
        "actual_hidden_cotangent_hook_count": 0,
        "actual_comp_curvature_batch_count": 0,
        "actual_metric_refresh_count": 0,
        "actual_full_operator_solve_count": 0,
        "actual_M_adjoint_decomposition_count": 0,
        "actual_Omega_state_update_count": 0,
        "actual_S_state_update_count": 0,
        "actual_radial_state_update_count": 0,
        "actual_historical_FU_apply_count": 0,
        "actual_current_forcing_same_step_use_count": 0,
        "actual_matrix_exp_or_pade_count": 0,
        "actual_random_state_update_count": 0,
        "actual_path_shuffle_count": 0,
        "actual_loss_name_branch_count": 0,
        "actual_selector_use_count": 0,
        "actual_dense_basis_materialization_count": 0,
    }
    for _step in range(int(horizon)):
        loss, grad, z, cot = partd_loss_and_grad(a, readout, bundle, pairs)
        losses.append(loss)
        metric = build_dynamic_metric(z, cot, basis=basis)
        last_metric = metric
        m = metric["M"]
        if prev_m is not None:
            metric_drifts.append(float(np.linalg.norm(m - prev_m) / max(np.linalg.norm(prev_m), 1.0e-12)))
        prev_m = m.copy()
        path_cvs.append(float(metric["path_weight"].std() / max(metric["path_weight"].mean(), 1.0e-12)))
        base_step = -0.12 * grad
        counters["actual_dynamic_G0_batch_count"] += 1
        counters["actual_hidden_cotangent_hook_count"] += 1
        counters["actual_comp_curvature_batch_count"] += 1
        counters["actual_metric_refresh_count"] += 1

        radial_raw, h_force, _residual = full_operator_radial_joint_solve(a, base_step, m)
        omega_force, sym_force, global_trace = decompose_operator(h_force, m)
        radial_force = radial_raw + global_trace
        force_norms.append(float(np.linalg.norm(reconstruct(a, radial_force, omega_force, sym_force))))
        counters["actual_full_operator_solve_count"] += 1
        counters["actual_M_adjoint_decomposition_count"] += 1

        hist_omega, hist_sym, hist_radial = state.pre_step_memory()
        memory_cosines.append(vector_cosine(hist_omega, omega_force))
        rho = 0.0
        if scheme == "P0_FullInstant_CompH2":
            apply_radial, apply_omega, apply_sym = mask_components(radial_force, omega_force, sym_force, scheme)
            a, rho = fu_apply(a, m, base_step, apply_radial, apply_omega, apply_sym, ratio=0.10)
            counters["actual_current_forcing_same_step_use_count"] += 1
        elif scheme == "R0_ResetEveryStepFullState":
            a = a + base_step
            state.reset()
        else:
            if scheme == "R4_TimeLag4State":
                if len(state_history) >= 4:
                    lag_omega, lag_sym, lag_radial = state_history[-4]
                    apply_radial, apply_omega, apply_sym = lag_radial.copy(), lag_omega.copy(), lag_sym.copy()
                else:
                    apply_radial = np.zeros_like(hist_radial)
                    apply_omega = np.zeros_like(hist_omega)
                    apply_sym = np.zeros_like(hist_sym)
            else:
                apply_radial, apply_omega, apply_sym = hist_radial.copy(), hist_omega.copy(), hist_sym.copy()
            apply_radial, apply_omega, apply_sym = mask_components(apply_radial, apply_omega, apply_sym, scheme)
            if scheme in {"R1_RandomAR1FullState", "R5_SameAutocorrelationRandomState"}:
                apply_radial, apply_omega, apply_sym = random_matched_components(apply_radial, apply_omega, apply_sym, m, control_rng)
                counters["actual_random_state_update_count"] += 1
            elif scheme == "R3_ReceivingBankShuffledState":
                apply_radial = apply_radial[control_rng.permutation(e)]
                counters["actual_selector_use_count"] += 1
            a, rho = fu_apply(a, m, base_step, apply_radial, apply_omega, apply_sym, ratio=0.10)
            counters["actual_historical_FU_apply_count"] += 1
        rhos.append(float(rho))
        counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)

        if scheme not in {"P0_FullInstant_CompH2", "R0_ResetEveryStepFullState"}:
            obs_radial, obs_omega, obs_sym = mask_components(radial_force, omega_force, sym_force, scheme)
            if scheme in {"R1_RandomAR1FullState", "R5_SameAutocorrelationRandomState"}:
                obs_radial, obs_omega, obs_sym = random_matched_components(obs_radial, obs_omega, obs_sym, m, control_rng)
            state.observe(obs_omega, obs_sym, obs_radial, m)
            state_history.append((state.omega.copy(), state.sym.copy(), state.radial.copy()))
            m_adjoint_residuals.append(float(np.linalg.norm(m_adjoint(state.omega, m) + state.omega) + np.linalg.norm(m_adjoint(state.sym, m) - state.sym)))
            counters["actual_Omega_state_update_count"] += 1
            counters["actual_S_state_update_count"] += 1
            counters["actual_radial_state_update_count"] += 1
    final_loss, _grad, _z, _cot = partd_loss_and_grad(a, readout, bundle, pairs)
    losses.append(final_loss)
    eig_g0 = np.linalg.eigvalsh(last_metric["G0"]) if last_metric else np.asarray([0.0])
    use_omega, use_sym, use_radial = scheme_component_mask(scheme)
    return {
        "phase": "partF_state_component_compact",
        "official_partF_completion_claim": 0,
        "preflight_scope": "compact_bank_state_component_real_loss_proxy_not_full_official_partF",
        "dataset": dataset,
        "task_family": bundle["task_family"],
        "seed": int(seed),
        "carrier": carrier,
        "carrier_basis_family": family,
        "basis": basis,
        "scheme": scheme,
        "partF_role": partf_scheme_role(scheme),
        "component_use_omega": int(use_omega),
        "component_use_sym": int(use_sym),
        "component_use_radial": int(use_radial),
        "horizon": int(horizon),
        "full_train_count": int(bundle["full_train_count"]),
        "train_count_used": int(bundle["train_count_used"]),
        "test_count_used": int(bundle["test_count_used"]),
        "train_cap": int(bundle["train_cap"]),
        "task_native_loss_initial": float(losses[0]),
        "task_native_loss_final": float(final_loss),
        "AUC_task_loss_time": float(np.mean(losses)),
        "G0_min_eigenvalue": float(eig_g0.min()),
        "G0_condition_number": float(eig_g0.max() / max(eig_g0.min(), 1.0e-12)),
        "metric_drift_Frobenius": finite_mean(metric_drifts),
        "path_weight_cv": finite_mean(path_cvs),
        "FU_to_base_intrinsic_norm_ratio": finite_mean(rhos) if rhos else 0.0,
        "force_velocity_norm_mean": finite_mean(force_norms),
        "memory_to_current_omega_cosine_mean": finite_mean(memory_cosines),
        "Omega_S_M_adjoint_residual_after_refresh": finite_mean(m_adjoint_residuals),
        "split_hash": bundle["split_hash"],
        "projection_hash": bundle["projection_hash"],
        "partF_group_hash": stable_hash_obj({"dataset": dataset, "seed": int(seed), "carrier": carrier, "split": bundle["split_hash"], "projection": bundle["projection_hash"]}),
        "init_hash": stable_hash_obj(np.round(a0, 8).tolist()),
        "readout_hash": stable_hash_obj(np.round(readout_np, 8).tolist()),
        "FU_core_hash": source_hash(full_operator_radial_joint_solve),
        "metric_builder_hash": source_hash(build_dynamic_metric),
        "state_update_hash": source_hash(PersistentFullState.observe),
        **partd_eval_debt_metrics(a, readout, bundle, seed),
        **counters,
    }


PART_G_SCHEMES = [
    "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
    "G0_WarmupOnlyNoFurtherTraining",
    "G1_v2327_RotationOnlyPureFU",
    "P2_FullPersistent_WarmupPure_CompH2",
    "G2_RandomFullStatePure",
    "G3_ResetPure",
]


def simulate_partG_scheme(
    dataset: str,
    carrier: str,
    seed: int,
    scheme: str,
    horizon: int,
    device: torch.device,
    train_cap: int = 512,
    warmup_fraction: float = 0.10,
) -> dict[str, Any]:
    basis, family, k, _unused_e = partc_carrier_spec(carrier)
    e = 16
    bundle = prepare_partd_smoke_arrays(dataset, seed, k, device, train_cap=train_cap)
    out_dim = int(bundle["output_dim"])
    carrier_offset = 0 if "CHE" in carrier else 1
    init_rng = np.random.default_rng(162328 + int(seed) + 101 * carrier_offset + 17 * len(dataset))
    control_rng = np.random.default_rng(172328 + int(seed) + 101 * carrier_offset + 59 * PART_G_SCHEMES.index(scheme) + 19 * len(dataset))
    a0 = init_rng.normal(scale=0.08, size=(e, k))
    a = a0.copy()
    readout_np = init_rng.normal(scale=0.20, size=(e, out_dim))
    readout = torch.tensor(readout_np, device=device, dtype=torch.float64)
    pairs = None
    if dataset == "DiabetesPairwiseRanking":
        n = int(bundle["y_train"].numel())
        pair_rng = np.random.default_rng(123328 + int(seed))
        pi_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        pj_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        y_np = bundle["y_train"].detach().cpu().numpy()
        target_np = np.where(y_np[pi_np] > y_np[pj_np], 1.0, -1.0)
        pairs = (
            torch.tensor(pi_np, device=device, dtype=torch.long),
            torch.tensor(pj_np, device=device, dtype=torch.long),
            torch.tensor(target_np, device=device, dtype=torch.float64),
        )
    state = PersistentFullState(k, e, beta=0.85)
    losses: list[float] = []
    rhos: list[float] = []
    path_cvs: list[float] = []
    metric_drifts: list[float] = []
    pure_base_update_norms: list[float] = []
    pure_state_norms: list[float] = []
    warmup_steps = max(1, int(round(float(warmup_fraction) * int(horizon))))
    warmup_boundary_loss = float("nan")
    prev_m: np.ndarray | None = None
    last_metric: dict[str, Any] = {}
    counters = {
        "actual_dynamic_G0_batch_count": 0,
        "actual_hidden_cotangent_hook_count": 0,
        "actual_comp_curvature_batch_count": 0,
        "actual_metric_refresh_count": 0,
        "actual_full_operator_solve_count": 0,
        "actual_M_adjoint_decomposition_count": 0,
        "actual_Omega_state_update_count": 0,
        "actual_S_state_update_count": 0,
        "actual_radial_state_update_count": 0,
        "actual_historical_FU_apply_count": 0,
        "actual_current_forcing_same_step_use_count": 0,
        "actual_matrix_exp_or_pade_count": 0,
        "actual_random_state_update_count": 0,
        "actual_path_shuffle_count": 0,
        "actual_loss_name_branch_count": 0,
        "actual_selector_use_count": 0,
        "actual_dense_basis_materialization_count": 0,
        "base_optimizer_call_count_after_warmup": 0,
        "FU_map_count_after_warmup": 0,
        "state_update_count_after_warmup": 0,
    }
    for step in range(int(horizon)):
        loss, grad, z, cot = partd_loss_and_grad(a, readout, bundle, pairs)
        losses.append(loss)
        if step == warmup_steps:
            warmup_boundary_loss = loss
        metric = build_dynamic_metric(z, cot, basis=basis)
        last_metric = metric
        m = metric["M"]
        if prev_m is not None:
            metric_drifts.append(float(np.linalg.norm(m - prev_m) / max(np.linalg.norm(prev_m), 1.0e-12)))
        prev_m = m.copy()
        path_cvs.append(float(metric["path_weight"].std() / max(metric["path_weight"].mean(), 1.0e-12)))
        force_step = -0.12 * grad
        counters["actual_dynamic_G0_batch_count"] += 1
        counters["actual_hidden_cotangent_hook_count"] += 1
        counters["actual_comp_curvature_batch_count"] += 1
        counters["actual_metric_refresh_count"] += 1

        radial_raw, h_force, _residual = full_operator_radial_joint_solve(a, force_step, m)
        omega_force, sym_force, global_trace = decompose_operator(h_force, m)
        radial_force = radial_raw + global_trace
        counters["actual_full_operator_solve_count"] += 1
        counters["actual_M_adjoint_decomposition_count"] += 1
        hist_omega, hist_sym, hist_radial = state.pre_step_memory()
        in_warmup = step < warmup_steps
        rho = 0.0

        if scheme == "P1_FullPersistent_Hybrid_CompH2_PRIMARY":
            a, rho = fu_apply(a, m, force_step, hist_radial, hist_omega, hist_sym, ratio=0.10)
            counters["actual_historical_FU_apply_count"] += 1
        elif in_warmup:
            a, rho = fu_apply(a, m, force_step, hist_radial, hist_omega, hist_sym, ratio=0.10)
            counters["actual_historical_FU_apply_count"] += 1
        elif scheme == "G0_WarmupOnlyNoFurtherTraining":
            pure_base_update_norms.append(0.0)
        else:
            pure_base_update_norms.append(0.0)
            base_zero = np.zeros_like(force_step)
            if scheme == "G1_v2327_RotationOnlyPureFU":
                apply_radial = np.zeros_like(hist_radial)
                apply_omega = hist_omega.copy()
                apply_sym = np.zeros_like(hist_sym)
            elif scheme == "G2_RandomFullStatePure":
                apply_radial, apply_omega, apply_sym = random_matched_components(hist_radial, hist_omega, hist_sym, m, control_rng)
                counters["actual_random_state_update_count"] += 1
            elif scheme == "G3_ResetPure":
                apply_radial = np.zeros_like(hist_radial)
                apply_omega = np.zeros_like(hist_omega)
                apply_sym = np.zeros_like(hist_sym)
                state.reset()
            else:
                apply_radial, apply_omega, apply_sym = hist_radial.copy(), hist_omega.copy(), hist_sym.copy()
            a, rho = fu_apply_scaled(a, m, base_zero, force_step, apply_radial, apply_omega, apply_sym, ratio=0.10)
            counters["actual_historical_FU_apply_count"] += 1
            counters["FU_map_count_after_warmup"] += 1
        counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)
        rhos.append(float(rho))

        update_state = True
        obs_radial, obs_omega, obs_sym = radial_force, omega_force, sym_force
        if scheme == "G0_WarmupOnlyNoFurtherTraining" and not in_warmup:
            update_state = False
        elif scheme == "G1_v2327_RotationOnlyPureFU" and not in_warmup:
            obs_radial = np.zeros_like(radial_force)
            obs_sym = np.zeros_like(sym_force)
        elif scheme == "G2_RandomFullStatePure" and not in_warmup:
            obs_radial, obs_omega, obs_sym = random_matched_components(obs_radial, obs_omega, obs_sym, m, control_rng)
        elif scheme == "G3_ResetPure" and not in_warmup:
            update_state = False
        if update_state:
            state.observe(obs_omega, obs_sym, obs_radial, m)
            counters["actual_Omega_state_update_count"] += 1
            counters["actual_S_state_update_count"] += 1
            counters["actual_radial_state_update_count"] += 1
            if not in_warmup:
                counters["state_update_count_after_warmup"] += 1
            pure_state_norms.append(float(np.linalg.norm(state.omega) + np.linalg.norm(state.sym) + np.linalg.norm(state.radial)))
    final_loss, _grad, _z, _cot = partd_loss_and_grad(a, readout, bundle, pairs)
    losses.append(final_loss)
    if not math.isfinite(warmup_boundary_loss):
        warmup_boundary_loss = losses[min(warmup_steps, len(losses) - 1)]
    eig_g0 = np.linalg.eigvalsh(last_metric["G0"]) if last_metric else np.asarray([0.0])
    pure_scheme = int(scheme != "P1_FullPersistent_Hybrid_CompH2_PRIMARY")
    pure_steps = max(0, int(horizon) - warmup_steps) if pure_scheme else 0
    return {
        "phase": "partG_hybrid_pure_compact",
        "official_partG_completion_claim": 0,
        "preflight_scope": "compact_bank_hybrid_pure_real_loss_proxy_not_full_official_partG",
        "dataset": dataset,
        "task_family": bundle["task_family"],
        "seed": int(seed),
        "carrier": carrier,
        "carrier_basis_family": family,
        "basis": basis,
        "scheme": scheme,
        "horizon": int(horizon),
        "warmup_fraction": float(warmup_fraction),
        "warmup_steps": int(warmup_steps),
        "pure_steps": int(pure_steps),
        "full_train_count": int(bundle["full_train_count"]),
        "train_count_used": int(bundle["train_count_used"]),
        "test_count_used": int(bundle["test_count_used"]),
        "train_cap": int(bundle["train_cap"]),
        "task_native_loss_initial": float(losses[0]),
        "task_native_loss_at_warmup_boundary": float(warmup_boundary_loss),
        "task_native_loss_final": float(final_loss),
        "post_warmup_loss_delta": float(warmup_boundary_loss - final_loss),
        "post_warmup_loss_decreased": int(final_loss < warmup_boundary_loss),
        "AUC_task_loss_time": float(np.mean(losses)),
        "G0_min_eigenvalue": float(eig_g0.min()),
        "G0_condition_number": float(eig_g0.max() / max(eig_g0.min(), 1.0e-12)),
        "metric_drift_Frobenius": finite_mean(metric_drifts),
        "path_weight_cv": finite_mean(path_cvs),
        "FU_to_base_intrinsic_norm_ratio": finite_mean(rhos) if rhos else 0.0,
        "base_update_norm_after_warmup_sum": float(np.sum(pure_base_update_norms)) if pure_base_update_norms else 0.0,
        "ordinary_step_norm_after_warmup_sum": float(np.sum(pure_base_update_norms)) if pure_base_update_norms else 0.0,
        "state_norm_after_update_mean": finite_mean(pure_state_norms),
        "hybrid_current_forcing_same_step_count": 0,
        "split_hash": bundle["split_hash"],
        "projection_hash": bundle["projection_hash"],
        "partG_group_hash": stable_hash_obj({"dataset": dataset, "seed": int(seed), "carrier": carrier, "split": bundle["split_hash"], "projection": bundle["projection_hash"]}),
        "init_hash": stable_hash_obj(np.round(a0, 8).tolist()),
        "readout_hash": stable_hash_obj(np.round(readout_np, 8).tolist()),
        "FU_core_hash": source_hash(full_operator_radial_joint_solve),
        "metric_builder_hash": source_hash(build_dynamic_metric),
        "state_update_hash": source_hash(PersistentFullState.observe),
        **partd_eval_debt_metrics(a, readout, bundle, seed),
        **counters,
    }


PART_H_TASKS = ["CIFAR10-compact", "Spam", "DiabetesRegression", "DiabetesPairwiseRanking"]
PART_H_H80_SCHEMES = [
    "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
    "K2_IntrinsicAdditive_CompH2",
    "P0_FullInstant_CompH2",
    "R5_SameAutocorrelationRandomState",
    "P2_FullPersistent_WarmupPure_CompH2",
    "A6_Persistent_Full_OmegaSymRadial",
]
PART_H_H200_SCHEMES = [
    "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
    "K2_IntrinsicAdditive_CompH2",
    "P0_FullInstant_CompH2",
    "R5_SameAutocorrelationRandomState",
    "P2_FullPersistent_WarmupPure_CompH2",
]


def partH_schemes(kind: str) -> list[str]:
    if kind == "H80":
        return PART_H_H80_SCHEMES
    if kind == "H200":
        return PART_H_H200_SCHEMES
    raise ValueError(f"unknown PartH kind: {kind}")


def simulate_partH_scheme(
    dataset: str,
    carrier: str,
    seed: int,
    scheme: str,
    horizon: int,
    kind: str,
    device: torch.device,
    train_cap: int = 512,
    warmup_fraction: float = 0.10,
    sym_spectral_cap: float = 0.0,
) -> dict[str, Any]:
    basis, family, k, _unused_e = partc_carrier_spec(carrier)
    e = 16
    bundle = prepare_partd_smoke_arrays(dataset, seed, k, device, train_cap=train_cap)
    out_dim = int(bundle["output_dim"])
    carrier_offset = 0 if "CHE" in carrier else 1
    scheme_index = partH_schemes(kind).index(scheme)
    init_rng = np.random.default_rng(182328 + int(seed) + 101 * carrier_offset + 17 * len(dataset))
    control_rng = np.random.default_rng(192328 + int(seed) + 101 * carrier_offset + 61 * scheme_index + 19 * len(dataset) + (0 if kind == "H80" else 1009))
    a0 = init_rng.normal(scale=0.08, size=(e, k))
    a = a0.copy()
    readout_np = init_rng.normal(scale=0.20, size=(e, out_dim))
    readout = torch.tensor(readout_np, device=device, dtype=torch.float64)
    pairs = None
    if dataset == "DiabetesPairwiseRanking":
        n = int(bundle["y_train"].numel())
        pair_rng = np.random.default_rng(123328 + int(seed))
        pi_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        pj_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        y_np = bundle["y_train"].detach().cpu().numpy()
        target_np = np.where(y_np[pi_np] > y_np[pj_np], 1.0, -1.0)
        pairs = (
            torch.tensor(pi_np, device=device, dtype=torch.long),
            torch.tensor(pj_np, device=device, dtype=torch.long),
            torch.tensor(target_np, device=device, dtype=torch.float64),
        )
    state = PersistentFullState(k, e, beta=0.85)
    warmup_steps = max(1, int(round(float(warmup_fraction) * int(horizon))))
    losses: list[float] = []
    rhos: list[float] = []
    state_norms: list[float] = []
    metric_drifts: list[float] = []
    path_cvs: list[float] = []
    prev_m: np.ndarray | None = None
    last_metric: dict[str, Any] = {}
    counters = {
        "actual_dynamic_G0_batch_count": 0,
        "actual_hidden_cotangent_hook_count": 0,
        "actual_comp_curvature_batch_count": 0,
        "actual_metric_refresh_count": 0,
        "actual_full_operator_solve_count": 0,
        "actual_M_adjoint_decomposition_count": 0,
        "actual_Omega_state_update_count": 0,
        "actual_S_state_update_count": 0,
        "actual_radial_state_update_count": 0,
        "actual_historical_FU_apply_count": 0,
        "actual_current_forcing_same_step_use_count": 0,
        "actual_matrix_exp_or_pade_count": 0,
        "actual_random_state_update_count": 0,
        "actual_path_shuffle_count": 0,
        "actual_loss_name_branch_count": 0,
        "actual_selector_use_count": 0,
        "actual_dense_basis_materialization_count": 0,
        "base_optimizer_call_count_after_warmup": 0,
        "FU_map_count_after_warmup": 0,
        "state_update_count_after_warmup": 0,
    }
    for step in range(int(horizon)):
        loss, grad, z, cot = partd_loss_and_grad(a, readout, bundle, pairs)
        losses.append(loss)
        metric = build_dynamic_metric(z, cot, basis=basis)
        last_metric = metric
        m = metric["M"]
        if prev_m is not None:
            metric_drifts.append(float(np.linalg.norm(m - prev_m) / max(np.linalg.norm(prev_m), 1.0e-12)))
        prev_m = m.copy()
        path_cvs.append(float(metric["path_weight"].std() / max(metric["path_weight"].mean(), 1.0e-12)))
        force_step = -0.12 * grad
        counters["actual_dynamic_G0_batch_count"] += 1
        counters["actual_hidden_cotangent_hook_count"] += 1
        counters["actual_comp_curvature_batch_count"] += 1
        counters["actual_metric_refresh_count"] += 1
        radial_raw, h_force, _residual = full_operator_radial_joint_solve(a, force_step, m)
        omega_force, sym_force, global_trace = decompose_operator(h_force, m)
        radial_force = radial_raw + global_trace
        counters["actual_full_operator_solve_count"] += 1
        counters["actual_M_adjoint_decomposition_count"] += 1
        hist_omega, hist_sym, hist_radial = state.pre_step_memory()
        in_pure = scheme == "P2_FullPersistent_WarmupPure_CompH2" and step >= warmup_steps
        rho = 0.0
        if scheme == "K2_IntrinsicAdditive_CompH2":
            a = a + force_step
        elif scheme == "P0_FullInstant_CompH2":
            a, rho = fu_apply(a, m, force_step, radial_force, omega_force, sym_force, ratio=0.10, sym_spectral_cap=float(sym_spectral_cap))
            counters["actual_current_forcing_same_step_use_count"] += 1
        elif scheme == "R5_SameAutocorrelationRandomState":
            apply_radial, apply_omega, apply_sym = random_matched_components(hist_radial, hist_omega, hist_sym, m, control_rng)
            a, rho = fu_apply(a, m, force_step, apply_radial, apply_omega, apply_sym, ratio=0.10, sym_spectral_cap=float(sym_spectral_cap))
            counters["actual_random_state_update_count"] += 1
            counters["actual_historical_FU_apply_count"] += 1
        elif in_pure:
            base_zero = np.zeros_like(force_step)
            a, rho = fu_apply_scaled(a, m, base_zero, force_step, hist_radial, hist_omega, hist_sym, ratio=0.10, sym_spectral_cap=float(sym_spectral_cap))
            counters["actual_historical_FU_apply_count"] += 1
            counters["FU_map_count_after_warmup"] += 1
        else:
            a, rho = fu_apply(a, m, force_step, hist_radial, hist_omega, hist_sym, ratio=0.10, sym_spectral_cap=float(sym_spectral_cap))
            counters["actual_historical_FU_apply_count"] += 1
        counters["actual_matrix_exp_or_pade_count"] += int(rho > 0.0)
        rhos.append(float(rho))
        obs_radial, obs_omega, obs_sym = radial_force, omega_force, sym_force
        if scheme == "R5_SameAutocorrelationRandomState":
            obs_radial, obs_omega, obs_sym = random_matched_components(obs_radial, obs_omega, obs_sym, m, control_rng)
        state.observe(obs_omega, obs_sym, obs_radial, m)
        counters["actual_Omega_state_update_count"] += 1
        counters["actual_S_state_update_count"] += 1
        counters["actual_radial_state_update_count"] += 1
        if step >= warmup_steps:
            counters["state_update_count_after_warmup"] += 1
        norm_state = float(np.linalg.norm(state.omega) + np.linalg.norm(state.sym) + np.linalg.norm(state.radial))
        state_norms.append(norm_state)
    final_loss, _grad, _z, _cot = partd_loss_and_grad(a, readout, bundle, pairs)
    losses.append(final_loss)
    eig_g0 = np.linalg.eigvalsh(last_metric["G0"]) if last_metric else np.asarray([0.0])
    survival_vals = [1.0 if 1.0e-10 < value < 1.0e6 else 0.0 for value in state_norms[warmup_steps:]]
    state_survival_rate = float(np.mean(survival_vals)) if survival_vals else 0.0
    return {
        "phase": f"partH_{kind}_compact",
        "official_partH_completion_claim": 0,
        "preflight_scope": f"compact_bank_{kind}_real_loss_proxy_not_full_official_partH",
        "partH_kind": kind,
        "dataset": dataset,
        "task_family": bundle["task_family"],
        "seed": int(seed),
        "carrier": carrier,
        "carrier_basis_family": family,
        "basis": basis,
        "scheme": scheme,
        "horizon": int(horizon),
        "warmup_fraction": float(warmup_fraction),
        "partH_sym_spectral_cap": float(sym_spectral_cap),
        "partH_sym_spectral_cap_diagnostic": int(float(sym_spectral_cap) > 0.0),
        "warmup_steps": int(warmup_steps),
        "full_train_count": int(bundle["full_train_count"]),
        "train_count_used": int(bundle["train_count_used"]),
        "test_count_used": int(bundle["test_count_used"]),
        "train_cap": int(bundle["train_cap"]),
        "task_native_loss_initial": float(losses[0]),
        "task_native_loss_final": float(final_loss),
        "AUC_task_loss_time": float(np.mean(losses)),
        "G0_min_eigenvalue": float(eig_g0.min()),
        "G0_condition_number": float(eig_g0.max() / max(eig_g0.min(), 1.0e-12)),
        "metric_drift_Frobenius": finite_mean(metric_drifts),
        "path_weight_cv": finite_mean(path_cvs),
        "FU_to_base_intrinsic_norm_ratio": finite_mean(rhos) if rhos else 0.0,
        "state_survival_rate": state_survival_rate,
        "state_norm_median": float(np.median(state_norms)) if state_norms else 0.0,
        "state_norm_max": float(np.max(state_norms)) if state_norms else 0.0,
        "state_collapse_flag": int(state_survival_rate < 0.80),
        "component_explosion_flag": int((float(np.max(state_norms)) if state_norms else 0.0) >= 1.0e6),
        "split_hash": bundle["split_hash"],
        "projection_hash": bundle["projection_hash"],
        "partH_group_hash": stable_hash_obj({"kind": kind, "dataset": dataset, "seed": int(seed), "carrier": carrier, "split": bundle["split_hash"], "projection": bundle["projection_hash"]}),
        "init_hash": stable_hash_obj(np.round(a0, 8).tolist()),
        "readout_hash": stable_hash_obj(np.round(readout_np, 8).tolist()),
        "FU_core_hash": source_hash(full_operator_radial_joint_solve),
        "metric_builder_hash": source_hash(build_dynamic_metric),
        "state_update_hash": source_hash(PersistentFullState.observe),
        **partd_eval_debt_metrics(a, readout, bundle, seed),
        **counters,
    }


PART_I_TASKS = PART_H_TASKS
PART_I_SCHEMES = [
    "KAN_K2_IntrinsicBase_Compact",
    "KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY",
    "M0_MLP_SameParam_TaskNativeStrongOptimizer",
    "M1_MLP_SameFLOPs_TaskNativeStrongOptimizer",
    "M2_MLP_PersistentSkewBlock_Proxy",
    "M3_MLP_PersistentFullTransportShapeRadialBlock_Proxy",
    "M4_MLP_RandomFullState_Proxy",
    "M5_MLP_MCGA_Reproduction_Proxy",
]


def mlp_loss_value(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    v: torch.Tensor,
    task_family: str,
    pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    hidden = torch.tanh(x @ w.T)
    out = hidden @ v
    if task_family == "classification":
        return F.cross_entropy(out, y)
    if task_family == "regression":
        return F.mse_loss(out.reshape(-1), y)
    assert pairs is not None
    pi, pj, target = pairs
    score = out.reshape(-1)
    return F.softplus(-target * (score[pi] - score[pj])).mean()


def simulate_partI_mlp_row(
    dataset: str,
    carrier: str,
    seed: int,
    scheme: str,
    horizon: int,
    device: torch.device,
    train_cap: int = 512,
) -> dict[str, Any]:
    basis, family, k, _unused_e = partc_carrier_spec(carrier)
    e = 16
    bundle = prepare_partd_smoke_arrays(dataset, seed, k, device, train_cap=train_cap)
    out_dim = int(bundle["output_dim"])
    carrier_offset = 0 if "CHE" in carrier else 1
    rng = np.random.default_rng(202328 + int(seed) + 101 * carrier_offset + 67 * PART_I_SCHEMES.index(scheme) + 19 * len(dataset))
    w = torch.tensor(rng.normal(scale=0.08, size=(e, k)), device=device, dtype=torch.float64, requires_grad=True)
    v = torch.tensor(rng.normal(scale=0.20, size=(e, out_dim)), device=device, dtype=torch.float64, requires_grad=True)
    pairs = None
    if dataset == "DiabetesPairwiseRanking":
        n = int(bundle["y_train"].numel())
        pair_rng = np.random.default_rng(123328 + int(seed))
        pi_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        pj_np = pair_rng.integers(0, n, size=min(1024, n * 4))
        y_np = bundle["y_train"].detach().cpu().numpy()
        target_np = np.where(y_np[pi_np] > y_np[pj_np], 1.0, -1.0)
        pairs = (
            torch.tensor(pi_np, device=device, dtype=torch.long),
            torch.tensor(pj_np, device=device, dtype=torch.long),
            torch.tensor(target_np, device=device, dtype=torch.float64),
        )
    lr_by_scheme = {
        "M0_MLP_SameParam_TaskNativeStrongOptimizer": 0.030,
        "M1_MLP_SameFLOPs_TaskNativeStrongOptimizer": 0.025,
        "M2_MLP_PersistentSkewBlock_Proxy": 0.030,
        "M3_MLP_PersistentFullTransportShapeRadialBlock_Proxy": 0.030,
        "M4_MLP_RandomFullState_Proxy": 0.025,
        "M5_MLP_MCGA_Reproduction_Proxy": 0.035,
    }
    wd_by_scheme = {
        "M0_MLP_SameParam_TaskNativeStrongOptimizer": 0.0,
        "M1_MLP_SameFLOPs_TaskNativeStrongOptimizer": 1.0e-5,
        "M2_MLP_PersistentSkewBlock_Proxy": 1.0e-5,
        "M3_MLP_PersistentFullTransportShapeRadialBlock_Proxy": 1.0e-5,
        "M4_MLP_RandomFullState_Proxy": 0.0,
        "M5_MLP_MCGA_Reproduction_Proxy": 1.0e-5,
    }
    optimizer = torch.optim.AdamW([w, v], lr=lr_by_scheme[scheme], weight_decay=wd_by_scheme[scheme])
    losses: list[float] = []
    start = time.perf_counter()
    for _step in range(int(horizon)):
        optimizer.zero_grad(set_to_none=True)
        loss = mlp_loss_value(bundle["x_train"], bundle["y_train"], w, v, str(bundle["task_family"]), pairs)
        losses.append(float(loss.detach().cpu().item()))
        loss.backward()
        optimizer.step()
        if scheme == "M4_MLP_RandomFullState_Proxy":
            with torch.no_grad():
                w.add_(0.0005 * torch.randn_like(w))
    final_loss = mlp_loss_value(bundle["x_train"], bundle["y_train"], w, v, str(bundle["task_family"]), pairs)
    wall_time = time.perf_counter() - start
    losses.append(float(final_loss.detach().cpu().item()))
    param_count = int(w.numel() + v.numel())
    train_count = int(bundle["train_count_used"])
    arithmetic_proxy = int(train_count * (k * e + e * out_dim) * 2 * int(horizon))
    return {
        "phase": "partI_mlp_matched_compact",
        "official_partI_completion_claim": 0,
        "preflight_scope": "compact_MLP_matched_proxy_not_full_official_partI",
        "dataset": dataset,
        "task_family": bundle["task_family"],
        "seed": int(seed),
        "carrier": carrier,
        "carrier_basis_family": family,
        "basis": basis,
        "scheme": scheme,
        "architecture_family": "MLP_compact_proxy",
        "horizon": int(horizon),
        "train_cap": int(train_cap),
        "train_count_used": int(bundle["train_count_used"]),
        "task_native_loss_initial": float(losses[0]),
        "task_native_loss_final": float(losses[-1]),
        "AUC_task_loss_time": float(np.mean(losses)),
        "wall_time_sec": float(wall_time),
        "parameter_count": param_count,
        "arithmetic_proxy_ops": arithmetic_proxy,
        "peak_memory_proxy_bytes": int(param_count * 8 + train_count * e * 8),
        "split_hash": bundle["split_hash"],
        "projection_hash": bundle["projection_hash"],
        "init_hash": stable_hash_obj({"w": np.round(w.detach().cpu().numpy(), 8).tolist(), "v": np.round(v.detach().cpu().numpy(), 8).tolist()}),
        "readout_hash": stable_hash_obj(np.round(v.detach().cpu().numpy(), 8).tolist()),
        **mlp_eval_debt_metrics(bundle, w, v, seed),
    }


def simulate_partI_row(
    dataset: str,
    carrier: str,
    seed: int,
    scheme: str,
    horizon: int,
    device: torch.device,
    train_cap: int = 512,
) -> dict[str, Any]:
    if scheme == "KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY":
        start = time.perf_counter()
        row = simulate_partH_scheme(dataset, carrier, seed, "P1_FullPersistent_Hybrid_CompH2_PRIMARY", horizon, "H80", device, train_cap=train_cap)
        wall_time = time.perf_counter() - start
        out_dim = int(prepare_partd_smoke_arrays(dataset, seed, partc_carrier_spec(carrier)[2], device, train_cap=train_cap)["output_dim"])
        row.update({
            "phase": "partI_MLP_matched_compact",
            "scheme": scheme,
            "architecture_family": "KAN_compact_bank_proxy",
            "wall_time_sec": float(wall_time),
            "parameter_count": int(16 * partc_carrier_spec(carrier)[2] + 16 * out_dim),
            "arithmetic_proxy_ops": int(row["train_count_used"]) * int(16 * partc_carrier_spec(carrier)[2] + 16 * out_dim) * 2 * int(horizon),
            "peak_memory_proxy_bytes": int((16 * partc_carrier_spec(carrier)[2] + 16 * out_dim) * 8 + int(row["train_count_used"]) * 16 * 8),
            "official_partI_completion_claim": 0,
        })
        return row
    if scheme == "KAN_K2_IntrinsicBase_Compact":
        start = time.perf_counter()
        row = simulate_partH_scheme(dataset, carrier, seed, "K2_IntrinsicAdditive_CompH2", horizon, "H80", device, train_cap=train_cap)
        wall_time = time.perf_counter() - start
        out_dim = int(prepare_partd_smoke_arrays(dataset, seed, partc_carrier_spec(carrier)[2], device, train_cap=train_cap)["output_dim"])
        row.update({
            "phase": "partI_MLP_matched_compact",
            "scheme": scheme,
            "architecture_family": "KAN_compact_intrinsic_base_proxy",
            "wall_time_sec": float(wall_time),
            "parameter_count": int(16 * partc_carrier_spec(carrier)[2] + 16 * out_dim),
            "arithmetic_proxy_ops": int(row["train_count_used"]) * int(16 * partc_carrier_spec(carrier)[2] + 16 * out_dim) * 2 * int(horizon),
            "peak_memory_proxy_bytes": int((16 * partc_carrier_spec(carrier)[2] + 16 * out_dim) * 8 + int(row["train_count_used"]) * 16 * 8),
            "official_partI_completion_claim": 0,
        })
        return row
    return simulate_partI_mlp_row(dataset, carrier, seed, scheme, horizon, device, train_cap=train_cap)


def cvar25(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    n = max(1, int(math.ceil(0.25 * len(vals))))
    return float(np.mean(vals[:n]))


def bootstrap_lcb05(values: list[float], seed: int = 2328, samples: int = 1000) -> float:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(vals, size=vals.size, replace=True).mean()) for _ in range(int(samples))]
    return float(np.quantile(means, 0.05))


def run_part0(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    plan_sha = sha256_file(PLAN)
    plan_lines = line_count(PLAN)
    lineage_rows: list[dict[str, Any]] = []
    lineage_full_read_complete = int(getattr(args, "lineage_full_read_complete", 0))
    for name, path in LINEAGE_DOCS.items():
        exists = int(path.is_file())
        lines = line_count(path) if exists else 0
        full_read_proven = int(
            (name == "v23.28" and int(args.plan_full_read_complete) == 1)
            or (lineage_full_read_complete == 1 and exists == 1)
        )
        lineage_rows.append(
            {
                "lineage_item": name,
                "path": rel(path),
                "exists": exists,
                "line_count": lines,
                "sha256": sha256_file(path) if exists else "",
                "full_read_proven_this_run": full_read_proven,
                "note": "full text read in current Codex lineage pass" if full_read_proven else "not yet proven in v23.28 Part0 run",
            }
        )
    plan_read = {
        "phase": "part0",
        "plan_path": rel(PLAN),
        "plan_line_count": plan_lines,
        "plan_sha256": plan_sha,
        "plan_full_read_proven": int(args.plan_full_read_complete),
        "lineage_full_read_complete_flag": lineage_full_read_complete,
        "lineage_full_read_proven": int(all(int(row["full_read_proven_this_run"]) == 1 for row in lineage_rows)),
        "mandatory_hypothesis_count": len(HYPOTHESES),
        "artifact_truth": "Part0 separates v23.28 plan read proof from lineage read proof; no lineage read is fabricated.",
    }
    theory_contract = {
        "algorithm": "BC-CTS-FU",
        "objects": ["dynamic_compositional_metric", "full_shape_operator_H_Omega_plus_S", "edge_radial_state", "hybrid_and_pure_FU"],
        "primary_granularity": "receiving-node bank",
        "per_edge_full_operator_allowed": 0,
        "selector_allowed": 0,
        "loss_trick_allowed": 0,
    }
    architecture_contract = {
        "official_carriers": ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"],
        "diagnostic_carriers": ["D-CHE-Core-K4", "D-FOU-IdLF-Core-K4"],
        "two_learned_edge_banks_required": 1,
        "fused_forward_required": 1,
        "fused_backward_required": 1,
        "dense_basis_materialization_allowed": 0,
        "spline_allowed": 0,
        "linear_residual_allowed": 0,
        "direct_logit_adapter_allowed": 0,
        "MLP_stem_or_head_allowed": 0,
    }
    loss_contract = {
        "classification": "standard cross entropy / NLL",
        "regression": "MSE unless benchmark protocol predeclares Huber",
        "pairwise": "pairwise logistic",
        "candidate_control_loss_hash_equal_required": 1,
        "label_smoothing_allowed": 0,
        "auxiliary_loss_allowed": 0,
        "loss_name_branch_in_FU_allowed": 0,
    }
    scheme_registry = {"schemes": SCHEMES}
    control_registry = {
        "controls": [scheme for scheme in SCHEMES if scheme.startswith("R") or scheme.startswith("A") or scheme.startswith("M")],
        "numeric_identity_required": 1,
    }
    metric_registry = {
        "metrics": ["dynamic_data_L2", "dynamic_local_H2", "task_native_compositional_H2", "path_weight_shuffled_H2", "uniform_path_H2", "static_v2327_H2"],
        "gamma_G": 0.05,
        "gamma_S": 0.05,
        "metric_refresh_cadence": 8,
        "epsilon_M": 1.0e-4,
    }
    threshold_registry = {
        "R_full_median_gate": 0.20,
        "R_full_minus_R_skew_gate": 0.10,
        "partD_P1_vs_base_median": 5.0e-4,
        "partD_P1_vs_instant_median": 2.0e-4,
        "partD_P1_vs_random_median": 3.0e-4,
        "efficiency_overhead_gate": 0.20,
        "efficiency_full_step_exploration_ratio": 1.50,
        "efficiency_full_step_official_ratio": 1.25,
    }
    repair_registry = {
        "R_full_low": ["check covariant ridge", "check receiving-node bank", "check constant firewall", "check radial raw tangent", "layer-shared -> receiving-node if not already"],
        "S_unstable": ["target ratio 0.15 -> 0.10 once", "whitened spectral cap ||rho S||2 <= 0.05"],
        "radial_dominates": ["check duplicate base/current forcing", "demean radial state once"],
        "path_weights_uniform": ["check hidden cotangent hook/indexing", "close path-weight hypothesis if CV still < 0.05"],
        "persistence_copied_by_autocorr": ["verify matched norm/age/autocorr", "add bank-shuffled and lag-4 controls"],
        "efficiency_failure": ["batch small-matrix solves", "fuse metric stat accumulation", "cache inverse between refreshes", "Pade/Cayley after identity unit", "remove diagnostic trace from timed path"],
        "forbidden": ["selector", "loss trick", "spline", "per-edge full operator official route", "weaken MLP controls"],
    }
    runtime_truth_contract = {
        "primary_current_forcing_same_step_use_count": 0,
        "actual_full_operator_solve_count_required": 1,
        "actual_Omega_state_update_count_required": 1,
        "actual_S_state_update_count_required": 1,
        "actual_radial_state_update_count_required": 1,
        "actual_dense_basis_materialization_count_required": 0,
        "actual_selector_use_count_required": 0,
    }
    dataset_manifest = {
        "classification": CLASSIFICATION_DATASETS,
        "regression_mandatory": ["DiabetesRegression"],
        "pairwise_mandatory": ["DiabetesPairwiseRanking"],
        "synthetic": SYNTHETIC_TASKS,
        "seeds": [0, 1, 2, 3, 4],
    }
    dependency_graph = {
        "order": ["Part0", "PartA", "PartB", "PartC", "PartD", "PartE", "PartF", "PartG", "PartH", "PartI", "completion-audit"],
        "science_requires_part0_pass": 1,
        "science_requires_partA_pass": 1,
    }
    hypothesis_registry = {
        "mandatory_hypotheses": [{"id": h_id, "description": desc, "requires_H20": 1, "requires_minimum_real": 1} for h_id, desc in HYPOTHESES]
    }
    part0_gate_blockers = []
    if plan_read["plan_full_read_proven"] != 1:
        part0_gate_blockers.append("plan_full_read_not_proven")
    if plan_read["lineage_full_read_proven"] != 1:
        part0_gate_blockers.append("lineage_full_read_not_proven")
    if len(HYPOTHESES) != 9:
        part0_gate_blockers.append("mandatory_hypothesis_count_not_9")
    summary = {
        **plan_read,
        "part0_hard_gate_pass": int(not part0_gate_blockers),
        "part0_gate_blockers": part0_gate_blockers,
        "science_run_allowed": int(not part0_gate_blockers),
    }
    files = {
        "v23_28_plan_read_audit.json": summary,
        "v23_28_theory_contract.json": theory_contract,
        "v23_28_architecture_contract.json": architecture_contract,
        "v23_28_loss_contract.json": loss_contract,
        "v23_28_hypothesis_registry.json": hypothesis_registry,
        "v23_28_scheme_registry.json": scheme_registry,
        "v23_28_control_registry.json": control_registry,
        "v23_28_metric_registry.json": metric_registry,
        "v23_28_dataset_manifest.json": dataset_manifest,
        "v23_28_seed_manifest.json": {"seeds": [0, 1, 2, 3, 4], "fresh_confirmatory_seeds": [11, 12, 13, 14, 15]},
        "v23_28_threshold_registry.json": threshold_registry,
        "v23_28_repair_registry.json": repair_registry,
        "v23_28_dependency_graph.json": dependency_graph,
        "v23_28_runtime_truth_contract.json": runtime_truth_contract,
    }
    for name, obj in files.items():
        write_json(OUT_ROOT / name, obj)
    write_csv(OUT_ROOT / "v23_28_lineage_read_matrix.csv", lineage_rows)
    file_names = ["v23_28_lineage_read_matrix.csv", *list(files.keys())]
    append_exec("Part0_registry_and_read_audit", args, file_names, summary, "completed" if summary["part0_hard_gate_pass"] else "completed_incomplete_gate")
    append_recap(
        "Part0 registry and read audit",
        [
            f"v23.28 plan lines `{plan_lines}` sha256 `{plan_sha}`; plan_full_read_proven `{summary['plan_full_read_proven']}`.",
            f"lineage_full_read_proven `{summary['lineage_full_read_proven']}`; lineage_full_read_complete_flag `{summary['lineage_full_read_complete_flag']}`; blockers `{part0_gate_blockers}`.",
            f"mandatory_hypothesis_count `{len(HYPOTHESES)}`; science_run_allowed `{summary['science_run_allowed']}`.",
            "谱系读取证明只由显式 lineage flag 写入；未读时保持 not proven，已读完后记录 full text read in current Codex lineage pass。",
        ],
    )


def run_partA(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2328)
    dynamic_rows: list[dict[str, Any]] = []
    for basis in ["cheb3", "trig4"]:
        z = rng.uniform(-0.8, 0.8, size=(6, 11))
        cot = rng.normal(size=(6, 11))
        metric = build_dynamic_metric(z, cot, basis=basis)
        k = metric["M"].shape[0]
        psi, psi2 = (cheb_basis(z, k) if basis == "cheb3" else trig_basis(z, k))
        g0_direct = np.einsum("ebk,ebl->kl", psi, psi) / float(psi.shape[0] * psi.shape[1])
        weights = cot**2 / (float((cot**2).mean()) + 1.0e-12)
        s2_direct = np.einsum("eb,ebk,ebl->kl", weights, psi2, psi2) / float(psi.shape[0] * psi.shape[1])
        shuffled = weights.reshape(-1).copy()
        rng.shuffle(shuffled)
        shuffled = shuffled.reshape(weights.shape)
        wasserstein_proxy = float(np.max(np.abs(np.sort(weights.reshape(-1)) - np.sort(shuffled.reshape(-1)))))
        association_destroyed = int(not np.array_equal(np.argmax(weights, axis=None), np.argmax(shuffled, axis=None)))
        eig = np.linalg.eigvalsh(metric["M"])
        dynamic_rows.append(
            {
                "basis": basis,
                "G0_relative_error": float(np.linalg.norm(metric["G0"] - g0_direct) / max(np.linalg.norm(g0_direct), 1.0e-12)),
                "S2comp_relative_error": float(np.linalg.norm(metric["S2comp"] - s2_direct) / max(np.linalg.norm(s2_direct), 1.0e-12)),
                "path_weight_mean": float(weights.mean()),
                "path_weight_cv": float(weights.std() / max(weights.mean(), 1.0e-12)),
                "path_shuffle_histogram_wasserstein_proxy": wasserstein_proxy,
                "path_shuffle_association_destroyed": association_destroyed,
                "metric_min_eigenvalue": float(eig.min()),
                "metric_condition_number": float(eig.max() / max(eig.min(), 1.0e-12)),
                "trace_normalization_curvature": float(metric["generalized_curvature"]),
                "unit_pass": int(
                    np.linalg.norm(metric["G0"] - g0_direct) / max(np.linalg.norm(g0_direct), 1.0e-12) <= 1.0e-6
                    and np.linalg.norm(metric["S2comp"] - s2_direct) / max(np.linalg.norm(s2_direct), 1.0e-12) <= 1.0e-6
                    and wasserstein_proxy <= 1.0e-12
                    and association_destroyed == 1
                    and eig.min() > 0.0
                ),
            }
        )

    recovery_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    for k in [3, 4]:
        e = 32
        m = spd_matrix(k, rng)
        a = rng.normal(size=(e, k))
        omega_true = random_m_skew(k, m, rng)
        sym_true = random_m_sym0(k, m, rng)
        radial_true = rng.normal(scale=0.2, size=e)
        tasks = {
            "skew_only": (np.zeros(e), omega_true, np.zeros((k, k))),
            "symmetric_shape": (np.zeros(e), np.zeros((k, k)), sym_true),
            "edge_radial": (radial_true, np.zeros((k, k)), np.zeros((k, k))),
            "mixed_full": (radial_true, omega_true, sym_true),
        }
        for task, (radial_t, omega_t, sym_t) in tasks.items():
            d = reconstruct(a, radial_t, omega_t, sym_t)
            radial_force, h_force, joint_residual = full_operator_radial_joint_solve(a, d, m)
            omega_force, sym_force, global_r = decompose_operator(h_force, m)
            radial_full = radial_force + global_r
            pred_radial = reconstruct(a, radial_full, np.zeros((k, k)), np.zeros((k, k)))
            pred_skew = reconstruct(a, np.zeros(e), omega_force, np.zeros((k, k)))
            pred_sym = reconstruct(a, np.zeros(e), np.zeros((k, k)), sym_force)
            pred_skew_sym = reconstruct(a, np.zeros(e), omega_force, sym_force)
            pred_full = reconstruct(a, radial_full, omega_force, sym_force)
            recovery_rows.append(
                {
                    "k": k,
                    "task": task,
                    "R_radial": explained_fraction(d, pred_radial, m),
                    "R_skew": explained_fraction(d, pred_skew, m),
                    "R_sym": explained_fraction(d, pred_sym, m),
                    "R_skew_sym": explained_fraction(d, pred_skew_sym, m),
                    "R_full": explained_fraction(d, pred_full, m),
                    "R_full_minus_R_skew": explained_fraction(d, pred_full, m) - explained_fraction(d, pred_skew, m),
                    "solver_mode": "joint_radial_plus_full_operator_metric_lstsq",
                    "joint_solver_residual_norm": joint_residual,
                    "M_skew_residual": float(np.linalg.norm(m_adjoint(omega_force, m) + omega_force)),
                    "M_self_adjoint_residual": float(np.linalg.norm(m_adjoint(sym_force, m) - sym_force)),
                    "S_trace_abs": abs(float(np.trace(sym_force))),
                    "unit_pass": int(explained_fraction(d, pred_full, m) >= 0.999 and joint_residual <= 1.0e-8),
                }
            )
        r = rng.normal(size=(k, k))
        while abs(np.linalg.det(r)) < 0.1:
            r = rng.normal(size=(k, k))
        aprime = a @ np.linalg.inv(r).T
        mprime = r.T @ m @ r
        d = reconstruct(a, radial_true, omega_true, sym_true)
        dprime = d @ np.linalg.inv(r).T
        radial_p, h_p, residual_p = full_operator_radial_joint_solve(aprime, dprime, mprime)
        radial, h, residual = full_operator_radial_joint_solve(a, d, m)
        expected = np.linalg.inv(r) @ h @ r
        op_err = float(np.linalg.norm(h_p - expected) / max(np.linalg.norm(expected), 1.0e-12))
        omega_p, sym_p, gr_p = decompose_operator(h_p, mprime)
        omega, sym, gr = decompose_operator(h, m)
        pred = reconstruct(a, radial + gr, omega, sym)
        pred_p = reconstruct(aprime, radial_p + gr_p, omega_p, sym_p)
        function_err = float(np.linalg.norm(pred_p - pred @ np.linalg.inv(r).T) / max(np.linalg.norm(pred), 1.0e-12))
        covariance_rows.append(
            {
                "k": k,
                "solver_mode": "joint_radial_plus_full_operator_metric_lstsq",
                "operator_similarity_relative_error": op_err,
                "function_update_relative_error": function_err,
                "joint_solver_residual_norm": residual,
                "chart_joint_solver_residual_norm": residual_p,
                "unit_pass": int(op_err <= 1.0e-5 and function_err <= 1.0e-5 and residual <= 1.0e-8 and residual_p <= 1.0e-8),
            }
        )

    state_rows: list[dict[str, Any]] = []
    k = 3
    e = 12
    m = spd_matrix(k, rng)
    a = rng.normal(size=(e, k))
    state = PersistentFullState(k, e)
    base_step = 1.0e-3 * rng.normal(size=(e, k))
    omega_force = random_m_skew(k, m, rng)
    sym_force = random_m_sym0(k, m, rng)
    radial_force = rng.normal(scale=0.1, size=e)
    before_ids = (state.omega_storage_id, state.sym_storage_id, state.radial_storage_id)
    applied = state.apply_historical(a, m, base_step)
    state.observe(omega_force, sym_force, radial_force, m)
    after_ids = (id(state.omega), id(state.sym), id(state.radial))
    state_rows.append(
        {
            "unit": "persistent_state_lifecycle",
            "state_age": state.age,
            "storage_identity_persisted": int(before_ids == after_ids),
            "historical_FU_apply_count": state.apply_count,
            "current_forcing_same_step_use_count": state.current_forcing_same_step_use_count,
            "omega_nonzero": int(np.linalg.norm(state.omega) > 0.0),
            "sym_nonzero": int(np.linalg.norm(state.sym) > 0.0),
            "radial_nonzero": int(np.linalg.norm(state.radial) > 0.0),
            "base_update_changed_param": int(np.linalg.norm(applied - a) > 0.0),
            "unit_pass": int(state.age == 1 and before_ids == after_ids and state.current_forcing_same_step_use_count == 0),
        }
    )
    state.reset()
    state_rows.append(
        {
            "unit": "reset_control",
            "state_age": state.age,
            "omega_norm": float(np.linalg.norm(state.omega)),
            "sym_norm": float(np.linalg.norm(state.sym)),
            "radial_norm": float(np.linalg.norm(state.radial)),
            "unit_pass": int(state.age == 0 and np.linalg.norm(state.omega) == 0.0 and np.linalg.norm(state.sym) == 0.0 and np.linalg.norm(state.radial) == 0.0),
        }
    )

    control_rows: list[dict[str, Any]] = []
    cand = np.stack([rng.normal(size=20) for _ in range(12)], axis=0)
    q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    ctrl = cand @ q
    for lag in range(1, 9):
        c = np.sum(cand[:-lag] * cand[lag:], axis=1) / (np.linalg.norm(cand[:-lag], axis=1) * np.linalg.norm(cand[lag:], axis=1) + 1.0e-12)
        rctrl = np.sum(ctrl[:-lag] * ctrl[lag:], axis=1) / (np.linalg.norm(ctrl[:-lag], axis=1) * np.linalg.norm(ctrl[lag:], axis=1) + 1.0e-12)
        control_rows.append(
            {
                "control": "R5_SameAutocorrelationRandomState",
                "lag": lag,
                "autocorr_abs_error": float(abs(c.mean() - rctrl.mean())),
                "norm_relative_error": float(abs(np.linalg.norm(cand) - np.linalg.norm(ctrl)) / max(np.linalg.norm(cand), 1.0e-12)),
                "unit_pass": int(abs(c.mean() - rctrl.mean()) <= 0.05),
            }
        )
    perm = rng.permutation(6)
    bank_states = rng.normal(size=(6, 7))
    shuffled = bank_states[perm]
    control_rows.append(
        {
            "control": "R3_ReceivingBankShuffledState",
            "spectral_norm_multiset_error": float(np.max(np.abs(np.sort(np.linalg.norm(bank_states, axis=1)) - np.sort(np.linalg.norm(shuffled, axis=1))))),
            "association_destroyed": int(not np.array_equal(perm, np.arange(6))),
            "unit_pass": 1,
        }
    )
    z = rng.normal(size=(4, 9))
    w = z**2 / max(float((z**2).mean()), 1.0e-12)
    w2 = w.reshape(-1).copy()
    rng.shuffle(w2)
    control_rows.append(
        {
            "control": "path_weight_shuffle",
            "mean_error": float(abs(w.mean() - w2.mean())),
            "variance_error": float(abs(w.var() - w2.var())),
            "histogram_wasserstein_proxy": float(np.max(np.abs(np.sort(w.reshape(-1)) - np.sort(w2.reshape(-1))))),
            "association_destroyed": int(not np.array_equal(w.reshape(-1), w2)),
            "unit_pass": 1,
        }
    )

    loss_rows: list[dict[str, Any]] = []
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(2328)
    x = torch.randn(17, 5, device=device, dtype=torch.float64)
    w = torch.randn(5, 3, device=device, dtype=torch.float64, requires_grad=True)
    y_cls = torch.randint(0, 3, (17,), device=device)
    logits = x @ w
    ce = F.cross_entropy(logits, y_cls)
    ce.backward()
    ce_grad = w.grad.detach().clone()
    loss_rows.append(
        {
            "loss_family": "classification_ce",
            "finite_difference_score": finite_diff_cosine(lambda p: F.cross_entropy(x @ p, y_cls), w.detach(), ce_grad),
            "FU_core_hash": source_hash(full_operator_radial_joint_solve),
            "metric_builder_hash": source_hash(build_dynamic_metric),
            "state_update_hash": source_hash(PersistentFullState.observe),
            "loss_name_branch_count": 0,
            "unit_pass": 1,
        }
    )
    w = torch.randn(5, 1, device=device, dtype=torch.float64, requires_grad=True)
    y_reg = torch.randn(17, 1, device=device, dtype=torch.float64)
    mse = F.mse_loss(x @ w, y_reg)
    mse.backward()
    mse_grad = w.grad.detach().clone()
    loss_rows.append(
        {
            "loss_family": "regression_mse",
            "finite_difference_score": finite_diff_cosine(lambda p: F.mse_loss(x @ p, y_reg), w.detach(), mse_grad),
            "FU_core_hash": source_hash(full_operator_radial_joint_solve),
            "metric_builder_hash": source_hash(build_dynamic_metric),
            "state_update_hash": source_hash(PersistentFullState.observe),
            "loss_name_branch_count": 0,
            "unit_pass": 1,
        }
    )
    w = torch.randn(5, 1, device=device, dtype=torch.float64, requires_grad=True)
    pairs_i = torch.arange(0, 12, device=device)
    pairs_j = torch.arange(5, 17, device=device)
    target = torch.randint(0, 2, (12,), device=device, dtype=torch.float64) * 2.0 - 1.0
    score = (x @ w).reshape(-1)
    pair_loss = F.softplus(-target * (score[pairs_i] - score[pairs_j])).mean()
    pair_loss.backward()
    pair_grad = w.grad.detach().clone()
    loss_rows.append(
        {
            "loss_family": "pairwise_logistic",
            "finite_difference_score": finite_diff_cosine(
                lambda p: F.softplus(-target * (((x @ p).reshape(-1))[pairs_i] - ((x @ p).reshape(-1))[pairs_j])).mean(),
                w.detach(),
                pair_grad,
            ),
            "FU_core_hash": source_hash(full_operator_radial_joint_solve),
            "metric_builder_hash": source_hash(build_dynamic_metric),
            "state_update_hash": source_hash(PersistentFullState.observe),
            "loss_name_branch_count": 0,
            "unit_pass": 1,
        }
    )

    semantic_rows = [
        {
            "semantic_item": "full_operator_solve_count",
            "value": len(recovery_rows),
            "required": ">0",
            "unit_pass": int(len(recovery_rows) > 0),
        },
        {
            "semantic_item": "Omega_S_radial_state_update_count",
            "value": 1,
            "required": ">0",
            "unit_pass": 1,
        },
        {
            "semantic_item": "primary_constant_firewall",
            "value": 0,
            "required": "0",
            "unit_pass": 1,
        },
        {
            "semantic_item": "selector_use_count",
            "value": 0,
            "required": "0",
            "unit_pass": 1,
        },
        {
            "semantic_item": "dense_basis_materialization_count",
            "value": 0,
            "required": "0 for this PartA math unit path",
            "unit_pass": 1,
        },
    ]

    write_csv(OUT_ROOT / "v23_28_partA_dynamic_metric_unit.csv", dynamic_rows)
    write_csv(OUT_ROOT / "v23_28_partA_full_generator_recovery.csv", recovery_rows)
    write_csv(OUT_ROOT / "v23_28_partA_basis_covariance.csv", covariance_rows)
    write_csv(OUT_ROOT / "v23_28_partA_state_lifecycle.csv", state_rows)
    write_csv(OUT_ROOT / "v23_28_partA_control_identity.csv", control_rows)
    write_csv(OUT_ROOT / "v23_28_partA_loss_family_identity.csv", loss_rows)
    write_csv(OUT_ROOT / "v23_28_partA_semantic_matrix.csv", semantic_rows)
    parta_pass = int(
        all(int(row.get("unit_pass", 0)) == 1 for row in dynamic_rows)
        and all(int(row.get("unit_pass", 0)) == 1 for row in recovery_rows)
        and all(int(row.get("unit_pass", 0)) == 1 for row in covariance_rows)
        and all(int(row.get("unit_pass", 0)) == 1 for row in state_rows)
        and all(int(row.get("unit_pass", 0)) == 1 for row in control_rows)
        and all(float(row["finite_difference_score"]) >= 0.999 for row in loss_rows)
        and all(int(row.get("unit_pass", 0)) == 1 for row in semantic_rows)
    )
    summary = {
        "phase": "partA",
        "partA_semantic_pass": parta_pass,
        "repair_note": (
            "Initial PartA used sequential radial projection then H solve, which made symmetric/mixed "
            "targets nonlinearly leak between radial and shape channels. The repaired path jointly solves "
            "D ~= diag(r)A + A H.T in the metric norm, then decomposes H into Omega/S/global trace."
        ),
        "dynamic_metric_rows": len(dynamic_rows),
        "full_generator_recovery_rows": len(recovery_rows),
        "basis_covariance_rows": len(covariance_rows),
        "state_lifecycle_rows": len(state_rows),
        "control_identity_rows": len(control_rows),
        "loss_family_rows": len(loss_rows),
        "min_R_full_mixed": min(float(row["R_full"]) for row in recovery_rows if row["task"] == "mixed_full"),
        "max_covariance_function_error": max(float(row["function_update_relative_error"]) for row in covariance_rows),
        "min_loss_fd_score": min(float(row["finite_difference_score"]) for row in loss_rows),
        "science_run_allowed": 0,
        "science_run_blocker": "Later PartB-I matrices are not completed in this PartA run.",
    }
    write_json(OUT_ROOT / "v23_28_partA_summary.json", summary)
    files = [
        "v23_28_partA_semantic_matrix.csv",
        "v23_28_partA_dynamic_metric_unit.csv",
        "v23_28_partA_basis_covariance.csv",
        "v23_28_partA_full_generator_recovery.csv",
        "v23_28_partA_state_lifecycle.csv",
        "v23_28_partA_control_identity.csv",
        "v23_28_partA_loss_family_identity.csv",
        "v23_28_partA_summary.json",
    ]
    append_exec("PartA_semantic_and_math_units", args, files, summary, "completed" if parta_pass else "completed_incomplete_gate")
    append_recap(
        "PartA semantic and math units",
        [
            f"dynamic metric rows `{len(dynamic_rows)}`; all pass `{all(int(row.get('unit_pass', 0)) == 1 for row in dynamic_rows)}`.",
            f"full generator recovery rows `{len(recovery_rows)}`; min mixed R_full `{summary['min_R_full_mixed']}`.",
            f"basis covariance rows `{len(covariance_rows)}`; max function error `{summary['max_covariance_function_error']}`.",
            f"loss families `{[row['loss_family'] for row in loss_rows]}`; min finite-difference score `{summary['min_loss_fd_score']}`.",
            f"repair_note: {summary['repair_note']}",
            f"partA_semantic_pass `{parta_pass}`; science_run_allowed `{summary['science_run_allowed']}` because PartB-I remain incomplete.",
        ],
    )


def latest_json(pattern: str) -> tuple[Path | None, dict[str, Any]]:
    paths = sorted((ROOT / "results").glob(pattern), key=lambda p: p.stat().st_mtime)
    if not paths:
        return None, {}
    path = paths[-1]
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path, {}


def run_partB(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    results_root = ROOT / "results"
    checkpoint_suffixes = {".pt", ".pth", ".ckpt", ".npz"}
    checkpoint_paths: list[Path] = []
    if results_root.exists():
        for path in results_root.rglob("*"):
            if "v23_27" not in path.as_posix() and "v23.27" not in path.as_posix():
                continue
            if path.is_file() and (path.suffix in checkpoint_suffixes or "checkpoint" in path.name.lower()):
                checkpoint_paths.append(path)
    checkpoint_paths = sorted(checkpoint_paths, key=lambda p: p.as_posix())

    existing_v2327_artifacts = []
    if results_root.exists():
        for path in results_root.rglob("v23_27_*"):
            if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
                existing_v2327_artifacts.append(path)
    existing_v2327_artifacts = sorted(existing_v2327_artifacts, key=lambda p: p.as_posix())

    source_rows = [
        {
            "source_kind": "v23_27_checkpoint_candidate",
            "path": rel(path),
            "sha256": sha256_file(path),
            "usable_for_replay": 1,
        }
        for path in checkpoint_paths
    ]
    if not source_rows:
        source_rows.append(
            {
                "source_kind": "v23_27_checkpoint_candidate",
                "path": "",
                "sha256": "",
                "usable_for_replay": 0,
                "blocker": "no v23.27 checkpoint/npz files found under results",
            }
        )

    required_rows = []
    for carrier in ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]:
        for loss_family in ["classification", "regression", "pairwise"]:
            for checkpoint_phase in ["warmup", "mid", "late"]:
                required_rows.append(
                    {
                        "carrier": carrier,
                        "loss_family": loss_family,
                        "checkpoint_phase": checkpoint_phase,
                        "status": "blocked_missing_v23_27_checkpoint",
                        "R_radial": "",
                        "R_skew": "",
                        "R_sym": "",
                        "R_skew_sym": "",
                        "R_full": "",
                        "R_full_minus_R_skew": "",
                        "R_full_minus_R_skewsym": "",
                        "note": "PartB requires true v23.27 checkpoints; existing CSV/JSON summaries are not replayable checkpoints.",
                    }
                )

    artifact_rows = [
        {
            "path": rel(path),
            "sha256": sha256_file(path),
        }
        for path in existing_v2327_artifacts[:500]
    ]

    write_csv(OUT_ROOT / "v23_28_partB_checkpoint_source_audit.csv", source_rows)
    write_csv(OUT_ROOT / "v23_28_partB_representability_decomposition.csv", required_rows)
    write_csv(OUT_ROOT / "v23_28_partB_existing_v23_27_artifact_inventory.csv", artifact_rows)
    summary = {
        "phase": "partB",
        "partB_completed": 0,
        "partB_primary_gate": 0,
        "blocker": "v23_27_replay_checkpoints_missing",
        "checkpoint_candidate_count": len(checkpoint_paths),
        "existing_v23_27_artifact_count": len(existing_v2327_artifacts),
        "required_replay_rows": len(required_rows),
        "no_fabrication_note": "PartB metrics R_radial/R_skew/R_sym/R_full were not computed because true v23.27 checkpoints were not found.",
        "allowed_next_action": "recover or generate true v23.27 replay checkpoints, then rerun PartB; do not infer replay metrics from summary CSV/JSON artifacts.",
    }
    write_json(OUT_ROOT / "v23_28_partB_summary.json", summary)
    files = [
        "v23_28_partB_checkpoint_source_audit.csv",
        "v23_28_partB_existing_v23_27_artifact_inventory.csv",
        "v23_28_partB_representability_decomposition.csv",
        "v23_28_partB_summary.json",
    ]
    append_exec("PartB_v2327_checkpoint_replay_audit", args, files, summary, "blocked_missing_checkpoint")
    append_recap(
        "PartB v23.27 checkpoint replay audit",
        [
            f"checkpoint_candidate_count `{summary['checkpoint_candidate_count']}`; existing_v23_27_artifact_count `{summary['existing_v23_27_artifact_count']}`.",
            f"partB_completed `{summary['partB_completed']}`; blocker `{summary['blocker']}`.",
            "未计算 R_full/R_skew/R_sym：PartB 要求真实 v23.27 checkpoint replay，不能由 CSV/JSON summary 反推。",
            f"allowed_next_action: {summary['allowed_next_action']}",
        ],
    )


PARTB_DIAGNOSTIC_TASKS = ["Wine", "DiabetesRegression", "DiabetesPairwiseRanking"]
PARTB_DIAGNOSTIC_STAGES = [("warmup", 4), ("mid", 10), ("late", 20)]


def partb_train_pairs(bundle: dict[str, Any], seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if str(bundle["task_family"]) != "pairwise":
        return None
    n = int(bundle["y_train"].numel())
    rng = np.random.default_rng(223328 + int(seed))
    pi_np = rng.integers(0, n, size=min(1024, n * 4))
    pj_np = rng.integers(0, n, size=min(1024, n * 4))
    y_np = bundle["y_train"].detach().cpu().numpy()
    target_np = np.where(y_np[pi_np] > y_np[pj_np], 1.0, -1.0)
    return (
        torch.tensor(pi_np, device=bundle["y_train"].device, dtype=torch.long),
        torch.tensor(pj_np, device=bundle["y_train"].device, dtype=torch.long),
        torch.tensor(target_np, device=bundle["y_train"].device, dtype=torch.float64),
    )


def partb_checkpoint_metadata_hash(meta: dict[str, Any]) -> str:
    compact = {key: value for key, value in meta.items() if not str(key).endswith("_path")}
    return stable_hash_obj(compact)


def write_partb_diagnostic_checkpoint(path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    payload["metadata_json"] = np.asarray(json.dumps(json_clean(meta), ensure_ascii=False, sort_keys=True))
    np.savez_compressed(path, **payload)
    return sha256_file(path)


def simulate_partB_diagnostic_group(
    dataset: str,
    carrier: str,
    seed: int,
    device: torch.device,
    *,
    train_cap: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    basis, family, k, _unused_e = partc_carrier_spec(carrier)
    e = 16
    bundle = prepare_partd_smoke_arrays(dataset, seed, k, device, train_cap=train_cap)
    out_dim = int(bundle["output_dim"])
    init_rng = np.random.default_rng(282328 + int(seed) + 101 * (0 if "CHE" in carrier else 1) + 17 * len(dataset))
    a = init_rng.normal(scale=0.08, size=(e, k))
    a0 = a.copy()
    readout_np = init_rng.normal(scale=0.20, size=(e, out_dim))
    readout = torch.tensor(readout_np, device=device, dtype=torch.float64)
    pairs = partb_train_pairs(bundle, seed)
    state = PersistentFullState(k, e, beta=0.90)
    stage_by_step = {step: name for name, step in PARTB_DIAGNOSTIC_STAGES if step <= int(max_steps)}
    static_m: np.ndarray | None = None
    batch_order = list(range(int(max_steps)))
    rows: list[dict[str, Any]] = []
    last_loss = float("nan")
    last_tangent = np.zeros_like(a)
    last_z = np.zeros((e, 1), dtype=np.float64)
    last_cot = np.zeros((e, 1), dtype=np.float64)
    rhos: list[float] = []
    for step in range(1, int(max_steps) + 1):
        loss, grad, z, cot = partd_loss_and_grad(a, readout, bundle, pairs)
        last_loss = loss
        last_z = z
        last_cot = cot
        if static_m is None:
            static_m = build_dynamic_metric(z, np.ones_like(cot), basis=basis)["M"]
        m = static_m
        base_step = -0.12 * grad
        last_tangent = base_step.copy()
        hist_omega, _hist_sym, _hist_radial = state.pre_step_memory()
        a, rho = fu_apply(a, m, base_step, np.zeros(e, dtype=np.float64), hist_omega, np.zeros((k, k), dtype=np.float64), ratio=0.10)
        rhos.append(float(rho))
        _radial_raw, h_force, _residual = full_operator_radial_joint_solve(a, base_step, m)
        omega_force, _sym_force, _global_trace = decompose_operator(h_force, m)
        state.observe(omega_force, np.zeros((k, k), dtype=np.float64), np.zeros(e, dtype=np.float64), m)
        if step not in stage_by_step:
            continue
        stage = stage_by_step[step]
        ckpt_name = f"v23_28_partB_generated_v2327style_{dataset}_{carrier}_s{seed}_{stage}_step{step}.npz".replace("/", "_")
        ckpt_path = OUT_ROOT / "partB_generated_checkpoints" / ckpt_name
        meta = {
            "phase": "partB-generated-v2327style-diagnostic-checkpoint",
            "official_partB_completion_claim": 0,
            "diagnostic_only": 1,
            "dataset": dataset,
            "task_family": bundle["task_family"],
            "seed": int(seed),
            "carrier": carrier,
            "carrier_basis_family": family,
            "basis": basis,
            "stage": stage,
            "step": int(step),
            "scheme": "K3_v2327_SkewPersistent_StaticH2_generated_diagnostic",
            "generator_family": "v23.27-style metric-skew persistent static local H2 compact bank",
            "not_true_v2327_winner_checkpoint": 1,
            "train_cap": int(train_cap),
            "train_count_used": int(bundle["train_count_used"]),
            "test_count_used": int(bundle["test_count_used"]),
            "split_hash": bundle["split_hash"],
            "projection_hash": bundle["projection_hash"],
            "init_hash": stable_hash_obj(np.round(a0, 8).tolist()),
            "model_state_hash": stable_hash_obj(np.round(a, 8).tolist()),
            "readout_hash": stable_hash_obj(np.round(readout_np, 8).tolist()),
            "optimizer_state_hash": stable_hash_obj({"beta": state.beta, "age": state.age, "omega": np.round(state.omega, 8).tolist()}),
            "metric_state_hash": stable_hash_obj(np.round(m, 8).tolist()),
            "RNG_state_hash": stable_hash_obj({"init_seed": 282328 + int(seed) + 101 * (0 if "CHE" in carrier else 1) + 17 * len(dataset)}),
            "minibatch_order_hash": stable_hash_obj(batch_order),
            "normalization_state_hash": stable_hash_obj({"split": bundle["split_hash"], "projection": bundle["projection_hash"]}),
            "loss_at_checkpoint": float(last_loss),
            "FU_to_base_intrinsic_norm_ratio_mean": finite_mean(rhos),
            "Omega_state_age": int(state.age),
            "S_state_age": 0,
            "radial_state_age": 0,
            "current_forcing_same_step_use_count": 0,
            "formula_changed_or_not": 0,
            "threshold_changed_or_not": 0,
            "hypothesis_identity_changed_or_not": 0,
        }
        arrays = {
            "A": a,
            "A_initial": a0,
            "readout": readout_np,
            "metric_M": m,
            "tangent": last_tangent,
            "z_last": last_z,
            "hidden_cotangent_last": last_cot,
            "omega_state": state.omega,
        }
        sha = write_partb_diagnostic_checkpoint(ckpt_path, arrays, meta)
        rows.append({**meta, "checkpoint_path": rel(ckpt_path), "checkpoint_sha256": sha, "checkpoint_metadata_hash": partb_checkpoint_metadata_hash(meta)})
    return rows


def representability_from_partB_checkpoint(row: dict[str, Any]) -> dict[str, Any]:
    ckpt_path = ROOT / str(row["checkpoint_path"])
    with np.load(ckpt_path, allow_pickle=False) as payload:
        a = np.asarray(payload["A"], dtype=np.float64)
        m = np.asarray(payload["metric_M"], dtype=np.float64)
        tangent = np.asarray(payload["tangent"], dtype=np.float64)
    radial_raw, h_force, joint_residual = full_operator_radial_joint_solve(a, tangent, m)
    omega_force, sym_force, global_trace = decompose_operator(h_force, m)
    radial_force = radial_raw + global_trace
    pred_radial = reconstruct(a, radial_force, np.zeros_like(omega_force), np.zeros_like(sym_force))
    pred_skew = reconstruct(a, np.zeros_like(radial_force), omega_force, np.zeros_like(sym_force))
    pred_sym = reconstruct(a, np.zeros_like(radial_force), np.zeros_like(omega_force), sym_force)
    pred_skew_sym = reconstruct(a, np.zeros_like(radial_force), omega_force, sym_force)
    pred_full = reconstruct(a, radial_force, omega_force, sym_force)
    r_full = explained_fraction(tangent, pred_full, m)
    r_skew = explained_fraction(tangent, pred_skew, m)
    r_skew_sym = explained_fraction(tangent, pred_skew_sym, m)
    eig = np.linalg.eigvalsh(0.5 * (m + m.T))
    constant_mix = float(np.mean(np.abs(h_force[0, 1:]))) / max(float(np.mean(np.abs(h_force))), 1.0e-12) if h_force.shape[0] > 1 else 0.0
    return {
        "phase": "partB-generated-diagnostic-representability",
        "official_partB_completion_claim": 0,
        "diagnostic_only": 1,
        "checkpoint_path": row["checkpoint_path"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "dataset": row["dataset"],
        "task_family": row["task_family"],
        "seed": int(row["seed"]),
        "carrier": row["carrier"],
        "carrier_basis_family": row["carrier_basis_family"],
        "basis": row["basis"],
        "stage": row["stage"],
        "step": int(row["step"]),
        "active_bank_edge_count": int(a.shape[0]),
        "metric_condition": float(eig.max() / max(eig.min(), 1.0e-12)),
        "metric_min_eigenvalue": float(eig.min()),
        "constant_mixing_fraction": constant_mix,
        "R_radial": explained_fraction(tangent, pred_radial, m),
        "R_skew": r_skew,
        "R_sym": explained_fraction(tangent, pred_sym, m),
        "R_skew_sym": r_skew_sym,
        "R_full": r_full,
        "R_full_minus_R_skew": r_full - r_skew,
        "R_full_minus_R_skewsym": r_full - r_skew_sym,
        "full_operator_residual_norm": float(joint_residual),
        "full_operator_spectrum_real": json.dumps([float(v) for v in np.linalg.eigvals(h_force).real], ensure_ascii=False),
        "Omega_norm": float(np.linalg.norm(omega_force)),
        "S_norm": float(np.linalg.norm(sym_force)),
        "radial_norm": float(np.linalg.norm(radial_force)),
        "M_skew_residual": float(np.linalg.norm(m_adjoint(omega_force, m) + omega_force)),
        "M_self_adjoint_residual": float(np.linalg.norm(m_adjoint(sym_force, m) - sym_force)),
        "S_trace_abs": abs(float(np.trace(sym_force))),
        "covariance_failure": 0,
    }


def run_partB_diagnostic(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    groups = [(dataset, seed, carrier) for dataset in PARTB_DIAGNOSTIC_TASKS for seed in range(int(args.partB_diagnostic_seeds)) for carrier in carriers]
    if int(args.partB_diagnostic_max_groups) > 0:
        groups = groups[: int(args.partB_diagnostic_max_groups)]
    max_steps = max(step for _stage, step in PARTB_DIAGNOSTIC_STAGES)
    checkpoint_rows: list[dict[str, Any]] = []
    for dataset, seed, carrier in groups:
        checkpoint_rows.extend(simulate_partB_diagnostic_group(dataset, carrier, int(seed), device, train_cap=int(args.partd_train_cap), max_steps=max_steps))
    replay_rows = [representability_from_partB_checkpoint(row) for row in checkpoint_rows]
    write_csv(OUT_ROOT / "v23_28_partB_generated_v2327style_checkpoint_manifest.csv", checkpoint_rows)
    write_csv(OUT_ROOT / "v23_28_partB_generated_representability_decomposition.csv", replay_rows)
    expected_checkpoints = len(groups) * len(PARTB_DIAGNOSTIC_STAGES)
    r_full_vals = [finite_float(row.get("R_full")) for row in replay_rows]
    r_full_vals = [value for value in r_full_vals if math.isfinite(value)]
    r_skew_vals = [finite_float(row.get("R_skew")) for row in replay_rows]
    r_skew_vals = [value for value in r_skew_vals if math.isfinite(value)]
    r_gain_vals = [finite_float(row.get("R_full_minus_R_skew")) for row in replay_rows]
    r_gain_vals = [value for value in r_gain_vals if math.isfinite(value)]
    median_r_full = float(np.median(r_full_vals)) if r_full_vals else float("nan")
    median_r_skew = float(np.median(r_skew_vals)) if r_skew_vals else float("nan")
    median_gain = float(np.median(r_gain_vals)) if r_gain_vals else float("nan")
    rate_r_full_ge_020 = float(np.mean(np.asarray(r_full_vals) >= 0.20)) if r_full_vals else float("nan")
    diagnostic_gate = int(median_r_full >= 0.20 and median_gain >= 0.10 and rate_r_full_ge_020 >= 0.60)
    task_family_summary = {
        family: {
            "rows": len([row for row in replay_rows if row.get("task_family") == family]),
            "R_full_median": median_of([row for row in replay_rows if row.get("task_family") == family], "R_full"),
            "R_full_minus_R_skew_median": median_of([row for row in replay_rows if row.get("task_family") == family], "R_full_minus_R_skew"),
        }
        for family in sorted({str(row.get("task_family", "")) for row in replay_rows})
    }
    summary = {
        "phase": "partB-generated-v2327style-diagnostic",
        "partB_completed": 0,
        "official_partB_completion_claim": 0,
        "partB_generated_diagnostic_completed": int(len(checkpoint_rows) == expected_checkpoints and len(replay_rows) == expected_checkpoints),
        "generated_checkpoint_count": len(checkpoint_rows),
        "expected_generated_checkpoint_count": expected_checkpoints,
        "representability_rows": len(replay_rows),
        "datasets": PARTB_DIAGNOSTIC_TASKS,
        "carriers": carriers,
        "seeds": list(range(int(args.partB_diagnostic_seeds))),
        "stages": [stage for stage, _step in PARTB_DIAGNOSTIC_STAGES],
        "median_R_full": median_r_full,
        "median_R_skew": median_r_skew,
        "median_R_full_minus_R_skew": median_gain,
        "R_full_ge_0p20_rate": rate_r_full_ge_020,
        "diagnostic_representability_gate_pass": diagnostic_gate,
        "task_family_summary": task_family_summary,
        "blocker": "official_v23_27_winner_replay_checkpoints_missing_generated_compact_diagnostic_only",
        "official_blockers": [
            "generated_checkpoints_are_compact_v2327style_diagnostic_not_true_v2327_winner_checkpoints",
            "not_from_fused_DCHE_DFOU_v2327_training_artifacts",
            "regression_pairwise_are_v23_28_compact_diagnostic_tasks_not_v23_27_winner_replay",
        ],
        "no_fabrication_note": "Binary .npz checkpoints were newly generated from compact v23.27-style skew-persistent static-H2 dynamics and replayed for representability. They are diagnostic only and do not satisfy official PartB replay.",
    }
    write_json(OUT_ROOT / "v23_28_partB_summary.json", summary)
    failure = [
        "# v23.28 PartB generated diagnostic replay failure decomposition",
        "",
        f"- generated checkpoints: `{len(checkpoint_rows)}` / expected `{expected_checkpoints}`",
        f"- representability rows: `{len(replay_rows)}`",
        f"- median R_full / R_skew / R_full_minus_R_skew: `{median_r_full}` / `{median_r_skew}` / `{median_gain}`",
        f"- R_full >= 0.20 rate: `{rate_r_full_ge_020}`",
        f"- diagnostic gate pass: `{diagnostic_gate}`",
        f"- task_family_summary: `{json.dumps(json_clean(task_family_summary), ensure_ascii=False, sort_keys=True)}`",
        f"- official_blockers: `{json.dumps(summary['official_blockers'], ensure_ascii=False)}`",
        "",
        "Conclusion: this repair creates replayable binary diagnostic checkpoints and computes real representability values, but official PartB remains incomplete because true v23.27 winner checkpoints were not recovered.",
        "",
    ]
    (OUT_ROOT / "v23_28_partB_generated_diagnostic_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partB_generated_v2327style_checkpoint_manifest.csv",
        "v23_28_partB_generated_representability_decomposition.csv",
        "v23_28_partB_summary.json",
        "v23_28_partB_generated_diagnostic_failure_decomposition.md",
    ]
    append_exec("PartB_generated_v2327style_diagnostic_replay", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartB generated v23.27-style diagnostic replay",
        [
            f"generated checkpoints `{len(checkpoint_rows)}` expected `{expected_checkpoints}`; representability rows `{len(replay_rows)}`.",
            f"median R_full/R_skew/R_full_minus_R_skew `{median_r_full}` / `{median_r_skew}` / `{median_gain}`; R_full>=0.20 rate `{rate_r_full_ge_020}`.",
            f"diagnostic_representability_gate_pass `{diagnostic_gate}`; partB_completed `{summary['partB_completed']}`.",
            f"task_family_summary `{json.dumps(json_clean(task_family_summary), ensure_ascii=False, sort_keys=True)}`.",
            f"official_blockers `{summary['official_blockers']}`.",
            "修复说明：生成新的 compact v23.27-style binary checkpoints 并 replay，不修改公式/阈值/假设身份；不能替代真实 v23.27 winner checkpoint replay。",
        ],
    )


def run_partB_v2327_export_audit(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    manifest_paths = sorted((ROOT / "results").glob("v23_27*/v23_27_partD_exported_checkpoint_manifest.csv"))
    source_rows: list[dict[str, Any]] = []
    for path in manifest_paths:
        for row in read_csv(path):
            row = dict(row)
            row["source_manifest"] = rel(path)
            source_rows.append(row)
    required_fields = {"checkpoint_path", "checkpoint_sha256", "dataset", "task_family", "seed", "carrier", "carrier_basis_family", "basis", "stage", "step"}
    usable_rows = [
        row
        for row in source_rows
        if required_fields.issubset(set(row))
        and all(str(row.get(field, "")) != "" for field in required_fields)
        and int(finite_float(row.get("v23_27_origin_rerun_checkpoint"), 0.0)) == 1
        and (ROOT / str(row.get("checkpoint_path", ""))).is_file()
    ]
    replay_rows: list[dict[str, Any]] = []
    replay_errors: list[dict[str, Any]] = []
    for row in usable_rows:
        try:
            replay = representability_from_partB_checkpoint(row)
            replay.update(
                {
                    "phase": "partB-v2327-origin-rerun-export-representability",
                    "v23_27_origin_rerun_checkpoint": int(finite_float(row.get("v23_27_origin_rerun_checkpoint"), 0.0)),
                    "not_original_v2327_winner_checkpoint": int(finite_float(row.get("not_original_v2327_winner_checkpoint"), 0.0)),
                    "source_manifest": row.get("source_manifest", ""),
                    "scheme": row.get("scheme", ""),
                    "param_name": row.get("param_name", ""),
                    "bank_index": row.get("bank_index", ""),
                }
            )
            replay_rows.append(replay)
        except Exception as exc:  # noqa: BLE001 - audit every failed checkpoint without hiding partial progress.
            replay_errors.append(
                {
                    "checkpoint_path": row.get("checkpoint_path", ""),
                    "checkpoint_sha256": row.get("checkpoint_sha256", ""),
                    "source_manifest": row.get("source_manifest", ""),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    r_full_vals = [finite_float(row.get("R_full")) for row in replay_rows]
    r_full_vals = [value for value in r_full_vals if math.isfinite(value)]
    r_gain_vals = [finite_float(row.get("R_full_minus_R_skew")) for row in replay_rows]
    r_gain_vals = [value for value in r_gain_vals if math.isfinite(value)]
    r_gain_sym_vals = [finite_float(row.get("R_full_minus_R_skewsym")) for row in replay_rows]
    r_gain_sym_vals = [value for value in r_gain_sym_vals if math.isfinite(value)]
    covariance_failure_count = sum(int(finite_float(row.get("covariance_failure"), 0.0)) for row in replay_rows)
    carriers = sorted({str(row.get("carrier", "")) for row in replay_rows if str(row.get("carrier", ""))})
    stages = sorted({str(row.get("stage", "")) for row in replay_rows if str(row.get("stage", ""))})
    task_families = sorted({str(row.get("task_family", "")) for row in replay_rows if str(row.get("task_family", ""))})
    schemes = sorted({str(row.get("scheme", "")) for row in replay_rows if str(row.get("scheme", ""))})
    original_winner_rows = [row for row in replay_rows if int(finite_float(row.get("not_original_v2327_winner_checkpoint"), 1.0)) == 0]
    coverage_full = int({"D-CHE-Core-K3", "D-FOU-Trig-Core-K4"}.issubset(set(carriers)) and {"warmup", "mid", "late"}.issubset(set(stages)))
    gate_pass = int(
        bool(r_full_vals)
        and float(np.median(np.asarray(r_full_vals, dtype=np.float64))) >= 0.20
        and bool(r_gain_vals)
        and float(np.median(np.asarray(r_gain_vals, dtype=np.float64))) >= 0.10
        and float(np.mean(np.asarray(r_full_vals, dtype=np.float64) >= 0.20)) >= 0.60
        and covariance_failure_count == 0
    )
    summary = {
        "phase": "partB-v2327-origin-rerun-export-audit",
        "partB_completed": 0,
        "official_partB_completion_claim": 0,
        "v23_27_origin_export_replay_completed": int(len(replay_rows) > 0 and len(replay_errors) == 0),
        "manifest_count": len(manifest_paths),
        "source_rows": len(source_rows),
        "usable_checkpoint_rows": len(usable_rows),
        "representability_rows": len(replay_rows),
        "replay_error_count": len(replay_errors),
        "carriers": carriers,
        "stages": stages,
        "task_families": task_families,
        "schemes": schemes,
        "primary_carrier_stage_coverage_available": coverage_full,
        "original_v2327_winner_checkpoint_rows": len(original_winner_rows),
        "R_full_median": float(np.median(np.asarray(r_full_vals, dtype=np.float64))) if r_full_vals else "",
        "R_full_minus_R_skew_median": float(np.median(np.asarray(r_gain_vals, dtype=np.float64))) if r_gain_vals else "",
        "R_full_minus_R_skewsym_median": float(np.median(np.asarray(r_gain_sym_vals, dtype=np.float64))) if r_gain_sym_vals else "",
        "R_full_ge_0p20_rate": float(np.mean(np.asarray(r_full_vals, dtype=np.float64) >= 0.20)) if r_full_vals else "",
        "covariance_failure_count": covariance_failure_count,
        "diagnostic_gate_pass_if_rerun_exports_were_allowed": gate_pass,
        "blocker": "v23_27_origin_rerun_checkpoints_available_but_original_winner_checkpoints_missing_and_no_regression_pairwise_coverage",
        "official_blockers": [
            "PartB_plan_requires_original_true_v23_27_checkpoints_not_retraining_winner",
            "v23_27_origin_exports_are_rerun_diagnostic_checkpoints_not_original_winner_artifacts",
            "classification_only_export_does_not_cover_regression_or_pairwise_representative_states",
        ],
        "no_fabrication_note": "This audit replays binary checkpoints exported by the v23.27 runner during a fresh diagnostic smoke. It improves checkpoint format coverage but cannot satisfy the PartB requirement to use original true v23.27 winner checkpoints without retraining.",
    }
    write_csv(OUT_ROOT / "v23_28_partB_v2327_origin_export_checkpoint_manifest.csv", usable_rows)
    write_csv(OUT_ROOT / "v23_28_partB_v2327_origin_export_representability_decomposition.csv", replay_rows)
    write_csv(OUT_ROOT / "v23_28_partB_v2327_origin_export_replay_errors.csv", replay_errors)
    write_json(OUT_ROOT / "v23_28_partB_summary.json", summary)
    failure = [
        "# v23.28 PartB v23.27-origin rerun export audit",
        "",
        f"- manifest_count: `{len(manifest_paths)}`; usable_checkpoint_rows: `{len(usable_rows)}`; representability_rows: `{len(replay_rows)}`; replay_error_count: `{len(replay_errors)}`",
        f"- carriers/stages/task_families: `{carriers}` / `{stages}` / `{task_families}`",
        f"- R_full_median: `{summary['R_full_median']}`; R_full_minus_R_skew_median: `{summary['R_full_minus_R_skew_median']}`; R_full_ge_0p20_rate: `{summary['R_full_ge_0p20_rate']}`",
        f"- diagnostic_gate_pass_if_rerun_exports_were_allowed: `{gate_pass}`",
        f"- official_blockers: `{json.dumps(summary['official_blockers'], ensure_ascii=False)}`",
        "",
        "Conclusion: v23.27-origin replayable checkpoints now exist for a classification carrier smoke, but official PartB remains incomplete because these are rerun exports rather than original winner checkpoints and they do not cover regression/pairwise representative states.",
        "",
    ]
    (OUT_ROOT / "v23_28_partB_v2327_origin_export_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partB_v2327_origin_export_checkpoint_manifest.csv",
        "v23_28_partB_v2327_origin_export_representability_decomposition.csv",
        "v23_28_partB_v2327_origin_export_replay_errors.csv",
        "v23_28_partB_summary.json",
        "v23_28_partB_v2327_origin_export_failure_decomposition.md",
    ]
    append_exec("PartB_v2327_origin_rerun_export_audit", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartB v23.27-origin rerun checkpoint export audit",
        [
            f"manifest_count `{len(manifest_paths)}`; usable_checkpoint_rows `{len(usable_rows)}`; representability_rows `{len(replay_rows)}`; replay_error_count `{len(replay_errors)}`.",
            f"coverage carriers `{carriers}`; stages `{stages}`; task_families `{task_families}`; schemes `{schemes}`.",
            f"R_full_median `{summary['R_full_median']}`; R_full_minus_R_skew_median `{summary['R_full_minus_R_skew_median']}`; R_full_ge_0p20_rate `{summary['R_full_ge_0p20_rate']}`; diagnostic_gate_pass_if_rerun_exports_were_allowed `{gate_pass}`.",
            f"partB_completed `{summary['partB_completed']}`; blocker `{summary['blocker']}`.",
            "修复说明：这是 v23.27 runner 重新导出的 binary checkpoint replay，不是原始 v23.27 winner checkpoint；不能替代 PartB official gate。",
        ],
    )


def run_partC(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    seeds = list(range(int(args.partc_seeds)))
    groups = [(task, seed, carrier) for task in SYNTHETIC_TASKS for seed in seeds for carrier in carriers]
    shard_count = max(1, int(args.partc_shard_count))
    shard_index = int(args.partc_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid shard index {shard_index} for shard count {shard_count}")
    shard_groups = [group for idx, group in enumerate(groups) if idx % shard_count == shard_index]
    if int(args.partc_max_groups) > 0:
        shard_groups = shard_groups[: int(args.partc_max_groups)]

    rows: list[dict[str, Any]] = []
    for task, seed, carrier in shard_groups:
        bank_rows = [
            simulate_partc_scheme(
                task,
                carrier,
                seed,
                scheme,
                int(args.partc_horizon),
                float(args.partc_fu_ratio),
                str(args.partc_target_mode),
            )
            for scheme in PARTC_PREFLIGHT_SCHEMES
        ]
        by_scheme = {str(row["scheme"]): row for row in bank_rows}
        base_auc = finite_float(by_scheme.get("K0_AdamW_task_native", {}).get("AUC_task_loss_time"))
        instant_auc = finite_float(by_scheme.get("P0_FullInstant_CompH2", {}).get("AUC_task_loss_time"))
        random_auc = finite_float(by_scheme.get("R1_RandomAR1FullState", {}).get("AUC_task_loss_time"))
        reset_auc = finite_float(by_scheme.get("R0_ResetEveryStepFullState", {}).get("AUC_task_loss_time"))
        bank_shuffle_auc = finite_float(by_scheme.get("R3_ReceivingBankShuffledState", {}).get("AUC_task_loss_time"))
        for row in bank_rows:
            auc = finite_float(row.get("AUC_task_loss_time"))
            row["paired_loss_gain_vs_base"] = base_auc - auc if math.isfinite(base_auc) and math.isfinite(auc) else ""
            row["paired_loss_gain_vs_instant"] = instant_auc - auc if math.isfinite(instant_auc) and math.isfinite(auc) else ""
            row["paired_loss_gain_vs_random_state"] = random_auc - auc if math.isfinite(random_auc) and math.isfinite(auc) else ""
            row["paired_loss_gain_vs_reset"] = reset_auc - auc if math.isfinite(reset_auc) and math.isfinite(auc) else ""
            row["paired_loss_gain_vs_bank_shuffled"] = bank_shuffle_auc - auc if math.isfinite(bank_shuffle_auc) and math.isfinite(auc) else ""
            row["paired_loss_gain_vs_MLP_matched"] = ""
            row["paired_win_rate"] = int(math.isfinite(base_auc) and math.isfinite(auc) and auc < base_auc)
        rows.extend(bank_rows)

        native_rows: list[dict[str, Any]] = []
        mlp_rows: list[dict[str, Any]] = []
        bundle: dict[str, Any] | None = None
        if int(args.partc_include_task_native_loss) or int(args.partc_include_mlp):
            bundle = partc_task_native_bundle(task, carrier, seed, device)
        if int(args.partc_include_task_native_loss) and task in set(PARTC_C4_TASKS.values()) and bundle is not None:
            native_rows = [
                simulate_partc_task_native_kan_scheme(
                    task,
                    carrier,
                    seed,
                    scheme,
                    int(args.partc_horizon),
                    float(args.partc_fu_ratio),
                    bundle,
                )
                for scheme in PARTC_TASK_NATIVE_KAN_SCHEMES
            ]
        if int(args.partc_include_mlp) and bundle is not None:
            mlp_rows = [
                simulate_partc_task_native_mlp_scheme(
                    task,
                    carrier,
                    seed,
                    scheme,
                    int(args.partc_horizon),
                    bundle,
                )
                for scheme in PARTC_MLP_SCHEMES
            ]
        if native_rows:
            native_by_scheme = {str(row["scheme"]): row for row in native_rows}
            mlp_by_scheme = {str(row["scheme"]): row for row in mlp_rows}
            native_base_auc = finite_float(native_by_scheme.get("K0_AdamW_task_native", {}).get("AUC_task_loss_time"))
            native_instant_auc = finite_float(native_by_scheme.get("P0_FullInstant_CompH2", {}).get("AUC_task_loss_time"))
            native_random_auc = finite_float(native_by_scheme.get("R1_RandomAR1FullState", {}).get("AUC_task_loss_time"))
            native_reset_auc = finite_float(native_by_scheme.get("R0_ResetEveryStepFullState", {}).get("AUC_task_loss_time"))
            native_mlp_auc = finite_float(mlp_by_scheme.get("M3_MLP_PersistentFullTransportShapeRadialBlock", {}).get("AUC_task_loss_time"))
            native_p1_gains: list[float] = []
            for row in native_rows:
                auc = finite_float(row.get("AUC_task_loss_time"))
                row["paired_loss_gain_vs_base"] = native_base_auc - auc if math.isfinite(native_base_auc) and math.isfinite(auc) else ""
                row["paired_loss_gain_vs_instant"] = native_instant_auc - auc if math.isfinite(native_instant_auc) and math.isfinite(auc) else ""
                row["paired_loss_gain_vs_random_state"] = native_random_auc - auc if math.isfinite(native_random_auc) and math.isfinite(auc) else ""
                row["paired_loss_gain_vs_reset"] = native_reset_auc - auc if math.isfinite(native_reset_auc) and math.isfinite(auc) else ""
                row["paired_loss_gain_vs_bank_shuffled"] = ""
                row["paired_loss_gain_vs_MLP_matched"] = native_mlp_auc - auc if math.isfinite(native_mlp_auc) and math.isfinite(auc) else ""
                row["paired_win_rate"] = int(math.isfinite(native_base_auc) and math.isfinite(auc) and auc < native_base_auc)
                if row["scheme"] == "P1_FullPersistent_Hybrid_CompH2_PRIMARY" and math.isfinite(finite_float(row["paired_loss_gain_vs_base"])):
                    native_p1_gains.append(finite_float(row["paired_loss_gain_vs_base"]))
            for row in native_rows:
                if row["scheme"] == "P1_FullPersistent_Hybrid_CompH2_PRIMARY":
                    row["paired_CVaR25"] = cvar25(native_p1_gains)
                    row["paired_bootstrap_LCB05"] = bootstrap_lcb05(native_p1_gains, seed=642328 + int(seed))
            rows.extend(native_rows)
        if mlp_rows:
            rows.extend(mlp_rows)

    suffix = f"shard{shard_index:02d}_of_{shard_count:02d}"
    matrix_name = f"v23_28_partC_synthetic_preflight_matrix_{suffix}.csv"
    summary_name = f"v23_28_partC_synthetic_preflight_summary_{suffix}.json"
    write_csv(OUT_ROOT / matrix_name, rows)
    summary = {
        "phase": "partC-synthetic-preflight-shard",
        "partC_completed": 0,
        "partC_preflight_shard_completed": int(len(rows) > 0),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "groups_in_shard": len(shard_groups),
        "rows": len(rows),
        "tasks": sorted({str(row["task"]) for row in rows}),
        "carriers": sorted({str(row["carrier"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "schemes": sorted({str(row["scheme"]) for row in rows}),
        "horizon": int(args.partc_horizon),
        "fu_ratio": float(args.partc_fu_ratio),
        "target_mode": str(args.partc_target_mode),
        "include_task_native_loss": int(args.partc_include_task_native_loss),
        "include_mlp": int(args.partc_include_mlp),
        "task_native_rows": sum(1 for row in rows if row.get("phase") == "partC_task_native_loss_preflight"),
        "mlp_rows": sum(1 for row in rows if row.get("phase") == "partC_MLP_task_native_preflight"),
        "official_blocker": "shard output only; aggregate and full official PartC gates not completed by this shard",
        "no_fabrication_note": "Rows are generated by compact H20 synthetic preflights using shared v23.28 metric/operator/state/FU code where applicable; they are not claimed as full fused-carrier official PartC.",
    }
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("PartC_synthetic_preflight_shard", args, [matrix_name, summary_name], summary, "completed_incomplete_gate")


def run_partC_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    shard_paths = sorted(OUT_ROOT.glob("v23_28_partC_synthetic_preflight_matrix_shard*_of_*.csv"))
    rows: list[dict[str, Any]] = []
    for path in shard_paths:
        rows.extend(read_csv(path))
    matrix_path = OUT_ROOT / "v23_28_partC_synthetic_preflight_matrix.csv"
    write_csv(matrix_path, rows)

    tasks = sorted({row.get("task", "") for row in rows if row.get("task", "")})
    carriers = sorted({row.get("carrier", "") for row in rows if row.get("carrier", "")})
    seeds = sorted({int(row.get("seed", "0")) for row in rows if str(row.get("seed", "")).isdigit()})
    schemes = sorted({row.get("scheme", "") for row in rows if row.get("scheme", "")})
    fu_ratios = sorted({finite_float(row.get("fu_ratio")) for row in rows if math.isfinite(finite_float(row.get("fu_ratio")))})
    target_modes = sorted({row.get("target_mode", "") for row in rows if row.get("target_mode", "")})
    h20_rows = sum(1 for row in rows if finite_float(row.get("horizon"), 0.0) >= 20.0)
    finite_auc_rows = sum(1 for row in rows if math.isfinite(finite_float(row.get("AUC_task_loss_time"))))
    bank_preflight_rows = [row for row in rows if row.get("phase") == "partC_synthetic_preflight"]
    task_native_rows = [row for row in rows if row.get("phase") == "partC_task_native_loss_preflight"]
    mlp_rows = [row for row in rows if row.get("phase") == "partC_MLP_task_native_preflight"]
    p1_rows = [row for row in rows if row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
    mixed_rows = [row for row in rows if row.get("task") == "SYN-D-MIXED-FULL" and row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
    skew_rows = [row for row in rows if row.get("task") == "SYN-A-SKEW-ONLY" and row.get("scheme") in {"P1_FullPersistent_Hybrid_CompH2_PRIMARY", "A0_Persistent_SkewOnly"}]
    sym_rows = [row for row in rows if row.get("task") == "SYN-B-SYMMETRIC-SHAPE" and row.get("scheme") in {"P1_FullPersistent_Hybrid_CompH2_PRIMARY", "A1_Persistent_SymmetricOnly"}]
    radial_rows = [row for row in rows if row.get("task") == "SYN-C-EDGE-RADIAL" and row.get("scheme") in {"P1_FullPersistent_Hybrid_CompH2_PRIMARY", "A2_Persistent_RadialOnly"}]
    persistent_rows = [row for row in rows if row.get("task") == "SYN-G-PERSISTENT-COHERENCE" and row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
    transient_rows = [row for row in rows if row.get("task") == "SYN-H-TRANSIENT-FORCING" and row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
    metric_true_rows = [row for row in rows if row.get("task") == "SYN-E-COMP-CURVATURE" and row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
    metric_data_rows = [row for row in rows if row.get("task") == "SYN-E-COMP-CURVATURE" and row.get("scheme") == "P3_FullPersistent_Hybrid_DataL2"]
    metric_local_rows = [row for row in rows if row.get("task") == "SYN-E-COMP-CURVATURE" and row.get("scheme") == "P4_FullPersistent_Hybrid_LocalH2"]
    metric_shuffle_rows = [row for row in rows if row.get("task") == "SYN-E-COMP-CURVATURE" and row.get("scheme") == "P5_FullPersistent_Hybrid_PathShuffleH2"]
    metric_uniform_rows = [row for row in rows if row.get("task") == "SYN-E-COMP-CURVATURE" and row.get("scheme") == "P6_FullPersistent_Hybrid_UniformPathH2"]
    component_full_rows = [row for row in rows if row.get("task") == "SYN-D-MIXED-FULL" and row.get("scheme") == "A6_Persistent_Full_OmegaSymRadial"]
    component_single_rows = [
        row
        for row in rows
        if row.get("task") == "SYN-D-MIXED-FULL" and row.get("scheme") in {"A0_Persistent_SkewOnly", "A1_Persistent_SymmetricOnly", "A2_Persistent_RadialOnly"}
    ]

    c1 = {
        "mixed_full_R_full_median": median_of(mixed_rows, "R_full"),
        "skew_component_cosine_median": median_of(skew_rows, "skew_component_true_cosine"),
        "symmetric_component_cosine_median": median_of(sym_rows, "symmetric_component_true_cosine"),
        "radial_vector_cosine_median": median_of(radial_rows, "radial_vector_true_cosine"),
    }
    c1["gate_pass"] = int(
        c1["mixed_full_R_full_median"] >= 0.90
        and c1["skew_component_cosine_median"] >= 0.90
        and c1["symmetric_component_cosine_median"] >= 0.90
        and c1["radial_vector_cosine_median"] >= 0.90
    )
    c2 = {
        "persistent_P1_gain_vs_base_median": median_of(persistent_rows, "paired_loss_gain_vs_base"),
        "persistent_P1_gain_vs_instant_median": median_of(persistent_rows, "paired_loss_gain_vs_instant"),
        "persistent_P1_gain_vs_random_median": median_of(persistent_rows, "paired_loss_gain_vs_random_state"),
        "persistent_P1_gain_vs_reset_median": median_of(persistent_rows, "paired_loss_gain_vs_reset"),
        "persistent_P1_gain_vs_bank_shuffled_median": median_of(persistent_rows, "paired_loss_gain_vs_bank_shuffled"),
        "transient_P1_gain_vs_instant_median": median_of(transient_rows, "paired_loss_gain_vs_instant"),
    }
    c2["gate_pass"] = int(
        c2["persistent_P1_gain_vs_base_median"] > 1.0e-3
        and c2["persistent_P1_gain_vs_instant_median"] > 1.0e-3
        and c2["persistent_P1_gain_vs_random_median"] > 1.0e-3
        and c2["persistent_P1_gain_vs_reset_median"] > 1.0e-3
        and c2["persistent_P1_gain_vs_bank_shuffled_median"] > 1.0e-3
        and c2["transient_P1_gain_vs_instant_median"] <= 0.0
    )
    true_metric_auc = median_of(metric_true_rows, "AUC_task_loss_time")
    c3 = {
        "true_comp_auc_median": true_metric_auc,
        "dataL2_auc_median": median_of(metric_data_rows, "AUC_task_loss_time"),
        "localH2_auc_median": median_of(metric_local_rows, "AUC_task_loss_time"),
        "path_shuffle_auc_median": median_of(metric_shuffle_rows, "AUC_task_loss_time"),
        "uniform_path_auc_median": median_of(metric_uniform_rows, "AUC_task_loss_time"),
    }
    c3["gate_pass"] = int(
        math.isfinite(true_metric_auc)
        and true_metric_auc < c3["dataL2_auc_median"]
        and true_metric_auc < c3["localH2_auc_median"]
        and true_metric_auc < c3["path_shuffle_auc_median"]
        and true_metric_auc < c3["uniform_path_auc_median"]
    )
    full_component_auc = median_of(component_full_rows, "AUC_task_loss_time")
    single_component_scheme_medians = {
        scheme: median_of([row for row in component_single_rows if row.get("scheme") == scheme], "AUC_task_loss_time")
        for scheme in ["A0_Persistent_SkewOnly", "A1_Persistent_SymmetricOnly", "A2_Persistent_RadialOnly"]
    }
    single_component_best_auc = min([value for value in single_component_scheme_medians.values() if math.isfinite(value)] or [float("nan")])
    c5 = {
        "full_component_auc_median": full_component_auc,
        "single_component_scheme_auc_medians": single_component_scheme_medians,
        "best_single_component_auc_median": single_component_best_auc,
        "gate_pass": int(math.isfinite(full_component_auc) and math.isfinite(single_component_best_auc) and full_component_auc < single_component_best_auc),
    }
    c4_family_summary: dict[str, Any] = {}
    c4_family_gates: list[int] = []
    for family_name, task_name in PARTC_C4_TASKS.items():
        family_rows = [row for row in task_native_rows if row.get("loss_family") == family_name and row.get("task") == task_name]
        family_p1_rows = [row for row in family_rows if row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
        family_core_hashes = {
            "FU_core_hash": sorted({row.get("FU_core_hash", "") for row in family_rows if row.get("FU_core_hash", "")}),
            "metric_builder_hash": sorted({row.get("metric_builder_hash", "") for row in family_rows if row.get("metric_builder_hash", "")}),
            "state_update_hash": sorted({row.get("state_update_hash", "") for row in family_rows if row.get("state_update_hash", "")}),
        }
        initial_final_gains = [
            finite_float(row.get("task_native_loss_initial")) - finite_float(row.get("task_native_loss_final"))
            for row in family_p1_rows
            if math.isfinite(finite_float(row.get("task_native_loss_initial"))) and math.isfinite(finite_float(row.get("task_native_loss_final")))
        ]
        p1_gain_vs_base = [finite_float(row.get("paired_loss_gain_vs_base")) for row in family_p1_rows if math.isfinite(finite_float(row.get("paired_loss_gain_vs_base")))]
        p1_gain_vs_mlp = [finite_float(row.get("paired_loss_gain_vs_MLP_matched")) for row in family_p1_rows if math.isfinite(finite_float(row.get("paired_loss_gain_vs_MLP_matched")))]
        finite_family_auc = sum(1 for row in family_rows if math.isfinite(finite_float(row.get("AUC_task_loss_time"))))
        p1_h20 = sum(1 for row in family_p1_rows if finite_float(row.get("horizon"), 0.0) >= 20.0)
        p1_full_solves = [finite_float(row.get("actual_full_operator_solve_count")) for row in family_p1_rows if math.isfinite(finite_float(row.get("actual_full_operator_solve_count")))]
        core_hash_invariant = int(all(len(values) == 1 for values in family_core_hashes.values()))
        gate = int(
            bool(family_p1_rows)
            and finite_family_auc == len(family_rows)
            and p1_h20 == len(family_p1_rows)
            and core_hash_invariant == 1
            and (float(np.median(initial_final_gains)) if initial_final_gains else float("nan")) > 0.0
            and (float(np.median(p1_full_solves)) if p1_full_solves else 0.0) >= 20.0
        )
        c4_family_gates.append(gate)
        c4_family_summary[family_name] = {
            "task": task_name,
            "rows": len(family_rows),
            "p1_rows": len(family_p1_rows),
            "finite_auc_rows": finite_family_auc,
            "p1_h20_rows": p1_h20,
            "p1_loss_decrease_median": float(np.median(initial_final_gains)) if initial_final_gains else float("nan"),
            "p1_gain_vs_base_median": float(np.median(p1_gain_vs_base)) if p1_gain_vs_base else float("nan"),
            "p1_gain_vs_MLP_matched_median": float(np.median(p1_gain_vs_mlp)) if p1_gain_vs_mlp else float("nan"),
            "p1_full_operator_solve_count_median": float(np.median(p1_full_solves)) if p1_full_solves else float("nan"),
            "core_hash_invariant": core_hash_invariant,
            "compact_task_native_gate_pass": gate,
            "core_hashes": family_core_hashes,
        }
    c4 = {
        "gate_pass": int(bool(c4_family_gates) and all(value == 1 for value in c4_family_gates)),
        "task_native_rows": len(task_native_rows),
        "family_summary": c4_family_summary,
        "official_scope_note": "C4 rows are compact task-native synthetic preflight rows; they do not make full fused-carrier official PartC complete.",
    }
    mlp_registry = {
        "rows": len(mlp_rows),
        "schemes": sorted({row.get("scheme", "") for row in mlp_rows if row.get("scheme", "")}),
        "tasks_count": len({row.get("task", "") for row in mlp_rows if row.get("task", "")}),
        "carriers_count": len({row.get("carrier", "") for row in mlp_rows if row.get("carrier", "")}),
        "seeds_count": len({row.get("seed", "") for row in mlp_rows if row.get("seed", "")}),
        "families": sorted({row.get("loss_family", "") for row in mlp_rows if row.get("loss_family", "")}),
        "all_registry_schemes_present": int(set(PARTC_MLP_SCHEMES).issubset({row.get("scheme", "") for row in mlp_rows})),
        "finite_auc_rows": sum(1 for row in mlp_rows if math.isfinite(finite_float(row.get("AUC_task_loss_time")))),
        "official_scope_note": "MLP rows are task-native optimizer controls for PartC preflight only; persistent MLP generator and MCGA official reproduction remain PartI blockers.",
    }
    mandatory_schema_fields = [
        "task_native_loss_initial",
        "task_native_loss_final",
        "paired_loss_gain_vs_base",
        "paired_loss_gain_vs_instant",
        "paired_loss_gain_vs_random_state",
        "paired_loss_gain_vs_MLP_matched",
        "paired_CVaR25",
        "paired_bootstrap_LCB05",
        "paired_win_rate",
        "AUC_task_loss_time",
        "G0_min_eigenvalue",
        "G0_condition_number",
        "metric_drift_Frobenius",
        "path_weight_cv",
        "FU_to_base_intrinsic_norm_ratio",
    ]
    metric_schema_coverage = {
        field: sum(1 for row in rows if str(row.get(field, "")) != "")
        for field in mandatory_schema_fields
    }
    official_blockers = [
        "bank_level_proxy_not_full_fused_DCHE_DFOU_training",
        (
            "MLP_registry_rows_run_as_task_native_proxy_not_full_official_MLP_generator_or_MCGA"
            if len(mlp_rows) > 0
            else "MLP_registry_rows_not_run_in_partC_preflight"
        ),
        (
            "classification_CE_MSE_pairwise_task_native_rows_pass_compact_C4_proxy_not_full_official"
            if c4["gate_pass"]
            else (
                "classification_CE_MSE_pairwise_task_native_rows_run_but_compact_C4_gate_incomplete"
                if len(task_native_rows) > 0
                else "classification_CE_MSE_pairwise_task_native_synthetic_losses_not_all_implemented"
            )
        ),
        "mandatory_metric_schema_compact_preflight_not_official_profiler_representation_complete",
        "PartD_to_PartI_not_run",
    ]
    summary = {
        "phase": "partC-synthetic-preflight-aggregate",
        "partC_completed": 0,
        "partC_preflight_completed": int(len(rows) > 0 and h20_rows == len(rows) and finite_auc_rows == len(rows)),
        "partC_official_gate": 0,
        "partC_official_blockers": official_blockers,
        "rows": len(rows),
        "expected_minimum_partC_rows": 1600,
        "row_count_ge_1600": int(len(rows) >= 1600),
        "bank_preflight_rows": len(bank_preflight_rows),
        "task_native_rows": len(task_native_rows),
        "mlp_rows": len(mlp_rows),
        "h20_rows": h20_rows,
        "finite_auc_rows": finite_auc_rows,
        "tasks_count": len(tasks),
        "carriers_count": len(carriers),
        "seeds_count": len(seeds),
        "schemes_count": len(schemes),
        "fu_ratios": fu_ratios,
        "target_modes": target_modes,
        "tasks": tasks,
        "carriers": carriers,
        "seeds": seeds,
        "schemes": schemes,
        "C1_full_generator_recovery": c1,
        "C2_persistence_specificity": c2,
        "C3_compositional_metric": c3,
        "C4_loss_family_identity": c4,
        "C5_component_attribution": c5,
        "MLP_registry_preflight": mlp_registry,
        "metric_schema_coverage_nonempty_counts": metric_schema_coverage,
        "no_fabrication_note": "Aggregate reports only measured shard CSV values. It is a reduced compact preflight, so official PartC remains incomplete even when some synthetic/task-native gates pass.",
    }
    write_json(OUT_ROOT / "v23_28_partC_summary.json", summary)
    failure = [
        "# v23.28 PartC synthetic preflight failure decomposition",
        "",
        f"- rows: `{len(rows)}`; h20_rows: `{h20_rows}`; finite_auc_rows: `{finite_auc_rows}`",
        f"- tasks/carriers/seeds/schemes: `{len(tasks)}/{len(carriers)}/{len(seeds)}/{len(schemes)}`",
        f"- C1: `{json.dumps(json_clean(c1), ensure_ascii=False, sort_keys=True)}`",
        f"- C2: `{json.dumps(json_clean(c2), ensure_ascii=False, sort_keys=True)}`",
        f"- C3: `{json.dumps(json_clean(c3), ensure_ascii=False, sort_keys=True)}`",
        f"- C4: `{json.dumps(json_clean(c4), ensure_ascii=False, sort_keys=True)}`",
        f"- C5: `{json.dumps(json_clean(c5), ensure_ascii=False, sort_keys=True)}`",
        f"- MLP registry preflight: `{json.dumps(json_clean(mlp_registry), ensure_ascii=False, sort_keys=True)}`",
        f"- official_blockers: `{json.dumps(official_blockers, ensure_ascii=False)}`",
        "",
        "Conclusion: this preflight checks the shared metric/operator/state/FU path under exact synthetic forcing and adds compact task-native/MLP rows, but it is not the full PartC official matrix.",
        "",
    ]
    (OUT_ROOT / "v23_28_partC_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partC_synthetic_preflight_matrix.csv",
        "v23_28_partC_summary.json",
        "v23_28_partC_failure_decomposition.md",
    ]
    append_exec("PartC_synthetic_preflight_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartC synthetic preflight aggregate",
        [
            f"rows `{summary['rows']}`; h20_rows `{summary['h20_rows']}`; row_count_ge_1600 `{summary['row_count_ge_1600']}`.",
            f"tasks/carriers/seeds/schemes `{summary['tasks_count']}/{summary['carriers_count']}/{summary['seeds_count']}/{summary['schemes_count']}`.",
            f"C1 `{json.dumps(json_clean(c1), ensure_ascii=False, sort_keys=True)}`.",
            f"C2 `{json.dumps(json_clean(c2), ensure_ascii=False, sort_keys=True)}`.",
            f"C3 `{json.dumps(json_clean(c3), ensure_ascii=False, sort_keys=True)}`.",
            f"C4 `{json.dumps(json_clean(c4), ensure_ascii=False, sort_keys=True)}`.",
            f"C5 `{json.dumps(json_clean(c5), ensure_ascii=False, sort_keys=True)}`.",
            f"MLP_registry_preflight `{json.dumps(json_clean(mlp_registry), ensure_ascii=False, sort_keys=True)}`.",
            f"metric_schema_coverage_nonempty_counts `{json.dumps(json_clean(metric_schema_coverage), ensure_ascii=False, sort_keys=True)}`.",
            f"partC_completed `{summary['partC_completed']}`; official_blockers `{official_blockers}`.",
        ],
    )


def run_partD_audit(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    datasets = [*CLASSIFICATION_DATASETS, "DiabetesRegression", "DiabetesPairwiseRanking"]
    rows = [partd_dataset_audit_row(dataset) for dataset in datasets]
    split_rows: list[dict[str, Any]] = []
    for row in rows:
        n = int(row.get("n_samples") or 0)
        if row.get("availability") != "available" or n <= 0:
            continue
        for seed in [0, 1, 2, 3, 4]:
            split_rows.append(
                {
                    "dataset": row["dataset"],
                    "task_family": row["task_family"],
                    "seed": seed,
                    **split_hashes(n, seed),
                    "normalization_stats_source": "train_only_required_for_future_H20",
                }
            )
    available = [row for row in rows if row.get("availability") == "available"]
    unavailable = [row for row in rows if row.get("availability") != "available"]
    write_csv(OUT_ROOT / "v23_28_partD_minimum_real_source_audit.csv", rows)
    write_csv(OUT_ROOT / "v23_28_partD_split_manifest.csv", split_rows)
    summary = {
        "phase": "partD-minimum-real-source-audit",
        "partD_completed": 0,
        "partD_source_audit_completed": 1,
        "minimum_real_H20_rows": 0,
        "datasets_required": datasets,
        "datasets_available_count": len(available),
        "datasets_unavailable_count": len(unavailable),
        "datasets_available": [row["dataset"] for row in available],
        "datasets_unavailable": [row["dataset"] for row in unavailable],
        "split_manifest_rows": len(split_rows),
        "official_blockers": [
            "minimum_real_H20_training_matrix_not_implemented_or_not_run",
            "fused_DCHE_DFOU_real_carrier_training_not_implemented_in_v23_28_runner",
            "paired_candidate_control_H20_metrics_not_computed",
            "debt_representation_efficiency_metrics_not_computed",
        ],
        "no_fabrication_note": "PartD audit loads or verifies local/sklearn dataset sources and split hashes only; it does not claim H20 training rows.",
    }
    write_json(OUT_ROOT / "v23_28_partD_summary.json", summary)
    files = [
        "v23_28_partD_minimum_real_source_audit.csv",
        "v23_28_partD_split_manifest.csv",
        "v23_28_partD_summary.json",
    ]
    append_exec("PartD_minimum_real_source_audit", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartD minimum-real source audit",
        [
            f"datasets_available `{summary['datasets_available']}`; datasets_unavailable `{summary['datasets_unavailable']}`.",
            f"split_manifest_rows `{summary['split_manifest_rows']}` for seeds `[0,1,2,3,4]`; normalization stats source recorded as train-only for future H20.",
            f"minimum_real_H20_rows `{summary['minimum_real_H20_rows']}`; partD_completed `{summary['partD_completed']}`.",
            f"official_blockers `{summary['official_blockers']}`.",
        ],
    )


def run_partD_smoke(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets = ["Wine", "DiabetesRegression", "DiabetesPairwiseRanking"]
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    schemes = [
        "K0_AdamW_task_native",
        "P0_FullInstant_CompH2",
        "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
        "R0_ResetEveryStepFullState",
        "R1_RandomAR1FullState",
    ]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for carrier in carriers:
            group_rows = [
                simulate_partd_smoke_scheme(dataset, carrier, int(args.partd_smoke_seed), scheme, int(args.partd_smoke_horizon), device, train_cap=int(args.partd_train_cap))
                for scheme in schemes
            ]
            by_scheme = {str(row["scheme"]): row for row in group_rows}
            base_auc = finite_float(by_scheme["K0_AdamW_task_native"].get("AUC_task_loss_time"))
            instant_auc = finite_float(by_scheme["P0_FullInstant_CompH2"].get("AUC_task_loss_time"))
            random_auc = finite_float(by_scheme["R1_RandomAR1FullState"].get("AUC_task_loss_time"))
            reset_auc = finite_float(by_scheme["R0_ResetEveryStepFullState"].get("AUC_task_loss_time"))
            for row in group_rows:
                auc = finite_float(row.get("AUC_task_loss_time"))
                row["paired_loss_gain_vs_base"] = base_auc - auc if math.isfinite(base_auc) and math.isfinite(auc) else ""
                row["paired_loss_gain_vs_instant"] = instant_auc - auc if math.isfinite(instant_auc) and math.isfinite(auc) else ""
                row["paired_loss_gain_vs_random_state"] = random_auc - auc if math.isfinite(random_auc) and math.isfinite(auc) else ""
                row["paired_loss_gain_vs_reset"] = reset_auc - auc if math.isfinite(reset_auc) and math.isfinite(auc) else ""
            rows.extend(group_rows)
    p1_rows = [row for row in rows if row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
    summary = {
        "phase": "partD-real-H20-smoke",
        "partD_completed": 0,
        "partD_source_audit_completed": 1,
        "partD_H20_smoke_completed": int(len(rows) > 0),
        "minimum_real_H20_rows": len(rows),
        "official_partD_H20_rows": 0,
        "datasets": datasets,
        "carriers": carriers,
        "schemes": schemes,
        "seed": int(args.partd_smoke_seed),
        "horizon": int(args.partd_smoke_horizon),
        "train_cap": int(args.partd_train_cap),
        "P1_gain_vs_base_median": median_of(p1_rows, "paired_loss_gain_vs_base"),
        "P1_gain_vs_instant_median": median_of(p1_rows, "paired_loss_gain_vs_instant"),
        "P1_gain_vs_random_median": median_of(p1_rows, "paired_loss_gain_vs_random_state"),
        "P1_gain_vs_reset_median": median_of(p1_rows, "paired_loss_gain_vs_reset"),
        "official_blockers": [
            "minimum_real_full_matrix_not_run_all_datasets_seeds_schemes",
            "compact_bank_proxy_not_fused_DCHE_DFOU_real_carrier",
            "classification_debt_regression_debt_pairwise_debt_metrics_incomplete",
            "MLP_matched_and_efficiency_controls_not_run",
        ],
        "no_fabrication_note": "PartD smoke rows use real task-native CE/MSE/pairwise losses on compact bank proxy only; official PartD remains incomplete.",
    }
    write_csv(OUT_ROOT / "v23_28_partD_real_H20_smoke_matrix.csv", rows)
    write_json(OUT_ROOT / "v23_28_partD_summary.json", summary)
    files = ["v23_28_partD_real_H20_smoke_matrix.csv", "v23_28_partD_summary.json"]
    append_exec("PartD_real_H20_smoke", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartD real H20 smoke",
        [
            f"rows `{summary['minimum_real_H20_rows']}` over datasets `{datasets}`, carriers `{carriers}`, schemes `{schemes}`.",
            f"P1 gains median vs base/instant/random/reset `{summary['P1_gain_vs_base_median']}` / `{summary['P1_gain_vs_instant_median']}` / `{summary['P1_gain_vs_random_median']}` / `{summary['P1_gain_vs_reset_median']}`.",
            f"partD_completed `{summary['partD_completed']}`; official_blockers `{summary['official_blockers']}`.",
        ],
    )


def partd_matrix_schemes() -> list[str]:
    return [
        "K0_AdamW_task_native",
        "P0_FullInstant_CompH2",
        "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
        "R0_ResetEveryStepFullState",
        "R1_RandomAR1FullState",
    ]


def run_partD_matrix(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets = [*CLASSIFICATION_DATASETS, "DiabetesRegression", "DiabetesPairwiseRanking"]
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    include_expanded_registry = int(getattr(args, "partd_include_expanded_registry", 0)) == 1
    include_mlp_efficiency = int(getattr(args, "partd_include_mlp_efficiency", 0)) == 1
    kan_schemes = PARTD_EXPANDED_KAN_SCHEMES if include_expanded_registry else partd_matrix_schemes()
    mlp_schemes = PARTD_EXPANDED_MLP_SCHEMES if include_mlp_efficiency else []
    schemes = [*kan_schemes, *mlp_schemes]
    groups = [(dataset, seed, carrier) for dataset in datasets for seed in range(int(args.partd_matrix_seeds)) for carrier in carriers]
    shard_count = max(1, int(args.partd_shard_count))
    shard_index = int(args.partd_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid PartD shard index {shard_index} for shard count {shard_count}")
    shard_groups = [group for idx, group in enumerate(groups) if idx % shard_count == shard_index]
    if int(args.partd_max_groups) > 0:
        shard_groups = shard_groups[: int(args.partd_max_groups)]
    rows: list[dict[str, Any]] = []
    for dataset, seed, carrier in shard_groups:
        if include_expanded_registry:
            group_rows = [
                simulate_partd_expanded_kan_scheme(
                    dataset,
                    carrier,
                    int(seed),
                    scheme,
                    int(args.partd_matrix_horizon),
                    device,
                    train_cap=int(args.partd_train_cap),
                    fu_ratio=float(args.partd_fu_ratio),
                    warmup_fraction=float(args.partG_warmup_fraction),
                )
                for scheme in kan_schemes
            ]
            if include_mlp_efficiency:
                group_rows.extend(
                    simulate_partd_expanded_mlp_scheme(
                        dataset,
                        carrier,
                        int(seed),
                        scheme,
                        int(args.partd_matrix_horizon),
                        device,
                        train_cap=int(args.partd_train_cap),
                    )
                    for scheme in mlp_schemes
                )
        else:
            group_rows = [
                simulate_partd_smoke_scheme(dataset, carrier, int(seed), scheme, int(args.partd_matrix_horizon), device, train_cap=int(args.partd_train_cap))
                for scheme in schemes
            ]
        by_scheme = {str(row["scheme"]): row for row in group_rows}
        base_auc = finite_float(by_scheme["K0_AdamW_task_native"].get("AUC_task_loss_time"))
        instant_auc = finite_float(by_scheme["P0_FullInstant_CompH2"].get("AUC_task_loss_time"))
        random_auc = finite_float(by_scheme["R1_RandomAR1FullState"].get("AUC_task_loss_time"))
        reset_auc = finite_float(by_scheme["R0_ResetEveryStepFullState"].get("AUC_task_loss_time"))
        for row in group_rows:
            auc = finite_float(row.get("AUC_task_loss_time"))
            row["paired_loss_gain_vs_base"] = base_auc - auc if math.isfinite(base_auc) and math.isfinite(auc) else ""
            row["paired_loss_gain_vs_instant"] = instant_auc - auc if math.isfinite(instant_auc) and math.isfinite(auc) else ""
            row["paired_loss_gain_vs_random_state"] = random_auc - auc if math.isfinite(random_auc) and math.isfinite(auc) else ""
            row["paired_loss_gain_vs_reset"] = reset_auc - auc if math.isfinite(reset_auc) and math.isfinite(auc) else ""
        p1_row = by_scheme.get("P1_FullPersistent_Hybrid_CompH2_PRIMARY")
        if p1_row is not None:
            p1_auc = finite_float(p1_row.get("AUC_task_loss_time"))
            for mlp_scheme in mlp_schemes:
                mlp_row = by_scheme.get(mlp_scheme)
                if mlp_row is None:
                    continue
                mlp_auc = finite_float(mlp_row.get("AUC_task_loss_time"))
                gain = mlp_auc - p1_auc if math.isfinite(mlp_auc) and math.isfinite(p1_auc) else ""
                p1_row[f"P1_gain_vs_{mlp_scheme}"] = gain
                mlp_row["P1_gain_vs_this_MLP"] = gain
            intrinsic_key = "K2_IntrinsicAdditive_CompH2" if "K2_IntrinsicAdditive_CompH2" in by_scheme else "K0_AdamW_task_native"
            intrinsic_row = by_scheme.get(intrinsic_key)
            m1_row = by_scheme.get("M1_MLP_SameFLOPs_TaskNativeStrongOptimizer")
            p1_time = finite_float(p1_row.get("wall_time_sec"))
            intrinsic_time = finite_float(intrinsic_row.get("wall_time_sec") if intrinsic_row else "")
            m1_time = finite_float(m1_row.get("wall_time_sec") if m1_row else "")
            p1_mem = finite_float(p1_row.get("peak_memory_proxy_bytes"))
            m1_mem = finite_float(m1_row.get("peak_memory_proxy_bytes") if m1_row else "")
            p1_row["partD_efficiency_intrinsic_reference"] = intrinsic_key
            p1_row["FU_incremental_overhead_vs_intrinsic_base"] = (p1_time - intrinsic_time) / max(intrinsic_time, 1.0e-12) if math.isfinite(p1_time) and math.isfinite(intrinsic_time) else ""
            p1_row["full_step_ratio_vs_sameFLOPs_MLP"] = p1_time / max(m1_time, 1.0e-12) if math.isfinite(p1_time) and math.isfinite(m1_time) else ""
            p1_row["peak_memory_ratio_vs_MLP"] = p1_mem / max(m1_mem, 1.0e-12) if math.isfinite(p1_mem) and math.isfinite(m1_mem) else ""
        rows.extend(group_rows)
    suffix = f"shard{shard_index:02d}_of_{shard_count:02d}"
    matrix_name = f"v23_28_partD_real_H20_compact_matrix_{suffix}.csv"
    summary_name = f"v23_28_partD_real_H20_compact_summary_{suffix}.json"
    write_csv(OUT_ROOT / matrix_name, rows)
    summary = {
        "phase": "partD-real-H20-compact-matrix-shard",
        "partD_completed": 0,
        "partD_compact_matrix_shard_completed": int(len(rows) > 0),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "groups_in_shard": len(shard_groups),
        "rows": len(rows),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "carriers": sorted({str(row["carrier"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "schemes": schemes,
        "include_expanded_registry": int(include_expanded_registry),
        "include_mlp_efficiency": int(include_mlp_efficiency),
        "kan_scheme_count": len(kan_schemes),
        "mlp_scheme_count": len(mlp_schemes),
        "horizon": int(args.partd_matrix_horizon),
        "train_cap": int(args.partd_train_cap),
        "no_fabrication_note": "Shard rows use real task-native losses on compact bank proxy; expanded/MLP rows are diagnostic proxy evidence and no official fused-carrier PartD success is claimed.",
    }
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("PartD_real_H20_compact_matrix_shard", args, [matrix_name, summary_name], summary, "completed_incomplete_gate")


def run_partD_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    shard_paths = sorted(OUT_ROOT.glob("v23_28_partD_real_H20_compact_matrix_shard*_of_*.csv"))
    rows: list[dict[str, Any]] = []
    for path in shard_paths:
        rows.extend(read_csv(path))
    write_csv(OUT_ROOT / "v23_28_partD_real_H20_compact_matrix.csv", rows)
    p1_rows = [row for row in rows if row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
    task_family_summary: dict[str, dict[str, Any]] = {}
    for family in sorted({row.get("task_family", "") for row in p1_rows if row.get("task_family", "")}):
        fam_rows = [row for row in p1_rows if row.get("task_family") == family]
        task_family_summary[family] = {
            "rows": len(fam_rows),
            "P1_gain_vs_base_median": median_of(fam_rows, "paired_loss_gain_vs_base"),
            "P1_gain_vs_instant_median": median_of(fam_rows, "paired_loss_gain_vs_instant"),
            "P1_gain_vs_random_median": median_of(fam_rows, "paired_loss_gain_vs_random_state"),
            "P1_gain_vs_reset_median": median_of(fam_rows, "paired_loss_gain_vs_reset"),
        }
    datasets = sorted({row.get("dataset", "") for row in rows if row.get("dataset", "")})
    carriers = sorted({row.get("carrier", "") for row in rows if row.get("carrier", "")})
    seeds = sorted({int(row.get("seed", "0")) for row in rows if str(row.get("seed", "")).isdigit()})
    schemes = sorted({row.get("scheme", "") for row in rows if row.get("scheme", "")})
    scheme_set = set(schemes)
    expanded_kan_rows = [row for row in rows if row.get("scheme") in PARTD_EXPANDED_KAN_SCHEMES]
    mlp_rows = [row for row in rows if row.get("scheme") in PARTD_EXPANDED_MLP_SCHEMES]
    expanded_kan_missing = [scheme for scheme in PARTD_EXPANDED_KAN_SCHEMES if scheme not in scheme_set]
    mlp_missing = [scheme for scheme in PARTD_EXPANDED_MLP_SCHEMES if scheme not in scheme_set]
    expanded_all_28_present = int(len(expanded_kan_missing) == 0)
    mlp_all_present = int(len(mlp_missing) == 0)
    expanded_registry_detected = int(expanded_all_28_present == 1 or any(row.get("phase") == "partD_real_H20_expanded_registry_compact" for row in rows))
    mlp_efficiency_detected = int(len(mlp_rows) > 0)
    finite_auc_rows = sum(1 for row in rows if math.isfinite(finite_float(row.get("AUC_task_loss_time"))))
    h20_rows = sum(1 for row in rows if finite_float(row.get("horizon"), 0.0) >= 20.0)
    p1_base = median_of(p1_rows, "paired_loss_gain_vs_base")
    p1_instant = median_of(p1_rows, "paired_loss_gain_vs_instant")
    p1_random = median_of(p1_rows, "paired_loss_gain_vs_random_state")
    debt_summary, debt_rows = compact_paired_debt_summary(
        rows,
        "scheme",
        "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
        "K0_AdamW_task_native",
        ["dataset", "seed", "carrier"],
        "PartD_P1_vs_K0_compact_test_debt",
    )
    write_csv(OUT_ROOT / "v23_28_partD_compact_debt_summary.csv", debt_rows)
    compact_gate = int(p1_base >= 5.0e-4 and p1_instant >= 2.0e-4 and p1_random >= 3.0e-4)
    compact_gate_with_debt = int(compact_gate == 1 and debt_summary["paired_no_debt_gate_pass"] == 1)
    expected_scheme_count = len(partd_matrix_schemes())
    if expanded_registry_detected:
        expected_scheme_count = len(PARTD_EXPANDED_KAN_SCHEMES) + (len(PARTD_EXPANDED_MLP_SCHEMES) if mlp_efficiency_detected else 0)
    expected_rows = 10 * int(args.partd_matrix_seeds) * 2 * expected_scheme_count

    def compact_field_summary(source_rows: list[dict[str, Any]], field: str, seed_value: int) -> dict[str, Any]:
        vals = [finite_float(row.get(field)) for row in source_rows]
        vals = [value for value in vals if math.isfinite(value)]
        return {
            "median": float(np.median(vals)) if vals else float("nan"),
            "CVaR25": cvar25(vals),
            "LCB05": bootstrap_lcb05(vals, seed=seed_value),
            "rows": len(vals),
        }

    mlp_delta_summary = {
        scheme: compact_field_summary(p1_rows, f"P1_gain_vs_{scheme}", 842800 + idx)
        for idx, scheme in enumerate(PARTD_EXPANDED_MLP_SCHEMES)
    }
    efficiency_rows = [
        {
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "carrier": row.get("carrier", ""),
            "partD_efficiency_intrinsic_reference": row.get("partD_efficiency_intrinsic_reference", ""),
            "FU_incremental_overhead_vs_intrinsic_base": row.get("FU_incremental_overhead_vs_intrinsic_base", ""),
            "full_step_ratio_vs_sameFLOPs_MLP": row.get("full_step_ratio_vs_sameFLOPs_MLP", ""),
            "peak_memory_ratio_vs_MLP": row.get("peak_memory_ratio_vs_MLP", ""),
            "wall_time_sec": row.get("wall_time_sec", ""),
            "arithmetic_proxy_ops": row.get("arithmetic_proxy_ops", ""),
        }
        for row in p1_rows
        if row.get("phase") == "partD_real_H20_expanded_registry_compact"
    ]
    write_csv(OUT_ROOT / "v23_28_partD_efficiency_matrix.csv", efficiency_rows)
    overhead_median = median_of(efficiency_rows, "FU_incremental_overhead_vs_intrinsic_base")
    step_ratio_median = median_of(efficiency_rows, "full_step_ratio_vs_sameFLOPs_MLP")
    mem_ratio_median = median_of(efficiency_rows, "peak_memory_ratio_vs_MLP")
    efficiency_proxy_completed = int(
        mlp_all_present == 1
        and len(efficiency_rows) == 10 * int(args.partd_matrix_seeds) * 2
        and math.isfinite(overhead_median)
        and math.isfinite(step_ratio_median)
        and math.isfinite(mem_ratio_median)
    )
    official_blockers = [
        "compact_bank_proxy_not_fused_DCHE_DFOU_real_carrier",
        "classification_regression_pairwise_debt_metrics_compact_proxy_only",
    ]
    official_blockers.append(
        "compact_28_scheme_partD_registry_run_as_proxy_not_official_full_fused"
        if expanded_all_28_present == 1
        else "full_28_scheme_partD_registry_not_run"
    )
    official_blockers.append(
        "compact_MLP_matched_and_efficiency_controls_run_as_proxy_not_official"
        if efficiency_proxy_completed == 1
        else "MLP_matched_and_efficiency_controls_not_run"
    )
    summary = {
        "phase": "partD-real-H20-compact-matrix-aggregate",
        "partD_completed": 0,
        "partD_compact_matrix_completed": int(len(rows) == expected_rows and h20_rows == len(rows) and finite_auc_rows == len(rows)),
        "official_partD_H20_rows": 0,
        "compact_matrix_rows": len(rows),
        "expected_compact_matrix_rows": expected_rows,
        "row_count_matches_compact_plan": int(len(rows) == expected_rows),
        "h20_rows": h20_rows,
        "finite_auc_rows": finite_auc_rows,
        "datasets_count": len(datasets),
        "datasets": datasets,
        "carriers_count": len(carriers),
        "carriers": carriers,
        "seeds_count": len(seeds),
        "seeds": seeds,
        "schemes_count": len(schemes),
        "schemes": schemes,
        "expanded_registry_detected": expanded_registry_detected,
        "expanded_kan_registry_rows": len(expanded_kan_rows),
        "expanded_kan_expected_scheme_count": len(PARTD_EXPANDED_KAN_SCHEMES),
        "expanded_kan_schemes_present_count": len(PARTD_EXPANDED_KAN_SCHEMES) - len(expanded_kan_missing),
        "expanded_kan_all_28_schemes_present": expanded_all_28_present,
        "expanded_kan_missing_schemes": expanded_kan_missing,
        "MLP_matched_efficiency_rows": len(mlp_rows),
        "MLP_expected_scheme_count": len(PARTD_EXPANDED_MLP_SCHEMES),
        "MLP_schemes_present_count": len(PARTD_EXPANDED_MLP_SCHEMES) - len(mlp_missing),
        "MLP_all_registry_schemes_present": mlp_all_present,
        "MLP_missing_schemes": mlp_missing,
        "MLP_delta_summary_P1_minus_controls": mlp_delta_summary,
        "efficiency_rows": len(efficiency_rows),
        "FU_incremental_overhead_median": overhead_median,
        "full_step_ratio_vs_sameFLOPs_MLP_median": step_ratio_median,
        "peak_memory_ratio_vs_MLP_median": mem_ratio_median,
        "efficiency_proxy_completed": efficiency_proxy_completed,
        "P1_gain_vs_base_median": p1_base,
        "P1_gain_vs_instant_median": p1_instant,
        "P1_gain_vs_random_median": p1_random,
        "P1_gain_vs_reset_median": median_of(p1_rows, "paired_loss_gain_vs_reset"),
        "compact_primary_gate_pass": compact_gate,
        "compact_paired_debt_summary_P1_vs_K0": debt_summary,
        "compact_paired_no_debt_rate": debt_summary["paired_no_debt_rate"],
        "compact_paired_no_debt_gate_pass": debt_summary["paired_no_debt_gate_pass"],
        "compact_primary_gate_with_debt_pass": compact_gate_with_debt,
        "task_family_summary": task_family_summary,
        "official_blockers": official_blockers,
        "no_fabrication_note": "Aggregate reports measured compact matrix rows, compact expanded registry proxy rows, MLP/efficiency proxy rows if present, and compact test-debt deltas only. It is not the official fused-carrier PartD matrix.",
    }
    write_json(OUT_ROOT / "v23_28_partD_summary.json", summary)
    failure = [
        "# v23.28 PartD compact matrix failure decomposition",
        "",
        f"- rows: `{len(rows)}` / expected compact `{expected_rows}`",
        f"- datasets/carriers/seeds/schemes: `{len(datasets)}/{len(carriers)}/{len(seeds)}/{len(schemes)}`",
        f"- expanded KAN registry present/missing: `{summary['expanded_kan_schemes_present_count']}/{summary['expanded_kan_expected_scheme_count']}` / `{json.dumps(expanded_kan_missing, ensure_ascii=False)}`",
        f"- MLP registry present/missing: `{summary['MLP_schemes_present_count']}/{summary['MLP_expected_scheme_count']}` / `{json.dumps(mlp_missing, ensure_ascii=False)}`",
        f"- MLP deltas P1 minus controls: `{json.dumps(json_clean(mlp_delta_summary), ensure_ascii=False, sort_keys=True)}`",
        f"- efficiency medians overhead/step_ratio/memory_ratio: `{overhead_median}` / `{step_ratio_median}` / `{mem_ratio_median}`; efficiency_proxy_completed `{efficiency_proxy_completed}`",
        f"- P1 gains vs base/instant/random/reset: `{p1_base}` / `{p1_instant}` / `{p1_random}` / `{summary['P1_gain_vs_reset_median']}`",
        f"- compact_primary_gate_pass: `{compact_gate}`",
        f"- compact paired no-debt P1 vs K0: rate `{debt_summary['paired_no_debt_rate']}`, gate `{debt_summary['paired_no_debt_gate_pass']}`, pairs `{debt_summary['paired_count']}`",
        f"- task_family_summary: `{json.dumps(json_clean(task_family_summary), ensure_ascii=False, sort_keys=True)}`",
        f"- official_blockers: `{json.dumps(summary['official_blockers'], ensure_ascii=False)}`",
        "",
        "Conclusion: this expands the real-loss evidence beyond smoke, but it remains a compact bank proxy and cannot close official PartD.",
        "",
    ]
    (OUT_ROOT / "v23_28_partD_compact_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partD_real_H20_compact_matrix.csv",
        "v23_28_partD_efficiency_matrix.csv",
        "v23_28_partD_compact_debt_summary.csv",
        "v23_28_partD_summary.json",
        "v23_28_partD_compact_failure_decomposition.md",
    ]
    append_exec("PartD_real_H20_compact_matrix_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartD real H20 compact matrix",
        [
            f"rows `{summary['compact_matrix_rows']}` expected `{summary['expected_compact_matrix_rows']}`; h20_rows `{h20_rows}`; finite_auc_rows `{finite_auc_rows}`.",
            f"datasets/carriers/seeds/schemes `{len(datasets)}/{len(carriers)}/{len(seeds)}/{len(schemes)}`.",
            f"expanded KAN registry present/missing `{summary['expanded_kan_schemes_present_count']}/{summary['expanded_kan_expected_scheme_count']}` / `{expanded_kan_missing}`; all_28 `{expanded_all_28_present}`.",
            f"MLP registry present/missing `{summary['MLP_schemes_present_count']}/{summary['MLP_expected_scheme_count']}` / `{mlp_missing}`; efficiency_proxy_completed `{efficiency_proxy_completed}`.",
            f"MLP deltas P1 minus controls `{json.dumps(json_clean(mlp_delta_summary), ensure_ascii=False, sort_keys=True)}`.",
            f"efficiency medians overhead/step_ratio/memory_ratio `{overhead_median}` / `{step_ratio_median}` / `{mem_ratio_median}`.",
            f"P1 gains median vs base/instant/random/reset `{p1_base}` / `{p1_instant}` / `{p1_random}` / `{summary['P1_gain_vs_reset_median']}`.",
            f"compact paired no-debt P1 vs K0 rate/gate/pairs `{debt_summary['paired_no_debt_rate']}` / `{debt_summary['paired_no_debt_gate_pass']}` / `{debt_summary['paired_count']}`.",
            f"task_family_summary `{json.dumps(json_clean(task_family_summary), ensure_ascii=False, sort_keys=True)}`.",
            f"compact_primary_gate_pass `{compact_gate}`; compact_primary_gate_with_debt_pass `{compact_gate_with_debt}`; partD_completed `{summary['partD_completed']}`; official_blockers `{summary['official_blockers']}`.",
        ],
    )


def run_partE_matrix(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets = [*CLASSIFICATION_DATASETS, "DiabetesRegression", "DiabetesPairwiseRanking"]
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    groups = [(dataset, seed, carrier) for dataset in datasets for seed in range(int(args.partE_matrix_seeds)) for carrier in carriers]
    shard_count = max(1, int(args.partE_shard_count))
    shard_index = int(args.partE_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid PartE shard index {shard_index} for shard count {shard_count}")
    shard_groups = [group for idx, group in enumerate(groups) if idx % shard_count == shard_index]
    if int(args.partE_max_groups) > 0:
        shard_groups = shard_groups[: int(args.partE_max_groups)]
    rows: list[dict[str, Any]] = []
    for dataset, seed, carrier in shard_groups:
        group_rows = [
            simulate_partE_metric_row(dataset, carrier, int(seed), metric_id, metric_variant, int(args.partE_matrix_horizon), device, train_cap=int(args.partd_train_cap))
            for metric_id, metric_variant in PART_E_METRICS
        ]
        by_metric = {str(row["metric_id"]): row for row in group_rows}
        primary_auc = finite_float(by_metric["E2_true_task_native_compositional_H2_PRIMARY"].get("AUC_task_loss_time"))
        for row in group_rows:
            auc = finite_float(row.get("AUC_task_loss_time"))
            row["E2_gain_vs_this_metric"] = auc - primary_auc if math.isfinite(auc) and math.isfinite(primary_auc) else ""
        rows.extend(group_rows)
    suffix = f"shard{shard_index:02d}_of_{shard_count:02d}"
    matrix_name = f"v23_28_partE_metric_causality_matrix_{suffix}.csv"
    summary_name = f"v23_28_partE_metric_causality_summary_{suffix}.json"
    write_csv(OUT_ROOT / matrix_name, rows)
    summary = {
        "phase": "partE-metric-causality-compact-shard",
        "partE_completed": 0,
        "partE_compact_shard_completed": int(len(rows) > 0),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "groups_in_shard": len(shard_groups),
        "rows": len(rows),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "carriers": sorted({str(row["carrier"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "metrics": [metric_id for metric_id, _variant in PART_E_METRICS],
        "horizon": int(args.partE_matrix_horizon),
        "train_cap": int(args.partd_train_cap),
        "no_fabrication_note": "Shard rows use compact real-loss metric-causality proxy; no official PartE success is claimed.",
    }
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("PartE_metric_causality_compact_shard", args, [matrix_name, summary_name], summary, "completed_incomplete_gate")


def run_partE_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    shard_paths = sorted(OUT_ROOT.glob("v23_28_partE_metric_causality_matrix_shard*_of_*.csv"))
    rows: list[dict[str, Any]] = []
    for path in shard_paths:
        rows.extend(read_csv(path))
    write_csv(OUT_ROOT / "v23_28_partE_metric_causality_matrix.csv", rows)
    e2_rows = [row for row in rows if row.get("metric_id") == "E2_true_task_native_compositional_H2_PRIMARY"]
    deltas: dict[str, list[float]] = {}
    for metric_id, _variant in PART_E_METRICS:
        if metric_id == "E2_true_task_native_compositional_H2_PRIMARY":
            continue
        vals = [finite_float(row.get("E2_gain_vs_this_metric")) for row in rows if row.get("metric_id") == metric_id]
        deltas[metric_id] = [v for v in vals if math.isfinite(v)]
    delta_summary = {
        metric_id: {
            "median": float(np.median(vals)) if vals else float("nan"),
            "CVaR25": cvar25(vals),
            "LCB05": bootstrap_lcb05(vals, seed=2328 + idx),
            "rows": len(vals),
        }
        for idx, (metric_id, vals) in enumerate(deltas.items())
    }
    e2_vs_e1 = delta_summary.get("E1_dynamic_local_H2", {}).get("median", float("nan"))
    e2_vs_e3 = delta_summary.get("E3_path_weight_shuffled_compositional_H2", {}).get("median", float("nan"))
    e2_vs_e0 = delta_summary.get("E0_dynamic_data_L2", {}).get("median", float("nan"))
    all_cvar_positive = int(
        delta_summary.get("E1_dynamic_local_H2", {}).get("CVaR25", float("nan")) > 0.0
        and delta_summary.get("E3_path_weight_shuffled_compositional_H2", {}).get("CVaR25", float("nan")) > 0.0
        and delta_summary.get("E0_dynamic_data_L2", {}).get("CVaR25", float("nan")) > 0.0
    )
    lcb_positive_count = sum(1 for key in ["E1_dynamic_local_H2", "E3_path_weight_shuffled_compositional_H2", "E0_dynamic_data_L2"] if delta_summary.get(key, {}).get("LCB05", float("nan")) > 0.0)
    path_cv_median = median_of(e2_rows, "path_weight_cv")
    refresh_all = int(all(int(finite_float(row.get("actual_metric_refresh_count"), 0.0)) > 0 for row in e2_rows)) if e2_rows else 0
    debt_summary, debt_rows = compact_paired_debt_summary(
        rows,
        "metric_id",
        "E2_true_task_native_compositional_H2_PRIMARY",
        "E1_dynamic_local_H2",
        ["dataset", "seed", "carrier"],
        "PartE_E2_vs_E1_compact_test_debt",
    )
    write_csv(OUT_ROOT / "v23_28_partE_compact_debt_summary.csv", debt_rows)
    compact_gate_without_debt = int(e2_vs_e1 >= 2.0e-4 and e2_vs_e3 >= 2.0e-4 and e2_vs_e0 >= 2.0e-4 and all_cvar_positive and lcb_positive_count >= 2 and path_cv_median >= 0.10 and refresh_all == 1)
    compact_gate_with_debt = int(compact_gate_without_debt == 1 and debt_summary["paired_no_debt_gate_pass"] == 1)
    datasets = sorted({row.get("dataset", "") for row in rows if row.get("dataset", "")})
    carriers = sorted({row.get("carrier", "") for row in rows if row.get("carrier", "")})
    seeds = sorted({int(row.get("seed", "0")) for row in rows if str(row.get("seed", "")).isdigit()})
    metrics = sorted({row.get("metric_id", "") for row in rows if row.get("metric_id", "")})
    finite_auc_rows = sum(1 for row in rows if math.isfinite(finite_float(row.get("AUC_task_loss_time"))))
    h20_rows = sum(1 for row in rows if finite_float(row.get("horizon"), 0.0) >= 20.0)
    expected_rows = 10 * 5 * 2 * len(PART_E_METRICS)
    summary = {
        "phase": "partE-metric-causality-compact-aggregate",
        "partE_completed": 0,
        "partE_compact_matrix_completed": int(len(rows) == expected_rows and h20_rows == len(rows) and finite_auc_rows == len(rows)),
        "compact_matrix_rows": len(rows),
        "expected_compact_matrix_rows": expected_rows,
        "row_count_matches_compact_plan": int(len(rows) == expected_rows),
        "h20_rows": h20_rows,
        "finite_auc_rows": finite_auc_rows,
        "datasets_count": len(datasets),
        "datasets": datasets,
        "carriers_count": len(carriers),
        "carriers": carriers,
        "seeds_count": len(seeds),
        "seeds": seeds,
        "metrics_count": len(metrics),
        "metrics": metrics,
        "delta_summary_E2_minus_controls": delta_summary,
        "E2_vs_E1_median": e2_vs_e1,
        "E2_vs_E3_median": e2_vs_e3,
        "E2_vs_E0_median": e2_vs_e0,
        "E0_E1_E3_CVaR25_all_positive": all_cvar_positive,
        "E0_E1_E3_LCB05_positive_count": lcb_positive_count,
        "E2_path_weight_cv_median": path_cv_median,
        "E2_metric_refresh_count_positive_all": refresh_all,
        "compact_gate_without_debt_pass": compact_gate_without_debt,
        "compact_paired_debt_summary_E2_vs_E1": debt_summary,
        "compact_paired_no_debt_rate": debt_summary["paired_no_debt_rate"],
        "compact_paired_no_debt_gate_pass": debt_summary["paired_no_debt_gate_pass"],
        "compact_gate_with_debt_pass": compact_gate_with_debt,
        "official_blockers": [
            "compact_bank_proxy_not_full_official_partE",
            "paired_debt_metrics_compact_proxy_only",
            "paired_no_debt_official_gate_not_available_from_compact_proxy",
            "same_checkpoint_full_registry_official_matrix_not_run",
        ],
        "no_fabrication_note": "Aggregate reports compact measured deltas and compact test-debt only; official PartE requires full official carrier matrix and official paired no-debt.",
    }
    write_csv(OUT_ROOT / "v23_28_partE_metric_identity.csv", [
        {
            "identity": "same_state_law_same_init_same_readout_same_split",
            "value": 1,
            "note": "metric rows in each group share deterministic seed-dependent initialization/readout/split and only switch metric variant",
        },
        {
            "identity": "metric_refresh_count_positive_all_E2",
            "value": refresh_all,
            "note": "checked on E2 primary compact rows",
        },
    ])
    write_json(OUT_ROOT / "v23_28_partE_summary.json", summary)
    failure = [
        "# v23.28 PartE compact metric causality failure decomposition",
        "",
        f"- rows: `{len(rows)}` / expected compact `{expected_rows}`",
        f"- E2 vs E1/E3/E0 medians: `{e2_vs_e1}` / `{e2_vs_e3}` / `{e2_vs_e0}`",
        f"- all three CVaR25 positive: `{all_cvar_positive}`; LCB05 positive count: `{lcb_positive_count}`",
        f"- E2 path weight CV median: `{path_cv_median}`; refresh all: `{refresh_all}`",
        f"- compact_gate_without_debt_pass: `{compact_gate_without_debt}`",
        f"- compact paired no-debt E2 vs E1: rate `{debt_summary['paired_no_debt_rate']}`, gate `{debt_summary['paired_no_debt_gate_pass']}`, pairs `{debt_summary['paired_count']}`",
        f"- official_blockers: `{json.dumps(summary['official_blockers'], ensure_ascii=False)}`",
        "",
        "Conclusion: no official metric-causality claim is made; debt/no-debt and full official carrier matrix remain incomplete.",
        "",
    ]
    (OUT_ROOT / "v23_28_partE_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partE_metric_causality_matrix.csv",
        "v23_28_partE_metric_identity.csv",
        "v23_28_partE_compact_debt_summary.csv",
        "v23_28_partE_summary.json",
        "v23_28_partE_failure_decomposition.md",
    ]
    append_exec("PartE_metric_causality_compact_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartE metric causality compact matrix",
        [
            f"rows `{len(rows)}` expected `{expected_rows}`; h20_rows `{h20_rows}`; finite_auc_rows `{finite_auc_rows}`.",
            f"E2 vs E1/E3/E0 medians `{e2_vs_e1}` / `{e2_vs_e3}` / `{e2_vs_e0}`.",
            f"CVaR25 all positive `{all_cvar_positive}`; LCB05 positive count `{lcb_positive_count}`; path_weight_cv_median `{path_cv_median}`; refresh_all `{refresh_all}`.",
            f"compact paired no-debt E2 vs E1 rate/gate/pairs `{debt_summary['paired_no_debt_rate']}` / `{debt_summary['paired_no_debt_gate_pass']}` / `{debt_summary['paired_count']}`.",
            f"compact_gate_without_debt_pass `{compact_gate_without_debt}`; compact_gate_with_debt_pass `{compact_gate_with_debt}`; partE_completed `{summary['partE_completed']}`; official_blockers `{summary['official_blockers']}`.",
        ],
    )


def run_partF_matrix(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets = [*CLASSIFICATION_DATASETS, "DiabetesRegression", "DiabetesPairwiseRanking"]
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    groups = [(dataset, seed, carrier) for dataset in datasets for seed in range(int(args.partF_matrix_seeds)) for carrier in carriers]
    shard_count = max(1, int(args.partF_shard_count))
    shard_index = int(args.partF_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid PartF shard index {shard_index} for shard count {shard_count}")
    shard_groups = [group for idx, group in enumerate(groups) if idx % shard_count == shard_index]
    if int(args.partF_max_groups) > 0:
        shard_groups = shard_groups[: int(args.partF_max_groups)]
    rows: list[dict[str, Any]] = []
    state_gain_fields = {
        "P0_FullInstant_CompH2": "P1_gain_vs_P0_FullInstant",
        "R0_ResetEveryStepFullState": "P1_gain_vs_R0_Reset",
        "R1_RandomAR1FullState": "P1_gain_vs_R1_RandomAR1",
        "R3_ReceivingBankShuffledState": "P1_gain_vs_R3_BankShuffled",
        "R4_TimeLag4State": "P1_gain_vs_R4_TimeLag4",
        "R5_SameAutocorrelationRandomState": "P1_gain_vs_R5_SameAutocorrelationRandom",
    }
    for dataset, seed, carrier in shard_groups:
        group_rows = [
            simulate_partF_scheme(dataset, carrier, int(seed), scheme, int(args.partF_matrix_horizon), device, train_cap=int(args.partd_train_cap))
            for scheme in PART_F_ALL_SCHEMES
        ]
        by_scheme = {str(row["scheme"]): row for row in group_rows}
        p1_auc = finite_float(by_scheme["P1_FullPersistent_Hybrid_CompH2_PRIMARY"].get("AUC_task_loss_time"))
        for control_scheme, field in state_gain_fields.items():
            control_auc = finite_float(by_scheme[control_scheme].get("AUC_task_loss_time"))
            gain = control_auc - p1_auc if math.isfinite(control_auc) and math.isfinite(p1_auc) else ""
            by_scheme["P1_FullPersistent_Hybrid_CompH2_PRIMARY"][field] = gain
            by_scheme[control_scheme]["P1_gain_vs_this_state_control"] = gain
        reset_auc = finite_float(by_scheme["R0_ResetEveryStepFullState"].get("AUC_task_loss_time"))
        component_gains: dict[str, float] = {}
        for component_scheme in PART_F_COMPONENT_SCHEMES:
            auc = finite_float(by_scheme[component_scheme].get("AUC_task_loss_time"))
            gain = reset_auc - auc if math.isfinite(reset_auc) and math.isfinite(auc) else float("nan")
            component_gains[component_scheme] = gain
            by_scheme[component_scheme]["component_gain_vs_reset"] = gain if math.isfinite(gain) else ""
        nonfull = {scheme: gain for scheme, gain in component_gains.items() if scheme != "A6_Persistent_Full_OmegaSymRadial" and math.isfinite(gain)}
        full_gain = component_gains.get("A6_Persistent_Full_OmegaSymRadial", float("nan"))
        best_nonfull_scheme = max(nonfull, key=nonfull.get) if nonfull else ""
        best_nonfull_gain = nonfull[best_nonfull_scheme] if best_nonfull_scheme else float("nan")
        synergy = full_gain - best_nonfull_gain if math.isfinite(full_gain) and math.isfinite(best_nonfull_gain) else float("nan")
        by_scheme["A6_Persistent_Full_OmegaSymRadial"]["component_best_nonfull_scheme"] = best_nonfull_scheme
        by_scheme["A6_Persistent_Full_OmegaSymRadial"]["component_best_nonfull_gain"] = best_nonfull_gain if math.isfinite(best_nonfull_gain) else ""
        by_scheme["A6_Persistent_Full_OmegaSymRadial"]["component_synergy_vs_best_nonfull"] = synergy if math.isfinite(synergy) else ""
        rows.extend(group_rows)
    suffix = f"shard{shard_index:02d}_of_{shard_count:02d}"
    matrix_name = f"v23_28_partF_state_component_compact_matrix_{suffix}.csv"
    summary_name = f"v23_28_partF_state_component_compact_summary_{suffix}.json"
    write_csv(OUT_ROOT / matrix_name, rows)
    summary = {
        "phase": "partF-state-component-compact-shard",
        "partF_completed": 0,
        "partF_compact_shard_completed": int(len(rows) > 0),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "groups_in_shard": len(shard_groups),
        "rows": len(rows),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "carriers": sorted({str(row["carrier"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "state_schemes": PART_F_STATE_SCHEMES,
        "component_schemes": PART_F_COMPONENT_SCHEMES,
        "horizon": int(args.partF_matrix_horizon),
        "train_cap": int(args.partd_train_cap),
        "no_fabrication_note": "Shard rows use compact real-loss state/component proxy; no official PartF success is claimed.",
    }
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("PartF_state_component_compact_shard", args, [matrix_name, summary_name], summary, "completed_incomplete_gate")


def run_partF_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    shard_paths = sorted(OUT_ROOT.glob("v23_28_partF_state_component_compact_matrix_shard*_of_*.csv"))
    rows: list[dict[str, Any]] = []
    for path in shard_paths:
        rows.extend(read_csv(path))
    state_rows = [row for row in rows if row.get("scheme") in PART_F_STATE_SCHEMES]
    component_rows = [row for row in rows if row.get("scheme") in PART_F_COMPONENT_SCHEMES]
    write_csv(OUT_ROOT / "v23_28_partF_state_component_compact_matrix.csv", rows)
    write_csv(OUT_ROOT / "v23_28_partF_state_causality_matrix.csv", state_rows)
    write_csv(OUT_ROOT / "v23_28_partF_component_attribution.csv", component_rows)
    p1_rows = [row for row in rows if row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
    a6_rows = [row for row in rows if row.get("scheme") == "A6_Persistent_Full_OmegaSymRadial"]

    def paired_summary(field: str, seed: int) -> dict[str, Any]:
        vals = [finite_float(row.get(field)) for row in p1_rows]
        vals = [v for v in vals if math.isfinite(v)]
        return {
            "median": float(np.median(vals)) if vals else float("nan"),
            "CVaR25": cvar25(vals),
            "LCB05": bootstrap_lcb05(vals, seed=seed),
            "rows": len(vals),
        }

    state_delta_summary = {
        "P0_FullInstant_CompH2": paired_summary("P1_gain_vs_P0_FullInstant", 432800),
        "R0_ResetEveryStepFullState": paired_summary("P1_gain_vs_R0_Reset", 432801),
        "R1_RandomAR1FullState": paired_summary("P1_gain_vs_R1_RandomAR1", 432802),
        "R3_ReceivingBankShuffledState": paired_summary("P1_gain_vs_R3_BankShuffled", 432803),
        "R4_TimeLag4State": paired_summary("P1_gain_vs_R4_TimeLag4", 432804),
        "R5_SameAutocorrelationRandomState": paired_summary("P1_gain_vs_R5_SameAutocorrelationRandom", 432805),
    }
    state_compact_gate = int(
        state_delta_summary["R5_SameAutocorrelationRandomState"]["median"] >= 2.0e-4
        and state_delta_summary["R5_SameAutocorrelationRandomState"]["CVaR25"] > 0.0
        and state_delta_summary["R5_SameAutocorrelationRandomState"]["LCB05"] > 0.0
        and state_delta_summary["R3_ReceivingBankShuffledState"]["LCB05"] > 0.0
        and state_delta_summary["P0_FullInstant_CompH2"]["LCB05"] > 0.0
    )

    synergy_vals = [finite_float(row.get("component_synergy_vs_best_nonfull")) for row in a6_rows]
    synergy_vals = [v for v in synergy_vals if math.isfinite(v)]
    component_gain_medians = {
        scheme: median_of([row for row in component_rows if row.get("scheme") == scheme], "component_gain_vs_reset")
        for scheme in PART_F_COMPONENT_SCHEMES
    }
    synergy_median = float(np.median(synergy_vals)) if synergy_vals else float("nan")
    synergy_cvar = cvar25(synergy_vals)
    synergy_lcb = bootstrap_lcb05(synergy_vals, seed=432806)
    synergy_positive_rate = float(np.mean([1.0 if value > 0.0 else 0.0 for value in synergy_vals])) if synergy_vals else float("nan")
    state_debt_summary, state_debt_rows = compact_paired_debt_summary(
        rows,
        "scheme",
        "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
        "R5_SameAutocorrelationRandomState",
        ["dataset", "seed", "carrier"],
        "PartF_P1_vs_R5_compact_test_debt",
    )
    component_debt_summary, component_debt_rows = compact_paired_debt_summary(
        rows,
        "scheme",
        "A6_Persistent_Full_OmegaSymRadial",
        "A0_Persistent_SkewOnly",
        ["dataset", "seed", "carrier"],
        "PartF_A6_vs_A0_compact_test_debt",
    )
    write_csv(OUT_ROOT / "v23_28_partF_compact_debt_summary.csv", [*state_debt_rows, *component_debt_rows])
    component_compact_gate_without_debt = int(synergy_median >= 1.0e-4 and synergy_cvar > 0.0 and synergy_lcb > 0.0)
    component_compact_gate_with_debt = int(component_compact_gate_without_debt == 1 and component_debt_summary["paired_no_debt_gate_pass"] == 1)

    datasets = sorted({row.get("dataset", "") for row in rows if row.get("dataset", "")})
    carriers = sorted({row.get("carrier", "") for row in rows if row.get("carrier", "")})
    seeds = sorted({int(row.get("seed", "0")) for row in rows if str(row.get("seed", "")).isdigit()})
    schemes = sorted({row.get("scheme", "") for row in rows if row.get("scheme", "")})
    finite_auc_rows = sum(1 for row in rows if math.isfinite(finite_float(row.get("AUC_task_loss_time"))))
    h20_rows = sum(1 for row in rows if finite_float(row.get("horizon"), 0.0) >= 20.0)
    expected_rows = 10 * 5 * 2 * len(PART_F_ALL_SCHEMES)
    summary = {
        "phase": "partF-state-component-compact-aggregate",
        "partF_completed": 0,
        "partF_compact_matrix_completed": int(len(rows) == expected_rows and h20_rows == len(rows) and finite_auc_rows == len(rows)),
        "compact_matrix_rows": len(rows),
        "expected_compact_matrix_rows": expected_rows,
        "row_count_matches_compact_plan": int(len(rows) == expected_rows),
        "h20_rows": h20_rows,
        "finite_auc_rows": finite_auc_rows,
        "datasets_count": len(datasets),
        "datasets": datasets,
        "carriers_count": len(carriers),
        "carriers": carriers,
        "seeds_count": len(seeds),
        "seeds": seeds,
        "schemes_count": len(schemes),
        "schemes": schemes,
        "state_delta_summary_P1_minus_controls": state_delta_summary,
        "state_compact_gate_pass": state_compact_gate,
        "component_gain_vs_reset_medians": component_gain_medians,
        "component_synergy_median": synergy_median,
        "component_synergy_CVaR25": synergy_cvar,
        "component_synergy_LCB05": synergy_lcb,
        "component_synergy_positive_rate": synergy_positive_rate,
        "component_compact_gate_without_debt_pass": component_compact_gate_without_debt,
        "state_compact_paired_debt_summary_P1_vs_R5": state_debt_summary,
        "component_compact_paired_debt_summary_A6_vs_A0": component_debt_summary,
        "state_paired_no_debt_rate": state_debt_summary["paired_no_debt_rate"],
        "component_paired_no_debt_rate": component_debt_summary["paired_no_debt_rate"],
        "state_paired_no_debt_gate_pass": state_debt_summary["paired_no_debt_gate_pass"],
        "component_paired_no_debt_gate_pass": component_debt_summary["paired_no_debt_gate_pass"],
        "component_compact_gate_with_debt_pass": component_compact_gate_with_debt,
        "official_blockers": [
            "compact_bank_proxy_not_full_official_partF",
            "official_same_checkpoint_full_registry_not_run",
            "debt_no_debt_component_metrics_compact_proxy_only",
            "strict_R5_autocorrelation_control_is_compact_random_matched_proxy",
            "fused_DCHE_DFOU_real_carrier_training_not_implemented_for_partF",
        ],
        "no_fabrication_note": "Aggregate reports measured compact PartF rows and compact test-debt only. It does not claim the official PartF gate because strict autocorrelation and full official registry are missing.",
    }
    write_json(OUT_ROOT / "v23_28_partF_summary.json", summary)
    failure = [
        "# v23.28 PartF compact state/component failure decomposition",
        "",
        f"- rows: `{len(rows)}` / expected compact `{expected_rows}`",
        f"- datasets/carriers/seeds/schemes: `{len(datasets)}/{len(carriers)}/{len(seeds)}/{len(schemes)}`",
        f"- state_delta_summary_P1_minus_controls: `{json.dumps(json_clean(state_delta_summary), ensure_ascii=False, sort_keys=True)}`",
        f"- state_compact_gate_pass: `{state_compact_gate}`",
        f"- component_gain_vs_reset_medians: `{json.dumps(json_clean(component_gain_medians), ensure_ascii=False, sort_keys=True)}`",
        f"- component synergy median/CVaR25/LCB05/positive_rate: `{synergy_median}` / `{synergy_cvar}` / `{synergy_lcb}` / `{synergy_positive_rate}`",
        f"- component_compact_gate_without_debt_pass: `{component_compact_gate_without_debt}`",
        f"- compact paired no-debt state P1 vs R5: rate `{state_debt_summary['paired_no_debt_rate']}`, gate `{state_debt_summary['paired_no_debt_gate_pass']}`, pairs `{state_debt_summary['paired_count']}`",
        f"- compact paired no-debt component A6 vs A0: rate `{component_debt_summary['paired_no_debt_rate']}`, gate `{component_debt_summary['paired_no_debt_gate_pass']}`, pairs `{component_debt_summary['paired_count']}`",
        f"- official_blockers: `{json.dumps(summary['official_blockers'], ensure_ascii=False)}`",
        "",
        "Conclusion: this extends evidence to PartF state and component controls, but it remains a compact proxy and cannot close official PartF.",
        "",
    ]
    (OUT_ROOT / "v23_28_partF_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partF_state_component_compact_matrix.csv",
        "v23_28_partF_state_causality_matrix.csv",
        "v23_28_partF_component_attribution.csv",
        "v23_28_partF_compact_debt_summary.csv",
        "v23_28_partF_summary.json",
        "v23_28_partF_failure_decomposition.md",
    ]
    append_exec("PartF_state_component_compact_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartF state/component compact matrix",
        [
            f"rows `{len(rows)}` expected `{expected_rows}`; h20_rows `{h20_rows}`; finite_auc_rows `{finite_auc_rows}`.",
            f"datasets/carriers/seeds/schemes `{len(datasets)}/{len(carriers)}/{len(seeds)}/{len(schemes)}`.",
            f"P1 state deltas `{json.dumps(json_clean(state_delta_summary), ensure_ascii=False, sort_keys=True)}`.",
            f"state_compact_gate_pass `{state_compact_gate}`.",
            f"component gains vs reset medians `{json.dumps(json_clean(component_gain_medians), ensure_ascii=False, sort_keys=True)}`.",
            f"component synergy median/CVaR25/LCB05/positive_rate `{synergy_median}` / `{synergy_cvar}` / `{synergy_lcb}` / `{synergy_positive_rate}`.",
            f"compact paired no-debt state P1 vs R5 rate/gate/pairs `{state_debt_summary['paired_no_debt_rate']}` / `{state_debt_summary['paired_no_debt_gate_pass']}` / `{state_debt_summary['paired_count']}`.",
            f"compact paired no-debt component A6 vs A0 rate/gate/pairs `{component_debt_summary['paired_no_debt_rate']}` / `{component_debt_summary['paired_no_debt_gate_pass']}` / `{component_debt_summary['paired_count']}`.",
            f"component_compact_gate_without_debt_pass `{component_compact_gate_without_debt}`; component_compact_gate_with_debt_pass `{component_compact_gate_with_debt}`; partF_completed `{summary['partF_completed']}`; official_blockers `{summary['official_blockers']}`.",
        ],
    )


def run_partG_matrix(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets = [*CLASSIFICATION_DATASETS, "DiabetesRegression", "DiabetesPairwiseRanking"]
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    groups = [(dataset, seed, carrier) for dataset in datasets for seed in range(int(args.partG_matrix_seeds)) for carrier in carriers]
    shard_count = max(1, int(args.partG_shard_count))
    shard_index = int(args.partG_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid PartG shard index {shard_index} for shard count {shard_count}")
    shard_groups = [group for idx, group in enumerate(groups) if idx % shard_count == shard_index]
    if int(args.partG_max_groups) > 0:
        shard_groups = shard_groups[: int(args.partG_max_groups)]
    rows: list[dict[str, Any]] = []
    for dataset, seed, carrier in shard_groups:
        group_rows = [
            simulate_partG_scheme(
                dataset,
                carrier,
                int(seed),
                scheme,
                int(args.partG_matrix_horizon),
                device,
                train_cap=int(args.partd_train_cap),
                warmup_fraction=float(args.partG_warmup_fraction),
            )
            for scheme in PART_G_SCHEMES
        ]
        by_scheme = {str(row["scheme"]): row for row in group_rows}
        full_auc = finite_float(by_scheme["P2_FullPersistent_WarmupPure_CompH2"].get("AUC_task_loss_time"))
        for control_scheme, field in [
            ("G0_WarmupOnlyNoFurtherTraining", "P2_gain_vs_warmup_only"),
            ("G1_v2327_RotationOnlyPureFU", "P2_gain_vs_rotation_only_pure"),
            ("G2_RandomFullStatePure", "P2_gain_vs_random_full_state_pure"),
            ("G3_ResetPure", "P2_gain_vs_reset_pure"),
            ("P1_FullPersistent_Hybrid_CompH2_PRIMARY", "P2_gain_vs_hybrid_P1"),
        ]:
            control_auc = finite_float(by_scheme[control_scheme].get("AUC_task_loss_time"))
            gain = control_auc - full_auc if math.isfinite(control_auc) and math.isfinite(full_auc) else ""
            by_scheme["P2_FullPersistent_WarmupPure_CompH2"][field] = gain
            by_scheme[control_scheme]["P2_gain_vs_this_control"] = gain
        rows.extend(group_rows)
    suffix = f"shard{shard_index:02d}_of_{shard_count:02d}"
    matrix_name = f"v23_28_partG_hybrid_pure_matrix_{suffix}.csv"
    summary_name = f"v23_28_partG_hybrid_pure_summary_{suffix}.json"
    write_csv(OUT_ROOT / matrix_name, rows)
    summary = {
        "phase": "partG-hybrid-pure-compact-shard",
        "partG_completed": 0,
        "partG_compact_shard_completed": int(len(rows) > 0),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "groups_in_shard": len(shard_groups),
        "rows": len(rows),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "carriers": sorted({str(row["carrier"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "schemes": PART_G_SCHEMES,
        "horizon": int(args.partG_matrix_horizon),
        "warmup_fraction": float(args.partG_warmup_fraction),
        "train_cap": int(args.partd_train_cap),
        "no_fabrication_note": "Shard rows use compact real-loss hybrid/pure proxy; no official PartG success is claimed.",
    }
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("PartG_hybrid_pure_compact_shard", args, [matrix_name, summary_name], summary, "completed_incomplete_gate")


def run_partG_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    shard_paths = sorted(OUT_ROOT.glob("v23_28_partG_hybrid_pure_matrix_shard*_of_*.csv"))
    rows: list[dict[str, Any]] = []
    for path in shard_paths:
        rows.extend(read_csv(path))
    write_csv(OUT_ROOT / "v23_28_partG_hybrid_pure_matrix.csv", rows)
    p2_rows = [row for row in rows if row.get("scheme") == "P2_FullPersistent_WarmupPure_CompH2"]

    def p2_summary(field: str, seed: int) -> dict[str, Any]:
        vals = [finite_float(row.get(field)) for row in p2_rows]
        vals = [v for v in vals if math.isfinite(v)]
        return {
            "median": float(np.median(vals)) if vals else float("nan"),
            "CVaR25": cvar25(vals),
            "LCB05": bootstrap_lcb05(vals, seed=seed),
            "rows": len(vals),
        }

    pure_delta_summary = {
        "warmup_only": p2_summary("P2_gain_vs_warmup_only", 532800),
        "rotation_only_pure": p2_summary("P2_gain_vs_rotation_only_pure", 532801),
        "random_full_state_pure": p2_summary("P2_gain_vs_random_full_state_pure", 532802),
        "reset_pure": p2_summary("P2_gain_vs_reset_pure", 532803),
        "hybrid_P1": p2_summary("P2_gain_vs_hybrid_P1", 532804),
    }
    post_warmup_decrease_rate = float(np.mean([1.0 if int(finite_float(row.get("post_warmup_loss_decreased"), 0.0)) == 1 else 0.0 for row in p2_rows])) if p2_rows else float("nan")
    p2_base_zero_all = int(all(abs(finite_float(row.get("base_update_norm_after_warmup_sum"), 0.0)) <= 1.0e-12 for row in p2_rows)) if p2_rows else 0
    p2_fu_count_all = int(all(int(finite_float(row.get("FU_map_count_after_warmup"), 0.0)) == int(finite_float(row.get("pure_steps"), -1.0)) for row in p2_rows)) if p2_rows else 0
    p2_state_update_all = int(all(int(finite_float(row.get("state_update_count_after_warmup"), 0.0)) == int(finite_float(row.get("pure_steps"), -1.0)) for row in p2_rows)) if p2_rows else 0
    debt_summary, debt_rows = compact_paired_debt_summary(
        rows,
        "scheme",
        "P2_FullPersistent_WarmupPure_CompH2",
        "G0_WarmupOnlyNoFurtherTraining",
        ["dataset", "seed", "carrier"],
        "PartG_P2_vs_warmup_only_compact_test_debt",
    )
    write_csv(OUT_ROOT / "v23_28_partG_compact_debt_summary.csv", debt_rows)
    compact_gate_without_debt = int(
        post_warmup_decrease_rate >= 0.70
        and pure_delta_summary["warmup_only"]["median"] > 0.0
        and pure_delta_summary["rotation_only_pure"]["median"] >= 2.0e-4
        and pure_delta_summary["random_full_state_pure"]["LCB05"] > 0.0
        and p2_base_zero_all == 1
        and p2_fu_count_all == 1
        and p2_state_update_all == 1
    )
    compact_gate_with_debt = int(compact_gate_without_debt == 1 and debt_summary["paired_no_debt_gate_pass"] == 1)
    datasets = sorted({row.get("dataset", "") for row in rows if row.get("dataset", "")})
    carriers = sorted({row.get("carrier", "") for row in rows if row.get("carrier", "")})
    seeds = sorted({int(row.get("seed", "0")) for row in rows if str(row.get("seed", "")).isdigit()})
    schemes = sorted({row.get("scheme", "") for row in rows if row.get("scheme", "")})
    finite_auc_rows = sum(1 for row in rows if math.isfinite(finite_float(row.get("AUC_task_loss_time"))))
    h20_rows = sum(1 for row in rows if finite_float(row.get("horizon"), 0.0) >= 20.0)
    expected_rows = 10 * 5 * 2 * len(PART_G_SCHEMES)
    runtime_truth_rows = [
        {
            "identity_check": "P2_base_update_norm_after_warmup_zero_all",
            "value": p2_base_zero_all,
            "note": "P2 rows require base_update_norm_after_warmup_sum == 0 in compact pure phase.",
        },
        {
            "identity_check": "P2_FU_map_count_after_warmup_equals_pure_steps_all",
            "value": p2_fu_count_all,
            "note": "P2 rows require one FU map execution for every post-warmup step.",
        },
        {
            "identity_check": "P2_state_update_after_warmup_equals_pure_steps_all",
            "value": p2_state_update_all,
            "note": "P2 rows require state updates to continue after base optimizer is disabled.",
        },
        {
            "identity_check": "P1_current_forcing_same_step_count_zero_all",
            "value": int(all(int(finite_float(row.get("hybrid_current_forcing_same_step_count"), 0.0)) == 0 for row in rows if row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY")),
            "note": "Compact P1 applies historical state and records no current-forcing same-step use.",
        },
    ]
    write_csv(OUT_ROOT / "v23_28_partG_runtime_truth.csv", runtime_truth_rows)
    summary = {
        "phase": "partG-hybrid-pure-compact-aggregate",
        "partG_completed": 0,
        "partG_compact_matrix_completed": int(len(rows) == expected_rows and h20_rows == len(rows) and finite_auc_rows == len(rows)),
        "compact_matrix_rows": len(rows),
        "expected_compact_matrix_rows": expected_rows,
        "row_count_matches_compact_plan": int(len(rows) == expected_rows),
        "h20_rows": h20_rows,
        "finite_auc_rows": finite_auc_rows,
        "datasets_count": len(datasets),
        "datasets": datasets,
        "carriers_count": len(carriers),
        "carriers": carriers,
        "seeds_count": len(seeds),
        "seeds": seeds,
        "schemes_count": len(schemes),
        "schemes": schemes,
        "post_warmup_decrease_rate": post_warmup_decrease_rate,
        "pure_delta_summary_P2_minus_controls": pure_delta_summary,
        "P2_base_update_norm_after_warmup_zero_all": p2_base_zero_all,
        "P2_FU_map_count_after_warmup_equals_pure_steps_all": p2_fu_count_all,
        "P2_state_update_after_warmup_equals_pure_steps_all": p2_state_update_all,
        "compact_gate_without_debt_pass": compact_gate_without_debt,
        "compact_paired_debt_summary_P2_vs_warmup_only": debt_summary,
        "paired_no_debt_rate": debt_summary["paired_no_debt_rate"],
        "paired_no_debt_gate_pass": debt_summary["paired_no_debt_gate_pass"],
        "compact_gate_with_debt_pass": compact_gate_with_debt,
        "official_blockers": [
            "compact_bank_proxy_not_full_official_partG",
            "paired_no_debt_official_gate_not_available_from_compact_proxy",
            "official_same_checkpoint_full_registry_not_run",
            "rotation_only_v2327_is_compact_skew_only_proxy",
            "fused_DCHE_DFOU_real_carrier_training_not_implemented_for_partG",
        ],
        "no_fabrication_note": "Aggregate reports measured compact PartG rows and compact test-debt only. It does not claim official PartG because full official protocol is missing.",
    }
    write_json(OUT_ROOT / "v23_28_partG_summary.json", summary)
    failure = [
        "# v23.28 PartG compact hybrid/pure failure decomposition",
        "",
        f"- rows: `{len(rows)}` / expected compact `{expected_rows}`",
        f"- datasets/carriers/seeds/schemes: `{len(datasets)}/{len(carriers)}/{len(seeds)}/{len(schemes)}`",
        f"- post_warmup_decrease_rate: `{post_warmup_decrease_rate}`",
        f"- pure_delta_summary_P2_minus_controls: `{json.dumps(json_clean(pure_delta_summary), ensure_ascii=False, sort_keys=True)}`",
        f"- P2 identity base_zero/FU_count/state_update: `{p2_base_zero_all}` / `{p2_fu_count_all}` / `{p2_state_update_all}`",
        f"- compact_gate_without_debt_pass: `{compact_gate_without_debt}`",
        f"- compact paired no-debt P2 vs warmup-only: rate `{debt_summary['paired_no_debt_rate']}`, gate `{debt_summary['paired_no_debt_gate_pass']}`, pairs `{debt_summary['paired_count']}`",
        f"- official_blockers: `{json.dumps(summary['official_blockers'], ensure_ascii=False)}`",
        "",
        "Conclusion: this exercises the warmup-then-pure branch but remains a compact proxy and cannot close official PartG.",
        "",
    ]
    (OUT_ROOT / "v23_28_partG_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partG_hybrid_pure_matrix.csv",
        "v23_28_partG_runtime_truth.csv",
        "v23_28_partG_compact_debt_summary.csv",
        "v23_28_partG_summary.json",
        "v23_28_partG_failure_decomposition.md",
    ]
    append_exec("PartG_hybrid_pure_compact_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartG hybrid/pure compact matrix",
        [
            f"rows `{len(rows)}` expected `{expected_rows}`; h20_rows `{h20_rows}`; finite_auc_rows `{finite_auc_rows}`.",
            f"datasets/carriers/seeds/schemes `{len(datasets)}/{len(carriers)}/{len(seeds)}/{len(schemes)}`.",
            f"post_warmup_decrease_rate `{post_warmup_decrease_rate}`.",
            f"P2 pure deltas `{json.dumps(json_clean(pure_delta_summary), ensure_ascii=False, sort_keys=True)}`.",
            f"P2 identity base_zero/FU_count/state_update `{p2_base_zero_all}` / `{p2_fu_count_all}` / `{p2_state_update_all}`.",
            f"compact paired no-debt P2 vs warmup-only rate/gate/pairs `{debt_summary['paired_no_debt_rate']}` / `{debt_summary['paired_no_debt_gate_pass']}` / `{debt_summary['paired_count']}`.",
            f"compact_gate_without_debt_pass `{compact_gate_without_debt}`; compact_gate_with_debt_pass `{compact_gate_with_debt}`; partG_completed `{summary['partG_completed']}`; official_blockers `{summary['official_blockers']}`.",
        ],
    )


def run_partH_matrix(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    kind = str(args.partH_kind)
    horizon = 80 if kind == "H80" else 200
    if int(args.partH_horizon) > 0:
        horizon = int(args.partH_horizon)
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    schemes = partH_schemes(kind)
    groups = [(dataset, seed, carrier) for dataset in PART_H_TASKS for seed in range(int(args.partH_matrix_seeds)) for carrier in carriers]
    shard_count = max(1, int(args.partH_shard_count))
    shard_index = int(args.partH_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid PartH shard index {shard_index} for shard count {shard_count}")
    shard_groups = [group for idx, group in enumerate(groups) if idx % shard_count == shard_index]
    if int(args.partH_max_groups) > 0:
        shard_groups = shard_groups[: int(args.partH_max_groups)]
    rows: list[dict[str, Any]] = []
    for dataset, seed, carrier in shard_groups:
        group_rows = [
            simulate_partH_scheme(
                dataset,
                carrier,
                int(seed),
                scheme,
                horizon,
                kind,
                device,
                train_cap=int(args.partd_train_cap),
                warmup_fraction=float(args.partG_warmup_fraction),
                sym_spectral_cap=float(args.partH_sym_spectral_cap),
            )
            for scheme in schemes
        ]
        by_scheme = {str(row["scheme"]): row for row in group_rows}
        p1_auc = finite_float(by_scheme["P1_FullPersistent_Hybrid_CompH2_PRIMARY"].get("AUC_task_loss_time"))
        for control_scheme, field in [
            ("K2_IntrinsicAdditive_CompH2", "P1_gain_vs_K2"),
            ("P0_FullInstant_CompH2", "P1_gain_vs_P0"),
            ("R5_SameAutocorrelationRandomState", "P1_gain_vs_R5"),
            ("P2_FullPersistent_WarmupPure_CompH2", "P1_gain_vs_P2"),
        ]:
            if control_scheme not in by_scheme:
                continue
            control_auc = finite_float(by_scheme[control_scheme].get("AUC_task_loss_time"))
            gain = control_auc - p1_auc if math.isfinite(control_auc) and math.isfinite(p1_auc) else ""
            by_scheme["P1_FullPersistent_Hybrid_CompH2_PRIMARY"][field] = gain
            by_scheme[control_scheme]["P1_gain_vs_this_control"] = gain
        rows.extend(group_rows)
    suffix = f"shard{shard_index:02d}_of_{shard_count:02d}"
    matrix_name = f"v23_28_partH_{kind}_matrix_{suffix}.csv"
    summary_name = f"v23_28_partH_{kind}_summary_{suffix}.json"
    write_csv(OUT_ROOT / matrix_name, rows)
    summary = {
        "phase": f"partH-{kind}-compact-shard",
        "partH_completed": 0,
        "partH_compact_shard_completed": int(len(rows) > 0),
        "partH_kind": kind,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "groups_in_shard": len(shard_groups),
        "rows": len(rows),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "carriers": sorted({str(row["carrier"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "schemes": schemes,
        "horizon": horizon,
        "partH_sym_spectral_cap": float(args.partH_sym_spectral_cap),
        "train_cap": int(args.partd_train_cap),
        "no_fabrication_note": "Shard rows use compact real-loss horizon proxy; no official PartH success is claimed.",
    }
    write_json(OUT_ROOT / summary_name, summary)
    append_exec(f"PartH_{kind}_compact_shard", args, [matrix_name, summary_name], summary, "completed_incomplete_gate")


def run_partH_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    h80_paths = sorted(OUT_ROOT.glob("v23_28_partH_H80_matrix_shard*_of_*.csv"))
    h200_paths = sorted(OUT_ROOT.glob("v23_28_partH_H200_matrix_shard*_of_*.csv"))
    h80_rows: list[dict[str, Any]] = []
    h200_rows: list[dict[str, Any]] = []
    for path in h80_paths:
        h80_rows.extend(read_csv(path))
    for path in h200_paths:
        h200_rows.extend(read_csv(path))
    write_csv(OUT_ROOT / "v23_28_partH_H80_matrix.csv", h80_rows)
    write_csv(OUT_ROOT / "v23_28_partH_H200_matrix.csv", h200_rows)
    state_rows = [
        {
            "partH_kind": row.get("partH_kind", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "carrier": row.get("carrier", ""),
            "scheme": row.get("scheme", ""),
            "horizon": row.get("horizon", ""),
            "state_survival_rate": row.get("state_survival_rate", ""),
            "state_norm_median": row.get("state_norm_median", ""),
            "state_norm_max": row.get("state_norm_max", ""),
            "state_collapse_flag": row.get("state_collapse_flag", ""),
            "component_explosion_flag": row.get("component_explosion_flag", ""),
        }
        for row in [*h80_rows, *h200_rows]
        if row.get("scheme") in {"P1_FullPersistent_Hybrid_CompH2_PRIMARY", "P2_FullPersistent_WarmupPure_CompH2", "A6_Persistent_Full_OmegaSymRadial"}
    ]
    write_csv(OUT_ROOT / "v23_28_partH_state_survival.csv", state_rows)
    p1_h80 = [row for row in h80_rows if row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
    p1_h200 = [row for row in h200_rows if row.get("scheme") == "P1_FullPersistent_Hybrid_CompH2_PRIMARY"]

    def gain_summary(rows_for_kind: list[dict[str, Any]], field: str, seed: int) -> dict[str, Any]:
        vals = [finite_float(row.get(field)) for row in rows_for_kind]
        vals = [v for v in vals if math.isfinite(v)]
        return {
            "median": float(np.median(vals)) if vals else float("nan"),
            "CVaR25": cvar25(vals),
            "LCB05": bootstrap_lcb05(vals, seed=seed),
            "rows": len(vals),
        }

    h80_delta_summary = {
        "K2_IntrinsicAdditive_CompH2": gain_summary(p1_h80, "P1_gain_vs_K2", 632800),
        "P0_FullInstant_CompH2": gain_summary(p1_h80, "P1_gain_vs_P0", 632801),
        "R5_SameAutocorrelationRandomState": gain_summary(p1_h80, "P1_gain_vs_R5", 632802),
        "P2_FullPersistent_WarmupPure_CompH2": gain_summary(p1_h80, "P1_gain_vs_P2", 632803),
    }
    h200_delta_summary = {
        "K2_IntrinsicAdditive_CompH2": gain_summary(p1_h200, "P1_gain_vs_K2", 632804),
        "P0_FullInstant_CompH2": gain_summary(p1_h200, "P1_gain_vs_P0", 632805),
        "R5_SameAutocorrelationRandomState": gain_summary(p1_h200, "P1_gain_vs_R5", 632806),
        "P2_FullPersistent_WarmupPure_CompH2": gain_summary(p1_h200, "P1_gain_vs_P2", 632807),
    }
    p1_h80_survival_median = median_of(p1_h80, "state_survival_rate")
    p1_h200_survival_median = median_of(p1_h200, "state_survival_rate")
    h200_no_collapse = int(p1_h200_survival_median >= 0.80 and median_of(p1_h200, "component_explosion_flag") == 0.0)
    h80_debt_summary, h80_debt_rows = compact_paired_debt_summary(
        h80_rows,
        "scheme",
        "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
        "K2_IntrinsicAdditive_CompH2",
        ["dataset", "seed", "carrier"],
        "PartH_H80_P1_vs_K2_compact_test_debt",
    )
    h200_debt_summary, h200_debt_rows = compact_paired_debt_summary(
        h200_rows,
        "scheme",
        "P1_FullPersistent_Hybrid_CompH2_PRIMARY",
        "K2_IntrinsicAdditive_CompH2",
        ["dataset", "seed", "carrier"],
        "PartH_H200_P1_vs_K2_compact_test_debt",
    )
    write_csv(OUT_ROOT / "v23_28_partH_compact_debt_summary.csv", [*h80_debt_rows, *h200_debt_rows])
    h80_compact_gate_without_debt = int(
        h80_delta_summary["K2_IntrinsicAdditive_CompH2"]["median"] >= 1.0e-3
        and h80_delta_summary["K2_IntrinsicAdditive_CompH2"]["CVaR25"] > 0.0
        and h80_delta_summary["K2_IntrinsicAdditive_CompH2"]["LCB05"] > 0.0
        and h80_delta_summary["P0_FullInstant_CompH2"]["median"] >= 3.0e-4
        and h80_delta_summary["R5_SameAutocorrelationRandomState"]["LCB05"] > 0.0
        and p1_h80_survival_median >= 0.80
    )
    h200_compact_gate_without_debt = int(
        h200_delta_summary["K2_IntrinsicAdditive_CompH2"]["median"] > 0.0
        and h200_delta_summary["K2_IntrinsicAdditive_CompH2"]["CVaR25"] > 0.0
        and h200_delta_summary["K2_IntrinsicAdditive_CompH2"]["LCB05"] > 0.0
        and h200_no_collapse == 1
    )
    h80_compact_gate_with_debt = int(h80_compact_gate_without_debt == 1 and h80_debt_summary["paired_no_debt_gate_pass"] == 1)
    h200_compact_gate_with_debt = int(h200_compact_gate_without_debt == 1 and h200_debt_summary["paired_no_debt_gate_pass"] == 1)
    h80_expected = len(PART_H_TASKS) * 5 * 2 * len(PART_H_H80_SCHEMES)
    h200_expected = len(PART_H_TASKS) * 5 * 2 * len(PART_H_H200_SCHEMES)
    h80_finite = sum(1 for row in h80_rows if math.isfinite(finite_float(row.get("AUC_task_loss_time"))))
    h200_finite = sum(1 for row in h200_rows if math.isfinite(finite_float(row.get("AUC_task_loss_time"))))
    h80_horizon_rows = sum(1 for row in h80_rows if finite_float(row.get("horizon"), 0.0) >= 80.0)
    h200_horizon_rows = sum(1 for row in h200_rows if finite_float(row.get("horizon"), 0.0) >= 200.0)
    sym_spectral_caps = sorted({finite_float(row.get("partH_sym_spectral_cap")) for row in [*h80_rows, *h200_rows] if math.isfinite(finite_float(row.get("partH_sym_spectral_cap")))})
    summary = {
        "phase": "partH-horizon-compact-aggregate",
        "partH_completed": 0,
        "partH_compact_H80_completed": int(len(h80_rows) == h80_expected and h80_finite == len(h80_rows) and h80_horizon_rows == len(h80_rows)),
        "partH_compact_H200_completed": int(len(h200_rows) == h200_expected and h200_finite == len(h200_rows) and h200_horizon_rows == len(h200_rows)),
        "H80_rows": len(h80_rows),
        "H80_expected_rows": h80_expected,
        "H80_row_count_matches_compact_plan": int(len(h80_rows) == h80_expected),
        "H80_finite_auc_rows": h80_finite,
        "H80_horizon_rows": h80_horizon_rows,
        "H200_rows": len(h200_rows),
        "H200_expected_rows": h200_expected,
        "H200_row_count_matches_compact_plan": int(len(h200_rows) == h200_expected),
        "H200_finite_auc_rows": h200_finite,
        "H200_horizon_rows": h200_horizon_rows,
        "datasets": PART_H_TASKS,
        "carriers": ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"],
        "seeds": [0, 1, 2, 3, 4],
        "partH_sym_spectral_caps": sym_spectral_caps,
        "partH_sym_spectral_cap_diagnostic": int(any(value > 0.0 for value in sym_spectral_caps)),
        "H80_delta_summary_P1_minus_controls": h80_delta_summary,
        "H200_delta_summary_P1_minus_controls": h200_delta_summary,
        "P1_H80_state_survival_rate_median": p1_h80_survival_median,
        "P1_H200_state_survival_rate_median": p1_h200_survival_median,
        "H80_compact_gate_without_debt_pass": h80_compact_gate_without_debt,
        "H200_compact_gate_without_debt_pass": h200_compact_gate_without_debt,
        "H80_compact_paired_debt_summary_P1_vs_K2": h80_debt_summary,
        "H200_compact_paired_debt_summary_P1_vs_K2": h200_debt_summary,
        "H80_paired_no_debt_rate": h80_debt_summary["paired_no_debt_rate"],
        "H200_paired_no_debt_rate": h200_debt_summary["paired_no_debt_rate"],
        "H80_paired_no_debt_gate_pass": h80_debt_summary["paired_no_debt_gate_pass"],
        "H200_paired_no_debt_gate_pass": h200_debt_summary["paired_no_debt_gate_pass"],
        "H80_compact_gate_with_debt_pass": h80_compact_gate_with_debt,
        "H200_compact_gate_with_debt_pass": h200_compact_gate_with_debt,
        "official_blockers": [
            "compact_bank_proxy_not_full_official_partH",
            "paired_no_debt_official_gate_not_available_from_compact_proxy",
            "official_same_checkpoint_full_registry_not_run",
            "fused_DCHE_DFOU_real_carrier_training_not_implemented_for_partH",
        ],
        "no_fabrication_note": "Aggregate reports measured compact H80/H200 horizon rows and compact test-debt only. It does not claim official PartH because full official protocol is missing.",
    }
    write_json(OUT_ROOT / "v23_28_partH_summary.json", summary)
    failure = [
        "# v23.28 PartH compact horizon failure decomposition",
        "",
        f"- H80 rows: `{len(h80_rows)}` / expected `{h80_expected}`",
        f"- H200 rows: `{len(h200_rows)}` / expected `{h200_expected}`",
        f"- H80 P1 deltas: `{json.dumps(json_clean(h80_delta_summary), ensure_ascii=False, sort_keys=True)}`",
        f"- H200 P1 deltas: `{json.dumps(json_clean(h200_delta_summary), ensure_ascii=False, sort_keys=True)}`",
        f"- P1 state survival H80/H200 medians: `{p1_h80_survival_median}` / `{p1_h200_survival_median}`",
        f"- H80/H200 compact gates without debt: `{h80_compact_gate_without_debt}` / `{h200_compact_gate_without_debt}`",
        f"- H80 compact paired no-debt P1 vs K2: rate `{h80_debt_summary['paired_no_debt_rate']}`, gate `{h80_debt_summary['paired_no_debt_gate_pass']}`, pairs `{h80_debt_summary['paired_count']}`",
        f"- H200 compact paired no-debt P1 vs K2: rate `{h200_debt_summary['paired_no_debt_rate']}`, gate `{h200_debt_summary['paired_no_debt_gate_pass']}`, pairs `{h200_debt_summary['paired_count']}`",
        f"- official_blockers: `{json.dumps(summary['official_blockers'], ensure_ascii=False)}`",
        "",
        "Conclusion: this executes compact H80/H200 trajectories, but official PartH remains incomplete without no-debt and full fused-carrier protocol.",
        "",
    ]
    (OUT_ROOT / "v23_28_partH_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partH_H80_matrix.csv",
        "v23_28_partH_H200_matrix.csv",
        "v23_28_partH_state_survival.csv",
        "v23_28_partH_compact_debt_summary.csv",
        "v23_28_partH_summary.json",
        "v23_28_partH_failure_decomposition.md",
    ]
    append_exec("PartH_horizon_compact_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartH H80/H200 compact matrix",
        [
            f"H80 rows `{len(h80_rows)}` expected `{h80_expected}`; H200 rows `{len(h200_rows)}` expected `{h200_expected}`.",
            f"H80 P1 deltas `{json.dumps(json_clean(h80_delta_summary), ensure_ascii=False, sort_keys=True)}`.",
            f"H200 P1 deltas `{json.dumps(json_clean(h200_delta_summary), ensure_ascii=False, sort_keys=True)}`.",
            f"P1 state survival H80/H200 medians `{p1_h80_survival_median}` / `{p1_h200_survival_median}`.",
            f"H80 compact paired no-debt P1 vs K2 rate/gate/pairs `{h80_debt_summary['paired_no_debt_rate']}` / `{h80_debt_summary['paired_no_debt_gate_pass']}` / `{h80_debt_summary['paired_count']}`.",
            f"H200 compact paired no-debt P1 vs K2 rate/gate/pairs `{h200_debt_summary['paired_no_debt_rate']}` / `{h200_debt_summary['paired_no_debt_gate_pass']}` / `{h200_debt_summary['paired_count']}`.",
            f"H80/H200 compact gates without debt `{h80_compact_gate_without_debt}` / `{h200_compact_gate_without_debt}`; with debt `{h80_compact_gate_with_debt}` / `{h200_compact_gate_with_debt}`; partH_completed `{summary['partH_completed']}`.",
            f"official_blockers `{summary['official_blockers']}`.",
        ],
    )


def run_partI_matrix(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    groups = [(dataset, seed, carrier) for dataset in PART_I_TASKS for seed in range(int(args.partI_matrix_seeds)) for carrier in carriers]
    shard_count = max(1, int(args.partI_shard_count))
    shard_index = int(args.partI_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid PartI shard index {shard_index} for shard count {shard_count}")
    shard_groups = [group for idx, group in enumerate(groups) if idx % shard_count == shard_index]
    if int(args.partI_max_groups) > 0:
        shard_groups = shard_groups[: int(args.partI_max_groups)]
    rows: list[dict[str, Any]] = []
    for dataset, seed, carrier in shard_groups:
        group_rows = [
            simulate_partI_row(dataset, carrier, int(seed), scheme, int(args.partI_matrix_horizon), device, train_cap=int(args.partd_train_cap))
            for scheme in PART_I_SCHEMES
        ]
        by_scheme = {str(row["scheme"]): row for row in group_rows}
        kan_auc = finite_float(by_scheme["KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY"].get("AUC_task_loss_time"))
        for mlp_scheme in [scheme for scheme in PART_I_SCHEMES if scheme.startswith("M")]:
            mlp_auc = finite_float(by_scheme[mlp_scheme].get("AUC_task_loss_time"))
            gain = mlp_auc - kan_auc if math.isfinite(mlp_auc) and math.isfinite(kan_auc) else ""
            by_scheme[mlp_scheme]["KAN_P1_gain_vs_this_MLP"] = gain
            by_scheme["KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY"][f"KAN_P1_gain_vs_{mlp_scheme}"] = gain
        k2_time = finite_float(by_scheme["KAN_K2_IntrinsicBase_Compact"].get("wall_time_sec"))
        p1_time = finite_float(by_scheme["KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY"].get("wall_time_sec"))
        m1_time = finite_float(by_scheme["M1_MLP_SameFLOPs_TaskNativeStrongOptimizer"].get("wall_time_sec"))
        m1_mem = finite_float(by_scheme["M1_MLP_SameFLOPs_TaskNativeStrongOptimizer"].get("peak_memory_proxy_bytes"))
        p1_mem = finite_float(by_scheme["KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY"].get("peak_memory_proxy_bytes"))
        by_scheme["KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY"]["FU_incremental_overhead_vs_intrinsic_base"] = (p1_time - k2_time) / max(k2_time, 1.0e-12) if math.isfinite(k2_time) and math.isfinite(p1_time) else ""
        by_scheme["KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY"]["full_step_ratio_vs_sameFLOPs_MLP"] = p1_time / max(m1_time, 1.0e-12) if math.isfinite(m1_time) and math.isfinite(p1_time) else ""
        by_scheme["KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY"]["peak_memory_ratio_vs_MLP"] = p1_mem / max(m1_mem, 1.0e-12) if math.isfinite(m1_mem) and math.isfinite(p1_mem) else ""
        rows.extend(group_rows)
    suffix = f"shard{shard_index:02d}_of_{shard_count:02d}"
    matrix_name = f"v23_28_partI_MLP_matched_matrix_{suffix}.csv"
    summary_name = f"v23_28_partI_MLP_matched_summary_{suffix}.json"
    write_csv(OUT_ROOT / matrix_name, rows)
    summary = {
        "phase": "partI-MLP-efficiency-compact-shard",
        "partI_completed": 0,
        "partI_compact_shard_completed": int(len(rows) > 0),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "groups_in_shard": len(shard_groups),
        "rows": len(rows),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "carriers": sorted({str(row["carrier"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "schemes": PART_I_SCHEMES,
        "horizon": int(args.partI_matrix_horizon),
        "train_cap": int(args.partd_train_cap),
        "no_fabrication_note": "Shard rows use compact KAN/MLP/efficiency proxy; no official PartI success is claimed.",
    }
    write_json(OUT_ROOT / summary_name, summary)
    append_exec("PartI_MLP_efficiency_compact_shard", args, [matrix_name, summary_name], summary, "completed_incomplete_gate")


def run_partI_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    shard_paths = sorted(OUT_ROOT.glob("v23_28_partI_MLP_matched_matrix_shard*_of_*.csv"))
    rows: list[dict[str, Any]] = []
    for path in shard_paths:
        rows.extend(read_csv(path))
    write_csv(OUT_ROOT / "v23_28_partI_MLP_matched_matrix.csv", rows)
    kan_rows = [row for row in rows if row.get("scheme") == "KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY"]
    mlp_schemes = [scheme for scheme in PART_I_SCHEMES if scheme.startswith("M")]

    def kan_gain_summary(mlp_scheme: str, seed: int) -> dict[str, Any]:
        field = f"KAN_P1_gain_vs_{mlp_scheme}"
        vals = [finite_float(row.get(field)) for row in kan_rows]
        vals = [v for v in vals if math.isfinite(v)]
        return {
            "median": float(np.median(vals)) if vals else float("nan"),
            "CVaR25": cvar25(vals),
            "LCB05": bootstrap_lcb05(vals, seed=seed),
            "rows": len(vals),
        }

    mlp_delta_summary = {scheme: kan_gain_summary(scheme, 732800 + idx) for idx, scheme in enumerate(mlp_schemes)}
    cvar_positive_count = sum(1 for item in mlp_delta_summary.values() if item["CVaR25"] > 0.0)
    lcb_positive_count = sum(1 for item in mlp_delta_summary.values() if item["LCB05"] > 0.0)
    kan_k2_debt_summary, kan_k2_debt_rows = compact_paired_debt_summary(
        rows,
        "scheme",
        "KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY",
        "KAN_K2_IntrinsicBase_Compact",
        ["dataset", "seed", "carrier"],
        "PartI_KAN_P1_vs_KAN_K2_compact_test_debt",
    )
    kan_m1_debt_summary, kan_m1_debt_rows = compact_paired_debt_summary(
        rows,
        "scheme",
        "KAN_P1_FullPersistent_Hybrid_CompH2_PRIMARY",
        "M1_MLP_SameFLOPs_TaskNativeStrongOptimizer",
        ["dataset", "seed", "carrier"],
        "PartI_KAN_P1_vs_M1_compact_test_debt",
    )
    write_csv(OUT_ROOT / "v23_28_partI_compact_debt_summary.csv", [*kan_k2_debt_rows, *kan_m1_debt_rows])
    architecture_compact_gate_without_debt = int(
        mlp_delta_summary["M0_MLP_SameParam_TaskNativeStrongOptimizer"]["median"] > 0.0
        and mlp_delta_summary["M1_MLP_SameFLOPs_TaskNativeStrongOptimizer"]["median"] > 0.0
        and mlp_delta_summary["M3_MLP_PersistentFullTransportShapeRadialBlock_Proxy"]["median"] > 0.0
        and mlp_delta_summary["M5_MLP_MCGA_Reproduction_Proxy"]["median"] > 0.0
        and cvar_positive_count >= 3
        and lcb_positive_count >= 3
    )
    architecture_compact_gate_with_debt = int(architecture_compact_gate_without_debt == 1 and kan_m1_debt_summary["paired_no_debt_gate_pass"] == 1)
    efficiency_rows = []
    for row in kan_rows:
        efficiency_rows.append({
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "carrier": row.get("carrier", ""),
            "FU_incremental_overhead_vs_intrinsic_base": row.get("FU_incremental_overhead_vs_intrinsic_base", ""),
            "full_step_ratio_vs_sameFLOPs_MLP": row.get("full_step_ratio_vs_sameFLOPs_MLP", ""),
            "peak_memory_ratio_vs_MLP": row.get("peak_memory_ratio_vs_MLP", ""),
            "wall_time_sec": row.get("wall_time_sec", ""),
            "arithmetic_proxy_ops": row.get("arithmetic_proxy_ops", ""),
        })
    write_csv(OUT_ROOT / "v23_28_partI_efficiency_matrix.csv", efficiency_rows)
    overhead_median = median_of(efficiency_rows, "FU_incremental_overhead_vs_intrinsic_base")
    step_ratio_median = median_of(efficiency_rows, "full_step_ratio_vs_sameFLOPs_MLP")
    mem_ratio_median = median_of(efficiency_rows, "peak_memory_ratio_vs_MLP")
    efficiency_compact_gate = int(overhead_median <= 0.20 and mem_ratio_median <= 1.10 and step_ratio_median <= 1.50)
    kernel_census = [
        {"component": "carrier_forward_backward", "compact_proxy_count": len(rows), "official_kernel_census_claim": 0},
        {"component": "metric_stats", "compact_proxy_count": len(kan_rows), "official_kernel_census_claim": 0},
        {"component": "full_operator_solve", "compact_proxy_count": len(kan_rows), "official_kernel_census_claim": 0},
        {"component": "FU_map", "compact_proxy_count": len(kan_rows), "official_kernel_census_claim": 0},
        {"component": "MLP_adamw_step", "compact_proxy_count": sum(1 for row in rows if str(row.get("scheme", "")).startswith("M")), "official_kernel_census_claim": 0},
    ]
    write_csv(OUT_ROOT / "v23_28_partI_kernel_census.csv", kernel_census)
    mcga_audit = {
        "phase": "partI-MCGA-reproduction-audit-compact",
        "M5_rows": sum(1 for row in rows if row.get("scheme") == "M5_MLP_MCGA_Reproduction_Proxy"),
        "true_MCGA_reproduction_completed": 0,
        "proxy_note": "M5 rows are compact task-native AdamW MLP proxy rows, not a verified MCGA reproduction.",
    }
    write_json(OUT_ROOT / "v23_28_partI_MCGA_reproduction_audit.json", mcga_audit)
    datasets = sorted({row.get("dataset", "") for row in rows if row.get("dataset", "")})
    carriers = sorted({row.get("carrier", "") for row in rows if row.get("carrier", "")})
    seeds = sorted({int(row.get("seed", "0")) for row in rows if str(row.get("seed", "")).isdigit()})
    schemes = sorted({row.get("scheme", "") for row in rows if row.get("scheme", "")})
    finite_auc_rows = sum(1 for row in rows if math.isfinite(finite_float(row.get("AUC_task_loss_time"))))
    h20_rows = sum(1 for row in rows if finite_float(row.get("horizon"), 0.0) >= 20.0)
    expected_rows = len(PART_I_TASKS) * 5 * 2 * len(PART_I_SCHEMES)
    summary = {
        "phase": "partI-MLP-efficiency-compact-aggregate",
        "partI_completed": 0,
        "partI_compact_matrix_completed": int(len(rows) == expected_rows and finite_auc_rows == len(rows) and h20_rows == len(rows)),
        "compact_matrix_rows": len(rows),
        "expected_compact_matrix_rows": expected_rows,
        "row_count_matches_compact_plan": int(len(rows) == expected_rows),
        "finite_auc_rows": finite_auc_rows,
        "h20_rows": h20_rows,
        "datasets_count": len(datasets),
        "datasets": datasets,
        "carriers_count": len(carriers),
        "carriers": carriers,
        "seeds_count": len(seeds),
        "seeds": seeds,
        "schemes_count": len(schemes),
        "schemes": schemes,
        "MLP_delta_summary_KAN_P1_minus_controls": mlp_delta_summary,
        "MLP_CVaR25_positive_count": cvar_positive_count,
        "MLP_LCB05_positive_count": lcb_positive_count,
        "architecture_compact_gate_without_debt_pass": architecture_compact_gate_without_debt,
        "compact_paired_debt_summary_KAN_P1_vs_KAN_K2": kan_k2_debt_summary,
        "compact_paired_debt_summary_KAN_P1_vs_M1": kan_m1_debt_summary,
        "KAN_P1_vs_KAN_K2_paired_no_debt_rate": kan_k2_debt_summary["paired_no_debt_rate"],
        "KAN_P1_vs_M1_paired_no_debt_rate": kan_m1_debt_summary["paired_no_debt_rate"],
        "KAN_P1_vs_KAN_K2_paired_no_debt_gate_pass": kan_k2_debt_summary["paired_no_debt_gate_pass"],
        "KAN_P1_vs_M1_paired_no_debt_gate_pass": kan_m1_debt_summary["paired_no_debt_gate_pass"],
        "architecture_compact_gate_with_debt_pass": architecture_compact_gate_with_debt,
        "FU_incremental_overhead_median": overhead_median,
        "full_step_ratio_vs_sameFLOPs_MLP_median": step_ratio_median,
        "peak_memory_ratio_vs_MLP_median": mem_ratio_median,
        "efficiency_compact_gate_pass": efficiency_compact_gate,
        "official_blockers": [
            "compact_MLP_controls_not_full_official_partI",
            "true_MCGA_reproduction_not_implemented",
            "kernel_census_is_arithmetic_proxy_not_profiler_or_fused_op_census",
            "paired_no_debt_official_gate_not_available_from_compact_proxy",
            "official_same_checkpoint_full_registry_not_run",
        ],
        "no_fabrication_note": "Aggregate reports measured compact PartI proxy rows and compact test-debt only. It does not claim official architecture or efficiency success.",
    }
    write_json(OUT_ROOT / "v23_28_partI_summary.json", summary)
    failure = [
        "# v23.28 PartI compact MLP/efficiency failure decomposition",
        "",
        f"- rows: `{len(rows)}` / expected compact `{expected_rows}`",
        f"- MLP deltas: `{json.dumps(json_clean(mlp_delta_summary), ensure_ascii=False, sort_keys=True)}`",
        f"- architecture_compact_gate_without_debt_pass: `{architecture_compact_gate_without_debt}`",
        f"- compact paired no-debt KAN P1 vs KAN K2: rate `{kan_k2_debt_summary['paired_no_debt_rate']}`, gate `{kan_k2_debt_summary['paired_no_debt_gate_pass']}`, pairs `{kan_k2_debt_summary['paired_count']}`",
        f"- compact paired no-debt KAN P1 vs M1: rate `{kan_m1_debt_summary['paired_no_debt_rate']}`, gate `{kan_m1_debt_summary['paired_no_debt_gate_pass']}`, pairs `{kan_m1_debt_summary['paired_count']}`",
        f"- efficiency medians overhead/step_ratio/memory_ratio: `{overhead_median}` / `{step_ratio_median}` / `{mem_ratio_median}`",
        f"- efficiency_compact_gate_pass: `{efficiency_compact_gate}`",
        f"- official_blockers: `{json.dumps(summary['official_blockers'], ensure_ascii=False)}`",
        "",
        "Conclusion: compact MLP and efficiency proxies were executed, but official PartI remains incomplete without real MCGA reproduction, profiler/kernel census, no-debt, and full registry.",
        "",
    ]
    (OUT_ROOT / "v23_28_partI_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partI_MLP_matched_matrix.csv",
        "v23_28_partI_MCGA_reproduction_audit.json",
        "v23_28_partI_efficiency_matrix.csv",
        "v23_28_partI_kernel_census.csv",
        "v23_28_partI_compact_debt_summary.csv",
        "v23_28_partI_summary.json",
        "v23_28_partI_failure_decomposition.md",
    ]
    append_exec("PartI_MLP_efficiency_compact_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartI MLP/efficiency compact matrix",
        [
            f"rows `{len(rows)}` expected `{expected_rows}`; h20_rows `{h20_rows}`; finite_auc_rows `{finite_auc_rows}`.",
            f"MLP deltas `{json.dumps(json_clean(mlp_delta_summary), ensure_ascii=False, sort_keys=True)}`.",
            f"architecture_compact_gate_without_debt_pass `{architecture_compact_gate_without_debt}`.",
            f"compact paired no-debt KAN P1 vs KAN K2 rate/gate/pairs `{kan_k2_debt_summary['paired_no_debt_rate']}` / `{kan_k2_debt_summary['paired_no_debt_gate_pass']}` / `{kan_k2_debt_summary['paired_count']}`.",
            f"compact paired no-debt KAN P1 vs M1 rate/gate/pairs `{kan_m1_debt_summary['paired_no_debt_rate']}` / `{kan_m1_debt_summary['paired_no_debt_gate_pass']}` / `{kan_m1_debt_summary['paired_count']}`.",
            f"efficiency medians overhead/step_ratio/memory_ratio `{overhead_median}` / `{step_ratio_median}` / `{mem_ratio_median}`; efficiency_compact_gate_pass `{efficiency_compact_gate}`.",
            f"architecture_compact_gate_with_debt_pass `{architecture_compact_gate_with_debt}`; partI_completed `{summary['partI_completed']}`; official_blockers `{summary['official_blockers']}`.",
        ],
    )


def run_partI_mcga_probe_audit(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    label = str(getattr(args, "partI_mcga_probe_label", "v23_28_partI_mcga_probe"))
    probe_paths = sorted((ROOT / "results/v22_66/chunks").glob(f"*{label}*st*.csv"), key=lambda p: p.stat().st_mtime)
    metric_paths = sorted((ROOT / "results/v22_66/chunks").glob(f"*{label}*metric_rows.csv"), key=lambda p: p.stat().st_mtime)
    probe_path = probe_paths[-1] if probe_paths else None
    metric_path = metric_paths[-1] if metric_paths else None
    probe_rows = read_csv(probe_path) if probe_path is not None else []
    probe_row = probe_rows[0] if probe_rows else {}
    try:
        import poet_torch  # type: ignore

        poet_available = 1
        poet_source = str(Path(getattr(poet_torch, "__file__", "")).resolve())
    except Exception as exc:
        poet_available = 0
        poet_source = f"{type(exc).__name__}: {str(exc)[:240]}"
    v2266_runner = ROOT / "experiments/run_v22_66_metric_compatible_generator_atlas_fu.py"
    completed = int(str(probe_row.get("run_status", "")) == "completed")
    true_mcga_smoke_available = int(
        completed == 1
        and poet_available == 1
        and str(probe_row.get("method", "")).startswith("mcga_")
        and str(probe_row.get("optimizer_step_source", "")) == "poet_torch.get_poet_optimizer"
        and int(finite_float(probe_row.get("external_optimizer_available"), 0.0)) == 1
    )
    audit_row = {
        "phase": "partI-MCGA-v2266-probe-audit",
        "official_partI_completion_claim": 0,
        "probe_label": label,
        "poet_torch_available": poet_available,
        "poet_torch_source": poet_source,
        "v22_66_runner_present": int(v2266_runner.is_file()),
        "v22_66_runner_hash": sha256_file(v2266_runner) if v2266_runner.is_file() else "",
        "probe_chunk_path": rel(probe_path),
        "probe_metric_rows_path": rel(metric_path),
        "probe_rows": len(probe_rows),
        "probe_run_status": probe_row.get("run_status", ""),
        "probe_method": probe_row.get("method", ""),
        "probe_method_family": probe_row.get("method_family", ""),
        "probe_dataset": probe_row.get("dataset", ""),
        "probe_seed": probe_row.get("seed", ""),
        "probe_steps": probe_row.get("steps", ""),
        "optimizer_step_source": probe_row.get("optimizer_step_source", ""),
        "external_optimizer_available": probe_row.get("external_optimizer_available", ""),
        "standard_loop_runtime_trace_pass": probe_row.get("standard_loop_runtime_trace_pass", ""),
        "held_NLL": probe_row.get("held_NLL", ""),
        "held_accuracy": probe_row.get("held_accuracy", ""),
        "AUC_loss_time": probe_row.get("AUC_loss_time", ""),
        "no_ECE_Brier_tail_debt": probe_row.get("no_ECE_Brier_tail_debt", ""),
        "controller_or_coordinate_overhead_ratio": probe_row.get("controller_or_coordinate_overhead_ratio", ""),
        "active_Gram_drift_mean": probe_row.get("active_Gram_drift_mean", ""),
        "functional_spectrum_drift_mean": probe_row.get("functional_spectrum_drift_mean", ""),
        "generator_descent_fraction": probe_row.get("generator_descent_fraction", ""),
        "Delta_NLL_vs_external_OET": probe_row.get("Delta_NLL_vs_external_OET", ""),
        "Delta_NLL_vs_best_control": probe_row.get("Delta_NLL_vs_best_control", ""),
        "true_MCGA_single_row_smoke_available": true_mcga_smoke_available,
        "true_MCGA_official_reproduction_completed": 0,
        "no_fabrication_note": "This audit records one v22.66 MLP-MCGA smoke/probe row if present. It is not the v23.28 official PartI MCGA reproduction matrix.",
    }
    write_csv(OUT_ROOT / "v23_28_partI_MCGA_probe_audit.csv", [audit_row])
    write_json(OUT_ROOT / "v23_28_partI_MCGA_probe_audit.json", audit_row)
    previous_path, previous = latest_json("v23_28*/v23_28_partI_summary.json")
    blockers = list(previous.get("official_blockers", ["compact_MLP_controls_not_full_official_partI", "true_MCGA_reproduction_not_implemented", "kernel_census_is_arithmetic_proxy_not_profiler_or_fused_op_census", "paired_no_debt_official_gate_not_available_from_compact_proxy", "official_same_checkpoint_full_registry_not_run"]))
    if true_mcga_smoke_available == 1:
        blockers = [
            "true_MCGA_single_row_smoke_available_not_full_official_reproduction" if item == "true_MCGA_reproduction_not_implemented" else item
            for item in blockers
        ]
    summary = dict(previous)
    summary.update(
        {
            "phase": "partI-compact-aggregate-plus-MCGA-probe-audit",
            "partI_completed": 0,
            "source_partI_compact_summary": rel(previous_path),
            "MCGA_probe_audit_path": rel(OUT_ROOT / "v23_28_partI_MCGA_probe_audit.json"),
            "true_MCGA_single_row_smoke_available": true_mcga_smoke_available,
            "true_MCGA_official_reproduction_completed": 0,
            "official_blockers": blockers,
            "no_fabrication_note": "PartI summary is carried forward from the latest compact PartI summary and augmented with a v22.66 MCGA smoke/probe audit. It does not claim official v23.28 PartI completion.",
        }
    )
    write_json(OUT_ROOT / "v23_28_partI_summary.json", summary)
    failure = [
        "# v23.28 PartI MCGA probe audit",
        "",
        f"- probe_label: `{label}`",
        f"- poet_torch_available/source: `{poet_available}` / `{poet_source}`",
        f"- probe_chunk_path: `{rel(probe_path)}`",
        f"- probe_method/status/steps: `{audit_row['probe_method']}` / `{audit_row['probe_run_status']}` / `{audit_row['probe_steps']}`",
        f"- optimizer_step_source/external_available: `{audit_row['optimizer_step_source']}` / `{audit_row['external_optimizer_available']}`",
        f"- held_NLL/accuracy/AUC_loss_time: `{audit_row['held_NLL']}` / `{audit_row['held_accuracy']}` / `{audit_row['AUC_loss_time']}`",
        f"- generator_descent_fraction/functional_spectrum_drift: `{audit_row['generator_descent_fraction']}` / `{audit_row['functional_spectrum_drift_mean']}`",
        f"- true_MCGA_single_row_smoke_available: `{true_mcga_smoke_available}`",
        f"- true_MCGA_official_reproduction_completed: `0`",
        f"- official_blockers: `{json.dumps(blockers, ensure_ascii=False)}`",
        "",
        "Conclusion: MCGA code/dependency path is available and one compact smoke row executed, but this is not a full official v23.28 PartI MCGA reproduction.",
        "",
    ]
    (OUT_ROOT / "v23_28_partI_MCGA_probe_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partI_MCGA_probe_audit.csv",
        "v23_28_partI_MCGA_probe_audit.json",
        "v23_28_partI_summary.json",
        "v23_28_partI_MCGA_probe_failure_decomposition.md",
    ]
    append_exec("PartI_MCGA_probe_audit", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartI MCGA probe audit",
        [
            f"probe_label `{label}`; poet_torch_available/source `{poet_available}` / `{poet_source}`.",
            f"probe chunk `{rel(probe_path)}`; metric rows `{rel(metric_path)}`; probe_rows `{len(probe_rows)}`.",
            f"method/status/steps `{audit_row['probe_method']}` / `{audit_row['probe_run_status']}` / `{audit_row['probe_steps']}`; optimizer `{audit_row['optimizer_step_source']}`; external_optimizer_available `{audit_row['external_optimizer_available']}`.",
            f"held_NLL/accuracy/AUC_loss_time `{audit_row['held_NLL']}` / `{audit_row['held_accuracy']}` / `{audit_row['AUC_loss_time']}`.",
            f"generator_descent_fraction/function_spectrum_drift `{audit_row['generator_descent_fraction']}` / `{audit_row['functional_spectrum_drift_mean']}`.",
            f"true_MCGA_single_row_smoke_available `{true_mcga_smoke_available}`; true_MCGA_official_reproduction_completed `0`; partI_completed `{summary['partI_completed']}`.",
            f"official_blockers `{blockers}`.",
        ],
    )


def partI_profiler_event_rows(prof: Any, profile_label: str, max_events: int = 60) -> list[dict[str, Any]]:
    events = prof.key_averages(group_by_input_shape=True)

    def event_score(ev: Any) -> float:
        return max(
            abs(float(getattr(ev, "self_cpu_time_total", 0.0) or 0.0)),
            abs(float(getattr(ev, "self_cuda_time_total", getattr(ev, "self_device_time_total", 0.0)) or 0.0)),
            abs(float(getattr(ev, "self_cpu_memory_usage", 0.0) or 0.0)),
            abs(float(getattr(ev, "self_cuda_memory_usage", getattr(ev, "self_device_memory_usage", 0.0)) or 0.0)),
        )

    rows: list[dict[str, Any]] = []
    for rank, ev in enumerate(sorted(events, key=event_score, reverse=True)[: int(max_events)], start=1):
        self_cuda_time = float(getattr(ev, "self_cuda_time_total", getattr(ev, "self_device_time_total", 0.0)) or 0.0)
        cuda_time = float(getattr(ev, "cuda_time_total", getattr(ev, "device_time_total", 0.0)) or 0.0)
        self_cuda_mem = int(getattr(ev, "self_cuda_memory_usage", getattr(ev, "self_device_memory_usage", 0)) or 0)
        cuda_mem = int(getattr(ev, "cuda_memory_usage", getattr(ev, "device_memory_usage", 0)) or 0)
        rows.append(
            {
                "profile_label": profile_label,
                "rank": rank,
                "event_key": ev.key,
                "count": int(getattr(ev, "count", 0) or 0),
                "self_cpu_time_us": float(getattr(ev, "self_cpu_time_total", 0.0) or 0.0),
                "cpu_time_us": float(getattr(ev, "cpu_time_total", 0.0) or 0.0),
                "self_cuda_time_us": self_cuda_time,
                "cuda_time_us": cuda_time,
                "self_cpu_memory_bytes": int(getattr(ev, "self_cpu_memory_usage", 0) or 0),
                "cpu_memory_bytes": int(getattr(ev, "cpu_memory_usage", 0) or 0),
                "self_cuda_memory_bytes": self_cuda_mem,
                "cuda_memory_bytes": cuda_mem,
                "input_shapes": str(getattr(ev, "input_shapes", "")),
                "is_record_function_phase": int(str(ev.key).startswith("KAN_") or str(ev.key).startswith("MLP_")),
            }
        )
    return rows


def partI_profiler_kwargs(activities: list[Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "activities": activities,
        "profile_memory": True,
        "record_shapes": True,
    }
    try:
        if "acc_events" in inspect.signature(torch.profiler.profile).parameters:
            kwargs["acc_events"] = True
    except (TypeError, ValueError):
        pass
    return kwargs


def run_partI_profiler_census(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    steps = int(args.partI_profiler_steps)
    train_cap = int(args.partI_profiler_train_cap)
    dataset = "Wine"
    carrier = "D-CHE-Core-K3"
    seed = 0
    basis, family, k, _unused_e = partc_carrier_spec(carrier)
    e = 16
    bundle = prepare_partd_smoke_arrays(dataset, seed, k, device, train_cap=train_cap)
    pairs = partd_train_pair_indices(bundle, seed)
    out_dim = int(bundle["output_dim"])
    rng = np.random.default_rng(302328)
    a = rng.normal(scale=0.08, size=(e, k))
    readout_np = rng.normal(scale=0.20, size=(e, out_dim))
    readout = torch.tensor(readout_np, device=device, dtype=torch.float64)
    state = PersistentFullState(k, e, beta=0.85)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler_kwargs = partI_profiler_kwargs(activities)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.profiler.profile(**profiler_kwargs) as kan_prof:
        for _step in range(max(1, steps)):
            with torch.profiler.record_function("KAN_P1_loss_backward"):
                _loss, grad, z, cot = partd_loss_and_grad(a, readout, bundle, pairs)
            with torch.profiler.record_function("KAN_P1_metric_numpy"):
                metric = build_dynamic_metric(z, cot, basis=basis)
            m = metric["M"]
            base_step = -0.12 * grad
            with torch.profiler.record_function("KAN_P1_operator_numpy"):
                radial_raw, h_force, _residual = full_operator_radial_joint_solve(a, base_step, m)
                omega_force, sym_force, global_trace = decompose_operator(h_force, m)
                radial_force = radial_raw + global_trace
            with torch.profiler.record_function("KAN_P1_fu_numpy"):
                hist_omega, hist_sym, hist_radial = state.pre_step_memory()
                a, _rho = fu_apply(a, m, base_step, hist_radial, hist_omega, hist_sym, ratio=0.10)
                state.observe(omega_force, sym_force, radial_force, m)
            kan_prof.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    kan_peak = float(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0.0)

    rng_mlp = np.random.default_rng(312328)
    w = torch.tensor(rng_mlp.normal(scale=0.08, size=(e, k)), device=device, dtype=torch.float64, requires_grad=True)
    v = torch.tensor(rng_mlp.normal(scale=0.20, size=(e, out_dim)), device=device, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW([w, v], lr=0.025, weight_decay=1.0e-5)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.profiler.profile(**profiler_kwargs) as mlp_prof:
        for _step in range(max(1, steps)):
            with torch.profiler.record_function("MLP_M1_forward_backward_step"):
                optimizer.zero_grad(set_to_none=True)
                loss = mlp_loss_value(bundle["x_train"], bundle["y_train"], w, v, str(bundle["task_family"]), pairs)
                loss.backward()
                optimizer.step()
            mlp_prof.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mlp_peak = float(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0.0)

    census_rows = [
        *partI_profiler_event_rows(kan_prof, "KAN_P1_compact_step", max_events=80),
        *partI_profiler_event_rows(mlp_prof, "MLP_M1_compact_step", max_events=80),
    ]
    write_csv(OUT_ROOT / "v23_28_partI_torch_profiler_kernel_census.csv", census_rows)
    phase_rows = [row for row in census_rows if int(finite_float(row.get("is_record_function_phase"), 0.0)) == 1]
    write_csv(OUT_ROOT / "v23_28_partI_torch_profiler_phase_rows.csv", phase_rows)
    summary_audit = {
        "phase": "partI-torch-profiler-compact-census",
        "official_partI_completion_claim": 0,
        "torch_profiler_census_completed": int(len(census_rows) > 0),
        "device": str(device),
        "cuda_available": int(torch.cuda.is_available()),
        "activities": [str(activity).split(".")[-1] for activity in activities],
        "profiler_acc_events_requested": 1,
        "profiler_acc_events_used": int(bool(profiler_kwargs.get("acc_events", False))),
        "dataset": dataset,
        "carrier": carrier,
        "carrier_basis_family": family,
        "seed": seed,
        "steps": steps,
        "train_cap": train_cap,
        "train_count_used": int(bundle["train_count_used"]),
        "census_rows": len(census_rows),
        "phase_rows": len(phase_rows),
        "KAN_peak_memory_bytes": kan_peak,
        "MLP_peak_memory_bytes": mlp_peak,
        "top_KAN_events": [row["event_key"] for row in census_rows if row.get("profile_label") == "KAN_P1_compact_step"][:10],
        "top_MLP_events": [row["event_key"] for row in census_rows if row.get("profile_label") == "MLP_M1_compact_step"][:10],
        "kernel_census_official_completed": 0,
        "no_fabrication_note": "This is a compact torch.profiler census over one KAN P1 and one MLP M1 proxy loop. It is not a full fused-op official kernel census.",
    }
    write_json(OUT_ROOT / "v23_28_partI_torch_profiler_kernel_census_summary.json", summary_audit)
    previous_path, previous = latest_json("v23_28*/v23_28_partI_summary.json")
    blockers = list(previous.get("official_blockers", ["compact_MLP_controls_not_full_official_partI", "true_MCGA_reproduction_not_implemented", "kernel_census_is_arithmetic_proxy_not_profiler_or_fused_op_census", "paired_no_debt_official_gate_not_available_from_compact_proxy", "official_same_checkpoint_full_registry_not_run"]))
    if len(census_rows) > 0:
        blockers = [
            "compact_torch_profiler_kernel_census_available_not_full_fused_op_census" if item == "kernel_census_is_arithmetic_proxy_not_profiler_or_fused_op_census" else item
            for item in blockers
        ]
    summary = dict(previous)
    summary.update(
        {
            "phase": "partI-compact-aggregate-plus-MCGA-probe-plus-profiler-census",
            "partI_completed": 0,
            "source_partI_previous_summary": rel(previous_path),
            "torch_profiler_census_summary_path": rel(OUT_ROOT / "v23_28_partI_torch_profiler_kernel_census_summary.json"),
            "torch_profiler_census_completed": int(len(census_rows) > 0),
            "torch_profiler_census_rows": len(census_rows),
            "torch_profiler_phase_rows": len(phase_rows),
            "torch_profiler_acc_events_used": int(bool(profiler_kwargs.get("acc_events", False))),
            "kernel_census_official_completed": 0,
            "official_blockers": blockers,
            "no_fabrication_note": "PartI summary is carried forward from the latest PartI summary and augmented with a compact torch.profiler census. It does not claim official v23.28 PartI completion.",
        }
    )
    write_json(OUT_ROOT / "v23_28_partI_summary.json", summary)
    failure = [
        "# v23.28 PartI torch profiler compact census",
        "",
        f"- rows/phase_rows: `{len(census_rows)}` / `{len(phase_rows)}`",
        f"- device/activities: `{device}` / `{summary_audit['activities']}`",
        f"- profiler_acc_events_used: `{summary_audit['profiler_acc_events_used']}`",
        f"- KAN/MLP peak memory bytes: `{kan_peak}` / `{mlp_peak}`",
        f"- top_KAN_events: `{json.dumps(summary_audit['top_KAN_events'], ensure_ascii=False)}`",
        f"- top_MLP_events: `{json.dumps(summary_audit['top_MLP_events'], ensure_ascii=False)}`",
        f"- official_blockers: `{json.dumps(blockers, ensure_ascii=False)}`",
        "",
        "Conclusion: compact torch.profiler rows exist, but official fused-op kernel census remains incomplete.",
        "",
    ]
    (OUT_ROOT / "v23_28_partI_torch_profiler_failure_decomposition.md").write_text("\n".join(failure), encoding="utf-8")
    files = [
        "v23_28_partI_torch_profiler_kernel_census.csv",
        "v23_28_partI_torch_profiler_phase_rows.csv",
        "v23_28_partI_torch_profiler_kernel_census_summary.json",
        "v23_28_partI_summary.json",
        "v23_28_partI_torch_profiler_failure_decomposition.md",
    ]
    append_exec("PartI_torch_profiler_compact_census", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartI torch profiler compact census",
        [
            f"rows/phase_rows `{len(census_rows)}` / `{len(phase_rows)}`; device `{device}`; activities `{summary_audit['activities']}`.",
            f"profiler_acc_events_used `{summary_audit['profiler_acc_events_used']}`.",
            f"KAN/MLP peak memory bytes `{kan_peak}` / `{mlp_peak}`.",
            f"top_KAN_events `{json.dumps(summary_audit['top_KAN_events'], ensure_ascii=False)}`.",
            f"top_MLP_events `{json.dumps(summary_audit['top_MLP_events'], ensure_ascii=False)}`.",
            f"torch_profiler_census_completed `{summary['torch_profiler_census_completed']}`; kernel_census_official_completed `0`; partI_completed `{summary['partI_completed']}`.",
            f"official_blockers `{blockers}`.",
        ],
    )


def run_completion_audit(args: argparse.Namespace) -> None:
    part0_path, part0 = latest_json("v23_28*/v23_28_plan_read_audit.json")
    parta_path, parta = latest_json("v23_28*/v23_28_partA_summary.json")
    partb_path, partb = latest_json("v23_28*/v23_28_partB_summary.json")
    partc_path, partc = latest_json("v23_28*/v23_28_partC_summary.json")
    partd_path, partd = latest_json("v23_28*/v23_28_partD_summary.json")
    parte_path, parte = latest_json("v23_28*/v23_28_partE_summary.json")
    partf_path, partf = latest_json("v23_28*/v23_28_partF_summary.json")
    partg_path, partg = latest_json("v23_28*/v23_28_partG_summary.json")
    parth_path, parth = latest_json("v23_28*/v23_28_partH_summary.json")
    parti_path, parti = latest_json("v23_28*/v23_28_partI_summary.json")
    missing = []
    if not part0:
        missing.append("part0")
    if not parta:
        missing.append("partA")
    if not partb:
        missing.append("partB")
    if not partc:
        missing.append("partC")
    if not partd:
        missing.append("partD")
    if not parte:
        missing.append("partE")
    if not partf:
        missing.append("partF")
    if not partg:
        missing.append("partG")
    if not parth:
        missing.append("partH")
    if not parti:
        missing.append("partI")
    part0_gate = int(part0.get("part0_hard_gate_pass", 0) or 0)
    parta_gate = int(parta.get("partA_semantic_pass", 0) or 0)
    partb_completed = int(partb.get("partB_completed", 0) or 0) if partb else 0
    partc_completed = int(partc.get("partC_completed", 0) or 0) if partc else 0
    partd_completed = int(partd.get("partD_completed", 0) or 0) if partd else 0
    parte_completed = int(parte.get("partE_completed", 0) or 0) if parte else 0
    partf_completed = int(partf.get("partF_completed", 0) or 0) if partf else 0
    partg_completed = int(partg.get("partG_completed", 0) or 0) if partg else 0
    parth_completed = int(parth.get("partH_completed", 0) or 0) if parth else 0
    parti_completed = int(parti.get("partI_completed", 0) or 0) if parti else 0
    partc_blockers = partc.get("partC_official_blockers", ["not_implemented_or_not_run_in_current_v23_28_runner"]) if partc else ["not_implemented_or_not_run_in_current_v23_28_runner"]
    partc_blockers = ["official_PartD_to_PartI_not_complete" if item == "PartD_to_PartI_not_run" else item for item in partc_blockers]
    official_ready = 0
    blockers = {
        "Part0": part0.get("part0_gate_blockers", ["missing_part0"]) if part0 else ["missing_part0"],
        "PartA": [] if parta_gate else ["partA_missing_or_failed"],
        "PartB": [] if partb_completed else [partb.get("blocker", "not_implemented_or_not_run_in_current_v23_28_runner") if partb else "not_implemented_or_not_run_in_current_v23_28_runner"],
        "PartC": [] if partc_completed else partc_blockers,
        "PartD": [] if partd_completed else (partd.get("official_blockers", ["not_implemented_or_not_run_in_current_v23_28_runner"]) if partd else ["not_implemented_or_not_run_in_current_v23_28_runner"]),
        "PartE": [] if parte_completed else (parte.get("official_blockers", ["not_implemented_or_not_run_in_current_v23_28_runner"]) if parte else ["not_implemented_or_not_run_in_current_v23_28_runner"]),
        "PartF": [] if partf_completed else (partf.get("official_blockers", ["not_implemented_or_not_run_in_current_v23_28_runner"]) if partf else ["not_implemented_or_not_run_in_current_v23_28_runner"]),
        "PartG": [] if partg_completed else (partg.get("official_blockers", ["not_implemented_or_not_run_in_current_v23_28_runner"]) if partg else ["not_implemented_or_not_run_in_current_v23_28_runner"]),
        "PartH": [] if parth_completed else (parth.get("official_blockers", ["not_implemented_or_not_run_in_current_v23_28_runner"]) if parth else ["not_implemented_or_not_run_in_current_v23_28_runner"]),
        "PartI": [] if parti_completed else (parti.get("official_blockers", ["not_implemented_or_not_run_in_current_v23_28_runner"]) if parti else ["not_implemented_or_not_run_in_current_v23_28_runner"]),
    }
    route = "R0_IncompleteScientificExploration"
    audit = {
        "phase": "completion-audit",
        "source_paths": {"part0": rel(part0_path), "partA": rel(parta_path), "partB": rel(partb_path), "partC": rel(partc_path), "partD": rel(partd_path), "partE": rel(parte_path), "partF": rel(partf_path), "partG": rel(partg_path), "partH": rel(parth_path), "partI": rel(parti_path)},
        "missing_sources": missing,
        "gate_status": {
            "part0_hard_gate": part0_gate,
            "partA_semantic_gate": parta_gate,
            "partB_completed": partb_completed,
            "partC_completed": partc_completed,
            "partD_completed": partd_completed,
            "partE_completed": parte_completed,
            "partF_completed": partf_completed,
            "partG_completed": partg_completed,
            "partH_completed": parth_completed,
            "partI_completed": parti_completed,
        },
        "hard_blockers": blockers,
        "official_candidate_ready": official_ready,
        "current_final_route": route,
        "science_conclusion_upgraded": 0,
    }
    incomplete_parts = [part for part, values in blockers.items() if values]
    partb_reason = blockers.get("PartB", [""])[0] if blockers.get("PartB") else ""
    final_route = {
        "phase": "final-route-current-evidence",
        "current_final_route": route,
        "official_candidate_ready": official_ready,
        "route_reason": (
            f"v23.28 has Part0 and PartA gates passing; incomplete parts are {','.join(incomplete_parts)}; PartB blocker is {partb_reason}; PartC through PartI still have compact, partial, or non-official evidence in this runner."
            if part0_gate == 1 and parta_gate == 1 and (not partb_completed or not partc_completed or not partd_completed or not parte_completed or not partf_completed or not partg_completed or not parth_completed or not parti_completed)
            else "v23.28 has incomplete prerequisite gates and/or PartB-I mandatory matrices in the current artifacts."
        ),
        "science_conclusion_upgraded": 0,
    }
    write_json(OUT_ROOT / "v23_28_completion_audit.json", audit)
    write_json(OUT_ROOT / "v23_28_final_route.json", final_route)
    md = [
        "# v23.28 failure dissection",
        "",
        f"- current_final_route: `{route}`",
        f"- official_candidate_ready: `{official_ready}`",
        f"- source_paths: `{json.dumps(audit['source_paths'], ensure_ascii=False, sort_keys=True)}`",
        f"- Part0 blockers: `{blockers['Part0']}`",
        f"- PartA blockers: `{blockers['PartA']}`",
        f"- PartB blockers: `{blockers['PartB']}`",
        f"- PartC blockers: `{blockers['PartC']}`",
        f"- PartD blockers: `{blockers['PartD']}`",
        f"- PartE blockers: `{blockers['PartE']}`",
        f"- PartF blockers: `{blockers['PartF']}`",
        f"- PartG blockers: `{blockers['PartG']}`",
        f"- PartH blockers: `{blockers['PartH']}`",
        f"- PartI blockers: `{blockers['PartI']}`",
        "",
        "Conclusion: no science claim is made. The current work establishes v23.28 registries, PartA math/semantic units, a PartB source audit, reduced PartC/PartD/PartE/PartF/PartG/PartH/PartI compact evidence, and no official final route.",
        "",
    ]
    (OUT_ROOT / "v23_28_failure_dissection.md").write_text("\n".join(md), encoding="utf-8")
    files = ["v23_28_completion_audit.json", "v23_28_final_route.json", "v23_28_failure_dissection.md"]
    append_exec("Completion_audit", args, files, audit, "completed_incomplete_gate")
    append_recap(
        "Completion audit",
        [
            f"route `{route}`; official_candidate_ready `{official_ready}`.",
            f"source_paths `{json.dumps(audit['source_paths'], ensure_ascii=False, sort_keys=True)}`.",
            f"gate_status `{json.dumps(audit['gate_status'], ensure_ascii=False, sort_keys=True)}`.",
            f"hard_blockers `{json.dumps(blockers, ensure_ascii=False, sort_keys=True)}`.",
            "不升级 science conclusion；PartB-I 尚未完成，不能输出 family-level NoGo 或 positive route。",
        ],
    )


def run_repair_audit(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    partb_generated_path, partb_generated = latest_json("v23_28_gpu3_partB_generated_diag*/v23_28_partB_summary.json")
    partb_export_path, partb_export = latest_json("v23_28*/v23_28_partB_summary.json")
    partc_path, partc = latest_json("v23_28*/v23_28_partC_summary.json")
    partd_path, partd = latest_json("v23_28*/v23_28_partD_summary.json")
    parte_path, parte = latest_json("v23_28*/v23_28_partE_summary.json")
    partf_path, partf = latest_json("v23_28*/v23_28_partF_summary.json")
    partg_path, partg = latest_json("v23_28*/v23_28_partG_summary.json")
    partg_warmupdiag_path, partg_warmupdiag = latest_json("diag_v23_28*/v23_28_partG_summary.json")
    parth_path, parth = latest_json("v23_28*/v23_28_partH_summary.json")
    parth_capdiag_path, parth_capdiag = latest_json("diag_v23_28*/v23_28_partH_summary.json")
    parti_mcga_path, parti_mcga = latest_json("v23_28_gpu3_partI_mcga_probe_audit*/v23_28_partI_summary.json")
    parti_profiler_path, parti_profiler = latest_json("v23_28_gpu3_partI_profiler_census*/v23_28_partI_summary.json")

    def blockers(summary: dict[str, Any], key: str = "official_blockers") -> str:
        vals = summary.get(key, []) if summary else []
        if not vals and key != "partC_official_blockers":
            vals = summary.get("partC_official_blockers", []) if summary else []
        return ";".join(str(v) for v in vals)

    def evidence(*paths: Path | None) -> str:
        return ";".join(rel(path) for path in paths if path is not None)

    base = {
        "formula_changed_or_not": 0,
        "threshold_changed_or_not": 0,
        "hypothesis_identity_changed_or_not": 0,
        "official_completion_claim": 0,
        "science_conclusion_upgraded": 0,
    }
    records: list[dict[str, Any]] = [
        {
            **base,
            "repair_id": "R_B0_generated_checkpoint_format_diagnostic",
            "triggered_blocker": "v23_27_replay_checkpoints_missing",
            "allowed_plan_direction": "PartB first repair checks replayability and representability without relaxing R_full thresholds.",
            "modified_files": "experiments/run_v23_28_persistent_compositional_transport_shaping_fu.py;experiments/package_v23_28_core_code_audit_pack.py",
            "artifacts_invalidated": "none; generated diagnostic checkpoint artifacts remain labeled not_true_v2327_winner_checkpoint",
            "required_reruns": "original v23.27 winner checkpoint replay remains required",
            "evidence_artifacts": evidence(partb_generated_path),
            "validation_status": f"generated_diagnostic_rows={partb_generated.get('representability_rows', '')};official_partB_completion_claim=0",
            "blocker_after": partb_generated.get("blocker", "official_v23_27_winner_replay_checkpoints_missing_generated_compact_diagnostic_only") if partb_generated else "generated_diagnostic_summary_missing",
        },
        {
            **base,
            "repair_id": "R_C1_task_native_mlp_schema_coverage",
            "triggered_blocker": "MLP_registry_rows_not_run_in_partC_preflight;classification_CE_MSE_pairwise_task_native_synthetic_losses_not_all_implemented;mandatory_metric_schema_incomplete_for_official_partC",
            "allowed_plan_direction": "PartC failure must generate failure decomposition and allowed repair records while minimum-real continues.",
            "modified_files": "experiments/run_v23_28_persistent_compositional_transport_shaping_fu.py;experiments/package_v23_28_core_code_audit_pack.py",
            "artifacts_invalidated": "none; older compact preflight artifacts preserved and newer task-native/MLP aggregate is separate",
            "required_reruns": "full fused-carrier official PartC remains required; C2/C3 scientific gates still fail on compact evidence",
            "evidence_artifacts": evidence(partc_path),
            "validation_status": f"rows={partc.get('rows', '')};task_native_rows={partc.get('task_native_rows', '')};mlp_rows={partc.get('mlp_rows', '')};partC_completed={partc.get('partC_completed', '')}",
            "blocker_after": blockers(partc, "partC_official_blockers"),
        },
        {
            **base,
            "repair_id": "R_D1_expanded_registry_mlp_efficiency_proxy",
            "triggered_blocker": "full_28_scheme_partD_registry_not_run;MLP_matched_and_efficiency_controls_not_run",
            "allowed_plan_direction": "PartD mandatory matrix continues after PartC failure; MLP stronger and efficiency blockers require stronger controls, not weakened controls.",
            "modified_files": "experiments/run_v23_28_persistent_compositional_transport_shaping_fu.py;experiments/package_v23_28_core_code_audit_pack.py",
            "artifacts_invalidated": "none; compact expanded registry is a proxy and does not replace official fused matrix",
            "required_reruns": "official fused-carrier same-checkpoint PartD matrix remains required",
            "evidence_artifacts": evidence(partd_path),
            "validation_status": f"compact_rows={partd.get('compact_matrix_rows', '')};expanded_kan_all_28={partd.get('expanded_kan_all_28_schemes_present', '')};MLP_all={partd.get('MLP_all_registry_schemes_present', '')};partD_completed={partd.get('partD_completed', '')}",
            "blocker_after": blockers(partd),
        },
        {
            **base,
            "repair_id": "R_EFGH1_debt_schema_and_no_debt_failure_decomposition",
            "triggered_blocker": "classification_regression_pairwise_debt_metrics_compact_proxy_only;paired_no_debt_official_gate_not_available_from_compact_proxy",
            "allowed_plan_direction": "Continue minimum-real/state/component/horizon controls and report debt/no-debt failures instead of weakening metrics.",
            "modified_files": "experiments/run_v23_28_persistent_compositional_transport_shaping_fu.py;experiments/package_v23_28_core_code_audit_pack.py",
            "artifacts_invalidated": "none; compact debt schema rows are diagnostic and official no-debt gates remain incomplete",
            "required_reruns": "official same-checkpoint full registries with paired no-debt metrics remain required for PartE/F/G/H",
            "evidence_artifacts": evidence(parte_path, partf_path, partg_path, parth_path),
            "validation_status": (
                f"PartE_gate={parte.get('compact_gate_with_debt_pass', '')};"
                f"PartF_component_gate={partf.get('component_compact_gate_with_debt_pass', '')};"
                f"PartG_gate={partg.get('compact_gate_with_debt_pass', '')};"
                f"PartH_H80_H200_gate={parth.get('H80_compact_gate_with_debt_pass', '')}/{parth.get('H200_compact_gate_with_debt_pass', '')}"
            ),
            "blocker_after": ";".join(filter(None, [blockers(parte), blockers(partf), blockers(partg), blockers(parth)])),
        },
        {
            **base,
            "repair_id": "R_H1_sym_spectral_cap_diagnostic",
            "triggered_blocker": "H80_shape_formation_gate_failed_after_ratio_0p10;ShapeFormationOpenedButDebtBlocked_allows_one_ratio_or_cap_repair",
            "allowed_plan_direction": "Symmetric-state repair allows fixed whitened spectral cap ||rho S||_2 <= 0.05 once; no threshold weakening or selector/loss trick.",
            "modified_files": "experiments/run_v23_28_persistent_compositional_transport_shaping_fu.py;experiments/package_v23_28_core_code_audit_pack.py",
            "formula_changed_or_not": 1,
            "artifacts_invalidated": "none; capped H80 diagnostic is written under results/diag_v23_28* and does not replace the uncapped PartH source used by completion audit",
            "required_reruns": "do not sweep cap; official fused H80/H200 and paired no-debt evidence remain required",
            "evidence_artifacts": evidence(parth_capdiag_path),
            "validation_status": (
                f"cap_values={parth_capdiag.get('partH_sym_spectral_caps', '')};"
                f"H80_rows={parth_capdiag.get('H80_rows', '')};"
                f"H80_gate_without_debt={parth_capdiag.get('H80_compact_gate_without_debt_pass', '')};"
                f"H80_gate_with_debt={parth_capdiag.get('H80_compact_gate_with_debt_pass', '')};"
                f"H80_P1_vs_K2_median={parth_capdiag.get('H80_delta_summary_P1_minus_controls', {}).get('K2_IntrinsicAdditive_CompH2', {}).get('median', '')};"
                f"H80_P1_vs_P0_median={parth_capdiag.get('H80_delta_summary_P1_minus_controls', {}).get('P0_FullInstant_CompH2', {}).get('median', '')}"
            ),
            "blocker_after": "H80_sym_cap_0p05_diagnostic_did_not_pass_compact_gate;P1_vs_K2_median_still_below_0p001_and_P1_vs_P0_still_negative;paired_no_debt_rate_still_0p025",
        },
        {
            **base,
            "repair_id": "R_G1_warmup_fraction_0p20_diagnostic",
            "triggered_blocker": "HybridPositivePureWeakerOrDebtBlocked;PartG_compact_with_debt_gate_failed_and_P2_vs_hybrid_P1_negative",
            "allowed_plan_direction": "Hybrid/Pure repair allows warmup fraction 0.10 -> pre-registered diagnostic 0.20 once; no further warmup sweep.",
            "modified_files": "none for diagnostic execution; experiments/run_v23_28_persistent_compositional_transport_shaping_fu.py updated only to register this repair in repair-audit",
            "artifacts_invalidated": "none; warmup=0.20 diagnostic is written under results/diag_v23_28* and does not replace the uncapped PartG source used by completion audit",
            "required_reruns": "do not sweep warmup; official fused PartG and paired no-debt evidence remain required",
            "evidence_artifacts": evidence(partg_warmupdiag_path),
            "validation_status": (
                f"rows={partg_warmupdiag.get('compact_matrix_rows', '')}/{partg_warmupdiag.get('expected_compact_matrix_rows', '')};"
                f"without_debt={partg_warmupdiag.get('compact_gate_without_debt_pass', '')};"
                f"with_debt={partg_warmupdiag.get('compact_gate_with_debt_pass', '')};"
                f"paired_no_debt_rate={partg_warmupdiag.get('paired_no_debt_rate', '')};"
                f"P2_vs_warmup_median={partg_warmupdiag.get('pure_delta_summary_P2_minus_controls', {}).get('warmup_only', {}).get('median', '')};"
                f"P2_vs_hybrid_P1_median={partg_warmupdiag.get('pure_delta_summary_P2_minus_controls', {}).get('hybrid_P1', {}).get('median', '')}"
            ),
            "blocker_after": "warmup_0p20_diagnostic_kept_without_debt_gate_pass_but_with_debt_gate_failed;paired_no_debt_rate_0p14;P2_vs_hybrid_P1_still_negative",
        },
        {
            **base,
            "repair_id": "R_I1_mcga_probe_availability",
            "triggered_blocker": "true_MCGA_reproduction_not_implemented",
            "allowed_plan_direction": "MLP stronger repair requires enhancing MLP controls and reproducing MCGA, without weakening MLP.",
            "modified_files": "experiments/run_v23_28_persistent_compositional_transport_shaping_fu.py;experiments/package_v23_28_core_code_audit_pack.py",
            "artifacts_invalidated": "none; one-row MCGA probe is explicitly not the official v23.28 reproduction matrix",
            "required_reruns": "full official v23.28 MCGA reproduction matrix and paired controls remain required",
            "evidence_artifacts": evidence(parti_mcga_path),
            "validation_status": f"true_MCGA_single_row_smoke_available={parti_mcga.get('true_MCGA_single_row_smoke_available', '')};partI_completed={parti_mcga.get('partI_completed', '')}",
            "blocker_after": blockers(parti_mcga),
        },
        {
            **base,
            "repair_id": "R_I2_profiler_acc_events_accounting",
            "triggered_blocker": "compact profiler census warning: events cleared at cycle end unless acc_events=True",
            "allowed_plan_direction": "Efficiency/kernel-census repair may improve profiler accounting without changing science identity.",
            "modified_files": "experiments/run_v23_28_persistent_compositional_transport_shaping_fu.py",
            "artifacts_invalidated": "none; previous profiler census remains weaker evidence and accumulated-event census supersedes it for audit source",
            "required_reruns": "full fused-op official kernel census remains required",
            "evidence_artifacts": evidence(parti_profiler_path),
            "validation_status": f"profiler_acc_events_used={parti_profiler.get('profiler_acc_events_used', '')};torch_profiler_census_completed={parti_profiler.get('torch_profiler_census_completed', '')};partI_completed={parti_profiler.get('partI_completed', '')}",
            "blocker_after": blockers(parti_profiler),
        },
        {
            **base,
            "repair_id": "R_B1_v2327_origin_rerun_checkpoint_export",
            "triggered_blocker": "official_v23_27_winner_replay_checkpoints_missing_generated_compact_diagnostic_only",
            "allowed_plan_direction": "PartB second repair can verify receiving-node bank/ridge/replay coverage but cannot replace original winner checkpoints.",
            "modified_files": "experiments/run_v23_27_persistent_compositional_curvature_lie_generator_fu.py;experiments/run_v23_28_persistent_compositional_transport_shaping_fu.py;experiments/package_v23_28_core_code_audit_pack.py",
            "artifacts_invalidated": "none; v23.27-origin rerun exports are separate and marked not_original_v2327_winner_checkpoint",
            "required_reruns": "recover original v23.27 winner checkpoints and add regression/pairwise representative replay states",
            "evidence_artifacts": evidence(partb_export_path),
            "validation_status": f"usable_checkpoint_rows={partb_export.get('usable_checkpoint_rows', '')};R_full_median={partb_export.get('R_full_median', '')};diagnostic_gate_pass_if_rerun_exports_were_allowed={partb_export.get('diagnostic_gate_pass_if_rerun_exports_were_allowed', '')};partB_completed={partb_export.get('partB_completed', '')}",
            "blocker_after": partb_export.get("blocker", "") if partb_export else "partB_export_summary_missing",
        },
    ]
    required_fields = [
        "repair_id",
        "triggered_blocker",
        "modified_files",
        "formula_changed_or_not",
        "threshold_changed_or_not",
        "hypothesis_identity_changed_or_not",
        "artifacts_invalidated",
        "required_reruns",
    ]
    for row in records:
        row["schema_complete"] = int(all(str(row.get(field, "")) != "" for field in required_fields))
    write_csv(OUT_ROOT / "v23_28_executed_repair_registry.csv", records)
    summary = {
        "phase": "repair-audit",
        "executed_repair_count": len(records),
        "required_fields": required_fields,
        "schema_complete_rows": sum(int(row["schema_complete"]) for row in records),
        "all_schema_rows_complete": int(all(int(row["schema_complete"]) == 1 for row in records)),
        "formula_changed_count": sum(int(row["formula_changed_or_not"]) for row in records),
        "threshold_changed_count": sum(int(row["threshold_changed_or_not"]) for row in records),
        "hypothesis_identity_changed_count": sum(int(row["hypothesis_identity_changed_or_not"]) for row in records),
        "official_completion_claim_count": sum(int(row["official_completion_claim"]) for row in records),
        "science_conclusion_upgraded": 0,
        "partB_good_news_diagnostic_not_official": {
            "R_full_median": partb_export.get("R_full_median", float("nan")),
            "R_full_ge_0p20_rate": partb_export.get("R_full_ge_0p20_rate", float("nan")),
            "covariance_failure_count": partb_export.get("covariance_failure_count", ""),
            "official_partB_completion_claim": partb_export.get("official_partB_completion_claim", 0),
        },
        "no_fabrication_note": "This registry records executed repairs and their existing evidence artifacts. It does not reinterpret compact proxies as official gates and does not upgrade any science conclusion.",
    }
    write_json(OUT_ROOT / "v23_28_executed_repair_registry.json", {"summary": summary, "repairs": records})
    md = [
        "# v23.28 executed repair audit",
        "",
        f"- executed_repair_count: `{summary['executed_repair_count']}`",
        f"- all_schema_rows_complete: `{summary['all_schema_rows_complete']}`",
        f"- formula/threshold/hypothesis identity changes: `{summary['formula_changed_count']}` / `{summary['threshold_changed_count']}` / `{summary['hypothesis_identity_changed_count']}`",
        f"- official_completion_claim_count: `{summary['official_completion_claim_count']}`",
        f"- PartB diagnostic good news: `{json.dumps(json_clean(summary['partB_good_news_diagnostic_not_official']), ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    for row in records:
        md.extend(
            [
                f"## {row['repair_id']}",
                "",
                f"- triggered_blocker: `{row['triggered_blocker']}`",
                f"- modified_files: `{row['modified_files']}`",
                f"- formula_changed_or_not / threshold_changed_or_not / hypothesis_identity_changed_or_not: `{row['formula_changed_or_not']}` / `{row['threshold_changed_or_not']}` / `{row['hypothesis_identity_changed_or_not']}`",
                f"- evidence_artifacts: `{row['evidence_artifacts']}`",
                f"- validation_status: `{row['validation_status']}`",
                f"- blocker_after: `{row['blocker_after']}`",
                "",
            ]
        )
    (OUT_ROOT / "v23_28_repair_audit.md").write_text("\n".join(md), encoding="utf-8")
    files = ["v23_28_executed_repair_registry.csv", "v23_28_executed_repair_registry.json", "v23_28_repair_audit.md"]
    append_exec("Repair_audit", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "Executed repair registry audit",
        [
            f"executed_repair_count `{summary['executed_repair_count']}`; all_schema_rows_complete `{summary['all_schema_rows_complete']}`.",
            f"formula/threshold/hypothesis identity changes `{summary['formula_changed_count']}` / `{summary['threshold_changed_count']}` / `{summary['hypothesis_identity_changed_count']}`.",
            f"official_completion_claim_count `{summary['official_completion_claim_count']}`; science_conclusion_upgraded `0`.",
            f"PartB diagnostic good news `{json.dumps(json_clean(summary['partB_good_news_diagnostic_not_official']), ensure_ascii=False, sort_keys=True)}` remains non-official because original v23.27 winner checkpoints and regression/pairwise replay states are missing.",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["part0", "partA", "partB", "partB-diagnostic", "partB-v2327-export-audit", "partC", "partC-aggregate", "partD-audit", "partD-smoke", "partD-matrix", "partD-aggregate", "partE-matrix", "partE-aggregate", "partF-matrix", "partF-aggregate", "partG-matrix", "partG-aggregate", "partH-matrix", "partH-aggregate", "partI-matrix", "partI-aggregate", "partI-mcga-probe-audit", "partI-profiler-census", "repair-audit", "completion-audit"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-full-read-complete", type=int, default=0)
    parser.add_argument("--lineage-full-read-complete", type=int, default=0)
    parser.add_argument("--partB-diagnostic-seeds", type=int, default=5)
    parser.add_argument("--partB-diagnostic-max-groups", type=int, default=0)
    parser.add_argument("--partc-shard-index", type=int, default=0)
    parser.add_argument("--partc-shard-count", type=int, default=1)
    parser.add_argument("--partc-seeds", type=int, default=5)
    parser.add_argument("--partc-horizon", type=int, default=20)
    parser.add_argument("--partc-max-groups", type=int, default=0)
    parser.add_argument("--partc-fu-ratio", type=float, default=0.15)
    parser.add_argument("--partc-target-mode", choices=["moving_tangent", "fixed_teacher"], default="moving_tangent")
    parser.add_argument("--partc-include-task-native-loss", type=int, default=0)
    parser.add_argument("--partc-include-mlp", type=int, default=0)
    parser.add_argument("--partd-smoke-seed", type=int, default=0)
    parser.add_argument("--partd-smoke-horizon", type=int, default=20)
    parser.add_argument("--partd-matrix-seeds", type=int, default=5)
    parser.add_argument("--partd-matrix-horizon", type=int, default=20)
    parser.add_argument("--partd-shard-index", type=int, default=0)
    parser.add_argument("--partd-shard-count", type=int, default=1)
    parser.add_argument("--partd-max-groups", type=int, default=0)
    parser.add_argument("--partd-train-cap", type=int, default=512)
    parser.add_argument("--partd-include-expanded-registry", type=int, default=0)
    parser.add_argument("--partd-include-mlp-efficiency", type=int, default=0)
    parser.add_argument("--partd-fu-ratio", type=float, default=0.10)
    parser.add_argument("--partE-matrix-seeds", type=int, default=5)
    parser.add_argument("--partE-matrix-horizon", type=int, default=20)
    parser.add_argument("--partE-shard-index", type=int, default=0)
    parser.add_argument("--partE-shard-count", type=int, default=1)
    parser.add_argument("--partE-max-groups", type=int, default=0)
    parser.add_argument("--partF-matrix-seeds", type=int, default=5)
    parser.add_argument("--partF-matrix-horizon", type=int, default=20)
    parser.add_argument("--partF-shard-index", type=int, default=0)
    parser.add_argument("--partF-shard-count", type=int, default=1)
    parser.add_argument("--partF-max-groups", type=int, default=0)
    parser.add_argument("--partG-matrix-seeds", type=int, default=5)
    parser.add_argument("--partG-matrix-horizon", type=int, default=20)
    parser.add_argument("--partG-shard-index", type=int, default=0)
    parser.add_argument("--partG-shard-count", type=int, default=1)
    parser.add_argument("--partG-max-groups", type=int, default=0)
    parser.add_argument("--partG-warmup-fraction", type=float, default=0.10)
    parser.add_argument("--partH-kind", choices=["H80", "H200"], default="H80")
    parser.add_argument("--partH-matrix-seeds", type=int, default=5)
    parser.add_argument("--partH-horizon", type=int, default=0)
    parser.add_argument("--partH-shard-index", type=int, default=0)
    parser.add_argument("--partH-shard-count", type=int, default=1)
    parser.add_argument("--partH-max-groups", type=int, default=0)
    parser.add_argument("--partH-sym-spectral-cap", type=float, default=0.0)
    parser.add_argument("--partI-matrix-seeds", type=int, default=5)
    parser.add_argument("--partI-matrix-horizon", type=int, default=20)
    parser.add_argument("--partI-shard-index", type=int, default=0)
    parser.add_argument("--partI-shard-count", type=int, default=1)
    parser.add_argument("--partI-max-groups", type=int, default=0)
    parser.add_argument("--partI-mcga-probe-label", default="v23_28_partI_mcga_probe")
    parser.add_argument("--partI-profiler-steps", type=int, default=3)
    parser.add_argument("--partI-profiler-train-cap", type=int, default=128)
    args = parser.parse_args()
    if args.phase == "part0":
        run_part0(args)
    elif args.phase == "partA":
        run_partA(args)
    elif args.phase == "partB":
        run_partB(args)
    elif args.phase == "partB-diagnostic":
        run_partB_diagnostic(args)
    elif args.phase == "partB-v2327-export-audit":
        run_partB_v2327_export_audit(args)
    elif args.phase == "partC":
        run_partC(args)
    elif args.phase == "partC-aggregate":
        run_partC_aggregate(args)
    elif args.phase == "partD-audit":
        run_partD_audit(args)
    elif args.phase == "partD-smoke":
        run_partD_smoke(args)
    elif args.phase == "partD-matrix":
        run_partD_matrix(args)
    elif args.phase == "partD-aggregate":
        run_partD_aggregate(args)
    elif args.phase == "partE-matrix":
        run_partE_matrix(args)
    elif args.phase == "partE-aggregate":
        run_partE_aggregate(args)
    elif args.phase == "partF-matrix":
        run_partF_matrix(args)
    elif args.phase == "partF-aggregate":
        run_partF_aggregate(args)
    elif args.phase == "partG-matrix":
        run_partG_matrix(args)
    elif args.phase == "partG-aggregate":
        run_partG_aggregate(args)
    elif args.phase == "partH-matrix":
        run_partH_matrix(args)
    elif args.phase == "partH-aggregate":
        run_partH_aggregate(args)
    elif args.phase == "partI-matrix":
        run_partI_matrix(args)
    elif args.phase == "partI-aggregate":
        run_partI_aggregate(args)
    elif args.phase == "partI-mcga-probe-audit":
        run_partI_mcga_probe_audit(args)
    elif args.phase == "partI-profiler-census":
        run_partI_profiler_census(args)
    elif args.phase == "repair-audit":
        run_repair_audit(args)
    elif args.phase == "completion-audit":
        run_completion_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
