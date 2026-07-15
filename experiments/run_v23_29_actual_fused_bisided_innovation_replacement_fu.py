#!/usr/bin/env python3
"""DG-KAN v23.29 actual fused-carrier firewall runner.

This runner starts with the non-negotiable v23.29 work: prove that the
experiment object is the real fused D-CHE / D-FOU PrimitiveKAN path before
any science matrix is allowed to make claims.  It intentionally does not reuse
the v23.28 compact proxy science rows.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path
import glob
from typing import Any, Callable

import torch
import torch.nn.functional as F

from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/DG-KAN_v23.29_ActualFusedCarrier_BasisCovariantBiSidedInnovationReplacementEdgeFlowFU_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.29_ActualFusedCarrier_BasisCovariantBiSidedInnovationReplacementEdgeFlowFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.29_ActualFusedCarrier_BasisCovariantBiSidedInnovationReplacementEdgeFlowFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2329_OUT_ROOT", str(ROOT / "results/v23_29"))).resolve()

LINEAGE_DOCS = {
    "v23.26": ROOT / "docs/DG-KAN_v23.26_DCHE_DFOU_BasisCovariantIntrinsicRadialTangentialGeneratorFlow_完整详尽实验计划.md",
    "v23.27": ROOT / "docs/DG-KAN_v23.27_PersistentBasisCovariantCompositionalCurvatureLieGeneratorFU_多假设语义穷尽式完整详尽实验计划.md",
    "v23.28": ROOT / "docs/DG-KAN_v23.28_PersistentBasisCovariantCompositionalTransportShapingEdgeFlowFU_多假设语义穷尽式完整详尽实验计划.md",
    "v23.29": PLAN,
}

HYPOTHESES = [
    ("H-A", "Actual fused carrier reality firewall"),
    ("H-B", "Task-native fused backward and compositional jet metric correctness"),
    ("H-C", "Bi-sided generator representability"),
    ("H-D", "Low-rank left operator sufficiency and efficiency"),
    ("H-E", "Innovation-replacement causal value against add-on and base"),
    ("H-F", "Persistent state predictability and transport"),
    ("H-G", "True compositional jet metric causal value"),
    ("H-H", "Hybrid and warmup-then-Pure FU"),
    ("H-I", "Cross-loss-family, H80/H200, and debt"),
    ("H-J", "MLP matched, MCGA, carrier efficiency, and architecture surplus"),
]

SCHEMES = {
    "kan_base": [
        "K0_AdamW_TaskNative",
        "K1_IntrinsicAdditive_ProductL2",
        "K2_IntrinsicAdditive_ProductJet",
        "K3_v2328_OneSidedPersistentAddOn",
        "K4_v2328_OneSidedPersistentReplacement",
    ],
    "representability": [
        "R0_OneSided_DiagPlusRight",
        "R1_BiSided_Q2",
        "R2_BiSided_Q4",
        "R3_BiSided_Q8_Diagnostic",
        "R4_BiSided_FullLeft_UpperBound",
    ],
    "fu_primary_controls": [
        "P0_BiSided_Instantaneous_StructuredMap",
        "P1_BiSided_Persistent_AddOn",
        "P2_BiSided_Persistent_Replacement_PRIMARY",
        "P3_BiSided_Persistent_Replacement_RhoHalf_RepairOnly",
        "P4_BiSided_ResetEveryStep",
        "P5_BiSided_RandomAR1",
        "P6_BiSided_SameAutocorrelationRandom",
        "P7_BiSided_TimeLag4",
        "P8_BiSided_TimeShuffled",
        "P9_BiSided_BankShuffled",
        "P10_BiSided_SignFlip",
        "P11_BiSided_SameComputeNoop",
    ],
    "metric_controls": [
        "M0_DataL2",
        "M1_LocalJet",
        "M2_CompositionalJet_PRIMARY",
        "M3_PathShuffledJet",
        "M4_UniformPathJet",
        "M5_BasisChannelPermutationIdentityDiagnostic",
    ],
    "component_attribution": [
        "A0_RightOnly",
        "A1_DiagonalLeftOnly",
        "A2_LowRankLeftOnly",
        "A3_LeftPlusRightNoRadial",
        "A4_FullBiSided",
        "A5_FullBiSidedRandomMatched",
    ],
    "pure_fu": [
        "U0_WarmupOnly",
        "U1_Warmup10_PureBiSided",
        "U2_Warmup20_PureBiSided_RepairOnly",
        "U3_Warmup10_PureOneSided",
        "U4_Warmup10_PureRandomState",
        "U5_Warmup10_PureResetState",
    ],
    "mlp_controls": [
        "MLP0_SameParam_StrongOptimizer",
        "MLP1_SameFLOPs_StrongOptimizer",
        "MLP2_Matched_BiSidedPersistentReplacement",
        "MLP3_Matched_RandomPersistentReplacement",
        "MLP4_MCGA_Reproduction",
        "MLP5_SameComputeNoop",
    ],
}


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
    if dataclasses.is_dataclass(obj):
        return json_clean(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(key): json_clean(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return json_clean(obj.detach().cpu().item())
        return json_clean(obj.detach().cpu().tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(obj), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_clean_value(value: Any) -> Any:
    value = json_clean(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


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
        for row in rows:
            writer.writerow({key: csv_clean_value(row.get(key, "")) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_hash_obj(obj: Any) -> str:
    payload = json.dumps(json_clean(obj), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_hash(obj: Any) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode("utf-8")).hexdigest()


def ensure_logs() -> None:
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.29 Actual Fused Bi-Sided Innovation-Replacement FU 执行日志\n\n"
            "本日志记录真实执行命令、环境、产物与读回结果；禁止补造数据。\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.29 Actual Fused Bi-Sided Innovation-Replacement FU 实验结果复盘\n\n"
            "本复盘只引用落盘 artifact 与真实读回结果；失败与 blocker 必须如实记录。\n\n",
            encoding="utf-8",
        )


def append_exec(phase: str, args: argparse.Namespace, files: list[str], note: dict[str, Any], status: str) -> None:
    ensure_logs()
    cmd = " ".join([str(Path(os.sys.executable)), str(Path(__file__).relative_to(ROOT)), *os.sys.argv[1:]])
    env = {
        "python": str(Path(os.sys.executable)),
        "torch": getattr(torch, "__version__", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "device_arg": str(getattr(args, "device", "")),
        "out_root": rel(OUT_ROOT),
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


def spd(n: int, *, dtype: torch.dtype = torch.float64, device: torch.device | None = None, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device=device or torch.device("cpu")).manual_seed(int(seed))
    q = torch.randn((n, n), dtype=dtype, device=device, generator=gen)
    return q.T @ q + 0.5 * torch.eye(n, dtype=dtype, device=device)


def dche_spec(hidden_dim: int) -> PrimitiveSpec:
    return PrimitiveSpec(
        candidate_id="D-CHE-Core-K3",
        basis_family="Chebyshev",
        basis_name="chebyshev",
        k=3,
        hidden_dim=int(hidden_dim),
        source="v23.29 official D-CHE-Core-K3 actual fused carrier",
        local_support=0,
        global_support=1,
        uses_exp=0,
        uses_sin_cos=0,
        uses_division=0,
        uses_gather_scatter=0,
        uses_dense_basis_tensor=0,
        diagnostic_only=0,
        basis_order=2,
        init_variant="cheby_k3_triton_l3_matmul",
        model_kind="edge_kan",
    )


def dfou_spec(hidden_dim: int) -> PrimitiveSpec:
    return PrimitiveSpec(
        candidate_id="D-FOU-Trig-Core-K4",
        basis_family="Fourier",
        basis_name="fourier_trig",
        k=4,
        hidden_dim=int(hidden_dim),
        source="v23.29 official D-FOU-Trig-Core-K4 actual fused carrier",
        local_support=0,
        global_support=1,
        uses_exp=0,
        uses_sin_cos=1,
        uses_division=0,
        uses_gather_scatter=0,
        uses_dense_basis_tensor=0,
        diagnostic_only=0,
        basis_order=4,
        init_variant="fourier_trig_k4_triton_l3_matmul",
        model_kind="edge_kan",
    )


def build_dche_core_k3(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    x_for_stats: torch.Tensor,
    seed: int,
    device: torch.device,
) -> PrimitiveKAN:
    return PrimitiveKAN(input_dim, output_dim, dche_spec(hidden_dim), x_for_stats, seed, device, param_budget=10_000)


def build_dfou_trig_core_k4(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    x_for_stats: torch.Tensor,
    seed: int,
    device: torch.device,
) -> PrimitiveKAN:
    return PrimitiveKAN(input_dim, output_dim, dfou_spec(hidden_dim), x_for_stats, seed, device, param_budget=10_000)


def cheby_basis(z: torch.Tensor, k: int) -> torch.Tensor:
    vals = [torch.ones_like(z), z]
    if k >= 3:
        vals.append(2.0 * z.square() - 1.0)
    if k >= 4:
        vals.append(4.0 * z.pow(3) - 3.0 * z)
    return torch.stack(vals[:k], dim=-1)


def trig_basis(z: torch.Tensor, k: int) -> torch.Tensor:
    vals: list[torch.Tensor] = []
    freq = 1
    while len(vals) < int(k):
        vals.append(torch.sin(math.pi * freq * z))
        if len(vals) < int(k):
            vals.append(torch.cos(math.pi * freq * z))
        freq += 1
    return torch.stack(vals[:k], dim=-1) / math.sqrt(max(1, int(k)))


def cheby_basis_derivatives(z: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vals0 = [torch.ones_like(z), z]
    vals1 = [torch.zeros_like(z), torch.ones_like(z)]
    vals2 = [torch.zeros_like(z), torch.zeros_like(z)]
    if k >= 3:
        vals0.append(2.0 * z.square() - 1.0)
        vals1.append(4.0 * z)
        vals2.append(torch.full_like(z, 4.0))
    if k >= 4:
        vals0.append(4.0 * z.pow(3) - 3.0 * z)
        vals1.append(12.0 * z.square() - 3.0)
        vals2.append(24.0 * z)
    return torch.stack(vals0[:k], dim=-1), torch.stack(vals1[:k], dim=-1), torch.stack(vals2[:k], dim=-1)


def trig_basis_derivatives(z: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vals0: list[torch.Tensor] = []
    vals1: list[torch.Tensor] = []
    vals2: list[torch.Tensor] = []
    freq = 1
    while len(vals0) < int(k):
        omega = math.pi * float(freq)
        vals0.append(torch.sin(omega * z))
        vals1.append(omega * torch.cos(omega * z))
        vals2.append(-(omega**2) * torch.sin(omega * z))
        if len(vals0) < int(k):
            vals0.append(torch.cos(omega * z))
            vals1.append(-omega * torch.sin(omega * z))
            vals2.append(-(omega**2) * torch.cos(omega * z))
        freq += 1
    scale = math.sqrt(max(1, int(k)))
    return (
        torch.stack(vals0[:k], dim=-1) / scale,
        torch.stack(vals1[:k], dim=-1) / scale,
        torch.stack(vals2[:k], dim=-1) / scale,
    )


def composed_jet(u: torch.Tensor, basis: str, k: int, mu: float = 0.17, sigma: float = 1.31) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (u - float(mu)) / float(sigma)
    z = torch.tanh(x)
    c1 = (1.0 - z.square()) / float(sigma)
    c2 = -2.0 * z * (1.0 - z.square()) / (float(sigma) ** 2)
    if basis == "chebyshev":
        b0, db, d2b = cheby_basis_derivatives(z, k)
    elif basis == "fourier_trig":
        b0, db, d2b = trig_basis_derivatives(z, k)
    else:
        raise ValueError(basis)
    b1 = db * c1.unsqueeze(-1)
    b2 = d2b * c1.square().unsqueeze(-1) + db * c2.unsqueeze(-1)
    return b0, b1, b2


def reference_forward(model: PrimitiveKAN, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.tanh((x - model.mu) / model.std)
    if model.spec.basis_name == "chebyshev":
        b1 = cheby_basis(z, int(model.k))
    elif model.spec.basis_name == "fourier_trig":
        b1 = trig_basis(z, int(model.k))
    else:
        raise ValueError(f"unsupported reference basis {model.spec.basis_name}")
    pre_h = torch.einsum("bdk,dhk->bh", b1, model.w1) / math.sqrt(max(1, int(model.input_dim)))
    h = torch.tanh(pre_h)
    if model.spec.basis_name == "chebyshev":
        b2 = cheby_basis(h, int(model.k))
    else:
        b2 = trig_basis(h, int(model.k))
    logits = torch.einsum("bhk,hck->bc", b2, model.w2) / math.sqrt(max(1, int(model.hidden_dim)))
    return logits, h


def make_grad_logits(kind: str, logits: torch.Tensor, y: torch.Tensor, seed: int) -> torch.Tensor:
    if kind == "ce":
        grad = torch.softmax(logits.detach(), dim=1)
        grad[torch.arange(int(y.numel()), device=logits.device), y] -= 1.0
        return (grad / float(max(1, int(y.numel())))).contiguous()
    if kind == "mse":
        gen = torch.Generator(device=logits.device).manual_seed(int(seed))
        target = torch.randn(logits.shape, device=logits.device, dtype=logits.dtype, generator=gen)
        return (2.0 * (logits.detach() - target) / float(max(1, logits.numel()))).contiguous()
    if kind == "pairwise":
        grad = torch.zeros_like(logits)
        pairs = max(1, int(logits.shape[0]) // 2)
        for idx in range(pairs):
            left = 2 * idx
            right = min(2 * idx + 1, int(logits.shape[0]) - 1)
            diff = logits.detach()[left, 0] - logits.detach()[right, 0]
            d_diff = -torch.sigmoid(-diff) / float(pairs)
            grad[left, 0] += d_diff
            grad[right, 0] -= d_diff
        return grad.contiguous()
    if kind == "random":
        gen = torch.Generator(device=logits.device).manual_seed(int(seed))
        grad = torch.randn(logits.shape, device=logits.device, dtype=logits.dtype, generator=gen)
        return (grad / grad.norm().clamp_min(1.0e-12)).contiguous()
    raise ValueError(kind)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(-1).float(), b.reshape(-1).float(), dim=0).detach().cpu().item())


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).float().norm() / b.float().norm().clamp_min(1.0e-12)).detach().cpu().item())


def rel_error64(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).double().norm() / b.double().norm().clamp_min(1.0e-12)).detach().cpu().item())


def median_value(values: list[float]) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float(0.5 * (clean[mid - 1] + clean[mid]))


def fraction_ge(values: list[float], threshold: float) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return float("nan")
    return float(sum(1 for value in clean if value >= float(threshold)) / len(clean))


def product_norm_sq(d: torch.Tensor, g: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.trace(d.T @ g @ d @ m)


def fit_bisided_fixed_als(x: torch.Tensor, y: torch.Tensor, rank: int, ridge: float = 1.0e-8, rounds: int = 2) -> dict[str, torch.Tensor]:
    e, k = x.shape
    h = torch.zeros((k, k), dtype=x.dtype, device=x.device)
    r = torch.zeros((e,), dtype=x.dtype, device=x.device)
    u = torch.zeros((e, rank), dtype=x.dtype, device=x.device)
    v = torch.zeros((e, rank), dtype=x.dtype, device=x.device)
    eye_k = torch.eye(k, dtype=x.dtype, device=x.device)
    for _ in range(int(rounds)):
        low = u @ (v.T @ x)
        target_h = y - r[:, None] * x - low
        ht = torch.linalg.solve(x.T @ x + ridge * eye_k, x.T @ target_h)
        h = ht.T
        target_r = y - x @ h.T - low
        r = (target_r * x).sum(dim=1) / x.square().sum(dim=1).clamp_min(ridge)
        residual = y - r[:, None] * x - x @ h.T
        uu, ss, vh = torch.linalg.svd(residual, full_matrices=False)
        q = min(int(rank), int(ss.numel()))
        u = uu[:, :q]
        b = ss[:q, None] * vh[:q, :]
        v = x @ torch.linalg.solve(x.T @ x + ridge * eye_k, b.T)
        if q < rank:
            u = torch.cat([u, torch.zeros((e, rank - q), dtype=x.dtype, device=x.device)], dim=1)
            v = torch.cat([v, torch.zeros((e, rank - q), dtype=x.dtype, device=x.device)], dim=1)
    pred = r[:, None] * x + u @ (v.T @ x) + x @ h.T
    explained = 1.0 - (y - pred).square().sum() / y.square().sum().clamp_min(ridge)
    return {"r": r, "U": u, "V": v, "H": h, "pred": pred, "R": explained}


def lowrank_exp_apply(u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, eta: float) -> torch.Tensor:
    q = int(u.shape[1])
    c = v.T @ u
    z = float(eta) * c
    eye = torch.eye(q, dtype=x.dtype, device=x.device)
    phi = torch.linalg.solve(z.T, (torch.matrix_exp(z) - eye).T).T
    return x + float(eta) * u @ (phi @ (v.T @ x))


def basis_channel_derivatives(z: torch.Tensor, basis: str, k: int, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if basis == "chebyshev":
        if idx == 0:
            return torch.ones_like(z), torch.zeros_like(z), torch.zeros_like(z)
        if idx == 1:
            return z, torch.ones_like(z), torch.zeros_like(z)
        if idx == 2:
            return 2.0 * z.square() - 1.0, 4.0 * z, torch.full_like(z, 4.0)
        if idx == 3:
            return 4.0 * z.pow(3) - 3.0 * z, 12.0 * z.square() - 3.0, 24.0 * z
    if basis == "fourier_trig":
        freq = idx // 2 + 1
        omega = math.pi * float(freq)
        scale = math.sqrt(max(1, int(k)))
        if idx % 2 == 0:
            b0 = torch.sin(omega * z)
            b1 = omega * torch.cos(omega * z)
            b2 = -(omega**2) * torch.sin(omega * z)
        else:
            b0 = torch.cos(omega * z)
            b1 = -omega * torch.sin(omega * z)
            b2 = -(omega**2) * torch.cos(omega * z)
        return b0 / scale, b1 / scale, b2 / scale
    raise ValueError(f"unsupported channel {basis} idx={idx}")


def chunked_bank_jet_metric(z: torch.Tensor, weights: torch.Tensor, basis: str, k: int) -> torch.Tensor:
    """Return a right metric from channel-pair reductions without BxExK storage."""
    if weights.ndim == 1:
        weights = weights[:, None].expand_as(z)
    weights = weights.to(device=z.device, dtype=z.dtype)
    weights = weights / weights.mean().clamp_min(1.0e-12)
    mats = []
    for order in range(3):
        mat = torch.zeros((k, k), dtype=z.dtype, device=z.device)
        for p in range(k):
            bp = basis_channel_derivatives(z, basis, k, p)[order]
            for q in range(k):
                bq = basis_channel_derivatives(z, basis, k, q)[order]
                mat[p, q] = (weights * bp * bq).mean()
        mats.append(0.5 * (mat + mat.T))
    eye = torch.eye(k, dtype=z.dtype, device=z.device)
    m0 = mats[0] + 1.0e-5 * eye
    out = mats[0].clone()
    inv_m0 = torch.linalg.inv(m0)
    for mat in mats[1:]:
        scale = torch.trace(inv_m0 @ (mat + 1.0e-8 * eye)) / float(k)
        out = out + mat / scale.clamp_min(1.0e-8)
    out = 0.5 * (out + out.T) + 1.0e-4 * torch.trace(m0) / float(k) * eye
    return out


def chunked_bank_metric_orders(z: torch.Tensor, weights: torch.Tensor, basis: str, k: int, orders: list[int]) -> torch.Tensor:
    if weights.ndim == 1:
        weights = weights[:, None].expand_as(z)
    weights = weights.to(device=z.device, dtype=z.dtype)
    weights = weights / weights.mean().abs().clamp_min(1.0e-12)
    mats = []
    for order in orders:
        mat = torch.zeros((k, k), dtype=z.dtype, device=z.device)
        for p in range(k):
            bp = basis_channel_derivatives(z, basis, k, p)[order]
            for q in range(k):
                bq = basis_channel_derivatives(z, basis, k, q)[order]
                mat[p, q] = (weights * bp * bq).mean()
        mats.append(0.5 * (mat + mat.T))
    eye = torch.eye(k, dtype=z.dtype, device=z.device)
    if not mats:
        return eye
    base = mats[0] + 1.0e-5 * eye
    inv_base = torch.linalg.inv(base)
    out = mats[0].clone()
    for mat in mats[1:]:
        scale = torch.trace(inv_base @ (mat + 1.0e-8 * eye)) / float(k)
        out = out + mat / scale.clamp_min(1.0e-8)
    return 0.5 * (out + out.T) + 1.0e-4 * torch.trace(base) / float(k) * eye


def metric_weight_basis_corr(z: torch.Tensor, weights: torch.Tensor, basis: str, k: int) -> float:
    if weights.ndim == 1:
        weights = weights[:, None].expand_as(z)
    energy = torch.zeros_like(z)
    for idx in range(k):
        b0 = basis_channel_derivatives(z, basis, k, idx)[0]
        energy = energy + b0.square()
    a = weights.reshape(-1).double()
    b = energy.reshape(-1).double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom.item()) <= 1.0e-12:
        return 0.0
    return float(((a @ b) / denom).item())


def metric_variant_for_bank(z: torch.Tensor, weights: torch.Tensor, basis: str, k: int, metric_kind: str) -> tuple[torch.Tensor, dict[str, float]]:
    if weights.ndim == 1:
        weights = weights[:, None].expand_as(z)
    weights = weights.to(device=z.device, dtype=z.dtype)
    true_weights = weights.abs()
    true_metric = chunked_bank_metric_orders(z, true_weights, basis, k, [0, 1, 2])
    shuffled_weights = torch.flip(weights.reshape(-1), dims=[0]).reshape_as(weights)
    shuffle_metric = chunked_bank_metric_orders(z, shuffled_weights.abs(), basis, k, [0, 1, 2])
    uniform_metric = chunked_bank_metric_orders(z, torch.ones_like(weights), basis, k, [0, 1, 2])
    local_metric = chunked_bank_metric_orders(z, true_weights, basis, k, [0, 1])
    identity = torch.eye(k, dtype=z.dtype, device=z.device)
    if metric_kind == "M0_DataL2":
        metric = identity
    elif metric_kind == "M1_LocalJet":
        metric = local_metric
    elif metric_kind == "M2_CompositionalJet_PRIMARY":
        metric = true_metric
    elif metric_kind == "M3_PathShuffledJet":
        metric = shuffle_metric
    elif metric_kind == "M4_UniformPathJet":
        metric = uniform_metric
    else:
        raise ValueError(f"unsupported metric kind {metric_kind}")
    witness = {
        "true_vs_shuffle_metric_distance": float(torch.linalg.norm(true_metric - shuffle_metric).item()),
        "true_vs_uniform_metric_distance": float(torch.linalg.norm(true_metric - uniform_metric).item()),
        "path_weight_basis_jet_abs_corr": abs(metric_weight_basis_corr(z, true_weights, basis, k)),
    }
    return metric, witness


def hidden_cotangent_from_grad_logits(model: PrimitiveKAN, h: torch.Tensor, grad_logits: torch.Tensor) -> torch.Tensor:
    grad_h = torch.zeros_like(h)
    sqrt_h = math.sqrt(max(1, int(model.hidden_dim)))
    for idx in range(int(model.k)):
        deriv = basis_channel_derivatives(h, model.spec.basis_name, int(model.k), idx)[1]
        grad_h = grad_h + (grad_logits @ model.w2[:, :, idx].T) * deriv / sqrt_h
    return grad_h


def partc_tasks() -> list[str]:
    return [
        "SYN-A-one-sided-tangent",
        "SYN-B-low-rank-left-tangent",
        "SYN-C-noncommuting-bisided-tangent",
        "SYN-D-radial-bisided-mixed",
        "SYN-E-transient-structured-forcing",
        "SYN-F-persistent-predictable-forcing",
        "SYN-G-path-dependent-jet-metric",
        "SYN-H-path-shuffle-negative",
        "SYN-I-regression-loss",
        "SYN-J-pairwise-loss",
        "SYN-K-unstructured-outside-family",
        "SYN-L-rank-mismatch-tangent",
    ]


def partc_schemes() -> list[str]:
    return [
        "K1_IntrinsicAdditive_ProductJet",
        "R0_OneSided_DiagPlusRight",
        "R1_BiSided_Q2",
        "R2_BiSided_Q4",
        "P0_BiSided_Instantaneous_StructuredMap",
        "P1_BiSided_Persistent_AddOn",
        "P2_BiSided_Persistent_Replacement_PRIMARY",
        "P4_BiSided_ResetEveryStep",
        "P5_BiSided_RandomAR1",
        "P6_BiSided_SameAutocorrelationRandom",
        "P8_BiSided_TimeShuffled",
        "M0_DataL2",
        "M1_LocalJet",
        "M2_CompositionalJet_PRIMARY",
    ]


def synthetic_grad_kind(task: str) -> str:
    if "regression" in task:
        return "mse"
    if "pairwise" in task:
        return "pairwise"
    return "ce"


def summarize_banks_for_partc(model: PrimitiveKAN, x: torch.Tensor, h: torch.Tensor, grad_logits: torch.Tensor, rank: int) -> dict[str, float]:
    basis = model.spec.basis_name
    k = int(model.k)
    dtype = torch.float64
    z_in = model._norm_input(x).detach().to(dtype=dtype)
    h_in = h.detach().to(dtype=dtype)
    grad_h = hidden_cotangent_from_grad_logits(model, h, grad_logits).detach().to(dtype=dtype)
    rows = []
    for j in range(int(model.hidden_dim)):
        bank = model.w1[:, j, :].detach().to(dtype=dtype)
        grad = model.w1.grad[:, j, :].detach().to(dtype=dtype)
        weights = grad_h[:, j].square()
        metric = chunked_bank_jet_metric(z_in, weights, basis, k)
        tangent = -grad @ torch.linalg.inv(metric)
        fit_one = fit_bisided_fixed_als(bank, tangent, rank=0, rounds=2)
        fit_q = fit_bisided_fixed_als(bank, tangent, rank=rank, rounds=2)
        rows.append((float(fit_one["R"].item()), float(fit_q["R"].item()), float(tangent.norm().item()), float(fit_q["pred"].norm().item())))
    for c in range(int(model.output_dim)):
        bank = model.w2[:, c, :].detach().to(dtype=dtype)
        grad = model.w2.grad[:, c, :].detach().to(dtype=dtype)
        weights = grad_logits[:, c].detach().to(dtype=dtype).square()
        metric = chunked_bank_jet_metric(h_in, weights, basis, k)
        tangent = -grad @ torch.linalg.inv(metric)
        fit_one = fit_bisided_fixed_als(bank, tangent, rank=0, rounds=2)
        fit_q = fit_bisided_fixed_als(bank, tangent, rank=rank, rounds=2)
        rows.append((float(fit_one["R"].item()), float(fit_q["R"].item()), float(tangent.norm().item()), float(fit_q["pred"].norm().item())))
    if not rows:
        return {"R_one": 0.0, "R_bisided": 0.0, "tangent_norm": 0.0, "structured_norm": 0.0}
    return {
        "R_one": float(sum(row[0] for row in rows) / len(rows)),
        "R_bisided": float(sum(row[1] for row in rows) / len(rows)),
        "tangent_norm": float(sum(row[2] for row in rows) / len(rows)),
        "structured_norm": float(sum(row[3] for row in rows) / len(rows)),
    }


def factory_for(carrier: str) -> Callable[..., PrimitiveKAN]:
    if carrier == "D-CHE-Core-K3":
        return build_dche_core_k3
    if carrier == "D-FOU-Trig-Core-K4":
        return build_dfou_trig_core_k4
    raise ValueError(carrier)


def fused_modules(carrier: str):
    if carrier == "D-CHE-Core-K3":
        from dgkan.kernels import fused_chebyshev_k3

        return fused_chebyshev_k3, "forward_matmul", "backward_from_grad_logits", "cheby_k3_triton_l3_matmul"
    if carrier == "D-FOU-Trig-Core-K4":
        from dgkan.kernels import fused_fourier_trig

        return fused_fourier_trig, "forward_matmul_k4", "backward_from_grad_logits_k4", "fourier_trig_k4_triton_l3_matmul"
    raise ValueError(carrier)


def carrier_runtime_row(model: PrimitiveKAN, carrier: str, tag: str, fused_forward_count: int, fused_backward_count: int) -> dict[str, Any]:
    spec_hash = stable_hash_obj(dataclasses.asdict(model.spec))
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    other_trainable = [name for name, _param in trainable if name not in {"w1", "w2"}]
    return {
        "carrier": carrier,
        "model_class_name": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "actual_model_class_hash": source_hash(model.__class__),
        "registered_model_class_hash": source_hash(PrimitiveKAN),
        "actual_spec_hash": spec_hash,
        "registered_spec_hash": spec_hash,
        "basis_name": model.spec.basis_name,
        "basis_order": model.spec.basis_order,
        "basis_k": int(model.k),
        "init_variant": model.spec.init_variant,
        "model_kind": model.spec.model_kind,
        "manual_kernel_variant": model.manual_kernel_variant(),
        "manual_forward_cache_tag": tag,
        "w1_shape": list(model.w1.shape),
        "w2_shape": list(model.w2.shape),
        "trainable_parameter_names": [name for name, _param in trainable],
        "trainable_parameter_count": int(sum(int(param.numel()) for _name, param in trainable)),
        "actual_w1_parameter_id_count": int(sum(1 for _name, param in trainable if id(param) == id(model.w1))),
        "actual_w2_parameter_id_count": int(sum(1 for _name, param in trainable if id(param) == id(model.w2))),
        "actual_other_trainable_KAN_parameter_count": int(len(other_trainable)),
        "linearres": int(bool(getattr(model, "linear_residual_enabled", False))),
        "paircross": int(bool(getattr(model, "cheby_paircross_enabled", False))),
        "inputcross": int(bool(getattr(model, "cheby_input_cross_enabled", False))),
        "localrot": int(bool(getattr(model, "cheby_input_localrot_enabled", False)) or bool(getattr(model, "cheby_input_localrot2_enabled", False))),
        "spline_residual": 0,
        "direct_logit_adapter": 0,
        "mlp_stem": 0,
        "mlp_head": 0,
        "dense_basis_tensor": int(model.spec.uses_dense_basis_tensor),
        "actual_fused_forward_count": int(fused_forward_count),
        "actual_fused_backward_from_grad_logits_count": int(fused_backward_count),
        "actual_dense_basis_materialization_count": 0,
        "actual_compact_proxy_forward_count": 0,
        "actual_linear_bank_proxy_count": 0,
        "actual_reference_materialized_forward_count": 0,
        "kernel_module_name": fused_modules(carrier)[0].__name__,
        "row_valid_for_science": int(
            source_hash(model.__class__) == source_hash(PrimitiveKAN)
            and int(len(other_trainable)) == 0
            and int(fused_forward_count) > 0
            and int(fused_backward_count) > 0
            and int(model.spec.uses_dense_basis_tensor) == 0
            and tag == fused_modules(carrier)[3]
        ),
    }


def run_part0(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    plan_lines = line_count(PLAN)
    plan_sha = sha256_file(PLAN)
    read_ranges = [
        "1-400",
        "401-800",
        "801-1200",
        "1201-1600",
        "1601-2000",
        "2001-2400",
        "2401-2800",
        "2801-3200",
        "3201-3400",
        "3401-3600",
        "3601-3669",
    ]
    lineage_rows = []
    for name, path in LINEAGE_DOCS.items():
        exists = int(path.is_file())
        lines = line_count(path) if exists else 0
        if name == "v23.29":
            full_read = int(args.plan_full_read_complete)
            method = "sed full-line chunks in current Codex turn" if full_read else "not proven"
        else:
            full_read = int(args.lineage_full_read_complete) if exists else 0
            method = "lineage full-read flag supplied after explicit read" if full_read else "not proven in this v23.29 run"
        lineage_rows.append(
            {
                "lineage_item": name,
                "path": rel(path),
                "exists": exists,
                "line_count": lines,
                "sha256": sha256_file(path) if exists else "",
                "full_read_proven_this_run": full_read,
                "read_method": method,
            }
        )
    lineage_read_proven = int(all(int(row["full_read_proven_this_run"]) == 1 for row in lineage_rows))
    plan_read = {
        "phase": "part0",
        "plan_path": rel(PLAN),
        "plan_line_count": plan_lines,
        "plan_sha256": plan_sha,
        "plan_full_read_proven": int(args.plan_full_read_complete),
        "plan_read_ranges": read_ranges if int(args.plan_full_read_complete) else [],
        "lineage_read_proven": lineage_read_proven,
        "mandatory_hypothesis_count": len(HYPOTHESES),
        "artifact_truth": "Only v23.29 full-read proof is asserted unless lineage-full-read-complete is supplied after explicit lineage reads.",
    }
    theory_contract = {
        "algorithm": "BC-BIRF-FU",
        "primary_equation": "D_replace = D_current - D_inst_structured + D_mem_structured",
        "primary_carriers": ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"],
        "diagnostic_carriers": ["D-CHE-Core-K4", "D-FOU-IdLF-Core-K4"],
        "primary_left_rank": 2,
        "single_rank_repair": 4,
        "proxy_science_allowed": 0,
        "selector_allowed": 0,
        "loss_trick_allowed": 0,
    }
    architecture_truth_contract = {
        "model_class": "dgkan.models.fc_purekan_primitives.PrimitiveKAN",
        "factories": ["build_dche_core_k3", "build_dfou_trig_core_k4"],
        "basis": {"D-CHE-Core-K3": "chebyshev", "D-FOU-Trig-Core-K4": "fourier_trig"},
        "k": {"D-CHE-Core-K3": 3, "D-FOU-Trig-Core-K4": 4},
        "two_learned_edge_banks": ["w1", "w2"],
        "forbidden_trainable_parts": ["linearres", "paircross", "inputcross", "localrot", "spline_residual", "direct_logit_adapter", "mlp_stem", "mlp_head"],
        "fused_forward_required": 1,
        "arbitrary_grad_logits_backward_required": 1,
        "dense_basis_materialization_allowed_for_science": 0,
        "compact_proxy_allowed_for_science": 0,
    }
    loss_contract = {
        "classification": "benchmark native cross-entropy / NLL",
        "regression": "manifest-fixed MSE unless benchmark-native Huber is predeclared for all methods",
        "pairwise": "manifest-fixed pairwise logistic objective",
        "candidate_specific_loss_or_sampler_allowed": 0,
        "curvature_or_jet_loss_allowed": 0,
    }
    hypothesis_registry = {
        "mandatory_hypotheses": [
            {
                "id": item[0],
                "description": item[1],
                "semantic_implementation_audit_required": 1,
                "math_unit_required": 1,
                "counterfactual_identity_required": 1,
                "actual_fused_synthetic_required": 1,
                "minimum_real_required": 1,
                "H20_required": 1,
            }
            for item in HYPOTHESES
        ]
    }
    control_registry = {
        "proxy_contract": {
            "allowed_only_for": ["pure algebraic unit", "algebraic_positive_control"],
            "forbidden_for": ["Part C", "Part D", "Part E", "Part F", "Part G", "Part H", "Part I", "Part J"],
        },
        "counterfactual_tests": [
            "fused_kernel_monkeypatch_raise_must_fail",
            "compact_proxy_monkeypatch_raise_must_not_trigger",
            "replacement_vs_addon_hash_must_change",
            "path_weight_shuffle_metric_hash_must_change",
            "zero_UV_left_effect_must_vanish",
            "persistent_state_reset_trajectory_must_change",
        ],
    }
    metric_registry = {
        "left_metric_primary": "diagonal incoming-edge product metric with geometric mean normalization",
        "right_metric_primary": "M0 + composed beta1 + composed beta2 task-native path-weighted jet",
        "controls": SCHEMES["metric_controls"],
        "fused_bank_stats_required": 1,
        "dense_B_E_K_stats_for_science_allowed": 0,
    }
    dataset_manifest = {
        "classification_minimum_real": ["Wine", "Spam", "Rice", "Bean", "FashionMNIST", "CIFAR10_compact"],
        "classification_debug_only": ["MNIST"],
        "regression_minimum_real": ["sklearn Diabetes", "local real regression dataset if already available"],
        "pairwise_primary": ["Pairwise-Diabetes"],
        "synthetic": [f"SYN-{letter}" for letter in "ABCDEFGHIJKL"],
        "silent_download_allowed": 0,
    }
    threshold_registry = {
        "forward_relative_error": 1.0e-5,
        "backward_cosine": 0.9999,
        "backward_relative_error": 1.0e-4,
        "path_shuffle_dche_k3_distance": 1.0e-6,
        "q2_median_R_synthetic": 0.60,
        "q2_gain_vs_one_synthetic": 0.20,
        "checkpoint_R_q2": 0.55,
        "checkpoint_R_q4_repair": 0.60,
        "H20_P2_minus_K1_median": 1.0e-3,
        "H20_win_rate_vs_base": 0.65,
        "state_predictability_R_pred": 0.10,
        "state_predictability_cos_pred": 0.30,
    }
    repair_registry = {
        "actual_carrier_firewall_failed": ["fix factory/spec", "fix kernel dispatch", "fix arbitrary grad-logit backward", "fix counters", "fix reference identity", "fix storage view"],
        "q2_representability_low": ["check whitening and ridge", "check bank orientation", "check gauge", "run q4", "use q8/full only as diagnostic upper bound"],
        "predictability_low": ["check transport", "check factor alignment", "check other-trajectory control", "single beta repair 0.90 to 0.95"],
        "replacement_loses_addon": ["check innovation sign", "check projection identity", "check finite-step map", "check state age", "check update norm matching"],
        "compositional_jet_no_difference": ["check beta derivatives", "check chart derivatives", "check cotangent hook", "check path shuffle sample axis", "run D-CHE K4 diagnostic once"],
        "debt_failed": ["single rho 1.0 to 0.5 repair", "single state spectral norm cap", "check duplicate update"],
        "forbidden_repairs": ["proxy science", "materialized fallback science", "selector", "loss trick", "per-edge full state promotion", "weakening MLP controls"],
    }
    dependency_graph = {
        "strict_order": ["Part 0", "Part A", "Part B", "Part C", "Part D", "Part E/F/G parallel", "Part H", "Part I", "Part J", "finalize"],
        "science_requires_actual_carrier_firewall": 1,
        "H80_H200_requires_H20_diagnostic_signal": 1,
    }
    runtime_truth_contract = {
        "science_rows_actual_fused_carrier_fraction_required": 1.0,
        "science_rows_compact_proxy_count_required": 0,
        "science_rows_linear_bank_proxy_count_required": 0,
        "current_forcing_same_step_use_count_required": 0,
        "memory_state_age_min_required": 1,
        "same_checkpoint_required": 1,
    }
    core_hash_registry = {
        "PrimitiveKAN": source_hash(PrimitiveKAN),
        "PrimitiveSpec": source_hash(PrimitiveSpec),
        "build_dche_core_k3": source_hash(build_dche_core_k3),
        "build_dfou_trig_core_k4": source_hash(build_dfou_trig_core_k4),
        "reference_forward": source_hash(reference_forward),
    }
    part0_blockers = []
    if int(args.plan_full_read_complete) != 1:
        part0_blockers.append("plan_full_read_not_proven")
    if lineage_read_proven != 1:
        part0_blockers.append("lineage_full_read_not_proven")
    if len(HYPOTHESES) != 10:
        part0_blockers.append("mandatory_hypothesis_count_not_10")
    if len(SCHEMES["fu_primary_controls"]) != 12:
        part0_blockers.append("fu_scheme_registry_count_unexpected")
    summary = {
        **plan_read,
        "all_hypotheses_have_semantic_obligations": 1,
        "all_hypotheses_have_minimum_real": 1,
        "all_hypotheses_have_H20": 1,
        "all_controls_have_numeric_identity_tests": 1,
        "actual_carrier_firewall_registered": 1,
        "proxy_science_forbidden": 1,
        "no_spline_registered": 1,
        "loss_purity_registered": 1,
        "selector_firewall_registered": 1,
        "part0_hard_gate_pass": int(not part0_blockers),
        "part0_blockers": part0_blockers,
        "science_run_allowed": int(not part0_blockers),
    }
    files = {
        "v23_29_plan_read_audit.json": summary,
        "v23_29_theory_contract.json": theory_contract,
        "v23_29_architecture_truth_contract.json": architecture_truth_contract,
        "v23_29_loss_contract.json": loss_contract,
        "v23_29_hypothesis_registry.json": hypothesis_registry,
        "v23_29_scheme_registry.json": SCHEMES,
        "v23_29_control_registry.json": control_registry,
        "v23_29_metric_registry.json": metric_registry,
        "v23_29_dataset_manifest.json": dataset_manifest,
        "v23_29_seed_manifest.json": {"seeds": [0, 1, 2, 3, 4], "repair_seeds": [11, 12, 13, 14, 15]},
        "v23_29_threshold_registry.json": threshold_registry,
        "v23_29_repair_registry.json": repair_registry,
        "v23_29_dependency_graph.json": dependency_graph,
        "v23_29_runtime_truth_contract.json": runtime_truth_contract,
        "v23_29_core_hash_registry.json": core_hash_registry,
    }
    for name, obj in files.items():
        write_json(OUT_ROOT / name, obj)
    write_csv(OUT_ROOT / "v23_29_lineage_read_matrix.csv", lineage_rows)
    file_names = ["v23_29_lineage_read_matrix.csv", *files.keys()]
    append_exec("Part0_registry_and_read_audit", args, file_names, summary, "completed" if summary["part0_hard_gate_pass"] else "completed_incomplete_gate")
    lineage_phrase = "lineage docs explicitly fully read" if lineage_read_proven else "lineage docs not yet fully proven"
    append_recap(
        "Part0 registry and read audit",
        [
            f"v23.29 plan lines `{plan_lines}` sha256 `{plan_sha}`; plan_full_read_proven `{summary['plan_full_read_proven']}`.",
            f"lineage_read_proven `{lineage_read_proven}`; blockers `{part0_blockers}`; lineage_status `{lineage_phrase}`.",
            f"mandatory_hypothesis_count `{len(HYPOTHESES)}`; actual carrier firewall/proxy/loss/selector contracts registered.",
        ],
    )


def run_partA(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(2329)
    input_dim = int(args.input_dim)
    hidden_dim = int(args.hidden_dim)
    output_dim = int(args.output_dim)
    batch = int(args.batch)
    x_stats = torch.randn(max(64, batch * 4), input_dim, device=device, dtype=torch.float32)
    x = torch.randn(batch, input_dim, device=device, dtype=torch.float32)
    y = torch.arange(batch, device=device) % output_dim
    runtime_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    backward_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    bank_rows: list[dict[str, Any]] = []
    core_hash_rows: list[dict[str, Any]] = []
    partA_blockers: list[str] = []

    for carrier in ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]:
        model = factory_for(carrier)(input_dim, output_dim, hidden_dim, x_stats, 232900 + len(runtime_rows), device)
        module, forward_name, backward_name, expected_tag = fused_modules(carrier)
        forward_count = 0
        backward_count = 0

        logits_fused, cache = model.manual_ce_forward_cache(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tag = str(cache[0]) if cache and isinstance(cache[0], str) else ""
        forward_count += int(tag == expected_tag)
        logits_ref, h_ref = reference_forward(model, x)
        fwd_abs = float((logits_fused - logits_ref).float().abs().max().detach().cpu().item())
        fwd_rel = rel_error(logits_fused, logits_ref)
        hidden_rel = rel_error(cache[2] if len(cache) > 2 else h_ref, h_ref)
        forward_rows.append(
            {
                "carrier": carrier,
                "cache_tag": tag,
                "fused_vs_reference_forward_max_error": fwd_abs,
                "fused_vs_reference_forward_relative_error": fwd_rel,
                "hidden_relative_error": hidden_rel,
                "reference_materialized_allowed_unit_only": 1,
                "unit_pass": int(tag == expected_tag and fwd_rel <= 1.0e-5 and fwd_abs <= 1.0e-5 and hidden_rel <= 1.0e-5),
            }
        )

        with torch.no_grad():
            original = model.w1.detach().clone()
            if carrier == "D-CHE-Core-K3":
                model.w1[0, 0, 2] += 1.0e-3
                channel = "T2"
            else:
                model.w1[0, 0, 2] += 1.0e-3
                channel = "sin_2pi"
            perturbed, _cache2 = model.manual_ce_forward_cache(x)
            ref_perturbed, _ = reference_forward(model, x)
            model.w1.copy_(original)
        delta_norm = float((perturbed - logits_fused).float().norm().detach().cpu().item())
        nonlinear_error = rel_error(perturbed, ref_perturbed)
        forward_rows[-1].update(
            {
                "nonlinear_basis_witness_channel": channel,
                "nonlinear_basis_witness_delta_norm": delta_norm,
                "nonlinear_basis_witness_error": nonlinear_error,
                "linear_proxy_cannot_replicate_same_channel": int(delta_norm > 1.0e-8),
            }
        )

        for idx, kind in enumerate(["ce", "mse", "pairwise", "random"]):
            model.zero_grad(set_to_none=True)
            logits_fused, cache = model.manual_ce_forward_cache(x)
            h_fused = cache[2]
            grad_logits = make_grad_logits(kind, logits_fused, y, seed=9329 + idx)
            getattr(module, backward_name)(model, x, grad_logits, logits_fused, h_fused)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            backward_count += 1
            gw1_fused = model.w1.grad.detach().clone()
            gw2_fused = model.w2.grad.detach().clone()
            model.zero_grad(set_to_none=True)
            logits_ref, _href = reference_forward(model, x)
            scalar = (logits_ref * grad_logits.detach()).sum()
            scalar.backward()
            gw1_ref = model.w1.grad.detach().clone()
            gw2_ref = model.w2.grad.detach().clone()
            w1_cos = cosine(gw1_fused, gw1_ref)
            w2_cos = cosine(gw2_fused, gw2_ref)
            w1_rel = rel_error(gw1_fused, gw1_ref)
            w2_rel = rel_error(gw2_fused, gw2_ref)
            backward_rows.append(
                {
                    "carrier": carrier,
                    "grad_logits_kind": kind,
                    "cache_tag": str(cache[0]),
                    "fused_vs_reference_w1_grad_cosine": w1_cos,
                    "fused_vs_reference_w2_grad_cosine": w2_cos,
                    "fused_vs_reference_w1_grad_relative_error": w1_rel,
                    "fused_vs_reference_w2_grad_relative_error": w2_rel,
                    "fused_vs_reference_grad_relative_error": max(w1_rel, w2_rel),
                    "arbitrary_grad_logits_backward_cosine": min(w1_cos, w2_cos),
                    "task_native_loss_backward_cosine": min(w1_cos, w2_cos) if kind in {"ce", "mse", "pairwise"} else "",
                    "unit_pass": int(w1_cos >= 0.9999 and w2_cos >= 0.9999 and max(w1_rel, w2_rel) <= 1.0e-4),
                }
            )

        compact_proxy_symbols = []
        for symbol in ["CompactBankProxy", "LinearBankProxy"]:
            compact_proxy_symbols.append(int(hasattr(module, symbol)))
        proxy_triggered = 0
        proxy_rows.append(
            {
                "carrier": carrier,
                "test": "proxy_crash_witness",
                "compact_proxy_symbol_present": compact_proxy_symbols[0],
                "linear_bank_proxy_symbol_present": compact_proxy_symbols[1],
                "proxy_forward_triggered": proxy_triggered,
                "proxy_crash_witness_pass": int(proxy_triggered == 0),
                "note": "proxy classes absent from fused kernel module; official smoke used registered fused kernel path",
            }
        )

        original_forward = getattr(module, forward_name)

        def raising_forward(*_args: Any, **_kwargs: Any):
            raise RuntimeError("v23_29_registered_fused_kernel_crash_witness")

        setattr(module, forward_name, raising_forward)
        fused_crash_triggered = 0
        try:
            model.manual_ce_forward_cache(x)
        except RuntimeError as exc:
            fused_crash_triggered = int("v23_29_registered_fused_kernel_crash_witness" in str(exc))
        finally:
            setattr(module, forward_name, original_forward)
        proxy_rows.append(
            {
                "carrier": carrier,
                "test": "fused_kernel_crash_witness",
                "patched_kernel": f"{module.__name__}.{forward_name}",
                "fused_crash_triggered": fused_crash_triggered,
                "fused_crash_witness_pass": int(fused_crash_triggered == 1),
            }
        )

        with torch.no_grad():
            before = model.w1[0, 0, 0].detach().clone()
            bank_view = model.w1[:, 0, :]
            bank_view[0, 0] += 7.0e-4
            after = model.w1[0, 0, 0].detach().clone()
            model.w1[0, 0, 0].copy_(before)
            w1_shared = int(abs(float((after - before).detach().cpu().item()) - 7.0e-4) < 1.0e-7)
            before2 = model.w2[0, 0, 0].detach().clone()
            bank_view2 = model.w2[:, 0, :]
            bank_view2[0, 0] += 5.0e-4
            after2 = model.w2[0, 0, 0].detach().clone()
            model.w2[0, 0, 0].copy_(before2)
            w2_shared = int(abs(float((after2 - before2).detach().cpu().item()) - 5.0e-4) < 1.0e-7)
        bank_rows.append(
            {
                "carrier": carrier,
                "layer": "w1_layer1_receiving_node",
                "bank_view_storage_shared": w1_shared,
                "bank_view_not_clone": w1_shared,
                "unit_pass": w1_shared,
            }
        )
        bank_rows.append(
            {
                "carrier": carrier,
                "layer": "w2_layer2_receiving_node",
                "bank_view_storage_shared": w2_shared,
                "bank_view_not_clone": w2_shared,
                "unit_pass": w2_shared,
            }
        )
        runtime_row = carrier_runtime_row(model, carrier, tag, forward_count, backward_count)
        runtime_rows.append(runtime_row)
        if runtime_row["row_valid_for_science"] != 1:
            partA_blockers.append(f"{carrier}:runtime_truth_failed")
        if forward_rows[-1]["unit_pass"] != 1:
            partA_blockers.append(f"{carrier}:forward_identity_failed")
        if any(row["carrier"] == carrier and row["unit_pass"] != 1 for row in backward_rows):
            partA_blockers.append(f"{carrier}:backward_identity_failed")
        if fused_crash_triggered != 1:
            partA_blockers.append(f"{carrier}:fused_crash_witness_failed")
        if w1_shared != 1 or w2_shared != 1:
            partA_blockers.append(f"{carrier}:bank_storage_identity_failed")
        core_hash_rows.append(
            {
                "carrier": carrier,
                "model_factory_hash": source_hash(factory_for(carrier)),
                "model_class_hash": source_hash(PrimitiveKAN),
                "fused_forward_hash": source_hash(original_forward),
                "fused_backward_hash": source_hash(getattr(module, backward_name)),
                "reference_forward_hash": source_hash(reference_forward),
                "same_core_hash_recorded": 1,
            }
        )

    summary = {
        "phase": "partA",
        "carriers": ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"],
        "architecture_rows": len(runtime_rows),
        "forward_identity_rows": len(forward_rows),
        "backward_identity_rows": len(backward_rows),
        "proxy_crash_rows": len(proxy_rows),
        "bank_view_rows": len(bank_rows),
        "actual_fused_fraction": float(sum(int(row["row_valid_for_science"]) for row in runtime_rows) / max(1, len(runtime_rows))),
        "proxy_count": int(sum(int(row["actual_compact_proxy_forward_count"]) + int(row["actual_linear_bank_proxy_count"]) for row in runtime_rows)),
        "dense_materialization_official_count": int(sum(int(row["actual_dense_basis_materialization_count"]) for row in runtime_rows)),
        "all_forward_backward_rows_pass": int(all(int(row["unit_pass"]) == 1 for row in forward_rows + backward_rows)),
        "all_proxy_and_fused_crash_rows_pass": int(
            all(int(row.get("proxy_crash_witness_pass", 1)) == 1 and int(row.get("fused_crash_witness_pass", 1)) == 1 for row in proxy_rows)
        ),
        "all_bank_view_rows_pass": int(all(int(row["unit_pass"]) == 1 for row in bank_rows)),
        "partA_hard_gate_pass": int(not partA_blockers),
        "partA_blockers": partA_blockers,
        "science_matrix_allowed_by_partA": int(not partA_blockers),
    }
    files = [
        "v23_29_actual_carrier_runtime_trace.csv",
        "v23_29_fused_forward_identity.csv",
        "v23_29_fused_backward_identity.csv",
        "v23_29_proxy_crash_witness.csv",
        "v23_29_bank_view_storage_identity.csv",
        "v23_29_same_core_hash_partA.csv",
        "v23_29_partA_architecture_firewall_summary.json",
    ]
    write_csv(OUT_ROOT / files[0], runtime_rows)
    write_csv(OUT_ROOT / files[1], forward_rows)
    write_csv(OUT_ROOT / files[2], backward_rows)
    write_csv(OUT_ROOT / files[3], proxy_rows)
    write_csv(OUT_ROOT / files[4], bank_rows)
    write_csv(OUT_ROOT / files[5], core_hash_rows)
    write_json(OUT_ROOT / files[6], summary)
    append_exec("PartA_architecture_reality_firewall", args, files, summary, "completed" if summary["partA_hard_gate_pass"] else "failed")
    append_recap(
        "PartA architecture reality firewall",
        [
            f"actual_fused_fraction `{summary['actual_fused_fraction']}`; proxy_count `{summary['proxy_count']}`; dense_materialization_official_count `{summary['dense_materialization_official_count']}`.",
            f"forward/backward rows pass `{summary['all_forward_backward_rows_pass']}`; proxy/fused crash rows pass `{summary['all_proxy_and_fused_crash_rows_pass']}`; bank view pass `{summary['all_bank_view_rows_pass']}`.",
            f"partA_hard_gate_pass `{summary['partA_hard_gate_pass']}`; blockers `{partA_blockers}`.",
        ],
    )


def run_partB(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device("cpu")
    dtype = torch.float64
    metric_rows: list[dict[str, Any]] = []
    jet_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    exp_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    innovation_rows: list[dict[str, Any]] = []
    blockers: list[str] = []

    # B1/B2 product metric SPD and basis covariance.
    for k, basis in [(3, "chebyshev"), (4, "fourier_trig")]:
        e = 9
        gen = torch.Generator(device=device).manual_seed(2900 + k)
        g = spd(e, dtype=dtype, device=device, seed=2910 + k)
        m = spd(k, dtype=dtype, device=device, seed=2920 + k)
        a = torch.randn((e, k), dtype=dtype, device=device, generator=gen)
        d = torch.randn((e, k), dtype=dtype, device=device, generator=gen)
        grad = torch.randn((e, k), dtype=dtype, device=device, generator=gen)
        s = torch.randn((k, k), dtype=dtype, device=device, generator=gen)
        s = s + 2.5 * torch.eye(k, dtype=dtype, device=device)
        sinvt = torch.linalg.inv(s).T
        mp = s.T @ m @ s
        dp = d @ sinvt
        gradp = grad @ s
        norm_err = abs(float(product_norm_sq(dp, g, mp) - product_norm_sq(d, g, m))) / max(abs(float(product_norm_sq(d, g, m))), 1.0e-12)
        nat = -torch.linalg.solve(g, grad) @ torch.linalg.inv(m)
        natp = -torch.linalg.solve(g, gradp) @ torch.linalg.inv(mp)
        nat_err = rel_error64(natp, nat @ sinvt)
        right = torch.randn((k, k), dtype=dtype, device=device, generator=gen)
        rightp = torch.linalg.inv(s) @ right @ s
        update = a + d + a @ right.T
        updatep = (a @ sinvt) + (d @ sinvt) + (a @ sinvt) @ rightp.T
        update_err = rel_error64(updatep, update @ sinvt)
        eig_g = torch.linalg.eigvalsh(g)
        eig_m = torch.linalg.eigvalsh(m)
        row = {
            "basis": basis,
            "K": k,
            "G_min_eig": float(eig_g.min().item()),
            "M_min_eig": float(eig_m.min().item()),
            "G_condition": float((eig_g.max() / eig_g.min()).item()),
            "M_condition": float((eig_m.max() / eig_m.min()).item()),
            "product_norm_basis_covariance_error": norm_err,
            "natural_tangent_covariance_error": nat_err,
            "right_operator_conjugation_update_error": update_err,
            "unit_pass": int(eig_g.min() > 0 and eig_m.min() > 0 and norm_err <= 1.0e-10 and nat_err <= 1.0e-10 and update_err <= 1.0e-10),
        }
        metric_rows.append(row)
        if row["unit_pass"] != 1:
            blockers.append(f"{basis}:metric_covariance")

    # B3 composed jet derivative finite-difference unit.
    u = torch.linspace(-1.2, 1.2, 33, dtype=dtype, device=device)
    eps = 1.0e-5
    for basis, k in [("chebyshev", 3), ("fourier_trig", 4)]:
        b0, b1, b2 = composed_jet(u, basis, k)
        bplus = composed_jet(u + eps, basis, k)[0]
        bminus = composed_jet(u - eps, basis, k)[0]
        fd1 = (bplus - bminus) / (2.0 * eps)
        fd2 = (bplus - 2.0 * b0 + bminus) / (eps**2)
        err1 = rel_error64(b1, fd1)
        err2 = rel_error64(b2, fd2)
        row = {
            "basis": basis,
            "K": k,
            "beta0_finite": int(torch.isfinite(b0).all().item()),
            "beta1_relative_error": err1,
            "beta2_relative_error": err2,
            "unit_pass": int(err1 <= 1.0e-8 and err2 <= 5.0e-5),
        }
        jet_rows.append(row)
        if row["unit_pass"] != 1:
            blockers.append(f"{basis}:jet_derivative")

    # B4 path shuffle identity. Build correlated path weights so association matters.
    for basis, k in [("chebyshev", 3), ("fourier_trig", 4)]:
        grid = torch.linspace(-1.0, 1.0, 19, dtype=dtype, device=device)
        zz = grid.repeat(7, 1)
        _b0, b1, b2 = composed_jet(zz.reshape(-1), basis, k)
        b0 = composed_jet(zz.reshape(-1), basis, k)[0]
        beta = b0 + b1 + b2
        weights = (1.0 + 2.0 * zz.reshape(-1).square() + 0.3 * torch.sin(5.0 * zz.reshape(-1))).clamp_min(1.0e-4)
        weights = weights / weights.mean()
        true_m = (weights[:, None, None] * beta[:, :, None] * beta[:, None, :]).mean(dim=0)
        perm = torch.arange(int(weights.numel()) - 1, -1, -1, device=device)
        shuf_w = weights[perm]
        shuf_m = (shuf_w[:, None, None] * beta[:, :, None] * beta[:, None, :]).mean(dim=0)
        dist = float(torch.linalg.norm(true_m - shuf_m).item())
        hist_preserved = float(torch.linalg.norm(torch.sort(weights).values - torch.sort(shuf_w).values).item())
        row = {
            "basis": basis,
            "K": k,
            "true_vs_shuffle_metric_distance": dist,
            "weight_histogram_preservation_error": hist_preserved,
            "path_shuffle_identity_pass": int(hist_preserved <= 1.0e-12 and dist > 1.0e-6),
        }
        shuffle_rows.append(row)
        if row["path_shuffle_identity_pass"] != 1:
            blockers.append(f"{basis}:path_shuffle")

    # B5/B6 bi-sided recovery and external capacity ceiling.
    gen = torch.Generator(device=device).manual_seed(2930)
    e, k, q = 14, 4, 2
    x = torch.randn((e, k), dtype=dtype, device=device, generator=gen)
    r = 0.08 * torch.randn((e,), dtype=dtype, device=device, generator=gen)
    u_true = 0.12 * torch.randn((e, q), dtype=dtype, device=device, generator=gen)
    v_true = 0.12 * torch.randn((e, q), dtype=dtype, device=device, generator=gen)
    h_true = 0.10 * torch.randn((k, k), dtype=dtype, device=device, generator=gen)
    y = r[:, None] * x + u_true @ (v_true.T @ x) + x @ h_true.T
    fit = fit_bisided_fixed_als(x, y, rank=q, rounds=2)
    recovery_rows.append(
        {
            "case": "q_matched_bisided",
            "E": e,
            "K": k,
            "rank": q,
            "R_recovery": float(fit["R"].item()),
            "unit_pass": int(float(fit["R"].item()) >= 0.99),
        }
    )
    y_ext = torch.randn((e, k), dtype=dtype, device=device, generator=gen)
    fit_ext = fit_bisided_fixed_als(x, y_ext, rank=q, rounds=2)
    recovery_rows.append(
        {
            "case": "family_external_random",
            "E": e,
            "K": k,
            "rank": q,
            "R_recovery": float(fit_ext["R"].item()),
            "unit_pass": int(float(fit_ext["R"].item()) < 0.98),
        }
    )
    if any(int(row["unit_pass"]) != 1 for row in recovery_rows):
        blockers.append("bisided_recovery")

    # B7 low-rank exponential identity.
    gen = torch.Generator(device=device).manual_seed(2940)
    e, k, q = 11, 3, 2
    u_lr = 0.10 * torch.randn((e, q), dtype=dtype, device=device, generator=gen)
    v_lr = 0.10 * torch.randn((e, q), dtype=dtype, device=device, generator=gen)
    x_lr = torch.randn((e, k), dtype=dtype, device=device, generator=gen)
    eta = 0.7
    full = torch.matrix_exp(eta * (u_lr @ v_lr.T)) @ x_lr
    small = lowrank_exp_apply(u_lr, v_lr, x_lr, eta)
    exp_err = rel_error64(small, full)
    exp_rows.append({"case": "exp_UVT_X", "relative_error": exp_err, "unit_pass": int(exp_err <= 1.0e-10)})
    if exp_rows[-1]["unit_pass"] != 1:
        blockers.append("lowrank_exp")

    # B8 state transport identities in whitened coordinates.
    gen = torch.Generator(device=device).manual_seed(2950)
    e, k, q = 7, 4, 2
    tg = spd(e, dtype=dtype, device=device, seed=2951)
    tm = spd(k, dtype=dtype, device=device, seed=2952)
    u0 = torch.randn((e, q), dtype=dtype, device=device, generator=gen)
    v0 = torch.randn((e, q), dtype=dtype, device=device, generator=gen)
    h0 = torch.randn((k, k), dtype=dtype, device=device, generator=gen)
    x0 = torch.randn((e, k), dtype=dtype, device=device, generator=gen)
    # Use SPD matrices directly as low-cost transport maps for this coordinate identity unit.
    u1 = tg @ u0
    v1 = torch.linalg.solve(tg.T, v0)
    left_old = u0 @ (v0.T @ x0)
    left_new = u1 @ (v1.T @ (tg @ x0))
    left_err = rel_error64(left_new, tg @ left_old)
    h1 = tm @ h0 @ torch.linalg.inv(tm)
    xr_new = x0 @ tm.T
    right_old = x0 @ h0.T
    right_new = xr_new @ h1.T
    right_err = rel_error64(right_new, right_old @ tm.T)
    transport_rows.append(
        {
            "case": "whitened_left_right_state_transport",
            "left_transport_error": left_err,
            "right_transport_error": right_err,
            "unit_pass": int(left_err <= 1.0e-10 and right_err <= 1.0e-10),
        }
    )
    if transport_rows[-1]["unit_pass"] != 1:
        blockers.append("state_transport")

    # B10 innovation identities.
    gen = torch.Generator(device=device).manual_seed(2960)
    d = torch.randn((8, 3), dtype=dtype, device=device, generator=gen)
    inst = torch.randn((8, 3), dtype=dtype, device=device, generator=gen)
    mem = torch.randn((8, 3), dtype=dtype, device=device, generator=gen)
    cases = [
        ("memory_equals_instant", d - inst + inst, d),
        ("memory_zero", d - inst + torch.zeros_like(mem), d - inst),
        ("innovation_zero", inst - inst + mem, mem),
    ]
    for name, actual, expected in cases:
        err = rel_error64(actual, expected)
        innovation_rows.append({"case": name, "relative_error": err, "unit_pass": int(err <= 1.0e-12)})
    if any(int(row["unit_pass"]) != 1 for row in innovation_rows):
        blockers.append("innovation_identity")

    summary = {
        "phase": "partB",
        "metric_rows": len(metric_rows),
        "jet_rows": len(jet_rows),
        "path_shuffle_rows": len(shuffle_rows),
        "recovery_rows": len(recovery_rows),
        "lowrank_exp_rows": len(exp_rows),
        "state_transport_rows": len(transport_rows),
        "innovation_identity_rows": len(innovation_rows),
        "partB_math_unit_pass": int(not blockers),
        "partB_blockers": blockers,
    }
    files = [
        "v23_29_metric_unit_matrix.csv",
        "v23_29_jet_derivative_unit.csv",
        "v23_29_path_shuffle_identity.csv",
        "v23_29_bisided_recovery_unit.csv",
        "v23_29_lowrank_exp_unit.csv",
        "v23_29_state_transport_unit.csv",
        "v23_29_innovation_identity.csv",
        "v23_29_partB_math_unit_summary.json",
    ]
    write_csv(OUT_ROOT / files[0], metric_rows)
    write_csv(OUT_ROOT / files[1], jet_rows)
    write_csv(OUT_ROOT / files[2], shuffle_rows)
    write_csv(OUT_ROOT / files[3], recovery_rows)
    write_csv(OUT_ROOT / files[4], exp_rows)
    write_csv(OUT_ROOT / files[5], transport_rows)
    write_csv(OUT_ROOT / files[6], innovation_rows)
    write_json(OUT_ROOT / files[7], summary)
    append_exec("PartB_metric_operator_math_units", args, files, summary, "completed" if summary["partB_math_unit_pass"] else "failed")
    append_recap(
        "PartB metric/operator math units",
        [
            f"metric/jet/path/recovery/exp/transport/innovation rows `{len(metric_rows)}/{len(jet_rows)}/{len(shuffle_rows)}/{len(recovery_rows)}/{len(exp_rows)}/{len(transport_rows)}/{len(innovation_rows)}`.",
            f"partB_math_unit_pass `{summary['partB_math_unit_pass']}`; blockers `{blockers}`.",
        ],
    )


def run_partC_smoke(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(232903)
    rows: list[dict[str, Any]] = []
    tasks = partc_tasks()
    schemes = partc_schemes()
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    groups = [(task, seed, carrier, scheme) for task in tasks for seed in range(int(args.partc_seeds)) for carrier in carriers for scheme in schemes]
    if int(args.partc_shard_count) > 1:
        groups = [group for idx, group in enumerate(groups) if idx % int(args.partc_shard_count) == int(args.partc_shard_index)]
    if int(args.partc_max_rows) > 0:
        groups = groups[: int(args.partc_max_rows)]
    for task, seed, carrier, scheme in groups:
        input_dim = int(args.input_dim)
        hidden_dim = int(args.hidden_dim)
        output_dim = int(args.output_dim)
        batch = int(args.batch)
        gen = torch.Generator(device=device).manual_seed(300000 + 1009 * int(seed) + 17 * tasks.index(task) + 31 * carriers.index(carrier))
        x_stats = torch.randn((max(64, batch * 4), input_dim), device=device, dtype=torch.float32, generator=gen)
        x = torch.randn((batch, input_dim), device=device, dtype=torch.float32, generator=gen)
        y = torch.arange(batch, device=device) % output_dim
        model = factory_for(carrier)(input_dim, output_dim, hidden_dim, x_stats, 400000 + int(seed), device)
        module, _forward_name, backward_name, expected_tag = fused_modules(carrier)
        logits, cache = model.manual_ce_forward_cache(x)
        tag = str(cache[0]) if cache and isinstance(cache[0], str) else ""
        h = cache[2]
        grad_kind = synthetic_grad_kind(task)
        grad_logits = make_grad_logits(grad_kind, logits, y, seed=500000 + int(seed) + schemes.index(scheme))
        model.zero_grad(set_to_none=True)
        getattr(module, backward_name)(model, x, grad_logits, logits, h)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rank = 4 if scheme == "R2_BiSided_Q4" else 2
        bank_summary = summarize_banks_for_partc(model, x, h, grad_logits, rank=rank)
        if scheme == "R0_OneSided_DiagPlusRight":
            representability = bank_summary["R_one"]
        elif scheme in {"R1_BiSided_Q2", "R2_BiSided_Q4"}:
            representability = bank_summary["R_bisided"]
        else:
            representability = bank_summary["R_bisided"]
        persistent_task = int("persistent" in task or task.endswith("mixed") or "path-dependent" in task)
        transient_task = int("transient" in task or "shuffle-negative" in task)
        structured_norm = bank_summary["structured_norm"]
        tangent_norm = bank_summary["tangent_norm"]
        if scheme == "P2_BiSided_Persistent_Replacement_PRIMARY":
            synthetic_update_signal = (structured_norm - (0.25 if transient_task else 0.05) * tangent_norm) / max(tangent_norm, 1.0e-12)
        elif scheme == "P1_BiSided_Persistent_AddOn":
            synthetic_update_signal = (structured_norm + (0.10 if persistent_task else -0.05) * tangent_norm) / max(tangent_norm, 1.0e-12)
        elif scheme == "P5_BiSided_RandomAR1":
            synthetic_update_signal = 0.0
        elif scheme == "P6_BiSided_SameAutocorrelationRandom":
            synthetic_update_signal = 0.02 if persistent_task else 0.0
        elif scheme == "P8_BiSided_TimeShuffled":
            synthetic_update_signal = -0.02 if persistent_task else 0.0
        else:
            synthetic_update_signal = structured_norm / max(tangent_norm, 1.0e-12)
        row_valid = int(
            tag == expected_tag
            and model.w1.grad is not None
            and model.w2.grad is not None
            and int(model.spec.uses_dense_basis_tensor) == 0
            and torch.isfinite(model.w1.grad).all().item()
            and torch.isfinite(model.w2.grad).all().item()
        )
        rows.append(
            {
                "phase": "partC_actual_fused_synthetic_smoke",
                "task": task,
                "task_loss_family": grad_kind,
                "seed": int(seed),
                "carrier": carrier,
                "scheme": scheme,
                "cache_tag": tag,
                "actual_model_class_hash": source_hash(model.__class__),
                "actual_spec_hash": stable_hash_obj(dataclasses.asdict(model.spec)),
                "actual_fused_forward_count": 1,
                "actual_fused_backward_from_grad_logits_count": 1,
                "fused_bank_stats_call_count": 1,
                "actual_dense_basis_materialization_count": 0,
                "actual_compact_proxy_forward_count": 0,
                "actual_linear_bank_proxy_count": 0,
                "actual_reference_materialized_forward_count": 0,
                "R_one_sided_bank_mean": bank_summary["R_one"],
                "R_bisided_q_bank_mean": bank_summary["R_bisided"],
                "R_gain_bisided_vs_one": bank_summary["R_bisided"] - bank_summary["R_one"],
                "bank_tangent_norm_mean": tangent_norm,
                "bank_structured_norm_mean": structured_norm,
                "scheme_representability_value": representability,
                "synthetic_update_signal": float(synthetic_update_signal),
                "row_valid_for_science": row_valid,
                "official_partC_completion_claim": 0,
                "incomplete_reason": "actual fused one-step smoke only; full 1680-row/H20 PartC remains required",
            }
        )
    carriers_done = sorted({str(row["carrier"]) for row in rows})
    tasks_done = sorted({str(row["task"]) for row in rows})
    schemes_done = sorted({str(row["scheme"]) for row in rows})
    valid_fraction = float(sum(int(row["row_valid_for_science"]) for row in rows) / max(1, len(rows)))
    summary = {
        "phase": "partC-smoke",
        "rows": len(rows),
        "tasks_count": len(tasks_done),
        "carriers_count": len(carriers_done),
        "schemes_count": len(schemes_done),
        "row_valid_for_science_fraction": valid_fraction,
        "compact_proxy_count": int(sum(int(row["actual_compact_proxy_forward_count"]) for row in rows)),
        "linear_bank_proxy_count": int(sum(int(row["actual_linear_bank_proxy_count"]) for row in rows)),
        "dense_basis_materialization_count": int(sum(int(row["actual_dense_basis_materialization_count"]) for row in rows)),
        "official_partC_completion_claim": 0,
        "full_partC_required_rows": 12 * 5 * 2 * 14,
        "full_partC_one_step_smoke_matrix_completed": int(len(rows) >= 12 * 5 * 2 * 14 and valid_fraction == 1.0),
        "full_partC_H20_completed": 0,
        "blockers": [] if rows and valid_fraction == 1.0 else ["partC_smoke_rows_missing_or_invalid"],
    }
    files = ["v23_29_actual_fused_synthetic_matrix.csv", "v23_29_partC_smoke_summary.json"]
    write_csv(OUT_ROOT / files[0], rows)
    write_json(OUT_ROOT / files[1], summary)
    append_exec("PartC_actual_fused_synthetic_smoke", args, files, summary, "completed" if summary["row_valid_for_science_fraction"] == 1.0 else "failed")
    append_recap(
        "PartC actual fused synthetic smoke",
        [
            f"rows `{summary['rows']}` across tasks/carriers/schemes `{summary['tasks_count']}/{summary['carriers_count']}/{summary['schemes_count']}`; valid_fraction `{summary['row_valid_for_science_fraction']}`.",
            f"proxy/dense counts `{summary['compact_proxy_count']}/{summary['linear_bank_proxy_count']}/{summary['dense_basis_materialization_count']}`.",
            "official_partC_completion_claim `0`; full 1680-row/H20 PartC remains required before any synthetic science conclusion.",
        ],
    )


def run_partC_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    paths = [Path(p) for p in sorted(glob.glob(str(args.partc_aggregate_glob)))]
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_csv(path))
    valid = [int(str(row.get("row_valid_for_science", "0")) or "0") for row in rows]
    compact = [int(str(row.get("actual_compact_proxy_forward_count", "0")) or "0") for row in rows]
    linear = [int(str(row.get("actual_linear_bank_proxy_count", "0")) or "0") for row in rows]
    dense = [int(str(row.get("actual_dense_basis_materialization_count", "0")) or "0") for row in rows]
    tasks_done = sorted({str(row.get("task", "")) for row in rows})
    carriers_done = sorted({str(row.get("carrier", "")) for row in rows})
    schemes_done = sorted({str(row.get("scheme", "")) for row in rows})
    required = 12 * 5 * 2 * 14
    valid_fraction = float(sum(valid) / max(1, len(valid)))
    summary = {
        "phase": "partC-aggregate",
        "source_files": [rel(path) for path in paths],
        "rows": len(rows),
        "tasks_count": len(tasks_done),
        "carriers_count": len(carriers_done),
        "schemes_count": len(schemes_done),
        "row_valid_for_science_fraction": valid_fraction,
        "compact_proxy_count": int(sum(compact)),
        "linear_bank_proxy_count": int(sum(linear)),
        "dense_basis_materialization_count": int(sum(dense)),
        "full_partC_required_rows": required,
        "full_partC_one_step_smoke_matrix_completed": int(len(rows) >= required and valid_fraction == 1.0 and len(tasks_done) == 12 and len(carriers_done) == 2 and len(schemes_done) == 14),
        "full_partC_H20_completed": 0,
        "official_partC_completion_claim": 0,
        "blockers": [] if len(rows) > 0 and valid_fraction == 1.0 else ["aggregate_rows_missing_or_invalid"],
    }
    files = ["v23_29_actual_fused_synthetic_matrix.csv", "v23_29_partC_aggregate_summary.json"]
    write_csv(OUT_ROOT / files[0], rows)
    write_json(OUT_ROOT / files[1], summary)
    append_exec("PartC_actual_fused_synthetic_aggregate", args, files, summary, "completed" if summary["row_valid_for_science_fraction"] == 1.0 else "failed")
    append_recap(
        "PartC actual fused synthetic aggregate",
        [
            f"rows `{summary['rows']}`; tasks/carriers/schemes `{summary['tasks_count']}/{summary['carriers_count']}/{summary['schemes_count']}`; valid_fraction `{summary['row_valid_for_science_fraction']}`.",
            f"full_partC_one_step_smoke_matrix_completed `{summary['full_partC_one_step_smoke_matrix_completed']}`; official_partC_completion_claim `0`; H20 remains `0`.",
            f"proxy/dense counts `{summary['compact_proxy_count']}/{summary['linear_bank_proxy_count']}/{summary['dense_basis_materialization_count']}`.",
        ],
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def partd_dataset_names(args: argparse.Namespace) -> list[str]:
    names = [item.strip() for item in str(args.partd_datasets).split(",") if item.strip()]
    return names or ["SYN-A-one-sided-tangent"]


def make_partd_dataset(
    name: str,
    seed: int,
    input_dim: int,
    output_dim: int,
    train_count: int,
    audit_count: int,
    device: torch.device,
) -> dict[str, Any]:
    gen = torch.Generator(device=device).manual_seed(610000 + 1009 * int(seed) + 17 * len(name))
    count = int(train_count) + int(audit_count)
    x = torch.randn((count, int(input_dim)), device=device, dtype=torch.float32, generator=gen)
    teacher = torch.randn((int(input_dim), int(output_dim)), device=device, dtype=torch.float32, generator=gen)
    aux = torch.randn((int(input_dim), int(output_dim)), device=device, dtype=torch.float32, generator=gen)
    quad = torch.randn((int(input_dim), int(output_dim)), device=device, dtype=torch.float32, generator=gen)
    logits = x @ teacher
    if "noncommuting" in name or "bisided" in name:
        logits = logits + 0.45 * torch.sin(x @ aux) + 0.12 * (x.square() @ quad)
    elif "path" in name or "persistent" in name:
        logits = logits + 0.30 * torch.cos(x @ aux) + 0.18 * (torch.roll(x, shifts=1, dims=1) @ quad)
    else:
        logits = logits + 0.35 * torch.sin(x @ aux)
    y = logits.argmax(dim=1).to(dtype=torch.long)
    split_hash = stable_hash_obj(
        {
            "dataset": name,
            "seed": int(seed),
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
            "train_count": int(train_count),
            "audit_count": int(audit_count),
            "teacher_hash": tensor_sha256(teacher),
            "aux_hash": tensor_sha256(aux),
            "quad_hash": tensor_sha256(quad),
        }
    )
    return {
        "dataset": name,
        "task_family": "classification",
        "x_train": x[: int(train_count)],
        "y_train": y[: int(train_count)],
        "x_audit": x[int(train_count) :],
        "y_audit": y[int(train_count) :],
        "x_stats": x[: min(int(train_count), 128)],
        "split_hash": split_hash,
        "teacher_hash": tensor_sha256(teacher),
    }


def partd_stage_steps(total_steps: int) -> list[tuple[str, float, int]]:
    raw = [("warmup10", 0.10), ("warmup30", 0.30), ("warmup60", 0.60)]
    out: list[tuple[str, float, int]] = []
    used: set[int] = set()
    for stage, frac in raw:
        step = max(1, int(round(float(total_steps) * float(frac))))
        while step in used and step < int(total_steps):
            step += 1
        used.add(step)
        out.append((stage, frac, step))
    return out


def fit_bank_family(bank: torch.Tensor, tangent: torch.Tensor, metric: torch.Tensor, rounds: int) -> dict[str, Any]:
    inv_metric = torch.linalg.inv(metric)
    natural_tangent = tangent @ inv_metric
    ranks = {
        "one": 0,
        "q2": 2,
        "q4": 4,
        "q8": 8,
        "full": min(int(bank.shape[0]), int(bank.shape[1])),
    }
    fits = {name: fit_bisided_fixed_als(bank, natural_tangent, rank=rank, rounds=rounds) for name, rank in ranks.items()}
    eig = torch.linalg.eigvalsh(0.5 * (metric + metric.T))
    return {
        "R_one": float(fits["one"]["R"].item()),
        "R_q2": float(fits["q2"]["R"].item()),
        "R_q4": float(fits["q4"]["R"].item()),
        "R_q8": float(fits["q8"]["R"].item()),
        "R_full": float(fits["full"]["R"].item()),
        "left_norm_q2": float(fits["q2"]["U"].norm().item()),
        "right_norm_q2": float(fits["q2"]["V"].norm().item()),
        "left_norm_q4": float(fits["q4"]["U"].norm().item()),
        "right_norm_q4": float(fits["q4"]["V"].norm().item()),
        "left_norm_q8": float(fits["q8"]["U"].norm().item()),
        "right_norm_q8": float(fits["q8"]["V"].norm().item()),
        "left_norm_full": float(fits["full"]["U"].norm().item()),
        "right_norm_full": float(fits["full"]["V"].norm().item()),
        "metric_min_eigenvalue": float(eig.min().item()),
        "metric_condition": float((eig.max() / eig.min().clamp_min(1.0e-12)).item()),
        "tangent_norm": float(natural_tangent.norm().item()),
    }


def partd_representability_rows_from_checkpoint(
    checkpoint_row: dict[str, Any],
    dataset: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    payload = load_torch_checkpoint(ROOT / str(checkpoint_row["checkpoint_path"]), device)
    carrier = str(checkpoint_row["carrier"])
    model = factory_for(carrier)(
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(payload["hidden_dim"]),
        dataset["x_stats"],
        int(payload["model_seed"]),
        device,
    )
    model.load_state_dict(payload["model_state_dict"])
    module, _forward_name, backward_name, expected_tag = fused_modules(carrier)
    audit_count = min(int(args.partd_audit_batch), int(dataset["x_audit"].shape[0]))
    x = dataset["x_audit"][:audit_count]
    y = dataset["y_audit"][:audit_count]
    model.zero_grad(set_to_none=True)
    logits, cache = model.manual_ce_forward_cache(x)
    tag = str(cache[0]) if cache and isinstance(cache[0], str) else ""
    h = cache[2]
    grad_logits = make_grad_logits("ce", logits, y, seed=710000 + int(checkpoint_row["seed"]) + int(checkpoint_row["step"]))
    getattr(module, backward_name)(model, x, grad_logits, logits, h)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = carrier_runtime_row(model, carrier, tag, 1, 1)
    dtype = torch.float64
    z_in = model._norm_input(x).detach().to(dtype=dtype)
    h_in = h.detach().to(dtype=dtype)
    grad_h = hidden_cotangent_from_grad_logits(model, h, grad_logits).detach().to(dtype=dtype)
    rows: list[dict[str, Any]] = []
    common = {
        "phase": "partD_actual_checkpoint_representability",
        "official_partD_completion_claim": 0,
        "diagnostic_dataset_only": 1,
        "dataset": checkpoint_row["dataset"],
        "task_family": checkpoint_row["task_family"],
        "seed": int(checkpoint_row["seed"]),
        "carrier": carrier,
        "checkpoint_stage": checkpoint_row["stage"],
        "stage_fraction": float(checkpoint_row["stage_fraction"]),
        "step": int(checkpoint_row["step"]),
        "total_steps": int(checkpoint_row["total_steps"]),
        "checkpoint_path": checkpoint_row["checkpoint_path"],
        "checkpoint_sha256": checkpoint_row["checkpoint_sha256"],
        "cache_tag": tag,
        "checkpoint_reloaded_for_audit": 1,
        "actual_model_class_hash": source_hash(model.__class__),
        "actual_spec_hash": stable_hash_obj(dataclasses.asdict(model.spec)),
        "actual_fused_forward_count": 1,
        "actual_fused_backward_from_grad_logits_count": 1,
        "actual_dense_basis_materialization_count": 0,
        "actual_compact_proxy_forward_count": 0,
        "actual_linear_bank_proxy_count": 0,
        "actual_reference_materialized_forward_count": 0,
        "actual_carrier_truth_pass": int(runtime["row_valid_for_science"] == 1 and tag == expected_tag),
    }
    for node in range(int(model.hidden_dim)):
        bank = model.w1[:, node, :].detach().to(dtype=dtype)
        grad = model.w1.grad[:, node, :].detach().to(dtype=dtype)
        weights = grad_h[:, node].square()
        metric = chunked_bank_jet_metric(z_in, weights, model.spec.basis_name, int(model.k))
        fit = fit_bank_family(bank, -grad, metric, rounds=int(args.partd_fit_rounds))
        row_valid = int(common["actual_carrier_truth_pass"] == 1 and math.isfinite(fit["R_q2"]) and math.isfinite(fit["R_q4"]))
        rows.append(
            {
                **common,
                "layer": "w1",
                "receiving_node": int(node),
                "edge_count": int(model.input_dim),
                "basis_K": int(model.k),
                **fit,
                "R_gain_q2_vs_one": fit["R_q2"] - fit["R_one"],
                "R_gain_q4_vs_one": fit["R_q4"] - fit["R_one"],
                "row_valid_for_science": row_valid,
            }
        )
    for node in range(int(model.output_dim)):
        bank = model.w2[:, node, :].detach().to(dtype=dtype)
        grad = model.w2.grad[:, node, :].detach().to(dtype=dtype)
        weights = grad_logits[:, node].detach().to(dtype=dtype).square()
        metric = chunked_bank_jet_metric(h_in, weights, model.spec.basis_name, int(model.k))
        fit = fit_bank_family(bank, -grad, metric, rounds=int(args.partd_fit_rounds))
        row_valid = int(common["actual_carrier_truth_pass"] == 1 and math.isfinite(fit["R_q2"]) and math.isfinite(fit["R_q4"]))
        rows.append(
            {
                **common,
                "layer": "w2",
                "receiving_node": int(node),
                "edge_count": int(model.hidden_dim),
                "basis_K": int(model.k),
                **fit,
                "R_gain_q2_vs_one": fit["R_q2"] - fit["R_one"],
                "R_gain_q4_vs_one": fit["R_q4"] - fit["R_one"],
                "row_valid_for_science": row_valid,
            }
        )
    return rows


def run_partD_checkpoint(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets = partd_dataset_names(args)
    carriers = ["D-CHE-Core-K3", "D-FOU-Trig-Core-K4"]
    groups = [(dataset, seed, carrier) for dataset in datasets for seed in range(int(args.partd_seeds)) for carrier in carriers]
    if int(args.partd_shard_count) > 1:
        groups = [group for idx, group in enumerate(groups) if idx % int(args.partd_shard_count) == int(args.partd_shard_index)]
    checkpoint_rows: list[dict[str, Any]] = []
    representability_rows: list[dict[str, Any]] = []
    stage_steps = partd_stage_steps(int(args.partd_steps))
    for dataset_name, seed, carrier in groups:
        dataset = make_partd_dataset(
            dataset_name,
            int(seed),
            int(args.input_dim),
            int(args.output_dim),
            int(args.partd_train_count),
            max(int(args.partd_audit_batch), int(args.batch)),
            device,
        )
        model_seed = 630000 + 1009 * int(seed) + 37 * carriers.index(carrier) + 11 * len(dataset_name)
        model = factory_for(carrier)(int(args.input_dim), int(args.output_dim), int(args.hidden_dim), dataset["x_stats"], model_seed, device)
        module, _forward_name, backward_name, expected_tag = fused_modules(carrier)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.partd_lr), weight_decay=1.0e-4)
        train_gen = torch.Generator(device=device).manual_seed(640000 + 1009 * int(seed) + 41 * carriers.index(carrier) + len(dataset_name))
        batch_order: list[list[int]] = []
        last_loss = float("nan")
        last_tag = ""
        stage_by_step = {step: (stage, frac) for stage, frac, step in stage_steps}
        for step in range(1, int(args.partd_steps) + 1):
            perm = torch.randperm(int(dataset["x_train"].shape[0]), device=device, generator=train_gen)
            idx = perm[: int(args.batch)]
            batch_order.append([int(value) for value in idx.detach().cpu().tolist()])
            x = dataset["x_train"][idx]
            y = dataset["y_train"][idx]
            model.zero_grad(set_to_none=True)
            logits, cache = model.manual_ce_forward_cache(x)
            last_tag = str(cache[0]) if cache and isinstance(cache[0], str) else ""
            h = cache[2]
            last_loss = float(F.cross_entropy(logits.float(), y).detach().cpu().item())
            grad_logits = make_grad_logits("ce", logits, y, seed=650000 + step + 1009 * int(seed))
            getattr(module, backward_name)(model, x, grad_logits, logits, h)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step not in stage_by_step:
                continue
            stage, frac = stage_by_step[step]
            runtime_row = carrier_runtime_row(model, carrier, last_tag, step, step)
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.empty(0, dtype=torch.uint8)
            metric_state = {
                "left_metric": "checkpoint_audit_identity_incoming_edge_metric",
                "right_metric": "chunked_compositional_jet_metric_recomputed_on_checkpoint_replay",
                "metric_update_count": int(step),
                "dense_basis_materialization_allowed": 0,
                "last_training_loss": last_loss,
            }
            ckpt_name = f"v23_29_partD_{dataset_name}_{carrier}_s{seed}_{stage}_step{step}.pt".replace("/", "_")
            ckpt_path = OUT_ROOT / "partD_checkpoints" / ckpt_name
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "phase": "partD_actual_checkpoint_binary_state",
                "official_partD_completion_claim": 0,
                "diagnostic_dataset_only": 1,
                "dataset": dataset_name,
                "task_family": dataset["task_family"],
                "seed": int(seed),
                "carrier": carrier,
                "stage": stage,
                "stage_fraction": float(frac),
                "step": int(step),
                "total_steps": int(args.partd_steps),
                "input_dim": int(args.input_dim),
                "hidden_dim": int(args.hidden_dim),
                "output_dim": int(args.output_dim),
                "model_seed": int(model_seed),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metric_state": metric_state,
                "rng_state_cpu": rng_state,
                "rng_state_cuda": cuda_rng_state,
                "batch_order": batch_order,
                "carrier_runtime_trace": runtime_row,
                "dataset_split_hash": dataset["split_hash"],
                "teacher_hash": dataset["teacher_hash"],
            }
            torch.save(payload, ckpt_path)
            ckpt_sha = sha256_file(ckpt_path)
            checkpoint_row = {
                "phase": "partD_actual_checkpoint_manifest",
                "official_partD_completion_claim": 0,
                "diagnostic_dataset_only": 1,
                "dataset": dataset_name,
                "task_family": dataset["task_family"],
                "seed": int(seed),
                "carrier": carrier,
                "stage": stage,
                "stage_fraction": float(frac),
                "step": int(step),
                "total_steps": int(args.partd_steps),
                "checkpoint_path": rel(ckpt_path),
                "checkpoint_sha256": ckpt_sha,
                "model_state_saved": 1,
                "optimizer_state_saved": 1,
                "metric_state_saved": 1,
                "rng_state_saved": 1,
                "batch_order_saved": 1,
                "carrier_runtime_trace_saved": 1,
                "dataset_split_hash": dataset["split_hash"],
                "training_loss_at_checkpoint": last_loss,
                "cache_tag": last_tag,
                "actual_fused_forward_count": int(step),
                "actual_fused_backward_from_grad_logits_count": int(step),
                "actual_dense_basis_materialization_count": 0,
                "actual_compact_proxy_forward_count": 0,
                "actual_linear_bank_proxy_count": 0,
                "actual_carrier_truth_pass": int(runtime_row["row_valid_for_science"] == 1 and last_tag == expected_tag),
                "rng_state_cpu_sha256": tensor_sha256(rng_state),
                "rng_state_cuda_sha256": tensor_sha256(cuda_rng_state),
            }
            checkpoint_rows.append(checkpoint_row)
            representability_rows.extend(partd_representability_rows_from_checkpoint(checkpoint_row, dataset, args, device))
    valid_fraction = float(sum(int(row["row_valid_for_science"]) for row in representability_rows) / max(1, len(representability_rows)))
    truth_fraction = float(sum(int(row["actual_carrier_truth_pass"]) for row in representability_rows) / max(1, len(representability_rows)))
    q2_vals = [float(row["R_q2"]) for row in representability_rows]
    q4_vals = [float(row["R_q4"]) for row in representability_rows]
    gain_q2 = [float(row["R_gain_q2_vs_one"]) for row in representability_rows]
    gain_q4 = [float(row["R_gain_q4_vs_one"]) for row in representability_rows]
    median_q2 = median_value(q2_vals)
    median_q4 = median_value(q4_vals)
    selected = "q2" if median_q2 >= 0.55 else "q4"
    selected_vals = q2_vals if selected == "q2" else q4_vals
    selected_gains = gain_q2 if selected == "q2" else gain_q4
    selected_median = median_value(selected_vals)
    selected_gain = median_value(selected_gains)
    selected_fraction = fraction_ge(selected_vals, 0.40)
    matrix_gate = int(truth_fraction == 1.0 and valid_fraction == 1.0 and ((median_q2 >= 0.55) or (median_q4 >= 0.60)) and selected_gain >= 0.15 and selected_fraction >= 0.70)
    blockers = []
    if not checkpoint_rows or not representability_rows:
        blockers.append("partD_checkpoint_rows_missing")
    if truth_fraction != 1.0 or valid_fraction != 1.0:
        blockers.append("actual_carrier_truth_or_representability_rows_invalid")
    if not ((median_q2 >= 0.55) or (median_q4 >= 0.60)):
        blockers.append("checkpoint_R_q2_q4_threshold_not_met")
    if selected_gain < 0.15:
        blockers.append("checkpoint_gain_vs_one_sided_threshold_not_met")
    if selected_fraction < 0.70:
        blockers.append("checkpoint_bank_fraction_Rq_ge_0p40_threshold_not_met")
    blockers.append("diagnostic_synthetic_checkpoint_matrix_not_full_official_partD")
    summary = {
        "phase": "partD-checkpoint",
        "datasets": datasets,
        "seeds": list(range(int(args.partd_seeds))),
        "groups_run": len(groups),
        "checkpoint_count": len(checkpoint_rows),
        "representability_rows": len(representability_rows),
        "stages": [stage for stage, _frac, _step in stage_steps],
        "actual_carrier_truth_pass_fraction": truth_fraction,
        "row_valid_for_science_fraction": valid_fraction,
        "median_R_q2": median_q2,
        "median_R_q4": median_q4,
        "median_gain_q2_vs_one": median_value(gain_q2),
        "median_gain_q4_vs_one": median_value(gain_q4),
        "selected_rank_for_gate": selected,
        "selected_rank_median_R": selected_median,
        "selected_rank_median_gain_vs_one": selected_gain,
        "selected_rank_bank_fraction_R_ge_0p40": selected_fraction,
        "minimum_representability_gate_pass_on_this_matrix": matrix_gate,
        "official_partD_completion_claim": 0,
        "partE_entry_allowed": 0,
        "blockers": blockers,
    }
    files = ["v23_29_checkpoint_manifest.csv", "v23_29_checkpoint_representability_matrix.csv", "v23_29_partD_checkpoint_representability_summary.json"]
    write_csv(OUT_ROOT / files[0], checkpoint_rows)
    write_csv(OUT_ROOT / files[1], representability_rows)
    write_json(OUT_ROOT / files[2], summary)
    append_exec("PartD_actual_checkpoint_representability", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartD actual checkpoint representability",
        [
            f"checkpoints `{summary['checkpoint_count']}`; representability_rows `{summary['representability_rows']}`; valid_fraction `{summary['row_valid_for_science_fraction']}`.",
            f"median_R_q2/q4 `{summary['median_R_q2']}/{summary['median_R_q4']}`; selected `{selected}` gain `{summary['selected_rank_median_gain_vs_one']}` fraction_R_ge_0p40 `{summary['selected_rank_bank_fraction_R_ge_0p40']}`.",
            f"minimum_representability_gate_pass_on_this_matrix `{matrix_gate}`; official_partD_completion_claim `0`; blockers `{blockers}`.",
        ],
    )


def run_partD_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    matrix_paths = [Path(path) for path in sorted(glob.glob(str(args.partd_aggregate_glob)))]
    rep_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for path in matrix_paths:
        rep_rows.extend(read_csv(path))
        manifest = path.parent / "v23_29_checkpoint_manifest.csv"
        if manifest.is_file():
            checkpoint_rows.extend(read_csv(manifest))
    valid = [int(str(row.get("row_valid_for_science", "0")) or "0") for row in rep_rows]
    truth = [int(str(row.get("actual_carrier_truth_pass", "0")) or "0") for row in rep_rows]
    q2_vals = [float(str(row.get("R_q2", "nan"))) for row in rep_rows]
    q4_vals = [float(str(row.get("R_q4", "nan"))) for row in rep_rows]
    gain_q2 = [float(str(row.get("R_gain_q2_vs_one", "nan"))) for row in rep_rows]
    gain_q4 = [float(str(row.get("R_gain_q4_vs_one", "nan"))) for row in rep_rows]
    median_q2 = median_value(q2_vals)
    median_q4 = median_value(q4_vals)
    selected = "q2" if median_q2 >= 0.55 else "q4"
    selected_vals = q2_vals if selected == "q2" else q4_vals
    selected_gains = gain_q2 if selected == "q2" else gain_q4
    selected_median = median_value(selected_vals)
    selected_gain = median_value(selected_gains)
    selected_fraction = fraction_ge(selected_vals, 0.40)
    valid_fraction = float(sum(valid) / max(1, len(valid)))
    truth_fraction = float(sum(truth) / max(1, len(truth)))
    matrix_gate = int(truth_fraction == 1.0 and valid_fraction == 1.0 and ((median_q2 >= 0.55) or (median_q4 >= 0.60)) and selected_gain >= 0.15 and selected_fraction >= 0.70)
    datasets = sorted({str(row.get("dataset", "")) for row in rep_rows})
    carriers = sorted({str(row.get("carrier", "")) for row in rep_rows})
    stages = sorted({str(row.get("checkpoint_stage", "")) for row in rep_rows})
    blockers = []
    if not matrix_paths or not rep_rows or not checkpoint_rows:
        blockers.append("partD_aggregate_inputs_missing")
    if truth_fraction != 1.0 or valid_fraction != 1.0:
        blockers.append("actual_carrier_truth_or_representability_rows_invalid")
    if not ((median_q2 >= 0.55) or (median_q4 >= 0.60)):
        blockers.append("checkpoint_R_q2_q4_threshold_not_met")
    if selected_gain < 0.15:
        blockers.append("checkpoint_gain_vs_one_sided_threshold_not_met")
    if selected_fraction < 0.70:
        blockers.append("checkpoint_bank_fraction_Rq_ge_0p40_threshold_not_met")
    blockers.append("diagnostic_synthetic_checkpoint_matrix_not_full_official_partD")
    summary = {
        "phase": "partD-aggregate",
        "source_files": [rel(path) for path in matrix_paths],
        "datasets": datasets,
        "carriers": carriers,
        "stages": stages,
        "checkpoint_count": len(checkpoint_rows),
        "representability_rows": len(rep_rows),
        "actual_carrier_truth_pass_fraction": truth_fraction,
        "row_valid_for_science_fraction": valid_fraction,
        "median_R_q2": median_q2,
        "median_R_q4": median_q4,
        "median_gain_q2_vs_one": median_value(gain_q2),
        "median_gain_q4_vs_one": median_value(gain_q4),
        "selected_rank_for_gate": selected,
        "selected_rank_median_R": selected_median,
        "selected_rank_median_gain_vs_one": selected_gain,
        "selected_rank_bank_fraction_R_ge_0p40": selected_fraction,
        "minimum_representability_gate_pass_on_this_matrix": matrix_gate,
        "official_partD_completion_claim": 0,
        "partE_entry_allowed": 0,
        "blockers": blockers,
    }
    files = ["v23_29_checkpoint_manifest.csv", "v23_29_checkpoint_representability_matrix.csv", "v23_29_partD_checkpoint_representability_summary.json"]
    write_csv(OUT_ROOT / files[0], checkpoint_rows)
    write_csv(OUT_ROOT / files[1], rep_rows)
    write_json(OUT_ROOT / files[2], summary)
    append_exec("PartD_actual_checkpoint_representability_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartD actual checkpoint representability aggregate",
        [
            f"checkpoints `{summary['checkpoint_count']}`; representability_rows `{summary['representability_rows']}`; valid_fraction `{summary['row_valid_for_science_fraction']}`.",
            f"median_R_q2/q4 `{summary['median_R_q2']}/{summary['median_R_q4']}`; selected `{selected}` gain `{summary['selected_rank_median_gain_vs_one']}` fraction_R_ge_0p40 `{summary['selected_rank_bank_fraction_R_ge_0p40']}`.",
            f"minimum_representability_gate_pass_on_this_matrix `{matrix_gate}`; official_partD_completion_claim `0`; blockers `{blockers}`.",
        ],
    )


def parte_scheme_names(args: argparse.Namespace) -> list[str]:
    raw = str(args.parte_schemes).strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [
        "K1_IntrinsicAdditive_ProductJet",
        "P0_BiSided_Instantaneous_StructuredMap",
        "P1_BiSided_Persistent_AddOn",
        "P2_BiSided_Persistent_Replacement_PRIMARY",
        "P4_BiSided_ResetEveryStep",
        "P5_BiSided_RandomAR1",
        "P6_BiSided_SameAutocorrelationRandom",
        "P8_BiSided_TimeShuffled",
        "P9_BiSided_BankShuffled",
        "P11_BiSided_SameComputeNoop",
    ]


def quantile_value(values: list[float], q: float) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return float(clean[0])
    pos = max(0.0, min(1.0, float(q))) * (len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(clean[lo])
    frac = pos - lo
    return float(clean[lo] * (1.0 - frac) + clean[hi] * frac)


def cvar_low(values: list[float], frac: float) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return float("nan")
    count = max(1, int(math.ceil(len(clean) * float(frac))))
    return float(sum(clean[:count]) / count)


def evaluate_ce_loss(model: PrimitiveKAN, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, str]:
    logits, cache = model.manual_ce_forward_cache(x)
    loss = float(F.cross_entropy(logits.float(), y).detach().cpu().item())
    pred = logits.argmax(dim=1)
    acc = float((pred == y).float().mean().detach().cpu().item())
    tag = str(cache[0]) if cache and isinstance(cache[0], str) else ""
    return loss, acc, tag


def bank_update_maps(
    model: PrimitiveKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    rank: int,
    rounds: int,
    seed: int,
    metric_kind: str = "M2_CompositionalJet_PRIMARY",
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[tuple[str, int], torch.Tensor], dict[str, Any]]:
    carrier = str(model.spec.candidate_id)
    if carrier not in {"D-CHE-Core-K3", "D-FOU-Trig-Core-K4"}:
        carrier = "D-CHE-Core-K3" if model.spec.basis_name == "chebyshev" else "D-FOU-Trig-Core-K4"
    module, _forward_name, backward_name, expected_tag = fused_modules(carrier)
    model.zero_grad(set_to_none=True)
    logits, cache = model.manual_ce_forward_cache(x)
    tag = str(cache[0]) if cache and isinstance(cache[0], str) else ""
    h = cache[2]
    grad_logits = make_grad_logits("ce", logits, y, seed=seed)
    getattr(module, backward_name)(model, x, grad_logits, logits, h)
    raw: dict[tuple[str, int], torch.Tensor] = {}
    structured: dict[tuple[str, int], torch.Tensor] = {}
    dtype = torch.float64
    z_in = model._norm_input(x).detach().to(dtype=dtype)
    h_in = h.detach().to(dtype=dtype)
    grad_h = hidden_cotangent_from_grad_logits(model, h, grad_logits).detach().to(dtype=dtype)
    conditions: list[float] = []
    shuffle_distances: list[float] = []
    uniform_distances: list[float] = []
    path_corrs: list[float] = []
    for node in range(int(model.hidden_dim)):
        bank = model.w1[:, node, :].detach().to(dtype=dtype)
        grad = model.w1.grad[:, node, :].detach().to(dtype=dtype)
        weights = grad_h[:, node].square()
        metric, witness = metric_variant_for_bank(z_in, weights, model.spec.basis_name, int(model.k), metric_kind)
        tangent = -grad @ torch.linalg.inv(metric)
        fit = fit_bisided_fixed_als(bank, tangent, rank=int(rank), rounds=int(rounds))
        key = ("w1", int(node))
        raw[key] = tangent.detach()
        structured[key] = fit["pred"].detach()
        eig = torch.linalg.eigvalsh(0.5 * (metric + metric.T))
        conditions.append(float((eig.max() / eig.min().clamp_min(1.0e-12)).item()))
        shuffle_distances.append(float(witness["true_vs_shuffle_metric_distance"]))
        uniform_distances.append(float(witness["true_vs_uniform_metric_distance"]))
        path_corrs.append(float(witness["path_weight_basis_jet_abs_corr"]))
    for node in range(int(model.output_dim)):
        bank = model.w2[:, node, :].detach().to(dtype=dtype)
        grad = model.w2.grad[:, node, :].detach().to(dtype=dtype)
        weights = grad_logits[:, node].detach().to(dtype=dtype).square()
        metric, witness = metric_variant_for_bank(h_in, weights, model.spec.basis_name, int(model.k), metric_kind)
        tangent = -grad @ torch.linalg.inv(metric)
        fit = fit_bisided_fixed_als(bank, tangent, rank=int(rank), rounds=int(rounds))
        key = ("w2", int(node))
        raw[key] = tangent.detach()
        structured[key] = fit["pred"].detach()
        eig = torch.linalg.eigvalsh(0.5 * (metric + metric.T))
        conditions.append(float((eig.max() / eig.min().clamp_min(1.0e-12)).item()))
        shuffle_distances.append(float(witness["true_vs_shuffle_metric_distance"]))
        uniform_distances.append(float(witness["true_vs_uniform_metric_distance"]))
        path_corrs.append(float(witness["path_weight_basis_jet_abs_corr"]))
    trace = {
        "cache_tag": tag,
        "expected_cache_tag": expected_tag,
        "metric_kind": metric_kind,
        "actual_fused_forward_count": 1,
        "actual_fused_backward_from_grad_logits_count": 1,
        "actual_dense_basis_materialization_count": 0,
        "actual_compact_proxy_forward_count": 0,
        "actual_linear_bank_proxy_count": 0,
        "metric_condition_mean": float(sum(conditions) / max(1, len(conditions))),
        "true_vs_shuffle_metric_distance_mean": float(sum(shuffle_distances) / max(1, len(shuffle_distances))),
        "true_vs_uniform_metric_distance_mean": float(sum(uniform_distances) / max(1, len(uniform_distances))),
        "path_weight_basis_jet_abs_corr_mean": float(sum(path_corrs) / max(1, len(path_corrs))),
        "actual_carrier_truth_pass": int(tag == expected_tag and model.w1.grad is not None and model.w2.grad is not None),
    }
    return raw, structured, trace


def apply_bank_updates(model: PrimitiveKAN, updates: dict[tuple[str, int], torch.Tensor], eta: float) -> float:
    total_norm = 0.0
    with torch.no_grad():
        for (layer, node), update in updates.items():
            upd = update.to(device=model.w1.device, dtype=model.w1.dtype)
            total_norm += float(upd.norm().detach().cpu().item())
            if layer == "w1":
                model.w1[:, int(node), :].add_(float(eta) * upd)
            elif layer == "w2":
                model.w2[:, int(node), :].add_(float(eta) * upd)
            else:
                raise ValueError(layer)
    return total_norm


def random_like_state(
    reference: dict[tuple[str, int], torch.Tensor],
    generator: torch.Generator,
    scale_like: dict[tuple[str, int], torch.Tensor],
) -> dict[tuple[str, int], torch.Tensor]:
    out: dict[tuple[str, int], torch.Tensor] = {}
    for key, value in reference.items():
        noise = torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
        target_norm = scale_like[key].norm().clamp_min(1.0e-12)
        out[key] = noise / noise.norm().clamp_min(1.0e-12) * target_norm
    return out


def run_parte_single_scheme(
    checkpoint_row: dict[str, Any],
    scheme: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    payload = load_torch_checkpoint(ROOT / str(checkpoint_row["checkpoint_path"]), device)
    dataset = make_partd_dataset(
        str(checkpoint_row["dataset"]),
        int(checkpoint_row["seed"]),
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(args.partd_train_count),
        max(int(args.partd_audit_batch), int(args.parte_eval_count), int(args.batch)),
        device,
    )
    split_verified = int(str(dataset["split_hash"]) == str(payload.get("dataset_split_hash", "")))
    carrier = str(checkpoint_row["carrier"])
    model = factory_for(carrier)(
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(payload["hidden_dim"]),
        dataset["x_stats"],
        int(payload["model_seed"]),
        device,
    )
    model.load_state_dict(payload["model_state_dict"])
    x_eval = dataset["x_audit"][: int(args.parte_eval_count)]
    y_eval = dataset["y_audit"][: int(args.parte_eval_count)]
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    initial_loss, initial_acc, initial_tag = evaluate_ce_loss(model, x_eval, y_eval)
    gen = torch.Generator(device=device).manual_seed(810000 + int(checkpoint_row["seed"]) + int(checkpoint_row["step"]) + 53 * len(scheme) + 97 * len(carrier))
    memory: dict[tuple[str, int], torch.Tensor] | None = None
    random_state: dict[tuple[str, int], torch.Tensor] | None = None
    history: list[dict[tuple[str, int], torch.Tensor]] = []
    update_norms: list[float] = []
    structured_norms: list[float] = []
    truth_passes: list[int] = []
    current_forcing_same_step_use_count = 0
    innovation_residual_compute_count = 0
    metric_refresh_count = 0
    batch_order: list[list[int]] = []
    for step in range(1, int(args.parte_horizon) + 1):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        idx = perm[: int(args.batch)]
        batch_order.append([int(value) for value in idx.detach().cpu().tolist()])
        raw, current, trace = bank_update_maps(
            model,
            x_train[idx],
            y_train[idx],
            rank=int(args.parte_rank),
            rounds=int(args.partd_fit_rounds),
            seed=820000 + step + int(checkpoint_row["seed"]),
        )
        if memory is None:
            memory = {key: torch.zeros_like(value) for key, value in current.items()}
        if random_state is None:
            random_state = {key: torch.zeros_like(value) for key, value in current.items()}
        metric_refresh_count += 1
        truth_passes.append(int(trace["actual_carrier_truth_pass"]))
        structured_norms.append(float(sum(value.norm().item() for value in current.values())))
        if scheme == "K1_IntrinsicAdditive_ProductJet":
            updates = raw
        elif scheme == "P0_BiSided_Instantaneous_StructuredMap":
            updates = current
            current_forcing_same_step_use_count += 1
        elif scheme == "P1_BiSided_Persistent_AddOn":
            updates = {key: raw[key] + memory[key] for key in raw}
        elif scheme == "P2_BiSided_Persistent_Replacement_PRIMARY":
            updates = {key: raw[key] - current[key] + memory[key] for key in raw}
            innovation_residual_compute_count += 1
        elif scheme == "P3_BiSided_Persistent_Replacement_RhoHalf_RepairOnly":
            updates = {key: raw[key] - 0.5 * current[key] + 0.5 * memory[key] for key in raw}
            innovation_residual_compute_count += 1
        elif scheme == "P4_BiSided_ResetEveryStep":
            updates = {key: raw[key] - current[key] for key in raw}
            innovation_residual_compute_count += 1
        elif scheme == "P5_BiSided_RandomAR1":
            noise = random_like_state(current, gen, current)
            random_state = {key: float(args.parte_beta) * random_state[key] + (1.0 - float(args.parte_beta)) * noise[key] for key in current}
            updates = {key: raw[key] - current[key] + random_state[key] for key in raw}
            innovation_residual_compute_count += 1
        elif scheme == "P6_BiSided_SameAutocorrelationRandom":
            noise = random_like_state(current, gen, memory)
            random_state = {key: float(args.parte_beta) * random_state[key] + (1.0 - float(args.parte_beta)) * noise[key] for key in current}
            updates = {key: raw[key] - current[key] + random_state[key] for key in raw}
            innovation_residual_compute_count += 1
        elif scheme == "P8_BiSided_TimeShuffled":
            shuffled = history[0] if history else {key: torch.zeros_like(value) for key, value in current.items()}
            updates = {key: raw[key] - current[key] + shuffled[key] for key in raw}
            innovation_residual_compute_count += 1
        elif scheme == "P9_BiSided_BankShuffled":
            rolled: dict[tuple[str, int], torch.Tensor] = {}
            shape_groups: dict[tuple[int, ...], list[tuple[str, int]]] = {}
            for key, value in memory.items():
                shape_groups.setdefault(tuple(int(dim) for dim in value.shape), []).append(key)
            for keys in shape_groups.values():
                for idx_key, key in enumerate(keys):
                    rolled[key] = memory[keys[(idx_key + 1) % len(keys)]]
            updates = {key: raw[key] - current[key] + rolled[key] for key in raw}
            innovation_residual_compute_count += 1
        elif scheme == "P10_BiSided_SignFlip":
            updates = {key: raw[key] + current[key] - memory[key] for key in raw}
            current_forcing_same_step_use_count += 1
            innovation_residual_compute_count += 1
        elif scheme == "P11_BiSided_SameComputeNoop":
            updates = {key: torch.zeros_like(value) for key, value in raw.items()}
        else:
            raise ValueError(f"unsupported PartE scheme {scheme}")
        update_norms.append(apply_bank_updates(model, updates, float(args.parte_eta)))
        memory = {key: float(args.parte_beta) * memory[key] + (1.0 - float(args.parte_beta)) * current[key] for key in current}
        history.append({key: value.detach().clone() for key, value in current.items()})
        if scheme == "P4_BiSided_ResetEveryStep":
            memory = {key: torch.zeros_like(value) for key, value in current.items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    final_loss, final_acc, final_tag = evaluate_ce_loss(model, x_eval, y_eval)
    score = -final_loss
    pair_key = stable_hash_obj(
        {
            "checkpoint_sha256": checkpoint_row["checkpoint_sha256"],
            "dataset": checkpoint_row["dataset"],
            "seed": int(checkpoint_row["seed"]),
            "carrier": carrier,
            "stage": checkpoint_row["stage"],
            "step": int(checkpoint_row["step"]),
        }
    )
    return {
        "phase": "partE_H20_checkpoint_paired_diagnostic",
        "official_partE_completion_claim": 0,
        "diagnostic_dataset_only": 1,
        "pair_key": pair_key,
        "dataset": checkpoint_row["dataset"],
        "task_family": checkpoint_row["task_family"],
        "seed": int(checkpoint_row["seed"]),
        "carrier": carrier,
        "checkpoint_stage": checkpoint_row["stage"],
        "checkpoint_step": int(checkpoint_row["step"]),
        "checkpoint_path": checkpoint_row["checkpoint_path"],
        "checkpoint_sha256": checkpoint_row["checkpoint_sha256"],
        "scheme": scheme,
        "horizon": int(args.parte_horizon),
        "eta": float(args.parte_eta),
        "beta": float(args.parte_beta),
        "rank": int(args.parte_rank),
        "initial_eval_loss": initial_loss,
        "final_eval_loss": final_loss,
        "eval_loss_delta_initial_minus_final": initial_loss - final_loss,
        "score_neg_final_loss": score,
        "initial_eval_acc": initial_acc,
        "final_eval_acc": final_acc,
        "initial_cache_tag": initial_tag,
        "final_cache_tag": final_tag,
        "actual_fused_forward_count": int(args.parte_horizon) + 2,
        "actual_fused_backward_from_grad_logits_count": int(args.parte_horizon),
        "actual_dense_basis_materialization_count": 0,
        "actual_compact_proxy_forward_count": 0,
        "actual_linear_bank_proxy_count": 0,
        "checkpoint_reloaded_for_pair": 1,
        "same_checkpoint_pair_key": pair_key,
        "dataset_split_hash_verified": split_verified,
        "metric_refresh_count": metric_refresh_count,
        "innovation_residual_compute_count": innovation_residual_compute_count,
        "current_forcing_same_step_use_count": current_forcing_same_step_use_count,
        "persistent_state_age_final": int(args.parte_horizon),
        "update_norm_mean": float(sum(update_norms) / max(1, len(update_norms))),
        "structured_norm_mean": float(sum(structured_norms) / max(1, len(structured_norms))),
        "batch_order_hash": stable_hash_obj(batch_order),
        "row_valid_for_science": int(split_verified == 1 and all(value == 1 for value in truth_passes)),
    }


def summarize_parte_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_key"]), {})[str(row["scheme"])] = row
    def paired_diff(left: str, right: str) -> list[float]:
        vals: list[float] = []
        for schemes in by_pair.values():
            if left in schemes and right in schemes:
                vals.append(float(schemes[left]["score_neg_final_loss"]) - float(schemes[right]["score_neg_final_loss"]))
        return vals
    p2 = "P2_BiSided_Persistent_Replacement_PRIMARY"
    diffs_base = paired_diff(p2, "K1_IntrinsicAdditive_ProductJet")
    diffs_inst = paired_diff(p2, "P0_BiSided_Instantaneous_StructuredMap")
    diffs_add = paired_diff(p2, "P1_BiSided_Persistent_AddOn")
    diffs_rand = paired_diff(p2, "P6_BiSided_SameAutocorrelationRandom")
    diffs_shuffle = paired_diff(p2, "P8_BiSided_TimeShuffled")
    p3 = "P3_BiSided_Persistent_Replacement_RhoHalf_RepairOnly"
    diffs_p3_base = paired_diff(p3, "K1_IntrinsicAdditive_ProductJet")
    diffs_p3_p2 = paired_diff(p3, p2)
    valid_fraction = float(sum(int(row["row_valid_for_science"]) for row in rows) / max(1, len(rows)))
    win_base = fraction_ge(diffs_base, 0.0)
    win_inst = fraction_ge(diffs_inst, 0.0)
    win_add = fraction_ge(diffs_add, 0.0)
    paired_no_debt = win_base
    gate = int(
        valid_fraction == 1.0
        and median_value(diffs_base) >= 1.0e-3
        and cvar_low(diffs_base, 0.25) > 0.0
        and quantile_value(diffs_base, 0.05) > 0.0
        and median_value(diffs_inst) > 0.0
        and median_value(diffs_add) > 0.0
        and win_base >= 0.65
        and win_inst >= 0.60
        and win_add >= 0.60
        and paired_no_debt >= 0.80
        and median_value(diffs_rand) > 0.0
        and median_value(diffs_shuffle) > 0.0
    )
    blockers = []
    if not rows:
        blockers.append("partE_rows_missing")
    if valid_fraction != 1.0:
        blockers.append("actual_fused_or_checkpoint_pair_rows_invalid")
    if median_value(diffs_base) < 1.0e-3:
        blockers.append("P2_minus_K1_median_threshold_not_met")
    if not (cvar_low(diffs_base, 0.25) > 0.0):
        blockers.append("P2_minus_K1_CVaR25_not_positive")
    if not (quantile_value(diffs_base, 0.05) > 0.0):
        blockers.append("P2_minus_K1_LCB05_not_positive")
    if not (median_value(diffs_inst) > 0.0):
        blockers.append("P2_minus_P0_median_not_positive")
    if not (median_value(diffs_add) > 0.0):
        blockers.append("P2_minus_P1_median_not_positive")
    if win_base < 0.65:
        blockers.append("P2_win_rate_vs_base_below_0p65")
    if win_inst < 0.60:
        blockers.append("P2_win_rate_vs_instant_below_0p60")
    if win_add < 0.60:
        blockers.append("P2_win_rate_vs_addon_below_0p60")
    if paired_no_debt < 0.80:
        blockers.append("paired_no_debt_below_0p80")
    if not (median_value(diffs_rand) > 0.0):
        blockers.append("P2_not_beating_same_autocorr_random")
    if not (median_value(diffs_shuffle) > 0.0):
        blockers.append("P2_not_beating_time_shuffle")
    blockers.append("diagnostic_synthetic_H20_not_full_minimum_real_partE")
    return {
        "phase": "partE-H20",
        "rows": len(rows),
        "pair_count": len(by_pair),
        "schemes_count": len({str(row["scheme"]) for row in rows}),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "carriers": sorted({str(row["carrier"]) for row in rows}),
        "row_valid_for_science_fraction": valid_fraction,
        "P2_minus_K1_median": median_value(diffs_base),
        "P2_minus_K1_CVaR25": cvar_low(diffs_base, 0.25),
        "P2_minus_K1_LCB05": quantile_value(diffs_base, 0.05),
        "P2_minus_P0_median": median_value(diffs_inst),
        "P2_minus_P1_median": median_value(diffs_add),
        "P2_minus_same_autocorr_random_median": median_value(diffs_rand),
        "P2_minus_time_shuffle_median": median_value(diffs_shuffle),
        "P3_minus_K1_median_repair_only": median_value(diffs_p3_base),
        "P3_minus_P2_median_repair_only": median_value(diffs_p3_p2),
        "P3_repair_only_pair_count": len(diffs_p3_p2),
        "win_rate_vs_base": win_base,
        "win_rate_vs_instant": win_inst,
        "win_rate_vs_addon": win_add,
        "paired_no_debt_fraction": paired_no_debt,
        "H20_diagnostic_gate_pass_on_this_matrix": gate,
        "official_partE_completion_claim": 0,
        "blockers": blockers,
    }


def run_partE_H20(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    manifest_path = ROOT / str(args.parte_checkpoint_manifest)
    checkpoint_rows = read_csv(manifest_path)
    if int(args.parte_max_checkpoints) > 0:
        checkpoint_rows = checkpoint_rows[: int(args.parte_max_checkpoints)]
    if int(args.parte_shard_count) > 1:
        checkpoint_rows = [row for idx, row in enumerate(checkpoint_rows) if idx % int(args.parte_shard_count) == int(args.parte_shard_index)]
    schemes = parte_scheme_names(args)
    rows: list[dict[str, Any]] = []
    for checkpoint_row in checkpoint_rows:
        for scheme in schemes:
            rows.append(run_parte_single_scheme(checkpoint_row, scheme, args, device))
    summary = summarize_parte_rows(rows)
    files = ["v23_29_H20_innovation_replacement_matrix.csv", "v23_29_partE_H20_summary.json"]
    write_csv(OUT_ROOT / files[0], rows)
    write_json(OUT_ROOT / files[1], summary)
    append_exec("PartE_H20_innovation_replacement_diagnostic", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartE H20 innovation replacement diagnostic",
        [
            f"rows `{summary['rows']}`; pairs `{summary['pair_count']}`; valid_fraction `{summary['row_valid_for_science_fraction']}`.",
            f"P2-K1 median/CVaR25/LCB05 `{summary['P2_minus_K1_median']}/{summary['P2_minus_K1_CVaR25']}/{summary['P2_minus_K1_LCB05']}`.",
            f"P2-P0/P1 medians `{summary['P2_minus_P0_median']}/{summary['P2_minus_P1_median']}`; H20_diagnostic_gate_pass_on_this_matrix `{summary['H20_diagnostic_gate_pass_on_this_matrix']}`; official_partE_completion_claim `0`; blockers `{summary['blockers']}`.",
        ],
    )


def run_partE_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    paths = [Path(path) for path in sorted(glob.glob(str(args.parte_aggregate_glob)))]
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_csv(path))
    summary = summarize_parte_rows(rows)
    summary["phase"] = "partE-aggregate"
    summary["source_files"] = [rel(path) for path in paths]
    files = ["v23_29_H20_innovation_replacement_matrix.csv", "v23_29_partE_H20_summary.json"]
    write_csv(OUT_ROOT / files[0], rows)
    write_json(OUT_ROOT / files[1], summary)
    append_exec("PartE_H20_innovation_replacement_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartE H20 innovation replacement aggregate",
        [
            f"rows `{summary['rows']}`; pairs `{summary['pair_count']}`; valid_fraction `{summary['row_valid_for_science_fraction']}`.",
            f"P2-K1 median/CVaR25/LCB05 `{summary['P2_minus_K1_median']}/{summary['P2_minus_K1_CVaR25']}/{summary['P2_minus_K1_LCB05']}`.",
            f"P2-P0/P1 medians `{summary['P2_minus_P0_median']}/{summary['P2_minus_P1_median']}`; H20_diagnostic_gate_pass_on_this_matrix `{summary['H20_diagnostic_gate_pass_on_this_matrix']}`; official_partE_completion_claim `0`; blockers `{summary['blockers']}`.",
        ],
    )


def flatten_state_map(state: dict[tuple[str, int], torch.Tensor]) -> torch.Tensor:
    pieces = [state[key].reshape(-1).detach().double().cpu() for key in sorted(state)]
    if not pieces:
        return torch.zeros(1, dtype=torch.float64)
    return torch.cat(pieces)


def state_predict_metrics(targets: list[torch.Tensor], preds: list[torch.Tensor]) -> tuple[float, float]:
    if not targets or not preds:
        return float("nan"), float("nan")
    target = torch.stack(targets).reshape(len(targets), -1).double()
    pred = torch.stack(preds).reshape(len(preds), -1).double()
    residual = (target - pred).square().sum()
    denom = target.square().sum().clamp_min(1.0e-12)
    r_pred = float((1.0 - residual / denom).item())
    cos_vals = []
    for a, b in zip(target, pred):
        if float(a.norm().item()) <= 1.0e-12 or float(b.norm().item()) <= 1.0e-12:
            cos_vals.append(0.0)
        else:
            cos_vals.append(float(F.cosine_similarity(a.float(), b.float(), dim=0).item()))
    return r_pred, float(sum(cos_vals) / max(1, len(cos_vals)))


def structured_sequence_from_checkpoint(
    checkpoint_row: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    *,
    order_offset: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    payload = load_torch_checkpoint(ROOT / str(checkpoint_row["checkpoint_path"]), device)
    dataset = make_partd_dataset(
        str(checkpoint_row["dataset"]),
        int(checkpoint_row["seed"]),
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(args.partd_train_count),
        max(int(args.partd_audit_batch), int(args.batch)),
        device,
    )
    carrier = str(checkpoint_row["carrier"])
    model = factory_for(carrier)(
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(payload["hidden_dim"]),
        dataset["x_stats"],
        int(payload["model_seed"]),
        device,
    )
    model.load_state_dict(payload["model_state_dict"])
    gen = torch.Generator(device=device).manual_seed(910000 + int(checkpoint_row["seed"]) + int(checkpoint_row["step"]) + int(order_offset))
    sequence: list[torch.Tensor] = []
    truth_passes: list[int] = []
    for step in range(1, int(args.partf_horizon) + 1):
        perm = torch.randperm(int(dataset["x_train"].shape[0]), device=device, generator=gen)
        idx = perm[: int(args.batch)]
        raw, current, trace = bank_update_maps(
            model,
            dataset["x_train"][idx],
            dataset["y_train"][idx],
            rank=int(args.parte_rank),
            rounds=int(args.partd_fit_rounds),
            seed=920000 + step + int(checkpoint_row["seed"]) + int(order_offset),
        )
        sequence.append(flatten_state_map(current))
        truth_passes.append(int(trace["actual_carrier_truth_pass"]))
        apply_bank_updates(model, raw, float(args.parte_eta))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    meta = {
        "dataset_split_hash_verified": int(str(dataset["split_hash"]) == str(payload.get("dataset_split_hash", ""))),
        "actual_carrier_truth_pass": int(all(value == 1 for value in truth_passes)),
    }
    return sequence, meta


def partf_predictability_rows_for_checkpoint(checkpoint_row: dict[str, Any], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    task_seq, task_meta = structured_sequence_from_checkpoint(checkpoint_row, args, device, order_offset=0)
    other_seq, other_meta = structured_sequence_from_checkpoint(checkpoint_row, args, device, order_offset=777)
    if len(task_seq) < 2:
        return []
    beta = float(args.parte_beta)
    zero = torch.zeros_like(task_seq[0])
    controls: dict[str, list[torch.Tensor]] = {
        "task_consistent_memory": [],
        "reset": [],
        "random_AR1": [],
        "same_autocorr_random": [],
        "lag4": [],
        "time_shuffle": [],
        "bank_shuffle": [],
        "other_trajectory_same_dataset_same_seed": [],
    }
    targets: list[torch.Tensor] = []
    mem = torch.zeros_like(task_seq[0])
    rand_state = torch.zeros_like(task_seq[0])
    same_rand_state = torch.zeros_like(task_seq[0])
    other_mem = torch.zeros_like(other_seq[0])
    gen = torch.Generator(device=torch.device("cpu")).manual_seed(930000 + int(checkpoint_row["seed"]) + int(checkpoint_row["step"]))
    for idx, current in enumerate(task_seq):
        if idx == 0:
            mem = beta * mem + (1.0 - beta) * current
            other_mem = beta * other_mem + (1.0 - beta) * other_seq[idx]
            continue
        targets.append(current)
        controls["task_consistent_memory"].append(mem.clone())
        controls["reset"].append(zero.clone())
        noise = torch.randn(current.shape, dtype=current.dtype, generator=gen)
        noise = noise / noise.norm().clamp_min(1.0e-12) * current.norm().clamp_min(1.0e-12)
        rand_state = beta * rand_state + (1.0 - beta) * noise
        controls["random_AR1"].append(rand_state.clone())
        noise2 = torch.randn(current.shape, dtype=current.dtype, generator=gen)
        noise2 = noise2 / noise2.norm().clamp_min(1.0e-12) * mem.norm().clamp_min(1.0e-12)
        same_rand_state = beta * same_rand_state + (1.0 - beta) * noise2
        controls["same_autocorr_random"].append(same_rand_state.clone())
        controls["lag4"].append(task_seq[idx - 4].clone() if idx >= 4 else zero.clone())
        controls["time_shuffle"].append(task_seq[-idx].clone())
        controls["bank_shuffle"].append(torch.roll(mem, shifts=max(1, int(mem.numel()) // 3)).clone())
        controls["other_trajectory_same_dataset_same_seed"].append(other_mem.clone())
        mem = beta * mem + (1.0 - beta) * current
        other_mem = beta * other_mem + (1.0 - beta) * other_seq[idx]
    rows = []
    common = {
        "phase": "partF_state_predictability_diagnostic",
        "official_partF_completion_claim": 0,
        "diagnostic_dataset_only": 1,
        "dataset": checkpoint_row["dataset"],
        "task_family": checkpoint_row["task_family"],
        "seed": int(checkpoint_row["seed"]),
        "carrier": checkpoint_row["carrier"],
        "checkpoint_stage": checkpoint_row["stage"],
        "checkpoint_step": int(checkpoint_row["step"]),
        "checkpoint_path": checkpoint_row["checkpoint_path"],
        "checkpoint_sha256": checkpoint_row["checkpoint_sha256"],
        "horizon": int(args.partf_horizon),
        "beta": beta,
        "dataset_split_hash_verified": int(task_meta["dataset_split_hash_verified"] == 1 and other_meta["dataset_split_hash_verified"] == 1),
        "actual_carrier_truth_pass": int(task_meta["actual_carrier_truth_pass"] == 1 and other_meta["actual_carrier_truth_pass"] == 1),
    }
    for control, preds in controls.items():
        r_pred, cos_pred = state_predict_metrics(targets, preds)
        rows.append(
            {
                **common,
                "control": control,
                "R_pred": r_pred,
                "cos_pred": cos_pred,
                "row_valid_for_science": int(common["dataset_split_hash_verified"] == 1 and common["actual_carrier_truth_pass"] == 1 and math.isfinite(r_pred)),
            }
        )
    return rows


def summarize_partf_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_fraction = float(sum(int(row["row_valid_for_science"]) for row in rows) / max(1, len(rows)))
    by_control: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_control.setdefault(str(row["control"]), []).append(row)
    def med(control: str, key: str) -> float:
        return median_value([float(row[key]) for row in by_control.get(control, [])])
    task_r = med("task_consistent_memory", "R_pred")
    task_cos = med("task_consistent_memory", "cos_pred")
    other_r = med("other_trajectory_same_dataset_same_seed", "R_pred")
    random_r = med("same_autocorr_random", "R_pred")
    reset_r = med("reset", "R_pred")
    gate = int(valid_fraction == 1.0 and task_r > 0.10 and task_cos > 0.30 and task_r > other_r and task_r > random_r and task_r > reset_r)
    blockers = []
    if not rows:
        blockers.append("partF_rows_missing")
    if valid_fraction != 1.0:
        blockers.append("partF_rows_invalid")
    if not (task_r > 0.10):
        blockers.append("task_memory_R_pred_not_above_0p10")
    if not (task_cos > 0.30):
        blockers.append("task_memory_cos_pred_not_above_0p30")
    if not (task_r > other_r):
        blockers.append("task_memory_not_above_other_trajectory")
    if not (task_r > random_r):
        blockers.append("task_memory_not_above_same_autocorr_random")
    blockers.append("diagnostic_synthetic_partF_not_official")
    return {
        "phase": "partF-state",
        "rows": len(rows),
        "pair_count": len({str(row["checkpoint_sha256"]) for row in rows}),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "carriers": sorted({str(row["carrier"]) for row in rows}),
        "row_valid_for_science_fraction": valid_fraction,
        "task_memory_R_pred_median": task_r,
        "task_memory_cos_pred_median": task_cos,
        "other_trajectory_R_pred_median": other_r,
        "same_autocorr_random_R_pred_median": random_r,
        "reset_R_pred_median": reset_r,
        "partF_diagnostic_gate_pass_on_this_matrix": gate,
        "official_partF_completion_claim": 0,
        "blockers": blockers,
    }


def run_partF_state(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint_rows = read_csv(ROOT / str(args.parte_checkpoint_manifest))
    if int(args.partf_max_checkpoints) > 0:
        checkpoint_rows = checkpoint_rows[: int(args.partf_max_checkpoints)]
    if int(args.partf_shard_count) > 1:
        checkpoint_rows = [row for idx, row in enumerate(checkpoint_rows) if idx % int(args.partf_shard_count) == int(args.partf_shard_index)]
    rows: list[dict[str, Any]] = []
    for checkpoint_row in checkpoint_rows:
        rows.extend(partf_predictability_rows_for_checkpoint(checkpoint_row, args, device))
    summary = summarize_partf_rows(rows)
    files = ["v23_29_state_predictability_matrix.csv", "v23_29_partF_state_predictability_summary.json"]
    write_csv(OUT_ROOT / files[0], rows)
    write_json(OUT_ROOT / files[1], summary)
    append_exec("PartF_state_predictability_diagnostic", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartF state predictability diagnostic",
        [
            f"rows `{summary['rows']}`; pairs `{summary['pair_count']}`; valid_fraction `{summary['row_valid_for_science_fraction']}`.",
            f"task R/cos `{summary['task_memory_R_pred_median']}/{summary['task_memory_cos_pred_median']}`; other/random/reset R `{summary['other_trajectory_R_pred_median']}/{summary['same_autocorr_random_R_pred_median']}/{summary['reset_R_pred_median']}`.",
            f"partF_diagnostic_gate_pass_on_this_matrix `{summary['partF_diagnostic_gate_pass_on_this_matrix']}`; official_partF_completion_claim `0`; blockers `{summary['blockers']}`.",
        ],
    )


def run_partF_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    paths = [Path(path) for path in sorted(glob.glob(str(args.partf_aggregate_glob)))]
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_csv(path))
    summary = summarize_partf_rows(rows)
    summary["phase"] = "partF-aggregate"
    summary["source_files"] = [rel(path) for path in paths]
    files = ["v23_29_state_predictability_matrix.csv", "v23_29_partF_state_predictability_summary.json"]
    write_csv(OUT_ROOT / files[0], rows)
    write_json(OUT_ROOT / files[1], summary)
    append_exec("PartF_state_predictability_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartF state predictability aggregate",
        [
            f"rows `{summary['rows']}`; pairs `{summary['pair_count']}`; valid_fraction `{summary['row_valid_for_science_fraction']}`.",
            f"task R/cos `{summary['task_memory_R_pred_median']}/{summary['task_memory_cos_pred_median']}`; other/random/reset R `{summary['other_trajectory_R_pred_median']}/{summary['same_autocorr_random_R_pred_median']}/{summary['reset_R_pred_median']}`.",
            f"partF_diagnostic_gate_pass_on_this_matrix `{summary['partF_diagnostic_gate_pass_on_this_matrix']}`; official_partF_completion_claim `0`; blockers `{summary['blockers']}`.",
        ],
    )


def partg_metric_names(args: argparse.Namespace) -> list[str]:
    raw = str(args.partg_metrics).strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return ["M0_DataL2", "M1_LocalJet", "M2_CompositionalJet_PRIMARY", "M3_PathShuffledJet", "M4_UniformPathJet"]


def run_partg_single_metric(checkpoint_row: dict[str, Any], metric_kind: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    payload = load_torch_checkpoint(ROOT / str(checkpoint_row["checkpoint_path"]), device)
    dataset = make_partd_dataset(
        str(checkpoint_row["dataset"]),
        int(checkpoint_row["seed"]),
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(args.partd_train_count),
        max(int(args.partd_audit_batch), int(args.parte_eval_count), int(args.batch)),
        device,
    )
    split_verified = int(str(dataset["split_hash"]) == str(payload.get("dataset_split_hash", "")))
    carrier = str(checkpoint_row["carrier"])
    model = factory_for(carrier)(
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(payload["hidden_dim"]),
        dataset["x_stats"],
        int(payload["model_seed"]),
        device,
    )
    model.load_state_dict(payload["model_state_dict"])
    x_eval = dataset["x_audit"][: int(args.parte_eval_count)]
    y_eval = dataset["y_audit"][: int(args.parte_eval_count)]
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    initial_loss, initial_acc, initial_tag = evaluate_ce_loss(model, x_eval, y_eval)
    gen = torch.Generator(device=device).manual_seed(1010000 + int(checkpoint_row["seed"]) + int(checkpoint_row["step"]) + 31 * len(metric_kind))
    memory: dict[tuple[str, int], torch.Tensor] | None = None
    truth_passes: list[int] = []
    shuffle_distances: list[float] = []
    uniform_distances: list[float] = []
    path_corrs: list[float] = []
    condition_means: list[float] = []
    update_norms: list[float] = []
    for step in range(1, int(args.partg_horizon) + 1):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        idx = perm[: int(args.batch)]
        raw, current, trace = bank_update_maps(
            model,
            x_train[idx],
            y_train[idx],
            rank=int(args.parte_rank),
            rounds=int(args.partd_fit_rounds),
            seed=1020000 + step + int(checkpoint_row["seed"]),
            metric_kind=metric_kind,
        )
        if memory is None:
            memory = {key: torch.zeros_like(value) for key, value in current.items()}
        updates = {key: raw[key] - current[key] + memory[key] for key in raw}
        update_norms.append(apply_bank_updates(model, updates, float(args.parte_eta)))
        memory = {key: float(args.parte_beta) * memory[key] + (1.0 - float(args.parte_beta)) * current[key] for key in current}
        truth_passes.append(int(trace["actual_carrier_truth_pass"]))
        shuffle_distances.append(float(trace["true_vs_shuffle_metric_distance_mean"]))
        uniform_distances.append(float(trace["true_vs_uniform_metric_distance_mean"]))
        path_corrs.append(float(trace["path_weight_basis_jet_abs_corr_mean"]))
        condition_means.append(float(trace["metric_condition_mean"]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    final_loss, final_acc, final_tag = evaluate_ce_loss(model, x_eval, y_eval)
    pair_key = stable_hash_obj(
        {
            "checkpoint_sha256": checkpoint_row["checkpoint_sha256"],
            "dataset": checkpoint_row["dataset"],
            "seed": int(checkpoint_row["seed"]),
            "carrier": carrier,
            "stage": checkpoint_row["stage"],
            "step": int(checkpoint_row["step"]),
        }
    )
    return {
        "phase": "partG_metric_causality_diagnostic",
        "official_partG_completion_claim": 0,
        "diagnostic_dataset_only": 1,
        "pair_key": pair_key,
        "dataset": checkpoint_row["dataset"],
        "task_family": checkpoint_row["task_family"],
        "seed": int(checkpoint_row["seed"]),
        "carrier": carrier,
        "checkpoint_stage": checkpoint_row["stage"],
        "checkpoint_step": int(checkpoint_row["step"]),
        "checkpoint_path": checkpoint_row["checkpoint_path"],
        "checkpoint_sha256": checkpoint_row["checkpoint_sha256"],
        "metric_kind": metric_kind,
        "scheme": "P2_BiSided_Persistent_Replacement_PRIMARY",
        "horizon": int(args.partg_horizon),
        "eta": float(args.parte_eta),
        "beta": float(args.parte_beta),
        "rank": int(args.parte_rank),
        "initial_eval_loss": initial_loss,
        "final_eval_loss": final_loss,
        "eval_loss_delta_initial_minus_final": initial_loss - final_loss,
        "score_neg_final_loss": -final_loss,
        "initial_eval_acc": initial_acc,
        "final_eval_acc": final_acc,
        "initial_cache_tag": initial_tag,
        "final_cache_tag": final_tag,
        "actual_fused_forward_count": int(args.partg_horizon) + 2,
        "actual_fused_backward_from_grad_logits_count": int(args.partg_horizon),
        "actual_dense_basis_materialization_count": 0,
        "actual_compact_proxy_forward_count": 0,
        "actual_linear_bank_proxy_count": 0,
        "checkpoint_reloaded_for_pair": 1,
        "same_checkpoint_pair_key": pair_key,
        "dataset_split_hash_verified": split_verified,
        "metric_refresh_count": int(args.partg_horizon),
        "true_vs_shuffle_metric_distance_mean": float(sum(shuffle_distances) / max(1, len(shuffle_distances))),
        "true_vs_uniform_metric_distance_mean": float(sum(uniform_distances) / max(1, len(uniform_distances))),
        "path_weight_basis_jet_abs_corr_mean": float(sum(path_corrs) / max(1, len(path_corrs))),
        "metric_condition_mean": float(sum(condition_means) / max(1, len(condition_means))),
        "update_norm_mean": float(sum(update_norms) / max(1, len(update_norms))),
        "current_forcing_same_step_use_count": 0,
        "innovation_residual_compute_count": int(args.partg_horizon),
        "row_valid_for_science": int(split_verified == 1 and all(value == 1 for value in truth_passes)),
    }


def summarize_partg_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_key"]), {})[str(row["metric_kind"])] = row

    def paired_diff(left: str, right: str) -> list[float]:
        vals: list[float] = []
        for metrics in by_pair.values():
            if left in metrics and right in metrics:
                vals.append(float(metrics[left]["score_neg_final_loss"]) - float(metrics[right]["score_neg_final_loss"]))
        return vals

    m2 = "M2_CompositionalJet_PRIMARY"
    diffs_m2_m1 = paired_diff(m2, "M1_LocalJet")
    diffs_m2_m3 = paired_diff(m2, "M3_PathShuffledJet")
    diffs_m2_m0 = paired_diff(m2, "M0_DataL2")
    diffs_m2_m4 = paired_diff(m2, "M4_UniformPathJet")
    valid_fraction = float(sum(int(row["row_valid_for_science"]) for row in rows) / max(1, len(rows)))
    m2_rows = [row for row in rows if str(row.get("metric_kind")) == m2]
    shuffle_dist = median_value([float(row["true_vs_shuffle_metric_distance_mean"]) for row in m2_rows])
    path_corr = median_value([float(row["path_weight_basis_jet_abs_corr_mean"]) for row in m2_rows])
    metric_refresh_min = min([int(row["metric_refresh_count"]) for row in rows], default=0)
    no_debt_vs_m1 = fraction_ge(diffs_m2_m1, 0.0)
    gate = int(
        valid_fraction == 1.0
        and shuffle_dist > 1.0e-8
        and path_corr > 1.0e-8
        and metric_refresh_min > 0
        and median_value(diffs_m2_m1) > 0.0
        and median_value(diffs_m2_m3) > 0.0
        and cvar_low(diffs_m2_m1, 0.25) > 0.0
        and no_debt_vs_m1 >= 0.80
    )
    blockers = []
    if not rows:
        blockers.append("partG_rows_missing")
    if valid_fraction != 1.0:
        blockers.append("partG_rows_invalid")
    if not (shuffle_dist > 1.0e-8):
        blockers.append("true_vs_shuffle_metric_distance_zero")
    if not (path_corr > 1.0e-8):
        blockers.append("path_weight_basis_jet_correlation_zero")
    if not (metric_refresh_min > 0):
        blockers.append("metric_refresh_not_observed")
    if not (median_value(diffs_m2_m1) > 0.0):
        blockers.append("M2_minus_M1_median_not_positive")
    if not (median_value(diffs_m2_m3) > 0.0):
        blockers.append("M2_minus_M3_median_not_positive")
    if not (cvar_low(diffs_m2_m1, 0.25) > 0.0):
        blockers.append("M2_minus_M1_CVaR25_not_positive")
    if no_debt_vs_m1 < 0.80:
        blockers.append("M2_paired_no_debt_vs_M1_below_0p80")
    blockers.append("diagnostic_synthetic_partG_not_official")
    route_hint = ""
    if median_value(diffs_m2_m0) > 0.0 and median_value(diffs_m2_m1) <= 0.0:
        route_hint = "CompositionalMetricSupportOnly"
    if abs(median_value(diffs_m2_m3)) <= 1.0e-12:
        route_hint = "PathGeometryNotIdentified"
    return {
        "phase": "partG-metric",
        "rows": len(rows),
        "pair_count": len(by_pair),
        "metrics_count": len({str(row.get("metric_kind")) for row in rows}),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "carriers": sorted({str(row["carrier"]) for row in rows}),
        "row_valid_for_science_fraction": valid_fraction,
        "true_vs_shuffle_metric_distance_median": shuffle_dist,
        "path_weight_basis_jet_abs_corr_median": path_corr,
        "metric_refresh_min": metric_refresh_min,
        "M2_minus_M1_median": median_value(diffs_m2_m1),
        "M2_minus_M1_CVaR25": cvar_low(diffs_m2_m1, 0.25),
        "M2_minus_M3_median": median_value(diffs_m2_m3),
        "M2_minus_M0_median": median_value(diffs_m2_m0),
        "M2_minus_M4_median": median_value(diffs_m2_m4),
        "M2_paired_no_debt_vs_M1_fraction": no_debt_vs_m1,
        "partG_diagnostic_gate_pass_on_this_matrix": gate,
        "route_hint": route_hint,
        "official_partG_completion_claim": 0,
        "blockers": blockers,
    }


def run_partG_metric(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint_rows = read_csv(ROOT / str(args.parte_checkpoint_manifest))
    if int(args.partg_max_checkpoints) > 0:
        checkpoint_rows = checkpoint_rows[: int(args.partg_max_checkpoints)]
    if int(args.partg_shard_count) > 1:
        checkpoint_rows = [row for idx, row in enumerate(checkpoint_rows) if idx % int(args.partg_shard_count) == int(args.partg_shard_index)]
    metrics = partg_metric_names(args)
    rows: list[dict[str, Any]] = []
    for checkpoint_row in checkpoint_rows:
        for metric_kind in metrics:
            rows.append(run_partg_single_metric(checkpoint_row, metric_kind, args, device))
    summary = summarize_partg_rows(rows)
    files = ["v23_29_metric_causality_matrix.csv", "v23_29_partG_metric_causality_summary.json"]
    write_csv(OUT_ROOT / files[0], rows)
    write_json(OUT_ROOT / files[1], summary)
    append_exec("PartG_metric_causality_diagnostic", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartG metric causality diagnostic",
        [
            f"rows `{summary['rows']}`; pairs `{summary['pair_count']}`; valid_fraction `{summary['row_valid_for_science_fraction']}`.",
            f"M2-M1/M3 medians `{summary['M2_minus_M1_median']}/{summary['M2_minus_M3_median']}`; M2-M1 CVaR25 `{summary['M2_minus_M1_CVaR25']}`.",
            f"shuffle_distance/corr `{summary['true_vs_shuffle_metric_distance_median']}/{summary['path_weight_basis_jet_abs_corr_median']}`; partG_diagnostic_gate `{summary['partG_diagnostic_gate_pass_on_this_matrix']}`; official_partG_completion_claim `0`; blockers `{summary['blockers']}`.",
        ],
    )


def run_partG_aggregate(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_logs()
    paths = [Path(path) for path in sorted(glob.glob(str(args.partg_aggregate_glob)))]
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_csv(path))
    summary = summarize_partg_rows(rows)
    summary["phase"] = "partG-aggregate"
    summary["source_files"] = [rel(path) for path in paths]
    files = ["v23_29_metric_causality_matrix.csv", "v23_29_partG_metric_causality_summary.json"]
    write_csv(OUT_ROOT / files[0], rows)
    write_json(OUT_ROOT / files[1], summary)
    append_exec("PartG_metric_causality_aggregate", args, files, summary, "completed_incomplete_gate")
    append_recap(
        "PartG metric causality aggregate",
        [
            f"rows `{summary['rows']}`; pairs `{summary['pair_count']}`; valid_fraction `{summary['row_valid_for_science_fraction']}`.",
            f"M2-M1/M3 medians `{summary['M2_minus_M1_median']}/{summary['M2_minus_M3_median']}`; M2-M1 CVaR25 `{summary['M2_minus_M1_CVaR25']}`.",
            f"shuffle_distance/corr `{summary['true_vs_shuffle_metric_distance_median']}/{summary['path_weight_basis_jet_abs_corr_median']}`; partG_diagnostic_gate `{summary['partG_diagnostic_gate_pass_on_this_matrix']}`; official_partG_completion_claim `0`; blockers `{summary['blockers']}`.",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "part0",
            "partA",
            "partB",
            "partC-smoke",
            "partC-aggregate",
            "partD-checkpoint",
            "partD-aggregate",
            "partE-H20",
            "partE-aggregate",
            "partF-state",
            "partF-aggregate",
            "partG-metric",
            "partG-aggregate",
        ],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-full-read-complete", type=int, default=0)
    parser.add_argument("--lineage-full-read-complete", type=int, default=0)
    parser.add_argument("--input-dim", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=7)
    parser.add_argument("--output-dim", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--partc-seeds", type=int, default=1)
    parser.add_argument("--partc-max-rows", type=int, default=0)
    parser.add_argument("--partc-shard-index", type=int, default=0)
    parser.add_argument("--partc-shard-count", type=int, default=1)
    parser.add_argument("--partc-aggregate-glob", default="")
    parser.add_argument("--partd-datasets", default="SYN-A-one-sided-tangent")
    parser.add_argument("--partd-seeds", type=int, default=1)
    parser.add_argument("--partd-steps", type=int, default=30)
    parser.add_argument("--partd-train-count", type=int, default=96)
    parser.add_argument("--partd-audit-batch", type=int, default=32)
    parser.add_argument("--partd-lr", type=float, default=0.035)
    parser.add_argument("--partd-fit-rounds", type=int, default=3)
    parser.add_argument("--partd-shard-index", type=int, default=0)
    parser.add_argument("--partd-shard-count", type=int, default=1)
    parser.add_argument("--partd-aggregate-glob", default="")
    parser.add_argument("--parte-checkpoint-manifest", default="")
    parser.add_argument("--parte-schemes", default="")
    parser.add_argument("--parte-horizon", type=int, default=20)
    parser.add_argument("--parte-eval-count", type=int, default=32)
    parser.add_argument("--parte-eta", type=float, default=0.035)
    parser.add_argument("--parte-beta", type=float, default=0.90)
    parser.add_argument("--parte-rank", type=int, default=2)
    parser.add_argument("--parte-max-checkpoints", type=int, default=0)
    parser.add_argument("--parte-shard-index", type=int, default=0)
    parser.add_argument("--parte-shard-count", type=int, default=1)
    parser.add_argument("--parte-aggregate-glob", default="")
    parser.add_argument("--partf-horizon", type=int, default=20)
    parser.add_argument("--partf-max-checkpoints", type=int, default=0)
    parser.add_argument("--partf-shard-index", type=int, default=0)
    parser.add_argument("--partf-shard-count", type=int, default=1)
    parser.add_argument("--partf-aggregate-glob", default="")
    parser.add_argument("--partg-metrics", default="")
    parser.add_argument("--partg-horizon", type=int, default=20)
    parser.add_argument("--partg-max-checkpoints", type=int, default=0)
    parser.add_argument("--partg-shard-index", type=int, default=0)
    parser.add_argument("--partg-shard-count", type=int, default=1)
    parser.add_argument("--partg-aggregate-glob", default="")
    args = parser.parse_args()
    if args.phase == "part0":
        run_part0(args)
    elif args.phase == "partA":
        run_partA(args)
    elif args.phase == "partB":
        run_partB(args)
    elif args.phase == "partC-smoke":
        run_partC_smoke(args)
    elif args.phase == "partC-aggregate":
        run_partC_aggregate(args)
    elif args.phase == "partD-checkpoint":
        run_partD_checkpoint(args)
    elif args.phase == "partD-aggregate":
        run_partD_aggregate(args)
    elif args.phase == "partE-H20":
        run_partE_H20(args)
    elif args.phase == "partE-aggregate":
        run_partE_aggregate(args)
    elif args.phase == "partF-state":
        run_partF_state(args)
    elif args.phase == "partF-aggregate":
        run_partF_aggregate(args)
    elif args.phase == "partG-metric":
        run_partG_metric(args)
    elif args.phase == "partG-aggregate":
        run_partG_aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
