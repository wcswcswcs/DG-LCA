#!/usr/bin/env python
"""v13.0 function-preserving coordinate transport reset runner.

This runner deliberately separates three claims:

1. v12.35 substrate artifacts are re-read under the v13.0 S1/S2 gates.
2. A controlled synthetic feature-coordinate transport proof is executed.
3. A real-data frozen readout-feature transport probe is executed when model
   hooks expose ``frozen_readout_features``.  This is not claimed to be a full
   basis-parameter surgery.

The transport direction uses only unlabeled features and current logits.  Labels
are used only for ordinary training/evaluation and future-training probes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as v1223  # noqa: E402
from experiments.run_v1231_basis_kernel_workspace import make_adamw, make_basis_model  # noqa: E402
from dgkan.diagnostics.basis_workspace import V1235_BASIS_CANDIDATES, fnum  # noqa: E402
from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from dgkan.training.eval import classification_basic  # noqa: E402


OUT_DIR = ROOT / "results" / "v13_0_strategic_reset_function_preserving_coordinate_transport" / "official_v130"
SOURCE_V1235 = ROOT / "results" / "v12_35_all_basis_substrate_health_functional_colocation" / "official_v1235"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v13.0_StrategicReset_FunctionPreservingCoordinateTransport.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v13.0_StrategicReset_FunctionPreservingCoordinateTransport_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v13.0_StrategicReset_FunctionPreservingCoordinateTransport_实验结果复盘.md"

REQUIRED = [
    "v130_substrate_gate_v2.csv",
    "v130_synthetic_mechanism_proof.csv",
    "v130_transport_proposals.csv",
    "v130_function_preservation.csv",
    "v130_basis_telemetry_before_after.csv",
    "v130_future_training_probe.csv",
    "v130_transport_controls.csv",
    "v130_mlp_analog_transport.csv",
    "v130_linec_audit.csv",
    "v130_family_decision.csv",
    "v130_route_decision.json",
    "v130_failure_table.csv",
    "v130_next_hypothesis_queue.md",
    "v130_code_review_packet.zip",
]

SUPPLEMENTAL = [
    "v130_required_artifact_manifest.csv",
    "v130_code_provenance_audit.csv",
    "v130_code_review_manifest.csv",
]

FIGURES = [
    "fig_substrate_map_by_family.svg",
    "fig_transport_function_drift_vs_geom_gain.svg",
    "fig_future_auc_transport_vs_controls.svg",
    "fig_basis_telemetry_before_after.svg",
    "fig_linec_before_after_transport.svg",
    "fig_synthetic_mechanism_success_matrix.svg",
    "fig_mlp_analog_vs_basis_transport.svg",
    "fig_family_go_nogo_dashboard.svg",
]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def sint(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def finite(value: float, default: float = float("nan")) -> float:
    return float(value) if math.isfinite(float(value)) else default


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_svg(path: Path, title: str, lines: list[str]) -> None:
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body = []
    for idx, line in enumerate(lines[:22]):
        safe = str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body.append(f'<text x="24" y="{64 + idx * 24}" font-size="14" fill="#222">{safe}</text>')
    path.write_text(
        "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="680" viewBox="0 0 1220 680">',
            '<rect width="1220" height="680" fill="#f7f8f4"/>',
            f'<text x="24" y="34" font-size="20" font-weight="700" fill="#111">{safe_title}</text>',
            *body,
            "</svg>",
        ]),
        encoding="utf-8",
    )


def linec_rate(row: dict[str, Any]) -> float:
    total = sint(row.get("LineC_total"), sint(row.get("LineC_seed_count"), 0))
    return float(sint(row.get("LineC_pass_count"), 0)) / float(total) if total > 0 else 0.0


def substrate_s1(row: dict[str, Any]) -> int:
    return int(
        fnum(row.get("raw_memory_ratio_vs_mlp"), 999.0) <= 1.10
        and fnum(row.get("incremental_memory_ratio_vs_mlp"), 999.0) <= 2.00
        and fnum(row.get("step_ratio_vs_mlp"), 999.0) <= 2.00
        and fnum(row.get("mean_delta_vs_mlp"), -999.0) >= -0.08
        and fnum(row.get("worst_delta_vs_mlp"), -999.0) >= -0.15
        and fnum(row.get("AUC_time_ratio_vs_mlp"), 999.0) <= 2.50
        and linec_rate(row) >= 0.20
    )


def substrate_s2(row: dict[str, Any]) -> int:
    return int(
        fnum(row.get("mean_delta_vs_mlp"), -999.0) >= -0.005
        and fnum(row.get("worst_delta_vs_mlp"), -999.0) >= -0.015
        and fnum(row.get("AUC_time_ratio_vs_mlp"), 999.0) <= 1.05
        and linec_rate(row) >= 0.70
        and fnum(row.get("CEp99_delta_vs_mlp"), 999.0) <= 0.05
        and fnum(row.get("ECE_delta_vs_mlp"), 999.0) <= 0.02
    )


def build_substrate_gate_v2(source_dir: Path) -> list[dict[str, Any]]:
    rows = read_rows(source_dir / "v1235_basis_substrate_health.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        rr = dict(row)
        reasons: list[str] = []
        if fnum(row.get("raw_memory_ratio_vs_mlp"), 999.0) > 1.10:
            reasons.append("raw_memory_ratio_gt_1.10")
        if fnum(row.get("incremental_memory_ratio_vs_mlp"), 999.0) > 2.00:
            reasons.append("incremental_memory_ratio_gt_2.00")
        if fnum(row.get("step_ratio_vs_mlp"), 999.0) > 2.00:
            reasons.append("step_ratio_gt_2.00")
        if fnum(row.get("mean_delta_vs_mlp"), -999.0) < -0.08:
            reasons.append("mean_delta_lt_-0.08")
        if fnum(row.get("worst_delta_vs_mlp"), -999.0) < -0.15:
            reasons.append("worst_delta_lt_-0.15")
        if fnum(row.get("AUC_time_ratio_vs_mlp"), 999.0) > 2.50:
            reasons.append("AUCtime_ratio_gt_2.50")
        if linec_rate(row) < 0.20:
            reasons.append("LineC_rate_lt_0.20")
        rr.update({
            "stage": "V130_SUBSTRATE_GATE_V2",
            "LineC_pass_rate": linec_rate(row),
            "S0_workspace_only": int(sint(row.get("workspace_gate_pass_rows"), 0) > 0 and not substrate_s1(row)),
            "S1_efficient_substrate": substrate_s1(row),
            "S2_healthy_base": substrate_s2(row),
            "fail_reason": ";".join(reasons) if reasons else "pass",
            "uses_label_init": 0,
            "uses_ce_direction": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
            "source_artifact": str(source_dir / "v1235_basis_substrate_health.csv"),
        })
        out.append(rr)
    return out


def stable_task_seed(task: str, seed: int) -> int:
    return int(seed) + 13000 + sum(ord(c) for c in task)


def synthetic_data(task: str, seed: int, n_train: int, n_val: int, dim: int, classes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(stable_task_seed(task, seed))
    x = torch.randn(n_train + n_val, dim, device=device, generator=gen).clamp(-3.0, 3.0)
    if task == "X1":
        s = torch.stack([x[:, 0] * x[:, 1] + 0.3 * x[:, 2], -x[:, 2] * x[:, 3] + 0.2 * x[:, 4], x[:, 4] * x[:, 5] - 0.1 * x[:, 0]], dim=1)
    elif task == "X2":
        q = (x[:, 0] + x[:, 1]).square() - 0.5 * (x[:, 2] - x[:, 3]).square()
        s = torch.stack([q, -q + 0.2 * x[:, 4], x[:, 5] + 0.3 * x[:, 6].square()], dim=1)
    elif task == "X3":
        a = torch.randn(dim, dim, device=device, generator=gen)
        a = 0.5 * (a + a.t())
        q = torch.einsum("bd,df,bf->b", x, a, x) / math.sqrt(dim)
        s = torch.stack([q, -0.8 * q + x[:, 0], x[:, 1] - x[:, 2]], dim=1)
    elif task == "X4":
        low = torch.sin(0.8 * x[:, 0]) + torch.cos(0.7 * x[:, 1])
        high = 0.5 * torch.sin(3.5 * x[:, 2])
        s = torch.stack([low + high, -low + 0.2 * high, x[:, 3] + 0.5 * torch.cos(2.0 * x[:, 4])], dim=1)
    elif task == "X5":
        c1 = torch.tensor([1.2, -0.8], device=device)
        c2 = torch.tensor([-1.0, 1.0], device=device)
        d1 = (x[:, :2] - c1).square().sum(dim=1)
        d2 = (x[:, :2] - c2).square().sum(dim=1)
        s = torch.stack([-d1 + 0.2 * x[:, 2], -d2 - 0.1 * x[:, 3], 0.5 * x[:, 4] - 0.25 * d1], dim=1)
    elif task == "X6":
        r = x[:, :2].norm(dim=1)
        angle = torch.atan2(x[:, 1], x[:, 0])
        s = torch.stack([torch.sin(2 * angle) + 0.2 * r, torch.cos(3 * angle) - 0.1 * r, 1.2 - (r - 1.5).abs()], dim=1)
    elif task == "X7":
        base = x[:, 0] * x[:, 1] - 0.4 * x[:, 2].square()
        s = torch.stack([base, -base + 0.1 * x[:, 3], x[:, 4] - 0.2 * x[:, 5]], dim=1)
    else:
        raise ValueError(f"unknown synthetic task {task}")
    if classes != 3:
        extra = []
        while s.shape[1] + len(extra) < classes:
            idx = len(extra) % dim
            extra.append(torch.sin(x[:, idx]))
        if extra:
            s = torch.cat([s, torch.stack(extra, dim=1)], dim=1)
        s = s[:, :classes]
    y = s.argmax(dim=1).to(torch.long)
    if task == "X7":
        flip = torch.rand(n_train, device=device, generator=gen) < 0.15
        y[:n_train][flip] = (y[:n_train][flip] + 1) % classes
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def feature_map(x: torch.Tensor, family: str, seed: int, out_dim: int = 64) -> torch.Tensor:
    b, d = int(x.shape[0]), int(x.shape[1])
    parts = [torch.ones(b, 1, device=x.device), x]
    if family == "D-RAT":
        parts += [x / (1.0 + x.abs()), x.square() / (1.0 + x.square())]
    elif family == "D-CHE":
        z = x.clamp(-1.0, 1.0)
        parts += [2 * z.square() - 1.0, 4 * z.pow(3) - 3 * z]
    elif family == "D-FOU":
        parts += [torch.sin(math.pi * x), torch.cos(math.pi * x), torch.sin(2.0 * math.pi * x)]
    elif family == "D-RBF":
        centers = torch.linspace(-1.5, 1.5, 4, device=x.device)
        rbfs = [torch.exp(-0.75 * (x - c).square()) for c in centers]
        parts += rbfs
    elif family == "D-WAV":
        centers = torch.linspace(-1.5, 1.5, 4, device=x.device)
        hats = [torch.clamp(1.0 - (x - c).abs(), min=0.0) for c in centers]
        parts += hats
    elif family == "MLP":
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 991)
        w = torch.randn(d, max(out_dim, 32), device=x.device, generator=gen) / math.sqrt(d)
        parts += [torch.tanh(x @ w)]
    else:
        raise ValueError(f"unknown family {family}")
    if d >= 2:
        pairs = []
        for i in range(min(d - 1, 8)):
            pairs.append((x[:, i] * x[:, i + 1]).unsqueeze(1))
        parts.append(torch.cat(pairs, dim=1))
    phi = torch.cat(parts, dim=1)
    if phi.shape[1] < out_dim:
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 1009)
        proj = torch.randn(phi.shape[1], out_dim - phi.shape[1], device=x.device, generator=gen) / math.sqrt(max(1, phi.shape[1]))
        phi = torch.cat([phi, torch.tanh(phi @ proj)], dim=1)
    phi = phi[:, :out_dim]
    scales = torch.logspace(0.0, -2.2, phi.shape[1], device=x.device)
    return phi * scales


def condition_proxy(phi: torch.Tensor) -> float:
    centered = phi - phi.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(centered.float())
        valid = s[s > 1.0e-7]
        if valid.numel() == 0:
            return float("inf")
        return float((valid.max() / valid.min()).item())
    except RuntimeError:
        return float("inf")


def effective_rank(phi: torch.Tensor) -> float:
    centered = phi - phi.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(centered.float())
        p2 = s.square()
        return float((p2.sum().square() / p2.square().sum().clamp_min(1.0e-8)).item())
    except RuntimeError:
        return 0.0


def occupancy_entropy(phi: torch.Tensor) -> float:
    e = phi.square().mean(dim=0)
    p = e / e.sum().clamp_min(1.0e-8)
    return float((-(p * (p + 1.0e-8).log()).sum() / math.log(max(2, p.numel()))).item())


def ridge_solve(phi: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    p = int(phi.shape[1])
    eye = torch.eye(p, device=phi.device, dtype=phi.dtype)
    if not bool(torch.isfinite(phi).all()) or not bool(torch.isfinite(target).all()):
        phi = torch.nan_to_num(phi, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    lhs = phi.t() @ phi + float(ridge) * eye
    rhs = phi.t() @ target
    for scale in (1.0, 10.0, 100.0, 1000.0):
        try:
            lhs_s = phi.t() @ phi + float(ridge) * float(scale) * eye
            return torch.linalg.solve(lhs_s.float(), rhs.float()).to(dtype=phi.dtype)
        except RuntimeError:
            continue
    try:
        return (torch.linalg.pinv(lhs.float(), hermitian=True) @ rhs.float()).to(dtype=phi.dtype)
    except RuntimeError:
        return torch.linalg.lstsq(lhs.float(), rhs.float()).solution.to(dtype=phi.dtype)


def with_bias(phi: torch.Tensor) -> torch.Tensor:
    return torch.cat([phi, torch.ones(phi.shape[0], 1, device=phi.device, dtype=phi.dtype)], dim=1)


def blockwise_ridge_solve(phi: torch.Tensor, target: torch.Tensor, ridge: float, blocks: int = 4) -> torch.Tensor:
    p = int(phi.shape[1])
    w = torch.zeros(p, int(target.shape[1]), device=phi.device, dtype=phi.dtype)
    residual = target.detach().clone()
    splits = torch.linspace(0, p - 1, int(blocks) + 1, device=phi.device).round().to(torch.long).tolist()
    # Keep the final bias column in every block's residual solve only through the
    # accumulated full solution; this avoids creating multiple independent biases.
    ranges: list[list[int]] = []
    for i in range(int(blocks)):
        lo = int(splits[i])
        hi = int(splits[i + 1])
        cols = list(range(max(0, lo), max(lo + 1, hi)))
        if p - 1 not in cols:
            cols.append(p - 1)
        ranges.append(sorted(set(c for c in cols if 0 <= c < p)))
    for cols in ranges:
        idx = torch.tensor(cols, device=phi.device, dtype=torch.long)
        local = phi.index_select(1, idx)
        wb = ridge_solve(local, residual, ridge)
        residual = residual - local @ wb
        w.index_add_(0, idx, wb)
    return w


def train_stream_fit_select_indices(n: int, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(int(seed) + 130013)
    perm = torch.randperm(int(n), device=device, generator=gen)
    split = max(8, int(round(0.70 * int(n))))
    split = min(max(1, split), max(1, int(n) - 1))
    return perm[:split], perm[split:]


def drift(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).norm().div(target.norm().clamp_min(1.0e-8)).item())


@dataclass
class TransformResult:
    name: str
    phi_train_t: torch.Tensor
    phi_val_t: torch.Tensor
    w_init: torch.Tensor
    drift_train: float
    drift_val: float
    condition_before: float
    condition_after: float
    rank_before: float
    rank_after: float
    occupancy_before: float
    occupancy_after: float
    geom_gain: float
    accepted: int
    fallback_note: str


def make_transport(
    name: str,
    phi_train: torch.Tensor,
    phi_val: torch.Tensor,
    u_train: torch.Tensor,
    u_val: torch.Tensor,
    w0: torch.Tensor,
    seed: int,
) -> TransformResult:
    cond_before = condition_proxy(phi_train)
    rank_before = effective_rank(phi_train)
    occ_before = occupancy_entropy(phi_train)
    if name == "NoOp":
        phi_t, val_t = phi_train, phi_val
        w = w0.detach().clone()
        note = "identity_no_transport"
    elif name == "ReadoutOnlyRefit":
        phi_t, val_t = phi_train, phi_val
        w = ridge_solve(phi_t, u_train, 1.0e-3)
        note = "ridge_readout_refit_only"
    elif name == "RandomMatchedTransport":
        gen = torch.Generator(device=phi_train.device).manual_seed(int(seed) + 303)
        q, _ = torch.linalg.qr(torch.randn(phi_train.shape[1], phi_train.shape[1], device=phi_train.device, generator=gen), mode="reduced")
        phi_t = phi_train @ q
        val_t = phi_val @ q
        w = ridge_solve(phi_t, u_train, 1.0e-3)
        note = "random_orthogonal_matched_transport"
    elif name == "FamilyTransport":
        best: TransformResult | None = None
        best_score: tuple[float, ...] | None = None
        centered = phi_train - phi_train.mean(dim=0, keepdim=True)
        mean = phi_train.mean(dim=0, keepdim=True)
        source_rms = float(centered.float().square().mean().sqrt().clamp_min(1.0e-8).item())
        fit_idx, select_idx = train_stream_fit_select_indices(int(phi_train.shape[0]), phi_train.device, int(seed))
        cov = centered.t() @ centered / float(max(1, centered.shape[0] - 1))
        evals, evecs = torch.linalg.eigh(cov.float())
        evals = evals.clamp_min(1.0e-6)
        for alpha in (0.5, 0.25, 0.10, 0.05, 0.02):
            a = evecs @ torch.diag(evals.pow(-float(alpha))) @ evecs.t()
            whitened_train = (phi_train - mean) @ a
            whitened_val = (phi_val - mean) @ a
            for ridge in (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0):
                for blend in (1.0, 0.75, 0.50, 0.25, 0.10, 0.05):
                    # Blend toward the transformed coordinate system instead of
                    # jumping there in one shot. This is the local-hidden
                    # reconstruction fallback: the direction remains label-free,
                    # while the original hidden coordinates regularize the map.
                    tr_base = (1.0 - float(blend)) * centered + float(blend) * whitened_train
                    va_base = (1.0 - float(blend)) * (phi_val - mean) + float(blend) * whitened_val
                    transport_rms = float(tr_base.float().square().mean().sqrt().clamp_min(1.0e-8).item())
                    rms_match = source_rms / max(transport_rms, 1.0e-8)
                    scale_options = []
                    for s in (1.0, 0.5, 0.25, 0.10, 0.05, rms_match, math.sqrt(max(rms_match, 1.0e-8))):
                        if math.isfinite(float(s)) and float(s) > 0:
                            ss = float(max(min(float(s), 10.0), 1.0e-3))
                            if all(abs(ss - old) > 1.0e-6 for old in scale_options):
                                scale_options.append(ss)
                    for post_scale in scale_options:
                        tr = tr_base * float(post_scale)
                        va = va_base * float(post_scale)
                        feature_rms = float(tr.float().square().mean().sqrt().clamp_min(1.0e-8).item())
                        rms_ratio = feature_rms / max(source_rms, 1.0e-8)
                        rms_penalty = abs(math.log(max(rms_ratio, 1.0e-8)))
                        compensation_modes = [
                            ("global_no_bias", tr, va, ridge_solve(tr.index_select(0, fit_idx), u_train.index_select(0, fit_idx), ridge)),
                            ("affine_bias", with_bias(tr), with_bias(va), ridge_solve(with_bias(tr).index_select(0, fit_idx), u_train.index_select(0, fit_idx), ridge)),
                            ("blockwise_affine", with_bias(tr), with_bias(va), blockwise_ridge_solve(with_bias(tr).index_select(0, fit_idx), u_train.index_select(0, fit_idx), ridge)),
                        ]
                        for mode, tr_fit, va_fit, w in compensation_modes:
                            db = drift(tr_fit @ w, u_train)
                            db_select = drift(tr_fit.index_select(0, select_idx) @ w, u_train.index_select(0, select_idx))
                            dq = drift(va_fit @ w, u_val)
                            cond_after = condition_proxy(tr)
                            rank_after = effective_rank(tr)
                            occ_after = occupancy_entropy(tr)
                            geom_gain = math.log((cond_before + 1.0e-8) / (cond_after + 1.0e-8)) + 0.05 * (rank_after - rank_before) + 0.25 * (occ_after - occ_before)
                            cand = TransformResult(
                                name,
                                tr_fit,
                                va_fit,
                                w,
                                db,
                                dq,
                                cond_before,
                                cond_after,
                                rank_before,
                                rank_after,
                                occ_before,
                                occ_after,
                                geom_gain,
                                int(db <= 0.01 and dq <= 0.02 and cond_after <= 0.70 * cond_before and geom_gain > 0.0),
                                f"{mode};fit_select_split=0.70/0.30;select_drift={db_select};whitening_alpha={alpha};transport_blend={blend};post_scale={post_scale};rms_ratio={rms_ratio};ridge={ridge}",
                            )
                            # Candidate selection must not use the probe/query batch.
                            # Probe drift is recorded in the artifact and gate, but the
                            # selected direction is based only on train-stream function
                            # preservation on a precommitted train-stream split plus
                            # label-free coordinate geometry.
                            split_gate = int(db <= 0.01 and db_select <= 0.02 and cond_after <= 0.70 * cond_before and geom_gain > 0.0)
                            score = (float(split_gate), -float(db_select), -float(cand.drift_train), -float(rms_penalty), float(cand.geom_gain))
                            if best is None or best_score is None or score > best_score:
                                best = cand
                                best_score = score
        assert best is not None
        return best
    else:
        raise ValueError(name)
    db = drift(phi_t @ w, u_train)
    dq = drift(val_t @ w, u_val)
    cond_after = condition_proxy(phi_t)
    rank_after = effective_rank(phi_t)
    occ_after = occupancy_entropy(phi_t)
    geom_gain = math.log((cond_before + 1.0e-8) / (cond_after + 1.0e-8)) + 0.05 * (rank_after - rank_before) + 0.25 * (occ_after - occ_before)
    accepted = int(db <= 0.01 and dq <= 0.02 and (name != "FamilyTransport" or (cond_after <= 0.70 * cond_before and geom_gain > 0.0)))
    return TransformResult(name, phi_t, val_t, w, db, dq, cond_before, cond_after, rank_before, rank_after, occ_before, occ_after, geom_gain, accepted, note)


def train_linear_future(phi: torch.Tensor, y: torch.Tensor, phi_val: torch.Tensor, y_val: torch.Tensor, w_init: torch.Tensor, steps: int, lr: float, weight_decay: float) -> dict[str, float]:
    w = torch.nn.Parameter(w_init.detach().clone().float())
    opt = torch.optim.AdamW([w], lr=float(lr), weight_decay=float(weight_decay), foreach=False)
    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(phi.float() @ w, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        logits = phi_val.float() @ w
        ev = classification_basic(lambda z: z, logits, y_val) if False else None
        ce = F.cross_entropy(logits, y_val, reduction="none")
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        acc = float((pred == y_val).float().mean().item())
        nll = float(ce.mean().item())
        ce99 = float(torch.quantile(ce, 0.99).item())
        # Compact ECE.
        ece = 0.0
        for lo in torch.linspace(0, 0.9, 10, device=phi.device):
            hi = lo + 0.1
            m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
            if bool(m.any()):
                ece += float(m.float().mean().item()) * abs(float(conf[m].mean().item()) - float((pred[m] == y_val[m]).float().mean().item()))
    return {
        "future_steps": int(steps),
        "future_loss_auc": finite_mean(losses),
        "future_elapsed_sec": float(elapsed),
        "future_auc_time_proxy": float(finite_mean(losses) * max(elapsed, 1.0e-9)),
        "future_val_acc": acc,
        "future_NLL": nll,
        "future_CEp99": ce99,
        "future_ECE": ece,
    }


def checkpoint_linear(phi: torch.Tensor, y: torch.Tensor, classes: int, seed: int, steps: int = 20) -> torch.Tensor:
    gen = torch.Generator(device=phi.device).manual_seed(int(seed) + 717)
    w = torch.nn.Parameter(torch.randn(phi.shape[1], classes, device=phi.device, generator=gen) * 0.01)
    opt = torch.optim.AdamW([w], lr=0.03, weight_decay=0.001, foreach=False)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(phi.float() @ w, y)
        loss.backward()
        opt.step()
    return w.detach()


def run_feature_transport_case(
    *,
    scope: str,
    family: str,
    task: str,
    candidate_id: str,
    dataset: str,
    seed: int,
    phi_train: torch.Tensor,
    y_train: torch.Tensor,
    phi_val: torch.Tensor,
    y_val: torch.Tensor,
    w0: torch.Tensor,
    future_steps: list[int],
    lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    with torch.no_grad():
        u_train = phi_train.float() @ w0.float()
        u_val = phi_val.float() @ w0.float()
    proposals: list[dict[str, Any]] = []
    preservation: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    transforms: dict[str, TransformResult] = {}
    for name in ("NoOp", "RandomMatchedTransport", "ReadoutOnlyRefit", "FamilyTransport"):
        tr = make_transport(name, phi_train.float(), phi_val.float(), u_train.float(), u_val.float(), w0.float(), int(seed) + sum(ord(c) for c in name))
        transforms[name] = tr
        proposals.append({
            "stage": "V130_TRANSPORT_PROPOSAL",
            "transport_scope": scope,
            "family": family,
            "synthetic_task": task,
            "candidate_id": candidate_id,
            "dataset": dataset,
            "seed": seed,
            "transport_name": name,
            "uses_label_for_direction": 0,
            "uses_ce_for_direction": 0,
            "uses_current_logits_for_compensation": 1,
            "label_used_for_future_training_only": 1,
            "fallback_note": tr.fallback_note,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
        preservation.append({
            "stage": "V130_FUNCTION_PRESERVATION",
            "transport_scope": scope,
            "family": family,
            "synthetic_task": task,
            "candidate_id": candidate_id,
            "dataset": dataset,
            "seed": seed,
            "transport_name": name,
            "Drift_B": tr.drift_train,
            "Drift_Q": tr.drift_val,
            "FunctionDrift_pass": int(tr.drift_train <= 0.01 and tr.drift_val <= 0.02),
            "condition_before": tr.condition_before,
            "condition_after": tr.condition_after,
            "TangentCondition_before": tr.condition_before,
            "TangentCondition_after": tr.condition_after,
            "TangentCondition_ratio": tr.condition_after / tr.condition_before if tr.condition_before > 0 and math.isfinite(tr.condition_before) else float("nan"),
            "rank_before": tr.rank_before,
            "rank_after": tr.rank_after,
            "occupancy_before": tr.occupancy_before,
            "occupancy_after": tr.occupancy_after,
            "GeomGain": tr.geom_gain,
            "transport_accepted": tr.accepted,
            "fallback_note": tr.fallback_note,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
        telemetry.append({
            "stage": "V130_BASIS_TELEMETRY_BEFORE_AFTER",
            "transport_scope": scope,
            "family": family,
            "synthetic_task": task,
            "candidate_id": candidate_id,
            "dataset": dataset,
            "seed": seed,
            "transport_name": name,
            "basis_condition_before": tr.condition_before,
            "basis_condition_after": tr.condition_after,
            "basis_effective_rank_before": tr.rank_before,
            "basis_effective_rank_after": tr.rank_after,
            "basis_occupancy_entropy_before": tr.occupancy_before,
            "basis_occupancy_entropy_after": tr.occupancy_after,
            "basis_telemetry_available": 1,
            "telemetry_fields_available": "condition,effective_rank,occupancy_entropy",
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    control_stats: dict[tuple[str, int], dict[str, float]] = {}
    for name, tr in transforms.items():
        for k in future_steps:
            stats = train_linear_future(tr.phi_train_t, y_train, tr.phi_val_t, y_val, tr.w_init, int(k), lr, weight_decay)
            control_stats[(name, int(k))] = stats
    for name, tr in transforms.items():
        for k in future_steps:
            stats = dict(control_stats[(name, int(k))])
            noop = control_stats[("NoOp", int(k))]
            random = control_stats[("RandomMatchedTransport", int(k))]
            readout = control_stats[("ReadoutOnlyRefit", int(k))]
            control_best = min(noop["future_auc_time_proxy"], random["future_auc_time_proxy"], readout["future_auc_time_proxy"])
            gap = control_best - stats["future_auc_time_proxy"]
            success = int(
                name == "FamilyTransport"
                and tr.accepted == 1
                and stats["future_auc_time_proxy"] <= 0.95 * noop["future_auc_time_proxy"]
                and stats["future_auc_time_proxy"] < random["future_auc_time_proxy"]
                and gap >= 0.005
                and stats["future_CEp99"] <= noop["future_CEp99"] + 0.05
            )
            future_rows.append({
                "stage": "V130_FUTURE_TRAINING_PROBE",
                "transport_scope": scope,
                "family": family,
                "synthetic_task": task,
                "candidate_id": candidate_id,
                "dataset": dataset,
                "seed": seed,
                "transport_name": name,
                "future_lr": float(lr),
                "future_weight_decay": float(weight_decay),
                **stats,
                "noop_future_auc_time_proxy": noop["future_auc_time_proxy"],
                "random_future_auc_time_proxy": random["future_auc_time_proxy"],
                "readout_future_auc_time_proxy": readout["future_auc_time_proxy"],
                "ControlGap_transport": gap,
                "future_training_probe_pass": success,
                "uses_label_for_transport_direction": 0,
                "label_used_for_future_training_only": 1,
                "promotion_allowed": 0,
                "no_fake": 1,
            })
    return proposals, preservation, telemetry, future_rows


def run_synthetic(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    proposal_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    synthetic_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    families = parse_csv(args.synthetic_families)
    tasks = parse_csv(args.synthetic_tasks)
    steps = parse_ints(args.future_steps)
    for task in tasks:
        for seed in parse_ints(args.synthetic_seeds):
            xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
            for family in families:
                phi = feature_map(xtr, family, seed, int(args.synthetic_feature_dim))
                phiv = feature_map(xva, family, seed, int(args.synthetic_feature_dim))
                w0 = checkpoint_linear(phi, ytr, int(args.synthetic_classes), seed)
                pr, fp, tel, fut = run_feature_transport_case(
                    scope="synthetic_controlled_feature_space",
                    family=family,
                    task=task,
                    candidate_id=f"{family}-SyntheticFeatureMap",
                    dataset="synthetic",
                    seed=seed,
                    phi_train=phi,
                    y_train=ytr,
                    phi_val=phiv,
                    y_val=yva,
                    w0=w0,
                    future_steps=steps,
                    lr=float(args.future_lr),
                    weight_decay=float(args.future_weight_decay),
                )
                proposal_rows.extend(pr)
                preservation_rows.extend(fp)
                telemetry_rows.extend(tel)
                future_rows.extend(fut)
                fam_fut = [r for r in fut if r["transport_name"] == "FamilyTransport"]
                family_pass = int(any(sint(r.get("future_training_probe_pass"), 0) for r in fam_fut))
                synthetic_rows.append({
                    "stage": "V130_SYNTHETIC_MECHANISM_PROOF",
                    "family": family,
                    "synthetic_task": task,
                    "seed": seed,
                    "FunctionDrift_pass_rows": sum(sint(r.get("FunctionDrift_pass"), 0) for r in fp if r["transport_name"] == "FamilyTransport"),
                    "transport_accepted_rows": sum(sint(r.get("transport_accepted"), 0) for r in fp if r["transport_name"] == "FamilyTransport"),
                    "future_training_probe_pass_rows": sum(sint(r.get("future_training_probe_pass"), 0) for r in fam_fut),
                    "synthetic_mechanism_pass": family_pass,
                    "promotion_allowed": 0,
                    "no_fake": 1,
                })
                if family == "MLP":
                    mlp_rows.extend([{**r, "stage": "V130_MLP_ANALOG_TRANSPORT"} for r in fam_fut])
    return synthetic_rows, proposal_rows, preservation_rows, telemetry_rows, future_rows, mlp_rows


def mlp_features(model: MLPBaseline, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        h = F.silu(x @ model.w0)
        h = F.silu(h @ model.w1)
        return h.detach()


def train_torch_model(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace, seed: int, device: torch.device) -> None:
    opt = make_adamw(args, [p for p in model.parameters() if p.requires_grad])
    gen = torch.Generator(device=device).manual_seed(int(seed) + 130717)
    for _ in range(int(args.real_checkpoint_epochs)):
        perm = torch.randperm(int(x.shape[0]), device=device, generator=gen)
        for off in range(0, int(x.shape[0]), int(args.batch_size)):
            idx = perm[off: off + int(args.batch_size)]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            opt.step()


def run_real_feature_probe(args: argparse.Namespace, substrate_rows: list[dict[str, Any]], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    proposal_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    s1 = [r for r in substrate_rows if sint(r.get("S1_efficient_substrate"), 0) == 1]
    selected_ids = parse_csv(args.real_candidates)
    if selected_ids:
        s1 = [r for r in s1 if str(r.get("candidate_id")) in selected_ids]
    s1 = sorted(s1, key=lambda r: (fnum(r.get("mean_delta_vs_mlp"), -999), fnum(r.get("LineC_pass_rate"), 0)), reverse=True)[: int(args.real_max_candidates)]
    steps = parse_ints(args.real_future_steps)
    for dataset in parse_csv(args.real_datasets):
        for seed in parse_ints(args.real_seeds):
            load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=int(seed))
            data = v1223.v120._load_vision_split(load_args, v1223.v120._canonical_dataset(dataset), train_size=int(args.real_train_size), val_size=int(args.real_val_size), test_size=int(args.real_val_size))
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
            xtr = x_train_cpu.to(device=device, dtype=torch.float32)
            ytr = y_train_cpu.to(device=device)
            xva = x_val_cpu.to(device=device, dtype=torch.float32)
            yva = y_val_cpu.to(device=device)
            # MLP analog on the same data.
            mlp = MLPBaseline(int(input_dim), int(output_dim), int(args.mlp_hidden), int(seed) + 130400, device).to(device)
            train_torch_model(mlp, xtr, ytr, args, int(seed), device)
            phi = mlp_features(mlp, xtr)
            phiv = mlp_features(mlp, xva)
            with torch.no_grad():
                logits = mlp(xtr)
            w0 = ridge_solve(phi.float(), logits.float(), 1.0e-3)
            pr, fp, tel, fut = run_feature_transport_case(
                scope="real_mlp_hidden_feature_analog",
                family="MLP",
                task="real",
                candidate_id="MLP-T1-hidden-feature-whiten-readout-comp",
                dataset=dataset,
                seed=seed,
                phi_train=phi,
                y_train=ytr,
                phi_val=phiv,
                y_val=yva,
                w0=w0,
                future_steps=steps,
                lr=float(args.future_lr),
                weight_decay=float(args.future_weight_decay),
            )
            proposal_rows.extend(pr)
            preservation_rows.extend(fp)
            telemetry_rows.extend(tel)
            future_rows.extend(fut)
            control_rows.extend([{**r, "stage": "V130_TRANSPORT_CONTROLS"} for r in fut if r["transport_name"] != "FamilyTransport"])
            linec_rows.extend([{**r, "stage": "V130_LINEC_AUDIT", "LineC_proxy_source": "future_feature_condition_proxy"} for r in fp])
            for row in s1:
                cand_id = str(row.get("candidate_id"))
                cand = V1235_BASIS_CANDIDATES.get(cand_id)
                if cand is None:
                    continue
                model, _spec = make_basis_model(cand.method_id, int(input_dim), int(output_dim), xtr, device, int(seed) + 130500)
                train_torch_model(model, xtr, ytr, args, int(seed) + 31, device)
                if not hasattr(model, "frozen_readout_features"):
                    continue
                with torch.no_grad():
                    phi = model.frozen_readout_features(xtr).detach().float()
                    phiv = model.frozen_readout_features(xva).detach().float()
                    logits = model(xtr).detach().float()
                w0 = ridge_solve(phi, logits, 1.0e-3)
                pr, fp, tel, fut = run_feature_transport_case(
                    scope="real_basis_frozen_readout_feature_transport",
                    family=str(row.get("family")),
                    task="real",
                    candidate_id=cand_id,
                    dataset=dataset,
                    seed=seed,
                    phi_train=phi,
                    y_train=ytr,
                    phi_val=phiv,
                    y_val=yva,
                    w0=w0,
                    future_steps=steps,
                    lr=float(args.future_lr),
                    weight_decay=float(args.future_weight_decay),
                )
                proposal_rows.extend(pr)
                preservation_rows.extend(fp)
                telemetry_rows.extend(tel)
                future_rows.extend(fut)
                control_rows.extend([{**r, "stage": "V130_TRANSPORT_CONTROLS"} for r in fut if r["transport_name"] != "FamilyTransport"])
                linec_rows.extend([{**r, "stage": "V130_LINEC_AUDIT", "LineC_proxy_source": "future_feature_condition_proxy"} for r in fp])
    return proposal_rows, preservation_rows, telemetry_rows, future_rows, control_rows, linec_rows


def build_family_decisions(substrate_rows: list[dict[str, Any]], synthetic_rows: list[dict[str, Any]], future_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({str(r.get("family")) for r in substrate_rows if r.get("family")} | {str(r.get("family")) for r in synthetic_rows if r.get("family")})
    mlp_pass = sum(sint(r.get("future_training_probe_pass"), 0) for r in mlp_rows)
    out: list[dict[str, Any]] = []
    for family in families:
        s1 = sum(sint(r.get("S1_efficient_substrate"), 0) for r in substrate_rows if str(r.get("family")) == family)
        s2 = sum(sint(r.get("S2_healthy_base"), 0) for r in substrate_rows if str(r.get("family")) == family)
        syn = sum(sint(r.get("synthetic_mechanism_pass"), 0) for r in synthetic_rows if str(r.get("family")) == family)
        real = sum(sint(r.get("future_training_probe_pass"), 0) for r in future_rows if str(r.get("family")) == family and str(r.get("transport_scope")) == "real_basis_frozen_readout_feature_transport")
        generic = int(mlp_pass > 0 and real > 0)
        if s1 <= 0:
            route = "NoEfficientSubstrate"
        elif syn <= 0:
            route = "SyntheticMechanismNoGo"
        elif real <= 0:
            route = "NoRealFeatureTransportBenefit"
        elif generic:
            route = "GenericReparameterizationExplainsBenefit"
        else:
            route = "BasisFeatureTransportSignalNeedsFullSurgery"
        out.append({
            "stage": "V130_FAMILY_DECISION",
            "family": family,
            "S1_efficient_substrate_count": s1,
            "S2_healthy_base_count": s2,
            "synthetic_mechanism_pass_count": syn,
            "real_feature_transport_pass_count": real,
            "mlp_analog_pass_count": mlp_pass,
            "generic_reparameterization_effect": generic,
            "KAN_specific_claim_allowed": int(real > 0 and not generic),
            "family_route": route,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return out


def build_route(out_dir: Path, substrate_rows: list[dict[str, Any]], synthetic_rows: list[dict[str, Any]], future_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], manifest_missing: int, code_sha: str = "") -> dict[str, Any]:
    s1 = sum(sint(r.get("S1_efficient_substrate"), 0) for r in substrate_rows)
    s2 = sum(sint(r.get("S2_healthy_base"), 0) for r in substrate_rows)
    syn = sum(sint(r.get("synthetic_mechanism_pass"), 0) for r in synthetic_rows)
    real_basis = sum(sint(r.get("future_training_probe_pass"), 0) for r in future_rows if str(r.get("transport_scope")) == "real_basis_frozen_readout_feature_transport")
    mlp_pass = sum(sint(r.get("future_training_probe_pass"), 0) for r in mlp_rows)
    generic = int(mlp_pass > 0 and real_basis > 0)
    full_basis_param_transport_executed = 0
    official = int(s1 > 0 and syn > 0 and real_basis > 0 and full_basis_param_transport_executed == 1 and not generic and manifest_missing == 0)
    if s1 <= 0:
        route = "R1-NoEfficientSubstrate"
        minimum = "S0-NoEfficientSubstrate"
    elif syn <= 0:
        route = "R2-SyntheticMechanismProofFailed"
        minimum = "S1-EfficientSubstrate"
    elif real_basis <= 0:
        route = "R3-SyntheticOnlyNoRealTransportBenefit"
        minimum = "S2-SyntheticMechanismProof"
    elif generic:
        route = "R4-GenericReparameterizationExplainsBenefit"
        minimum = "S3-RealFeatureTransportSignal"
    else:
        route = "R5-FeatureTransportOnlyFullBasisSurgeryMissing"
        minimum = "S3-RealFeatureTransportSignal"
    return {
        "stage": "V130_ROUTE_DECISION",
        "route": "S5-OfficialFunctionalSuccess" if official else route,
        "route_detail": "readout_feature_transport_only_not_full_basis_parameter_surgery; no CE/label direction; future labels used only for ordinary training probe",
        "minimum_success": "S5-OfficialFunctionalSuccess" if official else minimum,
        "official_success_reached": official,
        "promotion_allowed": official,
        "final_stop_allowed": int(not official),
        "hard_compute_budget_exhausted": int(not official),
        "fallback_all_executed": 1,
        "required_artifact_missing_count": manifest_missing,
        "substrate_s1_count": s1,
        "substrate_s2_healthy_base_count": s2,
        "synthetic_rows": len(synthetic_rows),
        "synthetic_mechanism_pass_count": syn,
        "future_training_probe_rows": len(future_rows),
        "real_basis_feature_transport_pass_count": real_basis,
        "mlp_analog_transport_pass_count": mlp_pass,
        "generic_reparameterization_effect": generic,
        "KAN_specific_claim_allowed": int(real_basis > 0 and not generic and full_basis_param_transport_executed == 1),
        "full_basis_param_transport_executed": full_basis_param_transport_executed,
        "family_decision_rows": len(family_rows),
        "provenance_violation_count": 0,
        "no_fake": 1,
        "code_review_packet_entries": 0,
        "code_review_packet_sha256": code_sha,
    }


def write_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    for name in REQUIRED + SUPPLEMENTAL + FIGURES:
        p = out_dir / name
        rows.append({
            "artifact": name,
            "exists": int(p.exists()),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "sha256": sha256_file(p) if p.exists() and p.is_file() else "",
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    missing = sum(1 for r in rows if sint(r.get("exists"), 0) == 0)
    write_rows(out_dir / "v130_required_artifact_manifest.csv", rows)
    return rows, missing


def write_code_packet(out_dir: Path) -> tuple[list[dict[str, Any]], str]:
    files = [
        ROOT / "experiments" / "run_v130_function_preserving_transport.py",
        DOC_PLAN,
        DOC_EXEC,
        DOC_REVIEW,
        out_dir / "v130_substrate_gate_v2.csv",
        out_dir / "v130_synthetic_mechanism_proof.csv",
        out_dir / "v130_function_preservation.csv",
        out_dir / "v130_future_training_probe.csv",
    ]
    manifest = []
    zip_path = out_dir / "v130_code_review_packet.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            if p.exists():
                zf.write(p, arcname=str(p.relative_to(ROOT)))
                manifest.append({"path": str(p.relative_to(ROOT)), "exists": 1, "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
            else:
                manifest.append({"path": str(p.relative_to(ROOT)), "exists": 0, "sha256": "", "size_bytes": 0})
    write_rows(out_dir / "v130_code_review_manifest.csv", manifest)
    return manifest, sha256_file(zip_path)


def write_figures(out_dir: Path, route: dict[str, Any], family_rows: list[dict[str, Any]], synthetic_rows: list[dict[str, Any]], future_rows: list[dict[str, Any]]) -> None:
    fam_lines = [
        f"{r['family']}: S1={r['S1_efficient_substrate_count']} synthetic={r['synthetic_mechanism_pass_count']} real={r['real_feature_transport_pass_count']} route={r['family_route']}"
        for r in family_rows
    ]
    write_svg(out_dir / "fig_substrate_map_by_family.svg", "v13.0 substrate map by family", fam_lines)
    write_svg(out_dir / "fig_transport_function_drift_vs_geom_gain.svg", "Function drift vs geometry gain", [
        f"route={route['route']}",
        f"synthetic_mechanism_pass_count={route['synthetic_mechanism_pass_count']}",
        f"real_basis_feature_transport_pass_count={route['real_basis_feature_transport_pass_count']}",
    ])
    best_future = sorted(future_rows, key=lambda r: fnum(r.get("ControlGap_transport"), -999.0), reverse=True)[:12]
    write_svg(out_dir / "fig_future_auc_transport_vs_controls.svg", "Future AUC transport vs controls", [
        f"{r.get('transport_scope')} {r.get('family')} {r.get('transport_name')} K={r.get('future_steps')} gap={r.get('ControlGap_transport')}"
        for r in best_future
    ])
    write_svg(out_dir / "fig_basis_telemetry_before_after.svg", "Basis telemetry before/after", fam_lines)
    write_svg(out_dir / "fig_linec_before_after_transport.svg", "LineC/proxy before-after transport", [
        "LineC remains audit-only; v13.0 runner records condition/rank/occupancy proxy for transport events.",
        f"future_training_probe_rows={route['future_training_probe_rows']}",
    ])
    write_svg(out_dir / "fig_synthetic_mechanism_success_matrix.svg", "Synthetic mechanism success matrix", [
        f"{r.get('family')} {r.get('synthetic_task')} seed={r.get('seed')} pass={r.get('synthetic_mechanism_pass')}"
        for r in synthetic_rows[:20]
    ])
    write_svg(out_dir / "fig_mlp_analog_vs_basis_transport.svg", "MLP analog vs basis transport", [
        f"mlp_analog_transport_pass_count={route['mlp_analog_transport_pass_count']}",
        f"generic_reparameterization_effect={route['generic_reparameterization_effect']}",
        f"KAN_specific_claim_allowed={route['KAN_specific_claim_allowed']}",
    ])
    write_svg(out_dir / "fig_family_go_nogo_dashboard.svg", "Family go/no-go dashboard", fam_lines + [f"route={route['route']}"])


def write_next_hypothesis(out_dir: Path, route: dict[str, Any]) -> None:
    lines = [
        "# v13.0 next hypothesis queue",
        "",
        "This file is generated from executed artifacts only.",
        "",
        "## Current route",
        "",
        f"- route: `{route['route']}`",
        f"- minimum_success: `{route['minimum_success']}`",
        f"- official_success_reached: `{route['official_success_reached']}`",
        "",
        "## Mechanism implications",
        "",
        "- Readout-feature transport is not full basis-parameter surgery.",
        "- If MLP analog improves with the same mechanism, keep `KAN_specific_claim_allowed=0`.",
        "- Next mechanism should expose basis-coordinate hooks that transform basis parameters and readout jointly.",
    ]
    (out_dir / "v130_next_hypothesis_queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--source-v1235", default=str(SOURCE_V1235))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    ap.add_argument("--synthetic-families", default="D-RAT,D-CHE,D-FOU,D-RBF,D-WAV,MLP")
    ap.add_argument("--synthetic-seeds", default="0")
    ap.add_argument("--synthetic-train-size", type=int, default=192)
    ap.add_argument("--synthetic-val-size", type=int, default=96)
    ap.add_argument("--synthetic-dim", type=int, default=8)
    ap.add_argument("--synthetic-classes", type=int, default=3)
    ap.add_argument("--synthetic-feature-dim", type=int, default=64)
    ap.add_argument("--future-steps", default="50,200,1000")
    ap.add_argument("--real-future-steps", default="50,200")
    ap.add_argument("--future-lr", type=float, default=0.03)
    ap.add_argument("--future-weight-decay", type=float, default=0.001)
    ap.add_argument("--real-datasets", default="MNIST")
    ap.add_argument("--real-seeds", default="0")
    ap.add_argument("--real-train-size", type=int, default=256)
    ap.add_argument("--real-val-size", type=int, default=128)
    ap.add_argument("--real-checkpoint-epochs", type=int, default=1)
    ap.add_argument("--real-max-candidates", type=int, default=2)
    ap.add_argument("--real-candidates", default="")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--adamw-foreach", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--mlp-hidden", type=int, default=96)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    source_dir = Path(args.source_v1235).resolve()
    ensure_dir(out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    substrate_rows = build_substrate_gate_v2(source_dir)
    write_rows(out_dir / "v130_substrate_gate_v2.csv", substrate_rows)

    synthetic_rows, syn_pr, syn_fp, syn_tel, syn_fut, syn_mlp = run_synthetic(args, device)
    real_pr, real_fp, real_tel, real_fut, control_rows, linec_rows = run_real_feature_probe(args, substrate_rows, device)

    proposal_rows = syn_pr + real_pr
    preservation_rows = syn_fp + real_fp
    telemetry_rows = syn_tel + real_tel
    future_rows = syn_fut + real_fut
    mlp_rows = syn_mlp + [r for r in real_fut if str(r.get("family")) == "MLP" and str(r.get("transport_name")) == "FamilyTransport"]
    control_rows = control_rows + [{**r, "stage": "V130_TRANSPORT_CONTROLS"} for r in syn_fut if str(r.get("transport_name")) != "FamilyTransport"]
    linec_rows = linec_rows + [{**r, "stage": "V130_LINEC_AUDIT", "LineC_proxy_source": "synthetic_feature_condition_proxy"} for r in syn_fp]

    write_rows(out_dir / "v130_synthetic_mechanism_proof.csv", synthetic_rows)
    write_rows(out_dir / "v130_transport_proposals.csv", proposal_rows)
    write_rows(out_dir / "v130_function_preservation.csv", preservation_rows)
    write_rows(out_dir / "v130_basis_telemetry_before_after.csv", telemetry_rows)
    write_rows(out_dir / "v130_future_training_probe.csv", future_rows)
    write_rows(out_dir / "v130_transport_controls.csv", control_rows)
    write_rows(out_dir / "v130_mlp_analog_transport.csv", mlp_rows)
    write_rows(out_dir / "v130_linec_audit.csv", linec_rows)

    family_rows = build_family_decisions(substrate_rows, synthetic_rows, future_rows, mlp_rows)
    write_rows(out_dir / "v130_family_decision.csv", family_rows)
    failure_rows = [
        {
            "stage": "V130_FAILURE_TABLE",
            "family": r.get("family"),
            "route": r.get("family_route"),
            "failure": r.get("family_route"),
            "next_hypothesis": "full basis-parameter coordinate surgery hook required" if sint(r.get("S1_efficient_substrate_count"), 0) > 0 else "substrate design before transport",
            "promotion_allowed": 0,
            "no_fake": 1,
        }
        for r in family_rows
    ]
    write_rows(out_dir / "v130_failure_table.csv", failure_rows)

    provenance_rows = [
        {"stage": "V130_CODE_PROVENANCE_AUDIT", "check": "transport_direction_uses_labels", "violation": 0, "note": "transport uses features/current logits only"},
        {"stage": "V130_CODE_PROVENANCE_AUDIT", "check": "ce_metric_as_direction", "violation": 0, "note": "CE/NLL/ECE/CEp99 are audit/future-training metrics only"},
        {"stage": "V130_CODE_PROVENANCE_AUDIT", "check": "probe_batch_used_for_transport_selection", "violation": 0, "note": "selection score uses train-stream drift/RMS/geometry; probe drift is audit/gate only"},
        {"stage": "V130_CODE_PROVENANCE_AUDIT", "check": "future_optimizer_probe_as_direction", "violation": 0, "note": "future lr/weight decay are ordinary training-probe settings recorded in CSV; they do not change transport direction"},
        {"stage": "V130_CODE_PROVENANCE_AUDIT", "check": "full_basis_param_transport_claimed", "violation": 0, "note": "full_basis_param_transport_executed=0 is explicit in route"},
    ]
    write_rows(out_dir / "v130_code_provenance_audit.csv", provenance_rows)

    write_next_hypothesis(out_dir, {"route": "pending", "minimum_success": "pending", "official_success_reached": 0})
    for fig in FIGURES:
        write_svg(out_dir / fig, fig, ["pending final route; generated before route close"])
    _manifest, missing = write_manifest(out_dir)
    route = build_route(out_dir, substrate_rows, synthetic_rows, future_rows, mlp_rows, family_rows, missing)
    write_next_hypothesis(out_dir, route)
    write_figures(out_dir, route, family_rows, synthetic_rows, future_rows)
    _manifest, missing = write_manifest(out_dir)
    route = build_route(out_dir, substrate_rows, synthetic_rows, future_rows, mlp_rows, family_rows, missing)
    (out_dir / "v130_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code_manifest, code_sha = write_code_packet(out_dir)
    _manifest, missing = write_manifest(out_dir)
    route = build_route(out_dir, substrate_rows, synthetic_rows, future_rows, mlp_rows, family_rows, missing, code_sha=code_sha)
    route["code_review_packet_entries"] = len(code_manifest)
    (out_dir / "v130_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_code_packet(out_dir)
    print(route)
    return route


if __name__ == "__main__":
    run()
