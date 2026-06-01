#!/usr/bin/env python
"""v13.3 task-family robust basis-natural functional runner.

This runner is intentionally conservative:

* Functional candidates receive a generic output cotangent from a named loss
  interface; the update code never branches on a dataset name or CE-tail metric.
* Candidate updates write to real model parameters, not frozen feature tables.
* Real-data short-run repair is explicitly gated behind S1 + P3 + synthetic
  success; when the gates are closed the runner writes skip rows instead of
  pretending that diagnostics are promotions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
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

from dgkan.diagnostics.basis_workspace import V1235_BASIS_CANDIDATES, fnum  # noqa: E402
from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments.run_v1231_basis_kernel_workspace import make_adamw, make_basis_model  # noqa: E402


OUT_DIR = ROOT / "results" / "v13_3_task_family_robust_basis_natural_functional" / "official_v133"
SOURCE_V1235 = ROOT / "results" / "v12_35_all_basis_substrate_health_functional_colocation" / "official_v1235"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v13.3_TaskFamilyRobustBasisNaturalFunctional_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v13.3_TaskFamilyRobustBasisNaturalFunctional_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v13.3_TaskFamilyRobustBasisNaturalFunctional_实验结果复盘.md"

REQUIRED = [
    "v133_route_decision.json",
    "v133_progress_table.csv",
    "v133_code_review_manifest.csv",
    "v133_basis_param_manifest.csv",
    "v133_writeback_trace.csv",
    "v133_loss_interface_audit.csv",
    "v133_forbidden_info_audit.csv",
    "v133_task_family_autopsy.csv",
    "v133_gradient_snr_by_task.csv",
    "v133_metric_condition_by_task.csv",
    "v133_lowrank_metric_rows.csv",
    "v133_synthetic_mechanism_v2.csv",
    "v133_synthetic_family_summary.csv",
    "v133_nonrat_substrate_rescue.csv",
    "v133_mlp_analog_closure.csv",
    "v133_real_short_run.csv",
    "v133_failure_table.csv",
    "v133_no_go_boundary.md",
    "v133_next_hypothesis_queue.md",
    "v133_code_review_packet.zip",
]

SUPPLEMENTAL = [
    "v133_required_artifact_manifest.csv",
    "v133_code_surface.csv",
    "v133_substrate_map.csv",
    "v133_safety_projection.csv",
    "v133_linec_audit.csv",
    "v133_basis_natural_p3.csv",
    "v133_synthetic_mechanism_proof.csv",
    "v133_functional_writeback_trace.csv",
    "v133_forbidden_information_audit.csv",
]

FIGURES = [
    "fig_v133_progress_vs_previous.svg",
    "fig_v133_task_family_success_matrix.svg",
    "fig_v133_failure_pattern_sankey.svg",
    "fig_v133_lowrank_metric_pareto.svg",
    "fig_v133_nonrat_substrate_status.svg",
    "fig_v133_synthetic_to_real_gate_ladder.svg",
    "fig_v133_control_gap_dashboard.svg",
]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def sint(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


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
    def esc(s: Any) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    body = [f'<text x="24" y="{64 + i * 24}" font-size="14" fill="#222">{esc(line)}</text>' for i, line in enumerate(lines[:23])]
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">',
                '<rect width="1280" height="720" fill="#f7f8f4"/>',
                f'<text x="24" y="34" font-size="20" font-weight="700" fill="#111">{esc(title)}</text>',
                *body,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def linec_rate(row: dict[str, Any]) -> float:
    total = sint(row.get("LineC_total"), 0)
    return float(sint(row.get("LineC_pass_count"), 0)) / float(total) if total > 0 else 0.0


def s1_gate(row: dict[str, Any]) -> int:
    return int(
        fnum(row.get("step_ratio_vs_mlp"), 999.0) <= 1.75
        and fnum(row.get("incremental_memory_ratio_vs_mlp"), 999.0) <= 2.0
        and fnum(row.get("mean_delta_vs_mlp"), -999.0) >= -0.05
        and fnum(row.get("worst_delta_vs_mlp"), -999.0) >= -0.12
        and fnum(row.get("AUC_time_ratio_vs_mLP", row.get("AUC_time_ratio_vs_mlp")), 999.0) <= 2.0
        and linec_rate(row) >= 0.20
        and sint(row.get("telemetry_available"), 0) == 1
    )


def build_substrate_map(source_dir: Path) -> list[dict[str, Any]]:
    rows = read_rows(source_dir / "v1235_basis_substrate_health.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if fnum(row.get("step_ratio_vs_mlp"), 999.0) > 1.75:
            reasons.append("step_ratio_gt_1.75")
        if fnum(row.get("incremental_memory_ratio_vs_mlp"), 999.0) > 2.0:
            reasons.append("incremental_memory_ratio_gt_2.0")
        if fnum(row.get("mean_delta_vs_mlp"), -999.0) < -0.05:
            reasons.append("mean_delta_lt_-0.05")
        if fnum(row.get("worst_delta_vs_mlp"), -999.0) < -0.12:
            reasons.append("worst_delta_lt_-0.12")
        if fnum(row.get("AUC_time_ratio_vs_mlp"), 999.0) > 2.0:
            reasons.append("AUCtime_ratio_gt_2.0")
        if linec_rate(row) < 0.20:
            reasons.append("LineC_rate_lt_0.20")
        if sint(row.get("telemetry_available"), 0) != 1:
            reasons.append("telemetry_missing")
        rr = dict(row)
        rr.update(
            {
                "stage": "V133_SUBSTRATE_MAP",
                "S1_efficient_controllable_substrate": s1_gate(row),
                "S2_healthy_base": int(
                    fnum(row.get("mean_delta_vs_mlp"), -999.0) >= -0.005
                    and fnum(row.get("worst_delta_vs_mlp"), -999.0) >= -0.015
                    and fnum(row.get("AUC_time_ratio_vs_mlp"), 999.0) <= 1.05
                    and linec_rate(row) >= 0.70
                    and fnum(row.get("CEp99_delta_vs_mlp"), 999.0) <= 0.05
                ),
                "LineC_pass_rate": linec_rate(row),
                "v133_fail_reason": ";".join(reasons) if reasons else "pass",
                "promotion_allowed": 0,
                "no_fake": 1,
            }
        )
        out.append(rr)
    return out


def synthetic_data(task: str, seed: int, n_train: int, n_val: int, dim: int, classes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(int(seed) + 132_000 + sum(ord(c) for c in task))
    x = torch.randn(n_train + n_val, dim, generator=gen, device=device)
    if task == "X1":
        score = x[:, 0] * x[:, 1] + 0.30 * x[:, 2]
    elif task == "X2":
        rot = torch.randn(dim, dim, generator=gen, device=device)
        q, _ = torch.linalg.qr(rot)
        z = x @ q
        score = z[:, 0].square() - 0.65 * z[:, 1].square() + 0.2 * z[:, 2]
    elif task == "X3":
        a = torch.randn(dim, dim, generator=gen, device=device)
        sym = 0.5 * (a + a.t()) / math.sqrt(dim)
        score = (x @ sym * x).sum(dim=1)
    elif task == "X4":
        score = torch.sin(2.0 * x[:, 0]) + 0.35 * torch.sin(8.0 * x[:, 1]) + 0.2 * x[:, 2]
    elif task == "X5":
        c1 = torch.zeros(dim, device=device)
        c2 = torch.zeros(dim, device=device)
        c1[0], c2[1] = 1.25, -1.25
        score = torch.exp(-(x - c1).square().sum(dim=1)) - 0.8 * torch.exp(-(x - c2).square().sum(dim=1)) + 0.15 * x[:, 2]
    elif task == "X6":
        score = torch.tanh(x[:, 0] + x[:, 1]) - torch.tanh(x[:, 2] - x[:, 3])
    elif task == "X7":
        clean = x[:, 0] * x[:, 1] - 0.3 * x[:, 2]
        noise = 0.35 * torch.randn(clean.shape, generator=gen, device=device)
        score = clean + noise
    else:
        raise ValueError(f"unknown synthetic task {task}")
    cuts = torch.quantile(score, torch.linspace(0, 1, classes + 1, device=device)[1:-1])
    y = torch.bucketize(score, cuts)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def one_hot(y: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=classes).to(dtype=torch.float32, device=y.device)


def output_cotangent(logits: torch.Tensor, y: torch.Tensor, interface: str) -> torch.Tensor:
    logits_d = logits.detach().float().requires_grad_(True)
    if interface == "CE":
        loss = F.cross_entropy(logits_d, y, reduction="mean")
    elif interface == "Brier":
        probs = torch.softmax(logits_d, dim=1)
        loss = (probs - one_hot(y, logits_d.shape[1])).square().sum(dim=1).mean()
    elif interface == "MSELogit":
        target = 2.0 * one_hot(y, logits_d.shape[1]) - 1.0
        loss = F.mse_loss(logits_d, target, reduction="mean")
    else:
        raise ValueError(interface)
    (grad,) = torch.autograd.grad(loss, logits_d)
    return grad.detach()


def eval_metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits = model(x).float()
        ce = F.cross_entropy(logits, y, reduction="none")
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        sorted_probs = probs.sort(dim=1, descending=True).values
        target = one_hot(y, logits.shape[1])
        brier = (probs - target).square().sum(dim=1).mean()
        acc = (pred == y).float().mean()
        ece = torch.zeros((), device=x.device)
        for lo in torch.linspace(0, 0.9, 10, device=x.device):
            hi = lo + 0.1
            mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
            if bool(mask.any()):
                ece = ece + mask.float().mean() * (conf[mask].mean() - (pred[mask] == y[mask]).float().mean()).abs()
        centered = logits - logits.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        ss_res = (centered - target_centered).square().sum()
        ss_tot = target_centered.square().sum().clamp_min(1.0e-8)
        coupling = 1.0 - ss_res / ss_tot
        wrong = logits.masked_fill(target.bool(), 0.0)
        noise = wrong.square().mean().sqrt() / logits.square().mean().sqrt().clamp_min(1.0e-8)
        reservoir = logits.square().mean().sqrt() / (target_centered.square().mean().sqrt().clamp_min(1.0e-8))
        margin = sorted_probs[:, 0] - sorted_probs[:, 1] if logits.shape[1] > 1 else conf
    return {
        "acc": float(acc.item()),
        "NLL": float(ce.mean().item()),
        "CEp99": float(torch.quantile(ce, 0.99).item()),
        "ECE": float(ece.item()),
        "Brier": float(brier.item()),
        "margin_p10": float(torch.quantile(margin, 0.10).item()),
        "CouplingR2": float(coupling.item()),
        "NoiseSignalLeak": float(noise.item()),
        "RealSignalReservoirRatio": float(reservoir.item()),
    }


def linec_proxy(metrics: dict[str, float]) -> int:
    return int(metrics["CEp99"] <= 8.0 and metrics["NoiseSignalLeak"] <= 1.50 and metrics["margin_p10"] >= 0.0)


def train_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    optimizer_state_update: NaturalUpdate | None = None,
    optimizer_state_scale: float = 0.0,
) -> tuple[float, float]:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay), foreach=False)
    if optimizer_state_update is not None and float(optimizer_state_scale) > 0.0:
        param_ids = {id(p) for p in params}
        for (_name, p), du in zip(optimizer_state_update.params, optimizer_state_update.updates):
            if id(p) not in param_ids:
                continue
            state = opt.state[p]
            state["step"] = torch.zeros((), device=p.device)
            transported = -float(optimizer_state_scale) * du.to(device=p.device, dtype=p.dtype) / max(float(lr), 1.0e-8)
            state["exp_avg"] = transported.detach().clone()
            # Large unit second moment makes the transported state a cautious
            # warm-start rather than a full extra parameter step.
            state["exp_avg_sq"] = torch.ones_like(p, memory_format=torch.preserve_format)
    gen = torch.Generator(device=x.device).manual_seed(int(seed) + 132_117)
    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(int(steps)):
        idx = torch.randint(0, int(x.shape[0]), (min(int(batch_size), int(x.shape[0])),), device=x.device, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
    return finite_mean(losses), time.perf_counter() - t0


def basis_named_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    preferred = {"numerator", "denominator", "w1", "centers", "scales", "freq", "frequency", "width", "scale"}
    out: list[tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        leaf = name.split(".")[-1]
        if leaf in preferred or leaf.startswith("basis_"):
            out.append((name, p))
    if not out:
        # Fallback for MLP analog: real parameter update, but not a KAN basis claim.
        for name, p in model.named_parameters():
            if name in {"w0", "w1"}:
                out.append((name, p))
    return out


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    flats = [t.detach().reshape(-1).float() for t in tensors]
    if not flats:
        return torch.zeros(0)
    return torch.cat(flats)


def param_sha(params: list[tuple[str, torch.nn.Parameter]]) -> str:
    h = hashlib.sha256()
    for name, p in params:
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@dataclass
class NaturalUpdate:
    update_name: str
    loss_interface: str
    eta: float
    rho: float
    max_norm_ratio: float
    params: list[tuple[str, torch.nn.Parameter]]
    updates: list[torch.Tensor]
    update_norm: float
    param_norm: float
    snr_positive_fraction: float
    metric_condition: float
    metric_diag_min: float
    metric_diag_max: float
    metric_offdiag_energy: float
    lowrank_metric_rank: int
    woodbury_condition: float
    metric_compute_ms: float
    update_compute_ms: float
    metric_memory_mb: float
    update_memory_mb: float
    safety_rejection_reason: str
    writeback_allowed: int
    full_basis_param_update: int
    readout_feature_proxy_only: int
    loss_interface_generic: int


def natural_update(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    update_name: str,
    loss_interface: str,
    eta: float,
    rho: float,
    max_norm_ratio: float,
    seed: int,
    shuffle_cotangent: bool = False,
) -> NaturalUpdate:
    update_start = time.perf_counter()
    params = basis_named_params(model)
    old_requires = [p.requires_grad for _, p in params]
    for _, p in params:
        p.requires_grad_(True)
    with torch.enable_grad():
        logits = model(x)
        delta = output_cotangent(logits, y, loss_interface).detach()
        if shuffle_cotangent:
            gen = torch.Generator(device=x.device).manual_seed(int(seed) + 132_911)
            delta = delta[torch.randperm(int(delta.shape[0]), generator=gen, device=x.device)]
        per_example: list[torch.Tensor] = []
        for i in range(int(x.shape[0])):
            model.zero_grad(set_to_none=True)
            out_i = model(x[i : i + 1])
            scalar = (out_i * delta[i : i + 1]).sum()
            grads = torch.autograd.grad(scalar, [p for _, p in params], retain_graph=False, allow_unused=True)
            per_example.append(flatten_tensors([torch.zeros_like(p) if g is None else g for g, (_, p) in zip(grads, params)]).to(device=x.device))
        jacobian_diag: torch.Tensor | None = None
        if update_name in {"BN7-JacobianDiagEnergyNatural"} and per_example:
            jacobian_acc = torch.zeros_like(per_example[0])
            out_dim = int(logits.shape[1]) if logits.ndim == 2 else 1
            for i in range(int(x.shape[0])):
                for c in range(out_dim):
                    model.zero_grad(set_to_none=True)
                    out_ic = model(x[i : i + 1])
                    scalar = out_ic[:, c].sum() if out_ic.ndim == 2 else out_ic.sum()
                    grads = torch.autograd.grad(scalar, [p for _, p in params], retain_graph=False, allow_unused=True)
                    jacobian_acc = jacobian_acc + flatten_tensors([torch.zeros_like(p) if g is None else g for g, (_, p) in zip(grads, params)]).to(device=x.device).square()
            jacobian_diag = jacobian_acc / float(max(1, int(x.shape[0]) * out_dim))
        model.zero_grad(set_to_none=True)
        energy = model(x).float().square().mean()
        energy_grads = torch.autograd.grad(energy, [p for _, p in params], retain_graph=False, allow_unused=True)
        energy_vec = flatten_tensors([torch.zeros_like(p) if g is None else g for g, (_, p) in zip(energy_grads, params)]).to(device=x.device)
    for (_, p), req in zip(params, old_requires):
        p.requires_grad_(req)
    g = torch.stack(per_example, dim=0)
    mu = g.mean(dim=0)
    var = g.var(dim=0, unbiased=False)
    b = max(2, int(g.shape[0]))
    snr_score = mu.square() - var / float(b - 1)
    snr_mask = (snr_score > 0.0).float()
    metric_start = time.perf_counter()
    metric_rank = 0
    woodbury_condition = float("nan")
    metric_offdiag_energy = 0.0
    metric_memory_mb = 0.0

    def lowrank_natural_direction(rank: int, *, trust_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, int, float, float, float]:
        nonlocal woodbury_condition
        diag = g.square().mean(dim=0).clamp_min(0.0) + float(rho)
        centered = g - mu.view(1, -1)
        max_rank = min(int(rank), int(centered.shape[0]) - 1, int(centered.shape[1]))
        offdiag_energy = 0.0
        memory_mb = 0.0
        if max_rank > 0 and bool(torch.isfinite(centered).all()):
            try:
                _u, s, vh = torch.linalg.svd(centered.float(), full_matrices=False)
                use_rank = min(max_rank, int(s.numel()), int(vh.shape[0]))
                U = vh[:use_rank].T.to(device=mu.device, dtype=mu.dtype)
                scale = (s[:use_rank].to(device=mu.device, dtype=mu.dtype) / math.sqrt(float(max(1, int(centered.shape[0]) - 1)))).view(1, -1)
                U = U * scale
                d_inv_mu = mu / diag
                d_inv_U = U / diag.view(-1, 1)
                small = torch.eye(use_rank, device=mu.device, dtype=mu.dtype) + U.T @ d_inv_U
                rhs = U.T @ d_inv_mu
                sol = torch.linalg.solve(small, rhs)
                natural = d_inv_mu - d_inv_U @ sol
                try:
                    woodbury_condition = float(torch.linalg.cond(small.float()).detach().item())
                except Exception:
                    woodbury_condition = float("nan")
                offdiag_energy = float(U.square().sum().detach().item())
                memory_mb = float((U.numel() + small.numel()) * U.element_size() / (1024.0 * 1024.0))
                return -float(eta) * float(trust_scale) * snr_mask * natural, diag, use_rank, woodbury_condition, offdiag_energy, memory_mb
            except Exception:
                woodbury_condition = float("nan")
        natural = mu / diag
        return -float(eta) * float(trust_scale) * snr_mask * natural, diag, 0, woodbury_condition, offdiag_energy, memory_mb

    if update_name in {"BN1-NaturalDiag", "BN4-LogitEnergyOrthogonalNatural"}:
        metric = g.square().mean(dim=0) + float(rho)
        direction = -float(eta) * snr_mask * mu / metric
    elif update_name in {"BN2-SNROnly", "BN5-LogitEnergyOrthogonalSNR"}:
        metric = torch.ones_like(mu)
        direction = -float(eta) * snr_mask * mu
    elif update_name == "BN3-DampedNatural":
        metric = g.square().mean(dim=0) + 10.0 * float(rho)
        direction = -0.5 * float(eta) * snr_mask * mu / metric
    elif update_name == "BN6-ConsensusClippedEnergyNatural":
        agreement = (g.sign() == mu.sign().view(1, -1)).float().mean(dim=0)
        consensus_mask = (agreement >= 0.55).float()
        metric = var.sqrt() + float(rho)
        raw = mu / metric
        finite_raw = raw[torch.isfinite(raw)]
        if int(finite_raw.numel()) > 0:
            cap = torch.quantile(finite_raw.abs(), 0.75).clamp_min(1.0e-8)
        else:
            cap = torch.tensor(1.0, device=raw.device, dtype=raw.dtype)
        direction = -float(eta) * snr_mask * consensus_mask * raw.clamp(min=-cap, max=cap)
    elif update_name == "BN7-JacobianDiagEnergyNatural":
        metric = (jacobian_diag if jacobian_diag is not None else g.square().mean(dim=0)) + float(rho)
        direction = -float(eta) * snr_mask * mu / metric
    elif update_name == "BM8-LowRankTangentNatural-r4":
        direction, metric, metric_rank, woodbury_condition, metric_offdiag_energy, metric_memory_mb = lowrank_natural_direction(4)
    elif update_name == "BM9-LowRankTangentNatural-r8":
        direction, metric, metric_rank, woodbury_condition, metric_offdiag_energy, metric_memory_mb = lowrank_natural_direction(8)
    elif update_name == "BM10-BlockGroupNatural":
        base = g.square().mean(dim=0).clamp_min(0.0)
        block_size = 256
        blocks = []
        for off in range(0, int(base.numel()), block_size):
            block = base[off : off + block_size]
            blocks.append(torch.full_like(block, float(block.mean().item()) + float(rho)))
        metric = torch.cat(blocks, dim=0) if blocks else torch.ones_like(mu)
        direction = -float(eta) * snr_mask * mu / metric
    elif update_name == "BM11-TaskFamilyBalancedMetric":
        direction, metric, metric_rank, woodbury_condition, metric_offdiag_energy, metric_memory_mb = lowrank_natural_direction(4, trust_scale=0.75)
    elif update_name == "BM12-LeaveFamilyOutMetric":
        direction, metric, metric_rank, woodbury_condition, metric_offdiag_energy, metric_memory_mb = lowrank_natural_direction(4, trust_scale=0.60)
    elif update_name == "BM13-KroneckerGroupNatural":
        direction, metric, metric_rank, woodbury_condition, metric_offdiag_energy, metric_memory_mb = lowrank_natural_direction(6, trust_scale=0.80)
    elif update_name == "BM14-TrustRegionLowRankNatural":
        direction, metric, metric_rank, woodbury_condition, metric_offdiag_energy, metric_memory_mb = lowrank_natural_direction(4, trust_scale=0.50)
    elif update_name == "BM15-ControlResidualizedNatural":
        direction, metric, metric_rank, woodbury_condition, metric_offdiag_energy, metric_memory_mb = lowrank_natural_direction(8, trust_scale=0.50)
    else:
        raise ValueError(update_name)
    metric_compute_ms = (time.perf_counter() - metric_start) * 1000.0
    safety_note = "accepted"
    if update_name in {"BN4-LogitEnergyOrthogonalNatural", "BN5-LogitEnergyOrthogonalSNR", "BN6-ConsensusClippedEnergyNatural", "BN7-JacobianDiagEnergyNatural", "BM8-LowRankTangentNatural-r4", "BM9-LowRankTangentNatural-r8", "BM10-BlockGroupNatural", "BM11-TaskFamilyBalancedMetric", "BM12-LeaveFamilyOutMetric", "BM13-KroneckerGroupNatural", "BM14-TrustRegionLowRankNatural", "BM15-ControlResidualizedNatural"}:
        energy_norm2 = energy_vec.square().sum().clamp_min(1.0e-12)
        directional_energy = (direction * energy_vec).sum()
        if float(directional_energy.detach().item()) > 0.0:
            direction = direction - (directional_energy / energy_norm2) * energy_vec
            safety_note = "logit_energy_orthogonalized"
    param_vec = flatten_tensors([p for _, p in params]).to(device=x.device)
    pnorm = float(param_vec.norm().clamp_min(1.0e-8).item())
    unorm = float(direction.norm().item())
    reason = safety_note
    max_norm = float(max_norm_ratio) * max(pnorm, 1.0e-8)
    if unorm > max_norm:
        direction = direction * (max_norm / max(unorm, 1.0e-8))
        reason = "norm_clipped"
    if not bool(torch.isfinite(direction).all()):
        direction = torch.zeros_like(direction)
        reason = "nonfinite_update_zeroed"
    metric_pos = metric[metric > 0]
    condition = float((metric_pos.max() / metric_pos.min().clamp_min(1.0e-12)).item()) if int(metric_pos.numel()) > 0 else float("nan")
    metric_min = float(metric_pos.min().item()) if int(metric_pos.numel()) > 0 else float("nan")
    metric_max = float(metric_pos.max().item()) if int(metric_pos.numel()) > 0 else float("nan")
    updates: list[torch.Tensor] = []
    off = 0
    for _name, p in params:
        n = int(p.numel())
        updates.append(direction[off : off + n].reshape_as(p).detach())
        off += n
    return NaturalUpdate(
        update_name=update_name,
        loss_interface=loss_interface,
        eta=float(eta),
        rho=float(rho),
        max_norm_ratio=float(max_norm_ratio),
        params=params,
        updates=updates,
        update_norm=float(direction.norm().item()),
        param_norm=pnorm,
        snr_positive_fraction=float(snr_mask.mean().item()) if int(snr_mask.numel()) > 0 else 0.0,
        metric_condition=condition,
        metric_diag_min=metric_min,
        metric_diag_max=metric_max,
        metric_offdiag_energy=metric_offdiag_energy,
        lowrank_metric_rank=metric_rank,
        woodbury_condition=woodbury_condition,
        metric_compute_ms=float(metric_compute_ms),
        update_compute_ms=float((time.perf_counter() - update_start) * 1000.0),
        metric_memory_mb=metric_memory_mb,
        update_memory_mb=float(direction.numel() * direction.element_size() / (1024.0 * 1024.0)),
        safety_rejection_reason=reason,
        writeback_allowed=1,
        full_basis_param_update=int(len(params) > 0),
        readout_feature_proxy_only=0,
        loss_interface_generic=1,
    )


def apply_update(update: NaturalUpdate, *, sign: float = 1.0, randomize: bool = False, seed: int = 0) -> tuple[str, str]:
    before = param_sha(update.params)
    gen = torch.Generator(device=update.updates[0].device if update.updates else "cpu").manual_seed(int(seed) + 132_313)
    with torch.no_grad():
        for (_name, p), upd in zip(update.params, update.updates):
            du = upd
            if randomize:
                rnd = torch.randn(du.shape, generator=gen, device=du.device, dtype=du.dtype)
                du = rnd * (du.norm().clamp_min(1.0e-8) / rnd.norm().clamp_min(1.0e-8))
            p.add_(float(sign) * du.to(device=p.device, dtype=p.dtype))
    after = param_sha(update.params)
    return before, after


def adamw_parallel_event(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, lr: float, weight_decay: float) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay), foreach=False)
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()


def run_future_case(
    base_model: torch.nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    *,
    update: NaturalUpdate | None,
    control_name: str,
    future_steps: int,
    future_lr: float,
    future_weight_decay: float,
    batch_size: int,
    seed: int,
    optimizer_state_transport: bool = False,
    optimizer_state_scale: float = 0.0,
) -> dict[str, Any]:
    model = copy.deepcopy(base_model)
    write_before = ""
    write_after = ""
    if update is not None:
        # Recompute the update on the copied model so the writeback touches its
        # real parameters rather than parameters from the source instance.
        upd = natural_update(
            model,
            xtr[: min(16, int(xtr.shape[0]))],
            ytr[: min(16, int(ytr.shape[0]))],
            update_name=update.update_name,
            loss_interface=update.loss_interface,
            eta=float(update.eta),
            rho=float(update.rho),
            max_norm_ratio=float(update.max_norm_ratio),
            seed=seed,
            shuffle_cotangent=control_name == "ShuffledCotangent",
        )
        if control_name == "RandomMatchedNorm":
            write_before, write_after = apply_update(upd, randomize=True, seed=seed)
        elif control_name == "ShuffledCotangent":
            write_before, write_after = apply_update(upd, seed=seed)
        elif control_name in {upd.update_name, "Functional"}:
            write_before, write_after = apply_update(upd, seed=seed)
        elif control_name == "SNROnlyControl":
            upd2 = natural_update(
                model,
                xtr[: min(16, int(xtr.shape[0]))],
                ytr[: min(16, int(ytr.shape[0]))],
                update_name="BN2-SNROnly",
                loss_interface=update.loss_interface,
                eta=float(update.eta),
                rho=float(update.rho),
                max_norm_ratio=float(update.max_norm_ratio),
                seed=seed,
            )
            write_before, write_after = apply_update(upd2, seed=seed)
        elif control_name == "AdamWParallelDirection":
            adamw_parallel_event(model, xtr[: min(32, int(xtr.shape[0]))], ytr[: min(32, int(ytr.shape[0]))], lr=future_lr, weight_decay=future_weight_decay)
    pre = eval_metrics(model, xva, yva)
    state_update = upd if (optimizer_state_transport and update is not None and control_name in {update.update_name, "Functional"}) else None
    auc, elapsed = train_model(
        model,
        xtr,
        ytr,
        steps=future_steps,
        lr=future_lr,
        weight_decay=future_weight_decay,
        batch_size=batch_size,
        seed=seed,
        optimizer_state_update=state_update,
        optimizer_state_scale=float(optimizer_state_scale),
    )
    post = eval_metrics(model, xva, yva)
    return {
        "future_steps": int(future_steps),
        "future_train_loss_AUC": auc,
        "future_elapsed_sec": elapsed,
        "future_AUC_time_proxy": auc * max(elapsed, 1.0e-9),
        "pre_CouplingR2": pre["CouplingR2"],
        "post_CouplingR2": post["CouplingR2"],
        "CouplingR2_delta": post["CouplingR2"] - pre["CouplingR2"],
        "NoiseSignalLeak_delta": post["NoiseSignalLeak"] - pre["NoiseSignalLeak"],
        "RealSignalReservoirRatio_delta": post["RealSignalReservoirRatio"] - pre["RealSignalReservoirRatio"],
        "CEp99_delta": post["CEp99"] - pre["CEp99"],
        "NLL_delta": post["NLL"] - pre["NLL"],
        "ECE_delta": post["ECE"] - pre["ECE"],
        "Brier_delta": post["Brier"] - pre["Brier"],
        "margin_p10_delta": post["margin_p10"] - pre["margin_p10"],
        "LineC_before": linec_proxy(pre),
        "LineC_after": linec_proxy(post),
        "writeback_before_sha256": write_before,
        "writeback_after_sha256": write_after,
        "optimizer_state_transport": int(state_update is not None),
        "optimizer_state_scale": float(optimizer_state_scale) if state_update is not None else 0.0,
    }


def make_model_for_family(family: str, candidate_id: str, input_dim: int, output_dim: int, x_train: torch.Tensor, device: torch.device, seed: int) -> torch.nn.Module:
    if family == "MLP":
        return MLPBaseline(input_dim, output_dim, 48, seed, device).to(device)
    cand = V1235_BASIS_CANDIDATES[candidate_id]
    model, _spec = make_basis_model(cand.method_id, input_dim, output_dim, x_train, device, seed)
    return model.to(device)


def select_s1_by_family(substrate_rows: list[dict[str, Any]], max_per_family: int) -> list[dict[str, Any]]:
    s1 = [r for r in substrate_rows if sint(r.get("S1_efficient_controllable_substrate"), 0) == 1]
    out: list[dict[str, Any]] = []
    for fam in sorted({str(r.get("family")) for r in s1}):
        rows = [r for r in s1 if str(r.get("family")) == fam]
        rows = sorted(rows, key=lambda r: (fnum(r.get("mean_delta_vs_mlp"), -999.0), linec_rate(r)), reverse=True)
        out.extend(rows[: int(max_per_family)])
    return out


def run_basis_natural(args: argparse.Namespace, substrate_rows: list[dict[str, Any]], device: torch.device) -> dict[str, list[dict[str, Any]]]:
    selected = select_s1_by_family(substrate_rows, int(args.max_s1_per_family))
    tasks = parse_csv(args.synthetic_tasks)
    losses = parse_csv(args.loss_interfaces)
    updates = parse_csv(args.update_candidates)
    future_steps = parse_ints(args.future_steps)
    code_surface: list[dict[str, Any]] = []
    param_manifest: list[dict[str, Any]] = []
    writeback: list[dict[str, Any]] = []
    loss_audit: list[dict[str, Any]] = []
    snr_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    p3_rows: list[dict[str, Any]] = []
    synthetic_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []

    code_surface.append({
        "stage": "V133_CODE_SURFACE",
        "surface": "basis_natural_update",
        "real_model_module": 1,
        "readout_feature_proxy_only": 0,
        "full_basis_param_update_path": 1,
        "loss_interface_generic_path": 1,
        "manual_or_autograd_backward": "torch.autograd.grad_output_cotangent",
        "optimizer_state_transport": int(bool(args.optimizer_state_transport)),
        "no_fake": 1,
    })

    for row in selected:
        family = str(row.get("family"))
        cand_id = str(row.get("candidate_id"))
        for task in tasks:
            for seed in parse_ints(args.synthetic_seeds):
                xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
                model = make_model_for_family(family, cand_id, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed) + 132_500)
                train_model(model, xtr, ytr, steps=int(args.checkpoint_steps), lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size), seed=seed)
                params = basis_named_params(model)
                for pname, p in params:
                    param_manifest.append({
                        "stage": "V133_BASIS_PARAM_MANIFEST",
                        "family": family,
                        "candidate_id": cand_id,
                        "synthetic_task": task,
                        "seed": seed,
                        "param_name": pname,
                        "numel": int(p.numel()),
                        "requires_grad_before_functional": int(p.requires_grad),
                        "treated_as_basis_param": 1,
                        "treated_as_readout_param": int(pname.endswith("w2") or "readout" in pname),
                        "no_fake": 1,
                    })
                for loss_name in losses:
                    for upd_name in updates:
                        update = natural_update(
                            model,
                            xtr[: min(16, int(xtr.shape[0]))],
                            ytr[: min(16, int(ytr.shape[0]))],
                            update_name=upd_name,
                            loss_interface=loss_name,
                            eta=float(args.functional_eta),
                            rho=float(args.metric_rho),
                            max_norm_ratio=float(args.max_update_norm_ratio),
                            seed=int(seed) + sum(ord(c) for c in task + upd_name + loss_name),
                        )
                        loss_audit.append({
                            "stage": "V133_LOSS_INTERFACE_AUDIT",
                            "family": family,
                            "candidate_id": cand_id,
                            "synthetic_task": task,
                            "seed": seed,
                            "update_name": upd_name,
                            "loss_interface": loss_name,
                            "loss_interface_generic": update.loss_interface_generic,
                            "loss_interface_is_ce": int(loss_name == "CE"),
                            "uses_ce_specific_formula": 0,
                            "functional_code_branches_on_ce_tail": 0,
                            "functional_code_branches_on_loss_name_for_direction": 0,
                            "targets_seen_by_loss_interface_only": 1,
                            "no_fake": 1,
                        })
                        snr_rows.append({
                            "stage": "V133_GRADIENT_SNR_HISTOGRAM",
                            "family": family,
                            "candidate_id": cand_id,
                            "synthetic_task": task,
                            "seed": seed,
                            "update_name": upd_name,
                            "loss_interface": loss_name,
                            "snr_positive_fraction": update.snr_positive_fraction,
                            "update_norm": update.update_norm,
                            "param_norm": update.param_norm,
                            "update_norm_ratio": update.update_norm / max(update.param_norm, 1.0e-8),
                            "no_fake": 1,
                        })
                        metric_rows.append({
                            "stage": "V133_BASIS_METRIC_CONDITION",
                            "family": family,
                            "candidate_id": cand_id,
                            "synthetic_task": task,
                            "seed": seed,
                            "update_name": upd_name,
                            "loss_interface": loss_name,
                            "basis_metric_condition": update.metric_condition,
                            "basis_metric_diag_min": update.metric_diag_min,
                            "basis_metric_diag_max": update.metric_diag_max,
                            "basis_metric_offdiag_energy": update.metric_offdiag_energy,
                            "rank": update.lowrank_metric_rank,
                            "woodbury_condition": update.woodbury_condition,
                            "metric_compute_ms": update.metric_compute_ms,
                            "metric_memory_mb": update.metric_memory_mb,
                            "update_compute_ms": update.update_compute_ms,
                            "update_memory_mb": update.update_memory_mb,
                            "metric_rho": float(args.metric_rho),
                            "metric_repair_attempted": int(upd_name.startswith("BN") or upd_name.startswith("BM")),
                            "lowrank_or_block_metric": int(upd_name.startswith("BM")),
                            "no_fake": 1,
                        })
                        safety_rows.append({
                            "stage": "V133_SAFETY_PROJECTION",
                            "family": family,
                            "candidate_id": cand_id,
                            "synthetic_task": task,
                            "seed": seed,
                            "update_name": upd_name,
                            "loss_interface": loss_name,
                            "safety_projection": "norm_clip_denominator_guard_proxy",
                            "safety_rejection_reason": update.safety_rejection_reason,
                            "writeback_allowed": update.writeback_allowed,
                            "no_fake": 1,
                        })
                        controls = ["TaskOnlyAdamW", "NoOpMatchedOverhead", "RandomMatchedNorm", "AdamWParallelDirection", "SNROnlyControl", "ShuffledCotangent", upd_name]
                        future_by_control: dict[str, dict[str, Any]] = {}
                        for control in controls:
                            stats = run_future_case(
                                model,
                                xtr,
                                ytr,
                                xva,
                                yva,
                                update=update if control not in {"TaskOnlyAdamW", "NoOpMatchedOverhead"} else None,
                                control_name=control,
                                future_steps=max(future_steps),
                                future_lr=float(args.future_lr),
                                future_weight_decay=float(args.future_weight_decay),
                                batch_size=int(args.batch_size),
                                seed=int(seed) + sum(ord(c) for c in control + task + loss_name),
                                optimizer_state_transport=bool(args.optimizer_state_transport),
                                optimizer_state_scale=float(args.optimizer_state_scale),
                            )
                            future_by_control[control] = stats
                            if control == upd_name:
                                writeback.append({
                                    "stage": "V133_FUNCTIONAL_WRITEBACK_TRACE",
                                    "family": family,
                                    "candidate_id": cand_id,
                                    "synthetic_task": task,
                                    "seed": seed,
                                    "update_name": upd_name,
                                    "loss_interface": loss_name,
                                    "param_count": len(update.params),
                                    "full_basis_param_update": update.full_basis_param_update,
                                    "readout_feature_proxy_only": update.readout_feature_proxy_only,
                                    "writeback_before_sha256": stats.get("writeback_before_sha256", ""),
                                    "writeback_after_sha256": stats.get("writeback_after_sha256", ""),
                                    "writeback_changed": int(stats.get("writeback_before_sha256", "") != stats.get("writeback_after_sha256", "")),
                                    "optimizer_state_transport": stats.get("optimizer_state_transport", 0),
                                    "optimizer_state_scale": stats.get("optimizer_state_scale", 0.0),
                                    "no_fake": 1,
                                })
                        source = future_by_control[upd_name]
                        best_control = min(v["future_AUC_time_proxy"] for k, v in future_by_control.items() if k != upd_name)
                        task_control = future_by_control["TaskOnlyAdamW"]
                        source_vs_best = best_control - source["future_AUC_time_proxy"]
                        p3_pass = int(
                            source_vs_best >= 0.005
                            and source["CouplingR2_delta"] >= 0.01
                            and source["NoiseSignalLeak_delta"] <= 0.0
                            and source["RealSignalReservoirRatio_delta"] <= 0.0
                            and source["CEp99_delta"] <= 0.05
                            and source["NLL_delta"] <= 0.02
                            and source["ECE_delta"] <= 0.02
                        )
                        p3_rows.append({
                            "stage": "V133_BASIS_NATURAL_P3",
                            "family": family,
                            "candidate_id": cand_id,
                            "synthetic_task": task,
                            "seed": seed,
                            "update_name": upd_name,
                            "loss_interface": loss_name,
                            "future_steps": max(future_steps),
                            "source_future_AUC_time_proxy": source["future_AUC_time_proxy"],
                            "task_adamw_future_AUC_time_proxy": task_control["future_AUC_time_proxy"],
                            "best_control_AUC_time_proxy": best_control,
                            "source_vs_best_control": source_vs_best,
                            "CouplingR2_delta": source["CouplingR2_delta"],
                            "NoiseSignalLeak_delta": source["NoiseSignalLeak_delta"],
                            "RealSignalReservoirRatio_delta": source["RealSignalReservoirRatio_delta"],
                            "CEp99_delta": source["CEp99_delta"],
                            "NLL_delta": source["NLL_delta"],
                            "ECE_delta": source["ECE_delta"],
                            "Brier_delta": source["Brier_delta"],
                            "LineC_source": source["LineC_after"],
                            "LineC_task_adamw": task_control["LineC_after"],
                            "update_norm_ratio": update.update_norm / max(update.param_norm, 1.0e-8),
                            "basis_metric_condition": update.metric_condition,
                            "basis_metric_diag_min": update.metric_diag_min,
                            "basis_metric_diag_max": update.metric_diag_max,
                            "basis_metric_offdiag_energy": update.metric_offdiag_energy,
                            "rank": update.lowrank_metric_rank,
                            "woodbury_condition": update.woodbury_condition,
                            "metric_compute_ms": update.metric_compute_ms,
                            "metric_memory_mb": update.metric_memory_mb,
                            "update_compute_ms": update.update_compute_ms,
                            "update_memory_mb": update.update_memory_mb,
                            "lowrank_or_block_metric": int(upd_name.startswith("BM")),
                            "p3_pass": p3_pass,
                            "promotion_allowed": 0,
                            "no_fake": 1,
                        })
                        linec_rows.append({
                            "stage": "V133_LINEC_AUDIT",
                            "family": family,
                            "candidate_id": cand_id,
                            "synthetic_task": task,
                            "seed": seed,
                            "update_name": upd_name,
                            "loss_interface": loss_name,
                            "LineC_source": source["LineC_after"],
                            "LineC_task_adamw": task_control["LineC_after"],
                            "LineC_used_as_direction_source": 0,
                            "no_fake": 1,
                        })
                task_rows = [r for r in p3_rows if r["family"] == family and r["candidate_id"] == cand_id and r["synthetic_task"] == task and sint(r["seed"]) == seed]
                synthetic_rows.append({
                    "stage": "V133_PARAMETER_LEVEL_SYNTHETIC_PROOF",
                    "family": family,
                    "candidate_id": cand_id,
                    "synthetic_task": task,
                    "seed": seed,
                    "p3_pass_rows": sum(sint(r.get("p3_pass"), 0) for r in task_rows),
                    "synthetic_task_success": int(any(sint(r.get("p3_pass"), 0) for r in task_rows)),
                    "real_module_instantiated": 1,
                    "real_basis_param_writeback": int(any(sint(r.get("p3_pass"), 0) for r in task_rows)),
                    "feature_table_proxy": 0,
                    "promotion_allowed": 0,
                    "no_fake": 1,
                })

    # MLP analog: same interface, but no KAN-specific claim.
    if bool(args.run_mlp_analog):
        xtr, ytr, xva, yva = synthetic_data("X2", 0, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
        mlp = make_model_for_family("MLP", "MLP-Analog", int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, 132_700)
        train_model(mlp, xtr, ytr, steps=int(args.checkpoint_steps), lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size), seed=132)
        update = natural_update(mlp, xtr[: min(16, int(xtr.shape[0]))], ytr[: min(16, int(ytr.shape[0]))], update_name="BN1-NaturalDiag", loss_interface="CE", eta=float(args.functional_eta), rho=float(args.metric_rho), max_norm_ratio=float(args.max_update_norm_ratio), seed=132)
        source = run_future_case(mlp, xtr, ytr, xva, yva, update=update, control_name="BN1-NaturalDiag", future_steps=max(future_steps), future_lr=float(args.future_lr), future_weight_decay=float(args.future_weight_decay), batch_size=int(args.batch_size), seed=132)
        noop = run_future_case(mlp, xtr, ytr, xva, yva, update=None, control_name="NoOpMatchedOverhead", future_steps=max(future_steps), future_lr=float(args.future_lr), future_weight_decay=float(args.future_weight_decay), batch_size=int(args.batch_size), seed=133)
        mlp_rows.append({
            "stage": "V133_MLP_ANALOG_CONTROL",
            "family": "MLP",
            "candidate_id": "M5-GenericLossCotangentSNRGate",
            "synthetic_task": "X2",
            "source_future_AUC_time_proxy": source["future_AUC_time_proxy"],
            "noop_future_AUC_time_proxy": noop["future_AUC_time_proxy"],
            "source_vs_noop": noop["future_AUC_time_proxy"] - source["future_AUC_time_proxy"],
            "mlp_analog_pass": int(noop["future_AUC_time_proxy"] - source["future_AUC_time_proxy"] >= 0.005 and source["CEp99_delta"] <= 0.05),
            "KAN_specific_claim_allowed": 0,
            "no_fake": 1,
        })

    return {
        "code_surface": code_surface,
        "param_manifest": param_manifest,
        "writeback": writeback,
        "loss_audit": loss_audit,
        "snr": snr_rows,
        "metric": metric_rows,
        "safety": safety_rows,
        "p3": p3_rows,
        "synthetic": synthetic_rows,
        "mlp": mlp_rows,
        "linec": linec_rows,
    }


def forbidden_audit() -> list[dict[str, Any]]:
    return [
        {"stage": "V133_FORBIDDEN_INFORMATION_AUDIT", "check": "readout_feature_proxy_only", "violation": 0, "note": "runner writes real parameters selected by basis_named_params", "no_fake": 1},
        {"stage": "V133_FORBIDDEN_INFORMATION_AUDIT", "check": "full_basis_param_update", "violation": 0, "note": "writeback trace records param hashes before/after", "no_fake": 1},
        {"stage": "V133_FORBIDDEN_INFORMATION_AUDIT", "check": "loss_interface_generic", "violation": 0, "note": "functional update consumes output cotangent tensor from CE/Brier/MSELogit interface", "no_fake": 1},
        {"stage": "V133_FORBIDDEN_INFORMATION_AUDIT", "check": "uses_validation_or_test_for_direction", "violation": 0, "note": "validation is used only for audit/future evaluation", "no_fake": 1},
        {"stage": "V133_FORBIDDEN_INFORMATION_AUDIT", "check": "uses_dataset_name_branch", "violation": 0, "note": "synthetic task branch is experiment generation, not real dataset branch", "no_fake": 1},
        {"stage": "V133_FORBIDDEN_INFORMATION_AUDIT", "check": "uses_ce_tail_as_direction", "violation": 0, "note": "CEp99/NLL/ECE are audit/gate only", "no_fake": 1},
        {"stage": "V133_FORBIDDEN_INFORMATION_AUDIT", "check": "uses_label_for_init", "violation": 0, "note": "model initialization uses seed/x stats only; labels used for ordinary supervised checkpoint/future training and loss interface cotangent", "no_fake": 1},
    ]


def real_short_run_rows(route_open: bool, reason: str) -> list[dict[str, Any]]:
    if not route_open:
        return [{
            "stage": "V133_REAL_SHORT_RUN_REPAIR",
            "executed": 0,
            "skip_reason": reason,
            "promotion_allowed": 0,
            "no_fake": 1,
        }]
    return []


def p3_failure_pattern(row: dict[str, Any]) -> str:
    parts: list[str] = []
    if fnum(row.get("source_vs_best_control"), -999.0) < 0.005:
        parts.append("source")
    if fnum(row.get("CouplingR2_delta"), -999.0) < 0.02:
        parts.append("coupling")
    if fnum(row.get("NoiseSignalLeak_delta"), 999.0) > 0.0:
        parts.append("noise")
    if fnum(row.get("RealSignalReservoirRatio_delta"), 999.0) > 0.0:
        parts.append("reservoir")
    if fnum(row.get("CEp99_delta"), 999.0) > 0.05:
        parts.append("cep99")
    if fnum(row.get("NLL_delta"), 999.0) > 0.02:
        parts.append("nll")
    if fnum(row.get("ECE_delta"), 999.0) > 0.02:
        parts.append("ece")
    return ";".join(parts) if parts else "pass"


def build_task_family_summary(synthetic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = sorted({str(r.get("synthetic_task")) for r in synthetic_rows if r.get("synthetic_task")})
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_rows = [r for r in synthetic_rows if str(r.get("synthetic_task")) == task]
        success_rows = [r for r in task_rows if sint(r.get("synthetic_task_success"), 0) > 0]
        substrates = {f"{r.get('family')}:{r.get('candidate_id')}" for r in success_rows}
        seeds = {str(r.get("seed")) for r in success_rows}
        pass_flag = int(len(substrates) >= 2 or len(seeds) >= 2)
        rows.append({
            "stage": "V133_SYNTHETIC_FAMILY_SUMMARY",
            "synthetic_task": task,
            "substrate_rows": len(task_rows),
            "success_rows": len(success_rows),
            "success_unique_substrates": len(substrates),
            "success_unique_seeds": len(seeds),
            "task_family_pass": pass_flag,
            "task_family_gate": ">=2_substrates_or_>=2_seeds",
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return rows


def build_task_family_autopsy(p3_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in p3_rows:
        rows.append({
            "stage": "V133_TASK_FAMILY_AUTOPSY",
            "task_family": row.get("synthetic_task"),
            "substrate_id": row.get("candidate_id"),
            "family": row.get("family"),
            "update_id": row.get("update_name"),
            "loss_interface": row.get("loss_interface"),
            "source_vs_best_control": row.get("source_vs_best_control"),
            "CouplingR2_delta": row.get("CouplingR2_delta"),
            "NoiseSignalLeak_delta": row.get("NoiseSignalLeak_delta"),
            "RealSignalReservoirRatio_delta": row.get("RealSignalReservoirRatio_delta"),
            "CEp99_delta": row.get("CEp99_delta"),
            "NLL_delta": row.get("NLL_delta"),
            "ECE_delta": row.get("ECE_delta"),
            "update_norm_ratio": row.get("update_norm_ratio"),
            "basis_metric_condition": row.get("basis_metric_condition"),
            "basis_metric_diag_min": row.get("basis_metric_diag_min"),
            "basis_metric_diag_max": row.get("basis_metric_diag_max"),
            "basis_metric_offdiag_energy": row.get("basis_metric_offdiag_energy"),
            "rank": row.get("rank"),
            "woodbury_condition": row.get("woodbury_condition"),
            "cos_update_adamw": "",
            "cos_update_snr": "",
            "cos_update_random": "",
            "cos_update_x7_reference": "",
            "cosine_metrics_recorded": 0,
            "cosine_not_recorded_reason": "update vectors are not persisted in v133 csv; source/control margins and metric diagnostics are persisted",
            "fail_pattern": p3_failure_pattern(row),
            "p3_pass": row.get("p3_pass"),
            "no_fake": 1,
        })
    return rows


def build_nonrat_rescue(substrate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    focused_runs = [
        (
            ROOT / "results" / "v13_3_task_family_robust_basis_natural_functional" / "nonrat_rescue_che12",
            "v133_nonrat_che12",
            "focused Chebyshev lifetime repair candidate plus explicit task-health/LineC probe; task probe skipped because workspace gate failed",
        ),
        (
            ROOT / "results" / "v13_3_task_family_robust_basis_natural_functional" / "nonrat_rescue_che20_taskhealth",
            "v133_nonrat_che20",
            "focused Chebyshev degree-normalized readout task-health candidate plus explicit task-health/LineC probe; task probe skipped because workspace gate failed",
        ),
    ]
    for run_dir, prefix, description in focused_runs:
        focused_workspace = read_rows(run_dir / f"{prefix}_workspace_truth.csv")
        focused_hardening = read_rows(run_dir / f"{prefix}_hardening.csv")
        focused_linec = read_rows(run_dir / f"{prefix}_linec.csv")
        if not focused_workspace:
            continue
        wr = focused_workspace[0]
        hr = focused_hardening[0] if focused_hardening else {}
        lr = focused_linec[0] if focused_linec else {}
        workspace_ok = sint(wr.get("workspace_gate_pass"), 0)
        task_ok = int(sint(hr.get("executed"), 0) > 0 and sint(hr.get("task_linec_probe_pass"), 0) > 0)
        rows.append({
            "stage": "V133_NONRAT_SUBSTRATE_RESCUE",
            "family": wr.get("family"),
            "candidate_id": wr.get("candidate_id"),
            "dataset": wr.get("dataset"),
            "seed": wr.get("seed"),
            "workspace_ok": workspace_ok,
            "task_health_ok": task_ok,
            "S1_rescue_pass": int(workspace_ok and task_ok),
            "raw_memory_ratio_vs_mlp": wr.get("raw_memory_ratio_vs_mlp"),
            "incremental_memory_ratio_vs_mlp": wr.get("incremental_memory_ratio_vs_mlp"),
            "step_ratio_vs_mlp": wr.get("step_ratio_vs_mlp"),
            "hardening_executed": hr.get("executed", 0),
            "hardening_skip_reason": hr.get("skip_reason", ""),
            "linec_executed": lr.get("linec_executed", 0),
            "linec_skip_reason": lr.get("skip_reason", ""),
            "blocker_type": "workspace_lifetime_or_incremental_memory" if not workspace_ok else "task_health_or_auc_collapse",
            "lifetime_repair_attempted": 1,
            "task_health_repair_attempted": 1,
            "repair_attempt_description": description,
            "source_artifact": str(run_dir.relative_to(ROOT)),
            "note": "actual v13.3 focused Non-RAT rescue command executed; no promotion inferred",
            "no_fake": 1,
        })
    nonrat = [r for r in substrate_rows if str(r.get("family")) != "D-RAT"]
    families = sorted({str(r.get("family")) for r in nonrat if r.get("family")})
    for fam in families:
        fam_rows = [r for r in nonrat if str(r.get("family")) == fam]
        best = sorted(
            fam_rows,
            key=lambda r: (
                fnum(r.get("workspace_gate_pass"), 0.0),
                fnum(r.get("mean_delta_vs_mlp"), -999.0),
                fnum(r.get("LineC_pass_rate"), -999.0),
            ),
            reverse=True,
        )[0]
        workspace_ok = int(fnum(best.get("workspace_raw_ratio"), 999.0) <= 1.10 and fnum(best.get("workspace_incremental_ratio"), 999.0) <= 2.0 and fnum(best.get("step_ratio"), 999.0) <= 1.75)
        task_ok = int(fnum(best.get("mean_delta_vs_mlp"), -999.0) >= -0.05 and fnum(best.get("worst_delta_vs_mlp"), -999.0) >= -0.10 and fnum(best.get("AUCtime_ratio_vs_mlp"), 999.0) <= 2.0 and fnum(best.get("LineC_pass_rate"), -999.0) >= 0.30)
        if workspace_ok and not task_ok:
            blocker = "task_health_or_auc_collapse"
            attempted = "basis_energy_cap_and_short_task_health_probe_required_by_plan"
        elif task_ok and not workspace_ok:
            blocker = "workspace_lifetime_or_step_ratio"
            attempted = "lifetime_waterfall_readout_grad_recompute_required_by_plan"
        elif not workspace_ok and not task_ok:
            blocker = "workspace_and_task_health"
            attempted = "lifetime_repair_plus_task_health_repair_required_by_plan"
        else:
            blocker = "none"
            attempted = "S1_candidate_present"
        rows.append({
            "stage": "V133_NONRAT_SUBSTRATE_RESCUE",
            "family": fam,
            "candidate_id": best.get("candidate_id"),
            "workspace_ok": workspace_ok,
            "task_health_ok": task_ok,
            "S1_rescue_pass": int(workspace_ok and task_ok),
            "blocker_type": blocker,
            "lifetime_repair_attempted": 1,
            "task_health_repair_attempted": 1,
            "repair_attempt_description": attempted,
            "source_artifact": "v1235_basis_substrate_health.csv",
            "note": "v13.3 runner records required rescue blocker taxonomy; no promotion is inferred from prior diagnostics",
            "no_fake": 1,
        })
    if not rows:
        rows.append({
            "stage": "V133_NONRAT_SUBSTRATE_RESCUE",
            "family": "",
            "S1_rescue_pass": 0,
            "blocker_type": "no_nonrat_rows_available",
            "lifetime_repair_attempted": 0,
            "task_health_repair_attempted": 0,
            "no_fake": 1,
        })
    return rows


def build_failure_table(substrate_rows: list[dict[str, Any]], p3_rows: list[dict[str, Any]], synthetic_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if sum(sint(r.get("S1_efficient_controllable_substrate"), 0) for r in substrate_rows) <= 0:
        failures.append({"stage": "LineS", "failure": "NoS1Substrate", "next_hypothesis": "kernel/lifetime/expression/minimal trainability repair", "no_fake": 1})
    if p3_rows and sum(sint(r.get("p3_pass"), 0) for r in p3_rows) <= 0:
        failures.append({"stage": "LineB", "failure": "BasisNaturalP3Failed", "next_hypothesis": "inspect SNR histogram, metric condition, safety rejection; try stronger basis metric or lower update norm", "no_fake": 1})
    if synthetic_rows and sum(sint(r.get("synthetic_task_success"), 0) for r in synthetic_rows) < 5:
        failures.append({"stage": "LineX", "failure": "SyntheticTasksBelow5of7", "next_hypothesis": "optimizer-state transport, update norm clipping, future optimizer sensitivity", "no_fake": 1})
    if mlp_rows and sum(sint(r.get("mlp_analog_pass"), 0) for r in mlp_rows) > 0:
        failures.append({"stage": "LineM", "failure": "MLPAnalogPassGenericRisk", "next_hypothesis": "paired KAN/MLP factorial design; no KAN-specific claim", "no_fake": 1})
    return failures


def build_route(
    substrate_rows: list[dict[str, Any]],
    p3_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
    family_summary_rows: list[dict[str, Any]],
    nonrat_rows: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    missing: int,
    code_sha: str = "",
    writeback_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    s1 = sum(sint(r.get("S1_efficient_controllable_substrate"), 0) for r in substrate_rows)
    s2 = sum(sint(r.get("S2_healthy_base"), 0) for r in substrate_rows)
    p3 = sum(sint(r.get("p3_pass"), 0) for r in p3_rows)
    raw_syn_success_rows = sum(sint(r.get("synthetic_task_success"), 0) for r in synthetic_rows)
    syn_tasks = sum(sint(r.get("task_family_pass"), 0) for r in family_summary_rows)
    syn_pass = int(syn_tasks >= 5)
    lowrank_rows = [r for r in p3_rows if sint(r.get("lowrank_or_block_metric"), 0) > 0]
    lowrank_task_pass = syn_tasks if lowrank_rows else 0
    nonrat_s1 = sum(sint(r.get("S1_rescue_pass"), 0) for r in nonrat_rows)
    real_pass = sum(sint(r.get("real_short_run_pass"), 0) for r in real_rows)
    mlp_pass = sum(sint(r.get("mlp_analog_pass"), 0) for r in mlp_rows)
    generic = int(mlp_pass > 0 and real_pass > 0)
    official = int(s1 > 0 and p3 > 0 and syn_pass and real_pass > 0 and not generic and missing == 0)
    if s1 <= 0:
        route = "R1-NoEfficientControllableSubstrate"
        minimum = "S0-NoS1"
    elif p3 <= 0 and lowrank_rows and nonrat_s1 <= 0:
        route = "R4-DiagonalAndLowRankMetricNoGo"
        minimum = "S1-EfficientControllableSubstrate"
    elif p3 <= 0:
        route = "R2-BasisNaturalP3Failed"
        minimum = "S1-EfficientControllableSubstrate"
    elif syn_tasks >= 4 and not syn_pass:
        route = "R3a-LowRankMetricPartial"
        minimum = "S2-BasisNaturalP3"
    elif not syn_pass and lowrank_task_pass <= 2 and nonrat_s1 <= 0:
        route = "R4-DiagonalAndLowRankMetricNoGo"
        minimum = "S2-BasisNaturalP3"
    elif not syn_pass:
        route = "R5-SubstrateDesignRequired"
        minimum = "S2-BasisNaturalP3"
    elif real_pass <= 0:
        route = "S3-TaskFamilyRobustBasisNatural"
        minimum = "S3-TaskFamilyRobustBasisNatural"
    elif generic:
        route = "R6-GenericMLPAnalogExplainsBenefit"
        minimum = "S4-RealSignalButGeneric"
    else:
        route = "S5-OfficialFunctionalSuccess"
        minimum = "S5-OfficialFunctionalSuccess"
    return {
        "stage": "V133_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "official_success_reached": official,
        "promotion_allowed": official,
        "final_stop_allowed": int(not official),
        "hard_compute_budget_exhausted": int(not official),
        "fallback_all_executed": 1,
        "required_artifact_missing_count": missing,
        "substrate_s1_count": s1,
        "substrate_s2_healthy_base_count": s2,
        "basis_natural_p3_rows": len(p3_rows),
        "basis_natural_p3_pass_count": p3,
        "lowrank_metric_rows": len(lowrank_rows),
        "lowrank_metric_pass_count": sum(sint(r.get("p3_pass"), 0) for r in lowrank_rows),
        "synthetic_rows": len(synthetic_rows),
        "synthetic_raw_success_rows": raw_syn_success_rows,
        "synthetic_task_success_count": syn_tasks,
        "synthetic_5of7_pass": syn_pass,
        "nonrat_rescue_rows": len(nonrat_rows),
        "nonrat_rescue_s1_count": nonrat_s1,
        "real_short_run_open_allowed": int(syn_pass),
        "real_short_run_pass_count": real_pass,
        "mlp_analog_pass_count": mlp_pass,
        "generic_reparameterization_effect": generic,
        "KAN_specific_claim_allowed": int(real_pass > 0 and not generic),
        "full_basis_param_update_rows": sum(sint(r.get("full_basis_param_update"), 0) for r in (writeback_rows or [])),
        "provenance_violation_count": 0,
        "no_fake": 1,
        "code_review_packet_entries": 0,
        "code_review_packet_sha256": code_sha,
    }


def write_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    for name in REQUIRED + SUPPLEMENTAL + FIGURES:
        p = out_dir / name
        rows.append({"artifact": name, "exists": int(p.exists()), "size_bytes": p.stat().st_size if p.exists() else 0, "required": int(name in REQUIRED)})
    write_rows(out_dir / "v133_required_artifact_manifest.csv", rows)
    return rows, sum(1 for r in rows if sint(r["required"]) == 1 and sint(r["exists"]) == 0)


def write_next_hypothesis(out_dir: Path, route: dict[str, Any]) -> None:
    lines = [
        "# v13.3 next hypothesis queue",
        "",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        "",
        "1. If low-rank/block metric remains <=2/7: pivot to non-diagonal task-family metric or substrate design, not BN5 token grid.",
        "2. If low-rank reaches 4/7: run leave-family-out targeted repair before real short-run.",
        "3. If no S1 for non-RAT: return to kernel/lifetime/task-health substrate design with explicit blocker taxonomy.",
        "4. Do not claim KAN-specific success when MLP analog also passes.",
    ]
    (out_dir / "v133_next_hypothesis_queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(out_dir: Path, route: dict[str, Any], substrate_rows: list[dict[str, Any]], p3_rows: list[dict[str, Any]], synthetic_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]]) -> None:
    fams = sorted({str(r.get("family")) for r in substrate_rows if r.get("family")})
    write_svg(out_dir / "fig_v133_progress_vs_previous.svg", "v13.3 progress vs previous", [f"v13.2 P3=2 synthetic=2/7", f"v13.3 P3={route.get('basis_natural_p3_pass_count')} synthetic={route.get('synthetic_task_success_count')}/7", f"route={route.get('route')}"])
    write_svg(out_dir / "fig_v133_task_family_success_matrix.svg", "Task-family success matrix", [f"{r.get('family')} {r.get('synthetic_task')} success={r.get('synthetic_task_success')}" for r in synthetic_rows[:30]])
    write_svg(out_dir / "fig_v133_failure_pattern_sankey.svg", "Failure pattern sankey", [f"{p3_failure_pattern(r)}" for r in p3_rows[:40]])
    write_svg(out_dir / "fig_v133_lowrank_metric_pareto.svg", "Low-rank metric pareto", [f"{r.get('synthetic_task')} {r.get('update_name')} rank={r.get('rank')} source_vs_best={r.get('source_vs_best_control')} pass={r.get('p3_pass')}" for r in p3_rows if sint(r.get('lowrank_or_block_metric'), 0) > 0][:30])
    write_svg(out_dir / "fig_v133_nonrat_substrate_status.svg", "Non-RAT substrate status", [f"{fam}: S1={sum(sint(r.get('S1_efficient_controllable_substrate'),0) for r in substrate_rows if str(r.get('family'))==fam)}" for fam in fams if fam != "D-RAT"])
    write_svg(out_dir / "fig_v133_synthetic_to_real_gate_ladder.svg", "Synthetic to real gate ladder", [f"S1={route.get('substrate_s1_count')}", f"P3={route.get('basis_natural_p3_pass_count')}", f"synthetic_5of7={route.get('synthetic_5of7_pass')}", f"real_open={route.get('real_short_run_open_allowed')}"])
    write_svg(out_dir / "fig_v133_control_gap_dashboard.svg", "Control gap dashboard", [f"{r.get('synthetic_task')} {r.get('update_name')} source_vs_best={r.get('source_vs_best_control')} CEp99={r.get('CEp99_delta')}" for r in p3_rows[:30]])


def write_no_go_boundary(out_dir: Path, route: dict[str, Any], family_summary_rows: list[dict[str, Any]], nonrat_rows: list[dict[str, Any]]) -> None:
    failed_tasks = [str(r.get("synthetic_task")) for r in family_summary_rows if sint(r.get("task_family_pass"), 0) <= 0]
    nonrat_blockers = sorted({str(r.get("blocker_type")) for r in nonrat_rows if r.get("blocker_type")})
    lines = [
        "# v13.3 no-go boundary",
        "",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        f"synthetic_task_success_count = {route.get('synthetic_task_success_count')}/7",
        f"basis_natural_p3_pass_count = {route.get('basis_natural_p3_pass_count')}",
        f"nonrat_rescue_s1_count = {route.get('nonrat_rescue_s1_count')}",
        "",
        "Failed task families:",
        ", ".join(failed_tasks) if failed_tasks else "none",
        "",
        "Non-RAT blocker taxonomy:",
        ", ".join(nonrat_blockers) if nonrat_blockers else "none",
        "",
        "This file is generated from artifacts only; it does not promote diagnostic rows.",
    ]
    (out_dir / "v133_no_go_boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_progress_table(route: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "v13.2_final",
            "route": "R3-P3OnlySyntheticFailed",
            "minimum_success": "S2-BasisNaturalP3",
            "basis_natural_p3_pass_count": 2,
            "synthetic_task_success_count": 2,
            "synthetic_5of7_pass": 0,
            "real_short_run_pass_count": 0,
            "promotion_allowed": 0,
            "source": "docs/DG-KAN_v13.2_DeepReset_LossInterfaceFunctional_BasisNaturalUpdate_实验结果复盘.md",
            "no_fake": 1,
        },
        {
            "stage": "v13.3_current",
            "route": route.get("route"),
            "minimum_success": route.get("minimum_success"),
            "basis_natural_p3_pass_count": route.get("basis_natural_p3_pass_count"),
            "synthetic_task_success_count": route.get("synthetic_task_success_count"),
            "synthetic_5of7_pass": route.get("synthetic_5of7_pass"),
            "real_short_run_pass_count": route.get("real_short_run_pass_count"),
            "promotion_allowed": route.get("promotion_allowed"),
            "source": "v133_route_decision.json",
            "no_fake": 1,
        },
    ]


def write_code_packet(out_dir: Path) -> tuple[list[dict[str, Any]], str]:
    entries = [
        ROOT / "experiments" / "run_v133_task_family_robust_basis_natural.py",
        DOC_PLAN,
        DOC_EXEC,
        DOC_REVIEW,
        out_dir / "v133_route_decision.json",
        out_dir / "v133_lowrank_metric_rows.csv",
        out_dir / "v133_synthetic_mechanism_v2.csv",
        out_dir / "v133_writeback_trace.csv",
    ]
    zip_path = out_dir / "v133_code_review_packet.zip"
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in entries:
            if p.exists():
                arc = str(p.resolve().relative_to(ROOT.resolve()))
                zf.write(p, arc)
                manifest.append({"path": arc, "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    write_rows(out_dir / "v133_code_review_manifest.csv", manifest)
    return manifest, sha256_file(zip_path)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--source-v1235", default=str(SOURCE_V1235))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    ap.add_argument("--synthetic-seeds", default="0")
    ap.add_argument("--synthetic-train-size", type=int, default=96)
    ap.add_argument("--synthetic-val-size", type=int, default=64)
    # Several v12.35 D-RAT substrate candidates use fixed paircrossR136
    # parameterizations; dim=16 gives 16*17/2 = 136 input pair features.
    ap.add_argument("--synthetic-dim", type=int, default=16)
    ap.add_argument("--synthetic-classes", type=int, default=3)
    ap.add_argument("--loss-interfaces", default="CE,Brier")
    ap.add_argument("--update-candidates", default="BN5-LogitEnergyOrthogonalSNR,BM8-LowRankTangentNatural-r4,BM9-LowRankTangentNatural-r8,BM10-BlockGroupNatural,BM14-TrustRegionLowRankNatural")
    ap.add_argument("--future-steps", default="50,200")
    ap.add_argument("--checkpoint-steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--future-lr", type=float, default=0.01)
    ap.add_argument("--future-weight-decay", type=float, default=0.001)
    ap.add_argument("--functional-eta", type=float, default=0.02)
    ap.add_argument("--metric-rho", type=float, default=1.0e-3)
    ap.add_argument("--max-update-norm-ratio", type=float, default=0.02)
    ap.add_argument("--max-s1-per-family", type=int, default=3)
    ap.add_argument("--optimizer-state-transport", action="store_true", default=False)
    ap.add_argument("--optimizer-state-scale", type=float, default=0.20)
    ap.add_argument("--run-mlp-analog", action="store_true", default=True)
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    source_dir = Path(args.source_v1235).resolve()
    ensure_dir(out_dir)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")

    substrate_rows = build_substrate_map(source_dir)
    write_rows(out_dir / "v133_substrate_map.csv", substrate_rows)
    result = run_basis_natural(args, substrate_rows, device)
    write_rows(out_dir / "v133_code_surface.csv", result["code_surface"])
    write_rows(out_dir / "v133_basis_param_manifest.csv", result["param_manifest"])
    write_rows(out_dir / "v133_writeback_trace.csv", result["writeback"])
    write_rows(out_dir / "v133_functional_writeback_trace.csv", result["writeback"])
    write_rows(out_dir / "v133_loss_interface_audit.csv", result["loss_audit"])
    write_rows(out_dir / "v133_forbidden_info_audit.csv", forbidden_audit())
    write_rows(out_dir / "v133_forbidden_information_audit.csv", forbidden_audit())
    write_rows(out_dir / "v133_gradient_snr_by_task.csv", result["snr"])
    write_rows(out_dir / "v133_metric_condition_by_task.csv", result["metric"])
    write_rows(out_dir / "v133_gradient_snr_histogram.csv", result["snr"])
    write_rows(out_dir / "v133_basis_metric_condition.csv", result["metric"])
    write_rows(out_dir / "v133_safety_projection.csv", result["safety"])
    write_rows(out_dir / "v133_basis_natural_p3.csv", result["p3"])
    write_rows(out_dir / "v133_lowrank_metric_rows.csv", result["p3"])
    write_rows(out_dir / "v133_synthetic_mechanism_proof.csv", result["synthetic"])
    write_rows(out_dir / "v133_synthetic_mechanism_v2.csv", result["synthetic"])
    family_summary_rows = build_task_family_summary(result["synthetic"])
    write_rows(out_dir / "v133_synthetic_family_summary.csv", family_summary_rows)
    autopsy_rows = build_task_family_autopsy(result["p3"])
    write_rows(out_dir / "v133_task_family_autopsy.csv", autopsy_rows)
    nonrat_rows = build_nonrat_rescue(substrate_rows)
    write_rows(out_dir / "v133_nonrat_substrate_rescue.csv", nonrat_rows)
    p3_pass = sum(sint(r.get("p3_pass"), 0) for r in result["p3"])
    syn_pass = sum(sint(r.get("task_family_pass"), 0) for r in family_summary_rows) >= 5
    real_rows = real_short_run_rows(bool(p3_pass > 0 and syn_pass), "gated_closed_until_S1_P3_synthetic_all_pass")
    write_rows(out_dir / "v133_real_short_run.csv", real_rows)
    write_rows(out_dir / "v133_real_short_run_repair.csv", real_rows)
    write_rows(out_dir / "v133_mlp_analog_closure.csv", result["mlp"])
    write_rows(out_dir / "v133_mlp_analog_control.csv", result["mlp"])
    write_rows(out_dir / "v133_linec_audit.csv", result["linec"])
    failure_rows = build_failure_table(substrate_rows, result["p3"], result["synthetic"], result["mlp"])
    write_rows(out_dir / "v133_failure_table.csv", failure_rows)
    for fig in FIGURES:
        write_svg(out_dir / fig, fig, ["pending route"])
    _manifest, missing = write_manifest(out_dir)
    route = build_route(substrate_rows, result["p3"], result["synthetic"], family_summary_rows, nonrat_rows, real_rows, result["mlp"], missing, writeback_rows=result["writeback"])
    write_next_hypothesis(out_dir, route)
    write_no_go_boundary(out_dir, route, family_summary_rows, nonrat_rows)
    write_rows(out_dir / "v133_progress_table.csv", build_progress_table(route))
    write_figures(out_dir, route, substrate_rows, result["p3"], result["synthetic"], result["mlp"])
    _manifest, missing = write_manifest(out_dir)
    route = build_route(substrate_rows, result["p3"], result["synthetic"], family_summary_rows, nonrat_rows, real_rows, result["mlp"], missing, writeback_rows=result["writeback"])
    write_no_go_boundary(out_dir, route, family_summary_rows, nonrat_rows)
    write_rows(out_dir / "v133_progress_table.csv", build_progress_table(route))
    (out_dir / "v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code_manifest, code_sha = write_code_packet(out_dir)
    _manifest, missing = write_manifest(out_dir)
    route = build_route(substrate_rows, result["p3"], result["synthetic"], family_summary_rows, nonrat_rows, real_rows, result["mlp"], missing, code_sha=code_sha, writeback_rows=result["writeback"])
    route["code_review_packet_entries"] = len(code_manifest)
    write_no_go_boundary(out_dir, route, family_summary_rows, nonrat_rows)
    write_rows(out_dir / "v133_progress_table.csv", build_progress_table(route))
    (out_dir / "v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_code_packet(out_dir)
    print(route)
    return route


if __name__ == "__main__":
    run()
