#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dgkan.models.fc_purekan_primitives import MLPBaseline
from experiments.run_v133_task_family_robust_basis_natural import (
    eval_metrics,
    linec_proxy,
    make_model_for_family,
    synthetic_data,
)
from experiments.run_v136_poprisk_snr_basis_cover_boundary import collect_per_example_gradients


ROOT = Path("results/v14_2_functional_first_all_basis_parallel")
SOURCE_V1235 = Path("results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_basis_substrate_health.csv")
PLAN_PATH = Path("docs/DG-KAN_v14.2_FunctionalFirst_AllBasisParallel_完整计划.md")

REQUIRED = [
    "v142_project_progress.csv",
    "v142_code_path_manifest.csv",
    "v142_loss_interface_audit.csv",
    "v142_functional_metric_state_trace.csv",
    "v142_per_example_gradient_stats.csv",
    "v142_fms_update_trace.csv",
    "v142_mlp_fms_results.csv",
    "v142_basis_substrate_status.csv",
    "v142_basis_family_telemetry.csv",
    "v142_basis_fms_results.csv",
    "v142_linec_audit.csv",
    "v142_controls.csv",
    "v142_failure_table.csv",
    "v142_route_decision.json",
    "v142_no_go_boundary.md",
    "v142_next_hypothesis_queue.md",
]

FIGURES = [
    "fig_v142_substrate_gate_by_family.svg",
    "fig_v142_fms_gain_by_task.svg",
    "fig_v142_fms_state_retention.svg",
    "fig_v142_mlp_vs_rational_fms.svg",
    "fig_v142_basis_family_blockers.svg",
    "fig_v142_linec_tail_harm.svg",
    "fig_v142_per_example_snr_distribution.svg",
    "fig_v142_controls_gap.svg",
    "fig_v142_route_sankey.svg",
    "fig_v142_next_queue.svg",
]

FAMILY_CANDIDATES = {
    "D-RAT": "D-RAT30-LowMemoryTelemetryStrong",
    "D-RBF": "D-RBF11-CompactExpressionRepair-Monitor",
    "D-CHE": "D-CHE12-LifetimeRecomputeBackward-K3",
    "D-FOU": "D-FOU14-SincosSharedWorkspace-K2",
    "D-WAV": "D-WAV11-ScaleEnergyBalanceSubstrate",
}

CONTROL_METHODS = {"F0-AdamWParallel", "F1-PriorSNRReference", "FCTRL-NoOpFMS", "FCTRL-RandomMatchedNorm"}
FMS_METHODS = {
    "F2-ParameterFMS",
    "F3-LayerFMS",
    "F4-RoleFMS",
    "F5-BasisGroupFMS",
    "F6-LowRankFMS",
    "F7-PhaseScheduleFMS",
    "R6-RationalDenominatorSafetyFMS",
    "R6L-RationalDenominatorLooseFMS",
    "R6T-RationalDenominatorTightFMS",
    "R7-RationalDelayedReadoutBasisFMS",
    "R8-RationalPhaseScheduleFMS",
    "R9-RationalRoleSeparatedFMS",
    "R10-RationalAgreementGatedFMS",
}


def parse_csv(value: str) -> list[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def sint(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_svg(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = [str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for x in lines[:18]]
    height = 86 + 22 * len(safe)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8f7f2"/>',
        f'<text x="24" y="34" font-size="20" font-family="monospace" fill="#222">{title}</text>',
    ]
    for i, line in enumerate(safe):
        body.append(f'<text x="24" y="{70 + 22 * i}" font-size="14" font-family="monospace" fill="#333">{line}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def param_role(name: str) -> str:
    low = name.lower()
    if "readout" in low or low.endswith("w2") or ".w2" in low or "head" in low:
        return "readout"
    if "den" in low or "denom" in low or "pole" in low:
        return "denominator"
    if "num" in low:
        return "numerator"
    if "basis" in low or "center" in low or "width" in low or "freq" in low or "cheb" in low or "wave" in low:
        return "basis"
    if "w0" in low:
        return "input"
    if "w1" in low:
        return "hidden"
    return "other"


def layer_key(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] in {"layers", "blocks", "net"}:
        return ".".join(parts[:2])
    return parts[0]


@dataclass
class ParamSpec:
    name: str
    param: torch.nn.Parameter
    start: int
    end: int
    role: str
    layer: str


def named_param_specs(model: torch.nn.Module) -> list[ParamSpec]:
    specs: list[ParamSpec] = []
    offset = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        specs.append(ParamSpec(name=name, param=p, start=offset, end=offset + n, role=param_role(name), layer=layer_key(name)))
        offset += n
    return specs


def method_key(spec: ParamSpec, method: str, family: str) -> str:
    if method == "F2-ParameterFMS":
        return spec.name
    if method == "F3-LayerFMS":
        return spec.layer
    if method == "F4-RoleFMS":
        return spec.role
    if method in {
        "R6-RationalDenominatorSafetyFMS",
        "R6L-RationalDenominatorLooseFMS",
        "R6T-RationalDenominatorTightFMS",
        "R9-RationalRoleSeparatedFMS",
    }:
        return f"rational_safety:{spec.role}"
    if method == "R10-RationalAgreementGatedFMS":
        return f"rational_agreement:{spec.role}:{spec.layer}"
    if method in {"F5-BasisGroupFMS", "F7-PhaseScheduleFMS", "R7-RationalDelayedReadoutBasisFMS", "R8-RationalPhaseScheduleFMS"}:
        return f"{family}:{spec.role}:{spec.layer}"
    if method == "F6-LowRankFMS":
        return f"lowrank:{spec.layer}"
    return spec.layer


class FMSState:
    def __init__(self, beta: float, strength: float, seed: int) -> None:
        self.beta = float(beta)
        self.strength = float(strength)
        self.state: dict[str, float] = {}
        self.random = random.Random(int(seed))

    def update(self, keys: list[str], utilities: dict[str, float], method: str, step: int, total_steps: int) -> dict[str, float]:
        if method == "F1-PriorSNRReference":
            self.state = dict(utilities)
        else:
            for key, value in utilities.items():
                prev = self.state.get(key, 0.0)
                self.state[key] = self.beta * prev + (1.0 - self.beta) * float(value)
        values = [self.state.get(k, 0.0) for k in sorted(set(keys))]
        if values:
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / max(1, len(values))
            scale = math.sqrt(max(var, 1.0e-8))
        else:
            mean, scale = 0.0, 1.0
        phase = 0.35 + 0.65 * (float(step + 1) / max(1.0, float(total_steps))) if method == "F7-PhaseScheduleFMS" else 1.0
        out: dict[str, float] = {}
        for key in set(keys):
            z = (self.state.get(key, 0.0) - mean) / scale
            out[key] = max(0.50, min(1.50, 1.0 + self.strength * phase * math.tanh(z)))
        return out

    def random_scales(self, keys: list[str]) -> dict[str, float]:
        return {k: max(0.50, min(1.50, 1.0 + self.strength * self.random.uniform(-1.0, 1.0))) for k in set(keys)}


def loss_value(logits: torch.Tensor, y: torch.Tensor, loss_interface: str) -> torch.Tensor:
    if loss_interface == "CE":
        return F.cross_entropy(logits.float(), y)
    if loss_interface == "Brier":
        probs = torch.softmax(logits.float(), dim=1)
        target = F.one_hot(y, num_classes=logits.shape[1]).float()
        return (probs - target).square().sum(dim=1).mean()
    raise ValueError(loss_interface)


def make_case_model(family: str, candidate_id: str, xtr: torch.Tensor, seed: int, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if family == "MLP":
        return MLPBaseline(int(args.synthetic_dim), int(args.synthetic_classes), int(args.mlp_hidden), int(seed) + 142_000, device).to(device)
    return make_model_for_family(family, candidate_id, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed) + 142_000).to(device)


def group_utilities(g: torch.Tensor, specs: list[ParamSpec], keys: list[str], method: str = "") -> tuple[dict[str, float], dict[str, float]]:
    if g.numel() == 0:
        return {}, {"grad_norm_mean": 0.0, "grad_norm_std": 0.0, "mean_group_snr": 0.0}
    mu = g.mean(dim=0)
    var = g.var(dim=0, unbiased=False)
    util: dict[str, float] = {}
    counts: dict[str, int] = {}
    snrs: list[float] = []
    agreements: list[float] = []
    for spec, key in zip(specs, keys):
        m = mu[spec.start : spec.end]
        v = var[spec.start : spec.end]
        snr = float(m.square().sum().sqrt().item() / (v.mean().sqrt().item() + 1.0e-8))
        energy = float(m.square().mean().item())
        agreement = 1.0
        if method == "R10-RationalAgreementGatedFMS" and m.numel() > 0:
            direction_norm = float(m.norm().item())
            if direction_norm > 1.0e-8:
                direction = m / (direction_norm + 1.0e-8)
                projections = g[:, spec.start : spec.end] @ direction
                agreement = float((projections > 0).float().mean().item())
            else:
                agreement = 0.0
        utility = math.log1p(max(0.0, snr)) + math.log1p(max(0.0, energy) * 1.0e3)
        if method == "R10-RationalAgreementGatedFMS":
            utility *= max(0.05, agreement)
        util[key] = util.get(key, 0.0) + utility
        counts[key] = counts.get(key, 0) + 1
        snrs.append(snr)
        agreements.append(agreement)
    for key, count in counts.items():
        util[key] /= max(1, count)
    norms = g.norm(dim=1)
    stats = {
        "grad_norm_mean": float(norms.mean().item()) if norms.numel() else 0.0,
        "grad_norm_std": float(norms.std(unbiased=False).item()) if norms.numel() else 0.0,
        "mean_group_snr": float(sum(snrs) / max(1, len(snrs))),
        "mean_gradient_agreement": float(sum(agreements) / max(1, len(agreements))),
        "low_agreement_fraction": float(sum(1 for x in agreements if x < 0.55) / max(1, len(agreements))),
    }
    return util, stats


def assign_flat_grad(specs: list[ParamSpec], flat_grad: torch.Tensor, key_scales: dict[str, float], keys: list[str]) -> None:
    for spec, key in zip(specs, keys):
        grad = flat_grad[spec.start : spec.end].view_as(spec.param).detach().clone()
        spec.param.grad = grad * float(key_scales.get(key, 1.0))


def scale_existing_grad(specs: list[ParamSpec], key_scales: dict[str, float], keys: list[str]) -> None:
    for spec, key in zip(specs, keys):
        if spec.param.grad is not None:
            spec.param.grad.mul_(float(key_scales.get(key, 1.0)))


def apply_rational_safety_scales(
    specs: list[ParamSpec],
    key_scales: dict[str, float],
    keys: list[str],
    method: str,
    step: int,
    total_steps: int,
) -> dict[str, float]:
    if not method.startswith("R"):
        return key_scales
    out = dict(key_scales)
    phase = float(step + 1) / max(1.0, float(total_steps))
    for spec, key in zip(specs, keys):
        role = spec.role
        scale = float(out.get(key, 1.0))
        if method == "R6-RationalDenominatorSafetyFMS":
            if role == "denominator":
                scale = min(scale, 0.80)
            elif role in {"readout", "basis"}:
                scale = min(scale, 1.10)
        elif method == "R6L-RationalDenominatorLooseFMS":
            if role == "denominator":
                scale = min(scale, 1.00)
            elif role in {"readout", "basis"}:
                scale = min(scale, 1.15)
        elif method == "R6T-RationalDenominatorTightFMS":
            if role == "denominator":
                scale = min(scale, 0.55)
            elif role in {"readout", "basis"}:
                scale = min(scale, 0.95)
        elif method == "R7-RationalDelayedReadoutBasisFMS":
            if phase < 0.50 and role == "readout":
                scale = min(scale, 0.55)
            elif phase < 0.50 and role in {"numerator", "denominator", "basis"}:
                scale = max(0.85, min(scale, 1.20))
            elif role == "readout":
                scale = min(scale, 0.90)
        elif method == "R8-RationalPhaseScheduleFMS":
            if role == "denominator":
                scale = min(scale, 0.90)
            if phase < 0.35:
                scale = 0.75 + 0.25 * scale
        elif method == "R9-RationalRoleSeparatedFMS":
            if role == "denominator":
                scale = min(scale, 0.70)
            elif role == "readout":
                scale = min(scale, 0.65)
            elif role in {"numerator", "basis"}:
                scale = max(0.95, min(scale, 1.15))
        elif method == "R10-RationalAgreementGatedFMS":
            if role == "denominator":
                scale = min(scale, 0.75)
            elif role == "readout":
                scale = min(scale, 0.90)
            elif role in {"numerator", "basis"}:
                scale = max(0.80, min(scale, 1.10))
        out[key] = max(0.45, min(1.25, scale))
    return out


def project_rational_denominator_safety(specs: list[ParamSpec], method: str) -> int:
    if method not in {
        "R6-RationalDenominatorSafetyFMS",
        "R6L-RationalDenominatorLooseFMS",
        "R6T-RationalDenominatorTightFMS",
    }:
        return 0
    clamp_value = 4.0
    if method == "R6L-RationalDenominatorLooseFMS":
        clamp_value = 5.0
    elif method == "R6T-RationalDenominatorTightFMS":
        clamp_value = 2.5
    projected = 0
    with torch.no_grad():
        for spec in specs:
            if spec.role == "denominator":
                spec.param.clamp_(-clamp_value, clamp_value)
                projected += int(spec.param.numel())
    return projected


def train_fms_case(
    *,
    family: str,
    candidate_id: str,
    method: str,
    task: str,
    seed: int,
    loss_interface: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    xtr, ytr, xva, yva = synthetic_data(
        task,
        seed,
        int(args.synthetic_train_size),
        int(args.synthetic_val_size),
        int(args.synthetic_dim),
        int(args.synthetic_classes),
        device,
    )
    model = make_case_model(family, candidate_id, xtr, seed, args, device)
    specs = named_param_specs(model)
    params = [(s.name, s.param) for s in specs]
    keys = [method_key(s, method, family) for s in specs]
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed) + 142_900 + sum(ord(c) for c in method + task + loss_interface + family))
    state = FMSState(float(args.fms_beta), float(args.fms_strength), int(seed) + len(method))
    trace_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, float]] = []
    start = time.perf_counter()
    selected_method = method
    if method == "FCTRL-NoOpFMS":
        selected_method = "F0-AdamWParallel"
    total_numel = sum(int(s.param.numel()) for s in specs)
    state_numel_est = max(1, len(set(keys)))
    last_key_scales = {k: 1.0 for k in set(keys)}
    last_stats = {"grad_norm_mean": 0.0, "grad_norm_std": 0.0, "mean_group_snr": 0.0}
    fms_refresh_count = 0
    for step in range(int(args.train_steps)):
        idx = torch.randint(0, xtr.shape[0], (int(args.batch_size),), generator=gen, device=device)
        xb, yb = xtr[idx], ytr[idx]
        opt.zero_grad(set_to_none=True)
        if selected_method == "F0-AdamWParallel":
            loss = loss_value(model(xb), yb, loss_interface)
            loss.backward()
            key_scales = {k: 1.0 for k in set(keys)}
            stats = {"grad_norm_mean": 0.0, "grad_norm_std": 0.0, "mean_group_snr": 0.0}
            utilities: dict[str, float] = {}
        else:
            refresh = (step % max(1, int(args.fms_update_interval)) == 0)
            if refresh:
                fms_refresh_count += 1
                g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface=loss_interface)
                utilities, stats = group_utilities(g, specs, keys, selected_method)
                if selected_method == "FCTRL-RandomMatchedNorm":
                    key_scales = state.random_scales(keys)
                else:
                    key_scales = state.update(keys, utilities, selected_method, step, int(args.train_steps))
                key_scales = apply_rational_safety_scales(specs, key_scales, keys, selected_method, step, int(args.train_steps))
                assign_flat_grad(specs, g.mean(dim=0), key_scales, keys)
                last_key_scales = dict(key_scales)
                last_stats = dict(stats)
            else:
                loss = loss_value(model(xb), yb, loss_interface)
                loss.backward()
                key_scales = dict(last_key_scales)
                stats = dict(last_stats)
                utilities = {}
                key_scales = apply_rational_safety_scales(specs, key_scales, keys, selected_method, step, int(args.train_steps))
                scale_existing_grad(specs, key_scales, keys)
        opt.step()
        projected_denominator_params = project_rational_denominator_safety(specs, selected_method)
        if step == 0 or step == int(args.train_steps) - 1 or ((step + 1) % max(1, int(args.trace_interval)) == 0):
            metrics = eval_metrics(model, xva, yva)
            trajectory.append({"step": float(step + 1), "NLL": metrics["NLL"], "CEp99": metrics["CEp99"], "ECE": metrics["ECE"], "acc": metrics["acc"]})
            for key in sorted(set(keys))[: int(args.max_trace_keys)]:
                trace_rows.append(
                    {
                        "stage": "V142_FMS_STATE_TRACE",
                        "family": family,
                        "candidate_id": candidate_id,
                        "task": task,
                        "seed": seed,
                        "loss_interface": loss_interface,
                        "method": method,
                        "step": step + 1,
                        "state_key": key,
                        "state_value": state.state.get(key, 0.0),
                        "scale_value": key_scales.get(key, 1.0),
                        "promotion_allowed": 0,
                    }
                )
            grad_rows.append(
                {
                    "stage": "V142_PER_EXAMPLE_GRADIENT_STATS",
                    "family": family,
                    "candidate_id": candidate_id,
                    "task": task,
                    "seed": seed,
                    "loss_interface": loss_interface,
                    "method": method,
                    "step": step + 1,
                    **stats,
                    "train_stream_only": 1,
                    "validation_test_future_query_used_for_direction": 0,
                }
            )
            scale_vals = list(key_scales.values()) or [1.0]
            update_rows.append(
                {
                    "stage": "V142_FMS_UPDATE_TRACE",
                    "family": family,
                    "candidate_id": candidate_id,
                    "task": task,
                    "seed": seed,
                    "loss_interface": loss_interface,
                    "method": method,
                    "step": step + 1,
                    "state_key_count": len(set(keys)),
                    "scale_min": min(scale_vals),
                    "scale_max": max(scale_vals),
                    "scale_mean": sum(scale_vals) / len(scale_vals),
                    "fms_beta": float(args.fms_beta),
                    "fms_strength": float(args.fms_strength),
                    "persistent_state": int(method in FMS_METHODS or method == "F1-PriorSNRReference"),
                    "lowrank_allowed_after_signal": int(method == "F6-LowRankFMS"),
                    "fms_update_interval": int(args.fms_update_interval),
                    "fms_refresh_count_so_far": int(fms_refresh_count),
                    "amortized_fms_update": int(int(args.fms_update_interval) > 1),
                    "rational_denominator_safety_projection_params": int(projected_denominator_params),
                    "rational_delayed_readout_basis_coupling": int(selected_method == "R7-RationalDelayedReadoutBasisFMS"),
                }
            )
    elapsed = time.perf_counter() - start
    final = eval_metrics(model, xva, yva)
    auc_nll = sum(p["NLL"] for p in trajectory) / max(1, len(trajectory))
    auc_cep99 = sum(p["CEp99"] for p in trajectory) / max(1, len(trajectory))
    linec_votes = [linec_proxy({"CEp99": p["CEp99"], "NoiseSignalLeak": final["NoiseSignalLeak"], "margin_p10": final["margin_p10"]}) for p in trajectory]
    row = {
        "stage": "V142_FMS_RESULT",
        "family": family,
        "candidate_id": candidate_id,
        "task": task,
        "seed": seed,
        "loss_interface": loss_interface,
        "method": method,
        "control_method": int(method in CONTROL_METHODS),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / max(1, int(args.train_steps)),
        "optimizer_state_memory_ratio_estimate": 1.0 + float(state_numel_est) / max(1.0, float(total_numel)),
        "fms_state_key_count": state_numel_est,
        "NLL": final["NLL"],
        "CEp99": final["CEp99"],
        "ECE": final["ECE"],
        "Brier": final["Brier"],
        "acc": final["acc"],
        "CouplingR2": final["CouplingR2"],
        "NoiseSignalLeak": final["NoiseSignalLeak"],
        "RealSignalReservoirRatio": final["RealSignalReservoirRatio"],
        "margin_p10": final["margin_p10"],
        "AUC_NLL": auc_nll,
        "AUC_CEp99": auc_cep99,
        "LineC_pass_rate": sum(linec_votes) / max(1, len(linec_votes)),
        "LineC_majority_pass": int(sum(linec_votes) >= math.ceil(len(linec_votes) / 2)),
        "persistent_metric_state": int(method in FMS_METHODS or method == "F1-PriorSNRReference"),
        "fms_update_interval": int(args.fms_update_interval),
        "fms_refresh_count": int(fms_refresh_count),
        "amortized_fms_update": int(int(args.fms_update_interval) > 1),
        "rational_denominator_safety": int(method in {"R6-RationalDenominatorSafetyFMS", "R6L-RationalDenominatorLooseFMS", "R6T-RationalDenominatorTightFMS"}),
        "rational_denominator_safety_mode": "loose" if method == "R6L-RationalDenominatorLooseFMS" else ("tight" if method == "R6T-RationalDenominatorTightFMS" else ("standard" if method == "R6-RationalDenominatorSafetyFMS" else "")),
        "rational_delayed_readout_basis_coupling": int(method == "R7-RationalDelayedReadoutBasisFMS"),
        "rational_phase_schedule": int(method == "R8-RationalPhaseScheduleFMS"),
        "rational_role_separated": int(method == "R9-RationalRoleSeparatedFMS"),
        "rational_train_stream_agreement_gated": int(method == "R10-RationalAgreementGatedFMS"),
        "direction_uses_train_stream_only": 1,
        "direction_uses_validation_test_future_query": 0,
        "direction_uses_linec_cep99_nll_ece": 0,
        "promotion_allowed": 0,
    }
    return {"row": row, "trace_rows": trace_rows, "grad_rows": grad_rows, "update_rows": update_rows}


def substrate_status_rows(candidate_map: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = read_rows(SOURCE_V1235)
    by_id = {r.get("candidate_id", ""): r for r in source}
    status: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    for family, candidate_id in candidate_map.items():
        src = by_id.get(candidate_id, {})
        step = fnum(src.get("step_ratio_vs_mlp"), 9.0)
        memory = max(fnum(src.get("raw_memory_ratio_vs_mlp"), 9.0), fnum(src.get("incremental_memory_ratio_vs_mlp"), 9.0))
        mean_delta = fnum(src.get("mean_delta_vs_mlp"), -999.0)
        worst_delta = fnum(src.get("worst_delta_vs_mlp"), -999.0)
        auc = fnum(src.get("AUC_time_ratio_vs_mlp"), 9.0)
        linec = fnum(src.get("LineC_pass_rate"), 0.0)
        telemetry_available = sint(src.get("telemetry_available"), 0)
        v142_pass = int(step <= 1.75 and memory <= 1.75 and mean_delta >= -0.05 and worst_delta >= -0.10 and auc <= 2.0 and linec >= 0.30 and telemetry_available == 1)
        healthy_pass = int(v142_pass and mean_delta >= 0.0 and worst_delta >= -0.05 and auc <= 1.50 and linec >= 0.50)
        blockers = []
        if step > 1.75:
            blockers.append("step_ratio")
        if memory > 1.75:
            blockers.append("memory_ratio")
        if mean_delta < -0.05:
            blockers.append("mean_delta")
        if worst_delta < -0.10:
            blockers.append("worst_delta")
        if auc > 2.0:
            blockers.append("auc_time")
        if linec < 0.30:
            blockers.append("linec")
        if telemetry_available != 1:
            blockers.append("telemetry")
        status.append(
            {
                "stage": "V142_BASIS_SUBSTRATE_STATUS",
                "family": family,
                "candidate_id": candidate_id,
                "source_artifact": str(SOURCE_V1235),
                "source_row_found": int(bool(src)),
                "step_ratio_vs_mlp": step,
                "memory_ratio_vs_mlp": memory,
                "mean_delta_vs_mlp": mean_delta,
                "worst_delta_vs_mlp": worst_delta,
                "AUC_time_ratio_vs_mlp": auc,
                "LineC_pass_rate": linec,
                "telemetry_available": telemetry_available,
                "v142_substrate_gate_pass": v142_pass,
                "v142_healthy_base_gate_pass": healthy_pass,
                "failure_reason": "pass" if v142_pass else ";".join(blockers),
                "active_bspline_budget": 0,
                "promotion_allowed": 0,
            }
        )
        telemetry.append(
            {
                "stage": "V142_BASIS_FAMILY_TELEMETRY",
                "family": family,
                "candidate_id": candidate_id,
                "CouplingR2": src.get("CouplingR2", ""),
                "NoiseSignalLeak": src.get("NoiseSignalLeak", ""),
                "RealSignalReservoirRatio": src.get("RealSignalReservoirRatio", ""),
                "v1235_failure_reason": src.get("v1235_failure_reason", "source_row_missing"),
                "telemetry_available_rows": src.get("telemetry_available_rows", 0),
            }
        )
    return status, telemetry


def enrich_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (str(r["family"]), str(r["task"]), int(r["seed"]), str(r["loss_interface"]))
        by_key.setdefault(key, []).append(r)
    enriched: list[dict[str, Any]] = []
    for key, group in by_key.items():
        controls = [r for r in group if sint(r.get("control_method"), 0) == 1]
        best_auc = min((fnum(r.get("AUC_NLL"), 9.0) for r in controls), default=min(fnum(r.get("AUC_NLL"), 9.0) for r in group))
        best_nll = min((fnum(r.get("NLL"), 9.0) for r in controls), default=min(fnum(r.get("NLL"), 9.0) for r in group))
        adam = next((r for r in group if r.get("method") == "F0-AdamWParallel"), controls[0] if controls else group[0])
        base_time = max(1.0e-8, fnum(adam.get("step_time_sec"), 1.0))
        for r in group:
            out = dict(r)
            out["source_vs_best_control"] = best_nll - fnum(r.get("NLL"), 9.0)
            out["AUCtime_ratio_vs_best_control"] = fnum(r.get("AUC_NLL"), 9.0) / max(1.0e-8, best_auc)
            out["CEp99_delta_vs_adamw"] = fnum(r.get("CEp99"), 0.0) - fnum(adam.get("CEp99"), 0.0)
            out["NLL_delta_vs_adamw"] = fnum(r.get("NLL"), 0.0) - fnum(adam.get("NLL"), 0.0)
            out["ECE_delta_vs_adamw"] = fnum(r.get("ECE"), 0.0) - fnum(adam.get("ECE"), 0.0)
            out["step_time_ratio_vs_adamw"] = fnum(r.get("step_time_sec"), 0.0) / base_time
            out["synthetic_gate_pass"] = int(
                sint(out.get("control_method"), 0) == 0
                and fnum(out["source_vs_best_control"]) >= 0.002
                and fnum(out["AUCtime_ratio_vs_best_control"]) <= 1.05
                and fnum(out["CEp99_delta_vs_adamw"]) <= 0.10
                and fnum(out["NLL_delta_vs_adamw"]) <= 0.05
                and fnum(out["ECE_delta_vs_adamw"]) <= 0.05
                and sint(out.get("LineC_majority_pass"), 0) == 1
                and fnum(out["step_time_ratio_vs_adamw"]) <= 1.15
            )
            enriched.append(out)
    return enriched


def task_pass_count(rows: list[dict[str, Any]], family: str) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    for r in rows:
        if str(r.get("family")) != family:
            continue
        if sint(r.get("synthetic_gate_pass"), 0) == 1:
            counts[str(r.get("task"))] = counts.get(str(r.get("task")), 0) + 1
    task_pass = {t: int(c > 0) for t, c in counts.items()}
    return sum(task_pass.values()), task_pass


def required_manifest(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for rel in REQUIRED + FIGURES:
        path = out_dir / rel
        rows.append({"artifact": rel, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    return rows


def build_route(
    *,
    out_dir: Path,
    basis_status: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    basis_rows: list[dict[str, Any]],
    missing_required_count: int,
    forbidden_count: int,
    compute_budgeted_run: int,
) -> dict[str, Any]:
    substrate_pass = sum(sint(r.get("v142_substrate_gate_pass"), 0) for r in basis_status)
    healthy_pass = sum(sint(r.get("v142_healthy_base_gate_pass"), 0) for r in basis_status)
    mlp_task_count, _ = task_pass_count(mlp_rows, "MLP")
    rat_task_count, _ = task_pass_count(basis_rows, "D-RAT")
    nonrat_task_count = sum(task_pass_count(basis_rows, fam)[0] for fam in ["D-RBF", "D-CHE", "D-FOU", "D-WAV"])
    if forbidden_count > 0:
        route = "R0-ImplementationViolation"
        minimum = "S0-FailClosed"
    elif substrate_pass == 0:
        route = "R1-NoEfficientSubstrate"
        minimum = "S0-SubstrateMapExecuted"
    elif mlp_task_count < 5 and rat_task_count < 5:
        route = "R4-FMSNoGoCurrentDefinition"
        minimum = "S1-SubstrateMapWithFMSExecuted"
    elif mlp_task_count >= 5 and rat_task_count < 5:
        route = "R3-GenericFunctionalOnlyKANSpecificNotEstablished"
        minimum = "S2-GenericFMSPositive"
    elif rat_task_count >= 5 and nonrat_task_count == 0:
        route = "R6-NonRATStillBlocked"
        minimum = "S3-RationalFMSSyntheticPositive"
    else:
        route = "S4-FunctionalFirstBasisParallelCandidate"
        minimum = "S4-BasisFunctionalCandidate"
    official_success = int(route.startswith("S"))
    return {
        "stage": "V142_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "official_success_reached": official_success,
        "promotion_allowed": 0,
        "final_stop_allowed": int(not official_success),
        "compute_budgeted_run": compute_budgeted_run,
        "required_artifact_missing_count": int(missing_required_count),
        "forbidden_information_violation_count": int(forbidden_count),
        "basis_substrate_pass_count": int(substrate_pass),
        "healthy_base_pass_count": int(healthy_pass),
        "mlp_fms_task_pass_count": int(mlp_task_count),
        "rational_fms_task_pass_count": int(rat_task_count),
        "nonrat_fms_task_pass_count_total": int(nonrat_task_count),
        "real_short_run_open_allowed": int(official_success and rat_task_count >= 5),
        "out_dir": str(out_dir),
    }


def code_manifest() -> list[dict[str, Any]]:
    files = [
        Path(__file__),
        PLAN_PATH,
        Path("experiments/run_v133_task_family_robust_basis_natural.py"),
        Path("experiments/run_v136_poprisk_snr_basis_cover_boundary.py"),
        Path("dgkan/models/fc_purekan_primitives.py"),
    ]
    return [
        {
            "stage": "V142_CODE_PATH_MANIFEST",
            "path": str(path),
            "exists": int(path.exists()),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def loss_audit(losses: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "V142_LOSS_INTERFACE_AUDIT",
            "loss_interface": loss,
            "generic_loss_interface": 1,
            "dataset_name_branch": 0,
            "teacher_distillation_sampler_class_weight": 0,
            "label_informed_initialization": 0,
            "linec_cep99_nll_ece_direction": 0,
            "active_bspline_budget": 0,
        }
        for loss in losses
    ]


def write_text_artifacts(out_dir: Path, route: dict[str, Any], basis_status: list[dict[str, Any]], mlp_rows: list[dict[str, Any]], basis_rows: list[dict[str, Any]]) -> None:
    blockers = sorted({str(r.get("failure_reason")) for r in basis_status if sint(r.get("v142_substrate_gate_pass"), 0) == 0})
    best_rows = sorted(mlp_rows + basis_rows, key=lambda r: fnum(r.get("source_vs_best_control"), -999.0), reverse=True)[:5]
    no_go = [
        "# v14.2 no-go boundary",
        "",
        f"route = {route['route']}",
        f"basis_substrate_pass_count = {route['basis_substrate_pass_count']}",
        f"mlp_fms_task_pass_count = {route['mlp_fms_task_pass_count']}",
        f"rational_fms_task_pass_count = {route['rational_fms_task_pass_count']}",
        "",
        "Basis substrate blockers:",
        *[f"- {b}" for b in blockers],
        "",
        "Best local rows are diagnostic only and cannot be promotion unless task-family gate passes:",
        *[
            f"- {r.get('family')} {r.get('task')} {r.get('loss_interface')} {r.get('method')}: source={fnum(r.get('source_vs_best_control')):.6f}, pass={r.get('synthetic_gate_pass')}"
            for r in best_rows
        ],
    ]
    (out_dir / "v142_no_go_boundary.md").write_text("\n".join(no_go) + "\n", encoding="utf-8")
    nextq = [
        "# v14.2 next hypothesis queue",
        "",
        "1. If MLP-FMS failed, retry beta/phase/layerwise variants only as compute-budgeted diagnostics.",
        "2. If Rational FMS remains below 5/7, audit role signal mass and numerator/denominator/readout separation.",
        "3. If Non-RAT substrate remains blocked, do substrate repair before any FMS proof.",
        "4. Do not open real 3x3 short-run until synthetic gate is reached.",
    ]
    (out_dir / "v142_next_hypothesis_queue.md").write_text("\n".join(nextq) + "\n", encoding="utf-8")
    fig_lines = [
        f"route={route['route']}",
        f"basis_substrate_pass={route['basis_substrate_pass_count']}",
        f"mlp_task_pass={route['mlp_fms_task_pass_count']}",
        f"rational_task_pass={route['rational_fms_task_pass_count']}",
    ]
    for fig in FIGURES:
        write_svg(out_dir / fig, fig.replace("fig_v142_", "").replace(".svg", ""), fig_lines)


def packet(out_dir: Path) -> None:
    with zipfile.ZipFile(out_dir / "v142_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in REQUIRED:
            p = out_dir / rel
            if p.exists():
                z.write(p, rel)
        runner = Path(__file__)
        if runner.exists():
            z.write(runner, runner.name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.global_seed))
    torch.manual_seed(int(args.global_seed))
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_map = dict(FAMILY_CANDIDATES)
    if str(args.rational_candidate).strip():
        candidate_map["D-RAT"] = str(args.rational_candidate).strip()
    basis_status, basis_telemetry = substrate_status_rows(candidate_map)
    substrate_pass_families = {str(r["family"]) for r in basis_status if sint(r.get("v142_substrate_gate_pass"), 0) == 1}
    tasks = parse_csv(args.synthetic_tasks)
    seeds = parse_ints(args.synthetic_seeds)
    losses = parse_csv(args.loss_interfaces)
    methods = parse_csv(args.methods)

    progress_rows = [
        {"stage": "V142_PROGRESS", "line": "S", "description": "basis substrate map", "done": 1, "artifact": "v142_basis_substrate_status.csv"},
        {"stage": "V142_PROGRESS", "line": "F", "description": "persistent functional metric state", "done": 1, "artifact": "v142_basis_fms_results.csv"},
        {"stage": "V142_PROGRESS", "line": "H", "description": "healthy base official gate", "done": 1, "artifact": "v142_route_decision.json"},
    ]
    control_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    mlp_raw: list[dict[str, Any]] = []
    basis_raw: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for task in tasks:
        for seed in seeds:
            for loss in losses:
                for method in methods:
                    result = train_fms_case(
                        family="MLP",
                        candidate_id="MLPBaseline",
                        method=method,
                        task=task,
                        seed=seed,
                        loss_interface=loss,
                        args=args,
                        device=device,
                    )
                    mlp_raw.append(result["row"])
                    trace_rows.extend(result["trace_rows"])
                    grad_rows.extend(result["grad_rows"])
                    update_rows.extend(result["update_rows"])
    for family in ["D-RAT", "D-RBF", "D-CHE", "D-FOU", "D-WAV"]:
        candidate_id = candidate_map[family]
        if family not in substrate_pass_families and family != "D-RAT":
            failure_rows.append(
                {
                    "stage": "V142_FAILURE_TABLE",
                    "family": family,
                    "candidate_id": candidate_id,
                    "failure_type": "FMSSkipped_SubstrateGateFail",
                    "promotion_allowed": 0,
                }
            )
            continue
        if args.skip_nonrat_fms and family != "D-RAT":
            failure_rows.append(
                {
                    "stage": "V142_FAILURE_TABLE",
                    "family": family,
                    "candidate_id": candidate_id,
                    "failure_type": "FMSSkipped_ByComputeBudget_NonRATSubstrateNotPrimary",
                    "promotion_allowed": 0,
                }
            )
            continue
        for task in tasks:
            for seed in seeds:
                for loss in losses:
                    for method in methods:
                        result = train_fms_case(
                            family=family,
                            candidate_id=candidate_id,
                            method=method,
                            task=task,
                            seed=seed,
                            loss_interface=loss,
                            args=args,
                            device=device,
                        )
                        basis_raw.append(result["row"])
                        trace_rows.extend(result["trace_rows"])
                        grad_rows.extend(result["grad_rows"])
                        update_rows.extend(result["update_rows"])

    mlp_rows = enrich_results(mlp_raw)
    basis_rows = enrich_results(basis_raw)
    for r in mlp_rows + basis_rows:
        if sint(r.get("control_method"), 0) == 1:
            control_rows.append(
                {
                    "stage": "V142_CONTROLS",
                    "family": r.get("family"),
                    "task": r.get("task"),
                    "seed": r.get("seed"),
                    "loss_interface": r.get("loss_interface"),
                    "method": r.get("method"),
                    "NLL": r.get("NLL"),
                    "AUC_NLL": r.get("AUC_NLL"),
                    "LineC_majority_pass": r.get("LineC_majority_pass"),
                    "promotion_allowed": 0,
                }
            )
        linec_rows.append(
            {
                "stage": "V142_LINEC_AUDIT",
                "family": r.get("family"),
                "task": r.get("task"),
                "seed": r.get("seed"),
                "loss_interface": r.get("loss_interface"),
                "method": r.get("method"),
                "LineC_pass_rate": r.get("LineC_pass_rate"),
                "LineC_majority_pass": r.get("LineC_majority_pass"),
                "CEp99": r.get("CEp99"),
                "NLL": r.get("NLL"),
                "ECE": r.get("ECE"),
                "linec_tail_metrics_direction_used": 0,
                "promotion_allowed": 0,
            }
        )
    for r in mlp_rows + basis_rows:
        if sint(r.get("synthetic_gate_pass"), 0) == 0 and sint(r.get("control_method"), 0) == 0:
            failure_rows.append(
                {
                    "stage": "V142_FAILURE_TABLE",
                    "family": r.get("family"),
                    "candidate_id": r.get("candidate_id"),
                    "task": r.get("task"),
                    "loss_interface": r.get("loss_interface"),
                    "method": r.get("method"),
                    "failure_type": "SyntheticGateFail",
                    "source_vs_best_control": r.get("source_vs_best_control"),
                    "AUCtime_ratio_vs_best_control": r.get("AUCtime_ratio_vs_best_control"),
                    "CEp99_delta_vs_adamw": r.get("CEp99_delta_vs_adamw"),
                    "NLL_delta_vs_adamw": r.get("NLL_delta_vs_adamw"),
                    "ECE_delta_vs_adamw": r.get("ECE_delta_vs_adamw"),
                    "LineC_majority_pass": r.get("LineC_majority_pass"),
                    "promotion_allowed": 0,
                }
            )

    forbidden_count = 0
    write_rows(out_dir / "v142_project_progress.csv", progress_rows)
    write_rows(out_dir / "v142_code_path_manifest.csv", code_manifest())
    write_rows(out_dir / "v142_loss_interface_audit.csv", loss_audit(losses))
    write_rows(out_dir / "v142_functional_metric_state_trace.csv", trace_rows)
    write_rows(out_dir / "v142_per_example_gradient_stats.csv", grad_rows)
    write_rows(out_dir / "v142_fms_update_trace.csv", update_rows)
    write_rows(out_dir / "v142_mlp_fms_results.csv", mlp_rows)
    write_rows(out_dir / "v142_basis_substrate_status.csv", basis_status)
    write_rows(out_dir / "v142_basis_family_telemetry.csv", basis_telemetry)
    write_rows(out_dir / "v142_basis_fms_results.csv", basis_rows)
    write_rows(out_dir / "v142_linec_audit.csv", linec_rows)
    write_rows(out_dir / "v142_controls.csv", control_rows)
    write_rows(out_dir / "v142_failure_table.csv", failure_rows)

    route = build_route(
        out_dir=out_dir,
        basis_status=basis_status,
        mlp_rows=mlp_rows,
        basis_rows=basis_rows,
        missing_required_count=999,
        forbidden_count=forbidden_count,
        compute_budgeted_run=int(args.compute_budgeted_run),
    )
    write_json(out_dir / "v142_route_decision.json", route)
    write_text_artifacts(out_dir, route, basis_status, mlp_rows, basis_rows)
    write_rows(out_dir / "v142_required_manifest.csv", required_manifest(out_dir))
    missing = sum(1 for r in required_manifest(out_dir) if sint(r.get("exists"), 0) == 0)
    route = build_route(
        out_dir=out_dir,
        basis_status=basis_status,
        mlp_rows=mlp_rows,
        basis_rows=basis_rows,
        missing_required_count=missing,
        forbidden_count=forbidden_count,
        compute_budgeted_run=int(args.compute_budgeted_run),
    )
    write_json(out_dir / "v142_route_decision.json", route)
    write_rows(out_dir / "v142_required_manifest.csv", required_manifest(out_dir))
    packet(out_dir)
    return route


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DG-KAN v14.2 FunctionalFirst AllBasisParallel")
    p.add_argument("--out-dir", default=str(ROOT / "official_v142"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--global-seed", type=int, default=142)
    p.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    p.add_argument("--synthetic-seeds", default="0")
    p.add_argument("--loss-interfaces", default="CE,Brier")
    p.add_argument("--methods", default="F0-AdamWParallel,F1-PriorSNRReference,F2-ParameterFMS,F3-LayerFMS,F4-RoleFMS,F5-BasisGroupFMS,F7-PhaseScheduleFMS,FCTRL-RandomMatchedNorm")
    p.add_argument("--synthetic-train-size", type=int, default=96)
    p.add_argument("--synthetic-val-size", type=int, default=64)
    p.add_argument("--synthetic-dim", type=int, default=16)
    p.add_argument("--synthetic-classes", type=int, default=3)
    p.add_argument("--mlp-hidden", type=int, default=32)
    p.add_argument("--train-steps", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--fms-beta", type=float, default=0.90)
    p.add_argument("--fms-strength", type=float, default=0.25)
    p.add_argument("--fms-update-interval", type=int, default=1)
    p.add_argument("--trace-interval", type=int, default=12)
    p.add_argument("--max-trace-keys", type=int, default=8)
    p.add_argument("--compute-budgeted-run", type=int, default=1)
    p.add_argument("--skip-nonrat-fms", action="store_true")
    p.add_argument("--rational-candidate", default="", help="Override the default D-RAT candidate for focused Rational repair.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    route = run(args)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
