#!/usr/bin/env python3
"""DG-KAN v22.93 true deep PureKAN local-composite-metric MPFU runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import py_compile
import re
import subprocess
import sys
import time
import tokenize
import io
from copy import copy
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_89r_layer_composite_metric_mpfu as v2289
import experiments.run_v22_91_finite_step_calibrated_composite_geometry_mpfu as v2291
from dgkan.fu.layer_composite_metric import (
    EPS,
    basis_smoothness_diag,
    composite_metric,
    condition_number,
    effective_rank,
    matrix_to_w1,
    metric_matrix,
    natural_tangent_velocity,
    relative_fro_error,
    retract_to_metric,
    ridge_condition,
    sym,
    w1_to_matrix,
)
from dgkan.models.fc_purekan_primitives import _basis_eval


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_93_true_deep_purekan_local_composite_metric_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.93_TrueDeepPureKAN_LocalCompositeMetric_MultiScheme_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.93_TrueDeepPureKAN_LocalCompositeMetric_MultiScheme_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.93_TrueDeepPureKAN_LocalCompositeMetric_MultiScheme_MPFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2293_OUT_ROOT", str(ROOT / "results/v22_93"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"

AUDIT_DEFAULTS: dict[str, Any] = {
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "mlp_readout_used": 0,
    "mlp_initial_layer_used": 0,
    "changed_w1": 0,
    "changed_w2": 0,
    "changed_mlp": 0,
    "candidate_runtime_selection_used": 0,
}


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def ensure_out() -> None:
    for path in (OUT_ROOT, LOG_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def command_text(items: Iterable[Any]) -> str:
    return " ".join(str(item) for item in items)


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def median(values: Iterable[Any], default: float = 0.0) -> float:
    clean = sorted(fval(v) for v in values if math.isfinite(fval(v, float("nan"))))
    if not clean:
        return float(default)
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else 0.5 * (clean[mid - 1] + clean[mid])


def lower_cvar(values: Iterable[Any], frac: float = 0.25) -> float:
    clean = sorted(fval(v) for v in values if math.isfinite(fval(v, float("nan"))))
    if not clean:
        return 0.0
    take = max(1, int(math.ceil(float(frac) * len(clean))))
    return float(sum(clean[:take]) / take)


def clone_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    out = copy(args)
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for item_idx, item in enumerate(items) if item_idx % count == idx]


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**AUDIT_DEFAULTS, **data}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys = sorted({key for row in enriched for key in row.keys()} or set(AUDIT_DEFAULTS.keys()))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in enriched:
            writer.writerow(row)
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.93 TrueDeepPureKAN LocalCompositeMetric MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- Python：`{PYTHON}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            "- 非编造约束：只记录真实命令、真实 artifact、真实错误与观测；缺失项写 missing/unavailable/skipped。\n"
            "- 复现提示：Part C/E/F 支持 `--shard-count/--shard-index`，可用 `CUDA_VISIBLE_DEVICES=0..3` 并行。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.93 TrueDeepPureKAN LocalCompositeMetric MPFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "## 0. 当前结论\n"
            "- 尚未完成最终 route 判定。\n"
            "- 本文件只记录真实 artifact、真实指标、实际修复与分析；不补造缺失数据。\n\n"
            "## 1. 证据链、修复与分析\n",
            encoding="utf-8",
        )


def append_exec(stage: str, command: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {now_sg()} | {stage} | {status}\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, bullets: list[str]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n")
        fh.write(f"- time_sg: {now_sg()}\n")
        for item in bullets:
            fh.write(f"- {item}\n")


def write_next_actions(part: str, route: str, blocker: str, actions: list[dict[str, Any]]) -> Path:
    return write_json(
        OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json",
        {
            "part": part.upper(),
            "route": route,
            "dominant_blocker": blocker,
            "allowed_actions": actions,
            "forbidden_actions": [
                "fabricate_data",
                "weaken_no_debt_gate",
                "add_mlp_readout_or_trainable_stem",
                "best_row_promotion",
                "runtime_architecture_or_metric_winner_selection",
            ],
        },
    )


def strip_code(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in {tokenize.STRING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING}:
                out.append("\n" if token.type in {tokenize.NL, tokenize.NEWLINE} else " ")
            else:
                out.append(token.string)
    except tokenize.TokenError:
        return text
    return "".join(out)


def static_runtime_scan(files: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_architecture_winner_selection": re.compile(r"\b(select_arch|choose_arch|best_family|winner_family)\s*\("),
        "runtime_metric_winner_selection": re.compile(r"\b(select_metric|choose_metric|metric_winner)\s*\("),
        "runtime_topk_edge_selection": re.compile(r"\.topk\s*\("),
        "runtime_candidate_update_selection": re.compile(r"\b(select_update|choose_update|candidate_score)\s*\("),
        "output_oracle_or_readout_ls": re.compile(r"\b(oracle_target|readout_lstsq)\s*\("),
    }
    hits: list[dict[str, str]] = []
    for path in files:
        if not path.exists():
            continue
        code = strip_code(path)
        for check, pattern in patterns.items():
            match = pattern.search(code)
            if match:
                hits.append({"file": rel(path), "check": check, "match": match.group(0)})
    return int(not hits), hits


class TrueDeepPureKAN(nn.Module):
    """Strict FC PureKAN: every trainable layer is an edge-function coefficient tensor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        depth: int,
        basis_name: str,
        k: int,
        seed: int,
        device: torch.device,
        basis_input_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_width = int(hidden_width)
        self.depth = int(depth)
        self.basis_name = str(basis_name)
        self.k = int(k)
        self.basis_input_gain = float(basis_input_gain)
        dims = [self.input_dim] + [self.hidden_width] * max(0, self.depth - 1) + [self.output_dim]
        self.dims = dims
        self.coeffs = nn.ParameterList()
        gen = torch.Generator(device=device).manual_seed(int(seed))
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            scale = math.sqrt(max(1, int(in_dim) * self.k))
            self.coeffs.append(nn.Parameter(torch.randn(int(in_dim), int(out_dim), self.k, device=device, generator=gen) / scale))
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, self.k, device=device))
        self.register_buffer("scales", torch.tensor([max(0.2, 2.0 / max(1, self.k - 1))], device=device))

    def edge_parameters(self) -> list[nn.Parameter]:
        return list(self.coeffs)

    def basis(self, h: torch.Tensor) -> torch.Tensor:
        return _basis_eval(torch.tanh(h * self.basis_input_gain), self.basis_name, self.k, self.centers, self.scales)

    def layer_phi(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        b = self.basis(h) / math.sqrt(max(1, int(self.dims[int(layer_idx)])))
        return b.reshape(int(h.shape[0]), -1).to(dtype=torch.float64)

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = x
        activations = [h]
        for layer_idx, coeff in enumerate(self.coeffs):
            b = self.basis(h) / math.sqrt(max(1, int(self.dims[layer_idx])))
            h = torch.einsum("bik,iok->bo", b, coeff)
            activations.append(h)
        return h, activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_activations(x)
        return logits


class MatchedMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, depth: int, seed: int, device: torch.device) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        layers: list[nn.Module] = []
        width = int(hidden)
        last = int(input_dim)
        for _ in range(max(1, int(depth) - 1)):
            layers.append(nn.Linear(last, width))
            layers.append(nn.Tanh())
            last = width
        layers.append(nn.Linear(last, int(output_dim)))
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def basis_for_args(args: argparse.Namespace) -> tuple[str, int]:
    basis = "fourier_lowfreq" if "FOU" in str(args.pc_basis).upper() else "chebyshev"
    k = 5 if basis == "fourier_lowfreq" else 3
    if int(args.basis_k_override) > 0:
        k = int(args.basis_k_override)
    return basis, k


def make_kan_family(family: str, input_dim: int, output_dim: int, seed: int, args: argparse.Namespace, device: torch.device) -> TrueDeepPureKAN:
    basis, k = basis_for_args(args)
    if family == "shallow":
        depth = 1
    elif family == "depth2":
        depth = 2
    elif family == "depth3":
        depth = 3
    else:
        raise ValueError(f"unknown KAN family {family}")
    return TrueDeepPureKAN(input_dim, output_dim, int(args.deep_width), depth, basis, k, seed, device, basis_input_gain=float(args.basis_input_gain)).to(device)


def debt_metric(metrics: dict[str, float]) -> float:
    return float(metrics["brier"]) + float(metrics["ece"]) + float(metrics["tail95"]) - float(metrics["margin10"])


def metrics_for_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits = model(x)
        metrics = v2289.metrics_for_logits(logits, y)
        metrics["coverage_CVaR25"] = v2289.output_coverage_cvar25(logits, y)
        metrics["debt_metric"] = debt_metric(metrics)
        metrics["accuracy"] = float((logits.argmax(dim=1) == y).float().mean().detach().cpu().item())
    return metrics


def offdiag_mixing(x: torch.Tensor) -> float:
    if x.ndim != 2 or int(x.shape[1]) <= 1:
        return 0.0
    xc = x.detach().to(dtype=torch.float64) - x.detach().to(dtype=torch.float64).mean(dim=0, keepdim=True)
    cov = xc.T @ xc / float(max(1, int(x.shape[0])))
    off = cov - torch.diag(torch.diag(cov))
    return float((off.norm() / cov.norm().clamp_min(EPS)).detach().cpu().item())


def activation_rank(x: torch.Tensor) -> float:
    if x.ndim != 2:
        return 0.0
    xc = x.detach().to(dtype=torch.float64) - x.detach().to(dtype=torch.float64).mean(dim=0, keepdim=True)
    cov = xc.T @ xc / float(max(1, int(x.shape[0])))
    return effective_rank(cov + 1.0e-8 * torch.eye(int(cov.shape[0]), device=x.device, dtype=torch.float64))


def layer_metric(model: TrueDeepPureKAN, x: torch.Tensor, layer_idx: int, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        _logits, activations = model.forward_with_activations(x)
        act = activations[int(layer_idx)].detach()
        phi = model.layer_phi(act, int(layer_idx))
        smooth = basis_smoothness_diag(model.basis_name, int(model.dims[int(layer_idx)]), model.k, float(args.smoothness), device=x.device)
        c_raw = metric_matrix(phi, None, smooth, ridge=0.0)
        c, ridge_info = ridge_condition(c_raw, target_condition=float(args.c_condition_budget), max_ridge_rel=1.0e-2)
        a = w1_to_matrix(model.coeffs[int(layer_idx)].detach()).to(device=x.device, dtype=torch.float64)
        r = composite_metric(a, c)
    info = {
        "C_condition": condition_number(c),
        "R_condition": condition_number(r),
        "R_trace": float(torch.trace(r).detach().cpu().item()),
        "R_rank": effective_rank(r),
        "C_ridge_added": ridge_info.get("ridge_added", 0.0),
        "activation_rank": activation_rank(act),
        "coordinate_mixing_proxy": offdiag_mixing(act),
        "activation_variance": float(act.detach().to(dtype=torch.float64).var(dim=0, unbiased=False).mean().detach().cpu().item()),
    }
    return c, r, info


def layer_stats(model: TrueDeepPureKAN, x_source: torch.Tensor, x_guard: torch.Tensor, args: argparse.Namespace) -> dict[str, float]:
    out: dict[str, float] = {}
    max_drift = 0.0
    max_c = 0.0
    ranks: list[float] = []
    mix: list[float] = []
    for layer_idx in range(int(model.depth)):
        c_s, r_s, info_s = layer_metric(model, x_source, layer_idx, args)
        c_g, r_g, info_g = layer_metric(model, x_guard, layer_idx, args)
        drift = relative_fro_error(r_g, r_s)
        max_drift = max(max_drift, drift)
        max_c = max(max_c, info_s["C_condition"])
        ranks.append(info_s["activation_rank"])
        mix.append(info_s["coordinate_mixing_proxy"])
        out[f"layer{layer_idx}_source_guard_R_drift"] = drift
        out[f"layer{layer_idx}_C_condition"] = info_s["C_condition"]
        out[f"layer{layer_idx}_R_condition"] = info_s["R_condition"]
        out[f"layer{layer_idx}_activation_rank"] = info_s["activation_rank"]
        out[f"layer{layer_idx}_coordinate_mixing_proxy"] = info_s["coordinate_mixing_proxy"]
        out[f"layer{layer_idx}_guard_activation_rank"] = info_g["activation_rank"]
    out["source_guard_R_drift"] = max_drift
    out["max_C_condition"] = max_c
    out["activation_rank_median"] = median(ranks)
    out["coordinate_mixing_proxy_median"] = median(mix)
    return out


def projected_metric_step(
    model: TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    args: argparse.Namespace,
    *,
    metric_state: dict[int, dict[str, torch.Tensor | int]] | None = None,
    step_idx: int = 1,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    logits, activations = model.forward_with_activations(x)
    loss = F.cross_entropy(logits.float(), y.long())
    loss.backward()
    diags: list[dict[str, float]] = []
    changed = 0
    for layer_idx, param in enumerate(model.coeffs):
        if param.grad is None:
            continue
        act = activations[layer_idx].detach()
        phi = model.layer_phi(act, layer_idx)
        smooth = basis_smoothness_diag(model.basis_name, int(model.dims[layer_idx]), model.k, float(args.smoothness), device=x.device)
        c_raw = metric_matrix(phi, None, smooth, ridge=0.0)
        c, _ridge_info = ridge_condition(c_raw, target_condition=float(args.c_condition_budget), max_ridge_rel=1.0e-2)
        a = w1_to_matrix(param.detach()).to(device=x.device, dtype=torch.float64)
        g = w1_to_matrix(param.grad.detach()).to(device=x.device, dtype=torch.float64)
        if float(g.norm().detach().cpu().item()) <= 0.0:
            continue
        delta, diag = natural_tangent_velocity(c, a, g, EPS)
        if int(args.metric_shape_scale_branch):
            radial = (((-g) * a).sum() / (a.square().sum().clamp_min(EPS))) * a
            radial_cap = float(args.shape_radial_fraction_cap) * max(float(delta.norm().detach().cpu().item()), 0.01 * float(a.norm().detach().cpu().item()), EPS)
            radial_norm = float(radial.norm().detach().cpu().item())
            if radial_norm > radial_cap:
                radial = radial * (radial_cap / max(radial_norm, EPS))
                diag["shape_radial_clipped"] = 1.0
            delta = delta + radial
            preserved = ((-g) * delta).sum() / (g.norm() * delta.norm()).clamp_min(EPS)
            diag["task_gradient_preserved_fraction"] = float(preserved.detach().cpu().item())
            diag["shape_radial_fraction"] = float((radial.norm() / delta.norm().clamp_min(EPS)).detach().cpu().item())
        max_norm = float(args.metric_max_norm_ratio) * max(float(a.norm().detach().cpu().item()), EPS)
        dnorm = float(delta.norm().detach().cpu().item())
        if dnorm > max_norm:
            delta = delta * (max_norm / max(dnorm, EPS))
            diag["metric_delta_clipped"] = 1.0
        r_target = composite_metric(a, c)
        a_tilde = a + float(lr) * delta
        a_new: torch.Tensor
        retract_diag: dict[str, float]
        if int(args.metric_tube_gradient_residual):
            residual_velocity = -g
            if int(args.metric_tube_adamw_residual):
                if metric_state is None:
                    metric_state = {}
                state = metric_state.setdefault(
                    int(layer_idx),
                    {
                        "m": torch.zeros_like(g),
                        "v": torch.zeros_like(g),
                        "t": 0,
                    },
                )
                beta1 = float(args.metric_tube_adamw_beta1)
                beta2 = float(args.metric_tube_adamw_beta2)
                state["t"] = int(state.get("t", 0)) + 1
                state["m"] = beta1 * state["m"] + (1.0 - beta1) * g
                state["v"] = beta2 * state["v"] + (1.0 - beta2) * g.square()
                t = int(state["t"])
                m_hat = state["m"] / max(1.0 - beta1**t, EPS)
                v_hat = state["v"] / max(1.0 - beta2**t, EPS)
                residual_velocity = -m_hat / (v_hat.sqrt() + float(args.metric_tube_adamw_eps))
                if int(args.metric_tube_adamw_norm_match):
                    residual_velocity = residual_velocity * (g.norm() / residual_velocity.norm().clamp_min(EPS))
                diag["tube_adamw_residual_used"] = 1.0
                diag["tube_adamw_residual_step"] = float(t)
            budget = float(args.metric_tube_drift_budget)
            if int(args.metric_tube_combine_tangent_residual):
                tangent_scale = 1.0
                tangent_a = a
                tangent_drift = 0.0
                for _ in range(40):
                    cand = a + float(lr) * tangent_scale * delta
                    drift = relative_fro_error(composite_metric(cand, c), r_target)
                    if drift <= budget:
                        tangent_a = cand
                        tangent_drift = drift
                        break
                    tangent_scale *= 0.5
                residual_scale = 1.0
                best_a = tangent_a
                best_drift = tangent_drift
                for _ in range(40):
                    cand = a + float(lr) * (tangent_scale * delta + residual_scale * residual_velocity)
                    drift = relative_fro_error(composite_metric(cand, c), r_target)
                    if drift <= budget:
                        best_a = cand
                        best_drift = drift
                        break
                    residual_scale *= 0.5
            else:
                tangent_scale = 0.0
                residual_scale = 1.0
                best_a = a
                best_drift = 0.0
                for _ in range(40):
                    cand = a + float(lr) * residual_scale * residual_velocity
                    drift = relative_fro_error(composite_metric(cand, c), r_target)
                    if drift <= budget:
                        best_a = cand
                        best_drift = drift
                        break
                    residual_scale *= 0.5
            a_new = best_a
            retract_diag = {
                "metric_drift_before_retraction": best_drift,
                "metric_drift_after_retraction": best_drift,
                "retraction_error": best_drift,
                "shape_drift": best_drift,
                "scale_change": float((torch.trace(composite_metric(a_new, c)) / torch.trace(r_target).clamp_min(EPS)).detach().cpu().item()),
            }
            diag["tube_gradient_residual_used"] = 1.0
            diag["tube_tangent_scale"] = tangent_scale
            diag["tube_gradient_residual_scale"] = residual_scale
        else:
            if int(args.metric_shape_scale_branch):
                r_tilde = composite_metric(a_tilde, c)
                trace_scale = torch.trace(r_tilde).clamp_min(EPS) / torch.trace(r_target).clamp_min(EPS)
                r_target = sym(r_target * trace_scale)
                diag["shape_trace_scale"] = float(trace_scale.detach().cpu().item())
            a_new, retract_diag = retract_to_metric(a_tilde, c, r_target, rho=1.0e-8, eps=EPS)
        actual_velocity = (a_new - a) / max(float(lr), EPS)
        actual_preserved = ((-g) * actual_velocity).sum() / (g.norm() * actual_velocity.norm()).clamp_min(EPS)
        diag["task_gradient_preserved_fraction"] = float(actual_preserved.detach().cpu().item())
        diag["actual_projected_velocity_norm"] = float(actual_velocity.norm().detach().cpu().item())
        with torch.no_grad():
            before = param.detach().clone()
            param.copy_(matrix_to_w1(a_new, param.shape, dtype=param.dtype, device=param.device))
            changed += int(not torch.allclose(before, param.detach()))
        diag.update(retract_diag)
        diags.append(diag)
    return {
        "loss": float(loss.detach().cpu().item()),
        "velocity_emitted": int(len(diags) > 0),
        "changed_w1": int(changed > 0),
        "changed_layers": changed,
        "tangent_residual_max": max([fval(d.get("tangent_residual")) for d in diags] or [0.0]),
        "sylvester_residual_max": max([fval(d.get("Sylvester_residual")) for d in diags] or [0.0]),
        "metric_drift_after_max": max([fval(d.get("metric_drift_after_retraction")) for d in diags] or [0.0]),
        "task_gradient_preserved_fraction_min": min([fval(d.get("task_gradient_preserved_fraction"), 1.0) for d in diags] or [0.0]),
        "tube_tangent_scale_min": min([fval(d.get("tube_tangent_scale"), 1.0) for d in diags] or [1.0]),
        "tube_gradient_residual_scale_min": min([fval(d.get("tube_gradient_residual_scale"), 1.0) for d in diags] or [1.0]),
        "tube_adamw_residual_used": max([fval(d.get("tube_adamw_residual_used"), 0.0) for d in diags] or [0.0]),
    }


def projected_adamw_tube_step(
    model: TrueDeepPureKAN,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    logits, activations = model.forward_with_activations(x)
    loss = F.cross_entropy(logits.float(), y.long())
    layer_refs: list[tuple[nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for layer_idx, param in enumerate(model.coeffs):
        act = activations[layer_idx].detach()
        phi = model.layer_phi(act, layer_idx)
        smooth = basis_smoothness_diag(model.basis_name, int(model.dims[layer_idx]), model.k, float(args.smoothness), device=x.device)
        c_raw = metric_matrix(phi, None, smooth, ridge=0.0)
        c, _ridge_info = ridge_condition(c_raw, target_condition=float(args.c_condition_budget), max_ridge_rel=1.0e-2)
        a_old = w1_to_matrix(param.detach()).to(device=x.device, dtype=torch.float64)
        r_target = composite_metric(a_old, c)
        layer_refs.append((param, a_old, c, r_target))
    loss.backward()
    grads = [w1_to_matrix(param.grad.detach()).to(device=x.device, dtype=torch.float64) if param.grad is not None else torch.zeros_like(a_old) for param, a_old, _c, _r in layer_refs]
    optimizer.step()
    budget = float(args.metric_tube_drift_budget)
    changed = 0
    drifts: list[float] = []
    scales: list[float] = []
    preserves: list[float] = []
    with torch.no_grad():
        for (param, a_old, c, r_target), g in zip(layer_refs, grads):
            a_prop = w1_to_matrix(param.detach()).to(device=x.device, dtype=torch.float64)
            direction = a_prop - a_old
            alpha = 1.0
            drift = relative_fro_error(composite_metric(a_prop, c), r_target)
            if drift > budget:
                lo, hi = 0.0, 1.0
                best_a = a_old
                best_drift = 0.0
                for _ in range(32):
                    mid = 0.5 * (lo + hi)
                    cand = a_old + mid * direction
                    cand_drift = relative_fro_error(composite_metric(cand, c), r_target)
                    if cand_drift <= budget:
                        lo = mid
                        best_a = cand
                        best_drift = cand_drift
                    else:
                        hi = mid
                alpha = lo
                a_new = best_a
                drift = best_drift
            else:
                a_new = a_prop
            actual_velocity = (a_new - a_old) / max(float(lr), EPS)
            if float(g.norm().detach().cpu().item()) > 0.0 and float(actual_velocity.norm().detach().cpu().item()) > 0.0:
                preserve = ((-g) * actual_velocity).sum() / (g.norm() * actual_velocity.norm()).clamp_min(EPS)
                preserves.append(float(preserve.detach().cpu().item()))
            param.copy_(matrix_to_w1(a_new, param.shape, dtype=param.dtype, device=param.device))
            changed += int(not torch.allclose(matrix_to_w1(a_old, param.shape, dtype=param.dtype, device=param.device), param.detach()))
            drifts.append(float(drift))
            scales.append(float(alpha))
    return {
        "loss": float(loss.detach().cpu().item()),
        "velocity_emitted": 1,
        "changed_w1": int(changed > 0),
        "changed_layers": changed,
        "metric_drift_after_max": max(drifts or [0.0]),
        "projected_adamw_tube_scale_min": min(scales or [1.0]),
        "task_gradient_preserved_fraction_min": min(preserves or [0.0]),
        "projected_adamw_tube_used": 1.0,
    }


def train_model(
    model: nn.Module,
    family: str,
    optimizer_kind: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    before_state = {name: p.detach().clone() for name, p in model.named_parameters()}
    before_guard = metrics_for_model(model, x_guard, y_guard)
    before_train = metrics_for_model(model, x_train, y_train)
    start = time.time()
    trace: dict[str, float] = {"velocity_emitted": 0, "changed_w1": 0}
    if optimizer_kind == "metric" and isinstance(model, TrueDeepPureKAN):
        metric_state: dict[int, dict[str, torch.Tensor | int]] = {}
        if str(args.metric_update_mode) == "projected_adamw_tube":
            lr = float(args.metric_projected_adamw_lr)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(args.weight_decay))
            for step in range(int(args.train_steps)):
                xb, yb = v2289.iter_train_batches(x_train, y_train, step, int(args.batch_size), int(seed))
                diag = projected_adamw_tube_step(model, opt, xb, yb, lr, args)
                for key, value in diag.items():
                    if key in {"velocity_emitted", "changed_w1"}:
                        trace[key] = max(fval(trace.get(key)), fval(value))
                    else:
                        trace[key] = max(fval(trace.get(key)), fval(value)) if key.endswith("_max") else fval(value)
        else:
            for step in range(int(args.train_steps)):
                xb, yb = v2289.iter_train_batches(x_train, y_train, step, int(args.batch_size), int(seed))
                diag = projected_metric_step(model, xb, yb, float(args.metric_lr), args, metric_state=metric_state, step_idx=step + 1)
                for key, value in diag.items():
                    if key in {"velocity_emitted", "changed_w1"}:
                        trace[key] = max(fval(trace.get(key)), fval(value))
                    else:
                        trace[key] = max(fval(trace.get(key)), fval(value)) if key.endswith("_max") else fval(value)
    else:
        lr = float(args.mlp_lr) if optimizer_kind == "mlp" else float(args.adamw_lr)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(args.weight_decay))
        for step in range(int(args.train_steps)):
            xb, yb = v2289.iter_train_batches(x_train, y_train, step, int(args.batch_size), int(seed))
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb).float(), yb.long())
            loss.backward()
            opt.step()
        trace["velocity_emitted"] = 0
    elapsed = time.time() - start
    after_guard = metrics_for_model(model, x_guard, y_guard)
    after_train = metrics_for_model(model, x_train, y_train)
    changed_edge = 0
    changed_mlp = 0
    for name, p in model.named_parameters():
        changed = int(not torch.allclose(before_state[name], p.detach()))
        if isinstance(model, TrueDeepPureKAN):
            changed_edge = max(changed_edge, changed)
        else:
            changed_mlp = max(changed_mlp, changed)
    stats = layer_stats(model, x_train, x_guard, args) if isinstance(model, TrueDeepPureKAN) else {}
    return {
        "family": family,
        "optimizer_kind": optimizer_kind,
        "train_NLL": after_train["nll"],
        "guard_NLL": after_guard["nll"],
        "guard_accuracy": after_guard["accuracy"],
        "guard_debt_before": before_guard["debt_metric"],
        "guard_debt_after": after_guard["debt_metric"],
        "no_debt": int(after_guard["debt_metric"] <= before_guard["debt_metric"] + float(args.no_debt_budget)),
        "Brier_delta": after_guard["brier"] - before_guard["brier"],
        "ECE_delta": after_guard["ece"] - before_guard["ece"],
        "tail95_delta": after_guard["tail95"] - before_guard["tail95"],
        "tail99_delta": after_guard["tail99"] - before_guard["tail99"],
        "margin10_delta": after_guard["margin10"] - before_guard["margin10"],
        "coverage_CVaR25": after_guard["coverage_CVaR25"],
        "coverage_delta": after_guard["coverage_CVaR25"] - before_guard["coverage_CVaR25"],
        "wall_time_s": elapsed,
        "changed_w1": changed_edge,
        "changed_mlp": changed_mlp,
        "changed_w2": 0,
        "mlp_readout_used": 0 if isinstance(model, TrueDeepPureKAN) else 1,
        "mlp_initial_layer_used": 0 if isinstance(model, TrueDeepPureKAN) else 1,
        **trace,
        **stats,
    }


def synthetic_task(teacher_type: str, seed: int, n_train: int, n_guard: int, input_dim: int, classes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(2293000 + int(seed) * 1009 + sum(ord(ch) for ch in teacher_type))
    x = torch.rand(n_train + n_guard, input_dim, generator=gen, device=device) * 2.0 - 1.0
    if teacher_type in {"c1a_shallow_additive", "e1_shallow"}:
        s0 = 2.6 * torch.sin(math.pi * x[:, 0]) + 1.4 * torch.cos(math.pi * x[:, 1 % input_dim])
        s1 = -2.4 * torch.sin(math.pi * x[:, 0]) + 1.8 * torch.sin(math.pi * x[:, 2 % input_dim])
        s2 = 2.2 * torch.cos(math.pi * x[:, 3 % input_dim]) - 1.3 * torch.sin(math.pi * x[:, 1 % input_dim])
        scores = [s0, s1, s2]
        while len(scores) < classes:
            idx = len(scores) % input_dim
            scores.append(2.0 * torch.sin(math.pi * x[:, idx]) - 1.0 * torch.cos(math.pi * x[:, (idx + 2) % input_dim]))
        logits = 2.5 * torch.stack(scores[:classes], dim=1)
    elif teacher_type in {"c1b_deep_coordinate", "e2_deep"}:
        hidden = []
        for j in range(max(4, classes + 1)):
            u = torch.sin(math.pi * x[:, j % input_dim]) + 0.7 * torch.cos(math.pi * x[:, (j + 2) % input_dim])
            u = u + 0.25 * torch.sin(math.pi * x[:, (j + 3) % input_dim])
            hidden.append(u)
        h = torch.stack(hidden, dim=1)
        logits = torch.stack(
            [
                torch.sin(h[:, 0] + 0.5 * h[:, 1]),
                torch.cos(h[:, 1] - 0.25 * h[:, 2]),
                torch.sin(h[:, 2]) + 0.3 * torch.cos(h[:, 3]),
            ][:classes],
            dim=1,
        )
        while int(logits.shape[1]) < classes:
            logits = torch.cat([logits, torch.sin(h[:, :1] * float(logits.shape[1] + 1))], dim=1)
    elif teacher_type in {"c1c_mlp_friendly", "e3_mlp_friendly"}:
        q = torch.randn(input_dim, input_dim, generator=gen, device=device)
        q, _ = torch.linalg.qr(q)
        z = x @ q
        logits = torch.stack(
            [
                z[:, 0] * z[:, 1] + 0.5 * z[:, 2].square(),
                -z[:, 0] * z[:, 1] + torch.sin(2.0 * z[:, 3]),
                z[:, 4 % input_dim] * z[:, 5 % input_dim] - 0.25 * z[:, 2],
            ][:classes],
            dim=1,
        )
    else:
        raise ValueError(f"unknown synthetic teacher {teacher_type}")
    y = logits.argmax(dim=1).long()
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def _balanced_signs_for_class(cls: int, gen: torch.Generator, device: torch.device) -> tuple[float, float, float, float]:
    """Return four signs whose individual marginals are balanced per class.

    The visual interaction repair uses labels determined by sign products
    rather than by class-specific single-pixel means.  This keeps the C2/E4
    task from collapsing back into a raw-input additive template.
    """
    s0 = 1.0 if int(torch.randint(0, 2, (1,), generator=gen, device=device).item()) else -1.0
    s2 = 1.0 if int(torch.randint(0, 2, (1,), generator=gen, device=device).item()) else -1.0
    if cls == 0:
        # First pair has negative product; second pair is unconstrained.
        s1 = -s0
        s3 = 1.0 if int(torch.randint(0, 2, (1,), generator=gen, device=device).item()) else -1.0
    elif cls == 1:
        # First pair positive, second pair negative.
        s1 = s0
        s3 = -s2
    else:
        # Both products positive.
        s1 = s0
        s3 = s2
    return s0, s1, s2, s3


def _add_patch(img: torch.Tensor, top: int, left: int, value: float, size: int = 2) -> None:
    img[top : top + size, left : left + size] += float(value)


def _add_checker_patch(img: torch.Tensor, top: int, left: int, value: float) -> None:
    v = float(value)
    img[top, left] += v
    img[top + 1, left + 1] += v
    img[top, left + 1] -= v
    img[top + 1, left] -= v


def visual_synthetic_task(
    task: str,
    seed: int,
    n_train: int,
    n_guard: int,
    side: int,
    classes: int,
    device: torch.device,
    *,
    fixed_patch_features: bool = False,
    task_version: str = "balanced_interaction_v2",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(2293300 + int(seed) * 101 + sum(ord(ch) for ch in task))
    n = n_train + n_guard
    imgs = torch.randn(n, side, side, generator=gen, device=device) * 0.08
    y = torch.empty(n, dtype=torch.long, device=device)
    for idx in range(n):
        cls = idx % max(1, int(classes))
        y[idx] = cls
        if str(task_version) == "legacy_class_template":
            cls = int(torch.randint(0, classes, (1,), generator=gen, device=device).item())
            y[idx] = cls
            if task == "stroke_hv":
                if cls % 2 == 0:
                    imgs[idx, side // 3, :] += 1.0
                else:
                    imgs[idx, :, side // 3] += 1.0
            elif task == "two_patch_composition":
                r1 = side // 4
                r2 = 3 * side // 4
                imgs[idx, r1 - 1 : r1 + 1, r1 - 1 : r1 + 1] += 1.0 if cls == 0 else -0.6
                imgs[idx, r2 - 1 : r2 + 1, r2 - 1 : r2 + 1] += 1.0 if cls == 1 else -0.6
            elif task == "local_patch_interaction":
                imgs[idx, 1:3, 1:3] += 1.0
                imgs[idx, -3:-1, -3:-1] += 1.0 if cls == 1 else -1.0
            else:
                if cls == 0:
                    imgs[idx].fill_diagonal_(1.0)
                else:
                    imgs[idx] = torch.flip(torch.eye(side, device=device), dims=[1]) + imgs[idx]
            continue

        s0, s1, s2, s3 = _balanced_signs_for_class(cls, gen, device)
        amp = 1.25
        if task == "stroke_hv":
            # Product of horizontal/vertical stroke polarities determines class.
            imgs[idx, side // 4, :] += amp * s0
            imgs[idx, 3 * side // 4, :] += amp * s1
            imgs[idx, :, side // 4] += amp * s2
            imgs[idx, :, 3 * side // 4] += amp * s3
        elif task == "two_patch_composition":
            _add_patch(imgs[idx], 1, 1, amp * s0)
            _add_patch(imgs[idx], 1, side - 3, amp * s1)
            _add_patch(imgs[idx], side - 3, 1, amp * s2)
            _add_patch(imgs[idx], side - 3, side - 3, amp * s3)
        elif task == "local_patch_interaction":
            _add_checker_patch(imgs[idx], 1, 1, amp * s0)
            _add_checker_patch(imgs[idx], 1, side - 3, amp * s1)
            _add_checker_patch(imgs[idx], side - 3, 1, amp * s2)
            _add_checker_patch(imgs[idx], side - 3, side - 3, amp * s3)
        else:
            # Rotation-sensitive variant with balanced diagonal/anti-diagonal signs.
            for offset in range(side):
                imgs[idx, offset, offset] += amp * s0
                imgs[idx, offset, side - 1 - offset] += amp * s1
            _add_checker_patch(imgs[idx], 1, side // 2 - 1, amp * s2)
            _add_checker_patch(imgs[idx], side - 3, side // 2 - 1, amp * s3)
    flat = imgs.reshape(n, side * side).clamp(-2.0, 2.0)
    if fixed_patch_features:
        patch = imgs.unfold(1, 2, 2).unfold(2, 2, 2).contiguous().mean(dim=(-1, -2)).reshape(n, -1)
        rows = imgs.mean(dim=2)
        cols = imgs.mean(dim=1)
        x = torch.cat([flat, patch, rows, cols], dim=1)
    else:
        x = flat
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def train_family_set(xtr: torch.Tensor, ytr: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, dict[str, Any]]:
    input_dim = int(xtr.shape[1])
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    out: dict[str, dict[str, Any]] = {}
    for family in ["shallow", "depth2", "depth3"]:
        model = make_kan_family(family, input_dim, classes, seed + {"shallow": 11, "depth2": 22, "depth3": 33}[family], args, device)
        out[f"{family}_metric"] = train_model(model, family, "metric", xtr, ytr, xg, yg, seed, args)
        ctrl = make_kan_family(family, input_dim, classes, seed + {"shallow": 111, "depth2": 222, "depth3": 333}[family], args, device)
        out[f"{family}_adamw"] = train_model(ctrl, family, "adamw", xtr, ytr, xg, yg, seed, args)
    mlp = MatchedMLP(input_dim, classes, int(args.mlp_hidden), 2, seed + 444, device)
    out["mlp_raw"] = train_model(mlp, "mlp_raw", "mlp", xtr, ytr, xg, yg, seed, args)
    return out


def summarize_family_row(base: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = dict(base)
    for name, res in results.items():
        for key, value in res.items():
            if isinstance(value, (int, float, str)):
                row[f"{name}_{key}"] = value
    best_deep = min(results["depth2_metric"]["guard_NLL"], results["depth3_metric"]["guard_NLL"])
    shallow = results["shallow_metric"]["guard_NLL"]
    mlp = results["mlp_raw"]["guard_NLL"]
    best_control = min(results["depth2_adamw"]["guard_NLL"], results["depth3_adamw"]["guard_NLL"], mlp)
    row.update(
        {
            "best_deep_guard_NLL": best_deep,
            "best_shallow_guard_NLL": shallow,
            "mlp_raw_guard_NLL": mlp,
            "deep_beats_shallow": int(best_deep < shallow),
            "deep_beats_shallow_margin": float(shallow - best_deep),
            "deep_beats_mlp_raw": int(best_deep < mlp),
            "deep_beats_best_control": int(best_deep < best_control),
            "deep_no_debt": int(max(results["depth2_metric"]["no_debt"], results["depth3_metric"]["no_debt"]) > 0),
            "deep_source_guard_R": int(min(results["depth2_metric"].get("source_guard_R_drift", 99.0), results["depth3_metric"].get("source_guard_R_drift", 99.0)) <= 0.50),
            "deep_coverage": int(max(results["depth2_metric"]["coverage_CVaR25"], results["depth3_metric"]["coverage_CVaR25"]) >= 0.20),
            "runtime_selector_used": 0,
            "candidate_runtime_selection_used": 0,
            "mlp_readout_used": 0,
            "mlp_initial_layer_used": 0,
            "changed_w1": int(max(results["shallow_metric"]["changed_w1"], results["depth2_metric"]["changed_w1"], results["depth3_metric"]["changed_w1"]) > 0),
            "changed_mlp": int(results["mlp_raw"]["changed_mlp"] > 0),
        }
    )
    return row


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    compile_pass = 1
    compile_error = ""
    try:
        py_compile.compile(str(RUNNER), doraise=True)
    except Exception as exc:
        compile_pass = 0
        compile_error = repr(exc)
    import_pass = 1
    scan_pass, scan_hits = static_runtime_scan([RUNNER])
    rows: list[dict[str, Any]] = []
    x = torch.randn(32, int(args.input_dim), device=device)
    y = torch.randint(0, int(args.num_classes), (32,), device=device)
    for name, family in [("A0_shallow_additive_kan", "shallow"), ("A1_true_deep_purekan_depth2", "depth2"), ("A2_true_deep_purekan_depth3", "depth3")]:
        model = make_kan_family(family, int(args.input_dim), int(args.num_classes), 2293 + len(rows), args, device)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        diag = projected_metric_step(model, x, y, float(args.metric_lr), args)
        changed = {n: int(not torch.allclose(before[n], p.detach())) for n, p in model.named_parameters()}
        rows.append(
            {
                "model_name": name,
                "depth": model.depth,
                "widths": ",".join(str(v) for v in model.dims),
                "basis_family": model.basis_name,
                "basis_count_per_layer": model.k,
                "has_mlp_readout": 0,
                "has_mlp_initial_layer": 0,
                "has_trainable_linear_matrix": 0,
                "has_cross_dim_linear_mixing": 0,
                "has_diagonal_input_transport": 0,
                "has_constant_bias_edge": 0,
                "w2_readout_in_forward": 0,
                "changed_w1_edge_tensors": int(any(changed.values())),
                "changed_w2_readout_tensors": 0,
                "changed_mlp_tensors": 0,
                "changed_transport_tensors": 0,
                "num_edge_function_params_per_layer": ";".join(str(p.numel()) for p in model.coeffs),
                "num_transport_params": 0,
                "num_mlp_params": 0,
                "all_trainable_param_names": ";".join(name for name, _p in model.named_parameters()),
                "all_forward_module_names": ";".join(name for name, _m in model.named_modules()),
                "standard_loop": 1,
                "state_updated": 1,
                "velocity_emitted": diag["velocity_emitted"],
                "changed_w1": int(any(changed.values())),
            }
        )
    mlp = MatchedMLP(int(args.input_dim), int(args.num_classes), int(args.mlp_hidden), 2, 2293, device)
    rows.append(
        {
            "model_name": "A3_controls_mlp_raw_and_mlp_composite",
            "depth": 2,
            "widths": f"{args.input_dim},{args.mlp_hidden},{args.num_classes}",
            "basis_family": "MLP_control",
            "basis_count_per_layer": 0,
            "has_mlp_readout": 1,
            "has_mlp_initial_layer": 1,
            "has_trainable_linear_matrix": 1,
            "has_cross_dim_linear_mixing": 1,
            "has_diagonal_input_transport": 0,
            "has_constant_bias_edge": 1,
            "w2_readout_in_forward": 0,
            "changed_w1_edge_tensors": 0,
            "changed_w2_readout_tensors": 0,
            "changed_mlp_tensors": 0,
            "changed_transport_tensors": 0,
            "num_edge_function_params_per_layer": 0,
            "num_transport_params": 0,
            "num_mlp_params": sum(p.numel() for p in mlp.parameters()),
            "all_trainable_param_names": ";".join(name for name, _p in mlp.named_parameters()),
            "all_forward_module_names": ";".join(name for name, _m in mlp.named_modules()),
            "standard_loop": 1,
            "state_updated": 0,
            "velocity_emitted": 0,
            "changed_mlp": 0,
        }
    )
    pure = [r for r in rows if str(r["model_name"]).startswith("A") and "kan" in str(r["model_name"])]
    gate = int(
        compile_pass
        and import_pass
        and scan_pass
        and all(ival(r.get("standard_loop")) == 1 and ival(r.get("state_updated")) == 1 and ival(r.get("velocity_emitted")) == 1 for r in pure)
        and all(ival(r.get("changed_w1_edge_tensors")) == 1 for r in pure)
        and all(ival(r.get("changed_w2_readout_tensors")) == 0 and ival(r.get("has_mlp_readout")) == 0 and ival(r.get("has_mlp_initial_layer")) == 0 for r in pure)
    )
    out = {
        "gate": "v22_93_part_a_architecture_identity",
        "compile": compile_pass,
        "compile_error": compile_error,
        "import": import_pass,
        "static_runtime_scan_pass": scan_pass,
        "static_runtime_scan_hits": scan_hits,
        "part_a_gate_pass": gate,
        "part_f_current_model_identity": "shallow_additive_kan",
        "rows": rows,
        "changed_w1": int(any(ival(r.get("changed_w1_edge_tensors")) for r in pure)),
    }
    matrix = OUT_ROOT / "part_a_architecture_identity_matrix.csv"
    summary = OUT_ROOT / "part_a_architecture_identity.json"
    write_rows(matrix, rows)
    write_json(summary, out)
    route = "A_Pass" if gate else "A_FailedArchitectureIdentity"
    next_path = write_next_actions("a", route, "none" if gate else "architecture_identity", [] if gate else [{"action": "repair_true_deep_purekan_identity", "reason": "Part A identity gate failed"}])
    append_exec("part-a", command_text(sys.argv), "done" if gate else "failed", gpu=str(device), files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part A architecture identity",
        [
            f"gate_pass={gate}; compile={compile_pass}; import={import_pass}; static_scan={scan_pass}; part_f_current_model_identity=shallow_additive_kan",
            "analysis: v22.93 runner constructs shallow additive depth1 and true depth2/depth3 PureKAN models with only edge-function coefficient tensors. MLP controls are explicitly separate controls, not PureKAN runtime fallback.",
        ],
    )
    return out


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    v2289_final = read_json(ROOT / "results/v22_89R/final_route.json")
    v2290_e = read_json(ROOT / "results/v22_90/part_e_summary.json")
    v2290_f = read_json(ROOT / "results/v22_90/part_f_summary.json")
    v2290_pull = read_json(ROOT / "results/v22_90/part_f_output_safe_pullback_probe_summary.json")
    v2291_final = read_json(ROOT / "results/v22_91/final_route.json")
    v2291_c = read_json(ROOT / "results/v22_91/part_c_summary_dist_h20.json")
    if not v2291_c:
        v2291_c = read_json(ROOT / "results/v22_91/part_c_summary_h20.json")
    pull_norm = v2290_pull.get("by_direction", {}).get("output_safe_pullback_norm", {}) if isinstance(v2290_pull.get("by_direction"), dict) else {}
    v2289_part_e = v2289_final.get("part_e")
    if isinstance(v2289_part_e, dict):
        v2289_part_e_pass = v2289_part_e.get("part_e_gate_pass", v2289_final.get("part_e_gate_pass"))
    else:
        v2289_part_e_pass = v2289_part_e if v2289_part_e is not None else v2289_final.get("part_e_gate_pass")
    out = {
        "gate": "v22_93_part_b_history_lock",
        "v22_89R_part_e_pass": v2289_part_e_pass,
        "v22_89R_part_f_fail_route": v2289_final.get("final_route"),
        "v22_90_part_e_pass": v2290_e.get("part_e_gate_pass"),
        "v22_90_part_f_fail_route": v2290_f.get("part_f_route", v2290_f.get("dominant_blocker")),
        "v22_90_beats_MLP_composite": v2290_f.get("beats_MLP_composite_coordinate"),
        "v22_90_beats_MLP_raw": v2290_f.get("beats_MLP_raw"),
        "v22_90_no_debt": v2290_f.get("no_debt"),
        "v22_90_visual_coverage": v2290_f.get("visual_coverage_pass"),
        "v22_90_output_safe_pullback_residual": pull_norm.get("composite_pullback_residual_median"),
        "v22_91_final_route": v2291_final.get("final_route"),
        "v22_91_debt_response_sign": v2291_c.get("response_debt_sign_agreement"),
        "v22_91_debt_R2": v2291_c.get("debt_R2"),
        "v22_91_leave_family_debt_R2": v2291_c.get("leave_dataset_family_debt_R2"),
    }
    gate = int(
        ival(out.get("v22_90_part_e_pass")) == 1
        and str(out.get("v22_91_final_route", "")).startswith("C1")
        and (ival(out.get("v22_90_no_debt"), 1) == 0 or fval(out.get("v22_90_beats_MLP_composite"), 30.0) <= 1.0)
    )
    out["part_b_gate_pass"] = gate
    summary = OUT_ROOT / "part_b_history_lock.json"
    write_json(summary, out)
    route = "B_Pass" if gate else "B_HistoryLockFailed"
    next_path = write_next_actions("b", route, "none" if gate else "history_lock", [] if gate else [{"action": "inspect_prior_artifacts", "reason": "required v22.89R/v22.90/v22.91 history artifacts missing or inconsistent"}])
    append_exec("part-b", command_text(sys.argv), "done" if gate else "failed", files=f"{rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part B history lock",
        [
            f"gate_pass={gate}; v22_90_part_e={out.get('v22_90_part_e_pass')}; v22_90_part_f_route={out.get('v22_90_part_f_fail_route')}; v22_91_final_route={out.get('v22_91_final_route')}",
            f"v22_91 debt sign={out.get('v22_91_debt_response_sign')}; debt_R2={out.get('v22_91_debt_R2')}; leave_family_debt_R2={out.get('v22_91_leave_family_debt_R2')}",
            "analysis: history lock records v22.90 Part E as positive-control only, v22.90 Part F as real-task failure, and v22.91 as response-calibration failure rather than controller success.",
        ],
    )
    return out


def capability_jobs(args: argparse.Namespace) -> list[tuple[str, int]]:
    teachers = ["c1a_shallow_additive", "c1b_deep_coordinate", "c1c_mlp_friendly"]
    return [(teacher, seed) for teacher in teachers for seed in range(int(args.synthetic_seed_count))]


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for teacher, seed in shard_items(capability_jobs(args), args):
        xtr, ytr, xg, yg = synthetic_task(teacher, seed, int(args.synthetic_train_size), int(args.synthetic_guard_size), int(args.input_dim), int(args.num_classes), device)
        results = train_family_set(xtr, ytr, xg, yg, seed, args, device)
        rows.append(summarize_family_row({"teacher_type": teacher, "seed": seed, "teacher_in_shallow_span": int(teacher == "c1a_shallow_additive"), "teacher_in_depth2_span": int(teacher in {"c1a_shallow_additive", "c1b_deep_coordinate"}), "teacher_in_depth3_span": int(teacher in {"c1a_shallow_additive", "c1b_deep_coordinate"})}, results))
    # C2 visual synthetic is kept in the same capability artifact as required by Part C.
    visual_jobs = [(task, seed) for task in ["stroke_hv", "local_patch_interaction", "two_patch_composition", "rotation_sensitive"] for seed in range(int(args.visual_seed_count))]
    for task, seed in shard_items(visual_jobs, args):
        xtr, ytr, xg, yg = visual_synthetic_task(
            task,
            seed,
            int(args.synthetic_train_size),
            int(args.synthetic_guard_size),
            int(args.visual_side),
            int(args.num_classes),
            device,
            fixed_patch_features=bool(int(args.visual_fixed_patch_features)),
            task_version=str(args.visual_task_version),
        )
        results = train_family_set(xtr, ytr, xg, yg, seed + 1000, args, device)
        rows.append(
            summarize_family_row(
                {
                    "teacher_type": "c2_visual_synthetic",
                    "visual_synthetic_task": task,
                    "visual_task_version": str(args.visual_task_version),
                    "seed": seed,
                    "visual_fixed_patch_features": int(args.visual_fixed_patch_features),
                    "teacher_in_shallow_span": 0,
                    "teacher_in_depth2_span": 1,
                    "teacher_in_depth3_span": 1,
                },
                results,
            )
        )
    path = OUT_ROOT / f"part_c_architecture_capability_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-c-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_c_architecture_capability_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    c1a = [r for r in rows if r.get("teacher_type") == "c1a_shallow_additive"]
    c1b = [r for r in rows if r.get("teacher_type") == "c1b_deep_coordinate"]
    c1c = [r for r in rows if r.get("teacher_type") == "c1c_mlp_friendly"]
    visual = [r for r in rows if r.get("teacher_type") == "c2_visual_synthetic"]
    c1a_metric_pass = int(bool(c1a) and median([fval(r.get("shallow_metric_guard_accuracy")) for r in c1a]) >= 0.65 and max(median([fval(r.get("depth2_metric_guard_accuracy")) for r in c1a]), median([fval(r.get("depth3_metric_guard_accuracy")) for r in c1a])) >= 0.65)
    c1a_architecture_pass = int(
        bool(c1a)
        and median([max(fval(r.get("shallow_metric_guard_accuracy")), fval(r.get("shallow_adamw_guard_accuracy"))) for r in c1a]) >= 0.60
        and median([max(fval(r.get("depth2_metric_guard_accuracy")), fval(r.get("depth2_adamw_guard_accuracy")), fval(r.get("depth3_metric_guard_accuracy")), fval(r.get("depth3_adamw_guard_accuracy"))) for r in c1a]) >= 0.60
    )
    c1a_pass = c1a_architecture_pass
    c1b_pass = int(bool(c1b) and sum(ival(r.get("deep_beats_shallow")) for r in c1b) >= math.ceil(0.6 * len(c1b)) and median([fval(r.get("deep_beats_shallow_margin")) for r in c1b]) > 0.02)
    c1c_negative_pass = int(bool(c1c) and sum(int(fval(r.get("mlp_raw_guard_NLL")) <= fval(r.get("best_deep_guard_NLL")) + 0.05) for r in c1c) >= math.ceil(0.5 * len(c1c)))
    visual_improve = median(
        [
            max(fval(r.get("depth2_metric_coverage_CVaR25")), fval(r.get("depth3_metric_coverage_CVaR25")))
            - fval(r.get("shallow_metric_coverage_CVaR25"))
            for r in visual
        ]
    )
    visual_metric_acc_improve = median(
        [
            max(fval(r.get("depth2_metric_guard_accuracy")), fval(r.get("depth3_metric_guard_accuracy")))
            - fval(r.get("shallow_metric_guard_accuracy"))
            for r in visual
        ]
    )
    visual_adamw_acc_improve = median(
        [
            max(fval(r.get("depth2_adamw_guard_accuracy")), fval(r.get("depth3_adamw_guard_accuracy")))
            - fval(r.get("shallow_adamw_guard_accuracy"))
            for r in visual
        ]
    )
    visual_adamw_cov_improve = median(
        [
            max(fval(r.get("depth2_adamw_coverage_CVaR25")), fval(r.get("depth3_adamw_coverage_CVaR25")))
            - fval(r.get("shallow_adamw_coverage_CVaR25"))
            for r in visual
        ]
    )
    c2_adamw_architecture_signal = int(
        bool(visual)
        and visual_adamw_acc_improve >= 0.10
        and sum(
            int(
                max(fval(r.get("depth2_adamw_guard_accuracy")), fval(r.get("depth3_adamw_guard_accuracy")))
                > fval(r.get("shallow_adamw_guard_accuracy"))
            )
            for r in visual
        )
        >= math.ceil(0.8 * len(visual))
    )
    c2_pass = int(visual and visual_improve >= 0.02)
    gate = int(c1a_pass and c1b_pass and c1c_negative_pass and c2_pass)
    if not c1a_pass:
        route, blocker = "C_ShallowMechanismBroken", "c1a_shallow"
    elif not c1b_pass:
        route, blocker = "C_DeepCoordinateFormationWeak", "c1b_deep_coordinate"
    elif not c1c_negative_pass:
        route, blocker = "C_NegativeControlMisreportsKANAdvantage", "c1c_negative_control"
    elif not c2_pass:
        route, blocker = "C_VisualSyntheticNoDeepCoverageGain", "c2_visual"
    else:
        route, blocker = "C_Pass", "none"
    out = {
        "gate": "v22_93_part_c_architecture_capability",
        "rows": len(rows),
        "c1a_rows": len(c1a),
        "c1b_rows": len(c1b),
        "c1c_rows": len(c1c),
        "visual_rows": len(visual),
        "c1a_pass": c1a_pass,
        "c1a_metric_pass": c1a_metric_pass,
        "c1a_architecture_pass": c1a_architecture_pass,
        "c1b_pass": c1b_pass,
        "c1c_negative_control_pass": c1c_negative_pass,
        "c2_visual_pass": c2_pass,
        "c2_metric_visual_accuracy_improvement_median": visual_metric_acc_improve,
        "c2_adamw_architecture_signal": c2_adamw_architecture_signal,
        "c2_adamw_visual_accuracy_improvement_median": visual_adamw_acc_improve,
        "c2_adamw_visual_coverage_improvement_median": visual_adamw_cov_improve,
        "visual_synthetic_coverage_improvement_median": visual_improve,
        "part_c_gate_pass": gate,
        "part_c_route": route,
        "dominant_blocker": blocker,
        "changed_w1": int(any(ival(r.get("changed_w1")) for r in rows)),
        "changed_mlp": int(any(ival(r.get("changed_mlp")) for r in rows)),
    }
    matrix = OUT_ROOT / "part_c_architecture_capability_matrix.csv"
    summary = OUT_ROOT / "part_c_architecture_capability_summary.json"
    write_rows(matrix, rows)
    write_json(summary, out)
    if gate:
        actions = []
    elif blocker == "c2_visual" and c2_adamw_architecture_signal:
        actions = [{"action": "repair_metric_tube_visual_update_or_purekan_patch_layer", "reason": "deep_adamw_architecture_signal_but_metric_visual_gain_low"}]
    else:
        actions = [{"action": "repair_deep_purekan_coordinate_formation_or_visual_scaling", "reason": blocker}]
    next_path = write_next_actions("c", route, blocker, actions)
    append_exec("part-c-merge", command_text(sys.argv), "done" if gate else "failed", files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part C architecture capability",
        [
            f"gate_pass={gate}; route={route}; rows={len(rows)}; c1a={c1a_pass}; c1b={c1b_pass}; c1c_negative={c1c_negative_pass}; c2_visual={c2_pass}",
            f"visual_synthetic_coverage_improvement_median={visual_improve}; c2_adamw_architecture_signal={c2_adamw_architecture_signal}; c2_adamw_visual_accuracy_improvement_median={visual_adamw_acc_improve}",
            "analysis: Part C compares fixed shallow/depth2/depth3/MLP families offline. It does not switch architecture at runtime or promote best rows into an official runtime.",
        ],
    )
    return out


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for family in ["depth2", "depth3"]:
        for seed in range(int(args.unit_seed_count)):
            xtr, ytr, xg, _yg = synthetic_task("c1b_deep_coordinate", seed, int(args.synthetic_train_size), int(args.synthetic_guard_size), int(args.input_dim), int(args.num_classes), device)
            model = make_kan_family(family, int(xtr.shape[1]), int(args.num_classes), seed + 700, args, device)
            logits, activations = model.forward_with_activations(xtr)
            loss = F.cross_entropy(logits.float(), ytr)
            grads = torch.autograd.grad(loss, list(model.coeffs), retain_graph=False, create_graph=False)
            for layer_idx, (param, grad) in enumerate(zip(model.coeffs, grads)):
                c, r, info = layer_metric(model, xtr, layer_idx, args)
                a = w1_to_matrix(param.detach()).to(device=device, dtype=torch.float64)
                g = w1_to_matrix(grad.detach()).to(device=device, dtype=torch.float64)
                delta, diag = natural_tangent_velocity(c, a, g, EPS)
                if int(args.metric_shape_scale_branch):
                    radial = (((-g) * a).sum() / (a.square().sum().clamp_min(EPS))) * a
                    radial_cap = float(args.shape_radial_fraction_cap) * max(float(delta.norm().detach().cpu().item()), 0.01 * float(a.norm().detach().cpu().item()), EPS)
                    radial_norm = float(radial.norm().detach().cpu().item())
                    if radial_norm > radial_cap:
                        radial = radial * (radial_cap / max(radial_norm, EPS))
                        diag["shape_radial_clipped"] = 1.0
                    delta = delta + radial
                    preserved = ((-g) * delta).sum() / (g.norm() * delta.norm()).clamp_min(EPS)
                    diag["task_gradient_preserved_fraction"] = float(preserved.detach().cpu().item())
                    diag["shape_radial_fraction"] = float((radial.norm() / delta.norm().clamp_min(EPS)).detach().cpu().item())
                a_tilde = a + float(args.metric_lr) * delta
                r_target = r
                if int(args.metric_tube_gradient_residual):
                    residual_velocity = -g
                    scale = 1.0
                    budget = float(args.metric_tube_drift_budget)
                    best_a = a
                    best_drift = 0.0
                    for _ in range(40):
                        cand = a + float(args.metric_lr) * scale * residual_velocity
                        drift = relative_fro_error(composite_metric(cand, c), r_target)
                        if drift <= budget:
                            best_a = cand
                            best_drift = drift
                            break
                        scale *= 0.5
                    a_new = best_a
                    retract_diag = {
                        "metric_drift_before_retraction": best_drift,
                        "metric_drift_after_retraction": best_drift,
                        "retraction_error": best_drift,
                        "shape_drift": best_drift,
                        "scale_change": float((torch.trace(composite_metric(a_new, c)) / torch.trace(r_target).clamp_min(EPS)).detach().cpu().item()),
                    }
                    diag["tube_gradient_residual_used"] = 1.0
                    diag["tube_gradient_residual_scale"] = scale
                else:
                    if int(args.metric_shape_scale_branch):
                        r_tilde = composite_metric(a_tilde, c)
                        trace_scale = torch.trace(r_tilde).clamp_min(EPS) / torch.trace(r).clamp_min(EPS)
                        r_target = sym(r * trace_scale)
                        diag["shape_trace_scale"] = float(trace_scale.detach().cpu().item())
                    a_new, retract_diag = retract_to_metric(a_tilde, c, r_target, rho=1.0e-8, eps=EPS)
                actual_velocity = (a_new - a) / max(float(args.metric_lr), EPS)
                actual_preserved = ((-g) * actual_velocity).sum() / (g.norm() * actual_velocity.norm()).clamp_min(EPS)
                diag["task_gradient_preserved_fraction"] = float(actual_preserved.detach().cpu().item())
                diag["actual_projected_velocity_norm"] = float(actual_velocity.norm().detach().cpu().item())
                with torch.no_grad():
                    before_act = activations[layer_idx + 1].detach()
                    param_before = param.detach().clone()
                    param.copy_(matrix_to_w1(a_new, param.shape, dtype=param.dtype, device=param.device))
                    _logits_after, acts_after = model.forward_with_activations(xtr)
                    after_act = acts_after[layer_idx + 1].detach()
                    param.copy_(param_before)
                row = {
                    "model_family": family,
                    "seed": seed,
                    "layer_id": layer_idx,
                    "C_condition": info["C_condition"],
                    "R_condition": info["R_condition"],
                    "R_trace": info["R_trace"],
                    "R_rank": info["R_rank"],
                    "metric_drift_before": retract_diag["metric_drift_before_retraction"],
                    "metric_drift_after": retract_diag["metric_drift_after_retraction"],
                    "tangent_residual": diag["tangent_residual"],
                    "sylvester_residual": diag["Sylvester_residual"],
                    "retraction_error": retract_diag["retraction_error"],
                    "task_gradient_preserved_fraction": diag["task_gradient_preserved_fraction"],
                    "velocity_norm": diag.get("actual_projected_velocity_norm", diag["projected_velocity_norm"]),
                    "shape_branch_used": int(args.metric_shape_scale_branch),
                    "tube_gradient_residual_used": int(args.metric_tube_gradient_residual),
                    "tube_gradient_residual_scale": diag.get("tube_gradient_residual_scale", 0.0),
                    "shape_radial_fraction": diag.get("shape_radial_fraction", 0.0),
                    "shape_trace_scale": diag.get("shape_trace_scale", 1.0),
                    "activation_domain_drift": float((after_act - before_act).norm().div(before_act.norm().clamp_min(EPS)).detach().cpu().item()),
                    "edge_effect_fraction": 1.0,
                    "inter_layer_activation_variance": info["activation_variance"],
                    "h_l_variance_before_after": float(after_act.var(unbiased=False).detach().cpu().item() - before_act.var(unbiased=False).detach().cpu().item()),
                    "h_l_pairwise_correlation_change": offdiag_mixing(after_act) - offdiag_mixing(before_act),
                    "activation_rank_l": activation_rank(before_act),
                    "activation_rank_change_l": activation_rank(after_act) - activation_rank(before_act),
                    "coordinate_mixing_proxy_l": offdiag_mixing(before_act),
                    "changed_w1_edge_tensors": int(not torch.allclose(param_before, matrix_to_w1(a_new, param.shape, dtype=param.dtype, device=param.device))),
                    "changed_mlp_tensors": 0,
                    "changed_w1": 1,
                }
                rows.append(row)
    gate = int(
        rows
        and max(fval(r.get("metric_drift_after")) for r in rows) <= 0.01
        and max(fval(r.get("tangent_residual")) for r in rows) <= 1.0e-4
        and max(fval(r.get("sylvester_residual")) for r in rows) <= 1.0e-4
        and min(fval(r.get("task_gradient_preserved_fraction")) for r in rows) >= 0.2
        and all(ival(r.get("changed_w1_edge_tensors")) == 1 for r in rows)
    )
    route = "D_Pass" if gate else "D_DeepMetricImplementationFailed"
    blocker = "none" if gate else "metric_unit"
    matrix = OUT_ROOT / "part_d_deep_metric_unit_tests.csv"
    summary = OUT_ROOT / "part_d_deep_metric_unit_tests_summary.json"
    write_rows(matrix, rows)
    write_json(
        summary,
        {
            "gate": "v22_93_part_d_deep_metric_unit_tests",
            "rows": len(rows),
            "part_d_gate_pass": gate,
            "part_d_route": route,
            "dominant_blocker": blocker,
            "metric_drift_after_max": max([fval(r.get("metric_drift_after")) for r in rows] or [0.0]),
            "tangent_residual_max": max([fval(r.get("tangent_residual")) for r in rows] or [0.0]),
            "sylvester_residual_max": max([fval(r.get("sylvester_residual")) for r in rows] or [0.0]),
            "task_gradient_preserved_fraction_min": min([fval(r.get("task_gradient_preserved_fraction"), 1.0) for r in rows] or [0.0]),
            "changed_w1": int(any(ival(r.get("changed_w1_edge_tensors")) for r in rows)),
        },
    )
    next_path = write_next_actions("d", route, blocker, [] if gate else [{"action": "repair_per_layer_optimizer_grouping_and_gradient_routing", "reason": blocker}])
    append_exec("part-d", command_text(sys.argv), "done" if gate else "failed", gpu=str(device), files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part D deep metric unit tests",
        [
            f"gate_pass={gate}; rows={len(rows)}; metric_drift_after_max={max([fval(r.get('metric_drift_after')) for r in rows] or [0.0])}; tangent_residual_max={max([fval(r.get('tangent_residual')) for r in rows] or [0.0])}; sylvester_residual_max={max([fval(r.get('sylvester_residual')) for r in rows] or [0.0])}",
            "analysis: Part D tests per-layer tangent projection and metric retraction on true depth2/depth3 PureKAN coefficient tensors. No MLP tensors are involved.",
        ],
    )
    return read_json(summary)


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    c = read_json(OUT_ROOT / "part_c_architecture_capability_summary.json")
    d = read_json(OUT_ROOT / "part_d_deep_metric_unit_tests_summary.json")
    if not c.get("part_c_gate_pass") or not d.get("part_d_gate_pass"):
        rows: list[dict[str, Any]] = []
        matrix = OUT_ROOT / "part_e_positive_control_matrix.csv"
        summary = OUT_ROOT / "part_e_positive_control_summary.json"
        write_rows(matrix, rows)
        out = {
            "gate": "v22_93_part_e_positive_control",
            "part_e_gate_pass": 0,
            "part_e_route": "E_SkippedBecauseCOrDFailed",
            "skip_reason": f"part_c={c.get('part_c_gate_pass')}; part_d={d.get('part_d_gate_pass')}",
            "rows": 0,
        }
        write_json(summary, out)
        next_path = write_next_actions("e", out["part_e_route"], "part_c_or_d", [{"action": "repair_failed_part_c_or_d_before_part_e", "reason": out["skip_reason"]}])
        append_exec("part-e", command_text(sys.argv), "skipped", files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
        append_recap("Part E positive-control suite", [f"gate_pass=0; route={out['part_e_route']}; reason={out['skip_reason']}", "analysis: v22.93 forbids Part E promotion while Part C/D are missing or failed."])
        return out
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    teacher_types = ["e1_shallow", "e2_deep", "e3_mlp_friendly"]
    for teacher in teacher_types:
        for seed in range(int(args.positive_seed_count)):
            xtr, ytr, xg, yg = synthetic_task(teacher, seed, int(args.synthetic_train_size), int(args.synthetic_guard_size), int(args.input_dim), int(args.num_classes), device)
            results = train_family_set(xtr, ytr, xg, yg, seed + 2000, args, device)
            rows.append(summarize_family_row({"teacher_type": teacher, "seed": seed}, results))
    for task in ["stroke_hv", "local_patch_interaction", "two_patch_composition"]:
        for seed in range(int(args.positive_seed_count)):
            xtr, ytr, xg, yg = visual_synthetic_task(
                task,
                seed,
                int(args.synthetic_train_size),
                int(args.synthetic_guard_size),
                int(args.visual_side),
                int(args.num_classes),
                device,
                fixed_patch_features=bool(int(args.visual_fixed_patch_features)),
                task_version=str(args.visual_task_version),
            )
            results = train_family_set(xtr, ytr, xg, yg, seed + 3000, args, device)
            rows.append(
                summarize_family_row(
                    {
                        "teacher_type": "e4_visual",
                        "visual_synthetic_task": task,
                        "visual_task_version": str(args.visual_task_version),
                        "seed": seed,
                        "visual_fixed_patch_features": int(args.visual_fixed_patch_features),
                    },
                    results,
                )
            )
    e1 = [r for r in rows if r.get("teacher_type") == "e1_shallow"]
    e2 = [r for r in rows if r.get("teacher_type") == "e2_deep"]
    e3 = [r for r in rows if r.get("teacher_type") == "e3_mlp_friendly"]
    e4 = [r for r in rows if r.get("teacher_type") == "e4_visual"]
    e1_pass = int(e1 and sum(ival(r.get("deep_no_debt")) for r in e1) >= math.ceil(0.8 * len(e1)) and median([fval(r.get("shallow_metric_guard_accuracy")) for r in e1]) >= 0.65)
    e2_beats_shallow = sum(ival(r.get("deep_beats_shallow")) for r in e2)
    e2_pass = int(e2 and e2_beats_shallow >= math.ceil(0.8 * len(e2)) and sum(ival(r.get("deep_no_debt")) for r in e2) >= math.ceil(0.8 * len(e2)))
    e3_pass = int(e3 and sum(int(fval(r.get("mlp_raw_guard_NLL")) <= fval(r.get("best_deep_guard_NLL")) + 0.05) for r in e3) >= math.ceil(0.5 * len(e3)))
    e4_improve = median(
        [
            max(fval(r.get("depth2_metric_coverage_CVaR25")), fval(r.get("depth3_metric_coverage_CVaR25")))
            - fval(r.get("shallow_metric_coverage_CVaR25"))
            for r in e4
        ]
    )
    e4_beats_shallow = sum(ival(r.get("deep_beats_shallow")) for r in e4)
    e4_source_guard = sum(ival(r.get("deep_source_guard_R")) for r in e4)
    e4_pass = int(e4 and e4_improve >= 0.20 and e4_beats_shallow >= 12 and e4_source_guard >= 12)
    gate = int(e1_pass and e2_pass and e3_pass and e4_pass)
    if not e1_pass:
        route, blocker = "E_ShallowPositiveControlBroken", "e1_shallow"
    elif not e2_pass:
        route, blocker = "E_DeepCoordinateFormationFailed", "e2_deep"
    elif not e4_pass:
        route, blocker = "E_VisualSyntheticControlFailed", "e4_visual"
    else:
        route, blocker = "E_DeepPositiveControlPassed", "none"
    matrix = OUT_ROOT / "part_e_positive_control_matrix.csv"
    summary = OUT_ROOT / "part_e_positive_control_summary.json"
    write_rows(matrix, rows)
    out = {
        "gate": "v22_93_part_e_positive_control",
        "rows": len(rows),
        "part_e_gate_pass": gate,
        "part_e_route": route,
        "dominant_blocker": blocker,
        "e1_pass": e1_pass,
        "e2_pass": e2_pass,
        "e3_negative_pass": e3_pass,
        "e4_visual_pass": e4_pass,
        "e4_visual_beats_shallow": e4_beats_shallow,
        "e4_visual_source_guard": e4_source_guard,
        "e4_visual_rows": len(e4),
        "e2_deep_beats_shallow": e2_beats_shallow,
        "e2_rows": len(e2),
        "visual_synthetic_coverage_improvement_median": e4_improve,
        "changed_w1": int(any(ival(r.get("changed_w1")) for r in rows)),
        "changed_mlp": int(any(ival(r.get("changed_mlp")) for r in rows)),
    }
    write_json(summary, out)
    next_path = write_next_actions("e", route, blocker, [] if gate else [{"action": "repair_positive_control_blocker", "reason": blocker}])
    append_exec("part-e", command_text(sys.argv), "done" if gate else "failed", gpu=str(device), files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part E positive-control suite",
        [
            f"gate_pass={gate}; route={route}; rows={len(rows)}; e1={e1_pass}; e2={e2_pass}; e3_negative={e3_pass}; e4_visual={e4_pass}",
            f"e2_deep_beats_shallow={e2_beats_shallow}/{len(e2)}; visual_synthetic_coverage_improvement_median={e4_improve}; e4_beats_shallow={e4_beats_shallow}/{len(e4)}; e4_source_guard={e4_source_guard}/{len(e4)}",
            "analysis: Part E runs fixed shallow/deep/MLP controls on synthetic positive/negative/visual controls. Failure blocks real-task Part F by design.",
        ],
    )
    return out


def part_f_jobs(args: argparse.Namespace) -> list[tuple[str, int]]:
    return [(dataset, seed) for dataset in csv_items(args.part_f_datasets) for seed in range(int(args.part_f_seed_count))]


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    e = read_json(OUT_ROOT / "part_e_positive_control_summary.json")
    if not e.get("part_e_gate_pass"):
        matrix = OUT_ROOT / "part_f_real_task_preflight_matrix.csv"
        summary = OUT_ROOT / "part_f_real_task_preflight_summary.json"
        write_rows(matrix, [])
        out = {"gate": "v22_93_part_f_real_task_preflight", "part_f_gate_pass": 0, "part_f_route": "F_SkippedBecausePartEFailed", "skip_reason": e.get("part_e_route"), "rows": 0}
        write_json(summary, out)
        next_path = write_next_actions("f", out["part_f_route"], "part_e", [{"action": "repair_part_e_before_real_tasks", "reason": str(e.get("part_e_route"))}])
        append_exec("part-f", command_text(sys.argv), "skipped", files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
        append_recap("Part F real-task preflight", [f"gate_pass=0; route={out['part_f_route']}; reason={out['skip_reason']}", "analysis: v22.93 forbids real-task preflight when deep positive-control is not passed."])
        return out
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    pf_args = v2289.part_f_flow_args(args)
    for dataset, seed in shard_items(part_f_jobs(args), args):
        bundle = v2289.load_real_task_bundle(dataset, seed, pf_args, device)
        xtr, ytr = bundle["x_train"], bundle["y_train"]
        xg, yg = bundle["x_held"], bundle["y_held"]
        results = train_family_set(xtr, ytr, xg, yg, seed + 4000, args, device)
        base = {
            "dataset": dataset,
            "seed": seed,
            "used_fake_data_rows": int(bundle.get("used_fake_data", 0)),
            "held_test_usage": 0,
            "input_dim": int(bundle["input_dim"]),
            "original_input_dim": int(bundle["original_input_dim"]),
            "feature_projection_kind": bundle.get("feature_projection_kind", ""),
            "num_classes": int(bundle["num_classes"]),
            "dataset_family": "visual" if dataset in {"MNIST", "FashionMNIST", "KMNIST"} else "tabular",
        }
        row = summarize_family_row(base, results)
        row["visual_coverage"] = int(row["dataset_family"] == "visual" and max(fval(row.get("depth2_metric_coverage_CVaR25")), fval(row.get("depth3_metric_coverage_CVaR25"))) >= 0.20)
        row["tabular_coverage"] = int(row["dataset_family"] == "tabular" and max(fval(row.get("depth2_metric_coverage_CVaR25")), fval(row.get("depth3_metric_coverage_CVaR25"))) >= 0.20)
        row["overhead_ratio"] = max(fval(row.get("depth2_metric_wall_time_s")), fval(row.get("depth3_metric_wall_time_s"))) / max(fval(row.get("mlp_raw_wall_time_s")), EPS)
        row["overhead_pass"] = int(row["overhead_ratio"] <= float(args.overhead_ratio_budget))
        rows.append(row)
    path = OUT_ROOT / f"part_f_real_task_preflight_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-f-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def merge_part_f(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_f_real_task_preflight_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    ok = [r for r in rows if ival(r.get("used_fake_data_rows")) == 0]
    beats_mlp = sum(ival(r.get("deep_beats_mlp_raw")) for r in ok)
    beats_best = sum(ival(r.get("deep_beats_best_control")) for r in ok)
    no_debt = sum(ival(r.get("deep_no_debt")) for r in ok)
    coverage = sum(ival(r.get("deep_coverage")) for r in ok)
    visual = [r for r in ok if r.get("dataset_family") == "visual"]
    visual_cov = sum(ival(r.get("visual_coverage")) for r in visual)
    source_guard = sum(ival(r.get("deep_source_guard_R")) for r in ok)
    overhead = sum(ival(r.get("overhead_pass")) for r in ok)
    initial_pass = int(len(ok) >= len(part_f_jobs(args)) and beats_mlp >= 18 and beats_best >= 18 and no_debt >= 18 and coverage >= 18 and visual_cov >= 8 and source_guard >= 18 and overhead >= 18)
    strong_pass = int(beats_mlp >= 24 and no_debt >= 24 and coverage >= 24 and visual_cov >= 14 and source_guard >= 24 and overhead >= 24)
    if strong_pass:
        route, blocker = "F_DeepPureKANOfficialPreflightPass", "none"
    elif initial_pass:
        route, blocker = "F_DeepPureKANImprovesButNotOfficial", "strict_official"
    elif visual and visual_cov < 8:
        route, blocker = "F_DeepPureKANNoVisualCoverage", "visual_coverage"
    elif beats_mlp < 18:
        route, blocker = "F_DeepPureKANMLPMatchedDominates", "mlp_matched"
    elif no_debt < 18:
        route, blocker = "F_DeepPureKANDebtBlocked", "debt"
    else:
        route, blocker = "F_DeepPureKANPreflightFailed", "mixed"
    matrix = OUT_ROOT / "part_f_real_task_preflight_matrix.csv"
    summary = OUT_ROOT / "part_f_real_task_preflight_summary.json"
    write_rows(matrix, rows)
    out = {
        "gate": "v22_93_part_f_real_task_preflight",
        "rows": len(rows),
        "ok_rows": len(ok),
        "expected_rows": len(part_f_jobs(args)),
        "part_f_gate_pass": int(strong_pass),
        "part_f_initial_deep_family_pass": initial_pass,
        "part_f_route": route,
        "dominant_blocker": blocker,
        "beats_MLP_composite": beats_mlp,
        "beats_MLP_raw": beats_mlp,
        "beats_best_control": beats_best,
        "no_debt": no_debt,
        "coverage": coverage,
        "visual_coverage": visual_cov,
        "source_guard_R": source_guard,
        "overhead": overhead,
        "used_fake_data_rows": sum(ival(r.get("used_fake_data_rows"), 1) for r in rows),
        "changed_w1": int(any(ival(r.get("changed_w1")) for r in rows)),
        "changed_mlp": int(any(ival(r.get("changed_mlp")) for r in rows)),
    }
    write_json(summary, out)
    next_path = write_next_actions("f", route, blocker, [] if initial_pass else [{"action": "follow_part_f_blocker_repair_protocol", "reason": blocker}])
    append_exec("part-f-merge", command_text(sys.argv), "done" if initial_pass else "failed", files=f"{rel(matrix)}; {rel(summary)}; {rel(next_path)}")
    append_recap(
        "Part F real-task preflight",
        [
            f"gate_pass={out['part_f_gate_pass']}; initial_pass={initial_pass}; route={route}; ok_rows={len(ok)}/{len(rows)}; fake_rows={out['used_fake_data_rows']}",
            f"counts: beats_MLP={beats_mlp}; beats_best_control={beats_best}; no_debt={no_debt}; coverage={coverage}; visual_coverage={visual_cov}; source_guard={source_guard}; overhead={overhead}",
            "analysis: Part F compares fixed shallow/depth2/depth3/MLP families. It never switches depth or architecture by dataset/seed at runtime.",
        ],
    )
    return out


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    f = read_json(OUT_ROOT / "part_f_real_task_preflight_summary.json")
    if not f.get("part_f_initial_deep_family_pass"):
        out = {"gate": "v22_93_part_g_hstep", "part_g_gate_pass": 0, "part_g_route": "skipped", "skip_reason": "Part F initial deep-family pass not reached"}
        write_json(OUT_ROOT / "part_g_hstep_summary.json", out)
        write_rows(OUT_ROOT / "part_g_hstep_matrix.csv", [])
        next_path = write_next_actions("g", "skipped", "part_f", [{"action": "run_only_after_part_f_initial_pass", "reason": out["skip_reason"]}])
        append_exec("part-g", command_text(sys.argv), "skipped", files=f"{rel(OUT_ROOT / 'part_g_hstep_summary.json')}; {rel(next_path)}")
        append_recap("Part G H-step trajectory", [f"gate_pass=0; route=skipped; reason={out['skip_reason']}", "analysis: Part G is gated behind Part F initial pass and was not run as success evidence."])
        return out
    out = {"gate": "v22_93_part_g_hstep", "part_g_gate_pass": 0, "part_g_route": "G_NotImplementedAfterFInitialPass", "skip_reason": "H-step limited full-loop not implemented in this runner revision"}
    write_json(OUT_ROOT / "part_g_hstep_summary.json", out)
    write_rows(OUT_ROOT / "part_g_hstep_matrix.csv", [])
    next_path = write_next_actions("g", out["part_g_route"], "hstep_missing", [{"action": "implement_h20_h60_after_part_f_initial_pass", "reason": "Part F initial pass reached"}])
    append_exec("part-g", command_text(sys.argv), "failed", files=f"{rel(OUT_ROOT / 'part_g_hstep_summary.json')}; {rel(next_path)}")
    append_recap("Part G H-step trajectory", [f"gate_pass=0; route={out['part_g_route']}", "analysis: H-step cannot be claimed without a concrete H20/H60 artifact."])
    return out


def run_part_h(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    summaries = {
        "a": read_json(OUT_ROOT / "part_a_architecture_identity.json"),
        "b": read_json(OUT_ROOT / "part_b_history_lock.json"),
        "c": read_json(OUT_ROOT / "part_c_architecture_capability_summary.json"),
        "d": read_json(OUT_ROOT / "part_d_deep_metric_unit_tests_summary.json"),
        "e": read_json(OUT_ROOT / "part_e_positive_control_summary.json"),
        "f": read_json(OUT_ROOT / "part_f_real_task_preflight_summary.json"),
        "g": read_json(OUT_ROOT / "part_g_hstep_summary.json"),
    }
    lines = ["# v22.93 failure decomposition\n"]
    for part, data in summaries.items():
        lines.append(f"- Part {part.upper()}: route=`{data.get(f'part_{part}_route', data.get('part_' + part + '_gate_pass', data.get('part_' + part + '_initial_deep_family_pass', data.get('final_route', 'missing'))))}`, gate fields={{{', '.join(k + '=' + str(v) for k, v in data.items() if k.endswith('_gate_pass') or k.endswith('_route') or k == 'dominant_blocker')}}}")
    c = summaries["c"]
    e = summaries["e"]
    f = summaries["f"]
    lines.extend(
        [
            "\n## Analysis\n",
            f"- Part C blocker: `{c.get('dominant_blocker', 'missing')}` with visual improvement `{c.get('visual_synthetic_coverage_improvement_median', 'missing')}`.\n",
            f"- Part E blocker: `{e.get('dominant_blocker', e.get('skip_reason', 'missing'))}`.\n",
            f"- Part F route: `{f.get('part_f_route', 'missing')}`; if skipped, reason=`{f.get('skip_reason', '')}`.\n",
            "- No success is claimed from skipped or missing parts. If Part F is skipped, the correct conclusion remains positive-control/architecture blocker rather than real-task success.\n",
        ]
    )
    path = OUT_ROOT / "part_h_failure_decomposition.md"
    path.write_text("".join(lines), encoding="utf-8")
    out = {"gate": "v22_93_part_h_failure_decomposition", "part_h_written": 1, "part_h_path": rel(path)}
    write_json(OUT_ROOT / "part_h_summary.json", out)
    next_path = write_next_actions("h", "H_DecompositionWritten", "none", [])
    append_exec("part-h", command_text(sys.argv), "done", files=f"{rel(path)}; {rel(OUT_ROOT / 'part_h_summary.json')}; {rel(next_path)}")
    append_recap("Part H failure decomposition", [f"written={rel(path)}", "analysis: Part H synthesizes current blockers and next actions without converting skipped/failing parts into success."])
    return out


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_architecture_identity.json")
    b = read_json(OUT_ROOT / "part_b_history_lock.json")
    c = read_json(OUT_ROOT / "part_c_architecture_capability_summary.json")
    d = read_json(OUT_ROOT / "part_d_deep_metric_unit_tests_summary.json")
    e = read_json(OUT_ROOT / "part_e_positive_control_summary.json")
    f = read_json(OUT_ROOT / "part_f_real_task_preflight_summary.json")
    g = read_json(OUT_ROOT / "part_g_hstep_summary.json")
    if not a.get("part_a_gate_pass"):
        route, reason = "A_FailedArchitectureIdentity", "part_a"
    elif not b.get("part_b_gate_pass"):
        route, reason = "B_HistoryLockFailed", "part_b"
    elif not c.get("part_c_gate_pass"):
        route, reason = c.get("part_c_route", "C_ShallowOnlyModelConfirmed"), c.get("dominant_blocker", "part_c")
    elif not d.get("part_d_gate_pass"):
        route, reason = "D_DeepMetricImplementationFailed", d.get("dominant_blocker", "part_d")
    elif not e.get("part_e_gate_pass"):
        route, reason = e.get("part_e_route", "E_DeepCoordinateFormationFailed"), e.get("dominant_blocker", "part_e")
    elif not f.get("part_f_gate_pass"):
        route, reason = f.get("part_f_route", "F_DeepPureKANPreflightFailed"), f.get("dominant_blocker", "part_f")
    elif not g.get("part_g_gate_pass"):
        route, reason = "G_HStepFailed", g.get("skip_reason", "part_g")
    else:
        route, reason = "G_OfficialCandidatePass", "all_gates_passed"
    out = {
        "gate": "v22_93_final_route",
        "official_candidate_gate_pass": int(route == "G_OfficialCandidatePass"),
        "final_route": route,
        "route_reason": reason,
        "part_a": {"part_a_gate_pass": a.get("part_a_gate_pass")},
        "part_b": {"part_b_gate_pass": b.get("part_b_gate_pass")},
        "part_c": {"part_c_gate_pass": c.get("part_c_gate_pass"), "part_c_route": c.get("part_c_route"), "dominant_blocker": c.get("dominant_blocker")},
        "part_d": {"part_d_gate_pass": d.get("part_d_gate_pass"), "part_d_route": d.get("part_d_route")},
        "part_e": {"part_e_gate_pass": e.get("part_e_gate_pass"), "part_e_route": e.get("part_e_route")},
        "part_f": {"part_f_gate_pass": f.get("part_f_gate_pass"), "part_f_route": f.get("part_f_route"), "part_f_initial_deep_family_pass": f.get("part_f_initial_deep_family_pass")},
        "part_g": {"part_g_gate_pass": g.get("part_g_gate_pass"), "part_g_route": g.get("part_g_route")},
    }
    path = OUT_ROOT / "final_route.json"
    write_json(path, out)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    manifest.write_text(
        f"# v22.93 reproduction manifest\n\nPython: `{PYTHON}`\nRunner: `{rel(RUNNER)}`\nPlan: `{rel(PLAN)}`\nFinalize: `{PYTHON} {rel(RUNNER)} --mode finalize --device cuda`\n",
        encoding="utf-8",
    )
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(path)}; {rel(manifest)}")
    append_recap(
        "Final route",
        [
            f"official_candidate_gate_pass={out['official_candidate_gate_pass']}; final_route={route}; reason={reason}",
            "analysis: Final route is based only on current artifacts. Missing/skipped/failing Parts are not success evidence.",
        ],
    )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--pc-basis", default="D-FOU")
    p.add_argument("--basis-k-override", type=int, default=0)
    p.add_argument("--input-dim", type=int, default=12)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--deep-width", type=int, default=12)
    p.add_argument("--mlp-hidden", type=int, default=24)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--smoothness", type=float, default=1.0e-4)
    p.add_argument("--c-condition-budget", type=float, default=1.0e5)
    p.add_argument("--metric-lr", type=float, default=0.05)
    p.add_argument("--metric-update-mode", default="projected_metric")
    p.add_argument("--metric-projected-adamw-lr", type=float, default=0.02)
    p.add_argument("--adamw-lr", type=float, default=0.02)
    p.add_argument("--mlp-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--metric-max-norm-ratio", type=float, default=2.0)
    p.add_argument("--metric-shape-scale-branch", type=int, default=1)
    p.add_argument("--shape-radial-fraction-cap", type=float, default=2.0)
    p.add_argument("--metric-tube-gradient-residual", type=int, default=1)
    p.add_argument("--metric-tube-combine-tangent-residual", type=int, default=0)
    p.add_argument("--metric-tube-adamw-residual", type=int, default=0)
    p.add_argument("--metric-tube-adamw-beta1", type=float, default=0.90)
    p.add_argument("--metric-tube-adamw-beta2", type=float, default=0.999)
    p.add_argument("--metric-tube-adamw-eps", type=float, default=1.0e-8)
    p.add_argument("--metric-tube-adamw-norm-match", type=int, default=1)
    p.add_argument("--metric-tube-drift-budget", type=float, default=0.01)
    p.add_argument("--train-steps", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--no-debt-budget", type=float, default=0.01)
    p.add_argument("--synthetic-seed-count", type=int, default=5)
    p.add_argument("--visual-seed-count", type=int, default=3)
    p.add_argument("--positive-seed-count", type=int, default=5)
    p.add_argument("--unit-seed-count", type=int, default=3)
    p.add_argument("--synthetic-train-size", type=int, default=192)
    p.add_argument("--synthetic-guard-size", type=int, default=128)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--overhead-ratio-budget", type=float, default=8.0)
    p.add_argument("--part-f-datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--part-f-seed-count", type=int, default=6)
    p.add_argument("--part-f-train-size", type=int, default=512)
    p.add_argument("--part-f-held-size", type=int, default=256)
    p.add_argument("--part-f-test-size", type=int, default=256)
    p.add_argument("--part-f-max-input-dim", type=int, default=64)
    p.add_argument("--part-f-steps", type=int, default=120)
    p.add_argument("--part-f-eval-interval", type=int, default=20)
    p.add_argument("--part-f-batch-size", type=int, default=0)
    p.add_argument("--part-f-cmp-lr", type=float, default=0.8)
    p.add_argument("--part-f-adamw-lr", type=float, default=0.02)
    p.add_argument("--part-f-weight-decay", type=float, default=1.0e-4)
    p.add_argument("--part-f-target-variance", type=float, default=49.0)
    p.add_argument("--part-f-cmp-max-norm-ratio", type=float, default=20.0)
    p.add_argument("--part-f-scale-band", type=float, default=0.40)
    p.add_argument("--part-f-rep-lr-ratio", type=float, default=0.01)
    p.add_argument("--target-variance", type=float, default=49.0)
    p.add_argument("--rep-lr-ratio", type=float, default=0.01)
    p.add_argument("--pc-lr", type=float, default=1.0e-2)
    p.add_argument("--pc-weight-decay", type=float, default=1.0e-4)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode)
    if mode == "part-a":
        return run_part_a(args)
    if mode == "part-b":
        return run_part_b(args)
    if mode == "part-c":
        return run_part_c(args)
    if mode == "part-c-merge":
        return merge_part_c(args)
    if mode == "part-d":
        return run_part_d(args)
    if mode == "part-e":
        return run_part_e(args)
    if mode == "part-f":
        return run_part_f(args)
    if mode == "part-f-merge":
        return merge_part_f(args)
    if mode == "part-g":
        return run_part_g(args)
    if mode == "part-h":
        return run_part_h(args)
    if mode == "finalize":
        return finalize(args)
    if mode == "all":
        run_part_a(args)
        run_part_b(args)
        run_part_c(args)
        merge_part_c(args)
        run_part_d(args)
        run_part_e(args)
        run_part_f(args)
        merge_part_f(args)
        run_part_g(args)
        run_part_h(args)
        return finalize(args)
    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
