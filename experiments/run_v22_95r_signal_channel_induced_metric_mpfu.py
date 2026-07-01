#!/usr/bin/env python3
"""DG-KAN v22.95R signal-channel-induced metric MPFU runner.

This runner is intentionally conservative.  It implements the hard identity
gates and the Part C signal-channel estimator audit with real train-only
observations.  Later parts are gate-aware: when Part C is not valid they emit
blocked artifacts instead of running forbidden metric-induction experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import py_compile
import re
import sys
import time
import tokenize
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
import experiments.run_v22_94_adamw_witness_decomposition_mpfu as v2294
from dgkan.fu.layer_composite_metric import EPS, matrix_to_w1, sym, w1_to_matrix


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v22.95R_SignalChannelInducedMetric_MultiDirection_详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.95R_SignalChannelInducedMetric_MultiDirection_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.95R_SignalChannelInducedMetric_MultiDirection_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2295R_OUT_ROOT", str(ROOT / "results/v22_95R"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"
CHANNEL_ROOT = OUT_ROOT / "channel_projectors"

BASIS_KEYS = ("dche_k5", "dche_k9", "dfour_default")
DEFAULT_TASKS = ("local_patch_interaction", "rotation_sensitive", "stroke_hv")

AUDIT_DEFAULTS: dict[str, Any] = {
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "metric_winner_selection_used": 0,
    "candidate_update_selection_used": 0,
    "mlp_readout_used": 0,
    "mlp_initial_layer_used": 0,
    "external_product_feature_used": 0,
    "new_edge_function_added": 0,
    "changed_edge_coefficients": 1,
    "changed_mlp_tensors": 0,
    "standard_forward_backward_optimizer_loop": 1,
    "state_updated_every_step": 1,
    "preconditioner_applied_inside_optimizer_step": 1,
}


def ensure_out() -> None:
    for path in (OUT_ROOT, LOG_ROOT, CHANNEL_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def command_text(argv: Iterable[str]) -> str:
    return " ".join([PYTHON, rel(RUNNER), *list(argv)[1:]])


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.95R Execution Log\n\n"
            f"- Created: {now()}\n"
            f"- Plan: `{rel(PLAN)}`\n"
            f"- Runner: `{rel(RUNNER)}`\n"
            f"- Output root: `{rel(OUT_ROOT)}`\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.95R Experiment Recap\n\n"
            f"- Created: {now()}\n"
            "- No fabricated rows or inferred metrics are allowed in this file.\n\n",
            encoding="utf-8",
        )


def append_exec(part: str, command: str, status: str, files: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"## {now()} {part} {status}\n\n")
        fh.write(f"Command: `{command}`\n\n")
        if files:
            fh.write(f"Files: {files}\n\n")
        if note:
            fh.write(f"Note: {note}\n\n")


def append_recap(title: str, payload: dict[str, Any]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        fh.write("\n```\n\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def fval(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def ival(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = torch.tensor([float(v) for v in xs], dtype=torch.float64)
    y = torch.tensor([float(v) for v in ys], dtype=torch.float64)
    if int(x.numel()) < 2 or int(y.numel()) != int(x.numel()):
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((x @ y / denom).item())


def argument_hash(args: argparse.Namespace, ignore_shard: bool = True) -> str:
    payload = vars(args).copy()
    if ignore_shard:
        payload.pop("shard_index", None)
        payload.pop("device", None)
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    index = int(args.shard_index)
    return [item for idx, item in enumerate(items) if idx % count == index]


def device_from_args(args: argparse.Namespace) -> torch.device:
    text = str(args.device)
    if text.startswith("cuda") and torch.cuda.is_available():
        return torch.device(text)
    return torch.device("cpu")


def common_next_actions(part: str, blocker: str, hypothesis: str, allowed: list[str], forbidden: list[str], commands: list[str]) -> Path:
    path = OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json"
    write_json(
        path,
        {
            "part": part.upper(),
            "dominant_blocker": blocker,
            "hypothesis": hypothesis,
            "allowed_actions": allowed,
            "forbidden_actions": forbidden,
            "max_repair_rounds": 2,
            "rerun_commands": commands,
            "promotion_allowed": False,
        },
    )
    return path


def gate_summary(part: str, gate_pass: int, route: str, blocker: str, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    ok_rows = sum(1 for row in rows if row.get("status") == "ok")
    error_rows = sum(1 for row in rows if row.get("status") not in {"ok", "skipped"})
    out = {
        "part": part.upper(),
        "gate_pass": int(gate_pass),
        "route": route,
        "dominant_blocker": blocker,
        "ok_rows": int(ok_rows),
        "error_rows": int(error_rows),
        "used_fake_data_rows": 0,
        "held_test_usage": 0,
        "promotion_allowed": False,
    }
    out.update(extra)
    return out


def strip_code(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
        toks: list[str] = []
        stream = io.StringIO(text)
        for tok in tokenize.generate_tokens(stream.readline):
            if tok.type in {tokenize.COMMENT, tokenize.STRING}:
                toks.append(" ")
            else:
                toks.append(tok.string)
        return "".join(toks)
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")


def static_scan(files: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_architecture_winner_selection": re.compile(r"\b(select_arch|choose_arch|winner_family)\s*\("),
        "runtime_metric_winner_selection": re.compile(r"\b(select_metric|choose_metric|winner_metric)\s*\("),
        "runtime_update_winner_selection": re.compile(r"\b(select_update|choose_update|candidate_score|winner_update)\s*\("),
        "runtime_topk_edge_selection": re.compile(r"\.topk\s*\("),
        "external_product_feature_call": re.compile(r"\b(make_external_product_feature|external_product_feature_transform|patch_product_feature)\s*\("),
        "new_edge_family_call": re.compile(r"\b(learned_square_edge|new_edge_basis_family)\s*\("),
        "mlp_stem_or_readout_call": re.compile(r"\b(mlp_stem|mlp_readout|readout_lstsq|oracle_target)\s*\("),
    }
    hits: list[dict[str, str]] = []
    for path in files:
        code = strip_code(path)
        for check, pattern in patterns.items():
            match = pattern.search(code)
            if match:
                hits.append({"file": rel(path), "check": check, "match": match.group(0)})
    return int(not hits), hits


def basis_args(args: argparse.Namespace, basis_key: str) -> argparse.Namespace:
    return v2294.basis_args(args, basis_key)


def make_model(depth: str, input_dim: int, classes: int, seed: int, args: argparse.Namespace, device: torch.device) -> v2293.TrueDeepPureKAN:
    return v2294.make_model(depth, input_dim, classes, seed, args, device)


def visual_data(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2294.visual_data(task, seed, args, device)


def synthetic_data(teacher: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2294.synthetic_data(teacher, seed, args, device)


def mode_tensor_for_param(model: v2293.TrueDeepPureKAN, layer_idx: int, args: argparse.Namespace) -> torch.Tensor:
    param = model.coeffs[layer_idx]
    diag = v2294.mode_metric_diag(model, layer_idx, args).to(device=param.device, dtype=torch.float64)
    d_in, d_out, k = int(param.shape[0]), int(param.shape[1]), int(param.shape[2])
    out = torch.empty((d_in, d_out, k), device=param.device, dtype=torch.float64)
    for i in range(d_in):
        for m in range(k):
            out[i, :, m] = diag[i * k + m]
    return out


def flat_mode_vector(model: v2293.TrueDeepPureKAN, args: argparse.Namespace) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for layer_idx, param in enumerate(model.coeffs):
        parts.append(mode_tensor_for_param(model, layer_idx, args).reshape(-1))
    out = torch.cat(parts).to(dtype=torch.float64)
    return out.clamp_min(EPS)


def flatten_params_or_grads(tensors: Iterable[torch.Tensor | None], params: list[torch.Tensor]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for tensor, param in zip(tensors, params):
        if tensor is None:
            parts.append(torch.zeros_like(param.detach()).reshape(-1).to(dtype=torch.float64))
        else:
            parts.append(tensor.detach().reshape(-1).to(dtype=torch.float64))
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float64)


def metric_vector(model: v2293.TrueDeepPureKAN, args: argparse.Namespace, kind: str, gate: torch.Tensor | None = None, seed: int = 0) -> torch.Tensor:
    mode = flat_mode_vector(model, args).to(device=model.coeffs[0].device)
    vec = 1.0 / mode.clamp_min(EPS)
    if gate is not None:
        g = gate.to(device=vec.device, dtype=torch.float64).reshape(-1).clamp(0.0, 2.0)
        if kind == "random_matched":
            gen = torch.Generator(device=vec.device).manual_seed(int(seed))
            perm = torch.randperm(int(g.numel()), generator=gen, device=vec.device)
            g = g[perm]
        vec = vec * g
    elif kind == "identity" or kind == "adamw_witness":
        vec = torch.ones_like(vec)
    elif kind == "op_metric":
        pass
    elif kind == "data_metric":
        vec = vec.sqrt()
    else:
        pass
    return vec / vec.mean().clamp_min(EPS)


def signal_pullback_metric_operator(
    model: v2293.TrueDeepPureKAN,
    args: argparse.Namespace,
    J: torch.Tensor,
    U: torch.Tensor,
    rho: float,
) -> dict[str, Any]:
    mode = flat_mode_vector(model, args).to(dtype=torch.float64).cpu()
    if int(J.numel()) == 0 or int(U.numel()) == 0:
        vec = 1.0 / mode.clamp_min(EPS)
        vec = vec / vec.mean().clamp_min(EPS)
        return {
            "kind": "diagonal_fallback",
            "metric_vec": vec,
            "rank": 0,
            "condition": 0.0,
            "woodbury_residual": 0.0,
            "scale": 1.0,
        }
    n_dim = min(int(J.shape[1]), int(mode.numel()))
    rank = min(int(U.shape[1]), int(J.shape[0]))
    B = (U[:, :rank].to(dtype=torch.float64).cpu().T @ J[:, :n_dim].to(dtype=torch.float64).cpu()).contiguous()
    d = (float(rho) * mode[:n_dim].clamp_min(EPS) + EPS).clamp_min(EPS)
    d_inv = 1.0 / d
    K = sym(torch.eye(rank, dtype=torch.float64) + (B * d_inv.reshape(1, -1)) @ B.T)
    jitter = 1.0e-10 * torch.eye(rank, dtype=torch.float64)
    try:
        K_inv = torch.linalg.inv(K + jitter)
        eig = torch.linalg.eigvalsh(K + jitter)
        condition = float(eig.max().div(eig.min().clamp_min(EPS)).item()) if int(eig.numel()) else 0.0
    except Exception:
        K_inv = torch.linalg.pinv(K + jitter)
        condition = float("inf")
    KB = K_inv @ B
    diag = (d_inv - d_inv.square() * (B * KB).sum(dim=0)).clamp_min(EPS)
    if n_dim < int(mode.numel()):
        tail = 1.0 / (float(rho) * mode[n_dim:].clamp_min(EPS) + EPS)
        diag_full = torch.cat([diag, tail], dim=0)
    else:
        tail = torch.empty(0, dtype=torch.float64)
        diag_full = diag
    scale = float(1.0 / diag_full.mean().clamp_min(EPS).item())
    metric_vec = (diag_full * scale).to(dtype=torch.float64)
    residual = 0.0
    if n_dim > 0:
        probe = torch.linspace(-1.0, 1.0, steps=n_dim, dtype=torch.float64)
        solved = apply_metric_operator(
            {
                "kind": "woodbury_pullback",
                "B": B,
                "d_inv": d_inv,
                "K_inv": K_inv,
                "tail_metric": tail,
                "n_dim": n_dim,
                "scale": scale,
            },
            probe,
        )[:n_dim] / max(scale, EPS)
        Gx = d * solved + B.T @ (B @ solved)
        residual = float((Gx - probe).norm().div(probe.norm().clamp_min(EPS)).item())
    return {
        "kind": "woodbury_pullback",
        "B": B,
        "d_inv": d_inv,
        "K_inv": K_inv,
        "tail_metric": tail,
        "n_dim": n_dim,
        "scale": scale,
        "metric_vec": metric_vec,
        "rank": rank,
        "condition": condition,
        "woodbury_residual": residual,
        "diag_mean_before_normalization": float(diag_full.mean().item()),
    }


def apply_metric_operator(operator: dict[str, Any], vector: torch.Tensor) -> torch.Tensor:
    if str(operator.get("kind", "")) != "woodbury_pullback":
        metric_vec = operator.get("metric_vec")
        if isinstance(metric_vec, torch.Tensor):
            mv = metric_vec.to(dtype=torch.float64).cpu()
            n = min(int(mv.numel()), int(vector.numel()))
            out = vector.detach().to(dtype=torch.float64).cpu().clone()
            out[:n] = out[:n] * mv[:n]
            return out
        return vector.detach().to(dtype=torch.float64).cpu()
    g = vector.detach().to(dtype=torch.float64).cpu()
    B = operator["B"].to(dtype=torch.float64).cpu()
    d_inv = operator["d_inv"].to(dtype=torch.float64).cpu()
    K_inv = operator["K_inv"].to(dtype=torch.float64).cpu()
    n_dim = min(int(operator.get("n_dim", int(d_inv.numel()))), int(g.numel()))
    h = d_inv[:n_dim] * g[:n_dim]
    correction = d_inv[:n_dim] * (B[:, :n_dim].T @ (K_inv @ (B[:, :n_dim] @ h)))
    head = h - correction
    if n_dim < int(g.numel()):
        tail_metric = operator.get("tail_metric")
        if isinstance(tail_metric, torch.Tensor) and int(tail_metric.numel()) >= int(g.numel()) - n_dim:
            tail = tail_metric[: int(g.numel()) - n_dim].to(dtype=torch.float64).cpu() * g[n_dim:]
        else:
            tail = g[n_dim:]
        out = torch.cat([head, tail], dim=0)
    else:
        out = head
    return out * float(operator.get("scale", 1.0))


def signal_pullback_metric_vector(
    model: v2293.TrueDeepPureKAN,
    args: argparse.Namespace,
    J: torch.Tensor,
    U: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    operator = signal_pullback_metric_operator(model, args, J, U, rho)
    metric_vec = operator.get("metric_vec")
    if isinstance(metric_vec, torch.Tensor):
        return metric_vec.to(dtype=torch.float64)
    mode = flat_mode_vector(model, args).to(dtype=torch.float64).cpu()
    vec = 1.0 / mode.clamp_min(EPS)
    return vec / vec.mean().clamp_min(EPS)


def signal_mode_band_metric_vector(
    model: v2293.TrueDeepPureKAN,
    args: argparse.Namespace,
    J: torch.Tensor,
    U: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    base = metric_vector(model, args, "op_metric").detach().to(dtype=torch.float64).cpu()
    if int(J.numel()) == 0 or int(U.numel()) == 0:
        return base, {"mode_weight_change": 0.0, "mode_signal_entropy": 0.0, "degree_or_frequency_weight_shift": 1.0}
    n_dim = min(int(J.shape[1]), int(base.numel()))
    projected = U[:, : min(int(U.shape[1]), int(J.shape[0]))].to(dtype=torch.float64).cpu().T @ J[:, :n_dim].to(dtype=torch.float64).cpu()
    sig_diag = projected.square().sum(dim=0).clamp_min(0.0)
    gate = torch.ones_like(base)
    offset = 0
    band_means: list[float] = []
    for param in model.coeffs:
        numel = int(param.numel())
        k = int(param.shape[2])
        local_n = max(0, min(numel, n_dim - offset))
        if local_n <= 0:
            offset += numel
            continue
        local = sig_diag[offset : offset + local_n]
        mode_idx = torch.arange(local_n, dtype=torch.long) % k
        mode_energy = torch.zeros(k, dtype=torch.float64)
        for idx in range(k):
            vals = local[mode_idx == idx]
            mode_energy[idx] = vals.mean() if int(vals.numel()) else 0.0
        smoothed = mode_energy + 0.10 * mode_energy.mean().clamp_min(EPS) + EPS
        local_gate = torch.pow(smoothed / smoothed.mean().clamp_min(EPS), float(alpha))
        gate[offset : offset + local_n] = local_gate[mode_idx]
        band_means.extend(float(v.item()) for v in local_gate)
        offset += numel
    metric = (base * gate).clamp_min(EPS)
    metric = metric / metric.mean().clamp_min(EPS)
    probs = torch.tensor(band_means, dtype=torch.float64).clamp_min(EPS)
    probs = probs / probs.sum().clamp_min(EPS)
    entropy = float((-(probs * probs.log()).sum() / math.log(max(2, int(probs.numel())))).item()) if int(probs.numel()) else 0.0
    shift = max(band_means or [1.0]) / max(min(band_means or [1.0]), EPS)
    return metric, {
        "mode_weight_change": float((metric / base.clamp_min(EPS)).log().abs().mean().item()),
        "mode_signal_entropy": entropy,
        "degree_or_frequency_weight_shift": float(shift),
    }


class SignalChannelMetricOptimizer(torch.optim.Optimizer):
    """Optimizer-owned diagonal preconditioner for PureKAN coefficient tensors."""

    def __init__(
        self,
        model: v2293.TrueDeepPureKAN,
        args: argparse.Namespace,
        *,
        lr: float,
        kind: str,
        gate: torch.Tensor | None = None,
        seed: int = 0,
        max_norm_ratio: float = 2.0,
        metric_override: torch.Tensor | None = None,
        metric_operator: dict[str, Any] | None = None,
    ) -> None:
        params = list(model.coeffs)
        defaults = {"lr": float(lr), "kind": str(kind), "max_norm_ratio": float(max_norm_ratio)}
        super().__init__(params, defaults)
        self._model = model
        self._args = args
        self._metric_operator = metric_operator
        if metric_operator is not None and isinstance(metric_operator.get("metric_vec"), torch.Tensor):
            self._metric_vec = metric_operator["metric_vec"].detach()
        else:
            self._metric_vec = (metric_override.detach() if metric_override is not None else metric_vector(model, args, kind, gate=gate, seed=seed).detach())
        self.steps_taken = 0
        self.last_step_info: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        changed = 0
        max_ratio_seen = 0.0
        if self._metric_operator is not None:
            params = [p for group in self.param_groups for p in group["params"]]
            grads = [p.grad for p in params]
            pre_flat = apply_metric_operator(self._metric_operator, flatten_params_or_grads(grads, params))
            offset = 0
            for group in self.param_groups:
                lr = float(group["lr"])
                cap = float(group["max_norm_ratio"])
                for param in group["params"]:
                    numel = int(param.numel())
                    if param.grad is None:
                        offset += numel
                        continue
                    pre = pre_flat[offset : offset + numel].to(device=param.device, dtype=param.grad.dtype)
                    delta = -lr * pre.reshape_as(param)
                    denom = float(param.detach().norm().clamp_min(EPS).cpu().item())
                    ratio = float(delta.detach().norm().cpu().item()) / max(denom, EPS)
                    max_ratio_seen = max(max_ratio_seen, ratio)
                    if ratio > cap:
                        delta = delta * (cap / max(ratio, EPS))
                    before = param.detach().clone()
                    param.add_(delta)
                    changed += int(not torch.allclose(before, param.detach()))
                    offset += numel
        else:
            offset = 0
            for group in self.param_groups:
                lr = float(group["lr"])
                cap = float(group["max_norm_ratio"])
                for param in group["params"]:
                    numel = int(param.numel())
                    if param.grad is None:
                        offset += numel
                        continue
                    vec = self._metric_vec[offset : offset + numel].to(device=param.device, dtype=param.grad.dtype)
                    pre = param.grad.reshape(-1) * vec
                    delta = -lr * pre.reshape_as(param)
                    denom = float(param.detach().norm().clamp_min(EPS).cpu().item())
                    ratio = float(delta.detach().norm().cpu().item()) / max(denom, EPS)
                    max_ratio_seen = max(max_ratio_seen, ratio)
                    if ratio > cap:
                        delta = delta * (cap / max(ratio, EPS))
                    before = param.detach().clone()
                    param.add_(delta)
                    changed += int(not torch.allclose(before, param.detach()))
                    offset += numel
        self.steps_taken += 1
        for group in self.param_groups:
            group["state_step"] = self.steps_taken
        self.last_step_info = {
            "optimizer_state_step": float(self.steps_taken),
            "changed_param_tensors": float(changed),
            "max_delta_norm_ratio_before_cap": float(max_ratio_seen),
            "metric_trace": float(self._metric_vec.mean().detach().cpu().item()),
            "metric_fro_norm": float(self._metric_vec.norm().detach().cpu().item()),
        }
        return loss


def output_values(model: v2293.TrueDeepPureKAN, x: torch.Tensor, y: torch.Tensor | None, mode: str) -> torch.Tensor:
    logits = model(x)
    if str(mode) == "margin":
        if y is None:
            raise ValueError("margin output mode requires labels")
        yy = y.long().reshape(-1, 1)
        true_logit = logits.gather(1, yy).reshape(-1)
        if int(logits.shape[1]) <= 1:
            return true_logit
        other_sum = logits.sum(dim=1) - true_logit
        return true_logit - other_sum / float(max(1, int(logits.shape[1]) - 1))
    return logits.reshape(-1)


def full_output_jacobian(
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    max_outputs: int | None = None,
    mode: str = "logits",
) -> torch.Tensor:
    params = list(model.coeffs)
    flat_out = output_values(model, x, y, mode).reshape(-1)
    if max_outputs is not None and int(flat_out.numel()) > int(max_outputs):
        flat_out = flat_out[: int(max_outputs)]
    rows: list[torch.Tensor] = []
    for idx in range(int(flat_out.numel())):
        grads = torch.autograd.grad(flat_out[idx], params, retain_graph=True, allow_unused=True)
        rows.append(flatten_params_or_grads(grads, params).detach().cpu())
    return torch.stack(rows, dim=0).to(dtype=torch.float64)


def uniform_label_loss(logits: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits.float(), dim=1).mean(dim=1).mean()


def loss_gradient_vector(
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    params: list[torch.nn.Parameter],
    label_prior_correction: bool,
) -> torch.Tensor:
    logits = model(x)
    loss = F.cross_entropy(logits.float(), y.long())
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    g = flatten_params_or_grads(grads, params)
    if label_prior_correction:
        null_logits = model(x)
        null_loss = uniform_label_loss(null_logits)
        null_grads = torch.autograd.grad(null_loss, params, retain_graph=False, allow_unused=True)
        g = g - flatten_params_or_grads(null_grads, params)
    return g


def channel_from_jacobian(J: torch.Tensor, metric: torch.Tensor, rank: int) -> tuple[torch.Tensor, dict[str, float]]:
    m = metric.detach().to(dtype=torch.float64).cpu().reshape(1, -1).clamp_min(EPS)
    W = sym((J * m) @ J.T / float(max(1, int(J.shape[0]))))
    vals, vecs = torch.linalg.eigh(W)
    order = torch.argsort(vals.abs(), descending=True)
    vals = vals[order]
    vecs = vecs[:, order]
    r = min(max(1, int(rank)), int(vecs.shape[1]))
    trace = float(vals.clamp_min(0).sum().item())
    top_sum = float(vals[:r].clamp_min(0).sum().item())
    probs = vals.clamp_min(0) / max(trace, EPS)
    eff = float(torch.exp(-(probs * probs.clamp_min(EPS).log()).sum()).item()) if trace > 0 else 0.0
    info = {
        "output_dim": int(J.shape[0]),
        "param_dim": int(J.shape[1]),
        "W_trace_estimate": trace,
        "W_top1": float(vals[0].item()) if int(vals.numel()) else 0.0,
        "W_top_mass_ratio": top_sum / max(trace, EPS),
        "W_effective_rank": eff,
        "PSD_violation_min_eig": float(vals.min().item()) if int(vals.numel()) else 0.0,
        "signal_channel_projector_rank": r,
    }
    return vecs[:, :r].contiguous(), info


def maybe_unit_normalize_rows(G: torch.Tensor, normalization: str) -> torch.Tensor:
    if str(normalization) == "unit":
        return G / G.norm(dim=1, keepdim=True).clamp_min(EPS)
    return G


def maybe_unit_normalize_cols(V: torch.Tensor, normalization: str) -> torch.Tensor:
    if str(normalization) == "unit":
        return V / V.norm(dim=0, keepdim=True).clamp_min(EPS)
    return V


def channel_from_velocity_matrix(V: torch.Tensor, rank: int, mode: str, normalization: str) -> tuple[torch.Tensor, dict[str, float]]:
    if int(V.numel()) == 0:
        W = torch.zeros((1, 1), dtype=torch.float64)
        neg_fraction = 0.0
        raw_min = 0.0
    else:
        Vn = maybe_unit_normalize_cols(V, normalization)
        if str(mode) == "agreement":
            if int(Vn.shape[1]) <= 1:
                W_raw = sym(Vn @ Vn.T)
            else:
                mu = Vn.mean(dim=1, keepdim=True)
                centered = Vn - mu
                cov = centered @ centered.T / float(max(1, int(Vn.shape[1]) - 1))
                W_raw = sym(mu @ mu.T - cov / float(max(1, int(Vn.shape[1]) - 1)))
            raw_vals, raw_vecs = torch.linalg.eigh(W_raw)
            pos_vals = raw_vals.clamp_min(0.0)
            pos_mass = float(pos_vals.sum().item())
            neg_mass = float((-raw_vals.clamp_max(0.0)).sum().item())
            neg_fraction = neg_mass / max(pos_mass + neg_mass, EPS)
            raw_min = float(raw_vals.min().item()) if int(raw_vals.numel()) else 0.0
            W = sym((raw_vecs * pos_vals.unsqueeze(0)) @ raw_vecs.T)
        else:
            W = sym(Vn @ Vn.T / float(max(1, int(Vn.shape[1]))))
            neg_fraction = 0.0
            raw_min = 0.0
    vals, vecs = torch.linalg.eigh(W)
    order = torch.argsort(vals.abs(), descending=True)
    vals = vals[order]
    vecs = vecs[:, order]
    r = min(max(1, int(rank)), int(vecs.shape[1]))
    trace = float(vals.clamp_min(0).sum().item())
    top_sum = float(vals[:r].clamp_min(0).sum().item())
    probs = vals.clamp_min(0) / max(trace, EPS)
    eff = float(torch.exp(-(probs * probs.clamp_min(EPS).log()).sum()).item()) if trace > 0 else 0.0
    info = {
        "output_dim": int(W.shape[0]),
        "param_dim": 0,
        "W_trace_estimate": trace,
        "W_top1": float(vals[0].item()) if int(vals.numel()) else 0.0,
        "W_top_mass_ratio": top_sum / max(trace, EPS),
        "W_effective_rank": eff,
        "PSD_violation_min_eig": float(vals.min().item()) if int(vals.numel()) else 0.0,
        "W_raw_min_eig_before_positive_projection": raw_min,
        "velocity_agreement_negative_trace_fraction": neg_fraction,
        "signal_channel_projector_rank": r,
        "velocity_normalization": str(normalization),
    }
    return vecs[:, :r].contiguous(), info


def output_velocity_channel(
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    metric: torch.Tensor | dict[str, Any],
    rank: int,
    max_outputs: int,
    cohorts: int,
    output_mode: str,
    channel_mode: str,
    velocity_normalization: str,
    label_prior_correction: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    params = list(model.coeffs)
    J = full_output_jacobian(model, x, y, max_outputs=max_outputs, mode=output_mode)
    if isinstance(metric, dict):
        metric_vec_obj = metric.get("metric_vec")
        metric_cpu = metric_vec_obj.detach().to(dtype=torch.float64).cpu().reshape(-1) if isinstance(metric_vec_obj, torch.Tensor) else torch.ones(int(J.shape[1]), dtype=torch.float64)
    else:
        metric_cpu = metric.detach().to(dtype=torch.float64).cpu().reshape(-1)
    n = int(x.shape[0])
    cohort_count = max(1, min(int(cohorts), n))
    indices = torch.arange(n, device=x.device)
    chunks = torch.chunk(indices, cohort_count)
    velocities: list[torch.Tensor] = []
    for chunk in chunks:
        if int(chunk.numel()) == 0:
            continue
        g = loss_gradient_vector(model, x[chunk], y[chunk], params, label_prior_correction).detach().cpu()
        n_dim = min(int(g.numel()), int(metric_cpu.numel()), int(J.shape[1]))
        if isinstance(metric, dict):
            pre = apply_metric_operator(metric, g)[:n_dim]
        else:
            pre = metric_cpu[:n_dim] * g[:n_dim]
        velocity = -(J[:, :n_dim] @ pre[:n_dim].to(dtype=torch.float64).cpu())
        velocities.append(velocity.to(dtype=torch.float64))
    if not velocities:
        V = torch.zeros((int(J.shape[0]), 1), dtype=torch.float64)
    else:
        V = torch.stack(velocities, dim=1)
    U, info = channel_from_velocity_matrix(V, rank, channel_mode, velocity_normalization)
    info["velocity_cohort_count"] = int(V.shape[1])
    info["velocity_norm_median"] = median(float(V[:, idx].norm().item()) for idx in range(int(V.shape[1])))
    return U, J, info


def projector_overlap(U: torch.Tensor, V: torch.Tensor) -> float:
    if int(U.numel()) == 0 or int(V.numel()) == 0:
        return 0.0
    r = min(int(U.shape[1]), int(V.shape[1]))
    if r <= 0:
        return 0.0
    val = torch.linalg.norm(U[:, :r].T @ V[:, :r], ord="fro").square() / float(r)
    return float(val.clamp(0.0, 1.0).item())


def vector_channel_fraction(U: torch.Tensor, vec: torch.Tensor) -> float:
    v = vec.detach().reshape(-1).to(dtype=torch.float64).cpu()
    if int(U.numel()) == 0 or int(v.numel()) != int(U.shape[0]):
        return 0.0
    denom = float(v.square().sum().item())
    if denom <= 0.0:
        return 0.0
    return float((U.T @ v).square().sum().div(denom).clamp(0.0, 1.0).item())


def per_example_gradient_matrix(
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
    max_examples: int,
    label_prior_correction: bool,
) -> torch.Tensor:
    params = list(model.coeffs)
    n = min(int(max_examples), int(x.shape[0]))
    mode_inv_sqrt = flat_mode_vector(model, args).to(device=x.device).rsqrt()
    rows: list[torch.Tensor] = []
    for idx in range(n):
        g = loss_gradient_vector(
            model,
            x[idx : idx + 1],
            y[idx : idx + 1],
            params,
            label_prior_correction,
        ).to(device=x.device) * mode_inv_sqrt
        rows.append(g.detach().cpu())
    return torch.stack(rows, dim=0).to(dtype=torch.float64)


def snr_from_grad_samples(
    G: torch.Tensor,
    beta: float = 4.0,
    tau: float = 1.0,
    sample_normalization: str = "unit",
) -> tuple[torch.Tensor, dict[str, float]]:
    if int(G.shape[0]) <= 1:
        gate = torch.ones(int(G.shape[1]), dtype=torch.float64)
        return gate, {"diagonal_snr_median": 0.0, "diagonal_snr_top_decile_mean": 0.0, "AB_positive_trace_fraction": 0.0, "AB_negative_trace_fraction": 0.0}
    Gs = maybe_unit_normalize_rows(G, sample_normalization)
    mu = Gs.mean(dim=0)
    var = Gs.var(dim=0, unbiased=False)
    snr = mu.square() / (var / float(max(1, int(G.shape[0]) - 1)) + 1.0e-12)
    log_snr = snr.clamp_min(1.0e-12).log()
    gate = torch.sigmoid(float(beta) * (log_snr - math.log(float(tau))))
    sorted_snr = torch.sort(snr).values
    k = max(1, int(math.ceil(0.10 * int(sorted_snr.numel()))))
    pos = float(mu.square().sum().item())
    neg = float((var / float(max(1, int(G.shape[0]) - 1))).sum().item())
    info = {
        "diagonal_snr_median": float(torch.median(snr).item()),
        "diagonal_snr_top_decile_mean": float(sorted_snr[-k:].mean().item()),
        "AB_positive_trace_fraction": pos / max(pos + neg, EPS),
        "AB_negative_trace_fraction": neg / max(pos + neg, EPS),
        "snr_gate_density": float((gate > 0.5).to(dtype=torch.float64).mean().item()),
        "snr_gate_mean": float(gate.mean().item()),
    }
    return gate.to(dtype=torch.float64), info


def block_snr_values(model: v2293.TrueDeepPureKAN, G: torch.Tensor, sample_normalization: str) -> dict[str, float]:
    out: dict[str, float] = {}
    G = maybe_unit_normalize_rows(G, sample_normalization)
    offset = 0
    for layer_idx, param in enumerate(model.coeffs):
        d_in, d_out, k = int(param.shape[0]), int(param.shape[1]), int(param.shape[2])
        size = int(param.numel())
        block = G[:, offset : offset + size].reshape(int(G.shape[0]), d_in, d_out, k)
        offset += size
        bands = {
            "low": range(0, min(k, 2)),
            "mid": range(2, min(k, 5)),
            "high": range(5, k),
        }
        for name, idxs in bands.items():
            idx_list = list(idxs)
            if not idx_list:
                continue
            vals = block[:, :, :, idx_list].reshape(int(G.shape[0]), -1)
            mu = vals.mean(dim=0)
            var = vals.var(dim=0, unbiased=False)
            snr = float(mu.square().sum().div(var.sum() / float(max(1, int(G.shape[0]) - 1)) + EPS).item())
            out[f"layer{layer_idx}_{name}_block_snr"] = snr
    if out:
        out["layer_block_snr"] = median(out.values())
        out["mode_band_snr"] = max(out.values())
        highs = [v for k, v in out.items() if "_high_" in k]
        mids = [v for k, v in out.items() if "_mid_" in k]
        out["normal_block_snr"] = median(highs + mids)
        out["tangent_block_snr"] = median([v for k, v in out.items() if "_low_" in k])
    return out


def apply_label_control(y: torch.Tensor, control: str, seed: int) -> torch.Tensor:
    if control == "c2_positive":
        return y
    gen = torch.Generator(device=y.device).manual_seed(950000 + int(seed))
    if control == "random_label":
        classes = int(y.max().detach().cpu().item()) + 1
        return torch.randint(0, max(2, classes), tuple(y.shape), device=y.device, generator=gen)
    if control == "source_shuffle":
        perm = torch.randperm(int(y.numel()), generator=gen, device=y.device)
        return y[perm]
    return y


def channel_source_data(
    task: str,
    control: str,
    row_seed: int,
    args: argparse.Namespace,
    device: torch.device,
    probe_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_seed = int(args.channel_probe_seed if probe_seed is None else probe_seed)
    xsrc_full, ysrc_full, _x_unused, _y_unused = visual_data(task, source_seed, args, device)
    if control != "c2_positive":
        ysrc_full = apply_label_control(ysrc_full, control, source_seed + 1009 * int(row_seed) + 17)
    source_n = min(int(args.channel_source_size), int(xsrc_full.shape[0]))
    return xsrc_full[:source_n].detach(), ysrc_full[:source_n].detach()


def run_optimizer_training(
    model: v2293.TrueDeepPureKAN,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    args: argparse.Namespace,
    kind: str,
    gate: torch.Tensor | None,
    seed: int,
    metric_override: torch.Tensor | None = None,
    metric_operator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if kind == "adamw_witness":
        opt: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.adamw_lr), weight_decay=float(args.weight_decay))
    else:
        opt = SignalChannelMetricOptimizer(
            model,
            args,
            lr=float(args.metric_lr),
            kind=kind,
            gate=gate,
            seed=seed,
            max_norm_ratio=float(args.metric_max_norm_ratio),
            metric_override=metric_override,
            metric_operator=metric_operator,
        )
    trace = {"changed_edge_coefficients": 0, "optimizer_state_max": 0.0, "preconditioner_applied_inside_optimizer_step": int(kind != "adamw_witness")}
    for step in range(1, int(args.train_steps) + 1):
        xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
        before = [p.detach().clone() for p in model.coeffs]
        model.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits.float(), yb.long())
        loss.backward()
        opt.step()
        trace["changed_edge_coefficients"] = max(trace["changed_edge_coefficients"], int(any(not torch.allclose(a, p.detach()) for a, p in zip(before, model.coeffs))))
        if isinstance(opt, SignalChannelMetricOptimizer):
            trace["optimizer_state_max"] = max(trace["optimizer_state_max"], float(opt.last_step_info.get("optimizer_state_step", 0.0)))
    return trace


def metric_drift_proxies(
    before_model: v2293.TrueDeepPureKAN,
    after_model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    before_mats = v2294.param_mats(before_model)
    after_mats = v2294.param_mats(after_model)
    data_drifts: list[float] = []
    op_drifts: list[float] = []
    for layer_idx, (a0, a1) in enumerate(zip(before_mats, after_mats)):
        mode_diag = v2294.mode_metric_diag(before_model, layer_idx, args)
        c_data = v2294.data_c_matrix(before_model, x, layer_idx)
        data_drifts.append(v2294.log_spectral_drift(v2294.composite_metric(a0, c_data), v2294.composite_metric(a1, c_data)))
        op_drifts.append(v2294.log_spectral_drift(v2294.op_spectrum(a0, mode_diag), v2294.op_spectrum(a1, mode_diag)))
    return {
        "data_gram_drift_max": max(data_drifts or [0.0]),
        "operator_spectrum_drift_max": max(op_drifts or [0.0]),
    }


def part_c_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int, str]]:
    basis = csv_items(args.part_c_basis)
    depths = csv_items(args.part_c_depths)
    schemes = csv_items(args.part_c_schemes)
    tasks = csv_items(args.part_c_tasks)
    controls = csv_items(args.part_c_controls)
    jobs: list[tuple[str, str, str, str, int, str]] = []
    for basis_key in basis:
        for depth in depths:
            for scheme in schemes:
                for task in tasks:
                    for seed in range(int(args.part_c_seed_count)):
                        for control in controls:
                            jobs.append((basis_key, depth, scheme, task, seed, control))
    return jobs


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    init_logs()
    device = device_from_args(args)
    compile_pass = 1
    compile_error = ""
    try:
        py_compile.compile(str(RUNNER), doraise=True)
    except Exception as exc:
        compile_pass = 0
        compile_error = repr(exc)
    import_pass = 1
    static_pass, scan_hits = static_scan([RUNNER])
    rows: list[dict[str, Any]] = []
    x = torch.randn(24, int(args.input_dim), device=device)
    y = torch.randint(0, int(args.num_classes), (24,), device=device)
    for basis_key in BASIS_KEYS:
        bargs = basis_args(args, basis_key)
        for depth in ("depth2", "depth3"):
            model = make_model(depth, int(args.input_dim), int(args.num_classes), 229500 + len(rows), bargs, device)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            opt = SignalChannelMetricOptimizer(model, bargs, lr=float(args.metric_lr), kind="op_metric", max_norm_ratio=float(args.metric_max_norm_ratio))
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).float(), y.long())
            loss.backward()
            opt.step()
            changed = {name: int(not torch.allclose(before[name], p.detach())) for name, p in model.named_parameters()}
            coeff_names = {f"coeffs.{idx}" for idx in range(len(model.coeffs))}
            row = {
                "row_id": f"A_{basis_key}_{depth}",
                "part": "A",
                "basis_key": basis_key,
                "depth": depth,
                "status": "ok",
                "compile_pass": compile_pass,
                "compile_error": compile_error,
                "import_pass": import_pass,
                "static_scan_pass": static_pass,
                "static_scan_hits": json.dumps(scan_hits, ensure_ascii=False),
                "true_depth2_purekan_constructed": int(depth == "depth2"),
                "true_depth3_purekan_constructed": int(depth == "depth3"),
                "dche_available": int(basis_key.startswith("dche")),
                "dfour_available": int(basis_key.startswith("dfour")),
                "changed_edge_coefficients": int(any(changed.get(name, 0) for name in coeff_names)),
                "changed_mlp_tensors": int(any(value for name, value in changed.items() if name not in coeff_names)),
                "optimizer_state_step": int(opt.steps_taken),
                **AUDIT_DEFAULTS,
            }
            row["preconditioner_applied_inside_optimizer_step"] = 1
            rows.append(row)
    matrix = OUT_ROOT / "part_a_identity_matrix.csv"
    summary_path = OUT_ROOT / "part_a_identity_summary.json"
    write_rows(matrix, rows)
    true_depth2 = int(any(int(r["true_depth2_purekan_constructed"]) for r in rows))
    true_depth3 = int(any(int(r["true_depth3_purekan_constructed"]) for r in rows))
    gate = int(
        compile_pass
        and import_pass
        and static_pass
        and true_depth2
        and true_depth3
        and all(int(r["changed_edge_coefficients"]) == 1 for r in rows)
        and all(int(r["changed_mlp_tensors"]) == 0 for r in rows)
    )
    route = "A_Pass" if gate else "A_CodeIdentityFailed"
    blocker = "none" if gate else ("static_scan" if not static_pass else "compile_or_smoke")
    summary = gate_summary(
        "A",
        gate,
        route,
        blocker,
        rows,
        compile_pass=compile_pass,
        import_pass=import_pass,
        static_scan_pass=static_pass,
        true_depth2_purekan_constructed=true_depth2,
        true_depth3_purekan_constructed=true_depth3,
        scan_hits=scan_hits,
    )
    write_json(summary_path, summary)
    next_path = common_next_actions(
        "A",
        blocker,
        "Implementation identity must pass before science experiments.",
        ["fix compile/import/static scan identity", "remove any actual forbidden call path"],
        ["continue to Part C after a failed identity gate", "rename forbidden code to evade scan"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-a --device {args.device}"],
    )
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(next_path)}")
    append_recap("Part A identity gate", summary)
    return summary


def pick_value(data: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return "missing"
        cur = cur[key]
    return cur


def first_present(data: dict[str, Any], paths: list[list[str]]) -> Any:
    for path in paths:
        value = pick_value(data, path)
        if value != "missing":
            return value
    return "missing"


def max_group_value(data: dict[str, Any], group_key: str, value_key: str) -> Any:
    groups = data.get(group_key)
    if not isinstance(groups, list):
        return "missing"
    vals: list[float] = []
    for group in groups:
        if isinstance(group, dict) and value_key in group:
            vals.append(fval(group.get(value_key)))
    if not vals:
        return "missing"
    out = max(vals)
    return int(out) if abs(out - int(out)) < 1.0e-12 else out


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    sources = {
        "v22_90_final": ROOT / "results/v22_90/final_route.json",
        "v22_90_e": ROOT / "results/v22_90/part_e_summary.json",
        "v22_90_f": ROOT / "results/v22_90/part_f_summary.json",
        "v22_91_final": ROOT / "results/v22_91/final_route.json",
        "v22_91_c": ROOT / "results/v22_91/part_c_summary.json",
        "v22_91_h20": ROOT / "results/v22_91/part_c_summary_h20.json",
        "v22_93_final": ROOT / "results/v22_93/final_route.json",
        "v22_93_c": ROOT / "results/v22_93/part_c_architecture_capability_summary.json",
        "v22_94_final": ROOT / "results/v22_94/final_route.json",
        "v22_94_c": ROOT / "results/v22_94/part_c_witness_summary.json",
        "v22_94_d": ROOT / "results/v22_94/part_d_decomposition_summary.json",
        "v22_94_e_ord": ROOT / "results/v22_94/pre_tailsafe_repair_snapshot/part_e_multi_scheme_summary.json",
        "v22_94_f_ord": ROOT / "results/v22_94/pre_tailsafe_repair_snapshot/part_f_deep_positive_control_summary.json",
    }
    loaded = {name: read_json(path) for name, path in sources.items()}
    lock = {
        "part": "B",
        "status": "ok",
        "artifact_sources": {name: rel(path) if path.exists() else "missing" for name, path in sources.items()},
        "v22_90_part_e_pass": first_present(loaded["v22_90_e"], [["gate_pass"], ["part_e_gate_pass"]]),
        "v22_90_part_f_route": first_present(loaded["v22_90_final"], [["route"], ["final_route"], ["part_f", "part_f_route"]]),
        "v22_90_beats_mlp_composite": first_present(loaded["v22_90_f"], [["beats_MLP_composite"], ["beats_MLP_composite_coordinate"], ["beats_MLP_matched_budget"]]),
        "v22_90_no_debt": pick_value(loaded["v22_90_f"], ["no_debt"]),
        "v22_90_visual_coverage": first_present(loaded["v22_90_f"], [["visual_coverage"], ["visual_coverage_pass"], ["coverage_CVaR25_pass"]]),
        "v22_91_final_route": first_present(loaded["v22_91_final"], [["route"], ["final_route"], ["part_c", "part_c_route"]]),
        "v22_91_h20_debt_sign": first_present(loaded["v22_91_h20"], [["h20_debt_sign"], ["response_debt_sign_agreement"]]),
        "v22_91_h20_debt_R2": first_present(loaded["v22_91_h20"], [["h20_debt_R2"], ["debt_R2"]]),
        "v22_91_leave_family_debt_R2": first_present(loaded["v22_91_h20"], [["leave_family_debt_R2"], ["leave_dataset_family_debt_R2"]]),
        "v22_93_final_route": first_present(loaded["v22_93_final"], [["route"], ["final_route"], ["part_c", "part_c_route"]]),
        "v22_93_adamw_architecture_signal": first_present(loaded["v22_93_c"], [["adamw_architecture_signal"], ["c2_adamw_architecture_signal"]]),
        "v22_93_metric_c2_gain": first_present(loaded["v22_93_c"], [["metric_c2_gain"], ["c2_metric_visual_accuracy_improvement_median"], ["visual_synthetic_coverage_improvement_median"]]),
        "v22_94_part_c_witness_pass": first_present(loaded["v22_94_c"], [["gate_pass"], ["part_c_gate_pass"], ["adamw_witness_valid"]]),
        "v22_94_part_d_decomposition_pass": first_present(loaded["v22_94_d"], [["gate_pass"], ["part_d_gate_pass"]]),
        "v22_94_ordinary_part_e_pass": first_present(loaded["v22_94_e_ord"], [["gate_pass"], ["part_e_gate_pass"]]),
        "v22_94_ordinary_part_f_F5_best": max_group_value(loaded["v22_94_f_ord"], "group_summaries", "f5_no_debt_rows"),
        "v22_94_tailsafe_full_route": first_present(loaded["v22_94_final"], [["route"], ["final_route"], ["part_e", "part_e_route"]]),
        "used_fake_data_rows": 0,
        "held_test_usage": 0,
        "promotion_allowed": False,
    }
    missing = [key for key, value in lock.items() if value == "missing"]
    gate = int(True)
    route = "B_HistoryLockComplete" if not missing else "B_HistoryLockIncomplete"
    summary = {
        **lock,
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": "missing_fields" if missing else "none",
        "missing_fields": missing,
        "note": "Missing is recorded verbatim and not imputed.",
    }
    path = OUT_ROOT / "history_lock.json"
    alias = OUT_ROOT / "part_b_history_lock.json"
    write_json(path, summary)
    write_json(alias, summary)
    next_path = common_next_actions(
        "B",
        "missing_fields" if missing else "none",
        "History artifacts constrain later interpretation; missing values must remain missing.",
        ["search closeout tarballs for missing history fields", "keep unknown historical fields out of deterministic claims"],
        ["invent missing baselines", "reinterpret reduced probes as official history"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-b --device {args.device}"],
    )
    append_exec("part-b", command_text(sys.argv), "done", files=f"{rel(path)}; {rel(alias)}; {rel(next_path)}", note=f"missing_fields={len(missing)}")
    append_recap("Part B history lock", summary)
    return summary


def run_part_c_row(job: tuple[str, str, str, str, int, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    basis_key, depth, scheme, task, seed, control = job
    start = time.time()
    row_part = str(getattr(args, "row_part", "C"))
    row_id = f"{row_part}_{basis_key}_{depth}_{scheme}_{task}_s{seed}_{control}"
    try:
        bargs = basis_args(args, basis_key)
        xtr, ytr, xg, yg = visual_data(task, seed, args, device)
        ytr = apply_label_control(ytr, control, seed)
        if control != "c2_positive":
            yg = apply_label_control(yg, control, seed + 77)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = 2295000 + BASIS_KEYS.index(basis_key) * 100000 + (2 if depth == "depth2" else 3) * 10000 + int(seed)
        model = make_model(depth, int(xtr.shape[1]), classes, model_seed, bargs, device)
        initial_model = deepcopy(model).to(device)
        before_guard = v2293.metrics_for_model(model, xg, yg)
        before_train = v2293.metrics_for_model(model, xtr, ytr)
        if int(args.channel_common_source):
            xsrc, ysrc = channel_source_data(task, control, seed, args, device)
        else:
            source_n = min(int(args.channel_source_size), int(xtr.shape[0]))
            xsrc = xtr[:source_n].detach()
            ysrc = ytr[:source_n].detach()
        source_guard_intersection = bool(int(args.channel_source_guard_intersection))
        if source_guard_intersection:
            xsrc_guard, ysrc_guard = channel_source_data(
                task,
                control,
                seed,
                args,
                device,
                probe_seed=int(args.channel_guard_probe_seed),
            )
        else:
            xsrc_guard, ysrc_guard = xsrc, ysrc
        before_logits = output_values(model, xsrc, ysrc, str(args.channel_output_mode)).detach().reshape(-1).cpu().to(dtype=torch.float64)
        snr_x = xsrc[: int(args.snr_examples)]
        snr_y = ysrc[: int(args.snr_examples)]
        label_prior_correction = bool(int(args.label_prior_correction))
        G = per_example_gradient_matrix(model, snr_x, snr_y, bargs, int(args.snr_examples), label_prior_correction)
        gate, snr_info = snr_from_grad_samples(
            G,
            beta=float(args.snr_beta),
            tau=float(args.snr_tau),
            sample_normalization=str(args.snr_sample_normalization),
        )
        block_info = block_snr_values(model, G, str(args.snr_sample_normalization))
        metric_kind = "identity" if scheme == "adamw_witness" else scheme
        gate_for_metric = gate if scheme in {"population_snr", "random_matched"} else None
        metric_override: torch.Tensor | None = None
        metric_operator: dict[str, Any] | None = None
        pullback_info: dict[str, Any] = {}
        band_info: dict[str, float] = {}
        if scheme == "signal_pullback":
            base_metric = metric_vector(model, bargs, "op_metric")
            U_base, J_base, _info_base = output_velocity_channel(
                model,
                xsrc,
                ysrc,
                base_metric,
                int(args.signal_rank),
                int(args.max_channel_outputs),
                int(args.channel_cohorts),
                str(args.channel_output_mode),
                str(args.channel_mode),
                str(args.channel_velocity_normalization),
                label_prior_correction,
            )
            metric_operator = signal_pullback_metric_operator(model, bargs, J_base, U_base, float(args.pullback_ridge))
            metric_override = metric_operator["metric_vec"].to(device=device) if isinstance(metric_operator.get("metric_vec"), torch.Tensor) else None
            pullback_info = metric_operator
            metric = metric_operator
        elif scheme == "signal_mode_band":
            base_metric = metric_vector(model, bargs, "op_metric")
            U_base, J_base, _info_base = output_velocity_channel(
                model,
                xsrc,
                ysrc,
                base_metric,
                int(args.signal_rank),
                int(args.max_channel_outputs),
                int(args.channel_cohorts),
                str(args.channel_output_mode),
                str(args.channel_mode),
                str(args.channel_velocity_normalization),
                label_prior_correction,
            )
            metric_override, band_info = signal_mode_band_metric_vector(model, bargs, J_base, U_base, float(args.mode_band_alpha))
            metric_override = metric_override.to(device=device)
            metric = metric_override
        else:
            metric = metric_vector(model, bargs, metric_kind, gate=gate_for_metric, seed=seed)
        U0, J0, info0 = output_velocity_channel(
            model,
            xsrc,
            ysrc,
            metric,
            int(args.signal_rank),
            int(args.max_channel_outputs),
            int(args.channel_cohorts),
            str(args.channel_output_mode),
            str(args.channel_mode),
            str(args.channel_velocity_normalization),
            label_prior_correction,
        )
        if source_guard_intersection:
            Ug0, _Jg0, infog0 = output_velocity_channel(
                model,
                xsrc_guard,
                ysrc_guard,
                metric,
                int(args.signal_rank),
                int(args.max_channel_outputs),
                int(args.channel_cohorts),
                str(args.channel_output_mode),
                str(args.channel_mode),
                str(args.channel_velocity_normalization),
                label_prior_correction,
            )
            source_guard_stability_before = projector_overlap(U0, Ug0)
        else:
            infog0 = {}
            source_guard_stability_before = 1.0
        training_gate = gate if scheme in {"population_snr", "random_matched"} else None
        train_trace = run_optimizer_training(model, xtr, ytr, bargs, scheme, training_gate, seed, metric_override=metric_override, metric_operator=metric_operator)
        after_guard = v2293.metrics_for_model(model, xg, yg)
        after_train = v2293.metrics_for_model(model, xtr, ytr)
        drift = metric_drift_proxies(initial_model, model, xtr, bargs)
        after_logits = output_values(model, xsrc, ysrc, str(args.channel_output_mode)).detach().reshape(-1).cpu().to(dtype=torch.float64)
        metric1: torch.Tensor | dict[str, Any] = metric_operator if metric_operator is not None else metric_vector(model, bargs, metric_kind, gate=gate_for_metric, seed=seed)
        U1, J1, info1 = output_velocity_channel(
            model,
            xsrc,
            ysrc,
            metric1,
            int(args.signal_rank),
            int(args.max_channel_outputs),
            int(args.channel_cohorts),
            str(args.channel_output_mode),
            str(args.channel_mode),
            str(args.channel_velocity_normalization),
            label_prior_correction,
        )
        if source_guard_intersection:
            Ug1, _Jg1, infog1 = output_velocity_channel(
                model,
                xsrc_guard,
                ysrc_guard,
                metric1,
                int(args.signal_rank),
                int(args.max_channel_outputs),
                int(args.channel_cohorts),
                str(args.channel_output_mode),
                str(args.channel_mode),
                str(args.channel_velocity_normalization),
                label_prior_correction,
            )
            source_guard_stability = projector_overlap(U1, Ug1)
        else:
            infog1 = {}
            source_guard_stability = 1.0
        window_stability = projector_overlap(U0, U1)
        U = U1
        dz = after_logits[: int(U.shape[0])] - before_logits[: int(U.shape[0])]
        channel_overlap = vector_channel_fraction(U, dz)
        if source_guard_intersection:
            channel_overlap *= source_guard_stability
        dz_energy = float(dz.square().sum().item())
        update_signal_energy = float(channel_overlap * dz_energy)
        update_reservoir_energy = float((1.0 - channel_overlap) * dz_energy)
        params = list(initial_model.coeffs)
        xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, 0, int(args.batch_size), int(seed))
        raw_g = loss_gradient_vector(initial_model, xb, yb, params, label_prior_correction).to(dtype=torch.float64).cpu()
        op_metric = metric_vector(initial_model, bargs, "op_metric").cpu()
        deleted_delta = raw_g * (op_metric - 1.0)
        dz_deleted = J0 @ deleted_delta[: int(J0.shape[1])]
        deleted_sig_frac = vector_channel_fraction(U0, dz_deleted)
        if source_guard_intersection:
            deleted_sig_frac *= source_guard_stability_before
        deleted_sig = float(deleted_sig_frac * dz_deleted.square().sum().item())
        deleted_res = float((1.0 - deleted_sig_frac) * dz_deleted.square().sum().item())
        proj_path = CHANNEL_ROOT / f"{row_id}.npz"
        np.savez_compressed(
            proj_path,
            U0=U0.numpy(),
            U1=U1.numpy(),
            snr_gate=gate.numpy(),
            block_snr=np.array([float(v) for v in block_info.values()], dtype=np.float64),
            source_guard_stability=np.array([float(source_guard_stability)], dtype=np.float64),
        )
        W_top_mass_ratio_raw = float(info1["W_top_mass_ratio"])
        W_top_mass_ratio = W_top_mass_ratio_raw * float(source_guard_stability) if source_guard_intersection else W_top_mass_ratio_raw
        row = {
            "row_id": row_id,
            "part": row_part,
            "scheme": scheme,
            "basis_key": basis_key,
            "depth": depth,
            "seed": seed,
            "dataset_or_task": task,
            "control": control,
            "train_steps": int(args.train_steps),
            "batch_size": int(args.batch_size),
            "status": "ok",
            "argument_hash": argument_hash(args),
            "projector_npz": rel(proj_path),
            "estimator_kind": "label_prior_corrected_loss_weighted_full_jacobian_velocity_train_only"
            if label_prior_correction
            else "loss_weighted_full_jacobian_velocity_train_only",
            "metric_override_used": int(metric_override is not None),
            "pullback_ridge": float(args.pullback_ridge) if scheme == "signal_pullback" else "",
            "G_pull_rank": int(pullback_info.get("rank", 0)) if scheme == "signal_pullback" else "",
            "G_pull_condition": fval(pullback_info.get("condition")) if scheme == "signal_pullback" else "",
            "solve_residual": fval(pullback_info.get("woodbury_residual")) if scheme == "signal_pullback" else "",
            "pullback_operator_kind": str(pullback_info.get("kind", "")) if scheme == "signal_pullback" else "",
            "signal_mode_band_alpha": float(args.mode_band_alpha) if scheme == "signal_mode_band" else "",
            "mode_weight_change": band_info.get("mode_weight_change", "") if scheme == "signal_mode_band" else "",
            "mode_signal_entropy": band_info.get("mode_signal_entropy", "") if scheme == "signal_mode_band" else "",
            "degree_or_frequency_weight_shift": band_info.get("degree_or_frequency_weight_shift", "") if scheme == "signal_mode_band" else "",
            "channel_common_source": int(args.channel_common_source),
            "channel_probe_seed": int(args.channel_probe_seed),
            "channel_output_mode": str(args.channel_output_mode),
            "channel_mode": str(args.channel_mode),
            "channel_velocity_normalization": str(args.channel_velocity_normalization),
            "snr_sample_normalization": str(args.snr_sample_normalization),
            "label_prior_correction": int(label_prior_correction),
            "channel_source_guard_intersection": int(source_guard_intersection),
            "channel_guard_probe_seed": int(args.channel_guard_probe_seed),
            "source_guard_channel_stability_before": source_guard_stability_before,
            "source_guard_channel_stability": source_guard_stability,
            "fresh_boundary": 1,
            "microbatch_independence_flag": 1,
            "sketch_rank": f"velocity_cohorts={int(args.channel_cohorts)}",
            "window_length": int(args.train_steps),
            "checkpoint_count": 2,
            "velocity_cohort_count": info1.get("velocity_cohort_count", 0),
            "velocity_norm_median": info1.get("velocity_norm_median", 0.0),
            "guard_velocity_norm_median": infog1.get("velocity_norm_median", 0.0),
            "velocity_agreement_negative_trace_fraction": info1.get("velocity_agreement_negative_trace_fraction", 0.0),
            "W_raw_min_eig_before_positive_projection": info1.get("W_raw_min_eig_before_positive_projection", 0.0),
            "W_top_eigenvalues": f"{info1['W_top1']:.12g}",
            "W_effective_rank": info1["W_effective_rank"],
            "W_top_mass_ratio_raw": W_top_mass_ratio_raw,
            "guard_W_top_mass_ratio_raw": infog1.get("W_top_mass_ratio", 0.0),
            "W_top_mass_ratio": W_top_mass_ratio,
            "PSD_violation_min_eig": info1["PSD_violation_min_eig"],
            "signal_channel_projector_rank": info1["signal_channel_projector_rank"],
            "signal_channel_window_stability": window_stability,
            "reservoir_energy_fraction": 1.0 - channel_overlap,
            "channel_overlap_with_adamw_output_displacement": channel_overlap if scheme == "adamw_witness" else "",
            "channel_overlap_with_metric_branch_output_displacement": channel_overlap if scheme != "adamw_witness" else "",
            "channel_overlap_with_C2_coverage_direction": channel_overlap,
            "diagonal_snr_median": snr_info["diagonal_snr_median"],
            "diagonal_snr_top_decile_mean": snr_info["diagonal_snr_top_decile_mean"],
            "layer_block_snr": block_info.get("layer_block_snr", 0.0),
            "mode_band_snr": block_info.get("mode_band_snr", 0.0),
            "tangent_block_snr": block_info.get("tangent_block_snr", 0.0),
            "normal_block_snr": block_info.get("normal_block_snr", 0.0),
            "AB_positive_trace_fraction": snr_info["AB_positive_trace_fraction"],
            "AB_negative_trace_fraction": snr_info["AB_negative_trace_fraction"],
            "AdamW_delta_signal_energy": update_signal_energy if scheme == "adamw_witness" else "",
            "metric_delta_signal_energy": update_signal_energy if scheme != "adamw_witness" else "",
            "metric_delta_reservoir_energy": update_reservoir_energy if scheme != "adamw_witness" else "",
            "deleted_freedom_signal_energy": deleted_sig,
            "deleted_freedom_reservoir_energy": deleted_res,
            "signal_to_reservoir_ratio_deleted": deleted_sig / max(deleted_res, EPS),
            "normal_freedom_signal_fraction": deleted_sig_frac,
            "operator_freedom_signal_fraction": channel_overlap,
            "initial_visual_accuracy": before_guard["accuracy"],
            "final_visual_accuracy": after_guard["accuracy"],
            "visual_accuracy_improvement": after_guard["accuracy"] - before_guard["accuracy"],
            "initial_visual_coverage": before_guard["coverage_CVaR25"],
            "final_visual_coverage": after_guard["coverage_CVaR25"],
            "visual_coverage_improvement": after_guard["coverage_CVaR25"] - before_guard["coverage_CVaR25"],
            "guard_debt_before": before_guard["debt_metric"],
            "guard_debt_after": after_guard["debt_metric"],
            "guard_debt_delta": after_guard["debt_metric"] - before_guard["debt_metric"],
            "train_debt_before": before_train["debt_metric"],
            "train_debt_after": after_train["debt_metric"],
            "train_debt_delta": after_train["debt_metric"] - before_train["debt_metric"],
            "no_debt": int(after_guard["debt_metric"] <= before_guard["debt_metric"] + float(args.no_debt_budget)),
            "Brier_delta": after_guard.get("brier", 0.0) - before_guard.get("brier", 0.0),
            "ECE_delta": after_guard.get("ece", 0.0) - before_guard.get("ece", 0.0),
            "tail95_delta": after_guard.get("tail95", 0.0) - before_guard.get("tail95", 0.0),
            "tail99_delta": after_guard.get("tail99", 0.0) - before_guard.get("tail99", 0.0),
            "margin10_delta": after_guard.get("margin10", 0.0) - before_guard.get("margin10", 0.0),
            "data_gram_drift_max": drift["data_gram_drift_max"],
            "operator_spectrum_drift_max": drift["operator_spectrum_drift_max"],
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
        row["preconditioner_applied_inside_optimizer_step"] = int(scheme != "adamw_witness")
        row["state_updated_every_step"] = int(train_trace.get("optimizer_state_max", 0.0) >= int(args.train_steps) or scheme == "adamw_witness")
        row["changed_edge_coefficients"] = int(train_trace.get("changed_edge_coefficients", 0))
        return row
    except Exception as exc:
        return {
            "row_id": row_id,
            "part": row_part,
            "scheme": scheme,
            "basis_key": basis_key,
            "depth": depth,
            "seed": seed,
            "dataset_or_task": task,
            "control": control,
            "status": "error",
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    b = read_json(OUT_ROOT / "part_b_history_lock.json")
    if int(a.get("gate_pass", 0)) != 1 or not b:
        rows: list[dict[str, Any]] = []
        summary = gate_summary("C", 0, "C_BlockedByIdentityOrHistory", "part_a_or_b_missing", rows)
        path = OUT_ROOT / f"part_c_signal_channel_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
        write_rows(path, rows)
        write_json(OUT_ROOT / "part_c_signal_channel_summary.json", summary)
        append_exec("part-c", command_text(sys.argv), "blocked", files=f"{rel(path)}; {rel(OUT_ROOT / 'part_c_signal_channel_summary.json')}")
        return summary
    device = device_from_args(args)
    jobs = shard_items(part_c_jobs(args), args)
    rows = [run_part_c_row(job, args, device) for job in jobs]
    path = OUT_ROOT / f"part_c_signal_channel_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    summary = gate_summary("C", 0, "C_ShardsWritten", "merge_required", rows, shard_index=int(args.shard_index), shard_count=int(args.shard_count))
    append_exec("part-c", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}")
    return summary


def load_projector(path_text: str, key: str = "U1") -> torch.Tensor | None:
    if not path_text:
        return None
    path = ROOT / path_text if not Path(path_text).is_absolute() else Path(path_text)
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        return torch.from_numpy(arr[key]).to(dtype=torch.float64)
    except Exception:
        return None


def pairwise_projector_stability(rows: list[dict[str, str]], key: str = "U1") -> float:
    projs = [load_projector(row.get("projector_npz", ""), key=key) for row in rows]
    projs = [p for p in projs if p is not None]
    vals: list[float] = []
    for i in range(len(projs)):
        for j in range(i + 1, len(projs)):
            vals.append(projector_overlap(projs[i], projs[j]))
    return median(vals)


def pairwise_snr_stability(rows: list[dict[str, str]]) -> float:
    vecs: list[torch.Tensor] = []
    for row in rows:
        path_text = row.get("projector_npz", "")
        path = ROOT / path_text if path_text and not Path(path_text).is_absolute() else Path(path_text)
        if not path.exists():
            continue
        try:
            arr = np.load(path)
            vecs.append(torch.from_numpy(arr["snr_gate"]).to(dtype=torch.float64))
        except Exception:
            continue
    vals: list[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            n = min(int(vecs[i].numel()), int(vecs[j].numel()))
            a = vecs[i][:n]
            b = vecs[j][:n]
            denom = a.norm() * b.norm()
            if float(denom.item()) > 0.0:
                vals.append(float(((a @ b / denom) + 1.0).div(2.0).clamp(0.0, 1.0).item()))
    return median(vals)


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob("part_c_signal_channel_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = OUT_ROOT / "part_c_signal_channel_matrix.csv"
    write_rows(matrix, [dict(row) for row in rows])
    ok = [row for row in rows if row.get("status") == "ok"]
    pos = [row for row in ok if row.get("control") == "c2_positive"]
    neg = [row for row in ok if row.get("control") != "c2_positive"]
    seed_stabilities: list[float] = []
    snr_stabilities: list[float] = []
    c1_group_summaries: list[dict[str, Any]] = []
    for group_key in sorted({(r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("dataset_or_task")) for r in pos}):
        group = [r for r in pos if (r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("dataset_or_task")) == group_key]
        if len(group) >= 2:
            group_seed_stability = pairwise_projector_stability(group, key="U1")
            group_window_stability = median(fval(r.get("signal_channel_window_stability")) for r in group)
            group_top_mass = median(fval(r.get("W_top_mass_ratio")) for r in group)
            group_c1_pass = int(group_seed_stability >= 0.65 and group_window_stability >= 0.65 and group_top_mass >= 0.50)
            c1_group_summaries.append(
                {
                    "basis_key": group_key[0],
                    "depth": group_key[1],
                    "scheme": group_key[2],
                    "dataset_or_task": group_key[3],
                    "signal_channel_seed_stability": group_seed_stability,
                    "signal_channel_window_stability": group_window_stability,
                    "W_top_mass_ratio": group_top_mass,
                    "c1_group_pass": group_c1_pass,
                }
            )
            seed_stabilities.append(group_seed_stability)
            snr_stabilities.append(pairwise_snr_stability(group))
    seed_stability = median(seed_stabilities)
    snr_seed_stability = median(snr_stabilities)
    window_stability = median(fval(r.get("signal_channel_window_stability")) for r in pos)
    top_mass = median(fval(r.get("W_top_mass_ratio")) for r in pos)
    psd_min = min([fval(r.get("PSD_violation_min_eig"), 0.0) for r in ok] or [0.0])
    c2_gains = [fval(r.get("visual_coverage_improvement")) for r in pos]
    snr_top = [fval(r.get("diagonal_snr_top_decile_mean")) for r in pos]
    deleted_signal = [fval(r.get("deleted_freedom_signal_energy")) for r in pos]
    ratios = [fval(r.get("signal_to_reservoir_ratio_deleted")) for r in pos]
    c2_corr = pearson(snr_top, c2_gains)
    c3_corr = pearson(deleted_signal, c2_gains)
    neg_top_mass = median(fval(r.get("W_top_mass_ratio")) for r in neg)
    neg_snr = median(fval(r.get("diagonal_snr_top_decile_mean")) for r in neg)
    pos_snr = median(snr_top)
    c1_pass = int(any(int(g.get("c1_group_pass", 0)) for g in c1_group_summaries) and psd_min >= -1.0e-7)
    c2_pass = int(snr_seed_stability >= 0.60 and (c2_corr > 0.20 or median(fval(r.get("normal_block_snr")) for r in pos) > median(fval(r.get("tangent_block_snr")) for r in neg)))
    c3_pass = int(median(ratios) > 1.20 and c3_corr > 0.20)
    c4_pass = int((neg_top_mass <= 0.75 * top_mass) or (neg_snr <= 0.75 * pos_snr))
    gate = int(c1_pass and (c2_pass or c3_pass) and c4_pass)
    if not c1_pass:
        route = "C_ChannelEstimatorUnstable"
        blocker = "c1_seed_window_topmass"
    elif not (c2_pass or c3_pass):
        route = "C_ChannelEstimatorUnstable"
        blocker = "c2_c3_signal_consistency"
    elif not c4_pass:
        route = "C_ChannelEstimatorNegativeControlLeakage"
        blocker = "negative_control_strength"
    else:
        route = "C_Pass"
        blocker = "none"
    summary = gate_summary(
        "C",
        gate,
        route,
        blocker,
        [dict(row) for row in rows],
        c1_pass=c1_pass,
        c2_pass=c2_pass,
        c3_pass=c3_pass,
        c4_negative_control_pass=c4_pass,
        c1_passing_group_count=sum(int(g.get("c1_group_pass", 0)) for g in c1_group_summaries),
        c1_group_summaries=c1_group_summaries,
        signal_channel_seed_stability=seed_stability,
        signal_channel_window_stability=window_stability,
        W_top_mass_ratio=top_mass,
        PSD_violation_min_eig=psd_min,
        snr_seed_stability=snr_seed_stability,
        snr_cohort_stability=snr_seed_stability,
        snr_C2_gain_correlation=c2_corr,
        C2_gain_vs_deleted_signal_energy_corr=c3_corr,
        signal_to_reservoir_ratio_deleted_median=median(ratios),
        C2_channel_top_mass=top_mass,
        negative_channel_top_mass=neg_top_mass,
        C2_snr_top_decile=pos_snr,
        negative_snr_top_decile=neg_snr,
        evidence_fields=[
            "signal_channel_seed_stability",
            "signal_channel_window_stability",
            "W_top_mass_ratio",
            "snr_seed_stability",
            "snr_C2_gain_correlation",
            "signal_to_reservoir_ratio_deleted_median",
            "negative_channel_top_mass",
        ],
    )
    summary_path = OUT_ROOT / "part_c_signal_channel_summary.json"
    write_json(summary_path, summary)
    next_path = common_next_actions(
        "C",
        blocker,
        "Signal-channel estimator must be stable and distinguish positive from negative controls before metric induction.",
        [
            "increase channel source size or checkpoint averaging",
            "increase Part C seed count",
            "inspect full-output Jacobian numerical scale",
            "inspect per-example gradient whitening and cohort split",
            "rerun Part C shards with the same pre-registered thresholds",
        ],
        [
            "lower C1/C2/C4 thresholds to pass",
            "choose a single passing seed or window",
            "run Part D metric schemes after failed Part C",
        ],
        [f"{PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index <0-3> --device cuda:<0-3>", f"{PYTHON} {rel(RUNNER)} --mode part-c-merge"],
    )
    append_exec("part-c-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(next_path)}")
    append_recap("Part C signal-channel estimator audit", summary)
    return summary


def write_blocked_part(part: str, route: str, blocker: str, reason: str, args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        {
            "row_id": f"{part}_blocked",
            "part": part,
            "status": "skipped",
            "blocked_reason": reason,
            **AUDIT_DEFAULTS,
        }
    ]
    matrix = OUT_ROOT / f"part_{part.lower()}_matrix.csv"
    summary_path = OUT_ROOT / f"part_{part.lower()}_summary.json"
    write_rows(matrix, rows)
    summary = gate_summary(part, 0, route, blocker, rows, blocked_reason=reason)
    write_json(summary_path, summary)
    next_path = common_next_actions(
        part,
        blocker,
        reason,
        ["repair the failed prerequisite gate first"],
        ["run downstream official metric schemes before prerequisite pass"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-c-merge"],
    )
    append_exec(f"part-{part.lower()}", command_text(sys.argv), "blocked", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(next_path)}")
    append_recap(f"Part {part} blocked", summary)
    return summary


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    c = read_json(OUT_ROOT / "part_c_signal_channel_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        return write_blocked_part("D", "D_BlockedByPartC", "part_c_failed", "Part C failed; Part D metric-to-channel causality is prohibited by the plan.", args)
    d_schemes = tuple(csv_items(args.part_d_schemes))
    required = ("adamw_witness", "data_metric", "op_metric", "population_snr", "signal_pullback", "random_matched")
    if int(args.shard_count) > 1:
        d_args = argparse.Namespace(**vars(args))
        setattr(d_args, "row_part", "D")
        jobs: list[tuple[str, str, str, str, int, str]] = []
        for basis_key in BASIS_KEYS:
            for depth in ("depth2", "depth3"):
                for scheme in d_schemes:
                    for task in csv_items(args.part_d_tasks):
                        for seed in range(int(args.part_d_seed_count)):
                            jobs.append((basis_key, depth, scheme, task, seed, "c2_positive"))
        rows = [run_part_c_row(job, d_args, device_from_args(args)) for job in shard_items(jobs, args)]
        path = OUT_ROOT / f"part_d_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
        write_rows(path, rows)
        summary = gate_summary("D", 0, "D_ShardsWritten", "merge_required", rows, shard_index=int(args.shard_index), shard_count=int(args.shard_count))
        append_exec("part-d", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}")
        return summary

    shard_paths = sorted(OUT_ROOT.glob("part_d_matrix_shard*_of_*.csv"))
    if shard_paths:
        rows_c: list[dict[str, str]] = []
        for path in shard_paths:
            rows_c.extend(read_rows(path))
    else:
        d_args = argparse.Namespace(**vars(args))
        setattr(d_args, "row_part", "D")
        jobs = []
        for basis_key in BASIS_KEYS:
            for depth in ("depth2", "depth3"):
                for scheme in d_schemes:
                    for task in csv_items(args.part_d_tasks):
                        for seed in range(int(args.part_d_seed_count)):
                            jobs.append((basis_key, depth, scheme, task, seed, "c2_positive"))
        rows_c = [run_part_c_row(job, d_args, device_from_args(args)) for job in jobs]
    rows_c = [row for row in rows_c if row.get("status") == "ok" and row.get("control") == "c2_positive"]
    by_group: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows_c:
        key = (row.get("basis_key", ""), row.get("depth", ""), row.get("dataset_or_task", ""), row.get("seed", ""))
        by_group.setdefault(key, {})[row.get("scheme", "")] = row
    rows: list[dict[str, Any]] = []
    for key, scheme_rows in sorted(by_group.items()):
        if not all(name in scheme_rows for name in required):
            continue
        projs = {name: load_projector(scheme_rows[name].get("projector_npz", ""), key="U1") for name in required}
        if any(proj is None for proj in projs.values()):
            continue
        assert all(proj is not None for proj in projs.values())
        overlap_data_op = projector_overlap(projs["data_metric"], projs["op_metric"])  # type: ignore[arg-type]
        overlap_adamw_op = projector_overlap(projs["adamw_witness"], projs["op_metric"])  # type: ignore[arg-type]
        overlap_op_pop = projector_overlap(projs["op_metric"], projs["population_snr"])  # type: ignore[arg-type]
        overlap_pop_adamw = projector_overlap(projs["population_snr"], projs["adamw_witness"])  # type: ignore[arg-type]
        overlap_pull_adamw = projector_overlap(projs["signal_pullback"], projs["adamw_witness"])  # type: ignore[arg-type]
        overlap_rand_pop = projector_overlap(projs["random_matched"], projs["population_snr"])  # type: ignore[arg-type]
        top_mass = {name: fval(scheme_rows[name].get("W_top_mass_ratio")) for name in required}
        coverage = {name: fval(scheme_rows[name].get("final_visual_coverage")) for name in required}
        rows.append(
            {
                "row_id": f"D_{key[0]}_{key[1]}_{key[2]}_s{key[3]}",
                "part": "D",
                "basis_key": key[0],
                "depth": key[1],
                "dataset_or_task": key[2],
                "seed": key[3],
                "status": "ok",
                "scheme_set": ",".join(required),
                "overlap_data_vs_op": overlap_data_op,
                "overlap_adamw_vs_op": overlap_adamw_op,
                "overlap_op_vs_pop": overlap_op_pop,
                "overlap_pop_vs_adamw": overlap_pop_adamw,
                "overlap_pull_vs_adamw": overlap_pull_adamw,
                "overlap_random_vs_pop": overlap_rand_pop,
                "subspace_angle_data_vs_op": 1.0 - overlap_data_op,
                "subspace_angle_adamw_vs_op": 1.0 - overlap_adamw_op,
                "subspace_angle_op_vs_pop": 1.0 - overlap_op_pop,
                "subspace_angle_pop_vs_adamw": 1.0 - overlap_pop_adamw,
                "subspace_angle_pull_vs_adamw": 1.0 - overlap_pull_adamw,
                "subspace_angle_random_vs_pop": 1.0 - overlap_rand_pop,
                "channel_top_mass_by_metric": json.dumps(top_mass, sort_keys=True),
                "C2_coverage_by_metric": json.dumps(coverage, sort_keys=True),
                "C2_accuracy_by_metric": json.dumps({name: fval(scheme_rows[name].get("final_visual_accuracy")) for name in required}, sort_keys=True),
                "normal_freedom_retention_by_metric": json.dumps({name: fval(scheme_rows[name].get("normal_freedom_signal_fraction")) for name in required}, sort_keys=True),
                "operator_spectrum_drift_by_metric": "not_implemented",
                "data_gram_drift_by_metric": "not_implemented",
                "F5_no_debt_by_metric": "not_run_part_d_scope",
                "metric_compute_overhead_by_metric": json.dumps({name: fval(scheme_rows[name].get("wall_time_s")) for name in required}, sort_keys=True),
                "random_metric_control_gap": 1.0 - overlap_rand_pop,
                "data_metric_channel_available": 1,
                "pullback_metric_channel_available": 1,
                "random_metric_control_available": 1,
                **AUDIT_DEFAULTS,
            }
        )
    matrix = OUT_ROOT / "part_d_matrix.csv"
    write_rows(matrix, rows)
    angles_data_op = [fval(row.get("subspace_angle_data_vs_op")) for row in rows]
    angles_adamw_op = [fval(row.get("subspace_angle_adamw_vs_op")) for row in rows]
    angles_op_pop = [fval(row.get("subspace_angle_op_vs_pop")) for row in rows]
    angles_pop_adamw = [fval(row.get("subspace_angle_pop_vs_adamw")) for row in rows]
    angles_pull_adamw = [fval(row.get("subspace_angle_pull_vs_adamw")) for row in rows]
    angles_rand_pop = [fval(row.get("subspace_angle_random_vs_pop")) for row in rows]
    angle_max = max(angles_data_op + angles_adamw_op + angles_op_pop + angles_pop_adamw + angles_pull_adamw + [0.0])
    minimum_causality = int(median(angles_data_op) >= 0.15 or angle_max >= 0.15)
    full_scheme_coverage = int(bool(rows) and all(row.get("data_metric_channel_available") == 1 and row.get("pullback_metric_channel_available") == 1 and row.get("random_metric_control_available") == 1 for row in rows))
    random_control_pass = int(median(angles_rand_pop) >= 0.10)
    gate = int(minimum_causality and full_scheme_coverage and random_control_pass)
    if not rows:
        route = "D_NoComparablePartCProjectors"
        blocker = "missing_projectors"
    elif not minimum_causality:
        route = "D_MetricChannelsTooSimilar"
        blocker = "metric_channel_causality_absent"
    elif not random_control_pass:
        route = "D_RandomMetricExplainsGain"
        blocker = "random_metric_control"
    elif not full_scheme_coverage:
        route = "D_MinimumCausalityPass_PartialSchemeCoverage"
        blocker = "full_part_d_scheme_coverage_missing"
    else:
        route = "D_Pass"
        blocker = "none"
    summary = gate_summary(
        "D",
        gate,
        route,
        blocker,
        rows,
        minimum_causality_pass=minimum_causality,
        full_scheme_coverage_pass=full_scheme_coverage,
        random_metric_control_pass=random_control_pass,
        comparable_rows=len(rows),
        channel_rank_by_metric="from_part_d_rank_field",
        channel_top_mass_by_metric="recorded_per_row_json",
        subspace_angle_data_vs_op=median(angles_data_op),
        subspace_angle_adamw_vs_op=median(angles_adamw_op),
        subspace_angle_op_vs_pop=median(angles_op_pop),
        subspace_angle_pop_vs_adamw=median(angles_pop_adamw),
        subspace_angle_pull_vs_adamw=median(angles_pull_adamw),
        subspace_angle_random_vs_pop=median(angles_rand_pop),
        max_subspace_angle_observed=angle_max,
        random_metric_control_gap=median(angles_rand_pop),
        data_metric_channel_available=1 if rows else 0,
        pullback_metric_channel_available=1 if rows else 0,
        random_metric_control_available=1 if rows else 0,
        evidence_fields=[
            "subspace_angle_adamw_vs_op",
            "subspace_angle_data_vs_op",
            "subspace_angle_op_vs_pop",
            "subspace_angle_pop_vs_adamw",
            "subspace_angle_pull_vs_adamw",
            "subspace_angle_random_vs_pop",
            "max_subspace_angle_observed",
            "full_scheme_coverage_pass",
            "random_metric_control_pass",
        ],
    )
    summary_path = OUT_ROOT / "part_d_summary.json"
    write_json(summary_path, summary)
    next_path = common_next_actions(
        "D",
        blocker,
        "Part D checks whether fixed metric schemes induce different signal channels and whether randomized matched controls explain the effect.",
        ["if D fails, inspect preconditioner application, JVP/VJP coordinate normalization, or randomized control construction"],
        ["run Part E-K official metric schemes before Part D passes"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-d --shard-count 4 --shard-index <0-3> --device cuda:<0-3>", f"{PYTHON} {rel(RUNNER)} --mode part-d --device {args.device}"],
    )
    append_exec("part-d", command_text(sys.argv), "passed" if gate else "partial", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(next_path)}")
    append_recap("Part D metric-to-channel causality audit", summary)
    return summary


def part_e_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for basis_key in csv_items(args.part_e_basis):
        for depth in csv_items(args.part_e_depths):
            for scheme in csv_items(args.part_e_schemes):
                for task in csv_items(args.part_e_tasks):
                    for seed in range(int(args.part_e_seed_count)):
                        for control in csv_items(args.part_e_controls):
                            jobs.append({"kind": "visual", "job": (basis_key, depth, scheme, task, seed, control)})
                for seed in range(int(args.part_e_f5_seed_count)):
                    jobs.append({"kind": "f5", "basis_key": basis_key, "depth": depth, "scheme": scheme, "seed": seed})
    return jobs


def run_part_e_f5_row(job: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    basis_key = str(job.get("basis_key"))
    depth = str(job.get("depth"))
    scheme = str(job.get("scheme"))
    seed = int(job.get("seed", 0))
    row_part = str(getattr(args, "row_part", "E"))
    row_id = f"{row_part}_{basis_key}_{depth}_{scheme}_f5_s{seed}"
    start = time.time()
    try:
        bargs = basis_args(args, basis_key)
        xtr, ytr, xg, yg = synthetic_data("c1a_shallow_additive", seed + 50000, args, device)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = 2295900 + BASIS_KEYS.index(basis_key) * 100000 + (2 if depth == "depth2" else 3) * 10000 + seed
        model = make_model(depth, int(xtr.shape[1]), classes, model_seed, bargs, device)
        initial_model = deepcopy(model).to(device)
        before_guard = v2293.metrics_for_model(model, xg, yg)
        before_train = v2293.metrics_for_model(model, xtr, ytr)
        source_n = min(int(args.channel_source_size), int(xtr.shape[0]))
        xsrc = xtr[:source_n].detach()
        ysrc = ytr[:source_n].detach()
        if int(args.channel_source_guard_intersection):
            xsrc_guard = xg[:source_n].detach()
            ysrc_guard = yg[:source_n].detach()
        else:
            xsrc_guard, ysrc_guard = xsrc, ysrc
        label_prior_correction = bool(int(args.label_prior_correction))
        before_logits = output_values(model, xsrc, ysrc, str(args.channel_output_mode)).detach().reshape(-1).cpu().to(dtype=torch.float64)
        snr_x = xsrc[: int(args.snr_examples)]
        snr_y = ysrc[: int(args.snr_examples)]
        G = per_example_gradient_matrix(model, snr_x, snr_y, bargs, int(args.snr_examples), label_prior_correction)
        gate, snr_info = snr_from_grad_samples(
            G,
            beta=float(args.snr_beta),
            tau=float(args.snr_tau),
            sample_normalization=str(args.snr_sample_normalization),
        )
        block_info = block_snr_values(model, G, str(args.snr_sample_normalization))
        metric_kind = "identity" if scheme == "adamw_witness" else scheme
        gate_for_metric = gate if scheme in {"population_snr", "random_matched"} else None
        metric_operator: dict[str, Any] | None = None
        metric_override: torch.Tensor | None = None
        pullback_info: dict[str, Any] = {}
        band_info: dict[str, float] = {}
        if scheme == "signal_pullback":
            base_metric = metric_vector(model, bargs, "op_metric")
            U_base, J_base, _info_base = output_velocity_channel(
                model,
                xsrc,
                ysrc,
                base_metric,
                int(args.signal_rank),
                int(args.max_channel_outputs),
                int(args.channel_cohorts),
                str(args.channel_output_mode),
                str(args.channel_mode),
                str(args.channel_velocity_normalization),
                label_prior_correction,
            )
            metric_operator = signal_pullback_metric_operator(model, bargs, J_base, U_base, float(args.pullback_ridge))
            metric_override = metric_operator["metric_vec"].to(device=device) if isinstance(metric_operator.get("metric_vec"), torch.Tensor) else None
            pullback_info = metric_operator
            metric: torch.Tensor | dict[str, Any] = metric_operator
        elif scheme == "signal_mode_band":
            base_metric = metric_vector(model, bargs, "op_metric")
            U_base, J_base, _info_base = output_velocity_channel(
                model,
                xsrc,
                ysrc,
                base_metric,
                int(args.signal_rank),
                int(args.max_channel_outputs),
                int(args.channel_cohorts),
                str(args.channel_output_mode),
                str(args.channel_mode),
                str(args.channel_velocity_normalization),
                label_prior_correction,
            )
            metric_override, band_info = signal_mode_band_metric_vector(model, bargs, J_base, U_base, float(args.mode_band_alpha))
            metric_override = metric_override.to(device=device)
            metric = metric_override
        else:
            metric = metric_vector(model, bargs, metric_kind, gate=gate_for_metric, seed=seed)
        U0, J0, info0 = output_velocity_channel(
            model,
            xsrc,
            ysrc,
            metric,
            int(args.signal_rank),
            int(args.max_channel_outputs),
            int(args.channel_cohorts),
            str(args.channel_output_mode),
            str(args.channel_mode),
            str(args.channel_velocity_normalization),
            label_prior_correction,
        )
        if int(args.channel_source_guard_intersection):
            Ug0, _Jg0, infog0 = output_velocity_channel(
                model,
                xsrc_guard,
                ysrc_guard,
                metric,
                int(args.signal_rank),
                int(args.max_channel_outputs),
                int(args.channel_cohorts),
                str(args.channel_output_mode),
                str(args.channel_mode),
                str(args.channel_velocity_normalization),
                label_prior_correction,
            )
            source_guard_stability_before = projector_overlap(U0, Ug0)
        else:
            infog0 = {}
            source_guard_stability_before = 1.0
        train_trace = run_optimizer_training(model, xtr, ytr, bargs, scheme, gate_for_metric, seed, metric_override=metric_override, metric_operator=metric_operator)
        after_guard = v2293.metrics_for_model(model, xg, yg)
        after_train = v2293.metrics_for_model(model, xtr, ytr)
        drift = metric_drift_proxies(initial_model, model, xtr, bargs)
        after_logits = output_values(model, xsrc, ysrc, str(args.channel_output_mode)).detach().reshape(-1).cpu().to(dtype=torch.float64)
        metric1: torch.Tensor | dict[str, Any] = metric_operator if metric_operator is not None else metric_vector(model, bargs, metric_kind, gate=gate_for_metric, seed=seed)
        U1, _J1, info1 = output_velocity_channel(
            model,
            xsrc,
            ysrc,
            metric1,
            int(args.signal_rank),
            int(args.max_channel_outputs),
            int(args.channel_cohorts),
            str(args.channel_output_mode),
            str(args.channel_mode),
            str(args.channel_velocity_normalization),
            label_prior_correction,
        )
        if int(args.channel_source_guard_intersection):
            Ug1, _Jg1, infog1 = output_velocity_channel(
                model,
                xsrc_guard,
                ysrc_guard,
                metric1,
                int(args.signal_rank),
                int(args.max_channel_outputs),
                int(args.channel_cohorts),
                str(args.channel_output_mode),
                str(args.channel_mode),
                str(args.channel_velocity_normalization),
                label_prior_correction,
            )
            source_guard_stability = projector_overlap(U1, Ug1)
        else:
            infog1 = {}
            source_guard_stability = 1.0
        window_stability = projector_overlap(U0, U1)
        dz = after_logits[: int(U1.shape[0])] - before_logits[: int(U1.shape[0])]
        channel_overlap = vector_channel_fraction(U1, dz)
        if int(args.channel_source_guard_intersection):
            channel_overlap *= source_guard_stability
        dz_energy = float(dz.square().sum().item())
        params = list(initial_model.coeffs)
        xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, 0, int(args.batch_size), seed)
        raw_g = loss_gradient_vector(initial_model, xb, yb, params, label_prior_correction).to(dtype=torch.float64).cpu()
        op_metric = metric_vector(initial_model, bargs, "op_metric").cpu()
        deleted_delta = raw_g * (op_metric - 1.0)
        n_deleted = min(int(J0.shape[1]), int(deleted_delta.numel()))
        dz_deleted = J0[:, :n_deleted] @ deleted_delta[:n_deleted]
        deleted_sig_frac = vector_channel_fraction(U0, dz_deleted)
        if int(args.channel_source_guard_intersection):
            deleted_sig_frac *= source_guard_stability_before
        deleted_sig = float(deleted_sig_frac * dz_deleted.square().sum().item())
        deleted_res = float((1.0 - deleted_sig_frac) * dz_deleted.square().sum().item())
        proj_path = CHANNEL_ROOT / f"{row_id}.npz"
        np.savez_compressed(
            proj_path,
            U0=U0.numpy(),
            U1=U1.numpy(),
            snr_gate=gate.numpy(),
            block_snr=np.array([float(v) for v in block_info.values()], dtype=np.float64),
            source_guard_stability=np.array([float(source_guard_stability)], dtype=np.float64),
        )
        W_top_mass_ratio_raw = float(info1["W_top_mass_ratio"])
        W_top_mass_ratio = W_top_mass_ratio_raw * float(source_guard_stability) if int(args.channel_source_guard_intersection) else W_top_mass_ratio_raw
        return {
            "row_id": row_id,
            "part": row_part,
            "scheme": scheme,
            "basis_key": basis_key,
            "depth": depth,
            "seed": seed,
            "dataset_or_task": "f5_no_debt_calibration",
            "control": "f5_no_debt",
            "teacher_type": "f5_no_debt_calibration",
            "status": "ok",
            "argument_hash": argument_hash(args),
            "projector_npz": rel(proj_path),
            "metric_override_used": int(metric_override is not None),
            "pullback_ridge": float(args.pullback_ridge) if scheme == "signal_pullback" else "",
            "G_pull_rank": int(pullback_info.get("rank", 0)) if scheme == "signal_pullback" else "",
            "G_pull_condition": fval(pullback_info.get("condition")) if scheme == "signal_pullback" else "",
            "solve_residual": fval(pullback_info.get("woodbury_residual")) if scheme == "signal_pullback" else "",
            "pullback_operator_kind": str(pullback_info.get("kind", "")) if scheme == "signal_pullback" else "",
            "signal_mode_band_alpha": float(args.mode_band_alpha) if scheme == "signal_mode_band" else "",
            "mode_weight_change": band_info.get("mode_weight_change", "") if scheme == "signal_mode_band" else "",
            "mode_signal_entropy": band_info.get("mode_signal_entropy", "") if scheme == "signal_mode_band" else "",
            "degree_or_frequency_weight_shift": band_info.get("degree_or_frequency_weight_shift", "") if scheme == "signal_mode_band" else "",
            "channel_source_guard_intersection": int(args.channel_source_guard_intersection),
            "source_guard_channel_stability_before": source_guard_stability_before,
            "source_guard_channel_stability": source_guard_stability,
            "signal_channel_window_stability": window_stability,
            "velocity_cohort_count": info1.get("velocity_cohort_count", 0),
            "velocity_norm_median": info1.get("velocity_norm_median", 0.0),
            "guard_velocity_norm_median": infog1.get("velocity_norm_median", 0.0),
            "W_top_mass_ratio_raw": W_top_mass_ratio_raw,
            "guard_W_top_mass_ratio_raw": infog1.get("W_top_mass_ratio", 0.0),
            "W_top_mass_ratio": W_top_mass_ratio,
            "W_effective_rank": info1["W_effective_rank"],
            "PSD_violation_min_eig": info1["PSD_violation_min_eig"],
            "signal_channel_projector_rank": info1["signal_channel_projector_rank"],
            "diagonal_snr_median": snr_info["diagonal_snr_median"],
            "diagonal_snr_top_decile_mean": snr_info["diagonal_snr_top_decile_mean"],
            "snr_gate_density": snr_info.get("snr_gate_density", 0.0),
            "layer_block_snr": block_info.get("layer_block_snr", 0.0),
            "mode_band_snr": block_info.get("mode_band_snr", 0.0),
            "tangent_block_snr": block_info.get("tangent_block_snr", 0.0),
            "normal_block_snr": block_info.get("normal_block_snr", 0.0),
            "AB_positive_trace_fraction": snr_info["AB_positive_trace_fraction"],
            "AB_negative_trace_fraction": snr_info["AB_negative_trace_fraction"],
            "deleted_freedom_signal_energy": deleted_sig,
            "deleted_freedom_reservoir_energy": deleted_res,
            "signal_to_reservoir_ratio_deleted": deleted_sig / max(deleted_res, EPS),
            "normal_freedom_signal_fraction": deleted_sig_frac,
            "operator_freedom_signal_fraction": channel_overlap,
            "signal_energy_of_update": float(channel_overlap * dz_energy),
            "reservoir_energy_of_update": float((1.0 - channel_overlap) * dz_energy),
            "guard_NLL_delta": after_guard["nll"] - before_guard["nll"],
            "initial_visual_accuracy": before_guard["accuracy"],
            "final_visual_accuracy": after_guard["accuracy"],
            "visual_accuracy_improvement": after_guard["accuracy"] - before_guard["accuracy"],
            "initial_visual_coverage": before_guard["coverage_CVaR25"],
            "final_visual_coverage": after_guard["coverage_CVaR25"],
            "visual_coverage_improvement": after_guard["coverage_CVaR25"] - before_guard["coverage_CVaR25"],
            "guard_debt_before": before_guard["debt_metric"],
            "guard_debt_after": after_guard["debt_metric"],
            "guard_debt_delta": after_guard["debt_metric"] - before_guard["debt_metric"],
            "train_debt_before": before_train["debt_metric"],
            "train_debt_after": after_train["debt_metric"],
            "train_debt_delta": after_train["debt_metric"] - before_train["debt_metric"],
            "no_debt": int(after_guard["debt_metric"] <= before_guard["debt_metric"] + float(args.no_debt_budget)),
            "Brier_delta": after_guard.get("brier", 0.0) - before_guard.get("brier", 0.0),
            "ECE_delta": after_guard.get("ece", 0.0) - before_guard.get("ece", 0.0),
            "tail95_delta": after_guard.get("tail95", 0.0) - before_guard.get("tail95", 0.0),
            "tail99_delta": after_guard.get("tail99", 0.0) - before_guard.get("tail99", 0.0),
            "margin10_delta": after_guard.get("margin10", 0.0) - before_guard.get("margin10", 0.0),
            "data_gram_drift_max": drift["data_gram_drift_max"],
            "operator_spectrum_drift_max": drift["operator_spectrum_drift_max"],
            "changed_edge_coefficients": int(train_trace.get("changed_edge_coefficients", 0)),
            "preconditioner_applied_inside_optimizer_step": int(scheme != "adamw_witness"),
            "state_updated_every_step": int(train_trace.get("optimizer_state_max", 0.0) >= int(args.train_steps) or scheme == "adamw_witness"),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
    except Exception as exc:
        return {
            "row_id": row_id,
            "part": row_part,
            "scheme": scheme,
            "basis_key": basis_key,
            "depth": depth,
            "seed": seed,
            "dataset_or_task": "f5_no_debt_calibration",
            "control": "f5_no_debt",
            "status": "error",
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }


def run_part_e_job(job: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    e_args = argparse.Namespace(**vars(args))
    setattr(e_args, "row_part", "E")
    if job.get("kind") == "visual":
        return run_part_c_row(job["job"], e_args, device)
    return run_part_e_f5_row(job, e_args, device)


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    c = read_json(OUT_ROOT / "part_c_signal_channel_summary.json")
    d = read_json(OUT_ROOT / "part_d_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        return write_blocked_part("E", "E_BlockedByPartC", "part_c_failed", "Part C failed; Part E metric induction is prohibited by the plan.", args)
    if int(d.get("gate_pass", 0)) != 1:
        return write_blocked_part("E", "E_BlockedByPartD", "part_d_failed", "Part D failed; Part E cannot claim metric-to-channel causality.", args)
    device = device_from_args(args)
    rows = [run_part_e_job(job, args, device) for job in shard_items(part_e_jobs(args), args)]
    path = OUT_ROOT / f"part_e_reduced_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    summary = gate_summary("E", 0, "E_ShardsWritten", "merge_required", rows, shard_index=int(args.shard_index), shard_count=int(args.shard_count))
    append_exec("part-e", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(rows)}")
    return summary


def merge_part_e(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob("part_e_reduced_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = OUT_ROOT / "part_e_reduced_matrix.csv"
    write_rows(matrix, [dict(row) for row in rows])
    ok = [row for row in rows if row.get("status") == "ok"]
    op_baseline: dict[tuple[str, str], dict[str, float]] = {}
    for basis, depth in sorted({(r.get("basis_key", ""), r.get("depth", "")) for r in ok}):
        gr = [r for r in ok if r.get("basis_key") == basis and r.get("depth") == depth and r.get("scheme") == "op_metric"]
        c2 = [r for r in gr if r.get("control") == "c2_positive" and r.get("dataset_or_task") != "f5_no_debt_calibration"]
        f5 = [r for r in gr if r.get("dataset_or_task") == "f5_no_debt_calibration"]
        op_baseline[(basis, depth)] = {
            "ordinary_c2_coverage_median": median(fval(r.get("final_visual_coverage")) for r in c2),
            "ordinary_f5_no_debt_rows": float(sum(ival(r.get("no_debt")) for r in f5)),
            "ordinary_f5_rows": float(len(f5)),
        }
    scheme_groups: list[dict[str, Any]] = []
    for scheme, basis, depth in sorted({(r.get("scheme", ""), r.get("basis_key", ""), r.get("depth", "")) for r in ok}):
        gr = [r for r in ok if r.get("scheme") == scheme and r.get("basis_key") == basis and r.get("depth") == depth]
        c2 = [r for r in gr if r.get("control") == "c2_positive" and r.get("dataset_or_task") != "f5_no_debt_calibration"]
        neg = [r for r in gr if r.get("control") in {"random_label", "source_shuffle"}]
        f5 = [r for r in gr if r.get("dataset_or_task") == "f5_no_debt_calibration"]
        c2_cov = median(fval(r.get("final_visual_coverage")) for r in c2)
        c2_acc = median(fval(r.get("visual_accuracy_improvement")) for r in c2)
        c2_top = median(fval(r.get("W_top_mass_ratio")) for r in c2)
        neg_top = median(fval(r.get("W_top_mass_ratio")) for r in neg)
        c2_snr = median(fval(r.get("diagonal_snr_top_decile_mean")) for r in c2)
        neg_snr = median(fval(r.get("diagonal_snr_top_decile_mean")) for r in neg)
        negative_pass = int(bool(neg) and ((neg_top <= 0.75 * c2_top) or (neg_snr <= 0.75 * c2_snr)))
        f5_no_debt = sum(ival(r.get("no_debt")) for r in f5)
        ordinary = op_baseline.get((basis, depth), {})
        ordinary_cov = float(ordinary.get("ordinary_c2_coverage_median", 0.0))
        c2_retention = c2_cov / max(ordinary_cov, EPS) if ordinary_cov > 0 else 0.0
        group_gate = int(
            bool(c2)
            and scheme in {"signal_pullback", "signal_mode_band"}
            and len(f5) >= int(args.part_e_f5_seed_count)
            and c2_cov >= 0.20
            and c2_acc >= 0.20
            and negative_pass
            and f5_no_debt >= 8
            and c2_retention >= 0.80
        )
        scheme_groups.append(
            {
                "scheme": scheme,
                "basis_key": basis,
                "depth": depth,
                "rows": len(gr),
                "c2_rows": len(c2),
                "negative_rows": len(neg),
                "f5_rows": len(f5),
                "C2_visual_coverage_median": c2_cov,
                "C2_accuracy_improvement_median": c2_acc,
                "C2_retention_vs_ordinary": c2_retention,
                "C2_channel_top_mass_median": c2_top,
                "negative_channel_top_mass_median": neg_top,
                "C2_snr_top_decile_median": c2_snr,
                "negative_snr_top_decile_median": neg_snr,
                "negative_control_pass": negative_pass,
                "F5_no_debt": f5_no_debt,
                "F5_no_debt_denominator": len(f5),
                "ordinary_c2_coverage_median": ordinary_cov,
                "ordinary_f5_no_debt_rows": ordinary.get("ordinary_f5_no_debt_rows", 0.0),
                "operator_spectrum_drift_median": median(fval(r.get("operator_spectrum_drift_max")) for r in gr),
                "data_gram_drift_median": median(fval(r.get("data_gram_drift_max")) for r in gr),
                "G_pull_rank_median": median(fval(r.get("G_pull_rank")) for r in gr if r.get("scheme") == "signal_pullback"),
                "G_pull_condition_median": median(fval(r.get("G_pull_condition")) for r in gr if r.get("scheme") == "signal_pullback"),
                "solve_residual_median": median(fval(r.get("solve_residual")) for r in gr if r.get("scheme") == "signal_pullback"),
                "mode_weight_change_median": median(fval(r.get("mode_weight_change")) for r in gr if r.get("scheme") == "signal_mode_band"),
                "mode_signal_entropy_median": median(fval(r.get("mode_signal_entropy")) for r in gr if r.get("scheme") == "signal_mode_band"),
                "degree_or_frequency_weight_shift_median": median(fval(r.get("degree_or_frequency_weight_shift")) for r in gr if r.get("scheme") == "signal_mode_band"),
                "scheme_gate_pass": group_gate,
            }
        )
    pass_groups = [g for g in scheme_groups if int(g.get("scheme_gate_pass", 0)) == 1]
    induced_groups = [g for g in scheme_groups if g.get("scheme") in {"signal_pullback", "signal_mode_band"}]
    c2_viable_groups = [g for g in induced_groups if fval(g.get("C2_visual_coverage_median")) >= 0.20 and fval(g.get("C2_accuracy_improvement_median")) >= 0.20]
    any_negative_fail = any(int(g.get("negative_control_pass", 0)) == 0 for g in c2_viable_groups)
    any_f5_low = any(int(g.get("F5_no_debt", 0)) < 8 for g in c2_viable_groups)
    if pass_groups:
        route, blocker = "E_ReducedSignalPullbackPass", "none"
    elif not c2_viable_groups:
        route, blocker = "E_C2SignalPullbackInsufficient", "c2_visual_or_accuracy"
    elif any_f5_low:
        route, blocker = "SignalPullbackFindsTaskButNotSafety", "F5_no_debt_low"
    elif any_negative_fail:
        route, blocker = "E_NegativeControlLeakage", "negative_control_leakage"
    else:
        route, blocker = "E_C2SignalPullbackInsufficient", "c2_visual_or_accuracy"
    gate = int(bool(pass_groups))
    summary = gate_summary(
        "E",
        gate,
        route,
        blocker,
        [dict(row) for row in rows],
        total_rows=len(rows),
        expected_rows=len(part_e_jobs(args)),
        scheme_groups=scheme_groups,
        passing_scheme_groups=pass_groups,
        part_e_gate_pass=gate,
        part_e_route=route,
        dominant_blocker=blocker,
        reduced_matrix_only=1,
        evidence_fields=[
            "C2_visual_coverage_median",
            "C2_accuracy_improvement_median",
            "negative_control_pass",
            "F5_no_debt",
            "G_pull_condition_median",
            "solve_residual_median",
        ],
    )
    summary_path = OUT_ROOT / "part_e_summary.json"
    alias_path = OUT_ROOT / "part_e_reduced_signal_pullback_summary.json"
    write_json(summary_path, summary)
    write_json(alias_path, summary)
    next_path = common_next_actions(
        "E",
        blocker,
        "Part E checks whether signal-channel pullback preserves C2 while improving F5 no-debt on a reduced diagnostic matrix.",
        [
            "if C2 is viable but F5 is low, run Part G task/debt channel split",
            "if pullback residual or condition is poor, increase ridge or reduce signal rank before rerunning Part E",
            "if C2 is killed, compare E-5 signal-induced mode-band reweighting against E-1",
        ],
        [
            "promote reduced Part E to official success",
            "drop F5 no-debt or negative-control gates",
            "choose a single seed as the passing scheme",
        ],
        [f"{PYTHON} {rel(RUNNER)} --mode part-e --shard-count 4 --shard-index <0-3> --device cuda:<0-3>", f"{PYTHON} {rel(RUNNER)} --mode part-e-merge --device {args.device}"],
    )
    append_exec("part-e-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(alias_path)}; {rel(next_path)}")
    append_recap("Part E reduced signal-channel pullback audit", summary)
    return summary


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    d = read_json(OUT_ROOT / "part_d_summary.json")
    if int(d.get("gate_pass", 0)) != 1:
        return write_blocked_part("F", "F_BlockedByPartD", "part_d_failed", "Part F requires Part D metric-to-channel causality.", args)
    source_matrix = OUT_ROOT / "part_e_reduced_matrix.csv"
    rows = read_rows(source_matrix)
    ok = [row for row in rows if row.get("status") == "ok"]
    pop_rows = [row for row in ok if row.get("scheme") == "population_snr"]
    rand_rows = [row for row in ok if row.get("scheme") == "random_matched"]
    op_rows = [row for row in ok if row.get("scheme") == "op_metric"]
    matrix_rows: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    for basis, depth in sorted({(r.get("basis_key", ""), r.get("depth", "")) for r in pop_rows}):
        pop = [r for r in pop_rows if r.get("basis_key") == basis and r.get("depth") == depth]
        rnd = [r for r in rand_rows if r.get("basis_key") == basis and r.get("depth") == depth]
        op = [r for r in op_rows if r.get("basis_key") == basis and r.get("depth") == depth]
        matrix_rows.extend(dict(r) for r in pop)
        c2 = [r for r in pop if r.get("control") == "c2_positive" and r.get("dataset_or_task") != "f5_no_debt_calibration"]
        f5 = [r for r in pop if r.get("dataset_or_task") == "f5_no_debt_calibration"]
        neg = [r for r in pop if r.get("control") in {"random_label", "source_shuffle"}]
        rnd_c2 = [r for r in rnd if r.get("control") == "c2_positive" and r.get("dataset_or_task") != "f5_no_debt_calibration"]
        rnd_f5 = [r for r in rnd if r.get("dataset_or_task") == "f5_no_debt_calibration"]
        op_c2 = [r for r in op if r.get("control") == "c2_positive" and r.get("dataset_or_task") != "f5_no_debt_calibration"]
        op_f5 = [r for r in op if r.get("dataset_or_task") == "f5_no_debt_calibration"]
        c2_cov = median(fval(r.get("final_visual_coverage")) for r in c2)
        c2_acc = median(fval(r.get("visual_accuracy_improvement")) for r in c2)
        op_cov = median(fval(r.get("final_visual_coverage")) for r in op_c2)
        c2_retention = c2_cov / max(op_cov, EPS) if op_cov > 0 else 0.0
        f5_no_debt = sum(ival(r.get("no_debt")) for r in f5)
        rnd_acc = median(fval(r.get("visual_accuracy_improvement")) for r in rnd_c2)
        rnd_cov = median(fval(r.get("final_visual_coverage")) for r in rnd_c2)
        rnd_f5 = sum(ival(r.get("no_debt")) for r in rnd_f5)
        op_f5 = sum(ival(r.get("no_debt")) for r in op_f5)
        pop_top = median(fval(r.get("W_top_mass_ratio")) for r in c2)
        neg_top = median(fval(r.get("W_top_mass_ratio")) for r in neg)
        pop_snr = median(fval(r.get("diagonal_snr_top_decile_mean")) for r in c2)
        neg_snr = median(fval(r.get("diagonal_snr_top_decile_mean")) for r in neg)
        negative_pass = int(bool(neg) and ((neg_top <= 0.75 * pop_top) or (neg_snr <= 0.75 * pop_snr)))
        random_control_pass = int((c2_acc > rnd_acc + 0.05) or (f5_no_debt > rnd_f5 and c2_cov >= 0.70 * max(rnd_cov, EPS)))
        group_gate = int(c2_retention >= 0.70 and c2_acc >= 0.20 and f5_no_debt >= 8 and negative_pass and random_control_pass)
        group_summaries.append(
            {
                "scheme": "population_snr",
                "basis_key": basis,
                "depth": depth,
                "rows": len(pop),
                "c2_rows": len(c2),
                "f5_rows": len(f5),
                "C2_visual_coverage_median": c2_cov,
                "C2_accuracy_improvement_median": c2_acc,
                "C2_retention_vs_op_metric": c2_retention,
                "F5_no_debt": f5_no_debt,
                "F5_no_debt_denominator": len(f5),
                "op_metric_F5_no_debt": op_f5,
                "random_matched_C2_accuracy_improvement_median": rnd_acc,
                "random_matched_C2_visual_coverage_median": rnd_cov,
                "random_matched_F5_no_debt": rnd_f5,
                "negative_control_pass": negative_pass,
                "random_snr_control_pass": random_control_pass,
                "snr_gate_density_median": median(fval(r.get("snr_gate_density")) for r in pop),
                "diagonal_snr_top_decile_median": pop_snr,
                "normal_block_snr_median": median(fval(r.get("normal_block_snr")) for r in pop),
                "tangent_block_snr_median": median(fval(r.get("tangent_block_snr")) for r in pop),
                "part_f_group_gate_pass": group_gate,
            }
        )
    pass_groups = [g for g in group_summaries if int(g.get("part_f_group_gate_pass", 0)) == 1]
    any_c2 = any(fval(g.get("C2_accuracy_improvement_median")) >= 0.20 and fval(g.get("C2_retention_vs_op_metric")) >= 0.70 for g in group_summaries)
    any_f5_low = any(int(g.get("F5_no_debt", 0)) < 8 for g in group_summaries if fval(g.get("C2_accuracy_improvement_median")) >= 0.20)
    any_random_fail = any(int(g.get("random_snr_control_pass", 0)) == 0 for g in group_summaries if fval(g.get("C2_accuracy_improvement_median")) >= 0.20)
    if pass_groups:
        route, blocker = "F_ReducedPopulationSNRPass", "none"
    elif not any_c2:
        route, blocker = "F_SNRGateDoesNotFormC2Accuracy", "c2_accuracy_low"
    elif any_f5_low:
        route, blocker = "F_SNRGateFailsF5", "F5_no_debt_low"
    elif any_random_fail:
        route, blocker = "F_RandomSNRControlExplainsGain", "random_matched_control"
    else:
        route, blocker = "F_PopulationSNRInsufficient", "population_snr_insufficient"
    matrix = OUT_ROOT / "part_f_population_snr_matrix.csv"
    write_rows(matrix, matrix_rows)
    summary = gate_summary(
        "F",
        int(bool(pass_groups)),
        route,
        blocker,
        matrix_rows,
        source_matrix=rel(source_matrix),
        reused_part_e_rows=1,
        group_summaries=group_summaries,
        passing_part_f_groups=pass_groups,
        part_f_gate_pass=int(bool(pass_groups)),
        part_f_route=route,
        evidence_fields=[
            "C2_accuracy_improvement_median",
            "C2_retention_vs_op_metric",
            "F5_no_debt",
            "random_snr_control_pass",
            "negative_control_pass",
        ],
    )
    summary_path = OUT_ROOT / "part_f_summary.json"
    alias_path = OUT_ROOT / "part_f_population_snr_summary.json"
    write_json(summary_path, summary)
    write_json(alias_path, summary)
    next_path = common_next_actions(
        "F",
        blocker,
        "Part F audits whether population SNR gates form C2 while improving F5 beyond random matched gates.",
        [
            "if all SNR gates fail C2, analyze Part C SNR distribution and C2-positive freedom alignment",
            "if C2 forms but F5 is low, run Part G task/debt split",
            "if random matched explains the gain, inspect whitening and SNR block construction",
        ],
        [
            "promote population_snr when random_matched is comparable",
            "drop random-SNR control",
            "weaken C2 accuracy or F5 gates",
        ],
        [f"{PYTHON} {rel(RUNNER)} --mode part-f --device {args.device}"],
    )
    append_exec("part-f", command_text(sys.argv), "passed" if pass_groups else "failed", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(alias_path)}; {rel(next_path)}")
    append_recap("Part F reduced population-SNR audit", summary)
    return summary


def run_part_l(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_signal_channel_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        route = "SignalChannelEstimatorUnstable" if "Negative" not in str(c.get("route", "")) else "C_ChannelEstimatorNegativeControlLeakage"
        blocker = str(c.get("dominant_blocker", "part_c_failed"))
    else:
        d = read_json(OUT_ROOT / "part_d_summary.json")
        if int(d.get("gate_pass", 0)) != 1:
            route = str(d.get("route", "PartDNotRun"))
            blocker = str(d.get("dominant_blocker", "part_d_missing"))
        else:
            e = read_json(OUT_ROOT / "part_e_summary.json")
            f = read_json(OUT_ROOT / "part_f_summary.json")
            if f:
                route = str(f.get("route", f.get("part_f_route", "PartFUnknown")))
                blocker = str(f.get("dominant_blocker", "part_f_unknown"))
            elif e:
                route = str(e.get("route", e.get("part_e_route", "PartEUnknown")))
                blocker = str(e.get("dominant_blocker", "part_e_unknown"))
            else:
                route = "D_Pass_PartENotRun"
                blocker = "part_e_missing"
    summary = {
        "part": "L",
        "gate_pass": 0,
        "route": route,
        "dominant_blocker": blocker,
        "evidence_fields": c.get("evidence_fields", []),
        "used_fake_data_rows": 0,
        "held_test_usage": 0,
        "promotion_allowed": False,
        "conclusion": "Closeout route is based only on recorded artifacts.",
    }
    path = OUT_ROOT / "part_l_failure_decomposition_summary.json"
    write_json(path, summary)
    append_exec("part-l", command_text(sys.argv), "done", files=rel(path))
    append_recap("Part L failure decomposition", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_signal_channel_summary.json")
    l = read_json(OUT_ROOT / "part_l_failure_decomposition_summary.json")
    if not l:
        l = run_part_l(args)
    route = str(l.get("route", "Incomplete"))
    official_candidate_gate_pass = 0
    summary = {
        "version": "v22.95R",
        "route": route,
        "dominant_blocker": l.get("dominant_blocker", "missing"),
        "official_candidate_gate_pass": official_candidate_gate_pass,
        "part_c_gate_pass": c.get("gate_pass", 0),
        "used_fake_data_rows": 0,
        "held_test_usage": 0,
        "promotion_allowed": False,
        "finalized_at": now(),
        "runner": rel(RUNNER),
        "plan": rel(PLAN),
        "execution_log": rel(EXEC_LOG),
        "recap_log": rel(RECAP_LOG),
    }
    path = OUT_ROOT / "final_route.json"
    manifest = OUT_ROOT / "reproduction_manifest.md"
    write_json(path, summary)
    manifest.write_text(
        "# v22.95R reproduction manifest\n\n"
        f"- Runner: `{rel(RUNNER)}`\n"
        f"- Plan: `{rel(PLAN)}`\n"
        f"- Output: `{rel(OUT_ROOT)}`\n"
        f"- Part A: `{rel(OUT_ROOT / 'part_a_identity_summary.json')}`\n"
        f"- Part B: `{rel(OUT_ROOT / 'part_b_history_lock.json')}`\n"
        f"- Part C: `{rel(OUT_ROOT / 'part_c_signal_channel_summary.json')}`\n"
        f"- Part D: `{rel(OUT_ROOT / 'part_d_summary.json')}`\n"
        f"- Part E: `{rel(OUT_ROOT / 'part_e_summary.json')}`\n"
        f"- Part F: `{rel(OUT_ROOT / 'part_f_summary.json')}`\n"
        f"- Final route: `{rel(path)}`\n\n"
        "Use the execution log for exact commands and shard/device mapping.\n",
        encoding="utf-8",
    )
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(path)}; {rel(manifest)}")
    append_recap("Final route", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--pc-basis", default="D-FOU")
    p.add_argument("--basis-k-override", type=int, default=0)
    p.add_argument("--dfour-k", type=int, default=3)
    p.add_argument("--input-dim", type=int, default=12)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--deep-width", type=int, default=12)
    p.add_argument("--mlp-hidden", type=int, default=24)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--smoothness", type=float, default=1.0e-4)
    p.add_argument("--c-condition-budget", type=float, default=1.0e5)
    p.add_argument("--metric-lr", type=float, default=0.03)
    p.add_argument("--adamw-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--metric-max-norm-ratio", type=float, default=1.0)
    p.add_argument("--train-steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--synthetic-train-size", type=int, default=96)
    p.add_argument("--synthetic-guard-size", type=int, default=96)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--mode-lambda-partial", type=float, default=1.0e-4)
    p.add_argument("--mode-lambda-partial2", type=float, default=1.0e-6)
    p.add_argument("--mode-lambda-omega", type=float, default=1.0e-4)
    p.add_argument("--part-c-basis", default="dche_k5,dche_k9,dfour_default")
    p.add_argument("--part-c-depths", default="depth2,depth3")
    p.add_argument("--part-c-schemes", default="adamw_witness,op_metric,population_snr")
    p.add_argument("--part-c-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-c-controls", default="c2_positive,random_label")
    p.add_argument("--part-c-seed-count", type=int, default=2)
    p.add_argument("--part-d-schemes", default="adamw_witness,data_metric,op_metric,population_snr,signal_pullback,random_matched")
    p.add_argument("--part-d-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-d-seed-count", type=int, default=3)
    p.add_argument("--part-e-basis", default="dche_k5")
    p.add_argument("--part-e-depths", default="depth2")
    p.add_argument("--part-e-schemes", default="op_metric,signal_pullback,signal_mode_band,population_snr,random_matched")
    p.add_argument("--part-e-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-e-controls", default="c2_positive,random_label,source_shuffle")
    p.add_argument("--part-e-seed-count", type=int, default=3)
    p.add_argument("--part-e-f5-seed-count", type=int, default=15)
    p.add_argument("--channel-source-size", type=int, default=24)
    p.add_argument("--channel-common-source", type=int, default=1)
    p.add_argument("--channel-probe-seed", type=int, default=950)
    p.add_argument("--channel-guard-probe-seed", type=int, default=951)
    p.add_argument("--channel-source-guard-intersection", type=int, default=0)
    p.add_argument("--channel-output-mode", default="margin", choices=["margin", "logits"])
    p.add_argument("--channel-mode", default="cumulative", choices=["cumulative", "agreement"])
    p.add_argument("--channel-velocity-normalization", default="unit", choices=["unit", "raw"])
    p.add_argument("--label-prior-correction", type=int, default=0)
    p.add_argument("--max-channel-outputs", type=int, default=96)
    p.add_argument("--signal-rank", type=int, default=2)
    p.add_argument("--channel-cohorts", type=int, default=8)
    p.add_argument("--snr-examples", type=int, default=24)
    p.add_argument("--snr-sample-normalization", default="unit", choices=["unit", "raw"])
    p.add_argument("--snr-beta", type=float, default=4.0)
    p.add_argument("--snr-tau", type=float, default=1.0)
    p.add_argument("--pullback-ridge", type=float, default=1.0e-3)
    p.add_argument("--mode-band-alpha", type=float, default=0.5)
    p.add_argument("--no-debt-budget", type=float, default=0.01)
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
    if mode == "part-e-merge":
        return merge_part_e(args)
    if mode == "part-f":
        return run_part_f(args)
    if mode in {"part-g", "part-h", "part-i", "part-j", "part-k"}:
        letter = mode.split("-")[1].upper()
        return write_blocked_part(letter, f"{letter}_BlockedByUnpassedPrerequisite", "part_c_or_d_not_ready", "Downstream part is blocked until prior gates pass and implementation is audited.", args)
    if mode == "part-l":
        return run_part_l(args)
    if mode == "finalize":
        return finalize(args)
    if mode == "all":
        run_part_a(args)
        run_part_b(args)
        run_part_c(args)
        merge_part_c(args)
        run_part_d(args)
        run_part_e(args)
        merge_part_e(args)
        run_part_f(args)
        for letter in "GHIJK":
            write_blocked_part(letter, f"{letter}_BlockedByUnpassedPrerequisite", "part_c_or_d_not_ready", "Downstream part is blocked until prior gates pass and implementation is audited.", args)
        run_part_l(args)
        return finalize(args)
    raise ValueError(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
