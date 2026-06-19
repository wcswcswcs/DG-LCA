#!/usr/bin/env python3
"""DG-KAN/KANbeFair bridge smoke with unified metrics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from dgkan.fu.mlp_adaptive_controller import MLPFUState, apply_mlp_fu_update  # noqa: E402
from dgkan.integration.kanbefair_adapter import canonical_dataset_name  # noqa: E402
from dgkan.models.fc_purekan_primitives import MLPBaseline, PrimitiveKAN, PrimitiveSpec  # noqa: E402
from experiments.run_v22_16_common import ce_cotangent, grad_energy_by_channel, selected_named_parameters  # noqa: E402
from experiments.run_v22_17_common import (  # noqa: E402
    OUT_ROOT,
    WORKTREE_ROOT,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--kanbefair-root", default=str(WORKTREE_ROOT))
    p.add_argument("--datasets", default="MNIST,FMNIST,KMNIST")
    p.add_argument("--models", default="DGMLP,DGMLP_FU,DGKAN_DFOU,DGKAN_DFOU_FU")
    p.add_argument("--seeds", default="0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--experiment-tag", default="v22_17_bridge")
    p.add_argument("--gate-margin", type=float, default=1.0e-8)
    p.add_argument("--actionable-gain-tolerance", type=float, default=0.002)
    p.add_argument("--gated-ratio-cap", type=float, default=0.01)
    p.add_argument("--actionable-ratio-cap", type=float, default=0.005)
    p.add_argument("--virtual-gate-scale", type=float, default=0.002)
    p.add_argument("--virtual-loss-tolerance", type=float, default=1.0e-5)
    p.add_argument("--output-suffix", default="")
    p.add_argument("--jvp-refresh-dim", type=int, default=8)
    p.add_argument("--jvp-refresh-min-history", type=int, default=8)
    p.add_argument("--jvp-refresh-history-window", type=int, default=32)
    p.add_argument("--jvp-refresh-scale", type=float, default=0.05)
    p.add_argument("--jvp-refresh-ratio-cap", type=float, default=0.05)
    p.add_argument("--jvp-refresh-ridge", type=float, default=1.0e-4)
    p.add_argument("--jvp-refresh-require-virtual-improvement", action="store_true")
    p.add_argument("--jvp-refresh-virtual-improvement-margin", type=float, default=0.0)
    p.add_argument("--benefit-p3-ratio-cap", type=float, default=0.01)
    p.add_argument("--benefit-p3-strong-ratio-cap", type=float, default=0.03)
    p.add_argument("--benefit-p3-min-score", type=float, default=0.0)
    p.add_argument("--benefit-p3-strong-min-score", type=float, default=0.005)
    p.add_argument("--benefit-p3-jvp-strong-scale", type=float, default=0.20)
    p.add_argument("--benefit-p3-jvp-strong-ratio-cap", type=float, default=0.20)
    p.add_argument("--benefit-p3-jvp-interval", type=int, default=1)
    p.add_argument("--benefit-policy-export-json", default="")
    p.add_argument("--benefit-policy-export-threshold", type=float, default=math.nan)
    return p


def _split(raw: str, cast: Any = str) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _batch_fingerprint(xb: torch.Tensor, yb: torch.Tensor) -> str:
    x_cpu = xb.detach().cpu().contiguous()
    y_cpu = yb.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(x_cpu.shape)).encode("utf-8"))
    h.update(x_cpu.numpy().tobytes())
    h.update(str(tuple(y_cpu.shape)).encode("utf-8"))
    h.update(y_cpu.numpy().tobytes())
    return h.hexdigest()


def _load_utils(root: Path) -> Any:
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import utils  # type: ignore

    return utils


class LoaderArgs:
    def __init__(self, dataset: str, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_batch_size = batch_size


def _subset_loader(loader: DataLoader, size: int, batch_size: int, seed: int, train: bool) -> DataLoader:
    ds = loader.dataset
    n = min(int(size), len(ds))
    gen = torch.Generator().manual_seed(int(seed) + (0 if train else 10000))
    idx = torch.randperm(len(ds), generator=gen)[:n].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=train, generator=gen, num_workers=0, drop_last=False)


def _get_loaders(root: Path, dataset: str, batch_size: int, train_size: int, test_size: int, seed: int) -> tuple[DataLoader, DataLoader, int, int]:
    utils = _load_utils(root)
    args = LoaderArgs(dataset=dataset, batch_size=batch_size)
    old_cwd = Path.cwd()
    os.chdir(root / "src")
    try:
        train_loader, test_loader, num_classes, input_size = utils.get_loader(args, use_cuda=False)
    finally:
        os.chdir(old_cwd)
    return (
        _subset_loader(train_loader, train_size, batch_size, seed, True),
        _subset_loader(test_loader, test_size, batch_size, seed, False),
        int(num_classes),
        int(input_size),
    )


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _mlp_flops(input_dim: int, hidden: int, output_dim: int) -> int:
    return int(2 * input_dim * hidden + 2 * hidden * hidden + 2 * hidden * output_dim)


def _kan_flops(input_dim: int, hidden: int, output_dim: int, k: int) -> int:
    return int((input_dim * hidden + hidden * output_dim) * max(1, k) * 8)


def _make_kan(input_dim: int, output_dim: int, hidden: int, seed: int, device: torch.device, model_name: str, x_stats: torch.Tensor) -> PrimitiveKAN:
    carrier = "D-CHE" if "DCHE" in model_name else "D-FOU"
    basis_name = "chebyshev" if carrier == "D-CHE" else "fourier_lowfreq"
    init_variant = "cheby_k3_triton_l3_gradbuf" if carrier == "D-CHE" else "fourier_k3_triton_l3_matmul"
    k = 3
    budget = input_dim * hidden + hidden * hidden + hidden * output_dim
    spec = PrimitiveSpec(
        candidate_id=f"v22.17-{carrier}-bridge-no-dense",
        basis_family=carrier,
        basis_name=basis_name,
        k=k,
        hidden_dim=max(4, hidden),
        source="v22_17_kanbefair_bridge",
        local_support=0,
        global_support=1,
        uses_exp=0,
        uses_sin_cos=int(carrier == "D-FOU"),
        uses_division=0,
        uses_dense_basis_tensor=0,
        init_variant=init_variant,
    )
    return PrimitiveKAN(input_dim, output_dim, spec, x_stats.to(device), seed, device, param_budget=budget)


def _make_model(model_name: str, input_dim: int, output_dim: int, hidden: int, seed: int, device: torch.device, x_stats: torch.Tensor) -> torch.nn.Module:
    if model_name.startswith("DGMLP"):
        return MLPBaseline(input_dim, output_dim, hidden, seed, device).to(device)
    return _make_kan(input_dim, output_dim, hidden, seed, device, model_name, x_stats).to(device)


def _model_flops(model_name: str, input_dim: int, hidden: int, output_dim: int) -> int:
    if model_name.startswith("DGMLP"):
        return _mlp_flops(input_dim, hidden, output_dim)
    return _kan_flops(input_dim, hidden, output_dim, 3)


def _flat_grads(named: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    parts = []
    for _name, p in named:
        parts.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def _assign_flat_update(named: list[tuple[str, torch.nn.Parameter]], update_flat: torch.Tensor) -> None:
    flat = update_flat.reshape(-1)
    offset = 0
    for _name, p in named:
        n = int(p.numel())
        chunk = (-flat[offset : offset + n]).reshape_as(p).to(dtype=p.dtype, device=p.device)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += n


def _eval(model: torch.nn.Module, loader: DataLoader, device: torch.device, output_dim: int) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    losses_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    y_all: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            xb = x.to(device).float()
            yb = y.to(device).long()
            logits = model(xb).float()
            loss = F.cross_entropy(logits, yb, reduction="none")
            total += int(yb.numel())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
            losses_all.append(loss.detach().cpu())
            logits_all.append(logits.detach().cpu())
            y_all.append(yb.detach().cpu())
    losses = torch.cat(losses_all) if losses_all else torch.empty(0)
    logits_cat = torch.cat(logits_all) if logits_all else torch.empty(0, output_dim)
    y_cat = torch.cat(y_all) if y_all else torch.empty(0, dtype=torch.long)
    probs = torch.softmax(logits_cat.float(), dim=-1) if logits_cat.numel() else logits_cat
    target = F.one_hot(y_cat, num_classes=output_dim).float() if y_cat.numel() else torch.empty_like(probs)
    ece = 0.0
    if y_cat.numel():
        conf, pred = probs.max(dim=-1)
        ok = (pred == y_cat).float()
        for idx in range(10):
            lo = idx / 10.0
            hi = (idx + 1) / 10.0
            mask = (conf >= lo) & (conf <= hi if idx == 9 else conf < hi)
            if mask.any():
                ece += float(mask.float().mean().item()) * abs(float(conf[mask].mean().item()) - float(ok[mask].mean().item()))
    return {
        "loss": float(losses.mean().item()) if losses.numel() else 0.0,
        "accuracy": correct / max(1, total),
        "ECE": ece,
        "Brier": float((probs - target).square().sum(dim=-1).mean().item()) if y_cat.numel() else 0.0,
        "tail_loss_q95": float(torch.quantile(losses.float(), 0.95).item()) if losses.numel() else 0.0,
        "tail_loss_q99": float(torch.quantile(losses.float(), 0.99).item()) if losses.numel() else 0.0,
    }


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().reshape(-1)
    bv = b.detach().float().reshape(-1)
    n = min(av.numel(), bv.numel())
    if n == 0:
        return 0.0
    av = av[:n]
    bv = bv[:n]
    return float(torch.dot(av, bv).div(torch.linalg.vector_norm(av).clamp_min(1.0e-12) * torch.linalg.vector_norm(bv).clamp_min(1.0e-12)).clamp(-1, 1).item())


def _u_score(linear_gain: float, nds: float, tail: float, control: float, staleness: float) -> float:
    return float(linear_gain - 0.05 * nds - 0.02 * tail - 0.05 * control - 0.001 * staleness)


def _u_score_source(source_loss_gain: float, nds: float, tail: float, control: float, staleness: float) -> float:
    return float(source_loss_gain - 0.002 * nds - 0.001 * tail - 0.01 * control - 0.0005 * staleness)


def _controller_policy(model_name: str) -> str:
    upper = model_name.upper()
    if "_FU" not in model_name:
        return "adamw_baseline"
    if "BENEFIT_P3" in upper and "JVP" in upper and "STRONG" in upper:
        return "benefit_p3_jvp_current_step_strong"
    if "BENEFIT_P3" in upper and "JVP" in upper:
        return "benefit_p3_jvp_current_step"
    if "BENEFIT_P3" in upper and "STRONG" in upper:
        return "benefit_p3_current_step_strong"
    if "BENEFIT_P3" in upper:
        return "benefit_p3_current_step"
    if "CONTROL_STABLE_RANDOM" in upper:
        return "task_neutral_stable_random_control"
    if "CONTROL_RANDOM" in upper:
        return "task_neutral_random_control"
    if "CONTROL_SIGNFLIP" in upper:
        return "task_neutral_signflip_control"
    if "CONTROL_CORRUPT" in upper:
        return "task_neutral_corrupt_control"
    if "JVP_REFRESH" in upper:
        return "jvp_useful_history_refresh"
    if "NOOP" in model_name:
        return "noop_equivalence"
    if "VIRTUAL" in model_name:
        return "task_useful_virtual_train_loss_gate"
    if "ACTIONABLE" in model_name:
        return "task_useful_actionable_gain_gate"
    if "RELEASE" in model_name:
        return "task_useful_release_low_ratio"
    if "GATED" in model_name:
        return "task_useful_gate"
    return "legacy_always_on_low_ratio"


def _controller_variant(policy: str, model_name: str) -> str:
    if policy == "task_neutral_random_control":
        return "M9 MLP+RandomSourceControl"
    if policy == "task_neutral_stable_random_control":
        return "M10 MLP+StableRandomSourceControl"
    if policy == "task_neutral_signflip_control":
        return "M11 MLP+SignFlipSourceControl"
    if policy == "task_neutral_corrupt_control":
        return "M12 MLP+CorruptSourceControl"
    if policy == "jvp_useful_history_refresh":
        return "M6-JVP MLP+JVPUsefulHistoryRefresh"
    if policy.startswith("benefit_p3"):
        return "M5 MLP+SourceLossGatedPredictiveProx"
    if "release" in policy:
        return "M6 MLP+SourceReleaseController"
    if "actionable" in policy or "virtual" in policy or "gated" in policy:
        return "M5 MLP+SourceLossGatedPredictiveProx"
    return "M7 MLP+RealSourceManifold-k8"


def _ratio_cap_for_policy(policy: str, args: argparse.Namespace) -> float:
    if policy.startswith("task_neutral_"):
        return float(args.actionable_ratio_cap)
    if policy == "jvp_useful_history_refresh":
        return float(args.jvp_refresh_ratio_cap)
    if policy in {"benefit_p3_current_step_strong", "benefit_p3_jvp_current_step_strong"}:
        return float(args.benefit_p3_strong_ratio_cap)
    if policy in {"benefit_p3_current_step", "benefit_p3_jvp_current_step"}:
        return float(args.benefit_p3_ratio_cap)
    if policy == "legacy_always_on_low_ratio":
        return 0.05
    if policy == "task_useful_gate":
        return float(args.gated_ratio_cap)
    if policy in {"task_useful_actionable_gain_gate", "task_useful_virtual_train_loss_gate"}:
        return float(args.actionable_ratio_cap)
    if policy == "task_useful_release_low_ratio":
        return min(float(args.gated_ratio_cap), 0.01)
    return 0.0


def _base_model_for_delta(model_name: str) -> str:
    if model_name.startswith("DGMLP"):
        return "DGMLP"
    if model_name.startswith("DGKAN_DFOU"):
        return "DGKAN_DFOU"
    if model_name.startswith("DGKAN_DCHE"):
        return "DGKAN_DCHE"
    return model_name


def _is_fu_model(model_name: str) -> bool:
    return "_FU" in model_name


def _source_candidate_type(model_name: str) -> str:
    upper = model_name.upper()
    if "CONTROL" in model_name.upper():
        return "trained_task_neutral_control"
    if "JVP_REFRESH" in upper or ("BENEFIT_P3" in upper and "JVP" in upper):
        return "jvp_useful_history"
    if "SPLIT" in upper:
        return "split_consensus_param"
    if "PARAM" in upper:
        return "param_update"
    return "output_delta"


def _benefit_p3_min_score(policy: str, args: argparse.Namespace) -> float:
    if policy.endswith("_strong"):
        return float(args.benefit_p3_strong_min_score)
    return float(args.benefit_p3_min_score)


def _runtime_benefit_p3_score(
    *,
    source_loss_gain: float,
    u_task_pre: float,
    random_u_pre: float,
    nds: float,
    tail_proxy: float,
    risk_score: float,
    update_ratio: float,
    virtual_loss_delta: float,
    predicted_gain_base: float,
    predicted_gain_guided: float,
    source_age: float,
) -> float:
    advantage = float(u_task_pre) - float(random_u_pre)
    normalized_gain_delta = (float(predicted_gain_guided) - float(predicted_gain_base)) / max(1.0, abs(float(predicted_gain_base)))
    return float(
        1000.0 * max(0.0, float(source_loss_gain))
        + 0.25 * max(0.0, advantage)
        + 0.20 * normalized_gain_delta
        - 0.50 * max(0.0, float(virtual_loss_delta))
        - 0.05 * max(0.0, float(update_ratio) - 0.01)
        - 0.005 * max(0.0, float(risk_score))
        - 0.005 * max(0.0, float(nds))
        - 0.0002 * max(0.0, float(tail_proxy))
        - 0.0005 * max(0.0, float(source_age))
    )


def _safe_runtime_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _load_benefit_policy_export(raw_path: str, threshold_override: float) -> dict[str, Any] | None:
    if not str(raw_path or "").strip():
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    data = json.loads(path.read_text(encoding="utf-8"))
    feature_names = [str(x) for x in data.get("feature_names", [])]
    coef = [_safe_runtime_float(x) for x in data.get("coef", [])]
    mean = [_safe_runtime_float(x) for x in data.get("mean", [])]
    scale = [_safe_runtime_float(x, 1.0) or 1.0 for x in data.get("scale", [])]
    if not feature_names or not (len(feature_names) == len(coef) == len(mean) == len(scale)):
        raise ValueError(f"invalid benefit policy export shape: {path}")
    threshold = threshold_override if math.isfinite(float(threshold_override)) else _safe_runtime_float(data.get("accept_threshold"), 0.0)
    return {
        "path": str(path),
        "feature_names": feature_names,
        "coef": coef,
        "mean": mean,
        "scale": scale,
        "intercept": _safe_runtime_float(data.get("intercept"), 0.0),
        "accept_threshold": threshold,
        "horizon": data.get("horizon", ""),
        "target_mode": data.get("target_mode", ""),
        "feature_set": data.get("feature_set", ""),
        "source_C_exploration_pass": int(_safe_runtime_float(data.get("C_exploration_pass"), 0.0)),
        "source_C_official_pass": int(_safe_runtime_float(data.get("C_official_pass"), 0.0)),
    }


def _benefit_policy_feature_row(
    *,
    candidate_diag: dict[str, Any],
    source_signal: torch.Tensor,
    u_task_pre: float,
    nds_pre: float,
    state_source_age: float,
    virtual_delta: float,
) -> dict[str, float]:
    source_norm = float(torch.linalg.vector_norm(source_signal.detach().float()).item()) if torch.is_tensor(source_signal) else 0.0
    return {
        "U_task_usefulness": float(u_task_pre),
        "source_loss_boundary_distance": _safe_runtime_float(candidate_diag.get("source_loss_boundary_distance"), 0.0),
        "source_age": _safe_runtime_float(candidate_diag.get("source_age"), state_source_age),
        "source_norm": source_norm,
        "J_reachable_residual_estimate": _safe_runtime_float(candidate_diag.get("jacobian_reachable_projection_residual"), 0.0),
        "JBa_source_cosine_estimate": _safe_runtime_float(candidate_diag.get("JBa_source_cosine"), 0.0),
        "risk_score": _safe_runtime_float(candidate_diag.get("risk_score"), 0.0),
        "destructive_projection": _safe_runtime_float(candidate_diag.get("destructive_projection"), 0.0),
        "NDS": float(nds_pre),
        "control_projection_fraction": _safe_runtime_float(candidate_diag.get("control_projection_fraction"), 0.0),
        "virtual_train_loss_delta_current_step": float(virtual_delta),
        "controller_to_base_update_ratio": _safe_runtime_float(candidate_diag.get("controller_to_base_update_ratio"), 0.0),
        "basis_channel_energy_estimate": _safe_runtime_float(candidate_diag.get("basis_channel_energy_estimate"), 0.0),
        "readout_leakage_estimate": _safe_runtime_float(candidate_diag.get("readout_leakage_estimate"), 0.0),
    }


def _benefit_policy_export_score(policy: dict[str, Any] | None, features: dict[str, float]) -> float | None:
    if not policy:
        return None
    score = float(policy["intercept"])
    for name, coef, mean, scale in zip(policy["feature_names"], policy["coef"], policy["mean"], policy["scale"]):
        value = _safe_runtime_float(features.get(str(name)), 0.0)
        score += float(coef) * ((value - float(mean)) / max(abs(float(scale)), 1.0e-12))
    return float(score)


def _flat_from_grad_list(grads: tuple[torch.Tensor | None, ...], named: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    parts = []
    for grad, (_name, p) in zip(grads, named):
        parts.append(torch.zeros_like(p).reshape(-1) if grad is None else grad.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def _unit_tensor(v: torch.Tensor) -> torch.Tensor:
    vf = v.detach().float().reshape(-1)
    return vf / torch.linalg.vector_norm(vf).clamp_min(1.0e-12)


def _add_direction(named: list[tuple[str, torch.nn.Parameter]], direction: torch.Tensor, scale: float) -> None:
    offset = 0
    with torch.no_grad():
        for _name, p in named:
            n = int(p.numel())
            p.add_(direction[offset : offset + n].reshape_as(p).to(p.device, p.dtype), alpha=scale)
            offset += n


def _finite_effect(model: torch.nn.Module, named: list[tuple[str, torch.nn.Parameter]], xb: torch.Tensor, direction: torch.Tensor, eps: float = 1.0e-3) -> torch.Tensor:
    with torch.no_grad():
        base = model(xb).detach().float().reshape(-1)
        _add_direction(named, direction, eps)
        moved = model(xb).detach().float().reshape(-1)
        _add_direction(named, direction, -eps)
    return (moved - base) / eps


def _virtual_train_loss(model: torch.nn.Module, named: list[tuple[str, torch.nn.Parameter]], xb: torch.Tensor, yb: torch.Tensor, direction: torch.Tensor, scale: float) -> float:
    with torch.no_grad():
        _add_direction(named, direction.detach().float().reshape(-1), float(scale))
    try:
        with torch.no_grad():
            loss = float(F.cross_entropy(model(xb).float(), yb).item())
    finally:
        with torch.no_grad():
            _add_direction(named, direction.detach().float().reshape(-1), -float(scale))
    return loss


def _direction_chunks(named: list[tuple[str, torch.nn.Parameter]], direction: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    flat = direction.detach().float().reshape(-1).to(device)
    chunks: list[torch.Tensor] = []
    offset = 0
    for _name, param in named:
        n = int(param.numel())
        chunks.append(flat[offset : offset + n].reshape_as(param).to(device=device, dtype=param.dtype))
        offset += n
    return tuple(chunks)


def _jvp_logits(
    model: torch.nn.Module,
    named: list[tuple[str, torch.nn.Parameter]],
    xb: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    try:
        from torch.func import functional_call, jvp

        param_map = dict(model.named_parameters())
        names = [name for name, _param in named]
        base_tuple = tuple(param_map[name] for name in names)
        tangent_tuple = _direction_chunks(named, direction, xb.device)

        def logits_fn(*selected_values: torch.Tensor) -> torch.Tensor:
            patched = dict(param_map)
            for name, value in zip(names, selected_values):
                patched[name] = value
            return functional_call(model, patched, (xb,), strict=False).float().reshape(-1)

        _base, tangent = jvp(logits_fn, base_tuple, tangent_tuple)
        return tangent.detach().float().reshape(-1), "torch_func_jvp"
    except Exception:
        return _finite_effect(model, named, xb, direction, eps=1.0e-3).detach().float().reshape(-1), "finite_diff_fallback"


def _candidate_source_loss(
    model: torch.nn.Module,
    named: list[tuple[str, torch.nn.Parameter]],
    xb: torch.Tensor,
    delta: torch.Tensor,
    candidate: torch.Tensor,
) -> float:
    effect, _kind = _jvp_logits(model, named, xb, _unit_tensor(candidate).to(xb.device))
    delta_cpu = delta.detach().float().cpu().reshape(-1)
    effect_cpu = effect.detach().float().cpu().reshape(-1)
    return float((-(delta_cpu * effect_cpu)).mean().item())


def _jvp_history_refresh_update(
    model: torch.nn.Module,
    named: list[tuple[str, torch.nn.Parameter]],
    xb: torch.Tensor,
    yb: torch.Tensor,
    delta: torch.Tensor,
    grad_flat: torch.Tensor,
    update: torch.Tensor,
    history: list[torch.Tensor],
    args: argparse.Namespace,
    *,
    scale_override: float | None = None,
    ratio_cap_override: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    diag: dict[str, Any] = {
        "controller_policy": "jvp_useful_history_refresh",
        "source_candidate_type": "jvp_useful_history",
        "intervention_flag": 0,
        "lambda_t": 0.0,
        "risk_score": 0.0,
        "source_release_count": 0,
        "source_refresh_count": 0,
        "gate_reject_reason": "",
    }
    if not named or grad_flat.numel() == 0 or update.numel() == 0:
        diag["gate_reject_reason"] = "empty_selected_grad"
        return update, diag
    if len(history) < int(args.jvp_refresh_min_history):
        diag["gate_reject_reason"] = "insufficient_jvp_history"
        diag["jvp_history_count"] = len(history)
        return update, diag

    hist = [h.detach().float().cpu() for h in history[-int(args.jvp_refresh_history_window) :] if h.numel() == update.numel()]
    if len(hist) < int(args.jvp_refresh_min_history):
        diag["gate_reject_reason"] = "insufficient_matching_jvp_history"
        diag["jvp_history_count"] = len(hist)
        return update, diag

    base_candidate = update.detach().float().cpu()
    base_score = _candidate_source_loss(model, named, xb, delta.detach().cpu(), base_candidate)
    scores = [_candidate_source_loss(model, named, xb, delta.detach().cpu(), h) for h in hist]
    best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
    best_score = float(scores[best_idx])
    if best_score <= 0.0:
        diag.update(
            {
                "gate_reject_reason": "nonpositive_jvp_history_source_loss",
                "jvp_history_count": len(hist),
                "jvp_history_best_source_loss": best_score,
                "jvp_current_base_source_loss": base_score,
            }
        )
        return update, diag

    target = hist[best_idx]
    target_unit = _unit_tensor(target).to(xb.device)
    base_effect, jvp_kind = _jvp_logits(model, named, xb, target_unit)
    cols = [_unit_tensor(v).to(xb.device) for v in hist[-int(args.jvp_refresh_dim) :]]
    effects: list[torch.Tensor] = []
    for vec in cols:
        eff, kind = _jvp_logits(model, named, xb, vec)
        jvp_kind = jvp_kind if kind == jvp_kind else f"{jvp_kind}+{kind}"
        effects.append(eff.detach().float().cpu())
    if not effects:
        diag["gate_reject_reason"] = "empty_jvp_effect_basis"
        return update, diag

    e_mat = torch.stack(effects, dim=1).float()
    tgt = base_effect.detach().float().cpu().reshape(-1)
    eye = torch.eye(e_mat.shape[1], dtype=e_mat.dtype)
    try:
        coeff = torch.linalg.solve(e_mat.T @ e_mat + float(args.jvp_refresh_ridge) * eye, e_mat.T @ tgt)
    except RuntimeError:
        coeff = torch.linalg.lstsq(e_mat, tgt).solution
    proj = e_mat @ coeff
    residual = float(torch.linalg.vector_norm(tgt - proj).div(torch.linalg.vector_norm(tgt).clamp_min(1.0e-12)).item())
    ctrl = torch.zeros_like(base_candidate)
    for c, vec in zip(coeff, cols):
        ctrl += float(c.item()) * vec.detach().float().cpu()
    ctrl_norm = float(torch.linalg.vector_norm(ctrl).item())
    ratio_cap = float(args.jvp_refresh_ratio_cap) if ratio_cap_override is None else float(ratio_cap_override)
    scale = float(args.jvp_refresh_scale) if scale_override is None else float(scale_override)
    cap = min(1.0, ratio_cap / max(ctrl_norm, 1.0e-12))
    ctrl = ctrl * cap
    scaled_ctrl = ctrl.to(update.device) * scale
    guided = update.detach().float() + scaled_ctrl.to(update.device)

    base_virtual_loss = _virtual_train_loss(model, named, xb, yb, update, float(args.virtual_gate_scale))
    guided_virtual_loss = _virtual_train_loss(model, named, xb, yb, guided, float(args.virtual_gate_scale))
    predicted_gain_base = float((-torch.dot(grad_flat.detach().float().reshape(-1), update.detach().float().reshape(-1))).item())
    predicted_gain_guided = float((-torch.dot(grad_flat.detach().float().reshape(-1), guided.detach().float().reshape(-1))).item())
    ctrl_effect, ctrl_kind = _jvp_logits(model, named, xb, ctrl.to(xb.device))
    jvp_kind = jvp_kind if ctrl_kind == jvp_kind else f"{jvp_kind}+{ctrl_kind}"
    source_gain = float((-(delta.detach().float().cpu().reshape(-1) * ctrl_effect.detach().float().cpu().reshape(-1))).mean().item())
    diag.update(
        {
            "jvp_reference": jvp_kind,
            "jvp_history_count": len(hist),
            "jvp_history_best_source_loss": best_score,
            "jvp_current_base_source_loss": base_score,
            "jvp_history_gain_vs_current_base": best_score - base_score,
            "jacobian_reachable_projection_residual": residual,
            "controller_update_norm": float(torch.linalg.vector_norm(scaled_ctrl.detach().float()).item()),
            "controller_to_base_update_ratio": float(torch.linalg.vector_norm(scaled_ctrl.detach().float()).div(torch.linalg.vector_norm(update.detach().float()).clamp_min(1.0e-12)).item()),
            "controller_cap_scale": cap,
            "controller_source_loss_gain": source_gain,
            "predicted_gain_base": predicted_gain_base,
            "predicted_gain_guided": predicted_gain_guided,
            "virtual_loss_base": base_virtual_loss,
            "virtual_loss_guided": guided_virtual_loss,
            "source_refresh_count": 1,
            "source_age": 0,
        }
    )
    if source_gain <= 0.0:
        diag["gate_reject_reason"] = "jvp_controller_source_loss_nonpositive"
        return update, diag
    if guided_virtual_loss > base_virtual_loss + float(args.virtual_loss_tolerance):
        diag["gate_reject_reason"] = "jvp_virtual_train_loss_gate_failed"
        return update, diag
    if bool(getattr(args, "jvp_refresh_require_virtual_improvement", False)):
        margin = float(getattr(args, "jvp_refresh_virtual_improvement_margin", 0.0))
        if guided_virtual_loss >= base_virtual_loss - margin:
            diag["gate_reject_reason"] = "jvp_virtual_train_loss_not_improved"
            return update, diag
    if predicted_gain_guided < predicted_gain_base * (1.0 - float(args.actionable_gain_tolerance)):
        diag["gate_reject_reason"] = "jvp_predicted_gain_gate_failed"
        return update, diag
    diag["intervention_flag"] = 1
    return guided.to(dtype=grad_flat.dtype), diag


def _pca_basis(vectors: list[torch.Tensor], dim: int) -> torch.Tensor | None:
    if len(vectors) < 2:
        return None
    mat = torch.stack([v.detach().float().cpu().reshape(-1) for v in vectors])
    mat = mat - mat.mean(dim=0, keepdim=True)
    try:
        _u, _s, vh = torch.linalg.svd(mat, full_matrices=False)
    except RuntimeError:
        return None
    return vh[: min(dim, vh.shape[0])].contiguous()


def _projection_residual(target: torch.Tensor, basis: torch.Tensor) -> float:
    target = target.detach().float().cpu().reshape(-1)
    basis = basis.detach().float().cpu()
    if basis.numel() == 0:
        return 1.0
    coeff = basis @ target
    proj = coeff @ basis
    return float(torch.linalg.vector_norm(target - proj).div(torch.linalg.vector_norm(target).clamp_min(1.0e-12)).item())


def _slice_direction(
    full_named: list[tuple[str, torch.nn.Parameter]],
    block_named: list[tuple[str, torch.nn.Parameter]],
    direction: torch.Tensor,
) -> torch.Tensor | None:
    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, param in full_named:
        n = int(param.numel())
        offsets[name] = (offset, n)
        offset += n
    chunks: list[torch.Tensor] = []
    flat = direction.detach().float().reshape(-1)
    for name, _param in block_named:
        item = offsets.get(name)
        if item is None:
            return None
        start, n = item
        chunks.append(flat[start : start + n])
    return torch.cat(chunks) if chunks else None


def _effect_matrix_stats(eff_basis: torch.Tensor) -> dict[str, Any]:
    if eff_basis.numel() == 0:
        return {"basis_condition_number": "", "basis_stability_energy": "", "jacobian_effect_rank": 0}
    try:
        s = torch.linalg.svdvals(eff_basis.detach().float().cpu())
    except RuntimeError:
        return {"basis_condition_number": "", "basis_stability_energy": "", "jacobian_effect_rank": "", "blocker": "EffectSVDFailed"}
    if s.numel() == 0:
        return {"basis_condition_number": "", "basis_stability_energy": "", "jacobian_effect_rank": 0}
    s_max = float(s.max().item())
    rank = int((s > max(1.0e-8, s_max * 1.0e-4)).sum().item())
    s_min = float(s[s > max(1.0e-8, s_max * 1.0e-4)].min().item()) if rank else 0.0
    cond = (s_max / max(s_min, 1.0e-12)) if rank else ""
    energy = float(eff_basis.detach().float().cpu().square().mean().item())
    return {"basis_condition_number": cond, "basis_stability_energy": energy, "jacobian_effect_rank": rank}


def _controllability_diag(
    model: torch.nn.Module,
    named: list[tuple[str, torch.nn.Parameter]],
    xb: torch.Tensor,
    history: list[torch.Tensor],
    useful_flags: list[int],
    dim: int,
) -> dict[str, Any]:
    useful = [v.detach().cpu() for v, flag in zip(history, useful_flags) if int(flag)]
    pool = useful if len(useful) >= max(2, dim) else [v.detach().cpu() for v in history]
    basis = _pca_basis(pool[:-1], dim) if len(pool) >= 3 else None
    if basis is None:
        return {
            "dim": dim,
            "useful_source_count": len(useful),
            "raw_history_projection_residual": "",
            "jacobian_reachable_projection_residual": "",
            "controllability_ratio": "",
            "JBa_source_cosine": "",
            "controls_pass_count": "",
        }
    target = pool[-1]
    raw_res = _projection_residual(target, basis)
    effects = []
    xb_small = xb[: min(16, xb.shape[0])]
    for vec in basis:
        effects.append(_finite_effect(model, named, xb_small, vec.to(xb.device)))
    target_effect = _finite_effect(model, named, xb_small, target.to(xb.device))
    eff_basis = torch.stack([e.detach().float().cpu().reshape(-1) for e in effects], dim=1)
    tgt = target_effect.detach().float().cpu().reshape(-1)
    try:
        sol = torch.linalg.lstsq(eff_basis, tgt).solution
        proj = eff_basis @ sol
        reach_res = float(torch.linalg.vector_norm(tgt - proj).div(torch.linalg.vector_norm(tgt).clamp_min(1.0e-12)).item())
        j_cos = float(torch.dot(proj, tgt).div(torch.linalg.vector_norm(proj).clamp_min(1.0e-12) * torch.linalg.vector_norm(tgt).clamp_min(1.0e-12)).clamp(-1, 1).item())
    except RuntimeError:
        reach_res = float("nan")
        j_cos = float("nan")
    gen = torch.Generator().manual_seed(2217)
    random_basis = torch.randn(basis.shape, generator=gen, dtype=basis.dtype)
    random_basis = torch.linalg.qr(random_basis.T, mode="reduced").Q.T[: basis.shape[0]]
    random_res = _projection_residual(target, random_basis)
    return {
        "dim": dim,
        "useful_source_count": len(useful),
        "raw_history_projection_residual": raw_res,
        "jacobian_reachable_projection_residual": reach_res,
        "controllability_ratio": raw_res / reach_res if reach_res and reach_res == reach_res else "",
        "JBa_source_cosine": j_cos,
        "random_basis_projection_residual": random_res,
        "controls_pass_count": int(random_res <= raw_res),
        **_effect_matrix_stats(eff_basis),
        "blocker": "ControllabilityResidualHigh" if reach_res == reach_res and reach_res > 0.50 else "",
    }


def _controllability_diag_rows(
    model: torch.nn.Module,
    named: list[tuple[str, torch.nn.Parameter]],
    xb: torch.Tensor,
    history: list[torch.Tensor],
    useful_flags: list[int],
    dim: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "manifold_family": "PositiveUsefulnessPCA_finite_diff_J",
            **_controllability_diag(model, named, xb, history, useful_flags, dim),
        }
    ]
    useful = [v.detach().cpu() for v, flag in zip(history, useful_flags) if int(flag)]
    pool = useful if len(useful) >= max(2, dim) else [v.detach().cpu() for v in history]
    if len(pool) < 3:
        rows.append(
            {
                "manifold_family": "JacobianEffectPCA_finite_diff_J",
                "dim": dim,
                "useful_source_count": len(useful),
                "raw_history_projection_residual": "",
                "jacobian_reachable_projection_residual": "",
                "controllability_ratio": "",
                "JBa_source_cosine": "",
                "random_basis_projection_residual": "",
                "controls_pass_count": "",
                "blocker": "InsufficientHistoryForJSpacePCA",
            }
        )
        return rows
    target = pool[-1]
    source_basis = _pca_basis(pool[:-1], dim)
    raw_res = _projection_residual(target, source_basis) if source_basis is not None else ""
    xb_small = xb[: min(16, xb.shape[0])]
    effect_history = []
    for vec in pool[:-1][-min(16, len(pool) - 1) :]:
        effect_history.append(_finite_effect(model, named, xb_small, vec.to(xb.device)).detach().float().cpu().reshape(-1))
    target_effect = _finite_effect(model, named, xb_small, target.to(xb.device)).detach().float().cpu().reshape(-1)
    effect_basis = _pca_basis(effect_history, dim)
    if effect_basis is None:
        rows.append(
            {
                "manifold_family": "JacobianEffectPCA_finite_diff_J",
                "dim": dim,
                "useful_source_count": len(useful),
                "raw_history_projection_residual": raw_res,
                "jacobian_reachable_projection_residual": "",
                "controllability_ratio": "",
                "JBa_source_cosine": "",
                "random_basis_projection_residual": "",
                "controls_pass_count": "",
                "blocker": "EffectPCAFailed",
            }
        )
        return rows
    effect_res = _projection_residual(target_effect, effect_basis)
    coeff = effect_basis @ target_effect
    proj = coeff @ effect_basis
    effect_cos = float(
        torch.dot(proj, target_effect)
        .div(torch.linalg.vector_norm(proj).clamp_min(1.0e-12) * torch.linalg.vector_norm(target_effect).clamp_min(1.0e-12))
        .clamp(-1, 1)
        .item()
    )
    gen = torch.Generator().manual_seed(2218)
    random_basis = torch.randn(effect_basis.shape, generator=gen, dtype=effect_basis.dtype)
    random_basis = torch.linalg.qr(random_basis.T, mode="reduced").Q.T[: effect_basis.shape[0]]
    random_res = _projection_residual(target_effect, random_basis)
    effect_stats = _effect_matrix_stats(effect_basis.T)
    blocker = ""
    if effect_stats.get("jacobian_effect_rank") != "" and int(effect_stats.get("jacobian_effect_rank") or 0) < min(dim, effect_basis.shape[0]):
        blocker = "ControllabilityRankBlocked"
    elif effect_res > 0.50:
        blocker = "ControllabilityResidualHigh"
    rows.append(
        {
            "manifold_family": "JacobianEffectPCA_finite_diff_J",
            "dim": min(dim, effect_basis.shape[0]),
            "history_window": len(effect_history),
            "useful_source_count": len(useful),
            "raw_history_projection_residual": raw_res,
            "jacobian_reachable_projection_residual": effect_res,
            "controllability_ratio": (float(raw_res) / effect_res) if raw_res != "" and effect_res else "",
            "JBa_source_cosine": effect_cos,
            "JBa_source_loss": "",
            "U_task_usefulness_after_projection": "",
            "basis_condition_number": "",
            "basis_stability_energy": "",
            "controller_to_base_update_ratio": "",
            "source_func_h3200": "",
            "source_loss_h3200": "",
            "source_func_h4800": "",
            "source_loss_h4800": "",
            "random_basis_projection_residual": random_res,
            "controls_pass_count": int(random_res <= effect_res),
            **effect_stats,
            "blocker": blocker,
        }
    )
    full_names = {name for name, _param in named}
    for block_name, block_named in [
        ("basis", selected_named_parameters(model, "basis")),
        ("readout", selected_named_parameters(model, "readout")),
    ]:
        if not block_named or any(name not in full_names for name, _param in block_named):
            continue
        block_pool: list[torch.Tensor] = []
        for vec in pool:
            sliced = _slice_direction(named, block_named, vec)
            if sliced is None or sliced.numel() == 0:
                block_pool = []
                break
            block_pool.append(sliced.cpu())
        if len(block_pool) < 3:
            continue
        block_diag = _controllability_diag(model, block_named, xb, block_pool, [1] * len(block_pool), dim=min(dim, len(block_pool) - 1))
        block_res = block_diag.get("jacobian_reachable_projection_residual", "")
        block_rank = block_diag.get("jacobian_effect_rank", "")
        block_blocker = block_diag.get("blocker", "")
        if block_rank != "" and int(block_rank or 0) < int(block_diag.get("dim") or dim):
            block_blocker = "ControllabilityRankBlocked"
        elif block_res != "" and float(block_res) > 0.50:
            block_blocker = "ControllabilityResidualHigh"
        rows.append(
            {
                "manifold_family": f"BlockJacobianEffectPCA_{block_name}",
                "parameter_block": block_name,
                **block_diag,
                "blocker": block_blocker,
            }
        )
    return rows


def _out_path(name: str, output_suffix: str) -> Path:
    path = OUT_ROOT / name
    suffix = str(output_suffix or "").strip().strip("_")
    if not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _train_one(dataset: str, seed: int, model_name: str, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_loader, test_loader, output_dim, input_dim = _get_loaders(Path(args.kanbefair_root), dataset, args.batch_size, args.train_size, args.test_size, seed)
    first_x, first_y = next(iter(train_loader))
    initial_batch_fingerprint = _batch_fingerprint(first_x, first_y)
    model = _make_model(model_name, input_dim, output_dim, args.hidden, seed + 2217, device, first_x.float()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    fu_enabled = _is_fu_model(model_name)
    controller_policy = _controller_policy(model_name)
    source_candidate_type = _source_candidate_type(model_name)
    benefit_policy_export = getattr(args, "_benefit_policy_export", None)
    selector = "basis" if model_name.startswith("DGKAN") and fu_enabled else "all"
    selected = selected_named_parameters(model, selector)
    all_params = [p for p in model.parameters() if p.requires_grad]
    state = MLPFUState()
    train_iter = iter(train_loader)
    losses: list[float] = []
    source_rows: list[dict[str, Any]] = []
    neutral_control_rows: list[dict[str, Any]] = []
    history: list[torch.Tensor] = []
    useful_flags: list[int] = []
    intervention_count = 0
    overhead_time = 0.0
    total_step_time = 0.0
    last_diag: dict[str, Any] = {}
    xb_last = first_x.to(device).float()
    gate_accept_count = 0
    gate_reject_count = 0
    gate_reject_reason_counts: dict[str, int] = {}
    predicted_gain_base_values: list[float] = []
    predicted_gain_guided_values: list[float] = []
    split_cosine_values: list[float] = []
    controller_source_gain_values: list[float] = []
    virtual_loss_base_values: list[float] = []
    virtual_loss_guided_values: list[float] = []
    jvp_refresh_source_gain_values: list[float] = []
    jvp_refresh_update_ratio_values: list[float] = []
    benefit_p3_score_values: list[float] = []
    benefit_p3_accept_score_values: list[float] = []
    benefit_policy_export_score_values: list[float] = []
    benefit_policy_export_accept_score_values: list[float] = []
    for step in range(1, int(args.steps) + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        xb_last = xb
        model.train()
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits_before = model(xb).float()
        loss = F.cross_entropy(logits_before, yb)
        split_source_flat: torch.Tensor | None = None
        split_consensus_cosine: float | str = ""
        if fu_enabled and source_candidate_type == "split_consensus_param" and xb.shape[0] >= 2:
            mid = int(xb.shape[0] // 2)
            params_for_grad = [p for _n, p in selected]
            loss_a = F.cross_entropy(logits_before[:mid], yb[:mid])
            loss_b = F.cross_entropy(logits_before[mid:], yb[mid:])
            grads_a = torch.autograd.grad(loss_a, params_for_grad, retain_graph=True, allow_unused=True)
            grads_b = torch.autograd.grad(loss_b, params_for_grad, retain_graph=True, allow_unused=True)
            update_a = -_flat_from_grad_list(grads_a, selected).to(device).float()
            update_b = -_flat_from_grad_list(grads_b, selected).to(device).float()
            if update_a.numel() and update_b.numel():
                split_consensus_cosine = _cos(update_a, update_b)
                if float(split_consensus_cosine) > 0.0:
                    split_source_flat = _unit_tensor(update_a) + _unit_tensor(update_b)
                else:
                    split_source_flat = _unit_tensor(update_a + update_b)
        loss.backward()
        delta = ce_cotangent(logits_before.detach(), yb).reshape(-1)
        source_signal = (-delta).detach()
        grad_flat = _flat_grads(selected).detach()
        update = -grad_flat.float()
        update_norm2 = float(torch.dot(update.reshape(-1), update.reshape(-1)).item()) if update.numel() else 0.0
        source_loss_gain = float((delta.detach().reshape(-1).square()).mean().item())
        source_signal_for_controller = source_signal
        controller_source_loss_gain = source_loss_gain
        if fu_enabled and grad_flat.numel() and source_candidate_type == "param_update":
            source_signal_for_controller = update.detach().clone()
        elif fu_enabled and grad_flat.numel() and source_candidate_type == "split_consensus_param":
            source_signal_for_controller = (split_source_flat if split_source_flat is not None and split_source_flat.numel() == grad_flat.numel() else update.detach().clone())
        if fu_enabled and grad_flat.numel() and source_signal_for_controller.numel() == grad_flat.numel():
            src_unit_for_gain = _unit_tensor(source_signal_for_controller).to(grad_flat.device)
            controller_source_loss_gain = float((-torch.dot(grad_flat.detach().float().reshape(-1), src_unit_for_gain)).div(max(1.0, float(grad_flat.numel()) ** 0.5)).item())
        if controller_source_loss_gain == controller_source_loss_gain:
            controller_source_gain_values.append(float(controller_source_loss_gain))
        if isinstance(split_consensus_cosine, float):
            split_cosine_values.append(float(split_consensus_cosine))
        nds_pre = float(source_signal.abs().mean().item())
        tail_proxy_pre = float(torch.quantile(delta.detach().abs().float(), 0.95).item()) if delta.numel() else 0.0
        gen_pre = torch.Generator(device=delta.device).manual_seed(550000 + int(seed) + step)
        random_source_pre = torch.randn(source_signal.numel(), generator=gen_pre, device=source_signal.device)
        random_source_loss_gain = float((-(delta.to(random_source_pre.device).reshape(-1) * random_source_pre.reshape(-1))).mean().item())
        u_task_pre = _u_score_source(controller_source_loss_gain, nds_pre, tail_proxy_pre, 0.0, float(state.source_age))
        random_u_pre = _u_score_source(random_source_loss_gain, float(random_source_pre.abs().mean().item()), tail_proxy_pre, 0.0, float(state.source_age))
        diag: dict[str, Any] = {
            "risk_score": 0.0,
            "lambda_t": 0.0,
            "intervention_flag": 0,
            "source_func": 0.0,
            "source_loss_gain": source_loss_gain,
            "controller_source_loss_gain": controller_source_loss_gain,
            "source_age": 0,
            "source_release_count": 0,
            "source_refresh_count": 0,
            "destructive_projection": 0.0,
            "NDS": nds_pre,
            "control_projection_fraction": 0.0,
            "controller_to_base_update_ratio": 0.0,
            "controller_policy": controller_policy,
            "source_candidate_type": source_candidate_type,
            "split_consensus_cosine": split_consensus_cosine,
            "task_useful_gate_pre": int(u_task_pre > random_u_pre + float(args.gate_margin)),
            "U_task_usefulness_pre": u_task_pre,
            "random_control_U_pre": random_u_pre,
            "gate_reject_reason": "",
            "predicted_gain_base": "",
            "predicted_gain_guided": "",
            "virtual_loss_base": "",
            "virtual_loss_guided": "",
            "benefit_p3_score": "",
            "benefit_p3_manual_score": "",
            "benefit_p3_min_score": "",
            "runtime_per_step_policy_integrated": int(controller_policy.startswith("benefit_p3")),
            "benefit_policy_export_runtime_integrated": int(bool(benefit_policy_export) and controller_policy.startswith("benefit_p3")),
            "benefit_policy_export_score": "",
            "benefit_policy_export_threshold": "",
        }
        if fu_enabled and grad_flat.numel():
            if controller_policy.startswith("benefit_p3_jvp_current_step") and step % max(1, int(args.benefit_p3_jvp_interval)) == 0:
                t0 = time.perf_counter()
                guided, candidate_diag = _jvp_history_refresh_update(
                    model,
                    selected,
                    xb,
                    yb,
                    delta.detach(),
                    grad_flat,
                    update,
                    history,
                    args,
                    scale_override=float(args.benefit_p3_jvp_strong_scale) if controller_policy.endswith("_strong") else None,
                    ratio_cap_override=float(args.benefit_p3_jvp_strong_ratio_cap) if controller_policy.endswith("_strong") else None,
                )
                candidate_diag["controller_policy"] = controller_policy
                candidate_diag["task_useful_gate_pre"] = int(u_task_pre > random_u_pre + float(args.gate_margin))
                candidate_diag["U_task_usefulness_pre"] = u_task_pre
                candidate_diag["random_control_U_pre"] = random_u_pre
                candidate_diag["split_consensus_cosine"] = split_consensus_cosine
                candidate_diag.setdefault("NDS", nds_pre)
                candidate_diag.setdefault("source_age", 0)
                candidate_diag.setdefault("control_projection_fraction", 0.0)
                candidate_diag.setdefault("source_loss_gain", source_loss_gain)
                virtual_delta = float(candidate_diag.get("virtual_loss_guided", 0.0) or 0.0) - float(candidate_diag.get("virtual_loss_base", 0.0) or 0.0)
                manual_score = _runtime_benefit_p3_score(
                    source_loss_gain=float(candidate_diag.get("controller_source_loss_gain", source_loss_gain) or 0.0),
                    u_task_pre=u_task_pre,
                    random_u_pre=random_u_pre,
                    nds=nds_pre,
                    tail_proxy=tail_proxy_pre,
                    risk_score=float(candidate_diag.get("risk_score", 0.0) or 0.0),
                    update_ratio=float(candidate_diag.get("controller_to_base_update_ratio", 0.0) or 0.0),
                    virtual_loss_delta=virtual_delta,
                    predicted_gain_base=float(candidate_diag.get("predicted_gain_base", 0.0) or 0.0),
                    predicted_gain_guided=float(candidate_diag.get("predicted_gain_guided", 0.0) or 0.0),
                    source_age=float(candidate_diag.get("source_age", state.source_age) or 0.0),
                )
                score = manual_score
                min_score = _benefit_p3_min_score(controller_policy, args)
                candidate_diag["benefit_p3_manual_score"] = manual_score
                if benefit_policy_export:
                    export_features = _benefit_policy_feature_row(
                        candidate_diag=candidate_diag,
                        source_signal=source_signal,
                        u_task_pre=u_task_pre,
                        nds_pre=nds_pre,
                        state_source_age=float(state.source_age),
                        virtual_delta=virtual_delta,
                    )
                    export_score = _benefit_policy_export_score(benefit_policy_export, export_features)
                    if export_score is not None:
                        score = export_score
                        min_score = float(benefit_policy_export["accept_threshold"])
                        candidate_diag["benefit_policy_export_score"] = export_score
                        candidate_diag["benefit_policy_export_threshold"] = min_score
                        candidate_diag["benefit_policy_export_runtime_integrated"] = 1
                        candidate_diag["benefit_policy_export_path"] = benefit_policy_export["path"]
                        candidate_diag["benefit_policy_export_horizon"] = benefit_policy_export.get("horizon", "")
                        candidate_diag["benefit_policy_export_target_mode"] = benefit_policy_export.get("target_mode", "")
                        candidate_diag["benefit_policy_export_feature_set"] = benefit_policy_export.get("feature_set", "")
                        for feature_name, feature_value in export_features.items():
                            candidate_diag[f"benefit_policy_feature_{feature_name}"] = feature_value
                        benefit_policy_export_score_values.append(export_score)
                candidate_diag["benefit_p3_score"] = score
                candidate_diag["benefit_p3_min_score"] = min_score
                candidate_diag["runtime_per_step_policy_integrated"] = 1
                benefit_p3_score_values.append(score)
                if candidate_diag.get("virtual_loss_base") != "":
                    virtual_loss_base_values.append(float(candidate_diag.get("virtual_loss_base", 0.0)))
                if candidate_diag.get("virtual_loss_guided") != "":
                    virtual_loss_guided_values.append(float(candidate_diag.get("virtual_loss_guided", 0.0)))
                if candidate_diag.get("controller_source_loss_gain") != "":
                    jvp_refresh_source_gain_values.append(float(candidate_diag.get("controller_source_loss_gain", 0.0)))
                if candidate_diag.get("controller_to_base_update_ratio") != "":
                    jvp_refresh_update_ratio_values.append(float(candidate_diag.get("controller_to_base_update_ratio", 0.0)))
                reject_reason = str(candidate_diag.get("gate_reject_reason", ""))
                if not reject_reason and score <= min_score:
                    reject_reason = "benefit_policy_export_score_gate_failed" if benefit_policy_export else "benefit_p3_score_gate_failed"
                    candidate_diag["gate_reject_reason"] = reject_reason
                if reject_reason:
                    candidate_diag["intervention_flag"] = 0
                    candidate_diag["lambda_t"] = 0.0
                    gate_reject_count += 1
                    gate_reject_reason_counts[reject_reason] = gate_reject_reason_counts.get(reject_reason, 0) + 1
                    diag.update(candidate_diag)
                else:
                    diag = candidate_diag
                    _assign_flat_update(selected, guided)
                    gate_accept_count += 1
                    benefit_p3_accept_score_values.append(score)
                    if benefit_policy_export and candidate_diag.get("benefit_policy_export_score") != "":
                        benefit_policy_export_accept_score_values.append(float(candidate_diag.get("benefit_policy_export_score", 0.0)))
                    predicted_gain_base_values.append(float(candidate_diag.get("predicted_gain_base", 0.0)))
                    predicted_gain_guided_values.append(float(candidate_diag.get("predicted_gain_guided", 0.0)))
                overhead_time += time.perf_counter() - t0
            elif controller_policy.startswith("benefit_p3_jvp_current_step"):
                reject_reason = "benefit_p3_jvp_interval_skip"
                diag["gate_reject_reason"] = reject_reason
                diag["benefit_p3_min_score"] = _benefit_p3_min_score(controller_policy, args)
                diag["runtime_per_step_policy_integrated"] = 1
                diag["benefit_policy_export_runtime_integrated"] = int(bool(benefit_policy_export))
                if benefit_policy_export:
                    diag["benefit_policy_export_threshold"] = float(benefit_policy_export["accept_threshold"])
                    diag["benefit_policy_export_path"] = benefit_policy_export["path"]
                gate_reject_count += 1
                gate_reject_reason_counts[reject_reason] = gate_reject_reason_counts.get(reject_reason, 0) + 1
            elif controller_policy == "jvp_useful_history_refresh":
                t0 = time.perf_counter()
                guided, candidate_diag = _jvp_history_refresh_update(
                    model,
                    selected,
                    xb,
                    yb,
                    delta.detach(),
                    grad_flat,
                    update,
                    history,
                    args,
                )
                candidate_diag["task_useful_gate_pre"] = int(u_task_pre > random_u_pre + float(args.gate_margin))
                candidate_diag["U_task_usefulness_pre"] = u_task_pre
                candidate_diag["random_control_U_pre"] = random_u_pre
                candidate_diag["split_consensus_cosine"] = split_consensus_cosine
                candidate_diag.setdefault("NDS", nds_pre)
                candidate_diag.setdefault("source_age", 0)
                candidate_diag.setdefault("control_projection_fraction", 0.0)
                candidate_diag.setdefault("source_loss_gain", source_loss_gain)
                if candidate_diag.get("virtual_loss_base") != "":
                    virtual_loss_base_values.append(float(candidate_diag.get("virtual_loss_base", 0.0)))
                if candidate_diag.get("virtual_loss_guided") != "":
                    virtual_loss_guided_values.append(float(candidate_diag.get("virtual_loss_guided", 0.0)))
                if candidate_diag.get("controller_source_loss_gain") != "":
                    jvp_refresh_source_gain_values.append(float(candidate_diag.get("controller_source_loss_gain", 0.0)))
                if candidate_diag.get("controller_to_base_update_ratio") != "":
                    jvp_refresh_update_ratio_values.append(float(candidate_diag.get("controller_to_base_update_ratio", 0.0)))
                reject_reason = str(candidate_diag.get("gate_reject_reason", ""))
                if reject_reason:
                    gate_reject_count += 1
                    gate_reject_reason_counts[reject_reason] = gate_reject_reason_counts.get(reject_reason, 0) + 1
                    diag.update(candidate_diag)
                else:
                    diag = candidate_diag
                    _assign_flat_update(selected, guided)
                    gate_accept_count += 1
                    predicted_gain_base_values.append(float(candidate_diag.get("predicted_gain_base", 0.0)))
                    predicted_gain_guided_values.append(float(candidate_diag.get("predicted_gain_guided", 0.0)))
                    overhead_time += time.perf_counter() - t0
            elif controller_policy.startswith("benefit_p3_current_step"):
                t0 = time.perf_counter()
                old_state = copy.deepcopy(state)
                variant = _controller_variant(controller_policy, model_name)
                guided, new_state, candidate_diag = apply_mlp_fu_update(
                    grad_flat,
                    state,
                    variant=variant,
                    source_signal=source_signal_for_controller,
                    source_loss_gain=controller_source_loss_gain,
                    step=step,
                    seed=seed,
                    ratio_cap=_ratio_cap_for_policy(controller_policy, args),
                )
                predicted_gain_base = float((-torch.dot(grad_flat.detach().float().reshape(-1), update.reshape(-1))).item())
                predicted_gain_guided = float((-torch.dot(grad_flat.detach().float().reshape(-1), guided.detach().float().reshape(-1))).item())
                virtual_loss_base = _virtual_train_loss(model, selected, xb, yb, update, float(args.virtual_gate_scale))
                virtual_loss_guided = _virtual_train_loss(model, selected, xb, yb, guided, float(args.virtual_gate_scale))
                virtual_delta = float(virtual_loss_guided) - float(virtual_loss_base)
                candidate_diag["controller_policy"] = controller_policy
                candidate_diag["source_candidate_type"] = source_candidate_type
                candidate_diag["split_consensus_cosine"] = split_consensus_cosine
                candidate_diag["controller_source_loss_gain"] = controller_source_loss_gain
                candidate_diag["task_useful_gate_pre"] = int(u_task_pre > random_u_pre + float(args.gate_margin))
                candidate_diag["U_task_usefulness_pre"] = u_task_pre
                candidate_diag["random_control_U_pre"] = random_u_pre
                candidate_diag["predicted_gain_base"] = predicted_gain_base
                candidate_diag["predicted_gain_guided"] = predicted_gain_guided
                candidate_diag["virtual_loss_base"] = virtual_loss_base
                candidate_diag["virtual_loss_guided"] = virtual_loss_guided
                manual_score = _runtime_benefit_p3_score(
                    source_loss_gain=controller_source_loss_gain,
                    u_task_pre=u_task_pre,
                    random_u_pre=random_u_pre,
                    nds=nds_pre,
                    tail_proxy=tail_proxy_pre,
                    risk_score=float(candidate_diag.get("risk_score", 0.0) or 0.0),
                    update_ratio=float(candidate_diag.get("controller_to_base_update_ratio", 0.0) or 0.0),
                    virtual_loss_delta=virtual_delta,
                    predicted_gain_base=predicted_gain_base,
                    predicted_gain_guided=predicted_gain_guided,
                    source_age=float(candidate_diag.get("source_age", state.source_age) or 0.0),
                )
                score = manual_score
                min_score = _benefit_p3_min_score(controller_policy, args)
                candidate_diag["benefit_p3_manual_score"] = manual_score
                if benefit_policy_export:
                    export_features = _benefit_policy_feature_row(
                        candidate_diag=candidate_diag,
                        source_signal=source_signal,
                        u_task_pre=u_task_pre,
                        nds_pre=nds_pre,
                        state_source_age=float(state.source_age),
                        virtual_delta=virtual_delta,
                    )
                    export_score = _benefit_policy_export_score(benefit_policy_export, export_features)
                    if export_score is not None:
                        score = export_score
                        min_score = float(benefit_policy_export["accept_threshold"])
                        candidate_diag["benefit_policy_export_score"] = export_score
                        candidate_diag["benefit_policy_export_threshold"] = min_score
                        candidate_diag["benefit_policy_export_runtime_integrated"] = 1
                        candidate_diag["benefit_policy_export_path"] = benefit_policy_export["path"]
                        candidate_diag["benefit_policy_export_horizon"] = benefit_policy_export.get("horizon", "")
                        candidate_diag["benefit_policy_export_target_mode"] = benefit_policy_export.get("target_mode", "")
                        candidate_diag["benefit_policy_export_feature_set"] = benefit_policy_export.get("feature_set", "")
                        for feature_name, feature_value in export_features.items():
                            candidate_diag[f"benefit_policy_feature_{feature_name}"] = feature_value
                        benefit_policy_export_score_values.append(export_score)
                candidate_diag["benefit_p3_score"] = score
                candidate_diag["benefit_p3_min_score"] = min_score
                candidate_diag["runtime_per_step_policy_integrated"] = 1
                benefit_p3_score_values.append(score)
                virtual_loss_base_values.append(float(virtual_loss_base))
                virtual_loss_guided_values.append(float(virtual_loss_guided))
                gain_failed = predicted_gain_guided < predicted_gain_base * (1.0 - float(args.actionable_gain_tolerance))
                virtual_failed = float(virtual_loss_guided) > float(virtual_loss_base) + float(args.virtual_loss_tolerance)
                score_failed = score <= min_score
                if gain_failed or virtual_failed or score_failed:
                    state = old_state
                    reject_reason = (
                        "virtual_train_loss_gate_failed"
                        if virtual_failed
                        else "predicted_gain_gate_failed"
                        if gain_failed
                        else "benefit_policy_export_score_gate_failed"
                        if benefit_policy_export
                        else "benefit_p3_score_gate_failed"
                    )
                    candidate_diag["intervention_flag"] = 0
                    candidate_diag["lambda_t"] = 0.0
                    gate_reject_count += 1
                    gate_reject_reason_counts[reject_reason] = gate_reject_reason_counts.get(reject_reason, 0) + 1
                    diag.update(candidate_diag)
                    diag["gate_reject_reason"] = reject_reason
                else:
                    state = new_state
                    diag = candidate_diag
                    _assign_flat_update(selected, guided)
                    gate_accept_count += 1
                    benefit_p3_accept_score_values.append(score)
                    if benefit_policy_export and candidate_diag.get("benefit_policy_export_score") != "":
                        benefit_policy_export_accept_score_values.append(float(candidate_diag.get("benefit_policy_export_score", 0.0)))
                    predicted_gain_base_values.append(predicted_gain_base)
                    predicted_gain_guided_values.append(predicted_gain_guided)
                overhead_time += time.perf_counter() - t0
            else:
                reject_reason = ""
                if controller_policy == "noop_equivalence":
                    reject_reason = "policy_noop_equivalence"
                elif controller_policy in {"task_useful_gate", "task_useful_actionable_gain_gate", "task_useful_release_low_ratio"}:
                    if not (u_task_pre > random_u_pre + float(args.gate_margin)):
                        reject_reason = "task_usefulness_gate_failed"
                    elif source_loss_gain <= 0.0:
                        reject_reason = "nonpositive_source_loss_gain"
                if reject_reason:
                    gate_reject_count += 1
                    gate_reject_reason_counts[reject_reason] = gate_reject_reason_counts.get(reject_reason, 0) + 1
                    diag["gate_reject_reason"] = reject_reason
                else:
                    t0 = time.perf_counter()
                    old_state = copy.deepcopy(state)
                    variant = _controller_variant(controller_policy, model_name)
                    guided, new_state, candidate_diag = apply_mlp_fu_update(
                        grad_flat,
                        state,
                        variant=variant,
                        source_signal=source_signal_for_controller,
                        source_loss_gain=controller_source_loss_gain,
                        step=step,
                        seed=seed,
                        ratio_cap=_ratio_cap_for_policy(controller_policy, args),
                    )
                    predicted_gain_base = float((-torch.dot(grad_flat.detach().float().reshape(-1), update.reshape(-1))).item())
                    predicted_gain_guided = float((-torch.dot(grad_flat.detach().float().reshape(-1), guided.detach().float().reshape(-1))).item())
                    candidate_diag["controller_policy"] = controller_policy
                    candidate_diag["source_candidate_type"] = source_candidate_type
                    candidate_diag["split_consensus_cosine"] = split_consensus_cosine
                    candidate_diag["controller_source_loss_gain"] = controller_source_loss_gain
                    candidate_diag["task_useful_gate_pre"] = int(u_task_pre > random_u_pre + float(args.gate_margin))
                    candidate_diag["U_task_usefulness_pre"] = u_task_pre
                    candidate_diag["random_control_U_pre"] = random_u_pre
                    candidate_diag["predicted_gain_base"] = predicted_gain_base
                    candidate_diag["predicted_gain_guided"] = predicted_gain_guided
                    virtual_loss_base: float | str = ""
                    virtual_loss_guided: float | str = ""
                    if controller_policy == "task_useful_virtual_train_loss_gate":
                        virtual_loss_base = _virtual_train_loss(model, selected, xb, yb, update, float(args.virtual_gate_scale))
                        virtual_loss_guided = _virtual_train_loss(model, selected, xb, yb, guided, float(args.virtual_gate_scale))
                        candidate_diag["virtual_loss_base"] = virtual_loss_base
                        candidate_diag["virtual_loss_guided"] = virtual_loss_guided
                        virtual_loss_base_values.append(float(virtual_loss_base))
                        virtual_loss_guided_values.append(float(virtual_loss_guided))
                    gain_failed = controller_policy in {"task_useful_actionable_gain_gate", "task_useful_virtual_train_loss_gate"} and predicted_gain_guided < predicted_gain_base * (1.0 - float(args.actionable_gain_tolerance))
                    virtual_failed = controller_policy == "task_useful_virtual_train_loss_gate" and float(virtual_loss_guided) > float(virtual_loss_base) + float(args.virtual_loss_tolerance)
                    if gain_failed or virtual_failed:
                        state = old_state
                        reject_reason = "virtual_train_loss_gate_failed" if virtual_failed else "predicted_gain_gate_failed"
                        gate_reject_count += 1
                        gate_reject_reason_counts[reject_reason] = gate_reject_reason_counts.get(reject_reason, 0) + 1
                        diag["gate_reject_reason"] = reject_reason
                        diag["predicted_gain_base"] = predicted_gain_base
                        diag["predicted_gain_guided"] = predicted_gain_guided
                        diag["virtual_loss_base"] = virtual_loss_base
                        diag["virtual_loss_guided"] = virtual_loss_guided
                    else:
                        state = new_state
                        diag = candidate_diag
                        _assign_flat_update(selected, guided)
                        gate_accept_count += 1
                        predicted_gain_base_values.append(predicted_gain_base)
                        predicted_gain_guided_values.append(predicted_gain_guided)
                    overhead_time += time.perf_counter() - t0
        torch.nn.utils.clip_grad_norm_(all_params, 2.0)
        opt.step()
        with torch.no_grad():
            logits_after = model(xb).float()
        effect = (logits_after - logits_before.detach()).reshape(-1)
        source_func = _cos(effect, source_signal.to(effect.device)) if effect.numel() == source_signal.numel() else float(diag.get("source_func", 0.0))
        source_loss_after = float((-(delta.to(effect.device).reshape(-1) * effect.detach().reshape(-1))).mean().item()) if effect.numel() == delta.numel() else source_loss_gain
        linear_gain = max(0.0, update_norm2)
        nds = float(diag.get("NDS", 0.0))
        tail_proxy = tail_proxy_pre
        staleness = float(diag.get("source_age", 0.0))
        u_task = _u_score_source(source_loss_after, nds, tail_proxy, float(diag.get("control_projection_fraction", 0.0)), staleness)
        gen = torch.Generator(device=delta.device).manual_seed(550000 + int(seed) + step)
        random_source = torch.randn(source_signal.numel(), generator=gen, device=source_signal.device)
        random_source_loss = float((-(delta.to(random_source.device).reshape(-1) * random_source.reshape(-1))).mean().item())
        random_source_nds = float(random_source.abs().mean().item())
        random_source_norm = float(torch.linalg.vector_norm(random_source.float()).item())
        random_u = _u_score_source(random_source_loss, random_source_nds, tail_proxy, 0.0, staleness)
        signflip_u = _u_score_source(-source_loss_after, nds, tail_proxy, 0.0, staleness)
        stable_random_u = 0.5 * random_u
        useful = int(u_task > random_u + 1.0e-8)
        if grad_flat.numel():
            history.append((-grad_flat.detach().float().cpu()))
            useful_flags.append(useful)
            max_history = max(32, int(args.jvp_refresh_history_window))
            if len(history) > max_history:
                history.pop(0)
                useful_flags.pop(0)
        losses.append(float(loss.detach().item()))
        intervention_count += int_flag(diag.get("intervention_flag", 0))
        total_step_time += time.perf_counter() - step_start
        last_diag = diag
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            source_row = {
                "dataset": dataset,
                "seed": seed,
                "model_family": "MLP" if model_name.startswith("DGMLP") else "KAN",
                "model_name": model_name,
                "step": step,
                "source_id": f"{dataset}/seed{seed}/{model_name}/step{step}",
                "experiment_tag": args.experiment_tag,
                "controller_policy": controller_policy,
                "source_candidate_type": source_candidate_type,
                "split_consensus_cosine": diag.get("split_consensus_cosine", split_consensus_cosine),
                "source_norm": float(torch.linalg.vector_norm(source_signal.float()).item()),
                "controller_source_norm": float(torch.linalg.vector_norm(source_signal_for_controller.detach().float()).item()) if torch.is_tensor(source_signal_for_controller) else "",
                "linear_task_gain": linear_gain,
                "NDS": nds,
                "TailRisk_proxy": tail_proxy,
                "ControlProjection": diag.get("control_projection_fraction", 0.0),
                "Staleness": staleness,
                "U_task_usefulness": u_task,
                "U_task_usefulness_pre": diag.get("U_task_usefulness_pre", u_task_pre),
                "source_func_t": source_func,
                "source_loss_t": source_loss_after,
                "controller_source_loss_gain": diag.get("controller_source_loss_gain", controller_source_loss_gain),
                "source_loss_after_50_analysis_only": "",
                "source_loss_after_100_analysis_only": "",
                "source_loss_after_200_analysis_only": "",
                "source_func_after_50_analysis_only": "",
                "source_func_after_100_analysis_only": "",
                "source_func_after_200_analysis_only": "",
                "source_loss_flip_count": "",
                "source_age": diag.get("source_age", ""),
                "source_refresh_reason": "release_or_refresh" if int(diag.get("source_release_count", 0)) or int(diag.get("source_refresh_count", 0)) else "",
                "random_control_U": random_u,
                "random_control_U_pre": diag.get("random_control_U_pre", random_u_pre),
                "stable_random_U": stable_random_u,
                "signflip_U": signflip_u,
                "risk_score": diag.get("risk_score", 0.0),
                "lambda_t": diag.get("lambda_t", 0.0),
                "intervention_flag": diag.get("intervention_flag", 0),
                "task_useful_gate_pre": diag.get("task_useful_gate_pre", ""),
                "gate_reject_reason": diag.get("gate_reject_reason", ""),
                "predicted_gain_base": diag.get("predicted_gain_base", ""),
                "predicted_gain_guided": diag.get("predicted_gain_guided", ""),
                "virtual_loss_base": diag.get("virtual_loss_base", ""),
                "virtual_loss_guided": diag.get("virtual_loss_guided", ""),
                "benefit_p3_score": diag.get("benefit_p3_score", ""),
                "benefit_p3_manual_score": diag.get("benefit_p3_manual_score", ""),
                "benefit_p3_min_score": diag.get("benefit_p3_min_score", ""),
                "runtime_per_step_policy_integrated": diag.get("runtime_per_step_policy_integrated", int(controller_policy.startswith("benefit_p3"))),
                "benefit_policy_export_runtime_integrated": diag.get("benefit_policy_export_runtime_integrated", int(bool(benefit_policy_export) and controller_policy.startswith("benefit_p3"))),
                "benefit_policy_export_score": diag.get("benefit_policy_export_score", ""),
                "benefit_policy_export_threshold": diag.get("benefit_policy_export_threshold", ""),
                "benefit_policy_export_path": diag.get("benefit_policy_export_path", benefit_policy_export["path"] if benefit_policy_export else ""),
                "benefit_policy_export_horizon": diag.get("benefit_policy_export_horizon", benefit_policy_export.get("horizon", "") if benefit_policy_export else ""),
                "benefit_policy_export_target_mode": diag.get("benefit_policy_export_target_mode", benefit_policy_export.get("target_mode", "") if benefit_policy_export else ""),
                "benefit_policy_export_feature_set": diag.get("benefit_policy_export_feature_set", benefit_policy_export.get("feature_set", "") if benefit_policy_export else ""),
                "benefit_policy_feature_U_task_usefulness": diag.get("benefit_policy_feature_U_task_usefulness", ""),
                "benefit_policy_feature_source_age": diag.get("benefit_policy_feature_source_age", ""),
                "benefit_policy_feature_source_norm": diag.get("benefit_policy_feature_source_norm", ""),
                "benefit_policy_feature_risk_score": diag.get("benefit_policy_feature_risk_score", ""),
                "benefit_policy_feature_NDS": diag.get("benefit_policy_feature_NDS", ""),
                "benefit_policy_feature_virtual_train_loss_delta_current_step": diag.get("benefit_policy_feature_virtual_train_loss_delta_current_step", ""),
                "benefit_policy_feature_controller_to_base_update_ratio": diag.get("benefit_policy_feature_controller_to_base_update_ratio", ""),
                "jvp_reference": diag.get("jvp_reference", ""),
                "jvp_history_count": diag.get("jvp_history_count", ""),
                "jvp_history_best_source_loss": diag.get("jvp_history_best_source_loss", ""),
                "jvp_current_base_source_loss": diag.get("jvp_current_base_source_loss", ""),
                "jvp_history_gain_vs_current_base": diag.get("jvp_history_gain_vs_current_base", ""),
                "jacobian_reachable_projection_residual": diag.get("jacobian_reachable_projection_residual", ""),
                "controller_cap_scale": diag.get("controller_cap_scale", ""),
                "analysis_only_future_label_used_for_runtime_direction": 0,
            }
            source_rows.append(source_row)
            control_specs = [
                ("random", random_u, random_source_loss, random_source_norm, random_source_nds),
                ("stable_random", stable_random_u, 0.5 * random_source_loss, 0.5 * random_source_norm, 0.5 * random_source_nds),
                ("signflip", signflip_u, -source_loss_after, float(torch.linalg.vector_norm(source_signal.float()).item()), nds),
            ]
            for control_name, control_u, control_loss, control_norm, control_nds in control_specs:
                control_row = dict(source_row)
                control_row.update(
                    {
                        "source_id": f"{dataset}/seed{seed}/{model_name}/step{step}/neutral:{control_name}",
                        "source_candidate_type": f"task_neutral_control_{control_name}",
                        "neutral_control_source_type": control_name,
                        "source_norm": control_norm,
                        "controller_source_norm": "",
                        "linear_task_gain": 0.0,
                        "NDS": control_nds,
                        "ControlProjection": 0.0,
                        "U_task_usefulness": control_u,
                        "U_task_usefulness_pre": control_u,
                        "source_func_t": "",
                        "source_loss_t": control_loss,
                        "controller_source_loss_gain": "",
                        "random_control_U": random_u,
                        "random_control_U_pre": random_u,
                        "stable_random_U": stable_random_u,
                        "signflip_U": signflip_u,
                        "risk_score": 0.0,
                        "lambda_t": 0.0,
                        "intervention_flag": 0,
                        "task_useful_gate_pre": int(control_u > random_u + 1.0e-8),
                        "gate_reject_reason": "task_neutral_control_eval_only",
                        "predicted_gain_base": "",
                        "predicted_gain_guided": "",
                        "virtual_loss_base": "",
                        "virtual_loss_guided": "",
                        "official_task_neutral_control": 1,
                    }
                )
                neutral_control_rows.append(control_row)
    metrics = _eval(model, test_loader, device, output_dim)
    train_eval = _eval(model, train_loader, device, output_dim)
    basis_energy, readout_energy = grad_energy_by_channel(model)
    param_count = _count_parameters(model)
    flops = _model_flops(model_name, input_dim, args.hidden, output_dim)
    ctrl_rows = _controllability_diag_rows(model, selected, xb_last, history, useful_flags, dim=min(4, max(1, len(history) - 1)))
    manual_kernel_variant = model.manual_kernel_variant() if hasattr(model, "manual_kernel_variant") else ""
    uses_dense_basis_tensor_spec = getattr(getattr(model, "spec", None), "uses_dense_basis_tensor", "")
    final = {
        "dataset": dataset,
        "seed": seed,
        "model_name": model_name,
        "experiment_tag": args.experiment_tag,
        "implementation_mode": "kanbefair_bridge_no_dense_basis" if model_name.startswith("DGKAN") else "mlp_reference",
        "controller_policy": controller_policy,
        "source_candidate_type": source_candidate_type,
        "model_origin": "DGKAN_bridge",
        "model_family": "MLP" if model_name.startswith("DGMLP") else "KAN",
        "is_dgkan_strict_fc_purekan": int(model_name.startswith("DGKAN")),
        "uses_kanbefair_loader": 1,
        "uses_kanbefair_baseline_model": 0,
        "uses_bspline_official_path": 0,
        "uses_readout_diagnostic": 0,
        "basis_native_claim_allowed": int(model_name.startswith("DGKAN")),
        "uses_dense_basis_tensor_spec": uses_dense_basis_tensor_spec if model_name.startswith("DGKAN") else "",
        "manual_kernel_variant": manual_kernel_variant,
        "smoke_only": 1,
        "hidden": args.hidden,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "initial_data_batch_fingerprint": initial_batch_fingerprint,
        "steps": args.steps,
        "final_train_loss": train_eval["loss"],
        "final_test_loss_NLL": metrics["loss"],
        "final_test_accuracy": metrics["accuracy"],
        "AUC_loss_time": sum(losses) / max(1, len(losses)),
        "ECE": metrics["ECE"],
        "Brier": metrics["Brier"],
        "tail_loss_q95": metrics["tail_loss_q95"],
        "tail_loss_q99": metrics["tail_loss_q99"],
        "param_count": param_count,
        "flops": flops,
        "runtime_per_step_ms": 1000.0 * total_step_time / max(1, args.steps),
        "full_loop_step_ms": 1000.0 * total_step_time / max(1, args.steps),
        "controller_overhead_ratio": overhead_time / max(1.0e-12, total_step_time),
        "source_func_h_final": source_rows[-1]["source_func_t"] if source_rows else "",
        "source_loss_h_final": source_rows[-1]["source_loss_t"] if source_rows else "",
        "U_task_usefulness_mean": sum(float(r["U_task_usefulness"]) for r in source_rows) / max(1, len(source_rows)),
        "intervention_count": intervention_count,
        "gate_accept_count": gate_accept_count,
        "gate_reject_count": gate_reject_count,
        "gate_reject_reason_counts": ";".join(f"{k}:{v}" for k, v in sorted(gate_reject_reason_counts.items())),
        "mean_predicted_gain_base": sum(predicted_gain_base_values) / max(1, len(predicted_gain_base_values)) if predicted_gain_base_values else "",
        "mean_predicted_gain_guided": sum(predicted_gain_guided_values) / max(1, len(predicted_gain_guided_values)) if predicted_gain_guided_values else "",
        "mean_controller_source_loss_gain": sum(controller_source_gain_values) / max(1, len(controller_source_gain_values)) if controller_source_gain_values else "",
        "mean_split_consensus_cosine": sum(split_cosine_values) / max(1, len(split_cosine_values)) if split_cosine_values else "",
        "mean_virtual_loss_base": sum(virtual_loss_base_values) / max(1, len(virtual_loss_base_values)) if virtual_loss_base_values else "",
        "mean_virtual_loss_guided": sum(virtual_loss_guided_values) / max(1, len(virtual_loss_guided_values)) if virtual_loss_guided_values else "",
        "mean_jvp_refresh_source_loss_gain": sum(jvp_refresh_source_gain_values) / max(1, len(jvp_refresh_source_gain_values)) if jvp_refresh_source_gain_values else "",
        "mean_jvp_refresh_update_ratio": sum(jvp_refresh_update_ratio_values) / max(1, len(jvp_refresh_update_ratio_values)) if jvp_refresh_update_ratio_values else "",
        "mean_benefit_p3_score": sum(benefit_p3_score_values) / max(1, len(benefit_p3_score_values)) if benefit_p3_score_values else "",
        "mean_benefit_p3_accept_score": sum(benefit_p3_accept_score_values) / max(1, len(benefit_p3_accept_score_values)) if benefit_p3_accept_score_values else "",
        "mean_benefit_policy_export_score": sum(benefit_policy_export_score_values) / max(1, len(benefit_policy_export_score_values)) if benefit_policy_export_score_values else "",
        "mean_benefit_policy_export_accept_score": (
            sum(benefit_policy_export_accept_score_values) / max(1, len(benefit_policy_export_accept_score_values)) if benefit_policy_export_accept_score_values else ""
        ),
        "runtime_per_step_policy_integrated": int(controller_policy.startswith("benefit_p3")),
        "benefit_policy_export_runtime_integrated": int(bool(benefit_policy_export) and controller_policy.startswith("benefit_p3")),
        "benefit_policy_export_json": benefit_policy_export["path"] if benefit_policy_export else "",
        "benefit_policy_export_threshold": float(benefit_policy_export["accept_threshold"]) if benefit_policy_export else "",
        "benefit_policy_export_horizon": benefit_policy_export.get("horizon", "") if benefit_policy_export else "",
        "benefit_policy_export_target_mode": benefit_policy_export.get("target_mode", "") if benefit_policy_export else "",
        "benefit_policy_export_feature_set": benefit_policy_export.get("feature_set", "") if benefit_policy_export else "",
        "benefit_policy_export_source_C_exploration_pass": benefit_policy_export.get("source_C_exploration_pass", "") if benefit_policy_export else "",
        "benefit_policy_export_source_C_official_pass": benefit_policy_export.get("source_C_official_pass", "") if benefit_policy_export else "",
        "final_jvp_history_count": last_diag.get("jvp_history_count", ""),
        "final_jvp_gate_reject_reason": last_diag.get("gate_reject_reason", "") if controller_policy == "jvp_useful_history_refresh" else "",
        "source_release_count": last_diag.get("source_release_count", 0),
        "source_refresh_count": last_diag.get("source_refresh_count", 0),
        "basis_channel_energy_fraction": basis_energy,
        "readout_channel_energy_fraction": readout_energy,
        "uses_dense_output_jacobian_official": 0,
        "analytic_vs_autograd_rel_error_audit": "",
    }
    return final, source_rows, neutral_control_rows, ctrl_rows, {"dataset": dataset, "seed": seed, "model_name": model_name, "param_count": param_count, "flops": flops, "input_size": input_dim, "output_size": output_dim}


def _add_deltas(rows: list[dict[str, Any]]) -> None:
    by_key = {(r["dataset"], str(r["seed"]), r["model_name"]): r for r in rows}
    mlpfu_candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row["model_name"].startswith("DGMLP_FU"):
            key = (row["dataset"], str(row["seed"]))
            prev = mlpfu_candidates.get(key)
            if prev is None or finite_float(row.get("final_test_loss_NLL")) < finite_float(prev.get("final_test_loss_NLL")):
                mlpfu_candidates[key] = row
    for row in rows:
        dataset, seed, model = row["dataset"], str(row["seed"]), row["model_name"]
        base_name = _base_model_for_delta(model)
        if model.startswith("DGMLP_FU"):
            base = by_key.get((dataset, seed, base_name))
            if base:
                row["NLL_delta_vs_MLP_AdamW"] = finite_float(row["final_test_loss_NLL"]) - finite_float(base["final_test_loss_NLL"])
                row["accuracy_delta_vs_MLP_AdamW"] = finite_float(row["final_test_accuracy"]) - finite_float(base["final_test_accuracy"])
                row["AUC_loss_time_delta_vs_MLP_AdamW"] = finite_float(row["AUC_loss_time"]) - finite_float(base["AUC_loss_time"])
        if model.startswith("DGKAN_DFOU_FU") or model.startswith("DGKAN_DCHE_FU"):
            base = by_key.get((dataset, seed, base_name))
            mlpfu = mlpfu_candidates.get((dataset, seed)) or by_key.get((dataset, seed, "DGMLP_FU"))
            if base:
                row["NLL_delta_vs_KAN_AdamW"] = finite_float(row["final_test_loss_NLL"]) - finite_float(base["final_test_loss_NLL"])
                row["accuracy_delta_vs_KAN_AdamW"] = finite_float(row["final_test_accuracy"]) - finite_float(base["final_test_accuracy"])
                row["AUC_loss_time_delta_vs_KAN_AdamW"] = finite_float(row["AUC_loss_time"]) - finite_float(base["AUC_loss_time"])
            if mlpfu:
                row["KAN_vs_MLPFU_NLL_delta"] = finite_float(row["final_test_loss_NLL"]) - finite_float(mlpfu["final_test_loss_NLL"])
                row["KAN_vs_MLPFU_accuracy_delta"] = finite_float(row["final_test_accuracy"]) - finite_float(mlpfu["final_test_accuracy"])


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    args._benefit_policy_export = _load_benefit_policy_export(args.benefit_policy_export_json, float(args.benefit_policy_export_threshold))
    device = _device(args.device)
    datasets = [canonical_dataset_name(x) for x in _split(args.datasets)]
    models = _split(args.models)
    seeds = _split(args.seeds, int)
    final_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    neutral_control_rows: list[dict[str, Any]] = []
    ctrl_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            for model in models:
                try:
                    row, src, neutral_src, ctrl, param = _train_one(dataset, seed, model, args, device)
                    final_rows.append(row)
                    source_rows.extend(src)
                    neutral_control_rows.extend(neutral_src)
                    for ctrl_row in ctrl:
                        ctrl_rows.append({"dataset": dataset, "seed": seed, "model_name": model, **ctrl_row})
                    param_rows.append(param)
                except Exception as exc:
                    failure_rows.append({"stage": "DGKAN_bridge", "dataset": dataset, "seed": seed, "model_name": model, "failure_type": type(exc).__name__, "error": str(exc)})
    _add_deltas(final_rows)
    write_rows(_out_path("v22_17_kanbefair_dgkan_bridge_matrix.csv", args.output_suffix), final_rows)
    write_rows(_out_path("v22_17_kanbefair_external_task_matrix.csv", args.output_suffix), final_rows)
    write_rows(_out_path("v22_17_kanbefair_param_flops_matching_matrix.csv", args.output_suffix), param_rows)
    write_rows(_out_path("v22_17_task_useful_source_matrix.csv", args.output_suffix), source_rows)
    write_rows(_out_path("v22_17_source_usefulness_timeseries.csv", args.output_suffix), source_rows)
    write_rows(_out_path("v22_17_source_controls_matrix.csv", args.output_suffix), source_rows)
    write_rows(_out_path("v22_17_task_neutral_control_source_matrix.csv", args.output_suffix), neutral_control_rows)
    write_rows(_out_path("v22_17_controllability_manifold_matrix.csv", args.output_suffix), ctrl_rows)
    write_rows(_out_path("v22_17_jacobian_reachability_matrix.csv", args.output_suffix), ctrl_rows)
    write_rows(_out_path("v22_17_raw_vs_reachable_residual.csv", args.output_suffix), ctrl_rows)
    write_rows(_out_path("v22_17_manifold_controls_matrix.csv", args.output_suffix), ctrl_rows)
    write_rows(_out_path("v22_17_mlp_fu_task_matrix.csv", args.output_suffix), [r for r in final_rows if r.get("model_family") == "MLP"])
    write_rows(_out_path("v22_17_kan_basis_controllability_matrix.csv", args.output_suffix), [r for r in final_rows if r.get("model_family") == "KAN"])
    write_rows(_out_path("v22_17_real_efficiency_controller_loop.csv", args.output_suffix), final_rows)
    write_rows(_out_path("v22_17_kan_vs_mlpfu_interaction.csv", args.output_suffix), [r for r in final_rows if r.get("model_name") == "DGKAN_DFOU_FU"])
    if failure_rows:
        write_rows(_out_path("v22_17_kanbefair_bridge_failures.csv", args.output_suffix), failure_rows)
    cmd_parts = [
        sys.executable,
        "experiments/run_v22_17_kanbefair_dgkan_eval.py",
        "--kanbefair-root",
        args.kanbefair_root,
        "--datasets",
        args.datasets,
        "--models",
        args.models,
        "--seeds",
        args.seeds,
        "--device",
        args.device,
        "--train-size",
        str(args.train_size),
        "--test-size",
        str(args.test_size),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--steps",
        str(args.steps),
        "--hidden",
        str(args.hidden),
        "--lr",
        str(args.lr),
        "--log-interval",
        str(args.log_interval),
        "--experiment-tag",
        args.experiment_tag,
        "--gate-margin",
        str(args.gate_margin),
        "--actionable-gain-tolerance",
        str(args.actionable_gain_tolerance),
        "--gated-ratio-cap",
        str(args.gated_ratio_cap),
        "--actionable-ratio-cap",
        str(args.actionable_ratio_cap),
        "--virtual-gate-scale",
        str(args.virtual_gate_scale),
        "--virtual-loss-tolerance",
        str(args.virtual_loss_tolerance),
        "--jvp-refresh-dim",
        str(args.jvp_refresh_dim),
        "--jvp-refresh-min-history",
        str(args.jvp_refresh_min_history),
        "--jvp-refresh-history-window",
        str(args.jvp_refresh_history_window),
        "--jvp-refresh-scale",
        str(args.jvp_refresh_scale),
        "--jvp-refresh-ratio-cap",
        str(args.jvp_refresh_ratio_cap),
        "--jvp-refresh-ridge",
        str(args.jvp_refresh_ridge),
        "--benefit-p3-ratio-cap",
        str(args.benefit_p3_ratio_cap),
        "--benefit-p3-strong-ratio-cap",
        str(args.benefit_p3_strong_ratio_cap),
        "--benefit-p3-min-score",
        str(args.benefit_p3_min_score),
        "--benefit-p3-strong-min-score",
        str(args.benefit_p3_strong_min_score),
        "--benefit-p3-jvp-strong-scale",
        str(args.benefit_p3_jvp_strong_scale),
        "--benefit-p3-jvp-strong-ratio-cap",
        str(args.benefit_p3_jvp_strong_ratio_cap),
        "--benefit-p3-jvp-interval",
        str(args.benefit_p3_jvp_interval),
    ]
    if args.benefit_policy_export_json:
        cmd_parts.extend(["--benefit-policy-export-json", args.benefit_policy_export_json])
        if math.isfinite(float(args.benefit_policy_export_threshold)):
            cmd_parts.extend(["--benefit-policy-export-threshold", str(args.benefit_policy_export_threshold)])
    if args.output_suffix:
        cmd_parts.extend(["--output-suffix", args.output_suffix])
    bridge_file = _out_path("v22_17_kanbefair_dgkan_bridge_matrix.csv", args.output_suffix)
    source_file = _out_path("v22_17_task_useful_source_matrix.csv", args.output_suffix)
    ctrl_file = _out_path("v22_17_controllability_manifold_matrix.csv", args.output_suffix)
    append_exec(
        " ".join(shlex.quote(part) for part in cmd_parts),
        task_id="E-F-G-bridge-smoke",
        status="pass" if final_rows and not failure_rows else "partial",
        gpu=args.device.replace("cuda:", "") if args.device.startswith("cuda:") else "",
        exit_code=0,
        files=f"{bridge_file.relative_to(ROOT)}, {source_file.relative_to(ROOT)}, {ctrl_file.relative_to(ROOT)}",
        note="fixed-step smoke with unified metrics; not an official full-proof run",
    )


if __name__ == "__main__":
    main()
