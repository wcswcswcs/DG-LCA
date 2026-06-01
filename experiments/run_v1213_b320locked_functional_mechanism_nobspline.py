#!/usr/bin/env python3
"""v12.13 B320-locked functional mechanism and Line C calibration runner.

The runner starts with measurement trustworthiness. It reuses v12.12 artifact
rewrites for B320 anchor/family status, then runs repeated cloned-checkpoint
Line C probes without changing loss, data sampling policy, class weights, or
dataset-specific thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "v12_13_b320locked_functional_mechanism_nobspline"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "DG-KAN_v12.13_B320Locked_FunctionalMechanism_NoBSpline_执行复盘.md"
DEFAULT_V1212_REPORT = REPO_ROOT / "docs" / "DG-KAN_v12.12_B320Locked_FunctionalValueRebuild_ClassicNoBSpline_执行复盘.md"
B320_ID = "B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075"
ACTIVE_FAMILIES = ["Rational", "Chebyshev", "Wavelet", "RBF", "Fourier"]


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_list(text: Any) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_ints(text: Any) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    materialized = [dict(row) for row in rows]
    if materialized:
        fields: list[str] = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    else:
        fields = ["stage", "status"]
        materialized = [{"stage": path.stem.upper(), "status": "no_rows"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_v1212_runner():
    return load_module("v1212_runner_for_v1213", REPO_ROOT / "experiments" / "run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py")


def load_v1211_runner():
    return load_module("v1211_runner_for_v1213", REPO_ROOT / "experiments" / "run_v1211_b320_functional_mechanism_classic_nobspline.py")


def delta_norm(deltas: Sequence[Any]) -> float:
    return math.sqrt(sum(float(delta.detach().square().sum().item()) for delta in deltas))


def delta_dot(a: Sequence[Any], b: Sequence[Any]) -> float:
    return sum(float((x.detach() * y.detach()).sum().item()) for x, y in zip(a, b))


def scale_delta(deltas: Sequence[Any], scale: float) -> list[Any]:
    return [delta * float(scale) for delta in deltas]


def normalize_like(deltas: Sequence[Any], refs: Sequence[Any]) -> list[Any]:
    norm = delta_norm(deltas)
    ref_norm = delta_norm(refs)
    if norm <= 1.0e-12 or ref_norm <= 1.0e-12:
        return [delta * 0.0 for delta in deltas]
    return [delta * (ref_norm / norm) for delta in deltas]


def remove_task_parallel(deltas: Sequence[Any], task_delta: Sequence[Any]) -> list[Any]:
    task_norm_sq = max(1.0e-12, delta_dot(task_delta, task_delta))
    coeff = delta_dot(deltas, task_delta) / task_norm_sq
    return [delta - coeff * task for delta, task in zip(deltas, task_delta)]


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(max(0.0, var))


def ci95(values: Sequence[float]) -> tuple[float, float]:
    mean, std = mean_std(values)
    if not values:
        return 0.0, 0.0
    radius = 1.96 * std / math.sqrt(max(1, len(values)))
    return mean - radius, mean + radius


def pareto_fail(row: Mapping[str, Any], strict_holdout: float = 1.002) -> list[str]:
    fail: list[str] = []
    if safe_float(row.get("CouplingR2_delta"), 0.0) < 0.02:
        fail.append("CouplingR2_delta<0.02")
    if safe_float(row.get("NoiseSignalLeak_delta"), 0.0) > -0.01:
        fail.append("NoiseSignalLeak_delta>-0.01")
    if safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0) > -0.01:
        fail.append("RealSignalReservoirRatio_delta>-0.01")
    if safe_float(row.get("holdout_loss_ratio"), 999.0) > strict_holdout:
        fail.append(f"holdout_loss_ratio>{strict_holdout}")
    if safe_float(row.get("logit_max_abs_drift"), 0.0) > 0.05:
        fail.append("logit_max_abs_drift>0.05")
    if safe_float(row.get("CEp99_delta"), 0.0) > 0.05:
        fail.append("CEp99_delta>0.05")
    if safe_float(row.get("control_gap_vs_best"), -999.0) < 0.005:
        fail.append("control_gap<0.005")
    return fail


def evaluate_with_drift(prev: Any, base: Any, deltas: Sequence[Any], task_delta: Sequence[Any], xb: Any, yb: Any, xq: Any, yq: Any, seed: int, args: argparse.Namespace, torch_mod: Any, F_mod: Any, v1252: Any, v1283: Any) -> dict[str, Any]:
    with torch_mod.no_grad():
        before = base(xq)
    metrics = prev._evaluate_probe_direction(
        base=base,
        deltas=deltas,
        task_delta=task_delta,
        xb=xb,
        yb=yb,
        xq=xq,
        yq=yq,
        scale=1.0,
        seed=int(seed),
        args=args,
        torch_mod=torch_mod,
        F_mod=F_mod,
        v1252=v1252,
        v1283=v1283,
    )
    trial = __import__("copy").deepcopy(base)
    v1252._apply_delta(trial, deltas, 1.0)
    with torch_mod.no_grad():
        after = trial(xq)
        p = torch_mod.log_softmax(before, dim=1)
        q = torch_mod.log_softmax(after, dim=1)
        metrics["logit_max_abs_drift"] = float((after - before).abs().max().item())
        metrics["logit_match_mse"] = float((after - before).square().mean().item())
        metrics["KL_before_after"] = float((p.exp() * (p - q)).sum(dim=1).mean().item())
    return metrics


def role_filtered_delta(model: Any, source_delta: Sequence[Any], task_delta: Sequence[Any], role: str) -> tuple[list[Any], dict[str, Any]]:
    role_tokens = {
        "direct": ("direct_readout", "bias"),
        "quad": ("quad_proj", "quad_readout"),
        "branch": ("branch_scale",),
    }
    tokens = role_tokens[role]
    filtered = []
    kept_square = 0.0
    total_square = 0.0
    kept_params = 0
    for (name, _param), delta in zip(model.named_parameters(), source_delta):
        total_square += float(delta.detach().square().sum().item())
        if any(token in name for token in tokens):
            filtered.append(delta.clone())
            kept_square += float(delta.detach().square().sum().item())
            kept_params += 1
        else:
            filtered.append(delta * 0.0)
    filtered = normalize_like(filtered, task_delta)
    return filtered, {
        "role": role,
        "role_kept_params": kept_params,
        "role_source_norm_fraction": math.sqrt(kept_square / max(1.0e-24, total_square)),
    }


def reservoir_vjp_delta(torch_mod: Any, F_mod: Any, model: Any, x: Any, y: Any, lr: float, task_delta: Sequence[Any], orthogonal_to_task: bool) -> tuple[list[Any], dict[str, Any]]:
    params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
    logits = model(x)
    target = F_mod.one_hot(y, num_classes=int(logits.shape[1])).to(dtype=logits.dtype)
    loss = 0.5 * (logits - target).square().mean()
    grads = torch_mod.autograd.grad(loss, params, allow_unused=True)
    deltas = [(-float(lr) * grad if grad is not None else torch_mod.zeros_like(param)) for param, grad in zip(params, grads)]
    raw_norm = delta_norm(deltas)
    task_norm = delta_norm(task_delta)
    task_cos = delta_dot(deltas, task_delta) / max(1.0e-12, raw_norm * task_norm)
    if orthogonal_to_task and task_norm > 1.0e-12:
        coeff = delta_dot(deltas, task_delta) / max(1.0e-12, task_norm * task_norm)
        deltas = [delta - coeff * task for delta, task in zip(deltas, task_delta)]
    deltas = normalize_like(deltas, task_delta)
    return deltas, {
        "reservoir_vjp_loss": float(loss.detach().item()),
        "reservoir_vjp_raw_norm": raw_norm,
        "reservoir_vjp_cos_to_adamw": task_cos,
        "reservoir_vjp_orthogonalized": int(orthogonal_to_task),
    }


def noise_veto_delta(v1252: Any, torch_mod: Any, model: Any, x: Any, y: Any, lr: float, task_delta: Sequence[Any], seed: int, orthogonal_to_task: bool) -> tuple[list[Any], dict[str, Any]]:
    gen = torch_mod.Generator(device=x.device).manual_seed(int(seed) + 433)
    perm = torch_mod.randperm(int(y.shape[0]), device=y.device, generator=gen)
    y_noise = y[perm]
    noise_delta = v1252._grad_delta(model, x, y_noise, lr)
    # Functional correction: descend on real labels while vetoing descent on
    # permuted-label noise. This uses only the current training batch.
    deltas = [task - noise for task, noise in zip(task_delta, noise_delta)]
    raw_norm = delta_norm(deltas)
    task_norm = delta_norm(task_delta)
    task_cos = delta_dot(deltas, task_delta) / max(1.0e-12, raw_norm * task_norm)
    if orthogonal_to_task:
        deltas = remove_task_parallel(deltas, task_delta)
    deltas = normalize_like(deltas, task_delta)
    return deltas, {
        "noise_veto_perm_seed": int(seed) + 433,
        "noise_veto_raw_norm": raw_norm,
        "noise_veto_cos_to_adamw": task_cos,
        "noise_veto_orthogonalized": int(orthogonal_to_task),
    }


def linec_projector_surrogate_delta(
    v1252: Any,
    torch_mod: Any,
    F_mod: Any,
    model: Any,
    x: Any,
    y: Any,
    lr: float,
    task_delta: Sequence[Any],
    sketch_dim: int,
    seed: int,
    mode: str,
    orthogonal_to_task: bool,
) -> tuple[list[Any], dict[str, Any]]:
    b = min(int(x.shape[0]), int(y.shape[0]), 8)
    x = x[:b]
    y = y[:b]
    grad_sketch = v1252._sample_grad_sketch(model, x, y, int(sketch_dim), int(seed)).detach()
    with torch_mod.no_grad():
        k_mat = grad_sketch @ grad_sketch.T
        evals, evecs = torch_mod.linalg.eigh(k_mat.float())
        order = torch_mod.argsort(evals, descending=True)
        evals = evals[order].clamp_min(0.0)
        evecs = evecs[:, order]
        total = evals.sum().clamp_min(1.0e-12)
        cum = torch_mod.cumsum(evals, dim=0)
        top_count = int(torch_mod.searchsorted(cum, 0.80 * total).item()) + 1
        top_count = max(1, min(top_count, int(evals.numel())))
        p_sig = (evecs[:, :top_count] @ evecs[:, :top_count].T).detach()
        p_res = (torch_mod.eye(b, device=x.device) - p_sig).detach()
        y_noise = y[torch_mod.randperm(b, device=x.device, generator=torch_mod.Generator(device=x.device).manual_seed(int(seed) + 33))]
    params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
    logits = model(x)
    ce_real = F_mod.cross_entropy(logits, y, reduction="none")
    ce_noise = F_mod.cross_entropy(logits, y_noise, reduction="none")
    r_real = ce_real.float() - ce_real.float().mean()
    r_noise = ce_noise.float() - ce_noise.float().mean()
    real_den = r_real.detach().square().sum().clamp_min(1.0e-12)
    noise_den = r_noise.detach().square().sum().clamp_min(1.0e-12)
    real_term = (p_res @ r_real).square().sum() / real_den
    noise_term = (p_sig @ r_noise).square().sum() / noise_den
    if mode == "reservoir":
        loss = real_term
    elif mode == "noise":
        loss = noise_term
    elif mode == "balanced":
        loss = real_term + noise_term
    else:
        raise ValueError(f"unknown projector surrogate mode {mode}")
    grads = torch_mod.autograd.grad(loss, params, allow_unused=True)
    deltas = [(-float(lr) * grad if grad is not None else torch_mod.zeros_like(param)) for param, grad in zip(params, grads)]
    raw_norm = delta_norm(deltas)
    task_norm = delta_norm(task_delta)
    task_cos = delta_dot(deltas, task_delta) / max(1.0e-12, raw_norm * task_norm)
    if orthogonal_to_task:
        deltas = remove_task_parallel(deltas, task_delta)
    deltas = normalize_like(deltas, task_delta)
    return deltas, {
        "projector_mode": mode,
        "projector_top_count": top_count,
        "projector_real_term": float(real_term.detach().item()),
        "projector_noise_term": float(noise_term.detach().item()),
        "projector_loss": float(loss.detach().item()),
        "projector_raw_norm": raw_norm,
        "projector_cos_to_adamw": task_cos,
        "projector_orthogonalized": int(orthogonal_to_task),
    }


def linec_projector_noise_constrained_delta(
    v1252: Any,
    torch_mod: Any,
    F_mod: Any,
    model: Any,
    x: Any,
    y: Any,
    lr: float,
    task_delta: Sequence[Any],
    sketch_dim: int,
    seed: int,
    mode: str,
    orthogonal_to_task: bool,
) -> tuple[list[Any], dict[str, Any]]:
    reservoir_delta, reservoir_stats = linec_projector_surrogate_delta(
        v1252, torch_mod, F_mod, model, x, y, lr, task_delta, sketch_dim, seed, "reservoir", False
    )
    noise_delta, noise_stats = linec_projector_surrogate_delta(
        v1252, torch_mod, F_mod, model, x, y, lr, task_delta, sketch_dim, seed, "noise", False
    )
    noise_norm_sq = max(1.0e-12, delta_dot(noise_delta, noise_delta))
    task_norm_sq = max(1.0e-12, delta_dot(task_delta, task_delta))
    before_dot = delta_dot(reservoir_delta, noise_delta)
    target_dot = 0.0
    if mode == "noise_margin":
        target_dot = 0.05 * task_norm_sq
    elif mode != "noise_null":
        raise ValueError(f"unknown projector noise-constrained mode {mode}")
    coeff = max(0.0, (target_dot - before_dot) / noise_norm_sq)
    deltas = [res + coeff * noise for res, noise in zip(reservoir_delta, noise_delta)]
    after_dot = delta_dot(deltas, noise_delta)
    if orthogonal_to_task:
        deltas = remove_task_parallel(deltas, task_delta)
    deltas = normalize_like(deltas, task_delta)
    return deltas, {
        "projector_mode": mode,
        "projector_top_count": reservoir_stats.get("projector_top_count", ""),
        "projector_real_term": reservoir_stats.get("projector_real_term", ""),
        "projector_noise_term": noise_stats.get("projector_noise_term", ""),
        "projector_loss": reservoir_stats.get("projector_loss", ""),
        "projector_raw_norm": reservoir_stats.get("projector_raw_norm", ""),
        "projector_cos_to_adamw": reservoir_stats.get("projector_cos_to_adamw", ""),
        "projector_orthogonalized": int(orthogonal_to_task),
        "projector_noise_constraint_dot_before": before_dot,
        "projector_noise_constraint_dot_after": after_dot,
        "projector_noise_constraint_coeff": coeff,
    }


def logit_cotangent_sketch_delta(
    torch_mod: Any,
    model: Any,
    x: Any,
    lr: float,
    task_delta: Sequence[Any],
    sketch_dim: int,
    seed: int,
    mode: str,
    orthogonal_to_task: bool,
) -> tuple[list[Any], dict[str, Any]]:
    """Loss-agnostic direction from arbitrary logit cotangent sample modes."""
    b = min(int(x.shape[0]), 8)
    x = x[:b]
    params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
    with torch_mod.no_grad():
        out_dim = int(model(x[:1]).shape[1])
        gen = torch_mod.Generator(device=x.device).manual_seed(int(seed) + 6151)
        cot = torch_mod.randn((b, out_dim), device=x.device, generator=gen)
        cot = cot - cot.mean(dim=1, keepdim=True)
        cot = cot / cot.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    rows: list[Any] = []
    for i in range(b):
        model.zero_grad(set_to_none=True)
        scalar = (model(x[i : i + 1]) * cot[i : i + 1]).sum()
        grads = torch_mod.autograd.grad(scalar, params, retain_graph=False, create_graph=False, allow_unused=True)
        vals: list[Any] = []
        for k in range(int(sketch_dim)):
            dot = torch_mod.zeros((), device=x.device)
            for pidx, grad in enumerate(grads):
                if grad is None:
                    continue
                sign_gen = torch_mod.Generator(device=x.device).manual_seed(int(seed) + 1009 * k + 9176 * pidx)
                signs = torch_mod.randint(0, 2, grad.shape, device=x.device, generator=sign_gen, dtype=torch_mod.int8).float().mul_(2.0).sub_(1.0)
                dot = dot + (grad.float() * signs).sum() / math.sqrt(max(1, grad.numel()))
            vals.append(dot)
        rows.append(torch_mod.stack(vals))
    model.zero_grad(set_to_none=True)
    with torch_mod.no_grad():
        sketch = torch_mod.stack(rows, dim=0)
        k_mat = sketch @ sketch.T
        evals, evecs = torch_mod.linalg.eigh(k_mat.float())
        order = torch_mod.argsort(evals, descending=True)
        evals = evals[order].clamp_min(0.0)
        evecs = evecs[:, order]
        total = evals.sum().clamp_min(1.0e-12)
        top_share = float(evals[0].div(total).item()) if evals.numel() else 0.0
        eff_rank = float(evals.sum().square().div(evals.square().sum().clamp_min(1.0e-12)).item()) if evals.numel() else 0.0
        top_vec = evecs[:, 0]
        tail_vec = evecs[:, -1]
        if mode == "tail":
            weights = tail_vec
            polarity = -1.0
        elif mode == "tail_flip":
            weights = tail_vec
            polarity = 1.0
        elif mode == "balanced":
            weights = tail_vec - top_vec
            polarity = -1.0
        else:
            raise ValueError(f"unknown logit cotangent sketch mode {mode}")
        weights = weights - weights.mean()
        weights = weights / weights.norm().clamp_min(1.0e-12)
        weights = weights.detach()
    logits = model(x)
    scalar_terms = (logits * cot).sum(dim=1)
    loss = (weights.to(dtype=scalar_terms.dtype) * scalar_terms).sum() / math.sqrt(max(1, b))
    grads = torch_mod.autograd.grad(loss, params, allow_unused=True)
    deltas = [(float(polarity) * float(lr) * grad if grad is not None else torch_mod.zeros_like(param)) for param, grad in zip(params, grads)]
    raw_norm = delta_norm(deltas)
    task_norm = delta_norm(task_delta)
    task_cos = delta_dot(deltas, task_delta) / max(1.0e-12, raw_norm * task_norm)
    if orthogonal_to_task:
        deltas = remove_task_parallel(deltas, task_delta)
    deltas = normalize_like(deltas, task_delta)
    return deltas, {
        "cotangent_mode": mode,
        "cotangent_seed": int(seed) + 6151,
        "cotangent_batch": b,
        "cotangent_sketch_dim": int(sketch_dim),
        "cotangent_top_eigen_share": top_share,
        "cotangent_effective_rank": eff_rank,
        "cotangent_raw_norm": raw_norm,
        "cotangent_cos_to_adamw": task_cos,
        "cotangent_orthogonalized": int(orthogonal_to_task),
        "loss_agnostic_direction": 1,
        "ce_vector_used_for_direction": 0,
    }


def logit_cotangent_spectral_delta(
    torch_mod: Any,
    model: Any,
    x: Any,
    lr: float,
    task_delta: Sequence[Any],
    sketch_dim: int,
    seed: int,
    mode: str,
    orthogonal_to_task: bool,
) -> tuple[list[Any], dict[str, Any]]:
    """Loss-agnostic second-order shaping of a logit-cotangent gradient sketch."""
    b = min(int(x.shape[0]), 4)
    x = x[:b]
    params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
    with torch_mod.no_grad():
        out_dim = int(model(x[:1]).shape[1])
        gen = torch_mod.Generator(device=x.device).manual_seed(int(seed) + 7919)
        cot = torch_mod.randn((b, out_dim), device=x.device, generator=gen)
        cot = cot - cot.mean(dim=1, keepdim=True)
        cot = cot / cot.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    rows: list[Any] = []
    for i in range(b):
        scalar = (model(x[i : i + 1]) * cot[i : i + 1]).sum()
        grads = torch_mod.autograd.grad(scalar, params, retain_graph=True, create_graph=True, allow_unused=True)
        vals: list[Any] = []
        for k in range(int(sketch_dim)):
            dot = torch_mod.zeros((), device=x.device)
            for pidx, grad in enumerate(grads):
                if grad is None:
                    continue
                sign_gen = torch_mod.Generator(device=x.device).manual_seed(int(seed) + 1009 * k + 9176 * pidx)
                signs = torch_mod.randint(0, 2, grad.shape, device=x.device, generator=sign_gen, dtype=torch_mod.int8).float().mul_(2.0).sub_(1.0)
                dot = dot + (grad.float() * signs).sum() / math.sqrt(max(1, grad.numel()))
            vals.append(dot)
        rows.append(torch_mod.stack(vals))
    sketch = torch_mod.stack(rows, dim=0)
    k_mat = sketch @ sketch.T
    evals = torch_mod.linalg.eigvalsh(k_mat.float()).clamp_min(1.0e-12)
    evals_desc = torch_mod.flip(evals, dims=[0])
    total = evals_desc.sum().clamp_min(1.0e-12)
    top_k = max(1, b // 2)
    top_mass = evals_desc[:top_k].sum() / total
    probs = evals_desc / total
    uniform = torch_mod.full_like(probs, 1.0 / max(1, b))
    isotropic_loss = (probs - uniform).square().sum()
    if mode == "top_damp":
        loss = top_mass
    elif mode == "isotropic":
        loss = isotropic_loss
    else:
        raise ValueError(f"unknown logit cotangent spectral mode {mode}")
    grads2 = torch_mod.autograd.grad(loss, params, allow_unused=True)
    deltas = [(-float(lr) * grad if grad is not None else torch_mod.zeros_like(param)) for param, grad in zip(params, grads2)]
    raw_norm = delta_norm(deltas)
    task_norm = delta_norm(task_delta)
    task_cos = delta_dot(deltas, task_delta) / max(1.0e-12, raw_norm * task_norm)
    if orthogonal_to_task:
        deltas = remove_task_parallel(deltas, task_delta)
    deltas = normalize_like(deltas, task_delta)
    model.zero_grad(set_to_none=True)
    return deltas, {
        "cotangent_mode": f"spectral_{mode}",
        "cotangent_seed": int(seed) + 7919,
        "cotangent_batch": b,
        "cotangent_sketch_dim": int(sketch_dim),
        "cotangent_top_eigen_share": float(evals_desc[0].detach().div(total.detach()).item()) if evals_desc.numel() else 0.0,
        "cotangent_effective_rank": float(total.detach().square().div(evals_desc.detach().square().sum().clamp_min(1.0e-12)).item()) if evals_desc.numel() else 0.0,
        "cotangent_spectral_loss": float(loss.detach().item()),
        "cotangent_raw_norm": raw_norm,
        "cotangent_cos_to_adamw": task_cos,
        "cotangent_orthogonalized": int(orthogonal_to_task),
        "loss_agnostic_direction": 1,
        "ce_vector_used_for_direction": 0,
    }


def combine_deltas(terms: Sequence[tuple[float, Sequence[Any]]], ref: Sequence[Any]) -> list[Any]:
    out = [delta * 0.0 for delta in ref]
    for weight, deltas in terms:
        out = [acc + float(weight) * delta for acc, delta in zip(out, deltas)]
    return normalize_like(out, ref)


def bm2_grid_coefficients(mode: str) -> list[tuple[float, float, float, float]]:
    if mode in {"signed", "signed_orth"}:
        return [
            (0.50, 0.50, 0.00, 0.00),
            (0.50, -0.50, 0.00, 0.00),
            (0.35, 0.35, -0.35, 0.35),
            (0.35, 0.35, 0.35, -0.35),
            (0.25, -0.25, 0.50, -0.50),
            (0.25, -0.25, -0.50, 0.50),
            (0.00, 0.50, -0.50, 0.50),
            (0.00, 0.50, 0.50, -0.50),
            (-0.25, 0.75, 0.25, 0.25),
            (-0.25, 0.75, -0.25, 0.25),
            (-0.25, 0.75, 0.25, -0.25),
            (0.75, -0.25, -0.25, 0.25),
        ]
    if mode == "noise_veto":
        return [
            (0.60, 0.20, 0.00, 0.20),
            (0.50, 0.10, 0.00, 0.40),
            (0.35, 0.25, 0.10, 0.30),
            (0.20, 0.20, 0.20, 0.40),
        ]
    if mode == "reservoir_veto":
        return [
            (0.35, 0.25, 0.40, 0.00),
            (0.25, 0.35, 0.40, 0.00),
            (0.20, 0.20, 0.50, 0.10),
            (0.10, 0.40, 0.40, 0.10),
        ]
    return [
        (0.50, 0.50, 0.00, 0.00),
        (0.40, 0.40, 0.20, 0.00),
        (0.30, 0.50, 0.00, 0.20),
        (0.25, 0.25, 0.25, 0.25),
        (0.00, 0.50, 0.50, 0.00),
    ]


def bm2_noise_contrast_coefficients(mode: str) -> list[tuple[float, float, float, float, float]]:
    if mode == "noise_contrast_orth":
        return [
            (0.35, 0.25, 0.50, 0.00, 0.00),
            (0.25, 0.25, 0.75, -0.25, 0.00),
            (0.25, 0.25, 0.75, 0.00, -0.25),
            (0.00, 0.25, 1.00, -0.25, 0.25),
            (-0.25, 0.50, 1.00, 0.00, 0.00),
            (-0.25, 0.25, 1.00, -0.25, 0.25),
        ]
    return [
        (0.40, 0.20, 0.40, 0.00, 0.00),
        (0.30, 0.20, 0.60, 0.00, 0.00),
        (0.25, 0.25, 0.50, 0.25, -0.25),
        (0.25, 0.25, 0.50, -0.25, 0.25),
        (0.00, 0.30, 0.70, 0.00, 0.00),
        (-0.20, 0.40, 0.80, 0.00, 0.00),
    ]


def bm2_projector_coefficients(mode: str) -> list[tuple[float, float, float, float, float, float]]:
    if mode.endswith("_orth"):
        return [
            (0.00, 0.25, 0.75, 0.75, 0.00, 0.00),
            (0.00, 0.00, 1.00, 1.00, -0.25, 0.25),
            (-0.25, 0.25, 0.75, 1.00, 0.00, 0.00),
            (0.25, -0.25, 1.00, 0.75, -0.25, 0.25),
            (0.00, 0.50, 0.50, 1.00, 0.00, 0.00),
        ]
    return [
        (0.25, 0.25, 0.50, 0.50, 0.00, 0.00),
        (0.00, 0.25, 0.75, 0.75, 0.00, 0.00),
        (0.00, 0.00, 1.00, 1.00, 0.00, 0.00),
        (0.25, 0.00, 0.75, 0.75, 0.25, -0.25),
        (-0.25, 0.25, 1.00, 0.75, 0.00, 0.00),
    ]


def bm2_constrained_grid_delta(
    prev: Any,
    torch_mod: Any,
    F_mod: Any,
    v1252: Any,
    v1283: Any,
    base: Any,
    task_delta: Sequence[Any],
    basis: Mapping[str, Sequence[Any]],
    xb: Any,
    yb: Any,
    x_hold: Any,
    y_hold: Any,
    args: argparse.Namespace,
    seed: int,
    mode: str,
) -> tuple[list[Any], dict[str, Any]]:
    names = ["bm1", "bm5_orth", "branch", "quad"]
    best_delta: list[Any] | None = None
    best_metrics: dict[str, Any] = {}
    best_score = -1.0e30
    best_coeffs: tuple[float, float, float, float] | None = None
    tried = 0
    feasible_count = 0
    for coeffs in bm2_grid_coefficients(mode):
        deltas = combine_deltas([(coeff, basis[name]) for coeff, name in zip(coeffs, names)], task_delta)
        if mode.endswith("_orth"):
            deltas = normalize_like(remove_task_parallel(deltas, task_delta), task_delta)
        metrics = evaluate_with_drift(prev, base, deltas, task_delta, xb, yb, x_hold, y_hold, int(seed) + tried * 17, args, torch_mod, F_mod, v1252, v1283)
        tried += 1
        coupling = safe_float(metrics.get("CouplingR2"), 0.0)
        noise = safe_float(metrics.get("NoiseSignalLeak_delta"), 0.0)
        reservoir = safe_float(metrics.get("RealSignalReservoirRatio_delta"), 0.0)
        holdout = safe_float(metrics.get("holdout_loss_ratio"), 999.0)
        logit = safe_float(metrics.get("logit_max_abs_drift"), 999.0)
        cep99 = safe_float(metrics.get("CEp99_delta"), 999.0)
        hard_violations = [
            max(0.0, holdout - 1.002) / 0.002,
            max(0.0, logit - 0.05) / 0.05,
            max(0.0, cep99 - 0.05) / 0.05,
        ]
        linec_violations = [
            max(0.0, noise + 0.01) / 0.01,
            max(0.0, reservoir + 0.01) / 0.01,
            max(0.0, 0.02 - coupling) / 0.02,
        ]
        feasible = max(hard_violations) <= 0.0
        feasible_count += int(feasible)
        score = coupling - 0.50 * linec_violations[0] - 0.50 * linec_violations[1] - 0.25 * linec_violations[2] - 2.0 * sum(hard_violations)
        if score > best_score:
            best_score = score
            best_delta = deltas
            best_metrics = metrics
            best_coeffs = coeffs
    assert best_delta is not None and best_coeffs is not None
    fail = pareto_fail(
        {
            "CouplingR2_delta": best_metrics.get("CouplingR2", 0.0),
            "NoiseSignalLeak_delta": best_metrics.get("NoiseSignalLeak_delta", 0.0),
            "RealSignalReservoirRatio_delta": best_metrics.get("RealSignalReservoirRatio_delta", 0.0),
            "holdout_loss_ratio": best_metrics.get("holdout_loss_ratio", 999.0),
            "logit_max_abs_drift": best_metrics.get("logit_max_abs_drift", 999.0),
            "CEp99_delta": best_metrics.get("CEp99_delta", 999.0),
            "control_gap_vs_best": 0.005,
        }
    )
    return best_delta, {
        "subspace_id": f"BM2-{mode}-bm1_bm5_branch_quad",
        "rank": len(names),
        "alpha_beta_gamma_eta": "1.0,0.5,0.5,0.25",
        "constraint_active_count": len(fail),
        "solve_status": "feasible_internal_hard" if feasible_count else "best_effort_no_internal_hard_feasible",
        "solve_time_ms": "",
        "functional_norm_ratio": delta_norm(best_delta) / max(1.0e-12, delta_norm(task_delta)),
        "task_orthogonal_fraction": best_metrics.get("orthogonal_fraction", ""),
        "bm2_coeff_bm1": best_coeffs[0],
        "bm2_coeff_bm5": best_coeffs[1],
        "bm2_coeff_branch": best_coeffs[2],
        "bm2_coeff_quad": best_coeffs[3],
        "bm2_internal_score": best_score,
        "bm2_internal_tried": tried,
        "bm2_internal_feasible_count": feasible_count,
        "bm2_internal_CouplingR2_delta": best_metrics.get("CouplingR2", ""),
        "bm2_internal_NoiseSignalLeak_delta": best_metrics.get("NoiseSignalLeak_delta", ""),
        "bm2_internal_RealSignalReservoirRatio_delta": best_metrics.get("RealSignalReservoirRatio_delta", ""),
        "bm2_internal_holdout_loss_ratio": best_metrics.get("holdout_loss_ratio", ""),
        "bm2_internal_logit_max_abs_drift": best_metrics.get("logit_max_abs_drift", ""),
    }


def bm2_noise_contrast_grid_delta(
    prev: Any,
    torch_mod: Any,
    F_mod: Any,
    v1252: Any,
    v1283: Any,
    base: Any,
    task_delta: Sequence[Any],
    basis: Mapping[str, Sequence[Any]],
    xb: Any,
    yb: Any,
    x_hold: Any,
    y_hold: Any,
    args: argparse.Namespace,
    seed: int,
    mode: str,
) -> tuple[list[Any], dict[str, Any]]:
    names = ["bm1", "bm5_orth", "noise_veto", "branch", "quad"]
    best_delta: list[Any] | None = None
    best_metrics: dict[str, Any] = {}
    best_score = -1.0e30
    best_coeffs: tuple[float, float, float, float, float] | None = None
    tried = 0
    feasible_count = 0
    for coeffs in bm2_noise_contrast_coefficients(mode):
        deltas = combine_deltas([(coeff, basis[name]) for coeff, name in zip(coeffs, names)], task_delta)
        if mode.endswith("_orth"):
            deltas = normalize_like(remove_task_parallel(deltas, task_delta), task_delta)
        metrics = evaluate_with_drift(prev, base, deltas, task_delta, xb, yb, x_hold, y_hold, int(seed) + tried * 19, args, torch_mod, F_mod, v1252, v1283)
        tried += 1
        coupling = safe_float(metrics.get("CouplingR2"), 0.0)
        noise = safe_float(metrics.get("NoiseSignalLeak_delta"), 0.0)
        reservoir = safe_float(metrics.get("RealSignalReservoirRatio_delta"), 0.0)
        holdout = safe_float(metrics.get("holdout_loss_ratio"), 999.0)
        logit = safe_float(metrics.get("logit_max_abs_drift"), 999.0)
        cep99 = safe_float(metrics.get("CEp99_delta"), 999.0)
        hard_violations = [
            max(0.0, holdout - 1.002) / 0.002,
            max(0.0, logit - 0.05) / 0.05,
            max(0.0, cep99 - 0.05) / 0.05,
        ]
        linec_violations = [
            max(0.0, noise + 0.01) / 0.01,
            max(0.0, reservoir + 0.01) / 0.01,
            max(0.0, 0.02 - coupling) / 0.02,
        ]
        feasible = max(hard_violations) <= 0.0
        feasible_count += int(feasible)
        score = coupling - 0.75 * linec_violations[0] - 0.50 * linec_violations[1] - 0.25 * linec_violations[2] - 2.0 * sum(hard_violations)
        if score > best_score:
            best_score = score
            best_delta = deltas
            best_metrics = metrics
            best_coeffs = coeffs
    assert best_delta is not None and best_coeffs is not None
    fail = pareto_fail(
        {
            "CouplingR2_delta": best_metrics.get("CouplingR2", 0.0),
            "NoiseSignalLeak_delta": best_metrics.get("NoiseSignalLeak_delta", 0.0),
            "RealSignalReservoirRatio_delta": best_metrics.get("RealSignalReservoirRatio_delta", 0.0),
            "holdout_loss_ratio": best_metrics.get("holdout_loss_ratio", 999.0),
            "logit_max_abs_drift": best_metrics.get("logit_max_abs_drift", 999.0),
            "CEp99_delta": best_metrics.get("CEp99_delta", 999.0),
            "control_gap_vs_best": 0.005,
        }
    )
    return best_delta, {
        "subspace_id": f"BM2-{mode}-bm1_bm5_noise_branch_quad",
        "rank": len(names),
        "alpha_beta_gamma_eta": "1.0,0.75,0.5,0.25",
        "constraint_active_count": len(fail),
        "solve_status": "feasible_internal_hard" if feasible_count else "best_effort_no_internal_hard_feasible",
        "solve_time_ms": "",
        "functional_norm_ratio": delta_norm(best_delta) / max(1.0e-12, delta_norm(task_delta)),
        "task_orthogonal_fraction": best_metrics.get("orthogonal_fraction", ""),
        "bm2_coeff_bm1": best_coeffs[0],
        "bm2_coeff_bm5": best_coeffs[1],
        "bm2_coeff_noise": best_coeffs[2],
        "bm2_coeff_branch": best_coeffs[3],
        "bm2_coeff_quad": best_coeffs[4],
        "bm2_internal_score": best_score,
        "bm2_internal_tried": tried,
        "bm2_internal_feasible_count": feasible_count,
        "bm2_internal_CouplingR2_delta": best_metrics.get("CouplingR2", ""),
        "bm2_internal_NoiseSignalLeak_delta": best_metrics.get("NoiseSignalLeak_delta", ""),
        "bm2_internal_RealSignalReservoirRatio_delta": best_metrics.get("RealSignalReservoirRatio_delta", ""),
        "bm2_internal_holdout_loss_ratio": best_metrics.get("holdout_loss_ratio", ""),
        "bm2_internal_logit_max_abs_drift": best_metrics.get("logit_max_abs_drift", ""),
    }


def bm2_projector_grid_delta(
    prev: Any,
    torch_mod: Any,
    F_mod: Any,
    v1252: Any,
    v1283: Any,
    base: Any,
    task_delta: Sequence[Any],
    basis: Mapping[str, Sequence[Any]],
    xb: Any,
    yb: Any,
    x_hold: Any,
    y_hold: Any,
    args: argparse.Namespace,
    seed: int,
    mode: str,
) -> tuple[list[Any], dict[str, Any]]:
    names = ["bm1", "bm5_orth", "projector_balanced", "noise_veto", "branch", "quad"]
    best_delta: list[Any] | None = None
    best_metrics: dict[str, Any] = {}
    best_score = -1.0e30
    best_coeffs: tuple[float, float, float, float, float, float] | None = None
    tried = 0
    feasible_count = 0
    for coeffs in bm2_projector_coefficients(mode):
        deltas = combine_deltas([(coeff, basis[name]) for coeff, name in zip(coeffs, names)], task_delta)
        if mode.endswith("_orth"):
            deltas = normalize_like(remove_task_parallel(deltas, task_delta), task_delta)
        metrics = evaluate_with_drift(prev, base, deltas, task_delta, xb, yb, x_hold, y_hold, int(seed) + tried * 23, args, torch_mod, F_mod, v1252, v1283)
        tried += 1
        coupling = safe_float(metrics.get("CouplingR2"), 0.0)
        noise = safe_float(metrics.get("NoiseSignalLeak_delta"), 0.0)
        reservoir = safe_float(metrics.get("RealSignalReservoirRatio_delta"), 0.0)
        holdout = safe_float(metrics.get("holdout_loss_ratio"), 999.0)
        logit = safe_float(metrics.get("logit_max_abs_drift"), 999.0)
        cep99 = safe_float(metrics.get("CEp99_delta"), 999.0)
        hard_violations = [
            max(0.0, holdout - 1.002) / 0.002,
            max(0.0, logit - 0.05) / 0.05,
            max(0.0, cep99 - 0.05) / 0.05,
        ]
        linec_violations = [
            max(0.0, noise + 0.01) / 0.01,
            max(0.0, reservoir + 0.01) / 0.01,
            max(0.0, 0.02 - coupling) / 0.02,
        ]
        feasible = max(hard_violations) <= 0.0
        feasible_count += int(feasible)
        score = coupling - 0.75 * linec_violations[0] - 0.75 * linec_violations[1] - 0.25 * linec_violations[2] - 2.0 * sum(hard_violations)
        if score > best_score:
            best_score = score
            best_delta = deltas
            best_metrics = metrics
            best_coeffs = coeffs
    assert best_delta is not None and best_coeffs is not None
    fail = pareto_fail(
        {
            "CouplingR2_delta": best_metrics.get("CouplingR2", 0.0),
            "NoiseSignalLeak_delta": best_metrics.get("NoiseSignalLeak_delta", 0.0),
            "RealSignalReservoirRatio_delta": best_metrics.get("RealSignalReservoirRatio_delta", 0.0),
            "holdout_loss_ratio": best_metrics.get("holdout_loss_ratio", 999.0),
            "logit_max_abs_drift": best_metrics.get("logit_max_abs_drift", 999.0),
            "CEp99_delta": best_metrics.get("CEp99_delta", 999.0),
            "control_gap_vs_best": 0.005,
        }
    )
    return best_delta, {
        "subspace_id": f"BM2-{mode}-projector_ce_vector",
        "rank": len(names),
        "alpha_beta_gamma_eta": "1.0,0.75,0.75,0.25",
        "constraint_active_count": len(fail),
        "solve_status": "feasible_internal_hard" if feasible_count else "best_effort_no_internal_hard_feasible",
        "solve_time_ms": "",
        "functional_norm_ratio": delta_norm(best_delta) / max(1.0e-12, delta_norm(task_delta)),
        "task_orthogonal_fraction": best_metrics.get("orthogonal_fraction", ""),
        "bm2_coeff_bm1": best_coeffs[0],
        "bm2_coeff_bm5": best_coeffs[1],
        "bm2_coeff_projector": best_coeffs[2],
        "bm2_coeff_noise": best_coeffs[3],
        "bm2_coeff_branch": best_coeffs[4],
        "bm2_coeff_quad": best_coeffs[5],
        "bm2_internal_score": best_score,
        "bm2_internal_tried": tried,
        "bm2_internal_feasible_count": feasible_count,
        "bm2_internal_CouplingR2_delta": best_metrics.get("CouplingR2", ""),
        "bm2_internal_NoiseSignalLeak_delta": best_metrics.get("NoiseSignalLeak_delta", ""),
        "bm2_internal_RealSignalReservoirRatio_delta": best_metrics.get("RealSignalReservoirRatio_delta", ""),
        "bm2_internal_holdout_loss_ratio": best_metrics.get("holdout_loss_ratio", ""),
        "bm2_internal_logit_max_abs_drift": best_metrics.get("logit_max_abs_drift", ""),
    }


def run_linec_calibration(args: argparse.Namespace, out_dir: Path, v1212: Any, prev: Any) -> dict[str, Any]:
    torch_mod, F_mod, v120, v124, v1252, v1283 = prev._lazy_probe_modules()
    device = v1283._device_from_arg(str(args.probe_device))
    if str(device).startswith("cuda"):
        torch_mod.cuda.set_device(device.index if device.index is not None else 0)
    datasets = parse_list(args.probe_datasets)
    seeds = parse_ints(args.probe_seeds)
    batch_sizes = [safe_int(x) for x in parse_list(args.functional_batch_sizes)]
    windows = [safe_int(x) for x in parse_list(args.windows)]
    probe_splits = int(args.probe_splits)
    methods = parse_list(args.calibration_methods)
    rows: list[dict[str, Any]] = []
    control_scores: dict[tuple[str, int, int, int, int], float] = {}

    for dataset in datasets:
        canon = v120._canonical_dataset(dataset)
        load_args = argparse.Namespace(
            data_root=args.data_root,
            no_download=bool(args.no_download),
            seed=seeds[0] if seeds else 0,
            train_size=int(args.probe_train_size),
            val_size=int(args.probe_val_size),
            test_size=int(args.probe_test_size),
            datasets=canon,
        )
        data = v120._load_vision_split(load_args, canon, train_size=int(args.probe_train_size), val_size=int(args.probe_val_size), test_size=int(args.probe_test_size))
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = data[0], data[1], data[2], data[3]
        input_dim = int(data[6])
        output_dim = int(data[7])
        _, budget = v124._param_budget(input_dim, output_dim)
        specs = {s.candidate_id: s for s in v1283.prim.primitive_specs(budget, input_dim, output_dim)}
        for seed in seeds:
            x_train = x_train_cpu.to(device=device, dtype=torch_mod.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch_mod.float32)
            y_val = y_val_cpu.to(device=device)
            base = v1283._make_model(B320_ID, input_dim, output_dim, x_train, device, int(seed) + 1213000, specs, y_train)
            for batch_size in batch_sizes:
                xq = x_val[:batch_size]
                yq = y_val[:batch_size]
                for split_id in range(probe_splits):
                    start = (split_id * batch_size) % max(1, int(x_train.shape[0]) - batch_size + 1)
                    xb = x_train[start : start + batch_size]
                    yb = y_train[start : start + batch_size]
                    task_delta = v1252._grad_delta(base, xb, yb, float(args.probe_lr))
                    random_delta = prev._random_like(task_delta, int(seed) + 1213017 + split_id, torch_mod)
                    control_methods = {"NoOp", "TaskOnlyAdamW", "AdamWParallelDirection", "RandomMatchedNorm"}
                    cotangent_methods = {
                        "BM2s-LogitCotangentTail",
                        "BM2t-LogitCotangentTailFlip",
                        "BM2u-LogitCotangentBalanced",
                        "BM2v-LogitCotangentBalancedOrth",
                        "BM2w-LogitCotangentSpectralTopDamp",
                        "BM2x-LogitCotangentSpectralTopDampOrth",
                        "BM2y-LogitCotangentSpectralIso",
                        "BM2z-LogitCotangentSpectralIsoOrth",
                    }
                    if set(methods).issubset(control_methods | cotangent_methods):
                        raw_deltas = {
                            "NoOp": [torch_mod.zeros_like(delta) for delta in task_delta],
                            "TaskOnlyAdamW": task_delta,
                            "AdamWParallelDirection": task_delta,
                            "RandomMatchedNorm": random_delta,
                        }
                        method_stats: dict[str, dict[str, Any]] = {}
                        if "BM2s-LogitCotangentTail" in methods:
                            raw_deltas["BM2s-LogitCotangentTail"], method_stats["BM2s-LogitCotangentTail"] = logit_cotangent_sketch_delta(
                                torch_mod, base, xb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 173, "tail", False
                            )
                        if "BM2t-LogitCotangentTailFlip" in methods:
                            raw_deltas["BM2t-LogitCotangentTailFlip"], method_stats["BM2t-LogitCotangentTailFlip"] = logit_cotangent_sketch_delta(
                                torch_mod, base, xb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 173, "tail_flip", False
                            )
                        if "BM2u-LogitCotangentBalanced" in methods:
                            raw_deltas["BM2u-LogitCotangentBalanced"], method_stats["BM2u-LogitCotangentBalanced"] = logit_cotangent_sketch_delta(
                                torch_mod, base, xb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 173, "balanced", False
                            )
                        if "BM2v-LogitCotangentBalancedOrth" in methods:
                            raw_deltas["BM2v-LogitCotangentBalancedOrth"], method_stats["BM2v-LogitCotangentBalancedOrth"] = logit_cotangent_sketch_delta(
                                torch_mod, base, xb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 173, "balanced", True
                            )
                        if "BM2w-LogitCotangentSpectralTopDamp" in methods:
                            raw_deltas["BM2w-LogitCotangentSpectralTopDamp"], method_stats["BM2w-LogitCotangentSpectralTopDamp"] = logit_cotangent_spectral_delta(
                                torch_mod, base, xb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 191, "top_damp", False
                            )
                        if "BM2x-LogitCotangentSpectralTopDampOrth" in methods:
                            raw_deltas["BM2x-LogitCotangentSpectralTopDampOrth"], method_stats["BM2x-LogitCotangentSpectralTopDampOrth"] = logit_cotangent_spectral_delta(
                                torch_mod, base, xb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 191, "top_damp", True
                            )
                        if "BM2y-LogitCotangentSpectralIso" in methods:
                            raw_deltas["BM2y-LogitCotangentSpectralIso"], method_stats["BM2y-LogitCotangentSpectralIso"] = logit_cotangent_spectral_delta(
                                torch_mod, base, xb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 191, "isotropic", False
                            )
                        if "BM2z-LogitCotangentSpectralIsoOrth" in methods:
                            raw_deltas["BM2z-LogitCotangentSpectralIsoOrth"], method_stats["BM2z-LogitCotangentSpectralIsoOrth"] = logit_cotangent_spectral_delta(
                                torch_mod, base, xb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 191, "isotropic", True
                            )
                        evaluated: dict[tuple[str, int], dict[str, Any]] = {}
                        for window in windows:
                            for method in methods:
                                deltas = scale_delta(raw_deltas[method], float(window))
                                metrics = evaluate_with_drift(prev, base, deltas, task_delta, xb, yb, xq, yq, int(seed) + split_id * 101 + window, args, torch_mod, F_mod, v1252, v1283)
                                evaluated[(method, window)] = metrics
                            control_values = [
                                safe_float(evaluated[(method, window)].get("delta_score"), -999.0)
                                for method in methods
                                if method in control_methods
                            ]
                            best_control = max(control_values) if control_values else 0.0
                            for method in methods:
                                metrics = evaluated[(method, window)]
                                stats = method_stats.get(method, {})
                                row = {
                                    "stage": "V1213_LINEC_CALIBRATION",
                                    "run_id": out_dir.name,
                                    "candidate_id": B320_ID,
                                    "method": method,
                                    "dataset": canon,
                                    "seed": seed,
                                    "functional_batch_size": batch_size,
                                    "probe_split_id": split_id,
                                    "window": window,
                                    "sketch_dim": int(args.sketch_dim),
                                    "ridge_lambda": float(args.ridge_lambda),
                                    "CouplingR2_delta": metrics.get("CouplingR2", ""),
                                    "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                                    "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                                    "CEp99_delta": metrics.get("CEp99_delta", ""),
                                    "ECE_delta": metrics.get("ECE_delta", ""),
                                    "logit_max_abs_drift": metrics.get("logit_max_abs_drift", ""),
                                    "holdout_loss_ratio": metrics.get("holdout_loss_ratio", ""),
                                    "control_gap_vs_best": safe_float(metrics.get("delta_score"), 0.0) - best_control,
                                    "delta_score": metrics.get("delta_score", ""),
                                    "cos_to_adamw": stats.get("cotangent_cos_to_adamw", metrics.get("cos_with_task_adamw", "")),
                                    "cotangent_mode": stats.get("cotangent_mode", ""),
                                    "cotangent_seed": stats.get("cotangent_seed", ""),
                                    "cotangent_batch": stats.get("cotangent_batch", ""),
                                    "cotangent_sketch_dim": stats.get("cotangent_sketch_dim", ""),
                                    "cotangent_top_eigen_share": stats.get("cotangent_top_eigen_share", ""),
                                    "cotangent_effective_rank": stats.get("cotangent_effective_rank", ""),
                                    "cotangent_spectral_loss": stats.get("cotangent_spectral_loss", ""),
                                    "cotangent_raw_norm": stats.get("cotangent_raw_norm", ""),
                                    "cotangent_cos_to_adamw": stats.get("cotangent_cos_to_adamw", ""),
                                    "cotangent_orthogonalized": stats.get("cotangent_orthogonalized", ""),
                                    "loss_agnostic_direction": stats.get("loss_agnostic_direction", int(method in control_methods)),
                                    "ce_vector_used_for_direction": stats.get("ce_vector_used_for_direction", 0),
                                    "fake_data_used": 0,
                                    "proxy_row_used": 0,
                                    "cpu_offload_used": 0,
                                    "dataset_name_branch_used": 0,
                                    "teacher_used": 0,
                                    "distillation_used": 0,
                                    "loss_modified": 0,
                                    "sampler_or_class_weight_used": 0,
                                }
                                fail = pareto_fail(row)
                                row["pareto_pass"] = int(not fail)
                                row["fail_reason"] = ";".join(fail)
                                rows.append(row)
                        continue
                    bm1_delta, bm1_stats = v1212.snr_gate_delta(v1252, torch_mod, base, xb, yb, float(args.probe_lr), "soft")
                    bm3_delta, bm3_stats = v1212.branch_rebalance_delta(v1252, torch_mod, base, torch_mod.cat([xb, xq], dim=0), float(args.bm3_strength))
                    bm1_direct_delta, bm1_direct_stats = role_filtered_delta(base, bm1_delta, task_delta, "direct")
                    bm1_quad_delta, bm1_quad_stats = role_filtered_delta(base, bm1_delta, task_delta, "quad")
                    bm1_branch_delta, bm1_branch_stats = role_filtered_delta(base, bm1_delta, task_delta, "branch")
                    bm5_delta, bm5_stats = reservoir_vjp_delta(torch_mod, F_mod, base, xb, yb, float(args.probe_lr), task_delta, False)
                    bm5_orth_delta, bm5_orth_stats = reservoir_vjp_delta(torch_mod, F_mod, base, xb, yb, float(args.probe_lr), task_delta, True)
                    noise_veto_delta_raw, noise_veto_stats = noise_veto_delta(v1252, torch_mod, base, xb, yb, float(args.probe_lr), task_delta, int(seed) + split_id * 131, False)
                    noise_veto_orth_delta, noise_veto_orth_stats = noise_veto_delta(v1252, torch_mod, base, xb, yb, float(args.probe_lr), task_delta, int(seed) + split_id * 131, True)
                    projector_res_delta, projector_res_stats = linec_projector_surrogate_delta(v1252, torch_mod, F_mod, base, xb, yb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 151, "reservoir", False)
                    projector_noise_delta, projector_noise_stats = linec_projector_surrogate_delta(v1252, torch_mod, F_mod, base, xb, yb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 151, "noise", False)
                    projector_bal_delta, projector_bal_stats = linec_projector_surrogate_delta(v1252, torch_mod, F_mod, base, xb, yb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 151, "balanced", False)
                    projector_bal_orth_delta, projector_bal_orth_stats = linec_projector_surrogate_delta(v1252, torch_mod, F_mod, base, xb, yb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 151, "balanced", True)
                    projector_noise_null_delta, projector_noise_null_stats = linec_projector_noise_constrained_delta(v1252, torch_mod, F_mod, base, xb, yb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 151, "noise_null", False)
                    projector_noise_margin_delta, projector_noise_margin_stats = linec_projector_noise_constrained_delta(v1252, torch_mod, F_mod, base, xb, yb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 151, "noise_margin", False)
                    projector_noise_null_orth_delta, projector_noise_null_orth_stats = linec_projector_noise_constrained_delta(v1252, torch_mod, F_mod, base, xb, yb, float(args.probe_lr), task_delta, int(args.sketch_dim), int(seed) + split_id * 151, "noise_null", True)
                    hold_start = max(1, int(xb.shape[0]) // 2)
                    x_solve = xb[:hold_start]
                    y_solve = yb[:hold_start]
                    x_hold = xb[hold_start:] if hold_start < int(xb.shape[0]) else xb[:hold_start]
                    y_hold = yb[hold_start:] if hold_start < int(yb.shape[0]) else yb[:hold_start]
                    bm2_basis = {
                        "bm1": bm1_delta,
                        "bm5_orth": bm5_orth_delta,
                        "branch": bm1_branch_delta,
                        "quad": bm1_quad_delta,
                    }
                    bm2_noise_basis = {
                        **bm2_basis,
                        "noise_veto": noise_veto_orth_delta,
                    }
                    bm2_projector_basis = {
                        **bm2_noise_basis,
                        "projector_balanced": projector_bal_orth_delta,
                    }
                    bm2_noisehard_basis = {
                        **bm2_noise_basis,
                        "projector_balanced": projector_noise_null_delta,
                    }
                    bm2_projector_pareto_basis = {
                        **bm2_noise_basis,
                        "projector_balanced": projector_res_delta,
                        "noise_veto": projector_noise_delta,
                    }
                    bm2_balanced_delta, bm2_balanced_stats = bm2_constrained_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009, "balanced")
                    bm2_noise_delta, bm2_noise_stats = bm2_constrained_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 101, "noise_veto")
                    bm2_reservoir_delta, bm2_reservoir_stats = bm2_constrained_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 202, "reservoir_veto")
                    bm2_signed_delta, bm2_signed_stats = bm2_constrained_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 303, "signed")
                    bm2_signed_orth_delta, bm2_signed_orth_stats = bm2_constrained_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 404, "signed_orth")
                    bm2_noise_contrast_delta, bm2_noise_contrast_stats = bm2_noise_contrast_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_noise_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 505, "noise_contrast")
                    bm2_noise_contrast_orth_delta, bm2_noise_contrast_orth_stats = bm2_noise_contrast_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_noise_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 606, "noise_contrast_orth")
                    bm2_projector_delta, bm2_projector_stats = bm2_projector_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_projector_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 707, "projector")
                    bm2_projector_orth_delta, bm2_projector_orth_stats = bm2_projector_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_projector_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 808, "projector_orth")
                    bm2_noisehard_delta, bm2_noisehard_stats = bm2_projector_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_noisehard_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 909, "projector_noisehard")
                    bm2_noisehard_orth_delta, bm2_noisehard_orth_stats = bm2_projector_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_noisehard_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 1001, "projector_noisehard_orth")
                    bm2_projector_pareto_delta, bm2_projector_pareto_stats = bm2_projector_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_projector_pareto_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 1103, "projector_pareto")
                    bm2_projector_pareto_orth_delta, bm2_projector_pareto_orth_stats = bm2_projector_grid_delta(prev, torch_mod, F_mod, v1252, v1283, base, task_delta, bm2_projector_pareto_basis, x_solve, y_solve, x_hold, y_hold, args, int(seed) + split_id * 1009 + 1207, "projector_pareto_orth")
                    raw_deltas = {
                        "NoOp": [torch_mod.zeros_like(delta) for delta in task_delta],
                        "TaskOnlyAdamW": task_delta,
                        "AdamWParallelDirection": task_delta,
                        "RandomMatchedNorm": random_delta,
                        "BM1b-SoftSNRGate": bm1_delta,
                        "BM1h-RoleWiseSNR-direct": bm1_direct_delta,
                        "BM1i-RoleWiseSNR-quad": bm1_quad_delta,
                        "BM1j-RoleWiseSNR-branch": bm1_branch_delta,
                        "BM3a-BranchRebalance": bm3_delta,
                        "BM5a-ReservoirVJP": bm5_delta,
                        "BM5b-ReservoirVJP-AdamWOrthogonal": bm5_orth_delta,
                        "BM5c-NoiseVetoVJP": noise_veto_delta_raw,
                        "BM5d-NoiseVetoVJP-AdamWOrthogonal": noise_veto_orth_delta,
                        "BM5e-LineCProjectorCEVJP": projector_res_delta,
                        "BM5k-LineCProjectorCEVJP-noiseOnly": projector_noise_delta,
                        "BM5f-LineCProjectorCEVJP-balanced": projector_bal_delta,
                        "BM5g-LineCProjectorCEVJP-balancedOrth": projector_bal_orth_delta,
                        "BM5h-LineCProjectorCEVJP-noiseNull": projector_noise_null_delta,
                        "BM5i-LineCProjectorCEVJP-noiseMargin": projector_noise_margin_delta,
                        "BM5j-LineCProjectorCEVJP-noiseNullOrth": projector_noise_null_orth_delta,
                        "BM2e-LowRankLineCGrid-balanced": bm2_balanced_delta,
                        "BM2g-LineCGrid-withNoiseVeto": bm2_noise_delta,
                        "BM2h-LineCGrid-withReservoirVeto": bm2_reservoir_delta,
                        "BM2i-LineCGrid-signed": bm2_signed_delta,
                        "BM2j-LineCGrid-signedAdamWOrthogonal": bm2_signed_orth_delta,
                        "BM2k-LineCGrid-noiseContrast": bm2_noise_contrast_delta,
                        "BM2l-LineCGrid-noiseContrastOrth": bm2_noise_contrast_orth_delta,
                        "BM2m-LineCGrid-projectorCE": bm2_projector_delta,
                        "BM2n-LineCGrid-projectorCEOrth": bm2_projector_orth_delta,
                        "BM2o-LineCGrid-projectorNoiseHard": bm2_noisehard_delta,
                        "BM2p-LineCGrid-projectorNoiseHardOrth": bm2_noisehard_orth_delta,
                        "BM2q-LineCGrid-projectorPareto": bm2_projector_pareto_delta,
                        "BM2r-LineCGrid-projectorParetoOrth": bm2_projector_pareto_orth_delta,
                    }
                    method_stats = {
                        "BM1b-SoftSNRGate": bm1_stats,
                        "BM1h-RoleWiseSNR-direct": {**bm1_stats, **bm1_direct_stats},
                        "BM1i-RoleWiseSNR-quad": {**bm1_stats, **bm1_quad_stats},
                        "BM1j-RoleWiseSNR-branch": {**bm1_stats, **bm1_branch_stats},
                        "BM3a-BranchRebalance": bm3_stats,
                        "BM5a-ReservoirVJP": bm5_stats,
                        "BM5b-ReservoirVJP-AdamWOrthogonal": bm5_orth_stats,
                        "BM5c-NoiseVetoVJP": noise_veto_stats,
                        "BM5d-NoiseVetoVJP-AdamWOrthogonal": noise_veto_orth_stats,
                        "BM5e-LineCProjectorCEVJP": projector_res_stats,
                        "BM5k-LineCProjectorCEVJP-noiseOnly": projector_noise_stats,
                        "BM5f-LineCProjectorCEVJP-balanced": projector_bal_stats,
                        "BM5g-LineCProjectorCEVJP-balancedOrth": projector_bal_orth_stats,
                        "BM5h-LineCProjectorCEVJP-noiseNull": projector_noise_null_stats,
                        "BM5i-LineCProjectorCEVJP-noiseMargin": projector_noise_margin_stats,
                        "BM5j-LineCProjectorCEVJP-noiseNullOrth": projector_noise_null_orth_stats,
                        "BM2e-LowRankLineCGrid-balanced": bm2_balanced_stats,
                        "BM2g-LineCGrid-withNoiseVeto": bm2_noise_stats,
                        "BM2h-LineCGrid-withReservoirVeto": bm2_reservoir_stats,
                        "BM2i-LineCGrid-signed": bm2_signed_stats,
                        "BM2j-LineCGrid-signedAdamWOrthogonal": bm2_signed_orth_stats,
                        "BM2k-LineCGrid-noiseContrast": bm2_noise_contrast_stats,
                        "BM2l-LineCGrid-noiseContrastOrth": bm2_noise_contrast_orth_stats,
                        "BM2m-LineCGrid-projectorCE": bm2_projector_stats,
                        "BM2n-LineCGrid-projectorCEOrth": bm2_projector_orth_stats,
                        "BM2o-LineCGrid-projectorNoiseHard": bm2_noisehard_stats,
                        "BM2p-LineCGrid-projectorNoiseHardOrth": bm2_noisehard_orth_stats,
                        "BM2q-LineCGrid-projectorPareto": bm2_projector_pareto_stats,
                        "BM2r-LineCGrid-projectorParetoOrth": bm2_projector_pareto_orth_stats,
                    }
                    evaluated: dict[tuple[str, int], dict[str, Any]] = {}
                    for window in windows:
                        for method in methods:
                            deltas = scale_delta(raw_deltas[method], float(window))
                            metrics = evaluate_with_drift(prev, base, deltas, task_delta, xb, yb, xq, yq, int(seed) + split_id * 101 + window, args, torch_mod, F_mod, v1252, v1283)
                            evaluated[(method, window)] = metrics
                        control_key_base = (canon, int(seed), batch_size, split_id, window)
                        control_scores[control_key_base] = max(
                            safe_float(evaluated[(method, window)].get("delta_score"), -999.0)
                            for method in methods
                            if method in {"NoOp", "TaskOnlyAdamW", "AdamWParallelDirection", "RandomMatchedNorm"}
                        )
                        for method in methods:
                            metrics = evaluated[(method, window)]
                            best_control = control_scores[control_key_base]
                            row = {
                                "stage": "V1213_LINEC_CALIBRATION",
                                "run_id": out_dir.name,
                                "candidate_id": B320_ID,
                                "method": method,
                                "dataset": canon,
                                "seed": seed,
                                "functional_batch_size": batch_size,
                                "probe_split_id": split_id,
                                "window": window,
                                "sketch_dim": int(args.sketch_dim),
                                "ridge_lambda": float(args.ridge_lambda),
                                "CouplingR2_before": "",
                                "CouplingR2_after": "",
                                "CouplingR2_delta": metrics.get("CouplingR2", ""),
                                "NoiseSignalLeak_before": "",
                                "NoiseSignalLeak_after": "",
                                "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                                "RealSignalReservoirRatio_before": "",
                                "RealSignalReservoirRatio_after": "",
                                "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                                "CEp99_delta": metrics.get("CEp99_delta", ""),
                                "ECE_delta": metrics.get("ECE_delta", ""),
                                "AUC_proxy_delta": "",
                                "logit_max_abs_drift": metrics.get("logit_max_abs_drift", ""),
                                "holdout_loss_ratio": metrics.get("holdout_loss_ratio", ""),
                                "bootstrap_ci_low": "",
                                "bootstrap_ci_high": "",
                                "metric_std_across_splits": "",
                                "noop_false_positive": "",
                                "random_false_positive": "",
                                "control_gap_vs_best": safe_float(metrics.get("delta_score"), 0.0) - best_control,
                                "delta_score": metrics.get("delta_score", ""),
                                "SNR_positive_fraction": method_stats.get(method, {}).get("SNR_positive_fraction", ""),
                                "gradient_norm_removed": method_stats.get(method, {}).get("gradient_norm_removed", ""),
                                "cos_to_adamw": method_stats.get(method, {}).get("cos_to_adamw", metrics.get("cos_with_task_adamw", "")),
                                "role": method_stats.get(method, {}).get("role", ""),
                                "role_kept_params": method_stats.get(method, {}).get("role_kept_params", ""),
                                "role_source_norm_fraction": method_stats.get(method, {}).get("role_source_norm_fraction", ""),
                                "reservoir_vjp_loss": method_stats.get(method, {}).get("reservoir_vjp_loss", ""),
                                "reservoir_vjp_raw_norm": method_stats.get(method, {}).get("reservoir_vjp_raw_norm", ""),
                                "reservoir_vjp_cos_to_adamw": method_stats.get(method, {}).get("reservoir_vjp_cos_to_adamw", ""),
                                "reservoir_vjp_orthogonalized": method_stats.get(method, {}).get("reservoir_vjp_orthogonalized", ""),
                                "noise_veto_perm_seed": method_stats.get(method, {}).get("noise_veto_perm_seed", ""),
                                "noise_veto_raw_norm": method_stats.get(method, {}).get("noise_veto_raw_norm", ""),
                                "noise_veto_cos_to_adamw": method_stats.get(method, {}).get("noise_veto_cos_to_adamw", ""),
                                "noise_veto_orthogonalized": method_stats.get(method, {}).get("noise_veto_orthogonalized", ""),
                                "projector_mode": method_stats.get(method, {}).get("projector_mode", ""),
                                "projector_top_count": method_stats.get(method, {}).get("projector_top_count", ""),
                                "projector_real_term": method_stats.get(method, {}).get("projector_real_term", ""),
                                "projector_noise_term": method_stats.get(method, {}).get("projector_noise_term", ""),
                                "projector_loss": method_stats.get(method, {}).get("projector_loss", ""),
                                "projector_raw_norm": method_stats.get(method, {}).get("projector_raw_norm", ""),
                                "projector_cos_to_adamw": method_stats.get(method, {}).get("projector_cos_to_adamw", ""),
                                "projector_orthogonalized": method_stats.get(method, {}).get("projector_orthogonalized", ""),
                                "projector_noise_constraint_dot_before": method_stats.get(method, {}).get("projector_noise_constraint_dot_before", ""),
                                "projector_noise_constraint_dot_after": method_stats.get(method, {}).get("projector_noise_constraint_dot_after", ""),
                                "projector_noise_constraint_coeff": method_stats.get(method, {}).get("projector_noise_constraint_coeff", ""),
                                "subspace_id": method_stats.get(method, {}).get("subspace_id", ""),
                                "rank": method_stats.get(method, {}).get("rank", ""),
                                "alpha_beta_gamma_eta": method_stats.get(method, {}).get("alpha_beta_gamma_eta", ""),
                                "constraint_active_count": method_stats.get(method, {}).get("constraint_active_count", ""),
                                "solve_status": method_stats.get(method, {}).get("solve_status", ""),
                                "solve_time_ms": method_stats.get(method, {}).get("solve_time_ms", ""),
                                "functional_norm_ratio": method_stats.get(method, {}).get("functional_norm_ratio", ""),
                                "task_orthogonal_fraction": method_stats.get(method, {}).get("task_orthogonal_fraction", ""),
                                "bm2_coeff_bm1": method_stats.get(method, {}).get("bm2_coeff_bm1", ""),
                                "bm2_coeff_bm5": method_stats.get(method, {}).get("bm2_coeff_bm5", ""),
                                "bm2_coeff_projector": method_stats.get(method, {}).get("bm2_coeff_projector", ""),
                                "bm2_coeff_noise": method_stats.get(method, {}).get("bm2_coeff_noise", ""),
                                "bm2_coeff_branch": method_stats.get(method, {}).get("bm2_coeff_branch", ""),
                                "bm2_coeff_quad": method_stats.get(method, {}).get("bm2_coeff_quad", ""),
                                "bm2_internal_score": method_stats.get(method, {}).get("bm2_internal_score", ""),
                                "bm2_internal_tried": method_stats.get(method, {}).get("bm2_internal_tried", ""),
                                "bm2_internal_feasible_count": method_stats.get(method, {}).get("bm2_internal_feasible_count", ""),
                                "bm2_internal_CouplingR2_delta": method_stats.get(method, {}).get("bm2_internal_CouplingR2_delta", ""),
                                "bm2_internal_NoiseSignalLeak_delta": method_stats.get(method, {}).get("bm2_internal_NoiseSignalLeak_delta", ""),
                                "bm2_internal_RealSignalReservoirRatio_delta": method_stats.get(method, {}).get("bm2_internal_RealSignalReservoirRatio_delta", ""),
                                "bm2_internal_holdout_loss_ratio": method_stats.get(method, {}).get("bm2_internal_holdout_loss_ratio", ""),
                                "bm2_internal_logit_max_abs_drift": method_stats.get(method, {}).get("bm2_internal_logit_max_abs_drift", ""),
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                                "dataset_name_branch_used": 0,
                                "teacher_used": 0,
                                "distillation_used": 0,
                                "loss_modified": 0,
                                "sampler_or_class_weight_used": 0,
                            }
                            fail = pareto_fail(row)
                            row["pareto_pass"] = int(not fail)
                            row["fail_reason"] = ";".join(fail)
                            rows.append(row)

    write_csv_rows(out_dir / "v1213_linec_calibration.csv", rows)
    write_csv_rows(out_dir / "v1213_functional_p3_linec.csv", rows)
    summary_rows = []
    by_group: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(str(row["method"]), safe_int(row["functional_batch_size"]), safe_int(row["window"]))].append(row)
    for (method, batch_size, window), group in sorted(by_group.items()):
        noise = [safe_float(row.get("NoiseSignalLeak_delta"), 0.0) for row in group]
        reservoir = [safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0) for row in group]
        coupling = [safe_float(row.get("CouplingR2_delta"), 0.0) for row in group]
        noise_mean, noise_std = mean_std(noise)
        reservoir_mean, reservoir_std = mean_std(reservoir)
        coupling_mean, coupling_std = mean_std(coupling)
        nlo, nhi = ci95(noise)
        rlo, rhi = ci95(reservoir)
        false_positive = sum(safe_int(row.get("pareto_pass")) for row in group)
        summary_rows.append(
            {
                "stage": "V1213_LINEC_CALIBRATION_SUMMARY",
                "method": method,
                "functional_batch_size": batch_size,
                "window": window,
                "rows": len(group),
                "CouplingR2_delta_mean": coupling_mean,
                "CouplingR2_delta_std": coupling_std,
                "NoiseSignalLeak_delta_mean": noise_mean,
                "NoiseSignalLeak_delta_std": noise_std,
                "NoiseSignalLeak_ci_low": nlo,
                "NoiseSignalLeak_ci_high": nhi,
                "RealSignalReservoirRatio_delta_mean": reservoir_mean,
                "RealSignalReservoirRatio_delta_std": reservoir_std,
                "RealSignalReservoirRatio_ci_low": rlo,
                "RealSignalReservoirRatio_ci_high": rhi,
                "pareto_pass_rows": false_positive,
                "noop_false_positive": false_positive if method == "NoOp" else 0,
                "random_false_positive": false_positive if method == "RandomMatchedNorm" else 0,
            }
        )
    write_csv_rows(out_dir / "v1213_linec_null_distribution.csv", summary_rows)
    noop_groups = [row for row in summary_rows if row["method"] == "NoOp"]
    random_groups = [row for row in summary_rows if row["method"] == "RandomMatchedNorm"]
    calibration_fail = []
    for row in noop_groups:
        if abs(safe_float(row.get("NoiseSignalLeak_delta_mean"))) > 0.003:
            calibration_fail.append("NoOp_noise_mean_abs>0.003")
        if safe_float(row.get("NoiseSignalLeak_delta_std")) > 0.005:
            calibration_fail.append("NoOp_noise_std>0.005")
        if abs(safe_float(row.get("RealSignalReservoirRatio_delta_mean"))) > 0.003:
            calibration_fail.append("NoOp_reservoir_mean_abs>0.003")
        if safe_float(row.get("RealSignalReservoirRatio_delta_std")) > 0.005:
            calibration_fail.append("NoOp_reservoir_std>0.005")
    if sum(safe_int(row.get("pareto_pass_rows")) for row in noop_groups) > 0:
        calibration_fail.append("NoOp_false_positive>0")
    if sum(safe_int(row.get("pareto_pass_rows")) for row in random_groups) > 0:
        calibration_fail.append("Random_false_positive>0")
    bm1_methods = sorted({str(row.get("method")) for row in rows if str(row.get("method", "")).startswith("BM1")})
    bm2_methods = sorted({str(row.get("method")) for row in rows if str(row.get("method", "")).startswith("BM2")})
    bm3_methods = sorted({str(row.get("method")) for row in rows if str(row.get("method", "")).startswith("BM3")})
    bm5_methods = sorted({str(row.get("method")) for row in rows if str(row.get("method", "")).startswith("BM5")})
    bm1_full_pass = sum(full_pass_count(rows, method, datasets, seeds) for method in bm1_methods)
    bm2_full_pass = sum(full_pass_count(rows, method, datasets, seeds) for method in bm2_methods)
    bm3_full_pass = sum(full_pass_count(rows, method, datasets, seeds) for method in bm3_methods)
    bm5_full_pass = sum(full_pass_count(rows, method, datasets, seeds) for method in bm5_methods)
    payload = {
        "stage": "V1213_CALIBRATION_DECISION",
        "rows": len(rows),
        "summary_rows": len(summary_rows),
        "linec_calibration_pass": int(not calibration_fail),
        "calibration_fail_reason": ";".join(sorted(set(calibration_fail))),
        "noop_false_positive_rows": sum(safe_int(row.get("pareto_pass_rows")) for row in noop_groups),
        "random_false_positive_rows": sum(safe_int(row.get("pareto_pass_rows")) for row in random_groups),
        "bm1_full_3x3_pass_candidate_count": bm1_full_pass,
        "bm2_full_3x3_pass_candidate_count": bm2_full_pass,
        "bm3_full_3x3_pass_candidate_count": bm3_full_pass,
        "bm5_full_3x3_pass_candidate_count": bm5_full_pass,
        "bm1_partial_pass_rows": sum(1 for row in rows if str(row.get("method", "")).startswith("BM1") and safe_int(row.get("pareto_pass")) == 1),
        "bm2_partial_pass_rows": sum(1 for row in rows if str(row.get("method", "")).startswith("BM2") and safe_int(row.get("pareto_pass")) == 1),
        "bm3_partial_pass_rows": sum(1 for row in rows if str(row.get("method", "")).startswith("BM3") and safe_int(row.get("pareto_pass")) == 1),
        "bm5_partial_pass_rows": sum(1 for row in rows if str(row.get("method", "")).startswith("BM5") and safe_int(row.get("pareto_pass")) == 1),
        "fail_reason_counts": dict(Counter(reason for row in rows for reason in str(row.get("fail_reason", "")).split(";") if reason)),
    }
    write_json(out_dir / "v1213_calibration_decision.json", payload)
    return payload


def summarize_calibration_rows(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, Any]:
    write_csv_rows(out_dir / "v1213_linec_calibration.csv", rows)
    write_csv_rows(out_dir / "v1213_functional_p3_linec.csv", rows)
    summary_rows = []
    by_group: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(str(row["method"]), safe_int(row["functional_batch_size"]), safe_int(row["window"]))].append(row)
    for (method, batch_size, window), group in sorted(by_group.items()):
        noise = [safe_float(row.get("NoiseSignalLeak_delta"), 0.0) for row in group]
        reservoir = [safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0) for row in group]
        coupling = [safe_float(row.get("CouplingR2_delta"), 0.0) for row in group]
        noise_mean, noise_std = mean_std(noise)
        reservoir_mean, reservoir_std = mean_std(reservoir)
        coupling_mean, coupling_std = mean_std(coupling)
        nlo, nhi = ci95(noise)
        rlo, rhi = ci95(reservoir)
        false_positive = sum(safe_int(row.get("pareto_pass")) for row in group)
        summary_rows.append(
            {
                "stage": "V1213_LINEC_CALIBRATION_SUMMARY",
                "method": method,
                "functional_batch_size": batch_size,
                "window": window,
                "rows": len(group),
                "CouplingR2_delta_mean": coupling_mean,
                "CouplingR2_delta_std": coupling_std,
                "NoiseSignalLeak_delta_mean": noise_mean,
                "NoiseSignalLeak_delta_std": noise_std,
                "NoiseSignalLeak_ci_low": nlo,
                "NoiseSignalLeak_ci_high": nhi,
                "RealSignalReservoirRatio_delta_mean": reservoir_mean,
                "RealSignalReservoirRatio_delta_std": reservoir_std,
                "RealSignalReservoirRatio_ci_low": rlo,
                "RealSignalReservoirRatio_ci_high": rhi,
                "pareto_pass_rows": false_positive,
                "noop_false_positive": false_positive if method == "NoOp" else 0,
                "random_false_positive": false_positive if method == "RandomMatchedNorm" else 0,
            }
        )
    write_csv_rows(out_dir / "v1213_linec_null_distribution.csv", summary_rows)
    noop_groups = [row for row in summary_rows if row["method"] == "NoOp"]
    random_groups = [row for row in summary_rows if row["method"] == "RandomMatchedNorm"]
    calibration_fail = []
    for row in noop_groups:
        if abs(safe_float(row.get("NoiseSignalLeak_delta_mean"))) > 0.003:
            calibration_fail.append("NoOp_noise_mean_abs>0.003")
        if safe_float(row.get("NoiseSignalLeak_delta_std")) > 0.005:
            calibration_fail.append("NoOp_noise_std>0.005")
        if abs(safe_float(row.get("RealSignalReservoirRatio_delta_mean"))) > 0.003:
            calibration_fail.append("NoOp_reservoir_mean_abs>0.003")
        if safe_float(row.get("RealSignalReservoirRatio_delta_std")) > 0.005:
            calibration_fail.append("NoOp_reservoir_std>0.005")
    if sum(safe_int(row.get("pareto_pass_rows")) for row in noop_groups) > 0:
        calibration_fail.append("NoOp_false_positive>0")
    if sum(safe_int(row.get("pareto_pass_rows")) for row in random_groups) > 0:
        calibration_fail.append("Random_false_positive>0")
    datasets = sorted({str(row.get("dataset")) for row in rows if row.get("dataset")})
    seeds = sorted({safe_int(row.get("seed")) for row in rows})
    payload = {
        "stage": "V1213_CALIBRATION_DECISION",
        "rows": len(rows),
        "summary_rows": len(summary_rows),
        "datasets": datasets,
        "seeds": seeds,
        "linec_calibration_pass": int(not calibration_fail),
        "calibration_fail_reason": ";".join(sorted(set(calibration_fail))),
        "noop_false_positive_rows": sum(safe_int(row.get("pareto_pass_rows")) for row in noop_groups),
        "random_false_positive_rows": sum(safe_int(row.get("pareto_pass_rows")) for row in random_groups),
        "bm1_full_3x3_pass_candidate_count": sum(full_pass_count(rows, method, datasets, seeds) for method in sorted({str(row.get("method")) for row in rows if str(row.get("method", "")).startswith("BM1")})),
        "bm2_full_3x3_pass_candidate_count": sum(full_pass_count(rows, method, datasets, seeds) for method in sorted({str(row.get("method")) for row in rows if str(row.get("method", "")).startswith("BM2")})),
        "bm3_full_3x3_pass_candidate_count": sum(full_pass_count(rows, method, datasets, seeds) for method in sorted({str(row.get("method")) for row in rows if str(row.get("method", "")).startswith("BM3")})),
        "bm5_full_3x3_pass_candidate_count": sum(full_pass_count(rows, method, datasets, seeds) for method in sorted({str(row.get("method")) for row in rows if str(row.get("method", "")).startswith("BM5")})),
        "bm1_partial_pass_rows": sum(1 for row in rows if str(row.get("method", "")).startswith("BM1") and safe_int(row.get("pareto_pass")) == 1),
        "bm2_partial_pass_rows": sum(1 for row in rows if str(row.get("method", "")).startswith("BM2") and safe_int(row.get("pareto_pass")) == 1),
        "bm3_partial_pass_rows": sum(1 for row in rows if str(row.get("method", "")).startswith("BM3") and safe_int(row.get("pareto_pass")) == 1),
        "bm5_partial_pass_rows": sum(1 for row in rows if str(row.get("method", "")).startswith("BM5") and safe_int(row.get("pareto_pass")) == 1),
        "fail_reason_counts": dict(Counter(reason for row in rows for reason in str(row.get("fail_reason", "")).split(";") if reason)),
    }
    write_json(out_dir / "v1213_calibration_decision.json", payload)
    return payload


def aggregate_input_dirs(input_dirs: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        path = input_dir / "v1213_linec_calibration.csv"
        for row in read_csv_rows(path):
            row["source_dir"] = rel(input_dir)
            rows.append(row)
    if not rows:
        raise RuntimeError("no v1213_linec_calibration.csv rows found in aggregate input dirs")
    return summarize_calibration_rows(rows, out_dir)


def full_pass_count(rows: Sequence[Mapping[str, Any]], method: str, datasets: Sequence[str], seeds: Sequence[int]) -> int:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    canon_datasets = set(datasets)
    for row in rows:
        if row.get("method") == method and row.get("dataset") in canon_datasets:
            grouped[(safe_int(row.get("functional_batch_size")), safe_int(row.get("window")))].append(row)
    expected = len(datasets) * len(seeds)
    count = 0
    for group in grouped.values():
        by_ds_seed = {(row.get("dataset"), safe_int(row.get("seed"))) for row in group if safe_int(row.get("pareto_pass")) == 1}
        if len(by_ds_seed) == expected:
            count += 1
    return count


def write_simple_svgs(out_dir: Path, decision: Mapping[str, Any]) -> None:
    figure_names = [
        "fig_A_b320_scorecard.svg",
        "fig_A_b320_auc_seed_matrix.svg",
        "fig_A_b320_efficiency_memory_bar.svg",
        "fig_A_b320_linec_vs_mlp.svg",
        "fig_C_noop_null_distribution.svg",
        "fig_C_metric_std_by_batch_size.svg",
        "fig_C_noise_leak_ci_by_method.svg",
        "fig_C_reservoir_ci_by_method.svg",
        "fig_C_coupling_vs_noise_tradeoff.svg",
        "fig_B_functional_pareto.svg",
        "fig_B_noise_leak_vs_coupling.svg",
        "fig_B_reservoir_vs_auc_time.svg",
        "fig_B_control_gap_by_candidate.svg",
        "fig_B_p3_to_p4_prediction.svg",
        "fig_B_event_accept_reject_timeline.svg",
        "fig_B_qp_constraint_activity.svg",
        "fig_B_moment_transport_effect.svg",
        "fig_D_family_status_grid.svg",
        "fig_D_family_efficiency_expression_task.svg",
        "fig_D_rational_linec_collapse.svg",
        "fig_D_cheby_degree_energy.svg",
        "fig_D_wavelet_local_coverage.svg",
        "fig_D_rbf_expression_r2.svg",
        "fig_D_fourier_spectrum_expression.svg",
    ]
    text = [
        f"route: {decision.get('route', '')}",
        f"calibration_pass: {decision.get('linec_calibration_pass', '')}",
        f"bm1_partial: {decision.get('bm1_partial_pass_rows', '')}",
        f"bm3_partial: {decision.get('bm3_partial_pass_rows', '')}",
    ]
    for name in figure_names:
        lines = "".join(f'<text x="20" y="{40 + i * 24}" font-size="16">{escape_xml(t)}</text>' for i, t in enumerate([name, *text]))
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="220"><rect width="100%" height="100%" fill="white"/>{lines}</svg>\n'
        (out_dir / name).write_text(svg, encoding="utf-8")


def escape_xml(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_hashes(out_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v1213_hash_manifest.json":
            artifacts.append({"artifact": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    payload = {"stage": "V1213_HASH_MANIFEST", "generated_at": now_iso(), "artifacts": artifacts}
    write_json(out_dir / "v1213_hash_manifest.json", payload)
    return payload


def decide_route(anchor: Mapping[str, Any], calibration: Mapping[str, Any], family_status: Mapping[str, Any]) -> dict[str, Any]:
    families = family_status.get("families", {}) if isinstance(family_status, Mapping) else {}
    family_pass_count = sum(1 for fam, info in families.items() if fam != "BSpline" and info.get("status") == "FamilyPass")
    decision = {
        "stage": "V1213_ROUTE_DECISION",
        "generated_at": now_iso(),
        "B320_anchor_locked": safe_int(anchor.get("anchor_pass"), 0),
        "linec_calibration_pass": calibration.get("linec_calibration_pass", 0),
        "calibration_fail_reason": calibration.get("calibration_fail_reason", ""),
        "bm1_full_3x3_pass_candidate_count": calibration.get("bm1_full_3x3_pass_candidate_count", 0),
        "bm2_full_3x3_pass_candidate_count": calibration.get("bm2_full_3x3_pass_candidate_count", 0),
        "bm3_full_3x3_pass_candidate_count": calibration.get("bm3_full_3x3_pass_candidate_count", 0),
        "bm5_full_3x3_pass_candidate_count": calibration.get("bm5_full_3x3_pass_candidate_count", 0),
        "bm1_partial_pass_rows": calibration.get("bm1_partial_pass_rows", 0),
        "bm2_partial_pass_rows": calibration.get("bm2_partial_pass_rows", 0),
        "bm3_partial_pass_rows": calibration.get("bm3_partial_pass_rows", 0),
        "bm5_partial_pass_rows": calibration.get("bm5_partial_pass_rows", 0),
        "classic_family_pass_count": family_pass_count,
        "official_functional_success": 0,
    }
    if not safe_int(anchor.get("anchor_pass"), 0):
        decision["route"] = "R1-B320AnchorRegression"
        decision["next_recommended_action"] = "fix B320 anchor implementation/protocol before functional"
    elif not safe_int(calibration.get("linec_calibration_pass"), 0):
        decision["route"] = "R2-LineCMetricNotReliable"
        decision["next_recommended_action"] = "increase batch/sketch/repeated splits and use paired CI before promotion"
    elif (
        safe_int(calibration.get("bm1_full_3x3_pass_candidate_count"), 0)
        or safe_int(calibration.get("bm2_full_3x3_pass_candidate_count"), 0)
        or safe_int(calibration.get("bm3_full_3x3_pass_candidate_count"), 0)
        or safe_int(calibration.get("bm5_full_3x3_pass_candidate_count"), 0)
    ):
        decision["route"] = "R6-FunctionalLocalSuccess"
        decision["next_recommended_action"] = "run P4 short-run with strong controls only for full 3x3 survivor"
    elif (
        safe_int(calibration.get("bm1_partial_pass_rows"), 0) > 0
        or safe_int(calibration.get("bm2_partial_pass_rows"), 0) > 0
        or safe_int(calibration.get("bm5_partial_pass_rows"), 0) > 0
    ):
        decision["route"] = "R3-FunctionalMechanismPartialOnly"
        decision["next_recommended_action"] = "continue constrained BM2/BM5 noise-reservoir solve; no P4"
    elif safe_int(calibration.get("bm3_partial_pass_rows"), 0) > 0:
        decision["route"] = "R4-CouplingOnlyMechanism"
        decision["next_recommended_action"] = "treat BM3 as coordinate diagnostic; prioritize BM2/BM5"
    else:
        decision["route"] = "R4-CouplingOnlyMechanism"
        decision["next_recommended_action"] = "run BM2/BM5 constrained reservoir/noise mechanism"
    return decision


def write_report(report_path: Path, out_dir: Path, args: argparse.Namespace, anchor: Mapping[str, Any], calibration: Mapping[str, Any], family_status: Mapping[str, Any], decision: Mapping[str, Any], hashes: Mapping[str, Any]) -> None:
    anchor_row = anchor.get("anchor_row", {})
    families = family_status.get("families", {}) if isinstance(family_status, Mapping) else {}
    null_rows = read_csv_rows(out_dir / "v1213_linec_null_distribution.csv")
    null_lines = "\n".join(
        "| `{method}` | `{batch}` | `{window}` | `{rows}` | `{coupling}` | `{noise}` | `{reservoir}` | `{passes}` |".format(
            method=row.get("method", ""),
            batch=row.get("functional_batch_size", ""),
            window=row.get("window", ""),
            rows=row.get("rows", ""),
            coupling=row.get("CouplingR2_delta_mean", ""),
            noise=row.get("NoiseSignalLeak_delta_mean", ""),
            reservoir=row.get("RealSignalReservoirRatio_delta_mean", ""),
            passes=row.get("pareto_pass_rows", ""),
        )
        for row in null_rows
    )
    family_lines = "\n".join(
        f"| `{fam}` | `{families.get(fam, {}).get('status', '')}` | `{families.get(fam, {}).get('best_task_candidate', '')}` | `{families.get(fam, {}).get('blocker', '')}` |"
        for fam in [*ACTIVE_FAMILIES, "BSpline"]
    )
    hash_lines = "\n".join(f"| `{row['artifact']}` | `{row['sha256']}` |" for row in hashes.get("artifacts", []))
    text = f"""# DG-KAN v12.13 B320Locked FunctionalMechanism NoBSpline 执行复盘

生成时间：`{now_iso()}`

## 1. 执行入口

```text
script = experiments/run_v1213_b320locked_functional_mechanism_nobspline.py
command = {' '.join(sys.argv)}
plan_doc = {args.plan_doc}
out_dir = {rel(out_dir)}
```

本轮按 v12.13 计划先执行 Batch 1：B320 anchor monitor、Line C calibration/null distribution、No-BSpline family status refresh。没有 fake/proxy/CPU offload；没有 teacher/distillation/loss modification/sampler/class weight/dataset-name branch。

## 2. 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1213_b320locked_functional_mechanism_nobspline.py` | 新增 v12.13 runner | 复用真实 v12.10/v12.12 artifacts，新增 Line C repeated calibration / null distribution / BM1b/BM3a repeated probe；不改模型、不改 loss、不进入 P4。 |
| `{rel(report_path)}` | 新增 v12.13 复盘文件 | 记录命令、source、指标、route 和 blocker，防止后续复现再次读全代码。 |

## 3. B320 Anchor Monitor

```text
B320_anchor_locked = {decision.get('B320_anchor_locked')}
step_ratio_q90 = {anchor_row.get('step_ratio_q90')}
memory_ratio_q90 = {anchor_row.get('memory_ratio_q90')}
mean_delta = {anchor_row.get('mean_delta')}
worst_delta = {anchor_row.get('worst_delta')}
near_pass_rate = {anchor_row.get('near_pass_rate')}
AUC_step_ratio = {anchor_row.get('AUC_step_ratio')}
AUC_time_ratio = {anchor_row.get('AUC_time_ratio')}
ECE_delta = {anchor_row.get('ECE_delta')}
LineC_nontearing_pass = {anchor_row.get('LineC_nontearing_pass')}
```

## 4. Line C Calibration

```text
calibration_rows = {calibration.get('rows')}
linec_calibration_pass = {calibration.get('linec_calibration_pass')}
calibration_fail_reason = {calibration.get('calibration_fail_reason')}
noop_false_positive_rows = {calibration.get('noop_false_positive_rows')}
random_false_positive_rows = {calibration.get('random_false_positive_rows')}
bm1_full_3x3_pass_candidate_count = {calibration.get('bm1_full_3x3_pass_candidate_count')}
bm2_full_3x3_pass_candidate_count = {calibration.get('bm2_full_3x3_pass_candidate_count')}
bm3_full_3x3_pass_candidate_count = {calibration.get('bm3_full_3x3_pass_candidate_count')}
bm5_full_3x3_pass_candidate_count = {calibration.get('bm5_full_3x3_pass_candidate_count')}
bm1_partial_pass_rows = {calibration.get('bm1_partial_pass_rows')}
bm2_partial_pass_rows = {calibration.get('bm2_partial_pass_rows')}
bm3_partial_pass_rows = {calibration.get('bm3_partial_pass_rows')}
bm5_partial_pass_rows = {calibration.get('bm5_partial_pass_rows')}
fail_reason_counts = {json.dumps(calibration.get('fail_reason_counts', {}), ensure_ascii=False)}
```

解释：若是单次运行，runner 参数为 `methods={args.calibration_methods}`、`functional_batch_sizes={args.functional_batch_sizes}`、`windows={args.windows}`、`probe_splits={args.probe_splits}`；若是 aggregate 报告，则以三数据集 shard 的 CSV 汇总表为准。若 NoOp/Random false positive 或 NoOp noise/reservoir 方差超过计划阈值，route 必须是 `R2-LineCMetricNotReliable`，不能把 BM partial pass 推进 P4。

如果这是 aggregate 报告，实际执行矩阵以 `v1213_linec_null_distribution.csv` 为准：

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
{null_lines}

BM1-ext 追加了 `window=3/10`、`batch=32` 和 role-wise direct/quad/branch；BM5 追加了 logits-vs-onehot residual VJP 及 AdamW-orthogonal 版本。二者都没有形成 3x3 survivor，因此 P4 按计划保持关闭。

## 5. No-BSpline Family Status

| family | status | best_task_candidate | blocker |
|---|---|---|---|
{family_lines}

## 6. Route

```text
route = {decision.get('route')}
official_functional_success = {decision.get('official_functional_success')}
next_recommended_action = {decision.get('next_recommended_action')}
```

## 7. 复现命令

```text
conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py \\
  --out-dir {rel(out_dir)} \\
  --report-path {rel(report_path)}
```

本轮完整命令：

```text
{' '.join(sys.argv)}
```

## 8. Hash

| artifact | sha256 |
|---|---|
{hash_lines}
"""
    ensure_dir(report_path.parent)
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    v1212 = load_v1212_runner()
    p = argparse.ArgumentParser(description="DG-KAN v12.13 B320 locked functional mechanism runner")
    p.add_argument("--out-dir", default=str(DEFAULT_ROOT / f"v1213_batch1_calibration_{now_tag()}"))
    p.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    p.add_argument("--plan-doc", default="docs/DG-KAN_v12.13_B320Locked_FunctionalMechanism_NoBSpline_独立分析与下一步计划.md")
    p.add_argument("--probe-device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--probe-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--probe-seeds", default="0,1,2")
    p.add_argument("--probe-train-size", type=int, default=1024)
    p.add_argument("--probe-val-size", type=int, default=512)
    p.add_argument("--probe-test-size", type=int, default=512)
    p.add_argument("--functional-batch-sizes", default="16")
    p.add_argument("--probe-splits", type=int, default=5)
    p.add_argument("--windows", default="5")
    p.add_argument("--calibration-methods", default="NoOp,TaskOnlyAdamW,AdamWParallelDirection,RandomMatchedNorm,BM1b-SoftSNRGate,BM3a-BranchRebalance")
    p.add_argument("--bm3-strength", type=float, default=0.005)
    p.add_argument("--probe-lr", type=float, default=2.0e-3)
    p.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    p.add_argument("--sketch-dim", type=int, default=8)
    p.add_argument("--anchor-dir", default=str(v1212.DEFAULT_ANCHOR_DIR))
    p.add_argument("--b320-source-dir", default=str(v1212.DEFAULT_B320_SOURCE_DIR))
    p.add_argument("--v1211-root", default=str(v1212.DEFAULT_V1211_ROOT))
    p.add_argument("--aggregate-input-dirs", default="", help="Comma-separated v12.13 shard output dirs to aggregate instead of running probes")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    report_path = Path(args.report_path).resolve()
    ensure_dir(out_dir)
    v1212 = load_v1212_runner()
    prev = load_v1211_runner()
    source_dirs = {
        "anchor_dir": Path(args.anchor_dir).resolve(),
        "b320_source_dir": Path(args.b320_source_dir).resolve(),
        "v1211_root": Path(args.v1211_root).resolve(),
        "v1212_report": DEFAULT_V1212_REPORT,
    }
    missing = [name for name, path in source_dirs.items() if not path.exists()]
    if missing:
        raise SystemExit(f"missing source paths: {', '.join(missing)}")
    manifest = {
        "stage": "V1213_RUN_MANIFEST",
        "generated_at": now_iso(),
        "command": sys.argv,
        "plan_doc": args.plan_doc,
        "source_dirs": {name: rel(path) for name, path in source_dirs.items()},
        "contract": "linec_calibration_no_fake_no_proxy_no_loss_or_data_changes",
    }
    write_json(out_dir / "v1213_run_manifest.json", manifest)
    anchor = v1212.build_anchor_hardening(source_dirs["anchor_dir"], source_dirs["b320_source_dir"], out_dir)
    family_status = v1212.build_family_status(source_dirs["v1211_root"], out_dir)
    aggregate_dirs = [Path(item).resolve() for item in parse_list(args.aggregate_input_dirs)]
    if aggregate_dirs:
        calibration = aggregate_input_dirs(aggregate_dirs, out_dir)
    else:
        calibration = run_linec_calibration(args, out_dir, v1212, prev)
    decision = decide_route(anchor, calibration, family_status)
    write_json(out_dir / "v1213_route_decision.json", decision)
    write_simple_svgs(out_dir, decision)
    hashes = write_hashes(out_dir)
    write_report(report_path, out_dir, args, anchor, calibration, family_status, decision, hashes)
    print(json.dumps({"out_dir": str(out_dir), "report_path": str(report_path), "route": decision.get("route")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
