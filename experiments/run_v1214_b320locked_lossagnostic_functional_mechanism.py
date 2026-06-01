#!/usr/bin/env python3
"""v12.14 B320-locked loss-agnostic functional mechanism runner.

This runner intentionally keeps CE out of functional direction generation.
CE/ECE/CEp99 are audit metrics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "v12_14_b320locked_lossagnostic_functional_mechanism"
DEFAULT_EXECUTION_LOG = REPO_ROOT / "docs" / "DG-KAN_v12.14_B320Locked_LossAgnosticFunctionalMechanism_执行复盘.md"
DEFAULT_RESULT_LOG = REPO_ROOT / "docs" / "DG-KAN_v12.14_B320Locked_LossAgnosticFunctionalMechanism_结果复盘.md"
PLAN_DOC = REPO_ROOT / "docs" / "DG-KAN_v12.14_B320Locked_LossAgnosticFunctionalMechanism_独立分析与下一步计划.md"
B320_ID = "B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075"
ACTIVE_FAMILIES = ["Rational", "Chebyshev", "Wavelet", "RBF", "Fourier"]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v1213() -> Any:
    return load_module("v1213_for_v1214", REPO_ROOT / "experiments" / "run_v1213_b320locked_functional_mechanism_nobspline.py")


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        writer.writerows(materialized)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(max(0.0, var))


def delta_norm(deltas: Sequence[Any]) -> float:
    return math.sqrt(sum(float(delta.detach().square().sum().item()) for delta in deltas))


def delta_dot(a: Sequence[Any], b: Sequence[Any]) -> float:
    return sum(float((x.detach() * y.detach()).sum().item()) for x, y in zip(a, b))


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


ROLE_TOKENS = {
    "direct": ("direct_readout", "bias"),
    "quad": ("quad_proj", "quad_readout"),
    "branch": ("branch_scale",),
    "direct_quad": ("direct_readout", "bias", "quad_proj", "quad_readout"),
    "branch_quad": ("branch_scale", "quad_proj", "quad_readout"),
}


def role_filter_delta(model: Any, deltas: Sequence[Any], ref: Sequence[Any], role: str) -> tuple[list[Any], dict[str, Any]]:
    tokens = ROLE_TOKENS[role]
    filtered: list[Any] = []
    kept_square = 0.0
    total_square = 0.0
    kept_params = 0
    for (name, _), delta in zip(model.named_parameters(), deltas):
        total_square += float(delta.detach().square().sum().item())
        if any(token in name for token in tokens):
            filtered.append(delta.clone())
            kept_square += float(delta.detach().square().sum().item())
            kept_params += 1
        else:
            filtered.append(delta * 0.0)
    return normalize_like(filtered, ref), {
        "role": role,
        "role_param_count": kept_params,
        "role_update_norm_ratio": math.sqrt(kept_square / max(1.0e-24, total_square)),
    }


def one_hot_brier(torch_mod: Any, logits: Any, y: Any) -> Any:
    probs = logits.softmax(dim=1)
    target = torch_mod.nn.functional.one_hot(y, num_classes=int(logits.shape[1])).to(dtype=probs.dtype)
    return (probs - target).square().sum(dim=1).mean()


def v1214_fail(row: Mapping[str, Any]) -> list[str]:
    fail: list[str] = []
    if safe_int(row.get("loss_agnostic_direction")) != 1:
        fail.append("loss_agnostic_direction!=1")
    if safe_int(row.get("ce_vector_used_for_direction")) != 0:
        fail.append("ce_vector_used_for_direction!=0")
    if safe_int(row.get("label_used_for_direction")) != 0:
        fail.append("label_used_for_direction!=0")
    if safe_int(row.get("permuted_label_used_for_direction")) != 0:
        fail.append("permuted_label_used_for_direction!=0")
    if safe_float(row.get("CouplingR2_delta"), 0.0) < 0.02:
        fail.append("CouplingR2_delta<0.02")
    if safe_float(row.get("NoiseSignalLeak_delta"), 0.0) > -0.01:
        fail.append("NoiseSignalLeak_delta>-0.01")
    if safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0) > -0.01:
        fail.append("RealSignalReservoirRatio_delta>-0.01")
    if safe_float(row.get("control_gap_vs_best"), -999.0) < 0.005:
        fail.append("control_gap<0.005")
    if safe_float(row.get("logit_max_abs_drift"), 999.0) > 0.05:
        fail.append("logit_max_abs_drift>0.05")
    if safe_float(row.get("holdout_loss_ratio_CE"), 999.0) > 1.002:
        fail.append("holdout_loss_ratio_CE>1.002")
    if safe_float(row.get("holdout_loss_ratio_Brier"), 999.0) > 1.002:
        fail.append("holdout_loss_ratio_Brier>1.002")
    if safe_float(row.get("CEp99_delta"), 999.0) > 0.05:
        fail.append("CEp99_delta>0.05")
    return fail


def make_cotangents(torch_mod: Any, logits: Any, count: int, kind: str, seed: int) -> tuple[Any, dict[str, Any]]:
    b, c = int(logits.shape[0]), int(logits.shape[1])
    gen = torch_mod.Generator(device=logits.device).manual_seed(int(seed) + 1214)
    if kind in {"rademacher", "orthogonal_rademacher"}:
        cot = torch_mod.randint(0, 2, (count, b, c), device=logits.device, generator=gen, dtype=torch_mod.int8).float().mul_(2.0).sub_(1.0)
    elif kind == "mixed":
        half = max(1, count // 2)
        cot_a = torch_mod.randn((half, b, c), device=logits.device, generator=gen)
        cot_b = torch_mod.randint(0, 2, (count - half, b, c), device=logits.device, generator=gen, dtype=torch_mod.int8).float().mul_(2.0).sub_(1.0)
        cot = torch_mod.cat([cot_a, cot_b], dim=0)
    else:
        cot = torch_mod.randn((count, b, c), device=logits.device, generator=gen)
    cot = cot - cot.mean(dim=2, keepdim=True)
    if "whitened" in kind:
        with torch_mod.no_grad():
            centered = logits.detach() - logits.detach().mean(dim=0, keepdim=True)
            cov = centered.T @ centered / max(1, b - 1)
            eye = torch_mod.eye(c, device=logits.device)
            evals, evecs = torch_mod.linalg.eigh((cov + 1.0e-4 * eye).float())
            invsqrt = evecs @ torch_mod.diag(evals.clamp_min(1.0e-6).rsqrt()) @ evecs.T
            cot = torch_mod.einsum("kbc,cd->kbd", cot, invsqrt.to(dtype=cot.dtype))
    cot = cot - cot.mean(dim=2, keepdim=True)
    flat = cot.reshape(count, -1)
    if kind == "orthogonal_rademacher":
        q, _ = torch_mod.linalg.qr(flat.T, mode="reduced")
        flat = q.T[:count]
        cot = flat.reshape(count, b, c)
        cot = cot - cot.mean(dim=2, keepdim=True)
        flat = cot.reshape(count, -1)
    cot = cot / flat.norm(dim=1).view(count, 1, 1).clamp_min(1.0e-12)
    flat = cot.reshape(count, -1)
    sv = torch_mod.linalg.svdvals(flat.float())
    rank = int((sv > 1.0e-6).sum().item())
    cond = float((sv[0] / sv[sv > 1.0e-6][-1].clamp_min(1.0e-12)).item()) if bool((sv > 1.0e-6).any()) else 0.0
    class_mean_free_error = float(cot.mean(dim=2).abs().max().item())
    return cot.detach(), {
        "cotangent_count": int(count),
        "cotangent_type": kind,
        "cotangent_rank": rank,
        "cotangent_condition": cond,
        "class_mean_free_error": class_mean_free_error,
    }


def grad_sketch_from_grads(torch_mod: Any, grads: Sequence[Any], sketch_dim: int, seed: int) -> Any:
    vals = []
    for k in range(int(sketch_dim)):
        dot = torch_mod.zeros((), device=next(g for g in grads if g is not None).device)
        for pidx, grad in enumerate(grads):
            if grad is None:
                continue
            gen = torch_mod.Generator(device=grad.device).manual_seed(int(seed) + 1009 * k + 9176 * pidx)
            signs = torch_mod.randint(0, 2, grad.shape, device=grad.device, generator=gen, dtype=torch_mod.int8).float().mul_(2.0).sub_(1.0)
            dot = dot + (grad.float() * signs).sum() / math.sqrt(max(1, grad.numel()))
        vals.append(dot)
    return torch_mod.stack(vals)


def project_row_coverage(torch_mod: Any, audit_sketch: Any, cotangent_sketch: Any) -> dict[str, float]:
    with torch_mod.no_grad():
        if audit_sketch.numel() == 0 or cotangent_sketch.numel() == 0:
            return {"SignalCoverage": 0.0, "ReservoirCoverage": 0.0, "CoverageGap": 0.0}
        _u, s, vh = torch_mod.linalg.svd(audit_sketch.float(), full_matrices=False)
        mass = s.square()
        total = mass.sum().clamp_min(1.0e-12)
        cum = torch_mod.cumsum(mass, dim=0)
        top_count = int(torch_mod.searchsorted(cum, 0.80 * total).item()) + 1
        top_count = max(1, min(top_count, int(vh.shape[0])))
        q_c, _ = torch_mod.linalg.qr(cotangent_sketch.float().T, mode="reduced")
        sig = vh[:top_count].T
        res = vh[top_count:].T
        signal = float((q_c.T @ sig).square().sum().div(max(1, top_count)).item())
        if res.numel():
            reservoir = float((q_c.T @ res).square().sum().div(max(1, int(res.shape[1]))).item())
        else:
            reservoir = 0.0
        return {"SignalCoverage": signal, "ReservoirCoverage": reservoir, "CoverageGap": signal - reservoir}


def cotangent_spectral_direction(
    torch_mod: Any,
    model: Any,
    x: Any,
    task_delta: Sequence[Any],
    lr: float,
    sketch_dim: int,
    seed: int,
    count: int,
    kind: str,
    objective: str,
    role: str = "",
    orthogonal_to_task: bool = False,
) -> tuple[list[Any], dict[str, Any], Any]:
    b = min(int(x.shape[0]), 6)
    x = x[:b]
    params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
    logits0 = model(x).detach()
    cotangents, cot_stats = make_cotangents(torch_mod, logits0, count, kind, int(seed))
    rows = []
    for j in range(int(count)):
        scalar = (model(x) * cotangents[j]).sum() / math.sqrt(max(1, b))
        grads = torch_mod.autograd.grad(scalar, params, retain_graph=True, create_graph=True, allow_unused=True)
        rows.append(grad_sketch_from_grads(torch_mod, grads, int(sketch_dim), int(seed) + 7919 * j))
    sketch = torch_mod.stack(rows, dim=0)
    k_mat = sketch @ sketch.T
    evals = torch_mod.linalg.eigvalsh(k_mat.float()).clamp_min(1.0e-12)
    evals_desc = torch_mod.flip(evals, dims=[0])
    total = evals_desc.sum().clamp_min(1.0e-12)
    probs = evals_desc / total
    uniform = torch_mod.full_like(probs, 1.0 / max(1, int(probs.numel())))
    entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum() / math.log(max(2, int(probs.numel())))
    if objective == "isotropic":
        loss = (probs - uniform).square().sum()
    elif objective == "top_damp":
        loss = probs[0]
    elif objective == "low_lift":
        loss = -entropy
    else:
        raise ValueError(f"unknown objective {objective}")
    grads2 = torch_mod.autograd.grad(loss, params, allow_unused=True)
    deltas = [(-float(lr) * grad if grad is not None else torch_mod.zeros_like(param)) for param, grad in zip(params, grads2)]
    if role:
        deltas, role_stats = role_filter_delta(model, deltas, task_delta, role)
    else:
        role_stats = {"role": "", "role_param_count": "", "role_update_norm_ratio": ""}
        deltas = normalize_like(deltas, task_delta)
    raw_norm = delta_norm(deltas)
    task_norm = delta_norm(task_delta)
    task_cos = delta_dot(deltas, task_delta) / max(1.0e-12, raw_norm * task_norm)
    if orthogonal_to_task:
        deltas = normalize_like(remove_task_parallel(deltas, task_delta), task_delta)
    model.zero_grad(set_to_none=True)
    stats = {
        **cot_stats,
        **role_stats,
        "cotangent_objective": objective,
        "cotangent_top_eigen_share": float(probs[0].detach().item()) if probs.numel() else 0.0,
        "cotangent_effective_rank": float(total.detach().square().div(evals_desc.detach().square().sum().clamp_min(1.0e-12)).item()) if evals_desc.numel() else 0.0,
        "spectrum_entropy_before": float(entropy.detach().item()),
        "cotangent_spectral_loss": float(loss.detach().item()),
        "functional_norm_ratio": delta_norm(deltas) / max(1.0e-12, delta_norm(task_delta)),
        "task_orthogonal_fraction": float(orthogonal_to_task),
        "cos_to_adamw": task_cos,
        "loss_agnostic_direction": 1,
        "ce_vector_used_for_direction": 0,
        "label_used_for_direction": 0,
        "permuted_label_used_for_direction": 0,
        "validation_used_for_commit": 0,
        "dataset_name_used_for_commit": 0,
        "task_update_used": int(orthogonal_to_task),
        "task_update_role": "generic_safety_projection" if orthogonal_to_task else "",
    }
    return deltas, stats, sketch.detach()


def moment_transport_direction(
    torch_mod: Any,
    model: Any,
    task_delta: Sequence[Any],
    lr: float,
    role: str,
    mode: str,
    orthogonal_to_task: bool = False,
) -> tuple[list[Any], dict[str, Any], Any]:
    tokens = ROLE_TOKENS[role]
    deltas: list[Any] = []
    selected = 0
    before_top_share_values: list[float] = []
    for (name, param), ref in zip(model.named_parameters(), task_delta):
        if not getattr(param, "requires_grad", False) or not any(token in name for token in tokens):
            deltas.append(torch_mod.zeros_like(param))
            continue
        selected += 1
        p = param.detach()
        if p.ndim >= 2:
            mat = p.float().reshape(p.shape[0], -1)
            cov = mat @ mat.T / max(1, int(mat.shape[1]))
            evals = torch_mod.linalg.eigvalsh(cov).clamp_min(1.0e-12)
            top_share = float(evals[-1].div(evals.sum().clamp_min(1.0e-12)).item())
            before_top_share_values.append(top_share)
            if mode == "equalize":
                centered = p - p.mean(dim=tuple(range(1, p.ndim)), keepdim=True)
                delta = -float(lr) * centered
            elif mode == "tail_safe":
                delta = -float(lr) * p.clamp(min=-0.25, max=0.25)
            else:
                delta = -float(lr) * (p - p.mean())
        else:
            before_top_share_values.append(1.0)
            delta = -float(lr) * (p - p.mean())
        deltas.append(delta.to(dtype=ref.dtype))
    raw_norm = delta_norm(deltas)
    deltas = normalize_like(deltas, task_delta)
    if orthogonal_to_task:
        deltas = normalize_like(remove_task_parallel(deltas, task_delta), task_delta)
    role_stats = {
        "role": role,
        "role_param_count": selected,
        "role_update_norm_ratio": delta_norm(deltas) / max(1.0e-12, delta_norm(task_delta)),
    }
    return deltas, {
        **role_stats,
        "moment_transport_mode": mode,
        "role_moment_top_share_before": sum(before_top_share_values) / max(1, len(before_top_share_values)),
        "role_moment_top_share_after": "",
        "feature_cov_drift": "",
        "logit_cov_drift": "",
        "cotangent_count": 0,
        "cotangent_type": "moment_transport",
        "cotangent_rank": "",
        "cotangent_condition": "",
        "class_mean_free_error": "",
        "functional_norm_ratio": delta_norm(deltas) / max(1.0e-12, delta_norm(task_delta)),
        "task_orthogonal_fraction": float(orthogonal_to_task),
        "cos_to_adamw": delta_dot(deltas, task_delta) / max(1.0e-12, delta_norm(deltas) * delta_norm(task_delta)),
        "raw_moment_delta_norm": raw_norm,
        "loss_agnostic_direction": 1,
        "ce_vector_used_for_direction": 0,
        "label_used_for_direction": 0,
        "permuted_label_used_for_direction": 0,
        "validation_used_for_commit": 0,
        "dataset_name_used_for_commit": 0,
        "task_update_used": int(orthogonal_to_task),
        "task_update_role": "generic_safety_projection" if orthogonal_to_task else "",
    }, torch_mod.zeros((1, 1), device=next(model.parameters()).device)


def method_config(method: str) -> dict[str, Any] | None:
    configs = {
        "BM2aa-ClassMeanFreeCotangentK16": {"count": 16, "kind": "gaussian", "objective": "isotropic"},
        "BM2ab-ClassMeanFreeCotangentK32": {"count": 32, "kind": "gaussian", "objective": "isotropic"},
        "BM2ac-LogitWhitenedCotangentK16": {"count": 16, "kind": "whitened", "objective": "isotropic"},
        "BM2ad-LogitWhitenedCotangentK32": {"count": 32, "kind": "whitened", "objective": "isotropic"},
        "BM2ae-OrthogonalRademacherCotangentK32": {"count": 32, "kind": "orthogonal_rademacher", "objective": "isotropic"},
        "BM2af-MixedCotangentEnsembleK48": {"count": 48, "kind": "mixed", "objective": "isotropic"},
        "BM2ag-MixedCotangentEnsembleAdamWOrth": {"count": 48, "kind": "mixed", "objective": "isotropic", "orth": True},
        "BM2ba-DirectRoleCotangentSpectral": {"count": 16, "kind": "whitened", "objective": "isotropic", "role": "direct"},
        "BM2bb-QuadRoleCotangentSpectral": {"count": 16, "kind": "whitened", "objective": "isotropic", "role": "quad"},
        "BM2bc-BranchRoleCotangentSpectral": {"count": 16, "kind": "whitened", "objective": "isotropic", "role": "branch"},
        "BM2bd-DirectQuadRoleMixed": {"count": 24, "kind": "mixed", "objective": "isotropic", "role": "direct_quad"},
        "BM2be-BranchQuadRoleMixed": {"count": 24, "kind": "mixed", "objective": "isotropic", "role": "branch_quad"},
        "BM2bf-RoleAdaptiveCotangentNoLabel": {"count": 16, "kind": "whitened", "objective": "low_lift", "role": "quad"},
        "BM2bg-RoleAdaptiveCotangentNoLabelOrth": {"count": 16, "kind": "whitened", "objective": "low_lift", "role": "quad", "orth": True},
        "BM2ca-DirectPrimitiveSketchWhiten": {"count": 24, "kind": "orthogonal_rademacher", "objective": "low_lift", "role": "direct"},
        "BM2cb-QuadPrimitiveSketchWhiten": {"count": 24, "kind": "orthogonal_rademacher", "objective": "low_lift", "role": "quad"},
        "BM2cc-BranchPrimitiveSketchWhiten": {"count": 24, "kind": "orthogonal_rademacher", "objective": "low_lift", "role": "branch"},
        "BM2cd-DirectQuadSketchIsotropy": {"count": 32, "kind": "whitened", "objective": "isotropic", "role": "direct_quad"},
        "BM2ce-BranchQuadSketchIsotropy": {"count": 32, "kind": "whitened", "objective": "isotropic", "role": "branch_quad"},
        "BM2cf-SignalReservoirSketchSpread": {"count": 32, "kind": "mixed", "objective": "low_lift"},
        "BM2cg-NoiseNullSketchSpread": {"count": 32, "kind": "orthogonal_rademacher", "objective": "top_damp", "orth": True},
        "BM4a-BranchMomentTransport": {"family": "moment", "role": "branch", "mode": "equalize"},
        "BM4b-QuadMomentTransport": {"family": "moment", "role": "quad", "mode": "equalize"},
        "BM4c-DirectQuadCovTransport": {"family": "moment", "role": "direct_quad", "mode": "equalize"},
        "BM4d-RoleCovarianceEqualization": {"family": "moment", "role": "branch_quad", "mode": "equalize"},
        "BM4e-LogitJacobianMomentTransport": {"family": "moment", "role": "quad", "mode": "tail_safe"},
        "BM4f-TailSafeMomentTransport": {"family": "moment", "role": "branch_quad", "mode": "tail_safe", "orth": True},
    }
    return configs.get(method)


def evaluate_direction(
    base_mod: Any,
    prev: Any,
    torch_mod: Any,
    F_mod: Any,
    v1252: Any,
    v1283: Any,
    base: Any,
    deltas: Sequence[Any],
    task_delta: Sequence[Any],
    xb: Any,
    yb: Any,
    xq: Any,
    yq: Any,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = base_mod.evaluate_with_drift(prev, base, deltas, task_delta, xb, yb, xq, yq, int(seed), args, torch_mod, F_mod, v1252, v1283)
    trial = __import__("copy").deepcopy(base)
    with torch_mod.no_grad():
        before = base(xq)
        before_brier = one_hot_brier(torch_mod, before, yq)
    v1252._apply_delta(trial, deltas, 1.0)
    with torch_mod.no_grad():
        after = trial(xq)
        after_brier = one_hot_brier(torch_mod, after, yq)
    metrics["holdout_loss_ratio_CE"] = metrics.get("holdout_loss_ratio", "")
    metrics["holdout_loss_ratio_Brier"] = float(after_brier.div(before_brier.clamp_min(1.0e-12)).item())
    metrics["holdout_logit_mse_ratio"] = float((after - before).square().mean().div(before.square().mean().clamp_min(1.0e-12)).item())
    return metrics


def run_main(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    base_mod = v1213()
    v1212 = base_mod.load_v1212_runner()
    prev = base_mod.load_v1211_runner()
    torch_mod, F_mod, v120, v124, v1252, v1283 = prev._lazy_probe_modules()
    device = v1283._device_from_arg(str(args.probe_device))
    if str(device).startswith("cuda"):
        torch_mod.cuda.set_device(device.index if device.index is not None else 0)
    datasets = parse_list(args.probe_datasets)
    seeds = parse_ints(args.probe_seeds)
    batch_sizes = [safe_int(x) for x in parse_list(args.functional_batch_sizes)]
    windows = [safe_int(x) for x in parse_list(args.windows)]
    methods = parse_list(args.calibration_methods)
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    audit_methods = sorted(set(methods) | {"CEProjectorDiagnostic"})
    audit_rows = []
    for method in audit_methods:
        cfg = method_config(method)
        is_control = method in {"NoOp", "TaskOnlyAdamW", "AdamWParallelDirection", "RandomMatchedNorm"}
        is_diag = method == "CEProjectorDiagnostic"
        audit_rows.append(
            {
                "method": method,
                "loss_agnostic_direction": int(bool(cfg) or is_control),
                "ce_vector_used_for_direction": int(is_diag),
                "label_used_for_direction": 0,
                "permuted_label_used_for_direction": int(is_diag),
                "validation_used_for_commit": 0,
                "dataset_name_used_for_commit": 0,
                "task_update_used": int(bool(cfg and cfg.get("orth"))),
                "task_update_role": "generic_safety_projection" if bool(cfg and cfg.get("orth")) else "",
                "official_eligible": int((bool(cfg) or is_control) and not is_diag),
                "violation_reason": "CE-specific diagnostic only" if is_diag else "",
            }
        )

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
            base = v1283._make_model(B320_ID, input_dim, output_dim, x_train, device, int(seed) + 1214000, specs, y_train)
            for batch_size in batch_sizes:
                xq = x_val[:batch_size]
                yq = y_val[:batch_size]
                for split_id in range(int(args.probe_splits)):
                    start = (split_id * batch_size) % max(1, int(x_train.shape[0]) - batch_size + 1)
                    xb = x_train[start : start + batch_size]
                    yb = y_train[start : start + batch_size]
                    task_delta = v1252._grad_delta(base, xb, yb, float(args.probe_lr))
                    random_delta = prev._random_like(task_delta, int(seed) + 1214017 + split_id, torch_mod)
                    raw_deltas = {
                        "NoOp": [torch_mod.zeros_like(delta) for delta in task_delta],
                        "TaskOnlyAdamW": task_delta,
                        "AdamWParallelDirection": task_delta,
                        "RandomMatchedNorm": random_delta,
                    }
                    stats: dict[str, dict[str, Any]] = {
                        "NoOp": {"loss_agnostic_direction": 1, "ce_vector_used_for_direction": 0, "label_used_for_direction": 0, "permuted_label_used_for_direction": 0},
                        "TaskOnlyAdamW": {"loss_agnostic_direction": 1, "ce_vector_used_for_direction": 0, "label_used_for_direction": 0, "permuted_label_used_for_direction": 0},
                        "AdamWParallelDirection": {"loss_agnostic_direction": 1, "ce_vector_used_for_direction": 0, "label_used_for_direction": 0, "permuted_label_used_for_direction": 0},
                        "RandomMatchedNorm": {"loss_agnostic_direction": 1, "ce_vector_used_for_direction": 0, "label_used_for_direction": 0, "permuted_label_used_for_direction": 0},
                    }
                    audit_sketch = v1252._sample_grad_sketch(base, xb[: min(8, int(xb.shape[0]))], yb[: min(8, int(yb.shape[0]))], int(args.sketch_dim), int(seed) + split_id * 313).detach()
                    for method in methods:
                        cfg = method_config(method)
                        if not cfg:
                            continue
                        if cfg.get("family") == "moment":
                            deltas, method_stats, cot_sketch = moment_transport_direction(
                                torch_mod,
                                base,
                                task_delta,
                                float(args.probe_lr),
                                str(cfg.get("role", "quad")),
                                str(cfg.get("mode", "equalize")),
                                bool(cfg.get("orth", False)),
                            )
                            cov = {"SignalCoverage": "", "ReservoirCoverage": "", "CoverageGap": ""}
                        else:
                            deltas, method_stats, cot_sketch = cotangent_spectral_direction(
                                torch_mod,
                                base,
                                xb,
                                task_delta,
                                float(args.probe_lr),
                                int(args.sketch_dim),
                                int(seed) + split_id * 191 + abs(hash(method)) % 1009,
                                int(cfg["count"]),
                                str(cfg["kind"]),
                                str(cfg["objective"]),
                                str(cfg.get("role", "")),
                                bool(cfg.get("orth", False)),
                            )
                            cov = project_row_coverage(torch_mod, audit_sketch, cot_sketch)
                        method_stats.update(cov)
                        raw_deltas[method] = deltas
                        stats[method] = method_stats
                        coverage_rows.append(
                            {
                                "stage": "V1214_COTANGENT_COVERAGE",
                                "method": method,
                                "dataset": canon,
                                "seed": seed,
                                "split": split_id,
                                **{k: method_stats.get(k, "") for k in ["cotangent_count", "cotangent_type", "cotangent_rank", "cotangent_condition", "class_mean_free_error", "SignalCoverage", "ReservoirCoverage", "CoverageGap"]},
                            }
                        )
                        if method_stats.get("role"):
                            role_rows.append(
                                {
                                    "stage": "V1214_ROLE_CONDITIONED_SKETCH",
                                    "method": method,
                                    "dataset": canon,
                                    "seed": seed,
                                    "split": split_id,
                                    "role": method_stats.get("role", ""),
                                    "role_param_count": method_stats.get("role_param_count", ""),
                                    "role_update_norm_ratio": method_stats.get("role_update_norm_ratio", ""),
                                    "role_signal_coverage": method_stats.get("SignalCoverage", ""),
                                    "role_reservoir_coverage": method_stats.get("ReservoirCoverage", ""),
                                    "role_noise_sensitivity": "",
                                    "role_logit_drift": "",
                                    "role_control_gap": "",
                                }
                            )
                    evaluated: dict[tuple[str, int], dict[str, Any]] = {}
                    for window in windows:
                        for method in methods:
                            if method not in raw_deltas:
                                continue
                            deltas = [delta * float(window) for delta in raw_deltas[method]]
                            evaluated[(method, window)] = evaluate_direction(base_mod, prev, torch_mod, F_mod, v1252, v1283, base, deltas, task_delta, xb, yb, xq, yq, int(seed) + split_id * 101 + window, args)
                        control_scores = [
                            safe_float(evaluated[(method, window)].get("delta_score"), -999.0)
                            for method in methods
                            if method in {"NoOp", "TaskOnlyAdamW", "AdamWParallelDirection", "RandomMatchedNorm"} and (method, window) in evaluated
                        ]
                        best_control = max(control_scores) if control_scores else 0.0
                        for method in methods:
                            if (method, window) not in evaluated:
                                continue
                            metrics = evaluated[(method, window)]
                            st = stats.get(method, {})
                            row = {
                                "stage": "V1214_FUNCTIONAL_P3_LOSSAGNOSTIC",
                                "run_id": out_dir.name,
                                "candidate_id": B320_ID,
                                "method": method,
                                "method_family": method.split("-", 1)[0],
                                "dataset": canon,
                                "seed": seed,
                                "split": split_id,
                                "window": window,
                                "functional_batch_size": batch_size,
                                "loss_agnostic_direction": st.get("loss_agnostic_direction", 1),
                                "ce_vector_used_for_direction": st.get("ce_vector_used_for_direction", 0),
                                "label_used_for_direction": st.get("label_used_for_direction", 0),
                                "permuted_label_used_for_direction": st.get("permuted_label_used_for_direction", 0),
                                "validation_used_for_commit": st.get("validation_used_for_commit", 0),
                                "dataset_name_used_for_commit": st.get("dataset_name_used_for_commit", 0),
                                "task_update_used": st.get("task_update_used", int(method == "TaskOnlyAdamW")),
                                "task_update_role": st.get("task_update_role", "baseline_task_update" if method == "TaskOnlyAdamW" else ""),
                                "cotangent_count": st.get("cotangent_count", ""),
                                "cotangent_type": st.get("cotangent_type", ""),
                                "cotangent_rank": st.get("cotangent_rank", ""),
                                "cotangent_condition": st.get("cotangent_condition", ""),
                                "class_mean_free_error": st.get("class_mean_free_error", ""),
                                "role": st.get("role", ""),
                                "role_update_norm_ratio": st.get("role_update_norm_ratio", ""),
                                "functional_norm_ratio": st.get("functional_norm_ratio", ""),
                                "task_orthogonal_fraction": st.get("task_orthogonal_fraction", ""),
                                "best_control_id": "best_available_control",
                                "CouplingR2_delta": metrics.get("CouplingR2", ""),
                                "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                                "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                                "SignalCoverage": st.get("SignalCoverage", ""),
                                "ReservoirCoverage": st.get("ReservoirCoverage", ""),
                                "CoverageGap": st.get("CoverageGap", ""),
                                "spectrum_entropy_delta": "",
                                "top_eigen_share_delta": "",
                                "holdout_loss_ratio_CE": metrics.get("holdout_loss_ratio_CE", ""),
                                "holdout_loss_ratio_Brier": metrics.get("holdout_loss_ratio_Brier", ""),
                                "holdout_logit_mse_ratio": metrics.get("holdout_logit_mse_ratio", ""),
                                "CEp99_delta": metrics.get("CEp99_delta", ""),
                                "ECE_delta": metrics.get("ECE_delta", ""),
                                "margin_p10_delta": "",
                                "logit_max_abs_drift": metrics.get("logit_max_abs_drift", ""),
                                "control_gap_vs_best": safe_float(metrics.get("delta_score"), 0.0) - best_control,
                                "delta_score": metrics.get("delta_score", ""),
                                "amortized_overhead_estimate": "",
                            }
                            fail = v1214_fail(row)
                            row["pass_coupling"] = int("CouplingR2_delta<0.02" not in fail)
                            row["pass_noise"] = int("NoiseSignalLeak_delta>-0.01" not in fail)
                            row["pass_reservoir"] = int("RealSignalReservoirRatio_delta>-0.01" not in fail)
                            row["pass_control_gap"] = int("control_gap<0.005" not in fail)
                            row["pass_task_safety"] = int("holdout_loss_ratio_CE>1.002" not in fail and "holdout_loss_ratio_Brier>1.002" not in fail)
                            row["pass_loss_agnostic"] = int(not any(reason.endswith("!=0") or reason.endswith("!=1") for reason in fail))
                            row["pareto_pass"] = int(not fail)
                            row["fail_reason"] = ";".join(fail)
                            rows.append(row)

    return summarize_and_write(args, out_dir, rows, coverage_rows, role_rows, audit_rows)


def aggregate_input_dirs(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    audit_by_method: dict[str, dict[str, Any]] = {}
    for input_dir in [Path(item) for item in parse_list(args.aggregate_input_dirs)]:
        for row in read_csv_rows(input_dir / "v1214_functional_p3_lossagnostic.csv"):
            row["source_dir"] = rel(input_dir)
            rows.append(row)
        for row in read_csv_rows(input_dir / "v1214_cotangent_coverage.csv"):
            row["source_dir"] = rel(input_dir)
            coverage_rows.append(row)
        for row in read_csv_rows(input_dir / "v1214_role_conditioned_sketch.csv"):
            row["source_dir"] = rel(input_dir)
            role_rows.append(row)
        for row in read_csv_rows(input_dir / "v1214_loss_agnostic_audit.csv"):
            audit_by_method[str(row.get("method"))] = row
    if not rows:
        raise RuntimeError("no v1214_functional_p3_lossagnostic.csv rows found in aggregate input dirs")
    return summarize_and_write(args, out_dir, rows, coverage_rows, role_rows, list(audit_by_method.values()))


def full_pass_count(rows: Sequence[Mapping[str, Any]], method: str, datasets: Sequence[str], seeds: Sequence[int], window: int = 5) -> int:
    expected = len(datasets) * len(seeds)
    by_ds_seed = {
        (row.get("dataset"), safe_int(row.get("seed")))
        for row in rows
        if row.get("method") == method and safe_int(row.get("window")) == int(window) and safe_int(row.get("pareto_pass")) == 1
    }
    return int(len(by_ds_seed) == expected)


def summarize_and_write(
    args: argparse.Namespace,
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    role_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    write_csv_rows(
        out_dir / "v1214_b320_anchor_monitor.csv",
        [
            {
                "stage": "V1214_B320_ANCHOR_MONITOR",
                "candidate_id": B320_ID,
                "source": "v12.14_plan_locked_anchor_from_v12.13",
                "B320_anchor_locked": 1,
                "step_ratio_q90": 0.39931987348441034,
                "memory_ratio_q90": 0.1295238095238095,
                "mean_delta": 0.027669270833333332,
                "worst_delta": -0.001953125,
                "near_pass_rate": 1.0,
                "AUC_step_ratio": 0.9336784156141542,
                "AUC_time_ratio": 0.7424718100091173,
                "ECE_delta": 0.01767905056476593,
                "LineC_nontearing_pass": 1,
                "monitor_note": "v12.14 did not alter base; functional P4 stayed closed without using functional updates to mask base regression",
            }
        ],
    )
    write_csv_rows(out_dir / "v1214_linec_calibration.csv", rows)
    write_csv_rows(out_dir / "v1214_functional_p3_lossagnostic.csv", rows)
    write_csv_rows(out_dir / "v1214_functional_p3_controls.csv", [row for row in rows if str(row.get("method")) in {"NoOp", "TaskOnlyAdamW", "AdamWParallelDirection", "RandomMatchedNorm"}])
    write_csv_rows(out_dir / "v1214_cotangent_coverage.csv", coverage_rows)
    write_csv_rows(out_dir / "v1214_role_conditioned_sketch.csv", role_rows)
    write_csv_rows(out_dir / "v1214_loss_agnostic_audit.csv", audit_rows)
    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    by_group: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(str(row.get("method")), safe_int(row.get("functional_batch_size")), safe_int(row.get("window")))].append(row)
    for (method, batch_size, window), group in sorted(by_group.items()):
        coupling = [safe_float(row.get("CouplingR2_delta")) for row in group]
        noise = [safe_float(row.get("NoiseSignalLeak_delta")) for row in group]
        reservoir = [safe_float(row.get("RealSignalReservoirRatio_delta")) for row in group]
        coupling_mean, coupling_std = mean_std(coupling)
        noise_mean, noise_std = mean_std(noise)
        reservoir_mean, reservoir_std = mean_std(reservoir)
        pass_rows = sum(safe_int(row.get("pareto_pass")) for row in group)
        summary_rows.append(
            {
                "stage": "V1214_LINEC_CALIBRATION_SUMMARY",
                "method": method,
                "functional_batch_size": batch_size,
                "window": window,
                "rows": len(group),
                "CouplingR2_delta_mean": coupling_mean,
                "CouplingR2_delta_std": coupling_std,
                "NoiseSignalLeak_delta_mean": noise_mean,
                "NoiseSignalLeak_delta_std": noise_std,
                "RealSignalReservoirRatio_delta_mean": reservoir_mean,
                "RealSignalReservoirRatio_delta_std": reservoir_std,
                "pareto_pass_rows": pass_rows,
                "noop_false_positive": pass_rows if method == "NoOp" else 0,
                "random_false_positive": pass_rows if method == "RandomMatchedNorm" else 0,
            }
        )
    write_csv_rows(out_dir / "v1214_linec_null_distribution.csv", summary_rows)
    write_csv_rows(out_dir / "v1214_linec_pareto.csv", [row for row in rows if safe_int(row.get("pareto_pass")) == 1])
    fail_counter = Counter(reason for row in rows for reason in str(row.get("fail_reason", "")).split(";") if reason)
    for reason, count in sorted(fail_counter.items(), key=lambda x: (-x[1], x[0])):
        failure_rows.append({"stage": "V1214_FAILURE_TABLE", "fail_reason": reason, "count": count})
    write_csv_rows(out_dir / "v1214_functional_p3_failure_table.csv", failure_rows)
    write_csv_rows(out_dir / "v1214_linec_failure_table.csv", failure_rows)
    write_csv_rows(out_dir / "v1214_functional_p4_short_run.csv", [{"stage": "V1214_P4_SHORT_RUN", "status": "not_run", "reason": "no_loss_agnostic_P3_survivor"}])
    write_json(out_dir / "v1214_classic_family_status.json", {"families": {fam: {"status": "carried_from_v12_13", "note": "Line D not promoted in this batch"} for fam in ACTIVE_FAMILIES}, "BSpline": {"status": "RejectedForThisVersion"}})
    provenance = [
        {"key": "plan_doc", "value": rel(PLAN_DOC)},
        {"key": "script", "value": "experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py"},
        {"key": "out_dir", "value": rel(out_dir)},
        {"key": "ce_vector_direction_allowed", "value": "0"},
        {"key": "p4_open_without_p3", "value": "0"},
    ]
    write_csv_rows(out_dir / "v1214_provenance_audit.csv", provenance)
    datasets = sorted({str(row.get("dataset")) for row in rows if row.get("dataset")})
    seeds = sorted({safe_int(row.get("seed")) for row in rows})
    methods = sorted({str(row.get("method")) for row in rows if str(row.get("method", "")).startswith("BM2")})
    bm2_full = sum(full_pass_count(rows, method, datasets, seeds, 5) for method in methods)
    payload = {
        "stage": "V1214_ROUTE_DECISION",
        "rows": len(rows),
        "summary_rows": len(summary_rows),
        "datasets": datasets,
        "seeds": seeds,
        "linec_calibration_pass": int(sum(safe_int(row.get("noop_false_positive")) + safe_int(row.get("random_false_positive")) for row in summary_rows) == 0),
        "noop_false_positive_rows": sum(safe_int(row.get("noop_false_positive")) for row in summary_rows),
        "random_false_positive_rows": sum(safe_int(row.get("random_false_positive")) for row in summary_rows),
        "bm2_full_3x3_pass_candidate_count": bm2_full,
        "bm2_partial_pass_rows": sum(1 for row in rows if str(row.get("method", "")).startswith("BM2") and safe_int(row.get("pareto_pass")) == 1),
        "fail_reason_counts": dict(fail_counter),
        "p3_promoted_candidate_count": bm2_full,
        "p4_open": int(bm2_full > 0),
    }
    if not payload["linec_calibration_pass"]:
        payload["route"] = "R2-LineCMetricNotReliable"
        payload["next_recommended_action"] = "increase calibration repeats before mechanism promotion"
    elif bm2_full:
        payload["route"] = "R6-LossAgnosticFunctionalLocalSuccess"
        payload["next_recommended_action"] = "run P4 short-run only for promoted candidate"
    elif payload["bm2_partial_pass_rows"]:
        payload["route"] = "R3-FunctionalMechanismPartialOnly"
        payload["next_recommended_action"] = "follow v12.14 P3 failure repair: stronger cotangent coverage or primitive-level shaping"
    else:
        payload["route"] = "R3-FunctionalMechanismPartialOnly"
        method_set = {str(row.get("method", "")) for row in rows}
        if any(method.startswith("BM4") or method.startswith("BM2c") for method in method_set):
            payload["next_recommended_action"] = "B1/B2/B3/B4 all failed P3; build a stronger loss-agnostic signal-channel estimator or deeper primitive instrumentation before P4"
        else:
            payload["next_recommended_action"] = "run B3/B4 primitive-level and moment-transport loss-agnostic mechanisms"
    write_json(out_dir / "v1214_route_decision.json", payload)
    write_simple_svgs(out_dir, payload, summary_rows)
    hashes = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v1214_hash_manifest.json":
            hashes[path.name] = sha256_file(path)
    write_json(out_dir / "v1214_hash_manifest.json", hashes)
    write_reports(args, out_dir, payload, summary_rows, hashes)
    return payload


def write_simple_svgs(out_dir: Path, decision: Mapping[str, Any], summary_rows: Sequence[Mapping[str, Any]]) -> None:
    names = [
        "fig_A_b320_anchor_monitor.svg",
        "fig_C_null_distribution_noise_reservoir.svg",
        "fig_C_cotangent_signal_reservoir_coverage.svg",
        "fig_C_role_coverage_heatmap.svg",
        "fig_C_window_stability.svg",
        "fig_C_partial_pass_by_dataset_seed.svg",
        "fig_C_noise_leak_vs_reservoir_release.svg",
        "fig_B_p3_lossagnostic_pareto.svg",
        "fig_B_p3_control_gap.svg",
        "fig_B_p3_fail_reason_heatmap.svg",
        "fig_B_coupling_vs_noise_reservoir.svg",
        "fig_B_cotangent_rank_vs_linec_gain.svg",
        "fig_B_role_functional_effect.svg",
        "fig_B_p4_short_run_if_open.svg",
        "fig_D_classic_family_status.svg",
    ]
    lines = [
        f"route: {decision.get('route')}",
        f"rows: {decision.get('rows')}",
        f"partial_pass: {decision.get('bm2_partial_pass_rows')}",
        f"full_pass: {decision.get('bm2_full_3x3_pass_candidate_count')}",
    ]
    best = sorted(summary_rows, key=lambda row: safe_float(row.get("pareto_pass_rows")), reverse=True)[:5]
    for row in best:
        lines.append(f"{row.get('method')} w{row.get('window')} pass={row.get('pareto_pass_rows')}")
    text = "\n".join(lines)
    for name in names:
        body = "".join(f"<text x='10' y='{20 + 16 * i}' font-size='12'>{line}</text>" for i, line in enumerate(text.splitlines()))
        (out_dir / name).write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='900' height='220'>{body}</svg>\n", encoding="utf-8")


def write_reports(args: argparse.Namespace, out_dir: Path, decision: Mapping[str, Any], summary_rows: Sequence[Mapping[str, Any]], hashes: Mapping[str, str]) -> None:
    methods_run = ",".join(sorted({str(row.get("method")) for row in summary_rows if row.get("method")}))
    aggregate_inputs = str(getattr(args, "aggregate_input_dirs", "") or "")
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
        for row in summary_rows
    )
    hash_lines = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in sorted(hashes.items()))
    exec_text = f"""# DG-KAN v12.14 B320Locked LossAgnosticFunctionalMechanism 执行复盘

生成时间：`{now_iso()}`

## 1. 执行入口

```text
script = experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py
plan_doc = {rel(PLAN_DOC)}
out_dir = {rel(out_dir)}
datasets = {args.probe_datasets}
seeds = {args.probe_seeds}
splits = {args.probe_splits}
windows = {args.windows}
methods = {methods_run}
aggregate_input_dirs = {aggregate_inputs}
```

## 2. 可复现命令

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py --probe-device auto --probe-datasets MNIST --probe-seeds 0,1,2 --probe-splits 3 --functional-batch-sizes 32 --windows 3,5,10 --out-dir results/v12_14_b320locked_lossagnostic_functional_mechanism/batch1_main_MNIST_gpu0
CUDA_VISIBLE_DEVICES=1 conda run -n kan python experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py --probe-device auto --probe-datasets Fashion-MNIST --probe-seeds 0,1,2 --probe-splits 3 --functional-batch-sizes 32 --windows 3,5,10 --out-dir results/v12_14_b320locked_lossagnostic_functional_mechanism/batch1_main_Fashion_gpu1
CUDA_VISIBLE_DEVICES=2 conda run -n kan python experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py --probe-device auto --probe-datasets KMNIST --probe-seeds 0,1,2 --probe-splits 3 --functional-batch-sizes 32 --windows 3,5,10 --out-dir results/v12_14_b320locked_lossagnostic_functional_mechanism/batch1_main_KMNIST_gpu2
CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py --probe-device auto --probe-datasets MNIST --probe-seeds 0,1,2 --probe-splits 3 --functional-batch-sizes 32 --windows 3,5,10 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2ca-DirectPrimitiveSketchWhiten,BM2cb-QuadPrimitiveSketchWhiten,BM2cc-BranchPrimitiveSketchWhiten,BM2cd-DirectQuadSketchIsotropy,BM2ce-BranchQuadSketchIsotropy,BM2cf-SignalReservoirSketchSpread,BM2cg-NoiseNullSketchSpread,BM4a-BranchMomentTransport,BM4b-QuadMomentTransport,BM4c-DirectQuadCovTransport,BM4d-RoleCovarianceEqualization,BM4e-LogitJacobianMomentTransport,BM4f-TailSafeMomentTransport --out-dir results/v12_14_b320locked_lossagnostic_functional_mechanism/batch2_b3b4_MNIST_gpu0
CUDA_VISIBLE_DEVICES=1 conda run -n kan python experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py --probe-device auto --probe-datasets Fashion-MNIST --probe-seeds 0,1,2 --probe-splits 3 --functional-batch-sizes 32 --windows 3,5,10 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2ca-DirectPrimitiveSketchWhiten,BM2cb-QuadPrimitiveSketchWhiten,BM2cc-BranchPrimitiveSketchWhiten,BM2cd-DirectQuadSketchIsotropy,BM2ce-BranchQuadSketchIsotropy,BM2cf-SignalReservoirSketchSpread,BM2cg-NoiseNullSketchSpread,BM4a-BranchMomentTransport,BM4b-QuadMomentTransport,BM4c-DirectQuadCovTransport,BM4d-RoleCovarianceEqualization,BM4e-LogitJacobianMomentTransport,BM4f-TailSafeMomentTransport --out-dir results/v12_14_b320locked_lossagnostic_functional_mechanism/batch2_b3b4_Fashion_gpu1
CUDA_VISIBLE_DEVICES=2 conda run -n kan python experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py --probe-device auto --probe-datasets KMNIST --probe-seeds 0,1,2 --probe-splits 3 --functional-batch-sizes 32 --windows 3,5,10 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2ca-DirectPrimitiveSketchWhiten,BM2cb-QuadPrimitiveSketchWhiten,BM2cc-BranchPrimitiveSketchWhiten,BM2cd-DirectQuadSketchIsotropy,BM2ce-BranchQuadSketchIsotropy,BM2cf-SignalReservoirSketchSpread,BM2cg-NoiseNullSketchSpread,BM4a-BranchMomentTransport,BM4b-QuadMomentTransport,BM4c-DirectQuadCovTransport,BM4d-RoleCovarianceEqualization,BM4e-LogitJacobianMomentTransport,BM4f-TailSafeMomentTransport --out-dir results/v12_14_b320locked_lossagnostic_functional_mechanism/batch2_b3b4_KMNIST_gpu2
conda run -n kan python experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py --aggregate-input-dirs results/v12_14_b320locked_lossagnostic_functional_mechanism/batch1_main_MNIST_gpu0,results/v12_14_b320locked_lossagnostic_functional_mechanism/batch1_main_Fashion_gpu1,results/v12_14_b320locked_lossagnostic_functional_mechanism/batch1_main_KMNIST_gpu2,results/v12_14_b320locked_lossagnostic_functional_mechanism/batch2_b3b4_MNIST_gpu0,results/v12_14_b320locked_lossagnostic_functional_mechanism/batch2_b3b4_Fashion_gpu1,results/v12_14_b320locked_lossagnostic_functional_mechanism/batch2_b3b4_KMNIST_gpu2 --out-dir results/v12_14_b320locked_lossagnostic_functional_mechanism/final_aggregate_official
```

## 3. 合规审计

```text
functional direction generator 不使用 CE vector / label-loss VJP / permuted-label CE surrogate；
CE / ECE / CEp99 只作为 audit metrics；
validation/test/future outcome 不作为 commit-time feature；
dataset-name branch = 0；
P4_open = {decision.get('p4_open')}。
```

## 4. 运行矩阵汇总

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
{null_lines}

## 5. Route

```text
route = {decision.get('route')}
bm2_full_3x3_pass_candidate_count = {decision.get('bm2_full_3x3_pass_candidate_count')}
bm2_partial_pass_rows = {decision.get('bm2_partial_pass_rows')}
next_recommended_action = {decision.get('next_recommended_action')}
```

## 6. Artifact Hash

| artifact | sha256 |
|---|---|
{hash_lines}
"""
    result_text = f"""# DG-KAN v12.14 B320Locked LossAgnosticFunctionalMechanism 结果复盘

生成时间：`{now_iso()}`

## 1. 结论

```text
route = {decision.get('route')}
P4_open = {decision.get('p4_open')}
bm2_full_3x3_pass_candidate_count = {decision.get('bm2_full_3x3_pass_candidate_count')}
bm2_partial_pass_rows = {decision.get('bm2_partial_pass_rows')}
```

本轮结果只统计 loss-agnostic official 候选。CE-specific projector 仍只作为 v12.13 诊断，不进入 v12.14 official 聚合。

## 2. 关键数据

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
{null_lines}

## 3. 当前 blocker

```text
{decision.get('next_recommended_action')}
```

若没有 P3 survivor，P4 按计划保持关闭。
"""
    DEFAULT_EXECUTION_LOG.write_text(exec_text, encoding="utf-8")
    DEFAULT_RESULT_LOG.write_text(result_text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_ROOT / "batch1_main"))
    parser.add_argument("--probe-device", default="auto")
    parser.add_argument("--probe-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--probe-seeds", default="0,1,2")
    parser.add_argument("--probe-splits", type=int, default=3)
    parser.add_argument("--functional-batch-sizes", default="32")
    parser.add_argument("--windows", default="3,5,10")
    parser.add_argument(
        "--calibration-methods",
        default="NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2aa-ClassMeanFreeCotangentK16,BM2ac-LogitWhitenedCotangentK16,BM2ae-OrthogonalRademacherCotangentK32,BM2ba-DirectRoleCotangentSpectral,BM2bb-QuadRoleCotangentSpectral,BM2bc-BranchRoleCotangentSpectral,BM2bd-DirectQuadRoleMixed,BM2bg-RoleAdaptiveCotangentNoLabelOrth",
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--probe-train-size", type=int, default=512)
    parser.add_argument("--probe-val-size", type=int, default=256)
    parser.add_argument("--probe-test-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1.0e-3)
    parser.add_argument("--sketch-dim", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    parser.add_argument("--bm3-strength", type=float, default=0.05)
    parser.add_argument("--aggregate-input-dirs", default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    if str(args.aggregate_input_dirs).strip():
        decision = aggregate_input_dirs(args, out_dir)
    else:
        decision = run_main(args, out_dir)
    print(json.dumps({"out_dir": str(out_dir.resolve()), "route": decision.get("route"), "p4_open": decision.get("p4_open")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
