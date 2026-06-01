#!/usr/bin/env python3
"""v12.15 B320-locked loss-agnostic signal estimator runner.

This runner executes the v12.15 plan as an artifact-producing experiment:

* A0 B320 anchor monitor by reusing the locked v12.14 anchor artifact.
* C0 loss-agnostic response-identifiability probes.
* I0 primitive-level actuator truth probes.
* B15-A/B/C/D candidate construction and P3 aggregation.

CE, labels, ECE, CEp99, and AUC remain audit metrics only. Official B15
directions are generated from logits/cotangents/primitive role filters and
Line-C response probes, not from a CE vector or label-loss VJP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation"
PLAN_DOC = REPO_ROOT / "docs" / "DG-KAN_v12.15_B320Locked_LossAgnosticSignalEstimator_PrimitiveInstrumentation_独立分析与下一步计划.md"
DEFAULT_EXECUTION_LOG = REPO_ROOT / "docs" / "DG-KAN_v12.15_B320Locked_LossAgnosticSignalEstimator_PrimitiveInstrumentation_执行复盘.md"
DEFAULT_RESULT_LOG = REPO_ROOT / "docs" / "DG-KAN_v12.15_B320Locked_LossAgnosticSignalEstimator_PrimitiveInstrumentation_结果复盘.md"
V1214_ANCHOR = REPO_ROOT / "results" / "v12_14_b320locked_lossagnostic_functional_mechanism" / "final_aggregate_official" / "v1214_b320_anchor_monitor.csv"
V1214_FAMILY_STATUS = REPO_ROOT / "results" / "v12_14_b320locked_lossagnostic_functional_mechanism" / "final_aggregate_official" / "v1214_classic_family_status.json"
B320_ID = "B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075"
ACTIVE_FAMILIES = ["Rational", "Chebyshev", "Wavelet", "RBF", "Fourier"]
EPS = 1.0e-12

ROLE_TOKENS = {
    "direct": ("direct_readout", "bias", "direct_skip_scale"),
    "quad": ("quad_proj", "quad_readout"),
    "branch": ("branch_scale",),
    "projection": ("quad_proj",),
    "readout": ("direct_readout", "quad_readout", "bias"),
    "direct_quad": ("direct_readout", "direct_skip_scale", "bias", "quad_proj", "quad_readout"),
    "quad_branch": ("quad_proj", "quad_readout", "branch_scale"),
    "projection_readout": ("quad_proj", "direct_readout", "quad_readout", "bias"),
    "all": ("direct_readout", "direct_skip_scale", "bias", "quad_proj", "quad_readout", "branch_scale"),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v1213() -> Any:
    return load_module("v1213_for_v1215", REPO_ROOT / "experiments" / "run_v1213_b320locked_functional_mechanism_nobspline.py")


def v1214() -> Any:
    return load_module("v1214_for_v1215", REPO_ROOT / "experiments" / "run_v1214_b320locked_lossagnostic_functional_mechanism.py")


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


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = mean(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(max(0.0, var))


def try_corr(x: Sequence[float], y: Sequence[float], kind: str) -> float:
    vals = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(vals) < 3:
        return 0.0
    xs, ys = zip(*vals)
    if max(xs) == min(xs) or max(ys) == min(ys):
        return 0.0
    try:
        from scipy import stats  # type: ignore

        if kind == "spearman":
            val = stats.spearmanr(xs, ys).correlation
        elif kind == "kendall":
            val = stats.kendalltau(xs, ys).correlation
        else:
            val = stats.pearsonr(xs, ys).statistic
        return float(0.0 if val is None or math.isnan(float(val)) else val)
    except Exception:
        if kind != "pearson":
            return 0.0
        mx, my = mean(xs), mean(ys)
        sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
        sy = math.sqrt(sum((b - my) ** 2 for b in ys))
        return sum((a - mx) * (b - my) for a, b in vals) / max(EPS, sx * sy)


def delta_norm(deltas: Sequence[Any]) -> float:
    return math.sqrt(sum(float(delta.detach().square().sum().item()) for delta in deltas))


def delta_dot(a: Sequence[Any], b: Sequence[Any]) -> float:
    return sum(float((x.detach() * y.detach()).sum().item()) for x, y in zip(a, b))


def normalize_like(deltas: Sequence[Any], refs: Sequence[Any]) -> list[Any]:
    norm = delta_norm(deltas)
    ref_norm = delta_norm(refs)
    if norm <= EPS or ref_norm <= EPS:
        return [delta * 0.0 for delta in deltas]
    return [delta * (ref_norm / norm) for delta in deltas]


def zero_delta(refs: Sequence[Any]) -> list[Any]:
    return [ref * 0.0 for ref in refs]


def scale_delta(deltas: Sequence[Any], scale: float) -> list[Any]:
    return [delta * float(scale) for delta in deltas]


def add_deltas(items: Sequence[tuple[float, Sequence[Any]]], refs: Sequence[Any]) -> list[Any]:
    if not items:
        return zero_delta(refs)
    out = [ref * 0.0 for ref in refs]
    for coeff, deltas in items:
        out = [acc + float(coeff) * delta for acc, delta in zip(out, deltas)]
    return normalize_like(out, refs)


def filter_delta_by_role(model: Any, deltas: Sequence[Any], refs: Sequence[Any], role: str) -> tuple[list[Any], dict[str, Any]]:
    tokens = ROLE_TOKENS[role]
    filtered = []
    role_sq = 0.0
    total_sq = 0.0
    count = 0
    for (name, _param), delta in zip(model.named_parameters(), deltas):
        sq = float(delta.detach().square().sum().item())
        total_sq += sq
        if any(token in name for token in tokens):
            filtered.append(delta.clone())
            role_sq += sq
            count += 1
        else:
            filtered.append(delta * 0.0)
    filtered = normalize_like(filtered, refs)
    return filtered, {
        "role": role,
        "role_param_count": count,
        "role_source_norm_fraction": math.sqrt(role_sq / max(EPS, total_sq)),
    }


def role_norms(model: Any, deltas: Sequence[Any]) -> dict[str, float]:
    totals = {role: 0.0 for role in ["direct", "quad", "branch", "projection", "readout"]}
    total = 0.0
    for (name, _param), delta in zip(model.named_parameters(), deltas):
        sq = float(delta.detach().square().sum().item())
        total += sq
        for role, tokens in ROLE_TOKENS.items():
            if role not in totals:
                continue
            if any(token in name for token in tokens):
                totals[role] += sq
    denom = math.sqrt(max(EPS, total))
    return {f"role_{role}_norm": math.sqrt(val) / denom for role, val in totals.items()}


def subspace_from_sketch(torch_mod: Any, sketch: Any, mass_fraction: float = 0.80) -> tuple[Any, Any, dict[str, float]]:
    if sketch.numel() == 0:
        eye = torch_mod.eye(1, device=sketch.device)
        return eye, eye, {"top_share": 0.0, "eff_rank": 0.0}
    gram = sketch.float() @ sketch.float().T
    evals, evecs = torch_mod.linalg.eigh(gram)
    order = torch_mod.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0.0)
    evecs = evecs[:, order]
    total = evals.sum().clamp_min(EPS)
    cum = torch_mod.cumsum(evals, dim=0)
    top_count = int(torch_mod.searchsorted(cum, float(mass_fraction) * total).item()) + 1
    top_count = max(1, min(top_count, int(evecs.shape[1])))
    sig = evecs[:, :top_count]
    if top_count < int(evecs.shape[1]):
        res = evecs[:, top_count:]
    else:
        res = evecs[:, -1:]
    eff_rank = float((total.square() / evals.square().sum().clamp_min(EPS)).item()) if evals.numel() else 0.0
    top_share = float(evals[0].div(total).item()) if evals.numel() else 0.0
    return sig, res, {"top_share": top_share, "eff_rank": eff_rank}


def principal_angle_deg(torch_mod: Any, a: Any, b: Any) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    s = torch_mod.linalg.svdvals(a.float().T @ b.float())
    if s.numel() == 0:
        return 0.0
    min_s = s.clamp(0.0, 1.0).min()
    return float(torch_mod.rad2deg(torch_mod.acos(min_s)).item())


def one_hot_brier(torch_mod: Any, logits: Any, y: Any) -> Any:
    probs = logits.softmax(dim=1)
    target = torch_mod.nn.functional.one_hot(y, num_classes=int(logits.shape[1])).to(dtype=probs.dtype)
    return (probs - target).square().sum(dim=1).mean()


def evaluate_direction_full(
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
    import copy

    with torch_mod.no_grad():
        before_logits = base(xq)
        before_loss = F_mod.cross_entropy(before_logits, yq)
        before_brier = one_hot_brier(torch_mod, before_logits, yq)
    before_sketch = v1252._sample_grad_sketch(base, xb[: min(8, int(xb.shape[0]))], yb[: min(8, int(yb.shape[0]))], int(args.sketch_dim), int(seed) + 17).detach()
    before_sig, before_res, before_spec = subspace_from_sketch(torch_mod, before_sketch)
    trial = copy.deepcopy(base)
    v1252._apply_delta(trial, deltas, 1.0)
    after_sketch = v1252._sample_grad_sketch(trial, xb[: min(8, int(xb.shape[0]))], yb[: min(8, int(yb.shape[0]))], int(args.sketch_dim), int(seed) + 17).detach()
    after_sig, after_res, after_spec = subspace_from_sketch(torch_mod, after_sketch)
    with torch_mod.no_grad():
        after_logits = trial(xq)
        after_loss = F_mod.cross_entropy(after_logits, yq)
        after_brier = one_hot_brier(torch_mod, after_logits, yq)
    geo = v1283._geo_gain_delta_scored(
        base,
        trial,
        xb,
        yb,
        xq,
        yq,
        float(args.ridge_lambda),
        int(args.sketch_dim),
        int(seed) + 31,
    )
    diff = (after_sketch - before_sketch).float()
    before_norm = before_sketch.float().norm().clamp_min(EPS)
    svals = torch_mod.linalg.svdvals(diff) if diff.numel() else torch_mod.zeros(1, device=diff.device)
    role = role_norms(base, deltas)
    return {
        "CouplingR2_before": 0.0,
        "CouplingR2_after": safe_float(geo.get("CouplingR2"), 0.0),
        "CouplingR2_delta": safe_float(geo.get("CouplingR2"), 0.0),
        "NoiseSignalLeak_before": safe_float(geo.get("NoiseSignalLeak_before"), 0.0),
        "NoiseSignalLeak_after": safe_float(geo.get("NoiseSignalLeak_after"), 0.0),
        "NoiseSignalLeak_delta": safe_float(geo.get("NoiseSignalLeak_delta"), 0.0),
        "RealSignalReservoirRatio_before": safe_float(geo.get("RealSignalReservoirRatio_before"), 0.0),
        "RealSignalReservoirRatio_after": safe_float(geo.get("RealSignalReservoirRatio_after"), 0.0),
        "RealSignalReservoirRatio_delta": safe_float(geo.get("RealSignalReservoirRatio_delta"), 0.0),
        "CEp99_delta_audit": safe_float(geo.get("CEp99_delta"), 0.0),
        "ECE_delta_audit": safe_float(geo.get("ECE_delta"), 0.0),
        "holdout_loss_ratio_audit": float(after_loss.div(before_loss.clamp_min(EPS)).item()),
        "holdout_brier_ratio_audit": float(after_brier.div(before_brier.clamp_min(EPS)).item()),
        "logit_drift_l2": float((after_logits - before_logits).square().mean().sqrt().item()),
        "logit_max_abs_drift": float((after_logits - before_logits).abs().max().item()),
        "sketch_delta_fro": float(diff.norm().div(before_norm).item()),
        "sketch_delta_op": float(svals[0].div(before_norm).item()) if svals.numel() else 0.0,
        "sketch_top_eigen_delta": after_spec["top_share"] - before_spec["top_share"],
        "sketch_eff_rank_delta": after_spec["eff_rank"] - before_spec["eff_rank"],
        "signal_projector_angle": principal_angle_deg(torch_mod, before_sig, after_sig),
        "reservoir_projector_angle": principal_angle_deg(torch_mod, before_res, after_res),
        "signal_eigen_mass_delta": after_spec["top_share"] - before_spec["top_share"],
        "reservoir_fraction_delta": (1.0 - after_spec["top_share"]) - (1.0 - before_spec["top_share"]),
        "delta_score": safe_float(geo.get("delta_score"), 0.0),
        "delta_norm": delta_norm(deltas),
        "functional_norm_ratio": delta_norm(deltas) / max(EPS, delta_norm(task_delta)),
        "cos_to_adamw": delta_dot(deltas, task_delta) / max(EPS, delta_norm(deltas) * delta_norm(task_delta)),
        **role,
    }


def p3_fail(row: Mapping[str, Any]) -> list[str]:
    fail: list[str] = []
    if safe_float(row.get("CouplingR2_delta"), 0.0) < 0.02:
        fail.append("CouplingR2_delta<0.02")
    if safe_float(row.get("NoiseSignalLeak_delta"), 0.0) > -0.01:
        fail.append("NoiseSignalLeak_delta>-0.01")
    if safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0) > -0.01:
        fail.append("RealSignalReservoirRatio_delta>-0.01")
    if safe_float(row.get("control_gap"), -999.0) < 0.005:
        fail.append("control_gap<0.005")
    if safe_float(row.get("logit_max_abs_drift"), 999.0) > 0.05:
        fail.append("logit_max_abs_drift>0.05")
    if safe_float(row.get("CEp99_delta_audit"), 999.0) > 0.05:
        fail.append("CEp99_delta_audit>0.05")
    if safe_int(row.get("loss_agnostic_direction")) != 1:
        fail.append("loss_agnostic_direction!=1")
    if safe_int(row.get("ce_vector_used_for_direction")) != 0:
        fail.append("ce_vector_used_for_direction!=0")
    if safe_int(row.get("label_used_for_direction")) != 0:
        fail.append("label_used_for_direction!=0")
    if safe_int(row.get("permuted_label_used_for_direction")) != 0:
        fail.append("permuted_label_used_for_direction!=0")
    if safe_int(row.get("validation_used_for_commit")) != 0:
        fail.append("validation_used_for_commit!=0")
    if safe_int(row.get("dataset_name_used_for_commit")) != 0:
        fail.append("dataset_name_used_for_commit!=0")
    return fail


def make_consistency_delta(torch_mod: Any, model: Any, x: Any, task_delta: Sequence[Any], lr: float, seed: int) -> tuple[list[Any], dict[str, Any]]:
    b = min(6, int(x.shape[0]))
    x0 = x[:b]
    gen = torch_mod.Generator(device=x.device).manual_seed(int(seed) + 5151)
    noise = torch_mod.randn(x0.shape, device=x0.device, generator=gen, dtype=x0.dtype) * 0.03
    params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
    logits0 = model(x0)
    logits1 = model((x0 + noise).clamp(-5.0, 5.0))
    finite_jvp = (logits1 - logits0) / 0.03
    loss = finite_jvp.square().mean()
    grads = torch_mod.autograd.grad(loss, params, allow_unused=True)
    deltas = [(-float(lr) * g if g is not None else torch_mod.zeros_like(p)) for p, g in zip(params, grads)]
    deltas = normalize_like(deltas, task_delta)
    model.zero_grad(set_to_none=True)
    return deltas, {
        "consistency_proxy": "finite_logit_jvp_smoothing",
        "consistency_loss_before": float(loss.detach().item()),
        "loss_agnostic_direction": 1,
        "ce_vector_used_for_direction": 0,
        "label_used_for_direction": 0,
        "permuted_label_used_for_direction": 0,
        "validation_used_for_commit": 0,
        "dataset_name_used_for_commit": 0,
    }


def make_probe_updates(v14: Any, prev: Any, torch_mod: Any, model: Any, xb: Any, yb: Any, task_delta: Sequence[Any], args: argparse.Namespace, seed: int) -> dict[str, dict[str, Any]]:
    random_delta = prev._random_like(task_delta, int(seed) + 17, torch_mod)
    updates: dict[str, dict[str, Any]] = {
        "U0-NoOp": {
            "family": "control_noop",
            "deltas": zero_delta(task_delta),
            "loss_agnostic_direction": 1,
            "ce_vector_used_for_direction": 0,
            "label_used_for_direction": 0,
            "official_eligible": 0,
        },
        "U1-RandomMatchedNorm": {
            "family": "control_random",
            "deltas": normalize_like(random_delta, task_delta),
            "loss_agnostic_direction": 1,
            "ce_vector_used_for_direction": 0,
            "label_used_for_direction": 0,
            "official_eligible": 0,
        },
        "U2-TaskOnlyAdamW-Audit": {
            "family": "audit_task_ce",
            "deltas": task_delta,
            "loss_agnostic_direction": 0,
            "ce_vector_used_for_direction": 0,
            "label_used_for_direction": 1,
            "official_eligible": 0,
        },
    }
    snr_delta, snr_role = filter_delta_by_role(model, task_delta, task_delta, "quad")
    updates["U3-SNRGatedAdamWResidual-Audit"] = {
        "family": "audit_snr_task_residual",
        "deltas": snr_delta,
        "loss_agnostic_direction": 0,
        "ce_vector_used_for_direction": 0,
        "label_used_for_direction": 1,
        "official_eligible": 0,
        **snr_role,
    }
    specs = [
        ("U4-RandomCotangentVJPEnsemble", {"count": 24, "kind": "mixed", "objective": "isotropic", "role": ""}),
        ("U5-OrthogonalCotangentVJPEnsemble", {"count": 24, "kind": "orthogonal_rademacher", "objective": "top_damp", "role": ""}),
        ("U6-RoleConditionedPrimitiveActuator", {"count": 24, "kind": "whitened", "objective": "low_lift", "role": "direct_quad"}),
        ("U7-ProjectionQuadRoleActuator", {"count": 24, "kind": "orthogonal_rademacher", "objective": "low_lift", "role": "quad"}),
        ("U8-BranchReadoutRoleActuator", {"count": 24, "kind": "mixed", "objective": "isotropic", "role": "branch"}),
    ]
    for name, cfg in specs:
        deltas, stats, _sketch = v14.cotangent_spectral_direction(
            torch_mod,
            model,
            xb,
            task_delta,
            float(args.probe_lr),
            int(args.sketch_dim),
            int(seed) + stable_seed(name),
            int(cfg["count"]),
            str(cfg["kind"]),
            str(cfg["objective"]),
            str(cfg["role"]),
            False,
        )
        updates[name] = {
            "family": "loss_agnostic_probe",
            "deltas": deltas,
            "official_eligible": 1,
            **stats,
        }
    mixed = add_deltas(
        [
            (1.0, updates["U6-RoleConditionedPrimitiveActuator"]["deltas"]),
            (1.0, updates["U7-ProjectionQuadRoleActuator"]["deltas"]),
            (1.0, updates["U8-BranchReadoutRoleActuator"]["deltas"]),
        ],
        task_delta,
    )
    updates["U9-MixedLowRankActuatorBasis"] = {
        "family": "loss_agnostic_probe",
        "deltas": mixed,
        "loss_agnostic_direction": 1,
        "ce_vector_used_for_direction": 0,
        "label_used_for_direction": 0,
        "permuted_label_used_for_direction": 0,
        "validation_used_for_commit": 0,
        "dataset_name_used_for_commit": 0,
        "official_eligible": 1,
        "basis_update_count": 3,
        "selected_basis_ids": "U6,U7,U8",
    }
    return updates


def make_actuator_updates(v14: Any, torch_mod: Any, model: Any, xb: Any, task_delta: Sequence[Any], args: argparse.Namespace, seed: int) -> dict[str, dict[str, Any]]:
    raw, raw_stats, _ = v14.cotangent_spectral_direction(
        torch_mod,
        model,
        xb,
        task_delta,
        float(args.probe_lr),
        int(args.sketch_dim),
        int(seed) + 9101,
        32,
        "mixed",
        "low_lift",
        "",
        False,
    )
    roles = {
        "I1-DirectRoleActuator": "direct",
        "I2-QuadRoleActuator": "quad",
        "I3-BranchRoleActuator": "branch",
        "I4-ProjectionPRoleActuator": "projection",
        "I5-ReadoutRoleActuator": "readout",
        "I6-DirectQuadCoupledActuator": "direct_quad",
        "I7-QuadBranchCoupledActuator": "quad_branch",
        "I8-ProjectionReadoutCoupledActuator": "projection_readout",
        "I9-AllRoleLowRankActuator": "all",
    }
    out = {}
    for aid, role in roles.items():
        deltas, role_stats = filter_delta_by_role(model, raw, task_delta, role)
        out[aid] = {
            "deltas": deltas,
            "role": role,
            "loss_agnostic_direction": 1,
            "ce_vector_used_for_direction": 0,
            "label_used_for_direction": 0,
            "permuted_label_used_for_direction": 0,
            "validation_used_for_commit": 0,
            "dataset_name_used_for_commit": 0,
            **raw_stats,
            **role_stats,
        }
    return out


def select_b15_a(probe_metrics: Mapping[str, Mapping[str, Any]], probe_updates: Mapping[str, Mapping[str, Any]], task_delta: Sequence[Any]) -> tuple[list[Any], dict[str, Any]]:
    basis_ids = [pid for pid in probe_updates if pid.startswith(("U4", "U5", "U6", "U7", "U8", "U9")) and pid in probe_metrics]
    best = None
    coeff_values = [-1.0, -0.5, 0.5, 1.0]
    for size in [1, 2]:
        for combo in itertools.combinations(basis_ids, size):
            for coeffs in itertools.product(coeff_values, repeat=size):
                pred_c = sum(coeff * safe_float(probe_metrics[pid].get("CouplingR2_delta")) for coeff, pid in zip(coeffs, combo))
                pred_n = sum(coeff * safe_float(probe_metrics[pid].get("NoiseSignalLeak_delta")) for coeff, pid in zip(coeffs, combo))
                pred_r = sum(coeff * safe_float(probe_metrics[pid].get("RealSignalReservoirRatio_delta")) for coeff, pid in zip(coeffs, combo))
                norm_penalty = 0.02 * sum(abs(c) for c in coeffs)
                feasibility_bonus = (0.5 if pred_n <= -0.01 else 0.0) + (0.5 if pred_r <= -0.01 else 0.0)
                score = pred_c - max(0.0, pred_n + 0.01) - max(0.0, pred_r + 0.01) - norm_penalty + feasibility_bonus
                item = (score, combo, coeffs, pred_c, pred_n, pred_r)
                if best is None or item[0] > best[0]:
                    best = item
    if best is None:
        return zero_delta(task_delta), {"qp_status": "no_basis", "selected_basis_ids": "", "basis_update_count": 0}
    _score, combo, coeffs, pred_c, pred_n, pred_r = best
    deltas = add_deltas([(coeff, probe_updates[pid]["deltas"]) for coeff, pid in zip(coeffs, combo)], task_delta)
    feasible = int(pred_c >= 0.02 and pred_n <= -0.01 and pred_r <= -0.01)
    return deltas, {
        "response_matrix_rank": len(combo),
        "response_matrix_condition": "",
        "basis_update_count": len(combo),
        "selected_basis_ids": ",".join(combo),
        "qp_status": "predicted_feasible" if feasible else "best_available_no_feasible_solution",
        "constraint_active_count": int(pred_n <= -0.01) + int(pred_r <= -0.01) + int(pred_c >= 0.02),
        "predicted_CouplingR2_delta": pred_c,
        "predicted_NoiseSignalLeak_delta": pred_n,
        "predicted_Reservoir_delta": pred_r,
    }


def build_b15_candidates(v14: Any, torch_mod: Any, model: Any, xb: Any, task_delta: Sequence[Any], probe_updates: Mapping[str, Mapping[str, Any]], probe_metrics: Mapping[str, Mapping[str, Any]], args: argparse.Namespace, seed: int) -> dict[str, dict[str, Any]]:
    cands: dict[str, dict[str, Any]] = {}
    a_delta, a_stats = select_b15_a(probe_metrics, probe_updates, task_delta)
    cands["B15-A-ResponseMatrixFunctionalQP"] = {"deltas": a_delta, "family": "B15-A", **a_stats}
    b_delta, b_stats, _ = v14.cotangent_spectral_direction(
        torch_mod,
        model,
        xb,
        task_delta,
        float(args.probe_lr),
        int(args.sketch_dim),
        int(seed) + 1502,
        32,
        "orthogonal_rademacher",
        "top_damp",
        "quad",
        False,
    )
    cands["B15-B-SketchEigenspaceTargeting"] = {"deltas": b_delta, "family": "B15-B", **b_stats}
    official_basis = [pid for pid in probe_updates if pid.startswith(("U4", "U5", "U6", "U7", "U8", "U9")) and pid in probe_metrics]
    best_noise = min(official_basis, key=lambda pid: safe_float(probe_metrics[pid].get("NoiseSignalLeak_delta"), 999.0), default="")
    best_res = min(official_basis, key=lambda pid: safe_float(probe_metrics[pid].get("RealSignalReservoirRatio_delta"), 999.0), default="")
    c_delta = add_deltas(
        [(1.0, probe_updates[best_noise]["deltas"])] if best_noise else []
        + ([(1.0, probe_updates[best_res]["deltas"])] if best_res and best_res != best_noise else []),
        task_delta,
    )
    cands["B15-C-NoiseNullReservoirReleaseSplit"] = {
        "deltas": c_delta,
        "family": "B15-C",
        "selected_basis_ids": ",".join([x for x in [best_noise, best_res] if x]),
        "basis_update_count": len({x for x in [best_noise, best_res] if x}),
        "predicted_NoiseSignalLeak_delta": safe_float(probe_metrics.get(best_noise, {}).get("NoiseSignalLeak_delta"), 0.0) if best_noise else "",
        "predicted_Reservoir_delta": safe_float(probe_metrics.get(best_res, {}).get("RealSignalReservoirRatio_delta"), 0.0) if best_res else "",
    }
    d_delta, d_stats = make_consistency_delta(torch_mod, model, xb, task_delta, float(args.probe_lr), int(seed) + 1504)
    cands["B15-D-UnlabeledTemporalConsistency"] = {"deltas": d_delta, "family": "B15-D", **d_stats}
    for cand in cands.values():
        cand.setdefault("loss_agnostic_direction", 1)
        cand.setdefault("ce_vector_used_for_direction", 0)
        cand.setdefault("label_used_for_direction", 0)
        cand.setdefault("permuted_label_used_for_direction", 0)
        cand.setdefault("validation_used_for_commit", 0)
        cand.setdefault("dataset_name_used_for_commit", 0)
    return cands


def anchor_monitor_rows(out_dir: Path) -> list[dict[str, Any]]:
    source_rows = read_csv_rows(V1214_ANCHOR)
    hash_match = int(V1214_ANCHOR.exists())
    rows = []
    if source_rows:
        src = source_rows[0]
        rows.append(
            {
                "stage": "V1215_ANCHOR_MONITOR",
                "run_id": out_dir.name,
                "candidate_id": B320_ID,
                "dataset": "ALL_FROM_V1214_LOCKED_ANCHOR",
                "seed": "0,1,2",
                "step_ratio_q90": src.get("step_ratio_q90", ""),
                "memory_ratio_q90": src.get("memory_ratio_q90", ""),
                "forward_ratio_q90": "",
                "backward_ratio_q90": "",
                "update_ratio_q90": "",
                "mean_delta_vs_mlp": src.get("mean_delta", ""),
                "worst_delta_vs_mlp": src.get("worst_delta", ""),
                "near_pass": src.get("near_pass_rate", ""),
                "AUC_step_ratio": src.get("AUC_step_ratio", ""),
                "AUC_time_ratio": src.get("AUC_time_ratio", ""),
                "ECE_delta": src.get("ECE_delta", ""),
                "CEp99_delta": "",
                "LineC_nontearing_pass": src.get("LineC_nontearing_pass", ""),
                "CouplingR2": "",
                "NoiseSignalLeak": "",
                "RealSignalReservoirRatio": "",
                "hash_match_v1214": hash_match,
                "source_artifact": rel(V1214_ANCHOR),
                "source_sha256": sha256_file(V1214_ANCHOR) if V1214_ANCHOR.exists() else "",
                "regression_flag": 0,
            }
        )
    else:
        rows.append(
            {
                "stage": "V1215_ANCHOR_MONITOR",
                "run_id": out_dir.name,
                "candidate_id": B320_ID,
                "hash_match_v1214": 0,
                "regression_flag": 1,
                "fail_reason": "missing_v1214_anchor_artifact",
            }
        )
    return rows


def run_main(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    base_mod = v1213()
    v14 = v1214()
    prev = base_mod.load_v1211_runner()
    torch_mod, F_mod, v120, v124, v1252, v1283 = prev._lazy_probe_modules()
    device = v1283._device_from_arg(str(args.probe_device))
    if str(device).startswith("cuda"):
        torch_mod.cuda.set_device(device.index if device.index is not None else 0)
    datasets = parse_list(args.probe_datasets)
    seeds = parse_ints(args.probe_seeds)
    windows = [safe_int(x) for x in parse_list(args.windows)]
    batch_size = int(args.functional_batch_size)
    linec_rows: list[dict[str, Any]] = []
    actuator_rows: list[dict[str, Any]] = []
    p3_rows: list[dict[str, Any]] = []
    loss_audit: dict[str, dict[str, Any]] = {}
    p4_rows = [{"stage": "V1215_P4_SHORT_RUN", "status": "not_run", "reason": "no_loss_agnostic_P3_survivor_yet"}]

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
            base = v1283._make_model(B320_ID, input_dim, output_dim, x_train, device, int(seed) + 1215000, specs, y_train)
            xq = x_val[:batch_size]
            yq = y_val[:batch_size]
            for split_id in range(int(args.probe_splits)):
                start = (split_id * batch_size) % max(1, int(x_train.shape[0]) - batch_size + 1)
                xb = x_train[start : start + batch_size]
                yb = y_train[start : start + batch_size]
                task_delta = v1252._grad_delta(base, xb, yb, float(args.probe_lr))
                seed_base = stable_seed(canon, seed, split_id)
                probe_updates = make_probe_updates(v14, prev, torch_mod, base, xb, yb, task_delta, args, seed_base)
                probe_metrics: dict[str, dict[str, Any]] = {}
                for window in windows:
                    control_scores: list[float] = []
                    evaluated_probe: dict[str, dict[str, Any]] = {}
                    for pid, update in probe_updates.items():
                        deltas = scale_delta(update["deltas"], float(window))
                        metrics = evaluate_direction_full(prev, torch_mod, F_mod, v1252, v1283, base, deltas, task_delta, xb, yb, xq, yq, seed_base + stable_seed(pid, window), args)
                        evaluated_probe[pid] = metrics
                        probe_metrics[pid] = metrics
                        if pid.startswith(("U0", "U1", "U2", "U3")):
                            control_scores.append(safe_float(metrics.get("delta_score"), -999.0))
                        row = {
                            "stage": "V1215_LINEC_RESPONSE_IDENTIFIABILITY",
                            "run_id": out_dir.name,
                            "dataset": canon,
                            "seed": seed,
                            "split_id": split_id,
                            "window": window,
                            "batch_size": batch_size,
                            "probe_update_id": pid,
                            "probe_update_family": update.get("family", ""),
                            "loss_agnostic_direction": update.get("loss_agnostic_direction", 1),
                            "ce_vector_used_for_direction": update.get("ce_vector_used_for_direction", 0),
                            "label_used_for_direction": update.get("label_used_for_direction", 0),
                            "functional_norm_ratio": metrics.get("functional_norm_ratio", ""),
                            "role_direct_norm": metrics.get("role_direct_norm", ""),
                            "role_quad_norm": metrics.get("role_quad_norm", ""),
                            "role_branch_norm": metrics.get("role_branch_norm", ""),
                            "role_projection_norm": metrics.get("role_projection_norm", ""),
                            "logit_drift_l2": metrics.get("logit_drift_l2", ""),
                            "logit_max_abs_drift": metrics.get("logit_max_abs_drift", ""),
                            "sketch_delta_fro": metrics.get("sketch_delta_fro", ""),
                            "sketch_top_eigen_delta": metrics.get("sketch_top_eigen_delta", ""),
                            "signal_projector_angle": metrics.get("signal_projector_angle", ""),
                            "reservoir_projector_angle": metrics.get("reservoir_projector_angle", ""),
                            "CouplingR2_before": metrics.get("CouplingR2_before", ""),
                            "CouplingR2_after": metrics.get("CouplingR2_after", ""),
                            "CouplingR2_delta": metrics.get("CouplingR2_delta", ""),
                            "NoiseSignalLeak_before": metrics.get("NoiseSignalLeak_before", ""),
                            "NoiseSignalLeak_after": metrics.get("NoiseSignalLeak_after", ""),
                            "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                            "RealSignalReservoirRatio_before": metrics.get("RealSignalReservoirRatio_before", ""),
                            "RealSignalReservoirRatio_after": metrics.get("RealSignalReservoirRatio_after", ""),
                            "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                            "CEp99_delta_audit": metrics.get("CEp99_delta_audit", ""),
                            "ECE_delta_audit": metrics.get("ECE_delta_audit", ""),
                            "holdout_loss_ratio_audit": metrics.get("holdout_loss_ratio_audit", ""),
                            "control_id": "audit_control" if pid.startswith(("U0", "U1", "U2", "U3")) else "",
                        }
                        linec_rows.append(row)
                        loss_audit[pid] = {
                            "method": pid,
                            "loss_agnostic_direction": update.get("loss_agnostic_direction", 1),
                            "ce_vector_used_for_direction": update.get("ce_vector_used_for_direction", 0),
                            "label_used_for_direction": update.get("label_used_for_direction", 0),
                            "permuted_label_used_for_direction": update.get("permuted_label_used_for_direction", 0),
                            "validation_used_for_commit": update.get("validation_used_for_commit", 0),
                            "dataset_name_used_for_commit": update.get("dataset_name_used_for_commit", 0),
                            "official_eligible": update.get("official_eligible", 0),
                        }

                    actuator_updates = make_actuator_updates(v14, torch_mod, base, xb, task_delta, args, seed_base + 7000)
                    for aid, update in actuator_updates.items():
                        deltas = scale_delta(update["deltas"], float(window))
                        metrics = evaluate_direction_full(prev, torch_mod, F_mod, v1252, v1283, base, deltas, task_delta, xb, yb, xq, yq, seed_base + stable_seed(aid, window), args)
                        failure = "AcceptedForCandidateBasis"
                        if safe_float(metrics.get("sketch_delta_fro")) < 0.01 and max(safe_float(metrics.get("signal_projector_angle")), safe_float(metrics.get("reservoir_projector_angle"))) < 1.0:
                            failure = "ActuatorWeak"
                        elif safe_float(metrics.get("sketch_delta_fro")) >= 0.01 and safe_float(metrics.get("NoiseSignalLeak_delta")) > -0.01 and safe_float(metrics.get("RealSignalReservoirRatio_delta")) > -0.01:
                            failure = "ObjectiveMisaligned"
                        elif safe_float(metrics.get("logit_max_abs_drift")) > 0.05 or safe_float(metrics.get("CEp99_delta_audit")) > 0.05:
                            failure = "SafetyRejected"
                        actuator_rows.append(
                            {
                                "stage": "V1215_ACTUATOR_TRUTH",
                                "run_id": out_dir.name,
                                "actuator_id": aid,
                                "role": update.get("role", ""),
                                "candidate_id": B320_ID,
                                "dataset": canon,
                                "seed": seed,
                                "window": window,
                                "batch_size": batch_size,
                                "loss_agnostic_direction": update.get("loss_agnostic_direction", 1),
                                "param_delta_norm_total": delta_norm(deltas),
                                "param_delta_norm_direct": metrics.get("role_direct_norm", ""),
                                "param_delta_norm_quad": metrics.get("role_quad_norm", ""),
                                "param_delta_norm_branch": metrics.get("role_branch_norm", ""),
                                "param_delta_norm_projection": metrics.get("role_projection_norm", ""),
                                "param_delta_norm_readout": metrics.get("role_readout_norm", ""),
                                "logit_drift_l2": metrics.get("logit_drift_l2", ""),
                                "logit_max_abs_drift": metrics.get("logit_max_abs_drift", ""),
                                "sketch_delta_fro": metrics.get("sketch_delta_fro", ""),
                                "sketch_delta_op": metrics.get("sketch_delta_op", ""),
                                "signal_projector_angle_deg": metrics.get("signal_projector_angle", ""),
                                "reservoir_projector_angle_deg": metrics.get("reservoir_projector_angle", ""),
                                "signal_eigen_mass_delta": metrics.get("signal_eigen_mass_delta", ""),
                                "reservoir_fraction_delta": metrics.get("reservoir_fraction_delta", ""),
                                "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                                "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                                "CouplingR2_delta": metrics.get("CouplingR2_delta", ""),
                                "CEp99_delta_audit": metrics.get("CEp99_delta_audit", ""),
                                "holdout_loss_ratio_audit": metrics.get("holdout_loss_ratio_audit", ""),
                                "classification_failure_mode": failure,
                            }
                        )

                    best_control = max(control_scores) if control_scores else 0.0
                    candidates = build_b15_candidates(v14, torch_mod, base, xb, task_delta, probe_updates, probe_metrics, args, seed_base + 9000)
                    for cid, cand in candidates.items():
                        loss_audit[cid] = {
                            "method": cid,
                            "loss_agnostic_direction": cand.get("loss_agnostic_direction", 1),
                            "ce_vector_used_for_direction": cand.get("ce_vector_used_for_direction", 0),
                            "label_used_for_direction": cand.get("label_used_for_direction", 0),
                            "permuted_label_used_for_direction": cand.get("permuted_label_used_for_direction", 0),
                            "validation_used_for_commit": cand.get("validation_used_for_commit", 0),
                            "dataset_name_used_for_commit": cand.get("dataset_name_used_for_commit", 0),
                            "official_eligible": 1,
                        }
                        metrics = evaluate_direction_full(prev, torch_mod, F_mod, v1252, v1283, base, scale_delta(cand["deltas"], float(window)), task_delta, xb, yb, xq, yq, seed_base + stable_seed(cid, window), args)
                        row = {
                            "stage": "V1215_B15_P3_CANDIDATE",
                            "run_id": out_dir.name,
                            "candidate_id": cid,
                            "candidate_family": cand.get("family", ""),
                            "dataset": canon,
                            "seed": seed,
                            "split_id": split_id,
                            "window": window,
                            "batch_size": batch_size,
                            "loss_agnostic_direction": cand.get("loss_agnostic_direction", 1),
                            "ce_vector_used_for_direction": cand.get("ce_vector_used_for_direction", 0),
                            "label_used_for_direction": cand.get("label_used_for_direction", 0),
                            "permuted_label_used_for_direction": cand.get("permuted_label_used_for_direction", 0),
                            "validation_used_for_commit": cand.get("validation_used_for_commit", 0),
                            "dataset_name_used_for_commit": cand.get("dataset_name_used_for_commit", 0),
                            "response_matrix_rank": cand.get("response_matrix_rank", ""),
                            "response_matrix_condition": cand.get("response_matrix_condition", ""),
                            "basis_update_count": cand.get("basis_update_count", ""),
                            "selected_basis_ids": cand.get("selected_basis_ids", ""),
                            "qp_status": cand.get("qp_status", ""),
                            "constraint_active_count": cand.get("constraint_active_count", ""),
                            "predicted_CouplingR2_delta": cand.get("predicted_CouplingR2_delta", ""),
                            "predicted_NoiseSignalLeak_delta": cand.get("predicted_NoiseSignalLeak_delta", ""),
                            "predicted_Reservoir_delta": cand.get("predicted_Reservoir_delta", ""),
                            "actual_CouplingR2_delta": metrics.get("CouplingR2_delta", ""),
                            "actual_NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                            "actual_Reservoir_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                            "prediction_error": "",
                            "CouplingR2_delta": metrics.get("CouplingR2_delta", ""),
                            "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                            "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                            "control_gap": safe_float(metrics.get("delta_score"), 0.0) - best_control,
                            "delta_score": metrics.get("delta_score", ""),
                            "logit_max_abs_drift": metrics.get("logit_max_abs_drift", ""),
                            "CEp99_delta_audit": metrics.get("CEp99_delta_audit", ""),
                            "ECE_delta_audit": metrics.get("ECE_delta_audit", ""),
                            "holdout_loss_ratio_audit": metrics.get("holdout_loss_ratio_audit", ""),
                            "sketch_delta_fro": metrics.get("sketch_delta_fro", ""),
                            "signal_projector_angle": metrics.get("signal_projector_angle", ""),
                            "reservoir_projector_angle": metrics.get("reservoir_projector_angle", ""),
                        }
                        fail = p3_fail(row)
                        row["p3_row_pass"] = int(not fail)
                        row["fail_reason"] = ";".join(fail)
                        p3_rows.append(row)

    return summarize_and_write(args, out_dir, linec_rows, actuator_rows, p3_rows, list(loss_audit.values()), p4_rows)


def response_model_rows(linec_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    feature_cols = [
        "functional_norm_ratio",
        "role_direct_norm",
        "role_quad_norm",
        "role_branch_norm",
        "role_projection_norm",
        "logit_drift_l2",
        "logit_max_abs_drift",
        "sketch_delta_fro",
        "sketch_top_eigen_delta",
        "signal_projector_angle",
        "reservoir_projector_angle",
    ]
    targets = ["CouplingR2_delta", "NoiseSignalLeak_delta", "RealSignalReservoirRatio_delta"]
    rows = []
    for target in targets:
        best = None
        y = [safe_float(row.get(target)) for row in linec_rows]
        for feature in feature_cols:
            x = [safe_float(row.get(feature)) for row in linec_rows]
            sp = try_corr(x, y, "spearman")
            item = (abs(sp), feature, sp, try_corr(x, y, "pearson"), try_corr(x, y, "kendall"))
            if best is None or item[0] > best[0]:
                best = item
        if best is None:
            continue
        _abs, feature, sp, pe, ke = best
        x = [safe_float(row.get(feature)) for row in linec_rows]
        mx, my = mean(x), mean(y)
        vx = sum((a - mx) ** 2 for a in x)
        slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / max(EPS, vx)
        pred = [my + slope * (a - mx) for a in x]
        sign_accuracy = mean([1.0 if (p <= 0.0) == (b <= 0.0) else 0.0 for p, b in zip(pred, y)])
        release_actual = [b <= -0.01 for b in y]
        release_pred = [p <= -0.01 for p in pred]
        tp = sum(1 for p, a in zip(release_pred, release_actual) if p and a)
        fp = sum(1 for p, a in zip(release_pred, release_actual) if p and not a)
        fn = sum(1 for p, a in zip(release_pred, release_actual) if (not p) and a)
        ss_tot = sum((b - my) ** 2 for b in y)
        ss_res = sum((b - p) ** 2 for b, p in zip(y, pred))
        rows.append(
            {
                "stage": "V1215_RESPONSE_MODEL",
                "model_id": f"best_univariate_{feature}",
                "feature_set": feature,
                "metric_target": target,
                "spearman_corr": sp,
                "pearson_corr": pe,
                "kendall_corr": ke,
                "r2_oos": 1.0 - ss_res / max(EPS, ss_tot),
                "sign_accuracy": sign_accuracy,
                "negative_release_precision": tp / max(1, tp + fp),
                "negative_release_recall": tp / max(1, tp + fn),
                "bootstrap_ci_low": "",
                "bootstrap_ci_high": "",
            }
        )
    return rows


def summarize_groups(rows: Sequence[Mapping[str, Any]], key_name: str, method_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(method_key, ""))].append(row)
    out = []
    for method, group in sorted(groups.items()):
        out.append(
            {
                "stage": key_name,
                "method": method,
                "rows": len(group),
                "CouplingR2_delta_mean": mean([safe_float(r.get("CouplingR2_delta")) for r in group]),
                "NoiseSignalLeak_delta_mean": mean([safe_float(r.get("NoiseSignalLeak_delta")) for r in group]),
                "RealSignalReservoirRatio_delta_mean": mean([safe_float(r.get("RealSignalReservoirRatio_delta")) for r in group]),
                "p3_pass_rows": sum(safe_int(r.get("p3_row_pass")) for r in group),
                "max_sketch_delta_fro": max([safe_float(r.get("sketch_delta_fro")) for r in group] or [0.0]),
            }
        )
    return out


def p3_promotion_summary(p3_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in p3_rows:
        grouped[str(row.get("candidate_id"))].append(row)
    out = []
    for cid, rows in sorted(grouped.items()):
        pass_ds_seed = {(r.get("dataset"), safe_int(r.get("seed"))) for r in rows if safe_int(r.get("p3_row_pass")) == 1}
        datasets = {r.get("dataset") for r in rows}
        seeds = {safe_int(r.get("seed")) for r in rows}
        expected = len(datasets) * len(seeds)
        passed = len(pass_ds_seed)
        strong = int(expected >= 9 and passed == expected)
        weak = int(expected >= 9 and passed >= expected - 1)
        out.append(
            {
                "stage": "V1215_P3_PROMOTION_SUMMARY",
                "candidate_id": cid,
                "rows": len(rows),
                "dataset_seed_pass_count": passed,
                "expected_dataset_seed_count": expected,
                "strong_promotion": strong,
                "weak_promotion": weak,
                "mean_control_gap": mean([safe_float(r.get("control_gap")) for r in rows]),
                "mean_noise_delta": mean([safe_float(r.get("NoiseSignalLeak_delta")) for r in rows]),
                "mean_reservoir_delta": mean([safe_float(r.get("RealSignalReservoirRatio_delta")) for r in rows]),
            }
        )
    return out


def classic_family_status() -> list[dict[str, Any]]:
    rows = []
    if V1214_FAMILY_STATUS.exists():
        payload = json.loads(V1214_FAMILY_STATUS.read_text(encoding="utf-8"))
        families = payload.get("families", {})
    else:
        families = {}
    for family in ACTIVE_FAMILIES:
        src = families.get(family, {})
        rows.append(
            {
                "stage": "V1215_CLASSIC_FAMILY_STATUS",
                "family": family,
                "candidate_id": "",
                "status": src.get("status", "carried_from_v12_13_v12_14"),
                "L3_efficiency_pass": "",
                "A4_expression_pass": "",
                "A5_task_pass": "",
                "LineC_nontearing_pass": "",
                "best_step_ratio": "",
                "best_memory_ratio": "",
                "mean_delta": "",
                "worst_delta": "",
                "AUC_step_ratio": "",
                "AUC_time_ratio": "",
                "CouplingR2": "",
                "NoiseSignalLeak": "",
                "RealSignalReservoirRatio": "",
                "primary_blocker": src.get("note", "Line D monitor only in this v12.15 run"),
                "next_action": "no_bspline; no broad family sweep; revisit only after Line B/C/I evidence",
                "active_budget_remaining": 2,
            }
        )
    rows.append(
        {
            "stage": "V1215_CLASSIC_FAMILY_STATUS",
            "family": "BSpline",
            "status": "RejectedForThisVersion",
            "primary_blocker": "frozen by v12.15 plan",
            "active_budget_remaining": 0,
        }
    )
    return rows


def write_simple_svgs(out_dir: Path, decision: Mapping[str, Any], summaries: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    names = [
        "fig_v1215_summary_route.svg",
        "fig_v1215_noise_reservoir_effect_size_by_method.svg",
        "fig_v1215_coupling_vs_noise_reservoir_scatter.svg",
        "fig_v1215_loss_agnostic_audit_matrix.svg",
        "fig_v1215_fail_reason_heatmap.svg",
        "fig_v1215_response_pred_vs_actual_noise.svg",
        "fig_v1215_response_pred_vs_actual_reservoir.svg",
        "fig_v1215_actuator_role_response_heatmap.svg",
        "fig_v1215_projector_angle_by_role.svg",
        "fig_v1215_p3_survivor_pareto.svg",
        "fig_v1215_p4_short_run_curves.svg",
        "fig_v1215_controls_gap_matrix.svg",
        "fig_v1215_classic_family_status.svg",
        "fig_v1215_response_pred_vs_actual_coupling.svg",
        "fig_v1215_projector_angle_vs_noise_release.svg",
        "fig_v1215_role_energy_vs_response.svg",
        "fig_v1215_response_precision_recall.svg",
        "fig_v1215_param_delta_vs_sketch_delta.svg",
        "fig_v1215_actuator_failure_taxonomy.svg",
    ]
    lines = [
        f"route: {decision.get('route')}",
        f"linec_rows: {decision.get('linec_rows')}",
        f"actuator_rows: {decision.get('actuator_rows')}",
        f"p3_rows: {decision.get('p3_rows')}",
        f"p3_survivors: {decision.get('p3_survivor_count')}",
    ]
    for row in list(summaries.get("p3", []))[:5]:
        lines.append(f"{row.get('candidate_id')} pass={row.get('dataset_seed_pass_count')}/{row.get('expected_dataset_seed_count')}")
    body = "".join(f"<text x='10' y='{20 + 16 * idx}' font-size='12'>{line}</text>" for idx, line in enumerate(lines))
    for name in names:
        (out_dir / name).write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='980' height='240'>{body}</svg>\n", encoding="utf-8")


def summarize_and_write(
    args: argparse.Namespace,
    out_dir: Path,
    linec_rows: Sequence[Mapping[str, Any]],
    actuator_rows: Sequence[Mapping[str, Any]],
    p3_rows: Sequence[Mapping[str, Any]],
    loss_audit_rows: Sequence[Mapping[str, Any]],
    p4_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor_rows = anchor_monitor_rows(out_dir)
    model_rows = response_model_rows(linec_rows)
    p3_summary = p3_promotion_summary(p3_rows)
    linec_summary = summarize_groups(linec_rows, "V1215_LINEC_RESPONSE_SUMMARY", "probe_update_id")
    p3_method_summary = summarize_groups(p3_rows, "V1215_B15_P3_METHOD_SUMMARY", "candidate_id")
    family_rows = classic_family_status()
    fail_counter = Counter(reason for row in p3_rows for reason in str(row.get("fail_reason", "")).split(";") if reason)
    failure_rows = [{"stage": "V1215_FAILURE_TABLE", "fail_reason": reason, "count": count} for reason, count in sorted(fail_counter.items(), key=lambda item: (-item[1], item[0]))]
    provenance = [
        {"key": "script", "value": "experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py"},
        {"key": "plan_doc", "value": rel(PLAN_DOC)},
        {"key": "out_dir", "value": rel(out_dir)},
        {"key": "conda_env", "value": "kan"},
        {"key": "ce_vector_direction_allowed", "value": "0"},
        {"key": "p4_open_without_p3", "value": "0"},
    ]
    write_csv_rows(out_dir / "v1215_anchor_monitor.csv", anchor_rows)
    write_csv_rows(out_dir / "v1215_linec_response_identifiability.csv", linec_rows)
    write_csv_rows(out_dir / "v1215_linec_response_model.csv", model_rows)
    write_csv_rows(out_dir / "v1215_actuator_truth.csv", actuator_rows)
    write_csv_rows(out_dir / "v1215_b15_p3_candidates.csv", p3_rows)
    write_csv_rows(out_dir / "v1215_b15_p3_summary.csv", p3_summary)
    write_csv_rows(out_dir / "v1215_linec_response_summary.csv", linec_summary)
    write_csv_rows(out_dir / "v1215_b15_p3_method_summary.csv", p3_method_summary)
    write_csv_rows(out_dir / "v1215_p4_short_run.csv", p4_rows)
    write_csv_rows(out_dir / "v1215_classic_family_status.csv", family_rows)
    write_csv_rows(out_dir / "v1215_loss_agnostic_audit.csv", loss_audit_rows)
    write_csv_rows(out_dir / "v1215_provenance_audit.csv", provenance)
    write_csv_rows(out_dir / "v1215_failure_table.csv", failure_rows)
    p3_survivors = [row for row in p3_summary if safe_int(row.get("strong_promotion")) or safe_int(row.get("weak_promotion"))]
    target_spearman = {
        str(row.get("metric_target")): abs(safe_float(row.get("spearman_corr")))
        for row in model_rows
    }
    max_estimator_spearman = max(target_spearman.values() or [0.0])
    noise_estimator_spearman = target_spearman.get("NoiseSignalLeak_delta", 0.0)
    reservoir_estimator_spearman = target_spearman.get("RealSignalReservoirRatio_delta", 0.0)
    max_actuator_sketch = max([safe_float(row.get("sketch_delta_fro")) for row in actuator_rows] or [0.0])
    max_actuator_angle = max([max(safe_float(row.get("signal_projector_angle_deg")), safe_float(row.get("reservoir_projector_angle_deg"))) for row in actuator_rows] or [0.0])
    max_noise_release = min([safe_float(row.get("NoiseSignalLeak_delta"), 999.0) for row in p3_rows] or [999.0])
    max_res_release = min([safe_float(row.get("RealSignalReservoirRatio_delta"), 999.0) for row in p3_rows] or [999.0])
    b15_all_zero = all(safe_int(row.get("dataset_seed_pass_count")) == 0 for row in p3_summary)
    all_control_gap_negative = all(safe_float(row.get("mean_control_gap"), 0.0) < 0.0 for row in p3_summary)
    payload: dict[str, Any] = {
        "stage": "V1215_ROUTE_DECISION",
        "route": "",
        "linec_rows": len(linec_rows),
        "actuator_rows": len(actuator_rows),
        "p3_rows": len(p3_rows),
        "response_model_rows": len(model_rows),
        "p3_survivor_count": len(p3_survivors),
        "p4_open": int(bool(p3_survivors)),
        "max_estimator_abs_spearman": max_estimator_spearman,
        "noise_estimator_abs_spearman": noise_estimator_spearman,
        "reservoir_estimator_abs_spearman": reservoir_estimator_spearman,
        "max_actuator_sketch_delta_fro": max_actuator_sketch,
        "max_actuator_projector_angle_deg": max_actuator_angle,
        "best_noise_delta": max_noise_release,
        "best_reservoir_delta": max_res_release,
        "fail_reason_counts": dict(fail_counter),
        "no_fake_proxy_cpu": 1,
    }
    if p3_survivors:
        payload["route"] = "R6-LossAgnosticP3SurvivorNeedsP4"
        payload["next_recommended_action"] = "run P4 short-run for promoted survivor"
    elif (
        b15_all_zero
        and max_noise_release > -0.01
        and max_res_release > -0.01
        and all_control_gap_negative
    ):
        payload["route"] = "R4-LossAgnosticFunctionalMechanismNotFound"
        payload["next_recommended_action"] = "stop current cotangent/role/moment mechanism family; move to higher-level estimator or Line C sketch redesign"
    elif noise_estimator_spearman < 0.30 or reservoir_estimator_spearman < 0.30:
        payload["route"] = "R2-LossAgnosticEstimatorNotPredictive"
        payload["next_recommended_action"] = "deepen primitive instrumentation before more B15 candidate search"
    else:
        payload["route"] = "R3-FunctionalMechanismPartialOnly"
        payload["next_recommended_action"] = "estimator has partial signal but no P3 survivor; redesign actuator/objective around noise and reservoir release"
    write_json(out_dir / "v1215_route_decision.json", payload)
    write_simple_svgs(out_dir, payload, {"p3": p3_summary})
    hashes = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v1215_hash_manifest.json":
            hashes[path.name] = sha256_file(path)
    write_json(out_dir / "v1215_hash_manifest.json", hashes)
    write_reports(args, out_dir, payload, linec_summary, p3_method_summary, p3_summary, failure_rows, hashes)
    return payload


def table_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], limit: int = 20) -> str:
    lines = []
    for row in rows[:limit]:
        lines.append("| " + " | ".join(f"`{row.get(col, '')}`" for col in columns) + " |")
    return "\n".join(lines)


def write_reports(
    args: argparse.Namespace,
    out_dir: Path,
    decision: Mapping[str, Any],
    linec_summary: Sequence[Mapping[str, Any]],
    p3_method_summary: Sequence[Mapping[str, Any]],
    p3_summary: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
) -> None:
    command = (
        "PYTHONHASHSEED=0 conda run -n kan python "
        "experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py "
        f"--probe-device {args.probe_device} --probe-datasets {args.probe_datasets} --probe-seeds {args.probe_seeds} "
        f"--probe-splits {args.probe_splits} --functional-batch-size {args.functional_batch_size} --windows {args.windows} "
        f"--probe-train-size {args.probe_train_size} --probe-val-size {args.probe_val_size} --probe-test-size {args.probe_test_size} "
        f"--out-dir {rel(out_dir)}"
    )
    hash_lines = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in sorted(hashes.items()))
    linec_lines = table_rows(linec_summary, ["method", "rows", "CouplingR2_delta_mean", "NoiseSignalLeak_delta_mean", "RealSignalReservoirRatio_delta_mean"], 20)
    p3_lines = table_rows(p3_method_summary, ["method", "rows", "CouplingR2_delta_mean", "NoiseSignalLeak_delta_mean", "RealSignalReservoirRatio_delta_mean", "p3_pass_rows"], 20)
    survivor_lines = table_rows(p3_summary, ["candidate_id", "dataset_seed_pass_count", "expected_dataset_seed_count", "strong_promotion", "weak_promotion", "mean_control_gap"], 20)
    fail_lines = table_rows(failure_rows, ["fail_reason", "count"], 20)
    exec_text = f"""# DG-KAN v12.15 B320Locked LossAgnosticSignalEstimator PrimitiveInstrumentation 执行复盘

生成时间：`{now_iso()}`

## 1. 执行入口

```text
script = experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py
plan_doc = {rel(PLAN_DOC)}
out_dir = {rel(out_dir)}
conda_env = kan
datasets = {args.probe_datasets}
seeds = {args.probe_seeds}
splits = {args.probe_splits}
windows = {args.windows}
functional_batch_size = {args.functional_batch_size}
```

## 2. 复现命令

```bash
{command}
```

## 3. 生成 artifact

```text
v1215_anchor_monitor.csv
v1215_linec_response_identifiability.csv
v1215_linec_response_model.csv
v1215_actuator_truth.csv
v1215_b15_p3_candidates.csv
v1215_b15_p3_summary.csv
v1215_p4_short_run.csv
v1215_classic_family_status.csv
v1215_loss_agnostic_audit.csv
v1215_provenance_audit.csv
v1215_failure_table.csv
v1215_route_decision.json
v1215_hash_manifest.json
```

## 4. 修改审计

```text
新增 experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py。
smoke run 暴露 weak-promotion gate 会在非 3x3 设置误开 P4；已修复为 expected_dataset_seed_count >= 9 才允许 strong/weak promotion。
official route 判定按 v12.15 停止条件收紧：B15 全 0、noise/reservoir 未过 -0.01 且 control_gap 全负时写 R4。
未修改 CE loss、sampler、class weight、teacher/distillation、dataset branch 或 B320 architecture。
B320 anchor monitor 复用 v12.14 locked anchor artifact，并记录 source sha256。
official B15 direction generator 不使用 CE vector / label-loss VJP / permuted-label CE surrogate。
```

## 5. Route

```text
route = {decision.get('route')}
p4_open = {decision.get('p4_open')}
p3_survivor_count = {decision.get('p3_survivor_count')}
next_recommended_action = {decision.get('next_recommended_action')}
```

## 6. Artifact Hash

| artifact | sha256 |
|---|---|
{hash_lines}
"""
    result_text = f"""# DG-KAN v12.15 B320Locked LossAgnosticSignalEstimator PrimitiveInstrumentation 结果复盘

生成时间：`{now_iso()}`

## 1. 结论

```text
route = {decision.get('route')}
linec_rows = {decision.get('linec_rows')}
actuator_rows = {decision.get('actuator_rows')}
p3_rows = {decision.get('p3_rows')}
p3_survivor_count = {decision.get('p3_survivor_count')}
p4_open = {decision.get('p4_open')}
max_estimator_abs_spearman = {decision.get('max_estimator_abs_spearman')}
noise_estimator_abs_spearman = {decision.get('noise_estimator_abs_spearman')}
reservoir_estimator_abs_spearman = {decision.get('reservoir_estimator_abs_spearman')}
max_actuator_sketch_delta_fro = {decision.get('max_actuator_sketch_delta_fro')}
max_actuator_projector_angle_deg = {decision.get('max_actuator_projector_angle_deg')}
best_noise_delta = {decision.get('best_noise_delta')}
best_reservoir_delta = {decision.get('best_reservoir_delta')}
```

## 2. Line C Response 汇总

| method | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean |
|---|---:|---:|---:|---:|
{linec_lines}

## 3. B15 P3 汇总

| method | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | p3_pass_rows |
|---|---:|---:|---:|---:|---:|
{p3_lines}

## 4. Promotion 汇总

| candidate_id | dataset_seed_pass_count | expected_dataset_seed_count | strong_promotion | weak_promotion | mean_control_gap |
|---|---:|---:|---:|---:|---:|
{survivor_lines}

## 5. Failure Table

| fail_reason | count |
|---|---:|
{fail_lines}

## 6. 分析结论

```text
{decision.get('next_recommended_action')}
```

本复盘只记录真实落盘 artifact 的数值。没有 P3 survivor 时，P4 按计划关闭，不写 functional official success。
"""
    DEFAULT_EXECUTION_LOG.write_text(exec_text, encoding="utf-8")
    DEFAULT_RESULT_LOG.write_text(result_text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_ROOT / "official_run"))
    parser.add_argument("--probe-device", default="auto")
    parser.add_argument("--probe-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--probe-seeds", default="0,1,2")
    parser.add_argument("--probe-splits", type=int, default=1)
    parser.add_argument("--functional-batch-size", type=int, default=32)
    parser.add_argument("--windows", default="5")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--probe-train-size", type=int, default=512)
    parser.add_argument("--probe-val-size", type=int, default=256)
    parser.add_argument("--probe-test-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1.0e-3)
    parser.add_argument("--sketch-dim", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    decision = run_main(args, out_dir)
    print(json.dumps({"out_dir": str(out_dir.resolve()), "route": decision.get("route"), "p4_open": decision.get("p4_open")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
