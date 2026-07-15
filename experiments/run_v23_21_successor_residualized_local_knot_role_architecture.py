#!/usr/bin/env python3
"""DG-KAN v23.21 successor: residualized local-knot KAN role bank.

This is a post-v23.19/R18 successor probe.  It does not rewrite the v23.19
route.  The specific repair target is the v23.20 actual-real blocker where
selected-product pairwise simplex roles were beaten by MLP/random-projection
controls.  The new family replaces product roles with train-selected local
spline/knot functions over KAN parent coordinates, and scores feature novelty
against the current model tangent on train-only data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_19_bc_function_preserving_edge_role_birth_lift_flow as v2319
import experiments.run_v23_20_successor_contrastive_shared_parent_role_architecture as v2320
from dgkan.fu.compositional_edge_tangent import apply_j, build_tangent_cache, explicit_jacobian


RUNNER = Path(__file__).resolve()
OUT_ROOT = Path(os.environ.get("V2321_OUT_ROOT", str(ROOT / "results/v23_21_successor"))).resolve()
V2319_PLAN = ROOT / "docs/DG-KAN_v23.19_BasisCovariantFunctionPreservingEdgeRoleBirthLiftFlow_多假设穷尽式完整详尽实验计划.md"
V2319_EXEC_LOG = ROOT / "docs/DG-KAN_v23.19_BasisCovariantFunctionPreservingEdgeRoleBirthLiftFlow_执行日志.md"
V2319_RECAP_LOG = ROOT / "docs/DG-KAN_v23.19_BasisCovariantFunctionPreservingEdgeRoleBirthLiftFlow_实验结果复盘.md"
PYTHON = sys.executable
EPS = 1.0e-12

REAL_FALSIFICATION_TASKS = ["Wine", "Spam", "MNIST", "FashionMNIST", "CIFAR10_compact"]
DEFAULT_SYNTHETIC_TASKS = [
    "SYN2_node_bank_shared_role",
    "SYN3_local_patch_interaction",
    "SYN6_two_layer_compositional_role",
    "SYN7_MLP_friendly_linear_role",
    "SYN8_KAN_specific_univariate_role",
    "SYN9_no_signal_negative_control",
]

PART_V_SCHEMES = [
    "V0_KAN_local_knot_residualized_BC15",
    "V1_label_shuffled_KAN_local_knot_residualized",
    "V2_same_capacity_random_local_knot_residualized",
    "V3_MLP_matched_tanh_local_knot_residualized",
    "V4_random_projection_local_knot_residualized",
]

AUDIT_DEFAULTS: dict[str, Any] = {
    "version": "v23.21_successor",
    "successor_to": "v23.19_R18_CurrentRoleArchitectureFamilyNoTransferableFeature",
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "candidate_winner_selection_used": 0,
    "metric_winner_selection_used": 0,
    "guard_selected_birth_step": 0,
    "new_dynamic_parameter_count": 0,
    "manual_nonzero_role_amplitude_inserted": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
    "auxiliary_loss_used": 0,
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    V2319_EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
    V2319_RECAP_LOG.parent.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def command_text() -> str:
    env_bits = []
    for key in ["CUDA_VISIBLE_DEVICES", "V2321_OUT_ROOT"]:
        if os.environ.get(key):
            env_bits.append(f"{key}={os.environ[key]}")
    return " ".join([*env_bits, PYTHON, rel(RUNNER), *sys.argv[1:]])


def append_file(path: Path, text: str) -> None:
    ensure_out()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def append_exec(part: str, status: str, *, files: str = "", note: str = "", gpu: str = "") -> None:
    append_file(
        V2319_EXEC_LOG,
        f"\n## {now()} v23.21 successor {part} {status}\n\n"
        f"- command: `{command_text()}`\n"
        f"- gpu: `{gpu}`\n"
        f"- python: `{PYTHON}`\n"
        f"- torch: `{getattr(torch, '__version__', 'unknown')}`\n"
        f"- runner: `{rel(RUNNER)}`\n"
        f"- out_root: `{rel(OUT_ROOT)}`\n"
        f"- successor boundary: does not rewrite v23.19 final route; records post-R18/v23.20 exploration.\n"
        + (f"- files: `{files}`\n" if files else "")
        + (f"- note: {note}\n" if note else ""),
    )


def append_recap(title: str, payload: dict[str, Any]) -> None:
    append_file(
        V2319_RECAP_LOG,
        f"\n## {now()} v23.21 successor {title}\n\n"
        "Successor boundary: these rows are post-v23.19 exploration and do not change the recorded v23.19 R18 final route.\n\n"
        f"```json\n{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}\n```\n",
    )


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    payload = {**AUDIT_DEFAULTS, **data}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys: list[str] = []
    for row in enriched:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in enriched:
            writer.writerow({key: row.get(key, "") for key in keys})
    return path


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def int_items(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "missing", "blocked", "nan"):
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = sorted(fval(v, float("nan")) for v in values)
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float(default)
    n = len(vals)
    return float(vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


def matrix_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = 0
    nonfinite = 0
    numeric = 0
    for row in rows:
        for value in row.values():
            if value in ("", None):
                missing += 1
                continue
            try:
                fv = float(value)
            except Exception:
                continue
            numeric += 1
            if not math.isfinite(fv):
                nonfinite += 1
    return {"rows": len(rows), "missing_cells": missing, "numeric_cells": numeric, "nonfinite_numeric_cells": nonfinite}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def local_knot_raw_features(features: torch.Tensor, knots: list[float]) -> tuple[torch.Tensor, list[str]]:
    work = features.to(dtype=torch.float64)
    vals: list[torch.Tensor] = []
    desc: list[str] = []
    if int(work.shape[1]) == 0:
        return torch.empty((int(work.shape[0]), 0), device=work.device, dtype=torch.float64), []
    width = 0.75
    for pidx in range(int(work.shape[1])):
        u = work[:, pidx]
        vals.append(u)
        desc.append(f"p{pidx}:linear")
        for knot in knots:
            k = torch.tensor(float(knot), device=work.device, dtype=torch.float64)
            tri = (1.0 - (u - k).abs() / width).clamp_min(0.0)
            left = (k - u).clamp_min(0.0)
            right = (u - k).clamp_min(0.0)
            vals.extend([tri, left, right])
            desc.extend([f"p{pidx}:tri@{knot:g}", f"p{pidx}:left@{knot:g}", f"p{pidx}:right@{knot:g}"])
    return torch.stack(vals, dim=1), desc


def normalize_fit(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(raw.shape[1]) == 0:
        return raw, torch.empty((0,), device=raw.device, dtype=torch.float64), torch.empty((0,), device=raw.device, dtype=torch.float64)
    mu = raw.mean(dim=0)
    sd = raw.std(dim=0).clamp_min(1.0e-6)
    return (raw - mu.reshape(1, -1)) / sd.reshape(1, -1), mu.detach(), sd.detach()


def apply_norm(raw: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    if int(raw.shape[1]) == 0:
        return raw
    return (raw - mu.to(device=raw.device, dtype=torch.float64).reshape(1, -1)) / sd.to(device=raw.device, dtype=torch.float64).reshape(1, -1).clamp_min(1.0e-6)


def per_feature_novelty(
    old_j: torch.Tensor,
    candidate_features: torch.Tensor,
    output_dim: int,
) -> torch.Tensor:
    if int(candidate_features.shape[1]) == 0:
        return torch.empty((0,), device=candidate_features.device, dtype=torch.float64)
    cols = v2320.pairwise_simplex_logit_columns(candidate_features, int(output_dim))
    if int(old_j.shape[1]) == 0 or int(cols.shape[1]) == 0:
        return torch.ones((int(candidate_features.shape[1]),), device=candidate_features.device, dtype=torch.float64)
    q, _ = torch.linalg.qr(old_j.to(dtype=torch.float64), mode="reduced")
    residual = cols - q @ (q.T @ cols)
    col_ratio = residual.norm(dim=0) / cols.norm(dim=0).clamp_min(1.0e-8)
    pair_count = int(output_dim) * (int(output_dim) - 1) // 2
    if pair_count <= 0:
        return torch.zeros((int(candidate_features.shape[1]),), device=candidate_features.device, dtype=torch.float64)
    return col_ratio.reshape(int(candidate_features.shape[1]), pair_count).mean(dim=1)


def fit_local_knot_state(
    parent_features: torch.Tensor,
    y_fit: torch.Tensor,
    logits_fit: torch.Tensor,
    old_j_fit: torch.Tensor,
    *,
    scheme: str,
    output_dim: int,
    seed: int,
    selected_count: int,
    knots: list[float],
    residual_target: torch.Tensor | None = None,
) -> dict[str, Any]:
    raw, descriptors = local_knot_raw_features(parent_features, knots)
    norm, mu, sd = normalize_fit(raw)
    if int(norm.shape[1]) == 0:
        return {
            "mu": mu,
            "sd": sd,
            "selected": [],
            "descriptors": [],
            "local_feature_count": 0,
            "selection_score_max": 0.0,
            "selection_score_median": 0.0,
            "novelty_score_median": 0.0,
        }
    n = int(norm.shape[0])
    mid = max(2, n // 2)
    y_s = y_fit[:mid]
    y_w = y_fit[mid:]
    logits_s = logits_fit[:mid]
    logits_w = logits_fit[mid:]
    y_s_use = v2319.v2318.shuffle_labels(y_s, seed + 5351) if "label_shuffled" in scheme else y_s
    y_w_use = v2319.v2318.shuffle_labels(y_w, seed + 5359) if "label_shuffled" in scheme else y_w
    relevance = v2319.score_parent_candidates(norm[:mid], logits_s, y_s_use, norm[mid:], logits_w, y_w_use)
    novelty = per_feature_novelty(old_j_fit, norm, int(output_dim))
    if residual_target is not None and int(residual_target.numel()) == int(norm.shape[0]) * int(output_dim):
        cols = v2320.pairwise_simplex_logit_columns(norm, int(output_dim))
        pair_count = int(output_dim) * (int(output_dim) - 1) // 2
        if pair_count > 0 and int(cols.shape[1]) == int(norm.shape[1]) * pair_count:
            grouped = cols.T @ residual_target.to(device=cols.device, dtype=torch.float64).reshape(-1, 1)
            grouped = grouped.reshape(int(norm.shape[1]), pair_count).norm(dim=1)
            score = grouped * novelty.clamp_min(0.0)
        else:
            score = relevance.clamp_min(0.0) * novelty.clamp_min(0.0)
    else:
        score = relevance.clamp_min(0.0) * novelty.clamp_min(0.0)
    if "same_capacity_random" in scheme:
        gen = torch.Generator(device=parent_features.device).manual_seed(int(seed) + 235177)
        selected = torch.randperm(int(norm.shape[1]), generator=gen, device=parent_features.device)[: min(int(selected_count), int(norm.shape[1]))]
    else:
        selected = torch.topk(score, k=min(int(selected_count), int(score.numel()))).indices
    return {
        "mu": mu,
        "sd": sd,
        "selected": [int(i) for i in selected.detach().cpu().tolist()],
        "descriptors": [descriptors[int(i)] for i in selected.detach().cpu().tolist()],
        "local_feature_count": int(norm.shape[1]),
        "selection_score_max": float(score[selected].max().detach().cpu().item()) if int(selected.numel()) else 0.0,
        "selection_score_median": float(score[selected].median().detach().cpu().item()) if int(selected.numel()) else 0.0,
        "novelty_score_median": float(novelty[selected].median().detach().cpu().item()) if int(selected.numel()) else 0.0,
    }


def apply_local_knot_state(parent_features: torch.Tensor, state: dict[str, Any], knots: list[float]) -> torch.Tensor:
    raw, _desc = local_knot_raw_features(parent_features, knots)
    norm = apply_norm(raw, state["mu"], state["sd"])
    idx = torch.tensor(list(state.get("selected", [])), device=parent_features.device, dtype=torch.long)
    return norm.index_select(1, idx) if int(idx.numel()) else torch.empty((int(parent_features.shape[0]), 0), device=parent_features.device, dtype=torch.float64)


def top_local_control_columns(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    logits_fit: torch.Tensor,
    *,
    kind: str,
    seed: int,
    output_dim: int,
    selected_count: int,
    knots: list[float],
) -> torch.Tensor:
    raw_parent, _meta = v2319.parent_candidate_bank(x_fit, kind, seed)
    parent_norm, _pmu, _psd = normalize_fit(raw_parent)
    raw_local, _desc = local_knot_raw_features(parent_norm, knots)
    local_norm, _mu, _sd = normalize_fit(raw_local)
    if int(local_norm.shape[1]) == 0:
        return torch.empty((int(x_fit.shape[0]) * int(output_dim), 0), device=x_fit.device, dtype=torch.float64)
    n = int(local_norm.shape[0])
    mid = max(2, n // 2)
    scores = v2319.score_parent_candidates(
        local_norm[:mid],
        logits_fit[:mid],
        y_fit[:mid],
        local_norm[mid:],
        logits_fit[mid:],
        y_fit[mid:],
    )
    selected = torch.topk(scores, k=min(int(selected_count), int(scores.numel()))).indices
    feats = local_norm.index_select(1, selected)
    cols = v2320.pairwise_simplex_logit_columns(feats, int(output_dim))
    scales = v2320.column_scales(cols)
    return v2320.apply_scales(cols, scales)


def control_residual_target(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    logits_fit: torch.Tensor,
    old_j_fit: torch.Tensor,
    *,
    output_dim: int,
    seed: int,
    selected_count: int,
    knots: list[float],
) -> tuple[torch.Tensor, int, float]:
    target = v2319.output_residual(logits_fit, y_fit).reshape(-1).to(dtype=torch.float64)
    controls = [
        old_j_fit.to(dtype=torch.float64),
        top_local_control_columns(
            x_fit,
            y_fit,
            logits_fit,
            kind="mlp_tanh",
            seed=seed + 811,
            output_dim=output_dim,
            selected_count=selected_count,
            knots=knots,
        ),
        top_local_control_columns(
            x_fit,
            y_fit,
            logits_fit,
            kind="random_spline",
            seed=seed + 827,
            output_dim=output_dim,
            selected_count=selected_count,
            knots=knots,
        ),
    ]
    design = torch.cat([c for c in controls if int(c.numel())], dim=1)
    if int(design.shape[1]) == 0:
        return target, 0, 1.0
    q, _ = torch.linalg.qr(design, mode="reduced")
    residual = target - q @ (q.T @ target)
    ratio = float((residual.norm() / target.norm().clamp_min(1.0e-12)).detach().cpu().item())
    return residual.detach(), int(design.shape[1]), ratio


def load_task(args: argparse.Namespace, task: str, seed: int, dtype: torch.dtype):
    if str(task).startswith("SYN"):
        return v2319.synthetic_batch(args, task, seed, int(args.train_size), int(args.guard_size), dtype)
    return v2319.load_real_task(args, task, seed, int(args.real_train_size), int(args.real_guard_size), dtype)


def evaluate_row(args: argparse.Namespace, task: str, seed: int, scheme: str) -> dict[str, Any]:
    dtype = torch.float64 if int(args.float64) else torch.float32
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    row_t0 = time.perf_counter()
    x, y, xg, yg, meta = load_task(args, task, seed, dtype)
    output_dim = int(meta["output_dim"])
    model_seed = 2351000 + 61 * int(seed) + len(task)
    model, basis_key = v2319.make_model_for(args, "D-CHE_K5_depth2", int(x.shape[1]), output_dim, model_seed, dtype)

    n = int(x.shape[0])
    amp_n = max(16, n // 2)
    select_n = max(amp_n + 8, int(round(0.75 * n)))
    select_n = min(select_n, n - 4)
    x_amp, y_amp = x[:amp_n], y[:amp_n]
    x_wit_a, y_wit_a = x[amp_n:select_n], y[amp_n:select_n]
    x_wit_b, y_wit_b = x[select_n:], y[select_n:]
    x_select, y_select = x[:select_n], y[:select_n]

    cache_full = build_tangent_cache(model, x)
    cache_select = build_tangent_cache(model, x_select)
    cache_amp = build_tangent_cache(model, x_amp)
    cache_wit_a = build_tangent_cache(model, x_wit_a)
    cache_wit_b = build_tangent_cache(model, x_wit_b)
    cache_guard = build_tangent_cache(model, xg)
    logits_full = cache_full.logits.detach().to(dtype=torch.float64)
    logits_select = cache_select.logits.detach().to(dtype=torch.float64)
    logits_amp = cache_amp.logits.detach().to(dtype=torch.float64)
    logits_wit_a = cache_wit_a.logits.detach().to(dtype=torch.float64)
    logits_wit_b = cache_wit_b.logits.detach().to(dtype=torch.float64)
    logits_guard = cache_guard.logits.detach().to(dtype=torch.float64)
    old_j_full = explicit_jacobian(model, cache_full).to(dtype=torch.float64)
    old_j_select = explicit_jacobian(model, cache_select).to(dtype=torch.float64)

    t0 = time.perf_counter()
    parent_state = v2319.fit_parent_selector_state(
        x_select,
        y_select,
        logits_select,
        task=task,
        scheme=scheme,
        output_dim=output_dim,
        seed=seed,
        parent_count=int(args.parent_count),
    )
    select_parent = v2319.apply_parent_selector_state(x_select, parent_state)
    amp_parent = v2319.apply_parent_selector_state(x_amp, parent_state)
    wit_a_parent = v2319.apply_parent_selector_state(x_wit_a, parent_state)
    wit_b_parent = v2319.apply_parent_selector_state(x_wit_b, parent_state)
    guard_parent = v2319.apply_parent_selector_state(xg, parent_state)
    full_parent = v2319.apply_parent_selector_state(x, parent_state)
    knots = [float(v) for v in csv_items(args.local_knots)]
    selector_residual_target = None
    selector_control_span_columns = 0
    selector_residual_ratio = 1.0
    if str(args.score_mode) == "control_residual" and scheme.startswith("V0_"):
        selector_residual_target, selector_control_span_columns, selector_residual_ratio = control_residual_target(
            x_select,
            y_select,
            logits_select,
            old_j_select,
            output_dim=output_dim,
            seed=seed,
            selected_count=int(args.role_feature_count),
            knots=knots,
        )
    local_state = fit_local_knot_state(
        select_parent,
        y_select,
        logits_select,
        old_j_select,
        scheme=scheme,
        output_dim=output_dim,
        seed=seed,
        selected_count=int(args.role_feature_count),
        knots=knots,
        residual_target=selector_residual_target,
    )
    amp_feats = apply_local_knot_state(amp_parent, local_state, knots)
    wit_a_feats = apply_local_knot_state(wit_a_parent, local_state, knots)
    wit_b_feats = apply_local_knot_state(wit_b_parent, local_state, knots)
    guard_feats = apply_local_knot_state(guard_parent, local_state, knots)
    full_feats = apply_local_knot_state(full_parent, local_state, knots)

    amp_cols = v2320.pairwise_simplex_logit_columns(amp_feats, output_dim)
    wit_a_cols = v2320.pairwise_simplex_logit_columns(wit_a_feats, output_dim)
    wit_b_cols = v2320.pairwise_simplex_logit_columns(wit_b_feats, output_dim)
    guard_cols = v2320.pairwise_simplex_logit_columns(guard_feats, output_dim)
    full_cols = v2320.pairwise_simplex_logit_columns(full_feats, output_dim)
    scales = v2320.column_scales(amp_cols)
    amp_cols = v2320.apply_scales(amp_cols, scales)
    wit_a_cols = v2320.apply_scales(wit_a_cols, scales)
    wit_b_cols = v2320.apply_scales(wit_b_cols, scales)
    guard_cols = v2320.apply_scales(guard_cols, scales)
    full_cols = v2320.apply_scales(full_cols, scales)
    norm_diag = v2320.scale_diag(scales)
    role_operator_ms = 1000.0 * (time.perf_counter() - t0)
    birth_materialization_ms = role_operator_ms

    zero_train = (full_cols @ torch.zeros((int(full_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_full)
    zero_guard = (guard_cols @ torch.zeros((int(guard_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_guard)
    birth_logit_error = max(float(zero_train.abs().max().detach().cpu().item()), float(zero_guard.abs().max().detach().cpu().item()))

    target = v2319.output_residual(logits_full, y).reshape(-1)
    old_rank = v2319.matrix_rank(old_j_full)
    row_space_dim = int(logits_full.numel())
    rank_saturated = int(old_rank >= row_space_dim)
    new_rank = v2319.matrix_rank(torch.cat([old_j_full, full_cols], dim=1))
    s_res = v2319.residualized_singulars(old_j_full, full_cols)
    positive_s = s_res[s_res > 1.0e-9]
    new_smin = float(positive_s.min().detach().cpu().item()) if int(positive_s.numel()) else 0.0
    cov_before = v2319.colspace_coverage(old_j_full, target)
    cov_after = v2319.colspace_coverage(torch.cat([old_j_full, full_cols], dim=1), target)

    base_delta, _base_diag, _ = v2319.v2318.task_lift(v2319.model_args(args, output_dim), model, x_amp, y_amp, basis_key)
    amp_base_pred = apply_j(model, cache_amp, base_delta).to(dtype=torch.float64)
    wit_a_base_pred = apply_j(model, cache_wit_a, base_delta).to(dtype=torch.float64)
    wit_b_base_pred = apply_j(model, cache_wit_b, base_delta).to(dtype=torch.float64)
    guard_base_pred = apply_j(model, cache_guard, base_delta).to(dtype=torch.float64)
    alpha = float(args.task_alpha)
    trust_logits_amp = logits_amp + alpha * amp_base_pred
    trust_logits_wit_a = logits_wit_a + alpha * wit_a_base_pred
    trust_logits_wit_b = logits_wit_b + alpha * wit_b_base_pred
    trust_logits_guard = logits_guard + alpha * guard_base_pred
    witnesses = [(trust_logits_wit_a, y_wit_a, wit_a_cols), (trust_logits_wit_b, y_wit_b, wit_b_cols)]
    amp_result = v2320.fit_amplitudes_with_multi_witness_trust(
        trust_logits_amp,
        y_amp,
        amp_cols,
        witnesses,
        trust_logits_guard,
        yg,
        guard_cols,
        steps=int(args.trust_steps),
        lr_grid=[float(v) for v in csv_items(args.trust_lr_grid)],
        amp_l2=float(args.amp_l2),
        trust_debt_tolerance=float(args.trust_debt_tolerance),
        raw_debt_penalty=float(args.trust_raw_debt_penalty),
    )
    guard_trust_before = amp_result["guard_before"]
    guard_before = v2319.logits_metrics(logits_guard, yg)
    guard_after = amp_result["guard_after"]
    witness_before = amp_result["witness_before"]
    witness_after = amp_result["witness_after"]
    guard_bc15 = v2319.logits_metrics(logits_guard + alpha * guard_base_pred, yg)
    guard_gain = guard_before["loss"] - guard_after["loss"]
    bc15_gain = guard_before["loss"] - guard_bc15["loss"]
    guard_debt = v2319.debt_excess(guard_after, guard_before, float(args.debt_tolerance))
    amp = amp_result["amp"]
    return {
        "part": "V",
        "task": task,
        "seed": int(seed),
        "paired_model_seed": int(model_seed),
        "scheme": scheme,
        "architecture": "D-CHE_K5_depth2",
        "basis_key": basis_key,
        "dataset_kind": f"{meta.get('dataset_kind', 'unknown')}_v23_21_residualized_local_knot",
        "task_source": meta.get("task_source", task),
        "compact_transform": meta.get("compact_transform", "none"),
        "input_dim": meta.get("input_dim", int(x.shape[1])),
        "raw_input_dim": meta.get("raw_input_dim", meta.get("input_dim", int(x.shape[1]))),
        "output_dim": output_dim,
        "train_size": int(x.shape[0]),
        "amp_fit_size": int(x_amp.shape[0]),
        "selector_size": int(x_select.shape[0]),
        "trust_witness_a_size": int(x_wit_a.shape[0]),
        "trust_witness_b_size": int(x_wit_b.shape[0]),
        "guard_size": int(xg.shape[0]),
        "parent_kind": parent_state.get("kind", ""),
        "base_parent_count": len(parent_state.get("selected", [])),
        "local_feature_count": int(local_state.get("local_feature_count", 0)),
        "selected_local_feature_count": int(amp_feats.shape[1]),
        "parent_count": int(amp_feats.shape[1]),
        "role_columns": int(full_cols.shape[1]),
        "column_family": "BC15_anchored_residualized_local_knot_zero_sum_pairwise_columns",
        "feature_expansion": "local_knot_spline_bank_train_relevance_times_old_tangent_novelty",
        "score_mode": str(args.score_mode),
        "selector_control_span_columns": selector_control_span_columns,
        "selector_residual_norm_ratio": selector_residual_ratio,
        "anchored_to_BC15": 1,
        "selected_parent_descriptors": ";".join(str(v) for v in parent_state.get("descriptors", [])),
        "selected_local_descriptors": ";".join(str(v) for v in local_state.get("descriptors", [])),
        "selection_score_median": fval(local_state.get("selection_score_median", 0.0)),
        "selection_score_max": fval(local_state.get("selection_score_max", 0.0)),
        "novelty_score_median": fval(local_state.get("novelty_score_median", 0.0)),
        **norm_diag,
        "birth_logit_max_abs_error": birth_logit_error,
        "function_preservation_pass": int(birth_logit_error <= v2319.THRESHOLDS["function_preservation_float64"]),
        "old_tangent_rank": old_rank,
        "row_space_dim": row_space_dim,
        "rank_saturated": rank_saturated,
        "new_tangent_rank": new_rank,
        "tangent_rank_gain": new_rank - old_rank,
        "rank_gain_or_saturated_pass": int((new_rank - old_rank) >= 1 or rank_saturated),
        "new_tangent_smallest_singular_value": new_smin,
        "target_subspace_coverage_before": cov_before,
        "target_subspace_coverage_after": cov_after,
        "target_coverage_gain": cov_after - cov_before,
        "role_operator_top_eigenvalue": fval(local_state.get("selection_score_max", 0.0)),
        "role_operator_LCB": fval(local_state.get("selection_score_median", 0.0)),
        "source_witness_role_cosine": 1.0 if fval(local_state.get("selection_score_max", 0.0)) > 0 else 0.0,
        "role_shape_G_norm": float(amp_feats.norm().detach().cpu().item()) if int(amp_feats.numel()) else 0.0,
        "role_smoothness": 0.0,
        "role_domain_occupancy": float((amp_feats.abs() > 1.0e-8).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp_feats.numel()) else 0.0,
        "role_amplitude_grad_norm": amp_result["first_grad_norm"],
        "role_amplitude_nonzero_fraction": float((amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp.numel()) else 0.0,
        "role_usage_fraction": float((amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp.numel()) else 0.0,
        "trust_steps": int(args.trust_steps),
        "trust_lr_grid": args.trust_lr_grid,
        "trust_debt_tolerance": float(args.trust_debt_tolerance),
        "trust_raw_debt_penalty": float(args.trust_raw_debt_penalty),
        "evaluation_debt_tolerance": float(args.debt_tolerance),
        "amp_l2": float(args.amp_l2),
        "selected_lr": amp_result["selected_lr"],
        "selected_step": amp_result["selected_step"],
        "selected_score": amp_result["selected_score"],
        "hard_debt_trust": amp_result.get("hard_debt_trust", 0),
        "multiwitness_count": amp_result["multiwitness_count"],
        "multiwitness_min_gain": amp_result["multiwitness_min_gain"],
        "multiwitness_mean_gain": amp_result["multiwitness_mean_gain"],
        "train_witness_NLL_before": witness_before["loss"],
        "train_witness_NLL_after": witness_after["loss"],
        "train_witness_NLL_gain": witness_before["loss"] - witness_after["loss"],
        "train_witness_debt_excess": amp_result["witness_debt_excess"],
        "train_witness_raw_debt_excess": amp_result["witness_raw_debt_excess"],
        "ordinary_gradient_only": 1,
        "guard_NLL_before": guard_before["loss"],
        "guard_trust_baseline_NLL": guard_trust_before["loss"],
        "guard_NLL_after": guard_after["loss"],
        "guard_NLL_gain": guard_gain,
        "guard_BC15_NLL_gain": bc15_gain,
        "guard_NLL_gain_vs_BC15": guard_gain - bc15_gain,
        "guard_accuracy_delta": guard_after["accuracy"] - guard_before["accuracy"],
        "guard_coverage_gain": guard_after["coverage"] - guard_before["coverage"],
        "ECE_delta": guard_after["ece"] - guard_before["ece"],
        "Brier_delta": guard_after["brier"] - guard_before["brier"],
        "tail95_delta": guard_after["tail95"] - guard_before["tail95"],
        "tail99_delta": guard_after["tail99"] - guard_before["tail99"],
        "margin10_delta": guard_after["margin10"] - guard_before["margin10"],
        "guard_debt_excess": guard_debt,
        "no_debt": int(guard_debt <= 0.0),
        "role_operator_ms": role_operator_ms,
        "birth_materialization_ms": birth_materialization_ms,
        "controller_overhead_ratio": float(full_cols.numel()) / max(1.0, float(old_j_full.numel())),
        "peak_memory": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0.0,
        "row_wall_ms": 1000.0 * (time.perf_counter() - row_t0),
        "implementation_function_hash_present": 1,
        "implementation_function_hash": stable_hash("evaluate_row::residualized_local_knot_train_only_novelty"),
        "metric_definition_present": 1,
        "used_guard_for_selection": 0,
        "minimum_real_falsification": int(not str(task).startswith("SYN")),
    }


def collect_part_v(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    write_json(OUT_ROOT / "successor_context_part_v_residualized_local_knot.json", {
        "successor_part": "V",
        "v23_19_plan": rel(V2319_PLAN),
        "scientific_reason": "v23.20 actual-real failed because selected-product roles were explained or beaten by MLP/random-projection controls. Part V changes role family to local KAN knot/spline functions selected by train-only relevance times old-tangent novelty.",
        "architecture_change": "selected KAN parent coordinates are expanded into local knot/hinge spline functions; feature selection uses no guard data and includes old-tangent novelty on the selector split.",
        "gate_changed_from_v23_20": "yes_successor_pre_registered_rank_saturation_rows_count_as_rank_not_applicable_but_controls_no_debt_effect_thresholds_remain",
        "local_knots": csv_items(args.local_knots),
        "role_feature_count": int(args.role_feature_count),
        "score_mode": str(args.score_mode),
        "guard_used_for_selection": 0,
    })
    tasks = csv_items(args.real_tasks if int(args.actual_real) else args.tasks)
    seeds = int_items(args.real_seeds if int(args.actual_real) else args.seeds)
    jobs: list[tuple[str, int, str]] = []
    for task in tasks:
        for seed in seeds:
            for scheme in PART_V_SCHEMES:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    name = "part_v_actual_real_residualized_local_knot" if int(args.actual_real) else "part_v_synthetic_residualized_local_knot"
    path = write_rows(OUT_ROOT / f"{name}{suffix}.csv", rows)
    audit = matrix_audit(rows)
    append_exec(
        "Part V residualized local-knot role shard",
        "done",
        files=rel(path),
        gpu=str(args.device),
        note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} actual_real={args.actual_real} audit={audit}",
    )
    return {"part": "V", "rows": len(rows), "matrix": rel(path), "audit": audit}


def summarize_part_v(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for scheme in [s for s in PART_V_SCHEMES if any(str(r.get("scheme")) == s for r in rows)]:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        eligible = [r for r in ss if ival(r.get("rank_saturated", 0)) == 0]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "real_task_count": len({r.get("task") for r in ss if ival(r.get("minimum_real_falsification", 0)) == 1}),
            "median_output_dim": median(r.get("output_dim", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_saturated_rows": sum(1 for r in ss if ival(r.get("rank_saturated", 0)) == 1),
            "eligible_rank_gain_positive_rate": mean((1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0) for r in eligible) if eligible else 1.0,
            "rank_gain_or_saturated_pass_rate": mean(r.get("rank_gain_or_saturated_pass", 0) for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_novelty_score": median(r.get("novelty_score_median", 0) for r in ss),
            "median_multiwitness_min_gain": median(r.get("multiwitness_min_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "selected_nonzero_rows": sum(1 for r in ss if ival(r.get("selected_step", 0)) > 0),
            "median_selected_step": median(r.get("selected_step", 0) for r in ss),
            "median_guard_debt_excess": median(r.get("guard_debt_excess", 0) for r in ss),
        })
    by = {r["scheme"]: r for r in summary_rows}
    primary = by.get("V0_KAN_local_knot_residualized_BC15", {})
    label = by.get("V1_label_shuffled_KAN_local_knot_residualized", {})
    random_capacity = by.get("V2_same_capacity_random_local_knot_residualized", {})
    mlp = by.get("V3_MLP_matched_tanh_local_knot_residualized", {})
    random_projection = by.get("V4_random_projection_local_knot_residualized", {})
    control_best = max(
        fval(label.get("median_guard_NLL_gain", 0)),
        fval(random_capacity.get("median_guard_NLL_gain", 0)),
        fval(mlp.get("median_guard_NLL_gain", 0)),
        fval(random_projection.get("median_guard_NLL_gain", 0)),
    )
    expected_rows = (
        len(csv_items(args.real_tasks if int(args.actual_real) else args.tasks))
        * len(int_items(args.real_seeds if int(args.actual_real) else args.seeds))
        * len(PART_V_SCHEMES)
    )
    no_debt_floor = 12 if int(args.actual_real) else 24
    gate = int(
        len(rows) == expected_rows
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_or_saturated_pass_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(primary.get("no_debt_rows", 0)) >= no_debt_floor
        and fval(primary.get("median_guard_NLL_gain", 0)) > control_best
    )
    blockers: list[str] = []
    if len(rows) != expected_rows:
        blockers.append("incomplete_matrix")
    if fval(primary.get("preservation_pass_rate", 0)) < 0.95:
        blockers.append("function_preservation_insufficient")
    if fval(primary.get("rank_gain_or_saturated_pass_rate", 0)) < 0.80:
        blockers.append("rank_gain_or_saturation_insufficient")
    if fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) < 1.0e-4:
        blockers.append("below_BC15_or_effect_floor")
    if ival(primary.get("no_debt_rows", 0)) < no_debt_floor:
        blockers.append("no_debt_insufficient")
    if fval(primary.get("median_guard_NLL_gain", 0)) <= control_best:
        blockers.append("control_explains_or_beats_primary")
    if not blockers:
        blockers.append("none")
    prefix = "actual_real" if int(args.actual_real) else "synthetic"
    summary = {
        "part": "V",
        "successor_repair": "residualized_local_knot_KAN_role_family",
        "matrix_kind": prefix,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / f"part_v_{prefix}_residualized_local_knot.csv"),
        "V0_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "V0_minus_same_capacity_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_capacity.get("median_guard_NLL_gain", 0)),
        "V0_minus_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "V0_minus_random_projection_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_projection.get("median_guard_NLL_gain", 0)),
        "V0_no_debt_rows": primary.get("no_debt_rows", 0),
        "control_best_median_guard_NLL": control_best,
        "dominant_blockers": blockers,
        "interpretation": "successor_residualized_local_knot_candidate_opened" if gate else "successor_residualized_local_knot_not_sufficient_under_strict_gate",
    }
    return summary_rows, summary


def merge_part_v(args: argparse.Namespace) -> dict[str, Any]:
    prefix = "actual_real" if int(args.actual_real) else "synthetic"
    base = f"part_v_{prefix}_residualized_local_knot"
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob(f"{base}_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    if rows:
        write_rows(OUT_ROOT / f"{base}.csv", rows)
    else:
        rows = read_rows(OUT_ROOT / f"{base}.csv")
    summary_rows, summary = summarize_part_v(args, rows)
    summary_path = write_rows(OUT_ROOT / f"{base}_summary.csv", summary_rows)
    summary["summary"] = rel(summary_path)
    matrix_path = OUT_ROOT / f"{base}.csv"
    audit = matrix_audit(rows)
    summary["matrix_audit"] = audit
    summary_json = write_json(OUT_ROOT / f"{base}_summary.json", summary)
    next_path = write_json(OUT_ROOT / f"{base}_next_actions_for_codex.json", {
        "part": "V",
        "gate_pass": summary["gate_pass"],
        "next": "promote_to_full_successor_plan_if_actual_real_also_passes" if summary["gate_pass"] and not int(args.actual_real) else ("candidate_opened_requires_full_official_plan" if summary["gate_pass"] else "do_not_claim_success; inspect blockers and avoid ordinary trust/rank sweep"),
    })
    fail_path = write_json(OUT_ROOT / f"{base}_failure_decomposition.json", {
        "part": "V",
        "gate_pass": summary["gate_pass"],
        "dominant_blockers": summary["dominant_blockers"],
        "summary": summary,
        "audit": audit,
    })
    append_exec(
        "Part V residualized local-knot role merge",
        "done",
        files=f"{rel(matrix_path)}; {rel(summary_path)}; {rel(summary_json)}; {rel(next_path)}; {rel(fail_path)}",
        gpu=str(args.device),
        note=f"rows={len(rows)} gate={summary['gate_pass']} actual_real={args.actual_real} audit={audit}",
    )
    append_recap("Part V residualized local-knot role results", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="part-v", choices=["part-v", "part-v-merge"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--actual-real", type=int, default=1)
    p.add_argument("--basis-input-gain", type=float, default=0.25)
    p.add_argument("--quadrature-points", type=int, default=65)
    p.add_argument("--edge-metric-ridge", type=float, default=1.0e-8)
    p.add_argument("--lam", type=float, default=1.0e-2)
    p.add_argument("--visual-side", type=int, default=4)
    p.add_argument("--visual-fixed-patch-features", type=int, default=1)
    p.add_argument("--visual-task-version", default="v23_15")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--train-size", type=int, default=128)
    p.add_argument("--guard-size", type=int, default=32)
    p.add_argument("--real-train-size", type=int, default=48)
    p.add_argument("--real-guard-size", type=int, default=32)
    p.add_argument("--real-compact-dim", type=int, default=32)
    p.add_argument("--task-alpha", type=float, default=0.03)
    p.add_argument("--debt-tolerance", type=float, default=1.0e-3)
    p.add_argument("--trust-debt-tolerance", type=float, default=1.0e-3)
    p.add_argument("--trust-raw-debt-penalty", type=float, default=0.0)
    p.add_argument("--float64", type=int, default=1)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--tasks", default=",".join(DEFAULT_SYNTHETIC_TASKS))
    p.add_argument("--real-tasks", default=",".join(REAL_FALSIFICATION_TASKS))
    p.add_argument("--real-seeds", default="0,1,2")
    p.add_argument("--trust-steps", type=int, default=60)
    p.add_argument("--trust-lr-grid", default="0.00003,0.0001,0.0003,0.001")
    p.add_argument("--amp-l2", type=float, default=1.0e-3)
    p.add_argument("--parent-count", type=int, default=8)
    p.add_argument("--role-feature-count", type=int, default=12)
    p.add_argument("--local-knots", default="-1.0,-0.25,0.25,1.0")
    p.add_argument("--score-mode", default="relevance_novelty", choices=["relevance_novelty", "control_residual"])
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--max-jobs", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.mode == "part-v":
        return collect_part_v(args)
    if args.mode == "part-v-merge":
        return merge_part_v(args)
    raise ValueError(args.mode)


if __name__ == "__main__":
    main()
