#!/usr/bin/env python3
"""DG-KAN v23.20 successor: contrastive shared-parent role architecture.

This is not a retroactive v23.19 success claim.  It is a successor probe
triggered by the v23.19 R18 next action: use a stronger compositional
shared-parent role materialization while keeping function preservation,
train-only selection, ordinary amplitude training, and matched controls.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_19_bc_function_preserving_edge_role_birth_lift_flow as v2319
from dgkan.fu.compositional_edge_tangent import apply_j, build_tangent_cache, explicit_jacobian


RUNNER = Path(__file__).resolve()
OUT_ROOT = Path(os.environ.get("V2320_OUT_ROOT", str(ROOT / "results/v23_20_successor"))).resolve()
V2319_PLAN = ROOT / "docs/DG-KAN_v23.19_BasisCovariantFunctionPreservingEdgeRoleBirthLiftFlow_多假设穷尽式完整详尽实验计划.md"
V2319_EXEC_LOG = ROOT / "docs/DG-KAN_v23.19_BasisCovariantFunctionPreservingEdgeRoleBirthLiftFlow_执行日志.md"
V2319_RECAP_LOG = ROOT / "docs/DG-KAN_v23.19_BasisCovariantFunctionPreservingEdgeRoleBirthLiftFlow_实验结果复盘.md"
PYTHON = sys.executable
EPS = 1.0e-12

PART_P_SCHEMES = [
    "P0_KAN_spline_pair_simplex_hard_trust",
    "P1_label_shuffled_KAN_spline_pair_hard_trust",
    "P2_same_capacity_random_pair_hard_trust",
    "P3_MLP_matched_selected_tanh_pair_hard_trust",
    "P4_random_projection_spline_selected_pair_hard_trust",
    "P5_KAN_spline_pair_more_parents_hard_trust",
    "P6_oracle_task_pair_upper",
]

PART_Q_SCHEMES = [
    "Q0_KAN_spline_product_pair_simplex_buffered_trust",
    "Q1_label_shuffled_KAN_spline_product_pair_buffered_trust",
    "Q2_same_capacity_random_product_pair_buffered_trust",
    "Q3_MLP_matched_tanh_product_pair_buffered_trust",
    "Q4_random_projection_spline_product_pair_buffered_trust",
    "Q5_oracle_task_product_pair_upper",
]

PART_R_SCHEMES = [
    "R0_KAN_spline_selected_product_pair_small_trust",
    "R1_label_shuffled_KAN_spline_selected_product_pair",
    "R2_same_capacity_random_selected_product_pair",
    "R3_MLP_matched_tanh_selected_product_pair",
    "R4_random_projection_spline_selected_product_pair",
    "R5_oracle_task_selected_product_pair_upper",
]

PART_S_SCHEMES = [
    "S0_KAN_spline_selected_product_pair_multiwitness_trust",
    "S1_label_shuffled_KAN_spline_selected_product_pair_multiwitness",
    "S2_same_capacity_random_selected_product_pair_multiwitness",
    "S3_MLP_matched_tanh_selected_product_pair_multiwitness",
    "S4_random_projection_spline_selected_product_pair_multiwitness",
    "S5_oracle_task_selected_product_pair_multiwitness_upper",
]

PART_T_SCHEMES = [
    "T0_KAN_spline_selected_product_pair_BC15anchored_multiwitness",
    "T1_label_shuffled_KAN_spline_selected_product_pair_BC15anchored",
    "T2_same_capacity_random_selected_product_pair_BC15anchored",
    "T3_MLP_matched_tanh_selected_product_pair_BC15anchored",
    "T4_random_projection_spline_selected_product_pair_BC15anchored",
    "T5_oracle_task_selected_product_pair_BC15anchored_upper",
]

PART_U_REAL_SCHEMES = PART_T_SCHEMES[:5]
REAL_FALSIFICATION_TASKS = ["Wine", "Spam", "MNIST", "FashionMNIST", "CIFAR10_compact"]

DEFAULT_TASKS = [
    "SYN2_node_bank_shared_role",
    "SYN3_local_patch_interaction",
    "SYN6_two_layer_compositional_role",
    "SYN7_MLP_friendly_linear_role",
    "SYN8_KAN_specific_univariate_role",
    "SYN9_no_signal_negative_control",
]

AUDIT_DEFAULTS: dict[str, Any] = {
    "version": "v23.20_successor",
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
    for key in ["CUDA_VISIBLE_DEVICES", "V2320_OUT_ROOT"]:
        if os.environ.get(key):
            env_bits.append(f"{key}={os.environ[key]}")
    return " ".join([*env_bits, PYTHON, rel(RUNNER), *sys.argv[1:]])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metric_hash(name: str, definition: str) -> str:
    return stable_hash(f"v23.20_successor::{name}::{definition}")[:16]


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


def append_file(path: Path, text: str) -> None:
    ensure_out()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def append_exec(part: str, status: str, *, files: str = "", note: str = "", gpu: str = "") -> None:
    append_file(
        V2319_EXEC_LOG,
        f"\n## {now()} v23.20 successor {part} {status}\n\n"
        f"- command: `{command_text()}`\n"
        f"- gpu: `{gpu}`\n"
        f"- python: `{PYTHON}`\n"
        f"- torch: `{getattr(torch, '__version__', 'unknown')}`\n"
        f"- runner: `{rel(RUNNER)}`\n"
        f"- out_root: `{rel(OUT_ROOT)}`\n"
        f"- successor boundary: does not rewrite v23.19 final route; records post-R18 exploration.\n"
        + (f"- files: `{files}`\n" if files else "")
        + (f"- note: {note}\n" if note else ""),
    )


def append_recap(title: str, payload: dict[str, Any]) -> None:
    append_file(
        V2319_RECAP_LOG,
        f"\n## {now()} v23.20 successor {title}\n\n"
        "Successor boundary: these rows are post-v23.19 exploration and do not change the recorded v23.19 R18 final route.\n\n"
        f"```json\n{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}\n```\n",
    )


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


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = sorted(fval(v, float("nan")) for v in values)
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float(default)
    n = len(vals)
    return float(vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def int_items(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def pairwise_simplex_logit_columns(features: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Map parent functions to zero-sum class-pair logit columns."""
    cols: list[torch.Tensor] = []
    n = int(features.shape[0])
    cdim = int(output_dim)
    if int(features.shape[1]) == 0:
        return torch.empty((n * cdim, 0), device=features.device, dtype=torch.float64)
    scale = 1.0 / math.sqrt(2.0)
    for pidx in range(int(features.shape[1])):
        feat = features[:, pidx].to(dtype=torch.float64)
        for a in range(cdim):
            for b in range(a + 1, cdim):
                mat = torch.zeros((n, cdim), device=features.device, dtype=torch.float64)
                mat[:, a] = feat * scale
                mat[:, b] = -feat * scale
                cols.append(mat.reshape(-1))
    return torch.stack(cols, dim=1) if cols else torch.empty((n * cdim, 0), device=features.device, dtype=torch.float64)


def raw_compositional_features(features: torch.Tensor) -> torch.Tensor:
    vals: list[torch.Tensor] = []
    k = int(features.shape[1])
    if k == 0:
        return torch.empty((int(features.shape[0]), 0), device=features.device, dtype=torch.float64)
    work = features.to(dtype=torch.float64)
    for i in range(k):
        vals.append(work[:, i])
    for i in range(k):
        for j in range(i + 1, k):
            vals.append(work[:, i] * work[:, j])
    return torch.stack(vals, dim=1) if vals else torch.empty((int(features.shape[0]), 0), device=features.device, dtype=torch.float64)


def fit_compositional_feature_state(features: torch.Tensor) -> dict[str, torch.Tensor | int]:
    raw = raw_compositional_features(features)
    if int(raw.shape[1]) == 0:
        return {
            "mu": torch.empty((0,), device=features.device, dtype=torch.float64),
            "sd": torch.empty((0,), device=features.device, dtype=torch.float64),
            "base_count": int(features.shape[1]),
            "expanded_count": 0,
        }
    mu = raw.mean(dim=0)
    sd = raw.std(dim=0).clamp_min(1.0e-6)
    return {"mu": mu.detach(), "sd": sd.detach(), "base_count": int(features.shape[1]), "expanded_count": int(raw.shape[1])}


def apply_compositional_feature_state(features: torch.Tensor, state: dict[str, torch.Tensor | int]) -> torch.Tensor:
    raw = raw_compositional_features(features)
    if int(raw.shape[1]) == 0:
        return raw
    mu = state["mu"].to(device=features.device, dtype=torch.float64)  # type: ignore[union-attr]
    sd = state["sd"].to(device=features.device, dtype=torch.float64)  # type: ignore[union-attr]
    return (raw - mu.reshape(1, -1)) / sd.reshape(1, -1).clamp_min(1.0e-6)


def fit_selected_compositional_feature_state(
    fit_features: torch.Tensor,
    y_fit: torch.Tensor,
    logits_fit: torch.Tensor,
    *,
    scheme: str,
    seed: int,
    selected_count: int,
) -> dict[str, Any]:
    raw = raw_compositional_features(fit_features)
    if int(raw.shape[1]) == 0:
        return {
            "mu": torch.empty((0,), device=fit_features.device, dtype=torch.float64),
            "sd": torch.empty((0,), device=fit_features.device, dtype=torch.float64),
            "selected": [],
            "base_count": int(fit_features.shape[1]),
            "expanded_count": 0,
            "selection_score_max": 0.0,
            "selection_score_median": 0.0,
        }
    norm, mu, sd = v2319.normalize_parent_bank_fit(raw)
    n = int(norm.shape[0])
    mid = max(2, n // 2)
    y_s = y_fit[:mid]
    y_w = y_fit[mid:]
    logits_s = logits_fit[:mid]
    logits_w = logits_fit[mid:]
    y_s_use = v2319.v2318.shuffle_labels(y_s, seed + 701) if "label_shuffled" in scheme else y_s
    y_w_use = v2319.v2318.shuffle_labels(y_w, seed + 709) if "label_shuffled" in scheme else y_w
    if "same_capacity_random" in scheme:
        gen = torch.Generator(device=fit_features.device).manual_seed(int(seed) + 234777)
        selected = torch.randperm(int(norm.shape[1]), generator=gen, device=fit_features.device)[: min(int(selected_count), int(norm.shape[1]))]
        scores = torch.zeros(int(norm.shape[1]), device=fit_features.device, dtype=torch.float64)
    else:
        scores = v2319.score_parent_candidates(norm[:mid], logits_s, y_s_use, norm[mid:], logits_w, y_w_use)
        selected = torch.topk(scores, k=min(int(selected_count), int(scores.numel()))).indices
    return {
        "mu": mu.detach(),
        "sd": sd.detach(),
        "selected": [int(i) for i in selected.detach().cpu().tolist()],
        "base_count": int(fit_features.shape[1]),
        "expanded_count": int(selected.numel()),
        "selection_score_max": float(scores[selected].max().detach().cpu().item()) if int(selected.numel()) else 0.0,
        "selection_score_median": float(scores[selected].median().detach().cpu().item()) if int(selected.numel()) else 0.0,
    }


def apply_selected_compositional_feature_state(features: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    raw = raw_compositional_features(features)
    if int(raw.shape[1]) == 0:
        return raw
    mu = state["mu"].to(device=features.device, dtype=torch.float64)
    sd = state["sd"].to(device=features.device, dtype=torch.float64)
    norm = (raw - mu.reshape(1, -1)) / sd.reshape(1, -1).clamp_min(1.0e-6)
    idx = torch.tensor(list(state.get("selected", [])), device=features.device, dtype=torch.long)
    return norm.index_select(1, idx) if int(idx.numel()) else torch.empty((int(features.shape[0]), 0), device=features.device, dtype=torch.float64)


def column_scales(cols: torch.Tensor) -> torch.Tensor:
    if int(cols.shape[1]) == 0:
        return torch.empty((0,), device=cols.device, dtype=torch.float64)
    return cols.norm(dim=0).clamp_min(1.0e-8) / math.sqrt(max(1, int(cols.shape[0])))


def apply_scales(cols: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if int(cols.shape[1]) == 0:
        return cols
    return cols / scales.to(device=cols.device, dtype=torch.float64).reshape(1, -1).clamp_min(1.0e-8)


def scale_diag(scales: torch.Tensor) -> dict[str, float | int]:
    if int(scales.numel()) == 0:
        return {"column_normalized": 0, "column_scale_min": 0.0, "column_scale_max": 0.0}
    return {
        "column_normalized": 1,
        "column_scale_min": float(scales.min().detach().cpu().item()),
        "column_scale_max": float(scales.max().detach().cpu().item()),
    }


def debt_excess(after: dict[str, float], before: dict[str, float], tolerance: float) -> float:
    return v2319.debt_excess(after, before, tolerance)


def fit_amplitudes_with_buffered_witness_trust(
    fit_logits: torch.Tensor,
    y_fit: torch.Tensor,
    fit_cols: torch.Tensor,
    witness_logits: torch.Tensor,
    y_witness: torch.Tensor,
    witness_cols: torch.Tensor,
    guard_logits: torch.Tensor,
    y_guard: torch.Tensor,
    guard_cols: torch.Tensor,
    *,
    steps: int,
    lr_grid: list[float],
    amp_l2: float,
    trust_debt_tolerance: float,
    raw_debt_penalty: float,
) -> dict[str, Any]:
    """Train amplitudes on fit split and select snapshots by witness trust.

    v23.19's hard-debt trust only penalized witness debt after it crossed the
    tolerance.  Part P-R2 showed that this can pick high-gain snapshots close to
    the witness debt boundary and then fail no-debt on guard.  This scorer keeps
    the same train/witness-only protocol but applies a continuous penalty to raw
    witness debt inside the tolerance.
    """
    fit_logits = fit_logits.detach()
    witness_logits = witness_logits.detach()
    guard_logits = guard_logits.detach()
    fit_cols = fit_cols.detach()
    witness_cols = witness_cols.detach()
    guard_cols = guard_cols.detach()
    witness_before = v2319.logits_metrics(witness_logits, y_witness)
    guard_before = v2319.logits_metrics(guard_logits, y_guard)
    if int(fit_cols.shape[1]) == 0:
        amp0 = torch.zeros((0, 1), device=fit_logits.device, dtype=torch.float64)
        return {
            "amp": amp0,
            "first_grad_norm": 0.0,
            "selected_lr": 0.0,
            "selected_step": 0,
            "selected_score": 0.0,
            "witness_before": witness_before,
            "witness_after": witness_before,
            "guard_before": guard_before,
            "guard_after": guard_before,
            "witness_debt_excess": 0.0,
            "witness_raw_debt_excess": 0.0,
            "hard_debt_trust": 1,
        }
    zero_amp = torch.zeros((int(fit_cols.shape[1]), 1), device=fit_logits.device, dtype=torch.float64)
    best_amp = zero_amp.clone()
    best_after = witness_before
    best_score = 0.0
    best_lr = 0.0
    best_step = 0
    best_debt_excess = 0.0
    best_raw_debt_excess = 0.0
    first_grad_norm = 0.0
    snapshot_every = max(1, int(steps) // 12)
    for lr in lr_grid:
        amp = torch.zeros_like(zero_amp, requires_grad=True)
        opt = torch.optim.SGD([amp], lr=float(lr))
        for step in range(1, int(steps) + 1):
            opt.zero_grad(set_to_none=True)
            logits = (fit_logits.reshape(-1, 1) + fit_cols @ amp).reshape_as(fit_logits)
            loss = F.cross_entropy(logits.to(dtype=torch.float64), y_fit.long())
            if float(amp_l2) > 0.0:
                loss = loss + float(amp_l2) * amp.square().mean()
            loss.backward()
            if first_grad_norm == 0.0 and amp.grad is not None:
                first_grad_norm = float(amp.grad.detach().norm().cpu().item())
            opt.step()
            if step % snapshot_every != 0 and step != int(steps):
                continue
            with torch.no_grad():
                witness_logits_after = (witness_logits.reshape(-1, 1) + witness_cols @ amp.detach()).reshape_as(witness_logits)
                witness_after = v2319.logits_metrics(witness_logits_after, y_witness)
                gain = witness_before["loss"] - witness_after["loss"]
                tol_excess = debt_excess(witness_after, witness_before, trust_debt_tolerance)
                raw_excess = debt_excess(witness_after, witness_before, 0.0)
                amp_cost = float(amp_l2) * float(amp.detach().square().mean().cpu().item())
                score = gain - 10.0 * tol_excess - float(raw_debt_penalty) * raw_excess - amp_cost
                if tol_excess <= 0.0 and score > best_score:
                    best_score = float(score)
                    best_amp = amp.detach().clone()
                    best_after = witness_after
                    best_lr = float(lr)
                    best_step = int(step)
                    best_debt_excess = float(tol_excess)
                    best_raw_debt_excess = float(raw_excess)
    with torch.no_grad():
        guard_after_logits = (guard_logits.reshape(-1, 1) + guard_cols @ best_amp).reshape_as(guard_logits)
        guard_after = v2319.logits_metrics(guard_after_logits, y_guard)
    return {
        "amp": best_amp,
        "first_grad_norm": first_grad_norm,
        "selected_lr": best_lr,
        "selected_step": best_step,
        "selected_score": best_score,
        "witness_before": witness_before,
        "witness_after": best_after,
        "guard_before": guard_before,
        "guard_after": guard_after,
        "witness_debt_excess": best_debt_excess,
        "witness_raw_debt_excess": best_raw_debt_excess,
        "hard_debt_trust": 1,
    }


def fit_amplitudes_with_multi_witness_trust(
    fit_logits: torch.Tensor,
    y_fit: torch.Tensor,
    fit_cols: torch.Tensor,
    witnesses: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    guard_logits: torch.Tensor,
    y_guard: torch.Tensor,
    guard_cols: torch.Tensor,
    *,
    steps: int,
    lr_grid: list[float],
    amp_l2: float,
    trust_debt_tolerance: float,
    raw_debt_penalty: float,
) -> dict[str, Any]:
    fit_logits = fit_logits.detach()
    fit_cols = fit_cols.detach()
    guard_logits = guard_logits.detach()
    guard_cols = guard_cols.detach()
    witness_data = [(lg.detach(), yy, cc.detach(), v2319.logits_metrics(lg.detach(), yy)) for lg, yy, cc in witnesses]
    guard_before = v2319.logits_metrics(guard_logits, y_guard)
    if int(fit_cols.shape[1]) == 0:
        amp0 = torch.zeros((0, 1), device=fit_logits.device, dtype=torch.float64)
        return {
            "amp": amp0,
            "first_grad_norm": 0.0,
            "selected_lr": 0.0,
            "selected_step": 0,
            "selected_score": 0.0,
            "witness_before": witness_data[0][3] if witness_data else guard_before,
            "witness_after": witness_data[0][3] if witness_data else guard_before,
            "guard_before": guard_before,
            "guard_after": guard_before,
            "witness_debt_excess": 0.0,
            "witness_raw_debt_excess": 0.0,
            "multiwitness_min_gain": 0.0,
            "multiwitness_mean_gain": 0.0,
            "multiwitness_count": len(witness_data),
            "hard_debt_trust": 1,
        }
    zero_amp = torch.zeros((int(fit_cols.shape[1]), 1), device=fit_logits.device, dtype=torch.float64)
    best_amp = zero_amp.clone()
    best_score = 0.0
    best_lr = 0.0
    best_step = 0
    best_after = witness_data[0][3] if witness_data else guard_before
    best_tol_excess = 0.0
    best_raw_excess = 0.0
    best_min_gain = 0.0
    best_mean_gain = 0.0
    first_grad_norm = 0.0
    snapshot_every = max(1, int(steps) // 12)
    for lr in lr_grid:
        amp = torch.zeros_like(zero_amp, requires_grad=True)
        opt = torch.optim.SGD([amp], lr=float(lr))
        for step in range(1, int(steps) + 1):
            opt.zero_grad(set_to_none=True)
            logits = (fit_logits.reshape(-1, 1) + fit_cols @ amp).reshape_as(fit_logits)
            loss = F.cross_entropy(logits.to(dtype=torch.float64), y_fit.long())
            if float(amp_l2) > 0.0:
                loss = loss + float(amp_l2) * amp.square().mean()
            loss.backward()
            if first_grad_norm == 0.0 and amp.grad is not None:
                first_grad_norm = float(amp.grad.detach().norm().cpu().item())
            opt.step()
            if step % snapshot_every != 0 and step != int(steps):
                continue
            with torch.no_grad():
                gains: list[float] = []
                tol_excesses: list[float] = []
                raw_excesses: list[float] = []
                afters: list[dict[str, float]] = []
                for witness_logits, y_witness, witness_cols, before in witness_data:
                    after_logits = (witness_logits.reshape(-1, 1) + witness_cols @ amp.detach()).reshape_as(witness_logits)
                    after = v2319.logits_metrics(after_logits, y_witness)
                    afters.append(after)
                    gains.append(before["loss"] - after["loss"])
                    tol_excesses.append(debt_excess(after, before, trust_debt_tolerance))
                    raw_excesses.append(debt_excess(after, before, 0.0))
                min_gain = min(gains) if gains else 0.0
                mean_gain = sum(gains) / max(1, len(gains))
                max_tol_excess = max(tol_excesses) if tol_excesses else 0.0
                max_raw_excess = max(raw_excesses) if raw_excesses else 0.0
                amp_cost = float(amp_l2) * float(amp.detach().square().mean().cpu().item())
                score = min_gain + 0.25 * mean_gain - 10.0 * max_tol_excess - float(raw_debt_penalty) * max_raw_excess - amp_cost
                if max_tol_excess <= 0.0 and score > best_score:
                    best_score = float(score)
                    best_amp = amp.detach().clone()
                    best_lr = float(lr)
                    best_step = int(step)
                    best_after = afters[0] if afters else guard_before
                    best_tol_excess = float(max_tol_excess)
                    best_raw_excess = float(max_raw_excess)
                    best_min_gain = float(min_gain)
                    best_mean_gain = float(mean_gain)
    with torch.no_grad():
        guard_after_logits = (guard_logits.reshape(-1, 1) + guard_cols @ best_amp).reshape_as(guard_logits)
        guard_after = v2319.logits_metrics(guard_after_logits, y_guard)
    return {
        "amp": best_amp,
        "first_grad_norm": first_grad_norm,
        "selected_lr": best_lr,
        "selected_step": best_step,
        "selected_score": best_score,
        "witness_before": witness_data[0][3] if witness_data else guard_before,
        "witness_after": best_after,
        "guard_before": guard_before,
        "guard_after": guard_after,
        "witness_debt_excess": best_tol_excess,
        "witness_raw_debt_excess": best_raw_excess,
        "multiwitness_min_gain": best_min_gain,
        "multiwitness_mean_gain": best_mean_gain,
        "multiwitness_count": len(witness_data),
        "hard_debt_trust": 1,
    }


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


def evaluate_part_p_row(args: argparse.Namespace, task: str, seed: int, scheme: str) -> dict[str, Any]:
    dtype = torch.float64 if int(args.float64) else torch.float32
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    row_t0 = time.perf_counter()
    if str(task).startswith("SYN"):
        x, y, xg, yg, meta = v2319.synthetic_batch(args, task, seed, int(args.train_size), int(args.guard_size), dtype)
    else:
        x, y, xg, yg, meta = v2319.load_real_task(args, task, seed, int(args.real_train_size), int(args.real_guard_size), dtype)
    output_dim = int(meta["output_dim"])
    model_seed = 2342000 + 47 * int(seed) + len(task)
    model, basis_key = v2319.make_model_for(args, "D-CHE_K5_depth2", int(x.shape[1]), output_dim, model_seed, dtype)

    n = int(x.shape[0])
    fit_n = max(8, int(round(0.75 * n)))
    fit_n = min(fit_n, n - 4)
    x_fit, y_fit = x[:fit_n], y[:fit_n]
    x_wit, y_wit = x[fit_n:], y[fit_n:]

    cache_full = build_tangent_cache(model, x)
    cache_fit = build_tangent_cache(model, x_fit)
    cache_wit = build_tangent_cache(model, x_wit)
    cache_guard = build_tangent_cache(model, xg)
    logits_full = cache_full.logits.detach().to(dtype=torch.float64)
    logits_fit = cache_fit.logits.detach().to(dtype=torch.float64)
    logits_wit = cache_wit.logits.detach().to(dtype=torch.float64)
    logits_guard = cache_guard.logits.detach().to(dtype=torch.float64)

    parent_count = int(args.parent_count) * (2 if "more_parents" in scheme else 1)
    t0 = time.perf_counter()
    state = v2319.fit_parent_selector_state(
        x_fit,
        y_fit,
        logits_fit,
        task=task,
        scheme=scheme,
        output_dim=output_dim,
        seed=seed,
        parent_count=parent_count,
    )
    fit_feats = v2319.apply_parent_selector_state(x_fit, state)
    wit_feats = v2319.apply_parent_selector_state(x_wit, state)
    guard_feats = v2319.apply_parent_selector_state(xg, state)
    full_feats = v2319.apply_parent_selector_state(x, state)
    feature_expansion = "selected_parent_features"
    expanded_parent_count = int(fit_feats.shape[1])
    if scheme.startswith("R") or "selected_product_pair" in scheme:
        comp_state_selected = fit_selected_compositional_feature_state(
            fit_feats,
            y_fit,
            logits_fit,
            scheme=scheme,
            seed=seed,
            selected_count=int(args.expanded_parent_count),
        )
        fit_feats = apply_selected_compositional_feature_state(fit_feats, comp_state_selected)
        wit_feats = apply_selected_compositional_feature_state(wit_feats, comp_state_selected)
        guard_feats = apply_selected_compositional_feature_state(guard_feats, comp_state_selected)
        full_feats = apply_selected_compositional_feature_state(full_feats, comp_state_selected)
        feature_expansion = "base_plus_pairwise_products_cross_split_selected"
        expanded_parent_count = int(comp_state_selected["expanded_count"])
    elif scheme.startswith("Q") or "product_pair" in scheme:
        comp_state = fit_compositional_feature_state(fit_feats)
        fit_feats = apply_compositional_feature_state(fit_feats, comp_state)
        wit_feats = apply_compositional_feature_state(wit_feats, comp_state)
        guard_feats = apply_compositional_feature_state(guard_feats, comp_state)
        full_feats = apply_compositional_feature_state(full_feats, comp_state)
        feature_expansion = "base_plus_pairwise_products_train_normalized"
        expanded_parent_count = int(comp_state["expanded_count"])

    fit_cols = pairwise_simplex_logit_columns(fit_feats, output_dim)
    wit_cols = pairwise_simplex_logit_columns(wit_feats, output_dim)
    guard_cols = pairwise_simplex_logit_columns(guard_feats, output_dim)
    full_cols = pairwise_simplex_logit_columns(full_feats, output_dim)
    scales = column_scales(fit_cols)
    fit_cols = apply_scales(fit_cols, scales)
    wit_cols = apply_scales(wit_cols, scales)
    guard_cols = apply_scales(guard_cols, scales)
    full_cols = apply_scales(full_cols, scales)
    norm_diag = scale_diag(scales)
    role_operator_ms = 1000.0 * (time.perf_counter() - t0)
    birth_materialization_ms = role_operator_ms

    zero_train = (full_cols @ torch.zeros((int(full_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_full)
    zero_guard = (guard_cols @ torch.zeros((int(guard_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_guard)
    birth_logit_error = max(float(zero_train.abs().max().detach().cpu().item()), float(zero_guard.abs().max().detach().cpu().item()))

    old_j = explicit_jacobian(model, cache_full).to(dtype=torch.float64)
    target = v2319.output_residual(logits_full, y).reshape(-1)
    old_rank = v2319.matrix_rank(old_j)
    new_rank = v2319.matrix_rank(torch.cat([old_j, full_cols], dim=1))
    s_res = v2319.residualized_singulars(old_j, full_cols)
    positive_s = s_res[s_res > 1.0e-9]
    new_smin = float(positive_s.min().detach().cpu().item()) if int(positive_s.numel()) else 0.0
    cov_before = v2319.colspace_coverage(old_j, target)
    cov_after = v2319.colspace_coverage(torch.cat([old_j, full_cols], dim=1), target)

    lr_grid = [float(v) for v in csv_items(args.trust_lr_grid)]
    amp_result = fit_amplitudes_with_buffered_witness_trust(
        logits_fit,
        y_fit,
        fit_cols,
        logits_wit,
        y_wit,
        wit_cols,
        logits_guard,
        yg,
        guard_cols,
        steps=int(args.trust_steps),
        lr_grid=lr_grid,
        amp_l2=float(args.amp_l2),
        trust_debt_tolerance=float(args.trust_debt_tolerance),
        raw_debt_penalty=float(args.trust_raw_debt_penalty),
    )
    guard_before = amp_result["guard_before"]
    guard_after = amp_result["guard_after"]
    witness_before = amp_result["witness_before"]
    witness_after = amp_result["witness_after"]

    base_delta, _base_diag, _ = v2319.v2318.task_lift(v2319.model_args(args, output_dim), model, x_fit, y_fit, basis_key)
    guard_base_pred = apply_j(model, cache_guard, base_delta).to(dtype=torch.float64)
    guard_bc15 = v2319.logits_metrics(logits_guard + float(args.task_alpha) * guard_base_pred, yg)
    guard_gain = guard_before["loss"] - guard_after["loss"]
    bc15_gain = guard_before["loss"] - guard_bc15["loss"]
    guard_debt = debt_excess(guard_after, guard_before, float(args.debt_tolerance))
    amp = amp_result["amp"]

    role_score_max = fval(state.get("selection_score_max", 0.0))
    role_score_median = fval(state.get("selection_score_median", 0.0))
    no_debt = int(guard_debt <= 0.0)
    return {
        "part": "R" if scheme.startswith("R") else ("Q" if scheme.startswith("Q") else "P"),
        "task": task,
        "seed": int(seed),
        "paired_model_seed": int(model_seed),
        "scheme": scheme,
        "architecture": "D-CHE_K5_depth2",
        "basis_key": basis_key,
        "dataset_kind": "synthetic_v23_20_successor_contrastive_pairwise_shared_parent",
        "train_size": int(x.shape[0]),
        "fit_size": int(x_fit.shape[0]),
        "train_witness_size": int(x_wit.shape[0]),
        "guard_size": int(xg.shape[0]),
        "parent_kind": state.get("kind", ""),
        "parent_count": int(fit_feats.shape[1]),
        "base_parent_count": int(state.get("selected") and len(state.get("selected", [])) or fit_feats.shape[1]),
        "expanded_parent_count": expanded_parent_count,
        "role_columns": int(full_cols.shape[1]),
        "column_family": "zero_sum_pairwise_simplex_parent_columns",
        "feature_expansion": feature_expansion,
        "selected_parent_descriptors": ";".join(str(v) for v in state.get("descriptors", [])),
        "selection_score_median": role_score_median,
        "selection_score_max": role_score_max,
        **norm_diag,
        "birth_logit_max_abs_error": birth_logit_error,
        "function_preservation_pass": int(birth_logit_error <= v2319.THRESHOLDS["function_preservation_float64"]),
        "old_tangent_rank": old_rank,
        "new_tangent_rank": new_rank,
        "tangent_rank_gain": new_rank - old_rank,
        "new_tangent_smallest_singular_value": new_smin,
        "target_subspace_coverage_before": cov_before,
        "target_subspace_coverage_after": cov_after,
        "target_coverage_gain": cov_after - cov_before,
        "role_operator_top_eigenvalue": role_score_max,
        "role_operator_LCB": role_score_median,
        "source_witness_role_cosine": 1.0 if role_score_max > 0 else 0.0,
        "role_shape_G_norm": float(fit_feats.norm().detach().cpu().item()),
        "role_smoothness": 0.0,
        "role_domain_occupancy": float((fit_feats.abs() > 1.0e-8).to(dtype=torch.float64).mean().detach().cpu().item()) if int(fit_feats.numel()) else 0.0,
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
        "train_witness_NLL_before": witness_before["loss"],
        "train_witness_NLL_after": witness_after["loss"],
        "train_witness_NLL_gain": witness_before["loss"] - witness_after["loss"],
        "train_witness_debt_excess": amp_result["witness_debt_excess"],
        "train_witness_raw_debt_excess": amp_result["witness_raw_debt_excess"],
        "ordinary_gradient_only": 1,
        "guard_NLL_before": guard_before["loss"],
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
        "no_debt": no_debt,
        "role_operator_ms": role_operator_ms,
        "birth_materialization_ms": birth_materialization_ms,
        "controller_overhead_ratio": float(full_cols.numel()) / max(1.0, float(old_j.numel())),
        "peak_memory": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0.0,
        "row_wall_ms": 1000.0 * (time.perf_counter() - row_t0),
        "implementation_function_hash_present": 1,
        "implementation_function_hash": metric_hash("evaluate_part_p_row", "contrastive_zero_sum_pairwise_shared_parent_columns_with_hard_train_witness_debt_trust"),
        "metric_definition_present": 1,
        "used_guard_for_selection": 0,
        "manual_nonzero_role_amplitude_inserted": 0,
    }


def evaluate_part_s_row(args: argparse.Namespace, task: str, seed: int, scheme: str) -> dict[str, Any]:
    dtype = torch.float64 if int(args.float64) else torch.float32
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    row_t0 = time.perf_counter()
    if str(task).startswith("SYN"):
        x, y, xg, yg, meta = v2319.synthetic_batch(args, task, seed, int(args.train_size), int(args.guard_size), dtype)
    else:
        x, y, xg, yg, meta = v2319.load_real_task(args, task, seed, int(args.real_train_size), int(args.real_guard_size), dtype)
    output_dim = int(meta["output_dim"])
    model_seed = 2347000 + 53 * int(seed) + len(task)
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

    t0 = time.perf_counter()
    state = v2319.fit_parent_selector_state(
        x_select,
        y_select,
        logits_select,
        task=task,
        scheme=scheme,
        output_dim=output_dim,
        seed=seed,
        parent_count=int(args.parent_count),
    )
    select_feats = v2319.apply_parent_selector_state(x_select, state)
    amp_feats = v2319.apply_parent_selector_state(x_amp, state)
    wit_a_feats = v2319.apply_parent_selector_state(x_wit_a, state)
    wit_b_feats = v2319.apply_parent_selector_state(x_wit_b, state)
    guard_feats = v2319.apply_parent_selector_state(xg, state)
    full_feats = v2319.apply_parent_selector_state(x, state)
    comp_state = fit_selected_compositional_feature_state(
        select_feats,
        y_select,
        logits_select,
        scheme=scheme,
        seed=seed,
        selected_count=int(args.expanded_parent_count),
    )
    amp_feats = apply_selected_compositional_feature_state(amp_feats, comp_state)
    wit_a_feats = apply_selected_compositional_feature_state(wit_a_feats, comp_state)
    wit_b_feats = apply_selected_compositional_feature_state(wit_b_feats, comp_state)
    guard_feats = apply_selected_compositional_feature_state(guard_feats, comp_state)
    full_feats = apply_selected_compositional_feature_state(full_feats, comp_state)

    amp_cols = pairwise_simplex_logit_columns(amp_feats, output_dim)
    wit_a_cols = pairwise_simplex_logit_columns(wit_a_feats, output_dim)
    wit_b_cols = pairwise_simplex_logit_columns(wit_b_feats, output_dim)
    guard_cols = pairwise_simplex_logit_columns(guard_feats, output_dim)
    full_cols = pairwise_simplex_logit_columns(full_feats, output_dim)
    scales = column_scales(amp_cols)
    amp_cols = apply_scales(amp_cols, scales)
    wit_a_cols = apply_scales(wit_a_cols, scales)
    wit_b_cols = apply_scales(wit_b_cols, scales)
    guard_cols = apply_scales(guard_cols, scales)
    full_cols = apply_scales(full_cols, scales)
    norm_diag = scale_diag(scales)
    role_operator_ms = 1000.0 * (time.perf_counter() - t0)
    birth_materialization_ms = role_operator_ms

    zero_train = (full_cols @ torch.zeros((int(full_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_full)
    zero_guard = (guard_cols @ torch.zeros((int(guard_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_guard)
    birth_logit_error = max(float(zero_train.abs().max().detach().cpu().item()), float(zero_guard.abs().max().detach().cpu().item()))

    old_j = explicit_jacobian(model, cache_full).to(dtype=torch.float64)
    target = v2319.output_residual(logits_full, y).reshape(-1)
    old_rank = v2319.matrix_rank(old_j)
    new_rank = v2319.matrix_rank(torch.cat([old_j, full_cols], dim=1))
    s_res = v2319.residualized_singulars(old_j, full_cols)
    positive_s = s_res[s_res > 1.0e-9]
    new_smin = float(positive_s.min().detach().cpu().item()) if int(positive_s.numel()) else 0.0
    cov_before = v2319.colspace_coverage(old_j, target)
    cov_after = v2319.colspace_coverage(torch.cat([old_j, full_cols], dim=1), target)

    lr_grid = [float(v) for v in csv_items(args.trust_lr_grid)]
    anchored_to_bc15 = int(scheme.startswith("T"))
    base_delta, _base_diag, _ = v2319.v2318.task_lift(v2319.model_args(args, output_dim), model, x_amp, y_amp, basis_key)
    amp_base_pred = apply_j(model, cache_amp, base_delta).to(dtype=torch.float64)
    wit_a_base_pred = apply_j(model, cache_wit_a, base_delta).to(dtype=torch.float64)
    wit_b_base_pred = apply_j(model, cache_wit_b, base_delta).to(dtype=torch.float64)
    guard_base_pred = apply_j(model, cache_guard, base_delta).to(dtype=torch.float64)
    alpha = float(args.task_alpha)
    trust_logits_amp = logits_amp + alpha * amp_base_pred if anchored_to_bc15 else logits_amp
    trust_logits_wit_a = logits_wit_a + alpha * wit_a_base_pred if anchored_to_bc15 else logits_wit_a
    trust_logits_wit_b = logits_wit_b + alpha * wit_b_base_pred if anchored_to_bc15 else logits_wit_b
    trust_logits_guard = logits_guard + alpha * guard_base_pred if anchored_to_bc15 else logits_guard
    trust_witnesses = [(trust_logits_wit_a, y_wit_a, wit_a_cols), (trust_logits_wit_b, y_wit_b, wit_b_cols)]
    combined_witness_size = 0
    if int(args.combined_witness_trust):
        trust_wit_ab = torch.cat([trust_logits_wit_a, trust_logits_wit_b], dim=0)
        y_wit_ab = torch.cat([y_wit_a, y_wit_b], dim=0)
        wit_ab_cols = torch.cat([wit_a_cols, wit_b_cols], dim=0)
        trust_witnesses.append((trust_wit_ab, y_wit_ab, wit_ab_cols))
        combined_witness_size = int(y_wit_ab.numel())
    amp_result = fit_amplitudes_with_multi_witness_trust(
        trust_logits_amp,
        y_amp,
        amp_cols,
        trust_witnesses,
        trust_logits_guard,
        yg,
        guard_cols,
        steps=int(args.trust_steps),
        lr_grid=lr_grid,
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
    guard_debt = debt_excess(guard_after, guard_before, float(args.debt_tolerance))
    amp = amp_result["amp"]
    no_debt = int(guard_debt <= 0.0)
    role_score_max = fval(state.get("selection_score_max", 0.0))
    role_score_median = fval(state.get("selection_score_median", 0.0))
    product_score_max = fval(comp_state.get("selection_score_max", 0.0))
    product_score_median = fval(comp_state.get("selection_score_median", 0.0))
    return {
        "part": "T" if anchored_to_bc15 else "S",
        "task": task,
        "seed": int(seed),
        "paired_model_seed": int(model_seed),
        "scheme": scheme,
        "architecture": "D-CHE_K5_depth2",
        "basis_key": basis_key,
        "dataset_kind": f"{meta.get('dataset_kind', 'unknown')}_v23_20_successor_multiwitness_selected_product_pair",
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
        "trust_witness_combined_size": combined_witness_size,
        "combined_witness_trust_used": int(args.combined_witness_trust),
        "guard_size": int(xg.shape[0]),
        "parent_kind": state.get("kind", ""),
        "base_parent_count": int(comp_state.get("base_count", 0)),
        "expanded_parent_count": int(comp_state.get("expanded_count", 0)),
        "parent_count": int(amp_feats.shape[1]),
        "role_columns": int(full_cols.shape[1]),
        "column_family": "BC15_anchored_multiwitness_zero_sum_pairwise_selected_product_columns" if anchored_to_bc15 else "multiwitness_zero_sum_pairwise_selected_product_columns",
        "feature_expansion": "base_plus_pairwise_products_cross_split_selected_multiwitness",
        "anchored_to_BC15": anchored_to_bc15,
        "selected_parent_descriptors": ";".join(str(v) for v in state.get("descriptors", [])),
        "selection_score_median": max(role_score_median, product_score_median),
        "selection_score_max": max(role_score_max, product_score_max),
        **norm_diag,
        "birth_logit_max_abs_error": birth_logit_error,
        "function_preservation_pass": int(birth_logit_error <= v2319.THRESHOLDS["function_preservation_float64"]),
        "old_tangent_rank": old_rank,
        "new_tangent_rank": new_rank,
        "tangent_rank_gain": new_rank - old_rank,
        "new_tangent_smallest_singular_value": new_smin,
        "target_subspace_coverage_before": cov_before,
        "target_subspace_coverage_after": cov_after,
        "target_coverage_gain": cov_after - cov_before,
        "role_operator_top_eigenvalue": max(role_score_max, product_score_max),
        "role_operator_LCB": max(role_score_median, product_score_median),
        "source_witness_role_cosine": 1.0 if max(role_score_max, product_score_max) > 0 else 0.0,
        "role_shape_G_norm": float(amp_feats.norm().detach().cpu().item()),
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
        "no_debt": no_debt,
        "role_operator_ms": role_operator_ms,
        "birth_materialization_ms": birth_materialization_ms,
        "controller_overhead_ratio": float(full_cols.numel()) / max(1.0, float(old_j.numel())),
        "peak_memory": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0.0,
        "row_wall_ms": 1000.0 * (time.perf_counter() - row_t0),
        "implementation_function_hash_present": 1,
        "implementation_function_hash": metric_hash("evaluate_part_s_row", "multiwitness_train_only_selected_product_pairwise_simplex_trust"),
        "metric_definition_present": 1,
        "used_guard_for_selection": 0,
        "manual_nonzero_role_amplitude_inserted": 0,
    }


def collect_part_p(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    write_json(OUT_ROOT / "successor_context.json", {
        "successor_part": "P",
        "v23_19_plan": rel(V2319_PLAN),
        "v23_19_final_audit": rel(ROOT / "results/v23_19/final_requirement_exhaustion_audit.json"),
        "scientific_reason": "R18 recommended stronger compositional shared-parent role architecture; Part N/O exposed a debt/random-projection-control blocker.",
        "architecture_change": "selected parent features enter as zero-sum pairwise simplex logit/tangent columns instead of all-class independent columns.",
        "formula_changed_from_v23_19": "yes_successor_not_in_version_repair",
        "gate_changed_from_v23_19": "no_successor_gate_keeps_preservation_rank_guard_gain_no_debt_and_controls",
    })
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.tasks):
        for seed in int_items(args.seeds):
            for scheme in PART_P_SCHEMES:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_part_p_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_p_contrastive_shared_parent{suffix}.csv", rows)
    audit = matrix_audit(rows)
    append_exec(
        "Part P contrastive shared-parent shard",
        "done",
        files=rel(path),
        gpu=str(args.device),
        note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} audit={audit}",
    )
    return {"part": "P", "rows": len(rows), "matrix": rel(path), "audit": audit}


def collect_part_q(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    write_json(OUT_ROOT / "successor_context_part_q.json", {
        "successor_part": "Q",
        "v23_19_plan": rel(V2319_PLAN),
        "scientific_reason": "Part P had safe positive signal on univariate/shared-role tasks but weak transfer on local/compositional tasks; v23.19 R18 recommends stronger compositional shared-parent architecture.",
        "architecture_change": "train-selected parent functions are expanded with pairwise products before zero-sum pairwise simplex logit columns.",
        "formula_changed_from_v23_19": "yes_successor_not_in_version_repair",
        "gate_changed_from_part_p": "no",
    })
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.tasks):
        for seed in int_items(args.seeds):
            for scheme in PART_Q_SCHEMES:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_part_p_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_q_compositional_contrastive_shared_parent{suffix}.csv", rows)
    audit = matrix_audit(rows)
    append_exec(
        "Part Q compositional contrastive shared-parent shard",
        "done",
        files=rel(path),
        gpu=str(args.device),
        note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} audit={audit}",
    )
    return {"part": "Q", "rows": len(rows), "matrix": rel(path), "audit": audit}


def collect_part_r(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    write_json(OUT_ROOT / "successor_context_part_r.json", {
        "successor_part": "R",
        "v23_19_plan": rel(V2319_PLAN),
        "scientific_reason": "Part Q product expansion had effect but insufficient no-debt; Part R keeps product architecture but selects a fixed small product parent set on source/witness to reduce overfit/debt.",
        "architecture_change": "base+pairwise-product parent bank is cross-split scored and reduced to fixed expanded_parent_count before zero-sum pairwise simplex columns.",
        "formula_changed_from_part_q": "yes_selector_repair_no_guard_selection",
        "gate_changed_from_part_q": "no",
        "expanded_parent_count": int(args.expanded_parent_count),
    })
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.tasks):
        for seed in int_items(args.seeds):
            for scheme in PART_R_SCHEMES:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_part_p_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_r_selected_product_contrastive_shared_parent{suffix}.csv", rows)
    audit = matrix_audit(rows)
    append_exec(
        "Part R selected-product contrastive shared-parent shard",
        "done",
        files=rel(path),
        gpu=str(args.device),
        note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} expanded_parent_count={args.expanded_parent_count} audit={audit}",
    )
    return {"part": "R", "rows": len(rows), "matrix": rel(path), "audit": audit}


def collect_part_s(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    witness_note = "witness-A and witness-B"
    if int(args.combined_witness_trust):
        witness_note = "witness-A, witness-B, and their combined train-only witness"
    write_json(OUT_ROOT / "successor_context_part_s.json", {
        "successor_part": "S",
        "v23_19_plan": rel(V2319_PLAN),
        "scientific_reason": "P/Q/R showed useful small steps but poor guard no-debt transfer; Part S replaces single-witness snapshot trust with two train-only witnesses.",
        "architecture_change": f"same selected product pairwise simplex columns as Part R, but amplitude snapshots are selected by min/mean gain and debt across {witness_note}.",
        "gate_changed_from_part_r": "no",
        "expanded_parent_count": int(args.expanded_parent_count),
        "guard_used_for_selection": 0,
        "combined_witness_trust_used": int(args.combined_witness_trust),
    })
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.tasks):
        for seed in int_items(args.seeds):
            for scheme in PART_S_SCHEMES:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_part_s_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_s_multiwitness_selected_product_trust{suffix}.csv", rows)
    audit = matrix_audit(rows)
    append_exec(
        "Part S multiwitness selected-product trust shard",
        "done",
        files=rel(path),
        gpu=str(args.device),
        note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} expanded_parent_count={args.expanded_parent_count} audit={audit}",
    )
    return {"part": "S", "rows": len(rows), "matrix": rel(path), "audit": audit}


def collect_part_t(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    witness_note = "witness-A and witness-B"
    if int(args.combined_witness_trust):
        witness_note = "witness-A, witness-B, and their combined train-only witness"
    write_json(OUT_ROOT / "successor_context_part_t.json", {
        "successor_part": "T",
        "v23_19_plan": rel(V2319_PLAN),
        "scientific_reason": "Part S was safe but below BC15; Part T trains selected-product role amplitudes as a train-only residual correction on top of BC15 logits.",
        "architecture_change": f"BC15-anchored multiwitness selected-product pairwise simplex role exploitation with trust checked across {witness_note}.",
        "gate_changed_from_part_s": "no",
        "expanded_parent_count": int(args.expanded_parent_count),
        "guard_used_for_selection": 0,
        "combined_witness_trust_used": int(args.combined_witness_trust),
    })
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.tasks):
        for seed in int_items(args.seeds):
            for scheme in PART_T_SCHEMES:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_part_s_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_t_bc15anchored_multiwitness_selected_product{suffix}.csv", rows)
    audit = matrix_audit(rows)
    append_exec(
        "Part T BC15-anchored multiwitness selected-product shard",
        "done",
        files=rel(path),
        gpu=str(args.device),
        note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} expanded_parent_count={args.expanded_parent_count} audit={audit}",
    )
    return {"part": "T", "rows": len(rows), "matrix": rel(path), "audit": audit}


def collect_part_u(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    witness_note = "witness-A and witness-B"
    if int(args.combined_witness_trust):
        witness_note = "witness-A, witness-B, and their combined train-only witness"
    write_json(OUT_ROOT / "successor_context_part_u_real_falsification.json", {
        "successor_part": "U",
        "v23_19_plan": rel(V2319_PLAN),
        "scientific_reason": "Part T opened a synthetic successor candidate; Part U runs minimum real-task falsification before any broader promotion.",
        "architecture_change": f"no further architecture change from Part T; real tasks only; trust checks {witness_note}.",
        "real_tasks": csv_items(args.real_tasks),
        "seeds": int_items(args.real_seeds),
        "schemes": PART_U_REAL_SCHEMES,
        "guard_used_for_selection": 0,
        "combined_witness_trust_used": int(args.combined_witness_trust),
    })
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.real_tasks):
        for seed in int_items(args.real_seeds):
            for scheme in PART_U_REAL_SCHEMES:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        row = evaluate_part_s_row(args, task, seed, scheme)
        row["part"] = "U"
        row["minimum_real_falsification"] = 1
        rows.append(row)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_u_real_falsification_bc15anchored_multiwitness{suffix}.csv", rows)
    audit = matrix_audit(rows)
    append_exec(
        "Part U real falsification BC15-anchored multiwitness shard",
        "done",
        files=rel(path),
        gpu=str(args.device),
        note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} real_tasks={args.real_tasks} audit={audit}",
    )
    return {"part": "U", "rows": len(rows), "matrix": rel(path), "audit": audit}


def merge_csvs(pattern: str, out: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob(pattern)):
        rows.extend(read_rows(path))
    if rows:
        write_rows(out, rows)
    return rows


def summarize_part_p(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    schemes = [s for s in PART_P_SCHEMES if any(str(r.get("scheme")) == s for r in rows)]
    for scheme in schemes:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "median_parent_count": median(r.get("parent_count", 0) for r in ss),
            "median_role_columns": median(r.get("role_columns", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_train_witness_NLL_gain": median(r.get("train_witness_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "selected_nonzero_rows": sum(1 for r in ss if ival(r.get("selected_step", 0)) > 0),
            "median_selected_step": median(r.get("selected_step", 0) for r in ss),
            "median_guard_debt_excess": median(r.get("guard_debt_excess", 0) for r in ss),
            "median_selection_score_max": median(r.get("selection_score_max", 0) for r in ss),
            "median_column_scale_min": median(r.get("column_scale_min", 0) for r in ss),
            "median_column_scale_max": median(r.get("column_scale_max", 0) for r in ss),
        })
    by = {r["scheme"]: r for r in summary_rows}
    primary = by.get("P0_KAN_spline_pair_simplex_hard_trust", {})
    label = by.get("P1_label_shuffled_KAN_spline_pair_hard_trust", {})
    random_capacity = by.get("P2_same_capacity_random_pair_hard_trust", {})
    mlp = by.get("P3_MLP_matched_selected_tanh_pair_hard_trust", {})
    random_projection = by.get("P4_random_projection_spline_selected_pair_hard_trust", {})
    more_parents = by.get("P5_KAN_spline_pair_more_parents_hard_trust", {})
    oracle = by.get("P6_oracle_task_pair_upper", {})
    control_best = max(
        fval(label.get("median_guard_NLL_gain", 0)),
        fval(random_capacity.get("median_guard_NLL_gain", 0)),
        fval(mlp.get("median_guard_NLL_gain", 0)),
        fval(random_projection.get("median_guard_NLL_gain", 0)),
    )
    gate = int(
        len(rows) > 0
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(primary.get("no_debt_rows", 0)) >= 24
        and fval(primary.get("median_guard_NLL_gain", 0)) > control_best
    )
    blockers: list[str] = []
    if fval(primary.get("preservation_pass_rate", 0)) < 0.95:
        blockers.append("function_preservation_failed")
    if fval(primary.get("rank_gain_positive_rate", 0)) < 0.80:
        blockers.append("tangent_expansion_insufficient")
    if fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) < 1.0e-4:
        blockers.append("below_BC15_or_effect_floor")
    if ival(primary.get("no_debt_rows", 0)) < 24:
        blockers.append("no_debt_insufficient")
    if fval(primary.get("median_guard_NLL_gain", 0)) <= control_best:
        blockers.append("control_explains_or_beats_primary")
    if not blockers:
        blockers.append("none")
    summary = {
        "part": "P",
        "successor_repair": "contrastive_pairwise_simplex_shared_parent_role_architecture",
        "rows": len(rows),
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_p_contrastive_shared_parent.csv"),
        "P0_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "P0_minus_same_capacity_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_capacity.get("median_guard_NLL_gain", 0)),
        "P0_minus_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "P0_minus_random_projection_selected_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_projection.get("median_guard_NLL_gain", 0)),
        "P0_minus_more_parents_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(more_parents.get("median_guard_NLL_gain", 0)),
        "P0_minus_oracle_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(oracle.get("median_guard_NLL_gain", 0)),
        "P0_no_debt_rows": primary.get("no_debt_rows", 0),
        "oracle_median_guard_NLL_gain": oracle.get("median_guard_NLL_gain", 0),
        "control_best_median_guard_NLL": control_best,
        "dominant_blockers": blockers,
        "interpretation": "successor_candidate_opened" if gate else "successor_pairwise_architecture_not_sufficient_under_strict_gate",
    }
    return summary_rows, summary


def merge_part_p(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_p_contrastive_shared_parent_shard*_of_*.csv", OUT_ROOT / "part_p_contrastive_shared_parent.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_p_contrastive_shared_parent.csv")
    summary_rows, summary = summarize_part_p(rows)
    summary_path = write_rows(OUT_ROOT / "part_p_contrastive_shared_parent_summary.csv", summary_rows)
    summary["summary"] = rel(summary_path)
    matrix_path = OUT_ROOT / "part_p_contrastive_shared_parent.csv"
    audit = matrix_audit(rows)
    summary["matrix_audit"] = audit
    summary_path_json = write_json(OUT_ROOT / "part_p_contrastive_shared_parent_summary.json", summary)
    next_action = (
        "promote_to_full_v23_20_plan_with_real_falsification_and_official_controls"
        if ival(summary.get("gate_pass", 0))
        else "do_not_claim_success; inspect blockers and consider a new pre-registered estimator/materialization repair rather than ordinary sweeps"
    )
    next_path = write_json(OUT_ROOT / "part_p_next_actions_for_codex.json", {
        "part": "P",
        "gate_pass": summary["gate_pass"],
        "next": next_action,
        "prohibited_shortcuts": [
            "do_not_lower_no_debt_requirement",
            "do_not_remove_random_projection_control",
            "do_not_tune_on_guard",
            "do_not_call_success_without_real_falsification",
        ],
    })
    fail_path = write_json(OUT_ROOT / "part_p_failure_decomposition.json", {
        "part": "P",
        "gate_pass": summary["gate_pass"],
        "dominant_blockers": summary["dominant_blockers"],
        "summary": summary,
        "audit": audit,
    })
    append_exec(
        "Part P contrastive shared-parent merge",
        "done",
        files=f"{rel(matrix_path)}; {rel(summary_path)}; {rel(summary_path_json)}; {rel(next_path)}; {rel(fail_path)}",
        gpu=str(args.device),
        note=f"rows={len(rows)} gate={summary['gate_pass']} audit={audit}",
    )
    append_recap("Part P contrastive shared-parent results", summary)
    return summary


def summarize_part_q(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for scheme in [s for s in PART_Q_SCHEMES if any(str(r.get("scheme")) == s for r in rows)]:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "median_base_parent_count": median(r.get("base_parent_count", 0) for r in ss),
            "median_expanded_parent_count": median(r.get("expanded_parent_count", 0) for r in ss),
            "median_role_columns": median(r.get("role_columns", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_train_witness_NLL_gain": median(r.get("train_witness_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "selected_nonzero_rows": sum(1 for r in ss if ival(r.get("selected_step", 0)) > 0),
            "median_selected_step": median(r.get("selected_step", 0) for r in ss),
            "median_guard_debt_excess": median(r.get("guard_debt_excess", 0) for r in ss),
            "median_train_witness_raw_debt_excess": median(r.get("train_witness_raw_debt_excess", 0) for r in ss),
            "median_selection_score_max": median(r.get("selection_score_max", 0) for r in ss),
        })
    by = {r["scheme"]: r for r in summary_rows}
    primary = by.get("Q0_KAN_spline_product_pair_simplex_buffered_trust", {})
    label = by.get("Q1_label_shuffled_KAN_spline_product_pair_buffered_trust", {})
    random_capacity = by.get("Q2_same_capacity_random_product_pair_buffered_trust", {})
    mlp = by.get("Q3_MLP_matched_tanh_product_pair_buffered_trust", {})
    random_projection = by.get("Q4_random_projection_spline_product_pair_buffered_trust", {})
    oracle = by.get("Q5_oracle_task_product_pair_upper", {})
    control_best = max(
        fval(label.get("median_guard_NLL_gain", 0)),
        fval(random_capacity.get("median_guard_NLL_gain", 0)),
        fval(mlp.get("median_guard_NLL_gain", 0)),
        fval(random_projection.get("median_guard_NLL_gain", 0)),
    )
    gate = int(
        len(rows) > 0
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(primary.get("no_debt_rows", 0)) >= 24
        and fval(primary.get("median_guard_NLL_gain", 0)) > control_best
    )
    blockers: list[str] = []
    if fval(primary.get("preservation_pass_rate", 0)) < 0.95:
        blockers.append("function_preservation_failed")
    if fval(primary.get("rank_gain_positive_rate", 0)) < 0.80:
        blockers.append("tangent_expansion_insufficient")
    if fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) < 1.0e-4:
        blockers.append("below_BC15_or_effect_floor")
    if ival(primary.get("no_debt_rows", 0)) < 24:
        blockers.append("no_debt_insufficient")
    if fval(primary.get("median_guard_NLL_gain", 0)) <= control_best:
        blockers.append("control_explains_or_beats_primary")
    if not blockers:
        blockers.append("none")
    summary = {
        "part": "Q",
        "successor_repair": "compositional_product_pairwise_simplex_shared_parent_role_architecture",
        "rows": len(rows),
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_q_compositional_contrastive_shared_parent.csv"),
        "Q0_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "Q0_minus_same_capacity_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_capacity.get("median_guard_NLL_gain", 0)),
        "Q0_minus_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "Q0_minus_random_projection_selected_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_projection.get("median_guard_NLL_gain", 0)),
        "Q0_minus_oracle_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(oracle.get("median_guard_NLL_gain", 0)),
        "Q0_no_debt_rows": primary.get("no_debt_rows", 0),
        "oracle_median_guard_NLL_gain": oracle.get("median_guard_NLL_gain", 0),
        "control_best_median_guard_NLL": control_best,
        "dominant_blockers": blockers,
        "interpretation": "successor_compositional_candidate_opened" if gate else "successor_compositional_architecture_not_sufficient_under_strict_gate",
    }
    return summary_rows, summary


def merge_part_q(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_q_compositional_contrastive_shared_parent_shard*_of_*.csv", OUT_ROOT / "part_q_compositional_contrastive_shared_parent.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_q_compositional_contrastive_shared_parent.csv")
    summary_rows, summary = summarize_part_q(rows)
    summary_path = write_rows(OUT_ROOT / "part_q_compositional_contrastive_shared_parent_summary.csv", summary_rows)
    summary["summary"] = rel(summary_path)
    matrix_path = OUT_ROOT / "part_q_compositional_contrastive_shared_parent.csv"
    audit = matrix_audit(rows)
    summary["matrix_audit"] = audit
    summary_path_json = write_json(OUT_ROOT / "part_q_compositional_contrastive_shared_parent_summary.json", summary)
    next_path = write_json(OUT_ROOT / "part_q_next_actions_for_codex.json", {
        "part": "Q",
        "gate_pass": summary["gate_pass"],
        "next": "promote_to_full_v23_20_plan_with_real_falsification_and_official_controls" if summary["gate_pass"] else "do_not_claim_success; product shared-parent still fails strict successor gate",
    })
    fail_path = write_json(OUT_ROOT / "part_q_failure_decomposition.json", {
        "part": "Q",
        "gate_pass": summary["gate_pass"],
        "dominant_blockers": summary["dominant_blockers"],
        "summary": summary,
        "audit": audit,
    })
    append_exec(
        "Part Q compositional contrastive shared-parent merge",
        "done",
        files=f"{rel(matrix_path)}; {rel(summary_path)}; {rel(summary_path_json)}; {rel(next_path)}; {rel(fail_path)}",
        gpu=str(args.device),
        note=f"rows={len(rows)} gate={summary['gate_pass']} audit={audit}",
    )
    append_recap("Part Q compositional contrastive shared-parent results", summary)
    return summary


def summarize_part_r(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for scheme in [s for s in PART_R_SCHEMES if any(str(r.get("scheme")) == s for r in rows)]:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "median_base_parent_count": median(r.get("base_parent_count", 0) for r in ss),
            "median_expanded_parent_count": median(r.get("expanded_parent_count", 0) for r in ss),
            "median_role_columns": median(r.get("role_columns", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_train_witness_NLL_gain": median(r.get("train_witness_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "selected_nonzero_rows": sum(1 for r in ss if ival(r.get("selected_step", 0)) > 0),
            "median_selected_step": median(r.get("selected_step", 0) for r in ss),
            "median_guard_debt_excess": median(r.get("guard_debt_excess", 0) for r in ss),
            "median_train_witness_raw_debt_excess": median(r.get("train_witness_raw_debt_excess", 0) for r in ss),
            "median_selection_score_max": median(r.get("selection_score_max", 0) for r in ss),
        })
    by = {r["scheme"]: r for r in summary_rows}
    primary = by.get("R0_KAN_spline_selected_product_pair_small_trust", {})
    label = by.get("R1_label_shuffled_KAN_spline_selected_product_pair", {})
    random_capacity = by.get("R2_same_capacity_random_selected_product_pair", {})
    mlp = by.get("R3_MLP_matched_tanh_selected_product_pair", {})
    random_projection = by.get("R4_random_projection_spline_selected_product_pair", {})
    oracle = by.get("R5_oracle_task_selected_product_pair_upper", {})
    control_best = max(
        fval(label.get("median_guard_NLL_gain", 0)),
        fval(random_capacity.get("median_guard_NLL_gain", 0)),
        fval(mlp.get("median_guard_NLL_gain", 0)),
        fval(random_projection.get("median_guard_NLL_gain", 0)),
    )
    gate = int(
        len(rows) > 0
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(primary.get("no_debt_rows", 0)) >= 24
        and fval(primary.get("median_guard_NLL_gain", 0)) > control_best
    )
    blockers: list[str] = []
    if fval(primary.get("preservation_pass_rate", 0)) < 0.95:
        blockers.append("function_preservation_failed")
    if fval(primary.get("rank_gain_positive_rate", 0)) < 0.80:
        blockers.append("tangent_expansion_insufficient")
    if fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) < 1.0e-4:
        blockers.append("below_BC15_or_effect_floor")
    if ival(primary.get("no_debt_rows", 0)) < 24:
        blockers.append("no_debt_insufficient")
    if fval(primary.get("median_guard_NLL_gain", 0)) <= control_best:
        blockers.append("control_explains_or_beats_primary")
    if not blockers:
        blockers.append("none")
    summary = {
        "part": "R",
        "successor_repair": "cross_split_selected_compositional_product_pairwise_simplex_role_architecture",
        "rows": len(rows),
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_r_selected_product_contrastive_shared_parent.csv"),
        "R0_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "R0_minus_same_capacity_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_capacity.get("median_guard_NLL_gain", 0)),
        "R0_minus_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "R0_minus_random_projection_selected_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_projection.get("median_guard_NLL_gain", 0)),
        "R0_minus_oracle_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(oracle.get("median_guard_NLL_gain", 0)),
        "R0_no_debt_rows": primary.get("no_debt_rows", 0),
        "oracle_median_guard_NLL_gain": oracle.get("median_guard_NLL_gain", 0),
        "control_best_median_guard_NLL": control_best,
        "dominant_blockers": blockers,
        "interpretation": "successor_selected_product_candidate_opened" if gate else "successor_selected_product_architecture_not_sufficient_under_strict_gate",
    }
    return summary_rows, summary


def merge_part_r(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_r_selected_product_contrastive_shared_parent_shard*_of_*.csv", OUT_ROOT / "part_r_selected_product_contrastive_shared_parent.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_r_selected_product_contrastive_shared_parent.csv")
    summary_rows, summary = summarize_part_r(rows)
    summary_path = write_rows(OUT_ROOT / "part_r_selected_product_contrastive_shared_parent_summary.csv", summary_rows)
    summary["summary"] = rel(summary_path)
    matrix_path = OUT_ROOT / "part_r_selected_product_contrastive_shared_parent.csv"
    audit = matrix_audit(rows)
    summary["matrix_audit"] = audit
    summary_path_json = write_json(OUT_ROOT / "part_r_selected_product_contrastive_shared_parent_summary.json", summary)
    next_path = write_json(OUT_ROOT / "part_r_next_actions_for_codex.json", {
        "part": "R",
        "gate_pass": summary["gate_pass"],
        "next": "promote_to_full_v23_20_plan_with_real_falsification_and_official_controls" if summary["gate_pass"] else "do_not_claim_success; selected product repair still fails strict successor gate",
    })
    fail_path = write_json(OUT_ROOT / "part_r_failure_decomposition.json", {
        "part": "R",
        "gate_pass": summary["gate_pass"],
        "dominant_blockers": summary["dominant_blockers"],
        "summary": summary,
        "audit": audit,
    })
    append_exec(
        "Part R selected-product contrastive shared-parent merge",
        "done",
        files=f"{rel(matrix_path)}; {rel(summary_path)}; {rel(summary_path_json)}; {rel(next_path)}; {rel(fail_path)}",
        gpu=str(args.device),
        note=f"rows={len(rows)} gate={summary['gate_pass']} audit={audit}",
    )
    append_recap("Part R selected-product contrastive shared-parent results", summary)
    return summary


def summarize_part_s(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for scheme in [s for s in PART_S_SCHEMES if any(str(r.get("scheme")) == s for r in rows)]:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "median_base_parent_count": median(r.get("base_parent_count", 0) for r in ss),
            "median_expanded_parent_count": median(r.get("expanded_parent_count", 0) for r in ss),
            "median_role_columns": median(r.get("role_columns", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_multiwitness_min_gain": median(r.get("multiwitness_min_gain", 0) for r in ss),
            "median_multiwitness_mean_gain": median(r.get("multiwitness_mean_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "selected_nonzero_rows": sum(1 for r in ss if ival(r.get("selected_step", 0)) > 0),
            "median_selected_step": median(r.get("selected_step", 0) for r in ss),
            "median_guard_debt_excess": median(r.get("guard_debt_excess", 0) for r in ss),
            "median_train_witness_raw_debt_excess": median(r.get("train_witness_raw_debt_excess", 0) for r in ss),
            "median_selection_score_max": median(r.get("selection_score_max", 0) for r in ss),
        })
    by = {r["scheme"]: r for r in summary_rows}
    primary = by.get("S0_KAN_spline_selected_product_pair_multiwitness_trust", {})
    label = by.get("S1_label_shuffled_KAN_spline_selected_product_pair_multiwitness", {})
    random_capacity = by.get("S2_same_capacity_random_selected_product_pair_multiwitness", {})
    mlp = by.get("S3_MLP_matched_tanh_selected_product_pair_multiwitness", {})
    random_projection = by.get("S4_random_projection_spline_selected_product_pair_multiwitness", {})
    oracle = by.get("S5_oracle_task_selected_product_pair_multiwitness_upper", {})
    control_best = max(
        fval(label.get("median_guard_NLL_gain", 0)),
        fval(random_capacity.get("median_guard_NLL_gain", 0)),
        fval(mlp.get("median_guard_NLL_gain", 0)),
        fval(random_projection.get("median_guard_NLL_gain", 0)),
    )
    gate = int(
        len(rows) > 0
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(primary.get("no_debt_rows", 0)) >= 24
        and fval(primary.get("median_guard_NLL_gain", 0)) > control_best
    )
    blockers: list[str] = []
    if fval(primary.get("preservation_pass_rate", 0)) < 0.95:
        blockers.append("function_preservation_failed")
    if fval(primary.get("rank_gain_positive_rate", 0)) < 0.80:
        blockers.append("tangent_expansion_insufficient")
    if fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) < 1.0e-4:
        blockers.append("below_BC15_or_effect_floor")
    if ival(primary.get("no_debt_rows", 0)) < 24:
        blockers.append("no_debt_insufficient")
    if fval(primary.get("median_guard_NLL_gain", 0)) <= control_best:
        blockers.append("control_explains_or_beats_primary")
    if not blockers:
        blockers.append("none")
    summary = {
        "part": "S",
        "successor_repair": "multiwitness_train_only_selected_product_pairwise_simplex_trust",
        "rows": len(rows),
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_s_multiwitness_selected_product_trust.csv"),
        "S0_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "S0_minus_same_capacity_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_capacity.get("median_guard_NLL_gain", 0)),
        "S0_minus_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "S0_minus_random_projection_selected_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_projection.get("median_guard_NLL_gain", 0)),
        "S0_minus_oracle_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(oracle.get("median_guard_NLL_gain", 0)),
        "S0_no_debt_rows": primary.get("no_debt_rows", 0),
        "oracle_median_guard_NLL_gain": oracle.get("median_guard_NLL_gain", 0),
        "control_best_median_guard_NLL": control_best,
        "dominant_blockers": blockers,
        "interpretation": "successor_multiwitness_candidate_opened" if gate else "successor_multiwitness_trust_not_sufficient_under_strict_gate",
    }
    return summary_rows, summary


def merge_part_s(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_s_multiwitness_selected_product_trust_shard*_of_*.csv", OUT_ROOT / "part_s_multiwitness_selected_product_trust.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_s_multiwitness_selected_product_trust.csv")
    summary_rows, summary = summarize_part_s(rows)
    summary_path = write_rows(OUT_ROOT / "part_s_multiwitness_selected_product_trust_summary.csv", summary_rows)
    summary["summary"] = rel(summary_path)
    matrix_path = OUT_ROOT / "part_s_multiwitness_selected_product_trust.csv"
    audit = matrix_audit(rows)
    summary["matrix_audit"] = audit
    summary_path_json = write_json(OUT_ROOT / "part_s_multiwitness_selected_product_trust_summary.json", summary)
    next_path = write_json(OUT_ROOT / "part_s_next_actions_for_codex.json", {
        "part": "S",
        "gate_pass": summary["gate_pass"],
        "next": "promote_to_full_v23_20_plan_with_real_falsification_and_official_controls" if summary["gate_pass"] else "do_not_claim_success; multiwitness trust still fails strict successor gate",
    })
    fail_path = write_json(OUT_ROOT / "part_s_failure_decomposition.json", {
        "part": "S",
        "gate_pass": summary["gate_pass"],
        "dominant_blockers": summary["dominant_blockers"],
        "summary": summary,
        "audit": audit,
    })
    append_exec(
        "Part S multiwitness selected-product trust merge",
        "done",
        files=f"{rel(matrix_path)}; {rel(summary_path)}; {rel(summary_path_json)}; {rel(next_path)}; {rel(fail_path)}",
        gpu=str(args.device),
        note=f"rows={len(rows)} gate={summary['gate_pass']} audit={audit}",
    )
    append_recap("Part S multiwitness selected-product trust results", summary)
    return summary


def summarize_part_t(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for scheme in [s for s in PART_T_SCHEMES if any(str(r.get("scheme")) == s for r in rows)]:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "median_base_parent_count": median(r.get("base_parent_count", 0) for r in ss),
            "median_expanded_parent_count": median(r.get("expanded_parent_count", 0) for r in ss),
            "median_role_columns": median(r.get("role_columns", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_multiwitness_min_gain": median(r.get("multiwitness_min_gain", 0) for r in ss),
            "median_multiwitness_mean_gain": median(r.get("multiwitness_mean_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "selected_nonzero_rows": sum(1 for r in ss if ival(r.get("selected_step", 0)) > 0),
            "median_selected_step": median(r.get("selected_step", 0) for r in ss),
            "median_guard_debt_excess": median(r.get("guard_debt_excess", 0) for r in ss),
            "median_train_witness_raw_debt_excess": median(r.get("train_witness_raw_debt_excess", 0) for r in ss),
            "median_selection_score_max": median(r.get("selection_score_max", 0) for r in ss),
        })
    by = {r["scheme"]: r for r in summary_rows}
    primary = by.get("T0_KAN_spline_selected_product_pair_BC15anchored_multiwitness", {})
    label = by.get("T1_label_shuffled_KAN_spline_selected_product_pair_BC15anchored", {})
    random_capacity = by.get("T2_same_capacity_random_selected_product_pair_BC15anchored", {})
    mlp = by.get("T3_MLP_matched_tanh_selected_product_pair_BC15anchored", {})
    random_projection = by.get("T4_random_projection_spline_selected_product_pair_BC15anchored", {})
    oracle = by.get("T5_oracle_task_selected_product_pair_BC15anchored_upper", {})
    control_best = max(
        fval(label.get("median_guard_NLL_gain", 0)),
        fval(random_capacity.get("median_guard_NLL_gain", 0)),
        fval(mlp.get("median_guard_NLL_gain", 0)),
        fval(random_projection.get("median_guard_NLL_gain", 0)),
    )
    gate = int(
        len(rows) > 0
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(primary.get("no_debt_rows", 0)) >= 24
        and fval(primary.get("median_guard_NLL_gain", 0)) > control_best
    )
    blockers: list[str] = []
    if fval(primary.get("preservation_pass_rate", 0)) < 0.95:
        blockers.append("function_preservation_failed")
    if fval(primary.get("rank_gain_positive_rate", 0)) < 0.80:
        blockers.append("tangent_expansion_insufficient")
    if fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) < 1.0e-4:
        blockers.append("below_BC15_or_effect_floor")
    if ival(primary.get("no_debt_rows", 0)) < 24:
        blockers.append("no_debt_insufficient")
    if fval(primary.get("median_guard_NLL_gain", 0)) <= control_best:
        blockers.append("control_explains_or_beats_primary")
    if not blockers:
        blockers.append("none")
    summary = {
        "part": "T",
        "successor_repair": "BC15anchored_multiwitness_selected_product_pairwise_simplex_trust",
        "rows": len(rows),
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_t_bc15anchored_multiwitness_selected_product.csv"),
        "T0_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "T0_minus_same_capacity_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_capacity.get("median_guard_NLL_gain", 0)),
        "T0_minus_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "T0_minus_random_projection_selected_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_projection.get("median_guard_NLL_gain", 0)),
        "T0_minus_oracle_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(oracle.get("median_guard_NLL_gain", 0)),
        "T0_no_debt_rows": primary.get("no_debt_rows", 0),
        "oracle_median_guard_NLL_gain": oracle.get("median_guard_NLL_gain", 0),
        "control_best_median_guard_NLL": control_best,
        "dominant_blockers": blockers,
        "interpretation": "successor_BC15anchored_candidate_opened" if gate else "successor_BC15anchored_trust_not_sufficient_under_strict_gate",
    }
    return summary_rows, summary


def merge_part_t(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_t_bc15anchored_multiwitness_selected_product_shard*_of_*.csv", OUT_ROOT / "part_t_bc15anchored_multiwitness_selected_product.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_t_bc15anchored_multiwitness_selected_product.csv")
    summary_rows, summary = summarize_part_t(rows)
    summary_path = write_rows(OUT_ROOT / "part_t_bc15anchored_multiwitness_selected_product_summary.csv", summary_rows)
    summary["summary"] = rel(summary_path)
    matrix_path = OUT_ROOT / "part_t_bc15anchored_multiwitness_selected_product.csv"
    audit = matrix_audit(rows)
    summary["matrix_audit"] = audit
    summary_path_json = write_json(OUT_ROOT / "part_t_bc15anchored_multiwitness_selected_product_summary.json", summary)
    next_path = write_json(OUT_ROOT / "part_t_next_actions_for_codex.json", {
        "part": "T",
        "gate_pass": summary["gate_pass"],
        "next": "promote_to_full_v23_20_plan_with_real_falsification_and_official_controls" if summary["gate_pass"] else "do_not_claim_success; BC15-anchored multiwitness trust still fails strict successor gate",
    })
    fail_path = write_json(OUT_ROOT / "part_t_failure_decomposition.json", {
        "part": "T",
        "gate_pass": summary["gate_pass"],
        "dominant_blockers": summary["dominant_blockers"],
        "summary": summary,
        "audit": audit,
    })
    append_exec(
        "Part T BC15-anchored multiwitness selected-product merge",
        "done",
        files=f"{rel(matrix_path)}; {rel(summary_path)}; {rel(summary_path_json)}; {rel(next_path)}; {rel(fail_path)}",
        gpu=str(args.device),
        note=f"rows={len(rows)} gate={summary['gate_pass']} audit={audit}",
    )
    append_recap("Part T BC15-anchored multiwitness selected-product results", summary)
    return summary


def summarize_part_u(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for scheme in [s for s in PART_U_REAL_SCHEMES if any(str(r.get("scheme")) == s for r in rows)]:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "real_task_count": len({r.get("task") for r in ss}),
            "median_output_dim": median(r.get("output_dim", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
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
    primary = by.get("T0_KAN_spline_selected_product_pair_BC15anchored_multiwitness", {})
    label = by.get("T1_label_shuffled_KAN_spline_selected_product_pair_BC15anchored", {})
    random_capacity = by.get("T2_same_capacity_random_selected_product_pair_BC15anchored", {})
    mlp = by.get("T3_MLP_matched_tanh_selected_product_pair_BC15anchored", {})
    random_projection = by.get("T4_random_projection_spline_selected_product_pair_BC15anchored", {})
    control_best = max(
        fval(label.get("median_guard_NLL_gain", 0)),
        fval(random_capacity.get("median_guard_NLL_gain", 0)),
        fval(mlp.get("median_guard_NLL_gain", 0)),
        fval(random_projection.get("median_guard_NLL_gain", 0)),
    )
    expected_rows = len(csv_items(args.real_tasks)) * len(int_items(args.real_seeds)) * len(PART_U_REAL_SCHEMES)
    gate = int(
        len(rows) == expected_rows
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(primary.get("no_debt_rows", 0)) >= 12
        and fval(primary.get("median_guard_NLL_gain", 0)) > control_best
    )
    blockers: list[str] = []
    if len(rows) != expected_rows:
        blockers.append("incomplete_real_matrix")
    if fval(primary.get("preservation_pass_rate", 0)) < 0.95:
        blockers.append("function_preservation_insufficient")
    if fval(primary.get("rank_gain_positive_rate", 0)) < 0.80:
        blockers.append("tangent_rank_gain_insufficient")
    if fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) < 1.0e-4:
        blockers.append("below_BC15_or_effect_floor")
    if ival(primary.get("no_debt_rows", 0)) < 12:
        blockers.append("no_debt_insufficient")
    if fval(primary.get("median_guard_NLL_gain", 0)) <= control_best:
        blockers.append("control_explains_or_beats_primary")
    if not blockers:
        blockers.append("none")
    summary = {
        "part": "U",
        "successor_repair": "minimum_real_falsification_for_T_BC15anchored_multiwitness",
        "rows": len(rows),
        "expected_rows": expected_rows,
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_u_real_falsification_bc15anchored_multiwitness.csv"),
        "U_T0_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "U_T0_minus_same_capacity_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_capacity.get("median_guard_NLL_gain", 0)),
        "U_T0_minus_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "U_T0_minus_random_projection_selected_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_projection.get("median_guard_NLL_gain", 0)),
        "U_T0_no_debt_rows": primary.get("no_debt_rows", 0),
        "control_best_median_guard_NLL": control_best,
        "dominant_blockers": blockers,
        "interpretation": "minimum_real_falsification_passed_for_successor_candidate" if gate else "minimum_real_falsification_not_passed",
    }
    return summary_rows, summary


def merge_part_u(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_u_real_falsification_bc15anchored_multiwitness_shard*_of_*.csv", OUT_ROOT / "part_u_real_falsification_bc15anchored_multiwitness.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_u_real_falsification_bc15anchored_multiwitness.csv")
    summary_rows, summary = summarize_part_u(args, rows)
    summary_path = write_rows(OUT_ROOT / "part_u_real_falsification_bc15anchored_multiwitness_summary.csv", summary_rows)
    summary["summary"] = rel(summary_path)
    matrix_path = OUT_ROOT / "part_u_real_falsification_bc15anchored_multiwitness.csv"
    audit = matrix_audit(rows)
    summary["matrix_audit"] = audit
    summary_path_json = write_json(OUT_ROOT / "part_u_real_falsification_bc15anchored_multiwitness_summary.json", summary)
    next_path = write_json(OUT_ROOT / "part_u_next_actions_for_codex.json", {
        "part": "U",
        "gate_pass": summary["gate_pass"],
        "next": "promote_to_full_v23_20_official_plan" if summary["gate_pass"] else "do_not_claim_real_success; inspect real-task blocker before expansion",
    })
    fail_path = write_json(OUT_ROOT / "part_u_failure_decomposition.json", {
        "part": "U",
        "gate_pass": summary["gate_pass"],
        "dominant_blockers": summary["dominant_blockers"],
        "summary": summary,
        "audit": audit,
    })
    append_exec(
        "Part U real falsification BC15-anchored multiwitness merge",
        "done",
        files=f"{rel(matrix_path)}; {rel(summary_path)}; {rel(summary_path_json)}; {rel(next_path)}; {rel(fail_path)}",
        gpu=str(args.device),
        note=f"rows={len(rows)} gate={summary['gate_pass']} audit={audit}",
    )
    append_recap("Part U real falsification BC15-anchored multiwitness results", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="part-p", choices=["part-p", "part-p-merge", "part-q", "part-q-merge", "part-r", "part-r-merge", "part-s", "part-s-merge", "part-t", "part-t-merge", "part-u-real", "part-u-real-merge"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--basis-input-gain", type=float, default=0.25)
    p.add_argument("--quadrature-points", type=int, default=257)
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
    p.add_argument("--role-lr", type=float, default=0.20)
    p.add_argument("--debt-tolerance", type=float, default=1.0e-3)
    p.add_argument("--trust-debt-tolerance", type=float, default=1.0e-3)
    p.add_argument("--trust-raw-debt-penalty", type=float, default=0.0)
    p.add_argument("--combined-witness-trust", type=int, default=0)
    p.add_argument("--mlp-role-count", type=int, default=1)
    p.add_argument("--float64", type=int, default=1)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--real-tasks", default=",".join(REAL_FALSIFICATION_TASKS))
    p.add_argument("--real-seeds", default="0,1,2")
    p.add_argument("--trust-steps", type=int, default=60)
    p.add_argument("--trust-lr-grid", default="0.003,0.01,0.03")
    p.add_argument("--amp-l2", type=float, default=1.0e-3)
    p.add_argument("--parent-count", type=int, default=8)
    p.add_argument("--expanded-parent-count", type=int, default=8)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--max-jobs", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.mode == "part-p":
        return collect_part_p(args)
    if args.mode == "part-p-merge":
        return merge_part_p(args)
    if args.mode == "part-q":
        return collect_part_q(args)
    if args.mode == "part-q-merge":
        return merge_part_q(args)
    if args.mode == "part-r":
        return collect_part_r(args)
    if args.mode == "part-r-merge":
        return merge_part_r(args)
    if args.mode == "part-s":
        return collect_part_s(args)
    if args.mode == "part-s-merge":
        return merge_part_s(args)
    if args.mode == "part-t":
        return collect_part_t(args)
    if args.mode == "part-t-merge":
        return merge_part_t(args)
    if args.mode == "part-u-real":
        return collect_part_u(args)
    if args.mode == "part-u-real-merge":
        return merge_part_u(args)
    raise ValueError(args.mode)


if __name__ == "__main__":
    main()
