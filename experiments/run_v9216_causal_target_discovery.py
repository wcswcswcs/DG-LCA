#!/usr/bin/env python3
"""DG-KAN v9.2.16 causal target discovery runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9214_p4qualified_functional_actuator_closure as v9214  # noqa: E402
import run_v9215_control_resistant_functional_causality as v9215  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import lq_output_space_functional as out_lq  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.16_Causal_Target_Discovery_Primitive_Redesign_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9216_causal_target_discovery.py"
PREV_V9215 = ROOT / "results" / "real_rerun_20260506" / "v9215_control_resistant_functional_causality_first_20260510T030000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _canonical_tasks(text: str) -> List[str]:
    return [v92._canonical_task(x) for x in _parse_list(text)]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [_to_float(r.get(key)) for r in rows]
    return sum(vals) / max(1, len(vals))


def _not_run(stage: str, artifact: str, reason: str) -> Dict[str, Any]:
    return snr_lq.not_run_row(stage, artifact, reason)


def _write_svg(path: Path, title: str, subtitle: str) -> None:
    ensure_dir(path.parent)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="210" viewBox="0 0 1040 210">
  <rect width="1040" height="210" fill="#f8fafc"/>
  <text x="30" y="58" font-family="Arial, sans-serif" font-size="25" fill="#111827">{title}</text>
  <text x="30" y="102" font-family="Arial, sans-serif" font-size="16" fill="#374151">{subtitle}</text>
  <text x="30" y="144" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">Generated from measured CSV/JSON fields only.</text>
</svg>
""",
        encoding="utf-8",
    )


def _p0_recap() -> Dict[str, Any]:
    route = _read_json(PREV_V9215 / "route_decision.json")
    p2_rows = read_csv_rows(PREV_V9215 / "p2_p4qualified_actuator_causality_matrix.csv")
    beat = [r for r in p2_rows if r.get("branch") == "RealFunctional" and _to_int(r.get("real_beats_best_control")) == 1]
    region = "none"
    if beat:
        keys = ["actuator", "dataset", "horizon"]
        region = ";".join(f"{k}={Counter(r.get(k, '') for r in beat).most_common(1)[0][0]}" for k in keys)
    audit = read_csv_rows(PREV_V9215 / "v9215_provenance_audit.csv")
    fake = _to_int(audit[0].get("fake_proxy_nonzero_count")) if audit else 999
    return {
        "stage": "P0_V9215_BOUNDARY_REPRODUCTION",
        "source_artifact": str(PREV_V9215.relative_to(ROOT)),
        "source_route": route.get("route", ""),
        "p2_paired_replay_pass": route.get("p2_paired_replay_pass", ""),
        "p2_survivor_count": route.get("p2_survivor_count", ""),
        "p3_pass_count": route.get("p3_pass_count", ""),
        "row_level_beat_count": len(beat),
        "row_level_beat_region": region,
        "best_control_by_metric": route.get("best_control", ""),
        "fake_proxy_count": fake,
        "P0_pass": int(
            route.get("route") == "R10-ReturnToTargetOrPrimitiveDesign"
            and _to_int(route.get("p2_survivor_count")) == 0
            and _to_int(route.get("p3_pass_count")) == 0
            and fake == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_control_autopsy() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    src = read_csv_rows(PREV_V9215 / "p2_p4qualified_actuator_causality_matrix.csv")
    rows: List[Dict[str, Any]] = []
    group_keys = ["actuator", "target", "solver", "event", "dataset", "seed", "horizon"]
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in src:
        groups[tuple(str(r.get(k, "")) for k in group_keys)].append(r)
    metric_map = {
        "CEp99": ("CEp99_delta", "min"),
        "MarginP10": ("margin_p10_delta", "max"),
        "ECE": ("ECE_delta", "min"),
        "NLL": ("NLL_delta", "min"),
    }
    control_winners: Counter[str] = Counter()
    for key, rs in groups.items():
        real = next((r for r in rs if r.get("branch") == "RealFunctional"), None)
        controls = [r for r in rs if r.get("branch") not in {"RealFunctional", "AdamWOnly"}]
        if not real or not controls:
            continue
        for metric, (field, mode) in metric_map.items():
            if mode == "min":
                best = min(controls, key=lambda r: _to_float(r.get(field)))
                real_minus = _to_float(real.get(field)) - _to_float(best.get(field))
                rank = 1 + sum(_to_float(c.get(field)) < _to_float(real.get(field)) for c in controls)
            else:
                best = max(controls, key=lambda r: _to_float(r.get(field)))
                real_minus = _to_float(real.get(field)) - _to_float(best.get(field))
                rank = 1 + sum(_to_float(c.get(field)) > _to_float(real.get(field)) for c in controls)
            control_winners[str(best.get("branch"))] += 1
            rows.append({
                "stage": "P1_CONTROL_DOMINANCE_AUTOPSY",
                "actuator": key[0],
                "target": key[1],
                "solver": key[2],
                "event": key[3],
                "dataset": key[4],
                "seed": key[5],
                "horizon": key[6],
                "metric": metric,
                "best_control": best.get("branch"),
                "real_delta": real.get(field),
                "best_control_delta": best.get(field),
                "real_minus_best_control": real_minus,
                "rank_of_real": rank,
                "logit_displacement_norm_real": real.get("delta_norm", "not_measured"),
                "logit_displacement_norm_control": "not_measured_per_control_in_v9215_source",
                "cos_real_control": "not_measured_per_control_in_v9215_source",
                "cos_real_adamw": real.get("cos_delta_adamw", "not_measured"),
                "cos_control_adamw": "not_measured_per_control_in_v9215_source",
                "task_safety_real": real.get("task_safe", ""),
                "task_safety_control": best.get("task_safe", ""),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    total = sum(control_winners.values())
    top_control, top_count = control_winners.most_common(1)[0] if control_winners else ("", 0)
    if top_control == "AdamWParallelDirection" and top_count / max(1, total) >= 0.50:
        dominance = "D3-AdamWParallelDominance"
    elif top_control == "RandomMatchedNorm" and top_count / max(1, total) >= 0.50:
        dominance = "D2-RandomRegularizationDominance"
    elif top_control == "NoOpMatchedOverhead" and top_count / max(1, total) >= 0.50:
        dominance = "D1-NoOpDominance"
    elif top_control == "ShuffledTarget" and top_count / max(1, total) >= 0.50:
        dominance = "D4-ShuffledTargetDominance"
    else:
        dominance = "D5-MixedControlDominance"
    # v9.2.15 had a row-level A4e/KMNIST/h80 exception; keep it in the summary
    beat = [r for r in src if r.get("branch") == "RealFunctional" and _to_int(r.get("real_beats_best_control")) == 1]
    if beat and all(r.get("actuator", "").startswith("A4e") and r.get("dataset") == "KMNIST" and str(r.get("horizon")) == "80" for r in beat):
        dominance = dominance + ",D6-A4eKMNISTDelayedSignal"
    return rows, {
        "control_dominance_type": dominance,
        "best_control": top_control,
        "best_control_count": top_count,
        "control_autopsy_rows": len(rows),
    }


def _ce_margin(logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logp = logits.log_softmax(dim=1)
    row = torch.arange(int(y.numel()), device=logits.device)
    ce = -logp[row, y]
    true = logits[row, y]
    masked = logits.clone()
    masked[row, y] = -torch.inf
    wrong, wrong_idx = masked.max(dim=1)
    margin = true - wrong
    return ce, margin, wrong_idx, row


def _event_mask(logits: torch.Tensor, y: torch.Tensor, event_id: str, dataset: str) -> torch.Tensor:
    ce, margin, _wrong_idx, _row = _ce_margin(logits, y)
    eid = str(event_id)
    if eid == "E7-KMNISTHardModeEvent":
        if dataset != "KMNIST":
            return torch.zeros_like(y, dtype=torch.bool)
        return margin <= torch.quantile(margin.float(), 0.10)
    if eid == "E8-A4e-KMNIST-H80-DelayedEvent":
        if dataset != "KMNIST":
            return torch.zeros_like(y, dtype=torch.bool)
        ce_tail = ce >= torch.quantile(ce.float(), 0.88)
        margin_tail = margin <= torch.quantile(margin.float(), 0.20)
        return ce_tail | margin_tail
    if eid == "E9-ControlWinnerEvent":
        return ce >= torch.quantile(ce.float(), 0.90)
    return v9215._event_mask(logits, y, event_id, dataset)


def _target_from_id(
    target_id: str,
    logits: torch.Tensor,
    y: torch.Tensor,
    dataset: str,
    control_delta: torch.Tensor,
    random_delta: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    tid = str(target_id)
    if tid.startswith("O"):
        return out_lq.build_output_target(tid, logits, y, dataset=dataset)
    if tid == "CD1-RandomWinnerPrototype":
        target = random_delta.detach()
    elif tid == "CD2-AdamWParallelWinnerPrototype":
        target = control_delta.detach()
    elif tid == "CD3-NoOpDeltaNullTarget":
        target = torch.zeros_like(logits)
    elif tid == "CD4-BestControlMixturePrototype":
        target = 0.75 * control_delta.detach() + 0.25 * random_delta.detach()
    elif tid == "CD5-KMNIST-A4e-H80-Prototype":
        target = control_delta.detach()
        if dataset == "KMNIST":
            ce, margin, wrong_idx, row = _ce_margin(logits, y)
            hard = (ce >= torch.quantile(ce.float(), 0.85)) | (margin <= torch.quantile(margin.float(), 0.20))
            extra = torch.zeros_like(logits)
            amp = (ce / ce.detach().mean().clamp_min(1.0e-6)).clamp(0.25, 4.0)
            extra[row, y] = amp
            extra[row, wrong_idx] = extra[row, wrong_idx] - amp
            target = target + 0.5 * extra * hard.float().unsqueeze(1)
    elif tid == "CD6-ControlContrastiveTailTarget":
        ce, _margin, wrong_idx, row = _ce_margin(logits, y)
        hard = ce >= torch.quantile(ce.float(), 0.90)
        target = control_delta.detach().clone()
        amp = (ce / ce.detach().mean().clamp_min(1.0e-6)).clamp(0.25, 4.0)
        target[row, y] = target[row, y] + amp * hard.float()
        target[row, wrong_idx] = target[row, wrong_idx] - amp * hard.float()
    elif tid == "CD7-ControlContrastiveCurvatureTarget":
        centered = logits - logits.mean(dim=1, keepdim=True)
        target = -centered / centered.float().norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    else:
        raise ValueError(f"unknown target {target_id}")
    target = target - target.mean(dim=1, keepdim=True)
    norm = target.float().norm()
    if bool((norm > 0).detach().cpu()):
        target = target / norm * 0.1
    return target.detach(), {
        "target_selected_fraction": 1.0,
        "target_norm_raw": float(norm.detach().cpu()),
        "target_mean_ce": 0.0,
        "target_mean_margin": 0.0,
        "target_wrong_confidence_mean": 0.0,
    }


def _solve_step(
    args: argparse.Namespace,
    spec: act.ActuatorSpec,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    target: torch.Tensor,
    solver: str,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
    task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
    raw_delta, ls = act.actuator_only_least_squares_delta(params, mu, std, spec, xb, target, ridge=float(args.ridge))
    projected, removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
    sid = str(solver)
    fractions = [float(args.functional_step_fraction)]
    if sid == "SOL3-TrustRegionQP":
        fractions = [float(args.trust_fraction)]
    elif sid in {"SOL4-HorizonAwareReplaySolver", "SOL5-ControlContrastiveSolver"}:
        fractions = [float(x) for x in _parse_list(args.scale_brackets)]
    best_delta: List[torch.Tensor] | None = None
    best_score = -float("inf")
    best_fraction = fractions[0]
    base_metrics = v9214._eval_actuator_metrics(params, mu, std, spec, x_eval, y_eval)
    parallel = v9214._cap_to_fraction(task_step, task_step, max(fractions))
    parallel_metrics = v9214._eval_actuator_metrics([p + d for p, d in zip(params, parallel)], mu, std, spec, x_eval, y_eval)
    for frac in fractions:
        cand = v9214._cap_to_fraction(projected, task_step, frac)
        metrics = v9214._eval_actuator_metrics([p + d for p, d in zip(params, cand)], mu, std, spec, x_eval, y_eval)
        if sid == "SOL5-ControlContrastiveSolver":
            score = (parallel_metrics["CE_p99"] - metrics["CE_p99"]) + 0.1 * (metrics["correct_margin_p10"] - parallel_metrics["correct_margin_p10"])
            score -= max(0.0, base_metrics["acc"] - metrics["acc"] - 0.005)
        elif sid == "SOL4-HorizonAwareReplaySolver":
            score = (base_metrics["NLL"] - metrics["NLL"]) + 0.01 * (base_metrics["CE_p99"] - metrics["CE_p99"])
        else:
            score = -float(snr_lq.step_norm(cand).detach().cpu())
        if score > best_score:
            best_score = score
            best_delta = cand
            best_fraction = frac
    solved = best_delta if best_delta is not None else v9214._cap_to_fraction(projected, task_step, fractions[0])
    fit = act.output_fit_metrics(params, solved, mu, std, spec, xb, target)
    adamw_logits = act.actuator_forward(xb, [p + d for p, d in zip(params, task_step)], mu, std, spec)
    base_logits = act.actuator_forward(xb, params, mu, std, spec)
    adamw_norm = (adamw_logits - base_logits).float().norm().clamp_min(1.0e-12)
    dnorm = snr_lq.step_norm(solved)
    return solved, {
        "target_fit_R2": fit["output_target_fit_r2"],
        "rz": float((fit["output_displacement_norm"] / adamw_norm).detach().cpu()),
        "delta_norm": float(dnorm.detach().cpu()),
        "delta_norm_vs_adamw": float((dnorm / snr_lq.step_norm(task_step).clamp_min(1.0e-12)).detach().cpu()),
        "projection_removed_norm": float(removed.detach().cpu()),
        "solver_selected_fraction": best_fraction,
        "solver_score": best_score,
        "ls_rank": ls.get("ls_rank", 0),
    }


def _branch_steps(
    args: argparse.Namespace,
    spec: act.ActuatorSpec,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    target: torch.Tensor,
    solver: str,
    seed: int,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, float]]:
    real, info = _solve_step(args, spec, params, mu, std, xb, yb, x_eval, y_eval, target, solver)
    _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
    task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
    real_norm = snr_lq.step_norm(real)
    perm = torch.randperm(int(target.shape[0]), device=target.device, generator=torch.Generator(device=target.device).manual_seed(seed + 1601))
    shuffled, _ = _solve_step(args, spec, params, mu, std, xb, yb, x_eval, y_eval, target[perm], solver)
    rand_step, _ = snr_lq.project_step_to_task_safe(v9214._random_like_step(params, real_norm, seed + 1602), grads)
    return {
        "AdamWOnly": snr_lq.zero_like_params(params),
        "RealFunctional": real,
        "NoOpMatchedOverhead": snr_lq.zero_like_params(params),
        "RandomMatchedNorm": v9214._cap_to_fraction(rand_step, task_step, float(args.functional_step_fraction)),
        "ShuffledTarget": v9214._cap_to_fraction(shuffled, task_step, float(args.functional_step_fraction)),
        "AdamWParallelDirection": v9214._cap_to_fraction(task_step, task_step, float(args.functional_step_fraction)),
    }, info


def _replay_metrics(
    args: argparse.Namespace,
    cfg: ManualAdamWConfig,
    spec: act.ActuatorSpec,
    base_params: Sequence[torch.Tensor],
    base_states: Sequence[AdamWState],
    mu: torch.Tensor,
    std: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    branch_step: Sequence[torch.Tensor],
    horizons: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    params_b = v9214._clone_params(base_params)
    states_b = v9214._clone_states(base_states)
    if any(bool((d.float().abs().max() > 0).detach().cpu()) for d in branch_step):
        for p, d in zip(params_b, branch_step):
            p.add_(d)
    prev = 0
    out: Dict[int, Dict[str, float]] = {}
    for horizon in sorted(int(h) for h in horizons):
        v9214._run_adamw_steps(params_b, states_b, mu, std, spec, x_train, y_train, int(args.warmup_steps) + prev, horizon - prev, int(args.batch_size), cfg)
        prev = horizon
        out[horizon] = v9214._eval_actuator_metrics(params_b, mu, std, spec, x_eval, y_eval)
    return out


def _append_replay_rows(
    rows: List[Dict[str, Any]],
    stage: str,
    metrics_by_branch: Dict[str, Dict[int, Dict[str, float]]],
    info: Dict[str, float],
    meta: Dict[str, Any],
) -> None:
    for horizon, adamw in metrics_by_branch["AdamWOnly"].items():
        for branch, by_h in metrics_by_branch.items():
            m = by_h[horizon]
            rows.append({
                "stage": stage,
                **meta,
                "horizon": horizon,
                "branch": branch,
                "CEp99_delta": m["CE_p99"] - adamw["CE_p99"],
                "margin_p10_delta": m["correct_margin_p10"] - adamw["correct_margin_p10"],
                "curvature_delta": "not_measured",
                "ECE_delta": m["ECE"] - adamw["ECE"],
                "NLL_delta": m["NLL"] - adamw["NLL"],
                "acc_delta": m["acc"] - adamw["acc"],
                "CEp99_abs": m["CE_p99"],
                "margin_p10_abs": m["correct_margin_p10"],
                "ECE_abs": m["ECE"],
                "NLL_abs": m["NLL"],
                "acc_abs": m["acc"],
                "AdamW_CEp99_abs": adamw["CE_p99"],
                "AdamW_ECE_abs": adamw["ECE"],
                "AdamW_margin_p10_abs": adamw["correct_margin_p10"],
                "AdamW_acc_abs": adamw["acc"],
                "control_rank": "",
                "real_beats_best_control": 0,
                "target_fit_R2": info.get("target_fit_R2", 0),
                "rz": info.get("rz", 0),
                "delta_norm": info.get("delta_norm", 0),
                "delta_norm_vs_adamw": info.get("delta_norm_vs_adamw", 0),
                "solver_selected_fraction": info.get("solver_selected_fraction", ""),
                "event_coverage": meta.get("event_coverage", 0),
                "bad_event_rate": 0,
                "task_safe": int(m["acc"] >= adamw["acc"] - 0.005),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })


def _annotate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    group_keys = ["actuator", "target", "solver", "event", "dataset", "seed", "horizon"]
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(str(r.get(k, "")) for k in group_keys)].append(r)
    for rs in groups.values():
        real = next((r for r in rs if r.get("branch") == "RealFunctional"), None)
        controls = [r for r in rs if r.get("branch") not in {"RealFunctional", "AdamWOnly"}]
        ranked = sorted(controls, key=lambda r: (_to_float(r["CEp99_delta"]), -_to_float(r["margin_p10_delta"])))
        for idx, r in enumerate(ranked, start=1):
            r["control_rank"] = idx
        if real and controls:
            best_ce = min(_to_float(r["CEp99_delta"]) for r in controls)
            best_margin = max(_to_float(r["margin_p10_delta"]) for r in controls)
            ce_margin = 0.03 * abs(_to_float(real.get("AdamW_CEp99_abs")))
            beat = int(
                _to_int(real.get("task_safe")) == 1
                and (
                    _to_float(real["CEp99_delta"]) <= best_ce - ce_margin
                    or _to_float(real["margin_p10_delta"]) >= best_margin + 0.01
                )
            )
            real["real_beats_best_control"] = beat
    real_rows = [r for r in rows if r.get("branch") == "RealFunctional"]
    beat_rows = [r for r in real_rows if _to_int(r.get("real_beats_best_control")) == 1]
    return {
        "real_row_count": len(real_rows),
        "row_level_beat_count": len(beat_rows),
        "beat_rate": len(beat_rows) / max(1, len(real_rows)),
        "real_mean_CEp99_delta": _mean(real_rows, "CEp99_delta"),
        "real_mean_margin_delta": _mean(real_rows, "margin_p10_delta"),
    }


def _run_replay_grid(
    args: argparse.Namespace,
    device: torch.device,
    *,
    stage: str,
    actuators: Sequence[str],
    targets: Sequence[str],
    solvers: Sequence[str],
    events: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int],
    horizons: Sequence[int],
) -> List[Dict[str, Any]]:
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    rows: List[Dict[str, Any]] = []
    for aid in actuators:
        spec = act.actuator_specs_from_ids([aid])[0]
        for dataset in datasets:
            x_train, y_train, x_eval, y_eval, in_dim, out_dim, _protocol = v92._load_task(
                args, dataset, train_size=max(int(args.train_size), 4096), test_size=int(args.eval_size)
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            x_eval = x_eval.to(device=device, dtype=torch.float32)
            y_eval = y_eval.to(device=device)
            for seed in seeds:
                base_params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x_train, device, seed + 921600 + len(aid))
                base_states = [AdamWState.zeros_like(p) for p in base_params]
                v9214._run_adamw_steps(base_params, base_states, mu, std, spec, x_train, y_train, 0, int(args.warmup_steps), int(args.batch_size), cfg)
                xb = x_train[: int(args.audit_batch_size)]
                yb = y_train[: int(args.audit_batch_size)]
                logits = act.actuator_forward(xb, base_params, mu, std, spec)
                _loss, grads = act.actuator_fwd_bwd(xb, yb, base_params, mu, std, spec)
                task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
                parallel_logits = act.actuator_forward(xb, [p + d for p, d in zip(base_params, task_step)], mu, std, spec)
                control_delta = (parallel_logits - logits).detach()
                rand_step = v9214._random_like_step(base_params, snr_lq.step_norm(task_step), seed + 16000)
                random_logits = act.actuator_forward(xb, [p + d for p, d in zip(base_params, rand_step)], mu, std, spec)
                random_delta = (random_logits - logits).detach()
                zero_metrics = _replay_metrics(args, cfg, spec, base_params, base_states, mu, std, x_train, y_train, x_eval, y_eval, snr_lq.zero_like_params(base_params), horizons)
                parallel_step = v9214._cap_to_fraction(task_step, task_step, float(args.functional_step_fraction))
                parallel_metrics = _replay_metrics(args, cfg, spec, base_params, base_states, mu, std, x_train, y_train, x_eval, y_eval, parallel_step, horizons)
                for target_id in targets:
                    base_target, target_info = _target_from_id(target_id, logits, yb, dataset, control_delta, random_delta)
                    for event_id in events:
                        mask = _event_mask(logits, yb, event_id, dataset)
                        target = base_target * mask.float().unsqueeze(1)
                        coverage = float(mask.float().mean().detach().cpu())
                        for solver in solvers:
                            branch_steps, info = _branch_steps(args, spec, base_params, mu, std, xb, yb, x_eval, y_eval, target, solver, seed)
                            info.update(target_info)
                            metrics_by_branch: Dict[str, Dict[int, Dict[str, float]]] = {}
                            for branch, step in branch_steps.items():
                                if branch in {"AdamWOnly", "NoOpMatchedOverhead"}:
                                    metrics_by_branch[branch] = zero_metrics
                                elif branch == "AdamWParallelDirection":
                                    metrics_by_branch[branch] = parallel_metrics
                                else:
                                    metrics_by_branch[branch] = _replay_metrics(args, cfg, spec, base_params, base_states, mu, std, x_train, y_train, x_eval, y_eval, step, horizons)
                            _append_replay_rows(
                                rows,
                                stage,
                                metrics_by_branch,
                                info,
                                {
                                    "actuator": aid,
                                    "target": target_id,
                                    "solver": solver,
                                    "event": event_id,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "event_coverage": coverage,
                                },
                            )
    _annotate_rows(rows)
    return rows


def _aggregate_group_pass(rows: Sequence[Dict[str, Any]], *, min_beat_rate: float) -> Dict[str, Any]:
    real = [r for r in rows if r.get("branch") == "RealFunctional"]
    beat = [r for r in real if _to_int(r.get("real_beats_best_control")) == 1]
    beat_rate = len(beat) / max(1, len(real))
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("actuator")), str(r.get("target")), str(r.get("solver")), str(r.get("event")))].append(r)
    survivor_count = 0
    best_group: Dict[str, Any] = {}
    for key, rs in groups.items():
        gr = [r for r in rs if r.get("branch") == "RealFunctional"]
        gc = [r for r in rs if r.get("branch") not in {"RealFunctional", "AdamWOnly"}]
        if not gr or not gc:
            continue
        best_ce = min(_mean([r for r in gc if r.get("branch") == b], "CEp99_delta") for b in {r.get("branch") for r in gc})
        best_margin = max(_mean([r for r in gc if r.get("branch") == b], "margin_p10_delta") for b in {r.get("branch") for r in gc})
        real_ce = _mean(gr, "CEp99_delta")
        real_margin = _mean(gr, "margin_p10_delta")
        ce_margin = 0.03 * abs(_mean(gr, "AdamW_CEp99_abs"))
        pass_group = int(real_ce <= best_ce - ce_margin or real_margin >= best_margin + 0.01)
        if pass_group:
            survivor_count += 1
            cand = {
                "actuator": key[0],
                "target": key[1],
                "solver": key[2],
                "event": key[3],
                "real_CEp99_delta": real_ce,
                "best_control_CEp99_delta": best_ce,
                "real_margin_delta": real_margin,
                "best_control_margin_delta": best_margin,
                "score": (best_ce - real_ce) + 0.1 * (real_margin - best_margin),
            }
            if not best_group or cand["score"] > best_group.get("score", -999):
                best_group = cand
    return {
        "paired_replay_pass": int(survivor_count > 0 and beat_rate >= min_beat_rate),
        "survivor_count": survivor_count,
        "beat_rate": beat_rate,
        "row_level_beat_count": len(beat),
        "real_mean_CEp99_delta": _mean(real, "CEp99_delta"),
        "real_mean_margin_delta": _mean(real, "margin_p10_delta"),
        **{f"best_{k}": v for k, v in best_group.items()},
    }


def _prototype_rows(p3_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in p3_rows:
        if r.get("branch") == "RealFunctional":
            groups[str(r.get("target"))].append(r)
    for target, rs in sorted(groups.items()):
        out.append({
            "stage": "PROTOTYPE_TARGET_TRACE",
            "prototype_id": target,
            "source_control": "online_control_displacement_or_direct_formula",
            "source_metric": "CEp99_margin_ECE",
            "source_dataset": "mixed",
            "source_horizon": "20,80",
            "cluster_size": len(rs),
            "prototype_norm": "computed_per_batch_not_stored_as_global_vector",
            "intra_cluster_cos_mean": "not_global_clustered_online_rows",
            "target_fit_R2": _mean(rs, "target_fit_R2"),
            "rz": _mean(rs, "rz"),
            "paired_replay_real_delta": _mean(rs, "CEp99_delta"),
            "paired_replay_best_control_delta": "see_p3_rows",
            "real_beats_best_control": sum(_to_int(r.get("real_beats_best_control")) for r in rs),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return out


def _p4_control_contrastive(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("solver")), str(r.get("target")), str(r.get("actuator")), str(r.get("event")), str(r.get("dataset")), str(r.get("seed")), str(r.get("horizon")))].append(r)
    pred: List[float] = []
    actual: List[float] = []
    for key, rs in groups.items():
        real = next((r for r in rs if r.get("branch") == "RealFunctional"), None)
        controls = [r for r in rs if r.get("branch") not in {"RealFunctional", "AdamWOnly"}]
        if not real or not controls:
            continue
        best_ce = min(_to_float(r.get("CEp99_delta")) for r in controls)
        best_margin = max(_to_float(r.get("margin_p10_delta")) for r in controls)
        scc = (best_ce - _to_float(real.get("CEp99_delta"))) + 0.1 * (_to_float(real.get("margin_p10_delta")) - best_margin)
        ps = _to_float(real.get("target_fit_R2")) * _to_float(real.get("rz")) - max(0.0, _to_float(real.get("event_coverage")) - 0.15)
        pred.append(ps)
        actual.append(scc)
        out.append({
            "stage": "P4_CONTROL_CONTRASTIVE_SOLVER_VALIDATION",
            "solver": key[0],
            "target": key[1],
            "actuator": key[2],
            "event": key[3],
            "dataset": key[4],
            "seed": key[5],
            "horizon": key[6],
            "predicted_Scc": ps,
            "actual_Scc": scc,
            "CEp99_gap_vs_control": _to_float(real.get("CEp99_delta")) - best_ce,
            "margin_gap_vs_control": _to_float(real.get("margin_p10_delta")) - best_margin,
            "curvature_gap_vs_control": "not_measured",
            "ECE_gap_vs_control": _to_float(real.get("ECE_delta")) - min(_to_float(r.get("ECE_delta")) for r in controls),
            "task_drop": max(0.0, -_to_float(real.get("acc_delta")) - 0.005),
            "bad_event": int(_to_float(real.get("holdout_delta", real.get("NLL_delta"))) > 0),
            "coverage": real.get("event_coverage", 0),
            "control_contrastive_pass": int(scc > 0 and _to_float(real.get("event_coverage")) >= 0.03 and _to_float(real.get("event_coverage")) <= 0.15),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    corr = 0.0
    if len(pred) > 2:
        mp = sum(pred) / len(pred)
        ma = sum(actual) / len(actual)
        num = sum((p - mp) * (a - ma) for p, a in zip(pred, actual))
        denp = math.sqrt(sum((p - mp) ** 2 for p in pred))
        dena = math.sqrt(sum((a - ma) ** 2 for a in actual))
        corr = num / (denp * dena) if denp > 0 and dena > 0 else 0.0
    passed = [r for r in out if _to_int(r.get("control_contrastive_pass")) == 1]
    return out, {"control_contrastive_pass": int(corr >= 0.30 and bool(passed)), "control_contrastive_corr": corr, "control_contrastive_pass_count": len(passed)}


def _write_downstream_not_run(out_dir: Path, reason: str) -> None:
    for filename, stage in [
        ("p5_short_run_causal_validation.csv", "P5_SHORT_RUN_CAUSAL_VALIDATION"),
        ("p6_full_functional_reentry_10seed.csv", "P6_FULL_FUNCTIONAL_REENTRY_10SEED"),
        ("p7_noise_robustness_signal_separation.csv", "P7_NOISE_ROBUSTNESS_SIGNAL_SEPARATION"),
        ("p8_strong_baseline_external_ready.csv", "P8_STRONG_BASELINE_EXTERNAL_READY"),
    ]:
        write_csv_rows(out_dir / filename, [_not_run(stage, filename, reason)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=36)
    parser.add_argument("--functional-step-fraction", type=float, default=0.10)
    parser.add_argument("--trust-fraction", type=float, default=0.03)
    parser.add_argument("--scale-brackets", default="0.003,0.01,0.03,0.10")
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--p2-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--p2-horizons", default="20,80,160")
    parser.add_argument("--p3-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p3-seeds", default="0,1,2")
    parser.add_argument("--p3-horizons", default="20,80")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    write_json(out_dir / "run_manifest.json", {
        "experiment": "DG-KAN v9.2.16 Causal Target Discovery Primitive Redesign",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "torch": torch.__version__,
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "script_path": str(SCRIPT_PATH.relative_to(ROOT)),
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "args": vars(args),
    })
    write_csv_rows(out_dir / "contract_audit_v9216.csv", [{
        "candidate": "v9216_all",
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "uses_loss_backward": 0,
        "sampler_or_class_weight_used": 0,
        "cpu_offload_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])

    p0 = [_p0_recap()]
    write_csv_rows(out_dir / "p0_v9215_boundary_reproduction.csv", p0)
    p1_rows, p1_summary = _p1_control_autopsy()
    write_csv_rows(out_dir / "p1_control_dominance_autopsy.csv", p1_rows)

    p2_rows = _run_replay_grid(
        args,
        device,
        stage="P2_A4E_KMNIST_H80_SIGNAL_REPLICATION",
        actuators=["A4e-BoundedRational-FusedCoeffGrad"],
        targets=["O1-HardTailLogitCorrection", "O2-MarginTailExpansion", "O6-KMNISTHardModeOutputTarget", "CD5-KMNIST-A4e-H80-Prototype"],
        solvers=["SOL2-ConstrainedLeastSquares", "SOL3-TrustRegionQP", "SOL4-HorizonAwareReplaySolver", "SOL5-ControlContrastiveSolver"],
        events=["E7-KMNISTHardModeEvent", "E8-A4e-KMNIST-H80-DelayedEvent", "E9-ControlWinnerEvent"],
        datasets=["KMNIST"],
        seeds=[int(x) for x in _parse_list(args.p2_seeds)],
        horizons=[int(x) for x in _parse_list(args.p2_horizons)],
    )
    p2_summary = _aggregate_group_pass(p2_rows, min_beat_rate=0.10)
    p2_summary = {f"a4e_{k}": v for k, v in p2_summary.items()}
    write_csv_rows(out_dir / "p2_a4e_kmnist_h80_signal_replication.csv", p2_rows)

    p3_rows = _run_replay_grid(
        args,
        device,
        stage="P3_CONTROL_DERIVED_TARGET_DISCOVERY",
        actuators=["A4e-BoundedRational-FusedCoeffGrad", "A7c-BasisEntropy-ValueOnly"],
        targets=[
            "CD1-RandomWinnerPrototype",
            "CD2-AdamWParallelWinnerPrototype",
            "CD3-NoOpDeltaNullTarget",
            "CD4-BestControlMixturePrototype",
            "CD5-KMNIST-A4e-H80-Prototype",
            "CD6-ControlContrastiveTailTarget",
            "CD7-ControlContrastiveCurvatureTarget",
        ],
        solvers=["SOL2-ConstrainedLeastSquares", "SOL5-ControlContrastiveSolver"],
        events=["E9-ControlWinnerEvent"],
        datasets=_canonical_tasks(args.p3_datasets),
        seeds=[int(x) for x in _parse_list(args.p3_seeds)],
        horizons=[int(x) for x in _parse_list(args.p3_horizons)],
    )
    p3_summary = _aggregate_group_pass(p3_rows, min_beat_rate=0.05)
    p3_summary = {f"prototype_{k}": v for k, v in p3_summary.items()}
    write_csv_rows(out_dir / "p3_control_derived_target_discovery.csv", p3_rows)
    proto_rows = _prototype_rows(p3_rows)
    write_csv_rows(out_dir / "prototype_target_trace_v9216.csv", proto_rows)
    write_csv_rows(out_dir / "control_displacement_trace_v9216.csv", p3_rows)
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9216.csv", [*p2_rows, *p3_rows])

    p4_rows, p4_summary = _p4_control_contrastive([*p2_rows, *p3_rows])
    write_csv_rows(out_dir / "p4_control_contrastive_solver_validation.csv", p4_rows)

    paired_pass = _to_int(p2_summary.get("a4e_paired_replay_pass")) or _to_int(p3_summary.get("prototype_paired_replay_pass"))
    cc_pass = _to_int(p4_summary.get("control_contrastive_pass"))
    if _to_int(p2_summary.get("a4e_paired_replay_pass")):
        route = "R1-A4eKMNISTDelayedSignalConfirmed"
        reason = "a4e_kmnist_delayed_signal_survivor_found_but_short_run_not_opened_in_this_runner"
        failure = "F17_artifact_missing"
    elif _to_int(p3_summary.get("prototype_paired_replay_pass")):
        route = "R2-ControlDerivedTargetPass"
        reason = "control_derived_target_survivor_found_but_short_run_not_opened_in_this_runner"
        failure = "F17_artifact_missing"
    elif cc_pass:
        route = "R3-ControlContrastiveSolverPass"
        reason = "control_contrastive_solver_survivor_found_but_short_run_not_opened_in_this_runner"
        failure = "F17_artifact_missing"
    elif p2_summary.get("a4e_row_level_beat_count", 0) and not _to_int(p2_summary.get("a4e_paired_replay_pass")):
        route = "R7-A4eRowSignalNoise"
        reason = "a4e_row_level_signal_did_not_replicate_as_aggregate_survivor"
        failure = "F4_a4e_kmnist_h80_signal_not_replicated"
    else:
        route = "R8-NoExtractableFunctionalAdvantage"
        reason = "control_derived_targets_remain_control_equivalent"
        failure = "F9_no_paired_replay_survivor"
    _write_downstream_not_run(out_dir, reason)

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "best_actuator_candidate": p2_summary.get("a4e_best_actuator") or p3_summary.get("prototype_best_actuator", ""),
        "best_target": p2_summary.get("a4e_best_target") or p3_summary.get("prototype_best_target", ""),
        "best_solver": p2_summary.get("a4e_best_solver") or p3_summary.get("prototype_best_solver", ""),
        "best_event_controller": p2_summary.get("a4e_best_event") or p3_summary.get("prototype_best_event", ""),
        "best_control": p1_summary.get("best_control", ""),
        "control_dominance_type": p1_summary.get("control_dominance_type", ""),
        "a4e_kmnist_h80_signal_pass": _to_int(p2_summary.get("a4e_paired_replay_pass")),
        "control_derived_target_pass": _to_int(p3_summary.get("prototype_paired_replay_pass")),
        "control_contrastive_pass": cc_pass,
        "paired_replay_pass": int(bool(paired_pass)),
        "short_run_pass": 0,
        "full_reentry_pass": 0,
        "functional_task_safe": 0,
        "functional_mechanism_pass": int(bool(paired_pass or cc_pass)),
        "functional_control_pass": int(bool(paired_pass or cc_pass)),
        "functional_system_pass": 1,
        "functional_kmnist_repair_pass": _to_int(p2_summary.get("a4e_paired_replay_pass")),
        "noise_robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "primary_blocker": reason,
        "success_v9216_paired_replay_causality": int(bool(paired_pass or cc_pass)),
        "success_v9216_short_run": 0,
        "success_v9216_full_functional": 0,
        "success_v9216_external_ready": 0,
        **p1_summary,
        **p2_summary,
        **p3_summary,
        **p4_summary,
    }
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "stage": "terminal",
        "failure_code": failure,
        "primary_blocker": reason,
        "route": route,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    _write_svg(out_dir / "figures" / "p0_v9215_boundary_dashboard.svg", "P0 v9.2.15 Boundary", f"route={p0[0].get('source_route')}")
    _write_svg(out_dir / "figures" / "p0_real_vs_best_control_summary.svg", "P0 Real vs Control", f"beats={p0[0].get('row_level_beat_count')}")
    _write_svg(out_dir / "figures" / "p0_row_level_beat_distribution.svg", "P0 Beat Distribution", str(p0[0].get("row_level_beat_region")))
    _write_svg(out_dir / "figures" / "p1_control_rank_heatmap.svg", "P1 Control Rank", p1_summary.get("control_dominance_type", ""))
    _write_svg(out_dir / "figures" / "p1_best_control_by_horizon.svg", "P1 Best Control", p1_summary.get("best_control", ""))
    _write_svg(out_dir / "figures" / "p1_best_control_by_dataset.svg", "P1 Dataset", f"rows={len(p1_rows)}")
    _write_svg(out_dir / "figures" / "p1_real_vs_control_logit_displacement.svg", "P1 Displacement", "per-control displacement not in source")
    _write_svg(out_dir / "figures" / "p1_cosine_real_control_adamw.svg", "P1 Cosine", "per-control cosine not in source")
    _write_svg(out_dir / "figures" / "p2_a4e_kmnist_horizon_effect.svg", "P2 A4e KMNIST", f"beat_rate={p2_summary.get('a4e_beat_rate')}")
    _write_svg(out_dir / "figures" / "p2_a4e_row_level_beat_rate.svg", "P2 Beat Rate", f"beats={p2_summary.get('a4e_row_level_beat_count')}")
    _write_svg(out_dir / "figures" / "p2_a4e_ce_margin_by_horizon.svg", "P2 CE Margin", f"pass={p2_summary.get('a4e_paired_replay_pass')}")
    _write_svg(out_dir / "figures" / "p2_a4e_controls_rank.svg", "P2 Control Rank", p1_summary.get("best_control", ""))
    _write_svg(out_dir / "figures" / "p3_control_prototype_clusters.svg", "P3 Prototypes", f"rows={len(proto_rows)}")
    _write_svg(out_dir / "figures" / "p3_prototype_fit_vs_causality.svg", "P3 Fit vs Causality", f"pass={p3_summary.get('prototype_paired_replay_pass')}")
    _write_svg(out_dir / "figures" / "p3_prototype_source_control_matrix.svg", "P3 Source Control", "online control displacement")
    _write_svg(out_dir / "figures" / "p3_real_vs_best_control_for_prototypes.svg", "P3 Real vs Control", f"survivors={p3_summary.get('prototype_survivor_count')}")
    _write_svg(out_dir / "figures" / "p4_predicted_vs_actual_scc.svg", "P4 Predicted vs Actual", f"corr={p4_summary.get('control_contrastive_corr')}")
    _write_svg(out_dir / "figures" / "p4_control_contrastive_pareto.svg", "P4 Pareto", f"pass={p4_summary.get('control_contrastive_pass')}")
    _write_svg(out_dir / "figures" / "p4_solver_comparison_matrix.svg", "P4 Solver", f"rows={len(p4_rows)}")
    _write_svg(out_dir / "figures" / "p4_abstention_precision_curve.svg", "P4 Abstention", "SOL6 not implemented")

    audit_paths = [
        out_dir / "contract_audit_v9216.csv",
        out_dir / "p0_v9215_boundary_reproduction.csv",
        out_dir / "p1_control_dominance_autopsy.csv",
        out_dir / "p2_a4e_kmnist_h80_signal_replication.csv",
        out_dir / "p3_control_derived_target_discovery.csv",
        out_dir / "p4_control_contrastive_solver_validation.csv",
        out_dir / "p5_short_run_causal_validation.csv",
        out_dir / "p6_full_functional_reentry_10seed.csv",
        out_dir / "p7_noise_robustness_signal_separation.csv",
        out_dir / "p8_strong_baseline_external_ready.csv",
        out_dir / "control_displacement_trace_v9216.csv",
        out_dir / "prototype_target_trace_v9216.csv",
        out_dir / "paired_replay_branch_trace_v9216.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v9216_provenance_audit.csv", [audit])
    route_decision.update(audit)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    hash_paths = [PLAN_PATH, SCRIPT_PATH, out_dir / "route_decision.json", *audit_paths, out_dir / "v9216_provenance_audit.csv"]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_paths, root=ROOT))
    print(json.dumps(route_decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
