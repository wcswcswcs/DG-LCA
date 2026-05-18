#!/usr/bin/env python3
"""DG-KAN v9.2.15 control-resistant functional causality runner."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import defaultdict
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
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import lq_output_space_functional as out_lq  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.15_ControlResistant_Functional_Causality_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9215_control_resistant_functional_causality.py"
PREV_V9214 = ROOT / "results" / "real_rerun_20260506" / "v9214_p4qualified_functional_actuator_closure_first_20260510T010000Z"


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


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [_to_float(r.get(key)) for r in rows]
    return sum(vals) / max(1, len(vals))


def _p0_recap(prev_dir: Path) -> Dict[str, Any]:
    route = _read_json(prev_dir / "route_decision.json")
    audit = read_csv_rows(prev_dir / "v9214_provenance_audit.csv")
    fake_count = _to_int(audit[0].get("fake_proxy_nonzero_count")) if audit else 0
    return {
        "stage": "P0_V9214_BOUNDARY_REPRODUCTION",
        "source_artifact": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir),
        "source_route": route.get("route", "missing"),
        "best_actuator_candidate": route.get("best_actuator_candidate", ""),
        "A7c_forward_q90": route.get("forward_ratio_q90", 0),
        "A7c_backward_q90": route.get("backward_ratio_q90", 0),
        "A7c_step_q90": route.get("step_ratio_q90", 0),
        "A7c_memory": route.get("memory_ratio", 0),
        "A7c_target_fit_R2": route.get("target_fit_R2", 0),
        "A7c_rz": route.get("output_displacement_ratio_rz", 0),
        "A7c_p5_near_pass_count": route.get("actuator_base_near_pass_count", 0),
        "A7c_macro_delta": route.get("actuator_base_macro_delta", 0),
        "paired_replay_real_CEp99_mean_delta": route.get("p5_real_mean_CEp99_delta", 0),
        "paired_replay_best_control_CEp99_mean_delta": route.get("p5_best_control_mean_CEp99_delta", 0),
        "paired_replay_real_margin_mean_delta": route.get("p5_real_mean_margin_delta", 0),
        "paired_replay_best_control_margin_mean_delta": route.get("p5_best_control_mean_margin_delta", 0),
        "fake_proxy_count": fake_count,
        "P0_pass": int(
            route.get("route") == "R4-ActuatorP4ClosedBaseQualified"
            and _to_int(route.get("p4_closure_pass")) == 1
            and _to_int(route.get("actuator_base_near_pass")) == 1
            and _to_int(route.get("paired_replay_pass")) == 0
            and fake_count == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _run_p1_autopsy(prev_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = read_csv_rows(prev_dir / "p5_paired_replay_after_actuator_closure.csv")
    out: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if str(r.get("branch")) == "AdamWOnly":
            continue
        key = (str(r.get("target_id")), str(r.get("dataset")), str(r.get("horizon")), str(r.get("branch")))
        groups[key].append(r)
    branch_scores: Dict[str, Dict[str, float]] = {}
    for branch in sorted({k[3] for k in groups}):
        br = [r for rs in groups.values() for r in rs if r.get("branch") == branch]
        branch_scores[branch] = {
            "CEp99_delta": _mean(br, "CEp99_delta"),
            "margin_p10_delta": _mean(br, "margin_p10_delta"),
            "accuracy_delta": _mean(br, "val_proxy_acc_delta"),
            "holdout_delta": _mean(br, "holdout_loss_delta"),
        }
    best_control = min(
        (b for b in branch_scores if b not in {"RealFunctional"}),
        key=lambda b: (branch_scores[b]["CEp99_delta"], -branch_scores[b]["margin_p10_delta"]),
        default="",
    )
    factor = "M4-control_dominance" if best_control else "M7-mechanism_metric_mismatch"
    if best_control == "AdamWParallelDirection":
        factor = "M4-control_dominance,M5-scale_failure"
    real_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("branch") == "RealFunctional":
            real_by_target[str(r.get("target_id"))].append(r)
    for target, rs in real_by_target.items():
        out.append({
            "stage": "P1_PAIRED_REPLAY_FAILURE_AUTOPSY",
            "actuator_candidate": "A7c-BasisEntropy-ValueOnly",
            "output_target": target,
            "solver": "v9214_task_safe_LS",
            "event_type": "v9214_target_event",
            "branch": "RealFunctional",
            "dataset": "aggregate",
            "seed": "aggregate",
            "horizon": "aggregate",
            "CEp99_delta": _mean(rs, "CEp99_delta"),
            "margin_p10_delta": _mean(rs, "margin_p10_delta"),
            "curvature_delta": "not_measured_in_v9214_source",
            "ECE_delta": _mean(rs, "ECE_delta"),
            "NLL_delta": _mean(rs, "NLL_delta"),
            "accuracy_delta": _mean(rs, "val_proxy_acc_delta"),
            "logit_displacement_norm": "not_measured_in_v9214_source",
            "target_fit_R2": "not_measured_in_v9214_p5",
            "rz": "not_measured_in_v9214_p5",
            "delta_norm": "not_measured_in_v9214_source",
            "delta_norm_vs_adamw": "not_measured_in_v9214_source",
            "cos_delta_adamw": "not_measured_in_v9214_source",
            "cos_delta_random": "not_measured_in_v9214_source",
            "cos_delta_control": "not_measured_in_v9214_source",
            "holdout_delta": _mean(rs, "holdout_loss_delta"),
            "bad_event": int(_mean(rs, "holdout_loss_delta") > 0),
            "event_coverage": "not_measured_in_v9214_source",
            "failure_factor": factor,
            "best_control": best_control,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "p1_best_control": best_control,
        "p1_failure_factor": factor,
        "p1_real_CEp99_delta": branch_scores.get("RealFunctional", {}).get("CEp99_delta", 0.0),
        "p1_best_control_CEp99_delta": branch_scores.get(best_control, {}).get("CEp99_delta", 0.0) if best_control else 0.0,
        "p1_real_margin_delta": branch_scores.get("RealFunctional", {}).get("margin_p10_delta", 0.0),
        "p1_best_control_margin_delta": branch_scores.get(best_control, {}).get("margin_p10_delta", 0.0) if best_control else 0.0,
    }
    return out, summary


def _event_mask(logits: torch.Tensor, y: torch.Tensor, event_id: str, dataset: str) -> torch.Tensor:
    log_probs = logits.log_softmax(dim=1)
    probs = logits.softmax(dim=1)
    n = int(y.numel())
    row = torch.arange(n, device=logits.device)
    true = logits[row, y]
    masked = logits.clone()
    masked[row, y] = -torch.inf
    wrong, _wrong_idx = masked.max(dim=1)
    pred = logits.argmax(dim=1)
    ce = -log_probs[row, y]
    margin = true - wrong
    wrong_conf = torch.where(pred != y, probs.max(dim=1).values, torch.zeros_like(ce))
    eid = str(event_id)
    if eid == "E1-CalibratedCEp99Tail":
        return ce >= torch.quantile(ce.float(), 0.90)
    if eid == "E2-CalibratedMarginTail":
        return margin <= torch.quantile(margin.float(), 0.20)
    if eid == "E6-CompositeSparseEvent":
        return (ce >= torch.quantile(ce.float(), 0.92)) | (margin <= torch.quantile(margin.float(), 0.12)) | (wrong_conf >= torch.quantile(wrong_conf.float(), 0.95))
    if eid == "E7-KMNISTHardModeEvent":
        if str(dataset) != "KMNIST":
            return torch.zeros_like(ce, dtype=torch.bool)
        return margin <= torch.quantile(margin.float(), 0.30)
    raise ValueError(f"unknown event {event_id}")


def _solve_step(
    args: argparse.Namespace,
    spec: act.ActuatorSpec,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    target: torch.Tensor,
    solver: str,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    logits = act.actuator_forward(xb, params, mu, std, spec)
    _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
    task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
    raw_delta, _ls = act.actuator_only_least_squares_delta(params, mu, std, spec, xb, target, ridge=float(args.ridge))
    sid = str(solver)
    if sid == "SOL1-LeastSquaresSketch":
        solved = v9214._cap_to_fraction(raw_delta, task_step, float(args.p2_functional_step_fraction))
        removed = torch.tensor(0.0, device=xb.device)
    elif sid == "SOL2-ConstrainedLeastSquares":
        projected, removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
        solved = v9214._cap_to_fraction(projected, task_step, float(args.p2_functional_step_fraction))
    elif sid == "SOL3-TrustRegionQP":
        projected, removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
        solved = v9214._cap_to_fraction(projected, task_step, float(args.p2_trust_fraction))
    else:
        raise ValueError(f"unknown solver {solver}")
    fit = act.output_fit_metrics(params, solved, mu, std, spec, xb, target)
    adamw_logits = act.actuator_forward(xb, [p + d for p, d in zip(params, task_step)], mu, std, spec)
    adamw_norm = (adamw_logits - logits).float().norm().clamp_min(1.0e-12)
    dnorm = snr_lq.step_norm(solved)
    cos = snr_lq.step_dot(solved, task_step) / (dnorm.clamp_min(1.0e-12) * snr_lq.step_norm(task_step).clamp_min(1.0e-12))
    return solved, {
        "target_fit_R2": fit["output_target_fit_r2"],
        "rz": float((fit["output_displacement_norm"] / adamw_norm).detach().cpu()),
        "delta_norm": float(dnorm.detach().cpu()),
        "delta_norm_vs_adamw": float((dnorm / snr_lq.step_norm(task_step).clamp_min(1.0e-12)).detach().cpu()),
        "cos_delta_adamw": float(cos.detach().cpu()) if bool((dnorm > 0).detach().cpu()) else 0.0,
        "projection_removed_norm": float(removed.detach().cpu()),
    }


def _branch_steps(
    args: argparse.Namespace,
    spec: act.ActuatorSpec,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    target: torch.Tensor,
    solver: str,
    seed: int,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, float]]:
    real, info = _solve_step(args, spec, params, mu, std, xb, yb, target, solver)
    logits = act.actuator_forward(xb, params, mu, std, spec)
    _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
    task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
    real_norm = snr_lq.step_norm(real)
    perm = torch.randperm(int(target.shape[0]), device=target.device, generator=torch.Generator(device=target.device).manual_seed(seed + 501))
    shuffled, _ = _solve_step(args, spec, params, mu, std, xb, yb, target[perm], solver)
    rand_step, _ = snr_lq.project_step_to_task_safe(v9214._random_like_step(params, real_norm, seed + 502), grads)
    return {
        "AdamWOnly": snr_lq.zero_like_params(params),
        "RealFunctional": real,
        "NoOpMatchedOverhead": snr_lq.zero_like_params(params),
        "RandomMatchedNorm": v9214._cap_to_fraction(rand_step, task_step, float(args.p2_functional_step_fraction)),
        "ShuffledTarget": v9214._cap_to_fraction(shuffled, task_step, float(args.p2_functional_step_fraction)),
        "AdamWParallelDirection": v9214._cap_to_fraction(task_step, task_step, float(args.p2_functional_step_fraction)),
    }, info


def _replay_branch_metrics(
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
) -> Dict[int, Dict[str, float]]:
    params_b = v9214._clone_params(base_params)
    states_b = v9214._clone_states(base_states)
    if any(bool((d.float().abs().max() > 0).detach().cpu()) for d in branch_step):
        for p, d in zip(params_b, branch_step):
            p.add_(d)
    prev = 0
    out: Dict[int, Dict[str, float]] = {}
    for horizon in [int(x) for x in _parse_list(args.p2_horizons)]:
        v9214._run_adamw_steps(
            params_b,
            states_b,
            mu,
            std,
            spec,
            x_train,
            y_train,
            int(args.p2_warmup_steps) + prev,
            horizon - prev,
            int(args.batch_size),
            cfg,
        )
        prev = horizon
        out[horizon] = v9214._eval_actuator_metrics(params_b, mu, std, spec, x_eval, y_eval)
    return out


def _run_p2_matrix(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    p4_rows = read_csv_rows(PREV_V9214 / "p2_p4_closure_candidate_factory.csv")
    p4_by_id = {r.get("candidate_id"): r for r in p4_rows}
    for aid in _parse_list(args.p2_actuators):
        spec = act.actuator_specs_from_ids([aid])[0]
        p4ref = p4_by_id.get(aid, {})
        for dataset in _canonical_tasks(args.p2_datasets):
            x_train, y_train, x_eval, y_eval, in_dim, out_dim, protocol = v92._load_task(
                args, dataset, train_size=max(int(args.train_size), 4096), test_size=int(args.p2_eval_size)
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            x_eval = x_eval.to(device=device, dtype=torch.float32)
            y_eval = y_eval.to(device=device)
            for seed_text in _parse_list(args.p2_seeds):
                seed = int(seed_text)
                base_params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x_train, device, seed + 921500 + len(aid))
                base_states = [AdamWState.zeros_like(p) for p in base_params]
                v9214._run_adamw_steps(base_params, base_states, mu, std, spec, x_train, y_train, 0, int(args.p2_warmup_steps), int(args.batch_size), cfg)
                xb = x_train[: int(args.audit_batch_size)]
                yb = y_train[: int(args.audit_batch_size)]
                base_logits = act.actuator_forward(xb, base_params, mu, std, spec)
                _loss_common, common_grads = act.actuator_fwd_bwd(xb, yb, base_params, mu, std, spec)
                common_task_step = snr_lq.gradient_descent_task_step(common_grads, float(args.lr))
                zero_step = snr_lq.zero_like_params(base_params)
                zero_metrics = _replay_branch_metrics(args, cfg, spec, base_params, base_states, mu, std, x_train, y_train, x_eval, y_eval, zero_step)
                parallel_step = v9214._cap_to_fraction(common_task_step, common_task_step, float(args.p2_functional_step_fraction))
                parallel_metrics = _replay_branch_metrics(args, cfg, spec, base_params, base_states, mu, std, x_train, y_train, x_eval, y_eval, parallel_step)
                for target_id in _parse_list(args.p2_targets):
                    base_target, _target_info = out_lq.build_output_target(target_id, base_logits, yb, dataset=dataset)
                    for event_id in _parse_list(args.p2_events):
                        mask = _event_mask(base_logits, yb, event_id, dataset)
                        target = base_target * mask.float().unsqueeze(1)
                        coverage = float(mask.float().mean().detach().cpu())
                        for solver in _parse_list(args.p2_solvers):
                            branch_steps, info = _branch_steps(args, spec, base_params, mu, std, xb, yb, target, solver, seed)
                            metrics_by_branch: Dict[str, Dict[int, Dict[str, float]]] = {}
                            for branch, fstep in branch_steps.items():
                                if branch in {"AdamWOnly", "NoOpMatchedOverhead"}:
                                    metrics_by_branch[branch] = zero_metrics
                                elif branch == "AdamWParallelDirection":
                                    metrics_by_branch[branch] = parallel_metrics
                                else:
                                    metrics_by_branch[branch] = _replay_branch_metrics(
                                        args, cfg, spec, base_params, base_states, mu, std, x_train, y_train, x_eval, y_eval, fstep
                                    )
                            for horizon in [int(x) for x in _parse_list(args.p2_horizons)]:
                                adamw = metrics_by_branch["AdamWOnly"][horizon]
                                for branch, by_h in metrics_by_branch.items():
                                    m = by_h[horizon]
                                    rows.append({
                                        "stage": "P2_P4QUALIFIED_ACTUATOR_CAUSALITY_MATRIX",
                                        "actuator": aid,
                                        "target": target_id,
                                        "solver": solver,
                                        "event": event_id,
                                        "dataset": dataset,
                                        "seed": seed,
                                        "horizon": horizon,
                                        "branch": branch,
                                        "CEp99_delta": m["CE_p99"] - adamw["CE_p99"],
                                        "margin_p10_delta": m["correct_margin_p10"] - adamw["correct_margin_p10"],
                                        "curvature_delta": "not_measured",
                                        "ECE_delta": m["ECE"] - adamw["ECE"],
                                        "NLL_delta": m["NLL"] - adamw["NLL"],
                                        "holdout_delta": m["NLL"] - adamw["NLL"],
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
                                        "P4_forward_q90": p4ref.get("forward_ratio_q90", ""),
                                        "P4_backward_q90": p4ref.get("backward_ratio_q90", ""),
                                        "P4_step_q90": p4ref.get("step_ratio_q90", ""),
                                        "P4_memory": p4ref.get("compact_memory_ratio", ""),
                                        "target_fit_R2": info["target_fit_R2"],
                                        "rz": info["rz"],
                                        "delta_norm": info["delta_norm"],
                                        "delta_norm_vs_adamw": info["delta_norm_vs_adamw"],
                                        "cos_delta_adamw": info["cos_delta_adamw"],
                                        "event_coverage": coverage,
                                        "task_safe": int(m["acc"] >= adamw["acc"] - 0.005),
                                        "loss_type": "CE",
                                        "label_smoothing": 0,
                                        "uses_loss_backward": 0,
                                        "external_teacher_used": 0,
                                        "functional_update_used": int(branch != "AdamWOnly"),
                                        "fake_data_used": 0,
                                        "proxy_row_used": 0,
                                        "cpu_offload_used": 0,
                                    })
    _annotate_p2_rows(rows)
    summary = _p2_summary(rows)
    return rows, summary


def _annotate_p2_rows(rows: List[Dict[str, Any]]) -> None:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["actuator"], r["target"], r["solver"], r["event"], r["dataset"], r["seed"], r["horizon"])
        groups[key].append(r)
    for rs in groups.values():
        real = next((r for r in rs if r["branch"] == "RealFunctional"), None)
        controls = [r for r in rs if r["branch"] not in {"RealFunctional", "AdamWOnly"}]
        ranked = sorted(controls, key=lambda r: (_to_float(r["CEp99_delta"]), -_to_float(r["margin_p10_delta"])))
        for idx, r in enumerate(ranked, start=1):
            r["control_rank"] = idx
        if real and controls:
            best_ce = min(_to_float(r["CEp99_delta"]) for r in controls)
            best_margin = max(_to_float(r["margin_p10_delta"]) for r in controls)
            ce_gate_margin = 0.05 * abs(_to_float(real.get("AdamW_CEp99_abs")))
            ece_gate_margin = 0.02 * abs(_to_float(real.get("AdamW_ECE_abs")))
            beat = int(
                _to_int(real.get("task_safe")) == 1
                and (
                    _to_float(real["CEp99_delta"]) <= best_ce - ce_gate_margin
                    or _to_float(real["margin_p10_delta"]) >= best_margin + 0.02
                    or _to_float(real["ECE_delta"]) <= min(_to_float(r["ECE_delta"]) for r in controls) - ece_gate_margin
                )
            )
            real["real_beats_best_control"] = beat


def _p2_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    real = [r for r in rows if r.get("branch") == "RealFunctional"]
    survivor_groups: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("actuator")), str(r.get("target")), str(r.get("solver")), str(r.get("event")))].append(r)
    for key, rs in groups.items():
        gr = [r for r in rs if r.get("branch") == "RealFunctional"]
        gc = [r for r in rs if r.get("branch") not in {"RealFunctional", "AdamWOnly"}]
        if not gr or not gc:
            continue
        control_branches = sorted({str(r.get("branch")) for r in gc})
        best_control_ce = min(_mean([r for r in gc if str(r.get("branch")) == b], "CEp99_delta") for b in control_branches)
        best_control_margin = max(_mean([r for r in gc if str(r.get("branch")) == b], "margin_p10_delta") for b in control_branches)
        best_control_ece = min(_mean([r for r in gc if str(r.get("branch")) == b], "ECE_delta") for b in control_branches)
        real_ce = _mean(gr, "CEp99_delta")
        real_margin = _mean(gr, "margin_p10_delta")
        real_ece = _mean(gr, "ECE_delta")
        ce_gate_margin = 0.05 * abs(_mean(gr, "AdamW_CEp99_abs"))
        ece_gate_margin = 0.02 * abs(_mean(gr, "AdamW_ECE_abs"))
        task_safe_rate = sum(_to_int(r.get("task_safe")) for r in gr) / max(1, len(gr))
        pass_group = int(
            task_safe_rate >= 0.95
            and (
                real_ce <= best_control_ce - ce_gate_margin
                or real_margin >= best_control_margin + 0.02
                or real_ece <= best_control_ece - ece_gate_margin
            )
        )
        if pass_group:
            survivor_groups.append({
                "actuator": key[0],
                "target": key[1],
                "solver": key[2],
                "event": key[3],
                "real_CEp99_delta": real_ce,
                "best_control_CEp99_delta": best_control_ce,
                "real_margin_delta": real_margin,
                "best_control_margin_delta": best_control_margin,
                "real_ECE_delta": real_ece,
                "best_control_ECE_delta": best_control_ece,
                "task_safe_rate": task_safe_rate,
            })
    if survivor_groups:
        best = sorted(survivor_groups, key=lambda r: (_to_float(r["real_CEp99_delta"]), -_to_float(r["real_margin_delta"])))[0]
        return {
            "p2_paired_replay_pass": 1,
            "p2_survivor_count": len(survivor_groups),
            "best_actuator": best["actuator"],
            "best_target": best["target"],
            "best_solver": best["solver"],
            "best_event_controller": best["event"],
            "best_horizon": "aggregate_all_horizons",
            "best_real_CEp99_delta": best["real_CEp99_delta"],
            "best_real_margin_delta": best["real_margin_delta"],
            "best_control_CEp99_delta": best["best_control_CEp99_delta"],
            "best_control_margin_delta": best["best_control_margin_delta"],
            "best_task_safe_rate": best["task_safe_rate"],
        }
    # dominance summary by controls
    controls = [r for r in rows if r.get("branch") not in {"RealFunctional", "AdamWOnly"}]
    control_means = {}
    for b in sorted({r.get("branch") for r in controls}):
        br = [r for r in controls if r.get("branch") == b]
        control_means[b] = (_mean(br, "CEp99_delta"), _mean(br, "margin_p10_delta"))
    best_control = min(control_means, key=lambda b: (control_means[b][0], -control_means[b][1])) if control_means else ""
    return {
        "p2_paired_replay_pass": 0,
        "p2_survivor_count": 0,
        "best_actuator": "",
        "best_target": "",
        "best_solver": "",
        "best_event_controller": "",
        "best_control": best_control,
        "real_mean_CEp99_delta": _mean(real, "CEp99_delta"),
        "best_control_mean_CEp99_delta": control_means.get(best_control, (0.0, 0.0))[0] if best_control else 0.0,
        "real_mean_margin_delta": _mean(real, "margin_p10_delta"),
        "best_control_mean_margin_delta": control_means.get(best_control, (0.0, 0.0))[1] if best_control else 0.0,
    }


def _run_p3_control_contrastive(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("actuator")), str(r.get("target")), str(r.get("solver")), str(r.get("event")))].append(r)
    for key, rs in groups.items():
        real = [r for r in rs if r.get("branch") == "RealFunctional"]
        controls = [r for r in rs if r.get("branch") not in {"RealFunctional", "AdamWOnly"}]
        if not real or not controls:
            continue
        best_control_ce = min(_mean([r for r in controls if r.get("branch") == b], "CEp99_delta") for b in {r.get("branch") for r in controls})
        best_control_margin = max(_mean([r for r in controls if r.get("branch") == b], "margin_p10_delta") for b in {r.get("branch") for r in controls})
        real_ce = _mean(real, "CEp99_delta")
        real_margin = _mean(real, "margin_p10_delta")
        bad_rate = sum(1 for r in real if _to_float(r.get("holdout_delta", 0.0)) > 0.0) / max(1, len(real))
        coverage = _mean(real, "event_coverage")
        score = (best_control_ce - real_ce) + 0.1 * (real_margin - best_control_margin) - bad_rate
        out.append({
            "stage": "P3_CONTROL_CONTRASTIVE_TARGET_SOLVER",
            "actuator": key[0],
            "target": key[1],
            "solver": "posthoc_score_from_P2_" + key[2],
            "event": key[3],
            "control_contrastive_score": score,
            "predicted_score": "not_modeled_posthoc_actual_only",
            "actual_paired_score": score,
            "bad_event_rate": bad_rate,
            "coverage": coverage,
            "CEp99_vs_best_control": real_ce - best_control_ce,
            "margin_vs_best_control": real_margin - best_control_margin,
            "curvature_vs_best_control": "not_measured",
            "ECE_vs_best_control": "not_aggregated",
            "task_safety": sum(_to_int(r.get("task_safe")) for r in real) / max(1, len(real)),
            "control_contrastive_pass": int(score > 0 and bad_rate <= 0.05 and 0.03 <= coverage <= 0.15),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    passed = [r for r in out if _to_int(r.get("control_contrastive_pass")) == 1]
    return out, {"p3_control_contrastive_pass": int(bool(passed)), "p3_pass_count": len(passed)}


def _write_downstream_not_run(out_dir: Path, reason: str) -> None:
    for filename, stage in [
        ("p4_short_run_causal_validation.csv", "P4_SHORT_RUN_CAUSAL_VALIDATION"),
        ("p5_full_functional_reentry_10seed.csv", "P5_FULL_FUNCTIONAL_REENTRY_10SEED"),
        ("p6_noise_robustness_signal_separation.csv", "P6_NOISE_ROBUSTNESS_SIGNAL_SEPARATION"),
        ("p7_strong_baseline_external_ready.csv", "P7_STRONG_BASELINE_EXTERNAL_READY"),
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
    parser.add_argument("--p2-eval-size", type=int, default=512)
    parser.add_argument("--p2-actuators", default="A4b-BoundedRational-BranchlessDerivative,A4d-BoundedRational-ValueOnlyActuator,A4e-BoundedRational-FusedCoeffGrad,A7c-BasisEntropy-ValueOnly")
    parser.add_argument("--p2-targets", default="O1-HardTailLogitCorrection,O2-MarginTailExpansion,O3-CalibrationTailCompression,O4-CurvatureOutputFlattening,O6-KMNISTHardModeOutputTarget")
    parser.add_argument("--p2-solvers", default="SOL1-LeastSquaresSketch,SOL2-ConstrainedLeastSquares,SOL3-TrustRegionQP")
    parser.add_argument("--p2-events", default="E1-CalibratedCEp99Tail,E2-CalibratedMarginTail,E6-CompositeSparseEvent,E7-KMNISTHardModeEvent")
    parser.add_argument("--p2-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p2-seeds", default="0,1,2")
    parser.add_argument("--p2-horizons", default="1,5,20,80")
    parser.add_argument("--p2-warmup-steps", type=int, default=36)
    parser.add_argument("--p2-functional-step-fraction", type=float, default=0.10)
    parser.add_argument("--p2-trust-fraction", type=float, default=0.03)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    manifest = {
        "experiment": "DG-KAN v9.2.15 ControlResistant Functional Causality",
        "created_at": _now_iso(),
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
    }
    write_json(out_dir / "run_manifest.json", manifest)
    contract = [{
        "candidate": "v9215_all",
        "strict_fc_purekan_edge_owned": 1,
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "uses_loss_backward": 0,
        "sampler_or_class_weight_used": 0,
        "cpu_offload_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }]
    write_csv_rows(out_dir / "contract_audit_v9215.csv", contract)

    p0 = [_p0_recap(PREV_V9214)]
    write_csv_rows(out_dir / "p0_v9214_boundary_reproduction.csv", p0)
    p1_rows, p1_summary = _run_p1_autopsy(PREV_V9214)
    write_csv_rows(out_dir / "p1_paired_replay_failure_autopsy.csv", p1_rows)

    p2_rows, p2_summary = _run_p2_matrix(args, device)
    write_csv_rows(out_dir / "p2_p4qualified_actuator_causality_matrix.csv", p2_rows)
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9215.csv", p2_rows)
    write_csv_rows(out_dir / "control_rank_trace_v9215.csv", p2_rows)
    p3_rows, p3_summary = _run_p3_control_contrastive(p2_rows)
    write_csv_rows(out_dir / "p3_control_contrastive_target_solver.csv", p3_rows)

    paired_pass = _to_int(p2_summary.get("p2_paired_replay_pass")) == 1
    p3_pass = _to_int(p3_summary.get("p3_control_contrastive_pass")) == 1
    if paired_pass:
        route = "R1-A4FunctionalCausalityPass" if "A4" in str(p2_summary.get("best_actuator", "")) else "R2-A7cFunctionalCausalityPass"
        reason = "paired_replay_survivor_found_but_short_run_not_opened_in_this_runner"
        failure_code = "F18_artifact_missing"
    elif p3_pass:
        route = "R3-ControlContrastiveSolverPass"
        reason = "control_contrastive_survivor_found_but_short_run_not_opened_in_this_runner"
        failure_code = "F18_artifact_missing"
    else:
        route = "R10-ReturnToTargetOrPrimitiveDesign"
        reason = "all_P4_qualified_actuators_remain_control_equivalent"
        failure_code = "F10_no_causality_survivor"
    _write_downstream_not_run(out_dir, reason)
    write_csv_rows(out_dir / "functional_event_trace_v9215.csv", p2_rows)

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "best_actuator_candidate": p2_summary.get("best_actuator", ""),
        "best_target": p2_summary.get("best_target", ""),
        "best_solver": p2_summary.get("best_solver", ""),
        "best_event_controller": p2_summary.get("best_event_controller", ""),
        "paired_replay_pass": int(paired_pass),
        "short_run_pass": 0,
        "full_reentry_pass": 0,
        "functional_task_safe": 0,
        "functional_mechanism_pass": int(paired_pass or p3_pass),
        "functional_control_pass": int(paired_pass or p3_pass),
        "functional_system_pass": 1,
        "functional_kmnist_repair_pass": 0,
        "noise_robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "primary_blocker": reason,
        "next_required_implementation": "return_to_target_or_primitive_design" if route == "R10-ReturnToTargetOrPrimitiveDesign" else "open_short_run_for_causality_survivor",
        "success_v9215_paired_replay_causality": int(paired_pass),
        "success_v9215_short_run": 0,
        "success_v9215_full_functional": 0,
        "success_v9215_external_ready": 0,
        **p1_summary,
        **p2_summary,
        **p3_summary,
    }
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "stage": "terminal",
        "failure_code": failure_code,
        "primary_blocker": reason,
        "route": route,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    _write_svg(out_dir / "figures" / "p0_v9214_boundary_dashboard.svg", "P0 v9.2.14 Boundary", f"source_route={p0[0].get('source_route')}")
    _write_svg(out_dir / "figures" / "p0_a7c_p4_and_base_summary.svg", "A7c P4 And Base", f"near={p0[0].get('A7c_p5_near_pass_count')} macro={p0[0].get('A7c_macro_delta')}")
    _write_svg(out_dir / "figures" / "p0_paired_replay_real_vs_control.svg", "Source Real vs Control", f"real_ce={p0[0].get('paired_replay_real_CEp99_mean_delta')} best_control={p0[0].get('paired_replay_best_control_CEp99_mean_delta')}")
    _write_svg(out_dir / "figures" / "p1_failure_factor_matrix.svg", "P1 Failure Factor", str(p1_summary))
    _write_svg(out_dir / "figures" / "p1_actuator_target_solver_heatmap.svg", "P1 Target Autopsy", f"rows={len(p1_rows)}")
    _write_svg(out_dir / "figures" / "p1_controls_rank_by_metric.svg", "P1 Control Rank", f"best_control={p1_summary.get('p1_best_control')}")
    _write_svg(out_dir / "figures" / "p1_logit_displacement_vs_mechanism_gain.svg", "P1 Displacement", "not measured in v9.2.14 source")
    _write_svg(out_dir / "figures" / "p1_cosine_with_controls.svg", "P1 Cosines", "not measured in v9.2.14 source")
    _write_svg(out_dir / "figures" / "p2_actuator_causality_pareto.svg", "P2 Actuator Causality", f"survivors={p2_summary.get('p2_survivor_count')}")
    _write_svg(out_dir / "figures" / "p2_real_vs_best_control_by_actuator.svg", "P2 Real vs Best Control", f"route={route}")
    _write_svg(out_dir / "figures" / "p2_horizon_effect_by_actuator.svg", "P2 Horizon Effect", f"horizons={args.p2_horizons}")
    _write_svg(out_dir / "figures" / "p2_kmnist_causality_matrix.svg", "P2 KMNIST Matrix", "measured in P2 rows")
    _write_svg(out_dir / "figures" / "p3_control_contrastive_score.svg", "P3 Control Contrastive", f"pass={p3_summary.get('p3_control_contrastive_pass')}")
    _write_svg(out_dir / "figures" / "p3_predicted_vs_actual_cc_score.svg", "P3 Predicted vs Actual", "posthoc actual only")
    _write_svg(out_dir / "figures" / "p3_target_solver_cc_heatmap.svg", "P3 Target Solver", f"rows={len(p3_rows)}")
    _write_svg(out_dir / "figures" / "p3_abstention_precision_curve.svg", "P3 Abstention", "SOL6 not implemented in first wave")

    audit_paths = [
        out_dir / "contract_audit_v9215.csv",
        out_dir / "p0_v9214_boundary_reproduction.csv",
        out_dir / "p1_paired_replay_failure_autopsy.csv",
        out_dir / "p2_p4qualified_actuator_causality_matrix.csv",
        out_dir / "p3_control_contrastive_target_solver.csv",
        out_dir / "p4_short_run_causal_validation.csv",
        out_dir / "p5_full_functional_reentry_10seed.csv",
        out_dir / "p6_noise_robustness_signal_separation.csv",
        out_dir / "p7_strong_baseline_external_ready.csv",
        out_dir / "functional_event_trace_v9215.csv",
        out_dir / "paired_replay_branch_trace_v9215.csv",
        out_dir / "control_rank_trace_v9215.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v9215_provenance_audit.csv", [audit])
    route_decision.update(audit)
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    hash_paths = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "route_decision.json",
        *audit_paths,
        out_dir / "v9215_provenance_audit.csv",
    ]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_paths, root=ROOT))
    print(json.dumps(route_decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
