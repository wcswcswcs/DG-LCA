#!/usr/bin/env python3
"""DG-KAN v9.2.10 functional predictor repair runner.

This runner executes the gate-ordered first wave from the v9.2.10 plan:
P0 reproduction, P1 attribution, and P2 one-step direction/gate factory.  It
does not open short-run or full functional training unless P2 produces a real
one-step-safe survivor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
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
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import lq_functional_predictor as fp_lq  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.10_Functional_Predictor_Repair_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9210_functional_predictor_repair.py"
PREV_V929 = ROOT / "results" / "real_rerun_20260506" / "v929_snr_gated_functional_update_p2_warmema_20260509T193500Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _canonical_tasks(text: str) -> List[str]:
    return [v92._canonical_task(x) for x in _parse_list(text)]


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    tx = torch.tensor(list(xs), dtype=torch.float64)
    ty = torch.tensor(list(ys), dtype=torch.float64)
    vx = tx - tx.mean()
    vy = ty - ty.mean()
    denom = float(vx.square().sum().sqrt() * vy.square().sum().sqrt())
    if denom <= 1.0e-12:
        return 0.0
    return float((vx * vy).sum().item() / denom)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(device: torch.device, fn):
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0, out


def _write_svg(path: Path, title: str, subtitle: str) -> None:
    ensure_dir(path.parent)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="190" viewBox="0 0 980 190">
  <rect width="980" height="190" fill="#f8fafc"/>
  <text x="28" y="54" font-family="Arial, sans-serif" font-size="25" fill="#111827">{title}</text>
  <text x="28" y="96" font-family="Arial, sans-serif" font-size="16" fill="#374151">{subtitle}</text>
  <text x="28" y="136" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">Generated from measured CSV/JSON fields; no inferred pass values.</text>
</svg>
""",
        encoding="utf-8",
    )


def _not_run(stage: str, artifact: str, reason: str) -> Dict[str, Any]:
    return snr_lq.not_run_row(stage, artifact, reason)


def _candidate_specs(text: str) -> List[lq.LQSpec]:
    specs: List[lq.LQSpec] = []
    for name in _parse_list(text):
        if name == "LQ0":
            specs.append(lq.LQSpec("LQ0-LQ-t2-h256-AdamW", "t2", 256, "default", 1.0))
        elif name == "LQ1":
            specs.append(lq.LQSpec("LQ1-LQ-t2-h256-fanin-output-scale", "t2", 256, "default", 0.8))
        else:
            raise ValueError(f"unknown candidate alias {name}")
    return specs


def _eval_lq_metrics(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    fwd, _bwd = lq.functions_for_basis(basis)
    with torch.no_grad():
        logits = fwd(x, *params, mu, std, 2.0, 2.0)
    return v92._classification_metrics_from_logits(logits, y)


def _quadratic_active_fraction(rows: Sequence[Dict[str, Any]]) -> float:
    for row in rows:
        role = str(row.get("role", ""))
        if "quadratic" in role or "legendre_p2" in role:
            return _to_float(row.get("SNR_active_fraction_tau1"), 0.0)
    return _to_float(rows[-1].get("SNR_active_fraction_tau1"), 0.0) if rows else 0.0


def _quadratic_snr(rows: Sequence[Dict[str, Any]]) -> float:
    for row in rows:
        role = str(row.get("role", ""))
        if "quadratic" in role or "legendre_p2" in role:
            return _to_float(row.get("SNR_role"), 0.0)
    return _to_float(rows[-1].get("SNR_role"), 0.0) if rows else 0.0


def _gate_multiplier(gate_id: str, active_fraction: float, quadratic_snr: float, high_tau: float) -> float:
    if gate_id == "G1-RoleSNROnly":
        return float(active_fraction)
    if gate_id == "G2-HighConfidenceRoleSNR":
        return float(active_fraction) if float(quadratic_snr) >= float(high_tau) else 0.0
    if gate_id == "G4-HoldoutPrecheckGate":
        return float(active_fraction)
    if gate_id == "G6-ShuffledSNRControl":
        return float(active_fraction)
    if gate_id == "G7-InvertedSNRControl":
        return max(0.0, 1.0 - float(active_fraction))
    raise ValueError(f"unknown gate {gate_id}")


def _make_delta(
    *,
    direction_id: str,
    gate_id: str,
    params: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor],
    basis: str,
    task_step: Sequence[torch.Tensor],
    step_fraction: float,
    gate_multiplier: float,
    device: torch.device,
    seed: int,
) -> List[torch.Tensor]:
    if direction_id == "C0-NoOp":
        return snr_lq.zero_like_params(params)
    if direction_id == "C2-RandomDirectionMatchedNorm":
        raw = fp_lq.random_matched_direction(params, task_step, seed=seed, device=device)
        return snr_lq.scale_step(raw, float(step_fraction) * float(gate_multiplier))
    raw_direction = fp_lq.direction_for_id(direction_id, params, grads, basis)
    scaled = snr_lq.scale_direction_to_fraction_of_task_step(raw_direction, task_step, float(step_fraction))
    return snr_lq.scale_step(scaled, float(gate_multiplier))


def _run_one_step_factory(args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = _candidate_specs(args.candidates)
    snr_cfg = snr_lq.SNRConfig(
        microbatch_count=1,
        tau1=float(args.snr_tau1),
        tau2=float(args.snr_tau2),
        temperature=float(args.snr_smooth_temperature),
    )
    directions = _parse_list(args.directions)
    gates = _parse_list(args.gates)
    step_fracs = [float(x) for x in _parse_list(args.step_fractions)]
    for dataset in _canonical_tasks(args.datasets):
        min_train = max(int(args.train_size), int(args.microbatch_size) * 2 * (int(args.steps) + 2 + int(args.snr_warmup_steps)))
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, protocol = v92._load_task(
            args,
            dataset,
            train_size=min_train,
            test_size=256,
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        for seed_text in _parse_list(args.seeds):
            seed = int(seed_text)
            for spec in specs:
                torch.manual_seed(int(args.seed) + seed + 921000)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(args.seed) + seed + 921000)
                params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, seed + 92920)
                states = [AdamWState.zeros_like(p) for p in params]
                opt_cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
                scalar_state = snr_lq.ScalarRoleSNRState.zeros(
                    len(snr_lq.role_names_for_basis(spec.basis)),
                    beta=float(args.snr_ema_beta),
                    device=device,
                )
                _fwd, bwd = lq.functions_for_basis(spec.basis)
                for warm_step in range(int(args.snr_warmup_steps)):
                    gen = torch.Generator(device=device).manual_seed(9291000 + seed * 10000 + warm_step)
                    idx = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)[: int(args.batch_size)]
                    pack = bwd(x_train[idx], y_train[idx], *params, mu, std, 2.0, 2.0)
                    grads = [g.detach() for g in pack[1:]]
                    snr_lq.scalar_ema_snr_from_grads(grads=grads, scalar_state=scalar_state, basis=spec.basis, snr_cfg=snr_cfg)
                    v92._adamw_update_foreach_(params, grads, states, opt_cfg)
                for step in range(int(args.steps)):
                    gen = torch.Generator(device=device).manual_seed(9292000 + seed * 10000 + step)
                    perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
                    n_micro = int(args.microbatch_size)
                    train_idx = perm[:n_micro]
                    hold_idx = perm[n_micro : 2 * n_micro]
                    xb = x_train[train_idx]
                    yb = y_train[train_idx]
                    xh = x_train[hold_idx]
                    yh = y_train[hold_idx]
                    bwd_ms, pack = _time_ms(device, lambda: bwd(xb, yb, *params, mu, std, 2.0, 2.0))
                    grads = [g.detach() for g in pack[1:]]
                    snr_rows = snr_lq.scalar_ema_snr_from_grads(
                        grads=grads,
                        scalar_state=scalar_state,
                        basis=spec.basis,
                        snr_cfg=snr_cfg,
                    )
                    active_fraction = _quadratic_active_fraction(snr_rows)
                    quadratic_snr = _quadratic_snr(snr_rows)
                    before_metrics = _eval_lq_metrics(params, mu, std, spec.basis, xh, yh)
                    before_mech = fp_lq.mechanism_metrics(params, mu, std, spec.basis, xh)
                    task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
                    adamw_step_norm = snr_lq.step_norm(task_step)
                    for direction_id in directions:
                        for gate_id in gates:
                            if direction_id in {"C0-NoOp", "C2-RandomDirectionMatchedNorm"} and gate_id != "G1-RoleSNROnly":
                                continue
                            for step_fraction in step_fracs:
                                if direction_id == "C0-NoOp" and step_fraction != step_fracs[0]:
                                    continue
                                multiplier = _gate_multiplier(gate_id, active_fraction, quadratic_snr, float(args.high_snr_tau))
                                raw_delta = _make_delta(
                                    direction_id=direction_id,
                                    gate_id=gate_id,
                                    params=params,
                                    grads=grads,
                                    basis=spec.basis,
                                    task_step=task_step,
                                    step_fraction=step_fraction,
                                    gate_multiplier=multiplier,
                                    device=device,
                                        seed=9392000 + seed * 10000 + step,
                                )
                                if direction_id == "C0-NoOp":
                                    projected_delta = raw_delta
                                    removed_norm = torch.zeros((), device=device)
                                else:
                                    projected_delta, removed_norm = snr_lq.project_step_to_task_safe(raw_delta, grads)
                                proposed_params = snr_lq.apply_step(params, projected_delta)
                                guard_ms = 0.0
                                proposed_metrics = None
                                event_accepted = int(snr_lq.step_norm(projected_delta).detach().cpu() > 0)
                                if gate_id == "G4-HoldoutPrecheckGate" and direction_id != "C0-NoOp":
                                    guard_ms, proposed_metrics = _time_ms(
                                        device,
                                        lambda: _eval_lq_metrics(proposed_params, mu, std, spec.basis, xh, yh),
                                    )
                                    if proposed_metrics["loss"] > before_metrics["loss"]:
                                        event_accepted = 0
                                        final_delta = snr_lq.zero_like_params(params)
                                    else:
                                        final_delta = projected_delta
                                else:
                                    final_delta = projected_delta
                                after_params = snr_lq.apply_step(params, final_delta)
                                after_metrics = proposed_metrics if (gate_id == "G4-HoldoutPrecheckGate" and event_accepted) else _eval_lq_metrics(after_params, mu, std, spec.basis, xh, yh)
                                after_mech = fp_lq.mechanism_metrics(after_params, mu, std, spec.basis, xh)
                                actual_delta = after_metrics["loss"] - before_metrics["loss"]
                                proposed_delta = (proposed_metrics["loss"] - before_metrics["loss"]) if proposed_metrics is not None else actual_delta
                                g_dot = snr_lq.step_dot(grads, final_delta)
                                func_norm = snr_lq.step_norm(final_delta)
                                cos = g_dot / (snr_lq.step_norm(grads).clamp_min(1.0e-12) * func_norm.clamp_min(1.0e-12))
                                amortized_overhead = float(args.base_snr_overhead_ratio) + (guard_ms / max(bwd_ms, 1.0e-12))
                                rows.append(
                                    {
                                        "stage": "P2_DIRECTION_FACTORY_ONE_STEP",
                                        "candidate_id": spec.candidate_id,
                                        "dataset": dataset,
                                        "seed": seed,
                                        "protocol": protocol,
                                        "step": step,
                                        "direction_id": direction_id,
                                        "gate_id": gate_id,
                                        "step_fraction": step_fraction,
                                        "functional_mode": f"{direction_id}+{gate_id}+rho{step_fraction}",
                                        "route_eligible_direction": int(fp_lq.is_route_eligible_direction(direction_id) and direction_id not in {"C0-NoOp", "C2-RandomDirectionMatchedNorm"}),
                                        "event_accepted": event_accepted,
                                        "event_coverage_unit": int(event_accepted != 0),
                                        "active_fraction": active_fraction,
                                        "quadratic_role_snr": quadratic_snr,
                                        "gate_multiplier": multiplier,
                                        "predicted_population_improvement": float(g_dot.detach().cpu()),
                                        "actual_holdout_loss_before": before_metrics["loss"],
                                        "actual_holdout_loss_after": after_metrics["loss"],
                                        "actual_holdout_delta": actual_delta,
                                        "proposed_holdout_delta_before_guard": proposed_delta,
                                        "holdout_nonharm": int(actual_delta <= 0.0),
                                        "bad_step": int(actual_delta > 0.0),
                                        "task_gradient_dot_functional_step": float(g_dot.detach().cpu()),
                                        "projection_removed_norm": float(removed_norm.detach().cpu()),
                                        "functional_step_norm": float(func_norm.detach().cpu()),
                                        "adamw_step_norm": float(adamw_step_norm.detach().cpu()),
                                        "cos_functional_task": float(cos.detach().cpu()) if math.isfinite(float(cos.detach().cpu())) else 0.0,
                                        "CEp99_before": before_metrics["CE_p99"],
                                        "CEp99_after": after_metrics["CE_p99"],
                                        "CEp99_delta": after_metrics["CE_p99"] - before_metrics["CE_p99"],
                                        "margin_p10_before": before_metrics["correct_margin_p10"],
                                        "margin_p10_after": after_metrics["correct_margin_p10"],
                                        "margin_p10_delta": after_metrics["correct_margin_p10"] - before_metrics["correct_margin_p10"],
                                        "curvature_before": before_mech["curvature_proxy"],
                                        "curvature_after": after_mech["curvature_proxy"],
                                        "curvature_delta": after_mech["curvature_proxy"] - before_mech["curvature_proxy"],
                                        "local_lipschitz_before": before_mech["local_lipschitz_proxy"],
                                        "local_lipschitz_after": after_mech["local_lipschitz_proxy"],
                                        "local_lipschitz_delta": after_mech["local_lipschitz_proxy"] - before_mech["local_lipschitz_proxy"],
                                        "basis_usage_entropy_before": before_mech["basis_usage_entropy"],
                                        "basis_usage_entropy_after": after_mech["basis_usage_entropy"],
                                        "basis_usage_entropy_delta": after_mech["basis_usage_entropy"] - before_mech["basis_usage_entropy"],
                                        "bwd_time_ms": bwd_ms,
                                        "holdout_guard_time_ms": guard_ms,
                                        "amortized_overhead_ratio": amortized_overhead,
                                        "loss_type": "CE",
                                        "label_smoothing": 0,
                                        "external_teacher_used": 0,
                                        "self_teacher_used": 0,
                                        "teacher_logits_used": 0,
                                        "distillation_used": 0,
                                        "geometry_loss_used": 0,
                                        "sampler_changed": 0,
                                        "class_weight_used": 0,
                                        "uses_loss_backward": 0,
                                        "functional_update_used": int(direction_id != "C0-NoOp" and event_accepted),
                                        "fake_data_used": 0,
                                        "proxy_row_used": 0,
                                        "cpu_offload_used": 0,
                                    }
                                )
                    v92._adamw_update_foreach_(params, grads, states, opt_cfg)
    return rows


def _summarize_groups(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    out: List[Dict[str, Any]] = []
    for key, rs in groups.items():
        accepted = [r for r in rs if _to_int(r.get("event_accepted")) == 1 and _to_int(r.get("functional_update_used")) == 1]
        n = len(rs)
        an = len(accepted)
        eval_rows = accepted if accepted else rs
        bad = sum(_to_int(r.get("bad_step")) for r in eval_rows) / max(1, len(eval_rows))
        nonharm = sum(_to_int(r.get("holdout_nonharm")) for r in eval_rows) / max(1, len(eval_rows))
        corr = _pearson([_to_float(r.get("predicted_population_improvement")) for r in rs], [_to_float(r.get("actual_holdout_delta")) for r in rs])
        mean_delta = sum(_to_float(r.get("actual_holdout_delta")) for r in eval_rows) / max(1, len(eval_rows))
        ce_delta = sum(_to_float(r.get("CEp99_delta")) for r in eval_rows) / max(1, len(eval_rows))
        margin_delta = sum(_to_float(r.get("margin_p10_delta")) for r in eval_rows) / max(1, len(eval_rows))
        curv_delta = sum(_to_float(r.get("curvature_delta")) for r in eval_rows) / max(1, len(eval_rows))
        overhead = sum(_to_float(r.get("amortized_overhead_ratio")) for r in rs) / max(1, n)
        coverage = an / max(1, n)
        mechanism = int(ce_delta < 0.0 or margin_delta > 0.0 or curv_delta < 0.0)
        row = {k: v for k, v in zip(keys, key)}
        row.update(
            {
                "rows": n,
                "accepted_rows": an,
                "functional_event_coverage": coverage,
                "bad_step_rate_accepted": bad,
                "holdout_nonharm_fraction_accepted": nonharm,
                "prediction_corr_all": corr,
                "mean_holdout_delta_accepted": mean_delta,
                "CEp99_delta_accepted": ce_delta,
                "margin_p10_delta_accepted": margin_delta,
                "curvature_delta_accepted": curv_delta,
                "mechanism_benefit_pass": mechanism,
                "amortized_overhead_ratio": overhead,
                "route_eligible_direction": min(_to_int(r.get("route_eligible_direction")) for r in rs),
                "p2_safety_pass": int(an > 0 and bad <= 0.05 and nonharm >= 0.70),
                "p2_coverage_pass": int(coverage >= 0.05),
                "p2_system_pass": int(overhead <= 0.20),
            }
        )
        row["p2_survivor"] = int(
            row["route_eligible_direction"]
            and row["p2_safety_pass"]
            and row["p2_coverage_pass"]
            and row["p2_system_pass"]
            and row["mechanism_benefit_pass"]
        )
        out.append(row)
    return out


def _auc_binary(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)]
    pos = [p for p in pairs if p[1] == 1]
    neg = [p for p in pairs if p[1] == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for ps, _ in pos:
        for ns, _ in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / float(len(pos) * len(neg))


def _run_p3_predictor(rows: Sequence[Dict[str, Any]], best: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not best:
        return [_not_run("P3_CALIBRATED_FUNCTIONAL_PREDICTOR", "p3_calibrated_functional_predictor.csv", "no_P2_survivor")]
    chosen = [
        r
        for r in rows
        if str(r.get("direction_id")) == str(best.get("direction_id"))
        and str(r.get("gate_id")) == str(best.get("gate_id"))
        and abs(_to_float(r.get("step_fraction")) - _to_float(best.get("step_fraction"))) < 1.0e-12
    ]
    if not chosen:
        return [_not_run("P3_CALIBRATED_FUNCTIONAL_PREDICTOR", "p3_calibrated_functional_predictor.csv", "best_P2_event_rows_missing")]
    labels = [1 - _to_int(r.get("bad_step")) for r in chosen]
    pred_scores = [-_to_float(r.get("predicted_population_improvement")) for r in chosen]
    margin_scores = [_to_float(r.get("margin_p10_before")) for r in chosen]
    snr_scores = [_to_float(r.get("quadratic_role_snr")) for r in chosen]
    candidates = [
        ("PRED0-AllAcceptP2Survivor", [1 for _ in chosen], [1 for _ in chosen], 0.5),
        ("PRED1-PredictedImprovementScore", [int(s >= sorted(pred_scores)[max(0, int(0.10 * len(pred_scores)) - 1)]) for s in pred_scores], pred_scores, _auc_binary(pred_scores, labels)),
        ("PRED2-HighSNRTopQuartile", [int(s >= sorted(snr_scores)[max(0, int(0.75 * len(snr_scores)) - 1)]) for s in snr_scores], snr_scores, _auc_binary(snr_scores, labels)),
        ("PRED3-MarginRiskBottomQuartile", [int(s <= sorted(margin_scores)[min(len(margin_scores) - 1, int(0.25 * len(margin_scores)))]) for s in margin_scores], [-s for s in margin_scores], _auc_binary([-s for s in margin_scores], labels)),
    ]
    out: List[Dict[str, Any]] = []
    for predictor_id, accepts, scores, auc in candidates:
        accepted = [r for r, keep in zip(chosen, accepts) if keep]
        coverage = len(accepted) / max(1, len(chosen))
        bad = sum(_to_int(r.get("bad_step")) for r in accepted) / max(1, len(accepted))
        nonharm = sum(_to_int(r.get("holdout_nonharm")) for r in accepted) / max(1, len(accepted))
        mean_delta = sum(_to_float(r.get("actual_holdout_delta")) for r in accepted) / max(1, len(accepted))
        out.append(
            {
                "stage": "P3_CALIBRATED_FUNCTIONAL_PREDICTOR",
                "predictor_id": predictor_id,
                "source_direction": best.get("direction_id", ""),
                "source_gate": best.get("gate_id", ""),
                "source_step_fraction": best.get("step_fraction", ""),
                "rows": len(chosen),
                "accepted_rows": len(accepted),
                "coverage": coverage,
                "bad_step_rate": bad,
                "holdout_nonharm_fraction": nonharm,
                "mean_holdout_delta": mean_delta,
                "nonharm_auc": auc,
                "p3_calibration_pass": int(coverage >= 0.05 and bad <= 0.05 and nonharm >= 0.70),
                "loss_type": "CE",
                "label_smoothing": 0,
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "teacher_logits_used": 0,
                "distillation_used": 0,
                "geometry_loss_used": 0,
                "sampler_changed": 0,
                "class_weight_used": 0,
                "uses_loss_backward": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    return out


def _clone_params_and_states(params: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], List[AdamWState]]:
    cloned = [p.detach().clone() for p in params]
    return cloned, [AdamWState.zeros_like(p) for p in cloned]


def _apply_in_place(params: Sequence[torch.Tensor], step: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, d in zip(params, step):
            p.add_(d)


def _run_p4_short(
    args: argparse.Namespace,
    device: torch.device,
    best: Dict[str, Any] | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not best:
        return (
            [_not_run("P4_SHORT_RUN_FUNCTIONAL_SAFETY", "p4_short_run_functional_safety.csv", "no_P3_predictor_survivor")],
            [_not_run("FUNCTIONAL_EVENT_TRACE", "functional_event_trace_v9210.csv", "no_P3_predictor_survivor")],
        )
    horizons = [int(x) for x in _parse_list(args.p4_steps_list)]
    if args.p4_candidates == "best":
        specs = [lq.LQSpec("LQ0-LQ-t2-h256-AdamW", "t2", 256, "default", 1.0)]
    else:
        specs = _candidate_specs(args.p4_candidates)
    modes = [
        ("AdamWOnly", "C0-NoOp", "G1-RoleSNROnly", 0.0),
        ("BestFunctional", str(best.get("direction_id")), str(best.get("gate_id")), _to_float(best.get("step_fraction"))),
        ("GeometryD1Control", "D1-QuadraticCoeffDamping", "G1-RoleSNROnly", _to_float(best.get("step_fraction"))),
        ("RandomMatchedControl", "C2-RandomDirectionMatchedNorm", "G1-RoleSNROnly", _to_float(best.get("step_fraction"))),
    ]
    rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    snr_cfg = snr_lq.SNRConfig(
        microbatch_count=1,
        tau1=float(args.snr_tau1),
        tau2=float(args.snr_tau2),
        temperature=float(args.snr_smooth_temperature),
    )
    for dataset in _canonical_tasks(args.p4_datasets):
        min_train = max(int(args.train_size), int(args.batch_size) * (max(horizons) + 2))
        x_train, y_train, x_test, y_test, input_dim, output_dim, protocol = v92._load_task(
            args,
            dataset,
            train_size=min_train,
            test_size=int(args.p4_test_size),
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        x_test = x_test.to(device=device, dtype=torch.float32)
        y_test = y_test.to(device=device)
        for seed_text in _parse_list(args.p4_seeds):
            seed = int(seed_text)
            for spec in specs:
                init_params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, seed + 92920)
                max_horizon = max(horizons)
                mode_state: Dict[str, Dict[str, Any]] = {}
                for mode_id, direction_id, gate_id, rho in modes:
                    params, states = _clone_params_and_states(init_params)
                    mode_state[mode_id] = {
                        "params": params,
                        "states": states,
                        "scalar_state": snr_lq.ScalarRoleSNRState.zeros(
                            len(snr_lq.role_names_for_basis(spec.basis)),
                            beta=float(args.snr_ema_beta),
                            device=device,
                        ),
                        "direction_id": direction_id,
                        "gate_id": gate_id,
                        "rho": rho,
                        "functional_events": 0,
                        "functional_norm_sum": 0.0,
                        "active_fraction_sum": 0.0,
                        "steps": 0,
                        "elapsed_ms": 0.0,
                        "peak_memory_mb": 0.0,
                    }
                _fwd, bwd = lq.functions_for_basis(spec.basis)
                opt_cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
                for step in range(max_horizon):
                    gen = torch.Generator(device=device).manual_seed(9393000 + seed * 10000 + step)
                    idx = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)[: int(args.batch_size)]
                    xb = x_train[idx]
                    yb = y_train[idx]
                    for mode_id, direction_id, gate_id, rho in modes:
                        state = mode_state[mode_id]
                        params = state["params"]
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        _sync(device)
                        t0 = time.perf_counter()
                        pack = bwd(xb, yb, *params, mu, std, 2.0, 2.0)
                        grads = [g.detach() for g in pack[1:]]
                        snr_rows = snr_lq.scalar_ema_snr_from_grads(
                            grads=grads,
                            scalar_state=state["scalar_state"],
                            basis=spec.basis,
                            snr_cfg=snr_cfg,
                        )
                        active_fraction = _quadratic_active_fraction(snr_rows)
                        quadratic_snr = _quadratic_snr(snr_rows)
                        task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
                        if direction_id == "C0-NoOp":
                            functional_step = snr_lq.zero_like_params(params)
                            event_accepted = 0
                        else:
                            multiplier = _gate_multiplier(gate_id, active_fraction, quadratic_snr, float(args.high_snr_tau))
                            raw_delta = _make_delta(
                                direction_id=direction_id,
                                gate_id=gate_id,
                                params=params,
                                grads=grads,
                                basis=spec.basis,
                                task_step=task_step,
                                step_fraction=float(rho),
                                gate_multiplier=multiplier,
                                device=device,
                                seed=9493000 + seed * 10000 + step,
                            )
                            functional_step, _removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
                            event_accepted = int(snr_lq.step_norm(functional_step).detach().cpu() > 0)
                        v92._adamw_update_foreach_(params, grads, state["states"], opt_cfg)
                        if event_accepted:
                            _apply_in_place(params, functional_step)
                        _sync(device)
                        elapsed = (time.perf_counter() - t0) * 1000.0
                        peak_mb = 0.0
                        if device.type == "cuda":
                            peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                        state["elapsed_ms"] += elapsed
                        state["peak_memory_mb"] = max(float(state["peak_memory_mb"]), peak_mb)
                        state["functional_events"] += int(event_accepted)
                        state["functional_norm_sum"] += float(snr_lq.step_norm(functional_step).detach().cpu())
                        state["active_fraction_sum"] += float(active_fraction)
                        state["steps"] += 1
                    if (step + 1) in horizons:
                        base_state = mode_state["AdamWOnly"]
                        base_params = base_state["params"]
                        base_eval = _eval_lq_metrics(base_params, mu, std, spec.basis, x_test, y_test)
                        base_mech = fp_lq.mechanism_metrics(base_params, mu, std, spec.basis, x_test[: min(256, int(x_test.shape[0]))])
                        base_step_ms = float(base_state["elapsed_ms"]) / max(1, int(base_state["steps"]))
                        base_mem = max(1.0e-12, float(base_state["peak_memory_mb"]))
                        for mode_id, direction_id, gate_id, rho in modes:
                            state = mode_state[mode_id]
                            params = state["params"]
                            metrics = _eval_lq_metrics(params, mu, std, spec.basis, x_test, y_test)
                            mech = fp_lq.mechanism_metrics(params, mu, std, spec.basis, x_test[: min(256, int(x_test.shape[0]))])
                            avg_step_ms = float(state["elapsed_ms"]) / max(1, int(state["steps"]))
                            step_ratio = avg_step_ms / max(1.0e-12, base_step_ms)
                            mem_ratio = float(state["peak_memory_mb"]) / base_mem if device.type == "cuda" else 1.0
                            acc_delta = metrics["acc"] - base_eval["acc"]
                            ce_delta = metrics["CE_p99"] - base_eval["CE_p99"]
                            margin_delta = metrics["correct_margin_p10"] - base_eval["correct_margin_p10"]
                            curv_delta = mech["curvature_proxy"] - base_mech["curvature_proxy"]
                            task_safe = int(acc_delta >= -0.005)
                            mechanism_pass = int(ce_delta < 0.0 or margin_delta > 0.0 or curv_delta < 0.0)
                            rows.append(
                                {
                                    "stage": "P4_SHORT_RUN_FUNCTIONAL_SAFETY",
                                    "candidate_id": spec.candidate_id,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "protocol": protocol,
                                    "horizon_steps": step + 1,
                                    "mode_id": mode_id,
                                    "direction_id": direction_id,
                                    "gate_id": gate_id,
                                    "step_fraction": rho,
                                    "acc": metrics["acc"],
                                    "adamw_acc": base_eval["acc"],
                                    "acc_delta_vs_adamw": acc_delta,
                                    "loss": metrics["loss"],
                                    "adamw_loss": base_eval["loss"],
                                    "loss_delta_vs_adamw": metrics["loss"] - base_eval["loss"],
                                    "CEp99": metrics["CE_p99"],
                                    "adamw_CEp99": base_eval["CE_p99"],
                                    "CEp99_delta_vs_adamw": ce_delta,
                                    "margin_p10": metrics["correct_margin_p10"],
                                    "adamw_margin_p10": base_eval["correct_margin_p10"],
                                    "margin_p10_delta_vs_adamw": margin_delta,
                                    "curvature_proxy": mech["curvature_proxy"],
                                    "adamw_curvature_proxy": base_mech["curvature_proxy"],
                                    "curvature_delta_vs_adamw": curv_delta,
                                    "avg_step_ms": avg_step_ms,
                                    "step_time_ratio_vs_adamw": step_ratio,
                                    "peak_memory_mb": float(state["peak_memory_mb"]),
                                    "memory_ratio_vs_adamw": mem_ratio,
                                    "functional_events": int(state["functional_events"]),
                                    "event_coverage": int(state["functional_events"]) / max(1, int(state["steps"])),
                                    "mean_functional_step_norm": float(state["functional_norm_sum"]) / max(1, int(state["steps"])),
                                    "mean_active_fraction": float(state["active_fraction_sum"]) / max(1, int(state["steps"])),
                                    "task_safe_pass": task_safe,
                                    "mechanism_pass": mechanism_pass,
                                    "system_pass": int(step_ratio <= 1.20 and mem_ratio <= 1.05),
                                    "p4_row_pass": int(mode_id == "BestFunctional" and task_safe and mechanism_pass and step_ratio <= 1.20 and mem_ratio <= 1.05),
                                    "loss_type": "CE",
                                    "label_smoothing": 0,
                                    "external_teacher_used": 0,
                                    "self_teacher_used": 0,
                                    "teacher_logits_used": 0,
                                    "distillation_used": 0,
                                    "geometry_loss_used": 0,
                                    "sampler_changed": 0,
                                    "class_weight_used": 0,
                                    "uses_loss_backward": 0,
                                    "functional_update_used": int(mode_id != "AdamWOnly"),
                                    "fake_data_used": 0,
                                    "proxy_row_used": 0,
                                    "cpu_offload_used": 0,
                                }
                            )
                            trace_rows.append(
                                {
                                    "stage": "FUNCTIONAL_EVENT_TRACE",
                                    "candidate_id": spec.candidate_id,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "horizon_steps": step + 1,
                                    "mode_id": mode_id,
                                    "functional_events": int(state["functional_events"]),
                                    "event_coverage": int(state["functional_events"]) / max(1, int(state["steps"])),
                                    "mean_functional_step_norm": float(state["functional_norm_sum"]) / max(1, int(state["steps"])),
                                    "mean_active_fraction": float(state["active_fraction_sum"]) / max(1, int(state["steps"])),
                                    "fake_data_used": 0,
                                    "proxy_row_used": 0,
                                    "cpu_offload_used": 0,
                                }
                            )
    return rows, trace_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--previous-v929-dir", default=str(PREV_V929.relative_to(ROOT)))
    parser.add_argument("--candidates", default="LQ0,LQ1")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--snr-warmup-steps", type=int, default=36)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--snr-ema-beta", type=float, default=0.97)
    parser.add_argument("--snr-tau1", type=float, default=20.0)
    parser.add_argument("--snr-tau2", type=float, default=40.0)
    parser.add_argument("--snr-smooth-temperature", type=float, default=1.0)
    parser.add_argument("--high-snr-tau", type=float, default=80.0)
    parser.add_argument("--base-snr-overhead-ratio", type=float, default=0.126)
    parser.add_argument("--directions", default="D1-QuadraticCoeffDamping,D2-QuadraticCoeffSignFlip,D3-LiftDamping,D4-OutputLinearDamping,D5-AllOutputCoeffDamping,D6-AllParamNormDamping,D7-AdamWResidualDiagnostic,D8-FisherDiagQuadraticDamping,D9-SignalChannelProjection,C0-NoOp,C2-RandomDirectionMatchedNorm")
    parser.add_argument("--gates", default="G1-RoleSNROnly,G2-HighConfidenceRoleSNR,G4-HoldoutPrecheckGate")
    parser.add_argument("--step-fractions", default="0.01,0.03,0.10")
    parser.add_argument("--run-p3-p4", action="store_true")
    parser.add_argument("--p4-candidates", default="best")
    parser.add_argument("--p4-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p4-seeds", default="0,1,2")
    parser.add_argument("--p4-steps-list", default="50,240")
    parser.add_argument("--p4-test-size", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device_from_arg(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    prev_dir = Path(args.previous_v929_dir)
    if not prev_dir.is_absolute():
        prev_dir = ROOT / prev_dir
    prev_route = _read_json(prev_dir / "route_decision.json")

    write_json(
        out_dir / "run_manifest.json",
        {
            "created_utc": _now_iso(),
            "script": str(SCRIPT_PATH.relative_to(ROOT)),
            "plan": str(PLAN_PATH.relative_to(ROOT)),
            "device": str(device),
            "torch": torch.__version__,
            "args": vars(args),
            "source_artifacts": {"v929": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir)},
            "contract": {
                "loss_type": "CE",
                "label_smoothing": 0,
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "teacher_logits_used": 0,
                "distillation_used": 0,
                "geometry_loss_used": 0,
                "sampler_changed": 0,
                "class_weight_used": 0,
                "cpu_offload_used": 0,
                "uses_loss_backward": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            },
        },
    )

    base_gate = int(_to_int(prev_route.get("success_v929_p1_snr_gate", 0)) == 1 and _to_int(prev_route.get("p4_pass", 0)) == 1 and _to_int(prev_route.get("p5_near_pass", 0)) == 1)
    contract_rows = [
        {
            "stage": "P0_CONTRACT",
            "source_artifact": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir),
            "v929_route": prev_route.get("route", ""),
            "p4_pass": _to_int(prev_route.get("p4_pass", 0)),
            "p5_near_pass": _to_int(prev_route.get("p5_near_pass", 0)),
            "p1_snr_pass": _to_int(prev_route.get("success_v929_p1_snr_gate", 0)),
            "base_gate_pass": base_gate,
            "loss_type": "CE",
            "label_smoothing": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "teacher_logits_used": 0,
            "distillation_used": 0,
            "geometry_loss_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "uses_loss_backward": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    write_csv_rows(out_dir / "contract_audit_v9210.csv", contract_rows)

    if base_gate:
        p2_rows = _run_one_step_factory(args, device)
    else:
        p2_rows = [_not_run("P2_DIRECTION_FACTORY_ONE_STEP", "p2_direction_factory_one_step.csv", "P0_base_gate_failed")]
    write_csv_rows(out_dir / "p2_direction_factory_one_step.csv", p2_rows)

    measured = [r for r in p2_rows if str(r.get("status", "")) != "not_run"]
    summary_keys = ["direction_id", "gate_id", "step_fraction"]
    group_rows = _summarize_groups(measured, summary_keys) if measured else []
    write_csv_rows(out_dir / "p1_p2_failure_attribution.csv", _summarize_groups(measured, ["direction_id", "gate_id", "step_fraction", "dataset"]) if measured else [_not_run("P1_P2_FAILURE_ATTRIBUTION", "p1_p2_failure_attribution.csv", "P2_not_measured")])
    write_csv_rows(out_dir / "p0_v929_p2_reproduction.csv", [r for r in group_rows if r.get("direction_id") == "D1-QuadraticCoeffDamping" and r.get("gate_id") == "G1-RoleSNROnly" and abs(_to_float(r.get("step_fraction")) - 0.10) < 1.0e-12] or [_not_run("P0_V929_P2_REPRODUCTION", "p0_v929_p2_reproduction.csv", "reference_group_missing")])
    write_csv_rows(out_dir / "p2_direction_factory_summary.csv", group_rows if group_rows else [_not_run("P2_DIRECTION_FACTORY_SUMMARY", "p2_direction_factory_summary.csv", "P2_not_measured")])

    survivors = [r for r in group_rows if _to_int(r.get("p2_survivor")) == 1]
    non_holdout_survivors = [r for r in survivors if str(r.get("gate_id")) != "G4-HoldoutPrecheckGate"]
    holdout_safety = [r for r in group_rows if str(r.get("gate_id")) == "G4-HoldoutPrecheckGate" and _to_int(r.get("p2_safety_pass")) == 1 and _to_int(r.get("p2_coverage_pass")) == 1]
    best = sorted(survivors, key=lambda r: (_to_float(r.get("amortized_overhead_ratio")), -_to_float(r.get("functional_event_coverage"))))[0] if survivors else None
    ref = [r for r in group_rows if r.get("direction_id") == "D1-QuadraticCoeffDamping" and r.get("gate_id") == "G1-RoleSNROnly" and abs(_to_float(r.get("step_fraction")) - 0.10) < 1.0e-12]
    ref_row = ref[0] if ref else {}
    p0_repeat_pass = int(
        bool(ref_row)
        and abs(_to_float(ref_row.get("bad_step_rate_accepted")) - 0.819444) <= 0.10
        and abs(_to_float(ref_row.get("prediction_corr_all")) - 0.231090) <= 0.10
    )
    valid_p2 = int(bool(base_gate and p0_repeat_pass))
    p3_rows: List[Dict[str, Any]]
    p4_rows: List[Dict[str, Any]]
    event_trace_rows: List[Dict[str, Any]]
    p3_pass = 0
    p4_pass = 0
    p4_best_rows: List[Dict[str, Any]] = []
    if valid_p2 and non_holdout_survivors and args.run_p3_p4:
        p3_rows = _run_p3_predictor(measured, best)
        p3_pass = int(any(_to_int(r.get("p3_calibration_pass")) == 1 for r in p3_rows))
        if p3_pass:
            p4_rows, event_trace_rows = _run_p4_short(args, device, best)
            p4_best_rows = [r for r in p4_rows if str(r.get("mode_id")) == "BestFunctional"]
            p4_pass = int(bool(p4_best_rows) and all(_to_int(r.get("p4_row_pass")) == 1 for r in p4_best_rows))
        else:
            p4_rows = [_not_run("P4_SHORT_RUN_FUNCTIONAL_SAFETY", "p4_short_run_functional_safety.csv", "P3_calibrated_predictor_failed")]
            event_trace_rows = [_not_run("FUNCTIONAL_EVENT_TRACE", "functional_event_trace_v9210.csv", "P3_calibrated_predictor_failed")]
    elif valid_p2 and non_holdout_survivors:
        p3_rows = [_not_run("P3_CALIBRATED_FUNCTIONAL_PREDICTOR", "p3_calibrated_functional_predictor.csv", "P2_passed_but_P3_P4_not_requested")]
        p4_rows = [_not_run("P4_SHORT_RUN_FUNCTIONAL_SAFETY", "p4_short_run_functional_safety.csv", "P2_passed_but_P3_P4_not_requested")]
        event_trace_rows = [_not_run("FUNCTIONAL_EVENT_TRACE", "functional_event_trace_v9210.csv", "P2_passed_but_P3_P4_not_requested")]
    else:
        reason = "P2_one_step_survivor_not_available"
        p3_rows = [_not_run("P3_CALIBRATED_FUNCTIONAL_PREDICTOR", "p3_calibrated_functional_predictor.csv", reason)]
        p4_rows = [_not_run("P4_SHORT_RUN_FUNCTIONAL_SAFETY", "p4_short_run_functional_safety.csv", reason)]
        event_trace_rows = [_not_run("FUNCTIONAL_EVENT_TRACE", "functional_event_trace_v9210.csv", reason)]
    write_csv_rows(out_dir / "p3_calibrated_functional_predictor.csv", p3_rows)
    write_csv_rows(out_dir / "p4_short_run_functional_safety.csv", p4_rows)
    write_csv_rows(out_dir / "functional_event_trace_v9210.csv", event_trace_rows)

    if not base_gate:
        route = "R5-FunctionalUnsafe"
        primary_blocker = "P0_base_gate_failed"
        next_required = "restore_v929_base_gate"
        failure_code = "F2_base_gate_regression"
    elif not p0_repeat_pass:
        route = "R5-FunctionalUnsafe"
        primary_blocker = "v929_P2_failure_reproduction_unstable"
        next_required = "repair_measurement_before_direction_factory"
        failure_code = "F3_p2_reproduction_unstable"
    elif non_holdout_survivors:
        if args.run_p3_p4 and p3_pass and p4_pass:
            route = "R2-FunctionalShortRunSafe"
            primary_blocker = "P4_passed_but_P5_P7_not_executed_in_this_runner"
            next_required = "open_P5_full_functional_reentry_10seed"
            failure_code = "F18_artifact_missing"
        elif args.run_p3_p4 and p3_pass and not p4_pass:
            route = "R5-FunctionalUnsafe"
            primary_blocker = "P4_short_run_functional_safety_failed"
            next_required = "redesign_direction_or_strength_before_full_reentry"
            failure_code = "F5_short_run_safety_fail"
        elif args.run_p3_p4 and not p3_pass:
            route = "R6-SNRGateTooCoarse"
            primary_blocker = "P3_calibrated_predictor_failed"
            next_required = "repair_predictor_or_role_gate_before_short_run"
            failure_code = "F8_predictor_calibration_fail"
        else:
            route = "R1-FunctionalPredictorRepaired"
            primary_blocker = "P2_passed_but_P3_P7_not_executed_in_this_runner"
            next_required = "open_P3_calibrated_predictor_and_P4_short_run"
            failure_code = "F18_artifact_missing"
    elif survivors and best and str(best.get("gate_id")) == "G4-HoldoutPrecheckGate":
        route = "R8-HoldoutGuardRequired"
        primary_blocker = "only_exchangeability_holdout_precheck_survived_P2"
        next_required = "turn_holdout_guard_into_low_overhead_predictor_before_short_run"
        failure_code = "F8_predictor_calibration_fail"
    elif holdout_safety:
        route = "R8-HoldoutGuardRequired"
        primary_blocker = "holdout_precheck_improves_safety_but_no_mechanism_or_system_survivor"
        next_required = "repair_direction_mechanism_or_holdout_guard_overhead"
        failure_code = "F8_predictor_calibration_fail"
    else:
        route = "R7-DirectionWrong"
        primary_blocker = "all_tested_functional_directions_failed_one_step_safety"
        next_required = "redesign_population_risk_aligned_functional_direction"
        failure_code = "F4_direction_failure"

    downstream_reason = "P2_one_step_survivor_not_available"
    if non_holdout_survivors and not args.run_p3_p4:
        downstream_reason = "P2_passed_but_P3_to_P7_not_implemented_in_this_terminal_run"
    elif p4_pass:
        downstream_reason = "P4_short_run_passed_but_full_reentry_not_executed_in_this_runner"
    elif args.run_p3_p4 and p3_pass and not p4_pass:
        downstream_reason = "P4_short_run_functional_safety_failed"
    elif args.run_p3_p4 and not p3_pass:
        downstream_reason = "P3_calibrated_predictor_failed"
    elif route == "R8-HoldoutGuardRequired":
        downstream_reason = "only_holdout_guard_route_available_so_predictor_training_not_opened"
    downstream_specs = [
        ("P5_FULL_FUNCTIONAL_REENTRY_10SEED", "p5_full_functional_reentry_10seed.csv"),
        ("P6_NOISE_ROBUSTNESS_DIAGNOSTIC", "p6_noise_robustness_diagnostic.csv"),
        ("P7_STRONG_BASELINE_EXTERNAL_READY", "p7_strong_baseline_external_ready.csv"),
    ]
    for stage, name in downstream_specs:
        write_csv_rows(out_dir / name, [_not_run(stage, name, downstream_reason)])

    _write_svg(out_dir / "figures" / "p0_p2_repeat_dashboard.svg", "P0 v9.2.9 P2 Repeat", f"repeat_pass={p0_repeat_pass}, bad={_to_float(ref_row.get('bad_step_rate_accepted')):.4f}, corr={_to_float(ref_row.get('prediction_corr_all')):.4f}")
    _write_svg(out_dir / "figures" / "p1_failure_factor_heatmap.svg", "P1 Failure Attribution", f"groups={len(group_rows)}, survivors={len(survivors)}")
    _write_svg(out_dir / "figures" / "p2_direction_factory_pareto.svg", "P2 Direction Factory Pareto", f"route={route}, best={best.get('direction_id','') if best else 'none'}")
    _write_svg(out_dir / "figures" / "p2_safety_vs_coverage.svg", "P2 Safety vs Coverage", f"holdout_safety_groups={len(holdout_safety)}")

    failure_rows = [
        {
            "stage": "P2",
            "failure_code": failure_code,
            "route": route,
            "reason": primary_blocker,
            "reference_bad_step_rate": _to_float(ref_row.get("bad_step_rate_accepted")),
            "reference_prediction_corr": _to_float(ref_row.get("prediction_corr_all")),
            "survivor_count": len(survivors),
            "non_holdout_survivor_count": len(non_holdout_survivors),
            "holdout_safety_group_count": len(holdout_safety),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    route_json = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "best_functional_candidate": best.get("direction_id", "") if (valid_p2 and best) else "",
        "best_direction": best.get("direction_id", "") if (valid_p2 and best) else "",
        "best_gate": best.get("gate_id", "") if (valid_p2 and best) else "",
        "best_step_fraction": best.get("step_fraction", "") if (valid_p2 and best) else "",
        "best_predictor": "PRED0-AllAcceptP2Survivor" if p3_pass else "none",
        "p0_repeat_pass": p0_repeat_pass,
        "p2_group_count": len(group_rows),
        "p2_survivor_count": len(survivors),
        "p2_non_holdout_survivor_count": len(non_holdout_survivors),
        "p2_holdout_safety_group_count": len(holdout_safety),
        "p2_reference_bad_step_rate": _to_float(ref_row.get("bad_step_rate_accepted")),
        "p2_reference_prediction_corr": _to_float(ref_row.get("prediction_corr_all")),
        "p2_safety_pass": int(bool(valid_p2 and survivors)),
        "p3_calibration_pass": p3_pass,
        "p4_short_run_pass": p4_pass,
        "p4_short_run_row_count": len([r for r in p4_rows if str(r.get("status", "")) != "not_run"]),
        "p5_full_reentry_pass": 0,
        "functional_task_safe": int(bool(valid_p2 and non_holdout_survivors)),
        "functional_geometry_pass": int(bool(valid_p2 and best and _to_int(best.get("mechanism_benefit_pass")) == 1)),
        "functional_calibration_pass": p3_pass,
        "functional_kmnist_repair_pass": 0,
        "functional_control_pass": int(bool(p4_pass)),
        "functional_system_pass": int(bool(valid_p2 and best and _to_int(best.get("p2_system_pass")) == 1)),
        "noise_robustness_pass": 0,
        "quadratic_baseline_challenge_pass": 0,
        "external_fair_ready": 0,
        "primary_blocker": primary_blocker,
        "next_required_implementation": next_required,
        "success_v9210_functional_predictor": int(bool(valid_p2 and non_holdout_survivors)),
        "success_v9210_functional_advantage": 0,
        "success_v9210_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)

    audit_targets = [
        out_dir / "contract_audit_v9210.csv",
        out_dir / "p0_v929_p2_reproduction.csv",
        out_dir / "p1_p2_failure_attribution.csv",
        out_dir / "p2_direction_factory_one_step.csv",
        out_dir / "p2_direction_factory_summary.csv",
        out_dir / "p3_calibrated_functional_predictor.csv",
        out_dir / "p4_short_run_functional_safety.csv",
        out_dir / "p5_full_functional_reentry_10seed.csv",
        out_dir / "p6_noise_robustness_diagnostic.csv",
        out_dir / "p7_strong_baseline_external_ready.csv",
        out_dir / "functional_event_trace_v9210.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_targets)
    write_csv_rows(out_dir / "v9210_provenance_audit.csv", [{"stage": "NO_FAKE_AUDIT", "route": route, **audit, "fake_data_used": int(audit["fake_data_used"]), "proxy_row_used": int(audit["proxy_row_used"]), "cpu_offload_used": int(audit["cpu_offload_used"])}])
    route_json.update({"no_fake": bool(audit["no_fake"]), "no_proxy": bool(audit["no_proxy"]), "rows_checked": int(audit["rows_checked"])})
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)
    hash_rows = artifact_hash_rows(
        [
            PLAN_PATH,
            SCRIPT_PATH,
            ROOT / "dgkan" / "functional" / "lq_functional_predictor.py",
            ROOT / "dgkan" / "functional" / "snr_gated_lq.py",
            ROOT / "dgkan" / "models" / "fc_purekan_lq.py",
            out_dir / "route_decision.json",
            *audit_targets,
            out_dir / "v9210_provenance_audit.csv",
        ],
        root=ROOT,
    )
    write_csv_rows(out_dir / "artifact_hashes.csv", hash_rows)
    print(json.dumps(route_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
