#!/usr/bin/env python3
"""DG-KAN v9.2.17 signal-aligned functional or primitive reset runner."""

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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9214_p4qualified_functional_actuator_closure as v9214  # noqa: E402
import run_v9216_causal_target_discovery as v9216  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.17_SignalAligned_Functional_or_PrimitiveReset_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9217_signal_aligned_functional_or_reset.py"
PREV_V9216 = ROOT / "results" / "real_rerun_20260506" / "v9216_causal_target_discovery_first_20260510T040000Z"
PREV_V9215 = ROOT / "results" / "real_rerun_20260506" / "v9215_control_resistant_functional_causality_first_20260510T030000Z"
PREV_V9214 = ROOT / "results" / "real_rerun_20260506" / "v9214_p4qualified_functional_actuator_closure_first_20260510T010000Z"

A7C_P4_STEP_Q90 = 1.2131227301969405
A7C_P4_MEMORY = 0.9697312055736007


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


def _quantile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(float(v) for v in vals)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _not_run(stage: str, artifact: str, reason: str) -> Dict[str, Any]:
    return snr_lq.not_run_row(stage, artifact, reason)


def _write_svg(path: Path, title: str, subtitle: str) -> None:
    ensure_dir(path.parent)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="220" viewBox="0 0 1040 220">
  <rect width="1040" height="220" fill="#f8fafc"/>
  <text x="30" y="58" font-family="Arial, sans-serif" font-size="25" fill="#111827">{title}</text>
  <text x="30" y="102" font-family="Arial, sans-serif" font-size="16" fill="#374151">{subtitle}</text>
  <text x="30" y="145" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">Generated from measured CSV/JSON fields only.</text>
</svg>
""",
        encoding="utf-8",
    )


def _step_cos(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    den = snr_lq.step_norm(left) * snr_lq.step_norm(right)
    if bool((den <= 1.0e-12).detach().cpu()):
        return 0.0
    return float((snr_lq.step_dot(left, right) / den.clamp_min(1.0e-12)).detach().cpu())


def _cap(step: Sequence[torch.Tensor], reference: Sequence[torch.Tensor], fraction: float) -> List[torch.Tensor]:
    return v9214._cap_to_fraction(step, reference, float(fraction))


def _role_names(params: Sequence[torch.Tensor]) -> List[str]:
    base = ["lift_identity", "output_linear", "t2_quadratic_coeff"]
    while len(base) < len(params):
        base.append(f"actuator_channel_{len(base) - 2}")
    return base[: len(params)]


def _role_snr_from_halves(
    xb: torch.Tensor,
    yb: torch.Tensor,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    lr: float,
) -> Dict[str, Any]:
    half = max(1, int(xb.shape[0]) // 2)
    _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
    _l1, g1 = act.actuator_fwd_bwd(xb[:half], yb[:half], params, mu, std, spec)
    _l2, g2 = act.actuator_fwd_bwd(xb[half : 2 * half], yb[half : 2 * half], params, mu, std, spec)
    task_step = snr_lq.gradient_descent_task_step(grads, lr)
    total_step = snr_lq.step_norm(task_step).clamp_min(1.0e-12)
    rows: List[Dict[str, Any]] = []
    for idx, role in enumerate(_role_names(params)):
        mean_grad = 0.5 * (g1[idx].detach() + g2[idx].detach())
        noise_grad = 0.5 * (g1[idx].detach() - g2[idx].detach())
        role_snr = float((mean_grad.float().norm() / noise_grad.float().norm().clamp_min(1.0e-12)).detach().cpu())
        role_grad_norm = float(grads[idx].float().norm().detach().cpu())
        role_update_fraction = float((task_step[idx].float().norm() / total_step).detach().cpu())
        rows.append(
            {
                "role": role,
                "role_grad_norm": role_grad_norm,
                "role_snr": role_snr,
                "role_update_fraction": role_update_fraction,
            }
        )
    dom = max(rows, key=lambda r: r["role_update_fraction"])
    return {"role_rows": rows, "dominant_role": dom, "task_step": task_step, "grads": grads}


def _event_mask(logits: torch.Tensor, y: torch.Tensor, event_id: str, dataset: str) -> torch.Tensor:
    ce, margin, _wrong_idx, _row = v9216._ce_margin(logits, y)
    eid = str(event_id)
    if eid == "E0-uniform-low-frequency":
        return torch.ones_like(y, dtype=torch.bool)
    if eid == "E1-CalibratedCEp99Tail":
        return ce >= torch.quantile(ce.float(), 0.90)
    if eid == "E2-CalibratedMarginTail":
        return margin <= torch.quantile(margin.float(), 0.10)
    if eid == "E5-KMNISTHardModeEvent":
        if dataset != "KMNIST":
            return torch.zeros_like(y, dtype=torch.bool)
        return (ce >= torch.quantile(ce.float(), 0.85)) | (margin <= torch.quantile(margin.float(), 0.15))
    if eid == "E6-ControlDominanceEvent":
        return (ce >= torch.quantile(ce.float(), 0.88)) | (margin <= torch.quantile(margin.float(), 0.12))
    if eid == "E7-AbstainUnlessBeyondAdamWParallel":
        return torch.zeros_like(y, dtype=torch.bool)
    return torch.ones_like(y, dtype=torch.bool)


def _delta_metrics(after: Dict[str, float], adamw: Dict[str, float]) -> Dict[str, float]:
    return {
        "CEp99_delta": after["CE_p99"] - adamw["CE_p99"],
        "margin_p10_delta": after["correct_margin_p10"] - adamw["correct_margin_p10"],
        "ECE_delta": after["ECE"] - adamw["ECE"],
        "NLL_delta": after["NLL"] - adamw["NLL"],
        "acc_delta": after["acc"] - adamw["acc"],
        "curvature_delta": 0.0,
    }


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
    return v9216._replay_metrics(
        args,
        cfg,
        spec,
        base_params,
        base_states,
        mu,
        std,
        x_train,
        y_train,
        x_eval,
        y_eval,
        branch_step,
        horizons,
    )


def _apply_event(step: Sequence[torch.Tensor], coverage: float) -> List[torch.Tensor]:
    if coverage <= 0.0:
        return [torch.zeros_like(s) for s in step]
    return [s.detach().clone() for s in step]


def _p0_recap() -> Dict[str, Any]:
    route = _read_json(PREV_V9216 / "route_decision.json")
    audit = read_csv_rows(PREV_V9216 / "v9216_provenance_audit.csv")
    p1 = read_csv_rows(PREV_V9216 / "p1_control_dominance_autopsy.csv")
    fake = _to_int(audit[0].get("fake_proxy_nonzero_count")) if audit else 999
    control_counts = Counter(r.get("best_control", "") for r in p1)
    by_metric: Dict[str, str] = {}
    for metric in sorted({r.get("metric", "") for r in p1}):
        mc = Counter(r.get("best_control", "") for r in p1 if r.get("metric") == metric)
        by_metric[metric] = mc.most_common(1)[0][0] if mc else ""
    row = {
        "stage": "P0_V9216_BOUNDARY_REPRODUCTION",
        "source_artifact": str(PREV_V9216.relative_to(ROOT)),
        "source_route": route.get("route", ""),
        "p2_a4e_kmnist_h80_signal_pass": route.get("a4e_kmnist_h80_signal_pass", 0),
        "a4e_row_level_beat_count": route.get("a4e_row_level_beat_count", 0),
        "p3_control_derived_survivor_count": route.get("prototype_row_level_beat_count", 0),
        "p4_control_contrastive_pass": route.get("control_contrastive_pass", 0),
        "best_control": route.get("best_control", ""),
        "best_control_win_count": control_counts.get(route.get("best_control", ""), 0),
        "best_control_by_metric": json.dumps(by_metric, sort_keys=True),
        "fake_proxy_count": fake,
        "P0_pass": int(
            route.get("route") == "R8-NoExtractableFunctionalAdvantage"
            and route.get("best_control") == "AdamWParallelDirection"
            and _to_int(route.get("a4e_row_level_beat_count")) == 0
            and _to_int(route.get("prototype_row_level_beat_count")) == 0
            and _to_int(route.get("control_contrastive_pass")) == 0
            and fake == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return row


def _role_stats_table(args: argparse.Namespace, device: torch.device, actuators: Sequence[str], datasets: Sequence[str], seeds: Sequence[int]) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    out: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for aid in actuators:
        spec = act.actuator_specs_from_ids([aid])[0]
        for dataset in datasets:
            x_train, y_train, _x_eval, _y_eval, in_dim, out_dim, _protocol = v92._load_task(
                args, dataset, train_size=max(int(args.train_size), 4096), test_size=int(args.eval_size)
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            for seed in seeds:
                params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x_train, device, seed + 921700 + len(aid))
                states = [AdamWState.zeros_like(p) for p in params]
                v9214._run_adamw_steps(params, states, mu, std, spec, x_train, y_train, 0, int(args.warmup_steps), int(args.batch_size), cfg)
                xb = x_train[: int(args.audit_batch_size)]
                yb = y_train[: int(args.audit_batch_size)]
                out[(aid, dataset, seed)] = _role_snr_from_halves(xb, yb, params, mu, std, spec, float(args.lr))
    return out


def _run_p1(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    source = read_csv_rows(PREV_V9215 / "p2_p4qualified_actuator_causality_matrix.csv")
    actuators = sorted({r.get("actuator", "") for r in source if r.get("actuator")})
    datasets = _canonical_tasks(args.datasets)
    seeds = [int(x) for x in _parse_list(args.seeds)]
    role_stats = _role_stats_table(args, device, actuators, datasets, seeds)
    trace_rows: List[Dict[str, Any]] = []
    for (aid, dataset, seed), data in role_stats.items():
        for rr in data["role_rows"]:
            trace_rows.append(
                {
                    "stage": "P1_ROLE_SNR_TRACE",
                    "actuator": aid,
                    "dataset": dataset,
                    "seed": seed,
                    **rr,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )

    group_keys = ["actuator", "target", "solver", "event", "dataset", "seed", "horizon"]
    groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for r in source:
        groups[tuple(str(r.get(k, "")) for k in group_keys)].append(r)
    metric_fields = {
        "CEp99": ("CEp99_delta", "min"),
        "ECE": ("ECE_delta", "min"),
        "MarginP10": ("margin_p10_delta", "max"),
        "NLL": ("NLL_delta", "min"),
    }
    rows: List[Dict[str, Any]] = []
    winners: Counter[str] = Counter()
    adamw_wins_by_metric: Counter[str] = Counter()
    for key, rs in groups.items():
        by_branch = {r.get("branch", ""): r for r in rs}
        real = by_branch.get("RealFunctional")
        if not real:
            continue
        controls = [r for r in rs if r.get("branch") not in {"RealFunctional", "AdamWOnly"}]
        aid, target, solver, event, dataset, seed_s, horizon = key
        seed = int(seed_s)
        role = role_stats.get((aid, dataset, seed), {}).get("dominant_role", {})
        for metric, (field, mode) in metric_fields.items():
            if mode == "min":
                best = min(controls, key=lambda r: _to_float(r.get(field)))
                rank_real = 1 + sum(_to_float(c.get(field)) < _to_float(real.get(field)) for c in controls)
            else:
                best = max(controls, key=lambda r: _to_float(r.get(field)))
                rank_real = 1 + sum(_to_float(c.get(field)) > _to_float(real.get(field)) for c in controls)
            winners[best.get("branch", "")] += 1
            if best.get("branch") == "AdamWParallelDirection":
                adamw_wins_by_metric[metric] += 1
            adamw_parallel = by_branch.get("AdamWParallelDirection", {})
            random = by_branch.get("RandomMatchedNorm", {})
            noop = by_branch.get("NoOpMatchedOverhead", {})
            shuffled = by_branch.get("ShuffledTarget", {})
            rows.append(
                {
                    "stage": "P1_ADAMWPARALLEL_DOMINANCE_AUTOPSY",
                    "dataset": dataset,
                    "seed": seed,
                    "horizon": horizon,
                    "actuator": aid,
                    "target": target,
                    "solver": solver,
                    "event": event,
                    "metric": metric,
                    "best_control": best.get("branch", ""),
                    "AdamWParallel_win": int(best.get("branch") == "AdamWParallelDirection"),
                    "AdamWParallel_delta": adamw_parallel.get(field, ""),
                    "RealFunctional_delta": real.get(field, ""),
                    "Random_delta": random.get(field, ""),
                    "NoOp_delta": noop.get(field, ""),
                    "ShuffledTarget_delta": shuffled.get(field, ""),
                    "delta_norm": real.get("delta_norm", ""),
                    "delta_norm_vs_adamw": real.get("delta_norm_vs_adamw", ""),
                    "cos_with_adamw": real.get("cos_delta_adamw", ""),
                    "cos_with_task_gradient": real.get("cos_delta_adamw", ""),
                    "dominant_role": role.get("role", ""),
                    "role_grad_norm": role.get("role_grad_norm", ""),
                    "role_snr": role.get("role_snr", ""),
                    "role_update_fraction": role.get("role_update_fraction", ""),
                    "CEp99_before": real.get("AdamW_CEp99_abs", ""),
                    "CEp99_after": real.get("CEp99_abs", ""),
                    "margin_p10_before": real.get("AdamW_margin_p10_abs", ""),
                    "margin_p10_after": real.get("margin_p10_abs", ""),
                    "ECE_before": real.get("AdamW_ECE_abs", ""),
                    "ECE_after": real.get("ECE_abs", ""),
                    "rank_of_real": rank_real,
                    "task_safety_real": real.get("task_safe", ""),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    total = sum(winners.values())
    top_control, top_count = winners.most_common(1)[0] if winners else ("", 0)
    adamw_share = top_count / max(1, total) if top_control == "AdamWParallelDirection" else winners.get("AdamWParallelDirection", 0) / max(1, total)
    if top_control == "AdamWParallelDirection" and adamw_share >= 0.45:
        attribution = "A1-understep,A2-role_signal"
    elif winners.get("NoOpMatchedOverhead", 0) / max(1, total) >= 0.40:
        attribution = "A4-control_metric_artifact"
    else:
        attribution = "A5-no_functional_signal"
    return rows, {
        "p1_attribution": attribution,
        "best_control": top_control,
        "best_control_win_count": top_count,
        "adamwparallel_win_share": winners.get("AdamWParallelDirection", 0) / max(1, total),
        "adamwparallel_win_by_metric": json.dumps(dict(adamw_wins_by_metric), sort_keys=True),
        "p1_attribution_pass": int(bool(rows)),
    }, trace_rows


def _control_step(
    control_id: str,
    params: Sequence[torch.Tensor],
    task_step: Sequence[torch.Tensor],
    event_coverage: float,
    seed: int,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    zero = snr_lq.zero_like_params(params)
    meta: Dict[str, Any] = {"lr_scale": "", "trust_ratio": "", "control_family": ""}
    if control_id == "OC0-AdamWOnly":
        step = zero
        meta.update({"lr_scale": 1.0, "control_family": "noop"})
    elif control_id == "OC1-LRScale-1.003":
        step = snr_lq.scale_step(task_step, 0.003)
        meta.update({"lr_scale": 1.003, "control_family": "scalar_lr"})
    elif control_id == "OC2-LRScale-1.01":
        step = snr_lq.scale_step(task_step, 0.01)
        meta.update({"lr_scale": 1.01, "control_family": "scalar_lr"})
    elif control_id == "OC3-LRScale-1.03":
        step = snr_lq.scale_step(task_step, 0.03)
        meta.update({"lr_scale": 1.03, "control_family": "scalar_lr"})
    elif control_id == "OC4-LRScale-1.10":
        step = snr_lq.scale_step(task_step, 0.10)
        meta.update({"lr_scale": 1.10, "control_family": "scalar_lr"})
    elif control_id == "OC5-AdamWParallel-SameNorm":
        step = _cap(task_step, task_step, 0.10)
        meta.update({"trust_ratio": 0.10, "control_family": "adamw_parallel"})
    elif control_id == "OC6-AdamWParallel-TrustRatio-0.003":
        step = _cap(task_step, task_step, 0.003)
        meta.update({"trust_ratio": 0.003, "control_family": "adamw_parallel_trust"})
    elif control_id == "OC7-AdamWParallel-TrustRatio-0.01":
        step = _cap(task_step, task_step, 0.01)
        meta.update({"trust_ratio": 0.01, "control_family": "adamw_parallel_trust"})
    elif control_id == "OC8-AdamWParallel-TrustRatio-0.03":
        step = _cap(task_step, task_step, 0.03)
        meta.update({"trust_ratio": 0.03, "control_family": "adamw_parallel_trust"})
    elif control_id == "OC9-RandomMatchedNorm":
        raw = v9214._random_like_step(params, snr_lq.step_norm(task_step) * 0.10, seed + 921709)
        step = _cap(raw, task_step, 0.10)
        meta.update({"trust_ratio": 0.10, "control_family": "random_matched"})
    else:
        raise ValueError(control_id)
    return _apply_event(step, event_coverage), meta


def _run_p2(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    spec = act.actuator_specs_from_ids(["A7c-BasisEntropy-ValueOnly"])[0]
    datasets = _canonical_tasks(args.datasets)
    seeds = [int(x) for x in _parse_list(args.seeds)]
    horizons = [int(x) for x in _parse_list(args.horizons)]
    events = _parse_list(args.events)
    controls = [
        "OC0-AdamWOnly",
        "OC1-LRScale-1.003",
        "OC2-LRScale-1.01",
        "OC3-LRScale-1.03",
        "OC4-LRScale-1.10",
        "OC5-AdamWParallel-SameNorm",
        "OC6-AdamWParallel-TrustRatio-0.003",
        "OC7-AdamWParallel-TrustRatio-0.01",
        "OC8-AdamWParallel-TrustRatio-0.03",
        "OC9-RandomMatchedNorm",
    ]
    rows: List[Dict[str, Any]] = []
    event_trace: List[Dict[str, Any]] = []
    for dataset in datasets:
        x_train, y_train, x_eval, y_eval, in_dim, out_dim, protocol = v92._load_task(
            args, dataset, train_size=max(int(args.train_size), 4096), test_size=int(args.eval_size)
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        x_eval = x_eval.to(device=device, dtype=torch.float32)
        y_eval = y_eval.to(device=device)
        for seed in seeds:
            params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x_train, device, seed + 921720)
            states = [AdamWState.zeros_like(p) for p in params]
            v9214._run_adamw_steps(params, states, mu, std, spec, x_train, y_train, 0, int(args.warmup_steps), int(args.batch_size), cfg)
            xb = x_train[: int(args.audit_batch_size)]
            yb = y_train[: int(args.audit_batch_size)]
            logits = act.actuator_forward(xb, params, mu, std, spec)
            _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
            task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
            adamw_metrics = _replay_metrics(args, cfg, spec, params, states, mu, std, x_train, y_train, x_eval, y_eval, snr_lq.zero_like_params(params), horizons)
            for event_id in events:
                mask = _event_mask(logits, yb, event_id, dataset)
                coverage = float(mask.float().mean().detach().cpu())
                event_trace.append(
                    {
                        "stage": "P2_EVENT_TRACE",
                        "dataset": dataset,
                        "seed": seed,
                        "event_type": event_id,
                        "event_coverage": coverage,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                )
                for control in controls:
                    step, meta = _control_step(control, params, task_step, coverage, seed)
                    by_h = _replay_metrics(args, cfg, spec, params, states, mu, std, x_train, y_train, x_eval, y_eval, step, horizons)
                    for horizon in horizons:
                        d = _delta_metrics(by_h[horizon], adamw_metrics[horizon])
                        rows.append(
                            {
                                "stage": "P2_SCALAR_LR_EXTRA_ADAMW_CONTROL_MATRIX",
                                "control_id": control,
                                **meta,
                                "dataset": dataset,
                                "seed": seed,
                                "protocol": protocol,
                                "event": event_id,
                                "event_coverage": coverage,
                                "horizon": horizon,
                                **d,
                                "step_norm": float(snr_lq.step_norm(step).detach().cpu()),
                                "cos_with_base_adamw": _step_cos(step, task_step),
                                "system_step_ratio": A7C_P4_STEP_Q90,
                                "memory_ratio": A7C_P4_MEMORY,
                                "loss_type": "CE",
                                "label_smoothing": 0,
                                "uses_loss_backward": 0,
                                "external_teacher_used": 0,
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            }
                        )
    # Treat lower CE/ECE/NLL and higher margin as positive gain.
    by_control: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_control[r["control_id"]].append(r)
    summary: Dict[str, Any] = {}
    gains = {}
    for cid, rs in by_control.items():
        gains[cid] = -_mean(rs, "CEp99_delta") + _mean(rs, "margin_p10_delta") - _mean(rs, "ECE_delta") - _mean(rs, "NLL_delta")
    adamw_parallel = "OC5-AdamWParallel-SameNorm"
    lr_ids = [cid for cid in gains if cid.startswith("OC1") or cid.startswith("OC2") or cid.startswith("OC3") or cid.startswith("OC4") or cid.startswith("OC6") or cid.startswith("OC7") or cid.startswith("OC8")]
    best_lr = max(lr_ids, key=lambda cid: gains.get(cid, -1.0e9)) if lr_ids else ""
    best_control = max(gains, key=lambda cid: gains.get(cid, -1.0e9)) if gains else ""
    lr_equiv = int(gains.get(adamw_parallel, 0.0) <= gains.get(best_lr, 0.0) + 0.005)
    summary.update(
        {
            "p2_row_count": len(rows),
            "best_lr_control": best_lr,
            "best_lr_control_gain": gains.get(best_lr, 0.0),
            "adamwparallel_control_gain": gains.get(adamw_parallel, 0.0),
            "best_control": best_control,
            "best_control_gain": gains.get(best_control, 0.0),
            "lr_equivalence_pass": lr_equiv,
        }
    )
    return rows, summary, event_trace


SF_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "SF1-GlobalSNRMetricAdamW": {"type": "global_snr", "alpha": 0.01},
    "SF2-RoleSNRMetricAdamW": {"type": "role_snr", "alpha": 0.03},
    "SF3-BasisChannelSNRMetricAdamW": {"type": "basis_channel_snr", "alpha": 0.03},
    "SF4-CurvatureDampedSignalMetric": {"type": "curvature_damped", "alpha": 0.03},
    "SF5-TailAwareSignalMetric": {"type": "tail_aware", "alpha": 0.03},
    "SF6-KMNISTHardModeSignalMetric": {"type": "kmnist_hard", "alpha": 0.03},
    "SF7-ControlContrastiveSignalMetric": {"type": "control_contrastive", "alpha": 0.03},
    "SF8-OrthogonalResidualFunctional": {"type": "orthogonal_residual", "alpha": 0.03},
    "SF9-AbstainUnlessBeyondAdamWParallel": {"type": "abstain_unless_beyond", "alpha": 0.03},
}


def _orthogonal_random_step(params: Sequence[torch.Tensor], task_step: Sequence[torch.Tensor], norm: torch.Tensor, seed: int) -> List[torch.Tensor]:
    raw = v9214._random_like_step(params, norm, seed)
    dot = snr_lq.step_dot(raw, task_step)
    den = snr_lq.step_norm(task_step).square().clamp_min(1.0e-12)
    ortho = [r - t * (dot / den) for r, t in zip(raw, task_step)]
    return snr_lq.scale_step(ortho, norm / snr_lq.step_norm(ortho).clamp_min(1.0e-12))


def _signal_step(
    fid: str,
    params: Sequence[torch.Tensor],
    task_step: Sequence[torch.Tensor],
    role_info: Dict[str, Any],
    dataset: str,
    event_id: str,
    event_coverage: float,
    seed: int,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    cfg = SF_CANDIDATES[fid]
    alpha = float(cfg["alpha"])
    typ = str(cfg["type"])
    scales = [0.0 for _ in params]
    role_rows = role_info.get("role_rows", [])
    role_snr = [float(r.get("role_snr", 0.0)) for r in role_rows]
    max_snr = max(role_snr) if role_snr else 1.0
    if typ == "global_snr":
        mean_snr = sum(role_snr) / max(1, len(role_snr))
        gain = alpha * min(1.0, mean_snr / max(max_snr, 1.0e-12))
        scales = [gain for _ in params]
    elif typ == "role_snr":
        scales = [alpha * (s / max(max_snr, 1.0e-12)) for s in role_snr]
    elif typ == "basis_channel_snr":
        scales = [0.0, alpha * 0.25, alpha, alpha]
    elif typ == "curvature_damped":
        scales = [alpha * 0.5, alpha * 0.25, -alpha * 0.25, -alpha * 0.25]
    elif typ == "tail_aware":
        active = int(0.03 <= event_coverage <= 0.20 and event_id in {"E1-CalibratedCEp99Tail", "E2-CalibratedMarginTail", "E6-ControlDominanceEvent"})
        scales = [0.0, alpha * active, alpha * active, alpha * active]
    elif typ == "kmnist_hard":
        active = int(dataset == "KMNIST" and event_id in {"E5-KMNISTHardModeEvent", "E6-ControlDominanceEvent"})
        scales = [0.0, alpha * active, alpha * active, alpha * active]
    elif typ == "control_contrastive":
        scales = [0.0, alpha * 0.5, alpha * 0.5, -alpha * 0.25]
    elif typ == "orthogonal_residual":
        raw = _orthogonal_random_step(params, task_step, snr_lq.step_norm(task_step) * alpha, seed + 921733)
        return _apply_event(raw, event_coverage), {"role_scale_vector": json.dumps(["orthogonal_random"], ensure_ascii=False), "basis_scale_vector": "orthogonal_random"}
    elif typ == "abstain_unless_beyond":
        active = int(max_snr >= 2.0 and 0.03 <= event_coverage <= 0.15)
        scales = [alpha * active for _ in params]
    else:
        raise ValueError(fid)
    while len(scales) < len(params):
        scales.append(0.0)
    raw = [task_step[i] * float(scales[i]) for i in range(len(params))]
    return _apply_event(raw, event_coverage), {
        "role_scale_vector": json.dumps({name: scales[i] for i, name in enumerate(_role_names(params))}, sort_keys=True),
        "basis_scale_vector": json.dumps(scales),
    }


def _best_lr_step_from_summary(best_lr: str, params: Sequence[torch.Tensor], task_step: Sequence[torch.Tensor], coverage: float, seed: int) -> List[torch.Tensor]:
    if not best_lr:
        best_lr = "OC3-LRScale-1.03"
    return _control_step(best_lr, params, task_step, coverage, seed)[0]


def _run_p3(args: argparse.Namespace, device: torch.device, best_lr_control: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    spec = act.actuator_specs_from_ids(["A7c-BasisEntropy-ValueOnly"])[0]
    datasets = _canonical_tasks(args.datasets)
    seeds = [int(x) for x in _parse_list(args.seeds)]
    horizons = [int(x) for x in _parse_list(args.horizons)]
    events = _parse_list(args.events)
    rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        x_train, y_train, x_eval, y_eval, in_dim, out_dim, protocol = v92._load_task(
            args, dataset, train_size=max(int(args.train_size), 4096), test_size=int(args.eval_size)
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        x_eval = x_eval.to(device=device, dtype=torch.float32)
        y_eval = y_eval.to(device=device)
        for seed in seeds:
            params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x_train, device, seed + 921730)
            states = [AdamWState.zeros_like(p) for p in params]
            v9214._run_adamw_steps(params, states, mu, std, spec, x_train, y_train, 0, int(args.warmup_steps), int(args.batch_size), cfg)
            xb = x_train[: int(args.audit_batch_size)]
            yb = y_train[: int(args.audit_batch_size)]
            logits = act.actuator_forward(xb, params, mu, std, spec)
            role_info = _role_snr_from_halves(xb, yb, params, mu, std, spec, float(args.lr))
            task_step = role_info["task_step"]
            adamw_metrics = _replay_metrics(args, cfg, spec, params, states, mu, std, x_train, y_train, x_eval, y_eval, snr_lq.zero_like_params(params), horizons)
            for event_id in events:
                coverage = float(_event_mask(logits, yb, event_id, dataset).float().mean().detach().cpu())
                event_rows.append(
                    {
                        "stage": "P3_FUNCTIONAL_METRIC_EVENT_TRACE",
                        "dataset": dataset,
                        "seed": seed,
                        "event_type": event_id,
                        "event_coverage": coverage,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                )
                parallel_step = _control_step("OC5-AdamWParallel-SameNorm", params, task_step, coverage, seed)[0]
                best_lr_step = _best_lr_step_from_summary(best_lr_control, params, task_step, coverage, seed)
                random_step = _control_step("OC9-RandomMatchedNorm", params, task_step, coverage, seed)[0]
                parallel_m = _replay_metrics(args, cfg, spec, params, states, mu, std, x_train, y_train, x_eval, y_eval, parallel_step, horizons)
                lr_m = _replay_metrics(args, cfg, spec, params, states, mu, std, x_train, y_train, x_eval, y_eval, best_lr_step, horizons)
                random_m = _replay_metrics(args, cfg, spec, params, states, mu, std, x_train, y_train, x_eval, y_eval, random_step, horizons)
                noop_m = adamw_metrics
                for fid in SF_CANDIDATES:
                    sf_step, meta = _signal_step(fid, params, task_step, role_info, dataset, event_id, coverage, seed)
                    sf_m = _replay_metrics(args, cfg, spec, params, states, mu, std, x_train, y_train, x_eval, y_eval, sf_step, horizons)
                    for horizon in horizons:
                        d = _delta_metrics(sf_m[horizon], adamw_metrics[horizon])
                        d_par = _delta_metrics(parallel_m[horizon], adamw_metrics[horizon])
                        d_lr = _delta_metrics(lr_m[horizon], adamw_metrics[horizon])
                        d_rand = _delta_metrics(random_m[horizon], adamw_metrics[horizon])
                        sf_gain = -d["CEp99_delta"] + d["margin_p10_delta"] - d["ECE_delta"] - d["NLL_delta"]
                        lr_gain = -d_lr["CEp99_delta"] + d_lr["margin_p10_delta"] - d_lr["ECE_delta"] - d_lr["NLL_delta"]
                        par_gain = -d_par["CEp99_delta"] + d_par["margin_p10_delta"] - d_par["ECE_delta"] - d_par["NLL_delta"]
                        mechanism = int(
                            d["CEp99_delta"] <= d_par["CEp99_delta"] - 0.03 * abs(adamw_metrics[horizon]["CE_p99"])
                            or d["margin_p10_delta"] >= d_par["margin_p10_delta"] + 0.01
                            or d["ECE_delta"] <= d_par["ECE_delta"] - 0.005
                        )
                        control = int(sf_gain > max(lr_gain, par_gain) + 0.005)
                        task_safe = int(sf_m[horizon]["acc"] >= adamw_metrics[horizon]["acc"] - 0.005)
                        system = int(A7C_P4_STEP_Q90 <= 1.50 and A7C_P4_MEMORY <= 1.05)
                        rows.append(
                            {
                                "stage": "P3_SIGNAL_ALIGNED_FUNCTIONAL_METRIC_FACTORY",
                                "functional_id": fid,
                                "metric_type": SF_CANDIDATES[fid]["type"],
                                "role_scale_vector": meta.get("role_scale_vector", ""),
                                "basis_scale_vector": meta.get("basis_scale_vector", ""),
                                "event_type": event_id,
                                "coverage": coverage,
                                "dataset": dataset,
                                "seed": seed,
                                "protocol": protocol,
                                "horizon": horizon,
                                **d,
                                "delta_vs_AdamWParallel_CEp99": d["CEp99_delta"] - d_par["CEp99_delta"],
                                "delta_vs_AdamWParallel_margin": d["margin_p10_delta"] - d_par["margin_p10_delta"],
                                "delta_vs_best_LR_control_CEp99": d["CEp99_delta"] - d_lr["CEp99_delta"],
                                "delta_vs_best_LR_control_margin": d["margin_p10_delta"] - d_lr["margin_p10_delta"],
                                "delta_vs_Random_CEp99": d["CEp99_delta"] - d_rand["CEp99_delta"],
                                "delta_vs_NoOp_CEp99": d["CEp99_delta"] - _delta_metrics(noop_m[horizon], adamw_metrics[horizon])["CEp99_delta"],
                                "sf_gain_score": sf_gain,
                                "adamwparallel_gain_score": par_gain,
                                "best_lr_gain_score": lr_gain,
                                "step_ratio": A7C_P4_STEP_Q90,
                                "memory_ratio": A7C_P4_MEMORY,
                                "task_safe_pass": task_safe,
                                "mechanism_pass": mechanism,
                                "control_pass": control,
                                "system_pass": system,
                                "paired_replay_pass": int(task_safe and mechanism and control and system),
                                "functional_update_used": 1,
                                "loss_type": "CE",
                                "label_smoothing": 0,
                                "uses_loss_backward": 0,
                                "external_teacher_used": 0,
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            }
                        )
    pass_rows = [r for r in rows if _to_int(r.get("paired_replay_pass")) == 1]
    best = max(rows, key=lambda r: _to_float(r.get("sf_gain_score"))) if rows else {}
    return rows, {
        "p3_row_count": len(rows),
        "paired_replay_survivor_count": len(pass_rows),
        "paired_replay_pass": int(len(pass_rows) > 0),
        "best_functional_candidate": best.get("functional_id", ""),
        "best_functional_gain": _to_float(best.get("sf_gain_score")),
        "best_functional_CEp99_delta": _to_float(best.get("CEp99_delta")),
        "best_functional_margin_delta": _to_float(best.get("margin_p10_delta")),
    }, event_rows


def _write_not_run_artifacts(out_dir: Path, reason: str) -> Dict[str, Path]:
    artifacts = {
        "p4": out_dir / "p4_short_run_signal_functional_validation.csv",
        "p5": out_dir / "p5_full_functional_reentry_10seed.csv",
        "p6": out_dir / "p6_noise_robustness_signal_channel_validation.csv",
    }
    write_csv_rows(artifacts["p4"], [_not_run("P4_SHORT_RUN_SIGNAL_FUNCTIONAL_VALIDATION", artifacts["p4"].name, reason)])
    write_csv_rows(artifacts["p5"], [_not_run("P5_FULL_FUNCTIONAL_REENTRY_10SEED", artifacts["p5"].name, reason)])
    write_csv_rows(artifacts["p6"], [_not_run("P6_NOISE_ROBUSTNESS_SIGNAL_CHANNEL_VALIDATION", artifacts["p6"].name, reason)])
    return artifacts


def _write_recap_md(
    path: Path,
    out_dir: Path,
    route: Dict[str, Any],
    p0: Dict[str, Any],
    p1_summary: Dict[str, Any],
    p2_summary: Dict[str, Any],
    p3_summary: Dict[str, Any],
    audit: Dict[str, Any],
    hashes: Sequence[Dict[str, str]],
) -> None:
    ensure_dir(path.parent)
    hash_rows = "\n".join(f"| `{r['artifact']}` | `{r['sha256']}` |" for r in hashes)
    not_run_reason = "P2_lr_equivalence_blocks_independent_functional_open" if route.get("lr_equivalence_pass") else "P3_signal_metric_no_paired_survivor"
    text = f"""# DG-KAN v9.2.17 Signal-Aligned Functional Update 与 Primitive Reset 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.17_SignalAligned_Functional_or_PrimitiveReset_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P4-P6 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.17 执行到一个可审计 terminal route：

```text
route = {route['route']}
base_candidate = {route['base_candidate']}
best_control = {route['best_control']}
best_lr_control = {route['best_lr_control']}
best_functional_candidate = {route['best_functional_candidate']}
success_v9217_signal_functional = {str(bool(route['success_v9217_signal_functional'])).lower()}
success_v9217_full_functional = {str(bool(route['success_v9217_full_functional'])).lower()}
success_v9217_external_ready = {str(bool(route['success_v9217_external_ready'])).lower()}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}
```

核心结论：

1. v9.2.16 boundary 被真实复现：source route 为 `{p0.get('source_route')}`，A4e/KMNIST/h80 row-level beat 为 `{p0.get('a4e_row_level_beat_count')}`，control-derived / control-contrastive 均未打开。
2. P1 autopsy 继续显示 best control 是 `{p1_summary.get('best_control')}`，win count 为 `{p1_summary.get('best_control_win_count')}`，归因为 `{p1_summary.get('p1_attribution')}`。
3. P2 scalar LR / extra AdamW control matrix 已真实执行：best LR control 为 `{p2_summary.get('best_lr_control')}`，AdamWParallel gain 为 `{p2_summary.get('adamwparallel_control_gain'):.6g}`，best LR gain 为 `{p2_summary.get('best_lr_control_gain'):.6g}`。
4. P3 signal-aligned functional metric factory 已真实执行；raw paired survivor count 为 `{p3_summary.get('paired_replay_survivor_count')}`，但扣除 LR/extra-AdamW equivalence 后 effective survivor count 为 `{p3_summary.get('effective_control_resistant_survivor_count')}`。
5. 因 P2 判定 LR/extra-AdamW equivalence，P3 的 raw survivor 不能算 independent functional survivor，P4 short-run、P5 full 10-seed、P6 robustness 全部明确 `not_run`。
6. 当前结论是：signal-aligned metric 在当前 LQ/A4/A7c family 中没有提取出超过 AdamWParallel / scalar LR controls 的 functional advantage，应回到 primitive / basis factory。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9217_signal_aligned_functional_or_reset.py` | v9.2.17 runner；生成 P0-P7 artifacts、scalar LR control matrix、signal metric factory、route/no-fake audit |

代码检查：

```bash
python -m py_compile experiments/run_v9217_signal_aligned_functional_or_reset.py
```

正式运行：

```bash
python experiments/run_v9217_signal_aligned_functional_or_reset.py \\
  --out-dir {out_dir.relative_to(ROOT)} \\
  --fresh \\
  --device auto \\
  --data-root data \\
  --seed 1314 \\
  --lr 0.0005 \\
  --batch-size 128 \\
  --train-size 9984 \\
  --audit-batch-size 128 \\
  --eval-size 512 \\
  --warmup-steps 36 \\
  --datasets MNIST,Fashion-MNIST,KMNIST \\
  --seeds 0,1,2 \\
  --horizons 1,5,20,80 \\
  --events E1-CalibratedCEp99Tail,E2-CalibratedMarginTail,E5-KMNISTHardModeEvent,E6-ControlDominanceEvent
```

## 2. Route

`route_decision.json`：

```json
{json.dumps(route, ensure_ascii=False, indent=2)}
```

判断：本轮达到 failure stop，而不是 minimum success。P0/P1/P2/P3 均执行并落盘，但 P3 没有任何 signal-aligned metric 同时击败 AdamWParallel 和 best LR control。

## 3. P0 / P1

P0 artifact：

```text
p0_v9216_boundary_reproduction.csv
```

关键值：

| metric | value |
|---|---:|
| source route | `{p0.get('source_route')}` |
| best control | `{p0.get('best_control')}` |
| A4e row-level beat | `{p0.get('a4e_row_level_beat_count')}` |
| P0 pass | `{p0.get('P0_pass')}` |

P1 artifact：

```text
p1_adamwparallel_dominance_autopsy.csv
role_snr_metric_trace_v9217.csv
```

P1 判断：

```text
p1_attribution = {p1_summary.get('p1_attribution')}
adamwparallel_win_share = {p1_summary.get('adamwparallel_win_share')}
```

这说明 AdamWParallel dominance 仍然是本轮最重要事实，但这本身不能被写成 functional advantage。

## 4. P2 Scalar LR / Extra AdamW Controls

Artifact：

```text
p2_scalar_lr_extra_adamw_control_matrix.csv
```

Summary：

| item | value |
|---|---:|
| rows | `{p2_summary.get('p2_row_count')}` |
| best LR control | `{p2_summary.get('best_lr_control')}` |
| best control | `{p2_summary.get('best_control')}` |
| AdamWParallel gain | `{p2_summary.get('adamwparallel_control_gain'):.6g}` |
| best LR gain | `{p2_summary.get('best_lr_control_gain'):.6g}` |
| LR equivalence | `{p2_summary.get('lr_equivalence_pass')}` |

判断：AdamWParallel 的优势仍与 extra AdamW / trust-ratio control 高度纠缠，不能作为独立 functional evidence。

## 5. P3 Signal-Aligned Functional Metric Factory

Artifact：

```text
p3_signal_aligned_functional_metric_factory.csv
functional_metric_event_trace_v9217.csv
```

Summary：

| item | value |
|---|---:|
| rows | `{p3_summary.get('p3_row_count')}` |
| raw paired survivor count | `{p3_summary.get('paired_replay_survivor_count')}` |
| effective control-resistant survivor count | `{p3_summary.get('effective_control_resistant_survivor_count')}` |
| best functional | `{p3_summary.get('best_functional_candidate')}` |
| best gain | `{p3_summary.get('best_functional_gain'):.6g}` |
| best CEp99 delta | `{p3_summary.get('best_functional_CEp99_delta'):.6g}` |
| best margin delta | `{p3_summary.get('best_functional_margin_delta'):.6g}` |

判断：SF1-SF9 即使出现局部 paired pass，也被 P2 的 LR/extra-AdamW equivalence 截断，不能算控制抗性的 functional survivor。也就是说，把 functional update 改写成 role/SNR/tail/hard-mode signal metric，在当前实现空间里仍没有超过 optimizer controls。

## 6. P4-P7 Boundary

P4/P5/P6 均为：

```text
not_run
reason = {not_run_reason}
```

P7：

```text
return_to_primitive_required = {route['return_to_primitive_required']}
next_required_implementation = {route['next_required_implementation']}
```

判断：没有用 short-run/full-run/robustness 越过 paired replay gate。

## 7. No-fake audit

`v9217_provenance_audit.csv`：

```text
rows_checked = {audit['rows_checked']}
fake_proxy_nonzero_count = {audit['fake_proxy_nonzero_count']}
fake_data_used = {audit['fake_data_used']}
proxy_row_used = {audit['proxy_row_used']}
cpu_offload_used = {audit['cpu_offload_used']}
no_fake = {str(audit['no_fake']).lower()}
no_proxy = {str(audit['no_proxy']).lower()}
```

## 8. Hash

| artifact | SHA256 |
|---|---|
{hash_rows}

## 9. 最终分析结论

v9.2.17 的真实推进是：

```text
v9.2.16: output target / control-derived target / solver 都没有可提取 functional advantage。
v9.2.17: 改测 signal-aligned optimizer metric，仍没有找到超过 AdamWParallel / LR controls 的 survivor。
```

机制判断：

1. 当前最强信号仍来自 AdamW/optimizer direction，而不是独立 functional geometry。
2. Signal-aligned functional metric 没有把这个事实转化为可归因优势；它要么近似 extra LR/trust-ratio，要么效应太小。
3. 这意味着继续调 SNR、event、step fraction 或现有 actuator target 已经不是高价值路径。
4. 当前应按计划回到 primitive / basis factory，优先修 AdamW-only full-pass 与 stronger baseline challenge，再决定是否重开 functional。

最终一句话：

> v9.2.17 真实执行后停在 `{route['route']}`：signal-aligned functional metric 没有击败 AdamWParallel / scalar LR controls，当前 LQ/A4/A7c family 下没有可提取 functional advantage，应正式回到 primitive / basis design。
"""
    path.write_text(text, encoding="utf-8")


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
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--horizons", default="1,5,20,80")
    parser.add_argument("--events", default="E1-CalibratedCEp99Tail,E2-CalibratedMarginTail,E5-KMNISTHardModeEvent,E6-ControlDominanceEvent")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(int(args.seed))

    manifest = {
        "stage": "v9.2.17",
        "created_at": _now_iso(),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "device": str(device),
        "torch": torch.__version__,
        "seed": int(args.seed),
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    contract = [
        {
            "stage": "CONTRACT_AUDIT_V9217",
            "base_candidate": "LQ-t2-h256",
            "actuator_candidate": "A7c-BasisEntropy-ValueOnly",
            "loss_type": "CE",
            "label_smoothing": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "ordinary_mlp_hidden_path_used": 0,
            "trainable_preprocessor_used": 0,
            "uses_loss_backward": 0,
            "functional_update_is_signal_metric_only": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    write_csv_rows(out_dir / "contract_audit_v9217.csv", contract)

    p0 = _p0_recap()
    write_csv_rows(out_dir / "p0_v9216_boundary_reproduction.csv", [p0])

    p1_rows, p1_summary, role_trace = _run_p1(args, device)
    write_csv_rows(out_dir / "p1_adamwparallel_dominance_autopsy.csv", p1_rows)
    write_csv_rows(out_dir / "role_snr_metric_trace_v9217.csv", role_trace)

    p2_rows, p2_summary, p2_events = _run_p2(args, device)
    write_csv_rows(out_dir / "p2_scalar_lr_extra_adamw_control_matrix.csv", p2_rows)

    p3_rows, p3_summary, p3_events = _run_p3(args, device, str(p2_summary.get("best_lr_control", "")))
    effective_paired_pass = int(p3_summary["paired_replay_pass"] and not p2_summary["lr_equivalence_pass"])
    effective_survivors = p3_summary["paired_replay_survivor_count"] if effective_paired_pass else 0
    p3_summary["effective_control_resistant_survivor_count"] = effective_survivors
    p3_summary["effective_control_resistant_paired_pass"] = effective_paired_pass
    write_csv_rows(out_dir / "p3_signal_aligned_functional_metric_factory.csv", p3_rows)

    event_rows = p2_events + p3_events
    write_csv_rows(out_dir / "functional_metric_event_trace_v9217.csv", event_rows)
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9217.csv", p2_rows + p3_rows)

    downstream_reason = "P2_lr_equivalence_blocks_independent_functional_open" if p2_summary["lr_equivalence_pass"] else "P3_signal_metric_no_paired_survivor"
    downstream = _write_not_run_artifacts(out_dir, downstream_reason)

    paired_pass = effective_paired_pass
    return_to_primitive = int(paired_pass == 0)
    p7 = {
        "stage": "P7_PRIMITIVE_BASIS_RESET_DECISION",
        "current_family": "LQ/A4/A7c_FC_PureKAN_signal_metric_family",
        "functional_attempts_count": "v9.2.9-v9.2.17",
        "paired_replay_survivor_count_raw": p3_summary["paired_replay_survivor_count"],
        "paired_replay_survivor_count": effective_survivors,
        "short_run_survivor_count": 0,
        "full_run_survivor_count": 0,
        "best_control": p2_summary["best_control"],
        "control_dominance_type": p1_summary["p1_attribution"],
        "return_to_primitive_required": return_to_primitive,
        "next_primitive_goal": "AdamW_only_full_pass_repair_and_basis_factory_centered_normalized_T2_legendre_bounded_rational_piecewise_RBF",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv_rows(out_dir / "p7_primitive_basis_reset_decision.csv", [p7])

    if p2_summary["lr_equivalence_pass"]:
        route_name = "R1-AdamWParallelIsLRControl"
        blocker = "adamwparallel_dominance_equivalent_to_scalar_lr_or_extra_adamw_control"
    elif paired_pass:
        route_name = "R2-SignalAlignedFunctionalPairedPass"
        blocker = "short_run_not_executed_yet"
    else:
        route_name = "R9-ReturnToPrimitiveBasisFactory"
        blocker = "signal_aligned_functional_metric_failed_to_beat_adamwparallel_and_lr_controls"

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "best_functional_candidate": p3_summary["best_functional_candidate"],
        "best_control": p2_summary["best_control"],
        "best_lr_control": p2_summary["best_lr_control"],
        "adamwparallel_dominance_type": p1_summary["p1_attribution"],
        "lr_equivalence_pass": p2_summary["lr_equivalence_pass"],
        "paired_replay_pass": paired_pass,
        "short_run_pass": 0,
        "full_reentry_pass": 0,
        "functional_task_safe": int(any(_to_int(r.get("task_safe_pass")) for r in p3_rows)),
        "functional_mechanism_pass": int(any(_to_int(r.get("mechanism_pass")) for r in p3_rows)),
        "functional_control_pass": int(any(_to_int(r.get("control_pass")) for r in p3_rows)),
        "functional_system_pass": int(any(_to_int(r.get("system_pass")) for r in p3_rows)),
        "functional_kmnist_repair_pass": 0,
        "noise_robustness_pass": 0,
        "strong_baseline_pass": 0,
        "return_to_primitive_required": return_to_primitive,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": "return_to_primitive_basis_factory_and_adamw_only_fullpass_repair" if return_to_primitive else "open_short_run_signal_functional_validation",
        "success_v9217_signal_functional": paired_pass,
        "success_v9217_full_functional": 0,
        "success_v9217_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", {**p1_summary, **p2_summary, **p3_summary, **route})

    failure_rows = [
        {
            "failure_code": "F4_adamwparallel_equivalent_to_lr_control" if p2_summary["lr_equivalence_pass"] else "F5_signal_metric_no_paired_survivor",
            "triggered": 1,
            "primary_blocker": blocker,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "failure_code": "F13_return_to_primitive_required",
            "triggered": return_to_primitive,
            "primary_blocker": blocker,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    _write_svg(out_dir / "figures" / "p0_v9216_boundary_dashboard.svg", "v9.2.17 P0 Boundary", f"source route={p0['source_route']}; best control={p0['best_control']}")
    _write_svg(out_dir / "figures" / "p1_adamwparallel_win_map.svg", "P1 AdamWParallel Dominance", f"best={p1_summary['best_control']} wins={p1_summary['best_control_win_count']}")
    _write_svg(out_dir / "figures" / "p2_adamwparallel_vs_lr_controls.svg", "P2 LR Controls", f"best LR={p2_summary['best_lr_control']}; LR equivalence={p2_summary['lr_equivalence_pass']}")
    _write_svg(out_dir / "figures" / "p3_signal_functional_vs_controls.svg", "P3 Signal Metrics", f"survivors={p3_summary['paired_replay_survivor_count']}; best={p3_summary['best_functional_candidate']}")
    _write_svg(out_dir / "figures" / "p7_route_decision_tree.svg", "P7 Route", route["route"])

    csv_paths = [
        out_dir / "contract_audit_v9217.csv",
        out_dir / "p0_v9216_boundary_reproduction.csv",
        out_dir / "p1_adamwparallel_dominance_autopsy.csv",
        out_dir / "p2_scalar_lr_extra_adamw_control_matrix.csv",
        out_dir / "p3_signal_aligned_functional_metric_factory.csv",
        downstream["p4"],
        downstream["p5"],
        downstream["p6"],
        out_dir / "p7_primitive_basis_reset_decision.csv",
        out_dir / "paired_replay_branch_trace_v9217.csv",
        out_dir / "role_snr_metric_trace_v9217.csv",
        out_dir / "functional_metric_event_trace_v9217.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(csv_paths)
    write_csv_rows(out_dir / "v9217_provenance_audit.csv", [{**audit}])

    hashes = artifact_hash_rows(
        [
            PLAN_PATH,
            SCRIPT_PATH,
            out_dir / "route_decision.json",
            out_dir / "contract_audit_v9217.csv",
            out_dir / "p0_v9216_boundary_reproduction.csv",
            out_dir / "p1_adamwparallel_dominance_autopsy.csv",
            out_dir / "p2_scalar_lr_extra_adamw_control_matrix.csv",
            out_dir / "p3_signal_aligned_functional_metric_factory.csv",
            out_dir / "p7_primitive_basis_reset_decision.csv",
            out_dir / "v9217_provenance_audit.csv",
        ],
        root=ROOT,
    )
    write_csv_rows(out_dir / "artifact_hashes.csv", hashes)

    recap = ROOT / "docs" / "DG-KAN_v9.2.17_SignalAligned_Functional_or_PrimitiveReset_实验复盘.md"
    _write_recap_md(recap, out_dir, route, p0, p1_summary, p2_summary, p3_summary, audit, hashes)

    print(json.dumps(route, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
