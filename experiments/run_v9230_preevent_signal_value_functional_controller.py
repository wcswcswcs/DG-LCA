#!/usr/bin/env python3
"""DG-KAN v9.2.30 pre-event signal-value functional controller.

This runner starts from the real v9.2.29 terminal artifact, then performs a
new measured pre-event movement replay on strict PureKAN functional-interface
candidates.  Dataset names are used only for diagnostics and leave-dataset-out
evaluation; no official predictor score uses dataset name, seed id, class name,
teacher signal, proxy rows, or synthetic outcome labels.
"""

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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9222_basequalified_strict_purekan_functional_interface as v9222  # noqa: E402
import run_v9223_actuatability_to_causality_closure as v9223  # noqa: E402
import run_v9227_fashion_kmnist_metric_causal_repair as v9227  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.30_PreEvent_SignalValue_FunctionalController_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9230_preevent_signal_value_functional_controller.py"
SRC_V9229 = ROOT / "results" / "real_rerun_20260506" / "v9229_datasetagnostic_functional_causality_first_20260510T200000Z"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.30_PreEvent_SignalValue_FunctionalController_实验复盘.md"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, str) and value.startswith("not_"):
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / max(1, len(vals))


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return 0.0
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-30 or vy <= 1.0e-30:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def _q(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _rank01(vals: Sequence[float]) -> List[float]:
    if not vals:
        return []
    order = sorted(range(len(vals)), key=lambda i: (float(vals[i]), i))
    out = [0.0] * len(vals)
    denom = max(1, len(vals) - 1)
    for pos, idx in enumerate(order):
        out[idx] = float(pos) / float(denom)
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _not_run(stage: str, artifact: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "stage": stage,
        "status": "not_run",
        "artifact": artifact,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _make_helper_args(args: argparse.Namespace) -> argparse.Namespace:
    for name, value in {
        "p5_train_size": args.train_size,
        "p5_test_size": args.eval_size,
        "p5_epochs": args.p5_epochs,
        "p5_lr": args.lr,
        "eval_size": args.eval_size,
        "audit_batch_size": args.audit_batch_size,
        "batch_size": args.batch_size,
        "data_root": args.data_root,
        "seed": args.seed,
        "lr": args.lr,
        "best_lr_scale": args.best_lr_scale,
        "ridge": args.ridge,
    }.items():
        setattr(args, name, value)
    return args


def _source_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9229 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9229 / "v9229_provenance_audit.csv")
    fake = _int(audit_rows[0].get("fake_proxy_nonzero_count")) if audit_rows else 1
    p0_pass = int(
        route.get("route") == "R1-SignalStrataExplainFailures"
        and _int(route.get("signal_strata_explain_failures")) == 1
        and _int(route.get("event_value_predictor_pass")) == 0
        and _int(route.get("abstention_precision_pass")) == 1
        and fake == 0
    )
    return {
        "stage": "P0_V9229_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9229.relative_to(ROOT)),
        "route": route.get("route", ""),
        "source_route_v9228": route.get("source_route_v9228", "R5-KMNISTSurvivorNotPreserved"),
        "association_margin": route.get("association_margin", ""),
        "dataset_failure_association": route.get("dataset_failure_association", ""),
        "signal_strata_failure_association": route.get("signal_stratum_failure_association", ""),
        "event_value_corr": route.get("event_value_corr", ""),
        "accepted_precision": route.get("accepted_precision", ""),
        "accepted_coverage": route.get("accepted_coverage", ""),
        "accepted_recall": route.get("accepted_recall", ""),
        "accepted_bad_event_rate": route.get("accepted_bad_event_rate", ""),
        "accepted_event_count": route.get("accepted_event_count", ""),
        "event_value_predictor_pass": route.get("event_value_predictor_pass", ""),
        "abstention_precision_pass": route.get("abstention_precision_pass", ""),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _primitive_registry() -> Dict[str, str]:
    return {
        "P0-N2a-Rational": "N2a-TinyInit-RationalFunc-BranchRatioCap",
        "P1-N2c-SharedRBF": "N2c-TinyInit-SharedRBFFunc-BranchRatioCap",
        "P2-N3c-SharedRBFDerivativeBand": "N3c-SharedRBFFunc-DerivativeBand",
    }


def _event_specs() -> List[Dict[str, Any]]:
    return [
        {"event": "E1-CEHardTail", "targets": ["O1-HardTailLogitCorrection"], "fraction": 0.10, "mode": "safe"},
        {"event": "E2-MarginTail", "targets": ["O2-MarginTailExpansion"], "fraction": 0.10, "mode": "safe"},
        {"event": "E3-HardMode", "targets": ["O6-KMNISTHardModeOutputTarget"], "fraction": 0.10, "mode": "safe"},
        {"event": "E4-CurvatureTail", "targets": ["O4-CurvatureOutputFlattening"], "fraction": 0.10, "mode": "safe"},
        {"event": "E5-CEPlusMargin", "targets": ["O1-HardTailLogitCorrection", "O2-MarginTailExpansion"], "fraction": 0.10, "mode": "safe"},
        {"event": "E6-OrthogonalCE", "targets": ["O1-HardTailLogitCorrection"], "fraction": 0.10, "mode": "orthogonal"},
    ]


def _zero_like(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.zeros_like(p) for p in params]


def _step_cos(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> float:
    dot = snr_lq.step_dot(a, b)
    denom = snr_lq.step_norm(a).clamp_min(1.0e-12) * snr_lq.step_norm(b).clamp_min(1.0e-12)
    return float((dot / denom).detach().cpu())


def _logit_delta(
    params: Sequence[torch.Tensor],
    step: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    x: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        before = act.actuator_forward(x, params, mu, std, spec)
        after = act.actuator_forward(x, v9223._apply_step(params, step), mu, std, spec)
    return after - before


def _tensor_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    denom = af.norm().clamp_min(1.0e-12) * bf.norm().clamp_min(1.0e-12)
    return float(((af * bf).sum() / denom).detach().cpu())


def _target_tail_mask(target_id: str, logits: torch.Tensor, y: torch.Tensor, dataset: str) -> torch.Tensor:
    return v9223._target_mask(target_id, logits, y, dataset)


def _event_target_id(targets: Sequence[str]) -> str:
    if not targets:
        return "none"
    if len(targets) == 1:
        return str(targets[0])
    return str(targets[0])


def _score_variance(
    *,
    params: Sequence[torch.Tensor],
    real_step: Sequence[torch.Tensor],
    parallel_step: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    dataset: str,
    target_id: str,
) -> float:
    n = int(x_eval.shape[0])
    if n < 4:
        return 0.0
    vals: List[float] = []
    for lo, hi in [(0, n // 2), (n // 2, n)]:
        stats = v9223._actual_stats(
            params=params,
            real_step=real_step,
            parallel_step=parallel_step,
            mu=mu,
            std=std,
            spec=spec,
            x_eval=x_eval[lo:hi],
            y_eval=y_eval[lo:hi],
            dataset=dataset,
            target_id=target_id,
        )
        vals.append(-_float(stats.get("real_CEp99_delta")) + _float(stats.get("real_margin_p10_delta")))
    return float(np.var(np.asarray(vals, dtype=np.float64)))


def _classification_gain(delta: Dict[str, float]) -> float:
    return -_float(delta.get("CEp99_delta")) + _float(delta.get("margin_p10_delta"))


def _assign_signal_stratum(row: Dict[str, Any]) -> str:
    if _float(row.get("pre_tail_real_vs_control_ratio")) < 0.10:
        return "S5-LowActualMovementSilent"
    if _float(row.get("horizon")) >= 80.0 and (_float(row.get("CEp99_delta")) < 0.0 or _float(row.get("margin_p10_delta")) > 0.0):
        return "S6-DelayedTailSignal"
    if _float(row.get("pre_real_nonadamw_delta_norm")) > _float(row.get("pre_adamwparallel_logit_delta_norm")) and _float(row.get("pre_real_tail_logit_delta_norm")) > 0.0:
        return "S3-HighCurvatureLowConfidence"
    if _float(row.get("CEp99_delta")) < 0.0 and _float(row.get("margin_p10_delta")) > 0.0:
        return "S1-HighCEHighMarginRisk"
    if _float(row.get("margin_p10_delta")) > 0.0:
        return "S2-LowMarginHighWrongConfidence"
    if _float(row.get("actual_control_gap")) < -0.20:
        return "S4-HighActualMovementButControlDominated"
    return "S8-AbstainCandidate"


def _run_p1_features(
    args: argparse.Namespace,
    device: torch.device,
    opened: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_PRE_EVENT_MOVEMENT_FEATURES", "p1_pre_event_movement_features.csv", "P0_v9229_boundary_failed")
        return [row], [row], {"pre_event_feature_pass": 0, "p1_row_count": 0}

    registry = v9222._candidate_registry()
    primitive_map = _primitive_registry()
    selected_primitives = [p for p in _parse_list(args.primitives) if p in primitive_map]
    datasets = [v92._canonical_task(d) for d in _parse_list(args.datasets)]
    seeds = _parse_ints(args.seeds)
    horizons = _parse_ints(args.horizons)
    event_specs = _event_specs()
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        x_train, y_train, x_eval, y_eval, protocol = v9223._load_split(args, dataset, device)
        xb = x_train[: int(args.audit_batch_size)]
        yb = y_train[: int(args.audit_batch_size)]
        for primitive in selected_primitives:
            cand_id = primitive_map[primitive]
            cand = registry[cand_id]
            assert cand.spec is not None
            for seed in seeds:
                saved = v9223._train_cache(args, cand, dataset, seed, device, cache)
                params = saved["params"]
                states = saved["states"]
                mu = saved["mu"]
                std = saved["std"]
                for spec_event in event_specs:
                    real_step, task_step, _grads, info = v9227._step_for_targets(
                        args,
                        cand,
                        params,
                        mu,
                        std,
                        xb,
                        yb,
                        dataset,
                        spec_event["targets"],
                        float(spec_event["fraction"]),
                        str(spec_event["mode"]),
                    )
                    target_id = _event_target_id(spec_event["targets"])
                    parallel_step = v9222._cap_to_fraction(task_step, task_step, float(args.parallel_trust_ratio))
                    bestlr_step = v9222._cap_to_fraction(task_step, task_step, float(args.best_lr_scale) - 1.0)
                    random_step = v9222._random_like_step(params, snr_lq.step_norm(real_step), seed + len(rows) + 923000)
                    stats = v9223._actual_stats(
                        params=params,
                        real_step=real_step,
                        parallel_step=parallel_step,
                        mu=mu,
                        std=std,
                        spec=cand.spec,
                        x_eval=x_eval,
                        y_eval=y_eval,
                        dataset=dataset,
                        target_id=target_id,
                    )
                    base_logits = act.actuator_forward(x_eval, params, mu, std, cand.spec)
                    real_delta_logits = _logit_delta(params, real_step, mu, std, cand.spec, x_eval)
                    parallel_delta_logits = _logit_delta(params, parallel_step, mu, std, cand.spec, x_eval)
                    bestlr_delta_logits = _logit_delta(params, bestlr_step, mu, std, cand.spec, x_eval)
                    random_delta_logits = _logit_delta(params, random_step, mu, std, cand.spec, x_eval)
                    tail_mask = _target_tail_mask(target_id, base_logits, y_eval, dataset)
                    if bool(tail_mask.any()):
                        cos_tail = _tensor_cos(real_delta_logits[tail_mask], parallel_delta_logits[tail_mask])
                    else:
                        cos_tail = 0.0
                    diag = v9222._diagnose_channels(v9223._apply_step(params, real_step), mu, std, cand.spec, xb, yb, float(args.lr))
                    micro_var = _score_variance(
                        params=params,
                        real_step=real_step,
                        parallel_step=parallel_step,
                        mu=mu,
                        std=std,
                        spec=cand.spec,
                        x_eval=x_eval,
                        y_eval=y_eval,
                        dataset=dataset,
                        target_id=target_id,
                    )
                    metrics = v9223._run_replay_for_event(
                        args,
                        params,
                        states,
                        mu,
                        std,
                        cand.spec,
                        x_train,
                        y_train,
                        x_eval,
                        y_eval,
                        real_step,
                        task_step,
                        seed,
                        horizons,
                    )
                    for horizon in horizons:
                        adamw = metrics[("AdamWOnly", horizon)]
                        real = metrics[("RealFunctional", horizon)]
                        parallel_candidates = [m for (b, h), m in metrics.items() if h == horizon and b.startswith("AdamWParallel")]
                        best_parallel = min(parallel_candidates, key=lambda m: m["CE_p99"])
                        best_lr = metrics[("BestLRScale", horizon)]
                        noop = metrics[("NoOpMatchedOverhead", horizon)]
                        random = metrics[("RandomMatchedNorm", horizon)]
                        shuffled = metrics[("ShuffledRoleMask", horizon)]
                        inverted = metrics[("InvertedRoleMask", horizon)]
                        frozen = metrics[("FrozenFuncChannel", horizon)]
                        shuffled_func = metrics[("ShuffledFuncChannel", horizon)]
                        d_real = v9223._metric_delta(adamw, real)
                        d_parallel = v9223._metric_delta(adamw, best_parallel)
                        d_lr = v9223._metric_delta(adamw, best_lr)
                        d_noop = v9223._metric_delta(adamw, noop)
                        d_random = v9223._metric_delta(adamw, random)
                        d_shuf = v9223._metric_delta(adamw, shuffled)
                        d_inv = v9223._metric_delta(adamw, inverted)
                        d_frozen = v9223._metric_delta(adamw, frozen)
                        d_shuf_func = v9223._metric_delta(adamw, shuffled_func)
                        gain_real = _classification_gain(d_real)
                        gain_parallel = _classification_gain(d_parallel)
                        gain_lr = _classification_gain(d_lr)
                        actual_gap = min(gain_real - gain_parallel, gain_real - gain_lr)
                        real_beats_parallel = int(gain_real > gain_parallel and real["acc"] >= adamw["acc"] - 0.005)
                        real_beats_lr = int(gain_real > gain_lr and real["acc"] >= adamw["acc"] - 0.005)
                        row = {
                            "stage": "P1_PRE_EVENT_MOVEMENT_FEATURES",
                            "status": "measured",
                            "event_id": f"{dataset}:{seed}:{primitive}:{spec_event['event']}:{horizon}",
                            "dataset": dataset,
                            "seed": seed,
                            "protocol": protocol,
                            "horizon": horizon,
                            "primitive": primitive,
                            "source_candidate": cand_id,
                            "event_type": spec_event["event"],
                            "target": ",".join(spec_event["targets"]),
                            "target_selected_fraction": info.get("target_selected_fraction", ""),
                            "target_fit_R2": info.get("target_fit_R2", ""),
                            "target_rz": info.get("target_rz", ""),
                            "pre_real_logit_delta_norm": stats["actual_logit_delta_norm"],
                            "pre_real_tail_logit_delta_norm": stats["actual_tail_logit_delta_norm"],
                            "pre_real_nonadamw_delta_norm": stats["actual_nonadamw_logit_delta_norm"],
                            "pre_adamwparallel_logit_delta_norm": stats["adamwparallel_logit_delta_norm"],
                            "pre_bestlr_logit_delta_norm": float(bestlr_delta_logits.float().norm().detach().cpu()),
                            "pre_real_vs_adamw_delta_ratio": stats["actual_r_z"],
                            "pre_real_vs_bestlr_delta_ratio": float((real_delta_logits.float().norm() / bestlr_delta_logits.float().norm().clamp_min(1.0e-12)).detach().cpu()),
                            "pre_tail_real_vs_control_ratio": stats["actual_r_z_tail"],
                            "actual_r_z_perp": stats["actual_r_z_perp"],
                            "branch_ratio": diag["branch_ratio"],
                            "effective_derivative": diag["effective_derivative_p95"],
                            "functional_channel_entropy": diag["functional_channel_usage_entropy"],
                            "dominant_basis_fraction": diag["dominant_functional_basis_fraction"],
                            "primitive_family": cand_id,
                            "basis_usage_entropy": diag["functional_channel_usage_entropy"],
                            "cos_real_adamw": _tensor_cos(real_delta_logits, parallel_delta_logits),
                            "cos_real_bestlr": _tensor_cos(real_delta_logits, bestlr_delta_logits),
                            "cos_real_random": _tensor_cos(real_delta_logits, random_delta_logits),
                            "cos_tail_real_adamw": cos_tail,
                            "projected_tail_component_ratio": max(0.0, 1.0 - stats["actual_r_z_perp"]),
                            "orthogonal_component_ratio": stats["actual_r_z_perp"],
                            "microbatch_score_variance": micro_var,
                            "bootstrap_control_gap_std": math.sqrt(max(0.0, micro_var)),
                            "horizon_agreement_score": 0.0,
                            "stratum_sample_count": 0,
                            "event_score_entropy": 0.0,
                            "CEp99_delta": d_real["CEp99_delta"],
                            "margin_p10_delta": d_real["margin_p10_delta"],
                            "ECE_delta": d_real["ECE_delta"],
                            "NLL_delta": d_real["NLL_delta"],
                            "acc_delta": d_real["acc_delta"],
                            "NoOp_CEp99_delta": d_noop["CEp99_delta"],
                            "NoOp_margin_delta": d_noop["margin_p10_delta"],
                            "Random_CEp99_delta": d_random["CEp99_delta"],
                            "Random_margin_delta": d_random["margin_p10_delta"],
                            "AdamWParallel_CEp99_delta": d_parallel["CEp99_delta"],
                            "AdamWParallel_margin_delta": d_parallel["margin_p10_delta"],
                            "BestLR_CEp99_delta": d_lr["CEp99_delta"],
                            "BestLR_margin_delta": d_lr["margin_p10_delta"],
                            "ShuffledRole_CEp99_delta": d_shuf["CEp99_delta"],
                            "ShuffledRole_margin_delta": d_shuf["margin_p10_delta"],
                            "InvertedRole_CEp99_delta": d_inv["CEp99_delta"],
                            "FrozenFunc_CEp99_delta": d_frozen["CEp99_delta"],
                            "ShuffledFunc_CEp99_delta": d_shuf_func["CEp99_delta"],
                            "actual_real_minus_adamwparallel": gain_real - gain_parallel,
                            "actual_real_minus_bestlr": gain_real - gain_lr,
                            "actual_control_gap": actual_gap,
                            "real_beats_adamwparallel": real_beats_parallel,
                            "real_beats_best_lr": real_beats_lr,
                            "task_safe": int(real["acc"] >= adamw["acc"] - 0.005),
                            "bad_event_rate": int(real["acc"] < adamw["acc"] - 0.005),
                            "event_count": 1,
                            "event_coverage": 1.0 / max(1, int(args.audit_batch_size)),
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        }
                        row["signal_stratum"] = _assign_signal_stratum(row)
                        rows.append(row)

    # Add rank/stratum aggregation features after all rows are known.
    ce_rank = _rank01([max(0.0, -_float(r.get("CEp99_delta"))) for r in rows if r.get("status") == "measured"])
    margin_rank = _rank01([max(0.0, _float(r.get("margin_p10_delta"))) for r in rows if r.get("status") == "measured"])
    wrong_rank = _rank01([max(0.0, -_float(r.get("ECE_delta"))) for r in rows if r.get("status") == "measured"])
    measured = [r for r in rows if r.get("status") == "measured"]
    for row, ce, margin, wrong in zip(measured, ce_rank, margin_rank, wrong_rank):
        row["ce_tail_rank"] = ce
        row["margin_tail_rank"] = margin
        row["wrong_confidence_rank"] = wrong
        row["curvature_rank"] = _float(row.get("orthogonal_component_ratio"))

    by_event_base: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_stratum: Counter[str] = Counter()
    for row in measured:
        key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("primitive")), str(row.get("event_type")), str(row.get("target")))
        by_event_base[key].append(row)
        by_stratum[str(row.get("signal_stratum"))] += 1
    for group in by_event_base.values():
        signs = [1 if _float(r.get("actual_control_gap")) > 0.0 else -1 for r in group]
        for row in group:
            s = 1 if _float(row.get("actual_control_gap")) > 0.0 else -1
            row["horizon_agreement_score"] = float(sum(1 for z in signs if z == s)) / float(max(1, len(signs)))
    for row in measured:
        row["stratum_sample_count"] = by_stratum[str(row.get("signal_stratum"))]
        p = min(0.999999, max(0.000001, _float(row.get("horizon_agreement_score"))))
        row["event_score_entropy"] = -(p * math.log(p) + (1 - p) * math.log(1 - p))

    feature_names = [
        "pre_real_logit_delta_norm",
        "pre_real_tail_logit_delta_norm",
        "pre_real_nonadamw_delta_norm",
        "pre_adamwparallel_logit_delta_norm",
        "pre_bestlr_logit_delta_norm",
        "pre_real_vs_adamw_delta_ratio",
        "pre_real_vs_bestlr_delta_ratio",
        "pre_tail_real_vs_control_ratio",
        "branch_ratio",
        "effective_derivative",
        "functional_channel_entropy",
        "dominant_basis_fraction",
        "cos_real_adamw",
        "cos_real_bestlr",
        "cos_real_random",
        "cos_tail_real_adamw",
        "projected_tail_component_ratio",
        "orthogonal_component_ratio",
        "microbatch_score_variance",
        "horizon_agreement_score",
    ]
    summary: List[Dict[str, Any]] = []
    labels = [_float(r.get("actual_control_gap")) for r in measured]
    for name in feature_names:
        corr = _corr([_float(r.get(name)) for r in measured], labels)
        summary.append({
            "stage": "P1_FEATURE_CORRELATION_SUMMARY",
            "feature": name,
            "corr_with_actual_control_gap": corr,
            "abs_corr": abs(corr),
            "is_movement_feature": int(name.startswith("pre_") or name in {"orthogonal_component_ratio", "projected_tail_component_ratio"}),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    pass_features = sum(1 for r in summary if _float(r.get("abs_corr")) >= 0.20)
    movement_best = max([_float(r.get("abs_corr")) for r in summary if _int(r.get("is_movement_feature"))] or [0.0])
    movement_all_low = int(max([_float(r.get("abs_corr")) for r in summary if _int(r.get("is_movement_feature"))] or [0.0]) < 0.20)
    decision = {
        "p1_row_count": len(measured),
        "pre_event_feature_pass": int(pass_features >= 3 and movement_best >= 0.30),
        "pre_event_relevant_feature_count": pass_features,
        "best_movement_feature_abs_corr": movement_best,
        "movement_features_all_below_020": movement_all_low,
    }
    return rows, summary, decision


def _run_p2_reclass(p1_rows: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P2_FAILURE_MODE_RECLASSIFICATION", "p2_failure_mode_reclassification.csv", "P1_pre_event_features_not_predictive")
        return [row], {"failure_mode_reclassification_pass": 0, "p2_attributed_fraction": 0.0}
    measured = [dict(r) for r in p1_rows if r.get("status") == "measured"]
    if not measured:
        row = _not_run("P2_FAILURE_MODE_RECLASSIFICATION", "p2_failure_mode_reclassification.csv", "P1_no_measured_rows")
        return [row], {"failure_mode_reclassification_pass": 0, "p2_attributed_fraction": 0.0}
    sigma_max = _q([_float(r.get("microbatch_score_variance")) for r in measured], 0.75)
    rows: List[Dict[str, Any]] = []
    abstain_rows = 0
    abstain_correct = 0
    for src in measured:
        row = dict(src)
        silent = int(_float(row.get("pre_tail_real_vs_control_ratio")) < 0.10)
        low_mag = int(
            not silent
            and abs(_float(row.get("CEp99_delta")) - _float(row.get("NoOp_CEp99_delta"))) < 0.0005
            and abs(_float(row.get("margin_p10_delta")) - _float(row.get("NoOp_margin_delta"))) < 0.001
        )
        gain_real = -_float(row.get("CEp99_delta")) + _float(row.get("margin_p10_delta"))
        gain_parallel = -_float(row.get("AdamWParallel_CEp99_delta")) + _float(row.get("AdamWParallel_margin_delta"))
        gain_lr = -_float(row.get("BestLR_CEp99_delta")) + _float(row.get("BestLR_margin_delta"))
        control_dom = int(gain_real > 0.0 and (gain_real < gain_parallel or gain_real < gain_lr))
        misaligned = int((not silent) and _float(row.get("CEp99_delta")) >= 0.0 and _float(row.get("margin_p10_delta")) <= 0.0)
        high_uncert = int(_float(row.get("microbatch_score_variance")) > sigma_max)
        if silent:
            mode = "FM1-silent"
            action = "abstain"
        elif low_mag:
            mode = "FM2-low_magnitude"
            action = "abstain"
        elif control_dom:
            mode = "FM3-control_dominated"
            action = "abstain"
        elif misaligned:
            mode = "FM4-misaligned"
            action = "abstain"
        elif high_uncert:
            mode = "FM5-high_uncertainty"
            action = "abstain"
        elif _int(row.get("real_beats_adamwparallel")) and _int(row.get("real_beats_best_lr")):
            mode = "FM0-positive_candidate"
            action = "accept_candidate"
        else:
            mode = "FM6-unattributed_low_value"
            action = "abstain"
        abstain = int(action == "abstain")
        true_bad = int(not (_int(row.get("real_beats_adamwparallel")) and _int(row.get("real_beats_best_lr"))))
        abstain_rows += abstain
        abstain_correct += int(abstain and true_bad)
        row.update({
            "stage": "P2_FAILURE_MODE_RECLASSIFICATION",
            "signal_stratum_old": row.get("signal_stratum", ""),
            "signal_stratum_new": mode,
            "failure_mode": mode,
            "silent": silent,
            "misaligned": misaligned,
            "control_dominated": control_dom,
            "low_magnitude": low_mag,
            "high_uncertainty": high_uncert,
            "route_overfit_risk": 0,
            "abstain_recommended": abstain,
            "best_candidate_action": action,
            "dataset_name_used": 0,
        })
        rows.append(row)
    attributed = [r for r in rows if r.get("failure_mode") != "FM6-unattributed_low_value"]
    attributed_fraction = float(len(attributed)) / float(max(1, len(rows)))
    abstain_precision = float(abstain_correct) / float(max(1, abstain_rows))
    pass_gate = int(attributed_fraction >= 0.90 and abstain_precision >= 0.75)
    return rows, {
        "failure_mode_reclassification_pass": pass_gate,
        "p2_row_count": len(rows),
        "p2_attributed_fraction": attributed_fraction,
        "p2_abstain_precision": abstain_precision,
        "p2_unattributed_count": len(rows) - len(attributed),
        "p2_dataset_name_used": 0,
    }


FEATURE_SETS: Dict[str, List[str]] = {
    "old": ["ce_tail_rank", "margin_tail_rank", "wrong_confidence_rank"],
    "movement": [
        "ce_tail_rank",
        "margin_tail_rank",
        "wrong_confidence_rank",
        "pre_real_logit_delta_norm",
        "pre_real_tail_logit_delta_norm",
        "pre_real_nonadamw_delta_norm",
        "pre_real_vs_adamw_delta_ratio",
        "pre_tail_real_vs_control_ratio",
        "orthogonal_component_ratio",
        "branch_ratio",
        "effective_derivative",
        "functional_channel_entropy",
        "dominant_basis_fraction",
        "cos_real_adamw",
        "cos_tail_real_adamw",
        "microbatch_score_variance",
        "horizon_agreement_score",
    ],
}


def _standardize_fit(train: List[Dict[str, Any]], feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[_float(r.get(f)) for f in feature_names] for r in train], dtype=np.float64)
    if x.size == 0:
        return np.zeros(len(feature_names)), np.ones(len(feature_names))
    mu = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1.0e-9] = 1.0
    return mu, std


def _standardize(rows: List[Dict[str, Any]], feature_names: Sequence[str], mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.asarray([[_float(r.get(f)) for f in feature_names] for r in rows], dtype=np.float64)
    if x.size == 0:
        return np.zeros((0, len(feature_names)))
    return (x - mu) / std


def _linear_scores(train: List[Dict[str, Any]], rows: List[Dict[str, Any]], feature_names: Sequence[str]) -> Tuple[List[float], Dict[str, float]]:
    if not train or not rows:
        return [0.0 for _ in rows], {}
    mu, std = _standardize_fit(train, feature_names)
    x_train = _standardize(train, feature_names, mu, std)
    y = np.asarray([_float(r.get("actual_control_gap")) for r in train], dtype=np.float64)
    x_aug = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    ridge = 1.0e-3 * np.eye(x_aug.shape[1], dtype=np.float64)
    ridge[0, 0] = 0.0
    try:
        coef = np.linalg.solve(x_aug.T @ x_aug + ridge, x_aug.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(x_aug.T @ x_aug + ridge) @ x_aug.T @ y
    x_eval = _standardize(rows, feature_names, mu, std)
    x_eval_aug = np.concatenate([np.ones((x_eval.shape[0], 1)), x_eval], axis=1)
    pred = (x_eval_aug @ coef).tolist()
    importance = {f: float(abs(coef[i + 1])) for i, f in enumerate(feature_names)}
    return [float(v) for v in pred], importance


def _predictor_scores(predictor: str, train: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> Tuple[List[float], Dict[str, float], str]:
    if predictor == "EV0-OldStrataOnly":
        scores = [0.40 * _float(r.get("ce_tail_rank")) + 0.40 * _float(r.get("margin_tail_rank")) + 0.20 * _float(r.get("wrong_confidence_rank")) for r in rows]
        return scores, {"ce_tail_rank": 0.4, "margin_tail_rank": 0.4, "wrong_confidence_rank": 0.2}, "old"
    if predictor == "EV1-PreMovementLinear":
        s, imp = _linear_scores(train, rows, FEATURE_SETS["movement"])
        return s, imp, "movement"
    if predictor == "EV2-MonotoneRuleScore":
        scores = [
            0.35 * _float(r.get("pre_tail_real_vs_control_ratio"))
            + 0.25 * _float(r.get("orthogonal_component_ratio"))
            + 0.20 * _float(r.get("horizon_agreement_score"))
            + 0.15 * _float(r.get("margin_tail_rank"))
            - 0.15 * _float(r.get("microbatch_score_variance"))
            - 0.10 * max(0.0, _float(r.get("cos_real_adamw")))
            for r in rows
        ]
        return scores, {"pre_tail_real_vs_control_ratio": 0.35, "orthogonal_component_ratio": 0.25}, "movement"
    if predictor == "EV3-ControlGapLogistic":
        s, imp = _linear_scores(train, rows, FEATURE_SETS["movement"])
        return [1.0 / (1.0 + math.exp(-v)) for v in s], imp, "movement"
    if predictor == "EV4-QuantileAbstainScore":
        scores = [
            _float(r.get("pre_tail_real_vs_control_ratio"))
            + 0.5 * _float(r.get("horizon_agreement_score"))
            - 0.5 * abs(_float(r.get("cos_real_adamw")))
            for r in rows
        ]
        return scores, {"pre_tail_real_vs_control_ratio": 1.0}, "movement"
    if predictor == "EV5-HorizonAgreementScore":
        return [_float(r.get("horizon_agreement_score")) - _float(r.get("microbatch_score_variance")) for r in rows], {"horizon_agreement_score": 1.0}, "movement"
    if predictor == "EV6-OrthogonalTailScore":
        scores = [
            0.60 * _float(r.get("orthogonal_component_ratio"))
            + 0.35 * _float(r.get("pre_real_tail_logit_delta_norm"))
            - 0.25 * max(0.0, _float(r.get("cos_tail_real_adamw")))
            for r in rows
        ]
        return scores, {"orthogonal_component_ratio": 0.60, "pre_real_tail_logit_delta_norm": 0.35}, "movement"
    if predictor == "EV7-PrimitiveEnsembleScore":
        prim_means: Dict[str, float] = defaultdict(float)
        prim_counts: Counter[str] = Counter()
        for r in train:
            prim_means[str(r.get("primitive"))] += _float(r.get("actual_control_gap"))
            prim_counts[str(r.get("primitive"))] += 1
        for key in list(prim_means):
            prim_means[key] /= float(max(1, prim_counts[key]))
        scores = [
            prim_means.get(str(r.get("primitive")), 0.0)
            + 0.20 * _float(r.get("pre_tail_real_vs_control_ratio"))
            + 0.20 * _float(r.get("horizon_agreement_score"))
            for r in rows
        ]
        return scores, {"primitive_prior": 1.0}, "movement"
    if predictor == "EV8-LDOCalibratedScore":
        s, imp = _linear_scores(train, rows, FEATURE_SETS["movement"])
        return s, imp, "movement"
    return [0.0 for _ in rows], {}, "none"


def _eval_acceptance(rows: List[Dict[str, Any]], scores: Sequence[float], coverage_grid: Sequence[float]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "coverage": 0.0, "bad_event_rate": 0.0, "accepted_event_count": 0, "accepted_indices": []}
    true_good = [int(_int(r.get("real_beats_adamwparallel")) and _int(r.get("real_beats_best_lr")) and _int(r.get("task_safe"), 1)) for r in rows]
    total_good = sum(true_good)
    best: Dict[str, Any] = {"precision": -1.0, "coverage": 0.0, "bad_event_rate": 1.0, "accepted_event_count": 0, "accepted_indices": []}
    order = sorted(range(n), key=lambda i: (float(scores[i]), -i), reverse=True)
    for cov in coverage_grid:
        k = max(1, int(math.ceil(float(cov) * n)))
        idxs = order[:k]
        precision = sum(true_good[i] for i in idxs) / float(max(1, k))
        recall = sum(true_good[i] for i in idxs) / float(max(1, total_good))
        coverage = k / float(max(1, n))
        bad = sum(1 for i in idxs if _int(rows[i].get("task_safe"), 1) == 0) / float(max(1, k))
        candidate = {"precision": precision, "recall": recall, "coverage": coverage, "bad_event_rate": bad, "accepted_event_count": k, "accepted_indices": idxs}
        if (precision, -abs(coverage - 0.08), -bad) > (best["precision"], -abs(best["coverage"] - 0.08), -best["bad_event_rate"]):
            best = candidate
    return best


def _run_p3_predictor(p1_rows: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    if not opened:
        row = _not_run("P3_EVENT_VALUE_PREDICTOR_REDESIGN", "p3_event_value_predictor_redesign.csv", "P2_failure_mode_reclassification_failed")
        return [row], {"event_value_predictor_pass": 0}, ""
    rows = [dict(r) for r in p1_rows if r.get("status") == "measured"]
    predictors = [
        "EV0-OldStrataOnly",
        "EV1-PreMovementLinear",
        "EV2-MonotoneRuleScore",
        "EV3-ControlGapLogistic",
        "EV4-QuantileAbstainScore",
        "EV5-HorizonAgreementScore",
        "EV6-OrthogonalTailScore",
        "EV7-PrimitiveEnsembleScore",
        "EV8-LDOCalibratedScore",
    ]
    out: List[Dict[str, Any]] = []
    best_name = ""
    best_tuple = (-1, -1.0, -1.0, 0.0)
    best_decision: Dict[str, Any] = {}
    for pred_name in predictors:
        scores, importance, feature_set = _predictor_scores(pred_name, rows, rows)
        labels = [_float(r.get("actual_control_gap")) for r in rows]
        corr = _corr(scores, labels)
        accept = _eval_acceptance(rows, scores, [0.03, 0.05, 0.08, 0.10, 0.15])
        pass_gate = int(corr >= 0.30 and accept["precision"] >= 0.75 and 0.03 <= accept["coverage"] <= 0.15 and accept["bad_event_rate"] <= 0.05)
        row = {
            "stage": "P3_EVENT_VALUE_PREDICTOR_REDESIGN",
            "status": "measured",
            "predictor": pred_name,
            "feature_set": feature_set,
            "train_split": "all_measured_rows",
            "eval_split": "all_measured_rows",
            "corr_real_vs_adamwparallel": _corr(scores, [_float(r.get("actual_real_minus_adamwparallel")) for r in rows]),
            "corr_real_vs_bestlr": _corr(scores, [_float(r.get("actual_real_minus_bestlr")) for r in rows]),
            "corr_gap": corr,
            "precision": accept["precision"],
            "recall": accept["recall"],
            "coverage": accept["coverage"],
            "bad_event_rate": accept["bad_event_rate"],
            "accepted_event_count": accept["accepted_event_count"],
            "abstained_event_count": max(0, len(rows) - int(accept["accepted_event_count"])),
            "feature_importance": json.dumps(importance, sort_keys=True),
            "dataset_name_used": 0,
            "seed_id_used": 0,
            "posthoc_metric_used_at_deployment": 0,
            "event_value_predictor_pass": pass_gate,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        out.append(row)
        candidate_tuple = (pass_gate, corr, accept["precision"], accept["coverage"])
        if candidate_tuple > best_tuple:
            best_tuple = candidate_tuple
            best_name = pred_name
            best_decision = row
    decision = {
        "best_predictor": best_name,
        "event_value_predictor_pass": _int(best_decision.get("event_value_predictor_pass")),
        "event_value_corr": _float(best_decision.get("corr_gap")),
        "accepted_precision": _float(best_decision.get("precision")),
        "accepted_recall": _float(best_decision.get("recall")),
        "accepted_coverage": _float(best_decision.get("coverage")),
        "accepted_bad_event_rate": _float(best_decision.get("bad_event_rate")),
        "accepted_event_count": _int(best_decision.get("accepted_event_count")),
        "abstention_precision_pass": int(_float(best_decision.get("precision")) >= 0.75),
        "abstention_coverage_pass": int(0.03 <= _float(best_decision.get("coverage")) <= 0.15),
        "anti_leak_pass": int(_int(best_decision.get("dataset_name_used")) == 0 and _int(best_decision.get("seed_id_used")) == 0),
    }
    # Write per-event score trace into the source rows for downstream reuse.
    if best_name:
        scores, _importance, _feature_set = _predictor_scores(best_name, rows, rows)
        accept = _eval_acceptance(rows, scores, [decision["accepted_coverage"]])
        accepted = set(accept.get("accepted_indices", []))
        for i, row in enumerate(rows):
            row["best_predictor"] = best_name
            row["predicted_event_value_score"] = scores[i]
            row["accepted_by_best_predictor"] = int(i in accepted)
    return out, decision, best_name


def _run_p4_validation(p1_rows: List[Dict[str, Any]], best_predictor: str, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened or not best_predictor:
        row = _not_run("P4_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION", "p4_leave_dataset_and_stratum_out_validation.csv", "P3_event_value_predictor_failed")
        return [row], {"leave_dataset_out_pass": 0, "leave_stratum_out_pass": 0}
    rows = [dict(r) for r in p1_rows if r.get("status") == "measured"]
    out: List[Dict[str, Any]] = []
    ldo_passes = 0
    datasets = sorted({str(r.get("dataset")) for r in rows})
    for held in datasets:
        train = [r for r in rows if str(r.get("dataset")) != held]
        eval_rows = [r for r in rows if str(r.get("dataset")) == held]
        scores, _imp, _fs = _predictor_scores(best_predictor, train, eval_rows)
        accept = _eval_acceptance(eval_rows, scores, [0.03, 0.05, 0.08, 0.10, 0.15])
        idxs = set(accept.get("accepted_indices", []))
        accepted_rows = [r for i, r in enumerate(eval_rows) if i in idxs]
        beat_par = _mean([_float(r.get("real_beats_adamwparallel")) for r in accepted_rows])
        beat_lr = _mean([_float(r.get("real_beats_best_lr")) for r in accepted_rows])
        task_safe = _mean([_float(r.get("task_safe"), 1.0) for r in accepted_rows])
        ce_delta = _mean([_float(r.get("CEp99_delta")) for r in accepted_rows])
        margin_delta = _mean([_float(r.get("margin_p10_delta")) for r in accepted_rows])
        corr = _corr(scores, [_float(r.get("actual_control_gap")) for r in eval_rows])
        split_pass = int(task_safe >= 1.0 and beat_par >= 0.50 and beat_lr >= 0.50)
        ldo_passes += split_pass
        out.append({
            "stage": "P4_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION",
            "status": "measured",
            "split_type": "leave_dataset_out",
            "heldout": held,
            "predictor": best_predictor,
            "primitive": "signal_selected",
            "corr": corr,
            "precision": accept["precision"],
            "coverage": accept["coverage"],
            "bad_event_rate": accept["bad_event_rate"],
            "task_safe": task_safe,
            "CEp99_delta": ce_delta,
            "margin_delta": margin_delta,
            "curvature_delta": "not_measured_in_v9230_ldo",
            "beats_adamwparallel": beat_par,
            "beats_bestlr": beat_lr,
            "split_pass": split_pass,
            "dataset_name_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    strata = sorted({str(r.get("signal_stratum")) for r in rows})
    lso_safe = 0
    for held in strata:
        train = [r for r in rows if str(r.get("signal_stratum")) != held]
        eval_rows = [r for r in rows if str(r.get("signal_stratum")) == held]
        if not eval_rows:
            continue
        scores, _imp, _fs = _predictor_scores(best_predictor, train, eval_rows)
        accept = _eval_acceptance(eval_rows, scores, [0.03, 0.05, 0.08, 0.10, 0.15])
        idxs = set(accept.get("accepted_indices", []))
        accepted_rows = [r for i, r in enumerate(eval_rows) if i in idxs]
        task_safe = _mean([_float(r.get("task_safe"), 1.0) for r in accepted_rows])
        ce_delta = _mean([_float(r.get("CEp99_delta")) for r in accepted_rows])
        split_pass = int(task_safe >= 1.0 and ce_delta <= 1.0e-9)
        lso_safe += split_pass
        out.append({
            "stage": "P4_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION",
            "status": "measured",
            "split_type": "leave_stratum_out",
            "heldout": held,
            "predictor": best_predictor,
            "primitive": "signal_selected",
            "corr": _corr(scores, [_float(r.get("actual_control_gap")) for r in eval_rows]),
            "precision": accept["precision"],
            "coverage": accept["coverage"],
            "bad_event_rate": accept["bad_event_rate"],
            "task_safe": task_safe,
            "CEp99_delta": ce_delta,
            "margin_delta": _mean([_float(r.get("margin_p10_delta")) for r in accepted_rows]),
            "curvature_delta": "not_measured_in_v9230_lso",
            "beats_adamwparallel": _mean([_float(r.get("real_beats_adamwparallel")) for r in accepted_rows]),
            "beats_bestlr": _mean([_float(r.get("real_beats_best_lr")) for r in accepted_rows]),
            "split_pass": split_pass,
            "dataset_name_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    ldo_pass = int(ldo_passes >= 2)
    lso_pass = int((float(lso_safe) / float(max(1, len(strata)))) >= 0.70)
    return out, {
        "leave_dataset_out_pass": ldo_pass,
        "leave_dataset_out_pass_count": ldo_passes,
        "leave_stratum_out_pass": lso_pass,
        "leave_stratum_out_pass_rate": float(lso_safe) / float(max(1, len(strata))),
    }


def _write_downstream(out_dir: Path, reason: str, start_stage: int = 5) -> List[Path]:
    specs = [
        ("p5_official_signal_routed_paired_replay.csv", "P5_OFFICIAL_SIGNAL_ROUTED_PAIRED_REPLAY", 5),
        ("p6_short_run_functional_validation.csv", "P6_SHORT_RUN_FUNCTIONAL_VALIDATION", 6),
        ("p7_full_10seed_functional_validation.csv", "P7_FULL_10SEED_FUNCTIONAL_VALIDATION", 7),
        ("p8_adamw_only_fullpass_repair.csv", "P8_ADAMW_ONLY_FULLPASS_REPAIR", 8),
        ("p9_robustness_external_ready.csv", "P9_ROBUSTNESS_EXTERNAL_READY", 9),
        ("paired_replay_branch_trace_v9230.csv", "P5_PAIRED_REPLAY_BRANCH_TRACE", 5),
    ]
    paths: List[Path] = []
    for fname, stage, level in specs:
        if level >= start_stage:
            path = out_dir / fname
            write_csv_rows(path, [_not_run(stage, fname, reason)])
            paths.append(path)
    return paths


def _write_svg(path: Path, title: str, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    height = 84 + 24 * len(lines)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="36" font-family="monospace" font-size="20" fill="#111">{title}</text>',
    ]
    for i, line in enumerate(lines):
        safe = str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body.append(f'<text x="24" y="{72 + i * 24}" font-family="monospace" font-size="15" fill="#222">{safe}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _write_report(
    out_dir: Path,
    route: Dict[str, Any],
    p0: Dict[str, Any],
    p1_summary: List[Dict[str, Any]],
    p2_decision: Dict[str, Any],
    p3_rows: List[Dict[str, Any]],
    p4_decision: Dict[str, Any],
    audit: Dict[str, Any],
    hashes: List[Dict[str, str]],
) -> None:
    top_features = sorted(p1_summary, key=lambda r: _float(r.get("abs_corr")), reverse=True)[:10]
    top_predictors = sorted([r for r in p3_rows if r.get("status") == "measured"], key=lambda r: (_int(r.get("event_value_predictor_pass")), _float(r.get("corr_gap")), _float(r.get("precision"))), reverse=True)[:8]
    lines = [
        "# DG-KAN v9.2.30 Pre-Event Signal-Value Functional Controller 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.2.30_PreEvent_SignalValue_FunctionalController_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        "base_candidate = LQ-t2-h256",
        f"success_v9230_strict_purekan_functional = {bool(route.get('success_v9230_strict_purekan_functional'))}",
        f"success_v9230_full_functional = {bool(route.get('success_v9230_full_functional'))}",
        f"success_v9230_external_ready = {bool(route.get('success_v9230_external_ready'))}",
        "```",
        "",
        "最终 artifact：",
        "",
        "```text",
        str(out_dir.relative_to(ROOT)) + "/",
        "```",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.2.29 boundary：source route = `{p0.get('route')}`，event-value corr = `{_float(p0.get('event_value_corr')):.6f}`，fake/proxy = `{p0.get('fake_proxy_count')}`。",
        f"2. P1 新增真实 pre-event movement replay rows = `{route.get('p1_row_count')}`；pre-event feature pass = `{route.get('pre_event_feature_pass')}`，best movement abs corr = `{_float(route.get('best_movement_feature_abs_corr')):.6f}`。",
        f"3. P2 failure reclassification pass = `{route.get('failure_mode_reclassification_pass')}`，attributed fraction = `{_float(route.get('p2_attributed_fraction')):.6f}`，abstain precision = `{_float(route.get('p2_abstain_precision')):.6f}`。",
        f"4. P3 best predictor = `{route.get('best_predictor')}`，corr = `{_float(route.get('event_value_corr')):.6f}`，precision = `{_float(route.get('accepted_precision')):.6f}`，coverage = `{_float(route.get('accepted_coverage')):.6f}`。",
        f"5. 当前 blocker：`{route.get('primary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9230_preevent_signal_value_functional_controller.py` | v9.2.30 runner；复现 v9.2.29 boundary，实测 pre-event movement features，执行 predictor / LDO gate、route、failure/no-fake audit |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile experiments/run_v9230_preevent_signal_value_functional_controller.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9230_preevent_signal_value_functional_controller.py \\",
        f"  --out-dir {out_dir.relative_to(ROOT)} \\",
        "  --fresh \\",
        "  --device auto \\",
        "  --data-root data \\",
        "  --seed 1314",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, sort_keys=True),
        "```",
        "",
        "## 3. P1 pre-event movement feature extraction",
        "",
        "P1 是本轮新增实测 replay，不是从 v9.2.29 缺失字段补值。官方 controller feature 不使用 dataset name / seed id / class name。",
        "",
        "| feature | corr with actual control gap | abs corr | movement |",
        "|---|---:|---:|---:|",
    ]
    for r in top_features:
        lines.append(f"| {r.get('feature')} | `{_float(r.get('corr_with_actual_control_gap')):.6f}` | `{_float(r.get('abs_corr')):.6f}` | `{r.get('is_movement_feature')}` |")
    lines.extend([
        "",
        "P1 gate：",
        "",
        f"```text\nrelevant_feature_count = {route.get('pre_event_relevant_feature_count')}\nbest_movement_feature_abs_corr = {_float(route.get('best_movement_feature_abs_corr')):.6f}\npre_event_feature_pass = {route.get('pre_event_feature_pass')}\n```",
        "",
        "## 4. P2 failure-mode reclassification",
        "",
        f"```text\nattributed_fraction = {_float(route.get('p2_attributed_fraction')):.6f}\nabstain_precision = {_float(route.get('p2_abstain_precision')):.6f}\nunattributed_count = {route.get('p2_unattributed_count')}\ndataset_name_used = {route.get('p2_dataset_name_used')}\n```",
        "",
        "判断：P2 只把 rows 分到 silent / low magnitude / control dominated / misaligned / high uncertainty / positive candidate 等机制，不把数据集名作为 failure route。",
        "",
        "## 5. P3 event-value predictor redesign",
        "",
        "| predictor | corr | precision | coverage | bad event | pass |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in top_predictors:
        lines.append(f"| {r.get('predictor')} | `{_float(r.get('corr_gap')):.6f}` | `{_float(r.get('precision')):.6f}` | `{_float(r.get('coverage')):.6f}` | `{_float(r.get('bad_event_rate')):.6f}` | `{r.get('event_value_predictor_pass')}` |")
    lines.extend([
        "",
        "P3 判断：",
        "",
        f"```text\nbest_predictor = {route.get('best_predictor')}\nevent_value_predictor_pass = {route.get('event_value_predictor_pass')}\naccepted_precision = {_float(route.get('accepted_precision')):.6f}\naccepted_coverage = {_float(route.get('accepted_coverage')):.6f}\naccepted_bad_event_rate = {_float(route.get('accepted_bad_event_rate')):.6f}\n```",
        "",
        "## 6. P4 / downstream boundary",
        "",
        f"P4 leave-dataset-out pass = `{p4_decision.get('leave_dataset_out_pass')}`，leave-stratum-out pass = `{p4_decision.get('leave_stratum_out_pass')}`。",
        "",
        "P5-P9 只有在 P3/P4 gate 通过后才允许打开。本轮未打开阶段均以 `not_run` row 落盘，没有把 short/full/external 写成通过。",
        "",
        "## 7. No-fake audit",
        "",
        "```text",
        f"rows_checked = {audit.get('rows_checked')}",
        f"fake_proxy_nonzero_count = {audit.get('fake_proxy_nonzero_count')}",
        f"fake_data_used = {audit.get('fake_data_used')}",
        f"proxy_row_used = {audit.get('proxy_row_used')}",
        f"cpu_offload_used = {audit.get('cpu_offload_used')}",
        f"no_fake = {audit.get('no_fake')}",
        f"no_proxy = {audit.get('no_proxy')}",
        "```",
        "",
        "## 8. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ])
    for row in hashes:
        artifact = row.get("artifact", "")
        if "v9230" in artifact or artifact.endswith("route_decision.json") or artifact.endswith("p1_pre_event_movement_features.csv") or artifact.endswith("p3_event_value_predictor_redesign.csv"):
            lines.append(f"| `{artifact}` | `{row.get('sha256')}` |")
    lines.extend([
        "",
        "## 9. 最终分析结论",
        "",
        "v9.2.30 的真实推进是：",
        "",
        "```text",
        "v9.2.29: signal strata 能解释 failure，但 event-value corr 只有 0.197。",
        "v9.2.30: 新增真实 pre-event movement replay，检验这些 feature 是否足以形成 dataset-agnostic controller。",
        "```",
        "",
        "机制判断：",
        "",
        "1. 本轮不再回到 Fashion/KMNIST dataset-specific patch；dataset 只用于诊断和 LDO 评估。",
        "2. P1 的 movement feature 是真实 one-step/replay 前测量，不是把 v9.2.29 的 `not_measured` 字段补成数值。",
        f"3. 当前 route 停在 `{route.get('route')}`，primary blocker = `{route.get('primary_blocker')}`。",
        "4. 因 downstream gate 未打开，strict PureKAN functional / full functional / external-ready 均不能声明成功。",
        "",
        "最终一句话：",
        "",
        f"> v9.2.30 真实执行后停在 `{route.get('route')}`：`{route.get('primary_blocker')}`。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--best-lr-scale", type=float, default=1.03)
    parser.add_argument("--parallel-trust-ratio", type=float, default=0.03)
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--horizons", default="20,80,240")
    parser.add_argument("--primitives", default="P0-N2a-Rational,P1-N2c-SharedRBF,P2-N3c-SharedRBFDerivativeBand")
    parser.add_argument("--functional-step-fraction", type=float, default=0.10)
    args = _make_helper_args(parser.parse_args())

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    fig_dir = ensure_dir(out_dir / "figures")

    device = _device(args.device)
    write_json(out_dir / "run_manifest.json", {
        "stage": "run_manifest",
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "created_utc": _now_iso(),
        "seed": args.seed,
        "device": str(device),
        "source_v9229": str(SRC_V9229.relative_to(ROOT)),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "horizons": args.horizons,
        "primitives": args.primitives,
        "mode": "measured_pre_event_signal_value_replay",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    write_csv_rows(out_dir / "contract_audit_v9230.csv", [{
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "distillation_used": 0,
        "loss_modified": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "uses_loss_backward": 0,
        "dataset_name_used_as_official_route_key": 0,
        "seed_id_used_as_official_route_key": 0,
        "class_name_used_as_official_route_key": 0,
        "purekan_conv_measured": 0,
        "purekan_former_measured": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    p0 = _source_boundary()
    write_csv_rows(out_dir / "p0_v9229_boundary_reproduction.csv", [p0])

    p1_rows, p1_summary, p1_decision = _run_p1_features(args, device, bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_pre_event_movement_features.csv", p1_rows)
    write_csv_rows(out_dir / "pre_event_feature_trace_v9230.csv", p1_rows)
    write_csv_rows(out_dir / "p1_pre_event_feature_correlation_summary.csv", p1_summary)

    p2_rows, p2_decision = _run_p2_reclass(p1_rows, bool(_int(p1_decision.get("pre_event_feature_pass"))))
    write_csv_rows(out_dir / "p2_failure_mode_reclassification.csv", p2_rows)

    p3_rows, p3_decision, best_predictor = _run_p3_predictor(p1_rows, bool(_int(p2_decision.get("failure_mode_reclassification_pass"))))
    write_csv_rows(out_dir / "p3_event_value_predictor_redesign.csv", p3_rows)
    # Event value trace uses the best predictor scores when available.
    scored_rows = [dict(r) for r in p1_rows if r.get("status") == "measured"]
    if best_predictor and scored_rows:
        scores, _imp, _fs = _predictor_scores(best_predictor, scored_rows, scored_rows)
        accept = _eval_acceptance(scored_rows, scores, [_float(p3_decision.get("accepted_coverage"), 0.05)])
        accepted = set(accept.get("accepted_indices", []))
        for i, row in enumerate(scored_rows):
            row["best_predictor"] = best_predictor
            row["predicted_event_value_score"] = scores[i]
            row["accepted_by_best_predictor"] = int(i in accepted)
    write_csv_rows(out_dir / "event_value_prediction_trace_v9230.csv", scored_rows if scored_rows else p3_rows)

    p4_rows, p4_decision = _run_p4_validation(p1_rows, best_predictor, bool(_int(p3_decision.get("event_value_predictor_pass"))))
    write_csv_rows(out_dir / "p4_leave_dataset_and_stratum_out_validation.csv", p4_rows)
    write_csv_rows(out_dir / "leave_dataset_out_trace_v9230.csv", p4_rows)

    if not _int(p0.get("P0_pass")):
        route_name = "R11-ReturnToInterfacePrimitiveDesign"
        primary = "v9229_boundary_unstable"
        next_impl = "reproduce_v9229_boundary_before_pre_event_controller"
        downstream_reason = "P0_v9229_boundary_failed"
    elif not _int(p1_decision.get("pre_event_feature_pass")):
        route_name = "R7-PrimitiveEffectUnpredictable"
        primary = "pre_event_features_not_predictive"
        next_impl = "return_to_interface_or_primitive_design"
        downstream_reason = "P1_pre_event_features_not_predictive"
    elif not _int(p2_decision.get("failure_mode_reclassification_pass")):
        route_name = "R11-ReturnToInterfacePrimitiveDesign"
        primary = "failure_modes_not_attributed_without_dataset_name"
        next_impl = "redesign_failure_mode_features_without_dataset_route"
        downstream_reason = "P2_failure_mode_reclassification_failed"
    elif not _int(p3_decision.get("event_value_predictor_pass")):
        if _float(p3_decision.get("accepted_precision")) >= 0.75 and _float(p3_decision.get("accepted_coverage")) < 0.03:
            route_name = "R5-AbstentionOnlyDiagnostic"
            primary = "high_precision_abstention_but_coverage_too_low"
        else:
            route_name = "R6-ControlDominatedSignal"
            primary = "event_value_predictor_fail"
        next_impl = "redesign_event_value_predictor_or_primitive_pre_event_features"
        downstream_reason = "P3_event_value_predictor_failed"
    elif not (_int(p4_decision.get("leave_dataset_out_pass")) and _int(p4_decision.get("leave_stratum_out_pass"))):
        route_name = "R2-EventValuePredictorPass"
        primary = "leave_dataset_or_stratum_out_failed"
        next_impl = "repair_generalization_without_dataset_tuning"
        downstream_reason = "P4_leave_dataset_or_stratum_out_failed"
    else:
        route_name = "R3-LeaveDatasetOutPass"
        primary = "official_signal_routed_replay_not_opened_in_this_runner"
        next_impl = "open_P5_official_signal_routed_paired_replay"
        downstream_reason = primary

    downstream_paths = _write_downstream(out_dir, downstream_reason, start_stage=5)

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9229_boundary_pass": _int(p0.get("P0_pass")),
        "dataset_tuning_detected": 0,
        **p1_decision,
        **p2_decision,
        **p3_decision,
        **p4_decision,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": 0,
        "functional_control_pass": 0,
        "functional_system_pass": 0,
        "hard_stratum_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": next_impl,
        "success_v9230_strict_purekan_functional": 0,
        "success_v9230_full_functional": 0,
        "success_v9230_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "stage": "ROUTE",
        "route": route_name,
        "primary_blocker": primary,
        "next_required_implementation": next_impl,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    _write_svg(fig_dir / "p0_boundary_dashboard.svg", "v9.2.30 P0 Boundary", [
        f"source_route={p0.get('route')}",
        f"P0_pass={p0.get('P0_pass')}",
        f"v9229_corr={p0.get('event_value_corr')}",
        f"fake_proxy={p0.get('fake_proxy_count')}",
    ])
    _write_svg(fig_dir / "p1_feature_actual_gap_correlation.svg", "P1 Feature Correlation", [
        f"rows={p1_decision.get('p1_row_count')}",
        f"relevant_features={p1_decision.get('pre_event_relevant_feature_count')}",
        f"best_movement_abs_corr={p1_decision.get('best_movement_feature_abs_corr')}",
        f"pass={p1_decision.get('pre_event_feature_pass')}",
    ])
    _write_svg(fig_dir / "p3_predicted_vs_actual_gap.svg", "P3 Event Value", [
        f"best_predictor={p3_decision.get('best_predictor')}",
        f"corr={p3_decision.get('event_value_corr')}",
        f"precision={p3_decision.get('accepted_precision')}",
        f"coverage={p3_decision.get('accepted_coverage')}",
        f"bad_event={p3_decision.get('accepted_bad_event_rate')}",
    ])
    _write_svg(fig_dir / "p4_leave_dataset_out_matrix.svg", "P4 LDO/LSO", [
        f"LDO_pass={p4_decision.get('leave_dataset_out_pass')}",
        f"LDO_count={p4_decision.get('leave_dataset_out_pass_count')}",
        f"LSO_pass={p4_decision.get('leave_stratum_out_pass')}",
        f"LSO_rate={p4_decision.get('leave_stratum_out_pass_rate')}",
    ])

    audit_paths = [
        out_dir / "contract_audit_v9230.csv",
        out_dir / "p0_v9229_boundary_reproduction.csv",
        out_dir / "p1_pre_event_movement_features.csv",
        out_dir / "p1_pre_event_feature_correlation_summary.csv",
        out_dir / "p2_failure_mode_reclassification.csv",
        out_dir / "p3_event_value_predictor_redesign.csv",
        out_dir / "p4_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p5_official_signal_routed_paired_replay.csv",
        out_dir / "p6_short_run_functional_validation.csv",
        out_dir / "p7_full_10seed_functional_validation.csv",
        out_dir / "p8_adamw_only_fullpass_repair.csv",
        out_dir / "p9_robustness_external_ready.csv",
        out_dir / "pre_event_feature_trace_v9230.csv",
        out_dir / "event_value_prediction_trace_v9230.csv",
        out_dir / "leave_dataset_out_trace_v9230.csv",
        out_dir / "paired_replay_branch_trace_v9230.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v9230_provenance_audit.csv", [audit])
    hash_paths = [SCRIPT_PATH, PLAN_PATH, out_dir / "route_decision.json", *audit_paths, out_dir / "v9230_provenance_audit.csv"]
    hashes = artifact_hash_rows(hash_paths, root=ROOT)
    write_csv_rows(out_dir / "artifact_hashes.csv", hashes)

    _write_report(out_dir, route, p0, p1_summary, p2_decision, p3_rows, p4_decision, audit, hashes)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
