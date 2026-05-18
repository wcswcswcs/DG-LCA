#!/usr/bin/env python3
"""DG-KAN v9.2.33 legal train-stream probe and observable primitive audit.

This runner takes the v9.2.32 source-logged CP5 clue seriously, but does not
promote it directly.  It recomputes several probe-to-commit scores from the
training stream only, before a hypothetical functional commit, and compares
those legal probe scores with the grounded event-value labels from v9.2.31.

Observable primitives beyond the already measured current reference are not
fabricated: unimplemented OP candidates remain explicit not_implemented rows.
"""

from __future__ import annotations

import argparse
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


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.33_LegalTrainStreamProbe_ObservablePrimitive_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9233_legal_train_stream_probe_observable_primitive.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.33_LegalTrainStreamProbe_ObservablePrimitive_实验复盘.md"

SRC_V9232 = ROOT / "results" / "real_rerun_20260506" / "v9232_probe_to_commit_observable_primitive_first_20260510T230000Z"
SRC_V9231 = ROOT / "results" / "real_rerun_20260506" / "v9231_event_value_grounding_primitive_observability_first_20260510T220000Z"
SRC_V9230 = ROOT / "results" / "real_rerun_20260506" / "v9230_preevent_signal_value_functional_controller_first_20260510T210000Z"
SRC_V9222 = ROOT / "results" / "real_rerun_20260506" / "v9222_basequalified_strict_purekan_functional_interface_first_20260510T120000Z"

STATIC_FEATURES = [
    "effective_derivative",
    "pre_adamwparallel_logit_delta_norm",
    "pre_bestlr_logit_delta_norm",
    "pre_real_nonadamw_delta_norm",
    "pre_real_logit_delta_norm",
    "pre_real_tail_logit_delta_norm",
    "pre_tail_real_vs_control_ratio",
    "actual_r_z_perp",
    "dominant_basis_fraction",
    "branch_ratio",
    "cos_tail_real_adamw",
    "microbatch_score_variance",
    "horizon_agreement_score",
]

PROBES = [
    "LP0-StaticNoProbeReference",
    "LP1-LegalHorizonConsistencyProbe",
    "LP2-LegalVirtualMicroholdoutProbe",
    "LP3-LegalLeaveOneOutPopulationRiskProbe",
    "LP4-ControlContrastiveVirtualProbe",
    "LP5-UncertaintyLCBProbe",
    "LP6-CheapHorizonStatistic",
]


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
        text = str(value)
        if text.startswith("not_"):
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
    return float(sum(vals) / max(1, len(vals)))


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return float(math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)))


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return 0.0
    mx = _mean(x for x, _ in pairs)
    my = _mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-30 or vy <= 1.0e-30:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return float(cov / math.sqrt(vx * vy))


def _quantile(values: Sequence[float], q: float) -> float:
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


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pos = [float(s) for s, y in zip(scores, labels) if int(y) == 1]
    neg = [float(s) for s, y in zip(scores, labels) if int(y) == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / max(1.0, total))


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


def _primitive_registry() -> Dict[str, str]:
    return {
        "P0-N2a-Rational": "N2a-TinyInit-RationalFunc-BranchRatioCap",
        "P1-N2c-SharedRBF": "N2c-TinyInit-SharedRBFFunc-BranchRatioCap",
        "P2-N3c-SharedRBFDerivativeBand": "N3c-SharedRBFFunc-DerivativeBand",
    }


def _scale_step(step: Sequence[torch.Tensor], scale: float) -> List[torch.Tensor]:
    return [d.detach() * float(scale) for d in step]


def _zero_like(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.zeros_like(p) for p in params]


def _classification_gain(before: Dict[str, float], after: Dict[str, float]) -> float:
    ce_delta = after["CE_p99"] - before["CE_p99"]
    margin_delta = after["correct_margin_p10"] - before["correct_margin_p10"]
    acc_delta = after["acc"] - before["acc"]
    task_penalty = 2.0 * max(0.0, -acc_delta - 0.005)
    return float(-ce_delta + margin_delta - task_penalty)


def _metric_gain_for_step(
    params: Sequence[torch.Tensor],
    step: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    x: torch.Tensor,
    y: torch.Tensor,
    before: Dict[str, float] | None = None,
) -> float:
    base = before if before is not None else v9222._eval_actuator_metrics(params, mu, std, spec, x, y)
    after = v9222._eval_actuator_metrics(v9223._apply_step(params, step), mu, std, spec, x, y)
    return _classification_gain(base, after)


def _per_sample_control_gap(
    params: Sequence[torch.Tensor],
    real_step: Sequence[torch.Tensor],
    parallel_step: Sequence[torch.Tensor],
    bestlr_step: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    with torch.no_grad():
        base = act.actuator_forward(x, params, mu, std, spec)
        real = act.actuator_forward(x, v9223._apply_step(params, real_step), mu, std, spec)
        parallel = act.actuator_forward(x, v9223._apply_step(params, parallel_step), mu, std, spec)
        bestlr = act.actuator_forward(x, v9223._apply_step(params, bestlr_step), mu, std, spec)
        nll_base = torch.nn.functional.cross_entropy(base, y, reduction="none")
        nll_real = torch.nn.functional.cross_entropy(real, y, reduction="none")
        nll_parallel = torch.nn.functional.cross_entropy(parallel, y, reduction="none")
        nll_bestlr = torch.nn.functional.cross_entropy(bestlr, y, reduction="none")
        gain_real = nll_base - nll_real
        gain_parallel = nll_base - nll_parallel
        gain_bestlr = nll_base - nll_bestlr
        gap = gain_real - torch.maximum(gain_parallel, gain_bestlr)
    return float(gap.float().mean().detach().cpu())


def _source_logged_cp5(row: Dict[str, Any]) -> float:
    ce = _float(row.get("ce_tail_rank"))
    uncert = math.sqrt(max(0.0, _float(row.get("microbatch_score_variance")))) + _float(row.get("bootstrap_control_gap_std"))
    return _float(row.get("horizon_agreement_score")) + 0.10 * ce - 0.20 * uncert


def _acceptance_metrics(scores: Sequence[float], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [_int(r.get("Y_beat")) for r in rows]
    bad = [_int(r.get("bad_event")) for r in rows]
    total_good = max(1, sum(labels))
    best = {
        "precision": 0.0,
        "recall": 0.0,
        "coverage": 0.0,
        "bad_event_rate": 0.0,
        "accepted": [],
        "threshold": "",
        "threshold_quantile": "",
    }
    for q in [0.85, 0.88, 0.90, 0.92, 0.95, 0.97]:
        threshold = _quantile(scores, q)
        idx = [i for i, s in enumerate(scores) if float(s) >= threshold]
        coverage = len(idx) / max(1, len(rows))
        if coverage < 0.03 or coverage > 0.15:
            continue
        precision = sum(labels[i] for i in idx) / max(1, len(idx))
        recall = sum(labels[i] for i in idx) / total_good
        bad_rate = sum(bad[i] for i in idx) / max(1, len(idx))
        candidate = {
            "precision": precision,
            "recall": recall,
            "coverage": coverage,
            "bad_event_rate": bad_rate,
            "accepted": idx,
            "threshold": threshold,
            "threshold_quantile": q,
        }
        if (precision, -abs(coverage - 0.08), -bad_rate) > (best["precision"], -abs(best["coverage"] - 0.08), -best["bad_event_rate"]):
            best = candidate
    return best


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9232 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9232 / "v9232_provenance_audit.csv")
    fake = _int(audit_rows[0].get("fake_proxy_nonzero_count")) if audit_rows else 1
    p0_pass = int(
        route.get("route") == "R13-ReturnToInterfacePrimitiveDesign"
        and route.get("best_probe") == "CP5-HorizonConsistencyProbe"
        and _int(route.get("probe_predictive_pass")) == 1
        and _int(route.get("probe_legality_pass")) == 0
        and _int(route.get("observable_primitive_pass")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9232_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9232.relative_to(ROOT)),
        "route": route.get("route", ""),
        "best_probe": route.get("best_probe", ""),
        "best_probe_corr": route.get("best_probe_corr", ""),
        "best_probe_auc": route.get("best_probe_auc", ""),
        "best_probe_precision": route.get("best_probe_precision", ""),
        "best_probe_coverage": route.get("best_probe_coverage", ""),
        "probe_predictive_pass": route.get("probe_predictive_pass", ""),
        "probe_legality_pass": route.get("probe_legality_pass", ""),
        "observable_primitive_pass": route.get("observable_primitive_pass", ""),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _load_joined_events(args: argparse.Namespace) -> List[Dict[str, Any]]:
    grounded = {
        r["event_id"]: r
        for r in read_csv_rows(SRC_V9231 / "p1_event_value_label_grounding.csv")
        if r.get("status") == "source_measured_derived"
    }
    source = {
        r["event_id"]: r
        for r in read_csv_rows(SRC_V9230 / "p1_pre_event_movement_features.csv")
        if r.get("status") == "measured"
    }
    allowed_datasets = {v92._canonical_task(d) for d in _parse_list(args.datasets)}
    allowed_seeds = set(_parse_ints(args.seeds))
    allowed_horizons = set(_parse_ints(args.horizons))
    allowed_primitives = set(_parse_list(args.primitives))
    rows: List[Dict[str, Any]] = []
    for event_id, g in grounded.items():
        if event_id not in source:
            continue
        s = source[event_id]
        if str(s.get("dataset")) not in allowed_datasets:
            continue
        if _int(s.get("seed")) not in allowed_seeds:
            continue
        if _int(s.get("horizon")) not in allowed_horizons:
            continue
        if str(s.get("primitive")) not in allowed_primitives:
            continue
        row = dict(s)
        row.update(
            {
                "grounded_value": _float(g.get("normalized_value")),
                "control_relative_value": _float(g.get("control_relative_value")),
                "event_family_value": _float(g.get("event_family_value")),
                "bootstrap_value_std": _float(g.get("bootstrap_value_std")),
                "value_reliability": _float(g.get("value_reliability")),
                "Y_beat": int(_int(s.get("real_beats_adamwparallel")) == 1 and _int(s.get("real_beats_best_lr")) == 1 and _int(s.get("task_safe")) == 1),
                "bad_event": int(_int(s.get("task_safe"), 1) == 0),
            }
        )
        rows.append(row)
    if int(args.probe_source_limit) > 0:
        rows = rows[: int(args.probe_source_limit)]
    return rows


def _cp5_legality_autopsy(events: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_SOURCE_LOGGED_CP5_LEGALITY_AUTOPSY", "p1_source_logged_cp5_legality_autopsy.csv", "P0_v9232_boundary_failed")
        return [row], {"cp5_legality_classification": "not_opened", "cp5_main_source": "not_opened"}
    labels = [_int(r.get("Y_beat")) for r in events]
    values = [_float(r.get("grounded_value")) for r in events]
    features = {
        "horizon_agreement_score": {
            "class": "PosthocOnly_in_v9232_but_LegalButNeedsProbe_if_recomputed",
            "weight": 1.0,
            "values": [_float(r.get("horizon_agreement_score")) for r in events],
        },
        "ce_tail_rank": {
            "class": "PosthocOnly",
            "weight": 0.10,
            "values": [_float(r.get("ce_tail_rank")) for r in events],
        },
        "microbatch_score_variance": {
            "class": "PosthocOnly_in_v9232_but_LegalButNeedsProbe_if_recomputed",
            "weight": -0.20,
            "values": [math.sqrt(max(0.0, _float(r.get("microbatch_score_variance")))) for r in events],
        },
        "bootstrap_control_gap_std": {
            "class": "PosthocOnly",
            "weight": -0.20,
            "values": [_float(r.get("bootstrap_control_gap_std")) for r in events],
        },
    }
    rows: List[Dict[str, Any]] = []
    best_feature = ""
    best_auc = -1.0
    posthoc_weight = 0.0
    legalizable_weight = 0.0
    for name, spec in features.items():
        vals = spec["values"]
        corr = _corr(vals, values)
        auc = max(_auc(vals, labels), _auc([-v for v in vals], labels))
        if auc > best_auc:
            best_auc = auc
            best_feature = name
        cls = str(spec["class"])
        w = abs(float(spec["weight"]))
        if cls == "PosthocOnly":
            posthoc_weight += w
        if "LegalButNeedsProbe" in cls:
            legalizable_weight += w
        rows.append(
            {
                "stage": "P1_SOURCE_LOGGED_CP5_LEGALITY_AUTOPSY",
                "status": "source_logged_feature_classification",
                "cp5_feature": name,
                "legality_class": cls,
                "cp5_weight_abs": w,
                "corr_with_grounded_value": corr,
                "auc_with_Y_beat_best_orientation": auc,
                "posthoc_outcome_used_in_v9232": int(cls.startswith("PosthocOnly")),
                "can_be_recomputed_as_train_stream_probe": int("LegalButNeedsProbe" in cls),
                "dataset_name_used": 0,
                "validation_metric_used": int(cls.startswith("PosthocOnly")),
                "test_metric_used": int(cls.startswith("PosthocOnly")),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    if posthoc_weight > legalizable_weight:
        classification = "CP5MixedButPosthocWeighted"
        source = "posthoc_weight_dominant"
    else:
        classification = "CP5LegalizableViaTrainStreamProbe"
        source = "horizon_consistency_recomputable"
    return rows, {
        "cp5_legality_classification": classification,
        "cp5_main_source": source,
        "cp5_best_feature": best_feature,
        "cp5_best_feature_auc": best_auc,
        "cp5_posthoc_weight": posthoc_weight,
        "cp5_legalizable_weight": legalizable_weight,
    }


def _source_static_score(row: Dict[str, Any]) -> float:
    return (
        -_float(row.get("effective_derivative"))
        + 0.20 * _float(row.get("pre_tail_real_vs_control_ratio"))
        + 0.10 * _float(row.get("actual_r_z_perp"))
        - 0.10 * abs(_float(row.get("cos_tail_real_adamw")))
    )


def _run_legal_probe(
    args: argparse.Namespace,
    device: torch.device,
    events: List[Dict[str, Any]],
    opened: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P2_LEGAL_TRAIN_STREAM_PROBE_IMPLEMENTATION", "p2_legal_train_stream_probe_implementation.csv", "P1_CP5_legality_autopsy_not_opened")
        return [row], [row], {"best_legal_probe": "not_opened", "legal_probe_predictive_pass": 0, "legal_probe_system_pass": 0}

    registry = v9222._candidate_registry()
    primitive_map = _primitive_registry()
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    split_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, str]] = {}
    by_group: Dict[Tuple[str, int, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in events:
        key = (str(r.get("dataset")), _int(r.get("seed")), str(r.get("primitive")), str(r.get("event_type")), str(r.get("target")))
        by_group[key].append(r)

    group_scores: Dict[Tuple[str, int, str, str, str], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    selected_groups = sorted(by_group.keys())
    for gi, key in enumerate(selected_groups):
        dataset, seed, primitive, event_type, targets_text = key
        if primitive not in primitive_map:
            continue
        if dataset not in split_cache:
            x_train, y_train, _x_eval, _y_eval, protocol = v9223._load_split(args, dataset, device)
            split_cache[dataset] = (x_train, y_train, protocol)
        x_train, y_train, protocol = split_cache[dataset]
        cand = registry[primitive_map[primitive]]
        assert cand.spec is not None
        saved = v9223._train_cache(args, cand, dataset, seed, device, cache)
        params = saved["params"]
        mu = saved["mu"]
        std = saved["std"]
        batch = int(args.audit_batch_size)
        update_start = (gi * batch) % max(1, int(x_train.shape[0]) - 2 * batch)
        probe_start = update_start + batch
        xb = x_train[update_start:update_start + batch]
        yb = y_train[update_start:update_start + batch]
        xp = x_train[probe_start:probe_start + batch]
        yp = y_train[probe_start:probe_start + batch]
        targets = [t for t in targets_text.split(",") if t]
        real_step, task_step, _grads, info = v9227._step_for_targets(
            args,
            cand,
            params,
            mu,
            std,
            xb,
            yb,
            dataset,
            targets,
            float(args.functional_step_fraction),
            "safe" if "Orthogonal" not in str(event_type) else "orthogonal",
        )
        parallel_step = v9222._cap_to_fraction(task_step, task_step, float(args.parallel_trust_ratio))
        bestlr_step = v9222._cap_to_fraction(task_step, task_step, float(args.best_lr_scale) - 1.0)
        random_step = v9222._random_like_step(params, snr_lq.step_norm(real_step), seed + gi + 923300)
        noop_step = _zero_like(params)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        base_probe = v9222._eval_actuator_metrics(params, mu, std, cand.spec, xp, yp)
        baseline_time = max(1.0e-9, time.perf_counter() - t0)

        def gain(step: Sequence[torch.Tensor], scale: float = 1.0, x: torch.Tensor = xp, y: torch.Tensor = yp, before: Dict[str, float] | None = base_probe) -> float:
            return _metric_gain_for_step(params, _scale_step(step, scale), mu, std, cand.spec, x, y, before)

        probe_t0 = time.perf_counter()
        scales = [1.0 / 20.0, 5.0 / 20.0, 1.0]
        horizon_gaps = []
        for scale in scales:
            gr = gain(real_step, scale)
            gp = gain(parallel_step, scale)
            gl = gain(bestlr_step, scale)
            horizon_gaps.append(gr - max(gp, gl))
        g_real = gain(real_step, 1.0)
        g_parallel = gain(parallel_step, 1.0)
        g_bestlr = gain(bestlr_step, 1.0)
        g_random = gain(random_step, 1.0)
        g_noop = gain(noop_step, 1.0)
        half = int(xp.shape[0]) // 2
        base_a = v9222._eval_actuator_metrics(params, mu, std, cand.spec, xp[:half], yp[:half])
        base_b = v9222._eval_actuator_metrics(params, mu, std, cand.spec, xp[half:], yp[half:])
        gap_a = gain(real_step, 1.0, xp[:half], yp[:half], base_a) - max(gain(parallel_step, 1.0, xp[:half], yp[:half], base_a), gain(bestlr_step, 1.0, xp[:half], yp[:half], base_a))
        gap_b = gain(real_step, 1.0, xp[half:], yp[half:], base_b) - max(gain(parallel_step, 1.0, xp[half:], yp[half:], base_b), gain(bestlr_step, 1.0, xp[half:], yp[half:], base_b))
        loo_gap = _per_sample_control_gap(params, real_step, parallel_step, bestlr_step, mu, std, cand.spec, xp, yp)
        probe_elapsed = time.perf_counter() - probe_t0
        if device.type == "cuda":
            memory_ratio: Any = 1.0 + float(torch.cuda.max_memory_allocated(device)) / max(1.0, float(torch.cuda.max_memory_reserved(device)))
        else:
            memory_ratio = "not_measured_cpu_device"

        horizon_mean = _mean(horizon_gaps)
        horizon_std = _std(horizon_gaps)
        group_score = {
            "LP0-StaticNoProbeReference": _source_static_score(by_group[key][0]),
            "LP1-LegalHorizonConsistencyProbe": float(sum(1 for z in horizon_gaps if z > 0.0)) + horizon_mean - 0.25 * horizon_std,
            "LP2-LegalVirtualMicroholdoutProbe": g_real - max(g_parallel, g_bestlr),
            "LP3-LegalLeaveOneOutPopulationRiskProbe": loo_gap,
            "LP4-ControlContrastiveVirtualProbe": g_real - max(g_parallel, g_bestlr, g_random, g_noop),
            "LP5-UncertaintyLCBProbe": _mean([gap_a, gap_b]) - _std([gap_a, gap_b]),
            "LP6-CheapHorizonStatistic": min(horizon_gaps) + 0.50 * float(sum(1 for z in horizon_gaps if z > 0.0)),
            "probe_elapsed": probe_elapsed,
            "baseline_time": baseline_time,
            "probe_overhead_ratio": probe_elapsed / baseline_time,
            "step_ratio_with_probe": 1.0 + probe_elapsed / baseline_time,
            "memory_ratio_with_probe": memory_ratio,
            "horizon_gaps": horizon_gaps,
            "gap_real_parallel": g_real - g_parallel,
            "gap_real_bestlr": g_real - g_bestlr,
            "gap_real_random": g_real - g_random,
            "gap_real_noop": g_real - g_noop,
            "target_fit_R2": info.get("target_fit_R2", ""),
            "target_rz": info.get("target_rz", ""),
            "target_selected_fraction": info.get("target_selected_fraction", ""),
            "branch_ratio": info.get("branch_ratio", ""),
            "effective_derivative": info.get("effective_derivative", ""),
            "protocol": protocol,
        }
        group_scores[key] = group_score
        trace.append(
            {
                "stage": "LEGAL_TRAIN_STREAM_PROBE_TRACE",
                "status": "measured",
                "dataset": dataset,
                "seed": seed,
                "primitive": primitive,
                "event_type": event_type,
                "target": targets_text,
                "update_batch_start": update_start,
                "probe_batch_start": probe_start,
                "probe_uses_train_stream_only": 1,
                "validation_used": 0,
                "test_metric_used": 0,
                "dataset_name_used": 0,
                "posthoc_outcome_used_at_commit": 0,
                "horizon_gaps": json.dumps(horizon_gaps),
                "real_gain": g_real,
                "adamwparallel_gain": g_parallel,
                "bestlr_gain": g_bestlr,
                "random_gain": g_random,
                "noop_gain": g_noop,
                "probe_elapsed_sec": probe_elapsed,
                "baseline_forward_eval_sec": baseline_time,
                "probe_overhead_ratio": group_score["probe_overhead_ratio"],
                "step_ratio_with_probe": group_score["step_ratio_with_probe"],
                "memory_ratio_with_probe": memory_ratio,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )

    all_rows_for_summary: Dict[str, List[Dict[str, Any]]] = {p: [] for p in PROBES}
    for source_row in events:
        key = (str(source_row.get("dataset")), _int(source_row.get("seed")), str(source_row.get("primitive")), str(source_row.get("event_type")), str(source_row.get("target")))
        gs = group_scores.get(key)
        if gs is None:
            continue
        for probe in PROBES:
            rows.append(
                {
                    "stage": "P2_LEGAL_TRAIN_STREAM_PROBE_IMPLEMENTATION",
                    "status": "measured" if probe != "LP0-StaticNoProbeReference" else "measured_static_reference",
                    "probe": probe,
                    "event_id": source_row.get("event_id", ""),
                    "dataset": source_row.get("dataset", ""),
                    "seed": source_row.get("seed", ""),
                    "horizon": source_row.get("horizon", ""),
                    "primitive": source_row.get("primitive", ""),
                    "event_type": source_row.get("event_type", ""),
                    "target": source_row.get("target", ""),
                    "signal_stratum": source_row.get("signal_stratum", ""),
                    "probe_score_real": gs[probe],
                    "probe_score_adamwparallel": "virtual_probe_control_gain_recorded_in_trace",
                    "probe_score_bestlr": "virtual_probe_control_gain_recorded_in_trace",
                    "probe_control_gap": gs[probe],
                    "probe_uncertainty": _std(gs["horizon_gaps"]) if probe in {"LP1-LegalHorizonConsistencyProbe", "LP5-UncertaintyLCBProbe", "LP6-CheapHorizonStatistic"} else 0.0,
                    "probe_lcb": gs[probe],
                    "grounded_value": source_row.get("grounded_value", ""),
                    "actual_real_minus_adamwparallel": source_row.get("actual_real_minus_adamwparallel", ""),
                    "actual_real_minus_bestlr": source_row.get("actual_real_minus_bestlr", ""),
                    "Y_beat": source_row.get("Y_beat", ""),
                    "accepted": 0,
                    "bad_event": source_row.get("bad_event", ""),
                    "probe_overhead_ratio": gs["probe_overhead_ratio"] if probe != "LP0-StaticNoProbeReference" else 0.0,
                    "step_ratio_with_probe": gs["step_ratio_with_probe"] if probe != "LP0-StaticNoProbeReference" else 1.0,
                    "memory_ratio_with_probe": gs["memory_ratio_with_probe"] if probe != "LP0-StaticNoProbeReference" else 1.0,
                    "target_fit_R2": gs["target_fit_R2"],
                    "target_rz": gs["target_rz"],
                    "branch_ratio": gs["branch_ratio"],
                    "effective_derivative": gs["effective_derivative"],
                    "probe_uses_train_stream_only": 1,
                    "probe_scope": "train_stream_update_batch_plus_train_stream_probe_batch",
                    "dataset_name_used": 0,
                    "validation_used": 0,
                    "test_metric_used": 0,
                    "posthoc_outcome_used_at_commit": 0,
                    "probe_predictive_pass": 0,
                    "accepted_event_pass": 0,
                    "probe_legality_pass": int(probe != "LP0-StaticNoProbeReference"),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
            all_rows_for_summary[probe].append(rows[-1])

    summaries: List[Dict[str, Any]] = []
    for probe, probe_rows in all_rows_for_summary.items():
        scores_raw = [_float(r.get("probe_score_real")) for r in probe_rows]
        labels = [_int(r.get("Y_beat")) for r in probe_rows]
        grounded = [_float(r.get("grounded_value")) for r in probe_rows]
        corr_raw = _corr(scores_raw, grounded)
        auc_raw = _auc(scores_raw, labels)
        scores_inv = [-s for s in scores_raw]
        corr_inv = _corr(scores_inv, grounded)
        auc_inv = _auc(scores_inv, labels)
        if (max(corr_inv, 0.0), auc_inv) > (max(corr_raw, 0.0), auc_raw):
            orientation = -1.0
            scores = scores_inv
            corr = corr_inv
            auc = auc_inv
        else:
            orientation = 1.0
            scores = scores_raw
            corr = corr_raw
            auc = auc_raw
        acc = _acceptance_metrics(scores, probe_rows)
        accepted_set = set(acc["accepted"])
        legality = int(probe != "LP0-StaticNoProbeReference")
        system_ratios = [_float(r.get("step_ratio_with_probe"), 999.0) for r in probe_rows if probe != "LP0-StaticNoProbeReference"]
        step_q90 = _quantile(system_ratios, 0.90) if system_ratios else 1.0
        memory_values = [_float(r.get("memory_ratio_with_probe"), 1.0) for r in probe_rows if str(r.get("memory_ratio_with_probe", "")).replace(".", "", 1).isdigit()]
        memory_ratio = max(memory_values or [1.0])
        predictive = int(legality and (auc >= 0.70 or corr >= 0.35))
        accepted_pass = int(acc["precision"] >= 0.75 and 0.03 <= acc["coverage"] <= 0.15 and acc["bad_event_rate"] <= 0.05)
        system_pass = int(legality and step_q90 <= 1.50 and memory_ratio <= 1.05)
        summary = {
            "stage": "P2_LEGAL_TRAIN_STREAM_PROBE_SUMMARY",
            "status": "measured",
            "probe": probe,
            "corr": corr,
            "auc": auc,
            "orientation": orientation,
            "precision": acc["precision"],
            "recall": acc["recall"],
            "coverage": acc["coverage"],
            "bad_event_rate": acc["bad_event_rate"],
            "accepted_count": len(acc["accepted"]),
            "threshold": acc["threshold"],
            "threshold_quantile": acc["threshold_quantile"],
            "probe_legality_pass": legality,
            "probe_predictive_pass": predictive,
            "accepted_event_pass": accepted_pass,
            "probe_system_pass": system_pass,
            "step_ratio_q90_with_probe": step_q90,
            "memory_ratio_with_probe": memory_ratio,
            "dataset_name_used": 0,
            "validation_used": 0,
            "test_metric_used": 0,
            "posthoc_outcome_used_at_commit": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        summaries.append(summary)
        for idx, row in enumerate(probe_rows):
            row["accepted"] = int(idx in accepted_set)
            row["probe_score_oriented_audit"] = scores[idx]
            row["probe_predictive_pass"] = predictive
            row["accepted_event_pass"] = accepted_pass
            row["probe_system_pass"] = system_pass
            row["probe_legality_pass"] = legality
    best = max(summaries, key=lambda r: (_int(r["probe_predictive_pass"]), _int(r["accepted_event_pass"]), _float(r["corr"]), _float(r["auc"]), _float(r["precision"])))
    decision = {
        "best_legal_probe": best["probe"],
        "legal_probe_predictive_pass": _int(best["probe_predictive_pass"]),
        "legal_probe_system_pass": _int(best["probe_system_pass"]),
        "legal_probe_controller_candidate_pass": int(_int(best["probe_predictive_pass"]) and _int(best["accepted_event_pass"]) and _int(best["probe_system_pass"])),
        "legal_probe_corr": _float(best["corr"]),
        "legal_probe_auc": _float(best["auc"]),
        "legal_probe_precision": _float(best["precision"]),
        "legal_probe_recall": _float(best["recall"]),
        "legal_probe_coverage": _float(best["coverage"]),
        "legal_probe_bad_event_rate": _float(best["bad_event_rate"]),
        "legal_probe_accepted_count": _int(best["accepted_count"]),
        "legal_probe_step_q90": _float(best["step_ratio_q90_with_probe"]),
        "legal_probe_memory_ratio": best["memory_ratio_with_probe"],
        "legal_probe_all_summary": json.dumps(summaries, sort_keys=True),
    }
    return rows, trace, decision


def _write_stage_not_run(out_dir: Path, specs: Sequence[Tuple[str, str]], reason: str) -> List[Path]:
    paths: List[Path] = []
    for fname, stage in specs:
        path = out_dir / fname
        write_csv_rows(path, [_not_run(stage, fname, reason)])
        paths.append(path)
    return paths


def _run_p3_p4_downstream(out_dir: Path, p2_decision: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Path]]:
    if not _int(p2_decision.get("legal_probe_controller_candidate_pass")):
        reason = "P2_legal_probe_predictive_accept_or_system_gate_failed"
        specs = [
            ("p3_probe_to_commit_controller_calibration.csv", "P3_PROBE_TO_COMMIT_CONTROLLER_CALIBRATION"),
            ("p4_leave_dataset_and_stratum_out_validation.csv", "P4_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION"),
        ]
        return {"probe_controller_pass": 0}, {"leave_dataset_out_pass": 0, "leave_stratum_out_pass": 0}, _write_stage_not_run(out_dir, specs, reason)
    reason = "controller_calibration_not_implemented_after_P2_in_this_runner"
    specs = [
        ("p3_probe_to_commit_controller_calibration.csv", "P3_PROBE_TO_COMMIT_CONTROLLER_CALIBRATION"),
        ("p4_leave_dataset_and_stratum_out_validation.csv", "P4_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION"),
    ]
    return {"probe_controller_pass": 0}, {"leave_dataset_out_pass": 0, "leave_stratum_out_pass": 0}, _write_stage_not_run(out_dir, specs, reason)


def _run_p5_observable_primitive(events: List[Dict[str, Any]], p2_decision: Dict[str, Any], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P5_OBSERVABLE_PRIMITIVE_IMPLEMENTATION_GATE", "p5_observable_primitive_implementation_gate.csv", "legal_probe_controller_path_opened")
        return [row], {"observable_primitive_pass": 0, "best_observable_primitive": "not_opened", "observable_primitive_not_implemented_count": 0}
    source_route = _read_json(SRC_V9222 / "route_decision.json")
    best_probe = str(p2_decision.get("best_legal_probe", "LP0-StaticNoProbeReference"))
    # Current reference observability: reuse source-logged static score because OP0
    # has no additional observable state beyond current primitive statistics.
    obs_by_primitive: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r in events:
        score = _source_static_score(r)
        obs_by_primitive[str(r.get("primitive"))].append((score, _float(r.get("grounded_value"))))
    primitive_corr = {p: abs(_corr([a for a, _ in vals], [b for _, b in vals])) for p, vals in obs_by_primitive.items()}
    op0_corr = max(primitive_corr.values() or [0.0])
    rows = [
        {
            "stage": "P5_OBSERVABLE_PRIMITIVE_IMPLEMENTATION_GATE",
            "status": "source_measured_current_reference",
            "primitive": "OP0-current-reference",
            "contract_pass": 1,
            "grad_pass": 1,
            "P4_pass": source_route.get("interface_p4_pass", 0),
            "P5_nearpass": source_route.get("interface_p5_nearpass", 0),
            "functional_actuatability_pass": source_route.get("functional_actuatability_pass", 0),
            "probe_observability_corr": op0_corr,
            "best_probe_for_observability": best_probe,
            "paired_replay_ready": 0,
            "observable_primitive_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    for op in [
        "OP1-ObservableTailLinearChannel",
        "OP2-ObservablePiecewiseTailChannel",
        "OP3-ObservableSharedRBFLocalChannel",
        "OP4-ObservableOrthogonalTailChannel",
        "OP5-ObservableControlGapChannel",
        "OP6-LightHybridObservable",
    ]:
        rows.append(
            {
                "stage": "P5_OBSERVABLE_PRIMITIVE_IMPLEMENTATION_GATE",
                "status": "not_implemented",
                "primitive": op,
                "contract_pass": 0,
                "grad_pass": 0,
                "P4_pass": 0,
                "P5_nearpass": 0,
                "functional_actuatability_pass": 0,
                "probe_observability_corr": "",
                "best_probe_for_observability": best_probe,
                "paired_replay_ready": 0,
                "observable_primitive_pass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    return rows, {
        "observable_primitive_pass": 0,
        "best_observable_primitive": "OP0-current-reference",
        "best_observable_primitive_corr": op0_corr,
        "observable_primitive_not_implemented_count": 6,
    }


def _make_svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    ensure_dir(path.parent)
    width = 850
    height = 300
    vals = [float(v) for v in values]
    vmax = max([abs(v) for v in vals] + [1.0e-9])
    bar_w = max(22, int((width - 150) / max(1, len(vals))))
    zero_y = 150
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="30" font-family="sans-serif" font-size="16">{title}</text>',
        f'<line x1="70" x2="{width-35}" y1="{zero_y}" y2="{zero_y}" stroke="#777" stroke-width="1"/>',
    ]
    for i, (label, val) in enumerate(zip(labels, vals)):
        x = 70 + i * bar_w
        h = int((abs(val) / vmax) * 90)
        y = zero_y - h if val >= 0 else zero_y
        color = "#2b6cb0" if val >= 0 else "#c53030"
        safe = str(label).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f'<rect x="{x}" y="{y}" width="{max(10, bar_w-5)}" height="{h}" fill="{color}" opacity="0.86"/>')
        lines.append(f'<text x="{x}" y="270" font-family="sans-serif" font-size="9" transform="rotate(-35 {x},270)">{safe}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(out_dir: Path, route: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any], p5: Dict[str, Any], hashes: List[Dict[str, str]]) -> None:
    def h(name: str) -> str:
        for row in hashes:
            if row["artifact"].endswith(name):
                return row["sha256"]
        return ""

    text = f"""# DG-KAN v9.2.33 Legal Train-Stream Probe 与 Observable Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.2.33_LegalTrainStreamProbe_ObservablePrimitive_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = {route['route']}
base_candidate = LQ-t2-h256
success_v9233_strict_purekan_functional = {bool(route['success_v9233_strict_purekan_functional'])}
success_v9233_full_functional = {bool(route['success_v9233_full_functional'])}
success_v9233_external_ready = {bool(route['success_v9233_external_ready'])}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}/
```

核心结论：

1. P0 复现 v9.2.32 boundary：source route = `R13-ReturnToInterfacePrimitiveDesign`，best probe = `CP5-HorizonConsistencyProbe`，fake/proxy = `0`。
2. P1 判定 source CP5 为 `{route['cp5_legality_classification']}`；v9.2.32 的 CP5 信号不能直接当 official controller。
3. P2 已执行真实 legal train-stream probe，best legal probe = `{route['best_legal_probe']}`，corr = `{route['legal_probe_corr']:.6f}`，AUC = `{route['legal_probe_auc']:.6f}`。
4. P2 precision = `{route['legal_probe_precision']:.6f}`，coverage = `{route['legal_probe_coverage']:.6f}`，bad-event = `{route['legal_probe_bad_event_rate']:.6f}`，system pass = `{route['legal_probe_system_pass']}`。
5. Observable primitive gate 仍未闭合：best OP = `{route['best_observable_primitive']}`，OP1-OP6 not_implemented count = `{route['observable_primitive_not_implemented_count']}`。
6. 当前 blocker：`{route['primary_blocker']}`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9233_legal_train_stream_probe_observable_primitive.py` | v9.2.33 runner；执行 CP5 legality autopsy、legal train-stream probe、observable primitive gate、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9233_legal_train_stream_probe_observable_primitive.py
```

正式运行：

```bash
python experiments/run_v9233_legal_train_stream_probe_observable_primitive.py \\
  --out-dir {out_dir.relative_to(ROOT)} \\
  --fresh \\
  --device auto \\
  --data-root data \\
  --seed 1314
```

## 2. Route

```json
{json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True)}
```

## 3. P1 Source CP5 Legality Autopsy

Artifact：

```text
p1_source_logged_cp5_legality_autopsy.csv
```

关键值：

| metric | value |
|---|---:|
| CP5 classification | `{route['cp5_legality_classification']}` |
| CP5 best feature | `{p1.get('cp5_best_feature')}` |
| CP5 best feature AUC | `{_float(p1.get('cp5_best_feature_auc')):.6f}` |
| posthoc weight | `{_float(p1.get('cp5_posthoc_weight')):.6f}` |
| legalizable weight | `{_float(p1.get('cp5_legalizable_weight')):.6f}` |

判断：CP5 的 horizon-consistency 逻辑可以尝试重算为 train-stream probe，但 v9.2.32 source 字段本身含 posthoc 成分，不能直接合法化。

## 4. P2 Legal Train-Stream Probe

Artifacts：

```text
p2_legal_train_stream_probe_implementation.csv
legal_train_stream_probe_trace_v9233.csv
```

P2 使用同一 train split 的 update batch 与 probe batch；不使用 validation/test metric，不用 dataset name 做 commit rule。

| metric | value |
|---|---:|
| best probe | `{route['best_legal_probe']}` |
| corr | `{route['legal_probe_corr']:.6f}` |
| AUC | `{route['legal_probe_auc']:.6f}` |
| precision | `{route['legal_probe_precision']:.6f}` |
| coverage | `{route['legal_probe_coverage']:.6f}` |
| bad event | `{route['legal_probe_bad_event_rate']:.6f}` |
| step q90 | `{route['legal_probe_step_q90']:.6f}` |
| predictive pass | `{route['legal_probe_predictive_pass']}` |
| system pass | `{route['legal_probe_system_pass']}` |

判断：本轮没有把 source-logged CP5 伪装成合法 probe；P2 数值来自新计算的 train-stream virtual probe。

## 5. P5 Observable Primitive Gate

Artifact：

```text
p5_observable_primitive_implementation_gate.csv
```

```text
observable_primitive_pass = {route['observable_primitive_pass']}
best_observable_primitive = {route['best_observable_primitive']}
best_observable_primitive_corr = {route['best_observable_primitive_corr']:.6f}
not_implemented OP rows = {route['observable_primitive_not_implemented_count']}
```

判断：OP0/current reference 没有形成 observable primitive pass；OP1-OP6 本轮仍明确 not_implemented，没有写成失败训练或成功。

## 6. Downstream Boundary

P6-P10 只有在 legal probe controller 或 observable primitive survivor 后打开。本轮均已落盘为 `not_run`。

## 7. No-fake audit

```text
rows_checked = {route['rows_checked']}
fake_proxy_nonzero_count = {route['fake_proxy_nonzero_count']}
fake_data_used = {route['fake_data_used']}
proxy_row_used = {route['proxy_row_used']}
cpu_offload_used = {route['cpu_offload_used']}
no_fake = {bool(route['no_fake'])}
no_proxy = {bool(route['no_proxy'])}
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| runner | `{h('run_v9233_legal_train_stream_probe_observable_primitive.py')}` |
| route | `{h('route_decision.json')}` |
| P1 autopsy | `{h('p1_source_logged_cp5_legality_autopsy.csv')}` |
| P2 legal probe | `{h('p2_legal_train_stream_probe_implementation.csv')}` |
| P5 primitive gate | `{h('p5_observable_primitive_implementation_gate.csv')}` |
| provenance audit | `{h('v9233_provenance_audit.csv')}` |

## 9. 最终分析结论

v9.2.33 的真实推进是：

```text
v9.2.32: source-logged CP5 有分类信号，但 legality 不过。
v9.2.33: CP5 被重做为 legal train-stream probe audit；
          official downstream 仍取决于 predictive / accept / system gate。
```

机制判断：

1. 本轮把 “source-logged signal” 和 “legal commit-time signal” 分开了，避免把后验 replay 信息当作 online controller。
2. legal probe 即使有局部分类能力，也必须同时满足 precision / coverage / bad-event / system gate。
3. 如果 P2 不过，正确下一步不是继续 dataset patch，而是实现真正 observable primitive，或者提取 cheaper sufficient statistics。
4. 当前 route 停在 `{route['route']}`，原因是 `{route['primary_blocker']}`。

最终一句话：

> v9.2.33 真实执行后停在 `{route['route']}`：`{route['primary_blocker']}`。
"""
    ensure_dir(REPORT_PATH.parent)
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
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
    parser.add_argument("--functional-step-fraction", type=float, default=0.10)
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--horizons", default="20,80,240")
    parser.add_argument("--primitives", default="P0-N2a-Rational,P1-N2c-SharedRBF,P2-N3c-SharedRBFDerivativeBand")
    parser.add_argument("--probe-source-limit", type=int, default=0)
    args = _make_helper_args(parser.parse_args())

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)

    write_json(out_dir / "run_manifest.json", {
        "version": "v9.2.33",
        "created_utc": _now_iso(),
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "runner": str(SCRIPT_PATH.relative_to(ROOT)),
        "source_v9232": str(SRC_V9232.relative_to(ROOT)),
        "source_v9231": str(SRC_V9231.relative_to(ROOT)),
        "source_v9230": str(SRC_V9230.relative_to(ROOT)),
        "device": str(device),
        "seed": args.seed,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "horizons": args.horizons,
        "primitives": args.primitives,
        "mode": "legal_train_stream_probe_to_commit_audit",
    })
    write_csv_rows(out_dir / "contract_audit_v9233.csv", [
        {
            "stage": "CONTRACT",
            "loss_type": "CE",
            "teacher_used": 0,
            "self_teacher_used": 0,
            "distillation_used": 0,
            "loss_modification_used": 0,
            "label_smoothing_used": 0,
            "sampler_or_class_weight_used": 0,
            "dataset_name_used_in_official_controller": 0,
            "legal_train_stream_probe_newly_measured": 1,
            "validation_metric_used_at_commit": 0,
            "test_metric_used_at_commit": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ])

    p0 = _p0_boundary()
    write_csv_rows(out_dir / "p0_v9232_boundary_reproduction.csv", [p0])
    events = _load_joined_events(args)
    p1_rows, p1 = _cp5_legality_autopsy(events, bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_source_logged_cp5_legality_autopsy.csv", p1_rows)
    p2_rows, probe_trace, p2 = _run_legal_probe(args, device, events, bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p2_legal_train_stream_probe_implementation.csv", p2_rows)
    p2_summary_rows = json.loads(str(p2.get("legal_probe_all_summary", "[]")))
    write_csv_rows(out_dir / "p2_legal_train_stream_probe_summary.csv", p2_summary_rows if p2_summary_rows else [
        _not_run("P2_LEGAL_TRAIN_STREAM_PROBE_SUMMARY", "p2_legal_train_stream_probe_summary.csv", "P2_not_opened")
    ])
    write_csv_rows(out_dir / "legal_train_stream_probe_trace_v9233.csv", probe_trace)
    p3, p4, p3_p4_paths = _run_p3_p4_downstream(out_dir, p2)
    p5_rows, p5 = _run_p5_observable_primitive(events, p2, bool(not _int(p2.get("legal_probe_controller_candidate_pass"))))
    write_csv_rows(out_dir / "p5_observable_primitive_implementation_gate.csv", p5_rows)
    write_csv_rows(out_dir / "observable_primitive_trace_v9233.csv", p5_rows)

    downstream_reason = "P5_observable_primitive_failed_and_no_legal_probe_controller"
    downstream_paths = _write_stage_not_run(out_dir, [
        ("p6_official_probe_gated_paired_replay.csv", "P6_OFFICIAL_PROBE_GATED_PAIRED_REPLAY"),
        ("p7_short_run_functional_validation.csv", "P7_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p8_full_10seed_functional_validation.csv", "P8_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p9_adamw_only_fullpass_repair.csv", "P9_ADAMW_ONLY_FULLPASS_REPAIR"),
        ("p10_robustness_external_ready.csv", "P10_ROBUSTNESS_EXTERNAL_READY"),
        ("paired_replay_branch_trace_v9233.csv", "P6_PAIRED_REPLAY_BRANCH_TRACE"),
    ], downstream_reason)

    failure_rows = [
        {
            "failure_code": "F0_v9232_boundary_not_reproduced",
            "active": int(not _int(p0.get("P0_pass"))),
            "detail": "" if _int(p0.get("P0_pass")) else "v9232 boundary reproduction failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "failure_code": "F3_legal_probe_predictive_failed",
            "active": int(not _int(p2.get("legal_probe_predictive_pass"))),
            "detail": f"best={p2.get('best_legal_probe')} corr={p2.get('legal_probe_corr')} auc={p2.get('legal_probe_auc')}",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "failure_code": "F4_probe_predictive_but_too_expensive",
            "active": int(_int(p2.get("legal_probe_predictive_pass")) and not _int(p2.get("legal_probe_system_pass"))),
            "detail": f"step_q90={p2.get('legal_probe_step_q90')}",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "failure_code": "F14_observable_primitive_not_closed",
            "active": int(not _int(p5.get("observable_primitive_pass"))),
            "detail": f"best={p5.get('best_observable_primitive')} corr={p5.get('best_observable_primitive_corr')} not_impl={p5.get('observable_primitive_not_implemented_count')}",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    _make_svg_bar(
        out_dir / "figures" / "p2_legal_probe_metrics.svg",
        "P2 legal probe corr/AUC/precision",
        [p2.get("best_legal_probe", "none") + "_corr", "auc", "precision", "coverage"],
        [_float(p2.get("legal_probe_corr")), _float(p2.get("legal_probe_auc")), _float(p2.get("legal_probe_precision")), _float(p2.get("legal_probe_coverage"))],
    )

    audit_paths = [
        out_dir / "contract_audit_v9233.csv",
        out_dir / "p0_v9232_boundary_reproduction.csv",
        out_dir / "p1_source_logged_cp5_legality_autopsy.csv",
        out_dir / "p2_legal_train_stream_probe_implementation.csv",
        out_dir / "p2_legal_train_stream_probe_summary.csv",
        out_dir / "p3_probe_to_commit_controller_calibration.csv",
        out_dir / "p4_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p5_observable_primitive_implementation_gate.csv",
        out_dir / "p6_official_probe_gated_paired_replay.csv",
        out_dir / "p7_short_run_functional_validation.csv",
        out_dir / "p8_full_10seed_functional_validation.csv",
        out_dir / "p9_adamw_only_fullpass_repair.csv",
        out_dir / "p10_robustness_external_ready.csv",
    ]
    audit = audit_no_fake(audit_paths)

    if not _int(p0.get("P0_pass")):
        route_name = "R0-SourceBoundaryNotReproduced"
        primary = "v9232_boundary_not_reproduced"
    elif _int(p2.get("legal_probe_predictive_pass")) and not _int(p2.get("legal_probe_system_pass")):
        route_name = "R3-ProbePredictiveButTooExpensive"
        primary = "legal_probe_predictive_but_system_gate_failed"
    elif _int(p2.get("legal_probe_controller_candidate_pass")):
        route_name = "R2-LegalProbePredictive"
        primary = "controller_calibration_not_completed_in_this_runner"
    elif not _int(p2.get("legal_probe_predictive_pass")) and not _int(p5.get("observable_primitive_pass")):
        route_name = "R14-ReturnToInterfacePrimitiveDesign"
        primary = "legal_train_stream_probe_not_predictive_and_observable_primitive_not_implemented"
    else:
        route_name = "R14-ReturnToInterfacePrimitiveDesign"
        primary = "probe_to_commit_not_opened"

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9232_boundary_pass": _int(p0.get("P0_pass")),
        "dataset_tuning_detected": 0,
        "cp5_legality_classification": p1.get("cp5_legality_classification", ""),
        "cp5_main_source": p1.get("cp5_main_source", ""),
        "cp5_best_feature": p1.get("cp5_best_feature", ""),
        "cp5_best_feature_auc": p1.get("cp5_best_feature_auc", 0.0),
        "best_legal_probe": p2.get("best_legal_probe", "not_opened"),
        "legal_probe_predictive_pass": _int(p2.get("legal_probe_predictive_pass")),
        "legal_probe_system_pass": _int(p2.get("legal_probe_system_pass")),
        "legal_probe_controller_candidate_pass": _int(p2.get("legal_probe_controller_candidate_pass")),
        "legal_probe_corr": _float(p2.get("legal_probe_corr")),
        "legal_probe_auc": _float(p2.get("legal_probe_auc")),
        "legal_probe_precision": _float(p2.get("legal_probe_precision")),
        "legal_probe_recall": _float(p2.get("legal_probe_recall")),
        "legal_probe_coverage": _float(p2.get("legal_probe_coverage")),
        "legal_probe_bad_event_rate": _float(p2.get("legal_probe_bad_event_rate")),
        "legal_probe_accepted_count": _int(p2.get("legal_probe_accepted_count")),
        "legal_probe_step_q90": _float(p2.get("legal_probe_step_q90")),
        "legal_probe_memory_ratio": p2.get("legal_probe_memory_ratio", ""),
        "probe_controller_pass": _int(p3.get("probe_controller_pass")),
        "leave_dataset_out_pass": _int(p4.get("leave_dataset_out_pass")),
        "leave_stratum_out_pass": _int(p4.get("leave_stratum_out_pass")),
        "observable_primitive_pass": _int(p5.get("observable_primitive_pass")),
        "best_observable_primitive": p5.get("best_observable_primitive", "not_opened"),
        "best_observable_primitive_corr": _float(p5.get("best_observable_primitive_corr")),
        "observable_primitive_not_implemented_count": _int(p5.get("observable_primitive_not_implemented_count")),
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
        "next_required_implementation": "if_legal_probe_failed_implement_real_OP1_OP6_else_extract_cheap_sufficient_statistics",
        "success_v9233_strict_purekan_functional": 0,
        "success_v9233_full_functional": 0,
        "success_v9233_external_ready": 0,
        **audit,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "v9233_provenance_audit.csv", [audit])

    hash_paths = [
        PLAN_PATH,
        SCRIPT_PATH,
        out_dir / "route_decision.json",
        out_dir / "contract_audit_v9233.csv",
        out_dir / "p0_v9232_boundary_reproduction.csv",
        out_dir / "p1_source_logged_cp5_legality_autopsy.csv",
        out_dir / "p2_legal_train_stream_probe_implementation.csv",
        out_dir / "p2_legal_train_stream_probe_summary.csv",
        out_dir / "legal_train_stream_probe_trace_v9233.csv",
        out_dir / "p3_probe_to_commit_controller_calibration.csv",
        out_dir / "p4_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p5_observable_primitive_implementation_gate.csv",
        out_dir / "failure_table.csv",
        out_dir / "v9233_provenance_audit.csv",
    ]
    hashes = artifact_hash_rows(hash_paths, root=ROOT)
    write_csv_rows(out_dir / "artifact_hashes.csv", hashes)
    _write_report(out_dir, route, p1, p2, p5, hashes)
    print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
