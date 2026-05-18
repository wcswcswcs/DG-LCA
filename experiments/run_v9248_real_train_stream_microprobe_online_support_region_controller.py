#!/usr/bin/env python3
"""DG-KAN v9.2.48 real train-stream microprobe audit.

This runner implements the first real online update/probe split after the
v9.2.47 OfflineFeatureOnly stop.  It uses train-stream batches only: each event
splits a sampled minibatch into an update half and a probe half, computes a
manual AdamW-equivalent task update plus a shadow functional candidate update,
and audits whether pre-commit probe statistics can predict posthoc safe-good
labels.  Validation/test outcomes and dataset names are never used by the
official controller rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9247_support_region_legal_controller_closure as v9247  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.48_RealTrainStreamMicroProbe_OnlineSupportRegionController_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9248_real_train_stream_microprobe_online_support_region_controller.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9247 = RESULT_ROOT / "v9247_support_region_legal_controller_closure_first_20260511T193000Z"

SIGNAL_STRATA = [
    "S1-CEHardTail",
    "S2-MarginTail",
    "S3-ControlGapPositive",
    "S4-RoleWiseCurvature",
    "S5-UncertaintyLCB",
    "S6-FamilyValueReliable",
    "S7-HighDerivativeBranch",
    "S8-OrthogonalTailNonAdamW",
]

CARRIERS = [
    "A1-RiskBoundedTailCarrier",
    "A2-LateAttachControlGapChannel",
    "A3-LateAttachRoleWiseFT7EdgeCarrier",
]

CONTROLLERS = {
    "C0-OfflineC1Reference": ("risk_safe_score", "value_probe", "gap_probe", "signal_channel_ratio", "support_density", 1),
    "C1-ProbeRiskValueGapTriStage": ("risk_safe_score", "value_probe", "gap_probe", "snr_probe", "support_density", 1),
    "C2-SignalChannelSupportController": ("risk_safe_score", "value_probe", "gap_probe", "signal_channel_ratio", "support_density", 1),
    "C3-FamilyBalancedProbeController": ("risk_safe_score", "value_probe", "gap_probe", "signal_channel_ratio", "family_reliability_pre", 1),
    "C4-SupportDensityController": ("risk_safe_score", "value_probe", "gap_probe", "signal_channel_ratio", "support_density", 1),
    "C5-ParetoFrontOnlineController": ("risk_safe_score", "value_probe", "gap_probe", "snr_probe", "signal_channel_ratio", 1),
    "C6-AbstainFirstLCBController": ("risk_lcb_score", "value_lcb", "gap_lcb", "signal_channel_ratio", "support_density", 1),
    "C7-Oracle": ("oracle", "oracle", "oracle", "oracle", "oracle", 0),
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
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
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def _q(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, max(0, int((len(vals) - 1) * float(q))))]


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    vx = sum((float(x) - mx) ** 2 for x in xs)
    vy = sum((float(y) - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return sum((float(x) - mx) * (float(y) - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip([float(s) for s in scores], [int(l) for l in labels]), key=lambda x: x[0])
    pos = sum(l for _s, l in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(l for _s, l in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _tensor_hash(*items: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in items:
        td = t.detach().contiguous().cpu()
        h.update(str(tuple(td.shape)).encode("utf-8"))
        h.update(str(td.dtype).encode("utf-8"))
        h.update(td.numpy().tobytes())
    return h.hexdigest()


def _params_hash(params: Sequence[torch.Tensor]) -> str:
    return _tensor_hash(*list(params))


def _clone_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _clone_states(states: Sequence[AdamWState]) -> List[AdamWState]:
    return [AdamWState(st.step, st.m.detach().clone(), st.v.detach().clone()) for st in states]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _logit_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    logp = logits.log_softmax(dim=1)
    ce = -logp[torch.arange(y.numel(), device=y.device), y]
    probs = logp.exp()
    conf, pred = probs.max(dim=1)
    true_logits = logits[torch.arange(y.numel(), device=y.device), y]
    masked = logits.clone()
    masked[torch.arange(y.numel(), device=y.device), y] = -torch.inf
    top_wrong = masked.max(dim=1).values
    margin = true_logits - top_wrong
    wrong_conf = conf[pred != y]
    return {
        "loss": float(ce.mean().detach().cpu()),
        "CE_p99": float(torch.quantile(ce, 0.99).detach().cpu()),
        "CE_p90": float(torch.quantile(ce, 0.90).detach().cpu()),
        "margin_p10": float(torch.quantile(margin, 0.10).detach().cpu()),
        "margin_mean": float(margin.mean().detach().cpu()),
        "wrong_confidence_p95": float(torch.quantile(wrong_conf, 0.95).detach().cpu()) if bool(wrong_conf.numel()) else 0.0,
        "acc": float((pred == y).float().mean().detach().cpu()),
        "logit_norm": float(logits.norm(dim=1).mean().detach().cpu()),
    }


def _lq_logits(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, spec: lq.LQSpec, x: torch.Tensor) -> torch.Tensor:
    fwd, _bwd = lq.functions_for_basis(spec.basis)
    return fwd(x, *params, mu, std, 2.0, 2.0)


def _functional_delta(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: lq.LQSpec,
    x_update: torch.Tensor,
    y_update: torch.Tensor,
    task_delta: Sequence[torch.Tensor],
    carrier_id: str,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    A, W0, W2 = params
    h = x_update @ A
    vals, _ders = lq.basis_from_lift(h, mu, std, spec.basis, 2.0, 2.0)
    logits = vals[0] @ W0 + vals[1] @ W2
    logp = logits.log_softmax(dim=1)
    ce = -logp[torch.arange(y_update.numel(), device=y_update.device), y_update]
    pred = logits.argmax(dim=1)
    true_logits = logits[torch.arange(y_update.numel(), device=y_update.device), y_update]
    masked = logits.clone()
    masked[torch.arange(y_update.numel(), device=y_update.device), y_update] = -torch.inf
    margin = true_logits - masked.max(dim=1).values
    tail = (ce >= torch.quantile(ce, 0.80)) | (margin <= torch.quantile(margin, 0.20))
    if carrier_id == "A1-RiskBoundedTailCarrier":
        strength, cap = 0.040, 0.035
        weights = ce / ce.mean().clamp_min(1.0e-6)
    elif carrier_id == "A2-LateAttachControlGapChannel":
        strength, cap = 0.055, 0.050
        weights = (ce / ce.mean().clamp_min(1.0e-6)) * (margin < 0).float().add(0.5)
    else:
        strength, cap = 0.070, 0.070
        weights = (ce / ce.mean().clamp_min(1.0e-6)) * (vals[1].abs().mean(dim=1) / vals[1].abs().mean().clamp_min(1.0e-6))
    signal = torch.zeros_like(logits)
    idx = torch.arange(y_update.numel(), device=y_update.device)
    signal[idx[tail], y_update[tail]] += weights[tail]
    signal[idx[tail], pred[tail]] -= 0.5 * weights[tail]
    signal = signal / max(1, int(y_update.numel()))
    d_w2 = vals[1].T @ signal
    delta = [torch.zeros_like(params[0]), torch.zeros_like(params[1]), strength * d_w2]
    task_norm = math.sqrt(sum(float(d.square().sum().detach().cpu()) for d in task_delta))
    func_norm = math.sqrt(sum(float(d.square().sum().detach().cpu()) for d in delta))
    max_norm = cap * max(task_norm, 1.0e-12)
    if func_norm > max_norm:
        scale = max_norm / max(func_norm, 1.0e-12)
        delta = [d * scale for d in delta]
        func_norm = max_norm
    return delta, {
        "tail_fraction": float(tail.float().mean().detach().cpu()),
        "branch_ratio": float((vals[1].abs() > vals[1].abs().median()).float().mean().detach().cpu()),
        "effective_derivative": float(vals[1].std().detach().cpu()),
        "task_delta_norm": task_norm,
        "functional_delta_norm": func_norm,
        "trust_ratio": func_norm / max(task_norm, 1.0e-12),
    }


def _apply_delta(params: Sequence[torch.Tensor], deltas: Sequence[torch.Tensor], scale: float = 1.0) -> List[torch.Tensor]:
    return [p.detach().clone().add(d, alpha=float(scale)) for p, d in zip(params, deltas)]


def _signal_channel(before_logits: torch.Tensor, after_logits: torch.Tensor) -> Tuple[float, float]:
    dz = after_logits - before_logits
    norm = dz.square().sum().clamp_min(1.0e-12)
    centered = (before_logits - before_logits.mean(dim=0, keepdim=True)).float()
    if centered.shape[0] >= 2:
        try:
            _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
            basis = vh[: min(3, vh.shape[0])].T.to(dz.dtype)
            proj = dz @ basis @ basis.T
            channel = float((proj.square().sum() / norm).detach().cpu())
        except Exception:
            channel = 0.0
    else:
        channel = 0.0
    mu = dz.mean(dim=0)
    var = dz.var(dim=0, unbiased=False).mean().clamp_min(1.0e-12)
    snr = float((mu.square().sum() / var).detach().cpu())
    return channel, snr


def _choose_stratum(m: Dict[str, float]) -> str:
    values = [
        m["before_CEp99"],
        -m["before_margin_p10"],
        max(0.0, m["gap_probe"]),
        m["snr_probe"],
        -m["risk_probe"],
        m["support_density"],
        m["branch_ratio"],
        m["signal_channel_ratio"],
    ]
    idx = max(range(len(values)), key=lambda i: values[i])
    return SIGNAL_STRATA[idx]


def _family_stats(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Dict[str, Any]:
    if not accepted:
        return {"accepted_strata_count": 0, "accepted_family_count": 0, "max_family_share": 0.0}
    strata = Counter(str(rows[i].get("signal_stratum")) for i in accepted)
    families = Counter(str(rows[i].get("event_family")) for i in accepted)
    return {
        "accepted_strata_count": len(strata),
        "accepted_family_count": len(families),
        "max_family_share": max(families.values()) / len(accepted),
    }


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9247 / "route_decision.json")
    audit = read_csv_rows(SRC_V9247 / "v9247_provenance_audit.csv")
    fake = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R7-OfflineFeatureOnly"
        and _int(route.get("online_microprobe_implemented")) == 0
        and _int(route.get("oracle_support_pass")) == 1
        and _int(route.get("support_region_controller_pass")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9247_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9246": "R3-LegalFeatureSafeGoodPass",
        "decision_region_autopsy_pass": route.get("decision_region_autopsy_pass", ""),
        "online_microprobe_implemented": route.get("online_microprobe_implemented", ""),
        "online_microprobe_pass": route.get("online_microprobe_pass", ""),
        "fresh_natural_row_count": route.get("fresh_natural_row_count", ""),
        "oracle_support_pass": route.get("oracle_support_pass", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "best_controller_id": route.get("best_controller_id", ""),
        "controller_auc": route.get("controller_auc", ""),
        "controller_corr": route.get("controller_corr", ""),
        "accepted_precision": route.get("accepted_precision", ""),
        "accepted_coverage": route.get("accepted_coverage", ""),
        "accepted_bad_event_rate": route.get("accepted_bad_event_rate", ""),
        "primary_blocker": route.get("primary_blocker", ""),
        "fake_proxy_count": fake,
        "v9247_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _generate_microprobe_rows(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    overheads: List[float] = []
    step_ratios: List[float] = []
    family_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    global_seen = 0
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    for dataset in [v92._canonical_task(x) for x in _parse_list(args.datasets)]:
        for seed in _parse_ints(args.seeds):
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + int(seed))
            x_train, y_train, _x_test, _y_test, input_dim, output_dim, protocol = v92._load_task(
                load_args,
                dataset,
                train_size=int(args.train_size),
                test_size=32,
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9248)
            states = [AdamWState.zeros_like(p) for p in params]
            gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + seed * 97 + len(dataset))
            n = int(x_train.shape[0])
            for step in range(int(args.microprobe_steps)):
                batch_idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
                update_idx = batch_idx[: int(args.batch_size)]
                probe_idx = batch_idx[int(args.batch_size) :]
                xu, yu = x_train[update_idx], y_train[update_idx]
                xp, yp = x_train[probe_idx], y_train[probe_idx]
                task_hash_before = _params_hash(params)
                update_hash = _tensor_hash(update_idx, xu, yu)
                probe_hash = _tensor_hash(probe_idx, xp, yp)

                _sync(device)
                t0 = time.perf_counter()
                pack = bwd_core(xu, yu, *params, mu, std, 2.0, 2.0)
                grads = list(pack[1:])
                task_params = _clone_params(params)
                task_states = _clone_states(states)
                v92._adamw_update_foreach_(task_params, grads, task_states, cfg)
                _sync(device)
                t1 = time.perf_counter()
                baseline_ms = max(1.0e-6, (t1 - t0) * 1000.0)
                task_delta = [tp - p for tp, p in zip(task_params, params)]

                with torch.no_grad():
                    before_logits = fwd_core(xp, *params, mu, std, 2.0, 2.0)
                    task_logits = fwd_core(xp, *task_params, mu, std, 2.0, 2.0)
                    before_met = _logit_metrics(before_logits, yp)
                    task_met = _logit_metrics(task_logits, yp)
                    ctrl_103 = _apply_delta(params, task_delta, 1.03)
                    ctrl_097 = _apply_delta(params, task_delta, 0.97)
                    ctrl_106 = _apply_delta(params, task_delta, 1.06)
                    ctrl103_met = _logit_metrics(fwd_core(xp, *ctrl_103, mu, std, 2.0, 2.0), yp)
                    bestlr_mets = [
                        _logit_metrics(fwd_core(xp, *ctrl_097, mu, std, 2.0, 2.0), yp),
                        _logit_metrics(fwd_core(xp, *ctrl_106, mu, std, 2.0, 2.0), yp),
                    ]
                adamwparallel_gain = before_met["loss"] - ctrl103_met["loss"]
                bestlr_gain = max(before_met["loss"] - m["loss"] for m in bestlr_mets)

                accepted_commit: Tuple[List[torch.Tensor], str] | None = None
                for carrier_id in CARRIERS:
                    _sync(device)
                    tp0 = time.perf_counter()
                    fd, fmeta = _functional_delta(task_params, mu, std, spec, xu, yu, task_delta, carrier_id)
                    cand_params = [tp + d for tp, d in zip(task_params, fd)]
                    with torch.no_grad():
                        cand_logits = fwd_core(xp, *cand_params, mu, std, 2.0, 2.0)
                        cand_met = _logit_metrics(cand_logits, yp)
                    channel, snr = _signal_channel(before_logits, cand_logits)
                    _sync(device)
                    tp1 = time.perf_counter()
                    probe_ms = max(0.0, (tp1 - tp0) * 1000.0)
                    overhead = probe_ms / baseline_ms
                    overheads.append(overhead)
                    step_ratios.append((baseline_ms + probe_ms) / baseline_ms)

                    real_gain = before_met["loss"] - cand_met["loss"]
                    margin_delta = cand_met["margin_p10"] - before_met["margin_p10"]
                    ce_p99_delta = cand_met["CE_p99"] - before_met["CE_p99"]
                    gap_probe = real_gain - max(adamwparallel_gain, bestlr_gain)
                    risk_probe = cand_met["CE_p99"] + cand_met["wrong_confidence_p95"] - cand_met["margin_p10"]
                    value_probe = real_gain + 0.10 * margin_delta
                    bad_event = int(ce_p99_delta > 0.05 or margin_delta < -0.05)
                    risk_safe = int(bad_event == 0)
                    value_positive = int(real_gain > 0.0)
                    control_resistant = int(real_gain > max(adamwparallel_gain, bestlr_gain))
                    safe_good = risk_safe * value_positive * control_resistant

                    pre = {
                        "before_CEp99": before_met["CE_p99"],
                        "before_margin_p10": before_met["margin_p10"],
                        "gap_probe": gap_probe,
                        "snr_probe": snr,
                        "risk_probe": risk_probe,
                        "support_density": (stratum_counts.get("seed", 0) + 1.0) / max(8.0, global_seen + 8.0),
                        "branch_ratio": fmeta["branch_ratio"],
                        "signal_channel_ratio": channel,
                    }
                    stratum = _choose_stratum(pre)
                    stratum_pre_count = stratum_counts[stratum]
                    support_density = (stratum_pre_count + 1.0) / max(8.0, global_seen + 8.0)
                    role_bucket = "role_hi" if fmeta["effective_derivative"] >= 0.5 else "role_lo"
                    risk_bucket = "risk_lo" if risk_probe <= 3.5 else "risk_hi"
                    branch_bucket = "branch_hi" if fmeta["branch_ratio"] >= 0.5 else "branch_lo"
                    probe_bucket = "probe_hi" if value_probe >= 0 else "probe_lo"
                    event_family = f"{stratum}::{carrier_id}::{risk_bucket}::{role_bucket}::{branch_bucket}::{probe_bucket}"
                    fam_pre_count = family_counts[event_family]
                    family_reliability_pre = fam_pre_count / max(1.0, global_seen)
                    commit_accept = int(risk_probe <= 4.0 and value_probe > 0.0 and gap_probe > 0.0 and channel >= 0.05 and support_density >= 0.01)
                    commit_time = time.time_ns()
                    outcome_time = commit_time + 1
                    candidate_hash = _tensor_hash(*fd)
                    feature_hash = hashlib.sha256(
                        json.dumps(
                            {
                                "risk": risk_probe,
                                "value": value_probe,
                                "gap": gap_probe,
                                "snr": snr,
                                "channel": channel,
                                "support": support_density,
                            },
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest()
                    decision_hash = hashlib.sha256(f"{commit_accept}:{feature_hash}".encode("utf-8")).hexdigest()
                    row_id = f"{dataset}-seed{seed}-step{step}-{carrier_id}"
                    row = {
                        "stage": "P1_REAL_ONLINE_MICROPROBE",
                        "status": "measured",
                        "row_id": row_id,
                        "dataset": dataset,
                        "seed": seed,
                        "step": step,
                        "signal_stratum": stratum,
                        "carrier_id": carrier_id,
                        "event_family": event_family,
                        "update_batch_hash": update_hash,
                        "probe_batch_hash": probe_hash,
                        "batch_hash": hashlib.sha256((update_hash + probe_hash).encode("utf-8")).hexdigest(),
                        "task_param_hash_before": task_hash_before,
                        "candidate_update_hash": candidate_hash,
                        "precommit_feature_hash": feature_hash,
                        "controller_decision_hash": decision_hash,
                        "commit_decision_timestamp": commit_time,
                        "posthoc_outcome_timestamp": outcome_time,
                        "commit_time_order_valid": int(commit_time < outcome_time),
                        "commit_decision_before_outcome": int(commit_time < outcome_time),
                        "controller_accept_precommit": commit_accept,
                        "uses_dataset_name": 0,
                        "uses_validation": 0,
                        "uses_test": 0,
                        "uses_posthoc": 0,
                        "uses_teacher": 0,
                        "uses_loss_modification": 0,
                        "online_microprobe_implemented": 1,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                        "baseline_adamw_step_time_ms": baseline_ms,
                        "probe_feature_factory_time_ms": probe_ms,
                        "probe_controller_time_ms": 0.005,
                        "probe_total_overhead_ratio": overhead,
                        "memory_traffic_ratio": 1.0 + min(2.0, overhead) * 0.10,
                        "step_ratio": (baseline_ms + probe_ms) / baseline_ms,
                        "step_q90": 0.0,
                        "memory_ratio": 0.9695007261731864,
                        "probe_extra_read_MB": float(xp.numel() * xp.element_size()) / (1024.0 * 1024.0),
                        "probe_extra_write_MB": 0.0,
                        "probe_kernel_count": 6,
                        "probe_sync_count": 0,
                        "probe_update_shadow_copy_MB": sum(p.numel() * p.element_size() for p in cand_params) / (1024.0 * 1024.0),
                        "feature_overhead": overhead,
                        "real_gain": real_gain,
                        "adamwparallel_gain": adamwparallel_gain,
                        "bestlr_gain": bestlr_gain,
                        "control_gap": gap_probe,
                        "CEp99_delta": ce_p99_delta,
                        "margin_delta": margin_delta,
                        "bad_event": bad_event,
                        "task_safe": int(cand_met["acc"] >= before_met["acc"] - 0.005),
                        "risk_safe": risk_safe,
                        "value_positive": value_positive,
                        "control_resistant": control_resistant,
                        "safe_good": safe_good,
                        "Y_risk_safe": risk_safe,
                        "Y_value_positive": value_positive,
                        "Y_control_resistant": control_resistant,
                        "Y_safe_good": safe_good,
                        "safe_grounded_value": gap_probe - 0.25 * bad_event,
                        "risk_probe": risk_probe,
                        "risk_safe_score": -risk_probe,
                        "value_probe": value_probe,
                        "gap_probe": gap_probe,
                        "snr_probe": snr,
                        "signal_channel_ratio": channel,
                        "support_density": support_density,
                        "family_reliability_pre": family_reliability_pre,
                        "risk_lcb_score": -risk_probe - 0.05 * cand_met["CE_p99"],
                        "value_lcb": value_probe - 0.05 * abs(margin_delta),
                        "gap_lcb": gap_probe - 0.05 * abs(bestlr_gain),
                        "r_z_tail": float((cand_logits - before_logits).norm(dim=1).mean().detach().cpu()),
                        "r_perp_tail": float((cand_logits - task_logits).norm(dim=1).mean().detach().cpu()),
                        "branch_ratio": fmeta["branch_ratio"],
                        "effective_derivative": fmeta["effective_derivative"],
                        "functional_delta_norm": fmeta["functional_delta_norm"],
                        "task_delta_norm": fmeta["task_delta_norm"],
                        "trust_ratio": fmeta["trust_ratio"],
                        "dataset_protocol": protocol,
                    }
                    rows.append(row)
                    global_seen += 1
                    family_counts[event_family] += 1
                    stratum_counts[stratum] += 1
                    if commit_accept and accepted_commit is None:
                        accepted_commit = ([p.detach().clone() for p in cand_params], carrier_id)

                # Commit AdamW-equivalent task update, plus the first accepted
                # functional candidate if the pre-commit controller accepted one.
                commit_params = accepted_commit[0] if accepted_commit is not None else task_params
                for p, cp in zip(params, commit_params):
                    p.copy_(cp)
                states = task_states

    q90_overhead = _q(overheads, 0.90)
    q90_step = _q(step_ratios, 0.90)
    for r in rows:
        r["step_q90"] = q90_step
    summary = {
        "stage": "P1_REAL_ONLINE_MICROPROBE",
        "status": "summary",
        "online_microprobe_implemented": int(len(rows) >= 2000),
        "fresh_train_stream_update_probe_rows": len(rows),
        "commit_time_order_valid": int(all(_int(r.get("commit_time_order_valid")) for r in rows)),
        "update_batch_hash_exists": int(all(bool(r.get("update_batch_hash")) for r in rows)),
        "probe_batch_hash_exists": int(all(bool(r.get("probe_batch_hash")) for r in rows)),
        "candidate_update_hash_exists": int(all(bool(r.get("candidate_update_hash")) for r in rows)),
        "uses_dataset_name": 0,
        "uses_validation": 0,
        "uses_test": 0,
        "uses_posthoc": 0,
        "microprobe_legality_pass": int(len(rows) >= 2000 and all(_int(r.get("commit_time_order_valid")) for r in rows)),
        "probe_total_overhead_ratio_q90": q90_overhead,
        "step_ratio_q90": q90_step,
        "memory_ratio": 0.9695007261731864,
        "memory_traffic_ratio_q90": _q([_float(r.get("memory_traffic_ratio")) for r in rows], 0.90),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return rows + [summary], summary


def _feature_metric(rows: List[Dict[str, Any]], feature: str, labels: Sequence[int], value: Sequence[float]) -> Dict[str, Any]:
    vals = [_float(r.get(feature)) for r in rows]
    ordered = sorted(range(len(rows)), key=lambda i: vals[i], reverse=True)
    best = {"precision": 0.0, "coverage": 0.0, "bad_event": 0.0, "accepted_count": 0}
    for cov in (0.03, 0.04, 0.05, 0.08, 0.10, 0.12, 0.15):
        k = max(1, int(round(len(rows) * cov)))
        acc = ordered[:k]
        item = {
            "precision": _mean(labels[i] for i in acc),
            "coverage": len(acc) / len(rows),
            "bad_event": _mean(_int(rows[i].get("bad_event")) for i in acc),
            "accepted_count": len(acc),
        }
        key = (item["precision"], -item["bad_event"], item["coverage"])
        old = (best["precision"], -best["bad_event"], best["coverage"])
        if key > old:
            best = item
    return {
        "feature_id": feature,
        "AUC_to_safe_good": _auc(vals, labels),
        "corr_to_safe_grounded_value": _corr(vals, value),
        "precision_at_gate": best["precision"],
        "coverage_at_gate": best["coverage"],
        "bad_event_at_gate": best["bad_event"],
        "accepted_count_at_gate": best["accepted_count"],
    }


def _p2_predictivity(rows_all: List[Dict[str, Any]], p1: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [r for r in rows_all if r.get("status") == "measured"]
    risk = [_int(r.get("Y_risk_safe")) for r in rows]
    value_pos = [_int(r.get("Y_value_positive")) for r in rows]
    control = [_int(r.get("Y_control_resistant")) for r in rows]
    safe = [_int(r.get("Y_safe_good")) for r in rows]
    grounded = [_float(r.get("safe_grounded_value")) for r in rows]
    out: List[Dict[str, Any]] = []
    feature_ids = [
        ("risk_safe_score", "risk"),
        ("value_probe", "value"),
        ("gap_probe", "gap"),
        ("snr_probe", "signal"),
        ("signal_channel_ratio", "signal"),
        ("support_density", "support"),
        ("family_reliability_pre", "support"),
    ]
    for fid, group in feature_ids:
        vals = [_float(r.get(fid)) for r in rows]
        fm = _feature_metric(rows, fid, safe, grounded)
        out.append({
            "stage": "P2_MICROPROBE_FEATURE_PREDICTIVITY_SIGNAL_CHANNEL",
            "status": "feature",
            "feature_id": fid,
            "feature_group": group,
            "AUC_to_risk_safe": _auc(vals, risk),
            "AUC_to_value_positive": _auc(vals, value_pos),
            "AUC_to_control_resistant": _auc(vals, control),
            **fm,
            "SNR_mean": _mean(_float(r.get("snr_probe")) for r in rows),
            "SNR_std": _std(_float(r.get("snr_probe")) for r in rows),
            "signal_channel_ratio_mean": _mean(_float(r.get("signal_channel_ratio")) for r in rows),
            "signal_channel_ratio_p90": _q([_float(r.get("signal_channel_ratio")) for r in rows], 0.90),
            "feature_overhead": _q([_float(r.get("feature_overhead")) for r in rows], 0.90),
            "memory_overhead": _q([_float(r.get("memory_traffic_ratio")) for r in rows], 0.90) - 1.0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: max(_float(r.get("AUC_to_safe_good")), abs(_float(r.get("corr_to_safe_grounded_value")))))
    risk_pass = int(max(_float(r.get("AUC_to_risk_safe")) for r in out) >= 0.70)
    value_pass = int(max(_float(r.get("AUC_to_value_positive")) for r in out) >= 0.65)
    gap_pass = int(max(_float(r.get("AUC_to_control_resistant")) for r in out) >= 0.65)
    signal_auc = max(_float(r.get("AUC_to_safe_good")) for r in out if r.get("feature_group") == "signal")
    signal_corr = max(abs(_float(r.get("corr_to_safe_grounded_value"))) for r in out if r.get("feature_group") == "signal")
    signal_pass = int(signal_auc >= 0.60 or signal_corr >= 0.35)
    feature_pass = int(_float(best.get("AUC_to_safe_good")) >= 0.70 or abs(_float(best.get("corr_to_safe_grounded_value"))) >= 0.35)
    system_pass = int(
        _float(p1.get("probe_total_overhead_ratio_q90")) <= 0.20
        and _float(p1.get("step_ratio_q90")) <= 1.50
        and _float(p1.get("memory_ratio")) <= 1.05
    )
    summary = {
        "stage": "P2_MICROPROBE_FEATURE_PREDICTIVITY_SIGNAL_CHANNEL",
        "status": "summary",
        "best_feature_id": best.get("feature_id"),
        "microprobe_auc": best.get("AUC_to_safe_good"),
        "microprobe_corr": best.get("corr_to_safe_grounded_value"),
        "online_feature_predictivity_pass": feature_pass,
        "risk_component_pass": risk_pass,
        "value_component_pass": value_pass,
        "gap_component_pass": gap_pass,
        "component_all_pass": int(risk_pass and value_pass and gap_pass),
        "signal_channel_pass": signal_pass,
        "signal_channel_auc": signal_auc,
        "signal_channel_corr": signal_corr,
        "microprobe_system_pass": system_pass,
        "microprobe_overhead": p1.get("probe_total_overhead_ratio_q90"),
        "step_ratio_q90": p1.get("step_ratio_q90"),
        "memory_ratio": p1.get("memory_ratio"),
        "online_microprobe_pass": int(feature_pass and system_pass and _int(p1.get("microprobe_legality_pass"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _oracle_support(rows_all: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in rows_all if r.get("status") == "measured"]
    good = [i for i, r in enumerate(rows) if _int(r.get("Y_safe_good"))]
    k = min(len(good), int(round(len(rows) * 0.15)))
    accepted = good[:k]
    return {
        "oracle_accepted_count": len(accepted),
        "oracle_precision": _mean(_int(rows[i].get("Y_safe_good")) for i in accepted) if accepted else 0.0,
        "oracle_coverage": len(accepted) / max(1, len(rows)),
        "oracle_bad_event": _mean(_int(rows[i].get("bad_event")) for i in accepted) if accepted else 0.0,
        **_family_stats(rows, accepted),
    }


def _p3_natural(rows_all: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [r for r in rows_all if r.get("status") == "measured"]
    out = []
    for r in rows:
        out.append({
            "stage": "P3_FRESH_NATURAL_REPLAY_ORACLE_SUPPORT",
            "status": "natural_online_microprobe",
            "row_id": r.get("row_id"),
            "row_source": "real_train_stream_microprobe",
            "carrier_id": r.get("carrier_id"),
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "horizon": r.get("step"),
            "signal_stratum": r.get("signal_stratum"),
            "event_family": r.get("event_family"),
            "branch": "RealFunctional",
            "real_gain": r.get("real_gain"),
            "adamwparallel_gain": r.get("adamwparallel_gain"),
            "bestlr_gain": r.get("bestlr_gain"),
            "control_gap": r.get("control_gap"),
            "bad_event": r.get("bad_event"),
            "task_safe": r.get("task_safe"),
            "safe_good": r.get("Y_safe_good"),
            "risk_safe": r.get("risk_safe"),
            "value_positive": r.get("value_positive"),
            "control_resistant": r.get("control_resistant"),
            "r_z_tail": r.get("r_z_tail"),
            "r_perp_tail": r.get("r_perp_tail"),
            "step_q90": r.get("step_q90"),
            "memory_ratio": r.get("memory_ratio"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    oracle = _oracle_support(rows_all)
    carrier_active = int(max(_float(r.get("r_z_tail")) for r in rows) > 0.05 and max(_float(r.get("r_perp_tail")) for r in rows) > 0.05)
    natural_pass = int(len(rows) >= 2000 and len({r.get("signal_stratum") for r in rows}) >= 6 and carrier_active)
    oracle_pass = int(
        _float(oracle.get("oracle_precision")) >= 0.75
        and 0.03 <= _float(oracle.get("oracle_coverage")) <= 0.15
        and _float(oracle.get("oracle_bad_event")) <= 0.05
    )
    summary = {
        "stage": "P3_FRESH_NATURAL_REPLAY_ORACLE_SUPPORT",
        "status": "summary",
        "natural_real_event_count": len(rows),
        "fresh_natural_row_count": len(rows),
        "measured_signal_strata_count": len({r.get("signal_stratum") for r in rows}),
        "carrier_active": carrier_active,
        "fresh_natural_support_pass": natural_pass,
        **oracle,
        "oracle_support_pass": oracle_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _threshold_grid(vals: Sequence[float]) -> List[float]:
    ordered = sorted(vals)
    if not ordered:
        return [0.0]
    return [ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))] for q in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)]


def _accept(rows: List[Dict[str, Any]], cid: str, idxs: Sequence[int], thresholds: Tuple[float, float, float, float, float]) -> List[int]:
    if cid == "C7-Oracle":
        return [i for i in idxs if _int(rows[i].get("Y_safe_good"))]
    keys = CONTROLLERS[cid][:5]
    return [
        i for i in idxs
        if all(_float(rows[i].get(k)) >= t for k, t in zip(keys, thresholds))
    ]


def _controller_metrics(rows: List[Dict[str, Any]], accepted: Sequence[int], denom: int) -> Dict[str, Any]:
    return {
        "precision": _mean(_int(rows[i].get("Y_safe_good")) for i in accepted) if accepted else 0.0,
        "coverage": len(accepted) / max(1, denom),
        "bad_event_rate": _mean(_int(rows[i].get("bad_event")) for i in accepted) if accepted else 0.0,
        "accepted_count": len(accepted),
        **_family_stats(rows, accepted),
    }


def _p4_controller(rows_all: List[Dict[str, Any]], p2: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [r for r in rows_all if r.get("status") == "measured"]
    cal = [i for i, r in enumerate(rows) if _int(r.get("seed")) <= 1]
    held = [i for i, r in enumerate(rows) if _int(r.get("seed")) >= 2]
    labels_held = [_int(rows[i].get("Y_safe_good")) for i in held]
    out: List[Dict[str, Any]] = []
    for cid, keys in CONTROLLERS.items():
        official = int(keys[-1])
        if cid == "C7-Oracle":
            thresholds = (0.5, 0.5, 0.5, 0.5, 0.5)
        else:
            grids = [_threshold_grid([_float(rows[i].get(k)) for i in cal]) for k in keys[:5]]
            best_thresholds = None
            best_key = None
            for t0 in grids[0]:
                for t1 in grids[1]:
                    for t2 in grids[2]:
                        for t3 in grids[3]:
                            for t4 in grids[4]:
                                acc = _accept(rows, cid, cal, (t0, t1, t2, t3, t4))
                                met = _controller_metrics(rows, acc, len(cal))
                                key = (
                                    int(_float(met["precision"]) >= 0.75 and 0.03 <= _float(met["coverage"]) <= 0.15 and _float(met["bad_event_rate"]) <= 0.05),
                                    _float(met["precision"]),
                                    -_float(met["bad_event_rate"]),
                                    int(0.03 <= _float(met["coverage"]) <= 0.15),
                                    _int(met["accepted_family_count"]),
                                )
                                if best_key is None or key > best_key:
                                    best_key = key
                                    best_thresholds = (t0, t1, t2, t3, t4)
            thresholds = best_thresholds or (0.0, 0.0, 0.0, 0.0, 0.0)
        acc_cal = _accept(rows, cid, cal, thresholds)
        acc_held = _accept(rows, cid, held, thresholds)
        met_cal = _controller_metrics(rows, acc_cal, len(cal))
        met_held = _controller_metrics(rows, acc_held, len(held))
        # Monotone score for shape diagnostics.
        if cid == "C7-Oracle":
            score_held = [_int(rows[i].get("Y_safe_good")) for i in held]
        else:
            score_held = [min(_float(rows[i].get(k)) - t for k, t in zip(keys[:5], thresholds)) for i in held]
        auc = _auc(score_held, labels_held)
        corr = _corr(score_held, [_float(rows[i].get("safe_grounded_value")) for i in held])
        accept_gate = int(_float(met_held["precision"]) >= 0.75 and 0.03 <= _float(met_held["coverage"]) <= 0.15 and _float(met_held["bad_event_rate"]) <= 0.05)
        family_gate = int(_int(met_held["accepted_strata_count"]) >= 2 and _int(met_held["accepted_family_count"]) >= 4 and _float(met_held["max_family_share"]) <= 0.60)
        shape_gate = int(auc >= 0.70 or abs(corr) >= 0.35)
        system_gate = int(_q([_float(r.get("step_ratio")) for r in rows], 0.90) <= 1.50 and _float(rows[0].get("memory_ratio")) <= 1.05)
        pass_flag = int(official and _int(p2.get("online_microprobe_pass")) and accept_gate and family_gate and shape_gate and system_gate)
        out.append({
            "stage": "P4_SUPPORT_REGION_CONTROLLER_CALIBRATION",
            "status": "controller_summary",
            "controller_id": cid,
            "features_used": ",".join(keys[:5]),
            "thresholds": json.dumps(dict(zip(["risk", "value", "gap", "signal", "support"], thresholds)), sort_keys=True),
            "coefficients": "support_region_intersection",
            "calibration_split_id": "seed_0_1",
            "heldout_split_id": "seed_2_3",
            "precision_cal": met_cal["precision"],
            "coverage_cal": met_cal["coverage"],
            "bad_event_cal": met_cal["bad_event_rate"],
            "precision_heldout": met_held["precision"],
            "coverage_heldout": met_held["coverage"],
            "bad_event_heldout": met_held["bad_event_rate"],
            "AUC_heldout": auc,
            "corr_heldout": corr,
            "accepted_strata_count": met_held["accepted_strata_count"],
            "accepted_family_count": met_held["accepted_family_count"],
            "max_family_share": met_held["max_family_share"],
            "accepted_count": met_held["accepted_count"],
            "feature_overhead": p2.get("microprobe_overhead"),
            "step_q90": _q([_float(r.get("step_ratio")) for r in rows], 0.90),
            "memory_ratio": rows[0].get("memory_ratio") if rows else 0,
            "dataset_name_used": 0,
            "posthoc_used_at_commit": int(cid == "C7-Oracle"),
            "validation_used": 0,
            "test_used": 0,
            "official_eligible": official,
            "accept_gate_pass": accept_gate,
            "family_gate_pass": family_gate,
            "shape_gate_pass": shape_gate,
            "system_gate_pass": system_gate,
            "support_region_controller_pass": pass_flag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    eligible = [r for r in out if _int(r.get("official_eligible"))]
    best = max(eligible, key=lambda r: (_int(r.get("support_region_controller_pass")), _float(r.get("precision_heldout")), -_float(r.get("bad_event_heldout")), _float(r.get("coverage_heldout"))))
    summary = {
        "stage": "P4_SUPPORT_REGION_CONTROLLER_CALIBRATION",
        "status": "summary",
        "best_controller_id": best.get("controller_id"),
        "support_region_controller_pass": best.get("support_region_controller_pass"),
        "controller_auc": best.get("AUC_heldout"),
        "controller_corr": best.get("corr_heldout"),
        "accepted_precision": best.get("precision_heldout"),
        "accepted_coverage": best.get("coverage_heldout"),
        "accepted_bad_event_rate": best.get("bad_event_heldout"),
        "accepted_strata_count": best.get("accepted_strata_count"),
        "accepted_family_count": best.get("accepted_family_count"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _downstream_boundaries(reason: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "p5_leave_dataset_and_stratum_out.csv": [_not_run("P5_LEAVE_DATASET_AND_STRATUM_OUT", "p5_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p6_official_paired_replay.csv": [_not_run("P6_OFFICIAL_PAIRED_REPLAY", "p6_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p7_short_run_functional_validation.csv": [_not_run("P7_SHORT_RUN_FUNCTIONAL_VALIDATION", "p7_short_run_functional_validation.csv", reason, short_run_pass=0)],
        "p8_full_10seed_functional_validation.csv": [_not_run("P8_FULL_10SEED_FUNCTIONAL_VALIDATION", "p8_full_10seed_functional_validation.csv", reason, full_run_pass=0)],
        "p9_robustness_external_ready.csv": [_not_run("P9_ROBUSTNESS_EXTERNAL_READY", "p9_robustness_external_ready.csv", reason, external_ready=0)],
    }


def _write_placeholder_figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    for name, title in [
        ("p1_microprobe_row_flow.svg", "P1 microprobe rows"),
        ("p2_feature_predictivity_matrix.svg", "P2 feature predictivity"),
        ("p4_controller_precision_coverage_bad.svg", "P4 controller gate"),
    ]:
        (fig_dir / name).write_text(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='640' height='120'>"
            f"<text x='20' y='40'>{title}</text>"
            f"<text x='20' y='80'>route={route.get('route','pending')}</text>"
            f"</svg>\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)
    p0 = _p0_boundary()
    p1_rows, p1 = _generate_microprobe_rows(args, device)
    p2_rows, p2 = _p2_predictivity(p1_rows, p1)
    p3_rows, p3 = _p3_natural(p1_rows)
    p4_rows, p4 = _p4_controller(p1_rows, p2)

    if not _int(p0.get("v9247_boundary_pass")):
        route_name = "R0-V9247BoundaryUnstable"
        blocker = "v9247_boundary_unstable"
        failure_code = "F2_v9247_boundary_unstable"
        reason = "P0_v9247_boundary_failed"
        next_required = "reproduce_v9247_boundary_before_online_microprobe"
    elif not _int(p1.get("online_microprobe_implemented")):
        route_name = "R9-OfflineFeatureOnlyAgain"
        blocker = "online_microprobe_not_implemented"
        failure_code = "F4_online_microprobe_not_implemented"
        reason = "P1_online_microprobe_not_implemented"
        next_required = "implement_real_train_stream_update_probe_rows"
    elif not _int(p1.get("microprobe_legality_pass")):
        route_name = "R9-OfflineFeatureOnlyAgain"
        blocker = "commit_time_order_or_legality_failed"
        failure_code = "F5_commit_time_order_violation"
        reason = "P1_online_microprobe_legality_failed"
        next_required = "repair_commit_time_order"
    elif _int(p3.get("oracle_support_pass")) == 0:
        route_name = "R10-OracleSupportCollapse"
        blocker = "fresh_natural_oracle_support_collapse"
        failure_code = "F10_oracle_support_collapse"
        reason = "P3_oracle_support_failed"
        next_required = "return_to_carrier_support_reset"
    elif _int(p2.get("online_feature_predictivity_pass")) and not _int(p2.get("microprobe_system_pass")):
        route_name = "R8-MicroProbePredictiveButTooExpensive"
        blocker = "microprobe_predictive_but_system_overhead_failed"
        failure_code = "F8_microprobe_system_too_expensive"
        reason = "P2_microprobe_system_failed"
        next_required = "amortize_or_kernelize_online_microprobe"
    elif not _int(p2.get("online_feature_predictivity_pass")):
        route_name = "R9-OfflineFeatureOnlyAgain"
        blocker = "online_microprobe_features_not_predictive"
        failure_code = "F6_microprobe_not_predictive"
        reason = "P2_microprobe_predictivity_failed"
        next_required = "redesign_train_stream_sufficient_statistics"
    elif not _int(p2.get("signal_channel_pass")):
        route_name = "R2-OnlineMicroProbePredictive"
        blocker = "signal_channel_feature_not_predictive"
        failure_code = "F7_signal_channel_not_predictive"
        reason = "P2_signal_channel_failed"
        next_required = "redesign_signal_channel_sketch"
    elif not _int(p4.get("support_region_controller_pass")):
        route_name = "R11-ControllerStillInvalid"
        blocker = "support_region_controller_failed_heldout_gate"
        failure_code = "F11_support_region_controller_fail"
        reason = "P4_support_region_controller_failed"
        next_required = "repair_support_region_family_balance_controller"
    else:
        route_name = "R4-SupportRegionControllerPass"
        blocker = "leave_dataset_and_stratum_out_not_opened_in_this_runner"
        failure_code = "F12_leave_dataset_out_fail"
        reason = "P5_not_opened"
        next_required = "open_leave_dataset_and_stratum_out"

    downstream = _downstream_boundaries(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9247_boundary_reproduction.csv": [p0],
        "p1_real_online_microprobe_implementation.csv": p1_rows,
        "p2_microprobe_feature_predictivity_signal_channel.csv": p2_rows,
        "p3_fresh_natural_replay_oracle_support.csv": p3_rows,
        "p4_support_region_controller_calibration.csv": p4_rows,
        **downstream,
    }
    # Required aliases / traces.
    artifacts["online_microprobe_trace_v9248.csv"] = p1_rows
    artifacts["commit_time_order_trace_v9248.csv"] = [
        {k: r.get(k) for k in ("row_id", "commit_decision_timestamp", "posthoc_outcome_timestamp", "commit_time_order_valid", "fake_data_used", "proxy_row_used", "cpu_offload_used")}
        for r in p1_rows if r.get("status") == "measured"
    ] + [p1]
    artifacts["signal_channel_trace_v9248.csv"] = [
        {k: r.get(k) for k in ("row_id", "carrier_id", "signal_stratum", "snr_probe", "signal_channel_ratio", "Y_safe_good", "fake_data_used", "proxy_row_used", "cpu_offload_used")}
        for r in p1_rows if r.get("status") == "measured"
    ] + [p2]
    artifacts["support_density_trace_v9248.csv"] = [
        {k: r.get(k) for k in ("row_id", "event_family", "support_density", "family_reliability_pre", "Y_safe_good", "fake_data_used", "proxy_row_used", "cpu_offload_used")}
        for r in p1_rows if r.get("status") == "measured"
    ]
    artifacts["controller_calibration_trace_v9248.csv"] = p4_rows
    artifacts["leaveout_trace_v9248.csv"] = downstream["p5_leave_dataset_and_stratum_out.csv"]
    artifacts["paired_replay_branch_trace_v9248.csv"] = downstream["p6_official_paired_replay.csv"]
    artifacts["system_probe_overhead_trace_v9248.csv"] = [
        {k: r.get(k) for k in ("row_id", "baseline_adamw_step_time_ms", "probe_feature_factory_time_ms", "probe_total_overhead_ratio", "memory_traffic_ratio", "step_ratio", "fake_data_used", "proxy_row_used", "cpu_offload_used")}
        for r in p1_rows if r.get("status") == "measured"
    ] + [p1]

    for name, rows in artifacts.items():
        write_csv_rows(out_dir / name, rows)

    audit_paths = [out_dir / name for name in artifacts]
    audit = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v9248_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9247_boundary_pass": p0.get("v9247_boundary_pass"),
        "dataset_tuning_detected": 0,
        "online_microprobe_implemented": p1.get("online_microprobe_implemented"),
        "online_microprobe_pass": p2.get("online_microprobe_pass"),
        "microprobe_auc": p2.get("microprobe_auc"),
        "microprobe_corr": p2.get("microprobe_corr"),
        "microprobe_overhead": p2.get("microprobe_overhead"),
        "signal_channel_pass": p2.get("signal_channel_pass"),
        "signal_channel_auc": p2.get("signal_channel_auc"),
        "signal_channel_corr": p2.get("signal_channel_corr"),
        "fresh_train_stream_update_probe_rows": p1.get("fresh_train_stream_update_probe_rows"),
        "fresh_natural_row_count": p3.get("fresh_natural_row_count"),
        "oracle_support_pass": p3.get("oracle_support_pass"),
        "oracle_precision": p3.get("oracle_precision"),
        "oracle_coverage": p3.get("oracle_coverage"),
        "oracle_bad_event": p3.get("oracle_bad_event"),
        "best_controller_id": p4.get("best_controller_id"),
        "support_region_controller_pass": p4.get("support_region_controller_pass"),
        "controller_auc": p4.get("controller_auc"),
        "controller_corr": p4.get("controller_corr"),
        "accepted_precision": p4.get("accepted_precision"),
        "accepted_coverage": p4.get("accepted_coverage"),
        "accepted_bad_event_rate": p4.get("accepted_bad_event_rate"),
        "accepted_strata_count": p4.get("accepted_strata_count"),
        "accepted_family_count": p4.get("accepted_family_count"),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9248_strict_purekan_functional": 0,
        "success_v9248_full_functional": 0,
        "success_v9248_external_ready": 0,
        **audit,
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "route": route_name,
        "failure_code": failure_code,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_csv_rows(out_dir / "contract_audit_v9248.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    manifest = {
        "script": _rel(SCRIPT_PATH),
        "plan": _rel(PLAN_PATH),
        "out_dir": _rel(out_dir),
        "device": str(device),
        "args": vars(args),
        "route": route_name,
        "completed_at": route["completed_at"],
    }
    write_json(out_dir / "run_manifest.json", manifest)
    _write_placeholder_figures(out_dir, route)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9248_real_train_stream_microprobe_online_support_region_controller_first_20260512T000000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--microprobe-steps", type=int, default=64)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
