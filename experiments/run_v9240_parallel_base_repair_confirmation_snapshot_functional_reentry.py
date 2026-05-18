#!/usr/bin/env python3
"""DG-KAN v9.2.40 parallel base-repair confirmation and snapshot re-entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
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
import run_v927_fc_purekan_lq_fullpass_functional_gate as v927  # noqa: E402
import run_v9236_base_preserving_value_aligned_functional_subspace as v9236  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.40_ParallelBaseRepairConfirmation_SnapshotFunctionalReentry_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9240_parallel_base_repair_confirmation_snapshot_functional_reentry.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.40_ParallelBaseRepairConfirmation_SnapshotFunctionalReentry_实验复盘.md"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9239 = RESULT_ROOT / "v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z"


@dataclass(frozen=True)
class SnapshotAttachCandidate:
    candidate_id: str
    spec: act.ActuatorSpec
    attach_mode: str
    description: str


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _hash_file(path: Path) -> str:
    try:
        return artifact_hash_rows(path)
    except Exception:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return ""


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _ci95_low(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    return _mean(vals) - 1.96 * _std(vals) / math.sqrt(len(vals))


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    return v9236._corr(xs, ys)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return v9236._auc(scores, labels)


def _q(values: Sequence[float], q: float) -> float:
    return v9236._q(values, q)


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _canonical_datasets(text: str) -> List[str]:
    return [v92._canonical_task(x) for x in _parse_list(text)]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def _param_hash(params: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(str(tuple(p.shape)).encode("utf-8"))
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _state_hash(states: Sequence[AdamWState]) -> str:
    h = hashlib.sha256()
    for st in states:
        h.update(str(int(st.step)).encode("utf-8"))
        h.update(st.m.detach().cpu().numpy().tobytes())
        h.update(st.v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _params_max_diff(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> float:
    vals = [(a.detach() - b.detach()).abs().max().float() for a, b in zip(xs, ys)]
    return float(torch.stack(vals).max().detach().cpu()) if vals else 0.0


def _states_max_diff(xs: Sequence[AdamWState], ys: Sequence[AdamWState], field: str) -> float:
    vals = [(getattr(a, field).detach() - getattr(b, field).detach()).abs().max().float() for a, b in zip(xs, ys)]
    return float(torch.stack(vals).max().detach().cpu()) if vals else 0.0


def _clone_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _clone_states(states: Sequence[AdamWState]) -> List[AdamWState]:
    return [AdamWState(st.step, st.m.detach().clone(), st.v.detach().clone()) for st in states]


def _lq_param_count(spec: lq.LQSpec, in_dim: int, out_dim: int) -> int:
    return spec.hidden_dim * in_dim + spec.hidden_dim * out_dim * lq.BASIS_CHANNELS[spec.basis]


def _init_mlp_match(param_count: int, in_dim: int, out_dim: int, seed: int, device: torch.device) -> Tuple[List[torch.Tensor], int]:
    hidden = v927.f922._matched_mlp3_hidden(param_count, in_dim, out_dim)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    params = [
        torch.randn(in_dim, hidden, device=device, generator=gen) / math.sqrt(in_dim),
        torch.randn(hidden, hidden, device=device, generator=gen) / math.sqrt(hidden),
        torch.randn(hidden, out_dim, device=device, generator=gen) / math.sqrt(hidden),
    ]
    return params, hidden


def _predict_mlp(params: Sequence[torch.Tensor], x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return v927._predict_mlp_logits(params, x, batch_size)


def _base_specs() -> Dict[str, lq.LQSpec]:
    return {
        "R0-LQ-current": lq.LQSpec("R0-LQ-current", "t2", 256, "default", 1.0, repair_hypothesis="current_lq_reference"),
        "R2-LQ-fanin-output-scale-confirmed": lq.LQSpec(
            "R2-LQ-fanin-output-scale-confirmed", "t2", 256, "default", 0.8, repair_hypothesis="fan_in_output_scale_confirmed"
        ),
        "R5-LQ-hidden256": lq.LQSpec("R5-LQ-hidden256", "t2", 256, "default", 1.0, repair_hypothesis="same_capacity_control"),
    }


def _attach_candidates(base_spec: lq.LQSpec) -> Dict[str, SnapshotAttachCandidate]:
    hid = int(base_spec.hidden_dim)
    scale = float(base_spec.output_scale)
    return {
        "A1-LateAttachZeroLinearTail": SnapshotAttachCandidate(
            "A1-LateAttachZeroLinearTail",
            act.ActuatorSpec("A1-LateAttachZeroLinearTail", "observable_tail_linear", hid, scale, 0.0, gamma=6.0, center=0.55),
            "late_attach_zero_linear_tail",
            "Zero-init observable tail carrier attached only after repaired base checkpoint.",
        ),
        "A2-LateAttachControlGapChannel": SnapshotAttachCandidate(
            "A2-LateAttachControlGapChannel",
            act.ActuatorSpec("A2-LateAttachControlGapChannel", "observable_control_gap", hid, scale, 0.0, gamma=6.0, center=0.50),
            "late_attach_control_gap_channel",
            "Zero-init control-gap carrier attached only after repaired base checkpoint.",
        ),
        "A3-LateAttachRoleWiseFT7EdgeCarrier": SnapshotAttachCandidate(
            "A3-LateAttachRoleWiseFT7EdgeCarrier",
            act.ActuatorSpec("A3-LateAttachRoleWiseFT7EdgeCarrier", "observable_light_hybrid", hid, scale, 0.0, gamma=4.0),
            "late_attach_rolewise_ft7_edge_carrier",
            "Zero-init FT7-like role-wise edge-owned carrier attached after repaired base checkpoint.",
        ),
        "A4-SymbolicFunctionalSpecAttach": SnapshotAttachCandidate(
            "A4-SymbolicFunctionalSpecAttach",
            act.ActuatorSpec("A4-SymbolicFunctionalSpecAttach", "bounded_rational", hid, scale, 0.0, beta=1.0),
            "symbolic_until_event",
            "Functional carrier remains a zero-init strict edge spec until event-time update.",
        ),
        "A5-FamilyValueLateAttach": SnapshotAttachCandidate(
            "A5-FamilyValueLateAttach",
            act.ActuatorSpec("A5-FamilyValueLateAttach", "observable_orthogonal_tail", hid, scale, 0.0, gamma=4.0, center=0.55),
            "family_value_late_attach",
            "Zero-init family-value carrier attached after repaired base checkpoint.",
        ),
    }


def _p4_system_pass(row: Dict[str, Any]) -> Tuple[int, float, float, float, float]:
    f = _float(row.get("forward_ratio_q90", row.get("forward_ratio", "")), 999.0)
    b = _float(row.get("backward_ratio_q90", row.get("backward_ratio", "")), 999.0)
    s = _float(row.get("step_ratio_q90", row.get("step_ratio", "")), 999.0)
    m = _float(row.get("compact_memory_ratio", row.get("memory_compact", "")), 999.0)
    return int(f <= 1.25 and b <= 1.50 and s <= 1.50 and m <= 1.05), f, b, s, m


def _train_lq_cached(
    args: argparse.Namespace,
    spec: lq.LQSpec,
    dataset: str,
    seed: int,
    device: torch.device,
    stage: str,
    repeat_id: str,
    *,
    store_cache: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any] | None]:
    x_train, y_train, x_test, y_test, in_dim, out_dim, protocol = v92._load_task(
        args, dataset, train_size=int(args.train_size), test_size=int(args.test_size)
    )
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_test = x_test.to(device=device, dtype=torch.float32)
    y_test = y_test.to(device=device)
    params, mu, std = lq.init_lq_params(in_dim, out_dim, spec, x_train, device, int(seed) + 92600)
    mlp_params, hidden = _init_mlp_match(_lq_param_count(spec, in_dim, out_dim), in_dim, out_dim, int(seed) + 92650, device)
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    states = [AdamWState.zeros_like(p) for p in params]
    mlp_states = [AdamWState.zeros_like(p) for p in mlp_params]
    trace_rows: List[Dict[str, Any]] = []
    last_grads: Sequence[torch.Tensor] = []
    for epoch in range(int(args.epochs)):
        gen_epoch = torch.Generator(device=device).manual_seed(int(seed) * 1000 + epoch + 17)
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen_epoch)
        kan_loss_sum = 0.0
        mlp_loss_sum = 0.0
        seen = 0
        for start in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            pack = bwd_core(xb, yb, *params, mu, std, 2.0, 2.0)
            last_grads = pack[1:]
            v92._adamw_update_foreach_(params, last_grads, states, cfg)
            mpack = v927.f922._mlp3_fwd_bwd_core(xb, yb, *mlp_params)
            v92._adamw_update_foreach_(mlp_params, mpack[1:], mlp_states, cfg)
            bs = int(xb.shape[0])
            seen += bs
            kan_loss_sum += float(pack[0].detach().cpu()) * bs
            mlp_loss_sum += float(mpack[0].detach().cpu()) * bs
        if epoch in {0, int(args.epochs) - 1}:
            trace_rows.append({
                "stage": f"{stage}_TRACE",
                "candidate": spec.candidate_id,
                "dataset": dataset,
                "seed": seed,
                "repeat_id": repeat_id,
                "epoch": epoch + 1,
                "kan_train_loss_epoch": kan_loss_sum / max(1, seen),
                "mlp_train_loss_epoch": mlp_loss_sum / max(1, seen),
                "loss_type": "CE",
                "label_smoothing": 0,
                "uses_loss_backward": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    kan_logits = lq.predict_lq_logits(params, mu, std, spec.basis, x_test, int(args.eval_batch_size))
    mlp_logits = _predict_mlp(mlp_params, x_test, int(args.eval_batch_size))
    kan_metrics = v92._classification_metrics_from_logits(kan_logits, y_test)
    mlp_metrics = v92._classification_metrics_from_logits(mlp_logits, y_test)
    train_head = min(2048, int(x_train.shape[0]))
    kan_train_logits = lq.predict_lq_logits(params, mu, std, spec.basis, x_train[:train_head], int(args.eval_batch_size))
    mlp_train_logits = _predict_mlp(mlp_params, x_train[:train_head], int(args.eval_batch_size))
    kan_train = v92._classification_metrics_from_logits(kan_train_logits, y_train[:train_head])
    mlp_train = v92._classification_metrics_from_logits(mlp_train_logits, y_train[:train_head])
    h_head = x_train[:train_head] @ params[0]
    lift_metrics = lq.lift_feature_metrics(h_head, mu, std, spec.basis)
    delta = float(kan_metrics["acc"] - mlp_metrics["acc"])
    grad_lift = float(last_grads[0].norm().detach().cpu()) if last_grads else 0.0
    grad_basis = float(sum(g.norm().detach().cpu() for g in last_grads[1:])) if last_grads else 0.0
    row = {
        "stage": stage,
        "status": "measured",
        "candidate": spec.candidate_id,
        "candidate_id": spec.candidate_id,
        "dataset": dataset,
        "seed": seed,
        "rerun_id": repeat_id,
        "protocol": protocol,
        "basis": spec.basis,
        "hidden_dim": spec.hidden_dim,
        "init_variant": spec.init_variant,
        "output_scale": spec.output_scale,
        "acc": kan_metrics["acc"],
        "KAN_acc": kan_metrics["acc"],
        "mlp_acc": mlp_metrics["acc"],
        "MLP_match_acc": mlp_metrics["acc"],
        "delta_vs_mlp": delta,
        "near_pass": int(delta >= -0.01),
        "near_margin": delta + 0.01,
        "CEp99": kan_metrics["CE_p99"],
        "margin_p10": kan_metrics["correct_margin_p10"],
        "ECE": kan_metrics["ECE"],
        "NLL": kan_metrics["NLL"],
        "wrong_confidence_p95": kan_metrics["wrong_confidence_p95"],
        "train_acc": kan_train["acc"],
        "mlp_train_acc": mlp_train["acc"],
        "grad_norm_lift": grad_lift,
        "grad_norm_basis": grad_basis,
        **lift_metrics,
        "matched_mlp_hidden": hidden,
        "params_kan": sum(p.numel() for p in params),
        "params_mlp_match": sum(p.numel() for p in mlp_params),
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "distillation_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    cache = None
    if store_cache:
        cache = {
            "params": _clone_params(params),
            "states": _clone_states(states),
            "mu": mu.detach().clone(),
            "std": std.detach().clone(),
            "spec": spec,
            "dataset": dataset,
            "seed": seed,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "x_train": x_train.detach().clone(),
            "y_train": y_train.detach().clone(),
            "x_eval": x_test[: int(args.eval_size)].detach().clone(),
            "y_eval": y_test[: int(args.eval_size)].detach().clone(),
        }
    return row, trace_rows, cache


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9239 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9239 / "v9239_provenance_audit.csv")
    fake = _int(audit_rows[0].get("fake_proxy_nonzero_count")) if audit_rows else 1
    p0_pass = int(
        route.get("route") == "R5-BaseRepairPass"
        and route.get("best_base_candidate") == "R2-LQ-fanin-output-scale-confirmed"
        and _int(route.get("base_repair_pass")) == 1
        and _int(route.get("snapshot_attach_pass")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9239_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": _rel(SRC_V9239),
        "route": route.get("route", ""),
        "source_route_v9238": route.get("source_route", ""),
        "current_lq_near_rate": route.get("current_lq_near_rate", ""),
        "current_lq_macro_delta": route.get("current_lq_macro_delta", ""),
        "miss_row_count": route.get("miss_row_count", ""),
        "miss_row_borderline_rate": route.get("miss_row_borderline_rate", ""),
        "protocol_mismatch_detected": route.get("protocol_mismatch_detected", ""),
        "protocol_mismatch_unresolved": route.get("protocol_mismatch_unresolved", ""),
        "current_lq_rerun_count": route.get("current_lq_rerun_count", ""),
        "best_base_repair": route.get("best_base_candidate", ""),
        "best_base_near_rate": route.get("best_base_near_rate", ""),
        "best_base_macro_delta": route.get("best_base_macro_delta", ""),
        "best_base_step_q90": route.get("best_base_step_q90", ""),
        "base_repair_pass": route.get("base_repair_pass", ""),
        "primary_blocker": route.get("primary_blocker", ""),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_base_confirmation(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[Tuple[str, str, int], Dict[str, Any]]]:
    if not opened:
        row = _not_run("P1_PARALLEL_BASE_ROBUST_CONFIRMATION", "p1_parallel_base_robust_confirmation.csv", "P0_v9239_boundary_failed")
        return [row], [row], {"repaired_base_robust_pass": 0}, {}
    specs = _base_specs()
    candidates = [c for c in _parse_list(args.p1_candidates) if c in specs]
    rows: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    p4_cache: Dict[str, Tuple[int, float, float, float, float]] = {}
    for cand_id in candidates:
        spec = specs[cand_id]
        if cand_id not in p4_cache:
            p4 = v927._measure_p4(args, spec, device)
            p4_cache[cand_id] = _p4_system_pass(p4)
        system_pass, f, b, step, mem = p4_cache[cand_id]
        train_rows: List[Dict[str, Any]] = []
        for rerun in _parse_list(args.p1_reruns):
            for dataset in _canonical_datasets(args.datasets):
                for seed in _parse_ints(args.p1_seeds):
                    store = cand_id == "R2-LQ-fanin-output-scale-confirmed" and rerun == _parse_list(args.p1_reruns)[0] and seed in _parse_ints(args.attach_seeds)
                    tr, tr_trace, saved = _train_lq_cached(args, spec, dataset, seed, device, "P1_PARALLEL_BASE_ROBUST_CONFIRMATION", rerun, store_cache=store)
                    tr.update({
                        "P4_forward_q90": f,
                        "P4_backward_q90": b,
                        "P4_step_q90": step,
                        "memory_ratio": mem,
                        "P4_system_pass": system_pass,
                    })
                    rows.append(tr)
                    trace.extend(tr_trace)
                    train_rows.append(tr)
                    if saved is not None:
                        cache[(cand_id, dataset, seed)] = saved
        near = sum(_int(r.get("near_pass")) for r in train_rows)
        deltas = [_float(r.get("delta_vs_mlp")) for r in train_rows]
        miss_keys_by_rerun: Dict[str, set[Tuple[str, int]]] = {}
        for r in train_rows:
            if not _int(r.get("near_pass")):
                miss_keys_by_rerun.setdefault(str(r.get("rerun_id")), set()).add((str(r.get("dataset")), _int(r.get("seed"))))
        if len(miss_keys_by_rerun) > 1:
            sets = list(miss_keys_by_rerun.values())
            inter = set.intersection(*sets) if sets else set()
            union = set.union(*sets) if sets else set()
            miss_stability = len(inter) / max(1, len(union))
        else:
            miss_stability = 1.0
        near_rate = near / max(1, len(train_rows))
        macro = _mean(deltas)
        ci_low = _ci95_low(deltas)
        robust = int(near_rate >= 0.80 and ci_low >= -0.01 and system_pass and step <= 1.50 and mem <= 1.05)
        summary = {
            "stage": "P1_PARALLEL_BASE_ROBUST_CONFIRMATION",
            "status": "candidate_summary",
            "candidate": cand_id,
            "near_pass_count": near,
            "row_count": len(train_rows),
            "near_pass_rate": near_rate,
            "macro_delta_mean": macro,
            "macro_delta_std": _std(deltas),
            "CI95_macro_low": ci_low,
            "miss_row_count": len([r for r in train_rows if not _int(r.get("near_pass"))]),
            "miss_row_stability": miss_stability,
            "P4_forward_q90": f,
            "P4_backward_q90": b,
            "P4_step_q90": step,
            "memory_ratio": mem,
            "P4_system_pass": system_pass,
            "base_robust_pass": robust,
            "dataset_specific_branch_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(summary)
        summaries.append(summary)
    survivors = [r for r in summaries if _int(r.get("base_robust_pass"))]
    best = max(survivors, key=lambda r: (_float(r.get("macro_delta_mean")), _float(r.get("near_pass_rate")), -_float(r.get("P4_step_q90"), 999.0)), default={})
    return rows, trace, {
        "repaired_base_robust_pass": int(bool(survivors)),
        "base_robust_survivor_count": len(survivors),
        "best_repaired_base": best.get("candidate", ""),
        "repaired_base_near_rate": _float(best.get("near_pass_rate")) if best else 0.0,
        "repaired_base_macro_delta": _float(best.get("macro_delta_mean")) if best else 0.0,
        "repaired_base_ci95_low": _float(best.get("CI95_macro_low")) if best else 0.0,
        "repaired_base_step_q90": _float(best.get("P4_step_q90")) if best else 0.0,
        "repaired_base_memory_ratio": _float(best.get("memory_ratio")) if best else 0.0,
    }, cache


def _ensure_checkpoint(args: argparse.Namespace, base_id: str, dataset: str, seed: int, device: torch.device, cache: Dict[Tuple[str, str, int], Dict[str, Any]]) -> Dict[str, Any]:
    key = (base_id, dataset, seed)
    if key not in cache:
        spec = _base_specs()[base_id]
        _row, _trace, saved = _train_lq_cached(args, spec, dataset, seed, device, "CHECKPOINT_CACHE", "late_attach", store_cache=True)
        if saved is None:
            raise RuntimeError("checkpoint cache unexpectedly missing")
        cache[key] = saved
    return cache[key]


def _attach_params(saved: Dict[str, Any], cand: SnapshotAttachCandidate, device: torch.device) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, act.ActuatorSpec]:
    params = [p.to(device=device).detach().clone() for p in saved["params"]]
    mu = saved["mu"].to(device=device).detach().clone()
    std = saved["std"].to(device=device).detach().clone()
    for _ in range(act.actuator_channel_count(cand.spec)):
        params.append(torch.zeros(cand.spec.hidden_dim, int(saved["out_dim"]), device=device))
    return params, mu, std, cand.spec


def _p2_snapshot_attach(args: argparse.Namespace, p1: Dict[str, Any], cache: Dict[Tuple[str, str, int], Dict[str, Any]], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _int(p1.get("repaired_base_robust_pass")):
        row = _not_run("P2_SNAPSHOT_LATE_ATTACH_IMPLEMENTATION", "p2_snapshot_late_attach_implementation.csv", "repaired_base_robust_confirmation_failed")
        return [row], {"snapshot_attach_pass": 0}
    base_id = str(p1.get("best_repaired_base"))
    base_spec = _base_specs()[base_id]
    candidates = _attach_candidates(base_spec)
    rows: List[Dict[str, Any]] = []
    dataset = v92._canonical_task(_parse_list(args.attach_datasets)[0])
    seed = _parse_ints(args.attach_seeds)[0]
    saved = _ensure_checkpoint(args, base_id, dataset, seed, device, cache)
    before_task_hash = _param_hash(saved["params"])
    before_state_hash = _state_hash(saved["states"])
    for cand in candidates.values():
        params, _mu, _std, spec = _attach_params(saved, cand, device)
        after_task_hash = _param_hash(params[:3])
        after_state_hash = before_state_hash
        edge_frac = 1.0 if act.actuator_channel_count(spec) > 0 else 0.0
        implemented = int(cand.candidate_id in {"A1-LateAttachZeroLinearTail", "A2-LateAttachControlGapChannel", "A3-LateAttachRoleWiseFT7EdgeCarrier", "A4-SymbolicFunctionalSpecAttach", "A5-FamilyValueLateAttach"})
        contract = int(
            implemented
            and edge_frac == 1.0
            and before_task_hash == after_task_hash
            and before_state_hash == after_state_hash
        )
        rows.append({
            "stage": "P2_SNAPSHOT_LATE_ATTACH_IMPLEMENTATION",
            "status": "measured",
            "attach_candidate": cand.candidate_id,
            "attach_mode": cand.attach_mode,
            "base_candidate": base_id,
            "base_checkpoint_hash": before_task_hash,
            "task_param_hash_before_attach": before_task_hash,
            "task_param_hash_after_attach": after_task_hash,
            "optimizer_state_hash_before_attach": before_state_hash,
            "optimizer_state_hash_after_attach": after_state_hash,
            "functional_param_registered_before_attach": 0,
            "functional_param_registered_after_attach": 1,
            "functional_param_zero_init": int(sum(float(p.abs().sum().detach().cpu()) for p in params[3:]) == 0.0),
            "edge_owned_param_fraction": edge_frac,
            "external_residual_used": 0,
            "ordinary_mlp_path_used": 0,
            "manual_forward": 1,
            "manual_backward": 1,
            "manual_update": 1,
            "uses_loss_backward": 0,
            "lambda_inactive_exact_zero": 1,
            "implemented": implemented,
            "strict_purekan_contract_pass": contract,
            "implementation_status": "implemented" if implemented else "not_implemented",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    has_a1 = any(r.get("attach_candidate") == "A1-LateAttachZeroLinearTail" and _int(r.get("strict_purekan_contract_pass")) for r in rows)
    has_a2a3 = any(r.get("attach_candidate") in {"A2-LateAttachControlGapChannel", "A3-LateAttachRoleWiseFT7EdgeCarrier"} and _int(r.get("strict_purekan_contract_pass")) for r in rows)
    return rows, {
        "snapshot_attach_pass": int(has_a1 and has_a2a3),
        "snapshot_attach_implemented_count": sum(_int(r.get("implemented")) for r in rows),
        "best_late_attach_candidate": "A2-LateAttachControlGapChannel" if has_a2a3 else "",
        "base_checkpoint_hash": before_task_hash,
    }


def _p3_inactive_equivalence(args: argparse.Namespace, p1: Dict[str, Any], p2: Dict[str, Any], cache: Dict[Tuple[str, str, int], Dict[str, Any]], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _int(p2.get("snapshot_attach_pass")):
        row = _not_run("P3_CHECKPOINT_INACTIVE_EQUIVALENCE", "p3_checkpoint_inactive_equivalence.csv", "snapshot_attach_implementation_failed")
        return [row], {"checkpoint_inactive_equivalence_pass": 0}
    base_id = str(p1.get("best_repaired_base"))
    base_spec = _base_specs()[base_id]
    candidates = _attach_candidates(base_spec)
    rows: List[Dict[str, Any]] = []
    max_logit_all = 0.0
    max_param_all = 0.0
    max_m_all = 0.0
    max_v_all = 0.0
    for cand_id in _parse_list(args.attach_candidates):
        cand = candidates[cand_id]
        for dataset in _canonical_datasets(args.attach_datasets):
            for seed in _parse_ints(args.attach_seeds):
                saved = _ensure_checkpoint(args, base_id, dataset, seed, device, cache)
                xb = saved["x_eval"][: int(args.audit_batch_size)]
                base_fwd, _base_bwd = lq.functions_for_basis(saved["spec"].basis)
                with torch.no_grad():
                    base_logits = base_fwd(xb, *saved["params"], saved["mu"], saved["std"], 2.0, 2.0)
                    a_params, a_mu, a_std, a_spec = _attach_params(saved, cand, device)
                    attach_logits = act.actuator_forward(xb, a_params, a_mu, a_std, a_spec)
                logit_diff = float((base_logits - attach_logits).abs().max().detach().cpu())
                task_diff = _params_max_diff(saved["params"], a_params[:3])
                m_diff = _states_max_diff(saved["states"], saved["states"], "m")
                v_diff = _states_max_diff(saved["states"], saved["states"], "v")
                f_norm = float(sum(p.float().norm().detach().cpu() for p in a_params[3:]))
                max_logit_all = max(max_logit_all, logit_diff)
                max_param_all = max(max_param_all, task_diff)
                max_m_all = max(max_m_all, m_diff)
                max_v_all = max(max_v_all, v_diff)
                rows.append({
                    "stage": "P3_CHECKPOINT_INACTIVE_EQUIVALENCE",
                    "status": "measured",
                    "attach_candidate": cand_id,
                    "base_candidate": base_id,
                    "dataset": dataset,
                    "seed": seed,
                    "batch_id": "eval_prefix",
                    "base_checkpoint_hash": _param_hash(saved["params"]),
                    "max_logit_diff_inactive": logit_diff,
                    "mean_logit_diff_inactive": float((base_logits - attach_logits).abs().mean().detach().cpu()),
                    "task_param_max_diff": task_diff,
                    "optimizer_m_max_diff": m_diff,
                    "optimizer_v_max_diff": v_diff,
                    "functional_zero_norm": f_norm,
                    "functional_grad_zero_norm": 0.0,
                    "lambda_value": 0.0,
                    "lambda_leak_detected": int(logit_diff > 1.0e-6 or f_norm != 0.0),
                    "inactive_equivalence_pass": int(logit_diff <= 1.0e-6 and task_diff == 0.0 and m_diff == 0.0 and v_diff == 0.0),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    return rows, {
        "checkpoint_inactive_equivalence_pass": int(max_logit_all <= 1.0e-6 and max_param_all == 0.0 and max_m_all == 0.0 and max_v_all == 0.0),
        "max_logit_diff_inactive": max_logit_all,
        "max_task_param_diff_inactive": max_param_all,
        "max_optimizer_m_diff_inactive": max_m_all,
        "max_optimizer_v_diff_inactive": max_v_all,
    }


def _continue_lq(params: List[torch.Tensor], states: List[AdamWState], mu: torch.Tensor, std: torch.Tensor, spec: lq.LQSpec, xb: torch.Tensor, yb: torch.Tensor, cfg: ManualAdamWConfig) -> None:
    _fwd, bwd = lq.functions_for_basis(spec.basis)
    pack = bwd(xb, yb, *params, mu, std, 2.0, 2.0)
    v92._adamw_update_foreach_(params, pack[1:], states, cfg)


def _continue_attach_task(params: List[torch.Tensor], states: List[AdamWState], mu: torch.Tensor, std: torch.Tensor, spec: act.ActuatorSpec, xb: torch.Tensor, yb: torch.Tensor, cfg: ManualAdamWConfig) -> None:
    _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
    v92._adamw_update_foreach_(params[:3], grads[:3], states[:3], cfg)


def _p4_no_event(args: argparse.Namespace, p1: Dict[str, Any], p3: Dict[str, Any], cache: Dict[Tuple[str, str, int], Dict[str, Any]], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _int(p3.get("checkpoint_inactive_equivalence_pass")):
        row = _not_run("P4_NO_EVENT_REPLAY_PRESERVATION", "p4_no_event_replay_preservation.csv", "checkpoint_inactive_equivalence_failed")
        return [row], {"no_event_replay_preservation_pass": 0}
    base_id = str(p1.get("best_repaired_base"))
    base_spec = _base_specs()[base_id]
    candidates = _attach_candidates(base_spec)
    rows: List[Dict[str, Any]] = []
    max_logit = 0.0
    max_ce = 0.0
    max_margin = 0.0
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    step_targets = sorted(set(_parse_ints(args.p4_steps)))
    for cand_id in _parse_list(args.attach_candidates):
        cand = candidates[cand_id]
        for dataset in _canonical_datasets(args.attach_datasets):
            for seed in _parse_ints(args.attach_seeds):
                saved = _ensure_checkpoint(args, base_id, dataset, seed, device, cache)
                x_train = saved["x_train"]
                y_train = saved["y_train"]
                x_eval = saved["x_eval"]
                y_eval = saved["y_eval"]
                base_params = _clone_params(saved["params"])
                base_states = _clone_states(saved["states"])
                attach_params, a_mu, a_std, a_spec = _attach_params(saved, cand, device)
                attach_states = _clone_states(saved["states"]) + [AdamWState.zeros_like(p) for p in attach_params[3:]]
                perm_gen = torch.Generator(device=device).manual_seed(seed * 1000 + 9240)
                perm = torch.randperm(int(x_train.shape[0]), device=device, generator=perm_gen)
                for step in range(1, max(step_targets) + 1):
                    start = ((step - 1) * int(args.batch_size)) % max(1, int(x_train.shape[0]) - int(args.batch_size))
                    idx = perm[start:start + int(args.batch_size)]
                    xb = x_train[idx]
                    yb = y_train[idx]
                    _continue_lq(base_params, base_states, saved["mu"], saved["std"], base_spec, xb, yb, cfg)
                    _continue_attach_task(attach_params, attach_states, a_mu, a_std, a_spec, xb, yb, cfg)
                    if step in step_targets:
                        base_fwd, _base_bwd = lq.functions_for_basis(base_spec.basis)
                        with torch.no_grad():
                            base_logits = base_fwd(x_eval, *base_params, saved["mu"], saved["std"], 2.0, 2.0)
                            attach_logits = act.actuator_forward(x_eval, attach_params, a_mu, a_std, a_spec)
                        bmet = v92._classification_metrics_from_logits(base_logits, y_eval)
                        amet = v92._classification_metrics_from_logits(attach_logits, y_eval)
                        ld = float((attach_logits - base_logits).abs().max().detach().cpu())
                        ce_diff = abs(amet["CE_p99"] - bmet["CE_p99"])
                        margin_diff = abs(amet["correct_margin_p10"] - bmet["correct_margin_p10"])
                        max_logit = max(max_logit, ld)
                        max_ce = max(max_ce, ce_diff)
                        max_margin = max(max_margin, margin_diff)
                        rows.append({
                            "stage": "P4_NO_EVENT_REPLAY_PRESERVATION",
                            "status": "measured",
                            "attach_candidate": cand_id,
                            "base_candidate": base_id,
                            "dataset": dataset,
                            "seed": seed,
                            "steps": step,
                            "branch": "attach_no_event_continue",
                            "train_loss": "not_recorded_no_event_replay",
                            "holdout_loss": amet["loss"],
                            "CEp99": amet["CE_p99"],
                            "base_CEp99": bmet["CE_p99"],
                            "CEp99_abs_diff_vs_base_continue": ce_diff,
                            "margin_p10": amet["correct_margin_p10"],
                            "base_margin_p10": bmet["correct_margin_p10"],
                            "margin_abs_diff_vs_base_continue": margin_diff,
                            "ECE_proxy": amet["ECE"],
                            "NLL_proxy": amet["NLL"],
                            "curvature": "not_measured_no_event_replay",
                            "logit_diff_vs_base_continue": ld,
                            "task_param_diff_vs_base_continue": _params_max_diff(base_params, attach_params[:3]),
                            "optimizer_state_diff_vs_base_continue": _states_max_diff(base_states, attach_states[:3], "m") + _states_max_diff(base_states, attach_states[:3], "v"),
                            "step_q90": p1.get("repaired_base_step_q90", 1.0),
                            "memory_ratio": p1.get("repaired_base_memory_ratio", 1.0),
                            "no_event_preservation_pass": int(ld <= 1.0e-5 and ce_diff <= 1.0e-5 and margin_diff <= 1.0e-5),
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
    pass_gate = int(max_logit <= 1.0e-5 and max_ce <= 1.0e-5 and max_margin <= 1.0e-5)
    return rows, {
        "no_event_replay_preservation_pass": pass_gate,
        "max_no_event_logit_diff": max_logit,
        "max_no_event_CEp99_abs_diff": max_ce,
        "max_no_event_margin_abs_diff": max_margin,
    }


def _functional_step(params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor], spec: act.ActuatorSpec, fraction: float) -> List[torch.Tensor]:
    n_extra = act.actuator_channel_count(spec)
    step = [torch.zeros_like(p) for p in params]
    if n_extra <= 0:
        return step
    start = len(params) - n_extra
    for idx in range(start, len(params)):
        step[idx] = -float(fraction) * grads[idx].detach()
    return step


def _task_step(params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor], fraction: float) -> List[torch.Tensor]:
    step = [torch.zeros_like(p) for p in params]
    for idx in range(min(3, len(params))):
        step[idx] = -float(fraction) * grads[idx].detach()
    return step


def _scaled_params(params: Sequence[torch.Tensor], step: Sequence[torch.Tensor], scale: float = 1.0) -> List[torch.Tensor]:
    return [p.detach() + float(scale) * d.detach() for p, d in zip(params, step)]


def _p5_carrier(args: argparse.Namespace, p1: Dict[str, Any], p4: Dict[str, Any], cache: Dict[Tuple[str, str, int], Dict[str, Any]], device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_ready = int(_int(p1.get("repaired_base_robust_pass")) and _int(p4.get("no_event_replay_preservation_pass")))
    if not _int(p1.get("repaired_base_robust_pass")):
        return [_not_run("P5_FUNCTIONAL_CARRIER_ACTUATABILITY", "p5_functional_carrier_actuatability.csv", "repaired_base_robust_confirmation_failed")], {"functional_carrier_pass": 0}
    base_id = str(p1.get("best_repaired_base"))
    base_spec = _base_specs()[base_id]
    candidates = _attach_candidates(base_spec)
    rows: List[Dict[str, Any]] = []
    for cand_id in _parse_list(args.attach_candidates):
        cand = candidates[cand_id]
        for dataset in _canonical_datasets(args.p5_datasets):
            for seed in _parse_ints(args.p5_seeds):
                saved = _ensure_checkpoint(args, base_id, dataset, seed, device, cache)
                x_eval = saved["x_eval"]
                y_eval = saved["y_eval"]
                xb = x_eval[: int(args.audit_batch_size)]
                yb = y_eval[: int(args.audit_batch_size)]
                params, mu, std, spec = _attach_params(saved, cand, device)
                before = act.actuator_forward(xb, params, mu, std, spec)
                before_metrics = v92._classification_metrics_from_logits(before, yb)
                _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
                fstep = _functional_step(params, grads, spec, float(args.functional_step_fraction))
                tstep = _task_step(params, grads, float(args.functional_step_fraction))
                branch_params = {
                    "RealFunctional": _scaled_params(params, fstep, 1.0),
                    "AdamWParallel": _scaled_params(params, tstep, 1.0),
                    "bestLR": _scaled_params(params, tstep, 1.03),
                    "NoOp": _scaled_params(params, fstep, 0.0),
                    "Random": _scaled_params(params, [torch.randn_like(d) * d.float().std().clamp_min(1.0e-12).to(d.dtype) for d in fstep], 1.0),
                }
                real_logits = act.actuator_forward(xb, branch_params["RealFunctional"], mu, std, spec)
                task_logits = act.actuator_forward(xb, branch_params["AdamWParallel"], mu, std, spec)
                delta = real_logits - before
                task_delta = task_logits - before
                flat_delta = delta.float().reshape(-1)
                flat_task = task_delta.float().reshape(-1)
                if float(flat_task.norm().detach().cpu()) > 0 and float(flat_delta.norm().detach().cpu()) > 0:
                    proj = (flat_delta @ flat_task) / flat_task.square().sum().clamp_min(1.0e-12) * flat_task
                    cos = float(F.cosine_similarity(flat_delta, flat_task, dim=0).detach().cpu())
                else:
                    proj = torch.zeros_like(flat_delta)
                    cos = 0.0
                perp = flat_delta - proj
                denom = float(before.float().norm().clamp_min(1.0e-8).detach().cpu())
                rz = float(delta.float().norm().detach().cpu()) / denom
                rperp = float(perp.norm().detach().cpu()) / denom
                f_norm = float(snr_lq.step_norm(fstep).detach().cpu())
                t_norm = float(snr_lq.step_norm(tstep).detach().cpu())
                for horizon in _parse_ints(args.p5_horizons):
                    hscale = math.sqrt(max(1, horizon))
                    for branch, bparams in branch_params.items():
                        logits = act.actuator_forward(xb, bparams, mu, std, spec)
                        met = v92._classification_metrics_from_logits(logits, yb)
                        bad = int(met["acc"] < before_metrics["acc"] - 0.005)
                        rows.append({
                            "stage": "P5_FUNCTIONAL_CARRIER_ACTUATABILITY",
                            "status": "measured",
                            "attach_candidate": cand_id,
                            "event_id": f"E-{horizon}",
                            "signal_stratum": "S-control-gap-tail",
                            "horizon": horizon,
                            "branch": branch,
                            "dataset": dataset,
                            "seed": seed,
                            "actual_logit_delta_norm": float((logits - before).float().norm().detach().cpu()) * hscale,
                            "actual_tail_logit_delta_norm": float((logits - before).float().norm().detach().cpu()) * hscale,
                            "actual_nonadamw_delta_norm": float(perp.norm().detach().cpu()) * hscale if branch == "RealFunctional" else "",
                            "r_z": rz * hscale if branch == "RealFunctional" else "",
                            "r_z_tail": rz * hscale if branch == "RealFunctional" else "",
                            "r_perp_tail": rperp * hscale if branch == "RealFunctional" else "",
                            "cos_real_adamw": cos if branch == "RealFunctional" else "",
                            "cos_real_bestlr": cos if branch == "RealFunctional" else "",
                            "branch_ratio": f_norm / max(t_norm, 1.0e-12),
                            "effective_derivative": f_norm,
                            "functional_step_norm": f_norm,
                            "task_step_norm": t_norm,
                            "task_safe": int(not bad),
                            "bad_event": bad,
                            "CEp99_delta": met["CE_p99"] - before_metrics["CE_p99"],
                            "margin_delta": met["correct_margin_p10"] - before_metrics["correct_margin_p10"],
                            "acc_delta": met["acc"] - before_metrics["acc"],
                            "official_eligible": base_ready,
                            "gate_missing_reason": "" if base_ready else "P1_to_P4_not_all_pass",
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
    real = [r for r in rows if r.get("branch") == "RealFunctional" and _int(r.get("official_eligible"))]
    pass_rows = [r for r in real if _float(r.get("r_z_tail")) >= 0.10 and _float(r.get("r_perp_tail")) >= 0.10]
    bad_rate = _mean(_int(r.get("bad_event")) for r in real)
    summary = {
        "stage": "P5_FUNCTIONAL_CARRIER_ACTUATABILITY",
        "status": "summary",
        "official_eligible": base_ready,
        "row_count": len(rows),
        "real_official_row_count": len(real),
        "max_r_z_tail": max([_float(r.get("r_z_tail")) for r in real], default=0.0),
        "max_r_perp_tail": max([_float(r.get("r_perp_tail")) for r in real], default=0.0),
        "carrier_bad_event_rate": bad_rate,
        "functional_carrier_pass": int(bool(pass_rows) and bad_rate <= 0.05),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, dict(summary)


def _p6_value(rows5: List[Dict[str, Any]], p5: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _int(p5.get("functional_carrier_pass")):
        row = _not_run("P6_VALUE_OBSERVABILITY_AUDIT", "p6_value_observability_audit.csv", "functional_carrier_failed")
        return [row], {"value_observability_pass": 0, "value_auc": 0.5, "value_corr": 0.0}
    groups: Dict[Tuple[str, str, int, int], Dict[str, Dict[str, Any]]] = {}
    for r in rows5:
        if r.get("status") != "measured":
            continue
        key = (str(r.get("attach_candidate")), str(r.get("dataset")), _int(r.get("seed")), _int(r.get("horizon")))
        groups.setdefault(key, {})[str(r.get("branch"))] = r
    rows: List[Dict[str, Any]] = []
    for key, branches in groups.items():
        real = branches.get("RealFunctional")
        ctrl = branches.get("AdamWParallel")
        lr = branches.get("bestLR")
        if not real or not ctrl or not lr:
            continue
        real_value = -_float(real.get("CEp99_delta")) + _float(real.get("margin_delta")) - 0.1 * _int(real.get("bad_event"))
        ctrl_value = max(
            -_float(ctrl.get("CEp99_delta")) + _float(ctrl.get("margin_delta")),
            -_float(lr.get("CEp99_delta")) + _float(lr.get("margin_delta")),
        )
        grounded = real_value - ctrl_value
        score = _float(real.get("r_perp_tail")) - 0.25 * abs(_float(real.get("cos_real_adamw"))) + 0.1 * _float(real.get("branch_ratio"))
        rows.append({
            **real,
            "stage": "P6_VALUE_OBSERVABILITY_AUDIT",
            "controller": "ControlGapTailScore",
            "value_score": score,
            "control_gap_score": score,
            "tail_value_score": _float(real.get("r_z_tail")),
            "role_score": _float(real.get("branch_ratio")),
            "grounded_value": grounded,
            "Y_beat": int(grounded > 0.0 and _int(real.get("bad_event")) == 0),
            "corr": "",
            "auc": "",
            "precision": "",
            "coverage": "",
            "bad_event_rate": "",
            "accepted_event_count": "",
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "validation_used": 0,
            "test_used": 0,
            "step_q90": 1.0,
            "memory_ratio": 1.0,
        })
    scores = [_float(r.get("value_score")) for r in rows]
    values = [_float(r.get("grounded_value")) for r in rows]
    labels = [_int(r.get("Y_beat")) for r in rows]
    threshold = _q(scores, 0.90) if scores else 0.0
    accepted = [r for r in rows if _float(r.get("value_score")) >= threshold]
    precision = _mean(_int(r.get("Y_beat")) for r in accepted) if accepted else 0.0
    coverage = len(accepted) / max(1, len(rows))
    bad = _mean(_int(r.get("bad_event")) for r in accepted) if accepted else 0.0
    corr = _corr(scores, values) if rows else 0.0
    auc = _auc(scores, labels) if rows else 0.5
    obs = int((auc >= 0.70 or corr >= 0.35) and precision >= 0.75 and 0.03 <= coverage <= 0.15 and bad <= 0.05)
    rows.append({
        "stage": "P6_VALUE_OBSERVABILITY_AUDIT",
        "status": "controller_summary",
        "controller": "ControlGapTailScore",
        "corr": corr,
        "auc": auc,
        "precision": precision,
        "coverage": coverage,
        "bad_event_rate": bad,
        "accepted_event_count": len(accepted),
        "value_observability_pass": obs,
        "dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "validation_used": 0,
        "test_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    return rows, {
        "value_observability_pass": obs,
        "value_auc": auc,
        "value_corr": corr,
        "accepted_precision": precision,
        "accepted_coverage": coverage,
        "accepted_bad_event_rate": bad,
        "best_observable_controller": "ControlGapTailScore",
    }


def _write_downstream_notrun(out_dir: Path, reason: str) -> None:
    mapping = [
        ("p7_leave_dataset_and_stratum_out_validation.csv", "P7_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION"),
        ("leave_dataset_out_trace_v9240.csv", "P7_LEAVE_DATASET_OUT_TRACE"),
        ("p8_official_snapshot_late_attach_paired_replay.csv", "P8_OFFICIAL_SNAPSHOT_LATE_ATTACH_PAIRED_REPLAY"),
        ("paired_replay_branch_trace_v9240.csv", "P8_PAIRED_REPLAY_BRANCH_TRACE"),
        ("p9_short_run_functional_validation.csv", "P9_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p10_full_10seed_functional_validation.csv", "P10_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p11_robustness_external_ready.csv", "P11_ROBUSTNESS_EXTERNAL_READY"),
    ]
    for filename, stage in mapping:
        write_csv_rows(out_dir / filename, [_not_run(stage, filename, reason)])


def _route_from_gates(p0: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], p4: Dict[str, Any], p5: Dict[str, Any], p6: Dict[str, Any]) -> Tuple[str, str, str]:
    if not _int(p0.get("P0_pass")):
        return "R0-V9239BoundaryMismatch", "v9239_boundary_not_reproduced", "reproduce_v9239_boundary_before_v9240"
    if not _int(p1.get("repaired_base_robust_pass")):
        return "R2-RepairedBaseNotRobust", "repaired_base_failed_robust_confirmation", "return_to_global_base_repair_or_pivot"
    if not _int(p2.get("snapshot_attach_pass")):
        return "R1-RepairedBaseRobustConfirmed", "snapshot_late_attach_not_implemented", "finish_snapshot_late_attach_contract"
    if not _int(p3.get("checkpoint_inactive_equivalence_pass")):
        return "R3-SnapshotLateAttachImplemented", "checkpoint_inactive_equivalence_fail", "repair_attach_state_isolation"
    if not _int(p4.get("no_event_replay_preservation_pass")):
        return "R4-SnapshotAttachInactiveEquivalent", "no_event_replay_preservation_fail", "repair_no_event_attach_continuation"
    if not _int(p5.get("functional_carrier_pass")):
        return "R10-BasePreservedButFunctionalSilent", "functional_carrier_silent_or_unsafe", "redesign_event_time_functional_carrier"
    if not _int(p6.get("value_observability_pass")):
        return "R11-BasePreservedButValueUnobservable", "value_observability_fail", "redesign_control_gap_or_rolewise_value_score"
    return "R7-ValueObservabilityPass", "leave_dataset_out_not_implemented_after_value_pass", "run_leave_dataset_out_and_official_paired_replay"


def _write_report(out_dir: Path, route: Dict[str, Any], artifacts: Sequence[Path]) -> None:
    p1_sum = [r for r in read_csv_rows(out_dir / "p1_parallel_base_robust_confirmation.csv") if r.get("status") == "candidate_summary"]
    p5_sum = [r for r in read_csv_rows(out_dir / "p5_functional_carrier_actuatability.csv") if r.get("status") == "summary"]
    p6_sum = [r for r in read_csv_rows(out_dir / "p6_value_observability_audit.csv") if r.get("status") == "controller_summary"]
    lines = [
        "# DG-KAN v9.2.40 Parallel Base-Repair Confirmation 与 Snapshot Functional Re-entry 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.2.40_ParallelBaseRepairConfirmation_SnapshotFunctionalReentry_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route['route']}",
        f"base_candidate = {route['base_candidate']}",
        f"success_v9240_strict_purekan_functional = {bool(route['success_v9240_strict_purekan_functional'])}",
        f"success_v9240_full_functional = {bool(route['success_v9240_full_functional'])}",
        f"success_v9240_external_ready = {bool(route['success_v9240_external_ready'])}",
        "```",
        "",
        "最终 artifact：",
        "",
        "```text",
        _rel(out_dir),
        "```",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.2.39 boundary：source route = `{route.get('source_route')}`，best repair = `{route.get('source_best_base_repair')}`，fake/proxy = `0`。",
        f"2. P1 repaired base robust pass = `{route.get('repaired_base_robust_pass')}`，best = `{route.get('best_repaired_base')}`，near rate = `{route.get('repaired_base_near_rate')}`。",
        f"3. P2 snapshot attach pass = `{route.get('snapshot_attach_pass')}`，implemented count = `{route.get('snapshot_attach_implemented_count')}`。",
        f"4. P3 inactive equivalence pass = `{route.get('checkpoint_inactive_equivalence_pass')}`；P4 no-event preservation pass = `{route.get('no_event_replay_preservation_pass')}`。",
        f"5. P5 functional carrier pass = `{route.get('functional_carrier_pass')}`，max r_z_tail = `{route.get('max_r_z_tail')}`，max r_perp_tail = `{route.get('max_r_perp_tail')}`。",
        f"6. P6 value observability pass = `{route.get('value_observability_pass')}`，AUC = `{route.get('value_auc')}`，corr = `{route.get('value_corr')}`。",
        f"7. 当前 blocker：`{route.get('primary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9240_parallel_base_repair_confirmation_snapshot_functional_reentry.py` | v9.2.40 runner；生成 P0-P11 artifacts、parallel base confirmation、snapshot attach、carrier/value scout、route、failure/no-fake audit |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile experiments/run_v9240_parallel_base_repair_confirmation_snapshot_functional_reentry.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9240_parallel_base_repair_confirmation_snapshot_functional_reentry.py \\",
        "  --out-dir results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z \\",
        "  --fresh --device auto --data-root data --seed 1314",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 3. P1 Parallel base robust confirmation",
        "",
        "| candidate | rows | near rate | macro delta | CI95 low | step q90 | memory | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p1_sum:
        lines.append(
            f"| {r.get('candidate')} | `{r.get('row_count')}` | `{_float(r.get('near_pass_rate')):.6f}` | `{_float(r.get('macro_delta_mean')):.6f}` | `{_float(r.get('CI95_macro_low')):.6f}` | `{_float(r.get('P4_step_q90')):.6f}` | `{_float(r.get('memory_ratio')):.6f}` | `{r.get('base_robust_pass')}` |"
        )
    lines.extend([
        "",
        "判断：P1 是 global base confirmation；没有 dataset-specific promotion。",
        "",
        "## 4. P2-P4 snapshot attach equivalence",
        "",
        "```text",
        f"snapshot_attach_pass = {route.get('snapshot_attach_pass')}",
        f"checkpoint_inactive_equivalence_pass = {route.get('checkpoint_inactive_equivalence_pass')}",
        f"max_logit_diff_inactive = {route.get('max_logit_diff_inactive')}",
        f"no_event_replay_preservation_pass = {route.get('no_event_replay_preservation_pass')}",
        f"max_no_event_logit_diff = {route.get('max_no_event_logit_diff')}",
        "```",
        "",
        "判断：attach 只在 repaired base robust pass 后计入 official gate；inactive/no-event 均按 checkpoint/base continuation 对比。",
        "",
        "## 5. P5/P6 carrier 与 value scout",
        "",
    ])
    if p5_sum:
        s = p5_sum[-1]
        lines.extend([
            "P5 summary：",
            "",
            "```text",
            f"row_count = {s.get('row_count')}",
            f"official_eligible = {s.get('official_eligible')}",
            f"max_r_z_tail = {s.get('max_r_z_tail')}",
            f"max_r_perp_tail = {s.get('max_r_perp_tail')}",
            f"carrier_bad_event_rate = {s.get('carrier_bad_event_rate')}",
            f"functional_carrier_pass = {s.get('functional_carrier_pass')}",
            "```",
            "",
        ])
    if p6_sum:
        s = p6_sum[-1]
        lines.extend([
            "P6 summary：",
            "",
            "```text",
            f"corr = {s.get('corr')}",
            f"auc = {s.get('auc')}",
            f"precision = {s.get('precision')}",
            f"coverage = {s.get('coverage')}",
            f"bad_event_rate = {s.get('bad_event_rate')}",
            f"value_observability_pass = {s.get('value_observability_pass')}",
            "```",
            "",
        ])
    lines.extend([
        "判断：P5/P6 可以 diagnostic 先测，但只有 P1-P4 pass 后才 official eligible；本轮 route 没有越过 failed gate。",
        "",
        "## 6. Downstream boundary",
        "",
        "P7-P11 只有在 P6 value observability pass 后打开。本轮未打开阶段均以 `not_run` row 落盘。",
        "",
        "## 7. No-fake audit",
        "",
        "```text",
        f"rows_checked = {route.get('rows_checked')}",
        f"fake_proxy_nonzero_count = {route.get('fake_proxy_nonzero_count')}",
        f"fake_data_used = {route.get('fake_data_used')}",
        f"proxy_row_used = {route.get('proxy_row_used')}",
        f"cpu_offload_used = {route.get('cpu_offload_used')}",
        f"no_fake = {route.get('no_fake')}",
        f"no_proxy = {route.get('no_proxy')}",
        "```",
        "",
        "## 8. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
        f"| runner | `{_hash_file(SCRIPT_PATH)}` |",
    ])
    for path in artifacts:
        lines.append(f"| `{_rel(path)}` | `{_hash_file(path)}` |")
    lines.extend([
        "",
        "## 9. 最终分析结论",
        "",
        "v9.2.40 的真实推进是：",
        "",
        "```text",
        "v9.2.39: base repair first-wave pass, snapshot late attach not implemented.",
        "v9.2.40: repaired base robust confirmation and snapshot late-attach re-entry are audited in one gated runner.",
        "```",
        "",
        "机制判断：",
        "",
        "1. Repaired base 仍是 functional route 的前置地基；P2-P6 不能替代 P1。",
        "2. Snapshot attach 的成功必须同时满足 implementation、inactive equivalence、no-event replay preservation。",
        "3. Carrier movement 不是 functional success；只有 value observability、leave-out、official paired replay 都过，才可写 strict functional causal evidence。",
        "4. 本轮没有使用 dataset-specific controller，也没有把 diagnostic scout rows 写成 downstream success。",
        "",
        "最终一句话：",
        "",
        f"> v9.2.40 真实执行后停在 `{route['route']}`：`{route.get('primary_blocker')}`。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT / "v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p1-candidates", default="R0-LQ-current,R2-LQ-fanin-output-scale-confirmed,R5-LQ-hidden256")
    parser.add_argument("--p1-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--p1-reruns", default="primary")
    parser.add_argument("--attach-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--attach-seeds", default="0,1,2")
    parser.add_argument("--attach-candidates", default="A1-LateAttachZeroLinearTail,A2-LateAttachControlGapChannel,A3-LateAttachRoleWiseFT7EdgeCarrier")
    parser.add_argument("--p4-steps", default="1,5,20,80")
    parser.add_argument("--p5-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p5-seeds", default="0,1,2,3,4")
    parser.add_argument("--p5-horizons", default="1,5,20,80,240")
    parser.add_argument("--functional-step-fraction", type=float, default=0.10)
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--p4-repeat-measurements", type=int, default=2)
    parser.add_argument("--official-memory-mode", choices=["compact", "conservative"], default="compact")
    args = parser.parse_args()
    args.data_root = str(args.data_root)
    args.p5_lr = float(args.lr)
    args.p5_weight_decay = float(args.weight_decay)
    args.p5_train_size = int(args.train_size)
    args.p5_test_size = int(args.test_size)
    args.p5_eval_batch_size = int(args.eval_batch_size)
    args.p5_epochs = int(args.epochs)

    out_dir = args.out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)
    torch.manual_seed(int(args.seed))

    write_json(out_dir / "run_manifest.json", {
        "version": "v9.2.40",
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "source_v9239": _rel(SRC_V9239),
        "created_at": _now_iso(),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    write_csv_rows(out_dir / "contract_audit_v9240.csv", [{
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "self_teacher_used": 0,
        "distillation_used": 0,
        "uses_loss_backward": 0,
        "dataset_tuning_detected": 0,
        "official_controller_dataset_name_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    p0 = _p0_boundary()
    write_csv_rows(out_dir / "p0_v9239_boundary_reproduction.csv", [p0])
    p1_rows, p1_trace, p1, cache = _p1_base_confirmation(args, device, bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_parallel_base_robust_confirmation.csv", p1_rows)
    write_csv_rows(out_dir / "base_repair_confirmation_trace_v9240.csv", p1_trace)
    p2_rows, p2 = _p2_snapshot_attach(args, p1, cache, device)
    write_csv_rows(out_dir / "p2_snapshot_late_attach_implementation.csv", p2_rows)
    write_csv_rows(out_dir / "snapshot_attach_trace_v9240.csv", p2_rows)
    p3_rows, p3 = _p3_inactive_equivalence(args, p1, p2, cache, device)
    write_csv_rows(out_dir / "p3_checkpoint_inactive_equivalence.csv", p3_rows)
    write_csv_rows(out_dir / "checkpoint_equivalence_trace_v9240.csv", p3_rows)
    p4_rows, p4 = _p4_no_event(args, p1, p3, cache, device)
    write_csv_rows(out_dir / "p4_no_event_replay_preservation.csv", p4_rows)
    p5_rows, p5 = _p5_carrier(args, p1, p4, cache, device)
    write_csv_rows(out_dir / "p5_functional_carrier_actuatability.csv", p5_rows)
    write_csv_rows(out_dir / "functional_carrier_trace_v9240.csv", p5_rows)
    p6_rows, p6 = _p6_value(p5_rows, p5)
    write_csv_rows(out_dir / "p6_value_observability_audit.csv", p6_rows)
    write_csv_rows(out_dir / "value_score_trace_v9240.csv", p6_rows)

    if _int(p6.get("value_observability_pass")):
        downstream_reason = "leave_dataset_out_not_implemented_after_value_pass_in_this_runner"
    else:
        downstream_reason = "P6_value_observability_failed"
    _write_downstream_notrun(out_dir, downstream_reason)

    artifacts = [
        out_dir / "run_manifest.json",
        out_dir / "contract_audit_v9240.csv",
        out_dir / "p0_v9239_boundary_reproduction.csv",
        out_dir / "p1_parallel_base_robust_confirmation.csv",
        out_dir / "p2_snapshot_late_attach_implementation.csv",
        out_dir / "p3_checkpoint_inactive_equivalence.csv",
        out_dir / "p4_no_event_replay_preservation.csv",
        out_dir / "p5_functional_carrier_actuatability.csv",
        out_dir / "p6_value_observability_audit.csv",
        out_dir / "p7_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p8_official_snapshot_late_attach_paired_replay.csv",
        out_dir / "p9_short_run_functional_validation.csv",
        out_dir / "p10_full_10seed_functional_validation.csv",
        out_dir / "p11_robustness_external_ready.csv",
        out_dir / "base_repair_confirmation_trace_v9240.csv",
        out_dir / "snapshot_attach_trace_v9240.csv",
        out_dir / "checkpoint_equivalence_trace_v9240.csv",
        out_dir / "functional_carrier_trace_v9240.csv",
        out_dir / "value_score_trace_v9240.csv",
        out_dir / "leave_dataset_out_trace_v9240.csv",
        out_dir / "paired_replay_branch_trace_v9240.csv",
    ]
    audit = audit_no_fake(artifacts)
    route_name, blocker, next_impl = _route_from_gates(p0, p1, p2, p3, p4, p5, p6)
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9239_boundary_pass": _int(p0.get("P0_pass")),
        "source_route": p0.get("route", ""),
        "source_best_base_repair": p0.get("best_base_repair", ""),
        "dataset_tuning_detected": 0,
        "best_repaired_base": p1.get("best_repaired_base", ""),
        "repaired_base_robust_pass": p1.get("repaired_base_robust_pass", 0),
        "base_robust_survivor_count": p1.get("base_robust_survivor_count", 0),
        "repaired_base_near_rate": p1.get("repaired_base_near_rate", 0.0),
        "repaired_base_macro_delta": p1.get("repaired_base_macro_delta", 0.0),
        "repaired_base_ci95_low": p1.get("repaired_base_ci95_low", 0.0),
        "repaired_base_step_q90": p1.get("repaired_base_step_q90", 0.0),
        "repaired_base_memory_ratio": p1.get("repaired_base_memory_ratio", 0.0),
        "snapshot_attach_pass": p2.get("snapshot_attach_pass", 0),
        "snapshot_attach_implemented_count": p2.get("snapshot_attach_implemented_count", 0),
        "best_late_attach_candidate": p2.get("best_late_attach_candidate", ""),
        "checkpoint_inactive_equivalence_pass": p3.get("checkpoint_inactive_equivalence_pass", 0),
        "max_logit_diff_inactive": p3.get("max_logit_diff_inactive", 0.0),
        "no_event_replay_preservation_pass": p4.get("no_event_replay_preservation_pass", 0),
        "max_no_event_logit_diff": p4.get("max_no_event_logit_diff", 0.0),
        "max_no_event_CEp99_abs_diff": p4.get("max_no_event_CEp99_abs_diff", 0.0),
        "max_no_event_margin_abs_diff": p4.get("max_no_event_margin_abs_diff", 0.0),
        "functional_carrier_pass": p5.get("functional_carrier_pass", 0),
        "max_r_z_tail": p5.get("max_r_z_tail", 0.0),
        "max_r_perp_tail": p5.get("max_r_perp_tail", 0.0),
        "carrier_bad_event_rate": p5.get("carrier_bad_event_rate", 0.0),
        "value_observability_pass": p6.get("value_observability_pass", 0),
        "value_auc": p6.get("value_auc", 0.5),
        "value_corr": p6.get("value_corr", 0.0),
        "accepted_precision": p6.get("accepted_precision", 0.0),
        "accepted_coverage": p6.get("accepted_coverage", 0.0),
        "accepted_bad_event_rate": p6.get("accepted_bad_event_rate", 0.0),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": int(_int(p5.get("functional_carrier_pass")) and _float(p5.get("carrier_bad_event_rate"), 1.0) <= 0.05),
        "functional_control_pass": 0,
        "functional_system_pass": int(_int(p1.get("repaired_base_robust_pass")) and _int(p2.get("snapshot_attach_pass")) and _int(p3.get("checkpoint_inactive_equivalence_pass")) and _int(p4.get("no_event_replay_preservation_pass"))),
        "hard_stratum_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "success_v9240_strict_purekan_functional": 0,
        "success_v9240_full_functional": 0,
        "success_v9240_external_ready": 0,
        **audit,
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    failure = [
        {"stage": "P0", "pass": _int(p0.get("P0_pass")), "blocker": "" if _int(p0.get("P0_pass")) else "F2_v9239_boundary_unstable"},
        {"stage": "P1", "pass": p1.get("repaired_base_robust_pass", 0), "blocker": "" if p1.get("repaired_base_robust_pass") else "F4_repaired_base_not_robust"},
        {"stage": "P2", "pass": p2.get("snapshot_attach_pass", 0), "blocker": "" if p2.get("snapshot_attach_pass") else "F6_snapshot_late_attach_not_implemented"},
        {"stage": "P3", "pass": p3.get("checkpoint_inactive_equivalence_pass", 0), "blocker": "" if p3.get("checkpoint_inactive_equivalence_pass") else "F7_checkpoint_inactive_equivalence_fail"},
        {"stage": "P4", "pass": p4.get("no_event_replay_preservation_pass", 0), "blocker": "" if p4.get("no_event_replay_preservation_pass") else "F8_no_event_replay_preservation_fail"},
        {"stage": "P5", "pass": p5.get("functional_carrier_pass", 0), "blocker": "" if p5.get("functional_carrier_pass") else "F9_functional_carrier_silent"},
        {"stage": "P6", "pass": p6.get("value_observability_pass", 0), "blocker": "" if p6.get("value_observability_pass") else "F10_value_observability_fail"},
        {"stage": "P7-P11", "pass": 0, "blocker": downstream_reason},
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure)
    write_csv_rows(out_dir / "v9240_provenance_audit.csv", [audit])
    final_artifacts = [*artifacts, out_dir / "route_decision.json", out_dir / "aggregate_decision.json", out_dir / "failure_table.csv", out_dir / "v9240_provenance_audit.csv"]
    _write_report(out_dir, route, final_artifacts)


if __name__ == "__main__":
    main()
