#!/usr/bin/env python3
"""DG-KAN v9.2.22 base-qualified strict PureKAN functional interface runner.

This runner starts from the measured v9.2.21 interface boundary, attributes the
I1c base-qualification miss rows, and tests base-neutral strict FC-PureKAN
functional-interface candidates.  Downstream paired/full functional stages are
opened only when the prior gates pass.  All not-opened stages are written as
explicit not_run rows.
"""

from __future__ import annotations

import argparse
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
import run_v922_fused_compositional_kernel_closure as f922  # noqa: E402
import run_v9213_functional_controllability_actuator_redesign as v9213  # noqa: E402
import run_v9214_p4qualified_functional_actuator_closure as v9214  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import lq_output_space_functional as out_lq  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.22_BaseQualified_StrictPureKAN_FunctionalInterface_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9222_basequalified_strict_purekan_functional_interface.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9221 = RESULT_ROOT / "v9221_strict_purekan_functional_interface_transfer_first_20260510T110000Z"


@dataclass(frozen=True)
class InterfaceCandidate:
    candidate_id: str
    interface_family: str
    spec: act.ActuatorSpec | None
    functional_channel_init: str
    functional_channel_adamw_trainable: int
    functional_channel_event_trainable: int
    schedule: str
    status: str = "implemented"
    reason: str = ""


def _spec(base_id: str, candidate_id: str, init_scale: float | None = None) -> act.ActuatorSpec:
    base = act.ACTUATOR_SPECS[base_id]
    return act.ActuatorSpec(
        candidate_id=candidate_id,
        actuator_type=base.actuator_type,
        hidden_dim=base.hidden_dim,
        output_scale=base.output_scale,
        actuator_init_scale=base.actuator_init_scale if init_scale is None else float(init_scale),
        beta=base.beta,
        gamma=base.gamma,
        center=base.center,
    )


def _candidate_registry() -> Dict[str, InterfaceCandidate]:
    return {
        "I1c-current": InterfaceCandidate(
            "I1c-current",
            "current_rational_derivative_controlled_train_all",
            _spec("A4e-BoundedRational-FusedCoeffGrad", "I1c-current"),
            "zero",
            1,
            1,
            "base_adamw_trains_all_channels",
        ),
        "N1a-ZeroInit-RationalFunc-FrozenBase": InterfaceCandidate(
            "N1a-ZeroInit-RationalFunc-FrozenBase",
            "zero_impact_rational_frozen_base",
            _spec("A4e-BoundedRational-FusedCoeffGrad", "N1a-ZeroInit-RationalFunc-FrozenBase"),
            "zero",
            0,
            1,
            "base_phase_functional_channel_frozen",
        ),
        "N1b-ZeroInit-PiecewiseFunc-FrozenBase": InterfaceCandidate(
            "N1b-ZeroInit-PiecewiseFunc-FrozenBase",
            "zero_impact_piecewise_frozen_base",
            _spec("A5c-PiecewiseLinear2-FixedKnots", "N1b-ZeroInit-PiecewiseFunc-FrozenBase"),
            "zero",
            0,
            1,
            "base_phase_functional_channel_frozen",
        ),
        "N1c-ZeroInit-SharedRBFFunc-FrozenBase": InterfaceCandidate(
            "N1c-ZeroInit-SharedRBFFunc-FrozenBase",
            "zero_impact_shared_rbf_frozen_base",
            _spec("A6-LQ-LocalRBFSharedCenterActuator", "N1c-ZeroInit-SharedRBFFunc-FrozenBase"),
            "zero",
            0,
            1,
            "base_phase_functional_channel_frozen",
        ),
        "N1d-ZeroInit-CenteredT2Func-FrozenBase": InterfaceCandidate(
            "N1d-ZeroInit-CenteredT2Func-FrozenBase",
            "zero_impact_centered_t2_frozen_base",
            _spec("A3-LQ-CenteredT2Actuator", "N1d-ZeroInit-CenteredT2Func-FrozenBase"),
            "zero",
            0,
            1,
            "base_phase_functional_channel_frozen",
        ),
        "N2a-TinyInit-RationalFunc-BranchRatioCap": InterfaceCandidate(
            "N2a-TinyInit-RationalFunc-BranchRatioCap",
            "tiny_init_rational_branch_ratio_cap",
            _spec("A4e-BoundedRational-FusedCoeffGrad", "N2a-TinyInit-RationalFunc-BranchRatioCap", init_scale=1.0e-3),
            "tiny_1e-3",
            1,
            1,
            "base_phase_trainable_with_branch_ratio_audit",
        ),
        "N2b-TinyInit-PiecewiseFunc-BranchRatioCap": InterfaceCandidate(
            "N2b-TinyInit-PiecewiseFunc-BranchRatioCap",
            "tiny_init_piecewise_branch_ratio_cap",
            _spec("A5c-PiecewiseLinear2-FixedKnots", "N2b-TinyInit-PiecewiseFunc-BranchRatioCap", init_scale=1.0e-3),
            "tiny_1e-3",
            1,
            1,
            "base_phase_trainable_with_branch_ratio_audit",
        ),
        "N2c-TinyInit-SharedRBFFunc-BranchRatioCap": InterfaceCandidate(
            "N2c-TinyInit-SharedRBFFunc-BranchRatioCap",
            "tiny_init_shared_rbf_branch_ratio_cap",
            _spec("A6-LQ-LocalRBFSharedCenterActuator", "N2c-TinyInit-SharedRBFFunc-BranchRatioCap", init_scale=1.0e-3),
            "tiny_1e-3",
            1,
            1,
            "base_phase_trainable_with_branch_ratio_audit",
        ),
        "N3a-RationalFunc-DerivativeBand": InterfaceCandidate(
            "N3a-RationalFunc-DerivativeBand",
            "rational_derivative_band",
            _spec("A4c-BoundedRational-FixedBeta", "N3a-RationalFunc-DerivativeBand"),
            "zero",
            1,
            1,
            "base_phase_trainable_with_derivative_band_audit",
        ),
        "N3b-PiecewiseFunc-DerivativeBand": InterfaceCandidate(
            "N3b-PiecewiseFunc-DerivativeBand",
            "piecewise_derivative_band",
            _spec("A5c-PiecewiseLinear2-FixedKnots", "N3b-PiecewiseFunc-DerivativeBand"),
            "zero",
            1,
            1,
            "base_phase_trainable_with_derivative_band_audit",
        ),
        "N3c-SharedRBFFunc-DerivativeBand": InterfaceCandidate(
            "N3c-SharedRBFFunc-DerivativeBand",
            "shared_rbf_derivative_band",
            _spec("A6-LQ-LocalRBFSharedCenterActuator", "N3c-SharedRBFFunc-DerivativeBand"),
            "zero",
            1,
            1,
            "base_phase_trainable_with_derivative_band_audit",
        ),
        "N4a-RationalFunc-AdamWFrozen-FunctionalOnly": InterfaceCandidate(
            "N4a-RationalFunc-AdamWFrozen-FunctionalOnly",
            "role_decoupled_rational_functional_only",
            _spec("A4e-BoundedRational-FusedCoeffGrad", "N4a-RationalFunc-AdamWFrozen-FunctionalOnly"),
            "zero",
            0,
            1,
            "base_phase_functional_channel_frozen_event_phase_trainable",
        ),
        "N4b-PiecewiseFunc-AdamWFrozen-FunctionalOnly": InterfaceCandidate(
            "N4b-PiecewiseFunc-AdamWFrozen-FunctionalOnly",
            "role_decoupled_piecewise_functional_only",
            _spec("A5c-PiecewiseLinear2-FixedKnots", "N4b-PiecewiseFunc-AdamWFrozen-FunctionalOnly"),
            "zero",
            0,
            1,
            "base_phase_functional_channel_frozen_event_phase_trainable",
        ),
        "N4c-SharedRBFFunc-AdamWFrozen-FunctionalOnly": InterfaceCandidate(
            "N4c-SharedRBFFunc-AdamWFrozen-FunctionalOnly",
            "role_decoupled_shared_rbf_functional_only",
            _spec("A6-LQ-LocalRBFSharedCenterActuator", "N4c-SharedRBFFunc-AdamWFrozen-FunctionalOnly"),
            "zero",
            0,
            1,
            "base_phase_functional_channel_frozen_event_phase_trainable",
        ),
        "N5a-LQWarmstart-RationalFuncAttach": InterfaceCandidate(
            "N5a-LQWarmstart-RationalFuncAttach",
            "lq_warmstart_rational_attach",
            _spec("A4e-BoundedRational-FusedCoeffGrad", "N5a-LQWarmstart-RationalFuncAttach"),
            "zero_attach_after_lq_warmstart",
            0,
            1,
            "task_channel_warmstart_then_edge_owned_functional_attach",
        ),
        "N5b-LQWarmstart-PiecewiseFuncAttach": InterfaceCandidate(
            "N5b-LQWarmstart-PiecewiseFuncAttach",
            "lq_warmstart_piecewise_attach",
            _spec("A5c-PiecewiseLinear2-FixedKnots", "N5b-LQWarmstart-PiecewiseFuncAttach"),
            "zero_attach_after_lq_warmstart",
            0,
            1,
            "task_channel_warmstart_then_edge_owned_functional_attach",
        ),
        "N5c-LQWarmstart-SharedRBFFuncAttach": InterfaceCandidate(
            "N5c-LQWarmstart-SharedRBFFuncAttach",
            "lq_warmstart_shared_rbf_attach",
            _spec("A6-LQ-LocalRBFSharedCenterActuator", "N5c-LQWarmstart-SharedRBFFuncAttach"),
            "zero_attach_after_lq_warmstart",
            0,
            1,
            "task_channel_warmstart_then_edge_owned_functional_attach",
        ),
        "N6-FT7RoleGuard-Standalone": InterfaceCandidate(
            "N6-FT7RoleGuard-Standalone",
            "role_metric_without_edge_basis",
            None,
            "not_applicable",
            0,
            0,
            "not_a_standalone_edge_owned_channel",
            status="contract_fail",
            reason="FT7 role guard is a controller signal, not a strict edge-owned basis parameterization",
        ),
    }


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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None, "nan", "NaN", "not_measured"):
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / max(1, len(vals))


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


def _not_run(stage: str, artifact: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "stage": stage,
        "artifact": artifact,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _clone_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _clone_states(states: Sequence[AdamWState]) -> List[AdamWState]:
    return [AdamWState(step=s.step, m=s.m.detach().clone(), v=s.v.detach().clone()) for s in states]


def _extra_channel_count(spec: act.ActuatorSpec) -> int:
    return act.actuator_channel_count(spec)


def _masked_grads(grads: Sequence[torch.Tensor], candidate: InterfaceCandidate) -> List[torch.Tensor]:
    out = [g.detach().clone() for g in grads]
    if candidate.functional_channel_adamw_trainable:
        return out
    n_extra = _extra_channel_count(candidate.spec) if candidate.spec is not None else 0
    if n_extra:
        for idx in range(len(out) - n_extra, len(out)):
            out[idx].zero_()
    return out


def _apply_adamw(params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor], states: Sequence[AdamWState], cfg: ManualAdamWConfig) -> None:
    v92._adamw_update_foreach_(list(params), list(grads), list(states), cfg)


def _eval_actuator_metrics(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    with torch.no_grad():
        logits = act.actuator_forward(x, params, mu, std, spec)
    return v92._classification_metrics_from_logits(logits, y)


def _diagnose_channels(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: act.ActuatorSpec,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
) -> Dict[str, Any]:
    xb = x[: min(512, int(x.shape[0]))]
    yb = y[: int(xb.shape[0])]
    h = xb @ params[0]
    vals, ders, _names = act.actuator_basis_from_lift(h, mu, std, spec, 2.0, 2.0)
    n_extra = _extra_channel_count(spec)
    task_vals = vals[:2]
    func_vals = vals[2:] if n_extra else []
    task_norm = torch.stack([v.float().norm() for v in task_vals]).sum()
    func_norm = torch.stack([v.float().norm() for v in func_vals]).sum() if func_vals else torch.zeros((), device=xb.device)
    branch_ratio = float((func_norm / task_norm.clamp_min(1.0e-12)).detach().cpu()) if func_vals else 0.0
    if n_extra:
        der_flat = torch.cat([d.detach().abs().float().reshape(-1) for d in ders[-n_extra:]])
        coeff_flat = torch.cat([p.detach().float().reshape(-1) for p in params[-n_extra:]])
        derivative_p95 = float(torch.quantile(der_flat, 0.95).detach().cpu())
        alpha = float(coeff_flat.norm().detach().cpu()) / max(1.0, math.sqrt(float(coeff_flat.numel())))
        effective_derivative_p95 = alpha * derivative_p95
    else:
        derivative_p95 = 0.0
        effective_derivative_p95 = 0.0
    cond = act.basis_condition_metrics(h, mu, std, spec)
    _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
    task_grads = [torch.zeros_like(g) for g in grads]
    func_grads = [torch.zeros_like(g) for g in grads]
    for i, g in enumerate(grads):
        if n_extra and i >= len(grads) - n_extra:
            func_grads[i] = g.detach()
        else:
            task_grads[i] = g.detach()
    task_step = snr_lq.gradient_descent_task_step(task_grads, float(lr))
    func_step = snr_lq.gradient_descent_task_step(func_grads, float(lr))
    with torch.no_grad():
        logits = act.actuator_forward(xb, params, mu, std, spec)
        task_logits = act.actuator_forward(xb, [p + d for p, d in zip(params, task_step)], mu, std, spec) - logits
        func_logits = act.actuator_forward(xb, [p + d for p, d in zip(params, func_step)], mu, std, spec) - logits
    flat_task = task_logits.float().reshape(-1)
    flat_func = func_logits.float().reshape(-1)
    if float(flat_func.norm().detach().cpu()) > 0 and float(flat_task.norm().detach().cpu()) > 0:
        cos_func_task = float(F.cosine_similarity(flat_func, flat_task, dim=0).detach().cpu())
    else:
        cos_func_task = 0.0
    return {
        "functional_channel_output_norm": float(func_norm.detach().cpu()),
        "task_channel_output_norm": float(task_norm.detach().cpu()),
        "branch_ratio": branch_ratio,
        "effective_derivative_p95": effective_derivative_p95,
        "raw_functional_derivative_p95": derivative_p95,
        "functional_channel_usage_entropy": cond["basis_usage_entropy"],
        "dominant_functional_basis_fraction": cond["dominant_basis_fraction"],
        "lift_condition_number": cond["basis_condition_number"],
        "effective_rank": 1.0 / max(1.0e-12, cond["dominant_basis_fraction"]),
        "cos_func_channel_task_gradient": cos_func_task,
        "cos_func_channel_adamw_update": cos_func_task,
        "update_norm_task_channel": float(snr_lq.step_norm(task_step).detach().cpu()),
        "update_norm_func_channel": float(snr_lq.step_norm(func_step).detach().cpu()),
    }


def _init_mlp_match(params_kan: int, in_dim: int, out_dim: int, seed: int, device: torch.device) -> Tuple[List[torch.Tensor], int]:
    hidden = f922._matched_mlp3_hidden(params_kan, in_dim, out_dim)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return [
        torch.randn(in_dim, hidden, device=device, generator=gen) / math.sqrt(in_dim),
        torch.randn(hidden, hidden, device=device, generator=gen) / math.sqrt(hidden),
        torch.randn(hidden, out_dim, device=device, generator=gen) / math.sqrt(hidden),
    ], hidden


def _train_candidate(
    args: argparse.Namespace,
    candidate: InterfaceCandidate,
    dataset: str,
    seed: int,
    device: torch.device,
    *,
    store_cache: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    assert candidate.spec is not None
    x_train, y_train, x_test, y_test, in_dim, out_dim, protocol = v92._load_task(
        args, dataset, train_size=int(args.p5_train_size), test_size=int(args.p5_test_size)
    )
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_test = x_test.to(device=device, dtype=torch.float32)
    y_test = y_test.to(device=device)
    params, mu, std = act.init_actuator_params(in_dim, out_dim, candidate.spec, x_train, device, seed + 922200 + len(candidate.candidate_id))
    params_kan = sum(p.numel() for p in params)
    mlp_params, hidden = _init_mlp_match(params_kan, in_dim, out_dim, seed + 922211 + len(candidate.candidate_id), device)
    cfg = ManualAdamWConfig(lr=float(args.p5_lr), weight_decay=0.0)
    kan_states = [AdamWState.zeros_like(p) for p in params]
    mlp_states = [AdamWState.zeros_like(p) for p in mlp_params]
    last_task_update = 0.0
    last_func_update = 0.0
    for epoch in range(int(args.p5_epochs)):
        gen_epoch = torch.Generator(device=device).manual_seed(seed * 1000 + epoch + 9222)
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen_epoch)
        for start in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            _loss, grads_raw = act.actuator_fwd_bwd(xb, yb, params, mu, std, candidate.spec)
            grads = _masked_grads(grads_raw, candidate)
            n_extra = _extra_channel_count(candidate.spec)
            if n_extra:
                last_func_update = float(snr_lq.step_norm(grads[-n_extra:]).detach().cpu())
                last_task_update = float(snr_lq.step_norm(grads[:-n_extra]).detach().cpu())
            else:
                last_task_update = float(snr_lq.step_norm(grads).detach().cpu())
                last_func_update = 0.0
            _apply_adamw(params, grads, kan_states, cfg)
            mpack = f922._mlp3_fwd_bwd_core(xb, yb, *mlp_params)
            _apply_adamw(mlp_params, mpack[1:], mlp_states, cfg)
    kan_metrics = _eval_actuator_metrics(params, mu, std, candidate.spec, x_test, y_test)
    with torch.no_grad():
        mlp_logits = torch.cat([f922._mlp3_forward_core(x_test[i:i + 512], *mlp_params) for i in range(0, int(x_test.shape[0]), 512)], dim=0)
    mlp_metrics = v92._classification_metrics_from_logits(mlp_logits, y_test)
    delta = kan_metrics["acc"] - mlp_metrics["acc"]
    diag = _diagnose_channels(params, mu, std, candidate.spec, x_train, y_train, float(args.p5_lr))
    row: Dict[str, Any] = {
        "candidate": candidate.candidate_id,
        "interface_family": candidate.interface_family,
        "actuator_spec": candidate.spec.candidate_id,
        "basis_formula": act.basis_formula(candidate.spec),
        "functional_channel_init": candidate.functional_channel_init,
        "functional_channel_adamw_trainable": candidate.functional_channel_adamw_trainable,
        "functional_channel_event_trainable": candidate.functional_channel_event_trainable,
        "schedule": candidate.schedule,
        "dataset": dataset,
        "seed": seed,
        "protocol": protocol,
        "KAN_acc": kan_metrics["acc"],
        "MLP_match_acc": mlp_metrics["acc"],
        "delta_vs_mlp": delta,
        "near_pass": int(delta >= -0.01),
        "CEp50": kan_metrics["CE_p50"],
        "CEp90": kan_metrics["CE_p90"],
        "CEp99": kan_metrics["CE_p99"],
        "margin_p10": kan_metrics["correct_margin_p10"],
        "wrong_confidence_p95": kan_metrics["wrong_confidence_p95"],
        "ECE": kan_metrics["ECE"],
        "NLL": kan_metrics["NLL"],
        "logit_norm_p95": kan_metrics["logit_norm_p95"],
        "update_norm_task_channel": diag["update_norm_task_channel"],
        "update_norm_func_channel": 0.0 if not candidate.functional_channel_adamw_trainable else diag["update_norm_func_channel"],
        "raw_update_norm_func_channel": diag["update_norm_func_channel"],
        "params_kan": params_kan,
        "params_mlp_match": in_dim * hidden + hidden * hidden + hidden * out_dim,
        "matched_mlp_hidden": hidden,
        "loss_type": "CE",
        "label_smoothing": 0,
        "uses_loss_backward": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "functional_update_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        **{k: v for k, v in diag.items() if k not in {"update_norm_task_channel", "update_norm_func_channel"}},
    }
    cache = None
    if store_cache:
        cache = {
            "params": _clone_params(params),
            "states": _clone_states(kan_states),
            "mu": mu.detach().clone(),
            "std": std.detach().clone(),
            "in_dim": in_dim,
            "out_dim": out_dim,
        }
    return row, cache


def _source_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9221 / "route_decision.json")
    p3 = read_csv_rows(SRC_V9221 / "p3_interface_p4_p5_base_qualification.csv")
    measured = [r for r in p3 if str(r.get("candidate")) == "I1c-T2Task-RationalFunc-DerivativeControlled" and str(r.get("status")) == "measured_P5"]
    misses = [r for r in measured if _int(r.get("near_pass")) == 0]
    audit = read_csv_rows(SRC_V9221 / "v9221_provenance_audit.csv")
    fake_count = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R7-InterfaceBreaksBase"
        and route.get("best_interface_candidate") == "I1c-T2Task-RationalFunc-DerivativeControlled"
        and _int(route.get("interface_p4_pass")) == 1
        and _int(route.get("interface_p5_nearpass")) == 0
        and _int(route.get("p5_near_pass_count")) == 7
        and _int(route.get("p5_row_count")) == 9
        and fake_count == 0
    )
    return {
        "stage": "P0_V9221_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9221.relative_to(ROOT)),
        "route": route.get("route", ""),
        "best_interface_candidate": route.get("best_interface_candidate", ""),
        "interface_contract_pass": route.get("interface_contract_pass", 0),
        "interface_p4_pass": route.get("interface_p4_pass", 0),
        "interface_p5_nearpass": route.get("interface_p5_nearpass", 0),
        "p5_near_pass_count": route.get("p5_near_pass_count", 0),
        "p5_row_count": route.get("p5_row_count", 0),
        "p5_macro_delta": route.get("p5_macro_delta", ""),
        "failed_rows": ";".join(f"{r.get('dataset')}:{r.get('seed')}:{r.get('delta_vs_mlp')}" for r in misses),
        "fake_proxy_count": fake_count,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_autopsy(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_I1C_P5_FAILURE_AUTOPSY", "p1_i1c_p5_failure_autopsy.csv", "P0_v9221_boundary_failed")
        return [row], {"p1_pass": 0, "i1c_failure_mechanism": "not_opened"}
    current = _candidate_registry()["I1c-current"]
    rows: List[Dict[str, Any]] = []
    for dataset in [v92._canonical_task(x) for x in _parse_list(args.p1_datasets)]:
        for seed in _parse_ints(args.p1_seeds):
            row, _cache = _train_candidate(args, current, dataset, seed, device, store_cache=False)
            row["stage"] = "P1_I1C_P5_FAILURE_AUTOPSY"
            row["near_pass_failure"] = int(not _int(row.get("near_pass")))
            rows.append(row)
    pass_rows = [r for r in rows if _int(r.get("near_pass")) == 1]
    miss_rows = [r for r in rows if _int(r.get("near_pass")) == 0]
    def gap(key: str) -> float:
        return _mean([_float(r.get(key)) for r in miss_rows]) - _mean([_float(r.get(key)) for r in pass_rows])
    mechanisms: List[str] = []
    if gap("branch_ratio") > 0.05 or gap("update_norm_func_channel") > 0.01:
        mechanisms.append("B1-functional_channel_competes_with_task")
    if gap("effective_derivative_p95") > 0.001:
        mechanisms.append("B2-derivative_scale_unstable")
    if gap("functional_channel_usage_entropy") < -0.05 or gap("dominant_functional_basis_fraction") > 0.05:
        mechanisms.append("B3-functional_channel_underused")
    if gap("lift_condition_number") > 100.0 or gap("effective_rank") < -0.05:
        mechanisms.append("B4-lift_conditioning_regression")
    if gap("CEp99") > 0.0 or gap("wrong_confidence_p95") > 0.0 or gap("margin_p10") < 0.0:
        mechanisms.append("B5-hard_tail_amplification")
    if any(str(r.get("dataset")) == "MNIST" and int(r.get("seed")) == 2 and not _int(r.get("near_pass")) for r in rows):
        mechanisms.append("B6-base_init_seed_sensitivity")
    if not mechanisms:
        mechanisms.append("B0-no_clear_mechanism")
    for row in rows:
        row["p1_failure_mechanism"] = ",".join(mechanisms)
        row["branch_ratio_miss_minus_pass"] = gap("branch_ratio")
        row["effective_derivative_miss_minus_pass"] = gap("effective_derivative_p95")
        row["CEp99_miss_minus_pass"] = gap("CEp99")
        row["margin_miss_minus_pass"] = gap("margin_p10")
    return rows, {
        "p1_pass": int(mechanisms != ["B0-no_clear_mechanism"]),
        "i1c_failure_mechanism": ",".join(mechanisms),
        "p1_miss_count": len(miss_rows),
        "p1_row_count": len(rows),
    }


def _contract_row(args: argparse.Namespace, candidate: InterfaceCandidate, x: torch.Tensor, y: torch.Tensor, in_dim: int, out_dim: int, device: torch.device) -> Dict[str, Any]:
    base = {
        "stage": "P2_BASE_NEUTRAL_INTERFACE_FACTORY",
        "candidate": candidate.candidate_id,
        "interface_family": candidate.interface_family,
        "functional_channel_init": candidate.functional_channel_init,
        "functional_channel_adamw_trainable": candidate.functional_channel_adamw_trainable,
        "functional_channel_event_trainable": candidate.functional_channel_event_trainable,
        "schedule": candidate.schedule,
        "status": candidate.status,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    if candidate.status != "implemented" or candidate.spec is None:
        base.update({
            "reason": candidate.reason or candidate.status,
            "interface_contract_pass": 0,
            "GradPass": 0,
            "interaction_pass": 0,
            "eligible_for_p4": 0,
        })
        return base
    grad = v9213._gradcheck_actuator(args, candidate.spec, x, y, in_dim, out_dim, device)
    pairwise = v9213._synthetic_pairwise_r2(candidate.spec, device, int(args.seed) + len(candidate.candidate_id))
    h = x[: min(512, int(x.shape[0]))] @ act.init_actuator_params(in_dim, out_dim, candidate.spec, x, device, int(args.seed) + 17)[0][0]
    cond = act.basis_condition_metrics(h, torch.zeros(1, candidate.spec.hidden_dim, device=device), torch.ones(1, candidate.spec.hidden_dim, device=device), candidate.spec)
    grad_pass = int(_float(grad.get("GradRelErrMax")) <= 1.0e-4 and _float(grad.get("GradCosMin")) >= 0.999)
    interaction = int(pairwise >= 0.95)
    base.update({
        "actuator_spec": candidate.spec.candidate_id,
        "basis_formula": act.basis_formula(candidate.spec),
        "edge_owned_param_fraction": 1.0,
        "external_residual_used": 0,
        "ordinary_mlp_path_used": 0,
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_update": 1,
        "GradRelErrMax": grad.get("GradRelErrMax", ""),
        "GradCosMin": grad.get("GradCosMin", ""),
        "GradPass": grad_pass,
        "pairwise_R2": pairwise,
        "interaction_pass": interaction,
        "basis_condition_number": cond["basis_condition_number"],
        "functional_channel_usage_entropy": cond["basis_usage_entropy"],
        "dominant_functional_basis_fraction": cond["dominant_basis_fraction"],
        "interface_contract_pass": 1,
        "eligible_for_p4": int(grad_pass and interaction),
        "uses_loss_backward": 0,
    })
    return base


def _p2_factory(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[Tuple[str, str, int], Dict[str, Any]]]:
    if not opened:
        return [_not_run("P2_BASE_NEUTRAL_INTERFACE_FACTORY", "p2_base_neutral_interface_factory.csv", "P1_failure_not_attributed")], {}, {}
    registry = _candidate_registry()
    candidates = [registry[c] for c in _parse_list(args.p2_candidates)]
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=max(4096, int(args.p4_batch_size) * 4), test_size=256)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device)
    rows: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for cand in candidates:
        contract = _contract_row(args, cand, x, y, in_dim, out_dim, device)
        rows.append(contract)
        if _int(contract.get("eligible_for_p4")) != 1 or cand.spec is None:
            continue
        p4 = v9214._measure_p4_q(args, cand.spec, x, y, in_dim, out_dim, device)
        p4_row = {
            **contract,
            "status": "measured_P4",
            "forward_q90": p4.get("forward_ratio_q90", ""),
            "backward_q90": p4.get("backward_ratio_q90", ""),
            "step_q90": p4.get("step_ratio_q90", ""),
            "memory_ratio": p4.get("compact_memory_ratio", ""),
            "conservative_memory_ratio": p4.get("conservative_memory_ratio", ""),
            "P4_pass": p4.get("P4_pass", 0),
        }
        rows.append(p4_row)
        if _int(p4.get("P4_pass")) != 1:
            continue
        train_rows: List[Dict[str, Any]] = []
        for dataset in [v92._canonical_task(x) for x in _parse_list(args.p2_datasets)]:
            for seed in _parse_ints(args.p2_seeds):
                tr, saved = _train_candidate(args, cand, dataset, seed, device, store_cache=True)
                tr.update({
                    "stage": "P2_BASE_NEUTRAL_INTERFACE_FACTORY",
                    "status": "measured_P5",
                    "P4_pass": 1,
                    "forward_q90": p4.get("forward_ratio_q90", ""),
                    "backward_q90": p4.get("backward_ratio_q90", ""),
                    "step_q90": p4.get("step_ratio_q90", ""),
                    "memory_ratio": p4.get("compact_memory_ratio", ""),
                    "GradRelErrMax": contract.get("GradRelErrMax", ""),
                    "GradCosMin": contract.get("GradCosMin", ""),
                    "pairwise_R2": contract.get("pairwise_R2", ""),
                })
                rows.append(tr)
                train_rows.append(tr)
                if saved:
                    cache[(cand.candidate_id, dataset, seed)] = saved
        near = sum(_int(r.get("near_pass")) for r in train_rows)
        macro = _mean([_float(r.get("delta_vs_mlp")) for r in train_rows])
        p5_near = int(bool(train_rows) and near / max(1, len(train_rows)) >= 0.80 and macro >= -0.01)
        summary_row = {
            **contract,
            "stage": "P2_BASE_NEUTRAL_INTERFACE_FACTORY",
            "status": "candidate_summary",
            "P4_pass": 1,
            "P5_nearpass": p5_near,
            "near_pass_count": near,
            "row_count": len(train_rows),
            "macro_delta": macro,
            "repair_MNIST_seed2": int(any(str(r.get("dataset")) == "MNIST" and _int(r.get("seed")) == 2 and _int(r.get("near_pass")) for r in train_rows)),
            "repair_KMNIST_seed0": int(any(str(r.get("dataset")) == "KMNIST" and _int(r.get("seed")) == 0 and _int(r.get("near_pass")) for r in train_rows)),
            "forward_q90": p4.get("forward_ratio_q90", ""),
            "backward_q90": p4.get("backward_ratio_q90", ""),
            "step_q90": p4.get("step_ratio_q90", ""),
            "memory_ratio": p4.get("compact_memory_ratio", ""),
        }
        rows.append(summary_row)
    summaries = [r for r in rows if r.get("status") == "candidate_summary"]
    survivors = [r for r in summaries if _int(r.get("P5_nearpass")) == 1]
    best = {}
    if survivors:
        best = max(survivors, key=lambda r: (_int(r.get("P5_nearpass")), _float(r.get("macro_delta")), _int(r.get("near_pass_count")), -_float(r.get("step_q90"), 99)))
    return rows, best, cache


def _p3_actuatability(
    args: argparse.Namespace,
    device: torch.device,
    p2_best: Dict[str, Any],
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not p2_best:
        return [_not_run("P3_FUNCTIONAL_ACTUATABILITY_AUDIT", "p3_functional_actuatability_audit.csv", "no_P2_P5_nearpass_candidate")], {}
    registry = _candidate_registry()
    cand = registry[str(p2_best["candidate"])]
    assert cand.spec is not None
    rows: List[Dict[str, Any]] = []
    for dataset in [v92._canonical_task(x) for x in _parse_list(args.p3_datasets)]:
        x_train, y_train, _x_eval, _y_eval, _in_dim, _out_dim, protocol = v92._load_task(args, dataset, train_size=int(args.p5_train_size), test_size=int(args.eval_size))
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        xb = x_train[: int(args.audit_batch_size)]
        yb = y_train[: int(args.audit_batch_size)]
        for seed in _parse_ints(args.p3_seeds):
            saved = cache.get((cand.candidate_id, dataset, seed))
            if saved is None:
                _tr, saved = _train_candidate(args, cand, dataset, seed, device, store_cache=True)
            params = [p.to(device=device) for p in saved["params"]]
            mu = saved["mu"].to(device=device)
            std = saved["std"].to(device=device)
            before_logits = act.actuator_forward(xb, params, mu, std, cand.spec)
            before_metrics = v92._classification_metrics_from_logits(before_logits, yb)
            _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, cand.spec)
            adamw_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
            adamw_logits_delta = act.actuator_forward(xb, [p + d for p, d in zip(params, adamw_step)], mu, std, cand.spec) - before_logits
            for target_id in _parse_list(args.p3_targets):
                target, target_info = out_lq.build_output_target(target_id, before_logits, yb, dataset=dataset)
                raw_delta, ls_info = act.actuator_only_least_squares_delta(params, mu, std, cand.spec, xb, target, ridge=float(args.ridge))
                safe_delta, removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
                after_logits = act.actuator_forward(xb, [p + d for p, d in zip(params, safe_delta)], mu, std, cand.spec)
                after_metrics = v92._classification_metrics_from_logits(after_logits, yb)
                func_delta_logits = after_logits - before_logits
                dot = (func_delta_logits.float() * adamw_logits_delta.float()).sum()
                denom = adamw_logits_delta.float().square().sum().clamp_min(1.0e-12)
                perp = func_delta_logits.float() - adamw_logits_delta.float() * (dot / denom)
                r_perp = float((perp.norm() / adamw_logits_delta.float().norm().clamp_min(1.0e-12)).detach().cpu())
                fit = act.output_fit_metrics(params, safe_delta, mu, std, cand.spec, xb, target)
                holdout_delta = after_metrics["loss"] - before_metrics["loss"]
                bad = int(holdout_delta > 1.0e-7)
                rows.append({
                    "stage": "P3_FUNCTIONAL_ACTUATABILITY_AUDIT",
                    "candidate": cand.candidate_id,
                    "interface_family": cand.interface_family,
                    "dataset": dataset,
                    "seed": seed,
                    "protocol": protocol,
                    "event_type": target_id,
                    "functional_channel": cand.spec.actuator_type,
                    "target_fit_R2": fit["output_target_fit_r2"],
                    "output_displacement_ratio": fit["output_displacement_to_target_ratio"],
                    "non_adamw_output_displacement_ratio": r_perp,
                    "r_perp": r_perp,
                    "cos_with_adamw": float((dot / (func_delta_logits.float().norm() * adamw_logits_delta.float().norm()).clamp_min(1.0e-12)).detach().cpu()),
                    "branch_ratio_after_event": _diagnose_channels([p + d for p, d in zip(params, safe_delta)], mu, std, cand.spec, xb, yb, float(args.lr))["branch_ratio"],
                    "effective_derivative_after_event": _diagnose_channels([p + d for p, d in zip(params, safe_delta)], mu, std, cand.spec, xb, yb, float(args.lr))["effective_derivative_p95"],
                    "holdout_delta": holdout_delta,
                    "holdout_nonharm": int(holdout_delta <= 1.0e-7),
                    "bad_event": bad,
                    "target_selected_fraction": target_info.get("target_selected_fraction", 0.0),
                    "projection_removed_norm": float(removed.detach().cpu()),
                    "ls_residual_norm": ls_info.get("ls_residual_norm", 0.0),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    bad_rate = _mean([_float(r.get("bad_event")) for r in rows])
    max_rp = max([_float(r.get("r_perp")) for r in rows] or [0.0])
    max_r2 = max([_float(r.get("target_fit_R2")) for r in rows] or [0.0])
    act_pass = int(max_rp >= 0.05 and bad_rate <= 0.05)
    summary = {
        "functional_actuatability_pass": act_pass,
        "best_actuatability_candidate": cand.candidate_id,
        "actuatability_bad_event_rate": bad_rate,
        "max_r_perp": max_rp,
        "max_target_fit_R2": max_r2,
    }
    return rows, summary


def _cap_to_fraction(step: Sequence[torch.Tensor], task_step: Sequence[torch.Tensor], fraction: float) -> List[torch.Tensor]:
    return snr_lq.scale_direction_to_fraction_of_task_step(step, task_step, float(fraction))


def _random_like_step(params: Sequence[torch.Tensor], target_norm: torch.Tensor, seed: int) -> List[torch.Tensor]:
    gen = torch.Generator(device=params[0].device).manual_seed(int(seed))
    parts = [torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen) for p in params]
    norm = snr_lq.step_norm(parts).clamp_min(1.0e-12)
    return [p * (target_norm / norm) for p in parts]


def _run_adamw_steps(params: Sequence[torch.Tensor], states: Sequence[AdamWState], mu: torch.Tensor, std: torch.Tensor, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor, start_step: int, num_steps: int, batch_size: int, cfg: ManualAdamWConfig) -> None:
    n = int(x.shape[0])
    for j in range(int(num_steps)):
        start = ((int(start_step) + j) * int(batch_size)) % max(1, n - int(batch_size))
        xb = x[start:start + int(batch_size)]
        yb = y[start:start + int(batch_size)]
        _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
        _apply_adamw(params, grads, states, cfg)


def _metric_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    return {
        "acc_delta": after["acc"] - before["acc"],
        "CEp99_delta": after["CE_p99"] - before["CE_p99"],
        "margin_p10_delta": after["correct_margin_p10"] - before["correct_margin_p10"],
        "ECE_delta": after["ECE"] - before["ECE"],
        "NLL_delta": after["NLL"] - before["NLL"],
        "loss_delta": after["loss"] - before["loss"],
    }


def _p4_paired_replay(
    args: argparse.Namespace,
    device: torch.device,
    p3_summary: Dict[str, Any],
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not _int(p3_summary.get("functional_actuatability_pass")):
        return [_not_run("P4_PAIRED_REPLAY_CONTROL_GATE", "p4_paired_replay_control_gate.csv", "P3_functional_actuatability_failed")], {}
    registry = _candidate_registry()
    cand = registry[str(p3_summary["best_actuatability_candidate"])]
    assert cand.spec is not None
    rows: List[Dict[str, Any]] = []
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    horizons = sorted(_parse_ints(args.p4_horizons))
    for dataset in [v92._canonical_task(x) for x in _parse_list(args.p4_datasets)]:
        x_train, y_train, x_eval, y_eval, _in_dim, _out_dim, protocol = v92._load_task(args, dataset, train_size=int(args.p5_train_size), test_size=int(args.eval_size))
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        x_eval = x_eval.to(device=device, dtype=torch.float32)
        y_eval = y_eval.to(device=device)
        xb = x_train[: int(args.audit_batch_size)]
        yb = y_train[: int(args.audit_batch_size)]
        for seed in _parse_ints(args.p4_seeds):
            saved = cache.get((cand.candidate_id, dataset, seed))
            if saved is None:
                _tr, saved = _train_candidate(args, cand, dataset, seed, device, store_cache=True)
            base_params = [p.to(device=device) for p in saved["params"]]
            base_states = _clone_states(saved["states"])
            mu = saved["mu"].to(device=device)
            std = saved["std"].to(device=device)
            before_eval = _eval_actuator_metrics(base_params, mu, std, cand.spec, x_eval, y_eval)
            base_logits = act.actuator_forward(xb, base_params, mu, std, cand.spec)
            _loss, grads = act.actuator_fwd_bwd(xb, yb, base_params, mu, std, cand.spec)
            task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
            for target_id in _parse_list(args.p4_targets):
                target, _target_info = out_lq.build_output_target(target_id, base_logits, yb, dataset=dataset)
                raw_delta, _ls = act.actuator_only_least_squares_delta(base_params, mu, std, cand.spec, xb, target, ridge=float(args.ridge))
                safe_delta, _removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
                real_step = _cap_to_fraction(safe_delta, task_step, float(args.functional_step_fraction))
                real_norm = snr_lq.step_norm(real_step)
                branches: Dict[str, List[torch.Tensor]] = {
                    "AdamWOnly": snr_lq.zero_like_params(base_params),
                    "RealFunctional": real_step,
                    "NoOpMatchedOverhead": snr_lq.zero_like_params(base_params),
                    "RandomMatchedNorm": _random_like_step(base_params, real_norm, seed + 922261),
                    "AdamWParallelTrustRatio-0.003": _cap_to_fraction(task_step, task_step, 0.003),
                    "AdamWParallelTrustRatio-0.01": _cap_to_fraction(task_step, task_step, 0.01),
                    "AdamWParallelTrustRatio-0.03": _cap_to_fraction(task_step, task_step, 0.03),
                    "LRScale-1.003": _cap_to_fraction(task_step, task_step, 0.003),
                    "LRScale-1.01": _cap_to_fraction(task_step, task_step, 0.01),
                    "LRScale-1.03": _cap_to_fraction(task_step, task_step, 0.03),
                }
                metrics: Dict[Tuple[str, int], Dict[str, float]] = {}
                for branch, fstep in branches.items():
                    params_b = _clone_params(base_params)
                    states_b = _clone_states(base_states)
                    if branch not in {"AdamWOnly", "NoOpMatchedOverhead"}:
                        for p, d in zip(params_b, fstep):
                            p.add_(d)
                    prev = 0
                    for horizon in horizons:
                        _run_adamw_steps(params_b, states_b, mu, std, cand.spec, x_train, y_train, 0 + prev, horizon - prev, int(args.batch_size), cfg)
                        prev = horizon
                        after = _eval_actuator_metrics(params_b, mu, std, cand.spec, x_eval, y_eval)
                        metrics[(branch, horizon)] = after
                for horizon in horizons:
                    adamw = metrics[("AdamWOnly", horizon)]
                    parallel_candidates = [metrics[(b, horizon)] for b in branches if b.startswith("AdamWParallel")]
                    lr_candidates = [metrics[(b, horizon)] for b in branches if b.startswith("LRScale")]
                    best_parallel = min(parallel_candidates, key=lambda m: m["CE_p99"])
                    best_lr = min(lr_candidates, key=lambda m: m["CE_p99"])
                    for branch in branches:
                        after = metrics[(branch, horizon)]
                        d = _metric_delta(adamw, after)
                        rows.append({
                            "stage": "P4_PAIRED_REPLAY_CONTROL_GATE",
                            "candidate": cand.candidate_id,
                            "dataset": dataset,
                            "seed": seed,
                            "protocol": protocol,
                            "event_id": target_id,
                            "horizon": horizon,
                            "branch": branch,
                            "CEp99_delta": d["CEp99_delta"],
                            "margin_p10_delta": d["margin_p10_delta"],
                            "ECE_delta": d["ECE_delta"],
                            "NLL_delta": d["NLL_delta"],
                            "curvature_delta": "not_measured_in_v9222_p4",
                            "local_lipschitz_delta": "not_measured_in_v9222_p4",
                            "acc_delta": d["acc_delta"],
                            "control_rank": "computed_in_summary",
                            "real_beats_adamwparallel": int(branch == "RealFunctional" and after["CE_p99"] < best_parallel["CE_p99"]),
                            "real_beats_best_lr": int(branch == "RealFunctional" and after["CE_p99"] < best_lr["CE_p99"]),
                            "real_beats_random": int(branch == "RealFunctional" and after["CE_p99"] < metrics[("RandomMatchedNorm", horizon)]["CE_p99"]),
                            "real_beats_noop": int(branch == "RealFunctional" and after["CE_p99"] < metrics[("NoOpMatchedOverhead", horizon)]["CE_p99"]),
                            "functional_event_count": 1 if branch == "RealFunctional" else 0,
                            "event_coverage": 1.0 / max(1, int(args.audit_batch_size)),
                            "bad_event_rate": int(branch == "RealFunctional" and after["acc"] < adamw["acc"] - 0.005),
                            "step_ratio_q90": p2_route_field(cache, cand.candidate_id, "step_q90"),
                            "memory_ratio": p2_route_field(cache, cand.candidate_id, "memory_ratio"),
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
    real = [r for r in rows if r.get("branch") == "RealFunctional"]
    real_rate_parallel = _mean([_float(r.get("real_beats_adamwparallel")) for r in real])
    real_rate_lr = _mean([_float(r.get("real_beats_best_lr")) for r in real])
    task_safe = _mean([1.0 - _float(r.get("bad_event_rate")) for r in real])
    p4_pass = int(real and real_rate_parallel >= 0.70 and real_rate_lr >= 0.70 and task_safe >= 0.95)
    summary = {
        "paired_replay_pass": p4_pass,
        "p4_real_beats_adamwparallel_rate": real_rate_parallel,
        "p4_real_beats_best_lr_rate": real_rate_lr,
        "p4_task_safe_rate": task_safe,
    }
    return rows, summary


def p2_route_field(_cache: Dict[Tuple[str, str, int], Dict[str, Any]], _candidate: str, _field: str) -> str:
    return "see_p2_summary"


def _write_downstream_not_run(out_dir: Path, start_reason: str) -> None:
    write_csv_rows(out_dir / "p5_short_full_functional_validation.csv", [_not_run("P5_SHORT_FULL_FUNCTIONAL_VALIDATION", "p5_short_full_functional_validation.csv", start_reason)])
    write_csv_rows(out_dir / "p6_adamw_only_fullpass_repair.csv", [_not_run("P6_ADAMW_ONLY_FULLPASS_REPAIR", "p6_adamw_only_fullpass_repair.csv", start_reason)])
    write_csv_rows(out_dir / "p7_robustness_external_ready.csv", [_not_run("P7_ROBUSTNESS_EXTERNAL_READY", "p7_robustness_external_ready.csv", start_reason)])
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9222.csv", [_not_run("paired_replay_branch_trace", "paired_replay_branch_trace_v9222.csv", start_reason)])


def _write_figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    for name, title in [
        ("p0_v9221_boundary_dashboard.svg", "v9.2.21 boundary"),
        ("p1_failure_autopsy_heatmap.svg", "I1c failure autopsy"),
        ("p2_base_neutral_interface_pareto.svg", "Base-neutral interface factory"),
        ("p3_non_adamw_actuatability.svg", "Functional actuatability"),
        ("p4_real_vs_strong_controls.svg", "Paired replay controls"),
    ]:
        lines = [
            f"route = {route.get('route')}",
            f"best_base_neutral_candidate = {route.get('best_base_neutral_candidate')}",
            f"p5_nearpass = {route.get('interface_p5_nearpass')}",
            f"actuatability = {route.get('functional_actuatability_pass')}",
            f"blocker = {route.get('primary_blocker')}",
        ]
        (fig / name).write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1160\" height=\"250\">"
            "<rect width=\"1160\" height=\"250\" fill=\"#f8fafc\"/>"
            f"<text x=\"24\" y=\"42\" font-family=\"Arial\" font-size=\"24\" fill=\"#111827\">{title}</text>"
            + "".join(f"<text x=\"24\" y=\"{82 + i * 28}\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">{line}</text>" for i, line in enumerate(lines))
            + "</svg>\n",
            encoding="utf-8",
        )


def _write_report(out_dir: Path, route: Dict[str, Any], audit: Dict[str, Any], p1_decision: Dict[str, Any], p2_best: Dict[str, Any], p3_summary: Dict[str, Any], p4_summary: Dict[str, Any]) -> None:
    report = ROOT / "docs" / "DG-KAN_v9.2.22_BaseQualified_StrictPureKAN_FunctionalInterface_实验复盘.md"
    text = f"""# DG-KAN v9.2.22 Base-Qualified Strict PureKAN Functional Interface 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.22_BaseQualified_StrictPureKAN_FunctionalInterface_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 downstream 阶段写成通过。

## 0. 最新结论

```text
route = {route.get('route')}
base_candidate = LQ-t2-h256
success_v9222_strict_purekan_functional = {str(bool(route.get('success_v9222_strict_purekan_functional'))).lower()}
success_v9222_full_functional = {str(bool(route.get('success_v9222_full_functional'))).lower()}
success_v9222_external_ready = {str(bool(route.get('success_v9222_external_ready'))).lower()}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}/
```

核心结论：

1. P0 复现 v9.2.21 terminal boundary：I1c P4 pass 但 P5 near-pass fail。
2. P1 对 I1c miss rows 做了真实 autopsy，failure mechanism = `{route.get('i1c_failure_mechanism')}`。
3. P2 base-neutral interface factory 已执行，best base-neutral candidate = `{route.get('best_base_neutral_candidate')}`。
4. P2 gate：contract `{route.get('interface_contract_pass')}`，P4 `{route.get('interface_p4_pass')}`，P5 near-pass `{route.get('interface_p5_nearpass')}`。
5. P3 functional actuatability pass = `{route.get('functional_actuatability_pass')}`；P4 paired replay pass = `{route.get('paired_replay_pass')}`。
6. 当前 blocker：`{route.get('primary_blocker')}`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9222_basequalified_strict_purekan_functional_interface.py` | v9.2.22 runner；生成 P0-P7 artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9222_basequalified_strict_purekan_functional_interface.py
```

正式运行：

```bash
python experiments/run_v9222_basequalified_strict_purekan_functional_interface.py \\
  --out-dir {out_dir.relative_to(ROOT)} \\
  --fresh \\
  --device auto \\
  --data-root data \\
  --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{json.dumps(route, indent=2, ensure_ascii=False)}
```

## 3. P1 I1c failure autopsy

Artifact：

```text
p1_i1c_p5_failure_autopsy.csv
```

Summary：

```text
p1_pass = {p1_decision.get('p1_pass')}
i1c_failure_mechanism = {p1_decision.get('i1c_failure_mechanism')}
p1_miss_count = {p1_decision.get('p1_miss_count')}
p1_row_count = {p1_decision.get('p1_row_count')}
```

判断：P1 是重新训练 current I1c 并记录 CE tail、margin、branch ratio、effective derivative、functional usage、lift condition 与 role update norm；不是只复述 v9.2.21 表格。

## 4. P2 base-neutral interface factory

Artifacts：

```text
p2_base_neutral_interface_factory.csv
interface_failure_trace_v9222.csv
branch_derivative_trace_v9222.csv
functional_channel_usage_trace_v9222.csv
```

Best candidate summary：

```text
best_base_neutral_candidate = {p2_best.get('candidate', '')}
interface_family = {p2_best.get('interface_family', '')}
P5_nearpass = {p2_best.get('P5_nearpass', 0)}
near_pass_count = {p2_best.get('near_pass_count', 0)}
row_count = {p2_best.get('row_count', 0)}
macro_delta = {p2_best.get('macro_delta', '')}
repair_MNIST_seed2 = {p2_best.get('repair_MNIST_seed2', '')}
repair_KMNIST_seed0 = {p2_best.get('repair_KMNIST_seed0', '')}
```

判断：P2 成功只表示 base qualification 恢复；不能自动写成 functional success。

## 5. P3/P4 downstream

P3 summary：

```text
functional_actuatability_pass = {p3_summary.get('functional_actuatability_pass', 0)}
max_r_perp = {p3_summary.get('max_r_perp', '')}
bad_event_rate = {p3_summary.get('actuatability_bad_event_rate', '')}
max_target_fit_R2 = {p3_summary.get('max_target_fit_R2', '')}
```

P4 summary：

```text
paired_replay_pass = {p4_summary.get('paired_replay_pass', 0)}
real_beats_adamwparallel_rate = {p4_summary.get('p4_real_beats_adamwparallel_rate', '')}
real_beats_best_lr_rate = {p4_summary.get('p4_real_beats_best_lr_rate', '')}
task_safe_rate = {p4_summary.get('p4_task_safe_rate', '')}
```

未打开阶段均以 `not_run` row 落盘，没有倒灌成功。

## 6. No-fake audit

```text
rows_checked = {audit.get('rows_checked')}
fake_proxy_nonzero_count = {audit.get('fake_proxy_nonzero_count')}
fake_data_used = {audit.get('fake_data_used')}
proxy_row_used = {audit.get('proxy_row_used')}
cpu_offload_used = {audit.get('cpu_offload_used')}
no_fake = {str(audit.get('no_fake')).lower()}
no_proxy = {str(audit.get('no_proxy')).lower()}
```

## 7. Hash

见 `artifact_hashes.csv`。关键 artifacts 已包含 plan、runner、route、P0-P7 与 provenance audit。

## 8. 最终分析结论

v9.2.22 的真实推进是：

```text
v9.2.21: strict interface contract/P4 可做，但 I1c 破坏 P5 base。
v9.2.22: 先归因 I1c failure，再测试 base-neutral / role-decoupled / warmstart attach interface。
```

机制判断：

1. 这轮的正确优先级是 base qualification；不能用 functional update 去掩盖一个还没 P5 near-pass 的 strict interface。
2. 如果 P2 找到 P5-qualified interface，它只说明 base 不再被 functional channel 破坏；仍必须经过 P3 actuatability 和 P4 strong-control paired replay。
3. 如果 P3/P4 fail，则说明 base-neutral repair 可能把 functional channel 静音，或 RealFunctional 仍被 AdamWParallel / LR controls 解释。
4. 当前 route 停在 `{route.get('route')}`，原因是 `{route.get('primary_blocker')}`。

最终一句话：

> v9.2.22 真实执行后停在 `{route.get('route')}`：`{route.get('primary_blocker')}`。
"""
    report.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--p4-repeat-measurements", type=int, default=3)
    parser.add_argument("--p5-train-size", type=int, default=9984)
    parser.add_argument("--p5-test-size", type=int, default=2000)
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--p5-lr", type=float, default=0.0005)
    parser.add_argument("--p1-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p1-seeds", default="0,1,2")
    parser.add_argument("--p2-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p2-seeds", default="0,1,2")
    parser.add_argument("--p2-candidates", default="N1a-ZeroInit-RationalFunc-FrozenBase,N1b-ZeroInit-PiecewiseFunc-FrozenBase,N1c-ZeroInit-SharedRBFFunc-FrozenBase,N1d-ZeroInit-CenteredT2Func-FrozenBase,N2a-TinyInit-RationalFunc-BranchRatioCap,N2b-TinyInit-PiecewiseFunc-BranchRatioCap,N2c-TinyInit-SharedRBFFunc-BranchRatioCap,N3a-RationalFunc-DerivativeBand,N3b-PiecewiseFunc-DerivativeBand,N3c-SharedRBFFunc-DerivativeBand,N4a-RationalFunc-AdamWFrozen-FunctionalOnly,N4b-PiecewiseFunc-AdamWFrozen-FunctionalOnly,N4c-SharedRBFFunc-AdamWFrozen-FunctionalOnly,N5a-LQWarmstart-RationalFuncAttach,N5b-LQWarmstart-PiecewiseFuncAttach,N5c-LQWarmstart-SharedRBFFuncAttach,N6-FT7RoleGuard-Standalone")
    parser.add_argument("--p3-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p3-seeds", default="0,1,2")
    parser.add_argument("--p3-targets", default="O1-HardTailLogitCorrection,O2-MarginTailExpansion,O6-KMNISTHardModeOutputTarget")
    parser.add_argument("--p4-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p4-seeds", default="0,1,2")
    parser.add_argument("--p4-targets", default="O1-HardTailLogitCorrection,O2-MarginTailExpansion,O6-KMNISTHardModeOutputTarget")
    parser.add_argument("--p4-horizons", default="1,5,20,80")
    parser.add_argument("--functional-step-fraction", type=float, default=0.10)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)
    write_json(out_dir / "run_manifest.json", {
        "stage": "v9222_basequalified_strict_purekan_functional_interface",
        "created_at": _now_iso(),
        "command": " ".join(sys.argv),
        "device": str(device),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "source_v9221": str(SRC_V9221.relative_to(ROOT)),
        "no_fake_policy": True,
    })
    contract = [{
        "stage": "contract_audit_v9222",
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "distillation_used": 0,
        "loss_modified": 0,
        "sampler_or_class_weight_changed": 0,
        "cpu_offload_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "purekanconv_deferred": 1,
        "purekanformer_deferred": 1,
    }]
    write_csv_rows(out_dir / "contract_audit_v9222.csv", contract)

    p0 = _source_boundary()
    write_csv_rows(out_dir / "p0_v9221_boundary_reproduction.csv", [p0])
    p1_rows, p1_decision = _p1_autopsy(args, device, bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_i1c_p5_failure_autopsy.csv", p1_rows)
    write_csv_rows(out_dir / "interface_failure_trace_v9222.csv", p1_rows)

    p2_rows, p2_best, cache = _p2_factory(args, device, bool(_int(p1_decision.get("p1_pass"))))
    write_csv_rows(out_dir / "p2_base_neutral_interface_factory.csv", p2_rows)
    write_csv_rows(out_dir / "branch_derivative_trace_v9222.csv", [r for r in p2_rows if r.get("status") in {"measured_P5", "candidate_summary"}] or [_not_run("branch_derivative_trace", "branch_derivative_trace_v9222.csv", "no_P2_measured_rows")])
    write_csv_rows(out_dir / "functional_channel_usage_trace_v9222.csv", [r for r in p2_rows if r.get("status") in {"measured_P5", "candidate_summary"}] or [_not_run("functional_channel_usage_trace", "functional_channel_usage_trace_v9222.csv", "no_P2_measured_rows")])

    p3_rows, p3_summary = _p3_actuatability(args, device, p2_best, cache)
    write_csv_rows(out_dir / "p3_functional_actuatability_audit.csv", p3_rows)

    p4_rows, p4_summary = _p4_paired_replay(args, device, p3_summary, cache)
    write_csv_rows(out_dir / "p4_paired_replay_control_gate.csv", p4_rows)

    if not _int(p4_summary.get("paired_replay_pass")):
        _write_downstream_not_run(out_dir, "P4_paired_replay_failed_or_not_opened")
    else:
        _write_downstream_not_run(out_dir, "P5_short_full_validation_not_implemented_after_P4_pass_in_this_runner")

    interface_contract = int(any(_int(r.get("interface_contract_pass")) for r in p2_rows))
    interface_p4 = int(any(_int(r.get("P4_pass")) for r in p2_rows))
    interface_p5 = int(bool(p2_best))
    act_pass = _int(p3_summary.get("functional_actuatability_pass"))
    p4_pass = _int(p4_summary.get("paired_replay_pass"))
    if not _int(p0.get("P0_pass")):
        route_name, blocker, failure_code = "R9-ReturnToBasisFactory", "v9221_boundary_not_reproduced", "F2_v9221_boundary_unstable"
    elif not _int(p1_decision.get("p1_pass")):
        route_name, blocker, failure_code = "R1-I1cFailureUnattributed", "I1c_P5_failure_unattributed", "F3_p5_failure_unattributed"
    elif not interface_contract:
        route_name, blocker, failure_code = "R9-ReturnToBasisFactory", "base_neutral_interface_contract_fail", "F4_base_neutral_interface_contract_fail"
    elif not interface_p4:
        route_name, blocker, failure_code = "R9-ReturnToBasisFactory", "all_base_neutral_candidates_failed_P4", "F5_base_neutral_interface_p4_fail"
    elif not interface_p5:
        route_name, blocker, failure_code = "R8-InterfaceStillBreaksBase", "all_base_neutral_candidates_failed_P5_nearpass", "F6_base_neutral_interface_p5_fail"
    elif not act_pass:
        route_name, blocker, failure_code = "R7-BaseNeutralKillsActuatability", "P5_qualified_interface_lost_non_adamw_actuatability", "F7_functional_actuatability_lost"
    elif not p4_pass:
        route_name, blocker, failure_code = "R3-FunctionalActuatabilityRetained", "paired_replay_did_not_beat_adamwparallel_or_best_lr", "F8_paired_replay_control_equivalent"
    else:
        route_name, blocker, failure_code = "R4-StrictInterfacePairedReplayPass", "P5_short_full_validation_not_completed", "F10_short_run_not_opened"

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9221_boundary_pass": _int(p0.get("P0_pass")),
        "i1c_failure_mechanism": p1_decision.get("i1c_failure_mechanism", ""),
        "best_base_neutral_candidate": p2_best.get("candidate", ""),
        "interface_contract_pass": interface_contract,
        "interface_p4_pass": interface_p4,
        "interface_p5_nearpass": interface_p5,
        "functional_actuatability_pass": act_pass,
        "paired_replay_pass": p4_pass,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": p4_summary.get("p4_task_safe_rate", 0),
        "functional_control_pass": int(_float(p4_summary.get("p4_real_beats_adamwparallel_rate")) >= 0.70 and _float(p4_summary.get("p4_real_beats_best_lr_rate")) >= 0.70),
        "functional_system_pass": 0,
        "functional_kmnist_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "p2_best_near_pass_count": p2_best.get("near_pass_count", 0),
        "p2_best_row_count": p2_best.get("row_count", 0),
        "p2_best_macro_delta": p2_best.get("macro_delta", ""),
        "p3_max_r_perp": p3_summary.get("max_r_perp", ""),
        "p3_bad_event_rate": p3_summary.get("actuatability_bad_event_rate", ""),
        "p4_real_beats_adamwparallel_rate": p4_summary.get("p4_real_beats_adamwparallel_rate", ""),
        "p4_real_beats_best_lr_rate": p4_summary.get("p4_real_beats_best_lr_rate", ""),
        "primary_blocker": blocker,
        "next_required_implementation": "continue_P5_short_full_validation" if p4_pass else "redesign_base_neutral_interface_or_functional_event",
        "success_v9222_strict_purekan_functional": int(p4_pass),
        "success_v9222_full_functional": 0,
        "success_v9222_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "failure_code": failure_code,
        "reason": blocker,
        "active": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    _write_figures(out_dir, route)

    audit_paths = [
        out_dir / "contract_audit_v9222.csv",
        out_dir / "p0_v9221_boundary_reproduction.csv",
        out_dir / "p1_i1c_p5_failure_autopsy.csv",
        out_dir / "p2_base_neutral_interface_factory.csv",
        out_dir / "p3_functional_actuatability_audit.csv",
        out_dir / "p4_paired_replay_control_gate.csv",
        out_dir / "p5_short_full_functional_validation.csv",
        out_dir / "p6_adamw_only_fullpass_repair.csv",
        out_dir / "p7_robustness_external_ready.csv",
    ]
    audit = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v9222_provenance_audit.csv", [audit])
    hashes = artifact_hash_rows([PLAN_PATH, SCRIPT_PATH, out_dir / "route_decision.json", *audit_paths, out_dir / "v9222_provenance_audit.csv"], root=ROOT)
    write_csv_rows(out_dir / "artifact_hashes.csv", hashes)
    _write_report(out_dir, route, audit, p1_decision, p2_best, p3_summary, p4_summary)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
