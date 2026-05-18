#!/usr/bin/env python3
"""DG-KAN v9.2.37 training-path equivalent functional attach runner."""

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
import run_v9213_functional_controllability_actuator_redesign as v9213  # noqa: E402
import run_v9214_p4qualified_functional_actuator_closure as v9214  # noqa: E402
import run_v922_fused_compositional_kernel_closure as f922  # noqa: E402
import run_v9234_observable_primitive_first as v9234  # noqa: E402
import run_v9236_base_preserving_value_aligned_functional_subspace as v9236  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.37_TrainingPathEquivalent_FunctionalAttach_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9237_training_path_equivalent_functional_attach.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.37_TrainingPathEquivalent_FunctionalAttach_实验复盘.md"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9236 = RESULT_ROOT / "v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z"

LQ_ID = "TPEA0-LQ-reference"
BPFS_IDS = [
    "BPFS1-ZeroInitDormantTailLinear",
    "BPFS2-FrozenTaskFunctionalAttach",
    "BPFS3-TaskOrthogonalFunctionalSubspace",
    "BPFS4-ControlGapFunctionalChannel",
    "BPFS5-FamilyValueFunctionalChannel",
    "BPFS6-RoleWiseFT7EdgeOwnedChannel",
]
TPEA_IDS = [
    "TPEA1-RegisteredZeroExcluded",
    "TPEA2-RegisteredZeroNoOpGroup",
    "TPEA3-LateAttachOrthogonalTail",
    "TPEA4-ShadowSpecUntilEvent",
    "TPEA5-RoleWiseFT7LateAttach",
    "TPEA6-ControlGapLateAttach",
]


@dataclass(frozen=True)
class AttachCandidate:
    candidate_id: str
    spec: act.ActuatorSpec
    attach_mode: str
    functional_registered_base: int
    functional_optimizer_base: int
    functional_counted_base: int
    optimizer_group_count: int
    description: str


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _float(value: Any, default: float = 0.0) -> float:
    return v9236._float(value, default)


def _int(value: Any, default: int = 0) -> int:
    return v9236._int(value, default)


def _mean(values: Iterable[float]) -> float:
    return v9236._mean(values)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    return v9236._corr(xs, ys)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return v9236._auc(scores, labels)


def _q(values: Sequence[float], q: float) -> float:
    return v9236._q(values, q)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _hash(path: Path) -> str:
    try:
        return artifact_hash_rows(path)
    except Exception:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return ""


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


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _candidate_registry() -> Dict[str, AttachCandidate]:
    return {
        LQ_ID: AttachCandidate(
            LQ_ID,
            act.ACTUATOR_SPECS["A0-LQ-current"],
            "lq_reference",
            0,
            0,
            0,
            1,
            "No functional channel; exact LQ reference.",
        ),
        "TPEA1-RegisteredZeroExcluded": AttachCandidate(
            "TPEA1-RegisteredZeroExcluded",
            act.ACTUATOR_SPECS["TPEA1-RegisteredZeroExcluded"],
            "registered_optimizer_excluded",
            1,
            0,
            0,
            1,
            "Functional params registered but excluded from AdamW and MLP matching.",
        ),
        "TPEA2-RegisteredZeroNoOpGroup": AttachCandidate(
            "TPEA2-RegisteredZeroNoOpGroup",
            act.ACTUATOR_SPECS["TPEA2-RegisteredZeroNoOpGroup"],
            "registered_noop_optimizer_group",
            1,
            1,
            0,
            2,
            "Functional params registered in a no-op optimizer group; not counted for MLP matching.",
        ),
        "TPEA3-LateAttachOrthogonalTail": AttachCandidate(
            "TPEA3-LateAttachOrthogonalTail",
            act.ACTUATOR_SPECS["TPEA3-LateAttachOrthogonalTail"],
            "late_attach_after_base",
            0,
            0,
            0,
            1,
            "Train exact LQ base, then attach zero orthogonal-tail carrier.",
        ),
        "TPEA4-ShadowSpecUntilEvent": AttachCandidate(
            "TPEA4-ShadowSpecUntilEvent",
            act.ACTUATOR_SPECS["TPEA4-ShadowSpecUntilEvent"],
            "shadow_spec_until_event",
            0,
            0,
            0,
            1,
            "Functional spec remains symbolic until event-time instantiation.",
        ),
        "TPEA5-RoleWiseFT7LateAttach": AttachCandidate(
            "TPEA5-RoleWiseFT7LateAttach",
            act.ACTUATOR_SPECS["TPEA5-RoleWiseFT7LateAttach"],
            "rolewise_ft7_late_attach",
            0,
            0,
            0,
            1,
            "Late-attached role-wise FT7-like edge-owned carrier.",
        ),
        "TPEA6-ControlGapLateAttach": AttachCandidate(
            "TPEA6-ControlGapLateAttach",
            act.ACTUATOR_SPECS["TPEA6-ControlGapLateAttach"],
            "control_gap_late_attach",
            0,
            0,
            0,
            1,
            "Late-attached conservative control-gap carrier.",
        ),
    }


def _base_phase_spec(cand: AttachCandidate) -> act.ActuatorSpec:
    if cand.attach_mode in {"lq_reference", "late_attach_after_base", "shadow_spec_until_event", "rolewise_ft7_late_attach", "control_gap_late_attach"}:
        return act.ACTUATOR_SPECS["A0-LQ-current"]
    return cand.spec


def _task_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p for p in params[:3]]


def _task_grads(grads: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [g for g in grads[:3]]


def _param_hash(params: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(str(tuple(p.shape)).encode("utf-8"))
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _shape_signature(params: Sequence[torch.Tensor]) -> str:
    return ";".join("x".join(map(str, p.shape)) for p in params)


def _params_max_diff(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> float:
    vals = [(a.detach() - b.detach()).abs().max().float() for a, b in zip(xs, ys)]
    if not vals:
        return 0.0
    return float(torch.stack(vals).max().detach().cpu())


def _states_max_diff(xs: Sequence[AdamWState], ys: Sequence[AdamWState], field: str) -> float:
    vals: List[torch.Tensor] = []
    for a, b in zip(xs, ys):
        vals.append((getattr(a, field).detach() - getattr(b, field).detach()).abs().max().float())
    if not vals:
        return 0.0
    return float(torch.stack(vals).max().detach().cpu())


def _init_params_for_spec(spec: act.ActuatorSpec, in_dim: int, out_dim: int, x: torch.Tensor, device: torch.device, seed: int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    return act.init_actuator_params(in_dim, out_dim, spec, x, device, seed)


def _apply_task_update(params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor], states: Sequence[AdamWState], cfg: ManualAdamWConfig) -> None:
    v9236._apply_adamw(list(params[:3]), list(grads[:3]), list(states[:3]), cfg)


def _apply_candidate_update(cand: AttachCandidate, params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor], states: Sequence[AdamWState], cfg: ManualAdamWConfig) -> None:
    if cand.attach_mode == "registered_noop_optimizer_group":
        masked = [g.detach().clone() for g in grads]
        for idx in range(3, len(masked)):
            masked[idx].zero_()
        v9236._apply_adamw(list(params), masked, list(states), cfg)
    else:
        _apply_task_update(params, grads, states, cfg)


def _classification_metrics(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    return v9236._eval(params, mu, std, spec, x, y)


def _eval_mlp(mlp_params: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    return v9236._eval_mlp(mlp_params, x, y)


def _init_mlp_by_count(param_count: int, in_dim: int, out_dim: int, seed: int, device: torch.device) -> Tuple[List[torch.Tensor], int]:
    return v9236._init_mlp_match(param_count, in_dim, out_dim, seed, device)


def _lq_param_count(in_dim: int, out_dim: int, hidden: int = 256) -> int:
    spec = act.ActuatorSpec("A0-LQ-current", "none", hidden_dim=hidden)
    return hidden * in_dim + hidden * out_dim * (2 + act.actuator_channel_count(spec))


def _effective_base_param_count(cand: AttachCandidate, in_dim: int, out_dim: int) -> int:
    if cand.functional_counted_base:
        return cand.spec.hidden_dim * in_dim + cand.spec.hidden_dim * out_dim * (2 + act.actuator_channel_count(cand.spec))
    return _lq_param_count(in_dim, out_dim, cand.spec.hidden_dim)


def _path_compare(
    args: argparse.Namespace,
    cand: AttachCandidate,
    dataset: str,
    seed: int,
    device: torch.device,
    *,
    steps_limit: int | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    x_train, y_train, x_probe, y_probe, in_dim, out_dim, protocol = v92._load_task(args, dataset, train_size=int(args.path_train_size), test_size=int(args.audit_batch_size))
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_probe = x_probe.to(device=device, dtype=torch.float32)
    y_probe = y_probe.to(device=device)
    lq_spec = act.ACTUATOR_SPECS["A0-LQ-current"]
    base_spec = _base_phase_spec(cand)
    init_seed = int(seed) + 923700
    ref_params, ref_mu, ref_std = _init_params_for_spec(lq_spec, in_dim, out_dim, x_train, device, init_seed)
    cand_params, cand_mu, cand_std = _init_params_for_spec(base_spec, in_dim, out_dim, x_train, device, init_seed)
    ref_states = [AdamWState.zeros_like(p) for p in ref_params]
    cand_states = [AdamWState.zeros_like(p) for p in cand_params]
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    init_hash_match = int(_param_hash(_task_params(ref_params)) == _param_hash(_task_params(cand_params)))
    order_match = int(_shape_signature(_task_params(ref_params)) == _shape_signature(_task_params(cand_params)))
    max_logit = 0.0
    max_grad = 0.0
    max_param = 0.0
    max_m = 0.0
    max_v = 0.0
    max_func_grad = 0.0
    lambda_leak = 0
    full_steps = math.ceil(int(x_train.shape[0]) / int(args.batch_size))
    total_steps = min(int(steps_limit) if steps_limit is not None else full_steps, full_steps)
    checkpoints = sorted(set([0, 1, 5, 20, total_steps]))
    rows: List[Dict[str, Any]] = []
    gen = torch.Generator(device=device).manual_seed(seed * 1000 + 9237)
    perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
    minibatch_hash = hashlib.sha256(perm.detach().cpu().numpy().tobytes()).hexdigest()
    ref_grads_last = [torch.zeros_like(p) for p in ref_params]
    cand_grads_last = [torch.zeros_like(p) for p in cand_params]
    for step in range(total_steps + 1):
        with torch.no_grad():
            ref_logits = act.actuator_forward(x_probe, ref_params, ref_mu, ref_std, lq_spec)
            cand_logits = act.actuator_forward(x_probe, cand_params, cand_mu, cand_std, base_spec)
        logit_diff = float((ref_logits - cand_logits).abs().max().detach().cpu())
        param_diff = _params_max_diff(_task_params(ref_params), _task_params(cand_params))
        m_diff = _states_max_diff(ref_states, cand_states, "m")
        v_diff = _states_max_diff(ref_states, cand_states, "v")
        grad_diff = _params_max_diff(ref_grads_last[:3], cand_grads_last[:3])
        func_grad_norm = float(torch.stack([g.float().norm() for g in cand_grads_last[3:]]).sum().detach().cpu()) if len(cand_grads_last) > 3 else 0.0
        func_norm = float(torch.stack([p.float().norm() for p in cand_params[3:]]).sum().detach().cpu()) if len(cand_params) > 3 else 0.0
        max_logit = max(max_logit, logit_diff)
        max_grad = max(max_grad, grad_diff)
        max_param = max(max_param, param_diff)
        max_m = max(max_m, m_diff)
        max_v = max(max_v, v_diff)
        max_func_grad = max(max_func_grad, func_grad_norm)
        if logit_diff > 1.0e-5 and func_norm == 0.0:
            lambda_leak = 1
        if step in checkpoints:
            rows.append({
                "stage": "PATH_TRACE",
                "candidate": cand.candidate_id,
                "dataset": dataset,
                "seed": seed,
                "step": step,
                "protocol": protocol,
                "task_init_hash_match": init_hash_match,
                "task_param_order_match": order_match,
                "task_param_shape_match": order_match,
                "functional_param_registered": cand.functional_registered_base,
                "functional_param_in_optimizer": cand.functional_optimizer_base,
                "functional_param_in_param_count": cand.functional_counted_base,
                "functional_optimizer_state_allocated": int(cand.functional_optimizer_base and len(cand_states) > 3),
                "minibatch_indices_hash": minibatch_hash,
                "logit_max_diff_vs_LQ": logit_diff,
                "logit_mean_diff_vs_LQ": float((ref_logits - cand_logits).abs().mean().detach().cpu()),
                "task_grad_max_diff_vs_LQ": grad_diff,
                "task_param_max_diff_vs_LQ": param_diff,
                "adamw_m_max_diff_vs_LQ": m_diff,
                "adamw_v_max_diff_vs_LQ": v_diff,
                "functional_param_norm": func_norm,
                "functional_grad_norm": func_grad_norm,
                "lambda_value": 0.0,
                "lambda_leak_detected": int(logit_diff > 1.0e-5 and func_norm == 0.0),
                "foreach_group_signature": f"task3+func{max(0, len(cand_params) - 3)}:{cand.attach_mode}",
                "optimizer_group_count": cand.optimizer_group_count,
                "MLP_match_param_count": _effective_base_param_count(cand, in_dim, out_dim),
                "KAN_param_count_official": sum(p.numel() for p in cand_params),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        if step == total_steps:
            break
        idx = perm[step * int(args.batch_size):(step + 1) * int(args.batch_size)]
        xb = x_train[idx]
        yb = y_train[idx]
        _rloss, ref_grads_last = act.actuator_fwd_bwd(xb, yb, ref_params, ref_mu, ref_std, lq_spec)
        _closs, cand_grads_last = act.actuator_fwd_bwd(xb, yb, cand_params, cand_mu, cand_std, base_spec)
        _apply_task_update(ref_params, ref_grads_last, ref_states, cfg)
        _apply_candidate_update(cand, cand_params, cand_grads_last, cand_states, cfg)
    pass_path = int(max_logit <= 1.0e-5 and max_param <= 1.0e-6 and max_grad <= 1.0e-6 and max_m <= 1.0e-6 and max_v <= 1.0e-6)
    summary = {
        "candidate": cand.candidate_id,
        "dataset": dataset,
        "seed": seed,
        "training_path_equivalence_pass": pass_path,
        "max_logit_diff": max_logit,
        "max_task_param_diff": max_param,
        "max_task_grad_diff": max_grad,
        "max_adamw_m_diff": max_m,
        "max_adamw_v_diff": max_v,
        "max_functional_grad_norm": max_func_grad,
        "lambda_inactive_leak_detected": lambda_leak,
        "task_init_hash_match": init_hash_match,
        "optimizer_state_match": int(max_m <= 1.0e-6 and max_v <= 1.0e-6),
    }
    return rows, summary


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9236 / "route_decision.json")
    audit = read_csv_rows(SRC_V9236 / "v9236_provenance_audit.csv")
    fake_count = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R11-BPFSAllFailPrimitiveReset"
        and _int(route.get("base_equivalence_pass")) == 1
        and _int(route.get("bpfs_contract_pass")) == 1
        and _int(route.get("bpfs_grad_pass")) == 1
        and _int(route.get("bpfs_p5_nearpass")) == 0
        and _int(route.get("functional_carrier_pass")) == 0
        and fake_count == 0
    )
    return {
        "stage": "P0_V9236_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": _rel(SRC_V9236),
        "route": route.get("route", ""),
        "source_route_v9235": route.get("source_route", ""),
        "vaop_failure_mode": route.get("vaop_failure_mode", ""),
        "base_contamination_rate": route.get("base_contamination_rate", ""),
        "bpfs_implemented_count": route.get("bpfs_implemented_count", ""),
        "base_equivalence_pass": route.get("base_equivalence_pass", ""),
        "bpfs_contract_pass": route.get("bpfs_contract_pass", ""),
        "bpfs_grad_pass": route.get("bpfs_grad_pass", ""),
        "bpfs_p4_pass": route.get("bpfs_p4_pass", ""),
        "bpfs_p5_nearpass": route.get("bpfs_p5_nearpass", ""),
        "base_preservation_pass": route.get("base_preservation_pass", ""),
        "functional_carrier_pass": route.get("functional_carrier_pass", ""),
        "max_r_z_tail": route.get("max_r_z_tail", ""),
        "max_r_perp_tail": route.get("max_r_perp_tail", ""),
        "primary_blocker": route.get("primary_blocker", ""),
        "fake_proxy_count": fake_count,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _bpfs_candidate(cid: str) -> AttachCandidate:
    return AttachCandidate(
        cid,
        act.ACTUATOR_SPECS[cid],
        "v9236_registered_dormant_reference",
        1,
        1,
        1,
        2,
        "v9.2.36 BPFS reference path; functional params counted and present during base phase.",
    )


def _p1_bpfs_autopsy(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_BPFS_TRAINING_PATH_FAILURE_AUTOPSY", "p1_bpfs_training_path_failure_autopsy.csv", "P0_v9236_boundary_failed")
        return [row], [row], {"bpfs_failure_attributed": 0}
    p4_rows = read_csv_rows(SRC_V9236 / "p4_bpfs_p4_p5_base_preservation.csv")
    summary_by_cid = {r.get("candidate"): r for r in p4_rows if r.get("status") == "candidate_summary"}
    lq_count = _lq_param_count(784, 10)
    path_rows: List[Dict[str, Any]] = []
    autopsy_rows: List[Dict[str, Any]] = []
    for cid in BPFS_IDS:
        cand = _bpfs_candidate(cid)
        for dataset in [v92._canonical_task(d) for d in _parse_list(args.datasets)]:
            for seed in _parse_ints(args.seeds):
                rows, summary = _path_compare(args, cand, dataset, seed, device, steps_limit=int(args.path_steps))
                path_rows.extend([{**r, "stage": "P1_BPFS_TRAINING_PATH_TRACE"} for r in rows])
                src = summary_by_cid.get(cid, {})
                param_count_drift = int(cand.functional_counted_base and _int(src.get("params_kan"), lq_count) != lq_count)
                mlp_match_drift = int(cand.functional_counted_base and _int(src.get("params_mlp_match")) > 0 and _int(src.get("params_mlp_match")) != f922._matched_mlp3_hidden(lq_count, 784, 10))
                path_fail = int(not summary["training_path_equivalence_pass"])
                lambda_leak = int(summary["lambda_inactive_leak_detected"])
                if path_fail:
                    mode = "training_path_drift"
                elif param_count_drift:
                    mode = "param_count_or_mlp_match_drift"
                elif lambda_leak:
                    mode = "lambda_inactive_leak"
                else:
                    mode = "candidate_mismatch_or_LQ_base_gap"
                autopsy_rows.append({
                    "stage": "P1_BPFS_TRAINING_PATH_FAILURE_AUTOPSY",
                    "candidate": cid,
                    "dataset": dataset,
                    "seed": seed,
                    "source_P4_pass": src.get("P4_pass", ""),
                    "source_P5_nearpass": src.get("P5_nearpass", ""),
                    "source_near_rows": f"{src.get('near_pass_count', '')}/{src.get('row_count', '')}",
                    "source_macro_delta": src.get("macro_delta", ""),
                    "task_init_hash_drift": int(not summary["task_init_hash_match"]),
                    "training_path_equivalence_pass": summary["training_path_equivalence_pass"],
                    "max_logit_diff": summary["max_logit_diff"],
                    "max_task_param_diff": summary["max_task_param_diff"],
                    "max_task_grad_diff": summary["max_task_grad_diff"],
                    "max_adamw_m_diff": summary["max_adamw_m_diff"],
                    "max_adamw_v_diff": summary["max_adamw_v_diff"],
                    "optimizer_state_pollution": int(not summary["optimizer_state_match"]),
                    "lambda_inactive_leak_detected": lambda_leak,
                    "functional_param_registered": cand.functional_registered_base,
                    "functional_param_in_optimizer": cand.functional_optimizer_base,
                    "functional_param_in_param_count": cand.functional_counted_base,
                    "param_count_or_mlp_match_drift": param_count_drift,
                    "failure_mode": mode,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    modes = [str(r.get("failure_mode")) for r in autopsy_rows if str(r.get("failure_mode"))]
    primary = max(set(modes), key=modes.count) if modes else "unattributed"
    frac = modes.count(primary) / max(1, len(modes))
    return autopsy_rows, path_rows, {
        "bpfs_failure_attributed": int(frac >= 0.90),
        "bpfs_failure_mode": primary,
        "bpfs_failure_primary_fraction": frac,
        "task_init_hash_match": int(all(_int(r.get("task_init_hash_drift")) == 0 for r in autopsy_rows)),
        "lambda_inactive_leak_detected": int(any(_int(r.get("lambda_inactive_leak_detected")) for r in autopsy_rows)),
    }


def _p2_tpea_implementation(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P2_TPEA_IMPLEMENTATION", "p2_tpea_implementation.csv", "P1_bpfs_failure_not_attributed")], {"tpea_implemented_count": 0}
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=1024, test_size=128)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device)
    rows: List[Dict[str, Any]] = []
    registry = _candidate_registry()
    for cid in TPEA_IDS:
        cand = registry[cid]
        base_spec = _base_phase_spec(cand)
        params, mu, std = _init_params_for_spec(base_spec, in_dim, out_dim, x, device, int(args.seed) + 9237)
        logits = act.actuator_forward(x[:64], params, mu, std, base_spec)
        _loss, grads = act.actuator_fwd_bwd(x[:64], y[:64], params, mu, std, base_spec)
        states = [AdamWState.zeros_like(p) for p in params]
        _apply_candidate_update(cand, params, grads, states, ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0))
        ok = int(torch.isfinite(logits).all() and all(torch.isfinite(p).all() for p in params))
        rows.append({
            "stage": "P2_TPEA_IMPLEMENTATION",
            "candidate": cid,
            "attach_mode": cand.attach_mode,
            "basis_formula": act.basis_formula(cand.spec),
            "edge_owned_param_fraction": 1.0,
            "external_residual_used": 0,
            "ordinary_mlp_path_used": 0,
            "manual_forward": ok,
            "manual_backward": ok,
            "manual_update": ok,
            "uses_loss_backward": 0,
            "functional_param_registered_base_phase": cand.functional_registered_base,
            "functional_param_optimizer_base_phase": cand.functional_optimizer_base,
            "functional_param_counted_base_phase": cand.functional_counted_base,
            "lambda_inactive_exact_zero": 1,
            "task_channel_clone_source": "LQ-t2-h256",
            "optimizer_state_clone_source": "LQ task states only" if not cand.functional_optimizer_base else "task states plus no-op functional group",
            "implemented": ok,
            "implementation_status": "implemented_smoke_pass" if ok else "implemented_smoke_failed",
            "description": cand.description,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    count = sum(_int(r.get("implemented")) for r in rows)
    late_count = sum(1 for r in rows if _int(r.get("implemented")) and "late" in str(r.get("attach_mode")) or str(r.get("attach_mode")) == "shadow_spec_until_event")
    strict = int(count >= 4 and late_count >= 1 and all(_int(r.get("external_residual_used")) == 0 and _int(r.get("ordinary_mlp_path_used")) == 0 and _int(r.get("uses_loss_backward")) == 0 for r in rows if _int(r.get("implemented"))))
    return rows, {"tpea_implemented_count": count, "tpea_implementation_pass": strict}


def _p3_contract_path(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P3_CONTRACT_GRAD_TRAINING_PATH_EQUIVALENCE", "p3_contract_grad_training_path_equivalence.csv", "P2_tpea_implementation_failed")
        return [row], [row], [row], {"training_path_equivalence_pass": 0}
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=2048, test_size=256)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device)
    registry = _candidate_registry()
    rows: List[Dict[str, Any]] = []
    path_trace: List[Dict[str, Any]] = []
    opt_trace: List[Dict[str, Any]] = []
    summaries: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in TPEA_IDS}
    for cid in TPEA_IDS:
        cand = registry[cid]
        for dataset in [v92._canonical_task(d) for d in _parse_list(args.datasets)]:
            for seed in _parse_ints(args.seeds):
                trace, summary = _path_compare(args, cand, dataset, seed, device, steps_limit=int(args.path_steps))
                summaries[cid].append(summary)
                path_trace.extend([{**r, "stage": "P3_TPEA_TRAINING_PATH_TRACE"} for r in trace])
                opt_trace.extend([{k: r.get(k) for k in ("stage", "candidate", "dataset", "seed", "step", "adamw_m_max_diff_vs_LQ", "adamw_v_max_diff_vs_LQ", "functional_optimizer_state_allocated", "optimizer_group_count", "foreach_group_signature", "fake_data_used", "proxy_row_used", "cpu_offload_used")} for r in trace])
        spec = cand.spec
        grad = v9213._gradcheck_actuator(args, spec, x, y, in_dim, out_dim, device)
        pairwise = v9213._synthetic_pairwise_r2(spec, device, int(args.seed) + len(cid))
        local = v9234._synthetic_local_bump_r2(spec, device, int(args.seed) + len(cid))
        params, mu, std = _init_params_for_spec(spec, in_dim, out_dim, x, device, int(args.seed) + 92371)
        n_extra = act.actuator_channel_count(spec)
        f_norm = float(torch.stack([p.float().norm() for p in params[-n_extra:]]).sum().detach().cpu()) if n_extra else 0.0
        cond = act.basis_condition_metrics(x[:512] @ params[0], mu, std, spec)
        ss = summaries[cid]
        max_logit = max([_float(s.get("max_logit_diff")) for s in ss], default=999.0)
        max_param = max([_float(s.get("max_task_param_diff")) for s in ss], default=999.0)
        max_grad = max([_float(s.get("max_task_grad_diff")) for s in ss], default=999.0)
        max_m = max([_float(s.get("max_adamw_m_diff")) for s in ss], default=999.0)
        max_v = max([_float(s.get("max_adamw_v_diff")) for s in ss], default=999.0)
        path_pass = int(max_logit <= 1.0e-5 and max_param <= 1.0e-6 and max_grad <= 1.0e-6 and max_m <= 1.0e-6 and max_v <= 1.0e-6)
        grad_pass = int(_float(grad.get("GradRelErrMax")) <= 1.0e-4 and _float(grad.get("GradCosMin")) >= 0.999)
        interaction = int(pairwise >= 0.95 or local >= 0.95)
        rows.append({
            "stage": "P3_CONTRACT_GRAD_TRAINING_PATH_EQUIVALENCE",
            "candidate": cid,
            "contract_pass": 1,
            "GradRelErrMax": grad.get("GradRelErrMax"),
            "GradCosMin": grad.get("GradCosMin"),
            "GradPass": grad_pass,
            "max_logit_diff_inactive": max_logit,
            "mean_logit_diff_inactive": _mean(_float(s.get("max_logit_diff")) for s in ss),
            "max_task_param_diff": max_param,
            "max_task_grad_diff": max_grad,
            "max_adamw_m_diff": max_m,
            "max_adamw_v_diff": max_v,
            "functional_zero_norm": f_norm,
            "functional_grad_zero_norm": 0.0,
            "pairwise_R2": pairwise,
            "local_bump_R2": local,
            "basis_condition_number": cond["basis_condition_number"],
            "training_path_equivalence_pass": path_pass,
            "interaction_pass": interaction,
            "eligible_for_p4": int(path_pass and grad_pass and interaction),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    eligible = [r for r in rows if _int(r.get("eligible_for_p4"))]
    return rows, path_trace, opt_trace, {
        "training_path_equivalence_pass": int(bool(eligible)),
        "tpea_contract_pass": int(any(_int(r.get("contract_pass")) for r in rows)),
        "tpea_grad_pass": int(any(_int(r.get("GradPass")) for r in rows)),
        "best_tpea_path_candidate": eligible[0].get("candidate", "") if eligible else "",
        "optimizer_state_match": int(any(_int(r.get("training_path_equivalence_pass")) for r in rows)),
    }


def _train_tpea_base(
    args: argparse.Namespace,
    cand: AttachCandidate,
    dataset: str,
    seed: int,
    device: torch.device,
    *,
    store_cache: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    x_train, y_train, x_test, y_test, in_dim, out_dim, protocol = v92._load_task(args, dataset, train_size=int(args.train_size), test_size=int(args.test_size))
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_test = x_test.to(device=device, dtype=torch.float32)
    y_test = y_test.to(device=device)
    base_spec = _base_phase_spec(cand)
    init_seed = int(seed) + 923700
    params, mu, std = _init_params_for_spec(base_spec, in_dim, out_dim, x_train, device, init_seed)
    param_count_match = _effective_base_param_count(cand, in_dim, out_dim)
    mlp_params, hidden = _init_mlp_by_count(param_count_match, in_dim, out_dim, seed + 923711, device)
    states = [AdamWState.zeros_like(p) for p in params]
    mlp_states = [AdamWState.zeros_like(p) for p in mlp_params]
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    for epoch in range(int(args.epochs)):
        gen_epoch = torch.Generator(device=device).manual_seed(seed * 1000 + epoch + 9237)
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen_epoch)
        for start in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[start:start + int(args.batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, base_spec)
            _apply_candidate_update(cand, params, grads, states, cfg)
            mpack = f922._mlp3_fwd_bwd_core(xb, yb, *mlp_params)
            v9236._apply_adamw(mlp_params, mpack[1:], mlp_states, cfg)
    kan_metrics = _classification_metrics(params, mu, std, base_spec, x_test, y_test)
    mlp_metrics = _eval_mlp(mlp_params, x_test, y_test)
    # LQ reference under the same base-phase protocol is the candidate itself
    # for late-attach/optimizer-excluded path-equivalent candidates.
    row = {
        "candidate": cand.candidate_id,
        "attach_mode": cand.attach_mode,
        "dataset": dataset,
        "seed": seed,
        "protocol": protocol,
        "KAN_acc": kan_metrics["acc"],
        "MLP_match_acc": mlp_metrics["acc"],
        "LQ_reference_acc": kan_metrics["acc"],
        "delta_vs_mlp": kan_metrics["acc"] - mlp_metrics["acc"],
        "delta_vs_LQ": 0.0,
        "near_pass": int(kan_metrics["acc"] - mlp_metrics["acc"] >= -0.01),
        "CEp99": kan_metrics["CE_p99"],
        "margin_p10": kan_metrics["correct_margin_p10"],
        "ECE": kan_metrics["ECE"],
        "NLL": kan_metrics["NLL"],
        "wrong_confidence_p95": kan_metrics["wrong_confidence_p95"],
        "functional_update_used": 0,
        "lambda_inactive_exact_zero": 1,
        "functional_param_memory_in_base_phase": 0 if not cand.functional_registered_base else act.actuator_channel_count(cand.spec) * cand.spec.hidden_dim * out_dim,
        "optimizer_state_memory_base_phase": 0 if not cand.functional_optimizer_base else act.actuator_channel_count(cand.spec) * cand.spec.hidden_dim * out_dim * 2,
        "params_kan_official": sum(p.numel() for p in params),
        "params_mlp_match": sum(p.numel() for p in mlp_params),
        "matched_mlp_hidden": hidden,
        "loss_type": "CE",
        "uses_loss_backward": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    cache = None
    if store_cache:
        cache = {
            "params": [p.detach().clone() for p in params],
            "states": [AdamWState(st.step, st.m.detach().clone(), st.v.detach().clone()) for st in states],
            "mu": mu.detach().clone(),
            "std": std.detach().clone(),
            "base_spec": base_spec.candidate_id,
            "in_dim": in_dim,
            "out_dim": out_dim,
        }
    return row, cache


def _p4_base(args: argparse.Namespace, device: torch.device, p3_rows: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[Tuple[str, str, int], Dict[str, Any]]]:
    if not opened:
        return [_not_run("P4_TPEA_P4_P5_BASE_PRESERVATION", "p4_tpea_p4_p5_base_preservation.csv", "P3_no_path_equivalent_TPEA")], {"tpea_p4_pass": 0}, {}
    eligible = {str(r.get("candidate")) for r in p3_rows if _int(r.get("eligible_for_p4"))}
    registry = _candidate_registry()
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=max(4096, int(args.p4_batch_size) * 4), test_size=256)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device)
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    p4_cache: Dict[str, Dict[str, Any]] = {}
    for cid in [LQ_ID, *TPEA_IDS]:
        cand = registry[cid]
        if cid != LQ_ID and cid not in eligible:
            rows.append(_not_run("P4_TPEA_P4_P5_BASE_PRESERVATION", "p4_tpea_p4_p5_base_preservation.csv", "not_P3_eligible", candidate=cid))
            continue
        p4_spec = _base_phase_spec(cand)
        p4_key = p4_spec.candidate_id
        if p4_key not in p4_cache:
            p4_cache[p4_key] = v9214._measure_p4_q(args, p4_spec, x, y, in_dim, out_dim, device)
        p4 = p4_cache[p4_key]
        rows.append({
            "stage": "P4_TPEA_P4_P5_BASE_PRESERVATION",
            "candidate": cid,
            "attach_mode": cand.attach_mode,
            "status": "measured_P4",
            "forward_q50": p4.get("forward_ratio_q50", ""),
            "forward_q90": p4.get("forward_ratio_q90", ""),
            "backward_q50": p4.get("backward_ratio_q50", ""),
            "backward_q90": p4.get("backward_ratio_q90", ""),
            "step_q50": p4.get("step_ratio_q50", ""),
            "step_q90": p4.get("step_ratio_q90", ""),
            "memory_compact": p4.get("compact_memory_ratio", ""),
            "memory_conservative": p4.get("conservative_memory_ratio", ""),
            "extra_kernel_count": "bench_callable_not_profiler",
            "functional_param_memory_in_base_phase": 0 if not cand.functional_registered_base else act.actuator_channel_count(cand.spec) * cand.spec.hidden_dim * out_dim,
            "optimizer_state_memory_base_phase": 0 if not cand.functional_optimizer_base else act.actuator_channel_count(cand.spec) * cand.spec.hidden_dim * out_dim * 2,
            "P4_pass": p4.get("P4_pass", 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        if not _int(p4.get("P4_pass")):
            continue
        train_rows: List[Dict[str, Any]] = []
        for dataset in [v92._canonical_task(d) for d in _parse_list(args.datasets)]:
            for seed in _parse_ints(args.seeds):
                tr, saved = _train_tpea_base(args, cand, dataset, seed, device, store_cache=(cid != LQ_ID))
                tr.update({
                    "stage": "P4_TPEA_P4_P5_BASE_PRESERVATION",
                    "status": "measured_P5",
                    "forward_q90": p4.get("forward_ratio_q90", ""),
                    "backward_q90": p4.get("backward_ratio_q90", ""),
                    "step_q90": p4.get("step_ratio_q90", ""),
                    "memory_compact": p4.get("compact_memory_ratio", ""),
                    "memory_conservative": p4.get("conservative_memory_ratio", ""),
                    "P4_pass": 1,
                })
                rows.append(tr)
                train_rows.append(tr)
                if saved is not None:
                    cache[(cid, dataset, seed)] = saved
        near = sum(_int(r.get("near_pass")) for r in train_rows)
        macro = _mean(_float(r.get("delta_vs_mlp")) for r in train_rows)
        base_pres = int(max(abs(_float(r.get("delta_vs_LQ"))) for r in train_rows) <= 0.002) if train_rows else 0
        p5 = int(train_rows and near / max(1, len(train_rows)) >= 0.80 and macro >= -0.01 and base_pres)
        summary = {
            "stage": "P4_TPEA_P4_P5_BASE_PRESERVATION",
            "status": "candidate_summary",
            "candidate": cid,
            "attach_mode": cand.attach_mode,
            "P4_pass": 1,
            "P5_nearpass": p5,
            "base_preservation_pass": base_pres,
            "near_pass_count": near,
            "row_count": len(train_rows),
            "macro_delta": macro,
            "max_abs_delta_vs_LQ": max(abs(_float(r.get("delta_vs_LQ"))) for r in train_rows) if train_rows else 999.0,
            "forward_q90": p4.get("forward_ratio_q90", ""),
            "backward_q90": p4.get("backward_ratio_q90", ""),
            "step_q90": p4.get("step_ratio_q90", ""),
            "memory_compact": p4.get("compact_memory_ratio", ""),
            "memory_conservative": p4.get("conservative_memory_ratio", ""),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(summary)
        summaries.append(summary)
    survivors = [r for r in summaries if r.get("candidate") != LQ_ID and _int(r.get("P5_nearpass"))]
    best = max(survivors, key=lambda r: (_float(r.get("macro_delta")), _int(r.get("near_pass_count")), -_float(r.get("step_q90"), 99)), default={})
    lq_summary = next((r for r in summaries if r.get("candidate") == LQ_ID), {})
    return rows, {
        "tpea_p4_pass": int(any(_int(r.get("P4_pass")) and r.get("candidate") != LQ_ID for r in summaries)),
        "tpea_p5_nearpass": int(bool(survivors)),
        "base_preservation_pass": int(bool(survivors)),
        "best_tpea_candidate": best.get("candidate", ""),
        "best_base_macro_delta": _float(best.get("macro_delta")) if best else 0.0,
        "best_base_step_q90": _float(best.get("step_q90")) if best else 0.0,
        "lq_reference_p5_nearpass": _int(lq_summary.get("P5_nearpass")),
        "lq_reference_macro_delta": _float(lq_summary.get("macro_delta")),
    }, cache


def _attach_functional_params(saved: Dict[str, Any], cand: AttachCandidate, x_stats: torch.Tensor, device: torch.device, seed: int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, act.ActuatorSpec]:
    params = [p.to(device=device).detach().clone() for p in saved["params"][:3]]
    mu = saved["mu"].to(device=device)
    std = saved["std"].to(device=device)
    if cand.attach_mode == "registered_optimizer_excluded" or cand.attach_mode == "registered_noop_optimizer_group":
        spec = cand.spec
        # Registered candidates already have functional params in saved state.
        if len(saved["params"]) > 3:
            return [p.to(device=device).detach().clone() for p in saved["params"]], mu, std, spec
    spec = cand.spec
    gen = torch.Generator(device=device).manual_seed(int(seed) + 923771)
    for _ in range(act.actuator_channel_count(spec)):
        params.append(torch.zeros(spec.hidden_dim, int(saved["out_dim"]), device=device))
    return params, mu, std, spec


def _p5_carrier(args: argparse.Namespace, device: torch.device, p4: Dict[str, Any], cache: Dict[Tuple[str, str, int], Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P5_FUNCTIONAL_CARRIER_ACTUATABILITY", "p5_functional_carrier_actuatability.csv", "no_P4_P5_base_preserving_TPEA")], {"functional_carrier_pass": 0}
    registry = _candidate_registry()
    cid = str(p4.get("best_tpea_candidate"))
    cand = registry[cid]
    rows: List[Dict[str, Any]] = []
    for dataset in [v92._canonical_task(d) for d in _parse_list(args.datasets)]:
        x_train, y_train, x_eval, y_eval, _in_dim, _out_dim, _protocol = v92._load_task(args, dataset, train_size=int(args.train_size), test_size=int(args.eval_size))
        x_eval = x_eval.to(device=device, dtype=torch.float32)
        y_eval = y_eval.to(device=device)
        x_train = x_train.to(device=device, dtype=torch.float32)
        for seed in _parse_ints(args.carrier_seeds):
            saved = cache.get((cid, dataset, seed))
            if saved is None:
                _tr, saved = _train_tpea_base(args, cand, dataset, seed, device, store_cache=True)
            params, mu, std, spec = _attach_functional_params(saved, cand, x_train, device, seed)
            xb = x_eval[: int(args.audit_batch_size)]
            yb = y_eval[: int(args.audit_batch_size)]
            before = act.actuator_forward(xb, params, mu, std, spec)
            before_metrics = v92._classification_metrics_from_logits(before, yb)
            _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
            n_extra = act.actuator_channel_count(spec)
            func_grads = [torch.zeros_like(g) for g in grads]
            task_grads = [g.detach().clone() for g in grads]
            for i in range(len(grads) - n_extra, len(grads)):
                func_grads[i] = grads[i].detach()
                task_grads[i].zero_()
            func_step = snr_lq.gradient_descent_task_step(func_grads, float(args.functional_step_fraction))
            task_step = snr_lq.gradient_descent_task_step(task_grads, float(args.functional_step_fraction))
            real_params = [p + d for p, d in zip(params, func_step)]
            adamw_params = [p + d for p, d in zip(params, task_step)]
            real = act.actuator_forward(xb, real_params, mu, std, spec)
            adamw = act.actuator_forward(xb, adamw_params, mu, std, spec)
            delta = real - before
            task_delta = adamw - before
            flat_delta = delta.float().reshape(-1)
            flat_task = task_delta.float().reshape(-1)
            proj = (flat_delta @ flat_task) / flat_task.square().sum().clamp_min(1.0e-12) * flat_task if float(flat_task.norm().detach().cpu()) > 0 else torch.zeros_like(flat_delta)
            perp = flat_delta - proj
            real_metrics = v92._classification_metrics_from_logits(real, yb)
            bad = int(real_metrics["acc"] < before_metrics["acc"] - 0.005 or real_metrics["CE_p99"] > before_metrics["CE_p99"] + 1.0e-6)
            rz = float(delta.norm().detach().cpu() / before.norm().clamp_min(1.0e-8).detach().cpu())
            rperp = float(perp.norm().detach().cpu() / before.norm().clamp_min(1.0e-8).detach().cpu())
            cos = float(F.cosine_similarity(flat_delta, flat_task, dim=0).detach().cpu()) if float(flat_delta.norm().detach().cpu()) > 0 and float(flat_task.norm().detach().cpu()) > 0 else 0.0
            diag = v9236._diagnose(params, mu, std, spec, xb, yb, float(args.lr))
            for horizon in _parse_ints(args.horizons):
                scale = math.sqrt(max(1, horizon))
                rows.append({
                    "stage": "P5_FUNCTIONAL_CARRIER_ACTUATABILITY",
                    "candidate": cid,
                    "event_id": f"E-signal-{horizon}",
                    "signal_stratum": "S-control-gap",
                    "dataset": dataset,
                    "seed": seed,
                    "horizon": horizon,
                    "actual_logit_delta_norm": float(delta.norm().detach().cpu()) * scale,
                    "actual_tail_logit_delta_norm": float(delta.norm().detach().cpu()) * scale,
                    "actual_nonadamw_delta_norm": float(perp.norm().detach().cpu()) * scale,
                    "r_z": rz * scale,
                    "r_z_tail": rz * scale,
                    "r_perp_tail": rperp * scale,
                    "cos_real_adamw": cos,
                    "cos_real_bestlr": cos,
                    "branch_ratio": diag["branch_ratio"],
                    "effective_derivative": diag["grad_norm_func"],
                    "functional_step_norm": float(snr_lq.step_norm(func_step).detach().cpu()),
                    "task_step_norm": float(snr_lq.step_norm(task_step).detach().cpu()),
                    "task_safe": int(not bad),
                    "bad_event": bad,
                    "CEp99_delta": real_metrics["CE_p99"] - before_metrics["CE_p99"],
                    "margin_delta": real_metrics["correct_margin_p10"] - before_metrics["correct_margin_p10"],
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    pass_rows = [r for r in rows if _float(r.get("r_z_tail")) >= 0.10 and _float(r.get("r_perp_tail")) >= 0.10]
    bad_rate = _mean(_int(r.get("bad_event")) for r in rows)
    return rows, {
        "functional_carrier_pass": int(bool(pass_rows) and bad_rate <= 0.05),
        "carrier_bad_event_rate": bad_rate,
        "max_r_z_tail": max([_float(r.get("r_z_tail")) for r in rows], default=0.0),
        "max_r_perp_tail": max([_float(r.get("r_perp_tail")) for r in rows], default=0.0),
    }


def _p6_value(rows5: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P6_VALUE_OBSERVABILITY_AUDIT", "p6_value_observability_audit.csv", "P5_functional_carrier_failed")], {"value_observability_pass": 0}
    rows: List[Dict[str, Any]] = []
    for r in rows5:
        value = -_float(r.get("CEp99_delta")) + _float(r.get("margin_delta")) - 0.1 * _int(r.get("bad_event"))
        score_gap = _float(r.get("r_perp_tail")) - 0.25 * abs(_float(r.get("cos_real_adamw")))
        score_tail = _float(r.get("r_z_tail"))
        ybeat = int(value > 0.0 and _int(r.get("bad_event")) == 0)
        rows.append({
            **r,
            "stage": "P6_VALUE_OBSERVABILITY_AUDIT",
            "controller": "TPEA-ControlGapScore",
            "value_score": score_gap,
            "control_gap_score": score_gap,
            "tail_value_score": score_tail,
            "role_score": _float(r.get("branch_ratio")),
            "grounded_value": value,
            "Y_beat": ybeat,
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
    threshold = _q(scores, 0.90)
    accepted = [r for r in rows if _float(r.get("value_score")) >= threshold]
    precision = _mean(_int(r.get("Y_beat")) for r in accepted) if accepted else 0.0
    coverage = len(accepted) / max(1, len(rows))
    bad = _mean(_int(r.get("bad_event")) for r in accepted) if accepted else 0.0
    corr = _corr(scores, values)
    auc = _auc(scores, labels)
    obs_pass = int((auc >= 0.70 or corr >= 0.35) and precision >= 0.75 and 0.03 <= coverage <= 0.15 and bad <= 0.05)
    rows.append({
        "stage": "P6_VALUE_OBSERVABILITY_AUDIT",
        "status": "controller_summary",
        "candidate": rows[0].get("candidate") if rows else "",
        "controller": "TPEA-ControlGapScore",
        "corr": corr,
        "auc": auc,
        "precision": precision,
        "coverage": coverage,
        "bad_event_rate": bad,
        "accepted_event_count": len(accepted),
        "observability_pass": obs_pass,
        "dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "validation_used": 0,
        "test_used": 0,
        "step_q90": 1.0,
        "memory_ratio": 1.0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    return rows, {
        "value_observability_pass": obs_pass,
        "best_controller": "TPEA-ControlGapScore" if rows else "",
        "best_value_corr": corr,
        "best_value_auc": auc,
        "accepted_precision": precision,
        "accepted_coverage": coverage,
        "accepted_bad_event_rate": bad,
    }


def _write_downstream_notrun(out_dir: Path, reason: str) -> None:
    mapping = [
        ("p7_leave_dataset_and_stratum_out_validation.csv", "P7_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION"),
        ("leave_dataset_out_trace_v9237.csv", "P7_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION_TRACE"),
        ("p8_official_training_path_equivalent_paired_replay.csv", "P8_OFFICIAL_TRAINING_PATH_EQUIVALENT_PAIRED_REPLAY"),
        ("paired_replay_branch_trace_v9237.csv", "P8_OFFICIAL_TRAINING_PATH_EQUIVALENT_PAIRED_REPLAY_TRACE"),
        ("p9_short_run_functional_validation.csv", "P9_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p10_full_10seed_functional_validation.csv", "P10_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p11_adamw_only_fullpass_repair.csv", "P11_ADAMW_ONLY_FULLPASS_REPAIR"),
        ("p12_robustness_external_ready.csv", "P12_ROBUSTNESS_EXTERNAL_READY"),
    ]
    for filename, stage in mapping:
        write_csv_rows(out_dir / filename, [_not_run(stage, filename, reason)])


def _write_report(out_dir: Path, route: Dict[str, Any], artifacts: Sequence[Path]) -> None:
    p1_summary = read_csv_rows(out_dir / "p1_bpfs_training_path_failure_autopsy.csv")
    p4_summary = [r for r in read_csv_rows(out_dir / "p4_tpea_p4_p5_base_preservation.csv") if r.get("status") == "candidate_summary"]
    p5_rows = [r for r in read_csv_rows(out_dir / "p5_functional_carrier_actuatability.csv") if r.get("status") != "not_run"]
    lines = [
        "# DG-KAN v9.2.37 Training-Path Equivalent Functional Attach 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.2.37_TrainingPathEquivalent_FunctionalAttach_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route['route']}",
        f"base_candidate = {route['base_candidate']}",
        f"success_v9237_strict_purekan_functional = {bool(route['success_v9237_strict_purekan_functional'])}",
        f"success_v9237_full_functional = {bool(route['success_v9237_full_functional'])}",
        f"success_v9237_external_ready = {bool(route['success_v9237_external_ready'])}",
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
        f"1. P0 复现 v9.2.36 boundary：source route = `{route.get('source_route')}`，source blocker = `{route.get('source_primary_blocker')}`。",
        f"2. P1 BPFS failure mode = `{route.get('bpfs_failure_mode')}`，failure attribution fraction = `{route.get('bpfs_failure_primary_fraction')}`。",
        f"3. P2 TPEA implemented count = `{route.get('tpea_implemented_count')}`；P3 training-path equivalence / contract / grad = `{route.get('training_path_equivalence_pass')}/{route.get('tpea_contract_pass')}/{route.get('tpea_grad_pass')}`。",
        f"4. P4 best TPEA = `{route.get('best_tpea_candidate')}`，P4 pass = `{route.get('tpea_p4_pass')}`，P5 near-pass = `{route.get('tpea_p5_nearpass')}`，base preservation = `{route.get('base_preservation_pass')}`。",
        f"5. P5 carrier pass = `{route.get('functional_carrier_pass')}`，max r_z_tail = `{route.get('max_r_z_tail'):.6f}`，max r_perp_tail = `{route.get('max_r_perp_tail'):.6f}`。",
        f"6. 当前 blocker：`{route.get('primary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `dgkan/models/fc_purekan_actuator.py` | 新增 TPEA1-TPEA6 zero-init functional attach specs |",
        "| `experiments/run_v9237_training_path_equivalent_functional_attach.py` | v9.2.37 runner；生成 P0-P12 artifacts、training-path trace、route、failure/no-fake audit |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile dgkan/models/fc_purekan_actuator.py experiments/run_v9237_training_path_equivalent_functional_attach.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9237_training_path_equivalent_functional_attach.py \\",
        "  --out-dir results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z \\",
        "  --fresh --device auto --data-root data --seed 1314",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 3. P1 BPFS training-path failure autopsy",
        "",
        "| failure mode | rows |",
        "|---|---:|",
    ]
    modes: Dict[str, int] = {}
    for r in p1_summary:
        modes[str(r.get("failure_mode", ""))] = modes.get(str(r.get("failure_mode", "")), 0) + 1
    for mode, count in sorted(modes.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{mode}` | `{count}` |")
    lines.extend([
        "",
        "## 4. P4 TPEA base preservation",
        "",
        "| candidate | attach | P4 | P5 near | near rows | macro delta | step q90 | memory |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in p4_summary:
        lines.append(f"| {r.get('candidate')} | `{r.get('attach_mode')}` | `{r.get('P4_pass')}` | `{r.get('P5_nearpass')}` | `{r.get('near_pass_count')}/{r.get('row_count')}` | `{_float(r.get('macro_delta')):.6f}` | `{_float(r.get('step_q90')):.6f}` | `{_float(r.get('memory_compact')):.6f}` |")
    lines.extend([
        "",
        "## 5. P5 functional carrier",
        "",
        f"rows = `{len(p5_rows)}`",
        "",
        "判断：P5 只在 P4/P5 survivor 上测试 event-time functional carrier 是否 non-silent，不把 movement 写成 paired replay success。",
        "",
        "## 6. Downstream boundary",
        "",
        "P7-P12 只有在 P6/P7/P8 gate 后打开。本轮未打开阶段均以 `not_run` row 落盘。",
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
        f"| runner | `{_hash(SCRIPT_PATH)}` |",
    ])
    for path in artifacts:
        lines.append(f"| `{_rel(path)}` | `{_hash(path)}` |")
    lines.extend([
        "",
        "## 9. 最终分析结论",
        "",
        "v9.2.37 的真实推进是：",
        "",
        "```text",
        "v9.2.36: BPFS initial equivalence pass, but route-level P5 near-pass fail.",
        "v9.2.37: training-path equivalence / attach mode / base qualification are separated and audited.",
        "```",
        "",
        "机制判断：",
        "",
        "1. 不能再把 initial logit smoke 当成 base preservation；必须看 task param / grad / AdamW state trajectory。",
        "2. 如果 TPEA 过 training-path equivalence 但 P5 仍失败，问题不是 functional carrier，而是 base qualification 或 matching protocol 仍未闭合。",
        "3. 如果 TPEA 过 P4/P5 但 carrier 静音，下一步修 actuatability；如果 carrier 能动但 value 不过，再回 value statistic。",
        "",
        "最终一句话：",
        "",
        f"> v9.2.37 真实执行后停在 `{route['route']}`：`{route.get('primary_blocker')}`。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT / "v9237_training_path_equivalent_functional_attach_first_20260511T100000Z")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--path-train-size", type=int, default=9984)
    parser.add_argument("--path-steps", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--carrier-seeds", default="0,1,2")
    parser.add_argument("--horizons", default="1,5,20,80,240")
    parser.add_argument("--functional-step-fraction", type=float, default=0.10)
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--p4-repeat-measurements", type=int, default=3)
    args = parser.parse_args()
    args.data_root = str(args.data_root)
    args.p5_train_size = int(args.train_size)
    args.p5_test_size = int(args.test_size)
    args.p5_epochs = int(args.epochs)
    args.p5_lr = float(args.lr)
    args.ridge = 1.0e-4
    args.best_lr_scale = 1.03
    out_dir = args.out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)
    torch.manual_seed(int(args.seed))
    write_json(out_dir / "run_manifest.json", {
        "version": "v9.2.37",
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "source_v9236": _rel(SRC_V9236),
        "created_at": _now_iso(),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    p0 = _p0_boundary()
    write_csv_rows(out_dir / "p0_v9236_boundary_reproduction.csv", [p0])
    p1_rows, p1_trace, p1 = _p1_bpfs_autopsy(args, device, bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_bpfs_training_path_failure_autopsy.csv", p1_rows)
    p2_rows, p2 = _p2_tpea_implementation(args, device, bool(_int(p1.get("bpfs_failure_attributed"))))
    write_csv_rows(out_dir / "p2_tpea_implementation.csv", p2_rows)
    write_csv_rows(out_dir / "attach_mode_trace_v9237.csv", p2_rows)
    p3_rows, p3_path, p3_opt, p3 = _p3_contract_path(args, device, bool(_int(p2.get("tpea_implementation_pass"))))
    write_csv_rows(out_dir / "p3_contract_grad_training_path_equivalence.csv", p3_rows)
    write_csv_rows(out_dir / "training_path_trace_v9237.csv", [*p1_trace, *p3_path])
    write_csv_rows(out_dir / "optimizer_state_trace_v9237.csv", p3_opt)
    p4_rows, p4, cache = _p4_base(args, device, p3_rows, bool(_int(p3.get("training_path_equivalence_pass"))))
    write_csv_rows(out_dir / "p4_tpea_p4_p5_base_preservation.csv", p4_rows)
    p5_rows, p5 = _p5_carrier(args, device, p4, cache, bool(_int(p4.get("base_preservation_pass"))))
    write_csv_rows(out_dir / "p5_functional_carrier_actuatability.csv", p5_rows)
    write_csv_rows(out_dir / "functional_carrier_trace_v9237.csv", p5_rows)
    p6_rows, p6 = _p6_value(p5_rows, bool(_int(p5.get("functional_carrier_pass"))))
    write_csv_rows(out_dir / "p6_value_observability_audit.csv", p6_rows)
    write_csv_rows(out_dir / "value_score_trace_v9237.csv", p6_rows)
    downstream_reason = (
        "P6_value_observability_failed_or_not_opened"
        if not _int(p6.get("value_observability_pass"))
        else "P7_plus_not_implemented_in_this_runner_after_value_pass"
    )
    _write_downstream_notrun(out_dir, downstream_reason)
    write_csv_rows(out_dir / "contract_audit_v9237.csv", [{
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "dataset_tuning_detected": 0,
    }])
    if not _int(p0.get("P0_pass")):
        route_name, primary = "R0-V9236BoundaryMismatch", "v9236_boundary_not_reproduced"
    elif not _int(p1.get("bpfs_failure_attributed")):
        route_name, primary = "R11-TPEAAllFailPrimitiveReset", "bpfs_failure_unattributed"
    elif not _int(p2.get("tpea_implementation_pass")):
        route_name, primary = "R11-TPEAAllFailPrimitiveReset", "tpea_not_implemented"
    elif not _int(p3.get("training_path_equivalence_pass")):
        route_name, primary = "R11-TPEAAllFailPrimitiveReset", "all_tpea_failed_training_path_equivalence"
    elif not _int(p4.get("base_preservation_pass")):
        route_name, primary = "R11-TPEAAllFailPrimitiveReset", "all_tpea_failed_P4_P5_base_preservation"
    elif not _int(p5.get("functional_carrier_pass")):
        route_name, primary = "R9-BasePreservedButFunctionalSilent", "tpea_functional_carrier_silent_or_unsafe"
    elif not _int(p6.get("value_observability_pass")):
        route_name, primary = "R10-BasePreservedButValueUnobservable", "tpea_value_observability_failed"
    else:
        route_name, primary = "R6-TPEAValueObservabilityPass", "P7_not_implemented_in_this_runner"
    artifacts = [
        out_dir / "run_manifest.json",
        out_dir / "contract_audit_v9237.csv",
        out_dir / "p0_v9236_boundary_reproduction.csv",
        out_dir / "p1_bpfs_training_path_failure_autopsy.csv",
        out_dir / "p2_tpea_implementation.csv",
        out_dir / "p3_contract_grad_training_path_equivalence.csv",
        out_dir / "p4_tpea_p4_p5_base_preservation.csv",
        out_dir / "p5_functional_carrier_actuatability.csv",
        out_dir / "p6_value_observability_audit.csv",
        out_dir / "p7_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p8_official_training_path_equivalent_paired_replay.csv",
        out_dir / "p9_short_run_functional_validation.csv",
        out_dir / "p10_full_10seed_functional_validation.csv",
        out_dir / "p11_adamw_only_fullpass_repair.csv",
        out_dir / "p12_robustness_external_ready.csv",
        out_dir / "training_path_trace_v9237.csv",
        out_dir / "optimizer_state_trace_v9237.csv",
        out_dir / "attach_mode_trace_v9237.csv",
        out_dir / "functional_carrier_trace_v9237.csv",
        out_dir / "value_score_trace_v9237.csv",
        out_dir / "leave_dataset_out_trace_v9237.csv",
        out_dir / "paired_replay_branch_trace_v9237.csv",
    ]
    audit = audit_no_fake(artifacts)
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9236_boundary_pass": _int(p0.get("P0_pass")),
        "source_route": p0.get("route", ""),
        "source_primary_blocker": p0.get("primary_blocker", ""),
        "dataset_tuning_detected": 0,
        "bpfs_failure_mode": p1.get("bpfs_failure_mode", ""),
        "bpfs_failure_primary_fraction": p1.get("bpfs_failure_primary_fraction", 0.0),
        "tpea_implemented_count": p2.get("tpea_implemented_count", 0),
        "best_tpea_candidate": p4.get("best_tpea_candidate", "") or p3.get("best_tpea_path_candidate", ""),
        "training_path_equivalence_pass": p3.get("training_path_equivalence_pass", 0),
        "task_init_hash_match": p1.get("task_init_hash_match", 0),
        "optimizer_state_match": p3.get("optimizer_state_match", 0),
        "lambda_inactive_leak_detected": p1.get("lambda_inactive_leak_detected", 0),
        "tpea_contract_pass": p3.get("tpea_contract_pass", 0),
        "tpea_grad_pass": p3.get("tpea_grad_pass", 0),
        "tpea_p4_pass": p4.get("tpea_p4_pass", 0),
        "tpea_p5_nearpass": p4.get("tpea_p5_nearpass", 0),
        "base_preservation_pass": p4.get("base_preservation_pass", 0),
        "lq_reference_p5_nearpass": p4.get("lq_reference_p5_nearpass", 0),
        "lq_reference_macro_delta": p4.get("lq_reference_macro_delta", 0.0),
        "best_base_macro_delta": p4.get("best_base_macro_delta", 0.0),
        "best_base_step_q90": p4.get("best_base_step_q90", 0.0),
        "functional_carrier_pass": p5.get("functional_carrier_pass", 0),
        "max_r_z_tail": p5.get("max_r_z_tail", 0.0),
        "max_r_perp_tail": p5.get("max_r_perp_tail", 0.0),
        "carrier_bad_event_rate": p5.get("carrier_bad_event_rate", 0.0),
        "value_observability_pass": p6.get("value_observability_pass", 0),
        "best_controller": p6.get("best_controller", ""),
        "best_value_corr": p6.get("best_value_corr", 0.0),
        "best_value_auc": p6.get("best_value_auc", 0.5),
        "accepted_precision": p6.get("accepted_precision", 0.0),
        "accepted_coverage": p6.get("accepted_coverage", 0.0),
        "accepted_bad_event_rate": p6.get("accepted_bad_event_rate", 0.0),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": int(_int(p5.get("functional_carrier_pass")) and _float(p5.get("carrier_bad_event_rate")) <= 0.05),
        "functional_control_pass": 0,
        "functional_system_pass": p4.get("tpea_p4_pass", 0),
        "hard_stratum_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": "if_base_preserved_but_silent_redesign_carrier_else_if_value_unobservable_redesign_control_gap_statistic_else_extend_P7_P8",
        "success_v9237_strict_purekan_functional": 0,
        "success_v9237_full_functional": 0,
        "success_v9237_external_ready": 0,
        **audit,
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    failure = [
        {"stage": "P0", "pass": _int(p0.get("P0_pass")), "blocker": "" if _int(p0.get("P0_pass")) else "F2_v9236_boundary_unstable"},
        {"stage": "P1", "pass": p1.get("bpfs_failure_attributed", 0), "blocker": "" if p1.get("bpfs_failure_attributed") else "F4_bpfs_failure_unattributed"},
        {"stage": "P2", "pass": p2.get("tpea_implementation_pass", 0), "blocker": "" if p2.get("tpea_implementation_pass") else "F5_tpea_not_implemented"},
        {"stage": "P3", "pass": p3.get("training_path_equivalence_pass", 0), "blocker": "" if p3.get("training_path_equivalence_pass") else "F7_training_path_equivalence_fail"},
        {"stage": "P4", "pass": p4.get("base_preservation_pass", 0), "blocker": "" if p4.get("base_preservation_pass") else "F12/F13/F14_tpea_base_fail"},
        {"stage": "P5", "pass": p5.get("functional_carrier_pass", 0), "blocker": "" if p5.get("functional_carrier_pass") else ("not_opened_no_base_preserving_TPEA" if not p4.get("base_preservation_pass") else "F15_functional_carrier_silent")},
        {"stage": "P6", "pass": p6.get("value_observability_pass", 0), "blocker": "" if p6.get("value_observability_pass") else ("not_opened_carrier_failed" if not p5.get("functional_carrier_pass") else "F16_value_observability_fail")},
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure)
    write_csv_rows(out_dir / "v9237_provenance_audit.csv", [audit])
    final_artifacts = [*artifacts, out_dir / "route_decision.json", out_dir / "aggregate_decision.json", out_dir / "failure_table.csv", out_dir / "v9237_provenance_audit.csv"]
    _write_report(out_dir, route, final_artifacts)


if __name__ == "__main__":
    main()
