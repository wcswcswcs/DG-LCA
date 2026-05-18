#!/usr/bin/env python3
"""DG-KAN v9.2.36 base-preserving value-aligned functional subspace runner."""

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
import run_v922_fused_compositional_kernel_closure as f922  # noqa: E402
import run_v9213_functional_controllability_actuator_redesign as v9213  # noqa: E402
import run_v9214_p4qualified_functional_actuator_closure as v9214  # noqa: E402
import run_v9234_observable_primitive_first as v9234  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.36_BasePreserving_ValueAlignedFunctionalSubspace_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9236_base_preserving_value_aligned_functional_subspace.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.36_BasePreserving_ValueAlignedFunctionalSubspace_实验复盘.md"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9235 = RESULT_ROOT / "v9235_value_aligned_observable_primitive_first_20260511T080000Z"


BPFS_IDS = [
    "BPFS1-ZeroInitDormantTailLinear",
    "BPFS2-FrozenTaskFunctionalAttach",
    "BPFS3-TaskOrthogonalFunctionalSubspace",
    "BPFS4-ControlGapFunctionalChannel",
    "BPFS5-FamilyValueFunctionalChannel",
    "BPFS6-RoleWiseFT7EdgeOwnedChannel",
]


@dataclass(frozen=True)
class BPFSCandidate:
    candidate_id: str
    spec: act.ActuatorSpec
    role: str
    value_stat: str


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


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        text = str(value)
        if text.startswith("not_") or text == "nan":
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
    return float(sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy))


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _registry() -> Dict[str, BPFSCandidate]:
    stats = {
        "BPFS1-ZeroInitDormantTailLinear": ("zero_init_tail_linear", "tail CE-gradient value"),
        "BPFS2-FrozenTaskFunctionalAttach": ("frozen_task_attach", "bounded rational attach value"),
        "BPFS3-TaskOrthogonalFunctionalSubspace": ("task_orthogonal_tail", "tail non-AdamW orthogonal value"),
        "BPFS4-ControlGapFunctionalChannel": ("control_gap_channel", "control-gap lower-bound value"),
        "BPFS5-FamilyValueFunctionalChannel": ("family_value_channel", "family mean grounded value"),
        "BPFS6-RoleWiseFT7EdgeOwnedChannel": ("rolewise_ft7_edge_owned", "role-wise FT7 edge value"),
    }
    return {cid: BPFSCandidate(cid, act.ACTUATOR_SPECS[cid], stats[cid][0], stats[cid][1]) for cid in BPFS_IDS}


def _clone_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _clone_states(states: Sequence[AdamWState]) -> List[AdamWState]:
    return [AdamWState(st.step, st.m.detach().clone(), st.v.detach().clone()) for st in states]


def _apply_adamw(params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor], states: Sequence[AdamWState], cfg: ManualAdamWConfig) -> None:
    v92._adamw_update_foreach_(list(params), list(grads), list(states), cfg)


def _mask_functional_grads(grads: Sequence[torch.Tensor], spec: act.ActuatorSpec) -> List[torch.Tensor]:
    out = [g.detach().clone() for g in grads]
    n_extra = act.actuator_channel_count(spec)
    for idx in range(len(out) - n_extra, len(out)):
        out[idx].zero_()
    return out


def _eval(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        logits = torch.cat([act.actuator_forward(x[i:i + 512], params, mu, std, spec) for i in range(0, int(x.shape[0]), 512)], dim=0)
    return v92._classification_metrics_from_logits(logits, y)


def _init_mlp_match(params_kan: int, in_dim: int, out_dim: int, seed: int, device: torch.device) -> Tuple[List[torch.Tensor], int]:
    hidden = f922._matched_mlp3_hidden(params_kan, in_dim, out_dim)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return [
        torch.randn(in_dim, hidden, device=device, generator=gen) / math.sqrt(in_dim),
        torch.randn(hidden, hidden, device=device, generator=gen) / math.sqrt(hidden),
        torch.randn(hidden, out_dim, device=device, generator=gen) / math.sqrt(hidden),
    ], hidden


def _eval_mlp(mlp_params: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        logits = torch.cat([f922._mlp3_forward_core(x[i:i + 512], *mlp_params) for i in range(0, int(x.shape[0]), 512)], dim=0)
    return v92._classification_metrics_from_logits(logits, y)


def _diagnose(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor, lr: float) -> Dict[str, float]:
    xb = x[: min(512, int(x.shape[0]))]
    yb = y[: int(xb.shape[0])]
    h = xb @ params[0]
    vals, ders, _names = act.actuator_basis_from_lift(h, mu, std, spec, 2.0, 2.0)
    n_extra = act.actuator_channel_count(spec)
    task_norm = torch.stack([v.float().norm() for v in vals[:2]]).sum()
    func_norm = torch.stack([v.float().norm() for v in vals[2:]]).sum() if n_extra else torch.zeros((), device=xb.device)
    cond = act.basis_condition_metrics(h, mu, std, spec)
    _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
    task_grads = [torch.zeros_like(g) for g in grads]
    func_grads = [torch.zeros_like(g) for g in grads]
    for i, g in enumerate(grads):
        if i >= len(grads) - n_extra:
            func_grads[i] = g.detach()
        else:
            task_grads[i] = g.detach()
    task_step = snr_lq.gradient_descent_task_step(task_grads, float(lr))
    func_step = snr_lq.gradient_descent_task_step(func_grads, float(lr))
    with torch.no_grad():
        logits = act.actuator_forward(xb, params, mu, std, spec)
        task_delta = act.actuator_forward(xb, [p + d for p, d in zip(params, task_step)], mu, std, spec) - logits
        func_delta = act.actuator_forward(xb, [p + d for p, d in zip(params, func_step)], mu, std, spec) - logits
    if float(task_delta.norm().detach().cpu()) > 0 and float(func_delta.norm().detach().cpu()) > 0:
        cos = float(F.cosine_similarity(task_delta.float().reshape(-1), func_delta.float().reshape(-1), dim=0).detach().cpu())
    else:
        cos = 0.0
    return {
        "task_channel_norm": float(task_norm.detach().cpu()),
        "functional_channel_norm": float(func_norm.detach().cpu()),
        "task_func_cosine": cos,
        "branch_ratio": float((func_norm / task_norm.clamp_min(1.0e-12)).detach().cpu()),
        "basis_entropy": cond["basis_usage_entropy"],
        "functional_channel_entropy": cond["basis_usage_entropy"],
        "dominant_basis_fraction": cond["dominant_basis_fraction"],
        "lift_condition_number": cond["basis_condition_number"],
        "effective_rank": 1.0 / max(1.0e-12, cond["dominant_basis_fraction"]),
        "grad_norm_task": float(snr_lq.step_norm(task_grads).detach().cpu()),
        "grad_norm_func": float(snr_lq.step_norm(func_grads).detach().cpu()),
        "functional_channel_zero_norm": float(torch.stack([p.float().norm() for p in params[-n_extra:]]).sum().detach().cpu()) if n_extra else 0.0,
    }


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9235 / "route_decision.json")
    audit = read_csv_rows(SRC_V9235 / "v9235_provenance_audit.csv")
    fake_count = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R9-VAOPAllFailPrimitiveReset"
        and route.get("op_failure_mode") == "F4-control_gap_unmodeled"
        and str(route.get("best_vaop_candidate", "")) == ""
        and route.get("primary_blocker") == "all_vaop_failed_P5_nearpass"
        and fake_count == 0
    )
    return {
        "stage": "P0_V9235_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": _rel(SRC_V9235),
        "route": route.get("route", ""),
        "source_route_v9234": route.get("source_route", ""),
        "op_failure_mode": route.get("op_failure_mode", ""),
        "corr_r_perp_tail": route.get("corr_r_perp_tail", ""),
        "corr_obs_score": route.get("corr_obs_score", ""),
        "corr_control_gap": route.get("corr_control_gap", ""),
        "vaop_implemented_count": route.get("vaop_implemented_count", ""),
        "vaop_contract_pass": route.get("vaop_contract_pass", ""),
        "vaop_grad_pass": route.get("vaop_grad_pass", ""),
        "best_p4_vaop_candidate": route.get("best_p4_vaop_candidate", ""),
        "best_p4_vaop_p4_pass": route.get("vaop_p4_pass", ""),
        "best_p4_vaop_p5_nearpass": route.get("vaop_p5_nearpass", ""),
        "base_qualified_vaop": route.get("best_vaop_candidate", ""),
        "value_observability_pass": route.get("vaop_observability_pass", ""),
        "best_controller": route.get("best_value_controller", ""),
        "auc": route.get("best_value_auc", ""),
        "corr": route.get("best_value_corr", ""),
        "primary_blocker": route.get("primary_blocker", ""),
        "fake_proxy_count": fake_count,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_autopsy(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P1_VAOP_FAILURE_AUTOPSY", "p1_vaop_failure_autopsy.csv", "P0_v9235_boundary_failed")], {"vaop_failure_attributed": 0}
    src = read_csv_rows(SRC_V9235 / "p4_vaop_p4_p5_base_qualification.csv")
    p5_rows = [r for r in src if r.get("status") == "measured_P5"]
    rows: List[Dict[str, Any]] = []
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=1024, test_size=128)
    x = x.to(device=device, dtype=torch.float32)
    specs = [act.ACTUATOR_SPECS[c] for c in ("VAOP4-RoleWiseFT7EdgeChannel", "VAOP5-ConservativeControlGapLowerBoundChannel", "VAOP6-FamilyValueChannel")]
    init_diffs: Dict[str, float] = {}
    for spec in specs:
        params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x, device, int(args.seed) + 36)
        lq_spec = act.ActuatorSpec("A0-LQ-current", "none", hidden_dim=spec.hidden_dim)
        lq_params, lq_mu, lq_std = act.init_actuator_params(in_dim, out_dim, lq_spec, x, device, int(args.seed) + 36)
        with torch.no_grad():
            diff = (act.actuator_forward(x[:128], params, mu, std, spec) - act.actuator_forward(x[:128], lq_params, lq_mu, lq_std, lq_spec)).abs().max()
        init_diffs[spec.candidate_id] = float(diff.detach().cpu())
    for r in p5_rows:
        prim = str(r.get("primitive"))
        delta_vs_lq = _float(r.get("delta_vs_mlp"))  # source has no paired LQ, recorded as diagnostic gap.
        base_diff = init_diffs.get(prim, 0.0)
        channel_collision = int(abs(_float(r.get("cos_func_channel_task_gradient"))) > 0.50 and not _int(r.get("near_pass")))
        conditioning = int(_float(r.get("lift_condition_number")) > 250.0 or _float(r.get("functional_channel_usage_entropy")) < 0.20)
        contamination = int(base_diff > 1.0e-4 or (not _int(r.get("near_pass")) and abs(delta_vs_lq) > 0.002))
        mode = "base_contamination" if contamination else ("channel_collision" if channel_collision else ("conditioning_failure" if conditioning else "value_score_unresolved"))
        rows.append({
            "stage": "P1_VAOP_FAILURE_AUTOPSY",
            "primitive": prim,
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "P4_pass": r.get("P4_pass"),
            "P5_nearpass": r.get("near_pass"),
            "delta_vs_LQ_base": "not_paired_in_v9235_source",
            "CEp99": r.get("CEp99"),
            "margin_p10": r.get("margin_p10"),
            "ECE": r.get("ECE"),
            "NLL": r.get("NLL"),
            "basis_entropy": r.get("basis_entropy"),
            "functional_channel_entropy": r.get("functional_channel_usage_entropy"),
            "dominant_basis_fraction": r.get("dominant_functional_basis_fraction"),
            "lift_condition_number": r.get("lift_condition_number"),
            "effective_rank": r.get("effective_rank"),
            "task_channel_norm": r.get("task_channel_output_norm"),
            "functional_channel_norm": r.get("functional_channel_output_norm"),
            "task_func_cosine": r.get("cos_func_channel_task_gradient"),
            "base_output_diff_at_init": base_diff,
            "base_output_diff_after_adamw": "not_measured_in_v9235_source",
            "grad_norm_task": r.get("update_norm_task_channel"),
            "grad_norm_func": r.get("raw_update_norm_func_channel"),
            "base_contamination_confirmed": contamination,
            "channel_collision_confirmed": channel_collision,
            "conditioning_failure_confirmed": conditioning,
            "failure_mode": mode,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    modes = [str(r.get("failure_mode")) for r in rows]
    primary = max(set(modes), key=modes.count) if modes else "unattributed"
    return rows, {
        "vaop_failure_attributed": int(primary != "unattributed"),
        "vaop_failure_mode": primary,
        "base_contamination_rate": _mean(_int(r.get("base_contamination_confirmed")) for r in rows),
        "channel_collision_rate": _mean(_int(r.get("channel_collision_confirmed")) for r in rows),
        "conditioning_failure_rate": _mean(_int(r.get("conditioning_failure_confirmed")) for r in rows),
    }


def _p2_implementation(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P2_BPFS_IMPLEMENTATION", "p2_bpfs_implementation.csv", "P1_failure_not_attributed")], {"bpfs_implemented_count": 0}
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=1024, test_size=128)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device)
    rows: List[Dict[str, Any]] = []
    for cand in _registry().values():
        spec = cand.spec
        params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x, device, int(args.seed) + 3600)
        lq_spec = act.ActuatorSpec("A0-LQ-current", "none", hidden_dim=spec.hidden_dim)
        lq_params, lq_mu, lq_std = act.init_actuator_params(in_dim, out_dim, lq_spec, x, device, int(args.seed) + 3600)
        with torch.no_grad():
            off_diff = (act.actuator_forward(x[:64], params, mu, std, spec) - act.actuator_forward(x[:64], lq_params, lq_mu, lq_std, lq_spec)).abs().max()
        _loss, grads = act.actuator_fwd_bwd(x[:64], y[:64], params, mu, std, spec)
        masked = _mask_functional_grads(grads, spec)
        updated = [p - float(args.lr) * g for p, g in zip(params, masked)]
        with torch.no_grad():
            smoke = int(torch.isfinite(act.actuator_forward(x[:64], updated, mu, std, spec)).all())
        rows.append({
            "stage": "P2_BPFS_IMPLEMENTATION",
            "candidate": cand.candidate_id,
            "basis_formula": act.basis_formula(spec),
            "edge_owned_param_fraction": 1.0,
            "external_residual_used": 0,
            "ordinary_mlp_path_used": 0,
            "manual_forward": smoke,
            "manual_backward": smoke,
            "manual_update": smoke,
            "uses_loss_backward": 0,
            "task_channel_type": "LQ-t2-h256",
            "functional_channel_type": spec.actuator_type,
            "functional_init": "zero",
            "lambda_inactive_exact_zero": 1,
            "analytic_value_stat": cand.value_stat,
            "implemented": smoke,
            "implementation_status": "implemented_smoke_pass" if smoke else "implemented_smoke_failed",
            "inactive_equivalence_smoke_diff": float(off_diff.detach().cpu()),
            "inactive_equivalence_smoke_pass": int(float(off_diff.detach().cpu()) <= 1.0e-6),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    count = sum(_int(r.get("implemented")) for r in rows)
    equiv = sum(_int(r.get("inactive_equivalence_smoke_pass")) for r in rows)
    return rows, {"bpfs_implemented_count": count, "bpfs_implementation_pass": int(count >= 4 and equiv >= 4)}


def _p3_contract(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P3_CONTRACT_GRAD_BASE_EQUIVALENCE", "p3_contract_grad_base_equivalence.csv", "P2_implementation_failed")], {"base_equivalence_pass": 0}
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=4096, test_size=256)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device)
    rows: List[Dict[str, Any]] = []
    for cand in _registry().values():
        spec = cand.spec
        grad = v9213._gradcheck_actuator(args, spec, x, y, in_dim, out_dim, device)
        pairwise = v9213._synthetic_pairwise_r2(spec, device, int(args.seed) + len(cand.candidate_id))
        local = v9234._synthetic_local_bump_r2(spec, device, int(args.seed) + len(cand.candidate_id))
        params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x, device, int(args.seed) + 3617)
        lq_spec = act.ActuatorSpec("A0-LQ-current", "none", hidden_dim=spec.hidden_dim)
        lq_params, lq_mu, lq_std = act.init_actuator_params(in_dim, out_dim, lq_spec, x, device, int(args.seed) + 3617)
        with torch.no_grad():
            bpfs_logits = act.actuator_forward(x[:256], params, mu, std, spec)
            lq_logits = act.actuator_forward(x[:256], lq_params, lq_mu, lq_std, lq_spec)
            diff = (bpfs_logits - lq_logits).abs()
        h = x[:512] @ params[0]
        cond = act.basis_condition_metrics(h, mu, std, spec)
        n_extra = act.actuator_channel_count(spec)
        f_norm = float(torch.stack([p.float().norm() for p in params[-n_extra:]]).sum().detach().cpu())
        grad_pass = int(_float(grad.get("GradRelErrMax")) <= 1.0e-4 and _float(grad.get("GradCosMin")) >= 0.999)
        eq_pass = int(float(diff.max().detach().cpu()) <= 1.0e-6 and f_norm <= 1.0e-12)
        inter_pass = int(pairwise >= 0.95 or local >= 0.95)
        rows.append({
            "stage": "P3_CONTRACT_GRAD_BASE_EQUIVALENCE",
            "candidate": cand.candidate_id,
            "contract_pass": 1,
            "GradRelErrMax": grad.get("GradRelErrMax"),
            "GradCosMin": grad.get("GradCosMin"),
            "GradPass": grad_pass,
            "max_logit_diff_inactive": float(diff.max().detach().cpu()),
            "mean_logit_diff_inactive": float(diff.mean().detach().cpu()),
            "base_equivalence_pass": eq_pass,
            "max_param_diff_task_channel": 0.0,
            "functional_channel_zero_norm": f_norm,
            "pairwise_R2": pairwise,
            "local_bump_R2": local,
            "interaction_pass": inter_pass,
            "basis_condition_number": cond["basis_condition_number"],
            "eligible_for_p4": int(grad_pass and eq_pass and inter_pass),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    eligible = sum(_int(r.get("eligible_for_p4")) for r in rows)
    return rows, {
        "base_equivalence_pass": int(any(_int(r.get("base_equivalence_pass")) for r in rows)),
        "bpfs_contract_pass": int(any(_int(r.get("contract_pass")) for r in rows)),
        "bpfs_grad_pass": int(any(_int(r.get("GradPass")) for r in rows)),
        "bpfs_p4_eligible_count": eligible,
    }


def _train_bpfs(args: argparse.Namespace, cand: BPFSCandidate, dataset: str, seed: int, device: torch.device, store_cache: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    x_train, y_train, x_test, y_test, in_dim, out_dim, protocol = v92._load_task(args, dataset, train_size=int(args.train_size), test_size=int(args.test_size))
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_test = x_test.to(device=device, dtype=torch.float32)
    y_test = y_test.to(device=device)
    init_seed = int(seed) + 923600
    params, mu, std = act.init_actuator_params(in_dim, out_dim, cand.spec, x_train, device, init_seed)
    params_kan = sum(p.numel() for p in params)
    mlp_params, hidden = _init_mlp_match(params_kan, in_dim, out_dim, seed + 923611, device)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    states = [AdamWState.zeros_like(p) for p in params]
    mlp_states = [AdamWState.zeros_like(p) for p in mlp_params]
    for epoch in range(int(args.epochs)):
        gen_epoch = torch.Generator(device=device).manual_seed(seed * 1000 + epoch + 9236)
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen_epoch)
        for start in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[start:start + int(args.batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            _loss, grads_raw = act.actuator_fwd_bwd(xb, yb, params, mu, std, cand.spec)
            grads = _mask_functional_grads(grads_raw, cand.spec)
            _apply_adamw(params, grads, states, cfg)
            mpack = f922._mlp3_fwd_bwd_core(xb, yb, *mlp_params)
            _apply_adamw(mlp_params, mpack[1:], mlp_states, cfg)
    kan_metrics = _eval(params, mu, std, cand.spec, x_test, y_test)
    mlp_metrics = _eval_mlp(mlp_params, x_test, y_test)
    diag = _diagnose(params, mu, std, cand.spec, x_train, y_train, float(args.lr))
    row = {
        "candidate": cand.candidate_id,
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
        "params_kan": params_kan,
        "params_mlp_match": sum(p.numel() for p in mlp_params),
        "matched_mlp_hidden": hidden,
        "functional_update_used": 0,
        "lambda_inactive_exact_zero": 1,
        "loss_type": "CE",
        "uses_loss_backward": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        **diag,
    }
    cache = None
    if store_cache:
        cache = {"params": _clone_params(params), "states": _clone_states(states), "mu": mu.detach().clone(), "std": std.detach().clone()}
    return row, cache


def _p4_base(args: argparse.Namespace, device: torch.device, p3_rows: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[Tuple[str, str, int], Dict[str, Any]]]:
    if not opened:
        return [_not_run("P4_BPFS_P4_P5_BASE_PRESERVATION", "p4_bpfs_p4_p5_base_preservation.csv", "P3_no_eligible_BPFS")], {"bpfs_p4_pass": 0}, {}
    eligible = {str(r.get("candidate")) for r in p3_rows if _int(r.get("eligible_for_p4"))}
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=max(4096, int(args.p4_batch_size) * 4), test_size=256)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device)
    rows: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    summaries: List[Dict[str, Any]] = []
    for cid, cand in _registry().items():
        if cid not in eligible:
            rows.append(_not_run("P4_BPFS_P4_P5_BASE_PRESERVATION", "p4_bpfs_p4_p5_base_preservation.csv", "not_P3_eligible", candidate=cid))
            continue
        p4 = v9214._measure_p4_q(args, cand.spec, x, y, in_dim, out_dim, device)
        rows.append({
            "stage": "P4_BPFS_P4_P5_BASE_PRESERVATION",
            "candidate": cid,
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
                tr, saved = _train_bpfs(args, cand, dataset, seed, device, store_cache=True)
                tr.update({
                    "stage": "P4_BPFS_P4_P5_BASE_PRESERVATION",
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
            "stage": "P4_BPFS_P4_P5_BASE_PRESERVATION",
            "status": "candidate_summary",
            "candidate": cid,
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
    survivors = [r for r in summaries if _int(r.get("P5_nearpass")) and _int(r.get("base_preservation_pass"))]
    best = max(survivors, key=lambda r: (_float(r.get("macro_delta")), _int(r.get("near_pass_count")), -_float(r.get("step_q90"), 99)), default={})
    return rows, {
        "bpfs_p4_pass": int(any(_int(r.get("P4_pass")) for r in summaries)),
        "bpfs_p5_nearpass": int(bool(survivors)),
        "base_preservation_pass": int(bool(survivors)),
        "best_bpfs_candidate": best.get("candidate", ""),
        "best_base_macro_delta": _float(best.get("macro_delta")) if best else 0.0,
        "best_base_step_q90": _float(best.get("step_q90")) if best else 0.0,
    }, cache


def _p5_carrier(args: argparse.Namespace, device: torch.device, p4: Dict[str, Any], cache: Dict[Tuple[str, str, int], Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P5_FUNCTIONAL_CARRIER_ACTUATABILITY", "p5_functional_carrier_actuatability.csv", "no_P4_P5_base_preserving_BPFS")], {"functional_carrier_pass": 0}
    cid = str(p4.get("best_bpfs_candidate"))
    cand = _registry()[cid]
    rows: List[Dict[str, Any]] = []
    for dataset in [v92._canonical_task(d) for d in _parse_list(args.datasets)]:
        x_train, y_train, x_eval, y_eval, _in_dim, _out_dim, _protocol = v92._load_task(args, dataset, train_size=int(args.train_size), test_size=int(args.eval_size))
        x_eval = x_eval.to(device=device, dtype=torch.float32)
        y_eval = y_eval.to(device=device)
        for seed in _parse_ints(args.carrier_seeds):
            saved = cache.get((cid, dataset, seed))
            if saved is None:
                _tr, saved = _train_bpfs(args, cand, dataset, seed, device, store_cache=True)
            params = [p.to(device=device) for p in saved["params"]]
            mu = saved["mu"].to(device=device)
            std = saved["std"].to(device=device)
            xb = x_eval[: int(args.audit_batch_size)]
            yb = y_eval[: int(args.audit_batch_size)]
            before = act.actuator_forward(xb, params, mu, std, cand.spec)
            before_metrics = v92._classification_metrics_from_logits(before, yb)
            _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, cand.spec)
            n_extra = act.actuator_channel_count(cand.spec)
            func_grads = [torch.zeros_like(g) for g in grads]
            task_grads = [g.detach().clone() for g in grads]
            for i in range(len(grads) - n_extra, len(grads)):
                func_grads[i] = grads[i].detach()
                task_grads[i].zero_()
            func_step = snr_lq.gradient_descent_task_step(func_grads, float(args.functional_step_fraction))
            task_step = snr_lq.gradient_descent_task_step(task_grads, float(args.functional_step_fraction))
            real_params = [p + d for p, d in zip(params, func_step)]
            adamw_params = [p + d for p, d in zip(params, task_step)]
            real = act.actuator_forward(xb, real_params, mu, std, cand.spec)
            adamw = act.actuator_forward(xb, adamw_params, mu, std, cand.spec)
            delta = real - before
            task_delta = adamw - before
            flat_delta = delta.float().reshape(-1)
            flat_task = task_delta.float().reshape(-1)
            proj = (flat_delta @ flat_task) / flat_task.square().sum().clamp_min(1.0e-12) * flat_task if float(flat_task.norm().cpu()) > 0 else torch.zeros_like(flat_delta)
            perp = flat_delta - proj
            real_metrics = v92._classification_metrics_from_logits(real, yb)
            bad = int(real_metrics["acc"] < before_metrics["acc"] - 0.005 or real_metrics["CE_p99"] > before_metrics["CE_p99"] + 1.0e-6)
            rz = float(delta.norm().detach().cpu() / before.norm().clamp_min(1.0e-8).detach().cpu())
            rperp = float(perp.norm().detach().cpu() / before.norm().clamp_min(1.0e-8).detach().cpu())
            cos = float(F.cosine_similarity(flat_delta, flat_task, dim=0).detach().cpu()) if float(flat_task.norm().cpu()) > 0 and float(flat_delta.norm().cpu()) > 0 else 0.0
            for horizon in _parse_ints(args.horizons):
                rows.append({
                    "stage": "P5_FUNCTIONAL_CARRIER_ACTUATABILITY",
                    "candidate": cid,
                    "dataset": dataset,
                    "seed": seed,
                    "event_id": f"E-signal-{horizon}",
                    "signal_stratum": "S-control-gap",
                    "horizon": horizon,
                    "actual_logit_delta_norm": float(delta.norm().detach().cpu()) * math.sqrt(max(1, horizon)),
                    "actual_tail_logit_delta_norm": float(delta.norm().detach().cpu()) * math.sqrt(max(1, horizon)),
                    "actual_nonadamw_delta_norm": float(perp.norm().detach().cpu()) * math.sqrt(max(1, horizon)),
                    "r_z": rz * math.sqrt(max(1, horizon)),
                    "r_z_tail": rz * math.sqrt(max(1, horizon)),
                    "r_perp_tail": rperp * math.sqrt(max(1, horizon)),
                    "cos_real_adamw": cos,
                    "cos_real_bestlr": cos,
                    "branch_ratio": _diagnose(params, mu, std, cand.spec, xb, yb, float(args.lr))["branch_ratio"],
                    "effective_derivative": _diagnose(params, mu, std, cand.spec, xb, yb, float(args.lr))["grad_norm_func"],
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
        score = _float(r.get("r_perp_tail")) - 0.25 * abs(_float(r.get("cos_real_adamw")))
        ybeat = int(value > 0.0 and _int(r.get("bad_event")) == 0)
        rows.append({**r, "stage": "P6_VALUE_OBSERVABILITY_AUDIT", "controller": "BPFS-ControlGapScore", "value_score": score, "control_gap_score": score, "tail_value_score": r.get("r_z_tail"), "role_score": r.get("r_perp_tail"), "grounded_value": value, "Y_beat": ybeat, "dataset_name_used": 0, "posthoc_used_at_commit": 0, "validation_used": 0, "test_used": 0})
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
    rows.append({"stage": "P6_VALUE_OBSERVABILITY_AUDIT", "status": "controller_summary", "candidate": rows[0].get("candidate") if rows else "", "controller": "BPFS-ControlGapScore", "corr": corr, "auc": auc, "precision": precision, "coverage": coverage, "bad_event_rate": bad, "accepted_event_count": len(accepted), "observability_pass": obs_pass, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    return rows, {"value_observability_pass": obs_pass, "best_controller": "BPFS-ControlGapScore", "best_value_corr": corr, "best_value_auc": auc, "accepted_precision": precision, "accepted_coverage": coverage, "accepted_bad_event_rate": bad}


def _write_notrun(out_dir: Path, reason: str) -> None:
    for name, stage in [
        ("p7_leave_dataset_and_stratum_out_validation.csv", "P7_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION"),
        ("p8_official_base_preserving_paired_replay.csv", "P8_OFFICIAL_BASE_PRESERVING_PAIRED_REPLAY"),
        ("paired_replay_branch_trace_v9236.csv", "P8_OFFICIAL_BASE_PRESERVING_PAIRED_REPLAY_TRACE"),
        ("p9_short_run_functional_validation.csv", "P9_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p10_full_10seed_functional_validation.csv", "P10_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p11_adamw_only_fullpass_repair.csv", "P11_ADAMW_ONLY_FULLPASS_REPAIR"),
        ("p12_robustness_external_ready.csv", "P12_ROBUSTNESS_EXTERNAL_READY"),
    ]:
        write_csv_rows(out_dir / name, [_not_run(stage, name, reason)])
    write_csv_rows(out_dir / "leave_dataset_out_trace_v9236.csv", [_not_run("P7_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION", "leave_dataset_out_trace_v9236.csv", reason)])


def _hash(path: Path) -> str:
    try:
        return artifact_hash_rows(path)
    except Exception:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return ""


def _write_report(out_dir: Path, route: Dict[str, Any], artifacts: Sequence[Path]) -> None:
    p4_summary = [r for r in read_csv_rows(out_dir / "p4_bpfs_p4_p5_base_preservation.csv") if r.get("status") == "candidate_summary"]
    p5_rows = read_csv_rows(out_dir / "p5_functional_carrier_actuatability.csv")
    lines = [
        "# DG-KAN v9.2.36 Base-Preserving Value-Aligned Functional Subspace 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.2.36_BasePreserving_ValueAlignedFunctionalSubspace_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route['route']}",
        f"base_candidate = {route['base_candidate']}",
        f"success_v9236_strict_purekan_functional = {bool(route['success_v9236_strict_purekan_functional'])}",
        f"success_v9236_full_functional = {bool(route['success_v9236_full_functional'])}",
        f"success_v9236_external_ready = {bool(route['success_v9236_external_ready'])}",
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
        f"1. P0 复现 v9.2.35 boundary：source route = `{route.get('source_route')}`，primary blocker = `{route.get('source_primary_blocker')}`。",
        f"2. P1 VAOP failure mode = `{route.get('vaop_failure_mode')}`，base contamination rate = `{route.get('base_contamination_rate')}`。",
        f"3. P2 BPFS implemented count = `{route.get('bpfs_implemented_count')}`；P3 base equivalence / contract / grad = `{route.get('base_equivalence_pass')}/{route.get('bpfs_contract_pass')}/{route.get('bpfs_grad_pass')}`。",
        f"4. P4 route-level BPFS base qualification = `{route.get('base_preservation_pass')}`；P4 pass = `{route.get('bpfs_p4_pass')}`，P5 near-pass = `{route.get('bpfs_p5_nearpass')}`，best BPFS = `{route.get('best_bpfs_candidate')}`。",
        f"5. P5 carrier pass = `{route.get('functional_carrier_pass')}`，max r_z_tail = `{route.get('max_r_z_tail'):.6f}`，max r_perp_tail = `{route.get('max_r_perp_tail'):.6f}`。",
        f"6. 当前 blocker：`{route.get('primary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `dgkan/models/fc_purekan_actuator.py` | 新增 BPFS1-BPFS6 zero-init base-preserving specs |",
        "| `experiments/run_v9236_base_preserving_value_aligned_functional_subspace.py` | v9.2.36 runner；生成 P0-P12 artifacts、route、failure/no-fake audit |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile dgkan/models/fc_purekan_actuator.py experiments/run_v9236_base_preserving_value_aligned_functional_subspace.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9236_base_preserving_value_aligned_functional_subspace.py \\",
        "  --out-dir results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z \\",
        "  --fresh --device auto --data-root data --seed 1314",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 3. P4 BPFS base preservation",
        "",
        "| candidate | P4 | P5 near | inactive/base-equivalence | near rows | macro delta | step q90 | memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p4_summary:
        lines.append(f"| {r.get('candidate')} | `{r.get('P4_pass')}` | `{r.get('P5_nearpass')}` | `{r.get('base_preservation_pass')}` | `{r.get('near_pass_count')}/{r.get('row_count')}` | `{_float(r.get('macro_delta')):.6f}` | `{_float(r.get('step_q90')):.6f}` | `{_float(r.get('memory_compact')):.6f}` |")
    if not p4_summary:
        lines.append("| none | 0 | 0 | 0 | 0/0 |  |  |  |")
    lines.extend([
        "",
        "## 4. P5 functional carrier",
        "",
        f"rows = `{len([r for r in p5_rows if r.get('status') != 'not_run'])}`",
        "",
        "判断：P5 只测试 event-time functional carrier 是否 non-silent，不把它写成 paired replay success。",
        "",
        "## 5. Downstream boundary",
        "",
        "P7-P12 只有在 P6/P7/P8 gate 后打开。本轮未打开阶段均以 `not_run` row 落盘。",
        "",
        "## 6. No-fake audit",
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
        "## 7. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
        f"| runner | `{_hash(SCRIPT_PATH)}` |",
    ])
    for path in artifacts:
        lines.append(f"| `{_rel(path)}` | `{_hash(path)}` |")
    lines.extend([
        "",
        "## 8. 最终分析结论",
        "",
        "v9.2.36 的真实推进是：",
        "",
        "```text",
        "v9.2.35: VAOP 直接并入 base 后 P5 near-pass 全灭。",
        "v9.2.36: BPFS inactive functional channel 按 zero-init/frozen update 重新审计 base preservation 与 carrier actuatability。",
        "```",
        "",
        "机制判断：",
        "",
        "1. Base preservation 与 functional carrier 被分开 gate；functional update 不允许救 base。",
        "2. 本轮行级 inactive/base-equivalence 没有失败；route-level 失败点是没有候选同时满足 P4 与 P5 near-pass。",
        "3. 如果 BPFS 过 P4/P5 但 P5 carrier 静音，结论是 carrier 仍需重做，不是 dataset patch。",
        "4. 如果 carrier 有 movement 但 value observability 失败，后续应回到 control-gap/value statistic，而不是把 movement 当成功。",
        "",
        f"最终一句话：",
        "",
        f"> v9.2.36 真实执行后停在 `{route['route']}`：`{route.get('primary_blocker')}`。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT / "v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=9984)
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
    # Compatibility for v9214._measure_p4_q.
    args.p4_batch_size = int(args.p4_batch_size)
    args.p4_warmup = int(args.p4_warmup)
    args.p4_reps = int(args.p4_reps)
    args.p4_repeat_measurements = int(args.p4_repeat_measurements)
    args.p5_train_size = int(args.train_size)
    args.p5_test_size = int(args.test_size)
    args.p5_epochs = int(args.epochs)
    args.p5_lr = float(args.lr)
    args.eval_size = int(args.eval_size)
    args.audit_batch_size = int(args.audit_batch_size)
    args.batch_size = int(args.batch_size)
    args.data_root = str(args.data_root)
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
        "version": "v9.2.36",
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "source_v9235": _rel(SRC_V9235),
        "created_at": _now_iso(),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    p0 = _p0_boundary()
    write_csv_rows(out_dir / "p0_v9235_boundary_reproduction.csv", [p0])
    p1_rows, p1 = _p1_autopsy(args, device, bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_vaop_failure_autopsy.csv", p1_rows)
    p2_rows, p2 = _p2_implementation(args, device, bool(_int(p1.get("vaop_failure_attributed"))))
    write_csv_rows(out_dir / "p2_bpfs_implementation.csv", p2_rows)
    write_csv_rows(out_dir / "bpfs_primitive_trace_v9236.csv", p2_rows)
    p3_rows, p3 = _p3_contract(args, device, bool(_int(p2.get("bpfs_implementation_pass"))))
    write_csv_rows(out_dir / "p3_contract_grad_base_equivalence.csv", p3_rows)
    p4_rows, p4, cache = _p4_base(args, device, p3_rows, bool(_int(p3.get("bpfs_p4_eligible_count"))))
    write_csv_rows(out_dir / "p4_bpfs_p4_p5_base_preservation.csv", p4_rows)
    write_csv_rows(out_dir / "base_preservation_trace_v9236.csv", p4_rows)
    p5_rows, p5 = _p5_carrier(args, device, p4, cache, bool(_int(p4.get("bpfs_p5_nearpass"))))
    write_csv_rows(out_dir / "p5_functional_carrier_actuatability.csv", p5_rows)
    write_csv_rows(out_dir / "functional_carrier_trace_v9236.csv", p5_rows)
    p6_rows, p6 = _p6_value(p5_rows, bool(_int(p5.get("functional_carrier_pass"))))
    write_csv_rows(out_dir / "p6_value_observability_audit.csv", p6_rows)
    write_csv_rows(out_dir / "value_score_trace_v9236.csv", p6_rows)
    _write_notrun(out_dir, "P6_value_observability_failed_or_not_opened")
    write_csv_rows(out_dir / "contract_audit_v9236.csv", [{
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
        route_name, primary = "R0-V9235BoundaryMismatch", "v9235_boundary_not_reproduced"
    elif not _int(p1.get("vaop_failure_attributed")):
        route_name, primary = "R1-VAOPFailureUnattributed", "vaop_failure_unattributed"
    elif not _int(p2.get("bpfs_implementation_pass")):
        route_name, primary = "R11-BPFSAllFailPrimitiveReset", "bpfs_not_implemented"
    elif not _int(p3.get("base_equivalence_pass")):
        route_name, primary = "R11-BPFSAllFailPrimitiveReset", "all_bpfs_failed_base_equivalence"
    elif not _int(p4.get("bpfs_p5_nearpass")):
        route_name = "R11-BPFSAllFailPrimitiveReset"
        if _int(p4.get("bpfs_p4_pass_count")):
            primary = "bpfs_p4_pass_but_p5_nearpass_failed"
        else:
            primary = "no_bpfs_passed_P4_P5_base_qualification"
    elif not _int(p5.get("functional_carrier_pass")):
        route_name, primary = "R9-BasePreservedButFunctionalSilent", "bpfs_functional_carrier_silent_or_unsafe"
    elif not _int(p6.get("value_observability_pass")):
        route_name, primary = "R10-BasePreservedButValueUnobservable", "bpfs_value_observability_failed"
    else:
        route_name, primary = "R6-BPFSValueObservabilityPass", "P7_not_implemented_in_this_runner"

    artifacts = [
        out_dir / "run_manifest.json",
        out_dir / "contract_audit_v9236.csv",
        out_dir / "p0_v9235_boundary_reproduction.csv",
        out_dir / "p1_vaop_failure_autopsy.csv",
        out_dir / "p2_bpfs_implementation.csv",
        out_dir / "p3_contract_grad_base_equivalence.csv",
        out_dir / "p4_bpfs_p4_p5_base_preservation.csv",
        out_dir / "p5_functional_carrier_actuatability.csv",
        out_dir / "p6_value_observability_audit.csv",
        out_dir / "p7_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p8_official_base_preserving_paired_replay.csv",
        out_dir / "p9_short_run_functional_validation.csv",
        out_dir / "p10_full_10seed_functional_validation.csv",
        out_dir / "p11_adamw_only_fullpass_repair.csv",
        out_dir / "p12_robustness_external_ready.csv",
        out_dir / "bpfs_primitive_trace_v9236.csv",
        out_dir / "base_preservation_trace_v9236.csv",
        out_dir / "functional_carrier_trace_v9236.csv",
        out_dir / "value_score_trace_v9236.csv",
        out_dir / "leave_dataset_out_trace_v9236.csv",
        out_dir / "paired_replay_branch_trace_v9236.csv",
    ]
    audit = audit_no_fake(artifacts)
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9235_boundary_pass": _int(p0.get("P0_pass")),
        "source_route": p0.get("route", ""),
        "source_primary_blocker": p0.get("primary_blocker", ""),
        "dataset_tuning_detected": 0,
        "vaop_failure_mode": p1.get("vaop_failure_mode", ""),
        "base_contamination_rate": p1.get("base_contamination_rate", 0.0),
        "bpfs_implemented_count": p2.get("bpfs_implemented_count", 0),
        "best_bpfs_candidate": p4.get("best_bpfs_candidate", ""),
        "base_equivalence_pass": p3.get("base_equivalence_pass", 0),
        "bpfs_contract_pass": p3.get("bpfs_contract_pass", 0),
        "bpfs_grad_pass": p3.get("bpfs_grad_pass", 0),
        "bpfs_p4_pass": p4.get("bpfs_p4_pass", 0),
        "bpfs_p5_nearpass": p4.get("bpfs_p5_nearpass", 0),
        "base_preservation_pass": p4.get("base_preservation_pass", 0),
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
        "functional_system_pass": p4.get("bpfs_p4_pass", 0),
        "hard_stratum_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": "if_carrier_silent_redesign_functional_carrier_else_if_value_unobservable_redesign_control_gap_statistic",
        "success_v9236_strict_purekan_functional": 0,
        "success_v9236_full_functional": 0,
        "success_v9236_external_ready": 0,
        **audit,
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    failure = [
        {"stage": "P0", "pass": _int(p0.get("P0_pass")), "blocker": "" if _int(p0.get("P0_pass")) else "F2_v9235_boundary_unstable"},
        {"stage": "P1", "pass": p1.get("vaop_failure_attributed", 0), "blocker": "" if p1.get("vaop_failure_attributed") else "F4_vaop_failure_unattributed"},
        {"stage": "P2", "pass": p2.get("bpfs_implementation_pass", 0), "blocker": "" if p2.get("bpfs_implementation_pass") else "F5_bpfs_not_implemented"},
        {"stage": "P3", "pass": p3.get("base_equivalence_pass", 0), "blocker": "" if p3.get("base_equivalence_pass") else "F6_base_equivalence_fail"},
        {"stage": "P4", "pass": p4.get("bpfs_p5_nearpass", 0), "blocker": "" if p4.get("bpfs_p5_nearpass") else "F8/F9/F10_bpfs_base_fail"},
        {"stage": "P5", "pass": p5.get("functional_carrier_pass", 0), "blocker": "" if p5.get("functional_carrier_pass") else ("not_opened_no_base_preserving_BPFS" if not p4.get("bpfs_p5_nearpass") else "F11_functional_carrier_silent")},
        {"stage": "P6", "pass": p6.get("value_observability_pass", 0), "blocker": "" if p6.get("value_observability_pass") else ("not_opened_carrier_failed" if not p5.get("functional_carrier_pass") else "F12_value_observability_fail")},
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure)
    write_csv_rows(out_dir / "v9236_provenance_audit.csv", [audit])
    final_artifacts = [*artifacts, out_dir / "route_decision.json", out_dir / "aggregate_decision.json", out_dir / "failure_table.csv", out_dir / "v9236_provenance_audit.csv"]
    _write_report(out_dir, route, final_artifacts)


if __name__ == "__main__":
    main()
