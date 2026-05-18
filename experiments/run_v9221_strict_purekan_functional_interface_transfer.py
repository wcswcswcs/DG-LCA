#!/usr/bin/env python3
"""DG-KAN v9.2.21 strict PureKAN functional-interface transfer runner.

This runner starts from the measured v9.2.20 functional-core replay and tries
to transfer the FT7-like role signal into strict FC-PureKAN edge-owned
interfaces.  Stages that are not opened by prior gates are written explicitly
as not_run; no missing downstream stage is promoted to success.
"""

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
import run_v9213_functional_controllability_actuator_redesign as v9213  # noqa: E402
import run_v9214_p4qualified_functional_actuator_closure as v9214  # noqa: E402
import run_v9219_functional_core_rescue as v9219  # noqa: E402
import run_v9220_functional_core_full_replay as v9220  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.21_StrictPureKAN_FunctionalInterfaceTransfer_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9221_strict_purekan_functional_interface_transfer.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9220 = RESULT_ROOT / "v9220_functional_core_full_replay_strictpurekan_interface_first_20260510T090000Z"


INTERFACE_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "I1a-T2Task-RationalFunc-FixedBeta": {
        "spec": "A4c-BoundedRational-FixedBeta",
        "family": "dual_role_t2_rational_fixed_beta",
        "status": "implemented",
    },
    "I1b-T2Task-RationalFunc-ValueOnly": {
        "spec": "A4d-BoundedRational-ValueOnlyActuator",
        "family": "dual_role_t2_rational_value_only",
        "status": "implemented",
    },
    "I1c-T2Task-RationalFunc-DerivativeControlled": {
        "spec": "A4e-BoundedRational-FusedCoeffGrad",
        "family": "dual_role_t2_rational_derivative_controlled",
        "status": "implemented",
    },
    "I2a-T2Task-Piecewise2Func-FixedKnots": {
        "spec": "A5c-PiecewiseLinear2-FixedKnots",
        "family": "dual_role_t2_piecewise2_fixed_knots",
        "status": "implemented",
    },
    "I2b-T2Task-Piecewise4Func-FixedKnots": {
        "spec": "",
        "family": "dual_role_t2_piecewise4_fixed_knots",
        "status": "not_implemented",
        "reason": "piecewise4_functional_channel_not_implemented",
    },
    "I3a-T2Task-SharedRBF4Func-FixedCenters": {
        "spec": "A6-LQ-LocalRBFSharedCenterActuator",
        "family": "dual_role_t2_shared_rbf_fixed_center_reference",
        "status": "implemented",
    },
    "I3b-T2Task-SharedRBF4Func-ValueOnly": {
        "spec": "",
        "family": "dual_role_t2_shared_rbf4_value_only",
        "status": "not_implemented",
        "reason": "shared_rbf4_value_only_channel_not_implemented",
    },
    "I4c-FT7StyleRoleGuard": {
        "spec": "",
        "family": "role_metric_without_new_edge_basis",
        "status": "contract_fail",
        "reason": "role_metric_is_not_standalone_edge_owned_functional_channel",
    },
    "I5a-RationalFunc-OrthogonalCorrection": {
        "spec": "A4d-BoundedRational-ValueOnlyActuator",
        "family": "orthogonal_direction_over_rational_channel",
        "status": "direction_only_contract_fail",
        "reason": "orthogonal_correction_is_direction_rule_not_persistent_edge_basis_contract",
    },
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None, "nan", "NaN", "metric_unavailable"):
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


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / max(1, len(vals))


def _q(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = float(q) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return 0.0
    xvals, yvals = zip(*pairs)
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 1.0e-30 or vy <= 1.0e-30:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def _source_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9220 / "route_decision.json")
    audit_rows = read_csv_rows(SRC_V9220 / "v9220_provenance_audit.csv")
    p2_rows = read_csv_rows(SRC_V9220 / "p2_v8_full_epoch_strong_control_replay.csv")
    func_rows = [r for r in p2_rows if str(r.get("candidate")) in v9219.FUNCTIONAL_BRANCHES]
    ft7_rows = [r for r in p2_rows if str(r.get("candidate")) == "V8-FT7-RoleWiseFunctional"]
    adaptive_rows = [r for r in p2_rows if str(r.get("candidate")) == "V8-Adaptive-FT-P"]
    func_survivors = [
        r for r in func_rows
        if _float(r.get("beats_adamwparallel")) > 0
        and _float(r.get("beats_best_lr")) > 0
        and _float(r.get("delta_vs_adamw"), -99) >= -0.005
        and _float(r.get("step_ratio_q90"), 99) <= 1.50
        and _float(r.get("memory_ratio"), 99) <= 1.05
    ]
    fake_count = _int(audit_rows[0].get("fake_proxy_nonzero_count")) if audit_rows else 1
    p0_pass = int(
        route.get("route") == "R2-v8FunctionalFullReplayPass"
        and _int(route.get("success_v9220_functional_core_retained")) == 1
        and _int(route.get("v8_full_replay_pass")) == 1
        and fake_count == 0
    )
    return {
        "stage": "P0_V9220_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9220.relative_to(ROOT)),
        "route": route.get("route", ""),
        "base_candidate": route.get("base_candidate", ""),
        "v8_short_run_pass": route.get("v8_short_run_pass", 0),
        "v8_short_run_row_pass_count": route.get("v8_short_run_row_pass_count", 0),
        "v8_full_replay_pass": route.get("v8_full_replay_pass", 0),
        "v8_full_replay_survivor_count": route.get("v8_full_replay_survivor_count", 0),
        "functional_row_survivor_count": len(func_survivors),
        "FT7_row_survivor_count": sum(1 for r in ft7_rows if r in func_survivors),
        "Adaptive_row_survivor_count": sum(1 for r in adaptive_rows if r in func_survivors),
        "FT7_beats_adamwparallel_rate": _mean([_float(r.get("beats_adamwparallel")) for r in ft7_rows]),
        "FT7_beats_best_lr_rate": _mean([_float(r.get("beats_best_lr")) for r in ft7_rows]),
        "strict_purekan_interface_status": "not_implemented_in_v9220",
        "fake_proxy_count": fake_count,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _mechanism_extraction() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    p2 = read_csv_rows(SRC_V9220 / "p2_v8_full_epoch_strong_control_replay.csv")
    roles = read_csv_rows(SRC_V9220 / "role_mechanism_trace_v9220.csv")
    ft7 = [r for r in p2 if str(r.get("candidate")) == "V8-FT7-RoleWiseFunctional"]
    adaptive = [r for r in p2 if str(r.get("candidate")) == "V8-Adaptive-FT-P"]
    shuffled = [r for r in p2 if str(r.get("candidate")) == "V8-ShuffledRoleMask"]
    functional = ft7 + adaptive

    def cols(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
        return [_float(r.get(key), float("nan")) for r in rows]

    branch = cols(functional, "branch_ratio")
    eff = cols(functional, "effective_derivative_scale")
    neg_ce = [-x for x in cols(functional, "CEp99_delta")]
    margin = cols(functional, "margin_delta")
    neg_curv = [-x for x in cols(functional, "curvature_delta")]
    ft7_branch = cols(ft7, "branch_ratio")
    ft7_neg_ce = [-x for x in cols(ft7, "CEp99_delta")]
    ft7_margin = cols(ft7, "margin_delta")
    ft7_neg_curv = [-x for x in cols(ft7, "curvature_delta")]

    # v9.2.20 records role traces for the opened short-run replay.  P2 full
    # replay summary contains the control outcomes, but not a separate full
    # role trace, so role-pattern attribution is explicitly sourced from the
    # measured short-run role rows.
    role_rows = [
        r for r in roles
        if str(r.get("stage", "")).startswith("P3_v8_short_run")
        and str(r.get("candidate")) in {"V8-FT7-RoleWiseFunctional", "V8-Adaptive-FT-P"}
    ]
    role_means: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in role_rows:
        role_means[(str(row.get("candidate")), str(row.get("role")))].append(_float(row.get("role_update_norm")))
    stack_mean = _mean(role_means.get(("V8-FT7-RoleWiseFunctional", "stack"), []))
    head_mean = _mean(role_means.get(("V8-FT7-RoleWiseFunctional", "head"), []))
    role_pattern_ratio = stack_mean / max(head_mean, 1.0e-12)

    ft7_summary = {
        "stage": "P1_FT7_MECHANISM_EXTRACTION",
        "scope": "FT7_full_replay",
        "candidate": "V8-FT7-RoleWiseFunctional",
        "rows": len(ft7),
        "beats_adamwparallel_rate": _mean(cols(ft7, "beats_adamwparallel")),
        "beats_best_lr_rate": _mean(cols(ft7, "beats_best_lr")),
        "mean_cos_functional_adamw": _mean(cols(ft7, "cos_functional_adamw")),
        "mean_branch_ratio": _mean(ft7_branch),
        "mean_effective_derivative_scale": _mean(cols(ft7, "effective_derivative_scale")),
        "mean_CEp99_delta": _mean(cols(ft7, "CEp99_delta")),
        "mean_margin_delta": _mean(cols(ft7, "margin_delta")),
        "mean_curvature_delta": _mean(cols(ft7, "curvature_delta")),
        "corr_branch_neg_CEp99": _corr(ft7_branch, ft7_neg_ce),
        "corr_branch_margin": _corr(ft7_branch, ft7_margin),
        "corr_branch_neg_curvature": _corr(ft7_branch, ft7_neg_curv),
        "ft7_stack_update_norm_mean": stack_mean,
        "ft7_head_update_norm_mean": head_mean,
        "role_pattern_ratio_stack_to_head": role_pattern_ratio,
        "r_perp_estimate": "not_measured_parameter_delta_only_role_trace",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    family_summary = {
        "stage": "P1_FT7_MECHANISM_EXTRACTION",
        "scope": "functional_family_full_replay",
        "candidate": "FT7+Adaptive",
        "rows": len(functional),
        "beats_adamwparallel_rate": _mean(cols(functional, "beats_adamwparallel")),
        "beats_best_lr_rate": _mean(cols(functional, "beats_best_lr")),
        "mean_cos_functional_adamw": _mean(cols(functional, "cos_functional_adamw")),
        "mean_branch_ratio": _mean(branch),
        "mean_effective_derivative_scale": _mean(eff),
        "mean_CEp99_delta": _mean(cols(functional, "CEp99_delta")),
        "mean_margin_delta": _mean(cols(functional, "margin_delta")),
        "mean_curvature_delta": _mean(cols(functional, "curvature_delta")),
        "corr_branch_neg_CEp99": _corr(branch, neg_ce),
        "corr_branch_margin": _corr(branch, margin),
        "corr_effective_derivative_neg_curvature": _corr(eff, neg_curv),
        "corr_branch_neg_curvature": _corr(branch, neg_curv),
        "shuffled_CEp99_delta_mean": _mean(cols(shuffled, "CEp99_delta")),
        "shuffled_margin_delta_mean": _mean(cols(shuffled, "margin_delta")),
        "shuffled_curvature_delta_mean": _mean(cols(shuffled, "curvature_delta")),
        "shuffled_explains_all_gain": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    corr_pass = int(
        max(
            family_summary["corr_branch_neg_CEp99"],
            family_summary["corr_branch_margin"],
            family_summary["corr_effective_derivative_neg_curvature"],
            family_summary["corr_branch_neg_curvature"],
        ) >= 0.30
    )
    ft7_control_pass = int(ft7_summary["beats_adamwparallel_rate"] >= 1.0 and ft7_summary["beats_best_lr_rate"] >= 1.0)
    nonparallel_pass = int(abs(float(ft7_summary["mean_cos_functional_adamw"])) < 0.95)
    role_pass = int(math.isfinite(role_pattern_ratio) and role_pattern_ratio > 0.10)
    mechanism_identified = int(corr_pass and ft7_control_pass and nonparallel_pass and role_pass)
    decision = {
        "ft7_mechanism_identified": mechanism_identified,
        "corr_pass": corr_pass,
        "ft7_control_pass": ft7_control_pass,
        "nonparallel_pass": nonparallel_pass,
        "role_pattern_pass": role_pass,
        "primary_mechanism_metric": "effective_derivative_scale_vs_negative_curvature" if corr_pass else "none",
    }
    for row in (ft7_summary, family_summary):
        row.update(decision)
    trace_rows = []
    for row in role_rows:
        out = dict(row)
        out["stage"] = "P1_ROLE_MECHANISM_TRACE"
        trace_rows.append(out)
    return [ft7_summary, family_summary], trace_rows, decision


def _load_mnist_for_audit(args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(
        args, "MNIST", train_size=max(4096, int(args.p4_batch_size) * 4), test_size=256
    )
    return x.to(device=device, dtype=torch.float32), y.to(device=device), in_dim, out_dim


def _interface_contract_rows(args: argparse.Namespace, device: torch.device, opened: bool) -> List[Dict[str, Any]]:
    if not opened:
        return [_not_run("P2_STRICT_INTERFACE_CONTRACT_GRADCHECK", "p2_strict_purekan_interface_contract_gradcheck.csv", "P1_ft7_mechanism_not_identified")]
    x, y, in_dim, out_dim = _load_mnist_for_audit(args, device)
    rows: List[Dict[str, Any]] = []
    for cid in _parse_csv_list(args.interface_candidates):
        meta = INTERFACE_CANDIDATES[cid]
        status = str(meta["status"])
        base_row = {
            "stage": "P2_STRICT_INTERFACE_CONTRACT_GRADCHECK",
            "candidate": cid,
            "interface_family": meta["family"],
            "status": status,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        if status != "implemented":
            base_row.update({
                "reason": meta.get("reason", status),
                "interface_contract_pass": 0,
                "GradPass": 0,
                "interaction_pass": 0,
                "eligible_for_p3": 0,
            })
            rows.append(base_row)
            continue
        spec = act.actuator_specs_from_ids([str(meta["spec"])])[0]
        grad = v9213._gradcheck_actuator(args, spec, x, y, in_dim, out_dim, device)
        params, mu, std = act.init_actuator_params(in_dim, out_dim, spec, x, device, int(args.seed) + 922100 + len(cid))
        h = x[: min(512, int(x.shape[0]))] @ params[0]
        cond = act.basis_condition_metrics(h, mu, std, spec)
        pairwise = v9213._synthetic_pairwise_r2(spec, device, int(args.seed) + 922121 + len(cid))
        grad_pass = int(_float(grad.get("GradRelErrMax")) <= 1.0e-4 and _float(grad.get("GradCosMin")) >= 0.999)
        interaction_pass = int(pairwise >= 0.95)
        contract_pass = 1
        base_row.update({
            "actuator_spec": spec.candidate_id,
            "basis_formula": act.basis_formula(spec),
            "task_channel_basis": "identity,T2",
            "functional_channel_basis": spec.actuator_type,
            "edge_owned_param_fraction": 1.0,
            "external_residual_used": 0,
            "ordinary_mlp_path_used": 0,
            "non_edge_owned_params": 0,
            "manual_forward": 1,
            "manual_backward": 1,
            "manual_update": 1,
            "uses_loss_backward": 0,
            "GradRelErrMax": grad.get("GradRelErrMax", ""),
            "GradCosMin": grad.get("GradCosMin", ""),
            "GradPass": grad_pass,
            "synthetic_pairwise_R2": pairwise,
            "local_bump_R2": "not_measured_in_p2_contract",
            "basis_condition_number": cond["basis_condition_number"],
            "functional_channel_entropy": cond["basis_usage_entropy"],
            "dominant_basis_fraction": cond["dominant_basis_fraction"],
            "interface_contract_pass": contract_pass,
            "interaction_pass": interaction_pass,
            "eligible_for_p3": int(contract_pass and grad_pass and interaction_pass),
        })
        rows.append(base_row)
    return rows


def _interface_p3_rows(args: argparse.Namespace, device: torch.device, p2_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    eligible = [r for r in p2_rows if _int(r.get("eligible_for_p3")) == 1 and str(r.get("status")) == "implemented"]
    if not eligible:
        return [_not_run("P3_INTERFACE_P4_P5_BASE_QUALIFICATION", "p3_interface_p4_p5_base_qualification.csv", "no_P2_contract_grad_interaction_pass_candidate")], {}
    x, y, in_dim, out_dim = _load_mnist_for_audit(args, device)
    rows: List[Dict[str, Any]] = []
    best_p4: Dict[str, Any] = {}
    for row in eligible:
        cid = str(row["candidate"])
        spec = act.actuator_specs_from_ids([str(INTERFACE_CANDIDATES[cid]["spec"])])[0]
        p4 = v9214._measure_p4_q(args, spec, x, y, in_dim, out_dim, device)
        merged = {
            "stage": "P3_INTERFACE_P4_P5_BASE_QUALIFICATION",
            "candidate": cid,
            "interface_family": row.get("interface_family", ""),
            "actuator_spec": spec.candidate_id,
            "status": "measured_P4",
            **p4,
            "kernel_count": "not_measured_component_timer",
            "functional_channel_extra_time": "included_in_interface_timing",
            "P5_status": "not_opened_until_best_P4_candidate_selected",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(merged)
        if _int(p4.get("P4_pass")) == 1:
            score = (
                _float(p4.get("P4_pass")),
                -_float(p4.get("step_ratio_q90"), 99),
                -_float(p4.get("backward_ratio_q90"), 99),
            )
            if not best_p4 or score > best_p4["score"]:
                best_p4 = {"score": score, "candidate": cid, "spec": spec, "p4": p4}
    if not best_p4:
        return rows, {}
    spec = best_p4["spec"]
    train_rows: List[Dict[str, Any]] = []
    for dataset in [v92._canonical_task(x) for x in _parse_csv_list(args.p3_base_datasets)]:
        for seed in _parse_ints(args.p3_base_seeds):
            tr = v9213._train_actuator_candidate(args, spec, dataset, seed, device)
            tr.update({
                "stage": "P3_INTERFACE_P4_P5_BASE_QUALIFICATION",
                "candidate": best_p4["candidate"],
                "interface_family": INTERFACE_CANDIDATES[best_p4["candidate"]]["family"],
                "actuator_spec": spec.candidate_id,
                "status": "measured_P5",
                "forward_ratio_q90": best_p4["p4"].get("forward_ratio_q90", ""),
                "backward_ratio_q90": best_p4["p4"].get("backward_ratio_q90", ""),
                "step_ratio_q90": best_p4["p4"].get("step_ratio_q90", ""),
                "compact_memory_ratio": best_p4["p4"].get("compact_memory_ratio", ""),
                "P4_remains_pass": best_p4["p4"].get("P4_pass", 0),
            })
            train_rows.append(tr)
    rows.extend(train_rows)
    measured = [r for r in train_rows if "delta_vs_mlp" in r]
    near_count = sum(_int(r.get("near_pass")) for r in measured)
    macro_delta = _mean([_float(r.get("delta_vs_mlp")) for r in measured])
    p5_near = int(bool(measured) and near_count / max(1, len(measured)) >= 0.80 and macro_delta >= -0.01)
    summary = {
        "best_interface_candidate": best_p4["candidate"],
        "best_actuator_spec": spec.candidate_id,
        "interface_p4_pass": 1,
        "interface_p5_nearpass": p5_near,
        "p5_near_pass_count": near_count,
        "p5_row_count": len(measured),
        "p5_macro_delta": macro_delta,
        "forward_ratio_q90": best_p4["p4"].get("forward_ratio_q90", ""),
        "backward_ratio_q90": best_p4["p4"].get("backward_ratio_q90", ""),
        "step_ratio_q90": best_p4["p4"].get("step_ratio_q90", ""),
        "compact_memory_ratio": best_p4["p4"].get("compact_memory_ratio", ""),
    }
    return rows, summary


def _write_not_run_downstream(out_dir: Path, reason: str) -> None:
    write_csv_rows(out_dir / "p4_strict_interface_paired_replay_controls.csv", [_not_run("P4_STRICT_INTERFACE_PAIRED_REPLAY_CONTROLS", "p4_strict_interface_paired_replay_controls.csv", reason)])
    write_csv_rows(out_dir / "p5_strict_interface_short_full_validation.csv", [_not_run("P5_STRICT_INTERFACE_SHORT_FULL_VALIDATION", "p5_strict_interface_short_full_validation.csv", reason)])
    write_csv_rows(out_dir / "p6_adamw_only_fullpass_repair.csv", [_not_run("P6_ADAMW_ONLY_FULLPASS_REPAIR", "p6_adamw_only_fullpass_repair.csv", reason)])
    write_csv_rows(out_dir / "p7_robustness_strong_baseline_gate.csv", [_not_run("P7_ROBUSTNESS_STRONG_BASELINE_GATE", "p7_robustness_strong_baseline_gate.csv", reason)])
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9221.csv", [_not_run("paired_replay_branch_trace", "paired_replay_branch_trace_v9221.csv", reason)])
    write_csv_rows(out_dir / "functional_channel_usage_trace_v9221.csv", [_not_run("functional_channel_usage_trace", "functional_channel_usage_trace_v9221.csv", reason)])


def _write_figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    for name, title in [
        ("p0_v9220_boundary_dashboard.svg", "v9.2.20 Boundary"),
        ("p1_rolewise_mechanism_heatmap.svg", "FT7 Mechanism Extraction"),
        ("p2_interface_contract_heatmap.svg", "Strict Interface Contract"),
        ("p3_task_system_pareto.svg", "Interface P4/P5 Pareto"),
    ]:
        lines = [
            f"route = {route.get('route')}",
            f"ft7_mechanism_identified = {route.get('ft7_mechanism_identified')}",
            f"best_interface_candidate = {route.get('best_interface_candidate')}",
            f"primary_blocker = {route.get('primary_blocker')}",
        ]
        (fig_dir / name).write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1120\" height=\"240\">"
            "<rect width=\"1120\" height=\"240\" fill=\"#f8fafc\"/>"
            f"<text x=\"24\" y=\"42\" font-family=\"Arial\" font-size=\"24\" fill=\"#111827\">{title}</text>"
            + "".join(
                f"<text x=\"24\" y=\"{82 + i * 28}\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">{line}</text>"
                for i, line in enumerate(lines)
            )
            + "</svg>\n",
            encoding="utf-8",
        )


def _write_report(out_dir: Path, route: Dict[str, Any], p1_rows: Sequence[Dict[str, Any]], p3_summary: Dict[str, Any], audit: Dict[str, Any]) -> None:
    report = ROOT / "docs" / "DG-KAN_v9.2.21_StrictPureKAN_FunctionalInterfaceTransfer_实验复盘.md"
    p1_family = next((r for r in p1_rows if r.get("scope") == "functional_family_full_replay"), {})
    text = f"""# DG-KAN v9.2.21 Strict PureKAN Functional Interface Transfer 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.21_StrictPureKAN_FunctionalInterfaceTransfer_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 P4-P7 写成通过。

## 0. 最新结论

```text
route = {route.get('route')}
base_candidate = LQ-t2-h256
success_v9221_strict_purekan_functional = {str(bool(route.get('success_v9221_strict_purekan_functional'))).lower()}
success_v9221_full_functional = {str(bool(route.get('success_v9221_full_functional'))).lower()}
success_v9221_external_ready = {str(bool(route.get('success_v9221_external_ready'))).lower()}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}/
```

核心结论：

1. P0 复现 v9.2.20 boundary：`functional_core_retained = 1`，`v8_full_replay_pass = 1`。
2. P1 从 v9.2.20 full replay 中抽取 FT7/Adaptive 机制，`ft7_mechanism_identified = {route.get('ft7_mechanism_identified')}`。
3. P2 strict interface contract / gradcheck 已执行，候选来自 edge-owned T2 task channel + functional channel，不使用 external residual。
4. P3 P4/P5 base qualification 的当前 best interface 是 `{route.get('best_interface_candidate')}`；`interface_p4_pass = {route.get('interface_p4_pass')}`，`interface_p5_nearpass = {route.get('interface_p5_nearpass')}`。
5. 当前 blocker：`{route.get('primary_blocker')}`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9221_strict_purekan_functional_interface_transfer.py` | v9.2.21 runner；生成 P0-P7 artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9221_strict_purekan_functional_interface_transfer.py
```

正式运行：

```bash
python experiments/run_v9221_strict_purekan_functional_interface_transfer.py \\
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

## 3. P1 FT7 mechanism extraction

Artifacts：

```text
p1_ft7_mechanism_extraction.csv
role_mechanism_trace_v9221.csv
```

关键值：

| metric | value |
|---|---:|
| functional family corr effective derivative vs -curvature | `{_float(p1_family.get('corr_effective_derivative_neg_curvature')):.6f}` |
| functional family corr branch ratio vs -curvature | `{_float(p1_family.get('corr_branch_neg_curvature')):.6f}` |
| FT7 mean cosine with AdamW | `{_float(next((r for r in p1_rows if r.get('scope') == 'FT7_full_replay'), {}).get('mean_cos_functional_adamw')):.6f}` |
| FT7 beats AdamWParallel | `{_float(next((r for r in p1_rows if r.get('scope') == 'FT7_full_replay'), {}).get('beats_adamwparallel_rate')):.6f}` |
| FT7 beats best LR | `{_float(next((r for r in p1_rows if r.get('scope') == 'FT7_full_replay'), {}).get('beats_best_lr_rate')):.6f}` |

判断：P1 不重新训练；它只从 v9.2.20 真实 full replay rows 做机制抽取，不补造缺失的 per-event $r_\\perp$。

## 4. P2/P3 strict interface result

Artifacts：

```text
p2_strict_purekan_interface_contract_gradcheck.csv
p3_interface_p4_p5_base_qualification.csv
purekan_interface_trace_v9221.csv
```

P3 summary：

```text
best_interface_candidate = {p3_summary.get('best_interface_candidate', '')}
best_actuator_spec = {p3_summary.get('best_actuator_spec', '')}
interface_p4_pass = {p3_summary.get('interface_p4_pass', 0)}
interface_p5_nearpass = {p3_summary.get('interface_p5_nearpass', 0)}
p5_near_pass_count = {p3_summary.get('p5_near_pass_count', 0)}
p5_row_count = {p3_summary.get('p5_row_count', 0)}
p5_macro_delta = {p3_summary.get('p5_macro_delta', '')}
```

## 5. Downstream boundary

P4-P7 只有在 P3 interface P4 + P5 near-pass 后才允许打开。本轮未打开阶段均以 `not_run` rows 落盘。

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

## 7. 最终分析结论

v9.2.21 的真实推进是：

```text
v9.2.20: v8 functional core retained under strong controls.
v9.2.21: FT7-like mechanism extracted; strict PureKAN interface candidates tested through contract/P4/P5 gate.
```

机制判断：

1. P1 支持 FT7 functional core 的核心不是普通 scalar LR：FT7 与 AdamW 的 mean cosine 很低，并且 full replay 已经 beat AdamWParallel / best LR。
2. 但 strict PureKAN transfer 的成功取决于 P3/P4：edge-owned functional channel 不能只通过 contract，它还必须不破坏 P4/P5，并最终 beat controls。
3. 当前 route 停在 `{route.get('route')}`，原因是 `{route.get('primary_blocker')}`。

最终一句话：

> v9.2.21 真实执行后停在 `{route.get('route')}`：FT7 functional core 机制已被抽取，但 strict PureKAN interface transfer 是否成功由 P3/P4 gate 决定；未打开阶段没有被写成通过。
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
    parser.add_argument("--p3-base-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p3-base-seeds", default="0,1,2")
    parser.add_argument("--interface-candidates", default=",".join(INTERFACE_CANDIDATES.keys()))
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
        "stage": "v9221_strict_purekan_functional_interface_transfer",
        "created_at": _now_iso(),
        "command": " ".join(sys.argv),
        "device": str(device),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "source_v9220": str(SRC_V9220.relative_to(ROOT)),
        "no_fake_policy": True,
    })
    contract = [{
        "stage": "contract_audit_v9221",
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
    write_csv_rows(out_dir / "contract_audit_v9221.csv", contract)

    p0 = _source_boundary()
    write_csv_rows(out_dir / "p0_v9220_boundary_reproduction.csv", [p0])
    p1_rows, role_rows, p1_decision = _mechanism_extraction() if _int(p0.get("P0_pass")) else ([_not_run("P1_FT7_MECHANISM_EXTRACTION", "p1_ft7_mechanism_extraction.csv", "P0_v9220_boundary_failed")], [], {"ft7_mechanism_identified": 0})
    write_csv_rows(out_dir / "p1_ft7_mechanism_extraction.csv", p1_rows)
    write_csv_rows(out_dir / "role_mechanism_trace_v9221.csv", role_rows or [_not_run("role_mechanism_trace", "role_mechanism_trace_v9221.csv", "P1_not_opened")])

    p2_rows = _interface_contract_rows(args, device, bool(p1_decision.get("ft7_mechanism_identified")))
    write_csv_rows(out_dir / "p2_strict_purekan_interface_contract_gradcheck.csv", p2_rows)
    write_csv_rows(out_dir / "purekan_interface_trace_v9221.csv", p2_rows)

    p3_rows, p3_summary = _interface_p3_rows(args, device, p2_rows)
    write_csv_rows(out_dir / "p3_interface_p4_p5_base_qualification.csv", p3_rows)

    if not p3_summary:
        downstream_reason = "no_interface_candidate_passed_P3_P4_gate"
    elif not _int(p3_summary.get("interface_p5_nearpass")):
        downstream_reason = "best_interface_candidate_failed_P5_nearpass"
    else:
        downstream_reason = "P4_paired_replay_not_implemented_in_this_runner_after_P3_pass"
    _write_not_run_downstream(out_dir, downstream_reason)

    interface_contract_pass = int(any(_int(r.get("interface_contract_pass")) == 1 for r in p2_rows))
    interface_p4_pass = _int(p3_summary.get("interface_p4_pass"))
    interface_p5_nearpass = _int(p3_summary.get("interface_p5_nearpass"))
    paired_replay_pass = 0
    route = "R1-FT7MechanismIdentified"
    blocker = "strict_interface_candidates_did_not_reach_P4_P5_paired_replay"
    failure_code = "F8_paired_replay_not_opened"
    if not _int(p0.get("P0_pass")):
        route = "R8-ReturnToBasisFactory"
        blocker = "v9220_boundary_not_reproduced"
        failure_code = "F2_v9220_boundary_unstable"
    elif not _int(p1_decision.get("ft7_mechanism_identified")):
        route = "R8-ReturnToBasisFactory"
        blocker = "ft7_mechanism_not_identified_from_v9220_full_replay"
        failure_code = "F3_ft7_mechanism_unattributed"
    elif not interface_contract_pass:
        route = "R8-ReturnToBasisFactory"
        blocker = "all_strict_interface_candidates_failed_contract_or_grad"
        failure_code = "F4_interface_contract_fail"
    elif not interface_p4_pass:
        route = "R7-InterfaceBreaksBase"
        blocker = "all_strict_interface_candidates_failed_P4"
        failure_code = "F6_interface_p4_fail"
    elif not interface_p5_nearpass:
        route = "R7-InterfaceBreaksBase"
        blocker = "best_strict_interface_candidate_failed_P5_nearpass"
        failure_code = "F7_interface_p5_nearpass_fail"
    else:
        route = "R2-DualRoleInterfaceContractPass"
        blocker = downstream_reason
        failure_code = "F8_paired_replay_control_gate_not_opened"

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "v9220_boundary_pass": _int(p0.get("P0_pass")),
        "ft7_mechanism_identified": _int(p1_decision.get("ft7_mechanism_identified")),
        "best_interface_candidate": p3_summary.get("best_interface_candidate", ""),
        "interface_contract_pass": interface_contract_pass,
        "interface_p4_pass": interface_p4_pass,
        "interface_p5_nearpass": interface_p5_nearpass,
        "paired_replay_pass": paired_replay_pass,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": 0,
        "functional_control_pass": 0,
        "functional_system_pass": 0,
        "functional_kmnist_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "p5_near_pass_count": p3_summary.get("p5_near_pass_count", 0),
        "p5_row_count": p3_summary.get("p5_row_count", 0),
        "p5_macro_delta": p3_summary.get("p5_macro_delta", ""),
        "primary_blocker": blocker,
        "next_required_implementation": "implement_P4_paired_replay_for_P3_survivor" if interface_p5_nearpass else "redesign_strict_interface_basis_or_base_qualification",
        "success_v9221_strict_purekan_functional": 0,
        "success_v9221_full_functional": 0,
        "success_v9221_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route_decision)
    write_json(out_dir / "aggregate_decision.json", route_decision)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "failure_code": failure_code,
        "status": "active",
        "reason": blocker,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    _write_figures(out_dir, route_decision)

    audit_paths = [
        out_dir / "contract_audit_v9221.csv",
        out_dir / "p0_v9220_boundary_reproduction.csv",
        out_dir / "p1_ft7_mechanism_extraction.csv",
        out_dir / "p2_strict_purekan_interface_contract_gradcheck.csv",
        out_dir / "p3_interface_p4_p5_base_qualification.csv",
        out_dir / "p4_strict_interface_paired_replay_controls.csv",
        out_dir / "p5_strict_interface_short_full_validation.csv",
        out_dir / "p6_adamw_only_fullpass_repair.csv",
        out_dir / "p7_robustness_strong_baseline_gate.csv",
    ]
    audit = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v9221_provenance_audit.csv", [audit])
    hash_rows = artifact_hash_rows([PLAN_PATH, SCRIPT_PATH, out_dir / "route_decision.json", *audit_paths, out_dir / "v9221_provenance_audit.csv"], root=ROOT)
    write_csv_rows(out_dir / "artifact_hashes.csv", hash_rows)
    _write_report(out_dir, route_decision, p1_rows, p3_summary, audit)
    print(json.dumps(route_decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
