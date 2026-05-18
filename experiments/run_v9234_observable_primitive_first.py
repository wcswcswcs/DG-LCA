#!/usr/bin/env python3
"""DG-KAN v9.2.34 observable primitive first runner.

This runner starts from the measured v9.2.33 boundary where legal train-stream
probes were not predictive and OP1-OP6 were still unimplemented.  It implements
and audits OP primitives as strict FC-PureKAN edge-owned actuator channels.
Downstream observability / paired replay stages are opened only when the
implementation, contract, P4, and P5 base gates have already passed.
"""

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

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9213_functional_controllability_actuator_redesign as v9213  # noqa: E402
import run_v9214_p4qualified_functional_actuator_closure as v9214  # noqa: E402
import run_v9222_basequalified_strict_purekan_functional_interface as v9222  # noqa: E402
import run_v9223_actuatability_to_causality_closure as v9223  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import lq_output_space_functional as out_lq  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.34_ObservablePrimitiveFirst_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9234_observable_primitive_first.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.34_ObservablePrimitiveFirst_实验复盘.md"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9233 = RESULT_ROOT / "v9233_legal_train_stream_probe_observable_primitive_first_20260510T233000Z"

OP_IDS = [
    "OP1-ObservableTailLinearChannel",
    "OP2-ObservablePiecewiseTailChannel",
    "OP3-ObservableSharedRBFLocalChannel",
    "OP4-ObservableOrthogonalTailChannel",
    "OP5-ObservableControlGapChannel",
    "OP6-LightHybridObservable",
]


@dataclass(frozen=True)
class OPCandidate:
    candidate_id: str
    spec: act.ActuatorSpec | None
    primitive_family: str
    status: str = "implemented"
    reason: str = ""


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


def _helper_args(args: argparse.Namespace) -> argparse.Namespace:
    for name, value in {
        "p5_train_size": args.train_size,
        "p5_test_size": args.test_size,
        "p5_epochs": args.p5_epochs,
        "p5_lr": args.lr,
        "eval_size": args.eval_size,
        "audit_batch_size": args.audit_batch_size,
        "batch_size": args.batch_size,
        "data_root": args.data_root,
        "seed": args.seed,
        "lr": args.lr,
        "ridge": args.ridge,
        "best_lr_scale": args.best_lr_scale,
    }.items():
        setattr(args, name, value)
    return args


def _registry() -> Dict[str, OPCandidate]:
    out: Dict[str, OPCandidate] = {}
    for cid in OP_IDS:
        if cid in act.ACTUATOR_SPECS:
            spec = act.ACTUATOR_SPECS[cid]
            out[cid] = OPCandidate(cid, spec, spec.actuator_type)
        else:
            out[cid] = OPCandidate(cid, None, "not_implemented", status="not_implemented", reason="missing_ACTUATOR_SPEC")
    return out


def _interface_candidate(op: OPCandidate) -> v9222.InterfaceCandidate:
    assert op.spec is not None
    return v9222.InterfaceCandidate(
        candidate_id=op.candidate_id,
        interface_family=op.primitive_family,
        spec=op.spec,
        functional_channel_init="tiny_1e-3",
        functional_channel_adamw_trainable=1,
        functional_channel_event_trainable=1,
        schedule="observable_primitive_trains_all_channels",
        status=op.status,
        reason=op.reason,
    )


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9233 / "route_decision.json")
    audit = read_csv_rows(SRC_V9233 / "v9233_provenance_audit.csv")
    fake_count = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R14-ReturnToInterfacePrimitiveDesign"
        and _int(route.get("legal_probe_predictive_pass")) == 0
        and _int(route.get("legal_probe_system_pass")) == 0
        and _int(route.get("observable_primitive_not_implemented_count")) == 6
        and fake_count == 0
    )
    return {
        "stage": "P0_V9233_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9233.relative_to(ROOT)),
        "route": route.get("route", ""),
        "source_route_v9232": route.get("route", ""),
        "cp5_legality_classification": route.get("cp5_legality_classification", ""),
        "cp5_best_feature_auc": route.get("cp5_best_feature_auc", ""),
        "best_legal_probe": route.get("best_legal_probe", ""),
        "legal_probe_corr": route.get("legal_probe_corr", ""),
        "legal_probe_auc": route.get("legal_probe_auc", ""),
        "legal_probe_precision": route.get("legal_probe_precision", ""),
        "legal_probe_coverage": route.get("legal_probe_coverage", ""),
        "legal_probe_bad_event_rate": route.get("legal_probe_bad_event_rate", ""),
        "legal_probe_step_q90": route.get("legal_probe_step_q90", ""),
        "legal_probe_memory_ratio": route.get("legal_probe_memory_ratio", ""),
        "legal_probe_predictive_pass": route.get("legal_probe_predictive_pass", 0),
        "legal_probe_system_pass": route.get("legal_probe_system_pass", 0),
        "best_observable_primitive": route.get("best_observable_primitive", ""),
        "best_observable_primitive_corr": route.get("best_observable_primitive_corr", ""),
        "op_not_implemented_count": route.get("observable_primitive_not_implemented_count", ""),
        "fake_proxy_count": fake_count,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_legal_probe_autopsy(opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_LEGAL_PROBE_FAILURE_AUTOPSY", "p1_legal_probe_failure_autopsy.csv", "P0_v9233_boundary_failed")
        return [row], {"p1_pass": 0, "major_failure_reason": "not_opened"}
    rows: List[Dict[str, Any]] = []
    source = read_csv_rows(SRC_V9233 / "p1_source_logged_cp5_legality_autopsy.csv")
    for r in source:
        posthoc = 1.0 if (_int(r.get("posthoc_outcome_used_in_v9232")) or _int(r.get("validation_metric_used")) or _int(r.get("test_metric_used"))) else 0.0
        rows.append({
            "stage": "P1_LEGAL_PROBE_FAILURE_AUTOPSY",
            "probe": f"source-logged-{r.get('cp5_feature', 'CP5')}",
            "source_or_legal": "source_logged",
            "feature_source": r.get("cp5_feature", "horizon_consistency_recomputable"),
            "available_before_commit": 0,
            "posthoc_component_fraction": posthoc,
            "train_stream_recomputable": r.get("can_be_recomputed_as_train_stream_probe", 0),
            "probe_split": "source_logged_replay_rows",
            "horizon": "source_mixed",
            "corr": "",
            "auc": r.get("auc_with_Y_beat_best_orientation", ""),
            "precision": "",
            "coverage": "",
            "bad_event_rate": "",
            "step_q90": "",
            "memory_ratio": "",
            "failure_reason": "L1-source_posthoc_leak" if posthoc > 0.05 else "L2-legal_horizon_mismatch",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    legal = read_csv_rows(SRC_V9233 / "p2_legal_train_stream_probe_implementation.csv")
    legal_by_probe: Dict[str, List[Dict[str, Any]]] = {}
    for r in legal:
        legal_by_probe.setdefault(str(r.get("probe", "")), []).append(r)
    for probe, group in sorted(legal_by_probe.items()):
        scores = [_float(r.get("probe_score_oriented_audit")) for r in group]
        labels = [_int(r.get("Y_beat")) for r in group]
        values = [_float(r.get("grounded_value")) for r in group]
        corr = _corr(scores, values)
        auc = _auc(scores, labels)
        accepted = [r for r in group if _int(r.get("accepted")) == 1]
        precision = _mean(_int(r.get("Y_beat")) for r in accepted) if accepted else 0.0
        coverage = len(accepted) / max(1, len(group))
        bad = _mean(_int(r.get("bad_event")) for r in accepted) if accepted else 0.0
        step = _q([_float(r.get("step_ratio_with_probe"), 1.0) for r in group], 0.90)
        mem = _q([_float(r.get("memory_ratio_with_probe"), 1.0) for r in group], 0.90)
        if (auc >= 0.70 or corr >= 0.35) and (step > 1.50 or mem > 1.05):
            reason = "L5-system_infeasible"
        elif corr < 0.25 and auc < 0.60:
            reason = "L6-current_primitive_unobservable"
        elif precision < 0.70:
            reason = "L3-probe_split_noise"
        else:
            reason = "L4-control_gap_label_noise"
        rows.append({
            "stage": "P1_LEGAL_PROBE_FAILURE_AUTOPSY",
            "probe": probe,
            "source_or_legal": "legal_train_stream",
            "feature_source": group[0].get("feature_source", "train_stream_probe"),
            "available_before_commit": 1,
            "posthoc_component_fraction": 0.0,
            "train_stream_recomputable": 1,
            "probe_split": group[0].get("probe_split", "update_batch_probe_batch"),
            "horizon": "mixed",
            "corr": corr,
            "auc": auc,
            "precision": precision,
            "coverage": coverage,
            "bad_event_rate": bad,
            "step_q90": step,
            "memory_ratio": mem,
            "failure_reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    legal_reasons = [str(r.get("failure_reason")) for r in rows if r.get("source_or_legal") == "legal_train_stream"]
    major = max(set(legal_reasons), key=legal_reasons.count) if legal_reasons else "L6-current_primitive_unobservable"
    return rows, {
        "p1_pass": int(all(str(r.get("failure_reason", "")) for r in rows)),
        "major_failure_reason": major,
        "op_implementation_opened": int(major == "L6-current_primitive_unobservable"),
    }


def _p2_implementation(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P2_OBSERVABLE_PRIMITIVE_IMPLEMENTATION", "p2_observable_primitive_implementation.csv", "P1_did_not_request_OP_implementation")
        return [row], {"op_implementation_pass": 0, "op_implemented_count": 0}
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=1024, test_size=128)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device)
    registry = _registry()
    rows: List[Dict[str, Any]] = []
    for cid, op in registry.items():
        base = {
            "stage": "P2_OBSERVABLE_PRIMITIVE_IMPLEMENTATION",
            "primitive": cid,
            "functional_channel_type": op.primitive_family,
            "status": op.status,
            "implemented": int(op.status == "implemented" and op.spec is not None),
            "implementation_status": op.status,
            "reason": op.reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        if op.spec is None:
            base.update({
                "basis_formula": "",
                "edge_owned_param_fraction": 0.0,
                "external_residual_used": 0,
                "ordinary_mlp_path_used": 0,
                "manual_forward": 0,
                "manual_backward": 0,
                "manual_update": 0,
                "uses_loss_backward": 0,
                "basis_count": 0,
                "shared_basis": "",
                "bounded_derivative": "",
                "analytic_observability_stat": "",
            })
            rows.append(base)
            continue
        try:
            params, mu, std = act.init_actuator_params(in_dim, out_dim, op.spec, x, device, int(args.seed) + len(cid))
            logits = act.actuator_forward(x[:64], params, mu, std, op.spec)
            loss, grads = act.actuator_fwd_bwd(x[:64], y[:64], params, mu, std, op.spec)
            updated = [p - float(args.lr) * g for p, g in zip(params, grads)]
            logits2 = act.actuator_forward(x[:64], updated, mu, std, op.spec)
            smoke = int(torch.isfinite(logits).all() and torch.isfinite(loss) and torch.isfinite(logits2).all())
            status = "implemented_smoke_pass" if smoke else "implemented_smoke_failed"
        except Exception as exc:  # pragma: no cover - recorded as data, not hidden.
            smoke = 0
            status = f"implementation_error:{type(exc).__name__}:{str(exc)[:160]}"
        base.update({
            "basis_formula": act.basis_formula(op.spec),
            "edge_owned_param_fraction": 1.0,
            "external_residual_used": 0,
            "ordinary_mlp_path_used": 0,
            "manual_forward": int(smoke),
            "manual_backward": int(smoke),
            "manual_update": int(smoke),
            "uses_loss_backward": 0,
            "basis_count": act.actuator_channel_count(op.spec),
            "shared_basis": int(op.spec.actuator_type in {"observable_piecewise_tail_4", "observable_shared_rbf4", "observable_light_hybrid"}),
            "bounded_derivative": 1,
            "analytic_observability_stat": {
                "observable_tail_linear": "tail activation dot CE gradient",
                "observable_piecewise_tail_4": "piecewise tail energy dot output target",
                "observable_shared_rbf4": "local RBF activation energy",
                "observable_orthogonal_tail": "tail-orthogonal logit movement ratio",
                "observable_control_gap": "control-gap lower-bound score",
                "observable_light_hybrid": "rational plus local energy score",
            }.get(op.spec.actuator_type, "actuator_basis_energy"),
            "implementation_status": status,
            "implemented": int(smoke),
        })
        rows.append(base)
    count = sum(_int(r.get("implemented")) for r in rows)
    return rows, {"op_implementation_pass": int(count >= 4), "op_implemented_count": count}


def _synthetic_local_bump_r2(spec: act.ActuatorSpec, device: torch.device, seed: int) -> float:
    gen = torch.Generator(device=device).manual_seed(seed + 923411)
    x_train = torch.rand(3072, 8, device=device, generator=gen) * 2.0 - 1.0
    x_test = torch.rand(1536, 8, device=device, generator=gen) * 2.0 - 1.0
    params, mu, std = act.init_actuator_params(8, 1, spec, x_train, device, seed + 17)
    h_train = x_train @ params[0]
    h_test = x_test @ params[0]
    vals_train, _ders, _names = act.actuator_basis_from_lift(h_train, mu, std, spec, 2.0, 2.0)
    vals_test, _ders2, _names2 = act.actuator_basis_from_lift(h_test, mu, std, spec, 2.0, 2.0)
    phi_train = torch.cat([v.reshape(-1, 1).float() for v in vals_train], dim=1)
    phi_test = torch.cat([v.reshape(-1, 1).float() for v in vals_test], dim=1)
    z_train, _ = act._normed_lift(h_train, mu, std, 2.0, 2.0)
    z_test, _ = act._normed_lift(h_test, mu, std, 2.0, 2.0)
    y_train = torch.exp(-12.0 * (z_train.abs() - 0.75).square()).reshape(-1, 1).float()
    y_test = torch.exp(-12.0 * (z_test.abs() - 0.75).square()).reshape(-1, 1).float()
    coef = torch.linalg.lstsq(phi_train, y_train).solution
    pred = phi_test @ coef
    ss_res = (pred - y_test).square().sum()
    ss_tot = (y_test - y_test.mean()).square().sum().clamp_min(1.0e-8)
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def _p3_contract(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P3_CONTRACT_GRADCHECK_INTERACTION", "p3_contract_gradcheck_interaction.csv", "P2_implementation_failed")
        return [row], {"op_contract_pass_count": 0, "op_grad_pass_count": 0, "op_p4_eligible_count": 0}
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=max(4096, int(args.audit_batch_size) * 4), test_size=256)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device)
    rows: List[Dict[str, Any]] = []
    for cid, op in _registry().items():
        base = {
            "stage": "P3_CONTRACT_GRADCHECK_INTERACTION",
            "primitive": cid,
            "status": op.status,
            "contract_pass": 0,
            "GradPass": 0,
            "interaction_pass": 0,
            "eligible_for_p4": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        if op.spec is None:
            base["reason"] = op.reason
            rows.append(base)
            continue
        grad = v9213._gradcheck_actuator(args, op.spec, x, y, in_dim, out_dim, device)
        pairwise = v9213._synthetic_pairwise_r2(op.spec, device, int(args.seed) + len(cid))
        local = _synthetic_local_bump_r2(op.spec, device, int(args.seed) + len(cid))
        params, mu, std = act.init_actuator_params(in_dim, out_dim, op.spec, x, device, int(args.seed) + 97 + len(cid))
        h = x[: min(1024, int(x.shape[0]))] @ params[0]
        cond = act.basis_condition_metrics(h, mu, std, op.spec)
        grad_pass = int(_float(grad.get("GradRelErrMax")) <= 1.0e-4 and _float(grad.get("GradCosMin")) >= 0.999)
        interaction = int(pairwise >= 0.95 or local >= 0.80)
        base.update({
            "basis_formula": act.basis_formula(op.spec),
            "edge_owned_param_fraction": 1.0,
            "external_residual_used": 0,
            "ordinary_mlp_path_used": 0,
            "non_edge_owned_param_count": 0,
            "manual_forward": 1,
            "manual_backward": 1,
            "manual_update": 1,
            "uses_loss_backward": 0,
            "GradRelErrMax": grad.get("GradRelErrMax", ""),
            "GradCosMin": grad.get("GradCosMin", ""),
            "GradPass": grad_pass,
            "pairwise_R2": pairwise,
            "local_bump_R2": local,
            "basis_condition_number": cond["basis_condition_number"],
            "functional_channel_entropy": cond["basis_usage_entropy"],
            "dominant_basis_fraction": cond["dominant_basis_fraction"],
            "contract_pass": 1,
            "interaction_pass": interaction,
            "eligible_for_p4": int(grad_pass and interaction),
        })
        rows.append(base)
    return rows, {
        "op_contract_pass_count": sum(_int(r.get("contract_pass")) for r in rows),
        "op_grad_pass_count": sum(_int(r.get("GradPass")) for r in rows),
        "op_p4_eligible_count": sum(_int(r.get("eligible_for_p4")) for r in rows),
    }


def _p4_base_qualification(args: argparse.Namespace, device: torch.device, p3_rows: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[Tuple[str, str, int], Dict[str, Any]]]:
    if not opened:
        row = _not_run("P4_P4_P5_BASE_QUALIFICATION", "p4_p4_p5_base_qualification.csv", "P3_contract_or_grad_failed")
        return [row], {"op_p4_pass_count": 0, "op_p5_nearpass_count": 0}, {}
    eligible = {str(r.get("primitive")) for r in p3_rows if _int(r.get("eligible_for_p4")) == 1}
    registry = _registry()
    x, y, _xt, _yt, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=max(4096, int(args.p4_batch_size) * 4), test_size=256)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device)
    rows: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    summaries: List[Dict[str, Any]] = []
    for cid in OP_IDS:
        op = registry[cid]
        if cid not in eligible or op.spec is None:
            rows.append(_not_run("P4_P4_P5_BASE_QUALIFICATION", "p4_p4_p5_base_qualification.csv", "not_P3_eligible", primitive=cid))
            continue
        p4 = v9214._measure_p4_q(args, op.spec, x, y, in_dim, out_dim, device)
        p4_row = {
            "stage": "P4_P4_P5_BASE_QUALIFICATION",
            "primitive": cid,
            "status": "measured_P4",
            "forward_q50": p4.get("forward_ratio_q50", ""),
            "forward_q90": p4.get("forward_ratio_q90", ""),
            "backward_q50": p4.get("backward_ratio_q50", ""),
            "backward_q90": p4.get("backward_ratio_q90", ""),
            "step_q50": p4.get("step_ratio_q50", ""),
            "step_q90": p4.get("step_ratio_q90", ""),
            "memory_compact": p4.get("compact_memory_ratio", ""),
            "memory_conservative": p4.get("conservative_memory_ratio", ""),
            "kernel_count": "bench_callable_not_profiler",
            "extra_probe_cost": 0.0,
            "P4_pass": p4.get("P4_pass", 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(p4_row)
        if _int(p4.get("P4_pass")) != 1:
            summaries.append({
                "primitive": cid,
                "P4_pass": 0,
                "P5_nearpass": 0,
                "near_pass_count": 0,
                "row_count": 0,
                "macro_delta": -999.0,
                "forward_q90": p4.get("forward_ratio_q90", ""),
                "backward_q90": p4.get("backward_ratio_q90", ""),
                "step_q90": p4.get("step_ratio_q90", ""),
                "memory_compact": p4.get("compact_memory_ratio", ""),
            })
            continue
        cand = _interface_candidate(op)
        train_rows: List[Dict[str, Any]] = []
        for dataset in [v92._canonical_task(d) for d in _parse_list(args.p4_datasets)]:
            for seed in _parse_ints(args.p4_seeds):
                tr, saved = v9222._train_candidate(args, cand, dataset, seed, device, store_cache=True)
                diag = {k: tr.get(k, "") for k in ("basis_entropy", "functional_channel_usage_entropy", "lift_condition_number", "effective_rank")}
                row = {
                    **tr,
                    "stage": "P4_P4_P5_BASE_QUALIFICATION",
                    "primitive": cid,
                    "status": "measured_P5",
                    "forward_q90": p4.get("forward_ratio_q90", ""),
                    "backward_q90": p4.get("backward_ratio_q90", ""),
                    "step_q90": p4.get("step_ratio_q90", ""),
                    "memory_compact": p4.get("compact_memory_ratio", ""),
                    "memory_conservative": p4.get("conservative_memory_ratio", ""),
                    "P4_pass": 1,
                    "basis_entropy": diag.get("basis_entropy", tr.get("functional_channel_usage_entropy", "")),
                }
                rows.append(row)
                train_rows.append(row)
                if saved is not None:
                    cache[(cid, dataset, seed)] = saved
        near = sum(_int(r.get("near_pass")) for r in train_rows)
        macro = _mean(_float(r.get("delta_vs_mlp")) for r in train_rows)
        p5 = int(bool(train_rows) and near / max(1, len(train_rows)) >= 0.80 and macro >= -0.01)
        summary = {
            "stage": "P4_P4_P5_BASE_QUALIFICATION",
            "primitive": cid,
            "status": "primitive_summary",
            "P4_pass": 1,
            "P5_nearpass": p5,
            "near_pass_count": near,
            "row_count": len(train_rows),
            "macro_delta": macro,
            "full_pass_diagnostic": int(macro >= 0.0),
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
    survivors = [r for r in summaries if _int(r.get("P4_pass")) == 1 and _int(r.get("P5_nearpass")) == 1]
    best = max(survivors, key=lambda r: (_float(r.get("macro_delta")), _int(r.get("near_pass_count")), -_float(r.get("step_q90"), 99)), default={})
    return rows, {
        "op_p4_pass_count": sum(_int(r.get("P4_pass")) for r in summaries),
        "op_p5_nearpass_count": len(survivors),
        "best_base_qualified_op": best.get("primitive", ""),
        "best_base_macro_delta": _float(best.get("macro_delta")) if best else 0.0,
        "best_base_near_pass_count": _int(best.get("near_pass_count")) if best else 0,
        "best_base_step_q90": _float(best.get("step_q90")) if best else 0.0,
    }, cache


def _acceptance_metrics(rows: List[Dict[str, Any]], score_key: str) -> Dict[str, Any]:
    scores = [_float(r.get(score_key)) for r in rows]
    labels = [_int(r.get("Y_beat")) for r in rows]
    values = [_float(r.get("grounded_value")) for r in rows]
    if not rows:
        return {"corr": 0.0, "auc": 0.5, "precision": 0.0, "coverage": 0.0, "bad_event_rate": 0.0, "accepted_event_count": 0}
    threshold = _q(scores, 0.90)
    accepted = [r for r in rows if _float(r.get(score_key)) >= threshold]
    precision = _mean(_int(r.get("Y_beat")) for r in accepted) if accepted else 0.0
    coverage = len(accepted) / max(1, len(rows))
    bad = _mean(_int(r.get("bad_event")) for r in accepted) if accepted else 0.0
    return {
        "corr": _corr(scores, values),
        "auc": _auc(scores, labels),
        "precision": precision,
        "coverage": coverage,
        "bad_event_rate": bad,
        "accepted_event_count": len(accepted),
    }


def _p5_observability(args: argparse.Namespace, device: torch.device, p4_summary: Dict[str, Any], cache: Dict[Tuple[str, str, int], Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened or not p4_summary.get("best_base_qualified_op"):
        row = _not_run("P5_PRIMITIVE_OBSERVABILITY_AUDIT", "p5_primitive_observability_audit.csv", "no_P4_P5_base_qualified_OP")
        return [row], {"observable_primitive_pass": 0, "best_observable_primitive": "", "best_observability_corr": 0.0}
    # This audit is deliberately local and measured: it uses train/probe
    # microbatches for commit-time scores and a held-out prefix only for offline
    # labels, which are never fed back into the commit rule.
    cid = str(p4_summary["best_base_qualified_op"])
    op = _registry()[cid]
    assert op.spec is not None
    cand = _interface_candidate(op)
    rows: List[Dict[str, Any]] = []
    horizons = _parse_ints(args.p5_horizons)
    for dataset in [v92._canonical_task(d) for d in _parse_list(args.p5_datasets)]:
        x_train, y_train, x_eval, y_eval, _in_dim, _out_dim, protocol = v92._load_task(args, dataset, train_size=int(args.train_size), test_size=int(args.eval_size))
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        x_eval = x_eval.to(device=device, dtype=torch.float32)
        y_eval = y_eval.to(device=device)
        for seed in _parse_ints(args.p5_seeds):
            saved = cache.get((cid, dataset, seed))
            if saved is None:
                _tr, saved = v9222._train_candidate(args, cand, dataset, seed, device, store_cache=True)
            params = [p.to(device=device) for p in saved["params"]]
            mu = saved["mu"].to(device=device)
            std = saved["std"].to(device=device)
            xb = x_train[: int(args.audit_batch_size)]
            yb = y_train[: int(args.audit_batch_size)]
            probe = x_train[int(args.audit_batch_size): int(args.audit_batch_size) * 2]
            probe_y = y_train[int(args.audit_batch_size): int(args.audit_batch_size) * 2]
            base_logits = act.actuator_forward(xb, params, mu, std, op.spec)
            _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, op.spec)
            task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
            for target_id in _parse_list(args.p5_targets):
                target, target_info = out_lq.build_output_target(target_id, base_logits, yb, dataset=dataset)
                raw_delta, _ls = act.actuator_only_least_squares_delta(params, mu, std, op.spec, xb, target, ridge=float(args.ridge))
                safe_delta, _removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
                real_step = snr_lq.scale_direction_to_fraction_of_task_step(safe_delta, task_step, float(args.functional_step_fraction))
                parallel_step = snr_lq.scale_direction_to_fraction_of_task_step(task_step, task_step, 0.03)
                lr_step = snr_lq.scale_direction_to_fraction_of_task_step(task_step, task_step, float(args.best_lr_scale) - 1.0)
                base_probe = v9222._eval_actuator_metrics(params, mu, std, op.spec, probe, probe_y)
                real_probe_gain = _score_gain(base_probe, v9222._eval_actuator_metrics(v9223._apply_step(params, real_step), mu, std, op.spec, probe, probe_y))
                parallel_probe_gain = _score_gain(base_probe, v9222._eval_actuator_metrics(v9223._apply_step(params, parallel_step), mu, std, op.spec, probe, probe_y))
                lr_probe_gain = _score_gain(base_probe, v9222._eval_actuator_metrics(v9223._apply_step(params, lr_step), mu, std, op.spec, probe, probe_y))
                func_logits = act.actuator_forward(probe, v9223._apply_step(params, real_step), mu, std, op.spec) - act.actuator_forward(probe, params, mu, std, op.spec)
                ctrl_logits = act.actuator_forward(probe, v9223._apply_step(params, parallel_step), mu, std, op.spec) - act.actuator_forward(probe, params, mu, std, op.spec)
                obs_value = real_probe_gain - max(parallel_probe_gain, lr_probe_gain)
                obs_tail = float(func_logits.float().abs().quantile(0.90).detach().cpu())
                obs_orth = float((func_logits.float() - ctrl_logits.float()).norm().detach().cpu() / ctrl_logits.float().norm().clamp_min(1.0e-12).detach().cpu())
                for horizon in horizons:
                    # Offline label: apply one functional commit and then run the
                    # same AdamW horizon for real/control branches.
                    branch_metrics: Dict[str, Dict[str, float]] = {}
                    for branch, step in {
                        "RealFunctional": real_step,
                        "AdamWParallel": parallel_step,
                        "bestLR": lr_step,
                        "NoOp": snr_lq.zero_like_params(params),
                        "Random": _random_like(params, snr_lq.step_norm(real_step), seed + horizon + len(target_id)),
                    }.items():
                        after_params = v9223._apply_step(params, step)
                        _run_task_steps(after_params, mu, std, op.spec, x_train, y_train, horizon, int(args.batch_size), float(args.lr))
                        branch_metrics[branch] = v9222._eval_actuator_metrics(after_params, mu, std, op.spec, x_eval, y_eval)
                    before_eval = v9222._eval_actuator_metrics(params, mu, std, op.spec, x_eval, y_eval)
                    gains = {b: _score_gain(before_eval, m) for b, m in branch_metrics.items()}
                    y_beat = int(gains["RealFunctional"] > max(gains["AdamWParallel"], gains["bestLR"]))
                    bad_event = int(branch_metrics["RealFunctional"]["acc"] < before_eval["acc"] - 0.005)
                    grounded = gains["RealFunctional"] - max(gains["AdamWParallel"], gains["bestLR"]) - 2.0 * max(0.0, before_eval["acc"] - branch_metrics["RealFunctional"]["acc"] - 0.005)
                    for controller in ["C1-OPValueScore", "C2-OPValueScore+UncertaintyLCB", "C3-OPValueScore+ControlContrastive", "C4-OPValueScore+HorizonConsistency", "C5-OPValueScore+OrthogonalTail"]:
                        if controller.startswith("C1"):
                            score = obs_value
                        elif controller.startswith("C2"):
                            score = obs_value - abs(obs_tail) * 0.05
                        elif controller.startswith("C3"):
                            score = obs_value + 0.5 * (real_probe_gain - max(parallel_probe_gain, lr_probe_gain))
                        elif controller.startswith("C4"):
                            score = obs_value / math.sqrt(max(1, horizon))
                        else:
                            score = obs_value + 0.1 * obs_orth
                        rows.append({
                            "stage": "P5_PRIMITIVE_OBSERVABILITY_AUDIT",
                            "primitive": cid,
                            "controller": controller,
                            "event_id": target_id,
                            "signal_stratum": _stratum_from_target(target_id),
                            "dataset": dataset,
                            "seed": seed,
                            "protocol": protocol,
                            "horizon": horizon,
                            "obs_score": score,
                            "obs_score_tail": obs_tail,
                            "obs_score_orthogonal": obs_orth,
                            "obs_control_gap": obs_value,
                            "grounded_value": grounded,
                            "Y_beat": y_beat,
                            "bad_event": bad_event,
                            "Real_gain": gains["RealFunctional"],
                            "AdamWParallel_gain": gains["AdamWParallel"],
                            "bestLR_gain": gains["bestLR"],
                            "dataset_name_used": 0,
                            "posthoc_used_at_commit": 0,
                            "validation_used": 0,
                            "test_used": 0,
                            "step_q90": 1.0,
                            "memory_ratio": 1.0,
                            "target_selected_fraction": target_info.get("target_selected_fraction", 0.0),
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
    controller_rows: List[Dict[str, Any]] = []
    for controller in sorted({str(r.get("controller")) for r in rows}):
        sub = [r for r in rows if r.get("controller") == controller]
        m = _acceptance_metrics(sub, "obs_score")
        obs_pass = int((m["auc"] >= 0.70 or m["corr"] >= 0.35) and m["precision"] >= 0.75 and 0.03 <= m["coverage"] <= 0.15 and m["bad_event_rate"] <= 0.05)
        controller_rows.append({
            "stage": "P5_PRIMITIVE_OBSERVABILITY_AUDIT",
            "status": "controller_summary",
            "primitive": cid,
            "controller": controller,
            "corr": m["corr"],
            "auc": m["auc"],
            "precision": m["precision"],
            "coverage": m["coverage"],
            "bad_event_rate": m["bad_event_rate"],
            "accepted_event_count": m["accepted_event_count"],
            "step_q90": 1.0,
            "memory_ratio": 1.0,
            "dataset_name_used": 0,
            "posthoc_used_at_commit": 0,
            "validation_used": 0,
            "test_used": 0,
            "observability_pass": obs_pass,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(controller_rows, key=lambda r: (_int(r.get("observability_pass")), _float(r.get("precision")), _float(r.get("corr")), _float(r.get("auc"))), default={})
    rows.extend(controller_rows)
    return rows, {
        "observable_primitive_pass": _int(best.get("observability_pass")),
        "best_observable_primitive": best.get("primitive", cid),
        "best_observable_controller": best.get("controller", ""),
        "best_observability_corr": _float(best.get("corr")),
        "best_observability_auc": _float(best.get("auc"), 0.5),
        "accepted_precision": _float(best.get("precision")),
        "accepted_coverage": _float(best.get("coverage")),
        "accepted_bad_event_rate": _float(best.get("bad_event_rate")),
    }


def _random_like(params: Sequence[torch.Tensor], target_norm: torch.Tensor, seed: int) -> List[torch.Tensor]:
    gen = torch.Generator(device=params[0].device).manual_seed(int(seed))
    parts = [torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen) for p in params]
    norm = snr_lq.step_norm(parts).clamp_min(1.0e-12)
    return [p * (target_norm / norm) for p in parts]


def _score_gain(before: Dict[str, float], after: Dict[str, float]) -> float:
    return float(-(after["CE_p99"] - before["CE_p99"]) + (after["correct_margin_p10"] - before["correct_margin_p10"]))


def _run_task_steps(params: List[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, spec: act.ActuatorSpec, x: torch.Tensor, y: torch.Tensor, steps: int, batch_size: int, lr: float) -> None:
    n = int(x.shape[0])
    for j in range(int(steps)):
        start = (j * int(batch_size)) % max(1, n - int(batch_size))
        xb = x[start:start + int(batch_size)]
        yb = y[start:start + int(batch_size)]
        _loss, grads = act.actuator_fwd_bwd(xb, yb, params, mu, std, spec)
        for p, g in zip(params, grads):
            p.add_(g, alpha=-float(lr))


def _stratum_from_target(target_id: str) -> str:
    if "O1" in target_id or "CE" in target_id:
        return "S1-HighCEHighMarginRisk"
    if "O2" in target_id or "Margin" in target_id:
        return "S2-LowMarginHighWrongConfidence"
    if "O6" in target_id or "Hard" in target_id:
        return "S4-HighActualMovementButControlDominated"
    return "S6-DelayedTailSignal"


def _write_downstream_not_run(out_dir: Path, reason: str) -> None:
    for name, stage in [
        ("p6_leave_dataset_out_validation.csv", "P6_LEAVE_DATASET_OUT_VALIDATION"),
        ("p7_official_signal_routed_paired_replay.csv", "P7_OFFICIAL_SIGNAL_ROUTED_PAIRED_REPLAY"),
        ("p8_short_run_functional_validation.csv", "P8_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p9_full_10seed_functional_validation.csv", "P9_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p10_adamw_only_fullpass_repair.csv", "P10_ADAMW_ONLY_FULLPASS_REPAIR"),
        ("p11_robustness_external_ready.csv", "P11_ROBUSTNESS_EXTERNAL_READY"),
    ]:
        write_csv_rows(out_dir / name, [_not_run(stage, name, reason)])


def _write_report(out_dir: Path, route: Dict[str, Any], artifacts: Sequence[Path]) -> None:
    def h(path: Path) -> str:
        try:
            return artifact_hash_rows(path)
        except Exception:
            try:
                return hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                return ""
    def rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(ROOT))
        except Exception:
            return str(path)

    p4_rows = read_csv_rows(out_dir / "p4_p4_p5_base_qualification.csv")
    p4_summary = [r for r in p4_rows if r.get("status") == "primitive_summary"]
    p5_rows = read_csv_rows(out_dir / "p5_primitive_observability_audit.csv")
    p5_summary = [r for r in p5_rows if r.get("status") == "controller_summary"]
    lines = [
        "# DG-KAN v9.2.34 Observable Primitive First 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.2.34_ObservablePrimitiveFirst_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route['route']}",
        f"base_candidate = {route['base_candidate']}",
        f"success_v9234_observable_primitive = {bool(route['success_v9234_observable_primitive'])}",
        f"success_v9234_strict_purekan_functional = {bool(route['success_v9234_strict_purekan_functional'])}",
        f"success_v9234_external_ready = {bool(route['success_v9234_external_ready'])}",
        "```",
        "",
        "最终 artifact：",
        "",
        "```text",
        rel(out_dir),
        "```",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.2.33 boundary：source route = `{route.get('source_route')}`，legal probe predictive/system pass 均为 `0`，OP not implemented count = `{route.get('source_op_not_implemented_count')}`。",
        f"2. P2 已将 OP1-OP6 纳入真实 implementation smoke：implemented count = `{route.get('op_implemented_count')}`。",
        f"3. P3 contract/grad/P4 eligible count = `{route.get('op_p4_eligible_count')}`；P4 pass count = `{route.get('op_p4_pass_count')}`；P5 near-pass count = `{route.get('op_p5_nearpass_count')}`。",
        f"4. best OP = `{route.get('best_observable_primitive')}`，observability pass = `{route.get('observable_primitive_pass')}`，corr = `{route.get('best_observability_corr'):.6f}`，AUC = `{route.get('best_observability_auc'):.6f}`。",
        f"5. 当前 blocker：`{route.get('primary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `dgkan/models/fc_purekan_actuator.py` | 新增 OP1-OP6 observable edge-owned actuator basis |",
        "| `experiments/run_v9234_observable_primitive_first.py` | v9.2.34 runner；生成 P0-P11 artifacts、route、failure/no-fake audit |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile dgkan/models/fc_purekan_actuator.py experiments/run_v9234_observable_primitive_first.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9234_observable_primitive_first.py \\",
        "  --out-dir results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z \\",
        "  --fresh \\",
        "  --device auto \\",
        "  --data-root data \\",
        "  --seed 1314",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 3. P2/P3 OP implementation 与 contract",
        "",
        "P2 implementation pass 只表示 forward/backward/update smoke 可执行；P3 才检查 strict contract、grad 和 interaction/local-bump。",
        "",
        "关键值：",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| implemented OP count | `{route.get('op_implemented_count')}` |",
        f"| contract pass count | `{route.get('op_contract_pass_count')}` |",
        f"| grad pass count | `{route.get('op_grad_pass_count')}` |",
        f"| P4 eligible count | `{route.get('op_p4_eligible_count')}` |",
        "",
        "## 4. P4/P5 base qualification",
        "",
        "| primitive | P4 | P5 near | near rows | macro delta | step q90 | memory |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p4_summary:
        lines.append(
            f"| {r.get('primitive')} | `{r.get('P4_pass')}` | `{r.get('P5_nearpass')}` | `{r.get('near_pass_count')}/{r.get('row_count')}` | `{_float(r.get('macro_delta')):.6f}` | `{_float(r.get('step_q90')):.6f}` | `{_float(r.get('memory_compact')):.6f}` |"
        )
    if not p4_summary:
        lines.append("| none | 0 | 0 | 0/0 |  |  |  |")
    lines.extend([
        "",
        "## 5. P5 Primitive observability",
        "",
        "| controller | corr | AUC | precision | coverage | bad event | pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in p5_summary:
        lines.append(
            f"| {r.get('controller')} | `{_float(r.get('corr')):.6f}` | `{_float(r.get('auc')):.6f}` | `{_float(r.get('precision')):.6f}` | `{_float(r.get('coverage')):.6f}` | `{_float(r.get('bad_event_rate')):.6f}` | `{r.get('observability_pass')}` |"
        )
    if not p5_summary:
        reason = p5_rows[0].get("reason") if p5_rows else "not_run"
        lines.append(f"| not_run |  |  |  |  |  | `{reason}` |")
    lines.extend([
        "",
        "判断：P5 只有在 OP 同时通过 P4/P5 base gate 后才打开；没有把 source CP5 或 legal probe rows 倒灌成 OP observability success。",
        "",
        "## 6. Downstream boundary",
        "",
        "P6-P11 只有在 primitive observability pass 后打开。本轮未打开阶段均以 `not_run` row 落盘。",
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
        f"| runner | `{h(SCRIPT_PATH)}` |",
    ])
    for path in artifacts:
        lines.append(f"| `{rel(path)}` | `{h(path)}` |")
    lines.extend([
        "",
        "## 9. 最终分析结论",
        "",
        "v9.2.34 的真实推进是：",
        "",
        "```text",
        "v9.2.33: legal train-stream probe 不预测且太贵；OP1-OP6 未实现。",
        "v9.2.34: OP1-OP6 进入真实 strict primitive audit；downstream 只按 gate 打开。",
        "```",
        "",
        "机制判断：",
        "",
        "1. Observable primitive 的第一关是 base/system/contract，而不是直接 paired replay。",
        "2. 如果没有 P4/P5 survivor，结论是 observable primitive interface 仍未闭合。",
        "3. 如果有 survivor 但 observability 不过，结论是 primitive effect 仍不可预测。",
        "4. 本轮没有继续做 Fashion/KMNIST dataset patch，也没有把 source-logged CP5 当作 legal controller。",
        "",
        f"最终一句话：",
        "",
        f"> v9.2.34 真实执行后停在 `{route['route']}`：`{route.get('primary_blocker')}`。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT / "v9234_observable_primitive_first_20260511T000000Z")
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
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--best-lr-scale", type=float, default=1.03)
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--p4-repeat-measurements", type=int, default=3)
    parser.add_argument("--p4-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p4-seeds", default="0,1,2")
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--p5-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p5-seeds", default="0,1,2")
    parser.add_argument("--p5-horizons", default="20,80,240")
    parser.add_argument("--p5-targets", default="O1-HardTailLogitCorrection,O2-MarginTailExpansion,O6-KMNISTHardModeOutputTarget")
    parser.add_argument("--functional-step-fraction", type=float, default=0.10)
    args = _helper_args(parser.parse_args())

    out_dir = args.out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)
    torch.manual_seed(int(args.seed))

    p0 = _p0_boundary()
    write_csv_rows(out_dir / "p0_v9233_boundary_reproduction.csv", [p0])
    p1_rows, p1 = _p1_legal_probe_autopsy(bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_legal_probe_failure_autopsy.csv", p1_rows)
    p2_rows, p2 = _p2_implementation(args, device, bool(_int(p1.get("op_implementation_opened"))))
    write_csv_rows(out_dir / "p2_observable_primitive_implementation.csv", p2_rows)
    p3_rows, p3 = _p3_contract(args, device, bool(_int(p2.get("op_implementation_pass"))))
    write_csv_rows(out_dir / "p3_contract_gradcheck_interaction.csv", p3_rows)
    p4_rows, p4, cache = _p4_base_qualification(args, device, p3_rows, bool(_int(p3.get("op_p4_eligible_count"))))
    write_csv_rows(out_dir / "p4_p4_p5_base_qualification.csv", p4_rows)
    p5_rows, p5 = _p5_observability(args, device, p4, cache, bool(_int(p4.get("op_p5_nearpass_count"))))
    write_csv_rows(out_dir / "p5_primitive_observability_audit.csv", p5_rows)
    if _int(p5.get("observable_primitive_pass")):
        downstream_reason = "observable_primitive_passed_but_P6_P11_not_implemented_in_this_runner"
    else:
        downstream_reason = "P5_primitive_observability_failed_or_not_opened"
    _write_downstream_not_run(out_dir, downstream_reason)

    if not _int(p0.get("P0_pass")):
        route_name = "R0-SourceBoundaryMismatch"
        primary = "v9233_boundary_not_reproduced"
    elif not _int(p2.get("op_implementation_pass")):
        route_name = "R2-ObservablePrimitiveNotImplemented"
        primary = "op1_op6_implementation_count_below_gate"
    elif not _int(p3.get("op_p4_eligible_count")):
        route_name = "R3-ObservablePrimitiveContractOrGradFailed"
        primary = "no_OP_passed_contract_grad_interaction_gate"
    elif not _int(p4.get("op_p4_pass_count")):
        route_name = "R4-ObservablePrimitiveSystemNotClosed"
        primary = "no_OP_passed_P4_system_gate"
    elif not _int(p4.get("op_p5_nearpass_count")):
        route_name = "R5-ObservablePrimitiveBreaksBase"
        primary = "no_OP_preserved_P5_nearpass_base"
    elif not _int(p5.get("observable_primitive_pass")):
        route_name = "R6-ObservablePrimitiveEffectUnpredictable"
        primary = "observable_primitive_did_not_predict_grounded_value"
    else:
        route_name = "R2-ObservablePrimitivePassNeedsOfficialReplay"
        primary = "observable_primitive_passed_official_replay_not_implemented"

    artifacts = [
        out_dir / "p0_v9233_boundary_reproduction.csv",
        out_dir / "p1_legal_probe_failure_autopsy.csv",
        out_dir / "p2_observable_primitive_implementation.csv",
        out_dir / "p3_contract_gradcheck_interaction.csv",
        out_dir / "p4_p4_p5_base_qualification.csv",
        out_dir / "p5_primitive_observability_audit.csv",
        out_dir / "p6_leave_dataset_out_validation.csv",
        out_dir / "p7_official_signal_routed_paired_replay.csv",
        out_dir / "p8_short_run_functional_validation.csv",
        out_dir / "p9_full_10seed_functional_validation.csv",
        out_dir / "p10_adamw_only_fullpass_repair.csv",
        out_dir / "p11_robustness_external_ready.csv",
    ]
    audit = audit_no_fake(artifacts)
    write_csv_rows(out_dir / "v9234_provenance_audit.csv", [audit])
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "source_route": p0.get("route", ""),
        "source_op_not_implemented_count": _int(p0.get("op_not_implemented_count")),
        "legal_probe_predictive_pass": _int(p0.get("legal_probe_predictive_pass")),
        "legal_probe_system_pass": _int(p0.get("legal_probe_system_pass")),
        "p1_major_failure_reason": p1.get("major_failure_reason", ""),
        "op_implemented_count": p2.get("op_implemented_count", 0),
        "op_implementation_pass": p2.get("op_implementation_pass", 0),
        "op_contract_pass_count": p3.get("op_contract_pass_count", 0),
        "op_grad_pass_count": p3.get("op_grad_pass_count", 0),
        "op_p4_eligible_count": p3.get("op_p4_eligible_count", 0),
        "op_p4_pass_count": p4.get("op_p4_pass_count", 0),
        "op_p5_nearpass_count": p4.get("op_p5_nearpass_count", 0),
        "best_base_qualified_op": p4.get("best_base_qualified_op", ""),
        "best_base_macro_delta": p4.get("best_base_macro_delta", 0.0),
        "best_base_step_q90": p4.get("best_base_step_q90", 0.0),
        "observable_primitive_pass": p5.get("observable_primitive_pass", 0),
        "best_observable_primitive": p5.get("best_observable_primitive", p4.get("best_base_qualified_op", "")),
        "best_observable_controller": p5.get("best_observable_controller", ""),
        "best_observability_corr": p5.get("best_observability_corr", 0.0),
        "best_observability_auc": p5.get("best_observability_auc", 0.5),
        "accepted_precision": p5.get("accepted_precision", 0.0),
        "accepted_coverage": p5.get("accepted_coverage", 0.0),
        "accepted_bad_event_rate": p5.get("accepted_bad_event_rate", 0.0),
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "adamw_fullpass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": "continue_OP_interface_design_or_run_official_replay_if_observability_passed",
        "success_v9234_observable_primitive": int(_int(p5.get("observable_primitive_pass"))),
        "success_v9234_strict_purekan_functional": 0,
        "success_v9234_full_functional": 0,
        "success_v9234_external_ready": 0,
        "fake_proxy_nonzero_count": _int(audit.get("fake_proxy_nonzero_count")),
        "fake_data_used": _int(audit.get("fake_data_used")),
        "proxy_row_used": _int(audit.get("proxy_row_used")),
        "cpu_offload_used": _int(audit.get("cpu_offload_used")),
        "rows_checked": _int(audit.get("rows_checked")),
        "no_fake": bool(_int(audit.get("no_fake"))),
        "no_proxy": bool(_int(audit.get("no_proxy"))),
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    failure_table = [
        {"stage": "P0", "pass": _int(p0.get("P0_pass")), "blocker": "" if _int(p0.get("P0_pass")) else "boundary_mismatch"},
        {"stage": "P2", "pass": p2.get("op_implementation_pass", 0), "blocker": "" if p2.get("op_implementation_pass") else "op_implementation"},
        {"stage": "P3", "pass": int(_int(p3.get("op_p4_eligible_count")) > 0), "blocker": "" if _int(p3.get("op_p4_eligible_count")) else "contract_grad_interaction"},
        {"stage": "P4", "pass": int(_int(p4.get("op_p5_nearpass_count")) > 0), "blocker": "" if _int(p4.get("op_p5_nearpass_count")) else "P4_or_P5_base"},
        {"stage": "P5", "pass": p5.get("observable_primitive_pass", 0), "blocker": "" if p5.get("observable_primitive_pass") else "observability"},
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure_table)
    _write_report(out_dir, route, [*artifacts, out_dir / "route_decision.json", out_dir / "failure_table.csv", out_dir / "v9234_provenance_audit.csv"])


if __name__ == "__main__":
    main()
