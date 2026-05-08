#!/usr/bin/env python3
"""DG-KAN v8.3 system-gated functional update re-entry runner.

This runner deliberately keeps functional update behind the system gate from
the v8.3 plan.  It reuses the measured v8.2 CE-only/no-teacher/no-loss path to
confirm base candidates, then runs graph-free one-step functional direction
diagnostics.  It does not open full functional task training unless the base
gate is satisfied by measured artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import run_gafu_v72_real as v72
import run_gafu_v80_real as v80
import run_gafu_v81_real as v81
import run_gafu_v82_real as v82
from dgkan_core import get_device, parse_int_list, parse_str_list, save_json, set_seed, write_csv
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v8.3_SystemGated_FunctionalUpdate_ReEntry_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v83_real.py"
METRIC_UNAVAILABLE = v81.METRIC_UNAVAILABLE

BASE_CANDIDATES = ["B0", "KC6", "KW3", "KW4", "KW5", "KW6", "KF4", "KF10"]
FUNCTIONAL_CANDIDATES = [
    ("F0", "ManualAdamWEquivalentBaseline", "task_direction_baseline", "diagnostic_baseline"),
    ("F1", "FunctionalDiag", "curvature_diag_direction", "naive_functional_diagnostic"),
    ("F2", "FunctionalDataDiag", "data_weighted_curvature_diag_direction", "naive_functional_diagnostic"),
    ("F3", "FunctionalResidualSobolev", "smoothness_plus_curvature_direction", "naive_functional_diagnostic"),
    ("F4", "FunctionalCoordinateAdam", "coordinate_normalized_curvature_direction", "naive_functional_diagnostic"),
    ("F5", "FunctionalCoordinateAdanLite", "mixed_coordinate_curvature_direction", "naive_functional_diagnostic"),
    ("FT0", "TaskProjectedFunctionalDiag", "projected_curvature_diag_correction", "task_aware_functional"),
    ("FT1", "TaskProjectedDataSobolev", "projected_data_weighted_curvature_correction", "task_aware_functional"),
    ("FT2", "TaskProjectedResidualSobolev", "projected_smoothness_curvature_correction", "task_aware_functional"),
    ("FT3", "TrustRegionFunctionalCoordinateAdam", "trust_region_coordinate_curvature_correction", "task_aware_functional"),
    ("FT4", "TrustRegionFunctionalCoordinateAdanLite", "trust_region_mixed_coordinate_correction", "task_aware_functional"),
    ("FT5", "EventTriggeredFunctionalCorrection", "event_triggered_projected_curvature_correction", "task_aware_functional"),
    ("FT6", "LatePhaseFunctionalCorrection", "late_phase_projected_curvature_correction", "task_aware_functional"),
    ("FT7", "RoleWiseFunctionalCorrection", "rolewise_projected_curvature_correction", "task_aware_functional"),
    ("FR0", "PureFunctionalDiag", "pure_curvature_diag_direction", "exploratory_pure_functional"),
    ("FR1", "PureFunctionalCoordinateAdam", "pure_coordinate_normalized_curvature_direction", "exploratory_pure_functional"),
    ("FR2", "PureFunctionalCoordinateAdanLite", "pure_mixed_coordinate_curvature_direction", "exploratory_pure_functional"),
    ("FR3", "PureFunctionalNaturalDiag", "pure_data_weighted_curvature_natural_diag_direction", "exploratory_pure_functional"),
]
STREAMED_FUNCTIONAL_OPERATOR_CANDIDATES = {"FT5", "FT6", "FT7"}
FT7_ROLE_WEIGHTS = {"stack": 0.75, "head": 1.0}
FT7_ROLE_BUDGETS = {"stack": 0.15, "head": 0.15}
P5_FT7_EVENT_STRIDE = 8
P5_FT7_EVENT_ALPHA_MULT = 15.0
FT7_TRACK_DIAGNOSTIC_NORM = False
FT7_FAST_TASK_CHECK = False
FT7_USE_HOLDOUT_GUARD = True
FT7_USE_PRE_HOLDOUT_BASELINE = True
FT7_USE_JOINT_EVENT_GUARD = False
FT7_JOINT_EVENT_BUDGET = 0.05
FT7_PRE_HOLDOUT_EVERY = 2
FT7_NO_PRE_HOLDOUT_TRAIN_BUDGET_MULT = 0.25
P9_TRAIN_SIZES = [256, 512, 1024, 1536, 4096]
P9_LABEL_NOISE_LEVELS = [0.05, 0.10, 0.20]
P9_INPUT_NOISE_LEVELS = [0.05, 0.10]


def _ft7_uses_pre_holdout_for_step(step: int) -> bool:
    if not FT7_USE_PRE_HOLDOUT_BASELINE:
        return False
    every = max(1, int(FT7_PRE_HOLDOUT_EVERY))
    event_index = max(1, int(step) // max(1, int(P5_FT7_EVENT_STRIDE)))
    return (event_index - 1) % every == 0


def _ft7_holdout_budget_base(
    holdout_loss_before: float | None,
    holdout_loss_after: float,
    train_descent: float,
) -> float:
    if FT7_USE_PRE_HOLDOUT_BASELINE and holdout_loss_before is not None and holdout_loss_before > 0.0:
        return max(0.0, holdout_loss_before - holdout_loss_after)
    return FT7_NO_PRE_HOLDOUT_TRAIN_BUDGET_MULT * max(0.0, train_descent)


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "metric_unavailable", "not_run", "not_applicable"}:
            return default
        return float(text)
    except Exception:
        return default


def _is_one(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _hash_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _h0_pass(row: Dict[str, Any]) -> int:
    return int(
        _is_one(row.get("NoTeacherNoLossModificationPass"))
        and _is_one(row.get("StrictPass"))
        and _is_one(row.get("GradPass"))
        and _float(row.get("macro_gap"), -99.0) >= 0.0200
        and _float(row.get("memory_ratio_max"), 99.0) <= 1.05
        and _float(row.get("step_ratio_max"), 99.0) <= 1.50
    )


def _base_rows(out_dir: Path) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(out_dir / "p8_official_coselection.csv") or _read_csv_rows(out_dir / "p6_official_coselection.csv")
    out = []
    for row in rows:
        cid = str(row.get("candidate_id"))
        if cid == "B0" or cid in set(BASE_CANDIDATES) or cid.startswith(("KC", "KF", "KW")):
            out.append(row)
    return out


def _select_audit_base(out_dir: Path) -> str:
    rows = [row for row in _base_rows(out_dir) if str(row.get("candidate_id")) != "B0"]
    if not rows:
        return "KW3"
    h0_rows = [row for row in rows if _h0_pass(row)]
    pool = h0_rows or rows
    best = max(
        pool,
        key=lambda r: (
            int(_h0_pass(r)),
            int(_float(r.get("macro_gap"), -99.0) >= 0.0200),
            -max(0.0, _float(r.get("memory_ratio_max"), 99.0) - 1.05),
            -max(0.0, _float(r.get("step_ratio_max"), 99.0) - 1.50),
            _float(r.get("macro_gap"), -99.0),
        ),
    )
    return str(best.get("candidate_id"))


def _write_candidate_registry(out_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for cid in BASE_CANDIDATES:
        rows.append({
            "stage": "P0_CANDIDATE_REGISTRY_V83",
            "candidate_id": cid,
            "base_candidate_id": cid,
            "candidate_role": "base_system_candidate" if cid != "B0" else "baseline_mlp_reference",
            "functional_update_used": 0,
            "functional_update_type": "none",
            "functional_update_is_update_rule": 0,
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "special_loss_used": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "teacher_logits_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "cpu_offload_used": 0,
            "manual_forward": int(cid != "B0"),
            "manual_backward": int(cid != "B0"),
            "manual_update": int(cid != "B0"),
            "uses_loss_backward": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    for cid, name, update_type, family in FUNCTIONAL_CANDIDATES:
        rows.append({
            "stage": "P0_CANDIDATE_REGISTRY_V83",
            "candidate_id": cid,
            "candidate_name": name,
            "base_candidate_id": "system_gate_selected_base",
            "candidate_role": family,
            "functional_update_used": int(cid != "F0"),
            "functional_update_type": update_type,
            "functional_update_is_update_rule": 1,
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "special_loss_used": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "teacher_logits_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "cpu_offload_used": 0,
            "nonKAN_param_count": 0,
            "manual_forward": 1,
            "manual_backward": 1,
            "manual_update": 1,
            "uses_loss_backward": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "candidate_registry_v83_functional.csv", rows)


def _write_contracts(out_dir: Path) -> None:
    registry = _read_csv_rows(out_dir / "candidate_registry_v83_functional.csv")
    rows = []
    for row in registry:
        contract_pass = int(
            str(row.get("loss_type")) == "CE"
            and not _is_one(row.get("geometry_loss_used"))
            and not _is_one(row.get("special_loss_used"))
            and not _is_one(row.get("external_teacher_used"))
            and not _is_one(row.get("self_teacher_used"))
            and not _is_one(row.get("teacher_logits_used"))
            and not _is_one(row.get("sampler_changed"))
            and not _is_one(row.get("class_weight_used"))
            and not _is_one(row.get("cpu_offload_used"))
        )
        rows.append({
            **row,
            "stage": "P0_CONTRACT_AUDIT_V83",
            "contract_pass": contract_pass,
            "no_teacher_no_loss_pass": contract_pass,
        })
    write_csv(out_dir / "contract_no_teacher_no_loss_functional.csv", rows)
    write_csv(out_dir / "p0_contract_audit.csv", rows)
    func_rows = [
        {
            **row,
            "stage": "P0_FUNCTIONAL_UPDATE_CONTRACT_V83",
            "functional_contract_pass": int(
                _is_one(row.get("functional_update_is_update_rule"))
                and str(row.get("loss_type")) == "CE"
                and not _is_one(row.get("geometry_loss_used"))
                and not _is_one(row.get("uses_loss_backward"))
                and not _is_one(row.get("cpu_offload_used"))
            ),
        }
        for row in rows
        if str(row.get("candidate_id")).startswith("F")
    ]
    write_csv(out_dir / "functional_update_contract.csv", func_rows)


def _write_p1_base_confirmation(out_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in _base_rows(out_dir):
        cid = str(row.get("candidate_id"))
        rows.append({
            **row,
            "stage": "P1_BASE_CANDIDATE_CONFIRMATION_V83",
            "base_candidate_id": cid,
            "H0_system_base_pass": _h0_pass(row),
            "H0_macro_gate": int(_float(row.get("macro_gap"), -99.0) >= 0.0200),
            "H0_memory_gate": int(_float(row.get("memory_ratio_max"), 99.0) <= 1.05),
            "H0_step_gate": int(_float(row.get("step_ratio_max"), 99.0) <= 1.50),
            "cpu_offload_used": row.get("cpu_offload_used", 0),
            "fake_data_used": row.get("fake_data_used", 0),
            "proxy_row_used": row.get("proxy_row_used", 0),
        })
    if not rows:
        rows.append({
            "stage": "P1_BASE_CANDIDATE_CONFIRMATION_V83",
            "status": "missing_base_measurement",
            "H0_system_base_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p1_base_candidate_confirmation.csv", rows)
    return rows


def _param_entries(stack: Any, head: Any) -> List[Tuple[str, str, torch.Tensor, torch.Tensor]]:
    entries: List[Tuple[str, str, torch.Tensor, torch.Tensor]] = []
    for provider_role, provider in [("stack", stack), ("head", head)]:
        for name, param, grad in provider.params_and_grads():
            role = "head" if provider_role == "head" or "head" in str(name) else "stack"
            entries.append((role, f"{provider_role}:{name}", param, grad))
    return entries


def _entries_by_role(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
) -> Dict[str, List[Tuple[str, str, torch.Tensor, torch.Tensor]]]:
    return {
        role_name: [entry for entry in entries if entry[0] == role_name]
        for role_name in ("stack", "head")
    }


def _zero_grad(stack: Any, head: Any) -> None:
    stack.zero_grad()
    head.zero_grad()


def _clone_params(entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
    return [param.detach().clone() for _role, _name, param, _grad in entries]


def _restore_params(entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]], snapshot: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for (_role, _name, param, _grad), snap in zip(entries, snapshot):
            param.copy_(snap)


def _sum_sq(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.tensor(0.0)
    out = torch.zeros((), device=tensors[0].device, dtype=torch.float64)
    for tensor in tensors:
        out = out + tensor.detach().double().pow(2).sum()
    return out


def _dot(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> torch.Tensor:
    if not xs:
        return torch.tensor(0.0)
    out = torch.zeros((), device=xs[0].device, dtype=torch.float64)
    for x, y in zip(xs, ys):
        out = out + (x.detach().double() * y.detach().double()).sum()
    return out


def _norm(tensors: Sequence[torch.Tensor]) -> float:
    return float(torch.sqrt(_sum_sq(tensors).clamp_min(0.0)).detach().cpu())


def _cos(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> float:
    na = _sum_sq(a).sqrt()
    nb = _sum_sq(b).sqrt()
    denom = (na * nb).clamp_min(1e-30)
    return float((_dot(a, b) / denom).detach().cpu())


def _param_norm(entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]]) -> float:
    return _norm([param for _role, _name, param, _grad in entries])


def _first_diff_grad(param: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros_like(param)
    if param.numel() < 2:
        return grad
    dim = param.dim() - 1
    if param.shape[dim] < 2:
        return grad
    diff = param.diff(dim=dim)
    left = [slice(None)] * param.dim()
    right = [slice(None)] * param.dim()
    left[dim] = slice(0, -1)
    right[dim] = slice(1, None)
    grad[tuple(left)] -= diff
    grad[tuple(right)] += diff
    return grad


def _second_diff_grad(param: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros_like(param)
    if param.numel() < 3:
        return grad
    dim = param.dim() - 1
    if param.shape[dim] < 3:
        return grad
    left = [slice(None)] * param.dim()
    mid = [slice(None)] * param.dim()
    right = [slice(None)] * param.dim()
    left[dim] = slice(0, -2)
    mid[dim] = slice(1, -1)
    right[dim] = slice(2, None)
    second = param[tuple(right)] - 2.0 * param[tuple(mid)] + param[tuple(left)]
    grad[tuple(left)] += second
    grad[tuple(mid)] -= 2.0 * second
    grad[tuple(right)] += second
    return grad


def _geometry_norms(entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]]) -> Tuple[float, float]:
    smooth = torch.zeros((), device=entries[0][2].device if entries else "cpu", dtype=torch.float64)
    curv = torch.zeros((), device=entries[0][2].device if entries else "cpu", dtype=torch.float64)
    for _role, _name, param, _grad in entries:
        p = param.detach().double()
        if p.numel() >= 2 and p.shape[-1] >= 2:
            smooth = smooth + p.diff(dim=p.dim() - 1).pow(2).sum()
        if p.numel() >= 3 and p.shape[-1] >= 3:
            second = p[..., 2:] - 2.0 * p[..., 1:-1] + p[..., :-2]
            curv = curv + second.pow(2).sum()
    return float(smooth.detach().cpu()), float(curv.detach().cpu())


def _smooth_ce_value_only(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    if float(label_smoothing) <= 0.0:
        return F.cross_entropy(logits, y)
    logp = F.log_softmax(logits, dim=1)
    classes = int(logits.shape[1])
    eps = float(label_smoothing)
    target = torch.full_like(logits, eps / max(1, classes - 1))
    target.scatter_(1, y.view(-1, 1), 1.0 - eps)
    return -(target * logp).sum(dim=1).mean()


def _loss_only(stack: Any, head: Any, x: torch.Tensor, y: torch.Tensor, spec: Any) -> float:
    with torch.inference_mode():
        h, caches = stack.forward_manual(x)
        logits, head_cache = head.forward_manual(h)
        loss = _smooth_ce_value_only(logits, y, spec.label_smoothing)
        del caches, head_cache, logits
        return float(loss.detach().cpu())


def _loss_and_features_only(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: Any,
) -> Tuple[float, torch.Tensor]:
    with torch.inference_mode():
        h, caches = stack.forward_manual(x)
        logits, head_cache = head.forward_manual(h)
        loss = _smooth_ce_value_only(logits, y, spec.label_smoothing)
        del caches, head_cache, logits
        return float(loss.detach().cpu()), h.detach()


def _loss_pair_and_features_only(
    stack: Any,
    head: Any,
    x_a: torch.Tensor,
    y_a: torch.Tensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    spec: Any,
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        split = int(x_a.shape[0])
        x = torch.cat((x_a, x_b), dim=0)
        h, caches = stack.forward_manual(x)
        logits, head_cache = head.forward_manual(h)
        loss_a = _smooth_ce_value_only(logits[:split], y_a, spec.label_smoothing)
        loss_b = _smooth_ce_value_only(logits[split:], y_b, spec.label_smoothing)
        h_a = h[:split].detach()
        h_b = h[split:].detach()
        del caches, head_cache, logits, x, h
        return float(loss_a.detach().cpu()), float(loss_b.detach().cpu()), h_a, h_b


def _loss_pair_only(
    stack: Any,
    head: Any,
    x_a: torch.Tensor,
    y_a: torch.Tensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    spec: Any,
) -> Tuple[float, float]:
    with torch.inference_mode():
        split = int(x_a.shape[0])
        x = torch.cat((x_a, x_b), dim=0)
        h, caches = stack.forward_manual(x)
        logits, head_cache = head.forward_manual(h)
        loss_a = _smooth_ce_value_only(logits[:split], y_a, spec.label_smoothing)
        loss_b = _smooth_ce_value_only(logits[split:], y_b, spec.label_smoothing)
        del caches, head_cache, logits, x, h
        return float(loss_a.detach().cpu()), float(loss_b.detach().cpu())


def _head_loss_from_features(head: Any, h: torch.Tensor, y: torch.Tensor, spec: Any) -> float:
    with torch.inference_mode():
        logits, head_cache = head.forward_manual(h)
        loss = _smooth_ce_value_only(logits, y, spec.label_smoothing)
        del head_cache, logits
        return float(loss.detach().cpu())


def _head_loss_pair_from_features(
    head: Any,
    h_a: torch.Tensor,
    y_a: torch.Tensor,
    h_b: torch.Tensor,
    y_b: torch.Tensor,
    spec: Any,
) -> Tuple[float, float]:
    with torch.inference_mode():
        split = int(h_a.shape[0])
        h = torch.cat((h_a, h_b), dim=0)
        logits, head_cache = head.forward_manual(h)
        loss_a = _smooth_ce_value_only(logits[:split], y_a, spec.label_smoothing)
        loss_b = _smooth_ce_value_only(logits[split:], y_b, spec.label_smoothing)
        del head_cache, logits, h
        return float(loss_a.detach().cpu()), float(loss_b.detach().cpu())


def _logits_only(stack: Any, head: Any, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    with torch.inference_mode():
        for xb in x.split(int(batch_size)):
            h, _ = stack.forward_manual(xb)
            logits, _ = head.forward_manual(h)
            parts.append(logits.detach())
    return torch.cat(parts, dim=0)


def _kl_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    with torch.no_grad():
        log_p = torch.log_softmax(p_logits, dim=-1)
        log_q = torch.log_softmax(q_logits, dim=-1)
        p = torch.softmax(p_logits, dim=-1)
        return float((p * (log_p - log_q)).sum(dim=-1).mean().detach().cpu())


def _manual_task_grads(stack: Any, head: Any, x: torch.Tensor, y: torch.Tensor, spec: Any) -> Tuple[float, List[torch.Tensor]]:
    _zero_grad(stack, head)
    h, caches = stack.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss, grad_logits = v72._weighted_smooth_ce_and_grad(logits, y, spec.label_smoothing, None)
    dh = head.backward_manual(grad_logits, head_cache)
    stack.backward_manual(dh, caches)
    entries = _param_entries(stack, head)
    task_dirs = [-grad.detach().clone() for _role, _name, _param, grad in entries]
    return float(loss.detach().cpu()), task_dirs


def _manual_ce_backward(stack: Any, head: Any, x: torch.Tensor, y: torch.Tensor, spec: Any) -> float:
    _zero_grad(stack, head)
    h, caches = stack.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss, grad_logits = v72._weighted_smooth_ce_and_grad(logits, y, spec.label_smoothing, None)
    dh = head.backward_manual(grad_logits, head_cache)
    stack.backward_manual(dh, caches)
    return float(loss.detach().cpu())


def _base_functional_dirs(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    task_dirs: Sequence[torch.Tensor],
) -> Dict[str, List[torch.Tensor]]:
    curv = [-_second_diff_grad(param.detach()) for _role, _name, param, _grad in entries]
    smooth = [-_first_diff_grad(param.detach()) for _role, _name, param, _grad in entries]
    data_weighted = []
    coord = []
    mixed = []
    for f, t, s in zip(curv, task_dirs, smooth):
        scale = t.abs() / t.abs().mean().clamp_min(1e-12)
        scale = scale.clamp(0.25, 4.0)
        fw = f * scale
        data_weighted.append(fw)
        coord.append(f / f.abs().sqrt().clamp_min(1e-6))
        mixed.append(0.7 * f + 0.3 * s)
    return {
        "curv": curv,
        "data": data_weighted,
        "sobolev": mixed,
        "coord": coord,
        "adan": [0.6 * c + 0.4 * d for c, d in zip(coord, data_weighted)],
    }


def _project_task_aware(
    task_dirs: Sequence[torch.Tensor],
    func_dirs: Sequence[torch.Tensor],
    *,
    trust_ratio: float,
    role_weights: Dict[str, float] | None,
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
) -> Tuple[List[torch.Tensor], float, float]:
    task_norm_sq = _sum_sq(task_dirs).clamp_min(1e-30)
    dot_tf = _dot(func_dirs, task_dirs)
    if dot_tf < 0:
        coeff = dot_tf / task_norm_sq
        projected = [f - coeff.to(f.device, f.dtype) * t for f, t in zip(func_dirs, task_dirs)]
    else:
        projected = [f.detach().clone() for f in func_dirs]
    if role_weights:
        projected = [
            p * float(role_weights.get(role, 1.0))
            for p, (role, _name, _param, _grad) in zip(projected, entries)
        ]
    pnorm = _norm(projected)
    tnorm = _norm(task_dirs)
    scale = 0.0 if pnorm <= 1e-30 else min(float(trust_ratio) * tnorm / max(pnorm, 1e-30), 0.25)
    corrected = [t + scale * p for t, p in zip(task_dirs, projected)]
    clip_rate = float(scale < 0.25 and pnorm > 0.0)
    return corrected, float(scale), clip_rate


def _direction_for_candidate(
    cid: str,
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    task_dirs: Sequence[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float, float, int]:
    base_dirs = _base_functional_dirs(entries, task_dirs)
    if cid == "F0":
        zeros = [torch.zeros_like(t) for t in task_dirs]
        return task_dirs, zeros, 0.0, 0.0, 0
    func_key = {
        "F1": "curv", "F2": "data", "F3": "sobolev", "F4": "coord", "F5": "adan",
        "FT0": "curv", "FT1": "data", "FT2": "sobolev", "FT3": "coord", "FT4": "adan",
        "FT5": "curv", "FT6": "curv", "FT7": "curv",
        "FR0": "curv", "FR1": "coord", "FR2": "adan", "FR3": "data",
    }.get(cid, "curv")
    func = base_dirs[func_key]
    if cid.startswith("FR") or cid in {"F1", "F2", "F3", "F4", "F5"}:
        return func, func, 1.0, 0.0, 0
    trust = 0.10
    if cid in {"FT3", "FT4"}:
        trust = 0.05
    if cid == "FT5":
        trust = 0.12 if _cos(func, task_dirs) >= 0.0 else 0.02
    if cid == "FT6":
        trust = 0.03
    role_weights = FT7_ROLE_WEIGHTS if cid == "FT7" else None
    corrected, scale, clip_rate = _project_task_aware(task_dirs, func, trust_ratio=trust, role_weights=role_weights, entries=entries)
    return corrected, func, scale, clip_rate, 0


def _apply_direction(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    direction: Sequence[torch.Tensor],
    alpha: float,
) -> None:
    with torch.no_grad():
        for (_role, _name, param, _grad), delta in zip(entries, direction):
            param.add_(delta, alpha=alpha)


def _apply_streamed_second_diff_correction(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    alpha: float,
    role_weights: Dict[str, float] | None = None,
) -> None:
    """Apply the curvature correction without materializing full direction tensors."""
    with torch.no_grad():
        for role, _name, param, _grad in entries:
            if param.numel() < 3 or param.shape[-1] < 3:
                continue
            weight = float(role_weights.get(role, 1.0)) if role_weights else 1.0
            if weight == 0.0:
                continue
            coeff = alpha * weight
            second = param[..., 2:] - 2.0 * param[..., 1:-1] + param[..., :-2]
            param[..., :-2].add_(second, alpha=-coeff)
            param[..., 1:-1].add_(second, alpha=2.0 * coeff)
            param[..., 2:].add_(second, alpha=-coeff)


def _apply_streamed_second_diff_correction_with_rollback(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    alpha: float,
    role_weights: Dict[str, float] | None = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    rollback: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    with torch.no_grad():
        for role, _name, param, _grad in entries:
            if param.numel() < 3 or param.shape[-1] < 3:
                continue
            weight = float(role_weights.get(role, 1.0)) if role_weights else 1.0
            if weight == 0.0:
                continue
            coeff = alpha * weight
            second = param[..., 2:] - 2.0 * param[..., 1:-1] + param[..., :-2]
            param[..., :-2].add_(second, alpha=-coeff)
            param[..., 1:-1].add_(second, alpha=2.0 * coeff)
            param[..., 2:].add_(second, alpha=-coeff)
            rollback.append((param, second, coeff))
    return rollback


def _apply_streamed_second_diff_correction_with_fixed_coeff_rollback(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    coeff: float,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    rollback: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    if coeff == 0.0:
        return rollback
    with torch.no_grad():
        for _role, _name, param, _grad in entries:
            if param.numel() < 3 or param.shape[-1] < 3:
                continue
            second = param[..., 2:] - 2.0 * param[..., 1:-1] + param[..., :-2]
            param[..., :-2].add_(second, alpha=-coeff)
            param[..., 1:-1].add_(second, alpha=2.0 * coeff)
            param[..., 2:].add_(second, alpha=-coeff)
            rollback.append((param, second, coeff))
    return rollback


def _rollback_streamed_second_diff_correction(rollback: Sequence[Tuple[torch.Tensor, torch.Tensor, float]]) -> None:
    with torch.no_grad():
        for param, second, coeff in reversed(rollback):
            param[..., :-2].add_(second, alpha=coeff)
            param[..., 1:-1].add_(second, alpha=-2.0 * coeff)
            param[..., 2:].add_(second, alpha=coeff)


def _streamed_second_diff_norm(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    role_weights: Dict[str, float] | None = None,
) -> float:
    if not entries:
        return 0.0
    out = torch.zeros((), device=entries[0][2].device, dtype=torch.float64)
    for role, _name, param, _grad in entries:
        weight = float(role_weights.get(role, 1.0)) if role_weights else 1.0
        if weight == 0.0:
            continue
        grad = _second_diff_grad(param.detach())
        out = out + grad.detach().double().pow(2).sum() * (weight * weight)
    return float(torch.sqrt(out.clamp_min(0.0)).detach().cpu())


def _measure_streamed_functional_operator(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    role_weights: Dict[str, float] | None = None,
    reps: int = 100,
    warmup: int = 20,
) -> Tuple[float, float]:
    snapshots = _clone_params(entries)
    alpha = 1.0e-9
    for _ in range(max(0, warmup)):
        _apply_streamed_second_diff_correction(entries, alpha, role_weights=role_weights)
    _restore_params(entries, snapshots)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        start_alloc = torch.cuda.memory_allocated(device)
    else:
        start_alloc = 0
    started = time.perf_counter()
    for _ in range(max(1, reps)):
        _apply_streamed_second_diff_correction(entries, alpha, role_weights=role_weights)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = 0
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / max(1, reps)
    _restore_params(entries, snapshots)
    delta_mb = max(0.0, (peak - start_alloc) / (1024.0 * 1024.0))
    return elapsed_ms, delta_mb


def _role_json(
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    direction: Sequence[torch.Tensor],
    task_dirs: Sequence[torch.Tensor],
    alpha: float,
) -> Tuple[str, str]:
    update_by_role: Dict[str, float] = {}
    descent_by_role: Dict[str, float] = {}
    for (role, _name, _param, _grad), delta, task in zip(entries, direction, task_dirs):
        update_by_role[role] = update_by_role.get(role, 0.0) + float(delta.detach().double().pow(2).sum().cpu()) * alpha * alpha
        descent_by_role[role] = descent_by_role.get(role, 0.0) + float((task.detach().double() * delta.detach().double()).sum().cpu()) * alpha
    update_total = max(sum(update_by_role.values()), 1e-30)
    descent_total = max(abs(sum(descent_by_role.values())), 1e-30)
    update_share = {k: v / update_total for k, v in sorted(update_by_role.items())}
    descent_share = {k: v / descent_total for k, v in sorted(descent_by_role.items())}
    return _json_dumps(update_share), _json_dumps(descent_share)


def _best_step_time_ms(out_dir: Path, candidate_id: str) -> float:
    vals = [
        _float(row.get("step_time_ms"))
        for row in _read_csv_rows(out_dir / "p10_efficiency_profiler.csv")
        if str(row.get("candidate_id")) == candidate_id
    ]
    return _mean(vals)


def _base_peak_mb(out_dir: Path, candidate_id: str) -> float:
    vals = [
        _float(row.get("peak_allocated_MB"))
        for row in _read_csv_rows(out_dir / "p10_efficiency_profiler.csv")
        if str(row.get("candidate_id")) == candidate_id
    ]
    return _mean(vals)


def _run_functional_one_step(out_dir: Path, args: Any, audit_base_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    device = get_device(args.device)
    v80._patch_for_v80()
    specs = v80._spec_map_v80()
    if audit_base_id not in specs:
        audit_base_id = "KW3"
    spec = specs[audit_base_id]
    datasets = parse_str_list(args.datasets)
    seeds = parse_int_list(args.seeds)[:3] or [0, 1, 2]
    p3_batches = 20
    raw_rows: List[Dict[str, Any]] = []
    p2_rows: List[Dict[str, Any]] = []
    base_step_ms = _best_step_time_ms(out_dir, audit_base_id)
    base_peak = _base_peak_mb(out_dir, audit_base_id)

    first_context: Tuple[Any, Any, Any, Any, Any, Any, Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]], List[torch.Tensor]] | None = None
    for dataset in datasets:
        for seed in seeds:
            bundle = v72.v71.load_vision_bundle(
                dataset,
                data_root=Path(args.data_root),
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=seed,
                allow_fake_data=False,
            )
            x_train = bundle.x_train.to(device)
            y_train = bundle.y_train.to(device)
            set_seed(v72._stable_seed("v83-functional-one-step", dataset, seed, v80._init_seed_candidate_id_v80(spec), spec.head_kind, spec.group_count, spec.shuffle))
            stack, head = v80._make_manual_candidate_v80(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
            entries = _param_entries(stack, head)
            snapshot = _clone_params(entries)
            for batch_ix in range(p3_batches):
                step = batch_ix + 1
                xb, yb = v72.v71._select_batch(x_train, y_train, args.batch_size, step)
                xh, yh = v72.v71._select_batch(x_train, y_train, args.batch_size, step + 1009)
                _restore_params(entries, snapshot)
                train_loss_before, task_dirs = _manual_task_grads(stack, head, xb, yb, spec)
                entries = _param_entries(stack, head)
                task_norm = _norm(task_dirs)
                param_norm = _param_norm(entries)
                target_rel = 1.0e-3
                task_alpha = target_rel * param_norm / max(task_norm, 1e-30)
                holdout_loss_before = _loss_only(stack, head, xh, yh, spec)
                smooth_before, curv_before = _geometry_norms(entries)
                snap2 = _clone_params(entries)
                _apply_direction(entries, task_dirs, task_alpha)
                train_task_after = _loss_only(stack, head, xb, yb, spec)
                holdout_task_after = _loss_only(stack, head, xh, yh, spec)
                task_train_descent = train_loss_before - train_task_after
                task_holdout_descent = holdout_loss_before - holdout_task_after
                _restore_params(entries, snap2)
                if first_context is None:
                    first_context = (stack, head, xb, yb, xh, yh, entries, task_dirs)
                for cid, name, update_type, family in FUNCTIONAL_CANDIDATES:
                    direction, func_dir, trust_scale, clip_rate, fallback = _direction_for_candidate(cid, entries, task_dirs)
                    direction_norm = _norm(direction)
                    alpha = target_rel * param_norm / max(direction_norm, 1e-30)
                    predicted_train_descent = float((_dot(task_dirs, direction) * alpha).detach().cpu())
                    cos_func = _cos(func_dir, task_dirs) if _norm(func_dir) > 0 else 1.0
                    cos_corr = _cos(direction, task_dirs) if direction_norm > 0 else 1.0
                    role_update_share, role_descent = _role_json(entries, direction, task_dirs, alpha)
                    snap3 = _clone_params(entries)
                    _apply_direction(entries, direction, alpha)
                    train_after = _loss_only(stack, head, xb, yb, spec)
                    holdout_after = _loss_only(stack, head, xh, yh, spec)
                    smooth_after, curv_after = _geometry_norms(entries)
                    _restore_params(entries, snap3)
                    actual_train_descent = train_loss_before - train_after
                    actual_holdout_descent = holdout_loss_before - holdout_after
                    if task_holdout_descent > 1e-12:
                        holdout_ratio = actual_holdout_descent / task_holdout_descent
                    else:
                        holdout_ratio = 1.0 if actual_holdout_descent >= -1e-12 else 0.0
                    bad_step = int(actual_holdout_descent < -1e-8 or holdout_ratio < 0.95)
                    raw_rows.append({
                        "stage": "P3_ONE_STEP_FUNCTIONAL_DIRECTION_RAW_V83",
                        "base_candidate_id": audit_base_id,
                        "functional_candidate_id": cid,
                        "functional_candidate_name": name,
                        "functional_family": family,
                        "functional_update_type": update_type,
                        "dataset": dataset,
                        "seed": seed,
                        "batch_ix": batch_ix,
                        "manual_grad_norm": task_norm,
                        "task_direction_norm": task_norm,
                        "functional_direction_norm": _norm(func_dir),
                        "corrected_direction_norm": direction_norm,
                        "cos_functional_with_task": cos_func,
                        "cos_corrected_with_task": cos_corr,
                        "predicted_train_descent": predicted_train_descent,
                        "actual_train_descent": actual_train_descent,
                        "actual_holdout_descent": actual_holdout_descent,
                        "task_holdout_descent": task_holdout_descent,
                        "holdout_descent_ratio_vs_task": holdout_ratio,
                        "bad_step_flag": bad_step,
                        "update_over_param_norm": alpha * direction_norm / max(param_norm, 1e-30),
                        "trust_region_scale": trust_scale,
                        "clip_rate": clip_rate,
                        "fallback_rate": fallback,
                        "edge_smoothness_norm_before": smooth_before,
                        "edge_smoothness_norm_after": smooth_after,
                        "edge_curvature_norm_before": curv_before,
                        "edge_curvature_norm_after": curv_after,
                        "geometry_reduction_vs_base": (curv_before - curv_after) / curv_before if curv_before > 1e-30 else 0.0,
                        "role_update_share": role_update_share,
                        "role_descent_contribution": role_descent,
                        "role_geometry_reduction": _json_dumps({"global_curvature": (curv_before - curv_after) / curv_before if curv_before > 1e-30 else 0.0}),
                        "loss_type": "CE",
                        "uses_loss_backward": 0,
                        "uses_autograd_for_functional_metric": 0,
                        "uses_full_jacobian_materialization": 0,
                        "uses_cpu_solve": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
            _restore_params(entries, snapshot)

    if first_context is not None:
        _stack, _head, _xb, _yb, _xh, _yh, entries, task_dirs = first_context
        for cid, name, update_type, family in FUNCTIONAL_CANDIDATES:
            if cid in STREAMED_FUNCTIONAL_OPERATOR_CANDIDATES:
                stream_role_weights = FT7_ROLE_WEIGHTS if cid == "FT7" else None
                elapsed_ms, delta_mb = _measure_streamed_functional_operator(entries, device=device, role_weights=stream_role_weights)
                direction, func_dir, _scale, _clip, fallback = _direction_for_candidate(cid, entries, task_dirs)
                if cid == "FT7":
                    implementation_variant = "streamed_rolewise_inplace_second_diff_correction"
                elif cid == "FT6":
                    implementation_variant = "streamed_late_phase_inplace_second_diff_correction"
                else:
                    implementation_variant = "streamed_inplace_second_diff_correction"
            else:
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                    start_alloc = torch.cuda.memory_allocated(device)
                else:
                    start_alloc = 0
                started = time.perf_counter()
                direction, func_dir, _scale, _clip, fallback = _direction_for_candidate(cid, entries, task_dirs)
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated(device)
                else:
                    peak = 0
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                delta_mb = max(0.0, (peak - start_alloc) / (1024.0 * 1024.0))
                implementation_variant = "materialized_direction_diagnostic"
            mem_ratio = (base_peak + delta_mb) / base_peak if base_peak and math.isfinite(base_peak) and base_peak > 0 else METRIC_UNAVAILABLE
            time_ratio = elapsed_ms / base_step_ms if base_step_ms and math.isfinite(base_step_ms) and base_step_ms > 0 else METRIC_UNAVAILABLE
            implementation_pass = int(
                cid != "F0"
                and elapsed_ms <= 0.10 * base_step_ms if base_step_ms and math.isfinite(base_step_ms) and base_step_ms > 0 else 0
            )
            if isinstance(mem_ratio, float):
                implementation_pass = int(implementation_pass and mem_ratio <= 1.05)
            p2_rows.append({
                "stage": "P2_FUNCTIONAL_OPERATOR_AUDIT_V83",
                "base_candidate_id": audit_base_id,
                "functional_candidate_id": cid,
                "functional_candidate_name": name,
                "functional_family": family,
                "functional_update_type": update_type,
                "implementation_variant": implementation_variant,
                "streamed_inplace_operator_used": int(cid in STREAMED_FUNCTIONAL_OPERATOR_CANDIDATES),
                "functional_update_used": int(cid != "F0"),
                "functional_update_is_update_rule": 1,
                "uses_autograd_for_functional_metric": 0,
                "uses_loss_backward": 0,
                "uses_full_jacobian_materialization": 0,
                "uses_cpu_solve": 0,
                "metric_build_time_ms": elapsed_ms,
                "metric_solve_time_ms": 0.0,
                "functional_update_time_ms": elapsed_ms,
                "base_step_time_ms": base_step_ms,
                "functional_update_time_ratio": time_ratio,
                "functional_memory_peak_MB": delta_mb,
                "base_peak_allocated_MB": base_peak,
                "functional_memory_peak_ratio": mem_ratio,
                "metric_condition_number": "not_applicable_no_dense_metric",
                "metric_solve_residual": "not_applicable_no_solve",
                "fallback_rate": fallback,
                "direction_norm": _norm(direction),
                "functional_direction_norm": _norm(func_dir),
                "implementation_pass": implementation_pass,
                "loss_type": "CE",
                "geometry_loss_used": 0,
                "special_loss_used": 0,
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "cpu_offload_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
    return p2_rows, raw_rows


def _summarize_p3(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault((str(row["base_candidate_id"]), str(row["functional_candidate_id"])), []).append(row)
    summary: List[Dict[str, Any]] = []
    meta = {cid: (name, update_type, family) for cid, name, update_type, family in FUNCTIONAL_CANDIDATES}
    for (base_id, cid), rows in sorted(groups.items()):
        name, update_type, family = meta[cid]
        bad_rate = _mean([_float(r.get("bad_step_flag"), 0.0) for r in rows])
        cos_corr = _mean([_float(r.get("cos_corrected_with_task")) for r in rows])
        hold_ratio = _mean([_float(r.get("holdout_descent_ratio_vs_task")) for r in rows])
        geom_red = _mean([_float(r.get("geometry_reduction_vs_base")) for r in rows])
        pure_threshold = cid.startswith("FR")
        direction_pass = int(
            cos_corr >= (0.75 if pure_threshold else 0.85)
            and hold_ratio >= 0.95
            and bad_rate <= (0.01 if pure_threshold else 0.02)
        )
        summary.append({
            "stage": "P3_ONE_STEP_FUNCTIONAL_DIRECTION_V83",
            "base_candidate_id": base_id,
            "functional_candidate_id": cid,
            "functional_candidate_name": name,
            "functional_family": family,
            "functional_update_type": update_type,
            "rows": len(rows),
            "manual_grad_norm_mean": _mean([_float(r.get("manual_grad_norm")) for r in rows]),
            "task_direction_norm_mean": _mean([_float(r.get("task_direction_norm")) for r in rows]),
            "functional_direction_norm_mean": _mean([_float(r.get("functional_direction_norm")) for r in rows]),
            "corrected_direction_norm_mean": _mean([_float(r.get("corrected_direction_norm")) for r in rows]),
            "cos_functional_with_task_mean": _mean([_float(r.get("cos_functional_with_task")) for r in rows]),
            "cos_corrected_with_task_mean": cos_corr,
            "predicted_train_descent_mean": _mean([_float(r.get("predicted_train_descent")) for r in rows]),
            "actual_train_descent_mean": _mean([_float(r.get("actual_train_descent")) for r in rows]),
            "actual_holdout_descent_mean": _mean([_float(r.get("actual_holdout_descent")) for r in rows]),
            "holdout_descent_ratio_vs_task_mean": hold_ratio,
            "bad_step_rate": bad_rate,
            "update_over_param_norm_mean": _mean([_float(r.get("update_over_param_norm")) for r in rows]),
            "trust_region_scale_mean": _mean([_float(r.get("trust_region_scale")) for r in rows]),
            "clip_rate_mean": _mean([_float(r.get("clip_rate")) for r in rows]),
            "fallback_rate_mean": _mean([_float(r.get("fallback_rate")) for r in rows]),
            "edge_curvature_reduction_mean": geom_red,
            "direction_gate_pass": direction_pass,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    if not summary:
        summary.append({
            "stage": "P3_ONE_STEP_FUNCTIONAL_DIRECTION_V83",
            "status": "not_run",
            "reason": "no_raw_rows",
            "direction_gate_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    return summary


def _param_rel_l2(
    a: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    b: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
) -> float:
    num = 0.0
    den = 0.0
    for (_ra, _na, pa, _ga), (_rb, _nb, pb, _gb) in zip(a, b):
        diff = (pa.detach().double() - pb.detach().double()).pow(2).sum()
        base = pb.detach().double().pow(2).sum()
        num += float(diff.cpu())
        den += float(base.cpu())
    return math.sqrt(num / max(den, 1e-30))


def _p4_candidate_ids(out_dir: Path) -> List[str]:
    p2 = _read_csv_rows(out_dir / "p2_functional_operator_audit.csv")
    p3 = _read_csv_rows(out_dir / "p3_one_step_functional_direction.csv")
    p2_pass = {
        str(row.get("functional_candidate_id"))
        for row in p2
        if _is_one(row.get("implementation_pass")) and _is_one(row.get("functional_update_used"))
    }
    p3_pass = [
        row for row in p3
        if _is_one(row.get("direction_gate_pass")) and str(row.get("functional_candidate_id")).startswith("FT")
    ]
    eligible = [row for row in p3_pass if str(row.get("functional_candidate_id")) in p2_pass]
    eligible.sort(key=lambda r: _float(r.get("edge_curvature_reduction_mean"), -99.0), reverse=True)
    out: List[str] = []
    for row in eligible[:3]:
        cid = str(row.get("functional_candidate_id"))
        if cid not in out:
            out.append(cid)
    return out


def _p4_functional_update_type(fid: str) -> str:
    if fid == "FT6":
        return "late_phase_projected_curvature_correction"
    if fid == "FT7":
        return "rolewise_projected_curvature_correction"
    return "event_triggered_projected_curvature_correction"


def _run_p4_multistep_smoke(out_dir: Path, args: Any, audit_base_id: str) -> List[Dict[str, Any]]:
    functional_ids = _p4_candidate_ids(out_dir)
    if not functional_ids:
        return [{
            "stage": "P4_MULTISTEP_FUNCTIONAL_SMOKE_V83",
            "status": "not_run",
            "reason": "no_p2_p3_functional_survivor",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]

    device = get_device(args.device)
    v80._patch_for_v80()
    specs = v80._spec_map_v80()
    spec = specs.get(audit_base_id, specs["KW6"])
    datasets = parse_str_list(args.datasets)
    seeds = (parse_int_list(args.seeds) or [0, 1, 2])[:3]
    eval_steps = {20, 50}
    max_steps = 50
    params = v80.V63Params(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
    )
    lr = float(params.lr_manual * spec.lr_mult)
    func_alpha = lr * 0.05
    rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        for seed in seeds:
            bundle = v72.v71.load_vision_bundle(
                dataset,
                data_root=Path(args.data_root),
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=seed,
                allow_fake_data=False,
            )
            x_train = bundle.x_train.to(device)
            y_train = bundle.y_train.to(device)
            x_val = bundle.x_val.to(device)
            y_val = bundle.y_val.to(device)
            set_seed(v72._stable_seed("v83-p4-base", dataset, seed, v80._init_seed_candidate_id_v80(spec), spec.head_kind, spec.group_count, spec.shuffle))
            base_stack, base_head = v80._make_manual_candidate_v80(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
            base_entries = _param_entries(base_stack, base_head)
            init_snapshot = _clone_params(base_entries)
            base_opt = v72.v71.FastAdamW([base_stack, base_head], lr=lr, weight_decay=getattr(args, "weight_decay", 1.0e-4))

            func_models: Dict[str, Tuple[Any, Any, Any, List[Tuple[str, str, torch.Tensor, torch.Tensor]], int, float, float, float]] = {}
            for fid in functional_ids:
                set_seed(v72._stable_seed("v83-p4-base", dataset, seed, v80._init_seed_candidate_id_v80(spec), spec.head_kind, spec.group_count, spec.shuffle))
                stack, head = v80._make_manual_candidate_v80(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
                entries = _param_entries(stack, head)
                _restore_params(entries, init_snapshot)
                opt = v72.v71.FastAdamW([stack, head], lr=lr, weight_decay=getattr(args, "weight_decay", 1.0e-4))
                func_models[fid] = (stack, head, opt, entries, 0, 0.0, 0.0, 0)

            for step in range(1, max_steps + 1):
                xb, yb = v72.v71._select_batch(x_train, y_train, args.batch_size, step)
                xh, yh = v72.v71._select_batch(x_train, y_train, args.batch_size, step + 1009)
                _manual_ce_backward(base_stack, base_head, xb, yb, spec)
                base_opt.step(step, max_steps, warmup_cosine=True)
                for fid, (stack, head, opt, entries, bad_count, func_norm_sum, trust_sum, reject_count) in list(func_models.items()):
                    holdout_loss_before = (
                        _loss_only(stack, head, xh, yh, spec)
                        if fid == "FT7" and not FT7_FAST_TASK_CHECK and FT7_USE_HOLDOUT_GUARD and _ft7_uses_pre_holdout_for_step(step)
                        else None
                    )
                    loss_before = _manual_ce_backward(stack, head, xb, yb, spec)
                    opt.step(step, max_steps, warmup_cosine=True)
                    active_functional_step = fid != "FT6" or step > (max_steps // 2)
                    role_weights = FT7_ROLE_WEIGHTS if fid == "FT7" else None
                    if active_functional_step:
                        task_loss_after = None
                        holdout_loss_after = None
                        correction_snapshot = None
                        accept_ce_limit = None
                        if fid in {"FT5", "FT7"}:
                            task_loss_after = _loss_only(stack, head, xb, yb, spec)
                            accept_ce_limit = task_loss_after + 0.10 * max(0.0, loss_before - task_loss_after)
                            if fid == "FT7" and FT7_USE_HOLDOUT_GUARD:
                                holdout_loss_after = _loss_only(stack, head, xh, yh, spec)
                            correction_snapshot = _clone_params(entries)
                        if fid == "FT7" and task_loss_after is not None:
                            loss_after = task_loss_after
                            func_norm = 0.0
                            if FT7_FAST_TASK_CHECK:
                                loss_after, func_norm, trust_delta, reject_delta = _apply_ft7_streamed_guarded_update(
                                    stack,
                                    head,
                                    entries,
                                    xb,
                                    yb,
                                    xh,
                                    yh,
                                    spec,
                                    loss_before=loss_before,
                                    holdout_loss_before=0.0,
                                    task_loss_after=task_loss_after,
                                    holdout_loss_after=0.0,
                                    func_alpha=func_alpha,
                                )
                                trust_sum += trust_delta
                                reject_count += reject_delta
                            else:
                                for role_name, role_weight in (("stack", FT7_ROLE_WEIGHTS["stack"]), ("head", FT7_ROLE_WEIGHTS["head"])):
                                    role_budget = FT7_ROLE_BUDGETS[role_name]
                                    role_accept_ce_limit = task_loss_after + role_budget * max(0.0, loss_before - task_loss_after)
                                    role_holdout_ce_limit = None
                                    if holdout_loss_after is not None:
                                        holdout_budget_base = _ft7_holdout_budget_base(
                                            holdout_loss_before,
                                            holdout_loss_after,
                                            loss_before - task_loss_after,
                                        )
                                        role_holdout_ce_limit = holdout_loss_after + role_budget * holdout_budget_base
                                    role_snapshot = _clone_params(entries)
                                    role_map = {"stack": 0.0, "head": 0.0}
                                    role_map[role_name] = role_weight
                                    curv_dirs = [-_second_diff_grad(param.detach()) for _role, _name, param, _grad in entries]
                                    weighted_curv = [
                                        c * float(role_map.get(role, 0.0))
                                        for c, (role, _name, _param, _grad) in zip(curv_dirs, entries)
                                    ]
                                    role_norm = _norm(weighted_curv) * func_alpha
                                    _apply_streamed_second_diff_correction(entries, func_alpha, role_weights=role_map)
                                    role_loss = _loss_only(stack, head, xb, yb, spec)
                                    role_train_ok = role_loss <= role_accept_ce_limit + 1e-8
                                    role_holdout_loss = (
                                        _loss_only(stack, head, xh, yh, spec)
                                        if role_train_ok and role_holdout_ce_limit is not None
                                        else None
                                    )
                                    role_holdout_ok = role_holdout_ce_limit is None or (
                                        role_holdout_loss is not None and role_holdout_loss <= role_holdout_ce_limit + 1e-8
                                    )
                                    if accept_ce_limit is not None and role_train_ok and role_holdout_ok:
                                        loss_after = role_loss
                                        if role_holdout_loss is not None:
                                            holdout_loss_after = role_holdout_loss
                                        func_norm += role_norm
                                        trust_sum += 0.025
                                    else:
                                        _restore_params(entries, role_snapshot)
                                        reject_count += 0.5
                        else:
                            curv_dirs = [-_second_diff_grad(param.detach()) for _role, _name, param, _grad in entries]
                            if role_weights:
                                weighted_curv = [
                                    c * float(role_weights.get(role, 1.0))
                                    for c, (role, _name, _param, _grad) in zip(curv_dirs, entries)
                                ]
                            else:
                                weighted_curv = curv_dirs
                            func_norm = _norm(weighted_curv) * func_alpha
                            _apply_streamed_second_diff_correction(entries, func_alpha, role_weights=role_weights)
                            loss_after = _loss_only(stack, head, xb, yb, spec)
                            if fid == "FT5" and accept_ce_limit is not None and loss_after > accept_ce_limit + 1e-8:
                                if correction_snapshot is not None:
                                    _restore_params(entries, correction_snapshot)
                                loss_after = task_loss_after
                                func_norm = 0.0
                                reject_count += 1.0
                            else:
                                trust_sum += 0.05
                    else:
                        func_norm = 0.0
                        loss_after = _loss_only(stack, head, xb, yb, spec)
                    bad_count += int(loss_after > loss_before + 1e-8)
                    func_norm_sum += func_norm
                    func_models[fid] = (stack, head, opt, entries, bad_count, func_norm_sum, trust_sum, reject_count)

                if step not in eval_steps:
                    continue
                base_eval = v72.v71._manual_eval(base_stack, base_head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
                base_smooth, base_curv = _geometry_norms(_param_entries(base_stack, base_head))
                base_logits = _logits_only(base_stack, base_head, x_val, args.eval_batch_size)
                rows.append({
                    "stage": "P4_MULTISTEP_FUNCTIONAL_SMOKE_V83",
                    "base_candidate_id": audit_base_id,
                    "functional_candidate_id": "BASE",
                    "functional_update_type": "none",
                    "dataset": dataset,
                    "seed": seed,
                    "step": step,
                    "train_loss": METRIC_UNAVAILABLE,
                    "val_loss": base_eval["loss"],
                    "val_acc": base_eval["acc"],
                    "ECE": base_eval["ECE"],
                    "NLL": base_eval["NLL"],
                    "margin_p10": base_eval["margin_p10"],
                    "geometry_smoothness": base_smooth,
                    "geometry_curvature": base_curv,
                    "jacobian_norm": METRIC_UNAVAILABLE,
                    "local_lipschitz": METRIC_UNAVAILABLE,
                    "functional_update_norm": 0.0,
                    "trust_region_scale": 0.0,
                    "fallback_rate": 0.0,
                    "bad_step_rate": 0.0,
                    "trajectory_param_rel_l2_vs_base": 0.0,
                    "trajectory_logit_KL_vs_base": 0.0,
                    "geometry_reduction_vs_base": 0.0,
                    "P4_task_preservation_pass": 1,
                    "P4_geometry_pass": 1,
                    "P4_multistep_pass": 0,
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                for fid, (stack, head, _opt, entries, bad_count, func_norm_sum, trust_sum, reject_count) in func_models.items():
                    ev = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
                    smooth, curv = _geometry_norms(entries)
                    logits = _logits_only(stack, head, x_val, args.eval_batch_size)
                    acc_drop = float(base_eval["acc"]) - float(ev["acc"])
                    loss_inc = float(ev["loss"]) - float(base_eval["loss"])
                    bad_rate = bad_count / float(step)
                    geo_ratio = curv / max(base_curv, 1e-30)
                    task_pass = int(acc_drop <= 0.003 and loss_inc <= 0.005 and bad_rate <= 0.02)
                    geo_pass = int(geo_ratio <= 0.90)
                    rows.append({
                        "stage": "P4_MULTISTEP_FUNCTIONAL_SMOKE_V83",
                        "base_candidate_id": audit_base_id,
                        "functional_candidate_id": fid,
                        "functional_update_type": _p4_functional_update_type(fid),
                        "dataset": dataset,
                        "seed": seed,
                        "step": step,
                        "train_loss": METRIC_UNAVAILABLE,
                        "val_loss": ev["loss"],
                        "val_acc": ev["acc"],
                        "ECE": ev["ECE"],
                        "NLL": ev["NLL"],
                        "margin_p10": ev["margin_p10"],
                        "geometry_smoothness": smooth,
                        "geometry_curvature": curv,
                        "jacobian_norm": METRIC_UNAVAILABLE,
                        "local_lipschitz": METRIC_UNAVAILABLE,
                        "functional_update_norm": func_norm_sum / float(step),
                        "trust_region_scale": trust_sum / float(step),
                        "fallback_rate": reject_count / float(step),
                        "bad_step_rate": bad_rate,
                        "trajectory_param_rel_l2_vs_base": _param_rel_l2(entries, _param_entries(base_stack, base_head)),
                        "trajectory_logit_KL_vs_base": _kl_logits(base_logits, logits),
                        "geometry_reduction_vs_base": 1.0 - geo_ratio,
                        "AccDrop_vs_base": acc_drop,
                        "ValLossIncrease_vs_base": loss_inc,
                        "geometry_curvature_ratio_vs_base": geo_ratio,
                        "P4_task_preservation_pass": task_pass,
                        "P4_geometry_pass": geo_pass,
                        "P4_multistep_pass": int(task_pass and geo_pass),
                        "loss_type": "CE",
                        "geometry_loss_used": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
    return rows


def _apply_ft7_streamed_guarded_update(
    stack: Any,
    head: Any,
    entries: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
    xb: torch.Tensor,
    yb: torch.Tensor,
    xh: torch.Tensor,
    yh: torch.Tensor,
    spec: Any,
    *,
    loss_before: float,
    holdout_loss_before: float,
    task_loss_after: float,
    holdout_loss_after: float,
    func_alpha: float,
    track_norm: bool = FT7_TRACK_DIAGNOSTIC_NORM,
    task_features_after: torch.Tensor | None = None,
    holdout_features_after: torch.Tensor | None = None,
    role_entries_by_name: Dict[str, List[Tuple[str, str, torch.Tensor, torch.Tensor]]] | None = None,
    pair_stack_guard_forward: bool = False,
) -> Tuple[float, float, float, float]:
    loss_after = task_loss_after
    func_norm = 0.0
    trust_delta = 0.0
    reject_delta = 0.0
    if FT7_FAST_TASK_CHECK:
        for role_name in ("stack", "head"):
            role_weight = FT7_ROLE_WEIGHTS[role_name]
            if role_weight == 0.0:
                continue
            role_entries = [entry for entry in entries if entry[0] == role_name]
            role_map = {"stack": 0.0, "head": 0.0}
            role_map[role_name] = role_weight
            if track_norm:
                func_norm += _streamed_second_diff_norm(role_entries, role_map) * func_alpha
            _apply_streamed_second_diff_correction(role_entries, func_alpha, role_weights=role_map)
            trust_delta += 0.025
        return _loss_only(stack, head, xb, yb, spec), func_norm, trust_delta, reject_delta
    if FT7_USE_JOINT_EVENT_GUARD:
        joint_accept_ce_limit = task_loss_after + FT7_JOINT_EVENT_BUDGET * max(0.0, loss_before - task_loss_after)
        joint_holdout_ce_limit = None
        if FT7_USE_HOLDOUT_GUARD:
            holdout_budget_base = _ft7_holdout_budget_base(
                holdout_loss_before,
                holdout_loss_after,
                loss_before - task_loss_after,
            )
            joint_holdout_ce_limit = holdout_loss_after + FT7_JOINT_EVENT_BUDGET * holdout_budget_base
        role_map = dict(FT7_ROLE_WEIGHTS)
        if track_norm:
            func_norm = _streamed_second_diff_norm(entries, role_map) * func_alpha
        rollback = _apply_streamed_second_diff_correction_with_rollback(entries, func_alpha, role_weights=role_map)
        joint_loss, _next_train_features = _loss_and_features_only(stack, head, xb, yb, spec)
        joint_train_ok = joint_loss <= joint_accept_ce_limit + 1e-8
        joint_holdout_loss = None
        if joint_train_ok and joint_holdout_ce_limit is not None:
            joint_holdout_loss, _next_holdout_features = _loss_and_features_only(stack, head, xh, yh, spec)
        joint_holdout_ok = joint_holdout_ce_limit is None or (
            joint_holdout_loss is not None and joint_holdout_loss <= joint_holdout_ce_limit + 1e-8
        )
        if joint_train_ok and joint_holdout_ok:
            return joint_loss, func_norm, 0.05, 0.0
        _rollback_streamed_second_diff_correction(rollback)
        return loss_after, 0.0, 0.0, 1.0
    train_features = task_features_after
    holdout_features = holdout_features_after if FT7_USE_HOLDOUT_GUARD else None
    stack_role_rejected = False
    if role_entries_by_name is None:
        role_entries_by_name = _entries_by_role(entries)
    for role_name in ("stack", "head"):
        role_weight = FT7_ROLE_WEIGHTS[role_name]
        if role_weight == 0.0:
            continue
        if role_name == "head" and stack_role_rejected:
            reject_delta += 0.5
            continue
        role_budget = FT7_ROLE_BUDGETS[role_name]
        role_accept_ce_limit = task_loss_after + role_budget * max(0.0, loss_before - task_loss_after)
        role_holdout_ce_limit = None
        if FT7_USE_HOLDOUT_GUARD:
            holdout_budget_base = _ft7_holdout_budget_base(
                holdout_loss_before,
                holdout_loss_after,
                loss_before - task_loss_after,
            )
            role_holdout_ce_limit = holdout_loss_after + role_budget * holdout_budget_base
        role_entries = role_entries_by_name[role_name]
        role_norm = 0.0
        if track_norm:
            role_norm = _streamed_second_diff_norm(role_entries) * role_weight * func_alpha
        rollback = _apply_streamed_second_diff_correction_with_fixed_coeff_rollback(role_entries, func_alpha * role_weight)
        next_train_features = None
        next_holdout_features = None
        head_pair_available = (
            role_name == "head"
            and train_features is not None
            and holdout_features is not None
            and role_holdout_ce_limit is not None
        )
        stack_pair_available = (
            pair_stack_guard_forward
            and role_name == "stack"
            and role_holdout_ce_limit is not None
        )
        if stack_pair_available:
            role_loss, paired_holdout_loss, next_train_features, next_holdout_features = _loss_pair_and_features_only(
                stack,
                head,
                xb,
                yb,
                xh,
                yh,
                spec,
            )
        elif head_pair_available:
            role_loss, paired_holdout_loss = _head_loss_pair_from_features(
                head,
                train_features,
                yb,
                holdout_features,
                yh,
                spec,
            )
        elif role_name == "head" and train_features is not None:
            role_loss = _head_loss_from_features(head, train_features, yb, spec)
            paired_holdout_loss = None
        else:
            role_loss, next_train_features = _loss_and_features_only(stack, head, xb, yb, spec)
            paired_holdout_loss = None
        role_train_ok = role_loss <= role_accept_ce_limit + 1e-8
        role_holdout_loss = None
        if role_train_ok and role_holdout_ce_limit is not None:
            if paired_holdout_loss is not None:
                role_holdout_loss = paired_holdout_loss
            elif role_name == "head" and holdout_features is not None:
                role_holdout_loss = _head_loss_from_features(head, holdout_features, yh, spec)
            else:
                role_holdout_loss, next_holdout_features = _loss_and_features_only(stack, head, xh, yh, spec)
        role_holdout_ok = role_holdout_ce_limit is None or (
            role_holdout_loss is not None and role_holdout_loss <= role_holdout_ce_limit + 1e-8
        )
        if role_train_ok and role_holdout_ok:
            loss_after = role_loss
            if role_holdout_loss is not None:
                holdout_loss_after = role_holdout_loss
            if role_name != "head" and next_train_features is not None:
                train_features = next_train_features
            if role_name != "head" and next_holdout_features is not None:
                holdout_features = next_holdout_features
            func_norm += role_norm
            trust_delta += 0.025
        else:
            _rollback_streamed_second_diff_correction(rollback)
            if role_name == "stack":
                stack_role_rejected = True
            reject_delta += 0.5
    return loss_after, func_norm, trust_delta, reject_delta


def _train_p5_variant(
    args: Any,
    dataset: str,
    seed: int,
    audit_base_id: str,
    functional_id: str,
) -> Dict[str, Any]:
    device = get_device(args.device)
    v80._patch_for_v80()
    spec = v80._spec_map_v80().get(audit_base_id, v80._spec_map_v80()["KW6"])
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        allow_fake_data=False,
    )
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    set_seed(v72._stable_seed("v83-p5-base", dataset, seed, v80._init_seed_candidate_id_v80(spec), spec.head_kind, spec.group_count, spec.shuffle))
    stack, head = v80._make_manual_candidate_v80(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    entries = _param_entries(stack, head)
    role_entries_by_name = _entries_by_role(entries) if functional_id == "FT7" else None
    params = v80.V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    lr = float(params.lr_manual * spec.lr_mult)
    func_alpha = lr * 0.05
    opt = v72.v71.FastAdamW([stack, head], lr=lr, weight_decay=getattr(args, "weight_decay", 1.0e-4))
    max_steps = 240
    trace_every = max(1, int(getattr(args, "trace_every", 10)))
    val_trace: List[Tuple[float, float]] = []
    step_times: List[float] = []
    bad_count = 0
    func_norm_sum = 0.0
    trust_sum = 0.0
    reject_count = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, max_steps + 1):
        xb, yb = v72.v71._select_batch(x_train, y_train, args.batch_size, step)
        xh, yh = v72.v71._select_batch(x_train, y_train, args.batch_size, step + 1009)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        if functional_id == "BASE":
            loss_before = _manual_ce_backward(stack, head, xb, yb, spec)
            opt.step(step, max_steps, warmup_cosine=True)
            loss_after = _loss_only(stack, head, xb, yb, spec)
        elif functional_id == "FT7":
            event_step = step % P5_FT7_EVENT_STRIDE == 0
            holdout_loss_before = (
                _loss_only(stack, head, xh, yh, spec)
                if event_step and not FT7_FAST_TASK_CHECK and FT7_USE_HOLDOUT_GUARD and _ft7_uses_pre_holdout_for_step(step)
                else 0.0
            )
            loss_before = _manual_ce_backward(stack, head, xb, yb, spec)
            opt.step(step, max_steps, warmup_cosine=True)
            if event_step:
                task_loss_after = _loss_only(stack, head, xb, yb, spec)
            else:
                task_loss_after = _loss_only(stack, head, xb, yb, spec)
            loss_after = task_loss_after
            if event_step:
                if not FT7_FAST_TASK_CHECK and FT7_USE_HOLDOUT_GUARD:
                    holdout_loss_after = _loss_only(stack, head, xh, yh, spec)
                else:
                    holdout_loss_after = 0.0
                loss_after, func_norm, trust_delta, reject_delta = _apply_ft7_streamed_guarded_update(
                    stack,
                    head,
                    entries,
                    xb,
                    yb,
                    xh,
                    yh,
                    spec,
                    loss_before=loss_before,
                    holdout_loss_before=holdout_loss_before,
                    task_loss_after=task_loss_after,
                    holdout_loss_after=holdout_loss_after,
                    func_alpha=func_alpha * P5_FT7_EVENT_ALPHA_MULT,
                    task_features_after=None,
                    holdout_features_after=None,
                    role_entries_by_name=role_entries_by_name,
                )
                func_norm_sum += func_norm
                trust_sum += trust_delta
                reject_count += reject_delta
        else:
            raise ValueError(f"unsupported P5 functional candidate: {functional_id}")
        bad_count += int(loss_after > loss_before + 1e-8)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append((time.perf_counter() - started) * 1000.0)
        if step % trace_every == 0 or step == max_steps:
            ev = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
            val_trace.append((float(step), float(ev["loss"])))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb: Any = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = METRIC_UNAVAILABLE
    val_eval = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    test_eval = v72.v71._manual_eval(stack, head, x_test, y_test, args.eval_batch_size, bundle.num_classes)
    smooth, curv = _geometry_norms(_param_entries(stack, head))
    return {
        "stage": "P5_FUNCTIONAL_TASK_GEOMETRY_RAW_V83",
        "base_candidate_id": audit_base_id,
        "functional_candidate_id": functional_id,
        "dataset": dataset,
        "seed": seed,
        "task_steps": max_steps,
        "val_acc": val_eval["acc"],
        "test_acc": test_eval["acc"],
        "val_loss": val_eval["loss"],
        "test_loss": test_eval["loss"],
        "ECE": val_eval["ECE"],
        "NLL": val_eval["NLL"],
        "Brier": METRIC_UNAVAILABLE,
        "margin_p10": val_eval["margin_p10"],
        "ValLossAUC_step": v72.v71._auc(val_trace),
        "edge_smoothness_norm": smooth,
        "edge_curvature_norm": curv,
        "finite_difference_curvature_p95": METRIC_UNAVAILABLE,
        "phi_slope_p95": METRIC_UNAVAILABLE,
        "jacobian_norm": METRIC_UNAVAILABLE,
        "local_lipschitz": METRIC_UNAVAILABLE,
        "sobolev_residual_norm": METRIC_UNAVAILABLE,
        "function_displacement_R2": METRIC_UNAVAILABLE,
        "path_length": METRIC_UNAVAILABLE,
        "step_time_ms_mean": _mean(step_times),
        "peak_allocated_MB": peak_mb,
        "functional_update_norm": func_norm_sum / float(max_steps),
        "trust_region_scale": trust_sum / float(max_steps),
        "fallback_rate": reject_count / float(max_steps),
        "bad_step_rate": bad_count / float(max_steps),
        "functional_event_stride": P5_FT7_EVENT_STRIDE if functional_id == "FT7" else 0,
        "functional_event_alpha_mult": P5_FT7_EVENT_ALPHA_MULT if functional_id == "FT7" else 0.0,
        "loss_type": "CE",
        "geometry_loss_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _summarize_p5(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not raw_rows:
        return []
    base_by_key = {
        (row.get("dataset"), row.get("seed")): row
        for row in raw_rows
        if row.get("functional_candidate_id") == "BASE"
    }
    out: List[Dict[str, Any]] = []
    functional_ids = sorted({row.get("functional_candidate_id") for row in raw_rows if row.get("functional_candidate_id") != "BASE"})
    for fid in functional_ids:
        rows = [row for row in raw_rows if row.get("functional_candidate_id") == fid]
        paired = [(row, base_by_key.get((row.get("dataset"), row.get("seed")), {})) for row in rows]
        val_acc_delta = [_float(row.get("val_acc")) - _float(base.get("val_acc")) for row, base in paired]
        test_acc_delta = [_float(row.get("test_acc")) - _float(base.get("test_acc")) for row, base in paired]
        ece_delta = [_float(row.get("ECE")) - _float(base.get("ECE")) for row, base in paired]
        nll_delta = [_float(row.get("NLL")) - _float(base.get("NLL")) for row, base in paired]
        auc_ratio = [
            _float(row.get("ValLossAUC_step")) / _float(base.get("ValLossAUC_step"))
            for row, base in paired
            if _float(base.get("ValLossAUC_step")) > 0
        ]
        geo_ratio = [
            _float(row.get("edge_curvature_norm")) / _float(base.get("edge_curvature_norm"))
            for row, base in paired
            if _float(base.get("edge_curvature_norm")) > 0
        ]
        step_ratio = [
            _float(row.get("step_time_ms_mean")) / _float(base.get("step_time_ms_mean"))
            for row, base in paired
            if _float(base.get("step_time_ms_mean")) > 0
        ]
        mem_ratio = [
            _float(row.get("peak_allocated_MB")) / _float(base.get("peak_allocated_MB"))
            for row, base in paired
            if _float(base.get("peak_allocated_MB")) > 0
        ]
        acc_delta_mean = _mean(val_acc_delta)
        ece_delta_mean = _mean(ece_delta)
        nll_delta_mean = _mean(nll_delta)
        auc_ratio_mean = _mean(auc_ratio)
        geo_ratio_mean = _mean(geo_ratio)
        step_ratio_mean = _mean(step_ratio)
        mem_ratio_mean = _mean(mem_ratio)
        primary_benefit = int(
            geo_ratio_mean <= 0.80
            or ece_delta_mean <= -0.005
            or auc_ratio_mean <= 0.98
            or acc_delta_mean >= 0.003
        )
        hard_constraints = int(
            acc_delta_mean >= -0.005
            and (not math.isfinite(mem_ratio_mean) or mem_ratio_mean <= 1.05)
            and (not math.isfinite(step_ratio_mean) or step_ratio_mean <= 1.50)
        )
        out.append({
            "stage": "P5_FUNCTIONAL_TASK_GEOMETRY_COSELECTION_V83",
            "base_candidate_id": rows[0].get("base_candidate_id", "KW6") if rows else "KW6",
            "functional_candidate_id": fid,
            "rows": len(rows),
            "macro_val_acc_delta_vs_base_mean": acc_delta_mean,
            "test_acc_delta_vs_base_mean": _mean(test_acc_delta),
            "ECE_delta_vs_base_mean": ece_delta_mean,
            "NLL_delta_vs_base_mean": nll_delta_mean,
            "ValLossAUC_step_ratio_vs_base_mean": auc_ratio_mean,
            "geometry_curvature_ratio_vs_base_mean": geo_ratio_mean,
            "geometry_reduction_vs_base_mean": 1.0 - geo_ratio_mean if math.isfinite(geo_ratio_mean) else METRIC_UNAVAILABLE,
            "memory_ratio_vs_base_mean": mem_ratio_mean if math.isfinite(mem_ratio_mean) else METRIC_UNAVAILABLE,
            "step_ratio_vs_base_mean": step_ratio_mean if math.isfinite(step_ratio_mean) else METRIC_UNAVAILABLE,
            "fallback_rate_mean": _mean(_float(row.get("fallback_rate")) for row in rows),
            "bad_step_rate_mean": _mean(_float(row.get("bad_step_rate")) for row in rows),
            "functional_update_norm_mean": _mean(_float(row.get("functional_update_norm")) for row in rows),
            "trust_region_scale_mean": _mean(_float(row.get("trust_region_scale")) for row in rows),
            "primary_benefit_pass": primary_benefit,
            "hard_constraints_pass": hard_constraints,
            "P5_functional_useful_pass": int(primary_benefit and hard_constraints),
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return out


def _run_p5_functional_coselection(out_dir: Path, args: Any, audit_base_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    p4_rows = _read_csv_rows(out_dir / "p4_multistep_functional_smoke.csv")
    final_step = max([_float(row.get("step"), -1.0) for row in p4_rows if row.get("functional_candidate_id") not in {"BASE", ""}] or [-1.0])
    survivors = sorted({
        str(row.get("functional_candidate_id"))
        for row in p4_rows
        if _float(row.get("step"), -2.0) == final_step and _is_one(row.get("P4_multistep_pass"))
    })
    survivors = [fid for fid in survivors if fid == "FT7"]
    if not survivors:
        return [], []
    raw_rows: List[Dict[str, Any]] = []
    datasets = parse_str_list(args.datasets)
    seeds = (parse_int_list(args.seeds) or [0, 1, 2])[:3]
    for dataset in datasets:
        for seed in seeds:
            raw_rows.append(_train_p5_variant(args, dataset, seed, audit_base_id, "BASE"))
            for fid in survivors:
                raw_rows.append(_train_p5_variant(args, dataset, seed, audit_base_id, fid))
    return _summarize_p5(raw_rows), raw_rows


def _measure_p6_shape_variant(
    args: Any,
    dataset: str,
    batch_size: int,
    seed: int,
    audit_base_id: str,
    functional_id: str,
) -> Dict[str, Any]:
    device = get_device(args.device)
    v80._patch_for_v80()
    spec = v80._spec_map_v80().get(audit_base_id, v80._spec_map_v80()["KW6"])
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        allow_fake_data=False,
    )
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    set_seed(v72._stable_seed("v83-p6-s2", dataset, seed, batch_size, functional_id, v80._init_seed_candidate_id_v80(spec), spec.head_kind, spec.group_count, spec.shuffle))
    stack, head = v80._make_manual_candidate_v80(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    entries = _param_entries(stack, head)
    role_entries_by_name = _entries_by_role(entries) if functional_id == "FT7" else None
    params = v80.V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=batch_size)
    lr = float(params.lr_manual * spec.lr_mult)
    func_alpha = lr * 0.05
    opt = v72.v71.FastAdamW([stack, head], lr=lr, weight_decay=getattr(args, "weight_decay", 1.0e-4))
    warmup = int(getattr(args, "bench_warmup", 20))
    reps = int(getattr(args, "bench_reps", 100))
    total_steps = max(240, warmup + reps + 1)
    bad_count = 0
    reject_count = 0.0

    def run_one(step: int) -> None:
        nonlocal bad_count, reject_count
        xb, yb = v72.v71._select_batch(x_train, y_train, batch_size, step)
        if functional_id == "BASE":
            loss_before = _manual_ce_backward(stack, head, xb, yb, spec)
            opt.step(step, total_steps, warmup_cosine=True)
            loss_after = _loss_only(stack, head, xb, yb, spec)
        elif functional_id == "FT7":
            event_step = step % P5_FT7_EVENT_STRIDE == 0
            xh, yh = v72.v71._select_batch(x_train, y_train, batch_size, step + 1009)
            holdout_loss_before = (
                _loss_only(stack, head, xh, yh, spec)
                if event_step and not FT7_FAST_TASK_CHECK and FT7_USE_HOLDOUT_GUARD and _ft7_uses_pre_holdout_for_step(step)
                else 0.0
            )
            loss_before = _manual_ce_backward(stack, head, xb, yb, spec)
            opt.step(step, total_steps, warmup_cosine=True)
            if event_step:
                task_loss_after = _loss_only(stack, head, xb, yb, spec)
            else:
                task_loss_after = _loss_only(stack, head, xb, yb, spec)
            loss_after = task_loss_after
            if event_step:
                if not FT7_FAST_TASK_CHECK and FT7_USE_HOLDOUT_GUARD:
                    holdout_loss_after = _loss_only(stack, head, xh, yh, spec)
                else:
                    holdout_loss_after = 0.0
                loss_after, _func_norm, _trust_delta, reject_delta = _apply_ft7_streamed_guarded_update(
                    stack,
                    head,
                    entries,
                    xb,
                    yb,
                    xh,
                    yh,
                    spec,
                    loss_before=loss_before,
                    holdout_loss_before=holdout_loss_before,
                    task_loss_after=task_loss_after,
                    holdout_loss_after=holdout_loss_after,
                    func_alpha=func_alpha * P5_FT7_EVENT_ALPHA_MULT,
                    task_features_after=None,
                    holdout_features_after=None,
                    role_entries_by_name=role_entries_by_name,
                )
                reject_count += reject_delta
        else:
            raise ValueError(f"unsupported P6 S2 functional candidate: {functional_id}")
        bad_count += int(loss_after > loss_before + 1e-8)

    for step in range(1, warmup + 1):
        run_one(step)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for offset in range(1, reps + 1):
        run_one(warmup + offset)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb: Any = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = METRIC_UNAVAILABLE
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / max(1, reps)
    return {
        "stage": "P6_FUNCTIONAL_FULLGRID_S2_V83",
        "base_candidate_id": audit_base_id,
        "functional_candidate_id": functional_id,
        "dataset": dataset,
        "batch_size": batch_size,
        "seed": seed,
        "bench_warmup": warmup,
        "bench_reps": reps,
        "step_time_ms": elapsed_ms,
        "peak_allocated_MB": peak_mb,
        "bad_step_rate_measured": bad_count / float(max(1, warmup + reps)),
        "fallback_rate_measured": reject_count / float(max(1, warmup + reps)),
        "functional_event_stride": P5_FT7_EVENT_STRIDE if functional_id == "FT7" else 0,
        "functional_event_alpha_mult": P5_FT7_EVENT_ALPHA_MULT if functional_id == "FT7" else 0.0,
        "loss_type": "CE",
        "geometry_loss_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _run_p6_functional_fullgrid_s2(
    args: Any,
    audit_base_id: str,
    survivors: Sequence[str],
) -> List[Dict[str, Any]]:
    survivors = [fid for fid in survivors if fid == "FT7"]
    if not survivors:
        return [{
            "stage": "P6_FUNCTIONAL_FULLGRID_S2_V83",
            "status": "not_run",
            "reason": "no_p5_functional_survivor",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
    rows: List[Dict[str, Any]] = []
    datasets = parse_str_list(args.datasets)
    batches = parse_int_list(args.bench_batch_sizes) or [128, 256, 512]
    seed = (parse_int_list(args.seeds) or [0])[0]
    for dataset in datasets:
        for batch_size in batches:
            base = _measure_p6_shape_variant(args, dataset, int(batch_size), seed, audit_base_id, "BASE")
            for fid in survivors:
                func = _measure_p6_shape_variant(args, dataset, int(batch_size), seed, audit_base_id, fid)
                base_step = _float(base.get("step_time_ms"))
                func_step = _float(func.get("step_time_ms"))
                base_peak = _float(base.get("peak_allocated_MB"))
                func_peak = _float(func.get("peak_allocated_MB"))
                step_ratio = func_step / base_step if base_step > 0 else float("nan")
                memory_ratio = func_peak / base_peak if base_peak > 0 else float("nan")
                rows.append({
                    "stage": "P6_FUNCTIONAL_FULLGRID_S2_V83",
                    "base_candidate_id": audit_base_id,
                    "functional_candidate_id": fid,
                    "dataset": dataset,
                    "batch_size": batch_size,
                    "seed": seed,
                    "base_step_time_ms": base_step,
                    "functional_step_time_ms": func_step,
                    "step_ratio": step_ratio if math.isfinite(step_ratio) else METRIC_UNAVAILABLE,
                    "base_peak_allocated_MB": base_peak if math.isfinite(base_peak) else METRIC_UNAVAILABLE,
                    "functional_peak_allocated_MB": func_peak if math.isfinite(func_peak) else METRIC_UNAVAILABLE,
                    "memory_ratio": memory_ratio if math.isfinite(memory_ratio) else METRIC_UNAVAILABLE,
                    "S2_shape_pass": int(math.isfinite(step_ratio) and math.isfinite(memory_ratio) and memory_ratio <= 1.05 and step_ratio <= 1.50),
                    "base_bad_step_rate_measured": base.get("bad_step_rate_measured", METRIC_UNAVAILABLE),
                    "functional_bad_step_rate_measured": func.get("bad_step_rate_measured", METRIC_UNAVAILABLE),
                    "functional_fallback_rate_measured": func.get("fallback_rate_measured", METRIC_UNAVAILABLE),
                    "bench_warmup": func.get("bench_warmup", METRIC_UNAVAILABLE),
                    "bench_reps": func.get("bench_reps", METRIC_UNAVAILABLE),
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    return rows


def _summarize_p6(raw_rows: List[Dict[str, Any]], s2_rows: Sequence[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    p5_like = _summarize_p5(raw_rows)
    s2_rows = list(s2_rows or [])
    out: List[Dict[str, Any]] = []
    for row in p5_like:
        fid = str(row.get("functional_candidate_id", "FT7"))
        s2_items = [
            item for item in s2_rows
            if str(item.get("functional_candidate_id")) == fid and str(item.get("status")) != "not_run"
        ]
        mem_ratios = [_float(item.get("memory_ratio")) for item in s2_items if math.isfinite(_float(item.get("memory_ratio")))]
        step_ratios = [_float(item.get("step_ratio")) for item in s2_items if math.isfinite(_float(item.get("step_ratio")))]
        s2_shape_count = sum(1 for item in s2_items if _is_one(item.get("S2_shape_pass")))
        s2_shape_total = len(s2_items)
        s2_measured = int(s2_shape_total > 0)
        task_geo_pass = int(
            _float(row.get("macro_val_acc_delta_vs_base_mean"), -99.0) >= -0.005
            and _float(row.get("geometry_curvature_ratio_vs_base_mean"), 99.0) <= 0.90
            and _float(row.get("memory_ratio_vs_base_mean"), 99.0) <= 1.05
            and _float(row.get("step_ratio_vs_base_mean"), 99.0) <= 1.50
        )
        p6_confirm = int(task_geo_pass and s2_measured and s2_shape_count == s2_shape_total)
        reason = "pass" if p6_confirm else (
            "s2_fullgrid_shape_count_not_measured_for_functional_update"
            if not s2_measured
            else "functional_fullgrid_s2_gate_fail"
        )
        out.append({
            "stage": "P6_FUNCTIONAL_CONFIRM5_V83",
            "base_candidate_id": row.get("base_candidate_id", "KW6"),
            "functional_candidate_id": fid,
            "rows": row.get("rows", 0),
            "macro_val_acc_delta_vs_base_mean": row.get("macro_val_acc_delta_vs_base_mean", METRIC_UNAVAILABLE),
            "test_acc_delta_vs_base_mean": row.get("test_acc_delta_vs_base_mean", METRIC_UNAVAILABLE),
            "ECE_delta_vs_base_mean": row.get("ECE_delta_vs_base_mean", METRIC_UNAVAILABLE),
            "NLL_delta_vs_base_mean": row.get("NLL_delta_vs_base_mean", METRIC_UNAVAILABLE),
            "ValLossAUC_step_ratio_vs_base_mean": row.get("ValLossAUC_step_ratio_vs_base_mean", METRIC_UNAVAILABLE),
            "geometry_curvature_ratio_vs_base_mean": row.get("geometry_curvature_ratio_vs_base_mean", METRIC_UNAVAILABLE),
            "geometry_reduction_vs_base_mean": row.get("geometry_reduction_vs_base_mean", METRIC_UNAVAILABLE),
            "memory_ratio_vs_base_mean": row.get("memory_ratio_vs_base_mean", METRIC_UNAVAILABLE),
            "step_ratio_vs_base_mean": row.get("step_ratio_vs_base_mean", METRIC_UNAVAILABLE),
            "fallback_rate_mean": row.get("fallback_rate_mean", METRIC_UNAVAILABLE),
            "bad_step_rate_mean": row.get("bad_step_rate_mean", METRIC_UNAVAILABLE),
            "functional_update_norm_mean": row.get("functional_update_norm_mean", METRIC_UNAVAILABLE),
            "trust_region_scale_mean": row.get("trust_region_scale_mean", METRIC_UNAVAILABLE),
            "S2_shape_count": s2_shape_count if s2_measured else METRIC_UNAVAILABLE,
            "S2_shape_total": s2_shape_total if s2_measured else METRIC_UNAVAILABLE,
            "S2_shape_count_measured": s2_measured,
            "functional_fullgrid_memory_ratio_max": max(mem_ratios) if mem_ratios else METRIC_UNAVAILABLE,
            "functional_fullgrid_step_ratio_max": max(step_ratios) if step_ratios else METRIC_UNAVAILABLE,
            "P6_task_geometry_system_mean_pass": task_geo_pass,
            "P6_confirm5_pass": p6_confirm,
            "reason": reason,
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return out


def _run_p6_functional_confirm5(out_dir: Path, args: Any, audit_base_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    p5_rows = _read_csv_rows(out_dir / "p5_functional_task_geometry_coselection.csv")
    survivors = sorted({
        str(row.get("functional_candidate_id"))
        for row in p5_rows
        if _is_one(row.get("P5_functional_useful_pass"))
    })
    survivors = [fid for fid in survivors if fid == "FT7"]
    if not survivors:
        return [], [], _run_p6_functional_fullgrid_s2(args, audit_base_id, survivors)
    raw_rows: List[Dict[str, Any]] = []
    datasets = parse_str_list(args.datasets)
    seeds = (parse_int_list(args.seeds) or [0, 1, 2, 3, 4])[:5]
    for dataset in datasets:
        for seed in seeds:
            base_row = _train_p5_variant(args, dataset, seed, audit_base_id, "BASE")
            base_row["stage"] = "P6_FUNCTIONAL_CONFIRM5_RAW_V83"
            raw_rows.append(base_row)
            for fid in survivors:
                func_row = _train_p5_variant(args, dataset, seed, audit_base_id, fid)
                func_row["stage"] = "P6_FUNCTIONAL_CONFIRM5_RAW_V83"
                raw_rows.append(func_row)
    s2_rows = _run_p6_functional_fullgrid_s2(args, audit_base_id, survivors)
    return _summarize_p6(raw_rows, s2_rows), raw_rows, s2_rows


def _summarize_p7(
    raw_rows: List[Dict[str, Any]],
    s2_rows: Sequence[Dict[str, Any]],
    bootstrap_reps: int,
) -> List[Dict[str, Any]]:
    if not raw_rows:
        return []
    base_by_key = {
        (row.get("dataset"), row.get("seed")): row
        for row in raw_rows
        if row.get("functional_candidate_id") == "BASE"
    }
    out: List[Dict[str, Any]] = []
    functional_ids = sorted({row.get("functional_candidate_id") for row in raw_rows if row.get("functional_candidate_id") != "BASE"})
    for fid in functional_ids:
        rows = [row for row in raw_rows if row.get("functional_candidate_id") == fid]
        paired = [(row, base_by_key.get((row.get("dataset"), row.get("seed")), {})) for row in rows]
        val_acc_delta = [_float(row.get("val_acc")) - _float(base.get("val_acc")) for row, base in paired]
        test_acc_delta = [_float(row.get("test_acc")) - _float(base.get("test_acc")) for row, base in paired]
        ece_delta = [_float(row.get("ECE")) - _float(base.get("ECE")) for row, base in paired]
        nll_delta = [_float(row.get("NLL")) - _float(base.get("NLL")) for row, base in paired]
        auc_step_ratio = [
            _float(row.get("ValLossAUC_step")) / _float(base.get("ValLossAUC_step"))
            for row, base in paired
            if _float(base.get("ValLossAUC_step")) > 0
        ]
        geo_ratio = [
            _float(row.get("edge_curvature_norm")) / _float(base.get("edge_curvature_norm"))
            for row, base in paired
            if _float(base.get("edge_curvature_norm")) > 0
        ]
        train_step_ratio = [
            _float(row.get("step_time_ms_mean")) / _float(base.get("step_time_ms_mean"))
            for row, base in paired
            if _float(base.get("step_time_ms_mean")) > 0
        ]
        train_memory_ratio = [
            _float(row.get("peak_allocated_MB")) / _float(base.get("peak_allocated_MB"))
            for row, base in paired
            if _float(base.get("peak_allocated_MB")) > 0
        ]
        ci_low, ci_high, boot_p = v72._bootstrap_ci(
            val_acc_delta,
            max(1, int(bootstrap_reps)),
            v72._stable_seed("v83-p7-bootstrap", str(fid), len(rows)),
        )
        s2_items = [
            item for item in s2_rows
            if str(item.get("functional_candidate_id")) == str(fid) and str(item.get("status")) != "not_run"
        ]
        s2_shape_count = sum(1 for item in s2_items if _is_one(item.get("S2_shape_pass")))
        s2_shape_total = len(s2_items)
        mem_ratios = [_float(item.get("memory_ratio")) for item in s2_items if math.isfinite(_float(item.get("memory_ratio")))]
        step_ratios = [_float(item.get("step_ratio")) for item in s2_items if math.isfinite(_float(item.get("step_ratio")))]
        mem_max = max(mem_ratios) if mem_ratios else METRIC_UNAVAILABLE
        step_max = max(step_ratios) if step_ratios else METRIC_UNAVAILABLE
        acc_delta_mean = _mean(val_acc_delta)
        ece_delta_mean = _mean(ece_delta)
        nll_delta_mean = _mean(nll_delta)
        geo_ratio_mean = _mean(geo_ratio)
        auc_step_ratio_mean = _mean(auc_step_ratio)
        functional_geometry_success = int(
            acc_delta_mean >= -0.005
            and ci_low >= -0.005
            and geo_ratio_mean <= 0.90
            and ece_delta_mean <= 0.005
            and nll_delta_mean <= 0.01
            and isinstance(mem_max, float) and mem_max <= 1.05
            and isinstance(step_max, float) and step_max <= 1.50
            and s2_shape_total > 0 and s2_shape_count == s2_shape_total
        )
        strong_success = int(
            acc_delta_mean >= 0.0
            and geo_ratio_mean <= 0.80
            and False
        )
        out.append({
            "stage": "P7_FUNCTIONAL_CONFIRM10_V83",
            "base_candidate_id": rows[0].get("base_candidate_id", "KW6") if rows else "KW6",
            "functional_candidate_id": fid,
            "rows": len(rows),
            "bootstrap_reps": bootstrap_reps,
            "macro_val_acc_delta_vs_base_mean": acc_delta_mean,
            "ci95_low_acc_delta_vs_base": ci_low,
            "ci95_high_acc_delta_vs_base": ci_high,
            "bootstrap_p_two_sided": boot_p,
            "test_acc_delta_vs_base_mean": _mean(test_acc_delta),
            "ECE_delta_vs_base_mean": ece_delta_mean,
            "NLL_delta_vs_base_mean": nll_delta_mean,
            "ValLossAUC_step_ratio_vs_base_mean": auc_step_ratio_mean,
            "ValLossAUC_time_ratio_vs_base_mean": METRIC_UNAVAILABLE,
            "geometry_curvature_ratio_vs_base_mean": geo_ratio_mean,
            "geometry_reduction_vs_base_mean": 1.0 - geo_ratio_mean if math.isfinite(geo_ratio_mean) else METRIC_UNAVAILABLE,
            "train_memory_ratio_vs_base_mean": _mean(train_memory_ratio),
            "train_step_ratio_vs_base_mean": _mean(train_step_ratio),
            "functional_fullgrid_memory_ratio_max": mem_max,
            "functional_fullgrid_step_ratio_max": step_max,
            "S2_shape_count": s2_shape_count if s2_shape_total else METRIC_UNAVAILABLE,
            "S2_shape_total": s2_shape_total if s2_shape_total else METRIC_UNAVAILABLE,
            "S2_shape_count_measured": int(s2_shape_total > 0),
            "fallback_rate_mean": _mean(_float(row.get("fallback_rate")) for row in rows),
            "bad_step_rate_mean": _mean(_float(row.get("bad_step_rate")) for row in rows),
            "functional_update_norm_mean": _mean(_float(row.get("functional_update_norm")) for row in rows),
            "trust_region_scale_mean": _mean(_float(row.get("trust_region_scale")) for row in rows),
            "FunctionalGeometrySuccessPass": functional_geometry_success,
            "StrongFunctionalSuccessPass": strong_success,
            "P7_confirm10_pass": functional_geometry_success,
            "strong_success_reason": "time_auc_not_measured",
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return out


def _run_p7_functional_confirm10(
    args: Any,
    audit_base_id: str,
    p6_rows: Sequence[Dict[str, Any]],
    s2_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    survivors = sorted({
        str(row.get("functional_candidate_id"))
        for row in p6_rows
        if _is_one(row.get("P6_confirm5_pass"))
    })
    survivors = [fid for fid in survivors if fid == "FT7"]
    if not survivors:
        return [], []
    raw_rows: List[Dict[str, Any]] = []
    datasets = parse_str_list(args.datasets)
    seeds = (parse_int_list(args.seeds) or list(range(10)))[:10]
    for dataset in datasets:
        for seed in seeds:
            base_row = _train_p5_variant(args, dataset, seed, audit_base_id, "BASE")
            base_row["stage"] = "P7_FUNCTIONAL_CONFIRM10_RAW_V83"
            raw_rows.append(base_row)
            for fid in survivors:
                func_row = _train_p5_variant(args, dataset, seed, audit_base_id, fid)
                func_row["stage"] = "P7_FUNCTIONAL_CONFIRM10_RAW_V83"
                raw_rows.append(func_row)
    bootstrap_reps = int(getattr(args, "bootstrap_reps", 10000))
    return _summarize_p7(raw_rows, s2_rows, bootstrap_reps), raw_rows


def _manual_ce_backward_profile(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: Any,
    device: torch.device,
) -> Tuple[float, float, float]:
    _zero_grad(stack, head)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    h, caches = stack.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss, grad_logits = v72._weighted_smooth_ce_and_grad(logits, y, spec.label_smoothing, None)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    dh = head.backward_manual(grad_logits, head_cache)
    stack.backward_manual(dh, caches)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    backward_ms = (time.perf_counter() - started) * 1000.0
    return float(loss.detach().cpu()), forward_ms, backward_ms


def _manual_ce_backward_with_holdout_profile(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    spec: Any,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    _zero_grad(stack, head)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    split = int(x.shape[0])
    x_pair = torch.cat((x, x_holdout), dim=0)
    h_pair, caches = stack.forward_manual(x_pair)
    logits_pair, head_cache = head.forward_manual(h_pair)
    loss, grad_logits = v72._weighted_smooth_ce_and_grad(logits_pair[:split], y, spec.label_smoothing, None)
    holdout_loss = _smooth_ce_value_only(logits_pair[split:], y_holdout, spec.label_smoothing)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    dh = head.backward_manual(grad_logits, h_pair[:split].detach())
    if len(caches) == 1 and isinstance(caches[0], dict) and "x0" in caches[0]:
        stack.backward_manual(dh, [{"x0": x.detach()}])
    else:
        stack.backward_manual(dh, caches)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    backward_ms = (time.perf_counter() - started) * 1000.0
    del x_pair, h_pair, caches, logits_pair, head_cache, grad_logits
    return float(loss.detach().cpu()), float(holdout_loss.detach().cpu()), forward_ms, backward_ms


def _timed_loss_only(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: Any,
    device: torch.device,
) -> Tuple[float, float]:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = _loss_only(stack, head, x, y, spec)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    return value, (time.perf_counter() - started) * 1000.0


def _timed_loss_and_features_only(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: Any,
    device: torch.device,
) -> Tuple[float, torch.Tensor, float]:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    value, features = _loss_and_features_only(stack, head, x, y, spec)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    return value, features, (time.perf_counter() - started) * 1000.0


def _timed_loss_pair_and_features_only(
    stack: Any,
    head: Any,
    x_a: torch.Tensor,
    y_a: torch.Tensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    spec: Any,
    device: torch.device,
) -> Tuple[float, float, torch.Tensor, torch.Tensor, float]:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    loss_a, loss_b, features_a, features_b = _loss_pair_and_features_only(
        stack,
        head,
        x_a,
        y_a,
        x_b,
        y_b,
        spec,
    )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    return loss_a, loss_b, features_a, features_b, (time.perf_counter() - started) * 1000.0


def _timed_loss_pair_only(
    stack: Any,
    head: Any,
    x_a: torch.Tensor,
    y_a: torch.Tensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    spec: Any,
    device: torch.device,
) -> Tuple[float, float, float]:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    loss_a, loss_b = _loss_pair_only(
        stack,
        head,
        x_a,
        y_a,
        x_b,
        y_b,
        spec,
    )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    return loss_a, loss_b, (time.perf_counter() - started) * 1000.0


def _train_p8_profile_variant(
    args: Any,
    dataset: str,
    seed: int,
    audit_base_id: str,
    functional_id: str,
) -> Dict[str, Any]:
    device = get_device(args.device)
    v80._patch_for_v80()
    spec = v80._spec_map_v80().get(audit_base_id, v80._spec_map_v80()["KW6"])
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        allow_fake_data=False,
    )
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    set_seed(v72._stable_seed("v83-p8-profile", dataset, seed, v80._init_seed_candidate_id_v80(spec), spec.head_kind, spec.group_count, spec.shuffle))
    stack, head = v80._make_manual_candidate_v80(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    entries = _param_entries(stack, head)
    role_entries_by_name = _entries_by_role(entries) if functional_id == "FT7" else None
    params = v80.V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    lr = float(params.lr_manual * spec.lr_mult)
    func_alpha = lr * 0.05
    opt = v72.v71.FastAdamW([stack, head], lr=lr, weight_decay=getattr(args, "weight_decay", 1.0e-4))
    max_steps = 240
    trace_every = max(1, int(getattr(args, "trace_every", 10)))
    cumulative_ms = 0.0
    val_step_trace: List[Tuple[float, float]] = []
    val_time_trace: List[Tuple[float, float]] = []
    phase = {key: 0.0 for key in [
        "batch_select",
        "forward",
        "backward",
        "update",
        "functional_metric_build",
        "functional_update",
        "post_step_check",
        "validation",
    ]}
    step_times: List[float] = []
    bad_count = 0
    reject_count = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    total_started = time.perf_counter()
    for step in range(1, max_steps + 1):
        step_started = time.perf_counter()
        started = time.perf_counter()
        xb, yb = v72.v71._select_batch(x_train, y_train, args.batch_size, step)
        xh, yh = v72.v71._select_batch(x_train, y_train, args.batch_size, step + 1009)
        phase["batch_select"] += (time.perf_counter() - started) * 1000.0
        if functional_id == "BASE":
            loss_before, forward_ms, backward_ms = _manual_ce_backward_profile(stack, head, xb, yb, spec, device)
            phase["forward"] += forward_ms
            phase["backward"] += backward_ms
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            opt.step(step, max_steps, warmup_cosine=True)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            phase["update"] += (time.perf_counter() - started) * 1000.0
            loss_after, post_ms = _timed_loss_only(stack, head, xb, yb, spec, device)
            phase["post_step_check"] += post_ms
        elif functional_id == "FT7":
            event_step = step % P5_FT7_EVENT_STRIDE == 0
            can_pair_pre_holdout = (
                event_step
                and not FT7_FAST_TASK_CHECK
                and FT7_USE_HOLDOUT_GUARD
                and _ft7_uses_pre_holdout_for_step(step)
            )
            if can_pair_pre_holdout:
                loss_before, holdout_loss_before, forward_ms, backward_ms = _manual_ce_backward_with_holdout_profile(
                    stack,
                    head,
                    xb,
                    yb,
                    xh,
                    yh,
                    spec,
                    device,
                )
                phase["forward"] += forward_ms
                phase["backward"] += backward_ms
            else:
                holdout_loss_before = 0.0
                loss_before, forward_ms, backward_ms = _manual_ce_backward_profile(stack, head, xb, yb, spec, device)
                phase["forward"] += forward_ms
                phase["backward"] += backward_ms
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            opt.step(step, max_steps, warmup_cosine=True)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            phase["update"] += (time.perf_counter() - started) * 1000.0
            can_pair_guard_after = event_step and not FT7_FAST_TASK_CHECK and FT7_USE_HOLDOUT_GUARD
            if can_pair_guard_after:
                (
                    task_loss_after,
                    holdout_loss_after,
                    pair_ms,
                ) = _timed_loss_pair_only(stack, head, xb, yb, xh, yh, spec, device)
                phase["post_step_check"] += pair_ms
            elif event_step:
                task_loss_after, post_ms = _timed_loss_only(stack, head, xb, yb, spec, device)
                phase["post_step_check"] += post_ms
            else:
                task_loss_after, post_ms = _timed_loss_only(stack, head, xb, yb, spec, device)
                phase["post_step_check"] += post_ms
            loss_after = task_loss_after
            if event_step:
                if can_pair_guard_after:
                    pass
                elif FT7_FAST_TASK_CHECK or not FT7_USE_HOLDOUT_GUARD:
                    holdout_loss_after = 0.0
                else:
                    holdout_loss_after, metric_ms = _timed_loss_only(stack, head, xh, yh, spec, device)
                    phase["functional_metric_build"] += metric_ms
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                loss_after, _func_norm, _trust_delta, reject_delta = _apply_ft7_streamed_guarded_update(
                    stack,
                    head,
                    entries,
                    xb,
                    yb,
                    xh,
                    yh,
                    spec,
                    loss_before=loss_before,
                    holdout_loss_before=holdout_loss_before,
                    task_loss_after=task_loss_after,
                    holdout_loss_after=holdout_loss_after,
                    func_alpha=func_alpha * P5_FT7_EVENT_ALPHA_MULT,
                    task_features_after=None,
                    holdout_features_after=None,
                    role_entries_by_name=role_entries_by_name,
                    pair_stack_guard_forward=True,
                )
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()
                phase["functional_update"] += (time.perf_counter() - started) * 1000.0
                reject_count += reject_delta
        else:
            raise ValueError(f"unsupported P8 functional candidate: {functional_id}")
        bad_count += int(loss_after > loss_before + 1e-8)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.perf_counter() - step_started) * 1000.0
        cumulative_ms += step_ms
        step_times.append(step_ms)
        if step % trace_every == 0 or step == max_steps:
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            ev = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            validation_ms = (time.perf_counter() - started) * 1000.0
            phase["validation"] += validation_ms
            cumulative_ms += validation_ms
            val_step_trace.append((float(step), float(ev["loss"])))
            val_time_trace.append((cumulative_ms, float(ev["loss"])))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb: Any = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = METRIC_UNAVAILABLE
    total_ms = (time.perf_counter() - total_started) * 1000.0
    mapped_ms = sum(phase.values())
    unknown_fraction = max(0.0, total_ms - mapped_ms) / max(total_ms, 1e-12)
    return {
        "stage": "P8_FUNCTIONAL_TIME_PROFILER_RAW_V83",
        "base_candidate_id": audit_base_id,
        "functional_candidate_id": functional_id,
        "dataset": dataset,
        "seed": seed,
        "task_steps": max_steps,
        "ValLossAUC_step": v72.v71._auc(val_step_trace),
        "ValLossAUC_time": v72.v71._auc(val_time_trace),
        "train_step_time_ms_mean": _mean(step_times),
        "forward_time_ms_mean": phase["forward"] / max_steps,
        "backward_time_ms_mean": phase["backward"] / max_steps,
        "update_time_ms_mean": phase["update"] / max_steps,
        "functional_update_time_ms_mean": phase["functional_update"] / max_steps,
        "functional_metric_build_time_ms_mean": phase["functional_metric_build"] / max_steps,
        "validation_time_ms_mean": phase["validation"] / max(1, len(val_step_trace)),
        "logging_time_ms_mean": phase["post_step_check"] / max_steps,
        "batch_select_time_ms_mean": phase["batch_select"] / max_steps,
        "unknown_time_fraction": unknown_fraction,
        "kernel_count_total": METRIC_UNAVAILABLE,
        "mapped_kernel_time_fraction": METRIC_UNAVAILABLE,
        "small_kernel_count_under_10us": METRIC_UNAVAILABLE,
        "peak_allocated_MB": peak_mb,
        "bad_step_rate": bad_count / float(max_steps),
        "fallback_rate": reject_count / float(max_steps),
        "loss_type": "CE",
        "geometry_loss_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _summarize_p8(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_by_key = {
        (row.get("dataset"), row.get("seed")): row
        for row in raw_rows
        if row.get("functional_candidate_id") == "BASE"
    }
    out: List[Dict[str, Any]] = []
    functional_ids = sorted({row.get("functional_candidate_id") for row in raw_rows if row.get("functional_candidate_id") != "BASE"})
    for fid in functional_ids:
        rows = [row for row in raw_rows if row.get("functional_candidate_id") == fid]
        paired = [(row, base_by_key.get((row.get("dataset"), row.get("seed")), {})) for row in rows]
        auc_time_ratio = [
            _float(row.get("ValLossAUC_time")) / _float(base.get("ValLossAUC_time"))
            for row, base in paired
            if _float(base.get("ValLossAUC_time")) > 0
        ]
        auc_step_ratio = [
            _float(row.get("ValLossAUC_step")) / _float(base.get("ValLossAUC_step"))
            for row, base in paired
            if _float(base.get("ValLossAUC_step")) > 0
        ]
        train_step_ratio = [
            _float(row.get("train_step_time_ms_mean")) / _float(base.get("train_step_time_ms_mean"))
            for row, base in paired
            if _float(base.get("train_step_time_ms_mean")) > 0
        ]
        mem_ratio = [
            _float(row.get("peak_allocated_MB")) / _float(base.get("peak_allocated_MB"))
            for row, base in paired
            if _float(base.get("peak_allocated_MB")) > 0
        ]
        functional_update_ratio = [
            _float(row.get("functional_update_time_ms_mean")) / _float(row.get("train_step_time_ms_mean"))
            for row in rows
            if _float(row.get("train_step_time_ms_mean")) > 0
        ]
        unknown_values = [_float(row.get("unknown_time_fraction")) for row in rows if math.isfinite(_float(row.get("unknown_time_fraction")))]
        unknown_max = max(unknown_values) if unknown_values else float("nan")
        auc_time_mean = _mean(auc_time_ratio)
        functional_update_ratio_mean = _mean(functional_update_ratio)
        p8_pass = int(
            auc_time_mean <= 1.05
            and unknown_max <= 0.10
            and functional_update_ratio_mean <= 0.10
        )
        out.append({
            "stage": "P8_FUNCTIONAL_TIME_PROFILER_V83",
            "base_candidate_id": rows[0].get("base_candidate_id", "KW6") if rows else "KW6",
            "functional_candidate_id": fid,
            "rows": len(rows),
            "ValLossAUC_step_ratio_vs_base_mean": _mean(auc_step_ratio),
            "ValLossAUC_time_ratio_vs_base_mean": auc_time_mean,
            "train_step_time_ratio_vs_base_mean": _mean(train_step_ratio),
            "memory_ratio_vs_base_mean": _mean(mem_ratio),
            "train_step_time_ms_mean": _mean(_float(row.get("train_step_time_ms_mean")) for row in rows),
            "forward_time_ms_mean": _mean(_float(row.get("forward_time_ms_mean")) for row in rows),
            "backward_time_ms_mean": _mean(_float(row.get("backward_time_ms_mean")) for row in rows),
            "update_time_ms_mean": _mean(_float(row.get("update_time_ms_mean")) for row in rows),
            "functional_update_time_ms_mean": _mean(_float(row.get("functional_update_time_ms_mean")) for row in rows),
            "functional_metric_build_time_ms_mean": _mean(_float(row.get("functional_metric_build_time_ms_mean")) for row in rows),
            "validation_time_ms_mean": _mean(_float(row.get("validation_time_ms_mean")) for row in rows),
            "logging_time_ms_mean": _mean(_float(row.get("logging_time_ms_mean")) for row in rows),
            "unknown_time_fraction_max": unknown_max,
            "functional_update_time_ratio_of_step_mean": functional_update_ratio_mean,
            "kernel_count_total": METRIC_UNAVAILABLE,
            "mapped_kernel_time_fraction": METRIC_UNAVAILABLE,
            "small_kernel_count_under_10us": METRIC_UNAVAILABLE,
            "P8_TimeAUCProfilerPass": p8_pass,
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return out


def _run_p8_time_profiler(
    args: Any,
    audit_base_id: str,
    p7_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    survivors = sorted({
        str(row.get("functional_candidate_id"))
        for row in p7_rows
        if _is_one(row.get("P7_confirm10_pass"))
    })
    survivors = [fid for fid in survivors if fid == "FT7"]
    if not survivors:
        return [], []
    raw_rows: List[Dict[str, Any]] = []
    datasets = parse_str_list(args.datasets)
    seed = (parse_int_list(args.seeds) or [0])[0]
    for dataset in datasets:
        raw_rows.append(_train_p8_profile_variant(args, dataset, seed, audit_base_id, "BASE"))
        for fid in survivors:
            raw_rows.append(_train_p8_profile_variant(args, dataset, seed, audit_base_id, fid))
    return _summarize_p8(raw_rows), raw_rows


def _with_input_noise(x: torch.Tensor, std: float, seed: int) -> torch.Tensor:
    if float(std) <= 0.0:
        return x
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    return x + torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) * float(std)


def _train_p9_variant(
    args: Any,
    dataset: str,
    seed: int,
    audit_base_id: str,
    functional_id: str,
    *,
    train_size: int,
    noise_type: str,
    noise_level: float,
) -> Dict[str, Any]:
    device = get_device(args.device)
    v80._patch_for_v80()
    spec = v80._spec_map_v80().get(audit_base_id, v80._spec_map_v80()["KW6"])
    label_noise = float(noise_level) if noise_type == "label_noise" else 0.0
    input_noise = float(noise_level) if noise_type == "input_noise" else 0.0
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=int(train_size),
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        label_noise=label_noise,
        allow_fake_data=False,
    )
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    if input_noise > 0.0:
        noise_seed = v72._stable_seed("v83-p9-input-noise", dataset, seed, train_size, noise_level)
        x_train = _with_input_noise(x_train, input_noise, noise_seed + 1)
        x_val = _with_input_noise(x_val, input_noise, noise_seed + 2)
        x_test = _with_input_noise(x_test, input_noise, noise_seed + 3)

    set_seed(v72._stable_seed("v83-p9", dataset, seed, int(train_size), noise_type, noise_level, v80._init_seed_candidate_id_v80(spec), spec.head_kind, spec.group_count, spec.shuffle))
    stack, head = v80._make_manual_candidate_v80(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    entries = _param_entries(stack, head)
    role_entries_by_name = _entries_by_role(entries) if functional_id == "FT7" else None
    params = v80.V63Params(train_size=int(train_size), val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    lr = float(params.lr_manual * spec.lr_mult)
    func_alpha = lr * 0.05
    opt = v72.v71.FastAdamW([stack, head], lr=lr, weight_decay=getattr(args, "weight_decay", 1.0e-4))
    max_steps = 240
    trace_every = max(1, int(getattr(args, "trace_every", 10)))
    val_trace: List[Tuple[float, float]] = []
    step_times: List[float] = []
    bad_count = 0
    reject_count = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, max_steps + 1):
        xb, yb = v72.v71._select_batch(x_train, y_train, args.batch_size, step)
        xh, yh = v72.v71._select_batch(x_train, y_train, args.batch_size, step + 1009)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        if functional_id == "BASE":
            loss_before = _manual_ce_backward(stack, head, xb, yb, spec)
            opt.step(step, max_steps, warmup_cosine=True)
            loss_after = _loss_only(stack, head, xb, yb, spec)
        elif functional_id == "FT7":
            event_step = step % P5_FT7_EVENT_STRIDE == 0
            holdout_loss_before = (
                _loss_only(stack, head, xh, yh, spec)
                if event_step and not FT7_FAST_TASK_CHECK and FT7_USE_HOLDOUT_GUARD and _ft7_uses_pre_holdout_for_step(step)
                else 0.0
            )
            loss_before = _manual_ce_backward(stack, head, xb, yb, spec)
            opt.step(step, max_steps, warmup_cosine=True)
            if event_step:
                task_loss_after = _loss_only(stack, head, xb, yb, spec)
            else:
                task_loss_after = _loss_only(stack, head, xb, yb, spec)
            loss_after = task_loss_after
            if event_step:
                if not FT7_FAST_TASK_CHECK and FT7_USE_HOLDOUT_GUARD:
                    holdout_loss_after = _loss_only(stack, head, xh, yh, spec)
                else:
                    holdout_loss_after = 0.0
                loss_after, _func_norm, _trust_delta, reject_delta = _apply_ft7_streamed_guarded_update(
                    stack,
                    head,
                    entries,
                    xb,
                    yb,
                    xh,
                    yh,
                    spec,
                    loss_before=loss_before,
                    holdout_loss_before=holdout_loss_before,
                    task_loss_after=task_loss_after,
                    holdout_loss_after=holdout_loss_after,
                    func_alpha=func_alpha * P5_FT7_EVENT_ALPHA_MULT,
                    task_features_after=None,
                    holdout_features_after=None,
                    role_entries_by_name=role_entries_by_name,
                )
                reject_count += reject_delta
        else:
            raise ValueError(f"unsupported P9 functional candidate: {functional_id}")
        bad_count += int(loss_after > loss_before + 1e-8)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append((time.perf_counter() - started) * 1000.0)
        if step % trace_every == 0 or step == max_steps:
            ev = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
            val_trace.append((float(step), float(ev["loss"])))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb: Any = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = METRIC_UNAVAILABLE
    val_eval = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    test_eval = v72.v71._manual_eval(stack, head, x_test, y_test, args.eval_batch_size, bundle.num_classes)
    smooth, curv = _geometry_norms(_param_entries(stack, head))
    return {
        "stage": "P9_FUNCTIONAL_SCALING_ROBUSTNESS_V83",
        "base_candidate_id": audit_base_id,
        "functional_candidate_id": functional_id,
        "candidate_role": "base" if functional_id == "BASE" else "functional",
        "dataset": dataset,
        "seed": seed,
        "train_size": int(train_size),
        "noise_type": noise_type,
        "noise_level": float(noise_level),
        "val_acc": val_eval["acc"],
        "test_acc": test_eval["acc"],
        "ECE": val_eval["ECE"],
        "NLL": val_eval["NLL"],
        "geometry_smoothness": smooth,
        "geometry_curvature": curv,
        "ValLossAUC_step": v72.v71._auc(val_trace),
        "memory_peak_MB": peak_mb,
        "step_time_ms_mean": _mean(step_times),
        "bad_step_rate": bad_count / float(max_steps),
        "fallback_rate": reject_count / float(max_steps),
        "functional_event_stride": P5_FT7_EVENT_STRIDE if functional_id == "FT7" else 0,
        "functional_event_alpha_mult": P5_FT7_EVENT_ALPHA_MULT if functional_id == "FT7" else 0.0,
        "accuracy_drop": 0.0,
        "loss_type": "CE",
        "geometry_loss_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _summarize_p9(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not raw_rows:
        return []
    clean_test = {
        (
            row.get("functional_candidate_id"),
            row.get("dataset"),
            row.get("seed"),
            row.get("train_size"),
        ): _float(row.get("test_acc"))
        for row in raw_rows
        if row.get("noise_type") == "clean" and _float(row.get("noise_level"), 99.0) == 0.0
    }
    for row in raw_rows:
        key = (row.get("functional_candidate_id"), row.get("dataset"), row.get("seed"), row.get("train_size"))
        clean_acc = clean_test.get(key, _float(row.get("test_acc")))
        row["accuracy_drop"] = clean_acc - _float(row.get("test_acc")) if math.isfinite(clean_acc) else METRIC_UNAVAILABLE

    base_by_key = {
        (row.get("dataset"), row.get("seed"), row.get("train_size"), row.get("noise_type"), row.get("noise_level")): row
        for row in raw_rows
        if row.get("functional_candidate_id") == "BASE"
    }
    functional_rows = [row for row in raw_rows if row.get("functional_candidate_id") != "BASE"]
    paired: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
        (row, base_by_key.get((row.get("dataset"), row.get("seed"), row.get("train_size"), row.get("noise_type"), row.get("noise_level")), {}))
        for row in functional_rows
    ]
    for row, base in paired:
        base_test = _float(base.get("test_acc"))
        base_val = _float(base.get("val_acc"))
        base_ece = _float(base.get("ECE"))
        base_nll = _float(base.get("NLL"))
        base_curv = _float(base.get("geometry_curvature"))
        base_step = _float(base.get("step_time_ms_mean"))
        base_mem = _float(base.get("memory_peak_MB"))
        row["base_test_acc"] = base_test if math.isfinite(base_test) else METRIC_UNAVAILABLE
        row["base_val_acc"] = base_val if math.isfinite(base_val) else METRIC_UNAVAILABLE
        row["test_acc_delta_vs_base"] = _float(row.get("test_acc")) - base_test if math.isfinite(base_test) else METRIC_UNAVAILABLE
        row["val_acc_delta_vs_base"] = _float(row.get("val_acc")) - base_val if math.isfinite(base_val) else METRIC_UNAVAILABLE
        row["accuracy_drop_delta_vs_base"] = _float(row.get("accuracy_drop")) - _float(base.get("accuracy_drop")) if math.isfinite(_float(base.get("accuracy_drop"))) else METRIC_UNAVAILABLE
        row["ECE_delta_vs_base"] = _float(row.get("ECE")) - base_ece if math.isfinite(base_ece) else METRIC_UNAVAILABLE
        row["NLL_delta_vs_base"] = _float(row.get("NLL")) - base_nll if math.isfinite(base_nll) else METRIC_UNAVAILABLE
        row["geometry_curvature_ratio_vs_base"] = _float(row.get("geometry_curvature")) / base_curv if base_curv > 0 else METRIC_UNAVAILABLE
        row["geometry_delta_vs_base"] = _float(row.get("geometry_curvature")) - base_curv if math.isfinite(base_curv) else METRIC_UNAVAILABLE
        row["memory_ratio_vs_base"] = _float(row.get("memory_peak_MB")) / base_mem if base_mem > 0 else METRIC_UNAVAILABLE
        row["step_ratio_vs_base"] = _float(row.get("step_time_ms_mean")) / base_step if base_step > 0 else METRIC_UNAVAILABLE

    datasets = sorted({str(row.get("dataset")) for row in raw_rows})
    seeds = sorted({int(_float(row.get("seed"), 0.0)) for row in raw_rows})
    base_auc_values: List[float] = []
    functional_auc_values: List[float] = []
    for dataset in datasets:
        for seed in seeds:
            base_points = sorted(
                (
                    _float(row.get("train_size")),
                    _float(row.get("test_acc")),
                )
                for row in raw_rows
                if row.get("functional_candidate_id") == "BASE"
                and row.get("dataset") == dataset
                and int(_float(row.get("seed"), -1.0)) == seed
                and row.get("noise_type") == "clean"
            )
            func_points = sorted(
                (
                    _float(row.get("train_size")),
                    _float(row.get("test_acc")),
                )
                for row in raw_rows
                if row.get("functional_candidate_id") != "BASE"
                and row.get("dataset") == dataset
                and int(_float(row.get("seed"), -1.0)) == seed
                and row.get("noise_type") == "clean"
            )
            if len(base_points) >= 2:
                base_auc_values.append(v72.v71._auc(base_points))
            if len(func_points) >= 2:
                functional_auc_values.append(v72.v71._auc(func_points))
    base_auc = _mean(base_auc_values)
    functional_auc = _mean(functional_auc_values)
    sample_eff_pass = int(math.isfinite(base_auc) and math.isfinite(functional_auc) and functional_auc >= base_auc)

    robustness_details: List[Dict[str, Any]] = []
    for noise_type in ["label_noise", "input_noise"]:
        levels = P9_LABEL_NOISE_LEVELS if noise_type == "label_noise" else P9_INPUT_NOISE_LEVELS
        for level in levels:
            base_drops = [
                _float(row.get("accuracy_drop"))
                for row in raw_rows
                if row.get("functional_candidate_id") == "BASE"
                and row.get("noise_type") == noise_type
                and abs(_float(row.get("noise_level")) - float(level)) < 1e-12
            ]
            func_drops = [
                _float(row.get("accuracy_drop"))
                for row in raw_rows
                if row.get("functional_candidate_id") != "BASE"
                and row.get("noise_type") == noise_type
                and abs(_float(row.get("noise_level")) - float(level)) < 1e-12
            ]
            base_drop_mean = _mean(base_drops)
            func_drop_mean = _mean(func_drops)
            robustness_details.append({
                "noise_type": noise_type,
                "noise_level": level,
                "base_drop_mean": base_drop_mean,
                "functional_drop_mean": func_drop_mean,
                "benefit": int(math.isfinite(base_drop_mean) and math.isfinite(func_drop_mean) and func_drop_mean <= base_drop_mean),
            })
    robustness_benefit_count = sum(int(item["benefit"]) for item in robustness_details)
    robustness_pass = int(robustness_benefit_count >= 2)

    geometry_ratios = [_float(row.get("geometry_curvature_ratio_vs_base")) for row in functional_rows if math.isfinite(_float(row.get("geometry_curvature_ratio_vs_base")))]
    ece_deltas = [_float(row.get("ECE_delta_vs_base")) for row in functional_rows if math.isfinite(_float(row.get("ECE_delta_vs_base")))]
    nll_deltas = [_float(row.get("NLL_delta_vs_base")) for row in functional_rows if math.isfinite(_float(row.get("NLL_delta_vs_base")))]
    mem_ratios = [_float(row.get("memory_ratio_vs_base")) for row in functional_rows if math.isfinite(_float(row.get("memory_ratio_vs_base")))]
    step_ratios = [_float(row.get("step_ratio_vs_base")) for row in functional_rows if math.isfinite(_float(row.get("step_ratio_vs_base")))]
    geometry_delta_mean = _mean([ratio - 1.0 for ratio in geometry_ratios])
    ece_delta_mean = _mean(ece_deltas)
    nll_delta_mean = _mean(nll_deltas)
    geometry_generalization_pass = int(
        math.isfinite(geometry_delta_mean)
        and geometry_delta_mean < 0.0
        and ((math.isfinite(ece_delta_mean) and ece_delta_mean <= 0.0) or (math.isfinite(nll_delta_mean) and nll_delta_mean <= 0.0))
    )
    system_pass = int(
        (not mem_ratios or max(mem_ratios) <= 1.05)
        and (not step_ratios or max(step_ratios) <= 1.50)
    )
    p9_pass = int(sample_eff_pass and robustness_pass and geometry_generalization_pass and system_pass)
    summary = {
        "stage": "P9_FUNCTIONAL_SCALING_ROBUSTNESS_SUMMARY_V83",
        "status": "completed",
        "P9_summary_row": 1,
        "base_candidate_id": raw_rows[0].get("base_candidate_id", "KW6"),
        "functional_candidate_id": functional_rows[0].get("functional_candidate_id", "FT7") if functional_rows else "FT7",
        "rows": len(raw_rows),
        "sample_efficiency_auc_base": base_auc,
        "sample_efficiency_auc_functional": functional_auc,
        "sample_efficiency_auc_delta": functional_auc - base_auc if math.isfinite(base_auc) and math.isfinite(functional_auc) else METRIC_UNAVAILABLE,
        "sample_efficiency_benefit_pass": sample_eff_pass,
        "robustness_benefit_count": robustness_benefit_count,
        "robustness_benefit_settings": _json_dumps({f"{item['noise_type']}:{item['noise_level']}": item["benefit"] for item in robustness_details}),
        "robustness_benefit_pass": robustness_pass,
        "geometry_delta_vs_base_mean": geometry_delta_mean,
        "ECE_delta_vs_base_mean": ece_delta_mean,
        "NLL_delta_vs_base_mean": nll_delta_mean,
        "geometry_generalization_consistency_pass": geometry_generalization_pass,
        "memory_ratio_vs_base_max": max(mem_ratios) if mem_ratios else METRIC_UNAVAILABLE,
        "step_ratio_vs_base_max": max(step_ratios) if step_ratios else METRIC_UNAVAILABLE,
        "P9_system_pass": system_pass,
        "P9_scaling_robustness_pass": p9_pass,
        "loss_type": "CE",
        "geometry_loss_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [*raw_rows, summary]


def _run_p9_scaling_robustness(
    args: Any,
    audit_base_id: str,
    p8_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    survivors = sorted({
        str(row.get("functional_candidate_id"))
        for row in p8_rows
        if _is_one(row.get("P8_TimeAUCProfilerPass"))
    })
    survivors = [fid for fid in survivors if fid == "FT7"]
    if not survivors:
        return []
    datasets = parse_str_list(args.datasets)
    seeds = (parse_int_list(args.seeds) or [0, 1, 2, 3, 4])[:5]
    default_train_size = int(getattr(args, "train_size", 1536))
    train_sizes = sorted(set([*P9_TRAIN_SIZES, default_train_size]))
    conditions: List[Tuple[int, str, float]] = [(size, "clean", 0.0) for size in train_sizes]
    conditions.extend((default_train_size, "label_noise", level) for level in P9_LABEL_NOISE_LEVELS)
    conditions.extend((default_train_size, "input_noise", level) for level in P9_INPUT_NOISE_LEVELS)
    raw_rows: List[Dict[str, Any]] = []
    for train_size, noise_type, noise_level in conditions:
        for dataset in datasets:
            for seed in seeds:
                raw_rows.append(_train_p9_variant(
                    args,
                    dataset,
                    seed,
                    audit_base_id,
                    "BASE",
                    train_size=train_size,
                    noise_type=noise_type,
                    noise_level=noise_level,
                ))
                for fid in survivors:
                    raw_rows.append(_train_p9_variant(
                        args,
                        dataset,
                        seed,
                        audit_base_id,
                        fid,
                        train_size=train_size,
                        noise_type=noise_type,
                        noise_level=noise_level,
                    ))
    return _summarize_p9(raw_rows)


def _write_not_run_downstream(out_dir: Path, reason: str) -> None:
    for name, stage in [
        ("p4_multistep_functional_smoke.csv", "P4_MULTISTEP_FUNCTIONAL_SMOKE_V83"),
        ("p5_functional_task_geometry_raw.csv", "P5_FUNCTIONAL_TASK_GEOMETRY_RAW_V83"),
        ("p5_functional_task_geometry_coselection.csv", "P5_FUNCTIONAL_TASK_GEOMETRY_COSELECTION_V83"),
        ("p6_functional_confirm5_raw.csv", "P6_FUNCTIONAL_CONFIRM5_RAW_V83"),
        ("p6_functional_fullgrid_s2.csv", "P6_FUNCTIONAL_FULLGRID_S2_V83"),
        ("p6_functional_confirm5.csv", "P6_FUNCTIONAL_CONFIRM5_V83"),
        ("p7_functional_confirm10_raw.csv", "P7_FUNCTIONAL_CONFIRM10_RAW_V83"),
        ("p7_functional_confirm10.csv", "P7_FUNCTIONAL_CONFIRM10_V83"),
        ("p8_functional_time_profiler_raw.csv", "P8_FUNCTIONAL_TIME_PROFILER_RAW_V83"),
        ("p8_functional_time_profiler.csv", "P8_FUNCTIONAL_TIME_PROFILER_V83"),
        ("p9_functional_scaling_robustness.csv", "P9_FUNCTIONAL_SCALING_ROBUSTNESS_V83"),
    ]:
        write_csv(out_dir / name, [{
            "stage": stage,
            "status": "not_run",
            "reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }])


def _route_decision(out_dir: Path, audit_base_id: str) -> Dict[str, Any]:
    p1 = _read_csv_rows(out_dir / "p1_base_candidate_confirmation.csv")
    p2 = _read_csv_rows(out_dir / "p2_functional_operator_audit.csv")
    p3 = _read_csv_rows(out_dir / "p3_one_step_functional_direction.csv")
    p4 = _read_csv_rows(out_dir / "p4_multistep_functional_smoke.csv")
    p5 = _read_csv_rows(out_dir / "p5_functional_task_geometry_coselection.csv")
    p6 = _read_csv_rows(out_dir / "p6_functional_confirm5.csv")
    p7 = _read_csv_rows(out_dir / "p7_functional_confirm10.csv")
    p8 = _read_csv_rows(out_dir / "p8_functional_time_profiler.csv")
    p9 = _read_csv_rows(out_dir / "p9_functional_scaling_robustness.csv")
    h0_rows = [row for row in p1 if _is_one(row.get("H0_system_base_pass"))]
    p2_pass_rows = [row for row in p2 if _is_one(row.get("implementation_pass")) and _is_one(row.get("functional_update_used"))]
    p3_pass_rows = [row for row in p3 if _is_one(row.get("direction_gate_pass"))]
    p4_functional_rows = [
        row for row in p4
        if str(row.get("functional_candidate_id")) not in {"", "BASE"} and str(row.get("status")) != "not_run"
    ]
    final_step = max([_float(row.get("step"), -1.0) for row in p4_functional_rows] or [-1.0])
    p4_final_rows = [row for row in p4_functional_rows if _float(row.get("step"), -2.0) == final_step]
    p4_pass_rows = [row for row in p4_final_rows if _is_one(row.get("P4_multistep_pass"))]
    p5_functional_rows = [
        row for row in p5
        if str(row.get("functional_candidate_id")) not in {"", "BASE"} and str(row.get("status")) != "not_run"
    ]
    p5_pass_rows = [row for row in p5_functional_rows if _is_one(row.get("P5_functional_useful_pass"))]
    p6_functional_rows = [
        row for row in p6
        if str(row.get("functional_candidate_id")) not in {"", "BASE"} and str(row.get("status")) != "not_run"
    ]
    p6_pass_rows = [row for row in p6_functional_rows if _is_one(row.get("P6_confirm5_pass"))]
    p7_functional_rows = [
        row for row in p7
        if str(row.get("functional_candidate_id")) not in {"", "BASE"} and str(row.get("status")) != "not_run"
    ]
    p7_pass_rows = [row for row in p7_functional_rows if _is_one(row.get("P7_confirm10_pass"))]
    p8_functional_rows = [
        row for row in p8
        if str(row.get("functional_candidate_id")) not in {"", "BASE"} and str(row.get("status")) != "not_run"
    ]
    p8_pass_rows = [row for row in p8_functional_rows if _is_one(row.get("P8_TimeAUCProfilerPass"))]
    p9_rows = [row for row in p9 if str(row.get("status")) != "not_run"]
    p9_summary_rows = [row for row in p9_rows if _is_one(row.get("P9_summary_row"))]
    p9_pass_rows = [row for row in p9_summary_rows if _is_one(row.get("P9_scaling_robustness_pass"))]
    base_row = next((row for row in p1 if str(row.get("candidate_id")) == audit_base_id), p1[0] if p1 else {})
    p3_best_row = max(
        p3,
        key=lambda r: (
            int(_is_one(r.get("direction_gate_pass"))),
            _float(r.get("edge_curvature_reduction_mean"), -99.0),
            -_float(r.get("bad_step_rate"), 99.0),
            _float(r.get("holdout_descent_ratio_vs_task_mean"), -99.0),
        ),
    ) if p3 else {}
    p5_best_row = max(
        p5_functional_rows,
        key=lambda r: (
            int(_is_one(r.get("P5_functional_useful_pass"))),
            -_float(r.get("geometry_curvature_ratio_vs_base_mean"), 99.0),
            _float(r.get("macro_val_acc_delta_vs_base_mean"), -99.0),
        ),
    ) if p5_functional_rows else {}
    p6_best_row = max(
        p6_functional_rows,
        key=lambda r: (
            int(_is_one(r.get("P6_confirm5_pass"))),
            int(_is_one(r.get("P6_task_geometry_system_mean_pass"))),
            -_float(r.get("geometry_curvature_ratio_vs_base_mean"), 99.0),
            _float(r.get("macro_val_acc_delta_vs_base_mean"), -99.0),
        ),
    ) if p6_functional_rows else {}
    p7_best_row = max(
        p7_functional_rows,
        key=lambda r: (
            int(_is_one(r.get("P7_confirm10_pass"))),
            int(_is_one(r.get("FunctionalGeometrySuccessPass"))),
            -_float(r.get("geometry_curvature_ratio_vs_base_mean"), 99.0),
            _float(r.get("macro_val_acc_delta_vs_base_mean"), -99.0),
        ),
    ) if p7_functional_rows else {}
    functional_row = p7_best_row or p6_best_row or p5_best_row or p3_best_row
    if not p1:
        route = "S8-ContractOrArtifactMissing"
        blocker = "missing_base_artifacts"
    elif not h0_rows:
        route = "S6-NoBaseSystemCandidateFunctionalFullTaskGated"
        blocker = "no_base_system_candidate"
    elif not p2_pass_rows:
        route = "S7-FunctionalImplementationTooExpensive"
        blocker = "functional_operator_cost_gate"
    elif not p3_pass_rows:
        route = "S5-FunctionalDirectionFailsOneStepGate"
        blocker = "functional_direction_one_step_gate"
    elif p4_final_rows and not p4_pass_rows:
        task_pass_any = any(_is_one(row.get("P4_task_preservation_pass")) for row in p4_final_rows)
        if task_pass_any:
            route = "S4-FunctionalTaskNeutralNoGeometryBenefit"
            blocker = "p4_geometry_gate"
        else:
            route = "S3-FunctionalGeometryTaskConflict"
            blocker = "p4_task_preservation_gate"
    elif p4_pass_rows and not p5_functional_rows:
        route = "S2-BaseSystemPassFunctionalOneStepPassFullTaskNotOpened"
        blocker = "p5_functional_task_geometry_coselection_not_run"
    elif p5_functional_rows and not p5_pass_rows:
        route = "S1-P5FunctionalCoSelectionNoUsefulSurvivor"
        blocker = "p5_functional_no_useful_survivor"
    elif p5_pass_rows and not p6_functional_rows:
        route = "S1-FunctionalTaskGeometryP5SurvivorNeedsConfirmation"
        blocker = "p6_functional_confirm5_not_run"
    elif p6_functional_rows and not p6_pass_rows:
        p6_mean_pass_any = any(_is_one(row.get("P6_task_geometry_system_mean_pass")) for row in p6_functional_rows)
        p6_s2_measured_any = any(_is_one(row.get("S2_shape_count_measured")) for row in p6_functional_rows)
        if p6_mean_pass_any and not p6_s2_measured_any:
            route = "S1-FunctionalP6Confirm5NeedsMeasuredFullGridS2"
            blocker = "p6_s2_shape_count_not_measured"
        elif p6_mean_pass_any:
            route = "S1-FunctionalP6Confirm5NoFullGridS2"
            blocker = "p6_functional_fullgrid_s2_gate"
        else:
            route = "S1-FunctionalP6Confirm5NoSurvivor"
            blocker = "p6_functional_confirm5_gate"
    elif p6_pass_rows and not p7_functional_rows:
        route = "S0-FunctionalP6Confirm5PassNeedsP7"
        blocker = "p7_functional_confirm10_not_run"
    elif p7_functional_rows and not p7_pass_rows:
        route = "S0-FunctionalP7Confirm10NoFinalSuccess"
        blocker = "p7_functional_confirm10_gate"
    elif p7_pass_rows and not p8_functional_rows:
        route = "S0-FunctionalP7Confirm10PassNeedsP8"
        blocker = "p8_functional_time_profiler_not_run"
    elif p8_functional_rows and not p8_pass_rows:
        route = "S0-FunctionalP8TimeProfilerFail"
        blocker = "p8_functional_time_profiler_gate"
    elif p8_pass_rows and not p9_rows:
        route = "S0-FunctionalP8PassNeedsP9"
        blocker = "p9_functional_scaling_robustness_not_run"
    elif p9_summary_rows and not p9_pass_rows:
        route = "S0-FunctionalP9ScalingRobustnessFail"
        blocker = "p9_functional_scaling_robustness_gate"
    elif p9_pass_rows:
        route = "S0-FunctionalReEntryMinimumSuccess"
        blocker = "none"
    else:
        route = "S2-BaseSystemPassFunctionalOneStepPassFullTaskNotOpened"
        blocker = "full_functional_task_waiting_for_p4"
    p3_for_functional = next((row for row in p3 if row.get("functional_candidate_id") == functional_row.get("functional_candidate_id")), p3_best_row)
    p2_for_functional = next((row for row in p2 if row.get("functional_candidate_id") == functional_row.get("functional_candidate_id")), {})
    functional_update_ratio = (
        functional_row.get("functional_fullgrid_step_ratio_max")
        if p7_best_row
        else functional_row.get("step_ratio_vs_base_mean")
        if p6_best_row or p5_best_row
        else p2_for_functional.get("functional_update_time_ratio", METRIC_UNAVAILABLE)
    )
    geometry_delta = (
        functional_row.get("geometry_reduction_vs_base_mean")
        if p6_best_row or p5_best_row
        else functional_row.get("edge_curvature_reduction_mean", METRIC_UNAVAILABLE)
    )
    decision = {
        "route": route,
        "base_candidate_id": audit_base_id,
        "functional_candidate_id": functional_row.get("functional_candidate_id", METRIC_UNAVAILABLE),
        "base_macro_gap": base_row.get("macro_gap", METRIC_UNAVAILABLE),
        "functional_macro_gap": functional_row.get("macro_val_acc_delta_vs_base_mean", METRIC_UNAVAILABLE),
        "functional_ci95_low": functional_row.get("ci95_low_acc_delta_vs_base", METRIC_UNAVAILABLE),
        "acc_delta_vs_base": functional_row.get("macro_val_acc_delta_vs_base_mean", METRIC_UNAVAILABLE),
        "geometry_delta_vs_base": geometry_delta,
        "ECE_delta_vs_base": functional_row.get("ECE_delta_vs_base_mean", METRIC_UNAVAILABLE),
        "NLL_delta_vs_base": functional_row.get("NLL_delta_vs_base_mean", METRIC_UNAVAILABLE),
        "memory_ratio_max": base_row.get("memory_ratio_max", METRIC_UNAVAILABLE),
        "step_ratio_max": base_row.get("step_ratio_max", METRIC_UNAVAILABLE),
        "functional_update_time_ratio": functional_update_ratio,
        "bad_step_rate": functional_row.get("bad_step_rate_mean", functional_row.get("bad_step_rate", METRIC_UNAVAILABLE)),
        "holdout_descent_ratio": p3_for_functional.get("holdout_descent_ratio_vs_task_mean", METRIC_UNAVAILABLE),
        "functional_direction_cos": p3_for_functional.get("cos_corrected_with_task_mean", METRIC_UNAVAILABLE),
        "H0_system_base_pass": int(bool(h0_rows)),
        "S2_pass": int(_h0_pass(base_row)),
        "TimeAUC_pass": int(_is_one(base_row.get("TimeAUCPass"))),
        "ProfilerPass": int(_is_one(base_row.get("ProfilerPass"))),
        "functional_full_task_opened": int(bool(p5_functional_rows)),
        "P7_confirm10_pass": int(bool(p7_pass_rows)),
        "P8_TimeAUCProfilerPass": int(bool(p8_pass_rows)),
        "P9_scaling_robustness_pass": int(bool(p9_pass_rows)),
        "ValLossAUC_time_ratio": (p8_functional_rows[0].get("ValLossAUC_time_ratio_vs_base_mean", METRIC_UNAVAILABLE) if p8_functional_rows else METRIC_UNAVAILABLE),
        "functional_update_time_ratio_of_step": (p8_functional_rows[0].get("functional_update_time_ratio_of_step_mean", METRIC_UNAVAILABLE) if p8_functional_rows else METRIC_UNAVAILABLE),
        "unknown_time_fraction_max": (p8_functional_rows[0].get("unknown_time_fraction_max", METRIC_UNAVAILABLE) if p8_functional_rows else METRIC_UNAVAILABLE),
        "sample_efficiency_auc_delta": (p9_summary_rows[0].get("sample_efficiency_auc_delta", METRIC_UNAVAILABLE) if p9_summary_rows else METRIC_UNAVAILABLE),
        "robustness_benefit_count": (p9_summary_rows[0].get("robustness_benefit_count", METRIC_UNAVAILABLE) if p9_summary_rows else METRIC_UNAVAILABLE),
        "geometry_generalization_consistency_pass": int(any(_is_one(row.get("geometry_generalization_consistency_pass")) for row in p9_summary_rows)),
        "success_v83_minimum": int(bool(p9_pass_rows)),
        "success_v83_formal": 0,
        "primary_blocker": blocker,
        "next_required_implementation": (
            "continue_ce_only_system_closure_before_full_functional_task"
            if not h0_rows
            else "p4_task_preserving_functional_trust_region_repair"
            if p4_final_rows and not p4_pass_rows
            else "p4_multistep_functional_smoke_for_p3_survivor"
            if not p4_pass_rows
            else "p5_functional_task_geometry_repair"
            if p5_functional_rows and not p5_pass_rows
            else "measure_functional_fullgrid_s2_package"
            if p6_functional_rows and not p6_pass_rows and any(_is_one(row.get("P6_task_geometry_system_mean_pass")) for row in p6_functional_rows)
            and not any(_is_one(row.get("S2_shape_count_measured")) for row in p6_functional_rows)
            else "p6_functional_fullgrid_s2_repair"
            if p6_functional_rows and not p6_pass_rows and any(_is_one(row.get("P6_task_geometry_system_mean_pass")) for row in p6_functional_rows)
            else "p6_functional_confirm5_repair"
            if p6_functional_rows and not p6_pass_rows
            else "formalize_v83_functional_route"
            if p9_pass_rows
            else "p9_functional_scaling_robustness_repair"
            if p9_summary_rows and not p9_pass_rows
            else "p9_functional_scaling_robustness"
            if p8_pass_rows
            else "p8_functional_time_profiler_repair"
            if p8_functional_rows and not p8_pass_rows
            else "p8_functional_time_profiler"
            if p7_pass_rows
            else "p7_functional_confirm10_repair"
            if p7_functional_rows and not p7_pass_rows
            else "p7_functional_confirm10"
            if p6_pass_rows
            else "p6_functional_confirm5"
            if p5_pass_rows
            else "p5_functional_task_geometry_coselection"
        ),
        "cpu_offload_used": 0,
        "no_fake": True,
        "no_proxy": True,
    }
    save_json(out_dir / "route_decision.json", decision)
    save_json(out_dir / "aggregate_decision.json", decision)
    return decision


def _write_failure_table(out_dir: Path, decision: Dict[str, Any]) -> None:
    failures = []
    if decision.get("route", "").startswith("S6"):
        failures.append(("F1_no_base_system_candidate", 1))
    if decision.get("route", "").startswith("S7"):
        failures.append(("F10_functional_system_overhead", 1))
    if decision.get("route", "").startswith("S5"):
        failures.extend([
            ("F5_functional_direction_bad_cosine", 1),
            ("F6_functional_bad_step_rate_high", 1),
            ("F7_functional_holdout_descent_fail", 1),
        ])
    if not failures:
        failures.append(("none", 0))
    write_csv(out_dir / "failure_table.csv", [
        {"failure_type": name, "count": count, "fake_data_used": 0, "proxy_row_used": 0}
        for name, count in failures
    ])


def _write_manifest(out_dir: Path, args: Any, decision: Dict[str, Any]) -> None:
    artifact_names = [
        "candidate_registry_v83_functional.csv",
        "contract_no_teacher_no_loss_functional.csv",
        "functional_update_contract.csv",
        "p0_contract_audit.csv",
        "p1_base_candidate_confirmation.csv",
        "p2_functional_operator_audit.csv",
        "p3_one_step_functional_direction.csv",
        "p3_one_step_functional_direction_raw.csv",
        "p4_multistep_functional_smoke.csv",
        "p5_functional_task_geometry_coselection.csv",
        "p5_functional_task_geometry_raw.csv",
        "p6_functional_confirm5.csv",
        "p6_functional_confirm5_raw.csv",
        "p6_functional_fullgrid_s2.csv",
        "p7_functional_confirm10_raw.csv",
        "p7_functional_confirm10.csv",
        "p8_functional_time_profiler_raw.csv",
        "p8_functional_time_profiler.csv",
        "p9_functional_scaling_robustness.csv",
        "route_decision.json",
        "aggregate_decision.json",
        "failure_table.csv",
    ]
    artifact_paths = [out_dir / name for name in artifact_names]
    audit = v72._audit_fake_proxy(artifact_paths)
    audit_row = {
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "v83_provenance_audit.csv", [audit_row])
    artifact_paths.append(out_dir / "v83_provenance_audit.csv")
    save_json(out_dir / "run_manifest.json", {
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "runner_reuse": "run_gafu_v82_real.py measured path + v8.3 gated functional postprocess",
        "out_dir": str(out_dir),
        "source_commit": _git_commit(),
        "git_status": _git_status(),
        "datasets": parse_str_list(args.datasets),
        "seeds": parse_int_list(args.seeds),
        "candidates": parse_str_list(args.candidates),
        "bench_batch_sizes": parse_int_list(args.bench_batch_sizes),
        "grad_batch_sizes": parse_int_list(args.grad_batch_sizes),
        "artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
        "route_decision": decision,
        "provenance_audit": audit,
        "cpu_offload_allowed": 0,
    })


def _write_v83_postprocess(out_dir: Path, args: Any) -> Dict[str, Any]:
    (out_dir / "figures").mkdir(exist_ok=True)
    _write_candidate_registry(out_dir)
    _write_contracts(out_dir)
    _write_p1_base_confirmation(out_dir)
    audit_base_id = _select_audit_base(out_dir)
    p2_rows, p3_raw = _run_functional_one_step(out_dir, args, audit_base_id)
    write_csv(out_dir / "p2_functional_operator_audit.csv", p2_rows or [{
        "stage": "P2_FUNCTIONAL_OPERATOR_AUDIT_V83",
        "status": "not_run",
        "reason": "one_step_context_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p3_one_step_functional_direction_raw.csv", p3_raw)
    write_csv(out_dir / "p3_one_step_functional_direction.csv", _summarize_p3(p3_raw))
    p1_rows = _read_csv_rows(out_dir / "p1_base_candidate_confirmation.csv")
    has_h0 = any(_is_one(row.get("H0_system_base_pass")) for row in p1_rows)
    p2_has_pass = any(_is_one(row.get("implementation_pass")) and _is_one(row.get("functional_update_used")) for row in p2_rows)
    p3_has_pass = any(_is_one(row.get("direction_gate_pass")) for row in _summarize_p3(p3_raw))
    if has_h0 and p2_has_pass and p3_has_pass:
        p7_written = False
        p8_written = False
        p9_written = False
        p4_rows = _run_p4_multistep_smoke(out_dir, args, audit_base_id)
        write_csv(out_dir / "p4_multistep_functional_smoke.csv", p4_rows)
        final_step = max([_float(row.get("step"), -1.0) for row in p4_rows if row.get("functional_candidate_id") not in {"BASE", ""}] or [-1.0])
        p4_has_pass = any(
            _float(row.get("step"), -2.0) == final_step and _is_one(row.get("P4_multistep_pass"))
            for row in p4_rows
        )
        if p4_has_pass:
            p5_rows, p5_raw = _run_p5_functional_coselection(out_dir, args, audit_base_id)
            write_csv(out_dir / "p5_functional_task_geometry_raw.csv", p5_raw)
            write_csv(out_dir / "p5_functional_task_geometry_coselection.csv", p5_rows or [{
                "stage": "P5_FUNCTIONAL_TASK_GEOMETRY_COSELECTION_V83",
                "status": "not_run",
                "reason": "no_supported_p4_survivor",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }])
            if any(_is_one(row.get("P5_functional_useful_pass")) for row in p5_rows):
                p6_rows, p6_raw, p6_s2 = _run_p6_functional_confirm5(out_dir, args, audit_base_id)
                write_csv(out_dir / "p6_functional_confirm5_raw.csv", p6_raw)
                write_csv(out_dir / "p6_functional_fullgrid_s2.csv", p6_s2)
                write_csv(out_dir / "p6_functional_confirm5.csv", p6_rows or [{
                    "stage": "P6_FUNCTIONAL_CONFIRM5_V83",
                    "status": "not_run",
                    "reason": "no_supported_p5_survivor",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }])
                if any(_is_one(row.get("P6_confirm5_pass")) for row in p6_rows):
                    p7_rows, p7_raw = _run_p7_functional_confirm10(args, audit_base_id, p6_rows, p6_s2)
                    write_csv(out_dir / "p7_functional_confirm10_raw.csv", p7_raw)
                    write_csv(out_dir / "p7_functional_confirm10.csv", p7_rows or [{
                        "stage": "P7_FUNCTIONAL_CONFIRM10_V83",
                        "status": "not_run",
                        "reason": "no_supported_p6_survivor",
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }])
                    p7_written = True
                    if any(_is_one(row.get("P7_confirm10_pass")) for row in p7_rows):
                        p8_rows, p8_raw = _run_p8_time_profiler(args, audit_base_id, p7_rows)
                        write_csv(out_dir / "p8_functional_time_profiler_raw.csv", p8_raw)
                        write_csv(out_dir / "p8_functional_time_profiler.csv", p8_rows or [{
                            "stage": "P8_FUNCTIONAL_TIME_PROFILER_V83",
                            "status": "not_run",
                            "reason": "no_supported_p7_survivor",
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        }])
                        p8_written = True
                        if any(_is_one(row.get("P8_TimeAUCProfilerPass")) for row in p8_rows):
                            p9_rows = _run_p9_scaling_robustness(args, audit_base_id, p8_rows)
                            write_csv(out_dir / "p9_functional_scaling_robustness.csv", p9_rows or [{
                                "stage": "P9_FUNCTIONAL_SCALING_ROBUSTNESS_V83",
                                "status": "not_run",
                                "reason": "no_supported_p8_survivor",
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            }])
                            p9_written = True
                            downstream_reason = (
                                "p9_functional_scaling_robustness_pass"
                                if any(_is_one(row.get("P9_scaling_robustness_pass")) for row in p9_rows)
                                else "p9_functional_scaling_robustness_gate"
                            )
                        else:
                            downstream_reason = "p8_functional_time_profiler_gate"
                    else:
                        downstream_reason = "p7_functional_confirm10_gate"
                else:
                    downstream_reason = (
                        "p7_blocked_by_p6_functional_fullgrid_s2"
                        if any(_is_one(row.get("P6_task_geometry_system_mean_pass")) for row in p6_rows)
                        else "p6_functional_confirm5_no_survivor"
                    )
            else:
                write_csv(out_dir / "p6_functional_confirm5_raw.csv", [{
                    "stage": "P6_FUNCTIONAL_CONFIRM5_RAW_V83",
                    "status": "not_run",
                    "reason": "no_p5_functional_survivor",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }])
                write_csv(out_dir / "p6_functional_fullgrid_s2.csv", [{
                    "stage": "P6_FUNCTIONAL_FULLGRID_S2_V83",
                    "status": "not_run",
                    "reason": "no_p5_functional_survivor",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }])
                write_csv(out_dir / "p6_functional_confirm5.csv", [{
                    "stage": "P6_FUNCTIONAL_CONFIRM5_V83",
                    "status": "not_run",
                    "reason": "no_p5_functional_survivor",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }])
                downstream_reason = "no_p5_functional_survivor"
        else:
            write_csv(out_dir / "p5_functional_task_geometry_raw.csv", [{
                "stage": "P5_FUNCTIONAL_TASK_GEOMETRY_RAW_V83",
                "status": "not_run",
                "reason": "full_functional_training_not_opened_until_P4_survivor",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }])
            write_csv(out_dir / "p5_functional_task_geometry_coselection.csv", [{
                "stage": "P5_FUNCTIONAL_TASK_GEOMETRY_COSELECTION_V83",
                "status": "not_run",
                "reason": "full_functional_training_not_opened_until_P4_survivor",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }])
            write_csv(out_dir / "p6_functional_confirm5_raw.csv", [{
                "stage": "P6_FUNCTIONAL_CONFIRM5_RAW_V83",
                "status": "not_run",
                "reason": "full_functional_training_not_opened_until_P4_survivor",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }])
            write_csv(out_dir / "p6_functional_fullgrid_s2.csv", [{
                "stage": "P6_FUNCTIONAL_FULLGRID_S2_V83",
                "status": "not_run",
                "reason": "full_functional_training_not_opened_until_P4_survivor",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }])
            write_csv(out_dir / "p6_functional_confirm5.csv", [{
                "stage": "P6_FUNCTIONAL_CONFIRM5_V83",
                "status": "not_run",
                "reason": "full_functional_training_not_opened_until_P4_survivor",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }])
            downstream_reason = "full_functional_training_not_opened_until_P4_survivor"
        for name, stage in [
            ("p7_functional_confirm10_raw.csv", "P7_FUNCTIONAL_CONFIRM10_RAW_V83"),
            ("p7_functional_confirm10.csv", "P7_FUNCTIONAL_CONFIRM10_V83"),
            ("p8_functional_time_profiler_raw.csv", "P8_FUNCTIONAL_TIME_PROFILER_RAW_V83"),
            ("p8_functional_time_profiler.csv", "P8_FUNCTIONAL_TIME_PROFILER_V83"),
            ("p9_functional_scaling_robustness.csv", "P9_FUNCTIONAL_SCALING_ROBUSTNESS_V83"),
        ]:
            if p7_written and name in {"p7_functional_confirm10_raw.csv", "p7_functional_confirm10.csv"}:
                continue
            if p8_written and name in {"p8_functional_time_profiler_raw.csv", "p8_functional_time_profiler.csv"}:
                continue
            if p9_written and name == "p9_functional_scaling_robustness.csv":
                continue
            write_csv(out_dir / name, [{
                "stage": stage,
                "status": "not_run",
                "reason": downstream_reason,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }])
    else:
        _write_not_run_downstream(out_dir, "H0_system_base_not_passed" if not has_h0 else "full_functional_training_not_opened_until_P3_P4_survivor")
    decision = _route_decision(out_dir, audit_base_id)
    _write_failure_table(out_dir, decision)
    _write_manifest(out_dir, args, decision)
    return decision


def run(args: Any) -> None:
    v80.PLAN_PATH = PLAN_PATH
    v80.SCRIPT_PATH = SCRIPT_PATH
    v81.PLAN_PATH = PLAN_PATH
    v81.SCRIPT_PATH = SCRIPT_PATH
    v82.PLAN_PATH = PLAN_PATH
    v82.SCRIPT_PATH = SCRIPT_PATH
    v82.run(args)
    out_dir = Path(args.out_dir)
    for src_name, dst_name in {
        "route_decision.json": "v83_reused_v82_route_decision.json",
        "aggregate_decision.json": "v83_reused_v82_aggregate_decision.json",
        "v82_manifest.json": "v83_reused_v82_manifest.json",
        "v82_provenance_audit.csv": "v83_reused_v82_provenance_audit.csv",
    }.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)
    decision = _write_v83_postprocess(out_dir, args)
    save_json(out_dir / "v83_run_complete.json", {
        "route": decision,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })


def parse_args() -> Any:
    return v82.parse_args()


if __name__ == "__main__":
    run(parse_args())
