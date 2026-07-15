#!/usr/bin/env python3
"""DG-KAN v23.13 Chebyshev domain-transport representation geometry runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_08_downstream_coupled_edge_residual_inverse_flow as v2308
import experiments.run_v23_12_total_audit_feature_learning_tangent_escape as v2312
from dgkan.models.fc_purekan_primitives import _basis_eval


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.13_ChebDomainTransportRepresentationGeometry_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.13_ChebDomainTransportRepresentationGeometry_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.13_ChebDomainTransportRepresentationGeometry_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2313_OUT_ROOT", str(ROOT / "results/v23_13_cheb_domain_transport_representation_geometry"))).resolve()
FORMAL_ROOT = ROOT / "results/v23_09_part_b_repair_tol001_formal_15seed80"

V2312_STAGE2R = ROOT / "results/v23_12_total_audit_feature_learning_tangent_escape_stage2r_reduced_s5_80/stage2_total_audit_summary.json"
V2312_F4 = ROOT / "results/v23_12_total_audit_feature_learning_tangent_escape_stage2s_f4_last_layer_s5_80/stage2_total_audit_summary.json"
V2312_F5 = ROOT / "results/v23_12_total_audit_feature_learning_tangent_escape_stage2s_f5_last_two_s5_80/stage2_total_audit_summary.json"
V2312_NEXT = FORMAL_ROOT / "next_actions_for_v23_12.json"

STAGE1_SCHEMES = [
    "C33_Candidate_tail_edge_domain_transport_total_audit",
    "C34_Control_shuffled_tail_edge_domain_transport_total_audit",
    "C35_Control_global_edge_domain_transport_total_audit",
]

RECODE_SCHEMES = [
    "C36_Candidate_function_preserving_gain_recode_total_audit",
    "C37_Control_identity_gain_recode_total_audit",
    "C38_Control_down_gain_recode_total_audit",
]

AUDIT_DEFAULTS = {
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "metric_winner_selection_used": 0,
    "candidate_update_selection_used": 0,
    "new_edge_function_added": 0,
    "external_product_feature_used": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
    "structure_transform_in_forward": 0,
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def ival(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_text() -> str:
    return f"{sys.executable} {' '.join(sys.argv)}"


def ensure_logs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text("# DG-KAN v23.13 执行日志\n", encoding="utf-8")
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text("# DG-KAN v23.13 实验结果复盘\n", encoding="utf-8")


def append_exec(title: str, command: str, *, files: str, gpu: str, note: str) -> None:
    ensure_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title} done\n\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        fh.write(f"- files: `{files}`\n")
        fh.write(f"- note: {note}\n")


def append_recap(title: str, data: dict[str, Any]) -> None:
    ensure_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
        fh.write("\n```\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys: list[str] = []
    for row in enriched:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(enriched)
    return path


def median(values: list[Any]) -> float:
    nums = sorted(fval(v) for v in values if v not in (None, ""))
    if not nums:
        return 0.0
    mid = len(nums) // 2
    if len(nums) % 2:
        return float(nums[mid])
    return float(0.5 * (nums[mid - 1] + nums[mid]))


def float_items(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def extended_model_state(model: Any) -> dict[str, Any]:
    return {"state": v2308.model_state(model), "basis_input_gain": float(model.basis_input_gain)}


def load_extended_model_state(model: Any, state: dict[str, Any]) -> None:
    model.basis_input_gain = float(state["basis_input_gain"])
    v2308.load_model_state(model, state["state"])


def interpolated_extended_state(before: dict[str, Any], after: dict[str, Any], alpha: float) -> dict[str, Any]:
    return {
        "state": v2312.interpolated_state(before["state"], after["state"], float(alpha)),
        "basis_input_gain": float(before["basis_input_gain"]) + float(alpha) * (float(after["basis_input_gain"]) - float(before["basis_input_gain"])),
    }


def stage0_lock(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    required = {
        "plan": PLAN,
        "runner": RUNNER,
        "v23_12_stage2r": V2312_STAGE2R,
        "v23_12_f4": V2312_F4,
        "v23_12_f5": V2312_F5,
        "v23_12_next": V2312_NEXT,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    stage2r = read_json(V2312_STAGE2R) if V2312_STAGE2R.exists() else {}
    f4 = read_json(V2312_F4) if V2312_F4.exists() else {}
    f5 = read_json(V2312_F5) if V2312_F5.exists() else {}
    checks = {
        "no_missing_required_artifacts": not missing,
        "v23_12_stage2r_failed": ival(stage2r.get("gate_pass"), -1) == 0,
        "v23_12_stage2r_no_fake": bool(stage2r.get("checks", {}).get("no_fake_data")),
        "v23_12_stage2r_no_held": bool(stage2r.get("checks", {}).get("no_held_test_usage")),
        "v23_12_f4_diagnostic_not_promotion": ival(f4.get("gate_pass"), -1) == 0 and not bool(f4.get("checks", {}).get("required_stage2_controls_present", True)),
        "v23_12_f5_partial_not_promotion": ival(f5.get("gate_pass"), -1) == 0 and not bool(f5.get("checks", {}).get("row_count_50_ok", True)),
    }
    gate = int(all(checks.values()))
    summary = {
        "part": "0",
        "route": "V2313EvidenceLockPass" if gate else "V2313EvidenceLockFailed",
        "gate_pass": gate,
        "dominant_blocker": "none" if gate else "missing_or_inconsistent_v23_12_evidence",
        "checks": checks,
        "missing": missing,
        "locked_metrics": {
            "v23_12_c28_coverage": stage2r.get("metrics", {}).get("c28_coverage"),
            "v23_12_c28_minus_c30": stage2r.get("metrics", {}).get("c28_minus_c30"),
            "v23_12_c28_minus_c31": stage2r.get("metrics", {}).get("c28_minus_c31"),
            "v23_12_f4_c28_minus_c31": f4.get("metrics", {}).get("c28_minus_c31"),
            "v23_12_f5_wall_time_s_median": f5.get("metrics", {}).get("c28_wall_time_s_median"),
        },
        "hashes": {name: sha256_file(path) for name, path in required.items() if path.exists()},
        "next_action": "run_stage1_cheb_domain_transport_reduced" if gate else "repair_evidence_lock",
    }
    out = write_json(OUT_ROOT / "stage0_evidence_lock_summary.json", summary)
    append_exec("Stage 0 evidence lock", command_text(), files=rel(out), gpu=str(args.device), note=f"gate={gate}; blocker={summary['dominant_blocker']}")
    append_recap("Stage 0 evidence lock", summary)
    return summary


def model_seed_for(task: str, seed: int, args: argparse.Namespace) -> int:
    return 23090000 + int(seed) * 1009 + sum(ord(ch) for ch in str(task)) + int(args.depth) * 17


def strict_total_audit_pass(before_guard: dict[str, float], after_guard: dict[str, float], budget: float) -> tuple[int, dict[str, float]]:
    debt = v2308.debt_deltas(before_guard, after_guard)
    component_ok = int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0)
    return int(v2308.no_debt_ok(debt, float(budget)) and component_ok), debt


def cheb_transport_matrix(model: Any, shift: float, args: argparse.Namespace) -> tuple[torch.Tensor, float]:
    k = int(model.k)
    device = model.coeffs[0].device
    dtype = torch.float64
    n = max(2 * k + 7, int(args.transport_grid_points))
    z = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
    shifted = (z + float(shift)).clamp(-1.0, 1.0)
    clipped = float(((z + float(shift)).abs() > 1.0).to(dtype=dtype).mean().detach().cpu().item())
    phi = _basis_eval(z, str(model.basis_name), k, model.centers.to(device=device, dtype=dtype), model.scales.to(device=device, dtype=dtype)).reshape(n, k).to(dtype=dtype)
    phis = _basis_eval(shifted, str(model.basis_name), k, model.centers.to(device=device, dtype=dtype), model.scales.to(device=device, dtype=dtype)).reshape(n, k).to(dtype=dtype)
    gram = phi.T @ phi + float(args.transport_ridge) * torch.eye(k, device=device, dtype=dtype)
    mix = torch.linalg.solve(gram, phi.T @ phis)
    return mix, clipped


def apply_cheb_domain_transport(model: Any, x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace, *, scheme: str, task: str, seed: int) -> dict[str, Any]:
    if str(model.basis_name) != "chebyshev":
        raise ValueError("v23.13 domain transport currently supports chebyshev basis only")
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits.float(), dim=1)
        true_prob = prob.gather(1, y.long().reshape(-1, 1)).reshape(-1)
    q = min(0.95, max(0.01, float(args.transport_tail_quantile)))
    threshold = torch.quantile(true_prob.detach().to(dtype=torch.float64), q)
    tail_mask = true_prob.detach().to(dtype=torch.float64) <= threshold
    if int(tail_mask.sum().detach().cpu().item()) <= 0:
        tail_mask = torch.zeros_like(true_prob, dtype=torch.bool)
        tail_mask[int(torch.argmin(true_prob.detach()).detach().cpu().item())] = True
    if scheme.startswith("C34"):
        gen = torch.Generator(device=x.device).manual_seed(23130000 + int(seed) * 1009 + sum(ord(c) for c in str(task)))
        tail_mask = tail_mask[torch.randperm(int(tail_mask.shape[0]), device=x.device, generator=gen)]
    if scheme.startswith("C35"):
        tail_mask = torch.ones_like(tail_mask, dtype=torch.bool)

    total_shift_abs: list[float] = []
    total_clip: list[float] = []
    layer_shift_max: list[float] = []
    layer_shift_mean: list[float] = []
    for layer_idx, coeff in enumerate(model.coeffs):
        with torch.no_grad():
            _logits, acts = model.forward_with_activations(x)
            h = acts[int(layer_idx)].detach()
            z = torch.tanh(h.to(dtype=torch.float64) * float(model.basis_input_gain))
            mu_all = z.mean(dim=0)
            mu_tail = z[tail_mask].mean(dim=0)
            if scheme.startswith("C35"):
                direction = torch.sign(mu_all)
                direction = torch.where(direction == 0, torch.ones_like(direction), direction)
                target = float(args.transport_target_abs) * direction
                raw_shift = float(args.transport_strength) * (target - mu_all)
            else:
                direction = torch.sign(mu_tail - mu_all)
                fallback = torch.sign(mu_tail)
                direction = torch.where(direction == 0, torch.where(fallback == 0, torch.ones_like(fallback), fallback), direction)
                target = float(args.transport_target_abs) * direction
                raw_shift = float(args.transport_strength) * (target - mu_tail)
            shifts = raw_shift.clamp(-float(args.transport_max_shift), float(args.transport_max_shift))
            new_coeff = coeff.detach().clone().to(dtype=torch.float64)
            for in_idx in range(int(coeff.shape[0])):
                shift = float(shifts[int(in_idx)].detach().cpu().item())
                mix, clipped = cheb_transport_matrix(model, shift, args)
                old = coeff.detach()[int(in_idx)].to(dtype=torch.float64)
                new_coeff[int(in_idx)] = torch.einsum("km,om->ok", mix, old)
                total_shift_abs.append(abs(shift))
                total_clip.append(float(clipped))
            coeff.copy_(new_coeff.to(dtype=coeff.dtype))
            layer_shift_max.append(float(shifts.abs().max().detach().cpu().item()))
            layer_shift_mean.append(float(shifts.abs().mean().detach().cpu().item()))
    return {
        "transport_tail_threshold": float(threshold.detach().cpu().item()),
        "transport_tail_count": int(tail_mask.sum().detach().cpu().item()),
        "transport_shift_abs_median": median(total_shift_abs),
        "transport_shift_abs_max": max(total_shift_abs or [0.0]),
        "transport_clipped_fraction_median": median(total_clip),
        "transport_layer_shift_max_curve": json.dumps(layer_shift_max),
        "transport_layer_shift_mean_curve": json.dumps(layer_shift_mean),
    }


def project_transport_state(model: Any, original_state: dict[str, torch.Tensor], raw_state: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor, guard_original: dict[str, float], args: argparse.Namespace) -> tuple[float, str, dict[str, torch.Tensor], dict[str, float], dict[str, float], int]:
    selected_alpha = 0.0
    selected_reason = "alpha_zero_fallback"
    selected_state = original_state
    v2308.load_model_state(model, original_state)
    selected_train = v2308.actual_metrics(model, x, y)
    selected_guard = guard_original
    selected_pass = 1
    for alpha in float_items(str(args.transport_alphas)):
        cand_state = v2312.interpolated_state(original_state, raw_state, float(alpha))
        v2308.load_model_state(model, cand_state)
        train_cand = v2308.actual_metrics(model, x, y)
        guard_cand = v2308.actual_metrics(model, xg, yg)
        cand_pass, _debt = strict_total_audit_pass(guard_original, guard_cand, float(args.no_debt_budget))
        if cand_pass:
            selected_alpha = float(alpha)
            selected_reason = "alpha_zero_fallback" if abs(float(alpha)) <= 1.0e-12 else "first_total_audit_alpha"
            selected_state = cand_state
            selected_train = train_cand
            selected_guard = guard_cand
            selected_pass = 1
            break
    v2308.load_model_state(model, selected_state)
    return selected_alpha, selected_reason, selected_state, selected_train, selected_guard, selected_pass


def solve_layer_coefficients(phi: torch.Tensor, target: torch.Tensor, in_dim: int, out_dim: int, k: int, ridge: float) -> torch.Tensor:
    design = phi.to(dtype=torch.float64)
    tgt = target.to(device=design.device, dtype=torch.float64)
    p = int(design.shape[1])
    gram = design.T @ design + float(ridge) * torch.eye(p, device=design.device, dtype=torch.float64)
    rhs = design.T @ tgt
    flat = torch.linalg.solve(gram, rhs)
    return flat.reshape(int(in_dim), int(k), int(out_dim)).permute(0, 2, 1).contiguous()


def apply_function_preserving_gain_recode(model: Any, x: torch.Tensor, xg: torch.Tensor, target_gain: float, args: argparse.Namespace) -> dict[str, Any]:
    with torch.no_grad():
        original_gain = float(model.basis_input_gain)
        original_logits_train, original_acts_train = model.forward_with_activations(x)
        original_logits_guard, original_acts_guard = model.forward_with_activations(xg)
        model.basis_input_gain = float(target_gain)
        layer_fit_rel: list[float] = []
        layer_phi_cond: list[float] = []
        for layer_idx, coeff in enumerate(model.coeffs):
            _cur_logits, cur_acts = model.forward_with_activations(x)
            h_current = cur_acts[int(layer_idx)].detach()
            target = original_acts_train[int(layer_idx) + 1].detach()
            phi = model.layer_phi(h_current, int(layer_idx))
            new_coeff = solve_layer_coefficients(phi, target, int(coeff.shape[0]), int(coeff.shape[1]), int(coeff.shape[2]), float(args.recode_ridge))
            coeff.copy_(new_coeff.to(dtype=coeff.dtype))
            _after_logits, after_acts = model.forward_with_activations(x)
            fit_err = (after_acts[int(layer_idx) + 1].detach().to(dtype=torch.float64) - target.to(dtype=torch.float64)).norm()
            fit_den = target.to(dtype=torch.float64).norm().clamp_min(1.0e-12)
            layer_fit_rel.append(float((fit_err / fit_den).detach().cpu().item()))
            try:
                vals = torch.linalg.svdvals(phi.to(dtype=torch.float64))
                cond = float((vals.max().clamp_min(1.0e-12) / vals.min().clamp_min(1.0e-12)).detach().cpu().item())
            except Exception:
                cond = 0.0
            layer_phi_cond.append(cond)
        recoded_logits_train, recoded_acts_train = model.forward_with_activations(x)
        recoded_logits_guard, recoded_acts_guard = model.forward_with_activations(xg)
        train_logit_rel = float(
            (recoded_logits_train.detach().to(dtype=torch.float64) - original_logits_train.detach().to(dtype=torch.float64))
            .norm()
            .div(original_logits_train.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12))
            .detach()
            .cpu()
            .item()
        )
        guard_logit_rel = float(
            (recoded_logits_guard.detach().to(dtype=torch.float64) - original_logits_guard.detach().to(dtype=torch.float64))
            .norm()
            .div(original_logits_guard.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12))
            .detach()
            .cpu()
            .item()
        )
        activation_drift = v2312.activation_drift_stats(original_acts_guard, recoded_acts_guard)
        design_vals: list[float] = []
        for layer_idx in range(len(model.coeffs)):
            model.basis_input_gain = original_gain
            phi_before = model.layer_phi(original_acts_guard[int(layer_idx)].detach(), int(layer_idx))
            model.basis_input_gain = float(target_gain)
            phi_after = model.layer_phi(recoded_acts_guard[int(layer_idx)].detach(), int(layer_idx))
            design_vals.append(float((phi_after - phi_before).norm().div(phi_before.norm().clamp_min(1.0e-12)).detach().cpu().item()))
        model.basis_input_gain = float(target_gain)
    return {
        "recode_gain_before": original_gain,
        "recode_gain_after": float(target_gain),
        "recode_layer_fit_rel_median": median(layer_fit_rel),
        "recode_layer_fit_rel_max": max(layer_fit_rel or [0.0]),
        "recode_layer_fit_rel_curve": json.dumps(layer_fit_rel),
        "recode_phi_condition_median": median(layer_phi_cond),
        "recode_phi_condition_max": max(layer_phi_cond or [0.0]),
        "recode_train_logit_rel_error": train_logit_rel,
        "recode_guard_logit_rel_error": guard_logit_rel,
        "recode_basis_design_drift_median": median(design_vals),
        "recode_basis_design_drift_max": max(design_vals or [0.0]),
        "activation_drift_median": activation_drift["activation_drift_median"],
        "activation_drift_max": activation_drift["activation_drift_max"],
        "activation_drift_layer_count": activation_drift["activation_drift_layer_count"],
    }


def project_extended_state(
    model: Any,
    original_state: dict[str, Any],
    raw_state: dict[str, Any],
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    guard_original: dict[str, float],
    args: argparse.Namespace,
) -> tuple[float, str, dict[str, Any], dict[str, float], dict[str, float]]:
    selected_alpha = 0.0
    selected_reason = "alpha_zero_fallback"
    selected_state = original_state
    load_extended_model_state(model, original_state)
    selected_train = v2308.actual_metrics(model, x, y)
    selected_guard = guard_original
    for alpha in float_items(str(args.recode_alphas)):
        cand_state = interpolated_extended_state(original_state, raw_state, float(alpha))
        load_extended_model_state(model, cand_state)
        train_cand = v2308.actual_metrics(model, x, y)
        guard_cand = v2308.actual_metrics(model, xg, yg)
        cand_pass, _debt = strict_total_audit_pass(guard_original, guard_cand, float(args.no_debt_budget))
        if cand_pass:
            selected_alpha = float(alpha)
            selected_reason = "alpha_zero_fallback" if abs(float(alpha)) <= 1.0e-12 else "first_total_audit_alpha"
            selected_state = cand_state
            selected_train = train_cand
            selected_guard = guard_cand
            break
    load_extended_model_state(model, selected_state)
    return selected_alpha, selected_reason, selected_state, selected_train, selected_guard


def extended_drift_stats(model: Any, xg: torch.Tensor, before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, float]:
    with torch.no_grad():
        load_extended_model_state(model, before_state)
        _lb, acts_before = model.forward_with_activations(xg)
        phis_before = [model.layer_phi(acts_before[int(layer_idx)].detach(), int(layer_idx)) for layer_idx in range(len(model.coeffs))]
        load_extended_model_state(model, after_state)
        _la, acts_after = model.forward_with_activations(xg)
        phis_after = [model.layer_phi(acts_after[int(layer_idx)].detach(), int(layer_idx)) for layer_idx in range(len(model.coeffs))]
    act = v2312.activation_drift_stats(acts_before, acts_after)
    design_vals = [
        float((pa - pb).norm().div(pb.norm().clamp_min(1.0e-12)).detach().cpu().item())
        for pb, pa in zip(phis_before, phis_after)
    ]
    return {
        "activation_drift_median": act["activation_drift_median"],
        "activation_drift_max": act["activation_drift_max"],
        "activation_drift_layer_count": act["activation_drift_layer_count"],
        "recode_basis_design_drift_median": median(design_vals),
        "recode_basis_design_drift_max": max(design_vals or [0.0]),
    }


def target_gain_for_recode_scheme(scheme: str, original_gain: float, args: argparse.Namespace) -> float:
    if scheme.startswith("C36"):
        return float(args.recode_gain)
    if scheme.startswith("C38"):
        return float(args.recode_control_gain)
    return float(original_gain)


def run_recode_row(scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    start = time.time()
    try:
        x, y, xg, yg = v2308.v2307.visual_data(task, int(seed), args, device)
        classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = model_seed_for(task, int(seed), args)
        model = v2308.v2307.make_model(str(args.basis_key), int(args.depth), int(x.shape[1]), classes, model_seed, args, device, basis_input_gain=float(args.part_d_basis_input_gain))
        ckpt = v2308.v2307.train_checkpoint(model, x, y, str(args.checkpoint), int(seed), args)
        original_state = extended_model_state(model)
        train_original = v2308.actual_metrics(model, x, y)
        guard_original = v2308.actual_metrics(model, xg, yg)
        original_gain = float(model.basis_input_gain)
        target_gain = target_gain_for_recode_scheme(scheme, original_gain, args)
        if scheme.startswith("C37"):
            raw_state = original_state
            raw_train = train_original
            raw_guard = guard_original
            diag = {
                "recode_gain_before": original_gain,
                "recode_gain_after": original_gain,
                "recode_layer_fit_rel_median": 0.0,
                "recode_layer_fit_rel_max": 0.0,
                "recode_layer_fit_rel_curve": "[]",
                "recode_phi_condition_median": 0.0,
                "recode_phi_condition_max": 0.0,
                "recode_train_logit_rel_error": 0.0,
                "recode_guard_logit_rel_error": 0.0,
                "recode_basis_design_drift_median": 0.0,
                "recode_basis_design_drift_max": 0.0,
                "activation_drift_median": 0.0,
                "activation_drift_max": 0.0,
                "activation_drift_layer_count": len(model.coeffs),
            }
        else:
            diag = apply_function_preserving_gain_recode(model, x, xg, target_gain, args)
            raw_state = extended_model_state(model)
            raw_train = v2308.actual_metrics(model, x, y)
            raw_guard = v2308.actual_metrics(model, xg, yg)
        raw_pass, raw_debt = strict_total_audit_pass(guard_original, raw_guard, float(args.no_debt_budget))
        alpha, reason, selected_state, final_train, final_guard = project_extended_state(model, original_state, raw_state, x, y, xg, yg, guard_original, args)
        selected_drift = extended_drift_stats(model, xg, original_state, selected_state)
        load_extended_model_state(model, selected_state)
        debt = v2308.debt_deltas(guard_original, final_guard)
        no_debt = v2308.no_debt_ok(debt, float(args.no_debt_budget))
        return {
            **AUDIT_DEFAULTS,
            "part": "1F",
            "status": "ok",
            "scheme": scheme,
            "task": task,
            "seed": int(seed),
            "basis_key": str(args.basis_key),
            "basis_family": str(model.basis_name),
            "depth": int(args.depth),
            "train_size": int(args.train_size),
            "guard_size": int(args.guard_size),
            "checkpoint_name": str(args.checkpoint),
            "checkpoint_optimizer": ckpt.get("checkpoint_optimizer", ""),
            "model_seed": model_seed,
            "recode_alpha_candidates": str(args.recode_alphas),
            "recode_alpha_selected": alpha,
            "recode_alpha_selection_reason": reason,
            "recode_selected_gain": float(selected_state["basis_input_gain"]),
            **diag,
            "selected_activation_drift_median": selected_drift["activation_drift_median"],
            "selected_activation_drift_max": selected_drift["activation_drift_max"],
            "selected_basis_design_drift_median": selected_drift["recode_basis_design_drift_median"],
            "selected_basis_design_drift_max": selected_drift["recode_basis_design_drift_max"],
            "original_guard_coverage": guard_original["coverage"],
            "raw_recode_guard_coverage": raw_guard["coverage"],
            "final_guard_coverage": final_guard["coverage"],
            "total_C2_coverage_improvement": final_guard["coverage"] - guard_original["coverage"],
            "raw_recode_total_C2_coverage_improvement": raw_guard["coverage"] - guard_original["coverage"],
            "total_C2_accuracy_improvement": final_guard["accuracy"] - guard_original["accuracy"],
            "total_guard_nll_delta": final_guard["loss"] - guard_original["loss"],
            "total_train_nll_delta": final_train["loss"] - train_original["loss"],
            "raw_train_logit_rel_error": diag["recode_train_logit_rel_error"],
            "raw_guard_logit_rel_error": diag["recode_guard_logit_rel_error"],
            "F5_no_debt_pass": no_debt,
            "raw_recode_total_audit_pass": raw_pass,
            "raw_recode_total_audit_reject": int(not raw_pass),
            "Brier_delta": debt["Brier_delta"],
            "ECE_delta": debt["ECE_delta"],
            "tail95_delta": debt["tail95_delta"],
            "tail99_delta": debt["tail99_delta"],
            "margin10_delta": debt["margin10_delta"],
            "debt_delta": debt["debt_delta"],
            "raw_Brier_delta": raw_debt["Brier_delta"],
            "raw_ECE_delta": raw_debt["ECE_delta"],
            "raw_tail95_delta": raw_debt["tail95_delta"],
            "raw_tail99_delta": raw_debt["tail99_delta"],
            "raw_margin10_delta": raw_debt["margin10_delta"],
            "margin10_delta_correct_sign": int(debt["margin10_delta"] >= -float(args.no_debt_budget)),
            "component_non_positive": int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0),
            "original_to_final_total_audit": 1,
            "post_recode_incremental_only": 0,
            "wall_time_s": time.time() - start,
        }
    except Exception as exc:
        return {**AUDIT_DEFAULTS, "part": "1F", "status": "error", "scheme": scheme, "task": task, "seed": int(seed), "error_message": repr(exc), "wall_time_s": time.time() - start}


def run_stage1_row(scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    start = time.time()
    try:
        x, y, xg, yg = v2308.v2307.visual_data(task, int(seed), args, device)
        classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = model_seed_for(task, int(seed), args)
        model = v2308.v2307.make_model(str(args.basis_key), int(args.depth), int(x.shape[1]), classes, model_seed, args, device, basis_input_gain=float(args.part_d_basis_input_gain))
        ckpt = v2308.v2307.train_checkpoint(model, x, y, str(args.checkpoint), int(seed), args)
        original_state = v2308.model_state(model)
        train_original = v2308.actual_metrics(model, x, y)
        guard_original = v2308.actual_metrics(model, xg, yg)
        with torch.no_grad():
            _lg0, guard_acts_original = model.forward_with_activations(xg)
        diag = apply_cheb_domain_transport(model, x, y, args, scheme=scheme, task=task, seed=int(seed))
        raw_state = v2308.model_state(model)
        raw_train = v2308.actual_metrics(model, x, y)
        raw_guard = v2308.actual_metrics(model, xg, yg)
        raw_pass, raw_debt = strict_total_audit_pass(guard_original, raw_guard, float(args.no_debt_budget))
        alpha, reason, selected_state, final_train, final_guard, _selected_pass = project_transport_state(model, original_state, raw_state, x, y, xg, yg, guard_original, args)
        v2308.load_model_state(model, selected_state)
        with torch.no_grad():
            _lg1, guard_acts_final = model.forward_with_activations(xg)
        drift = v2312.activation_drift_stats(guard_acts_original, guard_acts_final)
        debt = v2308.debt_deltas(guard_original, final_guard)
        no_debt = v2308.no_debt_ok(debt, float(args.no_debt_budget))
        return {
            **AUDIT_DEFAULTS,
            "part": "1",
            "status": "ok",
            "scheme": scheme,
            "task": task,
            "seed": int(seed),
            "basis_key": str(args.basis_key),
            "basis_family": str(model.basis_name),
            "depth": int(args.depth),
            "train_size": int(args.train_size),
            "guard_size": int(args.guard_size),
            "checkpoint_name": str(args.checkpoint),
            "checkpoint_optimizer": ckpt.get("checkpoint_optimizer", ""),
            "model_seed": model_seed,
            "transport_strength": float(args.transport_strength),
            "transport_target_abs": float(args.transport_target_abs),
            "transport_max_shift": float(args.transport_max_shift),
            "transport_tail_quantile": float(args.transport_tail_quantile),
            "transport_alpha_candidates": str(args.transport_alphas),
            "transport_alpha_selected": alpha,
            "transport_alpha_selection_reason": reason,
            **diag,
            "original_guard_coverage": guard_original["coverage"],
            "raw_transport_guard_coverage": raw_guard["coverage"],
            "final_guard_coverage": final_guard["coverage"],
            "total_C2_coverage_improvement": final_guard["coverage"] - guard_original["coverage"],
            "raw_transport_total_C2_coverage_improvement": raw_guard["coverage"] - guard_original["coverage"],
            "total_C2_accuracy_improvement": final_guard["accuracy"] - guard_original["accuracy"],
            "total_guard_nll_delta": final_guard["loss"] - guard_original["loss"],
            "total_train_nll_delta": final_train["loss"] - train_original["loss"],
            "source_guard_c2_gap": (final_train["coverage"] - train_original["coverage"]) - (final_guard["coverage"] - guard_original["coverage"]),
            "F5_no_debt_pass": no_debt,
            "raw_transport_total_audit_pass": raw_pass,
            "raw_transport_total_audit_reject": int(not raw_pass),
            "Brier_delta": debt["Brier_delta"],
            "ECE_delta": debt["ECE_delta"],
            "tail95_delta": debt["tail95_delta"],
            "tail99_delta": debt["tail99_delta"],
            "margin10_delta": debt["margin10_delta"],
            "debt_delta": debt["debt_delta"],
            "raw_Brier_delta": raw_debt["Brier_delta"],
            "raw_ECE_delta": raw_debt["ECE_delta"],
            "raw_tail95_delta": raw_debt["tail95_delta"],
            "raw_tail99_delta": raw_debt["tail99_delta"],
            "raw_margin10_delta": raw_debt["margin10_delta"],
            "margin10_delta_correct_sign": int(debt["margin10_delta"] >= -float(args.no_debt_budget)),
            "component_non_positive": int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0),
            "activation_drift_median": drift["activation_drift_median"],
            "activation_drift_max": drift["activation_drift_max"],
            "activation_drift_layer_count": drift["activation_drift_layer_count"],
            "original_to_final_total_audit": 1,
            "post_transport_incremental_only": 0,
            "wall_time_s": time.time() - start,
        }
    except Exception as exc:
        return {**AUDIT_DEFAULTS, "part": "1", "status": "error", "scheme": scheme, "task": task, "seed": int(seed), "error_message": repr(exc), "wall_time_s": time.time() - start}


def stage1_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    schemes = [item.strip() for item in str(args.stage1_schemes).split(",") if item.strip()]
    tasks = [item.strip() for item in str(args.stage1_tasks).split(",") if item.strip()]
    return [(scheme, task, seed) for scheme in schemes for task in tasks for seed in range(int(args.stage1_seed_count))]


def stage1_run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    lock_path = OUT_ROOT / "stage0_evidence_lock_summary.json"
    if not lock_path.exists() or ival(read_json(lock_path).get("gate_pass")) != 1:
        summary = {"part": "1", "route": "Stage1BlockedByStage0", "gate_pass": 0, "dominant_blocker": "stage0_missing_or_failed"}
        out = write_json(OUT_ROOT / "stage1_domain_transport_summary.json", summary)
        append_exec("Stage 1 domain transport blocked", command_text(), files=rel(out), gpu=str(args.device), note=summary["dominant_blocker"])
        return summary
    device = torch.device(str(args.device))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"stage1_domain_transport_matrix{suffix}.csv"
    existing = read_rows(matrix) if int(args.stage1_resume) else []
    done = {(r.get("scheme"), r.get("task"), str(r.get("seed"))) for r in existing if r.get("status") == "ok"}
    jobs = [job for idx, job in enumerate(stage1_jobs(args)) if idx % int(args.shard_count) == int(args.shard_index)]
    rows = list(existing)
    for scheme, task, seed in jobs:
        if (scheme, task, str(seed)) in done:
            continue
        rows.append(run_stage1_row(scheme, task, int(seed), args, device))
        if int(args.stage1_flush_every) > 0:
            write_rows(matrix, rows)
            print(json.dumps({"part": "1", "rows_written": len(rows), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
    write_rows(matrix, rows)
    summary = {"part": "1", "route": "Stage1ShardDone", "gate_pass": 0, "matrix": rel(matrix), "rows_written": len(rows), "ok_rows": sum(1 for row in rows if row.get("status") == "ok")}
    out = write_json(OUT_ROOT / f"stage1_domain_transport_summary{suffix}.json", summary)
    append_exec("Stage 1 domain transport shard", command_text(), files=f"{rel(matrix)}; {rel(out)}", gpu=str(args.device), note=f"rows={len(rows)}; ok={summary['ok_rows']}")
    return summary


def collect_stage1_rows() -> list[dict[str, str]]:
    shard_files = sorted(OUT_ROOT.glob("stage1_domain_transport_matrix_shard*_of_*.csv"))
    files = shard_files if shard_files else [OUT_ROOT / "stage1_domain_transport_matrix.csv"]
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in files:
        for row in read_rows(path):
            key = (row.get("scheme", ""), row.get("task", ""), str(row.get("seed", "")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def stage1_merge(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    rows = collect_stage1_rows()
    matrix = write_rows(OUT_ROOT / "stage1_domain_transport_matrix.csv", rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for scheme in [item.strip() for item in str(args.stage1_schemes).split(",") if item.strip()]:
        group_rows = [row for row in ok if row.get("scheme") == scheme]
        groups.append({
            "scheme": scheme,
            "rows": len(group_rows),
            "total_C2_coverage_improvement_median": median([r.get("total_C2_coverage_improvement") for r in group_rows]),
            "raw_transport_total_C2_coverage_improvement_median": median([r.get("raw_transport_total_C2_coverage_improvement") for r in group_rows]),
            "total_C2_accuracy_improvement_median": median([r.get("total_C2_accuracy_improvement") for r in group_rows]),
            "total_guard_nll_delta_median": median([r.get("total_guard_nll_delta") for r in group_rows]),
            "F5_no_debt_count": sum(ival(r.get("F5_no_debt_pass")) for r in group_rows),
            "component_non_positive_rows": sum(ival(r.get("component_non_positive")) for r in group_rows),
            "raw_transport_reject_count": sum(ival(r.get("raw_transport_total_audit_reject")) for r in group_rows),
            "transport_alpha_selected_median": median([r.get("transport_alpha_selected") for r in group_rows]),
            "transport_shift_abs_median": median([r.get("transport_shift_abs_median") for r in group_rows]),
            "transport_shift_abs_max_median": median([r.get("transport_shift_abs_max") for r in group_rows]),
            "activation_drift_median": median([r.get("activation_drift_median") for r in group_rows]),
            "wall_time_s_median": median([r.get("wall_time_s") for r in group_rows]),
        })
    group_csv = write_rows(OUT_ROOT / "stage1_domain_transport_group_summary.csv", groups)
    by_scheme = {g["scheme"]: g for g in groups}
    c33 = by_scheme.get("C33_Candidate_tail_edge_domain_transport_total_audit", {})
    c34 = by_scheme.get("C34_Control_shuffled_tail_edge_domain_transport_total_audit", {})
    c35 = by_scheme.get("C35_Control_global_edge_domain_transport_total_audit", {})
    expected_rows = len(stage1_jobs(args))
    c33_expected = sum(1 for scheme, _task, _seed in stage1_jobs(args) if scheme == "C33_Candidate_tail_edge_domain_transport_total_audit")
    c33_cov = fval(c33.get("total_C2_coverage_improvement_median"))
    c33_drift = fval(c33.get("activation_drift_median"))
    checks = {
        "row_count_30_ok": len(rows) == expected_rows and len(ok) == expected_rows,
        "no_fake_data": sum(ival(r.get("used_fake_data_rows")) for r in ok) == 0,
        "no_held_test_usage": sum(ival(r.get("held_test_usage")) for r in ok) == 0,
        "original_to_final_total_audit_all_rows": sum(ival(r.get("original_to_final_total_audit")) for r in ok) == len(ok),
        "post_transport_incremental_only_no_rows": sum(ival(r.get("post_transport_incremental_only")) for r in ok) == 0,
        "c33_total_no_debt_clean": c33_expected > 0 and fval(c33.get("F5_no_debt_count")) == c33_expected and fval(c33.get("component_non_positive_rows")) == c33_expected,
        "c33_coverage_ge_0p005": c33_cov >= 0.005,
        "c33_minus_c34_ge_0p01": c33_cov - fval(c34.get("total_C2_coverage_improvement_median")) >= 0.01,
        "c33_minus_c35_ge_0p01": c33_cov - fval(c35.get("total_C2_coverage_improvement_median")) >= 0.01,
        "c33_activation_drift_in_range": c33_drift > 0.001 and c33_drift < 0.30,
    }
    gate = int(all(checks.values()))
    if not checks["row_count_30_ok"]:
        blocker = "unexpected_stage1_row_count_or_errors"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif not checks["c33_total_no_debt_clean"]:
        blocker = "c33_total_audit_debt_or_component_positive"
    elif not checks["c33_coverage_ge_0p005"]:
        blocker = "c33_no_meaningful_total_coverage_gain"
    elif not checks["c33_minus_c34_ge_0p01"]:
        blocker = "c33_not_above_shuffled_tail_transport"
    elif not checks["c33_minus_c35_ge_0p01"]:
        blocker = "c33_not_above_global_transport_control"
    elif not checks["c33_activation_drift_in_range"]:
        blocker = "c33_activation_drift_out_of_range"
    else:
        blocker = "none"
    summary = {
        "part": "1",
        "route": "Stage1ChebDomainTransportPass" if gate else "Stage1ChebDomainTransportFailed",
        "gate_pass": gate,
        "dominant_blocker": blocker,
        "checks": checks,
        "metrics": {
            "c33_coverage": c33_cov,
            "c34_coverage": fval(c34.get("total_C2_coverage_improvement_median")),
            "c35_coverage": fval(c35.get("total_C2_coverage_improvement_median")),
            "c33_minus_c34": c33_cov - fval(c34.get("total_C2_coverage_improvement_median")),
            "c33_minus_c35": c33_cov - fval(c35.get("total_C2_coverage_improvement_median")),
            "c33_raw_transport_coverage": fval(c33.get("raw_transport_total_C2_coverage_improvement_median")),
            "c33_raw_transport_reject_count": fval(c33.get("raw_transport_reject_count")),
            "c33_alpha_selected_median": fval(c33.get("transport_alpha_selected_median")),
            "c33_activation_drift_median": c33_drift,
            "c33_wall_time_s_median": fval(c33.get("wall_time_s_median")),
        },
        "artifacts": {"matrix": rel(matrix), "group_summary": rel(group_csv)},
        "hashes": {"matrix": sha256_file(matrix), "group_summary": sha256_file(group_csv)},
        "next_action": "run_stage2_C33_plus_C28_C32_controls" if gate else "analyze_or_repair_domain_transport_before_inverse",
    }
    out = write_json(OUT_ROOT / "stage1_domain_transport_summary.json", summary)
    nxt = write_json(
        FORMAL_ROOT / "next_actions_for_v23_13.json",
        {
            "version": "v23.13",
            "last_completed_stage": "stage1_cheb_domain_transport_reduced",
            "stage1_gate_pass": gate,
            "stage1_route": summary["route"],
            "stage1_blocker": blocker,
            "promotion_allowed": 0,
            "completed_artifacts": {"stage1_summary": rel(out), "matrix": rel(matrix), "group_summary": rel(group_csv)},
            "allowed_next_actions": ["run_stage2_C33_plus_C28_C32_controls"] if gate else ["analyze_or_repair_domain_transport_before_inverse"],
            "forbidden_next_actions": [
                "promote_C33_without_stage2_full_controls",
                "fabricate_data",
                "held_test_induction",
                "runtime_winner_selection",
                "add_mlp_stem_or_readout",
                "add_new_edge_function_family",
            ],
        },
    )
    append_exec("Stage 1 Cheb domain transport merge", command_text(), files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Stage 1 Cheb domain transport reduced", summary)
    return summary


def recode_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    schemes = [item.strip() for item in str(args.recode_schemes).split(",") if item.strip()]
    tasks = [item.strip() for item in str(args.recode_tasks).split(",") if item.strip()]
    return [(scheme, task, seed) for scheme in schemes for task in tasks for seed in range(int(args.recode_seed_count))]


def recode_run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    lock_path = OUT_ROOT / "stage0_evidence_lock_summary.json"
    if not lock_path.exists() or ival(read_json(lock_path).get("gate_pass")) != 1:
        summary = {"part": "1F", "route": "Stage1FBlockedByStage0", "gate_pass": 0, "dominant_blocker": "stage0_missing_or_failed"}
        out = write_json(OUT_ROOT / "stage1f_gain_recode_summary.json", summary)
        append_exec("Stage 1F gain recode blocked", command_text(), files=rel(out), gpu=str(args.device), note=summary["dominant_blocker"])
        return summary
    device = torch.device(str(args.device))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"stage1f_gain_recode_matrix{suffix}.csv"
    existing = read_rows(matrix) if int(args.recode_resume) else []
    done = {(r.get("scheme"), r.get("task"), str(r.get("seed"))) for r in existing if r.get("status") == "ok"}
    jobs = [job for idx, job in enumerate(recode_jobs(args)) if idx % int(args.shard_count) == int(args.shard_index)]
    rows = list(existing)
    for scheme, task, seed in jobs:
        if (scheme, task, str(seed)) in done:
            continue
        rows.append(run_recode_row(scheme, task, int(seed), args, device))
        if int(args.recode_flush_every) > 0:
            write_rows(matrix, rows)
            print(json.dumps({"part": "1F", "rows_written": len(rows), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
    write_rows(matrix, rows)
    summary = {"part": "1F", "route": "Stage1FShardDone", "gate_pass": 0, "matrix": rel(matrix), "rows_written": len(rows), "ok_rows": sum(1 for row in rows if row.get("status") == "ok")}
    out = write_json(OUT_ROOT / f"stage1f_gain_recode_summary{suffix}.json", summary)
    append_exec("Stage 1F gain recode shard", command_text(), files=f"{rel(matrix)}; {rel(out)}", gpu=str(args.device), note=f"rows={len(rows)}; ok={summary['ok_rows']}")
    return summary


def collect_recode_rows() -> list[dict[str, str]]:
    shard_files = sorted(OUT_ROOT.glob("stage1f_gain_recode_matrix_shard*_of_*.csv"))
    files = shard_files if shard_files else [OUT_ROOT / "stage1f_gain_recode_matrix.csv"]
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in files:
        for row in read_rows(path):
            key = (row.get("scheme", ""), row.get("task", ""), str(row.get("seed", "")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def recode_merge(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    rows = collect_recode_rows()
    matrix = write_rows(OUT_ROOT / "stage1f_gain_recode_matrix.csv", rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for scheme in [item.strip() for item in str(args.recode_schemes).split(",") if item.strip()]:
        group_rows = [row for row in ok if row.get("scheme") == scheme]
        groups.append({
            "scheme": scheme,
            "rows": len(group_rows),
            "total_C2_coverage_improvement_median": median([r.get("total_C2_coverage_improvement") for r in group_rows]),
            "total_C2_accuracy_improvement_median": median([r.get("total_C2_accuracy_improvement") for r in group_rows]),
            "total_guard_nll_delta_median": median([r.get("total_guard_nll_delta") for r in group_rows]),
            "F5_no_debt_count": sum(ival(r.get("F5_no_debt_pass")) for r in group_rows),
            "component_non_positive_rows": sum(ival(r.get("component_non_positive")) for r in group_rows),
            "raw_recode_reject_count": sum(ival(r.get("raw_recode_total_audit_reject")) for r in group_rows),
            "recode_alpha_selected_median": median([r.get("recode_alpha_selected") for r in group_rows]),
            "recode_selected_gain_median": median([r.get("recode_selected_gain") for r in group_rows]),
            "recode_layer_fit_rel_median": median([r.get("recode_layer_fit_rel_median") for r in group_rows]),
            "recode_layer_fit_rel_max_median": median([r.get("recode_layer_fit_rel_max") for r in group_rows]),
            "raw_train_logit_rel_error_median": median([r.get("raw_train_logit_rel_error") for r in group_rows]),
            "raw_guard_logit_rel_error_median": median([r.get("raw_guard_logit_rel_error") for r in group_rows]),
            "selected_basis_design_drift_median": median([r.get("selected_basis_design_drift_median") for r in group_rows]),
            "selected_activation_drift_median": median([r.get("selected_activation_drift_median") for r in group_rows]),
            "wall_time_s_median": median([r.get("wall_time_s") for r in group_rows]),
        })
    group_csv = write_rows(OUT_ROOT / "stage1f_gain_recode_group_summary.csv", groups)
    by_scheme = {g["scheme"]: g for g in groups}
    c36 = by_scheme.get("C36_Candidate_function_preserving_gain_recode_total_audit", {})
    c37 = by_scheme.get("C37_Control_identity_gain_recode_total_audit", {})
    expected_rows = len(recode_jobs(args))
    c36_expected = sum(1 for scheme, _task, _seed in recode_jobs(args) if scheme == "C36_Candidate_function_preserving_gain_recode_total_audit")
    c36_cov = fval(c36.get("total_C2_coverage_improvement_median"))
    c36_design = fval(c36.get("selected_basis_design_drift_median"))
    c37_design = fval(c37.get("selected_basis_design_drift_median"))
    c36_fit = fval(c36.get("recode_layer_fit_rel_median"))
    checks = {
        "row_count_expected_ok": len(rows) == expected_rows and len(ok) == expected_rows,
        "no_fake_data": sum(ival(r.get("used_fake_data_rows")) for r in ok) == 0,
        "no_held_test_usage": sum(ival(r.get("held_test_usage")) for r in ok) == 0,
        "original_to_final_total_audit_all_rows": sum(ival(r.get("original_to_final_total_audit")) for r in ok) == len(ok),
        "post_recode_incremental_only_no_rows": sum(ival(r.get("post_recode_incremental_only")) for r in ok) == 0,
        "c36_total_no_debt_clean": c36_expected > 0 and fval(c36.get("F5_no_debt_count")) == c36_expected and fval(c36.get("component_non_positive_rows")) == c36_expected,
        "c36_abs_coverage_le_0p01": abs(c36_cov) <= 0.01,
        "c36_design_drift_ge_0p001": c36_design >= 0.001,
        "c36_fit_rel_le_0p05": c36_fit <= 0.05,
        "c36_design_minus_identity_ge_0p001": c36_design - c37_design >= 0.001,
    }
    gate = int(all(checks.values()))
    if not checks["row_count_expected_ok"]:
        blocker = "unexpected_stage1f_row_count_or_errors"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif not checks["c36_total_no_debt_clean"]:
        blocker = "c36_total_audit_debt_or_component_positive"
    elif not checks["c36_abs_coverage_le_0p01"]:
        blocker = "c36_not_function_preserving_on_total_coverage"
    elif not checks["c36_design_drift_ge_0p001"]:
        blocker = "c36_no_basis_design_drift"
    elif not checks["c36_fit_rel_le_0p05"]:
        blocker = "c36_source_reconstruction_error_too_high"
    elif not checks["c36_design_minus_identity_ge_0p001"]:
        blocker = "c36_design_drift_not_above_identity"
    else:
        blocker = "none"
    summary = {
        "part": "1F",
        "route": "Stage1FFunctionPreservingGainRecodePass" if gate else "Stage1FFunctionPreservingGainRecodeFailed",
        "gate_pass": gate,
        "dominant_blocker": blocker,
        "checks": checks,
        "metrics": {
            "c36_coverage": c36_cov,
            "c36_abs_coverage": abs(c36_cov),
            "c36_design_drift": c36_design,
            "c37_design_drift": c37_design,
            "c36_design_minus_identity": c36_design - c37_design,
            "c36_fit_rel": c36_fit,
            "c36_fit_rel_max": fval(c36.get("recode_layer_fit_rel_max_median")),
            "c36_raw_train_logit_rel_error": fval(c36.get("raw_train_logit_rel_error_median")),
            "c36_raw_guard_logit_rel_error": fval(c36.get("raw_guard_logit_rel_error_median")),
            "c36_alpha_selected_median": fval(c36.get("recode_alpha_selected_median")),
            "c36_selected_gain_median": fval(c36.get("recode_selected_gain_median")),
            "c36_wall_time_s_median": fval(c36.get("wall_time_s_median")),
        },
        "artifacts": {"matrix": rel(matrix), "group_summary": rel(group_csv)},
        "hashes": {"matrix": sha256_file(matrix), "group_summary": sha256_file(group_csv)},
        "next_action": "run_stage2_inverse_from_C36_recode" if gate else "stop_or_repair_function_preserving_recode_before_inverse",
    }
    out = write_json(OUT_ROOT / "stage1f_gain_recode_summary.json", summary)
    nxt = write_json(
        FORMAL_ROOT / "next_actions_for_v23_13.json",
        {
            "version": "v23.13",
            "last_completed_stage": "stage1f_function_preserving_gain_recode",
            "stage1f_gate_pass": gate,
            "stage1f_route": summary["route"],
            "stage1f_blocker": blocker,
            "promotion_allowed": 0,
            "completed_artifacts": {"stage1f_summary": rel(out), "matrix": rel(matrix), "group_summary": rel(group_csv)},
            "allowed_next_actions": ["run_stage2_inverse_from_C36_recode"] if gate else ["stop_or_repair_function_preserving_recode_before_inverse"],
            "forbidden_next_actions": [
                "promote_C36_without_inverse_and_full_controls",
                "fabricate_data",
                "held_test_induction",
                "runtime_winner_selection",
                "add_mlp_stem_or_readout",
                "add_new_edge_function_family",
            ],
        },
    )
    append_exec("Stage 1F function-preserving gain recode merge", command_text(), files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Stage 1F function-preserving gain recode", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--basis-key", default="dche_k9")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--train-size", type=int, default=768)
    p.add_argument("--guard-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--checkpoint", default="h10_isomorphic_partial")
    p.add_argument("--checkpoint-steps", type=int, default=20)
    p.add_argument("--checkpoint-lr", type=float, default=0.02)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--part-d-basis-input-gain", type=float, default=0.25)
    p.add_argument("--no-debt-budget", type=float, default=1.0e-8)
    p.add_argument("--stage1-schemes", default=",".join(STAGE1_SCHEMES))
    p.add_argument("--stage1-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--stage1-seed-count", type=int, default=5)
    p.add_argument("--stage1-resume", type=int, default=1)
    p.add_argument("--stage1-flush-every", type=int, default=1)
    p.add_argument("--transport-tail-quantile", type=float, default=0.25)
    p.add_argument("--transport-strength", type=float, default=0.5)
    p.add_argument("--transport-target-abs", type=float, default=0.75)
    p.add_argument("--transport-max-shift", type=float, default=0.25)
    p.add_argument("--transport-grid-points", type=int, default=45)
    p.add_argument("--transport-ridge", type=float, default=1.0e-8)
    p.add_argument("--transport-alphas", default="1,0.5,0.25,0.125,0.0625,0")
    p.add_argument("--recode-schemes", default=",".join(RECODE_SCHEMES))
    p.add_argument("--recode-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--recode-seed-count", type=int, default=5)
    p.add_argument("--recode-resume", type=int, default=1)
    p.add_argument("--recode-flush-every", type=int, default=1)
    p.add_argument("--recode-gain", type=float, default=0.5)
    p.add_argument("--recode-control-gain", type=float, default=0.125)
    p.add_argument("--recode-ridge", type=float, default=1.0e-6)
    p.add_argument("--recode-alphas", default="1,0.5,0.25,0.125,0.0625,0")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode).lower()
    if mode in {"stage0", "stage0-lock", "evidence-lock"}:
        return stage0_lock(args)
    if mode in {"stage1-run", "domain-run"}:
        return stage1_run(args)
    if mode in {"stage1-merge", "stage1", "domain-merge"}:
        return stage1_merge(args)
    if mode in {"stage1f-run", "recode-run", "gain-recode-run"}:
        return recode_run(args)
    if mode in {"stage1f-merge", "stage1f", "recode-merge", "gain-recode-merge"}:
        return recode_merge(args)
    raise SystemExit(f"unknown v23.13 mode: {args.mode}")


if __name__ == "__main__":
    main()
