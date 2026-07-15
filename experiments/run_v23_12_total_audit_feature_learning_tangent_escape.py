#!/usr/bin/env python3
"""DG-KAN v23.12 total-audit feature-learning / tangent-escape runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_08_downstream_coupled_edge_residual_inverse_flow as v2308
import experiments.run_v23_09_trust_projected_downstream_efrf as v2309
from dgkan.fu import downstream_coupled_residual_inverse as dcerif


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.12_TotalAuditFeatureLearningTangentEscape_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.12_TotalAuditFeatureLearningTangentEscape_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.12_TotalAuditFeatureLearningTangentEscape_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2312_OUT_ROOT", str(ROOT / "results/v23_12_total_audit_feature_learning_tangent_escape"))).resolve()
FORMAL_ROOT = ROOT / "results/v23_09_part_b_repair_tol001_formal_15seed80"

V2311_LOCK = ROOT / "results/v23_11_feature_learning_tangent_escape_kan/part_0_evidence_lock_summary.json"
V2311_STAGE1 = ROOT / "results/v23_11_feature_learning_tangent_escape_kan/basis_gain_tangent_relocation_summary.json"
V2311_STAGE2A = ROOT / "results/v23_11_feature_learning_tangent_escape_kan/spectral_mode_reduced_summary.json"
V2311_STAGE2B = ROOT / "results/v23_11_feature_learning_tangent_escape_kan/tail_target_reduced_summary.json"
ROUTE_A = ROOT / "results/v23_10_efficiency_equivalence_task_curvature_efrf/route_a_efficiency_equivalence_summary.json"
DEFAULT_REDUCED_REF = ROOT / "results/v23_10_route_b_guard_penalty_reduced_s5_80/route_b_b3_guard_penalty_reduced_summary.json"
DEFAULT_REDUCED_GROUP = ROOT / "results/v23_10_route_b_guard_penalty_reduced_s5_80/part_c_component_attribution_group_summary.csv"

STAGE2_SCHEMES = [
    "C28_Candidate_tail_cvar_anchor_then_terminal_C15_total_audit",
    "C29_Control_tail_cvar_anchor_only_total_audit",
    "C30_Control_tail_cvar_anchor_then_matched_LocalEFRF_total_audit",
    "C31_Control_shuffled_tail_anchor_then_terminal_C15_total_audit",
    "C32_Control_tail_cvar_anchor_then_no_inverse_total_audit",
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
        EXEC_LOG.write_text(
            "# DG-KAN v23.12 Total-Audit Feature-Learning Tangent-Escape 执行日志\n\n"
            "- 原则：不造假；不把 v23.09/v23.10/v23.11 failure 改写成 success；不使用 held/test induction。\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.12 Total-Audit Feature-Learning Tangent-Escape 实验结果复盘\n\n"
            "- 原则：只记录真实 artifact 数据；不得把 diagnostic 写成 promotion。\n",
            encoding="utf-8",
        )


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


def mean(values: list[Any]) -> float:
    nums = [fval(v) for v in values if v not in (None, "")]
    return float(sum(nums) / max(1, len(nums)))


def float_items(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def stage0_evidence_lock(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    required = {
        "plan": PLAN,
        "runner": RUNNER,
        "v23_11_lock": V2311_LOCK,
        "v23_11_stage1": V2311_STAGE1,
        "v23_11_stage2a": V2311_STAGE2A,
        "v23_11_stage2b": V2311_STAGE2B,
        "route_a_efficiency_equivalence": ROUTE_A,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    route_a = read_json(ROUTE_A) if ROUTE_A.exists() else {}
    lock = read_json(V2311_LOCK) if V2311_LOCK.exists() else {}
    s1 = read_json(V2311_STAGE1) if V2311_STAGE1.exists() else {}
    s2a = read_json(V2311_STAGE2A) if V2311_STAGE2A.exists() else {}
    s2b = read_json(V2311_STAGE2B) if V2311_STAGE2B.exists() else {}
    checks = {
        "no_missing_required_artifacts": not missing,
        "v23_09_official_coverage_dominance_still_failed": ival(route_a.get("official_v23_09_part_c_gate_pass"), -1) == 0,
        "route_a_efficiency_equivalence_pass": ival(route_a.get("gate_pass"), -1) == 1 and "efficiency" in str(route_a.get("promotion_scope", "")).lower(),
        "v23_11_lock_pass": ival(lock.get("gate_pass"), -1) == 1,
        "v23_11_stage1_failed": ival(s1.get("gate_pass"), -1) == 0,
        "v23_11_stage2a_failed": ival(s2a.get("gate_pass"), -1) == 0,
        "v23_11_stage2b_failed": ival(s2b.get("gate_pass"), -1) == 0,
        "stage1_no_fake_or_held": bool(s1.get("checks", {}).get("no_fake_data")) and bool(s1.get("checks", {}).get("no_held_test_usage")),
        "stage2a_no_fake_or_held": bool(s2a.get("checks", {}).get("no_fake_data")) and bool(s2a.get("checks", {}).get("no_held_test_usage")),
        "stage2b_no_fake_or_held": bool(s2b.get("checks", {}).get("no_fake_data")) and bool(s2b.get("checks", {}).get("no_held_test_usage")),
    }
    gate = int(all(checks.values()))
    summary = {
        "part": "0",
        "gate_pass": gate,
        "route": "V2312EvidenceLockPass" if gate else "V2312EvidenceLockFailed",
        "dominant_blocker": "none" if gate else "missing_or_inconsistent_prior_evidence",
        "checks": checks,
        "missing": missing,
        "artifacts": {name: rel(path) for name, path in required.items()},
        "hashes": {name: sha256_file(path) for name, path in required.items() if path.exists()},
        "locked_metrics": {
            "route_a_c15_coverage": route_a.get("metrics", {}).get("c15_coverage"),
            "route_a_c15_minus_c3": route_a.get("metrics", {}).get("c15_minus_c3"),
            "stage1_blocker": s1.get("dominant_blocker"),
            "stage2a_blocker": s2a.get("dominant_blocker"),
            "stage2b_blocker": s2b.get("dominant_blocker"),
            "stage2b_tail_target_c15_minus_matched_c3": s2b.get("metrics", {}).get("tail_target_c15_minus_matched_c3"),
        },
        "next_action": "run_stage1_tail_cvar_feature_anchor_feasibility" if gate else "repair_evidence_lock_before_experiment",
        "scope": "evidence_lock_only_no_new_experimental_data",
    }
    out = write_json(OUT_ROOT / "stage0_evidence_lock_summary.json", summary)
    nxt = write_json(
        FORMAL_ROOT / "next_actions_for_v23_12.json",
        {
            "version": "v23.12",
            "last_completed_stage": "stage0_evidence_lock",
            "stage0_gate_pass": gate,
            "promotion_allowed": 0,
            "completed_artifacts": {"stage0_summary": rel(out)},
            "allowed_next_actions": ["run_stage1_tail_cvar_feature_anchor_feasibility"] if gate else ["repair_evidence_lock_before_experiment"],
            "forbidden_next_actions": [
                "promote_v23_09_coverage_dominance_without_new_passing_data",
                "measure_only_post_anchor_incremental_delta_for_feature_anchor_candidate",
                "fabricate_data",
                "held_test_induction",
                "runtime_winner_selection",
            ],
        },
    )
    append_exec("Stage 0 evidence lock", command_text(), files=f"{rel(out)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={summary['dominant_blocker']}")
    append_recap("Stage 0 evidence lock", summary)
    return summary


def stage1_jobs(args: argparse.Namespace) -> list[tuple[str, int]]:
    tasks = [item.strip() for item in str(args.stage1_tasks).split(",") if item.strip()]
    return [(task, seed) for task in tasks for seed in range(int(args.stage1_seed_count))]


def model_seed_for_part_c(task: str, seed: int, args: argparse.Namespace) -> int:
    key = str(args.part_c_model_seed_scheme_key)
    return 23083000 + int(seed) * 1009 + sum(ord(c) for c in key + task)


def activation_drift_stats(before: list[torch.Tensor], after: list[torch.Tensor]) -> dict[str, float]:
    vals: list[float] = []
    for b, a in zip(before[1:], after[1:]):
        bb = b.detach().to(dtype=torch.float64)
        aa = a.detach().to(device=bb.device, dtype=torch.float64)
        vals.append(float((aa - bb).norm().div(bb.norm().clamp_min(1.0e-12)).detach().cpu().item()))
    return {
        "activation_drift_median": median(vals),
        "activation_drift_max": max(vals or [0.0]),
        "activation_drift_layer_count": len(vals),
    }


def downstream_jacobian_drift_proxy(model: Any, xg: torch.Tensor, before_state: dict[str, torch.Tensor], after_state: dict[str, torch.Tensor], args: argparse.Namespace) -> float:
    n = min(int(args.stage1_jacobian_sketch_size), int(xg.shape[0]))
    if n <= 0 or len(model.coeffs) < 2:
        return 0.0
    xs = xg[:n]
    pen = max(0, len(model.coeffs) - 2)
    v2308.load_model_state(model, before_state)
    _lb, acts_b = model.forward_with_activations(xs)
    jb = dcerif.downstream_jacobian(model, acts_b[pen + 1], pen + 1)
    v2308.load_model_state(model, after_state)
    _la, acts_a = model.forward_with_activations(xs)
    ja = dcerif.downstream_jacobian(model, acts_a[pen + 1], pen + 1)
    return float((ja - jb).norm().div(jb.norm().clamp_min(1.0e-12)).detach().cpu().item())


def interpolated_state(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, base in before.items():
        target = after[name].to(device=base.device, dtype=base.dtype)
        out[name] = base + float(alpha) * (target - base)
    return out


def strict_total_audit_pass(before_guard: dict[str, float], after_guard: dict[str, float], budget: float) -> tuple[int, dict[str, float]]:
    debt = v2308.debt_deltas(before_guard, after_guard)
    component_ok = int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0)
    return int(v2308.no_debt_ok(debt, float(budget)) and component_ok), debt


def tail_cvar_anchor_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    step: int,
    seed: int = 0,
    task: str = "",
    shuffle_tail: bool = False,
) -> dict[str, Any]:
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    prob = torch.softmax(logits.float(), dim=1)
    true_prob = prob.gather(1, y.long().reshape(-1, 1)).reshape(-1)
    q = min(0.95, max(0.01, float(args.stage1_tail_quantile)))
    threshold = torch.quantile(true_prob.detach().to(dtype=torch.float64), q)
    mask = true_prob.detach().to(dtype=torch.float64) <= threshold
    if int(mask.sum().detach().cpu().item()) <= 0:
        mask = torch.zeros_like(true_prob, dtype=torch.bool)
        mask[int(torch.argmin(true_prob.detach()).detach().cpu().item())] = True
    if shuffle_tail:
        gen = torch.Generator(device=x.device).manual_seed(23120000 + int(seed) * 1009 + int(step) * 917 + sum(ord(c) for c in str(task)))
        mask = mask[torch.randperm(int(mask.shape[0]), device=x.device, generator=gen)]
    loss = F.cross_entropy(logits[mask].float(), y[mask].long())
    loss.backward()
    opt.step()
    return {
        "step": int(step),
        "tail_threshold": float(threshold.detach().cpu().item()),
        "tail_count": int(mask.sum().detach().cpu().item()),
        "tail_loss": float(loss.detach().cpu().item()),
        "tail_shuffled": int(shuffle_tail),
    }


def prepare_tail_cvar_anchor(task: str, seed: int, args: argparse.Namespace, device: torch.device, *, shuffle_tail: bool = False) -> dict[str, Any]:
    x, y, xg, yg = v2308.v2307.visual_data(task, int(seed), args, device)
    classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
    model_seed = model_seed_for_part_c(task, int(seed), args)
    model = v2308.v2307.make_model(
        str(args.basis_key),
        int(args.depth),
        int(x.shape[1]),
        classes,
        model_seed,
        args,
        device,
        basis_input_gain=float(args.part_d_basis_input_gain),
    )
    ckpt = v2308.v2307.train_checkpoint(model, x, y, str(args.stage1_checkpoint), int(seed), args)
    original_state = v2308.model_state(model)
    train_original = v2308.actual_metrics(model, x, y)
    guard_original = v2308.actual_metrics(model, xg, yg)
    with torch.no_grad():
        _lg0, guard_acts_original = model.forward_with_activations(xg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.stage1_anchor_lr), weight_decay=float(args.weight_decay))
    tail_counts: list[int] = []
    tail_losses: list[float] = []
    tail_thresholds: list[float] = []
    for step in range(1, int(args.stage1_anchor_steps) + 1):
        diag = tail_cvar_anchor_step(model, x, y, opt, args, step=step, seed=int(seed), task=task, shuffle_tail=shuffle_tail)
        tail_counts.append(int(diag["tail_count"]))
        tail_losses.append(float(diag["tail_loss"]))
        tail_thresholds.append(float(diag["tail_threshold"]))
    raw_anchor_state = v2308.model_state(model)
    train_raw_anchor = v2308.actual_metrics(model, x, y)
    guard_raw_anchor = v2308.actual_metrics(model, xg, yg)
    raw_debt = v2308.debt_deltas(guard_original, guard_raw_anchor)
    raw_no_debt = v2308.no_debt_ok(raw_debt, float(args.no_debt_budget))
    alphas = float_items(str(args.stage1_anchor_alphas))
    projection_enabled = not (len(alphas) == 1 and abs(float(alphas[0]) - 1.0) <= 1.0e-12)
    if not projection_enabled:
        selected_alpha = 1.0
        selected_state = raw_anchor_state
        train_raw_anchor = v2308.actual_metrics(model, x, y)
        guard_raw_anchor = v2308.actual_metrics(model, xg, yg)
        train_anchor = train_raw_anchor
        guard_anchor = guard_raw_anchor
        selected_no_debt = raw_no_debt
        selected_reason = "raw_anchor_no_projection"
    else:
        selected_alpha = 0.0
        selected_state = original_state
        train_anchor = train_original
        guard_anchor = guard_original
        selected_no_debt = 1
        selected_reason = "alpha_zero_fallback"
        for alpha in alphas:
            cand_state = interpolated_state(original_state, raw_anchor_state, float(alpha))
            v2308.load_model_state(model, cand_state)
            train_cand = v2308.actual_metrics(model, x, y)
            guard_cand = v2308.actual_metrics(model, xg, yg)
            cand_debt = v2308.debt_deltas(guard_original, guard_cand)
            cand_no_debt = v2308.no_debt_ok(cand_debt, float(args.no_debt_budget))
            if cand_no_debt:
                selected_alpha = float(alpha)
                selected_state = cand_state
                train_anchor = train_cand
                guard_anchor = guard_cand
                selected_no_debt = cand_no_debt
                selected_reason = "first_no_debt_alpha"
                break
    v2308.load_model_state(model, selected_state)
    with torch.no_grad():
        _lg1, guard_acts_anchor = model.forward_with_activations(xg)
    drift = activation_drift_stats(guard_acts_original, guard_acts_anchor)
    jac_drift = downstream_jacobian_drift_proxy(model, xg, original_state, selected_state, args)
    v2308.load_model_state(model, selected_state)
    return {
        "model": model,
        "x": x,
        "y": y,
        "xg": xg,
        "yg": yg,
        "classes": classes,
        "model_seed": model_seed,
        "ckpt": ckpt,
        "original_state": original_state,
        "anchor_state": selected_state,
        "raw_anchor_state": raw_anchor_state,
        "train_original": train_original,
        "guard_original": guard_original,
        "train_anchor": train_anchor,
        "guard_anchor": guard_anchor,
        "guard_acts_original": guard_acts_original,
        "anchor_drift": drift,
        "anchor_jacobian_drift": jac_drift,
        "anchor_diag": {
            "anchor_tail_count_median": median(tail_counts),
            "anchor_tail_count_min": min(tail_counts or [0]),
            "anchor_tail_loss_initial": tail_losses[0] if tail_losses else 0.0,
            "anchor_tail_loss_final": tail_losses[-1] if tail_losses else 0.0,
            "anchor_tail_threshold_initial": tail_thresholds[0] if tail_thresholds else 0.0,
            "anchor_tail_threshold_final": tail_thresholds[-1] if tail_thresholds else 0.0,
            "anchor_tail_shuffled": int(shuffle_tail),
            "anchor_alpha_candidates": str(args.stage1_anchor_alphas),
            "anchor_alpha_projection_used": int(projection_enabled),
            "anchor_alpha_selected": selected_alpha,
            "anchor_alpha_selection_reason": selected_reason,
            "raw_anchor_guard_coverage": guard_raw_anchor["coverage"],
            "raw_anchor_total_C2_coverage_improvement": guard_raw_anchor["coverage"] - guard_original["coverage"],
            "raw_anchor_F5_no_debt_pass": raw_no_debt,
            "raw_anchor_Brier_delta": raw_debt["Brier_delta"],
            "raw_anchor_ECE_delta": raw_debt["ECE_delta"],
            "raw_anchor_tail95_delta": raw_debt["tail95_delta"],
            "raw_anchor_tail99_delta": raw_debt["tail99_delta"],
            "raw_anchor_margin10_delta": raw_debt["margin10_delta"],
        },
    }


def run_stage1_row(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    start = time.time()
    try:
        ctx = prepare_tail_cvar_anchor(task, seed, args, device, shuffle_tail=False)
        train_original = ctx["train_original"]
        guard_original = ctx["guard_original"]
        train_anchor = ctx["train_anchor"]
        guard_anchor = ctx["guard_anchor"]
        debt = v2308.debt_deltas(guard_original, guard_anchor)
        no_debt = v2308.no_debt_ok(debt, float(args.no_debt_budget))
        total_cov = guard_anchor["coverage"] - guard_original["coverage"]
        total_acc = guard_anchor["accuracy"] - guard_original["accuracy"]
        total_loss = guard_anchor["loss"] - guard_original["loss"]
        drift = ctx["anchor_drift"]
        row = {
            "part": "1",
            "status": "ok",
            "scheme": "S1_tail_cvar_feature_anchor_total_audit",
            "task": task,
            "seed": int(seed),
            "basis_key": str(args.basis_key),
            "depth": int(args.depth),
            "train_size": int(args.train_size),
            "guard_size": int(args.guard_size),
            "checkpoint_name": str(args.stage1_checkpoint),
            "checkpoint_optimizer": ctx["ckpt"].get("checkpoint_optimizer", ""),
            "checkpoint_train_steps": ctx["ckpt"].get("checkpoint_train_steps", ""),
            "model_seed_scheme_key": str(args.part_c_model_seed_scheme_key),
            "model_seed": ctx["model_seed"],
            "anchor_steps": int(args.stage1_anchor_steps),
            "anchor_lr": float(args.stage1_anchor_lr),
            "anchor_tail_quantile": float(args.stage1_tail_quantile),
            **ctx["anchor_diag"],
            "original_train_coverage": train_original["coverage"],
            "anchor_train_coverage": train_anchor["coverage"],
            "original_guard_coverage": guard_original["coverage"],
            "anchor_guard_coverage": guard_anchor["coverage"],
            "total_C2_coverage_improvement": total_cov,
            "total_C2_accuracy_improvement": total_acc,
            "total_guard_nll_delta": total_loss,
            "total_train_nll_delta": train_anchor["loss"] - train_original["loss"],
            "source_guard_c2_gap": (train_anchor["coverage"] - train_original["coverage"]) - total_cov,
            "F5_no_debt_pass": no_debt,
            "Brier_delta": debt["Brier_delta"],
            "ECE_delta": debt["ECE_delta"],
            "tail95_delta": debt["tail95_delta"],
            "tail99_delta": debt["tail99_delta"],
            "margin10_delta": debt["margin10_delta"],
            "debt_delta": debt["debt_delta"],
            "margin10_delta_correct_sign": int(debt["margin10_delta"] >= -float(args.no_debt_budget)),
            "component_non_positive": int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0),
            "activation_drift_median": drift["activation_drift_median"],
            "activation_drift_max": drift["activation_drift_max"],
            "activation_drift_layer_count": drift["activation_drift_layer_count"],
            "downstream_jacobian_drift_proxy": ctx["anchor_jacobian_drift"],
            "original_to_anchor_total_audit": 1,
            "post_anchor_incremental_only": 0,
            "wall_time_s": time.time() - start,
        }
        return {**AUDIT_DEFAULTS, **row}
    except Exception as exc:
        return {**AUDIT_DEFAULTS, "part": "1", "status": "error", "scheme": "S1_tail_cvar_feature_anchor_total_audit", "task": task, "seed": int(seed), "error_message": repr(exc), "wall_time_s": time.time() - start}


def stage1_run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    lock_path = OUT_ROOT / "stage0_evidence_lock_summary.json"
    if not lock_path.exists() or ival(read_json(lock_path).get("gate_pass")) != 1:
        summary = {
            "part": "1",
            "route": "Stage1BlockedByEvidenceLock",
            "gate_pass": 0,
            "dominant_blocker": "stage0_evidence_lock_missing_or_failed",
        }
        write_json(OUT_ROOT / "stage1_anchor_feasibility_summary.json", summary)
        append_exec("Stage 1 anchor shard blocked", command_text(), files=rel(OUT_ROOT / "stage1_anchor_feasibility_summary.json"), gpu=str(args.device), note=summary["dominant_blocker"])
        return summary
    device = torch.device(str(args.device))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"stage1_anchor_feasibility_matrix{suffix}.csv"
    existing = read_rows(matrix) if int(args.stage1_resume) else []
    done = {(r.get("task"), str(r.get("seed"))) for r in existing if r.get("status") == "ok"}
    jobs = [job for idx, job in enumerate(stage1_jobs(args)) if idx % int(args.shard_count) == int(args.shard_index)]
    rows = list(existing)
    for task, seed in jobs:
        if (task, str(seed)) in done:
            continue
        rows.append(run_stage1_row(task, int(seed), args, device))
        if int(args.stage1_flush_every) > 0:
            write_rows(matrix, rows)
            print(json.dumps({"part": "1", "rows_written": len(rows), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
    write_rows(matrix, rows)
    summary = {
        "part": "1",
        "route": "Stage1ShardDone",
        "gate_pass": 0,
        "matrix": rel(matrix),
        "rows_written": len(rows),
        "ok_rows": sum(1 for row in rows if row.get("status") == "ok"),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
    }
    write_json(OUT_ROOT / f"stage1_anchor_feasibility_summary{suffix}.json", summary)
    append_exec("Stage 1 anchor shard", command_text(), files=rel(matrix), gpu=str(args.device), note=f"rows={len(rows)}; ok={summary['ok_rows']}")
    return summary


def collect_stage1_rows() -> list[dict[str, str]]:
    shard_files = sorted(OUT_ROOT.glob("stage1_anchor_feasibility_matrix_shard*_of_*.csv"))
    files = shard_files if shard_files else [OUT_ROOT / "stage1_anchor_feasibility_matrix.csv"]
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
    matrix = write_rows(OUT_ROOT / "stage1_anchor_feasibility_matrix.csv", rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    expected_rows = len(stage1_jobs(args))
    group = {
        "scheme": "S1_tail_cvar_feature_anchor_total_audit",
        "rows": len(ok),
        "total_C2_coverage_improvement_median": median([r.get("total_C2_coverage_improvement") for r in ok]),
        "total_C2_accuracy_improvement_median": median([r.get("total_C2_accuracy_improvement") for r in ok]),
        "total_guard_nll_delta_median": median([r.get("total_guard_nll_delta") for r in ok]),
        "F5_no_debt_count": sum(ival(r.get("F5_no_debt_pass")) for r in ok),
        "component_non_positive_rows": sum(ival(r.get("component_non_positive")) for r in ok),
        "activation_drift_median_median": median([r.get("activation_drift_median") for r in ok]),
        "activation_drift_max_median": median([r.get("activation_drift_max") for r in ok]),
        "downstream_jacobian_drift_proxy_median": median([r.get("downstream_jacobian_drift_proxy") for r in ok]),
        "anchor_tail_count_median": median([r.get("anchor_tail_count_median") for r in ok]),
        "anchor_tail_loss_initial_median": median([r.get("anchor_tail_loss_initial") for r in ok]),
        "anchor_tail_loss_final_median": median([r.get("anchor_tail_loss_final") for r in ok]),
        "anchor_alpha_projection_used_rows": sum(ival(r.get("anchor_alpha_projection_used")) for r in ok),
        "anchor_alpha_selected_median": median([r.get("anchor_alpha_selected") for r in ok]),
        "raw_anchor_total_C2_coverage_improvement_median": median([r.get("raw_anchor_total_C2_coverage_improvement") for r in ok]),
        "raw_anchor_F5_no_debt_count": sum(ival(r.get("raw_anchor_F5_no_debt_pass")) for r in ok),
        "wall_time_s_median": median([r.get("wall_time_s") for r in ok]),
    }
    group_csv = write_rows(OUT_ROOT / "stage1_anchor_feasibility_group_summary.csv", [group])
    checks = {
        "row_count_10_ok": len(rows) == expected_rows and len(ok) == expected_rows,
        "no_fake_data": sum(ival(r.get("used_fake_data_rows")) for r in ok) == 0,
        "no_held_test_usage": sum(ival(r.get("held_test_usage")) for r in ok) == 0,
        "original_to_anchor_total_audit_all_rows": sum(ival(r.get("original_to_anchor_total_audit")) for r in ok) == len(ok),
        "post_anchor_incremental_only_no_rows": sum(ival(r.get("post_anchor_incremental_only")) for r in ok) == 0,
        "guard_no_debt_all_rows": int(group["F5_no_debt_count"]) == expected_rows,
        "coverage_median_ge_0p005": fval(group["total_C2_coverage_improvement_median"]) >= 0.005,
        "activation_drift_median_gt_0p001": fval(group["activation_drift_median_median"]) > 0.001,
        "activation_drift_median_lt_0p20": fval(group["activation_drift_median_median"]) < 0.20,
    }
    gate = int(all(checks.values()))
    if not checks["row_count_10_ok"]:
        blocker = "unexpected_stage1_row_count_or_errors"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif not checks["original_to_anchor_total_audit_all_rows"] or not checks["post_anchor_incremental_only_no_rows"]:
        blocker = "total_audit_semantics_violation"
    elif not checks["guard_no_debt_all_rows"]:
        blocker = "tail_cvar_anchor_introduces_guard_debt"
    elif not checks["activation_drift_median_gt_0p001"]:
        blocker = "tail_cvar_anchor_did_not_move_representation"
    elif not checks["activation_drift_median_lt_0p20"]:
        blocker = "tail_cvar_anchor_representation_collapse_or_too_large_drift"
    elif not checks["coverage_median_ge_0p005"]:
        blocker = "tail_cvar_anchor_drift_without_coverage_gain"
    else:
        blocker = "none"
    summary = {
        "part": "1",
        "route": "Stage1TailCvarFeatureAnchorFeasibilityPass" if gate else "Stage1TailCvarFeatureAnchorFeasibilityFailed",
        "gate_pass": gate,
        "scope": "anchor_only_total_audit_reduced_diagnostic",
        "dominant_blocker": blocker,
        "checks": checks,
        "metrics": group,
        "artifacts": {
            "matrix": rel(matrix),
            "group_summary": rel(group_csv),
        },
        "hashes": {
            "matrix": sha256_file(matrix),
            "group_summary": sha256_file(group_csv),
        },
        "next_action": "implement_stage2_C28_C32_total_audit_controls" if gate else "repair_or_stop_tail_cvar_feature_anchor_route_based_on_blocker",
        "interpretation": (
            "Stage 1 tests whether a short train-only lower-CVaR true-probability anchor can move representation/tangent geometry "
            "with original-to-anchor total audit. It is not a promotion candidate."
        ),
    }
    out = write_json(OUT_ROOT / "stage1_anchor_feasibility_summary.json", summary)
    nxt = write_json(
        FORMAL_ROOT / "next_actions_for_v23_12.json",
        {
            "version": "v23.12",
            "last_completed_stage": "stage1_tail_cvar_feature_anchor_feasibility",
            "stage1_gate_pass": gate,
            "stage1_route": summary["route"],
            "stage1_blocker": blocker,
            "promotion_allowed": 0,
            "completed_artifacts": {
                "stage1_summary": rel(out),
                "matrix": rel(matrix),
                "group_summary": rel(group_csv),
            },
            "allowed_next_actions": (
                ["implement_stage2_C28_C32_total_audit_controls"]
                if gate
                else ["if_no_debt_but_drift_low_try_stronger_anchor_once", "if_debt_or_collapse_try_weaker_anchor_once", "otherwise_stop_tail_cvar_feature_anchor_route"]
            ),
            "forbidden_next_actions": [
                "promote_stage1_anchor_only_as_coverage_dominance",
                "measure_only_post_anchor_incremental_delta_for_feature_anchor_candidate",
                "fabricate_data",
                "held_test_induction",
                "runtime_winner_selection",
                "add_mlp_stem_or_readout",
                "add_new_edge_function_family",
            ],
        },
    )
    append_exec("Stage 1 anchor feasibility merge", command_text(), files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Stage 1 tail-CVaR feature-anchor feasibility", summary)
    return summary


def stage2_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    tasks = [item.strip() for item in str(args.stage2_tasks).split(",") if item.strip()]
    schemes = [item.strip() for item in str(args.stage2_schemes).split(",") if item.strip()]
    return [(scheme, task, seed) for scheme in schemes for task in tasks for seed in range(int(args.stage2_seed_count))]


def stage2_f_args(args: argparse.Namespace, *, trust_mode: str, trust_tolerance: float, alphas: str | None = None) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.part_f_steps = int(args.stage2_steps)
    out.part_f_residual_target = str(args.stage2_residual_target)
    out.part_f_lambda = float(args.stage2_lambda)
    out.part_f_alphas = str(alphas if alphas is not None else args.stage2_alphas)
    out.part_f_delta_scale = float(args.stage2_delta_scale)
    out.part_f_delta_sign = float(args.stage2_delta_sign)
    out.part_f_trust_mode = str(trust_mode)
    out.part_f_trust_tolerance = float(trust_tolerance)
    out.part_f_local_reference_target = str(args.stage2_local_reference_target)
    out.part_f_functionalgram_lr = float(args.stage2_functionalgram_lr)
    out.part_f_adam_lr = float(args.stage2_adam_lr)
    out.part_f_solver = str(args.stage2_solver)
    out.part_f_sketch_rank = int(args.stage2_sketch_rank)
    out.part_f_activation_drift_cap = float(args.stage2_activation_drift_cap)
    out.part_f_alignment_min = -2.0
    out.part_f_alignment_soft_floor = -2.0
    out.part_f_alignment_soft_ceiling = 1.0
    out.part_f_alignment_soft_min_scale = 0.25
    out.part_f_alignment_soft_power = 0.5
    out.part_f_tail99_step_budget = -1.0
    out.part_f_tail99_cumulative_budget = -1.0
    out.part_f_tail_correction_scales = "0.25,0.125,0.0625,0"
    out.part_f_tail_correction_target = "tail_weighted_norm"
    out.part_f_tail_correction_layer_group = "topdown"
    out.part_f_tail_correction_transport_min = 0.90
    out.part_f_checkpoint = str(args.stage1_checkpoint)
    return out


def run_multistep_from_current(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    scheme: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    train_start = v2308.actual_metrics(model, x, y)
    guard_start = v2308.actual_metrics(model, xg, yg)
    _logits0, acts0 = model.forward_with_activations(xg)
    finite_accept = 0
    finite_skip = 0
    finite_scales: list[float] = []
    reject_reasons: dict[str, int] = {}
    solve_residuals: list[float] = []
    conds: list[float] = []
    cg_iters: list[float] = []
    step_gains: list[float] = []
    activation_drift_curve: list[float] = []
    for step in range(1, int(args.part_f_steps) + 1):
        source_before = v2308.actual_metrics(model, x, y)
        step_guard_before = v2308.actual_metrics(model, xg, yg)
        base_state = v2308.model_state(model)
        if scheme in v2308.HYBRID_REFRESH_INTERVALS and (step % int(v2308.HYBRID_REFRESH_INTERVALS[scheme])) != 1:
            diag = {"solve_residual_max": 0.0, "condition_number_after_ridge_max": 0.0, "cg_iterations": 0.0}
            alpha, accepted, reject_reason, source_after, guard_after = v2308.choose_alpha_edge_optimizer_f(
                model,
                base_state,
                source_before,
                step_guard_before,
                x,
                y,
                xg,
                yg,
                args,
                use_population_gate=False,
                base_acts=acts0,
                tail99_reference=guard_start,
            )
        else:
            e_scheme, rank = v2308.f_scheme_to_e_scheme(scheme)
            eargs = v2308.f_e_args(args, sketch_rank=rank)
            raw_deltas, diag = v2308.e_delta_candidate(
                e_scheme,
                model,
                x,
                y,
                xg,
                yg,
                str(args.part_f_residual_target),
                eargs,
                device,
                seed=23120000 + int(step),
                task="stage2_total_audit",
                checkpoint=str(args.part_f_checkpoint),
            )
            deltas = v2308.scaled_deltas(raw_deltas, eargs)
            alpha, accepted, reject_reason, source_after, guard_after = v2308.choose_alpha_for_f_deltas(
                model,
                base_state,
                deltas,
                source_before,
                step_guard_before,
                x,
                y,
                xg,
                yg,
                eargs,
                acts0,
                tail99_reference=guard_start,
            )
        finite_accept += int(accepted)
        finite_skip += int(not accepted)
        finite_scales.append(float(alpha))
        reject_reasons[str(reject_reason)] = reject_reasons.get(str(reject_reason), 0) + 1
        solve_residuals.append(fval(diag.get("solve_residual_max")))
        conds.append(fval(diag.get("condition_number_after_ridge_max")))
        cg_iters.append(fval(diag.get("cg_iterations")))
        step_gains.append(guard_after["coverage"] - step_guard_before["coverage"])
        _lt, acts_t = model.forward_with_activations(xg)
        drift_vals = [
            float((after.detach().to(dtype=torch.float64) - before.detach().to(dtype=torch.float64)).norm().div(before.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12)).detach().cpu().item())
            for before, after in zip(acts0[1:], acts_t[1:])
        ]
        activation_drift_curve.append(max(drift_vals or [0.0]))
    train_final = v2308.actual_metrics(model, x, y)
    guard_final = v2308.actual_metrics(model, xg, yg)
    return {
        "start_train": train_start,
        "start_guard": guard_start,
        "final_train": train_final,
        "final_guard": guard_final,
        "finite_step_accept_count": finite_accept,
        "finite_step_accept_rate": finite_accept / max(1, int(args.part_f_steps)),
        "finite_step_scale_mean": mean(finite_scales),
        "finite_step_skip_count": finite_skip,
        "finite_step_reject_reasons": json.dumps(reject_reasons, sort_keys=True),
        "solve_residual_max": max(solve_residuals or [0.0]),
        "condition_number_median": median(conds),
        "cg_iteration_median": median(cg_iters),
        "one_step_gain_retention_curve": json.dumps(step_gains),
        "activation_drift_max": max(activation_drift_curve or [0.0]),
        "activation_drift_median": median(activation_drift_curve),
    }


def total_metric_row(
    *,
    scheme: str,
    source_scheme: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    ctx: dict[str, Any],
    final_train: dict[str, float],
    final_guard: dict[str, float],
    start_time: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    guard_original = ctx["guard_original"]
    train_original = ctx["train_original"]
    debt = v2308.debt_deltas(guard_original, final_guard)
    return {
        **AUDIT_DEFAULTS,
        "part": "2",
        "status": "ok",
        "scheme": scheme,
        "source_scheme": source_scheme,
        "task": task,
        "seed": int(seed),
        "basis_key": str(args.basis_key),
        "depth": int(args.depth),
        "train_size": int(args.train_size),
        "guard_size": int(args.guard_size),
        "checkpoint_name": str(args.stage1_checkpoint),
        "checkpoint_optimizer": ctx["ckpt"].get("checkpoint_optimizer", ""),
        "anchor_steps": int(args.stage1_anchor_steps),
        "anchor_lr": float(args.stage1_anchor_lr),
        "anchor_tail_quantile": float(args.stage1_tail_quantile),
        **ctx["anchor_diag"],
        "original_guard_coverage": guard_original["coverage"],
        "anchor_guard_coverage": ctx["guard_anchor"]["coverage"],
        "final_guard_coverage": final_guard["coverage"],
        "total_C2_coverage_improvement": final_guard["coverage"] - guard_original["coverage"],
        "anchor_to_final_C2_coverage_improvement": final_guard["coverage"] - ctx["guard_anchor"]["coverage"],
        "total_C2_accuracy_improvement": final_guard["accuracy"] - guard_original["accuracy"],
        "total_guard_nll_delta": final_guard["loss"] - guard_original["loss"],
        "total_train_nll_delta": final_train["loss"] - train_original["loss"],
        "source_guard_c2_gap": (final_train["coverage"] - train_original["coverage"]) - (final_guard["coverage"] - guard_original["coverage"]),
        "F5_no_debt_pass": v2308.no_debt_ok(debt, float(args.no_debt_budget)),
        "Brier_delta": debt["Brier_delta"],
        "ECE_delta": debt["ECE_delta"],
        "tail95_delta": debt["tail95_delta"],
        "tail99_delta": debt["tail99_delta"],
        "margin10_delta": debt["margin10_delta"],
        "debt_delta": debt["debt_delta"],
        "margin10_delta_correct_sign": int(debt["margin10_delta"] >= -float(args.no_debt_budget)),
        "component_non_positive": int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0),
        "original_to_final_total_audit": 1,
        "post_anchor_incremental_only": 0,
        "wall_time_s": time.time() - start_time,
        **extra,
    }


def run_stage2_row(scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    start = time.time()
    try:
        shuffle_anchor = scheme == "C31_Control_shuffled_tail_anchor_then_terminal_C15_total_audit"
        ctx = prepare_tail_cvar_anchor(task, seed, args, device, shuffle_tail=shuffle_anchor)
        model = ctx["model"]
        x, y, xg, yg = ctx["x"], ctx["y"], ctx["xg"], ctx["yg"]
        v2308.load_model_state(model, ctx["anchor_state"])
        if scheme in {"C29_Control_tail_cvar_anchor_only_total_audit", "C32_Control_tail_cvar_anchor_then_no_inverse_total_audit"}:
            source_scheme = "anchor_only" if scheme.startswith("C29") else "anchor_then_no_inverse"
            return total_metric_row(
                scheme=scheme,
                source_scheme=source_scheme,
                task=task,
                seed=seed,
                args=args,
                ctx=ctx,
                final_train=ctx["train_anchor"],
                final_guard=ctx["guard_anchor"],
                start_time=start,
                extra={
                    "terminal_total_audit_pass": 1,
                    "terminal_total_audit_reject": 0,
                    "finite_step_accept_count": 0,
                    "finite_step_accept_rate": 0.0,
                    "finite_step_scale_mean": 0.0,
                    "finite_step_skip_count": 0,
                    "finite_step_reject_reasons": json.dumps({"anchor_only_or_no_inverse": 1}, sort_keys=True),
                },
            )
        if scheme in {"C28_Candidate_tail_cvar_anchor_then_terminal_C15_total_audit", "C31_Control_shuffled_tail_anchor_then_terminal_C15_total_audit"}:
            source_scheme = str(args.stage2_c15_source_scheme)
            fargs = stage2_f_args(args, trust_mode="coverage_floor", trust_tolerance=1.0e9, alphas=str(args.stage2_alphas))
            diag = run_multistep_from_current(model, x, y, xg, yg, source_scheme, fargs, device)
            raw_final_state = v2308.model_state(model)
            raw_final_train, raw_final_guard = diag["final_train"], diag["final_guard"]
            raw_terminal_pass, raw_debt = strict_total_audit_pass(ctx["guard_original"], raw_final_guard, float(args.no_debt_budget))
            terminal_alphas = float_items(str(args.stage2_terminal_alphas))
            projection_enabled = not (len(terminal_alphas) == 1 and abs(float(terminal_alphas[0]) - 1.0) <= 1.0e-12)
            selected_alpha = 1.0 if raw_terminal_pass else 0.0
            selected_reason = "raw_inverse_pass" if raw_terminal_pass else "raw_inverse_rejected_anchor_rollback"
            final_train, final_guard = raw_final_train, raw_final_guard
            terminal_pass = raw_terminal_pass
            terminal_reject = int(not raw_terminal_pass)
            if projection_enabled:
                selected_alpha = 0.0
                selected_reason = "alpha_zero_fallback"
                final_train, final_guard = ctx["train_anchor"], ctx["guard_anchor"]
                terminal_pass = 0
                terminal_reject = 1
                for alpha in terminal_alphas:
                    if abs(float(alpha) - 1.0) <= 1.0e-12:
                        cand_state = raw_final_state
                        train_cand, guard_cand = raw_final_train, raw_final_guard
                    elif abs(float(alpha)) <= 1.0e-12:
                        cand_state = ctx["anchor_state"]
                        train_cand, guard_cand = ctx["train_anchor"], ctx["guard_anchor"]
                    else:
                        cand_state = interpolated_state(ctx["anchor_state"], raw_final_state, float(alpha))
                        v2308.load_model_state(model, cand_state)
                        train_cand = v2308.actual_metrics(model, x, y)
                        guard_cand = v2308.actual_metrics(model, xg, yg)
                    cand_pass, _cand_debt = strict_total_audit_pass(ctx["guard_original"], guard_cand, float(args.no_debt_budget))
                    if cand_pass:
                        selected_alpha = float(alpha)
                        selected_reason = "alpha_zero_fallback" if abs(float(alpha)) <= 1.0e-12 else "first_total_audit_alpha"
                        final_train, final_guard = train_cand, guard_cand
                        terminal_pass = 1
                        terminal_reject = 0
                        v2308.load_model_state(model, cand_state)
                        break
            elif terminal_reject:
                v2308.load_model_state(model, ctx["anchor_state"])
                final_train, final_guard = ctx["train_anchor"], ctx["guard_anchor"]
            extra = {
                key: value
                for key, value in diag.items()
                if key not in {"start_train", "start_guard", "final_train", "final_guard"}
            }
            extra.update({
                "terminal_total_audit_pass": terminal_pass,
                "terminal_total_audit_reject": terminal_reject,
                "inverse_terminal_alpha_candidates": str(args.stage2_terminal_alphas),
                "inverse_terminal_alpha_projection_used": int(projection_enabled),
                "inverse_terminal_alpha_selected": selected_alpha,
                "inverse_terminal_alpha_selection_reason": selected_reason,
                "raw_inverse_guard_coverage": raw_final_guard["coverage"],
                "raw_inverse_total_C2_coverage_improvement": raw_final_guard["coverage"] - ctx["guard_original"]["coverage"],
                "raw_inverse_terminal_total_audit_pass": raw_terminal_pass,
                "raw_inverse_terminal_total_audit_reject": int(not raw_terminal_pass),
                "raw_inverse_Brier_delta": raw_debt["Brier_delta"],
                "raw_inverse_ECE_delta": raw_debt["ECE_delta"],
                "raw_inverse_tail95_delta": raw_debt["tail95_delta"],
                "raw_inverse_tail99_delta": raw_debt["tail99_delta"],
                "raw_inverse_margin10_delta": raw_debt["margin10_delta"],
            })
            return total_metric_row(
                scheme=scheme,
                source_scheme=source_scheme,
                task=task,
                seed=seed,
                args=args,
                ctx=ctx,
                final_train=final_train,
                final_guard=final_guard,
                start_time=start,
                extra=extra,
            )
        if scheme == "C30_Control_tail_cvar_anchor_then_matched_LocalEFRF_total_audit":
            source_scheme = "F3_v23_07_local_EFRF_all_layer_GS_control"
            fargs = stage2_f_args(args, trust_mode=str(args.stage2_trust_mode), trust_tolerance=float(args.stage2_trust_tolerance), alphas=str(args.stage2_alphas))
            diag = run_multistep_from_current(model, x, y, xg, yg, source_scheme, fargs, device)
            extra = {
                key: value
                for key, value in diag.items()
                if key not in {"start_train", "start_guard", "final_train", "final_guard"}
            }
            extra.update({"terminal_total_audit_pass": v2308.no_debt_ok(v2308.debt_deltas(ctx["guard_original"], diag["final_guard"]), float(args.no_debt_budget)), "terminal_total_audit_reject": 0})
            return total_metric_row(
                scheme=scheme,
                source_scheme=source_scheme,
                task=task,
                seed=seed,
                args=args,
                ctx=ctx,
                final_train=diag["final_train"],
                final_guard=diag["final_guard"],
                start_time=start,
                extra=extra,
            )
        raise ValueError(f"unknown stage2 scheme {scheme}")
    except Exception as exc:
        return {**AUDIT_DEFAULTS, "part": "2", "status": "error", "scheme": scheme, "task": task, "seed": int(seed), "error_message": repr(exc), "wall_time_s": time.time() - start}


def stage2_run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    stage1_path = Path(str(args.stage2_stage1_summary))
    if not stage1_path.is_absolute():
        stage1_path = ROOT / stage1_path
    if not stage1_path.exists() or ival(read_json(stage1_path).get("gate_pass")) != 1:
        summary = {"part": "2", "route": "Stage2BlockedByStage1", "gate_pass": 0, "dominant_blocker": "stage1_summary_missing_or_failed"}
        write_json(OUT_ROOT / "stage2_total_audit_summary.json", summary)
        append_exec("Stage 2 total-audit shard blocked", command_text(), files=rel(OUT_ROOT / "stage2_total_audit_summary.json"), gpu=str(args.device), note=summary["dominant_blocker"])
        return summary
    device = torch.device(str(args.device))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"stage2_total_audit_matrix{suffix}.csv"
    existing = read_rows(matrix) if int(args.stage2_resume) else []
    done = {(r.get("scheme"), r.get("task"), str(r.get("seed"))) for r in existing if r.get("status") == "ok"}
    jobs = [job for idx, job in enumerate(stage2_jobs(args)) if idx % int(args.shard_count) == int(args.shard_index)]
    rows = list(existing)
    for scheme, task, seed in jobs:
        if (scheme, task, str(seed)) in done:
            continue
        rows.append(run_stage2_row(scheme, task, int(seed), args, device))
        if int(args.stage2_flush_every) > 0:
            write_rows(matrix, rows)
            print(json.dumps({"part": "2", "rows_written": len(rows), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
    write_rows(matrix, rows)
    summary = {"part": "2", "route": "Stage2ShardDone", "gate_pass": 0, "matrix": rel(matrix), "rows_written": len(rows), "ok_rows": sum(1 for row in rows if row.get("status") == "ok")}
    write_json(OUT_ROOT / f"stage2_total_audit_summary{suffix}.json", summary)
    append_exec("Stage 2 total-audit shard", command_text(), files=rel(matrix), gpu=str(args.device), note=f"rows={len(rows)}; ok={summary['ok_rows']}")
    return summary


def collect_stage2_rows() -> list[dict[str, str]]:
    shard_files = sorted(OUT_ROOT.glob("stage2_total_audit_matrix_shard*_of_*.csv"))
    files = shard_files if shard_files else [OUT_ROOT / "stage2_total_audit_matrix.csv"]
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


def row_for_scheme(rows: list[dict[str, str]], scheme: str) -> dict[str, str]:
    for row in rows:
        if row.get("scheme") == scheme:
            return row
    return {}


def stage2_merge(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    rows = collect_stage2_rows()
    matrix = write_rows(OUT_ROOT / "stage2_total_audit_matrix.csv", rows)
    ok = [row for row in rows if row.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for scheme in [item.strip() for item in str(args.stage2_schemes).split(",") if item.strip()]:
        group_rows = [row for row in ok if row.get("scheme") == scheme]
        groups.append({
            "scheme": scheme,
            "rows": len(group_rows),
            "total_C2_coverage_improvement_median": median([r.get("total_C2_coverage_improvement") for r in group_rows]),
            "total_C2_accuracy_improvement_median": median([r.get("total_C2_accuracy_improvement") for r in group_rows]),
            "total_guard_nll_delta_median": median([r.get("total_guard_nll_delta") for r in group_rows]),
            "F5_no_debt_count": sum(ival(r.get("F5_no_debt_pass")) for r in group_rows),
            "component_non_positive_rows": sum(ival(r.get("component_non_positive")) for r in group_rows),
            "terminal_total_audit_reject_count": sum(ival(r.get("terminal_total_audit_reject")) for r in group_rows),
            "anchor_alpha_selected_median": median([r.get("anchor_alpha_selected") for r in group_rows]),
            "raw_anchor_total_C2_coverage_improvement_median": median([r.get("raw_anchor_total_C2_coverage_improvement") for r in group_rows]),
            "inverse_terminal_alpha_selected_median": median([r.get("inverse_terminal_alpha_selected") for r in group_rows]),
            "inverse_terminal_projection_used_rows": sum(ival(r.get("inverse_terminal_alpha_projection_used")) for r in group_rows),
            "raw_inverse_total_C2_coverage_improvement_median": median([r.get("raw_inverse_total_C2_coverage_improvement") for r in group_rows]),
            "raw_inverse_terminal_pass_count": sum(ival(r.get("raw_inverse_terminal_total_audit_pass")) for r in group_rows),
            "raw_inverse_terminal_reject_count": sum(ival(r.get("raw_inverse_terminal_total_audit_reject")) for r in group_rows),
            "finite_step_accept_rate_median": median([r.get("finite_step_accept_rate") for r in group_rows]),
            "wall_time_s_median": median([r.get("wall_time_s") for r in group_rows]),
        })
    group_csv = write_rows(OUT_ROOT / "stage2_total_audit_group_summary.csv", groups)
    expected_rows = len(stage2_jobs(args))
    by_scheme = {row["scheme"]: row for row in groups}
    c28 = by_scheme.get("C28_Candidate_tail_cvar_anchor_then_terminal_C15_total_audit", {})
    c29 = by_scheme.get("C29_Control_tail_cvar_anchor_only_total_audit", {})
    c30 = by_scheme.get("C30_Control_tail_cvar_anchor_then_matched_LocalEFRF_total_audit", {})
    c31 = by_scheme.get("C31_Control_shuffled_tail_anchor_then_terminal_C15_total_audit", {})
    required_stage2_controls_present = all(
        scheme in by_scheme and int(fval(by_scheme.get(scheme, {}).get("rows"))) > 0
        for scheme in STAGE2_SCHEMES
    )
    default = read_json(DEFAULT_REDUCED_REF) if DEFAULT_REDUCED_REF.exists() else {}
    default_metrics = default.get("metrics", {})
    default_c3 = fval(default_metrics.get("C3_coverage"))
    default_c15 = fval(default_metrics.get("C15_coverage"))
    default_groups = read_rows(DEFAULT_REDUCED_GROUP)
    default_c15_wall = fval(row_for_scheme(default_groups, "C15_Candidate_terminal_audited_no_trust").get("wall_time_s_median"), 0.0)
    c28_cov = fval(c28.get("total_C2_coverage_improvement_median"))
    c28_expected_rows = sum(1 for scheme, _task, _seed in stage2_jobs(args) if scheme == "C28_Candidate_tail_cvar_anchor_then_terminal_C15_total_audit")
    checks = {
        "row_count_50_ok": len(rows) == expected_rows and len(ok) == expected_rows,
        "no_fake_data": sum(ival(r.get("used_fake_data_rows")) for r in ok) == 0,
        "no_held_test_usage": sum(ival(r.get("held_test_usage")) for r in ok) == 0,
        "original_to_final_total_audit_all_rows": sum(ival(r.get("original_to_final_total_audit")) for r in ok) == len(ok),
        "post_anchor_incremental_only_no_rows": sum(ival(r.get("post_anchor_incremental_only")) for r in ok) == 0,
        "required_stage2_controls_present": required_stage2_controls_present,
        "c28_terminal_total_no_debt_clean": c28_expected_rows > 0 and fval(c28.get("F5_no_debt_count")) == c28_expected_rows and fval(c28.get("terminal_total_audit_reject_count")) == 0,
        "c28_minus_c30_ge_0p01": c28_cov - fval(c30.get("total_C2_coverage_improvement_median")) >= 0.01,
        "c28_minus_c29_ge_0p01": c28_cov - fval(c29.get("total_C2_coverage_improvement_median")) >= 0.01,
        "c28_minus_c31_ge_0p03": c28_cov - fval(c31.get("total_C2_coverage_improvement_median")) >= 0.03,
        "c28_minus_default_c3_ge_0p01": c28_cov - default_c3 >= 0.01,
        "c28_minus_default_c15_ge_0p01": c28_cov - default_c15 >= 0.01,
        "c28_wall_time_le_5x_default_c15": default_c15_wall > 0.0 and fval(c28.get("wall_time_s_median")) <= 5.0 * default_c15_wall,
    }
    gate = int(all(checks.values()))
    if not checks["row_count_50_ok"]:
        blocker = "unexpected_stage2_row_count_or_errors"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif not checks["original_to_final_total_audit_all_rows"] or not checks["post_anchor_incremental_only_no_rows"]:
        blocker = "total_audit_semantics_violation"
    elif not checks["required_stage2_controls_present"]:
        blocker = "missing_required_stage2_controls_partial_diagnostic_only"
    elif not checks["c28_terminal_total_no_debt_clean"]:
        blocker = "c28_terminal_total_audit_reject_or_debt"
    elif not checks["c28_minus_c30_ge_0p01"]:
        blocker = "c28_does_not_beat_matched_localefrf_after_anchor"
    elif not checks["c28_minus_c29_ge_0p01"]:
        blocker = "c28_does_not_beat_anchor_only"
    elif not checks["c28_minus_c31_ge_0p03"]:
        blocker = "c28_does_not_beat_shuffled_tail_anchor_control"
    elif not checks["c28_minus_default_c3_ge_0p01"] or not checks["c28_minus_default_c15_ge_0p01"]:
        blocker = "c28_does_not_exceed_default_c3_c15_ceiling"
    elif not checks["c28_wall_time_le_5x_default_c15"]:
        blocker = "c28_wall_time_too_high_or_reference_missing"
    else:
        blocker = "none"
    summary = {
        "part": "2",
        "route": "Stage2C28TotalAuditReducedPass" if gate else "Stage2C28TotalAuditReducedFailed",
        "gate_pass": gate,
        "scope": "stage2_total_audit_reduced_not_full_promotion",
        "dominant_blocker": blocker,
        "checks": checks,
        "metrics": {
            "default_c3_coverage": default_c3,
            "default_c15_coverage": default_c15,
            "default_c15_wall_time_median": default_c15_wall,
            "c28_coverage": c28_cov,
            "c29_coverage": fval(c29.get("total_C2_coverage_improvement_median")),
            "c30_coverage": fval(c30.get("total_C2_coverage_improvement_median")),
            "c31_coverage": fval(c31.get("total_C2_coverage_improvement_median")),
            "c28_minus_c29": c28_cov - fval(c29.get("total_C2_coverage_improvement_median")),
            "c28_minus_c30": c28_cov - fval(c30.get("total_C2_coverage_improvement_median")),
            "c28_minus_c31": c28_cov - fval(c31.get("total_C2_coverage_improvement_median")),
            "c28_minus_default_c3": c28_cov - default_c3,
            "c28_minus_default_c15": c28_cov - default_c15,
            "c28_terminal_reject_count": fval(c28.get("terminal_total_audit_reject_count")),
            "c28_no_debt_count": fval(c28.get("F5_no_debt_count")),
            "c28_wall_time_s_median": fval(c28.get("wall_time_s_median")),
            "c28_inverse_terminal_alpha_selected_median": fval(c28.get("inverse_terminal_alpha_selected_median")),
            "c28_raw_inverse_coverage": fval(c28.get("raw_inverse_total_C2_coverage_improvement_median")),
            "c28_raw_inverse_terminal_reject_count": fval(c28.get("raw_inverse_terminal_reject_count")),
        },
        "artifacts": {"matrix": rel(matrix), "group_summary": rel(group_csv)},
        "hashes": {"matrix": sha256_file(matrix), "group_summary": sha256_file(group_csv)},
        "next_action": "consider_full_controls_for_C28_only_after_rechecking_no_selection_bias" if gate else "do_not_full_run_C28; analyze_stage2_blocker",
    }
    out = write_json(OUT_ROOT / "stage2_total_audit_summary.json", summary)
    nxt = write_json(
        FORMAL_ROOT / "next_actions_for_v23_12.json",
        {
            "version": "v23.12",
            "last_completed_stage": "stage2_c28_total_audit_reduced",
            "stage2_gate_pass": gate,
            "stage2_route": summary["route"],
            "stage2_blocker": blocker,
            "promotion_allowed": 0,
            "completed_artifacts": {"stage2_summary": rel(out), "matrix": rel(matrix), "group_summary": rel(group_csv)},
            "allowed_next_actions": ["consider_full_controls_for_C28_only_after_rechecking_no_selection_bias"] if gate else ["do_not_full_run_C28; analyze_stage2_blocker"],
            "forbidden_next_actions": [
                "promote_C28_without_full_C0_C10_C15_C28_C32_controls",
                "measure_only_post_anchor_incremental_delta_for_feature_anchor_candidate",
                "fabricate_data",
                "held_test_induction",
                "runtime_winner_selection",
                "add_mlp_stem_or_readout",
                "add_new_edge_function_family",
            ],
        },
    )
    append_exec("Stage 2 C28-C32 total-audit merge", command_text(), files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}; {rel(nxt)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Stage 2 C28-C32 total-audit reduced", summary)
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
    p.add_argument("--checkpoint-steps", type=int, default=20)
    p.add_argument("--checkpoint-lr", type=float, default=0.02)
    p.add_argument("--quadrature-points", type=int, default=257)
    p.add_argument("--functional-gram-ridge", type=float, default=1.0e-6)
    p.add_argument("--population-beta", type=float, default=1.0)
    p.add_argument("--population-weight-mode", default="sample_coherence")
    p.add_argument("--cg-tol", type=float, default=1.0e-6)
    p.add_argument("--cg-max-iter", type=int, default=512)
    p.add_argument("--part-c-lambda", type=float, default=1.0e-2)
    p.add_argument("--part-d-basis-input-gain", type=float, default=0.25)
    p.add_argument("--part-c-model-seed-scheme-key", default="v23_09_part_c_shared_init")
    p.add_argument("--no-debt-budget", type=float, default=1.0e-8)
    p.add_argument("--stage1-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--stage1-seed-count", type=int, default=5)
    p.add_argument("--stage1-checkpoint", default="h10_isomorphic_partial")
    p.add_argument("--stage1-anchor-steps", type=int, default=5)
    p.add_argument("--stage1-anchor-lr", type=float, default=0.002)
    p.add_argument("--stage1-tail-quantile", type=float, default=0.25)
    p.add_argument("--stage1-anchor-alphas", default="1")
    p.add_argument("--stage1-jacobian-sketch-size", type=int, default=64)
    p.add_argument("--stage1-resume", type=int, default=1)
    p.add_argument("--stage1-flush-every", type=int, default=1)
    p.add_argument("--stage2-schemes", default=",".join(STAGE2_SCHEMES))
    p.add_argument("--stage2-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--stage2-seed-count", type=int, default=5)
    p.add_argument("--stage2-steps", type=int, default=80)
    p.add_argument("--stage2-c15-source-scheme", default="F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between")
    p.add_argument("--stage2-stage1-summary", default="results/v23_12_total_audit_feature_learning_tangent_escape_stage1_projected_lr002_s5/stage1_anchor_feasibility_summary.json")
    p.add_argument("--stage2-residual-target", default="norm")
    p.add_argument("--stage2-lambda", type=float, default=10.0)
    p.add_argument("--stage2-alphas", default="1,0.5,0.25,0.125")
    p.add_argument("--stage2-terminal-alphas", default="1")
    p.add_argument("--stage2-delta-scale", type=float, default=1.0)
    p.add_argument("--stage2-delta-sign", type=float, default=1.0)
    p.add_argument("--stage2-trust-mode", default="pareto")
    p.add_argument("--stage2-trust-tolerance", type=float, default=0.002)
    p.add_argument("--stage2-local-reference-target", default="norm")
    p.add_argument("--stage2-functionalgram-lr", type=float, default=0.0075)
    p.add_argument("--stage2-adam-lr", type=float, default=0.02)
    p.add_argument("--stage2-solver", default="exact")
    p.add_argument("--stage2-sketch-rank", type=int, default=16)
    p.add_argument("--stage2-activation-drift-cap", type=float, default=-1.0)
    p.add_argument("--stage2-resume", type=int, default=1)
    p.add_argument("--stage2-flush-every", type=int, default=1)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode).lower()
    if mode in {"stage0", "stage0-lock", "evidence-lock"}:
        return stage0_evidence_lock(args)
    if mode in {"stage1-run", "anchor-run"}:
        return stage1_run(args)
    if mode in {"stage1-merge", "anchor-merge", "stage1"}:
        return stage1_merge(args)
    if mode in {"stage2-run", "total-audit-run"}:
        return stage2_run(args)
    if mode in {"stage2-merge", "stage2", "total-audit-merge"}:
        return stage2_merge(args)
    raise SystemExit(f"unknown v23.12 mode: {args.mode}")


if __name__ == "__main__":
    main()
