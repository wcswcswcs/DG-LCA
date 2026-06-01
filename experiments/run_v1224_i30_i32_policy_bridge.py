#!/usr/bin/env python
"""v12.24 I30/I32 continuation bridge.

I30 uses an unlabeled train-stream micro-probe to select actuator sign, norm
scale, and compensation mode before committing an update.  I32 adds an
unlabeled policy-compatibility probe before the same P4 role-policy audit.
Neither path reads labels/CE to choose the functional direction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp
import experiments.run_v1223_p4_compensation_modes as p4m
import experiments.run_v1223_p4_trainable_role_scan as role_scan
import experiments.run_v1224_train_stream_functional_bridge as bridge


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def covariance_tail(logits: torch.Tensor) -> float:
    x = logits.detach().float()
    if x.shape[0] < 3:
        return 0.0
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.transpose(0, 1) @ x / max(1, int(x.shape[0]) - 1)
    vals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = vals.sum().clamp_min(1.0e-12)
    if vals.numel() <= 3:
        return float((vals / total).sum().item())
    return float((vals[:-3].sum() / total).item())


def unlabeled_probe_score(
    model: torch.nn.Module,
    refs: list[tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace | None = None,
) -> dict[str, float]:
    drifts = []
    tails = []
    entropies = []
    conf_p95s = []
    logit_abs_p95s = []
    for x_ref, base_logits in refs:
        with torch.no_grad():
            logits = model(x_ref).detach()
            drift = float((logits - base_logits).abs().max().item())
            probs = torch.softmax(logits.float(), dim=-1).clamp_min(1.0e-8)
            entropy = float((-(probs * probs.log()).sum(dim=-1)).mean().item())
            conf = probs.max(dim=-1).values.detach().float()
            logit_abs = logits.detach().float().abs().flatten()
        drifts.append(drift)
        tails.append(covariance_tail(logits))
        entropies.append(entropy)
        conf_p95s.append(float(torch.quantile(conf, 0.95).item()))
        logit_abs_p95s.append(float(torch.quantile(logit_abs, 0.95).item()))
    max_drift = max(drifts) if drifts else float("nan")
    mean_tail = sum(tails) / len(tails) if tails else float("nan")
    mean_entropy = sum(entropies) / len(entropies) if entropies else float("nan")
    mean_conf_p95 = sum(conf_p95s) / len(conf_p95s) if conf_p95s else float("nan")
    mean_logit_abs_p95 = sum(logit_abs_p95s) / len(logit_abs_p95s) if logit_abs_p95s else float("nan")
    mode = str(getattr(args, "probe_score_mode", "default") if args is not None else "default")
    if mode in {"tail_safe", "contrast_tail_safe"}:
        score = (
            -float(getattr(args, "probe_drift_weight", 1.0)) * max_drift
            -float(getattr(args, "probe_tail_weight", 0.10)) * mean_tail
            -float(getattr(args, "probe_confidence_tail_weight", 0.50)) * mean_conf_p95
            -float(getattr(args, "probe_logit_abs_weight", 0.02)) * mean_logit_abs_p95
            +float(getattr(args, "probe_entropy_weight", 0.02)) * mean_entropy
        )
    else:
        score = -max_drift - 0.05 * mean_tail + 0.005 * mean_entropy
    return {
        "probe_score": score,
        "probe_max_logit_drift": max_drift,
        "probe_output_cov_tail": mean_tail,
        "probe_entropy": mean_entropy,
        "probe_confidence_p95": mean_conf_p95,
        "probe_logit_abs_p95": mean_logit_abs_p95,
        "probe_score_mode": mode,
    }



def apply_actuator_with_train_comp(
    base: torch.nn.Module,
    actuator: str,
    budget: float,
    sign: float,
    seed: int,
    opt_delta: dict[str, torch.Tensor],
    refs: list[tuple[torch.Tensor, torch.Tensor]],
    comp_mode: str,
    device: torch.device,
) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    trial = copy.deepcopy(base).to(device)
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(budget) * 100000) + len(actuator) * 17 + (1 if sign > 0 else 2))
    role = exp.apply_v1223_actuator(trial, actuator, float(budget), float(sign), gen, opt_delta)
    comp = {"direct_logit_compensation_applied": 0, "direct_logit_compensation_mode": "", "direct_logit_compensation_pre_drift": "", "direct_logit_compensation_post_drift": "", "direct_logit_compensation_norm": ""}
    if actuator in {"I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl"}:
        comp = bridge.apply_train_stream_compensation(trial, refs, comp_mode)
    return trial, role, comp


def select_i30(base, source, refs, opt_delta, seed: int, args, device) -> dict[str, Any]:
    base_budget = float(source["norm_budget"])
    base_sign = float(source["signed_direction"])
    candidates: list[dict[str, Any]] = []
    for actuator in ["I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease"]:
        for scale in parse_floats(args.i30_scales):
            for sign_mult in [1.0, -1.0]:
                for mode in ["ema", "median", "trimmed_mean"]:
                    budget = base_budget * float(scale)
                    sign = base_sign * sign_mult
                    trial, role, comp = apply_actuator_with_train_comp(base, actuator, budget, sign, seed, opt_delta, refs, mode, device)
                    score = unlabeled_probe_score(trial, refs, args)
                    if str(getattr(args, "probe_score_mode", "default")) == "contrast_tail_safe":
                        control_trial, _control_role, _control_comp = apply_actuator_with_train_comp(
                            base,
                            "TrainDirectLogitCompensatedRandomControl",
                            budget,
                            sign,
                            seed,
                            opt_delta,
                            refs,
                            mode,
                            device,
                        )
                        control_score = unlabeled_probe_score(control_trial, refs, args)
                        source_raw = float(score.get("probe_score", float("nan")))
                        control_raw = float(control_score.get("probe_score", float("nan")))
                        contrast = source_raw - control_raw
                        score["probe_source_raw_score"] = source_raw
                        score["probe_control_raw_score"] = control_raw
                        score["probe_source_control_score_delta"] = contrast
                        score["probe_score"] = source_raw + float(getattr(args, "probe_control_contrast_weight", 0.5)) * contrast
                    candidates.append({"actuator": actuator, "budget": budget, "sign": sign, "comp_mode": mode, "role": role, **comp, **score})
    candidates.sort(key=lambda r: (fnum(r.get("probe_score"), -999.0), -fnum(r.get("probe_max_logit_drift"), 999.0)), reverse=True)
    best = candidates[0]
    best["selection_rank"] = 1
    best["selection_candidates"] = len(candidates)
    return best


def run_policy_probe(model: torch.nn.Module, refs: list[tuple[torch.Tensor, torch.Tensor]], policy: str, steps: int, lr: float, device: torch.device) -> dict[str, Any]:
    params = [p for n, p in model.named_parameters() if p.requires_grad and role_scan.include_param(n, policy)]
    if not params or steps <= 0:
        return {"policy_probe_steps": int(steps), "policy_probe": policy, "policy_probe_loss": "", **unlabeled_probe_score(model, refs)}
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
    losses = []
    for step in range(int(steps)):
        x_ref, base_logits = refs[step % len(refs)]
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x_ref), base_logits)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
    return {"policy_probe_steps": int(steps), "policy_probe": policy, "policy_probe_loss": losses[-1] if losses else "", **unlabeled_probe_score(model, refs)}


def select_i32(base, source, refs, opt_delta, seed: int, args, device) -> dict[str, Any]:
    base_budget = float(source["norm_budget"])
    base_sign = float(source["signed_direction"])
    base_actuator = "I26-TrainDirectLogitCompensatedQuadRelease"
    candidates = []
    for mode in ["ema", "median", "trimmed_mean"]:
        trial0, role, comp = apply_actuator_with_train_comp(base, base_actuator, base_budget, base_sign, seed, opt_delta, refs, mode, device)
        for policy in parse_csv(args.i32_policies):
            for steps in parse_ints(args.i32_probe_steps):
                trial = copy.deepcopy(trial0).to(device)
                probe = run_policy_probe(trial, refs, policy, int(steps), float(args.i32_probe_lr), device)
                candidates.append({"actuator": base_actuator, "budget": base_budget, "sign": base_sign, "comp_mode": mode, "role": role, **comp, **probe})
    candidates.sort(key=lambda r: (fnum(r.get("probe_score"), -999.0), -fnum(r.get("probe_max_logit_drift"), 999.0)), reverse=True)
    best = candidates[0]
    best["selection_rank"] = 1
    best["selection_candidates"] = len(candidates)
    return best


def build_trial_for_selected(base, selected: dict[str, Any], refs, opt_delta, seed: int, device: torch.device) -> tuple[torch.nn.Module, str, dict[str, Any], dict[str, Any]]:
    trial, role, comp = apply_actuator_with_train_comp(
        base,
        str(selected["actuator"]),
        float(selected["budget"]),
        float(selected["sign"]),
        seed,
        opt_delta,
        refs,
        str(selected["comp_mode"]),
        device,
    )
    probe = {}
    if "policy_probe" in selected and selected.get("policy_probe"):
        probe = run_policy_probe(trial, refs, str(selected["policy_probe"]), int(float(selected["policy_probe_steps"])), 1.0e-3, device)
    return trial, role, comp, probe


def run_one(args, source: dict[str, Any], candidate_name: str, train_seed_base: int, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, _base_query_logits, opt_delta, dataset, seed = role_scan.build_context(args, source, device)
    refs = bridge.build_train_refs(base, x_train, int(args.compensation_batch), int(args.ensemble_count))
    if candidate_name == "I30-T1BGuidedDirectCompensation":
        selected = select_i30(base, source, refs, opt_delta, int(seed), args, device)
    elif candidate_name == "I32-PolicyAwareP3QuadProbe":
        selected = select_i32(base, source, refs, opt_delta, int(seed), args, device)
    else:
        raise ValueError(candidate_name)
    rows: list[dict[str, Any]] = []
    rows.append({"stage": "V1224_I30_I32_PRECOMMIT_SELECTION", "candidate_name": candidate_name, "dataset": dataset, "seed": seed, "train_seed_base": int(train_seed_base), "uses_label": 0, "uses_ce_vector": 0, "uses_query_batch": 0, "uses_train_batch": 1, "precommit_available": 1, "promotion_allowed": 0, "no_fake": 1, **selected})
    variants = [
        ("noop", "NoOpMatchedOverhead", 0.0, float(selected["sign"]), 0),
        ("source", str(selected["actuator"]), float(selected["budget"]), float(selected["sign"]), 1),
        ("same_compensation_control", "TrainDirectLogitCompensatedRandomControl", float(selected["budget"]), float(selected["sign"]), 1),
        ("adamw_parallel_control", "AdamWParallelDirection", float(selected["budget"]), float(selected["sign"]), 0),
        ("snr_only_control", "SNR-only", float(selected["budget"]), float(selected["sign"]), 0),
        ("random_norm_control", "RandomMatchedNorm", float(selected["budget"]), float(selected["sign"]), 0),
    ]
    finals: dict[str, dict[str, Any]] = {}
    models: dict[str, torch.nn.Module] = {}
    for idx, (kind, actuator, budget, sign, needs_comp) in enumerate(variants):
        if kind == "source":
            trial, role, comp, probe = build_trial_for_selected(base, selected, refs, opt_delta, int(seed), device)
        else:
            trial = copy.deepcopy(base).to(device)
            gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(budget) * 100000) + len(actuator) * 17 + (1 if sign > 0 else 2))
            role = exp.apply_v1223_actuator(trial, actuator, budget, sign, gen, opt_delta)
            comp = {"direct_logit_compensation_applied": 0, "direct_logit_compensation_mode": "", "direct_logit_compensation_pre_drift": "", "direct_logit_compensation_post_drift": "", "direct_logit_compensation_norm": ""}
            if needs_comp and actuator == "TrainDirectLogitCompensatedRandomControl":
                comp = bridge.apply_train_stream_compensation(trial, refs, str(selected["comp_mode"]))
            probe = {}
        init_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        final_eval, q90, auc = bridge.train_role(
            trial,
            x_train,
            y_train,
            x_val,
            y_val,
            int(args.batch_size),
            int(args.epochs),
            float(args.lr),
            float(args.weight_decay),
            int(seed) + int(train_seed_base) + idx + len(candidate_name),
            device,
            str(args.role_policy),
            float(args.label_smoothing),
        )
        row = {
            "stage": "V1224_I30_I32_BRIDGE_VARIANT",
            "candidate_name": candidate_name,
            "variant_kind": kind,
            "actuator_id": actuator,
            "role": role,
            "dataset": dataset,
            "seed": seed,
            "train_seed_base": int(train_seed_base),
            "norm_budget": budget,
            "signed_direction": sign,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "role_policy": str(args.role_policy),
            "label_smoothing": float(args.label_smoothing),
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "init_acc": init_eval.get("acc", ""),
            "init_NLL": init_eval.get("NLL", ""),
            "init_ECE": init_eval.get("ECE", ""),
            "init_CEp99": init_eval.get("CEp99", ""),
            "final_acc": final_eval.get("acc", ""),
            "final_NLL": final_eval.get("NLL", ""),
            "final_ECE": final_eval.get("ECE", ""),
            "final_CEp99": final_eval.get("CEp99", ""),
            "step_time_q90_ms": q90,
            "auc_error_time": auc,
            "uses_label": int(actuator == "AdamWParallelDirection"),
            "uses_ce_vector": int(actuator == "AdamWParallelDirection"),
            "uses_query_batch": 0,
            "uses_train_batch": int(needs_comp),
            "precommit_available": int(actuator != "AdamWParallelDirection"),
            "loss_agnostic_direction": int(actuator != "AdamWParallelDirection"),
            "promotion_allowed": 0,
            "no_fake": 1,
            **comp,
            **{f"selected_{k}": v for k, v in selected.items() if k in {"actuator", "budget", "sign", "comp_mode", "probe_score", "probe_max_logit_drift", "policy_probe", "policy_probe_steps"}},
            **probe,
        }
        rows.append(row)
        finals[kind] = row
        models[kind] = trial
    linec_rows, linec_pass_count, linec_all_pass = bridge.linec_seed_summary({"noop": models["noop"], "source": models["source"]}, xb, yb, xq, yq, int(seed), parse_ints(args.linec_seeds), int(args.linec_sketch_dim))
    for linec_row in linec_rows:
        rows.append({**linec_row, "candidate_name": candidate_name, "dataset": dataset, "seed": seed, "train_seed_base": int(train_seed_base), "promotion_allowed": 0, "no_fake": 1})
    src = finals["source"]
    noop = finals["noop"]
    controls = [finals[k] for k in ["same_compensation_control", "adamw_parallel_control", "snr_only_control", "random_norm_control"]]
    src_acc = fnum(src.get("final_acc"))
    noop_acc = fnum(noop.get("final_acc"))
    control_acc = max(fnum(c.get("final_acc")) for c in controls)
    task_pass = int(
        math.isfinite(src_acc)
        and src_acc >= max(noop_acc, control_acc) + 0.005
        and fnum(src.get("final_NLL")) <= fnum(noop.get("final_NLL"))
        and fnum(src.get("final_CEp99")) <= fnum(noop.get("final_CEp99")) + 0.05
        and fnum(src.get("final_ECE")) <= fnum(noop.get("final_ECE")) + 0.02
        and fnum(src.get("auc_error_time")) <= fnum(noop.get("auc_error_time"))
        and fnum(src.get("step_time_q90_ms")) <= fnum(noop.get("step_time_q90_ms")) * 1.05
    )
    linec_majority_pass = int(linec_pass_count >= math.ceil(len(parse_ints(args.linec_seeds)) / 2.0))
    summary = {
        "stage": "V1224_I30_I32_BRIDGE_SUMMARY",
        "candidate_name": candidate_name,
        "dataset": dataset,
        "seed": seed,
        "train_seed_base": int(train_seed_base),
        "source_final_acc": src_acc,
        "noop_final_acc": noop_acc,
        "best_control_final_acc": control_acc,
        "source_vs_noop_acc_delta": src_acc - noop_acc,
        "source_vs_best_control_acc_delta": src_acc - control_acc,
        "task_gate_pass": task_pass,
        "linec_seed_count": len(parse_ints(args.linec_seeds)),
        "linec_seed_pass_count": linec_pass_count,
        "linec_majority_pass": linec_majority_pass,
        "linec_all_pass": linec_all_pass,
        "strict_majority_pass": int(task_pass and linec_majority_pass),
        "strict_all_pass": int(task_pass and linec_all_pass),
        "selected_actuator": selected.get("actuator", ""),
        "selected_budget": selected.get("budget", ""),
        "selected_sign": selected.get("sign", ""),
        "selected_comp_mode": selected.get("comp_mode", ""),
        "selected_probe_score": selected.get("probe_score", ""),
        "uses_query_batch": 0,
        "uses_train_batch": 1,
        "precommit_available": 1,
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    rows.append(summary)
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-out-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--role-policy", default="quad_only")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--compensation-batch", type=int, default=32)
    ap.add_argument("--ensemble-count", type=int, default=8)
    ap.add_argument("--train-seed-bases", default="12240400,12241400,12242400")
    ap.add_argument("--linec-seeds", default="12239500,12240600,12241600,12242600,12243600")
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--candidates", default="I30-T1BGuidedDirectCompensation,I32-PolicyAwareP3QuadProbe")
    ap.add_argument("--i30-scales", default="0.5,1.0,1.5")
    ap.add_argument("--i32-policies", default="quad_only,quad_direct,direct_only")
    ap.add_argument("--i32-probe-steps", default="1,3,5")
    ap.add_argument("--i32-probe-lr", type=float, default=0.001)
    ap.add_argument("--probe-score-mode", default="default", choices=["default", "tail_safe", "contrast_tail_safe"])
    ap.add_argument("--probe-drift-weight", type=float, default=1.0)
    ap.add_argument("--probe-tail-weight", type=float, default=0.10)
    ap.add_argument("--probe-confidence-tail-weight", type=float, default=0.50)
    ap.add_argument("--probe-logit-abs-weight", type=float, default=0.02)
    ap.add_argument("--probe-entropy-weight", type=float, default=0.02)
    ap.add_argument("--probe-control-contrast-weight", type=float, default=0.5)
    ap.add_argument("--artifact-prefix", default="v1224_i30_i32_policy_bridge")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    exp.ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.24 I30/I32 bridge requires CUDA")
    torch.cuda.set_device(device)
    source = p4m.pick_source(Path(args.source_out_dir))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for candidate in parse_csv(args.candidates):
        for train_seed_base in parse_ints(args.train_seed_bases):
            rr, ss = run_one(args, source, candidate, train_seed_base, device)
            rows.extend(rr)
            summaries.append(ss)
            torch.cuda.empty_cache()
    aggregate_rows = []
    for candidate in sorted(set(str(s["candidate_name"]) for s in summaries)):
        group = [s for s in summaries if str(s["candidate_name"]) == candidate]
        strict_majority = sum(exp.safe_int(r.get("strict_majority_pass"), 0) for r in group)
        strict_all = sum(exp.safe_int(r.get("strict_all_pass"), 0) for r in group)
        aggregate_rows.append(
            {
                "stage": "V1224_I30_I32_BRIDGE_CANDIDATE_AGGREGATE",
                "candidate_name": candidate,
                "train_shuffle_rows": len(group),
                "train_shuffle_strict_majority_pass_count": strict_majority,
                "train_shuffle_strict_all_pass_count": strict_all,
                "train_shuffle_robust_majority_pass": int(strict_majority >= 2),
                "train_shuffle_robust_all_pass": int(strict_all >= 2),
                "best_source_vs_noop_acc_delta": max(fnum(r.get("source_vs_noop_acc_delta"), -999.0) for r in group),
                "best_source_vs_control_acc_delta": max(fnum(r.get("source_vs_best_control_acc_delta"), -999.0) for r in group),
                "uses_query_batch": 0,
                "uses_train_batch": 1,
                "precommit_available": 1,
                "promotion_allowed": 0,
                "no_fake": 1,
            }
        )
    rows.extend(aggregate_rows)
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    json_path = out_dir / f"{args.artifact_prefix}_summary.json"
    exp.write_csv_rows(csv_path, rows)
    result = {
        "stage": "V1224_I30_I32_BRIDGE_AGGREGATE",
        "artifact_csv": exp.rel(csv_path),
        "source_out_dir": exp.rel(args.source_out_dir),
        "source_dataset": source.get("dataset", ""),
        "source_seed": source.get("seed", ""),
        "source_actuator": source.get("actuator_id", ""),
        "source_norm_budget": source.get("norm_budget", ""),
        "source_signed_direction": source.get("signed_direction", ""),
        "candidate_rows": len(summaries),
        "aggregate_rows": len(aggregate_rows),
        "any_strict_majority_pass": int(any(exp.safe_int(r.get("strict_majority_pass"), 0) for r in summaries)),
        "any_strict_all_pass": int(any(exp.safe_int(r.get("strict_all_pass"), 0) for r in summaries)),
        "any_train_shuffle_robust_majority_pass": int(any(exp.safe_int(r.get("train_shuffle_robust_majority_pass"), 0) for r in aggregate_rows)),
        "any_train_shuffle_robust_all_pass": int(any(exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0) for r in aggregate_rows)),
        "best_candidate_aggregates": sorted(
            aggregate_rows,
            key=lambda r: (
                exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0),
                exp.safe_int(r.get("train_shuffle_robust_majority_pass"), 0),
                fnum(r.get("best_source_vs_noop_acc_delta"), -999.0),
                fnum(r.get("best_source_vs_control_acc_delta"), -999.0),
            ),
            reverse=True,
        ),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    exp.write_json(json_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
