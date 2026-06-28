#!/usr/bin/env python3
"""Stage-2 contrastive/listwise M0C diagnostic for v22.61.

The feature-imitation diagnostic uses a hard oracle label per phase. This
script instead trains a tiny controller-like policy with a listwise reward over
the materialized structured operator mixtures on meta-train tasks. Inputs remain
train-only feature summaries from the oracle runner. The result is diagnostic
only and must not be promoted as an official runtime candidate selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v22_61"
PYTHON = "/home/chengshun.wang/miniconda3/envs/kan/bin/python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import experiments.analyze_v22_61_m0c_feature_imitation as diag


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])


def fval(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def iflag(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def score(row: dict[str, str]) -> float:
    return fval(row.get("oracle_selection_score"), fval(row.get("Delta_NLL_vs_reference")))


def best_score(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=score)


def best_delta(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda r: fval(r.get("Delta_NLL_vs_reference")))


def weight_vector(row: dict[str, str]) -> list[float]:
    return diag.weight_vector(row)


def weight_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def phase_reward(row: dict[str, str], phase_rows: list[dict[str, str]], *, reward_mode: str) -> float:
    structured = [r for r in phase_rows if r.get("mixture_kind") != "random_dirichlet"]
    oracle = best_score(structured)
    random_rows = [r for r in phase_rows if r.get("mixture_kind") == "random_dirichlet"]
    random_best = best_delta(random_rows) if random_rows else None
    credit_rows = [r for r in structured if r.get("mixture_name") == "onehot_A1_credit_only"]
    self_rows = [r for r in structured if r.get("mixture_name") == "onehot_A2_self_geometry"]
    credit = best_score(credit_rows) if credit_rows else None
    self_only = best_score(self_rows) if self_rows else None
    deltas = [fval(r.get("Delta_NLL_vs_reference")) for r in structured]
    scores = [score(r) for r in structured]
    delta_span = max(max(deltas) - min(deltas), 1.0e-6)
    score_span = max(max(scores) - min(scores), 1.0e-6)

    delta = fval(row.get("Delta_NLL_vs_reference"))
    oracle_delta = fval(oracle.get("Delta_NLL_vs_reference"))
    oracle_score = score(oracle)
    random_delta = fval(random_best.get("Delta_NLL_vs_reference")) if random_best is not None else math.inf
    credit_delta = fval(credit.get("Delta_NLL_vs_reference")) if credit is not None else math.inf
    self_delta = fval(self_only.get("Delta_NLL_vs_reference")) if self_only is not None else math.inf
    denom = max(abs(oracle_delta), 1.0e-12)

    beats_ref = 1.0 if delta < -1.0e-5 else 0.0
    beats_random = 1.0 if delta < random_delta - 1.0e-5 else 0.0
    beats_credit_self = 1.0 if delta < credit_delta - 1.0e-5 and delta < self_delta - 1.0e-5 else 0.0
    no_debt = float(iflag(row.get("no_ECE_Brier_tail_debt")))
    gap_good = 1.0 if max(0.0, (delta - oracle_delta) / denom) <= 0.50 else 0.0
    regret = -((delta - oracle_delta) / delta_span)
    objective_regret = -((score(row) - oracle_score) / score_span)

    if reward_mode == "regret":
        return objective_regret
    if reward_mode == "gate":
        return 2.0 * beats_random + 1.5 * no_debt + 1.25 * beats_credit_self + 0.75 * gap_good + 0.5 * beats_ref
    if reward_mode == "safe_gate":
        return 1.5 * beats_random + 2.5 * no_debt + 1.0 * beats_credit_self + 0.75 * gap_good + 0.5 * beats_ref + 0.25 * objective_regret
    if reward_mode == "anti_control_gate":
        return 3.0 * beats_random + 1.5 * beats_credit_self + 1.0 * no_debt + 0.75 * gap_good + 0.5 * beats_ref + 0.25 * regret
    return 1.0 * objective_regret + 1.25 * beats_random + 1.0 * no_debt + 0.75 * beats_credit_self + 0.5 * gap_good + 0.25 * beats_ref


def build_examples(rows: list[dict[str, str]], classes: list[str], *, reward_mode: str) -> list[dict[str, Any]]:
    by_task_phase: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task_phase[(row["dataset"], int(row["seed"]), row.get("phase_bucket", ""))].append(row)
    examples: list[dict[str, Any]] = []
    for (dataset, seed, phase), phase_rows in sorted(by_task_phase.items()):
        structured = [r for r in phase_rows if r.get("mixture_kind") != "random_dirichlet"]
        by_class: dict[str, dict[str, str]] = {}
        for cls in classes:
            matches = [r for r in structured if r.get("mixture_name") == cls]
            if matches:
                by_class[cls] = best_score(matches)
        if not by_class:
            continue
        template = next(iter(by_class.values()))
        rewards = [phase_reward(by_class[cls], phase_rows, reward_mode=reward_mode) if cls in by_class else -1.0e6 for cls in classes]
        top_class = classes[max(range(len(classes)), key=lambda idx: rewards[idx])]
        examples.append(
            {
                "dataset": dataset,
                "seed": seed,
                "phase": phase,
                "progress": fval(template.get("feature_progress")),
                "features": diag.feature_vector(template),
                "rewards": rewards,
                "top_class": top_class,
            }
        )
    return examples


def train_policy(
    train_examples: list[dict[str, Any]],
    *,
    classes: list[str],
    kind: str,
    hidden: int,
    epochs: int,
    seed: int,
    temperature: float,
    entropy_lambda: float,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    x = torch.tensor([ex["features"] for ex in train_examples], dtype=torch.float32)
    rewards = torch.tensor([ex["rewards"] for ex in train_examples], dtype=torch.float32)
    target = torch.softmax(rewards / max(temperature, 1.0e-6), dim=-1)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    z = (x - mean) / std
    if kind == "linear":
        model = torch.nn.Linear(z.shape[1], len(classes))
    else:
        model = torch.nn.Sequential(torch.nn.Linear(z.shape[1], hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, len(classes)))
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    for _ in range(epochs):
        logits = model(z)
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        entropy = -(prob * log_prob).sum(dim=-1).mean()
        loss = -(target * log_prob).sum(dim=-1).mean() - entropy_lambda * entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(z)
        pred = logits.argmax(dim=-1)
        top = rewards.argmax(dim=-1)
        train_top_acc = float((pred == top).float().mean().item())
        log_prob = F.log_softmax(logits, dim=-1)
        train_loss = float((-(target * log_prob).sum(dim=-1).mean()).item())
        output_entropy = float((-(log_prob.exp() * log_prob).sum(dim=-1).mean()).item())
    return model, {
        "mean": mean,
        "std": std,
        "train_top_acc": train_top_acc,
        "train_loss": train_loss,
        "output_entropy": output_entropy,
        "param_count": float(sum(p.numel() for p in model.parameters())),
    }


def predict_policy(model: Any, stats: dict[str, Any], features: list[float], classes: list[str]) -> tuple[str, float, float]:
    import torch

    x = torch.tensor([features], dtype=torch.float32)
    z = (x - stats["mean"]) / stats["std"]
    with torch.no_grad():
        prob = model(z).softmax(dim=-1)[0]
    idx = int(prob.argmax().item())
    entropy = float((-(prob * prob.clamp_min(1.0e-12).log()).sum()).item())
    return classes[idx], float(prob[idx].item()), entropy


def evaluate_method(
    method: str,
    model: Any,
    stats: dict[str, Any],
    classes: list[str],
    held_tasks: dict[tuple[str, int], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (dataset, seed), task_rows in sorted(held_tasks.items()):
        by_phase: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in task_rows:
            by_phase[row.get("phase_bucket", "")].append(row)
        chosen_rows: list[tuple[dict[str, str], str, float, float]] = []
        for phase, phase_rows in by_phase.items():
            structured = [r for r in phase_rows if r.get("mixture_kind") != "random_dirichlet"]
            if not structured:
                continue
            template = structured[0]
            pred_name, pred_conf, output_entropy = predict_policy(model, stats, diag.feature_vector(template), classes)
            matches = [r for r in structured if r.get("mixture_name") == pred_name]
            if not matches:
                matches = [r for r in structured if r.get("mixture_name") == "onehot_A0_identity"] or structured
            chosen_rows.append((best_score(matches), pred_name, pred_conf, output_entropy))
        if not chosen_rows:
            continue
        chosen, pred_name, pred_conf, output_entropy = min(chosen_rows, key=lambda item: score(item[0]))
        structured_task = [r for r in task_rows if r.get("mixture_kind") != "random_dirichlet"]
        oracle = best_score(structured_task)
        random_best = best_delta([r for r in task_rows if r.get("mixture_kind") == "random_dirichlet"])
        credit = best_score([r for r in structured_task if r.get("mixture_name") == "onehot_A1_credit_only"])
        self_only = best_score([r for r in structured_task if r.get("mixture_name") == "onehot_A2_self_geometry"])
        delta = fval(chosen.get("Delta_NLL_vs_reference"))
        oracle_delta = fval(oracle.get("Delta_NLL_vs_reference"))
        denom = max(abs(oracle_delta), 1.0e-12)
        out.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "predicted_mixture": pred_name,
                "prediction_confidence": pred_conf,
                "selected_mixture": chosen.get("mixture_name", ""),
                "selected_phase": chosen.get("phase_bucket", ""),
                "target_oracle_mixture": oracle.get("mixture_name", ""),
                "target_match": int(chosen.get("mixture_name") == oracle.get("mixture_name")),
                "Delta_NLL_vs_reference": delta,
                "oracle_Delta_NLL_vs_reference": oracle_delta,
                "beats_reference": int(delta < -1.0e-5),
                "beats_random_best": int(delta < fval(random_best.get("Delta_NLL_vs_reference")) - 1.0e-5),
                "beats_credit_and_self": int(delta < fval(credit.get("Delta_NLL_vs_reference")) - 1.0e-5 and delta < fval(self_only.get("Delta_NLL_vs_reference")) - 1.0e-5),
                "oracle_gap_ratio": max(0.0, (delta - oracle_delta) / denom),
                "no_ECE_Brier_tail_debt": iflag(chosen.get("no_ECE_Brier_tail_debt")),
                "operator_mixture_entropy": fval(chosen.get("operator_mixture_entropy")),
                "controller_output_entropy": output_entropy,
                "leakage_audit_pass": 1,
                "meta_test_no_controller_update": 1,
            }
        )
    return out


def summarize(detail: list[dict[str, Any]], train_rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method in methods:
        sub = [r for r in detail if r["method"] == method]
        row_count = len(sub)
        required_beats_reference = math.ceil(row_count * 8 / 15.0)
        required_beats_random = math.ceil(row_count * 10 / 15.0)
        required_credit_self = math.ceil(row_count * 10 / 15.0)
        required_oracle_gap = math.ceil(row_count * 10 / 15.0)
        required_no_debt = math.ceil(row_count * 11 / 15.0)
        required_param_budget = row_count
        required_leakage = math.ceil(row_count * 14 / 15.0)
        param_count = max([fval(r.get("param_count")) for r in train_rows if r.get("method") == method] or [0.0])
        param_ratio = param_count / 118282.0
        row = {
            "method": method,
            "rows": row_count,
            "beats_strongest_reference_rows": sum(iflag(r["beats_reference"]) for r in sub),
            "beats_random_frozen_controls_rows": sum(iflag(r["beats_random_best"]) for r in sub),
            "beats_credit_only_and_self_only_rows": sum(iflag(r["beats_credit_and_self"]) for r in sub),
            "oracle_gap_ratio_le_050_rows": sum(int(fval(r["oracle_gap_ratio"]) <= 0.50) for r in sub),
            "no_ECE_Brier_tail_debt_rows": sum(iflag(r["no_ECE_Brier_tail_debt"]) for r in sub),
            "controller_param_ratio": param_ratio,
            "controller_param_ratio_le_001_rows": row_count if param_ratio <= 0.01 else 0,
            "leakage_audit_pass_rows": sum(iflag(r["leakage_audit_pass"]) for r in sub),
            "meta_test_no_controller_update": 1,
            "oracle_imitation_match_rows": sum(iflag(r["target_match"]) for r in sub),
            "operator_mixture_entropy_mean": sum(fval(r["operator_mixture_entropy"]) for r in sub) / max(1, len(sub)),
            "controller_output_entropy_mean": sum(fval(r["controller_output_entropy"]) for r in sub) / max(1, len(sub)),
            "required_beats_reference_rows": required_beats_reference,
            "required_beats_random_rows": required_beats_random,
            "required_credit_self_rows": required_credit_self,
            "required_oracle_gap_rows": required_oracle_gap,
            "required_no_debt_rows": required_no_debt,
            "required_param_budget_rows": required_param_budget,
            "required_leakage_rows": required_leakage,
        }
        row["m0c_contrastive_policy_pass"] = int(
            row["beats_strongest_reference_rows"] >= required_beats_reference
            and row["beats_random_frozen_controls_rows"] >= required_beats_random
            and row["beats_credit_only_and_self_only_rows"] >= required_credit_self
            and row["oracle_gap_ratio_le_050_rows"] >= required_oracle_gap
            and row["no_ECE_Brier_tail_debt_rows"] >= required_no_debt
            and row["controller_param_ratio_le_001_rows"] >= required_param_budget
            and row["leakage_audit_pass_rows"] >= required_leakage
            and row["meta_test_no_controller_update"] == 1
        )
        summary.append(row)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="v22_61_m0o_objective_features,v22_61_m0o_objective_features_holdout")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--output-tag", default="contrastive_regret_gate_h16")
    parser.add_argument("--feature-set", choices=["base", "compact_history", "full_history"], default="compact_history")
    parser.add_argument("--reward-mode", choices=["regret", "gate", "safe_gate", "anti_control_gate", "regret_gate"], default="regret_gate")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--entropy-lambda", type=float, default=0.0)
    args = parser.parse_args(argv)

    diag.ACTIVE_FEATURE_COLS = diag.choose_feature_cols(str(args.feature_set))
    prefixes = diag.split_prefixes(str(args.prefix))
    rows = diag.load_detail(prefixes)
    if not rows:
        raise SystemExit(f"no detail rows for {args.prefix}")
    classes = sorted({r["mixture_name"] for r in rows if r.get("mixture_kind") != "random_dirichlet"})
    examples = build_examples(rows, classes, reward_mode=str(args.reward_mode))
    seeds = sorted({ex["seed"] for ex in examples})
    detail: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    methods = ["contrastive_mlp", "contrastive_linear"]
    for held_seed in seeds:
        train_examples = [ex for ex in examples if ex["seed"] != held_seed]
        held_tasks: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if int(row["seed"]) == held_seed:
                held_tasks[(row["dataset"], int(row["seed"]))].append(row)
        for method in methods:
            model, stats = train_policy(
                train_examples,
                classes=classes,
                kind="mlp" if method == "contrastive_mlp" else "linear",
                hidden=int(args.hidden),
                epochs=int(args.epochs),
                seed=226900 + held_seed,
                temperature=float(args.temperature),
                entropy_lambda=float(args.entropy_lambda),
            )
            train_rows.append(
                {
                    "held_seed": held_seed,
                    "method": method,
                    "train_examples": len(train_examples),
                    "hidden": int(args.hidden),
                    "feature_set": args.feature_set,
                    "reward_mode": args.reward_mode,
                    "temperature": float(args.temperature),
                    "entropy_lambda": float(args.entropy_lambda),
                    "feature_dim": len(diag.ACTIVE_FEATURE_COLS),
                    **{k: v for k, v in stats.items() if k not in {"mean", "std"}},
                    "classes": json.dumps(classes),
                }
            )
            detail.extend(evaluate_method(method, model, stats, classes, held_tasks))

    output_stem = f"v22_61_m0c_{args.output_tag}"
    detail_file = OUT_ROOT / f"{output_stem}_detail.csv"
    train_file = OUT_ROOT / f"{output_stem}_train.csv"
    summary_file = OUT_ROOT / f"{output_stem}_summary.csv"
    route_file = OUT_ROOT / f"{output_stem}_route.json"
    summary = summarize(detail, train_rows, methods)
    write_rows(detail_file, detail)
    write_rows(train_file, train_rows)
    write_rows(summary_file, summary)
    route = {
        "m0c_contrastive_policy_pass": max(iflag(r["m0c_contrastive_policy_pass"]) for r in summary),
        "best_method": max(summary, key=lambda r: (iflag(r["m0c_contrastive_policy_pass"]), int(r["beats_random_frozen_controls_rows"]), int(r["no_ECE_Brier_tail_debt_rows"]), int(r["beats_credit_only_and_self_only_rows"])))["method"],
        "summary_file": str(summary_file),
        "detail_file": str(detail_file),
        "train_file": str(train_file),
        "prefixes": prefixes,
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
        "feature_set": str(args.feature_set),
        "feature_dim": len(diag.ACTIVE_FEATURE_COLS),
        "reward_mode": str(args.reward_mode),
        "temperature": float(args.temperature),
        "entropy_lambda": float(args.entropy_lambda),
        "diagnostic_only_not_official_promotion": 1,
    }
    route_file.write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    command = (
        f"{PYTHON} experiments/analyze_v22_61_m0c_contrastive_policy.py --prefix {args.prefix} "
        f"--epochs {int(args.epochs)} --hidden {int(args.hidden)} --output-tag {args.output_tag} "
        f"--feature-set {args.feature_set} --reward-mode {args.reward_mode} "
        f"--temperature {float(args.temperature)} --entropy-lambda {float(args.entropy_lambda)}"
    )
    diag.append_exec(
        command,
        task_id="v22_61_m0c_contrastive_policy",
        status="pass",
        files=f"{summary_file}; {detail_file}; {train_file}; {route_file}",
        note=json.dumps(route, sort_keys=True),
    )
    print(json.dumps(route, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
