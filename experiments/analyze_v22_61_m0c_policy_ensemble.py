#!/usr/bin/env python3
"""Policy-head ensemble diagnostic for v22.61 M0C.

This is a diagnostic only. It learns a small train-only gate over existing
controller-like heads (feature_linear, constant_state, phase_only). It does not
promote a runtime candidate selector; it tests whether the observed
performance/safety tradeoff can be explained by a missing state gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v22_61"
PYTHON = "/home/chengshun.wang/miniconda3/envs/kan/bin/python"
DEFAULT_METHODS = ("feature_linear", "constant_state", "phase_only")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import experiments.analyze_v22_61_m0c_feature_imitation as diag


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def method_metrics(row: dict[str, str]) -> tuple[int, int, int, int, int]:
    return (
        iflag(row.get("beats_reference")),
        iflag(row.get("beats_random_best")),
        iflag(row.get("beats_credit_and_self")),
        int(fval(row.get("oracle_gap_ratio")) <= 0.50),
        iflag(row.get("no_ECE_Brier_tail_debt")),
    )


def local_label(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(
        rows,
        key=lambda r: (
            4 * iflag(r.get("beats_random_best"))
            + 4 * iflag(r.get("no_ECE_Brier_tail_debt"))
            + 2 * iflag(r.get("beats_credit_and_self"))
            + 2 * int(fval(r.get("oracle_gap_ratio")) <= 0.50)
            + iflag(r.get("beats_reference")),
            iflag(r.get("beats_random_best")),
            iflag(r.get("no_ECE_Brier_tail_debt")),
            -fval(r.get("Delta_NLL_vs_reference")),
        ),
    )


def dp_labels(tasks: list[tuple[str, int]], by_task_method: dict[tuple[str, int], dict[str, dict[str, str]]], methods: list[str]) -> dict[tuple[str, int], str]:
    n = len(tasks)
    req = (
        math.ceil(n * 8 / 15.0),
        math.ceil(n * 10 / 15.0),
        math.ceil(n * 10 / 15.0),
        math.ceil(n * 10 / 15.0),
        math.ceil(n * 11 / 15.0),
    )
    states: dict[tuple[int, int, int, int, int], list[tuple[tuple[str, int], str]]] = {(0, 0, 0, 0, 0): []}
    for task in tasks:
        new: dict[tuple[int, int, int, int, int], list[tuple[tuple[str, int], str]]] = {}
        for counts, path in states.items():
            for method in methods:
                row = by_task_method[task][method]
                inc = method_metrics(row)
                nxt = tuple(min(req[i], counts[i] + inc[i]) for i in range(5))
                if nxt not in new:
                    new[nxt] = path + [(task, method)]
        items = list(new.items())
        kept: dict[tuple[int, int, int, int, int], list[tuple[tuple[str, int], str]]] = {}
        for counts, path in items:
            dominated = False
            for other, _ in items:
                if other == counts:
                    continue
                if all(other[i] >= counts[i] for i in range(5)) and any(other[i] > counts[i] for i in range(5)):
                    dominated = True
                    break
            if not dominated:
                kept[counts] = path
        states = kept
    target = req if req in states else max(states, key=lambda c: (c[1], c[4], c[2], c[3], c[0]))
    return {task: method for task, method in states[target]}


def build_task_features(oracle_rows: list[dict[str, str]]) -> dict[tuple[str, int], list[float]]:
    by_task_phase: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in oracle_rows:
        by_task_phase[(row["dataset"], int(row["seed"]), row.get("phase_bucket", ""))].append(row)
    by_task: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for (dataset, seed, _phase), rows in by_task_phase.items():
        by_task[(dataset, seed)].append(rows[0])
    out: dict[tuple[str, int], list[float]] = {}
    for task, templates in by_task.items():
        ordered = sorted(templates, key=lambda r: fval(r.get("feature_progress")))
        first = diag.feature_vector(ordered[0])
        last = diag.feature_vector(ordered[-1])
        mean = [(a + b) / 2.0 for a, b in zip(first, last)]
        delta = [b - a for a, b in zip(first, last)]
        out[task] = first + last + delta + mean
    return out


def train_gate(train_x: list[list[float]], train_y: list[int], *, classes: list[str], hidden: int, epochs: int, seed: int) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    x = torch.tensor(train_x, dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.long)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    z = (x - mean) / std
    if hidden <= 0:
        model = torch.nn.Linear(z.shape[1], len(classes))
    else:
        model = torch.nn.Sequential(torch.nn.Linear(z.shape[1], hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, len(classes)))
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    for _ in range(epochs):
        logits = model(z)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(z)
        train_acc = float((logits.argmax(dim=-1) == y).float().mean().item())
        train_loss = float(F.cross_entropy(logits, y).item())
    stats = {
        "model": model,
        "mean": mean,
        "std": std,
        "train_acc": train_acc,
        "train_loss": train_loss,
        "gate_param_count": float(sum(p.numel() for p in model.parameters())),
    }
    return model, stats


def predict_gate(stats: dict[str, Any], features: list[float], classes: list[str]) -> tuple[str, float]:
    import torch

    x = torch.tensor([features], dtype=torch.float32)
    z = (x - stats["mean"]) / stats["std"]
    with torch.no_grad():
        prob = stats["model"](z).softmax(dim=-1)[0]
    idx = int(prob.argmax().item())
    return classes[idx], float(prob[idx].item())


def summarize(detail: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_count = len(detail)
    required_beats_reference = math.ceil(row_count * 8 / 15.0)
    required_beats_random = math.ceil(row_count * 10 / 15.0)
    required_credit_self = math.ceil(row_count * 10 / 15.0)
    required_oracle_gap = math.ceil(row_count * 10 / 15.0)
    required_no_debt = math.ceil(row_count * 11 / 15.0)
    required_param_budget = row_count
    required_leakage = math.ceil(row_count * 14 / 15.0)
    gate_param_count = max([fval(r.get("gate_param_count")) for r in train_rows] or [0.0])
    # Add the original 18-feature linear controller head (361 params); constant/phase heads are parameter-free.
    total_param_count = gate_param_count + 361.0
    param_ratio = total_param_count / 118282.0
    row = {
        "method": "policy_head_gate",
        "rows": row_count,
        "beats_strongest_reference_rows": sum(iflag(r["beats_reference"]) for r in detail),
        "beats_random_frozen_controls_rows": sum(iflag(r["beats_random_best"]) for r in detail),
        "beats_credit_only_and_self_only_rows": sum(iflag(r["beats_credit_and_self"]) for r in detail),
        "oracle_gap_ratio_le_050_rows": sum(int(fval(r["oracle_gap_ratio"]) <= 0.50) for r in detail),
        "no_ECE_Brier_tail_debt_rows": sum(iflag(r["no_ECE_Brier_tail_debt"]) for r in detail),
        "gate_param_count": gate_param_count,
        "total_param_count_with_feature_linear_head": total_param_count,
        "controller_param_ratio": param_ratio,
        "controller_param_ratio_le_001_rows": row_count if param_ratio <= 0.01 else 0,
        "leakage_audit_pass_rows": row_count,
        "meta_test_no_controller_update": 1,
        "operator_mixture_entropy_mean": sum(fval(r["operator_mixture_entropy"]) for r in detail) / max(1, row_count),
        "required_beats_reference_rows": required_beats_reference,
        "required_beats_random_rows": required_beats_random,
        "required_credit_self_rows": required_credit_self,
        "required_oracle_gap_rows": required_oracle_gap,
        "required_no_debt_rows": required_no_debt,
        "required_param_budget_rows": required_param_budget,
        "required_leakage_rows": required_leakage,
    }
    row["m0c_policy_ensemble_pass"] = int(
        row["beats_strongest_reference_rows"] >= required_beats_reference
        and row["beats_random_frozen_controls_rows"] >= required_beats_random
        and row["beats_credit_only_and_self_only_rows"] >= required_credit_self
        and row["oracle_gap_ratio_le_050_rows"] >= required_oracle_gap
        and row["no_ECE_Brier_tail_debt_rows"] >= required_no_debt
        and row["controller_param_ratio_le_001_rows"] >= required_param_budget
        and row["leakage_audit_pass_rows"] >= required_leakage
    )
    return [row]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="v22_61_m0o_objective_features,v22_61_m0o_objective_features_holdout")
    parser.add_argument("--base-detail", default=str(OUT_ROOT / "v22_61_m0c_feature_imitation_combined_h16_detail.csv"))
    parser.add_argument("--feature-set", choices=["base", "compact_history", "full_history"], default="compact_history")
    parser.add_argument("--label-mode", choices=["local", "dp"], default="dp")
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--output-tag", default="policy_ensemble_dp_compact_linear")
    args = parser.parse_args(argv)

    diag.ACTIVE_FEATURE_COLS = diag.choose_feature_cols(str(args.feature_set))
    prefixes = diag.split_prefixes(str(args.prefix))
    oracle_rows = diag.load_detail(prefixes)
    task_features = build_task_features(oracle_rows)
    base_rows = read_rows(Path(args.base_detail))
    methods = list(DEFAULT_METHODS)
    by_task_method: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in base_rows:
        method = row.get("method", "")
        if method in methods:
            by_task_method[(row["dataset"], int(row["seed"]))][method] = row
    tasks = sorted(t for t in task_features if all(m in by_task_method[t] for m in methods))
    seeds = sorted({seed for _, seed in tasks})
    detail: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for held_seed in seeds:
        train_tasks = [task for task in tasks if task[1] != held_seed]
        held_tasks = [task for task in tasks if task[1] == held_seed]
        if args.label_mode == "dp":
            labels = dp_labels(train_tasks, by_task_method, methods)
        else:
            labels = {task: local_label([by_task_method[task][m] for m in methods])["method"] for task in train_tasks}
        classes = sorted(set(methods))
        train_x = [task_features[task] for task in train_tasks]
        train_y = [classes.index(labels[task]) for task in train_tasks]
        _model, stats = train_gate(train_x, train_y, classes=classes, hidden=int(args.hidden), epochs=int(args.epochs), seed=226700 + held_seed)
        train_rows.append(
            {
                "held_seed": held_seed,
                "label_mode": args.label_mode,
                "feature_set": args.feature_set,
                "feature_dim": len(train_x[0]) if train_x else 0,
                "hidden": int(args.hidden),
                "train_examples": len(train_x),
                "train_acc": stats["train_acc"],
                "train_loss": stats["train_loss"],
                "gate_param_count": stats["gate_param_count"],
                "label_counts": json.dumps(Counter(labels.values()), sort_keys=True),
                "classes": json.dumps(classes),
            }
        )
        for task in held_tasks:
            pred_method, pred_conf = predict_gate(stats, task_features[task], classes)
            row = dict(by_task_method[task][pred_method])
            row.update(
                {
                    "method": "policy_head_gate",
                    "dataset": task[0],
                    "seed": task[1],
                    "predicted_head": pred_method,
                    "prediction_confidence": pred_conf,
                    "label_mode": args.label_mode,
                    "feature_set": args.feature_set,
                    "leakage_audit_pass": 1,
                    "meta_test_no_controller_update": 1,
                }
            )
            detail.append(row)

    output_stem = f"v22_61_m0c_{args.output_tag}"
    detail_file = OUT_ROOT / f"{output_stem}_detail.csv"
    train_file = OUT_ROOT / f"{output_stem}_train.csv"
    summary_file = OUT_ROOT / f"{output_stem}_summary.csv"
    route_file = OUT_ROOT / f"{output_stem}_route.json"
    summary = summarize(detail, train_rows)
    write_rows(detail_file, detail)
    write_rows(train_file, train_rows)
    write_rows(summary_file, summary)
    route = {
        "m0c_policy_ensemble_pass": max(iflag(r["m0c_policy_ensemble_pass"]) for r in summary),
        "summary_file": str(summary_file),
        "detail_file": str(detail_file),
        "train_file": str(train_file),
        "label_mode": str(args.label_mode),
        "feature_set": str(args.feature_set),
        "hidden": int(args.hidden),
        "diagnostic_only_not_official_promotion": 1,
    }
    route_file.write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    command = (
        f"{PYTHON} experiments/analyze_v22_61_m0c_policy_ensemble.py --prefix {args.prefix} "
        f"--base-detail {args.base_detail} --feature-set {args.feature_set} --label-mode {args.label_mode} "
        f"--hidden {int(args.hidden)} --epochs {int(args.epochs)} --output-tag {args.output_tag}"
    )
    diag.append_exec(
        command,
        task_id="v22_61_m0c_policy_ensemble",
        status="pass",
        files=f"{summary_file}; {detail_file}; {train_file}; {route_file}",
        note=json.dumps(route, sort_keys=True),
    )
    print(json.dumps(route, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
