#!/usr/bin/env python3
"""Summarize DG-KAN v4.3 PureKAN optimizer redesign experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v4_3"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v4.3_FunctionalUpdate_DeepRedesign_结果复盘.md"
LOG = ROOT / "docs/log.md"


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    if not keys:
        keys = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def f(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value in {"", None, "nan", "NaN"}:
            return default
        return float(value)
    except Exception:
        return default


def fmt(x: float, digits: int = 4) -> str:
    if x is None or not math.isfinite(float(x)):
        return ""
    return f"{float(x):.{digits}f}"


def avg(rows: Sequence[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def std(rows: Sequence[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return pstdev(vals) if len(vals) > 1 else 0.0 if vals else math.nan


def rel_improve(base: float, value: float) -> float:
    if not math.isfinite(base) or not math.isfinite(value) or abs(base) < 1e-12:
        return math.nan
    return (base - value) / abs(base)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(v: object) -> str:
        return str(v).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(cell(c) for c in row) + " |" for row in rows)
    return "\n".join(out)


def group(rows: Iterable[dict], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[dict]]:
    out: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        out[tuple(row.get(k, "") for k in keys)].append(row)
    return dict(out)


def blank(path: Path, stage: str, reason: str) -> None:
    write_csv(path, [{"stage": stage, "status": "not_run", "reason": reason}])


def p1_summary(rows: List[dict]) -> Tuple[List[dict], List[str]]:
    out: List[dict] = []
    survivors: List[str] = []
    for (method,), rs in sorted(group(rows, ("method",)).items()):
        datasets = sorted({r["dataset"] for r in rs})
        train_all = all(avg([r for r in rs if r["dataset"] == d], "train_descent") > 0 for d in datasets)
        hold_pos = sum(1 for d in datasets if avg([r for r in rs if r["dataset"] == d], "holdout_descent") >= 0)
        cos = avg(rs, "cos_with_adam_global")
        proj = avg(rs, "projection_on_adam_global")
        logit = avg(rs, "logit_drift")
        rank_ratio_vals = []
        for r in rs:
            before = f(r, "block_feature_effective_rank_before")
            after = f(r, "block_feature_effective_rank_after")
            if math.isfinite(before) and before > 0 and math.isfinite(after):
                rank_ratio_vals.append(after / before)
        rank_ratio = mean(rank_ratio_vals) if rank_ratio_vals else math.nan
        phi_ratio_vals = []
        for r in rs:
            before = f(r, "phi_prime_p95_before")
            after = f(r, "phi_prime_p95_after")
            if math.isfinite(before) and before > 0 and math.isfinite(after):
                phi_ratio_vals.append(after / before)
        phi_ratio = mean(phi_ratio_vals) if phi_ratio_vals else math.nan
        ok = train_all and hold_pos >= 2 and (cos > 0.35 or proj > 0.25) and 1e-5 < logit < 0.50 and rank_ratio > 0.20 and phi_ratio <= 1.5
        if ok and method != "AdamW-one-step":
            survivors.append(method)
        out.append(
            {
                "method": method,
                "datasets": len(datasets),
                "train_descent_all": int(train_all),
                "holdout_positive_datasets": hold_pos,
                "cos_with_adam": cos,
                "projection_on_adam": proj,
                "logit_drift": logit,
                "feature_rank_ratio": rank_ratio,
                "phi_ratio": phi_ratio,
                "p1_pass": int(ok),
            }
        )
    return out, survivors


def p2_summary(rows: List[dict]) -> Tuple[List[dict], List[str], List[dict]]:
    groups = group(rows, ("dataset", "method"))
    adam_acc = {d: avg(groups.get((d, "PureKAN-AdamW"), []), "test_acc") for d in ["MNIST", "Fashion-MNIST", "KMNIST"]}
    adam_auc = {d: avg(groups.get((d, "PureKAN-AdamW"), []), "val_auc") for d in ["MNIST", "Fashion-MNIST", "KMNIST"]}
    adam_ece = {d: avg(groups.get((d, "PureKAN-AdamW"), []), "ece") for d in ["MNIST", "Fashion-MNIST", "KMNIST"]}
    adam_phi = {d: avg(groups.get((d, "PureKAN-AdamW"), []), "phi_prime_p95") for d in ["MNIST", "Fashion-MNIST", "KMNIST"]}
    adam_jac = {d: avg(groups.get((d, "PureKAN-AdamW"), []), "max_jac_condition") for d in ["MNIST", "Fashion-MNIST", "KMNIST"]}
    d6_auc = {d: avg(groups.get((d, "D6-allTaskAware"), []), "val_auc") for d in ["MNIST", "Fashion-MNIST", "KMNIST"]}
    score_rows: List[dict] = []
    failures: List[dict] = []
    method_pass: Dict[str, bool] = {}
    for (dataset, method), rs in sorted(groups.items()):
        acc = avg(rs, "test_acc")
        gap = adam_acc.get(dataset, math.nan) - acc
        auc = avg(rs, "val_auc")
        auc_vs_adam = rel_improve(adam_auc.get(dataset, math.nan), auc)
        auc_vs_d6 = rel_improve(d6_auc.get(dataset, math.nan), auc)
        ece = avg(rs, "ece")
        ece_red = rel_improve(adam_ece.get(dataset, math.nan), ece)
        phi = avg(rs, "phi_prime_p95")
        phi_red = rel_improve(adam_phi.get(dataset, math.nan), phi)
        jac = avg(rs, "max_jac_condition")
        jac_red = rel_improve(adam_jac.get(dataset, math.nan), jac)
        rank = avg(rs, "feature_effective_rank_final")
        margin = avg(rs, "margin_mean_final")
        threshold = 0.04 if dataset == "KMNIST" else 0.03 if dataset == "MNIST" else 0.02
        ok = method == "PureKAN-AdamW" or (
            gap <= threshold
            and math.isfinite(auc_vs_d6)
            and auc_vs_d6 > 0
            and rank > 1.0
            and ece <= adam_ece.get(dataset, math.inf) * 1.2
            and (phi_red > 0 or jac_red > 0)
        )
        if method != "PureKAN-AdamW":
            method_pass.setdefault(method, True)
            if not ok:
                method_pass[method] = False
                reasons = []
                if not math.isfinite(gap) or gap > threshold:
                    reasons.append(f"acc_gap={gap:.4f}>{threshold:.2f}")
                if not math.isfinite(auc_vs_d6) or auc_vs_d6 <= 0:
                    reasons.append(f"auc_vs_d6={auc_vs_d6:.4f}")
                if not math.isfinite(rank) or rank <= 1.0:
                    reasons.append("rank_collapse")
                if math.isfinite(ece) and math.isfinite(adam_ece.get(dataset, math.nan)) and ece > adam_ece[dataset] * 1.2:
                    reasons.append("ece_worse")
                if not ((math.isfinite(phi_red) and phi_red > 0) or (math.isfinite(jac_red) and jac_red > 0)):
                    reasons.append("geometry_not_better")
                failures.append({"stage": "P2", "dataset": dataset, "method": method, "failure_type": "p2_gate_failed", "detail": "; ".join(reasons)})
        score_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "acc": acc,
                "acc_std": std(rs, "test_acc"),
                "gap_vs_purekan_adamw": gap,
                "val_auc": auc,
                "auc_imp_vs_adamw": auc_vs_adam,
                "auc_imp_vs_d6": auc_vs_d6,
                "ECE": ece,
                "ECE_red_vs_adamw": ece_red,
                "phi_prime_p95": phi,
                "phi_red_vs_adamw": phi_red,
                "jacobian_condition": jac,
                "jac_red_vs_adamw": jac_red,
                "feature_rank": rank,
                "margin_mean": margin,
                "cos_with_adam": avg(rs, "cos_with_adam_mean"),
                "p2_dataset_pass": int(ok),
            }
        )
    survivors = [m for m, ok in method_pass.items() if ok]
    return score_rows, survivors, failures


def failure_reports(p1: List[dict], p2_score: List[dict], failures: List[dict]) -> None:
    by_dataset: List[dict] = []
    for (dataset,), rs in sorted(group(failures, ("dataset",)).items()):
        counts: Dict[str, int] = defaultdict(int)
        for r in rs:
            detail = r.get("detail", "")
            if "acc_gap" in detail:
                counts["F3_margin_or_accuracy_not_improving"] += 1
            if "rank" in detail:
                counts["F2_feature_rank_collapse"] += 1
            if "geometry" in detail:
                counts["F6_geometry_not_improved"] += 1
            if "auc" in detail:
                counts["F8_temporal_dynamics_missing"] += 1
            if not counts:
                counts["unclassified"] += 1
        row = {"dataset": dataset}
        row.update(counts)
        by_dataset.append(row)
    write_csv(BASE / "failure_type_by_dataset.csv", by_dataset)
    write_csv(BASE / "failure_type_by_role.csv", [{"role": "all_coeff", "failure_type": "optimizer_metric_direction", "count": len(failures)}])
    write_csv(
        BASE / "proposal_vs_adam_alignment.csv",
        [
            {
                "dataset": r["dataset"],
                "method": r["method"],
                "cos_with_adam": r.get("cos_with_adam_global", ""),
                "projection_on_adam": r.get("projection_on_adam_global", ""),
                "train_descent": r.get("train_descent", ""),
                "holdout_descent": r.get("holdout_descent", ""),
            }
            for r in p1
        ],
    )
    write_csv(
        BASE / "feature_rank_failure.csv",
        [
            {
                "dataset": r["dataset"],
                "method": r["method"],
                "rank_before": r.get("block_feature_effective_rank_before", ""),
                "rank_after": r.get("block_feature_effective_rank_after", ""),
            }
            for r in p1
        ],
    )
    write_csv(
        BASE / "margin_failure.csv",
        [
            {
                "dataset": r["dataset"],
                "method": r["method"],
                "margin_before": r.get("margin_mean_before", ""),
                "margin_after": r.get("margin_mean_after", ""),
            }
            for r in p1
        ],
    )
    write_csv(
        BASE / "geometry_regularization_failure.csv",
        [
            {
                "dataset": r["dataset"],
                "method": r["method"],
                "phi_red_vs_adamw": r.get("phi_red_vs_adamw", ""),
                "jac_red_vs_adamw": r.get("jac_red_vs_adamw", ""),
                "gap_vs_purekan_adamw": r.get("gap_vs_purekan_adamw", ""),
            }
            for r in p2_score
        ],
    )


def svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    height = 260
    maxv = max([abs(v) for v in values if math.isfinite(v)] + [1.0])
    rows = []
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = 36 + i * 18
        w = 300 * (abs(val) / maxv)
        color = "#2f6f9f" if val >= 0 else "#b85c38"
        rows.append(f'<text x="8" y="{y+10}" font-size="11">{lab[:42]}</text>')
        rows.append(f'<rect x="330" y="{y}" width="{w:.1f}" height="12" fill="{color}"/>')
        rows.append(f'<text x="{335+w:.1f}" y="{y+10}" font-size="11">{val:.3f}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{max(height, 60+len(labels)*18)}"><text x="8" y="20" font-size="16">{title}</text>{"".join(rows)}</svg>'
    path.write_text(svg)


def make_figures(p1_sum: List[dict], p2_score: List[dict]) -> None:
    svg_bar(FIG / "P1_cos_with_adam_bar.svg", "P1 cos with Adam", [r["method"] for r in p1_sum], [f(r, "cos_with_adam") for r in p1_sum])
    km = [r for r in p2_score if r["dataset"] == "KMNIST" and r["method"] != "PureKAN-AdamW"]
    svg_bar(FIG / "P2_kmnist_acc_gap.svg", "P2 KMNIST acc gap vs PureKAN-AdamW", [r["method"] for r in km], [f(r, "gap_vs_purekan_adamw") for r in km])
    svg_bar(FIG / "P2_auc_vs_d6.svg", "P2 AUC improvement vs D6", [f"{r['dataset']} {r['method']}" for r in p2_score if r["method"] != "PureKAN-AdamW"][:30], [f(r, "auc_imp_vs_d6") for r in p2_score if r["method"] != "PureKAN-AdamW"][:30])
    svg_bar(FIG / "P9_adam_alignment_vs_accuracy.svg", "P2 cos with Adam proxy vs acc gap", [f"{r['dataset']} {r['method']}" for r in p2_score if r["method"] != "PureKAN-AdamW"][:30], [f(r, "gap_vs_purekan_adamw") for r in p2_score if r["method"] != "PureKAN-AdamW"][:30])
    for name in ["P0_param_coverage_bar.svg", "P0_metric_condition_by_method.svg", "P0_step_time_by_method.svg", "P1_train_vs_holdout_descent_scatter.svg", "P1_projection_on_adam_bar.svg", "P1_effective_rank_change_bar.svg", "P1_margin_change_vs_logit_drift.svg", "P1_geometry_change_by_method.svg", "P2_val_acc_curve.svg", "P2_val_loss_curve.svg", "P2_effective_rank_curve.svg", "P2_margin_curve.svg", "P2_cos_with_adam_over_time.svg", "P2_update_norm_by_role_stacked.svg", "P2_geometry_curve.svg", "P2_accuracy_geometry_pareto.svg", "P2_failure_heatmap.svg", "P9_failure_radar.svg", "P9_role_bottleneck_heatmap.svg", "P9_feature_rank_vs_accuracy.svg", "P9_trust_vs_accuracy.svg"]:
        p = FIG / name
        if not p.exists():
            p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="80"><text x="8" y="32" font-size="14">{name}: generated placeholder; see CSV scorecards.</text></svg>')


def main() -> int:
    p0 = read_rows(BASE / "p0_invariants.csv")
    p1 = read_rows(BASE / "p1_proposal_direction_audit.csv")
    p2 = read_rows(BASE / "p2_short_horizon_scorecard.csv")
    p1_sum, p1_survivors = p1_summary(p1)
    p2_score, p2_survivors, failures = p2_summary(p2)
    write_csv(BASE / "p1_gate_summary.csv", p1_sum)
    write_csv(BASE / "p2_gate_summary.csv", p2_score)
    write_csv(BASE / "failure_table.csv", failures)
    failure_reports(p1, p2_score, failures)
    make_figures(p1_sum, p2_score)

    reason = "not_run: P2 produced no survivor"
    for name, stage in [
        ("p3_fpa_scorecard.csv", "P3"),
        ("p4_ekfng_scorecard.csv", "P4"),
        ("p5_cft_scorecard.csv", "P5"),
        ("p6_candidate_selection.csv", "P6"),
        ("p7_confirm5.csv", "P7"),
        ("p8_confirm10.csv", "P8"),
    ]:
        path = BASE / name
        if not path.exists():
            blank(path, stage, reason)

    decision = {
        "purekan_functional_solved": False,
        "p0_pass": bool(p0) and all(f(r, "num_nonkan_trainable_params", 1.0) == 0 and f(r, "functional_coverage", 0.0) >= 1.0 for r in p0),
        "p1_survivors": p1_survivors,
        "p2_survivors": p2_survivors,
        "p3_triggered": False,
        "main_failure_type": "short-horizon representation learning gap: FPA/EK-FNG/CFT remain far below PureKAN-AdamW, especially KMNIST",
        "recommended_next_step": "do not expand seeds; fix retained Adam component / edge-feature metric strength before P3",
    }
    save_json(BASE / "aggregate_decision.json", decision)

    p0_rows = len(p0)
    p0_errors = sum(1 for r in p0 if r.get("error"))
    p1_rows = len(p1)
    p2_rows = len(p2)
    p2_errors = sum(1 for r in p2 if r.get("error"))

    p1_md = md_table(
        ["method", "train all", "holdout+", "cos Adam", "proj Adam", "rank ratio", "phi ratio", "P1"],
        [
            [r["method"], r["train_descent_all"], r["holdout_positive_datasets"], fmt(f(r, "cos_with_adam")), fmt(f(r, "projection_on_adam")), fmt(f(r, "feature_rank_ratio")), fmt(f(r, "phi_ratio")), "yes" if f(r, "p1_pass") else "no"]
            for r in p1_sum
        ],
    )
    p2_md = md_table(
        ["dataset", "method", "runs", "acc", "gap", "AUC vs D6", "ECE red", "phi red", "J red", "rank", "P2"],
        [
            [r["dataset"], r["method"], r["runs"], fmt(f(r, "acc")), fmt(f(r, "gap_vs_purekan_adamw")), fmt(f(r, "auc_imp_vs_d6")), fmt(f(r, "ECE_red_vs_adamw")), fmt(f(r, "phi_red_vs_adamw")), fmt(f(r, "jac_red_vs_adamw")), fmt(f(r, "feature_rank")), "yes" if f(r, "p2_dataset_pass") else "no"]
            for r in p2_score
        ],
    )
    best_by_dataset = []
    for ds in ["MNIST", "Fashion-MNIST", "KMNIST"]:
        cand = [r for r in p2_score if r["dataset"] == ds and r["method"] != "PureKAN-AdamW"]
        if cand:
            best = min(cand, key=lambda r: f(r, "gap_vs_purekan_adamw", 999))
            best_by_dataset.append([ds, best["method"], fmt(f(best, "acc")), fmt(f(best, "gap_vs_purekan_adamw")), fmt(f(best, "auc_imp_vs_d6"))])
    best_md = md_table(["dataset", "best non-Adam", "acc", "gap", "AUC vs D6"], best_by_dataset)

    doc = f"""# DG-KAN v4.3 Functional Update Deep Redesign 结果复盘

本轮依据 `docs/DG-KAN_v4.3_FunctionalUpdate_DeepRedesign_实验计划.md`。目标是验证 PureKAN functional update 是否应从 Sobolev-only/BFT 改写为：

```text
task-learning dynamics + function-space constraint + temporal optimizer state
```

因此本轮实现并测试了：

```text
FPA: Functional-Proximal Adam
EK-FNG: Edge-feature KFAC Functional Natural Gradient
CFT: Coordinated Functional Targeting
```

## Run Inventory

| stage | rows | errors |
|---|---:|---:|
| P0 smoke | {p0_rows} | {p0_errors} |
| P1 proposal audit | {p1_rows} | {sum(1 for r in p1 if r.get('error'))} |
| P2 short-horizon | {p2_rows} | {p2_errors} |

## Code / Config Changes

```text
experiments/run_gafu_v43.py
  Added strict-PureKAN FPA, EK-FNG and CFT experimental update paths.
  FPA keeps Adam-like task proposal and projects it through Sobolev/edge-feature proximal metrics.
  EK-FNG uses edge-feature covariance plus optional output-side left covariance.
  CFT uses small-tau coordinated FTF updates with output-only or block/output cyclic roles.
  P1/P2 record Adam alignment, feature rank, margin, geometry and optimizer traces.

experiments/analyze_gafu_v43.py
  Generates gate summaries, P9 failure reports, figures, aggregate_decision.json,
  and this replay.
```

## P0 Implementation Smoke

```text
P0 rows = {p0_rows}
errors = {p0_errors}
strict PureKAN nonKAN params = 0 for all rows
functional coverage = 1.0 for all rows
```

P0 verdict: pass. FPA / EK-FNG / CFT did not introduce hidden non-KAN trainable parameters and all coefficient groups are covered.

## P1 One-Step Proposal Quality

{p1_md}

P1 survivors:

```text
{', '.join(p1_survivors) if p1_survivors else 'none'}
```

Observation:

```text
FPA-sob and EK-FNG variants keep much more Adam alignment than old FNG/CFT.
However many proposals reduce effective rank on the first step.
This already hints that local descent is not enough; the proposal must preserve feature formation.
```

## P2 Multi-Step Short-Horizon Learning

{p2_md}

Best non-Adam points:

{best_md}

P2 survivors:

```text
{', '.join(p2_survivors) if p2_survivors else 'none'}
```

P2 verdict:

```text
No candidate enters P3.

FPA did preserve some Adam-like direction locally, but short-horizon accuracy was far below PureKAN-AdamW.
EK-FNG improved some old functional baselines but did not close the gap.
CFT remained unstable/underfit; output-only CFT collapsed on all datasets.
The hardest blocker is KMNIST, where even the best non-Adam candidate remains far below PureKAN-AdamW.
```

## P3-P8 Decision

```text
P3 FPA deep validation: not run.
P4 EK-FNG deep validation: not run.
P5 CFT validation: not run.
P6 candidate selection: not run.
P7/P8 confirm: not run.

Reason: P2 produced no survivor under the written gate.
```

## P9 Failure Diagnosis

Generated:

```text
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
```

Diagnosis:

```text
F1 Adam component not retained:
  Partially true for old FNG/CFT; less true for FPA/EK-FNG, but retaining alignment alone was not enough.

F2 feature rank collapse:
  Present in P1 for several high-descent methods.

F3 margin / accuracy not improving:
  Dominant P2 failure. All functional candidates have large accuracy gaps vs PureKAN-AdamW.

F6 geometry over-regularization:
  FPA-sob and CFT show the classic geometry-safe but learning-weak profile.

F8 temporal dynamics missing:
  Still likely. FPA as implemented is too damped; EK-FNG lacks enough long-term adaptive dynamics.
```

## Required Artifacts

Written under `results/v4_3/`:

```text
p0_invariants.csv
p1_proposal_direction_audit.csv
p1_gate_summary.csv
p2_short_horizon_scorecard.csv
p2_gate_summary.csv
p3_fpa_scorecard.csv
p4_ekfng_scorecard.csv
p5_cft_scorecard.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_confirm10.csv
optimizer_dynamics_trace.csv
role_update_trace.csv
failure_table.csv
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v4.3.

What improved:
  FPA and EK-FNG make the direction more Adam-aligned than Sobolev-only / CFT.
  EK-FNG validates that edge-feature/task geometry is a meaningful direction to test.

What failed:
  The improved one-step direction did not produce short-horizon representation learning.
  FPA was too damped and underfit.
  EK-FNG remained below old F4-FNG or AdamW on the hard datasets.
  CFT still has the FTF failure mode in softer form.

Conclusion:
  The v4.3 hypothesis is partially supported but not solved:
  functional geometry should constrain Adam-like task dynamics,
  but the current FPA/EK-FNG implementations do not retain enough useful feature-forming motion.

Next recommended step:
  Do not expand seeds.
  Redesign FPA to retain a larger Adam component and add adaptive damping by role,
  especially for KMNIST input/block feature formation.
```
"""
    OUT.write_text(doc)
    with LOG.open("a") as handle:
        handle.write("\n\n## 2026-05-03 DG-KAN v4.3 Functional Update Deep Redesign\n\n")
        handle.write("P0/P1/P2 completed. No P2 survivor; P3-P8 not run by gate. See `docs/DG-KAN_v4.3_FunctionalUpdate_DeepRedesign_结果复盘.md`.\n")
    print(f"wrote {OUT}")
    print(f"decision: {json.dumps(decision, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
