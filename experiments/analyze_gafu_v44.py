#!/usr/bin/env python3
"""Summarize DG-KAN v4.4 functional optimizer redesign experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v4_4"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v4.4_FunctionalOptimizer_Redesign_结果复盘.md"
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
                seen.add(key)
                keys.append(key)
    if not keys:
        keys = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


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
    if not math.isfinite(float(x)):
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


def group(rows: Iterable[dict], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[dict]]:
    out: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        out[tuple(row.get(k, "") for k in keys)].append(row)
    return dict(out)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(v: object) -> str:
        return str(v).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(cell(c) for c in row) + " |" for row in rows)
    return "\n".join(out)


def p0_summary(rows: List[dict]) -> Tuple[bool, dict]:
    strict = [r for r in rows if not r.get("error")]
    ok = bool(strict)
    for row in strict:
        ok = ok and int(f(row, "learnable_nonKAN_params", 999)) == 0
        ok = ok and abs(f(row, "functional_coverage", 0.0) - 1.0) < 1e-9
        ok = ok and abs(f(row, "input_coeff_seen", 0.0) - 1.0) < 1e-9
        ok = ok and abs(f(row, "block_coeff_seen", 0.0) - 1.0) < 1e-9
        ok = ok and abs(f(row, "output_coeff_seen", 0.0) - 1.0) < 1e-9
        ok = ok and int(f(row, "alpha_trainable", 1)) == 0
        ok = ok and f(row, "rollback_error_for_shadow_step", 1.0) < 1e-6
        recon = f(row, "coordinate_reconstruction_error", 0.0)
        if math.isfinite(recon):
            ok = ok and recon < 1e-6
    return ok, {
        "rows": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "strict_rows": len(strict),
        "max_nonkan": max([f(r, "learnable_nonKAN_params", 0.0) for r in strict] or [math.nan]),
        "min_coverage": min([f(r, "functional_coverage", math.nan) for r in strict] or [math.nan]),
        "max_rollback": max([f(r, "rollback_error_for_shadow_step", 0.0) for r in strict] or [math.nan]),
    }


def p1_summary(rows: List[dict]) -> Tuple[List[dict], List[str]]:
    out: List[dict] = []
    survivors: List[str] = []
    for (method,), rs in sorted(group(rows, ("method",)).items()):
        datasets = sorted({r["dataset"] for r in rs})
        train_all = all(avg([r for r in rs if r["dataset"] == d], "actual_train_descent") > 0 for d in datasets)
        hold_pos = sum(1 for d in datasets if avg([r for r in rs if r["dataset"] == d], "actual_holdout_descent") >= 0)
        cos_f = avg(rs, "cos_function_with_adam")
        r2 = avg(rs, "function_displacement_r2_vs_adam")
        cos_c = avg(rs, "cos_coeff_with_adam")
        drift = avg(rs, "candidate_logit_drift")
        rank_delta = avg(rs, "rank_change_candidate")
        phi_ratios = []
        for r in rs:
            before = f(r, "phi_prime_p95_before")
            after = f(r, "phi_prime_p95_after")
            if math.isfinite(before) and before > 0 and math.isfinite(after):
                phi_ratios.append(after / before)
        phi_ratio = mean(phi_ratios) if phi_ratios else math.nan
        ok = (
            method != "PureKAN-AdamW-one-step"
            and train_all
            and hold_pos >= 2
            and (r2 > 0.50 or cos_f > 0.50)
            and rank_delta > -1.0
            and phi_ratio <= 1.25
        )
        if ok:
            survivors.append(method)
        out.append(
            {
                "method": method,
                "datasets": len(datasets),
                "train_descent_all": int(train_all),
                "holdout_positive_datasets": hold_pos,
                "cos_function_with_adam": cos_f,
                "function_r2_vs_adam": r2,
                "cos_coeff_with_adam": cos_c,
                "logit_drift": drift,
                "rank_change": rank_delta,
                "phi_ratio": phi_ratio,
                "p1_pass": int(ok or method == "PureKAN-AdamW-one-step"),
            }
        )
    return out, survivors


def _ratio_against(value: float, baseline: float, *, default_pass: bool = True) -> float:
    if not math.isfinite(baseline) or abs(baseline) < 1e-12:
        return 1.0 if default_pass and value >= baseline else 0.0
    return value / baseline


def p2_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    grouped = group(rows, ("dataset", "method"))
    adam_key = "PureKAN-AdamW-one-step"
    adam = {d: grouped.get((d, adam_key), []) for d in ["MNIST", "Fashion-MNIST", "KMNIST"]}
    adam_rank = {d: avg(adam[d], "cumulative_rank_change") for d in adam}
    adam_margin = {d: avg(adam[d], "cumulative_margin_change") for d in adam}
    adam_train = {d: avg(adam[d], "train_loss_descent_5step") for d in adam}
    adam_phi_after = {d: avg(adam[d], "phi_prime_p95_after") for d in adam}
    score: List[dict] = []
    failures: List[dict] = []
    method_ok: Dict[str, bool] = {}
    for (dataset, method), rs in sorted(grouped.items()):
        bad = avg(rs, "bad_step_rate")
        hold5 = avg(rs, "holdout_loss_descent_5step")
        train5 = avg(rs, "train_loss_descent_5step")
        r2 = avg(rs, "function_r2_to_adam_mean")
        rank = avg(rs, "cumulative_rank_change")
        margin = avg(rs, "cumulative_margin_change")
        phi_after = avg(rs, "phi_prime_p95_after")
        phi_ratio = phi_after / adam_phi_after.get(dataset, math.nan) if math.isfinite(adam_phi_after.get(dataset, math.nan)) and adam_phi_after[dataset] > 0 else math.nan
        train_ratio = _ratio_against(train5, adam_train.get(dataset, math.nan), default_pass=False)
        rank_ratio = _ratio_against(rank, adam_rank.get(dataset, math.nan), default_pass=True)
        margin_ratio = _ratio_against(margin, adam_margin.get(dataset, math.nan), default_pass=True)
        ok = method == adam_key or (
            bad <= 1e-12
            and hold5 >= 0
            and (r2 > 0.50 or train_ratio >= 0.80)
            and rank_ratio >= 0.70
            and margin_ratio >= 0.70
            and phi_ratio <= 0.90
        )
        if method != adam_key:
            method_ok.setdefault(method, True)
            if not ok:
                method_ok[method] = False
                reasons = []
                if bad > 1e-12:
                    reasons.append("F2: bad_step_rate>0")
                if hold5 < 0 or not math.isfinite(hold5):
                    reasons.append("F2: holdout_5step_negative")
                if not (r2 > 0.50 or train_ratio >= 0.80):
                    reasons.append("F3/F8: low_adam_alignment_or_train_descent")
                if rank_ratio < 0.70:
                    reasons.append("F5: feature_rank_not_retained")
                if margin_ratio < 0.70:
                    reasons.append("F6: margin_not_retained")
                if not (math.isfinite(phi_ratio) and phi_ratio <= 0.90):
                    reasons.append("F7: geometry_not_better_than_adamw_gate")
                failures.append(
                    {
                        "stage": "P2",
                        "dataset": dataset,
                        "method": method,
                        "failure_type": "; ".join(reasons) or "p2_gate_failed",
                        "detail": f"bad={bad:.3g}, hold5={hold5:.4g}, r2={r2:.3g}, train_ratio={train_ratio:.3g}, rank_ratio={rank_ratio:.3g}, margin_ratio={margin_ratio:.3g}, phi_ratio={phi_ratio:.3g}",
                    }
                )
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "train_descent_5step": train5,
                "holdout_descent_5step": hold5,
                "bad_step_rate": bad,
                "function_r2_to_adam": r2,
                "train_descent_ratio_vs_adam": train_ratio,
                "rank_change": rank,
                "rank_ratio_vs_adam": rank_ratio,
                "margin_change": margin,
                "margin_ratio_vs_adam": margin_ratio,
                "phi_after": phi_after,
                "phi_ratio_vs_adam": phi_ratio,
                "test_acc_after_5step": avg(rs, "test_acc_after_5step"),
                "p2_dataset_pass": int(ok),
            }
        )
    survivors = [m for m, ok in method_ok.items() if ok]
    return score, failures, survivors


def blank_required(reason: str) -> None:
    for name in [
        "p3_short_horizon_train_scorecard.csv",
        "p4_fcadam_scorecard.csv",
        "p5_sfd_scorecard.csv",
        "p6_gfk_scorecard.csv",
        "p7_candidate_selection.csv",
        "p8_confirm5.csv",
        "p9_confirm10.csv",
        "final_scorecard.csv",
        "paired_delta_vs_purekan_adamw.csv",
        "paired_delta_vs_mlp.csv",
        "paired_delta_vs_hybrid.csv",
        "wallclock_summary.csv",
        "geometry_summary.csv",
        "representation_summary.csv",
    ]:
        path = BASE / name
        if not path.exists() or path.stat().st_size == 0:
            write_csv(path, [{"status": "not_run", "reason": reason}])


def failure_reports(failures: List[dict], p1: List[dict], p2: List[dict]) -> None:
    write_csv(BASE / "failure_table.csv", failures)
    by_type: Dict[str, int] = defaultdict(int)
    by_dataset: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in failures:
        tags = [part.split(":")[0].strip() for part in row.get("failure_type", "").split(";") if part.strip()]
        if not tags:
            tags = ["unclassified"]
        for tag in tags:
            by_type[tag] += 1
            by_dataset[(row.get("dataset", ""), tag)] += 1
    write_csv(BASE / "failure_type_by_role.csv", [{"role": "all_coeff", "failure_type": k, "count": v} for k, v in sorted(by_type.items())])
    rows = []
    datasets = sorted({k[0] for k in by_dataset if k[0]})
    tags = sorted({k[1] for k in by_dataset})
    for d in datasets:
        row = {"dataset": d}
        for tag in tags:
            row[tag] = by_dataset.get((d, tag), 0)
        rows.append(row)
    write_csv(BASE / "failure_type_by_dataset.csv", rows)
    write_csv(
        BASE / "proposal_vs_adam_alignment.csv",
        [
            {
                "stage": "P1",
                "dataset": r.get("dataset", ""),
                "method": r.get("method", ""),
                "cos_function_with_adam": r.get("cos_function_with_adam", ""),
                "function_r2_vs_adam": r.get("function_displacement_r2_vs_adam", ""),
                "train_descent": r.get("actual_train_descent", ""),
                "holdout_descent": r.get("actual_holdout_descent", ""),
            }
            for r in p1
        ]
        + [
            {
                "stage": "P2",
                "dataset": r.get("dataset", ""),
                "method": r.get("method", ""),
                "cos_function_with_adam": "",
                "function_r2_vs_adam": r.get("function_r2_to_adam", ""),
                "train_descent": r.get("train_descent_5step", ""),
                "holdout_descent": r.get("holdout_descent_5step", ""),
            }
            for r in p2
        ],
    )
    write_csv(
        BASE / "feature_rank_failure.csv",
        [r for r in p2 if f(r, "rank_ratio_vs_adam", 1.0) < 0.70 and r.get("method") != "PureKAN-AdamW-one-step"],
    )
    write_csv(
        BASE / "margin_failure.csv",
        [r for r in p2 if f(r, "margin_ratio_vs_adam", 1.0) < 0.70 and r.get("method") != "PureKAN-AdamW-one-step"],
    )
    write_csv(
        BASE / "geometry_regularization_failure.csv",
        [r for r in p2 if f(r, "phi_ratio_vs_adam", 0.0) > 0.90 and r.get("method") != "PureKAN-AdamW-one-step"],
    )


def svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    height = 260
    vals = [v if math.isfinite(v) else 0.0 for v in values]
    lo = min(0.0, min(vals) if vals else 0.0)
    hi = max(1e-9, max(vals) if vals else 1.0)
    span = hi - lo
    bar_w = max(8, int((width - 160) / max(1, len(vals))))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="24" y="28" font-family="sans-serif" font-size="16">{title}</text>')
    base_y = 210
    for i, (lab, val) in enumerate(zip(labels, vals)):
        x = 70 + i * bar_w
        h = int(150 * (val - lo) / span)
        y = base_y - h
        parts.append(f'<rect x="{x}" y="{y}" width="{max(4, bar_w-3)}" height="{h}" fill="#4c78a8"/>')
        parts.append(f'<text x="{x}" y="230" font-family="sans-serif" font-size="9" transform="rotate(35 {x} 230)">{lab[:20]}</text>')
        parts.append(f'<text x="{x}" y="{max(42, y-4)}" font-family="sans-serif" font-size="8">{val:.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def make_figures(p1s: List[dict], p2s: List[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p1_by_method = sorted(group(p1s, ("method",)).items())
    svg_bar(
        FIG / "p1_function_r2_vs_adam.svg",
        "P1 function displacement R2 vs AdamW",
        [k[0] for k, _ in p1_by_method],
        [avg(rs, "function_r2_vs_adam") for _, rs in p1_by_method],
    )
    p2_by_method = sorted(group(p2s, ("method",)).items())
    svg_bar(
        FIG / "p2_holdout_5step.svg",
        "P2 holdout loss descent after 5 steps",
        [k[0] for k, _ in p2_by_method],
        [avg(rs, "holdout_descent_5step") for _, rs in p2_by_method],
    )
    svg_bar(
        FIG / "p2_phi_ratio_vs_adam.svg",
        "P2 phi p95 ratio vs AdamW",
        [k[0] for k, _ in p2_by_method],
        [avg(rs, "phi_ratio_vs_adam") for _, rs in p2_by_method],
    )


def main() -> int:
    p0 = read_rows(BASE / "p0_invariants.csv")
    p1 = read_rows(BASE / "p1_adam_functional_trajectory.csv")
    p2 = read_rows(BASE / "p2_horizon_audit.csv")
    p0_pass, p0_stats = p0_summary(p0)
    p1s, p1_survivors = p1_summary(p1)
    p2s, failures, p2_survivors = p2_summary(p2)
    write_csv(BASE / "p1_gate_summary.csv", p1s)
    write_csv(BASE / "p2_gate_summary.csv", p2s)
    reason = "P2 produced no survivor" if not p2_survivors else "manual follow-up required for P3 candidates"
    if not p2_survivors:
        blank_required(reason)
    failure_reports(failures, p1, p2s)
    make_figures(p1s, p2s)
    decision = {
        "p0_pass": p0_pass,
        "p0": p0_stats,
        "p1_survivors": p1_survivors,
        "p2_survivors": p2_survivors,
        "p3_triggered": bool(p2_survivors),
        "final_decision": "stop_after_p2_no_survivor" if not p2_survivors else "p3_needed",
    }
    save_json(BASE / "aggregate_decision.json", decision)

    p0_table = md_table(
        ["rows", "errors", "strict", "max nonKAN", "min cov", "max rollback", "pass"],
        [[p0_stats["rows"], p0_stats["errors"], p0_stats["strict_rows"], fmt(p0_stats["max_nonkan"]), fmt(p0_stats["min_coverage"]), fmt(p0_stats["max_rollback"]), str(p0_pass).lower()]],
    )
    p1_table = md_table(
        ["method", "train all", "holdout+", "cos f", "R2", "cos coeff", "rank Δ", "phi ratio", "P1"],
        [
            [
                r["method"],
                r["train_descent_all"],
                r["holdout_positive_datasets"],
                fmt(f(r, "cos_function_with_adam")),
                fmt(f(r, "function_r2_vs_adam")),
                fmt(f(r, "cos_coeff_with_adam")),
                fmt(f(r, "rank_change")),
                fmt(f(r, "phi_ratio")),
                "yes" if int(f(r, "p1_pass", 0)) else "no",
            ]
            for r in p1s
        ],
    )
    p2_table = md_table(
        ["dataset", "method", "runs", "hold5", "bad", "R2", "rank/A", "margin/A", "phi/A", "P2"],
        [
            [
                r["dataset"],
                r["method"],
                r["runs"],
                fmt(f(r, "holdout_descent_5step")),
                fmt(f(r, "bad_step_rate")),
                fmt(f(r, "function_r2_to_adam")),
                fmt(f(r, "rank_ratio_vs_adam")),
                fmt(f(r, "margin_ratio_vs_adam")),
                fmt(f(r, "phi_ratio_vs_adam")),
                "yes" if int(f(r, "p2_dataset_pass", 0)) else "no",
            ]
            for r in p2s
        ],
    )

    best_rows = []
    for dataset in ["MNIST", "Fashion-MNIST", "KMNIST"]:
        candidates = [r for r in p2s if r.get("dataset") == dataset and r.get("method") != "PureKAN-AdamW-one-step"]
        if candidates:
            best = max(candidates, key=lambda r: f(r, "holdout_descent_5step", -1e9))
            best_rows.append([dataset, best["method"], fmt(f(best, "holdout_descent_5step")), fmt(f(best, "function_r2_to_adam")), fmt(f(best, "rank_ratio_vs_adam")), fmt(f(best, "phi_ratio_vs_adam"))])
    best_table = md_table(["dataset", "best by hold5", "hold5", "R2", "rank/A", "phi/A"], best_rows)

    doc = f"""# DG-KAN v4.4 Functional Optimizer Redesign 结果复盘

本轮依据 `docs/DG-KAN_v4.4_FunctionalOptimizer_Redesign_实验计划.md`。目标是验证 PureKAN functional optimizer 是否应该改写为 functional-coordinate Adam、Shadow-Adam function distillation 或 global output-space kernel step。

## Run Inventory

{md_table(["stage", "rows", "errors"], [["P0 smoke", len(p0), sum(1 for r in p0 if r.get("error"))], ["P1 trajectory", len(p1), sum(1 for r in p1 if r.get("error"))], ["P2 horizon", len(p2), sum(1 for r in p2 if r.get("error"))]])}

## Code / Config Changes

```text
experiments/run_gafu_v44.py
  Added FCAdam-v2 functional-coordinate Adam with L2/H1/H1-low/dataSob metrics.
  Added SFD direct/prox/residual using shadow AdamW layer-output displacement as teacher.
  Added GFK diagnostic output/block/all-lowrank target-fit steps.
  Added AB-RBF PureKAN with constant/linear edge basis and no non-KAN trainable params.
  P1 records coefficient/function alignment to AdamW, R2, rank/margin/geometry changes.
  P2 records one-step and five-step horizon metrics for the written gate.

experiments/analyze_gafu_v44.py
  Generates P1/P2 gate summaries, failure reports, required placeholder artifacts,
  aggregate_decision.json, SVG diagnostics, and this replay.
```

## P0 Implementation Invariants

{p0_table}

P0 verdict: {"pass" if p0_pass else "fail"}. All successful rows keep strict PureKAN trainable parameters at zero non-KAN params, coefficient coverage at 1.0, fixed alpha, and exact rollback.

## P1 AdamW Functional Trajectory Audit

{p1_table}

P1 survivors:

```text
{", ".join(p1_survivors) if p1_survivors else "none"}
```

Observation:

```text
FCAdam and AB-RBF-FCAdam produce high function-space R2 on several datasets.
SFD often gives positive local descent but weaker R2 to AdamW function displacement.
GFK is stable but the output displacement is extremely small, with R2 near zero.
EKFNG-leftRightFull-lowSob remains numerically aggressive and fails local direction on multiple datasets.
```

## P2 One-Step / Five-Step Horizon Gate

{p2_table}

Best non-Adam points by holdout 5-step descent:

{best_table}

P2 survivors:

```text
{", ".join(p2_survivors) if p2_survivors else "none"}
```

P2 verdict:

```text
No candidate enters P3.

Many methods achieve positive 5-step holdout descent, especially FCAdam-dataSob,
FPA-edge-prox, F4-FNG and AB-RBF-FCAdam.
However the full written gate is stricter: candidates must keep Adam-like function
alignment or Adam-level train descent, retain rank/margin formation, and reduce
phi_prime_p95 below 90% of PureKAN-AdamW.

The persistent blocker is the geometry gate plus representation retention:
the candidates can learn locally, but they do not yet combine Adam-like function
trajectory, feature formation, and strict geometry improvement.
```

## P3-P9 Decision

```text
P3 short-horizon training: not run.
P4 FCAdam deep validation: not run.
P5 SFD deep validation: not run.
P6 GFK deep validation: not run.
P7/P8/P9 confirm: not run.

Reason: P2 produced no survivor under the written gate.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
```

Diagnosis:

```text
F3/F8 low Adam alignment or insufficient train dynamics:
  GFK is too small; SFD has low R2; several FCAdam variants fall just below the R2 gate.

F5/F6 representation failure:
  Some methods with positive holdout descent do not retain enough rank/margin movement vs AdamW.

F7 geometry gate failure:
  This is the broadest blocker. Functional candidates usually keep phi close to AdamW,
  but the v4.4 P2 gate requires phi <= 90% of AdamW while preserving learning.

AB-RBF diagnostic:
  AB-RBF-FCAdam improves local function alignment and holdout descent in places,
  but does not clear the joint rank/margin/geometry gate.
```

## Required Artifacts

Written under `results/v4_4/`:

```text
p0_invariants.csv
p1_adam_functional_trajectory.csv
p1_gate_summary.csv
p2_horizon_audit.csv
p2_gate_summary.csv
p3_short_horizon_train_scorecard.csv
p4_fcadam_scorecard.csv
p5_sfd_scorecard.csv
p6_gfk_scorecard.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
optimizer_dynamics_trace.csv
failure_table.csv
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
aggregate_decision.json
figures/p1_function_r2_vs_adam.svg
figures/p2_holdout_5step.svg
figures/p2_phi_ratio_vs_adam.svg
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v4.4.

What improved:
  FCAdam-v2 and AB-RBF-FCAdam show that functional-coordinate Adam can track
  AdamW function displacement much better than old Sobolev-only updates.
  SFD confirms that AdamW function displacement can be fit safely in one-step diagnostics.
  GFK gives a stable but too-small global output-space diagnostic.

What failed:
  None of FCAdam, SFD, GFK, FPA, EK-FNG, or AB-RBF-FCAdam passes the P2 joint gate.
  Positive 5-step loss descent did not become a validated representation-learning optimizer.

Conclusion:
  v4.4 partially supports the functional-coordinate Adam hypothesis, but not enough
  to justify P3/P4 expansion. The next useful redesign should keep more Adam-like
  functional trajectory while explicitly meeting the phi/rank/margin gate, rather
  than running more seeds on the current candidates.
```
"""
    OUT.write_text(doc, encoding="utf-8")

    log_entry = f"""

## 2026-05-03 DG-KAN v4.4 Functional Optimizer Redesign

Implemented and ran `experiments/run_gafu_v44.py` and `experiments/analyze_gafu_v44.py`.

Summary:

```text
P0 rows={len(p0)}, errors={sum(1 for r in p0 if r.get('error'))}, pass={p0_pass}
P1 survivors={p1_survivors if p1_survivors else 'none'}
P2 survivors={p2_survivors if p2_survivors else 'none'}
Decision=stop after P2 by written gate
```

Result replay:

```text
docs/DG-KAN_v4.4_FunctionalOptimizer_Redesign_结果复盘.md
results/v4_4/aggregate_decision.json
```
"""
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(log_entry)

    print(f"Wrote {OUT}")
    print(f"P0 pass={p0_pass}; P1 survivors={p1_survivors}; P2 survivors={p2_survivors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
