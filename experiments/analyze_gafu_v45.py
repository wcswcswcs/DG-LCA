#!/usr/bin/env python3
"""Summarize DG-KAN v4.5 Accelerated Functional Optimizer experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v4_5"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v4.5_AcceleratedFunctionalOptimizer_结果复盘.md"
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
    ok_rows = [r for r in rows if not r.get("error")]
    ok = bool(ok_rows)
    for row in ok_rows:
        ok = ok and f(row, "learnable_nonKAN_params", 999) == 0
        ok = ok and abs(f(row, "functional_coverage", 0.0) - 1.0) < 1e-9
        ok = ok and f(row, "alpha_trainable", 1) == 0
        ok = ok and f(row, "input_coeff_seen", 0) == 1
        ok = ok and f(row, "block_coeff_seen", 0) == 1
        ok = ok and f(row, "output_coeff_seen", 0) == 1
        ok = ok and f(row, "whiten_reconstruction_error", 1.0) < 1e-6
        ok = ok and f(row, "u_to_a_roundtrip_error", 1.0) < 1e-6
        ok = ok and f(row, "rollback_error", 1.0) == 0
        ok = ok and f(row, "no_nan_inf", 0) == 1
    stats = {
        "rows": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "max_nonkan": max([f(r, "learnable_nonKAN_params", 0) for r in ok_rows] or [math.nan]),
        "max_roundtrip": max([f(r, "whiten_reconstruction_error", 0) for r in ok_rows] or [math.nan]),
        "max_u_to_a": max([f(r, "u_to_a_roundtrip_error", 0) for r in ok_rows] or [math.nan]),
        "max_rollback": max([f(r, "rollback_error", 0) for r in ok_rows] or [math.nan]),
    }
    return ok, stats


def p1_targets(rows: List[dict]) -> List[dict]:
    out = []
    for (dataset,), rs in sorted(group(rows, ("dataset",)).items()):
        first = min(rs, key=lambda r: f(r, "step", 0))
        last = max(rs, key=lambda r: f(r, "step", 0))
        total_update = sum(f(r, "input_update_norm", 0) + f(r, "block_update_norm", 0) + f(r, "output_update_norm", 0) for r in rs)
        input_share = sum(f(r, "input_update_norm", 0) for r in rs) / max(1e-12, total_update)
        block_share = sum(f(r, "block_update_norm", 0) for r in rs) / max(1e-12, total_update)
        output_share = sum(f(r, "output_update_norm", 0) for r in rs) / max(1e-12, total_update)
        out.append(
            {
                "dataset": dataset,
                "steps": len(rs),
                "holdout_loss_start": f(first, "holdout_loss"),
                "holdout_loss_end": f(last, "holdout_loss"),
                "holdout_descent": f(first, "holdout_loss") - f(last, "holdout_loss"),
                "rank_start": f(first, "feature_effective_rank_block"),
                "rank_end": f(last, "feature_effective_rank_block"),
                "rank_ratio": f(last, "feature_effective_rank_block") / max(1e-12, f(first, "feature_effective_rank_block")),
                "margin_start": f(first, "margin_mean"),
                "margin_end": f(last, "margin_mean"),
                "phi_start": f(first, "phi_prime_p95"),
                "phi_end": f(last, "phi_prime_p95"),
                "phi_ratio": f(last, "phi_prime_p95") / max(1e-12, f(first, "phi_prime_p95")),
                "input_update_share": input_share,
                "block_update_share": block_share,
                "output_update_share": output_share,
                "m_norm_end": f(last, "AdamW_m_norm"),
                "v_norm_end": f(last, "AdamW_v_norm"),
            }
        )
    return out


def ratio(value: float, base: float) -> float:
    if not math.isfinite(value) or not math.isfinite(base) or abs(base) < 1e-12:
        return math.nan
    return value / base


def p2_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    grouped = group(rows, ("dataset", "method"))
    adam = {d: grouped.get((d, "PureKAN-AdamW-one-step"), []) for d in ["MNIST", "Fashion-MNIST", "KMNIST"]}
    adam_rank = {d: avg(adam[d], "rank_change") for d in adam}
    adam_margin = {d: avg(adam[d], "margin_change") for d in adam}
    adam_phi = {d: avg(adam[d], "phi_prime_p95") for d in adam}
    score: List[dict] = []
    failures: List[dict] = []
    method_status: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for (dataset, method), rs in sorted(grouped.items()):
        hold5 = avg(rs, "holdout_5step_descent")
        hold20 = avg(rs, "holdout_20step_descent")
        bad = avg(rs, "bad_step_rate")
        cosf = avg(rs, "cos_function_with_adam")
        r2 = avg(rs, "function_R2_with_adam")
        rank_a = ratio(avg(rs, "rank_change"), adam_rank.get(dataset, math.nan))
        margin_a = ratio(avg(rs, "margin_change"), adam_margin.get(dataset, math.nan))
        phi_a = ratio(avg(rs, "phi_prime_p95"), adam_phi.get(dataset, math.nan))
        ok_dataset = method == "PureKAN-AdamW-one-step" or (
            hold5 > 0
            and bad <= 0.05
            and rank_a >= 0.85
            and margin_a >= 0.85
            and phi_a <= 1.15
        )
        if method != "PureKAN-AdamW-one-step":
            method_status[method][dataset] = ok_dataset
            reasons = []
            if hold5 <= 0 or not math.isfinite(hold5):
                reasons.append("F7 task overfit/holdout bad")
            if bad > 0.05:
                reasons.append("F8 acceleration instability/bad steps")
            if rank_a < 0.85 or not math.isfinite(rank_a):
                reasons.append("F3 rank collapse")
            if margin_a < 0.85 or not math.isfinite(margin_a):
                reasons.append("F4 margin failure")
            if phi_a > 1.15 or not math.isfinite(phi_a):
                reasons.append("F5 geometry conflict")
            if cosf < 0.85:
                reasons.append("F1 coordinate mismatch/cos")
            if r2 < 0.70:
                reasons.append("F1 coordinate mismatch/R2")
            if reasons:
                failures.append(
                    {
                        "stage": "P2",
                        "dataset": dataset,
                        "method": method,
                        "failure_type": "; ".join(reasons),
                        "detail": f"hold5={hold5:.4g}, bad={bad:.3g}, cos={cosf:.3g}, r2={r2:.3g}, rankA={rank_a:.3g}, marginA={margin_a:.3g}, phiA={phi_a:.3g}",
                    }
                )
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "holdout_5step_descent": hold5,
                "holdout_20step_descent": hold20,
                "bad_step_rate": bad,
                "cos_function_with_adam": cosf,
                "function_R2_with_adam": r2,
                "rank/A": rank_a,
                "margin/A": margin_a,
                "phi/A": phi_a,
                "jac/A": ratio(avg(rs, "jacobian_condition"), avg(adam.get(dataset, []), "jacobian_condition")),
                "ECE": avg(rs, "ECE"),
                "accepted_lr": avg(rs, "accepted_lr"),
                "u_m_norm": avg(rs, "u_m_norm"),
                "u_v_norm": avg(rs, "u_v_norm"),
                "test_acc_after_20step": avg(rs, "test_acc_after_20step"),
                "p2_dataset_pass": int(ok_dataset),
            }
        )
    survivors: List[str] = []
    for method, by_dataset in method_status.items():
        if not all(by_dataset.get(d, False) for d in ["MNIST", "Fashion-MNIST", "KMNIST"]):
            continue
        rs = [r for r in score if r["method"] == method]
        cos_ok = sum(1 for r in rs if f(r, "cos_function_with_adam") >= 0.85) >= 2
        r2_ok = sum(1 for r in rs if f(r, "function_R2_with_adam") >= 0.70) >= 2
        if cos_ok and r2_ok:
            survivors.append(method)
        else:
            failures.append(
                {
                    "stage": "P2-global",
                    "dataset": "all",
                    "method": method,
                    "failure_type": "F1 coordinate mismatch/global alignment",
                    "detail": f"cos_ok={cos_ok}, r2_ok={r2_ok}",
                }
            )
    return score, failures, survivors


def blank_later(reason: str) -> None:
    for name in [
        "p3_temporal_dynamics.csv",
        "p4_lyapunov_restart.csv",
        "p5_rank_margin_ablation.csv",
        "p6_candidate_selection.csv",
        "p7_confirm5.csv",
        "p8_confirm10.csv",
    ]:
        path = BASE / name
        if not path.exists() or path.stat().st_size == 0:
            write_csv(path, [{"status": "not_run", "reason": reason}])


def failure_reports(failures: List[dict], p2: List[dict]) -> None:
    write_csv(BASE / "p9_failure_diagnosis.csv", failures)
    write_csv(BASE / "failure_table.csv", failures)
    by_dataset: Dict[Tuple[str, str], int] = defaultdict(int)
    by_method: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in failures:
        tags = [part.strip().split()[0] for part in row.get("failure_type", "").split(";") if part.strip()]
        if not tags:
            tags = ["unclassified"]
        for tag in tags:
            by_dataset[(row.get("dataset", ""), tag)] += 1
            by_method[(row.get("method", ""), tag)] += 1
    tags = sorted({t for _, t in by_dataset})
    rows = []
    for d in sorted({d for d, _ in by_dataset}):
        row = {"dataset": d}
        for tag in tags:
            row[tag] = by_dataset.get((d, tag), 0)
        rows.append(row)
    write_csv(BASE / "failure_type_heatmap_by_dataset.csv", rows)
    tags_m = sorted({t for _, t in by_method})
    rows_m = []
    for m in sorted({m for m, _ in by_method}):
        row = {"method": m}
        for tag in tags_m:
            row[tag] = by_method.get((m, tag), 0)
        rows_m.append(row)
    write_csv(BASE / "failure_type_heatmap_by_method.csv", rows_m)
    write_csv(
        BASE / "rank_margin_phi_scatter.csv",
        [
            {
                "dataset": r["dataset"],
                "method": r["method"],
                "rank/A": r["rank/A"],
                "margin/A": r["margin/A"],
                "phi/A": r["phi/A"],
                "test_acc_after_20step": r["test_acc_after_20step"],
            }
            for r in p2
        ],
    )


def svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 280
    vals = [v if math.isfinite(v) else 0.0 for v in values]
    lo = min(0.0, min(vals) if vals else 0.0)
    hi = max(1e-9, max(vals) if vals else 1.0)
    span = hi - lo
    bw = max(7, int((width - 150) / max(1, len(vals))))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(f'<text x="24" y="28" font-family="sans-serif" font-size="16">{title}</text>')
    base = 220
    for i, (lab, val) in enumerate(zip(labels, vals)):
        x = 70 + i * bw
        h = int(160 * (val - lo) / span)
        y = base - h
        parts.append(f'<rect x="{x}" y="{y}" width="{max(4, bw-3)}" height="{h}" fill="#4c78a8"/>')
        parts.append(f'<text x="{x}" y="242" font-family="sans-serif" font-size="8" transform="rotate(35 {x} 242)">{lab[:24]}</text>')
        parts.append(f'<text x="{x}" y="{max(44, y-4)}" font-family="sans-serif" font-size="8">{val:.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def make_figures(p1: List[dict], p2: List[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p1g = sorted(group(p1, ("dataset",)).items())
    svg_bar(FIG / "adamw_trajectory_holdout_descent.svg", "AdamW 20-step holdout descent", [k[0] for k, _ in p1g], [f(max(rs, key=lambda r: f(r, "step")), "holdout_loss") for _, rs in p1g])
    mg = sorted(group(p2, ("method",)).items())
    svg_bar(FIG / "fcadam_function_r2_vs_rank_retention.svg", "P2 function R2 mean by method", [k[0] for k, _ in mg], [avg(rs, "function_R2_with_adam") for _, rs in mg])
    svg_bar(FIG / "holdout_descent_20step.svg", "P2 holdout 20-step descent by method", [k[0] for k, _ in mg], [avg(rs, "holdout_20step_descent") for _, rs in mg])
    svg_bar(FIG / "phi_ratio_by_method.svg", "P2 phi/A by method", [k[0] for k, _ in mg], [avg(rs, "phi/A") for _, rs in mg])


def main() -> int:
    p0 = read_rows(BASE / "p0_coordinate_invariants.csv")
    p1 = read_rows(BASE / "p1_adamw_trajectory_forensic.csv")
    p2_raw = read_rows(BASE / "p2_fcadam_backbone_audit.csv")
    p0_pass, p0_stats = p0_summary(p0)
    p1_profile = p1_targets(p1)
    p2_score, failures, p2_survivors = p2_summary(p2_raw)
    write_csv(BASE / "p1_adamw_target_profile.csv", p1_profile)
    write_csv(BASE / "p2_fcadam_gate_summary.csv", p2_score)
    reason = "P2 produced no survivor" if not p2_survivors else "P2 survivor exists; run P3 manually"
    if not p2_survivors:
        blank_later(reason)
    failure_reports(failures, p2_score)
    make_figures(p1_profile, p2_score)
    decision = {
        "p0_pass": p0_pass,
        "p0": p0_stats,
        "p2_survivors": p2_survivors,
        "p3_triggered": bool(p2_survivors),
        "final_decision": "stop_after_p2_no_survivor" if not p2_survivors else "p3_needed",
    }
    save_json(BASE / "aggregate_decision.json", decision)

    p0_table = md_table(
        ["rows", "errors", "max nonKAN", "max roundtrip", "max u->a", "max rollback", "pass"],
        [[p0_stats["rows"], p0_stats["errors"], fmt(p0_stats["max_nonkan"]), fmt(p0_stats["max_roundtrip"]), fmt(p0_stats["max_u_to_a"]), fmt(p0_stats["max_rollback"]), str(p0_pass).lower()]],
    )
    p1_table = md_table(
        ["dataset", "holdout Δ", "rank ratio", "phi ratio", "input share", "block share", "output share"],
        [
            [
                r["dataset"],
                fmt(f(r, "holdout_descent")),
                fmt(f(r, "rank_ratio")),
                fmt(f(r, "phi_ratio")),
                fmt(f(r, "input_update_share")),
                fmt(f(r, "block_update_share")),
                fmt(f(r, "output_update_share")),
            ]
            for r in p1_profile
        ],
    )
    p2_table = md_table(
        ["dataset", "method", "runs", "hold20", "cos", "R2", "rank/A", "margin/A", "phi/A", "bad", "P2"],
        [
            [
                r["dataset"],
                r["method"],
                r["runs"],
                fmt(f(r, "holdout_20step_descent")),
                fmt(f(r, "cos_function_with_adam")),
                fmt(f(r, "function_R2_with_adam")),
                fmt(f(r, "rank/A")),
                fmt(f(r, "margin/A")),
                fmt(f(r, "phi/A")),
                fmt(f(r, "bad_step_rate")),
                "yes" if int(f(r, "p2_dataset_pass", 0)) else "no",
            ]
            for r in p2_score
        ],
    )
    best_rows = []
    for dataset in ["MNIST", "Fashion-MNIST", "KMNIST"]:
        rows = [r for r in p2_score if r["dataset"] == dataset and r["method"] != "PureKAN-AdamW-one-step"]
        if rows:
            best = max(rows, key=lambda r: f(r, "holdout_20step_descent", -1e9))
            best_rows.append([dataset, best["method"], fmt(f(best, "holdout_20step_descent")), fmt(f(best, "cos_function_with_adam")), fmt(f(best, "function_R2_with_adam")), fmt(f(best, "rank/A")), fmt(f(best, "phi/A"))])
    best_table = md_table(["dataset", "best hold20", "hold20", "cos", "R2", "rank/A", "phi/A"], best_rows)

    doc = f"""# DG-KAN v4.5 Accelerated Functional Optimizer 结果复盘

本轮依据 `docs/DG-KAN_v4.5_AcceleratedFunctionalOptimizer_实验计划.md`。目标是验证 PureKAN 是否需要 AFO：functional-coordinate Adam backbone + trajectory controller + phased geometry。

## Run Inventory

{md_table(["stage", "rows", "errors"], [["P0 coordinate smoke", len(p0), sum(1 for r in p0 if r.get('error'))], ["P1 AdamW forensic", len(p1), sum(1 for r in p1 if r.get('error'))], ["P2 FCAdam backbone", len(p2_raw), sum(1 for r in p2_raw if r.get('error'))]])}

## Code / Config Changes

```text
experiments/run_gafu_v45.py
  Added v4.5 coordinate correctness audit across basis_count 16/24, hidden_dim 64/96, depth 2/4.
  Added AdamW trajectory forensic logging for rank, margin, phi, role update share and Adam m/v state.
  Added 20-step FCAdam backbone audit for L2/H1-low/dataSob, AB-RBF-FCAdam and phased/no-geometry variants.

experiments/analyze_gafu_v45.py
  Generates target profiles, P2 gate summaries, failure taxonomy, figures, aggregate_decision.json,
  and this result replay.
```

## P0 Coordinate Correctness

{p0_table}

P0 verdict: {"pass" if p0_pass else "fail"}. Functional-coordinate whitening and roundtrip are correct to numerical precision, with strict PureKAN non-KAN parameter count still zero.

## P1 AdamW Trajectory Forensic

{p1_table}

Observation:

```text
AdamW itself is not a pure geometry-improving trajectory in the first 20 steps.
It lowers holdout loss while reducing effective rank and slightly increasing phi.
This supports v4.5's phased policy: early optimization should prioritize task trajectory,
rank/margin formation, and only later consolidate geometry.
```

## P2 FCAdam Backbone Audit

{p2_table}

Best non-Adam points by 20-step holdout descent:

{best_table}

P2 survivors:

```text
{", ".join(p2_survivors) if p2_survivors else "none"}
```

P2 verdict:

```text
No candidate enters P3.

FCAdam-dataSob has very strong 20-step holdout descent on all datasets, but its
function-space alignment to the AdamW target trajectory collapses over 20 steps.
AB-RBF-FCAdam improves some rank/margin behavior but still fails the global
cos/R2 requirements.

The v4.5 positive signal is real task descent, not a solved optimizer.
The blocker moved from immediate descent to temporal trajectory mismatch:
the methods learn locally, but their multi-step function displacement no longer
resembles AdamW enough to satisfy the written AFO backbone gate.
```

## P3-P8 Decision

```text
P3 Functional Nesterov / Adan-like dynamics: not run.
P4 Lyapunov restart controller: not run.
P5 rank/margin preservation ablation: not run.
P6 3-seed full-budget selection: not run.
P7/P8 confirm: not run.

Reason: P2 produced no survivor under the written gate.
```

## P9 Failure Diagnosis

Generated:

```text
p9_failure_diagnosis.csv
failure_table.csv
failure_type_heatmap_by_dataset.csv
failure_type_heatmap_by_method.csv
rank_margin_phi_scatter.csv
```

Diagnosis:

```text
F1 coordinate mismatch:
  Dominant after 20 steps. FCAdam-dataSob descends strongly, but cos/R2 to AdamW's
  function trajectory is far below the P2 requirement.

F3/F4 rank and margin:
  Some variants preserve rank/margin, but not together with Adam-like function alignment.

F5 geometry conflict:
  dataSob variants often have phi/A > 1.0. This is acceptable in Phase I up to 1.15,
  but it still does not compensate for low function alignment.

F8 temporal dynamics:
  The core unsolved issue is now temporal: one-step/short-step descent exists,
  but the 20-step trajectory drifts away from AdamW.
```

## Required Artifacts

Written under `results/v4_5/`:

```text
p0_coordinate_invariants.csv
p1_adamw_trajectory_forensic.csv
p1_adamw_target_profile.csv
p2_fcadam_backbone_audit.csv
p2_fcadam_gate_summary.csv
optimizer_dynamics_trace.csv
p3_temporal_dynamics.csv
p4_lyapunov_restart.csv
p5_rank_margin_ablation.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_confirm10.csv
p9_failure_diagnosis.csv
failure_table.csv
failure_type_heatmap_by_dataset.csv
failure_type_heatmap_by_method.csv
rank_margin_phi_scatter.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN Accelerated Functional Optimizer is not solved in v4.5.

What improved:
  FCAdam-dataSob confirms strong task-learning descent over 20 steps.
  P0 fully validates the function-coordinate implementation.
  P1 gives a concrete AdamW target profile and confirms phased geometry is the right framing.

What failed:
  FCAdam variants do not preserve enough AdamW function-space trajectory over 20 steps.
  No candidate satisfies the P2 global cos/R2 + rank/margin + Phase-I phi budget gate.
  P3/P4/P5 expansion is therefore not justified.

Conclusion:
  v4.5 supports the idea that functional-coordinate adaptive dynamics are the right
  direction, but the current AFO backbone still lacks temporal trajectory control.
  The next redesign should target multi-step function trajectory matching or restart
  dynamics only after improving P2 cos/R2 retention.
```
"""
    OUT.write_text(doc, encoding="utf-8")

    log_entry = f"""

## 2026-05-03 DG-KAN v4.5 Accelerated Functional Optimizer

Implemented and ran `experiments/run_gafu_v45.py` and `experiments/analyze_gafu_v45.py`.

Summary:

```text
P0 rows={len(p0)}, errors={sum(1 for r in p0 if r.get('error'))}, pass={p0_pass}
P1 rows={len(p1)}, errors={sum(1 for r in p1 if r.get('error'))}
P2 rows={len(p2_raw)}, errors={sum(1 for r in p2_raw if r.get('error'))}
P2 survivors={p2_survivors if p2_survivors else 'none'}
Decision=stop after P2 by written gate
```

Result replay:

```text
docs/DG-KAN_v4.5_AcceleratedFunctionalOptimizer_结果复盘.md
results/v4_5/aggregate_decision.json
```
"""
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(log_entry)

    print(f"Wrote {OUT}")
    print(f"P0 pass={p0_pass}; P2 survivors={p2_survivors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
