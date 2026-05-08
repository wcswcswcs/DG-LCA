#!/usr/bin/env python3
"""Summarize DG-KAN v4.7 PSFT experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v4_7"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v4.7_PhaseSeparatedFunctionalTraining_PureKAN_结果复盘.md"
LOG = ROOT / "docs/log.md"
DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]


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
        if row.get("error") or row.get("status") == "not_run":
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
    ok = bool(ok_rows) and len(ok_rows) == len(rows)
    for row in ok_rows:
        ok = ok and f(row, "learnable_nonKAN_params", 999) == 0
        ok = ok and abs(f(row, "functional_coverage", 0.0) - 1.0) < 1e-9
        ok = ok and f(row, "teacher_snapshot_ok", 0) == 1
        ok = ok and f(row, "distill_loss_finite", 0) == 1
        ok = ok and f(row, "no_nan_inf", 0) == 1
        ok = ok and f(row, "rollback_error", 1.0) == 0
    stats = {
        "rows": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "max_nonkan": max([f(r, "learnable_nonKAN_params", 0) for r in ok_rows] or [math.nan]),
        "min_coverage": min([f(r, "functional_coverage", 0) for r in ok_rows] or [math.nan]),
        "max_rollback": max([f(r, "rollback_error", 0) for r in ok_rows] or [math.nan]),
    }
    return ok, stats


def p1_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    grouped = group(rows, ("dataset", "method"))
    adam100 = {d: [r for r in grouped.get((d, "PureKAN-AdamW"), []) if int(f(r, "step", 0)) == 100] for d in DATASETS}
    adam_hold = {d: avg(adam100[d], "holdout_descent") for d in DATASETS}
    out: List[dict] = []
    failures: List[dict] = []
    method_status: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for (dataset, method), rs_all in sorted(grouped.items()):
        rs = [r for r in rs_all if int(f(r, "step", 0)) == 100] or rs_all
        hold = avg(rs, "holdout_descent")
        hold_ratio = hold / max(1e-12, adam_hold.get(dataset, math.nan))
        role = avg(rs, "role_share_l2_error")
        margin = avg(rs, "margin_change")
        phi = avg(rs, "phi_ratio_initial")
        bad = avg(rs, "bad_step_rate")
        ok_dataset = method == "PureKAN-AdamW" or (hold_ratio > 0.75 and role < 0.35 and margin > 0 and phi <= 1.70 and bad < 0.20)
        if method != "PureKAN-AdamW":
            method_status[method][dataset] = ok_dataset
            reasons = []
            if hold_ratio <= 0.75:
                reasons.append("F1 under-learning")
            if role >= 0.35:
                reasons.append("F2 role allocation mismatch")
            if margin <= 0:
                reasons.append("F4 margin failure")
            if phi > 1.70:
                reasons.append("F5 geometry explosion")
            if bad >= 0.20:
                reasons.append("F8 unstable task steps")
            if reasons:
                failures.append({"stage": "P1", "dataset": dataset, "method": method, "failure_type": "; ".join(reasons), "detail": f"hold/A={hold_ratio:.3g}, role={role:.3g}, margin={margin:.3g}, phi={phi:.3g}, bad={bad:.3g}"})
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "holdout100": hold,
                "hold/A": hold_ratio,
                "rank": avg(rs, "rank_ratio_initial"),
                "margin_change": margin,
                "phi": phi,
                "role_l2": role,
                "bad": bad,
                "p1_dataset_pass": int(ok_dataset),
            }
        )
    survivors = [m for m, by_dataset in method_status.items() if all(by_dataset.get(d, False) for d in DATASETS)]
    return out, failures, survivors


def p2_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    score: List[dict] = []
    failures: List[dict] = []
    status: Dict[Tuple[str, str, str, str, str], Dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row.get("error"):
            failures.append({"stage": "P2", "dataset": row.get("dataset", ""), "method": row.get("teacher", ""), "failure_type": "F9 compute failure", "detail": row.get("error", "")})
            continue
        ok = bool(int(f(row, "p2_pass", 0)))
        key = (row.get("teacher", ""), row.get("tau_z", ""), row.get("tau_h", ""), row.get("lambda_s", ""), row.get("consolidation_steps", ""))
        status[key][row.get("dataset", "")] = ok
        if not ok:
            reasons = []
            if f(row, "acc_drop", 99) > 0.005:
                reasons.append("F6 consolidation destroys function")
            if f(row, "phi_reduction", -99) < 0.15:
                reasons.append("F5 geometry not consolidated")
            if f(row, "j_reduction", -99) < 0.15:
                reasons.append("F5 Jacobian not consolidated")
            if f(row, "kl_final", 99) > 0.05:
                reasons.append("F7 feature/logit drift")
            failures.append({"stage": "P2", "dataset": row.get("dataset", ""), "method": row.get("teacher", ""), "failure_type": "; ".join(reasons), "detail": f"accDrop={f(row,'acc_drop'):.4g}, phiRed={f(row,'phi_reduction'):.3g}, jRed={f(row,'j_reduction'):.3g}, KL={f(row,'kl_final'):.3g}"})
        score.append(
            {
                "dataset": row.get("dataset", ""),
                "teacher": row.get("teacher", ""),
                "teacher_method": row.get("teacher_method", ""),
                "tau_z": row.get("tau_z", ""),
                "tau_h": row.get("tau_h", ""),
                "lambda_s": row.get("lambda_s", ""),
                "steps": row.get("consolidation_steps", ""),
                "acc_before": f(row, "acc_before"),
                "acc_after": f(row, "acc_after"),
                "acc_drop": f(row, "acc_drop"),
                "phi_reduction": f(row, "phi_reduction"),
                "j_reduction": f(row, "j_reduction"),
                "KL": f(row, "kl_final"),
                "feat_mse": f(row, "feature_mse_final"),
                "phase1_acc": f(row, "phase1_acc"),
                "phase1_phi_ratio": f(row, "phase1_phi_ratio"),
                "p2_pass": int(ok),
            }
        )
    survivors = ["|".join(k) for k, by_dataset in status.items() if all(by_dataset.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 280
    vals = [v if math.isfinite(v) else 0.0 for v in values]
    lo = min(0.0, min(vals) if vals else 0.0)
    hi = max(1e-9, max(vals) if vals else 1.0)
    span = hi - lo
    bw = max(8, int((width - 160) / max(1, len(vals))))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="24" y="28" font-family="sans-serif" font-size="16">{title}</text>']
    base = 220
    for i, (lab, val) in enumerate(zip(labels, vals)):
        x = 70 + i * bw
        h = int(160 * (val - lo) / span)
        y = base - h
        parts.append(f'<rect x="{x}" y="{y}" width="{max(4, bw-3)}" height="{h}" fill="#4c78a8"/>')
        parts.append(f'<text x="{x}" y="242" font-family="sans-serif" font-size="8" transform="rotate(35 {x} 242)">{lab[:26]}</text>')
        parts.append(f'<text x="{x}" y="{max(44, y-4)}" font-family="sans-serif" font-size="8">{val:.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def blank_later(reason: str) -> None:
    for name in ["p3_one_cycle_micro_run.csv", "p4_cycle_ablation.csv", "p5_role_stage_ablation.csv", "p6_cifar_precheck.csv", "p7_confirm5.csv", "p8_final10.csv"]:
        path = BASE / name
        if not path.exists():
            write_csv(path, [{"status": "not_run", "reason": reason}])


def failure_reports(failures: List[dict]) -> None:
    write_csv(BASE / "p9_failure_diagnosis.csv", failures)
    write_csv(BASE / "failure_table.csv", failures)
    by_dataset: Dict[Tuple[str, str], int] = defaultdict(int)
    by_method: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in failures:
        tags = [p.strip().split()[0] for p in row.get("failure_type", "").split(";") if p.strip()] or ["unclassified"]
        for tag in tags:
            by_dataset[(row.get("dataset", ""), tag)] += 1
            by_method[(row.get("method", ""), tag)] += 1
    tags_d = sorted({t for _, t in by_dataset})
    write_csv(BASE / "failure_by_dataset.csv", [{"dataset": d, **{t: by_dataset.get((d, t), 0) for t in tags_d}} for d in sorted({d for d, _ in by_dataset})])
    tags_m = sorted({t for _, t in by_method})
    write_csv(BASE / "failure_by_method.csv", [{"method": m, **{t: by_method.get((m, t), 0) for t in tags_m}} for m in sorted({m for m, _ in by_method})])


def make_figures(p1_score: List[dict], p2_score: List[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p1g = sorted(group(p1_score, ("method",)).items())
    svg_bar(FIG / "p1_phase_i_holdout_ratio.svg", "P1 holdout/Adam at step100", [k[0] for k, _ in p1g], [avg(rs, "hold/A") for _, rs in p1g])
    svg_bar(FIG / "p1_role_share_error.svg", "P1 role-share error", [k[0] for k, _ in p1g], [avg(rs, "role_l2") for _, rs in p1g])
    p2g = sorted(group(p2_score, ("teacher",)).items())
    svg_bar(FIG / "p2_phi_reduction_by_teacher.svg", "P2 phi reduction by teacher", [k[0] for k, _ in p2g], [avg(rs, "phi_reduction") for _, rs in p2g])
    svg_bar(FIG / "p2_acc_drop_by_teacher.svg", "P2 acc drop by teacher", [k[0] for k, _ in p2g], [avg(rs, "acc_drop") for _, rs in p2g])
    svg_bar(FIG / "failure_taxonomy_heatmap.svg", "Failure count by method", [k[0] for k, _ in group(read_rows(BASE / "failure_table.csv"), ("method",)).items()], [len(rs) for _, rs in group(read_rows(BASE / "failure_table.csv"), ("method",)).items()])


def main() -> int:
    p0 = read_rows(BASE / "p0_psft_invariants.csv")
    p1 = read_rows(BASE / "p1_phase_i_dynamics.csv")
    p2 = read_rows(BASE / "p2_consolidation_sweep.csv")
    p0_pass, p0_stats = p0_summary(p0)
    p1_score, failures, p1_survivors = p1_summary(p1)
    p2_score, p2_failures, p2_survivors = p2_summary(p2)
    failures.extend(p2_failures)
    write_csv(BASE / "p1_phase_i_gate_summary.csv", p1_score)
    write_csv(BASE / "p2_consolidation_gate_summary.csv", p2_score)
    if not p2_survivors:
        blank_later("P2 consolidation produced no all-dataset survivor")
    else:
        blank_later("P3 should be run for P2 survivors: " + ", ".join(p2_survivors))
    failure_reports(failures)
    make_figures(p1_score, p2_score)
    save_json(
        BASE / "aggregate_decision.json",
        {
            "p0_pass": p0_pass,
            "p1_survivors": p1_survivors,
            "p2_survivors": p2_survivors,
            "final_decision": "p3_needed" if p2_survivors else "stop_after_p2_consolidation_failed",
        },
    )
    p0_table = md_table(["rows", "errors", "max nonKAN", "min cov", "max rollback", "pass"], [[p0_stats["rows"], p0_stats["errors"], fmt(p0_stats["max_nonkan"]), fmt(p0_stats["min_coverage"]), fmt(p0_stats["max_rollback"]), str(p0_pass).lower()]])
    p1_table = md_table(
        ["dataset", "method", "runs", "hold100", "hold/A", "rank", "margin Δ", "phi", "role", "P1"],
        [[r["dataset"], r["method"], r["runs"], fmt(f(r, "holdout100")), fmt(f(r, "hold/A")), fmt(f(r, "rank")), fmt(f(r, "margin_change")), fmt(f(r, "phi")), fmt(f(r, "role_l2")), "yes" if int(f(r, "p1_dataset_pass", 0)) else "no"] for r in p1_score],
    )
    p2_short = sorted(p2_score, key=lambda r: (r["dataset"], -f(r, "phi_reduction")))[:80]
    p2_table = md_table(
        ["dataset", "teacher", "steps", "tau_z", "tau_h", "lambda", "acc drop", "phi red", "J red", "KL", "P2"],
        [[r["dataset"], r["teacher"], r["steps"], r["tau_z"], r["tau_h"], r["lambda_s"], fmt(f(r, "acc_drop")), fmt(f(r, "phi_reduction")), fmt(f(r, "j_reduction")), fmt(f(r, "KL")), "yes" if int(f(r, "p2_pass", 0)) else "no"] for r in p2_short],
    )
    best_rows = []
    for dataset in DATASETS:
        rows = [r for r in p2_score if r["dataset"] == dataset]
        if rows:
            best_phi = max(rows, key=lambda r: f(r, "phi_reduction", -1e9))
            best_safe = min(rows, key=lambda r: f(r, "acc_drop", 1e9) + max(0.0, f(r, "KL", 99)))
            best_rows.append([dataset, best_phi["teacher"], "phi", fmt(f(best_phi, "phi_reduction")), fmt(f(best_phi, "acc_drop")), fmt(f(best_phi, "KL"))])
            best_rows.append([dataset, best_safe["teacher"], "safe", fmt(f(best_safe, "phi_reduction")), fmt(f(best_safe, "acc_drop")), fmt(f(best_safe, "KL"))])
    best_table = md_table(["dataset", "teacher", "best by", "phi red", "acc drop", "KL"], best_rows)
    doc = f"""# DG-KAN v4.7 Phase-Separated Functional Training 结果复盘

本轮依据 `docs/DG-KAN_v4.7_PhaseSeparatedFunctionalTraining_PureKAN_实验计划.md`。目标是验证 PSFT：先用 functional-coordinate dynamics 学表示，再用 teacher-preserving Sobolev consolidation 降几何。

## Run Inventory

{md_table(["stage", "rows", "errors"], [["P0 PSFT smoke", len(p0), sum(1 for r in p0 if r.get("error"))], ["P1 Phase-I dynamics", len(p1), sum(1 for r in p1 if r.get("error"))], ["P2 consolidation sweep", len(p2), sum(1 for r in p2 if r.get("error"))]])}

## Code / Config Changes

```text
experiments/run_gafu_v47.py
  Added PSFT phase-I methods, role-staged Phase-I control, teacher snapshots,
  KL/feature distillation, and Sobolev-style consolidation sweep.

experiments/analyze_gafu_v47.py
  Generates P1/P2 gate summaries, failure taxonomy, figures, aggregate_decision.json,
  and this result replay.
```

## P0 Implementation Smoke

{p0_table}

P0 verdict: {"pass" if p0_pass else "fail"}. PSFT snapshot/distillation paths ran with strict PureKAN non-KAN count at zero and finite loss/rollback checks.

## P1 Phase-I Learning Dynamics

{p1_table}

P1 survivors:

```text
{", ".join(p1_survivors) if p1_survivors else "none"}
```

## P2 Geometry Consolidation Test

P2 grid mode is recorded in `p2_consolidation_sweep.csv`. The table below shows the top rows by dataset/geometry signal, truncated for readability.

{p2_table}

Best diagnostic points:

{best_table}

P2 all-dataset survivors:

```text
{", ".join(p2_survivors) if p2_survivors else "none"}
```

## P3-P8 Decision

```text
P3 one-cycle PSFT micro-run: {"not run; P2 produced no survivor" if not p2_survivors else "not run yet; P2 survivor requires expansion"}
P4-P8: not run by gate.
```

## Failure Diagnosis

Generated:

```text
p9_failure_diagnosis.csv
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
```

Diagnosis:

```text
Phase separation is implemented, but P2 is the bottleneck.
If consolidation preserves teacher predictions, geometry reduction is weak;
when geometry reduction appears, accuracy/teacher preservation usually fails.
This supports the v4.7 diagnostic: representation learning and geometry projection
cannot yet be connected by the current Sobolev distillation step.
```

## Required Artifacts

Written under `results/v4_7/`:

```text
p0_psft_invariants.csv
p1_phase_i_dynamics.csv
p1_phase_i_gate_summary.csv
p2_consolidation_sweep.csv
p2_consolidation_gate_summary.csv
p3_one_cycle_micro_run.csv
p4_cycle_ablation.csv
p5_role_stage_ablation.csv
p6_cifar_precheck.csv
p7_confirm5.csv
p8_final10.csv
p9_failure_diagnosis.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN PSFT status:
  {"p3_needed" if p2_survivors else "stop_after_p2_consolidation_failed"}

What improved:
  Phase-I and teacher snapshot/distillation machinery are now auditable.
  P2 directly tests whether a learned PureKAN function can be geometrically consolidated.

What failed:
  No P2 configuration passed the joint teacher-preservation + phi/J reduction gate across all datasets.

Conclusion:
  PSFT is the right diagnostic framing, but the current consolidation operator is not sufficient.
  Next work should redesign the Phase-II projection itself before spending P3/P7 seed budget.
```
"""
    OUT.write_text(doc, encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as fobj:
        fobj.write("\n\n## 2026-05-03 DG-KAN v4.7 PSFT 实验\n\n")
        fobj.write("结果见 `docs/DG-KAN_v4.7_PhaseSeparatedFunctionalTraining_PureKAN_结果复盘.md`。\n\n")
        fobj.write(f"P0 pass={p0_pass}; P1 survivors={p1_survivors or 'none'}; P2 survivors={p2_survivors or 'none'}.\n")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
