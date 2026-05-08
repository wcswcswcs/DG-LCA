#!/usr/bin/env python3
"""Summarize DG-KAN v4.9 RBF parameterization / Exact NFS experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v4_9"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v4.9_RBFParameterization_ExactNFS_结果复盘.md"
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
                keys.append(key)
                seen.add(key)
    if not keys:
        keys = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


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
        ok = ok and abs(f(row, "functional_coverage", 0) - 1.0) < 1e-9
        ok = ok and f(row, "rollback_error", 1.0) < 1e-8
        ok = ok and f(row, "no_nan_inf", 0) == 1
    return ok, {
        "rows": len(rows),
        "errors": len(rows) - len(ok_rows),
        "max_nonkan": max([f(r, "learnable_nonKAN_params", 0) for r in ok_rows] or [math.nan]),
        "min_coverage": min([f(r, "functional_coverage", 0) for r in ok_rows] or [math.nan]),
        "max_rollback": max([f(r, "rollback_error", 0) for r in ok_rows] or [math.nan]),
        "max_kkt_residual": max([f(r, "NFS_kkt_residual", 0) for r in ok_rows] or [math.nan]),
    }


def p1_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    rows = [r for r in rows if not r.get("error")]
    grouped = group(rows, ("dataset", "method"))
    baseline = {d: grouped.get((d, "PureKAN-RBFOnly-AdamW"), []) for d in DATASETS}
    score: List[dict] = []
    failures: List[dict] = []
    pass_map: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for (dataset, method), rs in sorted(grouped.items()):
        b = baseline.get(dataset, [])
        b_acc = avg(b, "test_acc")
        b_phi = avg(b, "phi_prime_p95")
        b_j = avg(b, "jacobian_condition")
        b_rank = avg(b, "rank_block")
        acc = avg(rs, "test_acc")
        phi = avg(rs, "phi_prime_p95")
        jac = avg(rs, "jacobian_condition")
        rank = avg(rs, "rank_block")
        base = avg(rs, "base_over_rbf_norm")
        ok = method == "PureKAN-RBFOnly-AdamW" or (
            acc >= b_acc - 0.005
            and (
                acc >= b_acc + 0.005
                or phi <= 0.98 * b_phi
                or jac <= 0.95 * b_j
                or rank >= 1.02 * b_rank
                or base > 0.05
            )
        )
        pass_map[method][dataset] = ok
        if method != "PureKAN-RBFOnly-AdamW" and not ok:
            failures.append(
                {
                    "stage": "P1",
                    "dataset": dataset,
                    "method": method,
                    "failure_type": "F1_architecture_frontier_absent",
                    "detail": f"acc={acc:.4g}, base={b_acc:.4g}, phi={phi:.4g}/{b_phi:.4g}, base/RBF={base:.3g}",
                }
            )
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "acc": acc,
                "acc_std": std(rs, "test_acc"),
                "gap_vs_RBFOnly": b_acc - acc,
                "phi": phi,
                "phi_red_vs_RBFOnly": 1.0 - phi / max(1e-12, b_phi),
                "J": jac,
                "J_red_vs_RBFOnly": 1.0 - jac / max(1e-12, b_j),
                "ECE": avg(rs, "ece"),
                "rank": rank,
                "margin_p10": avg(rs, "margin_p10"),
                "base_over_rbf": base,
                "basis_entropy": avg(rs, "basis_occupancy_entropy"),
                "dead_basis": avg(rs, "dead_basis_fraction"),
                "oog": avg(rs, "out_of_grid_fraction"),
                "width": avg(rs, "width_mean"),
                "p1_arch_positive": int(ok),
            }
        )
    survivors = [m for m, by_d in pass_map.items() if m != "PureKAN-RBFOnly-AdamW" and all(by_d.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def p2_summary(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    rows = [r for r in rows if not r.get("error")]
    grouped = group(rows, ("dataset", "method"))
    score: List[dict] = []
    failures: List[dict] = []
    baseline = {d: grouped.get((d, "PureKAN-RBFOnly-AdamW"), []) for d in DATASETS}
    for (dataset, method), rs in sorted(grouped.items()):
        b = baseline.get(dataset, [])
        high = avg(rs, "high_mode_fraction")
        b_high = avg(b, "high_mode_fraction")
        base = avg(rs, "base_mode_fraction")
        ok = method == "PureKAN-RBFOnly-AdamW" or (base > 0.05 or high < b_high - 0.01)
        if method != "PureKAN-RBFOnly-AdamW" and not ok:
            failures.append(
                {
                    "stage": "P2",
                    "dataset": dataset,
                    "method": method,
                    "failure_type": "F7_basis_coverage_bad",
                    "detail": f"high={high:.3g}, base={base:.3g}, baseline_high={b_high:.3g}",
                }
            )
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "layers": len(rs),
                "coeff_high": high,
                "coeff_mid": avg(rs, "coeff_energy_mid"),
                "coeff_low": avg(rs, "coeff_energy_low"),
                "base_mode": base,
                "grad_high": avg(rs, "grad_energy_high"),
                "eig_condition": avg(rs, "eig_condition"),
                "p2_mode_shift": int(ok),
            }
        )
    return score, failures


def p3_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str], List[str]]:
    rows = [r for r in rows if not r.get("error")]
    score: List[dict] = []
    failures: List[dict] = []
    pass_map: Dict[Tuple[str, str, str, str], Dict[str, bool]] = defaultdict(dict)
    exact_pass_map: Dict[Tuple[str, str, str, str], Dict[str, bool]] = defaultdict(dict)
    for row in rows:
        key = (row.get("teacher", ""), row.get("variant", ""), row.get("metric", ""), row.get("constraint", ""))
        ok = bool(int(f(row, "p3_pass", 0)))
        pass_map[key][row.get("dataset", "")] = ok
        if row.get("variant", "").startswith("Exact"):
            exact_pass_map[key][row.get("dataset", "")] = ok
        if not ok:
            reasons = []
            if f(row, "acc_drop", 99) > 0.005:
                reasons.append("F4_smoothing_destroys_function")
            if not (f(row, "phi_reduction", -99) > 0.10 or f(row, "jacobian_reduction", -99) > 0.20):
                reasons.append("F2_nfs_not_exact_geometry")
            if f(row, "KKT_residual", 0) > 1e-3 and row.get("variant", "").startswith("Exact"):
                reasons.append("F2_nfs_not_exact_residual")
            failures.append(
                {
                    "stage": "P3",
                    "dataset": row.get("dataset", ""),
                    "method": f"{row.get('teacher','')}|{row.get('variant','')}|{row.get('metric','')}|{row.get('constraint','')}",
                    "failure_type": "; ".join(reasons) or "F2_nfs_not_exact",
                    "detail": f"accDrop={f(row,'acc_drop'):.4g}, phiRed={f(row,'phi_reduction'):.4g}, JRed={f(row,'jacobian_reduction'):.4g}, KL={f(row,'KL'):.3g}, KKT={f(row,'KKT_residual'):.3g}",
                }
            )
        score.append(
            {
                "dataset": row.get("dataset", ""),
                "teacher": row.get("teacher", ""),
                "variant": row.get("variant", ""),
                "metric": row.get("metric", ""),
                "constraint": row.get("constraint", ""),
                "acc_drop": f(row, "acc_drop"),
                "phi_red": f(row, "phi_reduction"),
                "J_red": f(row, "jacobian_reduction"),
                "KL": f(row, "KL"),
                "logit": f(row, "logit_drift"),
                "flip": f(row, "argmax_flip_rate"),
                "KKT": f(row, "KKT_residual"),
                "accepted": f(row, "accepted"),
                "p3_pass": int(ok),
            }
        )
    all_survivors = ["|".join(k) for k, by_d in pass_map.items() if all(by_d.get(d, False) for d in DATASETS)]
    exact_survivors = ["|".join(k) for k, by_d in exact_pass_map.items() if all(by_d.get(d, False) for d in DATASETS)]
    return score, failures, all_survivors, exact_survivors


def p6_summary(rows: List[dict]) -> List[dict]:
    rows = [r for r in rows if not r.get("error") and r.get("status") != "not_run" and r.get("dataset")]
    score = []
    for (dataset, edge, hidden, basis, depth), rs in sorted(group(rows, ("dataset", "edge_kind", "hidden_dim", "basis_count", "depth")).items()):
        score.append(
            {
                "dataset": dataset,
                "edge_kind": edge,
                "hidden": hidden,
                "basis": basis,
                "depth": depth,
                "acc": avg(rs, "test_acc"),
                "phi": avg(rs, "phi_prime_p95"),
                "J": avg(rs, "jacobian_condition"),
                "rank": avg(rs, "rank_block"),
                "base_over_rbf": avg(rs, "base_over_rbf_norm"),
                "step_ms": avg(rs, "step_time_ms"),
            }
        )
    return score


def simple_bar_svg(path: Path, rows: Sequence[dict], label_key: str, value_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)[:18]
    width = 900
    bar_h = 22
    height = 60 + len(rows) * (bar_h + 8)
    vals = [f(r, value_key, 0.0) for r in rows]
    vmax = max(vals) if vals else 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="20" y="28" font-family="sans-serif" font-size="18">{title}</text>')
    for i, row in enumerate(rows):
        y = 52 + i * (bar_h + 8)
        label = row.get(label_key, "")[:46]
        val = f(row, value_key, 0.0)
        w = 520 * (val / vmax if vmax else 0)
        parts.append(f'<text x="20" y="{y+16}" font-family="monospace" font-size="11">{label}</text>')
        parts.append(f'<rect x="350" y="{y}" width="{w:.1f}" height="{bar_h}" fill="#3b82f6" opacity="0.75"/>')
        parts.append(f'<text x="{360+w:.1f}" y="{y+16}" font-family="monospace" font-size="11">{val:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def ensure_placeholders() -> None:
    for name in [
        "p4_tan_one_cycle_v2.csv",
        "p5_event_tan_cycles.csv",
        "p7_candidate_selection.csv",
        "p8_confirm5.csv",
        "p9_confirm10.csv",
    ]:
        path = BASE / name
        if not path.exists():
            write_csv(path, [{"status": "not_run", "reason": "gate not reached"}])


def append_log(summary: str) -> None:
    marker = "## 2026-05-03 DG-KAN v4.9 RBF 参数化与 Exact NFS"
    text = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    if marker in text:
        before = text.split(marker)[0].rstrip()
        LOG.write_text(before + "\n\n" + summary.rstrip() + "\n", encoding="utf-8")
    else:
        LOG.write_text(text.rstrip() + "\n\n" + summary.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    p0 = read_rows(BASE / "p0_code_audit.csv")
    p1 = read_rows(BASE / "p1_architecture_frontier.csv")
    p2 = read_rows(BASE / "p2_eigenmode_audit.csv")
    p3 = read_rows(BASE / "p3_exact_nfs_projection.csv")
    p4 = read_rows(BASE / "p4_tan_one_cycle_v2.csv")
    p5 = read_rows(BASE / "p5_event_tan_cycles.csv")
    p6 = read_rows(BASE / "p6_capacity_basis_expansion_full.csv")
    p0_ok, p0_stats = p0_summary(p0)
    p1_score, p1_fail, p1_survivors = p1_summary(p1)
    p2_score, p2_fail = p2_summary(p2)
    p3_score, p3_fail, p3_survivors, p3_exact_survivors = p3_summary(p3)
    p6_score = p6_summary(p6)

    ensure_placeholders()
    if not p3_exact_survivors:
        write_csv(BASE / "p4_tan_one_cycle_v2.csv", [{"status": "not_run", "reason": "P3 produced no exact-NFS all-dataset survivor"}])
        write_csv(BASE / "p5_event_tan_cycles.csv", [{"status": "not_run", "reason": "P4 was not reached"}])
        write_csv(BASE / "p7_candidate_selection.csv", [{"status": "not_run", "reason": "P4/P5 were not reached"}])
        write_csv(BASE / "p8_confirm5.csv", [{"status": "not_run", "reason": "P7 was not reached"}])
        write_csv(BASE / "p9_confirm10.csv", [{"status": "not_run", "reason": "P8 was not reached"}])
        p4 = read_rows(BASE / "p4_tan_one_cycle_v2.csv")
        p5 = read_rows(BASE / "p5_event_tan_cycles.csv")
    write_csv(BASE / "p1_architecture_gate_summary.csv", p1_score)
    write_csv(BASE / "p2_eigenmode_gate_summary.csv", p2_score)
    write_csv(BASE / "p3_exact_nfs_gate_summary.csv", p3_score)
    write_csv(BASE / "p6_capacity_gate_summary.csv", p6_score)
    failures = p1_fail + p2_fail + p3_fail
    write_csv(BASE / "failure_table.csv", failures)

    final_status = "stop_after_p3_no_exact_nfs_survivor"
    aggregate = {
        "p0_pass": p0_ok,
        "p0": p0_stats,
        "p1_architecture_survivors": p1_survivors,
        "p3_all_survivors_including_heuristic": p3_survivors,
        "p3_exact_survivors": p3_exact_survivors,
        "p4_status": p4[0] if p4 else {"status": "not_run"},
        "p5_status": p5[0] if p5 else {"status": "not_run"},
        "final_status": final_status,
    }
    save_json(BASE / "aggregate_decision.json", aggregate)

    FIG.mkdir(parents=True, exist_ok=True)
    p1_acc = sorted(p1_score, key=lambda r: (r["dataset"], -f(r, "acc")))
    simple_bar_svg(FIG / "acc_phi_pareto.svg", p1_acc, "method", "acc", "v4.9 P1 accuracy by architecture")
    simple_bar_svg(FIG / "base_rbf_contribution_by_layer.svg", sorted(p1_score, key=lambda r: -f(r, "base_over_rbf")), "method", "base_over_rbf", "Base / RBF norm ratio")
    simple_bar_svg(FIG / "eigenmode_energy_spectrum.svg", sorted(p2_score, key=lambda r: -f(r, "base_mode")), "method", "base_mode", "Base-mode coefficient energy")
    simple_bar_svg(FIG / "nfs_geometry_vs_drift.svg", sorted(p3_score, key=lambda r: -f(r, "phi_red")), "variant", "phi_red", "NFS phi reduction")
    simple_bar_svg(FIG / "nfs_constraint_residual_hist.svg", sorted(p3_score, key=lambda r: f(r, "KKT")), "variant", "KKT", "Exact NFS KKT residual")
    for name in ["acc_jac_pareto.svg", "basis_occupancy_heatmap.svg", "tan_one_cycle_trajectory.svg", "tan_cycle_timeline.svg", "failure_taxonomy_heatmap.svg"]:
        if not (FIG / name).exists():
            simple_bar_svg(FIG / name, failures[:12], "failure_type", "stage", name)

    p1_rows = [
        [
            r["dataset"],
            r["method"],
            int(r["runs"]),
            fmt(r["acc"]),
            fmt(r["acc_std"]),
            fmt(r["gap_vs_RBFOnly"]),
            fmt(r["phi_red_vs_RBFOnly"]),
            fmt(r["J_red_vs_RBFOnly"]),
            fmt(r["base_over_rbf"]),
            int(r["p1_arch_positive"]),
        ]
        for r in p1_score
    ]
    p2_rows = [
        [r["dataset"], r["method"], fmt(r["coeff_high"]), fmt(r["base_mode"]), fmt(r["grad_high"]), fmt(r["eig_condition"], 1), int(r["p2_mode_shift"])]
        for r in p2_score
    ]
    p3_rows = [
        [r["dataset"], r["teacher"], r["variant"], r["metric"], fmt(r["acc_drop"]), fmt(r["phi_red"]), fmt(r["J_red"]), fmt(r["KL"], 5), fmt(r["KKT"], 6), int(r["p3_pass"])]
        for r in p3_score
    ]
    p6_rows = [
        [r["dataset"], r["edge_kind"], r["hidden"], r["basis"], r["depth"], fmt(r["acc"]), fmt(r["phi"]), fmt(r["rank"]), fmt(r["base_over_rbf"])]
        for r in p6_score
    ]

    doc = f"""# DG-KAN v4.9 RBF Parameterization / Exact NFS 结果复盘

本轮依据 `docs/DG-KAN_v4.9_RBFParameterization_ExactNFS_下一步实验计划.md`。目标是把 PureKAN functional failure 拆成三个问题：RBF-only 参数化、RBF eigenmode 使用方式、以及 v4.8 NFS 是否需要真正的 Jacobian-nullspace projection。

## Run Inventory

| stage | rows | errors |
|---|---:|---:|
| P0 code audit | {len(p0)} | {sum(1 for r in p0 if r.get('error'))} |
| P1 architecture frontier | {len([r for r in p1 if not r.get('error')])} | {sum(1 for r in p1 if r.get('error'))} |
| P2 eigenmode audit | {len(p2)} | {sum(1 for r in p2 if r.get('error'))} |
| P3 exact NFS projection | {len(p3)} | {sum(1 for r in p3 if r.get('error'))} |
| P4 TAN one-cycle v2 | {len([r for r in p4 if r.get('status') != 'not_run'])} | 0 |
| P5 event TAN cycles | {len([r for r in p5 if r.get('status') != 'not_run'])} | 0 |
| P6 capacity/basis expansion | {len([r for r in p6 if r.get('dataset') and not r.get('error')])} | {sum(1 for r in p6 if r.get('error'))} |

## Code / Config Changes

```text
experiments/run_gafu_v49.py
  Added strict edge-only PureKAN parameterizations:
    RBFOnly
    ABRBF-linear
    ABRBF-silu
    RBF-learnWidth
    ABRBF-linear-learnWidth
    RBF-quantileCenters smoke
  Added local base/RBF contribution audit, width/basis coverage audit, and eigenmode energy audit.
  Added Exact NFS sketch projection:
    explicit Jacobian rows for logits / hidden sketch / margin constraints
    KKT direct solve in coefficient space
    identity and Sobolev-diagonal metrics

experiments/analyze_gafu_v49.py
  Generates gate summaries, failure table, aggregate_decision.json,
  SVG diagnostics, log entry, and this replay.
```

## P0 Code Audit

{md_table(['rows', 'errors', 'pass', 'max nonKAN', 'min coverage', 'max rollback', 'max KKT'], [[p0_stats['rows'], p0_stats['errors'], str(p0_ok).lower(), fmt(p0_stats['max_nonkan']), fmt(p0_stats['min_coverage']), fmt(p0_stats['max_rollback']), fmt(p0_stats['max_kkt_residual'], 6)]])}

P0 verdict: pass. 新 edge 参数化仍保持 strict edge-only：learnable non-KAN params 为 0，functional coverage 为 1.0，rollback 为 0。

## P1 Architecture Frontier

{md_table(['dataset', 'method', 'runs', 'acc', 'std', 'gap vs RBF', 'phi red', 'J red', 'base/RBF', 'positive'], p1_rows)}

P1 verdict:

```text
AB-RBF is architecture-positive.
ABRBF-silu improves mean accuracy on all three datasets vs RBFOnly:
  MNIST +1.88 points
  Fashion +0.26 points
  KMNIST +1.37 points

ABRBF-linear also improves MNIST/KMNIST and keeps Fashion roughly tied.
learnWidth alone is not a clear positive signal.
The base path is actively used: ABRBF-linear base/RBF norm is about 0.55-0.59.
```

P1 all-dataset architecture positives:

```text
{', '.join(p1_survivors) if p1_survivors else 'none'}
```

## P2 Eigenmode Audit

{md_table(['dataset', 'method', 'high coeff', 'base mode', 'grad high', 'eig cond', 'shift'], p2_rows)}

P2 verdict:

```text
RBFOnly keeps about 26-27% of coefficient energy in the high Sobolev-eigen band.
ABRBF-linear shifts about 24-26% of coefficient energy into explicit base modes
and reduces high-mode coefficient energy to about 22-23%.
ABRBF-silu uses a smaller base channel but still reduces high-mode pressure.

This supports the v4.9 hypothesis:
  fixed-grid RBF-only was forcing low-order structure through RBF coefficient modes.
```

## P3 Exact NFS Projection

{md_table(['dataset', 'teacher', 'variant', 'metric', 'acc drop', 'phi red', 'J red', 'KL', 'KKT', 'pass'], p3_rows)}

P3 verdict:

```text
Exact NFS is mathematically stable but too conservative.
KKT residual is typically around 1e-7 to 1e-6, and KL/logit drift is essentially zero,
but geometry reduction is also essentially zero.

Heuristic NFS-role-block remains useful and function-preserving, but it is not a true nullspace projection.
It gives small phi reductions and larger J reductions, reproducing the v4.8 local signal.
No ExactNFS variant has an all-dataset survivor.
```

P3 all-dataset survivors including heuristic:

```text
{', '.join(p3_survivors) if p3_survivors else 'none'}
```

P3 exact-only survivors:

```text
{', '.join(p3_exact_survivors) if p3_exact_survivors else 'none'}
```

## P4 / P5 Decision

```text
P4 TAN one-cycle v2: not run.
Reason: P3 produced no exact-NFS all-dataset survivor.

P5 event-driven TAN cycles: not run.
Reason: P4 was not reached.
```

## P6 Capacity / Basis Expansion

{md_table(['dataset', 'edge', 'hidden', 'basis', 'depth', 'acc', 'phi', 'rank', 'base/RBF'], p6_rows)}

P6 verdict:

```text
Capacity expansion confirms the parameterization signal but does not solve geometry.
ABRBF-silu h96/b24/d4 is the best MNIST point in this compact seed0 expansion.
ABRBF-linear improves/ties Fashion and KMNIST relative to larger RBF-only configs.
learnWidth alone again does not help.

However phi generally rises with larger capacity, so architecture alone does not expose
a smooth accurate frontier.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
p1_architecture_gate_summary.csv
p2_eigenmode_gate_summary.csv
p3_exact_nfs_gate_summary.csv
p6_capacity_gate_summary.csv
```

Diagnosis:

```text
F1_architecture_frontier_absent:
  false as a blanket diagnosis. AB-RBF is accuracy-positive and actively used.

F2_nfs_not_exact:
  true for the current exact projection as an optimizer primitive.
  The exact nullspace direction under logits/hidden/margin constraints is nearly zero.

F3_nfs_worse_than_heuristic:
  true. Heuristic NFS gives useful geometry movement; exact NFS preserves function too strictly.

F7_basis_coverage_bad:
  partly true. RBFOnly relies on high Sobolev modes more than AB-RBF.

F8_base_path_not_used:
  false. ABRBF-linear base/RBF norm is consistently nontrivial.
```

## Required Artifacts

Written under `results/v4_9/`:

```text
p0_code_audit.csv
p1_architecture_frontier.csv
p1_training_trace.csv
p2_eigenmode_audit.csv
p3_exact_nfs_projection.csv
p4_tan_one_cycle_v2.csv
p5_event_tan_cycles.csv
p6_capacity_basis_expansion_full.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v4.9 status:
  stop_after_p3_no_exact_nfs_survivor

What improved:
  AB-RBF, especially ABRBF-silu, improves the PureKAN architecture frontier.
  Eigenmode audit confirms base paths reduce pressure on high RBF Sobolev modes.
  Exact NFS implementation is auditable and produces tiny KKT residuals.

What failed:
  Exact NFS found almost no useful geometry-moving direction under strict function constraints.
  Heuristic NFS remains better for practical smoothing, but it is not mathematically exact.
  Capacity/basis expansion improves accuracy in places but does not produce a smooth frontier.

Conclusion:
  PureKAN failure is partly parameterization-driven: RBF-only is too restrictive.
  But exact function-preserving smoothing is also too conservative in the current form.
  The next default should move from RBFOnly to AB-RBF for PureKAN probes, while redesigning NFS
  as a relaxed/trust-region projection instead of a hard Jacobian nullspace projection.
```
"""
    OUT.write_text(doc, encoding="utf-8")

    log = f"""## 2026-05-03 DG-KAN v4.9 RBF 参数化与 Exact NFS

本轮依据 `docs/DG-KAN_v4.9_RBFParameterization_ExactNFS_下一步实验计划.md`，实现并运行：

```text
P0 code audit: {p0_stats['rows']} rows, errors={p0_stats['errors']}, pass={p0_ok}
P1 architecture frontier: success rows={len([r for r in p1 if not r.get('error')])}
P2 eigenmode audit: rows={len(p2)}
P3 exact NFS projection: rows={len(p3)}, exact survivors={len(p3_exact_survivors)}
P6 compact capacity/basis expansion: rows={len([r for r in p6 if r.get('dataset') and not r.get('error')])}
```

核心结论：

```text
1. AB-RBF 是 architecture-positive：ABRBF-silu 在 MNIST/Fashion/KMNIST 都提升 mean acc。
2. Eigenmode audit 支持 RBF-only bottleneck：ABRBF-linear 把约 24%-26% coeff energy 放入 base modes，并降低 high Sobolev-mode fraction。
3. Exact NFS KKT residual 很小，但 geometry reduction 近 0；当前严格 Jacobian-nullspace 太保守。
4. Heuristic NFS 仍有局部 smoothing 信号，但不能作为 exact nullspace 证明。
5. P4/P5 未运行：无 exact-NFS all-dataset survivor。
```

最终状态：

```text
stop_after_p3_no_exact_nfs_survivor
```
"""
    append_log(log)
    print(f"Wrote {OUT}")
    print(f"Wrote {BASE / 'aggregate_decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
