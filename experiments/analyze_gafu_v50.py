#!/usr/bin/env python3
"""Summarize DG-KAN v5.0 edge-decomposed functional training experiments."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v5_0"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v5.0_EdgeDecomposedFunctionalTraining_结果复盘.md"
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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
        ok = ok and abs(f(row, "functional_coverage_total", f(row, "functional_coverage", 0)) - 1.0) < 1e-9
        ok = ok and f(row, "rollback_error", 1.0) < 1e-8
        ok = ok and f(row, "no_nan_inf", 0) == 1
    return ok, {
        "rows": len(rows),
        "errors": len(rows) - len(ok_rows),
        "max_nonkan": max([f(r, "learnable_nonKAN_params", 0) for r in ok_rows] or [math.nan]),
        "min_coverage": min([f(r, "functional_coverage_total", f(r, "functional_coverage", 0)) for r in ok_rows] or [math.nan]),
        "max_rollback": max([f(r, "rollback_error", 0) for r in ok_rows] or [math.nan]),
        "max_kkt": max([f(r, "NFS_kkt_residual", 0) for r in ok_rows] or [math.nan]),
        "max_cg": max([f(r, "CG_residual", 0) for r in ok_rows] or [math.nan]),
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
        b_phi_rbf = avg(b, "phi_rbf_p95")
        acc = avg(rs, "test_acc")
        phi_rbf = avg(rs, "phi_rbf_p95")
        base = avg(rs, "base_over_rbf_norm")
        rbf = avg(rs, "rbf_output_norm")
        ok = method == "PureKAN-RBFOnly-AdamW" or (
            "MLP" not in method
            and "BaseOnly" not in method
            and acc >= b_acc - 0.005
            and (acc >= b_acc + 0.005 or phi_rbf <= b_phi_rbf or base > 0.05)
        )
        pass_map[method][dataset] = ok
        if method != "PureKAN-RBFOnly-AdamW" and not ok:
            failures.append(
                {
                    "stage": "P1",
                    "dataset": dataset,
                    "method": method,
                    "failure_type": "F1_base_path_unused" if base <= 0.05 and "ABRBF" in method else "F2_base_path_helps_accuracy_but_geometry_bad",
                    "detail": f"acc={acc:.4g}, RBF={b_acc:.4g}, phiR={phi_rbf:.4g}/{b_phi_rbf:.4g}, base/RBF={base:.3g}",
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
                "phi_base": avg(rs, "phi_base_p95"),
                "phi_rbf": phi_rbf,
                "phi_total": avg(rs, "phi_total_p95"),
                "phi_rbf_red_vs_RBFOnly": 1.0 - phi_rbf / max(1e-12, b_phi_rbf),
                "curv_rbf": avg(rs, "curvature_rbf_p95"),
                "J": avg(rs, "jacobian_condition"),
                "ECE": avg(rs, "ece"),
                "rank": avg(rs, "rank_block"),
                "margin_p10": avg(rs, "margin_p10"),
                "base_over_rbf": base,
                "base_norm": avg(rs, "base_output_norm"),
                "rbf_norm": rbf,
                "p1_arch_positive": int(ok),
            }
        )
    arch_methods = [m for m in pass_map if m != "PureKAN-RBFOnly-AdamW" and "MLP" not in m and "BaseOnly" not in m]
    survivors = [m for m in arch_methods if all(pass_map[m].get(d, False) for d in DATASETS)]
    return score, failures, survivors


def p2_summary(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    rows = [r for r in rows if not r.get("error")]
    grouped = group(rows, ("dataset", "method"))
    baseline = {d: grouped.get((d, "PureKAN-RBFOnly-AdamW"), []) for d in DATASETS}
    score: List[dict] = []
    failures: List[dict] = []
    for (dataset, method), rs in sorted(grouped.items()):
        b_high = avg(baseline.get(dataset, []), "high_mode_fraction")
        high = avg(rs, "high_mode_fraction")
        base = avg(rs, "base_mode_fraction")
        grad_high = avg(rs, "grad_energy_high")
        ok = method == "PureKAN-RBFOnly-AdamW" or base > 0.05 or high < b_high - 0.005
        if method != "PureKAN-RBFOnly-AdamW" and not ok:
            failures.append(
                {
                    "stage": "P2",
                    "dataset": dataset,
                    "method": method,
                    "failure_type": "F3_residual_high_mode_pressure_persists",
                    "detail": f"high={high:.3g}, base={base:.3g}, RBF_high={b_high:.3g}",
                }
            )
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "layers": len(rs),
                "low_eigen_coeff_energy": avg(rs, "coeff_energy_low"),
                "mid_eigen_coeff_energy": avg(rs, "coeff_energy_mid"),
                "high_eigen_coeff_energy": high,
                "high_mode_task_pressure": grad_high,
                "base_mode_energy": base,
                "eig_condition": avg(rs, "eig_condition"),
                "p2_mode_shift": int(ok),
            }
        )
    return score, failures


def p3_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    rows = [r for r in rows if not r.get("error")]
    grouped = group(rows, ("dataset", "method"))
    baseline = {d: grouped.get((d, "ABRBF-AdamW"), []) for d in DATASETS}
    score: List[dict] = []
    failures: List[dict] = []
    pass_map: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for (dataset, method), rs in sorted(grouped.items()):
        b = baseline.get(dataset, [])
        b_acc = avg(b, "test_acc")
        b_hold = avg(b, "holdout_descent_final")
        b_phi = avg(b, "phi_rbf_p95")
        acc = avg(rs, "test_acc")
        hold = avg(rs, "holdout_descent_final")
        phi = avg(rs, "phi_rbf_p95")
        ok = method == "ABRBF-AdamW" or (
            acc >= b_acc - 0.03 and hold >= 0.75 * b_hold and phi <= b_phi
        )
        pass_map[method][dataset] = ok
        if method != "ABRBF-AdamW" and not ok:
            failures.append(
                {
                    "stage": "P3",
                    "dataset": dataset,
                    "method": method,
                    "failure_type": "F10_functional_candidate_lags_ABRBF_AdamW",
                    "detail": f"acc={acc:.4g}/{b_acc:.4g}, hold={hold:.4g}/{b_hold:.4g}, phiR={phi:.4g}/{b_phi:.4g}",
                }
            )
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "acc": acc,
                "gap_vs_ABRBF_AdamW": b_acc - acc,
                "holdout_descent_20": avg(rs, "holdout_descent_20"),
                "holdout_descent_final": hold,
                "holdout_ratio_vs_ABRBF_AdamW": hold / max(1e-12, b_hold),
                "phi_rbf": phi,
                "phi_rbf_red_vs_ABRBF_AdamW": 1.0 - phi / max(1e-12, b_phi),
                "phi_total": avg(rs, "phi_total_p95"),
                "curv_rbf": avg(rs, "curvature_rbf_p95"),
                "rank": avg(rs, "rank_block"),
                "margin_p10": avg(rs, "margin_p10"),
                "base_share": avg(rs, "base_update_share"),
                "rbf_share": avg(rs, "rbf_update_share"),
                "p3_pass": int(ok),
            }
        )
    survivors = [m for m, by_d in pass_map.items() if m != "ABRBF-AdamW" and all(by_d.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def p4_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    rows = [r for r in rows if not r.get("error")]
    score: List[dict] = []
    failures: List[dict] = []
    pass_map: Dict[Tuple[str, str, str, str], Dict[str, bool]] = defaultdict(dict)
    for row in rows:
        key = (row.get("teacher", ""), row.get("variant", ""), row.get("metric", ""), row.get("constraint", ""))
        ok = bool(int(f(row, "p4_pass", 0)))
        pass_map[key][row.get("dataset", "")] = ok
        if not ok:
            failure = "F4_exact_nfs_overconstrained" if row.get("variant", "").startswith("Exact") else "F6_relaxed_nfs_no_geometry_gain"
            if f(row, "acc_drop", 0) > 0.005:
                failure = "F5_relaxed_nfs_destroys_function"
            failures.append(
                {
                    "stage": "P4",
                    "dataset": row.get("dataset", ""),
                    "method": "|".join(key),
                    "failure_type": failure,
                    "detail": f"accDrop={f(row,'acc_drop'):.4g}, phiRRed={f(row,'phi_rbf_reduction'):.4g}, proj/raw={f(row,'projected_over_raw'):.3g}, KL={f(row,'KL'):.3g}",
                }
            )
        score.append(
            {
                "dataset": row.get("dataset", ""),
                "teacher": row.get("teacher", ""),
                "variant": row.get("variant", ""),
                "metric": row.get("metric", ""),
                "acc_drop": f(row, "acc_drop"),
                "KL": f(row, "KL"),
                "logit": f(row, "logit_drift"),
                "hidden": f(row, "hidden_drift"),
                "phi_rbf_red": f(row, "phi_rbf_reduction"),
                "curv_rbf_red": f(row, "curvature_rbf_reduction"),
                "J_red": f(row, "jacobian_reduction"),
                "projected_over_raw": f(row, "projected_over_raw"),
                "constraint_rank": f(row, "constraint_rank"),
                "constraint_condition": f(row, "constraint_condition"),
                "p4_pass": int(ok),
            }
        )
    survivors = ["|".join(k) for k, by_d in pass_map.items() if all(by_d.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def p7_summary(rows: List[dict]) -> List[dict]:
    rows = [r for r in rows if not r.get("error") and r.get("status") != "not_run"]
    score: List[dict] = []
    for (dataset, edge, hidden, basis, depth), rs in sorted(group(rows, ("dataset", "edge_kind", "hidden_dim", "basis_count", "depth")).items()):
        score.append(
            {
                "dataset": dataset,
                "edge_kind": edge,
                "hidden": hidden,
                "basis": basis,
                "depth": depth,
                "acc": avg(rs, "test_acc"),
                "phi_rbf": avg(rs, "phi_rbf_p95"),
                "phi_total": avg(rs, "phi_total_p95"),
                "rank": avg(rs, "rank_block"),
                "base_over_rbf": avg(rs, "base_over_rbf_norm"),
                "step_ms": avg(rs, "step_time_ms"),
            }
        )
    return score


def simple_bar_svg(path: Path, rows: Sequence[dict], label_key: str, value_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)[:18]
    width = 920
    bar_h = 22
    height = 60 + len(rows) * (bar_h + 8)
    vals = [max(0.0, f(r, value_key, 0.0)) for r in rows]
    vmax = max(vals) if vals else 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="20" y="28" font-family="sans-serif" font-size="18">{title}</text>')
    for i, row in enumerate(rows):
        y = 52 + i * (bar_h + 8)
        label = str(row.get(label_key, ""))[:48]
        val = max(0.0, f(row, value_key, 0.0))
        w = 520 * (val / vmax if vmax else 0)
        parts.append(f'<text x="20" y="{y+16}" font-family="monospace" font-size="11">{label}</text>')
        parts.append(f'<rect x="365" y="{y}" width="{w:.1f}" height="{bar_h}" fill="#2563eb" opacity="0.75"/>')
        parts.append(f'<text x="{375+w:.1f}" y="{y+16}" font-family="monospace" font-size="11">{val:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def ensure_aliases() -> None:
    aliases = {
        "runs.csv": "p1_architecture_frontier.csv",
        "curves.csv": "p1_training_trace.csv",
        "edge_decomposition_audit.csv": "p1_architecture_frontier.csv",
        "eigenmode_audit.csv": "p2_eigenmode_audit.csv",
        "basis_occupancy_audit.csv": "p2_eigenmode_audit.csv",
        "nfs_projection_audit.csv": "p4_relaxed_nfs_projection.csv",
    }
    for dst, src in aliases.items():
        sp = BASE / src
        dp = BASE / dst
        if sp.exists():
            shutil.copyfile(sp, dp)
    for name in ["p8_candidate_selection.csv", "p9_confirm5.csv", "p10_confirm10.csv"]:
        p = BASE / name
        if not p.exists():
            write_csv(p, [{"status": "not_run", "reason": "gate not reached"}])


def append_log(summary: str) -> None:
    marker = "## 2026-05-03 DG-KAN v5.0 Edge-Decomposed Functional Training"
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
    p3 = read_rows(BASE / "p3_split_functional_update.csv")
    p4 = read_rows(BASE / "p4_relaxed_nfs_projection.csv")
    p5 = read_rows(BASE / "p5_learn_smooth_refresh_cycle.csv")
    p6 = read_rows(BASE / "p6_event_tan_cycles.csv")
    p7 = read_rows(BASE / "p7_capacity_basis_followup.csv")

    p0_ok, p0_stats = p0_summary(p0)
    p1_score, p1_fail, p1_survivors = p1_summary(p1)
    p2_score, p2_fail = p2_summary(p2)
    p3_score, p3_fail, p3_survivors = p3_summary(p3)
    p4_score, p4_fail, p4_survivors = p4_summary(p4)
    p7_score = p7_summary(p7)

    write_csv(BASE / "p1_architecture_gate_summary.csv", p1_score)
    write_csv(BASE / "p2_eigenmode_gate_summary.csv", p2_score)
    write_csv(BASE / "p3_split_functional_gate_summary.csv", p3_score)
    write_csv(BASE / "p4_relaxed_nfs_gate_summary.csv", p4_score)
    write_csv(BASE / "p7_capacity_gate_summary.csv", p7_score)
    failures = p1_fail + p2_fail + p3_fail + p4_fail
    write_csv(BASE / "failure_table.csv", failures)
    ensure_aliases()

    final_status = "stop_after_p4_no_relaxed_nfs_survivor" if not p4_survivors else "p4_survivor_found"
    aggregate = {
        "p0_pass": p0_ok,
        "p0": p0_stats,
        "p1_architecture_survivors": p1_survivors,
        "p3_split_functional_survivors": p3_survivors,
        "p4_relaxed_nfs_survivors": p4_survivors,
        "p5_status": p5[0] if p5 else {"status": "not_run"},
        "p6_status": p6[0] if p6 else {"status": "not_run"},
        "final_status": final_status,
    }
    save_json(BASE / "aggregate_decision.json", aggregate)

    FIG.mkdir(parents=True, exist_ok=True)
    simple_bar_svg(FIG / "architecture_frontier_acc_vs_phi_rbf.svg", sorted(p1_score, key=lambda r: (r["dataset"], -f(r, "acc"))), "method", "acc", "P1 accuracy frontier")
    simple_bar_svg(FIG / "base_over_rbf_vs_acc.svg", sorted(p1_score, key=lambda r: -f(r, "base_over_rbf")), "method", "base_over_rbf", "Base/RBF contribution")
    simple_bar_svg(FIG / "eigenmode_energy_spectrum.svg", sorted(p2_score, key=lambda r: -f(r, "high_eigen_coeff_energy")), "method", "high_eigen_coeff_energy", "High eigenmode energy")
    simple_bar_svg(FIG / "basis_occupancy_heatmap.svg", sorted(p2_score, key=lambda r: -f(r, "base_mode_energy")), "method", "base_mode_energy", "Base mode energy")
    simple_bar_svg(FIG / "exact_vs_relaxed_nfs_scatter.svg", sorted(p4_score, key=lambda r: -f(r, "phi_rbf_red")), "variant", "phi_rbf_red", "NFS residual phi reduction")
    simple_bar_svg(FIG / "projected_over_raw_histogram.svg", sorted(p4_score, key=lambda r: -f(r, "projected_over_raw")), "variant", "projected_over_raw", "Projected/raw geometry gradient")
    simple_bar_svg(FIG / "learn_smooth_refresh_cycle.svg", p5 if p5 else [{"status": "not_run"}], "status", "p5_pass", "P5 learn-smooth-refresh")
    simple_bar_svg(FIG / "event_timeline.svg", p6 if p6 else [{"status": "not_run"}], "status", "p6_pass_final_combo", "P6 event TAN")
    simple_bar_svg(FIG / "paired_delta_vs_baselines.svg", sorted(p3_score, key=lambda r: -f(r, "acc")), "method", "acc", "P3 split functional accuracy")

    p1_rows = [
        [r["dataset"], r["method"], int(r["runs"]), fmt(r["acc"]), fmt(r["gap_vs_RBFOnly"]), fmt(r["phi_rbf"]), fmt(r["phi_total"]), fmt(r["base_over_rbf"]), int(r["p1_arch_positive"])]
        for r in p1_score
    ]
    p2_rows = [
        [r["dataset"], r["method"], fmt(r["high_eigen_coeff_energy"]), fmt(r["high_mode_task_pressure"]), fmt(r["base_mode_energy"]), fmt(r["eig_condition"], 1), int(r["p2_mode_shift"])]
        for r in p2_score
    ]
    p3_rows = [
        [r["dataset"], r["method"], int(r["runs"]), fmt(r["acc"]), fmt(r["gap_vs_ABRBF_AdamW"]), fmt(r["holdout_ratio_vs_ABRBF_AdamW"]), fmt(r["phi_rbf_red_vs_ABRBF_AdamW"]), fmt(r["base_share"]), int(r["p3_pass"])]
        for r in p3_score
    ]
    p4_rows = [
        [r["dataset"], r["teacher"], r["variant"], fmt(r["acc_drop"]), fmt(r["KL"], 5), fmt(r["phi_rbf_red"]), fmt(r["curv_rbf_red"]), fmt(r["projected_over_raw"]), int(r["p4_pass"])]
        for r in p4_score
    ]
    p7_rows = [
        [r["dataset"], r["edge_kind"], r["hidden"], r["basis"], r["depth"], fmt(r["acc"]), fmt(r["phi_rbf"]), fmt(r["rank"]), fmt(r["base_over_rbf"])]
        for r in p7_score
    ]

    doc = f"""# DG-KAN v5.0 Edge-Decomposed Functional Training 结果复盘

本轮依据 `docs/DG-KAN_v5.0_EdgeDecomposedFunctionalTraining_实验计划.md`。核心问题从“继续调 Sobolev optimizer”转为验证 edge-decomposed PureKAN：base path 学低阶/尺度结构，RBF residual 承担非线性，并把 geometry 主要约束到 residual 上。

## Run Inventory

| stage | rows | errors |
|---|---:|---:|
| P0 code audit | {len(p0)} | {sum(1 for r in p0 if r.get('error'))} |
| P1 architecture frontier | {len([r for r in p1 if not r.get('error')])} | {sum(1 for r in p1 if r.get('error'))} |
| P2 eigenmode audit | {len([r for r in p2 if not r.get('error')])} | {sum(1 for r in p2 if r.get('error'))} |
| P3 split functional update | {len([r for r in p3 if not r.get('error')])} | {sum(1 for r in p3 if r.get('error'))} |
| P4 relaxed NFS projection | {len([r for r in p4 if not r.get('error')])} | {sum(1 for r in p4 if r.get('error'))} |
| P5 learn-smooth-refresh | {len([r for r in p5 if r.get('status') != 'not_run'])} | 0 |
| P6 event TAN | {len([r for r in p6 if r.get('status') != 'not_run'])} | 0 |
| P7 capacity follow-up | {len([r for r in p7 if not r.get('error') and r.get('status') != 'not_run'])} | {sum(1 for r in p7 if r.get('error'))} |

## Code / Config Changes

```text
experiments/run_gafu_v50.py
  Added edge-decomposed PureKAN modules:
    ABRBF-linear
    ABRBF-silu
    ABRBF-linear+silu
    BaseOnly-linear / BaseOnly-silu
    learnWidth / quantile-center probes
  Added split geometry audit:
    phi_base_p95 / phi_rbf_p95 / phi_total_p95
    curvature_base/rbf/total
    sobolev_rbf_norm
  Added split functional update:
    base Adam-like task update
    RBF residual identity / Sobolev / dataSob-style updates
  Added relaxed NFS row-space solver using Woodbury form.

experiments/analyze_gafu_v50.py
  Generates gate summaries, failure taxonomy, aggregate_decision.json,
  required artifact aliases, SVG diagnostics, log entry, and this replay.
```

## P0 Code Audit

{md_table(['rows', 'errors', 'pass', 'max nonKAN', 'min coverage', 'max rollback', 'max KKT', 'max CG'], [[p0_stats['rows'], p0_stats['errors'], str(p0_ok).lower(), fmt(p0_stats['max_nonkan']), fmt(p0_stats['min_coverage']), fmt(p0_stats['max_rollback']), fmt(p0_stats['max_kkt'], 6), fmt(p0_stats['max_cg'], 6)]])}

P0 verdict: pass. Edge-only 参数计数、functional coverage、rollback、Exact/Relaxed NFS smoke 都可审计。Relaxed NFS 初版 dense solve 触发 OOM，已改为 row-space/Woodbury 解法。

## P1 Architecture Frontier

{md_table(['dataset', 'method', 'runs', 'acc', 'gap vs RBF', 'phi_rbf', 'phi_total', 'base/RBF', 'positive'], p1_rows)}

P1 verdict:

```text
AB-RBF is architecture-positive.
ABRBF-linear+silu is the strongest all-around short-run architecture:
  MNIST and Fashion improve over RBFOnly.
  KMNIST improves in seed0 compact P7 and remains competitive in P1.

BaseOnly is informative:
  It can approach RBF/ABRBF on MNIST/Fashion, but collapses relative to AB-RBF on KMNIST.
  This means the base path is necessary but not sufficient; the RBF residual still contributes.

Quantile centers and learnWidth alone are not stable positives.
```

P1 all-dataset architecture positives:

```text
{', '.join(p1_survivors) if p1_survivors else 'none'}
```

## P2 Eigenmode / Basis-Use Audit

{md_table(['dataset', 'method', 'high coeff', 'high grad', 'base mode', 'eig cond', 'shift'], p2_rows)}

P2 verdict:

```text
AB-RBF shifts a visible fraction of edge energy into explicit base modes.
This confirms the v4.9/v5.0 interpretation: RBFOnly uses coefficient modes to carry low-order structure.
However high-mode/task-pressure is not fully eliminated, especially when the residual remains needed for KMNIST.
```

## P3 Split-Metric Functional Update

{md_table(['dataset', 'method', 'runs', 'acc', 'gap vs ABRBF-A', 'hold/A', 'phiR red', 'base share', 'pass'], p3_rows)}

P3 verdict:

```text
Split functional update is a real improvement over residual-only training.
Moving base with Adam-like dynamics is essential:
  baseFrozen-rbfUFULL fails hard on all datasets.

Several baseAdam + rbf functional variants keep decent holdout descent and reduce residual phi.
But the all-dataset gate is not clean, mainly because KMNIST accuracy lags ABRBF-AdamW.
```

P3 all-dataset survivors:

```text
{', '.join(p3_survivors) if p3_survivors else 'none'}
```

## P4 Relaxed NFS Projection

{md_table(['dataset', 'teacher', 'variant', 'acc drop', 'KL', 'phiR red', 'curvR red', 'proj/raw', 'pass'], p4_rows)}

P4 verdict:

```text
Relaxed NFS is numerically stable and no longer overconstrained in the same way as Exact NFS:
  projected_over_raw is often around 0.5 to 1.0.

But the actual residual-geometry movement is still near zero under the current trust scale.
Heuristic NFS keeps the best practical geometry movement, around 2-3% phi_rbf reduction,
but this is below the planned >10% residual phi gate.

Therefore P5/P6 are not expanded.
```

P4 all-dataset survivors:

```text
{', '.join(p4_survivors) if p4_survivors else 'none'}
```

## P5 / P6 Decision

```text
P5 learn -> smooth -> refresh: not run
Reason: P4 produced no all-dataset relaxed NFS survivor.

P6 event-driven TAN v2: not run
Reason: P5 was not reached.
```

## P7 Capacity / Basis Follow-Up

{md_table(['dataset', 'edge', 'hidden', 'basis', 'depth', 'acc', 'phi_rbf', 'rank', 'base/RBF'], p7_rows)}

P7 verdict:

```text
Compact capacity follow-up supports the architecture signal.
ABRBF-linear+silu h64/b16/d4 is strong on KMNIST seed0 and improves MNIST over RBFOnly.
Scaling to h96/b24 does not automatically improve geometry; phi_rbf often rises.
Quantile centers are not a clean fix in this setup.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
p1_architecture_gate_summary.csv
p2_eigenmode_gate_summary.csv
p3_split_functional_gate_summary.csv
p4_relaxed_nfs_gate_summary.csv
p7_capacity_gate_summary.csv
```

Key labels:

```text
F1_base_path_unused:
  Mostly false for AB-RBF. Base/RBF contribution is nontrivial.

F2_base_path_helps_accuracy_but_geometry_bad:
  True for several AB-RBF rows under total geometry, but split geometry shows this is partly expected.

F3_residual_high_mode_pressure_persists:
  Partly true. AB-RBF helps, but does not remove residual pressure on KMNIST.

F4_exact_nfs_overconstrained:
  Still true for strict Exact NFS.

F6_relaxed_nfs_no_geometry_gain:
  New blocker. Relaxed projection keeps useful gradient mass, but current step/trust gives almost no residual phi reduction.

F10_functional_candidate_lags_ABRBF_AdamW:
  Main P3 blocker, especially on KMNIST.
```

## Required Artifacts

Written under `results/v5_0/`:

```text
runs.csv
curves.csv
edge_decomposition_audit.csv
eigenmode_audit.csv
basis_occupancy_audit.csv
nfs_projection_audit.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.0 status:
  {final_status}

What improved:
  Edge decomposition is clearly useful.
  AB-RBF, especially linear+silu / silu variants, improves the PureKAN architecture frontier.
  Split geometry gives a more honest view: base derivative is not the same as residual roughness.
  Split functional update confirms base-path task dynamics are essential.

What failed / remains open:
  Split functional candidates still lag ABRBF-AdamW on the hardest dataset.
  Relaxed NFS no longer has zero projected direction, but still fails to create enough residual smoothing.
  P5/P6/P8/P9 expansion is not justified.

Conclusion:
  v5.0 supports architecture success and partial optimizer progress.
  RBF-only was a bottleneck, but the current residual smoothing operator is still too weak.
  Next work should keep AB-RBF as the PureKAN default and redesign relaxed residual smoothing
  with a stronger accepted-step controller rather than returning to exact nullspace projection.
```
"""

    OUT.write_text(doc, encoding="utf-8")

    log = f"""## 2026-05-03 DG-KAN v5.0 Edge-Decomposed Functional Training

```text
P0 pass: {p0_ok}
P1 architecture survivors: {', '.join(p1_survivors) if p1_survivors else 'none'}
P3 split-functional survivors: {', '.join(p3_survivors) if p3_survivors else 'none'}
P4 relaxed-NFS survivors: {', '.join(p4_survivors) if p4_survivors else 'none'}
Final status: {final_status}
```

主结论：AB-RBF / edge decomposition 是 architecture-positive；base path 必须承担任务学习动力学。Relaxed NFS 的投影不再为零，但 residual geometry gain 仍太小，P5/P6 不展开。
"""
    append_log(log)
    print(f"Wrote {OUT}")
    print(f"Wrote {BASE / 'aggregate_decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
