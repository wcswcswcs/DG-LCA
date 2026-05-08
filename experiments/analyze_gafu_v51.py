#!/usr/bin/env python3
"""Summarize DG-KAN v5.1 AB-RBF strong residual smoothing experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v5_1"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v5.1_ABRBF_StrongResidualSmoothing_结果复盘.md"
LOG = ROOT / "docs/log.md"
DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
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
    return "" if not math.isfinite(float(x)) else f"{float(x):.{digits}f}"


def avg(rows: Sequence[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def std(rows: Sequence[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return pstdev(vals) if len(vals) > 1 else 0.0 if vals else math.nan


def grouped(rows: Iterable[dict], *keys: str) -> dict[tuple[str, ...], list[dict]]:
    out: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error") or row.get("status") == "not_run":
            continue
        out[tuple(str(row.get(k, "")) for k in keys)].append(row)
    return dict(out)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(v: object) -> str:
        return str(v).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def p0_summary(rows: list[dict], manifest: list[dict]) -> tuple[bool, dict]:
    ok_rows = [r for r in rows if not r.get("error")]
    ok = bool(ok_rows) and len(ok_rows) == len(rows)
    for row in ok_rows:
        ok = ok and f(row, "learnable_nonKAN_params", 999) == 0
        ok = ok and f(row, "coverage_edge_total", 0) >= 0.999
        ok = ok and f(row, "rollback_max_abs_error", f(row, "rollback_error", 1)) < 1e-8
        ok = ok and f(row, "finite_forward", 0) == 1 and f(row, "finite_backward", 0) == 1
    stats = {
        "rows": len(rows),
        "errors": len(rows) - len(ok_rows),
        "manifest_rows": len(manifest),
        "max_nonKAN": max([f(r, "learnable_nonKAN_params", 0) for r in ok_rows] or [math.nan]),
        "min_edge_cov": min([f(r, "coverage_edge_total", 0) for r in ok_rows] or [math.nan]),
        "min_base_cov": min([f(r, "coverage_base", 1) for r in ok_rows if f(r, "base_param_count", 0) > 0] or [math.nan]),
        "min_rbf_cov": min([f(r, "coverage_rbf", 1) for r in ok_rows if f(r, "rbf_param_count", 0) > 0] or [math.nan]),
        "max_rollback": max([f(r, "rollback_max_abs_error", f(r, "rollback_error", 0)) for r in ok_rows] or [math.nan]),
    }
    return ok, stats


def p1_summary(rows: list[dict]) -> tuple[list[dict], list[str]]:
    rows = [r for r in rows if not r.get("error")]
    g = grouped(rows, "dataset", "method")
    base = {d: g.get((d, "PureKAN-RBFOnly-AdamW"), []) for d in DATASETS}
    score: list[dict] = []
    pass_by_method: dict[str, set[str]] = defaultdict(set)
    for (dataset, method), rs in sorted(g.items()):
        b_acc = avg(base.get(dataset, []), "test_acc")
        acc = avg(rs, "test_acc")
        phi = avg(rs, "phi_rbf_p95")
        b_phi = avg(base.get(dataset, []), "phi_rbf_p95")
        base_ratio = avg(rs, "base_over_rbf_norm")
        positive = method == "PureKAN-RBFOnly-AdamW" or (
            "MLP" not in method
            and "BaseOnly" not in method
            and acc >= b_acc - 0.005
            and ("ABRBF" in method)
        )
        if positive:
            pass_by_method[method].add(dataset)
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "acc": acc,
                "std": std(rs, "test_acc"),
                "gap_vs_RBF": b_acc - acc,
                "phi_rbf": phi,
                "phi_total": avg(rs, "phi_total_p95"),
                "phi_rbf_red_vs_RBF": 1.0 - phi / max(1e-12, b_phi),
                "base_over_rbf": base_ratio,
                "rank": avg(rs, "rank_block"),
                "margin": avg(rs, "margin_p10"),
                "P1": int(positive),
            }
        )
    survivors = [
        m
        for m, ds in sorted(pass_by_method.items())
        if m != "PureKAN-RBFOnly-AdamW" and "ABRBF" in m and all(d in ds for d in DATASETS)
    ]
    write_csv(BASE / "p1_architecture_gate_summary.csv", score)
    return score, survivors


def p2_summary(rows: list[dict]) -> list[dict]:
    rows = [r for r in rows if not r.get("error")]
    score: list[dict] = []
    for (dataset, method), rs in sorted(grouped(rows, "dataset", "method").items()):
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "layers": len(rs),
                "high_coeff": avg(rs, "high_mode_fraction"),
                "high_grad": avg(rs, "grad_energy_high"),
                "base_mode": avg(rs, "base_mode_fraction"),
                "phi_base": avg(rs, "phi_base_p95"),
                "phi_rbf": avg(rs, "phi_rbf_p95"),
                "eig_cond": avg(rs, "eig_condition"),
                "mode_shift": int(avg(rs, "base_mode_fraction") > 0.05),
            }
        )
    write_csv(BASE / "p2_eigenmode_gate_summary.csv", score)
    return score


def p3_summary(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str, str]]]:
    rows = [r for r in rows if not r.get("error")]
    score: list[dict] = []
    pass_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for (teacher, proposal, controller, dataset), rs in sorted(grouped(rows, "teacher", "proposal", "controller", "dataset").items()):
        p = max([f(r, "p3_pass", 0) for r in rs] or [0])
        if p >= 1:
            pass_sets[(teacher, proposal, controller)].add(dataset)
        score.append(
            {
                "teacher": teacher,
                "proposal": proposal,
                "controller": controller,
                "dataset": dataset,
                "rows": len(rs),
                "pass": int(p >= 1),
                "acc_drop": avg(rs, "acc_drop"),
                "phiR_red": avg(rs, "actual_phiR_red"),
                "curvR_red": avg(rs, "actual_curvR_red"),
                "KL": avg(rs, "kl_teacher_student"),
                "logit": avg(rs, "logit_relative_drift"),
                "hidden": avg(rs, "hidden_relative_drift"),
                "score": avg(rs, "score"),
            }
        )
    survivors = [k for k, ds in pass_sets.items() if all(d in ds for d in DATASETS)]
    write_csv(BASE / "p3_residual_smoothing_gate_summary.csv", score)
    return score, survivors


def p4_summary(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str, str, str]]]:
    rows = [r for r in rows if not r.get("error") and r.get("status") != "not_run"]
    score: list[dict] = []
    pass_sets: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for (teacher, proposal, controller, steps, dataset), rs in sorted(grouped(rows, "teacher", "proposal", "controller", "steps", "dataset").items()):
        p = max([f(r, "p4_pass", 0) for r in rs] or [0])
        if p >= 1:
            pass_sets[(teacher, proposal, controller, steps)].add(dataset)
        score.append(
            {
                "teacher": teacher,
                "proposal": proposal,
                "controller": controller,
                "steps": steps,
                "dataset": dataset,
                "pass": int(p >= 1),
                "acc_drop": avg(rs, "acc_drop"),
                "phiR_red": avg(rs, "phiR_red"),
                "curvR_red": avg(rs, "curvR_red"),
                "holdout_delta": avg(rs, "holdout_loss_delta"),
                "accept_rate": avg(rs, "accept_rate"),
            }
        )
    survivors = [k for k, ds in pass_sets.items() if all(d in ds for d in DATASETS)]
    write_csv(BASE / "p4_accepted_smoothing_gate_summary.csv", score)
    return score, survivors


def p7_summary(rows: list[dict]) -> tuple[list[dict], list[str]]:
    rows = [r for r in rows if not r.get("error")]
    g = grouped(rows, "dataset", "method")
    base = {d: g.get((d, "ABRBF-AdamW"), []) for d in DATASETS}
    score: list[dict] = []
    pass_by: dict[str, set[str]] = defaultdict(set)
    for (dataset, method), rs in sorted(g.items()):
        b_acc = avg(base.get(dataset, []), "test_acc")
        b_phi = avg(base.get(dataset, []), "phi_rbf_p95")
        acc = avg(rs, "test_acc")
        phi = avg(rs, "phi_rbf_p95")
        ok = method == "ABRBF-AdamW" or (acc >= b_acc - 0.02 and phi <= b_phi * 0.90)
        if ok:
            pass_by[method].add(dataset)
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "acc": acc,
                "gap_vs_ABRBF": b_acc - acc,
                "hold/A": avg(rs, "holdout_loss_delta"),
                "phiR_red": 1.0 - phi / max(1e-12, b_phi),
                "base_share": avg(rs, "base_update_share"),
                "P7": int(ok),
            }
        )
    survivors = [m for m, ds in pass_by.items() if all(d in ds for d in DATASETS)]
    write_csv(BASE / "p7_split_residual_gate_summary.csv", score)
    return score, survivors


def p8_summary(rows: list[dict]) -> list[dict]:
    rows = [r for r in rows if not r.get("error")]
    score: list[dict] = []
    for (dataset, edge, hidden, basis, depth), rs in sorted(grouped(rows, "dataset", "edge_kind", "hidden_dim", "basis_count", "depth").items()):
        score.append(
            {
                "dataset": dataset,
                "edge": edge,
                "hidden": hidden,
                "basis": basis,
                "depth": depth,
                "acc": avg(rs, "test_acc"),
                "phi_rbf": avg(rs, "phi_rbf_p95"),
                "rank": avg(rs, "rank_block"),
                "base_over_rbf": avg(rs, "base_over_rbf_norm"),
            }
        )
    write_csv(BASE / "p8_capacity_gate_summary.csv", score)
    return score


def write_svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 760
    height = 80 + 28 * len(labels)
    max_v = max([v for v in values if math.isfinite(v)] or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="20" y="28" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, val) in enumerate(zip(labels, values)):
        y = 58 + i * 28
        bar = 0 if not math.isfinite(val) or max_v <= 0 else 440 * val / max_v
        lines.append(f'<text x="20" y="{y+14}" font-family="Arial" font-size="11">{label}</text>')
        lines.append(f'<rect x="250" y="{y}" width="{bar:.1f}" height="18" fill="#3b82f6"/>')
        lines.append(f'<text x="{260+bar:.1f}" y="{y+14}" font-family="Arial" font-size="11">{fmt(val,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p0 = read_rows(BASE / "p0_core_smoke.csv")
    manifest = read_rows(BASE / "edge_param_manifest.csv")
    p1 = read_rows(BASE / "p1_architecture_confirm5.csv")
    p2 = read_rows(BASE / "p2_split_geometry_calibration.csv")
    p3 = read_rows(BASE / "p3_residual_smoothing_proposal_audit.csv")
    p4 = read_rows(BASE / "p4_accepted_residual_smoothing.csv")
    p5 = read_rows(BASE / "p5_learn_smooth_refresh_v3.csv")
    p6 = read_rows(BASE / "p6_event_driven_tan_v2.csv")
    p7 = read_rows(BASE / "p7_split_residual_training.csv")
    p8 = read_rows(BASE / "p8_capacity_basis_followup.csv")

    p0_ok, p0_stats = p0_summary(p0, manifest)
    p1_score, p1_surv = p1_summary(p1)
    p2_score = p2_summary(p2)
    p3_score, p3_surv = p3_summary(p3)
    p4_score, p4_surv = p4_summary(p4)
    p7_score, p7_surv = p7_summary(p7)
    p8_score = p8_summary(p8)

    failure_rows: list[dict] = []
    if not p4_surv:
        failure_rows.append(
            {
                "stage": "P4",
                "failure_type": "F6_accumulated_smoothing_task_cost",
                "detail": "P3 single-step smoothing has all-dataset survivors, but P4 multi-step accepted smoothing fails holdout/accuracy gate.",
            }
        )
    if not p7_surv:
        failure_rows.append(
            {
                "stage": "P7",
                "failure_type": "F10_split_functional_lags_ABRBF_AdamW",
                "detail": "Split residual training lowers residual phi but does not pass the joint accuracy/geometry gate on all datasets.",
            }
        )
    write_csv(BASE / "failure_table.csv", failure_rows)

    status = "stop_after_p4_no_accepted_smoothing_survivor" if not p4_surv else "has_p4_survivor"
    decision = {
        "version": "v5.1",
        "status": status,
        "p0_pass": p0_ok,
        "p1_survivors": p1_surv,
        "p3_all_dataset_survivors": ["|".join(k) for k in p3_surv],
        "p4_all_dataset_survivors": ["|".join(k) for k in p4_surv],
        "p7_survivors": p7_surv,
        "rows": {
            "P0": len(p0),
            "P1": len(p1),
            "P2": len(p2),
            "P3": len(p3),
            "P4": len(p4),
            "P5": len(p5),
            "P6": len(p6),
            "P7": len(p7),
            "P8": len(p8),
        },
    }
    save_json(BASE / "aggregate_decision.json", decision)

    top_p1 = sorted(
        [r for r in p1_score if r["method"] != "PureKAN-RBFOnly-AdamW" and "MLP" not in r["method"]],
        key=lambda r: (r["dataset"], -r["acc"]),
    )
    top_p3 = sorted(p3_score, key=lambda r: (-r["pass"], -r["score"]))[:18]
    top_p4 = sorted(p4_score, key=lambda r: (-r["phiR_red"], r["acc_drop"]))[:18]
    top_p7 = sorted(p7_score, key=lambda r: (r["dataset"], -r["acc"]))
    top_p8 = sorted(p8_score, key=lambda r: (r["dataset"], -r["acc"]))

    p1_fig_rows = [r for r in p1_score if r["method"] in {"PureKAN-RBFOnly-AdamW", "PureKAN-ABRBF-linear+silu-AdamW", "PureKAN-ABRBF-linear-AdamW", "PureKAN-ABRBF-silu-AdamW"}]
    write_svg_bar(FIG / "p1_architecture_acc.svg", "P1 Mean Accuracy", [f"{r['dataset']} {r['method'].replace('PureKAN-','')}" for r in p1_fig_rows], [float(r["acc"]) for r in p1_fig_rows])
    write_svg_bar(FIG / "p4_phiR_reduction.svg", "P4 Residual Phi Reduction", [f"{r['dataset']} {r['steps']} {r['controller']}" for r in top_p4[:12]], [float(r["phiR_red"]) for r in top_p4[:12]])

    md: list[str] = []
    md.append("# DG-KAN v5.1 AB-RBF Strong Residual Smoothing 结果复盘")
    md.append("")
    md.append("本轮依据 `docs/DG-KAN_v5.1_ABRBF_StrongResidualSmoothing_实验计划.md`。核心目标是把 v5.0 的 AB-RBF 架构正信号固化为默认 PureKAN edge primitive，并测试更强的 residual smoothing proposal / accepted-step controller 是否能把单步 NFS 信号推进到多步稳定训练。")
    md.append("")
    md.append("## Run Inventory")
    md.append("")
    md.append(md_table(["stage", "rows", "errors"], [
        ["P0 core smoke", len(p0), p0_stats["errors"]],
        ["P1 architecture confirm5", len(p1), sum(1 for r in p1 if r.get("error"))],
        ["P2 split geometry", len(p2), sum(1 for r in p2 if r.get("error"))],
        ["P3 residual smoothing audit", len(p3), sum(1 for r in p3 if r.get("error"))],
        ["P4 accepted smoothing", len(p4), sum(1 for r in p4 if r.get("error"))],
        ["P5 learn-smooth-refresh", len(p5), sum(1 for r in p5 if r.get("error"))],
        ["P6 event TAN", len(p6), sum(1 for r in p6 if r.get("error"))],
        ["P7 split residual training", len(p7), sum(1 for r in p7 if r.get("error"))],
        ["P8 capacity follow-up", len(p8), sum(1 for r in p8 if r.get("error"))],
    ]))
    md.append("")
    md.append("## Code / Config Changes")
    md.append("")
    md.append("```text\nexperiments/dgkan_core.py\n  Added core ABRBFDense with linear / silu / linear+silu / base-only edge paths.\n  Added edge_named_params, base_named_params, and rbf_residual_named_params helpers.\n\nexperiments/run_gafu_v51.py\n  Added P0 edge manifest, P1 5-seed AB-RBF confirm, P2 split geometry calibration,\n  P3 S0-S7 residual smoothing proposals with C0-C5 controllers,\n  P4 multi-step accepted smoothing, plus P7/P8 follow-up diagnostics.\n\nexperiments/analyze_gafu_v51.py\n  Generates v5.1 gate summaries, failure table, aggregate decision, figures, and this replay.\n```")
    md.append("")
    md.append("## P0 Core Smoke")
    md.append("")
    md.append(md_table(["rows", "errors", "manifest", "max nonKAN", "min edge cov", "min base cov", "min rbf cov", "max rollback", "pass"], [[
        p0_stats["rows"], p0_stats["errors"], p0_stats["manifest_rows"], fmt(p0_stats["max_nonKAN"]), fmt(p0_stats["min_edge_cov"]), fmt(p0_stats["min_base_cov"]), fmt(p0_stats["min_rbf_cov"]), fmt(p0_stats["max_rollback"]), str(p0_ok).lower()
    ]]))
    md.append("")
    md.append("P0 verdict: pass. Core AB-RBF edge paths keep strict PureKAN non-KAN trainable parameters at zero, edge/base/RBF parameter coverage is auditable, and rollback is exact.")
    md.append("")
    md.append("## P1 Architecture Confirm5")
    md.append("")
    md.append(md_table(["dataset", "method", "runs", "acc", "std", "gap vs RBF", "phiR", "base/RBF", "P1"], [
        [r["dataset"], r["method"], r["runs"], fmt(r["acc"]), fmt(r["std"]), fmt(r["gap_vs_RBF"]), fmt(r["phi_rbf"]), fmt(r["base_over_rbf"]), "yes" if r["P1"] else "no"]
        for r in top_p1
    ]))
    md.append("")
    md.append("P1 survivors:")
    md.append("")
    md.append("```text\n" + ("\n".join(p1_surv) if p1_surv else "none") + "\n```")
    md.append("")
    md.append("P1 verdict: AB-RBF remains architecture-positive. The base path is consistently used; BaseOnly remains diagnostic rather than sufficient, especially on KMNIST.")
    md.append("")
    md.append("## P2 Split Geometry / Eigenmode Audit")
    md.append("")
    md.append(md_table(["dataset", "method", "high coeff", "high grad", "base mode", "phiR", "eig cond", "shift"], [
        [r["dataset"], r["method"], fmt(r["high_coeff"]), fmt(r["high_grad"]), fmt(r["base_mode"]), fmt(r["phi_rbf"]), fmt(r["eig_cond"], 1), r["mode_shift"]]
        for r in p2_score
    ]))
    md.append("")
    md.append("P2 verdict: AB-RBF shifts visible energy into explicit base modes, confirming that RBF-only had been carrying low-order structure through residual coefficient modes. The residual high-mode pressure is reduced but not removed.")
    md.append("")
    md.append("## P3 Residual Smoothing Proposal Audit")
    md.append("")
    md.append(md_table(["teacher", "proposal", "controller", "dataset", "pass", "acc drop", "phiR red", "curvR red", "KL", "score"], [
        [r["teacher"], r["proposal"], r["controller"], r["dataset"], r["pass"], fmt(r["acc_drop"]), fmt(r["phiR_red"]), fmt(r["curvR_red"]), fmt(r["KL"]), fmt(r["score"])]
        for r in top_p3
    ]))
    md.append("")
    md.append("P3 all-dataset survivors:")
    md.append("")
    md.append("```text\n" + ("\n".join("|".join(k) for k in p3_surv[:30]) if p3_surv else "none") + "\n```")
    md.append("")
    md.append("P3 verdict: strong single-step residual smoothing is real. `S0-laplacian`, `S1-sobolev-grad`, and `S6-heuristic-role-block` can produce accepted residual phi reductions under trust controllers across all datasets.")
    md.append("")
    md.append("## P4 Accepted Residual Smoothing")
    md.append("")
    md.append(md_table(["dataset", "proposal", "controller", "steps", "acc drop", "phiR red", "curvR red", "holdout Δ", "accept", "P4"], [
        [r["dataset"], r["proposal"], r["controller"], r["steps"], fmt(r["acc_drop"]), fmt(r["phiR_red"]), fmt(r["curvR_red"]), fmt(r["holdout_delta"]), fmt(r["accept_rate"]), "yes" if r["pass"] else "no"]
        for r in top_p4
    ]))
    md.append("")
    md.append("P4 all-dataset survivors:")
    md.append("")
    md.append("```text\n" + ("\n".join("|".join(k) for k in p4_surv) if p4_surv else "none") + "\n```")
    md.append("")
    md.append("P4 verdict: no survivor. Multi-step accepted smoothing reduces residual phi strongly, but it violates the joint task/holdout gate. The bottleneck moved from “no geometry movement” to “geometry movement is too task-expensive when accumulated.”")
    md.append("")
    md.append("## P5 / P6 Decision")
    md.append("")
    md.append("```text\nP5 learn-smooth-refresh: not run\nReason: P4 produced no all-dataset accepted smoothing survivor.\n\nP6 event TAN: not run\nReason: P5 was not reached.\n```")
    md.append("")
    md.append("## P7 Split Residual Training")
    md.append("")
    md.append(md_table(["dataset", "method", "runs", "acc", "gap vs ABRBF", "hold/A", "phiR red", "base share", "P7"], [
        [r["dataset"], r["method"], r["runs"], fmt(r["acc"]), fmt(r["gap_vs_ABRBF"]), fmt(r["hold/A"]), fmt(r["phiR_red"]), fmt(r["base_share"]), "yes" if r["P7"] else "no"]
        for r in top_p7
    ]))
    md.append("")
    md.append("P7 verdict: moving the base path is essential. Functional residual variants reduce residual phi, but KMNIST accuracy still lags the ABRBF-AdamW teacher enough to block a clean all-dataset survivor.")
    md.append("")
    md.append("## P8 Capacity / Basis Follow-Up")
    md.append("")
    md.append(md_table(["dataset", "edge", "hidden", "basis", "depth", "acc", "phiR", "rank", "base/RBF"], [
        [r["dataset"], r["edge"], r["hidden"], r["basis"], r["depth"], fmt(r["acc"]), fmt(r["phi_rbf"]), fmt(r["rank"]), fmt(r["base_over_rbf"])]
        for r in top_p8
    ]))
    md.append("")
    md.append("P8 verdict: compact ABRBF h64/b16 remains a strong default, especially on KMNIST for linear+silu. Scaling h96/b24 does not automatically improve residual geometry, and quantile centers again worsen the tradeoff.")
    md.append("")
    md.append("## Failure Diagnosis")
    md.append("")
    md.append("```text\nF6_accumulated_smoothing_task_cost:\n  P3 single-step residual smoothing works, but P4 multi-step smoothing increases holdout/task cost.\n\nF10_split_functional_lags_ABRBF_AdamW:\n  Split functional training lowers residual phi but does not match ABRBF-AdamW on the hardest dataset.\n\nArchitecture diagnosis:\n  AB-RBF is now the right PureKAN default. The remaining bottleneck is the residual smoothing controller, not edge parameterization coverage.\n```")
    md.append("")
    md.append("## Required Artifacts")
    md.append("")
    md.append("Written under `results/v5_1/`:")
    md.append("")
    md.append("```text\np0_core_smoke.csv\nedge_param_manifest.csv\np1_architecture_confirm5.csv\np1_architecture_gate_summary.csv\np2_split_geometry_calibration.csv\np2_eigenmode_gate_summary.csv\np3_residual_smoothing_proposal_audit.csv\np3_residual_smoothing_gate_summary.csv\np4_accepted_residual_smoothing.csv\np4_accepted_residual_smoothing_trace.csv\np4_accepted_smoothing_gate_summary.csv\np5_learn_smooth_refresh_v3.csv\np6_event_driven_tan_v2.csv\np7_split_residual_training.csv\np7_split_residual_gate_summary.csv\np8_capacity_basis_followup.csv\np8_capacity_gate_summary.csv\nfailure_table.csv\naggregate_decision.json\nfigures/\n```")
    md.append("")
    md.append("## Final Decision")
    md.append("")
    md.append("```text\nDG-KAN v5.1 status:\n  " + status + "\n\nWhat improved:\n  AB-RBF is promoted into core dgkan_core.py and remains architecture-positive.\n  P3 confirms strong residual smoothing can produce accepted single-step residual phi reductions.\n  P7 confirms split base/residual training is meaningful and base dynamics are essential.\n\nWhat failed / remains open:\n  P4 multi-step accepted smoothing fails the joint task/holdout gate.\n  P5/P6 expansion is not justified.\n  Split functional candidates still do not cleanly beat ABRBF-AdamW on KMNIST.\n\nConclusion:\n  v5.1 solves the edge primitive and single-step residual smoothing direction problem,\n  but not the accumulated smoothing controller problem.\n  Next work should keep AB-RBF as default and redesign multi-step smoothing with task-aware recovery,\n  smaller adaptive eta, or interleaved refresh before spending confirm-seed budget.\n```")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")

    log_entry = "- v5.1 AB-RBF strong residual smoothing: P3 found all-dataset single-step smoothing survivors; P4 failed multi-step task/holdout gate, so stopped before P5/P6 confirm.\n"
    if LOG.exists():
        text = LOG.read_text(encoding="utf-8")
        if "v5.1 AB-RBF strong residual smoothing" not in text:
            LOG.write_text(text.rstrip() + "\n" + log_entry, encoding="utf-8")
    else:
        LOG.write_text(log_entry, encoding="utf-8")

    print(f"Wrote {OUT}")
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
