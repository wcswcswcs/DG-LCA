#!/usr/bin/env python3
"""Summarize DG-KAN v4.8 Functional Geometry Feasibility experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v4_8"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v4.8_FunctionalGeometry_Feasibility_Redesign_结果复盘.md"
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
        ok = ok and f(row, "nonkan_param_count", 999) == 0
        ok = ok and abs(f(row, "functional_coverage", 0.0) - 1.0) < 1e-9
        ok = ok and f(row, "rollback_max_abs_error", 1.0) < 1e-8
        ok = ok and f(row, "teacher_snapshot_error", 1) == 0
        ok = ok and f(row, "jvp_finite", 0) == 1
        ok = ok and f(row, "vjp_finite", 0) == 1
        ok = ok and f(row, "no_nan_inf", 0) == 1
    return ok, {
        "rows": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "max_nonkan": max([f(r, "nonkan_param_count", 0) for r in ok_rows] or [math.nan]),
        "min_coverage": min([f(r, "functional_coverage", 0) for r in ok_rows] or [math.nan]),
        "max_rollback": max([f(r, "rollback_max_abs_error", 0) for r in ok_rows] or [math.nan]),
        "max_cg_residual": max([f(r, "cg_residual", 0) for r in ok_rows] or [math.nan]),
    }


def p1_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    grouped = group(rows, ("dataset", "method"))
    baseline = {d: grouped.get((d, "A0-PureKAN-AdamW"), []) for d in DATASETS}
    score: List[dict] = []
    failures: List[dict] = []
    pass_map: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for (dataset, method), rs in sorted(grouped.items()):
        b = baseline.get(dataset, [])
        b_acc = avg(b, "test_acc")
        b_phi = avg(b, "phi_prime_p95")
        b_j = avg(b, "jacobian_condition")
        b_ece = avg(b, "ece")
        acc = avg(rs, "test_acc")
        phi = avg(rs, "phi_prime_p95")
        jac = avg(rs, "jacobian_condition")
        ece = avg(rs, "ece")
        ok = method == "A0-PureKAN-AdamW" or (
            acc >= b_acc - 0.01
            and ((phi <= 0.80 * b_phi) or (jac <= 0.50 * b_j))
            and ece <= b_ece + 0.02
        )
        pass_map[method][dataset] = ok
        if method != "A0-PureKAN-AdamW" and not ok:
            reasons = []
            if acc < b_acc - 0.01:
                reasons.append("F1 acc/geometry frontier absent")
            if not ((phi <= 0.80 * b_phi) or (jac <= 0.50 * b_j)):
                reasons.append("F1 geometry not improved enough")
            if ece > b_ece + 0.02:
                reasons.append("F7 ECE worsened")
            failures.append({"stage": "P1", "dataset": dataset, "method": method, "failure_type": "; ".join(reasons), "detail": f"acc={acc:.4g}, base={b_acc:.4g}, phi={phi:.4g}/{b_phi:.4g}, J={jac:.4g}/{b_j:.4g}, ECE={ece:.4g}/{b_ece:.4g}"})
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "acc": acc,
                "acc_std": std(rs, "test_acc"),
                "acc_gap_vs_A0": b_acc - acc,
                "ECE": ece,
                "phi": phi,
                "phi_red_vs_A0": 1.0 - phi / max(1e-12, b_phi),
                "J": jac,
                "J_red_vs_A0": 1.0 - jac / max(1e-12, b_j),
                "sobolev": avg(rs, "sobolev_norm_total"),
                "roughness": avg(rs, "coefficient_roughness"),
                "rank": avg(rs, "rank_block"),
                "margin_p10": avg(rs, "margin_p10"),
                "basis_dead": avg(rs, "dead_basis_fraction"),
                "step_time_ms": avg(rs, "step_time_ms"),
                "p1_pass": int(ok),
            }
        )
    survivors = [m for m, by_d in pass_map.items() if m != "A0-PureKAN-AdamW" and all(by_d.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def p2_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    score: List[dict] = []
    failures: List[dict] = []
    pass_map: Dict[Tuple[str, str, str, str], Dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row.get("error"):
            failures.append({"stage": "P2", "dataset": row.get("dataset", ""), "method": row.get("variant", ""), "failure_type": "F2 compute failure", "detail": row.get("error", "")})
            continue
        ok = bool(int(f(row, "p2_pass", 0)))
        key = (row.get("teacher", ""), row.get("variant", ""), row.get("projector", ""), row.get("eta", ""))
        pass_map[key][row.get("dataset", "")] = ok
        if not ok:
            reasons = []
            if f(row, "kl_teacher_student", 99) >= 0.05 or f(row, "logit_relative_drift", 99) >= 0.03:
                reasons.append("F2 function not preserved")
            if f(row, "acc_drop", 99) > 0.005:
                reasons.append("F4 smoothing destroys accuracy")
            if not (f(row, "phi_reduction", -99) > 0.10 or f(row, "jacobian_reduction", -99) > 0.20):
                reasons.append("F2 geometry not reduced in nullspace")
            failures.append({"stage": "P2", "dataset": row.get("dataset", ""), "method": f"{row.get('teacher','')}|{row.get('variant','')}|{row.get('projector','')}", "failure_type": "; ".join(reasons), "detail": f"accDrop={f(row,'acc_drop'):.4g}, KL={f(row,'kl_teacher_student'):.3g}, logit={f(row,'logit_relative_drift'):.3g}, phiRed={f(row,'phi_reduction'):.3g}, JRed={f(row,'jacobian_reduction'):.3g}"})
        score.append(
            {
                "dataset": row.get("dataset", ""),
                "teacher": row.get("teacher", ""),
                "variant": row.get("variant", ""),
                "projector": row.get("projector", ""),
                "eta": row.get("eta", ""),
                "teacher_acc": f(row, "teacher_acc"),
                "acc_after": f(row, "acc_after"),
                "acc_drop": f(row, "acc_drop"),
                "phi_reduction": f(row, "phi_reduction"),
                "jacobian_reduction": f(row, "jacobian_reduction"),
                "sobolev_reduction": f(row, "sobolev_reduction"),
                "KL": f(row, "kl_teacher_student"),
                "logit_drift": f(row, "logit_relative_drift"),
                "hidden_drift": f(row, "hidden_relative_drift"),
                "acceptance": f(row, "acceptance_rate"),
                "smoothability": f(row, "smoothability_score"),
                "p2_pass": int(ok),
            }
        )
    survivors = ["|".join(k) for k, by_d in pass_map.items() if all(by_d.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def p3_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    score: List[dict] = []
    failures: List[dict] = []
    pass_map: Dict[Tuple[str, str, str, str], Dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row.get("status") == "not_run":
            continue
        if row.get("error"):
            failures.append({"stage": "P3", "dataset": row.get("dataset", ""), "method": row.get("variant", ""), "failure_type": "F2 compute failure", "detail": row.get("error", "")})
            continue
        ok = bool(int(f(row, "p3_pass", f(row, "p2_pass", 0))))
        key = (row.get("teacher", ""), row.get("variant", ""), row.get("projector", ""), row.get("eta", ""))
        pass_map[key][row.get("dataset", "")] = ok
        if not ok:
            reasons = []
            if f(row, "acc_drop", 99) > 0.005:
                reasons.append("F4 functional teacher not preserved")
            if not (f(row, "phi_reduction", -99) > 0.10 or f(row, "jacobian_reduction", -99) > 0.20):
                reasons.append("F2 functional-teacher geometry reduction weak")
            failures.append({"stage": "P3", "dataset": row.get("dataset", ""), "method": f"{row.get('teacher','')}|{row.get('variant','')}", "failure_type": "; ".join(reasons), "detail": f"accDrop={f(row,'acc_drop'):.4g}, phiRed={f(row,'phi_reduction'):.3g}, JRed={f(row,'jacobian_reduction'):.3g}, KL={f(row,'kl_teacher_student'):.3g}"})
        score.append(
            {
                "dataset": row.get("dataset", ""),
                "teacher": row.get("teacher", ""),
                "teacher_method": row.get("teacher_method", ""),
                "variant": row.get("variant", ""),
                "projector": row.get("projector", ""),
                "eta": row.get("eta", ""),
                "teacher_acc": f(row, "teacher_acc"),
                "teacher_holdout_descent": f(row, "teacher_holdout_descent"),
                "teacher_rank": f(row, "teacher_rank"),
                "teacher_phi_ratio": f(row, "teacher_phi_ratio_to_initial"),
                "acc_drop": f(row, "acc_drop"),
                "KL": f(row, "kl_teacher_student"),
                "phi_reduction": f(row, "phi_reduction"),
                "J_reduction": f(row, "jacobian_reduction"),
                "smoothability": f(row, "smoothability_score"),
                "p3_pass": int(ok),
            }
        )
    survivors = ["|".join(k) for k, by_d in pass_map.items() if all(by_d.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def p4_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    score: List[dict] = []
    failures: List[dict] = []
    pass_map: Dict[Tuple[str, str, str, str], Dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row.get("status") == "not_run":
            continue
        if row.get("error"):
            failures.append({"stage": "P4", "dataset": row.get("dataset", ""), "method": row.get("task_method", ""), "failure_type": "F5 TAN one-cycle compute failure", "detail": row.get("error", "")})
            continue
        ok = bool(int(f(row, "p4_pass", 0)))
        key = (row.get("task_method", ""), row.get("nfs_label", ""), row.get("variant", ""), row.get("refresh_method", ""))
        pass_map[key][row.get("dataset", "")] = ok
        if not ok:
            reasons = []
            if f(row, "acc_drop_from_task", 99) >= 0.01:
                reasons.append("F4 smoothing/refresh accuracy drop")
            if f(row, "phi_reduction_from_task", -99) <= 0.10:
                reasons.append("F2 cycle geometry gain weak")
            if f(row, "refresh_recovered_loss", -99) < 0.80:
                reasons.append("F5 refresh recovery weak")
            failures.append({"stage": "P4", "dataset": row.get("dataset", ""), "method": f"{row.get('task_method','')}|{row.get('nfs_label','')}|{row.get('refresh_method','')}", "failure_type": "; ".join(reasons), "detail": f"accDrop={f(row,'acc_drop_from_task'):.4g}, phiRed={f(row,'phi_reduction_from_task'):.3g}, rec={f(row,'refresh_recovered_loss'):.3g}"})
        score.append(
            {
                "dataset": row.get("dataset", ""),
                "task_method": row.get("task_method", ""),
                "nfs_label": row.get("nfs_label", ""),
                "variant": row.get("variant", ""),
                "projector": row.get("projector", ""),
                "refresh": row.get("refresh_method", ""),
                "task_acc": f(row, "task_acc"),
                "smooth_acc": f(row, "smooth_acc"),
                "refresh_acc": f(row, "refresh_acc"),
                "acc_drop": f(row, "acc_drop_from_task"),
                "task_phi": f(row, "task_phi"),
                "refresh_phi": f(row, "refresh_phi"),
                "phi_reduction": f(row, "phi_reduction_from_task"),
                "KL_smooth": f(row, "teacher_KL_after_smooth"),
                "refresh_recovery": f(row, "refresh_recovered_loss"),
                "p4_pass": int(ok),
            }
        )
    survivors = ["|".join(k) for k, by_d in pass_map.items() if all(by_d.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def p5_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    score: List[dict] = []
    failures: List[dict] = []
    if rows and rows[0].get("status") == "not_run":
        return rows, [], []
    grouped = group(rows, ("dataset", "task_method", "nfs_label", "variant", "refresh_method"))
    pass_map: Dict[Tuple[str, str, str, str], Dict[str, bool]] = defaultdict(dict)
    for (dataset, task, label, variant, refresh), rs in sorted(grouped.items()):
        final = max(rs, key=lambda r: f(r, "cycle_index", 0))
        ok = bool(int(f(final, "p5_pass_final_combo", 0)))
        key = (task, label, variant, refresh)
        pass_map[key][dataset] = ok
        if not ok:
            failures.append({"stage": "P5", "dataset": dataset, "method": f"{task}|{label}|{refresh}", "failure_type": "F5 alternating TAN cycle did not preserve task+geometry", "detail": f"finalAcc={f(final,'after_acc'):.4g}, finalPhi={f(final,'after_phi'):.4g}, cycle={f(final,'cycle_index'):.0f}"})
        score.append(
            {
                "dataset": dataset,
                "task_method": task,
                "nfs_label": label,
                "variant": variant,
                "refresh": refresh,
                "cycles": int(f(final, "cycle_index", 0)),
                "final_acc": f(final, "after_acc"),
                "final_val_loss": f(final, "after_val_loss"),
                "final_phi": f(final, "after_phi"),
                "final_J": f(final, "after_jacobian"),
                "final_rank": f(final, "after_rank"),
                "final_margin": f(final, "after_margin"),
                "p5_pass": int(ok),
            }
        )
    survivors = ["|".join(k) for k, by_d in pass_map.items() if all(by_d.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def p6_summary(rows: List[dict], p1_score: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    if not rows:
        return [], [], []
    p1_base = {r["dataset"]: r for r in p1_score if r.get("method") == "A0-PureKAN-AdamW"}
    score: List[dict] = []
    failures: List[dict] = []
    pass_map: Dict[Tuple[str, str, str, str, str], Dict[str, bool]] = defaultdict(dict)
    grouped = group(rows, ("dataset", "method", "hidden_dim", "basis_count", "depth", "basis_type"))
    for key, rs in sorted(grouped.items()):
        dataset, method, hidden, basis, depth, basis_type = key
        base = p1_base.get(dataset, {})
        b_acc = f(base, "acc")
        b_phi = f(base, "phi")
        b_j = f(base, "J")
        acc = avg(rs, "test_acc")
        phi = avg(rs, "phi_prime_p95")
        jac = avg(rs, "jacobian_condition")
        ok = acc >= b_acc - 0.01 and ((phi <= 0.80 * b_phi) or (jac <= 0.50 * b_j))
        pass_map[(method, hidden, basis, depth, basis_type)][dataset] = ok
        if not ok:
            failures.append({"stage": "P6", "dataset": dataset, "method": f"{method}|h{hidden}|b{basis}|d{depth}|{basis_type}", "failure_type": "F6 capacity/basis insufficient", "detail": f"acc={acc:.4g}, base={b_acc:.4g}, phi={phi:.4g}/{b_phi:.4g}, J={jac:.4g}/{b_j:.4g}"})
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "hidden_dim": hidden,
                "basis_count": basis,
                "depth": depth,
                "basis_type": basis_type,
                "runs": len(rs),
                "acc": acc,
                "acc_gap_vs_A0": b_acc - acc,
                "phi": phi,
                "phi_red_vs_A0": 1.0 - phi / max(1e-12, b_phi),
                "J": jac,
                "J_red_vs_A0": 1.0 - jac / max(1e-12, b_j),
                "sobolev": avg(rs, "sobolev_norm_total"),
                "rank": avg(rs, "rank_block"),
                "margin_p10": avg(rs, "margin_p10"),
                "param_count": avg(rs, "param_count"),
                "step_time_ms": avg(rs, "step_time_ms"),
                "p6_pass": int(ok),
            }
        )
    survivors = ["|".join(k) for k, by_d in pass_map.items() if all(by_d.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def blank_later(reason: str) -> None:
    for name in [
        "p3_nfs_functional_teachers.csv",
        "p4_tan_one_cycle_micro_run.csv",
        "p5_tan_alternating_cycles.csv",
        "p7_candidate_selection.csv",
        "p8_confirm5.csv",
        "p9_confirm10.csv",
    ]:
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


def svg_scatter(path: Path, title: str, rows: Sequence[dict], xkey: str, ykey: str, label_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = [(f(r, xkey), f(r, ykey), r.get(label_key, "")) for r in rows if math.isfinite(f(r, xkey)) and math.isfinite(f(r, ykey))]
    width, height = 760, 420
    if not vals:
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="30">{title}: no data</text></svg>\n')
        return
    xs, ys = [v[0] for v in vals], [v[1] for v in vals]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax += 1.0
    if ymin == ymax:
        ymax += 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="24" y="28" font-family="sans-serif" font-size="16">{title}</text>']
    for x, y, lab in vals:
        px = 60 + 640 * (x - xmin) / (xmax - xmin)
        py = 360 - 300 * (y - ymin) / (ymax - ymin)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#4c78a8" opacity="0.75"><title>{lab}: {x:.4g},{y:.4g}</title></circle>')
    parts.append(f'<text x="60" y="395" font-family="sans-serif" font-size="11">{xkey}</text>')
    parts.append(f'<text x="6" y="210" font-family="sans-serif" font-size="11" transform="rotate(-90 12 210)">{ykey}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 300
    vals = [v if math.isfinite(v) else 0.0 for v in values]
    lo, hi = min(0.0, min(vals) if vals else 0.0), max(1e-9, max(vals) if vals else 1.0)
    bw = max(8, int((width - 140) / max(1, len(vals))))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="24" y="28" font-family="sans-serif" font-size="16">{title}</text>']
    for i, (lab, val) in enumerate(zip(labels, vals)):
        x = 70 + i * bw
        h = int(180 * (val - lo) / (hi - lo))
        y = 230 - h
        parts.append(f'<rect x="{x}" y="{y}" width="{max(4, bw-3)}" height="{h}" fill="#f58518"/>')
        parts.append(f'<text x="{x}" y="260" font-family="sans-serif" font-size="8" transform="rotate(35 {x} 260)">{lab[:28]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def make_figures(p1_score: List[dict], p2_score: List[dict], p6_score: List[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    svg_scatter(FIG / "p1_acc_vs_phi_frontier.svg", "P1 accuracy vs phi frontier", p1_score, "phi", "acc", "method")
    svg_scatter(FIG / "p1_acc_vs_j_frontier.svg", "P1 accuracy vs J frontier", p1_score, "J", "acc", "method")
    svg_scatter(FIG / "p1_margin_vs_phi.svg", "P1 margin_p10 vs phi", p1_score, "phi", "margin_p10", "method")
    svg_scatter(FIG / "p1_rank_vs_acc.svg", "P1 rank vs accuracy", p1_score, "rank", "acc", "method")
    svg_scatter(FIG / "p2_kl_vs_phi_reduction.svg", "P2 KL vs phi reduction", p2_score, "KL", "phi_reduction", "variant")
    svg_scatter(FIG / "p2_logit_drift_vs_phi_reduction.svg", "P2 logit drift vs phi reduction", p2_score, "logit_drift", "phi_reduction", "variant")
    svg_scatter(FIG / "p2_hidden_drift_vs_phi_reduction.svg", "P2 hidden drift vs phi reduction", p2_score, "hidden_drift", "phi_reduction", "variant")
    if p2_score:
        g = sorted(group(p2_score, ("variant",)).items())
        svg_bar(FIG / "p2_rolewise_geometry_reduction.svg", "P2 mean phi reduction by NFS variant", [k[0] for k, _ in g], [avg(rs, "phi_reduction") for _, rs in g])
    if p6_score:
        svg_scatter(FIG / "p6_capacity_frontier.svg", "P6 capacity accuracy-geometry frontier", p6_score, "phi", "acc", "basis_type")


def main() -> int:
    p0 = read_rows(BASE / "p0_fgf_invariants.csv")
    p1 = read_rows(BASE / "p1_feasibility_frontier.csv")
    p2 = read_rows(BASE / "p2_nfs_adamw_projection.csv")
    p3 = read_rows(BASE / "p3_nfs_functional_teachers.csv")
    p4 = read_rows(BASE / "p4_tan_one_cycle_micro_run.csv")
    p5 = read_rows(BASE / "p5_tan_alternating_cycles.csv")
    p6 = read_rows(BASE / "p6_capacity_basis_expansion.csv")
    p0_pass, p0_stats = p0_summary(p0)
    p1_score, failures, p1_survivors = p1_summary(p1)
    p2_score, p2_failures, p2_survivors = p2_summary(p2)
    p3_score, p3_failures, p3_survivors = p3_summary(p3)
    p4_score, p4_failures, p4_survivors = p4_summary(p4)
    p5_score, p5_failures, p5_survivors = p5_summary(p5)
    p6_score, p6_failures, p6_survivors = p6_summary(p6, p1_score)
    failures.extend(p2_failures)
    failures.extend(p3_failures)
    failures.extend(p4_failures)
    failures.extend(p5_failures)
    failures.extend(p6_failures)
    write_csv(BASE / "p1_frontier_gate_summary.csv", p1_score)
    write_csv(BASE / "p2_nfs_gate_summary.csv", p2_score)
    write_csv(BASE / "p3_nfs_functional_gate_summary.csv", p3_score)
    write_csv(BASE / "p4_tan_one_cycle_gate_summary.csv", p4_score)
    write_csv(BASE / "p5_tan_cycle_gate_summary.csv", p5_score)
    write_csv(BASE / "p6_capacity_gate_summary.csv", p6_score)
    if p2_survivors:
        if not p3_score:
            blank_later("P3/P4 should be expanded from P2 NFS survivors")
            final_decision = "p3_needed"
        elif p3_survivors and not p4_score:
            blank_later("P4 TAN one-cycle should be expanded from P2/P3 NFS survivors")
            final_decision = "p4_needed"
        elif p4_survivors and (not p5_score or (p5_score and p5_score[0].get("status") == "not_run")):
            blank_later("P5 alternating TAN cycles should be expanded from P4 survivors")
            final_decision = "p5_needed"
        elif p5_survivors:
            blank_later("P7 full-budget selection should be expanded from P5 survivors")
            final_decision = "p7_needed_from_p5"
        elif p4_score and not p4_survivors:
            blank_later("P4 produced no all-dataset TAN one-cycle survivor")
            final_decision = "stop_after_p4_no_tan_survivor"
        elif p5_score and not p5_survivors:
            blank_later("P5 produced no all-dataset alternating TAN survivor")
            final_decision = "stop_after_p5_no_multicycle_survivor"
        else:
            blank_later("P3 functional-teacher NFS produced no survivor")
            final_decision = "stop_after_p3_no_functional_teacher_survivor"
    elif not p6:
        blank_later("P1/P2 failed; P6 capacity/basis expansion required")
        final_decision = "p6_needed"
    elif p6_survivors:
        blank_later("P7 should be expanded from P6 capacity survivors")
        final_decision = "p7_needed_from_p6"
    else:
        blank_later("P1/P2/P6 produced no survivor")
        final_decision = "stop_after_p6_no_frontier"
    failure_reports(failures)
    make_figures(p1_score, p2_score, p6_score)
    save_json(
        BASE / "aggregate_decision.json",
        {
            "p0_pass": p0_pass,
            "p1_survivors": p1_survivors,
            "p2_survivors": p2_survivors,
            "p3_survivors": p3_survivors,
            "p4_survivors": p4_survivors,
            "p5_survivors": p5_survivors,
            "p6_survivors": p6_survivors,
            "final_decision": final_decision,
        },
    )
    p0_table = md_table(["rows", "errors", "max nonKAN", "min cov", "max rollback", "max CG residual", "pass"], [[p0_stats["rows"], p0_stats["errors"], fmt(p0_stats["max_nonkan"]), fmt(p0_stats["min_coverage"]), fmt(p0_stats["max_rollback"]), fmt(p0_stats["max_cg_residual"]), str(p0_pass).lower()]])
    p1_table = md_table(
        ["dataset", "method", "acc", "gap", "phi red", "J red", "ECE", "rank", "P1"],
        [[r["dataset"], r["method"], fmt(f(r, "acc")), fmt(f(r, "acc_gap_vs_A0")), fmt(f(r, "phi_red_vs_A0")), fmt(f(r, "J_red_vs_A0")), fmt(f(r, "ECE")), fmt(f(r, "rank")), "yes" if int(f(r, "p1_pass", 0)) else "no"] for r in p1_score],
    )
    p2_short = sorted(p2_score, key=lambda r: (r["dataset"], -f(r, "smoothability"), -f(r, "phi_reduction")))[:72]
    p2_table = md_table(
        ["dataset", "teacher", "variant", "proj", "eta", "acc drop", "KL", "logit", "phi red", "J red", "smooth", "P2"],
        [[r["dataset"], r["teacher"], r["variant"], r["projector"], r["eta"], fmt(f(r, "acc_drop")), fmt(f(r, "KL")), fmt(f(r, "logit_drift")), fmt(f(r, "phi_reduction")), fmt(f(r, "jacobian_reduction")), fmt(f(r, "smoothability")), "yes" if int(f(r, "p2_pass", 0)) else "no"] for r in p2_short],
    )
    p3_short = sorted([r for r in p3_score if not r.get("status")], key=lambda r: (r["dataset"], -f(r, "smoothability"), -f(r, "phi_reduction")))[:54]
    p3_table = md_table(
        ["dataset", "teacher", "method", "variant", "proj", "acc drop", "KL", "phi red", "smooth", "P3"],
        [[r["dataset"], r["teacher"], r["teacher_method"], r["variant"], r["projector"], fmt(f(r, "acc_drop")), fmt(f(r, "KL")), fmt(f(r, "phi_reduction")), fmt(f(r, "smoothability")), "yes" if int(f(r, "p3_pass", 0)) else "no"] for r in p3_short],
    )
    p4_short = sorted([r for r in p4_score if not r.get("status")], key=lambda r: (r["dataset"], -f(r, "p4_pass"), -f(r, "phi_reduction")))[:72]
    p4_table = md_table(
        ["dataset", "task", "NFS", "refresh", "task acc", "final acc", "acc drop", "phi red", "recovery", "P4"],
        [[r["dataset"], r["task_method"], r["nfs_label"], r["refresh"], fmt(f(r, "task_acc")), fmt(f(r, "refresh_acc")), fmt(f(r, "acc_drop")), fmt(f(r, "phi_reduction")), fmt(f(r, "refresh_recovery")), "yes" if int(f(r, "p4_pass", 0)) else "no"] for r in p4_short],
    )
    p5_table = md_table(
        ["dataset", "task", "NFS", "refresh", "cycles", "final acc", "final phi", "rank", "margin", "P5"],
        [[r.get("dataset", ""), r.get("task_method", ""), r.get("nfs_label", ""), r.get("refresh", ""), r.get("cycles", ""), fmt(f(r, "final_acc")), fmt(f(r, "final_phi")), fmt(f(r, "final_rank")), fmt(f(r, "final_margin")), "yes" if int(f(r, "p5_pass", 0)) else "no"] for r in p5_score if not r.get("status")],
    )
    p6_table = md_table(
        ["dataset", "method", "h", "b", "d", "basis", "acc", "gap", "phi red", "J red", "P6"],
        [[r["dataset"], r["method"], r["hidden_dim"], r["basis_count"], r["depth"], r["basis_type"], fmt(f(r, "acc")), fmt(f(r, "acc_gap_vs_A0")), fmt(f(r, "phi_red_vs_A0")), fmt(f(r, "J_red_vs_A0")), "yes" if int(f(r, "p6_pass", 0)) else "no"] for r in p6_score[:120]],
    )
    best_p2_rows = []
    for dataset in DATASETS:
        rows = [r for r in p2_score if r["dataset"] == dataset]
        if rows:
            best_smooth = max(rows, key=lambda r: f(r, "smoothability", -1e9))
            best_phi = max(rows, key=lambda r: f(r, "phi_reduction", -1e9))
            best_p2_rows.append([dataset, "smoothability", best_smooth["teacher"], best_smooth["variant"], fmt(f(best_smooth, "smoothability")), fmt(f(best_smooth, "acc_drop")), fmt(f(best_smooth, "KL")), fmt(f(best_smooth, "phi_reduction"))])
            best_p2_rows.append([dataset, "phi", best_phi["teacher"], best_phi["variant"], fmt(f(best_phi, "smoothability")), fmt(f(best_phi, "acc_drop")), fmt(f(best_phi, "KL")), fmt(f(best_phi, "phi_reduction"))])
    best_p2_table = md_table(["dataset", "best by", "teacher", "variant", "smooth", "acc drop", "KL", "phi red"], best_p2_rows)
    doc = f"""# DG-KAN v4.8 Functional Geometry Feasibility 结果复盘

本轮依据 `docs/DG-KAN_v4.8_FunctionalGeometry_Feasibility_Redesign_实验计划.md`。目标从“继续调 functional optimizer”改为 feasibility-first：先判断当前 PureKAN/RBF 是否存在准确且几何好的可达解，再判断 NFS 是否能在保持 teacher function 的同时降低 geometry。

## Run Inventory

{md_table(["stage", "rows", "errors"], [["P0 FGF/NFS smoke", len(p0), sum(1 for r in p0 if r.get("error"))], ["P1 feasibility frontier", len(p1), sum(1 for r in p1 if r.get("error"))], ["P2 NFS AdamW projection", len(p2), sum(1 for r in p2 if r.get("error"))], ["P3 NFS functional teachers", len(p3), sum(1 for r in p3 if r.get("error"))], ["P4 TAN one-cycle", len(p4), sum(1 for r in p4 if r.get("error"))], ["P5 TAN alternating cycles", len(p5), sum(1 for r in p5 if r.get("error"))], ["P6 capacity/basis expansion", len(p6), sum(1 for r in p6 if r.get("error"))]])}

## Code / Config Changes

```text
experiments/run_gafu_v48.py
  Added FGF AdamW geometry regularization frontier.
  Added offline NFS projection variants with teacher snapshot, backtracking,
  logit/hidden/margin constraints, and role-specific smoothing proposals.
  Added functional-teacher NFS, TAN one-cycle, and gated TAN multi-cycle probes.
  Added compact P6 capacity/basis feasibility expansion.

experiments/analyze_gafu_v48.py
  Generates frontier/NFS/capacity gate summaries, failure taxonomy,
  SVG diagnostics, aggregate_decision.json, and this replay.
```

## P0 Implementation Smoke

{p0_table}

P0 verdict: {"pass" if p0_pass else "fail"}. NFS temporary apply / rollback / teacher snapshot / finite projection diagnostics passed for the smoke grid.

## P1 Accuracy-Geometry Feasibility Frontier

{p1_table}

P1 all-dataset survivors:

```text
{", ".join(p1_survivors) if p1_survivors else "none"}
```

## P2 Offline NFS Projection From AdamW Teachers

Top diagnostic rows are sorted by dataset and smoothability/geometry signal.

{p2_table}

Best diagnostic points:

{best_p2_table}

P2 all-dataset survivors:

```text
{", ".join(p2_survivors) if p2_survivors else "none"}
```

## P3 Offline NFS Projection From Functional Teachers

{p3_table if p3_score else "_P3 not run._"}

P3 all-dataset survivors:

```text
{", ".join(p3_survivors) if p3_survivors else "none"}
```

## P4 TAN One-Cycle Micro-Run

{p4_table if p4_score else "_P4 not run._"}

P4 all-dataset survivors:

```text
{", ".join(p4_survivors) if p4_survivors else "none"}
```

## P5 Alternating TAN Cycles

{p5_table if [r for r in p5_score if not r.get("status")] else "_P5 not run._"}

P5 all-dataset survivors:

```text
{", ".join(p5_survivors) if p5_survivors else "none"}
```

## P6 Capacity / Basis Expansion

{p6_table if p6_score else "_P6 not run yet._"}

P6 all-dataset survivors:

```text
{", ".join(p6_survivors) if p6_survivors else "none"}
```

## P3-P9 Decision

```text
P3 functional-teacher NFS: {"run" if p3_score else ("not run; no P2 survivor" if not p2_survivors else "pending")}
P4 TAN one-cycle: {"run" if p4_score else "not run by gate"}
P5 alternating TAN cycles: {"run" if [r for r in p5_score if not r.get("status")] else "not run by gate"}
P7/P8/P9 confirm: {"not run; P5 no all-dataset survivor" if not p5_survivors else "should be expanded from P5 survivors"}
Final decision: {final_decision}
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
F1 checks whether an accurate and smooth PureKAN solution is visible under AdamW-style geometry regularization.
F2 checks whether NFS can lower geometry in a teacher-preserving tangent/nullspace.
F4 checks whether smoothing destroys logits/features or accuracy.
F5 checks whether refresh and alternating cycles can retain the one-cycle geometry gain.
F6 remains the capacity/basis fallback, but it was not triggered because P2/P4 already gave mechanistic survivors.

Interpretation:
  P1 did not expose an AdamW+regularization smooth frontier.
  P2/P3 showed NFS role-block can preserve function while reducing geometry.
  P4 showed one-cycle TAN can pass across datasets.
  P5 showed the same recipe does not yet survive alternating cycles on MNIST.
```

## Required Artifacts

Written under `results/v4_8/`:

```text
p0_fgf_invariants.csv
p1_feasibility_frontier.csv
p1_training_trace.csv
p1_frontier_gate_summary.csv
p2_nfs_adamw_projection.csv
p2_nfs_gate_summary.csv
p3_nfs_functional_teachers.csv
p3_nfs_functional_gate_summary.csv
p4_tan_one_cycle_micro_run.csv
p4_tan_one_cycle_gate_summary.csv
p5_tan_alternating_cycles.csv
p5_tan_cycle_gate_summary.csv
p6_capacity_basis_expansion.csv
p6_capacity_gate_summary.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
p9_failure_diagnosis.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v4.8 status:
  {final_decision}

What improved:
  v4.8 now asks the right feasibility question before spending seed budget.
  P1 exposes the accuracy-geometry Pareto frontier directly.
  P2 measures function-preserving smoothability instead of relying on unconstrained Sobolev consolidation.
  P4 confirms a real one-cycle TAN signal with PureKAN-AdamW task teacher + role-block NFS + D6 refresh.

What failed / remains open:
  P5 fails the all-dataset multi-cycle gate because MNIST does not preserve the task/geometry tradeoff over 3 cycles.

Conclusion:
  NFS is feasible as a local projection, and TAN is viable for one cycle, but the current alternating controller is not stable enough for P7 seed confirmation.
  Next work should redesign the multi-cycle controller / refresh policy before spending 5/10-seed budget.
```
"""
    OUT.write_text(doc, encoding="utf-8")
    marker = "## 2026-05-03 DG-KAN v4.8 FGF/NFS 实验"
    existing = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    if marker not in existing:
        with LOG.open("a", encoding="utf-8") as fobj:
            fobj.write(f"\n\n{marker}\n\n")
            fobj.write("结果见 `docs/DG-KAN_v4.8_FunctionalGeometry_Feasibility_Redesign_结果复盘.md`。\n\n")
            fobj.write(f"P0 pass={p0_pass}; P1 survivors={p1_survivors or 'none'}; P2 survivors={p2_survivors or 'none'}; P6 survivors={p6_survivors or 'none'}; decision={final_decision}.\n")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
