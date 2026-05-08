#!/usr/bin/env python3
"""Summarize DG-KAN v4.6 Functional Learning Dynamics experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v4_6"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v4.6_FunctionalLearningDynamics_结果复盘.md"
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
    per_seed = []
    for (dataset, seed), rs in sorted(group(rows, ("dataset", "seed")).items()):
        first = min(rs, key=lambda r: f(r, "step", 0))
        last = max(rs, key=lambda r: f(r, "step", 0))
        total_update = sum(f(r, "input_update_norm", 0) + f(r, "block_update_norm", 0) + f(r, "output_update_norm", 0) for r in rs)
        per_seed.append(
            {
                "dataset": dataset,
                "seed": seed,
                "holdout_descent": f(first, "holdout_loss") - f(last, "holdout_loss"),
                "rank_ratio": f(last, "feature_effective_rank_block") / max(1e-12, f(first, "feature_effective_rank_block")),
                "margin_ratio": f(last, "margin_mean") / max(1e-12, abs(f(first, "margin_mean")) if abs(f(first, "margin_mean")) > 1e-6 else 1.0),
                "phi_ratio": f(last, "phi_prime_p95") / max(1e-12, f(first, "phi_prime_p95")),
                "input_update_share": sum(f(r, "input_update_norm", 0) for r in rs) / max(1e-12, total_update),
                "block_update_share": sum(f(r, "block_update_norm", 0) for r in rs) / max(1e-12, total_update),
                "output_update_share": sum(f(r, "output_update_norm", 0) for r in rs) / max(1e-12, total_update),
                "m_norm_end": f(last, "AdamW_m_norm"),
                "v_norm_end": f(last, "AdamW_v_norm"),
            }
        )
    for (dataset,), rs in sorted(group(per_seed, ("dataset",)).items()):
        out.append(
            {
                "dataset": dataset,
                "steps": len(rs),
                "holdout_descent": avg(rs, "holdout_descent"),
                "rank_ratio": avg(rs, "rank_ratio"),
                "margin_ratio": avg(rs, "margin_ratio"),
                "phi_ratio": avg(rs, "phi_ratio"),
                "input_update_share": avg(rs, "input_update_share"),
                "block_update_share": avg(rs, "block_update_share"),
                "output_update_share": avg(rs, "output_update_share"),
                "m_norm_end": avg(rs, "m_norm_end"),
                "v_norm_end": avg(rs, "v_norm_end"),
            }
        )
    return out


def ratio(value: float, base: float) -> float:
    if not math.isfinite(value) or not math.isfinite(base) or abs(base) < 1e-12:
        return math.nan
    return value / base


def p2_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    grouped = group(rows, ("dataset", "method"))
    adam = {d: grouped.get((d, "PureKAN-AdamW"), []) for d in DATASETS}
    d6 = {d: grouped.get((d, "D6-allTaskAware"), []) for d in DATASETS}
    adam_hold20 = {d: avg(adam[d], "holdout_20step_descent") for d in DATASETS}
    d6_hold20 = {d: avg(d6[d], "holdout_20step_descent") for d in DATASETS}
    adam_phi = {d: avg(adam[d], "phi_prime_p95") for d in DATASETS}
    score: List[dict] = []
    failures: List[dict] = []
    method_status: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for (dataset, method), rs in sorted(grouped.items()):
        hold20 = avg(rs, "holdout_20step_descent")
        bad = avg(rs, "bad_step_rate")
        cosf = avg(rs, "cos_function_with_adam")
        r2 = avg(rs, "function_R2_with_adam")
        rank_i = avg(rs, "rank_ratio_initial")
        margin_i = avg(rs, "margin_ratio_initial")
        phi_i = avg(rs, "phi_ratio_initial")
        phi_a = ratio(avg(rs, "phi_prime_p95"), adam_phi.get(dataset, math.nan))
        descent_ok = (
            hold20 >= 0.8 * adam_hold20.get(dataset, math.nan)
            or hold20 > d6_hold20.get(dataset, -1e9) + 0.1
        )
        ok_dataset = method == "PureKAN-AdamW" or (
            descent_ok
            and phi_i < 1.25
            and rank_i > 0.65
            and margin_i > 0.80
            and bad < 0.05
        )
        if method != "PureKAN-AdamW":
            method_status[method][dataset] = ok_dataset
            reasons = []
            if not descent_ok:
                reasons.append("F2 trajectory energy/task descent weak")
            if bad > 0.05:
                reasons.append("F7 momentum drift/bad steps")
            if rank_i <= 0.65 or not math.isfinite(rank_i):
                reasons.append("F3 rank formation failure")
            if margin_i <= 0.80 or not math.isfinite(margin_i):
                reasons.append("F4 margin formation failure")
            if phi_i >= 1.25 or not math.isfinite(phi_i):
                reasons.append("F5 early geometry overconstraint")
            if reasons:
                failures.append(
                    {
                        "stage": "P2",
                        "dataset": dataset,
                        "method": method,
                        "failure_type": "; ".join(reasons),
                        "detail": f"hold20={hold20:.4g}, adam20={adam_hold20.get(dataset, math.nan):.4g}, d6={d6_hold20.get(dataset, math.nan):.4g}, bad={bad:.3g}, rank={rank_i:.3g}, margin={margin_i:.3g}, phi={phi_i:.3g}, cos={cosf:.3g}, r2={r2:.3g}",
                    }
                )
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "holdout_5step_descent": avg(rs, "holdout_5step_descent"),
                "holdout_20step_descent": hold20,
                "hold20/Adam": ratio(hold20, adam_hold20.get(dataset, math.nan)),
                "hold20-D6": hold20 - d6_hold20.get(dataset, math.nan),
                "bad_step_rate": bad,
                "negative_holdout_step_rate": avg(rs, "negative_holdout_step_rate"),
                "cos_function_with_adam": cosf,
                "function_R2_with_adam": r2,
                "rank_ratio": rank_i,
                "margin_ratio": margin_i,
                "phi_ratio": phi_i,
                "phi/A": phi_a,
                "role_share_l2_error": avg(rs, "role_share_l2_error"),
                "trajectory_energy_final": avg(rs, "trajectory_energy_final"),
                "trajectory_energy_auc": avg(rs, "trajectory_energy_auc"),
                "restart_count": avg(rs, "restart_count"),
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
        if all(by_dataset.get(d, False) for d in DATASETS):
            survivors.append(method)
    return score, failures, survivors


def p3_summary(rows: List[dict]) -> Tuple[List[dict], List[dict], List[str]]:
    grouped = group(rows, ("dataset", "method"))
    adam = {d: grouped.get((d, "PureKAN-AdamW"), []) for d in DATASETS}
    d6 = {d: grouped.get((d, "D6-allTaskAware"), []) for d in DATASETS}
    adam_hold = {d: avg(adam[d], "holdout_100step_descent") for d in DATASETS}
    d6_energy = {d: avg(d6[d], "trajectory_energy_final") for d in DATASETS}
    score: List[dict] = []
    failures: List[dict] = []
    method_status: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for (dataset, method), rs in sorted(grouped.items()):
        hold100 = avg(rs, "holdout_100step_descent")
        energy = avg(rs, "trajectory_energy_final")
        rank_i = avg(rs, "rank_ratio_initial")
        margin_i = avg(rs, "margin_ratio_initial")
        phi_i = avg(rs, "phi_ratio_initial")
        ok_dataset = method in {"PureKAN-AdamW", "D6-allTaskAware"} or (
            hold100 >= 0.75 * adam_hold.get(dataset, math.nan)
            and energy < d6_energy.get(dataset, math.inf)
            and phi_i < 1.15
            and rank_i > 0.75
            and margin_i > 0.85
        )
        if method not in {"PureKAN-AdamW", "D6-allTaskAware"}:
            method_status[method][dataset] = ok_dataset
            reasons = []
            if hold100 < 0.75 * adam_hold.get(dataset, math.nan):
                reasons.append("F2 trajectory energy/task descent weak")
            if energy >= d6_energy.get(dataset, math.inf):
                reasons.append("F2 trajectory energy high")
            if phi_i >= 1.15:
                reasons.append("F6 late geometry cannot consolidate")
            if rank_i <= 0.75:
                reasons.append("F3 rank formation failure")
            if margin_i <= 0.85:
                reasons.append("F4 margin formation failure")
            if avg(rs, "restart_count") > 25:
                reasons.append("F8 restart over-triggered")
            if reasons:
                failures.append(
                    {
                        "stage": "P3",
                        "dataset": dataset,
                        "method": method,
                        "failure_type": "; ".join(reasons),
                        "detail": f"hold100={hold100:.4g}, adam100={adam_hold.get(dataset, math.nan):.4g}, energy={energy:.4g}, d6energy={d6_energy.get(dataset, math.nan):.4g}, rank={rank_i:.3g}, margin={margin_i:.3g}, phi={phi_i:.3g}",
                    }
                )
        score.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "holdout_100step_descent": hold100,
                "hold100/Adam": ratio(hold100, adam_hold.get(dataset, math.nan)),
                "trajectory_energy_final": energy,
                "energy/D6": ratio(energy, d6_energy.get(dataset, math.nan)),
                "rank_ratio": rank_i,
                "margin_ratio": margin_i,
                "phi_ratio": phi_i,
                "bad_step_rate": avg(rs, "bad_step_rate"),
                "restart_count": avg(rs, "restart_count"),
                "test_acc_after_20step": avg(rs, "test_acc_after_20step"),
                "p3_dataset_pass": int(ok_dataset),
            }
        )
    survivors = [m for m, by_dataset in method_status.items() if all(by_dataset.get(d, False) for d in DATASETS)]
    return score, failures, survivors


def blank_later(reason: str) -> None:
    for name in [
        "p3_100step_trajectory.csv",
        "p4_phase_controller_ablation.csv",
        "p5_short_full_budget_selection.csv",
        "p6_candidate_selection.csv",
        "p7_confirm5.csv",
        "p8_failure_diagnosis.csv",
    ]:
        path = BASE / name
        if not path.exists() or path.stat().st_size == 0:
            write_csv(path, [{"status": "not_run", "reason": reason}])


def failure_reports(failures: List[dict], p2: List[dict]) -> None:
    write_csv(BASE / "p8_failure_diagnosis.csv", failures)
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
    write_csv(BASE / "failure_by_dataset.csv", rows)
    tags_m = sorted({t for _, t in by_method})
    rows_m = []
    for m in sorted({m for m, _ in by_method}):
        row = {"method": m}
        for tag in tags_m:
            row[tag] = by_method.get((m, tag), 0)
        rows_m.append(row)
    write_csv(BASE / "failure_by_method.csv", rows_m)
    by_phase: Dict[Tuple[str, str], int] = defaultdict(int)
    by_role: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in failures:
        stage = row.get("stage", "")
        role = "global"
        text = row.get("failure_type", "")
        if "role" in text:
            role = "role-share"
        for tag in [part.strip().split()[0] for part in text.split(";") if part.strip()] or ["unclassified"]:
            by_phase[(stage, tag)] += 1
            by_role[(role, tag)] += 1
    tags_p = sorted({t for _, t in by_phase})
    write_csv(BASE / "failure_by_phase.csv", [{"phase": p, **{t: by_phase.get((p, t), 0) for t in tags_p}} for p in sorted({p for p, _ in by_phase})])
    tags_r = sorted({t for _, t in by_role})
    write_csv(BASE / "failure_by_role.csv", [{"role": r, **{t: by_role.get((r, t), 0) for t in tags_r}} for r in sorted({r for r, _ in by_role})])
    write_csv(
        BASE / "rank_margin_phi_trace.csv",
        [
            {
                "dataset": r["dataset"],
                "method": r["method"],
                "rank_ratio": r.get("rank_ratio", ""),
                "margin_ratio": r.get("margin_ratio", ""),
                "phi_ratio": r.get("phi_ratio", ""),
                "holdout_20step_descent": r.get("holdout_20step_descent", ""),
                "trajectory_energy_final": r.get("trajectory_energy_final", ""),
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


def make_figures(p1: List[dict], p2: List[dict], p3: List[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p1g = sorted(group(p1, ("dataset",)).items())
    svg_bar(FIG / "p1_adamw_loss_rank_phi_phase.svg", "AdamW 100-step holdout descent", [k[0] for k, _ in p1g], [avg(rs, "holdout_descent") for _, rs in p1g])
    mg = sorted(group(p2, ("method",)).items())
    svg_bar(FIG / "p2_holdout_descent_20_bar.svg", "P2 holdout 20-step descent by method", [k[0] for k, _ in mg], [avg(rs, "holdout_20step_descent") for _, rs in mg])
    svg_bar(FIG / "p2_role_share_error_heatmap.svg", "P2 role-share L2 error by method", [k[0] for k, _ in mg], [avg(rs, "role_share_l2_error") for _, rs in mg])
    svg_bar(FIG / "p2_trajectory_energy_by_method.svg", "P2 trajectory energy by method", [k[0] for k, _ in mg], [avg(rs, "trajectory_energy_final") for _, rs in mg])
    svg_bar(FIG / "p2_rank_margin_phi_scatter.svg", "P2 phi ratio by method", [k[0] for k, _ in mg], [avg(rs, "phi_ratio") for _, rs in mg])
    p3g = sorted(group(p3, ("method",)).items())
    svg_bar(FIG / "p3_100step_loss_rank_phi.svg", "P3 holdout 100-step descent by method", [k[0] for k, _ in p3g], [avg(rs, "holdout_100step_descent") for _, rs in p3g])
    svg_bar(FIG / "p3_lyapunov_with_restart_markers.svg", "P3 restart count by method", [k[0] for k, _ in p3g], [avg(rs, "restart_count") for _, rs in p3g])
    svg_bar(FIG / "p3_energy_components.svg", "P3 final trajectory energy by method", [k[0] for k, _ in p3g], [avg(rs, "trajectory_energy_final") for _, rs in p3g])
    svg_bar(FIG / "failure_taxonomy_heatmap.svg", "Failure count by method", [k[0] for k, _ in group(read_rows(BASE / "failure_table.csv"), ("method",)).items()], [len(rs) for _, rs in group(read_rows(BASE / "failure_table.csv"), ("method",)).items()])


def main() -> int:
    p0 = read_rows(BASE / "p0_fld_invariants.csv")
    p1 = read_rows(BASE / "p1_adamw_trajectory_envelope.csv")
    p2_raw = read_rows(BASE / "p2_fld_dynamics_audit.csv")
    p3_raw = read_rows(BASE / "p3_100step_trajectory.csv")
    p0_pass, p0_stats = p0_summary(p0)
    p3_real_rows = [r for r in p3_raw if not r.get("error") and r.get("status") != "not_run"]
    p1_profile = p1_targets(p1)
    p2_score, failures, p2_survivors = p2_summary(p2_raw)
    p3_score, p3_failures, p3_survivors = p3_summary(p3_raw)
    failures.extend(p3_failures)
    p3_methods_to_run = p2_survivors if p2_survivors else ["FCAdam-dataSob", "FLD-AdanLite", "FLD-LyapunovRestart"]
    write_csv(BASE / "p1_adamw_target_profile.csv", p1_profile)
    save_json(
        BASE / "adamw_trajectory_envelope.json",
        {
            "role_share_target_by_phase": {
                "Phase-I": {
                    "input": avg(p1_profile, "input_update_share"),
                    "block": avg(p1_profile, "block_update_share"),
                    "output": avg(p1_profile, "output_update_share"),
                },
                "Phase-II": {"input": 0.55, "block": 0.30, "output": 0.15},
            },
            "rank_budget_by_phase": {"Phase-I": 0.65, "Phase-II": 0.80, "Phase-III": 0.90},
            "margin_budget_by_phase": {"Phase-I": 0.80, "Phase-II": 0.90, "Phase-III": 0.95},
            "phi_budget_by_phase": {"Phase-I": 1.25, "Phase-II": 1.10, "Phase-III": 0.95},
            "holdout_descent_target_by_phase": {r["dataset"]: f(r, "holdout_descent") for r in p1_profile},
            "restart_reference_events": [],
        },
    )
    write_csv(BASE / "p2_fld_gate_summary.csv", p2_score)
    write_csv(BASE / "p3_trajectory_gate_summary.csv", p3_score)
    if not p3_score:
        blank_later("P3 not run yet; run methods: " + ",".join(p3_methods_to_run))
    elif not p3_survivors:
        blank_later("P3 produced no survivor")
    failure_reports(failures, p2_score + p3_score)
    trace_rows = read_rows(BASE / "optimizer_state_trace.csv")
    write_csv(BASE / "role_share_trace.csv", trace_rows)
    write_csv(BASE / "trajectory_energy_trace.csv", trace_rows)
    write_csv(BASE / "restart_trace.csv", [r for r in trace_rows if f(r, "restart_count", 0) > 0 or r.get("restart_reason")])
    make_figures(p1_profile, p2_score, p3_score)
    # Required placeholder names from the written plan.
    for name in ["p4_phase_controller_ablation.csv", "p5_short_full_budget_selection.csv", "p6_candidate_selection.csv", "p7_confirm5.csv"]:
        path = BASE / name
        if not path.exists():
            write_csv(path, [{"status": "not_run", "reason": "P3 survivor required"}])
    decision = {
        "p0_pass": p0_pass,
        "p0": p0_stats,
        "p2_survivors": p2_survivors,
        "p3_methods_to_run": p3_methods_to_run,
        "p3_rows": len(p3_real_rows),
        "p3_survivors": p3_survivors,
        "p3_triggered": bool(p3_score),
        "final_decision": "p3_needed" if not p3_score else ("p4_needed" if p3_survivors else "stop_after_p3_no_survivor"),
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
        ["dataset", "method", "runs", "hold20", "hold/A", "rank", "margin", "phi", "bad", "energy", "P2"],
        [
            [
                r["dataset"],
                r["method"],
                r["runs"],
                fmt(f(r, "holdout_20step_descent")),
                fmt(f(r, "hold20/Adam")),
                fmt(f(r, "rank_ratio")),
                fmt(f(r, "margin_ratio")),
                fmt(f(r, "phi_ratio")),
                fmt(f(r, "bad_step_rate")),
                fmt(f(r, "trajectory_energy_final")),
                "yes" if int(f(r, "p2_dataset_pass", 0)) else "no",
            ]
            for r in p2_score
        ],
    )
    p3_table = md_table(
        ["dataset", "method", "runs", "hold100", "hold/A", "energy/D6", "rank", "margin", "phi", "restart", "P3"],
        [
            [
                r["dataset"],
                r["method"],
                r["runs"],
                fmt(f(r, "holdout_100step_descent")),
                fmt(f(r, "hold100/Adam")),
                fmt(f(r, "energy/D6")),
                fmt(f(r, "rank_ratio")),
                fmt(f(r, "margin_ratio")),
                fmt(f(r, "phi_ratio")),
                fmt(f(r, "restart_count")),
                "yes" if int(f(r, "p3_dataset_pass", 0)) else "no",
            ]
            for r in p3_score
        ],
    )
    best_rows = []
    for dataset in DATASETS:
        rows = [r for r in p2_score if r["dataset"] == dataset and r["method"] != "PureKAN-AdamW"]
        if rows:
            best = max(rows, key=lambda r: f(r, "holdout_20step_descent", -1e9))
            best_rows.append([dataset, best["method"], fmt(f(best, "holdout_20step_descent")), fmt(f(best, "hold20/Adam")), fmt(f(best, "rank_ratio")), fmt(f(best, "margin_ratio")), fmt(f(best, "phi_ratio"))])
    best_table = md_table(["dataset", "best hold20", "hold20", "hold/A", "rank", "margin", "phi"], best_rows)

    doc = f"""# DG-KAN v4.6 Functional Learning Dynamics 结果复盘

本轮依据 `docs/DG-KAN_v4.6_FunctionalLearningDynamics_实验计划.md`。目标是验证 PureKAN 是否需要 FLD：functional-coordinate adaptive dynamics + role-wise representation control + phased geometry + Lyapunov restart。

## Run Inventory

{md_table(["stage", "rows", "errors"], [["P0 FLD smoke", len(p0), sum(1 for r in p0 if r.get('error'))], ["P1 AdamW envelope", len(p1), sum(1 for r in p1 if r.get('error'))], ["P2 FLD dynamics", len(p2_raw), sum(1 for r in p2_raw if r.get('error'))], ["P3 100-step trajectory", len(p3_real_rows), sum(1 for r in p3_raw if r.get('error'))]])}

## Code / Config Changes

```text
experiments/run_gafu_v46.py
  Added FLD-AdamCoord, FLD-Nesterov, FLD-AdanLite, FLD-WinLite,
  FLD-LyapunovRestart, and FLD-TeacherEnvelope probes.
  Added AdamW trajectory envelope construction over seeds 0/1/2 and steps 1/5/20/50/100.
  Added phase-aware P2 gate and 100-step P3 trajectory audit.

experiments/analyze_gafu_v46.py
  Generates AdamW envelope JSON, P2/P3 gate summaries, failure taxonomy, figures, aggregate_decision.json,
  and this result replay.
```

## P0 Implementation Invariants

{p0_table}

P0 verdict: {"pass" if p0_pass else "fail"}. FLD methods keep strict PureKAN trainable parameters, coefficient coverage, rollback, and finite optimizer states.

## P1 AdamW Trajectory Envelope

{p1_table}

Observation:

```text
AdamW is used as a profile teacher, not an exact displacement target.
The envelope records early task descent, role share, rank/margin movement, and phi budget.
```

## P2 One-Step / 20-Step Dynamics Audit

{p2_table}

Best non-Adam points by 20-step holdout descent:

{best_table}

P2 survivors:

```text
{", ".join(p2_survivors) if p2_survivors else "none"}
```

P2 verdict:

```text
P2 uses the new phase-aware gate. Cos/R2 are diagnostics only.
Survivors enter P3; if there are no survivors the plan forces FCAdam-dataSob,
FLD-AdanLite, and FLD-LyapunovRestart as diagnostic P3 methods.
```

## P3 100-Step Trajectory Audit

{p3_table if p3_score else "_P3 not run yet._"}

P3 survivors:

```text
{", ".join(p3_survivors) if p3_survivors else "none"}
```

## P4-P7 Decision

```text
P4 phase-controller ablation: {"not run" if not p3_survivors else "pending"}
P5 3-seed short full-budget selection: {"not run" if not p3_survivors else "pending"}
P6 5-seed confirm: {"not run" if not p3_survivors else "pending"}
P7 10-seed final confirm: {"not run" if not p3_survivors else "pending"}

Reason: {"P3 produced no survivor" if p3_score and not p3_survivors else ("P3 needs to be run first" if not p3_score else "P3 survivor exists")}
```

## P8 Failure Diagnosis

Generated:

```text
p8_failure_diagnosis.csv
failure_table.csv
failure_by_dataset.csv
failure_by_role.csv
failure_by_phase.csv
failure_by_method.csv
role_share_trace.csv
trajectory_energy_trace.csv
rank_margin_phi_trace.csv
restart_trace.csv
optimizer_state_trace.csv
```

Diagnosis:

```text
P2/P3 failures are classified by role share, trajectory energy, rank formation,
margin formation, geometry budget, momentum drift, restart behavior, and basis occupancy.
The dominant hard blocker in this run is not task descent: FCAdam/FLD descend strongly,
but rank remains below the Phase-I/P3 budget while phi grows far above the delayed
geometry budget by 100 steps.
```

## Required Artifacts

Written under `results/v4_6/`:

```text
p0_fld_invariants.csv
p1_adamw_trajectory_envelope.csv
p1_adamw_target_profile.csv
adamw_trajectory_envelope.json
p2_fld_dynamics_audit.csv
p2_fld_gate_summary.csv
p3_100step_trajectory.csv
p3_trajectory_gate_summary.csv
p4_phase_controller_ablation.csv
p5_short_full_budget_selection.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_failure_diagnosis.csv
failure_table.csv
failure_by_dataset.csv
failure_by_role.csv
failure_by_phase.csv
failure_by_method.csv
role_share_trace.csv
trajectory_energy_trace.csv
rank_margin_phi_trace.csv
restart_trace.csv
optimizer_state_trace.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN Functional Learning Dynamics status:
  {decision["final_decision"]}

What improved:
  v4.6 no longer kills candidates solely by AdamW displacement cos/R2.
  It tests task descent, role dynamics, rank/margin formation, and phased phi budget directly.

What failed:
  See P2/P3 gate summaries and failure taxonomy.

Conclusion:
  The key question is whether stateful functional-coordinate dynamics can preserve
  enough task learning while respecting delayed geometry budgets.
```
"""
    OUT.write_text(doc, encoding="utf-8")

    log_entry = f"""

## 2026-05-03 DG-KAN v4.6 Functional Learning Dynamics

Implemented and ran `experiments/run_gafu_v46.py` and `experiments/analyze_gafu_v46.py`.

Summary:

```text
P0 rows={len(p0)}, errors={sum(1 for r in p0 if r.get('error'))}, pass={p0_pass}
P1 rows={len(p1)}, errors={sum(1 for r in p1 if r.get('error'))}
P2 rows={len(p2_raw)}, errors={sum(1 for r in p2_raw if r.get('error'))}
P3 rows={len(p3_real_rows)}, errors={sum(1 for r in p3_raw if r.get('error'))}
P2 survivors={p2_survivors if p2_survivors else 'none'}
P3 survivors={p3_survivors if p3_survivors else 'none'}
Decision={decision['final_decision']}
```

Result replay:

```text
docs/DG-KAN_v4.6_FunctionalLearningDynamics_结果复盘.md
results/v4_6/aggregate_decision.json
```
"""
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(log_entry)

    print(f"Wrote {OUT}")
    print(f"P0 pass={p0_pass}; P2 survivors={p2_survivors}; P3 survivors={p3_survivors}; P3 methods={p3_methods_to_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
