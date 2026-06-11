#!/usr/bin/env python3
"""v22.08 retained-source observer no-commit audit."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_08_common import (  # noqa: E402
    PYTHON,
    V2207_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


SCORE_DESCRIPTIONS = {
    "C-O1_drift_diffusion_retention": "coherent train split drift over diffusion/noise and class-density consistency",
    "C-O2_micro_trajectory_invariance": "random/sign/corrupt control gap stability plus B2/B3 train split transfer",
    "C-O3_low_NDS_low_curvature": "low NDS score with first-order transfer and solver residual penalty",
    "C-O4_info_volume_no_fold": "information-volume/no-fold score plus signal-channel over reservoir separation",
    "C-O5_control_nullspace": "control-null residual proxy from target/control gaps and reservoir leakage penalty",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2207_OFFICIAL))
    p.add_argument("--top-k", type=int, default=20)
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _rank(values: list[float]) -> list[float]:
    pairs = sorted((v, i) for i, v in enumerate(values))
    out = [0.0] * len(values)
    pos = 0
    while pos < len(pairs):
        end = pos + 1
        while end < len(pairs) and pairs[end][0] == pairs[pos][0]:
            end += 1
        avg = 0.5 * (pos + end - 1) + 1.0
        for _v, idx in pairs[pos:end]:
            out[idx] = avg
        pos = end
    return out


def _auc(scores: list[float], labels: list[int]) -> float | str:
    valid = [(s, y) for s, y in zip(scores, labels) if math.isfinite(s)]
    if not valid:
        return ""
    scores = [s for s, _y in valid]
    labels = [y for _s, y in valid]
    pos = sum(labels)
    neg = len(labels) - pos
    if pos <= 0 or neg <= 0:
        return ""
    ranks = _rank(scores)
    pos_rank = sum(r for r, y in zip(ranks, labels) if y)
    return (pos_rank - pos * (pos + 1) / 2.0) / float(pos * neg)


def _precision(scores: list[float], labels: list[int], controls: list[int], k: int) -> tuple[float | str, float | str, float | str, list[int]]:
    valid = [(i, s) for i, s in enumerate(scores) if math.isfinite(s)]
    if not valid:
        return "", "", "", []
    order = [i for i, _s in sorted(valid, key=lambda item: item[1], reverse=True)[: min(k, len(valid))]]
    if not order:
        return "", "", "", []
    hits = sum(labels[i] for i in order)
    total_pos = sum(labels)
    control_fraction = sum(controls[i] for i in order) / float(len(order))
    return hits / float(len(order)), hits / float(total_pos) if total_pos else "", control_fraction, order


def _zscore(values: list[float], groups: list[tuple[str, str]]) -> list[float]:
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    for group in sorted(set(groups)):
        vals = [v for v, g in zip(values, groups) if g == group and math.isfinite(v)]
        if not vals:
            stats[group] = (0.0, 1.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        stats[group] = (mean, math.sqrt(var) if var > 1.0e-12 else 1.0)
    return [((_f(v, 0.0) - stats[g][0]) / stats[g][1]) for v, g in zip(values, groups)]


def _join(source_dir: Path) -> list[dict[str, Any]]:
    c0 = read_rows(source_dir / "v22_07_c0_no_commit_source_estimator_matrix.csv")
    c1_by_order = {str(r.get("job_order", "")): r for r in read_rows(source_dir / "v22_07_c1_target_contrast_matrix.csv")}
    c2_by_order = {str(r.get("job_order", "")): r for r in read_rows(source_dir / "v22_07_c2_metric_solver_matrix.csv")}
    rows: list[dict[str, Any]] = []
    for row in c0:
        if str(row.get("execution_status")) != "measured":
            continue
        merged = dict(row)
        c1 = c1_by_order.get(str(row.get("job_order", "")), {})
        c2 = c2_by_order.get(str(row.get("job_order", "")), {})
        for key in [
            "B2_transfer_gain",
            "B3_safety_gain",
            "random_target_gap",
            "sign_flip_gap",
            "corrupt_gap",
            "target_signal_projection",
            "target_reservoir_projection",
            "target_NDS",
            "target_norm",
        ]:
            merged[key] = c1.get(key, "")
        for key in ["projection_residual_Gf", "ActuationR2", "condition_estimate", "solve_time_ms"]:
            merged[key] = c2.get(key, "")
        rows.append(merged)
    return rows


def _build_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [(str(r.get("dataset", "")), str(r.get("seed", ""))) for r in rows]
    keys = [
        "E1_train_split_B2_transfer",
        "E2_gradient_drift_snr",
        "E3_signal_reservoir_projection",
        "E4_class_balanced_density",
        "E7_low_NDS_score",
        "E8_info_volume_no_fold",
        "B2_transfer_gain",
        "B3_safety_gain",
        "random_target_gap",
        "sign_flip_gap",
        "corrupt_gap",
        "target_signal_projection",
        "target_reservoir_projection",
        "projection_residual_Gf",
        "ActuationR2",
        "condition_estimate",
    ]
    z = {key: _zscore([_f(r.get(key)) for r in rows], groups) for key in keys}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        min_gap = min(z["random_target_gap"][i], z["sign_flip_gap"][i], z["corrupt_gap"][i])
        signal_gap = z["target_signal_projection"][i] - z["target_reservoir_projection"][i]
        co1 = z["E2_gradient_drift_snr"][i] + 0.35 * z["E4_class_balanced_density"][i] + 0.25 * signal_gap
        co2 = 0.45 * z["B2_transfer_gain"][i] + 0.30 * z["B3_safety_gain"][i] + 0.65 * min_gap + 0.20 * z["E1_train_split_B2_transfer"][i]
        co3 = z["E7_low_NDS_score"][i] + 0.25 * z["B2_transfer_gain"][i] - 0.30 * z["projection_residual_Gf"][i]
        co4 = z["E8_info_volume_no_fold"][i] + 0.45 * signal_gap - 0.15 * z["condition_estimate"][i]
        co5 = min_gap + 0.35 * signal_gap - 0.25 * z["target_reservoir_projection"][i]
        source = {h: _f(row.get(f"future_audit_source_h{h}"), -999.0) for h in [100, 400, 800, 1600, 3200]}
        item = {
            "job_order": row.get("job_order", ""),
            "v21_id": row.get("v21_id", ""),
            "mechanism": row.get("mechanism", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "is_control": row.get("is_control", ""),
            "future_audit_source_h100": row.get("future_audit_source_h100", ""),
            "future_audit_source_h400": row.get("future_audit_source_h400", ""),
            "future_audit_source_h800": row.get("future_audit_source_h800", ""),
            "future_audit_source_h1600": row.get("future_audit_source_h1600", ""),
            "future_audit_source_h3200": row.get("future_audit_source_h3200", ""),
            "official_early_chain_h100_h400_h800_positive": int(source[100] >= 0.005 and source[400] >= 0.005 and source[800] >= 0.005),
            "retained_h800_h3200_positive": int(source[800] >= 0.005 and source[3200] >= 0.005),
            "h3200_positive": int(source[3200] >= 0.005),
            "C-O1_drift_diffusion_retention": co1,
            "C-O2_micro_trajectory_invariance": co2,
            "C-O3_low_NDS_low_curvature": co3,
            "C-O4_info_volume_no_fold": co4,
            "C-O5_control_nullspace": co5,
            "coherent_drift_norm_proxy": row.get("E2_gradient_drift_snr", ""),
            "diffusion_variance_proxy": row.get("target_reservoir_projection", ""),
            "micro_trajectory_agreement_proxy": min_gap,
            "control_null_residual_fraction_proxy": co5,
            "uses_future_for_direction": 0,
            "future_source_used_as_audit_label_only": 1,
        }
        out.append(item)
    return out


def _heldout_precision(scores: list[float], labels: list[int], groups: list[str], k: int) -> float | str:
    vals: list[float] = []
    for group in sorted(set(groups)):
        idx = [i for i, g in enumerate(groups) if g == group and math.isfinite(scores[i])]
        if not idx:
            continue
        order = sorted(idx, key=lambda i: scores[i], reverse=True)[: min(k, len(idx))]
        if order:
            vals.append(sum(labels[i] for i in order) / float(len(order)))
    return min(vals) if vals else ""


def _summaries(matrix: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    labels = [int_flag(r.get("official_early_chain_h100_h400_h800_positive")) for r in matrix]
    h3200 = [int_flag(r.get("h3200_positive")) for r in matrix]
    retained = [int_flag(r.get("retained_h800_h3200_positive")) for r in matrix]
    controls = [int_flag(r.get("is_control")) for r in matrix]
    groups = [f"{r.get('dataset','')}:{r.get('seed','')}" for r in matrix]
    out: list[dict[str, Any]] = []
    for score_name, desc in SCORE_DESCRIPTIONS.items():
        scores = [_f(r.get(score_name)) for r in matrix]
        p20, recall20, control_frac, order = _precision(scores, labels, controls, top_k)
        heldout = _heldout_precision(scores, labels, groups, top_k)
        top = [matrix[i] for i in order]
        random_gap = sum(_f(r.get("micro_trajectory_agreement_proxy"), 0.0) for r in top) / len(top) if top else ""
        auc_early = _auc(scores, labels)
        blocker = []
        if sum(labels) == 0:
            blocker.append("no_positive_official_early_chain_label_rows")
        if _f(auc_early, -1.0) < 0.75:
            blocker.append("auc_official_early_chain_gate")
        if _f(p20, -1.0) < 0.50:
            blocker.append("precision_top20_gate")
        if _f(control_frac, 1.0) > 0.10:
            blocker.append("control_equivalent_fraction_gate")
        if _f(heldout, -1.0) < 0.40:
            blocker.append("heldout_precision_gate")
        if _f(random_gap, -999.0) <= 0.0:
            blocker.append("random_sign_corrupt_gap_gate")
        pass_flag = int(not blocker)
        out.append(
            {
                "observer_family": score_name,
                "description": desc,
                "rows": len(matrix),
                "positive_official_early_chain_rows": sum(labels),
                "positive_h3200_rows": sum(h3200),
                "positive_retained_h800_h3200_rows": sum(retained),
                "AUC_predict_official_early_chain": auc_early,
                "AUC_predict_h3200_positive": _auc(scores, h3200),
                "AUC_predict_retained_h800_h3200_positive": _auc(scores, retained),
                "precision_at_top20": p20,
                "recall_at_top20": recall20,
                "heldout_dataset_seed_precision_at_top20": heldout,
                "control_equivalent_fraction": control_frac,
                "top20_random_sign_corrupt_gap_proxy_mean": random_gap,
                "direction_allowed": 1,
                "C0_retained_source_observer_pass": pass_flag,
                "blocker": ";".join(dict.fromkeys(blocker)),
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    joined = _join(source_dir)
    matrix = _build_matrix(joined)
    summary = _summaries(matrix, int(args.top_k))
    passing = [r for r in summary if int_flag(r.get("C0_retained_source_observer_pass"))]
    best_family = max(summary, key=lambda r: _f(r.get("AUC_predict_retained_h800_h3200_positive"), -1.0)).get("observer_family", "") if summary else ""
    route = {
        "source_dir": str(source_dir),
        "observer_rows": len(matrix),
        "observer_family_rows": len(summary),
        "C0_retained_source_observer_pass_rows": len(passing),
        "best_observer_family": best_family,
        "official_early_chain_positive_rows": sum(int_flag(r.get("official_early_chain_h100_h400_h800_positive")) for r in matrix),
        "retained_h800_h3200_positive_rows": sum(int_flag(r.get("retained_h800_h3200_positive")) for r in matrix),
        "route": "C0-RetainedSourceObserverOpened" if passing else "RetainedSourceObserverLocalNoGo_v22.08",
        "blocker": "" if passing else "all_C-O_observers_failed_or_no_positive_official_early_chain_label_rows",
        "next_codex_action": "enter Line D selected fresh C2/C3" if passing else "stop same-family C-O observer scale/cap/floor tuning; require a genuinely new retained-source observability principle",
    }
    write_rows(out_dir / "v22_08_retained_source_observer_matrix.csv", matrix)
    write_rows(out_dir / "v22_08_retained_source_observer_summary.csv", summary)
    top_rows: list[dict[str, Any]] = []
    if best_family:
        top_rows = sorted(matrix, key=lambda r: _f(r.get(best_family), -999.0), reverse=True)[: int(args.top_k)]
    write_rows(out_dir / "v22_08_retained_source_observer_top_candidates.csv", top_rows)
    write_json(out_dir / "v22_08_retained_source_observer_route.json", route)
    simple_svg(out_dir / "figures/v22_08_observer_auc_precision_dashboard.svg", "v22.08 observer AUC/precision", summary, "AUC_predict_retained_h800_h3200_positive")
    simple_svg(out_dir / "figures/v22_08_drift_diffusion_vs_h3200.svg", "v22.08 drift diffusion vs h3200", matrix, "C-O1_drift_diffusion_retention")
    simple_svg(out_dir / "figures/v22_08_micro_trajectory_invariance.svg", "v22.08 micro trajectory invariance", matrix, "C-O2_micro_trajectory_invariance")
    simple_svg(out_dir / "figures/v22_08_NDS_vs_source_retention.svg", "v22.08 NDS vs source retention", matrix, "C-O3_low_NDS_low_curvature")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_retained_source_observer.py --source-dir {source_dir} --top-k {int(args.top_k)} --out-dir {out_dir}",
        status="completed",
        note=f"families={len(summary)} pass={len(passing)} official_early_positive={route['official_early_chain_positive_rows']} route={route['route']}",
    )


if __name__ == "__main__":
    main()

