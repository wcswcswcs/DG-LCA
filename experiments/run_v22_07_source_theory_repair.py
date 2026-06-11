#!/usr/bin/env python3
"""v22.07 source-observability theory repair audit.

After target-rescale, optimizer-integration, and semantic-target repairs failed
C3, this audit changes the observer target from "any h800 positive" to
"retained h800+h3200 positive".  Scores are built only from train-stream /
no-commit matrices.  Future source columns are used strictly as audit labels.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_07_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_rows, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--top-k", type=int, default=12)
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


def _spearman(scores: list[float], values: list[float]) -> float | str:
    valid = [(s, v) for s, v in zip(scores, values) if math.isfinite(s) and math.isfinite(v)]
    if len(valid) < 3:
        return ""
    sx = [s for s, _v in valid]
    sy = [v for _s, v in valid]
    rx = _rank(sx)
    ry = _rank(sy)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return ""
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(vx * vy)


def _precision(scores: list[float], labels: list[int], controls: list[int], k: int = 20) -> tuple[float | str, float | str, float | str]:
    valid = [(i, s) for i, s in enumerate(scores) if math.isfinite(s)]
    if not valid:
        return "", "", ""
    order = [i for i, _s in sorted(valid, key=lambda item: item[1], reverse=True)[: min(k, len(valid))]]
    if not order:
        return "", "", ""
    hits = sum(labels[i] for i in order)
    total_pos = sum(labels)
    control_fraction = sum(controls[i] for i in order) / float(len(order))
    return hits / float(len(order)), hits / float(total_pos) if total_pos else "", control_fraction


def _zscore(values: list[float], groups: list[tuple[str, str]]) -> list[float]:
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    for key in sorted(set(groups)):
        vals = [v for v, group in zip(values, groups) if group == key and math.isfinite(v)]
        if not vals:
            stats[key] = (0.0, 1.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        stats[key] = (mean, math.sqrt(var) if var > 1.0e-12 else 1.0)
    out = []
    for v, group in zip(values, groups):
        mean, std = stats[group]
        out.append((v - mean) / std if math.isfinite(v) else 0.0)
    return out


def _join(out_dir: Path) -> list[dict[str, Any]]:
    c0_rows = read_rows(out_dir / "v22_07_c0_no_commit_source_estimator_matrix.csv")
    c1_by_order = {str(r.get("job_order", "")): r for r in read_rows(out_dir / "v22_07_c1_target_contrast_matrix.csv")}
    c2_by_order = {str(r.get("job_order", "")): r for r in read_rows(out_dir / "v22_07_c2_metric_solver_matrix.csv")}
    rows: list[dict[str, Any]] = []
    for row in c0_rows:
        if str(row.get("execution_status")) != "measured":
            continue
        merged = dict(row)
        c1 = c1_by_order.get(str(row.get("job_order", "")), {})
        c2 = c2_by_order.get(str(row.get("job_order", "")), {})
        for key in [
            "target_norm",
            "target_signal_projection",
            "target_reservoir_projection",
            "B1_gain",
            "B2_transfer_gain",
            "B3_safety_gain",
            "random_target_gap",
            "sign_flip_gap",
            "corrupt_gap",
        ]:
            merged[key] = c1.get(key, "")
        for key in [
            "projection_residual_Gf",
            "ActuationR2",
            "ActuationCosine",
            "solve_time_ms",
            "condition_estimate",
            "parameter_update_cosine",
        ]:
            merged[key] = c2.get(key, "")
        rows.append(merged)
    return rows


def _labels(row: dict[str, Any]) -> dict[str, int]:
    source = {h: _f(row.get(f"future_audit_source_h{h}"), -999.0) for h in [100, 400, 800, 1600, 3200]}
    return {
        "h800_positive": int(source[800] >= 0.005),
        "h3200_positive": int(source[3200] >= 0.005),
        "retained_h800_h3200_positive": int(source[800] >= 0.005 and source[3200] >= 0.005),
        "mid_retained_h400_h800_h3200_positive": int(source[400] >= 0.005 and source[800] >= 0.005 and source[3200] >= 0.005),
        "official_early_chain_h100_h400_h800_positive": int(source[100] >= 0.005 and source[400] >= 0.005 and source[800] >= 0.005),
    }


def _build_scores(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    groups = [(str(r.get("dataset", "")), str(r.get("seed", ""))) for r in rows]
    keys = [
        "precommit_score",
        "make_update_wall_ms",
        "E1_train_split_B2_transfer",
        "E2_gradient_drift_snr",
        "E3_signal_reservoir_projection",
        "E4_class_balanced_density",
        "E5_hidden_readout_decomposition",
        "E6_optimizer_conflict_score",
        "E7_low_NDS_score",
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
    descriptions = {
        "Q0_precommit_signal_z": "dataset-z precommit score from no-commit E1/E3/low-NDS components",
        "Q1_transfer_signal_control_gap": "E1+B2+B3 plus random/sign/corrupt gap and signal-reservoir separation",
        "Q2_solver_feasible_retention": "Q1 plus projection residual, ActuationR2, and condition estimate; train-only solver feasibility",
        "Q3_conflict_safe_signal": "Q1 penalized by optimizer conflict and reservoir projection",
        "Q4_runtime_audit_only": "Q2 plus make_update_wall_ms; audit-only because runtime is not a source-direction signal",
        "Q5_moderate_solver_feasible_retention": "non-monotonic retained-source observer: moderate Q2 solver feasibility, not maximal transient actuation",
    }
    out = []
    for i, row in enumerate(rows):
        min_gap = min(z["random_target_gap"][i], z["sign_flip_gap"][i], z["corrupt_gap"][i])
        q0 = z["precommit_score"][i]
        q1 = (
            z["E1_train_split_B2_transfer"][i]
            + 0.60 * z["B2_transfer_gain"][i]
            + 0.30 * z["B3_safety_gain"][i]
            + 0.50 * min_gap
            + 0.35 * (z["target_signal_projection"][i] - z["target_reservoir_projection"][i])
            + 0.25 * z["E3_signal_reservoir_projection"][i]
        )
        q2 = q1 - 0.45 * z["projection_residual_Gf"][i] + 0.25 * z["ActuationR2"][i] - 0.15 * z["condition_estimate"][i]
        q3 = q1 - 0.35 * z["E6_optimizer_conflict_score"][i] - 0.25 * z["target_reservoir_projection"][i]
        q4 = q2 + 0.35 * z["make_update_wall_ms"][i]
        q5 = -abs(q2 - 2.0) + 0.20 * q1
        item = dict(row)
        item.update(_labels(row))
        item["Q0_precommit_signal_z"] = q0
        item["Q1_transfer_signal_control_gap"] = q1
        item["Q2_solver_feasible_retention"] = q2
        item["Q3_conflict_safe_signal"] = q3
        item["Q4_runtime_audit_only"] = q4
        item["Q5_moderate_solver_feasible_retention"] = q5
        item["Q4_runtime_direction_allowed"] = 0
        out.append(item)
    return out, descriptions


def _summaries(rows: list[dict[str, Any]], descriptions: dict[str, str]) -> list[dict[str, Any]]:
    controls = [int_flag(r.get("is_control")) for r in rows]
    labels = {
        "h800_positive": [int_flag(r.get("h800_positive")) for r in rows],
        "h3200_positive": [int_flag(r.get("h3200_positive")) for r in rows],
        "retained_h800_h3200_positive": [int_flag(r.get("retained_h800_h3200_positive")) for r in rows],
        "mid_retained_h400_h800_h3200_positive": [int_flag(r.get("mid_retained_h400_h800_h3200_positive")) for r in rows],
        "official_early_chain_h100_h400_h800_positive": [int_flag(r.get("official_early_chain_h100_h400_h800_positive")) for r in rows],
    }
    h3200_values = [_f(r.get("future_audit_source_h3200")) for r in rows]
    h800_values = [_f(r.get("future_audit_source_h800")) for r in rows]
    summary: list[dict[str, Any]] = []
    for score, desc in descriptions.items():
        values = [_f(r.get(score)) for r in rows]
        for label_name, labs in labels.items():
            p20, r20, control_fraction = _precision(values, labs, controls, 20)
            auc = _auc(values, labs)
            direction_allowed = 0 if score == "Q4_runtime_audit_only" else 1
            source_theory_pass = int(
                direction_allowed
                and label_name == "retained_h800_h3200_positive"
                and _f(auc, -1.0) >= 0.70
                and _f(p20, -1.0) >= 0.35
                and _f(control_fraction, 1.0) <= 0.50
            )
            limited_smoke_selection = int(
                direction_allowed
                and label_name == "retained_h800_h3200_positive"
                and _f(auc, -1.0) >= 0.75
                and _f(p20, -1.0) >= 0.25
                and _f(control_fraction, 1.0) <= 0.50
            )
            summary.append(
                {
                    "score": score,
                    "description": desc,
                    "label": label_name,
                    "positive_rows": sum(labs),
                    "rows": len(rows),
                    "AUC": auc,
                    "precision_at_top20": p20,
                    "recall_at_top20": r20,
                    "control_equivalent_fraction": control_fraction,
                    "Spearman_vs_source_h800": _spearman(values, h800_values),
                    "Spearman_vs_source_h3200": _spearman(values, h3200_values),
                    "direction_allowed": direction_allowed,
                    "source_theory_pass": source_theory_pass,
                    "limited_smoke_selection": limited_smoke_selection,
                    "blocker": "" if source_theory_pass else ("no_positive_label_rows" if not sum(labs) else "auc_or_precision_or_control_gate_failed"),
                }
            )
    return summary


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    joined = _join(out_dir)
    matrix, descriptions = _build_scores(joined)
    summary = _summaries(matrix, descriptions)
    legal_retained = [
        r
        for r in summary
        if str(r.get("label")) == "retained_h800_h3200_positive"
        and int_flag(r.get("direction_allowed"))
    ]
    best = sorted(
        legal_retained,
        key=lambda r: (int_flag(r.get("source_theory_pass")), _f(r.get("precision_at_top20"), -1.0), _f(r.get("AUC"), -1.0)),
        reverse=True,
    )[0] if legal_retained else {}
    smoke = sorted(
        [r for r in legal_retained if int_flag(r.get("limited_smoke_selection"))],
        key=lambda r: (_f(r.get("precision_at_top20"), -1.0), _f(r.get("AUC"), -1.0)),
        reverse=True,
    )
    smoke_best = smoke[0] if smoke else {}
    best_score = str((smoke_best or best).get("score", "Q2_solver_feasible_retention"))
    top = sorted(
        [r for r in matrix if not int_flag(r.get("is_control"))],
        key=lambda r: _f(r.get(best_score), -999.0),
        reverse=True,
    )[: int(args.top_k)]
    top_rows = []
    for rank, row in enumerate(top, start=1):
        top_rows.append(
            {
                "rank": rank,
                "selection_score": best_score,
                "selection_score_value": row.get(best_score, ""),
                "job_order": row.get("job_order", ""),
                "v21_id": row.get("v21_id", ""),
                "mechanism": row.get("mechanism", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "train_seed": row.get("train_seed", ""),
                "future_audit_source_h100": row.get("future_audit_source_h100", ""),
                "future_audit_source_h400": row.get("future_audit_source_h400", ""),
                "future_audit_source_h800": row.get("future_audit_source_h800", ""),
                "future_audit_source_h3200": row.get("future_audit_source_h3200", ""),
                "retained_h800_h3200_positive": row.get("retained_h800_h3200_positive", ""),
                "official_early_chain_h100_h400_h800_positive": row.get("official_early_chain_h100_h400_h800_positive", ""),
            }
        )
    label_counts = {
        "h800_positive": sum(int_flag(r.get("h800_positive")) for r in matrix),
        "h3200_positive": sum(int_flag(r.get("h3200_positive")) for r in matrix),
        "retained_h800_h3200_positive": sum(int_flag(r.get("retained_h800_h3200_positive")) for r in matrix),
        "mid_retained_h400_h800_h3200_positive": sum(int_flag(r.get("mid_retained_h400_h800_h3200_positive")) for r in matrix),
        "official_early_chain_h100_h400_h800_positive": sum(int_flag(r.get("official_early_chain_h100_h400_h800_positive")) for r in matrix),
    }
    route = {
        "rows": len(matrix),
        **label_counts,
        "best_legal_retained_score": best_score,
        "strict_best_legal_retained_score": best.get("score", ""),
        "strict_best_legal_retained_AUC": best.get("AUC", ""),
        "strict_best_legal_retained_precision_at_top20": best.get("precision_at_top20", ""),
        "strict_best_legal_retained_control_equiv": best.get("control_equivalent_fraction", ""),
        "smoke_best_retained_score": smoke_best.get("score", ""),
        "smoke_best_retained_AUC": smoke_best.get("AUC", ""),
        "smoke_best_retained_precision_at_top20": smoke_best.get("precision_at_top20", ""),
        "smoke_best_retained_control_equiv": smoke_best.get("control_equivalent_fraction", ""),
        "source_theory_pass": int_flag(best.get("source_theory_pass")),
        "limited_smoke_selection": int_flag(smoke_best.get("limited_smoke_selection")),
        "top_candidates": len(top_rows),
        "route": (
            "SourceTheoryRetainedObserverOpened"
            if int_flag(best.get("source_theory_pass"))
            else ("SourceTheoryModerateRetainedObserverSmokeOnly" if int_flag(smoke_best.get("limited_smoke_selection")) else "SourceTheoryRetainedObserverBlocked")
        ),
        "promotion_allowed": 0,
        "future_source_used_for_direction": 0,
        "blocker": "no_official_early_chain_positive_rows;strict_retained_observer_gate_failed" if not int_flag(best.get("source_theory_pass")) else "official_early_chain_absent_C3_smoke_required",
    }
    write_rows(out_dir / "v22_07_source_theory_repair_matrix.csv", matrix)
    write_rows(out_dir / "v22_07_source_theory_repair_summary.csv", summary)
    write_rows(out_dir / "v22_07_source_theory_top_candidates.csv", top_rows)
    write_json(out_dir / "v22_07_source_theory_repair_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_source_theory_repair.py --out-dir {out_dir} --top-k {args.top_k}",
        status="completed",
        note=(
            f"route={route['route']} retained_pos={route['retained_h800_h3200_positive']} "
            f"early_chain_pos={route['official_early_chain_h100_h400_h800_positive']} "
            f"best={best_score} smoke_auc={route['smoke_best_retained_AUC']} smoke_p20={route['smoke_best_retained_precision_at_top20']}"
        ),
    )


if __name__ == "__main__":
    main()
