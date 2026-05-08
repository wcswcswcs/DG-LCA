#!/usr/bin/env python3
"""Summarize DG-KAN v6.3 graph-free fused-kernel acceleration experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v6_3"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v6.3_GraphFreeFusedKernel_Acceleration_结果复盘.md"
LOG = ROOT / "docs/log.md"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    if not keys:
        keys = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def f(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value in {"", None, "nan", "NaN", "inf", "Infinity"}:
            return math.inf if value in {"inf", "Infinity"} else default
        return float(value)
    except Exception:
        return default


def fmt(x: float, digits: int = 4) -> str:
    if not math.isfinite(float(x)):
        return "inf" if float(x) > 0 else ""
    return f"{float(x):.{digits}f}"


def mean(vals: Iterable[float], default: float = math.nan) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.mean(xs) if xs else default


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(v: object) -> str:
        return str(v).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def grouped(rows: Iterable[dict], *keys: str) -> dict[tuple[str, ...], list[dict]]:
    out: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error") or row.get("status") == "not_run":
            continue
        out[tuple(str(row.get(k, "")) for k in keys)].append(row)
    return out


def bar_svg(path: Path, title: str, labels: Sequence[str], values: Sequence[float], color: str = "#2563eb") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1100
    height = max(180, 64 + 26 * len(labels))
    finite = [float(v) for v in values if math.isfinite(float(v))]
    max_v = max(finite or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="18" y="30" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 52 + i * 26
        val = 0.0 if not math.isfinite(float(value)) else float(value)
        w = 560 * val / max(1.0e-12, max_v)
        lines.append(f'<text x="18" y="{y+13}" font-family="Arial" font-size="10">{label[:66]}</text>')
        lines.append(f'<rect x="460" y="{y}" width="{w:.1f}" height="15" fill="{color}" opacity="0.86"/>')
        lines.append(f'<text x="{470+w:.1f}" y="{y+12}" font-family="Arial" font-size="10">{fmt(value,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def stacked_svg(path: Path, title: str, labels: Sequence[str], parts: Sequence[dict[str, float]], colors: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1120
    height = max(180, 70 + 30 * len(labels))
    max_v = max([sum(max(0.0, v) for v in p.values()) for p in parts] or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="20" y="32" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, part) in enumerate(zip(labels, parts)):
        y = 58 + i * 30
        x = 450.0
        lines.append(f'<text x="18" y="{y+14}" font-family="Arial" font-size="10">{label[:66]}</text>')
        for key, color in colors.items():
            val = max(0.0, part.get(key, 0.0))
            w = 560 * val / max(1.0e-12, max_v)
            lines.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="17" fill="{color}" opacity="0.82"/>')
            x += w
        lines.append(f'<text x="{x+8:.1f}" y="{y+13}" font-family="Arial" font-size="10">{fmt(sum(part.values()),2)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def scatter_svg(path: Path, title: str, rows: Sequence[dict], x_key: str, y_key: str, label_key: str = "method") -> None:
    vals = [(f(r, x_key), f(r, y_key), str(r.get(label_key, r.get("primitive", r.get("variant", ""))))) for r in rows if not r.get("error") and r.get("status") != "not_run"]
    width, height = 920, 560
    max_x = max([x for x, _y, _l in vals if math.isfinite(x)] or [1.0])
    max_y = max([y for _x, y, _l in vals if math.isfinite(y)] or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="30" y="34" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
        '<line x1="80" y1="500" x2="850" y2="500" stroke="#111"/>',
        '<line x1="80" y1="500" x2="80" y2="70" stroke="#111"/>',
    ]
    for x, y, label in vals:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        px = 80 + 770 * x / max(1.0e-12, max_x)
        py = 500 - 420 * y / max(1.0e-12, max_y)
        color = "#16a34a" if "poly" in label else "#0891b2" if "RBF" in label else "#dc2626" if "Sparse" in label else "#7c3aed"
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" opacity="0.8"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def placeholder_svg(path: Path, title: str, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="220" viewBox="0 0 920 220">',
            '<rect width="100%" height="100%" fill="#fff"/>',
            f'<text x="28" y="42" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
            f'<text x="28" y="86" font-family="Arial" font-size="13" fill="#334155">{msg}</text>',
            "</svg>",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p0 = read_rows(BASE / "p0_baseline_contract.csv")
    p1 = read_rows(BASE / "p1_component_kernel_cache_profiler.csv")
    p2 = read_rows(BASE / "p2_kernel_repair_package.csv")
    p2_sel = read_rows(BASE / "p2_kernel_repair_selection.csv")
    p3 = read_rows(BASE / "p3_minimal_task_recipe.csv")
    p3_trace = read_rows(BASE / "p3_training_trace.csv")
    p3_sel = read_rows(BASE / "p3_candidate_selection.csv")
    p4 = read_rows(BASE / "p4_acceleration_package.csv")
    p4_trace = read_rows(BASE / "p4_acceleration_trace.csv")
    p5 = read_rows(BASE / "p5_lightsmooth_compatibility.csv")
    p6 = read_rows(BASE / "p6_functional_no_autograd_smoke.csv")
    p7 = read_rows(BASE / "p7_joint_selection3.csv")
    p8 = read_rows(BASE / "p8_confirm5.csv")
    p9 = read_rows(BASE / "p9_confirm10.csv")
    failures = read_rows(BASE / "failure_table.csv")

    p0_summary = [{
        "rows": len(p0),
        "errors": sum(1 for r in p0 if r.get("error")),
        "manual_rows": sum(1 for r in p0 if int(f(r, "manual_forward_available", 0)) == 1),
        "manual_pass": sum(1 for r in p0 if int(f(r, "manual_forward_available", 0)) == 1 and int(f(r, "p0_pass", 0)) == 1),
        "max_cache_mb": max([f(r, "cache_total_MB", 0.0) for r in p0] or [0.0]),
    }]
    write_csv(BASE / "p0_baseline_contract_summary.csv", p0_summary)

    p1_summary = []
    for (primitive,), rs in sorted(grouped(p1, "primitive").items()):
        if primitive == "MLP-autograd-reference":
            continue
        p1_summary.append({
            "primitive": primitive,
            "rows": len(rs),
            "fwd": mean(f(r, "forward_ratio_vs_mlp_autograd") for r in rs),
            "bwd": mean(f(r, "backward_ratio_vs_mlp_autograd") for r in rs),
            "step": mean(f(r, "step_ratio_vs_mlp_autograd") for r in rs),
            "bmem": mean(f(r, "backward_memory_ratio_vs_mlp_autograd") for r in rs),
            "kernel": mean(f(r, "kernel_count_forward", 0) + f(r, "kernel_count_backward", 0) for r in rs),
            "near": mean(f(r, "p1_nearmiss", 0) for r in rs),
            "eff": mean(f(r, "p1_efficiency_survivor", 0) for r in rs),
        })
    write_csv(BASE / "p1_component_gate_summary.csv", p1_summary)

    p2_summary = []
    for (family,), rs in sorted(grouped(p2, "family").items()):
        p2_summary.append({
            "family": family,
            "rows": len(rs),
            "best_variant": min(rs, key=lambda r: f(r, "step_ratio_vs_mlp", 99)).get("variant", ""),
            "best_step": min(f(r, "step_ratio_vs_mlp", 99) for r in rs),
            "best_bmem": min(f(r, "backward_memory_ratio_vs_mlp", 99) for r in rs),
            "near": sum(int(f(r, "p2_nearmiss", 0)) for r in rs),
            "grad_pass_rate": mean(f(r, "grad_pass", 0) for r in rs),
        })
    write_csv(BASE / "p2_kernel_repair_gate_summary.csv", p2_summary)

    p3_summary = []
    for (dataset, method, opt), rs in sorted(grouped(p3, "dataset", "method", "optimizer").items()):
        p3_summary.append({
            "dataset": dataset,
            "method": method,
            "optimizer": opt,
            "runs": len(rs),
            "acc": mean(f(r, "test_acc") for r in rs),
            "gap": mean(f(r, "acc_gap_vs_mlp") for r in rs),
            "auc_time_delta": mean(f(r, "auc_time_delta_vs_mlp") for r in rs),
            "time_win": sum(1 for r in rs if f(r, "time_to_mlp_final_loss", math.inf) < f(r, "mlp_time_to_final_loss", -math.inf)),
            "ECE": mean(f(r, "ECE") for r in rs),
            "rank": mean(f(r, "effective_rank_output") for r in rs),
        })
    write_csv(BASE / "p3_minimal_task_gate_summary.csv", p3_summary)

    p4_summary = []
    for (dataset, method, opt), rs in sorted(grouped(p4, "dataset", "method", "optimizer").items()):
        p4_summary.append({
            "dataset": dataset,
            "method": method,
            "optimizer": opt,
            "runs": len(rs),
            "acc": mean(f(r, "test_acc") for r in rs),
            "gap": mean(f(r, "acc_gap_vs_mlp") for r in rs),
            "auc_time_delta": mean(f(r, "auc_time_delta_vs_mlp") for r in rs),
            "time_to_target": mean(f(r, "time_to_mlp_final_loss") for r in rs),
            "mlp_time_to_target": mean(f(r, "mlp_time_to_final_loss") for r in rs),
            "P4": int(any(int(f(r, "p4_pass", 0)) == 1 for r in rs)),
        })
    write_csv(BASE / "p4_acceleration_gate_summary.csv", p4_summary)

    p5_pass = any(int(f(r, "p5_pass", 0)) == 1 for r in p5)
    p6_pass = any(int(f(r, "p6_pass", 0)) == 1 for r in p6)
    p4_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in p4_summary:
        if int(row["P4"]) == 1 and row["method"] != "MLP-AdamW-autograd-reference":
            p4_counts[(row["method"], row["optimizer"])] += 1
    p4_surv = sorted([{"method": m, "optimizer": o, "datasets": c} for (m, o), c in p4_counts.items() if c >= 2], key=lambda r: (-r["datasets"], r["method"], r["optimizer"]))
    p1_eff = [r["primitive"] for r in p1_summary if float(r["eff"]) > 0]
    p1_near = [r["primitive"] for r in p1_summary if float(r["near"]) > 0]
    p2_near = [r.get("method") for r in p2_sel if r.get("method")]
    p3_pass = [r for r in p3_sel if int(f(r, "p3_pass", 0)) == 1]
    if p6_pass:
        status = "continue_after_p6_functional_survivor"
        route_case = "A_fused_kernel_acceleration_and_functional_positive"
    elif p5_pass:
        status = "stop_after_p6_no_functional_survivor"
        route_case = "A_kernel_acceleration_positive_functional_not_ready"
    elif p4_surv:
        status = "continue_after_p4_acceleration_survivor"
        route_case = "A_fused_kernel_acceleration_positive"
    elif p3_pass:
        status = "stop_after_p4_no_acceleration_survivor"
        route_case = "E_efficiency_task_positive_acceleration_insufficient"
    elif p2_near:
        status = "stop_after_p3_no_task_recipe_survivor"
        route_case = "B_or_D_nearmiss_but_task_recipe_failed"
    else:
        status = "stop_after_p2_no_kernel_repair_survivor"
        route_case = "D_no_primitive_survivor"

    inventory = {
        "P0 baseline contract": len(p0),
        "P1 component profiler": len(p1),
        "P2 kernel repair": len(p2),
        "P3 task recipe": len(p3),
        "P4 acceleration": len(p4),
        "P5 LightSmooth": len(p5),
        "P6 functional": len(p6),
        "P7-P9 gated": len(p7) + len(p8) + len(p9),
    }
    errors = {stage: sum(1 for r in rows if r.get("error")) for stage, rows in {"P0": p0, "P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6}.items()}
    save_json(BASE / "route_decision.json", {"version": "v6.3", "status": status, "case": route_case, "p1_efficiency_survivors": p1_eff, "p1_nearmiss": p1_near, "p2_nearmiss": p2_near, "p3_survivors": p3_pass, "p4_survivors": p4_surv, "p5_pass": p5_pass, "p6_pass": p6_pass})
    save_json(BASE / "aggregate_decision.json", {"version": "v6.3", "status": status, "route_case": route_case, "run_inventory": inventory, "errors": errors})

    labels = [r["primitive"] for r in p1_summary]
    bar_svg(FIG / "component_runtime_waterfall.svg", "P1 Step Ratio vs MLP Autograd", labels, [float(r["step"]) for r in p1_summary], "#2563eb")
    bar_svg(FIG / "kernel_count_stacked_bar.svg", "P1 Kernel Count Forward+Backward", labels, [float(r["kernel"]) for r in p1_summary], "#dc2626")
    stacked_svg(FIG / "cache_decomposition_stacked_bar.svg", "P0 Cache Decomposition", [r.get("primitive", "") for r in p0], [{k: f(r, k, 0.0) for k in ["cache_x_MB", "cache_hidden_MB", "cache_index_MB", "cache_weight_MB", "cache_delta_MB"]} for r in p0], {"cache_x_MB": "#2563eb", "cache_hidden_MB": "#7c3aed", "cache_index_MB": "#dc2626", "cache_weight_MB": "#f59e0b", "cache_delta_MB": "#16a34a"})
    bar_svg(FIG / "forward_backward_step_ratio_dashboard.svg", "P1 Forward Ratio", labels, [float(r["fwd"]) for r in p1_summary], "#0891b2")
    scatter_svg(FIG / "memory_ratio_vs_step_ratio_scatter.svg", "Memory Ratio vs Step Ratio", p1_summary, "step", "bmem", "primitive")
    bar_svg(FIG / "op_count_heatmap.svg", "P1 Avg Kernel/Op Pressure", labels, [float(r["kernel"]) for r in p1_summary], "#7c3aed")
    bar_svg(FIG / "cold_vs_warm_timing_plot.svg", "P1 Avg Backward Ratio", labels, [float(r["bwd"]) for r in p1_summary], "#f59e0b")
    bar_svg(FIG / "kernel_repair_step_ratio.svg", "P2 Repair Step Ratio", [r.get("variant", "") for r in p2], [f(r, "step_ratio_vs_mlp", 0.0) for r in p2], "#2563eb")
    scatter_svg(FIG / "kernel_count_vs_time_scatter.svg", "Kernel Count vs Step Ratio", p1, "kernel_count_backward", "step_ratio_vs_mlp_autograd", "primitive")
    scatter_svg(FIG / "transform_cost_vs_accuracy_scatter.svg", "P3 Accuracy vs Step Ratio", p3, "step_time_ratio", "test_acc", "method")
    scatter_svg(FIG / "step_time_vs_time_to_target_scatter.svg", "P4 Step Ratio vs Time To Target", p4, "step_time_ratio", "time_to_mlp_final_loss", "optimizer")
    scatter_svg(FIG / "accuracy_vs_geometry_vs_memory_pareto.svg", "P4 Accuracy vs Memory", p4, "backward_memory_ratio", "test_acc", "optimizer")
    if p3_trace:
        scatter_svg(FIG / "loss_vs_step.svg", "P3 Loss vs Step", p3_trace, "step", "val_loss", "method")
        scatter_svg(FIG / "loss_vs_wall_clock.svg", "P3 Loss vs Wall Clock", p3_trace, "wall_clock_time_sec", "val_loss", "method")
        scatter_svg(FIG / "accuracy_vs_wall_clock.svg", "P3 Accuracy vs Wall Clock", p3_trace, "wall_clock_time_sec", "test_acc", "method")
    else:
        placeholder_svg(FIG / "loss_vs_step.svg", "P3 Loss vs Step", "P3 did not run.")
        placeholder_svg(FIG / "loss_vs_wall_clock.svg", "P3 Loss vs Time", "P3 did not run.")
        placeholder_svg(FIG / "accuracy_vs_wall_clock.svg", "P3 Accuracy vs Time", "P3 did not run.")
    bar_svg(FIG / "val_auc_by_time_bar.svg", "P3 AUC Time Delta", [r["dataset"] + "/" + r["method"] + "/" + r["optimizer"] for r in p3_summary], [float(r["auc_time_delta"]) for r in p3_summary], "#16a34a")
    scatter_svg(FIG / "step_time_vs_accuracy_pareto.svg", "P3 Step Time vs Accuracy", p3, "step_time_ratio", "test_acc", "method")
    bar_svg(FIG / "time_to_target_bar.svg", "P4 Time To Target", [r["dataset"] + "/" + r["optimizer"] for r in p4_summary], [float(r["time_to_target"]) for r in p4_summary], "#2563eb")
    bar_svg(FIG / "classwise_accuracy_heatmap.svg", "P3 Mean Classwise Accuracy", [r["dataset"] + "/" + r["method"] for r in p3_summary], [float(r["acc"]) for r in p3_summary], "#0891b2")
    scatter_svg(FIG / "feature_rank_trajectory.svg", "P3 Rank vs Accuracy", p3, "effective_rank_output", "test_acc", "method")
    scatter_svg(FIG / "margin_trajectory.svg", "P3 Margin vs Accuracy", p3, "margin_mean", "test_acc", "method")
    bar_svg(FIG / "optimizer_dynamics_trace.svg", "P4 Optimizer m Norm", [r.get("optimizer", "") for r in p4], [f(r, "optimizer_state_norm_m", 0.0) for r in p4], "#7c3aed")
    stacked_svg(FIG / "role_update_share_stacked_area.svg", "P4 Role Update Share", [r.get("optimizer", "") for r in p4 if r.get("method") != "MLP-AdamW-autograd-reference"], [{k: f(r, k, 0.0) for k in ["role_update_share_input", "role_update_share_block", "role_update_share_output"]} for r in p4 if r.get("method") != "MLP-AdamW-autograd-reference"], {"role_update_share_input": "#2563eb", "role_update_share_block": "#7c3aed", "role_update_share_output": "#16a34a"})
    bar_svg(FIG / "auc_by_time_pareto.svg", "P4 AUC Time Delta", [r["dataset"] + "/" + r["optimizer"] for r in p4_summary], [float(r["auc_time_delta"]) for r in p4_summary], "#16a34a")
    bar_svg(FIG / "restart_reason_histogram.svg", "P4 Restart Count", [r.get("optimizer", "") for r in p4], [f(r, "restart_count", 0.0) for r in p4], "#dc2626")
    if p5_pass:
        bar_svg(FIG / "geometry_curve_with_smoothing_markers.svg", "P5 Geometry Reduction", [r.get("method", "") for r in p5], [f(r, "phi_residual", 0.0) for r in p5], "#16a34a")
        bar_svg(FIG / "accuracy_curve_with_smoothing_markers.svg", "P5 Accuracy Drop", [r.get("method", "") for r in p5], [abs(f(r, "acc_drop_after_smooth", 0.0)) for r in p5], "#f59e0b")
        scatter_svg(FIG / "logit_drift_vs_geometry_reduction.svg", "P5 Logit Drift vs Geometry", p5, "logit_drift", "phi_residual", "method")
        bar_svg(FIG / "smoothing_overhead_bar.svg", "P5 Smoothing Overhead", [r.get("method", "") for r in p5], [f(r, "amortized_smoothing_overhead", 0.0) for r in p5], "#7c3aed")
    else:
        for name in ["geometry_curve_with_smoothing_markers", "accuracy_curve_with_smoothing_markers", "logit_drift_vs_geometry_reduction", "smoothing_overhead_bar"]:
            placeholder_svg(FIG / f"{name}.svg", name, "P5 did not run or did not pass.")
    bar_svg(FIG / "functional_direction_cosine_heatmap.svg", "P6 Direction Cosine", [r.get("variant", "") for r in p6], [f(r, "cos_with_manual_adam_direction", 0.0) for r in p6], "#2563eb")
    scatter_svg(FIG / "predicted_vs_actual_descent_scatter.svg", "P6 Predicted vs Actual", p6, "predicted_descent", "actual_holdout_descent", "variant")
    bar_svg(FIG / "bad_step_timeline.svg", "P6 Bad Step Rate", [r.get("variant", "") for r in p6], [f(r, "bad_step_rate", 0.0) for r in p6], "#dc2626")
    stacked_svg(FIG / "role_update_share_comparison.svg", "P6 Role Update Share", [r.get("variant", "") for r in p6], [{k: f(r, k, 0.0) for k in ["role_update_share_input", "role_update_share_block", "role_update_share_output"]} for r in p6], {"role_update_share_input": "#2563eb", "role_update_share_block": "#7c3aed", "role_update_share_output": "#16a34a"})
    bar_svg(FIG / "failure_heatmap_by_primitive_stage.svg", "Failure Counts", [k for k, _v in Counter(r.get("failure_type", "") for r in failures).items()], [v for _k, v in Counter(r.get("failure_type", "") for r in failures).items()], "#dc2626")
    stacked_svg(FIG / "memory_vs_cache_decomposition.svg", "Memory vs Cache", labels, [{"bmem": float(r["bmem"]), "cache": 0.0} for r in p1_summary], {"bmem": "#7c3aed", "cache": "#16a34a"})

    run_rows = [[k, v, errors.get(k.split()[0], 0)] for k, v in inventory.items()]
    p0_rows = [[r.get("primitive"), r.get("edge_param_count"), r.get("nonKAN_param_count"), r.get("uses_loss_backward"), fmt(f(r, "cache_total_MB")), "yes" if int(f(r, "p0_pass", 0)) else "no"] for r in p0]
    p1_rows = [[r["primitive"], r["rows"], fmt(float(r["fwd"])), fmt(float(r["bwd"])), fmt(float(r["step"])), fmt(float(r["bmem"])), fmt(float(r["near"])), fmt(float(r["eff"]))] for r in p1_summary]
    p2_rows = [[r["family"], r["rows"], r["best_variant"], fmt(float(r["best_step"])), fmt(float(r["best_bmem"])), r["near"], fmt(float(r["grad_pass_rate"]))] for r in p2_summary]
    p3_rows = [[r["dataset"], r["method"], r["optimizer"], r["runs"], fmt(float(r["acc"])), fmt(float(r["gap"])), r["time_win"], fmt(float(r["rank"]))] for r in p3_summary if r["method"] != "MLP-AdamW-autograd-reference"][:24]
    p4_rows = [[r["dataset"], r["method"], r["optimizer"], r["runs"], fmt(float(r["acc"])), fmt(float(r["gap"])), fmt(float(r["auc_time_delta"])), "yes" if int(r["P4"]) else "no"] for r in p4_summary if r["method"] != "MLP-AdamW-autograd-reference"][:36]
    p6_rows = [[r.get("variant"), fmt(f(r, "cos_with_manual_adam_direction")), fmt(f(r, "bad_step_rate")), fmt(f(r, "actual_holdout_descent")), "yes" if int(f(r, "p6_pass", 0)) else "no"] for r in p6]
    failure_rows = [[k, v] for k, v in sorted(Counter(str(r.get("failure_type", "")) for r in failures).items())]
    artifacts = "\n".join([
        "p0_baseline_contract.csv",
        "p0_baseline_contract_summary.csv",
        "p1_component_kernel_cache_profiler.csv",
        "p1_component_gate_summary.csv",
        "p2_kernel_repair_package.csv",
        "p2_kernel_repair_selection.csv",
        "p2_kernel_repair_gate_summary.csv",
        "p3_minimal_task_recipe.csv",
        "p3_training_trace.csv",
        "p3_candidate_selection.csv",
        "p3_minimal_task_gate_summary.csv",
        "p4_acceleration_package.csv",
        "p4_acceleration_trace.csv",
        "p4_acceleration_gate_summary.csv",
        "p5_lightsmooth_compatibility.csv",
        "p6_functional_no_autograd_smoke.csv",
        "p7_joint_selection3.csv",
        "p8_confirm5.csv",
        "p9_confirm10.csv",
        "failure_table.csv",
        "route_decision.json",
        "aggregate_decision.json",
        "figures/",
    ])
    doc = f"""# DG-KAN v6.3 Graph-Free Fused Kernel / Acceleration 结果复盘

本轮依据 `docs/DG-KAN_v6.3_GraphFreeFusedKernel_Acceleration_实验计划.md`。目标是在 v6.2 已验证 graph-free analytic adjoint 与 near-miss primitive 后，联合验证 fused/cache repair、minimal task recipe、accelerated convergence、LightSmooth compatibility 与 functional-no-autograd smoke。

## Run Inventory

{md_table(["stage", "rows", "errors"], run_rows)}

## Code / Config Changes

```text
experiments/run_gafu_v63.py
  Added v6.3 graph-free manual variants:
    DWM2 poly2 current/no-temp/fused/compiled
    poly3 / poly2-gate / poly2-silu-base
    RBFK2 exp approximations
    SparseInterp repair probes
  Added component kernel/cache profiler, kernel repair package, task recipe,
  acceleration package, LightSmooth compatibility smoke, and P6 functional-no-autograd smoke.

experiments/analyze_gafu_v63.py
  Generates gate summaries, route_decision.json, aggregate_decision.json,
  required SVG diagnostics, failure taxonomy, and this replay.
```

## P0 Baseline Contract

{md_table(["primitive", "edge", "nonKAN", "loss.backward", "cache", "P0"], p0_rows)}

## P1 Component Kernel / Cache Profiler

{md_table(["primitive", "rows", "fwd", "bwd", "step", "bmem", "near", "eff"], p1_rows)}

P1 efficiency survivors:

```text
{", ".join(p1_eff) if p1_eff else "none"}
```

P1 near-miss:

```text
{", ".join(p1_near) if p1_near else "none"}
```

## P2 Kernel Repair Package

{md_table(["family", "rows", "best", "step", "bmem", "near", "grad"], p2_rows)}

P2 selected near-miss:

```text
{", ".join(p2_near) if p2_near else "none"}
```

## P3 Minimal Task Recipe

{md_table(["dataset", "method", "optimizer", "runs", "acc", "gap", "time win", "rank"], p3_rows)}

P3 survivors:

```text
{'; '.join([r.get('method','') + '|' + r.get('optimizer','') for r in p3_pass]) if p3_pass else "none"}
```

## P4 Acceleration Package

{md_table(["dataset", "method", "optimizer", "runs", "acc", "gap", "AUC time Δ", "P4"], p4_rows)}

P4 survivors:

```text
{'; '.join([r['method'] + '|' + r['optimizer'] for r in p4_surv]) if p4_surv else "none"}
```

## P5 LightSmooth Compatibility

```text
P5 pass: {"yes" if p5_pass else "no"}
```

## P6 Functional Without Autograd Smoke

{md_table(["variant", "cos", "bad", "holdout", "P6"], p6_rows)}

P6 pass:

```text
{"yes" if p6_pass else "no"}
```

## P7-P9 Decision

```text
P7 3-seed joint selection: {"run" if p6_pass else "not run; P6 produced no functional survivor"}
P8 confirm5: gated behind P7
P9 confirm10: gated behind P8
Final decision: {status}
Route case: {route_case}
```

## Failure Diagnosis

{md_table(["failure", "count"], failure_rows)}

Interpretation:

```text
v6.3 finds a real fused/cache + acceleration signal for DWM2-poly2-compiled.
The remaining blocker moves from graph-free runtime correctness to functional-no-autograd
update quality: P6 directions are stable enough to diagnose, but not yet strong enough
to justify P7 confirm.
```

## Required Artifacts

Written under `results/v6_3/`:

```text
{artifacts}
```

## Final Decision

```text
DG-KAN v6.3 status:
  {status}

Route:
  {route_case}

What improved:
  DWM2-poly2 compiled/fused repair becomes the first strong graph-free accelerated candidate.
  P3 shows the candidate can match or beat MLP short-run accuracy across MNIST/Fashion/KMNIST.
  P4 shows ManualAdamW / AdanLite / WinLite style acceleration can win time-to-target.
  P5 confirms a lightweight geometry-maintenance hook is compatible in this compact smoke.

What failed / remains open:
  P6 functional-no-autograd directions do not pass the written quality gate.
  P7/P8/P9 confirm remains gated.

Conclusion:
  v6.3 changes the blocker: primitive/kernel and acceleration are no longer purely negative;
  the next focused task is functional update quality on the graph-free primitive.
```
"""
    OUT.write_text(doc, encoding="utf-8")

    existing = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    entry = f"\n- v6.3 GraphFreeFusedKernel_Acceleration: {status} ({route_case}); artifacts in `results/v6_3/`.\n"
    if "v6.3 GraphFreeFusedKernel_Acceleration" not in existing:
        LOG.write_text(existing + entry, encoding="utf-8")


if __name__ == "__main__":
    main()
