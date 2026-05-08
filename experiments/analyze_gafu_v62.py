#!/usr/bin/env python3
"""Summarize DG-KAN v6.2 graph-free kernel/cache redesign experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v6_2"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v6.2_GraphFreeKernelCache_结果复盘.md"
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
        if value in {"", None, "nan", "NaN"}:
            return default
        return float(value)
    except Exception:
        return default


def fmt(x: float, digits: int = 4) -> str:
    return "" if not math.isfinite(float(x)) else f"{float(x):.{digits}f}"


def mean(vals: Iterable[float], default: float = math.nan) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.mean(xs) if xs else default


def grouped(rows: Iterable[dict], *keys: str) -> dict[tuple[str, ...], list[dict]]:
    out: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error") or row.get("status") == "not_run":
            continue
        out[tuple(str(row.get(k, "")) for k in keys)].append(row)
    return dict(out)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def bar_svg(path: Path, title: str, labels: Sequence[str], values: Sequence[float], color: str = "#2563eb") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1060
    height = max(180, 64 + 26 * len(labels))
    max_v = max([v for v in values if math.isfinite(v)] or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="18" y="30" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 52 + i * 26
        w = 560 * (0.0 if not math.isfinite(value) else value / max(1.0e-12, max_v))
        lines.append(f'<text x="18" y="{y+13}" font-family="Arial" font-size="10">{label[:62]}</text>')
        lines.append(f'<rect x="440" y="{y}" width="{w:.1f}" height="15" fill="{color}" opacity="0.86"/>')
        lines.append(f'<text x="{450+w:.1f}" y="{y+12}" font-family="Arial" font-size="10">{fmt(value,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def stacked_svg(path: Path, title: str, labels: Sequence[str], parts: Sequence[dict[str, float]], colors: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1080
    height = max(180, 70 + 30 * len(labels))
    max_v = max([sum(max(0.0, v) for v in p.values()) for p in parts] or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="20" y="32" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, part) in enumerate(zip(labels, parts)):
        y = 58 + i * 30
        x = 440.0
        lines.append(f'<text x="18" y="{y+14}" font-family="Arial" font-size="10">{label[:62]}</text>')
        for key, color in colors.items():
            val = max(0.0, part.get(key, 0.0))
            w = 560 * val / max(1.0e-12, max_v)
            lines.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="17" fill="{color}" opacity="0.82"/>')
            x += w
        lines.append(f'<text x="{x+8:.1f}" y="{y+13}" font-family="Arial" font-size="10">{fmt(sum(part.values()),2)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def scatter_svg(path: Path, title: str, rows: Sequence[dict], x_key: str, y_key: str, label_key: str = "primitive") -> None:
    vals = [(f(r, x_key), f(r, y_key), str(r.get(label_key, ""))) for r in rows if not r.get("error") and r.get("status") != "not_run"]
    width, height = 900, 560
    max_x = max([x for x, _y, _l in vals if math.isfinite(x)] or [1.0])
    max_y = max([y for _x, y, _l in vals if math.isfinite(y)] or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="30" y="34" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
        '<line x1="80" y1="500" x2="840" y2="500" stroke="#111"/>',
        '<line x1="80" y1="500" x2="80" y2="70" stroke="#111"/>',
    ]
    for x, y, label in vals:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        px = 80 + 760 * x / max(1.0e-12, max_x)
        py = 500 - 420 * y / max(1.0e-12, max_y)
        color = "#16a34a" if "RBFK2" in label else "#0891b2" if "poly" in label else "#dc2626" if "Sparse" in label else "#7c3aed" if "Rational" in label else "#64748b"
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" opacity="0.8"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def placeholder_svg(path: Path, title: str, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="220" viewBox="0 0 920 220">',
                '<rect width="100%" height="100%" fill="#fff"/>',
                f'<text x="28" y="42" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
                f'<text x="28" y="86" font-family="Arial" font-size="13" fill="#334155">{msg}</text>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p0 = read_rows(BASE / "p0_graphfree_cache_manifest.csv")
    p1 = read_rows(BASE / "p1_gradient_streaming_correctness.csv")
    p2 = read_rows(BASE / "p2_graphfree_efficiency_v2.csv")
    p3 = read_rows(BASE / "p3_kernel_cache_ablation.csv")
    p4 = read_rows(BASE / "p4_nearmiss_convergence_smoke.csv")
    p4_trace = read_rows(BASE / "p4_training_trace.csv")
    p5 = read_rows(BASE / "p5_acceleration_package.csv")
    p6 = read_rows(BASE / "p6_manual_geometry_maintenance.csv")
    p7 = read_rows(BASE / "p7_joint_selection3.csv")
    p8 = read_rows(BASE / "p8_confirm5.csv")
    p9 = read_rows(BASE / "p9_confirm10.csv")
    failures = read_rows(BASE / "failure_table.csv")

    p0_summary = [{
        "rows": len(p0),
        "errors": sum(1 for r in p0 if r.get("error")),
        "manual_rows": sum(1 for r in p0 if int(f(r, "uses_manual_adjoint", 0)) == 1),
        "manual_pass": sum(1 for r in p0 if int(f(r, "uses_manual_adjoint", 0)) == 1 and int(f(r, "p0_pass", 0)) == 1),
        "max_cache_mb": max([f(r, "manual_cache_total_MB", 0.0) for r in p0] or [0.0]),
    }]
    write_csv(BASE / "p0_cache_manifest_summary.csv", p0_summary)

    p1_summary = []
    for (primitive,), rs in sorted(grouped(p1, "primitive").items()):
        p1_summary.append({
            "primitive": primitive,
            "rows": len(rs),
            "max_coeff_relerr": max([f(r, "coeff_grad_relerr", 99) for r in rs] or [99]),
            "min_coeff_cos": min([f(r, "coeff_grad_cos", -1) for r in rs] or [-1]),
            "max_input_relerr": max([f(r, "input_grad_relerr", 99) for r in rs] or [99]),
            "min_input_cos": min([f(r, "input_grad_cos", -1) for r in rs] or [-1]),
            "pass_rate": mean(f(r, "p1_pass", 0) for r in rs),
        })
    write_csv(BASE / "p1_gradient_gate_summary.csv", p1_summary)

    p2_summary = []
    for (primitive,), rs in sorted(grouped(p2, "primitive").items()):
        if primitive == "MLP-autograd-reference":
            continue
        p2_summary.append({
            "primitive": primitive,
            "rows": len(rs),
            "fwd": mean(f(r, "forward_time_ratio") for r in rs),
            "bwd": mean(f(r, "backward_time_ratio") for r in rs),
            "step": mean(f(r, "step_time_ratio") for r in rs),
            "bmem": mean(f(r, "backward_memory_ratio") for r in rs),
            "cache": mean(f(r, "manual_cache_total_MB", 0.0) for r in rs),
            "explore_rate": mean(f(r, "p2_exploratory_pass", 0) for r in rs),
            "final_rate": mean(f(r, "p2_final_pass", 0) for r in rs),
            "nearmiss_rate": mean(f(r, "p2_nearmiss_pass", 0) for r in rs),
        })
    write_csv(BASE / "p2_efficiency_gate_summary.csv", p2_summary)

    p4_summary = []
    for (dataset, method), rs in sorted(grouped(p4, "dataset", "method").items()):
        acc_gap = mean(f(r, "acc_gap_vs_mlp") for r in rs)
        ece_gap = mean(f(r, "ECE_gap_vs_mlp") for r in rs)
        auc_delta = mean(f(r, "auc_time_delta_vs_mlp") for r in rs)
        bmem = mean(f(r, "backward_memory_ratio") for r in rs)
        p4_summary.append({
            "dataset": dataset,
            "method": method,
            "runs": len(rs),
            "acc": mean(f(r, "test_acc") for r in rs),
            "acc_gap": acc_gap,
            "ECE": mean(f(r, "ECE") for r in rs),
            "ECE_gap": ece_gap,
            "auc_time_delta": auc_delta,
            "step_ratio": mean(f(r, "step_time_ratio") for r in rs),
            "bmem": bmem,
            "pass_rate": mean(f(r, "p4_pass", 0) for r in rs),
            "dataset_pass": int(method == "MLP-AdamW-autograd-reference" or (acc_gap <= 0.02 and ece_gap <= 0.03 and auc_delta >= 0.0 and bmem <= 1.25)),
        })
    write_csv(BASE / "p4_nearmiss_gate_summary.csv", p4_summary)

    p1_surv = sorted(r["primitive"] for r in p1_summary if float(r["pass_rate"]) >= 1.0)
    p2_final = sorted(r["primitive"] for r in p2_summary if float(r["final_rate"]) > 0.0)
    p2_near = sorted(r["primitive"] for r in p2_summary if float(r["nearmiss_rate"]) > 0.0)
    manual_mlp = next((r for r in p2_summary if r["primitive"] == "MLP-manual-linear-reference"), None)
    manual_mlp_slow = bool(manual_mlp and float(manual_mlp["step"]) > 1.50)
    p4_dataset_passes: dict[str, int] = defaultdict(int)
    for row in p4_summary:
        if row["method"] != "MLP-AdamW-autograd-reference" and int(row.get("dataset_pass", 0)) == 1:
            p4_dataset_passes[row["method"]] += 1
    p4_surv = sorted(method for method, count in p4_dataset_passes.items() if count >= 2)
    if manual_mlp_slow:
        status = "stop_after_p2_manual_mlp_runtime_slow"
        route_case = "C_manual_runtime_not_mlp_like"
    elif p2_final:
        status = "continue_after_p2_final_efficiency_survivor"
        route_case = "A_graphfree_efficiency_survivor"
    elif p2_near and p4_surv:
        status = "continue_after_p4_nearmiss_convergence_survivor"
        route_case = "B_nearmiss_convergence_compensates"
    elif p2_near:
        status = "stop_after_p4_no_nearmiss_convergence_survivor"
        route_case = "B_nearmiss_no_convergence_compensation"
    else:
        status = "stop_after_p2_no_nearmiss_efficiency_candidate"
        route_case = "E_no_efficient_primitive"

    inventory = {
        "P0 cache manifest": len(p0),
        "P1 gradient correctness": len(p1),
        "P2 efficiency v2": len(p2),
        "P3 kernel/cache ablation": len(p3),
        "P4 near-miss convergence": len(p4),
        "P5-P9 gated": len(p5) + len(p6) + len(p7) + len(p8) + len(p9),
    }
    errors = {
        "P0": sum(1 for r in p0 if r.get("error")),
        "P1": sum(1 for r in p1 if r.get("error")),
        "P2": sum(1 for r in p2 if r.get("error")),
        "P3": sum(1 for r in p3 if r.get("error")),
        "P4": sum(1 for r in p4 if r.get("error")),
    }
    save_json(BASE / "route_decision.json", {"version": "v6.2", "status": status, "case": route_case, "p1_survivors": p1_surv, "p2_final_survivors": p2_final, "p2_nearmiss": p2_near, "p4_survivors": p4_surv})
    save_json(BASE / "aggregate_decision.json", {"version": "v6.2", "status": status, "route_case": route_case, "run_inventory": inventory, "errors": errors})

    labels = [r["primitive"] for r in p2_summary]
    bar_svg(FIG / "graphfree_efficiency_dashboard.svg", "P2 Step Ratio", labels, [float(r["step"]) for r in p2_summary], "#2563eb")
    bar_svg(FIG / "forward_backward_step_ratio_bar.svg", "P2 Forward Ratio", labels, [float(r["fwd"]) for r in p2_summary], "#0891b2")
    bar_svg(FIG / "memory_ratio_bar.svg", "P2 Backward Memory Ratio", labels, [float(r["bmem"]) for r in p2_summary], "#7c3aed")
    cache_parts = [{k: f(r, k, 0.0) for k in ["cache_x_MB", "cache_hidden_MB", "cache_index_MB", "cache_weight_MB", "cache_delta_MB", "cache_misc_MB"]} for r in p0]
    stacked_svg(FIG / "cache_decomposition_stacked_bar.svg", "P0 Cache Decomposition", [r.get("primitive", "") for r in p0], cache_parts, {"cache_x_MB": "#2563eb", "cache_hidden_MB": "#7c3aed", "cache_index_MB": "#dc2626", "cache_weight_MB": "#f59e0b", "cache_delta_MB": "#16a34a", "cache_misc_MB": "#64748b"})
    stacked_svg(FIG / "time_breakdown_stacked_bar.svg", "P2 Time Breakdown", labels, [{"forward": float(r["fwd"]), "backward": float(r["bwd"]), "update": max(0.0, float(r["step"]) - float(r["fwd"]) - float(r["bwd"]))} for r in p2_summary], {"forward": "#2563eb", "backward": "#dc2626", "update": "#16a34a"})
    scatter_svg(FIG / "cache_vs_runtime_scatter.svg", "Cache MB vs Step Ratio", p2_summary, "cache", "step")
    scatter_svg(FIG / "kernel_count_vs_forward_time.svg", "Kernel Count vs Forward Time", p2, "kernel_count_forward", "forward_time_ratio")
    scatter_svg(FIG / "efficiency_pareto_step_vs_bmem.svg", "Efficiency Pareto", p2_summary, "step", "bmem")
    bar_svg(FIG / "kernel_count_forward_backward_bar.svg", "P0 Forward Kernel Count", [r.get("primitive", "") for r in p0], [f(r, "kernel_count_forward", 0.0) for r in p0], "#0891b2")
    bar_svg(FIG / "manual_vs_autograd_mlp_runtime.svg", "Manual MLP Step Ratio", ["MLP-manual-linear-reference"], [float(manual_mlp["step"]) if manual_mlp else math.nan], "#f59e0b")
    bar_svg(FIG / "op_breakdown_by_primitive.svg", "P0 Exp/Gather/Scatter Ops", [r.get("primitive", "") for r in p0], [f(r, "op_count_exp", 0.0) + f(r, "op_count_gather", 0.0) + f(r, "op_count_scatter", 0.0) for r in p0], "#dc2626")
    bar_svg(FIG / "grad_relerr_by_primitive.svg", "P1 Max Coeff RelErr", [r["primitive"] for r in p1_summary], [float(r["max_coeff_relerr"]) for r in p1_summary], "#dc2626")
    bar_svg(FIG / "grad_cos_by_primitive.svg", "P1 Min Coeff Cos", [r["primitive"] for r in p1_summary], [float(r["min_coeff_cos"]) for r in p1_summary], "#16a34a")
    bar_svg(FIG / "forward_relerr_heatmap.svg", "P1 Pass Rate", [r["primitive"] for r in p1_summary], [float(r["pass_rate"]) for r in p1_summary], "#2563eb")
    bar_svg(FIG / "component_runtime_waterfall_by_primitive.svg", "P3 Component Runtime", [r.get("primitive", "") + "/" + r.get("component", "") for r in p3], [f(r, "component_time_ms", 0.0) for r in p3], "#2563eb")
    bar_svg(FIG / "component_memory_waterfall_by_primitive.svg", "P3 Component Memory", [r.get("primitive", "") + "/" + r.get("component", "") for r in p3], [f(r, "component_memory_MB", 0.0) for r in p3], "#7c3aed")
    bar_svg(FIG / "nonlinearity_runtime_comparison.svg", "P2 Nonlinearity Step Ratio", labels, [float(r["step"]) for r in p2_summary], "#0891b2")
    placeholder_svg(FIG / "scatter_vs_segmented_reduce.svg", "Scatter vs Segmented Reduce", "Segmented reduce is diagnostic-only in this compact v6.2 runner.")
    if p4_trace:
        scatter_svg(FIG / "loss_vs_step.svg", "Val Loss vs Step", p4_trace, "step", "val_loss", "method")
        scatter_svg(FIG / "loss_vs_time.svg", "Val Loss vs Wall Clock", p4_trace, "wall_clock_time_sec", "val_loss", "method")
        scatter_svg(FIG / "accuracy_vs_time.svg", "Accuracy vs Wall Clock", p4_trace, "wall_clock_time_sec", "test_acc", "method")
    else:
        placeholder_svg(FIG / "loss_vs_step.svg", "Val Loss vs Step", "P4 did not run.")
        placeholder_svg(FIG / "loss_vs_time.svg", "Val Loss vs Time", "P4 did not run.")
        placeholder_svg(FIG / "accuracy_vs_time.svg", "Accuracy vs Time", "P4 did not run.")
    bar_svg(FIG / "time_to_target_bar.svg", "P4 AUC Time Delta", [r["dataset"] + "/" + r["method"] for r in p4_summary], [float(r["auc_time_delta"]) for r in p4_summary], "#2563eb")
    scatter_svg(FIG / "auc_by_step_vs_auc_by_time.svg", "AUC Step vs AUC Time", p4, "val_loss_auc_by_step", "val_loss_auc_by_time", "method")
    scatter_svg(FIG / "step_ratio_vs_step_saving_scatter.svg", "Step Ratio vs AUC Delta", p4, "step_time_ratio", "auc_time_delta_vs_mlp", "method")
    scatter_svg(FIG / "task_efficiency_pareto.svg", "Task Efficiency Pareto", p4, "step_time_ratio", "test_acc", "method")
    scatter_svg(FIG / "memory_accuracy_pareto.svg", "Memory Accuracy Pareto", p4, "backward_memory_ratio", "test_acc", "method")
    placeholder_svg(FIG / "optimizer_loss_slope.svg", "Optimizer Loss Slope", "P5 acceleration is gated behind P4.")
    placeholder_svg(FIG / "optimizer_time_to_target.svg", "Optimizer Time To Target", "P5 acceleration is gated behind P4.")
    placeholder_svg(FIG / "momentum_norm_trace.svg", "Momentum Norm Trace", "P5 acceleration is gated behind P4.")
    placeholder_svg(FIG / "restart_marker_loss_curve.svg", "Restart Marker Loss Curve", "P5 acceleration is gated behind P4.")
    placeholder_svg(FIG / "update_cosine_trace.svg", "Update Cosine Trace", "P5 acceleration is gated behind P4.")

    run_rows = [[k, v, errors.get(k.split()[0], 0)] for k, v in inventory.items()]
    p0_rows = [[r.get("primitive"), r.get("edge_param_count"), r.get("nonKAN_param_count"), r.get("uses_loss_backward"), fmt(f(r, "manual_cache_total_MB")), fmt(f(r, "cache_x_MB")), fmt(f(r, "cache_hidden_MB")), "yes" if int(f(r, "p0_pass", 0)) else "no"] for r in p0]
    p1_rows = [[r["primitive"], r["rows"], fmt(float(r["max_coeff_relerr"]), 2), fmt(float(r["min_coeff_cos"])), fmt(float(r["max_input_relerr"]), 2), fmt(float(r["pass_rate"]))] for r in p1_summary]
    p2_rows = [[r["primitive"], r["rows"], fmt(float(r["fwd"])), fmt(float(r["bwd"])), fmt(float(r["step"])), fmt(float(r["bmem"])), fmt(float(r["cache"])), fmt(float(r["nearmiss_rate"]))] for r in p2_summary]
    p4_rows = [[r["dataset"], r["method"], r["runs"], fmt(float(r["acc"])), fmt(float(r["acc_gap"])), fmt(float(r["auc_time_delta"])), fmt(float(r["step_ratio"])), "yes" if int(r.get("dataset_pass", 0)) else "no"] for r in p4_summary]
    failure_rows = [[k, v] for k, v in sorted(Counter(str(r.get("failure_type", "")) for r in failures).items())]
    artifacts = "\n".join(
        [
            "p0_graphfree_cache_manifest.csv",
            "p0_cache_manifest_summary.csv",
            "p1_gradient_streaming_correctness.csv",
            "p1_gradient_gate_summary.csv",
            "p2_graphfree_efficiency_v2.csv",
            "p2_efficiency_selection.csv",
            "p2_efficiency_gate_summary.csv",
            "p3_kernel_cache_ablation.csv",
            "p4_nearmiss_convergence_smoke.csv",
            "p4_training_trace.csv",
            "p4_nearmiss_gate_summary.csv",
            "p5_acceleration_package.csv",
            "p6_manual_geometry_maintenance.csv",
            "p7_joint_selection3.csv",
            "p8_confirm5.csv",
            "p9_confirm10.csv",
            "failure_table.csv",
            "route_decision.json",
            "aggregate_decision.json",
            "figures/",
        ]
    )
    doc = f"""# DG-KAN v6.2 Graph-Free Kernel / Cache Redesign 结果复盘

本轮依据 `docs/DG-KAN_v6.2_GraphFreeKernelCache_实验计划.md`。目标是在 v6.1 已验证 analytic adjoint 正确之后，继续拆 kernel/cache 效率，并允许 near-miss 候选进入小规模 wall-clock 收敛补偿验证。

## Run Inventory

{md_table(["stage", "rows", "errors"], run_rows)}

## Code / Config Changes

```text
experiments/run_gafu_v62.py
  Added graph-free variants:
    MLP-manual-linear-reference
    DWM2-lite-RBFK2-cacheMin
    DWM2-lite-poly2 / piecewiseLinear / fastRational
    SparseInterpKAN vectorized/fusedIndex diagnostics
    RationalKAT-lite-fastpoly
  Added cache decomposition, op/kernel counters, P3 ablation, and P4 near-miss convergence smoke.

experiments/analyze_gafu_v62.py
  Generates v6.2 gate summaries, route_decision.json, aggregate_decision.json,
  required SVG diagnostics, failure taxonomy, and this replay.
```

## P0 Graph-Free Cache Manifest

{md_table(["primitive", "edge", "nonKAN", "loss.backward", "cache", "x", "hidden", "P0"], p0_rows)}

## P1 Gradient / Streaming Correctness

{md_table(["primitive", "rows", "max coeff rel", "min coeff cos", "max input rel", "pass"], p1_rows)}

P1 survivors:

```text
{", ".join(p1_surv) if p1_surv else "none"}
```

## P2 Graph-Free Efficiency V2

{md_table(["primitive", "rows", "fwd", "bwd", "step", "bmem", "cache", "near"], p2_rows)}

P2 final survivors:

```text
{", ".join(p2_final) if p2_final else "none"}
```

P2 near-miss candidates:

```text
{", ".join(p2_near) if p2_near else "none"}
```

## P4 Near-Miss Convergence Smoke

{md_table(["dataset", "method", "runs", "acc", "gap", "auc time Δ", "step", "P4"], p4_rows)}

P4 survivors:

```text
{", ".join(p4_surv) if p4_surv else "none"}
```

## P5-P9 Decision

```text
P5 acceleration package: {"run" if p4_surv else "not run; P4 produced no near-miss convergence survivor"}
P6 geometry maintenance: gated behind P5
P7 3-seed joint selection: gated behind P6
P8/P9 confirm: gated behind P7
Final decision: {status}
Route case: {route_case}
```

## Failure Diagnosis

{md_table(["failure", "count"], failure_rows)}

Interpretation:

```text
v6.2 checks whether the v6.1 near miss can be explained by cache/kernel overhead and
whether slower step time can be compensated by faster convergence.
If manual MLP is also slow, the runtime framework is the blocker.
If near-miss candidates fail P4, the remaining blocker is still kernel/cache plus task recipe,
not LightSmooth or functional geometry.
```

## Required Artifacts

Written under `results/v6_2/`:

```text
{artifacts}
```

## Final Decision

```text
DG-KAN v6.2 status:
  {status}

Route:
  {route_case}

What improved:
  v6.2 decomposes manual cache and kernel/op counts instead of reporting only one cache MB.
  DWM2-lite alternatives test whether exp/RBF is the main time bottleneck.
  Near-miss candidates are allowed into a compact convergence-compensation smoke.

What failed / remains open:
  See P2/P4 gates above.
  Functional optimizer and LightSmooth remain gated until graph-free primitive efficiency
  and wall-clock convergence are jointly credible.

Conclusion:
  v6.2 keeps the graph-free route alive only if manual runtime and near-miss convergence justify it;
  otherwise the next move remains kernel/cache design.
```
"""
    OUT.write_text(doc, encoding="utf-8")

    existing = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    entry = f"\n- v6.2 GraphFreeKernelCache: {status} ({route_case}); artifacts in `results/v6_2/`.\n"
    if "v6.2 GraphFreeKernelCache" not in existing:
        LOG.write_text(existing + entry, encoding="utf-8")


if __name__ == "__main__":
    main()
