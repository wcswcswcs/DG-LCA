#!/usr/bin/env python3
"""Summarize DG-KAN v6.1 graph-free analytic-adjoint experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v6_1"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v6.1_GraphFreeAnalyticAdjoint_结果复盘.md"
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


def write_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float], color: str = "#2563eb") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1000
    height = max(180, 64 + 26 * len(labels))
    finite = [v for v in values if math.isfinite(v)]
    max_v = max(finite or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="18" y="30" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 52 + i * 26
        bar = 0.0 if not math.isfinite(value) else 520 * value / max(1.0e-12, max_v)
        lines.append(f'<text x="18" y="{y+13}" font-family="Arial" font-size="10">{label[:58]}</text>')
        lines.append(f'<rect x="420" y="{y}" width="{bar:.1f}" height="15" fill="{color}" opacity="0.86"/>')
        lines.append(f'<text x="{430+bar:.1f}" y="{y+12}" font-family="Arial" font-size="10">{fmt(value,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scatter(path: Path, title: str, rows: Sequence[dict], x_key: str, y_key: str, label_key: str = "primitive") -> None:
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
        color = "#16a34a" if "Sparse" in label else "#dc2626" if "Rational" in label else "#0891b2" if "DWM2" in label else "#64748b"
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
    p0 = read_rows(BASE / "p0_graphfree_invariants.csv")
    p1 = read_rows(BASE / "p1_analytic_gradient_correctness.csv")
    p2 = read_rows(BASE / "p2_graphfree_microbenchmark.csv")
    p2_sel = read_rows(BASE / "p2_efficiency_selection.csv")
    p3 = read_rows(BASE / "p3_primitive_efficiency_selection.csv")
    p4 = read_rows(BASE / "p4_manual_training_smoke.csv")
    p5 = read_rows(BASE / "p5_manual_adam_convergence.csv")
    p6 = read_rows(BASE / "p6_functional_without_autograd.csv")
    p7 = read_rows(BASE / "p7_acceleration_package.csv")
    p8 = read_rows(BASE / "p8_manual_lightsmooth_geometry.csv")
    p9 = read_rows(BASE / "p9_joint_selection3.csv")
    p10a = read_rows(BASE / "p10_confirm5.csv")
    p10b = read_rows(BASE / "p10_confirm10.csv")
    failures = read_rows(BASE / "failure_table.csv")

    p0_summary = [{
        "rows": len(p0),
        "errors": sum(1 for r in p0 if r.get("error")),
        "manual_rows": sum(1 for r in p0 if int(f(r, "manual_backward_available", 0)) == 1),
        "manual_pass": sum(1 for r in p0 if int(f(r, "p0_pass", 0)) == 1),
        "max_rollback": max([f(r, "rollback_max_abs_error", 0.0) for r in p0] or [0.0]),
        "max_saved_tensors_manual": max([f(r, "saved_tensor_count", 0.0) for r in p0 if int(f(r, "manual_backward_available", 0)) == 1] or [0.0]),
    }]
    write_csv(BASE / "p0_graphfree_gate_summary.csv", p0_summary)

    p1_summary: list[dict] = []
    for (primitive,), rs in sorted(grouped(p1, "primitive").items()):
        p1_summary.append({
            "primitive": primitive,
            "rows": len(rs),
            "max_coeff_relerr": max([f(r, "grad_coeff_relerr", 99) for r in rs] or [99]),
            "min_coeff_cosine": min([f(r, "grad_coeff_cosine", -1) for r in rs] or [-1]),
            "max_input_relerr": max([f(r, "grad_input_relerr", 99) for r in rs] or [99]),
            "min_input_cosine": min([f(r, "grad_input_cosine", -1) for r in rs] or [-1]),
            "explore_rate": mean(f(r, "p1_exploratory_pass", 0) for r in rs),
            "final_rate": mean(f(r, "p1_final_pass", 0) for r in rs),
        })
    write_csv(BASE / "p1_gradient_gate_summary.csv", p1_summary)

    p2_summary: list[dict] = []
    for (primitive,), rs in sorted(grouped(p2, "primitive").items()):
        if primitive == "MLP-autograd-reference":
            continue
        p2_summary.append({
            "primitive": primitive,
            "rows": len(rs),
            "fwd": mean(f(r, "forward_time_ratio_vs_mlp") for r in rs),
            "bwd": mean(f(r, "backward_time_ratio_vs_mlp") for r in rs),
            "bmem": mean(f(r, "backward_memory_ratio_vs_mlp") for r in rs),
            "step": mean(f(r, "step_time_ratio_vs_mlp") for r in rs),
            "saved_mb": mean(f(r, "saved_tensor_total_mb", 0.0) for r in rs),
            "manual_cache_mb": mean(f(r, "manual_cache_mb", 0.0) for r in rs),
            "explore_rate": mean(f(r, "p2_exploratory_pass", 0) for r in rs),
            "final_rate": mean(f(r, "p2_final_pass", 0) for r in rs),
        })
    write_csv(BASE / "p2_efficiency_gate_summary.csv", p2_summary)

    p1_surv = sorted(r["primitive"] for r in p1_summary if float(r["explore_rate"]) >= 1.0)
    p2_surv = sorted({str(r.get("primitive")) for r in p2_sel if int(f(r, "p2_survivor", 0)) == 1})
    if not p1_surv:
        status = "stop_after_p1_grad_correctness_fail"
        route_case = "B_manual_adjoint_formula_failed"
    elif not p2_surv:
        status = "stop_after_p2_no_graphfree_efficiency_survivor"
        route_case = "D_no_primitive_passed_p2_efficiency"
    elif not any(int(f(r, "p3_pass", 0)) == 1 for r in p3):
        status = "stop_after_p3_no_descent_survivor"
        route_case = "C_efficiency_candidate_no_descent"
    else:
        status = "continue_to_manual_training_gates"
        route_case = "E_graphfree_candidate"

    inventory = {
        "P0 graph-free invariants": len(p0),
        "P1 analytic gradient": len(p1),
        "P2 graph-free microbenchmark": len(p2),
        "P3 primitive selection": len(p3),
        "P4 manual smoke": len(p4),
        "P5 manual Adam": len(p5),
        "P6 functional no autograd": len(p6),
        "P7-P10 gated": len(p7) + len(p8) + len(p9) + len(p10a) + len(p10b),
    }
    errors = {
        "P0": sum(1 for r in p0 if r.get("error")),
        "P1": sum(1 for r in p1 if r.get("error")),
        "P2": sum(1 for r in p2 if r.get("error")),
        "P3": sum(1 for r in p3 if r.get("error")),
        "P4-P10": sum(1 for rows in (p4, p5, p6, p7, p8, p9, p10a, p10b) for r in rows if r.get("error")),
    }
    route = {
        "version": "v6.1",
        "status": status,
        "case": route_case,
        "p1_survivors": p1_surv,
        "p2_survivors": p2_surv,
    }
    aggregate = {"version": "v6.1", "status": status, "route_case": route_case, "run_inventory": inventory, "errors": errors}
    save_json(BASE / "route_decision.json", route)
    save_json(BASE / "aggregate_decision.json", aggregate)

    write_bar(FIG / "graphfree_correctness_dashboard.svg", "P1 Max Coefficient RelErr", [r["primitive"] for r in p1_summary], [float(r["max_coeff_relerr"]) for r in p1_summary], "#dc2626")
    write_bar(FIG / "p1_gradient_relerr_heatmap.svg", "P1 Max Input RelErr", [r["primitive"] for r in p1_summary], [float(r["max_input_relerr"]) for r in p1_summary], "#f59e0b")
    write_bar(FIG / "p1_gradient_cosine_bar.svg", "P1 Min Coefficient Cosine", [r["primitive"] for r in p1_summary], [float(r["min_coeff_cosine"]) for r in p1_summary], "#16a34a")
    write_scatter(FIG / "p1_grad_norm_scatter.svg", "P1 Manual vs Autograd Grad Norm", p1, "grad_norm_manual", "grad_norm_autograd")
    write_scatter(FIG / "p2_efficiency_pareto.svg", "P2 Step Ratio vs Memory Ratio", p2, "step_time_ratio_vs_mlp", "backward_memory_ratio_vs_mlp")
    write_bar(FIG / "p2_step_time_stacked.svg", "P2 Mean Step Ratio", [r["primitive"] for r in p2_summary], [float(r["step"]) for r in p2_summary], "#2563eb")
    write_bar(FIG / "p2_memory_stacked.svg", "P2 Mean Backward Memory Ratio", [r["primitive"] for r in p2_summary], [float(r["bmem"]) for r in p2_summary], "#7c3aed")
    write_scatter(FIG / "p2_batch_scaling.svg", "P2 Batch Scaling", p2, "batch_size", "step_time_ms")
    write_bar(FIG / "p2_kernel_breakdown_heatmap.svg", "P2 Approx Kernel Count", [r.get("primitive", "") + "/" + r.get("shape", "") for r in p2 if r.get("shape") == "B512-H96-D4"], [f(r, "kernel_count", 0.0) for r in p2 if r.get("shape") == "B512-H96-D4"], "#0891b2")
    placeholder_svg(FIG / "task_efficiency_pareto.svg", "Task Efficiency Pareto", "Task recipes are gated behind P2 graph-free efficiency survival.")
    placeholder_svg(FIG / "convergence_speed_plots.svg", "Convergence Speed", "Manual training convergence is gated behind P3.")
    placeholder_svg(FIG / "geometry_plots.svg", "Geometry Plots", "LightSmooth/geometry remains gated behind graph-free task survival.")
    placeholder_svg(FIG / "representation_plots.svg", "Representation Plots", "Representation diagnostics are gated behind graph-free task survival.")
    write_bar(FIG / "failure_taxonomy_heatmap.svg", "Failure Taxonomy", list(Counter(r.get("failure_type", "") for r in failures).keys()), list(Counter(r.get("failure_type", "") for r in failures).values()), "#dc2626")
    placeholder_svg(FIG / "gate_dashboard.svg", "Gate Dashboard", f"Final status: {status}; route: {route_case}.")

    run_rows = [[k, v, errors.get(k.split()[0], 0)] for k, v in inventory.items()]
    p0_rows = [[r.get("primitive"), r.get("edge_param_count"), r.get("nonKAN_param_count"), r.get("uses_loss_backward"), r.get("saved_tensor_count"), fmt(f(r, "rollback_max_abs_error")), "yes" if int(f(r, "p0_pass", 0)) else "no"] for r in p0]
    p1_rows = [[r["primitive"], r["rows"], fmt(float(r["max_coeff_relerr"]), 2), fmt(float(r["min_coeff_cosine"])), fmt(float(r["max_input_relerr"]), 2), fmt(float(r["min_input_cosine"])), fmt(float(r["explore_rate"]))] for r in p1_summary]
    p2_rows = [[r["primitive"], r["rows"], fmt(float(r["fwd"])), fmt(float(r["bwd"])), fmt(float(r["bmem"])), fmt(float(r["step"])), fmt(float(r["manual_cache_mb"]), 2), fmt(float(r["explore_rate"]))] for r in p2_summary]
    failure_rows = [[k, v] for k, v in sorted(Counter(str(r.get("failure_type", "")) for r in failures).items())]
    artifacts = "\n".join(
        [
            "p0_graphfree_invariants.csv",
            "p0_graphfree_gate_summary.csv",
            "p1_analytic_gradient_correctness.csv",
            "p1_gradient_gate_summary.csv",
            "p2_graphfree_microbenchmark.csv",
            "p2_efficiency_selection.csv",
            "p2_efficiency_gate_summary.csv",
            "p3_primitive_efficiency_selection.csv",
            "p4_manual_training_smoke.csv",
            "p5_manual_adam_convergence.csv",
            "p6_functional_without_autograd.csv",
            "p7_acceleration_package.csv",
            "p8_manual_lightsmooth_geometry.csv",
            "p9_joint_selection3.csv",
            "p10_confirm5.csv",
            "p10_confirm10.csv",
            "failure_table.csv",
            "route_decision.json",
            "aggregate_decision.json",
            "figures/",
        ]
    )
    doc = f"""# DG-KAN v6.1 Graph-Free Analytic Adjoint 结果复盘

本轮依据 `docs/DG-KAN_v6.1_GraphFreeAnalyticAdjoint_实验计划.md`。核心目标是把“functional update rule”和“analytic adjoint / graph-free training”分开：先确认不依赖 `loss.backward()` 的 manual forward/backward/update 是否正确且更省显存，再决定是否进入 task recipe、LightSmooth 或 functional optimizer。

## Run Inventory

{md_table(["stage", "rows", "errors"], run_rows)}

## Code / Config Changes

```text
experiments/run_gafu_v61.py
  Added graph-free ManualLayer / ManualStack primitives for SparseInterp, LUT, RBF-lite, and Rational-lite.
  Added P0 graph-free invariants, P1 analytic gradient correctness, P2 graph-free efficiency benchmark,
  and gated P3-P10 artifacts.

experiments/analyze_gafu_v61.py
  Generates graph-free gate summaries, route_decision.json, aggregate_decision.json,
  required SVG diagnostics, failure taxonomy, and this replay.
```

## P0 Graph-Free Invariants

{md_table(["primitive", "edge", "nonKAN", "loss.backward", "saved", "rollback", "P0"], p0_rows)}

P0 verdict: pass for manual primitives when edge coverage is nonzero, rollback is exact, and no autograd saved-tensor path is used.

## P1 Analytic Gradient Correctness

{md_table(["primitive", "rows", "max coeff rel", "min coeff cos", "max input rel", "min input cos", "P1 rate"], p1_rows)}

P1 survivors:

```text
{", ".join(p1_surv) if p1_surv else "none"}
```

## P2 Graph-Free Efficiency Benchmark

{md_table(["primitive", "rows", "fwd", "bwd", "bmem", "step", "cache MB", "P2 rate"], p2_rows)}

P2 survivors:

```text
{", ".join(p2_surv) if p2_surv else "none"}
```

## P3-P10 Decision

```text
P3 primitive efficiency selection: {"run" if p2_surv else "not run; P2 produced no graph-free efficiency survivor"}
P4 manual training smoke: gated behind P3
P5 manual Adam convergence: gated behind P4
P6 functional without autograd: gated behind P5
P7/P8/P9/P10 confirm: gated behind P6
Final decision: {status}
Route case: {route_case}
```

## Failure Diagnosis

{md_table(["failure", "count"], failure_rows)}

Interpretation:

```text
v6.1 verifies the graph-free analytic-adjoint premise directly.
If P1 fails, the blocker is manual adjoint correctness.
If P1 passes but P2 fails, the blocker is still primitive/cache/kernel efficiency,
not LightSmooth or functional optimizer design.
```

## Required Artifacts

Written under `results/v6_1/`:

```text
{artifacts}
```

## Final Decision

```text
DG-KAN v6.1 status:
  {status}

Route:
  {route_case}

What improved:
  v6.1 adds a true graph-free manual-adjoint benchmark path.
  P1 compares manual gradients against autograd without making autograd the training path.
  P2 measures graph-free cache/memory/time ratios directly against an MLP autograd reference.

What failed / remains open:
  See P1-P3 gates above.
  Task recipe, LightSmooth, and functional optimizer remain gated until graph-free adjoint + efficiency survive.

Conclusion:
  v6.1 makes the next blocker explicit: either fix the manual adjoint formulas, or redesign the primitive/cache path
  until a graph-free candidate passes the P2 efficiency envelope.
```
"""
    OUT.write_text(doc, encoding="utf-8")

    existing = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    entry = f"\n- v6.1 GraphFreeAnalyticAdjoint: {status} ({route_case}); artifacts in `results/v6_1/`.\n"
    if "v6.1 GraphFreeAnalyticAdjoint" not in existing:
        LOG.write_text(existing + entry, encoding="utf-8")


if __name__ == "__main__":
    main()
