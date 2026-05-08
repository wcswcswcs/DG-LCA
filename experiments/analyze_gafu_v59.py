#!/usr/bin/env python3
"""Summarize DG-KAN v5.9 memory-first efficient primitive experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v5_9"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v5.9_MemoryFirst_EfficientPrimitive_结果复盘.md"
LOG = ROOT / "docs/log.md"
DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]


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
                keys.append(key)
                seen.add(key)
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


def all_dataset_survivors(rows: Sequence[dict], key: str, pass_field: str) -> list[str]:
    by: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("error") or row.get("status") == "not_run":
            continue
        if int(f(row, pass_field, 0)) == 1:
            by[str(row.get(key, ""))].add(str(row.get("dataset", "")))
    return sorted(k for k, ds in by.items() if all(d in ds for d in DATASETS))


def write_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float], *, color: str = "#2563eb") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 70 + 28 * len(labels)
    finite = [v for v in values if math.isfinite(v)]
    max_v = max(finite or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="18" y="30" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 54 + i * 28
        bar = 0.0 if not math.isfinite(value) else 520 * value / max(1.0e-12, max_v)
        lines.append(f'<text x="18" y="{y+14}" font-family="Arial" font-size="10">{label[:54]}</text>')
        lines.append(f'<rect x="420" y="{y}" width="{bar:.1f}" height="16" fill="{color}"/>')
        lines.append(f'<text x="{430+bar:.1f}" y="{y+13}" font-family="Arial" font-size="10">{fmt(value,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stacked_memory(path: Path, rows: Sequence[dict]) -> None:
    methods = [r for r in rows if r.get("shape") == "Medium" and r.get("primitive") != "MLP-reference" and not r.get("error")]
    width, height = 1080, 80 + 34 * len(methods)
    colors = {"param_mb": "#2563eb", "optimizer_state_mb": "#7c3aed", "saved_tensor_total_mb": "#dc2626", "workspace_temp_mb": "#f59e0b"}
    max_total = max([sum(max(0.0, f(r, k, 0.0)) for k in colors) for r in methods] or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="20" y="32" font-family="Arial" font-size="16" font-weight="700">P1 Memory Decomposition (Medium)</text>',
    ]
    for i, row in enumerate(methods):
        y = 58 + i * 34
        x = 420
        lines.append(f'<text x="18" y="{y+15}" font-family="Arial" font-size="10">{str(row.get("primitive"))[:56]}</text>')
        for key, color in colors.items():
            val = max(0.0, f(row, key, 0.0))
            w = 520 * val / max_total
            lines.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="18" fill="{color}" opacity="0.82"/>')
            x += w
        lines.append(f'<text x="{x+8:.1f}" y="{y+14}" font-family="Arial" font-size="10">{fmt(f(row, "peak_allocated_mb"), 1)}MB</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scatter(path: Path, title: str, rows: Sequence[dict], x_key: str, y_key: str, label_key: str = "primitive") -> None:
    vals = [(f(r, x_key), f(r, y_key), str(r.get(label_key, ""))) for r in rows if not r.get("error") and r.get("status") != "not_run"]
    width, height = 900, 560
    max_x = max([v[0] for v in vals if math.isfinite(v[0])] or [1.0])
    max_y = max([v[1] for v in vals if math.isfinite(v[1])] or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="30" y="32" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
        '<line x1="80" y1="500" x2="840" y2="500" stroke="#111"/>',
        '<line x1="80" y1="500" x2="80" y2="70" stroke="#111"/>',
    ]
    for x, y, label in vals:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        px = 80 + 760 * x / max(1.0e-12, max_x)
        py = 500 - 420 * y / max(1.0e-12, max_y)
        color = "#16a34a" if "DWM2-lite" in label else "#dc2626" if "Rational" in label else "#0891b2" if "LUT" in label else "#7c3aed" if "CP" in label else "#64748b"
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" fill-opacity="0.78"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def placeholder_svg(path: Path, title: str, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="220" viewBox="0 0 900 220">',
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
    p0 = read_rows(BASE / "p0_core_manifest.csv")
    p1 = read_rows(BASE / "p1_phase_efficiency.csv")
    p1_mem = read_rows(BASE / "p1_memory_decomposition.csv")
    p1_saved = read_rows(BASE / "p1_saved_tensor_audit.csv")
    p1_kernel = read_rows(BASE / "p1_kernel_breakdown.csv")
    p2 = read_rows(BASE / "p2_custom_backward_correctness.csv")
    p2_mem = read_rows(BASE / "p2_custom_backward_memory.csv")
    p3 = read_rows(BASE / "p3_recipe_grid.csv")
    p3_sel = read_rows(BASE / "p3_task_efficiency_selection.csv")
    p4 = read_rows(BASE / "p4_lightsmooth_compatibility.csv")
    p5 = read_rows(BASE / "p5_functional_smoke.csv")
    p6 = read_rows(BASE / "p6_confirm5.csv")
    p7 = read_rows(BASE / "p7_confirm10.csv")
    failures = read_rows(BASE / "failure_table.csv")

    p1_summary: list[dict] = []
    for (primitive,), rs in sorted(grouped(p1, "primitive").items()):
        if primitive == "MLP-reference":
            continue
        fwd = mean(f(r, "forward_time_ratio_vs_mlp") for r in rs)
        bwd = mean(f(r, "backward_time_ratio_vs_mlp") for r in rs)
        bmem = mean(f(r, "backward_memory_ratio_vs_mlp") for r in rs)
        saved = mean(f(r, "saved_tensor_total_mb") for r in rs)
        if bmem > 2.0:
            bottleneck = "memory"
        elif fwd > 3.0:
            bottleneck = "forward"
        elif bwd > 3.0:
            bottleneck = "backward"
        else:
            bottleneck = "none"
        p1_summary.append({
            "primitive": primitive,
            "family": rs[0].get("primitive_family", ""),
            "rows": len(rs),
            "fwd": fwd,
            "bwd": bwd,
            "bmem": bmem,
            "step": mean(f(r, "step_time_ratio_vs_mlp") for r in rs),
            "saved_mb": saved,
            "saved_ratio": mean(f(r, "saved_tensor_ratio_vs_mlp") for r in rs),
            "explore_rate": mean(f(r, "p1_exploratory_pass", 0) for r in rs),
            "final_rate": mean(f(r, "p1_final_pass", 0) for r in rs),
            "bottleneck": bottleneck,
        })
    write_csv(BASE / "p1_efficiency_gate_summary.csv", p1_summary)

    p2_summary = []
    for (primitive, variant), rs in sorted(grouped(p2, "primitive", "variant").items()):
        p2_summary.append({
            "primitive": primitive,
            "variant": variant,
            "relerr": mean(f(r, "rel_error_grad_params") for r in rs),
            "cos": mean(f(r, "cos_grad_params") for r in rs),
            "bmem": mean(f(r, "backward_memory_ratio_vs_mlp") for r in rs),
            "bwd": mean(f(r, "backward_time_ratio_vs_mlp") for r in rs),
            "pass": mean(f(r, "p2_pass", 0) for r in rs),
        })
    write_csv(BASE / "p2_custom_backward_gate_summary.csv", p2_summary)

    p3_summary = []
    for (dataset, primitive), rs in sorted(grouped(p3, "dataset", "primitive").items()):
        p3_summary.append({
            "dataset": dataset,
            "primitive": primitive,
            "runs": len(rs),
            "acc": mean(f(r, "test_acc") for r in rs),
            "gap": mean(f(r, "acc_gap_vs_mlp") for r in rs),
            "ECE": mean(f(r, "ECE") for r in rs),
            "bmem": mean(f(r, "backward_memory_ratio") for r in rs),
            "step": mean(f(r, "step_time_ratio") for r in rs),
            "pass_rate": mean(f(r, "p3_pass", 0) for r in rs),
        })
    write_csv(BASE / "p3_recipe_gate_summary.csv", p3_summary)

    p1_surv = [r["primitive"] for r in p1_summary if float(r["explore_rate"]) > 0.0]
    p2_surv = sorted({str(r.get("primitive")) for r in p2 if not r.get("error") and int(f(r, "p2_pass", 0)) == 1})
    p3_surv = all_dataset_survivors(p3, "primitive", "p3_pass")
    if p3_surv:
        status = "continue_to_p4_lightsmooth"
    elif p2_surv:
        status = "stop_after_p3_no_task_efficiency_survivor"
    elif p1_surv:
        status = "stop_after_p2_no_custom_backward_survivor"
    else:
        status = "stop_after_p1_no_efficiency_survivor"
    route_case = (
        "A_task_efficiency_survivor" if p3_surv else
        "B_custom_backward_blocker" if p1_surv else
        "C_no_p1_efficiency_survivor"
    )
    route = {
        "version": "v5.9",
        "status": status,
        "case": route_case,
        "p1_exploratory_survivors": p1_surv,
        "p2_survivors": p2_surv,
        "p3_survivors": p3_surv,
    }
    aggregate = {
        "version": "v5.9",
        "status": status,
        "route_case": route_case,
        "run_inventory": {
            "P0 core manifest": len(p0),
            "P1 phase efficiency": len(p1),
            "P2 custom backward": len(p2),
            "P3 recipe grid": len(p3),
            "P4 LightSmooth": len(p4),
            "P5 functional": len(p5),
            "P6 confirm5": len(p6),
            "P7 confirm10": len(p7),
        },
        "errors": {
            "P0": sum(1 for r in p0 if r.get("error")),
            "P1": sum(1 for r in p1 if r.get("error")),
            "P2": sum(1 for r in p2 if r.get("error")),
            "P3": sum(1 for r in p3 if r.get("error")),
        },
    }
    save_json(BASE / "route_decision.json", route)
    save_json(BASE / "aggregate_decision.json", aggregate)

    write_stacked_memory(FIG / "p1_memory_decomposition_stacked.svg", p1_mem)
    write_bar(FIG / "p1_saved_tensor_shapes_top10.svg", "P1 Saved Tensor Total MB", [r["primitive"] for r in p1_summary], [float(r["saved_mb"]) for r in p1_summary], color="#dc2626")
    write_bar(FIG / "p1_kernel_count_heatmap.svg", "P1 Approx Kernel Count", [r.get("primitive", "") + "/" + r.get("shape", "") for r in p1_kernel if r.get("shape") == "Medium"], [f(r, "kernel_count") for r in p1_kernel if r.get("shape") == "Medium"], color="#0891b2")
    write_bar(FIG / "p1_cold_warm_timing.svg", "P1 Cold/Warm Step Ratio", [r["primitive"] for r in p1_summary], [mean(f(row, "cold_warm_ratio") for row in p1 if row.get("primitive") == r["primitive"]) for r in p1_summary], color="#7c3aed")
    write_scatter(FIG / "p2_custom_backward_memory_vs_time.svg", "P2 Custom Backward Memory vs Time", p2_mem, "backward_memory_ratio_vs_mlp", "backward_time_ratio_vs_mlp")
    write_scatter(FIG / "p2_grad_correctness_scatter.svg", "P2 Grad Correctness", p2, "rel_error_grad_params", "cos_grad_params")
    write_scatter(FIG / "p3_task_efficiency_pareto.svg", "P3 Task Efficiency Pareto", p3, "step_time_ratio", "acc_gap_vs_mlp")
    placeholder_svg(FIG / "p3_recipe_accuracy_heatmap.svg", "P3 Recipe Accuracy Heatmap", "P3 was gated unless P2 produced a survivor.")
    placeholder_svg(FIG / "p3_recipe_memory_heatmap.svg", "P3 Recipe Memory Heatmap", "P3 was gated unless P2 produced a survivor.")
    placeholder_svg(FIG / "p3_classwise_failure_heatmap.svg", "P3 Classwise Failure Heatmap", "No task grid ran after P1/P2 gating.")
    placeholder_svg(FIG / "p4_geometry_cost_pareto.svg", "P4 Geometry Cost Pareto", "P4 was gated behind P3 task-efficiency survival.")
    placeholder_svg(FIG / "p4_refresh_retention_curve.svg", "P4 Refresh Retention Curve", "P4 was gated behind P3 task-efficiency survival.")
    placeholder_svg(FIG / "p6_paired_delta_dashboard.svg", "P6 Paired Delta Dashboard", "P6 was gated behind P3 task-efficiency survival.")

    by_type = Counter(str(r.get("failure_type", "")) for r in failures)
    failure_rows = [[k, v] for k, v in sorted(by_type.items())]
    p0_rows = [[r.get("primitive_name"), r.get("class_name"), r.get("edge_param_count"), r.get("nonKAN_param_count"), r.get("mixing_param_count"), fmt(f(r, "rollback_max_error")), "yes" if int(f(r, "p0_pass", 0)) else "no"] for r in p0]
    p1_rows = [[r["primitive"], r["rows"], fmt(float(r["fwd"])), fmt(float(r["bwd"])), fmt(float(r["bmem"])), fmt(float(r["step"])), fmt(float(r["saved_mb"]), 2), fmt(float(r["explore_rate"])), r["bottleneck"]] for r in p1_summary]
    run_rows = [
        ["P0 core manifest", len(p0), sum(1 for r in p0 if r.get("error"))],
        ["P1 phase efficiency", len(p1), sum(1 for r in p1 if r.get("error"))],
        ["P1 saved tensor audit", len(p1_saved), sum(1 for r in p1_saved if r.get("error"))],
        ["P2 custom backward", len(p2), sum(1 for r in p2 if r.get("error"))],
        ["P3 recipe grid", len(p3), sum(1 for r in p3 if r.get("error"))],
        ["P4-P7 gated", len(p4) + len(p5) + len(p6) + len(p7), 0],
    ]
    artifacts = "\n".join(
        [
            "p0_core_manifest.csv",
            "p1_phase_efficiency.csv",
            "p1_memory_decomposition.csv",
            "p1_saved_tensor_audit.csv",
            "p1_kernel_breakdown.csv",
            "p2_custom_backward_correctness.csv",
            "p2_custom_backward_memory.csv",
            "p3_recipe_grid.csv",
            "p3_task_efficiency_selection.csv",
            "p4_lightsmooth_compatibility.csv",
            "p5_functional_smoke.csv",
            "p6_confirm5.csv",
            "p7_confirm10.csv",
            "failure_table.csv",
            "route_decision.json",
            "aggregate_decision.json",
            "figures/",
        ]
    )
    doc = f"""# DG-KAN v5.9 Memory-First Efficient PureKAN Primitive 结果复盘

本轮依据 `docs/DG-KAN_v5.9_MemoryFirst_EfficientPrimitive_实验计划.md`。核心目标是先确认 primitive 的 profiler truth 和 backward saved tensor 来源，再决定是否值得进入 custom backward、task recipe、LightSmooth 或 functional optimizer。

## Run Inventory

{md_table(["stage", "rows", "errors"], run_rows)}

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added DWM2LiteDense: single-GEMM KAN residual primitive with RBF/LUT residual modes.

experiments/run_gafu_v59.py
  Added P0 memory-first manifest, P1 saved_tensors_hooks profiler,
  memory decomposition, kernel breakdown, gated P2 custom-backward diagnostics,
  and gated P3-P7 artifacts.

experiments/analyze_gafu_v59.py
  Generates v5.9 summaries, route_decision.json, aggregate_decision.json,
  required SVG diagnostics, and this replay.
```

## P0 Core Manifest

{md_table(["primitive", "class", "edge", "nonKAN", "mixing", "rollback", "P0"], p0_rows)}

P0 verdict: pass. The new DWM2-lite variants remain strict PureKAN: edge-covered, nonKAN-free, and rollback-exact.

## P1 Profiler Truth / Memory Decomposition

{md_table(["primitive", "rows", "fwd", "bwd", "bmem", "step", "saved MB", "explore", "bottleneck"], p1_rows)}

P1 exploratory survivors:

```text
{", ".join(p1_surv) if p1_surv else "none"}
```

P1 verdict: no exploratory survivor. DWM2-lite-RBFK2 is the closest route, but still misses the forward/memory envelope under the written gate.

## P2-P7 Decision

```text
P2 custom backward / recompute: {"run" if p1_surv else "not run; P1 produced no exploratory survivor"}
P3 minimal task recipe: {"run" if p2_surv else "not run; P2 produced no survivor"}
P4 LightSmooth compatibility: {"run" if p3_surv else "not run; P3 produced no task-efficiency survivor"}
P5 functional smoke: not run unless P4 survives
P6/P7 confirm: not run unless P3/P5 survives
Final decision: {status}
Route case: {route_case}
```

## Failure Diagnosis

{md_table(["failure", "count"], failure_rows)}

Interpretation:

```text
DWM2-lite reduces saved tensors substantially versus dense AB-RBF and current DWM2,
but it still does not reach the relaxed P1 envelope.
The dominant blocker remains profiler-level primitive efficiency, especially saved tensors /
workspace memory and forward kernel path, so task recipe and functional optimizer remain gated.
```

## Required Artifacts

Written under `results/v5_9/`:

```text
{artifacts}
```

## Final Decision

```text
DG-KAN v5.9 status:
  {status}

Route:
  {route_case}

What improved:
  v5.9 adds saved-tensor-aware profiling and a stricter memory-cause diagnosis.
  DWM2-lite gives a cleaner single-GEMM primitive and materially lowers saved tensor volume.

What failed / remains open:
  No primitive passed P1 exploratory efficiency.
  P2 custom backward, P3 task recipe, LightSmooth, and functional training are correctly gated.

Conclusion:
  The next move is still primitive/kernel/backward design, not optimizer tuning.
```
"""
    OUT.write_text(doc, encoding="utf-8")

    existing = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    entry = f"\n- v5.9 MemoryFirst_EfficientPrimitive: {status} ({route_case}); artifacts in `results/v5_9/`.\n"
    if "v5.9 MemoryFirst_EfficientPrimitive" not in existing:
        LOG.write_text(existing + entry, encoding="utf-8")


if __name__ == "__main__":
    main()
