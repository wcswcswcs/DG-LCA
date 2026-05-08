#!/usr/bin/env python3
"""Summarize DG-KAN v5.7 kernel-verified efficient PureKAN experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v5_7"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v5.7_KernelVerified_EfficientPureKAN_结果复盘.md"
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


def std(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


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


def all_dataset_survivors(rows: Sequence[dict], key_fields: Sequence[str], pass_field: str) -> list[tuple[str, ...]]:
    by: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        if row.get("error") or row.get("status") == "not_run":
            continue
        if int(f(row, pass_field, 0)) == 1:
            by[tuple(str(row.get(k, "")) for k in key_fields)].add(str(row.get("dataset", "")))
    return [key for key, ds in by.items() if all(d in ds for d in DATASETS)]


def write_svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
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
        lines.append(f'<text x="18" y="{y+14}" font-family="Arial" font-size="10">{label[:52]}</text>')
        lines.append(f'<rect x="400" y="{y}" width="{bar:.1f}" height="16" fill="#2563eb"/>')
        lines.append(f'<text x="{410+bar:.1f}" y="{y+13}" font-family="Arial" font-size="10">{fmt(value,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_scatter(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 560
    vals = [
        (f(r, "step_time_ratio", f(r, "step_time_ratio_vs_mlp")), f(r, "acc_gap_vs_mlp", 0.0), f(r, "backward_memory_ratio", f(r, "backward_memory_ratio_vs_mlp")), str(r.get("recipe", r.get("method", ""))))
        for r in rows
        if str(r.get("recipe", r.get("method", ""))) not in {"MLP-AdamW", "MLP-reference", ""}
    ]
    max_x = max([v[0] for v in vals if math.isfinite(v[0])] or [2.0])
    max_y = max([v[1] for v in vals if math.isfinite(v[1])] or [0.1])
    colors = {"DWM": "#16a34a", "RK": "#dc2626", "Rational": "#dc2626", "CP": "#7c3aed", "ABRBF": "#2563eb", "RBF": "#64748b"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="30" y="32" font-family="Arial" font-size="16" font-weight="700">Joint accuracy-efficiency Pareto</text>',
        '<line x1="80" y1="500" x2="840" y2="500" stroke="#111"/>',
        '<line x1="80" y1="500" x2="80" y2="70" stroke="#111"/>',
        '<text x="390" y="540" font-family="Arial" font-size="12">step ratio vs MLP</text>',
        '<text x="18" y="250" font-family="Arial" font-size="12" transform="rotate(-90 18,250)">accuracy gap vs MLP</text>',
    ]
    for x, y, mem, label in vals:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        px = 80 + 760 * x / max(1.0e-12, max_x)
        py = 500 - 420 * max(0.0, y) / max(1.0e-12, max_y)
        key = "DWM" if label.startswith("DWM") else "RK" if label.startswith("RK") or "Rational" in label else "CP" if label.startswith("CP") else "ABRBF" if "ABRBF" in label else "RBF"
        radius = 4 + min(8, max(0, mem if math.isfinite(mem) else 1))
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{colors[key]}" fill-opacity="0.76"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_recipe(rows: Sequence[dict], pass_field: str) -> list[dict]:
    out: list[dict] = []
    for (dataset, recipe), rs in sorted(grouped(rows, "dataset", "recipe").items()):
        if recipe == "MLP-AdamW":
            continue
        out.append(
            {
                "dataset": dataset,
                "recipe": recipe,
                "runs": len(rs),
                "acc": mean(f(r, "test_acc") for r in rs),
                "std": std(f(r, "test_acc") for r in rs),
                "gap": mean(f(r, "acc_gap_vs_mlp") for r in rs),
                "ECE": mean(f(r, "ECE") for r in rs),
                "step": mean(f(r, "step_time_ratio") for r in rs),
                "bmem": mean(f(r, "backward_memory_ratio") for r in rs),
                "pass_rate": mean(f(r, pass_field, 0) for r in rs),
                "pass": int(all(int(f(r, pass_field, 0)) == 1 for r in rs)),
            }
        )
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p0 = read_rows(BASE / "p0_core_profiler_consistency.csv")
    p1 = read_rows(BASE / "p1_phase_efficiency_v3.csv")
    cold = read_rows(BASE / "p1_cold_warm_compile_audit.csv")
    mem = read_rows(BASE / "p1_memory_decomposition.csv")
    p2 = read_rows(BASE / "p2_dwm_kernel_recipe.csv")
    p3 = read_rows(BASE / "p3_rational_kernel_recipe.csv")
    safety = read_rows(BASE / "p3_rational_safety.csv")
    p4 = read_rows(BASE / "p4_cp_rank_speed.csv")
    p5 = read_rows(BASE / "p5_joint_task_efficiency_selection.csv")
    p6 = read_rows(BASE / "p6_custom_backward_memory.csv")
    p7 = read_rows(BASE / "p7_functional_lightsmooth_compat.csv")
    p8 = read_rows(BASE / "p8_candidate_selection3.csv")
    p9 = read_rows(BASE / "p9_confirm5.csv")
    p10 = read_rows(BASE / "p10_confirm10.csv")
    failures = read_rows(BASE / "failure_table.csv")

    p1_summary: list[dict] = []
    for (method,), rows in sorted(grouped(p1, "method").items()):
        if method == "MLP-reference":
            continue
        p1_summary.append(
            {
                "method": method,
                "rows": len(rows),
                "fwd": mean(f(r, "forward_time_ratio_vs_mlp") for r in rows),
                "bwd": mean(f(r, "backward_time_ratio_vs_mlp") for r in rows),
                "bmem": mean(f(r, "backward_memory_ratio_vs_mlp") for r in rows),
                "step": mean(f(r, "step_time_ratio_vs_mlp") for r in rows),
                "cold_warm": mean(f(r, "cold_warm_step_ratio") for r in rows),
                "graph_breaks": sum(int(f(r, "graph_break_count", 0)) for r in rows),
                "recompiles": sum(int(f(r, "recompile_count", 0)) for r in rows),
                "explore_rate": mean(f(r, "p1_exploratory_pass", 0) for r in rows),
                "final_rate": mean(f(r, "p1_final_pass", 0) for r in rows),
                "bottleneck": "memory" if mean(f(r, "backward_memory_ratio_vs_mlp") for r in rows) > 1.2 else "time",
            }
        )
    write_csv(BASE / "p1_efficiency_gate_summary.csv", p1_summary)

    p2_summary = summarize_recipe(p2, "p2_pass")
    p3_summary = summarize_recipe(p3, "p3_pass")
    p4_summary = summarize_recipe(p4, "p4_pass")
    p5_summary = summarize_recipe(p5, "p5_pass")
    write_csv(BASE / "p2_dwm_gate_summary.csv", p2_summary)
    write_csv(BASE / "p3_rational_gate_summary.csv", p3_summary)
    write_csv(BASE / "p4_cp_gate_summary.csv", p4_summary)
    write_csv(BASE / "p5_joint_gate_summary.csv", p5_summary)

    p1_exploratory = [r["method"] for r in p1_summary if float(r["explore_rate"]) > 0.0]
    p1_final = [r["method"] for r in p1_summary if float(r["final_rate"]) > 0.0]
    p2_surv = all_dataset_survivors(p2, ["recipe"], "p2_pass")
    p3_surv = all_dataset_survivors(p3, ["recipe"], "p3_pass")
    p4_surv = all_dataset_survivors(p4, ["recipe"], "p4_pass")
    p5_surv = all_dataset_survivors(p5, ["recipe"], "p5_pass")

    if p5_surv:
        status = "continue_to_p6_custom_backward"
    elif p2_surv or p3_surv or p4_surv:
        status = "stop_after_p5_no_joint_survivor"
    elif p1_exploratory:
        status = "stop_after_p4_no_recipe_survivor"
    else:
        status = "stop_after_p1_no_efficiency_survivor"

    route_case = (
        "A_continue" if p5_surv else
        "B_recipe_blocker" if p1_exploratory and not p5_surv else
        "C_no_primitive_survivor"
    )
    interpretation = (
        "A kernel path appears in the profiler, but task recipe / all-dataset quality does not produce a joint survivor."
        if p1_exploratory and not p5_surv
        else "No strict PureKAN primitive reached the exploratory MLP-like efficiency envelope."
    )
    route = {
        "version": "v5.7",
        "status": status,
        "case": route_case,
        "interpretation": interpretation,
        "next_recommended_route": "Do not resume FGO-v3 yet; repair or redesign the efficient primitive recipe/kernel first.",
    }
    save_json(BASE / "route_decision.json", route)

    fail_by_type = Counter(str(r.get("primary_failure", "")) for r in failures)
    fail_by_method = Counter(str(r.get("method", "")) for r in failures)
    write_csv(BASE / "failure_by_type.csv", [{"failure": k, "count": v} for k, v in fail_by_type.most_common()])
    write_csv(BASE / "failure_by_method.csv", [{"method": k, "count": v} for k, v in fail_by_method.most_common()])

    ordered = [r["method"] for r in p1_summary]
    write_svg_bar(FIG / "p0_edge_manifest.svg", "P0 edge parameter count", [r.get("primitive_name", "") for r in p0], [f(r, "edge_param_count", 0) for r in p0])
    write_svg_bar(FIG / "p1_step_ratio.svg", "P1 warm step ratio", ordered, [float(r["step"]) for r in p1_summary])
    write_svg_bar(FIG / "p1_backward_memory_ratio.svg", "P1 backward memory ratio", ordered, [float(r["bmem"]) for r in p1_summary])
    write_svg_bar(FIG / "p1_cold_warm_ratio.svg", "P1 cold/warm step ratio", ordered, [float(r["cold_warm"]) for r in p1_summary])
    write_svg_bar(FIG / "p2_dwm_accuracy.svg", "P2 DWM accuracy", [r["recipe"] for r in p2_summary], [float(r["acc"]) for r in p2_summary])
    write_svg_bar(FIG / "p3_rational_accuracy.svg", "P3 Rational accuracy", [r["recipe"] for r in p3_summary], [float(r["acc"]) for r in p3_summary])
    write_svg_bar(FIG / "p4_cp_accuracy.svg", "P4 CP accuracy", [r["recipe"] for r in p4_summary], [float(r["acc"]) for r in p4_summary])
    write_svg_scatter(FIG / "joint_accuracy_efficiency_pareto.svg", p5_summary or p2_summary + p3_summary + p4_summary)

    agg = {
        "version": "v5.7",
        "status": status,
        "p1_exploratory_survivors": p1_exploratory,
        "p1_final_survivors": p1_final,
        "p2_dwm_survivors": ["|".join(x) for x in p2_surv],
        "p3_rational_survivors": ["|".join(x) for x in p3_surv],
        "p4_cp_survivors": ["|".join(x) for x in p4_surv],
        "p5_survivors": ["|".join(x) for x in p5_surv],
        "rows": {
            "P0": len(p0),
            "P1": len(p1),
            "P1 cold/warm": len(cold),
            "P1 memory": len(mem),
            "P2": len(p2),
            "P3": len(p3),
            "P3 safety": len(safety),
            "P4": len(p4),
            "P5": len(p5),
            "P6": len(p6),
            "P7": len(p7),
            "P8": len(p8),
            "P9": len(p9),
            "P10": len(p10),
        },
    }
    save_json(BASE / "aggregate_decision.json", agg)

    p0_rows = [
        [
            r.get("primitive_name"),
            r.get("primitive_class"),
            r.get("edge_param_count"),
            r.get("nonKAN_param_count"),
            r.get("mixing_param_count"),
            r.get("actual_backend"),
            r.get("graph_break_count"),
            fmt(f(r, "rollback_max_abs_error")),
            "yes" if int(f(r, "p0_pass", 0)) else "no",
        ]
        for r in p0
    ]
    p1_rows = [
        [
            r["method"],
            r["rows"],
            fmt(r["fwd"]),
            fmt(r["bwd"]),
            fmt(r["bmem"]),
            fmt(r["step"]),
            fmt(r["cold_warm"]),
            r["graph_breaks"],
            r["recompiles"],
            fmt(r["explore_rate"]),
            fmt(r["final_rate"]),
            r["bottleneck"],
        ]
        for r in p1_summary
    ]
    p2_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["ECE"]), fmt(r["step"]), fmt(r["bmem"]), "yes" if int(r["pass"]) else "no"] for r in p2_summary]
    p3_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["ECE"]), fmt(r["step"]), fmt(r["bmem"]), "yes" if int(r["pass"]) else "no"] for r in p3_summary]
    p4_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["ECE"]), fmt(r["step"]), fmt(r["bmem"]), "yes" if int(r["pass"]) else "no"] for r in p4_summary]
    p5_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["ECE"]), fmt(r["step"]), fmt(r["bmem"]), "yes" if int(r["pass"]) else "no"] for r in p5_summary]

    doc = f"""# DG-KAN v5.7 Kernel-Verified Efficient PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v5.7_KernelVerified_EfficientPureKAN_实验计划.md`。核心目标是先确认 profiler 没有把 cold compile / graph break 混入 steady-state，再判断 strict PureKAN primitive 是否存在 MLP-like kernel path 和可接受 accuracy。

## Run Inventory

{md_table(["stage", "rows", "errors"], [
    ["P0 core/profiler", len(p0), sum(1 for r in p0 if r.get("error"))],
    ["P1 efficiency v3", len(p1), sum(1 for r in p1 if r.get("error"))],
    ["P1 cold/warm audit", len(cold), sum(1 for r in cold if r.get("error"))],
    ["P1 memory decomposition", len(mem), sum(1 for r in mem if r.get("error"))],
    ["P2 DWM recipe", len(p2), sum(1 for r in p2 if r.get("error"))],
    ["P3 Rational recipe", len(p3), sum(1 for r in p3 if r.get("error"))],
    ["P4 CP rank/speed", len(p4), sum(1 for r in p4 if r.get("error"))],
    ["P5 joint selection", len(p5), sum(1 for r in p5 if r.get("error"))],
    ["P6-P10 gated", len(p6) + len(p7) + len(p8) + len(p9) + len(p10), sum(1 for r in p6 + p7 + p8 + p9 + p10 if r.get("error"))],
])}

## Code / Config Changes

```text
experiments/run_gafu_v57.py
  Added kernel-verified P0/P1 with cold-vs-warm compile audit,
  memory decomposition, graph-break/recompile fields, and gated DWM/RK/CP recipe probes.

experiments/analyze_gafu_v57.py
  Generates v5.7 gate summaries, failure taxonomy, route_decision.json,
  aggregate_decision.json, SVG diagnostics, and this replay.
```

## P0 Core / Profiler Consistency

{md_table(["primitive", "class", "edge", "nonKAN", "mixing", "backend", "breaks", "rollback", "P0"], p0_rows)}

P0 verdict: pass if each non-reference primitive is core-defined, edge-covered, nonKAN-free, rollback-exact, and has no recorded graph-break issue in the manifest.

## P1 Phase-Separated Efficiency Profiler V3

{md_table(["method", "rows", "fwd", "bwd", "bmem", "step", "cold/warm", "breaks", "recomp", "explore", "final", "bottleneck"], p1_rows)}

P1 exploratory survivors:

```text
{", ".join(p1_exploratory) or "none"}
```

P1 final survivors:

```text
{", ".join(p1_final) or "none"}
```

P1 verdict: v5.7 separates cold step from warm steady-state. Any large cold/warm ratio is now diagnostic rather than silently counted as steady-state.

## P2 DepthwiseMix Kernel / Recipe Repair

{md_table(["dataset", "recipe", "runs", "acc", "gap", "ECE", "step", "bmem", "P2"], p2_rows)}

P2 all-dataset survivors:

```text
{", ".join("|".join(x) for x in p2_surv) or "none"}
```

## P3 RationalKAT Kernel / Recipe Repair

{md_table(["dataset", "recipe", "runs", "acc", "gap", "ECE", "step", "bmem", "P3"], p3_rows)}

P3 all-dataset survivors:

```text
{", ".join("|".join(x) for x in p3_surv) or "none"}
```

## P4 CP-ABRBF Rank / Speed Reference

{md_table(["dataset", "recipe", "runs", "acc", "gap", "ECE", "step", "bmem", "P4"], p4_rows)}

P4 all-dataset survivors:

```text
{", ".join("|".join(x) for x in p4_surv) or "none"}
```

## P5 Joint Task-Efficiency Selection

{md_table(["dataset", "recipe", "runs", "acc", "gap", "ECE", "step", "bmem", "P5"], p5_rows) if p5_rows else "_P5 not run; no P2/P3/P4 all-dataset survivor._"}

P5 all-dataset survivors:

```text
{", ".join("|".join(x) for x in p5_surv) or "none"}
```

## P6-P10 Decision

```text
P6 custom backward/memory: {"run" if p6 and p6[0].get("status") != "not_run" else "not run"}
P7 functional / LightSmooth compatibility: {"run" if p7 and p7[0].get("status") != "not_run" else "not run"}
P8 3-seed selection: {"run" if p8 and p8[0].get("status") != "not_run" else "not run"}
P9/P10 confirm: {"run" if p9 and p9[0].get("status") != "not_run" else "not run"}

Reason: {route["interpretation"]}
```

## Failure Diagnosis

{md_table(["failure", "count"], [[k, v] for k, v in fail_by_type.most_common()])}

Route decision:

```text
{route["case"]}: {route["interpretation"]}
```

## Required Artifacts

Written under `results/v5_7/`:

```text
p0_core_profiler_consistency.csv
p1_phase_efficiency_v3.csv
p1_cold_warm_compile_audit.csv
p1_memory_decomposition.csv
p2_dwm_kernel_recipe.csv
p2_dwm_recipe_heatmap.csv
p3_rational_kernel_recipe.csv
p3_rational_safety.csv
p4_cp_rank_speed.csv
p5_joint_task_efficiency_selection.csv
p6_custom_backward_memory.csv
p7_functional_lightsmooth_compat.csv
p8_candidate_selection3.csv
p9_confirm5.csv
p10_confirm10.csv
failure_table.csv
failure_by_method.csv
failure_by_type.csv
route_decision.json
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.7 status:
  {status}

What improved:
  Profiler correctness is now explicit: cold compile, warm steady-state, memory decomposition,
  graph breaks, and recompiles are audited as first-class artifacts.
  DWM / RationalKAT / CP recipe probes remain gated behind strict task-efficiency checks.

What failed / remains open:
  See the P1-P5 gates above. Functional optimizer / LightSmooth stays gated until
  a primitive survives joint task + efficiency selection.

Conclusion:
  v5.7 keeps the kernel-first rule strict: a functional optimizer cannot rescue a primitive
  that lacks a verified MLP-like forward/backward/memory path.
```
"""

    OUT.write_text(doc, encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as fobj:
        fobj.write(f"\n- v5.7 kernel-verified efficient PureKAN: status={status}; P1 exploratory={len(p1_exploratory)}, P5 survivors={len(p5_surv)}.\n")


if __name__ == "__main__":
    main()
