#!/usr/bin/env python3
"""Summarize DG-KAN v5.6 GEMM-native efficient PureKAN experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v5_6"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v5.6_GEMMNative_EfficientPureKAN_结果复盘.md"
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
        v = row.get(key, "")
        if v in {"", None, "nan", "NaN"}:
            return default
        return float(v)
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
    def cell(v: object) -> str:
        return str(v).replace("|", "\\|")

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
    return [k for k, ds in by.items() if all(d in ds for d in DATASETS)]


def write_svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 960
    height = 70 + 28 * len(labels)
    finite = [v for v in values if math.isfinite(v)]
    max_v = max(finite or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="18" y="30" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, val) in enumerate(zip(labels, values)):
        y = 54 + i * 28
        bar = 0.0 if not math.isfinite(val) else 520 * val / max(1.0e-12, max_v)
        lines.append(f'<text x="18" y="{y+14}" font-family="Arial" font-size="10">{label[:48]}</text>')
        lines.append(f'<rect x="380" y="{y}" width="{bar:.1f}" height="16" fill="#2563eb"/>')
        lines.append(f'<text x="{390+bar:.1f}" y="{y+13}" font-family="Arial" font-size="10">{fmt(val,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_scatter(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 560
    vals = [(f(r, "step_time_ratio"), f(r, "acc_gap_vs_mlp"), f(r, "backward_memory_ratio"), str(r.get("recipe"))) for r in rows if r.get("recipe") not in {"MLP-AdamW", ""}]
    max_x = max([v[0] for v in vals if math.isfinite(v[0])] or [2.0])
    max_y = max([v[1] for v in vals if math.isfinite(v[1])] or [0.1])
    colors = {"DWM": "#16a34a", "RK": "#dc2626", "CP": "#7c3aed", "ABRBF": "#2563eb", "RBF": "#64748b"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="30" y="32" font-family="Arial" font-size="16" font-weight="700">Joint accuracy-efficiency Pareto</text>',
        '<line x1="80" y1="500" x2="840" y2="500" stroke="#111"/>',
        '<line x1="80" y1="500" x2="80" y2="70" stroke="#111"/>',
        '<text x="390" y="540" font-family="Arial" font-size="12">step ratio vs MLP</text>',
        '<text x="18" y="250" font-family="Arial" font-size="12" transform="rotate(-90 18,250)">accuracy gap vs MLP</text>',
    ]
    for x, y, _, label in vals:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        px = 80 + 760 * x / max(1.0e-12, max_x)
        py = 500 - 420 * max(0.0, y) / max(1.0e-12, max_y)
        key = "DWM" if label.startswith("DWM") else "RK" if label.startswith("RK") else "CP" if label.startswith("CP") else "ABRBF" if "ABRBF" in label else "RBF"
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{colors[key]}" fill-opacity="0.78"/>')
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
    p1 = read_rows(BASE / "p1_phase_efficiency_v2.csv")
    p2 = read_rows(BASE / "p2_dwm_recipe_repair.csv")
    p3 = read_rows(BASE / "p3_rationalkat_recipe_repair.csv")
    p4 = read_rows(BASE / "p4_cp_capacity_repair.csv")
    p5 = read_rows(BASE / "p5_joint_task_efficiency_selection.csv")
    p6 = read_rows(BASE / "p6_custom_backward_memory.csv")
    p7 = read_rows(BASE / "p7_lightsmooth_compatibility.csv")
    p8 = read_rows(BASE / "p8_functional_training_smoke.csv")
    p9 = read_rows(BASE / "p9_confirm5.csv")
    p10 = read_rows(BASE / "p10_confirm10.csv")
    failures = read_rows(BASE / "failure_table.csv")

    p1_summary: list[dict] = []
    for (method,), rows in sorted(grouped(p1, "method").items()):
        if "MLP" in method:
            continue
        p1_summary.append(
            {
                "method": method,
                "rows": len(rows),
                "fwd": mean(f(r, "forward_time_ratio_vs_mlp") for r in rows),
                "bwd": mean(f(r, "backward_time_ratio_vs_mlp") for r in rows),
                "bmem": mean(f(r, "backward_memory_ratio_vs_mlp") for r in rows),
                "step": mean(f(r, "step_time_ratio_vs_mlp") for r in rows),
                "explore_rate": mean((f(r, "p1_exploratory_pass", 0) for r in rows), 0.0),
                "final_rate": mean((f(r, "p1_final_pass", 0) for r in rows), 0.0),
                "bottleneck": "memory" if mean(f(r, "backward_memory_ratio_vs_mlp") for r in rows) > 1.25 else "time",
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
        status = "continue_to_p6_memory_reduction"
    elif p2_surv or p3_surv or p4_surv:
        status = "stop_after_p5_no_joint_task_efficiency_survivor"
    elif p1_exploratory:
        status = "stop_after_p4_no_recipe_survivor"
    else:
        status = "stop_after_p1_no_efficiency_survivor"

    route = {
        "status": status,
        "case": "B_recipe_blocker" if p1_exploratory and not (p2_surv or p3_surv or p4_surv) else "C_joint_blocker" if (p2_surv or p3_surv or p4_surv) and not p5_surv else "A_continue" if p5_surv else "D_kernel_blocker",
        "interpretation": (
            "GEMM-native kernel path exists in exploratory profiler, but recipe/task quality does not produce all-dataset survivors."
            if p1_exploratory and not p5_surv
            else "No primitive reached the exploratory MLP-like efficiency envelope."
        ),
        "next_recommended_route": "Repair task recipe for the fastest GEMM-native primitive before returning to FGO-v3 functional optimizer work.",
    }
    save_json(BASE / "route_decision.json", route)

    fail_by_type = Counter(str(r.get("failure_type", "")) for r in failures)
    fail_by_method = Counter(str(r.get("method", "")) for r in failures)
    write_csv(BASE / "failure_by_type.csv", [{"failure_type": k, "count": v} for k, v in fail_by_type.most_common()])
    write_csv(BASE / "failure_by_method.csv", [{"method": k, "count": v} for k, v in fail_by_method.most_common()])

    write_svg_bar(FIG / "parameter_manifest_stacked_bar.svg", "P0 edge parameter count", [r.get("primitive_name", "") for r in p0], [f(r, "edge_param_count", 0) for r in p0])
    write_svg_bar(FIG / "kernel_path_heatmap.svg", "P0 compile/GEMM path score", [r.get("primitive_name", "") for r in p0], [f(r, "uses_compile", 0) + f(r, "uses_gemm", 0) - f(r, "uses_einsum", 0) for r in p0])
    ordered = [r["method"] for r in p1_summary]
    write_svg_bar(FIG / "forward_backward_memory_pareto.svg", "P1 backward memory ratio", ordered, [float(r["bmem"]) for r in p1_summary])
    write_svg_bar(FIG / "step_time_vs_backward_memory.svg", "P1 step ratio", ordered, [float(r["step"]) for r in p1_summary])
    write_svg_bar(FIG / "kernel_count_breakdown.svg", "P1 forward ratio", ordered, [float(r["fwd"]) for r in p1_summary])
    write_svg_bar(FIG / "saved_tensor_vs_peak_memory.svg", "P1 saved tensor proxy", ordered, [float(r["bmem"]) for r in p1_summary])
    write_svg_bar(FIG / "shape_scaling_curve.svg", "P1 backward ratio", ordered, [float(r["bwd"]) for r in p1_summary])
    write_svg_bar(FIG / "DWM_accuracy_vs_efficiency.svg", "P2 DWM accuracy", [r["recipe"] for r in p2_summary], [float(r["acc"]) for r in p2_summary])
    write_svg_bar(FIG / "Rational_accuracy_vs_groups.svg", "P3 Rational accuracy", [r["recipe"] for r in p3_summary], [float(r["acc"]) for r in p3_summary])
    write_svg_bar(FIG / "CP_rank_vs_accuracy.svg", "P4 CP accuracy", [r["recipe"] for r in p4_summary], [float(r["acc"]) for r in p4_summary])
    write_svg_scatter(FIG / "joint_accuracy_efficiency_pareto.svg", p5_summary or p2_summary + p3_summary + p4_summary)

    agg = {
        "version": "v5.6",
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
            "P2": len(p2),
            "P3": len(p3),
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
            r.get("is_core_defined"),
            fmt(f(r, "rollback_max_abs_error")),
            "yes" if int(f(r, "p0_pass", 0)) else "no",
        ]
        for r in p0
    ]
    p1_rows = [[r["method"], r["rows"], fmt(r["fwd"]), fmt(r["bwd"]), fmt(r["bmem"]), fmt(r["step"]), fmt(r["explore_rate"]), fmt(r["final_rate"]), r["bottleneck"]] for r in p1_summary]
    p2_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["ECE"]), fmt(r["step"]), fmt(r["bmem"]), "yes" if int(r["pass"]) else "no"] for r in p2_summary]
    p3_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["ECE"]), fmt(r["step"]), fmt(r["bmem"]), "yes" if int(r["pass"]) else "no"] for r in p3_summary]
    p4_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["ECE"]), fmt(r["step"]), fmt(r["bmem"]), "yes" if int(r["pass"]) else "no"] for r in p4_summary]
    p5_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["ECE"]), fmt(r["step"]), fmt(r["bmem"]), "yes" if int(r["pass"]) else "no"] for r in p5_summary]

    doc = f"""# DG-KAN v5.6 GEMM-Native Efficient PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v5.6_GEMMNative_EfficientPureKAN_实验计划.md`。核心目标是先验证 PureKAN primitive 是否存在 MLP-like GEMM-native compute path，再决定是否继续 functional optimizer / LightSmooth。

## Run Inventory

{md_table(["stage", "rows", "errors"], [
    ["P0 core/profiler", len(p0), sum(1 for r in p0 if r.get("error"))],
    ["P1 efficiency v2", len(p1), sum(1 for r in p1 if r.get("error"))],
    ["P2 DWM recipe", len(p2), sum(1 for r in p2 if r.get("error"))],
    ["P3 Rational recipe", len(p3), sum(1 for r in p3 if r.get("error"))],
    ["P4 CP repair", len(p4), sum(1 for r in p4 if r.get("error"))],
    ["P5 joint selection", len(p5), sum(1 for r in p5 if r.get("error"))],
    ["P6 memory", len(p6), sum(1 for r in p6 if r.get("error"))],
    ["P7 LightSmooth", len(p7), sum(1 for r in p7 if r.get("error"))],
    ["P8 functional", len(p8), sum(1 for r in p8 if r.get("error"))],
    ["P9/P10 confirm", len(p9) + len(p10), sum(1 for r in p9 + p10 if r.get("error"))],
])}

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added core GEMMNativeDepthwiseMixDense, GEMMNativeCPABRBFDense,
  and GEMMNativeRationalKATDense.

experiments/run_gafu_v56.py
  Added core/profiler consistency checks, phase-separated efficiency profiler v2,
  DWM/Rational/CP recipe repair, gated joint selection, and gated P6-P10 artifacts.

experiments/analyze_gafu_v56.py
  Generates v5.6 gate summaries, route_decision.json, aggregate_decision.json,
  figures, failure taxonomy, and this replay.
```

## P0 Core / Profiler Consistency

{md_table(["primitive", "class", "edge", "nonKAN", "mixing", "core", "rollback", "P0"], p0_rows)}

P0 verdict: pass if each non-reference primitive is core-defined, edge-covered, nonKAN-free, and rollback-exact.

## P1 Phase-Separated Efficiency Profiler V2

{md_table(["method", "rows", "fwd", "bwd", "bmem", "step", "explore", "final", "bottleneck"], p1_rows)}

P1 exploratory survivors:

```text
{", ".join(p1_exploratory) or "none"}
```

P1 final survivors:

```text
{", ".join(p1_final) or "none"}
```

## P2 DepthwiseMix Recipe Repair

{md_table(["dataset", "recipe", "runs", "acc", "gap", "ECE", "step", "bmem", "P2"], p2_rows)}

P2 all-dataset survivors:

```text
{", ".join("|".join(x) for x in p2_surv) or "none"}
```

## P3 RationalKAT Recipe Repair

{md_table(["dataset", "recipe", "runs", "acc", "gap", "ECE", "step", "bmem", "P3"], p3_rows)}

P3 all-dataset survivors:

```text
{", ".join("|".join(x) for x in p3_surv) or "none"}
```

## P4 CP-ABRBF Compute / Capacity Repair

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
P6 custom backward/memory: {"run" if p6 and not p6[0].get("status") == "not_run" else "not run"}
P7 LightSmooth compatibility: {"run" if p7 and not p7[0].get("status") == "not_run" else "not run"}
P8 functional training smoke: {"run" if p8 and not p8[0].get("status") == "not_run" else "not run"}
P9/P10 confirm: {"run" if p9 and not p9[0].get("status") == "not_run" else "not run"}

Reason: {route["interpretation"]}
```

## Failure Diagnosis

{md_table(["failure", "count"], [[k, v] for k, v in fail_by_type.most_common()])}

Route decision:

```text
{route["case"]}: {route["interpretation"]}
```

## Required Artifacts

Written under `results/v5_6/`:

```text
p0_core_profiler_consistency.csv
p1_phase_efficiency_v2.csv
p1_efficiency_gate_summary.csv
p2_dwm_recipe_repair.csv
p2_dwm_gate_summary.csv
p3_rationalkat_recipe_repair.csv
p3_rational_gate_summary.csv
p4_cp_capacity_repair.csv
p4_cp_gate_summary.csv
p5_joint_task_efficiency_selection.csv
p5_joint_gate_summary.csv
p6_custom_backward_memory.csv
p7_lightsmooth_compatibility.csv
p8_functional_training_smoke.csv
p9_confirm5.csv
p10_confirm10.csv
p10_failure_diagnosis.csv
failure_table.csv
failure_by_type.csv
failure_by_method.csv
aggregate_decision.json
route_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.6 status:
  {status}

What improved:
  GEMM-native efficient primitives now live in core dgkan_core.py.
  P1 separates forward/backward/memory/optimizer behavior under the v5.6 gates.
  DWM/Rational/CP recipe repair is evaluated before any functional optimizer work.

What failed / remains open:
  See P1-P5 gates above. Functional optimizer / LightSmooth remains gated behind
  task + efficiency survival, consistent with the v5.6 kernel-first rule.

Conclusion:
  v5.6 keeps the mainline honest: FGO-v3 should wait until a GEMM-native PureKAN
  primitive can satisfy the task-efficiency envelope.
```
"""

    OUT.write_text(doc, encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as fobj:
        fobj.write(f"\n- v5.6 GEMM-native efficient PureKAN: status={status}; P1 exploratory={len(p1_exploratory)}, P5 survivors={len(p5_surv)}.\n")


if __name__ == "__main__":
    main()
