#!/usr/bin/env python3
"""Summarize DG-KAN v5.5 kernel-first efficient PureKAN experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v5_5"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v5.5_KernelFirst_EfficientPureKAN_结果复盘.md"
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
    width = 940
    height = 70 + 30 * len(labels)
    finite = [v for v in values if math.isfinite(v)]
    max_v = max(finite or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="18" y="30" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, val) in enumerate(zip(labels, values)):
        y = 58 + i * 30
        bar = 0.0 if not math.isfinite(val) else 520 * val / max(1.0e-12, max_v)
        lines.append(f'<text x="18" y="{y+15}" font-family="Arial" font-size="10">{label[:45]}</text>')
        lines.append(f'<rect x="360" y="{y}" width="{bar:.1f}" height="18" fill="#2563eb"/>')
        lines.append(f'<text x="{370+bar:.1f}" y="{y+14}" font-family="Arial" font-size="10">{fmt(val,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_scatter(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 560
    vals = [(f(r, "step_time_ratio"), f(r, "gap_vs_mlp"), f(r, "backward_memory_ratio"), str(r.get("method"))) for r in rows if str(r.get("method")) != "MLP-AdamW"]
    max_x = max([v[0] for v in vals if math.isfinite(v[0])] or [2.0])
    max_y = max([v[1] for v in vals if math.isfinite(v[1])] or [0.1])
    colors = {
        "RBFOnly": "#64748b",
        "ABRBF": "#2563eb",
        "CP": "#7c3aed",
        "DWM": "#16a34a",
        "RK": "#dc2626",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="30" y="32" font-family="Arial" font-size="16" font-weight="700">P5 accuracy-efficiency Pareto</text>',
        '<line x1="80" y1="500" x2="840" y2="500" stroke="#111"/>',
        '<line x1="80" y1="500" x2="80" y2="70" stroke="#111"/>',
        '<text x="400" y="540" font-family="Arial" font-size="12">step time ratio vs MLP</text>',
        '<text x="18" y="250" font-family="Arial" font-size="12" transform="rotate(-90 18,250)">accuracy gap vs MLP</text>',
    ]
    for x, y, mem, label in vals:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        px = 80 + 760 * x / max(1.0e-12, max_x)
        py = 500 - 420 * y / max(1.0e-12, max_y)
        key = "CP" if "CP" in label else "DWM" if "DWM" in label else "RK" if "RK" in label else "ABRBF" if "ABRBF" in label else "RBFOnly"
        radius = 4 + min(12, max(0, mem if math.isfinite(mem) else 1.0) * 2)
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{colors[key]}" fill-opacity="0.75"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p0 = read_rows(BASE / "p0_core_runner_consistency.csv")
    p1 = read_rows(BASE / "p1_phase_efficiency_profiler.csv")
    p2 = read_rows(BASE / "p2_depthwise_kernel_repair.csv")
    p3 = read_rows(BASE / "p3_cp_compute_graph_repair.csv")
    p4 = read_rows(BASE / "p4_rationalkat_kernel_recipe.csv")
    p5 = read_rows(BASE / "p5_efficient_accuracy_frontier.csv")
    p6 = read_rows(BASE / "p6_custom_backward_true_memory.csv")
    p7 = read_rows(BASE / "p7_functional_lightsmooth_smoke.csv")
    p8 = read_rows(BASE / "p8_candidate_selection3.csv")
    p9 = read_rows(BASE / "p9_confirm5.csv")
    failures = read_rows(BASE / "failure_table.csv") or read_rows(BASE / "p9_failure_diagnosis.csv")

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
                "bmem": mean(f(r, "backward_peak_ratio_vs_mlp") for r in rows),
                "step": mean(f(r, "step_time_ratio_vs_mlp") for r in rows),
                "early_pass_rate": mean((f(r, "p1_early_pass", 0) for r in rows), 0.0),
                "final_pass_rate": mean((f(r, "p1_final_pass", 0) for r in rows), 0.0),
                "bottleneck": "memory" if mean(f(r, "backward_peak_ratio_vs_mlp") for r in rows) > 1.25 else "time",
            }
        )
    write_csv(BASE / "p1_efficiency_gate_summary.csv", p1_summary)

    p2_summary: list[dict] = []
    for (variant,), rows in sorted(grouped(p2, "variant").items()):
        good = [r for r in rows if r.get("status") != "not_available"]
        p2_summary.append(
            {
                "variant": variant,
                "rows": len(rows),
                "step": mean(f(r, "step_ratio_vs_mlp") for r in good),
                "bmem": mean(f(r, "bmem_ratio_vs_mlp") for r in good),
                "pass_rate": mean((f(r, "p2_pass", 0) for r in good), 0.0),
                "P2": int(bool(good) and all(int(f(r, "p2_pass", 0)) == 1 for r in good)),
            }
        )
    write_csv(BASE / "p2_depthwise_gate_summary.csv", p2_summary)

    p3_summary: list[dict] = []
    for (variant,), rows in sorted(grouped(p3, "variant").items()):
        p3_summary.append(
            {
                "variant": variant,
                "rows": len(rows),
                "best_step": min([f(r, "step_ratio_vs_mlp", math.inf) for r in rows] or [math.nan]),
                "best_bmem": min([f(r, "bmem_ratio_vs_mlp", math.inf) for r in rows] or [math.nan]),
                "monotonic": int(mean((f(r, "rank_speed_monotonic", 0) for r in rows), 0.0) >= 1.0),
                "rank16_step": mean(f(r, "step_ratio_vs_mlp") for r in rows if int(f(r, "rank", 0)) == 16),
                "P3": int(any(int(f(r, "p3_pass", 0)) == 1 for r in rows)),
            }
        )
    write_csv(BASE / "p3_cp_gate_summary.csv", p3_summary)

    p4_kernel_summary: list[dict] = []
    for (variant,), rows in sorted(grouped([r for r in p4 if r.get("dataset") == "synthetic"], "variant").items()):
        usable = [r for r in rows if r.get("status") != "not_available"]
        p4_kernel_summary.append(
            {
                "variant": variant,
                "step": mean(f(r, "step_ratio_vs_mlp") for r in usable),
                "bmem": mean(f(r, "bmem_ratio_vs_mlp") for r in usable),
                "kernel_pass": int(any(int(f(r, "p4_kernel_pass", 0)) == 1 for r in usable)),
            }
        )
    write_csv(BASE / "p4_rational_kernel_gate_summary.csv", p4_kernel_summary)

    p4_recipe_summary: list[dict] = []
    for (dataset, recipe), rows in sorted(grouped([r for r in p4 if r.get("dataset") in DATASETS], "dataset", "recipe").items()):
        p4_recipe_summary.append({"dataset": dataset, "recipe": recipe, "acc": mean(f(r, "test_acc") for r in rows), "ECE": mean(f(r, "ECE") for r in rows)})
    write_csv(BASE / "p4_rational_recipe_summary.csv", p4_recipe_summary)

    p5_summary: list[dict] = []
    by_dataset_method = grouped(p5, "dataset", "method")
    for (dataset, method), rows in sorted(by_dataset_method.items()):
        p5_summary.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rows),
                "acc": mean(f(r, "test_acc") for r in rows),
                "std": std(f(r, "test_acc") for r in rows),
                "gap": mean(f(r, "acc_gap_vs_mlp") for r in rows),
                "ECE": mean(f(r, "ECE") for r in rows),
                "step": mean(f(r, "step_time_ratio") for r in rows),
                "bmem": mean(f(r, "backward_memory_ratio") for r in rows),
                "P5": int(all(int(f(r, "p5_pass", 0)) == 1 for r in rows) and method != "MLP-AdamW"),
            }
        )
    write_csv(BASE / "p5_accuracy_gate_summary.csv", p5_summary)

    p2_surv = [r["variant"] for r in p2_summary if int(r["P2"]) == 1]
    p3_surv = [r["variant"] for r in p3_summary if int(r["P3"]) == 1]
    p4_kernel_surv = [r["variant"] for r in p4_kernel_summary if int(r["kernel_pass"]) == 1]
    p5_surv = all_dataset_survivors(p5, ["method"], "p5_pass")

    if p5_surv:
        status = "continue_to_p6_true_memory_audit"
    elif p2_surv or p3_surv or p4_kernel_surv:
        status = "stop_after_p5_no_joint_accuracy_efficiency_survivor"
    else:
        status = "stop_after_p4_no_exploratory_efficiency_survivor"

    route = {
        "status": status,
        "case": "mixed_BC" if (p3_surv or p4_kernel_surv) else "D",
        "interpretation": (
            "Kernel repair exposes exploratory efficiency pockets, but task+efficiency does not survive P5."
            if p2_surv or p3_surv or p4_kernel_surv
            else "No current primitive family reached exploratory efficiency."
        ),
        "next_recommended_route": "Repair task recipe for the efficient kernel candidates, especially compiled DWM/RK, before functional optimizer work.",
    }
    save_json(BASE / "route_decision.json", route)

    fail_by_type = Counter(str(r.get("failure_type", "")) for r in failures)
    fail_by_method = Counter(str(r.get("method", "")) for r in failures)
    write_csv(BASE / "failure_by_type.csv", [{"failure_type": k, "count": v} for k, v in fail_by_type.most_common()])
    write_csv(BASE / "failure_by_method.csv", [{"method": k, "count": v} for k, v in fail_by_method.most_common()])

    # Required visual dashboard.
    ordered = [r["method"] for r in p1_summary]
    write_svg_bar(FIG / "p1_forward_ratio.svg", "P1 forward ratio by primitive", ordered, [float(r["fwd"]) for r in p1_summary])
    write_svg_bar(FIG / "p1_backward_ratio.svg", "P1 backward ratio by primitive", ordered, [float(r["bwd"]) for r in p1_summary])
    write_svg_bar(FIG / "p1_step_ratio.svg", "P1 step ratio by primitive", ordered, [float(r["step"]) for r in p1_summary])
    write_svg_bar(FIG / "p1_backward_memory_ratio.svg", "P1 backward memory ratio by primitive", ordered, [float(r["bmem"]) for r in p1_summary])
    write_svg_bar(FIG / "p2_dwm_step_ratio.svg", "P2 DepthwiseMix step ratio", [r["variant"] for r in p2_summary], [float(r["step"]) for r in p2_summary])
    write_svg_bar(FIG / "p3_cp_rank16_step_ratio.svg", "P3 CP rank16 step ratio", [r["variant"] for r in p3_summary], [float(r["rank16_step"]) for r in p3_summary])
    write_svg_bar(FIG / "p4_rational_step_ratio.svg", "P4 RationalKAT step ratio", [r["variant"] for r in p4_kernel_summary], [float(r["step"]) for r in p4_kernel_summary])
    write_svg_scatter(FIG / "p5_accuracy_efficiency_pareto.svg", p5_summary)

    agg = {
        "version": "v5.5",
        "status": status,
        "p2_depthwise_survivors": p2_surv,
        "p3_cp_survivors": p3_surv,
        "p4_rational_kernel_survivors": p4_kernel_surv,
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
        },
    }
    save_json(BASE / "aggregate_decision.json", agg)

    p0_rows = [
        [r.get("primitive_name"), r.get("class_name"), r.get("param_count_edge"), r.get("param_count_nonkan"), r.get("param_count_mixing"), fmt(f(r, "rollback_max_abs_error")), "yes" if int(f(r, "p0_pass", 0)) else "no"]
        for r in p0
    ]
    p1_rows = [[r["method"], r["rows"], fmt(r["fwd"]), fmt(r["bwd"]), fmt(r["bmem"]), fmt(r["step"]), fmt(r["early_pass_rate"]), fmt(r["final_pass_rate"]), r["bottleneck"]] for r in p1_summary]
    p2_rows = [[r["variant"], r["rows"], fmt(r["step"]), fmt(r["bmem"]), fmt(r["pass_rate"]), "yes" if int(r["P2"]) else "no"] for r in p2_summary]
    p3_rows = [[r["variant"], r["rows"], fmt(r["best_step"]), fmt(r["best_bmem"]), r["monotonic"], fmt(r["rank16_step"]), "yes" if int(r["P3"]) else "no"] for r in p3_summary]
    p4_rows = [[r["variant"], fmt(r["step"]), fmt(r["bmem"]), "yes" if int(r["kernel_pass"]) else "no"] for r in p4_kernel_summary]
    p4_recipe_rows = [[r["dataset"], r["recipe"], fmt(r["acc"]), fmt(r["ECE"])] for r in p4_recipe_summary]
    p5_rows = [[r["dataset"], r["method"], r["runs"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["ECE"]), fmt(r["step"]), fmt(r["bmem"]), "yes" if int(r["P5"]) else "no"] for r in p5_summary]

    doc = f"""# DG-KAN v5.5 Kernel-First Efficient PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v5.5_KernelFirst_EfficientPureKAN_实验计划.md`。核心目标是先回答：当前 PureKAN edge primitive 是否存在真正接近 MLP 的 kernel / compute path，再决定是否继续 functional optimizer / LightSmooth。

## Run Inventory

{md_table(["stage", "rows", "errors"], [
    ["P0 core consistency", len(p0), sum(1 for r in p0 if r.get("error"))],
    ["P1 phase efficiency", len(p1), sum(1 for r in p1 if r.get("error"))],
    ["P2 Depthwise repair", len(p2), sum(1 for r in p2 if r.get("error"))],
    ["P3 CP repair", len(p3), sum(1 for r in p3 if r.get("error"))],
    ["P4 RationalKAT", len(p4), sum(1 for r in p4 if r.get("error"))],
    ["P5 accuracy frontier", len(p5), sum(1 for r in p5 if r.get("error"))],
    ["P6 true memory", len(p6), sum(1 for r in p6 if r.get("error"))],
    ["P7 LightSmooth", len(p7), sum(1 for r in p7 if r.get("error"))],
    ["P8 selection", len(p8), sum(1 for r in p8 if r.get("error"))],
    ["P9 confirm/failure", len(p9), sum(1 for r in p9 if r.get("error"))],
])}

## Code / Config Changes

```text
experiments/run_gafu_v55.py
  Added phase-separated efficiency profiler and kernel repair probes for DWM / CP / RationalKAT.
  Added compact P5 efficient accuracy frontier and gated P6-P9 outputs.

experiments/analyze_gafu_v55.py
  Generates v5.5 gate summaries, route_decision.json, aggregate_decision.json, figures, and this replay.
```

## P0 Core / Runner Consistency

{md_table(["primitive", "class", "edge", "nonKAN", "mixing", "rollback", "P0"], p0_rows)}

P0 verdict: pass. All PureKAN candidates keep trainable parameters inside the audited edge system; MLP is retained only as reference.

## P1 Phase-Separated Efficiency Profiler

{md_table(["method", "rows", "fwd", "bwd", "bmem", "step", "early", "final", "bottleneck"], p1_rows)}

P1 survivors:

```text
{", ".join([r["method"] for r in p1_summary if float(r["early_pass_rate"]) > 0.0]) or "none"}
```

Observation:

```text
Several cleaned paths show exploratory speed pockets, but no primitive satisfies the final MLP-like envelope.
Backward peak memory remains the broadest blocker.
```

## P2 DepthwiseMix Kernel Repair

{md_table(["variant", "rows", "step", "bmem", "pass rate", "P2"], p2_rows)}

P2 survivors:

```text
{", ".join(p2_surv) or "none"}
```

Interpretation: compiled/vectorized DepthwiseMix exposes a real kernel-path gain, but its trainable accuracy path still needs P5 confirmation.

## P3 CP-ABRBF Compute Graph Repair

{md_table(["variant", "rows", "best step", "best bmem", "monotonic", "rank16 step", "P3"], p3_rows)}

P3 survivors:

```text
{", ".join(p3_surv) or "none"}
```

Interpretation: two-stage/compiled CP repairs the rank-speed behavior enough to reach exploratory gate, but memory and task quality remain joint blockers.

## P4 RationalKAT Kernel / Recipe Repair

Kernel audit:

{md_table(["variant", "step", "bmem", "kernel"], p4_rows)}

Recipe audit:

{md_table(["dataset", "recipe", "acc", "ECE"], p4_recipe_rows)}

P4 kernel survivors:

```text
{", ".join(p4_kernel_surv) or "none"}
```

P4 verdict: compiled RationalKAT is the best kernel signal, but compact recipes remain far below MLP on task quality, especially KMNIST.

## P5 Efficient Primitive Accuracy Frontier

{md_table(["dataset", "method", "runs", "acc", "gap", "ECE", "step", "bmem", "P5"], p5_rows)}

P5 all-dataset survivors:

```text
{", ".join("|".join(x) for x in p5_surv) or "none"}
```

P5 verdict:

```text
No method passes task + efficiency across all datasets.
CP-1-two-stage-r16 is the best task signal among efficient candidates on MNIST/Fashion,
but KMNIST remains below the required accuracy frontier.
DWM and RK kernel paths are promising for speed, but not yet for accuracy.
```

## P6-P9 Decision

```text
P6 true memory audit: not run; P5 produced no all-dataset efficient accuracy survivor.
P7 functional / LightSmooth smoke: not run.
P8 3-seed joint candidate selection: not run.
P9 confirm: not run.
```

## Failure Diagnosis

{md_table(["failure", "count"], [[k, v] for k, v in fail_by_type.most_common()])}

Route decision:

```text
{route["case"]}: {route["interpretation"]}
```

## Required Artifacts

Written under `results/v5_5/`:

```text
p0_core_runner_consistency.csv
p1_phase_efficiency_profiler.csv
p1_efficiency_gate_summary.csv
p2_depthwise_kernel_repair.csv
p2_depthwise_gate_summary.csv
p3_cp_compute_graph_repair.csv
p3_cp_gate_summary.csv
p4_rationalkat_kernel_recipe.csv
p4_rational_kernel_gate_summary.csv
p4_rational_recipe_summary.csv
p5_efficient_accuracy_frontier.csv
p5_accuracy_gate_summary.csv
p6_custom_backward_true_memory.csv
p7_functional_lightsmooth_smoke.csv
p8_candidate_selection3.csv
p9_confirm5.csv
p9_confirm10.csv
failure_table.csv
failure_by_type.csv
failure_by_method.csv
aggregate_decision.json
route_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.5 status:
  {status}

What improved:
  Kernel-first profiling now separates forward, backward, optimizer and memory phases.
  DWM compiled and CP two-stage/compiled expose exploratory efficiency improvements.
  RationalKAT compiled is the clearest memory/kernel signal.

What failed:
  No primitive passes the final efficiency envelope.
  P5 produced no task + efficiency all-dataset survivor.
  CP keeps the best accuracy signal but is not final efficient system.
  DWM/RK are better efficiency primitives but need architecture/recipe repair.

Conclusion:
  v5.5 supports the kernel-first diagnosis.
  The next step should repair the task recipe for efficient DWM/RK-style primitives
  or move to a new GEMM-native edge design before returning to functional optimizer work.
```
"""

    OUT.write_text(doc, encoding="utf-8")
    log_entry = f"\n- v5.5 kernel-first efficient PureKAN: status={status}; P2={len(p2_surv)} DWM survivors, P3={len(p3_surv)} CP survivors, P4={len(p4_kernel_surv)} Rational kernel survivors, P5 survivors={len(p5_surv)}.\n"
    with LOG.open("a", encoding="utf-8") as fobj:
        fobj.write(log_entry)


if __name__ == "__main__":
    main()
