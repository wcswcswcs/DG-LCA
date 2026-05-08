#!/usr/bin/env python3
"""Summarize DG-KAN v5.8 efficient primitive redesign experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v5_8"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v5.8_EfficientPrimitive_Redesign_结果复盘.md"
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
    return sorted(key for key, ds in by.items() if all(d in ds for d in DATASETS))


def any_pass(rows: Sequence[dict], pass_field: str) -> list[str]:
    return sorted(
        {
            str(row.get("recipe", row.get("method", "")))
            for row in rows
            if not row.get("error") and row.get("status") != "not_run" and int(f(row, pass_field, 0)) == 1
        }
    )


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
        lines.append(f'<text x="18" y="{y+14}" font-family="Arial" font-size="10">{label[:54]}</text>')
        lines.append(f'<rect x="420" y="{y}" width="{bar:.1f}" height="16" fill="#2563eb"/>')
        lines.append(f'<text x="{430+bar:.1f}" y="{y+13}" font-family="Arial" font-size="10">{fmt(value,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_scatter(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = [
        (
            f(r, "step_time_ratio", f(r, "step_time_ratio_vs_mlp")),
            f(r, "acc_gap_vs_mlp", 0.0),
            f(r, "backward_memory_ratio", f(r, "backward_memory_ratio_vs_mlp")),
            str(r.get("recipe", r.get("method", ""))),
        )
        for r in rows
        if str(r.get("recipe", r.get("method", ""))) not in {"MLP-AdamW", "MLP-reference", ""}
        and r.get("status") != "not_run"
        and not r.get("error")
    ]
    width, height = 900, 560
    max_x = max([v[0] for v in vals if math.isfinite(v[0])] or [2.0])
    max_y = max([v[1] for v in vals if math.isfinite(v[1])] or [0.1])
    colors = {"DWM2": "#16a34a", "RK2": "#dc2626", "LUT": "#0891b2", "CP": "#7c3aed"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="30" y="32" font-family="Arial" font-size="16" font-weight="700">Joint Accuracy-Efficiency View</text>',
        '<line x1="80" y1="500" x2="840" y2="500" stroke="#111"/>',
        '<line x1="80" y1="500" x2="80" y2="70" stroke="#111"/>',
        '<text x="386" y="540" font-family="Arial" font-size="12">step ratio vs MLP</text>',
        '<text x="18" y="278" font-family="Arial" font-size="12" transform="rotate(-90 18,278)">accuracy gap vs MLP</text>',
    ]
    for x, y, mem, label in vals:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        px = 80 + 760 * x / max(1.0e-12, max_x)
        py = 500 - 420 * max(0.0, y) / max(1.0e-12, max_y)
        key = "DWM2" if label.startswith("DWM2") else "RK2" if label.startswith("RK2") else "LUT" if label.startswith("LUT") else "CP"
        radius = 4 + min(8, max(0.0, mem if math.isfinite(mem) else 1.0))
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{colors[key]}" fill-opacity="0.75"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_memory_time(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = [
        (f(r, "backward_memory_ratio_vs_mlp"), f(r, "backward_time_ratio_vs_mlp"), str(r.get("method", "")))
        for r in rows
        if str(r.get("method", "")) != "MLP-reference" and not r.get("error")
    ]
    width, height = 900, 560
    max_x = max([v[0] for v in vals if math.isfinite(v[0])] or [1.5])
    max_y = max([v[1] for v in vals if math.isfinite(v[1])] or [2.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="30" y="32" font-family="Arial" font-size="16" font-weight="700">Memory-Time Frontier</text>',
        '<line x1="80" y1="500" x2="840" y2="500" stroke="#111"/>',
        '<line x1="80" y1="500" x2="80" y2="70" stroke="#111"/>',
        '<rect x="80" y="70" width="0" height="0" fill="none"/>',
        '<text x="356" y="540" font-family="Arial" font-size="12">backward memory ratio vs MLP</text>',
        '<text x="18" y="270" font-family="Arial" font-size="12" transform="rotate(-90 18,270)">backward time ratio vs MLP</text>',
    ]
    target_x = 80 + 760 * min(0.8, max_x) / max(1.0e-12, max_x)
    target_y = 500 - 420 * min(1.4, max_y) / max(1.0e-12, max_y)
    lines.append(f'<rect x="80" y="{target_y:.1f}" width="{max(0.0, target_x-80):.1f}" height="{500-target_y:.1f}" fill="#dcfce7" opacity="0.65"/>')
    for x, y, label in vals:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        px = 80 + 760 * x / max(1.0e-12, max_x)
        py = 500 - 420 * y / max(1.0e-12, max_y)
        color = "#16a34a" if label.startswith("DWM2") else "#dc2626" if "Rational" in label else "#0891b2" if "LUT" in label else "#7c3aed" if "CP" in label else "#64748b"
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" fill-opacity="0.8"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_placeholder_svg(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 220
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#fff"/>',
                f'<text x="28" y="42" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
                f'<text x="28" y="86" font-family="Arial" font-size="13" fill="#334155">{message}</text>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def recipe_summary(rows: Sequence[dict], pass_field: str) -> list[dict]:
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


def gate_files_exist() -> list[tuple[str, Path]]:
    return [
        ("p0_core_manifest.csv", BASE / "p0_core_manifest.csv"),
        ("p1_efficiency_decomposition.csv", BASE / "p1_efficiency_decomposition.csv"),
        ("p1_component_timing.csv", BASE / "p1_component_timing.csv"),
        ("p1_memory_decomposition.csv", BASE / "p1_memory_decomposition.csv"),
        ("p2_dwm_v2_recipe.csv", BASE / "p2_dwm_v2_recipe.csv"),
        ("p3_rationalkat_v2_recipe.csv", BASE / "p3_rationalkat_v2_recipe.csv"),
        ("p4_lutkan_feasibility.csv", BASE / "p4_lutkan_feasibility.csv"),
        ("p5_joint_task_efficiency.csv", BASE / "p5_joint_task_efficiency.csv"),
        ("p6_custom_backward_audit.csv", BASE / "p6_custom_backward_audit.csv"),
        ("p7_lightsmooth_compatibility.csv", BASE / "p7_lightsmooth_compatibility.csv"),
        ("p8_functional_training_smoke.csv", BASE / "p8_functional_training_smoke.csv"),
        ("p9_confirm5.csv", BASE / "p9_confirm5.csv"),
        ("p10_confirm10.csv", BASE / "p10_confirm10.csv"),
    ]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p0 = read_rows(BASE / "p0_core_manifest.csv")
    p1 = read_rows(BASE / "p1_efficiency_decomposition.csv")
    p2 = read_rows(BASE / "p2_dwm_v2_recipe.csv")
    p3 = read_rows(BASE / "p3_rationalkat_v2_recipe.csv")
    p4 = read_rows(BASE / "p4_lutkan_feasibility.csv")
    p5 = read_rows(BASE / "p5_joint_task_efficiency.csv")
    p6 = read_rows(BASE / "p6_custom_backward_audit.csv")
    p7 = read_rows(BASE / "p7_lightsmooth_compatibility.csv")
    p8 = read_rows(BASE / "p8_functional_training_smoke.csv")
    p9 = read_rows(BASE / "p9_confirm5.csv")
    p10 = read_rows(BASE / "p10_confirm10.csv")
    failures = read_rows(BASE / "failure_table.csv")

    p1_summary: list[dict] = []
    for (method,), rs in sorted(grouped(p1, "method").items()):
        if method == "MLP-reference":
            continue
        fwd = mean(f(r, "forward_time_ratio_vs_mlp") for r in rs)
        bwd = mean(f(r, "backward_time_ratio_vs_mlp") for r in rs)
        bmem = mean(f(r, "backward_memory_ratio_vs_mlp") for r in rs)
        step = mean(f(r, "step_time_ratio_vs_mlp") for r in rs)
        if bmem > 1.25:
            bottleneck = "memory"
        elif fwd > 2.0:
            bottleneck = "forward"
        elif bwd > 2.0:
            bottleneck = "backward"
        else:
            bottleneck = "none"
        p1_summary.append(
            {
                "method": method,
                "primitive_family": rs[0].get("primitive_family", ""),
                "rows": len(rs),
                "fwd": fwd,
                "bwd": bwd,
                "bmem": bmem,
                "step": step,
                "explore_rate": mean(f(r, "p1_exploratory_pass", 0) for r in rs),
                "final_rate": mean(f(r, "p1_final_pass", 0) for r in rs),
                "bottleneck": bottleneck,
            }
        )
    write_csv(BASE / "p1_efficiency_gate_summary.csv", p1_summary)

    p2_summary = recipe_summary(p2, "p2_pass")
    p3_summary = recipe_summary(p3, "p3_pass")
    p4_summary = recipe_summary(p4, "p4_pass")
    write_csv(BASE / "p2_dwm_v2_gate_summary.csv", p2_summary)
    write_csv(BASE / "p3_rationalkat_v2_gate_summary.csv", p3_summary)
    write_csv(BASE / "p4_lutkan_gate_summary.csv", p4_summary)

    p5_summary = [{"recipe": r.get("recipe", ""), "status": r.get("status", ""), "p5_pass": r.get("p5_pass", 0), "reason": r.get("reason", "")} for r in p5]
    write_csv(BASE / "p5_joint_gate_summary.csv", p5_summary)

    by_primitive = Counter(str(r.get("primitive", "")) for r in failures)
    by_dataset = Counter(str(r.get("dataset", "")) for r in failures)
    by_gate = Counter(str(r.get("gate", r.get("stage", ""))) for r in failures)
    write_csv(BASE / "failure_by_primitive.csv", [{"primitive": k, "count": v} for k, v in sorted(by_primitive.items())])
    write_csv(BASE / "failure_by_dataset.csv", [{"dataset": k, "count": v} for k, v in sorted(by_dataset.items())])
    write_csv(BASE / "failure_by_gate.csv", [{"gate": k, "count": v} for k, v in sorted(by_gate.items())])

    p1_exploratory = [str(r["method"]) for r in p1_summary if float(r["explore_rate"]) > 0.0]
    p1_final = [str(r["method"]) for r in p1_summary if float(r["final_rate"]) > 0.0]
    p2_surv = [s[0] for s in all_dataset_survivors(p2, ["recipe"], "p2_pass")]
    p3_surv = [s[0] for s in all_dataset_survivors(p3, ["recipe"], "p3_pass")]
    p4_surv = [s[0] for s in all_dataset_survivors(p4, ["recipe"], "p4_pass")]
    p5_surv = any_pass(p5, "p5_pass")

    if p5_surv:
        status = "continue_to_p6_custom_backward"
    elif p2_surv or p3_surv or p4_surv:
        status = "stop_after_p5_no_joint_survivor"
    elif p1_exploratory:
        status = "stop_after_p4_no_recipe_survivor"
    else:
        status = "stop_after_p1_no_efficiency_survivor"

    if any(r.startswith("DWM2") for r in p5_surv):
        route_case = "A_DWM_v2_survivor"
    elif any(r.startswith("RK2") for r in p5_surv):
        route_case = "B_RationalKAT_survivor"
    elif any(r.startswith("LUT") for r in p5_surv):
        route_case = "C_LUTKAN_survivor"
    elif any(r.startswith("CP") for r in p5_surv) or (p1_exploratory and all("CP" in r for r in p1_exploratory)):
        route_case = "D_CP_reference_only"
    else:
        route_case = "E_no_efficient_primitive"

    route = {
        "version": "v5.8",
        "status": status,
        "case": route_case,
        "p1_exploratory_survivors": p1_exploratory,
        "p1_final_survivors": p1_final,
        "p2_survivors": p2_surv,
        "p3_survivors": p3_surv,
        "p4_survivors": p4_surv,
        "p5_survivors": p5_surv,
    }
    aggregate = {
        "version": "v5.8",
        "status": status,
        "route_case": route_case,
        "run_inventory": {
            "P0 core manifest": len(p0),
            "P1 efficiency": len(p1),
            "P2 DWM-v2 recipe": len(p2),
            "P3 RationalKAT-v2 recipe": len(p3),
            "P4 LUTKAN": len(p4),
            "P5 joint selection": len(p5),
            "P6-P10 gated": len(p6) + len(p7) + len(p8) + len(p9) + len(p10),
        },
        "errors": {
            "P0": sum(1 for r in p0 if r.get("error")),
            "P1": sum(1 for r in p1 if r.get("error")),
            "P2": sum(1 for r in p2 if r.get("error")),
            "P3": sum(1 for r in p3 if r.get("error")),
            "P4": sum(1 for r in p4 if r.get("error")),
        },
    }
    save_json(BASE / "route_decision.json", route)
    save_json(BASE / "aggregate_decision.json", aggregate)

    write_svg_bar(FIG / "p1_step_ratio.svg", "P1 Step Ratio vs MLP", [r["method"] for r in p1_summary], [float(r["step"]) for r in p1_summary])
    write_svg_bar(FIG / "p1_memory_ratio.svg", "P1 Backward Memory Ratio vs MLP", [r["method"] for r in p1_summary], [float(r["bmem"]) for r in p1_summary])
    write_svg_bar(
        FIG / "p1_cold_warm_timing.svg",
        "P1 Cold/Warm Ratio",
        [r["method"] for r in p1_summary],
        [mean(f(row, "cold_warm_ratio") for row in p1 if row.get("method") == r["method"]) for r in p1_summary],
    )
    write_svg_memory_time(FIG / "p1_memory_time_frontier.svg", p1)
    write_svg_scatter(FIG / "joint_accuracy_efficiency.svg", p2 + p3 + p4)
    write_placeholder_svg(FIG / "p2_dwm_recipe_heatmap.svg", "DWM-v2 Recipe Heatmap", "P2 was gated because P1 produced no DWM-v2 exploratory efficiency survivor.")
    write_placeholder_svg(FIG / "p3_rational_recipe_heatmap.svg", "RationalKAT-v2 Recipe Heatmap", "P3 was gated because P1 produced no RationalKAT-v2 exploratory efficiency survivor.")
    write_placeholder_svg(FIG / "p4_lut_recipe_heatmap.svg", "LUTKAN Recipe Heatmap", "P4 was gated because P1 produced no LUTKAN exploratory efficiency survivor.")
    write_placeholder_svg(FIG / "classwise_failure_heatmap.svg", "Classwise Failure Heatmap", "Task recipes were not run after P1, so no classwise task failures were produced.")
    write_placeholder_svg(FIG / "geometry_efficiency.svg", "Geometry-Efficiency Plot", "Functional / geometry stages were gated behind primitive efficiency survival.")

    run_rows = [
        ["P0 core manifest", len(p0), sum(1 for r in p0 if r.get("error"))],
        ["P1 efficiency", len(p1), sum(1 for r in p1 if r.get("error"))],
        ["P2 DWM-v2 recipe", len(p2), sum(1 for r in p2 if r.get("error"))],
        ["P3 RationalKAT-v2 recipe", len(p3), sum(1 for r in p3 if r.get("error"))],
        ["P4 LUTKAN feasibility", len(p4), sum(1 for r in p4 if r.get("error"))],
        ["P5 joint selection", len(p5), sum(1 for r in p5 if r.get("error"))],
        ["P6-P10 gated", len(p6) + len(p7) + len(p8) + len(p9) + len(p10), 0],
    ]

    p0_rows = [
        [
            r.get("primitive"),
            r.get("primitive_class"),
            r.get("edge_param_count"),
            r.get("nonKAN_param_count"),
            r.get("mixing_param_count"),
            fmt(f(r, "rollback_error"), 4),
            "yes" if int(f(r, "p0_pass", 0)) == 1 else "no",
        ]
        for r in p0
    ]
    p1_rows = [
        [r["method"], r["rows"], fmt(float(r["fwd"])), fmt(float(r["bwd"])), fmt(float(r["bmem"])), fmt(float(r["step"])), fmt(float(r["explore_rate"])), fmt(float(r["final_rate"])), r["bottleneck"]]
        for r in p1_summary
    ]
    p2_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(float(r["acc"])), fmt(float(r["gap"])), fmt(float(r["ECE"])), fmt(float(r["step"])), fmt(float(r["bmem"])), "yes" if int(r["pass"]) else "no"] for r in p2_summary]
    p3_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(float(r["acc"])), fmt(float(r["gap"])), fmt(float(r["ECE"])), fmt(float(r["step"])), fmt(float(r["bmem"])), "yes" if int(r["pass"]) else "no"] for r in p3_summary]
    p4_rows = [[r["dataset"], r["recipe"], r["runs"], fmt(float(r["acc"])), fmt(float(r["gap"])), fmt(float(r["ECE"])), fmt(float(r["step"])), fmt(float(r["bmem"])), "yes" if int(r["pass"]) else "no"] for r in p4_summary]
    failure_rows = [[k, v] for k, v in sorted(Counter(str(r.get("failure_type", "")) for r in failures).items())]

    artifacts = "\n".join(name for name, _ in gate_files_exist()) + "\nfailure_table.csv\nfailure_by_primitive.csv\nfailure_by_dataset.csv\nfailure_by_gate.csv\nroute_decision.json\naggregate_decision.json\nfigures/"
    doc = f"""# DG-KAN v5.8 Efficient Primitive Redesign 结果复盘

本轮依据 `docs/DG-KAN_v5.8_EfficientPrimitive_Redesign_实验计划.md`。核心目标是在继续 functional optimizer / LightSmooth 之前，重新设计并验证更接近 MLP-like kernel path 的 strict PureKAN primitive。

## Run Inventory

{md_table(["stage", "rows", "errors"], run_rows)}

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added core efficient primitive candidates:
    DWM2Dense
    RationalKATV2Dense
    LUTKANDense
  Extended edge/base/RBF/mixing parameter discovery for the new primitives.

experiments/run_gafu_v58.py
  Added P0 core manifest, P1 small/medium/large efficiency decomposition,
  gated DWM-v2 / RationalKAT-v2 / LUTKAN recipe probes,
  joint selection, and required gated artifacts.

experiments/analyze_gafu_v58.py
  Generates v5.8 gate summaries, route_decision.json, aggregate_decision.json,
  failure taxonomy, figures, and this replay.
```

## P0 Core Manifest

{md_table(["primitive", "class", "edge", "nonKAN", "mixing", "rollback", "P0"], p0_rows)}

P0 verdict: core manifest passes when each strict PureKAN primitive is edge-covered, nonKAN-free, and rollback-exact.

## P1 Efficiency Decomposition

{md_table(["method", "rows", "fwd", "bwd", "bmem", "step", "explore", "final", "bottleneck"], p1_rows)}

P1 exploratory survivors:

```text
{", ".join(p1_exploratory) if p1_exploratory else "none"}
```

P1 final survivors:

```text
{", ".join(p1_final) if p1_final else "none"}
```

## P2 DWM-v2 Recipe

{md_table(["dataset", "recipe", "runs", "acc", "gap", "ECE", "step", "bmem", "P2"], p2_rows) if p2_rows else "_P2 not run or no recipe rows._"}

P2 all-dataset survivors:

```text
{", ".join(p2_surv) if p2_surv else "none"}
```

## P3 RationalKAT-v2 Recipe

{md_table(["dataset", "recipe", "runs", "acc", "gap", "ECE", "step", "bmem", "P3"], p3_rows) if p3_rows else "_P3 not run or no recipe rows._"}

P3 all-dataset survivors:

```text
{", ".join(p3_surv) if p3_surv else "none"}
```

## P4 LUTKAN Feasibility

{md_table(["dataset", "recipe", "runs", "acc", "gap", "ECE", "step", "bmem", "P4"], p4_rows) if p4_rows else "_P4 not run or no recipe rows._"}

P4 all-dataset survivors:

```text
{", ".join(p4_surv) if p4_surv else "none"}
```

## P5-P10 Decision

```text
P5 joint task-efficiency selection: {"run" if p5_surv else "not run / no survivor"}
P6 custom backward audit: {"run" if p5_surv else "not run"}
P7 LightSmooth compatibility: not run unless P6 survives
P8 functional training smoke: not run unless P7 survives
P9/P10 confirm: not run unless P8 survives
Final decision: {status}
Route case: {route_case}
```

## Failure Diagnosis

{md_table(["failure", "count"], failure_rows)}

## Required Artifacts

Written under `results/v5_8/`:

```text
{artifacts}
```

## Final Decision

```text
DG-KAN v5.8 status:
  {status}

Route:
  {route_case}

What improved:
  v5.8 adds three redesigned strict PureKAN primitive families in core.
  P1 now measures small/medium/large efficiency decomposition before recipe spending.
  Functional optimizer and LightSmooth remain correctly gated behind primitive survival.

What failed / remains open:
  See the P1-P5 gates above.
  If no P1/P5 survivor exists, the bottleneck remains primitive efficiency or task recipe,
  not functional smoothing.

Conclusion:
  v5.8 follows the kernel-first rule: do not resume FGO-v3 until an efficient primitive
  satisfies the joint task + efficiency envelope.
```
"""
    OUT.write_text(doc, encoding="utf-8")

    existing = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    entry = f"\n- v5.8 EfficientPrimitive_Redesign: {status} ({route_case}); artifacts in `results/v5_8/`.\n"
    if "v5.8 EfficientPrimitive_Redesign" not in existing:
        LOG.write_text(existing + entry, encoding="utf-8")


if __name__ == "__main__":
    main()
