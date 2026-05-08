#!/usr/bin/env python3
"""Summarize DG-KAN v3.9 Normalization / FNG PureKAN experiments."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results/gafu_v3_9_p0/runs.csv"
P1 = ROOT / "results/gafu_v3_9_p1_shadow/runs.csv"
P2 = ROOT / "results/gafu_v3_9_p2_norm/runs.csv"
P3 = ROOT / "results/gafu_v3_9_p3_fng_relaxed/runs.csv"
FIG_DIR = ROOT / "results/gafu_v3_9_figures"
OUT = ROOT / "docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_结果复盘.md"
LOG = ROOT / "docs/log.md"


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(x: float, digits: int = 4) -> str:
    if x is None or not math.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def fmt_intish(x: float) -> str:
    if x is None or not math.isfinite(x):
        return ""
    return str(int(round(x)))


def clean_method(name: str) -> str:
    out = name.replace("-alphaFixed1", "")
    out = out.replace("PureKAN-", "")
    return out


def display_norm_update(row: dict) -> str:
    method = row.get("method", "")
    update = row.get("norm_update_method", "")
    if update in {"", "none"} and "AdamW-LN" in method:
        return "adamw"
    return update


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(cell(c) for c in row) + " |" for row in rows)
    return "\n".join(out)


def group_rows(rows: Iterable[dict], keys: Sequence[str]) -> Dict[tuple, List[dict]]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    return dict(groups)


def agg(rows: Sequence[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def agg_std(rows: Sequence[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return pstdev(vals) if len(vals) > 1 else 0.0 if vals else math.nan


def rel_improve(base: float, value: float) -> float:
    if not math.isfinite(base) or not math.isfinite(value) or abs(base) <= 1e-12:
        return math.nan
    return (base - value) / abs(base)


def run_inventory(*sets: tuple[str, List[dict]]) -> str:
    rows = []
    for name, data in sets:
        rows.append([name, len(data), sum(1 for r in data if r.get("error"))])
    return md_table(["stage", "rows", "errors"], rows)


def p0_table(rows: List[dict]) -> str:
    table = []
    for r in sorted(rows, key=lambda x: (x.get("dataset", ""), x.get("method", ""))):
        fng_cond = max(
            f(r, "fng_input_combined_condition", 0.0),
            f(r, "fng_block_combined_condition", 0.0),
            f(r, "fng_output_combined_condition", 0.0),
        )
        table.append(
            [
                r.get("dataset", ""),
                clean_method(r.get("method", "")),
                r.get("pure_norm_mode", "") or r.get("norm_mode", ""),
                display_norm_update(r),
                fmt_intish(f(r, "learnable_nonkan_params")),
                fmt_intish(f(r, "learnable_nonkan_raw_params")),
                fmt(f(r, "functional_param_coverage"), 3),
                fmt(f(r, "input_kan_coeff_seen"), 3),
                fmt(f(r, "block_kan_coeff_seen"), 3),
                fmt(f(r, "output_kan_coeff_seen"), 3),
                fmt_intish(f(r, "norm_param_count")),
                fmt(f(r, "norm_param_coverage"), 3),
                r.get("fng_mode", ""),
                fmt(fng_cond, 2),
                fmt(f(r, "fng_fallback_rate"), 3),
            ]
        )
    return md_table(
        [
            "dataset",
            "method",
            "norm",
            "norm update",
            "nonKAN",
            "raw nonKAN",
            "func cov",
            "input coeff",
            "block coeff",
            "output coeff",
            "norm params",
            "norm cov",
            "FNG",
            "FNG cond",
            "fallback",
        ],
        table,
    )


def p1_role_cos(row: dict) -> float:
    vals = [
        f(row, "input_cos_raw_precond"),
        f(row, "block_cos_raw_precond_mean"),
        f(row, "output_cos_raw_precond"),
    ]
    vals = [v for v in vals if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def p1_row_cond(row: dict) -> float:
    vals = [
        f(row, "fng_input_combined_condition"),
        f(row, "fng_block_combined_condition"),
        f(row, "fng_output_combined_condition"),
    ]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else math.nan


def p1_table(rows: List[dict]) -> str:
    order = {"MNIST": 0, "Fashion-MNIST": 1, "KMNIST": 2}
    table = []
    for r in sorted(rows, key=lambda x: (order.get(x.get("dataset", ""), 9), x.get("method", ""))):
        table.append(
            [
                r.get("dataset", ""),
                r.get("method", ""),
                fmt(f(r, "shadow_actual_descent")),
                fmt(f(r, "shadow_val_descent")),
                fmt_intish(f(r, "shadow_bad_step_bool")),
                fmt(p1_role_cos(r)),
                fmt(p1_row_cond(r), 1),
                r.get("norm_mode", ""),
                r.get("norm_update_method", ""),
            ]
        )
    return md_table(
        ["dataset", "method", "train descent", "val descent", "bad", "mean cos", "max cond", "norm", "norm update"],
        table,
    )


def p1_gate_table(rows: List[dict]) -> tuple[str, List[str]]:
    groups = group_rows(rows, ("method",))
    table = []
    passed: List[str] = []
    for (method,), rs in sorted(groups.items()):
        if method == "AdamW-one-step":
            continue
        bad = sum(int(round(f(r, "shadow_bad_step_bool", 0.0))) for r in rs)
        min_train = min((f(r, "shadow_actual_descent") for r in rs), default=math.nan)
        val_ok = sum(1 for r in rs if f(r, "shadow_val_descent") >= 0.0)
        cos_mean = mean([p1_role_cos(r) for r in rs if math.isfinite(p1_role_cos(r))])
        max_cond = max([p1_row_cond(r) for r in rs if math.isfinite(p1_row_cond(r))], default=math.nan)
        ok = bad == 0 and min_train > 0.0 and val_ok >= 2 and cos_mean > 0.5 and max_cond < 1e5
        if ok:
            passed.append(method)
        table.append(
            [
                method,
                bad,
                fmt(min_train),
                f"{val_ok}/3",
                fmt(cos_mean),
                fmt(max_cond, 1),
                "yes" if ok else "no",
            ]
        )
    return md_table(["method", "bad", "min train", "val >=0", "mean cos", "max cond", "P1 pass"], table), passed


def score_table(
    rows: List[dict],
    *,
    adam_prefix: str,
    d0_prefix: str | None = None,
    d6_prefix: str | None = None,
    methods: Sequence[str] | None = None,
) -> str:
    groups = group_rows(rows, ("dataset", "method"))
    table = []
    for dataset in ["MNIST", "Fashion-MNIST", "KMNIST"]:
        data_methods = {m: rs for (d, m), rs in groups.items() if d == dataset}
        if not data_methods:
            continue
        adam_rows = next((rs for m, rs in data_methods.items() if m.startswith(adam_prefix)), [])
        d0_rows = next((rs for m, rs in data_methods.items() if d0_prefix and m.startswith(d0_prefix)), [])
        d6_rows = next((rs for m, rs in data_methods.items() if d6_prefix and m.startswith(d6_prefix)), [])
        adam_acc = agg(adam_rows, "test_acc")
        adam_auc = agg(adam_rows, "val_auc")
        adam_ece = agg(adam_rows, "ece")
        d0_acc = agg(d0_rows, "test_acc")
        d6_auc = agg(d6_rows, "val_auc")
        adam_time = agg(adam_rows, "step_time_ms")
        selected = methods or sorted(data_methods)
        for method in selected:
            rs = data_methods.get(method, [])
            if len(rs) < 1:
                continue
            acc = agg(rs, "test_acc")
            auc = agg(rs, "val_auc")
            ece = agg(rs, "ece")
            fng_cond = max(
                agg(rs, "fng_input_combined_condition"),
                agg(rs, "fng_block_combined_condition"),
                agg(rs, "fng_output_combined_condition"),
            )
            table.append(
                [
                    dataset,
                    clean_method(method),
                    len(rs),
                    fmt(acc),
                    fmt(agg_std(rs, "test_acc")),
                    fmt(adam_acc - acc),
                    fmt(acc - d0_acc) if d0_rows else "",
                    fmt(rel_improve(d6_auc, auc)) if d6_rows else "",
                    fmt(rel_improve(adam_auc, auc)),
                    fmt(rel_improve(adam_ece, ece)),
                    fmt(agg(rs, "phi_prime_p95")),
                    fmt(agg(rs, "max_jac_condition")),
                    fmt(fng_cond, 1),
                    fmt(agg(rs, "fng_bad_step_rate")),
                    fmt(agg(rs, "fng_fallback_rate")),
                    fmt(agg(rs, "learnable_nonkan_params"), 0),
                    fmt(agg(rs, "norm_param_coverage"), 3),
                    fmt(agg(rs, "step_time_ms") / adam_time if math.isfinite(adam_time) and adam_time else math.nan),
                ]
            )
    return md_table(
        [
            "dataset",
            "method",
            "runs",
            "acc",
            "std",
            "gap vs AdamW",
            "acc-D0",
            "AUC imp vs D6",
            "AUC imp vs AdamW",
            "ECE red",
            "phi p95",
            "J",
            "FNG cond",
            "bad",
            "fallback",
            "nonKAN",
            "norm cov",
            "time/A",
        ],
        table,
    )


def p3_gate_table(rows: List[dict], p1_passed: Sequence[str]) -> str:
    groups = group_rows(rows, ("dataset", "method"))
    methods = sorted({m for _, m in groups if not m.startswith("PureKAN-AdamW") and not m.startswith("D0-")})
    p1_set = set(p1_passed)
    table = []
    for method in methods:
        gaps = []
        auc_ok = []
        bads = []
        fallbacks = []
        geom_ok = []
        for dataset in ["MNIST", "Fashion-MNIST", "KMNIST"]:
            adam = next((rs for (d, m), rs in groups.items() if d == dataset and m.startswith("PureKAN-AdamW")), [])
            d6 = next((rs for (d, m), rs in groups.items() if d == dataset and m.startswith("D6-allTaskAware")), [])
            rs = groups.get((dataset, method), [])
            if not adam or not rs:
                continue
            gaps.append(agg(adam, "test_acc") - agg(rs, "test_acc"))
            auc_ok.append(agg(rs, "val_auc") < agg(d6, "val_auc") if d6 else False)
            bads.append(agg(rs, "fng_bad_step_rate"))
            fallbacks.append(agg(rs, "fng_fallback_rate"))
            geom_ok.append(agg(rs, "phi_prime_p95") <= agg(adam, "phi_prime_p95"))
        relaxed_gate = (
            len(gaps) == 3
            and gaps[0] < 0.02
            and gaps[1] < 0.02
            and gaps[2] < 0.03
            and all(auc_ok)
            and max([v for v in bads if math.isfinite(v)] or [0.0]) <= 0.0
            and max([v for v in fallbacks if math.isfinite(v)] or [0.0]) < 0.2
            and all(geom_ok)
        )
        formal = method in p1_set and relaxed_gate
        table.append(
            [
                clean_method(method),
                "yes" if method in p1_set else "no",
                fmt(max(gaps) if gaps else math.nan),
                "yes" if all(auc_ok) and auc_ok else "no",
                fmt(max([v for v in bads if math.isfinite(v)] or [math.nan])),
                fmt(max([v for v in fallbacks if math.isfinite(v)] or [math.nan])),
                "yes" if all(geom_ok) and geom_ok else "no",
                "yes" if formal else "no",
            ]
        )
    return md_table(
        ["method", "formal P1 pass", "max acc gap", "AUC < D6 all", "max bad", "max fallback", "geom <= AdamW", "enter P4"],
        table,
    )


def create_figures(p1: List[dict], p2: List[dict], p3: List[dict]) -> List[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        paths = create_svg_figures(p1, p2, p3)
        paths.append(f"matplotlib unavailable; SVG fallback used: {exc}")
        return paths

    paths: List[str] = []

    p1_groups = group_rows(p1, ("method",))
    methods = [m for (m,) in sorted(p1_groups) if m != "AdamW-one-step"]
    train = [agg(rs, "shadow_actual_descent") for (m,), rs in sorted(p1_groups.items()) if m != "AdamW-one-step"]
    val = [agg(rs, "shadow_val_descent") for (m,), rs in sorted(p1_groups.items()) if m != "AdamW-one-step"]
    bad = [agg(rs, "shadow_bad_step_bool") for (m,), rs in sorted(p1_groups.items()) if m != "AdamW-one-step"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(methods))
    ax.bar([i - 0.25 for i in x], train, width=0.25, label="train descent")
    ax.bar(x, val, width=0.25, label="val descent")
    ax.bar([i + 0.25 for i in x], bad, width=0.25, label="bad rate")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    ax.set_title("v3.9 P1 direction quality")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "p1_direction_quality.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    paths.append(str(out.relative_to(ROOT)))

    def scatter_score(rows: List[dict], out_name: str, title: str, baseline_prefix: str) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        groups = group_rows(rows, ("dataset", "method"))
        for ax, dataset in zip(axes, ["MNIST", "Fashion-MNIST", "KMNIST"]):
            adam = next((rs for (d, m), rs in groups.items() if d == dataset and m.startswith(baseline_prefix)), [])
            adam_auc = agg(adam, "val_auc")
            for (d, method), rs in groups.items():
                if d != dataset or len(rs) == 0:
                    continue
                auc_imp = rel_improve(adam_auc, agg(rs, "val_auc"))
                ax.scatter(agg(rs, "test_acc"), auc_imp, s=35)
                ax.annotate(clean_method(method)[:24], (agg(rs, "test_acc"), auc_imp), fontsize=6, alpha=0.8)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(dataset)
            ax.set_xlabel("test acc")
            ax.set_ylabel("AUC improvement vs AdamW")
        fig.suptitle(title)
        fig.tight_layout()
        out = FIG_DIR / out_name
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(str(out.relative_to(ROOT)))

    scatter_score(p2, "p2_norm_scorecard.png", "v3.9 P2 normalization ablation", "PureKAN-FixedNorm-AdamW")
    if p3:
        scatter_score(p3, "p3_fng_relaxed_scorecard.png", "v3.9 P3 relaxed FNG diagnostic", "PureKAN-AdamW")

    return paths


def svg_text(x: float, y: float, text: object, *, size: int = 11, anchor: str = "start") -> str:
    value = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-family="monospace">{value}</text>'


def create_svg_figures(p1: List[dict], p2: List[dict], p3: List[dict]) -> List[str]:
    paths: List[str] = []
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    p1_groups = group_rows(p1, ("method",))
    methods = [m for (m,) in sorted(p1_groups) if m != "AdamW-one-step"]
    values = []
    for method in methods:
        rs = p1_groups[(method,)]
        values.append((method, agg(rs, "shadow_actual_descent"), agg(rs, "shadow_val_descent"), agg(rs, "shadow_bad_step_bool")))
    width = 1120
    height = 80 + 28 * max(1, len(values))
    left = 330
    right = 1040
    numeric = [v for _, a, b, c in values for v in (a, b, c) if math.isfinite(v)]
    lo = min([-0.05] + numeric)
    hi = max([0.10] + numeric)
    span = max(1e-9, hi - lo)

    def xmap(v: float) -> float:
        return left + (v - lo) / span * (right - left)

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(20, 28, "v3.9 P1 direction quality SVG fallback", size=16),
        f'<line x1="{xmap(0):.1f}" y1="50" x2="{xmap(0):.1f}" y2="{height-20}" stroke="#222" stroke-width="1"/>',
    ]
    for idx, (method, train, val, bad) in enumerate(values):
        y = 66 + idx * 28
        elems.append(svg_text(20, y + 5, method[:42], size=10))
        for offset, value, color, label in [(-7, train, "#2563eb", "train"), (0, val, "#16a34a", "val"), (7, bad, "#dc2626", "bad")]:
            if not math.isfinite(value):
                continue
            x0 = xmap(0)
            x1 = xmap(value)
            elems.append(
                f'<rect x="{min(x0, x1):.1f}" y="{y+offset:.1f}" width="{abs(x1-x0):.1f}" height="5" fill="{color}"/>'
            )
        elems.append(svg_text(right + 10, y + 5, f"T={train:.3f} V={val:.3f} B={bad:.2f}", size=9))
    elems.append(svg_text(left, height - 6, f"range {lo:.3f} .. {hi:.3f}", size=9))
    elems.append("</svg>")
    out = FIG_DIR / "p1_direction_quality.svg"
    out.write_text("\n".join(elems))
    paths.append(str(out.relative_to(ROOT)))

    def scatter_svg(rows: List[dict], out_name: str, title: str, baseline_prefix: str) -> None:
        groups = group_rows(rows, ("dataset", "method"))
        panels = ["MNIST", "Fashion-MNIST", "KMNIST"]
        width = 1200
        height = 430
        elems = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            svg_text(20, 28, title, size=16),
        ]
        for pidx, dataset in enumerate(panels):
            items = []
            adam = next((rs for (d, m), rs in groups.items() if d == dataset and m.startswith(baseline_prefix)), [])
            adam_auc = agg(adam, "val_auc")
            for (d, method), rs in groups.items():
                if d != dataset or not rs:
                    continue
                items.append((method, agg(rs, "test_acc"), rel_improve(adam_auc, agg(rs, "val_auc"))))
            x0 = 35 + pidx * 390
            y0 = 58
            pw = 350
            ph = 300
            elems.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1"/>')
            elems.append(svg_text(x0 + 8, y0 + 18, dataset, size=13))
            xs = [x for _, x, _ in items if math.isfinite(x)]
            ys = [y for _, _, y in items if math.isfinite(y)]
            xmin = min(xs) - 0.005 if xs else 0.0
            xmax = max(xs) + 0.005 if xs else 1.0
            ymin = min([-0.05] + ys)
            ymax = max([0.05] + ys)
            if abs(xmax - xmin) < 1e-9:
                xmax = xmin + 0.01
            if abs(ymax - ymin) < 1e-9:
                ymax = ymin + 0.01

            def px(v: float) -> float:
                return x0 + 28 + (v - xmin) / (xmax - xmin) * (pw - 46)

            def py(v: float) -> float:
                return y0 + ph - 28 - (v - ymin) / (ymax - ymin) * (ph - 56)

            yzero = py(0.0)
            elems.append(f'<line x1="{x0+22}" y1="{yzero:.1f}" x2="{x0+pw-10}" y2="{yzero:.1f}" stroke="#94a3b8"/>')
            for method, acc, auc_imp in items:
                if not math.isfinite(acc) or not math.isfinite(auc_imp):
                    continue
                x = px(acc)
                y = py(auc_imp)
                elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#7c3aed" opacity="0.8"/>')
                elems.append(svg_text(x + 5, y - 4, clean_method(method)[:24], size=7))
            elems.append(svg_text(x0 + 28, y0 + ph - 6, "test acc", size=9))
            elems.append(svg_text(x0 + pw - 10, y0 + ph - 6, f"{xmax:.3f}", size=8, anchor="end"))
            elems.append(svg_text(x0 + 4, y0 + 36, "AUC imp", size=8))
        elems.append("</svg>")
        out = FIG_DIR / out_name
        out.write_text("\n".join(elems))
        paths.append(str(out.relative_to(ROOT)))

    scatter_svg(p2, "p2_norm_scorecard.svg", "v3.9 P2 normalization ablation", "PureKAN-FixedNorm-AdamW")
    if p3:
        scatter_svg(p3, "p3_fng_relaxed_scorecard.svg", "v3.9 P3 relaxed FNG diagnostic", "PureKAN-AdamW")
    return paths


def append_log(p1_passed: Sequence[str]) -> None:
    marker = "## 2026-05-02 DG-KAN v3.9 Normalization / FNG PureKAN"
    text = LOG.read_text() if LOG.exists() else ""
    if marker in text:
        return
    entry = f"""

{marker}

依据 `docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_实验计划.md`，实现并运行：

```text
P0 implementation smoke
P1 one-batch direction audit
P2 normalization ablation
P3 relaxed FNG diagnostic subset
```

关键结论：

```text
P0 pass: norm modes / functional norm / FNG update 覆盖正常。
P1 strict fail: no candidate passed the formal direction gate; no-bad-step FNG variants still had raw-precond cosine below 0.5.
P2: NoNorm fails hard, but AffineNorm/ScalarGain does not close the PureKAN functional gap, so normalization is necessary but not the missing key.
P3: relaxed diagnostic run did not justify P4/P5 expansion under the plan gate.
P4-P7 not run: no formal P1/P3 survivor.
```

复盘文档：

```text
docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_结果复盘.md
```
"""
    LOG.write_text(text.rstrip() + entry + "\n")


def main() -> int:
    p0 = read_rows(P0)
    p1 = read_rows(P1)
    p2 = read_rows(P2)
    p3 = read_rows(P3)
    p1_gate, p1_passed = p1_gate_table(p1)
    figures = create_figures(p1, p2, p3)

    p2_methods = [
        "PureKAN-FixedNorm-AdamW",
        "PureKAN-AffineNorm-AdamW-LN",
        "PureKAN-AffineNorm-FDiag-LN",
        "PureKAN-ScalarGainNorm-FDiag",
        "PureKAN-NoNorm-AdamW",
        "PureKAN-FixedNorm-D6-allTaskAware",
        "PureKAN-AffineNorm-AdamW-LN-D6",
        "PureKAN-AffineNorm-FDiag-LN-D6",
        "PureKAN-FixedNorm-FNG-leftFullRight",
        "PureKAN-AffineNorm-AdamW-LN-FNG-leftFullRight",
        "PureKAN-AffineNorm-FDiag-LN-FNG-leftFullRight",
    ]
    p3_methods = [
        "PureKAN-AdamW",
        "D0-allFullSobolev",
        "D6-allTaskAware",
        "F2-FNG-right-only",
        "F4-FNG-leftFull-right-noSob",
        "F4-FNG-leftFull-right-lowSob",
    ]

    doc = f"""# DG-KAN v3.9 Normalization / FNG PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_实验计划.md`。目标是判断 PureKAN functional update 追不上 PureKAN-AdamW 的主因，是缺少 learnable normalization，还是缺少真正 task/Fisher-aware 的 function-space metric。

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added PureKAN norm modes: fixed / affine_channel / scalar_gain / none.
  Added functional diagonal norm update for gamma/beta or scalar gain/bias.
  Added FNG/KFAC-style coefficient update metrics:
    fng_right
    fng_leftdiag_right
    fng_leftfull_right
    fng_leftlowrank_right
  Result rows now include norm audit, functional coverage, FNG metric conditions,
  FNG fallback/bad-step rates, timing, and per-role metric traces.

experiments/run_gafu_v39.py
  Added P0 smoke, P1 shadow audit, P2 normalization ablation,
  P3 FNG package, and optional P4 dynamics wiring.

experiments/analyze_gafu_v39.py
  Generates this replay, appends the log entry, and writes basic dashboards.
```

## Run Inventory

{run_inventory(("P0 smoke", p0), ("P1 shadow", p1), ("P2 norm", p2), ("P3 relaxed", p3))}

## P0 Implementation Smoke

{p0_table(p0)}

P0 verdict: pass. Strict PureKAN functional variants have zero learnable non-KAN parameters and full coefficient coverage. Diagnostic `AffineNorm-AdamW-LN` correctly exposes learnable non-KAN norm parameters and is not eligible for final PureKAN claims.

## P1 One-Batch Direction Audit

{p1_table(p1)}

### P1 Gate Check

{p1_gate}

P1 verdict: strict fail. FNG variants repair several bad-step cases compared with D0/D6, but no candidate satisfies the full direction gate because the mean raw/preconditioned cosine remains below `0.5`. Therefore no method formally enters P3/P4 under the written plan.

## P2 Normalization Ablation

{score_table(p2, adam_prefix="PureKAN-FixedNorm-AdamW", d6_prefix="PureKAN-FixedNorm-D6", methods=p2_methods)}

P2 verdict:

```text
NoNorm is a hard failure boundary, so normalization is necessary.
Learnable AffineNorm / ScalarGain can slightly help AdamW on MNIST/KMNIST,
but it does not close the PureKAN functional optimizer gap on Fashion/KMNIST.
AffineNorm-FDiag also does not rescue FNG-leftFullRight.

Conclusion: missing learnable normalization is not the main bottleneck.
The core issue remains KAN coefficient optimizer dynamics / metric direction.
```

## P3 Relaxed FNG Diagnostic

Because P1 had no formal survivor, this run is intentionally labeled relaxed diagnostic. It includes only the no-bad-step FNG variants plus baselines, to check whether the P1 local signal becomes a short-run training signal.

{score_table(p3, adam_prefix="PureKAN-AdamW", d0_prefix="D0-allFullSobolev", d6_prefix="D6-allTaskAware", methods=p3_methods)}

### P3 Gate Check

{p3_gate_table(p3, p1_passed)}

P3 verdict: no formal survivor enters P4. The relaxed run is useful diagnostically, but it cannot override the P1 gate. FNG improves some Sobolev-only trajectories, yet the joint accuracy/AUC/geometry requirements are not met across MNIST, Fashion-MNIST, and KMNIST.

## Visualizations

Generated dashboards:

```text
{chr(10).join(figures)}
```

## P4-P7 Decision

```text
P4 temporal dynamics:
  not run
  reason: no formal P1/P3 survivor.

P5 combined norm + FNG selection:
  not run
  reason: P3 entry gate was not reached.

P6 5-seed confirm / P7 10-seed final:
  not run by plan.
```

## Failure Diagnosis

```text
Normalization bottleneck:
  not primary.
  NoNorm fails, but learnable/functional norm does not recover PureKAN functional training.

FNG direction bottleneck:
  yes.
  FNG fixes many one-step bad directions, but the cosine gate and cross-dataset training gates fail.

Temporal dynamics bottleneck:
  possible but not yet justified.
  Since no candidate formally passed P1/P3, larger temporal-dynamics sweeps would be premature.

Architecture bottleneck:
  unlikely as the main cause.
  PureKAN-AdamW remains trainable; optimizer metric is the weak link.
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v3.9.

What is confirmed:
  1. Normalization is necessary, but learnable norm is not sufficient.
  2. FNG/KFAC-style task metrics repair part of the local descent problem.
  3. Current FNG still does not replace AdamW for full PureKAN training.

Accepted contribution:
  Hybrid DG-KAN branch functional update remains the valid mainline.

Next recommended direction:
  Redesign the PureKAN functional metric around safer layer-local/block-local objectives
  and only revisit temporal momentum/safeguards once a candidate passes the direction gate.
```
"""
    OUT.write_text(doc)
    append_log(p1_passed)
    print(f"wrote {OUT.relative_to(ROOT)}")
    for fig in figures:
        print(f"figure {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
