#!/usr/bin/env python3
"""Aggregate DG-KAN Optimizer v3.4 AFU experiment results."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgkan_core import add_baseline_comparisons, ensure_dir, read_csv, save_json, summarize_runs, write_csv


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results/gafu_v3_4_p0_smoke"
P2 = ROOT / "results/gafu_v3_4_p2_component3"
P3 = ROOT / "results/gafu_v3_4_p3_cumulative3"
P4 = ROOT / "results/gafu_v3_4_p4_expr_rescue"
P5 = ROOT / "results/gafu_v3_4_p5_confirm5"


def f(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_curve(text: Any) -> List[float]:
    if text in ("", None):
        return []
    return [float(x) for x in str(text).split(",") if x.strip()]


def method_rows(rows: Iterable[Dict[str, Any]], dataset: str, method: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("dataset") == dataset and r.get("method") == method]


def by_seed(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        out[int(row["seed"])] = row
    return out


def bootstrap_ci(values: List[float], *, n: int = 2000, seed: int = 0) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = vals[rng.integers(0, vals.size, size=(n, vals.size))].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def paired_delta_rows(rows: List[Dict[str, Any]], dataset: str, baseline: str, methods: List[str]) -> List[Dict[str, Any]]:
    base = by_seed(method_rows(rows, dataset, baseline))
    out: List[Dict[str, Any]] = []
    for method in methods:
        current = by_seed(method_rows(rows, dataset, method))
        seeds = sorted(set(base) & set(current))
        acc_delta = [f(current[s], "test_acc") - f(base[s], "test_acc") for s in seeds]
        auc_delta = [f(base[s], "val_auc") - f(current[s], "val_auc") for s in seeds]
        acc_lo, acc_hi = bootstrap_ci(acc_delta)
        auc_lo, auc_hi = bootstrap_ci(auc_delta)
        out.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "method": method,
                "seeds": ",".join(str(s) for s in seeds),
                "paired_acc_delta_mean": float(np.mean(acc_delta)) if acc_delta else float("nan"),
                "paired_acc_delta_ci95_lo": acc_lo,
                "paired_acc_delta_ci95_hi": acc_hi,
                "paired_auc_delta_mean": float(np.mean(auc_delta)) if auc_delta else float("nan"),
                "paired_auc_delta_ci95_lo": auc_lo,
                "paired_auc_delta_ci95_hi": auc_hi,
            }
        )
    return out


def first_reach(curve: List[float], target: float, *, mode: str) -> int:
    for idx, value in enumerate(curve, start=1):
        if (mode == "loss" and value <= target) or (mode == "acc" and value >= target):
            return idx
    return -1


def target_matched(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    methods_by_dataset = {
        "Fashion-MNIST": [
            "AdamW-alphaFixed1-alphaFixed1",
            "AFU-0-Hybrid-alphaFixed1",
            "AFU-LNOnly-alphaFixed1",
            "AFU-2-HeadCov-alphaFixed1",
        ],
        "KMNIST": [
            "AdamW-alphaFixed1-alphaFixed1",
            "AFU-0-Hybrid-alphaFixed1",
            "AFU-LNOnly-alphaFixed1",
            "AFU-2-HeadCov-alphaFixed1",
        ],
    }
    out: List[Dict[str, Any]] = []
    for dataset, methods in methods_by_dataset.items():
        adam = by_seed(method_rows(rows, dataset, methods[0]))
        for method in methods:
            for row in method_rows(rows, dataset, method):
                seed = int(row["seed"])
                if seed not in adam:
                    continue
                adam_loss_curve = parse_curve(adam[seed].get("val_loss_curve"))
                curve = parse_curve(row.get("val_loss_curve"))
                if not curve or not adam_loss_curve:
                    continue
                steps_per_epoch = math.ceil(f(row, "train_size") / f(row, "batch_size", 256.0))
                step_time_sec = f(row, "step_time_ms") / 1000.0
                relaxed_loss = 1.05 * adam_loss_curve[-1]
                relaxed_loss_epoch = first_reach(curve, relaxed_loss, mode="loss")
                out.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "reached_relaxed_loss": int(relaxed_loss_epoch > 0),
                        "epochs_to_relaxed_loss": relaxed_loss_epoch,
                        "time_to_relaxed_loss_sec": relaxed_loss_epoch * steps_per_epoch * step_time_sec
                        if relaxed_loss_epoch > 0
                        else float("nan"),
                        "step_time_ms": f(row, "step_time_ms"),
                    }
                )
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in out:
        groups.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)
    summary: List[Dict[str, Any]] = []
    for (dataset, method), group in groups.items():
        item: Dict[str, Any] = {"dataset": dataset, "method": method, "runs": len(group)}
        for key in ["reached_relaxed_loss", "epochs_to_relaxed_loss", "time_to_relaxed_loss_sec", "step_time_ms"]:
            vals = [f(row, key) for row in group if math.isfinite(f(row, key))]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                item[f"{key}_mean"] = float(arr.mean())
                item[f"{key}_std"] = float(arr.std(ddof=0))
        summary.append(item)
    return out, sorted(summary, key=lambda r: (str(r["dataset"]), str(r["method"])))


def md_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]], *, digits: int = 4) -> str:
    lines = ["| " + " | ".join(title for title, _ in columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            try:
                cells.append(f"{float(value):.{digits}f}")
            except (TypeError, ValueError):
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def p1_proxy_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        for group in ["kan_bias", "head", "stem", "ln"]:
            cos = f(row, f"afu_{group}_cos_raw_precond_mean")
            upd = f(row, f"afu_{group}_update_over_param_mean")
            cond = f(row, f"afu_{group}_metric_condition_mean")
            if not any(math.isfinite(x) for x in [cos, upd, cond]):
                continue
            out.append(
                {
                    "dataset": row.get("dataset"),
                    "method": row.get("method"),
                    "seed": row.get("seed"),
                    "group": group,
                    "direction_cos_raw_vs_afu": cos,
                    "update_over_param": upd,
                    "metric_condition": cond,
                    "p1_status": "proxy_from_training_audit",
                }
            )
    return out


def maybe_plots(expression_rows: List[Dict[str, Any]], target_summary: List[Dict[str, Any]]) -> None:
    p7 = ensure_dir(ROOT / "results/gafu_v3_4_p7_mechanism")
    p8 = ensure_dir(ROOT / "results/gafu_v3_4_p8_wallclock")
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    def scatter(path: Path, rows: List[Dict[str, Any]], x_key: str, y_key: str, hline: float | None = None) -> None:
        pts = [(f(r, x_key), f(r, y_key), str(r.get("dataset", "")), str(r.get("method", ""))) for r in rows]
        pts = [(x, y, ds, m) for x, y, ds, m in pts if math.isfinite(x) and math.isfinite(y)]
        if not pts:
            return
        w, h, pad = 900, 560, 70
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if hline is not None:
            y0, y1 = min(y0, hline), max(y1, hline)
        x0 -= 0.08 * max(1e-6, x1 - x0)
        x1 += 0.08 * max(1e-6, x1 - x0)
        y0 -= 0.08 * max(1e-6, y1 - y0)
        y1 += 0.08 * max(1e-6, y1 - y0)
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        draw.line((pad, h - pad, w - pad, h - pad), fill="black")
        draw.line((pad, pad, pad, h - pad), fill="black")

        def sx(x: float) -> float:
            return pad + (x - x0) / max(1e-9, x1 - x0) * (w - 2 * pad)

        def sy(y: float) -> float:
            return h - pad - (y - y0) / max(1e-9, y1 - y0) * (h - 2 * pad)

        if hline is not None:
            yy = sy(hline)
            draw.line((pad, yy, w - pad, yy), fill=(80, 80, 80), width=2)
        colors = {"Fashion-MNIST": (40, 110, 200), "KMNIST": (210, 90, 40)}
        for x, y, ds, method in pts:
            cx, cy = sx(x), sy(y)
            color = colors.get(ds, (80, 80, 80))
            draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=color, outline="black")
            draw.text((cx + 9, cy - 7), method[:18], fill=color)
        draw.text((pad, 20), f"{y_key} vs {x_key}", fill="black")
        img.save(path)

    scatter(p7 / "auc_vs_no_kan.png", expression_rows, "val_auc_improvement_vs_adamw_mean", "no_kan_drop_over_adamw_mean", hline=0.7)
    scatter(p7 / "phi_vs_no_kan.png", expression_rows, "phi_prime_reduction_vs_adamw_mean", "no_kan_drop_over_adamw_mean", hline=0.7)
    scatter(p8 / "auc_vs_step_time.png", target_summary, "step_time_ms_mean", "time_to_relaxed_loss_sec_mean")


def failure_rows(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_dataset_hybrid = {r["dataset"]: r for r in summary if r.get("method") == "AFU-0-Hybrid-alphaFixed1"}
    out: List[Dict[str, Any]] = []
    for row in summary:
        method = str(row.get("method", ""))
        if not method.startswith("AFU-") or method == "AFU-0-Hybrid-alphaFixed1":
            continue
        hybrid = by_dataset_hybrid.get(row.get("dataset"))
        if not hybrid:
            continue
        acc_gap_h = f(hybrid, "test_acc_mean") - f(row, "test_acc_mean")
        auc = f(row, "val_auc_improvement_vs_adamw_mean")
        no_kan = f(row, "no_kan_drop_over_adamw_mean")
        branch = f(row, "branch_over_adamw_mean")
        failure = ""
        if acc_gap_h > 0.015:
            failure = "nonkan_underfit"
        elif auc < 0.05:
            failure = "no_convergence_gain"
        elif no_kan < 0.70:
            failure = "expression_incomplete"
        elif branch > 1.0:
            failure = "branch_overactive"
        if failure:
            out.append(
                {
                    "dataset": row.get("dataset"),
                    "method": method,
                    "failure_type": failure,
                    "acc_gap_vs_hybrid": acc_gap_h,
                    "auc_improvement": auc,
                    "no_kan_ratio": no_kan,
                    "branch_over_adamw": branch,
                }
            )
    return out


def main() -> int:
    p0_rows = add_baseline_comparisons(read_csv(P0 / "runs.csv"))
    p2_rows = add_baseline_comparisons(read_csv(P2 / "runs.csv"))
    p3_rows = add_baseline_comparisons(read_csv(P3 / "runs.csv"))
    p4_rows = add_baseline_comparisons(read_csv(P4 / "runs.csv"))
    p5_rows = add_baseline_comparisons(read_csv(P5 / "runs.csv"))

    p1_dir = ensure_dir(ROOT / "results/gafu_v3_4_p1_direction_audit")
    p7_dir = ensure_dir(ROOT / "results/gafu_v3_4_p7_mechanism")
    p8_dir = ensure_dir(ROOT / "results/gafu_v3_4_p8_wallclock")

    p0_summary = summarize_runs(p0_rows, ["dataset", "method", "epochs"])
    p2_summary = summarize_runs(p2_rows, ["dataset", "method", "epochs"])
    p3_summary = summarize_runs(p3_rows, ["dataset", "method", "epochs"])
    p4_summary = summarize_runs(p4_rows, ["dataset", "method", "epochs"])
    p5_summary = summarize_runs(p5_rows, ["dataset", "method", "epochs"])
    write_csv(P0 / "p0_scorecard.csv", p0_summary)
    write_csv(P2 / "p2_scorecard.csv", p2_summary)
    write_csv(P3 / "p3_scorecard.csv", p3_summary)
    write_csv(P4 / "p4_scorecard.csv", p4_summary)
    write_csv(P5 / "p5_scorecard.csv", p5_summary)

    p1_proxy = p1_proxy_rows(p2_rows)
    write_csv(p1_dir / "direction_proxy_summary.csv", p1_proxy)

    paired_vs_adamw: List[Dict[str, Any]] = []
    paired_vs_hybrid: List[Dict[str, Any]] = []
    for dataset in ["Fashion-MNIST", "KMNIST"]:
        paired_vs_adamw.extend(
            paired_delta_rows(
                p5_rows,
                dataset,
                "AdamW-alphaFixed1-alphaFixed1",
                ["AFU-0-Hybrid-alphaFixed1", "AFU-LNOnly-alphaFixed1", "AFU-2-HeadCov-alphaFixed1"],
            )
        )
        paired_vs_hybrid.extend(
            paired_delta_rows(
                p5_rows,
                dataset,
                "AFU-0-Hybrid-alphaFixed1",
                ["AFU-LNOnly-alphaFixed1", "AFU-2-HeadCov-alphaFixed1"],
            )
        )
    write_csv(P5 / "paired_delta_vs_adamw.csv", paired_vs_adamw)
    write_csv(P5 / "paired_delta_vs_hybrid.csv", paired_vs_hybrid)

    target_rows, target_summary = target_matched(p5_rows)
    write_csv(p8_dir / "target_rows.csv", target_rows)
    write_csv(p8_dir / "target_summary.csv", target_summary)

    p7_methods = {
        "AdamW-alphaFixed1-alphaFixed1",
        "AFU-0-Hybrid-alphaFixed1",
        "AFU-LNOnly-alphaFixed1",
        "AFU-2-HeadCov-alphaFixed1",
        "AFU-5-FullDiagLite-alphaFixed1",
        "AFU-6-FullKFACLite-alphaFixed1",
    }
    mechanism = [r for r in p3_summary + p5_summary if r.get("method") in p7_methods]
    write_csv(p7_dir / "mechanism_summary.csv", mechanism)
    failures = failure_rows(p3_summary + p5_summary)
    write_csv(ROOT / "results/gafu_v3_4_failure_table.csv", failures)

    maybe_plots([r for r in p5_summary if str(r.get("method", "")).startswith("AFU")], target_summary)

    decision = {
        "p0": "passed implementation smoke without run failures",
        "p2": "HeadCov is expression-positive but KMNIST underfits; LNOnly is safest component; StemOnly fails hard.",
        "p3": "FullDiagLite/FullKFACLite fail as true all-functional optimizers due non-KAN underfit.",
        "p4": "branch_final_scale 0.875/0.90 does not rescue KMNIST without accuracy/AUC loss.",
        "p5": "No AFU candidate passes P5 medium across both datasets; P6 not run.",
        "final": "Partial AFU is useful as analysis evidence, but hybrid remains the default optimizer.",
    }
    save_json(ROOT / "results/gafu_v3_4_decision.json", decision)
    save_json(p1_dir / "aggregate_summary.json", {"direction_proxy_rows": p1_proxy, "note": "Proxy direction audit from training update metrics; explicit shadow-step was not expanded after P2/P3 failures."})
    save_json(p7_dir / "aggregate_summary.json", {"mechanism_summary": mechanism, "failure_table": failures})
    save_json(p8_dir / "aggregate_summary.json", {"target_summary": target_summary})

    score_cols = [
        ("dataset", "dataset"),
        ("method", "method"),
        ("runs", "runs"),
        ("acc", "test_acc_mean"),
        ("acc std", "test_acc_std"),
        ("gap", "acc_gap_vs_adamw_mean"),
        ("AUC imp", "val_auc_improvement_vs_adamw_mean"),
        ("phi red", "phi_prime_reduction_vs_adamw_mean"),
        ("J red", "jac_reduction_vs_adamw_mean"),
        ("branch/A", "branch_over_adamw_mean"),
        ("ECE red", "ece_reduction_vs_adamw_mean"),
        ("noKAN", "no_kan_drop_over_adamw_mean"),
        ("time/A", "time_ratio_vs_adamw_mean"),
    ]
    paired_cols = [
        ("dataset", "dataset"),
        ("baseline", "baseline"),
        ("method", "method"),
        ("acc delta", "paired_acc_delta_mean"),
        ("acc CI lo", "paired_acc_delta_ci95_lo"),
        ("acc CI hi", "paired_acc_delta_ci95_hi"),
        ("AUC delta", "paired_auc_delta_mean"),
        ("AUC CI lo", "paired_auc_delta_ci95_lo"),
        ("AUC CI hi", "paired_auc_delta_ci95_hi"),
    ]
    target_cols = [
        ("dataset", "dataset"),
        ("method", "method"),
        ("runs", "runs"),
        ("relaxed loss reached", "reached_relaxed_loss_mean"),
        ("epochs relaxed loss", "epochs_to_relaxed_loss_mean"),
        ("time relaxed loss", "time_to_relaxed_loss_sec_mean"),
        ("step ms", "step_time_ms_mean"),
    ]

    doc = f"""# DG-KAN Optimizer v3.4 AFU 结果复盘

## P2 Single-Component Scorecard

{md_table(p2_summary, score_cols)}

## P3 Cumulative AFU Scorecard

{md_table(p3_summary, score_cols)}

## P4 Expression Micro-Check

{md_table(p4_summary, score_cols)}

## P5 Partial AFU Confirm

{md_table(p5_summary, score_cols)}

## Paired Delta vs AdamW

{md_table(paired_vs_adamw, paired_cols)}

## Paired Delta vs Hybrid

{md_table(paired_vs_hybrid, paired_cols)}

## Target-Matched / Wall-Clock

{md_table(target_summary, target_cols)}

## Decision

```text
P0:
  AFU implementation smoke passed without run failures.
  Group metrics were finite. Stem/full AFU already showed early underfit.

P2:
  AFU-2 HeadCov is expression-positive:
    Fashion noKAN and ECE improve strongly.
    KMNIST noKAN/ECE improve, but accuracy drops by about 2.1 points vs Hybrid.
  AFU-LNOnly is the safest component:
    small accuracy cost, positive AUC, geometry preserved.
  AFU-StemOnly fails hard on both datasets.

P3:
  AFU-5 FullDiagLite and AFU-6 FullKFACLite fail as true all-functional optimizers.
  Removing non-KAN AdamW causes strong non-KAN underfit:
    Fashion drops about 4.8-4.9 points vs Hybrid.
    KMNIST drops about 16.2 points vs Hybrid.

P4:
  KMNIST branch_final_scale 0.875 / 0.90 does not solve the strict expression gate.
  Higher final scale trades away accuracy/AUC.

P5:
  AFU-LNOnly does not pass medium gate:
    Fashion is close, but KMNIST is 0.82 points below Hybrid.
  AFU-2 HeadCov does not pass because KMNIST underfits despite better AUC/noKAN/ECE.

P6:
  Not run. No AFU candidate passed P5 strongly enough.

Final:
  True all-functional update is not viable with the current non-KAN metrics.
  Partial AFU is informative, especially HeadCov for expression/calibration analysis and LNOnly as a safe component, but neither replaces Hybrid.
  Hybrid remains the main optimizer:
    KAN coeff uses Sobolev functional update.
    non-KAN parameters still need AdamW.
```
"""
    doc_path = ROOT / "docs/DG-KAN_Optimizer_v3.4_AFU_结果复盘.md"
    doc_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {doc_path}")
    print(f"Wrote {P5 / 'p5_scorecard.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
