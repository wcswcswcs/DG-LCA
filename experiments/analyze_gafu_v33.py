#!/usr/bin/env python3
"""Aggregate DG-KAN Optimizer v3.3 staged experiment results."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgkan_core import add_baseline_comparisons, ensure_dir, read_csv, save_json, summarize_runs, write_csv


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results/gafu_v3_3_p0_smoke"
P1 = ROOT / "results/gafu_v3_3_p1_ufull_sanity_a64"
P2 = ROOT / "results/gafu_v3_3_p2_ufull_expr3"
P3 = ROOT / "results/gafu_v3_3_p3_uo_smoke"
P5 = ROOT / "results/gafu_v3_3_p5_confirm5"
P6 = ROOT / "results/gafu_v3_3_p6_confirm10"


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


def paired_delta_rows(rows: List[Dict[str, Any]], dataset: str, methods: List[str]) -> List[Dict[str, Any]]:
    adam = by_seed(method_rows(rows, dataset, methods[0]))
    out: List[Dict[str, Any]] = []
    for method in methods[1:]:
        current = by_seed(method_rows(rows, dataset, method))
        seeds = sorted(set(adam) & set(current))
        acc_delta = [f(current[s], "test_acc") - f(adam[s], "test_acc") for s in seeds]
        auc_delta = [f(adam[s], "val_auc") - f(current[s], "val_auc") for s in seeds]
        no_kan_ratio = [f(current[s], "no_kan_drop_over_adamw") for s in seeds]
        acc_lo, acc_hi = bootstrap_ci(acc_delta)
        auc_lo, auc_hi = bootstrap_ci(auc_delta)
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "seeds": ",".join(str(s) for s in seeds),
                "paired_acc_delta_mean": float(np.mean(acc_delta)) if acc_delta else float("nan"),
                "paired_acc_delta_ci95_lo": acc_lo,
                "paired_acc_delta_ci95_hi": acc_hi,
                "paired_auc_delta_mean": float(np.mean(auc_delta)) if auc_delta else float("nan"),
                "paired_auc_delta_ci95_lo": auc_lo,
                "paired_auc_delta_ci95_hi": auc_hi,
                "no_kan_drop_over_adamw_mean": float(np.mean(no_kan_ratio)) if no_kan_ratio else float("nan"),
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
            "F-V3-HARD-base30-v32best-alphaFixed1",
            "U-FULL-f085-alphaFixed1",
        ],
        "KMNIST": [
            "AdamW-alphaFixed1-alphaFixed1",
            "K-V3-FULL-base-v32best-alphaFixed1",
            "U-FULL-f085-alphaFixed1",
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
                adam_acc_curve = parse_curve(adam[seed].get("val_acc_curve"))
                curve = parse_curve(row.get("val_loss_curve"))
                acc_curve = parse_curve(row.get("val_acc_curve"))
                if not curve or not adam_loss_curve:
                    continue
                steps_per_epoch = math.ceil(f(row, "train_size") / f(row, "batch_size", 256.0))
                step_time_sec = f(row, "step_time_ms") / 1000.0
                adam_final_loss = adam_loss_curve[-1]
                adam_final_acc = adam_acc_curve[-1] if adam_acc_curve else f(adam[seed], "val_acc")
                relaxed_loss = 1.05 * adam_final_loss
                relaxed_acc = adam_final_acc - 0.005
                loss_epoch = first_reach(curve, adam_final_loss, mode="loss")
                relaxed_loss_epoch = first_reach(curve, relaxed_loss, mode="loss")
                acc_epoch = first_reach(acc_curve, adam_final_acc, mode="acc")
                relaxed_acc_epoch = first_reach(acc_curve, relaxed_acc, mode="acc")

                def epoch_to_steps(epoch: int) -> float:
                    return float(epoch * steps_per_epoch) if epoch > 0 else float("nan")

                out.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "reached_adamw_final_loss": int(loss_epoch > 0),
                        "epochs_to_adamw_final_loss": loss_epoch,
                        "time_to_adamw_final_loss_sec": epoch_to_steps(loss_epoch) * step_time_sec if loss_epoch > 0 else float("nan"),
                        "reached_relaxed_loss": int(relaxed_loss_epoch > 0),
                        "epochs_to_relaxed_loss": relaxed_loss_epoch,
                        "time_to_relaxed_loss_sec": epoch_to_steps(relaxed_loss_epoch) * step_time_sec if relaxed_loss_epoch > 0 else float("nan"),
                        "reached_adamw_final_acc": int(acc_epoch > 0),
                        "epochs_to_adamw_final_acc": acc_epoch,
                        "time_to_adamw_final_acc_sec": epoch_to_steps(acc_epoch) * step_time_sec if acc_epoch > 0 else float("nan"),
                        "reached_relaxed_acc": int(relaxed_acc_epoch > 0),
                        "epochs_to_relaxed_acc": relaxed_acc_epoch,
                        "time_to_relaxed_acc_sec": epoch_to_steps(relaxed_acc_epoch) * step_time_sec if relaxed_acc_epoch > 0 else float("nan"),
                        "step_time_ms": f(row, "step_time_ms"),
                    }
                )
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in out:
        groups.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)
    metric_keys = [
        "reached_adamw_final_loss",
        "epochs_to_adamw_final_loss",
        "time_to_adamw_final_loss_sec",
        "reached_relaxed_loss",
        "epochs_to_relaxed_loss",
        "time_to_relaxed_loss_sec",
        "reached_adamw_final_acc",
        "epochs_to_adamw_final_acc",
        "time_to_adamw_final_acc_sec",
        "reached_relaxed_acc",
        "epochs_to_relaxed_acc",
        "time_to_relaxed_acc_sec",
        "step_time_ms",
    ]
    summary: List[Dict[str, Any]] = []
    for (dataset, method), group in groups.items():
        item: Dict[str, Any] = {"dataset": dataset, "method": method, "runs": len(group)}
        for key in metric_keys:
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


def maybe_plots(expression_rows: List[Dict[str, Any]], uo_rows: List[Dict[str, Any]], target_summary: List[Dict[str, Any]]) -> None:
    p7 = ensure_dir(ROOT / "results/gafu_v3_3_p7_expression")
    p8 = ensure_dir(ROOT / "results/gafu_v3_3_p8_uo_moment")
    p9 = ensure_dir(ROOT / "results/gafu_v3_3_p9_wallclock")
    try:
        import matplotlib.pyplot as plt
    except Exception:
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return

        def scatter_png(path: Path, rows: List[Dict[str, Any]], x_key: str, y_key: str, *, hline: float | None = None) -> None:
            w, h = 900, 560
            pad = 70
            points = [(f(row, x_key), f(row, y_key), str(row.get("dataset", "")), str(row.get("method", ""))) for row in rows]
            points = [(x, y, ds, m) for x, y, ds, m in points if math.isfinite(x) and math.isfinite(y)]
            if not points:
                return
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
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
            for x, y, ds, method in points:
                cx, cy = sx(x), sy(y)
                color = colors.get(ds, (80, 80, 80))
                draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=color, outline="black")
                draw.text((cx + 9, cy - 7), method[:18], fill=color)
            draw.text((pad, 20), f"{y_key} vs {x_key}", fill="black")
            draw.text((w // 2 - 80, h - 40), x_key, fill="black")
            draw.text((10, h // 2), y_key, fill="black")
            img.save(path)

        def bar_png(path: Path, rows: List[Dict[str, Any]], key: str) -> None:
            w, h = 1000, 560
            pad = 70
            vals = [f(row, key) for row in rows if math.isfinite(f(row, key))]
            if not vals:
                return
            img = Image.new("RGB", (w, h), "white")
            draw = ImageDraw.Draw(img)
            max_v = max(vals) * 1.15
            bar_w = max(8, int((w - 2 * pad) / max(1, len(rows)) * 0.65))
            for i, row in enumerate(rows):
                value = f(row, key)
                if not math.isfinite(value):
                    continue
                x = pad + i * (w - 2 * pad) / max(1, len(rows))
                y = h - pad - value / max(1e-9, max_v) * (h - 2 * pad)
                draw.rectangle((x, y, x + bar_w, h - pad), fill=(70, 130, 180), outline="black")
                draw.text((x, h - pad + 8), str(row.get("method", ""))[:12], fill="black")
            draw.line((pad, h - pad, w - pad, h - pad), fill="black")
            draw.line((pad, pad, pad, h - pad), fill="black")
            draw.text((pad, 20), key, fill="black")
            img.save(path)

        scatter_png(
            p7 / "branch_vs_no_kan.png",
            [r for r in expression_rows if not str(r.get("method", "")).startswith("AdamW")],
            "branch_over_adamw_mean",
            "no_kan_drop_over_adamw_mean",
            hline=0.7,
        )
        scatter_png(
            p8 / "moment_amp_vs_branch.png",
            [r for r in uo_rows if "UO" in str(r.get("method", ""))],
            "optimizer_moment_amplification_ratio_mean_mean",
            "branch_over_adamw_mean",
            hline=1.1,
        )
        bar_png(p9 / "time_to_relaxed_loss.png", target_summary, "time_to_relaxed_loss_sec_mean")
        return

    plt.figure(figsize=(6, 4))
    for row in expression_rows:
        if str(row.get("method", "")).startswith("AdamW"):
            continue
        plt.scatter(f(row, "branch_over_adamw_mean"), f(row, "no_kan_drop_over_adamw_mean"), s=70, label=f"{row['dataset']} {row['method'][:14]}")
    plt.axhline(0.7, color="black", linestyle="--", linewidth=1)
    plt.xlabel("branch_over_adamw")
    plt.ylabel("no_kan_drop_over_adamw")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(p7 / "branch_vs_no_kan.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    for row in uo_rows:
        if "UO" not in str(row.get("method", "")):
            continue
        plt.scatter(f(row, "optimizer_moment_amplification_ratio_mean_mean"), f(row, "branch_over_adamw_mean"), s=70, label=f"{row['dataset']} {row['method'][:14]}")
    plt.axhline(1.1, color="black", linestyle="--", linewidth=1)
    plt.xlabel("moment amplification")
    plt.ylabel("branch_over_adamw")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(p8 / "moment_amp_vs_branch.png", dpi=160)
    plt.close()

    labels = [f"{r['dataset']}\n{str(r['method'])[:16]}" for r in target_summary]
    vals = [f(r, "time_to_relaxed_loss_sec_mean") for r in target_summary]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("time_to_relaxed_loss_sec")
    plt.tight_layout()
    plt.savefig(p9 / "time_to_relaxed_loss.png", dpi=160)
    plt.close()


def main() -> int:
    p0_rows = add_baseline_comparisons(read_csv(P0 / "runs.csv"))
    p1_rows = add_baseline_comparisons(read_csv(P1 / "runs.csv"))
    p2_rows = add_baseline_comparisons(read_csv(P2 / "runs.csv"))
    p3_rows = add_baseline_comparisons(read_csv(P3 / "runs.csv"))
    p5_rows = add_baseline_comparisons(read_csv(P5 / "runs.csv"))
    p6_rows = add_baseline_comparisons(read_csv(P6 / "runs.csv"))

    p7_dir = ensure_dir(ROOT / "results/gafu_v3_3_p7_expression")
    p8_dir = ensure_dir(ROOT / "results/gafu_v3_3_p8_uo_moment")
    p9_dir = ensure_dir(ROOT / "results/gafu_v3_3_p9_wallclock")

    p6_summary = summarize_runs(p6_rows, ["dataset", "method", "epochs"])
    p5_summary = summarize_runs(p5_rows, ["dataset", "method", "epochs"])
    p2_summary = summarize_runs(p2_rows, ["dataset", "method", "epochs"])
    p3_summary = summarize_runs(p3_rows, ["dataset", "method", "epochs"])

    write_csv(P6 / "p6_final_scorecard.csv", p6_summary)
    write_csv(P5 / "p5_scorecard.csv", p5_summary)
    write_csv(P2 / "p2_scorecard.csv", p2_summary)
    write_csv(p8_dir / "uo_smoke_summary.csv", p3_summary)

    paired: List[Dict[str, Any]] = []
    paired.extend(
        paired_delta_rows(
            p6_rows,
            "Fashion-MNIST",
            [
                "AdamW-alphaFixed1-alphaFixed1",
                "F-V3-HARD-base30-v32best-alphaFixed1",
                "U-FULL-f085-alphaFixed1",
            ],
        )
    )
    paired.extend(
        paired_delta_rows(
            p6_rows,
            "KMNIST",
            [
                "AdamW-alphaFixed1-alphaFixed1",
                "K-V3-FULL-base-v32best-alphaFixed1",
                "U-FULL-f085-alphaFixed1",
            ],
        )
    )
    write_csv(P6 / "paired_delta_summary.csv", paired)

    target_rows, target_summary = target_matched(p6_rows)
    write_csv(p9_dir / "target_rows.csv", target_rows)
    write_csv(p9_dir / "target_summary.csv", target_summary)

    expression_rows = [
        row
        for row in p6_summary
        if row.get("method")
        in {
            "AdamW-alphaFixed1-alphaFixed1",
            "F-V3-HARD-base30-v32best-alphaFixed1",
            "K-V3-FULL-base-v32best-alphaFixed1",
            "U-FULL-f085-alphaFixed1",
        }
    ]
    write_csv(p7_dir / "expression_summary.csv", expression_rows)
    maybe_plots(expression_rows, p3_summary, target_summary)

    p0_pass = not any(row.get("error") for row in p0_rows)
    uo_failure = "UO methods quarantined at P3: branch_over_adamw and geometry spikes exceeded planned failure thresholds."
    decision = {
        "p0_no_errors": p0_pass,
        "p2_best": "U-FULL-f085-alphaFixed1",
        "p5_decision": "U-FULL-f085 passed 5-seed unified gate.",
        "p6_decision": "Unified full-start is confirmed as a cross-dataset Pareto profile; strict unified clean misses KMNIST no-KAN threshold by a small margin (0.696 < 0.700).",
        "uo_decision": uo_failure,
    }
    save_json(ROOT / "results/gafu_v3_3_decision.json", decision)
    save_json(p7_dir / "aggregate_summary.json", {"expression_summary": expression_rows})
    save_json(p8_dir / "aggregate_summary.json", {"uo_smoke_summary": p3_summary, "decision": uo_failure})
    save_json(p9_dir / "aggregate_summary.json", {"target_summary": target_summary})

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
        ("method", "method"),
        ("acc delta", "paired_acc_delta_mean"),
        ("acc CI lo", "paired_acc_delta_ci95_lo"),
        ("acc CI hi", "paired_acc_delta_ci95_hi"),
        ("AUC delta", "paired_auc_delta_mean"),
        ("AUC CI lo", "paired_auc_delta_ci95_lo"),
        ("AUC CI hi", "paired_auc_delta_ci95_hi"),
        ("noKAN", "no_kan_drop_over_adamw_mean"),
    ]
    uo_cols = [
        ("dataset", "dataset"),
        ("method", "method"),
        ("acc", "test_acc_mean"),
        ("AUC imp", "val_auc_improvement_vs_adamw_mean"),
        ("branch/A", "branch_over_adamw_mean"),
        ("phi red", "phi_prime_reduction_vs_adamw_mean"),
        ("J red", "jac_reduction_vs_adamw_mean"),
        ("clip", "trust_clip_rate_mean"),
        ("moment amp", "optimizer_moment_amplification_ratio_mean_mean"),
        ("moment cos", "optimizer_moment_cos_precond_current_mean_mean"),
        ("bad rate", "optimizer_bad_step_rate_mean"),
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

    doc = f"""# DG-KAN Optimizer v3.3 结果复盘

## Final Scorecard

{md_table(p6_summary, score_cols)}

## Paired Delta vs AdamW

{md_table(paired, paired_cols)}

## Unified Optimizer Smoke

{md_table(p3_summary, uo_cols)}

## Target-Matched / Wall-Clock

{md_table(target_summary, target_cols)}

## Decision

```text
P0:
  implementation smoke passed with no run failures and finite full-Sobolev metric statistics.

P1/P2:
  U-FULL-f085-alphaFixed1 was the best cross-dataset full-start candidate.
  In 3-seed P2 it passed Fashion and KMNIST gates, including no-KAN expression.

P3/P4:
  Unified AdamW-style optimizer variants are quarantined.
  PGAdam, NormPGAdam, and PostAdam produced branch over-activation and geometry spikes.
  PostAdam-trust030 was safer but still exceeded planned branch/trust/bad-step limits.
  P4 was therefore intentionally not expanded.

P5:
  U-FULL-f085-alphaFixed1 passed the 5-seed unified full-start gate.

P6:
  Fashion passes strict unified clean conditions.
  KMNIST matches/improves AdamW accuracy, improves AUC/geometry/ECE, and improves expression over v3.2 full-base.
  However KMNIST no-KAN ratio is 0.696, just below the strict 0.700 threshold.

Final:
  full_sobolev_gram from start + alphaFixed1 + branch_final_scale=0.85 is confirmed as the v3.3 unified Pareto profile.
  It is not claimed as strict unified clean default because KMNIST expression misses the no-KAN threshold by 0.004.
  Hybrid optimizer remains the mainline; unified AdamW-style moment dynamics are not accepted.

Wall-clock:
  True-Gram methods improve matched-step loss trajectories, but step time is about 1.37x-1.40x AdamW in this audit=64 run.
  Do not claim unconditional wall-clock faster convergence unless target_summary shows lower time-to-target.
```
"""
    doc_path = ROOT / "docs/DG-KAN_Optimizer_v3.3_结果复盘.md"
    doc_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {doc_path}")
    print(f"Wrote {P6 / 'p6_final_scorecard.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
