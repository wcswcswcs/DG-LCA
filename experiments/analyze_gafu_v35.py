#!/usr/bin/env python3
"""Aggregate DG-KAN Optimizer v3.5 U-FULL-f100 experiments."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgkan_core import add_baseline_comparisons, ensure_dir, read_csv, save_json, summarize_runs, write_csv


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results/gafu_v3_5_p0_config"
P1 = ROOT / "results/gafu_v3_5_p1_seed0"
P2 = ROOT / "results/gafu_v3_5_p2_confirm5"
P3 = ROOT / "results/gafu_v3_5_p3_confirm10"
P4 = ROOT / "results/gafu_v3_5_p4_small_data"
P5 = ROOT / "results/gafu_v3_5_p5_label_noise"
P6 = ROOT / "results/gafu_v3_5_p6_cifar_small"
P7 = ROOT / "results/gafu_v3_5_p7_speed"
P8 = ROOT / "results/gafu_v3_5_p8_rational"


def f(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_curve(text: Any) -> List[float]:
    if text in ("", None):
        return []
    out: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def method_rows(rows: Iterable[Dict[str, Any]], dataset: str, method: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("dataset") == dataset and r.get("method") == method]


def by_seed(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        try:
            out[int(row["seed"])] = row
        except (KeyError, TypeError, ValueError):
            pass
    return out


def bootstrap_ci(values: Sequence[float], *, n: int = 2000, seed: int = 0) -> Tuple[float, float]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = vals[rng.integers(0, vals.size, size=(n, vals.size))].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def md_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]], *, digits: int = 4) -> str:
    lines = ["| " + " | ".join(title for title, _ in columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells: List[str] = []
        for _, key in columns:
            value = row.get(key, "")
            try:
                cells.append(f"{float(value):.{digits}f}")
            except (TypeError, ValueError):
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_delta_rows(rows: List[Dict[str, Any]], dataset: str, baseline: str, methods: List[str]) -> List[Dict[str, Any]]:
    base = by_seed(method_rows(rows, dataset, baseline))
    out: List[Dict[str, Any]] = []
    for method in methods:
        current = by_seed(method_rows(rows, dataset, method))
        seeds = sorted(set(base) & set(current))
        acc_delta = [f(current[s], "test_acc") - f(base[s], "test_acc") for s in seeds]
        auc_delta = [f(base[s], "val_auc") - f(current[s], "val_auc") for s in seeds]
        no_kan_delta = [f(current[s], "no_kan_drop_over_adamw") for s in seeds]
        acc_lo, acc_hi = bootstrap_ci(acc_delta)
        auc_lo, auc_hi = bootstrap_ci(auc_delta)
        no_lo, no_hi = bootstrap_ci(no_kan_delta)
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
                "paired_noKAN_ratio_mean": float(np.mean(no_kan_delta)) if no_kan_delta else float("nan"),
                "paired_noKAN_ratio_ci95_lo": no_lo,
                "paired_noKAN_ratio_ci95_hi": no_hi,
            }
        )
    return out


def first_reach(curve: List[float], target: float, *, mode: str) -> int:
    for idx, value in enumerate(curve, start=1):
        if (mode == "loss" and value <= target) or (mode == "acc" and value >= target):
            return idx
    return -1


def target_matched(rows: List[Dict[str, Any]], methods_by_dataset: Dict[str, List[str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
                steps_per_epoch = math.ceil(f(row, "train_size") / max(1.0, f(row, "batch_size", 256.0)))
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


def mean_group(rows: List[Dict[str, Any]], group_keys: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k, "") for k in group_keys), []).append(row)
    out: List[Dict[str, Any]] = []
    for key, group in groups.items():
        item: Dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        item["runs"] = len(group)
        for metric in metrics:
            vals = [f(row, metric) for row in group if math.isfinite(f(row, metric))]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                item[f"{metric}_mean"] = float(arr.mean())
                item[f"{metric}_std"] = float(arr.std(ddof=0))
        out.append(item)
    return sorted(out, key=lambda r: tuple(str(r.get(k, "")) for k in group_keys))


def p0_check(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    ok = True
    for row in rows:
        if row.get("method") != "U-FULL-f100-alphaFixed1":
            continue
        item = {
            "dataset": row.get("dataset"),
            "method": row.get("method"),
            "branch_final_scale": f(row, "branch_final_scale"),
            "alpha_trainable": f(row, "alpha_trainable"),
            "alpha_final_mean": f(row, "alpha_final_mean"),
            "v3_metric_active": row.get("v3_metric_active"),
            "v3_metric_geometry": row.get("v3_metric_geometry"),
            "nonkan_update": row.get("nonkan_update"),
            "trust_clip_rate": f(row, "trust_clip_rate"),
            "v3_phase_final": row.get("v3_phase_final"),
            "v3_metric_mix_auc": f(row, "v3_metric_mix_auc"),
        }
        item["pass"] = (
            abs(item["branch_final_scale"] - 1.0) < 1e-9
            and int(item["alpha_trainable"]) == 0
            and abs(item["alpha_final_mean"] - 1.0) < 1e-6
            and item["v3_metric_active"] == "full_sobolev_gram"
            and item["v3_metric_geometry"] == "full_sobolev_gram"
            and item["nonkan_update"] == "adamw"
            and item["trust_clip_rate"] <= 0.01
        )
        ok = ok and bool(item["pass"])
        checks.append(item)
    return {"pass": ok and bool(checks), "checks": checks}


def f100_clean_pass(summary: List[Dict[str, Any]], *, level: str) -> Tuple[bool, Dict[str, str]]:
    needed_auc = 0.05 if level == "p1" else 0.08
    needed_phi = 0.20 if level == "p1" else 0.25
    needed_j = 0.20
    needed_ece = -1.0 if level == "p1" else 0.10
    status: Dict[str, str] = {}
    ok = True
    for dataset in ["Fashion-MNIST", "KMNIST"]:
        row = next((r for r in summary if r.get("dataset") == dataset and r.get("method") == "U-FULL-f100-alphaFixed1"), None)
        if not row:
            status[dataset] = "missing"
            ok = False
            continue
        gap = f(row, "acc_gap_vs_adamw_mean")
        auc = f(row, "val_auc_improvement_vs_adamw_mean")
        phi = f(row, "phi_prime_reduction_vs_adamw_mean")
        jac = f(row, "jac_reduction_vs_adamw_mean")
        branch = f(row, "branch_over_adamw_mean")
        ece = f(row, "ece_reduction_vs_adamw_mean")
        no_kan = f(row, "no_kan_drop_over_adamw_mean")
        acc_ok = gap <= (0.005 if dataset == "Fashion-MNIST" else 0.010)
        if level != "p1":
            acc_ok = gap <= (0.0 if dataset == "Fashion-MNIST" else 0.005)
        row_ok = acc_ok and auc > needed_auc and phi > needed_phi and jac > needed_j and ece > needed_ece
        if level != "p1":
            row_ok = row_ok and no_kan > 0.70
        if not (0.5 < branch < 0.95):
            row_ok = row_ok and bool(level == "p1" and branch <= 1.1)
        status[dataset] = "pass" if row_ok else (
            f"fail gap={gap:.4f} auc={auc:.4f} phi={phi:.3f} J={jac:.3f} "
            f"branch={branch:.3f} ece={ece:.3f} noKAN={no_kan:.3f}"
        )
        ok = ok and row_ok
    return ok, status


def choose_selected_profile(p2_summary: List[Dict[str, Any]]) -> str:
    p2_ok, _ = f100_clean_pass(p2_summary, level="p2")
    return "U-FULL-f100-alphaFixed1" if p2_ok else "U-FULL-f085-reference-alphaFixed1"


def phase_status(rows: List[Dict[str, Any]], *, label: str) -> str:
    if not rows:
        return f"{label}: not run"
    errors = [str(row.get("error", "")) for row in rows if str(row.get("error", ""))]
    if len(errors) == len(rows):
        return f"{label}: all failed ({errors[0][:160]})"
    if errors:
        return f"{label}: partial, {len(rows) - len(errors)}/{len(rows)} successful"
    return f"{label}: {len(rows)} successful rows"


def best_by_acc(summary: List[Dict[str, Any]], dataset: str, prefix: str = "") -> Dict[str, Any] | None:
    rows = [
        row
        for row in summary
        if row.get("dataset") == dataset
        and (not prefix or str(row.get("method", "")).startswith(prefix))
        and math.isfinite(f(row, "test_acc_mean"))
    ]
    return max(rows, key=lambda row: f(row, "test_acc_mean"), default=None)


def p6_decision(summary: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> str:
    status = phase_status(rows, label="P6 CIFAR-small")
    if "successful" not in status:
        return status
    base = best_by_acc(summary, "CIFAR10", "ConvStem-DGKAN-AdamW")
    f085 = best_by_acc(summary, "CIFAR10", "ConvStem-DGKAN-U-FULL-f085")
    f100 = best_by_acc(summary, "CIFAR10", "ConvStem-DGKAN-U-FULL-f100")
    parts = [status]
    if base and f085:
        parts.append(
            f"f085 acc={f(f085, 'test_acc_mean'):.4f}, gap_vs_conv_adamw={f(f085, 'acc_gap_vs_adamw_mean'):.4f}, "
            f"AUC_imp={f(f085, 'val_auc_improvement_vs_adamw_mean'):.4f}"
        )
    if base and f100:
        parts.append(
            f"f100 acc={f(f100, 'test_acc_mean'):.4f}, gap_vs_conv_adamw={f(f100, 'acc_gap_vs_adamw_mean'):.4f}, "
            f"AUC_imp={f(f100, 'val_auc_improvement_vs_adamw_mean'):.4f}"
        )
    return "; ".join(parts)


def p8_decision(summary: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> str:
    status = phase_status(rows, label="P8 Rational/KAT")
    if "successful" not in status:
        return status
    parts = [status]
    for dataset in ["Fashion-MNIST", "KMNIST"]:
        rat = best_by_acc(summary, dataset, "Rational-DGKAN-AdamW-triton")
        torch_rat = best_by_acc(summary, dataset, "Rational-DGKAN-AdamW-torch")
        rat_ufull = best_by_acc(summary, dataset, "Rational-DGKAN-U-FULL-triton")
        torch_ufull = best_by_acc(summary, dataset, "Rational-DGKAN-U-FULL-torch")
        if rat:
            parts.append(
                f"{dataset} triton acc={f(rat, 'test_acc_mean'):.4f}, gap_vs_mlp={f(rat, 'acc_gap_vs_adamw_mean'):.4f}, "
                f"time/MLP={f(rat, 'time_ratio_vs_adamw_mean'):.3f}, denom_min={f(rat, 'rational_denominator_min_mean'):.3f}"
            )
        if torch_rat:
            parts.append(f"{dataset} torch acc={f(torch_rat, 'test_acc_mean'):.4f}, time/MLP={f(torch_rat, 'time_ratio_vs_adamw_mean'):.3f}")
        if rat_ufull:
            parts.append(
                f"{dataset} U-FULL triton acc={f(rat_ufull, 'test_acc_mean'):.4f}, "
                f"AUC_imp={f(rat_ufull, 'val_auc_improvement_vs_adamw_mean'):.4f}, "
                f"condG={f(rat_ufull, 'v3_metric_condition_geometry_mean'):.1f}, "
                f"clip={f(rat_ufull, 'trust_clip_rate_mean'):.3f}, "
                f"denom_min={f(rat_ufull, 'rational_denominator_min_mean'):.3f}"
            )
        if torch_ufull:
            parts.append(
                f"{dataset} U-FULL torch acc={f(torch_ufull, 'test_acc_mean'):.4f}, "
                f"AUC_imp={f(torch_ufull, 'val_auc_improvement_vs_adamw_mean'):.4f}"
            )
    return "; ".join(parts)


def failure_rows(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in summary:
        method = str(row.get("method", ""))
        if method.lower().startswith("adamw"):
            continue
        failures: List[str] = []
        if f(row, "acc_gap_vs_adamw_mean") > 0.01:
            failures.append("accuracy_gap_large")
        if f(row, "val_auc_improvement_vs_adamw_mean") < 0:
            failures.append("auc_no_gain")
        if f(row, "no_kan_drop_over_adamw_mean") < 0.70:
            failures.append("expression_low")
        if f(row, "branch_over_adamw_mean") > 1.10:
            failures.append("branch_overactive")
        if f(row, "branch_over_adamw_mean") < 0.45:
            failures.append("branch_underactive")
        if f(row, "phi_prime_reduction_vs_adamw_mean") < 0.20 or f(row, "jac_reduction_vs_adamw_mean") < 0.20:
            failures.append("geometry_bad")
        if f(row, "ece_reduction_vs_adamw_mean") < 0:
            failures.append("ece_worse")
        if f(row, "time_ratio_vs_adamw_mean") > 1.5 and f(row, "val_auc_improvement_vs_adamw_mean") < 0.08:
            failures.append("wallclock_slow")
        for failure in failures:
            out.append(
                {
                    "dataset": row.get("dataset"),
                    "method": method,
                    "train_size": row.get("train_size", ""),
                    "label_noise": row.get("label_noise", ""),
                    "failure_type": failure,
                    "acc_gap": f(row, "acc_gap_vs_adamw_mean"),
                    "auc_imp": f(row, "val_auc_improvement_vs_adamw_mean"),
                    "phi_red": f(row, "phi_prime_reduction_vs_adamw_mean"),
                    "j_red": f(row, "jac_reduction_vs_adamw_mean"),
                    "branch_over_adamw": f(row, "branch_over_adamw_mean"),
                    "ece_red": f(row, "ece_reduction_vs_adamw_mean"),
                    "no_kan_ratio": f(row, "no_kan_drop_over_adamw_mean"),
                }
            )
    return out


def simple_plots(rows: List[Dict[str, Any]], small_summary: List[Dict[str, Any]], noise_summary: List[Dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    plot_dir = ensure_dir(ROOT / "results/gafu_v3_5_plots")

    def scatter(path: Path, pts: List[Tuple[float, float, str]], title: str, xlab: str, ylab: str) -> None:
        pts = [(x, y, label) for x, y, label in pts if math.isfinite(x) and math.isfinite(y)]
        if not pts:
            return
        w, h, pad = 920, 560, 70
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
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

        for x, y, label in pts:
            cx, cy = sx(x), sy(y)
            color = (45, 100, 190) if "Fashion" in label else (200, 80, 45)
            draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=color, outline="black")
            draw.text((cx + 8, cy - 7), label[:24], fill=color)
        draw.text((pad, 20), title, fill="black")
        draw.text((w // 2 - 50, h - 36), xlab, fill="black")
        draw.text((10, h // 2), ylab, fill="black")
        img.save(path)

    scatter(
        plot_dir / "clean_acc_vs_auc.png",
        [
            (f(r, "test_acc_mean"), f(r, "val_auc_improvement_vs_adamw_mean"), f"{r.get('dataset')} {str(r.get('method'))[:14]}")
            for r in rows
        ],
        "clean acc vs AUC improvement",
        "test_acc",
        "AUC improvement",
    )
    scatter(
        plot_dir / "small_data_acc_curve.png",
        [
            (f(r, "train_size"), f(r, "test_acc_mean"), f"{r.get('dataset')} {str(r.get('method'))[:12]}")
            for r in small_summary
        ],
        "small-data accuracy",
        "train_size",
        "test_acc",
    )
    scatter(
        plot_dir / "noise_ece_bar.png",
        [
            (f(r, "label_noise"), f(r, "ece_mean"), f"{r.get('dataset')} {str(r.get('method'))[:12]}")
            for r in noise_summary
        ],
        "label-noise ECE",
        "noise",
        "ECE",
    )


def main() -> int:
    p0_rows = add_baseline_comparisons(read_csv(P0 / "runs.csv"))
    p1_rows = add_baseline_comparisons(read_csv(P1 / "runs.csv"))
    p2_rows = add_baseline_comparisons(read_csv(P2 / "runs.csv"))
    p3_rows = add_baseline_comparisons(read_csv(P3 / "runs.csv"))
    p4_rows = add_baseline_comparisons(read_csv(P4 / "runs.csv"))
    p5_rows = add_baseline_comparisons(read_csv(P5 / "runs.csv"))
    p6_rows = add_baseline_comparisons(read_csv(P6 / "runs.csv"))
    p8_rows = add_baseline_comparisons(read_csv(P8 / "runs.csv"))

    for path in [P0, P1, P2, P3, P4, P5, P6, P7, P8]:
        ensure_dir(path)

    p0_summary = summarize_runs(p0_rows, ["dataset", "method", "epochs"])
    p1_summary = summarize_runs(p1_rows, ["dataset", "method", "epochs"])
    p2_summary = summarize_runs(p2_rows, ["dataset", "method", "epochs"])
    p3_summary = summarize_runs(p3_rows, ["dataset", "method", "epochs"])
    p4_summary = summarize_runs(p4_rows, ["dataset", "method", "train_size", "epochs"])
    p5_summary = summarize_runs(p5_rows, ["dataset", "method", "label_noise", "epochs"])
    p6_summary = summarize_runs(p6_rows, ["dataset", "method", "train_size", "epochs"])
    p8_summary = summarize_runs(p8_rows, ["dataset", "method", "epochs"])

    write_csv(P0 / "p0_scorecard.csv", p0_summary)
    write_csv(P1 / "p1_scorecard.csv", p1_summary)
    write_csv(P2 / "p2_scorecard.csv", p2_summary)
    write_csv(P3 / "p3_scorecard.csv", p3_summary)
    write_csv(P4 / "p4_scorecard.csv", p4_summary)
    write_csv(P5 / "p5_scorecard.csv", p5_summary)
    write_csv(P6 / "p6_scorecard.csv", p6_summary)
    write_csv(P8 / "p8_scorecard.csv", p8_summary)

    p0 = p0_check(p0_rows)
    p1_ok, p1_status = f100_clean_pass(p1_summary, level="p1")
    p2_ok, p2_status = f100_clean_pass(p2_summary, level="p2")
    selected = choose_selected_profile(p2_summary)

    paired_p2: List[Dict[str, Any]] = []
    for dataset in ["Fashion-MNIST", "KMNIST"]:
        paired_p2.extend(
            paired_delta_rows(
                p2_rows,
                dataset,
                "AdamW-alphaFixed1-alphaFixed1",
                [
                    "U-FULL-f085-reference-alphaFixed1",
                    "U-FULL-f100-alphaFixed1",
                    "F-V3-HARD-base30-v32best-alphaFixed1"
                    if dataset == "Fashion-MNIST"
                    else "K-V3-FULL-base-v32best-alphaFixed1",
                ],
            )
        )
    write_csv(P2 / "paired_delta.csv", paired_p2)

    target_rows, target_summary = target_matched(
        p2_rows if not p3_rows else p3_rows,
        {
            "Fashion-MNIST": [
                "AdamW-alphaFixed1-alphaFixed1",
                "U-FULL-f085-reference-alphaFixed1",
                "U-FULL-f100-alphaFixed1",
                "F-V3-HARD-base30-v32best-alphaFixed1",
            ],
            "KMNIST": [
                "AdamW-alphaFixed1-alphaFixed1",
                "U-FULL-f085-reference-alphaFixed1",
                "U-FULL-f100-alphaFixed1",
                "K-V3-FULL-base-v32best-alphaFixed1",
            ],
        },
    )
    write_csv(P7 / "target_rows.csv", target_rows)
    write_csv(P7 / "target_summary.csv", target_summary)

    noise_extra = mean_group(
        p5_rows,
        ["dataset", "method", "label_noise", "epochs"],
        [
            "train_acc_noisy_labels",
            "train_acc_clean_labels",
            "noise_memorization_rate",
            "corrupted_subset_train_acc_clean_label",
            "corrupted_subset_train_acc_noisy_label",
            "test_acc",
            "ece",
        ],
    )
    write_csv(P5 / "noise_memorization_summary.csv", noise_extra)

    all_summary = p1_summary + p2_summary + p3_summary + p4_summary + p5_summary + p6_summary + p8_summary
    failures = failure_rows(all_summary)
    write_csv(ROOT / "results/gafu_v3_5_failure_table.csv", failures)
    write_csv(P2 / "failure_table.csv", failure_rows(p2_summary))
    write_csv(P4 / "failure_table.csv", failure_rows(p4_summary))
    write_csv(P5 / "failure_table.csv", failure_rows(p5_summary))
    write_csv(P6 / "failure_table.csv", failure_rows(p6_summary))
    write_csv(P8 / "failure_table.csv", failure_rows(p8_summary))

    simple_plots(p2_summary, p4_summary, p5_summary)

    cifar_decision = p6_decision(p6_summary, p6_rows)
    rational_decision = p8_decision(p8_summary, p8_rows)
    speed_decision = "target-matched timing generated from P2/P3 rows; low-level kernel profiling not expanded"

    decision = {
        "p0_config_pass": p0["pass"],
        "p1_f100_pass": p1_ok,
        "p1_status": p1_status,
        "p2_f100_pass": p2_ok,
        "p2_status": p2_status,
        "selected_unified_profile": selected,
        "p3_trigger": "run P3" if p2_ok else "not triggered because P2 did not pass",
        "p3_rows": len(p3_rows),
        "p4_rows": len(p4_rows),
        "p5_rows": len(p5_rows),
        "p6_cifar": cifar_decision,
        "p7_speed": speed_decision,
        "p8_rational": rational_decision,
    }
    save_json(ROOT / "results/gafu_v3_5_decision.json", decision)
    save_json(P0 / "aggregate_summary.json", {"p0_check": p0, "summary": p0_summary})
    save_json(P1 / "aggregate_summary.json", {"summary": p1_summary, "p1_status": p1_status, "p1_f100_pass": p1_ok})
    save_json(P2 / "aggregate_summary.json", {"summary": p2_summary, "paired_delta": paired_p2, "p2_status": p2_status, "p2_f100_pass": p2_ok})
    save_json(P4 / "aggregate_summary.json", {"summary": p4_summary})
    save_json(P5 / "aggregate_summary.json", {"summary": p5_summary, "noise_memorization_summary": noise_extra})
    save_json(P6 / "aggregate_summary.json", {"summary": p6_summary, "decision": cifar_decision})
    save_json(P7 / "aggregate_summary.json", {"target_summary": target_summary, "note": speed_decision})
    save_json(P8 / "aggregate_summary.json", {"summary": p8_summary, "decision": rational_decision})

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
        ("noKAN", "paired_noKAN_ratio_mean"),
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
    noise_cols = [
        ("dataset", "dataset"),
        ("method", "method"),
        ("noise", "label_noise"),
        ("runs", "runs"),
        ("acc", "test_acc_mean"),
        ("AUC imp", "val_auc_improvement_vs_adamw_mean"),
        ("ECE", "ece_mean"),
        ("ECE red", "ece_reduction_vs_adamw_mean"),
        ("noKAN", "no_kan_drop_over_adamw_mean"),
    ]
    small_cols = [
        ("dataset", "dataset"),
        ("method", "method"),
        ("train", "train_size"),
        ("runs", "runs"),
        ("acc", "test_acc_mean"),
        ("gap", "acc_gap_vs_adamw_mean"),
        ("AUC imp", "val_auc_improvement_vs_adamw_mean"),
        ("phi red", "phi_prime_reduction_vs_adamw_mean"),
        ("J red", "jac_reduction_vs_adamw_mean"),
        ("branch/A", "branch_over_adamw_mean"),
        ("ECE red", "ece_reduction_vs_adamw_mean"),
        ("noKAN", "no_kan_drop_over_adamw_mean"),
        ("time/A", "time_ratio_vs_adamw_mean"),
    ]
    p8_cols = [
        ("dataset", "dataset"),
        ("method", "method"),
        ("runs", "runs"),
        ("acc", "test_acc_mean"),
        ("acc std", "test_acc_std"),
        ("gap vs MLP", "acc_gap_vs_adamw_mean"),
        ("AUC imp", "val_auc_improvement_vs_adamw_mean"),
        ("ECE red", "ece_reduction_vs_adamw_mean"),
        ("time/MLP", "time_ratio_vs_adamw_mean"),
        ("clip", "trust_clip_rate_mean"),
        ("condG", "v3_metric_condition_geometry_mean"),
        ("rawG", "update_raw_grad_norm_mean_mean"),
        ("dirN", "update_precond_direction_norm_mean_mean"),
        ("den min", "rational_denominator_min_mean"),
        ("den p01", "rational_denominator_p01_mean"),
        ("mem MB", "memory_peak_mb_mean"),
    ]

    doc = f"""# DG-KAN Optimizer v3.5 U-FULL f1 结果复盘

## P0 Config Freeze

```json
{json.dumps(p0, ensure_ascii=False, indent=2)}
```

## P1 Seed0 Sanity

{md_table(p1_summary, score_cols)}

P1 f100 pass: `{p1_ok}`

```json
{json.dumps(p1_status, ensure_ascii=False, indent=2)}
```

## P2 5-Seed Clean Confirm

{md_table(p2_summary, score_cols)}

P2 f100 pass: `{p2_ok}`

```json
{json.dumps(p2_status, ensure_ascii=False, indent=2)}
```

Selected unified profile: `{selected}`

## P2 Paired Delta vs AdamW

{md_table(paired_p2, paired_cols)}

## P3 10-Seed Final

Trigger decision: `{decision["p3_trigger"]}`

{md_table(p3_summary, score_cols) if p3_summary else "_P3 not run._"}

## P4 Small-Data Generalization

{md_table(p4_summary, small_cols)}

## P5 Label-Noise Generalization

{md_table(p5_summary, noise_cols)}

## P6 CIFAR-Small ConvStem Precheck

{md_table(p6_summary, small_cols)}

```text
{cifar_decision}
```

## P7 Target-Matched / Speed Proxy

{md_table(target_summary, target_cols)}

## P8 Rational/KAT Precheck

{md_table(p8_summary, p8_cols)}

```text
{rational_decision}
```

## Decision

```text
P0:
  config check pass = {p0["pass"]}

P1:
  U-FULL-f100 seed0 sanity pass = {p1_ok}

P2:
  U-FULL-f100 5-seed clean pass = {p2_ok}
  selected unified profile = {selected}

P3:
  {decision["p3_trigger"]}

P4/P5:
  See scorecards above for small-data and label-noise behavior.

P7:
  Target-matched wall-clock proxy was generated from clean confirm rows.
  Low-level kernel profiling and solve microbenchmarks were not expanded in this run.

P6/P8:
  CIFAR-small and Rational/KAT prechecks were included when their rows are present.
  Rational uses third_party/rational_kat_cu directly, with Triton autograd or the PyTorch fallback.
  No custom backward rewrite was accepted in this round.
```
"""
    doc_path = ROOT / "docs/DG-KAN_Optimizer_v3.5_UFULL_f1_结果复盘.md"
    doc_path.write_text(doc, encoding="utf-8")

    log_path = ROOT / "docs/log.md"
    marker = "## 2026-05-02 DG-KAN Optimizer v3.5 U-FULL f1 实验"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip()
    with log_path.open("w", encoding="utf-8") as f_log:
        if existing:
            f_log.write(existing)
            f_log.write("\n\n")
        f_log.write(marker)
        f_log.write("\n\n")
        f_log.write(doc)

    print(f"Wrote {doc_path}")
    print(f"Wrote {ROOT / 'results/gafu_v3_5_decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
