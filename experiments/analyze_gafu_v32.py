#!/usr/bin/env python3
"""Aggregate DG-KAN Optimizer v3.2 confirm and audit results."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgkan_core import add_baseline_comparisons, ensure_dir, read_csv, save_json, summarize_runs, write_csv


ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "results/gafu_v3_2_p2_fashion_confirm"
P3 = ROOT / "results/gafu_v3_2_p3_kmnist_confirm"
P5 = ROOT / "results/gafu_v3_2_p5_final_fixed"


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
        phi_delta = [f(adam[s], "phi_prime_p95") - f(current[s], "phi_prime_p95") for s in seeds]
        j_delta = [f(adam[s], "max_jac_condition") - f(current[s], "max_jac_condition") for s in seeds]
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
                "paired_phi_delta_mean": float(np.mean(phi_delta)) if phi_delta else float("nan"),
                "paired_j_delta_mean": float(np.mean(j_delta)) if j_delta else float("nan"),
            }
        )
    return out


def first_reach(curve: List[float], target: float, *, mode: str) -> int:
    if not curve:
        return -1
    for idx, value in enumerate(curve, start=1):
        if (mode == "loss" and value <= target) or (mode == "acc" and value >= target):
            return idx
    return -1


def target_matched(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    methods_by_dataset = {
        "Fashion-MNIST": [
            "AdamW-final-alphaFixed1",
            "GA-FU-v2-final-alphaFixed1",
            "F-V3-HARD-base30-final-alphaFixed1",
        ],
        "KMNIST": [
            "AdamW-final-alphaFixed1",
            "GA-FU-v2-final-alphaFixed1",
            "K-V3-FULL-base-final-alphaFixed1",
        ],
    }
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
                steps_per_epoch = math.ceil(f(row, "train_size") / 256.0)
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
                        "adam_final_loss": adam_final_loss,
                        "adam_final_acc": adam_final_acc,
                        "reached_adamw_final_loss": int(loss_epoch > 0),
                        "epochs_to_adamw_final_loss": loss_epoch,
                        "steps_to_adamw_final_loss": epoch_to_steps(loss_epoch),
                        "time_to_adamw_final_loss_sec": epoch_to_steps(loss_epoch) * step_time_sec if loss_epoch > 0 else float("nan"),
                        "reached_relaxed_loss": int(relaxed_loss_epoch > 0),
                        "epochs_to_relaxed_loss": relaxed_loss_epoch,
                        "steps_to_relaxed_loss": epoch_to_steps(relaxed_loss_epoch),
                        "time_to_relaxed_loss_sec": epoch_to_steps(relaxed_loss_epoch) * step_time_sec if relaxed_loss_epoch > 0 else float("nan"),
                        "reached_adamw_final_acc": int(acc_epoch > 0),
                        "epochs_to_adamw_final_acc": acc_epoch,
                        "steps_to_adamw_final_acc": epoch_to_steps(acc_epoch),
                        "time_to_adamw_final_acc_sec": epoch_to_steps(acc_epoch) * step_time_sec if acc_epoch > 0 else float("nan"),
                        "reached_relaxed_acc": int(relaxed_acc_epoch > 0),
                        "epochs_to_relaxed_acc": relaxed_acc_epoch,
                        "steps_to_relaxed_acc": epoch_to_steps(relaxed_acc_epoch),
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
        "steps_to_adamw_final_loss",
        "time_to_adamw_final_loss_sec",
        "reached_relaxed_loss",
        "epochs_to_relaxed_loss",
        "steps_to_relaxed_loss",
        "time_to_relaxed_loss_sec",
        "reached_adamw_final_acc",
        "epochs_to_adamw_final_acc",
        "steps_to_adamw_final_acc",
        "time_to_adamw_final_acc_sec",
        "reached_relaxed_acc",
        "epochs_to_relaxed_acc",
        "steps_to_relaxed_acc",
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
    summary = sorted(summary, key=lambda r: (str(r["dataset"]), str(r["method"])))
    return out, summary


def md_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]], *, digits: int = 4) -> str:
    lines = ["| " + " | ".join(title for title, _ in columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.{digits}f}")
            else:
                try:
                    cells.append(f"{float(value):.{digits}f}")
                except (TypeError, ValueError):
                    cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    p2_rows = add_baseline_comparisons(read_csv(P2 / "runs.csv"))
    p3_rows = add_baseline_comparisons(read_csv(P3 / "runs.csv"))
    p5_rows = add_baseline_comparisons(read_csv(P5 / "runs.csv"))
    p5_summary = summarize_runs(p5_rows, ["package", "dataset", "alpha_mode", "method", "label_noise", "train_size", "epochs"])

    p4_dir = ensure_dir(ROOT / "results/gafu_v3_2_p4_alpha_decision")
    p6_dir = ensure_dir(ROOT / "results/gafu_v3_2_p6_mechanism")
    p7_dir = ensure_dir(ROOT / "results/gafu_v3_2_p7_direction_audit")
    p8_dir = ensure_dir(ROOT / "results/gafu_v3_2_p8_target_matched")

    p4_summary = summarize_runs(p2_rows + p3_rows, ["dataset", "alpha_mode", "method", "epochs"])
    write_csv(p4_dir / "alpha_decision.csv", p4_summary)
    save_json(
        p4_dir / "aggregate_summary.json",
        {
            "decision": "fixed1_selected_for_P5; Fashion strong, KMNIST fixed1 full-base passed 5-seed medium but remains weaker on no-KAN ratio",
            "rows": p4_summary,
        },
    )

    mechanism_rows = [
        row
        for row in p5_summary
        if row.get("method")
        in {
            "AdamW-final-alphaFixed1",
            "GA-FU-v2-final-alphaFixed1",
            "F-V3-HARD-base30-final-alphaFixed1",
            "K-V3-FULL-base-final-alphaFixed1",
        }
    ]
    write_csv(p6_dir / "mechanism_summary.csv", mechanism_rows)
    save_json(
        p6_dir / "aggregate_summary.json",
        {
            "mechanism_rows": mechanism_rows,
            "notes": "Fashion v3 no-KAN drop ratio passes 0.7 AdamW; KMNIST v3 is below 0.7 and should be described as geometry/convergence evidence, not full expression proof.",
        },
    )

    direction_rows = [
        row
        for row in p5_summary
        if "V3" in str(row.get("method")) or "GA-FU" in str(row.get("method"))
    ]
    write_csv(p7_dir / "direction_summary.csv", direction_rows)
    save_json(
        p7_dir / "aggregate_summary.json",
        {
            "direction_rows": direction_rows,
            "notes": "Full true-Gram methods have positive AUC/geometry gains with no trust clipping; hard switch transition_loss_jump is negative on average for Fashion.",
        },
    )

    target_rows, target_summary = target_matched(p5_rows)
    write_csv(p8_dir / "target_by_run.csv", target_rows)
    write_csv(p8_dir / "target_summary.csv", target_summary)
    save_json(p8_dir / "aggregate_summary.json", {"target_summary": target_summary})

    paired = []
    paired.extend(
        paired_delta_rows(
            p5_rows,
            "Fashion-MNIST",
            [
                "AdamW-final-alphaFixed1",
                "GA-FU-v2-final-alphaFixed1",
                "F-V3-HARD-base30-final-alphaFixed1",
            ],
        )
    )
    paired.extend(
        paired_delta_rows(
            p5_rows,
            "KMNIST",
            [
                "AdamW-final-alphaFixed1",
                "GA-FU-v2-final-alphaFixed1",
                "K-V3-FULL-base-final-alphaFixed1",
            ],
        )
    )
    write_csv(P5 / "paired_delta_summary.csv", paired)

    doc = ROOT / "docs/DG-KAN_Optimizer_v3.2_结果复盘.md"
    score_rows = [
        row
        for row in p5_summary
        if row.get("method")
        in {
            "AdamW-final-alphaFixed1",
            "GA-FU-v2-final-alphaFixed1",
            "F-V3-HARD-base30-final-alphaFixed1",
            "K-V3-FULL-base-final-alphaFixed1",
        }
    ]
    columns = [
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
        ("noKAN", "no_kan_drop_mean"),
        ("margin", "kan_margin_contribution_mean_mean"),
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
    text = f"""# DG-KAN Optimizer v3.2 结果复盘

## Final Scorecard

{md_table(score_rows, columns)}

## Paired Delta vs AdamW

{md_table(paired, paired_cols)}

## Target-Matched / Wall-Clock

{md_table(target_summary, target_cols)}

## Decision

```text
Fashion:
  F-V3-HARD-base30-final-alphaFixed1 is confirmed over 10 seeds.
  It improves accuracy, validation-loss AUC, geometry, calibration, and keeps healthy branch utilization.

KMNIST:
  K-V3-FULL-base-final-alphaFixed1 is a geometry/convergence Pareto candidate.
  It approximately matches AdamW accuracy, improves AUC/J/ECE, but does not reach the stricter 9% AUC target and no-KAN ratio is below the 0.7 expression threshold.

Alpha mode:
  fixed1 is selected for P5 because it passed Fashion and reached the 5-seed KMNIST medium line.
  Final interpretation should still be dataset-specific: Fashion clean default, KMNIST Pareto/medium candidate.

Wall-clock:
  True-Gram methods improve matched-step loss trajectories, but step time is about 1.1x AdamW.
  Do not claim unconditional wall-clock faster convergence unless target_summary shows lower time-to-target.
```
"""
    doc.write_text(text, encoding="utf-8")
    print(f"wrote {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
