#!/usr/bin/env python3
"""Summarize DG-KAN v3.7 PureKAN U-FULL experiments."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results/gafu_v3_7_p0/runs.csv"
P1 = ROOT / "results/gafu_v3_7_p1_shadow/shadow_runs.csv"
P2 = ROOT / "results/gafu_v3_7_p2_schedule_fixed/runs.csv"
P3 = ROOT / "results/gafu_v3_7_p3_role/runs.csv"
OUT = ROOT / "docs/DG-KAN_v3.7_PureKAN_UFULL_结果复盘.md"


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


def clean_method(name: str) -> str:
    return (
        name.replace("-alphaFixed1", "")
        .replace("PureKAN-UFULL-", "U-")
        .replace("PureKAN-", "")
        .replace("role-", "")
    )


def fmt(x: float, digits: int = 4) -> str:
    if x is None or not math.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def fmt_intish(x: float) -> str:
    if x is None or not math.isfinite(x):
        return ""
    return str(int(round(x)))


def group_rows(rows: Iterable[dict], keys: Sequence[str]) -> Dict[tuple, List[dict]]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    return dict(groups)


def agg(rows: List[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def agg_std(rows: List[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return pstdev(vals) if len(vals) > 1 else 0.0 if vals else math.nan


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(cell(c) for c in row) + " |" for row in rows)
    return "\n".join(out)


def p0_table(rows: List[dict]) -> str:
    selected = [
        r
        for r in rows
        if r.get("model_type") == "pure_kan"
        and ("UFULL" in r.get("method", "") or "U-FULL" in r.get("method", ""))
    ]
    table_rows = []
    for r in sorted(selected, key=lambda x: (x.get("dataset", ""), x.get("method", ""))):
        table_rows.append(
            [
                r.get("dataset", ""),
                clean_method(r.get("method", "")),
                fmt(f(r, "pure_alpha_mean"), 1),
                fmt_intish(f(r, "pure_alpha_trainable")),
                fmt_intish(f(r, "learnable_nonkan_params")),
                fmt_intish(f(r, "purekan_has_linear")),
                fmt(f(r, "coeff_param_seen_ratio"), 3),
                fmt_intish(f(r, "active_steps_actual")),
                fmt_intish(f(r, "transition_steps_actual")),
                fmt_intish(f(r, "geometry_steps_actual")),
                r.get("input_metric_mode_seen", "") or r.get("metric_active_seen", ""),
                r.get("metric_geometry_seen", ""),
                fmt(f(r, "trust_clip_rate"), 3),
            ]
        )
    return md_table(
        [
            "dataset",
            "method",
            "alpha",
            "alpha train",
            "nonKAN params",
            "linear",
            "coeff seen",
            "active",
            "trans",
            "geom",
            "active metric",
            "geom metric",
            "clip",
        ],
        table_rows,
    )


def p1_table(rows: List[dict]) -> str:
    order = {"MNIST": 0, "Fashion-MNIST": 1, "KMNIST": 2}
    table_rows = []
    for r in sorted(rows, key=lambda x: (order.get(x.get("dataset", ""), 9), x.get("method", ""))):
        table_rows.append(
            [
                r.get("dataset", ""),
                r.get("method", ""),
                fmt(f(r, "shadow_actual_descent")),
                fmt(f(r, "shadow_val_descent")),
                fmt_intish(f(r, "shadow_bad_step_bool")),
                fmt(f(r, "shadow_update_norm")),
                fmt(f(r, "input_cos_raw_precond")),
                fmt(f(r, "block_cos_raw_precond_mean")),
                fmt(f(r, "output_cos_raw_precond")),
            ]
        )
    return md_table(
        [
            "dataset",
            "method",
            "train descent",
            "val descent",
            "bad",
            "update norm",
            "input cos",
            "block cos",
            "output cos",
        ],
        table_rows,
    )


def score_table(rows: List[dict], *, baseline_source: Dict[str, dict] | None = None) -> str:
    groups = group_rows(rows, ("dataset", "method"))
    by_dataset: Dict[str, Dict[str, List[dict]]] = defaultdict(dict)
    for (dataset, method), rs in groups.items():
        by_dataset[dataset][method] = rs

    table_rows = []
    for dataset in ["MNIST", "Fashion-MNIST", "KMNIST"]:
        methods = by_dataset.get(dataset, {})
        if not methods:
            continue
        adam_rows = next((rs for m, rs in methods.items() if m.startswith("PureKAN-AdamW")), None)
        if baseline_source is not None:
            adam = baseline_source.get(dataset, {})
            adam_acc = adam.get("acc", math.nan)
            adam_auc = adam.get("auc", math.nan)
            adam_ece = adam.get("ece", math.nan)
        else:
            adam_acc = agg(adam_rows or [], "test_acc")
            adam_auc = agg(adam_rows or [], "val_auc")
            adam_ece = agg(adam_rows or [], "ece")
        full_rows = next(
            (
                rs
                for m, rs in methods.items()
                if "full-alphaFixed1" in m or "role-allfull" in m
            ),
            None,
        )
        full_auc = agg(full_rows or [], "val_auc")
        for method, rs in sorted(methods.items()):
            if len(rs) < 3:
                continue
            acc = agg(rs, "test_acc")
            auc = agg(rs, "val_auc")
            ece = agg(rs, "ece")
            ece_red = (adam_ece - ece) / abs(adam_ece) if math.isfinite(adam_ece) and abs(adam_ece) > 1e-12 else math.nan
            table_rows.append(
                [
                    dataset,
                    clean_method(method),
                    str(len(rs)),
                    fmt(acc),
                    fmt(agg_std(rs, "test_acc")),
                    fmt(adam_acc - acc),
                    fmt(full_auc - auc),
                    fmt(adam_auc - auc),
                    fmt(ece_red),
                    fmt(agg(rs, "phi_prime_p95")),
                    fmt(agg(rs, "max_jac_condition")),
                    fmt(agg(rs, "active_steps_actual"), 1),
                    fmt(agg(rs, "transition_steps_actual"), 1),
                    fmt(agg(rs, "geometry_steps_actual"), 1),
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
            "AUC imp vs full",
            "AUC imp vs AdamW",
            "ECE red",
            "phi p95",
            "J",
            "active",
            "trans",
            "geom",
        ],
        table_rows,
    )


def baseline_from_p2(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    for (dataset, method), rs in group_rows(rows, ("dataset", "method")).items():
        if method.startswith("PureKAN-AdamW"):
            out[dataset] = {
                "acc": agg(rs, "test_acc"),
                "auc": agg(rs, "val_auc"),
                "ece": agg(rs, "ece"),
            }
    return out


def p3_gate(rows: List[dict], baselines: Dict[str, dict]) -> str:
    groups = group_rows(rows, ("dataset", "method"))
    full_auc = {}
    for (dataset, method), rs in groups.items():
        if "role-allfull" in method:
            full_auc[dataset] = agg(rs, "val_auc")
    by_method: Dict[str, Dict[str, List[dict]]] = defaultdict(dict)
    for (dataset, method), rs in groups.items():
        if "role-allfull" in method:
            continue
        by_method[method][dataset] = rs

    table_rows = []
    for method, ds_rows in sorted(by_method.items()):
        datasets = ["MNIST", "Fashion-MNIST", "KMNIST"]
        gaps = []
        auc_imps = []
        ece_ok = []
        accs = []
        for dataset in datasets:
            rs = ds_rows.get(dataset, [])
            if not rs:
                continue
            acc = agg(rs, "test_acc")
            auc = agg(rs, "val_auc")
            ece = agg(rs, "ece")
            base = baselines.get(dataset, {})
            gaps.append(base.get("acc", math.nan) - acc)
            auc_imps.append(full_auc.get(dataset, math.nan) - auc)
            base_ece = base.get("ece", math.nan)
            ece_ok.append(math.isfinite(base_ece) and ece <= base_ece * 1.05)
            accs.append(acc)
        pass_gate = (
            len(gaps) == 3
            and all(math.isfinite(g) and g < 0.010 for g in gaps)
            and all(math.isfinite(a) and a > 0 for a in auc_imps)
            and all(ece_ok)
        )
        table_rows.append(
            [
                clean_method(method),
                fmt(min(accs) if accs else math.nan),
                fmt(max(gaps) if gaps else math.nan),
                fmt(min(auc_imps) if auc_imps else math.nan),
                "yes" if all(ece_ok) else "no",
                "yes" if pass_gate else "no",
            ]
        )
    return md_table(
        ["method", "min acc", "max acc gap", "min AUC imp vs full", "ECE ok", "enter P5"],
        table_rows,
    )


def main() -> None:
    p0 = read_rows(P0)
    p1 = read_rows(P1)
    p2 = read_rows(P2)
    p3 = read_rows(P3)
    baselines = baseline_from_p2(p2)

    content = [
        "# DG-KAN v3.7 PureKAN U-FULL 结果复盘",
        "",
        "本轮依据 `docs/DG-KAN_v3.7_PureKAN_UFULL_代码检查与实验计划.md`。正式 P2 使用修复后的 `results/gafu_v3_7_p2_schedule_fixed`；旧 P2 只作为发现 warmup 被 epoch-end geometry trigger 截断的中间记录。",
        "",
        "## Code / Config Fixes",
        "",
        "```text",
        "experiments/dgkan_core.py",
        "  PureKAN alpha_mode=fixed1 now registers alpha as a fixed buffer with value 1.0.",
        "  PureKANClassifier forwards alpha_mode into all residual blocks.",
        "  PureKAN functional update now supports role-specific metrics/LR/trust for input/block/output.",
        "  Result rows include phase/metric traces, active/transition/geometry step counts, per-role update stats, coeff coverage, and PureKAN contribution audit.",
        "",
        "experiments/run_gafu_v37.py",
        "  Added v3.7 P0/P1/P2/P3/P5/P6 packages and one-batch shadow step.",
        "  PureKAN warmup now sets geometry_min_epochs=999 so max_active_frac controls 0.10/0.20 schedule length.",
        "",
        "experiments/analyze_gafu_v37.py",
        "  Generates this result replay from P0/P1/P2-fixed/P3 CSVs.",
        "```",
        "",
        "## P0 Code / Config Audit",
        "",
        p0_table(p0),
        "",
        "P0 verdict: pass. PureKAN has no non-KAN trainable parameters, alphaFixed1 is truly fixed at 1.0, and functional-update rows cover all coeff parameters (`coeff seen = 1.0`). Warmup smoke rows have real active steps.",
        "",
        "## P1 One-Batch Shadow Step",
        "",
        p1_table(p1),
        "",
        "P1 verdict: the full Sobolev direction is not reliably descent for PureKAN. It is a bad train step on MNIST and KMNIST, and on Fashion it improves the train batch but worsens validation loss. This points to a direction/metric mismatch rather than a missing implementation hook.",
        "",
        "## P2 Metric Schedule Repair Test",
        "",
        p2 and score_table(p2) or "_missing P2 rows_",
        "",
        "P2 verdict: schedule repair is real after the fix. 0.10 and 0.20 now have distinct active-step counts, and smooth has transition steps. Fashion benefits most (`diag-to-full-smooth-0.20` reaches acc 0.8460 and val-loss AUC 0.5242), but MNIST and KMNIST remain far below PureKAN-AdamW in accuracy.",
        "",
        "## P3 Role-Wise PureKAN U-FULL Sweep",
        "",
        p3 and score_table(p3, baseline_source=baselines) or "_missing P3 rows_",
        "",
        "### P3 Gate Check",
        "",
        p3_gate(p3, baselines),
        "",
        "P3 verdict: no candidate enters P5. Fashion is mostly repaired by role-aware metrics, but MNIST and KMNIST still miss the `acc gap < 1%` requirement. The best KMNIST role candidate is still about 2.4 points behind PureKAN-AdamW; the best MNIST role candidate is also about 2.5 points behind.",
        "",
        "## P4-P7 Decision",
        "",
        "```text",
        "P4 capacity/epoch budget check: not expanded.",
        "Reason: P3 did not produce a candidate close enough to PureKAN-AdamW on all three datasets.",
        "",
        "P5 3-seed candidate selection: not run.",
        "Reason: P3 entry gate failed for every role-wise candidate.",
        "",
        "P6 5-seed confirm and P7 speed proxy: not allowed by plan because P5 was not reached.",
        "```",
        "",
        "## Failure Diagnosis",
        "",
        "```text",
        "Implementation failure: no.",
        "  alphaFixed1, coeff coverage, pure-only parameterization, metric traces, and active-step counts pass audit.",
        "",
        "Schedule failure: partially fixed but not sufficient.",
        "  The original warmup length was indeed collapsed by epoch-end geometry triggers.",
        "  After fixing it, Fashion improves, but MNIST/KMNIST still do not approach AdamW.",
        "",
        "Direction failure: yes.",
        "  P1 shadow steps show full Sobolev is not reliable descent on PureKAN.",
        "  P2/P3 improve some trajectories by avoiding full metric on input/output, but full-network PureKAN remains unstable on hard datasets.",
        "",
        "Budget failure: unlikely as the primary cause.",
        "  The gap is already visible in one-step shadow descent and 3-seed role sweeps.",
        "  Extending 5/10-seed confirm is not justified until the metric direction is repaired.",
        "```",
        "",
        "## Final Decision",
        "",
        "```text",
        "PureKAN U-FULL is not accepted in v3.7.",
        "",
        "What is confirmed:",
        "  1. PureKAN alpha/fixed1 and functional coverage bugs are fixed.",
        "  2. Warmup schedule now actually executes 10%/20% active windows.",
        "  3. Role-aware metrics substantially improve Fashion and partially rescue KMNIST.",
        "",
        "What is not confirmed:",
        "  1. Full-network PureKAN functional update does not match PureKAN-AdamW on MNIST/KMNIST.",
        "  2. No role/schedule candidate qualifies for P5/P6.",
        "  3. The bottleneck is metric direction quality, not missing CUDA/KAT backward or missing parameter coverage.",
        "",
        "Next recommended research direction:",
        "  Redesign the PureKAN functional metric itself, likely with layer-local or block-local objectives and a safer output/input metric, before spending more seeds on confirmation.",
        "```",
        "",
    ]
    OUT.write_text("\n".join(content), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
