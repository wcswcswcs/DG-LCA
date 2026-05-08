#!/usr/bin/env python3
"""Summarize DG-KAN v3.8 GlobalTFU / depth-wise PureKAN experiments."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results/gafu_v3_8_p0/runs.csv"
P1 = ROOT / "results/gafu_v3_8_p1_shadow/shadow_runs.csv"
P2 = ROOT / "results/gafu_v3_8_p2_global/runs.csv"
P3 = ROOT / "results/gafu_v3_8_p3_depthwise/runs.csv"
OUT = ROOT / "docs/DG-KAN_v3.8_GlobalTFU_Depthwise_PureKAN_结果复盘.md"
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
    return (
        name.replace("-alphaFixed1", "")
        .replace("-smoke", "")
        .replace("PureKAN-", "")
    )


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


def agg(rows: List[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def agg_std(rows: List[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return pstdev(vals) if len(vals) > 1 else 0.0 if vals else math.nan


def p0_table(rows: List[dict]) -> str:
    table = []
    for r in sorted(rows, key=lambda x: (x.get("dataset", ""), x.get("method", ""))):
        if not r.get("method", "").startswith("D"):
            continue
        table.append(
            [
                r.get("dataset", ""),
                clean_method(r.get("method", "")),
                fmt(f(r, "pure_alpha_mean"), 1),
                fmt_intish(f(r, "pure_alpha_trainable")),
                fmt_intish(f(r, "learnable_nonkan_params")),
                fmt_intish(f(r, "purekan_has_linear")),
                fmt(f(r, "coeff_param_seen_ratio"), 3),
                r.get("input_metric_mode_seen", ""),
                r.get("shallow_metric_mode_seen", ""),
                r.get("deep_metric_mode_seen", ""),
                r.get("output_metric_mode_seen", ""),
                fmt(f(r, "pure_input_metric_condition"), 2),
                fmt(f(r, "pure_output_metric_condition"), 2),
            ]
        )
    return md_table(
        [
            "dataset",
            "method",
            "alpha",
            "alpha train",
            "nonKAN",
            "linear",
            "coeff seen",
            "input",
            "shallow",
            "deep",
            "output",
            "cond in",
            "cond out",
        ],
        table,
    )


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
                fmt(f(r, "input_cos_raw_precond")),
                fmt(f(r, "shallow_cos_raw_precond_mean")),
                fmt(f(r, "deep_cos_raw_precond_mean")),
                fmt(f(r, "output_cos_raw_precond")),
            ]
        )
    return md_table(
        ["dataset", "method", "train descent", "val descent", "bad", "input cos", "shallow cos", "deep cos", "output cos"],
        table,
    )


def score_rows(rows: List[dict], methods: Sequence[str] | None = None) -> List[List[str]]:
    groups = group_rows(rows, ("dataset", "method"))
    table = []
    for dataset in ["MNIST", "Fashion-MNIST", "KMNIST"]:
        dataset_methods = {m: rs for (d, m), rs in groups.items() if d == dataset}
        if not dataset_methods:
            continue
        adam = dataset_methods.get("PureKAN-AdamW-alphaFixed1", [])
        d0 = dataset_methods.get("D0-allFullSobolev-alphaFixed1", [])
        adam_acc = agg(adam, "test_acc")
        adam_auc = agg(adam, "val_auc")
        adam_ece = agg(adam, "ece")
        d0_acc = agg(d0, "test_acc")
        d0_auc = agg(d0, "val_auc")
        selected = methods or sorted(dataset_methods)
        for method in selected:
            rs = dataset_methods.get(method, [])
            if len(rs) < 3:
                continue
            acc = agg(rs, "test_acc")
            auc = agg(rs, "val_auc")
            ece = agg(rs, "ece")
            auc_imp_d0 = (d0_auc - auc) / abs(d0_auc) if math.isfinite(d0_auc) and abs(d0_auc) > 1e-12 else math.nan
            auc_imp_adam = (adam_auc - auc) / abs(adam_auc) if math.isfinite(adam_auc) and abs(adam_auc) > 1e-12 else math.nan
            ece_red = (adam_ece - ece) / abs(adam_ece) if math.isfinite(adam_ece) and abs(adam_ece) > 1e-12 else math.nan
            table.append(
                [
                    dataset,
                    clean_method(method),
                    str(len(rs)),
                    fmt(acc),
                    fmt(agg_std(rs, "test_acc")),
                    fmt(adam_acc - acc),
                    fmt(acc - d0_acc),
                    fmt(auc_imp_d0),
                    fmt(auc_imp_adam),
                    fmt(ece_red),
                    fmt(agg(rs, "phi_prime_p95")),
                    fmt(agg(rs, "max_jac_condition")),
                    fmt(agg(rs, "tfu_safeguard_fail_rate")),
                ]
            )
    return table


def gate_table(rows: List[dict]) -> str:
    groups = group_rows(rows, ("dataset", "method"))
    methods = [
        "D6-allTaskAware-alphaFixed1",
        "D1-frontTask-backSob-alphaFixed1",
        "D2-frontSob-backTask-alphaFixed1",
        "D3-taskSobTask-alphaFixed1",
        "D7-inputOutputTask-middleSob-alphaFixed1",
        "D8-inputDiag-blockTask-outputFisher-alphaFixed1",
    ]
    table = []
    for method in methods:
        gaps = []
        auc_imps = []
        improves = []
        for dataset in ["MNIST", "Fashion-MNIST", "KMNIST"]:
            adam = groups.get((dataset, "PureKAN-AdamW-alphaFixed1"), [])
            d0 = groups.get((dataset, "D0-allFullSobolev-alphaFixed1"), [])
            rs = groups.get((dataset, method), [])
            adam_acc = agg(adam, "test_acc")
            d0_acc = agg(d0, "test_acc")
            d0_auc = agg(d0, "val_auc")
            acc = agg(rs, "test_acc")
            auc = agg(rs, "val_auc")
            gaps.append(adam_acc - acc)
            improves.append(acc > d0_acc)
            auc_imps.append((d0_auc - auc) / abs(d0_auc) if math.isfinite(d0_auc) and abs(d0_auc) > 1e-12 else math.nan)
        enter = max(gaps) < 0.01 and min(auc_imps) > 0.20 and all(improves)
        table.append(
            [
                clean_method(method),
                fmt(max(gaps)),
                fmt(min(auc_imps)),
                "yes" if all(improves) else "no",
                "yes" if enter else "no",
            ]
        )
    return md_table(["method", "max gap vs AdamW", "min AUC imp vs D0", "improves D0 acc all", "enter P4"], table)


def main() -> int:
    p0 = read_rows(P0)
    p1 = read_rows(P1)
    p2 = read_rows(P2)
    p3 = read_rows(P3)
    p2_methods = [
        "PureKAN-AdamW-alphaFixed1",
        "D0-allFullSobolev-alphaFixed1",
        "D6-allTaskAware-alphaFixed1",
        "D6-noSob-alphaFixed1",
        "D6-lowSob-alphaFixed1",
        "D6-withSafeguard-alphaFixed1",
        "Hybrid-DGKAN-UFULL-f085-alphaFixed1",
        "MLP-AdamW-alphaFixed1",
    ]
    p3_methods = [
        "PureKAN-AdamW-alphaFixed1",
        "D0-allFullSobolev-alphaFixed1",
        "D6-allTaskAware-alphaFixed1",
        "D1-frontTask-backSob-alphaFixed1",
        "D2-frontSob-backTask-alphaFixed1",
        "D3-taskSobTask-alphaFixed1",
        "D7-inputOutputTask-middleSob-alphaFixed1",
        "D8-inputDiag-blockTask-outputFisher-alphaFixed1",
    ]
    doc = f"""# DG-KAN v3.8 GlobalTFU / Depthwise PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v3.8_修改版_GlobalTFU_Depthwise_PureKAN_实验计划.md`。目标是验证 PureKAN 的 task-aware functional update 是否能修复 v3.7 中 full Sobolev 方向不可靠的问题。

## Code / Config Changes

```text
experiments/dgkan_core.py
  TrainConfig / RuntimeState added TFU fields, role-specific input/shallow/deep/output metrics, LR multipliers and cosine safeguard counters.
  PureKAN functional update now supports tfu_task_diag and tfu_data_task_diag.
  Shallow/deep block roles are split from blocks.{{idx}} and also aggregated into block stats.
  Result rows include TFU metric min/max, per-role conditions, fallback counters, and safeguard fail rates.

experiments/run_gafu_v38.py
  Added P0 smoke, P1 shadow audit, P2 global TFU, P3 depth-wise sweep, and conditional P4 entry wiring.

experiments/analyze_gafu_v38.py
  Generates this replay and appends a concise log entry.
```

## P0 Implementation Smoke

{p0_table(p0)}

P0 verdict: pass. TFU rows have pure alpha fixed at 1.0, no trainable non-KAN parameters, no linear layer, full coefficient coverage, and finite role metric conditions.

## P1 One-Batch Direction Audit

{p1_table(p1)}

P1 verdict: strong positive direction signal. D0 full Sobolev remains a bad train step on MNIST/KMNIST, while D6 GlobalTFU has positive train and validation descent on all three datasets.

## P2 Global TFU Micro-Run

{md_table(['dataset', 'method', 'runs', 'acc', 'std', 'gap vs AdamW', 'acc minus D0', 'AUC imp vs D0', 'AUC imp vs AdamW', 'ECE red', 'phi p95', 'J', 'safeguard fail'], score_rows(p2, p2_methods))}

P2 verdict: D6 does not pass the global gate. It repairs one-batch direction and improves KMNIST substantially over D0, but the 3-seed accuracy gap vs PureKAN-AdamW is too large on MNIST and KMNIST. Fashion only passes the accuracy side for `D6-noSob` / `D6-lowSob`, not the unified D6 profile.

## P3 Depth-Wise Sweep

{md_table(['dataset', 'method', 'runs', 'acc', 'std', 'gap vs AdamW', 'acc minus D0', 'AUC imp vs D0', 'AUC imp vs AdamW', 'ECE red', 'phi p95', 'J', 'safeguard fail'], score_rows(p3, p3_methods))}

### P3 Gate Check

{gate_table(p3)}

P3 verdict: no method enters P4. `D3`/`D7` are best for Fashion and improve MNIST over D0, while `D8` is comparatively best on KMNIST among depth-wise variants. None satisfy the joint gate because MNIST/KMNIST still remain more than 1-2 points behind PureKAN-AdamW and AUC improvement vs D0 does not reach the required 20% on all datasets.

## Final Decision

```text
P0:
  Implementation smoke passed.

P1:
  GlobalTFU fixes the immediate direction failure seen in v3.7 one-batch audits.

P2:
  D6 GlobalTFU is not accepted as a full PureKAN optimizer.
  It improves direction and calibration/geometry but does not match PureKAN-AdamW accuracy.

P3:
  Depth-wise task placement helps identify useful structure:
    D3/D7: best Fashion and better MNIST than D0.
    D8: best KMNIST depth-wise candidate.
  No candidate reaches the P4 entry gate.

P4-P7:
  Not run by plan because P3 produced no qualifying candidate.

Final:
  v3.8 confirms that the v3.7 bottleneck is partly metric direction quality, not implementation coverage.
  Task-aware diagonal metrics are a real repair for local descent, but global training still underfits or becomes seed-sensitive.
  Next work should focus on layer-local/block-local objective design or adaptive per-role LR schedules before spending 5/10-seed confirmation budget.
```
"""
    OUT.write_text(doc, encoding="utf-8")

    log_entry = """
## 2026-05-02 DG-KAN v3.8 GlobalTFU / Depthwise PureKAN

依据 `docs/DG-KAN_v3.8_修改版_GlobalTFU_Depthwise_PureKAN_实验计划.md`，实现并运行：

```text
P0 smoke: 21 rows
P1 one-batch shadow: 24 rows
P2 global TFU: 72 rows
P3 depth-wise sweep: 72 rows
```

关键结论：

```text
P0 pass: PureKAN alphaFixed1 / no non-KAN params / coeff coverage / TFU role metrics 均正常。
P1 pass: D6 GlobalTFU 把 D0 full Sobolev 的 MNIST/KMNIST bad step 修成三数据集 train/val descent 全正。
P2 fail: D6 方向修复不能稳定转化为 3-seed accuracy gate。
P3 fail: D3/D7 对 Fashion/MNIST 有帮助，D8 对 KMNIST 相对最好，但无候选进入 P4。
P4-P7 not run: 按计划 gate 未达，不做大 seed confirm。
```

复盘文档：

```text
docs/DG-KAN_v3.8_GlobalTFU_Depthwise_PureKAN_结果复盘.md
```
"""
    existing = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    if "DG-KAN v3.8 GlobalTFU / Depthwise PureKAN" not in existing:
        LOG.write_text(existing.rstrip() + "\n\n" + log_entry.strip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
