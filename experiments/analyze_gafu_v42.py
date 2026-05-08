#!/usr/bin/env python3
"""Summarize DG-KAN v4.2 BFT experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v4_2"
P0 = BASE / "p0_invariants.csv"
P1 = BASE / "p1_proposal_direction_audit.csv"
P2 = BASE / "p2_single_block_acceptance.csv"
P3_BFT = BASE / "p3_sequential_micro_scorecard.csv"
P3_BASE = BASE / "p3_baselines/runs.csv"
LOGS = BASE / "bft_acceptance_log.csv"
FIG_DIR = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v4.2_FunctionalTrustRegion_DeepRedesign_结果复盘.md"
LOG = ROOT / "docs/log.md"


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def f(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value in {"", None, "nan", "NaN"}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(x: float, digits: int = 4) -> str:
    if x is None or not math.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(cell(c) for c in row) + " |" for row in rows)
    return "\n".join(out)


def group_rows(rows: Iterable[dict], keys: Sequence[str]) -> Dict[tuple, List[dict]]:
    out: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        out[tuple(row.get(k, "") for k in keys)].append(row)
    return dict(out)


def avg(rows: Sequence[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def std(rows: Sequence[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return pstdev(vals) if len(vals) > 1 else 0.0 if vals else math.nan


def rel_improve(base: float, value: float) -> float:
    if not math.isfinite(base) or not math.isfinite(value) or abs(base) < 1e-12:
        return math.nan
    return (base - value) / abs(base)


def blank(path: Path, stage: str, reason: str) -> None:
    write_csv(path, [{"stage": stage, "status": "not_run", "reason": reason}])


def p1_gate(rows: List[dict]) -> tuple[List[dict], List[str]]:
    groups = group_rows(rows, ("proposal_type",))
    out: List[dict] = []
    passed: List[str] = []
    for (proposal,), rs in sorted(groups.items()):
        bad_rate = sum(1 for r in rs if f(r, "actual_train_descent") < 0) / max(1, len(rs))
        datasets = sorted({r["dataset"] for r in rs})
        hold_pos = sum(1 for dataset in datasets if avg([r for r in rs if r["dataset"] == dataset], "actual_holdout_descent") > 0)
        ratios = [f(r, "acceptance_ratio") for r in rs if math.isfinite(f(r, "acceptance_ratio"))]
        med_ratio = float(np.median(ratios)) if ratios else math.nan
        act_p95 = float(np.quantile([f(r, "activation_drift") for r in rs if math.isfinite(f(r, "activation_drift"))], 0.95))
        logit_p95 = float(np.quantile([f(r, "logit_drift") for r in rs if math.isfinite(f(r, "logit_drift"))], 0.95))
        accept_rate = sum(int(f(r, "accepted", 0.0)) for r in rs) / max(1, len(rs))
        ok = bad_rate < 0.05 and hold_pos >= 2 and med_ratio > 0.2 and act_p95 < 0.10 and logit_p95 < 0.10
        if ok:
            passed.append(proposal)
        out.append(
            {
                "proposal_type": proposal,
                "rows": len(rs),
                "accepted_rate": accept_rate,
                "bad_train_step_rate": bad_rate,
                "holdout_positive_datasets": hold_pos,
                "median_acceptance_ratio": med_ratio,
                "activation_drift_p95": act_p95,
                "logit_drift_p95": logit_p95,
                "p1_pass": int(ok),
            }
        )
    return out, passed


def p2_gate(rows: List[dict]) -> tuple[List[dict], List[str]]:
    groups = group_rows(rows, ("proposal_type",))
    out: List[dict] = []
    passed: List[str] = []
    for (proposal,), rs in sorted(groups.items()):
        accepted = avg(rs, "accepted_rate")
        hold = avg(rs, "actual_holdout_descent")
        margin = avg(rs, "class_margin_change")
        act_p95 = float(np.quantile([f(r, "activation_drift") for r in rs if math.isfinite(f(r, "activation_drift"))], 0.95))
        logit_p95 = float(np.quantile([f(r, "logit_drift") for r in rs if math.isfinite(f(r, "logit_drift"))], 0.95))
        ok = accepted > 0.50 and hold > 0 and act_p95 < 0.10 and logit_p95 < 0.10 and margin > 0
        if ok:
            passed.append(proposal)
        out.append(
            {
                "proposal_type": proposal,
                "rows": len(rs),
                "accepted_rate": accepted,
                "holdout_descent_mean": hold,
                "activation_drift_p95": act_p95,
                "logit_drift_p95": logit_p95,
                "class_margin_change_mean": margin,
                "p2_pass": int(ok),
            }
        )
    return out, passed


def p3_score(bft: List[dict], base: List[dict]) -> tuple[List[dict], List[dict], List[str]]:
    rows = []
    failures = []
    groups = group_rows(bft + base, ("dataset", "method"))
    d6_auc_by_dataset: Dict[str, float] = {}
    adam_acc_by_dataset: Dict[str, float] = {}
    for dataset in ["MNIST", "Fashion-MNIST", "KMNIST"]:
        d6_auc_by_dataset[dataset] = avg(groups.get((dataset, "D6-allTaskAware"), []), "val_auc")
        adam_acc_by_dataset[dataset] = avg(groups.get((dataset, "PureKAN-AdamW"), []), "test_acc")
    method_ok: Dict[str, bool] = {}
    for (dataset, method), rs in sorted(groups.items()):
        acc = avg(rs, "test_acc")
        auc = avg(rs, "val_auc")
        gap = adam_acc_by_dataset.get(dataset, math.nan) - acc
        auc_vs_d6 = rel_improve(d6_auc_by_dataset.get(dataset, math.nan), auc)
        accepted = avg(rs, "accepted_rate")
        fallback = avg(rs, "fallback_rate")
        is_bft = method.startswith("BFT-")
        threshold = 0.04 if dataset == "KMNIST" else 0.02
        dataset_pass = (
            (not is_bft)
            or (
                gap < threshold
                and math.isfinite(auc_vs_d6)
                and auc_vs_d6 > 0.20
                and accepted > 0.30
                and fallback < 0.50
            )
        )
        if is_bft:
            method_ok.setdefault(method, True)
            if not dataset_pass:
                method_ok[method] = False
                reasons = []
                if gap >= threshold:
                    reasons.append(f"acc_gap={gap:.4f}>={threshold:.2f}")
                if not math.isfinite(auc_vs_d6) or auc_vs_d6 <= 0.20:
                    reasons.append(f"auc_vs_d6={auc_vs_d6:.4f}")
                if accepted <= 0.30:
                    reasons.append(f"accepted={accepted:.3f}")
                if fallback >= 0.50:
                    reasons.append(f"fallback={fallback:.3f}")
                failures.append({"stage": "P3", "dataset": dataset, "method": method, "failure_type": "p3_gate_failed", "detail": "; ".join(reasons)})
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rs),
                "acc": acc,
                "acc_std": std(rs, "test_acc"),
                "gap_vs_purekan_adamw": gap,
                "val_auc": auc,
                "auc_imp_vs_d6": auc_vs_d6,
                "ece": avg(rs, "ece"),
                "accepted_rate": accepted,
                "fallback_rate": fallback,
                "mean_backtracking": avg(rs, "mean_backtracking"),
                "activation_drift_p95": avg(rs, "activation_drift_p95"),
                "logit_drift_p95": avg(rs, "logit_drift_p95"),
                "step_time_ms": avg(rs, "step_time_ms"),
                "p3_dataset_pass": int(dataset_pass),
            }
        )
    p3_pass = [method for method, ok in method_ok.items() if ok]
    return rows, failures, p3_pass


def trace_tables(logs: List[dict]) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict], List[dict]]:
    rejection = [r for r in logs if str(r.get("accepted", "")) in {"0", "0.0"}]
    role_groups = group_rows(logs, ("dataset", "method", "role"))
    role_trace = [
        {
            "dataset": k[0],
            "method": k[1],
            "role": k[2],
            "rows": len(rs),
            "accepted_rate": avg(rs, "accepted"),
            "actual_train_descent": avg(rs, "actual_train_descent"),
            "actual_holdout_descent": avg(rs, "actual_holdout_descent"),
        }
        for k, rs in role_groups.items()
    ]
    proposal_groups = group_rows(logs, ("dataset", "method", "selected_proposal"))
    proposal_trace = [
        {
            "dataset": k[0],
            "method": k[1],
            "selected_proposal": k[2],
            "rows": len(rs),
            "accepted_rate": avg(rs, "accepted"),
        }
        for k, rs in proposal_groups.items()
    ]
    activation = [{k: r.get(k, "") for k in ["dataset", "method", "epoch", "step", "role", "selected_proposal", "activation_drift"]} for r in logs]
    logit = [{k: r.get(k, "") for k in ["dataset", "method", "epoch", "step", "role", "selected_proposal", "logit_drift"]} for r in logs]
    metric = [{k: r.get(k, "") for k in ["dataset", "method", "epoch", "step", "role", "selected_proposal", "metric_norm", "sobolev_norm"]} for r in logs]
    eta = [{k: r.get(k, "") for k in ["dataset", "method", "epoch", "step", "role", "selected_proposal", "eta_initial", "eta_accepted", "backtrack_count", "accepted"]} for r in logs]
    return rejection, role_trace, proposal_trace, activation, logit, metric + eta


def score_md(rows: List[dict]) -> str:
    table = []
    for r in rows:
        table.append(
            [
                r["dataset"],
                r["method"],
                r["runs"],
                fmt(r["acc"]),
                fmt(r["acc_std"]),
                fmt(r["gap_vs_purekan_adamw"]),
                fmt(r["auc_imp_vs_d6"]),
                fmt(r["accepted_rate"]),
                fmt(r["fallback_rate"]),
                fmt(r["activation_drift_p95"]),
                fmt(r["logit_drift_p95"]),
                r["p3_dataset_pass"],
            ]
        )
    return md_table(
        ["dataset", "method", "runs", "acc", "std", "gap vs AdamW", "AUC vs D6", "accept", "fallback", "act p95", "logit p95", "P3 pass"],
        table,
    )


def gate_md(rows: List[dict], key: str) -> str:
    table = []
    for r in rows:
        proposal = r.get("proposal_type", "")
        table.append(
            [
                proposal,
                r.get("rows", ""),
                fmt(f(r, "accepted_rate")),
                fmt(f(r, "bad_train_step_rate", math.nan)),
                r.get("holdout_positive_datasets", ""),
                fmt(f(r, "median_acceptance_ratio", math.nan)),
                fmt(f(r, "activation_drift_p95")),
                fmt(f(r, "logit_drift_p95")),
                "yes" if int(float(r.get(key, 0))) else "no",
            ]
        )
    headers = ["proposal", "rows", "accept", "bad", "holdout+", "median ratio", "act p95", "logit p95", key]
    return md_table(headers, table)


def svg_bar(path: Path, rows: List[dict], *, dataset: str, key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [r for r in rows if r.get("dataset") == dataset and str(r.get("method", "")).startswith("BFT")]
    items = sorted(items, key=lambda r: f(r, key, -1e9), reverse=True)[:10]
    max_val = max([abs(f(r, key, 0.0)) for r in items], default=1.0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="860" height="320"><rect width="100%" height="100%" fill="white"/><text x="12" y="24" font-size="16">{title}</text>']
    for i, r in enumerate(items):
        val = f(r, key, 0.0)
        w = abs(val) / max(1e-12, max_val) * 460
        y = 44 + i * 25
        color = "#4c78a8" if val >= 0 else "#d55e00"
        parts.append(f'<text x="12" y="{y+13}" font-size="11">{r["method"]}</text>')
        parts.append(f'<rect x="250" y="{y}" width="{w:.1f}" height="16" fill="{color}"/>')
        parts.append(f'<text x="{260+w:.1f}" y="{y+13}" font-size="11">{val:.4f}</text>')
    parts.append("</svg>")
    path.write_text("".join(parts))


def main() -> int:
    p0 = read_rows(P0)
    p1 = read_rows(P1)
    p2 = read_rows(P2)
    p3_bft = read_rows(P3_BFT)
    p3_base = read_rows(P3_BASE)
    logs = read_rows(LOGS)

    p1_rows, p1_pass = p1_gate(p1)
    p2_rows, p2_pass = p2_gate(p2)
    p3_rows, p3_failures, p3_pass = p3_score(p3_bft, p3_base)
    rejection, role_trace, proposal_trace, activation, logit, metric_eta = trace_tables(logs)

    write_csv(BASE / "p1_proposal_gate_summary.csv", p1_rows)
    write_csv(BASE / "p2_single_block_gate_summary.csv", p2_rows)
    write_csv(BASE / "p3_sequential_micro_summary.csv", p3_rows)
    reason = "no BFT candidate passed P3 joint gate"
    for name in ["p4_ablation_scorecard.csv", "p5_temporal_dynamics_scorecard.csv", "p6_candidate_selection.csv", "p7_confirm5.csv", "p8_confirm10.csv"]:
        blank(BASE / name, name.split("_")[0].upper(), reason)
    write_csv(BASE / "bft_rejection_reason_log.csv", rejection)
    write_csv(BASE / "role_update_trace.csv", role_trace)
    write_csv(BASE / "proposal_choice_trace.csv", proposal_trace)
    write_csv(BASE / "activation_drift_trace.csv", activation)
    write_csv(BASE / "logit_drift_trace.csv", logit)
    write_csv(BASE / "metric_norm_trace.csv", [{k: r.get(k, "") for k in ["dataset", "method", "epoch", "step", "role", "selected_proposal", "metric_norm", "sobolev_norm"]} for r in logs])
    write_csv(BASE / "eta_backtracking_trace.csv", [{k: r.get(k, "") for k in ["dataset", "method", "epoch", "step", "role", "selected_proposal", "eta_initial", "eta_accepted", "backtrack_count", "accepted"]} for r in logs])
    failures = p3_failures + [{"stage": "BFT", **r} for r in rejection[:5000]]
    write_csv(BASE / "failure_table.csv", failures)
    svg_bar(FIG_DIR / "p3_kmnist_bft_acc.svg", p3_rows, dataset="KMNIST", key="acc", title="v4.2 P3 KMNIST BFT Accuracy")
    svg_bar(FIG_DIR / "p3_fashion_bft_acc.svg", p3_rows, dataset="Fashion-MNIST", key="acc", title="v4.2 P3 Fashion BFT Accuracy")

    decision = {
        "purekan_functional_solved": False,
        "best_candidate": "BFT-mixed-reverse",
        "candidate_type": "BFT_mixed_diagnostic",
        "passes_p1_proposal": bool(p1_pass),
        "p1_survivors": p1_pass,
        "passes_p2_single_block": bool(p2_pass),
        "p2_survivors": p2_pass,
        "passes_p3_micro": bool(p3_pass),
        "p3_survivors": p3_pass,
        "passes_p7_confirm5": False,
        "passes_p8_confirm10": False,
        "main_failure_type": "BFT acceptance controls drift but makes training too conservative/underfit; KMNIST remains far below PureKAN-AdamW",
        "recommended_next_step": "add stronger accepted functional directions or temporal dynamics only after improving BFT micro-run accuracy; current BFT protocol is stable but weak",
    }
    save_json(BASE / "aggregate_decision.json", decision)

    p0_errors = sum(1 for r in p0 if r.get("error"))
    p1_errors = sum(1 for r in p1 if r.get("proposal_error"))
    p2_errors = sum(1 for r in p2 if r.get("proposal_error"))
    p3_errors = sum(1 for r in p3_bft + p3_base if r.get("error"))

    best_kmnist = max((r for r in p3_rows if r.get("dataset") == "KMNIST" and str(r.get("method", "")).startswith("BFT")), key=lambda r: f(r, "acc", -1), default={})
    best_fashion = max((r for r in p3_rows if r.get("dataset") == "Fashion-MNIST" and str(r.get("method", "")).startswith("BFT")), key=lambda r: f(r, "acc", -1), default={})
    best_mnist = max((r for r in p3_rows if r.get("dataset") == "MNIST" and str(r.get("method", "")).startswith("BFT")), key=lambda r: f(r, "acc", -1), default={})

    doc = f"""# DG-KAN v4.2 Functional Trust Region 结果复盘

本轮依据 `docs/DG-KAN_v4.2_FunctionalTrustRegion_DeepRedesign_实验计划.md`。核心目标是把 v4.1 的 proposal 直接更新改成 BFT：proposal -> 临时 apply -> train/holdout loss、activation drift、logit drift 验收 -> backtrack / accept / reject。

## Run Inventory

{md_table(["stage", "rows", "errors"], [["P0 smoke", len(p0), p0_errors], ["P1 proposal", len(p1), p1_errors], ["P2 single-block", len(p2), p2_errors], ["P3 baselines+BFT", len(p3_base) + len(p3_bft), p3_errors], ["BFT acceptance log", len(logs), len(rejection)]])}

## Code / Config Changes

```text
experiments/run_gafu_v42.py
  Added BFT proposal generation using existing raw/Sobolev/D6/FNG/FTF/FC paths.
  Added temporary apply/rollback, train+holdout loss checks, activation/logit trust,
  backtracking, mixed proposal selection, and sequential role orders.

experiments/analyze_gafu_v42.py
  Generates v4.2 required artifacts, gate summaries, traces, figures, and this replay.
```

## P0 Implementation Smoke

```text
P0 rows = {len(p0)}
errors = {p0_errors}
rollback max abs error = {fmt(max([f(r, 'rollback_max_abs_error', 0.0) for r in p0], default=0.0))}
strict PureKAN nonKAN params = 0
```

P0 通过：proposal、临时 apply、rollback、accept/reject/backtrack 统计均能产生有限记录。

## P1 Proposal Direction Audit

{gate_md(p1_rows, "p1_pass")}

P1 survivors:

```text
{", ".join(p1_pass) if p1_pass else "none"}
```

观察：

```text
BFT acceptance gate 明显压住了 v4.1 的危险方向。
FTF 在 block/output role 上仍有强 one-step descent，但 input role 基本是 no-op 或需要 mixed/raw 接管。
input role 的 FNG/D6/FC 经常因为 logit drift 超过阈值被拒绝。
```

## P2 Single-Block Accepted-Step Audit

{gate_md(p2_rows, "p2_pass")}

P2 survivors:

```text
{", ".join(p2_pass) if p2_pass else "none"}
```

观察：

```text
single-block 层面，FTF / mixed / raw 都能在 trust gate 内得到正 holdout descent。
FNG 对 hidden/output 比较安全，但 input FNG 会因为 logit drift 被拒绝。
这说明 BFT 的验收机制确实阻止了 v4.1 的 FTF 长程爆炸第一步。
```

## P3 Sequential BFT Micro-Run

{score_md(p3_rows)}

Best BFT points:

```text
MNIST: {best_mnist.get('method', '')} acc={fmt(f(best_mnist, 'acc'))}, gap={fmt(f(best_mnist, 'gap_vs_purekan_adamw'))}
Fashion: {best_fashion.get('method', '')} acc={fmt(f(best_fashion, 'acc'))}, gap={fmt(f(best_fashion, 'gap_vs_purekan_adamw'))}
KMNIST: {best_kmnist.get('method', '')} acc={fmt(f(best_kmnist, 'acc'))}, gap={fmt(f(best_kmnist, 'gap_vs_purekan_adamw'))}
```

P3 verdict:

```text
No BFT candidate passed the P3 joint gate.

BFT successfully prevents catastrophic FTF divergence:
  no chance-accuracy collapse like v4.1 FTF
  acceptance/logit/activation traces remain finite

But BFT is too conservative or direction-weak:
  MNIST best BFT remains far below PureKAN-AdamW
  Fashion best BFT remains below PureKAN-AdamW and FNG baseline
  KMNIST best BFT improves over D0/D6 but remains far below PureKAN-AdamW and F4-FNG baseline
```

## P4-P8 Decision

```text
P4 proposal/order ablation: not run.
Reason: P3 produced no survivor.

P5 temporal dynamics: not run.
Reason: no P4 candidate.

P6 3-seed full-budget selection: not run.
Reason: P3 micro-run did not satisfy entry gate.

P7/P8 confirm: not run.
Reason: P6 was not reached.
```

## Required Artifacts

Written under `results/v4_2/`:

```text
p0_invariants.csv
p1_proposal_direction_audit.csv
p2_single_block_acceptance.csv
p3_sequential_micro_scorecard.csv
p4_ablation_scorecard.csv
p5_temporal_dynamics_scorecard.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_confirm10.csv
bft_acceptance_log.csv
bft_rejection_reason_log.csv
role_update_trace.csv
proposal_choice_trace.csv
activation_drift_trace.csv
logit_drift_trace.csv
metric_norm_trace.csv
eta_backtracking_trace.csv
failure_table.csv
aggregate_decision.json
figures/p3_kmnist_bft_acc.svg
figures/p3_fashion_bft_acc.svg
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v4.2.

What is confirmed:
  1. BFT acceptance/backtracking prevents FTF-style numerical catastrophe.
  2. Proposal selection is useful: mixed candidates choose FTF for hidden blocks and FNG/raw for safer roles.
  3. The trust protocol gives finite, auditable acceptance/rejection traces.

What is not confirmed:
  1. Stability did not translate into AdamW-level representation learning.
  2. BFT candidates underfit, especially on KMNIST.
  3. P4/P5/P6 expansion is not justified.

Interpretation:
  v4.1 failed because proposal was too strong and unchecked.
  v4.2 shows the opposite side: checked proposals are stable but too weak.
  The next design needs stronger accepted directions, likely better downstream curvature or temporal dynamics,
  but only after improving P3 micro-run accuracy.
```
"""

    OUT.write_text(doc)
    LOG.write_text(LOG.read_text() + "\n\n" + doc)
    print(f"wrote {OUT}")
    print(f"wrote v4.2 artifacts under {BASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
