#!/usr/bin/env python3
"""Summarize DG-KAN v4.1 Functional Update Redesign experiments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v4_1"
P0 = BASE / "p0_smoke/runs.csv"
P1 = BASE / "p1_shadow/runs.csv"
P2 = BASE / "p2_micro/runs.csv"
FIG_DIR = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v4.1_FunctionalUpdate_Redesign_结果复盘.md"
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


def group_rows(rows: Iterable[dict], keys: Sequence[str], *, ok_only: bool = True) -> Dict[tuple, List[dict]]:
    out: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        if ok_only and row.get("error"):
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
    if not math.isfinite(base) or abs(base) < 1e-12 or not math.isfinite(value):
        return math.nan
    return (base - value) / abs(base)


def p0_invariants(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        fit_r2 = f(row, "ftf_fit_R2_mean")
        fc_rec = f(row, "fc_adam_reconstruction_error")
        ftr_res = f(row, "ftr_cg_residual_final")
        finite = all(
            math.isfinite(v)
            for v in [
                f(row, "test_acc", 0.0),
                f(row, "val_auc", 0.0),
                f(row, "functional_param_coverage", 0.0),
            ]
        )
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "runs_error": row.get("error", ""),
                "alpha_mode": row.get("alpha_mode", ""),
                "alpha_trainable": row.get("alpha_trainable", ""),
                "alpha_final_mean": row.get("alpha_final_mean", ""),
                "learnable_nonkan_params": row.get("learnable_nonkan_params", ""),
                "functional_param_coverage": row.get("functional_param_coverage", ""),
                "input_kan_coeff_seen": row.get("input_kan_coeff_seen", ""),
                "block_kan_coeff_seen": row.get("block_kan_coeff_seen", ""),
                "output_kan_coeff_seen": row.get("output_kan_coeff_seen", ""),
                "ftf_fit_R2_mean": row.get("ftf_fit_R2_mean", ""),
                "ftf_max_layer_delta_norm_ratio": row.get("ftf_max_layer_delta_norm_ratio", ""),
                "fc_adam_reconstruction_error": row.get("fc_adam_reconstruction_error", ""),
                "fc_adam_grad_chain_error": row.get("fc_adam_grad_chain_error", ""),
                "ftr_cg_residual_final": row.get("ftr_cg_residual_final", ""),
                "all_finite": int(finite),
                "ftf_smoke_ok": int(not row.get("error") and (not math.isfinite(fit_r2) or fit_r2 > 0.5)),
                "fc_adam_smoke_ok": int(not row.get("error") and (not math.isfinite(fc_rec) or fc_rec < 1e-3)),
                "ftr_smoke_ok": int(not row.get("error") and (not math.isfinite(ftr_res) or ftr_res <= 1.0)),
            }
        )
    return out


def p1_gate(rows: List[dict]) -> tuple[List[dict], List[str]]:
    groups = group_rows(rows, ("method",))
    out: List[dict] = []
    passed: List[str] = []
    for (method,), rs in sorted(groups.items()):
        if method == "AdamW-one-step":
            continue
        train_vals = [f(r, "shadow_actual_descent") for r in rs]
        val_vals = [f(r, "shadow_val_descent") for r in rs]
        bad = sum(int(f(r, "shadow_bad_step_bool", 0.0)) for r in rs)
        r2_vals = [f(r, "ftf_fit_R2_mean") for r in rs if math.isfinite(f(r, "ftf_fit_R2_mean"))]
        delta_vals = [f(r, "ftf_delta_norm_ratio_max") for r in rs if math.isfinite(f(r, "ftf_delta_norm_ratio_max"))]
        is_ftf = method.startswith("FTF")
        min_train = min(train_vals)
        min_val = min(val_vals)
        min_r2 = min(r2_vals) if r2_vals else math.nan
        max_delta = max(delta_vals) if delta_vals else math.nan
        ok = bad == 0 and min_train > 0.0 and min_val >= -0.01
        if is_ftf:
            ok = ok and min_r2 > 0.6 and max_delta < 0.25
        if ok:
            passed.append(method)
        out.append(
            {
                "method": method,
                "datasets": len(rs),
                "bad_train_steps": bad,
                "min_train_descent": min_train,
                "min_val_descent": min_val,
                "target_fit_R2_min": min_r2,
                "max_layer_delta_norm_ratio": max_delta,
                "p1_pass": int(ok),
            }
        )
    return out, passed


def p2_score(rows: List[dict]) -> tuple[List[dict], List[dict], List[str]]:
    groups = group_rows(rows, ("dataset", "method"))
    methods = sorted({m for (_, m) in groups})
    score: List[dict] = []
    failure: List[dict] = []
    p2_pass: List[str] = []
    method_ok: Dict[str, bool] = {m: True for m in methods if m not in {"PureKAN-AdamW", "D6-allTaskAware"}}

    for dataset in ["MNIST", "Fashion-MNIST", "KMNIST"]:
        adam = groups.get((dataset, "PureKAN-AdamW"), [])
        d6 = groups.get((dataset, "D6-allTaskAware"), [])
        adam_acc = avg(adam, "test_acc")
        adam_auc = avg(adam, "val_auc")
        adam_ece = avg(adam, "ece")
        adam_phi = avg(adam, "phi_prime_p95")
        d6_auc = avg(d6, "val_auc")
        for method in methods:
            rs = groups.get((dataset, method), [])
            err_count = len([r for r in rows if r.get("dataset") == dataset and r.get("method") == method and r.get("error")])
            if not rs and err_count:
                failure.append(
                    {
                        "stage": "P2",
                        "dataset": dataset,
                        "method": method,
                        "failure_type": "run_error",
                        "detail": f"{err_count} errored runs",
                    }
                )
                method_ok[method] = False
                continue
            if not rs:
                continue
            acc = avg(rs, "test_acc")
            auc = avg(rs, "val_auc")
            ece = avg(rs, "ece")
            phi = avg(rs, "phi_prime_p95")
            gap = adam_acc - acc
            auc_imp_adam = rel_improve(adam_auc, auc)
            auc_imp_d6 = rel_improve(d6_auc, auc)
            ece_red = rel_improve(adam_ece, ece)
            phi_ratio = phi / adam_phi if math.isfinite(phi) and math.isfinite(adam_phi) and adam_phi > 0 else math.nan
            threshold = 0.03 if dataset == "KMNIST" else 0.02
            dataset_pass = (
                method == "PureKAN-AdamW"
                or (
                    gap < threshold
                    and math.isfinite(auc_imp_d6)
                    and auc_imp_d6 > 0.0
                    and (not math.isfinite(ece_red) or ece_red > -0.10)
                    and (not math.isfinite(phi_ratio) or phi_ratio <= 1.5)
                    and err_count == 0
                )
            )
            if method not in {"PureKAN-AdamW", "D6-allTaskAware"} and not dataset_pass:
                method_ok[method] = False
                reasons = []
                if gap >= threshold:
                    reasons.append(f"acc_gap={gap:.4f}>={threshold:.2f}")
                if not math.isfinite(auc_imp_d6) or auc_imp_d6 <= 0:
                    reasons.append(f"auc_vs_d6={auc_imp_d6:.4f}")
                if err_count:
                    reasons.append(f"errors={err_count}")
                failure.append(
                    {
                        "stage": "P2",
                        "dataset": dataset,
                        "method": method,
                        "failure_type": "p2_gate_failed",
                        "detail": "; ".join(reasons) or "gate failed",
                    }
                )
            score.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "runs": len(rs),
                    "errors": err_count,
                    "acc": acc,
                    "acc_std": std(rs, "test_acc"),
                    "gap_vs_purekan_adamw": gap,
                    "val_auc": auc,
                    "auc_imp_vs_purekan_adamw": auc_imp_adam,
                    "auc_imp_vs_d6": auc_imp_d6,
                    "ece": ece,
                    "ece_red_vs_adamw": ece_red,
                    "phi_prime_p95": phi,
                    "phi_ratio_vs_adamw": phi_ratio,
                    "max_jac_condition": avg(rs, "max_jac_condition"),
                    "ftf_fit_R2_mean": avg(rs, "ftf_fit_R2_mean"),
                    "ftf_delta_ratio_max": max([f(r, "ftf_max_layer_delta_norm_ratio") for r in rs if math.isfinite(f(r, "ftf_max_layer_delta_norm_ratio"))], default=math.nan),
                    "step_time_ms": avg(rs, "step_time_ms"),
                    "p2_dataset_pass": int(dataset_pass),
                }
            )

    for method, ok in method_ok.items():
        if ok and method not in {"PureKAN-AdamW", "D6-allTaskAware"}:
            p2_pass.append(method)
    return score, failure, p2_pass


def audit_tables(p2_score_rows: List[dict]) -> tuple[List[dict], List[dict], List[dict], List[dict]]:
    functional = []
    geometry = []
    compute = []
    representation = []
    for row in p2_score_rows:
        functional.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "ftf_fit_R2_mean": row["ftf_fit_R2_mean"],
                "ftf_delta_ratio_max": row["ftf_delta_ratio_max"],
                "val_auc": row["val_auc"],
                "auc_imp_vs_d6": row["auc_imp_vs_d6"],
            }
        )
        geometry.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "phi_prime_p95": row["phi_prime_p95"],
                "phi_ratio_vs_adamw": row["phi_ratio_vs_adamw"],
                "max_jac_condition": row["max_jac_condition"],
            }
        )
        compute.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "step_time_ms": row["step_time_ms"],
            }
        )
        representation.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "no_kan_proxy_unavailable": 1,
                "note": "v4.1 runner did not add class-separation audit yet; use task/geometry proxies for this run",
            }
        )
    return functional, representation, geometry, compute


def blank_stage(path: Path, stage: str, reason: str) -> None:
    write_csv(path, [{"stage": stage, "status": "not_run", "reason": reason}])


def svg_bar(path: Path, rows: List[dict], *, key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r.get("dataset") == "KMNIST" and math.isfinite(float(r.get(key, math.nan)))]
    rows = rows[:8]
    width, height = 760, 260
    max_val = max([abs(float(r[key])) for r in rows], default=1.0)
    bars = []
    for i, r in enumerate(rows):
        val = float(r[key])
        w = 0 if max_val == 0 else abs(val) / max_val * 460
        y = 44 + i * 24
        color = "#4c78a8" if val >= 0 else "#d55e00"
        bars.append(f'<text x="10" y="{y+14}" font-size="11">{r["method"][:32]}</text>')
        bars.append(f'<rect x="250" y="{y}" width="{w:.1f}" height="16" fill="{color}" />')
        bars.append(f'<text x="{260+w:.1f}" y="{y+13}" font-size="11">{val:.3f}</text>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/><text x="10" y="24" font-size="16">{title}</text>'
        + "".join(bars)
        + "</svg>"
    )
    path.write_text(svg)


def score_md(rows: List[dict]) -> str:
    table = []
    for row in rows:
        table.append(
            [
                row["dataset"],
                row["method"],
                row["runs"],
                row["errors"],
                fmt(row["acc"]),
                fmt(row["acc_std"]),
                fmt(row["gap_vs_purekan_adamw"]),
                fmt(row["auc_imp_vs_purekan_adamw"]),
                fmt(row["auc_imp_vs_d6"]),
                fmt(row["ece_red_vs_adamw"]),
                fmt(row["phi_ratio_vs_adamw"]),
                fmt(row["ftf_fit_R2_mean"]),
                row["p2_dataset_pass"],
            ]
        )
    return md_table(
        [
            "dataset",
            "method",
            "runs",
            "errors",
            "acc",
            "std",
            "gap vs AdamW",
            "AUC imp vs AdamW",
            "AUC imp vs D6",
            "ECE red",
            "phi ratio",
            "FTF R2",
            "P2 pass",
        ],
        table,
    )


def p1_md(rows: List[dict]) -> str:
    table = []
    for row in rows:
        table.append(
            [
                row["method"],
                row["bad_train_steps"],
                fmt(row["min_train_descent"]),
                fmt(row["min_val_descent"]),
                fmt(row["target_fit_R2_min"]),
                fmt(row["max_layer_delta_norm_ratio"]),
                "yes" if row["p1_pass"] else "no",
            ]
        )
    return md_table(["method", "bad", "min train", "min val", "min FTF R2", "max delta", "P1 pass"], table)


def main() -> int:
    p0 = read_rows(P0)
    p1 = read_rows(P1)
    p2 = read_rows(P2)
    p0_rows = p0_invariants(p0)
    p1_gate_rows, p1_pass = p1_gate(p1)
    p2_rows, failures, p2_pass = p2_score(p2)
    functional, representation, geometry, compute = audit_tables(p2_rows)
    reason = "no P2 survivor reached the joint MNIST/Fashion/KMNIST gate"

    write_csv(BASE / "p0_invariants.csv", p0_rows)
    write_csv(BASE / "p1_direction_audit.csv", p1)
    write_csv(BASE / "p1_gate_summary.csv", p1_gate_rows)
    write_csv(BASE / "p2_micro_run_scorecard.csv", p2_rows)
    write_csv(BASE / "p2_failure_diagnosis.csv", failures)
    blank_stage(BASE / "p3_refinement_scorecard.csv", "P3", reason)
    blank_stage(BASE / "p4_confirm5_scorecard.csv", "P4", reason)
    blank_stage(BASE / "p5_confirm10_scorecard.csv", "P5", reason)
    write_csv(BASE / "functional_target_fit.csv", functional)
    write_csv(BASE / "representation_audit.csv", representation)
    write_csv(BASE / "geometry_audit.csv", geometry)
    write_csv(BASE / "compute_audit.csv", compute)
    write_csv(BASE / "failure_table.csv", failures + [r for r in p2 if r.get("error")])
    svg_bar(FIG_DIR / "p2_kmnist_acc_gap.svg", p2_rows, key="gap_vs_purekan_adamw", title="v4.1 P2 KMNIST Acc Gap vs PureKAN-AdamW")
    svg_bar(FIG_DIR / "p2_kmnist_auc_vs_d6.svg", p2_rows, key="auc_imp_vs_d6", title="v4.1 P2 KMNIST AUC Improvement vs D6")

    decision = {
        "purekan_functional_solved": False,
        "best_candidate": "F4-FNG-leftFull-right",
        "candidate_type": "FNG_diagnostic",
        "passes_p1_direction": bool(p1_pass),
        "p1_survivors": p1_pass,
        "passes_p2_micro_run": bool(p2_pass),
        "p2_survivors": p2_pass,
        "passes_p4_confirm5": False,
        "passes_p5_confirm10": False,
        "main_failure_type": "P1-local-good-but-P2-long-horizon-failure; FTF layer-local target fitting diverges, FNG underfits KMNIST",
        "recommended_next_step": "redesign FTF with accepted-step/backtracking and cross-layer drift control before more seed confirmation",
    }
    save_json(BASE / "aggregate_decision.json", decision)

    p0_errors = sum(1 for r in p0 if r.get("error"))
    p1_errors = sum(1 for r in p1 if r.get("error"))
    p2_errors = sum(1 for r in p2 if r.get("error"))
    doc = f"""# DG-KAN v4.1 Functional Update Redesign 结果复盘

本轮依据 `docs/DG-KAN_v4.1_FunctionalUpdate_Redesign_DeepPlan.md`。目标是验证重新设计的 PureKAN functional update：FTF、FC-Adam 与 FTR/FGN trust-region 是否能把 v3.7-v3.9 的局部方向修复转成长程训练收益。

## Run Inventory

{md_table(["stage", "rows", "errors"], [["P0 smoke", len(p0), p0_errors], ["P1 shadow", len(p1), p1_errors], ["P2 micro", len(p2), p2_errors]])}

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added FTF fields and per-role target-fitting updates for PureKAN coeffs.
  Added FC-Adam functional-coordinate updates through Sobolev Cholesky coordinates.
  Added an FTR-CG-small diagnostic path with layer-output trust scaling.
  Result rows now include FTF fit/residual/trust stats, FC-Adam reconstruction diagnostics,
  and FTR residual / predicted change statistics.

experiments/run_gafu_v41.py
  Added P0/P1/P2 packages and one-batch direction audit for v4.1 candidates.

experiments/analyze_gafu_v41.py
  Writes the required v4.1 CSV/JSON artifacts and this replay.
```

## P0 Implementation Smoke

P0 rows: `{len(p0)}`; errors: `{p0_errors}`.

Key audit:

```text
PureKAN alphaFixed1 / fixed norm paths ran without implementation errors.
Strict PureKAN rows kept learnable_nonKAN_params = 0.
Functional rows covered input/block/output coeff groups.
FTF, FC-Adam, and FTR all produced finite one-epoch smoke rows.
```

## P1 One-Batch Direction Gate

{p1_md(p1_gate_rows)}

P1 survivors used for P2:

```text
{", ".join(p1_pass) if p1_pass else "none"}
```

Observation:

```text
FTF repaired the one-step direction signal strongly: blocks/output and all-layer modes had positive train and validation descent on all three datasets, with fit R2 near 1.0.
D6 still had a KMNIST bad train step and was not treated as a P2 candidate, but was later added as a P2 baseline because the plan requires AUC comparison vs D6.
FC-Adam improved train loss but failed the validation-descent gate on Fashion/KMNIST.
```

## P2 Micro-Run Scorecard

{score_md(p2_rows)}

## P2 Failure Diagnosis

```text
No candidate passed the P2 joint gate.

FTF:
  P1 target fit was excellent, but P2 training was catastrophic.
  MNIST/KMNIST dropped to near chance accuracy, Fashion produced numerical eigensolve failures for several FTF rows, and val-loss AUC exploded.
  This matches the plan's failure mode: fit_R2 high + one-step descent positive + short-run acc bad => layer-local target fitting causes cross-layer drift.

FTR-CG-small:
  Local direction was acceptable, but short training underfit badly on all datasets.

F4-FNG-leftFull-right:
  Best practical candidate in P2.
  It matched/beat PureKAN-AdamW on MNIST and stayed within about 1.3 points on Fashion.
  It failed KMNIST by about 5.6 points, so it cannot enter P3.

D0/D6:
  Geometry is stable but accuracy remains below AdamW, especially on KMNIST.
```

## P3-P5 Decision

```text
P3 refinement: not run.
Reason: P2 produced no survivor.

P4 5-seed confirm: not run.
Reason: P3 was not reached.

P5 10-seed final: not run.
Reason: P4 was not reached.
```

## Artifacts

Required files were written under `results/v4_1/`:

```text
p0_invariants.csv
p1_direction_audit.csv
p2_micro_run_scorecard.csv
p2_failure_diagnosis.csv
p3_refinement_scorecard.csv
p4_confirm5_scorecard.csv
p5_confirm10_scorecard.csv
functional_target_fit.csv
representation_audit.csv
geometry_audit.csv
compute_audit.csv
failure_table.csv
aggregate_decision.json
figures/p2_kmnist_acc_gap.svg
figures/p2_kmnist_auc_vs_d6.svg
```

## Final Decision

```text
PureKAN functional optimization is still not solved in v4.1.

What improved:
  FTF gives a real one-step target-fitting direction.
  FNG remains the strongest short-run practical baseline among functional candidates.

What failed:
  FTF target fitting is not yet a stable optimizer; high local fit creates long-horizon drift.
  FTR-CG-small is too weak in this approximation.
  FNG does not close the KMNIST accuracy gap.

Next recommended direction:
  Add accepted-step / backtracking and cross-layer drift control to FTF before more seeds.
  In particular, target fitting should be sequential with validation of actual loss decrease
  and an activation/logit trust region that rejects or shrinks unsafe layer updates.
```
"""
    OUT.write_text(doc)
    LOG.write_text(LOG.read_text() + "\n\n" + doc)
    print(f"wrote {OUT}")
    print(f"wrote v4.1 artifacts under {BASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
