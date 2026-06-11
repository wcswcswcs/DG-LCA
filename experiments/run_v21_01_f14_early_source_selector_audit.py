#!/usr/bin/env python3
"""F14 offline early-source selector audit for v21.01.

This script does not run training. It audits existing v21.01 source-retention
artifacts for two questions:

1. Is there any train-only early-source selector that can predict h800 closure?
2. Is there a near-early-closure target that deserves independent confirmation?
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v21_01_common import (  # noqa: E402
    PYTHON,
    V2101_RECAP_DOC,
    append_exec,
    append_text,
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    read_json,
    read_rows,
    write_json,
    write_rows,
)


F14_ARTIFACTS = [
    "v21_01_f14_early_source_candidate_ranking.csv",
    "v21_01_f14_early_source_predictor_scan.csv",
    "v21_01_f14_early_source_decision.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-log", action="store_true")
    return p


def mean(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else ""


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j + 1) / 2.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def pearson(xs: list[float], ys: list[float]) -> float | str:
    if len(xs) < 3 or len(xs) != len(ys):
        return ""
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return ""
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(pairs: list[tuple[float, float]]) -> float | str:
    clean = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(clean) < 3:
        return ""
    xs, ys = zip(*clean)
    return pearson(ranks(list(xs)), ranks(list(ys)))


def auc_score(scores: list[float], labels: list[int]) -> float | str:
    pairs = [(s, int(y)) for s, y in zip(scores, labels) if math.isfinite(s)]
    pos = sum(1 for _, y in pairs if y == 1)
    neg = sum(1 for _, y in pairs if y == 0)
    if pos == 0 or neg == 0:
        return ""
    ranked = ranks([s for s, _ in pairs])
    rank_pos = sum(r for r, (_, y) in zip(ranked, pairs) if y == 1)
    return (rank_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def source(row: dict[str, str], horizon: int) -> float:
    return finite_float(row.get(f"source_h{horizon}"))


def grouped(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row.get(key, ""), []).append(row)
    return out


def append_unique_manifest(out_dir: Path, artifacts: list[str]) -> None:
    path = out_dir / "v21_01_required_artifact_manifest.csv"
    rows = read_rows(path)
    seen = {r.get("artifact") for r in rows}
    for artifact in artifacts:
        if artifact not in seen:
            rows.append({"artifact": artifact, "exists": int((out_dir / artifact).exists()), "required": 1})
        else:
            for row in rows:
                if row.get("artifact") == artifact:
                    row["exists"] = int((out_dir / artifact).exists())
                    row["required"] = row.get("required") or 1
    write_rows(path, rows)
    write_rows(out_dir / "required_manifest.csv", rows)


def render_section(decision: dict[str, Any], ranking: list[dict[str, Any]], predictors: list[dict[str, Any]]) -> str:
    rank_rows = "\n".join(
        "| {rank} | {v21_id} | {rows} | {h800_gap} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h6400_mean} | {h800_positive_rows} | {late_positive_chain} | {independent_confirmation_recommended} |".format(
            **r
        )
        for r in ranking[:12]
    )
    pred_rows = "\n".join(
        "| {predictor} | {kind} | {finite_rows} | {spearman_h800} | {auc_h800_positive} | {dataset_min_auc} | {offset_min_auc} | {pass} |".format(
            **r
        )
        for r in predictors
    )
    return f"""
## 2026-06-04 F14 Early-Source Selector / Near-Closure Audit

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- train-only selector pass: {decision.get('f14_train_only_selector_pass')}
- near-early-closure candidates: {decision.get('f14_near_early_closure_candidates')}
- F15 independent confirmation recommended: {decision.get('f14_f15_independent_recommended')}
- best candidate: `{decision.get('f14_best_candidate')}`

F14 没有新增训练 row。它读取已有 v21.01 source-retention matrix / summary，专门检查是否存在可提前使用的 train-only h800 selector，并排序所有 D-FOU target/KSW2 late-positive 候选。

### F14 Candidate Ranking

| rank | v21_id | rows | h800 gap | h800 | h1600 | h3200 | h6400 | h800+ rows | late chain | F15 recommended |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rank_rows}

### F14 Train-Only Predictor Scan

| predictor | kind | finite rows | Spearman h800 | AUC h800+ | dataset min AUC | offset min AUC | pass |
|---|---|---:|---:|---:|---:|---:|---:|
{pred_rows}

### F14 结论 / 下一步

- F14 没有找到合法 train-only early-source selector：所有 train-only predictors 的 pass 都为 0。
- 但 D-FOU `F3-T1-loss-cotangent-target` 是明确的 near-early-closure candidate：h800 gap 最小，且 h1600/h3200/h6400 均为正。
- 这不能 promotion，也不能写 retained source；它只允许一个计划内 F15 independent confirmation：用 x3 init offsets + matched random/sign-flip/corrupt/stable-random controls 检查它是否只是 offset0 近似闭合。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F14 Early-Source Selector / Near-Closure Audit"
    idx = text.find(marker)
    if idx >= 0:
        path.write_text(text[:idx].rstrip() + "\n" + section, encoding="utf-8")
    else:
        append_text(path, section)


def main() -> int:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    matrix = read_rows(out_dir / "v21_01_source_retention_matrix.csv")
    summary = read_rows(out_dir / "v21_01_source_retention_summary.csv")
    target_summary = []
    control_tokens = ("CTRL", "random", "sign", "corrupt", "TCTRL")
    for row in summary:
        if row.get("carrier") != "D-FOU":
            continue
        vid = row.get("v21_id", "")
        if not any(tok in vid for tok in ("F3-T", "F9-T", "F10-T", "KSW2")):
            continue
        if any(tok in vid for tok in control_tokens):
            continue
        h800 = finite_float(row.get("source_h800_mean"))
        h1600 = finite_float(row.get("source_h1600_mean"))
        h3200 = finite_float(row.get("source_h3200_mean"))
        h6400 = finite_float(row.get("source_h6400_mean"))
        late_chain = int(h1600 > 0.0 and h3200 > 0.0 and h6400 > 0.0)
        h800_gap = max(0.0, -h800) if math.isfinite(h800) else float("inf")
        recommend = int(late_chain and h800_gap <= 0.02 and int(float(row.get("rows", 0) or 0)) >= 9)
        target_summary.append(
            {
                "v21_id": vid,
                "rows": row.get("rows", ""),
                "h800_gap": h800_gap,
                "source_h800_mean": h800,
                "source_h1600_mean": h1600,
                "source_h3200_mean": h3200,
                "source_h6400_mean": h6400,
                "h800_positive_rows": row.get("source_h800_pass_count", ""),
                "late_positive_chain": late_chain,
                "independent_confirmation_recommended": recommend,
            }
        )
    ranking = sorted(target_summary, key=lambda r: (0 if int_flag(r.get("late_positive_chain")) else 1, finite_float(r.get("h800_gap"), 999.0)))
    for i, row in enumerate(ranking, start=1):
        row["rank"] = i

    target_rows = [
        r
        for r in matrix
        if r.get("carrier") == "D-FOU"
        and any(tok in r.get("v21_id", "") for tok in ("F3-T", "F9-T", "F10-T", "KSW2"))
        and not any(tok in r.get("v21_id", "") for tok in control_tokens)
    ]
    predictors = [
        ("train_loss_h100", "train_only"),
        ("train_loss_h400", "train_only"),
        ("train_loss_h800", "train_only"),
        ("train_loss_drop_h100_h800", "train_only"),
        ("CEp99_h800", "audit_metric"),
        ("ECE_h800", "audit_metric"),
        ("Brier_h800", "audit_metric"),
        ("LineC_channel_loss_h800", "audit_metric"),
        ("B2_transfer_gain_h800", "function_readback"),
        ("ActuationR2_h800", "function_readback"),
        ("source_h400", "source_readback_forbidden"),
    ]
    predictor_rows: list[dict[str, Any]] = []
    for pred, kind in predictors:
        pairs: list[tuple[float, float]] = []
        scores: list[float] = []
        labels: list[int] = []
        datasets: dict[str, tuple[list[float], list[int]]] = {}
        offsets: dict[str, tuple[list[float], list[int]]] = {}
        for row in target_rows:
            if pred == "train_loss_drop_h100_h800":
                value = finite_float(row.get("train_loss_h100")) - finite_float(row.get("train_loss_h800"))
            else:
                value = finite_float(row.get(pred))
            if not math.isfinite(value):
                continue
            h800 = source(row, 800)
            pairs.append((value, h800))
            label = int(h800 > 0.0)
            scores.append(value)
            labels.append(label)
            ds = row.get("dataset", "")
            of = row.get("init_seed_offset", "")
            datasets.setdefault(ds, ([], []))[0].append(value)
            datasets.setdefault(ds, ([], []))[1].append(label)
            offsets.setdefault(of, ([], []))[0].append(value)
            offsets.setdefault(of, ([], []))[1].append(label)
        auc = auc_score(scores, labels)
        ds_aucs = [auc_score(s, y) for s, y in datasets.values()]
        of_aucs = [auc_score(s, y) for s, y in offsets.values()]
        ds_f = [float(x) for x in ds_aucs if x != ""]
        of_f = [float(x) for x in of_aucs if x != ""]
        sp = spearman(pairs)
        pass_flag = int(
            kind == "train_only"
            and len(scores) >= 30
            and isinstance(sp, float)
            and sp >= 0.5
            and isinstance(auc, float)
            and auc >= 0.75
            and ds_f
            and min(ds_f) >= 0.65
            and of_f
            and min(of_f) >= 0.65
        )
        predictor_rows.append(
            {
                "predictor": pred,
                "kind": kind,
                "finite_rows": len(scores),
                "spearman_h800": sp,
                "auc_h800_positive": auc,
                "dataset_min_auc": min(ds_f) if ds_f else "",
                "offset_min_auc": min(of_f) if of_f else "",
                "pass": pass_flag,
            }
        )

    best = next((r for r in ranking if int_flag(r.get("independent_confirmation_recommended"))), ranking[0] if ranking else {})
    selector_pass = sum(int_flag(r.get("pass")) for r in predictor_rows)
    near = sum(int_flag(r.get("independent_confirmation_recommended")) for r in ranking)
    f15 = int(selector_pass == 0 and near > 0)
    decision = {
        "decision": "near_early_closure_candidate_requires_independent_confirmation" if f15 else "no_actionable_early_source_selector",
        "train_only_selector_pass": selector_pass,
        "near_early_closure_candidates": near,
        "f15_independent_recommended": f15,
        "best_candidate": best.get("v21_id", ""),
        "best_candidate_h800_gap": best.get("h800_gap", ""),
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v21_01_f14_early_source_candidate_ranking.csv", ranking)
    write_rows(out_dir / "v21_01_f14_early_source_predictor_scan.csv", predictor_rows)
    write_rows(out_dir / "v21_01_f14_early_source_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "promotion_allowed": 0,
            "f14_audit_completed": 1,
            "f14_train_only_selector_pass": selector_pass,
            "f14_near_early_closure_candidates": near,
            "f14_f15_independent_recommended": f15,
            "f14_best_candidate": best.get("v21_id", ""),
        }
    )
    write_json(out_dir / "v21_01_route_decision.json", route)
    write_json(out_dir / "route_decision.json", route)
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    seen = {r.get("artifact") for r in manifest}
    for artifact in F14_ARTIFACTS:
        if artifact not in seen:
            manifest.append({"artifact": artifact, "exists": int((out_dir / artifact).exists()), "required": 1})
        else:
            for row in manifest:
                if row.get("artifact") == artifact:
                    row["exists"] = int((out_dir / artifact).exists())
                    row["required"] = 1
    write_rows(out_dir / "v21_01_required_artifact_manifest.csv", manifest)
    write_rows(out_dir / "required_manifest.csv", manifest)
    section = render_section({**route, **{f"f14_{k}": v for k, v in decision.items()}}, ranking, predictor_rows)
    replace_section(V2101_RECAP_DOC, section)
    required = [r.get("artifact", "") for r in manifest if r.get("artifact")]
    build_packet(out_dir, required)
    if not args.no_log:
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v21_01_f14_early_source_selector_audit.py --out-dir {out_dir}",
            status="completed",
            note=f"selector_pass={selector_pass} near_candidates={near} best={best.get('v21_id','')} f15_recommended={f15}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
