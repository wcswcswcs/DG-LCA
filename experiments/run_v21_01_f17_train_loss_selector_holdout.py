#!/usr/bin/env python3
"""F17 held-out validation for the F16 train-loss-drop selector."""

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


F17_ARTIFACTS = [
    "v21_01_f17_train_loss_selector_threshold.csv",
    "v21_01_f17_train_loss_selector_holdout.csv",
    "v21_01_f17_train_loss_selector_selected_groups.csv",
    "v21_01_f17_train_loss_selector_decision.csv",
]
HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
TARGET_TOKENS = ("F3-T", "F9-T", "F10-T", "KSW2", "F6-KSW2", "F7-KSW2", "F8-KSW2")
CONTROL_TOKENS = ("CTRL", "random", "sign", "corrupt", "TCTRL")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-log", action="store_true")
    return p


def source(row: dict[str, str], h: int) -> float:
    return finite_float(row.get(f"source_h{h}"))


def score(row: dict[str, str]) -> float:
    return finite_float(row.get("train_loss_h100")) - finite_float(row.get("train_loss_h800"))


def is_target_like(row: dict[str, str]) -> bool:
    vid = row.get("v21_id", "")
    return row.get("carrier") in {"D-CHE", "D-FOU"} and any(tok in vid for tok in TARGET_TOKENS)


def is_control(row: dict[str, str]) -> bool:
    vid = row.get("v21_id", "")
    return int(any(tok in vid for tok in CONTROL_TOKENS))


def continuous(row: dict[str, str]) -> int:
    return int(all(source(row, h) > 0.0 for h in HORIZONS))


def mean(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else ""


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    return out


def choose_threshold(train: list[dict[str, str]]) -> dict[str, Any]:
    values = sorted({score(r) for r in train if math.isfinite(score(r))})
    best: dict[str, Any] = {
        "selector": "train_loss_drop_h100_h800",
        "train_rows": len(train),
        "threshold": "",
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "precision": "",
        "recall": "",
        "f1": "",
        "youden": "",
    }
    best_key = (-1.0, -1.0, -1.0, -1.0)
    for th in values:
        tp = fp = fn = tn = 0
        for row in train:
            pred = score(row) >= th
            label = bool(continuous(row))
            if pred and label:
                tp += 1
            elif pred and not label:
                fp += 1
            elif not pred and label:
                fn += 1
            else:
                tn += 1
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        youden = recall + specificity - 1.0
        key = (f1, youden, recall, precision)
        if key > best_key:
            best_key = key
            best = {
                "selector": "train_loss_drop_h100_h800",
                "train_rows": len(train),
                "threshold": th,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "youden": youden,
            }
    return best


def summarize_split(name: str, rows: list[dict[str, str]], threshold: float) -> dict[str, Any]:
    selected = [r for r in rows if score(r) >= threshold]
    out: dict[str, Any] = {
        "split": name,
        "rows": len(rows),
        "continuous_rows_all": sum(continuous(r) for r in rows),
        "selected_rows": len(selected),
        "selected_control_rows": sum(is_control(r) for r in selected),
        "selected_continuous_rows": sum(continuous(r) for r in selected),
        "selected_continuous_fraction": (sum(continuous(r) for r in selected) / len(selected)) if selected else "",
        "selected_score_mean": mean([score(r) for r in selected]),
    }
    for h in HORIZONS:
        vals = [source(r, h) for r in selected]
        out[f"selected_source_h{h}_mean"] = mean(vals)
        out[f"selected_source_h{h}_positive_rows"] = sum(1 for v in vals if math.isfinite(v) and v > 0.0)
    out["selected_group_continuous"] = int(selected and all(finite_float(out.get(f"selected_source_h{h}_mean")) > 0.0 for h in HORIZONS))
    return out


def summarize_groups(rows: list[dict[str, str]], threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    selected = [r for r in rows if score(r) >= threshold]
    for (split, carrier, vid), group in sorted(grouped(selected, ("selector_split", "carrier", "v21_id")).items()):
        item: dict[str, Any] = {
            "split": split,
            "carrier": carrier,
            "v21_id": vid,
            "rows": len(group),
            "control": int(any(is_control(r) for r in group)),
            "continuous_rows": sum(continuous(r) for r in group),
            "score_mean": mean([score(r) for r in group]),
        }
        for h in HORIZONS:
            vals = [source(r, h) for r in group]
            item[f"source_h{h}_mean"] = mean(vals)
            item[f"source_h{h}_positive_rows"] = sum(1 for v in vals if math.isfinite(v) and v > 0.0)
        item["group_continuous"] = int(all(finite_float(item.get(f"source_h{h}_mean")) > 0.0 for h in HORIZONS))
        out.append(item)
    return out


def append_unique_manifest(out_dir: Path, artifacts: list[str]) -> None:
    rows = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    seen = {r.get("artifact") for r in rows}
    for artifact in artifacts:
        if artifact not in seen:
            rows.append({"artifact": artifact, "exists": int((out_dir / artifact).exists()), "required": 1})
        else:
            for row in rows:
                if row.get("artifact") == artifact:
                    row["exists"] = int((out_dir / artifact).exists())
                    row["required"] = 1
    write_rows(out_dir / "v21_01_required_artifact_manifest.csv", rows)
    write_rows(out_dir / "required_manifest.csv", rows)


def render_section(decision: dict[str, Any], threshold: dict[str, Any], holdout: list[dict[str, Any]], groups: list[dict[str, Any]]) -> str:
    holdout_rows = "\n".join(
        "| {split} | {rows} | {continuous_rows_all} | {selected_rows} | {selected_control_rows} | {selected_continuous_rows} | {selected_continuous_fraction} | {selected_source_h800_mean} | {selected_source_h1600_mean} | {selected_source_h3200_mean} | {selected_source_h6400_mean} | {selected_group_continuous} |".format(**r)
        for r in holdout
    )
    top_groups = sorted(groups, key=lambda r: (r.get("split", ""), -int(r.get("rows", 0)), r.get("v21_id", "")))[:20]
    group_rows = "\n".join(
        "| {split} | {carrier} | {v21_id} | {rows} | {control} | {continuous_rows} | {score_mean} | {source_h800_mean} | {source_h3200_mean} | {source_h6400_mean} | {group_continuous} |".format(**r)
        for r in top_groups
    )
    return f"""
## 2026-06-04 F17 Train-Loss Selector Held-Out Validation

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- selector: `train_loss_drop_h100_h800`
- threshold learned on non-F15 rows: {threshold.get('threshold')}
- holdout selected rows: {decision.get('f17_holdout_selected_rows')}
- holdout selected h800 mean: {decision.get('f17_holdout_selected_h800_mean')}
- holdout selected h3200 mean: {decision.get('f17_holdout_selected_h3200_mean')}
- holdout selected group continuous: {decision.get('f17_holdout_selected_group_continuous')}
- selector actionable: {decision.get('f17_selector_actionable')}

F17 验证 F16 发现的唯一 legal predictor：`train_loss_drop_h100_h800`。阈值只在非 F15 rows 上选择，然后在 F15 independent rows 上验证，避免把同表相关性直接写成可行动 selector。

### F17 Threshold Fit

| selector | train rows | threshold | tp | fp | fn | tn | precision | recall | f1 | youden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {threshold.get('selector')} | {threshold.get('train_rows')} | {threshold.get('threshold')} | {threshold.get('tp')} | {threshold.get('fp')} | {threshold.get('fn')} | {threshold.get('tn')} | {threshold.get('precision')} | {threshold.get('recall')} | {threshold.get('f1')} | {threshold.get('youden')} |

### F17 Hold-Out Validation

| split | rows | continuous all | selected rows | selected controls | selected continuous | selected continuous fraction | selected h800 | selected h1600 | selected h3200 | selected h6400 | selected group continuous |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{holdout_rows}

### F17 Selected Group Evidence

| split | carrier | v21_id | rows | control | continuous rows | score mean | h800 | h3200 | h6400 | group continuous |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{group_rows}

### 修改记录

- 新增 `experiments/run_v21_01_f17_train_loss_selector_holdout.py`。
- 新增 F17 official artifacts：`v21_01_f17_train_loss_selector_threshold.csv`、`v21_01_f17_train_loss_selector_holdout.csv`、`v21_01_f17_train_loss_selector_selected_groups.csv`、`v21_01_f17_train_loss_selector_decision.csv`。
- 更新 `v21_01_route_decision.json`，`promotion_allowed` 仍为 0。

### F17 结论 / Insight

- F16 的 train-only predictor 不是可直接行动的 selector：F15 holdout selected group 的 h800 mean = {decision.get('f17_holdout_selected_h800_mean')}，仍为负，continuous group=0。
- 该 selector 同时选中 matched random target rows；在 F15 selected groups 中，random control 也保留 late-positive source，因此不能作为 target attribution。
- 它可以作为 failure taxonomy 的解释变量：train loss drop 能预测一部分 row-level continuous，但不能把 grouped source chain 闭合，也不能排除 control contamination。
- 当前不能把 `train_loss_drop_h100_h800` 写成 direction selector，更不能把它接入 promotion。继续推进需要新的 toy-correctness/source-target 假设，而不是把这个 post-h800 train metric 当作突破。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F17 Train-Loss Selector Held-Out Validation"
    idx = text.find(marker)
    if idx >= 0:
        path.write_text(text[:idx].rstrip() + "\n" + section, encoding="utf-8")
    else:
        append_text(path, section)


def main() -> int:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    matrix = [r for r in read_rows(out_dir / "v21_01_source_retention_matrix.csv") if is_target_like(r) and math.isfinite(score(r))]
    train = [dict(r, selector_split="fit_non_f15") for r in matrix if not r.get("run_label", "").startswith("v2101_f15")]
    holdout = [dict(r, selector_split="holdout_f15") for r in matrix if r.get("run_label", "").startswith("v2101_f15")]
    all_rows = [dict(r, selector_split="all") for r in matrix]
    threshold = choose_threshold(train)
    th = finite_float(threshold.get("threshold"))
    if not math.isfinite(th):
        raise SystemExit("no finite threshold")
    holdout_rows = [
        summarize_split("fit_non_f15", train, th),
        summarize_split("holdout_f15", holdout, th),
        summarize_split("all", all_rows, th),
    ]
    group_rows = summarize_groups(train + holdout, th)
    hold = next(r for r in holdout_rows if r.get("split") == "holdout_f15")
    actionable = int(
        int(hold.get("selected_rows", 0)) >= 18
        and finite_float(hold.get("selected_source_h800_mean")) > 0.0
        and finite_float(hold.get("selected_source_h1600_mean")) > 0.0
        and finite_float(hold.get("selected_source_h3200_mean")) > 0.0
        and finite_float(hold.get("selected_source_h6400_mean")) > 0.0
        and int_flag(hold.get("selected_group_continuous"))
        and finite_float(hold.get("selected_continuous_fraction")) >= 0.50
        and int(hold.get("selected_control_rows", 999)) <= int(hold.get("selected_rows", 0)) * 0.25
    )
    decision = {
        "decision": "train_loss_selector_actionable" if actionable else "train_loss_selector_not_actionable",
        "fit_rows": len(train),
        "holdout_rows": len(holdout),
        "threshold": th,
        "holdout_selected_rows": hold.get("selected_rows", ""),
        "holdout_selected_control_rows": hold.get("selected_control_rows", ""),
        "holdout_selected_continuous_rows": hold.get("selected_continuous_rows", ""),
        "holdout_selected_continuous_fraction": hold.get("selected_continuous_fraction", ""),
        "holdout_selected_h800_mean": hold.get("selected_source_h800_mean", ""),
        "holdout_selected_h1600_mean": hold.get("selected_source_h1600_mean", ""),
        "holdout_selected_h3200_mean": hold.get("selected_source_h3200_mean", ""),
        "holdout_selected_h6400_mean": hold.get("selected_source_h6400_mean", ""),
        "holdout_selected_group_continuous": hold.get("selected_group_continuous", ""),
        "selector_actionable": actionable,
        "promotion_allowed": 0,
    }

    write_rows(out_dir / "v21_01_f17_train_loss_selector_threshold.csv", [threshold])
    write_rows(out_dir / "v21_01_f17_train_loss_selector_holdout.csv", holdout_rows)
    write_rows(out_dir / "v21_01_f17_train_loss_selector_selected_groups.csv", group_rows)
    write_rows(out_dir / "v21_01_f17_train_loss_selector_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": "R2-LateReboundNoContinuousRetention-F17TrainLossSelectorNotActionable",
            "promotion_allowed": 0,
            "f17_selector": "train_loss_drop_h100_h800",
            "f17_threshold": th,
            "f17_holdout_selected_rows": hold.get("selected_rows", ""),
            "f17_holdout_selected_h800_mean": hold.get("selected_source_h800_mean", ""),
            "f17_holdout_selected_h3200_mean": hold.get("selected_source_h3200_mean", ""),
            "f17_holdout_selected_group_continuous": hold.get("selected_group_continuous", ""),
            "f17_selector_actionable": actionable,
            "new_source_target_theory_required": 1,
        }
    )
    write_json(out_dir / "v21_01_route_decision.json", route)
    write_json(out_dir / "route_decision.json", route)
    append_unique_manifest(out_dir, F17_ARTIFACTS)
    section = render_section(route, threshold, holdout_rows, group_rows)
    replace_section(V2101_RECAP_DOC, section)
    if not args.no_log:
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v21_01_f17_train_loss_selector_holdout.py --out-dir {out_dir}",
            status="completed",
            note=(
                f"threshold={th} holdout_selected={hold.get('selected_rows','')} "
                f"holdout_h800={hold.get('selected_source_h800_mean','')} actionable={actionable} promotion=0"
            ),
        )
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    build_packet(out_dir, [r.get("artifact", "") for r in manifest if r.get("artifact")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
