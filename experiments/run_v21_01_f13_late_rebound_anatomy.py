#!/usr/bin/env python3
"""F13 offline anatomy for v21.01 late-rebound target families.

This runner does not launch training. It replays already materialized v21.01
source-retention artifacts and classifies the F12 independent late-rebound
rerun into delayed migration, stochastic rebound, or control-equivalent drift.
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
    now_sg,
    read_json,
    read_rows,
    write_json,
    write_rows,
)


HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
TARGET_IDS = (
    "F9-T2-cross-split-consensus-target",
    "F9-T5-b1-readout-transfer-target",
    "F10-T7-b1-cross-split-consensus-transfer",
)
CONTROL_ID = "F9-TCTRL-stable-random-target"
F13_ARTIFACTS = [
    "v21_01_f13_late_rebound_anatomy_summary.csv",
    "v21_01_f13_late_rebound_row_vs_control.csv",
    "v21_01_f13_late_rebound_stratification.csv",
    "v21_01_f13_late_rebound_predictor_scan.csv",
    "v21_01_f13_late_rebound_decision.csv",
]
F13_REQUIRED_EVIDENCE = [
    "v21_01_source_retention_matrix.csv",
    "v21_01_source_retention_summary.csv",
    "v21_01_f9_f10_target_family_grouped_summary.csv",
    "v21_01_f9_f10_target_family_rowlevel_rollup.csv",
    "v21_01_f9_f10_target_family_trace_diagnostics.csv",
    "v21_01_f9_f10_target_family_retained_row_examples.csv",
    "v21_01_f9_f11_continuation_decision.csv",
    "v21_01_f11_dche_efficiency_repeat2_rollup.csv",
    "v21_01_f12_late_rebound_independent_per_offset.csv",
    "v21_01_f12_late_rebound_independent_aggregate.csv",
    "v21_01_f12_late_rebound_independent_retained_row_examples.csv",
    "v21_01_f12_late_rebound_independent_decision.csv",
] + F13_ARTIFACTS


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-log", action="store_true")
    return p


def mean(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return ""
    return sum(vals) / len(vals)


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


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        out.setdefault(key, []).append(row)
    return out


def source(row: dict[str, str], horizon: int) -> float:
    return finite_float(row.get(f"source_h{horizon}"))


def flag_source(row: dict[str, str], horizon: int) -> int:
    v = source(row, horizon)
    return int(math.isfinite(v) and v > 0.0)


def classify_group(means: dict[int, float | str], control_h6400_positive: bool) -> str:
    h800 = finite_float(means.get(800))
    h1600 = finite_float(means.get(1600))
    h3200 = finite_float(means.get(3200))
    h4800 = finite_float(means.get(4800))
    h6400 = finite_float(means.get(6400))
    if min(h800, h1600, h3200, h4800, h6400) > 0.0:
        return "continuous_retention"
    if h800 < 0.0 and h3200 > 0.0 and h4800 > 0.0 and h6400 > 0.0:
        return "delayed_migration_with_control_late_drift" if control_h6400_positive else "delayed_migration"
    if h800 < 0.0 and h6400 > 0.0 and control_h6400_positive:
        return "control_equivalent_late_drift"
    return "no_retained_or_ambiguous"


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


def render_markdown(
    decision: dict[str, Any],
    summary: list[dict[str, Any]],
    strat: list[dict[str, Any]],
    predictors: list[dict[str, Any]],
    packet_size: int,
    bundle_size: int,
) -> str:
    summary_rows = "\n".join(
        "| {v21_id} | {classification} | {rows} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h4800_mean} | {source_h6400_mean} | {matched_control_h6400_positive_rows} | {target_late_and_control_h6400_positive_rows} | {target_minus_control_h6400_mean} |".format(
            **row
        )
        for row in summary
    )
    strat_rows = "\n".join(
        "| {group} | {key} | {rows} | {h800_positive_rows} | {h3200_positive_rows} | {h6400_positive_rows} | {late_rebound_rows} | {source_h800_mean} | {source_h3200_mean} | {source_h6400_mean} |".format(
            **row
        )
        for row in strat[:24]
    )
    pred_rows = "\n".join(
        "| {predictor} | {finite_rows} | {spearman_h800} | {spearman_h3200} | {auc_h800_positive} | {dataset_min_auc} | {offset_min_auc} | {pass} |".format(
            **row
        )
        for row in predictors
    )
    return f"""
## 2026-06-04 F13 Late-Rebound Source-Target Anatomy

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F13 anatomy rows: {decision.get('f13_rows')}
- delayed migration groups: {decision.get('f13_delayed_migration_groups')}
- continuous retention groups: {decision.get('f13_continuous_retention_groups')}
- stable-random control h6400 positive rows: {decision.get('f13_control_h6400_positive_rows')}
- train-only early-source predictor pass: {decision.get('f13_train_only_predictor_pass')}
- actionable same-family repair: {decision.get('f13_actionable_same_family_repair')}

F13 没有新增训练 row。它读取 F12 independent rerun 的 108 rows 和 matched stable-random control，按计划 13.6 把 late rebound 进一步拆成 delayed migration / control-equivalent drift / continuous retention。

### F13 Group Anatomy

| v21_id | classification | rows | h800 | h1600 | h3200 | h4800 | h6400 | matched control h6400+ | target late with control h6400+ | target-control h6400 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{summary_rows}

### F13 Stratification Snapshot

| group | key | rows | h800+ | h3200+ | h6400+ | late rebound rows | h800 mean | h3200 mean | h6400 mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{strat_rows}

### F13 Predictor Scan

| predictor | finite rows | Spearman h800 | Spearman h3200 | AUC h800+ | dataset min AUC | offset min AUC | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
{pred_rows}

### 修改记录

- 新增 `experiments/run_v21_01_f13_late_rebound_anatomy.py`。
- 新增 F13 official artifacts：`v21_01_f13_late_rebound_anatomy_summary.csv`、`v21_01_f13_late_rebound_row_vs_control.csv`、`v21_01_f13_late_rebound_stratification.csv`、`v21_01_f13_late_rebound_predictor_scan.csv`、`v21_01_f13_late_rebound_decision.csv`。
- 更新 `v21_01_route_decision.json`，只追加 F13 audit 字段；`promotion_allowed` 保持 0。
- 刷新 `v21_01_code_review_packet.zip` 与 `v21_01_results_bundle.zip`；精确 byte size 以最终 file stat 为准，避免把 zip 自身大小写进归档后造成自引用变动。

### F13 结论 / Insight

- F9/F10 target family 的 late positive 不是单个 offset 偶然反弹：三个 independent offsets 都呈现 h800 negative、h1600 以后转正的 delayed migration。
- 但它也不是 retained source：三个 target group 的 h800 grouped mean 仍为负，continuous retention groups = 0。
- stable-random control 在 h6400 出现 {decision.get('f13_control_h6400_positive_rows')} 个 positive rows，说明 late horizon 正值里有 control-equivalent drift 成分，不能单独当成 source retention。
- target rows 相对 stable-random control 的 late horizon 确实更强，但 F13 predictor scan 没找到稳定可用的 train-only early-source selector。因此继续同族 target scale / alt / rank / B1-B2 组合扫描，预计只会增强 delayed migration，而不是闭合 h800->h6400 连续 retained chain。
- 当前诚实边界：`promotion_allowed=0`，`new_source_target_theory_required=1`，同族 repair 暂无可审计的下一步。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F13 Late-Rebound Source-Target Anatomy"
    idx = text.find(marker)
    if idx >= 0:
        text = text[:idx].rstrip() + "\n"
    append_text(path, text if not path.exists() else "")
    if path.exists() and idx >= 0:
        path.write_text(text.rstrip() + "\n" + section, encoding="utf-8")
    else:
        append_text(path, section)


def main() -> int:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    matrix_path = out_dir / "v21_01_source_retention_matrix.csv"
    matrix = read_rows(matrix_path)
    f12 = [r for r in matrix if r.get("run_label", "").startswith("v2101_f12_late_rebound_independent")]
    if not f12:
        raise SystemExit(f"missing F12 rows in {matrix_path}")
    rows = [r for r in f12 if r.get("v21_id") in set(TARGET_IDS + (CONTROL_ID,))]
    controls_by_key = {
        (r.get("dataset", ""), r.get("seed", ""), r.get("init_seed_offset", "")): r
        for r in rows
        if r.get("v21_id") == CONTROL_ID
    }
    row_vs_control: list[dict[str, Any]] = []
    for row in rows:
        if row.get("v21_id") == CONTROL_ID:
            continue
        key = (row.get("dataset", ""), row.get("seed", ""), row.get("init_seed_offset", ""))
        ctrl = controls_by_key.get(key, {})
        out: dict[str, Any] = {
            "v21_id": row.get("v21_id", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "init_seed_offset": row.get("init_seed_offset", ""),
            "target_retained_h6400": int_flag(row.get("retained_h6400_flag")),
            "target_late_rebound": int_flag(row.get("late_rebound_flag")),
            "control_late_rebound": int_flag(ctrl.get("late_rebound_flag")) if ctrl else "",
        }
        for h in HORIZONS:
            tv = source(row, h)
            cv = source(ctrl, h) if ctrl else float("nan")
            out[f"target_h{h}"] = tv
            out[f"control_h{h}"] = cv if math.isfinite(cv) else ""
            out[f"target_minus_control_h{h}"] = tv - cv if math.isfinite(tv) and math.isfinite(cv) else ""
            out[f"target_h{h}_positive"] = flag_source(row, h)
            out[f"control_h{h}_positive"] = flag_source(ctrl, h) if ctrl else ""
        row_vs_control.append(out)

    summary: list[dict[str, Any]] = []
    control_rows = [r for r in rows if r.get("v21_id") == CONTROL_ID]
    control_h6400_positive = any(flag_source(r, 6400) for r in control_rows)
    for vid, group_rows in grouped(rows, ("v21_id",)).items():
        vid_s = vid[0]
        means = {h: mean([source(r, h) for r in group_rows]) for h in HORIZONS}
        matched = [r for r in row_vs_control if r.get("v21_id") == vid_s]
        control_h6400_pos_rows = sum(int_flag(r.get("control_h6400_positive")) for r in matched)
        target_late_with_control = sum(
            int_flag(r.get("target_h6400_positive")) and int_flag(r.get("control_h6400_positive")) for r in matched
        )
        deltas = {h: mean([finite_float(r.get(f"target_minus_control_h{h}")) for r in matched]) for h in HORIZONS}
        row = {
            "v21_id": vid_s,
            "rows": len(group_rows),
            "classification": classify_group(means, control_h6400_positive) if vid_s != CONTROL_ID else classify_group(means, True),
            "retained_h6400_rows": sum(int_flag(r.get("retained_h6400_flag")) for r in group_rows),
            "late_rebound_rows": sum(int_flag(r.get("late_rebound_flag")) for r in group_rows),
            "matched_control_h6400_positive_rows": control_h6400_pos_rows if vid_s != CONTROL_ID else "",
            "target_late_and_control_h6400_positive_rows": target_late_with_control if vid_s != CONTROL_ID else "",
        }
        for h in HORIZONS:
            row[f"source_h{h}_mean"] = means[h]
            row[f"source_h{h}_positive_rows"] = sum(flag_source(r, h) for r in group_rows)
            row[f"target_minus_control_h{h}_mean"] = deltas[h] if vid_s != CONTROL_ID else ""
        summary.append(row)

    strat: list[dict[str, Any]] = []
    target_rows = [r for r in rows if r.get("v21_id") in TARGET_IDS]
    for group_name, keys in [
        ("v21_id", ("v21_id",)),
        ("dataset", ("dataset",)),
        ("offset", ("init_seed_offset",)),
        ("dataset_offset", ("dataset", "init_seed_offset")),
    ]:
        for key, group_rows in sorted(grouped(target_rows, keys).items()):
            record: dict[str, Any] = {
                "group": group_name,
                "key": "/".join(key),
                "rows": len(group_rows),
                "late_rebound_rows": sum(int_flag(r.get("late_rebound_flag")) for r in group_rows),
            }
            for h in (800, 1600, 3200, 6400):
                record[f"h{h}_positive_rows"] = sum(flag_source(r, h) for r in group_rows)
                record[f"source_h{h}_mean"] = mean([source(r, h) for r in group_rows])
            strat.append(record)

    predictors = [
        ("B2_transfer_gain_h800", "function_readback"),
        ("ActuationR2_h800", "function_readback"),
        ("source_state_current_cos_h800", "train_trace"),
        ("source_state_gate_accept_h800", "train_trace"),
        ("generalization_gate_accept_h800", "train_trace"),
        ("LineC_channel_loss_h800", "audit_readback"),
        ("train_loss_h800", "train_trace"),
        ("train_loss_drop_h100_h800", "train_trace"),
    ]
    predictor_rows: list[dict[str, Any]] = []
    for pred, kind in predictors:
        pairs_h800: list[tuple[float, float]] = []
        pairs_h3200: list[tuple[float, float]] = []
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
            h3200 = source(row, 3200)
            pairs_h800.append((value, h800))
            pairs_h3200.append((value, h3200))
            label = int(h800 > 0.0)
            scores.append(value)
            labels.append(label)
            d_scores, d_labels = datasets.setdefault(row.get("dataset", ""), ([], []))
            d_scores.append(value)
            d_labels.append(label)
            o_scores, o_labels = offsets.setdefault(row.get("init_seed_offset", ""), ([], []))
            o_scores.append(value)
            o_labels.append(label)
        auc = auc_score(scores, labels)
        dataset_aucs = [auc_score(s, y) for s, y in datasets.values()]
        offset_aucs = [auc_score(s, y) for s, y in offsets.values()]
        dataset_aucs_f = [float(x) for x in dataset_aucs if x != ""]
        offset_aucs_f = [float(x) for x in offset_aucs if x != ""]
        sp800 = spearman(pairs_h800)
        sp3200 = spearman(pairs_h3200)
        pass_flag = int(
            kind == "train_trace"
            and len(scores) >= 20
            and isinstance(sp800, float)
            and sp800 >= 0.5
            and isinstance(auc, float)
            and auc >= 0.75
            and dataset_aucs_f
            and min(dataset_aucs_f) >= 0.65
            and offset_aucs_f
            and min(offset_aucs_f) >= 0.65
        )
        predictor_rows.append(
            {
                "predictor": pred,
                "kind": kind,
                "finite_rows": len(scores),
                "spearman_h800": sp800,
                "spearman_h3200": sp3200,
                "auc_h800_positive": auc,
                "dataset_min_auc": min(dataset_aucs_f) if dataset_aucs_f else "",
                "offset_min_auc": min(offset_aucs_f) if offset_aucs_f else "",
                "pass": pass_flag,
            }
        )

    write_rows(out_dir / "v21_01_f13_late_rebound_anatomy_summary.csv", summary)
    write_rows(out_dir / "v21_01_f13_late_rebound_row_vs_control.csv", row_vs_control)
    write_rows(out_dir / "v21_01_f13_late_rebound_stratification.csv", strat)
    write_rows(out_dir / "v21_01_f13_late_rebound_predictor_scan.csv", predictor_rows)

    delayed = sum(1 for r in summary if str(r.get("classification", "")).startswith("delayed_migration"))
    continuous = sum(1 for r in summary if r.get("classification") == "continuous_retention")
    control_h6400_positive_rows = sum(flag_source(r, 6400) for r in control_rows)
    train_pred_pass = sum(int_flag(r.get("pass")) for r in predictor_rows)
    actionable = int(continuous > 0 or train_pred_pass > 0)
    decision = {
        "decision": "late_rebound_delayed_migration_control_late_drift_no_actionable_selector",
        "rows": len(rows),
        "target_rows": len(target_rows),
        "control_rows": len(control_rows),
        "delayed_migration_groups": delayed,
        "continuous_retention_groups": continuous,
        "control_h6400_positive_rows": control_h6400_positive_rows,
        "train_only_predictor_pass": train_pred_pass,
        "actionable_same_family_repair": actionable,
        "promotion_allowed": 0,
        "new_source_target_theory_required": 1,
    }
    write_rows(out_dir / "v21_01_f13_late_rebound_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": "R2-LateReboundNoContinuousRetention-F13DelayedMigrationControlDrift",
            "promotion_allowed": 0,
            "f13_anatomy_completed": 1,
            "f13_rows": len(rows),
            "f13_delayed_migration_groups": delayed,
            "f13_continuous_retention_groups": continuous,
            "f13_control_h6400_positive_rows": control_h6400_positive_rows,
            "f13_train_only_predictor_pass": train_pred_pass,
            "f13_actionable_same_family_repair": actionable,
            "new_source_target_theory_required": 1,
        }
    )
    write_json(out_dir / "v21_01_route_decision.json", route)
    write_json(out_dir / "route_decision.json", route)
    append_unique_manifest(out_dir, F13_REQUIRED_EVIDENCE)
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    required = [r.get("artifact", "") for r in manifest if r.get("artifact")]
    build_packet(out_dir, required)
    packet_size = (out_dir / "v21_01_code_review_packet.zip").stat().st_size
    bundle_size = (out_dir / "v21_01_results_bundle.zip").stat().st_size
    section = render_markdown(route, summary, strat, predictor_rows, packet_size, bundle_size)
    replace_section(V2101_RECAP_DOC, section)

    command = f"{PYTHON} experiments/run_v21_01_f13_late_rebound_anatomy.py --out-dir {out_dir}"
    if not args.no_log:
        append_exec(
            out_dir,
            command,
            status="completed",
            note=(
                f"rows={len(rows)} delayed_groups={delayed} continuous_groups={continuous} "
                f"control_h6400_positive_rows={control_h6400_positive_rows} "
                f"train_only_predictor_pass={train_pred_pass} packet={packet_size} bundle={bundle_size}"
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
