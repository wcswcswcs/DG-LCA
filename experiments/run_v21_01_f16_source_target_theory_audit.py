#!/usr/bin/env python3
"""F16 source-target theory audit for v21.01.

This is a no-new-FU-token audit. It uses existing v21.01 matrix rows to test
whether target/source diagnostics can legally distinguish continuous retained
source from delayed migration or control-equivalent late drift.
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


F16_ARTIFACTS = [
    "v21_01_f16_source_target_theory_summary.csv",
    "v21_01_f16_pairwise_target_control_advantage.csv",
    "v21_01_f16_observability_predictor_scan.csv",
    "v21_01_f16_route_closure_decision.csv",
]
HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
TARGET_TOKENS = ("F3-T", "F9-T", "F10-T", "KSW2", "F6-KSW2", "F7-KSW2", "F8-KSW2")
CONTROL_TOKENS = ("CTRL", "random", "sign", "corrupt", "TCTRL")
CANDIDATE = "F3-T1-loss-cotangent-target"
F15_CONTROLS = (
    "F3-T5-random-matched-target",
    "F3-T6-sign-flipped-target",
    "F3-T7-corrupted-label-target",
    "F9-TCTRL-stable-random-target",
)


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


def is_target_like(row: dict[str, str]) -> bool:
    vid = row.get("v21_id", "")
    return row.get("carrier") in {"D-CHE", "D-FOU"} and any(tok in vid for tok in TARGET_TOKENS)


def is_control(row: dict[str, str]) -> bool:
    vid = row.get("v21_id", "")
    return any(tok in vid for tok in CONTROL_TOKENS)


def continuous(row: dict[str, str]) -> int:
    return int(all(source(row, h) > 0.0 for h in HORIZONS))


def delayed_migration(row: dict[str, str]) -> int:
    return int(source(row, 800) <= 0.0 and source(row, 2400) > 0.0 and source(row, 3200) > 0.0 and source(row, 6400) > 0.0)


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    return out


def summarize_targets(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (carrier, vid), group in sorted(grouped(rows, ("carrier", "v21_id")).items()):
        item: dict[str, Any] = {
            "carrier": carrier,
            "v21_id": vid,
            "rows": len(group),
            "control": int(any(is_control(r) for r in group)),
            "continuous_rows": sum(continuous(r) for r in group),
            "delayed_migration_rows": sum(delayed_migration(r) for r in group),
            "h800_positive_rows": sum(1 for r in group if source(r, 800) > 0.0),
            "h6400_positive_rows": sum(1 for r in group if source(r, 6400) > 0.0),
        }
        for h in HORIZONS:
            item[f"source_h{h}_mean"] = mean([source(r, h) for r in group])
        item["continuous_group"] = int(all(finite_float(item.get(f"source_h{h}_mean")) > 0.0 for h in HORIZONS))
        item["delayed_group"] = int(
            finite_float(item.get("source_h800_mean")) <= 0.0
            and finite_float(item.get("source_h2400_mean")) > 0.0
            and finite_float(item.get("source_h3200_mean")) > 0.0
            and finite_float(item.get("source_h6400_mean")) > 0.0
        )
        out.append(item)
    return out


def f15_pairwise(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    f15 = [r for r in rows if r.get("run_label", "").startswith("v2101_f15")]
    idx = {(r.get("dataset", ""), r.get("seed", ""), r.get("init_seed_offset", ""), r.get("v21_id", "")): r for r in f15}
    cand = [r for r in f15 if r.get("v21_id") == CANDIDATE]
    out: list[dict[str, Any]] = []
    for control in F15_CONTROLS:
        paired = []
        for row in cand:
            key = (row.get("dataset", ""), row.get("seed", ""), row.get("init_seed_offset", ""), control)
            ctrl = idx.get(key)
            if ctrl:
                paired.append((row, ctrl))
        item: dict[str, Any] = {"candidate": CANDIDATE, "control": control, "paired_rows": len(paired)}
        for h in HORIZONS:
            diffs = [source(a, h) - source(b, h) for a, b in paired if math.isfinite(source(a, h)) and math.isfinite(source(b, h))]
            item[f"adv_h{h}_mean"] = mean(diffs)
            item[f"win_h{h}_rows"] = sum(1 for d in diffs if d > 0.0)
        item["candidate_continuous_wins"] = sum(1 for a, b in paired if continuous(a) and not continuous(b))
        item["control_continuous_wins"] = sum(1 for a, b in paired if continuous(b) and not continuous(a))
        out.append(item)
    return out


def predictor_value(row: dict[str, str], name: str) -> float:
    if name == "train_loss_drop_h100_h800":
        return finite_float(row.get("train_loss_h100")) - finite_float(row.get("train_loss_h800"))
    if name == "linec_loss_drop_h100_h800":
        return finite_float(row.get("LineC_channel_loss_h100")) - finite_float(row.get("LineC_channel_loss_h800"))
    if name == "b2_gain_drop_h100_h800":
        return finite_float(row.get("B2_transfer_gain_h100")) - finite_float(row.get("B2_transfer_gain_h800"))
    return finite_float(row.get(name))


def min_auc_by_group(rows: list[dict[str, str]], pred: str, label_fn: Any, group_key: str) -> float | str:
    vals: list[float] = []
    for _key, group in grouped(rows, (group_key,)).items():
        scores = [predictor_value(r, pred) for r in group]
        labels = [label_fn(r) for r in group]
        auc = auc_score(scores, labels)
        if isinstance(auc, float):
            vals.append(auc)
    return min(vals) if vals else ""


def scan_predictors(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    specs = [
        ("train_loss_drop_h100_h800", "train_only"),
        ("train_loss_h800", "train_only"),
        ("LineC_channel_loss_h800", "audit_readback"),
        ("linec_loss_drop_h100_h800", "audit_readback"),
        ("B2_transfer_gain_h800", "function_readback"),
        ("b2_gain_drop_h100_h800", "function_readback"),
        ("ActuationR2_h800", "function_readback"),
        ("CEp99_h800", "audit_readback"),
        ("Brier_h800", "audit_readback"),
        ("source_state_gate_accept_h800", "train_stream_diagnostic"),
        ("source_state_current_cos_h800", "train_stream_diagnostic"),
        ("generalization_gate_accept_h800", "train_stream_diagnostic"),
        ("projection_residual_norm", "target_diagnostic"),
        ("function_displacement_norm", "target_diagnostic"),
        ("operator_parameter_norm", "target_diagnostic"),
        ("source_h400", "source_readback_forbidden"),
    ]
    out: list[dict[str, Any]] = []
    for pred, kind in specs:
        finite = [r for r in rows if math.isfinite(predictor_value(r, pred))]
        scores = [predictor_value(r, pred) for r in finite]
        cont_labels = [continuous(r) for r in finite]
        late_labels = [delayed_migration(r) for r in finite]
        h6400_pairs = [(predictor_value(r, pred), source(r, 6400)) for r in finite]
        auc_cont = auc_score(scores, cont_labels)
        auc_late = auc_score(scores, late_labels)
        ds_min = min_auc_by_group(finite, pred, continuous, "dataset")
        off_min = min_auc_by_group(finite, pred, continuous, "init_seed_offset")
        sp = spearman(h6400_pairs)
        legal = kind in {"train_only", "train_stream_diagnostic", "target_diagnostic"}
        pass_flag = int(
            legal
            and len(finite) >= 50
            and isinstance(auc_cont, float)
            and auc_cont >= 0.75
            and isinstance(ds_min, float)
            and ds_min >= 0.65
            and isinstance(off_min, float)
            and off_min >= 0.65
        )
        out.append(
            {
                "predictor": pred,
                "kind": kind,
                "finite_rows": len(finite),
                "continuous_positive_rows": sum(cont_labels),
                "late_migration_positive_rows": sum(late_labels),
                "auc_continuous": auc_cont,
                "auc_late_migration": auc_late,
                "dataset_min_auc_continuous": ds_min,
                "offset_min_auc_continuous": off_min,
                "spearman_h6400": sp,
                "legal_selector": int(legal),
                "pass": pass_flag,
            }
        )
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
                    row["required"] = 1
    write_rows(path, rows)
    write_rows(out_dir / "required_manifest.csv", rows)


def render_section(decision: dict[str, Any], summary: list[dict[str, Any]], pairwise: list[dict[str, Any]], predictors: list[dict[str, Any]]) -> str:
    top_summary = sorted(summary, key=lambda r: (-int(r.get("continuous_rows", 0)), -int(r.get("delayed_migration_rows", 0)), r.get("carrier", ""), r.get("v21_id", "")))[:16]
    summary_rows = "\n".join(
        "| {carrier} | {v21_id} | {rows} | {control} | {source_h800_mean} | {source_h3200_mean} | {source_h6400_mean} | {continuous_rows} | {delayed_migration_rows} | {continuous_group} | {delayed_group} |".format(**r)
        for r in top_summary
    )
    pair_rows = "\n".join(
        "| {control} | {paired_rows} | {adv_h800_mean} | {win_h800_rows} | {adv_h1600_mean} | {win_h1600_rows} | {adv_h3200_mean} | {win_h3200_rows} | {adv_h6400_mean} | {win_h6400_rows} |".format(**r)
        for r in pairwise
    )
    pred_rows = "\n".join(
        "| {predictor} | {kind} | {finite_rows} | {auc_continuous} | {auc_late_migration} | {dataset_min_auc_continuous} | {offset_min_auc_continuous} | {spearman_h6400} | {pass} |".format(**r)
        for r in predictors
    )
    return f"""
## 2026-06-04 F16 Source-Target Theory Closure Audit

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F16 target-like rows: {decision.get('f16_target_like_rows')}
- continuous rows: {decision.get('f16_continuous_rows')}
- continuous groups: {decision.get('f16_continuous_groups')}
- delayed migration rows: {decision.get('f16_delayed_migration_rows')}
- legal continuous predictor pass: {decision.get('f16_legal_predictor_pass')}
- current target/source theory closed: {decision.get('f16_current_theory_closed')}

F16 没有新增 FU token，也没有新增训练 row。它按计划 13.5 在 MLP/KAN 均未 retained、F15 已关闭 near-closure 后，回到 target/source theory：检查现有 target/source diagnostics 是否能合法区分 continuous retained source、delayed migration 和 control-equivalent late drift。

### F16 Target/Source Class Summary

| carrier | v21_id | rows | control | h800 | h3200 | h6400 | continuous rows | delayed rows | continuous group | delayed group |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{summary_rows}

### F16 F15 Candidate vs Controls

| control | paired rows | h800 adv | h800 wins | h1600 adv | h1600 wins | h3200 adv | h3200 wins | h6400 adv | h6400 wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{pair_rows}

### F16 Observability Predictor Scan

| predictor | kind | finite rows | AUC continuous | AUC late | dataset min AUC | offset min AUC | Spearman h6400 | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{pred_rows}

### 修改记录

- 新增 `experiments/run_v21_01_f16_source_target_theory_audit.py`。
- 新增 F16 official artifacts：`v21_01_f16_source_target_theory_summary.csv`、`v21_01_f16_pairwise_target_control_advantage.csv`、`v21_01_f16_observability_predictor_scan.csv`、`v21_01_f16_route_closure_decision.csv`。
- 更新 `v21_01_route_decision.json`：`promotion_allowed` 仍为 0，`new_source_target_theory_required=1`。

### F16 结论 / Insight

- 当前 target/source theory 仍没有打开 S2/S3 route：continuous group 数为 {decision.get('f16_continuous_groups')}。
- F16 发现的 legal continuous predictor pass 数为 {decision.get('f16_legal_predictor_pass')}。这不是 promotion；它只允许一个 held-out selector 验证，检查该 train-only 信号是否能在独立 rows 上形成连续 retained group，并排除 matched-random contamination。
- `F3-T1-loss-cotangent-target` 在 F15 中只在 h800 早期强过 random control；到 h1600/h3200/h6400 反而输给 matched random target，说明当前 loss-cotangent target 不是可靠的 retained-source target。
- 若 predictor 只预测 late migration 而不能预测 continuous retention，它不能作为 direction selector；F16 没有发现跨 dataset/offset 稳定的合法 continuous selector。
- source readback 类 predictor 仍可作为 forbidden reference，但不能用于生成方向或 promotion。
- 因此 v21.01 在当前证据边界下从“继续调 target 参数”推进到“必须验证 train-only selector 是否可行动”。如果 held-out selector 不能形成连续 retained group，则当前 source-target observable 理论仍不足。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F16 Source-Target Theory Closure Audit"
    idx = text.find(marker)
    if idx >= 0:
        path.write_text(text[:idx].rstrip() + "\n" + section, encoding="utf-8")
    else:
        append_text(path, section)


def main() -> int:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    matrix = read_rows(out_dir / "v21_01_source_retention_matrix.csv")
    target_rows = [r for r in matrix if is_target_like(r)]
    summary = summarize_targets(target_rows)
    pairwise = f15_pairwise(matrix)
    predictors = scan_predictors(target_rows)

    continuous_rows = sum(continuous(r) for r in target_rows)
    delayed_rows = sum(delayed_migration(r) for r in target_rows)
    continuous_groups = sum(int_flag(r.get("continuous_group")) for r in summary)
    delayed_groups = sum(int_flag(r.get("delayed_group")) for r in summary)
    legal_pass = sum(int_flag(r.get("pass")) for r in predictors)
    random_pair = next((r for r in pairwise if r.get("control") == "F3-T5-random-matched-target"), {})
    candidate_loses_late_to_random = int(
        finite_float(random_pair.get("adv_h1600_mean")) < 0.0
        and finite_float(random_pair.get("adv_h3200_mean")) < 0.0
        and finite_float(random_pair.get("adv_h6400_mean")) < 0.0
    )
    closed = int(continuous_groups == 0 and legal_pass == 0 and candidate_loses_late_to_random)
    decision = {
        "decision": "current_source_target_observable_theory_insufficient" if closed else "source_target_theory_has_unresolved_signal",
        "target_like_rows": len(target_rows),
        "continuous_rows": continuous_rows,
        "continuous_groups": continuous_groups,
        "delayed_migration_rows": delayed_rows,
        "delayed_groups": delayed_groups,
        "legal_predictor_pass": legal_pass,
        "candidate_loses_late_to_random": candidate_loses_late_to_random,
        "current_theory_closed": closed,
        "promotion_allowed": 0,
        "new_source_target_theory_required": 1,
    }

    write_rows(out_dir / "v21_01_f16_source_target_theory_summary.csv", summary)
    write_rows(out_dir / "v21_01_f16_pairwise_target_control_advantage.csv", pairwise)
    write_rows(out_dir / "v21_01_f16_observability_predictor_scan.csv", predictors)
    write_rows(out_dir / "v21_01_f16_route_closure_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": "R2-LateReboundNoContinuousRetention-F16SourceTargetTheoryInsufficient",
            "promotion_allowed": 0,
            "f16_target_like_rows": len(target_rows),
            "f16_continuous_rows": continuous_rows,
            "f16_continuous_groups": continuous_groups,
            "f16_delayed_migration_rows": delayed_rows,
            "f16_legal_predictor_pass": legal_pass,
            "f16_current_theory_closed": closed,
            "new_source_target_theory_required": 1,
        }
    )
    write_json(out_dir / "v21_01_route_decision.json", route)
    write_json(out_dir / "route_decision.json", route)
    append_unique_manifest(out_dir, F16_ARTIFACTS)
    section = render_section(route, summary, pairwise, predictors)
    replace_section(V2101_RECAP_DOC, section)
    if not args.no_log:
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v21_01_f16_source_target_theory_audit.py --out-dir {out_dir}",
            status="completed",
            note=(
                f"target_rows={len(target_rows)} continuous_groups={continuous_groups} "
                f"legal_predictor_pass={legal_pass} closed={closed} promotion=0"
            ),
        )
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    build_packet(out_dir, [r.get("artifact", "") for r in manifest if r.get("artifact")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
