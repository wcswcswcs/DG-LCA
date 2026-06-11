#!/usr/bin/env python3
"""Summarize v21.01 F15 loss-target independent confirmation."""

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


F15_ARTIFACTS = [
    "v21_01_f15_loss_target_independent_aggregate.csv",
    "v21_01_f15_loss_target_independent_per_offset.csv",
    "v21_01_f15_loss_target_independent_row_examples.csv",
    "v21_01_f15_loss_target_independent_decision.csv",
]
HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
CANDIDATE = "F3-T1-loss-cotangent-target"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-log", action="store_true")
    return p


def source(row: dict[str, str], horizon: int) -> float:
    return finite_float(row.get(f"source_h{horizon}"))


def mean(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else ""


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    return out


def summarize_group(key: tuple[str, ...], rows: list[dict[str, str]], key_names: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {name: value for name, value in zip(key_names, key)}
    out["rows"] = len(rows)
    out["blocked_rows"] = sum(1 for r in rows if str(r.get("execution_status", "")).startswith("blocked"))
    for h in HORIZONS:
        xs = [source(r, h) for r in rows]
        out[f"source_h{h}_mean"] = mean(xs)
        out[f"source_h{h}_positive_rows"] = sum(1 for x in xs if math.isfinite(x) and x > 0.0)
    out["retained_h6400_rows"] = sum(int_flag(r.get("retained_h6400_flag")) for r in rows)
    out["late_rebound_rows"] = sum(int_flag(r.get("late_rebound_flag")) for r in rows)
    hvals = [finite_float(out.get(f"source_h{h}_mean")) for h in HORIZONS]
    out["continuous_group"] = int(all(v > 0.0 for v in hvals))
    out["productive_group"] = int(
        all(v >= 0.005 for v in hvals)
        and finite_float(out.get("source_h3200_mean")) >= 0.005
        and finite_float(out.get("source_h6400_mean")) >= 0.005
    )
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


def render_section(decision: dict[str, Any], aggregate: list[dict[str, Any]], per_offset: list[dict[str, Any]]) -> str:
    agg_rows = "\n".join(
        "| {v21_id} | {rows} | {source_h800_mean} | {source_h1600_mean} | {source_h2400_mean} | {source_h3200_mean} | {source_h4800_mean} | {source_h6400_mean} | {source_h800_positive_rows} | {retained_h6400_rows} | {late_rebound_rows} | {continuous_group} |".format(
            **r
        )
        for r in aggregate
    )
    per_rows = "\n".join(
        "| {init_seed_offset} | {v21_id} | {rows} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h6400_mean} | {retained_h6400_rows} | {late_rebound_rows} |".format(
            **r
        )
        for r in per_offset
    )
    return f"""
## 2026-06-04 F15 Loss-Target Near-Closure Independent Confirmation

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F15 rows: {decision.get('f15_rows')}
- blocked rows: {decision.get('f15_blocked_rows')}
- candidate: `{decision.get('f15_candidate')}`
- candidate h800 mean: {decision.get('f15_candidate_h800_mean')}
- candidate continuous group: {decision.get('f15_candidate_continuous_group')}
- matched random h6400 mean: {decision.get('f15_random_h6400_mean')}
- stable-random h6400 positive rows: {decision.get('f15_stable_random_h6400_positive_rows')}

F15 是 F14 推荐的唯一 near-early-closure 候选验证：`F3-T1-loss-cotangent-target` 在 offset0 h800 gap 只有 0.0156，因此按计划 13.6 做 x3 independent offsets，并加入 matched random / sign-flip / corrupted-label / stable-random controls。

### F15 Aggregate

| v21_id | rows | h800 | h1600 | h2400 | h3200 | h4800 | h6400 | h800+ rows | retained rows | late rebound rows | continuous group |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{agg_rows}

### F15 Per-Offset

| offset | v21_id | rows | h800 | h1600 | h3200 | h6400 | retained rows | late rebound rows |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
{per_rows}

### 修改记录

- 新增 `experiments/run_v21_01_f15_loss_target_independent_summary.py`。
- 新增 F15 official artifacts：`v21_01_f15_loss_target_independent_aggregate.csv`、`v21_01_f15_loss_target_independent_per_offset.csv`、`v21_01_f15_loss_target_independent_row_examples.csv`、`v21_01_f15_loss_target_independent_decision.csv`。
- F15 训练命令已写入执行日志；本节只汇总真实落盘 rows。
- 更新 `v21_01_route_decision.json`，`promotion_allowed` 仍为 0。

### F15 结论 / Insight

- F14 的 near-closure 候选没有独立转正：`F3-T1-loss-cotangent-target` independent h800 mean = {decision.get('f15_candidate_h800_mean')}，h1600 mean = {decision.get('f15_candidate_h1600_mean')}，不是连续 retained chain。
- 它仍然有 delayed migration：h2400/h3200/h4800/h6400 grouped mean 转正，但前两个 horizon 没闭合。
- matched random target 也出现 h1600-h6400 正值，且 retained rows 与 candidate 接近，说明 loss target 的 late positive 不是足够强的 attribution。
- stable-random control 仍有 h6400 positive rows，继续支持 F13 的 control-equivalent late drift 判定。
- 因此 F14 提出的唯一可行动 near-closure 线索已被 F15 关闭。当前 v21.01 不能继续同族 target/source repair；需要新的 source-target theory，而不是继续调现有 target 族。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F15 Loss-Target Near-Closure Independent Confirmation"
    idx = text.find(marker)
    if idx >= 0:
        path.write_text(text[:idx].rstrip() + "\n" + section, encoding="utf-8")
    else:
        append_text(path, section)


def main() -> int:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    rows = [
        r
        for r in read_rows(out_dir / "v21_01_source_retention_matrix.csv")
        if r.get("run_label", "").startswith("v2101_f15_loss_target_independent")
    ]
    if not rows:
        raise SystemExit("missing F15 rows")
    aggregate = [summarize_group(k, v, ("v21_id",)) for k, v in sorted(grouped(rows, ("v21_id",)).items())]
    per_offset = [
        summarize_group(k, v, ("init_seed_offset", "v21_id"))
        for k, v in sorted(grouped(rows, ("init_seed_offset", "v21_id")).items())
    ]
    examples = [
        {
            "v21_id": r.get("v21_id", ""),
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "init_seed_offset": r.get("init_seed_offset", ""),
            "source_h800": r.get("source_h800", ""),
            "source_h1600": r.get("source_h1600", ""),
            "source_h3200": r.get("source_h3200", ""),
            "source_h6400": r.get("source_h6400", ""),
            "retained_h6400_flag": r.get("retained_h6400_flag", ""),
            "late_rebound_flag": r.get("late_rebound_flag", ""),
        }
        for r in rows
        if int_flag(r.get("retained_h6400_flag")) or int_flag(r.get("late_rebound_flag"))
    ][:40]
    candidate = next((r for r in aggregate if r.get("v21_id") == CANDIDATE), {})
    random = next((r for r in aggregate if r.get("v21_id") == "F3-T5-random-matched-target"), {})
    stable = next((r for r in aggregate if r.get("v21_id") == "F9-TCTRL-stable-random-target"), {})
    continuous_groups = sum(int_flag(r.get("continuous_group")) for r in aggregate)
    blocked = sum(int(str(r.get("execution_status", "")).startswith("blocked")) for r in rows)
    decision = {
        "decision": "loss_target_near_closure_failed_independent_control_equivalent_late_drift",
        "rows": len(rows),
        "blocked_rows": blocked,
        "candidate": CANDIDATE,
        "candidate_h800_mean": candidate.get("source_h800_mean", ""),
        "candidate_h1600_mean": candidate.get("source_h1600_mean", ""),
        "candidate_h3200_mean": candidate.get("source_h3200_mean", ""),
        "candidate_h6400_mean": candidate.get("source_h6400_mean", ""),
        "candidate_continuous_group": candidate.get("continuous_group", ""),
        "continuous_groups": continuous_groups,
        "random_h6400_mean": random.get("source_h6400_mean", ""),
        "stable_random_h6400_positive_rows": stable.get("source_h6400_positive_rows", ""),
        "promotion_allowed": 0,
        "new_source_target_theory_required": 1,
    }
    write_rows(out_dir / "v21_01_f15_loss_target_independent_aggregate.csv", aggregate)
    write_rows(out_dir / "v21_01_f15_loss_target_independent_per_offset.csv", per_offset)
    write_rows(out_dir / "v21_01_f15_loss_target_independent_row_examples.csv", examples)
    write_rows(out_dir / "v21_01_f15_loss_target_independent_decision.csv", [decision])
    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": "R2-LateReboundNoContinuousRetention-F15LossTargetNoEarlyClosure",
            "promotion_allowed": 0,
            "f15_independent_completed": 1,
            "f15_rows": len(rows),
            "f15_blocked_rows": blocked,
            "f15_candidate": CANDIDATE,
            "f15_candidate_h800_mean": candidate.get("source_h800_mean", ""),
            "f15_candidate_h1600_mean": candidate.get("source_h1600_mean", ""),
            "f15_candidate_continuous_group": candidate.get("continuous_group", ""),
            "f15_random_h6400_mean": random.get("source_h6400_mean", ""),
            "f15_stable_random_h6400_positive_rows": stable.get("source_h6400_positive_rows", ""),
            "new_source_target_theory_required": 1,
        }
    )
    write_json(out_dir / "v21_01_route_decision.json", route)
    write_json(out_dir / "route_decision.json", route)
    append_unique_manifest(out_dir, F15_ARTIFACTS)
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    build_packet(out_dir, [r.get("artifact", "") for r in manifest if r.get("artifact")])
    section = render_section(route, aggregate, per_offset)
    replace_section(V2101_RECAP_DOC, section)
    if not args.no_log:
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v21_01_f15_loss_target_independent_summary.py --out-dir {out_dir}",
            status="completed",
            note=(
                f"rows={len(rows)} blocked={blocked} candidate_h800={candidate.get('source_h800_mean','')} "
                f"continuous_groups={continuous_groups} promotion=0"
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
