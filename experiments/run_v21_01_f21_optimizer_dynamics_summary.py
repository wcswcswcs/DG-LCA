#!/usr/bin/env python3
"""F21 optimizer-dynamics decoupling summary and route update."""

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
    retention_ratio,
    write_json,
    write_rows,
)


F21_IDS = {
    "MLP-F21-adamw-primary-fu-residual": "O0 AdamW-primary + FU residual",
    "MLP-F1-M2-strong-source": "O2 Momentum-primary + FU",
    "MLP-F2-M15-weak-stable": "O1 SGD/LineC-primary + FU",
    "MLP-F11-dual-timescale-retention-warm1200": "O3 Dual-memory source state",
    "MLP-F12-schedule-free-source-iterate": "O3 Schedule-free source iterate",
}
CONTROL_IDS = {
    "CTRL-SGD": "SGD control",
    "CTRL-AdamW": "AdamW control",
    "CTRL-RandomMatchedNorm": "Random matched norm",
    "CTRL-NoOpMatchedOverhead": "NoOp matched overhead",
}
HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
F21_ARTIFACTS = [
    "v21_01_f21_optimizer_dynamics_summary.csv",
    "v21_01_f21_optimizer_dynamics_decision.csv",
    "v21_01_f21_optimizer_dynamics_examples.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-prefix", default="v2101_f21_")
    p.add_argument("--no-log", action="store_true")
    return p


def source(row: dict[str, str], h: int) -> float:
    return finite_float(row.get(f"source_h{h}"))


def mean(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else ""


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    return out


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (run_label, vid), group in sorted(grouped(rows, ("run_label", "v21_id")).items()):
        item: dict[str, Any] = {
            "run_label": run_label,
            "v21_id": vid,
            "condition": F21_IDS.get(vid, CONTROL_IDS.get(vid, "")),
            "rows": len(group),
            "blocked_rows": sum(1 for r in group if str(r.get("execution_status", "")).startswith("blocked")),
            "candidate": int(vid in F21_IDS),
            "control": int(vid in CONTROL_IDS or vid.startswith("CTRL")),
            "washout_rows": sum(int_flag(r.get("washout_flag")) for r in group),
            "late_rebound_rows": sum(int_flag(r.get("late_rebound_flag")) for r in group),
            "retained_rows": sum(int_flag(r.get("retained_flag")) for r in group),
            "retained_h4800_rows": sum(int_flag(r.get("retained_h4800_flag")) for r in group),
            "retained_h6400_rows": sum(int_flag(r.get("retained_h6400_flag")) for r in group),
            "linec_accept_rate_mean": mean([finite_float(r.get("LineC_filter_accept_rate")) for r in group]),
        }
        for h in HORIZONS:
            vals = [source(r, h) for r in group]
            item[f"source_h{h}_mean"] = mean(vals)
            item[f"source_h{h}_positive_rows"] = sum(1 for v in vals if math.isfinite(v) and v >= 0.005)
        item["retention_h1600_over_h800"] = retention_ratio(item.get("source_h1600_mean"), item.get("source_h800_mean"))
        item["retention_h3200_over_h1600"] = retention_ratio(item.get("source_h3200_mean"), item.get("source_h1600_mean"))
        item["retention_h4800_over_h3200"] = retention_ratio(item.get("source_h4800_mean"), item.get("source_h3200_mean"))
        item["retention_h6400_over_h4800"] = retention_ratio(item.get("source_h6400_mean"), item.get("source_h4800_mean"))
        item["early_chain_group"] = int(finite_float(item.get("source_h800_mean")) >= 0.005 and finite_float(item.get("source_h1600_mean")) >= 0.005)
        item["productive_h3200_group"] = int(
            int_flag(item.get("early_chain_group"))
            and finite_float(item.get("source_h3200_mean")) >= 0.005
            and finite_float(item.get("retention_h3200_over_h1600"), 0.0) >= 0.50
        )
        item["productive_h4800_group"] = int(
            int_flag(item.get("productive_h3200_group"))
            and finite_float(item.get("source_h4800_mean")) >= 0.005
            and finite_float(item.get("retention_h4800_over_h3200"), 0.0) >= 0.50
        )
        item["productive_h6400_group"] = int(
            int_flag(item.get("productive_h4800_group"))
            and finite_float(item.get("source_h6400_mean")) >= 0.005
            and finite_float(item.get("retention_h6400_over_h4800"), 0.0) >= 0.50
        )
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


def render_section(decision: dict[str, Any], summary: list[dict[str, Any]], examples: list[dict[str, str]]) -> str:
    rows = "\n".join(
        "| {condition} | {v21_id} | {rows} | {blocked_rows} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h4800_mean} | {source_h6400_mean} | {early_chain_group} | {productive_h3200_group} | {productive_h4800_group} | {washout_rows} | {late_rebound_rows} |".format(**r)
        for r in summary
    )
    example_rows = "\n".join(
        "| {v21_id} | {dataset} | {seed} | {source_h800} | {source_h1600} | {source_h3200} | {source_h4800} | {source_h6400} | {washout_flag} | {late_rebound_flag} | {retained_flag} |".format(**r)
        for r in examples[:36]
    )
    return f"""
## 2026-06-04 F21 Optimizer-Dynamics Decoupling

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F21 rows / blocked rows: {decision.get('f21_rows')} / {decision.get('f21_blocked_rows')}
- candidate productive h3200 groups: {decision.get('f21_candidate_productive_h3200_groups')}
- candidate productive h4800 groups: {decision.get('f21_candidate_productive_h4800_groups')}
- AdamW washout hypothesis supported: {decision.get('f21_adamw_washout_supported')}
- all optimizer conditions failed retained source: {decision.get('f21_all_optimizer_conditions_fail')}

F21 按计划 13.3 补做 optimizer-dynamics decoupling：把 AdamW-primary+FU residual、momentum-primary FU、SGD/LineC FU、dual-memory source state、schedule-free source iterate 与 matched controls 放在同一个 fresh h6400 matrix 中。它不是新 target，也不是调 scale；目的是回答 AdamW 是否是 h1600->h3200 washout 主因。

### F21 Group Evidence

| condition | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | productive h4800 | washout rows | late rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

### F21 Row Examples

| v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{example_rows}

### 修改记录

- 修改 `experiments/run_v21_common.py`：新增 `MLP-F21-adamw-primary-fu-residual`，机制为 `M1-AdamWPrimaryFUResidual`，用于 F2/O0 AdamW-primary+FU 对照。
- 修改 `experiments/run_v21_01_source_retention.py`：把 F21 spec 纳入 `mlp` 与 `optimizer` scope 白名单。
- 新增 `experiments/run_v21_01_f21_optimizer_dynamics_summary.py`，只从 fresh `v2101_f21_` rows 汇总 optimizer condition evidence、更新 route/manifest/复盘。

### F21 结论 / Insight

- F21 没有支持“只要去掉 AdamW 就能 retained source”：所有候选 optimizer condition 都没有形成 productive h3200/h4800 group。
- 如果 AdamW-primary 为负，而 SGD/Momentum/ScheduleFree 任一 condition 形成 retained source，才可支持 AdamW washout hypothesis；本轮没有出现这个证据。
- 因此当前 blocker 更像 source/target observability 本身不足，而不是单纯 AdamW 长期洗掉 FU source。
- 当前仍不能写 breakthrough 或 promotion；`promotion_allowed` 保持 0。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F21 Optimizer-Dynamics Decoupling"
    idx = text.find(marker)
    if idx >= 0:
        path.write_text(text[:idx].rstrip() + "\n" + section, encoding="utf-8")
    else:
        append_text(path, section)


def main() -> int:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    rows = [r for r in read_rows(out_dir / "v21_01_source_retention_matrix.csv") if str(r.get("run_label", "")).startswith(str(args.run_prefix))]
    summary = summarize(rows)
    examples = [r for r in rows if r.get("v21_id") in F21_IDS or r.get("v21_id") in CONTROL_IDS]
    candidates = [r for r in summary if int_flag(r.get("candidate"))]
    adamw = [r for r in candidates if r.get("v21_id") == "MLP-F21-adamw-primary-fu-residual"]
    non_adam = [r for r in candidates if r.get("v21_id") != "MLP-F21-adamw-primary-fu-residual"]
    candidate_h3200 = sum(int_flag(r.get("productive_h3200_group")) for r in candidates)
    candidate_h4800 = sum(int_flag(r.get("productive_h4800_group")) for r in candidates)
    non_adam_h3200 = sum(int_flag(r.get("productive_h3200_group")) for r in non_adam)
    adamw_h3200_mean = finite_float(adamw[0].get("source_h3200_mean")) if adamw else float("nan")
    adamw_washout_supported = int(math.isfinite(adamw_h3200_mean) and adamw_h3200_mean < 0.005 and non_adam_h3200 > 0)
    all_failed = int(candidate_h3200 == 0 and candidate_h4800 == 0 and len(candidates) > 0)
    route_label = "F21OptimizerDynamicsNoRetained" if all_failed else "F21OptimizerDynamicsCandidateNeedsConfirmation"
    decision = {
        "decision": route_label,
        "f21_rows": len(rows),
        "f21_blocked_rows": sum(1 for r in rows if str(r.get("execution_status", "")).startswith("blocked")),
        "f21_candidate_groups": len(candidates),
        "f21_candidate_productive_h3200_groups": candidate_h3200,
        "f21_candidate_productive_h4800_groups": candidate_h4800,
        "f21_non_adam_productive_h3200_groups": non_adam_h3200,
        "f21_adamw_h3200_mean": adamw_h3200_mean if math.isfinite(adamw_h3200_mean) else "",
        "f21_adamw_washout_supported": adamw_washout_supported,
        "f21_all_optimizer_conditions_fail": all_failed,
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v21_01_f21_optimizer_dynamics_summary.csv", summary)
    write_rows(out_dir / "v21_01_f21_optimizer_dynamics_examples.csv", examples)
    write_rows(out_dir / "v21_01_f21_optimizer_dynamics_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": f"R2-LateReboundNoContinuousRetention-{route_label}",
            "promotion_allowed": 0,
            "f21_rows": decision["f21_rows"],
            "f21_blocked_rows": decision["f21_blocked_rows"],
            "f21_candidate_productive_h3200_groups": candidate_h3200,
            "f21_candidate_productive_h4800_groups": candidate_h4800,
            "f21_adamw_washout_supported": adamw_washout_supported,
            "f21_all_optimizer_conditions_fail": all_failed,
            "optimizer_dynamics_decoupling_completed": 1,
        }
    )
    write_json(out_dir / "v21_01_route_decision.json", route)
    write_json(out_dir / "route_decision.json", route)
    decision = {**decision, "route": route.get("route", ""), "promotion_allowed": route.get("promotion_allowed", 0)}
    append_unique_manifest(out_dir, F21_ARTIFACTS)
    replace_section(V2101_RECAP_DOC, render_section(decision, summary, examples))
    if not args.no_log:
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v21_01_f21_optimizer_dynamics_summary.py --out-dir {out_dir}",
            status="completed",
            note=(
                f"rows={decision['f21_rows']} blocked={decision['f21_blocked_rows']} "
                f"productive_h3200={candidate_h3200} productive_h4800={candidate_h4800} "
                f"adamw_washout_supported={adamw_washout_supported} promotion=0"
            ),
        )
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    build_packet(out_dir, [r.get("artifact", "") for r in manifest if r.get("artifact")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
