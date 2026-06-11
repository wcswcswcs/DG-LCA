#!/usr/bin/env python3
"""F20 margin split-consensus source-channel summary and route update."""

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


F20_IDS = {"MLP-F20-margin-split-consensus-source", "KSW4-margin-split-consensus-source"}
CONTROL_IDS = {
    "CTRL-SGD",
    "CTRL-AdamW",
    "CTRL-RandomMatchedNorm",
    "CTRL-NoOpMatchedOverhead",
}
HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
F20_ARTIFACTS = [
    "v21_01_f20_margin_source_summary.csv",
    "v21_01_f20_margin_source_decision.csv",
    "v21_01_f20_margin_source_examples.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-prefix", default="v2101_f20_")
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


def continuous_row(row: dict[str, str]) -> int:
    return int(
        source(row, 800) >= 0.005
        and source(row, 1600) >= 0.005
        and source(row, 3200) >= 0.005
        and finite_float(row.get("retention_h3200_over_h1600"), 0.0) >= 0.50
    )


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (run_label, carrier, vid), group in sorted(grouped(rows, ("run_label", "carrier", "v21_id")).items()):
        gate_fields = [f"source_state_gate_accept_h{h}" for h in HORIZONS]
        item: dict[str, Any] = {
            "run_label": run_label,
            "carrier": carrier,
            "v21_id": vid,
            "rows": len(group),
            "blocked_rows": sum(1 for r in group if str(r.get("execution_status", "")).startswith("blocked")),
            "candidate": int(vid in F20_IDS),
            "control": int(vid in CONTROL_IDS or vid.startswith("CTRL")),
            "continuous_rows": sum(continuous_row(r) for r in group),
            "late_rebound_rows": sum(int_flag(r.get("late_rebound_flag")) for r in group),
            "retained_rows": sum(int_flag(r.get("retained_flag")) for r in group),
            "retained_h6400_rows": sum(int_flag(r.get("retained_h6400_flag")) for r in group),
            "gate_accept_trace_sum": sum(int_flag(r.get(field)) for r in group for field in gate_fields),
            "gate_accept_final_rows": sum(int_flag(r.get("source_state_gate_accept")) for r in group),
            "consensus_density_mean": mean([finite_float(r.get("source_state_consensus_density")) for r in group]),
            "balance_mean": mean([finite_float(r.get("source_state_balance_mean")) for r in group]),
            "signal_gain_mean": mean([finite_float(r.get("source_state_signal_gain")) for r in group]),
            "corrupt_gain_mean": mean([finite_float(r.get("source_state_corrupt_gain")) for r in group]),
            "corrupt_cos_mean": mean([finite_float(r.get("source_state_corrupt_cos")) for r in group]),
        }
        for h in HORIZONS:
            vals = [source(r, h) for r in group]
            item[f"source_h{h}_mean"] = mean(vals)
            item[f"source_h{h}_positive_rows"] = sum(1 for v in vals if math.isfinite(v) and v >= 0.005)
            item[f"gate_accept_h{h}_rows"] = sum(int_flag(r.get(f"source_state_gate_accept_h{h}")) for r in group)
        item["retention_h1600_over_h800"] = retention_ratio(item.get("source_h1600_mean"), item.get("source_h800_mean"))
        item["retention_h3200_over_h1600"] = retention_ratio(item.get("source_h3200_mean"), item.get("source_h1600_mean"))
        item["early_chain_group"] = int(finite_float(item.get("source_h800_mean")) >= 0.005 and finite_float(item.get("source_h1600_mean")) >= 0.005)
        item["productive_h3200_group"] = int(
            int_flag(item.get("early_chain_group"))
            and finite_float(item.get("source_h3200_mean")) >= 0.005
            and finite_float(item.get("retention_h3200_over_h1600"), 0.0) >= 0.50
        )
        item["productive_h6400_group"] = int(
            int_flag(item.get("productive_h3200_group"))
            and finite_float(item.get("source_h4800_mean")) >= 0.005
            and finite_float(item.get("source_h6400_mean")) >= 0.005
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
        "| {run_label} | {carrier} | {v21_id} | {rows} | {blocked_rows} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h6400_mean} | {early_chain_group} | {productive_h3200_group} | {gate_accept_trace_sum} | {consensus_density_mean} | {signal_gain_mean} | {corrupt_gain_mean} |".format(**r)
        for r in summary
    )
    example_rows = "\n".join(
        "| {run_label} | {carrier} | {v21_id} | {dataset} | {seed} | {init_seed_offset} | {source_h800} | {source_h1600} | {source_h3200} | {source_h6400} | {source_state_gate_accept_h800} | {source_state_gate_accept_h1600} | {source_state_current_cos_h1600} | {retained_flag} | {late_rebound_flag} |".format(**r)
        for r in examples[:24]
    )
    return f"""
## 2026-06-04 F20 Margin Split-Consensus Source-Channel Estimator

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F20 rows / blocked rows: {decision.get('f20_rows')} / {decision.get('f20_blocked_rows')}
- candidate early-chain groups: {decision.get('f20_candidate_early_chain_groups')}
- candidate productive h3200 groups: {decision.get('f20_candidate_productive_h3200_groups')}
- control early-chain groups: {decision.get('f20_control_early_chain_groups')}
- full escalation recommended: {decision.get('f20_full_recommended')}

F20 是 F18/F19 target 语义失败后的 source-channel estimator redesign：不再直接拟合 function-space target，而是用 train split A/B 的 top-wrong margin 梯度一致性形成 slow source state，并用 corrupted-label train batch 做 gate。它不使用 validation/test/future/query 生成方向，也不是继续调 actuation rank、alt period 或 target threshold。

### F20 Group Evidence

| run_label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h6400 | early chain | productive h3200 | gate accepts | consensus density | signal gain | corrupt gain |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

### F20 Row Examples

| run_label | carrier | v21_id | dataset | seed | offset | h800 | h1600 | h3200 | h6400 | gate h800 | gate h1600 | cos h1600 | retained | late |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{example_rows}

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M68-MarginSplitConsensusSlowFU` semantic contract 与 source 描述。
- 修改 `experiments/run_v17_common.py`：新增 M68 训练分支，用 top-wrong margin split-consensus 梯度构造 slow source state，并记录 consensus density、signal/corrupt gain、corrupt cosine 与 gate accept。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 `MLP-F20-margin-split-consensus-source` 与 `KSW4-margin-split-consensus-source` specs。
- 新增 `experiments/run_v21_01_f20_margin_source_summary.py`，只从落盘 matrix 汇总 F20 结果并更新 route/manifest/复盘。

### F20 结论 / Insight

- F20 不是 breakthrough：没有形成 candidate productive h3200 group。
- 如果 `full escalation recommended=0`，说明 source-channel estimator 在 smoke 阶段没有形成 h800/h1600 grouped early-chain，不能升级 full。
- 如果 smoke 有 early-chain 但 full 后 productive h3200 仍为 0，则说明 margin split-consensus 仍没有解决 source retention dynamics。
- 当前仍不能把 gate accepts、single-row positive 或 late rebound 写成 promotion；`promotion_allowed` 保持 0。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F20 Margin Split-Consensus Source-Channel Estimator"
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
    examples = [r for r in rows if r.get("v21_id") in F20_IDS or r.get("v21_id") in CONTROL_IDS]
    candidates = [r for r in summary if int_flag(r.get("candidate"))]
    controls = [r for r in summary if int_flag(r.get("control"))]
    candidate_early = sum(int_flag(r.get("early_chain_group")) for r in candidates)
    candidate_productive = sum(int_flag(r.get("productive_h3200_group")) for r in candidates)
    control_early = sum(int_flag(r.get("early_chain_group")) for r in controls)
    full_recommended = int(
        candidate_early > 0
        and candidate_productive == 0
        and all(not math.isfinite(finite_float(r.get("source_h3200_mean"))) for r in candidates)
    )
    route_label = "F20MarginSourceSmokeNoEarlySignal" if candidate_early == 0 else "F20MarginSourceNoContinuousRetention"
    decision = {
        "decision": route_label,
        "f20_rows": len(rows),
        "f20_blocked_rows": sum(1 for r in rows if str(r.get("execution_status", "")).startswith("blocked")),
        "f20_candidate_groups": len(candidates),
        "f20_candidate_early_chain_groups": candidate_early,
        "f20_candidate_productive_h3200_groups": candidate_productive,
        "f20_control_early_chain_groups": control_early,
        "f20_full_recommended": full_recommended,
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v21_01_f20_margin_source_summary.csv", summary)
    write_rows(out_dir / "v21_01_f20_margin_source_examples.csv", examples)
    write_rows(out_dir / "v21_01_f20_margin_source_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": f"R2-LateReboundNoContinuousRetention-{route_label}",
            "promotion_allowed": 0,
            "f20_rows": decision["f20_rows"],
            "f20_blocked_rows": decision["f20_blocked_rows"],
            "f20_candidate_early_chain_groups": candidate_early,
            "f20_candidate_productive_h3200_groups": candidate_productive,
            "f20_full_recommended": full_recommended,
            "new_source_channel_estimator_required": int(candidate_productive == 0),
        }
    )
    write_json(out_dir / "v21_01_route_decision.json", route)
    write_json(out_dir / "route_decision.json", route)
    decision = {**decision, "route": route.get("route", ""), "promotion_allowed": route.get("promotion_allowed", 0)}
    append_unique_manifest(out_dir, F20_ARTIFACTS)
    replace_section(V2101_RECAP_DOC, render_section(decision, summary, examples))
    if not args.no_log:
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v21_01_f20_margin_source_summary.py --out-dir {out_dir}",
            status="completed",
            note=(
                f"rows={decision['f20_rows']} blocked={decision['f20_blocked_rows']} "
                f"candidate_early={candidate_early} productive_h3200={candidate_productive} "
                f"full_recommended={full_recommended} promotion=0"
            ),
        )
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    build_packet(out_dir, [r.get("artifact", "") for r in manifest if r.get("artifact")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
