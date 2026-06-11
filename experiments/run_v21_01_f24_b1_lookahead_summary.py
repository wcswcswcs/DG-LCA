#!/usr/bin/env python3
"""F24 B1 consensus target-to-retention lookahead selector summary."""

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


CANDIDATE = "F24-B1-consensus-lookahead-gated-transfer"
REFERENCE_IDS = {
    "F10-T7-b1-cross-split-consensus-transfer": "B1 consensus transfer reference",
    "KSW2-lowdegree-lowfreq-source-bank": "KSW2 source-bank reference",
    "F3-T5-random-matched-target": "random matched target control",
}
CONTROL_IDS = {
    "CTRL-SGD": "SGD control",
    "CTRL-RandomMatchedNorm": "Random matched norm",
    "CTRL-NoOpMatchedOverhead": "NoOp matched overhead",
}
HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
ARTIFACTS = [
    "v21_01_f24_b1_lookahead_summary.csv",
    "v21_01_f24_b1_lookahead_decision.csv",
    "v21_01_f24_b1_lookahead_examples.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-prefix", default="v2101_f24_")
    p.add_argument("--no-log", action="store_true")
    return p


def source(row: dict[str, str], h: int) -> float:
    return finite_float(row.get(f"source_h{h}"))


def mean(vals: list[float]) -> float | str:
    data = [v for v in vals if math.isfinite(v)]
    return sum(data) / len(data) if data else ""


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    return out


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (run_label, carrier, vid), group in sorted(grouped(rows, ("run_label", "carrier", "v21_id")).items()):
        item: dict[str, Any] = {
            "run_label": run_label,
            "carrier": carrier,
            "v21_id": vid,
            "condition": "F24 B1 consensus lookahead gate" if vid == CANDIDATE else REFERENCE_IDS.get(vid, CONTROL_IDS.get(vid, "")),
            "rows": len(group),
            "blocked_rows": sum(1 for r in group if str(r.get("execution_status", "")).startswith("blocked")),
            "candidate": int(vid == CANDIDATE),
            "reference": int(vid in REFERENCE_IDS),
            "control": int(vid in CONTROL_IDS or vid.startswith("CTRL")),
            "washout_rows": sum(int_flag(r.get("washout_flag")) for r in group),
            "late_rebound_rows": sum(int_flag(r.get("late_rebound_flag")) for r in group),
            "retained_rows": sum(int_flag(r.get("retained_flag")) for r in group),
            "gate_accept_h800_rows": sum(int_flag(r.get("source_state_gate_accept_h800")) for r in group),
            "gate_accept_h1600_rows": sum(int_flag(r.get("source_state_gate_accept_h1600")) for r in group),
            "lookahead_advantage_h1600_mean": mean([finite_float(r.get("source_state_balance_mean_h1600")) for r in group]),
            "current_cos_h1600_mean": mean([finite_float(r.get("source_state_current_cos_h1600")) for r in group]),
        }
        for h in HORIZONS:
            values = [source(r, h) for r in group]
            item[f"source_h{h}_mean"] = mean(values)
            item[f"source_h{h}_positive_rows"] = sum(1 for v in values if math.isfinite(v) and v >= 0.005)
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
        "| {condition} | {run_label} | {carrier} | {v21_id} | {rows} | {blocked_rows} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h4800_mean} | {source_h6400_mean} | {early_chain_group} | {productive_h3200_group} | {gate_accept_h800_rows} | {gate_accept_h1600_rows} | {lookahead_advantage_h1600_mean} |".format(**r)
        for r in summary
    )
    example_rows = "\n".join(
        "| {run_label} | {carrier} | {v21_id} | {dataset} | {seed} | {source_h800} | {source_h1600} | {source_h3200} | {source_h4800} | {source_h6400} | {washout_flag} | {late_rebound_flag} | {retained_flag} |".format(**r)
        for r in examples[:48]
    )
    return f"""
## 2026-06-04 F24 B1 Consensus Lookahead-Gated Transfer

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F24 rows / blocked rows: {decision.get('f24_rows')} / {decision.get('f24_blocked_rows')}
- F24 smoke rows / full rows: {decision.get('f24_smoke_rows')} / {decision.get('f24_full_rows')}
- candidate early-chain groups: {decision.get('f24_candidate_early_chain_groups')}
- candidate productive h3200 groups: {decision.get('f24_candidate_productive_h3200_groups')}
- full escalation recommended: {decision.get('f24_full_escalation_recommended')}

F24 是 target-to-retention 修复，不是继续 MLP matrix/optimizer 同族 sweep。它以 F10-T7 的 B1 cross-split consensus transfer 为候选 target，但每次提交前用 train split A/B 对比同范数 SGD/gradient residual；只有候选 target 在 train lookahead 上不输 gradient 且没有明显帮助 corrupted-label split 时才提交，否则退回普通 SGD。

### F24 Group Evidence

| condition | run label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | gate h800 | gate h1600 | lookahead advantage h1600 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

### F24 Row Examples

| run label | carrier | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{example_rows}

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M71-TrainLookaheadB1ConsensusTransferFU`，方向来自 train-stream B1 consensus transfer target，并写入 semantic contract。
- 修改 `experiments/run_v17_common.py`：新增 M71 训练分支，用 train split A/B 与同范数 gradient residual 做在线 lookahead 对照，并用 corrupted-label split 做坏方向过滤；记录 gate accept、target/gradient lookahead gain、current cosine 与 target diagnostics。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 `F24-B1-consensus-lookahead-gated-transfer` spec 与 v21.01 target 白名单。
- 新增 `experiments/run_v21_01_f24_b1_lookahead_summary.py`，只从 fresh `v2101_f24_` rows 汇总并更新 route/manifest/复盘。

### F24 结论 / Insight

- F24 没有形成 promotion-ready retained source；`promotion_allowed` 保持 0。
- 如果 candidate early-chain groups 为 0，说明 train-only target-vs-gradient selector 没有把 B1 late-rebound target 提前变成连续 h800/h1600 source。
- 如果 candidate h3200 groups 为 0，即便局部 rows 可能出现 late-positive 或 retained-like 信号，也不能通过 grouped 证据写 breakthrough。
- 当前 blocker 继续定位在 train-only source/target observability 与 target-to-retention dynamics；不能用 high ActuationR2 或 single-row positive 替代 retained chain。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F24 B1 Consensus Lookahead-Gated Transfer"
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
    examples = [r for r in rows if r.get("v21_id") in {CANDIDATE, *REFERENCE_IDS.keys(), *CONTROL_IDS.keys()}]
    candidates = [r for r in summary if int_flag(r.get("candidate"))]
    smoke = [r for r in rows if "smoke" in str(r.get("run_label", ""))]
    full = [r for r in rows if "full" in str(r.get("run_label", ""))]
    candidate_early = sum(int_flag(r.get("early_chain_group")) for r in candidates)
    candidate_h3200 = sum(int_flag(r.get("productive_h3200_group")) for r in candidates)
    full_recommended = int(bool(smoke) and not full and candidate_early > 0)
    if candidate_h3200 > 0:
        route_label = "F24B1LookaheadCandidateNeedsConfirmation"
    elif full:
        route_label = "F24B1LookaheadNoRetained"
    elif full_recommended:
        route_label = "F24B1LookaheadSmokeNeedsFull"
    else:
        route_label = "F24B1LookaheadSmokeNoEarlyChain"
    decision = {
        "decision": route_label,
        "f24_rows": len(rows),
        "f24_blocked_rows": sum(1 for r in rows if str(r.get("execution_status", "")).startswith("blocked")),
        "f24_smoke_rows": len(smoke),
        "f24_full_rows": len(full),
        "f24_candidate_groups": len(candidates),
        "f24_candidate_early_chain_groups": candidate_early,
        "f24_candidate_productive_h3200_groups": candidate_h3200,
        "f24_full_escalation_recommended": full_recommended,
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v21_01_f24_b1_lookahead_summary.csv", summary)
    write_rows(out_dir / "v21_01_f24_b1_lookahead_examples.csv", examples)
    write_rows(out_dir / "v21_01_f24_b1_lookahead_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": f"R2-LateReboundNoContinuousRetention-{route_label}",
            "promotion_allowed": 0,
            "f24_rows": decision["f24_rows"],
            "f24_blocked_rows": decision["f24_blocked_rows"],
            "f24_candidate_early_chain_groups": candidate_early,
            "f24_candidate_productive_h3200_groups": candidate_h3200,
            "f24_full_escalation_recommended": full_recommended,
            "target_to_retention_selector_completed": int(not full_recommended),
        }
    )
    write_json(out_dir / "v21_01_route_decision.json", route)
    write_json(out_dir / "route_decision.json", route)
    decision = {**decision, "route": route.get("route", ""), "promotion_allowed": route.get("promotion_allowed", 0)}
    append_unique_manifest(out_dir, ARTIFACTS)
    replace_section(V2101_RECAP_DOC, render_section(decision, summary, examples))
    if not args.no_log:
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v21_01_f24_b1_lookahead_summary.py --out-dir {out_dir}",
            status="completed",
            note=f"rows={decision['f24_rows']} blocked={decision['f24_blocked_rows']} early={candidate_early} productive_h3200={candidate_h3200} full_recommended={full_recommended} promotion=0",
        )
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    required = [str(r.get("artifact")) for r in manifest if int_flag(r.get("required", 1))]
    build_packet(out_dir, required)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
