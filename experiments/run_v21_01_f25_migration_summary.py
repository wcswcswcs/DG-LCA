#!/usr/bin/env python3
"""F25 loss-target warmup to B1-consensus delayed migration summary."""

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


CANDIDATE = "F25-loss-warm-to-b1-consensus-migration"
REFERENCE_IDS = {
    "F3-T1-loss-cotangent-target": "loss-cotangent warmup reference",
    "F10-T7-b1-cross-split-consensus-transfer": "B1 consensus migration reference",
    "F24-B1-consensus-lookahead-gated-transfer": "B1 lookahead gate reference",
}
CONTROL_IDS = {
    "F3-T5-random-matched-target": "random matched target control",
    "CTRL-SGD": "SGD control",
    "CTRL-RandomMatchedNorm": "Random matched norm",
    "CTRL-NoOpMatchedOverhead": "NoOp matched overhead",
}
HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
ARTIFACTS = [
    "v21_01_f25_migration_summary.csv",
    "v21_01_f25_migration_decision.csv",
    "v21_01_f25_migration_examples.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-prefix", default="v2101_f25_")
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
            "condition": "F25 loss-warm to B1 consensus migration" if vid == CANDIDATE else REFERENCE_IDS.get(vid, CONTROL_IDS.get(vid, "")),
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
            "migration_phase_h1600_mean": mean([finite_float(r.get("source_state_balance_mean_h1600")) for r in group]),
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
        "| {condition} | {run_label} | {carrier} | {v21_id} | {rows} | {blocked_rows} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h4800_mean} | {source_h6400_mean} | {early_chain_group} | {productive_h3200_group} | {gate_accept_h800_rows} | {gate_accept_h1600_rows} | {migration_phase_h1600_mean} |".format(**r)
        for r in summary
    )
    example_rows = "\n".join(
        "| {run_label} | {carrier} | {v21_id} | {dataset} | {seed} | {source_h800} | {source_h1600} | {source_h3200} | {source_h4800} | {source_h6400} | {washout_flag} | {late_rebound_flag} | {retained_flag} |".format(**r)
        for r in examples[:48]
    )
    return f"""
## 2026-06-04 F25 Loss-Warm To B1 Consensus Migration

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F25 rows / blocked rows: {decision.get('f25_rows')} / {decision.get('f25_blocked_rows')}
- F25 smoke rows / full rows: {decision.get('f25_smoke_rows')} / {decision.get('f25_full_rows')}
- candidate early-chain groups: {decision.get('f25_candidate_early_chain_groups')}
- candidate productive h3200 groups: {decision.get('f25_candidate_productive_h3200_groups')}
- full escalation recommended: {decision.get('f25_full_escalation_recommended')}

F25 是 delayed migration 测试：前 800 step 的 FU commit 用 loss-cotangent target 尝试 early closure，之后迁移到 B1 cross-split consensus transfer。非 FU commit step 使用 SGD；不使用 validation/test/future/query 生成方向。

### F25 Group Evidence

| condition | run label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | gate h800 | gate h1600 | migration phase h1600 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

### F25 Row Examples

| run label | carrier | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{example_rows}

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M72-LossWarmToB1ConsensusMigrationFU` 与 semantic contract。
- 修改 `experiments/run_v17_common.py`：新增 M72 两阶段训练分支，warmup 阶段使用 `M49-LossCotangentTargetFU`，迁移阶段使用 `M60-B1CrossSplitConsensusTransferFU`，并记录 migration phase/gate/current-cos 诊断。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 `F25-loss-warm-to-b1-consensus-migration` spec 和 v21.01 target 白名单。
- 新增 `experiments/run_v21_01_f25_migration_summary.py`，从 fresh `v2101_f25_` rows 汇总并更新 route/manifest/复盘。

### F25 结论 / Insight

- F25 只有在 grouped h800/h1600 同时为正时才允许升级 full；否则 smoke 只作为 delayed migration blocker。
- 如果 F25 仍无 early-chain，说明 loss-target early closure 与 B1 delayed target 不能简单串联成 retained source。
- 如果 controls 或 random matched target 同样出现 positive，则必须按 control-equivalent 处理，不能 promotion。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F25 Loss-Warm To B1 Consensus Migration"
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
        route_label = "F25MigrationCandidateNeedsConfirmation"
    elif full:
        route_label = "F25MigrationNoRetained"
    elif full_recommended:
        route_label = "F25MigrationSmokeNeedsFull"
    else:
        route_label = "F25MigrationSmokeNoEarlyChain"
    decision = {
        "decision": route_label,
        "f25_rows": len(rows),
        "f25_blocked_rows": sum(1 for r in rows if str(r.get("execution_status", "")).startswith("blocked")),
        "f25_smoke_rows": len(smoke),
        "f25_full_rows": len(full),
        "f25_candidate_groups": len(candidates),
        "f25_candidate_early_chain_groups": candidate_early,
        "f25_candidate_productive_h3200_groups": candidate_h3200,
        "f25_full_escalation_recommended": full_recommended,
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v21_01_f25_migration_summary.csv", summary)
    write_rows(out_dir / "v21_01_f25_migration_examples.csv", examples)
    write_rows(out_dir / "v21_01_f25_migration_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": f"R2-LateReboundNoContinuousRetention-{route_label}",
            "promotion_allowed": 0,
            "f25_rows": decision["f25_rows"],
            "f25_blocked_rows": decision["f25_blocked_rows"],
            "f25_candidate_early_chain_groups": candidate_early,
            "f25_candidate_productive_h3200_groups": candidate_h3200,
            "f25_full_escalation_recommended": full_recommended,
            "delayed_migration_writer_completed": int(not full_recommended),
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
            f"{PYTHON} experiments/run_v21_01_f25_migration_summary.py --out-dir {out_dir}",
            status="completed",
            note=f"rows={decision['f25_rows']} blocked={decision['f25_blocked_rows']} early={candidate_early} productive_h3200={candidate_h3200} full_recommended={full_recommended} promotion=0",
        )
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    required = [str(r.get("artifact")) for r in manifest if int_flag(r.get("required", 1))]
    build_packet(out_dir, required)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
