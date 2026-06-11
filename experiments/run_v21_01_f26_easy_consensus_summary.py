#!/usr/bin/env python3
"""F26-F33 B1 consensus target-to-retention summary."""

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


CANDIDATES = {
    "F26-easy-b1-consensus-transfer": "easy-example B1 consensus transfer",
    "F27-loss-warm-to-easy-b1-consensus-migration": "loss-warm to easy B1 consensus migration",
    "F28-loss-easy-b1-consensus-blend": "loss + easy B1 consensus blend",
    "F29-loss-warm-to-loss-easy-b1-consensus-blend": "loss-warm to loss + easy B1 consensus blend",
    "F30-gain-gated-loss-warm-b1-consensus": "gain-gated loss-warm to B1 consensus",
    "F31-gain-gated-loss-warm-blend-consensus": "gain-gated loss-warm to blended consensus",
    "F32-b1-consensus-b3-null-transfer": "B1 consensus transfer with B3-null safety projection",
    "F33-loss-warm-to-b1-consensus-b3-null": "loss-warm to B1 consensus with B3-null migration",
}
REFERENCE_IDS = {
    "F25-loss-warm-to-b1-consensus-migration": "loss-warm to raw B1 consensus reference",
    "F10-T7-b1-cross-split-consensus-transfer": "raw B1 consensus reference",
    "F3-T1-loss-cotangent-target": "loss-cotangent warmup reference",
}
CONTROL_IDS = {
    "F3-T5-random-matched-target": "random matched target control",
    "CTRL-SGD": "SGD control",
    "CTRL-RandomMatchedNorm": "Random matched norm",
    "CTRL-NoOpMatchedOverhead": "NoOp matched overhead",
}
HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
ARTIFACTS = [
    "v21_01_f26_easy_consensus_summary.csv",
    "v21_01_f26_easy_consensus_decision.csv",
    "v21_01_f26_easy_consensus_examples.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-prefix", default="v2101_f26_")
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
            "condition": CANDIDATES.get(vid, REFERENCE_IDS.get(vid, CONTROL_IDS.get(vid, ""))),
            "rows": len(group),
            "blocked_rows": sum(1 for r in group if str(r.get("execution_status", "")).startswith("blocked")),
            "candidate": int(vid in CANDIDATES),
            "reference": int(vid in REFERENCE_IDS),
            "control": int(vid in CONTROL_IDS or vid.startswith("CTRL")),
            "washout_rows": sum(int_flag(r.get("washout_flag")) for r in group),
            "late_rebound_rows": sum(int_flag(r.get("late_rebound_flag")) for r in group),
            "retained_rows": sum(int_flag(r.get("retained_flag")) for r in group),
            "gate_accept_h800_rows": sum(int_flag(r.get("source_state_gate_accept_h800")) for r in group),
            "gate_accept_h1600_rows": sum(int_flag(r.get("source_state_gate_accept_h1600")) for r in group),
            "easy_fraction_b1_h800_mean": mean([finite_float(r.get("target_easy_fraction_b1_h800")) for r in group]),
            "easy_fraction_b2_h800_mean": mean([finite_float(r.get("target_easy_fraction_b2_h800")) for r in group]),
            "target_consensus_density_h800_mean": mean([finite_float(r.get("target_consensus_density_h800")) for r in group]),
            "target_b3_null_rows_h800_mean": mean([finite_float(r.get("target_b3_null_rows_h800")) for r in group]),
            "target_b3_null_rows_h1600_mean": mean([finite_float(r.get("target_b3_null_rows_h1600")) for r in group]),
            "current_cos_h800_mean": mean([finite_float(r.get("source_state_current_cos_h800")) for r in group]),
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
        "| {condition} | {run_label} | {carrier} | {v21_id} | {rows} | {blocked_rows} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h4800_mean} | {source_h6400_mean} | {early_chain_group} | {productive_h3200_group} | {gate_accept_h800_rows} | {gate_accept_h1600_rows} | {easy_fraction_b1_h800_mean} | {easy_fraction_b2_h800_mean} | {target_b3_null_rows_h800_mean} | {target_b3_null_rows_h1600_mean} |".format(**r)
        for r in summary
    )
    example_rows = "\n".join(
        "| {run_label} | {carrier} | {v21_id} | {dataset} | {seed} | {source_h800} | {source_h1600} | {source_h3200} | {source_h4800} | {source_h6400} | {washout_flag} | {late_rebound_flag} | {retained_flag} |".format(**r)
        for r in examples[:48]
    )
    return f"""
## 2026-06-04 F26-F33 Easy/Blended/Gated/B3-Null B1 Consensus Target

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F26-F33 rows / blocked rows: {decision.get('f26_rows')} / {decision.get('f26_blocked_rows')}
- smoke rows / full rows: {decision.get('f26_smoke_rows')} / {decision.get('f26_full_rows')}
- candidate early-chain groups: {decision.get('f26_candidate_early_chain_groups')}
- candidate productive h3200 groups: {decision.get('f26_candidate_productive_h3200_groups')}
- full escalation recommended: {decision.get('f26_full_escalation_recommended')}

F26-F33 是新的 source/target theory 尝试，不是 M71/M72 的 scale/alt/warmup 小修。F26/F27 只用 train batch 内 low-loss / already-stable examples 生成 B1 split-consensus target；F28/F29 保留 loss-cotangent early closure，同时叠加 easy B1 consensus；F30/F31 增加 train-split B2 transfer precommit gate，只有 B2 gain 明确压过 B3 safety/noise 时才提交迁移 update；F32/F33 把 B3/safety split 作为 readout operator 的零位移约束，测试 late-rebound target 是否能被 source-channel/reservoir exclusion 提前成 h800/h1600 连续链。

### F26-F33 Group Evidence

| condition | run label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | gate h800 | gate h1600 | easy b1 h800 | easy b2 h800 | B3-null rows h800 | B3-null rows h1600 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

### F26-F33 Row Examples

| run label | carrier | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{example_rows}

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M73-EasyB1ConsensusTransferFU` 与 `M74-LossWarmToEasyB1ConsensusMigrationFU`，并新增 `easy_split_consensus` target kind。该 target 只使用当前 train batch 内低 loss 或已预测正确的样本，叠加 B1/B2 类方向一致 mask。
- 修改 `dgkan/fu/mechanisms.py`：新增 `M75-LossEasyB1ConsensusBlendFU` 与 `M76-LossWarmToLossEasyB1ConsensusBlendFU`，并新增 `loss_easy_consensus_blend` target kind。该 target 保留 loss-cotangent 分量，同时加入 easy B1 consensus 分量。
- 修改 `dgkan/fu/mechanisms.py`：新增 `M77-GainGatedLossWarmB1ConsensusMigrationFU` 与 `M78-GainGatedLossWarmBlendMigrationFU`，用于给迁移 target 提供 train-split B2 gain precommit 语义。
- 修改 `dgkan/fu/mechanisms.py`：新增 `M79-B1ConsensusB3NullTransferFU` 与 `M80-LossWarmToB1ConsensusB3NullMigrationFU`，并新增 `b1_b3zero` / `b1b2_b3zero` fit scope。该 scope 用 B1 target 求解 readout operator，同时把 B3 split 作为零位移约束纳入同一个 least-squares system。
- 修改 `experiments/run_v17_common.py`：扩展 M72/M74 two-phase migration branch；M74 warmup 后迁移到 M73 easy-consensus target。
- 修改 `experiments/run_v17_common.py`：继续扩展 M76 two-phase migration branch；M76 warmup 后迁移到 M75 blended target。
- 修改 `experiments/run_v17_common.py`：新增 M77/M78 gate 分支；gate reject 时不提交 FU residual，回退到普通 SGD step，并把 gate_accept/gain 写入 trace。
- 修改 `experiments/run_v17_common.py`：扩展 M80 two-phase migration branch；M80 warmup 后迁移到 M79 B3-null B1 consensus target。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F26-F33 specs 和 target scope 白名单。
- 修改 `experiments/run_v21_01_s06_truth_gate.py`：把 F26 summary runner 与 M73-M80 semantic contracts 纳入 S0.6。
- 新增 `experiments/run_v21_01_f26_easy_consensus_summary.py`，从 fresh `v2101_f26_` rows 汇总并更新 route/manifest/复盘。

### F26-F33 结论 / Insight

- 如果 grouped h800/h1600 仍不能同时为正，说明“只保留 train low-loss/easy examples 的 B1 consensus target”仍不能形成 early source chain。
- 如果 F27 比 F25 更差，说明 F25 的 near-chain 改善不是因为 hard examples 过多，而可能是 loss target 本身提供了早期 closure。
- 如果 F29 比 F27 更好但仍不能 early-chain，说明保留 loss target 可以缓解硬切换问题，但 easy-consensus 分量仍不是足够的 retained source selector。
- 如果 F30/F31 gate accept 很少且 h800 仍负，说明“无门槛提交”不是主要 blocker；如果 gate accept 很多但仍负，说明 train-split B2 gain 本身不能预测 early source retention。
- 如果 F32/F33 仍不能 early-chain，说明 B3/safety null 约束也不能把 B1 late-rebound target 转成 early retained source，blocker 更接近 B1 target 本身的早期相位问题，而不是 reservoir 泄漏。
- smoke positive 只允许升级 full，不允许直接写 breakthrough；controls 或 random target 若同步 positive，必须写 control-equivalent。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    old_marker = "\n## 2026-06-04 F26/F27 Easy-Example B1 Consensus Target"
    mid_marker = "\n## 2026-06-04 F26-F29 Easy/Blended B1 Consensus Target"
    gate_marker = "\n## 2026-06-04 F26-F31 Easy/Blended/Gated B1 Consensus Target"
    marker = "\n## 2026-06-04 F26-F33 Easy/Blended/Gated/B3-Null B1 Consensus Target"
    idx = text.find(marker)
    if idx < 0:
        idx = text.find(gate_marker)
    if idx < 0:
        idx = text.find(mid_marker)
    if idx < 0:
        idx = text.find(old_marker)
    if idx >= 0:
        path.write_text(text[:idx].rstrip() + "\n" + section, encoding="utf-8")
    else:
        append_text(path, section)


def main() -> int:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    rows = [r for r in read_rows(out_dir / "v21_01_source_retention_matrix.csv") if str(r.get("run_label", "")).startswith(str(args.run_prefix))]
    summary = summarize(rows)
    examples = [r for r in rows if r.get("v21_id") in {*CANDIDATES.keys(), *REFERENCE_IDS.keys(), *CONTROL_IDS.keys()}]
    candidates = [r for r in summary if int_flag(r.get("candidate"))]
    smoke = [r for r in rows if "smoke" in str(r.get("run_label", ""))]
    full = [r for r in rows if "full" in str(r.get("run_label", ""))]
    candidate_early = sum(int_flag(r.get("early_chain_group")) for r in candidates)
    candidate_h3200 = sum(int_flag(r.get("productive_h3200_group")) for r in candidates)
    full_recommended = int(bool(smoke) and not full and candidate_early > 0)
    if candidate_h3200 > 0:
        route_label = "F26EasyConsensusCandidateNeedsConfirmation"
    elif full:
        route_label = "F26EasyConsensusNoRetained"
    elif full_recommended:
        route_label = "F26EasyConsensusSmokeNeedsFull"
    else:
        route_label = "F26EasyConsensusSmokeNoEarlyChain"
    decision = {
        "decision": route_label,
        "f26_rows": len(rows),
        "f26_blocked_rows": sum(1 for r in rows if str(r.get("execution_status", "")).startswith("blocked")),
        "f26_smoke_rows": len(smoke),
        "f26_full_rows": len(full),
        "f26_candidate_groups": len(candidates),
        "f26_candidate_early_chain_groups": candidate_early,
        "f26_candidate_productive_h3200_groups": candidate_h3200,
        "f26_full_escalation_recommended": full_recommended,
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v21_01_f26_easy_consensus_summary.csv", summary)
    write_rows(out_dir / "v21_01_f26_easy_consensus_examples.csv", examples)
    write_rows(out_dir / "v21_01_f26_easy_consensus_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": f"R2-LateReboundNoContinuousRetention-{route_label}",
            "promotion_allowed": 0,
            "f26_rows": decision["f26_rows"],
            "f26_blocked_rows": decision["f26_blocked_rows"],
            "f26_candidate_early_chain_groups": candidate_early,
            "f26_candidate_productive_h3200_groups": candidate_h3200,
            "f26_full_escalation_recommended": full_recommended,
            "easy_consensus_target_completed": int(not full_recommended),
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
            f"{PYTHON} experiments/run_v21_01_f26_easy_consensus_summary.py --out-dir {out_dir}",
            status="completed",
            note=f"rows={decision['f26_rows']} blocked={decision['f26_blocked_rows']} early={candidate_early} productive_h3200={candidate_h3200} full_recommended={full_recommended} promotion=0",
        )
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    required = [str(r.get("artifact")) for r in manifest if int_flag(r.get("required", 1))]
    build_packet(out_dir, required)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
