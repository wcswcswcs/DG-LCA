#!/usr/bin/env python3
"""F34/F35 view-consistent source-target summary."""

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
    "F34-view-consistent-loss-b3-null": "view-consistent loss target with B3 null",
    "F35-loss-warm-to-view-consistent-loss": "loss-warm to view-consistent target",
}
REFERENCE_IDS = {
    "F3-T1-loss-cotangent-target": "loss-cotangent reference",
    "F25-loss-warm-to-b1-consensus-migration": "loss-warm B1 consensus reference",
    "F33-loss-warm-to-b1-consensus-b3-null": "loss-warm B1 consensus B3-null reference",
}
CONTROL_IDS = {
    "F3-T5-random-matched-target": "random matched target control",
    "F9-TCTRL-stable-random-target": "stable random target control",
    "CTRL-SGD": "SGD control",
    "CTRL-RandomMatchedNorm": "random matched norm control",
    "CTRL-NoOpMatchedOverhead": "NoOp matched overhead control",
}
HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
ARTIFACTS = [
    "v21_01_f34_view_consistency_summary.csv",
    "v21_01_f34_view_consistency_decision.csv",
    "v21_01_f34_view_consistency_examples.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-prefix", default="v2101_f34_")
    p.add_argument("--no-log", action="store_true")
    return p


def source(row: dict[str, str], h: int) -> float:
    return finite_float(row.get(f"source_h{h}"))


def mean(vals: list[float]) -> float | str:
    data = [v for v in vals if math.isfinite(v)]
    return sum(data) / len(data) if data else ""


def diag_value(row: dict[str, str], name: str, h: int = 800) -> float:
    value = finite_float(row.get(f"{name}_h{h}"))
    if math.isfinite(value):
        return value
    return finite_float(row.get(name))


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
            "view_stable_b1_h800_mean": mean([diag_value(r, "target_view_stable_fraction_b1") for r in group]),
            "view_stable_b2_h800_mean": mean([diag_value(r, "target_view_stable_fraction_b2") for r in group]),
            "view_align_b1_h800_mean": mean([diag_value(r, "target_view_alignment_b1_mean") for r in group]),
            "view_align_b2_h800_mean": mean([diag_value(r, "target_view_alignment_b2_mean") for r in group]),
            "target_consensus_density_h800_mean": mean([diag_value(r, "target_consensus_density") for r in group]),
            "target_b3_null_rows_h800_mean": mean([diag_value(r, "target_b3_null_rows") for r in group]),
            "gate_accept_h800_rows": sum(int_flag(r.get("source_state_gate_accept_h800")) for r in group),
            "gate_accept_h1600_rows": sum(int_flag(r.get("source_state_gate_accept_h1600")) for r in group),
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
        "| {condition} | {run_label} | {carrier} | {v21_id} | {rows} | {blocked_rows} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h4800_mean} | {source_h6400_mean} | {early_chain_group} | {productive_h3200_group} | {view_stable_b1_h800_mean} | {view_stable_b2_h800_mean} | {view_align_b1_h800_mean} | {view_align_b2_h800_mean} | {target_consensus_density_h800_mean} | {target_b3_null_rows_h800_mean} |".format(**r)
        for r in summary
    )
    example_rows = "\n".join(
        "| {run_label} | {carrier} | {v21_id} | {dataset} | {seed} | {source_h800} | {source_h1600} | {source_h3200} | {source_h4800} | {source_h6400} | {washout_flag} | {late_rebound_flag} | {retained_flag} |".format(**r)
        for r in examples[:48]
    )
    return f"""
## 2026-06-04 F34-F35 View-Consistent Loss Target

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F34/F35 rows / blocked rows: {decision.get('f34_rows')} / {decision.get('f34_blocked_rows')}
- smoke rows / full rows: {decision.get('f34_smoke_rows')} / {decision.get('f34_full_rows')}
- candidate early-chain groups: {decision.get('f34_candidate_early_chain_groups')}
- candidate productive h3200 groups: {decision.get('f34_candidate_productive_h3200_groups')}
- full escalation recommended: {decision.get('f34_full_escalation_recommended')}

F34/F35 是在 F17 selector 不可行动、F26-F33 B1 consensus/B3-null 仍无 early chain 后的新 source-target theory。它不使用 h800 source readback，不读 validation/test/future/query；只用当前 train batch 的 loss-cotangent target 与小输入扰动视角下的 per-sample/class 方向一致性，测试“对局部扰动稳定的 loss target”能否比 raw loss/random/B1 consensus 更早形成 h800/h1600 连续链。

### F34/F35 Group Evidence

| condition | run label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | view stable b1 | view stable b2 | view align b1 | view align b2 | consensus density | B3-null rows |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

### F34/F35 Row Examples

| run label | carrier | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{example_rows}

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `view_consistent_loss` target kind。该 target 对 B1/B2 train split 分别加入固定 seed 小输入扰动，只有 clean/perturbed loss-cotangent 方向 per-sample cosine 达标、且 B1/B2/扰动视角 class-sign 一致的分量才进入 target。
- 修改 `dgkan/fu/mechanisms.py`：新增 `M81-ViewConsistentLossTargetFU`，使用 `view_consistent_loss` + `b1_b3zero` fit scope；B3 split 作为零位移安全约束，不作为方向源。
- 修改 `dgkan/fu/mechanisms.py` 与 `experiments/run_v17_common.py`：新增 `M82-LossWarmToViewConsistentLossMigrationFU`，warmup 阶段用 `M49` loss-cotangent，warmup 后迁移到 `M81`。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F34/F35 specs，并纳入 target scope 白名单。
- 修改 `experiments/run_v21_01_s06_truth_gate.py`：把 F34 summary runner、M81/M82 合约与 M81 update-semantics smoke 纳入 S0.6。
- 新增 `experiments/run_v21_01_f34_view_consistency_summary.py`，只读取 fresh `v2101_f34_` rows 汇总 route、manifest 和复盘。

### F34/F35 结论 / Insight

- 若 F34/F35 grouped h800/h1600 仍不能同时为正，说明“train-only 扰动视角稳定性”也没有把 late-rebound target 转成 early retained source。
- 若 F35 相对 F34 改善 h800 但 h1600/h3200 断链，说明 loss-cotangent early closure 仍不能迁移成 retained source，blocker 继续指向 target-to-retention dynamics。
- 若 random/stable-random controls 同步变好，必须按 control-equivalent 处理，不能写 breakthrough。
- 只有 fresh full grouped h800/h1600/h3200/h4800 与 controls attribution 同时通过，才允许进入 promotion；本轮 smoke 只负责决定是否升级 full。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F34-F35 View-Consistent Loss Target"
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
    examples = [r for r in rows if r.get("v21_id") in {*CANDIDATES.keys(), *REFERENCE_IDS.keys(), *CONTROL_IDS.keys()}]
    candidates = [r for r in summary if int_flag(r.get("candidate"))]
    smoke = [r for r in rows if "smoke" in str(r.get("run_label", ""))]
    full = [r for r in rows if "full" in str(r.get("run_label", ""))]
    candidate_early = sum(int_flag(r.get("early_chain_group")) for r in candidates)
    candidate_h3200 = sum(int_flag(r.get("productive_h3200_group")) for r in candidates)
    full_recommended = int(bool(smoke) and not full and candidate_early > 0)
    if candidate_h3200 > 0:
        route_label = "F34ViewConsistencyCandidateNeedsConfirmation"
    elif full:
        route_label = "F34ViewConsistencyNoRetained"
    elif full_recommended:
        route_label = "F34ViewConsistencySmokeNeedsFull"
    else:
        route_label = "F34ViewConsistencySmokeNoEarlyChain"
    decision = {
        "decision": route_label,
        "f34_rows": len(rows),
        "f34_blocked_rows": sum(1 for r in rows if str(r.get("execution_status", "")).startswith("blocked")),
        "f34_smoke_rows": len(smoke),
        "f34_full_rows": len(full),
        "f34_candidate_groups": len(candidates),
        "f34_candidate_early_chain_groups": candidate_early,
        "f34_candidate_productive_h3200_groups": candidate_h3200,
        "f34_full_escalation_recommended": full_recommended,
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v21_01_f34_view_consistency_summary.csv", summary)
    write_rows(out_dir / "v21_01_f34_view_consistency_examples.csv", examples)
    write_rows(out_dir / "v21_01_f34_view_consistency_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": f"R2-LateReboundNoContinuousRetention-{route_label}",
            "promotion_allowed": 0,
            "f34_rows": decision["f34_rows"],
            "f34_blocked_rows": decision["f34_blocked_rows"],
            "f34_candidate_early_chain_groups": candidate_early,
            "f34_candidate_productive_h3200_groups": candidate_h3200,
            "f34_full_escalation_recommended": full_recommended,
            "view_consistency_target_completed": int(not full_recommended),
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
            f"{PYTHON} experiments/run_v21_01_f34_view_consistency_summary.py --out-dir {out_dir}",
            status="completed",
            note=f"rows={decision['f34_rows']} blocked={decision['f34_blocked_rows']} early={candidate_early} productive_h3200={candidate_h3200} full_recommended={full_recommended} promotion=0",
        )
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    required = [str(r.get("artifact")) for r in manifest if int_flag(r.get("required", 1))]
    build_packet(out_dir, required)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
