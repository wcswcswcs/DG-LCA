#!/usr/bin/env python3
"""F36/F37 low-bank source-channel writer summary."""

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
    "F36-lowbank-loss-b3-null": "low-bank loss target with B3 reservoir null",
    "F37-loss-warm-to-lowbank-loss-b3-null": "loss-warm to low-bank source-channel writer",
    "F38-gain-gated-lowbank-loss-b3-null": "gain-gated low-bank loss target",
    "F39-loss-warm-to-gated-lowbank-loss-b3-null": "loss-warm to gain-gated low-bank writer",
}
REFERENCE_IDS = {
    "F3-T1-loss-cotangent-target": "loss-cotangent reference",
    "F25-loss-warm-to-b1-consensus-migration": "loss-warm B1 consensus reference",
    "F33-loss-warm-to-b1-consensus-b3-null": "loss-warm B1 consensus B3-null reference",
    "F34-view-consistent-loss-b3-null": "view-consistent loss B3-null reference",
    "F35-loss-warm-to-view-consistent-loss": "loss-warm view-consistent reference",
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
    "v21_01_f36_lowbank_source_channel_summary.csv",
    "v21_01_f36_lowbank_source_channel_decision.csv",
    "v21_01_f36_lowbank_source_channel_examples.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-prefix", default="v2101_f36_")
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
            "source_channel_projection_h800_mean": mean([diag_value(r, "source_channel_projection") for r in group]),
            "source_channel_projection_h1600_mean": mean([diag_value(r, "source_channel_projection", 1600) for r in group]),
            "reservoir_projection_h800_mean": mean([diag_value(r, "reservoir_projection") for r in group]),
            "reservoir_projection_h1600_mean": mean([diag_value(r, "reservoir_projection", 1600) for r in group]),
            "source_bank_feature_count_h800_mean": mean([diag_value(r, "source_bank_feature_count") for r in group]),
            "source_bank_feature_count_h1600_mean": mean([diag_value(r, "source_bank_feature_count", 1600) for r in group]),
            "reservoir_bank_feature_count_h800_mean": mean([diag_value(r, "reservoir_bank_feature_count") for r in group]),
            "reservoir_bank_feature_count_h1600_mean": mean([diag_value(r, "reservoir_bank_feature_count", 1600) for r in group]),
            "source_bank_feature_norm_h800_mean": mean([diag_value(r, "source_bank_feature_norm") for r in group]),
            "reservoir_bank_feature_norm_h800_mean": mean([diag_value(r, "reservoir_bank_feature_norm") for r in group]),
            "low_degree_source_energy_h800_mean": mean([diag_value(r, "low_degree_source_energy") for r in group]),
            "low_degree_source_energy_h1600_mean": mean([diag_value(r, "low_degree_source_energy", 1600) for r in group]),
            "high_degree_reservoir_energy_h800_mean": mean([diag_value(r, "high_degree_reservoir_energy") for r in group]),
            "high_degree_reservoir_energy_h1600_mean": mean([diag_value(r, "high_degree_reservoir_energy", 1600) for r in group]),
            "degree_entropy_h800_mean": mean([diag_value(r, "degree_entropy") for r in group]),
            "target_b3_null_rows_h800_mean": mean([diag_value(r, "target_b3_null_rows") for r in group]),
            "target_b3_null_rows_h1600_mean": mean([diag_value(r, "target_b3_null_rows", 1600) for r in group]),
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
        "| {condition} | {run_label} | {carrier} | {v21_id} | {rows} | {blocked_rows} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h4800_mean} | {early_chain_group} | {productive_h3200_group} | {source_channel_projection_h800_mean} | {source_channel_projection_h1600_mean} | {reservoir_projection_h800_mean} | {reservoir_projection_h1600_mean} | {source_bank_feature_count_h800_mean} | {source_bank_feature_count_h1600_mean} | {reservoir_bank_feature_count_h800_mean} | {reservoir_bank_feature_count_h1600_mean} | {low_degree_source_energy_h800_mean} | {low_degree_source_energy_h1600_mean} | {high_degree_reservoir_energy_h800_mean} | {high_degree_reservoir_energy_h1600_mean} | {degree_entropy_h800_mean} | {target_b3_null_rows_h800_mean} | {target_b3_null_rows_h1600_mean} |".format(**r)
        for r in summary
    )
    example_rows = "\n".join(
        "| {run_label} | {carrier} | {v21_id} | {dataset} | {seed} | {source_h800} | {source_h1600} | {source_h3200} | {source_h4800} | {washout_flag} | {late_rebound_flag} | {retained_flag} |".format(**r)
        for r in examples[:60]
    )
    return f"""
## 2026-06-04 F36-F39 Low-Bank Source-Channel Writer

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F36-F39 rows / blocked rows: {decision.get('f36_rows')} / {decision.get('f36_blocked_rows')}
- smoke rows / full rows: {decision.get('f36_smoke_rows')} / {decision.get('f36_full_rows')}
- candidate early-chain groups: {decision.get('f36_candidate_early_chain_groups')}
- candidate productive h3200 groups: {decision.get('f36_candidate_productive_h3200_groups')}
- full escalation recommended: {decision.get('f36_full_escalation_recommended')}

F36-F39 是在 F34/F35 view-consistency 仍没有 early-chain 后，对计划 7.6 “低阶/低频 source bank + high-degree/high-frequency reservoir” 的直接实现与 gate 修复。方向仍只来自 train stream；B3 split 只作为零位移安全约束，不作为方向源；h800 source readback 只用于事后审计，不用于选择方向。F38/F39 只使用 train split B2/B3 precommit gains 决定是否提交 low-bank update，不使用 validation/test/future/query。

### F36-F39 Group Evidence

| condition | run label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | early chain | productive h3200 | source proj h800 | source proj h1600 | reservoir proj h800 | reservoir proj h1600 | source bank h800 | source bank h1600 | reservoir h800 | reservoir h1600 | source energy h800 | source energy h1600 | reservoir energy h800 | reservoir energy h1600 | entropy h800 | B3-null h800 | B3-null h1600 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

### F36-F39 Row Examples

| run label | carrier | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | washout | late | retained |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{example_rows}

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `low_bank`/`reservoir_bank` readout feature selection，并记录 `source_channel_projection`、`reservoir_projection`、source/reservoir feature norm、degree/band entropy 等诊断。
- 新增 `M83-LowBankLossB3NullFU`：只允许 readout low-bank feature 承接 loss-cotangent target，同时使用 `b1_b3zero`，把 B3 作为 reservoir null 约束。
- 新增 `M84-LossWarmToLowBankLossB3NullMigrationFU`：前 800 step 使用 `M49` loss-cotangent warmup，之后迁移到 `M83`，用于测试 early closure 是否能桥接到 low-bank source channel。
- 修改 `experiments/run_v17_common.py`：把 low-bank source/reservoir 诊断写入 trace，并把 M83 纳入 direct target-mechanism branch；这修复了初版 F36 direct writer 没有真正接入训练 loop 的实现 blocker。
- 修改 `experiments/run_v17_common.py`：把 M84/M85/M86 纳入 migration/gain-gated branch；M85 direct gated warmup 强制为 0，避免被错误当作 800-step warmup writer。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F36/F37 specs 和 target scope 白名单。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F38/F39 gain-gated low-bank specs 和 target scope 白名单。
- 修改 `experiments/run_v21_01_s06_truth_gate.py`：把 F36 summary runner、M83-M86 合约和 M83/M85 update-semantics smoke 纳入 S0.6。
- 新增 `experiments/run_v21_01_f36_lowbank_source_channel_summary.py`，只读取指定 `--run-prefix` 的 fresh rows 汇总 route、manifest 和复盘；本轮最终判定使用 `v2101_f36_fix_`，不把初版 M83 branch blocker run 混入最终统计。

### F36-F39 结论 / Insight

- 如果 F36-F39 source projection/gate 有效但 h800/h1600 仍不能同时为正，说明把 target 限制到低阶/低频 bank 并用 B2/B3 precommit gate，也没有解决 early-source observability。
- 如果 h1600/h3200 late-positive 改善但 h800 仍为负，则 blocker 仍是 early target phase mismatch，而不是 reservoir 泄漏。
- 如果 controls 同步变好，按 control-equivalent 处理，不能写 breakthrough。
- 只有 fresh full grouped h800/h1600/h3200/h4800 与 controls attribution 同时通过，才允许继续 promotion；本轮 smoke 只负责决定是否升级 full。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    markers = [
        "\n## 2026-06-04 F36-F39 Low-Bank Source-Channel Writer",
        "\n## 2026-06-04 F36-F37 Low-Bank Source-Channel Writer",
    ]
    idx = -1
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            break
    if idx >= 0:
        path.write_text(text[:idx].rstrip() + "\n" + section, encoding="utf-8")
    else:
        append_text(path, section)


def main() -> int:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    rows = [r for r in read_rows(out_dir / "v21_01_source_retention_matrix.csv") if str(r.get("run_label", "")).startswith(str(args.run_prefix))]
    summary = summarize(rows)
    candidate_summary = [r for r in summary if int_flag(r.get("candidate"))]
    examples = [
        r
        for r in rows
        if r.get("v21_id") in CANDIDATES
        or (r.get("v21_id") in REFERENCE_IDS and int_flag(r.get("late_rebound_flag")))
        or (r.get("v21_id") in CONTROL_IDS and r.get("seed") in {"0", "1"})
    ]
    write_rows(out_dir / ARTIFACTS[0], summary)
    write_rows(out_dir / ARTIFACTS[2], examples[:200])
    smoke_rows = sum(1 for r in rows if not math.isfinite(source(r, 3200)))
    full_rows = len(rows) - smoke_rows
    early_groups = sum(int_flag(r.get("early_chain_group")) for r in candidate_summary)
    h3200_groups = sum(int_flag(r.get("productive_h3200_group")) for r in candidate_summary)
    h4800_groups = sum(int_flag(r.get("productive_h4800_group")) for r in candidate_summary)
    blocked_rows = sum(int(r.get("blocked_rows", 0) or 0) for r in summary)
    full_recommended = int(early_groups > 0 and full_rows == 0 and blocked_rows == 0)

    base = read_json(out_dir / "v21_01_route_decision.json")
    base_route = str(base.get("route", "R0-InProgress"))
    if not rows:
        suffix = "F36LowBankNotRun"
    elif blocked_rows:
        suffix = "F36LowBankBlocked"
    elif h4800_groups:
        suffix = "F36LowBankProductiveH4800CandidateNeedsIndependentConfirmation"
    elif h3200_groups:
        suffix = "F36LowBankProductiveH3200Candidate"
    elif full_recommended:
        suffix = "F36LowBankSmokeNeedsFull"
    elif early_groups:
        suffix = "F36LowBankEarlyChainButNoH3200Closure"
    else:
        suffix = "F36LowBankSmokeNoEarlyChain"
    route = base_route if base_route.endswith(suffix) else f"{base_route}-{suffix}"
    decision = dict(base)
    decision.update(
        {
            "route": route,
            "promotion_allowed": int_flag(base.get("promotion_allowed")),
            "f36_rows": len(rows),
            "f36_blocked_rows": blocked_rows,
            "f36_smoke_rows": smoke_rows,
            "f36_full_rows": full_rows,
            "f36_candidate_early_chain_groups": early_groups,
            "f36_candidate_productive_h3200_groups": h3200_groups,
            "f36_candidate_productive_h4800_groups": h4800_groups,
            "f36_full_escalation_recommended": full_recommended,
            "lowbank_source_channel_completed": int(bool(rows) and blocked_rows == 0),
        }
    )
    write_rows(out_dir / ARTIFACTS[1], [decision])
    write_json(out_dir / "v21_01_route_decision.json", decision)
    write_json(out_dir / "route_decision.json", decision)
    write_json(out_dir / "gate_recompute.json", decision)
    append_unique_manifest(out_dir, ARTIFACTS)
    build_packet(out_dir, [r.get("artifact", "") for r in read_rows(out_dir / "v21_01_required_artifact_manifest.csv")])
    replace_section(V2101_RECAP_DOC, render_section(decision, summary, examples))
    if not args.no_log:
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v21_01_f36_lowbank_source_channel_summary.py --out-dir {out_dir}",
            status="completed",
            note=(
                f"rows={len(rows)} blocked={blocked_rows} early={early_groups} "
                f"productive_h3200={h3200_groups} full_recommended={full_recommended} promotion={decision.get('promotion_allowed')}"
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
