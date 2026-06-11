#!/usr/bin/env python3
"""v22.01 continuation summary for terminal-collapse repair candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.source_chain import RETENTION_EPS, classify_source_chain, source_chain_row  # noqa: E402
from experiments.run_v22_01_common import (  # noqa: E402
    V2201_RECAP_DOC,
    append_exec,
    append_text,
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    mean,
    read_rows,
    sha256_file,
    write_json,
    write_rows,
)


DEFAULT_NEW_IDS = (
    "MLP-F75-dataset-invariant-poprisk-slow-source",
    "MLP-F76-dataset-invariant-readout-consensus-source",
    "MLP-F77-source-conserving-optimizer-only",
)
CONTROL_IDS = ("CTRL-SGD", "CTRL-AdamW", "CTRL-RandomMatchedNorm", "CTRL-NoOpMatchedOverhead")
TAG_CONFIGS = {
    "f75_f77": {
        "continuation": "F75_F76_F77",
        "title": "F75-F77 terminal-collapse repair",
        "changes": [
            "- 新增 `M116-DatasetInvariantPopRiskSlowFU` / `MLP-F75-dataset-invariant-poprisk-slow-source`：用 train split exact PopRisk/SNR 的 sign-consensus 和慢状态作为 DatasetHeterogeneity 分支的 train-only invariant selector。\n",
            "- 新增 `M117-DatasetInvariantReadoutConsensusFU` / `MLP-F76-dataset-invariant-readout-consensus-source`：用 train split readout-gradient consensus，并投掉 corrupt-label 正投影，检验低容量 readout carrier 是否能承接跨数据集不变量。\n",
            "- 新增 `M118-SourceConservingOptimizerOnlyFU` / `MLP-F77-source-conserving-optimizer-only`：沿用 AdamW boundary + dual-timescale source warmup，但 h3200 后只允许 source-preserving projected optimizer；gate 不通过时 hold，不再 SGD fallback，用于验证 OptimizerWashout。\n",
            "- 扩展 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`，使 F75-F77 走既有 fresh source-retention runner 和 v22.01 source-chain reaggregation 口径。\n",
        ],
    },
    "f78_f80": {
        "continuation": "F78_F79_F80",
        "title": "F78-F80 retained-target/source-observability repair",
        "changes": [
            "- 新增 `M119-TerminalSourceConservingRouteFU` / `MLP-F78-terminal-source-conserving-route`：复用 F53 系列能到 h3200 的 AdamW-boundary + dual-timescale source path，但 terminal phase 只在 source-axis / source-projected / split-consensus 三个 source-conserving 候选之间做 train-only precommit；全部 reject 时 hold。\n",
            "- 新增 `M120-TrainLossRiskProfileRouteFU` / `MLP-F79-trainloss-risk-profile-route`：在 F78 route 上加入 train split A/B loss risk-profile；split risk 失衡时只允许 source-axis 保守候选，检验 DatasetHeterogeneity 是否需要 train-only invariant route 而不是 terminal floor。\n",
            "- 新增 `M121-DebtAwareSourceGateFU` / `MLP-F80-debt-aware-source-gate`：在 F78 route 上加入 train-only debt gate；只有当前 train loss debt 存在时才允许 projected/consensus catch-up，否则只保留 source-axis 候选，用于检验 DebtCollapse 假设。\n",
            "- 扩展 `experiments/run_v17_common.py` 的 terminal branch、`dgkan/fu/mechanisms.py` 的 manifest/contract、`experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py` 的 MLP scope，使 F78-F80 走 fresh source-retention runner 和 v22.01 source-chain reaggregation 口径。\n",
        ],
    },
    "f81_f83": {
        "continuation": "F81_F82_F83",
        "title": "F81-F83 ungated h3200 warm retained-target repair",
        "changes": [
            "- 新增 `M122-UngatedWarmTerminalSourceRouteFU` / `MLP-F81-ungated-warm-terminal-source-route`：先关闭早期 train-loss selector，强制复用 AdamW-boundary + fast source warm 到 h3200；terminal phase 再只允许 source-axis/source-projected/split-consensus route 或 hold，用于分离 early selector failure 与 terminal transition failure。\n",
            "- 新增 `M123-UngatedWarmRiskProfileRouteFU` / `MLP-F82-ungated-warm-risk-profile-route`：在 F81 的 h3200 warm 基础上，把 DatasetHeterogeneity fallback 落成 train split A/B risk-profile；risk 失衡时只允许 source-axis 保守候选。\n",
            "- 新增 `M124-UngatedWarmDebtRawBailoutFU` / `MLP-F83-ungated-warm-debt-raw-bailout`：在 F81 的 h3200 warm 基础上，只在 train-only loss debt 存在时允许 positive-only raw terminal bailout，否则走 source-conserving route/hold，用于检验 DebtCollapse 是否需要有限 raw catch-up。\n",
            "- 扩展 `experiments/run_v17_common.py` 的 terminal branch、`dgkan/fu/mechanisms.py` 的 manifest/contract、`experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py` 的 MLP scope，使 F81-F83 走 fresh source-retention runner 和 v22.01 source-chain reaggregation 口径。\n",
        ],
    },
    "f84_f86": {
        "continuation": "F84_F85_F86",
        "title": "F84-F86 terminal-collapse hold without future leakage",
        "changes": [
            "- 新增 `M125-TrainLossH2400CheckpointHoldFU` / `MLP-F84-h2400-terminal-hold-source`：复用 F53-style train-loss gated dual-timescale source path 到 h2400，之后不再提交 source/optimizer 更新，检验高 h2400 source 是否可通过 terminal hold 跨过 h4800。\n",
            "- 新增 `M126-TrainLossH2800CheckpointHoldFU` / `MLP-F85-h2800-terminal-hold-source`：把 hold 起点后移到 h2800，检验 h2400-h3200 的 source decay 是否来自过早冻结还是 terminal drift。\n",
            "- 新增 `M127-TrainLossH2400DebtBailoutFU` / `MLP-F86-h2400-debt-bailout-source`：h2400 后默认 hold，只在 train-only loss debt 明显存在时允许 0.03x positive-only lookahead micro bailout，检验 DebtCollapse 是否需要极小无泄漏 catch-up。\n",
            "- 扩展 `experiments/run_v17_common.py` 的 terminal branch、`dgkan/fu/mechanisms.py` 的 manifest/contract、`experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py` 的 MLP scope，使 F84-F86 走 fresh source-retention runner 和 v22.01 source-chain reaggregation 口径。\n",
        ],
    },
    "f53_f70_f74_replay": {
        "continuation": "F53_F70_F74_REPLAY",
        "title": "F53/F70/F74 fresh terminal-family replay",
        "changes": [
            "- 未新增机制；在当前代码与当前 matched-control attribution 口径下 fresh rerun `MLP-F53` / `MLP-F70` / `MLP-F74`，检验 v22.01 baseline terminal-source family 是否可复现。\n",
            "- replay 只作为校准与故障定位证据；若 fresh replay 不能复现 h3200 continuous chain，后续不能继续基于旧 raw matrix 的 F53 near-miss 设计 confirmation。\n",
        ],
    },
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--tag", default="f75_f77")
    p.add_argument("--new-ids", default="")
    p.add_argument("--title", default="")
    p.add_argument("--record-command", default="")
    p.add_argument("--skip-packet", action="store_true")
    return p


def resolve_new_ids(raw: str) -> tuple[str, ...]:
    ids = tuple(part.strip() for part in str(raw or "").split(",") if part.strip())
    return ids or DEFAULT_NEW_IDS


def artifact_name(tag: str, suffix: str) -> str:
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(tag or "continuation"))
    return f"v22_01_continuation_{safe_tag}_{suffix}"


def horizon_mean(rows: list[dict[str, Any]], step: int) -> float:
    return mean(rows, f"source_h{step}")


def is_control(row: dict[str, Any]) -> int:
    return int(str(row.get("v21_id", "")) in CONTROL_IDS)


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("carrier", "")),
        str(row.get("basis_repair_variant", row.get("variant", ""))),
        str(row.get("v21_id", "")),
    )


def summarize(
    source_dir: Path,
    out_dir: Path,
    *,
    tag: str,
    new_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw = read_rows(source_dir / "v21_01_source_retention_matrix.csv")
    wanted = set(new_ids) | set(CONTROL_IDS)
    enriched: list[dict[str, Any]] = []
    for row in raw:
        if str(row.get("v21_id", "")) not in wanted:
            continue
        item = dict(row)
        item["v22_id"] = item.get("v21_id", "")
        item["control_equivalent"] = is_control(item)
        item.update(source_chain_row(item, prefix="source_h"))
        enriched.append(item)
    write_rows(out_dir / artifact_name(tag, "source_chain.csv"), enriched)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[group_key(row)].append(row)
    summary: list[dict[str, Any]] = []
    for (carrier, variant, v22_id), rows in sorted(groups.items()):
        h = {f"h{step}": horizon_mean(rows, step) for step in (100, 400, 800, 1600, 2400, 3200, 4800, 6400)}
        decision = classify_source_chain(
            h["h100"],
            h["h400"],
            h["h800"],
            h["h1600"],
            h["h3200"],
            h["h4800"],
            control_equivalent=int(v22_id in CONTROL_IDS),
        )
        summary.append(
            {
                "carrier": carrier,
                "variant": variant,
                "v22_id": v22_id,
                "rows": len(rows),
                **h,
                "control_equivalent_group": int(v22_id in CONTROL_IDS),
                "early_source_chain_group": decision.early_source_chain,
                "continuous_retention_group": decision.continuous_retention_chain,
                "continuous_h4800_group": decision.continuous_h4800_chain,
                "productive_h3200_group": decision.continuous_retention_chain,
                "productive_h4800_group": decision.continuous_h4800_chain,
                "late_rebound_group": decision.late_rebound,
                "terminal_collapse_group": decision.terminal_collapse,
                "h1600_retention_ratio": decision.h1600_retention_ratio,
                "h3200_retention_ratio": decision.h3200_retention_ratio,
                "h4800_retention_ratio": decision.h4800_retention_ratio,
                "row_early_chain_count": sum(int_flag(r.get("early_source_chain")) for r in rows),
                "row_continuous_count": sum(int_flag(r.get("continuous_retention_chain")) for r in rows),
                "row_h4800_count": sum(int_flag(r.get("continuous_h4800_chain")) for r in rows),
                "row_h4800_positive_count": sum(finite_float(r.get("source_h4800")) >= RETENTION_EPS for r in rows),
                "source_chain_blocker": decision.blocker,
            }
        )
    write_rows(out_dir / artifact_name(tag, "source_chain_summary.csv"), summary)

    localization = []
    for row in enriched:
        if str(row.get("v21_id", "")) not in new_ids:
            continue
        localization.append(
            {
                "v22_id": row.get("v22_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "source_h100": row.get("source_h100", ""),
                "source_h400": row.get("source_h400", ""),
                "source_h800": row.get("source_h800", ""),
                "source_h1600": row.get("source_h1600", ""),
                "source_h3200": row.get("source_h3200", ""),
                "source_h4800": row.get("source_h4800", ""),
                "early_source_chain": row.get("early_source_chain", ""),
                "continuous_retention_chain": row.get("continuous_retention_chain", ""),
                "continuous_h4800_chain": row.get("continuous_h4800_chain", ""),
                "source_chain_blocker": row.get("source_chain_blocker", ""),
            }
        )
    write_rows(out_dir / artifact_name(tag, "row_localization.csv"), localization)

    candidates = [r for r in summary if str(r.get("v22_id", "")) in new_ids]
    candidate_h4800 = sum(int_flag(r.get("productive_h4800_group")) for r in candidates)
    candidate_h4800_values = [finite_float(r.get("h4800")) for r in candidates]
    candidate_h4800_values = [v for v in candidate_h4800_values if math.isfinite(v)]
    route = {
        "continuation": TAG_CONFIGS.get(tag, {}).get("continuation", str(tag).upper()),
        "source_dir": str(source_dir),
        "rows": len(enriched),
        "candidate_groups": len(candidates),
        "candidate_early_chain": sum(int_flag(r.get("early_source_chain_group")) for r in candidates),
        "candidate_continuous_h3200": sum(int_flag(r.get("continuous_retention_group")) for r in candidates),
        "candidate_h4800": candidate_h4800,
        "terminal_collapse_groups": sum(int_flag(r.get("terminal_collapse_group")) for r in candidates),
        "late_rebound_groups": sum(int_flag(r.get("late_rebound_group")) for r in candidates),
        "best_candidate_h4800": max(candidate_h4800_values) if candidate_h4800_values else "",
        "promotion_allowed": 0,
        "decision": (
            "ContinuationH4800CandidateNeedsIndependentConfirmation"
            if candidate_h4800
            else "ContinuationNoH4800SourceChain"
        ),
    }
    write_json(out_dir / artifact_name(tag, "route.json"), route)
    write_rows(out_dir / artifact_name(tag, "route.csv"), [route])
    return enriched, summary, route


def append_recap(
    out_dir: Path,
    source_dir: Path,
    summary: list[dict[str, Any]],
    route: dict[str, Any],
    *,
    tag: str,
    new_ids: tuple[str, ...],
    title: str = "",
) -> None:
    candidates = [r for r in summary if str(r.get("v22_id", "")) in new_ids]
    controls = [r for r in summary if str(r.get("v22_id", "")) in CONTROL_IDS]
    cfg = TAG_CONFIGS.get(tag, {})
    recap_title = title or str(cfg.get("title") or f"{tag} continuation")
    fields = [
        "v22_id",
        "rows",
        "h100",
        "h400",
        "h800",
        "h1600",
        "h3200",
        "h4800",
        "early_source_chain_group",
        "continuous_retention_group",
        "productive_h4800_group",
        "terminal_collapse_group",
        "source_chain_blocker",
    ]
    text = [
        f"\n## v22.01 continuation: {recap_title}\n\n",
        f"- source_dir: `{source_dir}`\n",
        f"- decision: `{route.get('decision')}`\n",
        f"- promotion_allowed: {route.get('promotion_allowed')}\n",
        f"- candidate groups early / continuous_h3200 / h4800: {route.get('candidate_early_chain')} / {route.get('candidate_continuous_h3200')} / {route.get('candidate_h4800')}\n",
        f"- terminal collapse groups: {route.get('terminal_collapse_groups')}\n",
        f"- best candidate h4800: {route.get('best_candidate_h4800')}\n\n",
        "### Continuation candidate evidence\n\n",
        md_table(candidates, fields, max_rows=20),
        "\n### Continuation controls\n\n",
        md_table(controls, fields, max_rows=10),
        "\n### 本轮修改记录\n\n",
    ]
    changes = list(cfg.get("changes", []))
    if not changes:
        changes = [
            f"- 新增/扩展 continuation `{route.get('continuation')}` 的机制与 runner scope；只读取 fresh matrix 聚合，不生成或篡改实验数据。\n"
        ]
    text.extend(changes)
    text.append(
        "- 扩展 `experiments/run_v22_01_continue_terminal_repair.py`：支持按 continuation tag / ids 读取落盘 fresh matrix，生成 source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。\n\n"
    )
    text.append("### 分析 / Insight / 结论\n\n")
    if int(route.get("candidate_h4800", 0) or 0) > 0:
        text.append(
            f"- {route.get('continuation')} 中至少一个 candidate 形成 grouped h4800 retained source chain；但这只是 continuation candidate，仍需 independent offset/control attribution，不能直接 promotion。\n"
        )
    else:
        text.append(
            f"- {route.get('continuation')} 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。\n"
        )
    if int(route.get("candidate_continuous_h3200", 0) or 0) > 0 and int(route.get("candidate_h4800", 0) or 0) == 0:
        text.append(
            "- 若本轮仍出现 h3200 continuous 但 h4800 失败，说明 DatasetHeterogeneity 的 train-only invariant selector 和 source-conserving optimizer 仍未解决 terminal collapse；下一步应进一步重定义 retained target，而不是继续 terminal floor/carrier 小扫。\n"
        )
    if int(route.get("candidate_continuous_h3200", 0) or 0) == 0:
        text.append(
            "- 若本轮连 h3200 continuous 都没有，说明当前 invariant/source-conserving 改法未复现 v22 中 F53/F70 的 h3200 source 保存能力；后续优先分析 row localization，而不是升级 confirmation。\n"
        )
    text.append(
        "- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。\n"
    )
    append_text(V2201_RECAP_DOC, "".join(text))


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    new_ids = resolve_new_ids(args.new_ids)
    enriched, summary, route = summarize(source_dir, out_dir, tag=args.tag, new_ids=new_ids)
    append_recap(out_dir, source_dir, summary, route, tag=args.tag, new_ids=new_ids, title=args.title)
    if args.record_command:
        append_exec(out_dir, args.record_command, status="completed", note=f"continuation summary rows={len(enriched)} decision={route.get('decision')}")
    if not args.skip_packet:
        zip_path, bundle = build_packet(out_dir)
        route["code_review_packet_zip"] = str(zip_path)
        route["code_review_packet_sha256"] = sha256_file(zip_path)
        route["results_bundle_zip"] = str(bundle)
        route["results_bundle_sha256"] = sha256_file(bundle)
        write_json(out_dir / artifact_name(args.tag, "route.json"), route)
        write_rows(out_dir / artifact_name(args.tag, "route.csv"), [route])


if __name__ == "__main__":
    main()
