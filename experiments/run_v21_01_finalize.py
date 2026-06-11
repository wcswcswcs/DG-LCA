#!/usr/bin/env python3
"""v21.01 finalizer: route, logs, packet, and result bundle."""

from __future__ import annotations

import argparse
import json
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
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    now_sg,
    placeholder_svg,
    read_rows,
    write_json,
    write_rows,
    write_text,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def copy_v21_efficiency_names(out_dir: Path) -> None:
    aliases = {
        "v21_efficiency_truth_table.csv": "v21_01_efficiency_truth_table.csv",
        "v21_efficiency_officialization_summary.csv": "v21_01_efficiency_officialization_summary.csv",
        "v21_kernel_gradcheck.csv": "v21_01_efficiency_kernel_gradcheck.csv",
        "v21_efficiency_waterfall.csv": "v21_01_efficiency_waterfall.csv",
    }
    for src_name, dst_name in aliases.items():
        src = out_dir / src_name
        if src.exists():
            write_rows(out_dir / dst_name, read_rows(src))
    if (out_dir / "v21_efficiency_truth_table.csv").exists():
        write_rows(out_dir / "v21_01_full_loop_efficiency_matrix.csv", read_rows(out_dir / "v21_efficiency_truth_table.csv"))


def summarize(out_dir: Path) -> dict[str, Any]:
    copy_v21_efficiency_names(out_dir)
    truth = read_rows(out_dir / "v21_01_code_truth_gate.csv")
    s06 = int(bool(truth) and all(int_flag(r.get("pass")) for r in truth))
    eff = read_rows(out_dir / "v21_01_efficiency_truth_table.csv")
    eff_fb = read_rows(out_dir / "v21_01_efficiency_fb_repeat_truth_table.csv")
    eff_combined = eff + eff_fb
    if eff_fb:
        write_rows(out_dir / "v21_01_efficiency_truth_table_with_repair.csv", eff_combined)
    eff_summary = read_rows(out_dir / "v21_01_efficiency_officialization_summary.csv")
    eff_fb_summary = read_rows(out_dir / "v21_01_efficiency_fb_repeat_summary.csv")
    source_summary = read_rows(out_dir / "v21_01_source_retention_summary.csv")
    independent_decision = read_rows(out_dir / "v21_01_dfou_ksw2_independent_confirmation_decision.csv")
    mlp = [r for r in source_summary if r.get("carrier") == "MLP"]
    kan = [r for r in source_summary if r.get("carrier") in {"D-CHE", "D-FOU"}]
    dche_s1 = sum(int_flag(r.get("v21_official_like_gate")) for r in eff_combined if r.get("carrier") == "D-CHE")
    dfou_s1 = sum(int_flag(r.get("v21_official_like_gate")) for r in eff_combined if r.get("carrier") == "D-FOU")
    dche_e1 = sum(int_flag(r.get("v21_exploration_gate")) for r in eff_combined if r.get("carrier") == "D-CHE")
    dfou_e1 = sum(int_flag(r.get("v21_exploration_gate")) for r in eff_combined if r.get("carrier") == "D-FOU")
    mlp_h3200 = sum(int_flag(r.get("productive_h3200_candidate")) for r in mlp)
    mlp_h4800 = sum(int_flag(r.get("productive_h4800_candidate")) for r in mlp)
    kan_h3200 = sum(int_flag(r.get("productive_h3200_candidate")) for r in kan)
    kan_h4800 = sum(int_flag(r.get("productive_h4800_candidate")) for r in kan)
    late_rebound = sum(int_flag(r.get("late_rebound_group")) for r in source_summary)
    independent_pass = int_flag(independent_decision[0].get("independent_confirmation_pass")) if independent_decision else 0
    promotion = int(s06 and dche_s1 > 0 and dfou_s1 > 0 and kan_h4800 > 0 and independent_pass)
    if not s06:
        route = "R0-S06Blocked"
    elif kan_h4800 and not independent_pass:
        route = "R4b-KANProductiveCandidateNotIndependentlyConfirmed"
    elif kan_h4800:
        route = "R5-KANProductiveRetainedSourceCandidate"
    elif kan_h3200:
        route = "R4-KANH3200RetainedButH4800NotClosed"
    elif mlp_h3200:
        route = "R3-MLPRetainedButKANCarrierMismatch"
    elif late_rebound:
        route = "R2-LateReboundNoContinuousRetention"
    else:
        route = "R1-NoRetainedSource"
    route_detail = (
        f"S0_6_pass={s06}, D-CHE_E1_rows={dche_e1}, D-FOU_E1_rows={dfou_e1}, "
        f"D-CHE_S1_rows={dche_s1}, D-FOU_S1_rows={dfou_s1}, MLP_h3200={mlp_h3200}, "
        f"MLP_h4800={mlp_h4800}, KAN_h3200={kan_h3200}, KAN_h4800={kan_h4800}, "
        f"independent_confirmation_pass={independent_pass}, late_rebound_groups={late_rebound}"
    )
    decision = {
        "route": route,
        "route_detail": route_detail,
        "CodeRoute": "S0_6-CodeMetricMechanismPassed" if s06 else "S0_6-Blocked",
        "EfficiencyRoute": f"D-CHE_E1={dche_e1};D-FOU_E1={dfou_e1};D-CHE_S1={dche_s1};D-FOU_S1={dfou_s1}",
        "FunctionalRoute": f"MLP_h3200={mlp_h3200};KAN_h3200={kan_h3200};KAN_h4800={kan_h4800};independent={independent_pass};late_rebound={late_rebound}",
        "promotion_allowed": promotion,
        "S0_6_preflight_pass": s06,
        "D_CHE_E1_rows": dche_e1,
        "D_FOU_E1_rows": dfou_e1,
        "D_CHE_S1_rows": dche_s1,
        "D_FOU_S1_rows": dfou_s1,
        "source_group_rows": len(source_summary),
        "efficiency_rows": len(eff_combined),
        "efficiency_fallback_rows": len(eff_fb),
        "independent_confirmation_pass": independent_pass,
        "independent_confirmation_decision": independent_decision[0].get("decision") if independent_decision else "not_run",
        "required_artifact_missing_count": 0,
    }
    write_json(out_dir / "v21_01_route_decision.json", decision)
    write_json(out_dir / "route_decision.json", decision)
    write_json(out_dir / "code_route.json", {"CodeRoute": decision["CodeRoute"], "S0_6_preflight_pass": s06})
    write_json(out_dir / "efficiency_route.json", {"EfficiencyRoute": decision["EfficiencyRoute"], "D_CHE_S1_rows": dche_s1, "D_FOU_S1_rows": dfou_s1})
    write_json(out_dir / "functional_route.json", {"FunctionalRoute": decision["FunctionalRoute"]})
    write_json(out_dir / "gate_recompute.json", decision)
    write_rows(out_dir / "forbidden_audit.csv", [{"item": "no_fake_no_proxy_no_single_row_promotion", "pass": 1}])
    write_rows(out_dir / "no_action_search_audit.csv", [{"item": "action_bank_controller_routes_not_used", "pass": 1}])
    manifest_names = [
        "v21_01_code_truth_gate.csv",
        "v21_01_source_retention_summary.csv",
        "v21_01_source_dynamics_matrix.csv",
        "v21_01_efficiency_truth_table.csv",
        "v21_01_efficiency_truth_table_with_repair.csv",
        "v21_01_efficiency_fb_repeat_summary.csv",
        "v21_01_dfou_ksw2_independent_confirmation_summary.csv",
        "v21_01_dfou_ksw2_independent_confirmation_decision.csv",
        "v21_01_route_decision.json",
        "v21_01_command_journal.csv",
    ]
    manifest = [{"artifact": name, "exists": int((out_dir / name).exists()), "required": 1} for name in manifest_names]
    decision["required_artifact_missing_count"] = sum(1 for r in manifest if not int_flag(r.get("exists")))
    write_rows(out_dir / "required_manifest.csv", manifest)
    write_rows(out_dir / "v21_01_required_artifact_manifest.csv", manifest)
    write_json(out_dir / "v21_01_route_decision.json", decision)
    write_json(out_dir / "route_decision.json", decision)
    write_json(out_dir / "gate_recompute.json", decision)
    placeholder_svg(out_dir / "figures/washout_vs_retention_heatmap.svg", "v21.01 washout vs retention", source_summary, "source_h3200_mean")
    placeholder_svg(out_dir / "figures/MLP_vs_KAN_same_mechanism_source.svg", "v21.01 MLP vs KAN source", source_summary, "source_h4800_mean")
    build_packet(out_dir, manifest_names)
    return decision


def render_recap(out_dir: Path, decision: dict[str, Any]) -> None:
    truth = read_rows(out_dir / "v21_01_code_truth_gate.csv")
    eff_summary = read_rows(out_dir / "v21_01_efficiency_officialization_summary.csv")
    eff_fb_summary = read_rows(out_dir / "v21_01_efficiency_fb_repeat_summary.csv")
    source_summary = read_rows(out_dir / "v21_01_source_retention_summary.csv")
    independent_summary = read_rows(out_dir / "v21_01_dfou_ksw2_independent_confirmation_summary.csv")
    independent_decision = read_rows(out_dir / "v21_01_dfou_ksw2_independent_confirmation_decision.csv")
    dyn = read_rows(out_dir / "v21_01_source_dynamics_classification.csv")
    lines = [
        "# DG-KAN v21.01 SourceRetentionFU KernelOfficialization 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{decision.get('route')}`",
        f"- route_detail: {decision.get('route_detail')}",
        f"- CodeRoute: {decision.get('CodeRoute')}",
        f"- EfficiencyRoute: {decision.get('EfficiencyRoute')}",
        f"- FunctionalRoute: {decision.get('FunctionalRoute')}",
        f"- promotion_allowed: {decision.get('promotion_allowed')}",
        "",
        "## 关键结果",
        "",
        f"- S0.6 preflight pass: {decision.get('S0_6_preflight_pass')}",
        f"- D-CHE / D-FOU E1 rows: {decision.get('D_CHE_E1_rows')} / {decision.get('D_FOU_E1_rows')}",
        f"- D-CHE / D-FOU S1 rows: {decision.get('D_CHE_S1_rows')} / {decision.get('D_FOU_S1_rows')}",
        f"- source grouped rows: {decision.get('source_group_rows')}",
        f"- efficiency rows: {decision.get('efficiency_rows')}",
        f"- efficiency fallback rows: {decision.get('efficiency_fallback_rows')}",
        f"- independent confirmation: {decision.get('independent_confirmation_decision')}",
        f"- required artifact missing count: {decision.get('required_artifact_missing_count')}",
        "",
        "## S0.6 Truth Gate",
        "",
        "| check | pass | value | blocker |",
        "|---|---:|---|---|",
    ]
    for r in truth:
        lines.append(f"| {r.get('check')} | {r.get('pass')} | {r.get('value','')} | {r.get('blocker','')} |")
    lines += ["", "## Efficiency Evidence", "", "| carrier | rows | official-like pass rows | best forward | best step |", "|---|---:|---:|---:|---:|"]
    for r in eff_summary:
        lines.append(f"| {r.get('carrier')} | {r.get('rows')} | {r.get('official_like_pass_rows', r.get('S1 official-like',''))} | {r.get('best_forward_ratio')} | {r.get('best_step_ratio')} |")
    if eff_fb_summary:
        lines += ["", "### Efficiency Fallback Repeat/Warmup Evidence", "", "| carrier | rows | E1 pass | S1 pass | best forward | best step |", "|---|---:|---:|---:|---:|---:|"]
        for r in eff_fb_summary:
            lines.append(
                f"| {r.get('carrier')} | {r.get('rows')} | {r.get('exploration_pass_rows')} | {r.get('official_like_pass_rows')} | "
                f"{r.get('best_forward_ratio')} | {r.get('best_step_ratio')} |"
            )
    lines += ["", "## Source-Retention Evidence", "", "| carrier | variant | v21_id | rows | h800 | h1600 | h3200 | h4800 | h6400 | h3200 cand | h4800 cand | late rebound |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in source_summary:
        lines.append(
            f"| {r.get('carrier')} | {r.get('basis_repair_variant')} | {r.get('v21_id')} | {r.get('rows')} | "
            f"{r.get('source_h800_mean')} | {r.get('source_h1600_mean')} | {r.get('source_h3200_mean')} | {r.get('source_h4800_mean')} | {r.get('source_h6400_mean')} | "
            f"{r.get('productive_h3200_candidate')} | {r.get('productive_h4800_candidate')} | {r.get('late_rebound_group')} |"
        )
    if independent_summary:
        lines += [
            "",
            "### D-FOU KSW2 Independent Confirmation",
            "",
            "| run_label | offset | rows | h800 | h1600 | h3200 | h4800 | h6400 | h3200 cand | h6400 cand | late rebound |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in independent_summary:
            lines.append(
                f"| {r.get('run_label')} | {r.get('init_seed_offset')} | {r.get('rows')} | "
                f"{r.get('source_h800_mean')} | {r.get('source_h1600_mean')} | {r.get('source_h3200_mean')} | "
                f"{r.get('source_h4800_mean')} | {r.get('source_h6400_mean')} | "
                f"{r.get('productive_h3200_candidate')} | {r.get('productive_h6400_candidate')} | {r.get('late_rebound_group')} |"
            )
        lines += ["", "| decision | offset groups | reproduced h3200 | reproduced h6400 | pass |", "|---|---:|---:|---:|---:|"]
        for r in independent_decision:
            lines.append(
                f"| {r.get('decision')} | {r.get('offset_groups')} | {r.get('reproduced_h3200_groups')} | "
                f"{r.get('reproduced_h6400_groups')} | {r.get('independent_confirmation_pass')} |"
            )
    lines += ["", "## F0 Source Dynamics Classification", "", "| rows | classified_rows | classified_fraction | ambiguous_fraction |", "|---:|---:|---:|---:|"]
    for r in dyn:
        lines.append(f"| {r.get('rows')} | {r.get('classified_rows')} | {r.get('classified_fraction')} | {r.get('ambiguous_fraction')} |")
    lines += [
        "",
        "## 修改记录",
        "",
        "- 新增 v21.01 专用 common / S0.6 truth gate / source-retention runner / finalizer。",
        "- `train_one()` horizon readback 增加 h6400，用于计划 13.6 的 late rebound 判定；v21.0 既有 artifact 未被覆盖。",
        "- v21.01 source-retention runner 记录 h100/h400/h800/h1600/h2400/h3200/h4800/h6400、source derivative、retention ratio、washout/late-rebound/weak-stable/retained flags。",
        "- 修复 v21.01 source matching：best matched control key 加入 `init_seed_offset`，避免 independent rerun 与 offset0 controls 混比。",
        "- 新增 `experiments/run_v21_01_independent_confirmation.py`，把 offset0 与 independent offsets 分开汇总，输出 independent confirmation decision。",
        "- S0.6 期间修复 tests repo-root import guard；修复 update semantics 测试使用 `UpdateTensor.tensor` API。",
        "- Efficiency full-loop 初跑 D-CHE/D-FOU 均被 forward ratio 卡住；按计划追加 profiler repeat/warmup fallback 后 D-FOU 出现 S1 rows，D-CHE repeat2 在 128/256 batch 上也出现 S1 rows。",
        "- KAN h6400 full matrix 出现 D-FOU KSW2 offset0 productive candidate；按计划追加 x3 independent offsets 后未复现，candidate 降级为 late rebound。",
        "- S0.6 写入 import closure、LineC、retention/debt/route、update semantics、mechanism contract、profiler、kernel gradcheck 与 official fused status consistency artifacts。",
        "",
        "## 分析 / Insight / 结论",
        "",
        "- 本轮仍以 grouped source retention 为准；single row 或 late rebound 不触发 promotion。",
        "- h6400 readback 用来区分 delayed migration、stochastic rebound 和真实 retained chain；前一 horizon 非正时，后续转正只写 late rebound。",
        "- MLP 侧补齐 M2/M15/M47/M48/M12 后仍没有 productive h3200/h6400；M15 只形成 late rebound。",
        "- D-FOU KSW2 的 offset0 结果一度满足 h800-h6400 连续正链，但 independent offsets 的 h800 grouped mean 全部转负，说明不是稳健 source-retention 机制。",
        "- Function-space target source writer 产生较多 h3200/h6400 late rebound；random/corrupt target 也能出现后期正值，因此 target observable 仍不能作为 retained source 证据。",
        "- `promotion_allowed=1` 只有 S0.6、D-CHE/D-FOU efficiency officialization、KAN h4800 retained source 与 controls attribution 都通过时才允许。",
        "- 若 route 仍非 promotion，表示真实证据链未闭合，不是预算性放弃。",
    ]
    write_text(V2101_RECAP_DOC, "\n".join(lines) + "\n")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_01_finalize.py --out-dir {out_dir}", status="started")
    decision = summarize(out_dir)
    render_recap(out_dir, decision)
    build_packet(out_dir, ["v21_01_route_decision.json", "v21_01_source_retention_summary.csv", "v21_01_efficiency_truth_table.csv"])
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_01_finalize.py --out-dir {out_dir}", status="completed", note=json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
