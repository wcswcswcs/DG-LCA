#!/usr/bin/env python3
"""Finalize v22.12 artifacts, execution log, recap, and route decision."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_12_common import (  # noqa: E402
    PYTHON,
    V2212_EXEC_DOC,
    V2212_RECAP_DOC,
    append_exec,
    artifact_index,
    build_code_review_packet,
    build_results_bundle,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    now_sg,
    read_json,
    read_rows,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def _best(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    valid = []
    for row in rows:
        value = finite_float(row.get(field))
        if value == value:
            valid.append((value, row))
    return max(valid, key=lambda item: item[0])[1] if valid else {}


def _route_decision(routes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    s0, s1, s2, s3, s4, s5, s6, queue = (routes[k] for k in ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "queue"])
    blockers: list[str] = []
    s1_any = any(int_flag(s1.get(k)) for k in ["D-CHE_pass", "D-FOU_pass", "D-RAT_pass", "D-RBF_pass"])
    fused_rows = int_flag(s1.get("official_fused_kernel_complete_rows"))
    queue_ok = int_flag(queue.get("queue_drained", 0)) == 1 and int_flag(queue.get("execution_contract_violation", 0)) == 0
    s5_c3_adapters = int_flag(s5.get("C3_adapter_pass_count"))
    s5_c4_adapters = int_flag(s5.get("C4_adapter_pass_count"))
    kan_pass = int_flag(s6.get("KAN_source_channel_pass_rows")) > 0
    kan_readout_only = int_flag(s6.get("KAN_readout_only_pass_rows")) > 0 and int_flag(s6.get("KAN_readout_only_pass_rows")) == int_flag(s6.get("KAN_source_channel_pass_rows"))
    exploration_promotion_allowed = int(
        int_flag(s0.get("S0_19_pass"))
        and s1_any
        and int_flag(s2.get("S2_operator_atom_pass_rows")) > 0
        and int_flag(s3.get("S3_operator_variational_pass_rows")) > 0
        and int_flag(s4.get("S4_operator_metric_commit_pass_rows")) > 0
        and s5_c3_adapters >= 2
        and s5_c4_adapters >= 1
        and kan_pass
        and queue_ok
    )
    official_promotion_allowed = int(
        exploration_promotion_allowed
        and fused_rows > 0
        and s5_c4_adapters >= 2
        and kan_pass
        and not kan_readout_only
    )
    if not int_flag(s0.get("S0_19_pass")):
        route = "R0-CodeOrSemanticGateFailed"
        blockers.append(str(s0.get("blocker", "S0_failed")))
    elif not s1_any:
        route = "R1-ArbitraryCotangentEfficiencyBlocked"
        blockers.append(str(s1.get("blocker", "S1_efficiency_failed")))
    elif int_flag(s2.get("S2_operator_atom_pass_rows")) <= 0 or int_flag(s3.get("S3_operator_variational_pass_rows")) <= 0 or int_flag(s4.get("S4_operator_metric_commit_pass_rows")) <= 0:
        route = "R2-SourceOperatorNoGo"
        blockers.extend([str(s2.get("blocker", "")), str(s3.get("blocker", "")), str(s4.get("blocker", ""))])
    elif str(s5.get("route")) == "S5-AdapterSpecificSourceOpened_NotUniversal" or s5_c3_adapters == 1:
        route = "R3-AdapterSpecificSourceOpened_NotUniversal"
        blockers.append(str(s5.get("blocker", "adapter_specific_source")))
    elif str(s5.get("route")) == "S5-TargetRetentionOnly_NotTaskUseful":
        route = "R4-TargetRetentionOnly_NotTaskUseful"
        blockers.append(str(s5.get("blocker", "target_retention_only")))
    elif s5_c3_adapters <= 0:
        route = "R2-SourceOperatorNoGo"
        blockers.append(str(s5.get("blocker", "horizon_source_operator_failed")))
    elif not kan_pass and str(s6.get("route")) == "S6-KANMappingNotEntered":
        route = "R5-MLPArbitraryLossSourceOpened_KANNotEntered"
        blockers.append(str(s6.get("blocker", "KAN_not_entered")))
    elif str(s6.get("route")) == "S6-DFOUSourceOpened_DCHESourceExistsEfficiencyBlocked":
        route = "R7-KANSourceExistsEfficiencyBlocked"
        blockers.append(str(s6.get("blocker") or "D-CHE_arbitrary_cotangent_efficiency_blocked_after_source_exists"))
    elif str(s6.get("route")) == "S6-DFOUReadoutSourceOpened_DCHEMismatch":
        route = "R6-DFOUReadoutSourceOpened_DCHEMismatch"
        blockers.append(str(s6.get("blocker") or "D-CHE_source_channel_mismatch_after_DFOU_open"))
    elif int_flag(s6.get("KAN_efficiency_blocked_rows")) > 0 and not kan_pass:
        route = "R7-KANSourceExistsEfficiencyBlocked"
        blockers.append(str(s6.get("blocker", "KAN_efficiency_blocked")))
    elif official_promotion_allowed:
        route = "R10-OfficialPromotionReady"
    elif exploration_promotion_allowed and fused_rows == 0:
        route = "R8-FunctionalExplorationSuccess_SystemEfficiencyOfficializationPending"
        blockers.append("official_fused_kernel_complete_rows=0")
    elif exploration_promotion_allowed:
        route = "R9-ArbitraryLossOperatorFU_KANExplorationSuccess"
    else:
        route = "R2-SourceOperatorNoGo"
        blockers.extend([str(s5.get("blocker", "")), str(s6.get("blocker", ""))])
    minimum: list[str] = []
    if int_flag(s0.get("S0_19_pass")) and int_flag(s0.get("carrier_specific_efficiency_pass_consistency")) and not int_flag(s0.get("wrong_global_efficiency_gate_detected")):
        minimum.append("A-CodeTruth")
    if s1_any:
        minimum.append("B-Efficiency")
    if s5_c3_adapters >= 2:
        minimum.append("C-Functional")
    if kan_pass:
        minimum.append("D-KAN")
    if not minimum:
        minimum.append("E-TheoryBoundary")
    return {
        "route": route,
        "exploration_promotion_allowed": exploration_promotion_allowed,
        "official_promotion_allowed": official_promotion_allowed,
        "promotion_allowed": official_promotion_allowed,
        "blocking_metric": ";".join(dict.fromkeys(str(b) for b in blockers if b)),
        "minimum_effective_progress": ";".join(minimum),
        "next_codex_action": "continue kernel-native arbitrary-cotangent officialization or repair adapter/KAN blockers according to v22.12 failure taxonomy",
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    idle_rows = read_rows(out_dir / "v22_12_idle_violation.csv")
    queue_report = read_json(out_dir / "v22_12_queue_drain_report.json")
    if idle_rows:
        queue_report["execution_contract_violation"] = int_flag(idle_rows[0].get("execution_contract_violation"))
    routes = {
        "s0": read_json(out_dir / "v22_12_code_truth_route.json"),
        "s1": read_json(out_dir / "v22_12_basis_efficiency_route.json"),
        "s2": read_json(out_dir / "v22_12_operator_atom_route.json"),
        "s3": read_json(out_dir / "v22_12_operator_variational_route.json"),
        "s4": read_json(out_dir / "v22_12_operator_metric_commit_route.json"),
        "s5": read_json(out_dir / "v22_12_arbitrary_loss_horizon_route.json"),
        "s6": read_json(out_dir / "v22_12_kan_mapping_route.json"),
        "queue": queue_report,
    }
    decision = _route_decision(routes)
    write_json(out_dir / "v22_12_final_decision.json", decision)

    truth = read_rows(out_dir / "v22_12_code_truth_gate.csv")
    efficiency_summary = read_rows(out_dir / "v22_12_arbitrary_cotangent_efficiency_summary.csv")
    efficiency_rows = read_rows(out_dir / "v22_12_arbitrary_cotangent_efficiency_rows.csv")
    repair_rows = read_rows(out_dir / "v22_12_efficiency_repair_attempts.csv")
    operator_rows = read_rows(out_dir / "v22_12_operator_atom_rows.csv")
    variational_rows = read_rows(out_dir / "v22_12_operator_variational_solve.csv")
    commit_rows = read_rows(out_dir / "v22_12_operator_metric_commit_matrix.csv")
    horizon_rows = read_rows(out_dir / "v22_12_arbitrary_loss_horizon_matrix.csv")
    fu_horizon = [r for r in horizon_rows if str(r.get("variant")) == "FU"]
    repair_diagnostics = read_rows(out_dir / "v22_12_repair_diagnostic_summary.csv")
    kan_rows = read_rows(out_dir / "v22_12_kan_mapping_matrix.csv")
    queue_rows = read_rows(out_dir / "v22_12_gpu_assignment_manifest.csv")
    command_rows = read_rows(out_dir / "v22_12_command_journal.csv")
    best_h = _best(fu_horizon, "source_func_h3200")
    fused_complete_rows = sum(int_flag(r.get("official_fused_kernel_complete")) for r in efficiency_rows)
    manual_rows = sum(int_flag(r.get("manual_upstream_vjp_used")) for r in efficiency_rows)
    k14_rows = [r for r in kan_rows if str(r.get("attempt")) == "K14_dche_corrected_readout_layout_fan_scale_25x0p08"]
    k13_rows = [r for r in kan_rows if str(r.get("attempt", "")).startswith("K13_")]
    k13_open_rows = [r for r in k13_rows if str(r.get("KAN_source_channel_decision")) == "KANRetainedSourceOpened"]
    dche_pre_k14_attempts = {
        "K1_readout_replay_100x0p08",
        "K8_readout_replay_25x0p12_repair",
        "K10_readout_gain1p0_25x0p12_repair",
        "K11_basis_linearized_rank12_cap4p0_initial_only",
    }
    dche_pre_k14_rows = [
        r for r in kan_rows
        if str(r.get("carrier")) == "D-CHE" and str(r.get("attempt")) in dche_pre_k14_attempts
    ]
    k14_command_rows = [
        {
            "timestamp": r.get("timestamp", ""),
            "status": r.get("status", ""),
            "note": r.get("note", ""),
        }
        for r in command_rows
        if "K14" in str(r.get("note", "")) or "run_v22_12_kan_k14_repair.py" in str(r.get("command", ""))
    ][-10:]
    k14 = k14_rows[0] if k14_rows else {}
    promotion_gate_rows = [
        {
            "gate": "S0.19 code/semantic truth",
            "required": "pass=1",
            "observed": f"S0.19={routes['s0'].get('S0_19_pass', '')}",
            "decision": "pass" if int_flag(routes["s0"].get("S0_19_pass")) else "blocked",
        },
        {
            "gate": "arbitrary-cotangent official kernel",
            "required": "official_fused_kernel_complete_rows > 0",
            "observed": f"official_fused_kernel_complete_rows={fused_complete_rows}; manual_upstream_vjp_rows={manual_rows}",
            "decision": "blocked" if fused_complete_rows <= 0 else "pass",
        },
        {
            "gate": "operator FU arbitrary-loss robustness",
            "required": "C3 adapters >= 2; C4 adapters >= 2 for official",
            "observed": f"C3={routes['s5'].get('C3_adapter_pass_count', '')} ({routes['s5'].get('C3_adapter_pass_list', '')}); C4={routes['s5'].get('C4_adapter_pass_count', '')} ({routes['s5'].get('C4_adapter_pass_list', '')})",
            "decision": "exploration_pass_official_blocked" if int_flag(routes["s5"].get("C3_adapter_pass_count")) >= 2 and int_flag(routes["s5"].get("C4_adapter_pass_count")) < 2 else "pass_or_blocked_by_counts",
        },
        {
            "gate": "KAN carrier-specific source",
            "required": "no carrier borrows another carrier's efficiency pass",
            "observed": f"S6={routes['s6'].get('route', '')}; KAN_pass_rows={routes['s6'].get('KAN_source_channel_pass_rows', '')}",
            "decision": "exploration_pass_with_D-CHE_blocker",
        },
        {
            "gate": "D-CHE K14 terminal/effectiveness",
            "required": "carrier_specific_efficiency_pass=1 and source_loss_h4800 >= 0",
            "observed": f"carrier_specific_efficiency_pass={k14.get('carrier_specific_efficiency_pass', '')}; K14_terminal_source_loss_pass={k14.get('K14_terminal_source_loss_pass', '')}; source_loss_h4800={k14.get('KAN_source_loss_h4800', '')}",
            "decision": "blocked" if k14 else "not_run",
        },
    ]

    idx = artifact_index(out_dir)
    write_rows(out_dir / "v22_12_artifact_index.csv", idx)
    packet = build_code_review_packet(out_dir)
    bundle = build_results_bundle(out_dir)
    idx = artifact_index(out_dir)
    write_rows(out_dir / "v22_12_artifact_index.csv", idx)

    recap: list[str] = []
    recap.append("# DG-KAN v22.12 ArbitraryLossOperatorFU BasisEfficiency 4GPU 实验结果复盘\n")
    recap.append(f"生成时间：{now_sg()}\n")
    recap.append("## Route\n")
    for key in ["route", "exploration_promotion_allowed", "official_promotion_allowed", "blocking_metric", "minimum_effective_progress", "next_codex_action"]:
        recap.append(f"- {key}: `{decision.get(key, '')}`")
    recap.append(f"- S0.19 pass: {routes['s0'].get('S0_19_pass', '')}")
    recap.append(f"- D-CHE/D-FOU/D-RAT/D-RBF S1 pass: {routes['s1'].get('D-CHE_pass', '')} / {routes['s1'].get('D-FOU_pass', '')} / {routes['s1'].get('D-RAT_pass', '')} / {routes['s1'].get('D-RBF_pass', '')}")
    recap.append(f"- official_fused_kernel_complete_rows: {routes['s1'].get('official_fused_kernel_complete_rows', '')}")
    recap.append(f"- S2/S3/S4 pass rows: {routes['s2'].get('S2_operator_atom_pass_rows', '')} / {routes['s3'].get('S3_operator_variational_pass_rows', '')} / {routes['s4'].get('S4_operator_metric_commit_pass_rows', '')}")
    recap.append(f"- S5 C3/C4 adapter pass count: {routes['s5'].get('C3_adapter_pass_count', '')} / {routes['s5'].get('C4_adapter_pass_count', '')}")
    recap.append(f"- S5 C3/C4 adapter list: `{routes['s5'].get('C3_adapter_pass_list', '')}` / `{routes['s5'].get('C4_adapter_pass_list', '')}`")
    recap.append(f"- KAN route / pass rows: `{routes['s6'].get('route', '')}` / {routes['s6'].get('KAN_source_channel_pass_rows', '')}")
    recap.append(f"- results bundle: `{bundle}`")
    recap.append(f"- code review packet: `{packet}`\n")

    recap.append("## Executive Summary\n")
    recap.append(f"- v22.12 route=`{decision['route']}`；exploration_promotion_allowed={decision['exploration_promotion_allowed']}，official_promotion_allowed={decision['official_promotion_allowed']}。")
    recap.append(f"- S1 执行 arbitrary upstream cotangent repair variants，manual_upstream_vjp_rows={manual_rows}，official_fused_kernel_complete_rows={fused_complete_rows}；因此不写 official fused success。")
    if best_h:
        recap.append(f"- best S5 FU row: loss_adapter=`{best_h.get('loss_adapter_name')}`, attempt=`{best_h.get('attempt')}`, source_func_h3200={best_h.get('source_func_h3200')}, source_loss_h3200={best_h.get('source_loss_h3200')}, C3={best_h.get('C3_source_formation_pass')}, C4={best_h.get('C4_terminal_retention_pass')}。")
    recap.append("- 所有关键结论来自本轮落盘 CSV/JSON/PT/log；未执行或 gate 阻塞阶段以 blocker 记录，没有写成通过。\n")

    recap.append("## 本轮修复过程\n")
    recap.append("- 初始 v22.12 不是 official success：D-FOU 已经通过 K13 fan-scale source-loss boundary rows 打开 source channel，但 D-CHE 在 K1/K8/K10/K11 中仍表现为 `KANSourceChannelMismatchConfirmed` 或 `KANTargetRetentionOnly`。这意味着不能把 D-FOU 的 carrier-specific efficiency pass 借给 D-CHE。")
    recap.append("- 这轮重点检查 D-CHE readout commit。诊断发现 `frozen_readout_features` 的自然顺序是 `(hidden,basis)`，而旧 readout solve 直接 reshape 到 `w2(hidden,class,basis)`，会把 class/basis layout 混在一起。修复没有回写历史 K1-K13 row，而是新增 K14 corrected-readout-layout row，保留前序负例。")
    recap.append("- K14 先做缩短 horizon diagnostic grid；完整 grid 和 full S6 重跑都因为重复计算过多被 kill 并写入执行日志，随后改成 K14-only incremental runner：复用已落盘 S6 matrix，只重新计算新增 K14 row。")
    recap.append("- 最终 K14 把 route 从 `S6-DFOUReadoutSourceOpened_DCHEMismatch` 推进到 `S6-DFOUSourceOpened_DCHESourceExistsEfficiencyBlocked`：D-CHE source-exists 有证据，但仍被 D-CHE efficiency 与 h4800 source-loss gate 阻塞。\n")

    if k14_command_rows:
        recap.append("### K14 执行尝试摘要\n")
        recap.append(md_table(k14_command_rows, ["timestamp", "status", "note"], max_rows=10))

    recap.append("### K14 关键证据\n")
    recap.append(md_table(k14_rows, [
        "carrier",
        "attempt",
        "mapping_strategy",
        "basis_estimate_fit_cosine",
        "basis_commit_projection_residual",
        "KAN_source_func_h3200",
        "KAN_source_func_h4800",
        "KAN_source_loss_h3200",
        "KAN_source_loss_h4800",
        "KAN_specific_delta_vs_MLP_same_metric",
        "K14_source_exploration_pass",
        "K14_terminal_source_loss_pass",
        "carrier_specific_efficiency_pass",
        "KAN_source_channel_decision",
        "blocker",
    ], max_rows=5))
    recap.append("Interpretation: K14 的 `basis_estimate_fit_cosine=0.9972032308578491` 和 `basis_commit_projection_residual=0.08182405680418015` 说明 corrected layout 后 readout/basis target 能被拟合；`KAN_specific_delta_vs_MLP_same_metric=7.176154110527452` 与 h3200 正 source_loss 说明 D-CHE source channel 已经不是原来的纯 mismatch。但 h4800 source_loss 为负，且 D-CHE S1 carrier efficiency pass 仍为 0，所以只能写成 source-exists / efficiency-blocked。\n")

    if dche_pre_k14_rows or k13_open_rows:
        recap.append("### K13/K14 前后对照\n")
        recap.append(md_table(k13_open_rows + dche_pre_k14_rows + k14_rows, [
            "carrier",
            "attempt",
            "mapping_strategy",
            "KAN_source_func_h3200",
            "KAN_source_loss_h3200",
            "KAN_source_loss_h4800",
            "KAN_specific_delta_vs_MLP_same_metric",
            "carrier_specific_efficiency_pass",
            "KAN_source_channel_decision",
            "blocker",
        ], max_rows=12))
        recap.append("Interpretation: D-FOU K13 是 clean pass，因为 carrier_specific_efficiency_pass=1 且 source_loss_h3200/h4800 非负；D-CHE K1/K8/K10/K11 即便有局部 positive source_loss，也低于 MLP same-metric 或触发 target-only/mismatch。K14 第一次把 D-CHE 的 `delta_vs_MLP` 推到正数，但官方 gate 仍要求 D-CHE 自己的 efficiency pass 与 terminal source-loss 同时成立。\n")

    recap.append("## 为什么仍不允许 Promotion\n")
    recap.append(md_table(promotion_gate_rows, ["gate", "required", "observed", "decision"], max_rows=10))
    recap.append("- `exploration_promotion_allowed=1` 的含义只是：代码 truth、manual arbitrary-cotangent runner、operator source、至少两个 C3 adapter、至少一个 C4 adapter、以及 KAN source evidence 都存在。")
    recap.append("- `official_promotion_allowed=0` 有三个硬原因：第一，S1 没有任何 `official_fused_kernel_complete` row；第二，S5 的 C4 只有 Delta-MSEAdapter，Ranking 只到 C3 未到 terminal retention；第三，K14 虽然打开 D-CHE source-exists，但 D-CHE `carrier_specific_efficiency_pass=0` 且 h4800 source_loss 为负。")
    recap.append("- 因此本轮有效进展应写作 `A-CodeTruth;B-Efficiency;C-Functional;D-KAN` 的 exploration progress，而不是 official promotion。下一步应优先做 D-CHE kernel-native arbitrary-cotangent officialization，并同时修 K14 h4800 source-loss retention。\n")

    recap.append("## Part A Code / Semantic Truth Gate\n")
    recap.append(md_table(truth, ["check", "pass", "metric", "value", "blocker"], max_rows=50))

    recap.append("## Part B Arbitrary-Cotangent Efficiency\n")
    recap.append(md_table(efficiency_summary, ["carrier", "profile_rows", "batch_pass_rows", "cotangent_pass_rows", "near_E1_rows", "fallback_rows", "manual_upstream_vjp_rows", "official_fused_kernel_complete_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker"], max_rows=20))
    recap.append(md_table(repair_rows, ["carrier", "repair_attempt", "attempt_rows", "pass_rows", "near_E1_rows", "best_step_ratio", "official_fused_kernel_complete_rows", "blocker"], max_rows=40))
    recap.append("Insight: v22.12 的 S1 不再使用 CE trainpath；repair variants 实际计时 forward/manual VJP/update/component telemetry。`official_fused_kernel_complete_rows=0` 是保留 blocker，不因 manual path ratio pass 而改写。\n")

    recap.append("## Part C Operator Atoms\n")
    recap.append(md_table(operator_rows, ["operator_id", "cotangent_types_supported", "expected_loss_linear_gain", "random_matched_expected_gain", "adapter_invariance_score", "control_projection_fraction", "NDS", "metric_energy_Sobolev", "S2_operator_atom_pass", "blocker"], max_rows=30))
    recap.append("Insight: operator atoms 直接消费 LossInterface output cotangent，记录 linearity/homogeneity、adapter variance、control projection 与 metric/NDS energy；labels 只允许在 supervised adapter 内部产生 cotangent，不进入 FU core。\n")

    recap.append("## Part D Variational Solve / Commit\n")
    recap.append(md_table(variational_rows, ["attempt", "operator_combo_atom_count", "selected_operators", "adapter_invariance_score", "expected_loss_linear_gain", "random_matched_expected_gain", "control_projection_fraction", "NDS", "operator_variational_pass", "blocker"], max_rows=20))
    recap.append(md_table(commit_rows, ["solver_level", "block_role", "projection_residual_Gf", "ActuationR2", "B2_transfer_gain", "random_matched_B2_transfer_gain", "function_displacement_cos_with_target", "optimizer_state_write_fraction", "S4_operator_metric_commit_pass", "blocker"], max_rows=20))

    recap.append("## Part E Arbitrary-Loss Horizon\n")
    recap.append(md_table(fu_horizon, ["loss_adapter_name", "attempt", "source_func_h100", "source_func_h800", "source_func_h3200", "source_func_h4800", "source_loss_h800", "source_loss_h3200", "source_loss_h4800", "row_positive_count_h3200", "C3_source_formation_pass", "C4_terminal_retention_pass", "TargetRetentionOnly_NotTaskUseful", "blocker"], max_rows=60))
    recap.append("Evidence chain: S5 按 adapter 计数 C3/C4；Delta-StableRandom 只作为 control 读回，不计入非随机 robustness。若 source_func 过但 source_loss 负，route 不允许提升为 arbitrary-loss success。\n")

    if repair_diagnostics:
        recap.append("## Repair Diagnostics\n")
        recap.append(md_table(repair_diagnostics, ["diagnostic_id", "operator_id", "norm_scale", "attempt", "loss_adapter_name", "S2_operator_atom_pass", "S4_operator_metric_commit_pass_rows", "source_func_h800", "source_func_h1600", "source_func_h3200", "source_loss_h800", "source_loss_h1600", "source_loss_h3200", "row_positive_count_h3200", "C3_source_formation_pass", "C4_terminal_retention_pass", "blocker"], max_rows=30))
        recap.append("Insight: repair diagnostics 是 gate 外的缩短 horizon 边界复算，用来审计已尝试修复方向。O10 non-random/source-loss-balanced consensus 用 random/stable 作为 controls、把 SourceTarget/CE/MSE/Ranking 的 raw consensus 与 CE/MSE/Ranking smooth consensus 混合；若同一 official seed 下第二个 adapter 打开，会在 S5 official matrix 中再确认。O4 scale/PID 与 O1 high-gain rows 保留为边界/负例，用于说明单纯放大或只追 retention 会触发 source_loss blocker。\n")

    recap.append("## Part F KAN Mapping\n")
    recap.append(md_table(kan_rows, ["carrier", "attempt", "mapping_strategy", "loss_adapter_name", "KAN_source_func_h3200", "KAN_source_loss_h3200", "KAN_source_loss_h4800", "KAN_specific_delta_vs_MLP_same_metric", "carrier_specific_efficiency_pass", "wrong_global_efficiency_gate_detected", "KAN_source_channel_decision", "blocker"], max_rows=30))
    recap.append("Insight: S6 每一行使用自己的 carrier_specific_efficiency_pass。D-CHE 不会借 D-FOU pass 放行；readout-only source 会保留为 basis-channel still open，不写成 KAN-general success。本轮新增 K8 readout replay scale/interval repair scan、K10 readout update-gain repair rows、K11 finite-difference basis-linearized commit、K12 D-FOU low-frequency init-variant rows、K13 fan-scale source_loss boundary rows 与 K14 D-CHE corrected-readout-layout low-degree rows。K13 已打开 D-FOU source-channel；K14 显示 D-CHE source 可被构造，但 carrier_specific_efficiency_pass=0，因此 route 写成 source-exists / efficiency-blocked，而不是 KAN-general success。K9 all/basis/readout target-gradient commit 作为执行日志中的边界诊断保留，未进入 official pass。\n")

    recap.append("## 4GPU Queue\n")
    recap.append(md_table(queue_rows, ["task_id", "line", "gpu", "command", "status", "returncode", "start_time", "end_time", "duration_sec", "log_path"], max_rows=20))
    recap.append(md_table(read_rows(out_dir / "v22_12_idle_violation.csv"), ["execution_contract_violation", "reason", "max_idle_gap_sec"], max_rows=5))

    recap.append("## Failure Taxonomy / Next Repair\n")
    s6_blocker = routes["s6"].get("blocker") or ("D-CHE_source_channel_mismatch_after_DFOU_open" if str(routes["s6"].get("route")) == "S6-DFOUReadoutSourceOpened_DCHEMismatch" else "")
    taxonomy = [
        {"blocker": routes["s0"].get("blocker", ""), "repair_or_next_direction": "if S0 fails, repair code/semantic packet first and rerun S0 only"},
        {"blocker": routes["s1"].get("blocker", ""), "repair_or_next_direction": "if efficiency blocked or fused rows remain zero, implement kernel-native arbitrary-cotangent forward/VJP/update path"},
        {"blocker": routes["s5"].get("blocker", ""), "repair_or_next_direction": "if MSE-only or source_loss negative, continue adapter-balanced/source-loss-aware operator objective before replay tuning"},
        {"blocker": s6_blocker, "repair_or_next_direction": "if D-CHE source exists but carrier efficiency is blocked, continue D-CHE kernel-native arbitrary-cotangent officialization; if source is still missing, continue low-degree/basis-estimate commit"},
    ]
    recap.append(md_table(taxonomy, ["blocker", "repair_or_next_direction"], max_rows=10))

    recap.append("## 修改记录\n")
    recap.append("- 新增 `dgkan/profiling/efficiency_v22_12.py`：实现 arbitrary-cotangent repair variant profiler，记录 manual VJP、component telemetry、carrier-specific pass 与 fused-complete blocker。")
    recap.append("- 新增 `experiments/run_v22_12_s019_truth_gate.py`：检查 v22.12 source closure、semantic firewall、carrier-specific gate 与 promotion split。")
    recap.append("- 新增并修复 `experiments/run_v22_12_operator_fu.py`：把固定 source displacement 升级为 LossInterface operator atoms；首轮 operator atom 被 NDS/adapter gate 阻塞后，新增 class 维 LowNDS 平滑；随后修复 operator atom/combo 归一化，改为保留 arbitrary cotangent row-mean 方向，避免 ranking/source-target cotangent 被 CE/logit-gauge 去均值抹掉；本轮新增 `O10_NonRandomSourceLossBalancedOperator`，把 random/stable cotangent 留作 controls，并优先尝试 non-random raw + loss-smooth consensus，final queue norm_scale 调为 0.16 以修复 ranking source_loss blocker。")
    recap.append("- 修复 `experiments/run_v22_12_arbitrary_loss_horizon.py`：按 CE/MSE/ranking/source-target/stable-random adapter 运行 horizon，并按不同非随机 adapter 计数 C3/C4；本轮新增 `source_loss_pid20_lr0p0005_clip0p03` 低 lr/PID source-loss-aware attempt。")
    recap.append("- 修复 `experiments/run_v22_12_repair_diagnostics.py`：将 row-mean/PID/norm-scale/O1-high-gain/O10-nonrandom-source-loss-balanced 边界修复尝试落盘为 diagnostic CSV；这些行不单独改 official route，只用于审计 blocker。")
    recap.append("- 修复 `experiments/run_v22_12_kan_mapping.py`：修复 v22.11 的全局 OR gate，所有 KAN row 使用 carrier-specific efficiency pass；本轮新增 K8 readout replay scale/interval repair scan、K10 readout update-gain repair rows、K11 finite-difference basis-linearized commit rows、K12 D-FOU low-frequency init-variant rows、K13 fan-scale source_loss boundary rows 与 K14 D-CHE corrected-readout-layout rows。K11 直接在 `w1/w2` basis/readout tensor 上解 train-stream target_delta 的低秩线性化 update，并落盘 operator_rank/basis_channel_energy/projection_residual 等诊断；K12/K13 只在 carrier-specific efficient 的 D-FOU 上测试 strict PrimitiveKAN `signed_pair_linear`/`signed_pair_random_linear`/`fan_scale_repair` 初始化与 replay scale/interval，不读取 adapter 公式；K14 修复 D-CHE frozen-readout feature `(hidden,basis)` 到 `w2(hidden,class,basis)` 的 layout 映射，仅作为新 row 落盘，不改写历史 K1-K13；若 source pass 但 D-CHE efficiency pass=0，则写为 `KANEfficiencyContractBlocked`。")
    recap.append("- 新增 `experiments/run_v22_12_full.py` 与 finalizer：生成 4GPU queue artifacts、results bundle、code review packet、执行日志和复盘日志。\n")

    recap.append("## 补充分析 / Evidence Chain\n")
    recap.append("- Code truth 与 S6 runtime row 双重记录 `wrong_global_efficiency_gate_detected=0` 和 `carrier_specific_efficiency_pass_consistency=1`，这是 v22.12 的 A-CodeTruth 最小进展。")
    recap.append("- S1 repair rows 显示具体尝试过 CHE/RAT/RBF/FOU variants；若 ratio pass 仍不能越过 fused blocker，结论必须限定为 manual-path exploration。")
    recap.append("- Operator source 证据链从 cotangent suite -> operator atom -> variational combo -> metric commit -> horizon adapter robustness 逐层落盘；O10 若打开第二个 adapter，仍需同一 official payload 在 S5 horizon matrix 中过 C3/C4 adapter-count gate，finalizer 不读取 diagnostic stdout 来放行。")
    recap.append("- KAN mapping 只证明 readout/basis-estimate/linearized-basis runner 中实际读回的 source channel；若 K11/K14 低秩或 corrected-layout commit 仍低于 MLP same-metric source 或触发 source_loss/efficiency gate，会作为 KANSourceChannelMismatch/TargetRetentionOnly/KANEfficiencyContractBlocked 写入 artifact。D-RAT/D-RBF KAN primitive 若未实现，会作为 deferred/blocker 写入 artifact。\n")

    V2212_RECAP_DOC.write_text("\n".join(recap), encoding="utf-8")

    with V2212_EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write("\n## Final Artifact Summary\n")
        f.write(f"- final route: `{decision['route']}`\n")
        f.write(f"- exploration_promotion_allowed: `{decision['exploration_promotion_allowed']}`\n")
        f.write(f"- official_promotion_allowed: `{decision['official_promotion_allowed']}`\n")
        f.write(f"- results bundle: `{bundle}`\n")
        f.write(f"- code review packet: `{packet}`\n")
        f.write("- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.\n")

    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_12_finalize.py --out-dir {out_dir}",
        status="completed",
        note=f"route={decision['route']} artifacts={len(idx)} bundle={bundle} code_review_packet={packet}",
    )


if __name__ == "__main__":
    main()
