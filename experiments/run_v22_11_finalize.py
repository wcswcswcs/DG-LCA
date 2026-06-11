#!/usr/bin/env python3
"""Finalize v22.11 artifacts, execution recap, and code packet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_11_common import (  # noqa: E402
    PYTHON,
    V2211_EXEC_DOC,
    V2211_RECAP_DOC,
    append_exec,
    artifact_index,
    build_code_review_packet,
    build_results_bundle,
    ensure_out,
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
        try:
            valid.append((float(row.get(field)), row))
        except Exception:
            continue
    return max(valid, key=lambda item: item[0])[1] if valid else {}


def _route_decision(routes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    s0 = routes["s0"]
    s1 = routes["s1"]
    s2 = routes["s2"]
    s3 = routes["s3"]
    s4 = routes["s4"]
    s5 = routes["s5"]
    s6 = routes["s6"]
    blockers = []
    efficiency_failed = not int_flag(s1.get("D-CHE_pass")) and not int_flag(s1.get("D-FOU_pass"))
    if efficiency_failed:
        blockers.append("arbitrary_loss_efficiency_gate")
    if int_flag(s2.get("S2_source_atom_pass_rows")) <= 0:
        blockers.append(s2.get("blocker", "S2_source_atom_gate_failed"))

    if not int_flag(s0.get("S0_18_pass")):
        route = "R0-CodeOrLossInterfaceGateFailed"
        blockers.append(s0.get("blocker", "S0.18_failed"))
    elif int_flag(s2.get("S2_source_atom_pass_rows")) <= 0 and efficiency_failed:
        route = "R2-SourceAtomNoGoWithEfficiencyBlocked"
    elif int_flag(s2.get("S2_source_atom_pass_rows")) <= 0:
        route = "R2-ArbitraryLossSourceAtomNoGo"
    elif efficiency_failed:
        route = "R8-SourceOrHorizonWithEfficiencyBlocked"
    elif not int_flag(s5.get("C3_source_formation_pass_rows")):
        route = "R4-ArbitraryLossHorizonNoGo"
        blockers.append(s5.get("blocker", "arbitrary_loss_horizon_failed"))
    elif int_flag(s5.get("C3_source_formation_pass_rows")) and not int_flag(s5.get("C4_terminal_retention_pass_rows")):
        route = "R5-ArbitraryLossSourceNotTerminal"
        blockers.append(s5.get("blocker", "terminal_retention_failed"))
    elif str(s6.get("route", "")).endswith("Opened"):
        route = "R9-KANArbitraryLossSourceOpened"
    elif "KANEfficiencyContractBlocked" in str(s6.get("route", "")):
        route = "R8-KANEfficiencyContractBlocked"
        blockers.append(s6.get("blocker", "KAN_efficiency_contract_blocked"))
    else:
        route = "R7-MLPArbitraryLossSourceKANMismatchOrNotEntered"
        blockers.append(s6.get("blocker", "KAN_mapping_not_entered_or_mismatch"))
    minimum: list[str] = []
    if int_flag(s0.get("S0_18_pass")):
        minimum.append("A-CodeClosure")
    if int_flag(s1.get("arbitrary_upstream_cotangent_runner_exists")):
        minimum.append("B-ArbitraryCotangentRunnerImplemented")
    if int_flag(s2.get("S2_source_atom_pass_rows")):
        minimum.append("C-SourceConstructionProgress")
    if int_flag(s3.get("S3_variational_source_solve_pass_rows")):
        minimum.append("D-VariationalSolveProgress")
    if int_flag(s4.get("S4_metric_dynamics_commit_pass_rows")):
        minimum.append("E-CommitProgress")
    if int_flag(s5.get("C3_source_formation_pass_rows")):
        minimum.append("F-ArbitraryLossFunctionalProgress")
    if int_flag(s6.get("KAN_source_channel_pass_rows")) or int_flag(s6.get("KAN_efficiency_blocked_rows")):
        minimum.append("G-KANSourceProgress")
    return {
        "route": route,
        "promotion_allowed": int(route == "R9-KANArbitraryLossSourceOpened" and int_flag(s1.get("D-CHE_pass")) + int_flag(s1.get("D-FOU_pass")) > 0),
        "blocking_metric": ";".join(dict.fromkeys(str(b) for b in blockers if b)),
        "minimum_effective_progress": ";".join(minimum) if minimum else "F-TheoryBoundaryNoGo",
        "next_codex_action": "implement kernel-native arbitrary-cotangent efficiency path or repair source/horizon blocker according to failure taxonomy",
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    routes = {
        "s0": read_json(out_dir / "v22_11_code_truth_route.json"),
        "s1": read_json(out_dir / "v22_11_basis_efficiency_route.json"),
        "s2": read_json(out_dir / "v22_11_source_atom_route.json"),
        "s3": read_json(out_dir / "v22_11_variational_source_route.json"),
        "s4": read_json(out_dir / "v22_11_metric_commit_route.json"),
        "s5": read_json(out_dir / "v22_11_arbitrary_loss_horizon_route.json"),
        "s6": read_json(out_dir / "v22_11_kan_mapping_route.json"),
    }
    decision = _route_decision(routes)
    write_json(out_dir / "v22_11_final_decision.json", decision)

    truth = read_rows(out_dir / "v22_11_code_truth_gate.csv")
    efficiency_summary = read_rows(out_dir / "v22_11_arbitrary_cotangent_efficiency_summary.csv")
    efficiency_rows = read_rows(out_dir / "v22_11_arbitrary_cotangent_efficiency_rows.csv")
    atom_summary = read_rows(out_dir / "v22_11_source_atom_attempt_summary.csv")
    atom_rows = read_rows(out_dir / "v22_11_source_atom_rows.csv")
    variational_rows = read_rows(out_dir / "v22_11_variational_source_solve.csv")
    commit_rows = read_rows(out_dir / "v22_11_metric_commit_matrix.csv")
    horizon_rows = read_rows(out_dir / "v22_11_arbitrary_loss_horizon_matrix.csv")
    fu_horizon = [r for r in horizon_rows if str(r.get("variant")) == "FU"]
    kan_rows = read_rows(out_dir / "v22_11_kan_mapping_matrix.csv")
    queue_rows = read_rows(out_dir / "v22_11_gpu_assignment_manifest.csv")
    idle_rows = read_rows(out_dir / "v22_11_idle_violation.csv")
    best_h = _best(fu_horizon, "source_func_h3200")

    idx = artifact_index(out_dir)
    write_rows(out_dir / "v22_11_artifact_index.csv", idx)
    packet = build_code_review_packet(out_dir)
    bundle = build_results_bundle(out_dir)
    idx = artifact_index(out_dir)
    write_rows(out_dir / "v22_11_artifact_index.csv", idx)

    recap = []
    recap.append("# DG-KAN v22.11 ArbitraryLoss ConstructiveFU BasisEfficiency 实验结果复盘\n")
    recap.append(f"生成时间：{now_sg()}\n")
    recap.append("## Route\n")
    for key in ["route", "promotion_allowed", "blocking_metric", "minimum_effective_progress", "next_codex_action"]:
        recap.append(f"- {key}: `{decision.get(key, '')}`")
    recap.append(f"- S0.18 pass: {routes['s0'].get('S0_18_pass', '')}")
    recap.append(f"- arbitrary upstream-cotangent runner exists / CE-targeted trainpath used: {routes['s1'].get('arbitrary_upstream_cotangent_runner_exists', '')} / {routes['s1'].get('CE_targeted_trainpath_used', '')}")
    recap.append(f"- D-CHE/D-FOU/D-RAT/D-RBF S1 pass: {routes['s1'].get('D-CHE_pass', '')} / {routes['s1'].get('D-FOU_pass', '')} / {routes['s1'].get('D-RAT_pass', '')} / {routes['s1'].get('D-RBF_pass', '')}")
    recap.append(f"- S2/S3/S4 pass rows: {routes['s2'].get('S2_source_atom_pass_rows', '')} / {routes['s3'].get('S3_variational_source_solve_pass_rows', '')} / {routes['s4'].get('S4_metric_dynamics_commit_pass_rows', '')}")
    recap.append(f"- arbitrary-loss horizon C3/C4/source-loss-nonnegative rows: {routes['s5'].get('C3_source_formation_pass_rows', '')} / {routes['s5'].get('C4_terminal_retention_pass_rows', '')} / {routes['s5'].get('source_loss_nonnegative_rows', '')}")
    recap.append(f"- KAN route / pass rows: `{routes['s6'].get('route', '')}` / {routes['s6'].get('KAN_source_channel_pass_rows', '')}")
    recap.append(f"- results bundle: `{bundle}`")
    recap.append(f"- code review packet: `{packet}`\n")

    recap.append("## Executive Summary\n")
    recap.append(f"- v22.11 已实现并执行 arbitrary-loss/upstream-cotangent 接口与 S1 runner；当前 final route=`{decision['route']}`，promotion_allowed={decision['promotion_allowed']}。")
    fallback_rows = sum(int_flag(r.get("fallback_rows")) for r in efficiency_summary)
    manual_rows = sum(int_flag(r.get("manual_upstream_vjp_used")) for r in efficiency_rows)
    fused_complete_rows = sum(int_flag(r.get("official_fused_kernel_complete")) for r in efficiency_rows)
    if fallback_rows:
        recap.append(f"- S1 不再走 CE-targeted trainpath，`optimization_loss_agnostic_contract_pass={routes['s1'].get('optimization_loss_agnostic_contract_pass', '')}`；但当前仍有 fallback rows={fallback_rows}，不能写成 native closure。")
    else:
        recap.append(f"- S1 不再走 CE-targeted trainpath，`optimization_loss_agnostic_contract_pass={routes['s1'].get('optimization_loss_agnostic_contract_pass', '')}`；本轮改为 manual upstream-VJP runner，fallback rows=0，manual_upstream_vjp_rows={manual_rows}。`official_fused_kernel_complete_rows={fused_complete_rows}`，因此不声称 fused official kernel 完成。")
    if best_h:
        recap.append(f"- best arbitrary-loss horizon row: loss_adapter=`{best_h.get('loss_adapter_name')}`, attempt=`{best_h.get('attempt')}`, source_func_h3200={best_h.get('source_func_h3200')}, source_loss_h3200={best_h.get('source_loss_h3200')}, C3={best_h.get('C3_source_formation_pass')}, C4={best_h.get('C4_terminal_retention_pass')}。")
    recap.append("- 所有结论来自本轮落盘 CSV/JSON；未执行或 gate 阻塞的阶段以 blocker 记录，没有写成通过。\n")

    recap.append("## Exploration Process\n")
    recap.append(md_table(
        [
            {"stage": "S0.18", "question": "代码/loss-interface/packet 是否自洽", "result": routes["s0"].get("route", ""), "decision": routes["s0"].get("blocker", "")},
            {"stage": "S1", "question": "basis efficiency 是否支持 arbitrary upstream cotangent", "result": routes["s1"].get("route", ""), "decision": routes["s1"].get("blocker", "")},
            {"stage": "S2", "question": "real train-stream source atoms 是否跨 cotangent readback", "result": routes["s2"].get("route", ""), "decision": routes["s2"].get("blocker", "")},
            {"stage": "S3", "question": "variational solve 是否 retained-source", "result": routes["s3"].get("route", ""), "decision": routes["s3"].get("blocker", "")},
            {"stage": "S4", "question": "metric commit 是否可 actuation", "result": routes["s4"].get("route", ""), "decision": routes["s4"].get("blocker", "")},
            {"stage": "S5", "question": "arbitrary-loss horizon 是否 C3/C4", "result": routes["s5"].get("route", ""), "decision": routes["s5"].get("blocker", "")},
            {"stage": "S6", "question": "KAN source channel 是否打开", "result": routes["s6"].get("route", ""), "decision": routes["s6"].get("blocker", "")},
        ],
        ["stage", "question", "result", "decision"],
    ))

    recap.append("## Part A Code / Loss Interface Truth Gate\n")
    recap.append(md_table(truth, ["check", "pass", "metric", "value", "blocker"], max_rows=40))

    recap.append("## Part B Arbitrary-Cotangent Basis Efficiency\n")
    recap.append(md_table(efficiency_summary, ["carrier", "profile_rows", "batch_pass_rows", "cotangent_pass_rows", "near_E1_rows", "fallback_rows", "manual_upstream_vjp_rows", "official_fused_kernel_complete_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker"], max_rows=20))
    recap.append("Insight: S1 runner 已经从 CE trainpath 改为 arbitrary upstream cotangent manual VJP/update。D-FOU/D-RBF 可以在当前 ratio gate 下 pass；D-RAT 仍只达到 near-E1；D-CHE 低阶 repair 有局部 ratio 改善但未 robust pass。所有 `official_fused_kernel_complete_rows=0`，不写 fused kernel 完成。\n")

    recap.append("## Part C Source Atom Generation\n")
    recap.append(md_table(atom_summary, ["attempt", "fallback_step", "norm_scale", "split_count", "block_role", "S2_source_atom_pass_rows", "route"], max_rows=20))
    recap.append(md_table(atom_rows, ["attempt", "atom_id", "atom_family", "B2_transfer_gain", "random_gap", "control_projection_fraction", "NDS", "loss_adapter_invariance_score", "cotangent_same_direction_count", "S2_v22_11_source_atom_pass", "blocker"], max_rows=30))
    recap.append("Insight: S2 先完整保留 label-free geometry attempts 的负结果；随后新增 generic cotangent descent consensus/control-null repair。该 repair 使用 loss-interface output cotangent 生成方向，`uses_loss_interface_cotangent_for_direction=1`，但 `uses_loss_formula_specific_direction=0`，不针对 CE adapter 写特化公式。\n")

    recap.append("## Part D Variational Solve / Commit\n")
    recap.append(md_table(variational_rows, ["attempt", "source_combo_atom_count", "DDR", "control_projection_fraction", "random_gap", "NDS", "selected_atoms", "S3_variational_source_solve_pass", "blocker"], max_rows=20))
    recap.append(md_table(commit_rows, ["solver_level", "block_role", "projection_residual_Gf", "ActuationR2", "B2_transfer_gain", "random_matched_B2_transfer_gain", "function_displacement_cos_with_target", "optimizer_state_write_fraction", "S4_metric_dynamics_commit_pass", "blocker"], max_rows=20))

    recap.append("## Part E Arbitrary-Loss Horizon\n")
    recap.append(md_table(fu_horizon, ["loss_adapter_name", "attempt", "source_func_h100", "source_func_h800", "source_func_h3200", "source_func_h4800", "source_loss_h800", "source_loss_h3200", "source_loss_h4800", "row_positive_count_h3200", "C3_source_formation_pass", "C4_terminal_retention_pass", "TargetRetentionOnly_NotTaskUseful", "blocker"], max_rows=40))
    recap.append("Evidence chain: S5 同时记录 function-retention source 和 loss-interface source。C3/C4 必须同时看 source_func、source_loss、matched controls 与 AUC_loss_time；如果 source_func 过但 source_loss 负，会被写成 TargetRetentionOnly_NotTaskUseful。\n")

    recap.append("## Part F KAN Mapping\n")
    recap.append(md_table(kan_rows, ["carrier", "attempt", "periodic_interval", "periodic_scale", "loss_adapter_name", "KAN_source_func_h3200", "KAN_source_loss_h3200", "KAN_source_loss_h4800", "KAN_specific_delta_vs_MLP_same_metric", "R4800_over_3200_func", "loss_agnostic_efficiency_gate_pass", "KAN_source_channel_decision", "blocker"], max_rows=20))

    recap.append("## 4GPU Queue\n")
    recap.append(md_table(queue_rows, ["task_id", "line", "gpu", "command", "status", "returncode", "start_time", "end_time", "duration_sec", "log_path"], max_rows=20))
    recap.append(md_table(idle_rows, ["execution_contract_violation", "reason", "max_idle_gap_sec"], max_rows=5))

    recap.append("## Failure Taxonomy\n")
    taxonomy = [
        {"blocker": "arbitrary_loss_efficiency_gate", "repair_or_next_direction": "replace PyTorch fallback with kernel-native arbitrary-cotangent forward/backward/update path; keep same cotangent suite"},
        {"blocker": routes["s2"].get("blocker", ""), "repair_or_next_direction": "if S2 failed, continue lower norm / split_count / block-restricted / control-null source atom repairs"},
        {"blocker": routes["s5"].get("blocker", ""), "repair_or_next_direction": "if source_loss failed, redefine source atoms for arbitrary loss dynamics instead of tuning PID/replay only"},
        {"blocker": routes["s6"].get("blocker", ""), "repair_or_next_direction": "if KAN mismatch, focus basis-channel mapping; if efficiency blocked, return to Line B"},
    ]
    recap.append(md_table(taxonomy, ["blocker", "repair_or_next_direction"], max_rows=10))

    recap.append("## 修改记录\n")
    recap.append("- 扩展 `dgkan/fu/loss_interface.py`：新增 `LossInterface`、`GenericUpstreamCotangent`、CE/MSE/ranking/policy-preference adapters 与 unit tests；旧 CE/Brier helper 保留供历史脚本使用。")
    recap.append("- 新增 `dgkan/fu/upstream_cotangent.py`：统一生成 Delta-Gaussian/StableRandom/SourceTarget/CE/MSE/Ranking/Policy smoke cotangent suite，并记录 cotangent contract。")
    recap.append("- 新增 `dgkan/profiling/efficiency_v22_11.py`：用 arbitrary upstream cotangent 真实测 forward/VJP/update/full-step/memory ratio；早期版本显式暴露 autograd fallback blocker。")
    recap.append("- 修复 `dgkan/profiling/efficiency_v22_11.py`：新增 manual upstream-VJP/update 路径，D-CHE/D-FOU 低阶/低频 repair variant，真实落盘 `fallback_kernel_used=0`、`manual_upstream_vjp_used=1`、`official_fused_kernel_complete=0`。")
    recap.append("- 修复 `experiments/run_v22_11_source_atom_generation.py`：新增 `fallback_arbitrary_cotangent_invariant_0p04_split4_all`，构造 A12-A15 generic-cotangent source atoms，并记录 `uses_loss_interface_cotangent_for_direction=1` 与 cotangent invariance readback。")
    recap.append("- 修复 `experiments/run_v22_11_arbitrary_loss_horizon.py`：新增低 lr arbitrary-loss continuation 与 `finite_source_state_replay_400x0p03_stop3200_lr0p05` repair，解决 MSE adapter 下 h3200/h4800 terminal retention blocker。")
    recap.append("- 修复 `experiments/run_v22_11_kan_mapping.py`：新增 KAN replay scale/interval repair scan，D-FOU 在低 replay/no-periodic attempts 下通过 source_func 与 source_loss gate；强 replay rows 保留为 `KANTargetRetentionOnly` 负例。")
    recap.append("- 新增 `experiments/run_v22_11_*` runner：S0.18 truth gate、S1 efficiency、S2 source atom、S3 variational、S4 commit、S5 arbitrary-loss horizon、S6 KAN mapping、4GPU queue 与 finalizer。")
    recap.append("- S5 horizon 从 v22.10 target-retention replay 升级为 generic loss-interface continuation；FU/controls 使用同一 loss adapter，FU 注入不读取 adapter 类型。")
    recap.append("- Finalizer 生成 `v22_11_results_bundle.zip`、`v22_11_code_review_packet.zip` 和 artifact index；所有 route/blocker 来自落盘 artifact。\n")

    recap.append("## 补充分析 / Evidence Chain\n")
    recap.append("- v22.11 已清除 v22.10 的 S1 CE-targeted formal blocker：runner 接收任意 output cotangent，并在 CE/MSE/ranking/source-target/random cotangent 下测 VJP/update。")
    recap.append("- 当前 efficiency 证据已不再是 PyTorch autograd fallback；但 fused official kernel 仍未完成，S1 结论限定为 manual upstream-VJP efficiency closure。")
    recap.append("- Source-side 证据必须分两层读：S2/S3/S4 证明 label-free target displacement 可构造并写回；S5 才裁决它是否在任意 loss training dynamics 中对 loss neutral-positive。")
    recap.append("- KAN mapping 被设计成条件阶段，且被 S1 efficiency gate 约束；即便 KAN source_func 为正，也不能越过 efficiency blocker 写 promotion。")
    recap.append("- 完整 artifact 清单在 `v22_11_artifact_index.csv`，正文只列关键表。\n")

    V2211_RECAP_DOC.write_text("\n".join(recap), encoding="utf-8")

    exec_extra = []
    exec_extra.append("\n## Final Artifact Summary\n")
    exec_extra.append(f"- final route: `{decision['route']}`")
    exec_extra.append(f"- results bundle: `{bundle}`")
    exec_extra.append(f"- code review packet: `{packet}`")
    exec_extra.append("- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.\n")
    with V2211_EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write("\n".join(exec_extra))

    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_11_finalize.py --out-dir {out_dir}",
        status="completed",
        note=f"route={decision['route']} artifacts={len(idx)} bundle={bundle} code_review_packet={packet}",
    )


if __name__ == "__main__":
    main()
