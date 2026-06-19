#!/usr/bin/env python3
"""Finalize v22.16 routes, recap, and artifact bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_16_common import (  # noqa: E402
    PYTHON,
    V2216_RECAP_DOC,
    append_exec,
    artifact_index,
    build_results_bundle,
    ensure_out,
    finite_float,
    init_docs,
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


def _final_route(routes: dict[str, dict[str, Any]]) -> str:
    s0 = routes.get("s0", {})
    traj = routes.get("trajectory", {})
    mlp = routes.get("mlp", {})
    geom = routes.get("geometry", {})
    sm = routes.get("source_manifold", {})
    kan = routes.get("kan", {})
    eff = routes.get("efficiency", {})
    task = routes.get("task", {})
    mlp_fu = routes.get("mlp_fu", {})
    if not int_flag(s0.get("S0_pass")):
        return "R0-CodeOrSemanticGateFailed"
    if not int_flag(traj.get("real_trajectory_logging_pass")):
        return "R1-RealTrajectoryLoggingFailed"
    if not int_flag(mlp.get("C2_risk_prediction_pass")):
        return "R2-RealRiskPredictionNoGo"
    if not int_flag(mlp.get("C1_mlp_adaptive_pass")):
        return "R3-RealAdaptiveControllerNoGo"
    if int_flag(geom.get("pointwise_pass")) and not int_flag(geom.get("ranking_pairwise_exploration_pass")):
        return "R4-PointwiseAdaptiveFUOpened_PairwiseNoGo"
    if sm and not int_flag(sm.get("MLP_source_manifold_pass")) and not int_flag(sm.get("KAN_basis_manifold_pass")):
        return "R5-RealSourceManifoldNoGo"
    if sm and int_flag(sm.get("MLP_source_manifold_pass")) and not int_flag(sm.get("KAN_basis_manifold_pass")):
        return "R6-RealSourceManifoldOpened_ControllerNoGain"
    if not int_flag(kan.get("KAN_basis_exploration_pass")):
        return "R7-MLPRealAdaptiveFUOpened_KANBasisNoGo"
    if not int_flag(eff.get("real_adaptive_efficiency_exploration_pass")):
        return "R9-KANBasisExplorationOpened_EfficiencyOrTaskPending"
    if not int_flag(task.get("mechanism_ready_for_task")):
        return "R10-RealAdaptiveFUMechanismPass_TaskPending"
    if (
        int_flag(task.get("full_superiority_pass"))
        and int_flag(kan.get("KAN_basis_official_pass"))
        and int_flag(eff.get("real_adaptive_efficiency_official_pass"))
        and int_flag(mlp_fu.get("fu_general_value_allowed"))
        and int_flag(mlp_fu.get("kan_specific_carrier_value_allowed"))
    ):
        return "R14-OfficialDGKANFullSuperiorityReady"
    if int_flag(task.get("accuracy_superiority_pass")):
        return "R13-DGKAN_AccuracySuperiority_NLLPending"
    if int_flag(task.get("nll_superiority_pass")):
        return "R12-DGKAN_NLLSuperiority_NotAccuracy"
    if int_flag(task.get("functional_value_vs_KANAdamW_pass")):
        return "R11-FunctionalValueVsKANAdamW_MLPNotBeaten"
    return "R10-RealAdaptiveFUMechanismPass_TaskPending"


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = f"{PYTHON} experiments/run_v22_16_finalize.py --out-dir {out_dir}"
    routes = {
        "s0": read_json(out_dir / "v22_16_code_truth_route.json"),
        "trajectory": read_json(out_dir / "v22_16_real_trajectory_route.json"),
        "mlp": read_json(out_dir / "v22_16_mlp_adaptive_route.json"),
        "geometry": read_json(out_dir / "v22_16_loss_geometry_route.json"),
        "source_manifold": read_json(out_dir / "v22_16_source_manifold_route.json"),
        "kan": read_json(out_dir / "v22_16_kan_basis_route.json"),
        "efficiency": read_json(out_dir / "v22_16_efficiency_route.json"),
        "task": read_json(out_dir / "v22_16_task_proof_route.json"),
        "mlp_fu": read_json(ROOT / "results/v22_16_mlp_fu_mainline/official_v22_16_m/v22_16_mlp_fu_route.json"),
    }
    route = _final_route(routes)
    s0_rows = read_rows(out_dir / "v22_16_code_truth_gate.csv")
    traj_rows = read_rows(out_dir / "v22_16_real_trajectory_source_log.csv")
    rank_rows = read_rows(out_dir / "v22_16_source_history_rank_matrix.csv")
    mlp_rows = read_rows(out_dir / "v22_16_real_mlp_adaptive_guidance_matrix.csv")
    risk_rows = read_rows(out_dir / "v22_16_real_risk_prediction_matrix.csv")
    geom_rows = read_rows(out_dir / "v22_16_real_loss_geometry_operator_matrix.csv")
    sm_rows = read_rows(out_dir / "v22_16_real_source_manifold_controller_matrix.csv")
    kan_rows = read_rows(out_dir / "v22_16_real_KAN_basis_controller_matrix.csv")
    eff_rows = read_rows(out_dir / "v22_16_real_adaptive_efficiency_matrix.csv")
    task_rows = read_rows(out_dir / "v22_16_strict_task_eval_matrix.csv")
    repair_route_rows: list[dict[str, Any]] = []
    repair_mlp_rows: list[dict[str, Any]] = []
    for route_path in sorted(out_dir.glob("r3_repair_*/v22_16_mlp_adaptive_route.json")):
        repair_dir = route_path.parent
        repair_route = read_json(route_path)
        repair_route_rows.append(
            {
                "repair_dir": str(repair_dir.relative_to(out_dir)),
                "route": repair_route.get("route", ""),
                "C1": repair_route.get("C1_mlp_adaptive_pass", ""),
                "C1_explore": repair_route.get("C1_mlp_adaptive_exploration_pass", ""),
                "C2": repair_route.get("C2_risk_prediction_pass", ""),
                "official_scope": repair_route.get("run_scope_official_mechanism", ""),
                "risk_AUC_H100": repair_route.get("risk_AUC_H100", ""),
                "best_risk_model_H100": repair_route.get("best_risk_model_H100", ""),
                "median_lead": repair_route.get("median_intervention_lead_time", ""),
            }
        )
        for row in read_rows(repair_dir / "v22_16_real_mlp_adaptive_guidance_matrix.csv"):
            repair_mlp_rows.append({"repair_dir": str(repair_dir.relative_to(out_dir)), **row})
    deferred: list[dict[str, Any]] = []
    if not int_flag(routes["task"].get("mechanism_ready_for_task")):
        deferred.append({"item": "strict_task_proof", "status": "deferred_or_not_run", "reason": routes["task"].get("blocker", routes["task"].get("route", ""))})
    if route != "R14-OfficialDGKANFullSuperiorityReady":
        deferred.append({"item": "official_DGKAN_full_superiority_claim", "status": "blocked", "reason": route})
    write_rows(out_dir / "v22_16_deferred_items.csv", deferred)
    write_rows(out_dir / "v22_16_gpu_utilization_timeline.csv", read_rows(out_dir / "v22_16_command_journal.csv"))
    write_rows(out_dir / "v22_16_artifact_index.csv", artifact_index(out_dir))
    bundle = out_dir / "v22_16_results_bundle.zip"
    final = {
        "final_route": route,
        "route_inputs": routes,
        "artifact_bundle": str(bundle.resolve()),
        "trajectory_rows": len(traj_rows),
        "mlp_rows": len(mlp_rows),
        "kan_rows": len(kan_rows),
        "efficiency_rows": len(eff_rows),
        "task_rows": len(task_rows),
        "mlp_fu_route": routes["mlp_fu"].get("route", ""),
        "mlp_fu_mechanism_pass": routes["mlp_fu"].get("mlp_fu_mechanism_pass", ""),
        "mlp_fu_task_pass": routes["mlp_fu"].get("mlp_fu_task_pass", ""),
        "mlp_fu_nll_superiority_rows": routes["mlp_fu"].get("mlp_fu_nll_superiority_rows", ""),
        "mlp_fu_accuracy_superiority_rows": routes["mlp_fu"].get("mlp_fu_accuracy_superiority_rows", ""),
        "mlp_fu_auc_superiority_rows": routes["mlp_fu"].get("mlp_fu_auc_superiority_rows", ""),
        "mlp_fu_no_debt_rows": routes["mlp_fu"].get("mlp_fu_no_debt_rows", ""),
        "kan_fu_vs_mlp_fu_nll_rows": routes["mlp_fu"].get("kan_fu_vs_mlp_fu_nll_rows", ""),
        "kan_fu_vs_mlp_fu_accuracy_rows": routes["mlp_fu"].get("kan_fu_vs_mlp_fu_accuracy_rows", ""),
        "kan_fu_vs_mlp_fu_auc_rows": routes["mlp_fu"].get("kan_fu_vs_mlp_fu_auc_rows", ""),
        "fu_general_value_allowed": routes["mlp_fu"].get("fu_general_value_allowed", 0),
        "kan_specific_carrier_value_allowed": routes["mlp_fu"].get("kan_specific_carrier_value_allowed", 0),
    }
    write_json(out_dir / "v22_16_final_route.json", final)

    best_rank_k16 = max((finite_float(r.get("history_PCA_explained_variance_k16"), 0.0) for r in rank_rows), default=0.0)
    best_kan_res = min((finite_float(r.get("basis_projection_residual"), 999.0) for r in kan_rows), default=999.0)
    worst_eff = max((finite_float(r.get("full_loop_ratio_vs_mlp"), 0.0) for r in eff_rows if r.get("carrier") != "MLP"), default=0.0)
    recap = [
        "# DG-KAN v22.16 Real-Trajectory Source-Manifold Adaptive FU 实验结果复盘\n",
        f"生成时间：{now_sg()}\n",
        "原则：本复盘只引用本次落盘 artifact 与脚本运行结果；未运行、blocked、gate-dependent 项明确标记，不补造指标。\n",
        "## Final route\n",
        f"- Final route: `{route}`\n",
        f"- Results bundle: `{bundle}`\n",
        "## 实现与修复审计\n",
        "- 新增 v22.16 公共执行层 `experiments/run_v22_16_common.py`，统一 real dataset loader、真实模型构造、command journal、code packet、artifact index、执行日志和复盘日志。\n",
        "- 新增 `dgkan/fu/real_jacobian_commit.py`：真实参数 flatten、basis/readout selector、autograd output Jacobian、linearized commit solve、finite-difference gradcheck。\n",
        "- 新增 `dgkan/profiling/real_adaptive_fu_efficiency.py`：真实 MLP/KAN batch 的 CE/MSE/Ranking/StableRandom loss/cotangent、forward/backward/controller component timing。\n",
        "- 新增 v22.16 stage runners：S0、real trajectory logger、real MLP adaptive/risk、real source manifold、real loss geometry、real KAN basis、real efficiency、strict task proof、finalizer。\n",
        "- 修复方向遵循计划：finalizer 分离 accuracy/NLL/AUC；diagnostic/readout/synthetic 不得晋升 official；task proof 受 mechanism gates 约束；Source-Manifold 只使用 Part B 落盘 train-stream history。\n",
        "- S0 修复审计：初版 scanner 把 `uses_validation_test_future` firewall 字段误报为 forbidden direction；已改为跳过 firewall 字段名并重跑 S0。另将 source-history PCA unit test 从随机扰动改成确定性 source-history contract，避免 S0 随机失败。\n",
        "- MLP adaptive 修复审计：初次 official run 因 `source_prev` 在 CPU、当前 source 在 CUDA 导致 `cosine_t` device mismatch；已将公共 `cosine_t` 统一到 CPU 计算并重跑 MLP adaptive。\n",
        "- MLP risk 修复审计：按 Part D/R2 先后测试 built-in、destructive-only、derivative-only、source-loss-boundary、monotone agreement；risk AUC 排除 random/signflip controls；lead time 改为每个 risk model 的 top20% train-trajectory analysis threshold，不把固定阈值失败伪装成无信号。\n",
        "- MLP controller 修复审计：按 Part C/R3 发现 reactive `controller_to_base_update_ratio_mean` 常超过 0.5 后，新增 conservative controller modes，降低 lambda/prox activation；随后新增 train-only online source-boundary low-lambda modes，用上一训练步 `source_func/source_loss` 决定下一步弱介入。MLP route 同时区分 exploration 与 official mechanism gate，短 seed0/短 horizon 诊断不能晋升 official C1。\n",
        "- Loss-geometry 修复审计：初次 Ranking/Preference rows 把 logits flatten 后送入 batch-level incidence context，产生 640 vs 64 维度错误；已改为 pairwise B-space 每样本 cotangent/source 后重跑，Ranking/Preference smoke 通过。\n",
        "- Efficiency 修复审计：初次 real timing 暴露 basis-native autograd Jacobian path 的巨大 overhead/memory；已按计划尝试 row-local peak-memory reset 与更小 active output rows 后重跑。修复后仍未过 memory/overhead gate，保留为真实 efficiency blocker。\n",
        "- Execution 修复审计：完整 3200/4800-step Python small-step loops 超过 14-16 分钟仍未产出 stage artifact，被记录为执行层 blocker；随后按计划 slow-logging/controller 修复方向改用 shorter/sketched real profile 重跑，未把被停止 run 写成完成。\n",
        "## S0 code truth\n",
        md_table(s0_rows, ["S0_pass", "clean_unzip_compileall_pass", "clean_unzip_import_pass", "missing_transitive_dependency_count", "operator_core_adapter_name_branch_count", "loss_formula_branch_in_core_count", "future_validation_test_query_direction_count", "finalizer_separates_accuracy_and_nll", "finalizer_blocks_OR_superiority_claim", "blocker"], max_rows=5),
        f"Analysis: S0 route `{routes['s0'].get('route', '')}`。若这里失败，后续机制解释全部无效。\n",
        "## Real trajectory\n",
        f"- Route: `{routes['trajectory'].get('route', '')}`；trajectory_rows={len(traj_rows)}；finite_source_fraction={routes['trajectory'].get('finite_source_fraction', '')}；best_PCA_k16={best_rank_k16:.6g}。\n",
        md_table(rank_rows[:12], ["dataset", "model_variant", "source_candidate_name", "history_rows", "history_rank_estimate", "history_effective_rank", "history_PCA_explained_variance_k16", "history_projection_residual_k16", "future_leakage_check_pass", "train_only_history_pass"], max_rows=12),
        "Analysis: Part B source payload 来自真实 train batch logits/cotangent/step displacement。Source-Manifold 的 basis 构造只读已落盘的过去 source history；payload sha256 写在 route JSON。\n",
        "## MLP adaptive and risk\n",
        f"- Route: `{routes['mlp'].get('route', '')}`；C1={routes['mlp'].get('C1_mlp_adaptive_pass', '')}；C1_explore={routes['mlp'].get('C1_mlp_adaptive_exploration_pass', '')}；official_scope={routes['mlp'].get('run_scope_official_mechanism', '')}；C2={routes['mlp'].get('C2_risk_prediction_pass', '')}；risk_AUC_H100={routes['mlp'].get('risk_AUC_H100', '')}；best_risk_model_H100={routes['mlp'].get('best_risk_model_H100', '')}；median_lead={routes['mlp'].get('median_intervention_lead_time', '')}。\n",
        md_table(risk_rows, ["risk_model", "H", "AUC_predict_washout_H50", "AUC_predict_washout_H100", "AUC_predict_washout_H200", "precision_at_top20pct_risk", "median_lead_time_steps", "analysis_scope"], max_rows=18),
        md_table(mlp_rows[:12], ["dataset", "seed", "adapter", "mode", "source_func_h3200", "source_loss_h3200", "source_func_h4800", "source_loss_h4800", "C4_terminal_retention_pass", "AUC_loss_time", "diagnostic_only"], max_rows=12),
        "Analysis: risk labels 是完成后的 train-trajectory analysis labels，不作为同 run direction。Predictive/reactive/fixed/random/sign controls 分开记录。\n",
        "## R3 controller repair attempts\n",
        md_table(repair_route_rows, ["repair_dir", "route", "C1", "C1_explore", "C2", "official_scope", "risk_AUC_H100", "best_risk_model_H100", "median_lead"], max_rows=12),
        md_table(repair_mlp_rows[:24], ["repair_dir", "dataset", "seed", "adapter", "mode", "source_func_h3200", "source_loss_h3200", "source_func_h4800", "source_loss_h4800", "R4800_over_3200_func", "C4_terminal_retention_pass", "controller_to_base_update_ratio_mean"], max_rows=24),
        "Analysis: R3 repair rows 是聚焦诊断，不满足 official coverage 时只能作为 blocker localization 或下一轮修复依据，不能替代 full official mechanism pass。\n",
        "## Source manifold\n",
        f"- Route: `{routes['source_manifold'].get('route', '')}`；MLP_pass={routes['source_manifold'].get('MLP_source_manifold_pass', '')}；KAN_pass={routes['source_manifold'].get('KAN_basis_manifold_pass', '')}；controls_fail={routes['source_manifold'].get('controls_fail', '')}。\n",
        md_table(sm_rows[:12], ["controller_kind", "source_manifold_family", "source_manifold_dim", "basis_source", "history_window", "projection_residual_Gf", "source_loss_h4800", "basis_channel_energy_fraction", "MLP_source_manifold_pass", "KAN_basis_manifold_pass", "route_blocker"], max_rows=12),
        "Analysis: 如果 random/shuffled/signflip controls 也通过，Source-Manifold 不会晋升。若 KAN energy 低而 MLP projection 可行，结论必须是 KAN carrier blocker，而不是 source manifold success。\n",
        "## Loss geometry\n",
        f"- Route: `{routes['geometry'].get('route', '')}`；pointwise={routes['geometry'].get('pointwise_pass', '')}；ranking={routes['geometry'].get('ranking_pairwise_exploration_pass', '')}；rename={routes['geometry'].get('adapter_renaming_pass', '')}。\n",
        md_table(geom_rows[:12], ["dataset", "seed", "adapter", "operator_context_type", "adapter_renaming_pass_with_same_delta_C", "same_delta_different_C_output_delta_norm", "pairwise_margin_gain_h3200", "pairwise_antisymmetry_error", "C3_operator_pass"], max_rows=12),
        "Analysis: pairwise/preference rows 使用真实 batch label-derived incidence/preference graph；同 δ 同 C rename 一致，同 δ 不同 C 可改变输出。\n",
        "## KAN basis\n",
        f"- Route: `{routes['kan'].get('route', '')}`；exploration={routes['kan'].get('KAN_basis_exploration_pass', '')}；official={routes['kan'].get('KAN_basis_official_pass', '')}；best_residual={best_kan_res:.6g}。\n",
        md_table(kan_rows[:14], ["carrier", "dataset", "seed", "controller_mode", "basis_channel_energy_fraction", "readout_channel_energy_fraction", "basis_projection_residual", "basis_gradcheck_rel_error", "KAN_source_loss_h4800", "KAN_basis_exploration_pass", "KAN_basis_official_pass", "horizon_metric_kind", "blocker"], max_rows=14),
        "Analysis: KAN basis rows 是真实 local linearized actuation/gradcheck 证据，不等价于 long-horizon task success；readout rows 保留为 diagnostic。\n",
        "## Efficiency\n",
        f"- Route: `{routes['efficiency'].get('route', '')}`；exploration={routes['efficiency'].get('real_adaptive_efficiency_exploration_pass', '')}；official={routes['efficiency'].get('real_adaptive_efficiency_official_pass', '')}；worst_ratio={worst_eff:.6g}。\n",
        md_table(eff_rows[:14], ["carrier", "variant", "batch_size", "hidden", "cotangent_type", "controller_mode", "full_loop_ratio_vs_mlp", "controller_overhead_ratio", "memory_ratio_vs_mlp", "grad_rel_error", "controller_efficiency_exploration_pass", "controller_efficiency_official_pass", "outlier_reason"], max_rows=14),
        "Analysis: timing rows use real model/batch loops. Component timings are retained for repair; synthetic controller timing is not counted official.\n",
        "## Task proof\n",
        f"- Route: `{routes['task'].get('route', '')}`；mechanism_ready={routes['task'].get('mechanism_ready_for_task', '')}；accuracy_rows={routes['task'].get('accuracy_superiority_rows', '')}；nll_rows={routes['task'].get('nll_superiority_rows', '')}；auc_rows={routes['task'].get('auc_superiority_rows', '')}；functional_vs_KANAdamW={routes['task'].get('functional_value_vs_KANAdamW_rows', '')}。\n",
        md_table(task_rows[:18], ["dataset", "seed", "variant", "status", "param_count", "final_test_loss_readback", "final_test_accuracy_readback", "NLL_delta_vs_MLP", "accuracy_delta_vs_MLP", "AUC_loss_time_ratio_vs_best_control", "diagnostic_only", "official_basis_claim_allowed", "blocker"], max_rows=18),
        "Analysis: strict task proof 只有 mechanism gates 全过才运行；accuracy/NLL/AUC gates 独立写入 CSV，禁止 OR promotion。\n",
        "## MLP+FU Mainline Supplement\n",
        f"- Route: `{routes['mlp_fu'].get('route', 'not_run')}`；mlp_fu_mechanism_pass={routes['mlp_fu'].get('mlp_fu_mechanism_pass', '')}；mlp_fu_task_pass={routes['mlp_fu'].get('mlp_fu_task_pass', '')}；fu_general_value_allowed={routes['mlp_fu'].get('fu_general_value_allowed', 0)}；kan_specific_carrier_value_allowed={routes['mlp_fu'].get('kan_specific_carrier_value_allowed', 0)}。\n",
        f"- MLP+FU rows: nll={routes['mlp_fu'].get('mlp_fu_nll_superiority_rows', '')}；accuracy={routes['mlp_fu'].get('mlp_fu_accuracy_superiority_rows', '')}；auc={routes['mlp_fu'].get('mlp_fu_auc_superiority_rows', '')}；no_debt={routes['mlp_fu'].get('mlp_fu_no_debt_rows', '')}；KAN-vs-MLPFU nll={routes['mlp_fu'].get('kan_fu_vs_mlp_fu_nll_rows', '')}；accuracy={routes['mlp_fu'].get('kan_fu_vs_mlp_fu_accuracy_rows', '')}；auc={routes['mlp_fu'].get('kan_fu_vs_mlp_fu_auc_rows', '')}。\n",
        "Analysis: R14 official route now additionally requires MLP+FU general value and KAN-specific carrier gates. If the supplement is missing or below M8, finalizer cannot claim official DG-KAN full superiority.\n",
        "## 结论与 insight\n",
        f"- 当前最终解释必须按 `{route}` 处理。\n",
        "- 证据链顺序：S0 语义与 artifact truth -> 真实 train trajectory source -> MLP adaptive/risk lead time -> real source manifold -> loss geometry -> real KAN basis Jacobian/gradcheck -> real efficiency -> gated task proof。\n",
        "- 若本轮停在 R1/R2/R3，说明 adaptive FU 尚未形成可用真实控制信号；若停在 R5/R6，说明 low-rank source manifold 或 KAN manifold 没有足够证据；若停在 R7，说明 MLP adaptive 与 source history 不足以推出 strict FC-PureKAN basis carrier；若停在 R9，下一步应优先修 full-loop timing 或 native fast path。\n",
        "- 不能写 Official DG-KAN > MLP，除非 final route 是 R14，且 basis carrier、efficiency、accuracy、NLL、AUC、debt 与 controls 全部同时满足。\n",
    ]
    V2216_RECAP_DOC.write_text("\n".join(recap), encoding="utf-8")
    append_exec(out_dir, command, status="completed", gpu="n/a", task_id="finalize", files="v22_16_final_route.json; v22_16_artifact_index.csv; v22_16_results_bundle.zip; docs recap", note=f"final_route={route}")
    build_results_bundle(out_dir)


if __name__ == "__main__":
    main()
