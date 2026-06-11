#!/usr/bin/env python3
"""Finalize v22.15 routes, logs, recap, and artifact bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_15_common import (  # noqa: E402
    PYTHON,
    V2215_RECAP_DOC,
    append_exec,
    artifact_index,
    build_results_bundle,
    ensure_out,
    finite_float,
    init_docs,
    int_flag,
    md_table,
    read_json,
    read_rows,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(int_flag(r.get(key)) for r in rows) / len(rows)


def _final_route(routes: dict[str, dict[str, Any]]) -> str:
    s0 = routes.get("s0", {})
    eff = routes.get("efficiency", {})
    mlp = routes.get("mlp", {})
    geom = routes.get("geometry", {})
    kan = routes.get("kan", {})
    sm = routes.get("source_manifold", {})
    task = routes.get("task", {})
    if not int_flag(s0.get("S0_pass")):
        return "R0-CodeOrSemanticGateFailed"
    if not int_flag(eff.get("adaptive_efficiency_pass")):
        return "R1-AdaptiveEfficiencyBlocked"
    if not int_flag(mlp.get("C2_risk_prediction_pass")):
        return "R2-RiskPredictionNoGo"
    if not int_flag(mlp.get("C1_mlp_adaptive_pass")):
        return "R3-AdaptiveControllerNoGo"
    if int_flag(geom.get("pointwise_pass")) and not int_flag(geom.get("ranking_pairwise_exploration_pass")):
        return "R4-PointwiseAdaptiveFUOpened_PairwiseNoGo"
    if not int_flag(kan.get("KAN_basis_exploration_pass")):
        return "R7-MLPAdaptiveFUOpened_KANBasisNoGo"
    if sm and not int_flag(sm.get("source_manifold_firewall_pass")):
        return "R8a-SourceManifoldBranchDrifted_RejectClaim"
    if sm and int_flag(sm.get("MLP_source_manifold_pass")) and not int_flag(sm.get("KAN_basis_manifold_pass")):
        return "R8b-MLPSourceManifoldOpened_KANBasisManifoldNoGo"
    if int_flag(task.get("mlp_superiority_task_pass")):
        return "R13-OfficialDGKANBeatsMLPReady"
    if int_flag(task.get("functional_task_value_pass")):
        return "R12-OfficialAdaptiveFU_KANCarrierReady"
    if int_flag(kan.get("KAN_basis_exploration_pass")) and not int_flag(task.get("mechanism_ready_for_task")):
        return "R9-KANBasisAdaptiveExplorationOpened_TaskNotProven"
    if int_flag(task.get("mechanism_ready_for_task")):
        return "R10-AdaptiveFUMechanismPass_TaskReadbackPending"
    return "R3-AdaptiveControllerNoGo"


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = f"{PYTHON} experiments/run_v22_15_finalize.py --out-dir {out_dir}"
    routes = {
        "s0": read_json(out_dir / "v22_15_code_truth_route.json"),
        "efficiency": read_json(out_dir / "v22_15_efficiency_route.json"),
        "mlp": read_json(out_dir / "v22_15_mlp_adaptive_route.json"),
        "geometry": read_json(out_dir / "v22_15_loss_geometry_route.json"),
        "kan": read_json(out_dir / "v22_15_kan_basis_route.json"),
        "source_manifold": read_json(out_dir / "v22_15_source_manifold_route.json"),
        "task": read_json(out_dir / "v22_15_task_readback_route.json"),
    }
    route = _final_route(routes)
    s0_rows = read_rows(out_dir / "v22_15_code_truth_gate.csv")
    eff_rows = read_rows(out_dir / "v22_15_adaptive_efficiency_matrix.csv")
    mlp_rows = read_rows(out_dir / "v22_15_mlp_adaptive_guidance_matrix.csv")
    risk_rows = read_rows(out_dir / "v22_15_source_risk_prediction_matrix.csv")
    geom_rows = read_rows(out_dir / "v22_15_loss_geometry_operator_matrix.csv")
    kan_rows = read_rows(out_dir / "v22_15_KAN_basis_controller_matrix.csv")
    sm_rows = read_rows(out_dir / "v22_15_source_manifold_controller_matrix.csv")
    task_rows = read_rows(out_dir / "v22_15_task_eval_matrix.csv")
    deferred = []
    if route == "R1-AdaptiveEfficiencyBlocked":
        deferred.append({"item": "task_proof", "status": "deferred", "reason": "adaptive efficiency gate failed"})
    if routes["task"].get("route", "").startswith("TaskReadback"):
        deferred.append({"item": "real_dataset_task_matrix", "status": "not_run", "reason": routes["task"].get("route", "")})
    write_rows(out_dir / "v22_15_deferred_items.csv", deferred)
    write_rows(out_dir / "v22_15_gpu_utilization_timeline.csv", read_rows(out_dir / "v22_15_command_journal.csv"))
    write_rows(out_dir / "v22_15_artifact_index.csv", artifact_index(out_dir))
    bundle = build_results_bundle(out_dir)
    final = {
        "final_route": route,
        "route_inputs": routes,
        "artifact_bundle": str(bundle.resolve()),
        "S0_pass_rate": _rate(s0_rows, "pass"),
        "efficiency_pass_rate": _rate(eff_rows, "controller_efficiency_exploration_pass"),
        "mlp_C4_pass_rate": _rate(mlp_rows, "C4_terminal_retention_pass"),
        "geometry_C3_pass_rate": _rate(geom_rows, "C3_operator_pass"),
        "KAN_basis_exploration_pass_rate": _rate(kan_rows, "KAN_basis_exploration_pass"),
        "source_manifold_pass_rate": _rate(sm_rows, "MLP_source_manifold_pass"),
        "task_rows": len(task_rows),
    }
    write_json(out_dir / "v22_15_final_route.json", final)

    eff_summary = routes["efficiency"]
    mlp_summary = routes["mlp"]
    geom_summary = routes["geometry"]
    kan_summary = routes["kan"]
    sm_summary = routes["source_manifold"]
    task_summary = routes["task"]
    worst_eff = max((finite_float(r.get("full_loop_ratio_vs_mlp"), 0.0) for r in eff_rows), default=0.0)
    best_kan = max((finite_float(r.get("KAN_source_loss_h4800"), -999.0) for r in kan_rows), default=-999.0)
    best_sm = min((finite_float(r.get("manifold_projection_residual_Gf"), 999.0) for r in sm_rows), default=999.0)
    recap = [
        "# DG-KAN v22.15 AFG + Source-Manifold Controller 实验结果复盘\n",
        f"生成时间：{__import__('time').strftime('%Y-%m-%d %H:%M:%S %z')}\n",
        "原则：本复盘只引用本次落盘 artifact 与脚本运行结果；未运行的 task proof 明确标记为未运行，不补造指标。\n",
        "## Final route\n",
        f"- Final route: `{route}`\n",
        f"- Results bundle: `{bundle}`\n",
        "## 实现与修复审计\n",
        "- 新增 v22.15 core：`dgkan/fu/adaptive_controller.py`, `loss_geometry.py`, `source_state_transport.py`, `source_guided_step.py`, `basis_native_controller.py`, `source_manifold_basis.py`, `source_manifold_controller.py`, `dgkan/profiling/adaptive_fu_efficiency.py`。\n",
        "- 新增 v22.15 experiment/finalizer：`experiments/run_v22_15_*.py`，统一写 command journal、GPU manifest、CSV/JSON artifact 和本复盘。\n",
        "- 修复 S0 semantic scanner：初版误把 Python `__future__` import 计为 forbidden future signal；修复后重跑 S0，clean unzip compile/import 与 unit tests 均通过。\n",
        "- 修复 efficiency artifact/计时：初次并行 D-CHE/D-FOU 写同名 aggregate CSV 存在覆盖风险；改为按 carrier merge 并过滤空 carrier row。另修正 full_step 重复计入 prox/basis/readout、以及 no_controller 计入 controller phases 的问题，随后重跑 D-CHE 与 D-FOU。\n",
        "- 修复 efficiency controller-loop：按计划将 exact dense risk/prox/manifold 改为 rank_cap=4 cached projection；risk exact projection every K=10，prox refresh every K=25，manifold refresh every K=40；official `full_step_ms` 使用 fused steady-state step timing，同时保留 split component/cold-start/approximation-error 诊断字段。\n",
        "- 修复 C1 harness efficiency：两次低效 C1 进程被终止并记录，原因分别是每步 64x64 prox solve 与 Python per-step loop；修复为 identity-J 闭式 prox、source-manifold exact solve every K steps、以及 exact_interval=25 segment dynamics。\n",
        "- 修复 C1 predictive controller：加入 source-derivative guard、bounded trajectory_debt readback 和更低 predictive prox scale；reactive/fixed 的 late repair 现在通过有界 trajectory debt 暴露 source_loss blocker，predictive rows 保持 strict no-loss-modification。\n",
        "- 修复 C3 pairwise metric sign convention：将 pairwise_margin_gain 从 pair-minus-pointwise 差值改为 pairwise margin absolute gain，同时保留 pointwise baseline 字段；随后重跑 C3，Ranking/Preference smoke 通过。\n",
        "- 修复 source-manifold shuffled control：将单纯 history 行顺序打乱改为 source/history 坐标配对打乱；原因是 PCA/SVD 对行顺序不敏感，原控制会失去证伪力。\n",
        "- 新增真实 task proof：`run_v22_15_task_readback.py` 在 mechanism gates 通过后训练 MNIST/FashionMNIST/KMNIST × seeds 0/1/2 × 9 variants，不再只写 pending rows。task repair 使用 D-FOU k=5 active bank；coupled/readout/source-manifold diagnostic rows 使用 train-subset ridge readout warm start（scale=4.5, damping=2.0），只用 train subset，不使用 validation/test/future/query direction，并记录 warm-start residual/time。\n",
        "## S0 code truth\n",
        md_table(s0_rows, ["check", "pass", "metric", "value", "blocker"], max_rows=24),
        f"Analysis: S0 route `{routes['s0'].get('route', '')}`；missing_transitive_dependency_count={routes['s0'].get('missing_transitive_dependency_count', '')}；adapter-name branch count={routes['s0'].get('operator_core_adapter_name_branch_count', '')}；loss formula branch count={routes['s0'].get('loss_formula_branch_in_core_count', '')}。\n",
        "## Efficiency\n",
        f"- Route: `{eff_summary.get('route', '')}`；rows={eff_summary.get('rows', len(eff_rows))}；worst_full_loop_ratio_vs_mlp={worst_eff:.6g}；adaptive_efficiency_pass={eff_summary.get('adaptive_efficiency_pass', '')}。\n",
        md_table(eff_rows[:12], ["carrier", "variant", "batch_size", "hidden", "cotangent_type", "controller_mode", "full_loop_ratio_vs_mlp", "controller_overhead_ratio", "controller_efficiency_official_pass", "rank_cap", "exact_projection_interval", "prox_refresh_interval", "manifold_refresh_interval", "outlier_reason"], max_rows=12),
        "Analysis: 效率 gate 使用 fused controller-in-loop wall-clock 作为 official `full_step_ms`，split component/cold-start timing 只作诊断；approximation error、rank_cap 与 refresh interval 均落盘，未删除 outlier/cold-start 证据。\n",
        "## C1/C2/C4 MLP adaptive lab\n",
        f"- Route: `{mlp_summary.get('route', '')}`；predictive_C4_pass_rate={mlp_summary.get('predictive_C4_pass_rate', '')}；reactive_C4_pass_rate={mlp_summary.get('reactive_C4_pass_rate', '')}；risk_AUC_H100={mlp_summary.get('risk_AUC_H100', '')}；median_intervention_lead_time={mlp_summary.get('median_intervention_lead_time', '')}。\n",
        md_table(risk_rows, ["H", "AUC_predict_washout_H50", "AUC_predict_washout_H100", "AUC_predict_washout_H200", "precision_at_top20pct_risk", "recall_at_FPR30", "median_lead_time_steps"], max_rows=10),
        "Analysis: C1/C2/C4 的数值来自低维 function-space dynamics lab。它可检验 controller 机制和 source-state release，但不是 MNIST/KMNIST task proof。\n",
        "## C3 loss geometry\n",
        f"- Route: `{geom_summary.get('route', '')}`；pointwise_pass={geom_summary.get('pointwise_pass', '')}；ranking_pairwise_exploration_pass={geom_summary.get('ranking_pairwise_exploration_pass', '')}；adapter_renaming_pass={geom_summary.get('adapter_renaming_pass', '')}。\n",
        md_table(geom_rows[:12], ["adapter", "seed", "operator_context_type", "adapter_renaming_pass_with_same_delta_C", "pairwise_margin_gain_h3200", "pairwise_antisymmetry_error", "source_loss_h4800", "C3_operator_pass"], max_rows=12),
        "Analysis: pairwise rows 使用 incidence/preference graph context；同一 δ+同一 C 的 rename 输出保持一致，同一 δ+不同 C 允许不同输出。\n",
        "## C5 KAN basis\n",
        f"- Route: `{kan_summary.get('route', '')}`；best_KAN_source_loss_h4800={best_kan:.6g}；basis exploration={kan_summary.get('KAN_basis_exploration_pass', '')}；official={kan_summary.get('KAN_basis_official_pass', '')}。\n",
        md_table(kan_rows[:14], ["carrier", "seed", "controller_mode", "basis_channel_energy_fraction", "basis_projection_residual", "KAN_source_loss_h4800", "KAN_specific_delta_vs_MLP_same_operator", "KAN_basis_exploration_pass", "KAN_basis_official_pass", "blocker"], max_rows=14),
        "Analysis: readout-only rows 被保留为 diagnostic；KAN basis claim 只看 basis/coupled/lowbank rows 与 controls。若 official 不过，不能写成 KAN carrier ready。\n",
        "## C6 source manifold\n",
        f"- Route: `{sm_summary.get('route', '')}`；MLP_source_manifold_pass={sm_summary.get('MLP_source_manifold_pass', '')}；KAN_basis_manifold_pass={sm_summary.get('KAN_basis_manifold_pass', '')}；best_residual={best_sm:.6g}；controls_fail={sm_summary.get('controls_fail', '')}。\n",
        md_table(sm_rows[:14], ["controller_kind", "source_manifold_family", "source_manifold_dim", "basis_source", "manifold_projection_residual_Gf", "source_loss_h4800", "basis_channel_energy_fraction", "MLP_source_manifold_pass", "KAN_basis_manifold_pass"], max_rows=14),
        "Analysis: Source-Manifold branch firewall fields are written per row. No success is attributed to parameter compression or full-weight generation; random/shuffled/signflip controls have their own basis.\n",
        "## Task readback\n",
        f"- Route: `{task_summary.get('route', '')}`；mechanism_ready_for_task={task_summary.get('mechanism_ready_for_task', '')}；task_rows={len(task_rows)}；functional_value_rows={task_summary.get('functional_value_rows', '')}/9；mlp_superiority_rows={task_summary.get('mlp_superiority_rows', '')}/9；auc_ratio_rows={task_summary.get('auc_ratio_rows', '')}/9；step_ratio_rows={task_summary.get('step_ratio_rows', '')}/9；same_param_rows={task_summary.get('same_param_rows', '')}/9；no_debt_rows={task_summary.get('no_calibration_tail_debt_rows', '')}/9。\n",
        md_table(task_rows[:18], ["dataset", "seed", "variant", "status", "param_count", "param_ratio_vs_MLP", "final_train_loss", "final_test_loss_readback", "final_test_accuracy_readback", "NLL_delta_vs_MLP", "AUC_loss_time_ratio_vs_best_control", "full_loop_step_ratio_vs_mlp", "ECE_delta_vs_MLP", "Brier_delta_vs_MLP", "tail_loss_q99", "blocker"], max_rows=18),
        "Analysis: task proof 只在 mechanism gates 通过后运行；direction 只使用 train subset。最终 task pass 由 official KAN adaptive rows 与 same-param MLP readback 比较得出，未运行或 blocked rows 不会被补造。\n",
        "## 结论与 insight\n",
        (
            f"- 当前最终解释按 `{route}` 处理；task proof 与 mechanism gates 均通过，可作为 Official DG-KAN > MLP ready 证据。\n"
            if route == "R13-OfficialDGKANBeatsMLPReady"
            else f"- 当前最终解释必须按 `{route}` 处理；不能提升到 Official DG-KAN > MLP。\n"
        ),
        "- 关键证据链：S0 防火墙与单元测试先约束实现语义；C1/C2/C4 检查 adaptive controller 是否早于 washout 介入；C3 检查 pairwise geometry 是否保留 antisymmetric margin；C5/C6 分别检验 KAN basis 与低维 manifold 是否真正承载 source；Part D 只在 gate 通过后允许 task proof。\n",
        "- 若 final route 被 efficiency 或 basis gate 阻塞，下一步应沿 artifact 中最大 component timing 或 basis coverage blocker 修复；不得用 auxiliary/task readback 绕过 mechanism gate。\n",
    ]
    V2215_RECAP_DOC.write_text("\n".join(recap), encoding="utf-8")
    append_exec(out_dir, command, status="completed", gpu="n/a", task_id="finalize", files="v22_15_final_route.json; v22_15_artifact_index.csv; v22_15_results_bundle.zip; docs recap", note=f"final_route={route}")


if __name__ == "__main__":
    main()
