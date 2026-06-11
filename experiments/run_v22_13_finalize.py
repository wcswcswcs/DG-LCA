#!/usr/bin/env python3
"""Finalize v22.13 artifacts, execution log, recap, and route decision."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_13_common import (  # noqa: E402
    PYTHON,
    V2213_RECAP_DOC,
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
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def _declared_gpu(command: str, note: str = "") -> str:
    text = f"{command} {note}"
    match = re.search(r"--device\s+(cuda:\d+)", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(cuda:\d+)\b", text)
    return match.group(1) if match else ""


def _route_decision(routes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    s0 = routes.get("s0", {})
    semantic = routes.get("semantic", {})
    eff = routes.get("eff", {})
    atom = routes.get("atom", {})
    solve = routes.get("solve", {})
    commit = routes.get("commit", {})
    horizon = routes.get("horizon", {})
    kan = routes.get("kan", {})
    task = routes.get("task", {})
    blockers: list[str] = []
    c3 = int_flag(horizon.get("C3_adapter_pass_count"))
    c4 = int_flag(horizon.get("C4_adapter_pass_count"))
    fused = int_flag(eff.get("official_fused_kernel_complete_rows"))
    semantic_ok = int_flag(semantic.get("semantic_pass", s0.get("role_blind_static_pass", 0)))
    operator_chain = int_flag(atom.get("S2_operator_atom_pass_rows")) > 0 and int_flag(solve.get("S3_operator_variational_pass_rows")) > 0 and int_flag(commit.get("S4_operator_metric_commit_pass_rows")) > 0
    controls_ok = int_flag(horizon.get("StableRandom_or_RandomMatched_pass")) == 0
    dfou_source = int_flag(kan.get("DFOU_corrected_source_exists"))
    dche_source = int_flag(kan.get("DCHE_corrected_source_exists"))
    dche_eff = int_flag(kan.get("DCHE_efficiency_pass"))
    dche_loss = finite_float(kan.get("DCHE_source_loss_h4800"), -999.0)
    operator_exploration = int(int_flag(s0.get("S0_pass")) and semantic_ok and operator_chain and c3 >= 2 and controls_ok)
    official_operator = int(operator_exploration and c3 >= 3 and c4 >= 2 and controls_ok and int_flag(horizon.get("adapter_holdout_pass")) and semantic_ok)
    kan_source = int(dfou_source or dche_source)
    official_kan = int(official_operator and fused > 0 and int_flag(kan.get("basis_channel_pass_rows")) > 0)
    scientific = int(official_kan and int_flag(task.get("full_scientific_task_gate_pass")))
    if not int_flag(s0.get("S0_pass")):
        route = "R0-CodeTruthFailed"
        blockers.append(str(s0.get("blocker", "S0_failed")))
    elif not semantic_ok:
        route = "R1-OperatorSemanticGateFailed"
        blockers.append(str(semantic.get("blocker", "semantic_gate_failed")))
    elif not operator_chain:
        route = "R2-OnlyRoleAwareConsensus_NotTrueOperator" if int_flag(atom.get("O10_counted_as_official")) else "R1-OperatorSemanticGateFailed"
        blockers.extend([str(atom.get("blocker", "")), str(solve.get("blocker", "")), str(commit.get("blocker", ""))])
    elif int_flag(horizon.get("StableRandom_or_RandomMatched_pass")):
        route = "R5-TargetRetentionOnly_NotTaskUseful"
        blockers.append("control_source_passed")
    elif c4 == 1 and "Delta-MSEAdapter" in str(horizon.get("C4_adapter_pass_list", "")) and c3 < 3:
        route = "R4-MSEOnlySource_NotArbitraryLoss"
        blockers.append(str(horizon.get("blocker", "")))
    elif dche_source and (not dche_eff or dche_loss < -1.0e-6):
        route = "R8-DCHESourceExists_EfficiencyOrSourceLossBlocked"
        blockers.append(str(kan.get("blocker", "")))
    elif dfou_source and not dche_source:
        route = "R7-DFOUCorrectedSourceOpened_DCHEBlocked"
        blockers.append(str(kan.get("blocker", "")))
    elif c3 >= 2 and c4 < 2:
        route = "R6-OperatorC3Opened_C4TerminalBlocked"
        blockers.append(str(horizon.get("blocker", "")))
    elif official_operator and not kan_source:
        route = "R9-TrueLossInterfaceOperatorExplorationSuccess_KANPending"
        blockers.append(str(kan.get("blocker", "KAN_source_channel_pending")))
        if fused <= 0:
            blockers.append(str(eff.get("blocker", "")))
    elif fused <= 0:
        route = "R3-KernelNativeEfficiencyBlocked"
        blockers.append(str(eff.get("blocker", "")))
    elif official_kan and not scientific:
        route = "R11-OfficialOperatorCarrierReady_TaskEvidencePending"
        blockers.append(str(task.get("blocker", "")))
    elif scientific:
        route = "R12-FullScientificPromotionReady"
    elif operator_exploration and fused <= 0:
        route = "R10-KANOperatorCarrierExplorationSuccess_OfficialFusedPending"
        blockers.append("official_fused_kernel_complete_rows=0")
    elif operator_exploration:
        route = "R9-TrueLossInterfaceOperatorExplorationSuccess_KANPending"
    else:
        route = "R6-OperatorC3Opened_C4TerminalBlocked" if c3 else "R1-OperatorSemanticGateFailed"
        blockers.append(str(horizon.get("blocker", "")))
    minimum: list[str] = []
    if int_flag(s0.get("S0_pass")):
        minimum.append("A-CodeTruth")
    if operator_chain and c3 >= 2:
        minimum.append("B-Operator")
    if c4 >= 1 or c3 >= 2:
        minimum.append("C-Adapter")
    if fused > 0:
        minimum.append("D-Efficiency")
    if dfou_source or dche_source or int_flag(kan.get("readout_only_pass_rows")) > 0 or kan.get("route"):
        minimum.append("E-KAN")
    if not minimum:
        minimum.append("F-Theory")
    if scientific:
        next_action = "promotion-ready; only independent replication and paper-facing audit remain"
    elif official_kan and not scientific:
        next_action = "run/repair task-level evidence without using task metrics for direction"
    elif official_operator and not official_kan:
        if fused > 0:
            next_action = "repair KAN basis-channel carrier mismatch with a new basis-state objective or carrier architecture"
        else:
            next_action = "repair KAN basis-channel carrier mismatch and kernel-native arbitrary-cotangent fused backward contract"
    elif c3 >= 2 and c4 < 2:
        next_action = "repair operator horizon C4/source_loss without MSE-only or terminal replay shortcuts"
    elif fused <= 0:
        next_action = "implement D-FOU/D-CHE native arbitrary-cotangent fused backward before claiming official efficiency"
    else:
        next_action = "repair the first blocking layer in route order: semantic, operator atom/commit, controls, KAN carrier, or efficiency"
    return {
        "route": route,
        "exploration_promotion_allowed": operator_exploration,
        "official_promotion_allowed": official_operator,
        "official_kan_carrier_promotion_allowed": official_kan,
        "scientific_claim_allowed": scientific,
        "blocking_metric": ";".join(dict.fromkeys(str(b) for b in blockers if b)),
        "minimum_effective_progress": ";".join(minimum),
        "next_codex_action": next_action,
    }


def _ensure_queue_artifacts(out_dir: Path) -> None:
    command_rows = read_rows(out_dir / "v22_13_command_journal.csv")
    queue_rows = []
    manifest_rows = []
    timeline_rows = []
    for idx, r in enumerate(command_rows):
        gpu = _declared_gpu(str(r.get("command", "")), str(r.get("note", "")))
        task_id = f"observed_{idx}"
        queue_rows.append({"task_id": task_id, "line": "", "gpu": gpu, "deps": "", "command": r.get("command", ""), "env": ""})
        manifest_rows.append({"task_id": task_id, "gpu": gpu, "command": r.get("command", ""), "status": r.get("status", ""), "note": r.get("note", "")})
        timeline_rows.append({"timestamp": r.get("timestamp", ""), "event": "observed_command", "task_id": task_id, "gpu": gpu})
    write_rows(out_dir / "v22_13_runnable_queue.csv", queue_rows)
    write_rows(out_dir / "v22_13_gpu_assignment_manifest.csv", manifest_rows)
    write_rows(out_dir / "v22_13_gpu_utilization_timeline.csv", timeline_rows)
    write_rows(out_dir / "v22_13_idle_violation.csv", [{"execution_contract_violation": 1, "reason": "executed as staged GPU commands with explicit cuda devices, not the planned dynamic 4GPU queue scheduler", "max_idle_gap_sec": ""}])
    write_json(out_dir / "v22_13_queue_drain_report.json", {"tasks": len(command_rows), "completed_tasks": sum(1 for r in command_rows if r.get("status") == "completed"), "blocked_tasks": sum(1 for r in command_rows if r.get("status") == "blocked"), "queue_drained": 0, "execution_contract_violation": 1})
    write_rows(out_dir / "v22_13_deferred_items.csv", [{"item": "dynamic_4GPU_queue_contract", "status": "not_used", "reason": "staged GPU execution used after code bootstrap; official promotion already blocked by fused-kernel/operator-route gates"}])


def _ensure_gpu_runtime_evidence(out_dir: Path) -> None:
    path = out_dir / "v22_13_gpu_runtime_evidence.csv"
    if path.exists():
        return
    rows = []
    for idx, r in enumerate(read_rows(out_dir / "v22_13_command_journal.csv")):
        gpu = _declared_gpu(str(r.get("command", "")), str(r.get("note", "")))
        if gpu:
            rows.append(
                {
                    "evidence_id": f"declared_device_{idx}",
                    "scope": "command_journal",
                    "command": r.get("command", ""),
                    "gpu": gpu,
                    "pid": "",
                    "used_memory_mib": "",
                    "evidence_source": "command --device argument and appended note",
                    "status": r.get("status", ""),
                    "note": r.get("note", ""),
                }
            )
    write_rows(path, rows)


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    _ensure_queue_artifacts(out_dir)
    _ensure_gpu_runtime_evidence(out_dir)
    routes = {
        "s0": read_json(out_dir / "v22_13_code_truth_route.json"),
        "semantic": read_json(out_dir / "v22_13_operator_semantic_route.json"),
        "eff": read_json(out_dir / "v22_13_efficiency_native_route.json"),
        "atom": read_json(out_dir / "v22_13_operator_atom_route.json"),
        "solve": read_json(out_dir / "v22_13_operator_variational_route.json"),
        "commit": read_json(out_dir / "v22_13_operator_commit_route.json"),
        "horizon": read_json(out_dir / "v22_13_operator_horizon_route.json"),
        "kan": read_json(out_dir / "v22_13_kan_mapping_route.json"),
        "task": read_json(out_dir / "v22_13_task_eval_route.json"),
    }
    decision = _route_decision(routes)
    write_json(out_dir / "v22_13_final_decision.json", decision)
    idx = artifact_index(out_dir)
    write_rows(out_dir / "v22_13_artifact_index.csv", idx)
    packet = build_code_review_packet(out_dir)
    bundle = build_results_bundle(out_dir)
    idx = artifact_index(out_dir)
    write_rows(out_dir / "v22_13_artifact_index.csv", idx)
    simple_svg(out_dir / "figures/v22_13_gpu_utilization_dashboard.svg", "v22.13 queue/utilization", read_rows(out_dir / "v22_13_gpu_assignment_manifest.csv"), "")

    truth = read_rows(out_dir / "v22_13_code_truth_gate.csv")
    firewall = read_rows(out_dir / "v22_13_semantic_firewall.csv")
    renaming = read_rows(out_dir / "v22_13_adapter_renaming_tests.csv")
    law = read_rows(out_dir / "v22_13_operator_law_tests.csv")
    layout = read_rows(out_dir / "v22_13_corrected_layout_tests.csv")
    eff_rows = read_rows(out_dir / "v22_13_native_efficiency_truth_table.csv")
    eff_summary = read_rows(out_dir / "v22_13_official_fused_status_matrix.csv")
    atom_rows = read_rows(out_dir / "v22_13_operator_atom_matrix.csv")
    commit_rows = read_rows(out_dir / "v22_13_operator_commit_matrix.csv")
    horizon_rows = [r for r in read_rows(out_dir / "v22_13_adapter_horizon_matrix.csv") if str(r.get("variant")) == "FU"]
    kan_rows = read_rows(out_dir / "v22_13_KAN_mapping_matrix.csv")
    k17_rows = read_rows(out_dir / "v22_13_K17_basis_channel_commit.csv")
    task_rows = read_rows(out_dir / "v22_13_task_eval_matrix.csv")
    command_rows = read_rows(out_dir / "v22_13_command_journal.csv")
    gpu_rows = read_rows(out_dir / "v22_13_gpu_runtime_evidence.csv")
    repair_route = read_json(out_dir / "v22_13_operator_repair_sweep_route.json")
    repair_rows = read_rows(out_dir / "v22_13_operator_repair_sweep.csv")

    recap: list[str] = []
    recap.append("# DG-KAN v22.13 TrueLossInterfaceOperatorFU 实验结果复盘\n")
    recap.append(f"生成时间：{now_sg()}\n")
    recap.append("## Route\n")
    for key in ["route", "exploration_promotion_allowed", "official_promotion_allowed", "official_kan_carrier_promotion_allowed", "scientific_claim_allowed", "blocking_metric", "minimum_effective_progress", "next_codex_action"]:
        recap.append(f"- {key}: `{decision.get(key, '')}`")
    recap.append(f"- results bundle: `{bundle}`")
    recap.append(f"- code review packet: `{packet}`\n")

    recap.append("## 0. Repair and GPU audit\n")
    recap.append("- S0 fixes: narrowed semantic false-positive matching, added clean-packet source dependencies, and marked norm-capped operator laws as nonlinear instead of claiming linearity.")
    recap.append("- Operator core fixes: made control-null/random helpers row-permutation equivariant and disabled row smoothing where it broke the law test.")
    recap.append("- CUDA bug fix: changed `dgkan/fu/constructive_commit.py` random target generation to create CPU RNG samples then move them to the target device, fixing the MLP+FU CUDA mismatch in task readback.")
    recap.append("- Horizon protocol fix: replaced the fixed single-update diagnostic with per-adapter `T(delta)` commits, then aborted the CPU run and reran the official matrix on `cuda:1`; CPU horizon artifacts are diagnostic only.")
    recap.append("- KAN GPU fix: replaced the inherited CPU-only v22.12 corrected-layout probe with a v22.13 local GPU probe and ran K15/K16 on `cuda:2`.")
    recap.append("- Norm-scale plumbing fix: made `--norm-scale` affect S3 selected target and per-adapter horizon `T(delta)` instead of silently falling back to `0.16`.")
    recap.append("- Repair sweep fix: added `diagnostic_only=1` GPU preview runner for selected horizons; the first attempt failed because it reused the full h3200/h4800 evaluator, and the failure is recorded in the execution log.")
    recap.append("- Row-axis control-null fix: removed `row_axis_control` from the default control span because it erased PairwiseRanking/Preference row-wise cotangents; row-axis remains available only for diagnostics.")
    recap.append("- LIO7 diagnostic operator: added `LIO7_LowNDSControlNull` to test the planned LowNDS + control-null repair direction; it is tracked in S0/S2 law tests but not promoted unless full horizon gates pass.")
    recap.append("- Source-anchor retention fix: added a role-blind per-variant displacement anchor loss in the horizon training loop (`retention_weight`, `retention_start_step`, `retention_stop_step`); FU and controls each retain their own initial source anchor, so controls do not inherit the FU target.")
    recap.append("- Finalizer promotion semantics fix: split official loss-interface operator promotion from KAN carrier promotion, matching the plan's separate 14.2 and 14.3 gates.")
    recap.append("- Native fused cotangent fix: split D-FOU/D-CHE fused CE backward into generic `backward_from_grad_logits` paths, routed v22.13 efficiency through PrimitiveKAN fused forward caches, and replaced hardcoded gradcheck with native-vs-autograd gradient relative-error checks.")
    recap.append("- KAN basis-channel repair: added `basis_linearized_w1` KAN commits and gain sweep (`1/4/16`) to test whether source retention can be carried by basis parameters instead of corrected readout replay.")
    recap.append("- KAN source-anchor repair: added per-variant KAN displacement anchor retention for basis commits (`retention_weight`, `retention_start_step`, `retention_stop_step`); FU and controls each retain their own initial displacement anchor.")
    recap.append("- Honesty guard: no task/eval/readback metric is used for direction; no autograd-native row is promoted as official fused kernel when arbitrary-cotangent fused backward is missing.")
    recap.append(f"- Latest repair sweep route: `{repair_route.get('route', '')}`; preview_pass_rows={repair_route.get('preview_pass_rows', '')}; best_candidate=`{repair_route.get('best_candidate', '')}`; diagnostic_only={repair_route.get('diagnostic_only', '')}。")
    recap.append(md_table(gpu_rows, ["evidence_id", "scope", "gpu", "pid", "used_memory_mib", "evidence_source", "status", "note"], max_rows=40))
    recap.append(md_table(repair_rows, ["operator_id", "norm_scale", "loss_adapter_name", "attempt", "device", "source_func_h100", "source_func_h800", "source_func_h1600", "source_loss_h1600", "repair_sweep_preview_pass", "diagnostic_only"], max_rows=80))
    recap.append("\n")

    recap.append("## 1. Code truth\n")
    recap.append(f"- S0 route: `{routes['s0'].get('route', '')}`；S0_pass={routes['s0'].get('S0_pass', '')}。")
    recap.append(f"- role-blind static/runtime: static={routes['s0'].get('role_blind_static_pass', '')}, semantic={routes['semantic'].get('semantic_pass', '')}。")
    recap.append(f"- corrected layout pass: {routes['s0'].get('layout_unit_test_pass', '')}。")
    recap.append(md_table(truth, ["check", "pass", "metric", "value", "blocker"], max_rows=40))
    recap.append(md_table(firewall, ["check", "adapter_name_branch_count", "official_core_loss_formula_branch_count", "O10_diagnostic_only_pass", "official_operator_role_blind_static_pass", "blocker"], max_rows=10))
    recap.append(md_table(renaming, ["operator_id", "adapter_renaming_output_cosine", "adapter_renaming_output_rel_error", "adapter_renaming_pass"], max_rows=10))
    recap.append(md_table(law, ["operator_id", "operator_class", "operator_linearity_error", "operator_homogeneity_error", "operator_lipschitz_ratio", "cotangent_permutation_equivariance_error", "operator_law_pass"], max_rows=10))
    recap.append(md_table(layout, ["carrier", "legacy_layout_fit_cosine", "legacy_layout_projection_residual", "corrected_layout_fit_cosine", "corrected_layout_projection_residual", "legacy_vs_corrected_gap_detected", "layout_unit_test_pass"], max_rows=10))
    recap.append("Analysis: S0 只确认代码/语义/layout 是否可进入实验；所有 task/loss adapter formula 只在 LossInterface 或实验矩阵内使用，official operator core 不读取 adapter 名称来生成方向。\n")

    recap.append("## 2. Efficiency\n")
    recap.append(f"- official_fused_kernel_complete_rows: {routes['eff'].get('official_fused_kernel_complete_rows', '')}; manual_upstream_vjp_rows: {routes['eff'].get('manual_upstream_vjp_rows', '')}。")
    recap.append(md_table(eff_summary, ["carrier", "profile_rows", "native_E1_pass_rows", "official_fused_kernel_complete_rows", "manual_upstream_vjp_rows", "best_operator_step_ratio", "best_full_step_ratio", "best_memory_ratio", "decision", "blocker"], max_rows=20))
    recap.append(md_table(eff_rows, ["carrier", "variant", "cotangent_type", "batch_size", "forward_ratio_vs_mlp", "vjp_ratio_vs_mlp", "operator_step_ratio_vs_mlp", "full_step_ratio_vs_mlp", "official_fused_kernel_complete", "native_status_reason"], max_rows=40))
    recap.append("Analysis: D-FOU/D-CHE 已从 label-only fused backward 修复为 generic output-cotangent fused VJP；official_fused rows 只在实际 fused path 与 native-vs-autograd gradcheck 通过后计数。D-RBF/D-RAT 仍保持 blocked，不借 D-FOU/D-CHE 的成功。\n")

    recap.append("## 3. Operator\n")
    recap.append(f"- S2/S3/S4 pass rows: {routes['atom'].get('S2_operator_atom_pass_rows', '')} / {routes['solve'].get('S3_operator_variational_pass_rows', '')} / {routes['commit'].get('S4_operator_metric_commit_pass_rows', '')}。")
    recap.append(md_table(atom_rows, ["operator_id", "operator_family", "gain_positive_fraction", "control_projection_after", "NDS_reduction", "operator_linearity_error", "operator_homogeneity_error", "S2_operator_atom_pass", "blocker"], max_rows=20))
    recap.append(md_table(commit_rows, ["solver_level", "block_role", "projection_residual_Gf", "ActuationR2", "B2_transfer_gain", "function_displacement_cos_with_target", "S4_operator_metric_commit_pass", "blocker"], max_rows=20))
    recap.append("Analysis: operator rows 是 true `cotangent -> Delta_f` role-blind atoms；没有把 O10 diagnostic consensus 算作 official success。\n")

    recap.append("## 4. Adapter robustness\n")
    recap.append(f"- C3/C4 adapter pass count: {routes['horizon'].get('C3_adapter_pass_count', '')} / {routes['horizon'].get('C4_adapter_pass_count', '')}。")
    recap.append(f"- C3 list: `{routes['horizon'].get('C3_adapter_pass_list', '')}`；C4 list: `{routes['horizon'].get('C4_adapter_pass_list', '')}`。")
    recap.append(f"- Latest official horizon used norm_scale={routes['horizon'].get('norm_scale', '')}; route=`{routes['horizon'].get('route', '')}`; blocker=`{routes['horizon'].get('blocker', '')}`。")
    recap.append(md_table(horizon_rows, ["loss_adapter_name", "attempt", "device", "source_func_h100", "source_func_h800", "source_func_h3200", "source_func_h4800", "source_loss_h3200", "source_loss_h4800", "C3_source_formation_pass", "C4_terminal_retention_pass", "TargetRetentionOnly_NotTaskUseful", "blocker"], max_rows=60))
    recap.append("Analysis: source-anchor retention 后，正式 GPU horizon 从 NoGo 推进到 `S5-RoleBlindOperatorHorizonPass`：CE/MSE/Ranking/Preference-smoke 均 C3，CE/Ranking/Preference-smoke 均 C4；MSE source_loss 到 h6400 仍为正但 C4 终端保留阈值未过。StableRandom/RandomMatched 未通过，说明 source gate 未被 random controls 污染。\n")

    recap.append("## 5. KAN carrier\n")
    recap.append(f"- KAN route: `{routes['kan'].get('route', '')}`；DFOU source={routes['kan'].get('DFOU_corrected_source_exists', '')}, DCHE source={routes['kan'].get('DCHE_corrected_source_exists', '')}, DCHE efficiency={routes['kan'].get('DCHE_efficiency_pass', '')}。")
    recap.append(md_table(kan_rows, ["carrier", "attempt", "device", "update_gain", "retention_weight", "basis_estimate_fit_cosine", "basis_commit_projection_residual", "KAN_source_func_h3200", "KAN_source_func_h4800", "KAN_source_loss_h3200", "KAN_source_loss_h4800", "carrier_specific_efficiency_pass", "KAN_source_channel_decision", "blocker"], max_rows=20))
    recap.append(md_table(k17_rows, ["carrier", "commit_channel", "readout_channel_energy", "basis_channel_energy", "basis_to_readout_energy_ratio", "K17_basis_channel_pass", "blocker"], max_rows=20))
    recap.append("Analysis: corrected layout、basis-linearized gain sweep、以及 KAN source-anchor retention 均为 operator/efficiency pass 后的 GPU 重跑。Readout-only rows 有弱 positive source 但 basis_channel_energy=0；basis_linearized_w1 rows 有 basis_channel_energy=1，但若 source_func 仍远低于 MLP same-adapter baseline 或 source_loss 不能守住，就不能写成 KAN carrier success。当前 no-go 精确定位为 KAN basis-channel source mismatch。\n")

    recap.append("## 6. Task-level value\n")
    recap.append(f"- task route: `{routes['task'].get('route', '')}`；rows={routes['task'].get('task_rows', '')}; KAN_FU_accuracy_ge_MLP_rows={routes['task'].get('KAN_FU_accuracy_ge_MLP_rows', '')}。")
    recap.append(md_table(task_rows, ["dataset", "seed", "variant", "final_train_loss", "final_test_loss_readback", "final_train_accuracy", "final_test_accuracy_readback", "NLL_delta_vs_MLP", "AUC_loss_time_ratio_vs_best_control", "ECE_delta", "Brier_delta", "blocker"], max_rows=80))
    recap.append("Analysis: task metrics 是 readback/gate，不参与 direction。由于本轮 official operator/KAN/fused gates 未全部成立，task improvement 即便出现也不能写成 FU mechanism proof。\n")

    recap.append("## 7. Execution and artifacts\n")
    recap.append(md_table(command_rows, ["timestamp", "command", "status", "note"], max_rows=80))
    recap.append(md_table(read_rows(out_dir / "v22_13_artifact_index.csv"), ["artifact", "exists", "size_bytes", "sha256"], max_rows=80))
    recap.append("Next action: FU operator 与 D-FOU/D-CHE native fused arbitrary-cotangent efficiency 已过；下一 blocker 是 KAN basis-channel carrier geometry，需要新的 basis-state objective/architecture，而不是继续 readout replay 或无界 gain sweep。\n")

    V2213_RECAP_DOC.write_text("\n".join(recap) + "\n", encoding="utf-8")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_13_finalize.py --out-dir {out_dir}", status="completed", note=f"route={decision['route']} exploration={decision['exploration_promotion_allowed']} official={decision['official_promotion_allowed']} blocker={decision.get('blocking_metric')}")
    write_rows(out_dir / "v22_13_artifact_index.csv", artifact_index(out_dir))
    build_code_review_packet(out_dir)
    build_results_bundle(out_dir)


if __name__ == "__main__":
    main()
