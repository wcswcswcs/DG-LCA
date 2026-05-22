# DG-KAN v12.10 B320 Functional ClassicNoBSpline 执行复盘

> 本复盘由 `run_v1210_b320_functional_classic_nobspline.py` 生成。所有数值来自本轮 `results` artifact 或显式复用的 raw artifact；没有编造数据。B-spline 在本版按计划冻结，不作为 active family 候选。

## 0. 当前结论

```text
route = R2-B320BasePassLineCNeedsRepair
raw_runner_returncode = 0
raw_dir = results/v12_10_b320_functional_classic_nobspline/v1210_family_cheb_fourier_extended_20260522T000000Z/raw_v1283_reuse
B320_v1210_strict_hardening_pass = 1
B320_strict_fail_reason = 
B320_F3_step_ratio_q90 = 0.7191994446336335
B320_F3_memory_ratio_q90 = 0.1295238095238095
B320_mean_delta = 0.0625
B320_worst_delta = 0.0625
B320_near_pass_rate = 1.0
B320_max_ECE_delta = -0.15041980147361755
B320_max_AUC_step = 0.6276671677833036
B320_max_AUC_time = 0.6276671677833036
B320_LineC_nontearing_pass = 0
official_functional_success = 0
functional_P3_official_candidate_pass = 1
functional_P3_best_update = F14-OrthBranchScaleDampingOnly
functional_P3_control_gap = 0.32235952500038023
functional_P3_fail_reason = 
classic_family_pass_count_excluding_bspline = 1
BSpline.status = FamilyFrozen_KernelBlocked_RejectedForThisVersion
next_recommended_action = repair train-probe coupling/noise leakage before functional official success claim
artifact_dir = /workspace/DG-LCA/DG-LCA/results/v12_10_b320_functional_classic_nobspline/v1210_family_cheb_fourier_extended_20260522T000000Z
```

## 1. 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 新增 v12.10 wrapper | 固化 B320 exact/B321/B314/B109 对照协议，调用真实 raw runner，生成 v1210 route/policy/provenance/hash/复盘，不把 raw v1283 字段直接冒充 v12.10 official 结论。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 `F11-TaskMinusBranchScaleDamping` functional diagnostic bracket | 在 F10 branch damping 未通过 B320 P3 gate 后，增加 role-wise 反向 bracket，检查方向符号是否导致 CouplingR2/control gap 失败；不改 CE loss、sampler/class weight、teacher/distillation 或 dataset branch。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 修正 delta-score reservoir 项并新增 `F12/F13` negative geometry-only diagnostics | 按 v12.10 P3 的 reservoir-release 定义，把 score 从奖励 RealSignalReservoirRatio 增加改为奖励 reservoir release；新增负向 SNR/orthogonal geometry-only bracket，用于确认是否存在不依赖 AdamW 的 reservoir-release 方向。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 `F14/F15` AdamW-orthogonal branch diagnostics | 将 branch-scale damping residual 显式去除与 task/AdamW step 的平行分量，测试 role-wise orthogonal functional 是否能越过 C0/C3 controls；仍不改 loss、数据、采样或标签权重。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | P3 functional backtracking 增加 holdout safety lower bound | F14 暴露强信号但一步过强后，按 v12.10 F-F4 推荐方向加入 norm-budget downscale：functional 只接受 `0.95 <= holdout_loss_after/before <= 1.05`；controls 保持原审计语义。 |
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 新增 P4 short-run candidate loop | P3 打开后可用 `--run-p4-short 1` 运行 B320/FHQ/manual-AdamW 短跑；B320 路径不使用 `loss.backward()` 更新，functional 事件只作为训练内维护事件，并与 NoOp/Random/AdamWParallel/SNR controls 对比。 |
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 新增 P4 fixed-lambda runtime-cost repair | 当 P4 runtime backtracking 因 deepcopy/holdout evaluation 触发 overhead blocker 时，可用 `--p4-runtime-backtracking 0 --p4-fixed-lambda ...` 运行固定步长维护事件；这是成本修复，不改变数据、loss、label、sampler 或 gate。 |
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 新增 P4 branch-mode bracket | `--p4-functional-mode` 支持 direct/quad/both/negative/F15-style blend，用同一 short-run gate 检查 P3 方向是否只是 branch role 选择错误；不改变数据、loss、sampler 或 gate。 |
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 扩展 v12.10 非 B-spline active family 候选 | 将已实现的 Chebyshev K4/localrot2 与 Fourier h8 stronger linear residual 候选加入 active list，按同一 L3/A4 gate 验证 ExpressionBlocked 是否可修复；仍排除 B-spline。 |
| `docs/DG-KAN_v12.10_B320_Functional_ClassicNoBSpline_执行复盘.md` | 新增本复盘文件 | 只记录真实落盘 artifact 的关键数值、blocker 和后续动作；缺失项写 not_measured 或沿用 raw runner 返回状态，不补造数据。 |

## 2. B320 F3 Efficiency

| candidate_id | implementation_id | step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
| --- | --- | --- | --- | --- |
| `B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075` | `F3-triton-workspace-forward-delta-readout-proj-grad-learnableP` | `0.7191994446336335` | `0.1295238095238095` | `1` |

Hardening gate summary:

| candidate_id | impl_path | step_ratio_q90 | memory_ratio_q90 | mean_delta | worst_delta | near_pass_rate | AUC_step_ratio | AUC_time_ratio | ECE_delta | strict_pass | strict_fail_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075` | `F3-triton-workspace-forward-delta-readout-proj-grad-learnableP` | `0.7191994446336335` | `0.1295238095238095` | `0.0625` | `0.0625` | `1.0` | `0.6276671677833036` | `0.6276671677833036` | `-0.15041980147361755` | `1` | `` |

## 3. B320 Task / AUC

| stage | dataset | seed | mean_delta | worst_delta | near_pass_rate | max_ECE_delta | max_AUC_step | max_AUC_time | val_acc_delta_vs_mlp | ECE_delta_vs_mlp | AUC_step_ratio | AUC_time_ratio | A5_autopsy_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `V1283_B109_AUC_ATTRIBUTION_SUMMARY` | `` | `` | `0.0625` | `0.0625` | `1.0` | `` | `` | `` | `` | `` | `` | `` | `1` |

## 4. Family Policy / Status

## 4. Functional P3 Gate

| candidate_id | best_control | best_control_delta_score | best_functional_update | best_functional_delta_score | control_gap | CouplingR2_best_control | CouplingR2_after | CouplingR2_delta_vs_best_control | NoiseSignalLeak_delta | official_candidate_pass | fail_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075` | `C0-TaskOnlyAdamW` | `0.14382719965480906` | `F14-OrthBranchScaleDampingOnly` | `0.4661867246551893` | `0.32235952500038023` | `0.14573264093898874` | `0.28490542625621773` | `0.139172785317229` | `-0.5948874950408936` | `1` | `` |

## 4.1 Functional P4 Short Run

| method | dataset | seed | acc_delta_vs_B320 | AUC_time_delta_vs_B320 | ECE_delta_vs_B320 | CEp99_delta_vs_B320 | CouplingR2_delta_vs_B320 | NoiseSignalLeak_delta_vs_B320 | amortized_overhead_ratio | control_gap_vs_best | strict_pass | fail_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `` | `` | `` | `` | `` | `` | `` | `` | `` | `` | `` | `0` | `` |

## 5. Family Policy / Status

| family | status | active_followup | codex_budget | best_L3_manual_step_ratio | blocker |
| --- | --- | --- | --- | --- | --- |
| `Chebyshev` | `ExpressionBlocked` | `` | `` | `0.6521527773287539` | `family-specific fused L3 efficiency passed but A4 expression failed` |
| `Fourier` | `ExpressionBlocked` | `` | `` | `0.7572990222904608` | `family-specific fused L3 efficiency passed but A4 expression failed` |
| `RBF` | `KernelBlocked` | `` | `` | `1.0596900696617313` | `L3 manual analytic attempted; fused family-specific backward/update kernel missing` |
| `Rational` | `FamilyPass` | `` | `` | `0.8075464622174309` | `family-specific fused L3 efficiency, A4 expression, and A5 task passed; family Line C/Functional route is still not integrated` |
| `Wavelet` | `KernelBlocked` | `` | `` | `1.858184210046784` | `L3 manual analytic attempted; fused family-specific backward/update kernel missing` |
| `BSpline` | `FamilyFrozen_KernelBlocked_RejectedForThisVersion` | `0` | `0` | `` | `excluded from active v12.10 follow-up by plan; no B-spline candidates scheduled in this run` |

## 6. Provenance / Hash

| stage | artifact | rows_checked | fake_data_used_sum | proxy_row_used_sum | cpu_offload_used_sum | no_fake_pass |
| --- | --- | --- | --- | --- | --- | --- |
| `V1210_PROVENANCE_AUDIT` | `v1210_route_config.json` | `1` | `0` | `0` | `0` | `1` |
| `V1210_PROVENANCE_AUDIT` | `v1210_family_policy.json` | `15` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_auc_attribution.csv` | `4` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_auc_autopsy_trace.csv` | `3` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_fullstep_profile.csv` | `6` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_functional_control_matrix.csv` | `4` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_functional_delta_score_control_matrix.csv` | `3` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_functional_delta_score_repair.csv` | `99` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_functional_delta_score_repair_summary.csv` | `1` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_functional_direction_audit.csv` | `28` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_functional_five_step.csv` | `28` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_functional_one_step.csv` | `28` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_functional_route.json` | `0` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_fused_backward_correctness.csv` | `20` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_linec_coupling.csv` | `5` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_linec_diagnostics.csv` | `5` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_linec_functional_reentry_summary.csv` | `1` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_linec_noise_leak.csv` | `5` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_linec_signal_reservoir.csv` | `5` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_task_autopsy_final.csv` | `3` | `0` | `0` | `0` | `1` |

Selected hashes:

| artifact | sha256_prefix |
| --- | --- |
| `v1210_b320_auc_attribution.csv` | `b74e8fcf090f` |
| `v1210_b320_base_hardening.csv` | `2840da6fa651` |
| `v1210_b320_efficiency_profile.csv` | `189259c61c55` |
| `v1210_b320_task_final.csv` | `67a9f24b381a` |
| `v1210_b320_task_trace.csv` | `00e0b58896d2` |
| `v1210_family_component_profile.csv` | `1520941ffd74` |
| `v1210_family_efficiency.csv` | `a14f05cc4f31` |
| `v1210_family_expression.csv` | `34f726e3fecf` |
| `v1210_family_failure_table.csv` | `541f92234dbe` |
| `v1210_family_gradcheck.csv` | `4b21fb85133b` |
| `v1210_family_linec_diagnostics.csv` | `e6e715b0734a` |
| `v1210_family_manifest_raw.csv` | `8fec05e40023` |
| `v1210_family_policy.csv` | `b43640f47ce4` |
| `v1210_family_policy.json` | `6fa077ee3ab7` |
| `v1210_family_status.json` | `8511bc36f183` |
| `v1210_family_task_triage.csv` | `d8b41dce5c5d` |
| `v1210_functional_delta_score_repair_summary.csv` | `67f44be68665` |
| `v1210_functional_one_five_step.csv` | `236d5b632df7` |
| `v1210_functional_p3_gate.csv` | `a590650744af` |
| `v1210_functional_reentry_summary.csv` | `9622ca538979` |

## 7. 分析结论

1. B-spline 已按 v12.10 计划从 active classic family 中排除；本轮没有调度 B-spline 候选，因此不能把 B-spline 写成新失败实验，只能写成 `FamilyFrozen_KernelBlocked_RejectedForThisVersion`。
2. B320 hardening 是否通过只看 `v1210_route_decision.json` 中的 strict gates 聚合；如果 raw runner 未完成或任一 gate 超限，本复盘不写 official base lock。
3. Functional official success 仍以 raw strong-control artifact 与 `v1210_functional_p3_gate.csv` 为准；没有通过 strong controls 与 Line C gate 前不写成功。
4. 后续修复方向必须沿计划继续：先修 B320 strict hardening 或 Line C nontearing blocker，再进入 functional official re-entry；classic family 只在非 B-spline active families 内推进。
