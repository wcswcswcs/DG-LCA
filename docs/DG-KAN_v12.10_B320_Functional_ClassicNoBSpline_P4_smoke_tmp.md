# DG-KAN v12.10 B320 Functional ClassicNoBSpline 执行复盘

> 本复盘由 `run_v1210_b320_functional_classic_nobspline.py` 生成。所有数值来自本轮 `results` artifact 或显式复用的 raw artifact；没有编造数据。B-spline 在本版按计划冻结，不作为 active family 候选。

## 0. 当前结论

```text
route = R5-B320P4ShortRunFailed
raw_runner_returncode = 0
raw_dir = results/v12_10_b320_functional_classic_nobspline/v1210_b320_f14_backtrack_20260522T000000Z/raw_v1283_reuse
B320_v1210_strict_hardening_pass = 1
B320_strict_fail_reason = 
B320_F3_step_ratio_q90 = 0.7938351796804363
B320_F3_memory_ratio_q90 = 0.1295238095238095
B320_mean_delta = 0.022526041666666666
B320_worst_delta = 0.0
B320_near_pass_rate = 1.0
B320_max_ECE_delta = 0.019615232944488525
B320_max_AUC_step = 0.9809975399787746
B320_max_AUC_time = 0.9804242794311487
B320_LineC_nontearing_pass = 1
official_functional_success = 0
functional_P3_official_candidate_pass = 1
functional_P3_best_update = F14d-OrthDirectBranchDampingOnly
functional_P3_control_gap = 0.2704752677133553
functional_P3_fail_reason = 
classic_family_pass_count_excluding_bspline = 0
BSpline.status = FamilyFrozen_KernelBlocked_RejectedForThisVersion
next_recommended_action = autopsy P4 short-run failure; continue plan-consistent functional or classic-family repair without claiming official functional success
artifact_dir = /workspace/DG-LCA/DG-LCA/results/v12_10_b320_functional_classic_nobspline/v1210_p4_smoke_20260522T000000Z
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
| `docs/DG-KAN_v12.10_B320_Functional_ClassicNoBSpline_执行复盘.md` | 新增本复盘文件 | 只记录真实落盘 artifact 的关键数值、blocker 和后续动作；缺失项写 not_measured 或沿用 raw runner 返回状态，不补造数据。 |

## 2. B320 F3 Efficiency

| candidate_id | implementation_id | step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
| --- | --- | --- | --- | --- |
| `B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075` | `F3-triton-workspace-forward-delta-readout-proj-grad-learnableP` | `0.7938351796804363` | `0.1295238095238095` | `1` |

Hardening gate summary:

| candidate_id | impl_path | step_ratio_q90 | memory_ratio_q90 | mean_delta | worst_delta | near_pass_rate | AUC_step_ratio | AUC_time_ratio | ECE_delta | strict_pass | strict_fail_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075` | `F3-triton-workspace-forward-delta-readout-proj-grad-learnableP` | `0.7938351796804363` | `0.1295238095238095` | `0.022526041666666666` | `0.0` | `1.0` | `0.9809975399787746` | `0.9804242794311487` | `0.019615232944488525` | `1` | `` |

## 3. B320 Task / AUC

| stage | dataset | seed | mean_delta | worst_delta | near_pass_rate | max_ECE_delta | max_AUC_step | max_AUC_time | val_acc_delta_vs_mlp | ECE_delta_vs_mlp | AUC_step_ratio | AUC_time_ratio | A5_autopsy_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `V1283_B109_AUC_ATTRIBUTION_SUMMARY` | `` | `` | `0.022526041666666666` | `0.0` | `1.0` | `` | `` | `` | `` | `` | `` | `` | `1` |

## 4. Family Policy / Status

## 4. Functional P3 Gate

| candidate_id | best_control | best_control_delta_score | best_functional_update | best_functional_delta_score | control_gap | CouplingR2_best_control | CouplingR2_after | CouplingR2_delta_vs_best_control | NoiseSignalLeak_delta | official_candidate_pass | fail_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075` | `C0-TaskOnlyAdamW` | `0.112838347053971` | `F14d-OrthDirectBranchDampingOnly` | `0.3833136147673263` | `0.2704752677133553` | `0.11466433534666509` | `0.33411868278337853` | `0.21945434743671344` | `-0.021574735641479492` | `1` | `` |

## 4.1 Functional P4 Short Run

| method | dataset | seed | acc_delta_vs_B320 | AUC_time_delta_vs_B320 | ECE_delta_vs_B320 | CEp99_delta_vs_B320 | CouplingR2_delta_vs_B320 | NoiseSignalLeak_delta_vs_B320 | amortized_overhead_ratio | control_gap_vs_best | strict_pass | fail_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `B320-AdamW` | `MNIST` | `0` | `0.0` | `0.0` | `0.0` | `0.0` | `1.0` | `0.04832930862903595` | `1.0` | `0.29533170346830806` | `0` | `` |
| `B320-NoOpMatchedOverhead` | `MNIST` | `0` | `0.0` | `-239.51713540958812` | `0.0` | `0.0` | `0.7046682071247248` | `0.04832921922206879` | `0.011573360293580498` | `0.0` | `0` | `` |
| `B320-RandomMatchedNorm` | `MNIST` | `0` | `-0.0078125` | `-240.25073084490188` | `-0.002646327018737793` | `-0.055325984954833984` | `0.3312186766314066` | `0.04267725721001625` | `0.008499846367578357` | `-0.3799582869222752` | `0` | `` |
| `B320-AdamWParallelMaintenance` | `MNIST` | `0` | `0.0` | `-240.51364932225948` | `-1.9490718841552734e-05` | `0.00013589859008789062` | `0.0005297428787102154` | `0.04833012819290161` | `0.007461291517372637` | `-0.7041393732168474` | `0` | `` |
| `B320-SNR-only` | `MNIST` | `0` | `0.0` | `-240.0122391500749` | `4.76837158203125e-07` | `6.67572021484375e-06` | `1.264024618952142e-05` | `0.048331014811992645` | `0.009530168260327035` | `-0.7046598658635397` | `0` | `` |
| `B320-bestFunctional` | `MNIST` | `0` | `0.0` | `-238.245926418103` | `0.02059304714202881` | `-0.296722412109375` | `0.3497443983707694` | `0.062232863157987595` | `0.016467032791446345` | `-0.40655296090291737` | `0` | `ECE_delta>0.005;NoiseSignalLeak_delta>-0.01;control_gap_vs_best<=0` |
| `MLP-AdamW` | `MNIST` | `0` | `-0.3515625` | `-222.78317655295814` | `0.10621118545532227` | `-1.4356510639190674` | `0.31303166035437235` | `0.6429837346076965` | `0.036080336027433775` | `-2.432881724158544` | `0` | `` |
| `MLP-analogFunctional` | `MNIST` | `0` | `-0.3515625` | `-239.34636372784897` | `0.1062127947807312` | `-1.4356048107147217` | `0.31303264767120464` | `0.6429560780525208` | `0.005494209326950406` | `-2.4328490867753354` | `0` | `` |

## 5. Family Policy / Status

| family | status | active_followup | codex_budget | best_L3_manual_step_ratio | blocker |
| --- | --- | --- | --- | --- | --- |
| `Chebyshev` | `ExpressionBlocked` | `` | `` | `0.6858152590294013` | `family-specific fused L3 efficiency passed but A4 expression failed` |
| `Fourier` | `ExpressionBlocked` | `` | `` | `1.0140202390867186` | `family-specific fused L3 efficiency passed but A4 expression failed` |
| `RBF` | `KernelBlocked` | `` | `` | `0.9889286521226709` | `L3 manual analytic attempted; fused family-specific backward/update kernel missing` |
| `Rational` | `TaskBlocked` | `` | `` | `0.8001309156911405` | `family-specific fused L3 efficiency and A4 expression passed; A5 fused task triage failed` |
| `Wavelet` | `KernelBlocked` | `` | `` | `1.849628978261043` | `L3 manual analytic attempted; fused family-specific backward/update kernel missing` |
| `BSpline` | `FamilyFrozen_KernelBlocked_RejectedForThisVersion` | `0` | `0` | `` | `excluded from active v12.10 follow-up by plan; no B-spline candidates scheduled in this run` |

## 6. Provenance / Hash

| stage | artifact | rows_checked | fake_data_used_sum | proxy_row_used_sum | cpu_offload_used_sum | no_fake_pass |
| --- | --- | --- | --- | --- | --- | --- |
| `V1210_PROVENANCE_AUDIT` | `v1210_route_config.json` | `1` | `0` | `0` | `0` | `1` |
| `V1210_PROVENANCE_AUDIT` | `v1210_family_policy.json` | `10` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_auc_attribution.csv` | `62` | `0` | `0` | `0` | `1` |
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_auc_autopsy_trace.csv` | `270` | `0` | `0` | `0` | `1` |
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
| `V1283_PROVENANCE_AUDIT` | `v1283_b109_task_autopsy_final.csv` | `90` | `0` | `0` | `0` | `1` |

Selected hashes:

| artifact | sha256_prefix |
| --- | --- |
| `v1210_b320_auc_attribution.csv` | `74e7b8490d09` |
| `v1210_b320_base_hardening.csv` | `43c17b712596` |
| `v1210_b320_efficiency_profile.csv` | `2cbe001df86f` |
| `v1210_b320_task_final.csv` | `ea713160fd63` |
| `v1210_b320_task_trace.csv` | `1bb5ac1ce83f` |
| `v1210_family_component_profile.csv` | `bd15b7cc5526` |
| `v1210_family_efficiency.csv` | `b66b8094da0a` |
| `v1210_family_expression.csv` | `82757bd7a585` |
| `v1210_family_failure_table.csv` | `1ff709326dda` |
| `v1210_family_gradcheck.csv` | `4bc639c1996e` |
| `v1210_family_linec_diagnostics.csv` | `c83a5b5edea5` |
| `v1210_family_manifest_raw.csv` | `8fec05e40023` |
| `v1210_family_policy.csv` | `d37e4d2f6018` |
| `v1210_family_policy.json` | `85450d757a0b` |
| `v1210_family_status.json` | `1e1051526fac` |
| `v1210_family_task_triage.csv` | `b02ee518e18f` |
| `v1210_functional_delta_score_repair_summary.csv` | `9a0c9310ff60` |
| `v1210_functional_one_five_step.csv` | `a3c3beaf0408` |
| `v1210_functional_p3_gate.csv` | `a41ef210f95a` |
| `v1210_functional_reentry_summary.csv` | `005d36fc4e88` |

## 7. 分析结论

1. B-spline 已按 v12.10 计划从 active classic family 中排除；本轮没有调度 B-spline 候选，因此不能把 B-spline 写成新失败实验，只能写成 `FamilyFrozen_KernelBlocked_RejectedForThisVersion`。
2. B320 hardening 是否通过只看 `v1210_route_decision.json` 中的 strict gates 聚合；如果 raw runner 未完成或任一 gate 超限，本复盘不写 official base lock。
3. Functional official success 仍以 raw strong-control artifact 与 `v1210_functional_p3_gate.csv` 为准；没有通过 strong controls 与 Line C gate 前不写成功。
4. 后续修复方向必须沿计划继续：先修 B320 strict hardening 或 Line C nontearing blocker，再进入 functional official re-entry；classic family 只在非 B-spline active families 内推进。
