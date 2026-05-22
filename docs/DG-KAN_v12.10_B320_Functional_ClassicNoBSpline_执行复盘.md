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
artifact_dir = /workspace/DG-LCA/DG-LCA/results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_taskminusquad010_lambda003125_event24_20260522T000000Z
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
| `B320-AdamW` | `MNIST` | `0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `1.0` | `-3.216040118891872e-05` | `0` | `` |
| `B320-NoOpMatchedOverhead` | `MNIST` | `0` | `0.0` | `-15.267892042066746` | `-7.599592208862305e-07` | `-3.3855438232421875e-05` | `3.226440836290312e-05` | `4.470348358154297e-06` | `0.7097697281927482` | `-4.366341184169897e-06` | `0` | `` |
| `B320-RandomMatchedNorm` | `MNIST` | `0` | `0.0` | `-15.283497329748714` | `-1.4901161193847656e-08` | `9.5367431640625e-07` | `7.150357361762616e-08` | `0.0` | `0.680532467602672` | `-3.211869993768879e-05` | `0` | `` |
| `B320-AdamWParallelMaintenance` | `MNIST` | `0` | `0.0` | `-15.293734822108583` | `-3.2782554626464844e-07` | `-1.3828277587890625e-05` | `3.4216761433669696e-05` | `2.0563602447509766e-06` | `0.6817863207477214` | `0.0` | `0` | `` |
| `B320-SNR-only` | `MNIST` | `0` | `0.0` | `-15.27466691119886` | `1.043081283569336e-07` | `-1.1920928955078125e-05` | `1.1284962223578887e-05` | `2.2351741790771484e-06` | `0.7123906776864181` | `-2.3274525917549305e-05` | `0` | `` |
| `B320-bestFunctional` | `MNIST` | `0` | `0.0` | `-15.293270443840393` | `-1.0132789611816406e-06` | `-6.008148193359375e-05` | `3.183268860718069e-05` | `5.602836608886719e-06` | `0.6794223515471006` | `-5.930549190624745e-06` | `0` | `CouplingR2_delta<0.02;NoiseSignalLeak_delta>-0.01;control_gap_vs_best<=0` |
| `MLP-AdamW` | `MNIST` | `0` | `-0.015625` | `-14.696317379684736` | `0.008287623524665833` | `1.1600863933563232` | `0.3403713370477618` | `-0.1952829360961914` | `0.9230357883175431` | `0.2613421457923477` | `0` | `` |
| `MLP-analogFunctional` | `MNIST` | `0` | `-0.015625` | `-15.195040020043685` | `0.008287683129310608` | `1.1600849628448486` | `0.34037159940218387` | `-0.1952831894159317` | `0.9217357785179171` | `0.2613421697281907` | `0` | `` |
| `B320-AdamW` | `MNIST` | `1` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `1.0` | `0.06031945150427015` | `0` | `` |
| `B320-NoOpMatchedOverhead` | `MNIST` | `1` | `0.0` | `0.061388063559470796` | `-1.6391277313232422e-07` | `-1.8596649169921875e-05` | `8.830045882590731e-06` | `2.562999725341797e-06` | `1.054947778846278` | `-0.0010623450090433972` | `0` | `` |
| `B320-RandomMatchedNorm` | `MNIST` | `1` | `0.0` | `0.0688711381526115` | `0.0` | `-6.9141387939453125e-06` | `7.3722483475702205e-06` | `2.175569534301758e-06` | `1.037320416672045` | `-0.008546489969528087` | `0` | `` |
| `B320-AdamWParallelMaintenance` | `MNIST` | `1` | `0.0` | `0.060329211248082604` | `-1.043081283569336e-07` | `2.2172927856445312e-05` | `9.759743812454147e-06` | `-3.1888484954833984e-06` | `1.0450892511391359` | `0.0` | `0` | `` |
| `B320-SNR-only` | `MNIST` | `1` | `0.0` | `0.0760137835140691` | `-1.043081283569336e-07` | `-2.2411346435546875e-05` | `1.109083127837085e-05` | `2.8908252716064453e-06` | `1.0216310780808302` | `-0.01568613200379218` | `0` | `` |
| `B320-bestFunctional` | `MNIST` | `1` | `0.0` | `0.05753645091782866` | `-1.043081283569336e-07` | `1.52587890625e-05` | `1.1532602158359495e-05` | `-2.652406692504883e-06` | `1.0104722654283538` | `0.002794533188599846` | `0` | `AUC_time_delta>0;CouplingR2_delta<0.02;NoiseSignalLeak_delta>-0.01` |
| `MLP-AdamW` | `MNIST` | `1` | `-0.0234375` | `0.11959766655858425` | `0.01337142288684845` | `2.137019634246826` | `0.29719538370662457` | `-0.4480555299669504` | `1.3353283002671967` | `0.11340951791931639` | `0` | `` |
| `MLP-analogFunctional` | `MNIST` | `1` | `-0.0234375` | `0.15293130682701153` | `0.013371393084526062` | `2.1370248794555664` | `0.29719523558249306` | `-0.4480554983019829` | `1.3086727614204026` | `0.08007581893372478` | `0` | `` |
| `B320-AdamW` | `MNIST` | `2` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `1.0` | `0.055504995512619115` | `0` | `` |
| `B320-NoOpMatchedOverhead` | `MNIST` | `2` | `0.0` | `0.05550903993222034` | `1.4901161193847656e-08` | `-1.1444091796875e-05` | `5.668646171352876e-06` | `1.6093254089355469e-06` | `1.053563150050007` | `0.0` | `0` | `` |
| `B320-RandomMatchedNorm` | `MNIST` | `2` | `0.0` | `0.06441004882713944` | `2.9802322387695312e-08` | `-4.76837158203125e-06` | `5.2701477749605985e-06` | `5.364418029785156e-07` | `1.0310375158218634` | `-0.008900349410870728` | `0` | `` |
| `B320-AdamWParallelMaintenance` | `MNIST` | `2` | `0.0` | `0.06400417033209738` | `4.470348358154297e-08` | `9.5367431640625e-06` | `5.3791755975085565e-06` | `2.950429916381836e-06` | `1.1322087678918586` | `-0.008496969591215042` | `0` | `` |
| `B320-SNR-only` | `MNIST` | `2` | `0.0` | `0.06837115015433007` | `2.0116567611694336e-07` | `1.0013580322265625e-05` | `3.871460434345941e-06` | `-3.8743019104003906e-07` | `1.0361062330224637` | `-0.012862752567854213` | `0` | `` |
| `B320-bestFunctional` | `MNIST` | `2` | `0.0` | `0.056382693030556236` | `-1.4901161193847656e-08` | `-7.62939453125e-06` | `9.476324756341192e-06` | `3.3676624298095703e-06` | `1.0177860730824515` | `-0.0008716186579329765` | `0` | `AUC_time_delta>0;CouplingR2_delta<0.02;NoiseSignalLeak_delta>-0.01;control_gap_vs_best<=0` |
| `MLP-AdamW` | `MNIST` | `2` | `-0.025390625` | `0.1464874019594109` | `0.005107328295707703` | `2.3679447174072266` | `0.2785545418710361` | `-0.09066860377788544` | `1.3728604601033645` | `0.03685215005837672` | `0` | `` |
| `MLP-analogFunctional` | `MNIST` | `2` | `-0.025390625` | `0.18887412195659387` | `0.00510735809803009` | `2.3679561614990234` | `0.27855452091239874` | `-0.09066881239414215` | `1.3643430619277184` | `-0.005534799513700339` | `0` | `` |

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
| `v1210_family_policy.json` | `288d5cefd177` |
| `v1210_family_status.json` | `e23a23a23402` |
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

## 8. P4 修复尝试汇总

以下汇总只记录已经落盘 artifact 中的真实统计；`pass` 为 `B320-bestFunctional` 在 3 datasets x 3 seeds focused P4 short-run 中的 strict pass 行数。

| run | artifact_dir | 修改/目的 | B320-bestFunctional 关键结果 | 结论 |
|---|---|---|---|---|
| P4 smoke | `results/v12_10_b320_functional_classic_nobspline/v1210_p4_smoke_20260522T000000Z` | 首次接入 P4 short-run，修复 wrapper 未把 repo root 加入 `sys.path` 导致 `dgkan` import failure 的集成 bug | MNIST seed0/epochs1 smoke，`strict_pass=0`，fail reason 包含 ECE、NoiseSignalLeak、control gap | 仅用于集成验证，不作为 official 结论。 |
| P4 runtime backtracking | `results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_short_focused_deltafix_20260522T000000Z` | 使用 P3 F14d 方向，运行时 deepcopy/backtracking；同时修正 `NoiseSignalLeak_delta_vs_B320` 从绝对值改为 method minus B320 | pass `0/9`; mean acc_delta `-0.004991319444444444`; mean ECE_delta `0.025381920238335926`; mean CouplingR2_delta `0.36917729919583625`; mean NoiseSignalLeak_delta `0.03409979140592946`; mean overhead `6.380373793551417`; mean control_gap `-0.41952276882641354` | 真实 P4 失败；overhead、ECE、noise、control gap 同时不合格。 |
| P4 fixed lambda 0.0625 event24 | `results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_fixedlambda00625_event24_20260522T000000Z` | 成本修复：取消 runtime backtracking，固定 lambda=0.0625，每 24 step 一次维护事件 | pass `0/9`; mean acc_delta `0.0008680555555555555`; mean ECE_delta `0.005468514230516221`; mean CouplingR2_delta `0.3436475765798721`; mean NoiseSignalLeak_delta `0.0019353313578499688`; mean overhead `0.9682600191267977`; mean control_gap `0.04549171525738877` | overhead 已基本修复，但 AUC_time/ECE/noise/control gap 仍有行失败。 |
| P4 fixed lambda 0.03125 event24 | `results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_fixedlambda003125_event24_20260522T000000Z` | 强度下调 bracket，测试是否减少 ECE/noise 伤害 | pass `0/9`; mean acc_delta `0.00021701388888888888`; mean ECE_delta `0.0024165179994371203`; mean CouplingR2_delta `0.3323082514010865`; mean NoiseSignalLeak_delta `0.0010810624808073044`; mean overhead `0.9397015470003984`; mean control_gap `0.0911458039600335` | ECE/overhead 更健康，但 NoiseSignalLeak 未达到 `<= -0.01`，且部分 AUC_time/control gap 行失败。 |
| P4 fixed lambda 0.125 event24 | `results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_fixedlambda0125_event24_20260522T000000Z` | 强度上调 bracket，测试 noise release 是否需要更大步长 | pass `0/9`; mean acc_delta `-0.00021701388888888888`; mean ECE_delta `0.01205528030792872`; mean CouplingR2_delta `0.3492542392990327`; mean NoiseSignalLeak_delta `0.025746248041590054`; mean overhead `0.9828280667402844`; mean control_gap `-0.0018515857353599964` | 强度上调加重 ECE/noise，方向存在 task-safety tradeoff。 |
| P4 fixed lambda -0.03125 event24 | `results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_fixedlambda_neg003125_event24_20260522T000000Z` | 符号 sanity bracket，测试反向维护事件 | pass `0/9`; mean acc_delta `0.0008680555555555555`; mean ECE_delta `-0.002325753370920817`; mean CouplingR2_delta `0.332280062393609`; mean NoiseSignalLeak_delta `-0.0017077341261837217`; mean overhead `0.9948969507759843`; mean control_gap `0.0904913539929198`; mean CEp99_delta `0.09461243947347005` | 反向改善 ECE，但 CEp99、AUC_time、NoiseSignalLeak 仍失败。 |
| P4 quad lambda 0.03125 event24 | `results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_quad_lambda003125_event24_20260522T000000Z` | 新增 `--p4-functional-mode quad`，测试 P3 branch role 是否选错 | pass `0/9`; mean acc_delta `0.0006510416666666666`; mean ECE_delta `0.000674578050772349`; mean CouplingR2_delta `0.3339308300837576`; mean NoiseSignalLeak_delta `0.0012956234729952281`; mean overhead `0.9736231935975211`; mean control_gap `0.27855775692183754` | control gap、ECE、overhead 均健康，但 AUC_time 与 NoiseSignalLeak 未过。 |
| P4 neg-quad lambda 0.03125 event24 | `results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_negquad_lambda003125_event24_20260522T000000Z` | quad 符号 sanity bracket | pass `0/9`; mean acc_delta `0.0`; mean ECE_delta `-0.0008270897799068027`; mean CouplingR2_delta `0.33389939226294857`; mean NoiseSignalLeak_delta `-0.0014243870973587036`; mean overhead `0.9770560071625773`; mean control_gap `0.2801329136210396`; mean CEp99_delta `0.05871412489149305` | ECE 和 control gap 更好，但 NoiseSignalLeak 仍没到 `<= -0.01`，CEp99 开始变差。 |
| P4 neg-quad lambda 0.0625 event24 | `results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_negquad_lambda00625_event24_20260522T000000Z` | neg-quad 强度上调 bracket | pass `0/9`; mean acc_delta `0.0`; mean ECE_delta `-0.0015358469552463954`; mean CouplingR2_delta `0.3361842821693879`; mean NoiseSignalLeak_delta `-0.0030022936148775946`; mean overhead `0.9807937339566579`; mean control_gap `0.19237797240014332`; mean CEp99_delta `0.133310423956977` | noise release 增强但仍不足，同时 CEp99/AUC_time blocker 更明显。 |
| P4 task-minus-quad010 lambda 0.03125 event24 | `results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_taskminusquad010_lambda003125_event24_20260522T000000Z` | F15-style task-safe blend bracket | pass `0/9`; mean acc_delta `0.0`; mean ECE_delta `-6.573067771063911e-07`; mean CouplingR2_delta `3.9426649639554924e-05`; mean NoiseSignalLeak_delta `2.2382785876592e-06`; mean overhead `0.9681528084457871`; mean control_gap `4.812251550096828e-05` | task-safe 但几何效应塌缩，`CouplingR2_delta<0.02`。 |

当前 P4 结论：

```text
B320 base hardening = pass
B320 P3 diagnostic gate = pass
B320 P4 short-run = fail after runtime-backtracking, fixed-lambda, strength/sign bracket, branch-mode bracket, and F15-style blend bracket
official_functional_success = 0
route = R5-B320P4ShortRunFailed
classic_family_pass_count_excluding_bspline = 0
```

## 9. Classic Family Extended Focused Run

Artifact:

```text
results/v12_10_b320_functional_classic_nobspline/v1210_family_cheb_fourier_extended_20260522T000000Z
```

Protocol note:

```text
This was a focused family repair run: MNIST only, seed=0, epochs=1, train_size=512, val_size=256, test_size=256.
It must not be used to override the full v12.10 B320 hardening / official functional / official Rational conclusions.
```

Code modification audited:

```text
experiments/run_v1210_b320_functional_classic_nobspline.py
  Added already-implemented non-B-spline candidates to ACTIVE_FAMILY_IDS:
  B3f, B3al, B3am, B4v, B4w.
  Purpose: test v12.10 Chebyshev/Fourier expression repair directions under the same L3/A4 gates.
```

Key measured results:

| family | candidate | L3 step_ratio | memory_ratio | L3 pass | A4 pass | key expression deltas |
|---|---:|---:|---:|---:|---:|---|
| Chebyshev | B3f | `0.6521527773287539` | `1.0047619047619047` | `1` | `0` | E1 `-0.052822113037109375`, E6 `-0.05070286989212036`, E8 `-0.07686823606491089` |
| Chebyshev | B3y | `0.703215514033744` | `0.8523809523809524` | `1` | `0` | E1 `-0.05701035261154175`, E6 `-0.08060640096664429`, E8 `-0.09051793813705444` |
| Chebyshev | B3z | `0.6701433852509462` | `0.7814285714285715` | `1` | `0` | E1 `-0.05769479274749756`, E6 `-0.08526718616485596`, E8 `-0.09092313051223755` |
| Chebyshev | B3aa | `0.8797328506995857` | `0.7928571428571428` | `1` | `0` | E1 `-0.05089282989501953`, E6 `-0.08163946866989136`, E8 `-0.09169262647628784` |
| Chebyshev | B3al | `0.9247399367865786` | `0.7847619047619048` | `1` | `0` | E1 `-0.053018033504486084`, E6 `-0.07943016290664673`, E8 `-0.09940272569656372` |
| Chebyshev | B3am | `0.9279382278963169` | `0.780952380952381` | `1` | `0` | E1 `-0.05739927291870117`, E6 `-0.08377701044082642`, E8 `-0.09486734867095947` |
| Fourier | B4p | `1.079937292097852` | `0.7857142857142857` | `1` | `0` | E1 `-0.2300693392753601`, E6 `-0.3443470001220703`, E8 `-0.4360584616661072` |
| Fourier | B4q | `0.9876859032424359` | `0.5966666666666667` | `1` | `0` | E1 `-0.2284194827079773`, E6 `-0.3804120421409607`, E8 `-0.4018245339393616` |
| Fourier | B4v | `0.8183551696105327` | `0.6309523809523809` | `1` | `0` | E1 `-0.29307687282562256`, E6 `-0.755325436592102`, E8 `-0.4833430051803589` |
| Fourier | B4w | `0.7572990222904608` | `0.12428571428571429` | `1` | `0` | E1 `-0.19420772790908813`, E6 `-0.5619633793830872`, E8 `-0.5250858664512634` |

Focused family status:

```text
Chebyshev = ExpressionBlocked
Fourier = ExpressionBlocked
RBF = KernelBlocked
Wavelet = KernelBlocked
BSpline = FamilyFrozen_KernelBlocked_RejectedForThisVersion
```

Important audit warning:

```text
The focused run's route JSON reports Rational.status = FamilyPass and classic_family_pass_count_excluding_bspline = 1.
This is not promoted to v12.10 official family success because the focused protocol used MNIST seed0 / epochs1 only.
The full v12.10/v12.9 evidence still says Rational remains not official family success under the stricter multi-dataset protocol.
```

Extended family conclusion:

```text
Chebyshev L3 is now comfortably open, including K4/localrot2 brackets, but A4 remains failed.
Fourier h8 stronger linear residual variants improved L3 efficiency, especially B4w memory_ratio = 0.12428571428571429, but A4 expression became worse on E6/E8.
Per v12.10 plan, Chebyshev/Fourier remain ExpressionBlocked after K4 + paircross/localrot and low-frequency + residual brackets.
No non-B-spline classic family official success is claimed from this focused run.
```

## 10. RBF / Wavelet Stream-Recompute Focused Repair

Artifact:

```text
results/v12_10_b320_functional_classic_nobspline/v1210_family_rbf_wavelet_recompute2_20260522T000000Z
```

Protocol note:

```text
This was a focused blocker repair run: MNIST only, seed=0, epochs=1, train_size=512, val_size=256, test_size=256.
It is used only to audit the RBF/Wavelet compact-local repair direction and cannot override full v12.10 official conclusions.
```

Code modifications audited:

```text
dgkan/models/fc_purekan_primitives.py
  Added _basis_derivative_channel.
  Added stream-recompute manual CE path for fastkan_rbf / compact_rbf / hat_wavelet.
  The path caches only z/h and recomputes each basis channel during backward instead of storing dense b1/b2/db2.
  Added B2s-GaussianRBF-stream-K4-recompute to cover v12.10 RBF-F2.

experiments/run_v1283_b109_classic_family_functional_geometry.py
  Added B2s to _family_plan so the focused run actually measures RBF K_active=4.

experiments/run_v1210_b320_functional_classic_nobspline.py
  Added B2s to ACTIVE_FAMILY_IDS.
  Added --family-candidate-ids so focused family repair runs can audit only the intended subset.
```

Local CPU gradient sanity before CUDA focused run:

```text
B2r fastkan_rbf_stream_recompute: grad_relerr_max = 9.045955096098623e-08, output_max_abs_error = 0.0
B2s compact_rbf_stream_recompute: grad_relerr_max = 9.533125222560557e-08, output_max_abs_error = 0.0
B5h hat_wavelet_stream_recompute: grad_relerr_max = 1.4681359061796684e-07, output_max_abs_error = 0.0
```

Measured CUDA focused results:

| family | candidate | manual variant | analytic gradcheck | L3 step_ratio | L3 memory_ratio | official fused L3 | official efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| RBF | B2r | `fastkan_rbf_stream_recompute` | `1`, grad_relerr `1.3815819954743347e-07`, output error `0.0` | `1.3178717497236523` | `1.6919047619047618` | `0` | `0` |
| RBF | B2s | `compact_rbf_stream_recompute` | `1`, grad_relerr `1.352966165768521e-07`, output error `0.0` | `2.0863001291704486` | `1.2923809523809524` | `0` | `0` |
| Wavelet | B5h | `hat_wavelet_stream_recompute` | `1`, grad_relerr `1.323139429132425e-07`, output error `0.0` | `2.8633910169000014` | `2.039047619047619` | `0` | `0` |

Focused family status from artifact:

```text
RBF.status = KernelBlocked
RBF.best_L3_manual_step_ratio = 1.3178717497236523
RBF.blocker = L3 manual analytic attempted; fused family-specific backward/update kernel missing

Wavelet.status = KernelBlocked
Wavelet.best_L3_manual_step_ratio = 2.8633910169000014
Wavelet.blocker = L3 manual analytic attempted; fused family-specific backward/update kernel missing

classic_family_pass_count_excluding_bspline = 0
official_functional_success = 0
```

Conclusion:

```text
The stream-recompute repair is mathematically correct for the tested batches, but it did not open the v12.10 RBF/Wavelet L3 gate.
RBF K_active=2 and K_active=4 both fail focused L3 step/memory gates and still lack a family-specific fused backward/update kernel.
Wavelet hat/triangle-local also fails focused L3 step/memory gates and still lacks a family-specific fused backward/update kernel.
Per v12.10 sections 9.5 and 9.6, RBF/FastKAN and Wavelet remain KernelBlocked and should be frozen for this version unless a real fused kernel is implemented first.
No A4/A5 or family success is claimed from this focused repair.
```

## 11. P4 Event-Gate Tail/Noise Veto Brackets

Code modification audited:

```text
experiments/run_v1210_b320_functional_classic_nobspline.py
  Added --p4-event-gate tail_noise / tail_noise_reservoir.
  Each maintenance event checks CEp99, margin_p10, NoiseSignalLeak, and optionally RealSignalReservoirRatio before acceptance.
  Failed events are rolled back.
  This does not change CE loss, data, labels, sampler, class weights, teacher/distillation, or the final strict P4 gate.
```

Artifacts:

```text
results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_negquad00625_eventgate_tailnoise_20260522T000000Z
results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_quad003125_eventgate_tailnoise_20260522T000000Z
```

Key results:

| run | pass | mean acc_delta | mean AUC_time_delta | mean ECE_delta | mean CEp99_delta | mean CouplingR2_delta | mean NoiseSignalLeak_delta | mean overhead | mean control_gap | mean accepted/rejected events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neg-quad lambda 0.0625 + tail_noise gate | `0/9` | `0.00021701388888888888` | `0.9395459706930827` | `-0.0002055838704109192` | `0.07037003835042317` | `0.11814711580036175` | `-0.0010472467790047328` | `1.0848134126098878` | `-0.15158445854740554` | `0.3333333333333333 / 0.6666666666666666` |
| quad lambda 0.03125 + tail_noise gate | `0/9` | `0.00043402777777777775` | `0.7570811730205015` | `0.0005526898635758294` | `-0.040127330356174044` | `0.23367458725472692` | `0.001631012186408043` | `0.9985525114092676` | `0.09281487670041508` | `0.6666666666666666 / 0.3333333333333333` |

Failure interpretation:

```text
neg-quad + tail_noise gate rejected many MNIST/KMNIST events and left geometry mostly collapsed there; Fashion-MNIST accepted events but still failed AUC_time / CEp99 / NoiseSignalLeak rows.
quad + tail_noise gate kept CE tail healthier and preserved overhead, but still failed all B320-bestFunctional rows because AUC_time_delta stayed positive and NoiseSignalLeak_delta did not reach <= -0.01.
Event-level veto is therefore a real measured negative result for P4 official confirmation, not a success.
official_functional_success remains 0.
```

## 12. RBF/FastKAN Family-Specific Fused L3 Kernel

Artifact:

```text
results/v12_10_b320_functional_classic_nobspline/v1210_family_rbf_fused_l3_20260522T000000Z
```

Protocol note:

```text
Focused implementation run: MNIST only, seed=0, epochs=1, train_size=512, val_size=256, test_size=256.
The result is valid for auditing the new RBF fused L3 path and A4 transition, but it is not a full multi-dataset family official success claim.
```

Code modifications audited:

```text
dgkan/kernels/fused_rbf.py
  Added fixed-center Gaussian RBF/FastKAN Triton K2/K4 forward + backward kernels.
  Backward recomputes basis in Triton and writes w1.grad / w2.grad; no dense [B,D,K] or [B,H,K] basis cache.

dgkan/models/fc_purekan_primitives.py
  B2r now uses init_variant=rbf_k2_triton_l3_matmul.
  B2s now uses init_variant=rbf_k4_triton_l3_matmul.
  manual_ce_forward_cache / manual_ce_backward_from_cache dispatch to fused_rbf for these variants.

experiments/run_v1283_b109_classic_family_functional_geometry.py
  Added rbf_k2_triton_l3_matmul and rbf_k4_triton_l3_matmul to official fused L3 kernel variants.
```

Pre-run CUDA gradient sanity:

```text
B2r rbf_k2_triton_l3_matmul: grad_relerr_max = 2.2664832499685872e-07, output_max_abs_error = 3.650784492492676e-07
B2s rbf_k4_triton_l3_matmul: grad_relerr_max = 2.449252747283026e-07, output_max_abs_error = 3.0547380447387695e-07
```

Focused measured results:

| candidate | manual variant | analytic gradcheck | L3 step_ratio | L3 memory_ratio | official fused L3 | official efficiency | A4 |
|---|---|---:|---:|---:|---:|---:|---:|
| B2r | `rbf_k2_triton_l3_matmul` | `1`, grad_relerr `5.058263354840165e-07`, output error `5.21540641784668e-07` | `0.7103699836137742` | `0.9990476190476191` | `1` | `1` | `0` |
| B2s | `rbf_k4_triton_l3_matmul` | `1`, grad_relerr `1.210070422530407e-06`, output error `6.407499313354492e-07` | `0.6920597147480739` | `1.0047619047619047` | `1` | `1` | `0` |

A4 failure details:

```text
B2r A4 summary: E1 delta = -0.9968013763427734, E2 delta = -0.17229336500167847, E6 delta = -0.9954899549484253, E8 delta = -0.9957923889160156.
B2s A4 summary: E1 delta = -0.17386770248413086, E2 delta = -0.07675796747207642, E6 delta = -0.19729214906692505, E8 delta = -0.2404266595840454.
Both rows emitted A4_expression_gate_fail_after_l3_efficiency.
```

Updated RBF family status:

```text
RBF.status = ExpressionBlocked
RBF.L3_fused_kernel_measured = true
RBF.L3_official_efficiency_candidate_exists = true
RBF.best_L3_manual_step_ratio = 0.6920597147480739
RBF.blocker = family-specific fused L3 efficiency passed but A4 expression failed
```

Conclusion:

```text
The requested real family-specific fused backward path was implemented for RBF/FastKAN K2/K4 and passed focused gradient + L3 efficiency checks.
This resolves the prior RBF KernelBlocked state in the focused run, but RBF is not a family success because A4 expression fails.
Next legitimate RBF work, if continued, must repair expression capacity without losing the fused L3 step/memory gate.
No official functional success or family success is claimed from this run.
```

## 14. P4 Tail/Noise/Reservoir Veto Bracket

Artifacts:

```text
fresh raw attempt:
results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_negquad00625_eventgate_tailnoise_reservoir_20260522T000000Z

P3-pass raw reuse attempt:
results/v12_10_b320_functional_classic_nobspline/v1210_b320_p4_negquad00625_eventgate_tailnoise_reservoir_reusep3_20260522T000000Z
```

Protocol note:

```text
This bracket follows F-F2/F-F3 after P4 failure: add event-level CE tail, noise leak, and reservoir-release veto.
It does not change CE loss, labels, data, sampler/class weights, teacher/distillation, or the final strict P4 gate.
```

Fresh raw attempt result:

```text
raw_mode = ran
route = R3-B320AnchorReadyFunctionalStillBlocked
functional_P3_official_candidate_pass = 0
v1210_functional_short_run.csv = not generated
official_functional_success = 0

Interpretation: this run did not enter P4 because the fresh raw P3 gate was closed. It is recorded as a not-run attempt, not a P4 negative/positive result.
```

P3-pass raw reuse measured result:

```text
raw_mode = reused
run mode = p4-functional-mode quad, p4-fixed-lambda -0.0625, p4-event-gate tail_noise_reservoir, event_every_steps = 24
B320-bestFunctional strict pass = 0 / 9
official_functional_success = 0
```

Aggregate B320-bestFunctional metrics from the reuse run:

| metric | mean |
|---|---:|
| acc_delta_vs_B320 | `0.0` |
| AUC_time_delta_vs_B320 | `0.8153835439055539` |
| ECE_delta_vs_B320 | `2.5663110944959854e-08` |
| CEp99_delta_vs_B320 | `2.7020772298177085e-06` |
| CouplingR2_delta_vs_B320 | `1.0306551399527938e-05` |
| RealSignalReservoirRatio_delta_vs_B320 | `1.3278590308295356e-06` |
| NoiseSignalLeak_delta_vs_B320 | `6.796585188971626e-07` |
| amortized_overhead_ratio | `0.9816752089468932` |
| control_gap_vs_best | `-0.013803450879180236` |
| functional_event_count | `1.0` |
| accepted_event_count | `0.0` |
| rejected_event_count | `1.0` |

Event-gate rejection counts for B320-bestFunctional:

```text
event_CEp99_delta>epsilon;event_RealSignalReservoirRatio_release_insufficient:1 -> 3 rows
event_RealSignalReservoirRatio_release_insufficient:1 -> 3 rows
event_margin_p10_drop;event_RealSignalReservoirRatio_release_insufficient:1 -> 2 rows
event_CEp99_delta>epsilon;event_margin_p10_drop;event_NoiseSignalLeak_increase;event_RealSignalReservoirRatio_release_insufficient:1 -> 1 row
```

Strict P4 fail reasons:

```text
5 rows: AUC_time_delta>0; CouplingR2_delta<0.02; NoiseSignalLeak_delta>-0.01; control_gap_vs_best<=0
2 rows: AUC_time_delta>0; CouplingR2_delta<0.02; NoiseSignalLeak_delta>-0.01
1 row: CouplingR2_delta<0.02; NoiseSignalLeak_delta>-0.01; control_gap_vs_best<=0
1 row: AUC_time_delta>0; CouplingR2_delta<0.02; NoiseSignalLeak_delta>-0.01; amortized_overhead>1.05; control_gap_vs_best<=0
```

Conclusion:

```text
The reservoir-release veto is a measured negative result.
It made the P4 maintenance event fully rejected in all 9 B320-bestFunctional rows, leaving the method effectively near no-op: task/ECE/CE tail stayed safe, but CouplingR2_delta collapsed to about 1.03e-05, NoiseSignalLeak_delta did not reach <= -0.01, AUC_time remained positive, and control_gap_vs_best was negative on average.
official_functional_success remains 0.
```

## 13. Wavelet Hat Family-Specific Fused L3 Kernel

Artifact:

```text
results/v12_10_b320_functional_classic_nobspline/v1210_family_wavelet_fused_l3_20260522T000000Z
```

Protocol note:

```text
Focused implementation run: MNIST only, seed=0, epochs=1, train_size=512, val_size=256, test_size=256.
The result audits the new Wavelet hat/triangle fused L3 path required by v12.10 WAV-F1, but it is not a full multi-dataset family official success claim.
```

Code modifications audited:

```text
dgkan/kernels/fused_hat_wavelet.py
  Added fixed-center hat/triangle Wavelet Triton K4 forward + backward kernels.
  Backward recomputes the local hat basis and derivative in Triton and writes w1.grad / w2.grad; no dense [B,D,K] or [B,H,K] basis cache.

dgkan/models/fc_purekan_primitives.py
  B5h now uses init_variant=hat_wavelet_k4_triton_l3_matmul.
  manual_ce_forward_cache / manual_ce_backward_from_cache dispatch to fused_hat_wavelet for B5h.

experiments/run_v1283_b109_classic_family_functional_geometry.py
  Added hat_wavelet_k4_triton_l3_matmul to official fused L3 kernel variants.

experiments/run_v1210_b320_functional_classic_nobspline.py
  Added this Wavelet fused L3 edit to the v12.10 audit table.
```

Verification:

```text
python -m py_compile dgkan/kernels/fused_hat_wavelet.py dgkan/models/fc_purekan_primitives.py experiments/run_v1283_b109_classic_family_functional_geometry.py experiments/run_v1210_b320_functional_classic_nobspline.py
status = pass
```

Pre-run CUDA gradient sanity:

```text
B5h hat_wavelet_k4_triton_l3_matmul:
  grad_relerr_max = 4.708418828158756e-07
  grad_cos_min = 0.9999999403953552
  output_max_abs_error = 7.82310962677002e-08
```

Focused measured results:

| candidate | manual variant | analytic gradcheck | L3 step_ratio | L3 memory_ratio | official fused L3 | official efficiency | A4 |
|---|---|---:|---:|---:|---:|---:|---:|
| B5h | `hat_wavelet_k4_triton_l3_matmul` | `1`, grad_relerr `1.0833267651833012e-06`, output error `8.568167686462402e-08` | `0.648993247415022` | `1.0047619047619047` | `1` | `1` | `0` |

A4 failure details:

```text
B5h A4 summary row:
E1 delta = -1.0239673852920532, B1 val R2 = -0.027661681175231934, frozen R2 = -0.1063694953918457
E2 delta = -0.06650412082672119, B1 val R2 = 0.9272632598876953, frozen R2 = 0.9245912432670593
E6 delta = -0.999286949634552, B1 val R2 = -0.0034426450729370117, frozen R2 = -0.2150346040725708
E8 delta = -0.9814637303352356, B1 val R2 = 0.012902319431304932, frozen R2 = -1.0236592292785645

B2 high-budget key rows:
E1 delta = -0.9984725117683411, val R2 = -0.0018302202224731445
E2 delta = -0.06518763303756714, val R2 = 0.9293475151062012
E6 delta = -0.9981919527053833, val R2 = -0.002756357192993164
E8 delta = -0.9824029803276062, val R2 = 0.013824164867401123
```

Updated Wavelet family status:

```text
Wavelet.status = ExpressionBlocked
Wavelet.L3_fused_kernel_measured = true
Wavelet.L3_official_efficiency_candidate_exists = true
Wavelet.best_L3_manual_step_ratio = 0.648993247415022
Wavelet.blocker = family-specific fused L3 efficiency passed but A4 expression failed
```

Conclusion:

```text
The requested real family-specific fused backward path was implemented for Wavelet hat K4 and passed focused gradient + L3 efficiency checks.
This resolves the prior Wavelet KernelBlocked state in the focused run, but Wavelet is not a family success because A4 expression fails.
No official functional success or family success is claimed from this run.
```
