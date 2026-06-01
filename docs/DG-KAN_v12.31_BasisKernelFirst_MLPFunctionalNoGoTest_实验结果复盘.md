# DG-KAN v12.31 BasisKernelFirst MLPFunctionalNoGoTest 实验结果复盘

生成时间：2026-05-27（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic/near-pass 写成 promotion。

## 1. 计划理解

v12.31 的目标不是继续扩大 v12.30 的 alias/token grid，而是把 classic no-BSpline basis 线升级到 kernel/workspace memory-first 审计，并在 MLP 上执行 loss-agnostic functional value source 的 no-go / 新机制微探针。

硬约束：

```text
1. strict FC-PureKAN / classic no-BSpline active basis。
2. no teacher / distillation / loss modification / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target。
5. CE/NLL/ECE/CEp99 只能作为审计和坏化约束。
6. blocker 后必须执行计划要求的 fallback，不允许把 diagnostic/smoke/skipped row 写成 promotion。
```

## 2. 当前状态

实验正在执行中。本文件后续只记录已经生成 artifact 的真实结果。

## 3. 本轮代码修改

### 3.1 Basis workspace core

新增文件：

```text
dgkan/diagnostics/basis_workspace.py
```

核心内容：

```text
1. V1231_BASIS_CANDIDATES：D-RAT7..D-RAT12、D-WAV6..D-WAV9、D-CHE6..D-CHE9、D-RBF7..D-RBF10、D-FOU6..D-FOU9 映射表。
2. measured_phase_step：记录 CUDA forward/backward/update peak、raw peak、incremental peak、step time。
3. static_workspace_accounting：输出 basis_activation/readout_grad/optimizer_state/temp workspace 等静态 component 估计；该字段只作审计，不作 gate source。
4. workspace_gate / workspace_strong_gate：按 v12.31 plan 的 raw/incremental/step ratio 阈值判定。
5. component_peak_rows / microkernel_correctness_rows：生成 component profiler 与 implementation readback rows。
```

重要审计说明：

```text
v12.31 当前没有新增真实 fused CUDA/Triton microkernel；D-RAT7 等候选先映射到现有 dgkan PrimitiveSpec 低内存路径做 workspace truth audit。
因此 exact_kernel_implemented=0，不把这些 alias 描述成已经实现了 plan 字面意义的 fused kernel。
```

### 3.2 MLP functional core

修改文件：

```text
dgkan/functional/mlp_functional.py
```

新增 M-G candidates：

```text
M-G1-UnlabeledCloneResponseCov
M-G2-RandomCotangentResponseStability
M-G3-InputPerturbationConsistencyTransport
M-G4-HiddenCovarianceTransportWithControlResidual
M-G5-ActivationSubspaceGuardThenTaskNeutral
```

合法性说明：

```text
M-G1..M-G5 只读取模型状态、unlabeled train-stream input / activation / logits response；
label/CE 只存在于 supervised AdamW 训练和 C3 AdamWParallelDirection control，不进入 functional direction source。
```

### 3.3 v12.31 runners/finalizer

新增文件：

```text
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_finalize_basis_kernel_first.py
```

设计说明：

```text
1. basis runner 只做 orchestration；核心 workspace accounting 在 dgkan/diagnostics/basis_workspace.py。
2. workspace gate 不过时，不进入 task hardening；hardening/LineC artifact 写 explicit skip row。
3. finalizer 生成 route decision、fallback manifest、provenance audit、no-go boundary、next hypothesis queue、figures 与 code review packet。
```

## 4. Provenance / import audit

语法检查：

```text
py_compile pass
```

Import/provenance 审计结果：

```text
basis_candidates = 22
basis_exact_kernel_sum = 0
mg_candidates = 5
controls = 5
adyn = 5
adyn_uses_y_for_stats = 0
adyn_forbidden = 0
```

解释：

```text
1. v12.31 basis candidate registry 覆盖 22 个 D-RAT/D-WAV/D-CHE/D-RBF/D-FOU workspace audit candidates。
2. exact_kernel_sum=0 是刻意审计字段：本轮尚未新增真实 fused kernel；不能把现有低内存 primitive path 写成已经完成 plan 字面 fused kernel。
3. M-G functional sources 已进入 dgkan.functional.mlp_functional；C3 仍是 non-promotable CE/label control。
4. A-DYN monitor uses_y_for_stats=0 且 forbidden_token_present=0。
```

## 5. Smoke checks

已执行三个 smoke：

```text
basis workspace smoke rows = 2, workspace_gate_pass_rows = 1, hardening_executed_rows = 1
MLP functional smoke candidate_rows = 1, control_rows = 6, linec_rows = 4
A-DYN smoke rows = 4, summary_rows = 4
```

解释：

```text
1. Smoke 证明 v12.31 basis runner、M-G functional source、A-DYN monitor 调用链可运行。
2. Basis smoke 中 1 个小规模 row 过 workspace gate，只能说明 smoke 尺度下入口可用；不能作为 official workspace success。
3. MLP/A-DYN smoke 均不作为 official evidence。
```

## 6. Line D basis workspace truth audit

### 6.1 重要实现修复

首次 official run 后发现：

```text
1. MLP task reference 只跑 1 epoch，而 basis hardening 跑 3 epoch，task delta 不公平。
2. workspace step 直接测首步，可能混入 first-call compile / lazy allocation。
3. row-level family_near_pass 缺少跨 dataset/seed worst/AUC 聚合，不能作为 FamilyNearPass。
```

修复内容：

```text
1. MLP workspace profile 与 MLP task reference 分离。
2. MLP task reference 使用 fresh MLP 并训练同等 hardening_epochs。
3. workspace phase 前增加 workspace_warmup_steps。
4. hardening row 改为 task_linec_probe_pass；family_near_pass 固定为 0，finalizer 只把它作为 probe，不作 promotion。
5. finalizer route 改为：workspace 打开但 task/LineC 未闭合时，route=R2-BasisTaskLineCNotColocated，minimum_success=S1-BasisWorkspaceOpened。
```

合理性：

```text
这是审计口径修复，不降低任何 gate；修复后旧首次 run 不进入最终 artifact，已重跑 canonical v1231_basis_*。
```

### 6.2 Workspace truth 结果

执行规模：

```text
candidates = 22
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val = 256
workspace_rows = 198
```

总体结果：

```text
workspace_gate_pass_rows = 36
workspace_strong_gate_pass_rows = 22
hardening_executed_rows = 36
family_near_pass_rows = 0
```

Family status：

| family | status | rows | workspace pass | strong pass | task probe pass | min raw ratio | min incremental ratio | min step ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D-RAT | TaskLineCNotColocated | 54 | 36 | 22 | 20 | 0.7030934708989138 | 1.3255813953488371 | 0.5972796919231077 |
| D-CHE | WorkspaceBlocked | 36 | 0 | 0 | 0 | 0.9490445859872612 | 6.819767441860465 | 0.7927786921543468 |
| D-FOU | WorkspaceBlocked | 36 | 0 | 0 | 0 | 0.9228825604042744 | 3.9244186046511627 | 0.7347246970844235 |
| D-RBF | WorkspaceBlocked | 36 | 0 | 0 | 0 | 0.9103191732027864 | 3.5456810631229234 | 0.9492694811328597 |
| D-WAV | WorkspaceBlocked | 36 | 0 | 0 | 0 | 0.9233387727886859 | 3.5456810631229234 | 1.1220340571266207 |

解释：

```text
1. v12.31 的真正正进展是 Rational workspace 打开：D-RAT 有 36/54 workspace pass、22/54 strong pass。
2. Chebyshev/Fourier/RBF/Wavelet 仍被 incremental memory blocker 卡住，没有进入 task hardening。
3. Rational task/LineC 有局部 probe signal，但不是 FamilyNearPass。
```

Rational hardening probe aggregate：

| candidate | executed | skipped | mean_delta_vs_MLP | worst_delta_vs_MLP | LineC pass | aggregate probe pass | FamilyNearPass |
|---|---:|---:|---:|---:|---:|---:|---:|
| D-RAT10-FusedDenNumReadoutGrad | 6 | 3 | 0.0032552083333333335 | -0.0234375 | 17/18 | 0 | 0 |
| D-RAT11-LowMemGroupSharedDenomPlusLineC | 6 | 3 | 0.0032552083333333335 | -0.0234375 | 17/18 | 0 | 0 |
| D-RAT12-RationalWorkspaceMinStrongDiag | 6 | 3 | 0.0032552083333333335 | -0.0234375 | 17/18 | 0 | 0 |
| D-RAT7-FusedGroupRationalNoMaterialize | 6 | 3 | 0.0032552083333333335 | -0.0234375 | 17/18 | 0 | 0 |
| D-RAT8-RecomputeDenominatorBackward | 6 | 3 | 0.0026041666666666665 | -0.0234375 | 16/18 | 0 | 0 |
| D-RAT9-ChunkedReadoutGradNoPersistentBasis | 6 | 3 | -0.15169270833333334 | -0.2265625 | 3/18 | 0 | 0 |

为什么不 promotion：

```text
1. 每个 D-RAT candidate 都有 3 个 dataset/seed row 因 workspace gate fail 被 skip。
2. v12.31 workspace runner 未测 AUC_time_ratio_vs_MLP，不能满足计划中的完整 FamilyNearPass gate。
3. D-RAT7/D-RAT10/D-RAT11/D-RAT12 的 task/LineC probe 很接近，但仍只能写成 S1 workspace opened + R2 task/LineC not colocated。
```

## 7. Line M M-G MLP functional no-go test

执行规模：

```text
windows = 3,5,10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_seed_bases = 12310400,12311400,12312400
candidates = M-G1..M-G5
candidate_rows = 405
control_rows = 2430
linec_rows = 4050
gate_rows = 15
```

总结果：

```text
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
```

Aggregate by source/window：

| candidate | window | mean_vs_noop | mean_vs_control | mean_CouplingR2_delta | max LineC |
|---|---:|---:|---:|---:|---:|
| M-G1-UnlabeledCloneResponseCov | 3 | 0.00028935185185185184 | -0.0011574074074074073 | 0.00042905512746847443 | 5/5 |
| M-G1-UnlabeledCloneResponseCov | 5 | -0.00028935185185185184 | -0.0011574074074074073 | -0.0006227479107438948 | 5/5 |
| M-G1-UnlabeledCloneResponseCov | 10 | 0.0 | -0.002025462962962963 | -0.001180848620549578 | 5/5 |
| M-G2-RandomCotangentResponseStability | 3 | 0.00028935185185185184 | -0.0011574074074074073 | 0.00020349301429235645 | 5/5 |
| M-G2-RandomCotangentResponseStability | 5 | -0.00043402777777777775 | -0.0013020833333333333 | -0.00042640203816811866 | 5/5 |
| M-G2-RandomCotangentResponseStability | 10 | 0.00014467592592592592 | -0.001880787037037037 | -0.00031161301514741037 | 5/5 |
| M-G3-InputPerturbationConsistencyTransport | 3 | 0.0 | -0.0014467592592592592 | -0.0002838888976915146 | 5/5 |
| M-G3-InputPerturbationConsistencyTransport | 5 | -0.0005787037037037037 | -0.0014467592592592592 | -0.0004932307176376419 | 5/5 |
| M-G3-InputPerturbationConsistencyTransport | 10 | -0.00014467592592592592 | -0.002170138888888889 | -0.0005069240879798229 | 5/5 |
| M-G4-HiddenCovarianceTransportWithControlResidual | 3 | 0.00043402777777777775 | -0.0010127314814814814 | 0.0002135505542001181 | 5/5 |
| M-G4-HiddenCovarianceTransportWithControlResidual | 5 | -0.00043402777777777775 | -0.0013020833333333333 | -0.00019793418837720279 | 5/5 |
| M-G4-HiddenCovarianceTransportWithControlResidual | 10 | 0.00014467592592592592 | -0.001880787037037037 | -0.0008416951862234066 | 5/5 |
| M-G5-ActivationSubspaceGuardThenTaskNeutral | 3 | 0.0005787037037037037 | -0.0008680555555555555 | 0.00017259685195885547 | 5/5 |
| M-G5-ActivationSubspaceGuardThenTaskNeutral | 5 | -0.00014467592592592592 | -0.0010127314814814814 | -0.0006858368460149559 | 5/5 |
| M-G5-ActivationSubspaceGuardThenTaskNeutral | 10 | 0.00014467592592592592 | -0.001880787037037037 | -0.000494443712774231 | 5/5 |

最接近 row：

```text
candidate = M-G4-HiddenCovarianceTransportWithControlResidual
window = 10
dataset = KMNIST
seed = 0
train_seed_base = 12312400
source_vs_noop = +0.00390625
source_vs_control = +0.00390625
LineC_pass_count = 5/5
CouplingR2_delta = -0.0009028726468589098
exploration_gate_pass = 0
```

解释：M-G family 局部 row 能达到 LineC 5/5 与 positive control margin，但 CouplingR2_delta 为负，平均 source_vs_control 也为负，因此不能写 MLP functional generic positive。

## 8. Line A A-DYN monitor

执行规模：

```text
candidates = A-DYN1..A-DYN5
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
epochs = 3
rows = 63
summary_rows = 7
```

相对 MLP-same-param：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp |
|---|---:|---:|
| A-DYN1-LearnableSignalFrameWarmup | -0.012152777777777778 | -0.06640625 |
| A-DYN4-OvercompleteRankGuardFrame | -0.020833333333333332 | -0.08984375 |
| A-DYN2-EarlySelfPredictiveFrame | -0.06684027777777778 | -0.19921875 |
| A-DYN3-OptimizerObservableFrameRefresh | -0.06770833333333333 | -0.15234375 |
| A-DYN5-RoleEnergyBalancedFHQMonitor | -0.2404513888888889 | -0.359375 |

结论：A-DYN monitor 没有恢复 label-free near-anchor，Line F official re-entry 仍不允许。

## 9. Final route

最终 route：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 198
basis_workspace_pass_count = 36
basis_workspace_strong_pass_count = 22
basis_family_near_pass_count = 0
basis_hardening_executed_rows = 36
mlp_functional_candidate_rows = 405
mlp_functional_control_rows = 2430
mlp_functional_linec_rows = 4050
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 78
code_review_packet_sha256 = 以最终 v1231_route_decision.json 为准
```

## 10. Required artifacts

主要产物：

```text
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_route_decision.json
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_required_artifact_manifest.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_fallback_manifest.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_basis_workspace_truth.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_basis_component_peak.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_basis_microkernel_correctness.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_basis_hardening.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_basis_linec.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_family_status.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_mlp_functional_candidates.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_mlp_functional_controls.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_mlp_functional_linec.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_mlp_functional_gate.csv
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_code_review_packet.zip
```

## 11. 最终科学结论

v12.31 没有达成 S5，也没有产生 classic FamilyNearPass 或 MLP functional generic positive。

已闭合事实：

```text
1. Basis kernel/workspace P0 已执行 22 candidates x 3 datasets x 3 seeds。
2. Rational workspace 明显打开：36/54 workspace pass，22/54 strong pass。
3. Chebyshev/Fourier/RBF/Wavelet 的 raw ratio 可低于 1.5，但 incremental ratio 仍高于 3.0，workspace gate 未开。
4. Rational task/LineC probe 有局部正信号，D-RAT7/10/11/12 aggregate mean_delta 约 +0.0033、worst -0.0234、LineC 17/18；但有 skipped rows 且 AUC_time 未测，不能写 FamilyNearPass。
5. M-G1..M-G5 在 MLP windows=3/5/10 下没有 exploration/official pass。
6. A-DYN monitor 没有 label-free near-anchor。
7. Provenance audit 通过，required artifacts 缺失为 0。
```

新增 no-go boundary：

```text
1. v12.31 证明 Rational workspace blocker 可被现有 low-memory path 局部打开，但 task/LineC/AUC 完整闭合仍未完成。
2. 非 Rational basis family 仍主要被 incremental memory gate 卡住，不能直接进入 task grid。
3. MLP loss-agnostic M-G response/covariance/value sources 仍不能击败 matched controls。
4. A-DYN 仍只能作为 monitor，不允许 functional promotion。
```

最终合法状态：

```text
R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
```

下一步不应继续扩 M-G 单项 objective，也不应把非 Rational family 直接 task grid 化。更合理的方向是：对 Rational 做真正 AUC-time hardening 与 skipped-row workspace repair；或者实现真实 fused/recompute kernel 后重新 P0，再谈 FamilyNearPass。

## 12. 用户继续要求后的 Rational workspace / AUC repair

用户再次要求确认 v12.31 是否达成目标，若未达成则继续。读取 route 后确认：

```text
route = R2-BasisTaskLineCNotColocated
official_success_reached = 0
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
promotion_allowed = 0
```

因此本轮按计划继续 Rational repair，而不是继续扩 M-G 单项 objective 或把非 Rational family 直接 task grid 化。

### 12.1 本轮代码修改

修改/新增文件：

```text
dgkan/diagnostics/basis_workspace.py
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_rational_auc_hardening.py
experiments/run_v1231_finalize_basis_kernel_first.py
```

修改内容：

```text
1. 新增 measured_phase_window()，用于 post-warmup 多步 workspace profile。
2. workspace runner 新增 --workspace-profile-steps，MLP/basis workspace profile 改为多步 q90/max 聚合。
3. 新增 train_epoch_timed / trajectory_auc / trajectory_time_auc，把 AUC 轨迹 helper 放入 dgkan。
4. 新增 run_v1231_rational_auc_hardening.py，仅执行 workspace-pass Rational candidates 的 trajectory/AUC/LineC audit。
5. finalizer 纳入 Rational repair artifacts 与新的 core symbol map。
```

合理性：

```text
1. 多步 workspace profile 是审计口径修复，不降低 workspace gate。
2. AUC runner 只在 workspace repair pass 后执行；skipped row 不会被写成 success。
3. AUC_time 使用 epoch mean NLL 与 step q90 time 的保守 audit；不是基于 validation/test 的 commit direction。
4. CEp99/NLL/ECE 仍只作为坏化约束；没有用 CE tail 设计 functional direction。
```

### 12.2 Rational multi-step workspace repair

执行规模：

```text
candidates = D-RAT7,D-RAT8,D-RAT10,D-RAT11,D-RAT12
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
workspace_warmup_steps = 2
workspace_profile_steps = 3
rows = 45
```

结果：

| candidate | rows | workspace pass | strong pass | max raw ratio | max incremental ratio | max step ratio |
|---|---:|---:|---:|---:|---:|---:|
| D-RAT7-FusedGroupRationalNoMaterialize | 9 | 9 | 3 | 0.9112617119985446 | 1.3255813953488371 | 1.6783035167257876 |
| D-RAT8-RecomputeDenominatorBackward | 9 | 9 | 5 | 0.9200855089602474 | 1.6029900332225913 | 1.5264153578052289 |
| D-RAT10-FusedDenNumReadoutGrad | 9 | 9 | 8 | 0.9112617119985446 | 1.3255813953488371 | 1.3361446474069723 |
| D-RAT11-LowMemGroupSharedDenomPlusLineC | 9 | 9 | 4 | 0.9201082507049941 | 1.6038205980066444 | 1.5346359425049412 |
| D-RAT12-RationalWorkspaceMinStrongDiag | 9 | 9 | 6 | 0.9124442827253707 | 1.3496677740863787 | 1.47068722160661 |

结论：多步 profile 修复了此前 D-RAT candidates 的 skipped rows；Rational workspace repair 达到 `45/45` pass、`26/45` strong pass。这是 v12.31 的真实正进展，但还不是 FamilyNearPass。

同一 run 的 3-epoch task/LineC probe：

| candidate | executed | mean_delta_vs_MLP | worst_delta_vs_MLP | LineC pass | probe pass rows |
|---|---:|---:|---:|---:|---:|
| D-RAT7-FusedGroupRationalNoMaterialize | 9 | -0.008246527777777778 | -0.05078125 | 25/27 | 5 |
| D-RAT8-RecomputeDenominatorBackward | 9 | -0.009114583333333334 | -0.05078125 | 25/27 | 4 |
| D-RAT10-FusedDenNumReadoutGrad | 9 | -0.008246527777777778 | -0.05078125 | 25/27 | 5 |
| D-RAT11-LowMemGroupSharedDenomPlusLineC | 9 | -0.008246527777777778 | -0.05078125 | 25/27 | 5 |
| D-RAT12-RationalWorkspaceMinStrongDiag | 9 | -0.008680555555555556 | -0.05078125 | 25/27 | 4 |

解释：workspace 已经打开，但 worst_delta 仍低于 `-0.035`，因此继续进入 AUC/trajectory repair。

### 12.3 Rational AUC repair family

执行了四组 Rational AUC / trajectory repairs：

```text
1. 8 epoch, lr=0.002
2. 4 epoch, lr=0.002
3. 3 epoch, lr=0.002
4. 3 epoch, lr=0.0015
```

全部使用：

```text
train_size = 512
val_size = 256
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
linec_seeds = 12319500,12320600,12321600
workspace_csv = v1231_rational_workspace_repair_workspace_truth.csv
```

各组最关键结果：

| run | best candidate | mean_delta_vs_MLP | worst_delta_vs_MLP | max_AUC_time_ratio_vs_MLP | max_CEp99_delta_vs_MLP | LineC pass | near pass |
|---|---|---:|---:|---:|---:|---:|---:|
| 8 epoch lr=0.002 | D-RAT10 | -0.023871527777777776 | -0.1015625 | 1.1970999939645166 | 3.6048946380615234 | 27/27 | 0 |
| 4 epoch lr=0.002 | D-RAT10 | 0.0008680555555555555 | -0.05078125 | 1.2026178692412284 | 5.015604496002197 | 24/27 | 0 |
| 3 epoch lr=0.002 | D-RAT11 | 0.011284722222222222 | -0.0234375 | 1.6854124735148666 | 3.396552562713623 | 25/27 | 0 |
| 3 epoch lr=0.0015 | D-RAT10 | 0.016059027777777776 | -0.01953125 | 1.089952397444073 | 2.6084342002868652 | 20/27 | 0 |

解释：

```text
1. 8 epoch：LineC 全过，但 task/worst/AUC 坏。
2. 4 epoch：mean 接近 0，但 worst/AUC/CEp99 tail 仍坏。
3. 3 epoch：task/worst 明显修复，但 AUC_time 和 CEp99 tail 仍坏。
4. 3 epoch + lr=0.0015 是最接近结果：mean/worst/AUC_time/LineC 同时接近或通过，但 CEp99 tail 仍严重坏化，因此 near pass=0。
```

最接近但不能 promotion 的行：

```text
run = v1231_rational_auc_repair_e3_lr15
candidate = D-RAT10-FusedDenNumReadoutGrad
mean_delta_vs_MLP = +0.016059027777777776
worst_delta_vs_MLP = -0.01953125
max_AUC_time_ratio_vs_MLP = 1.089952397444073
LineC_pass_count = 20/27
max_CEp99_delta_vs_MLP = +2.6084342002868652
candidate_auc_near_pass = 0
```

为什么不 promotion：

```text
1. CEp99_delta 远高于坏化约束，不能把它写成 FamilyNearPass。
2. CEp99/NLL/ECE 在计划中只能作为 audit 与 badness constraint，不能用来设计 CE-tail 方向或修改 loss。
3. 继续用同一 Rational alias 做 epoch/lr 小网格会变成低价值调参，而不是新的 kernel/workspace 或 task-trajectory 机制。
```

### 12.4 Final route after repairs

Finalizer 纳入 repair artifacts 后结果：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 36
basis_family_near_pass_count = 0
rational_workspace_repair_pass_count = 45
rational_auc_near_pass_count = 0
rational_auc_repair_summary_rows = 20
mlp_functional_exploration_pass_rows = 0
line_a_near_anchor_pass_count = 0
code_review_packet_entries = 106
```

最终科学判断：

```text
v12.31 没有达成 S5；
没有 classic FamilyNearPass；
没有 MLP functional generic positive；
不允许 promotion；
合法 route 仍是 R2-BasisTaskLineCNotColocated；
允许 final stop。
```

本轮新增 no-go boundary：

```text
1. Rational workspace blocker 已经被多步 profile repair 进一步打开，45/45 pass。
2. Rational task/worst/LineC/AUC 通过 early-stop 和 conservative LR 可以接近 near-pass。
3. 但 CEp99 tail 严重坏化仍没有合法修复机制；按硬约束不能用 CE-tail 设计方向或改 loss。
4. 我现在不确定继续在同一 D-RAT alias + epoch/lr 局部网格上推进能形成有效机制；继续应转向真正的 rational tail/denominator stability kernel 或合法 calibration-preserving basis mechanism。
```

## 13. 用户再次要求后的 Rational tailnorm output-geometry repair

用户再次要求 v12.31 未达成则继续。复核 route：

```text
route = R2-BasisTaskLineCNotColocated
official_success_reached = 0
rational_auc_near_pass_count = 0
promotion_allowed = 0
```

上一轮最接近结果是 `D-RAT10` 的 `3 epoch + lr=0.0015`，但 CEp99 tail 明显坏化。本轮没有使用 CEp99 作为 direction source，也没有修改 loss；而是复用已有 PrimitiveSpec 里的 label-free stop-gradient logit RMS norm / RMS mix output-geometry repair。

### 13.1 本轮代码修改

修改文件：

```text
dgkan/diagnostics/basis_workspace.py
experiments/run_v1231_finalize_basis_kernel_first.py
```

新增 candidates：

```text
D-RAT13-RationalLogitBatchRMSNormSG
D-RAT14-RationalLogitBatchRMSMixSG025
D-RAT15-RationalLogitBatchRMSMixSG050
```

映射说明：

```text
D-RAT13 -> B7md：B7lp anchor + batch-level stop-gradient logit RMS normalization
D-RAT14 -> B7mg：25% batch RMS norm mix + original logits
D-RAT15 -> B7mh：50% batch RMS norm mix + original logits
```

合理性：

```text
1. 这些 mechanism 来自已有 dgkan PrimitiveSpec，不在 runner 中硬写核心逻辑。
2. 它们使用 stop-gradient logit RMS / mix，是 label-free output geometry mechanism。
3. 本轮只做 diagnostic repair；promotion gate 不降低，CEp99 仍只作为坏化约束。
```

### 13.2 Workspace result

执行规模：

```text
candidates = D-RAT13,D-RAT14,D-RAT15
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
workspace_profile_steps = 3
rows = 27
```

结果：

| candidate | rows | workspace pass | strong pass | max step ratio | skipped rows |
|---|---:|---:|---:|---:|---|
| D-RAT13-RationalLogitBatchRMSNormSG | 9 | 8 | 2 | 1.8077725373657476 | Fashion-MNIST seed 0 |
| D-RAT14-RationalLogitBatchRMSMixSG025 | 9 | 9 | 4 | 1.714252430390903 | none |
| D-RAT15-RationalLogitBatchRMSMixSG050 | 9 | 9 | 4 | 1.441377746831491 | none |

结论：tailnorm family 大部分能过 workspace gate，但不是全面 strong pass；D-RAT13 有一个 row 因 step ratio 超过 1.75 被 skip。

### 13.3 AUC / task / LineC result

执行设置：

```text
epochs = 3
lr = 0.0015
workspace_csv = v1231_rational_tailnorm_workspace_repair_workspace_truth.csv
```

Summary：

| candidate | executed | skipped | mean_delta_vs_MLP | worst_delta_vs_MLP | max_AUC_time_ratio_vs_MLP | max_CEp99_delta_vs_MLP | LineC pass | near pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D-RAT13-RationalLogitBatchRMSNormSG | 8 | 1 | -0.07177734375 | -0.1171875 | 3.117835794920232 | 0.1939530372619629 | 6/24 | 0 |
| D-RAT14-RationalLogitBatchRMSMixSG025 | 9 | 0 | -0.004774305555555556 | -0.0390625 | 1.735411371440807 | 2.5851240158081055 | 25/27 | 0 |
| D-RAT15-RationalLogitBatchRMSMixSG050 | 9 | 0 | -0.021701388888888888 | -0.05078125 | 1.5925863296468374 | 2.3133020401000977 | 19/27 | 0 |

解释：

```text
1. D-RAT13 的 CEp99 tail 明显改善，但 task / AUC / LineC 全面失败，且有 skipped row。
2. D-RAT14 保留了较好的 LineC 与接近的 mean_delta，但 worst、AUC_time、CEp99 仍失败。
3. D-RAT15 介于两者之间，仍未达成 near-pass。
4. 因此 stop-gradient logit RMS / RMS mix 没有解决 v12.31 的 task-LineC-AUC-tail 同位问题。
```

### 13.4 Final route after tailnorm

Finalizer 纳入 tailnorm artifacts 后：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
rational_workspace_repair_pass_count = 45
rational_tailnorm_workspace_repair_pass_count = 26
rational_auc_near_pass_count = 0
rational_auc_repair_summary_rows = 23
mlp_functional_exploration_pass_rows = 0
line_a_near_anchor_pass_count = 0
code_review_packet_entries = 117
```

最终判断不变：

```text
v12.31 没有达成 S5；
没有 classic FamilyNearPass；
没有 MLP functional generic positive；
不允许 promotion；
合法 route 仍是 R2-BasisTaskLineCNotColocated；
允许 final stop。
```

本轮新增 no-go boundary：

```text
1. label-free output geometry tailnorm 可以降低 CEp99 tail，但会牺牲 task/AUC/LineC。
2. RMS mix 能保留部分 task/LineC，但 CEp99/AUC/worst 仍不闭合。
3. 当前已经测试过 workspace repair、early trajectory、conservative LR、output-geometry RMS norm/mix；我现在不确定继续在同一 Rational alias family 上做局部组合能形成有效机制。
4. 下一步若继续，应进入新的机制版本：真正 denominator/tail stability kernel 或更强的 label-free calibration-preserving basis mechanism，而不是继续排列 D-RAT token。
```

## 14. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v12.31 是否达成目标，若未达成则继续。读取最终 `v1231_route_decision.json` 后，结论没有变化：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
basis_workspace_pass_count = 36
basis_workspace_strong_pass_count = 22
rational_workspace_repair_pass_count = 45
rational_tailnorm_workspace_repair_pass_count = 26
rational_auc_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

因此最终判断仍是：

```text
v12.31 没有达成 S5；
没有 classic FamilyNearPass；
没有 MLP functional generic positive；
不允许 promotion；
合法 route 仍是 R2-BasisTaskLineCNotColocated；
允许 final stop。
```

没有新增实验的原因：v12.31 已执行计划 fallback depth 6，完成 basis workspace truth、basis task hardening、MLP functional no-go test、Line A monitor、Rational multi-step workspace repair、Rational AUC/trajectory repair、conservative LR repair、tailnorm output-geometry repair 和 finalizer/provenance 审计。当前已经满足 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1`。我已经不确定继续在同一 Rational alias / tailnorm token family 上扩局部变体会形成有效机制；继续执行会变成低价值网格搜索。

## 15. 用户再次追问后的 stop-contract 复核 2

再次读取最终 `v1231_route_decision.json` 后，结论没有变化：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
rational_workspace_repair_pass_count = 45
rational_tailnorm_workspace_repair_pass_count = 26
rational_auc_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

最终判断仍是：

```text
v12.31 没有达成 S5；
没有 classic FamilyNearPass；
没有 MLP functional generic positive；
不允许 promotion；
合法 route 仍是 R2-BasisTaskLineCNotColocated；
允许 final stop。
```

没有新增实验的原因不变：计划内 fallback 和已追加的 Rational repair / tailnorm repair 均已执行，当前没有明确的新机制假设。继续在同一 Rational alias / tailnorm token family 上排列局部组合会变成低价值网格搜索。

## 16. 用户再次追问后的 stop-contract 复核 3

再次读取最终 `v1231_route_decision.json` 后，结论仍未变化：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
rational_workspace_repair_pass_count = 45
rational_tailnorm_workspace_repair_pass_count = 26
rational_auc_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

因此最终判断仍是：

```text
v12.31 没有达成 S5；
没有 classic FamilyNearPass；
没有 MLP functional generic positive；
不允许 promotion；
合法 route 仍是 R2-BasisTaskLineCNotColocated；
允许 final stop。
```

本次没有新增训练实验。补充审计说明：曾尝试整理执行日志后半段顺序，但整理命令因文档内嵌 `PY` here-doc 与外层 here-doc 冲突而失败；该失败没有写入文件、没有改变任何实验 CSV/JSON 指标，也没有启动新增训练。随后确认无遗留训练进程，并把说明追加进执行日志。

最新 code packet sha 以最终 `v1231_route_decision.json` 为准。
