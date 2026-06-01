# DG-KAN v14.12.1 FunctionalContinueOpen TransferObservability AllBasis 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 synthetic S3、real-lite diagnostic、substrate eligibility 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v14.12.1 的核心修正是 promotion fail-closed、exploration continue-open。Line E2/V2 即使没有达到 promotion-enabling gate，也必须继续执行 F-Diag、Line D/M/C，并输出 failure taxonomy。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py
```

实现：

```text
1. Line R provenance / no-action-search / forbidden-information audit。
2. Line E2 synthetic-real predictivity rebuild。
3. Line V2 train-stream proxy expansion，包含 split-window、micro-horizon h1/h2/h4 proxy、drift-diffusion、recovery-lag、state disagreement、degree stability。
4. Line F-Diag D-CHE real-lite diagnostic。
5. Line D all-basis cross-version reconciliation，并复写 v14.11 D-FOU/RBF/WAV hardening rows。
6. Line M MLP/generic control replay。
7. Line C tail/LineC/failure taxonomy。
8. required manifest、figures、code review packet、执行日志和复盘日志。
9. 发现 required manifest 自身首次生成顺序导致 route 被误判 R0 后，修复为预创建 manifest 占位再 finalizer。
10. 按 Case D 补跑 v14.9 exact lineage replay：
    D-FOU26-NoMaterializeLifetimeAuditV2、
    D-RBF25-WidthConditionGuardNoTaskBranch。
11. 用户再次要求继续后，按 F-Diag protocol alignment 补跑：
    train_steps=200,
    fms_update_interval=200,
    linec_seeds=12319500,12319501,12319502。
12. protocol alignment 后仍未达 4/9，因此继续做全局 FMS amplitude sensitivity：
    strength=0.02 与 strength=0.10。
13. 用户再次要求继续后，补齐计划 V2-B：
    新增实际 bounded micro-horizon h=1/2/4 probe；
    只使用 synthetic train split B1/B2 和 train-stream gradient/FMS state；
    不提交 probed update，不使用 validation/test/LineC/tail/AUC/calibration 生成方向。
14. 随后修正 F-Diag D4 语义：
    FMS-D4 仍使用预注册 token，但开启 actual_micro_horizon_gating；
    gate 只由当前 train batch B1/B2 micro-horizon loss-integral / recovery-lag 计算；
    不读取 real fail pattern 或任何 audit metric。
15. D4 h=1 未打开后，继续做全局 horizon sensitivity：
    fdiag_micro_horizon_steps=2 与 4；
    仍为全局设置，不按 dataset/seed 分支。
```

F-Diag method alias：

```text
FMS-D1 -> F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection
FMS-D2 -> F-CHE-RT1-TrainSplitAgreement
FMS-D3 -> F-CHE6-PhaseScheduleDegreeFMS
FMS-D4 -> F-CHE-RT4-CompositeTransferTrust + actual_micro_horizon_gating
FMS-D5 -> F-CHE-RT3-DegreeEnergySafetyProjection
```

这些 alias 记录在 `v1412_1_method_surface_manifest.csv`，不新增 F-CHE8/F-CHE9，不启动 controller/action bank。

## 3. Line E2 结果

```text
best_feature = D-CHE_degree_energy_delta
best_auc_to_real_pass = 0.6448134467450608
best_spearman_to_real_source = 0.025091648721839714
line_e2_exploration_gate_pass = 1
```

## 4. Line V2 结果

```text
best_proxy = V2-D_actual_micro_horizon_h1_recovery_lag
best_proxy_auc = 0.6565
line_v2_exploration_gate_pass = 1
control_explains_count = 3
```

### 4.1 Actual micro-horizon probe

```text
micro_horizon_probe_rows = 630
V2-B_actual_micro_horizon_h1_loss_integral: rows=210, AUC=0.65375, pass=1
V2-D_actual_micro_horizon_h1_recovery_lag: rows=210, AUC=0.6565, pass=1
V2-B_actual_micro_horizon_h2_loss_integral: rows=210, AUC=0.611, pass=1
V2-D_actual_micro_horizon_h2_recovery_lag: rows=210, AUC=0.621, pass=1
V2-B_actual_micro_horizon_h4_loss_integral: rows=210, AUC=0.589, pass=0
V2-D_actual_micro_horizon_h4_recovery_lag: rows=210, AUC=0.5875, pass=0
```

判断：

```text
actual micro-horizon probe 是 observability diagnostic；
它不执行 real training，不改变 F-Diag method surface，不允许 promotion。
```

## 5. F-Diag real-lite 结果

```text
real_lite_dataset_seed_pass_count = 1/9
real_lite_fms_mean_source_vs_best_control = -0.03228502604696486
real_lite_source_fail_count = 40
real_lite_auctime_fail_count = 35
real_lite_tail_fail_count = 48
real_lite_linec_fail_count = 19
```

## 6. Line D all-basis reconciliation

```text
best_non_dche_family = D-FOU
best_non_dche_dataset_seed_pass_count = 2/9
historical_best_non_dche_dataset_seed_pass_count = 6/9
line_d_official_fms_eligible_family_count = 0
```

## 6.1 F-Diag strength sensitivity

```text
strength0.02: pass=1/9, mean_source=-0.03222594194942051, route=R4-AllBasisSubstrateBlocked
strength0.10: pass=1/9, mean_source=-0.03235950536198086, route=R4-AllBasisSubstrateBlocked
```

判断：

```text
strength 0.02 / 0.10 均没有打开 real-lite >=4/9；
唯一 pass 仍来自 Fashion-MNIST seed0 的 FMS-D5-RecoveryLagSuppressed；
因此当前 blocker 不是简单 FMS amplitude。
```

## 6.2 F-Diag D4 actual micro-horizon gating

```text
rows=9, pass=0/9, mean_source=-0.03750330540868971, best_source=0.005054891109466553, source_fail=8, AUC_fail=8, tail_fail=10, LineC_fail=7
```

判断：

```text
actual D4 micro-horizon gating 没有打开 real-lite；
D4 仍被 source/AUC/tail/LineC 混合阻断，且没有达到 4/9 exploration threshold。
```

### 6.3 D4 horizon sensitivity

```text
h1: route=R4-AllBasisSubstrateBlocked, real_lite_pass=1/9, D4_pass=0/9, D4_mean_source=-0.03750330540868971, overall_mean_source=-0.03228502604696486
h2: route=R4-AllBasisSubstrateBlocked, real_lite_pass=1/9, D4_pass=0/9, D4_mean_source=-0.037503298785951406, overall_mean_source=-0.0322850247224172
h4: route=R4-AllBasisSubstrateBlocked, real_lite_pass=1/9, D4_pass=0/9, D4_mean_source=-0.03750327891773648, overall_mean_source=-0.032285020748774214
```

判断：

```text
h=2 / h=4 均没有提高 real-lite pass；
micro-horizon 长度不是当前有效杠杆。
```

## 7. 最终 route

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 8. 科学结论

```text
1. v14.12.1 已执行 Line R/E2/V2/F-Diag/D/M/C/Z。
   用户继续要求后，已补齐实际 micro-horizon h=1/2/4 train-stream probe。
2. 当前结果没有达成 S5，promotion_allowed = 0。
3. real-lite diagnostic 不能被写成 official real success。
4. all-basis reconciliation 显示 D-CHE 外仍没有 official FMS eligible family；
   v14.9 D-FOU/RBF historical 6/9 在 v14.12.1 exact replay 下没有复现到 >=6/9。
5. 后续若继续，需要新的计划或在 v14.12.1 允许的 alias/proxy 框架内继续，不允许 action/controller/reset route。
```

## 9. 用户再次追问后的状态复核

本次没有新增训练；只复核最新 artifact 和计划 Case B/C/D/E 边界，确认是否仍有不越界的继续分支。

复核对象：

```text
results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1/v1412_1_route_decision.json
results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1/v1412_1_required_manifest.csv
results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1/v1412_1_train_stream_proxy_expansion.csv
results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1/v1412_1_real_lite_dche_fms_diagnostic.csv
results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1/v1412_1_allbasis_reconciliation.csv
```

复核结果：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
line_v2_best_auc = 0.6565
line_v2_exploration_gate_pass = 1
line_v2_micro_horizon_probe_rows = 630
real_lite_dataset_seed_pass_count = 1/9
line_d_best_non_dche_dataset_seed_pass_count = 2/9
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
manifest_missing_sum = 0
```

计划分支复核：

```text
Case B 已覆盖：
  split B1/B2 agreement、actual micro-horizon h=1/2/4、
  recovery-lag、AdamW/FMS state disagreement、degree-energy stability proxy。

Case C 已覆盖：
  controls / source / AUCtime / tail-calibration / LineC failure taxonomy。

Case D 已覆盖：
  v14.9 exact candidate replay、gate/run-budget/finalizer mismatch 检查；
  D-FOU/D-RBF historical 6/9 没有在 v14.12.1 replay 中复现到 >=6/9。

Case E 不触发：
  D-CHE no-regression check 没有显示 substrate regression；
  因此不启动 D-CHE lineage repair。
```

当前停止边界：

```text
我现在已经不确定如何在 v14.12.1 当前计划内继续安全推进到 S4/S5，
而不新增 method token/action/controller/reset route，
不使用 real dataset/seed fail pattern 作方向，
不把 LineC/CEp99/NLL/ECE/AUCtime audit metric 变成 direction，
不把 real-lite 1/9、V2 diagnostic AUC 0.6565 或 Line D 2/9 写成 official success。

因此本次不再启动新的 v14.12.1 训练。
当前合法结论仍是：
  route = R4-AllBasisSubstrateBlocked
  minimum_success = S3-DCHESyntheticFMSPass
  official_s5_reached = 0
  promotion_allowed = 0
```

## 11. 用户再次追问后的三次状态复核

本次没有新增代码修改，也没有新增训练；重点重新读取计划原文中 Line F-Diag、Line D、Case B/C/D/E 与 route definitions，确认是否还有未执行的合法修复方向。

复核到的计划边界：

```text
1. F-Diag real-lite <4/9 时，Case C 只允许：
   controls explainability / source / AUCtime / tail-calibration / failure taxonomy。
   不允许新增 method token。
2. D-FOU/RBF/WAV cross-version 不一致时，Case D 只允许：
   replay v14.9 exact IDs / 对齐 gate / 对齐 run budget / 检查 finalizer mismatch。
3. D-CHE substrate regression 时才触发 Case E。
4. D-FOU/RBF/WAV 未达到 >=6/9 前，不得进入 official FMS proof。
```

artifact 覆盖复核：

```text
v1412_1_d_fou_hardening.csv rows = 45
v1412_1_d_rbf_hardening.csv rows = 45
v1412_1_d_wav_monitor.csv rows = 27
v1412_1_allbasis_reconciliation.csv:
  D-FOU v1412_replay_dataset_seed_pass_count = 2
  D-RBF v1412_replay_dataset_seed_pass_count = 1
  D-WAV v1412_replay_dataset_seed_pass_count = missing / not opened
v1412_1_dche_no_regression.csv rows = 3
v1412_1_failure_taxonomy.csv rows = 45
```

三次复核结论：

```text
1. v14.12.1 仍没有达成 S4/S5。
2. F-Diag <4/9 的 Case C 已通过 failure taxonomy / controls / blocker 分解覆盖。
3. Line D 的 D-FOU/RBF/WAV 允许方向已通过 hardening artifacts 与 v14.9 exact replay 覆盖，
   但 best 非 D-CHE family 仍未达到 >=6/9 exploration gate。
4. Case E 不触发，因为 D-CHE no-regression check 没有显示 substrate regression。
5. 当前继续推进需要新的计划，而不是在 v14.12.1 内新增 method/action/controller/reset route。
```

当前合法结论仍是：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0
```

## 12. 用户再次追问后的代码层方法面复核

本次没有新增代码修改，也没有新增训练；进一步读取 v14.12.1 runner 中实际注册的 method alias / Line D candidates，并与已产出 artifact 对齐，确认是否存在“计划允许但漏跑”的 method 或 substrate candidate。

代码层 method surface：

```text
FMS-D1-GenericValueDegreeSafety -> F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection
FMS-D2-TrainStreamTransferUtilityGated -> F-CHE-RT1-TrainSplitAgreement
FMS-D3-ContinuousLowAmplitudeDegreePhase -> F-CHE6-PhaseScheduleDegreeFMS
FMS-D4-MicroHorizonLossIntegralGated -> F-CHE-RT4-CompositeTransferTrust + actual_micro_horizon_gating
FMS-D5-RecoveryLagSuppressed -> F-CHE-RT3-DegreeEnergySafetyProjection
```

artifact 覆盖：

```text
v1412_1_method_surface_manifest.csv rows = 15
v1412_1_real_lite_dche_fms_diagnostic.csv rows = 45
  methods = FMS-D1..D5
  datasets = MNIST,Fashion-MNIST,KMNIST
  seeds = 0,1,2
  candidate = D-CHE17-HighDegreeLateEnableSubstrate

v1412_1_real_lite_controls.csv rows = 45
  controls = C0..C4

v1412_1_d_fou_hardening.csv rows = 45
  candidates = D-FOU26,D-FOU27,D-FOU28,D-FOU29,D-FOU30

v1412_1_d_rbf_hardening.csv rows = 45
  candidates = D-RBF25,D-RBF26,D-RBF27,D-RBF28,D-RBF29

v1412_1_d_wav_monitor.csv rows = 27
  candidates = D-WAV25,D-WAV26,D-WAV27
```

判断：

```text
1. F-Diag 的预注册 FMS-D1..D5 没有漏跑。
2. Line D 中 v14.9 exact replay candidates 与 v14.12.1 allowed hardening candidates 没有漏跑。
3. D-FOU/D-RBF/D-WAV artifacts 均保持 official_fms_proof_executed = 0；
   这是因为 substrate exploration gate 未达 >=6/9，符合计划约束。
4. 继续推进需要新的 transfer-observable FMS definition 或新的计划；
   不能在 v14.12.1 当前计划内临时新增 method/action/controller/reset route。
```

当前合法结论不变：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0
```

## 10. 用户再次追问后的二次状态复核

本次没有新增代码修改，也没有新增训练；只再次读取最新 route / manifest / proxy / real-lite / all-basis reconciliation，并对照计划 Case B/C/D/E 与 hard-stop 条款确认是否还有可继续执行且不越界的分支。

复核结果：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
line_e2_exploration_gate_pass = 1
line_v2_best_auc = 0.6565
line_v2_exploration_gate_pass = 1
line_v2_micro_horizon_probe_rows = 630
real_lite_dataset_seed_pass_count = 1/9
line_d_best_non_dche_dataset_seed_pass_count = 2/9
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
manifest_missing_sum = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

二次判断：

```text
1. v14.12.1 仍没有达成 S4/S5。
2. Line V2 的 best AUC = 0.6565 只允许 exploration-open，不允许 promotion。
3. F-Diag real-lite 仍只有 1/9，低于继续写成 transfer success 的任何门槛。
4. Line D 非 D-CHE best replay 仍只有 2/9，不能打开 D-FOU/D-RBF/D-WAV official FMS proof。
5. required artifacts 缺失仍为 0，forbidden / no-action-search violation 仍为 0。
```

停止边界不变：

```text
我现在仍不确定如何在 v14.12.1 当前计划内继续安全推进到 S4/S5，
而不新增 method token / action token / controller / reset route，
不使用 real dataset/seed fail pattern 作方向，
不把 LineC/CEp99/NLL/ECE/AUCtime audit metric 变成 direction，
不把 real-lite 1/9、V2 diagnostic AUC 0.6565 或 Line D 2/9 写成 official success。

因此本次不再启动新的 v14.12.1 训练。
当前合法结论仍是：
  route = R4-AllBasisSubstrateBlocked
  minimum_success = S3-DCHESyntheticFMSPass
  official_s5_reached = 0
  promotion_allowed = 0
```
