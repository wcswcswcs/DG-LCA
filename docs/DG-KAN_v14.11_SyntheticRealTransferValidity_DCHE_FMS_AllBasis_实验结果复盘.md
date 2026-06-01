# DG-KAN v14.11 SyntheticRealTransferValidity DCHE FMS AllBasis 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入本轮实际 artifact 中的结果；不虚构成功、不补填未执行 real 训练、不把 synthetic S3 / real 2/9 / substrate eligibility 写成 promotion。

## 1. 计划理解

v14.11 的目标是验证 synthetic S3 是否能预测 real transfer，并审计 train-stream-only real-transfer value proxy 是否可观测。只有 Line E 或 Line V 通过，才允许继续 D-CHE FMS definition reset；否则 fail-closed。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py
```

实现：

```text
1. Line R provenance / no-action-search / forbidden-information audit。
2. Line E synthetic-to-real alignment 与 predictivity summary。
   继续推进后补入 v14.3 Rational K8 synthetic/real 和 v14.4 Rational S4b real reference。
3. 继续推进后补入 v14.5 action-bank oracle diagnostics / missing-state reference，
   只作为 audit-only upper-bound 证据，不作为方向源。
4. Line V train-stream telemetry proxy validity。
5. D-CHE synthetic/real/MLP control artifact replay。
6. v14.9 all-basis substrate status replay。
7. 继续推进后执行 v14.11 Line D substrate-only hardening：
   D-FOU27..30 / D-RBF26..29 / D-WAV25..27。
8. LineC/tail failure taxonomy、figures、required manifest、code review packet。
```

修复：

```text
1. 首次 Line R scan 使用源码字符串搜索，误把审计规则文本本身判为 F-CHE8/controller/action-token 证据。
2. 已改为读取 v14.10/v14.11 artifact 中的 method tokens 与 flag columns：
   controller_executed、is_action_token_extension、uses_dataset_name_branch、uses_seed_specific_scale。
3. 修复后 no_action_search_violation_count = 0，
   forbidden_information_violation_count = 0。
4. split-batch probe 首次运行遇到 direct script import path blocker；
   已在 runner 开头加入 repo root 到 sys.path。
5. 调整 code review packet 生成顺序，确保最终执行日志/复盘日志写入后再打包。
```

合法性：

```text
1. 没有新增 F-CHE8/F-CHE9。
2. 没有 controller / action token / reset route。
3. 没有执行新的 real 训练。
4. LineC / CEp99 / NLL / ECE / AUCtime 只作为 audit / gate。
5. promotion_allowed = 0。
```

## 3. Line E 结果

```text
split = D-CHE-internal-all
rows = 3780
positive_real_pass_rows = 168
spearman_synthetic_source_real_source = -0.00516569843263
auc_predict_synthetic_features_to_real_pass = 0.543990270527
predictivity_gate_pass = 0
```

判断：

```text
Line E gate 要求 AUC >= 0.70 且 Spearman >= 0.30。
本轮 Line E pass = 0.
```

v14.5 oracle reference：

```text
v145_oracle_reference_rows = 20
v145_missing_state_reference_rows = 50
v145_best_oracle_dataset_seed_pass_count = 7
v145_action_bank_upper_bound_pass = 0
```

判断：

```text
v14.5 action-bank oracle diagnostics 是计划 Line E 的输入源之一。
本轮只作为 audit-only reference 读取；oracle_uses_audit_metric = 1 的历史 oracle 不允许成为 v14.11 direction。
best oracle 仍未达到 9/9，因此不能改写 S5，也不能解除 v14.11 的 R1/R2 blocker。
```

## 4. Line V 结果

```text
synthetic U_train AUC = 0.362215061463
real U_train AUC = 0.473837209302
U_DCHE_minus_U_MLP = -0.0517916755588
split_batch_probe_rows = 420
split_batch_probe_auc_U_train_to_synthetic_pass = 0.394617496121
split_batch_probe_gate_pass = 0
split_batch_control_probe_rows = 1260
split_batch_control_explains_fms_count = 3
Line V pass = 0
```

判断：

```text
Line V 使用已有 train-stream/model-state telemetry 做 proxy audit。
追加的 split-batch probe 使用 synthetic train-stream B1/B2 one-step counterfactual U_train，并记录 pre-registered controls。
两者都没有使用 validation/test/future/query 或 LineC/tail/AUC/calibration 生成方向。
```

## 5. Line D 继续推进结果

执行范围：

```text
D-FOU27-LowFreqIdentityResidualV2
D-FOU28-BandwiseSNRSafeWarmup
D-FOU29-PhaseStableBandMixNoHighFreq
D-FOU30-NoMaterializeLifetimeV3
D-RBF26-ActiveCenterOccupancyV2
D-RBF27-WidthConditionIdentityResidual
D-RBF28-CompactBumpNoDenseMaterialization
D-RBF29-GaussianLocalK4TaskHealth
D-WAV25-TriangularSupportV3
D-WAV26-ScaleOccupancyNoTailTarget
D-WAV27-LocalSupportOverlapDamping
```

结果：

```text
line_d_v1411_route = R8-NonRATSubstrateStillMissing
line_d_v1411_candidate_rows = 99
line_d_v1411_best_family_dataset_seed_pass_count = 0
line_d_v1411_exploration_open_family_count = 0
line_d_v1411_official_fms_eligible_family_count = 0
```

判断：

```text
这轮 Line D 是 substrate-only hardening，不执行 FMS proof。
新增 D-FOU/RBF/WAV candidates 没有打开 exploration 或 official FMS eligibility。
因此 D-FOU/D-RBF/D-WAV 仍不能进入 official FMS proof，也不能改变 v14.11 D-CHE real-transfer route。
```

## 6. 最终 route

```text
route = R1-SyntheticGateNotPredictiveForRealTransfer
minimum_success = S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count = 5
real_dataset_seed_pass_count = 2
v145_best_oracle_dataset_seed_pass_count = 7
v145_action_bank_upper_bound_pass = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 7. 科学结论

```text
1. v14.10 的 D-CHE synthetic S3 仍是已达成事实，但 v14.11 没有把它改写成 real success。
2. v14.10 best real 仍只有 2/9，未达到 S4/S5。
3. v14.11 Line E 与 Line V 均未打开继续 D-CHE FMS definition reset 的合法门。
4. 因此本轮 fail-closed，不执行新的 F-CHE token / action search / real 训练。
5. promotion_allowed = 0。
```

## 8. Blocker 与停止边界

```text
Line E blocker:
  synthetic source 与 real source 的 Spearman 接近 0 且为负；
  AUC_predict 只有约 0.544，低于 0.70。
  补入 Rational reference 后仍以 D-CHE internal gate 为准；
  v14.11 要求至少 D-CHE internal split 通过，因此 route 不变。

Line V blocker:
  telemetry-only U_train 对 synthetic pass 的 AUC 约 0.362；
  对 real pass 的 AUC 约 0.474；
  split-batch B1/B2 U_train 对 synthetic pass 的 AUC 约 0.394617496121；
  split-batch control_explains_fms_count = 3；
  D-CHE mean U_train 低于 MLP/generic control。

Line D blocker:
  v14.11 允许的 D-FOU/RBF/WAV substrate candidates 共 99 rows；
  best family dataset-seed pass count = 0/9；
  exploration_open_family_count = 0；
  official_fms_eligible_family_count = 0。

因此继续训练会违反 v14.11：
  不能新增 F-CHE8/F-CHE9；
  不能用 real fail pattern 或 LineC/tail/AUC/calibration 指标反推方向；
  不能把 D-CHE synthetic S3 或 real 2/9 写成 S4/S5。
```

## 9. 用户再次追问后的最终状态复核

本次没有新增代码修改，也没有新增训练；只复核最新 artifact 和计划 stop/go 边界，确认是否还有可继续执行且不越界的分支。

复核对象：

```text
results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411/v1411_route_decision.json
results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411/v1411_progress_table.csv
results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411/v1411_failure_taxonomy.csv
results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411/v1411_required_artifact_manifest.csv
```

复核结论：

```text
v14.11 没有达成最终目标。

route = R1-SyntheticGateNotPredictiveForRealTransfer
minimum_success = S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count = 5/7
real_dataset_seed_pass_count = 2/9
line_e_pass = 0
line_v_pass = 0
line_fche_executed = 0
line_d_v1411_route = R8-NonRATSubstrateStillMissing
line_d_v1411_best_family_dataset_seed_pass_count = 0/9
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
```

已尝试但未打开目标：

```text
1. Line E D-CHE synthetic-to-real predictivity audit。
2. v14.3/v14.4 Rational reference 扩展。
3. v14.5 oracle / missing-state audit-only reference。
4. Line V telemetry proxy。
5. Line V split-batch B1/B2 U_train probe。
6. split-batch NoOp / RandomMatched / SameActiveFraction controls。
7. v14.11 Line D D-FOU27..30 / D-RBF26..29 / D-WAV25..27 substrate-only hardening。
```

当前停止边界：

```text
我现在已经不确定如何在 v14.11 当前计划内继续安全推进到 S4/S5，
而不新增 F-CHE8/F-CHE9 或 action token，
不启动 controller/reset route，
不使用 real dataset/seed fail pattern 作方向，
不把 LineC/CEp99/NLL/ECE/AUCtime audit metric 变成 direction，
不把 synthetic S3、real 2/9、v14.5 oracle 7/9 或 Line D local rows 写成 official success。

因此本次不再启动新的 v14.11 训练。
当前合法结论仍是：
  route = R1-SyntheticGateNotPredictiveForRealTransfer
  minimum_success = S3-DCHESyntheticFMSPass
  official_s5_reached = 0
  promotion_allowed = 0
```

## 10. 用户再次追问后的二次状态复核

本次没有新增代码修改，也没有新增训练；只再次读取最新 route / manifest，确认状态是否变化。

复核结果：

```text
route = R1-SyntheticGateNotPredictiveForRealTransfer
minimum_success = S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count = 5/7
real_dataset_seed_pass_count = 2/9
line_e_pass = 0
line_v_pass = 0
line_fche_executed = 0
line_d_v1411_route = R8-NonRATSubstrateStillMissing
line_d_v1411_best_family_dataset_seed_pass_count = 0/9
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
manifest_missing_sum = 0
```

结论不变：

```text
v14.11 没有达成 S4/S5。
当前计划内能合法执行的 Line E / Line V / v14.5 oracle reference / Line D substrate-only hardening 都已覆盖。
继续推进需要新的计划，而不是在 v14.11 内新增 F-CHE/action/controller 或使用 real fail audit pattern 反推方向。
```
