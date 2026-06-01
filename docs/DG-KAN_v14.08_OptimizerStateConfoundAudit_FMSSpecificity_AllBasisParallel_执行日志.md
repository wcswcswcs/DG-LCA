# DG-KAN v14.8 OptimizerStateConfoundAudit FMSSpecificity AllBasisParallel 执行日志

生成时间：2026-05-29（Asia/Singapore）

本日志记录实际执行过的命令、代码文件和 artifact 位置；不补写未执行命令。

## 1. 计划与上下文读取

计划文件：

```text
docs/DG-KAN_v14.8_OptimizerStateConfoundAudit_FMSSpecificity_AllBasisParallel_完整计划.md
```

相关输入 artifacts：

```text
results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461_after_p1_transport/
results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/p1_transport_zero_moment_reset_v1461/
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/official_v147/
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/repair_v147_event_only_transport/
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/repair_v147_event_only_delta_top25/
```

## 2. 代码修改

修改文件：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

新增文件：

```text
experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py
```

修改目的：

```text
1. 给 v144 train_case 增加 AdamW beta1 / beta2 控制。
2. 增加 generic optimizer moment reset controls。
3. 增加 optimizer-state transport post-event recovery window。
4. 增加 v144 overhead 粗粒度计时字段：
   per_example_gradient_time_sec
   fms_state_update_time_sec
   projection_time_sec
   state_transport_time_sec
   optimizer_update_time_sec
   sync_time_sec
   artifact_logging_time_sec
   linec_audit_time_sec
5. 新增 v14.8 runner，输出计划要求的 confound audit artifacts。
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py
```

结果：

```text
py_compile pass
```

## 3. Smoke

执行命令：

```bash
conda run -n kan python experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py \
  --out-dir results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel/smoke_v148 \
  --datasets MNIST \
  --seeds 0 \
  --methods G0-RAT-AdamW,G1-RAT-AdamW-Beta1Zero,S0-RAT-FMS-NoStateTransport,S1-RAT-FMS-ZeroMomentResetAffectedEventOnly,S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow \
  --mlp-methods M0-MLP-AdamW,M5-MLP-AdamW-Beta1Zero \
  --train-steps 4 \
  --batch-size 8 \
  --linec-mode none \
  --compute-budgeted-run 1
```

输出目录：

```text
results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel/smoke_v148/
```

结果：

```text
route = R3-GenericOptimizerStateResetConfound
expected_dataset_seed_count = 1
best_fms_method = S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow
best_fms_strict_pass_count = 1 / 1
best_fms_endpoint_vs_adamw_pass_count = 1 / 1
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

说明：

```text
smoke 只验证 runner / artifact surface 可执行；
不能作为 official success 或 confound scientific conclusion。
```

Timing 埋点后补 smoke：

```bash
conda run -n kan python experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py \
  --out-dir results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel/smoke_v148_timing \
  --datasets MNIST \
  --seeds 0 \
  --methods G0-RAT-AdamW,S1-RAT-FMS-ZeroMomentResetAffectedEventOnly \
  --mlp-methods M0-MLP-AdamW \
  --train-steps 4 \
  --batch-size 8 \
  --linec-mode none \
  --compute-budgeted-run 1
```

结果：

```text
route = R3-GenericOptimizerStateResetConfound
expected_dataset_seed_count = 1
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 4. Official v14.8

执行命令：

```bash
conda run -n kan python experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py \
  --out-dir results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel/official_v148 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods G0-RAT-AdamW,G1-RAT-AdamW-Beta1Zero,G2-RAT-AdamW-Beta1Half,G3-RAT-AdamW-PeriodicMomentReset,G4-RAT-AdamW-EventMatchedRandomReset,G5-RAT-AdamW-FullMomentResetAtFMSIntervalsNoFMS,G6-RAT-AdamW-RMSPropLikeNoMomentum,G7-RAT-AdamW-NoMomentumWarmupThenAdamW,S0-RAT-FMS-NoStateTransport,S1-RAT-FMS-ZeroMomentResetAffectedEventOnly,S2-RAT-FMS-ZeroMomentResetAllFMSRolesEventOnly,S3-RAT-FMS-ProjectedMomentTransportEventOnly,S4-RAT-FMS-RMSRecomputeEventOnly,S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow \
  --mlp-methods M0-MLP-AdamW,M1-MLP-FMS-NoStateTransport,M2-MLP-FMS-ZeroMomentResetAffected,M3-MLP-FMS-ZeroMomentResetRandomCoords,M4-MLP-FMS-FullAdamWStateReset,M5-MLP-AdamW-Beta1Zero,M6-MLP-AdamW-PeriodicMomentReset \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

输出目录：

```text
results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel/official_v148/
```

最终 route：

```text
route = R3-GenericOptimizerStateResetConfound
expected_dataset_seed_count = 9
best_fms_method = S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow
best_fms_strict_pass_count = 0 / 9
best_fms_endpoint_vs_adamw_pass_count = 6 / 9
generic_optimizer_state_confound = 1
mlp_analog_confound = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

Required artifact manifest：

```text
required_count = 25
missing = 0
```

## 5. 用户再次追问后的 stop-go 复核

本次没有新增训练；直接复核 v14.8 完整计划第 15-17 节的 route / no-go 规则。

复核命令：

```bash
cat results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel/official_v148/v148_route_decision.json
sed -n '760,900p' docs/DG-KAN_v14.8_OptimizerStateConfoundAudit_FMSSpecificity_AllBasisParallel_完整计划.md
```

当前 route：

```text
route = R3-GenericOptimizerStateResetConfound
best_fms_method = S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow
best_fms_strict_pass_count = 0 / 9
best_fms_endpoint_vs_adamw_pass_count = 6 / 9
generic_optimizer_state_confound = 1
official_s5_reached = 0
promotion_allowed = 0
```

计划原文边界：

```text
如果是 generic optimizer-state reset / momentum suppression confound，
functional update 主线必须从 optimizer-state reset route 撤出，
回到 FMS definition / substrate / all-basis parallel repair，
而不是继续用 action 或 reset 小修拖延。
```

复核结论：

```text
v14.8 diagnostic 目标已经完成：
回答为 generic optimizer-state reset confound。

v14.8 没有达成 S5 official success。
但继续在 v14.8 内新增 reset variant、调 grid、启动 controller、
或用 audit metric 反推坐标都会违反计划。

因此本次不新增训练，不补填成功，不 promotion。
```

## 6. 用户再次追问后的最终复核

本次仍没有新增训练；再次按计划关键条款复核是否还有 v14.8 内合法继续项。

复核命令：

```bash
rg -n "R3|GenericOptimizer|不能继续|Line D|all-basis|substrate|Codex 禁止|最终判断" \
  docs/DG-KAN_v14.8_OptimizerStateConfoundAudit_FMSSpecificity_AllBasisParallel_完整计划.md

cat results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel/official_v148/v148_route_decision.json
```

复核结果：

```text
1. 当前 route 仍为 R3-GenericOptimizerStateResetConfound。
2. v14.8 计划第 17 节明确：
   若为 generic optimizer-state reset / momentum suppression confound，
   主线必须从 optimizer-state reset route 撤出，
   回到 FMS definition / substrate / all-basis parallel repair。
3. v14.8 计划禁止继续 reset/action 小修、grid、controller、
   或 audit-metric direction。
4. Line D 本轮要求的是 all-basis substrate status；
   v148_all_basis_substrate_status.csv 已输出。
5. 因此继续在 v14.8 内新增训练会违反计划边界。
```

最终执行状态：

```text
不新增训练。
不补填成功。
不 promotion。
下一步需要新预注册计划。
```
