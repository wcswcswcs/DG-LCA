# DG-KAN v14.8 OptimizerStateConfoundAudit FMSSpecificity AllBasisParallel 实验结果复盘

生成时间：2026-05-29（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功，不把 endpoint-only、control-equivalent、diagnostic 或 smoke 写成 promotion。

## 1. 计划理解

v14.8 的目标不是继续 action search，也不是让 zero-moment reset 通过某个局部 gate，而是回答：

```text
v14.7 的 endpoint gain 到底是 FMS-specific，
还是 generic optimizer-state reset / momentum suppression confound？
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline。
2. 不使用 teacher / distillation / loss modification / sampler / class weight。
3. 不使用 dataset-name branch / seed-specific scaling / label-informed initialization。
4. functional direction 不使用 validation / test / future / query。
5. LineC / CEp99 / NLL / ECE / AUCtime / Brier 只作为 audit / gate。
6. 不新增 K-RT / K-AUC / K-FL action token。
7. 不启动 controller。
8. 不做 strength / lambda / lr / refresh grid search。
9. 不把 random/full/generic reset controls 也能复现的结果写成 FMS-specific。
10. 不降低 step_time gate。
```

计划必须执行：

```text
Line Q: v14.6.1 / v14.7 semantic diff。
Line G: generic optimizer-state reset controls。
Line S: FMS specificity test。
Line A: affected-coordinate autopsy。
Line E: overhead / step-time audit。
Line M: MLP analog controls。
Line D: all-basis substrate status。
Line Z: final route / no-go / next hypothesis。
```

## 2. 本轮代码修改

修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

新增：

```text
experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py
```

主要实现：

```text
1. v144 train_case 支持 AdamW beta1 / beta2 控制。
2. v144 train_case 支持 generic optimizer moment reset controls：
   periodic_moment_reset
   event_matched_random_reset
   full_moment_reset_at_fms_intervals
   rmsprop_like_no_momentum
   no_momentum_warmup_then_adamw
3. v144 train_case 支持 optimizer-state transport recovery window。
4. v14.8 runner 输出计划要求的 v148 artifacts 和 route。
5. route 保持 promotion_allowed = 0，除非 official S5 全部 gate 真实达成。
```

语法检查：

```text
py_compile pass
```

## 3. Smoke

执行规模：

```text
datasets = MNIST
seeds = 0
methods = G0,G1,S0,S1,S5
mlp_methods = M0,M5
train_steps = 4
batch_size = 8
linec_mode = none
compute_budgeted_run = 1
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

解释：

```text
smoke 只证明 v14.8 runner / artifact surface 可执行；
不能作为 official success 或 scientific no-go 证据。
```

## 4. Official v14.8 confound audit

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
generic controls = G0..G7
FMS specificity methods = S0..S5
MLP analog controls = M0..M6
train_steps = 200
batch_size = 32
lr = 0.005
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.5
linec_mode = exact
compute_budgeted_run = 1
```

输出目录：

```text
results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel/official_v148/
```

route：

```text
route = R3-GenericOptimizerStateResetConfound
minimum_success = S4c-MechanismSufficientDiagnostic
expected_dataset_seed_count = 9
best_fms_method = S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow
best_fms_strict_pass_count = 0 / 9
best_fms_endpoint_vs_adamw_pass_count = 6 / 9
generic_optimizer_state_confound = 1
mlp_analog_confound = 0
affected_mask_conclusion = A2-AffectedSetIntrinsicallyBroad|A3-SparseAffectedMaskDestroysValue
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 5. Line Q：Semantic diff

关键对照：

| qid | run | strict | endpoint | route | controls endpoint full |
|---|---|---:|---:|---|---:|
| Q0 | v14.6.1 after-P1 diagnostic old metrics | - | 9 | S4c-MechanismSufficientDiagnostic | - |
| Q1 | v14.6.1 P1 zero under v144 runner | not available under v14.7 strict | 9 | S5 in v144 runner | - |
| Q2 | v14.7 every-step K2 | 0 | 9 | R2-ResetWorksButNotFMSSpecific | 3 |
| Q3 | v14.7 event-only K2 | 0 | 4 | R1-ZeroMomentResetNotReplicated | 0 |
| Q4 | v14.7 delta_top25 K2 | 0 | 2 | R1-ZeroMomentResetNotReplicated | 0 |
| Q5 | v14.8 S1 event-only affected | 0 | 4 | current audit | - |
| Q5 | v14.8 S5 post-event window | 0 | 6 | current audit | - |

判断：

```text
1. v14.6.1 的 diagnostic 9/9 不能直接升级为 v14.7/v14.8 strict S5。
2. v14.7 every-step endpoint 9/9 被 controls 复现。
3. v14.7 event-only / sparse reset 均未复现 9/9。
4. v14.8 S5 recovery window 把 endpoint 提到 6/9，但 strict 仍为 0/9。
```

## 6. Line G：Generic optimizer-state reset controls

| method | strict | endpoint-vs-AdamW | efficiency | mean source | median AUC vs AdamW | median step |
|---|---:|---:|---:|---:|---:|---:|
| G0 AdamW | 0 | 0 | 9 | -0.304291 | 1.000000 | 1.000000 |
| G1 Beta1Zero | 0 | 5 | 9 | -0.257457 | 0.922067 | 0.934421 |
| G2 Beta1Half | 0 | 5 | 9 | -0.066033 | 0.948498 | 0.930727 |
| G3 PeriodicMomentReset | 0 | 6 | 9 | -0.162511 | 0.958836 | 0.933953 |
| G4 EventMatchedRandomReset | 0 | 7 | 9 | -0.142358 | 0.944246 | 0.929910 |
| G5 FullMomentResetAtFMSIntervalsNoFMS | 0 | 6 | 9 | -0.162511 | 0.958836 | 0.929744 |
| G6 RMSPropLikeNoMomentum | 0 | 5 | 9 | -0.257457 | 0.922067 | 0.968622 |
| G7 NoMomentumWarmupThenAdamW | 0 | 6 | 9 | -0.143280 | 0.873071 | 0.942701 |

判断：

```text
1. S5 的 endpoint-vs-AdamW = 6/9。
2. Generic G4 event-matched random reset endpoint-vs-AdamW = 7/9，
   超过 S5 的 endpoint coverage。
3. G3/G5/G7 均达到 6/9 endpoint，与 S5 相当。
4. 因此 FMS-specific claim 被 generic optimizer-state reset confound 阻断。
```

## 7. Line S：FMS specificity test

| method | strict | endpoint-vs-AdamW | specificity | efficiency | mean source | median AUC vs best | median AUC vs AdamW | median step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 NoStateTransport | 0 | 2 | 1 | 2 | -0.163891 | 1.191321 | 1.012485 | 1.290792 |
| S1 EventOnlyAffectedZeroReset | 0 | 4 | 4 | 1 | -0.019915 | 1.145349 | 0.962900 | 1.295999 |
| S2 EventOnlyAllRolesZeroReset | 0 | 4 | 4 | 1 | -0.019913 | 1.145349 | 0.962898 | 1.295007 |
| S3 ProjectedMomentTransport | 0 | 0 | 0 | 2 | -0.906318 | 1.569404 | 1.362555 | 1.286782 |
| S4 RMSRecompute | 0 | 0 | 0 | 1 | -551467.586376 | 3231.406175 | 2648.723201 | 1.290566 |
| S5 PostEventRecoveryWindow | 0 | 6 | 5 | 1 | -0.004105 | 1.094224 | 0.928741 | 1.295544 |

S5 row-level blockers：

| dataset | seed | blocker | source | AUC |
|---|---:|---|---:|---:|
| MNIST | 0 | source,AUCtime,specificity | -0.028465 | 1.087139 |
| MNIST | 1 | source,AUCtime,step_time,specificity | -0.001261 | 1.187501 |
| MNIST | 2 | step_time | 0.030311 | 0.979631 |
| Fashion-MNIST | 0 | source,AUCtime,step_time,specificity | -0.130422 | 1.085591 |
| Fashion-MNIST | 1 | source,NLL_tail,ECE_tail,step_time,specificity | -0.317890 | 0.996850 |
| Fashion-MNIST | 2 | AUCtime,step_time | 0.017303 | 1.278345 |
| KMNIST | 0 | AUCtime,ECE_tail,step_time | 0.053367 | 1.108301 |
| KMNIST | 1 | AUCtime,step_time | 0.046069 | 1.133051 |
| KMNIST | 2 | AUCtime,step_time | 0.294041 | 1.094224 |

判断：

```text
1. S5 post-event recovery window 是本轮最强 FMS candidate，
   但 strict = 0/9。
2. S5 endpoint 6/9 被 generic controls 追平或超过。
3. S1/S2 几乎等价，说明 affected-only 仍接近 all-role。
4. S3/S4 没有打开机制；S4 RMS recompute 明显退化。
```

## 8. Line A：Affected-coordinate autopsy

| run | affected median | reset median | conclusion |
|---|---:|---:|---|
| v147 event-only K2 | 0.994661 | 0.994661 | A2-AffectedSetIntrinsicallyBroad |
| v147 delta_top25 K2 | 0.994765 | 0.248744 | A3-SparseAffectedMaskDestroysValue |
| v148 S1 current | 0.994661 | 0.994661 | A2-AffectedSetIntrinsicallyBroad |

判断：

```text
1. affected coordinate set 仍几乎覆盖全参数。
2. 这支持 A2：当前 FMS 影响是全局的，affected-only reset 不是真稀疏 reset。
3. v14.7 的 delta_top25 已证明 sparse reset fraction 降到约 25% 后 endpoint 退化，
   因此也支持 A3。
4. 当前没有 A1 implementation bug 证据；不能继续扫 topK / threshold。
```

## 9. Line E：Overhead audit

本轮补了粗粒度计时埋点；v148_overhead_breakdown.csv 中：

```text
instrumentation_available = 1
linec_audit_time_excluded = 1
```

代表性 timing：

| method | dataset | seed | step ratio | per-example grad | FMS update | projection | state transport | optimizer |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G4 random reset | MNIST | 1 | 0.914992 | 0.000000 | 0.000000 | 0.000000 | 0.000285 | 0.057986 |
| S1 event-only | MNIST | 1 | 1.262043 | 0.118536 | 0.001341 | 0.141367 | 0.001782 | 0.053314 |
| S5 post-event | MNIST | 1 | 1.295544 | 0.129745 | 0.001604 | 0.142942 | 0.006462 | 0.054430 |
| S5 post-event | KMNIST | 0 | 1.349946 | 0.140501 | 0.001332 | 0.142607 | 0.005682 | 0.053358 |

判断：

```text
1. S5 median step_time_ratio = 1.295544 > 1.25。
2. 主要额外时间来自 per-example gradient 和 projection；
   state transport 本身约 0.005-0.006 sec，不是最大项。
3. 由于 route 已经是 R3 generic confound，不进入 overhead optimization。
4. 不降低 step_time gate。
```

## 10. Line M：MLP analog controls

| method | strict | endpoint-vs-AdamW | efficiency | mean source | median AUC vs AdamW | median step |
|---|---:|---:|---:|---:|---:|---:|
| M0 MLP AdamW | 0 | 0 | 9 | -0.096389 | 1.000000 | 1.000000 |
| M1 MLP FMS no transport | 0 | 4 | 0 | -0.068154 | 0.982640 | 1.260415 |
| M2 MLP zero affected | 0 | 3 | 0 | -0.056845 | 0.989475 | 1.260104 |
| M3 MLP random reset | 0 | 3 | 0 | -0.030566 | 1.003156 | 1.266902 |
| M4 MLP full reset | 0 | 3 | 0 | -0.056845 | 0.989475 | 1.258281 |
| M5 MLP beta1 zero | 0 | 3 | 9 | -0.056643 | 0.999714 | 0.998726 |
| M6 MLP periodic reset | 0 | 4 | 9 | -0.075500 | 0.987696 | 0.993246 |

判断：

```text
MLP analog 没有 strict success，也没有超过 S5 endpoint；
本轮 route 的主要阻断来自 RAT generic optimizer reset controls，
不是 MLP analog 完整复现。
```

## 11. Required artifacts

输出目录：

```text
results/v14_8_optimizer_state_confound_audit_fms_specificity_all_basis_parallel/official_v148/
```

artifact completeness：

```text
required_count = 25
required_artifact_missing_count = 0
```

主要产物：

```text
v148_route_decision.json
v148_progress_table.csv
v148_v1461_v147_semantic_diff.csv
v148_generic_optimizer_reset_controls.csv
v148_fms_specificity_results.csv
v148_affected_mask_autopsy.csv
v148_overhead_breakdown.csv
v148_mlp_analog_controls.csv
v148_all_basis_substrate_status.csv
v148_linec_tail_audit.csv
v148_no_action_search_audit.csv
v148_forbidden_information_audit.csv
v148_required_artifact_manifest.csv
v148_code_review_packet.zip
v148_no_go_boundary.md
v148_next_hypothesis_queue.md
fig_*.svg
```

## 12. 最终科学结论

v14.8 没有达成 S5-OfficialFunctionalSuccess。

最终合法 route：

```text
route = R3-GenericOptimizerStateResetConfound
official_s5_reached = 0
promotion_allowed = 0
```

闭合事实：

```text
1. v14.7 every-step endpoint 9/9 被 controls 复现；
   v14.8 继续支持 generic optimizer-state confound。

2. v14.8 S5 post-event recovery window 是当前最强 FMS candidate，
   但 strict = 0/9，endpoint = 6/9。

3. Generic optimizer-only controls 能达到相当或更高 endpoint：
   G4 event-matched random reset = 7/9；
   G3/G5/G7 = 6/9。

4. affected mask 仍几乎全局：
   v148 S1 affected median = 0.994661。

5. v14.7 sparse top25 已显示 reset fraction 降到约 25% 后退化；
   当前没有合法依据继续 mask grid。

6. S5 step_time 仍超过 strict gate：
   median step_time_ratio = 1.295544 > 1.25。

7. no-action-search audit 与 forbidden information audit 均为 0 violation。
```

No-go boundary：

```text
1. 不能把 v14.6.1 diagnostic 或 v14.7/v14.8 endpoint 写成 S5。
2. 不能把 generic/random/full reset controls 能解释的 gain 写成 FMS-specific。
3. 不能继续新增 action token 或启动 controller。
4. 不能调 strength/lambda/lr/refresh grid。
5. 不能用 audit metric 反推 affected coordinate 或 trigger。
6. 不能降低 step_time gate。
```

下一步：

```text
如果继续，必须新预注册：
1. 真正 train-stream-only 且机制清晰的 sparse affected-mask 定义；
2. 完整 timing instrumentation / overhead 降低机制；
3. 或者从 optimizer-state reset route 撤出，回到 FMS definition / substrate repair。
```

## 13. 用户再次追问后的 stop-go 复核

本次没有新增训练；只复核 v14.8 完整计划的 route / no-go 条款，确认是否还有计划内必须继续的合法 repair。

计划原文约束：

```text
R3-GenericOptimizerStateResetConfound:
  optimizer-only / random / full reset controls 解释了 endpoint 或 strict gain。

Codex 禁止：
  新增 K-RT / K-AUC / K-FL action token；
  启动 controller；
  调 strength / lambda / lr / refresh grid；
  用 audit metric 生成 direction；
  把 endpoint-only 写成 strict；
  把 random/full reset controls 也能复现的结果写成 FMS-specific。

最终判断：
  如果是 generic optimizer-state reset / momentum suppression confound，
  functional update 主线必须从 optimizer-state reset route 撤出，
  回到 FMS definition / substrate / all-basis parallel repair，
  而不是继续用 action 或 reset 小修拖延。
```

当前 artifact 事实：

```text
route = R3-GenericOptimizerStateResetConfound
best_fms_method = S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow
best_fms_strict_pass_count = 0 / 9
best_fms_endpoint_vs_adamw_pass_count = 6 / 9
G4-RAT-AdamW-EventMatchedRandomReset endpoint = 7 / 9
G3/G5/G7 endpoint = 6 / 9
generic_optimizer_state_confound = 1
official_s5_reached = 0
promotion_allowed = 0
```

复核结论：

```text
1. v14.8 的 diagnostic 目标已经达成：
   它回答了核心问题，当前 endpoint gain 被 generic optimizer-state reset confound 解释。

2. v14.8 没有达成 S5-OfficialFunctionalSuccess。

3. 计划内不允许继续 reset/action 小修：
   不能新增 variant、不能调 grid、不能启动 controller、
   不能用 audit metric 反推坐标或 trigger。

4. 因此本轮保持 no-go：
   official_s5_reached = 0
   promotion_allowed = 0

5. 后续若继续，必须另开新预注册计划，
   从 FMS definition / substrate / all-basis repair 方向推进，
   不能在 v14.8 内把 R3 改写成 success。
```

## 14. 用户再次追问后的最终复核

本次仍没有新增训练；再次复核 v14.8 完整计划是否要求继续。

当前 artifact 事实不变：

```text
route = R3-GenericOptimizerStateResetConfound
best_fms_method = S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow
best_fms_strict_pass_count = 0 / 9
best_fms_endpoint_vs_adamw_pass_count = 6 / 9
G4-RAT-AdamW-EventMatchedRandomReset endpoint = 7 / 9
generic_optimizer_state_confound = 1
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

复核判断：

```text
1. v14.8 没有达成 S5。
2. v14.8 的机制判别目标已经完成：
   endpoint gain 被 generic optimizer-state reset / momentum suppression confound 解释。
3. 计划第 17 节明确要求：
   若是 generic confound，主线必须从 optimizer-state reset route 撤出，
   回到 FMS definition / substrate / all-basis parallel repair。
4. Line D 的本轮 required artifact 是 all-basis substrate status；
   v148_all_basis_substrate_status.csv 已输出。
5. 继续在 v14.8 内新增 reset variant、调 grid、启动 controller、
   或用 audit metric 反推坐标，都会违反计划。
```

最终边界：

```text
本轮保持 no-go。
不新增训练。
不补填成功。
不 promotion。

下一步必须是新预注册计划，而不是在 v14.8 内继续小修。
```
