# DG-KAN v12.16 B320Locked ExplicitSignalReservoirFunctional 实验结果复盘

生成时间：`2026-05-24T02:42:34Z`

对应计划：

```text
docs/DG-KAN_v12.16_B320Locked_ExplicitSignalReservoirFunctional_实验结果分析与下一步计划.md
```

对应执行审计：

```text
docs/DG-KAN_v12.16_B320Locked_ExplicitSignalReservoirFunctional_执行复盘.md
```

本复盘只记录真实落盘 artifact；不把 smoke 当 official，不把 P1/P2/P3 fail 后的 gated-not-run 写成 B16 success，不编造 P3/P4 数据。

## 1. 总结论

v12.16 official route：

```text
route = R2-EstimatorStillWeak
p0_pass = 1
p1_pass = 0
p2_pass = 0
p3_response_pass = 0
p4_open = 0
p3_survivor_count = 0
```

修复 run route：

```text
route = R2-EstimatorStillWeak
p0_pass = 1
p1_pass = 0
p2_pass = 0
p3_response_pass = 0
p4_open = 0
p3_survivor_count = 0
```

本轮最重要的真实结论：

```text
1. B320 anchor 仍通过，base 不是 blocker。
2. P1 的 Line C v2 sketch / target reconstruction 仍不能预测 noise/reservoir hard release。
3. P2 explicit target calibration 不成立，sign accuracy / Spearman / joint release rows 均未过 gate。
4. P3 actuator response matrix rank/condition 数值不差，但 safe actuator combo gate 0/9。
5. 因 P1/P2/P3 全部前置失败，B16 functional candidates 正确 gated not-run，P4 short-run 正确关闭。
```

因此 v12.16 没有 functional survivor，也没有 P4 资格。当前 blocker 仍是：

```text
loss-agnostic target visibility 不足；
当前 explicit sketch target 看不到可部署的 noise/reservoir release basis；
actuator movement 仍不能稳定变成 hard gate release。
```

## 2. P0 Anchor Monitor

P0 复用 v12.14/v12.15 locked B320 anchor artifact，不小修 base。

official：

```text
p0_pass = 1
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta_vs_mlp = 0.027669270833333332
worst_delta_vs_mlp = -0.001953125
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
LineC_nontearing_pass = 1
```

结论：

```text
B320 base 仍可作为 functional experiment anchor。
本轮失败不应通过 B320 architecture 小修解释。
```

## 3. P1 Line C v2 Sketch / Target Reconstruction

official 设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
windows = 3,5,10
functional_batch_size = 32
sketch_dim = 12
output_subspace_rank = 3
rows = 864
```

P1 family 结果：

| sketch_family | Noise Spearman | Reservoir Spearman | precision | recall | NoOp FP | Random FP |
|---|---:|---:|---:|---:|---:|---:|
| `multi_window_gradient` | `0.02414884548498602` | `-0.15995361757658177` | `0.0` | `0.0` | `0` | `0` |
| `role_conditioned` | `0.01283158228062369` | `-0.08782059412204585` | `0.0` | `0.0` | `0` | `0` |
| `stable_unstable_output` | `0.04916432196353432` | `-0.09742148273573316` | `0.0` | `0.0` | `0` | `0` |
| `persistent_projector` | `0.014865046065642` | `-0.10175821389746577` | `0.0` | `0.0` | `0` | `0` |

official P1 route fields：

```text
p1_best_noise_abs_spearman = 0.04916432196353432
p1_best_reservoir_abs_spearman = 0.15995361757658177
p1_best_negative_release_precision = 0.0
p1_best_negative_release_recall = 0.0
p1_pass = 0
p1_strong_pass = 0
```

关键解释：

```text
P1 不是没有 hard-release 样本。
official v1216_linec_v2_sketch_targets.csv 中：
  Noise hard support = 80 / 864
  Reservoir hard support = 76 / 864

但 v12.16 loss-agnostic sketch target 没有预测到这些 release：
  negative_release_precision = 0
  negative_release_recall = 0
```

这说明失败不是单纯因为数据里没有 release 事件，而是当前 target / feature 看不到它们。

## 4. P1 修复 run

修复设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
windows = 3,5,10
functional_batch_size = 64
sketch_dim = 24
output_subspace_rank = 5
rows = 864
```

修复方向对应计划：

```text
增加 sketch_dim；
保留 window ensemble；
保留 role-conditioned sketch；
保留 stable/unstable output sketch；
改变 rank cutoff；
增大 batch_size 检查方差。
```

修复结果：

| sketch_family | Noise Spearman | Reservoir Spearman | precision | recall | NoOp FP | Random FP |
|---|---:|---:|---:|---:|---:|---:|
| `multi_window_gradient` | `0.03484404943854978` | `-0.15058756569705098` | `0.0` | `0.0` | `0` | `0` |
| `role_conditioned` | `0.034752997323167445` | `-0.17545880235411232` | `0.0` | `0.0` | `0` | `0` |
| `stable_unstable_output` | `-0.0006568276168539488` | `-0.15593755180311364` | `0.0` | `0.0` | `0` | `0` |
| `persistent_projector` | `0.07642684028921456` | `-0.06332247369742607` | `0.0` | `0.0` | `0` | `0` |

修复 route fields：

```text
p1_best_noise_abs_spearman = 0.07642684028921456
p1_best_reservoir_abs_spearman = 0.17545880235411232
p1_best_negative_release_precision = 0.0
p1_best_negative_release_recall = 0.0
p1_pass = 0
```

解释：

```text
sketch_dim / rank / batch 增大没有修复 P1。
Noise Spearman 仅从 0.049 提到 0.076；
Reservoir Spearman 仅从 0.160 提到 0.175；
precision / recall 仍为 0。
```

因此，当前 P1 blocker 不是简单的 sketch_dim 太小、rank cutoff 太窄或 batch_size 太小，而是 target construction 本身仍未显式捕捉 release basis。

## 5. P2 Explicit Target Calibration

official：

```text
p2_pass = 0
p2_strong_pass = 0
p2_noise_sign_accuracy = 0.5252525252525253
p2_reservoir_sign_accuracy = 0.3838383838383838
p2_noise_spearman = 0.06204081632653061
p2_reservoir_spearman = 0.06430426716141002
p2_near_joint_release_rows = 0
```

repair：

```text
p2_pass = 0
p2_strong_pass = 0
p2_noise_sign_accuracy = 0.5050505050505051
p2_reservoir_sign_accuracy = 0.41414141414141414
p2_noise_spearman = 0.009672232529375387
p2_reservoir_spearman = 0.16084106369820653
p2_near_joint_release_rows = 0
```

P2 gate 要求：

```text
noise sign_accuracy >= 0.65
reservoir sign_accuracy >= 0.65
noise Spearman >= 0.35
reservoir Spearman >= 0.35
至少 3/9 rows 同时：
  NoiseSignalLeak_delta < -0.005
  RealSignalReservoirRatio_delta < -0.005
  logit drift <= 0.05
```

结论：

```text
P2 explicit target calibration 未成立。
当前 T1/T2/T3/T4/T5 proxy target 不能稳定预测或制造 joint noise/reservoir release。
```

## 6. P3 Fused Primitive Actuator Response Matrix

official：

```text
rows = 99
p3_response_expected_count = 9
p3_response_pass_count = 0
p3_response_pass = 0
p3_max_response_rank = 5
p3_min_response_condition = 5.958653351168656
p3_max_actuator_sketch_delta_fro = 0.01025579497218132
p3_max_actuator_projector_angle_deg = 7.517968654632568
p3_best_noise_delta = -0.004974812269210815
p3_best_reservoir_delta = -0.0043097734451293945
```

official classification：

```text
ActuatorResponseWeak = 88
SafetyRejected = 9
ObjectiveMisaligned = 2
```

official extrema：

```text
max sketch:
  actuator_id = A10-MixedLowRankPrimitive
  dataset = KMNIST
  seed = 1
  sketch_delta_fro = 0.01025579497218132
  projector_angle = 0.6192882657051086 / 0.6177064180374146
  NoiseSignalLeak_delta = -0.002143383026123047
  Reservoir_delta = 0.0010202527046203613
  CouplingR2_delta = 0.3576848402009114
  classification = ObjectiveMisaligned

max angle:
  actuator_id = A4-ProjectionP
  dataset = KMNIST
  seed = 1
  sketch_delta_fro = 0.0045374492183327675
  projector_angle = 7.517915725708008 / 7.517968654632568
  NoiseSignalLeak_delta = -0.00005099177360534668
  Reservoir_delta = -0.0003514289855957031
  CouplingR2_delta = 0.25673007215202404
  classification = ObjectiveMisaligned
```

repair：

```text
rows = 99
p3_response_expected_count = 9
p3_response_pass_count = 0
p3_response_pass = 0
p3_max_response_rank = 5
p3_min_response_condition = 8.123402310333205
p3_max_actuator_sketch_delta_fro = 0.00858838576823473
p3_max_actuator_projector_angle_deg = 2.553436517715454
p3_best_noise_delta = -0.003841221332550049
p3_best_reservoir_delta = -0.006813347339630127
```

repair classification：

```text
ActuatorResponseWeak = 92
SafetyRejected = 4
ObjectiveMisaligned = 3
```

P3 解释：

```text
response matrix rank / condition 本身不是主要 blocker；
official rank = 5, condition 最小约 5.96。

但 safe actuator combo gate 仍是 0/9：
  没有 6/9 dataset-seed 同时满足 sketch_delta_fro >= 0.01、
  projector_angle >= 1 degree、logit_drift <= 0.05。

同时 actuator response matrix 中 hard release support 为：
  official noise hard support = 0 / 99
  official reservoir hard support = 0 / 99
  repair noise hard support = 0 / 99
  repair reservoir hard support = 0 / 99
```

因此 P3 也不能进入 B16 candidate construction。

## 7. P4 / P4 Short-Run

official 和 repair 均为 gated not-run：

```text
candidate_id = B16-NOT-RUN
candidate_family = gated_not_run
strong_promotion = 0
weak_promotion = 0
fail_reason = P1_estimator_gate_failed;P2_explicit_target_calibration_failed;P3_actuator_response_gate_failed
```

P4 short-run：

```text
method = P4_NOT_OPENED
reason = no_v1216_functional_P3_survivor
```

解释：

```text
没有 P3 survivor，按 v12.16 gate 不能跑 P4。
如果此时跑 P4，会把未过前置因果验证的 movement 写成在线收益，是错误路线。
```

## 8. Line D

v12.16 runner 输出了 `v1216_classic_family_status.csv`，但没有重新跑 Line D family targeted repair。

原因：

```text
1. v12.15 Line D focused repair 已补跑，0 FamilyPass，且 coupling_collapse。
2. v12.16 Wave 1 functional gates P1/P2/P3 已失败。
3. 当前 runner 没有实现文档中新的 family-specific candidate IDs。
4. 不允许把 v12.15 的 focused repair 重新包装进 B16/P3。
```

这不是 functional result 的数据缺口；它是 route gate 下的低优先级 carry-forward。若后续单独开 Line D，需要先实现新的 family hypothesis，而不是重复 B7lu/B7lv、B2aa/B2ab 等旧 focused candidates。

## 9. Loss-Agnostic / Provenance Audit

本轮 official audit artifact：

```text
v1216_loss_agnostic_audit.csv
v1216_provenance_audit.csv
```

审计结论：

```text
official direction rows 均记录：
loss_agnostic_direction = 1
ce_vector_used_for_direction = 0
label_used_for_direction = 0
validation_used_for_commit = 0
dataset_name_used_for_commit = 0
no_fake = 1
no_proxy = 1
cpu_offload_used = 0
```

CEp99、Brier、holdout loss 只作为 audit 字段；没有作为 functional direction objective。

## 10. 本轮到底推进了什么

相比 v12.15，v12.16 的真实推进是：

```text
1. 把 v12.15 的“explicit target 应该可见”做成了可执行 P1/P2/P3 artifact contract。
2. 实测了 multi-window、role-conditioned、stable/unstable output、persistent projector 四类 P1 sketch target。
3. 实测了 A1..A11 fused primitive actuator response matrix。
4. 验证了增大 sketch_dim/rank/batch 仍不能修复 P1。
5. 确认当前 blocker 更靠前：不是 B16 solver 不够强，而是 P1/P2 target visibility 本身失败。
```

v12.16 的负结果很明确：

```text
当前 explicit signal/reservoir functional mechanism 仍未建立。
P1 target reconstruction 失败；
P2 calibration 失败；
P3 response gate 失败；
B16/P4 不允许打开。
```

## 11. 下一步

当前 route：

```text
R2-EstimatorStillWeak
```

下一步不应继续：

```text
重复 B16 candidate grid；
重复 K6 sign / budget；
重复 + / - selector；
继续增大 CouplingR2 权重；
放宽 -0.01 hard gate；
用 CE vector / label residual 构造 target；
因为单行 sketch 或 angle 高就打开 P4。
```

下一步应做：

```text
重新设计 Line C sketch construction 本身；
让 noise/reservoir release basis 在 direction generation 阶段可见；
优先考虑 primitive-role gradient sketch 的 persistent basis，而不是 output-logit stable/unstable proxy；
在 P1 precision/recall 不再为 0 前，不进入 B16/P4。
```

更具体的推荐方向：

```text
1. 不再只用 output covariance stable/unstable basis；
2. 构造 primitive-role gradient sketch 的 persistent aligned basis；
3. 对 noise/reservoir release 做 batch-to-batch matched finite-difference target；
4. 将 actual release support 与 predicted release support 做 calibration，而不是只做 continuous score correlation；
5. 先把 P1 negative_release_precision / recall 从 0 拉起来，再考虑 P2/P3/P4。
```
