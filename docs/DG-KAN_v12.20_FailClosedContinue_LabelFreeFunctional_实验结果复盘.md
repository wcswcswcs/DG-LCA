# DG-KAN v12.20 Fail-Closed Continue 实验结果复盘

> 结论先行：v12.20 按计划完成了 fail-closed continuation。promotion 仍全部关闭，但不再是 v12.19 那种 gate fail 后直接停止；本轮完成了 Line A upper-bound、Line C target reset、Line T visibility/no-go、Line I exploratory capacity、Line B P3/P4 not-opened audit、Line D monitor 和审计 zip。
> 最终 route：`R2-LabelFreeSignalFrameMissing`。
> minimum success：`Success D`，即完成 continuation 后得到 no-go 证据，而不是 functional success。

---

## 1. 最终 route

最终文件：

```text
results/v12_20_failclosed_continue_label_free_functional/official_continuation/v1220_route_decision.json
```

关键字段：

```text
route = R2-LabelFreeSignalFrameMissing
minimum_success = Success D
fail_reason = label-free signal frame attempts and legal visibility/C-T1 diagnostics did not pass; continuation diagnostics completed
line_r_pass = 1
linea_rows = 270
label_free_candidate_rows = 13
label_free_official_pass_count = 0
label_free_best_candidate = A1-noYForStats
upper_bound_rows = 6
linec_target_reset_rows = 216
c_t1_spearman_noise = 0.03645397607489652
c_t1_spearman_reservoir = 0.1970492654002053
c_t1_value_source_pass = 0
T1A_visibility_pass = 0
T1A_auc_joint_min = 0.0
T1A_precision_joint_min = 0.0
T1A_recall_joint_min = 0.0
T2_upper_bound_auc_joint_min = 0.03773584905660377
actuator_capacity_rows = 198
actuator_safe_movement_rows = 0
actuator_release_audit_rows = 0
p4_open = 0
continuation_missing_count = 0
required_artifact_missing_count = 0
```

解释：

```text
Promotion gate 没有放水：
  label-free base 没过；
  C-T1 target 没过；
  T1A hard visibility 没过；
  exploratory actuator 没有 safe movement / audit release；
  P3/P4 未打开。

Continuation queue 已执行：
  Line A / C / T / I / B / D 全部有 continuation artifact。
```

---

## 2. Implementation Readback / Code Rationale

### 2.1 Line R / continuation queue

新增 runner：

```text
experiments/run_v1220_failclosed_continue_label_free_functional.py
```

职责：

```text
1. 汇总 Line A official continuation 结果；
2. 写 v1220_code_review_manifest.csv；
3. 写 continuation_execution_manifest；
4. 执行 C-T1 target reset correlation；
5. 执行 T1A/T1B/T2 visibility v4 与 no-go；
6. 执行 Line I exploratory actuator capacity；
7. 写 P3/P4 not-opened audit；
8. 写 classic monitor；
9. 生成 figures/hash/zip。
```

审计结果：

```text
line_r_rows = 11
line_r_missing_refs = 0
line_r_pass = 1
```

### 2.2 Label-free architecture changes

文件：

```text
dgkan/models/fc_purekan_primitives.py
```

新增 label-free projector tokens：

```text
multiframebankp:
  PCA, SRHT/Rademacher, augmentation-stable, local-block, low-frequency frames.

selfcondresp:
  activation covariance eigenspace + random-cotangent residual frame.

covadaptp:
  covariance/PCA initialized projector, paired with unlabeled epoch adaptation in runner.
```

这些路径不读取：

```text
label
CE vector
validation/test
dataset name
future outcome
```

### 2.3 Label-free runner changes

文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增 candidates：

```text
A23-MultiFrameBank-labelFree
A24-MultiFrameBankDirect125-labelFree
A25-SelfConditionResidualP-labelFree
A26-CovAdaptP-labelFree
A27-MultiFrameLowFreqBias-labelFree
A28-unsupervisedClusterTrainProbe-diagnostic
A29-oracleSmallLabelTrainProbe-diagnostic
```

新增 upper-bound diagnostic y_stats：

```text
unsupervised_kmeans:
  A28；用 x_train 无监督聚类产生伪类；diagnostic only。

small_label_fraction:
  A29；少量真实 label oracle；upper-bound only。
```

新增 repair：

```text
apply_unlabeled_projector_adaptation()
```

用于 A26，在 epoch 开始时用 unlabeled activation covariance 更新 `quad_proj`。不读 label/CE。

### 2.4 数值 blocker 修复

文件：

```text
experiments/run_v1252_efficiency_functional_manifold.py
```

修复：

```text
_ridge_coupling:
  solve -> lstsq -> pinv fallback
```

原因：

```text
small diagnostic batch 下 x.T @ x + ridge I 仍可能 singular；
原实现直接 RuntimeError；
fallback 仍求解同一个 ridge system，不跳行、不造数据。
```

文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

修复：

```text
signal_reservoir_metrics_detailed:
  Gram matrix 对称化；
  nonfinite grad sketch 返回 NaN metrics；
  eigh fallback: jitter sweep -> CPU eigh -> NaN metrics。
```

这些修复使 blocker 显式失败或可计算，不把失败写成通过。

### 2.5 Visibility target join 修复

首次 official runner 中 soft-release regression 为空。核验发现：

```text
T1 feature row_id unique = 1728
T3 target row_id unique = 1728
完整 row_id intersect = 1
```

原因：

```text
T1 feature row_id 与 T3 target row_id 最后一段 index 不一致。
```

修复：

```text
write_visibility_v4 按稳定键 join：
source_run::dataset::seed::window::method::sketch_id
```

修复后 soft-release regression 产生实际数值：

```text
T1A soft_target_r2_noise_min = -11956.931957256009
T1A soft_target_r2_reservoir_min = -401245.5453505141
T1A+T1B soft_target_r2_noise_min = -11956.931957256009
T1A+T1B soft_target_r2_reservoir_min = -401245.5453505141
T2 upper-bound soft_target_r2_noise_min = -13474.094195762538
T2 upper-bound soft_target_r2_reservoir_min = -261669.11702240884
```

这些强负 R2 表明 soft target 也不可见，不是 pass。

---

## 3. Line A：Label-Free Base Architecture Redesign

官方 continuation 运行：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
epochs = 3 and 8
train_size = 1024
val_size = 512
test_size = 512
batch_size = 128
LineC batch = 64
LineC sketch_batch = 32
LineC sketch_dim = 24
```

总行数：

```text
linea_rows = 270
```

整体候选摘要：

| candidate | family | mean_delta_vs_A0 | worst_delta_vs_A0 | AUC_time_ratio_vs_mlp | LineC all pass | official |
|---|---|---:|---:|---:|---:|---:|
| A1-noYForStats | carried-forward control | -0.0030381944444444445 | -0.03125 | 1.083605297058177 | 0 | 0 |
| A5-orthogonalP-labelFree | carried-forward control | -0.012152777777777778 | -0.056640625 | 0.9963883569477223 | 0 | 0 |
| A23-MultiFrameBank-labelFree | A-F1 | -0.7388237847222222 | -0.828125 | 0.0 | 0 | 0 |
| A24-MultiFrameBankDirect125-labelFree | A-F1 | -0.7388237847222222 | -0.828125 | 0.0 | 0 | 0 |
| A25-SelfConditionResidualP-labelFree | A-F2 | -0.029513888888888888 | -0.052734375 | 1.1845646746035297 | 0 | 0 |
| A26-CovAdaptP-labelFree | A-F3 | -0.030164930555555556 | -0.08984375 | 1.0336809114813008 | 0 | 0 |
| A27-MultiFrameLowFreqBias-labelFree | A-F1 | -0.7388237847222222 | -0.828125 | 0.0 | 0 | 0 |
| A28-unsupervisedClusterTrainProbe-diagnostic | A-UB4 | -0.038736979166666664 | -0.1015625 | 1.5649744184776775 | 0 | 0 |
| A29-oracleSmallLabelTrainProbe-diagnostic | A-UB5 | -0.004123263888888889 | -0.0234375 | 0.9535006537974816 | 0 | 0 |

Line A 判断：

```text
label_free_official_pass_count = 0
best_label_free_candidate = A1-noYForStats
```

A1 mean delta 接近 gate，但：

```text
worst_delta_vs_A0 = -0.03125 < -0.015
AUC_time_ratio_vs_mlp = 1.083605297058177 > 1.0
LineC_nontearing_all_pass = 0
```

所以 A1 不能 promotion。

A29 small-label oracle 接近 task mean gate：

```text
mean_delta_vs_A0 = -0.004123263888888889
```

但：

```text
uses_label = 1
promotion_allowed = 0
LineC all pass = 0
worst_delta_vs_A0 = -0.0234375
```

它支持“少量 supervised signal 仍有上界价值”，但不能作为 label-free claim。

新增 A23/A24/A27 multi-frame bank 明显失败，说明简单 frame bank 不能替代 supervised trainprobe signal。

---

## 4. Line C：Target Reset Diagnostic

产物：

```text
v1220_linec_target_reset.csv
```

行数：

```text
linec_target_reset_rows = 216
```

C-T1 loss-agnostic geometric target 与 audit target 的 Spearman：

```text
c_t1_spearman_noise = 0.03645397607489652
c_t1_spearman_reservoir = 0.1970492654002053
c_t1_value_source_pass = 0
```

计划 gate 要求：

```text
Spearman(G_LA, Delta NoiseSignalLeak_audit) <= -0.30
Spearman(G_LA, Delta RealSignalReservoirRatio_audit) <= -0.30
```

实际结果为正相关且远未达到负相关 gate。解释：

```text
当前 C-T1 几何代理不能作为 functional value source；
CE-residual audit target 仍只能作为 audit / non-harm，不应作为方向源。
```

---

## 5. Line T：Visibility v4 / No-Go

产物：

```text
v1220_visibility_scores.csv
v1220_visibility_no_go_upper_bound.csv
v1220_visibility_leaveout.csv
```

T1A only：

```text
rows = 1728
feature_columns = 31
long_feature_rows = 53568
AUC_joint_min = 0.0
precision_at_k_joint_min = 0.0
recall_at_k_joint_min = 0.0
soft_target_r2_noise_min = -11956.931957256009
soft_target_r2_reservoir_min = -401245.5453505141
visibility_pass = 0
```

T1B only：

```text
rows = 0
visibility_pass = 0
not_promotable_reason = actual optimizer-update vector features were not present in v12.16-v12.19 visibility artifacts; scored as unavailable, not promotable
```

解释：本轮没有伪造 T1B optimizer-update features；它被明确记录为 unavailable。未来若要真正推进 T1B，需要在训练 runner 中保存 opaque optimizer update features。

T1A + T1B：

```text
rows = 1728
feature_columns = 31
AUC_joint_min = 0.0
precision_at_k_joint_min = 0.0
recall_at_k_joint_min = 0.0
soft_target_r2_noise_min = -11956.931957256009
soft_target_r2_reservoir_min = -401245.5453505141
visibility_pass = 0
not_promotable_reason = T1B unavailable in inherited artifacts; this equals T1A and is not a T1B promotion result
```

T2 upper-bound diagnostic：

```text
rows = 1728
feature_columns = 41
long_feature_rows = 70848
AUC_joint_min = 0.03773584905660377
precision_at_k_joint_min = 0.0
recall_at_k_joint_min = 0.0
soft_target_r2_noise_min = -13474.094195762538
soft_target_r2_reservoir_min = -261669.11702240884
diagnostic_only = 1
```

Line T 判断：

```text
硬 release 不可见；
soft release 也不可见；
T2 clone-probe upper-bound 也没有救回；
support_concentrated = 0
control_false_positive_rate = 0.0
```

所以失败不是 support 集中或 control false positive，而是当前 legal T1 feature family 和 inherited T2 diagnostic 对 target 都没有可用可见性。

---

## 6. Line I：Exploratory Actuator Capacity

产物：

```text
v1220_actuator_capacity.csv
```

来源：

```text
v1217_actuator_response_dictionary.csv
```

结果：

```text
actuator_capacity_rows = 198
actuator_safe_movement_rows = 0
actuator_release_audit_rows = 0
promotion_allowed = 0 for all rows
```

示例：

```text
A1-DirectRole:
  sketch_delta_fro = 0.0010618959786370397
  projector_angle_deg = 0.08156468719244003
  logit_max_abs_drift = 0.007739067077636719
  NoiseSignalLeak_delta_audit = 0.00010123848915100098
  RealSignalReservoirRatio_delta_audit = 2.1517276763916016e-05
  safe_movement_pass = 0
  release_audit_pass = 0

A4-ProjectionP:
  sketch_delta_fro = 0.0046552433632314205
  projector_angle_deg = 0.1453699767589569
  logit_max_abs_drift = 0.04452776908874512
  NoiseSignalLeak_delta_audit = -0.0001470521092414856
  RealSignalReservoirRatio_delta_audit = -0.001662135124206543
  safe_movement_pass = 0
  release_audit_pass = 0
```

Line I 判断：

```text
exploratory capacity 已记录；
没有 actuator 同时满足 movement / projector angle / audit release；
不能打开 official actuator gate。
```

---

## 7. Line B：Functional P3/P4

产物：

```text
v1220_functional_p3.csv
v1220_functional_p4_short.csv
```

P3 candidate families 均记录为 not opened：

```text
B1-C-T1-geometric-stabilizer
B2-AdamW-orthogonal-projector-stabilizer
B3-signal-frame-occupancy-rebalance
B4-primitive-role-energy-transport
B5-low-frequency-event-geometry-maintenance
B6-T1B-update-spectrum-preconditioner
```

共同原因：

```text
P3 not opened: T1/C-T1/I official gates did not pass; exploratory actuator capacity recorded separately
```

P4：

```text
p4_open = 0
P4_pass = 0
failure_reason = P4 not opened after completed continuation: official T1/C-T1/I gates did not pass
```

这不是 fail-fast；因为 Line A/C/T/I/B continuation artifacts 已全部生成。

---

## 8. Line D：Classic No-BSpline Monitor

产物：

```text
v1220_classic_family_status.csv
v1220_classic_family_new_hypothesis.csv
v1220_classic_family_linec.csv
```

结果：

```text
classic_status_rows = 12
classic_new_hypothesis_count = 0
line_d_role = monitor_only
```

解释：

```text
没有新 classic loss-agnostic hypothesis；
按计划不重复旧 family candidate；
BSpline 继续 frozen。
```

---

## 9. Continuation Manifest

产物：

```text
v1220_continuation_execution_manifest.csv
```

结果：

```text
Line A continuation_executed = 1
Line C continuation_executed = 1
Line T continuation_executed = 1
Line I continuation_executed = 1
Line B continuation_executed = 1
Line D continuation_executed = 1
continuation_missing_count = 0
```

Stop audit：

```text
route = R2-LabelFreeSignalFrameMissing
minimum_success = Success D
promotion_allowed = 0
final_stop_allowed = 1
```

所以本轮可以停止，不是因为某个 gate fail 后直接停，而是因为所有预注册 continuation diagnostics 已完成。

---

## 10. 审计 zip

路径：

```text
results/v12_20_failclosed_continue_label_free_functional/official_continuation/v1220_code_review_packet.zip
```

sha256：

```text
e4dac025396e626032bd1fddbf639f81dec0e263c3ab4d8b4f6fa636841f68d4
```

自检：

```text
entries = 46
contains code/experiments/run_v1220_failclosed_continue_label_free_functional.py = True
contains code/experiments/run_v1218_b320_label_free_ablation.py = True
contains code/dgkan/models/fc_purekan_primitives.py = True
contains code/experiments/run_v1252_efficiency_functional_manifold.py = True
contains review_artifacts/v1220_route_decision.json = True
contains review_artifacts/v1220_label_free_base_candidates.csv = True
contains review_artifacts/v1220_visibility_scores.csv = True
contains review_artifacts/v1220_actuator_capacity.csv = True
```

---

## 11. 本轮结论

### 11.1 Success A 未达到

没有 label-free B320-like base 同时通过 task + Line C。

```text
label_free_official_pass_count = 0
```

A1 是 best，但 Line C 不过，AUC time 超 gate，worst delta 不过。

### 11.2 Success B 未达到

C-T1 target reset 没有找到 deployable value source。

```text
c_t1_spearman_noise = 0.03645397607489652
c_t1_spearman_reservoir = 0.1970492654002053
c_t1_value_source_pass = 0
```

方向不但没有达到 `<= -0.30`，还呈弱正相关。

### 11.3 Success C 未达到

T1A/T1B visibility 没有通过。

```text
T1A_auc_joint_min = 0.0
T1A_precision_joint_min = 0.0
T1A_recall_joint_min = 0.0
```

T2 upper-bound diagnostic 也没救回：

```text
T2_upper_bound_auc_joint_min = 0.03773584905660377
```

### 11.4 Success D 达到

本轮完成了强 no-go continuation：

```text
Line A upper-bound completed
Line C target reset completed
Line T hard/soft/no-go visibility completed
Line I exploratory actuator capacity completed
Line B P3/P4 not-opened audit completed
Line D monitor completed
continuation_missing_count = 0
```

因此 `Success D` 成立：

```text
hard CE-residual release is not visible from legal T1 even with upper-bound diagnostics;
target reset is justified.
```

更准确地说，当前 C-T1 也没有成为新 value source，所以不只是 old hard release 不可见，而是本轮尝试的 deployable geometric reset 也失败。

---

## 12. 下一步建议

不要继续在 B320 trainprobe patch 上小修。v12.20 的结果支持：

```text
1. supervised trainprobe signal 仍是 B320-current 的关键；
2. simple multi-frame / self-conditioning / covadapt projector 不能替代；
3. small-label oracle 有 task 上界迹象，但不是 label-free，Line C 也不过；
4. hard release 和 soft release 都不可见；
5. inherited actuator dictionary 没有足够 movement/release capacity。
```

下一轮优先级：

```text
1. 重新设计 label-free signal-frame architecture，而不是再追加 projector token。
2. 在训练 runner 中原生记录 T1B opaque optimizer-update features；不要事后伪造。
3. 重新定义 C-T1，使其来自可验证的 precommit geometric invariant，而不是当前 P-basis heuristic。
4. 设计更强的 primitive-level actuator basis；当前 actuator movement 太弱。
5. P4 继续关闭，直到 Line A 或 C-T1/T1/I 至少一条合法路径打开。
```
