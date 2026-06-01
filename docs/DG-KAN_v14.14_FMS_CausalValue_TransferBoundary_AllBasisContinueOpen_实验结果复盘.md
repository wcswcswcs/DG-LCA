# DG-KAN v14.14 FMS CausalValue TransferBoundary AllBasisContinueOpen 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入本轮实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 weak proxy / boundary real-lite / substrate replay 写成 promotion。

## 1. 计划理解

v14.14 的目标是验证 v14.13 E3/V3 observability 是否具有独立 causal value，并在 ControlEquivalent 成立时只执行预注册 B1/B2/B3 boundary FMS。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py
```

修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 boundary_abstention_gating。
  该 gate 只使用当前 train-stream generic/projected update 的 cosine、value retention、
  degree projection rejection 计算 trust，把 B1 变成 lower-plasticity boundary。
  不读取 validation/test/future/query，也不使用 LineC/CEp99/NLL/ECE/AUCtime 生成方向。
```

实现：

```text
1. Line R provenance / forbidden / no-action-search audit。
2. Line E4 observability robustness and matched-control deconfounding。
3. Line Q FMS causal value vs matched controls。
4. Line B/F bounded D-CHE B1/B2/B3 boundary real-lite。
5. Line D all-basis cross-version reconciliation replay。
6. Line C LineC/tail/failure taxonomy。
7. required manifest、figures、code review packet、执行日志和复盘日志。
```

## 3. Line E4 结果

```text
best_feature = E4-K-recovery_lag
best_auc_mean = 0.9090909090909091
best_leaveout_auc_min = 0.5
best_spearman = 0.034221800155956356
best_incremental_auc_over_controls = 0.40909090909090906
exploration_gate_pass = 0
promotion_enabling_gate_pass = 0
observability_explained_by_controls = 0
```

## 4. Line Q 结果

```text
source_vs_best_control_mean = -0.032626233498255414
control_equivalent_fraction = 0.8888888888888888
bad_event_fraction = 0.9555555555555556
causal_exploration_gate_pass = 0
strong_causal_gate_pass = 0
q_route_hint = R3-FMSDirectionControlEquivalent
v3_dominant_breakpoint = V3-B6-ControlEquivalent
```

判断：

```text
Line Q 是 matched-control causal-value audit，不允许 promotion。
若 control-equivalent 仍占主导，只允许进入 Line B，不允许扩 FMS-M6/M7。
```

## 5. Line B/F boundary real-lite 结果

```text
boundary_reused_existing = 0
boundary_real_lite_pass_count = 0 / 9
boundary_mean_source_vs_best_control = -0.032401322214691726
boundary_control_equivalent_fraction = 0.9259259259259259
boundary_control_equivalent_decrease_fraction = 0.0
boundary_exploration_gate_pass = 0
boundary_meaningful_gate_pass = 0
boundary_s4_gate_pass = 0
source_fail_count = 25
auctime_fail_count = 26
tail_fail_count = 24
linec_fail_count = 10
```

B/F method surface：

```text
F-B1-AbstentionBoundaryFMS -> F-CHE-RT1-TrainSplitAgreement + boundary_abstention_gating
F-B2-ConstraintOnlyBoundaryFMS -> F-CHE-FB2-DegreeConstraintOnly
F-B3-ContinuousLowAmplitudeBoundaryFMS -> F-CHE6-PhaseScheduleDegreeFMS with one pre-registered low amplitude
```

## 6. Line D all-basis 结果

```text
best_non_dche_family = D-FOU
best_non_dche_dataset_seed_pass_count = 2 / 9
line_d_official_fms_eligible_family_count = 0
```

### 6.1 Boundary method summary

| method | rows | strict pass | dataset-seed pass | mean source vs best control | mean NLL |
|---|---:|---:|---:|---:|---:|
| F-B1-AbstentionBoundaryFMS | 9 | 0 | 0 | -0.046176854107115 | 0.6764623125394186 |
| F-B2-ConstraintOnlyBoundaryFMS | 9 | 0 | 0 | -0.023050483730104234 | 0.6533359421624078 |
| F-B3-ContinuousLowAmplitudeBoundaryFMS | 9 | 0 | 0 | -0.027976628806855943 | 0.6582620872391595 |

判断：

```text
B1/B2/B3 均没有打开 real-lite row；
best boundary method 仍低于 best matched control。
因此 boundary reset 没有达到 >=3/9 weak exploration gate。
```

### 6.2 Boundary control summary

| control | rows | mean source vs AdamW | mean NLL |
|---|---:|---:|---:|
| C0-D-CHE-AdamW | 9 | 0.0 | 0.6665280792448256 |
| C1-D-CHE-AdamW-NoOpMatchedOverhead | 9 | 0.0047211481465233695 | 0.6618069310983022 |
| C2-D-CHE-AdamW-RandomMatchedNorm | 9 | -0.0003910263379414876 | 0.666919105582767 |
| C3-D-CHE-AdamW-AdamWParallelDirectionControl | 9 | 0.015354070398542616 | 0.651174008846283 |
| C4-D-CHE-AdamW-GenericOptimizerStateControl | 9 | 0.022337244616614446 | 0.6441908346282111 |
| C5-D-CHE-AdamW-SameActiveFractionControl | 9 | 0.0080412228902181 | 0.6584868563546075 |

判断：

```text
GenericOptimizerStateControl / AdamWParallelDirectionControl 仍强于 B1/B2/B3。
这支持 Q 的 ControlEquivalent / control-explained blocker。
```

### 6.3 Line D reconciliation status

```text
D-FOU: v14.9 = 6/9, v14.13 replay/extra best = 2/9, status = CandidateMismatch
D-RBF: v14.9 = 6/9, v14.13 replay/extra best = 1/9, status = CandidateMismatch
D-WAV: v14.9 = 2/9, v14.13 replay/extra best = 0/9, status = CandidateMismatch
```

判断：

```text
非 D-CHE basis 没有达到 >=6/9 substrate exploration gate；
D-FOU/D-RBF/D-WAV 仍不得进入 official FMS proof。
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
1. v14.14 已执行 Line R/E4/Q/B/F/D/C/Z。
2. E4/Q 只提供 causal-value / control deconfound 诊断，不允许 promotion。
3. Boundary real-lite 未达到 9/9，因此没有 official S5。
4. D-CHE 外 basis 未达到 >=6/9 substrate exploration gate 时，不允许 official FMS proof。
5. 当前不能新增 F-CHE8/F-CHE9、FMS-M6/M7、action token、controller 或 reset route；
   也不能用 real fail pattern 或 audit metric 反推方向。
```

## 9. Blocker 与停止边界

已覆盖的 v14.14 分支：

```text
1. Line E4 observability deconfound：
   raw best AUC = 0.9090909090909091，
   但 leaveout min = 0.5，Spearman = 0.034221800155956356，
   promotion-enabling gate = 0。
2. Line Q matched controls：
   control_equivalent_fraction = 0.8888888888888888，
   bad_event_fraction = 0.9555555555555556，
   causal exploration gate = 0。
3. Line B/F boundary reset：
   F-B1/F-B2/F-B3 全部执行，
   real-lite pass = 0/9。
4. Line D：
   非 D-CHE best = 2/9，
   official_fms_eligible_family_count = 0。
```

当前合法结论：

```text
v14.14 没有达成 S4-lite / S4 / S5。
继续推进需要新增 FMS-M6/M7 或 F-CHE8/F-CHE9、
新增 action token / controller / reset route、
或使用 real fail pattern / audit metric 反推方向。
这些均违反 v14.14 当前计划。
因此本轮不再启动新的 v14.14 训练。
```

## 10. 用户再次追问后的状态复核

本次没有新增训练；重新读取 v14.14 完整计划中的 Line B/F/D、route definitions、Stop/continue contract，并复核最新 artifact，确认是否还有未执行且不越界的分支。

计划原文关键边界：

```text
1. Line B/F 只允许：
   F-B1-AbstentionBoundaryFMS
   F-B2-ConstraintOnlyBoundaryFMS
   F-B3-ContinuousLowAmplitudeBoundaryFMS
2. Line B 不允许：
   B4/B5/B6、dataset/seed 分支、audit metric direction、
   action bank / controller / reset route、strength/lambda/interval grid。
3. Line D non-D-CHE <6/9 不是 hard stop，
   但不允许进入 official FMS proof。
4. 如果 current method family 的 legal fallback 已覆盖，
   必须进入下一版机制重写，而不是在同一 runner 临时补 token。
```

最新 artifact 复核：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
e4_best_feature = E4-K-recovery_lag
e4_best_auc_mean = 0.9090909090909091
e4_best_leaveout_auc_min = 0.5
e4_exploration_gate_pass = 0
q_control_equivalent_fraction = 0.8888888888888888
q_bad_event_fraction = 0.9555555555555556
boundary_real_lite_pass_count = 0/9
boundary_control_equivalent_fraction = 0.9259259259259259
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 2/9
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
manifest_missing_sum = 0
```

覆盖性复核：

```text
1. B/F methods F-B1/F-B2/F-B3 已执行，共 27 rows。
2. B/F controls C0..C5 已执行，共 54 rows。
3. Q groups G0/G1/G2/G3/G4/G5/G6/G7/G8 已输出。
4. Line D D-FOU/D-RBF/D-WAV replay/hardening artifact 已合入，
   v1414_d_all_basis_substrate_hardening.csv rows = 135。
5. required manifest 缺失为 0；forbidden / no-action-search violation 为 0。
```

结论：

```text
v14.14 仍没有达成 S4-lite / S4 / S5。
当前计划内可合法执行的 fallback 已覆盖。
继续推进需要新增 FMS-M6/M7、F-CHE8/F-CHE9、B4/B5/B6、
action token / controller / reset route，
或使用 real fail pattern / audit metric 反推方向；这些都违反 v14.14 当前计划。
因此本次不启动新的 v14.14 训练。
当前合法结论仍是：
  route = R4-AllBasisSubstrateBlocked
  minimum_success = S3-DCHESyntheticFMSPass
  official_s5_reached = 0
  promotion_allowed = 0
```

## 11. 用户再次追问后的代码层覆盖复核

本次没有新增训练；只读取 v14.14 runner、完整计划与最新 official artifacts，确认是否存在“计划允许但漏跑”的 method / control / Line D candidate。

代码层 method surface 复核：

```text
BOUNDARY_ALIASES =
  F-B1-AbstentionBoundaryFMS -> F-CHE-RT1-TrainSplitAgreement
  F-B2-ConstraintOnlyBoundaryFMS -> F-CHE-FB2-DegreeConstraintOnly
  F-B3-ContinuousLowAmplitudeBoundaryFMS -> F-CHE6-PhaseScheduleDegreeFMS

Q_GROUPS =
  G0-D-CHE-AdamW-baseline
  G2-RandomMatchedNorm
  G3-SameActiveFractionControl
  G6-AdamWParallelDirectionControl
  G7-NoOpMatchedOverhead
  G8-GenericOptimizerStateControl

G4 / G5 在 E4/Q 中作为 matched controls 写入：
  G4-SameDegreeProjectionRejectionControl
  G5-SameValueRetentionRandomDirection
```

artifact 覆盖复核：

```text
v1414_method_surface_manifest.csv rows = 11
methods =
  C0-D-CHE-AdamW
  C1-D-CHE-AdamW-NoOpMatchedOverhead
  C2-D-CHE-AdamW-RandomMatchedNorm
  C3-D-CHE-AdamW-AdamWParallelDirectionControl
  C4-D-CHE-AdamW-GenericOptimizerStateControl
  C5-D-CHE-AdamW-SameActiveFractionControl
  F-B1-AbstentionBoundaryFMS
  F-B2-ConstraintOnlyBoundaryFMS
  F-B3-ContinuousLowAmplitudeBoundaryFMS
  G4-SameDegreeProjectionRejectionControl
  G5-SameValueRetentionRandomDirection

v1414_q_fms_vs_matched_controls.csv rows = 189
Q groups =
  G0-D-CHE-AdamW-baseline
  G1-D-CHE-FMS-M1..M5
  G2-RandomMatchedNorm
  G3-SameActiveFractionControl
  G4-SameDegreeProjectionRejectionControl
  G5-SameValueRetentionRandomDirection
  G6-AdamWParallelDirectionControl
  G7-NoOpMatchedOverhead
  G8-GenericOptimizerStateControl

v1414_b_boundary_fms_real_lite.csv rows = 27
boundary methods =
  F-B1-AbstentionBoundaryFMS
  F-B2-ConstraintOnlyBoundaryFMS
  F-B3-ContinuousLowAmplitudeBoundaryFMS

v1414_b_boundary_controls.csv rows = 54
boundary controls = C0..C5

v1414_d_all_basis_substrate_hardening.csv rows = 135
Line D candidates =
  D-FOU26-NoMaterializeLifetimeAuditV2
  D-FOU27-LowFreqIdentityResidualV2
  D-FOU28-BandwiseSNRSafeWarmup
  D-FOU29-PhaseStableBandMixNoHighFreq
  D-FOU30-NoMaterializeLifetimeV3
  D-FOU31-HighFrequencyQuarantine
  D-RBF25-WidthConditionGuardNoTaskBranch
  D-RBF26-ActiveCenterOccupancyV2
  D-RBF27-WidthConditionIdentityResidual
  D-RBF28-CompactBumpNoDenseMaterialization
  D-RBF29-GaussianLocalK4TaskHealth
  D-WAV25-TriangularSupportV3
  D-WAV26-ScaleOccupancyNoTailTarget
  D-WAV27-LocalSupportOverlapDamping
  D-WAV28-LocalTailCoverageAudit
```

最新 route / audit 复核：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
e4_exploration_gate_pass = 0
q_causal_exploration_gate_pass = 0
q_route_hint = R3-FMSDirectionControlEquivalent
boundary_real_lite_pass_count = 0/9
boundary_mean_source_vs_best_control = -0.032401322214691726
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
official_s5_reached = 0
promotion_allowed = 0
manifest_missing_sum = 0
forbidden_violation_sum = 0
no_action_violation_sum = 0
```

复核结论：

```text
1. v14.14 仍没有达成 S4-lite / S4 / S5。
2. Line B/F 允许的 F-B1/F-B2/F-B3 已全部执行。
3. Q matched controls G4/G5 与 C0..C5 controls 已覆盖。
4. Line D 允许的 D-FOU/D-RBF/D-WAV substrate candidates 已覆盖。
5. 没有发现当前计划内仍可合法补跑的预注册分支。
6. 继续推进需要新增 FMS-M6/M7、F-CHE8/F-CHE9、B4/B5/B6、
   action token / controller / reset route，
   或使用 real fail pattern / audit metric 反推方向；这些均违反 v14.14 当前计划。
```

当前合法结论不变：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0
```

## 12. 用户再次追问后的二次状态复核

本次没有新增代码修改，也没有新增训练；只再次读取 latest official artifacts，并对照 v14.14 完整计划中的 Line B/F allowed definitions、Line D gate、Stop/continue contract 与 Codex must-not 条款，确认是否仍有可继续执行且不越界的分支。

最新 artifact 复核：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
e4_best_auc_mean = 0.9090909090909091
e4_best_leaveout_auc_min = 0.5
e4_exploration_gate_pass = 0
q_control_equivalent_fraction = 0.8888888888888888
q_causal_exploration_gate_pass = 0
boundary_real_lite_pass_count = 0/9
boundary_mean_source_vs_best_control = -0.032401322214691726
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
official_s5_reached = 0
promotion_allowed = 0
manifest_missing_sum = 0
forbidden_violation_sum = 0
no_action_violation_sum = 0
```

覆盖矩阵复核：

```text
Line B/F allowed definitions:
  F-B1-AbstentionBoundaryFMS
  F-B2-ConstraintOnlyBoundaryFMS
  F-B3-ContinuousLowAmplitudeBoundaryFMS
  artifact rows = 27

Boundary controls:
  C0-D-CHE-AdamW
  C1-D-CHE-AdamW-NoOpMatchedOverhead
  C2-D-CHE-AdamW-RandomMatchedNorm
  C3-D-CHE-AdamW-AdamWParallelDirectionControl
  C4-D-CHE-AdamW-GenericOptimizerStateControl
  C5-D-CHE-AdamW-SameActiveFractionControl
  artifact rows = 54

Q groups:
  G0-D-CHE-AdamW-baseline
  G1-D-CHE-FMS-M1..M5
  G2-RandomMatchedNorm
  G3-SameActiveFractionControl
  G4-SameDegreeProjectionRejectionControl
  G5-SameValueRetentionRandomDirection
  G6-AdamWParallelDirectionControl
  G7-NoOpMatchedOverhead
  G8-GenericOptimizerStateControl
  artifact rows = 189

Line D candidate_id count = 15:
  D-FOU26..31
  D-RBF25..29
  D-WAV25..28
  artifact rows = 135
```

计划边界复核：

```text
1. Line B/F 不允许 B4/B5/B6、F-CHE8/F-CHE9、FMS-M6/M7/M8。
2. 不允许 dataset/seed 分支、strength/lambda/interval grid。
3. 不允许 action token / controller / action bank / reset route。
4. 不允许 real fail pattern 作方向。
5. 不允许 LineC/CEp99/NLL/ECE/AUCtime 作方向。
6. Line D non-D-CHE <6/9 不是 hard stop，
   但不允许进入 official FMS proof。
```

二次复核结论：

```text
v14.14 仍没有达成 S4-lite / S4 / S5。
当前计划内 Line B/F、Line Q controls、Line D all-basis continuation 均已覆盖。
没有发现仍可合法补跑的预注册 method/control/substrate candidate。
继续推进需要下一版机制重写计划，而不是在 v14.14 内临时扩 token / controller / action / audit-directed route。
```

当前合法结论仍是：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0
```

## 13. 用户再次追问后的 Line Z / no-go 三次复核

本次没有新增代码修改，也没有新增训练；重点复核 route definitions、Line Z no-go artifact、next hypothesis queue、required manifest 与 forbidden/no-action audit，确认当前是否仍属于“继续执行预注册 fallback”还是“v14.14 计划已闭合”。

route / gate 复核：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
e4_exploration_gate_pass = 0
q_causal_exploration_gate_pass = 0
q_route_hint = R3-FMSDirectionControlEquivalent
boundary_exploration_gate_pass = 0
boundary_meaningful_gate_pass = 0
boundary_s4_gate_pass = 0
boundary_real_lite_pass_count = 0/9
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

required / audit 复核：

```text
required_artifact_rows = 29
required_artifact_missing_count = 0
manifest_missing_sum = 0
forbidden_violation_sum = 0
no_action_violation_sum = 0
```

Line Z artifacts：

```text
v1414_no_go_boundary.md:
  1. v14.14 已执行 Line R/E4/Q/B/F/D/C/Z。
  2. E4/Q 只提供 causal-value / control deconfound 诊断，不允许 promotion。
  3. Boundary real-lite 未达到 9/9，因此没有 official S5。
  4. D-CHE 外 basis 未达到 >=6/9 substrate exploration gate 时，不允许 official FMS proof。
  5. 当前不能新增 F-CHE8/F-CHE9、FMS-M6/M7、action token、controller 或 reset route；
     也不能用 real fail pattern 或 audit metric 反推方向。

v1414_next_hypothesis_queue.md:
  下一步需要新的 transfer-boundary causal-value hypothesis 或新的 all-basis substrate carrier；
  不能新增 action/controller/reset route，不能使用 audit metric 生成方向。
```

覆盖复核：

```text
v1414_method_surface_manifest.csv rows = 11
v1414_b_boundary_fms_real_lite.csv rows = 27
v1414_b_boundary_controls.csv rows = 54
v1414_q_fms_vs_matched_controls.csv rows = 189
v1414_d_all_basis_substrate_hardening.csv rows = 135
Line D candidate_id count = 15
```

三次复核结论：

```text
v14.14 没有达成 S4-lite / S4 / S5。
所有预注册 fallback 已执行到 Line Z。
当前计划内没有可继续执行且不越界的分支。
继续推进需要下一版 transfer-boundary causal-value hypothesis 或新的 all-basis substrate carrier；
不能在 v14.14 内临时新增 token / controller / action / reset route，
也不能用 real fail pattern 或 audit metric 反推方向。
```

当前合法结论仍是：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0
```
