# DG-KAN v14.13 TransferMechanismClarification FunctionalContinueOpen AllBasisParallel 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入本轮实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 weak proxy / real-lite / substrate replay 写成 promotion。

## 1. 计划理解

v14.13 的目标是定位 transfer-observability 到 FMS-realization 的断点，并在不新增 action token / controller / F-CHE8/F-CHE9 的前提下继续 functional exploration。

## 2. 代码修改

新增：

```text
experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py
```

实现：

```text
1. Line R provenance / forbidden / no-action-search audit。
2. Line E3 transfer-observability robust rebuild。
3. Line V3 proxy-to-effect mechanism chain decomposition。
4. Line F3 D-CHE FMS-M1..M5 bounded real-lite。
5. Line D all-basis cross-version reconciliation replay。
6. Line M MLP/generic control replay。
7. Line C LineC/tail/failure taxonomy。
8. required manifest、figures、code review packet、执行日志和复盘日志。
```

F3 method mapping：

```text
FMS-M1 -> F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection
FMS-M2 -> F-CHE6-PhaseScheduleDegreeFMS
FMS-M3 -> F-CHE-RT4-CompositeTransferTrust + actual_micro_horizon_gating
FMS-M4 -> F-CHE-RT3-DegreeEnergySafetyProjection
FMS-M5 -> F-CHE-RT2-ValueRetentionTrust
```

## 3. Line E3 结果

```text
best_feature = degree_projection_rejection_fraction
best_auc_mean = 0.75
best_leaveout_auc_min = 0.7586206896551724
exploration_gate_pass = 1
promotion_enabling_gate_pass = 0
```

## 4. Line V3 结果

```text
breakpoint_coverage = 1
ambiguous_rows_fraction = 0.0
dominant_breakpoint = V3-B6-ControlEquivalent
```

判断：

```text
Line V3 输出了断点分类；这只是 mechanism diagnosis，不允许 promotion。
```

## 5. Line F3 real-lite 结果

```text
f3_real_lite_pass_count = 1 / 9
f3_mean_source_vs_best_control = -0.032626233498255414
f3_source_fail_count = 40
f3_auctime_fail_count = 41
f3_tail_fail_count = 47
f3_linec_fail_count = 18
f3_reused_existing = 1
```

判断：

```text
F3 未达到 S4-lite 的 >=4/9 real-lite exploration gate。
real-lite 不能写成 official real success，也不能 promotion。
```

## 6. Line D all-basis 结果

```text
best_non_dche_family = D-FOU
best_non_dche_dataset_seed_pass_count = 2 / 9
line_d_official_fms_eligible_family_count = 0
```

判断：

```text
D-CHE 之外没有 basis 达到 >=6/9 exploration substrate gate；
因此不能进入 D-FOU/D-RBF/D-WAV official FMS proof。
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
1. v14.13 已执行 Line R/E3/V3/F3/D/M/C/Z。
2. E3/V3 提供了机制断点诊断，但没有打开 promotion。
3. F3 real-lite 未达到 >=4/9，因此没有 S4-lite。
4. All-basis parallel 仍 blocked，D-CHE 外没有 family 达到 >=6/9。
5. 当前不能新增 method/action/controller/reset route，也不能把 diagnostics 写成 success。
```

## 9. 用户再次追问后的控制组与 Line D 补齐

本次复核 v14.13 计划后发现两个仍可合法继续的点：

```text
1. F3 controls 显式列出 SameActiveFractionControl；
   初始 v14.13 runner 只把它记录为 not executable。
2. Line D 候选方向显式列出：
   D-FOU-HighFrequencyQuarantine
   D-WAV-LocalTailCoverageAudit
   初始 v14.13 只复写了 v14.12.1/v14.11 artifacts。
```

### 9.1 SameActiveFractionControl 补齐

代码修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 C5-D-CHE-AdamW-SameActiveFractionControl。
  该 control 使用当前 train batch / FMS state 估计参考 active key 数量，
  然后随机选择同数量 key 施加 random matched scale。

experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py
  将 C5 加入 D_CHE_CONTROLS。
```

合法性：

```text
1. C5 是 control，不是新 FMS mechanism。
2. 不新增 F-CHE8/F-CHE9 / FMS-M6。
3. 不新增 action token / controller。
4. 不使用 validation/test/future/query。
5. 不使用 LineC/CEp99/NLL/ECE/AUCtime 生成方向。
```

补齐后 F3 结果：

```text
v1413_method_surface_manifest.csv rows = 17
C0..C5 controls executed_in_f3 = 1
FMS-M1..M5 executed_in_f3 = 1
not_executed SameActiveFractionControl rows = 0
```

F3 method summary：

| method | rows | strict pass | dataset-seed pass | mean source vs best control |
|---|---:|---:|---:|---:|
| FMS-M1-GenericValueOnlyDegreeSafety | 9 | 0 | 0 | -0.026791 |
| FMS-M2-DegreeContinuousBoundary | 9 | 0 | 0 | -0.028283 |
| FMS-M3-MicroHorizonGatedBoundary | 9 | 0 | 0 | -0.038901 |
| FMS-M4-RecoveryLagSuppressedBoundary | 9 | 1 | 1 | -0.028261 |
| FMS-M5-ProjectionRetentionFloor | 9 | 0 | 0 | -0.040895 |

control summary：

| control | rows | mean NLL | mean source vs AdamW |
|---|---:|---:|---:|
| C0-D-CHE-AdamW | 9 | 0.666528 | 0.000000 |
| C1-D-CHE-AdamW-NoOpMatchedOverhead | 9 | 0.661807 | 0.004721 |
| C2-D-CHE-AdamW-RandomMatchedNorm | 9 | 0.666917 | -0.000389 |
| C3-D-CHE-AdamW-AdamWParallelDirectionControl | 9 | 0.651174 | 0.015354 |
| C4-D-CHE-AdamW-GenericOptimizerStateControl | 9 | 0.644191 | 0.022337 |
| C5-D-CHE-AdamW-SameActiveFractionControl | 9 | 0.658488 | 0.008040 |

判断：

```text
补齐 C5 后，F3 仍只有 1/9；
mean source vs best control = -0.032626233498255414。
SameActiveFractionControl 没有打开 FMS success，ControlEquivalent blocker 仍成立。
```

### 9.2 Line D extra substrate-only hardening

代码修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 D-FOU31-HighFrequencyQuarantine。
  新增 D-WAV28-LocalTailCoverageAudit。

experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py
  新增读取 line_d_extra_highfreq_tail_v1413 artifact。
  将 extra D-FOU/D-WAV rows 合入 v1413_d_fou_hardening.csv / v1413_d_wav_monitor.csv。
  v1413_d0_cross_version_reconciliation.csv 新增：
    v1413_extra_dataset_seed_pass_count
    v1413_extra_best_candidate
    v1413_best_replay_or_extra_dataset_seed_pass_count。
```

执行范围：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
candidates =
  D-FOU31-HighFrequencyQuarantine
  D-WAV28-LocalTailCoverageAudit
train_size = 256
val_size = 128
epochs = 1
linec_seeds = 12319500,12319501,12319502
official_fms_proof_executed = 0
promotion_allowed = 0
```

extra Line D route：

```text
route = R8-NonRATSubstrateStillMissing
candidate_rows = 18
linec_rows = 54
best_family = D-FOU
best_family_dataset_seed_pass_count = 0/9
exploration_open_family_count = 0
official_fms_eligible_family_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

extra candidate summary：

| candidate | rows | pass | best delta vs MLP | best NLL ratio | best LineC | median step ratio | main blocker |
|---|---:|---:|---:|---:|---:|---:|---|
| D-FOU31-HighFrequencyQuarantine | 9 | 0 | -0.156250 | 1.564545 | 0.666667 | 0.410670 | task delta fail |
| D-WAV28-LocalTailCoverageAudit | 9 | 0 | -0.187500 | 1.563148 | 0.666667 | 0.521715 | task delta + LineC fail |

合入 v14.13 finalizer 后：

```text
v1413_d_fou_hardening.csv rows = 54
v1413_d_wav_monitor.csv rows = 36
v1413_d0_cross_version_reconciliation:
  D-FOU v1412_replay = 2/9, v1413_extra = 0/9, best = 2/9
  D-RBF v1412_replay = 1/9, v1413_extra = 0/9, best = 1/9
  D-WAV v1412_replay = missing, v1413_extra = 0/9, best = 0/9
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
```

判断：

```text
Line D 计划中剩余的 HighFrequencyQuarantine / LocalTailCoverageAudit 方向已执行。
它们没有打开 >=6/9 exploration substrate gate。
因此 D-FOU/D-RBF/D-WAV 仍不能进入 official FMS proof。
```

## 10. 更新后的最终 route

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
f3_real_lite_pass_count = 1/9
f3_mean_source_vs_best_control = -0.032626233498255414
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 2/9
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

required manifest：

```text
path = results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413/v1413_required_artifact_manifest.csv
rows = 35
missing_required_rows = 0
```

当前停止边界：

```text
v14.13 没有达成 S4/S5。
现在已覆盖：
  Line R/E3/V3/F3/D/M/C/Z；
  F3 SameActiveFractionControl；
  Line D HighFrequencyQuarantine；
  Line D LocalTailCoverageAudit。

继续推进需要新增 FMS-M6/M7 或 F-CHE8/F-CHE9、
新增 action token / controller / reset route、
或使用 real fail pattern / audit metric 反推方向。
这些均违反 v14.13 当前计划。
```

## 11. 用户再次追问后的状态复核

本次没有新增训练；重新读取 v14.13 完整计划中的 F3 fallback ladder、Line D gate、Stop/continue contract，并复核最新 artifact，确认是否还有未执行且不越界的分支。

计划原文关键边界：

```text
1. F3 <3/9 时：
   Do not add methods；
   Use Line V3 failure taxonomy；
   若 ControlEquivalent dominates，report FMS value no-go；
   Continue Line D all-basis；do not stop whole experiment。
2. FMS-M1..M5 不允许扩成 M6/M7/M8，
   除非 Line Z 判定进入下一版重写计划。
3. D-FOU/RBF/WAV 未过 dataset_seed_pass_count >= 6/9 substrate gate，
   不允许 official FMS proof。
4. Promotion 只能在 full S5 gate 达成后打开。
```

最新 artifact 复核：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
e3_exploration_gate_pass = 1
e3_promotion_enabling_gate_pass = 0
v3_dominant_breakpoint = V3-B6-ControlEquivalent
f3_real_lite_pass_count = 1/9
f3_mean_source_vs_best_control = -0.032626233498255414
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
manifest_missing_sum = 0
```

覆盖性复核：

```text
1. F3 FMS-M1..M5 已执行。
2. F3 controls AdamW / NoOpMatchedOverhead / RandomMatchedNorm /
   GenericMLPFMS / SameActiveFractionControl 已执行。
3. V3 breakpoint taxonomy 已输出，dominant = ControlEquivalent。
4. Line D D-FOU / D-RBF / D-WAV cross-version replay 与 v14.13 extra hardening 已执行。
5. Line D best non-D-CHE 仍只有 2/9，低于 >=6/9 exploration substrate gate。
6. required manifest 缺失为 0；forbidden / no-action-search violation 为 0。
```

结论：

```text
v14.13 仍没有达成 S4/S5。
当前计划内可合法执行的 fallback 已覆盖。
继续推进需要新增 FMS-M6/M7、F-CHE8/F-CHE9、action token / controller / reset route，
或使用 real fail pattern / audit metric 反推方向；这些都违反 v14.13 当前计划。
因此本次不启动新的 v14.13 训练。
当前合法结论仍是：
  route = R4-AllBasisSubstrateBlocked
  minimum_success = S3-DCHESyntheticFMSPass
  official_s5_reached = 0
  promotion_allowed = 0
```

## 12. 用户再次追问后的覆盖矩阵复核

本次没有新增训练；按计划原文中的 method / control / Line D candidate 列表，对实际 artifact 做覆盖矩阵复核，确认是否存在“计划允许但漏跑”的分支。

F3 method / control 覆盖：

```text
v1413_f3_dche_fms_real_lite.csv rows = 45
F3 methods =
  FMS-M1-GenericValueOnlyDegreeSafety
  FMS-M2-DegreeContinuousBoundary
  FMS-M3-MicroHorizonGatedBoundary
  FMS-M4-RecoveryLagSuppressedBoundary
  FMS-M5-ProjectionRetentionFloor

v1413_f3_dche_fms_controls.csv rows = 54
controls =
  C0-D-CHE-AdamW
  C1-D-CHE-AdamW-NoOpMatchedOverhead
  C2-D-CHE-AdamW-RandomMatchedNorm
  C3-D-CHE-AdamW-AdamWParallelDirectionControl
  C4-D-CHE-AdamW-GenericOptimizerStateControl
  C5-D-CHE-AdamW-SameActiveFractionControl
```

Line D candidate 覆盖：

```text
v1413_d_fou_hardening.csv rows = 54
D-FOU candidates =
  D-FOU26-NoMaterializeLifetimeAuditV2
  D-FOU27-LowFreqIdentityResidualV2
  D-FOU28-BandwiseSNRSafeWarmup
  D-FOU29-PhaseStableBandMixNoHighFreq
  D-FOU30-NoMaterializeLifetimeV3
  D-FOU31-HighFrequencyQuarantine

v1413_d_rbf_hardening.csv rows = 45
D-RBF candidates =
  D-RBF25-WidthConditionGuardNoTaskBranch
  D-RBF26-ActiveCenterOccupancyV2
  D-RBF27-WidthConditionIdentityResidual
  D-RBF28-CompactBumpNoDenseMaterialization
  D-RBF29-GaussianLocalK4TaskHealth

v1413_d_wav_monitor.csv rows = 36
D-WAV candidates =
  D-WAV25-TriangularSupportV3
  D-WAV26-ScaleOccupancyNoTailTarget
  D-WAV27-LocalSupportOverlapDamping
  D-WAV28-LocalTailCoverageAudit
```

最新 route 仍为：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
f3_real_lite_pass_count = 1/9
f3_mean_source_vs_best_control = -0.032626233498255414
v3_dominant_breakpoint = V3-B6-ControlEquivalent
line_d_best_non_dche_dataset_seed_pass_count = 2/9
official_s5_reached = 0
promotion_allowed = 0
manifest_missing_sum = 0
```

复核结论：

```text
v14.13 完整计划内的 F3 method / control 和 Line D candidate 已覆盖。
不存在仍可合法补跑的预注册 method 或 substrate candidate。
继续推进只能依赖新增 FMS-M6/M7、F-CHE8/F-CHE9、action/controller/reset route，
或用 audit/fail pattern 反推方向；这些均不允许。
因此本次仍不启动新训练，合法结论不变。
```

## 13. 用户再次追问后的 V3 gate 复核

本次没有新增训练；专门复核 v14.13 计划中 Line V3 的 ambiguous diagnostics 分支，确认是否需要继续补充 V3 diagnostics。

计划 gate：

```text
breakpoint_coverage = 1
ambiguous_rows_fraction <= 0.20
controls_explained_rows recorded
```

最新 artifact：

```text
v3_breakpoint_coverage = 1
v3_ambiguous_rows_fraction = 0.0
v3_dominant_breakpoint = V3-B6-ControlEquivalent
v1413_v3_proxy_to_effect_chain.csv rows = 45
```

V3 breakpoint summary：

| breakpoint | rows | fraction | strict pass rows | mean source |
|---|---:|---:|---:|---:|
| V3-B3-EventTooWeak | 1 | 0.022222222222222223 | 0 | 0.022882819175720215 |
| V3-B5-MicroHorizonGoodRealBad | 7 | 0.15555555555555556 | 0 | 0.005020201206207275 |
| V3-B6-ControlEquivalent | 36 | 0.8 | 0 | -0.04305057144827313 |
| V3-Pass | 1 | 0.022222222222222223 | 1 | 0.02361583709716797 |

结论：

```text
V3 ambiguous diagnostics 分支不触发。
当前主要断点是 ControlEquivalent，而非 V3 未解释。
因此没有合法的 V3 diagnostic 补跑分支。
最终 route 仍是 R4-AllBasisSubstrateBlocked；
official_s5_reached = 0；
promotion_allowed = 0。
```

## 14. 用户再次追问后的 Line Z / no-go 复核

本次没有新增训练；复核 v14.13 route definitions、Line Z/no-go artifact、required manifest、forbidden/no-action audit，确认当前是否属于“必须继续到预注册 fallback”还是“当前计划闭合后等待下一版计划”。

计划 route 边界：

```text
S4-lite 要求 F3 real_lite_pass_count >= 4/9。
S4 exploration 要求 F3 real_lite_pass_count >= 6/9。
S5 要求 full real 9/9 official gate。
R4-AllBasisSubstrateBlocked 表示：
  D-CHE no S4/S5 and no other basis reaches >=6/9 substrate exploration。
```

v1413_no_go_boundary.md：

```text
1. v14.13 已执行 Line R/E3/V3/F3/D/M/C/Z。
2. E3/V3 提供了机制断点诊断，但没有打开 promotion。
3. F3 real-lite 未达到 >=4/9，因此没有 S4-lite。
4. All-basis parallel 仍 blocked，D-CHE 外没有 family 达到 >=6/9。
5. 当前不能新增 method/action/controller/reset route，也不能把 diagnostics 写成 success。
```

v1413_next_hypothesis_queue.md：

```text
下一步需要新的 transfer-observable FMS definition 或新的 all-basis substrate carrier；
不能新增 action/controller/reset route，不能使用 audit metric 生成方向。
```

最新核验：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
e3_promotion_enabling_gate_pass = 0
v3_breakpoint_coverage = 1
v3_ambiguous_rows_fraction = 0.0
v3_dominant_breakpoint = V3-B6-ControlEquivalent
f3_real_lite_pass_count = 1/9
f3_mean_source_vs_best_control = -0.032626233498255414
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
forbidden audit violation sum = 0
no-action-search audit violation sum = 0
required manifest missing sum = 0
```

结论：

```text
v14.13 没有达成 S4-lite / S4 / S5。
所有预注册 fallback 已执行到 Line Z。
当前不再启动新训练；继续推进需要下一版计划，而不是在 v14.13 内临时扩 token / controller / action / audit-directed route。
```

## 15. 用户再次追问后的 route precedence 复核

本次没有新增训练；复核 `R4-AllBasisSubstrateBlocked` 与 `R5-GenericFMSConfound` 的关系，避免把 ControlEquivalent breakpoint 误判成还需要继续跑的新分支。

计划 route 定义：

```text
R4-AllBasisSubstrateBlocked:
  D-CHE no S4/S5 and no other basis reaches >=6/9 substrate exploration.

R5-GenericFMSConfound:
  MLP/generic controls explain observed gains.
```

最新 artifact：

```text
route = R4-AllBasisSubstrateBlocked
v3_dominant_breakpoint = V3-B6-ControlEquivalent
V3-B6 rows = 36 / 45
V3-B6 mean_source_vs_best_control = -0.04305057144827313
F3 pass rows = 1 / 9
f3_mean_source_vs_best_control = -0.032626233498255414
line_d_best_non_dche_dataset_seed_pass_count = 2 / 9
```

判断：

```text
1. V3-B6-ControlEquivalent 已记录 controls_explained_rows，说明当前 FMS event 多数被 control 等价解释。
2. 但 F3 没有形成整体 observed gains：
   real-lite 只有 1/9，mean source vs best control 为负。
3. Line D 仍未达到 >=6/9 substrate exploration gate。
4. 因此 primary route 保持 R4-AllBasisSubstrateBlocked 是合法的；
   ControlEquivalent 是 V3 机制断点，不打开 R5 后的任何新训练分支。
```

结论：

```text
route precedence 没有发现需要修复的 artifact 错误。
当前合法结论仍是：
  route = R4-AllBasisSubstrateBlocked
  minimum_success = S3-DCHESyntheticFMSPass
  official_s5_reached = 0
  promotion_allowed = 0
```
