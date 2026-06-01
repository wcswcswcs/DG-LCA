# DG-KAN v14.9：FMS Causal Specificity Audit + All-Basis Parallel Substrate 完整计划

> 版本：v14.9 execution plan  
> 生成时间：2026-05-29 Asia/Singapore  
> 公式格式：Typora 友好，仅使用 `$...$` 与 `$$...$$`  
> 核心修正：v14.8 已经把 `zero_moment_reset / optimizer-state transport` 路线打回 diagnostic 层。下一步不能继续寻找“好动作”、不能继续扩 reset/action token、不能启动 controller。v14.9 的目标是用因果分解明确：FMS 是否有不被 generic optimizer-state reset 解释的特异性价值；同时继续 all-basis substrate 并行修复。

---

# 0. 项目总目标与当前进展

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个局部 positive 的 update trick。项目要证明的是：

$$
\boxed{
\text{label-free strict FC-PureKAN substrate/base}
+
\text{functional update}
>
\text{same substrate/base + ordinary backprop / AdamW controls}
}
$$

更具体地说，最终系统必须同时满足：

```text
1. 表达力不打折，不靠弱模型降低 gate；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹健康，AUC-step / AUC-time 不输；
4. 几何健康：CouplingR2、NoiseSignalLeak、ReservoirRatio、tail/calibration 不坏；
5. functional update 的收益必须击败 NoOp / Random / AdamWParallel / generic optimizer-state controls；
6. functional direction 不得使用 validation/test/future/query，不得使用 LineC / CEp99 / NLL / ECE / AUCtime audit metric 生成方向；
7. 不允许 label-informed initialization，不允许 dataset-name branch，不允许 seed-specific scaling；
8. 不允许把 endpoint-only、diagnostic、oracle、control-equivalent、harmless-null 写成 promotion。
```

当前项目的真实状态是：

```text
1. Rational 仍是最稳定的 functional substrate。
2. FMS / PopRisk infrastructure 已经成熟，真实写回、per-example gradient、persistent state、projection trace、LineC/tail audit 都可执行。
3. v14.3-v14.5 曾经把 Rational-FMS 推到 synthetic S3 和 real S4b，best real 3x3 到 6/9。
4. v14.6.1 的 zero_moment_reset diagnostic 曾把 coverage 推到 9/9，但 v14.7/v14.8 证明这不能直接 promotion。
5. v14.8 证明 endpoint gain 被 generic optimizer-state reset / momentum suppression confound 解释；best FMS strict 为 0/9，endpoint 为 6/9；generic optimizer-only control G4 endpoint 为 7/9。
6. 因此 v14.9 不再优化 zero reset，不再扩 action bank；必须先做 FMS causal specificity audit。
7. Non-RAT basis 仍未形成 official FMS substrate，但不能放弃；all-basis substrate repair 继续并行。
```

---

# 1. v14.8 独立分析：为什么不能继续 reset / action 小修

v14.8 的合法 route 是：

```text
route = R3-GenericOptimizerStateResetConfound
official_s5_reached = 0
promotion_allowed = 0
best_fms_method = S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow
best_fms_strict_pass_count = 0/9
best_fms_endpoint_vs_adamw_pass_count = 6/9
G4-RAT-AdamW-EventMatchedRandomReset endpoint = 7/9
G3/G5/G7 endpoint = 6/9
generic_optimizer_state_confound = 1
```

这说明现在不能再说：

```text
zero_moment_reset 已经找到 functional 突破；
只是 step_time 差一点；
只要再调 mask / strength / refresh / recovery window 就行；
只要启动 controller 就能选对。
```

更准确的判断是：

$$
\boxed{
\text{当前 reset / momentum suppression 产生的 endpoint gain 不是 FMS-specific；}
\text{它可以由 generic optimizer-state reset controls 解释。}
}
$$

v14.8 还暴露了三个硬 blocker：

```text
1. FMS specificity blocker:
   FMS candidate endpoint 6/9，不如 G4 generic optimizer reset 7/9；strict 全部 0/9。

2. affected-coordinate blocker:
   affected mask median 约 0.995，说明 affected-only reset 近似全局 reset；
   v14.7 top25 sparse reset 已经退化，不能继续 mask grid。

3. efficiency blocker:
   best S5 path median step_time_ratio = 1.295544 > 1.25；
   但 route 已是 generic confound，因此不应先做 overhead optimization。
```

所以 v14.9 的首要任务不是修 reset，而是回答：

$$
\boxed{
\text{FMS 本身是否仍有不被 generic optimizer-state dynamics 解释的 causal value？}
}
$$

---

# 2. 必须吸收的历史教训：不要重演 v9 / v12 action-search 错误

过去 v9 / v12 系列最严重的错误是 action-centric illusion：

```text
1. 看到某个局部 good event，就继续扩 action。
2. 看到 oracle frontier，就以为 controller 可学。
3. 看到 actuatability，就以为有 causal value。
4. 看到 harmless-null / endpoint improvement，就以为 safe-good。
5. 没有 apply / outcome / controls / runtime materialization，就开始谈 controller。
6. risk / support / value / control 混成一个分数。
7. 对失败 dataset / seed 写补丁，误入 dataset-specific path。
```

v14.9 必须写死以下禁止项：

```text
1. 不新增 K-RT / K-AUC / K-FL / reset action token。
2. 不启动 controller。
3. 不做 strength / lambda / lr / refresh / mask topK grid。
4. 不把 endpoint-only 写成 strict。
5. 不把 random/full/generic reset controls 也能复现的结果写成 FMS-specific。
6. 不用 LineC / CEp99 / NLL / ECE / AUCtime / Brier audit metric 生成方向。
7. 不按 dataset / seed 写 branch。
8. 不拼接不同 run 的局部 positive。
9. 不降低 step_time / memory / S5 gate。
10. Non-RAT substrate 未过 gate，不进入 official FMS proof。
```

若 Codex 违反任一条，route 直接为：

```text
R0-ActionSearchOrAuditLeakViolation
```

---

# 3. v14.9 的核心假设

## H1：v14.8 的 gain 主要来自 generic optimizer-state dynamics，而非 FMS-specific value

需要用 difference-in-differences 检验：

$$
\Delta_{FMS\text{-}specific}
=
[(FMS + T) - (FMS)]
-
[(AdamW + T_{matched}) - (AdamW)]
$$

其中 $T$ 是同语义、同频率、同开销的 state transport / momentum suppression。若：

$$
\Delta_{FMS\text{-}specific} \le 0,
$$

则说明 transport 不是 FMS-specific，只是 generic optimizer reset。

## H2：当前 FMS value path 可能仍存在，但被 optimizer-state confound 掩盖

v14.3-v14.5 的 S3/S4b 说明 FMS 不是完全没信号。v14.9 要把 value 与 optimizer-state reset 分开：

```text
FMS value source:
  train-stream per-example gradient / FMS metric state / PopRisk signal。

Optimizer-state effects:
  reset / beta1 suppression / periodic reset / no-momentum warmup / full reset。
```

如果 FMS value 在统一 optimizer-state 条件下仍然优于 AdamW / Random / Generic controls，才有资格继续。

## H3：affected mask 过宽可能是实现语义问题，也可能是 FMS 本身 diffuse

v14.8 的 affected median 约 0.995。v14.9 不允许继续 topK grid，只允许做语义审计：

```text
A1: implementation bug，affected mask 应该稀疏但实现全局化；
A2: FMS update 本身就是全局 diffuse，没有 sparse affected support；
A3: sparse support 会破坏 value，因此不能走 affected-only route。
```

只有 A1 允许代码修复；A2/A3 都不允许继续 mask search。

## H4：all-basis substrate 仍是并行刚需

Functional update 不能只在 Rational 上循环。Rational 是主 carrier，但 RBF/FastKAN、Chebyshev、Fourier、Wavelet 必须继续作为 active substrate line。Non-RAT 不能未过 substrate 就 official FMS，但必须持续推进：

```text
RBF/FastKAN: task collapse / center occupancy / width condition；
Chebyshev: exact path后 full-step/update lifetime；
Fourier: low-frequency task-health / incremental memory；
Wavelet: compact workspace 已有线索，需 seed-robust task-health / LineC。
```

---

# 4. Line R：implementation / provenance / no-action-search audit

## 4.1 目标

保证 v14.9 不是又一次 reset/action 小修。必须审计：

```text
1. 没有新增 action token；
2. 没有 controller；
3. 没有 grid search；
4. 没有 audit metric direction；
5. 没有 dataset/seed branch；
6. 没有 endpoint-only promotion；
7. 所有 controls 与 FMS candidate 在同一 budget、同一 strict gate 下比较。
```

## 4.2 必须输出

```text
v149_no_action_search_audit.csv
v149_forbidden_information_audit.csv
v149_code_review_manifest.csv
v149_route_decision.json
v149_required_artifact_manifest.csv
v149_code_review_packet.zip
```

## 4.3 Pass gate

```text
no_action_search_violation_count = 0
forbidden_information_violation_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0 unless S5 full gate passes
```

---

# 5. Line Q：v14.6.1 / v14.7 / v14.8 semantic diff closure

## 5.1 目标

解释为什么：

```text
v14.6.1 diagnostic zero reset looked like 9/9；
v14.7 every-step reset endpoint 9/9 but control-confounded；
v14.7 event-only reset only 4/9；
v14.8 S5 post-event recovery strict 0/9 endpoint 6/9；
generic controls G4 reached 7/9。
```

## 5.2 必须记录字段

```text
run_id
semantics_type: diagnostic_old / every_step / event_only / post_event_window / generic_control
strict_pass_count
endpoint_vs_adamw_pass_count
source_mean
AUCtime_median
tail_fail_count
LineC_fail_count
step_time_median
controls_endpoint_full_count
specificity_pass_count
```

## 5.3 判定

```text
R1-DiagnosticMetricInflation:
  only old metric passes; strict never passes.

R2-ContinuousMomentumSuppressionNotFMSEvent:
  every-step passes, event-only/post-event fails.

R3-GenericOptimizerStateResetConfound:
  generic controls match or exceed FMS endpoint/strict.
```

v14.8 已经指向 R3；v14.9 只做 final closure，不再在此处寻新 action。

---

# 6. Line G：generic optimizer-state reset controls closure

## 6.1 目标

把 generic reset / momentum suppression 的上界测清楚，避免把 generic optimizer trick 写成 FMS。

## 6.2 Methods

```text
G0-RAT-AdamW
G1-RAT-AdamW-Beta1Zero
G2-RAT-AdamW-Beta1Half
G3-RAT-AdamW-PeriodicMomentReset
G4-RAT-AdamW-EventMatchedRandomReset
G5-RAT-AdamW-FullMomentResetAtFMSIntervalsNoFMS
G6-RAT-AdamW-RMSPropLikeNoMomentum
G7-RAT-AdamW-NoMomentumWarmupThenAdamW
G8-RAT-AdamW-MatchedOverheadNoStateChange
```

注意：G8 是 overhead-only control，不改变 optimizer state，只匹配计算开销。

## 6.3 指标

```text
strict_pass_count
endpoint_vs_adamw_pass_count
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
step_time_ratio
memory_ratio
```

## 6.4 Gate

如果任何 generic control 满足：

$$
endpoint_{generic} \ge endpoint_{FMS}
$$

或：

$$
strict_{generic} \ge strict_{FMS},
$$

则当前 state-transport route 不允许写成 FMS-specific。

---

# 7. Line F：FMS-specific causal decomposition

## 7.1 目标

在相同 optimizer-state condition 下检验 FMS 是否仍有 causal value。

## 7.2 Factorial design

比较四组：

```text
A: AdamW
B: AdamW + generic transport T
C: FMS + no transport
D: FMS + same transport T
```

计算：

$$
\Delta_{generic} = B - A
$$

$$
\Delta_{FMS} = C - A
$$

$$
\Delta_{interaction} = D - C - (B - A)
$$

其中 interaction 才是 FMS-state-transport 的特异性增益。若：

$$
\Delta_{interaction} \le 0,
$$

则不能继续 state-transport route。

## 7.3 Methods

```text
F0-RAT-AdamW
F1-RAT-AdamW-GenericBestTransport
F2-RAT-FMS-NoTransport
F3-RAT-FMS-GenericBestTransportMatched
F4-RAT-FMS-ValuePathOnly-NoStateTransport
F5-RAT-FMS-DirectionRemoved-StateOnly
F6-RAT-FMS-RandomDirectionMatchedState
F7-RAT-FMS-AdamWParallelDirectionControl
```

F5/F6 是关键 deconfound：

```text
F5 只保留 FMS timing/state，移除 FMS direction；
F6 用 random matched norm direction 替代 FMS direction；
若 F5/F6 与 F3 接近，则 FMS direction 没有特异性。
```

## 7.4 主要指标

```text
fms_specific_source_delta
fms_specific_auc_delta
fms_specific_tail_delta
fms_specific_linec_delta
interaction_strict_pass_count
interaction_endpoint_pass_count
control_equivalent_flag
harmless_null_flag
```

## 7.5 Pass / no-go

探索通过：

```text
interaction_endpoint_pass_count >= 6/9
and fms_specific_source_delta_mean > 0
and generic controls do not match interaction rows
```

Official S5 通过：

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
generic/random/full reset controls fail
MLP analog does not explain gain
promotion_allowed = 1 only after code/provenance pass
```

No-go：

```text
R3-GenericOptimizerStateResetConfound:
  generic controls explain FMS endpoint or strict gain.

R4-FMSDirectionControlEquivalent:
  random/matched direction with same state semantics explains gain.

R5-HarmlessNullNoValue:
  FMS does not harm but source <= controls.
```

---

# 8. Line A：affected-coordinate semantics audit

## 8.1 目标

不做 mask grid，只判定 affected mask 的语义。

## 8.2 必须记录

```text
affected_fraction
rolewise_affected_fraction
source_value_mass_in_affected
source_value_mass_in_unaffected
cos_fms_update_affected_vs_full
cos_fms_update_unaffected_vs_full
sparse_top25_endpoint_delta
sparse_top25_source_delta
sparse_top25_auc_delta
mask_definition_uses_audit_metric
```

## 8.3 三分路由

```text
A1-AffectedMaskImplementationBug:
  affected fraction should be sparse but implementation marks broad due to bug.
  允许代码修复，但不允许 new action。

A2-AffectedSetIntrinsicallyBroad:
  FMS update truly diffuse; affected-only reset is semantically all-role reset.
  停止 affected-only route。

A3-SparseAffectedMaskDestroysValue:
  sparse mask destroys source/AUC value.
  停止 sparse mask search。
```

只有 A1 才允许修代码。A2/A3 都不允许继续 mask search。

---

# 9. Line E：overhead / timing instrumentation

## 9.1 目标

只有当 Line F 证明 FMS-specific value 之后，才允许做 overhead optimization。若 Line F no-go，则 overhead optimization 也关闭。

## 9.2 分解字段

```text
per_example_gradient_time
fms_state_update_time
basis_projection_time
state_transport_time
optimizer_update_time
linec_audit_time
artifact_logging_time
cuda_sync_time
python_overhead_time
```

## 9.3 Gate

```text
step_time_ratio <= 1.25
memory_ratio <= 1.25
```

不允许降低 gate。若 FMS-specific value 成立但 timing fail，route：

```text
R6-FMSValueButOverheadBlocked
```

---

# 10. Line M：MLP analog controls

## 10.1 目标

判断发现是否是 KAN-specific。

## 10.2 Methods

```text
M0-MLP-AdamW
M1-MLP-FMS-NoTransport
M2-MLP-FMS-MatchedGenericTransport
M3-MLP-RandomDirectionMatchedTransport
M4-MLP-FMS-DirectionRemoved-StateOnly
M5-MLP-Beta1Zero
M6-MLP-PeriodicMomentReset
```

## 10.3 判定

如果 MLP analog 也达到同等 strict / endpoint，并且 difference-in-differences 不低于 Rational，则不能 claim KAN-specific，只能写：

```text
GenericOptimizerOrFMSMechanismObserved_KANSpecificNotEstablished
```

---

# 11. Line D：all-basis substrate parallel repair

Functional 主线不能让其他 basis 停摆。v14.9 继续 all-basis 并行，但 Non-RAT 未过 substrate 不允许 official FMS。

## 11.1 Rational

角色：主 FMS carrier，主要用于 causal specificity audit。

记录：

```text
D-RAT no-regression task/AUC/LineC/tail
rational denominator / derivative telemetry
group diversity
FMS value retention
```

## 11.2 Fourier

当前状态：低频方向是最有希望的 Non-RAT，但 single-config 只有 3/9，audit best 4/9；不能拼接配置。

v14.9 只允许 substrate repair：

```text
D-FOU23-LowFreqIdentityResidualUnified
D-FOU24-BandwiseSNRWarmupNoFMS
D-FOU25-PhaseStableBandMixNoHighFreq
D-FOU26-NoMaterializeLifetimeAuditV2
```

Gate：

```text
full 3x3 substrate >= 6/9 for exploration
full 3x3 substrate = 9/9 for official FMS eligibility
```

## 11.3 Wavelet

当前状态：有 compact workspace 与 seed-local signal，但 full 3x3 substrate 不稳。

v14.9 只允许 substrate repair：

```text
D-WAV23-TriangularSupportStableV2
D-WAV24-ScaleOccupancyHardeningV2
D-WAV25-LocalTailCoverageWithoutAuditMetric
D-WAV26-ReservoirStableTrainEntropyGeometry
```

不得把 seed0 / seed-specific output geometry 写成 success。

## 11.4 RBF / FastKAN

当前状态：workspace 可动，但 task collapse。

v14.9 repair：

```text
D-RBF23-CompactBumpIdentityResidual
D-RBF24-ActiveCenterOccupancyNoDense
D-RBF25-WidthConditionGuardNoTaskBranch
D-RBF26-FastKANGaussianLocalK4NoDense
```

必须记录：

```text
center_occupancy_entropy
empty_center_fraction
width_condition
out_of_grid_fraction
task_health
LineC_pass_rate
```

## 11.5 Chebyshev

当前状态：exact/no-materialize 有工程信号，但 full-step/update lifetime 与 task-health 不闭合。

v14.9 repair：

```text
D-CHE23-LowDegreeIdentityResidual
D-CHE24-HighDegreeLateEnable
D-CHE25-DegreeEnergyDampingNoAudit
D-CHE26-RecurrenceLifetimeV2
```

---

# 12. Line C：geometry / tail audit only

Line C 继续记录，但不得生成 direction。

必须记录：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
NLL
ECE
Brier
margin_p10
AUC_step
AUC_time
LineC_pass
```

重点解释：

```text
1. source fail 是 value 不足；
2. AUCtime fail 是 trajectory cost；
3. CEp99/NLL/ECE fail 是 tail/calibration；
4. LineC fail 是 signal/reservoir/noise geometry；
5. step_time fail 是 deployability。
```

---

# 13. 必须生成的 CSV / JSON artifacts

```text
v149_route_decision.json
v149_progress_table.csv
v149_no_action_search_audit.csv
v149_forbidden_information_audit.csv
v149_semantic_diff.csv
v149_generic_optimizer_controls.csv
v149_fms_specificity_factorial.csv
v149_difference_in_differences_summary.csv
v149_affected_mask_semantics.csv
v149_overhead_breakdown.csv
v149_mlp_analog_controls.csv
v149_all_basis_substrate_status.csv
v149_linec_tail_audit.csv
v149_failure_taxonomy.csv
v149_no_go_boundary.md
v149_next_hypothesis_queue.md
v149_required_artifact_manifest.csv
v149_code_review_packet.zip
```

---

# 14. 必须生成的可视化

```text
fig_v149_diff_in_diff_fms_specificity.svg
fig_v149_generic_reset_vs_fms_endpoint_strict.svg
fig_v149_affected_mask_semantics.svg
fig_v149_step_time_breakdown.svg
fig_v149_mlp_vs_rat_fms_specificity.svg
fig_v149_all_basis_substrate_matrix.svg
fig_v149_linec_tail_failure_heatmap.svg
fig_v149_route_decision_tree.svg
```

---

# 15. 最终 route 定义

```text
S5-OfficialFunctionalSuccess:
  strict 9/9 + source/AUC/tail/LineC/efficiency/control/provenance 全过。

S4d-FMSSpecificExplorationPositive:
  difference-in-differences endpoint >=6/9，generic controls 不能解释，但 S5 未过。

R1-DiagnosticMetricInflation:
  旧 diagnostic 过，strict 不过。

R2-ContinuousMomentumSuppressionNotFMSEvent:
  every-step reset 过，event/post-event semantics 不过。

R3-GenericOptimizerStateResetConfound:
  generic optimizer reset/momentum suppression controls 解释 gain。

R4-FMSDirectionControlEquivalent:
  random/matched direction + same state semantics 解释 gain。

R5-HarmlessNullNoValue:
  不伤但 source/control gap 不成立。

R6-FMSValueButOverheadBlocked:
  FMS-specific value 成立，但 step_time/memory 不过。

R7-AffectedMaskSemanticNoGo:
  affected mask intrinsically broad 或 sparse mask destroys value。

R8-NonRATSubstrateStillMissing:
  Rational 仍唯一 substrate；Non-RAT 不允许 official FMS。

R0-ActionSearchOrAuditLeakViolation:
  任意违反禁止项。
```

---

# 16. Codex failure 后允许做什么

如果 Line F no-go：

```text
不要继续 reset/action 小修。
必须输出 no-go boundary，说明是 generic confound、direction-equivalent、harmless-null、overhead、还是 affected-mask语义失败。
```

如果 generic controls 解释 gain：

```text
停止 optimizer-state reset route。
回到 FMS definition：重新定义 FMS value，不允许继续 reset。
```

如果 FMS-specific value 成立但 overhead fail：

```text
只允许做 implementation optimization，不允许改 FMS mechanism。
必须先拆 overhead components，再做 kernel-level optimization。
```

如果 Non-RAT substrate fail：

```text
只允许 substrate repair。
不允许 Non-RAT FMS proof。
不允许拼接不同配置局部 positive。
```

---

# 17. 本计划的核心结论

v14.9 不再问：

```text
zero reset 怎么调才过？
controller 怎么选 action？
affected mask topK 怎么选？
K-RT / K-AUC / K-FL 再加哪个？
```

v14.9 只问：

$$
\boxed{
\text{FMS 是否存在不被 generic optimizer-state dynamics 解释的 causal value？}
}
$$

如果答案是否定的，那么 optimizer-state reset route 关闭；functional update 不能继续靠 reset/action 小修拖延。如果答案是肯定的，再进入 overhead 与 official S5 closure。

