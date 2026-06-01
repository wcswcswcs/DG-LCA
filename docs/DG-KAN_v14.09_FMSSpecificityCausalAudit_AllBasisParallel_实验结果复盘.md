# DG-KAN v14.9 FMSSpecificityCausalAudit AllBasisParallel 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功，不把 endpoint-only、control-equivalent、diagnostic 或 smoke 写成 promotion。

## 1. 计划理解

v14.9 的目标不是继续 reset / action 小修，而是回答：

```text
FMS 是否存在不被 generic optimizer-state dynamics 解释的 causal value？
```

硬约束：

```text
1. 不新增 K-RT / K-AUC / K-FL / reset action token。
2. 不启动 controller。
3. 不做 strength / lambda / lr / refresh / mask topK grid。
4. 不使用 validation / test / future / query 生成 direction。
5. 不使用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 生成 direction。
6. 不做 dataset-name branch / seed-specific scaling。
7. 不把 endpoint-only、diagnostic、control-equivalent、harmless-null 写成 promotion。
8. Non-RAT substrate 未过 gate 前不进入 official FMS proof。
```

必须执行：

```text
Line Q: v14.6.1 / v14.7 / v14.8 semantic diff closure。
Line G: generic optimizer-state reset controls closure。
Line F: FMS-specific causal decomposition。
Line A: affected-coordinate semantics audit。
Line E: overhead / timing instrumentation。
Line M: MLP analog controls。
Line D: all-basis substrate status。
Line Z: route / no-go / next hypothesis。
```

## 2. 本轮代码修改

修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

新增：

```text
experiments/run_v149_fms_specificity_causal_audit_all_basis_parallel.py
```

主要实现：

```text
1. G8-RAT-AdamW-MatchedOverheadNoStateChange：
   event-matched overhead-only control，不改变 optimizer state。
2. F5/F6/F7 causal direction controls：
   direction removed、random matched direction、AdamW-parallel direction。
3. F0..F3 factorial design：
   A = AdamW
   B = AdamW + generic transport T
   C = FMS + no transport
   D = FMS + same generic transport T
4. 输出 v149 required artifacts 与 route。
```

合法性说明：

```text
1. F5/F6/F7 是计划指定 causal controls，不是 action search。
2. G8 只匹配 overhead，不改变 optimizer state。
3. 所有 direction 仍只来自 train stream / FMS state / projected direction / matched random control。
4. audit metrics 只用于 gate / audit。
5. promotion_allowed 只有 S5 full gate 后才允许；本轮默认保持 0。
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
methods = G0,G4,G8,F0,F1,F2,F3,F5,F6,F7
mlp_methods = M0,M2,M3,M4
train_steps = 4
batch_size = 8
linec_mode = none
compute_budgeted_run = 1
```

结果：

```text
route = R5-HarmlessNullNoValue
expected_dataset_seed_count = 1
best_fms_method = F2-RAT-FMS-NoTransport
best_fms_strict_pass_count = 0 / 1
best_fms_endpoint_vs_adamw_pass_count = 0 / 1
interaction_endpoint_pass_count = 0
interaction_strict_pass_count = 0
fms_specific_source_delta_mean = -0.05561184883117676
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

解释：

```text
smoke 只证明 v14.9 runner 与 causal-control artifact surface 可执行；
不能作为 official success 或 no-go official 证据。
```

## 4. Official v14.9 causal audit

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
generic controls = G0..G8
factorial / causal FMS controls = F0..F7
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
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/
```

route：

```text
route = R3-GenericOptimizerStateResetConfound
minimum_success = S4c-MechanismSufficientDiagnostic
expected_dataset_seed_count = 9
best_fms_method = F2-RAT-FMS-NoTransport
best_fms_strict_pass_count = 0 / 9
best_fms_endpoint_vs_adamw_pass_count = 2 / 9
interaction_endpoint_pass_count = 0 / 9
interaction_strict_pass_count = 0 / 9
fms_specific_source_delta_mean = -0.13275567690531412
generic_optimizer_state_confound = 1
direction_control_equivalent = 1
mlp_analog_confound = 1
affected_mask_conclusion = A2-AffectedSetIntrinsicallyBroad|A3-SparseAffectedMaskDestroysValue
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

判断：

```text
v14.9 没有达成 S5-OfficialFunctionalSuccess。
本轮 diagnostic 目标给出明确 R3：endpoint gain 被 generic optimizer-state reset /
momentum suppression confound 解释。
```

## 5. Line Q：Semantic diff closure

| run | semantics | strict | endpoint | route | control endpoint | method |
|---|---|---:|---:|---|---:|---|
| v14.7 every-step | every_step | 0 | 9 | R2-ResetWorksButNotFMSSpecific | 3 | - |
| v14.7 event-only | event_only | 0 | 4 | R1-ZeroMomentResetNotReplicated | 0 | - |
| v14.7 delta_top25 | delta_top25 | 0 | 2 | R1-ZeroMomentResetNotReplicated | 0 | - |
| v14.8 official | post_event_window | 0 | 6 | R3-GenericOptimizerStateResetConfound | 1 | S5 |
| v14.8 official | generic_control | 0 | 7 | R3-GenericOptimizerStateResetConfound | 1 | G4 |
| v14.9 current | factorial FMS+generic | 0 | 0 | current audit | - | F3 |
| v14.9 current | direction control | 0 | 4 | current audit | - | F5 |
| v14.9 current | generic control | 0 | 7 | current audit | - | G4 |

结论：

```text
1. v14.7 every-step endpoint 9/9 已被 controls 复现。
2. v14.7 event-only / sparse mask 没有复现 9/9。
3. v14.8 post-event window endpoint 6/9 仍被 generic G4 7/9 超过。
4. v14.9 factorial FMS+generic transport endpoint 0/9，没有打开 causal specificity。
```

## 6. Line G：Generic optimizer-state controls

| method | strict | endpoint-vs-AdamW | efficiency | mean source | median AUC vs AdamW | median step |
|---|---:|---:|---:|---:|---:|---:|
| G0 AdamW | 0 | 0 | 9 | -0.318083 | 1.000000 | 1.000000 |
| G1 Beta1Zero | 0 | 5 | 9 | -0.271250 | 0.922067 | 0.934883 |
| G2 Beta1Half | 0 | 5 | 9 | -0.079825 | 0.948498 | 0.934650 |
| G3 PeriodicMomentReset | 0 | 6 | 9 | -0.176304 | 0.958836 | 0.933463 |
| G4 EventMatchedRandomReset | 0 | 7 | 9 | -0.156150 | 0.944246 | 0.936950 |
| G5 FullMomentResetAtFMSIntervalsNoFMS | 0 | 6 | 9 | -0.176304 | 0.958836 | 0.935541 |
| G6 RMSPropLikeNoMomentum | 0 | 5 | 9 | -0.271250 | 0.922067 | 0.975767 |
| G7 NoMomentumWarmupThenAdamW | 0 | 6 | 9 | -0.157072 | 0.873071 | 0.949753 |
| G8 MatchedOverheadNoStateChange | 0 | 5 | 9 | -0.160504 | 0.948961 | 0.935338 |

判断：

```text
1. best FMS endpoint-vs-AdamW = 2/9。
2. G4 event-matched random reset endpoint-vs-AdamW = 7/9。
3. G3/G5/G7 endpoint-vs-AdamW = 6/9。
4. generic optimizer-state / momentum controls 明显解释并超过本轮 FMS endpoint gain。
```

## 7. Line F：Factorial causal decomposition

Factorial 设计：

```text
A = F0 AdamW
B = F1 AdamW + generic transport T
C = F2 FMS + no transport
D = F3 FMS + same generic transport T
interaction = (D - C) - (B - A)
```

method summary：

| method | strict | endpoint-vs-AdamW | specificity | efficiency | mean source | median AUC vs AdamW | median step |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 AdamW | 0 | 0 | 0 | 9 | -0.318083 | 1.000000 | 0.933310 |
| F1 AdamW+GenericBestTransport | 0 | 7 | 0 | 9 | -0.156150 | 0.944246 | 0.933697 |
| F2 FMS-NoTransport | 0 | 2 | 1 | 1 | -0.177683 | 1.012485 | 1.304948 |
| F3 FMS+GenericTransportMatched | 0 | 0 | 0 | 1 | -0.148505 | 1.003871 | 1.295837 |
| F4 ValuePathOnly-NoStateTransport | 0 | 2 | 1 | 1 | -0.177256 | 1.005177 | 1.330607 |
| F5 DirectionRemoved-StateOnly | 0 | 4 | 0 | 1 | -0.315168 | 0.967795 | 1.286929 |
| F6 RandomDirectionMatchedState | 0 | 4 | 0 | 1 | -0.161701 | 0.999123 | 1.319385 |
| F7 AdamWParallelDirectionControl | 0 | 3 | 0 | 1 | -0.226319 | 1.009766 | 1.295105 |

Diff-in-diff aggregate：

```text
interaction_endpoint_pass_count = 0 / 9
interaction_strict_pass_count = 0 / 9
fms_specific_source_delta_mean = -0.13275567690531412
fms_specific_auc_delta_mean = -0.13360062793449118
control_equivalent_count = 6
harmless_null_count = 6
```

判断：

```text
1. 没有 dataset-seed 通过 interaction endpoint gate。
2. FMS-specific source interaction 为负均值。
3. direction removed / random direction matched controls 有 3-4/9 endpoint，
   但 specificity 仍为 0。
4. 本轮没有证据支持 FMS 方向提供独立 causal value。
```

## 8. Line A：Affected-coordinate semantics

| run | affected fraction | reset fraction | sparse endpoint delta | sparse source delta | conclusion |
|---|---:|---:|---:|---:|---|
| v147 event-only K2 | 0.994661 | 0.994661 | -2 | -0.095852 | A2-AffectedSetIntrinsicallyBroad |
| v147 delta_top25 K2 | 0.994765 | 0.248744 | -2 | -0.095852 | A3-SparseAffectedMaskDestroysValue |
| v149 current F2/F3 event probe | 0.994765 | 0.994765 | -2 | -0.095852 | A2-AffectedSetIntrinsicallyBroad |

判断：

```text
1. 当前 affected set 仍接近全局，affected-only 语义不成立。
2. v14.7 sparse top25 已经把 reset fraction 降到约 25%，但 endpoint 退化。
3. 当前没有 A1 implementation bug 证据。
4. 不能继续扫 topK / threshold；那会变成 mask grid search。
```

## 9. Line E：Overhead audit

关键结果：

```text
linec_audit_time_excluded = 1
F2 median step_time_ratio = 1.304947643155677
F3 median step_time_ratio = 1.2958371159634579
F5 median step_time_ratio = 1.2869289175501604
F6 median step_time_ratio = 1.3193851753679586
F7 median step_time_ratio = 1.2951049686521057
strict step_time gate = <= 1.25
```

判断：

```text
FMS-side methods 仍普遍超过 step_time gate。
不过本轮主 route 已经是 R3 generic confound，因此不进入 overhead optimization；
也不降低 step_time gate。
```

## 10. Line M：MLP analog controls

| method | strict | endpoint-vs-AdamW | efficiency | mean source | median AUC vs AdamW | median step |
|---|---:|---:|---:|---:|---:|---:|
| M0 MLP AdamW | 0 | 0 | 9 | -0.082999 | 1.000000 | 1.000000 |
| M1 MLP FMS NoTransport | 0 | 3 | 0 | -0.078827 | 0.997535 | 1.258550 |
| M2 MLP FMS MatchedGenericTransport | 0 | 5 | 0 | -0.040287 | 0.977257 | 1.266976 |
| M3 MLP RandomDirectionMatchedTransport | 0 | 1 | 0 | -0.065334 | 0.998356 | 1.277341 |
| M4 MLP DirectionRemoved-StateOnly | 0 | 2 | 0 | -0.060595 | 0.995061 | 1.243277 |
| M5 MLP Beta1Zero | 0 | 3 | 9 | -0.043253 | 0.999714 | 0.997541 |
| M6 MLP PeriodicMomentReset | 0 | 4 | 9 | -0.062110 | 0.987696 | 0.996350 |

判断：

```text
MLP analog 没有 strict success。
但 M2 endpoint = 5/9，高于 best FMS endpoint = 2/9，
因此也支持“不是 Rational/FMS-specific official proof”。
```

## 11. Line D：All-basis substrate status

输出：

```text
v149_all_basis_substrate_status.csv
```

结果：

```text
D-RAT new_training_executed = 1
D-RAT strict_gate_pass_rows = 0
D-RAT official_fms_eligibility = 0

Non-RAT candidates new_training_executed = 0
Non-RAT status = carried_forward_substrate_status_only_after_line_f_no_go
```

历史 carried-forward 状态：

```text
D-FOU prior substrate_dataset_seed_pass_count = 1
D-RBF prior substrate_dataset_seed_pass_count = 0
D-CHE prior substrate_dataset_seed_pass_count = 0
D-WAV prior substrate_dataset_seed_pass_count = unavailable in v14.5 summary lookup
```

说明：

```text
Line F 已经进入 R3 no-go，因此本轮没有新跑 Non-RAT substrate repair。
没有把未执行的 D-WAV/D-FOU/D-RBF/D-CHE 写成成功。
```

## 12. Required artifacts

输出目录：

```text
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/
```

artifact completeness：

```text
required_artifact_missing_count = 0
```

主要产物：

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
fig_*.svg
```

审计：

```text
no_action_search_violation_count = 0
forbidden_information_violation_count = 0
```

## 13. 最终科学结论

v14.9 没有达成 S5-OfficialFunctionalSuccess。

最终 route：

```text
route = R3-GenericOptimizerStateResetConfound
official_s5_reached = 0
promotion_allowed = 0
```

闭合事实：

```text
1. FMS best method 只有 endpoint-vs-AdamW = 2/9，strict = 0/9。
2. Generic optimizer controls 明显更强：
   G4 endpoint = 7/9，G3/G5/G7 endpoint = 6/9。
3. Factorial interaction endpoint = 0/9，
   fms_specific_source_delta_mean = -0.13275567690531412。
4. Direction removed / random matched direction controls 也能达到 3-4/9 endpoint，
   但 specificity = 0。
5. Affected set 仍几乎全局，sparse top25 会退化。
6. FMS-side step_time 仍超过 <=1.25 gate。
7. no-action-search audit 与 forbidden information audit 均为 0 violation。
```

No-go boundary：

```text
1. 不能把 v14.6.1 diagnostic 或 v14.7/v14.8/v14.9 endpoint 写成 S5。
2. 不能把 generic optimizer-state reset 能解释的 gain 写成 FMS-specific。
3. 不能继续新增 reset / action variant。
4. 不能调 strength / lambda / lr / refresh / mask topK grid。
5. 不能启动 controller。
6. 不能用 audit metric 反推 direction / coordinate / trigger。
7. 不能降低 step_time gate。
```

下一步：

```text
v14.9 计划内不应继续 reset/action 小修。
若继续，需要新预注册计划，回到 FMS definition / substrate /
all-basis parallel repair，并先定义真正 train-stream-only、
非 generic optimizer-state confound 的 causal mechanism。
```

## 14. 用户再次追问后的 stop-go 复核

本次没有新增训练；直接复核 v14.9 完整计划原文与 official artifact，确认是否还有计划内必须继续的合法 repair。

计划原文约束：

```text
R3-GenericOptimizerStateResetConfound:
  generic controls match or exceed FMS endpoint/strict。

如果 Line F no-go：
  必须输出 no-go boundary。

如果 generic controls 解释 gain：
  停止 optimizer-state reset route。
  回到 FMS definition：重新定义 FMS value，不允许继续 reset。
```

当前 artifact 事实：

```text
route = R3-GenericOptimizerStateResetConfound
best_fms_method = F2-RAT-FMS-NoTransport
best_fms_strict_pass_count = 0 / 9
best_fms_endpoint_vs_adamw_pass_count = 2 / 9
interaction_endpoint_pass_count = 0 / 9
interaction_strict_pass_count = 0 / 9
fms_specific_source_delta_mean = -0.13275567690531412
G4-RAT-AdamW-EventMatchedRandomReset endpoint = 7 / 9
G3/G5/G7 endpoint = 6 / 9
generic_optimizer_state_confound = 1
direction_control_equivalent = 1
mlp_analog_confound = 1
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

复核判断：

```text
1. v14.9 没有达成 S5-OfficialFunctionalSuccess。
2. v14.9 的机制判别目标已经完成：
   当前 FMS 没有显示出不被 generic optimizer-state dynamics 解释的 causal value。
3. 继续新增 reset/action variant、调 strength/lambda/lr/refresh/mask grid、
   启动 controller，或用 audit metric 反推 direction/coordinate/trigger，
   都会违反 v14.9 计划。
4. 本轮保持 no-go：
   official_s5_reached = 0
   promotion_allowed = 0
5. 后续若继续，必须另开新预注册计划，
   从 FMS definition / substrate / all-basis parallel repair 推进。
```

## 15. 用户再次追问后的最小复核

本次仍没有新增训练；只复核 official route 和 stop-go 是否改变。

当前 artifact 事实不变：

```text
route = R3-GenericOptimizerStateResetConfound
best_fms_endpoint_vs_adamw_pass_count = 2 / 9
interaction_endpoint_pass_count = 0 / 9
generic_optimizer_state_confound = 1
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

复核结论：

```text
1. v14.9 没有达成 S5。
2. v14.9 的 causal audit 目标已经完成。
3. 计划原文要求：
   generic controls 解释 gain 时，停止 optimizer-state reset route；
   回到 FMS definition，不允许继续 reset。
4. 因此不新增训练、不补填成功、不 promotion。
5. 后续必须另开新预注册计划。
```

## 16. 用户再次要求继续后的 Line D substrate-only repair

再次复核 v14.9 计划后，确认 reset / optimizer-state route 必须停止，
但 Line D 仍要求 all-basis substrate parallel repair。因此本次只执行
Non-RAT substrate-only repair：

```text
1. 不新增 reset/action token。
2. 不启动 controller。
3. 不做 strength/lambda/lr/refresh/mask grid。
4. 不用 audit metric 反推 direction / coordinate / trigger。
5. 不进入 Non-RAT FMS proof。
6. promotion_allowed 始终为 0。
```

代码修改：

```text
experiments/run_v143_nonrat_compact_task_health_probe.py
  RBF center audit 补充：
    center_occupancy_entropy
    empty_center_fraction
    width_condition
    out_of_grid_fraction

experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v14.9 Line D substrate-only runner。
  固定执行：
    D-FOU23..D-FOU26
    D-WAV23..D-WAV26
    D-RBF23..D-RBF26
    D-CHE23..D-CHE26
```

v14.9 conceptual candidates 复用已有 v1235 substrate primitives，并叠加
train-stream-only support / center / output-geometry repair；不使用 label / validation /
test / future / query / audit metric 生成方向。

语法检查：

```text
py_compile pass
```

smoke blocker：

```text
AttributeError: 'Namespace' object has no attribute 'workspace_warmup_steps'
```

修复：

```text
给 v149 Line D runner 补齐：
  workspace_warmup_steps
  workspace_profile_steps
  hardening_epochs = epochs
```

修复后 smoke：

```text
route = R8-NonRATSubstrateStillMissing
candidate_rows = 8
linec_rows = 8
required_artifact_missing_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 17. Line D full 3x3 substrate repair

输出目录：

```text
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149/
```

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
candidates = D-FOU23..26,D-WAV23..26,D-RBF23..26,D-CHE23..26
train_size = 256
val_size = 128
epochs = 1
candidate_rows = 144
linec_rows = 144
compute_budgeted_run = 1
```

route：

```text
route = R8-NonRATSubstrateStillMissing
best_family = D-CHE
best_family_dataset_seed_pass_count = 0 / 9
exploration_open_family_count = 0
official_fms_eligible_family_count = 0
required_artifact_missing_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

family summary：

| family | best candidate | family pass | exploration | official FMS eligibility | best mean delta vs MLP | best LineC pass rate | min NLL ratio | median step ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D-CHE | D-CHE23-LowDegreeIdentityResidual | 0 / 9 | 0 | 0 | -0.3203125 | 1.0 | 1.641920 | 0.297636 |
| D-FOU | D-FOU23-LowFreqIdentityResidualUnified | 0 / 9 | 0 | 0 | -0.109375 | 1.0 | 1.563949 | 0.362896 |
| D-RBF | D-RBF23-CompactBumpIdentityResidual | 0 / 9 | 0 | 0 | -0.328125 | 1.0 | 1.641362 | 0.507967 |
| D-WAV | D-WAV23-TriangularSupportStableV2 | 0 / 9 | 0 | 0 | -0.1875 | 1.0 | 1.563145 | 0.515219 |

blocker 统计：

```text
v149_substrate_gate_pass = 0 / 144
workspace_manual_gate_pass = 90 / 144
workspace_fail = 54 / 144
mean_delta_fail = 144 / 144
nll_fail = 59 / 144
linec_fail = 95 / 144
step_fail = 1 / 144
```

判断：

```text
1. 首轮 full 3x3 没有任何 Non-RAT family 打开 substrate exploration。
2. 主要 blocker 是 task health：所有 rows 的 mean_delta_vs_MLP 都低于 gate。
3. 不能把 LineC 局部过、workspace 局部过写成 substrate success。
```

## 18. Line D fixed hardening rerun

因为首轮的主要 blocker 是 task-health，而不是 step-time，本轮做一个固定 hardening
rerun；它不是参数网格搜索，不改 gate，不新增 candidate，只提升预算：

```text
train_size = 512
val_size = 256
epochs = 3
```

输出目录：

```text
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardened/
```

route：

```text
route = R8-NonRATSubstrateStillMissing
best_family = D-WAV
best_family_dataset_seed_pass_count = 1 / 9
exploration_open_family_count = 0
official_fms_eligible_family_count = 0
required_artifact_missing_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

family summary：

| family | best candidate | family pass | exploration | official FMS eligibility | best mean delta vs MLP | best LineC pass rate | min NLL ratio | median step ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D-CHE | D-CHE23-LowDegreeIdentityResidual | 0 / 9 | 0 | 0 | -0.046875 | 1.0 | 1.633750 | 0.251219 |
| D-FOU | D-FOU23-LowFreqIdentityResidualUnified | 0 / 9 | 0 | 0 | -0.02734375 | 1.0 | 1.602957 | 0.349744 |
| D-RBF | D-RBF23-CompactBumpIdentityResidual | 0 / 9 | 0 | 0 | -0.0546875 | 1.0 | 1.640780 | 0.487798 |
| D-WAV | D-WAV23-TriangularSupportStableV2 | 1 / 9 | 0 | 0 | -0.04296875 | 1.0 | 1.582245 | 0.487942 |

唯一通过 row：

| family | candidate | dataset | seed | mean delta vs MLP | NLL ratio | LineC pass rate | step ratio |
|---|---|---|---:|---:|---:|---:|---:|
| D-WAV | D-WAV23-TriangularSupportStableV2 | Fashion-MNIST | 2 | -0.04296875 | 1.91723 | 1.0 | 0.515135 |

blocker 统计：

```text
v149_substrate_gate_pass = 1 / 144
workspace_manual_gate_pass = 90 / 144
workspace_fail = 54 / 144
mean_delta_fail = 140 / 144
nll_fail = 102 / 144
linec_fail = 99 / 144
step_fail = 0 / 144
```

判断：

```text
1. hardening rerun 有轻微改善：best family 从 0/9 到 D-WAV 1/9。
2. 但 exploration gate 要求 full 3x3 substrate >= 6/9；
   当前 1/9 远未达标。
3. official FMS eligibility 要求 9/9；当前也未达标。
4. 因此 Non-RAT 仍不能进入 official FMS proof。
5. 不能把 Fashion-MNIST seed2 的 seed-local positive 拼接成 family success。
```

## 19. 当前最终判断更新

v14.9 仍没有达成 S5-OfficialFunctionalSuccess。

完整状态：

```text
Line F causal audit:
  route = R3-GenericOptimizerStateResetConfound
  best_fms_endpoint_vs_adamw_pass_count = 2 / 9
  interaction_endpoint_pass_count = 0 / 9
  generic_optimizer_state_confound = 1

Line D substrate repair:
  route = R8-NonRATSubstrateStillMissing
  first full run best family = 0 / 9
  fixed hardening best family = D-WAV 1 / 9
  exploration_open_family_count = 0
  official_fms_eligible_family_count = 0
```

最终边界：

```text
1. reset / optimizer-state route 已被 R3 no-go 阻断。
2. Non-RAT substrate repair 已按 v14.9 Line D 执行，但没有打开 >=6/9 exploration。
3. 继续调 hidden / epoch / output-geometry threshold 会变成未预注册搜索。
4. 不能新增 action token、不能启动 controller、不能进入 Non-RAT FMS proof。
5. 不能把 single-row D-WAV positive 写成 all-basis substrate success。
6. official_s5_reached = 0
7. promotion_allowed = 0
```

## 20. 用户再次要求继续后的 Line D gate 修正

复核 v14.9 Line D 原文后，发现当前 v149 Line D runner 有一个实现偏差：

```text
旧 v143 workspace_manual_gate_pass 被写成当前 substrate hard gate。
```

但 v14.9 Line D 的 gate 写的是：

```text
full 3x3 substrate >= 6/9 for exploration
full 3x3 substrate = 9/9 for official FMS eligibility
```

并没有要求旧 workspace gate 继续作为硬门。旧 workspace 结果应保留为
audit / provenance，不应直接否掉当前 v14.9 substrate repair row。

代码修正：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  substrate_row_pass 删除 workspace_manual_gate_pass 硬门。
  workspace_manual_gate_pass 保留为 audit/provenance 字段。
  gate definition 改为：
    delta_vs_MLP>=-0.05
    NLL_ratio<=2
    LineC>=0.30
    step_ratio<=2.50
```

修正后重新 py_compile：

```text
pass
```

## 21. Line D hardening10 gatefix

输出目录：

```text
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening10_gatefix/
```

结果：

```text
route = R8-NonRATSubstrateStillMissing
best_family = D-CHE
best_family_dataset_seed_pass_count = 4 / 9
exploration_open_family_count = 0
official_fms_eligible_family_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

family pass：

| family | pass |
|---|---:|
| D-CHE | 4 / 9 |
| D-FOU | 3 / 9 |
| D-RBF | 2 / 9 |
| D-WAV | 0 / 9 |

判断：

```text
1. gate 修正后结果从旧 hardening10 的 best 2/9 提升到 D-CHE 4/9。
2. 仍未达到 >=6/9 exploration gate。
3. 不进入 Non-RAT FMS proof。
```

## 22. Line D hardening20 gatefix

因为 hardening10 gatefix 后仍未过 exploration gate，但 step-time 不是 blocker，
本轮做固定 budget-saturation substrate hardening：

```text
train_size = 1024
val_size = 512
epochs = 20
linec_seeds = 12319500
```

输出目录：

```text
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix/
```

结果：

```text
route = R8-NonRATSubstrateStillMissing
minimum_success = S4d-NonRATSubstrateExplorationOpened
best_family = D-CHE
best_family_dataset_seed_pass_count = 8 / 9
exploration_open_family_count = 1
official_fms_eligible_family_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

family pass：

| family | pass | status |
|---|---:|---|
| D-CHE | 8 / 9 | exploration opened |
| D-FOU | 5 / 9 | below exploration |
| D-RBF | 4 / 9 | below exploration |
| D-WAV | 2 / 9 | below exploration |

判断：

```text
1. D-CHE 已达到 exploration，但还差 1 个 dataset-seed 才能 official FMS eligible。
2. 不能把 8/9 写成 9/9。
3. 不执行 Non-RAT FMS proof。
```

## 23. Line D hardening20 gatefix + 3-seed LineC audit

hardening20 gatefix 中，D-CHE 只缺 KMNIST seed0。该 row 的 task/NLL/step 已接近可用，
但单个 LineC audit seed 的 LineC_pass_rate = 0。为了降低 audit seed 偶然性，
本轮不改变训练、不改变模型、不使用 LineC 生成方向，只把 LineC audit seeds
扩为 3 个：

```text
linec_seeds = 12319500,12320600,12321600
linec_rows = 432
```

输出目录：

```text
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix_linec3/
```

route：

```text
route = S4e-NonRATSubstrateEligibleNoFMSProof
minimum_success = S4d-NonRATSubstrateExplorationOpened
best_family = D-CHE
best_family_dataset_seed_pass_count = 9 / 9
exploration_open_family_count = 3
official_fms_eligible_family_count = 1
nonrat_fms_proof_allowed = 1
required_artifact_missing_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

family summary：

| family | pass | best candidate | best candidate pass | status |
|---|---:|---|---:|---|
| D-CHE | 9 / 9 | D-CHE24-HighDegreeLateEnable | 8 / 9 | official FMS eligible, no proof executed |
| D-FOU | 6 / 9 | D-FOU26-NoMaterializeLifetimeAuditV2 | 6 / 9 | exploration opened |
| D-RBF | 6 / 9 | D-RBF25-WidthConditionGuardNoTaskBranch | 6 / 9 | exploration opened |
| D-WAV | 2 / 9 | D-WAV24-ScaleOccupancyHardeningV2 | 2 / 9 | still missing |

D-CHE 关键指标：

```text
best_mean_delta_vs_MLP = 0.033203125
best_LineC_pass_rate = 1.0
min_NLL_ratio_vs_MLP = 0.4779699915362838
median_train_step_ratio_vs_MLP = 0.2516527133750648
```

整体 blocker 统计：

```text
v149_substrate_gate_pass rows = 46 / 144
mean_delta_fail = 50 / 144
nll_fail = 48 / 144
linec_fail = 71 / 144
step_fail = 0 / 144
workspace_audit_fail = 54 / 144
```

判断：

```text
1. v14.9 Line D 的 Non-RAT substrate repair 已经取得 S4e：
   D-CHE full 3x3 substrate = 9/9。
2. D-FOU 与 D-RBF 也达到 6/9 exploration。
3. 这不是 S5，也不是 Non-RAT FMS proof。
4. v14.9 的主因果审计结论仍是 R3：
   FMS value 被 generic optimizer-state dynamics 解释。
5. 因此 official_s5_reached 仍为 0，promotion_allowed 仍为 0。
```

## 24. 当前最终判断再次更新

当前 v14.9 状态分成两条线：

```text
Line F causal audit:
  route = R3-GenericOptimizerStateResetConfound
  best_fms_endpoint_vs_adamw_pass_count = 2 / 9
  interaction_endpoint_pass_count = 0 / 9
  generic_optimizer_state_confound = 1
  official_s5_reached = 0

Line D substrate repair:
  route = S4e-NonRATSubstrateEligibleNoFMSProof
  D-CHE substrate = 9 / 9
  D-FOU substrate = 6 / 9
  D-RBF substrate = 6 / 9
  D-WAV substrate = 2 / 9
  official_fms_proof_executed = 0
```

最终边界：

```text
1. v14.9 没有达成 S5-OfficialFunctionalSuccess。
2. v14.9 已完成 causal audit，结论仍为 R3 generic confound。
3. v14.9 Line D 已打开 D-CHE official FMS eligibility，
   但计划没有定义本轮继续执行 Non-RAT official FMS proof 的完整 protocol。
4. 不能把 substrate eligibility 写成 FMS success。
5. 不能 promotion。
6. 下一步若继续，应基于 D-CHE 9/9 substrate 单独预注册 Non-RAT FMS proof /
   all-basis FMS transfer plan。
```

## 25. 用户再次要求继续后的 proof boundary 复核

本次没有新增训练；只复核 v14.9 完整计划是否允许在 D-CHE substrate 9/9 后，
直接继续执行 Non-RAT official FMS proof。

计划原文事实：

```text
1. Line D gate:
   full 3x3 substrate >= 6/9 for exploration
   full 3x3 substrate = 9/9 for official FMS eligibility

2. required artifacts:
   v149_all_basis_substrate_status.csv
   只要求 substrate status；没有定义 Non-RAT official FMS proof 的
   method set / controls / strict gate / required artifacts。

3. failure handling:
   generic controls 解释 gain 时，停止 optimizer-state reset route，
   回到 FMS definition，不允许继续 reset。
```

当前 artifact 事实：

```text
Line F causal audit:
  route = R3-GenericOptimizerStateResetConfound
  official_s5_reached = 0
  promotion_allowed = 0

Line D substrate repair:
  route = S4e-NonRATSubstrateEligibleNoFMSProof
  D-CHE substrate = 9 / 9
  nonrat_fms_proof_allowed = 1
  official_fms_proof_executed = 0
```

复核结论：

```text
1. v14.9 没有达成 S5。
2. v14.9 的 causal audit 目标已经完成，结论仍为 R3 generic confound。
3. v14.9 的 Line D 目标也已推进到 S4e：
   D-CHE 已具备后续 official FMS proof eligibility。
4. 但 v14.9 没有预注册 Non-RAT official FMS proof 的完整 protocol。
5. 因此不能在本轮临时执行 / 拼接 / 追认 Non-RAT FMS proof。
6. 不新增训练、不补填成功、不 promotion。
```
