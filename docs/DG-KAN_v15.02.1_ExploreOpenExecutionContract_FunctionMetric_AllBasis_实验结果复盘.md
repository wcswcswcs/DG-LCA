# DG-KAN v15.02.1 ExploreOpenExecutionContract FunctionMetric AllBasis 实验结果复盘

生成时间：2026-05-31（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 function-metric fallback、proximal diagnostic、substrate replay 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v15.02.1 的目标是在 Explore-Open contract 下执行 Line R/G/P/X/D/M/C/Z，不允许因单线失败早停，并要求 fallback ladder 与 exhaustion certificate 完整落盘。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v15.02.1 预注册 substrate-only candidates:
  D-FOU47..51, D-RBF45..49, D-WAV41..44。
```

过程说明：Line G/P/M 为实际 low-budget real-data training；Line D 为 v149 substrate-only actual reconfirmation；Line X/C 为 audit only。

## 3. Line G Function-Metric 结果

```text
line_g_candidate_count = 6
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = -0.07151905742902605
control_equivalent_fraction = 1.0
bad_event_fraction = 1.0
line_g_exploration_gate_pass = 0
line_g_meaningful_gate_pass = 0
line_g_s4_gate_pass = 0
best_g_method = G8-D-CHE-CombinedAdamV_FunctionMetric_Alignment
```

Method summary：

| method | rows | strict pass | dataset-seed pass | mean source vs best control | mean AUCtime ratio |
|---|---:|---:|---:|---:|---:|
| G3-D-CHE-AdamVFunctionMetric | 9 | 0 | 0 | -0.9870692425303988 | 1.2755210583337682 |
| G4-D-CHE-DegreeRoleFunctionMetric | 9 | 0 | 0 | -0.9870454006724887 | 1.2755140115270658 |
| G5-D-CHE-OutputJacobianDiagMetric | 9 | 0 | 0 | -0.9870806866221957 | 1.2755245173752257 |
| G6-D-CHE-GradientSNRMetric | 126 | 0 | 0 | -0.07179423788237194 | 1.0199413573201734 |
| G7-D-CHE-CurvatureClippedFunctionMetric | 9 | 0 | 0 | -0.9870868060323927 | 1.2755164207469958 |
| G8-D-CHE-CombinedAdamV_FunctionMetric_Alignment | 126 | 0 | 0 | -0.07151905742902605 | 1.0199079495433119 |

## 4. Line P Function-Space Proximal 结果

```text
line_p_candidate_count = 5
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = -0.03708608945210775
control_equivalent_fraction = 0.6666666666666666
bad_event_fraction = 1.0
line_p_exploration_gate_pass = 0
best_p_method = P4-D-CHE-RandomSketchFunctionProx
```

## 5. Line X transfer operator audit

```text
line_x_rows = 10
positive_looking_rows = 10
x_transfer_supported_count = 0
x_failure_class_rows = 10
x_local_positive_no_transfer_rows = 10
line_x_route = R-X-LocalPositiveNoTransfer
```

## 6. Line D all-basis substrate 结果

```text
line_d_source = v1521_actual_v149_substrate_acceleration
line_d_rows = 126
best_non_dche_family = D-FOU
best_non_dche_dataset_seed_pass_count = 0 / 9
line_d_official_fms_eligible_family_count = 0
line_d_route = R-D-AllBasisSubstrateExhausted
```

Line D family summary：

| family | rows | pass | max mean delta vs MLP | best LineC pass rate | official eligibility |
|---|---:|---:|---:|---:|---:|
| D-FOU | 45 | 0/9 | -0.109375 | 1.0 | 0 |
| D-RBF | 45 | 0/9 | -0.328125 | 1.0 | 0 |
| D-WAV | 36 | 0/9 | -0.0234375 | 1.0 | 0 |

Line D family-specific fallback certificate：

| family | main candidates | family-specific fallback count | fallbacks executed |
|---|---:|---:|---|
| D-FOU | 5 | 5 | lineage_replay_checked,gate_mismatch_checked,budget_mismatch_checked,low_frequency_task_health_hardening_checked,phase_drift_band_occupancy_decomposition_checked |
| D-RBF | 5 | 8 | lineage_replay_checked,gate_mismatch_checked,budget_mismatch_checked,center_occupancy_collapse_audit_checked,width_condition_collapse_audit_checked,identity_residual_ablation_checked,gaussian_local_support_replay_checked,task_health_workspace_conflict_decomposition_checked |
| D-WAV | 4 | 7 | lineage_replay_checked,gate_mismatch_checked,budget_mismatch_checked,scale_occupancy_audit_checked,support_overlap_audit_checked,low_scale_only_replay_checked,local_tail_coverage_decomposition_checked |

## 7. Line M generic controls

```text
line_m_rows = 72
positive_looking_rows = 266
positive_rows_checked_by_m = 266
generic_control_explains_positive_fraction = 1.0
generic_control_explains_positive = 1
positive_row_control_map_rows = 266
positive_row_map_missing_control_rows = 0
positive_row_map_generic_explained_rows = 266
positive_row_map_matched_control_count_min = 8
positive_row_map_matched_control_count_max = 8
line_m_route = R-M-GenericOptimizerExplainsGain
```

判断：Line M 只作为 generic / MLP confound audit；不写成 KAN-specific promotion。

## 8. Rational / D-CHE no-regression monitor

```text
v1521_dche_no_regression_monitor.csv rows = 1
v1521_rational_no_regression_monitor.csv rows = 8
monitor_promotion_allowed = 0
```

判断：no-regression monitor 只用于 Line D/Rational 稳定性复核，不打开 reset route，不写成 promotion。

## 9. Line Z no-go 分类

```text
promotion_no_go active=1 reason=S5 official gate not reached
exploration_no_go active=1 reason=G/P/D exhaustion certificates complete without exploration gate
budget_deferred active=0 reason=no remaining pre-registered v15.02.1 budget-deferred branch
implementation_blocker active=0 reason=none
theoretical_no_go active=1 reason=current function-metric/proximal definition exhausted; next version needs new theory-level functional definition or substrate carrier
```

## 10. 最终 route

```text
route = R15_2_1-CurrentFunctionalDefinitionNoGo
minimum_success = S1-ExploreOpenExecutionContractExecuted
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 11. 科学结论

```text
1. v15.02.1 已执行 Line R/G/P/X/D/M/C/Z，并生成 required artifacts。
2. Line G/P fallback ladder 与 exhaustion certificate 已落盘；未达成 S2/S3/S4/S5。
3. Line X 只作为 transfer operator audit，不打开 promotion。
4. Line M controls 已检查 positive-looking rows；generic/MLP controls 只作为 confound audit。
5. Rational / D-CHE no-regression monitor 已落盘；不作为 promotion。
6. Line D 非 D-CHE basis 未达到 >=6/9 substrate exploration gate。
7. 当前 route = R15_2_1-CurrentFunctionalDefinitionNoGo，promotion_allowed = 0。
```

## 12. 用户再次追问后的覆盖修复

本次没有新增训练；复核完整计划后发现并修复 finalizer / audit / logging 层覆盖缺口：

```text
1. v1521_line_c_linec_tail_audit.csv 补齐 source_vs_control、AUCtime_ratio、LineC_fail_reason。
2. v1521_forbidden_information_audit.csv / v1521_no_action_search_audit.csv 补齐计划 5.3 的 readback 字段。
3. v1521_method_surface_manifest.csv 中 Line D candidates 的 executed_rows 从空值修正为实际 9 rows/candidate。
4. route decision 显式加入 Line M controls explain positives 条件：
   positive_rows_checked_by_m = 266
   generic_control_explains_positive = 1
   v1521_line_m_positive_row_control_map.csv rows = 266, missing_control_rows = 0, generic_explained_rows = 266
5. 以上修复复用已落盘训练 artifact，不新增 FU/F-CHE token，不新增 action/controller/reset route。
6. 再次追问后修复 gate summary：只用 effect rows 计算 G/P route 数值，P-FB1/P-FB2/P-FB3 audit rows 不再污染 gate。
7. 补齐 Rational no-regression 与 D-CHE no-regression monitor，并写入 Line Z no-go taxonomy。
8. 将 no-regression monitor 与 Line Z taxonomy 纳入 required manifest，避免完整性审计漏项。
9. 再次追问后补齐 Line M positive-looking row control map，逐行记录 best generic/MLP matched control。
10. 再次追问后补齐 Line D family-specific fallback certificate，显式记录 D-FOU/D-RBF/D-WAV 计划内 fallback 覆盖。
11. 再次追问后补齐 Line X per-row failure_class：
    X-Fail-LocalPositiveNoTransfer rows = 10
```

修复后当前合法结论仍为：

```text
route = R15_2_1-CurrentFunctionalDefinitionNoGo
minimum_success = S1-ExploreOpenExecutionContractExecuted
official_s5_reached = 0
promotion_allowed = 0
```

## 13. 用户再次追问后的 method/control surface 反方复核

本次没有新增训练；按完整计划第 6/7/9/10/15 节复核 method、control、candidate 与 fallback ladder 的实际 artifact 覆盖，确认是否仍有计划内漏跑分支。

```text
Line G main/control surface missing = 0 (G0..G8)
Line G fallback ladder missing = 0 (G-FB1..G-FB5)
Line P main/control surface missing = 0 (P0..P5)
Line P fallback ladder missing = 0 (P-FB1..P-FB4)
Line M required generic controls missing = 0 (8/8)
Line D candidates missing = 0 (D-FOU47..51, D-RBF45..49, D-WAV41..44)
```

## 14. 用户再次追问后的独立 gate 重算复核

本次没有新增训练；从 official artifacts 重算 S2/S3/S4/S5 与 no-go route 条件，确认 route JSON 没有过早 no-go 或漏 promotion。

```text
S2-FunctionMetricExplorationPositive: recomputed_pass=0 route_consistent=1 details=g_pass=0;p_pass=0;real_lite=0;source_mean=-0.03708608945210775;control_equiv=0.6666666666666666
S3-FunctionMetricMeaningful: recomputed_pass=0 route_consistent=1 details=g_meaningful=0;p_meaningful=0
S4-RealTransferExploration: recomputed_pass=0 route_consistent=1 details=g_s4=0;p_s4=0
S5-Official: recomputed_pass=0 route_consistent=1 details=g_s5=0;p_s5=0
R-G-FunctionMetricExhausted: recomputed_pass=1 route_consistent=1 details=g_exhausted=1;g_pass=0
R-P-ProximalMetricExhausted: recomputed_pass=1 route_consistent=1 details=p_exhausted=1;p_pass=0
R-X-LocalPositiveNoTransfer: recomputed_pass=1 route_consistent=1 details=x_positive_rows=10;x_supported=0
R-D-AllBasisSubstrateExhausted: recomputed_pass=1 route_consistent=1 details=best_non_dche_dataset_seed_pass_count=0;eligible_family_count=0
R-M-GenericOptimizerExplainsGain: recomputed_pass=1 route_consistent=1 details=positive_rows_checked=266;generic_control_explains_positive=1
R15_2_1-CurrentFunctionalDefinitionNoGo: recomputed_pass=1 route_consistent=1 details=route=R15_2_1-CurrentFunctionalDefinitionNoGo;S2=0;S3=0;S4=0;S5=0;G_exhausted=1;P_exhausted=1;D_count=0;M_explains=1
gate_route_inconsistent_rows = 0
```

## 15. 用户再次追问后的执行合同闭合复核

本次没有新增训练；新增 v1521_execution_contract_coverage_audit.csv，把完整计划中的探索合同义务逐项机器化核对。

```text
Line R artifact/provenance/audit: status=1 details=forbidden/no-action audit files present
Line G main plus fallback ladder: status=1 details=surface=1;fallback=1;certificate=1;exhausted=1
Line P main plus fallback ladder: status=1 details=surface=1;fallback=1;certificate=1;exhausted=1
Line X transfer/operator audit: status=1 details=rows=10;transfer_supported=0;failure_class_complete=1
Line D all-basis fresh hardening and exhaustion: status=1 details=surface=1;family_cert=1;monitors=1;best_count=0
Line M positive-looking matched controls: status=1 details=surface=1;map_rows=266;checked=266;generic_explains=1
Line C/tail audit: status=1 details=rows=1323;fail_reason_field=1
Per-fail failure taxonomy: status=1 details=G=321;P=35236;X=10;D=126;C=1323
Line Z no-go taxonomy and next queue: status=1 details=taxonomy_rows=5;route=R15_2_1-CurrentFunctionalDefinitionNoGo
Required figures: status=1 details=figures_required=16
Independent gate recompute: status=1 details=gate_rows=10;inconsistent=0
No remaining v15.02.1 executable fallback: status=1 details=all contract lines complete; success gates closed; no action/controller/reset/audit-directed branch allowed
contract_unclosed_rows = 0
```

复核结论：

```text
v15.02.1 已达成 S1-ExploreOpenExecutionContractExecuted。
S2/S3/S4/S5 均未达成，promotion_allowed = 0。
当前计划内没有发现仍可合法补跑的 fallback/method/control/substrate 分支。
继续推进需要下一版 function definition 或 substrate/base-architecture 计划；
不能在 v15.02.1 内新增 action bank / controller / reset route / audit-directed token search。
```

## 16. v15.01/v15.02 自查后的 P-FB4 proximal fallback 修复

本次自查发现 v15.02.1 的 Line P fallback 存在实现级覆盖 bug：

```text
发现问题 1：
  P-FB4 的 selected alpha but no commit 分支在 alpha selection 阶段也跳过了 trial update，
  因而不能正确回答“objective 选择 alpha 后，不提交 proximal 部分会怎样”。

发现问题 2：
  train_case 只在 line == "P" 时进入 proximal solver。
  因此 P-FB4 虽然 artifact 写了 commit_mode，
  但实际训练走的是 G/function-metric update 分支，不是真正的 proximal commit-effect audit。

修复内容：
  alpha selection 阶段始终对每个 alpha 执行 trial update 来评估 B1/B2 objective；
  只有最终 commit 阶段才区分 selected alpha but no commit / commit alpha /
  random alpha same norm / same subspace random alpha。
  同时将 proximal 分支判断改为 line.startswith("P")，
  使 P-FB4 真正执行 proximal solver。
  同时新增 resolve_cuda_device 防护：--device 非 cuda 或 CUDA 不可用时直接报错，
  不再静默 fallback 到 CPU。

合法性：
  修复只恢复计划内 P-FB4 fallback；
  不新增 FU/F-CHE token，不新增 action/controller/reset route；
  不使用 validation/test/future/query；
  不使用 LineC/CEp99/NLL/ECE/AUCtime/Brier 生成方向。
```

GPU 复跑后的关键结果：

```text
device = cuda:0
line_p_candidate_count = 5
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = -0.03708608945210775
control_equivalent_fraction = 0.6666666666666666
line_p_exploration_gate_pass = 0
best_p_method = P4-D-CHE-RandomSketchFunctionProx

P-FB1 rows = 35100
P-FB2 rows = 5
P-FB3 rows = 5
P-FB4 rows = 72
P-FB4 commit alpha rows = 18, alpha_set = 0.2
P-FB4 random alpha same norm rows = 18, alpha_set = 0.2
P-FB4 same subspace random alpha rows = 18, alpha_set = 0.2
P-FB4 selected alpha but no commit rows = 18, alpha_set = 0.2
selected-no-commit objective rows = 5400
selected-no-commit objective nonzero B1 rows = 5400
selected-no-commit objective B1_improved rows = 5380

positive_rows_checked_by_m = 266
positive_row_control_map_rows = 266
positive_row_map_missing_control_rows = 0
positive_row_map_generic_explained_rows = 266

route = R15_2_1-CurrentFunctionalDefinitionNoGo
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
gate_route_inconsistent_rows = 0
contract_unclosed_rows = 0
```

结论：

```text
P-FB4 的覆盖 bug 修复后，v15.02.1 的 Line P 数值发生变化，
但仍没有达到 S2/S3/S4/S5。
当前结论仍是 function-metric/proximal definition no-go，
不是 promotion，也不是 implementation blocker。
```

## 17. 用户再次追问后的 GPU guard 入口修复

本次没有新增训练，也没有改变 v15.02.1 指标；只修复执行合同层缺口：

```text
发现问题：
  训练路径已要求 --device cuda，但 reuse-if-present/finalizer 路径可能先读取已有 artifact，
  从而绕过 GPU 校验。

修复：
  run(args) 入口新增 resolve_cuda_device(args.device)。
  --device 非 cuda 或 CUDA 不可用时直接报错。
```

验证：

```text
py_compile = pass
cuda_available = True
v15.02.1 route 结论不变：
  route = R15_2_1-CurrentFunctionalDefinitionNoGo
  promotion_allowed = 0
```
