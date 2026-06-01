# DG-KAN v14.16 DecisiveFMSFork MetricState AllBasisAcceleration 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 replay diagnostic、metric-state real-lite、substrate replay 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v14.16 的目标是对当前 FMS definition 做 decisive exit test：判断 direction generator、metric-state、transfer predictor、D-CHE real-lite、all-basis carrier 是否任一具备因果价值。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py
```

修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 MS1-ParameterMetricState、MS2-DegreeRoleMetricState、MS3-BasisGroupMetricState。
  新增 matched controls：
    C6-D-CHE-SameTCRandomDirection
    C7-D-CHE-SameMetricScaleRandomPermutation
    M8-MLP-MetricState-MS1
    M9-MLP-MetricState-MS2
    M10-MLP-SameMetricScaleRandomPermutation
    M11-MLP-AdamWParallelDirectionControl
  metric-state utility 只使用 train-stream per-example gradient drift/diffusion；
  不读取 validation/test/future/query，也不使用 LineC/CEp99/NLL/ECE/AUCtime 生成方向。
```

过程修正：

```text
首次 v14.16 finalizer 发现 required manifest 自身在写入前被 self-check 计为 missing=1。
已修正为 manifest 自身生成时按 exists=1 记录，并重写 route / manifest / 两份日志。
该修正只影响 artifact completeness 计数，不改变任何实验指标。

二次复核发现 Line A 初版只从 v14.14/v14.15 replay artifact 构造 available matrix，
不能满足计划要求的 full selector-direction-boundary contrast table。
已改为实际执行 3 selector × 7 direction × 3 boundary × 3 dataset × 3 seed = 567 rows；
其中 D4/D5 是 matched-random controls，不是新 FMS token。

三次复核发现 Line C 的 leave-time-budget-out 仍是 unavailable，
且 runner 中 --compute-budgeted-run 尚未接入。
已按计划内 fixed half-budget probe 补齐：
  full-budget Line A + fixed half-budget Line A 用于 leave-time-budget-out；
  v14.16 actual Line A 可观测特征用于 leave-method-out / leave-control-out；
  P2/P7 因当前 actual artifact 无 recovery_lag / micro_horizon_loss_integral，
  仍只记录 unavailable，不补填 AUC。

四次复核发现 Line E 初版只读取 v14.15 all-basis replay，
而完整计划第 10 节要求用并行预算重新确认 D-FOU / D-RBF / D-WAV。
已修改 runner：
  新增 --line-e-out；
  若存在 v149_line_d_substrate_repair_results.csv，则优先读取 v14.16 实跑 substrate reconfirmation；
  输出 line_e_source / line_e_rows / v1416_substrate_gate_pass；
  只有缺失实跑 artifact 时才 fallback 到 v14.15 replay。
该修正不新增 FMS/F-CHE token，不执行 Non-D-CHE official FMS proof。
```

## 3. Line A Current FMS Direction Exit Test

```text
planned_rows = 567
available_rows = 567
mean_source_vs_best_control = -0.01974100978286178
control_equivalent_fraction = 0.8888888888888888
bad_event_fraction = 0.9629629629629629
real_lite_pass_count = 0 / 9
line_a_gate_pass = 0
line_a_route = R-A-CurrentFMSDirectionNoGo
```

判断：Line A 没有证明 current FMS direction 有独立 causal value；567 个 contrast cell 均为实际执行，不补填数据。

## 4. Line B Metric-State FMS Rebuild

```text
metric_state_rows = 27
metric_state_controls_rows = 54
population_risk_trace_rows = 486
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = -0.034055363248895715
control_equivalent_fraction = 0.9259259259259259
line_b_exploration_gate_pass = 0
line_b_strong_gate_pass = 0
line_b_route = R-B-MetricStateFMSNoGo
```

判断：MS1/MS2/MS3 没有打开 >=3/9 exploration gate；不得调 beta/clip/grid，也不得新增 MS4。

## 5. Line C Transfer Predictor Deconfounding

```text
best_feature = P1-degree_projection_rejection_fraction
best_auc_mean = 0.5
best_leaveout_min = 0.5
best_spearman_source = 0.05597363779181217
best_raw_feature = P2-recovery_lag
best_complete_feature = P1-degree_projection_rejection_fraction
leaveout_rows = 63
leaveout_available_rows = 57
complete_feature_count = 7
time_budget_probe_rows = 567
line_c_gate_pass = 0
line_c_route = R-C-TransferPredictorNotRobust
```

判断：Line C 已补齐计划内 method/control/time-budget split 的可执行部分；
无法从 actual artifact 观测的 P2/P7 不做补填。
predictor 仍未达到 complete leaveout 且 leaveout_min>=0.60 且 Spearman>=0.20；
不能用于 gating / abstention / boundary。

## 6. Line D D-CHE Real-Lite Verification

```text
eligible_mechanism_count = 0
real_lite_pass_count = 0 / 9
line_d_route = R-D-DCHENoRealLiteTransfer
```

判断：没有 Line A/B 机制通过入口 gate，因此 Line D 不打开新 D-CHE real-lite，以避免把失败机制继续 real-lite 搜索。

## 7. Line E All-Basis Substrate Acceleration

```text
best_family = D-FOU
line_e_source = v1416_actual_v149_substrate_reconfirmation
line_e_rows = 126
best_family_dataset_seed_pass_count = 0 / 9
line_e_gate_pass = 0
line_e_route = R-E-AllBasisCarrierBlocked
```

判断：D-FOU / D-RBF / D-WAV 均未达到 >=6/9 substrate exploration gate；不得进入 Non-D-CHE official FMS proof。

## 8. Line M Controls

```text
line_m_rows = 81
generic_control_explains_gain = 0
line_m_route = M-ControlsCompleteNoPositive
```

判断：MLP/generic controls 只作为 confound audit，不写成 KAN-specific success。

## 9. 最终 route

```text
route = R16-CurrentFMSDefinitionNoCausalValue
minimum_success = S0-ParallelForkExecuted
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 10. 科学结论

```text
1. v14.16 已执行 Line R/A/B/C/D/E/M/Z，并生成 required artifacts。
2. Current FMS direction no-go；Metric-State MS1/MS2/MS3 也未打开 exploration gate。
3. Transfer predictor 不稳健，不能用于 gating。
4. D-CHE 下没有可进入 Line D 的 passing mechanism。
5. Non-D-CHE all-basis carrier 仍 blocked。
6. 当前结论是退出 current FMS definition，而不是继续新增 token/controller/action/reset route。
```

## 11. 用户再次追问后的 Line E actual reconfirmation

本次继续推进不是新增 token，而是修复计划覆盖性问题：

```text
发现问题：
  Line E 原先使用 v14.15 all-basis replay 作为 fallback；
  这不足以满足 v14.16 计划中“用并行预算重新确认 all-basis substrate”的要求。

代码修改：
  experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py
    新增 DEFAULT_LINE_E_OUT / LINE_E_CANDIDATES。
    build_line_e 优先读取 --line-e-out 下的 v149_line_d_substrate_repair_results.csv。
    写入 line_e_source、line_e_rows、v1416_substrate_gate_pass。
    若实跑 artifact 缺失，才回退到 v14.15 replay。
```

实际执行的 v14.16 Line E substrate-only reconfirmation：

```text
out_dir = results/v14_16_decisive_fms_fork_metric_state_allbasis/line_e_v1416_substrate_acceleration
candidate_rows = 126
linec_rows = 378
families =
  D-FOU: 45 rows
  D-RBF: 45 rows
  D-WAV: 36 rows
official_fms_proof_executed = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

Line E family summary：

| family | rows | reported best candidate | family pass | max mean delta vs MLP | candidate for max mean delta | best LineC pass rate | official eligibility |
|---|---:|---|---:|---:|---|---:|---:|
| D-FOU | 45 | D-FOU32-LowFreqIdentityResidualV3 | 0/9 | -0.109375 | D-FOU33/D-FOU35 | 1.0 | 0 |
| D-RBF | 45 | D-RBF30-ActiveCenterOccupancyV3 | 0/9 | -0.328125 | D-RBF33 | 1.0 | 0 |
| D-WAV | 36 | D-WAV29-TriangularSupportV4 | 0/9 | -0.0234375 | D-WAV31 | 1.0 | 0 |

合入 official_v1416 finalizer 后：

```text
line_e_source = v1416_actual_v149_substrate_reconfirmation
line_e_rows = 126
line_e_best_family = D-FOU
line_e_best_count = 0 / 9
line_e_gate_pass = 0
line_e_route = R-E-AllBasisCarrierBlocked
route = R16-CurrentFMSDefinitionNoCausalValue
minimum_success = S0-ParallelForkExecuted
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

结论：

```text
v14.16 计划内的 Line E actual reconfirmation 已补齐。
D-FOU / D-RBF / D-WAV 都没有达到 >=6/9 exploration substrate gate；
因此不能进入 Non-D-CHE official FMS proof。
A/B/C/D/E/M/Z 全部闭合后，当前仍只能得到 S0-ParallelForkExecuted；
没有 S1/S2/S3/S4/S5，promotion_allowed 仍为 0。
继续推进需要下一版 optimizer/theory-level functional update 或 substrate/base-architecture 计划，
不能在 v14.16 内新增 FMS-M6/M7、F-CHE8/F-CHE9、action token、controller、action bank、reset route，
也不能用 LineC/CEp99/NLL/ECE/AUCtime 反推方向。
```

## 12. 用户再次追问后的计划覆盖矩阵复核

本次没有新增训练；重新读取 v14.16 完整计划、official artifacts 和 actual Line E artifact，确认是否仍有可执行且不越界的分支。

Line A 覆盖：

```text
v1416_a_current_fms_exit_matrix.csv rows = 567
selectors =
  S0-always-on
  S1-predictor-high-score
  S4-NoOp-safe-abstention
directions =
  D0-AdamW
  D1-current-FMS
  D2-random-matched-norm
  D3-same-active-fraction-random
  D4-same-projection-rejection-random
  D5-same-value-retention-random
  D6-AdamWParallelDirection-control
boundaries =
  B0-none
  B1-degree-projection-safety
  B4-value-retention-floor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

Line B / M control 覆盖：

```text
v1416_b_metric_state_results.csv rows = 27
metric-state methods =
  MS1-ParameterMetricState
  MS2-DegreeRoleMetricState
  MS3-BasisGroupMetricState

v1416_b_metric_state_controls.csv rows = 54
D-CHE controls =
  C0-D-CHE-AdamW
  C1-D-CHE-AdamW-NoOpMatchedOverhead
  C2-D-CHE-AdamW-RandomMatchedNorm
  C5-D-CHE-AdamW-SameActiveFractionControl
  C6-D-CHE-SameTCRandomDirection
  C7-D-CHE-SameMetricScaleRandomPermutation

v1416_m_mlp_generic_controls.csv rows = 81
MLP/generic controls =
  M0-MLP-AdamW
  M8-MLP-MetricState-MS1
  M9-MLP-MetricState-MS2
  M6-MLP-SameActiveFractionControl
  M10-MLP-SameMetricScaleRandomPermutation
  M3-MLP-RandomMatchedNorm
  M5-MLP-NoOpMatchedOverhead
  M4-MLP-GenericOptimizerStateControl
  M11-MLP-AdamWParallelDirectionControl
```

Line C 覆盖：

```text
v1416_c_transfer_predictor_leaveout.csv rows = 63
available rows = 57
planned leaveout splits =
  leave-dataset-out: 9/9 available
  leave-seed-out: 9/9 available
  leave-loss-interface-out: 9/9 available
  leave-method-out: 7/9 available
  leave-control-out: 7/9 available
  leave-synthetic-family-out: 9/9 available
  leave-time-budget-out: 7/9 available
unavailable features =
  P2-recovery_lag
  P7-micro_horizon_loss_integral
reason =
  current v14.16 actual Line A artifact does not contain recovery_lag /
  micro_horizon_loss_integral telemetry; no AUC is imputed.
```

Line E 覆盖：

```text
actual Line E rows = 126
D-FOU candidates = D-FOU32..36, rows = 45, pass = 0/9
D-RBF candidates = D-RBF30..34, rows = 45, pass = 0/9
D-WAV candidates = D-WAV29..32, rows = 36, pass = 0/9
line_e_source = v1416_actual_v149_substrate_reconfirmation
```

required / audit 复核：

```text
v1416_required_artifact_manifest.csv rows = 38, missing_sum = 0
v1416_forbidden_information_audit.csv rows = 31, violation_sum = 0
v1416_no_action_search_audit.csv rows = 31, violation_sum = 0
required figures = 11/11 present
```

最终判断：

```text
v14.16 达成最小成功 S0-ParallelForkExecuted；
没有达成 S1-FMSDirectionCausalValue；
没有达成 S2-MetricStateFMSPositive；
没有达成 S3-TransferPredictorRobust；
没有达成 S4-AllBasisAlternativeCarrier；
没有达成 S4-DCHENewRealLitePositive；
没有达成 S5-OfficialFunctionalSuccess。

当前计划内没有发现仍可合法补跑的 method/control/substrate 分支。
继续推进必须进入下一版 optimizer/theory-level functional update
或 substrate/base-architecture bottleneck 计划；
不能在 v14.16 内新增 FMS/F-CHE token、controller、action bank、reset route，
也不能用 audit metric 反推方向。
```

## 13. 用户再次追问后的实现级 selector / boundary 复核

本次没有新增训练；专门复核 Line A 的 selector / boundary 是否真的进入训练逻辑，而不是只作为 artifact metadata 写入。

代码层复核：

```text
experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py
  run_actual_line_a 为每个 cell 设置：
    local_args.line_a_selector_id
    local_args.line_a_direction_id
    local_args.line_a_boundary_id

experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  train_model_case 读取 line_a_boundary_id：
    B1-degree-projection-safety
    B4-value-retention-floor
  并基于当前 train-stream projection stats 调整 projected_flat。

  train_model_case 读取 line_a_selector_id：
    S1-predictor-high-score
    S4-NoOp-safe-abstention
  并基于当前 train-stream utilities / projection stats 决定是否跳过 optimizer step。
```

实际 artifact 中的 selector 行为：

```text
v1416_a_current_fms_exit_matrix.csv rows = 567

selector active fraction:
  S0-always-on:
    n = 189
    min = 1.0
    max = 1.0
    mean = 1.0

  S1-predictor-high-score:
    n = 189
    min = 0.0
    max = 0.015
    mean = 0.010714285714285714

  S4-NoOp-safe-abstention:
    n = 189
    min = 1.0
    max = 1.0
    mean = 1.0
```

Current FMS direction source_vs_best_control by selector：

```text
S0-always-on:
  rows = 27
  mean = -0.015283273326026069
  min = -0.0378987193107605
  max = 0.011716127395629883

S1-predictor-high-score:
  rows = 27
  mean = -0.028656482696533203
  min = -0.08471512794494629
  max = 0.019894123077392578

S4-NoOp-safe-abstention:
  rows = 27
  mean = -0.015283273326026069
  min = -0.0378987193107605
  max = 0.011716127395629883
```

复核判断：

```text
1. Line A selector / boundary 不是 metadata-only，已经进入训练 step。
2. S1 selector 实际大幅 abstain，但没有改善 current FMS direction；
   mean source_vs_best_control 比 S0 更低。
3. S4 safe-abstention 在当前 projection stats 下几乎等价于 S0，
   没有打开 boundary/curriculum value。
4. 若要把 S1 改成 recovery_lag / micro_horizon predictor gating，
   需要新增当前 v14.16 actual artifact 不具备的 telemetry 或重开 predictor-gating route；
   这会违反 Line C fail 后 predictor 不能用于 gating 的计划边界。
5. 因此本轮没有新的合法训练补跑分支。
```

当前合法结论仍为：

```text
route = R16-CurrentFMSDefinitionNoCausalValue
minimum_success = S0-ParallelForkExecuted
official_s5_reached = 0
promotion_allowed = 0
```

## 14. 用户再次追问后的 Line B / Line M control surface 复核

本次没有新增训练；复核 Line B 计划中的 controls 与实际 artifact 的映射，避免把 `C4 MLP-MetricState` 误判成漏跑。

计划 Line B controls：

```text
C0 AdamW
C1 SameT(c)RandomDirection
C2 SameActiveFraction
C3 SameMetricScaleRandomPermutation
C4 MLP-MetricState
C5 NoOpMatchedOverhead
```

实际 artifact 覆盖：

```text
v1416_b_metric_state_results.csv rows = 27
Line B metric-state methods =
  MS1-ParameterMetricState
  MS2-DegreeRoleMetricState
  MS3-BasisGroupMetricState

v1416_b_metric_state_controls.csv rows = 54
D-CHE controls =
  C0-D-CHE-AdamW
  C1-D-CHE-AdamW-NoOpMatchedOverhead
  C2-D-CHE-AdamW-RandomMatchedNorm
  C5-D-CHE-AdamW-SameActiveFractionControl
  C6-D-CHE-SameTCRandomDirection
  C7-D-CHE-SameMetricScaleRandomPermutation

v1416_m_mlp_generic_controls.csv rows = 81
MLP metric-state controls covering Line B C4 =
  M8-MLP-MetricState-MS1
  M9-MLP-MetricState-MS2
```

Line B / MLP metric-state 数值复核：

| method | rows | mean source vs best control | strict pass | mean NLL |
|---|---:|---:|---:|---:|
| MS1-ParameterMetricState | 9 | -0.023307979106903076 | 0 | 0.8350219064288669 |
| MS2-DegreeRoleMetricState | 9 | -0.040673659907446966 | 0 | 0.8523875872294108 |
| MS3-BasisGroupMetricState | 9 | -0.0381844507323371 | 0 | 0.8498983780543009 |
| M8-MLP-MetricState-MS1 | 9 | -0.18170246150758532 | 0 | 1.2892921831872728 |
| M9-MLP-MetricState-MS2 | 9 | -0.12233130799399482 | 0 | 1.2299210296736822 |

判断：

```text
1. Line B 的 D-CHE matched controls 已覆盖 SameT(c)RandomDirection、
   SameActiveFraction、SameMetricScaleRandomPermutation、NoOpMatchedOverhead 和 AdamW。
2. 计划中的 C4 MLP-MetricState 由 Line M 的 M8/M9 覆盖；
   它是 generic/MLP control surface，不写入 D-CHE control CSV。
3. D-CHE MS1/MS2/MS3 自身已全部低于 D-CHE best controls；
   即便只看 Line B 内部 gate，也没有 S2。
4. MLP metric-state controls 也没有 positive-looking success；
   不能作为 KAN-specific promotion，且不打开 R-M 后续分支。
5. 因此不存在“Line B control 漏跑导致不该停止”的问题。
```

当前合法结论仍为：

```text
route = R16-CurrentFMSDefinitionNoCausalValue
minimum_success = S0-ParallelForkExecuted
official_s5_reached = 0
promotion_allowed = 0
```
