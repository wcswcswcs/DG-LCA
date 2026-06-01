# DG-KAN v14.15 ParallelFMSCausalFork TransferObservability AllBasis 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 replay diagnostic、real-lite、local positive、substrate replay 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v14.15 将 v14.14 的串行 no-go 拆成并行 decisive fork：Line P predictor robustness、Line I intervention causality、Line B boundary/curriculum role reset、Line F D-CHE bounded real-lite、Line D all-basis substrate acceleration、Line M controls、Line C/Z。

## 2. 本轮代码修改

新增：

```text
experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v14.15 预注册 substrate-only candidates：
  D-FOU32..36、D-RBF30..34、D-WAV29..32。
  不执行 official FMS proof，不新增 action/controller/reset route。

experiments/run_v143_nonrat_compact_task_health_probe.py
  新增 D-FOU Fourier train-stream telemetry：
  band_energy_low / band_energy_mid / band_energy_high、
  phase_drift、high_freq_ratio、bandwise_snr。
  telemetry_source = train_stream_layer1_basis_channel_split；
  只使用 train-stream layer1 basis activation，不使用 labels / validation / test /
  future / query / LineC / CEp99 / NLL / ECE / AUCtime 生成方向。
  同时补齐 RBF quantile_width100 repair 解析，
  使 D-RBF34-CenterOccupancyWarmupNoTaskBranch 不再因 parser 缺口 blocked。

experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  为 v14.15 Line M 补齐 MLP/generic controls：
  M5-MLP-NoOpMatchedOverhead、
  M6-MLP-SameActiveFractionControl、
  M7-MLP-BoundaryOnly。
  这些是 control surface，不是 D-CHE FMS token；
  不新增 F-CHE8/F-CHE9、FMS-M6/M7、action/controller/reset route。
```

过程修正：

```text
首次 Line D 命令只覆盖最低完成要求中的 D-FOU32..36 / D-RBF30..34。
复核计划 8.4 后确认 D-WAV29..32 也是预注册 low-budget all-basis 分支。
已修正执行命令并重跑 Line D 全量候选：
  D-FOU32..36、D-RBF30..34、D-WAV29..32。
最终 Line D candidate rows = 261；
其中 v14.14 baseline rows = 135；
v14.15 new rows = 126；
line_d_summary_rows = 3；
v1415_allbasis_substrate_acceleration.csv total rows = 264；
best_non_dche_dataset_seed_pass_count = 2 / 9。
```

## 3. Line P 结果

```text
best_feature = P2-recovery_lag
best_auc_mean = 0.9090909090909091
best_leaveout_min = 0.5
best_incremental_auc = 0.40909090909090906
best_spearman_source = 0.034221800155956356
line_p_exploration_gate_pass = 0
line_p_promotion_enabling_gate_pass = 0
line_p_route = R-P-PredictorWeak
leaveout_rows = 54
leaveout_planned_splits = leave-dataset-out,leave-seed-out,leave-loss-interface-out,leave-method-out,leave-control-out,leave-synthetic-family-out
leaveout_available_rows = 36
leaveout_unavailable_rows = 18
```

判断：

```text
Line P 按计划补齐六类 planned leaveout split 记录。
其中 method/control leaveout 在 v14.14 E4 artifact 中不存在；
本轮只标记 unavailable，不补填 AUC，不参与 gate。
leave-synthetic-family-out 映射自 v14.14 leave_task_family_out，
并保留 mapping_note 方便审计。
```

## 4. Line I 结果

```text
line_i_rows = 270
line_i_selector_ids = S0-always-on,S1-E4-predictor-high-score,S2-inverse-E4-predictor,S3-random-matched-active-fraction,S4-NoOp-safe-abstention-selector
line_i_boundary_ids = B0-no-boundary,B1-degree-projection-safety,B2-low-amplitude-boundary,B3-rejection-fraction-cap,B4-value-retention-floor
line_i_selector_replay_rows = 45
line_i_value_retention_floor_rows = 9
line_i_strict_selector_contrast_available = 0
mean_direction_incremental_gain = -0.01765242298444112
control_equivalent_fraction = 0.9466666666666667
bad_event_fraction = 0.5022222222222222
real_lite_pass_count = 0
line_i_exploration_gate_pass = 0
line_i_route = R-I-DirectionControlEquivalent
```

判断：

```text
补充复核后，Line I 增加 S1/S2 selector replay：
  S1-E4-predictor-high-score
  S2-inverse-E4-predictor
这些行只重建 E4-K recovery_lag selector 分组，不是新 real training。
同时补入 B4-value-retention-floor，
来源是 v14.13 FMS-M5-ProjectionRetentionFloor 真实 artifact replay。
由于同方向 random selector 未执行，
strict_selector_contrast_available = 0；
因此 selector replay 不能打开 causal promotion。
```

## 5. Line B 结果

```text
line_b_executed_boundary_count = 5
best_boundary = BND3-DegreeEnergyLimiter
best_source_vs_best_control = -0.023050483730104234
min_control_equivalent_fraction = 0.7777777777777778
line_b_any_gate_pass = 0
line_b_route = R-B-BoundaryIsHarmlessNull
```

## 6. Line F 结果

```text
line_f_candidate_count = 3
line_f_real_lite_pass_count = 0 / 9
source_vs_best_control_mean = -0.032401322214691726
source_vs_best_control_min = -0.12490177154541016
control_equivalent_fraction = 0.9259259259259259
line_f_meaningful_gate_pass = 0
line_f_s4_gate_pass = 0
line_f_route = R-F-RealLiteBelow4
```

## 7. Line D 结果

```text
line_d_source = v1414_baseline_plus_v1415_new_line_d_v149_substrate_acceleration
line_d_rows = 261
line_d_v1414_baseline_rows = 135
line_d_v1415_new_rows = 126
line_d_fou_telemetry_required_rows = 45
line_d_fou_telemetry_available_rows = 45
line_d_fou_telemetry_missing_rows = 0
line_d_fou_telemetry_source = train_stream_layer1_basis_channel_split
line_d_rbf_residual_required_rows = 45
line_d_rbf_residual_available_rows = 45
line_d_rbf_residual_missing_rows = 0
line_d_rbf_residual_source = no_linear_residual_present,train_stream_logit_residual_norm_ratio
line_d_v1415_gate_definition = step_ratio<=1.75 & memory_ratio<=1.75 & mean_delta>=-0.05 & worst_delta>=-0.10 & LineC_pass_rate>=0.30
best_non_dche_family = D-FOU
best_non_dche_dataset_seed_pass_count = 2 / 9
official_fms_eligible_family_count = 0
line_d_route = R-D-SubstrateStillBlocked
```

Line D telemetry / visualization coverage:

```text
D-FOU Fourier telemetry required/available/missing =
  45 /
  45 /
  0
telemetry_source = train_stream_layer1_basis_channel_split
D-RBF residual_over_base required/available/missing =
  45 /
  45 /
  0
residual_source = no_linear_residual_present,train_stream_logit_residual_norm_ratio
v14.15 substrate gate definition =
  step_ratio<=1.75 & memory_ratio<=1.75 & mean_delta>=-0.05 & worst_delta>=-0.10 & LineC_pass_rate>=0.30
fig_d_family_pass_heatmap.svg = generated
fig_d_workspace_task_pareto.svg = generated
fig_d_nonrat_telemetry_matrix.svg = generated
fig_d_linec_tail_task_breakdown.svg = generated
```

## 7.1 Line M control 覆盖

```text
line_m_required_control_count = 7
line_m_executed_required_control_count = 7
line_m_missing_required_control_count = 0
line_m_missing_required_controls = 
line_m_route = M-ControlsCompleteNoGenericPositive
```

## 8. 最终 route

```text
route = R15-ParallelForkNoCausalValueAllBasisBlocked
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 9. 科学结论

```text
1. v14.15 已执行 Line R/P/I/B/F/D/M/C/Z。
2. 本轮没有达成 S4-lite / S4 / S5，promotion_allowed = 0。
3. Line I 未证明 direction 有独立 causal value；Line B 未打开 boundary/curriculum gate。
4. Line F D-CHE real-lite 仍低于 >=4/9 meaningful gate。
5. Line D 非 D-CHE substrate 仍低于 >=6/9 exploration gate。
6. Line M 已补齐计划要求的 7 类 MLP/generic controls；
   没有把 MLP/generic control 写成 promotion。
7. 继续推进需要下一版 functional causal theory 或新的 substrate carrier；
   不能在 v14.15 内新增 token/controller/action/reset route 或 audit-directed branch。
```
