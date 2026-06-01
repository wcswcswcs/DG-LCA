# DG-KAN v15.0 FunctionUpdate AllBasis 并行加速实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 optimizer trace、real-lite、substrate replay 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v15.0 的目标是把 function update 放回 optimizer-aware 语境，验证 second moment、alignment、decoupled regularization 和 basis-safe projection 是否能让 D-CHE FU 超过 matched controls。

## 2. 本轮代码修改

新增：

```text
experiments/run_v150_function_update_allbasis_parallel.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v15.0 预注册 substrate-only candidates：
  D-FOU37..41、D-RBF35..39、D-WAV33..36。
  不执行 Non-D-CHE official FMS proof，不新增 action/controller/reset route。
```

过程修正：

```text
1. smoke run 发现 v150_route_decision.json 在首次 manifest 自检前尚未写入，
   会导致临时 route = R0-ArtifactOrProvenanceViolation。
   已修正 finalizer 顺序：manifest 完成后重新计算 route 并重写 route/docs。
   该修正只影响 artifact completeness route，不改变任何实验指标。
2. 执行日志初版命令使用泛化 python 前缀；
   已改为记录实际解释器 /home/chengshun.wang/miniconda3/envs/kan/bin/python。
3. 用户再次追问后复核发现 FU3 初版把 hard/soft 合成单个 update，
   不满足计划中 hard 与 soft 必须同时测试的要求。
   已改为同一 method token 下写入 fu3_alignment_variant=hard/soft 两组实际训练行；
   M2-MLP-CautiousAdamW 同步补齐 hard/soft generic control。
4. 复核发现 v150_linec_tail_audit.csv 初版缺少 Line C 计划点名的 delta/proxy 字段。
   已补齐 CouplingR2_delta、NoiseSignalLeak_delta、RealSignalReservoirRatio_delta、
   CEp99_delta、NLL_delta、ECE_delta、Brier_delta、margin_p10_delta、
   signal_channel_energy、reservoir_energy、noise_leakage_proxy；这些字段只作 audit，不生成 direction。
5. 用户再次追问后发现 Rational no-regression monitor 未单独落表。
   已新增 v150_rational_no_regression_monitor.csv，读取 v14.4 official D-RAT artifact 作为 replay monitor；
   new_training_executed=0、reset_route_used=0、official_fms_proof_executed=0、promotion_allowed=0。
```

实现：

```text
1. Line R forbidden/no-action audit。
2. Line O optimizer mechanism manifest。
3. Line FU D-CHE FU0..FU8 + C0..C9 matched controls。
4. Line M MLP/generic optimizer controls M0..M7。
5. Line K KAN-specificity 与 decoupled decay deconfound。
6. Line D all-basis substrate-only v15 reconfirmation。
7. Line C tail/LineC audit、required manifest、figures、code review packet、执行日志和复盘日志。
```

## 3. Line O 结果

```text
decoupled_decay_implemented = 1
second_moment_implemented = 1
alignment_trace_available = 1
controls_available = 1
s1_optimizer_mechanism_implemented = 1
```

## 4. Line FU 结果

```text
line_fu_candidate_count = 9
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = -0.19696081876754762
control_equivalent_fraction = 0.8222222222222222
bad_event_fraction = 0.9444444444444444
line_fu_exploration_gate_pass = 0
line_fu_meaningful_gate_pass = 0
line_fu_s4_gate_pass = 0
best_fu_method = FU4-D-CHE-MGUPFunctionUpdate
best_fu_mean_source_vs_best_control = 0.2216603292359246
```

FU method summary：

| method | rows | strict pass | dataset-seed pass | mean source vs best control |
|---|---:|---:|---:|---:|
| FU0-D-CHE-AdamW | 9 | 0 | 0 | -0.017848 |
| FU1-D-CHE-RawFunctionUpdate | 9 | 0 | 0 | -1.156716 |
| FU2-D-CHE-AdamV2FunctionUpdate | 9 | 0 | 0 | -0.035111 |
| FU3-D-CHE-CautiousFunctionUpdate | 18 | 0 | 0 | -0.235107 |
| FU4-D-CHE-MGUPFunctionUpdate | 9 | 0 | 0 | 0.221660 |
| FU5-D-CHE-SophiaDiagClippedFunctionUpdate | 9 | 0 | 0 | -0.222945 |
| FU6-D-CHE-BlockSecondMomentFunctionUpdate | 9 | 0 | 0 | -0.115237 |
| FU7-D-CHE-DecoupledDecayPlusFunctionUpdate | 9 | 0 | 0 | -0.013650 |
| FU8-D-CHE-SOAPLiteDiagnosticFunctionUpdate | 9 | 0 | 0 | -0.159547 |

判断：

```text
FU1..FU8 只有超过 C0..C9 matched controls 才能写成 FU value。
本轮 best matched control = C0-D-CHE-AdamW；
未达到 S5 时 promotion_allowed 必须保持 0。
```

## 5. Line K / M 结果

```text
decoupled_decay_route = K-DecoupledDecayNotSufficient
decoupled_decay_explains_fraction = 0.2222222222222222
generic_optimizer_explains_gain = 1
best_mlp_control = M2-MLP-CautiousAdamW
best_mlp_source_vs_adamw = 0.3097565174102783
```

判断：MLP/generic optimizer controls 只作为 confound audit；即使出现 positive，也不能写成 KAN-specific promotion。

## 6. Line D all-basis 结果

```text
line_d_source = results/v15_0_function_update_allbasis_parallel/line_d_v150_allbasis_substrate/v149_line_d_substrate_repair_results.csv
line_d_rows = 126
best_non_dche_family = D-FOU
best_non_dche_dataset_seed_pass_count = 0 / 9
line_d_exploration_family_count = 0
line_d_official_fms_eligible_family_count = 0
line_d_route = R7-AllBasisCarrierBlocked
```

Line D family summary：

| family | rows | best candidate | family pass | official eligibility |
|---|---:|---|---:|---:|
| D-FOU | 45 | D-FOU37-LowFreqIdentityResidualV4 | 0/9 | 0 |
| D-RBF | 45 | D-RBF35-ActiveCenterSecondMoment | 0/9 | 0 |
| D-WAV | 36 | D-WAV33-TriangularSupportV4 | 0/9 | 0 |

## 6.1 Rational no-regression monitor

```text
monitor_source = v14.4_official_replay
new_training_executed = 0
reset_route_used = 0
official_fms_proof_executed = 0
promotion_allowed = 0
monitor_rows = 8
source_route = R2-RealTransferFail
best_method = K-RT1-TrainSplitAgreement
best_dataset_seed_pass_count = 3 / 9
```

判断：Rational monitor 只是 no-regression replay，不打开 reset route，不进入 v15.0 promotion。

## 7. 最终 route

```text
route = R8-CurrentFUDefinitionNoGo
minimum_success = S1-OptimizerMechanismImplemented
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 8. 科学结论

```text
1. v15.0 已执行 Line R/O/FU/M/K/D/C/Z，并生成 required artifacts。
2. 当前 minimum_success = S1-OptimizerMechanismImplemented；promotion_allowed = 0。
3. D-CHE FU real-lite pass = 0/9。
4. Non-D-CHE best substrate = D-FOU 0/9。
5. 未达到 S5 时不能把 decoupled decay、MLP/generic control 或 local positive row 写成 official functional success。
6. 若继续推进，需要下一版 substrate/base-architecture 或 theory-level functional update 计划，不能在 v15.0 内临时新增 FU9/FU10、F-CHE8/F-CHE9、action/controller/reset route。
```

## 9. 用户再次追问后的 Line K 控制面补齐

本次继续推进时发现 v15.0 计划 1.1 里还点名了：

```text
random-matched decoupled decay
```

此前已补齐：

```text
standard decoupled decay
cautious decoupled decay
coupled L2
no-decay FU baseline
```

但还没有实际执行 random-matched decoupled decay control。该项属于 control/audit surface，可在 `C6-SameDecoupledDecayNoFU` 族下补齐，不新增 FU/F-CHE token。

代码修改：

```text
experiments/run_v150_function_update_allbasis_parallel.py
  C6-SameDecoupledDecayNoFU 新增 decay_control_variant：
    standard
    cautious
    random_matched

  random_matched variant 使用当前 train-stream gradient 只确定 matched active count，
  再随机选择同数量 decay 坐标执行 decoupled shrink。
  它不使用 validation/test/future/query，不使用 LineC/CEp99/NLL/ECE/AUCtime 生成 direction。

  build_decay_deconfound 新增：
    random_matched_decoupled_decay_no_fu_gain
    random_matched_decay_active_fraction
    random_matched_decay_explains_gain
```

执行结果：

```text
v150_dche_fu_controls.csv rows = 108
C6-SameDecoupledDecayNoFU rows = 27
  standard = 9
  cautious = 9
  random_matched = 9

v150_decoupled_decay_deconfound.csv rows = 9
standard decay explains rows = 0
cautious decay explains rows = 0
random_matched decay explains rows = 2
any decay control explains rows = 2
decoupled_decay_explains_fraction = 0.2222222222222222
random_matched_decay_explains_fraction = 0.2222222222222222
decoupled_decay_route = K-DecoupledDecayNotSufficient
```

判断：

```text
random-matched decoupled decay 解释了 2/9 个 FU7 positive row，
但整体低于 confound route 阈值，不能把 FU7 写成 decoupled decay success，
也不能打开 promotion。
Line K 仍为 K-DecoupledDecayNotSufficient。
```

## 10. 最终覆盖复核

```text
v150_dche_fu_results.csv rows = 90
  FU0/FU1/FU2/FU4/FU5/FU6/FU7/FU8 = 9 rows each
  FU3-D-CHE-CautiousFunctionUpdate = 18 rows
    hard = 9
    soft = 9

v150_dche_fu_controls.csv rows = 108
  C0/C1/C2/C3/C4/C5/C7/C8/C9 = 9 rows each
  C6-SameDecoupledDecayNoFU = 27 rows
    standard = 9
    cautious = 9
    random_matched = 9

v150_mlp_controls.csv rows = 81
  M0/M1/M3/M4/M5/M6/M7 = 9 rows each
  M2-MLP-CautiousAdamW = 18 rows
    hard = 9
    soft = 9

Line O traces:
  v150_second_moment_trace.csv rows = 1917
  v150_alignment_trace.csv rows = 837
  v150_curvature_diag_trace.csv rows = 837
  v150_block_preconditioner_trace.csv rows = 837
  v150_schedulefree_control_trace.csv rows = 837

Line C:
  v150_linec_tail_audit.csv rows = 675
  required delta/proxy fields present
  linec_metric_used_as_direction sum = 0

Line D:
  v150_allbasis_substrate_results.csv rows = 126
  D-FOU37..41 / D-RBF35..39 / D-WAV33..36 已执行
  best_non_dche_family = D-FOU
  best_non_dche_dataset_seed_pass_count = 0 / 9

Rational monitor:
  v150_rational_no_regression_monitor.csv rows = 8
  best_method = K-RT1-TrainSplitAgreement
  best_dataset_seed_pass_count = 3 / 9
  new_training_executed = 0
  reset_route_used = 0
```

required / audit 复核：

```text
v150_required_manifest.csv rows = 39
required_artifact_missing_count = 0
manifest_missing_sum = 0

v150_forbidden_information_audit.csv rows = 42
forbidden_information_violation_count = 0
violation_sum = 0

v150_no_action_search_audit.csv rows = 42
no_action_search_violation_count = 0
violation_sum = 0
```

最终判断：

```text
route = R8-CurrentFUDefinitionNoGo
minimum_success = S1-OptimizerMechanismImplemented
source_vs_best_control_mean = -0.19696081876754762
control_equivalent_fraction = 0.8222222222222222
best_fu_method = FU4-D-CHE-MGUPFunctionUpdate
best_fu_mean_source_vs_best_control = 0.2216603292359246
generic_optimizer_explains_gain = 1
official_s5_reached = 0
promotion_allowed = 0

v15.0 已达成 S1。
没有达成 S2 / S3 / S4 / S5。
当前计划内 FU/M/K/D/C/R/Z、standard/cautious/random-matched decay controls、
Rational no-regression monitor 均已覆盖。
没有发现仍可合法补跑的预注册 method/control/substrate/monitor 分支。
继续推进需要下一版 substrate/base-architecture 或 theory-level functional update 计划；
不能在 v15.0 内新增 FU9/FU10、F-CHE8/F-CHE9、action token、controller、action bank、reset route，
也不能使用 LineC/CEp99/NLL/ECE/AUCtime 反推 direction。
```
