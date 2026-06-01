# DG-KAN v14.10 NonRAT FMS Transfer + FMSDefinitionReset 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入本轮实际 artifact 中的结果；不虚构成功、不补填未执行 real 3x3、不把 substrate eligibility / repair / 局部 positive row 写成 promotion。

## 1. 计划理解

v14.10 的核心不是继续 Rational reset / action search，而是利用 v14.9 打开的 D-CHE 9/9 substrate eligibility，验证：

```text
D-CHE + FMS
是否能打过
D-CHE + AdamW / matched controls
```

硬边界：

```text
1. strict FC-PureKAN / no active B-spline。
2. 不使用 teacher / distillation / loss modification / sampler / class weight。
3. 不使用 dataset-name branch / seed-specific scale / label-informed initialization。
4. FMS direction 不使用 validation / test / future / query。
5. LineC / CEp99 / NLL / ECE / AUCtime / Brier 只作为 audit / gate，不作为 direction。
6. 不新增 action token，不启动 controller。
7. D-CHE substrate eligibility 不等于 FMS success。
8. D-CHE synthetic S3 不过时，不允许打开 real 3x3。
```

## 2. 本轮代码修改

新增：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

实现内容：

```text
1. v14.10 official artifact surface 和 required manifest。
2. D-CHE C0..C4 controls。
3. D-CHE F-CHE1..F-CHE7 FMS definition variants。
4. Case A fallback：FB1 / FB2 / FB3。
5. MLP/generic controls：M0 / M1 / M2 / M3 / M4。
6. D-CHE degree telemetry：
   degree_energy_before / after
   high_degree_energy_fraction
   degree_entropy
   degree_gate_active_fraction
7. projection/value-retention telemetry：
   cos_projected_vs_generic
   value_retention_after_degree_projection
   degree_projection_rejection_fraction
8. forbidden information audit / no-action-search audit。
9. 读取 v14.9 Line D substrate artifact，写入 all-basis substrate status。
10. synthetic S3 fail 后 fail-closed，不打开 real 3x3。
```

合法性：

```text
1. direction 只来自当前 train-stream supervised loss / per-example gradient / FMS state。
2. LineC/tail/calibration/AUC 不生成方向。
3. no_action_search_violation = 0。
4. forbidden_information_violation = 0。
5. promotion_allowed = 0。
```

语法检查：

```text
py_compile pass
```

## 3. Smoke

规模：

```text
task = X1
seed = 0
loss = CE
methods = C0,C2,F-CHE2
mlp_methods = M0,M1
train_steps = 4
batch_size = 8
synthetic_train_size = 32
synthetic_val_size = 16
skip_real = 1
compute_budgeted_run = 1
```

结果：

```text
route = R1-DCHESyntheticFMSFail
minimum_success = S0-ExecutionCompleteNoPromotion
synthetic_task_family_pass_count = 0
dche_substrate_official_fms_eligibility = 1
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

解释：

```text
smoke 只证明 runner / artifact surface 可执行，不作为 success 或 no-go official 证据。
```

## 4. Official compute-budgeted synthetic

规模：

```text
synthetic_tasks = X1..X7
synthetic_seeds = 0,1,2
loss_interfaces = CE,Brier
methods = C0..C4 + F-CHE1..F-CHE7
fallbacks = FB1,FB2,FB3
mlp_methods = M0..M4
train_steps = 200
batch_size = 32
synthetic_train_size = 96
synthetic_val_size = 64
fms_update_interval = 80
compute_budgeted_run = 1
```

route：

```text
route = R1-DCHESyntheticFMSFail
minimum_success = S0-ExecutionCompleteNoPromotion
synthetic_task_family_pass_count = 0
synthetic_task_pass = {}
dche_substrate_official_fms_eligibility = 1
dche_best_source_vs_adamw = 0.1652148962020874
mlp_best_source_vs_adamw = 0.2887003421783447
generic_fms_confound = 1
real_dataset_seed_pass_count = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

method summary：

| method | rows | strict pass | task pass | mean source | median AUC | median step | median memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-CHE1 | 42 | 0 | 0 | -0.039813 | 1.018485 | 1.400370 | 1.150207 |
| F-CHE2 | 42 | 0 | 0 | -0.037491 | 1.015735 | 1.401474 | 1.150207 |
| F-CHE3 | 42 | 0 | 0 | -0.034315 | 1.016506 | 1.398118 | 1.150207 |
| F-CHE4 | 42 | 0 | 0 | -0.027577 | 1.011317 | 1.396688 | 1.150207 |
| F-CHE5 | 42 | 0 | 0 | -0.040617 | 1.020044 | 1.393282 | 1.150207 |
| F-CHE6 | 42 | 0 | 0 | -0.036001 | 1.017620 | 1.393949 | 1.150207 |
| F-CHE7 | 42 | 0 | 0 | -0.029512 | 1.016207 | 1.399176 | 1.150207 |
| FB1 | 42 | 0 | 0 | -0.038867 | 1.017308 | 1.403134 | 1.150207 |
| FB2 | 42 | 0 | 0 | -0.027842 | 1.013301 | 1.402498 | 1.150207 |
| FB3 | 42 | 0 | 0 | -0.031921 | 1.017988 | 1.404420 | 1.150207 |

official failure decomposition：

```text
source fail rows = 377
AUCtime fail rows = 368
step fail rows = 420
CEp99 fail rows = 175
NLL fail rows = 103
ECE fail rows = 92
LineC fail rows = 0
```

判断：

```text
1. D-CHE substrate 没有回退：official_fms_eligibility = 1。
2. D-CHE FMS synthetic S3 没有成立：0/7 task families。
3. F-CHE rows 有局部 positive source，但不同时满足 source/AUC/tail/step gates。
4. MLP best source 高于 D-CHE best source，因此 generic_fms_confound = 1。
5. LineC 不是本轮 blocker。
6. real 3x3 没有打开。
```

## 5. Case A fallback 与 repair

计划 Case A 要求 S3 不过时先判断 failure source，并且只允许：

```text
F-CHE-FB1-ValuePathOnly
F-CHE-FB2-DegreeConstraintOnly
F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection
```

本轮 official run 已执行 FB1/FB2/FB3，仍未打开 S3。

因为 official failure 中所有 F-CHE/FB rows 都 step fail，继续做一个全局 repair：

```text
repair = amortized refresh
fms_update_interval = 200
methods = C0..C4 + F-CHE3 + F-CHE7
fallbacks = FB2 + FB3
```

合法性：

```text
1. 没有新增 F-CHE8。
2. 没有 action token。
3. 没有 controller。
4. 没有按 task / seed / dataset branch。
5. 没有使用 LineC/tail/AUC/calibration 生成方向。
```

repair route：

```text
route = R1-DCHESyntheticFMSFail
minimum_success = S0-ExecutionCompleteNoPromotion
synthetic_task_family_pass_count = 1
synthetic_task_pass = {X1: 1, X2: 0, X3: 0, X4: 0, X5: 0}
dche_substrate_official_fms_eligibility = 1
dche_best_source_vs_adamw = 0.13746285438537598
mlp_best_source_vs_adamw = 0.2882530689239502
generic_fms_confound = 1
promotion_allowed = 0
```

repair method summary：

| method | rows | strict pass | task pass | mean source | median AUC | median step | median memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-CHE3 | 42 | 2 | 0 | -0.033866 | 1.016605 | 1.158379 | 1.093692 |
| F-CHE7 | 42 | 2 | 0 | -0.028834 | 1.015783 | 1.159916 | 1.093692 |
| FB2 | 42 | 1 | 0 | -0.027454 | 1.013073 | 1.166259 | 1.093692 |
| FB3 | 42 | 2 | 0 | -0.031159 | 1.018443 | 1.165722 | 1.093692 |

strict pass rows in repair：

| task | seed | loss | method | source | AUC | CEp99 delta | NLL delta | ECE delta | step |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| X1 | 0 | Brier | F-CHE3 | 0.031314 | 0.990243 | -0.153654 | -0.033990 | -0.040904 | 1.155331 |
| X1 | 0 | Brier | F-CHE7 | 0.028959 | 0.989640 | -0.221328 | -0.031636 | -0.007445 | 1.171381 |
| X2 | 0 | Brier | F-CHE3 | 0.016147 | 0.985890 | -0.208917 | -0.043744 | 0.010724 | 1.157811 |
| X5 | 0 | CE | F-CHE7 | 0.005945 | 0.987589 | -0.169503 | -0.049598 | -0.037171 | 1.140672 |
| X1 | 2 | CE | FB2 | 0.010902 | 0.996242 | -0.382439 | -0.075444 | -0.041425 | 1.164534 |
| X3 | 2 | Brier | FB3 | 0.010864 | 0.980728 | -0.270859 | -0.063978 | -0.079098 | 1.155708 |
| X4 | 0 | CE | FB3 | 0.021979 | 0.996016 | -0.220342 | -0.052875 | -0.001492 | 1.156243 |

repair failure decomposition：

```text
source fail rows = 142
AUCtime fail rows = 141
CEp99 fail rows = 72
NLL fail rows = 35
ECE fail rows = 31
step fail rows = 0
LineC fail rows = 0
```

判断：

```text
1. amortized refresh 修掉 step blocker。
2. 但 source/AUC/tail 仍然主导失败。
3. strict pass rows 只形成 X1 一个 task-family pass，不满足 S3 >=5/7。
4. repair 后 D-CHE best source 仍低于 MLP best source，generic_fms_confound 仍为 1。
5. 不能打开 real 3x3。
```

## 6. All-basis substrate 状态

本轮 v14.10 runner 读取 v14.9 hardening20 gatefix LineC3 substrate artifact：

```text
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix_linec3/
```

写入：

```text
v1410_all_basis_substrate_status.csv
v1410_fourier_substrate_hardening.csv
v1410_rbf_substrate_hardening.csv
v1410_wavelet_substrate_hardening.csv
```

状态：

| family | dataset-seed pass | official FMS eligible | exploration | best candidate | best mean delta | best LineC | min NLL ratio | median step |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| D-CHE | 9 | 1 | 1 | D-CHE24-HighDegreeLateEnable | 0.033203 | 1.000000 | 0.477970 | 0.251653 |
| D-FOU | 6 | 0 | 1 | D-FOU26-NoMaterializeLifetimeAuditV2 | 0.029297 | 1.000000 | 0.481892 | 0.373275 |
| D-RBF | 6 | 0 | 1 | D-RBF25-WidthConditionGuardNoTaskBranch | 0.021484 | 1.000000 | 0.494566 | 0.503058 |
| D-WAV | 2 | 0 | 0 | D-WAV24-ScaleOccupancyHardeningV2 | 0.021484 | 0.666667 | 1.298239 | 0.505166 |

判断：

```text
D-CHE 是唯一 official FMS eligible substrate；
D-FOU/D-RBF 仍只是 exploration；
D-WAV 仍弱；
本轮未对 D-FOU/D-RBF/D-WAV 执行 official FMS proof。
```

## 7. Required artifacts

official required manifest：

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/official_v1410/v1410_required_artifact_manifest.csv
missing_required_rows = 0
lines including header = 36
```

repair required manifest：

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_amortized_interval200_v1410/v1410_required_artifact_manifest.csv
missing_required_rows = 0
lines including header = 36
```

主要产物：

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/smoke_v1410/
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/official_v1410/
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_amortized_interval200_v1410/
```

## 8. 最终科学结论

v14.10 没有达成 D-CHE synthetic S3，也没有打开 real 3x3，更没有达成 S5。

最终合法判断：

```text
route = R1-DCHESyntheticFMSFail
minimum_success = S0-ExecutionCompleteNoPromotion
official_s5_reached = 0
promotion_allowed = 0
real 3x3 executed = 0
```

已闭合事实：

```text
1. D-CHE substrate eligibility 仍成立：9/9，official_fms_eligibility = 1。
2. v14.10 official D-CHE FMS synthetic proof 失败：0/7 task families。
3. Case A FB1/FB2/FB3 已执行，没有打开 S3。
4. step-time blocker 可由全局 amortized refresh 修掉。
5. 修掉 step 后仍只有 1/7 task-family pass。
6. 剩余 blocker 是 source instability + AUCtime + tail。
7. LineC 不是本轮 blocker。
8. MLP/generic source 上界高于 D-CHE source，上报 generic_fms_confound = 1。
9. required artifacts 缺失为 0，forbidden information violation 为 0。
10. compute_budgeted_run = 1，因此 promotion_allowed 必须为 0。
```

no-go boundary：

```text
1. 不允许把 D-CHE substrate eligibility 写成 FMS success。
2. 不允许把 repair 后 X1 局部 positive 写成 S3。
3. 不允许打开 real 3x3，因为 synthetic S3 未过。
4. 不允许 promotion。
5. 不允许在 v14.10 内临时新增 F-CHE8 / action token / controller / reset route。
```

下一步需要新的 FMS definition，而不是继续小修：

```text
当前 D-CHE degree-FMS variants 和 FB1/FB2/FB3 都没有形成跨 X1..X7 的 causal value。
如果继续，应在下一版预注册新的 train-stream-only value definition，
同时明确如何同时解决 source instability、AUCtime 与 CEp99/NLL/ECE tail，
并保留 MLP/generic confound control。
```

## 9. 用户再次要求继续后的 candidate / memory / S3 repair

### 9.1 strength / interval sensitivity

在 `repair_amortized_interval200_v1410` 基础上测试：

```text
fms_strength = 0.02
fms_strength = 0.10
fms_update_interval = 120
```

结果：

```text
strength 0.02 / 0.10:
  route = R1-DCHESyntheticFMSFail
  synthetic_task_family_pass_count = 1
  pass = X1 only
  failure counts = source 142, AUCtime 141, CEp99 72, NLL 35, ECE 31, step 0, LineC 0

interval120:
  route = R1-DCHESyntheticFMSFail
  synthetic_task_family_pass_count = 0
  dche_best_source_vs_adamw = 0.1368476152420044
  mlp_best_source_vs_adamw = 0.28802919387817383
  generic_fms_confound = 1
```

判断：

```text
1. FMS amplitude 不是主 blocker。
2. 更频繁 refresh 没有修好 source/AUC/tail。
```

### 9.2 D-CHE candidate mismatch 修复

复核 v14.9 substrate artifact 后，发现 v14.10 初始 runner 使用的 D-CHE20 不是 v14.9 D-CHE official eligibility 的 base candidate。

代码修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  DEFAULT_D_CHE_CANDIDATE 改为 D-CHE17-HighDegreeLateEnableSubstrate。
  新增 --dche-candidate。
```

D-CHE17 seed0 batch32 结果：

```text
route = R1-DCHESyntheticFMSFail
synthetic_task_family_pass_count = 0
dche_best_source_vs_adamw = 0.3820805549621582
mlp_best_source_vs_adamw = 0.27991271018981934
generic_fms_confound = 0
```

判断：

```text
D-CHE17 比 D-CHE20 出现更强 D-CHE source 且不再被 MLP best source 解释；
但 batch32 materialized per-example gradient 的 memory ratio 约 1.773，超过 1.25。
```

### 9.3 batch sensitivity

结果：

| run | route | synthetic pass | key blocker |
|---|---|---:|---|
| D-CHE17 seed0 batch8 | R1 | 0/7 | memory fixed but source weaker |
| D-CHE17 seed012 batch8 | R1 | 2/7 | pass X5/X6 only |
| D-CHE17 seed012 batch8 interval80 | R1 | 0/7 | more refresh worse |
| D-CHE17 seed0 batch16 | R1 | 0/7 | memory ratio about 1.393 > 1.25 |

判断：

```text
batch8 能修 memory 但削弱 source；
batch16 / batch32 source 更强但 memory fail；
需要 memory repair，而不是单纯改 batch。
```

### 9.4 streaming per-example gradient memory repair

代码修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 collect_streaming_per_example_summary()。
  新增 --streaming-per-example-gradients。
  refresh 时流式累计 mean / variance / utilities，
  不再 materialize batch x params gradient matrix。
```

合法性：

```text
1. direction 仍只来自 current train batch supervised loss gradient。
2. 不使用 validation/test/future/query。
3. 不使用 LineC/CEp99/NLL/ECE/AUCtime 作方向。
```

结果：

| run | route | synthetic task pass | dche best source | generic confound | key |
|---|---|---:|---:|---:|---|
| seed0 streaming b32 selected | R1 | 0/7 | 0.3820803165435791 | 0 | memory median 1.012766 |
| seed012 streaming b32 selected | R1 | 3/7 | 0.3820803165435791 | 0 | pass X1/X3/X7 |
| seed012 streaming all-FCHE/FB 200-step | R1 | 4/7 | 0.4170483350753784 | 0 | pass X1/X3/X5/X7 |

判断：

```text
streaming per-example gradient 是有效 memory repair；
它把 D-CHE17 batch32 从 memory fail 修到 memory pass，
并把 synthetic 从 0/7 推到 3/7，再到 all-FCHE/FB 的 4/7。
```

### 9.5 synthetic 300-step S3

执行设置：

```text
D-CHE17-HighDegreeLateEnableSubstrate
synthetic_tasks = X1..X7
synthetic_seeds = 0,1,2
loss_interfaces = CE,Brier
methods = C0..C4 + F-CHE1..7
fallback_methods = FB1..FB3
train_steps = 300
batch_size = 32
fms_update_interval = 200
streaming_per_example_gradients = 1
skip_real = 1
compute_budgeted_run = 1
```

route：

```text
route = S3-DCHESyntheticFMSPass
minimum_success = S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count = 5
synthetic_task_pass = {X1:1, X2:0, X3:1, X4:1, X5:1, X6:0, X7:1}
dche_best_source_vs_adamw = 0.4909071922302246
mlp_best_source_vs_adamw = 0.3354611396789551
generic_fms_confound = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | rows | strict rows | dataset-seed pass | mean source | median AUCtime | median step | median memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-CHE1 | 42 | 4 | 3 | -0.083202 | 1.039167 | 1.196613 | 1.012766 |
| F-CHE2 | 42 | 2 | 2 | -0.115204 | 1.032649 | 1.196508 | 1.012766 |
| F-CHE3 | 42 | 9 | 3 | -0.041894 | 1.016937 | 1.196305 | 1.012766 |
| F-CHE4 | 42 | 0 | 0 | -0.120973 | 1.035280 | 1.195642 | 1.012766 |
| F-CHE5 | 42 | 0 | 0 | -0.090073 | 1.033341 | 1.196260 | 1.012766 |
| F-CHE6 | 42 | 1 | 1 | -0.091388 | 1.033835 | 1.195497 | 1.012766 |
| F-CHE7 | 42 | 0 | 0 | -0.084404 | 1.031702 | 1.195831 | 1.012766 |
| FB1 | 42 | 1 | 1 | -0.113363 | 1.030213 | 1.191993 | 1.012766 |
| FB2 | 42 | 1 | 1 | -0.068447 | 1.029777 | 1.192991 | 1.012766 |
| FB3 | 42 | 3 | 2 | -0.078308 | 1.023079 | 1.193221 | 1.012766 |

strict pass rows：

| task | seed | loss | method | source | LineC |
|---|---:|---|---|---:|---:|
| X1 | 0 | CE | F-CHE1 | 0.133960 | 1 |
| X1 | 0 | CE | F-CHE3 | 0.091831 | 1 |
| X1 | 0 | Brier | F-CHE3 | 0.490907 | 1 |
| X1 | 2 | CE | F-CHE3 | 0.046433 | 1 |
| X2 | 1 | CE | F-CHE1 | 0.280351 | 1 |
| X3 | 0 | CE | F-CHE3 | 0.091029 | 1 |
| X3 | 1 | CE | F-CHE3 | 0.180582 | 1 |
| X3 | 1 | Brier | F-CHE3 | 0.193098 | 1 |
| X4 | 2 | Brier | F-CHE1 | 0.111721 | 1 |
| X5 | 1 | CE | F-CHE2 | 0.023798 | 1 |
| X7 | 0 | CE | F-CHE2 | 0.081136 | 1 |
| X7 | 0 | CE | F-CHE3 | 0.103724 | 1 |
| X7 | 0 | Brier | F-CHE6 | 0.271424 | 1 |
| X7 | 1 | CE | F-CHE3 | 0.165380 | 1 |
| X7 | 1 | Brier | F-CHE1 | 0.088956 | 1 |
| X7 | 2 | CE | F-CHE3 | 0.105831 | 1 |
| X1 | 0 | CE | FB3 | 0.126133 | 1 |
| X1 | 2 | CE | FB3 | 0.095397 | 1 |
| X4 | 0 | CE | FB3 | 0.045198 | 1 |
| X5 | 0 | Brier | FB1 | 0.019689 | 1 |
| X6 | 0 | Brier | FB2 | 0.054800 | 1 |

failure counts：

```text
source = 364
AUCtime = 361
CEp99 = 195
NLL = 167
ECE = 84
LineC = 26
step = 3
memory = 0
```

判断：

```text
1. v14.10 首次达到 D-CHE synthetic S3。
2. S3 来自 D-CHE17 candidate alignment + streaming memory repair + all F-CHE/FB + 300-step window。
3. X2/X6 仍没有形成 task-family pass。
4. 这是 real gate open，不是 S5，也不是 promotion。
```

## 10. Real 3x3 transfer after S3

为避免把 real 绑定到 synthetic 的 300-step，新增 `--synthetic-source-dir` 读取已通过的 S3 artifact，再按计划 real 200/400 confirmation 执行。

### 10.1 real 200-step interval80

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
synthetic_task_family_pass_count = 5
real_dataset_seed_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

failure decomposition：

```text
source_fail = 66
AUC_fail = 63
CEp99_fail = 25
NLL_fail = 12
ECE_fail = 24
LineC_fail = 48
step_fail = 80
memory_fail = 0
control_equivalent = 0
```

判断：

```text
interval80 与 synthetic S3 protocol 不一致，并带来明显 step-time blocker。
```

### 10.2 real 200-step interval200 protocol alignment

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 2
strict unique dataset-seeds = KMNIST seed0, KMNIST seed2
required_artifact_missing_count = 0
promotion_allowed = 0
```

strict rows：

| dataset | seed | method | source vs AdamW | source vs best control | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | step | memory |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KMNIST | 0 | FB1 | 0.044367 | 0.019065 | 0.989251 | -0.309881 | -0.044367 | -0.017968 | 1 | 1.128855 | 1.024638 |
| KMNIST | 0 | FB2 | 0.050539 | 0.025236 | 0.980490 | -0.021430 | -0.050539 | -0.003587 | 1 | 1.091052 | 1.024638 |
| KMNIST | 2 | F-CHE3 | 0.043657 | 0.031367 | 0.985806 | -0.204150 | -0.043657 | 0.000631 | 1 | 1.087061 | 1.024638 |
| KMNIST | 2 | FB3 | 0.031847 | 0.019557 | 0.985845 | 0.047713 | -0.031847 | -0.014798 | 1 | 1.094837 | 1.024638 |

failure decomposition：

```text
source_fail = 66
AUC_fail = 62
CEp99_fail = 25
NLL_fail = 12
ECE_fail = 24
LineC_fail = 48
step_fail = 0
memory_fail = 0
control_equivalent = 0
```

判断：

```text
1. protocol alignment 修复 step blocker，并将 real 从 0/9 提到 2/9。
2. 仍未达到 S4 的 6/9。
3. real failures 现在主要来自 source_vs_best_control / AUC / tail / LineC，而非 step/memory。
4. pass 集中在 KMNIST seed0/seed2，不可写成 general real transfer。
```

### 10.3 real 400-step confirmation

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 1
promotion_allowed = 0
```

判断：

```text
400-step confirmation 没有改善 real transfer，反而从 2/9 降到 1/9；
因此 blocker 不是单纯训练窗口不够长。
```

## 11. Case B real-transfer fallback RT1/RT2/RT3

计划 Case B 允许：

```text
F-CHE-RT1-TrainSplitAgreement
F-CHE-RT2-ValueRetentionTrust
F-CHE-RT3-DegreeEnergySafetyProjection
```

代码修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 REAL_TRANSFER_METHODS。
  RT1: current train batch split agreement 缩放 FMS scale。
  RT2: train-gradient generic/projected value retention blend。
  RT3: degree-energy safety projection。
  no-action audit 标记 RT1/RT2/RT3 为 pre-registered method set。
```

合法性：

```text
1. RT1/RT2/RT3 direction 只使用 train-stream loss gradient、current model parameter role / degree state。
2. 不使用 validation/test/future/query。
3. 不使用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 生成方向。
4. 不新增 action token，不启动 controller，不使用 reset route。
```

RT run 结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

RT summary：

| method | rows | strict pass | mean source vs best control | median AUCtime | median step | median memory |
|---|---:|---:|---:|---:|---:|---:|
| RT1-TrainSplitAgreement | 9 | 0 | -0.008865 | 1.002341 | 1.083551 | 1.072877 |
| RT2-ValueRetentionTrust | 9 | 0 | -0.029664 | 1.009020 | 1.081008 | 1.024638 |
| RT3-DegreeEnergySafetyProjection | 9 | 0 | -0.007735 | 1.010209 | 1.080086 | 1.024638 |

RT top rows：

| dataset | seed | method | source vs best control | source vs AdamW | AUCtime | CEp99 delta | ECE delta | LineC | pass |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| KMNIST | 1 | RT3 | 0.037155 | 0.047428 | 0.990453 | 0.170992 | 0.030777 | 1 | 0 |
| MNIST | 1 | RT3 | 0.019072 | 0.062697 | 0.955080 | -0.028061 | -0.005602 | 0 | 0 |
| KMNIST | 1 | RT1 | 0.014709 | 0.024981 | 0.980546 | 0.576832 | -0.009235 | 1 | 0 |

RT failure counts：

```text
source_fail = 24
AUC_fail = 20
CEp99_fail = 10
NLL_fail = 3
ECE_fail = 9
LineC_fail = 11
step_fail = 0
memory_fail = 0
control_equivalent = 0
```

判断：

```text
1. RT1/RT2/RT3 没有打开 real transfer。
2. RT 的 step/memory 都过，但 source_vs_best_control、AUC、CEp99/ECE 与 LineC 仍混合阻断。
3. RT3 在 KMNIST seed1 有 source/AUC/LineC，但 CEp99/ECE fail。
4. RT1/RT2/RT3 不能写成 S4，更不能写成 S5。
```

## 12. 当前最终状态

最新 best synthetic artifact：

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410/v1410_route_decision.json
route = S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count = 5
promotion_allowed = 0
```

最新 best real artifact：

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps200_interval200_v1410/v1410_route_decision.json
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 2
promotion_allowed = 0
```

最终合法判断：

```text
v14.10 已达成 S3-DCHESyntheticFMSPass；
real 3x3 已合法打开并执行；
但 real transfer 未达 S4：
  best real_dataset_seed_pass_count = 2/9
S5 未达成：
  official_s5_reached = 0
promotion_allowed = 0
```

当前 no-go boundary：

```text
1. 不允许把 synthetic S3 写成 real success。
2. 不允许把 KMNIST seed0/seed2 局部 real pass 写成 S4。
3. 不允许把 RT top rows 写成 transfer success。
4. 不允许降低 source_vs_best_control / AUCtime / tail / LineC / step / memory gates。
5. 不允许使用 validation/test/LineC/tail audit 设计下一步 direction。
6. 不允许新增 action token / controller / reset route。
```

当前我已经按计划推进到：

```text
candidate alignment、
memory repair、
batch sensitivity、
full all-FCHE/FB synthetic S3、
real 200/400 confirmation、
Case B RT1/RT2/RT3 fallback。
```

在当前 v14.10 计划内，我现在不确定如何继续安全推进到 S4/S5，而不把 real fail 的 audit metric、dataset/seed pattern、或 control-comparison结果用作 direction。下一步需要新的 real-transfer value definition 计划，核心是 train-stream-only 地同时解决 source_vs_best_control、AUCtime、CEp99/ECE tail 与 LineC transfer。

## 13. 用户再次要求继续后的 post-plan RT4 composite repair

本节是在计划内 RT1/RT2/RT3 失败后执行的 post-plan diagnostic/repair。它不作为 promotion 依据，也不降低任何 gate。

### 13.1 代码修改

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 F-CHE-RT4-CompositeTransferTrust。
```

RT4 机制：

```text
1. 使用 current train batch split-gradient agreement。
2. 使用 train-gradient value retention trust。
3. 使用 degree-energy safety projection。
4. direction source 仍只来自 current train-stream supervised loss gradient。
5. 不使用 validation/test/future/query。
6. 不使用 LineC/CEp99/NLL/ECE/AUCtime/Brier 生成方向。
7. 不新增 action token / controller / reset route。
```

语法检查：

```text
py_compile pass
```

### 13.2 RT4 strength 0.05

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
minimum_success = S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count = 5
real_dataset_seed_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

RT4 summary：

| method | rows | strict pass | mean source vs best control | median AUCtime | median step | median memory |
|---|---:|---:|---:|---:|---:|---:|
| F-CHE-RT4-CompositeTransferTrust | 9 | 0 | -0.012427 | 1.000852 | 1.081033 | 1.072877 |

RT4 rows：

| dataset | seed | source vs best control | source vs AdamW | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KMNIST | 1 | 0.033159 | 0.043432 | 0.975165 | 0.587616 | -0.043432 | 0.000106 | 1 | 0 |
| Fashion-MNIST | 2 | 0.009919 | 0.086658 | 1.004848 | -0.679142 | -0.086658 | 0.000370 | 0 | 0 |
| KMNIST | 0 | 0.004710 | 0.030012 | 0.998141 | -0.383439 | -0.030012 | -0.002495 | 1 | 0 |
| Fashion-MNIST | 0 | -0.010917 | 0.020501 | 1.000852 | 0.166009 | -0.020501 | -0.007274 | 1 | 0 |
| MNIST | 0 | -0.017743 | 0.003531 | 1.010349 | 0.346716 | -0.003531 | 0.030791 | 1 | 0 |
| Fashion-MNIST | 1 | -0.019551 | -0.005549 | 0.996883 | -0.115746 | 0.005549 | 0.016640 | 0 | 0 |
| MNIST | 1 | -0.025550 | 0.018075 | 0.991423 | 0.063403 | -0.018075 | 0.017930 | 0 | 0 |
| MNIST | 2 | -0.038014 | 0.020839 | 1.014899 | -0.124096 | -0.020839 | 0.041912 | 0 | 0 |
| KMNIST | 2 | -0.047851 | -0.035561 | 1.022862 | -0.086472 | 0.035561 | 0.014523 | 1 | 0 |

failure counts：

```text
source_fail = 7
AUC_fail = 5
CEp99_fail = 4
NLL_fail = 1
ECE_fail = 2
LineC_fail = 4
step_fail = 0
memory_fail = 0
control_equivalent = 0
```

判断：

```text
1. RT4 没有打开 real transfer。
2. KMNIST seed0 接近 source gate，但 source_vs_best_control = 0.004710 < 0.005，不能算 pass。
3. KMNIST seed1 有 source/AUC/LineC，但 CEp99 fail。
4. step/memory 不再是 blocker。
```

### 13.3 RT4 strength 0.10

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

判断：

```text
RT4 strength 0.10 与 0.05 的 row-level 结果几乎一致；
说明 RT4 的 composite trust / safety clamp 主导更新，
FMS strength 不是有效杠杆。
```

### 13.4 最终判断更新

当前 best synthetic 仍为：

```text
route = S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count = 5/7
```

当前 best real 仍为：

```text
route = R2-DCHESyntheticDoesNotTransfer
best real_dataset_seed_pass_count = 2/9
```

最终判断不变：

```text
S3 已达成；
S4 未达成；
S5 未达成；
promotion_allowed = 0。
```

我现在已经尝试了：

```text
1. 计划内 Case B RT1/RT2/RT3。
2. post-plan RT4 composite trust。
3. RT4 global strength sensitivity。
```

这些都没有把 real transfer 推到 6/9。继续在 v14.10 内推进会很可能变成：

```text
1. 按 dataset/seed fail pattern 调方向；
2. 用 CEp99/ECE/LineC/AUCtime audit 指标设计方向；
3. 或新增 action/controller/reset route。
```

这些都违反当前计划边界。因此当前停止点仍为：

```text
route = R2-DCHESyntheticDoesNotTransfer
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0
```

## 14. 用户再次追问后的最终状态复核

本次没有新增代码修改，也没有新增训练；只复核最新 artifact，确认 v14.10 是否已经达成目标。

复核对象：

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410/v1410_route_decision.json
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps200_interval200_v1410/v1410_route_decision.json
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt123_steps200_interval200_v1410/v1410_route_decision.json
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt4_composite_steps200_interval200_v1410/v1410_route_decision.json
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt4_strength010_steps200_interval200_v1410/v1410_route_decision.json
```

复核结论：

```text
v14.10 没有达成最终目标。

best synthetic:
  route = S3-DCHESyntheticFMSPass
  synthetic_task_family_pass_count = 5/7

best real:
  route = R2-DCHESyntheticDoesNotTransfer
  real_dataset_seed_pass_count = 2/9

official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
generic_fms_confound = 0
```

已尝试但未打开 S4/S5：

```text
1. v14.9 D-CHE substrate candidate 对齐。
2. streaming per-example gradient memory repair。
3. D-CHE17 synthetic all-FCHE/FB 300-step S3。
4. real 3x3 200-step / 400-step transfer。
5. 计划内 Case B:
   F-CHE-RT1-TrainSplitAgreement
   F-CHE-RT2-ValueRetentionTrust
   F-CHE-RT3-DegreeEnergySafetyProjection
6. post-plan diagnostic:
   F-CHE-RT4-CompositeTransferTrust
7. RT4 strength sensitivity。
```

当前停止边界：

```text
我现在已经不确定如何在 v14.10 当前计划内继续安全推进到 S4/S5，
而不把 real dataset/seed fail pattern 用作方向选择、
不把 CEp99/ECE/LineC/AUCtime audit metric 变成方向源、
不新增 action/controller/reset route、
也不把 2/9 real transfer positive 写成 official success。

因此本次不再启动新的 v14.10 训练。
当前合法结论仍是：
  minimum_success = S3-DCHESyntheticFMSPass
  route = R2-DCHESyntheticDoesNotTransfer
  official_s5_reached = 0
  promotion_allowed = 0
```
