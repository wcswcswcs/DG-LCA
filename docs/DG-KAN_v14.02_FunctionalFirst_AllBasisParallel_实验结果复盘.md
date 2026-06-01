# DG-KAN v14.2 FunctionalFirst AllBasisParallel 实验结果复盘

生成时间：2026-05-29（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 smoke / compute-budgeted repair / generic MLP-FMS success 写成 KAN promotion。

## 1. 计划理解

v14.2 的目标不是继续 cover-objective 或 A-RCF 小修，而是执行：

```text
functional-first persistent metric state (FMS)
+ all-basis parallel substrate map / repair boundary
+ KAN-specific synthetic proof gate。
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 只能使用 train-stream gradient / parameter / group / basis telemetry。
5. LineC / CEp99 / NLL / ECE 只能作为 audit/gate，不能生成方向。
6. real 3x3 short-run 只能在 synthetic gate 达成后打开。
7. MLP-FMS 只能证明 generic functional control，不能写成 KAN-specific promotion。
```

v14.2 gate 关键点：

```text
S/F/H 三层：
S = basis substrate efficient / healthy map。
F = persistent Functional Metric State，不是一阶一次性 perturbation。
H = healthy official base 与 synthetic >=5/7 gate。
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v142_functional_first_all_basis_parallel.py
```

主要实现：

```text
1. 新增 v14.2 runner 与 artifact surface。
2. 实现 F0/F1/F2/F3/F4/F5/F7 与 random matched control：
   F0-AdamWParallel
   F1-PriorSNRReference
   F2-ParameterFMS
   F3-LayerFMS
   F4-RoleFMS
   F5-BasisGroupFMS
   F7-PhaseScheduleFMS
   FCTRL-RandomMatchedNorm
3. FMS 使用 train-stream per-example gradient telemetry 建立 persistent state；
   route / artifacts 记录 direction_uses_validation_test_future_query = 0。
4. 输出 v142 required artifacts 与 figures。
5. basis substrate status 读取 v12.35 official audited substrate-health map，并按 v14.2 threshold 重新判定。
6. Non-RAT substrate gate fail 时 fail-closed，不进入 FMS proof。
7. 不改变 synthetic / LineC / tail / promotion gate。
```

后续 blocker 修复：

```text
1. repo root sys.path 修复，保证直接运行 experiments/runner 可 import dgkan。
2. 默认 synthetic_dim 改为 16，避免 D-RAT30 在 dim=12 下 shape mismatch。
3. 新增 --fms-update-interval，实现 amortized persistent FMS update。
4. 新增 --rational-candidate，用于 Rational focused repair。
```

合法性说明：

```text
1. FMS direction 只使用当前 train batch loss gradient / parameter group telemetry。
2. CEp99 / NLL / ECE / LineC 只作为 gate 和 audit。
3. MLP-FMS positive 不写成 KAN-specific success。
4. compute-budgeted repair 不替代 official success。
5. promotion_allowed 始终为 0。
```

## 3. Smoke

执行规模：

```text
synthetic_tasks = X1
synthetic_seeds = 0
loss_interfaces = CE
methods = F0,F3,FCTRL
train_steps = 4
batch_size = 8
synthetic_dim = 16
skip_nonrat_fms = 1
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
minimum_success = S1-SubstrateMapWithFMSExecuted
required_artifact_missing_count = 0
basis_substrate_pass_count = 1
healthy_base_pass_count = 1
mlp_fms_task_pass_count = 0
rational_fms_task_pass_count = 0
promotion_allowed = 0
```

解释：

```text
smoke 只证明 runner / artifact / FMS surface 可执行；
不能作为 promotion 或 no-go official 证据。
```

## 4. Official v14.2 compute-budgeted 结果

执行规模：

```text
synthetic_tasks = X1..X7
synthetic_seeds = 0
loss_interfaces = CE,Brier
methods = F0,F1,F2,F3,F4,F5,F7,FCTRL
train_steps = 20
batch_size = 8
synthetic_dim = 16
skip_nonrat_fms = 1
compute_budgeted_run = 1
```

最终 route：

```text
route = R4-FMSNoGoCurrentDefinition
minimum_success = S1-SubstrateMapWithFMSExecuted
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
basis_substrate_pass_count = 1
healthy_base_pass_count = 1
mlp_fms_task_pass_count = 0
rational_fms_task_pass_count = 0
nonrat_fms_task_pass_count_total = 0
real_short_run_open_allowed = 0
```

Official substrate map：

| family | candidate | substrate pass | healthy pass | blocker |
|---|---|---:|---:|---|
| D-RAT | D-RAT30-LowMemoryTelemetryStrong | 1 | 1 | pass |
| D-RBF | D-RBF11-CompactExpressionRepair-Monitor | 0 | 0 | step/memory/mean/worst/AUC |
| D-CHE | D-CHE20-DegreeNormalizedReadoutHealthSubstrate | 0 | 0 | source row missing / telemetry fail |
| D-FOU | D-FOU20-LowFreqIdentityResidualHealthSubstrate | 0 | 0 | source row missing / telemetry fail |
| D-WAV | D-WAV16-SupportStableHatHealthSubstrate | 0 | 0 | source row missing / telemetry fail |

Official best local rows：

| family | task | loss | method | source_vs_best_control | AUC ratio | CEp99 delta | LineC | pass |
|---|---|---|---|---:|---:|---:|---:|---:|
| MLP | X4 | Brier | F2 | 0.071219 | 0.966851 | -0.679653 | 1 | 0 |
| MLP | X1 | CE | F4 | 0.060203 | 0.985488 | -0.510178 | 1 | 0 |
| D-RAT | X7 | CE | F3 | 1.145299 | 0.945859 | -3.797529 | 1 | 0 |
| D-RAT | X6 | Brier | F4 | 0.648767 | 0.708135 | -5.464330 | 1 | 0 |

判断：

```text
1. Official FMS 有局部 source/AUC positive row，但 synthetic gate row pass = 0。
2. MLP-FMS 与 Rational-FMS task-family count 都是 0/7。
3. Rational 局部 strong-source row 仍被 overhead / LineC / tail gate 拒绝。
4. Non-RAT basis substrate gate fail-closed，未进入 FMS proof。
```

## 5. Blocker repair：amortized persistent FMS

触发原因：

```text
official 中 FMS per-example gradient update overhead 是主要 gate blocker 之一；
计划建议 MLP-FMS fail 后尝试 beta / phase schedule / layerwise variants。
```

修复内容：

```text
新增 --fms-update-interval；
FMS state 只周期性用 train-stream per-example gradient 刷新；
非刷新 step 沿用 persistent state 缩放普通 train gradient。
```

### Repair 1：interval=20

结果：

```text
route = R4-FMSNoGoCurrentDefinition
mlp_fms_task_pass_count = 4
rational_fms_task_pass_count = 0
promotion_allowed = 0
```

### Repair 2：interval=40

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
minimum_success = S2-GenericFMSPositive
mlp_fms_task_pass_count = 5
rational_fms_task_pass_count = 1
promotion_allowed = 0
real_short_run_open_allowed = 0
```

MLP pass rows 示例：

| task | loss | method | source | AUC ratio | time ratio | CEp99 delta | LineC |
|---|---|---|---:|---:|---:|---:|---:|
| X1 | Brier | F3 | 0.080528 | 0.956038 | 1.096900 | -0.539703 | 1 |
| X2 | CE | F2 | 0.043705 | 0.997507 | 1.099447 | -0.561765 | 1 |
| X4 | Brier | F2 | 0.059377 | 0.940224 | 1.090584 | -0.208636 | 1 |
| X5 | Brier | F7 | 0.047095 | 0.971612 | 1.097620 | -0.290227 | 1 |
| X7 | Brier | F3 | 0.095034 | 0.963448 | 1.086704 | 0.003328 | 1 |

判断：

```text
amortized FMS 修复证明 generic MLP-FMS 可以达到 5/7；
但 Rational-FMS 仍只有 1/7，因此不能写成 KAN-specific success。
```

## 6. Rational candidate focused repair

触发原因：

```text
MLP-FMS 达到 5/7 后，计划要求检查 Rational-specific blocker：
role signal mass / denominator safety / readout-rational separation / candidate variants。
```

执行候选：

```text
D-RAT26-TangentTrustRegionNoCE
D-RAT28-GroupDiversityPreservingRational
D-RAT35-ReadoutRationalDecoupleNoCE
D-RAT39-DenDerivativeSubstrateRepairNoCE
```

结果汇总：

| repair | route | healthy pass | MLP task pass | Rational task pass | promotion |
|---|---|---:|---:|---:|---:|
| D-RAT26 interval40 | R3 | 0 | 5 | 1 | 0 |
| D-RAT28 interval40 | R3 | 0 | 5 | 2 | 0 |
| D-RAT35 interval40 | R3 | 1 | 5 | 1 | 0 |
| D-RAT39 interval40 | R3 | 1 | 5 | 0 | 0 |

D-RAT28 最接近 rows：

| task | loss | method | source | AUC ratio | time ratio | CEp99 delta | NLL delta | ECE delta | LineC | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| X3 | CE | F2 | 0.431406 | 0.824630 | 1.046346 | -8.173300 | -1.671995 | -0.145276 | 1 | 1 |
| X7 | CE | F3 | 0.179951 | 0.878277 | 0.114821 | -1.378347 | -0.260417 | 0.045831 | 1 | 1 |
| X2 | Brier | F2 | 1.149220 | 0.846310 | 1.037177 | -0.661253 | -1.859587 | -0.091046 | 0 | 0 |
| X4 | CE | F2 | 0.356762 | 0.877933 | 1.054729 | -3.307789 | -0.711871 | -0.112252 | 0 | 0 |

判断：

```text
1. D-RAT28 是本轮 Rational focused repair 中最好结果，但只有 2/7 task pass。
2. 多个 strong-source rows 被 LineC=0 或 tail gate 拒绝，不能算 family pass。
3. D-RAT35 / D-RAT39 的 healthy substrate 更好，但没有改善 Rational task pass。
```

## 7. Reduced plasticity repair

触发原因：

```text
D-RAT28 strong-source rows 主要被 LineC/tail gate 拒绝；
按计划尝试 state over-amplification / reduced plasticity repair。
```

执行：

```text
rational_candidate = D-RAT28
fms_strength = 0.10
fms_beta = 0.99
fms_update_interval = 40
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 5
rational_fms_task_pass_count = 0
promotion_allowed = 0
```

判断：

```text
reduced plasticity 没有打开 Rational gate；
说明问题不是简单的 state scale 过强。
```

## 8. Rational R6/R7/R8 safety repair

触发原因：

```text
计划要求在 MLP-FMS 过但 Rational-FMS 不过时检查：
denominator/slope safety、delayed readout-basis coupling、phase schedule。
```

代码修改：

```text
experiments/run_v142_functional_first_all_basis_parallel.py
  新增 R6-RationalDenominatorSafetyFMS。
  新增 R7-RationalDelayedReadoutBasisFMS。
  新增 R8-RationalPhaseScheduleFMS。
```

合法性说明：

```text
R6/R7/R8 只使用 train-stream gradient 与 model-state safety scale；
LineC / CEp99 / NLL / ECE 仍只作为 audit/gate。
```

执行规模：

```text
rational_candidate = D-RAT28
methods = F0,R6,R7,R8,FCTRL
train_steps = 40
fms_beta = 0.99
fms_update_interval = 40
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
mlp_fms_task_pass_count = 3
rational_fms_task_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

最接近 rows：

| task | loss | method | source | AUC ratio | time ratio | CEp99 delta | LineC | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| X2 | Brier | R7 | 2.018056 | 0.751557 | 1.220341 | -5.358440 | 0 | 0 |
| X1 | Brier | R8 | 0.988743 | 0.863154 | 1.129525 | 2.750771 | 0 | 0 |
| X4 | Brier | R7 | 0.925906 | 0.926960 | 1.176037 | 1.281361 | 0 | 0 |
| X7 | CE | R7 | 0.121764 | 0.848242 | 1.163730 | 0.194738 | 1 | 0 |

判断：

```text
R6/R7/R8 没有打开 Rational gate；
多个 row source 很强，但仍被 LineC / tail / overhead gate 拒绝。
本轮不能写成 KAN-specific success。
```

## 9. All-basis candidate remap 与 substrate sweep

触发原因：

```text
早期 v14.2 map 中 D-CHE / D-FOU / D-WAV 使用了 v13.10 名称，
在 v12.35 source substrate map 中 source_row_found=0。
本次改用 v12.35 中实际存在的候选，并额外对全部 44 个 v12.35 substrate rows 做 v14.2 gate sweep。
```

remap 后 official_v142_remap：

```text
route = R4-FMSNoGoCurrentDefinition
basis_substrate_pass_count = 1
healthy_base_pass_count = 1
mlp_fms_task_pass_count = 0
rational_fms_task_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

remap substrate map：

| family | candidate | source row | pass | step | memory | mean delta | AUC ratio | LineC | blocker |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| D-RAT | D-RAT30 | 1 | 1 | 1.233812 | 1.349668 | 0.009983 | 1.490471 | 0.851852 | pass |
| D-RBF | D-RBF11 | 1 | 0 | 4.430183 | 3.545681 | -0.453559 | 9.024233 | 0.518519 | step/memory/task/AUC |
| D-CHE | D-CHE12 | 1 | 0 | 1.596203 | 7.009136 | -999.0 | 9.0 | 0.0 | memory/task/AUC/LineC/telemetry |
| D-FOU | D-FOU14 | 1 | 0 | 1.417649 | 4.258306 | -0.314453 | 4.390794 | 0.0 | memory/task/AUC/LineC |
| D-WAV | D-WAV11 | 1 | 0 | 1.968991 | 4.25 | -0.546875 | 10.238310 | 0.0 | step/memory/task/AUC/LineC |

全 v12.35 substrate sweep：

```text
source_rows = 44
v142_pass_rows = 11
families_with_pass = D-RAT
nonrat_families_with_pass = []
new_training_executed = 0
promotion_allowed = 0
```

输出：

```text
results/v14_2_functional_first_all_basis_parallel/substrate_sweep_v142/v142_all_basis_substrate_sweep.csv
results/v14_2_functional_first_all_basis_parallel/substrate_sweep_v142/v142_all_basis_substrate_sweep_summary.csv
results/v14_2_functional_first_all_basis_parallel/substrate_sweep_v142/v142_all_basis_substrate_sweep_route.json
```

判断：

```text
Non-RAT 不是单个候选选错导致 fail；
在当前 v12.35 audited source map 中，按 v14.2 gate 重判后仍没有 Non-RAT family pass。
因此 Non-RAT 不能进入 FMS proof。
```

## 10. Longer amortized / role-separated Rational repair

### interval=80 phase probe

触发原因：

```text
Rational strong-source rows 仍被 LineC / tail / overhead 拒绝；
继续排除 FMS 刷新频率和 phase coupling 太早的解释。
```

执行规模：

```text
rational_candidate = D-RAT28
train_steps = 80
fms_update_interval = 80
methods = F0,F2,F3,F7,R7,R8,FCTRL
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 6
rational_fms_task_pass_count = 2
required_artifact_missing_count = 0
promotion_allowed = 0
```

最接近 rows：

| task | loss | method | source | AUC ratio | time ratio | CEp99 delta | LineC | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| X1 | Brier | R8 | 1.581616 | 0.823043 | 0.986222 | -3.640163 | 0 | 0 |
| X5 | Brier | F2 | 0.723700 | 1.012911 | 1.085110 | -4.879549 | 0 | 0 |
| X3 | Brier | F3 | 0.548404 | 0.969824 | 1.213593 | -13.210636 | 0 | 0 |
| X2 | Brier | R7 | 0.503088 | 0.795939 | 1.139526 | -0.324554 | 0 | 0 |

判断：

```text
interval=80 让 MLP-FMS 提升到 6/7，但 Rational 仍只有 2/7；
主要 blocker 仍是 LineC=0 或 overhead/tail gate。
```

### R9 role-separated FMS

代码修改：

```text
experiments/run_v142_functional_first_all_basis_parallel.py
  新增 R9-RationalRoleSeparatedFMS：
  denominator/readout plasticity lower；
  numerator/basis update preserved；
  no audit metric direction。
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 6
rational_fms_task_pass_count = 1
required_artifact_missing_count = 0
promotion_allowed = 0
```

最接近 rows：

| task | loss | method | source | AUC ratio | time ratio | CEp99 delta | LineC | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| X1 | Brier | R9 | 1.477934 | 0.845233 | 1.267785 | 5.658951 | 0 | 0 |
| X3 | Brier | R9 | 1.126712 | 0.867324 | 1.087281 | -5.677471 | 0 | 0 |
| X2 | CE | R9 | 0.317657 | 0.975481 | 1.139045 | -0.487528 | 1 | 1 |

判断：

```text
R9 没有打开 Rational gate，反而从 D-RAT28 最好 2/7 降到 1/7；
role separation 不是当前缺失的充分机制。
```

## 11. Required artifacts

official_v142 manifest：

```text
manifest_rows = 26
missing_required_rows = 0
```

主要产物：

```text
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_route_decision.json
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_project_progress.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_code_path_manifest.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_loss_interface_audit.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_functional_metric_state_trace.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_per_example_gradient_stats.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_fms_update_trace.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_mlp_fms_results.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_basis_substrate_status.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_basis_family_telemetry.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_basis_fms_results.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_linec_audit.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_controls.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_failure_table.csv
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_no_go_boundary.md
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_next_hypothesis_queue.md
results/v14_2_functional_first_all_basis_parallel/official_v142/v142_code_review_packet.zip
```

repair artifacts：

```text
results/v14_2_functional_first_all_basis_parallel/smoke_v142/
results/v14_2_functional_first_all_basis_parallel/repair_v142_amortized_beta099/
results/v14_2_functional_first_all_basis_parallel/repair_v142_amortized40_beta099/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat26_amortized40/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_amortized40/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat35_amortized40/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat39_amortized40/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_lowplasticity/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_r6r7r8/
results/v14_2_functional_first_all_basis_parallel/official_v142_remap/
results/v14_2_functional_first_all_basis_parallel/substrate_sweep_v142/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_interval80_phase/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_r9_rolesep/
```

## 12. 最终科学结论

v14.2 没有达成 KAN-specific S3/S4/S5；最终合法 official route：

```text
official_v142 route = R4-FMSNoGoCurrentDefinition
best repair route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
promotion_allowed = 0
real_short_run_open_allowed = 0
```

已闭合事实：

```text
1. All-basis substrate map 已生成；只有 D-RAT 通过 v14.2 substrate gate。
2. Non-RAT families 仍因 workspace / memory / telemetry / task-health blocker fail-closed。
3. Official non-amortized FMS 没有 synthetic task pass。
4. Amortized persistent FMS 让 MLP-FMS 达到 5/7，说明 generic FMS mechanism 可见。
5. Rational focused repair 最好只有 D-RAT28 的 2/7，低于 >=5/7 KAN-specific gate。
6. D-RAT strong-source rows 主要被 LineC/tail/overhead gate 拒绝，不能写成 pass。
7. reduced plasticity、R6 denominator safety、R7 delayed readout-basis、R8 phase schedule、R9 role separation 都没有打开 Rational gate。
8. 全 v12.35 substrate sweep 说明 Non-RAT families 按 v14.2 gate 仍没有 pass。
9. 所有 run required_artifact_missing_count = 0；promotion_allowed = 0。
```

no-go boundary：

```text
1. v14.2 当前没有 KAN-specific FMS proof。
2. generic MLP-FMS positive 不能替代 Rational/KAN-specific success。
3. Rational substrate 可产生局部 strong source，但无法稳定通过 LineC/tail/family gate。
4. Non-RAT substrate 仍不能进入 FMS proof。
5. 不允许 real 3x3 short-run。
```

最终判断：

```text
v14.2 未达成目标；
不允许 promotion；
不允许 real short-run；
当前可安全执行的 v14.2 修复路径已覆盖：
MLP beta/phase/layerwise repair、amortized FMS overhead repair、
Rational candidate repair、reduced-plasticity repair、
R6/R7/R8 Rational safety repair、all-basis substrate remap 与全 substrate sweep。
Rational interval=80 phase repair、R9 role-separated repair。
继续推进需要新的 Rational LineC/tail-safe FMS mechanism 或 Non-RAT substrate repair；
不能把 generic MLP-FMS success 或局部 Rational positive row 写成 KAN success。
```

## 13. 用户再次追问后的 Rational rejection / g8-g16 复核

用户再次要求确认 v14.2 是否达成目标，若未达成则继续。本次对照计划第 13.2 节后，继续补齐：

```text
1. Rational FMS rejection audit。
2. role signal mass audit。
3. denominator safety loosen/tighten diagnostic。
4. group granularity g8/g16 availability audit。
```

代码修改：

```text
新增：
experiments/analyze_v142_rational_rejection_audit.py
experiments/analyze_v142_group_granularity_availability.py

修改：
experiments/run_v142_functional_first_all_basis_parallel.py
  新增 R6L-RationalDenominatorLooseFMS
  新增 R6T-RationalDenominatorTightFMS
  新增 rational_denominator_safety_mode = loose / tight 审计字段
```

合法性说明：

```text
1. R6L/R6T 只使用 train-stream FMS state / parameter role telemetry。
2. 不使用 validation/test/future/query batch 生成方向。
3. 不使用 LineC / CEp99 / NLL / ECE 生成方向。
4. rejection / group granularity audit 只读取既有 artifact，不新增训练。
5. 不降低 synthetic / LineC / tail / promotion gate。
```

Rational rejection audit：

```text
diagnostic_route = D2-RationalLineCTailOverheadRejectedLocalSource
official_route_unchanged = R4-FMSNoGoCurrentDefinition
best_repair_route_observed = R3-GenericFunctionalOnlyKANSpecificNotEstablished
uses_existing_artifacts_only = 1
new_training_executed = 0
promotion_allowed = 0
total_rational_rows_audited = 336
synthetic_gate_pass_rows = 7
source_positive_rejected_rows = 56
linec_rejection_rows = 154
tail_rejection_rows = 129
step_time_rejection_rows = 85
```

R6 loosen/tighten 执行规模：

```text
candidate = D-RAT28-GroupDiversityPreservingRational
tasks = X1..X7
seeds = 0
losses = CE,Brier
methods = F0,R6L,R6T,FCTRL
train_steps = 80
batch_size = 8
compute_budgeted_run = 1
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
minimum_success = S2-GenericFMSPositive
mlp_fms_task_pass_count = 5
rational_fms_task_pass_count = 2
required_artifact_missing_count = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
```

R6L/R6T 摘要：

| method | task pass | median source | max source | median AUCtime | source-positive rows |
|---|---:|---:|---:|---:|---:|
| R6L loose | 2 | -0.022265 | 1.571163 | 1.042150 | 7 |
| R6T tight | 0 | -0.090489 | 1.694592 | 0.994096 | 7 |

role signal mass：

| method | role | mean scale | mean abs state |
|---|---|---:|---:|
| R6L | denominator | 0.809601 | 0.009761 |
| R6L | readout | 1.150000 | 0.049847 |
| R6T | denominator | 0.550000 | 0.010365 |
| R6T | readout | 0.950000 | 0.051395 |

判断：

```text
1. denominator loose/tight 让局部 source positive rows 仍存在，但没有超过既有 best 2/7。
2. R6L 的 pass 集中在 X5/X7，不能达到 >=5/7。
3. R6T 降低 overhead 一些，但 task pass = 0。
4. 主要拒绝原因仍是 LineC / tail / overhead 组合，不是单纯 denominator clamp 太松或太紧。
```

Group granularity g8/g16 audit：

```text
diagnostic_route = D3-GroupGranularityG8UnavailableG16TritonCoveredNoKANSpecificFMS
official_route_unchanged = R4-FMSNoGoCurrentDefinition
best_repair_route_observed = R3-GenericFunctionalOnlyKANSpecificNotEstablished
uses_existing_artifacts_only = 1
new_training_executed = 0
promotion_allowed = 0
g8_available_in_v1235_substrate_map = 0
g16_triton_candidate_count = 17
g16_triton_v142_substrate_pass_count = 11
best_existing_g16_fms_task_pass_count = 2
```

计划 RAT-A/B/C/D 覆盖情况：

| planned variant | availability | v14.2 substrate pass | best existing FMS |
|---|---|---:|---:|
| RAT-A GroupRational-Horner-g8 | unavailable | 0 | 0 |
| RAT-B GroupRational-Horner-g16 | unavailable | 0 | 0 |
| RAT-C GroupRational-TritonEval-g8 | unavailable | 0 | 0 |
| RAT-D GroupRational-TritonEval-g16 | available | 11 | 2 |

解释：

```text
当前 v12.35 substrate registry 中 D-RAT 全部为 G16 Triton path；
没有 g8，也没有 non-triton/Horner exact variant。
因此 RAT-D 已被现有 G16 Triton candidates 覆盖但仍未达成；
RAT-A/B/C 不能在当前 v14.2 runner 中作为合法 official repair 直接运行。
强行新增 g8/Horner candidate 会进入新 substrate/base map 工作，不能混写成 v14.2 当前 proof。
```

新增产物：

```text
results/v14_2_functional_first_all_basis_parallel/rational_rejection_audit_v142/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_r6_loose_tight/
results/v14_2_functional_first_all_basis_parallel/group_granularity_availability_v142/
```

最终判断更新：

```text
v14.2 仍未达成 KAN-specific S3/S4/S5；
official route 仍为 R4-FMSNoGoCurrentDefinition；
best repair route 仍为 R3-GenericFunctionalOnlyKANSpecificNotEstablished；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

当前边界：

```text
计划第 13.2 要求的 rejection audit、role mass、role separation、denominator loosen/tighten、
group granularity g8/g16 复核均已执行或按 current substrate map fail-closed。
我现在不确定如何在当前 v14.2 runner 内继续安全推进，而不把不存在的 g8/Horner substrate、
generic MLP-FMS、或局部 Rational positive row 编造成 KAN-specific success。
下一步需要新的 Rational LineC/tail-safe FMS mechanism 或新的 substrate/base map 计划。
```

## 14. 用户再次追问后的 R10 train-stream agreement-gated safety projection

用户再次要求未达成则继续。本次重新对照计划第 13.5 节：

```text
如果 FMS 改善 task 但 LineC / tail 坏，
必须先做 audit-only failure localization / safety projection rejection / leak source /
role or basis state over-amplification audit / reduced-plasticity phase schedule；
不能用 CEp99 / LineC 作为 direction target。
```

由于 R6L/R6T 仍有 source-positive rows 被 LineC/tail/overhead 拒绝，本次新增一个只使用 train-stream per-example gradient agreement 的安全投影诊断。

代码修改：

```text
experiments/run_v142_functional_first_all_basis_parallel.py
  新增 R10-RationalAgreementGatedFMS
  group_utilities() 增加 mean_gradient_agreement / low_agreement_fraction telemetry
  R10 使用当前 train batch 的 per-example gradient agreement 降低低一致性 role/layer utility
  新增 rational_train_stream_agreement_gated 审计字段
```

合法性说明：

```text
1. R10 direction 只使用 train-stream per-example gradients。
2. 不使用 validation/test/future/query batch。
3. 不使用 LineC / CEp99 / NLL / ECE 生成方向。
4. R10 是 compute-budgeted diagnostic，不能替代 official。
5. 不降低任何 gate。
```

R10 focused probe：

```text
candidate = D-RAT28-GroupDiversityPreservingRational
tasks = X1..X7
seed = 0
losses = CE,Brier
methods = F0,R10,FCTRL
train_steps = 80
batch_size = 8
compute_budgeted_run = 1
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
minimum_success = S1-SubstrateMapWithFMSExecuted
mlp_fms_task_pass_count = 2
rational_fms_task_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

R10 longer amortized probe：

```text
candidate = D-RAT28-GroupDiversityPreservingRational
tasks = X1..X7
seed = 0
losses = CE,Brier
methods = F0,R10,FCTRL
train_steps = 160
fms_update_interval = 160
batch_size = 8
compute_budgeted_run = 1
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
minimum_success = S1-SubstrateMapWithFMSExecuted
mlp_fms_task_pass_count = 4
rational_fms_task_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

R10 摘要：

| run | task pass | median source | max source | median AUCtime | source-positive rows |
|---|---:|---:|---:|---:|---:|
| R10 interval80 | 0 | -0.478749 | 1.301358 | 1.111727 | 5 |
| R10 interval160 | 0 | -0.056391 | 3.526281 | 1.014914 | 6 |

R10 failure reason counts：

```text
LineC = 19
source_vs_best_control = 17
AUCtime_ratio = 16
NLL_tail = 14
step_time = 13
CEp99_tail = 12
ECE_tail = 2
```

R10 后重跑 Rational rejection audit：

```text
diagnostic_route = D2-RationalLineCTailOverheadRejectedLocalSource
official_route_unchanged = R4-FMSNoGoCurrentDefinition
best_repair_route_observed = R3-GenericFunctionalOnlyKANSpecificNotEstablished
total_rational_rows_audited = 420
synthetic_gate_pass_rows = 7
source_positive_rejected_rows = 67
linec_rejection_rows = 173
tail_rejection_rows = 146
step_time_rejection_rows = 98
promotion_allowed = 0
```

判断：

```text
1. R10 能产生局部 source-positive rows，但没有任何 task-family pass。
2. 160-step 摊销版仍然 rational_fms_task_pass_count = 0。
3. 主要 blocker 仍是 LineC / source instability / AUCtime / tail 的组合。
4. 这说明 train-stream gradient agreement gate 不是足够的 LineC/tail-safe Rational FMS mechanism。
5. 不允许 promotion，不允许 real short-run。
```

最终判断更新：

```text
v14.2 仍未达成 KAN-specific S3/S4/S5；
official route 仍为 R4-FMSNoGoCurrentDefinition；
best repair route 仍为 R3-GenericFunctionalOnlyKANSpecificNotEstablished；
R10 repair route = R4-FMSNoGoCurrentDefinition；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

当前边界更新：

```text
当前 v14.2 内已执行：
MLP beta/phase/layerwise repair、
amortized FMS overhead repair、
Rational candidate repair、
reduced-plasticity repair、
R6/R6L/R6T denominator safety、
R7 delayed readout-basis、
R8 phase schedule、
R9 role-separated FMS、
R10 train-stream agreement-gated safety projection、
all-basis substrate remap 与 full substrate sweep、
g8/g16 availability audit。

我现在不确定如何在当前 v14.2 runner 内继续安全推进，
而不把 audit metric 作为 direction、不把不存在的 g8/Horner substrate 写成 official repair、
或把局部 Rational source-positive row 编造成 KAN-specific success。
下一步需要新的 Rational LineC/tail-safe FMS 计划或新的 substrate/base map 计划。
```

## 15. 用户再次追问后的 Non-RAT cross-plan substrate repair audit

用户再次要求未达成则继续。本次重新对照计划第 13.4 节：

```text
如果某 basis substrate 不过，不能跑该 basis 的 FMS official proof；
只能跑 substrate repair / telemetry diagnostic / minimal smoke。
```

由于 v14.2 official all-basis sweep 没有任何 Non-RAT family 打开 substrate gate，本次没有继续 Non-RAT FMS official proof，而是新增一个跨计划读取式 substrate repair audit：

```text
experiments/analyze_v142_nonrat_crossplan_substrate_repair.py
```

该脚本只读取 v12.34.2 foreach-off Non-RAT repair artifacts，用 v14.2 strict substrate gate 重新判定 raw memory / incremental memory / step / exact no-materialize；不运行训练，不生成方向，不允许 promotion。

输出：

```text
results/v14_2_functional_first_all_basis_parallel/nonrat_crossplan_substrate_repair_v142/v142_nonrat_crossplan_route.json
results/v14_2_functional_first_all_basis_parallel/nonrat_crossplan_substrate_repair_v142/v142_nonrat_crossplan_workspace_repair.csv
results/v14_2_functional_first_all_basis_parallel/nonrat_crossplan_substrate_repair_v142/v142_nonrat_crossplan_family_summary.csv
results/v14_2_functional_first_all_basis_parallel/nonrat_crossplan_substrate_repair_v142/v142_nonrat_crossplan_functional_summary.csv
```

route 结果：

```text
diagnostic_route = D4-NonRATCrossPlanRepairStillFailsV142SubstrateGate
official_route_unchanged = R4-FMSNoGoCurrentDefinition
best_repair_route_observed = R3-GenericFunctionalOnlyKANSpecificNotEstablished
uses_existing_artifacts_only = 1
new_training_executed = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
source_v12342_workspace_rows = 108
source_v12342_workspace_gate_pass_rows = 35
source_v12342_workspace_strong_gate_pass_rows = 0
v142_strict_workspace_pass_rows = 0
nonrat_families_with_v142_strict_workspace_pass = []
source_p3_rows = 135
source_p3_executed_rows = 60
source_p3_pass_rows = 0
source_p4_rows = 60
source_p4_executed_rows = 0
source_p4_pass_rows = 0
```

family summary：

| family | workspace rows | v12.34.2 workspace pass | v14.2 strict pass | exact no-materialize rows | best raw ratio | best incremental ratio | best step ratio | P3 executed | P3 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-CHE | 27 | 0 | 0 | 27 | 0.9503426374728398 | 4.229325215626585 | 1.0121619112102047 | 0 | 0 |
| D-FOU | 45 | 12 | 0 | 45 | 0.8871469162627444 | 2.311009639776763 | 0.9094278679292677 | 60 | 0 |
| D-RBF | 18 | 18 | 0 | 0 | 0.9184355674410831 | 2.165905631659056 | 1.2743499745399467 | 0 | 0 |
| D-WAV | 18 | 5 | 0 | 0 | 0.8873474845395287 | 2.165905631659056 | 1.3325370259191072 | 0 | 0 |

判断：

```text
1. 旧 v12.34.2 foreach-off repair 中有 35 个 workspace pass rows，但按 v14.2 strict gate 重新判定后 strict pass = 0。
2. D-FOU / D-RBF / D-WAV 的 raw memory 与 step 局部接近，但 incremental_memory_ratio 仍超过 1.75。
3. D-CHE 有 exact no-materialize rows，但 incremental_memory_ratio 最低仍为 4.2293。
4. D-FOU 曾执行 P3 functional repair 60 rows，但 P3 pass = 0，不能写成 Non-RAT functional proof。
5. Non-RAT 仍不能进入 v14.2 official FMS proof。
```

最终判断更新：

```text
v14.2 仍未达成 KAN-specific S3/S4/S5；
official route 仍为 R4-FMSNoGoCurrentDefinition；
best repair route 仍为 R3-GenericFunctionalOnlyKANSpecificNotEstablished；
Non-RAT cross-plan repair strict substrate pass = 0；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

当前边界更新：

```text
当前 v14.2 内已执行：
MLP beta/phase/layerwise repair、
amortized FMS overhead repair、
Rational candidate repair、
reduced-plasticity repair、
R6/R6L/R6T denominator safety、
R7 delayed readout-basis、
R8 phase schedule、
R9 role-separated FMS、
R10 train-stream agreement-gated safety projection、
all-basis substrate remap 与 full substrate sweep、
g8/g16 availability audit、
Non-RAT cross-plan substrate repair audit。

我现在不确定如何在当前 v14.2 runner 内继续安全推进，
而不把旧 Non-RAT workspace scout、generic MLP-FMS、audit metric、
不存在的 g8/Horner substrate、或局部 Rational source-positive row 编造成 KAN-specific success。
下一步需要新的 Rational LineC/tail-safe FMS 计划或新的 substrate/base map 计划。
```
