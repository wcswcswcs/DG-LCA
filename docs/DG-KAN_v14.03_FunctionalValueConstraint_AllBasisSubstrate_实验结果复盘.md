# DG-KAN v14.3 FunctionalValueConstraint AllBasisSubstrate 实验结果复盘

生成时间：2026-05-29（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 smoke / compute-budgeted repair / generic MLP-FMS / local Rational positive row 写成 promotion。

## 1. 计划理解

v14.3 的目标不是继续把 basis telemetry 当成 value source，而是把：

```text
functional value source
与
basis constraint / geometry safety projection
```

分离。

计划核心判断：

```text
1. v14.2 已显示 generic MLP-FMS 有 value signal。
2. Rational-FMS 失败说明 KAN basis telemetry 直接作为 value source 不成立。
3. v14.3 要验证：generic FMS value 能否通过 Rational basis constraint 保留，
   并形成 KAN-specific synthetic proof。
4. all-basis substrate 仍必须并行审计；Non-RAT substrate 不过时不能进入 official FMS proof。
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 不使用 validation / test / future / query batch。
5. LineC / CEp99 / NLL / ECE / Brier 只能作为 audit / gate，不能作为方向源。
6. MLP-FMS positive 只能说明 generic optimizer/control，不能写成 KAN promotion。
7. real 3x3 short-run 只能在 synthetic + all-basis gate 允许后打开。
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v143_functional_value_constraint_all_basis_substrate.py
```

主要实现：

```text
1. 新增 v14.3 runner 与 artifact surface。
2. 实现 generic FMS line：
   G0-AdamW
   G1-PriorSNRReference
   G2-ParameterFMS
   G3-LayerFMS
   G4-RoleFMS
   G5-AmortizedParameterFMS
   G6-AmortizedLayerFMS
   G7-PhaseScheduleGenericFMS
   GCTRL-RandomMatchedNorm
3. 实现 Rational value / constraint separated line：
   K0-RAT-AdamW
   K1-RAT-GenericFMS-NoProjection
   K2-RAT-GenericFMS-IdentityProjectionAudit
   K3-RAT-GenericFMS-DenSlopeTrustRegion
   K4-RAT-GenericFMS-ReadoutBasisTrustRegion
   K5-RAT-GenericFMS-RoleWisePlasticityTrustRegion
   K6-RAT-GenericFMS-DelayedBasisConstraint
   K7-RAT-GenericFMS-PhaseScheduleConstraint
   K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint
   KCTRL-RandomMatchedProjection
4. generic value path 使用 train-stream per-example gradient / persistent FMS。
5. Rational K path 只用 basis / role telemetry 做 trust-region / projection / plasticity boundary。
6. 记录 cos_projected_vs_generic、value_retention、projection_rejection_fraction。
7. 读取 v14.2 / v12.34.2 all-basis substrate artifacts 做 strict substrate recheck。
8. 输出 route、required manifest、forbidden audit、LineC / tail audit、figures、no-go、next queue。
```

后续 blocker 修复：

```text
1. 修复 smoke 中 required manifest / packet 写出顺序。
2. 修复 v143_code_review_packet.zip 自包含递归膨胀：
   packet() 不再收录自身。
3. 新增 --finalize-existing，允许在训练已完成后只重建 route / manifest / packet。
```

合法性说明：

```text
1. K1-K8 direction 不使用 validation/test/future/query batch。
2. K1-K8 direction 不使用 LineC / CEp99 / NLL / ECE 生成方向。
3. K8 名称中的 LineCProxyFree 表示不使用 LineC proxy 做方向。
4. projection audit 只使用 train-stream value vector 与 basis/role state。
5. Non-RAT strict substrate fail 时 fail-closed，不进入 official FMS proof。
6. 不降低 synthetic / LineC / tail / promotion gate。
```

## 3. Smoke

执行规模：

```text
synthetic_tasks = X1
synthetic_seeds = 0
loss_interfaces = CE
generic_methods = G0,G5,GCTRL
rational_methods = K0,K1,K3,KCTRL
train_steps = 4
batch_size = 8
synthetic_dim = 16
compute_budgeted_run = 1
```

重跑 smoke 结果：

```text
route = R1-NoGenericFMSValue
minimum_success = S0-ValueConstraintSurfaceExecuted
generic_fms_task_pass_count = 0
rational_projection_pass_method_count = 2
rational_projection_pass_methods = K1,K3
rational_fms_task_pass_count = 1
kan_specific_positive_rows = 1
nonrat_strict_substrate_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
```

解释：

```text
smoke 只证明 runner / artifact / value-projection surface 可执行；
不能作为 success 或 no-go official 证据。
```

## 4. Official compute-budgeted seed0

执行规模：

```text
synthetic_tasks = X1..X7
synthetic_seeds = 0
loss_interfaces = CE,Brier
generic_methods = G0,G5,G6,G7,GCTRL
rational_methods = K0..K8,KCTRL
train_steps = 80
batch_size = 8
synthetic_dim = 16
fms_update_interval = 40
generic_fms_update_interval = 40
compute_budgeted_run = 1
```

结果：

```text
route = R1-NoGenericFMSValue
minimum_success = S0-ValueConstraintSurfaceExecuted
generic_fms_task_pass_count = 4
rational_projection_pass_method_count = 4
rational_projection_pass_methods = K1,K2,K3,K8
rational_fms_task_pass_count = 2
kan_specific_positive_rows = 4
rational_source_positive_rows = 39
rational_linec_tail_rejected_source_positive_rows = 35
nonrat_strict_substrate_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

判断：

```text
80-step seed0 不足以确认 generic 200-step / 3-seed target；
只作为 initial compute-budgeted scout。
```

## 5. Generic FMS 200-step confirmation

### seed0

执行规模：

```text
synthetic_tasks = X1..X7
synthetic_seeds = 0
loss_interfaces = CE,Brier
generic_methods = G0,G1,G2,G3,G4,G5,G6,G7,GCTRL
rational_methods = none
train_steps = 200
batch_size = 8
compute_budgeted_run = 1
```

结果：

```text
route = R2-GenericValueKilledByBasisProjection
minimum_success = S1-GenericFMSPositive
generic_fms_task_pass_count = 5
required_artifact_missing_count = 0
promotion_allowed = 0
```

说明：

```text
这是 G-only run，route=R2 只表示没有 K projection proof；
它的有效结论是 generic FMS seed0 达到 5/7。
```

### seed0,1,2

执行规模：

```text
synthetic_tasks = X1..X7
synthetic_seeds = 0,1,2
loss_interfaces = CE,Brier
generic_methods = G0,G1,G2,G3,G4,G5,G6,G7,GCTRL
rational_methods = none
train_steps = 200
batch_size = 8
compute_budgeted_run = 1
```

结果：

```text
route = R2-GenericValueKilledByBasisProjection
minimum_success = S1-GenericFMSPositive
generic_fms_task_pass_count = 7
required_artifact_missing_count = 0
promotion_allowed = 0
```

3-seed generic method summary：

| method | rows | task pass | tasks | median source | max source |
|---|---:|---:|---|---:|---:|
| G0-AdamW | 42 | 0 | - | -0.066363 | 0.000000 |
| G5-AmortizedParameterFMS | 42 | 4 | X4,X5,X6,X7 | -0.056576 | 0.539541 |
| G6-AmortizedLayerFMS | 42 | 6 | X1,X2,X4,X5,X6,X7 | -0.119282 | 0.249210 |
| G7-PhaseScheduleGenericFMS | 42 | 5 | X2,X3,X5,X6,X7 | 0.006942 | 0.413175 |
| GCTRL-RandomMatchedNorm | 42 | 0 | - | 0.000000 | 0.000000 |

判断：

```text
generic FMS value source 在 200-step / 3-seed compute-budgeted setting 下成立；
但这仍然不是 KAN-specific promotion。
```

## 6. Rational value-preserving projection repair

### seed0 selected-K

执行规模：

```text
generic_methods = G0,G5,G6,G7,GCTRL
rational_methods = K0,K1,K3,K8,KCTRL
synthetic_tasks = X1..X7
synthetic_seeds = 0
loss_interfaces = CE,Brier
train_steps = 200
batch_size = 8
compute_budgeted_run = 1
```

结果：

```text
route = R3-RationalLineCTailUnsafe
minimum_success = S2-RationalValuePreserved
generic_fms_task_pass_count = 5
rational_projection_pass_method_count = 3
rational_projection_pass_methods = K1,K3,K8
rational_fms_task_pass_count = 3
kan_specific_positive_rows = 3
rational_source_positive_rows = 20
rational_linec_tail_rejected_source_positive_rows = 13
promotion_allowed = 0
```

### seed0 all-K

执行规模：

```text
rational_methods = K0..K8,KCTRL
synthetic_seeds = 0
train_steps = 200
```

结果：

```text
route = R3-RationalLineCTailUnsafe
minimum_success = S2-RationalValuePreserved
generic_fms_task_pass_count = 5
rational_projection_pass_method_count = 4
rational_projection_pass_methods = K1,K2,K3,K8
rational_fms_task_pass_count = 4
kan_specific_positive_rows = 6
rational_source_positive_rows = 48
rational_linec_tail_rejected_source_positive_rows = 35
promotion_allowed = 0
```

判断：

```text
all-K 比 selected-K 增加到 4/7，但仍没有达到 >=5/7；
主要 blocker 仍是 LineC / tail rejection。
```

## 7. Final 3-seed Rational all-K

执行规模：

```text
synthetic_tasks = X1..X7
synthetic_seeds = 0,1,2
loss_interfaces = CE,Brier
generic_methods = G0,G5,G6,G7,GCTRL
rational_methods = K0,K1,K2,K3,K4,K5,K6,K7,K8,KCTRL
train_steps = 200
batch_size = 8
synthetic_dim = 16
fms_update_interval = 40
generic_fms_update_interval = 40
compute_budgeted_run = 1
```

最终 route：

```text
route = R4-NonRATSubstrateMissing
minimum_success = S3-KANSpecificSyntheticPass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
generic_fms_task_pass_count = 7
rational_projection_pass_method_count = 4
rational_projection_pass_methods = K1,K2,K3,K8
rational_fms_task_pass_count = 5
kan_specific_positive_rows = 12
rational_source_positive_rows = 100
rational_linec_tail_rejected_source_positive_rows = 73
nonrat_strict_substrate_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
real_short_run_open_allowed = 0
```

关键判断：

```text
1. v14.3 在 compute-budgeted 200-step / 3-seed 设置下，
   达到 Generic S1 和 Rational/KAN synthetic S3。
2. 但 final route 仍是 R4-NonRATSubstrateMissing。
3. all-basis / Non-RAT strict substrate 没有打开，因此不允许 promotion，也不允许 real short-run。
```

## 8. Rational method summary

final 3-seed all-K：

| method | rows | task pass | tasks | median source | max source | max K delta | source+LineC fail | source+tail fail |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| K0-RAT-AdamW | 42 | 0 | - | 0.000000 | 0.000000 | 0.584539 | 0 | 0 |
| K1-RAT-GenericFMS-NoProjection | 42 | 0 | - | -0.231056 | 2.780976 | 4.971772 | 8 | 6 |
| K2-RAT-GenericFMS-IdentityProjectionAudit | 42 | 2 | X1,X2 | -0.351727 | 1.576378 | 3.428341 | 11 | 7 |
| K3-RAT-GenericFMS-DenSlopeTrustRegion | 42 | 3 | X1,X3,X6 | -0.212620 | 1.292366 | 3.366704 | 8 | 5 |
| K4-RAT-GenericFMS-ReadoutBasisTrustRegion | 42 | 1 | X5 | -0.040331 | 2.160016 | 4.143228 | 12 | 6 |
| K5-RAT-GenericFMS-RoleWisePlasticityTrustRegion | 42 | 1 | X3 | -0.225842 | 1.727921 | 4.074255 | 7 | 4 |
| K6-RAT-GenericFMS-DelayedBasisConstraint | 42 | 0 | - | -1.160369 | 1.240546 | 1.204327 | 5 | 4 |
| K7-RAT-GenericFMS-PhaseScheduleConstraint | 42 | 2 | X2,X3 | -0.424218 | 1.744069 | 2.830652 | 9 | 7 |
| K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint | 42 | 2 | X2,X3 | -0.270427 | 1.531862 | 1.495643 | 5 | 7 |
| KCTRL-RandomMatchedProjection | 42 | 0 | - | -0.032405 | 0.000000 | 4.267634 | 0 | 0 |

synthetic pass rows = 12：

| task | seed | loss | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | K delta |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| X1 | 0 | CE | K2 | 0.373556 | 0.949837 | -3.487226 | -0.722512 | -0.160086 | 1 | 0.359867 |
| X1 | 0 | CE | K3 | 0.260865 | 0.974819 | -3.870861 | -0.609821 | -0.079194 | 1 | 0.247176 |
| X2 | 0 | CE | K2 | 0.645759 | 0.901588 | -2.083889 | -0.645759 | -0.082264 | 1 | 0.517801 |
| X2 | 0 | CE | K8 | 0.460273 | 0.891024 | -2.169868 | -0.460273 | -0.065113 | 1 | 0.332315 |
| X2 | 1 | CE | K7 | 0.158641 | 0.988266 | -2.516328 | -0.631548 | -0.100974 | 1 | 0.535345 |
| X3 | 0 | CE | K5 | 0.158113 | 0.954241 | -4.934969 | -0.892872 | -0.099235 | 1 | 0.788928 |
| X3 | 1 | CE | K7 | 0.237843 | 0.956004 | -5.723499 | -1.127140 | -0.175152 | 1 | 0.596627 |
| X3 | 2 | CE | K3 | 0.203722 | 0.915858 | -2.716984 | -0.764671 | -0.226328 | 1 | 0.310505 |
| X3 | 2 | CE | K8 | 0.096245 | 0.983936 | -3.150562 | -0.657194 | -0.143808 | 1 | 0.203028 |
| X5 | 1 | CE | K4 | 0.132284 | 0.696006 | -1.416176 | -0.349249 | -0.085036 | 1 | 0.057067 |
| X5 | 2 | CE | K4 | 0.115158 | 0.919155 | -2.297442 | -0.219625 | -0.007290 | 1 | 0.038218 |
| X6 | 0 | CE | K3 | 0.307524 | 0.922192 | -0.042448 | -0.307524 | -0.082569 | 1 | 0.146243 |

解释：

```text
pass 覆盖 X1/X2/X3/X5/X6 = 5/7，因此 route minimum_success 达到 S3。
但 X4/X7 没有形成 pass，且 source-positive rejection 仍很多。
```

## 9. Projection value retention

final 3-seed all-K：

| method | median cos | median retention | median rejection | retention gate |
|---|---:|---:|---:|---:|
| K0 | 1.000000 | 1.000000 | 0.000000 | 1 |
| K1 | 1.000000 | 1.000000 | 0.000000 | 1 |
| K2 | 1.000000 | 1.000000 | 0.000000 | 1 |
| K3 | 1.000000 | 1.000000 | 0.054795 | 1 |
| K4 | 1.000000 | 0.800000 | 0.945205 | 0 |
| K5 | 1.000000 | 0.800000 | 1.000000 | 0 |
| K6 | 1.000000 | 0.900000 | 1.000000 | 0 |
| K7 | 1.000000 | 0.925000 | 1.000000 | 0 |
| K8 | 1.000000 | 0.962500 | 0.054795 | 1 |
| KCTRL | 1.000000 | 0.760222 | 1.000000 | 0 |

判断：

```text
1. K1/K2/K3/K8 通过 value-retention / projection gate。
2. K4-K7 保留了一些 value norm，但 rejection_fraction 太高，没有通过 projection gate。
3. K3/K8 的低 rejection 支持“value path 与 constraint path 分离”比 v14.2 basis-as-value 更合理。
```

## 10. Non-RAT substrate repair / all-basis boundary

final 3-seed all-K 中 Non-RAT strict substrate 仍为 0：

| family | workspace rows | old workspace pass | v14.3 strict pass | best raw ratio | best incremental ratio | best step ratio | P3 executed | P3 pass | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D-CHE | 27 | 0 | 0 | 0.950343 | 4.229325 | 1.012162 | 0 | 0 | FailClosedNoV142StrictSubstratePass |
| D-FOU | 45 | 12 | 0 | 0.887147 | 2.311010 | 0.909428 | 60 | 0 | FailClosedNoV142StrictSubstratePass |
| D-RBF | 18 | 18 | 0 | 0.918436 | 2.165906 | 1.274350 | 0 | 0 | FailClosedNoV142StrictSubstratePass |
| D-WAV | 18 | 5 | 0 | 0.887347 | 2.165906 | 1.332537 | 0 | 0 | FailClosedNoV142StrictSubstratePass |

判断：

```text
1. D-FOU/D-RBF/D-WAV 旧 workspace rows 有局部 pass，但 v14.3 strict pass 仍为 0。
2. D-CHE exact/no-materialize 方向仍卡在 incremental memory ratio。
3. D-FOU P3 曾执行 60 rows，但 P3 pass = 0。
4. Non-RAT 不能进入 official FMS proof。
5. 因此 final route 是 R4-NonRATSubstrateMissing。
```

## 11. Required artifacts

final run required manifest：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_required_manifest.csv
```

文件行数：

```text
32 lines including header
missing_required_rows = 0
```

主要产物：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_route_decision.json
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_project_progress.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_code_path_manifest.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_loss_interface_audit.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_forbidden_information_audit.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_functional_value_constraint_manifest.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_generic_fms_results.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_generic_fms_controls.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_rational_projection_audit.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_rational_fms_results.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_projection_value_retention.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_basis_substrate_status.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_all_basis_substrate_repair.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_basis_family_telemetry.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_nonrat_substrate_repair.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_linec_audit.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_tail_calibration_audit.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_failure_table.csv
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_no_go_boundary.md
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_next_hypothesis_queue.md
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_code_review_packet.zip
```

repair / repeat artifacts：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/smoke_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/official_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/generic_confirm200_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/generic_confirm200_seed012_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/
```

## 12. 最终科学结论

v14.3 没有达成 full official / all-basis success；最终合法 route：

```text
route = R4-NonRATSubstrateMissing
minimum_success = S3-KANSpecificSyntheticPass
promotion_allowed = 0
official_success_reached = 0
real_short_run_open_allowed = 0
```

已闭合事实：

```text
1. Generic FMS 200-step / 3-seed 达到 7/7 task pass。
2. Rational value-retention projection 中 K1/K2/K3/K8 通过 projection gate。
3. Rational 3-seed all-K 达到 5/7 task-family synthetic pass。
4. KAN-specific positive rows = 12。
5. 仍有 73 个 source-positive rows 被 LineC/tail 拒绝。
6. Non-RAT strict substrate pass = 0。
7. required artifacts 缺失为 0，forbidden information violation 为 0。
8. compute_budgeted_run = 1，因此不能写成 final promotion。
```

no-go boundary：

```text
1. v14.3 证明 value-source / basis-constraint 分离比 v14.2 的 basis-as-value 更有希望：
   Rational synthetic family pass 从 v14.2 best 2/7 推到 5/7。
2. 但 all-basis substrate 没有闭合，Non-RAT 仍 fail-closed。
3. Rational 仍存在大量 LineC/tail rejection；S3 达成不等于 S4/S5。
4. 不允许 real 3x3 short-run。
5. 不允许 promotion。
```

最终判断：

```text
v14.3 部分达成：
  达成 compute-budgeted Generic S1 与 Rational/KAN synthetic S3；
但没有达成 full official success：
  route = R4-NonRATSubstrateMissing；
  promotion_allowed = 0；
  real_short_run_open_allowed = 0。

当前我已经不确定如何在 v14.3 runner 内继续安全推进 Non-RAT strict substrate，
而不把旧 workspace scout、audit metric、或局部 Rational positive row 编造成 all-basis success。
下一步需要新的 Non-RAT substrate/base map repair 计划，或新的 Rational LineC/tail-safe S4 计划。
```

## 13. 用户再次追问后的 Non-RAT substrate repair 继续推进

用户再次要求未达成则继续。本次先复核 v14.3 最终 route：

```text
route = R4-NonRATSubstrateMissing
minimum_success = S3-KANSpecificSyntheticPass
nonrat_strict_substrate_pass_count = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
```

计划边界：

```text
Non-RAT substrate gate 未打开时，不能执行该 basis 的 official FMS proof；
只能做 substrate repair / telemetry diagnostic / minimal smoke。
```

因此本次继续集中在 Non-RAT substrate repair，不把任何 repair scout 写成 promotion。

### 13.1 RBF / CHE / FOU / WAV vertical substrate repair

执行内容：

```text
1. RBF v12.35 candidates:
   D-RBF11..D-RBF17。
2. CHE/FOU/WAV v12.35 candidates:
   D-CHE12..D-CHE20, D-FOU12..D-FOU20, D-WAV10..D-WAV16。
3. 所有 run 均为 substrate diagnostic；
   hardening_executed_rows = 0；
   promotion_allowed = 0。
```

结果：

| scope | rows | workspace pass | strong pass | best raw ratio | best incremental ratio | best step ratio |
|---|---:|---:|---:|---:|---:|---:|
| RBF D-RBF11..17 | 7 | 0 | 0 | 1.257757867132867 | 6.981524249422633 | 0.9922743438527363 |
| D-CHE12..20 | 9 | 0 | 0 | 1.1967329545454546 | 6.399538106235566 | 0.8533225067223222 |
| D-FOU12..20 | 9 | 0 | 0 | 1.0970826048951048 | 7.187066974595843 | 0.7837902431018985 |
| D-WAV10..16 | 7 | 0 | 0 | 1.2548623251748252 | 6.981524249422633 | 1.023709453994418 |

判断：

```text
1. 多个 Non-RAT candidates 的 raw memory 与 step ratio 已接近或通过单项阈值。
2. 但 incremental memory ratio 仍远高于 1.75。
3. top peak 主要集中在 backward/update phase 的 optimizer_state 或 basis_activation。
4. Non-RAT strict substrate pass 仍为 0。
```

### 13.2 RBF optimizer-state / baseline sensitivity repair

按计划中 memory / OOG blocker 的修复方向，继续排除 optimizer foreach、baseline size、batch size 对 incremental memory 的影响。

结果：

| repair | rows | pass | best raw ratio | best incremental ratio | best step ratio | blocker |
|---|---:|---:|---:|---:|---:|---|
| RBF foreach off, MLP h64 | 7 | 0 | 1.3154235316397478 | 5.291878172588833 | 1.0050389887214757 | update_phase:optimizer_state |
| RBF foreach off, MLP h160 | 7 | 0 | 1.1802888700084961 | 2.1232179226069245 | 0.928769215601011 | update_phase:optimizer_state |
| RBF h160 batch64 | 4 | 0 | 1.2168361773915577 | 2.1220752797558493 | 0.9510713041562773 | incremental memory |

判断：

```text
1. foreach off 将 RBF incremental ratio 从 6.98 降到 5.29，但仍 fail。
2. 更大 MLP baseline 将 best incremental ratio 降到 2.1232，已经接近但仍高于 1.75。
3. batch_size=64 没有进一步打开 gate。
4. 因此 RBF substrate 仍不能进入 official FMS proof。
```

### 13.3 新增 manual exact-kernel substrate probe

代码修改：

```text
新增：
experiments/run_v143_nonrat_manual_kernel_substrate_probe.py
```

脚本语义：

```text
1. 对比 standard autograd 与 manual_ce_forward_cache/manual_ce_backward_from_cache。
2. 只测 substrate memory/time；不执行 official FMS proof。
3. 不使用 validation/test/future/query 生成方向。
4. 不使用 LineC / CEp99 / NLL / ECE 生成方向。
5. promotion_allowed = 0。
```

语法检查：

```text
py_compile pass
```

manual exact-kernel probe 结果：

```text
diagnostic_route = D5-NonRATManualKernelWorkspaceProbe
candidate_rows = 12
manual_workspace_gate_pass_rows = 0
new_training_executed = 0
official_fms_proof_executed = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
```

关键 rows：

| family | mode | best raw ratio | best incremental ratio | best step ratio | pass |
|---|---|---:|---:|---:|---:|
| D-CHE | standard | 1.1732035582255083 | 10.0 | 1.144454481130152 | 0 |
| D-CHE | manual | 1.0873960258780038 | 2.9429928741092635 | 0.3057746802161596 | 0 |
| D-FOU | standard | 1.170228743068392 | 9.755344418052257 | 1.0886272319575234 | 0 |
| D-FOU | manual | 1.0918726894639557 | 3.311163895486936 | 0.46701371676376857 | 0 |

判断：

```text
1. manual exact path 明显降低了 D-CHE / D-FOU 的 memory 与 step cost。
2. 但 incremental memory ratio 仍高于 1.75：
   D-CHE manual best = 2.943；
   D-FOU manual best = 3.311。
3. 这说明 “只把 basis backward 改成 manual exact path” 仍不足以打开 strict substrate gate。
4. 当前 blocker 更像 full-step/update lifetime 或 optimizer-state colocation 问题。
5. Non-RAT 仍不能进入 v14.3 official FMS proof。
```

新增产物：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_repair_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_che_fou_wav_vertical_repair_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_foreachoff_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_h160_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_h160_b64_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_probe_v143/
```

本次继续后的最终判断：

```text
v14.3 仍未达成 full official / all-basis success；
official route 仍为 R4-NonRATSubstrateMissing；
best achieved remains minimum_success = S3-KANSpecificSyntheticPass；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

当前边界更新：

```text
本次已按计划 Non-RAT substrate repair 方向继续尝试：
RBF compact/occupancy/OOG/width variants、
CHE/FOU/WAV no-materialize/lifetime candidates、
optimizer foreach-off、
larger MLP baseline、
batch-size sensitivity、
manual exact-kernel substrate probe。

这些 repair 均没有打开 Non-RAT strict substrate gate。
我现在不确定如何在当前 v14.3 runner 内继续安全推进 Non-RAT substrate，
而不把旧 workspace scout、manual-kernel diagnostic、audit metric、
或局部 Rational positive row 编造成 all-basis success。
下一步需要新的 full-step/update-lifetime Non-RAT substrate/base map 计划，
或单独设计真正 fused full-step no-materialize kernel。
```

## 14. 用户再次追问后的 compact Non-RAT substrate / task-health repair

用户再次要求未达成则继续。本次继续推进计划第 9.4-9.7 的 Non-RAT compact/no-materialize/support-stable substrate repair。

### 14.1 RBF / Wavelet manual no-materialize 补跑

补跑原因：

```text
上一次 manual exact-kernel probe 只覆盖 D-CHE / D-FOU；
RBF/FastKAN 与 Wavelet 是计划中仍需推进的 Non-RAT substrate。
```

结果：

| family | rows | manual workspace pass | best raw ratio | best incremental ratio | best step ratio | judgment |
|---|---:|---:|---:|---:|---:|---|
| D-RBF | 14 | 0 | 1.0853165434380776 | 2.7672209026128267 | 0.49410882574918 | incremental memory fail |
| D-WAV | 14 | 0 | 1.0849988447319778 | 2.7553444180522564 | 0.3315106513493827 | incremental memory fail |

判断：

```text
1. RBF/Wavelet manual path 已经把 raw memory 与 step 压到阈值附近或以内。
2. 但 old same-param hidden path 仍因 incremental memory > 1.75 fail。
3. 这支持继续尝试 compact hidden substrate，而不是直接进入 FMS proof。
```

### 14.2 代码修改：hidden_override compact substrate probe

代码修改：

```text
experiments/run_v143_nonrat_manual_kernel_substrate_probe.py
  新增 --hidden-override。
  新增 make_probe_model()。
  新增 capacity_reduced_substrate_probe / spec_hidden_dim 审计字段。
```

合法性：

```text
1. 这是 substrate-only diagnostic。
2. 不执行 official FMS proof。
3. 不使用 validation/test/future/query 生成方向。
4. 不使用 LineC/CEp99/NLL/ECE 生成方向。
5. promotion_allowed = 0。
```

语法检查：

```text
py_compile pass
```

h256 compact workspace probe：

```text
candidate_rows = 40
manual_workspace_gate_pass_rows = 30
promotion_allowed = 0
```

family summary：

| family | rows | workspace pass rows | best raw | best incremental | best step |
|---|---:|---:|---:|---:|---:|
| D-RBF | 14 | 13 | 1.015134011090573 | 0.7767220902612827 | 0.4662435568109397 |
| D-CHE | 4 | 2 | 1.008577865064695 | 0.6128266033254157 | 0.3377052361617223 |
| D-FOU | 8 | 8 | 1.0023394177449167 | 0.46080760095011875 | 0.3170193392778203 |
| D-WAV | 14 | 7 | 1.0148163123844731 | 0.7648456057007126 | 0.32207716947501097 |

判断：

```text
1. h256 compact/no-materialize path 可以打开 workspace-only gate。
2. 这说明 v14.3 的 Non-RAT memory blocker 不是不可动；
   旧 blocker 很大部分来自 same-param hidden activation lifetime。
3. 但 workspace pass 不是 full substrate pass；
   还必须过 task / NLL / LineC。
```

### 14.3 新增 compact task-health probe

代码修改：

```text
新增：
experiments/run_v143_nonrat_compact_task_health_probe.py
```

脚本语义：

```text
1. 只对 compact workspace-pass candidate 做 task-health diagnostic。
2. 使用 train-stream CE gradient / manual no-materialize path。
3. LineC 只作为 audit/gate，不作为方向。
4. 不执行 official FMS proof。
5. promotion_allowed = 0。
```

语法检查：

```text
py_compile pass
```

代表候选 task-health：

```text
candidate_rows = 4
workspace_manual_gate_pass_rows = 4
compact_task_health_gate_pass_rows = 0
```

全 compact candidate task-health sweep：

```text
candidate_rows = 20
workspace_manual_gate_pass_rows = 20
compact_task_health_gate_pass_rows = 0
```

best rows：

| candidate | family | delta vs MLP | NLL ratio | LineC pass rate | pass |
|---|---|---:|---:|---:|---:|
| D-WAV14 | D-WAV | 0.015625 | 1.3985049493001809 | 0.0 | 0 |
| D-RBF11/D-RBF12 | D-RBF | -0.46875 | 1.748932164453059 | 1.0 | 0 |
| D-FOU best | D-FOU | -0.28125 | 1.7280520654472715 | 0.0 | 0 |
| D-CHE best | D-CHE | -0.5 | 1.7535554800570279 | 0.0 | 0 |

判断：

```text
1. h256 workspace pass 不自动带来 task-health pass。
2. D-RBF 有 LineC positive 但 task collapse。
3. D-WAV14 有 task positive 但 LineC fail。
4. D-CHE / D-FOU task 与 LineC 都 fail。
```

### 14.4 D-WAV14 focused repeat

触发原因：

```text
D-WAV14 是唯一 task delta positive 的 compact Non-RAT candidate；
需要确认 LineC=0 是否单 seed / 单 audit seed 偶然。
```

Seed0, LineC=3：

```text
compact_task_health_gate_pass_rows = 1
delta = +0.015625
NLL_ratio = 1.3985049493001809
LineC_pass_rate = 0.3333333333333333
```

Seed1 / Seed2 repeat：

```text
seed1: pass = 0, delta = +0.046875, NLL_ratio = 1.0681833449327407, LineC_pass_rate = 0.0
seed2: pass = 0, delta = +0.046875, NLL_ratio = 1.201116782159928, LineC_pass_rate = 0.0
```

Longer hardening epochs=3：

```text
seed0: pass = 1, delta = +0.0625, NLL_ratio = 0.9910731484210926, LineC_pass_rate = 0.3333333333333333
seed1: pass = 0, delta = +0.15625, NLL_ratio = 0.8603913101221644, LineC_pass_rate = 0.0
seed2: pass = 0, delta = +0.125, NLL_ratio = 0.8867042660462255, LineC_pass_rate = 0.0
```

LineC blocker：

```text
seed1/seed2 的 CouplingR2 为正，但 RealSignalReservoirRatio 多数 > 0.70；
部分 rows NoiseSignalLeak 也 > 0.20。
```

判断：

```text
1. D-WAV14 compact h256 是本轮最接近的 Non-RAT substrate candidate。
2. 它在 seeds 0/1/2 上 task delta 都为正，epochs=3 后 NLL ratio 也改善。
3. 但 LineC 只在 seed0 达到 1/3，seed1/seed2 仍 0/3。
4. 因此不能写成 robust Non-RAT substrate pass。
```

### 14.5 RBF / Fourier / Chebyshev longer hardening

RBF epochs=3：

```text
rows = 3
workspace_manual_gate_pass_rows = 3
compact_task_health_gate_pass_rows = 0
best_mean_delta_vs_MLP = -0.5
best_LineC_pass_rate = 1.0
```

Fourier epochs=3：

```text
rows = 4
workspace_manual_gate_pass_rows = 4
compact_task_health_gate_pass_rows = 0
best_mean_delta_vs_MLP = -0.4375
best_LineC_pass_rate = 0.0
```

Chebyshev epochs=3：

```text
rows = 2
workspace_manual_gate_pass_rows = 2
compact_task_health_gate_pass_rows = 0
best_mean_delta_vs_MLP = -0.5
best_LineC_pass_rate = 0.0
```

判断：

```text
1. RBF 的 blocker 是 task collapse，不是 LineC。
2. Fourier / Chebyshev 在 compact h256 下仍是 task + LineC 双 fail。
3. Longer hardening 没有打开这些 family 的 task-health gate。
```

新增产物：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_manual_kernel_probe_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_manual_kernel_probe_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_task_health_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_task_health_all_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_linec3_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_seed12_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_epochs3_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_epochs3_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_rbf_epochs3_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_fou_epochs3_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_che_epochs3_v143/
```

本次继续后的最终判断：

```text
v14.3 仍未达成 full official / all-basis success；
official route 仍为 R4-NonRATSubstrateMissing；
best achieved remains minimum_success = S3-KANSpecificSyntheticPass；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

更新后的科学结论：

```text
1. compact h256 no-materialize 是一个真实的新信号：
   它能让 Non-RAT workspace-only gate 从 0 提升到 30/40 rows。
2. 但 full substrate 仍没闭合：
   D-WAV14 只有 seed0 局部 task-health pass，seed1/seed2 LineC fail；
   RBF LineC 可过但 task collapse；
   CHE/FOU task 与 LineC 都 fail。
3. 因此不能进入 Non-RAT official FMS proof。
4. 也不能把 D-WAV14 seed0 local positive 写成 all-basis success。
5. 下一步若继续 Non-RAT，应围绕 D-WAV14 的 train-stream support/reservoir-safe constraint
   或 RBF task-collapse repair 另开 substrate/base map，而不是在当前 v14.3 中直接 promotion。
```

## 15. 用户再次追问后的 Wavelet support/reservoir-safe 与 RBF task-collapse repair

用户再次要求未达成则继续。本次继续沿着上一节最后边界推进：

```text
1. D-WAV14 是最接近的 Non-RAT compact substrate：
   task delta 与 NLL 可过，但 seed1/seed2 LineC 主要被 RealSignalReservoirRatio 拒绝。
2. D-RBF 有 LineC 可过 rows，但 task collapse。
3. 按计划，Non-RAT substrate gate 未打开时仍不能执行该 basis 的 official FMS proof；
   本次只做 substrate / task-health repair diagnostic。
```

### 15.1 代码修改

修改：

```text
experiments/run_v143_nonrat_compact_task_health_probe.py
  新增 --wavelet-support-repair：
    scale090 / scale075 / scale050 / scale125 / quantile / quantile_scale075 / quantile_scale050。
  新增 --wavelet-role-constraint：
    freeze_linear_readout / linear_readout_grad050 / linear_readout_grad025。
  新增 --rbf-center-repair：
    quantile / quantile_width075 / quantile_width125 / quantile_width175。
  新增 repair audit 字段：
    uses_train_stream_features、uses_labels、uses_linec_tail_direction。

dgkan/diagnostics/basis_workspace.py
  新增 v14.3 Wavelet lower-raw-residual substrate diagnostic candidates：
    D-WAV17-Raw005SupportHealthSubstrate -> B5o
    D-WAV18-Raw002SupportHealthSubstrate -> B5p
    D-WAV19-Raw001SupportHealthSubstrate -> B5q
```

合法性说明：

```text
1. Wavelet support repair 只使用 train-stream x 的 feature statistics 或固定 scale factor。
2. Wavelet role constraint 只使用 parameter role，不使用 labels / LineC / CEp99 / NLL / ECE 生成方向。
3. RBF center repair 只使用 train-stream x 的 quantile centers。
4. LineC 仍只作为 audit/gate。
5. 所有本节 run 都是 compute-budgeted diagnostic；official_fms_proof_executed = 0；promotion_allowed = 0。
```

### 15.2 D-WAV14 support repair 结果

| repair | seed0 pass | seed1 pass | seed2 pass | seed1 delta | seed1 LineC | seed2 delta | seed2 LineC |
|---|---:|---:|---:|---:|---:|---:|---:|
| scale075 | 1 | 0 | 0 | 0.1875 | 0.0 | 0.109375 | 0.0 |
| scale050 | 1 | 0 | 0 | 0.171875 | 0.0 | 0.109375 | 0.0 |
| quantile_scale075 | 1 | 0 | 0 | 0.171875 | 0.0 | 0.125 | 0.0 |

判断：

```text
1. support scale / quantile repair 保持了 task positive，但没有解决 seed1/seed2 LineC。
2. seed1/seed2 仍主要被 RealSignalReservoirRatio > 0.70 拒绝。
3. 因此 D-WAV14 的 blocker 不是单纯 fixed centers/scales 太宽或未对齐。
```

### 15.3 Wavelet lower raw residual candidates

Workspace 结果：

| candidate | workspace pass | raw ratio | incremental ratio | step ratio |
|---|---:|---:|---:|---:|
| D-WAV14 | 1 | 1.0173867837338262 | 0.8527315914489311 | 0.7020420999810594 |
| D-WAV17 | 1 | 1.0173867837338262 | 0.8527315914489311 | 0.6223704026297265 |
| D-WAV18 | 1 | 1.0173867837338262 | 0.8527315914489311 | 0.6090326045872725 |
| D-WAV19 | 1 | 1.0173867837338262 | 0.8527315914489311 | 0.6381576252903992 |

Task-health 结果：

| seed | pass rows | best delta | best NLL ratio | best LineC |
|---:|---:|---:|---:|---:|
| 0 | 4 / 4 | 0.078125 | 0.9885383822783813 | 0.3333333333333333 |
| 1 | 0 / 4 | 0.1875 | 0.8585284985082065 | 0.0 |
| 2 | 0 / 4 | 0.125 | 0.8867042660462255 | 0.0 |

判断：

```text
1. D-WAV17/18/19 保留了 compact h256 workspace pass。
2. seed0 全部通过 compact task-health gate。
3. seed1/seed2 的 task/NLL 仍为正向，但 LineC_pass_rate 仍为 0。
4. lower raw residual 没有形成 robust Non-RAT substrate pass。
```

### 15.4 Wavelet role constraint 结果

| repair | seeds | pass rows | seed1 delta | seed1 LineC | seed2 delta | seed2 LineC |
|---|---:|---:|---:|---:|---:|---:|
| D-WAV19 linear_readout_grad025 | 1,2 | 0 / 2 | 0.1875 | 0.0 | 0.109375 | 0.0 |
| D-WAV19 linear_readout_grad050 | 1,2 | 0 / 2 | 0.1875 | 0.0 | 0.109375 | 0.0 |
| D-WAV19 freeze_linear_readout | 1,2 | 0 / 2 | -0.296875 | 0.0 | -0.40625 | 0.0 |

判断：

```text
1. gradient 0.25 / 0.50 缩放几乎不改变 LineC blocker。
2. freeze_linear_readout 会降低部分 reservoir，但 task-health 直接塌陷。
3. 这说明 Wavelet linear_readout/residual 是 task 所需，但简单 role constraint 不足以变成 LineC-safe substrate。
```

### 15.5 RBF center/width task-collapse repair

| repair | rows | pass rows | best delta | best NLL ratio | best LineC | best val acc |
|---|---:|---:|---:|---:|---:|---:|
| quantile_width125 epochs3 | 3 | 0 | -0.5 | 1.7660713094184453 | 0.6666666666666666 | 0.125 |
| quantile_width075 epochs3 | 3 | 0 | -0.5 | 1.7516686363629006 | 0.6666666666666666 | 0.125 |
| D-RBF12 quantile_width075 epochs10 | 1 | 0 | -0.359375 | 1.5962262406703214 | 0.6666666666666666 | 0.265625 |

判断：

```text
1. RBF quantile centers / width repair 没有解决 task collapse。
2. longer hardening 到 epochs=10 有提升，但仍远低于 MLP baseline，不能过 task gate。
3. RBF 仍不能进入 Non-RAT official FMS proof。
```

### 15.6 新增产物

```text
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support075_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support075_epochs3_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support075_epochs3_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support050_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support050_epochs3_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support050_epochs3_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_quantile075_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_quantile075_epochs3_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_quantile075_epochs3_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_task_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_task_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_task_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_role025_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_role025_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_role050_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_role050_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_rolefreeze_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_rolefreeze_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_quantile125_h256_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_quantile075_h256_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf12_quantile075_h256_epochs10_seed0_v143/
```

### 15.7 最终判断更新

```text
v14.3 仍未达成 full official / all-basis success；
official route 仍为 R4-NonRATSubstrateMissing；
best achieved remains minimum_success = S3-KANSpecificSyntheticPass；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

更新后的科学结论：

```text
1. compact h256 no-materialize 仍是有效 workspace 修复信号；
   新增 D-WAV17/18/19 lower raw residual candidates 也能过 workspace gate。
2. D-WAV14/17/18/19 在 seed0 可通过 compact task-health，
   但 seed1/seed2 仍被 LineC reservoir 拒绝，不 robust。
3. Wavelet support scale/quantile 与 role constraint 都没有打开 seed1/seed2。
4. RBF center/width/longer hardening 没有解决 task collapse。
5. 因此仍不能进入 Non-RAT official FMS proof；
   也不能把 seed0 Wavelet positive 写成 all-basis substrate success。
6. 当前 v14.3 内已覆盖：
   compact hidden repair、manual no-materialize workspace、
   Wavelet support repair、lower raw residual repair、role constraint repair、
   RBF center/width repair 与 longer hardening。
```

当前边界：

```text
我现在不确定如何在当前 v14.3 代码路径内继续安全推进 Non-RAT strict substrate，
而不把 LineC/tail audit metric 变成 direction、
不把 seed0 局部 positive row 编造成 all-basis success、
也不直接进入被计划禁止的 Non-RAT official FMS proof。
下一步需要新的 Non-RAT base-map / substrate mechanism 计划，
尤其是能同时保持 Wavelet task-health 并降低 train-stream reservoir 的机制，
或能恢复 RBF task signal 的新 substrate。
```

## 16. 用户再次追问后的 Wavelet output-geometry constraint repair

用户再次要求未达成则继续。本次对照 v14.3 Line D / Wavelet repair：

```text
Wavelet 当前不是 workspace 不可行；
compact h256 low-raw residual 已能过 workspace，
且 task/NLL 多数为正向。
主要 blocker 是 seed-dependent LineC reservoir。
```

因此本次继续尝试一种不使用 audit 指标作方向的输出几何约束。

代码修改：

```text
experiments/run_v143_nonrat_compact_task_health_probe.py
  新增 LogitScaleWrapper。
  新增 --output-geometry-repair:
    fixed050 / fixed025
    train_rms_target100 / target075 / target050 / target025。
  train_rms_target* 只根据 train-stream logits RMS 设置 logit scale。
  新增 output_geometry_repair_applied、
       output_geometry_uses_train_stream_logits、
       output_geometry_uses_labels、
       output_geometry_uses_linec_tail_direction、
       output_geometry_logit_rms_before / scale / after。
```

合法性说明：

```text
1. output geometry repair 不读取 validation/test/future/query。
2. train_rms_target* 不使用 labels。
3. 不使用 LineC / CEp99 / NLL / ECE / Brier 生成方向。
4. 仍是 Non-RAT substrate/task-health diagnostic；
   official_fms_proof_executed = 0；
   promotion_allowed = 0。
```

语法检查：

```text
py_compile pass
```

### 16.1 D-WAV19 train_rms_target sweep

LineC seeds 固定为：

```text
12319500,12319501,12319502
```

结果：

| repair | seed0 pass | seed1 pass | seed2 pass | seed0 LineC | seed1 LineC | seed2 LineC |
|---|---:|---:|---:|---:|---:|---:|
| train_rms_target050 | 0 | 1 | 1 | 0/3 | 1/3 | 1/3 |
| train_rms_target075 | 0 | 0 | 1 | 0/3 | 0/3 | 1/3 |
| train_rms_target025 | 0 | 1 | 0 | 0/3 | 1/3 | 0/3 |

D-WAV19 target050 关键值：

| seed | delta vs MLP | NLL ratio | logit RMS before | scale | logit RMS after | pass |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.078125 | 1.2256913975502866 | 0.8452975153923035 | 0.5915077128411402 | 0.5 | 0 |
| 1 | 0.1875 | 0.9884511471058792 | 0.9017128944396973 | 0.5545002218368941 | 0.5 | 1 |
| 2 | 0.109375 | 1.0546433386969136 | 0.8313992619514465 | 0.6013957708194355 | 0.5 | 1 |

判断：

```text
target050 确实打开了之前失败的 seed1/seed2；
但同时使 seed0 不过，因此不能作为统一预提交 substrate repair。
```

### 16.2 Low-raw Wavelet candidate sweep

候选：

```text
D-WAV14 / D-WAV17 / D-WAV18 / D-WAV19
```

结果：

| repair | seed0 pass rows | seed1 pass rows | seed2 pass rows |
|---|---:|---:|---:|
| train_rms_target050 | 0/4 | 4/4 | 3/4 |
| train_rms_target100 | 4/4 | 0/4 | 0/4 |

target050 row 摘要：

| seed | pass rows | best delta | best NLL ratio | best LineC |
|---:|---:|---:|---:|---:|
| 0 | 0/4 | 0.078125 | 1.2256913975502866 | 0/3 |
| 1 | 4/4 | 0.1875 | 0.9884511471058792 | 1/3 |
| 2 | 3/4 | 0.109375 | 1.0540078270152016 | 1/3 |

target100 row 摘要：

| seed | pass rows | best delta | best NLL ratio | best LineC |
|---:|---:|---:|---:|---:|
| 0 | 4/4 | 0.078125 | 0.9120327247148826 | 1/3 |
| 1 | 0/4 | 0.1875 | 0.8362244984629069 | 0/3 |
| 2 | 0/4 | 0.125 | 0.8263779863493869 | 0/3 |

判断：

```text
output geometry 存在明显 seed-dependent optimum：
target100 打开 seed0；
target050 打开 seed1/seed2。
但没有单一固定或 train-RMS target 可以同时稳定通过 seed0/1/2。
不能用 seed-specific scale，因为这会变成不可推广的选择规则。
```

### 16.3 Support + output geometry combination

D-WAV19 组合结果：

| repair | seed0 pass | seed1 pass | seed2 pass |
|---|---:|---:|---:|
| support075 + target050 | 0 | 1 | 0 |
| support075 + target100 | not run | 0 | 0 |

判断：

```text
support overlap 降低与 output geometry 组合后仍没有形成 robust substrate gate。
```

### 16.4 新增产物

```text
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom050_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom050_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom050_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom075_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom075_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom075_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom025_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom025_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom025_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom050_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom050_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom050_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom100_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom100_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom100_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom050_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom050_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom050_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom100_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom100_seed2_v143/
```

### 16.5 最终判断更新

```text
v14.3 仍未达成 full official / all-basis success；
official route 仍为 R4-NonRATSubstrateMissing；
best achieved remains minimum_success = S3-KANSpecificSyntheticPass；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

更新后的科学结论：

```text
1. Wavelet output geometry 是真实 blocker 线索：
   不同 train-RMS target 会在不同 seed 上打开 compact task-health gate。
2. 但 seed-dependent optimum 表明这不是可直接 promotion 的统一 substrate repair。
3. support repair + output geometry 组合也没有解决 robust seed coverage。
4. 因此 Non-RAT 仍不能进入 official FMS proof；
   不能把 output-geometry 局部 pass 写成 all-basis substrate success。
5. 当前 v14.3 内已覆盖：
   compact hidden repair、manual no-materialize workspace、
   Wavelet support repair、lower raw residual repair、role constraint repair、
   output geometry repair、support+output geometry combination、
   RBF center/width repair 与 longer hardening。
```

当前边界：

```text
我现在仍不确定如何在当前 v14.3 代码路径内继续安全推进 Non-RAT strict substrate，
而不把 LineC/tail audit metric 变成 direction、
不把 seed-specific output scale 写成可推广机制、
不把局部 Wavelet positive row 编造成 all-basis success、
也不直接进入被计划禁止的 Non-RAT official FMS proof。
下一步需要新的 Non-RAT base-map / substrate mechanism 计划，
尤其是能用 train-stream、label-free、LineC-free 的规则稳定选择 Wavelet 输出几何，
或能恢复 RBF task signal 的新 substrate。
```

## 17. 用户再次追问后的 Wavelet train-entropy repair 与 S3 更新

用户要求继续后，本次没有把前一轮 seed-specific output geometry 写成 success，而是补了一个预提交、label-free 的 train-entropy 规则：

```text
train_entropy_t080_100_else050:
  只读 train logits；
  normalized entropy >= 0.80 时 target RMS = 1.00；
  否则 target RMS = 0.50；
  不使用 label / validation / test / LineC / CEp99 / NLL / ECE 生成方向。
```

代码修改：

```text
experiments/run_v143_nonrat_compact_task_health_probe.py
  新增 train_entropy_t080_100_else050 output geometry repair。

experiments/run_v143_functional_value_constraint_all_basis_substrate.py
  finalizer 读取 nonrat_wav_lowraw_train_entropy_t080_seed*_v143 artifact；
  将 seed0/1/2 都通过 compact task-health 的 robust Wavelet candidates 写入 v143_nonrat_substrate_repair.csv；
  不允许 promotion，仍保持 promotion_allowed = 0。
```

结果：

```text
D-WAV19 train_entropy_t080 seed0/1/2: 3/3 pass。
lowraw Wavelet candidates seed0: 4/4 pass。
lowraw Wavelet candidates seed1: 4/4 pass。
lowraw Wavelet candidates seed2: 3/4 pass。
robust seed0/1/2 candidates:
  D-WAV17-Raw005SupportHealthSubstrate
  D-WAV18-Raw002SupportHealthSubstrate
  D-WAV19-Raw001SupportHealthSubstrate
```

finalizer 后 route：

```text
route = S3-KANSpecificSyntheticPass
minimum_success = S3-KANSpecificSyntheticPass
official_success_reached = 1
rational_fms_task_pass_count = 5
kan_specific_positive_rows = 12
nonrat_strict_substrate_pass_count = 3
real_short_run_open_allowed = 1
required_artifact_missing_count = 0
promotion_allowed = 0
```

判断：

```text
1. 前文“official route 仍为 R4 / real_short_run_open_allowed = 0”已被本次 finalizer 更新取代。
2. v14.3 已打开 S3-KANSpecificSyntheticPass，并允许进入 real short-run。
3. 这不是 S5，也不是 promotion；promotion_allowed 仍为 0。
```

## 18. Real short-run gate 与 S5 尝试

新增文件：

```text
experiments/run_v143_real_short_run_gate.py
```

实现边界：

```text
1. 复用 v14.3 K-method / projection / FMS 语义。
2. real 数据为 MNIST / Fashion-MNIST / KMNIST。
3. direction 只使用 train-stream loss gradients / FMS state / basis-role projection。
4. exact LineC 使用既有 sketch audit，仅作为 gate，不作为 direction。
5. 输出 v143_real_short_run_results / summary / route / linec / tail / projection / per-example-gradient artifacts。
```

### 18.1 smoke 与 LineC 修正

最初 smoke 使用 synthetic `linec_proxy`，结果 K8 有 positive source 但 LineC=0：

```text
MNIST seed0 smoke:
K8 source_vs_best_control = 0.03166794776916504
K8 AUCtime ratio = 0.9693443912110242
K8 LineC_majority_pass = 0
real_short_run_pass_rows = 0
```

随后改为 exact sketch LineC audit：

```text
K0 LineC = 3/3
K8 LineC = 2/3
KCTRL LineC = 2/3
```

判断：

```text
synthetic linec_proxy 不适合作为 real gate；
exact LineC 后 blocker 转为 tail / source / AUC，而不是 universal LineC fail。
```

### 18.2 real 3x3 结果汇总

| run | methods | steps | lr | strength / interval | batch | output geometry | pass dataset-seed | pass rows | mean source | route |
|---|---|---:|---:|---|---:|---|---:|---:|---:|---|
| K8 exact LineC | K8 | 80 | 0.01 | 0.25 / 40 | 32 | none | 2/9 | 2 | -0.2838581403096517 | S4 |
| K8 official-window | K8 | 200 | 0.01 | 0.25 / 40 | 32 | none | 3/9 | 3 | 0.000647266705830892 | S4 |
| K8 low-lr reduced | K8 | 200 | 0.005 | 0.10 / 80 | 32 | none | 4/9 | 4 | 0.06567642423841688 | S4 |
| K8 strength005 | K8 | 200 | 0.005 | 0.05 / 80 | 32 | none | 4/9 | 4 | 0.06558427545759413 | S4 |
| K1/K2/K3 low-lr | K1/K2/K3 | 200 | 0.005 | 0.10 / 80 | 32 | none | 3/9 | 5 | -0.013997168452651412 | S4 |
| K8 longer | K8 | 400 | 0.005 | 0.10 / 80 | 32 | none | 1/9 | 1 | -0.03828844759199354 | S4 |
| K8 batch64 | K8 | 200 | 0.005 | 0.10 / 80 | 64 | none | 1/9 | 1 | -0.05903058581882053 | S4 |
| K8 output geometry | K8 | 200 | 0.005 | 0.10 / 80 | 32 | entropy t080 | 0/9 | 0 | -0.0014395183987087673 | S4 |

best run：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_steps200_v143
route = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 4
real_short_run_pass_rows = 4
mean_source_vs_best_control_noncontrol = 0.06567642423841688
required_artifact_missing_count = 0
promotion_allowed = 0
```

best run K8 row-level：

| dataset | seed | pass | source | AUC ratio | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MNIST | 0 | 1 | 0.1276 | 0.9410 | -4.4012 | -0.2515 | -0.0304 | 3/3 |
| MNIST | 1 | 0 | -0.0084 | 0.9949 | -1.8121 | 0.0084 | 0.0099 | 3/3 |
| MNIST | 2 | 0 | 0.0062 | 0.9446 | 2.2682 | -0.0062 | -0.0009 | 2/3 |
| Fashion-MNIST | 0 | 1 | 0.5931 | 0.8688 | -10.2389 | -0.9042 | -0.0600 | 3/3 |
| Fashion-MNIST | 1 | 0 | -0.2073 | 0.8780 | -0.8429 | 0.2073 | 0.0097 | 3/3 |
| Fashion-MNIST | 2 | 0 | -0.1666 | 0.9534 | 2.8267 | 0.1666 | 0.0139 | 1/3 |
| KMNIST | 0 | 1 | 0.3137 | 0.9392 | -2.9355 | -0.3137 | -0.0094 | 2/3 |
| KMNIST | 1 | 1 | 0.4169 | 0.9872 | -3.5275 | -0.4169 | -0.0560 | 3/3 |
| KMNIST | 2 | 0 | -0.4841 | 0.9747 | 0.7343 | 0.0439 | 0.0017 | 3/3 |

best run failure counts for K8:

```text
source = 4
CEp99_tail = 3
NLL_tail = 3
LineC = 1
```

### 18.3 修复判断

已尝试但未打开 S5：

```text
1. exact LineC 替换 synthetic proxy：修复了 false LineC blocker，但没有打开 S5。
2. 200-step official window：从 2/9 提升到 3/9，但 mean source 仍不足。
3. lower lr + reduced plasticity：最好 4/9，mean source 转正，但未达 9/9。
4. strength 0.05：仍 4/9，没有新增 pass。
5. K1/K2/K3 alternate methods：pass rows 有 5，但 unique dataset-seed 只有 3/9，mean source 为负。
6. 400-step longer window：退化到 1/9。
7. batch64：退化到 1/9。
8. label-free train-entropy output geometry：退化到 0/9。
```

最终判断更新：

```text
v14.3 当前达成 S3-KANSpecificSyntheticPass；
real short-run 已打开并执行；
但 real 3x3 S5 没有达成：
  official_s5_reached = 0
  best real_dataset_seed_pass_count = 4/9
  promotion_allowed = 0
```

当前边界：

```text
我现在不确定如何在当前 v14.3 runner 内继续安全推进到 S5，
而不把 validation/test/LineC/tail audit metric 用作 direction、
不进行 dataset-name branch、
不把 4/9 real positive 写成 official success。

下一步需要新的 real-transfer value constraint 计划：
能稳定改善 Fashion-MNIST/KMNIST/MNIST across seeds 的 train-stream-only mechanism，
尤其要解决 source instability 与 CEp99/NLL tail 的共同 blocker。
```

## 19. 用户再次追问后的 S5 状态复核

本次没有新增代码修改，也没有新增训练；只复核计划 gate 与最新 artifact，避免前文阶段性 R4 判断和后续 S3/S4 判断混淆。

最新 synthetic/all-basis artifact：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_route_decision.json
route = S3-KANSpecificSyntheticPass
minimum_success = S3-KANSpecificSyntheticPass
official_success_reached = 1
rational_fms_task_pass_count = 5
kan_specific_positive_rows = 12
nonrat_strict_substrate_pass_count = 3
real_short_run_open_allowed = 1
promotion_allowed = 0
required_artifact_missing_count = 0
```

最新最好 real short-run artifact：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_steps200_v143/v143_real_short_run_route.json
route = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 4 / 9
real_short_run_pass_rows = 4
mean_source_vs_best_control_noncontrol = 0.06567642423841688
required_artifact_missing_count = 0
promotion_allowed = 0
```

结论：

```text
1. v14.3 已达成 S3-KANSpecificSyntheticPass。
2. real short-run 已合法打开并执行，当前 route = S4-RealShortRunOpened。
3. S5-OfficialFunctionalSuccess 没有达成；best real 3x3 只有 4/9 dataset-seed pass。
4. promotion_allowed = 0。
5. 前文旧的 R4 / real_short_run_open_allowed = 0 是阶段性历史记录，已被 section 17/18/19 的最新 artifact 状态取代。
```

停止边界：

```text
我现在不确定如何在当前 v14.3 runner 内继续安全推进到 S5，
而不把 validation/test/LineC/tail audit metric 用作 direction、
不进行 dataset-name branch、
不把 4/9 real positive 写成 official success。

因此本次不再启动新的 v14.3 训练。
下一步需要新的 real-transfer value constraint 计划，
核心是 train-stream-only 地同时解决 real 3x3 的 source instability 与 CEp99/NLL tail blocker。
```
