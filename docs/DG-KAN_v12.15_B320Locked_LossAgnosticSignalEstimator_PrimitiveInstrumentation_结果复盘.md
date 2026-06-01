# DG-KAN v12.15 B320Locked LossAgnosticSignalEstimator PrimitiveInstrumentation 结果复盘

生成时间：`2026-05-24T01:49:32Z`

对应计划：`docs/DG-KAN_v12.15_B320Locked_LossAgnosticSignalEstimator_PrimitiveInstrumentation_独立分析与下一步计划.md`

对应执行审计：`docs/DG-KAN_v12.15_B320Locked_LossAgnosticSignalEstimator_PrimitiveInstrumentation_执行复盘.md`

结果来源：

```text
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5/v1215_route_decision.json
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5/v1215_anchor_monitor.csv
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5/v1215_linec_response_identifiability.csv
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5/v1215_linec_response_model.csv
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5/v1215_actuator_truth.csv
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5/v1215_b15_p3_candidates.csv
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5/v1215_b15_p3_summary.csv
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5/v1215_failure_table.csv
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5/v1215_loss_agnostic_audit.csv
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5/v1215_classic_family_status.csv
```

本文件只解释真实落盘 artifact，不新增实验数据，不修改 gate，不把 smoke / partial / coupling-only 结果写成 survivor。

## 1. 最终结论

v12.15 没有找到可进入 P4 的 loss-agnostic functional survivor。当前 route 为：

```text
route = R4-LossAgnosticFunctionalMechanismNotFound
p4_open = 0
p3_survivor_count = 0
linec_rows = 90
actuator_rows = 81
p3_rows = 36
```

解释：

```text
B320 base 仍是 locked anchor；
Line C response probes 说明 CouplingR2 可由 sketch_delta_fro 预测，但 NoiseSignalLeak 的 estimator 仍弱；
primitive actuator truth 显示所有 role actuator 都是 ActuatorWeak；
B15-A/B/C/D 全部 0/9 dataset-seed promotion；
NoiseSignalLeak、RealSignalReservoirRatio、control_gap 三个 hard gate 对所有 36 个 P3 row 都失败；
因此 P4 short-run 按计划关闭。
```

核心判断：

```text
当前 cotangent / role / moment / shallow primitive actuator family 可以制造 CouplingR2 位移，
但仍不能稳定驱动 noise/reservoir release，也不能击败 strong controls。
```

## 2. B320 Anchor 状态

v12.15 没有重新优化 B320，也没有用 functional update 掩盖 base regression。Anchor monitor 复用 v12.14 locked artifact，hash match 为 1：

```text
source_artifact = results/v12_14_b320locked_lossagnostic_functional_mechanism/final_aggregate_official/v1214_b320_anchor_monitor.csv
source_sha256 = 65974eedebae91e8d2800dec33bc03230a5b7509fb359d7577315419b7cecba0
hash_match_v1214 = 1
regression_flag = 0
```

B320 locked 指标：

```text
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta_vs_mlp = 0.027669270833333332
worst_delta_vs_mlp = -0.001953125
near_pass = 1.0
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
ECE_delta = 0.01767905056476593
LineC_nontearing_pass = 1
```

结论：本轮失败不是 B320 base regression，也不是需要继续修 B320 architecture。v12.15 的主 blocker 仍在 loss-agnostic signal estimator / primitive actuator。

## 3. Line C Response Identifiability

本轮 C0 运行：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
splits = 1
window = 5
batch_size = 32
rows = 90
```

### 3.1 Probe update 汇总

| method | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | max_sketch_delta_fro |
|---|---:|---:|---:|---:|---:|
| `U0-NoOp` | `9` | `0.0` | `0.0` | `0.0` | `0.0` |
| `U1-RandomMatchedNorm` | `9` | `0.0006437344345478429` | `-4.21040587955051e-06` | `2.1495752864413793e-05` | `0.00022624626581091434` |
| `U2-TaskOnlyAdamW-Audit` | `9` | `0.18454791837066026` | `-0.00015446957614686753` | `-0.001847935633526908` | `0.007331079337745905` |
| `U3-SNRGatedAdamWResidual-Audit` | `9` | `0.16695761924555833` | `-0.0007107764896419314` | `-0.0025259521272447375` | `0.006502765696495771` |
| `U4-RandomCotangentVJPEnsemble` | `9` | `0.08276374668993002` | `0.00040467911296420626` | `-0.0005522933271196154` | `0.003808514215052128` |
| `U5-OrthogonalCotangentVJPEnsemble` | `9` | `0.1045888667427449` | `0.0007300175105532011` | `-0.0006717758046256171` | `0.005921827629208565` |
| `U6-RoleConditionedPrimitiveActuator` | `9` | `0.10087649443257325` | `0.00010415435665183597` | `0.0003376934263441298` | `0.006373575888574123` |
| `U7-ProjectionQuadRoleActuator` | `9` | `0.10504745467402654` | `8.920538756582473e-05` | `0.00014090372456444634` | `0.0036166009958833456` |
| `U8-BranchReadoutRoleActuator` | `9` | `0.054011763505262454` | `-1.786649227142334e-05` | `1.7313493622673883e-05` | `0.0009353267960250378` |
| `U9-MixedLowRankActuatorBasis` | `9` | `0.0880465369978239` | `-4.555253932873408e-05` | `0.0001450445916917589` | `0.003121167654171586` |

### 3.2 Response model 结果

| target | best feature | spearman_corr | pearson_corr | sign_accuracy | negative_release_precision | negative_release_recall |
|---|---|---:|---:|---:|---:|---:|
| `CouplingR2_delta` | `sketch_delta_fro` | `0.8325876488279158` | `0.7450220117897491` | `0.9` | `0.0` | `0.0` |
| `NoiseSignalLeak_delta` | `role_direct_norm` | `0.1776506664644746` | `-0.012871677236849602` | `0.5` | `0.0` | `0.0` |
| `RealSignalReservoirRatio_delta` | `sketch_delta_fro` | `-0.30037490215465745` | `-0.46514792520497145` | `0.5444444444444444` | `0.0` | `0.0` |

解释：

```text
CouplingR2_delta 可被 sketch_delta_fro 较好预测，说明 C0 不是全噪声；
NoiseSignalLeak_delta 的 best Spearman 只有 0.17765，低于计划中的 0.30；
Reservoir 的 Spearman 约 0.30037，刚过数值阈值，但 negative_release_precision / recall 仍为 0；
两个 hard target 都没有形成可部署 release selector。
```

因此，v12.15 的 C0 结论不是“Line C metric 失效”，而是：

```text
当前 loss-agnostic observables 能解释输出耦合位移，
但不能可靠预测 NoiseSignalLeak <= -0.01 或 RealSignalReservoirRatio <= -0.01。
```

## 4. Line I Primitive Actuator Truth

本轮 I0 运行：

```text
actuator_rows = 81
actuator_ids = I1..I9
datasets x seeds = 3 x 3
window = 5
```

所有 actuator 的分类结果均为：

```text
ActuatorWeak = 81
ObjectiveMisaligned = 0
SafetyRejected = 0
AcceptedForCandidateBasis = 0
```

Role-level 汇总：

| actuator | rows | mean CouplingR2_delta | mean NoiseSignalLeak_delta | mean Reservoir_delta | max_sketch_delta_fro | max_projector_angle_deg | max_logit_drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| `I1-DirectRoleActuator` | `9` | `0.007326148609457005` | `0.00010202328364054362` | `0.00004337728023529053` | `0.0007408506935462356` | `0.15195108950138092` | `0.00380706787109375` |
| `I2-QuadRoleActuator` | `9` | `0.08386112797335477` | `-0.000026512063211864894` | `-0.0006492601500617133` | `0.002682015299797058` | `0.19179700314998627` | `0.03238105773925781` |
| `I3-BranchRoleActuator` | `9` | `0.057820527349093216` | `0.00025583555301030475` | `-0.00005572040875752767` | `0.0016414743149653077` | `0.15450507402420044` | `0.01628875732421875` |
| `I4-ProjectionPRoleActuator` | `9` | `0.07329988915199523` | `-0.00003411538071102566` | `-0.000125199556350708` | `0.003415588289499283` | `0.3800070583820343` | `0.028767108917236328` |
| `I5-ReadoutRoleActuator` | `9` | `0.06944210057495126` | `-0.00010383894873989953` | `-0.00017460187276204428` | `0.002621237188577652` | `0.2373882383108139` | `0.015362739562988281` |
| `I6-DirectQuadCoupledActuator` | `9` | `0.08388146903961284` | `0.0002958772497044669` | `-0.000034797522756788465` | `0.0038212935905903578` | `0.46688127517700195` | `0.03235673904418945` |
| `I7-QuadBranchCoupledActuator` | `9` | `0.08386908828401315` | `-0.0003818400825063388` | `0.000026954544915093316` | `0.0036211851984262466` | `0.7409852743148804` | `0.032502174377441406` |
| `I8-ProjectionReadoutCoupledActuator` | `9` | `0.08388146903961284` | `0.0003271231220828162` | `-0.00001808007558186849` | `0.0025524774100631475` | `0.2980513572692871` | `0.03235673904418945` |
| `I9-AllRoleLowRankActuator` | `9` | `0.08388657318706218` | `0.000010726766453848944` | `0.00020701024267408584` | `0.0031917046289891005` | `0.17132017016410828` | `0.03247809410095215` |

解释：

```text
logit drift 均未触发 0.05 safety blocker；
CEp99 / holdout audit 没有成为主失败原因；
真正问题是 sketch_delta_fro 最大只有 0.00382，projector angle 最大只有 0.74099 度，
低于计划中的 sketch_delta_fro >= 0.01 与 angle >= 1 度 actuator gate。
```

因此 Line I 明确支持 `ActuatorWeak`，不是 `SafetyRejected`。继续调小安全阈值、加 CE safety direction 或做 coefficient grid 都不是当前修复方向。

## 5. B15 Candidate 结果

B15-A/B/C/D 均按 v12.15 规则从 C0/I0 的 loss-agnostic probes 构造，没有使用 CE vector / label-loss VJP / validation/test/dataset-name branch。

### 5.1 P3 mean summary

| method | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | p3_pass_rows | max_sketch_delta_fro |
|---|---:|---:|---:|---:|---:|---:|
| `B15-A-ResponseMatrixFunctionalQP` | `9` | `0.10450844915366767` | `0.0006779738598399692` | `-0.0008775401446554395` | `0` | `0.004413261543959379` |
| `B15-B-SketchEigenspaceTargeting` | `9` | `0.11397181964840976` | `-0.000001497980621125963` | `0.00022135840521918403` | `0` | `0.004109886009246111` |
| `B15-C-NoiseNullReservoirReleaseSplit` | `9` | `0.10159635945790736` | `0.0001713616463045279` | `0.0008658601178063287` | `0` | `0.006989005953073502` |
| `B15-D-UnlabeledTemporalConsistency` | `9` | `0.03306356660957743` | `0.0000924355246954494` | `-0.00006248388025495742` | `0` | `0.0019148907158523798` |

### 5.2 Best-row extrema

| candidate | best_noise_delta | best_reservoir_delta | best_control_gap | max_logit_drift | max_CEp99_delta |
|---|---:|---:|---:|---:|---:|
| `B15-A-ResponseMatrixFunctionalQP` | `-0.0008949637413024902` | `-0.0024486780166625977` | `-0.046769863950319346` | `0.026134014129638672` | `0.005772590637207031` |
| `B15-B-SketchEigenspaceTargeting` | `-0.0018601864576339722` | `-0.0013582408428192139` | `-0.0008903844598354693` | `0.023835182189941406` | `0.0015861988067626953` |
| `B15-C-NoiseNullReservoirReleaseSplit` | `-0.0019992440938949585` | `-0.00019928812980651855` | `-0.019972895958527137` | `0.03661632537841797` | `0.0019083023071289062` |
| `B15-D-UnlabeledTemporalConsistency` | `-0.00030797719955444336` | `-0.0005331337451934814` | `-0.1258759443564671` | `0.012475967407226562` | `0.003845691680908203` |

解释：

```text
四个 B15 candidate 都能在 mean CouplingR2 上超过 0.02；
但 best_noise_delta 只到 -0.001999，best_reservoir_delta 只到 -0.002449，距离 -0.01 hard gate 仍差约 4-10 倍；
所有 candidate 的 best_control_gap 仍为负，说明效果仍可被 controls 解释；
logit drift 与 CEp99 没有成为主 blocker。
```

这排除了一个重要误解：本轮不是因为 safety 太严而失败，也不是因为 P3 gate 被 CEp99 / logit drift 卡死；失败来自 functional value 本身没有稳定进入 noise/reservoir release 通道。

## 6. P3 Promotion 与 Failure Table

Promotion summary：

| candidate_id | dataset_seed_pass_count | expected_dataset_seed_count | strong_promotion | weak_promotion | mean_control_gap |
|---|---:|---:|---:|---:|---:|
| `B15-A-ResponseMatrixFunctionalQP` | `0` | `9` | `0` | `0` | `-0.09585456127690224` |
| `B15-B-SketchEigenspaceTargeting` | `0` | `9` | `0` | `0` | `-0.08708917814324688` |
| `B15-C-NoiseNullReservoirReleaseSplit` | `0` | `9` | `0` | `0` | `-0.10020033887813308` |
| `B15-D-UnlabeledTemporalConsistency` | `0` | `9` | `0` | `0` | `-0.16740795076543694` |

Failure table：

| fail_reason | count |
|---|---:|
| `NoiseSignalLeak_delta>-0.01` | `36` |
| `RealSignalReservoirRatio_delta>-0.01` | `36` |
| `control_gap<0.005` | `36` |
| `CouplingR2_delta<0.02` | `2` |

解释：

```text
P3 的 36 个 candidate rows 全部同时失败 NoiseSignalLeak、Reservoir 和 control_gap；
只有 2 个 rows 失败 CouplingR2，说明 CouplingR2 已不是 hard blocker；
没有任何 candidate 达到 strong 或 weak promotion；
因此 P4 short-run 正确保持关闭。
```

## 7. Loss-Agnostic Audit

本轮 official eligible 的 U4-U9 与 B15-A/B/C/D 均满足：

```text
loss_agnostic_direction = 1
ce_vector_used_for_direction = 0
label_used_for_direction = 0
permuted_label_used_for_direction = 0
validation_used_for_commit = 0
dataset_name_used_for_commit = 0
```

`U2-TaskOnlyAdamW-Audit` 与 `U3-SNRGatedAdamWResidual-Audit` 被显式标记为 audit controls：

```text
loss_agnostic_direction = 0
label_used_for_direction = 1
official_eligible = 0
```

这两个 audit controls 没有进入 official B15 promotion claim。

## 8. 修改与修复审计

本轮新增：

```text
experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py
```

实现内容：

```text
A0 B320 anchor monitor；
C0 Line C response identifiability probes；
I0 primitive actuator truth probes；
B15-A ResponseMatrixFunctionalQP；
B15-B SketchEigenspaceTargeting；
B15-C NoiseNullReservoirReleaseSplit；
B15-D UnlabeledTemporalConsistency；
P3 3x3 promotion gate；
P4 open/close gate；
loss_agnostic_audit / provenance_audit / hash_manifest / route_decision；
执行复盘与结果复盘日志。
```

运行中发现并修复了两个 gate / reporting blocker：

```text
1. smoke run 暴露 weak-promotion gate 在非 3x3 设置下可能误开 P4。
   修复：只有 expected_dataset_seed_count >= 9 时才允许 strong/weak promotion。

2. 初版 route 判定在 CouplingR2 estimator 很强时过于宽松，会写 R3 partial。
   修复：按 v12.15 停止条件收紧；当 B15 全 0 pass、noise/reservoir 均未过 -0.01 且 control_gap 全负时，写 R4。
```

未修改：

```text
CE loss；
sampler；
class weight；
teacher / distillation；
dataset-name branch；
B320 architecture hyperparameters；
B-spline frozen policy。
```

## 8.1 Line D 补跑结果

原始执行确实漏掉了 Batch D focused repair；这不是计划漏项，而是执行漏项。漏项发现后已补跑 Line D official focused repair，并写回 v12.15 official 主目录：

```text
raw_dir = results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/lineD_official_focused_10_3x3
v1215_lineD_focused_repair_candidates.csv
v1215_lineD_focused_repair_family_status.csv
v1215_lineD_focused_repair_route.json
v1215_lineD_focused_repair_hash_manifest.json
```

补跑约束：

```text
不跑 B-spline；
不做全 family 大扫；
每个 active family 跑两个 focused repair；
所有结果来自真实 v1283 raw artifacts；
no_fake_pass = 1, no_proxy_pass = 1, no_cpu_offload_pass = 1。
```

### 8.1.1 Line D family 状态

| family | status | focused candidates | best L3 step ratio | A4 pass candidates | A5 pass candidates | blocker |
|---|---|---:|---:|---|---|---|
| `Rational` | `ExpressionBlocked` | `2` | `0.6000411442511098` | `` | `` | `L3 passed, A4 failed` |
| `RBF` | `ExpressionBlocked` | `2` | `0.48376549932276924` | `` | `` | `L3 passed, A4 failed` |
| `Chebyshev` | `TaskBlocked` | `2` | `0.5145371977455292` | `B3an` | `` | `L3/A4 passed, A5 failed` |
| `Fourier` | `ExpressionBlocked` | `2` | `0.47917220677116973` | `` | `` | `L3 passed, A4 failed` |
| `Wavelet` | `TaskBlocked` | `2` | `0.49426250776207276` | `B5p; B5q` | `` | `L3/A4 passed, A5 failed` |
| `BSpline` | `KernelBlocked` | `0` | `` | `` | `` | `frozen / not run` |

结论：

```text
lineD_completed = 1
lineD_focused_candidate_count = 10
lineD_family_pass_count = 0
lineD_family_linec_measured_count = 10
```

### 8.1.2 Candidate-level 结果

| family | candidate | L3 pass | step ratio | memory ratio | A4 | A5 | A5 mean delta | A5 worst delta | LineC CouplingR2 | LineC NoiseLeak | LineC Reservoir | LineC interpretation | failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `Chebyshev` | `B3ao` | `0` | `0.5145371977455292` | `1.3076190476190477` | `` | `` | `` | `` | `0.24499824647085133` | `0.2985441982746124` | `0.7571591138839722` | `coupling_collapse` | `L3_fused_kernel_efficiency_fail` |
| `RBF` | `B2aa` | `1` | `0.6312788268926616` | `1.0057142857142858` | `0` | `` | `` | `` | `0.022361796896410047` | `0.5679519176483154` | `0.4500548541545868` | `coupling_collapse` | `A4_expression_gate_fail_after_l3_efficiency` |
| `RBF` | `B2ab` | `1` | `0.48376549932276924` | `1.01` | `0` | `` | `` | `` | `0.050744908019064616` | `0.19130150973796844` | `0.5613080263137817` | `coupling_collapse` | `A4_expression_gate_fail_after_l3_efficiency` |
| `Chebyshev` | `B3an` | `1` | `0.5302275395798013` | `1.0276190476190477` | `1` | `0` | `-0.09331597222222222` | `-0.16796875` | `0.24450676085737766` | `0.1484554260969162` | `0.7575636506080627` | `coupling_collapse` | `A5_family_task_gate_fail` |
| `Fourier` | `B4v` | `1` | `0.5465024952147447` | `0.6309523809523809` | `0` | `` | `` | `` | `0.1051104966684111` | `0.28819921612739563` | `0.24673078954219818` | `coupling_collapse` | `A4_expression_gate_fail_after_l3_efficiency` |
| `Fourier` | `B4w` | `1` | `0.47917220677116973` | `0.12428571428571429` | `0` | `` | `` | `` | `0.1051104966684111` | `0.2881992757320404` | `0.24672988057136536` | `coupling_collapse` | `A4_expression_gate_fail_after_l3_efficiency` |
| `Wavelet` | `B5p` | `1` | `0.5305414416046225` | `1.0395238095238095` | `1` | `0` | `-0.09288194444444445` | `-0.14453125` | `0.2415788497830993` | `0.8224287629127502` | `0.6739217638969421` | `coupling_collapse` | `A5_family_task_gate_fail` |
| `Wavelet` | `B5q` | `1` | `0.49426250776207276` | `1.0395238095238095` | `1` | `0` | `-0.09288194444444445` | `-0.15234375` | `0.24572348473090033` | `0.823042094707489` | `0.6626569628715515` | `coupling_collapse` | `A5_family_task_gate_fail` |
| `Rational` | `B7lu` | `1` | `0.6235310596972393` | `0.9219047619047619` | `0` | `` | `` | `` | `0.2402957879132529` | `0.2299851030111313` | `0.21766093373298645` | `coupling_collapse` | `A4_expression_gate_fail_after_l3_efficiency` |
| `Rational` | `B7lv` | `1` | `0.6000411442511098` | `0.92` | `0` | `` | `` | `` | `0.24026501888557739` | `0.22998537123203278` | `0.21776250004768372` | `coupling_collapse` | `A4_expression_gate_fail_after_l3_efficiency` |

解释：

```text
1. Rational/RBF/Fourier 的 focused repairs 均能提供 L3 efficiency signal，但 A4 expression 没有打开。
2. Chebyshev B3an 与 Wavelet B5p/B5q 能打开 A4，但 3x3 A5 task gate 全失败。
3. 所有 10 个 Line C family summaries 都是 coupling_collapse。
4. 没有任何 Line D family 达到 FamilyPass 或 near-pass-to-functional 接入条件。
```

因此，Line D 补跑后的正式判断不是“未执行”，而是：

```text
Line D Batch D 已补跑完成，但 0/5 active family 通过；
当前 focused candidates 不应接入 B15/P3/P4；
主 route 仍保持 R4，P4 仍关闭。
```

## 9. 为什么不能进入 P4

v12.15 P4 最低条件要求：

```text
至少一个 loss-agnostic P3 survivor；
survivor 不是 single dataset / single seed；
NoOp / Random / AdamWParallel / SNR-only controls 被击败；
NoiseSignalLeak 和 RealSignalReservoirRatio 同时过 -0.01。
```

本轮实际结果：

```text
p3_survivor_count = 0
dataset_seed_pass_count = 0/9 for all B15 candidates
NoiseSignalLeak_delta>-0.01 = 36/36
RealSignalReservoirRatio_delta>-0.01 = 36/36
control_gap<0.005 = 36/36
```

因此如果继续跑 P4，会违反计划。P4 不开不是保守，而是防止把 one-step coupling movement 或 smoke artifact 写成 functional success。

## 10. 结论与下一步

v12.15 把 v12.14 的 blocker 进一步拆清楚了：

```text
1. Line C measurement 不是纯假阳性；CouplingR2_delta 可被 sketch_delta_fro 预测。
2. 当前 loss-agnostic estimator 不能可靠预测 NoiseSignalLeak / Reservoir release。
3. 当前 primitive actuators 全部 ActuatorWeak，没有真正改变 active gradient-sketch projector。
4. B15-A/B/C/D 都无法产生 control-resistant、noise/reservoir 双通过的 P3 survivor。
5. Line D Batch D 已补跑，10 个 focused repairs 产生 0 FamilyPass，且全部 family Line C summaries 为 coupling_collapse。
```

正式 route：

```text
R4-LossAgnosticFunctionalMechanismNotFound
```

下一步不应继续：

```text
cotangent count sweep；
role coefficient grid；
window/lambda 小网格；
CouplingR2-only promotion；
CE-specific projector 包装成 official；
B320 base 小修；
继续把本轮 Line D focused candidates 包装为 official functional/base repair。
```

下一步应转向：

```text
更高层 loss-agnostic signal estimator；
重新定义或扩展 Line C sketch construction；
更深 primitive-level instrumentation，使 actuator 真正达到 sketch_delta_fro >= 0.01 和 projector_angle >= 1 度；
Line D 若再开新版本，应先提出新的 expression/task/coupling-collapse repair hypothesis，而不是重复 B7lu/B7lv、B2aa/B2ab、B3an/B3ao、B4v/B4w、B5p/B5q；
在新的 estimator/actuator evidence 出现前，不再继续当前 cotangent / role / moment mechanism family。
```

## 11. v12.15 continuation 结果复盘

生成时间：`2026-05-24T02:10:28Z`

本节仍属于 v12.15。它不是 v12.16，也不是新版本命名。执行动机来自上文 route：

```text
1. 当前 cotangent / role / moment / shallow primitive actuator family 不能过 P3。
2. 不继续做小网格和 CouplingR2-only promotion。
3. 先尝试更深 primitive instrumentation；
4. 如果 actuator gate 打开，再用 gate-open actuator 构造 P3 continuation candidate。
```

新增代码：

```text
experiments/run_v1215_continuation_actuator_budget_repair.py
experiments/run_v1215_continuation_p3_from_fused_actuator.py
```

新增输出目录：

```text
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_actuator_budget_repair_3x3_b32_w5
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_fused_primitive_actuator_3x3_b32_w5
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_fused_quad_sign_3x3_b32_w5
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_bidir_selector_3x3_b32_w5
```

### 11.1 修复方向与修改审计

本轮 continuation 做了三类修改：

```text
1. Safety-capped actuator budget repair：
   对 J1..J8 cotangent-derived role actuator 加大 budget_multiplier = 2,4,8，
   但用 train-batch unlabeled logit drift cap 控制，不用 CE vector / label-loss VJP。

2. Fused primitive actuator：
   新增 K1..K6 primitive_param_space actuator，
   直接在 B320 active primitive role 参数上构造 loss-agnostic perturbation。
   目标是验证之前 ActuatorWeak 是否只是 cotangent VJP actuator 太浅。

3. Gate-open P3 continuation：
   只把 3x3 中实际打开 actuator gate 的 K6-FusedQuadSignBudgetCap 接入 B15-E P3，
   不把 smoke gate-open 或 partial actuator movement 写成 survivor。
```

实现中遇到并修复两个代码 blocker：

```text
1. cotangent count=64 在当前 logit cotangent shape 下 reshape 失败。
   修复：收回到 v12.15 已验证的 count=32。

2. safety cap 初版调用 prev._apply_delta，但 apply_delta 实际在 v1252。
   修复：safety_cap_delta / logit_drift_on_batch 改用 v1252._apply_delta。
```

这些修复只改变 continuation runner 的执行路径，不修改 B320 architecture，不修改 CE loss、sampler、class weight、teacher/distillation，也不引入 dataset-name branch。

### 11.2 Budget-only actuator repair 结果

来源：

```text
continuation_actuator_budget_repair_3x3_b32_w5/v1215_continuation_route_decision.json
continuation_actuator_budget_repair_3x3_b32_w5/v1215_continuation_actuator_budget_summary.csv
continuation_actuator_budget_repair_3x3_b32_w5/v1215_continuation_estimator_cv.csv
```

route：

```text
route = R4-ContinuationActuatorBudgetRepairStillWeak
repair_rows = 216
actuator_gate_pass_rows = 0
max_repair_sketch_delta_fro = 0.008185695856809616
max_repair_projector_angle_deg = 1.8827358484268188
best_repair_noise_delta = -0.006306260824203491
best_repair_reservoir_delta = -0.014425188302993774
max_noise_estimator_abs_spearman_oos = 0.12781885091005538
max_reservoir_estimator_abs_spearman_oos = 0.21536208820961664
hard_noise_release_actual_support = 0
hard_reservoir_release_actual_support = 1
p4_open = 0
official_p3_claim = 0
```

主要解释：

```text
1. budget-only 修复能把 projector_angle 推过 1 度，但 sketch_delta_fro 最高只有 0.0081857，仍低于 0.01。
2. 单行 reservoir 能到 -0.014425，但不是 actuator gate pass，也没有形成可部署 estimator。
3. higher-level estimator 的 OOS Spearman 仍弱：Noise 0.1278，Reservoir 0.2154。
4. 因此不能把 budget-only 结果接入 P3 survivor。
```

### 11.3 Fused primitive actuator 结果

来源：

```text
continuation_fused_primitive_actuator_3x3_b32_w5/v1215_continuation_route_decision.json
continuation_fused_primitive_actuator_3x3_b32_w5/v1215_continuation_actuator_budget_summary.csv
continuation_fused_primitive_actuator_3x3_b32_w5/v1215_continuation_estimator_cv.csv
```

route：

```text
route = R4-ContinuationActuatorGateOpenedButP3NotClaimed
repair_rows = 378
actuator_gate_pass_rows = 1
max_repair_sketch_delta_fro = 0.013528571464121342
max_repair_projector_angle_deg = 4.587743282318115
best_repair_noise_delta = -0.006748616695404053
best_repair_reservoir_delta = -0.014425188302993774
max_noise_estimator_abs_spearman_oos = 0.1506627825955557
max_reservoir_estimator_abs_spearman_oos = 0.15104559306239979
p4_open = 0
official_p3_claim = 0
```

gate-open 行：

| field | value |
|---|---:|
| `repair_id` | `K6-FusedQuadSignBudgetCap` |
| `repair_family` | `fused_quad_sign_budget_repair` |
| `role_recipe` | `primitive_param_sign:quad` |
| `dataset` | `KMNIST` |
| `seed` | `2` |
| `budget_multiplier` | `8.0` |
| `requested_logit_drift_train` | `0.1018221378326416` |
| `safety_cap_scale` | `0.4419471144277442` |
| `capped_logit_drift_train` | `0.045` |
| `logit_max_abs_drift` | `0.040654659271240234` |
| `sketch_delta_fro` | `0.010541788302361965` |
| `signal_projector_angle_deg` | `4.587743282318115` |
| `reservoir_projector_angle_deg` | `4.587273597717285` |
| `NoiseSignalLeak_delta` | `-0.0023810267448425293` |
| `RealSignalReservoirRatio_delta` | `-0.0028071701526641846` |
| `CouplingR2_delta` | `0.2872958679372558` |
| `classification_failure_mode` | `ActuatorGateOpenButObjectiveMisaligned` |

K6 family summary：

```text
rows = 27
actuator_gate_pass_rows = 1
mean_sketch_delta_fro = 0.006521449157002347
max_sketch_delta_fro = 0.013528571464121342
mean_projector_angle_deg = 0.6906187338409601
max_projector_angle_deg = 4.587743282318115
best_noise_delta = -0.003466993570327759
best_reservoir_delta = -0.005521595478057861
mean_CouplingR2_delta = 0.27418670364944897
failure_mode_counts = {
  "ActuatorGateOpenButObjectiveMisaligned": 1,
  "ActuatorPartiallyImproved": 4,
  "ActuatorWeak": 18,
  "SafetyRejected": 4
}
```

解释：

```text
Fused primitive actuator 证明 v12.15 的 ActuatorWeak 不是绝对不可破。
直接作用 active primitive quad 参数后，确实可以同时达到 sketch_delta_fro >= 0.01 和 projector_angle >= 1 度。

但 gate-open 行的 noise/reservoir release 只有 -0.00238 / -0.00281，远低于 -0.01；
分类为 ActuatorGateOpenButObjectiveMisaligned。
这把 blocker 从“actuator 完全太弱”推进为“actuator 可以动 active sketch，但 objective 仍没有对准 noise/reservoir release”。
```

### 11.4 Fixed K6 P3 结果

来源：

```text
continuation_p3_fused_quad_sign_3x3_b32_w5/v1215_continuation_p3_route_decision.json
continuation_p3_fused_quad_sign_3x3_b32_w5/v1215_continuation_p3_summary.csv
```

route：

```text
route = R4-ContinuationFusedActuatorP3Failed
candidate_rows = 9
p3_pass_rows = 0
p4_open = 0
official_p3_claim = 0
max_sketch_delta_fro = 0.014910714700818062
max_projector_angle_deg = 1.2201300859451294
best_noise_delta = -0.008338093757629395
best_reservoir_delta = -0.007685840129852295
```

summary：

| metric | value |
|---|---:|
| `candidate_id` | `B15-E-FusedQuadSignGateOpenActuator` |
| `rows` | `9` |
| `p3_pass_rows` | `0` |
| `strong_promotion` | `0` |
| `weak_promotion` | `0` |
| `mean_CouplingR2_delta` | `0.30535509533270794` |
| `mean_NoiseSignalLeak_delta` | `-0.00016311597492959764` |
| `mean_RealSignalReservoirRatio_delta` | `-0.0011343707640965779` |
| `best_noise_delta` | `-0.008338093757629395` |
| `best_reservoir_delta` | `-0.007685840129852295` |
| `mean_control_gap` | `-0.05098838506361029` |
| `best_control_gap` | `0.05770214754022507` |
| `max_logit_max_abs_drift` | `0.06440162658691406` |

fail counts：

```text
NoiseSignalLeak_delta>-0.01 = 9
RealSignalReservoirRatio_delta>-0.01 = 9
control_gap<0.005 = 7
logit_max_abs_drift>0.05 = 4
```

解释：

```text
Fixed K6 已经能把 sketch_delta_fro 推到 0.01491，说明 fused primitive actuator 不是 no-op。
但它仍不能把 NoiseSignalLeak / Reservoir 同时推过 -0.01；
best row 接近 hard gate，但所有 9 行都失败 noise/reservoir；
同时 4/9 行 holdout logit drift 超过 0.05。
因此不能开 P4，也不能写成 P3 survivor。
```

### 11.5 Bidirectional selector P3 结果

来源：

```text
continuation_p3_bidir_selector_3x3_b32_w5/v1215_continuation_p3_route_decision.json
continuation_p3_bidir_selector_3x3_b32_w5/v1215_continuation_p3_summary.csv
continuation_p3_bidir_selector_3x3_b32_w5/v1215_continuation_p3_selector_probes.csv
```

设置：

```text
candidate_mode = bidirectional_noise_reservoir_selector
selector_probe_budget_multiplier = 2.0
budget_multiplier = 8.0
target_logit_drift_train = 0.04
```

route：

```text
route = R4-ContinuationFusedActuatorP3Failed
candidate_rows = 9
p3_pass_rows = 0
p4_open = 0
official_p3_claim = 0
max_sketch_delta_fro = 0.010474889539182186
max_projector_angle_deg = 0.9291486144065857
best_noise_delta = -0.0035085678100585938
best_reservoir_delta = -0.002069234848022461
```

summary：

| metric | value |
|---|---:|
| `candidate_id` | `B15-E-FusedQuadSignGateOpenActuator` |
| `rows` | `9` |
| `p3_pass_rows` | `0` |
| `strong_promotion` | `0` |
| `weak_promotion` | `0` |
| `mean_CouplingR2_delta` | `0.26559342953255755` |
| `mean_NoiseSignalLeak_delta` | `-2.0682811737060547e-05` |
| `mean_RealSignalReservoirRatio_delta` | `0.00018160541852315268` |
| `best_noise_delta` | `-0.0035085678100585938` |
| `best_reservoir_delta` | `-0.002069234848022461` |
| `mean_control_gap` | `-0.091467112952939` |
| `best_control_gap` | `0.0012455436961674726` |
| `max_logit_max_abs_drift` | `0.0418238639831543` |

fail counts：

```text
NoiseSignalLeak_delta>-0.01 = 9
RealSignalReservoirRatio_delta>-0.01 = 9
control_gap<0.005 = 9
```

解释：

```text
Bidirectional selector 成功消除了 fixed K6 中的 holdout logit drift failure；
但代价是 effect size 明显变小，projector_angle 最高只有 0.929 度；
noise/reservoir release 更弱，所有 9 行仍失败。

这说明简单的 + / - direction selector 不是足够的 explicit noise/reservoir spectral target。
当前问题不是只差方向符号，而是 objective 本身仍没有构造出可重复的 noise/reservoir release basis。
```

### 11.6 continuation 后的正式判断

v12.15 continuation 的真实进展：

```text
1. 原始 I0 的“全部 ActuatorWeak”被细化：
   cotangent-derived budget repair 仍 weak；
   fused primitive param actuator 可以打开 actuator gate，但只在 1/378 行打开。

2. Actuator blocker 被部分解除：
   max_sketch_delta_fro 从原 I0 的 0.0038212935905903578 提高到 0.013528571464121342；
   max_projector_angle 从原 I0 的 0.7409852743148804 提高到 4.587743282318115。

3. 但 P3 blocker 没解除：
   Fixed K6 P3 = 0/9；
   Bidirectional selector P3 = 0/9；
   两次 P3 的 NoiseSignalLeak / Reservoir hard gate 都是 9/9 失败。

4. higher-level estimator 仍不够：
   fused continuation 中 Noise OOS Spearman 最大 0.1506627825955557；
   Reservoir OOS Spearman 最大 0.15104559306239979。
```

因此 continuation 后的 v12.15 route 是：

```text
R4-ContinuationFusedActuatorP3Failed
```

新的 blocker 判断：

```text
不是 B320 base；
不是 Line D 漏跑；
不是 safety 过严单独导致；
不是 actuator 完全动不了；
而是：即使 fused primitive actuator 能动 active sketch/projector，
当前 objective/selector 仍不能稳定把这个 movement 转化成 NoiseSignalLeak <= -0.01
和 RealSignalReservoirRatio <= -0.01 的 control-resistant release。
```

下一步不应继续：

```text
重复 budget_multiplier 2/4/8/12；
重复 K6 fixed sign；
重复 + / - bidirectional selector；
用 CouplingR2 高值包装为 P3；
因为 K6 单行 actuator gate-open 就打开 P4。
```

下一步应做：

```text
重新定义 explicit noise/reservoir spectral target；
或重建 Line C sketch construction，使 noise/reservoir basis 在 direction generation 时可见；
然后再把 fused primitive actuator 作为执行器，而不是把它本身当 value source。
```
