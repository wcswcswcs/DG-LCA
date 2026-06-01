# DG-KAN v14.4 RealTransferFMS AllBasisSubstrate 实验结果复盘

生成时间：2026-05-29（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 6/9 exploration positive 写成 S5，不把 MLP generic control 或局部 positive row 写成 promotion。

## 1. 计划理解

v14.4 的目标是在 v14.3 已经打开 S3 synthetic / S4 real short-run 的基础上，验证 real-transfer FMS 是否能在真实 3x3 上稳定达到 S5：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
S5 gate = 9/9 dataset-seed pass
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. direction 只能使用 train-stream loss gradient / FMS state / train proxy / basis-role projection。
5. validation / test / future / query batch 不能生成方向。
6. LineC / CEp99 / NLL / ECE / Brier 只能作为 audit / gate，不能作为方向源。
7. 6/9 但不到 9/9 只能写成 S4b-RealTransferExplorationPositive，不能写成 S5。
8. promotion_allowed 只有 S5 达成后才允许打开；本轮始终为 0。
```

计划中的 blocker 修复顺序：

```text
1. 若 real 仍停在 4/9，先分解 source / LineC / tail / AUCtime failure。
2. 按计划尝试 K-RT1..K-RT5：
   K-RT1 split agreement
   K-RT2 train-stream tail trust
   K-RT3 projection value retention
   K-RT4 source-tail co-state
   K-RT5 delayed basis constraint
3. 若仍不足，继续尝试低 plasticity、tail trust、split/delayed、refresh interval 等 train-stream-only 修复。
4. 不允许使用 audit metric 或 dataset-name branch 去调方向。
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

主要实现：

```text
1. 新增 v14.4 real-transfer runner 与 artifact surface。
2. 实现 Rational / KAN real-transfer methods：
   K0-RAT-AdamW
   K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint
   K-RT1-TrainSplitAgreement
   K-RT2-TrainStreamTailTrust
   K-RT3-ProjectionValueRetention
   K-RT4-SourceTailCoState
   K-RT5-DelayedBasisConstraint
   KCTRL-RandomMatchedProjection
3. 实现 MLP generic controls：
   MLP-AdamW
   MLP-FMS-Amortized
   MLP-FMS-SplitAgreement
   MLP-FMS-SourceTailCoState
4. K-RT direction 只使用 train-stream gradient / FMS state / train proxy：
   split agreement、train loss q95、train margin p10、train logit RMS、
   train entropy、projection retention、source-tail co-state。
5. exact LineC / CEp99 / NLL / ECE 只作为 audit 和 gate。
6. 输出 v144 route、manifest、forbidden audit、real results / summary、
   split agreement、projection retention、source-tail co-state、
   failure table、LineC / tail audit、all-basis status 和 figures。
```

后续 blocker 修复：

```text
1. smoke 首跑发现 eval_metrics import 缺失。
2. 修复：
   from experiments.run_v133_task_family_robust_basis_natural import eval_metrics
3. py_compile 通过。
```

合法性说明：

```text
1. v144_forbidden_information_audit.csv 记录：
   direction_uses_validation_test_future_query = 0
   direction_uses_linec_cep99_nll_ece = 0
2. 所有 repair 均为 compute-budgeted run。
3. 不降低 S5 gate，不把 S4b 写成 S5。
4. promotion_allowed 始终为 0。
```

## 3. Smoke

首跑 smoke：

```text
datasets = MNIST
seeds = 0
methods = K0,K-RT1,KCTRL
mlp_methods = MLP-AdamW,MLP-FMS-SplitAgreement
train_steps = 4
batch_size = 8
```

首跑 blocker：

```text
NameError: name 'eval_metrics' is not defined
```

修复后 smoke 结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
expected_dataset_seed_count = 1
real_dataset_seed_pass_count = 0
real_short_run_pass_rows = 0
mean_source_vs_best_control_noncontrol = -1.1983697414398193
required_artifact_missing_count = 0
promotion_allowed = 0
```

解释：

```text
smoke 只证明 runner / artifact / real-transfer surface 可执行；
不能作为 success 或 no-go official 证据。
```

## 4. Official compute-budgeted 结果

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K8,K-RT1,K-RT2,K-RT3,K-RT4,K-RT5,KCTRL
mlp_methods = MLP-AdamW,MLP-FMS-Amortized,MLP-FMS-SplitAgreement,MLP-FMS-SourceTailCoState
train_steps = 200
batch_size = 32
lr = 0.005
fms_strength = 0.10
fms_update_interval = 80
linec_mode = exact
compute_budgeted_run = 1
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 3 / 9
real_short_run_pass_rows = 7
mean_source_vs_best_control_noncontrol = -0.09544556670718723
required_artifact_missing_count = 0
promotion_allowed = 0
```

failure counts：

```text
AUCtime = 36
source = 31
CEp99_tail = 26
NLL_tail = 21
ECE_tail = 10
LineC = 1
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.114219 | 1.0620 | 9 |
| K-RT2-TrainStreamTailTrust | 0 | 0 | -0.0656 | 1.1027 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.02205 | 1.0508 | 9 |
| K-RT4-SourceTailCoState | 0 | 0 | -0.1845 | 1.0971 | 9 |
| K-RT5-DelayedBasisConstraint | 2 | 2 | -0.1042 | 1.0709 | 9 |
| K8 baseline | 1 | 1 | -0.1262 | 0.9864 | 8 |
| K0 / KCTRL | 0 | 0 | - | - | - |

判断：

```text
1. Official compute-budgeted run 没有达到 S5。
2. LineC 已不是主要 blocker；主要 blocker 是 source instability、AUCtime 与 tail。
3. 必须按计划继续 repair，不能停止在 3/9。
```

## 5. Repair 1：low plasticity / lower lambda

触发原因：

```text
official run 的 failure counts 显示 AUCtime / tail / source 同时阻塞；
计划建议降低过强 FMS plasticity 与 projection lambda，先减少 real-transfer 过冲。
```

执行规模：

```text
methods = K0,K-RT1,K-RT2,K-RT3,K-RT5,KCTRL
skip_mlp_control = 1
train_steps = 200
lr = 0.005
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.5
```

结果：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 9
mean_source_vs_best_control_noncontrol = -0.03511386281914181
required_artifact_missing_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.10150694184833103 | 1.0673764713627447 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.02155327796936035 | 1.0695467503638163 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.021980126698811848 | 1.050731870366617 | 9 |
| K-RT5-DelayedBasisConstraint | 4 | 4 | -0.08248191409640843 | 1.095324834920501 | 9 |
| K0-RAT-AdamW | 0 | 0 | - | - | - |
| KCTRL-RandomMatchedProjection | 0 | 0 | - | - | - |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-RT1 | 0.113161 | 0.989009 | -1.502634 | -0.224621 | -0.014783 | 1 |
| MNIST | 1 | K-RT1 | 0.225544 | 0.883245 | -6.995262 | -0.238807 | -0.021089 | 1 |
| MNIST | 1 | K-RT3 | 0.226790 | 0.898311 | -4.681700 | -0.240052 | -0.020003 | 1 |
| MNIST | 1 | K-RT5 | 0.182434 | 0.976370 | -5.420082 | -0.195696 | -0.028673 | 1 |
| MNIST | 2 | K-RT1 | 0.231780 | 0.943324 | -1.094994 | -0.298710 | -0.042674 | 1 |
| MNIST | 2 | K-RT5 | 0.057659 | 0.918473 | -0.084893 | -0.124589 | -0.018150 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.027344 | 0.908514 | -2.434628 | -0.047216 | 0.000262 | 1 |
| KMNIST | 0 | K-RT2 | 0.051924 | 0.989902 | -6.277489 | -0.051924 | -0.002071 | 1 |
| KMNIST | 1 | K-RT5 | 0.027283 | 0.965196 | -0.074760 | -0.518914 | -0.045063 | 1 |

remaining failed dataset-seeds：

```text
Fashion-MNIST seed0:
  best rows can be source/AUC positive, but CEp99 tail fails.
Fashion-MNIST seed2:
  best rows can be source/tail positive, but AUCtime fails.
KMNIST seed2:
  source can be positive, but AUCtime and CEp99 tail fail.
```

判断：

```text
1. low plasticity / lower lambda 将 best pass 从 3/9 提升到 6/9。
2. 这达到计划定义的 S4b exploration positive。
3. 但 S5 要求 9/9，因此 promotion_allowed 仍为 0。
4. 不能把 S4b 写成 official S5 success。
```

## 6. Repair 2：stronger train-stream tail trust

触发原因：

```text
lowplasticity 后剩余 blocker 仍包含 CEp99 / NLL tail；
按计划尝试更强的 train-stream tail trust 与更低 lambda。
```

执行规模：

```text
methods = K0,K-RT2,K-RT4,K-RT5,KCTRL
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.35
rt_state_beta = 0.80
rt_risk_scale = 8.0
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 5 / 9
real_short_run_pass_rows = 6
mean_source_vs_best_control_noncontrol = -0.05162593832722417
required_artifact_missing_count = 0
promotion_allowed = 0
```

判断：

```text
stronger tail trust 降低了过冲风险的一部分，但牺牲 source；
结果从 6/9 退到 5/9，没有打开 S5。
```

## 7. Repair 3：moderate split / delayed basis

触发原因：

```text
official 与 lowplasticity 均显示 K-RT1 / K-RT5 是相对最有效方法；
继续尝试更频繁 FMS refresh 与更温和 split agreement threshold。
```

执行规模：

```text
methods = K0,K-RT1,K-RT3,K-RT5,KCTRL
fms_strength = 0.05
fms_update_interval = 40
rt_lambda_max = 0.65
rt_agreement_a0 = -0.20
rt_agreement_a1 = 0.40
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 5 / 9
real_short_run_pass_rows = 8
mean_source_vs_best_control_noncontrol = -0.06597101467627066
required_artifact_missing_count = 0
promotion_allowed = 0
```

判断：

```text
moderate split/delayed 增加了 pass rows，但 unique dataset-seed pass 只有 5/9；
没有超过 lowplasticity best 6/9。
```

## 8. Repair 4：slow refresh

触发原因：

```text
检查 FMS refresh 频率是否导致 real-transfer source/tail 不稳定。
```

执行规模：

```text
methods = K0,K-RT1,K-RT2,K-RT3,K-RT5,KCTRL
fms_strength = 0.05
fms_update_interval = 160
rt_lambda_max = 0.5
```

结果：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 9
mean_source_vs_best_control_noncontrol = -0.03129612737231784
required_artifact_missing_count = 0
promotion_allowed = 0
```

判断：

```text
slow refresh 保持 6/9，但没有新增通过的 dataset-seed；
refresh interval 不是打开剩余 3/9 的充分机制。
```

## 9. Required artifacts

主要 run：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/smoke_v144/
results/v14_4_real_transfer_fms_all_basis_substrate/official_v144/
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_lowplasticity_lambda05/
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_tailtrust_strong/
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_split_delayed_moderate/
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_slowrefresh160_lambda05/
```

每个 run 均输出：

```text
v144_route_decision.json
v144_required_manifest.csv
v144_forbidden_information_audit.csv
v144_code_review_manifest.csv
v144_real_transfer_fms_results.csv
v144_real_transfer_fms_summary.csv
v144_train_stream_proxy.csv
v144_projection_value_retention.csv
v144_source_tail_costate.csv
v144_split_agreement.csv
v144_real_3x3_failure_table.csv
v144_wavelet_substrate_hardening.csv
v144_rbf_substrate_repair.csv
v144_chebyshev_lifetime_repair.csv
v144_fourier_lifetime_repair.csv
v144_all_basis_substrate_status.csv
v144_mlp_generic_fms_control.csv
v144_kan_specific_advantage.csv
v144_linec_audit.csv
v144_tail_calibration_audit.csv
fig_*.svg
```

artifact completeness：

```text
所有记录到 route 的 run：
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 10. 最终科学结论

v14.4 没有达成 S5-OfficialFunctionalSuccess；最终 best 合法 route：

```text
best route = S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
best real_short_run_pass_rows = 9
official_s5_reached = 0
promotion_allowed = 0
```

已闭合事实：

```text
1. v14.3 best real 3x3 为 4/9；v14.4 best repair 推进到 6/9。
2. K-RT1 / K-RT5 是本轮最稳定的 real-transfer methods。
3. 低 plasticity / lower lambda 是有效修复方向，说明原先存在过强 FMS / projection 过冲。
4. stronger tail trust、moderate split/delayed、slow refresh 都没有超过 6/9。
5. LineC 不是当前主要 blocker；主要 blocker 是 source instability、AUCtime、CEp99/NLL tail 的组合。
6. Fashion-MNIST seed0、Fashion-MNIST seed2、KMNIST seed2 仍未稳定通过。
7. required artifacts 缺失为 0；forbidden direction audit 没有发现使用 validation/test/future/query 或 LineC/tail metric 生成方向。
```

no-go boundary：

```text
1. 6/9 是 S4b exploration positive，不是 S5。
2. 不能把 pass rows = 9 写成 9/9，因为 unique dataset-seed pass 只有 6/9。
3. 不能用 CEp99/NLL/ECE/LineC audit metric 反向调方向。
4. 不能做 dataset-name branch 去专门修 Fashion-MNIST 或 KMNIST。
5. 不允许 promotion。
```

当前边界：

```text
本轮已按 v14.4 计划尝试：
K-RT1 split agreement、
K-RT2 train-stream tail trust、
K-RT3 projection value retention、
K-RT4 source-tail co-state、
K-RT5 delayed basis constraint、
low plasticity / lower lambda、
stronger train-stream tail trust、
moderate split/delayed、
slow FMS refresh。

我现在不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不把 CEp99/NLL/ECE/LineC audit metric 用作 direction、
不做 dataset-name branch、
不把 6/9 exploration positive 写成 S5 official success。

下一步需要新的 real-transfer 计划：
train-stream-only 地同时解决 remaining dataset-seed 的 source instability、
AUCtime overhead 与 CEp99/NLL tail blocker。
```

## 11. 用户再次追问后的继续修复

用户再次要求“未达成则继续”。本次重新对照 v14.4 failure-response rules：

```text
If source improves but tail fails:
  Try train-stream tail trust / risk state.
If tail improves but source fails:
  Try projection value retention / split agreement.
If LineC fails:
  Try delayed basis constraint or weaker constraint phase.
Do not use CEp99 / LineC target direction.
Do not use seed-specific scaling.
```

best lowplasticity 剩余失败集中在：

| dataset | seed | closest method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 0 | K-RT3 | 0.184902 | 0.873954 | 0.166046 | -0.473994 | -0.046205 | 1 | CEp99 tail |
| Fashion-MNIST | 2 | K-RT2 | 0.162050 | 1.232769 | -0.704851 | -0.162050 | -0.007385 | 1 | AUCtime |
| KMNIST | 2 | K-RT1 | 0.094257 | 1.110939 | 2.585526 | -0.171631 | -0.016695 | 1 | AUCtime + CEp99 tail |

判断：

```text
LineC 已基本不是剩余 blocker；
剩余是 source / AUCtime / CEp99 tail 的交叉问题。
```

### 11.1 lowplasticity + train-stream output geometry

触发原因：

```text
v14.4 runner 已实现 label-free train-stream output geometry；
该规则只使用 train logits entropy / RMS，不使用 labels、validation/test、LineC 或 tail audit。
用它检查是否能压低输出几何过冲，从而改善 CEp99 tail。
```

执行规模：

```text
methods = K0,K-RT1,K-RT2,K-RT3,K-RT5,KCTRL
train_steps = 200
lr = 0.005
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.5
output_geometry_repair = train_entropy_t080_100_else050
compute_budgeted_run = 1
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 0 / 9
real_short_run_pass_rows = 0
mean_source_vs_best_control_noncontrol = -0.0064664847320980495
required_artifact_missing_count = 0
promotion_allowed = 0
```

failure counts：

```text
LineC = 32
source = 27
AUCtime = 22
ECE_tail = 14
CEp99_tail = 9
NLL_tail = 4
```

判断：

```text
train-stream entropy output geometry 没有打开 S5；
它从 best 6/9 退化到 0/9，并引入大量 LineC / source failure。
因此不能作为 v14.4 real-transfer 修复。
```

### 11.2 ultralow plasticity

触发原因：

```text
lowplasticity 从 3/9 提升到 6/9，说明过强 FMS / projection 是 blocker；
继续检查更低 fms_strength 是否能压下 CEp99/AUCtime 过冲。
```

执行规模：

```text
methods = K0,K-RT1,K-RT2,K-RT3,K-RT5,KCTRL
train_steps = 200
lr = 0.005
fms_strength = 0.025
fms_update_interval = 80
rt_lambda_max = 0.5
compute_budgeted_run = 1
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 5 / 9
real_short_run_pass_rows = 8
mean_source_vs_best_control_noncontrol = -0.03621196746826172
required_artifact_missing_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.09185847308900622 | 1.0701243092652721 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.010098596413930258 | 1.0682185684632735 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.021993809276156955 | 1.0507222223219834 | 8 |
| K-RT5-DelayedBasisConstraint | 3 | 3 | -0.08508180247412787 | 1.0947689743379645 | 9 |

failure counts：

```text
AUCtime = 22
CEp99_tail = 20
source = 14
NLL_tail = 10
ECE_tail = 4
LineC = 1
```

判断：

```text
ultralow plasticity 从 best 6/9 退化到 5/9；
继续简单降低 FMS strength 会牺牲 source，不能打开 S5。
```

## 12. 最终判断更新

v14.4 仍未达成 S5-OfficialFunctionalSuccess；最终 best 合法 route 仍是：

```text
best route = S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
best real_short_run_pass_rows = 9
official_s5_reached = 0
promotion_allowed = 0
```

本次追问后新增闭合事实：

```text
1. train-stream entropy output geometry 不适合作为 v14.4 real-transfer 修复；
   它退化到 0/9。
2. ultralow plasticity 不超过 lowplasticity best；
   它退化到 5/9。
3. best 仍是 repair_v144_lowplasticity_lambda05 或 slowrefresh160 的 6/9。
4. S5 仍未达成，不允许 promotion。
```

当前边界更新：

```text
当前 v14.4 内已尝试：
K-RT1 split agreement、
K-RT2 train-stream tail trust、
K-RT3 projection value retention、
K-RT4 source-tail co-state、
K-RT5 delayed basis constraint、
low plasticity / lower lambda、
stronger train-stream tail trust、
moderate split/delayed、
slow FMS refresh、
train-stream entropy output geometry、
ultralow FMS plasticity。

我现在仍不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不把 CEp99/NLL/ECE/LineC audit metric 用作 direction、
不做 dataset-name branch、
不使用 seed-specific scaling、
不把 6/9 exploration positive 写成 S5 official success。
```

## 13. 用户再次追问后的计划覆盖复核

本次没有新增训练；只对照 v14.4 完整计划复核是否仍有未覆盖的合法推荐方向。

复核范围：

```text
Line K: Rational Real-Transfer Value Constraint
Line W: Wavelet substrate hardening
Line D: All-basis substrate repair
Line G: Generic MLP-FMS control
Line C/R/Z: audit / provenance / finalizer
Failure-response rules
```

复核结论：

```text
1. Line K 的 K-RT1..K-RT5 已全部实现并运行。
2. source improves but tail fails -> train-stream tail trust / risk state 已尝试。
3. tail improves but source fails -> projection retention / split agreement 已尝试。
4. LineC fails -> delayed basis / weaker phase / output geometry 已尝试。
5. 额外 low plasticity、ultralow plasticity、slow refresh 也已尝试。
6. Line W / D 是 bounded parallel，不阻塞 Rational S5；
   v14.3 已完成主要 Wavelet/RBF/CHE/FOU/WAV substrate repair，
   v14.4 artifacts 也保留 all-basis substrate status。
7. Line G generic controls 已在 official_v144 运行；
   后续 repair 使用 --skip-mlp-control 节省预算，但没有把 MLP 写成 KAN success。
8. Line C/R/Z 所需 audit / manifest / route 均已输出。
```

最新 route 汇总仍为：

```text
best route = S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
official_s5_reached = 0
promotion_allowed = 0
```

最终判断：

```text
v14.4 仍未达成 S5。
当前完整计划中的推荐修复方向已经覆盖；
新增 train-stream entropy output geometry 与 ultralow plasticity 均没有超过 6/9。

我现在不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不把 CEp99/NLL/ECE/LineC audit metric 用作 direction、
不做 dataset-name branch、
不使用 seed-specific scaling、
不把 S4b 6/9 编造成 S5 official success。
```

## 14. 用户再次追问后的 low-lr repair

用户再次要求继续。本次在 best lowplasticity 配置上尝试一个仍合法的窄修复：

```text
lr 从 0.005 降到 0.003；
fms_strength = 0.05；
rt_lambda_max = 0.5；
不使用 CEp99/NLL/ECE/LineC 作为 direction；
不做 dataset-name branch；
不做 seed-specific scaling。
```

触发原因：

```text
best lowplasticity 的剩余 blocker 是 CEp99/AUCtime/source 交叉。
降低 lr 可能减少 step-to-step 过冲和 tail 波动；
但必须全 3x3 统一运行，不能只针对失败 seed。
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 5 / 9
real_short_run_pass_rows = 5
mean_source_vs_best_control_noncontrol = 0.04683350192175971
required_artifact_missing_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 1 | 1 | 0.02774900197982788 | 1.0533332299728817 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.07578038507037693 | 1.0459694909969366 | 8 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.06553273068534003 | 1.0473440391487672 | 9 |
| K-RT5-DelayedBasisConstraint | 2 | 2 | 0.018271889951494005 | 1.031561781810233 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | K-RT1 | 0.065979 | 0.968803 | -1.047808 | -0.077472 | -0.015550 | 1 |
| Fashion-MNIST | 0 | K-RT3 | 0.284883 | 0.927610 | -0.864432 | -0.284883 | -0.044932 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.202265 | 0.934653 | -1.107346 | -0.202265 | -0.013246 | 1 |
| KMNIST | 0 | K-RT2 | 0.137554 | 0.976507 | -2.002703 | -0.137554 | -0.016042 | 1 |
| KMNIST | 1 | K-RT5 | 0.026668 | 0.968709 | -0.422087 | -0.111281 | -0.018716 | 1 |

failure counts：

```text
AUCtime = 24
CEp99_tail = 22
source = 10
NLL_tail = 3
ECE_tail = 2
LineC = 1
```

判断：

```text
1. lr=0.003 让 mean source 转正，但 unique dataset-seed pass 从 6/9 退到 5/9。
2. AUCtime 与 CEp99 tail 仍是主要 blocker。
3. 单纯降低 lr 不能打开 S5。
```

## 15. 最终判断更新

v14.4 仍未达成 S5-OfficialFunctionalSuccess；最终 best 合法 route 仍为：

```text
best route = S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
best real_short_run_pass_rows = 9
official_s5_reached = 0
promotion_allowed = 0
```

当前已尝试：

```text
K-RT1 split agreement、
K-RT2 train-stream tail trust、
K-RT3 projection value retention、
K-RT4 source-tail co-state、
K-RT5 delayed basis constraint、
low plasticity / lower lambda、
stronger train-stream tail trust、
moderate split/delayed、
slow FMS refresh、
train-stream entropy output geometry、
ultralow FMS plasticity、
low-lr repair。
```

当前边界：

```text
我现在仍不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不把 CEp99/NLL/ECE/LineC audit metric 用作 direction、
不做 dataset-name branch、
不使用 seed-specific scaling、
不把 6/9 exploration positive 写成 S5 official success。
```

## 16. 用户再次追问后的 single-refresh 修复探针

本次没有新增代码修改；继续在 v14.4 runner 内按 real-transfer blocker 做一个受限修复探针：

```text
repair_v144_single_refresh200_lambda05
lr = 0.005
fms_strength = 0.05
fms_update_interval = 200
rt_lambda_max = 0.5
methods = K0,K-RT1,K-RT2,K-RT3,K-RT5,KCTRL
```

合法性说明：

```text
1. 全 3x3 统一运行，不做 dataset-name branch。
2. 不使用 validation/test/future/query batch 生成 direction。
3. 不使用 LineC / CEp99 / NLL / ECE 生成 direction。
4. 只把 LineC / tail / AUC / source 作为 audit/gate。
5. 不把 S4b 写成 S5，不允许 promotion。
```

结果：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 9
mean_source_vs_best_control_noncontrol = -0.03191156850920783
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.08869858582814534 | 1.0727013843627924 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.02159310711754693 | 1.0695566280761937 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.02195682790544298 | 1.050712801731164 | 9 |
| K-RT5-DelayedBasisConstraint | 4 | 4 | -0.08249762323167589 | 1.0954464050942987 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-RT1 | 0.108119 | 0.991314 | -1.533506 | -0.219537 | -0.010946 | 1 |
| MNIST | 1 | K-RT1 | 0.223926 | 0.883772 | -6.946172 | -0.237098 | -0.017712 | 1 |
| MNIST | 1 | K-RT3 | 0.226802 | 0.898331 | -4.681934 | -0.239974 | -0.019997 | 1 |
| MNIST | 1 | K-RT5 | 0.182495 | 0.976379 | -5.419912 | -0.195667 | -0.028670 | 1 |
| MNIST | 2 | K-RT1 | 0.230200 | 0.943766 | -0.952721 | -0.297060 | -0.040520 | 1 |
| MNIST | 2 | K-RT5 | 0.057710 | 0.918460 | -0.084510 | -0.124570 | -0.018149 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.027604 | 0.908469 | -2.434698 | -0.047252 | 0.000261 | 1 |
| KMNIST | 0 | K-RT2 | 0.051920 | 0.989902 | -6.277149 | -0.051920 | -0.002072 | 1 |
| KMNIST | 1 | K-RT5 | 0.027316 | 0.965192 | -0.075369 | -0.518898 | -0.045067 | 1 |

failure counts：

```text
AUCtime = 22
CEp99_tail = 18
source = 14
NLL_tail = 9
ECE_tail = 4
LineC = 0
```

判断：

```text
1. single-refresh200 与 best lowplasticity / slowrefresh160 一样停在 6/9。
2. LineC 不再是本 run 的 blocker，但 AUCtime / CEp99 / source 仍阻断 S5。
3. single-refresh 降低 direction churn 的假设没有带来 9/9 coverage。
4. 因此本 run 不能写成 S5，也不能 promotion。
```

## 17. 当前最终判断

v14.4 仍未达成 S5-OfficialFunctionalSuccess；当前 best 合法状态：

```text
best route = S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
best real_short_run_pass_rows = 9
official_s5_reached = 0
promotion_allowed = 0
```

已覆盖的 v14.4 修复方向：

```text
1. K-RT1 split agreement。
2. K-RT2 train-stream tail trust。
3. K-RT3 projection value retention。
4. K-RT4 source-tail co-state。
5. K-RT5 delayed basis constraint。
6. low plasticity / lower lambda。
7. stronger tail trust。
8. moderate split/delayed gate。
9. slow FMS refresh。
10. train-stream entropy output geometry。
11. ultralow FMS plasticity。
12. low-lr repair。
13. single-refresh repair。
```

停止边界：

```text
我现在不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不把 CEp99/NLL/ECE/LineC audit metric 用作 direction、
不做 dataset-name branch、
不使用 seed-specific scaling、
不把 6/9 exploration positive 写成 S5 official success。
```

## 18. 用户再次追问后的 lowplasticity + K8/KRT4 覆盖修复

本次没有新增代码修改；继续按计划中的 real-transfer boundary 做一个未覆盖组合：

```text
repair_v144_lowplasticity_plus_k8_krt4_lambda05
lr = 0.005
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.5
methods = K0,K8,K-RT1,K-RT2,K-RT3,K-RT4,K-RT5,KCTRL
```

触发原因：

```text
1. best lowplasticity / slowrefresh / single-refresh 都为 6/9。
2. K8 与 K-RT4 已在 official_v144 中覆盖，但没有在 best lowplasticity 配置下与 K-RT1/2/3/5 同 run 覆盖。
3. 本次检查它们是否能补上缺失的 Fashion-MNIST seed0/2 或 KMNIST seed2。
```

合法性说明：

```text
1. 全 3x3 统一运行，不做 dataset-name branch。
2. 不使用 seed-specific scaling。
3. 不使用 validation/test/future/query batch 生成 direction。
4. 不使用 LineC / CEp99 / NLL / ECE / AUC 生成 direction。
5. 不拼接不同 run 的局部 pass。
```

结果：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 10
mean_source_vs_best_control_noncontrol = -0.06702906445220665
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint | 1 | 1 | -0.1262306637234158 | 0.9864509986186465 | 8 |
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.10150694184833103 | 1.0673764713627447 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.02155327796936035 | 1.0695467503638163 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.021980126698811848 | 1.050731870366617 | 9 |
| K-RT4-SourceTailCoState | 0 | 0 | -0.13548827171325684 | 1.0664973035841765 | 9 |
| K-RT5-DelayedBasisConstraint | 4 | 4 | -0.08248191409640843 | 1.095324834920501 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-RT1 | 0.113161 | 0.989009 | -1.502634 | -0.224621 | -0.014783 | 1 |
| MNIST | 1 | K8 | 0.087914 | 0.975628 | -5.826309 | -0.101176 | 0.002058 | 1 |
| MNIST | 1 | K-RT1 | 0.225544 | 0.883245 | -6.995262 | -0.238807 | -0.021089 | 1 |
| MNIST | 1 | K-RT3 | 0.226790 | 0.898311 | -4.681700 | -0.240052 | -0.020003 | 1 |
| MNIST | 1 | K-RT5 | 0.182434 | 0.976370 | -5.420082 | -0.195696 | -0.028673 | 1 |
| MNIST | 2 | K-RT1 | 0.231780 | 0.943324 | -1.094994 | -0.298710 | -0.042674 | 1 |
| MNIST | 2 | K-RT5 | 0.057659 | 0.918473 | -0.084893 | -0.124589 | -0.018150 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.027344 | 0.908514 | -2.434628 | -0.047216 | 0.000262 | 1 |
| KMNIST | 0 | K-RT2 | 0.051924 | 0.989902 | -6.277489 | -0.051924 | -0.002071 | 1 |
| KMNIST | 1 | K-RT5 | 0.027283 | 0.965196 | -0.074760 | -0.518914 | -0.045063 | 1 |

remaining best row by dataset-seed：

| dataset | seed | best method | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | K-RT3 | 0 | 0.184902 | 0.873954 | 0.166046 | -0.473994 | -0.046205 | 1 |
| Fashion-MNIST | 2 | K-RT2 | 0 | 0.162050 | 1.232769 | -0.704851 | -0.162050 | -0.007385 | 1 |
| KMNIST | 2 | K-RT1 | 0 | 0.094257 | 1.110939 | 2.585526 | -0.171631 | -0.016695 | 1 |

failure counts：

```text
source = 27
AUCtime = 34
CEp99_tail = 26
NLL_tail = 16
ECE_tail = 7
LineC = 1
```

判断：

```text
1. 加回 K8 与 K-RT4 后 pass rows 增至 10，
   但 unique dataset-seed pass 仍停在 6/9。
2. K8 只增加 MNIST seed1 的额外 pass，没有补上缺失 dataset-seed。
3. K-RT4 仍为 0 pass。
4. 剩余缺口仍是 Fashion-MNIST seed0 的 CEp99 tail、
   Fashion-MNIST seed2 的 AUCtime、
   KMNIST seed2 的 AUCtime + CEp99 tail。
5. 因此仍不是 S5，promotion_allowed 仍为 0。
```

## 19. 当前最终判断更新

v14.4 仍未达成 S5-OfficialFunctionalSuccess；当前 best 合法状态没有变化：

```text
best route = S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
best real_short_run_pass_rows = 10
official_s5_reached = 0
promotion_allowed = 0
```

已覆盖的 v14.4 修复方向更新为：

```text
1. K-RT1 split agreement。
2. K-RT2 train-stream tail trust。
3. K-RT3 projection value retention。
4. K-RT4 source-tail co-state。
5. K-RT5 delayed basis constraint。
6. K8 baseline in lowplasticity setting。
7. low plasticity / lower lambda。
8. stronger tail trust。
9. moderate split/delayed gate。
10. slow FMS refresh。
11. train-stream entropy output geometry。
12. ultralow FMS plasticity。
13. low-lr repair。
14. single-refresh repair。
15. lowplasticity + K8/KRT4 same-run coverage。
```

停止边界更新：

```text
我现在仍不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不把 CEp99/NLL/ECE/LineC/AUC audit metric 用作 direction、
不做 dataset-name branch、
不使用 seed-specific scaling、
不拼接不同 run 的局部 positive、
不把 6/9 exploration positive 写成 S5 official success。
```

## 20. 用户再次追问后的 K-RT6 ProjectionTailTrust 修复

用户再次要求未达成则继续。本次按计划 failure-response 继续尝试组合修复：

```text
1. 若 source 改善但 tail / AUC fail，尝试 train-stream tail trust / risk state。
2. 若 tail 改善但 source fail，尝试 projection value retention / split agreement。
```

因此新增：

```text
K-RT6-ProjectionTailTrust =
  K-RT3 ProjectionValueRetention
  +
  K-RT2 TrainStreamTailTrust
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-RT6-ProjectionTailTrust。
  rt_method_base 将 K-RT6 映射到 K8 Rational base。
  adaptive_projection_scales 将 K-RT6 走 K-RT3 value-retention projection。
  lambda_from_method 将 K-RT6 走 K-RT2 train-stream risk lambda。
```

合法性：

```text
1. K-RT6 direction 只使用 train-stream loss_q95 / margin_p10 / logit_rms、
   FMS value vector 与 projection retention。
2. 不使用 validation/test/future/query。
3. 不使用 CEp99 / NLL / ECE / LineC / AUCtime 生成方向。
4. 本轮仍是 compute-budgeted repair；promotion_allowed 必须保持 0。
```

语法检查：

```text
py_compile pass
```

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K-RT1,K-RT2,K-RT3,K-RT5,K-RT6,KCTRL
train_steps = 200
batch_size = 32
lr = 0.005
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.5
linec_mode = exact
compute_budgeted_run = 1
```

route：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 11
mean_source_vs_best_control_noncontrol = -0.04666680892308553
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.10150694184833103 | 1.0673764713627447 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.02155327796936035 | 1.0695467503638163 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.021980126698811848 | 1.050731870366617 | 9 |
| K-RT5-DelayedBasisConstraint | 4 | 4 | -0.08248191409640843 | 1.095324834920501 | 9 |
| K-RT6-ProjectionTailTrust | 2 | 2 | -0.09287859333886041 | 1.083393981663408 | 9 |
| K0-RAT-AdamW | 0 | 0 | -0.1188468403286404 | 1.0240534785437074 | 9 |
| KCTRL-RandomMatchedProjection | 0 | 0 | -0.04685921139187283 | 1.0 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-RT1 | 0.113161 | 0.989009 | -1.502634 | -0.224621 | -0.014783 | 1 |
| MNIST | 1 | K-RT1 | 0.225544 | 0.883245 | -6.995262 | -0.238807 | -0.021089 | 1 |
| MNIST | 1 | K-RT3 | 0.226790 | 0.898311 | -4.681700 | -0.240052 | -0.020003 | 1 |
| MNIST | 1 | K-RT5 | 0.182434 | 0.976370 | -5.420082 | -0.195696 | -0.028673 | 1 |
| MNIST | 1 | K-RT6 | 0.223744 | 0.908376 | -2.440582 | -0.237007 | -0.025605 | 1 |
| MNIST | 2 | K-RT1 | 0.231780 | 0.943324 | -1.094994 | -0.298710 | -0.042674 | 1 |
| MNIST | 2 | K-RT5 | 0.057659 | 0.918473 | -0.084893 | -0.124589 | -0.018150 | 1 |
| MNIST | 2 | K-RT6 | 0.058370 | 0.932749 | -0.313580 | -0.125301 | -0.024170 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.027344 | 0.908514 | -2.434628 | -0.047216 | 0.000262 | 1 |
| KMNIST | 0 | K-RT2 | 0.051924 | 0.989902 | -6.277489 | -0.051924 | -0.002071 | 1 |
| KMNIST | 1 | K-RT5 | 0.027283 | 0.965196 | -0.074760 | -0.518914 | -0.045063 | 1 |

remaining best row by dataset-seed：

| dataset | seed | best method | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | K-RT3 | 0 | 0.184902 | 0.873954 | 0.166046 | -0.473994 | -0.046205 | 1 |
| Fashion-MNIST | 2 | K-RT2 | 0 | 0.162050 | 1.232769 | -0.704851 | -0.162050 | -0.007385 | 1 |
| KMNIST | 2 | K-RT1 | 0 | 0.094257 | 1.110939 | 2.585526 | -0.171631 | -0.016695 | 1 |

K-RT6 row-level：

| dataset | seed | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MNIST | 0 | 0 | -0.123465 | 1.104864 | -2.497648 | 0.012005 | -0.002448 | 1 | source,AUCtime |
| MNIST | 1 | 1 | 0.223744 | 0.908376 | -2.440582 | -0.237007 | -0.025605 | 1 | - |
| MNIST | 2 | 1 | 0.058370 | 0.932749 | -0.313580 | -0.125301 | -0.024170 | 1 | - |
| Fashion-MNIST | 0 | 0 | -0.167308 | 1.071968 | -0.327671 | -0.121784 | -0.032406 | 1 | source,AUCtime |
| Fashion-MNIST | 1 | 0 | -0.194636 | 1.102639 | 3.312584 | 0.174764 | 0.014478 | 1 | source,AUCtime,CEp99,NLL |
| Fashion-MNIST | 2 | 0 | -0.203331 | 1.111799 | 0.893652 | 0.203331 | 0.017452 | 1 | source,AUCtime,CEp99,NLL |
| KMNIST | 0 | 0 | 0.055100 | 1.025365 | -3.250263 | -0.055100 | 0.018707 | 1 | AUCtime |
| KMNIST | 1 | 0 | -0.342099 | 1.083394 | 0.090408 | -0.149531 | -0.030250 | 1 | source,AUCtime,CEp99 |
| KMNIST | 2 | 0 | -0.142282 | 1.243287 | 2.074610 | 0.064909 | 0.022072 | 1 | source,AUCtime,CEp99,NLL,ECE |

failure counts：

```text
AUCtime = 29
CEp99_tail = 22
source = 20
NLL_tail = 12
ECE_tail = 5
LineC = 0
```

判断：

```text
1. K-RT6 使 pass rows 从 10 增至 11，但 unique dataset-seed 仍为 6/9。
2. K-RT6 自身只通过 MNIST seed1/seed2。
3. K-RT6 没有补上缺失的 Fashion-MNIST seed0/2 或 KMNIST seed2。
4. LineC failure 在本轮为 0，说明当前主 blocker 不是 LineC。
5. 剩余主 blocker 是 AUCtime、CEp99 tail 与 source instability 的共同失败。
6. 因此仍不能写成 S5，也不能 promotion。
```

新增产物：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_krt6_projection_tailtrust_lambda05/
```

## 21. 当前最终判断更新

v14.4 仍未达成 S5-OfficialFunctionalSuccess；当前 best 合法状态更新为：

```text
best route = S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
latest real_short_run_pass_rows = 11
official_s5_reached = 0
promotion_allowed = 0
```

已覆盖的 v14.4 修复方向更新为：

```text
1. K-RT1 split agreement。
2. K-RT2 train-stream tail trust。
3. K-RT3 projection value retention。
4. K-RT4 source-tail co-state。
5. K-RT5 delayed basis constraint。
6. K-RT6 projection + tail trust。
7. K8 baseline in lowplasticity setting。
8. low plasticity / lower lambda。
9. stronger tail trust。
10. moderate split/delayed gate。
11. slow FMS refresh。
12. train-stream entropy output geometry。
13. ultralow FMS plasticity。
14. low-lr repair。
15. single-refresh repair。
16. lowplasticity + K8/KRT4 same-run coverage。
```

停止边界更新：

```text
我现在仍不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不把 CEp99/NLL/ECE/LineC/AUC audit metric 用作 direction、
不做 dataset-name branch、
不使用 seed-specific scaling、
不拼接不同 run 的局部 positive、
不把 6/9 exploration positive 写成 S5 official success。

已尝试的合法 train-stream-only repair 没有补上
Fashion-MNIST seed0/2 与 KMNIST seed2。
下一步需要新的 real-transfer mechanism 计划，
尤其是同时降低 AUCtime overhead、CEp99 tail 与 source instability 的统一机制。
```

## 22. 用户再次追问后的 K-RT7 LateProjectionTailTrust 修复

用户再次要求未达成则继续。本次没有降低 gate，也没有用 audit metric 做方向，而是尝试一个 phase-only 的 late plasticity 机制：

```text
K-RT7-LateProjectionTailTrust =
  late FMS plasticity ramp
  +
  K-RT6 projection value retention
  +
  train-stream tail/risk lambda
```

核心假设：

```text
1. 当前很多失败是 AUC_NLL ratio > 1.0。
2. 如果前半程尽量保持 AdamW 学习曲线，后半程再注入 FMS，
   可能降低 early AUC damage，同时保留 final source。
3. phase ramp 只使用 train step phase；
   risk lambda 只使用 train-stream loss_q95 / margin_p10 / logit_rms。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-RT7-LateProjectionTailTrust。
  新增 train_step_phase 到 train-stream proxy。
  K-RT7 使用 K-RT3/K-RT6 的 projection value retention。
  K-RT7 的 lambda = late_phase_ramp * train_stream_risk_trust。
```

合法性：

```text
1. 不使用 validation/test/future/query。
2. 不使用 CEp99 / NLL / ECE / LineC / AUCtime 生成方向。
3. 不做 dataset-name branch。
4. 不使用 seed-specific scaling。
5. compute_budgeted_run = 1；promotion_allowed = 0。
```

语法检查：

```text
py_compile pass
```

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K-RT1,K-RT2,K-RT3,K-RT5,K-RT7,KCTRL
train_steps = 200
batch_size = 32
lr = 0.005
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.5
linec_mode = exact
compute_budgeted_run = 1
```

route：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 10
mean_source_vs_best_control_noncontrol = -0.043665132257673475
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.10150694184833103 | 1.0673764713627447 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.02155327796936035 | 1.0695467503638163 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.021980126698811848 | 1.050731870366617 | 9 |
| K-RT5-DelayedBasisConstraint | 4 | 4 | -0.08248191409640843 | 1.095324834920501 | 9 |
| K-RT7-LateProjectionTailTrust | 1 | 1 | -0.07787021001180013 | 1.0802075331995027 | 9 |
| K0-RAT-AdamW | 0 | 0 | -0.1188468403286404 | 1.0240534785437074 | 9 |
| KCTRL-RandomMatchedProjection | 0 | 0 | -0.04685921139187283 | 1.0 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-RT1 | 0.113161 | 0.989009 | -1.502634 | -0.224621 | -0.014783 | 1 |
| MNIST | 1 | K-RT1 | 0.225544 | 0.883245 | -6.995262 | -0.238807 | -0.021089 | 1 |
| MNIST | 1 | K-RT3 | 0.226790 | 0.898311 | -4.681700 | -0.240052 | -0.020003 | 1 |
| MNIST | 1 | K-RT5 | 0.182434 | 0.976370 | -5.420082 | -0.195696 | -0.028673 | 1 |
| MNIST | 1 | K-RT7 | 0.075135 | 0.992399 | -2.819923 | -0.088398 | -0.010688 | 1 |
| MNIST | 2 | K-RT1 | 0.231780 | 0.943324 | -1.094994 | -0.298710 | -0.042674 | 1 |
| MNIST | 2 | K-RT5 | 0.057659 | 0.918473 | -0.084893 | -0.124589 | -0.018150 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.027344 | 0.908514 | -2.434628 | -0.047216 | 0.000262 | 1 |
| KMNIST | 0 | K-RT2 | 0.051924 | 0.989902 | -6.277489 | -0.051924 | -0.002071 | 1 |
| KMNIST | 1 | K-RT5 | 0.027283 | 0.965196 | -0.074760 | -0.518914 | -0.045063 | 1 |

remaining best row by dataset-seed：

| dataset | seed | best method | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | K-RT3 | 0 | 0.184902 | 0.873954 | 0.166046 | -0.473994 | -0.046205 | 1 |
| Fashion-MNIST | 2 | K-RT7 | 0 | 0.233003 | 1.228868 | -0.889408 | -0.233003 | -0.032549 | 1 |
| KMNIST | 2 | K-RT1 | 0 | 0.094257 | 1.110939 | 2.585526 | -0.171631 | -0.016695 | 1 |

K-RT7 row-level：

| dataset | seed | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MNIST | 0 | 0 | 0.008138 | 1.080208 | -3.253248 | -0.119598 | 0.003864 | 1 | AUCtime |
| MNIST | 1 | 1 | 0.075135 | 0.992399 | -2.819923 | -0.088398 | -0.010688 | 1 | - |
| MNIST | 2 | 0 | 0.042569 | 0.939440 | 0.454446 | -0.109499 | -0.029132 | 1 | CEp99 |
| Fashion-MNIST | 0 | 0 | -0.016016 | 1.125437 | 0.857180 | -0.273077 | -0.048155 | 1 | source,AUCtime,CEp99 |
| Fashion-MNIST | 1 | 0 | -0.682969 | 1.141081 | 2.715660 | 0.663097 | 0.055254 | 1 | source,AUCtime,CEp99,NLL,ECE |
| Fashion-MNIST | 2 | 0 | 0.233003 | 1.228868 | -0.889408 | -0.233003 | -0.032549 | 1 | AUCtime |
| KMNIST | 0 | 0 | -0.325062 | 1.057435 | -3.872623 | 0.325062 | 0.030242 | 1 | source,AUCtime,NLL,ECE |
| KMNIST | 1 | 0 | 0.070650 | 1.043607 | -2.214352 | -0.562281 | -0.055730 | 1 | AUCtime |
| KMNIST | 2 | 0 | -0.106281 | 1.104521 | 0.881664 | 0.028907 | -0.003424 | 1 | source,AUCtime,CEp99,NLL |

failure counts：

```text
AUCtime = 29
CEp99_tail = 22
source = 18
NLL_tail = 12
ECE_tail = 6
LineC = 0
```

判断：

```text
1. K-RT7 没有打开 S5，unique dataset-seed 仍为 6/9。
2. K-RT7 自身只通过 MNIST seed1。
3. Fashion-MNIST seed2 的 source/tail 在 K-RT7 下更强，
   但 AUCtime = 1.228868，仍不能过 gate。
4. Fashion-MNIST seed0 仍卡在 CEp99 tail。
5. KMNIST seed2 仍卡在 AUCtime + CEp99/source。
6. late ramp 没有解决 AUCtime blocker。
7. 不能写成 S5，也不能 promotion。
```

新增产物：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_krt7_late_projection_tailtrust_lambda05/
```

## 23. 当前最终判断更新

v14.4 仍未达成 S5-OfficialFunctionalSuccess；当前 best 合法状态仍是：

```text
best route = S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
best real_short_run_pass_rows = 11
latest K-RT7 real_short_run_pass_rows = 10
official_s5_reached = 0
promotion_allowed = 0
```

已覆盖的 v14.4 修复方向更新为：

```text
1. K-RT1 split agreement。
2. K-RT2 train-stream tail trust。
3. K-RT3 projection value retention。
4. K-RT4 source-tail co-state。
5. K-RT5 delayed basis constraint。
6. K-RT6 projection + tail trust。
7. K-RT7 late projection + tail trust。
8. K8 baseline in lowplasticity setting。
9. low plasticity / lower lambda。
10. stronger tail trust。
11. moderate split/delayed gate。
12. slow FMS refresh。
13. train-stream entropy output geometry。
14. ultralow FMS plasticity。
15. low-lr repair。
16. single-refresh repair。
17. lowplasticity + K8/KRT4 same-run coverage。
```

停止边界更新：

```text
我现在不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5。

已尝试的合法 train-stream-only repair 包括：
split agreement、tail trust、projection retention、source-tail co-state、
delayed basis、late FMS ramp、low/ultralow plasticity、slow/single refresh、
train-stream entropy output geometry、K8/KRT4 coverage、K-RT6/K-RT7 组合。

它们没有补上 Fashion-MNIST seed0/2 与 KMNIST seed2。
继续推进需要新的 real-transfer mechanism 计划，
尤其是能同时降低 AUC_NLL、CEp99 tail 与 source instability 的统一机制。
不能把 audit metric 作为 direction，不能 dataset-name branch，
不能 seed-specific scaling，也不能把 6/9 S4b 写成 S5。
```
