# DG-KAN v14.7 OptimizerStateTransportFMS AllBasisParallel 实验结果复盘

生成时间：2026-05-29（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不把 v14.6.1 diagnostic positive 写成 v14.7 official promotion。

## 1. 计划理解

v14.7 的目标是把 v14.6.1 中的 P1 optimizer-state transport 机制做成 official confirmation：

```text
main candidate = K2-RAT-FMS-ZeroMomentReset-AffectedOnly
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
official gate = 9/9 strict pass
```

硬约束：

```text
1. 不新增 K-RT/K-AUC/K-FL action token。
2. 不使用 action bank 或 controller。
3. 不做 strength/lambda/lr/refresh grid search。
4. 不做 dataset-name branch 或 seed-specific scaling。
5. validation/test/future/query 不可生成 direction。
6. LineC/CEp99/NLL/ECE/AUCtime/Brier 只作为 audit/gate。
7. random/full reset controls 必须比较；若 controls 也能复现，不允许 KAN-specific promotion。
```

official strict gate：

```text
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
required artifacts complete
forbidden/no-action-search audit pass
```

## 2. 本轮代码修改

新增：

```text
experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py
```

修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

主要实现：

```text
1. v14.7 official runner 与 required artifacts。
2. K methods:
   K0-RAT-AdamW
   K1-RAT-FMS-NoStateTransport
   K2-RAT-FMS-ZeroMomentReset-AffectedOnly
   K3-RAT-FMS-ZeroMomentReset-AllFMSRoles
   K4-RAT-FMS-PartialMomentInterpolation
   K5-RAT-FMS-RMSRecomputeMicrobatch
   K6-RAT-FMS-MomentTransportProjectedGrad
   KCTRL-RandomMomentResetMatchedFraction
   KCTRL-ZeroMomentResetRandomCoords
   KCTRL-FullAdamWStateReset
   KCTRL-NoOpMatchedOverhead
3. MLP analog controls:
   M0-MLP-AdamW
   M1-MLP-FMS-NoStateTransport
   M2-MLP-FMS-ZeroMomentResetAffected
   M3-MLP-FMS-ZeroMomentResetRandomCoords
   M4-MLP-FMS-FullAdamWStateReset
4. optimizer-state transport scope:
   affected / all_fms_roles / random_matched /
   random_affected_fraction / full_adamw
5. strict gate、endpoint-vs-AdamW gate、specificity gate、efficiency gate。
6. route 复算模式：
   --reuse-existing-results 1
```

后续修复：

```text
1. 修复 load_real_split 调用签名。
2. 修复 write_svg 调用签名。
3. 修复 v14.7 关键实现 blocker：
   optimizer-state transport 从 every-step reset 改为只在 FMS refresh/event step 执行。
```

合法性说明：

```text
1. 没有新增 action token。
2. 没有启动 controller。
3. 没有调 strength/lambda/lr/refresh 网格。
4. 没有使用 audit metric 生成 direction。
5. promotion_allowed 始终为 0。
```

## 3. Smoke

默认 Python smoke blocker：

```text
ModuleNotFoundError: No module named 'torch'
```

修复：

```text
使用 conda run -n kan python。
```

runner glue blockers：

```text
1. load_real_split() takes 4 positional arguments but 8 were given
2. write_svg() missing 1 required positional argument: 'lines'
```

修复后 smoke：

```text
out_dir = results/v14_7_optimizer_state_transport_fms_all_basis_parallel/smoke_v147
route = R1-ZeroMomentResetNotReplicated
expected_dataset_seed_count = 1
k2_dataset_seed_pass_count = 0 / 1
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

解释：

```text
smoke 只证明 runner 和 artifact surface 可执行；
不作为 official 成功或失败定论。
```

## 4. Official v14.7 首跑

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K1,K2,K3,K4,K5,K6,KCTRL-Random,KCTRL-RandomCoords,KCTRL-FullReset,KCTRL-NoOp
mlp_methods = M0,M1,M2,M3,M4
train_steps = 200
batch_size = 32
lr = 0.005
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.5
linec_mode = exact
compute_budgeted_run = 1
```

首跑 route 复算后：

```text
route = R2-ResetWorksButNotFMSSpecific
minimum_success = S4c-MechanismSufficientDiagnostic
expected_dataset_seed_count = 9
k2_dataset_seed_pass_count = 0 / 9
k2_endpoint_vs_adamw_pass_count = 9 / 9
k2_specificity_pass_count = 0 / 9
k2_efficiency_pass_count = 1 / 9
control_endpoint_full_success_count = 3
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

K2 summary：

| method | strict pass | endpoint-vs-AdamW pass | specificity pass | efficiency pass | mean source vs best control | median AUC vs best control | median AUC vs AdamW | median step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K2 affected zero reset | 0/9 | 9/9 | 0/9 | 1/9 | -0.007801 | 1.006418 | 0.589830 | 1.496919 |

Controls：

| control | endpoint-vs-AdamW pass | median AUC vs AdamW | median step |
|---|---:|---:|---:|
| KCTRL-RandomMomentResetMatchedFraction | 9/9 | 0.579976 | 1.565960 |
| KCTRL-ZeroMomentResetRandomCoords | 9/9 | 0.579976 | 1.555645 |
| KCTRL-FullAdamWStateReset | 9/9 | 0.589830 | 1.509587 |

判断：

```text
1. K2 确实复制了 v14.6.1 的 endpoint-vs-AdamW 改善。
2. 但 random/full reset controls 也能复制 endpoint 改善。
3. K2 source_vs_best_control 没有优势，specificity pass = 0/9。
4. K2 step_time 也未过 strict gate，仅 1/9 efficiency pass。
5. 因此不能写成 S5，也不能 promotion。
```

实现层 blocker：

```text
首跑的 optimizer-state transport 是 every-step reset。
这不符合 v14.7 计划中“FMS event 后 optimizer-state transport”的语义，
并会把 K2 退化成 generic reset trick。
```

## 5. Repair：event-only optimizer-state transport

修复内容：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  optimizer-state transport 只在 refresh/FMS event step 执行。
```

执行规模：

```text
同 official_v147。
out_dir = results/v14_7_optimizer_state_transport_fms_all_basis_parallel/repair_v147_event_only_transport
```

结果：

```text
route = R1-ZeroMomentResetNotReplicated
minimum_success = S4c-MechanismSufficientDiagnostic
expected_dataset_seed_count = 9
k2_dataset_seed_pass_count = 0 / 9
k2_endpoint_vs_adamw_pass_count = 4 / 9
k2_specificity_pass_count = 0 / 9
k2_efficiency_pass_count = 2 / 9
control_endpoint_full_success_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

method summary：

| method | strict pass | endpoint-vs-AdamW pass | specificity pass | efficiency pass | mean source vs best control | median AUC vs best control | median AUC vs AdamW | median step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K2 affected zero reset | 0/9 | 4/9 | 0/9 | 2/9 | -0.059711 | 1.000002 | 0.962900 | 1.303882 |
| K3 all-FMS-role zero reset | 0/9 | 4/9 | 0/9 | 2/9 | -0.059709 | 1.000000 | 0.962898 | 1.293518 |
| K5 RMS recompute | 0/9 | 0/9 | 0/9 | 2/9 | -551467.626172 | 2841.746462 | 2648.723201 | 1.307796 |
| K6 projected moment transport | 0/9 | 0/9 | 0/9 | 2/9 | -0.946114 | 1.442027 | 1.362555 | 1.305575 |
| KCTRL random matched reset | 0/9 | 4/9 | 0/9 | 2/9 | -0.115331 | 1.031285 | 0.994311 | 1.302785 |
| KCTRL full reset | 0/9 | 4/9 | 0/9 | 2/9 | -0.059709 | 1.000000 | 0.962898 | 1.298604 |
| M2 MLP zero reset affected | 0/9 | 3/9 | 0/9 | 0/9 | -0.048693 | 1.018274 | 0.989475 | 1.266652 |

K2 row-level blocker：

```text
MNIST seed0: endpoint pass, strict fails source_vs_best_control.
MNIST seed1: endpoint pass, strict fails source + step_time.
MNIST seed2: endpoint pass, strict fails source.
Fashion-MNIST seed0: fails source + AUC + CEp99 + step.
Fashion-MNIST seed1: fails source + AUC + CEp99 + NLL + step.
Fashion-MNIST seed2: fails source + AUC + step.
KMNIST seed0: endpoint pass, strict fails source + AUC + step.
KMNIST seed1: fails source + CEp99 + step.
KMNIST seed2: fails source + AUC + CEp99 + step.
```

affected-coordinate audit：

```text
official_v147 continuous reset:
  K2 reset fraction median = 0.994765
repair_v147_event_only_transport:
  K2 reset fraction median = 0.994661
```

判断：

```text
1. event-only transport 更符合计划语义，但不能复现 v14.6.1 的 9/9 endpoint gain。
2. K2 与 K3 结果几乎相同，说明当前 affected coordinate identification 过宽；
   affected-only 近似 all-FMS-role reset。
3. K5/RMS 和 K6/projected moment transport 没有超过 K2。
4. reset controls 没有 strict full success，但 K2 也没有 strict success。
5. step_time gate 仍是 blocker：K2 median step = 1.303882 > 1.25。
6. 因此 v14.7 没有达成 S5。
```

## 6. Required artifacts

主要输出目录：

```text
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/smoke_v147/
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/official_v147/
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/repair_v147_event_only_transport/
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/smoke_v147_event_only/
```

每个完整 run 输出：

```text
v147_route_decision.json
v147_real_results.csv
v147_real_summary.csv
v147_optimizer_state_transport_manifest.csv
v147_fms_affected_coordinate_manifest.csv
v147_state_transport_controls.csv
v147_mlp_state_transport_control.csv
v147_all_basis_substrate_status.csv
v147_trajectory_autopsy.csv
v147_forbidden_information_audit.csv
v147_no_action_search_audit.csv
v147_required_artifact_manifest.csv
v147_code_review_packet.zip
fig_v147_auc_debt_before_after.svg
fig_v147_state_transport_gate_matrix.svg
fig_v147_controls_comparison.svg
fig_v147_all_basis_status.svg
```

artifact completeness：

```text
official_v147 required_artifact_missing_count = 0
repair_v147_event_only_transport required_artifact_missing_count = 0
```

## 7. All-basis status

v14.7 runner 写入：

```text
v147_all_basis_substrate_status.csv
```

结果：

```text
D-RAT-Monitor-v147 new_training_executed = 1
D-RAT strict_gate_pass_rows = 0
Non-RAT candidates new_training_executed = 0
```

说明：

```text
Rational official confirmation 未打开；
本轮没有把未执行的 D-WAV/D-FOU/D-RBF/D-CHE candidates 写成成功。
```

## 8. 最终科学结论

v14.7 没有达成 S5-OfficialFunctionalSuccess。

最终 best 合法状态：

```text
best official-style route = R2-ResetWorksButNotFMSSpecific
best plan-faithful repair route = R1-ZeroMomentResetNotReplicated
official_s5_reached = 0
promotion_allowed = 0
```

已闭合事实：

```text
1. every-step zero moment reset 可把 K2 endpoint-vs-AdamW 推到 9/9，
   但 random/full reset controls 也能做到，因此不是 FMS-specific official proof。

2. event-only optimizer-state transport 是更符合 v14.7 计划语义的实现，
   但 K2 endpoint-vs-AdamW 只有 4/9，strict 仍为 0/9。

3. 当前 affected-coordinate mask 过宽：
   K2 affected reset fraction median 约 0.995，
   affected-only 与 all-FMS-role reset 几乎等价。

4. step_time gate 未过：
   event-only K2 median step_time_ratio = 1.303882 > 1.25。

5. K5 RMS recompute 明显退化，K6 projected moment transport 也没有打开 coverage。

6. no-action-search audit 与 forbidden information audit 均为 0 violation。
```

当前 no-go boundary：

```text
1. 不能把 v14.6.1 S4c diagnostic 或 v14.7 endpoint-vs-AdamW 9/9 写成 S5。
2. 不能忽略 random/full reset controls 的复现能力。
3. 不能降低 step_time gate。
4. 不能通过调 strength/lambda/lr/refresh 网格补救。
5. 不能新增 action token 或启动 controller。
```

下一步需要新计划：

```text
1. 先解决 affected-coordinate identification 过宽问题；
   需要一个预注册的 train-stream-only sparse affected mask，
   不能用 audit metric 反推。

2. 同时需要降低 FMS per-example refresh overhead，
   否则 step_time_ratio <= 1.25 仍无法打开。

3. 如果继续 optimizer-state transport，应验证 bounded post-event recovery window，
   但这已经超出 v14.7 当前方法组，需要单独预注册，不能在本轮临时写成 success。
```

## 9. 用户再次要求继续后的 sparse affected-coordinate 修复

用户再次要求“没有达成则继续”。本次没有新增 action token，也没有调
strength / lambda / lr / refresh 网格，而是针对 v14.7 已发现的实现问题做
一个固定的 train-stream-only sparse affected-coordinate 修复。

触发原因：

```text
repair_v147_event_only_transport 显示：
  K2 affected coordinate fraction median 约 0.995。

这说明原 affected-coordinate identification 过宽，
K2 affected-only reset 几乎等价于 K3 all-FMS-role reset。

v14.7 计划 6.4 要求 affected coordinate 来自
train-stream FMS active mask / projected direction；
不能来自 AUCtime / LineC / CEp99 / NLL / ECE audit metric。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 optimizer-state transport scope：
    delta_top25
    random_delta_top25

experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py
  新增参数：
    --affected-coordinate-policy delta_nonzero|delta_top25

  当 affected-coordinate-policy = delta_top25：
    K2 / M2 使用 projected-gradient delta top-25% affected mask；
    random controls 使用 random_delta_top25 matched fraction。
```

合法性说明：

```text
1. 不新增 K-token action。
2. 不启动 controller。
3. 不使用 validation / test / future / query。
4. 不使用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 生成 coordinate。
5. 不做 dataset-name branch 或 seed-specific scaling。
6. 不降低 S5 gate。
```

执行规模：

```text
out_dir = results/v14_7_optimizer_state_transport_fms_all_basis_parallel/repair_v147_event_only_delta_top25
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K1,K2,K3,K4,K5,K6,KCTRL-random,KCTRL-randomcoords,KCTRL-fullreset,KCTRL-noop
mlp_methods = M0,M1,M2,M3,M4
affected_coordinate_policy = delta_top25
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
route = R1-ZeroMomentResetNotReplicated
minimum_success = S4c-MechanismSufficientDiagnostic
k2_dataset_seed_pass_count = 0 / 9
k2_endpoint_vs_adamw_pass_count = 2 / 9
k2_specificity_pass_count = 0 / 9
k2_efficiency_pass_count = 1 / 9
control_endpoint_full_success_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

method summary：

| method | strict dataset-seed pass | endpoint-vs-AdamW pass rows | specificity pass rows | efficiency pass rows | mean source | median AUC vs best | median AUC vs AdamW | median step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K2 affected zero reset, delta_top25 | 0 | 2 | 0 | 1 | -0.155563 | 1.028774 | 0.971016 | 1.302402 |
| K3 all-FMS-role zero reset | 0 | 4 | 0 | 2 | -0.051499 | 1.022013 | 0.966762 | 1.297901 |
| K4 partial interpolation | 0 | 0 | 0 | 2 | -0.510376 | 1.098305 | 1.107529 | 1.298971 |
| K5 RMS recompute | 0 | 0 | 0 | 2 | -551467.617962 | 928617.661868 | 839575.961846 | 1.298526 |
| K6 projected-grad moment | 0 | 0 | 0 | 2 | -0.937904 | 1.353446 | 1.265292 | 1.299424 |
| KCTRL random matched | 0 | 0 | 0 | 1 | -0.169125 | 1.028301 | 0.984587 | 1.330723 |
| KCTRL full AdamW reset | 0 | 4 | 0 | 2 | -0.051499 | 1.022013 | 0.966762 | 1.297901 |

mask audit：

```text
K2 original affected_param_fraction:
  min = 0.7298994660377502
  median = 0.9947654604911804
  max = 1.0

K2 actual transport_reset_param_fraction after delta_top25:
  min = 0.18247486650943756
  median = 0.2487437129020691
  max = 0.25
```

判断：

```text
1. delta_top25 确实把实际 reset 参数比例降到约 25%。
2. 但 K2 endpoint-vs-AdamW 从 event-only 的 4/9 退到 2/9。
3. K3 / KCTRL full reset 均为 endpoint-vs-AdamW 4/9，仍没有 strict S5。
4. K5 RMS recompute 继续明显退化。
5. K6 projected-grad moment transport 没有打开 coverage。
6. 因此 sparse affected-coordinate 修复没有达成 v14.7 official S5。
```

## 10. 当前最终判断更新

v14.7 仍未达成 S5-OfficialFunctionalSuccess。

当前完整 route 对照：

| run | route | K2 strict | K2 endpoint-vs-AdamW | K2 specificity | K2 efficiency | controls endpoint full success | promotion |
|---|---|---:|---:|---:|---:|---:|---:|
| official_v147 | R2-ResetWorksButNotFMSSpecific | 0/9 | 9/9 | 0/9 | 1/9 | 3 | 0 |
| repair_v147_event_only_transport | R1-ZeroMomentResetNotReplicated | 0/9 | 4/9 | 0/9 | 2/9 | 0 | 0 |
| repair_v147_event_only_delta_top25 | R1-ZeroMomentResetNotReplicated | 0/9 | 2/9 | 0/9 | 1/9 | 0 | 0 |

最终科学结论更新：

```text
1. v14.6.1 的 zero_moment_reset 仍只能作为 S4c mechanism diagnostic。
2. v14.7 official confirmation 未复现为 S5。
3. every-step reset 的 endpoint 9/9 被 random/full reset controls 复现，
   所以不能写成 KAN/FMS-specific official success。
4. event-only reset 更符合计划语义，但 K2 只有 endpoint 4/9。
5. delta_top25 sparse mask 降低了 reset fraction，却进一步退到 endpoint 2/9。
6. strict gate 始终为 0/9，promotion_allowed 始终为 0。
```

停止边界：

```text
我现在不确定如何在 v14.7 当前计划内继续安全推进到 S5。

原因：
1. 计划 stop-go 明确写明 K2 < 9/9 时应停止 zero_moment_reset route，
   不能扩 action token。
2. K3 / K5 / K6 没有打开 S5，也没有提供更强机制解释。
3. 继续调 strength / lambda / lr / refresh 会变成计划禁止的网格搜索。
4. 用 AUCtime / LineC / CEp99 / NLL / ECE 反推 coordinate 或 trigger 是禁止的。
5. 降低 step_time gate、拼接局部 positive、或把 diagnostic 写成 promotion 都是禁止的。

下一步若继续，需要新预注册计划：
bounded post-event recovery window、
更精确的 train-stream-only affected-coordinate identification、
以及 FMS refresh overhead 降低机制。
这些不能在 v14.7 本轮临时写成 success。
```

## 11. 用户再次追问后的 stop-go 复核

本次没有新增训练；只重新对照完整计划的 stop-go 条款，确认是否还有未覆盖的合法继续路径。

计划条款：

```text
If K2 9/9 and controls fail:
  S5 candidate, promotion allowed only after audit/code review.

If K2 <9/9:
  stop zero_moment_reset route; do not add action tokens.

If K3/K5/K6 beat K2:
  update mechanism interpretation; still no action search.

If all state transport fail:
  return to substrate or FMS design, not action search.
```

当前事实：

```text
official_v147:
  K2 strict = 0/9
  K2 endpoint-vs-AdamW = 9/9
  controls endpoint full success = 3
  route = R2-ResetWorksButNotFMSSpecific

repair_v147_event_only_transport:
  K2 strict = 0/9
  K2 endpoint-vs-AdamW = 4/9
  route = R1-ZeroMomentResetNotReplicated

repair_v147_event_only_delta_top25:
  K2 strict = 0/9
  K2 endpoint-vs-AdamW = 2/9
  route = R1-ZeroMomentResetNotReplicated
```

复核结论：

```text
1. K2 没有任何一轮达到 9/9 strict。
2. K3/K5/K6 没有打开 S5。
3. every-step reset 被 controls 复现，不能作为 FMS-specific official proof。
4. event-only 与 delta_top25 都没有复现 v14.6.1 diagnostic 9/9。
5. 继续新增 action token、启动 controller、调 grid、
   或用 audit metric 反推 coordinate 均违反 v14.7 计划。
```

最终状态保持：

```text
official_s5_reached = 0
promotion_allowed = 0
best legal route = R2-ResetWorksButNotFMSSpecific / R1-ZeroMomentResetNotReplicated
```

## 12. 用户再次追问后的计划原文复核

本次没有新增训练；直接复核完整计划原文的 failure route、stop-go 与结论模板。

原文约束：

```text
5.4 Pass / fail:
  action token / grid / controller / audit metric direction /
  dataset-seed branch / diagnostic promotion 均触发 R0。

6.7 failure route:
  R1-ZeroMomentResetNotReplicated:
    K2 < 9/9。

  R2-ResetWorksButNotFMSSpecific:
    K2 过，但 random moment reset / full state reset 也过。

  R3/R4/R5:
    只有 K3/K5/K6 分别 beat K2 时才更新机制解释。

10.2 Stop-go:
  If K2 <9/9:
    stop zero_moment_reset route; do not add action tokens.

11 Case B:
  v14.6.1 S4c diagnostic did not reproduce in official runner.
  Do not continue with state reset variants unless new mechanism evidence appears.
```

对照结果：

```text
official_v147:
  route = R2-ResetWorksButNotFMSSpecific
  K2 strict = 0/9
  K2 endpoint-vs-AdamW = 9/9
  controls endpoint full success = 3

repair_v147_event_only_transport:
  route = R1-ZeroMomentResetNotReplicated
  K2 strict = 0/9
  K2 endpoint-vs-AdamW = 4/9

repair_v147_event_only_delta_top25:
  route = R1-ZeroMomentResetNotReplicated
  K2 strict = 0/9
  K2 endpoint-vs-AdamW = 2/9
```

复核结论：

```text
1. 当前没有 K2 9/9。
2. 当前没有 K3/K5/K6 beat K2 并打开 S5 的证据。
3. 当前没有可支持继续 state reset variants 的新机制证据。
4. 因此继续在 v14.7 内训练新 variant 会违反 Case B / stop-go。
5. 本轮保持 no-go，不 promotion，不补填成功。
```
