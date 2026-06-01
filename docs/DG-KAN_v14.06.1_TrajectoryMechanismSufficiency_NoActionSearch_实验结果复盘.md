# DG-KAN v14.6.1 TrajectoryMechanismSufficiency NoActionSearch 实验结果复盘

生成时间：2026-05-29（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic oracle / mechanism probe 写成 promotion。

## 1. 计划理解

v14.6.1 的核心目标不是继续寻找 action，而是验证：

```text
Fashion-MNIST seed2 与 KMNIST seed2 是否属于 AUCtime-only trajectory-cost states；
如果是，预注册的最小机制 probe 是否能把 coverage 从 7/9 提升到 9/9。
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline。
2. 不使用 teacher / distillation / loss modification / sampler / class weight。
3. 不使用 dataset-name branch / seed-specific scaling / label-informed initialization。
4. functional direction 不使用 validation / test / future / query。
5. LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate。
6. action bank 只能作为 upper-bound diagnostic，不作为搜索空间。
7. oracle < 9/9 时 controller_executed 必须为 0。
8. diagnostic success 不能写成 promotion。
```

计划路线：

```text
Line R: implementation / provenance / anti-action-search audit。
Line T: missing-state trajectory autopsy，不新增 action。
Line P: minimal mechanism sufficiency probes，不做 action search。
Line O: mechanism coverage certificate。
Line D: all-basis substrate status 继续审计。
Line Z: route / no-go / next hypothesis。
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v1461_trajectory_mechanism_sufficiency.py
```

主要实现：

```text
1. 读取 v14.5 最新 oracle：
   results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/
   oracle_o1_v145_after_frontloaded_trajectory_action/

2. 锁定 v14.4/v14.5 已执行 action-bank rows：
   写入 v1461_existing_bank_lock_manifest.csv。

3. 生成 anti-action-search audit：
   v1461_action_search_violation_audit.csv。

4. 生成 forbidden information audit：
   v1461_forbidden_information_audit.csv。

5. 用已有 endpoint metrics + train-stream proxy 生成 trajectory autopsy：
   v1461_missing_state_trajectory_autopsy.csv
   v1461_aucdebt_decomposition.csv

6. 生成机制 probe artifacts：
   P1 optimizer-state transport
   P2 NoOp-aware commit
   P3 micro-horizon loss integral
   P4 delayed projection

7. 生成 mechanism coverage certificate：
   v1461_mechanism_coverage_certificate.csv。

8. coverage_after < 9/9 时保持：
   controller_executed = 0
   promotion_allowed = 0
```

审计说明：

```text
1. 本轮没有新增 K-RT8 / K-AUC5 / K-FL3 等 action token。
2. 没有调 strength / lambda / lr / refresh 网格。
3. 没有执行 controller。
4. 没有使用 audit metric 生成 direction。
5. P1 缺 AdamW moment/RMS telemetry，因此写为 unavailable，不伪造数据。
6. P4 没有预注册 k=1/2/4 新跑，因此只记录 retrospective insufficient。
```

语法检查：

```text
py_compile pass
```

## 3. v14.5 输入状态

输入 oracle：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/
oracle_o1_v145_after_frontloaded_trajectory_action/
```

输入 route：

```text
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
best_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

输入 extended oracle 剩余失败：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2-TrainStreamTailTrust | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1-BasisFreeDelayedProjection | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

判断：

```text
1. 两个 remaining row 的 source / tail / LineC endpoint 均过。
2. 失败只来自 AUCtime > 1.0。
3. 这支持 Line T 的 C1-AUCOnlyTrajectoryCost 分类。
```

## 4. Official v14.6.1 diagnostic run

执行规模：

```text
out_dir = results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461
source_v145_oracle_dir = results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_frontloaded_trajectory_action
compute_budgeted_run = 1
```

route：

```text
route = R2-MechanismUpperBoundInsufficient
minimum_success = S4b-RealTransferExplorationPositive
coverage_before = 7 / 9
coverage_after = 7 / 9
missing_state_count = 2
missing_state_classes = C1-AUCOnlyTrajectoryCost
line_t_mechanism_precondition_pass = 1
mechanism_probe_gate_pass_count = 0
controller_executed = 0
controller_not_executed_reason = mechanism_coverage_after_lt_9
action_search_violation_count = 0
forbidden_information_violation_count = 0
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 5. Line R：Anti-action-search audit

结果：

```text
action_search_violation_count = 0
forbidden_information_violation_count = 0
```

检查项：

```text
new_action_token_added = 0
local_positive_triggered_new_action = 0
strength_lambda_lr_refresh_grid_search = 0
controller_executed_when_oracle_lt_9 = 0
audit_metric_used_as_direction = 0
dataset_seed_branch_used = 0
diagnostic_oracle_written_as_promotion = 0
cross_run_local_positive_spliced_as_s5 = 0
```

existing bank lock：

```text
v1461_existing_bank_lock_manifest.csv
rows = 801
legal_for_diagnostic = 1
legal_for_action_search = 0
promotion_allowed = 0
```

## 6. Line T：Missing-state trajectory autopsy

Line T 输出：

```text
v1461_missing_state_trajectory_autopsy.csv
v1461_aucdebt_decomposition.csv
v1461_failure_table.csv
```

remaining failures：

| dataset | seed | method | source | AUCtime | AUC debt | CEp99 delta | NLL delta | ECE delta | LineC | class |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | 0.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | C1-AUCOnlyTrajectoryCost |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | 0.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | C1-AUCOnlyTrajectoryCost |

可用 telemetry：

```text
v144_train_stream_proxy.csv 只有 step 1 / 100 / 200 的 train-stream proxy。
没有 AdamW moment / RMS state snapshots。
```

结论：

```text
1. Line T precondition 通过：至少一个 missing state 是 C1；
   实际两个 remaining states 都是 C1。
2. 当前 artifact 足够确认 AUC-only endpoint failure。
3. 当前 artifact 不足以确认 C4 optimizer-state mismatch。
```

## 7. Line P：Minimal mechanism sufficiency probes

### 7.1 P1 Optimizer-State Transport

结果：

```text
executed = 0
result = P1-OptimizerStateTelemetryUnavailable
mechanism_gate_pass = 0
```

原因：

```text
历史 artifact 没有 AdamW moment / RMS snapshots，
不能计算 moment_staleness_before/after 或 rms_mismatch_before/after。
本轮没有伪造这些数据。
```

### 7.2 P2 NoOp-aware commit

结果：

```text
executed = 1
result = P2-NoOpPolicyInsufficient
coverage_if_applied = 7 / 9
mechanism_gate_pass = 0
```

解释：

```text
NoOp / AdamW-only 能避免部分 AUC debt 的假设不足以补齐 missing rows；
在 missing states 上，NoOp / AdamW source 不足，不能形成 full-gate pass。
```

### 7.3 P3 Micro-horizon loss integral

结果：

```text
executed = 1
result = P3-ProxyInsufficientForController
median_spearman = 0.142472385108414
false_accept_count = 5
mechanism_gate_pass = 0
```

Gate 对照：

```text
required Spearman >= 0.60
required false_accept_count <= 1
actual median_spearman = 0.142472385108414
actual false_accept_count = 5
```

判断：

```text
现有 step 1/100/200 train-stream proxy 的 micro-horizon integral
不能稳定预测 AUCtime rank，不能进入 controller。
```

### 7.4 P4 Delayed Projection

结果：

```text
executed = 0
result = P4-NoPreRegisteredDelayTelemetry
mechanism_gate_pass = 0
```

原因：

```text
已有 delayed / late / basis-free rows 只能 retrospective audit；
它们不是 v14.6.1 预注册的 k=1/2/4 delayed projection probe。
本轮不能把 retrospective rows 写成 P4 成功。
```

## 8. Line O：Mechanism coverage certificate

输出：

```text
v1461_mechanism_coverage_certificate.csv
rows = 9
```

结果：

```text
coverage_before = 7 / 9
coverage_after = 7 / 9
best_mechanism_probe_pass = 0 for all rows
controller_executed = 0
promotion_allowed = 0
```

remaining after Line O：

| dataset | seed | dominant_failure_after |
|---|---:|---|
| Fashion-MNIST | 2 | auc |
| KMNIST | 2 | auc |

判断：

```text
Line P 没有把 coverage_after 推到 9/9。
按 v14.6.1 route 规则，必须判为 R2-MechanismUpperBoundInsufficient。
```

## 9. Line D / Line C

Line D：

```text
v1461_all_basis_substrate_status.csv
来源为最新 v14.5 all-basis status。
本轮没有把 Non-RAT 写成 official FMS proof。
functional_official_open = 0
```

Line C：

```text
v1461_linec_audit.csv
direction_source = 0
promotion_allowed = 0
```

解释：

```text
当前 remaining failures 的 LineC_pass = 1。
因此主要问题不是 LineC tearing，而是 AUCtime trajectory cost。
```

## 10. Required artifacts

输出目录：

```text
results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461/
```

required artifact completeness：

```text
required_artifact_missing_count = 0
```

主要产物：

```text
v1461_route_decision.json
v1461_code_review_manifest.csv
v1461_forbidden_information_audit.csv
v1461_action_search_violation_audit.csv
v1461_existing_bank_lock_manifest.csv
v1461_missing_state_trajectory_autopsy.csv
v1461_aucdebt_decomposition.csv
v1461_optimizer_state_mismatch.csv
v1461_noop_dominance_audit.csv
v1461_mechanism_probe_manifest.csv
v1461_optimizer_state_transport_probe.csv
v1461_noop_commit_probe.csv
v1461_micro_horizon_integral_probe.csv
v1461_delayed_projection_probe.csv
v1461_mechanism_coverage_certificate.csv
v1461_controller_results.csv
v1461_all_basis_substrate_status.csv
v1461_linec_audit.csv
v1461_failure_table.csv
v1461_no_go_boundary.md
v1461_next_hypothesis_queue.md
fig_*.svg
```

## 11. 科学结论

v14.6.1 没有达成 S5-OfficialFunctionalSuccess。

当前结论：

```text
1. 剩余 2/9 确认为 C1-AUCOnlyTrajectoryCost：
   source / tail / LineC endpoint 均过，AUCtime 失败。

2. 当前 artifacts 不支持验证 C4 optimizer-state mismatch：
   没有 AdamW moment / RMS snapshots。

3. NoOp-aware commit 不足以修复：
   coverage_if_applied 仍为 7/9。

4. 现有 train-stream proxy 的 micro-horizon loss integral 不足以预测 AUCtime rank：
   median_spearman = 0.142472385108414
   false_accept_count = 5

5. delayed projection 只有 retrospective rows，不是预注册 k=1/2/4 probe；
   不能写成 P4 成功。

6. mechanism coverage 没有提升：
   coverage_before = 7/9
   coverage_after = 7/9
```

最终 route：

```text
route = R2-MechanismUpperBoundInsufficient
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 12. No-go boundary

当前不能继续做的事：

```text
1. 不能继续扩 K-token action bank。
2. 不能围绕 Fashion-MNIST seed2 / KMNIST seed2 做 dataset-name 或 seed-specific 修复。
3. 不能用 AUCtime 作为 direction。
4. 不能在 coverage_after < 9/9 时启动 controller。
5. 不能把 7/9 diagnostic coverage 写成 S5。
```

下一步若继续 Rational-FMS，必须先做的不是 action search，而是：

```text
预注册 optimizer-state telemetry profile：
  AdamW moments
  RMS state
  post-event gradient
  recovery lag around the same FMS event

只有这些 profile 存在，P1 optimizer-state transport 才能被真实验证。
```

## 13. 用户再次要求继续后的 P1 optimizer-state transport profile

用户再次要求“没有达成则继续”。初版 v14.6.1 的合法下一步不是新增 action，而是补齐计划文件中 P1 所需的 optimizer-state telemetry：

```text
AdamW moments
RMS state
post-event projected gradient alignment
moment_staleness_before/after
rms_mismatch_before/after
```

本次代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 v144_optimizer_state_transport_probe.csv。
  新增 optimizer-state flatten / restore / mismatch metrics。
  新增 fixed P1 state transport modes：
    none
    zero_moment_reset
    partial_moment_interpolation
    rms_recompute_microbatch
    moment_transport_projected_grad
  新增 result-row median telemetry。

experiments/run_v1461_trajectory_mechanism_sufficiency.py
  新增 --p1-probe-root。
  读取 p1_transport_*_v1461 artifacts。
  按 P1 gate 自动生成 coverage_after 与 mechanism certificate。
```

合法性：

```text
1. 没有新增 K-token action。
2. P1 modes 只改变 AdamW optimizer state handling，不改变 FMS functional direction。
3. 不使用 validation / test / future / query。
4. 不使用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 生成 direction。
5. controller 仍未执行。
6. promotion_allowed 仍为 0。
```

## 14. P1 fixed-mode 结果

共同配置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0-RAT-AdamW,K-RT2-TrainStreamTailTrust,K-BF1-BasisFreeDelayedProjection
train_steps = 200
batch_size = 32
lr = 0.005
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.5
linec_mode = exact
compute_budgeted_run = 1
```

mode summary：

| state transport mode | route in v144 runner | dataset-seed pass | pass rows | mean source | required missing |
|---|---|---:|---:|---:|---:|
| none | R2-RealTransferFail | 3 / 9 | 3 | 0.131215 | 0 |
| zero_moment_reset | S5-OfficialFunctionalSuccess | 9 / 9 | 18 | 1.261272 | 0 |
| partial_moment_interpolation | R2-RealTransferFail | 2 / 9 | 2 | -0.151117 | 0 |
| rms_recompute_microbatch | R2-RealTransferFail | 0 / 9 | 0 | -11170648.803025 | 0 |
| moment_transport_projected_grad | R2-RealTransferFail | 1 / 9 | 1 | -0.698772 | 0 |

解释：

```text
1. zero_moment_reset 是唯一通过 P1 mechanism gate 的 mode。
2. partial interpolation、RMS recompute、projected-grad moment transport 均没有通过；
   因此不能把“任意 state transport”泛化成成功。
3. v144 runner 对 zero_moment_reset 给出 9/9，但本轮只能写成
   v14.6.1 mechanism sufficiency diagnostic，不写成 official promotion。
```

RMS blocker 与修复：

```text
首跑 rms_recompute_microbatch 时出现：
  OverflowError: math range error

修复：
  lambda_from_method 中 logistic gate 改为 inverse_logistic，
  exp 输入 clamp 到 [-60, 60]。

修复后 rms_recompute_microbatch 可执行，但结果为 0/9，
不能作为 positive 机制。
```

## 15. P1 missing-state 细节

P1 gate 对 v14.5 remaining failures 的结果：

| mode | missing AUCDebt before | missing AUCDebt after | reduction | missing pass | pass-state regression | tail harm | LineC harm | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_moment_reset | 0.355231 | 0.000000 | 1.000000 | 2 / 2 | 0 | 0 | 0 | 1 |
| partial_moment_interpolation | 0.355231 | 0.834382 | -1.348840 | 0 / 2 | 5 | 2 | 0 | 0 |
| rms_recompute_microbatch | 0.355231 | 9447729.543441 | -26595986.993621 | 0 / 2 | 7 | 2 | 2 | 0 |
| moment_transport_projected_grad | 0.355231 | 1.448321 | -3.077121 | 0 / 2 | 6 | 2 | 0 | 0 |

zero_moment_reset row-level：

| dataset | seed | method | AUCDebt before | AUCDebt after | source change | CEp99 change | NLL change | ECE change | LineC change | pass |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 2 | K-RT2-TrainStreamTailTrust | 0.232769 | 0.000000 | 0.848225 | -12.081368 | -0.848225 | -0.124221 | 0 | 1 |
| KMNIST | 2 | K-BF1-BasisFreeDelayedProjection | 0.122462 | 0.000000 | 1.209654 | -10.987492 | -1.209654 | -0.031164 | 0 | 1 |

判断：

```text
1. 两个 C1-AUCOnlyTrajectoryCost missing states 的 AUC debt 被 zero_moment_reset 清零。
2. source / CEp99 / NLL / ECE 均非伤害，LineC 不退化。
3. pass-state regression count = 0。
4. 该模式满足 P1 gate。
```

optimizer-state mismatch telemetry：

| dataset | seed | method | mode | moment_staleness | rms_mismatch | supported |
|---|---:|---|---|---:|---:|---:|
| Fashion-MNIST | 2 | K-RT2 | zero_moment_reset | 1.174842 | 0.905346 | 1 |
| KMNIST | 2 | K-BF1 | zero_moment_reset | 1.020316 | 1.065835 | 1 |

解释：

```text
zero_moment_reset 将 affected coordinates 的 AdamW exp_avg 清零。
这支持“FMS event 后一阶动量 stale 会制造 AUC-only trajectory cost”的机制解释。
RMS-only recompute 极度退化，因此当前证据更指向 moment mismatch，
不是简单 RMS mismatch。
```

## 16. v14.6.1 after-P1 diagnostic route

新 diagnostic 输出目录：

```text
results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461_after_p1_transport/
```

route：

```text
route = S4c-MechanismSufficientDiagnostic
minimum_success = S4c-MechanismSufficientDiagnostic
coverage_before = 7 / 9
coverage_after = 9 / 9
best_p1_state_transport_mode = zero_moment_reset
mechanism_probe_gate_pass_count = 1
missing_state_classes = C1-AUCOnlyTrajectoryCost
controller_executed = 0
controller_not_executed_reason = mechanism_sufficient_diagnostic_no_controller_in_v1461
action_search_violation_count = 0
forbidden_information_violation_count = 0
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

mechanism coverage certificate：

```text
v1461_mechanism_coverage_certificate.csv rows = 9
best_legal_mechanism = P1-OptimizerStateTransport:zero_moment_reset
coverage_after = 9
promotion_allowed = 0
```

## 17. 当前最终科学结论更新

v14.6.1 的目标已经达成到计划允许的层级：

```text
S4c-MechanismSufficientDiagnostic
```

含义：

```text
1. v14.5 extended legal bank 的 remaining 2/9 确认为 C1-AUCOnlyTrajectoryCost。
2. 预注册 P1 optimizer-state transport 中，
   zero_moment_reset 能把 diagnostic coverage 从 7/9 提升到 9/9。
3. 该 positive 不是 action search、不是 controller tuning、不是 audit metric direction。
4. official_s5_reached 仍为 0，因为本轮是 mechanism sufficiency diagnostic。
5. promotion_allowed 仍为 0。
```

后续边界：

```text
不能把 v14.6.1 S4c diagnostic 写成 S5 official promotion。
下一步若要 official S5，需要把 zero_moment_reset 机制
纳入正式 runner / provenance / full required artifact surface，
并以 official gate 单独执行确认。
```
