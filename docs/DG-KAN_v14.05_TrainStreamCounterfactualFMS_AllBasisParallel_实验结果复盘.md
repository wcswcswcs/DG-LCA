# DG-KAN v14.5 TrainStreamCounterfactualFMS AllBasisParallel 实验结果复盘

生成时间：2026-05-29（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic oracle 写成 promotion，不把 7/9 upper bound 写成 S5。

## 1. 计划理解

v14.5 的目标不是继续追加固定 K-RT 小变体，而是先回答：

```text
v14.4 的 6/9 是因为 action bank 不够，
还是因为缺少合法 train-stream action selection policy？
```

因此第一步必须执行 Line O：

```text
Action-bank oracle decomposition
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. direction 不使用 validation / test / future / query batch。
5. LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate。
6. diagnostic oracle 可以使用 audit metric，但不能 promotion。
7. O1/O2 oracle < 9/9 时，不能调 controller policy。
8. promotion_allowed 必须保持 0。
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py
```

主要实现：

```text
1. 新增 v14.5 Line O oracle / finalizer runner。
2. 读取 v14.4 已执行合法 artifacts，不生成 imagined rows。
3. 构造 core action bank：
   A0 K0/NoOp
   A1 K8 lowplasticity
   A2 K-RT1
   A3 K-RT2
   A4 K-RT3
   A5 K-RT4
   A6 K-RT5
   A7 K-RT6
   A8 K-RT7
   A9 single-refresh200
   A10 slowrefresh160
4. 额外构造 extended legal-v144 diagnostic bank：
   包含所有已有 v14.4 非 smoke / 非 control real-transfer rows。
5. 对每个 dataset-seed 做 O1 run-level diagnostic oracle。
6. 输出 v145_action_bank_rows / oracle / summary / missing_state_classes。
7. 当 oracle upper bound < 9/9 时，不执行 controller，并输出空 controller artifact 表头。
8. 输出 required manifest、forbidden audit、LineC/tail/AUC audit、figures、no-go、next queue 和 code review packet。
```

合法性说明：

```text
1. 本轮没有新增训练。
2. 本轮没有调 controller policy。
3. oracle 使用 audit metric 只做 diagnostic upper-bound，不用于 promotion。
4. 没有使用 validation/test/future/query 生成 direction。
5. 没有使用 dataset-name branch 或 seed-specific scaling。
```

语法检查：

```text
py_compile pass
```

## 3. Line O Oracle 结果

执行目录：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/
```

route：

```text
route = R2-ActionBankUpperBoundInsufficient
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
promotion_allowed = 0
controller_executed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

oracle summary：

| oracle level | dataset-seed pass | pass rows | source fail | AUC fail | CEp99 fail | NLL fail | ECE fail | LineC fail | upper bound pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O1-core-action-bank | 6 / 9 | 6 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| O1-extended-legal-v144-bank | 7 / 9 | 7 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |

判断：

```text
1. 严格计划 action bank 的 O1 upper bound 只有 6/9。
2. 即使把所有已有合法 v14.4 repair rows 纳入 extended diagnostic bank，
   O1 upper bound 也只有 7/9。
3. 因此当前问题不是“controller 还没学会选现有 action”；
   当前 action family 本身没有覆盖到 9/9。
4. 按计划，不能继续调 K-CF policy。
```

## 4. Missing State Classes

core bank 未覆盖：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | dominant failure |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 0 | K-RT3 | 0.184902 | 0.873954 | 0.166046 | -0.473994 | -0.046205 | 1 | CEp99 |
| Fashion-MNIST | 2 | K-RT7 | 0.233003 | 1.228868 | -0.889408 | -0.233003 | -0.032549 | 1 | AUC |
| KMNIST | 2 | K0 | -0.077374 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1 | source |

extended legal-v144 bank 未覆盖：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | dominant failure |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-RT1 | 0.122882 | 1.071418 | -0.545938 | -0.122882 | -0.007228 | 1 | AUC |

解释：

```text
1. extended bank 已经把 Fashion-MNIST seed0 补到 pass。
2. 但 Fashion-MNIST seed2 与 KMNIST seed2 仍是 AUC-only fail：
   source、tail、LineC 都可以通过，AUCtime 仍 > 1.0。
3. 这说明继续在现有 action 间选 fixed run，理论 upper bound 也不到 S5。
```

## 5. Controller 未执行

计划第 13.1 规定：

```text
If oracle bank < 9/9:
  Codex must not tune policy.
```

因此：

```text
K-CF0..K-CF6 controller 没有执行；
v145_counterfactual_event_log.csv 只有表头；
v145_controller_real_results.csv 只有表头；
v145_controller_summary.csv 记录 controller_executed = 0。
```

这不是未完成，而是按 v14.5 gate fail-closed：

```text
policy learning cannot be expected to reach S5 without new actions.
```

## 6. 新 Action Family 边界

本轮按计划输出的新 action family 需求：

```text
source_tail_auc_colocation_action
```

该 action family 需要解决：

```text
1. train-stream-only 地保持 source positive。
2. train-stream-only 地压低 local trajectory / AUC cost。
3. train-stream-only 地保持 tail proxy 不恶化。
4. 不使用 CEp99 / NLL / ECE / LineC / AUCtime audit metric 作为 direction。
5. 不做 dataset-name branch。
6. 不做 seed-specific scaling。
```

当前不能继续做的事情：

```text
1. 不能只调 selection threshold。
2. 不能把 extended oracle 7/9 写成 S4c controller success。
3. 不能拼接不同 run 的局部 positive 写成 9/9。
4. 不能 promotion。
```

## 7. Required Artifacts

required manifest：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_required_manifest.csv
```

检查结果：

```text
required_artifact_missing_count = 0
```

主要产物：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_route_decision.json
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_action_bank_rows.csv
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_action_bank_oracle.csv
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_oracle_summary.csv
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_missing_state_classes.csv
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_counterfactual_event_log.csv
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_controller_real_results.csv
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_controller_summary.csv
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_controller_failure_table.csv
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_forbidden_information_audit.csv
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_no_go_boundary.md
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_next_hypothesis_queue.md
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_code_review_packet.zip
```

## 8. 最终科学结论

v14.5 没有达成 S5-OfficialFunctionalSuccess：

```text
route = R2-ActionBankUpperBoundInsufficient
official_s5_reached = 0
promotion_allowed = 0
```

已闭合事实：

```text
1. core action bank O1 upper bound = 6/9。
2. extended legal-v144 diagnostic bank O1 upper bound = 7/9。
3. 当前合法 action bank 没有 9/9 upper bound。
4. LineC 不是 remaining blocker；extended bank 剩余失败均为 AUC-only。
5. Controller policy 没有执行，因为计划要求 oracle < 9/9 时不能调 policy。
6. required artifacts 缺失为 0。
7. forbidden direction audit 没有发现违规。
```

no-go boundary：

```text
1. 7/9 是 diagnostic oracle upper bound，不是 controller success。
2. 不能把 action-bank oracle 写成 S5。
3. 不能继续调 selection threshold 伪装成新机制。
4. 下一步需要新的 action family，而不是 K-CF policy tuning。
```

当前边界：

```text
我现在不确定如何在当前 v14.5 action bank 内安全推进到 S5，
因为按实际 artifact 计算，现有 action bank 的 upper bound 已经低于 9/9。

继续推进需要先设计并执行新的 source_tail_auc_colocation_action
或同等级的新 functional action family；
否则 train-stream controller 即使选择 oracle best，也不能达到 9/9。
```

## 9. 用户再次追问后的新 action family 尝试

用户再次要求未达成则继续。本次没有继续调 controller policy，因为 v14.5 计划第 13.1 明确规定：

```text
If oracle bank < 9/9:
  Codex must not tune policy.
  It must propose new action family, not new selection threshold.
```

因此本次尝试一个新 action family：

```text
source_tail_auc_colocation_action
```

具体实现为两个统一 action 变体：

```text
K-AUC1-SourceTailAUCColocation:
  使用 train-stream loss relative progress EMA、
  train-stream tail/risk state、
  FMS value state、
  train step phase，
  共同决定 FMS lambda。

K-AUC2-UltraLateAUCGuard:
  使用更保守的 ultra-late phase ramp、
  train-stream tail/risk state、
  FMS value state，
  尝试减少早期 AUC trajectory 破坏。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-AUC1-SourceTailAUCColocation。
  新增 K-AUC2-UltraLateAUCGuard。
  lambda_from_method 新增 train_loss_rel_progress_ema 状态。
  v144_train_stream_proxy.csv 增加 train_loss_rel_progress_ema 字段。

experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py
  修正 oracle=9/9 时不能直接写成 S5 的 route 逻辑；
  diagnostic oracle 只能决定是否允许进入 controller。
```

合法性说明：

```text
1. K-AUC1 / K-AUC2 不读取 validation/test/future/query batch。
2. 不使用 CEp99 / NLL / ECE / LineC / AUCtime audit metric 生成 direction。
3. 不使用 dataset-name branch。
4. 不使用 seed-specific scaling。
5. 本轮仍为 compute-budgeted repair；promotion_allowed = 0。
```

语法检查：

```text
py_compile pass
```

### 9.1 K-AUC action family 3x3 结果

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K-AUC1,K-AUC2,KCTRL
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
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 1 / 9
real_short_run_pass_rows = 2
mean_source_vs_best_control_noncontrol = -0.0183285309208764
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|---:|
| K-AUC1-SourceTailAUCColocation | 1 | 1 | 0.006986 | 0.003751 | 1.033899 | 9 |
| K-AUC2-UltraLateAUCGuard | 1 | 1 | -0.043643 | 0.042665 | 1.092919 | 9 |
| K0-RAT-AdamW | 0 | 0 | -0.118847 | -0.066930 | 1.024053 | 9 |
| KCTRL-RandomMatchedProjection | 0 | 0 | -0.046859 | 0.000000 | 1.000000 | 9 |

K-AUC row-level 摘要：

| dataset | seed | method | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-AUC1 | 0 | 0.144467 | 1.002476 | -2.557758 | -0.255927 | -0.009203 | 1 |
| MNIST | 1 | K-AUC1 | 1 | 0.284434 | 0.896601 | -6.689432 | -0.297697 | -0.035577 | 1 |
| MNIST | 2 | K-AUC2 | 0 | 0.070961 | 0.954484 | 0.105145 | -0.137891 | -0.015093 | 1 |
| Fashion-MNIST | 0 | K-AUC1 | 0 | -0.089671 | 1.033899 | -2.724142 | -0.199421 | -0.005417 | 1 |
| Fashion-MNIST | 1 | K-AUC1 | 0 | 0.003751 | 0.878995 | 3.037252 | -0.023624 | -0.001445 | 1 |
| Fashion-MNIST | 2 | K-AUC1 | 0 | 0.026654 | 1.251899 | 2.295580 | -0.026654 | -0.018876 | 1 |
| KMNIST | 0 | K-AUC2 | 0 | 0.213956 | 1.088674 | -4.498878 | -0.213956 | 0.014888 | 1 |
| KMNIST | 1 | K-AUC2 | 0 | -0.005364 | 0.960444 | -2.840906 | -0.486266 | -0.045201 | 1 |
| KMNIST | 2 | K-AUC1 | 0 | 0.142251 | 1.022674 | 4.781481 | -0.219625 | -0.007390 | 1 |

判断：

```text
1. K-AUC1 / K-AUC2 没有打开 S5。
2. 新 action family 反而从 existing best 6/9 / oracle 7/9 退化到 1/9。
3. LineC 全部为 1，说明失败不是 LineC。
4. K-AUC1 在 KMNIST seed2 的 source 为正且 AUCtime 接近 1，
   但 CEp99 tail 大幅 fail。
5. Fashion-MNIST seed2 仍 AUCtime fail 且 CEp99 fail。
6. 该 action family 不能作为 v14.5 修复。
```

### 9.2 纳入新 action 后的 v14.5 oracle 复核

执行目录：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_auc_action/
```

route：

```text
route = R2-ActionBankUpperBoundInsufficient
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
promotion_allowed = 0
core_oracle_dataset_seed_pass_count = 6
extended_legal_v144_oracle_dataset_seed_pass_count = 7
best_oracle_dataset_seed_pass_count = 7
controller_executed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

oracle summary 仍为：

| oracle level | dataset-seed pass | source fail | AUC fail | CEp99 fail | LineC fail | upper bound pass |
|---|---:|---:|---:|---:|---:|---:|
| O1-core-action-bank | 6 / 9 | 1 | 1 | 1 | 0 | 0 |
| O1-extended-legal-v144-bank | 7 / 9 | 0 | 2 | 0 | 0 | 0 |

remaining extended missing states 没有变化：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | dominant failure |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-RT1 | 0.122882 | 1.071418 | -0.545938 | -0.122882 | -0.007228 | 1 | AUC |

判断：

```text
1. 新 action family 没有提高 extended legal action bank upper bound。
2. 当前 upper bound 仍为 7/9。
3. 因此按 v14.5 计划仍不能进入 controller policy tuning。
4. 不能写成 S4c 或 S5。
```

## 10. 当前最终判断更新

v14.5 仍未达成 S5-OfficialFunctionalSuccess：

```text
route = R2-ActionBankUpperBoundInsufficient
best_oracle_dataset_seed_pass_count = 7 / 9
official_s5_reached = 0
promotion_allowed = 0
controller_executed = 0
```

本次继续后新增闭合事实：

```text
1. 已按计划第 13.1 尝试新 action family，而不是调 selection threshold。
2. K-AUC1 / K-AUC2 没有超过既有 action bank。
3. 纳入新 action 后，extended oracle 仍为 7/9。
4. 剩余缺口仍为 Fashion-MNIST seed2 与 KMNIST seed2 的 AUC-only fail。
5. 我现在不确定如何在不使用 AUCtime/CEp99/NLL/ECE/LineC audit metric、
   不做 dataset-name branch、
   不做 seed-specific scaling 的条件下，
   继续设计能补上这两个 AUC-only states 的合法 action family。
```

当前 no-go boundary：

```text
1. 不能调 controller，因为 action bank upper bound < 9/9。
2. 不能把 diagnostic oracle 7/9 写成 controller success。
3. 不能继续用 audit metric 反向修 AUC。
4. 下一步需要新的机制计划，而不是当前 v14.5 runner 内的阈值修补。
```

## 11. 用户再次追问后的 final-pulse action family 尝试

用户再次要求未达成则继续。本次没有调 controller threshold，也没有使用 audit metric 做方向；继续围绕 remaining extended missing states 的 AUC-only fail 尝试更窄的 final-pulse action family。

触发判断：

```text
1. K-AUC1 / K-AUC2 纳入后 extended oracle 仍为 7/9。
2. 剩余 extended missing states：
   Fashion-MNIST seed2: source/tail/LineC 已过，AUCtime fail。
   KMNIST seed2: source/tail/LineC 已过，AUCtime fail。
3. 合法方向不能读取 AUCtime 作 direction；
   因此尝试只使用 train-step phase 与 train-stream proxy 的 final-pulse action，
   让 FMS 注入更晚、更短，以减少 AUC trajectory damage。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-AUC3-FinalPulseTailTrust：
    final-phase ramp * train-stream tail trust * train-stream source gate。
  新增 K-AUC4-FinalPulseIdentityProjection：
    final-phase ramp * train-stream source gate，
    使用 identity projection，避免 basis-role projection 额外扰动。
```

合法性说明：

```text
1. phase ramp 只使用 train step phase。
2. train-stream gate 只使用 loss_q95 / margin_p10 / logit_rms / relative loss progress。
3. 不使用 validation/test/future/query batch。
4. 不使用 AUCtime / CEp99 / NLL / ECE / LineC 生成方向。
5. 不做 dataset-name branch，不做 seed-specific scaling。
6. compute_budgeted_run = 1；promotion_allowed = 0。
```

### 11.1 Final-pulse action family 结果

执行目录：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_finalpulse_auc_action_family/
```

route：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 3 / 9
real_short_run_pass_rows = 5
mean_source_vs_best_control_noncontrol = -0.028244892756144207
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|---:|
| K-AUC3-FinalPulseTailTrust | 2 | 2 | -0.052240 | 0.036833 | 1.025997 | 9 |
| K-AUC4-FinalPulseIdentityProjection | 3 | 3 | -0.004250 | 0.014231 | 1.052606 | 9 |
| K0-RAT-AdamW | 0 | 0 | -0.118847 | -0.066930 | 1.024053 | 9 |
| KCTRL-RandomMatchedProjection | 0 | 0 | -0.046859 | 0.000000 | 1.000000 | 9 |

代表 row：

| dataset | seed | method | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-AUC4 | 1 | 0.051266 | 0.973404 | -3.533220 | -0.162726 | -0.020336 | 1 |
| MNIST | 1 | K-AUC4 | 1 | 0.162412 | 0.854996 | -5.121954 | -0.175675 | -0.013772 | 1 |
| MNIST | 2 | K-AUC4 | 1 | 0.096663 | 0.997566 | -1.125879 | -0.163594 | -0.024115 | 1 |
| Fashion-MNIST | 0 | K-AUC3 | 0 | 0.197641 | 1.025997 | -0.838678 | -0.486733 | -0.045440 | 1 |
| Fashion-MNIST | 2 | K-AUC4 | 0 | -0.113909 | 1.115469 | 0.201349 | 0.113909 | 0.009914 | 1 |
| KMNIST | 2 | K-AUC3 | 0 | 0.070830 | 1.090232 | -0.634985 | -0.148204 | -0.000080 | 1 |

判断：

```text
1. final-pulse action 没有打开 S5。
2. K-AUC4 能覆盖 MNIST 0/1/2，但不能补上 Fashion-MNIST seed2 或 KMNIST seed2。
3. LineC 全部通过，因此当前失败仍不是 LineC。
4. KMNIST seed2 在 K-AUC3 下 source/tail 已为正向，但 AUCtime = 1.090232，仍 fail。
5. Fashion-MNIST seed2 在 final-pulse 下 source/tail 反而失稳。
6. 该 action family 不能作为 v14.5 修复。
```

### 11.2 纳入 final-pulse 后的 v14.5 oracle 复核

执行目录：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_finalpulse_action/
```

route：

```text
route = R2-ActionBankUpperBoundInsufficient
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
promotion_allowed = 0
core_oracle_dataset_seed_pass_count = 6
extended_legal_v144_oracle_dataset_seed_pass_count = 7
best_oracle_dataset_seed_pass_count = 7
controller_executed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

oracle summary：

| oracle level | dataset-seed pass | source fail | AUC fail | CEp99 fail | LineC fail | upper bound pass |
|---|---:|---:|---:|---:|---:|---:|
| O1-core-action-bank | 6 / 9 | 1 | 1 | 1 | 0 | 0 |
| O1-extended-legal-v144-bank | 7 / 9 | 0 | 2 | 0 | 0 | 0 |

remaining extended missing states：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | dominant failure |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-RT1 | 0.122882 | 1.071418 | -0.545938 | -0.122882 | -0.007228 | 1 | AUC |

判断：

```text
1. 纳入 K-AUC3 / K-AUC4 后，extended legal action bank upper bound 没有提高。
2. 当前 best oracle 仍为 7/9。
3. 按 v14.5 计划，action-bank upper bound < 9/9 时不能进入 controller policy tuning。
4. 因此不能写成 S4c 或 S5，也不能 promotion。
```

## 12. 当前最终判断更新

v14.5 仍未达成 S5-OfficialFunctionalSuccess：

```text
route = R2-ActionBankUpperBoundInsufficient
best_oracle_dataset_seed_pass_count = 7 / 9
official_s5_reached = 0
promotion_allowed = 0
controller_executed = 0
```

当前已尝试：

```text
1. 按 v14.5 Line O 建立 core action bank oracle。
2. 纳入全部 legal v14.4 action rows 做 extended oracle。
3. 修复了 diagnostic oracle 不能误写 S5 的 route 判定。
4. 尝试 source-tail-AUC colocation action family：
   K-AUC1 / K-AUC2。
5. 尝试 final-pulse action family：
   K-AUC3 / K-AUC4。
```

最终 no-go boundary：

```text
1. 当前 action bank upper bound 只有 7/9；
   按计划不能执行 controller tuning。
2. 剩余缺口仍为 Fashion-MNIST seed2 与 KMNIST seed2 的 AUC-only fail。
3. 已尝试的合法 train-stream-only AUC-oriented action family 没有补上缺口。
4. 我现在不确定如何在当前 v14.5 runner 内继续安全推进，
   而不把 AUCtime / CEp99 / NLL / ECE / LineC audit metric 用作 direction、
   不做 dataset-name branch、
   不做 seed-specific scaling、
   不拼接不同 run 的局部 positive、
   不把 7/9 diagnostic oracle 写成 S5。
```

## 13. 用户再次追问后的 basis-free / curvature-safe action family 尝试

用户再次要求继续。本次重新对照 v14.5 计划中 oracle bank < 9/9 后的推荐新 action family：

```text
low-cost curvature-safe FMS
basis-free FMS + delayed projection
two-phase source-tail separation
```

前文已尝试 source-tail-AUC colocation 与 final-pulse，本节继续尝试前两个计划示例方向。

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-BF1-BasisFreeDelayedProjection：
    前 80% train phase 使用 identity projection，
    后段恢复 value-preserving basis projection。

  新增 K-CURV1-CurvatureSafeBasisFreeFMS：
    使用 train_grad_norm 的增长作为 low-cost curvature proxy，
    identity projection，
    train-stream loss progress / tail trust / source gate 控制 lambda。

  train-stream proxy 新增 train_grad_norm。
```

实现 blocker 与修复：

```text
首跑 K-CURV1 时，grad_growth 极端值导致 math.exp overflow。
修复：对 grad_growth 和 sigmoid exponent 做有限 clamp。
该修复不改变信息边界，只防止数值溢出。
```

合法性说明：

```text
1. K-BF1 / K-CURV1 只使用 train-stream loss / train proxy / gradient norm / phase / FMS value state。
2. 不使用 validation/test/future/query。
3. 不使用 AUCtime / CEp99 / NLL / ECE / LineC 生成方向。
4. 不做 dataset-name branch，不做 seed-specific scaling。
5. compute_budgeted_run = 1；promotion_allowed = 0。
```

### 13.1 Basis-free / curvature-safe 结果

执行目录：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_basisfree_curvature_action_family/
```

route：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 1 / 9
real_short_run_pass_rows = 2
mean_source_vs_best_control_noncontrol = -0.04684809843699137
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|---:|
| K-BF1-BasisFreeDelayedProjection | 1 | 1 | 0.003184 | -0.036571 | 1.087945 | 9 |
| K-CURV1-CurvatureSafeBasisFreeFMS | 1 | 1 | -0.096880 | -0.002920 | 1.032248 | 9 |
| K0-RAT-AdamW | 0 | 0 | -0.118847 | -0.066930 | 1.024053 | 9 |
| KCTRL-RandomMatchedProjection | 0 | 0 | -0.046859 | 0.000000 | 1.000000 | 9 |

关键 row：

| dataset | seed | method | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | K-BF1 | 1 | 0.171142 | 0.883369 | -5.870255 | -0.184405 | -0.022742 | 1 |
| MNIST | 1 | K-CURV1 | 1 | 0.272668 | 0.902394 | -5.379744 | -0.285930 | -0.032290 | 1 |
| Fashion-MNIST | 2 | K-CURV1 | 0 | 0.029584 | 1.078045 | -0.056999 | -0.029584 | 0.004322 | 1 |
| KMNIST | 2 | K-BF1 | 0 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 |

判断：

```text
1. K-BF1 / K-CURV1 没有打开 S5。
2. K-BF1 在 KMNIST seed2 上实现 source/tail/LineC 正向，
   但 AUCtime = 1.122462，仍 fail。
3. Fashion-MNIST seed2 仍没有形成 pass。
4. LineC 全部通过，主 blocker 仍是 AUC trajectory。
```

### 13.2 纳入 basis-free / curvature-safe 后的 oracle 复核

执行目录：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_basisfree_curvature_action/
```

route：

```text
route = R2-ActionBankUpperBoundInsufficient
extended_legal_v144_oracle_dataset_seed_pass_count = 7
best_oracle_dataset_seed_pass_count = 7
controller_executed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

remaining extended missing states：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | dominant failure |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

判断：

```text
1. K-BF1 改写了 KMNIST seed2 的 best oracle row，
   但没有让它通过 AUC gate。
2. extended oracle 仍为 7/9。
3. 因此仍不能进入 controller policy tuning。
```

## 14. 用户再次追问后的 two-phase source-tail action family 尝试

继续覆盖 v14.5 计划示例中的 two-phase source-tail separation。

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-2P1-TwoPhaseSourceTailSeparation：
    early phase: basis-free source phase；
    late phase: projected tail-trust phase。

  新增 K-2P2-TwoPhaseDelayedSourceTail：
    early NoOp；
    middle basis-free source phase；
    late projected tail-trust phase。
```

合法性说明：

```text
1. phase split 只使用 train step phase。
2. lambda 只使用 train-stream loss progress / source state / tail risk state。
3. 不使用 AUCtime / CEp99 / NLL / ECE / LineC 生成方向。
4. 不做 dataset-name branch，不做 seed-specific scaling。
```

### 14.1 Two-phase 结果

执行目录：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_twophase_action_family/
```

route：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 2 / 9
real_short_run_pass_rows = 3
mean_source_vs_best_control_noncontrol = -0.1334330207771725
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|---:|
| K-2P1-TwoPhaseSourceTailSeparation | 2 | 2 | -0.021079 | 0.058117 | 1.038791 | 9 |
| K-2P2-TwoPhaseDelayedSourceTail | 1 | 1 | -0.245787 | -0.129133 | 1.074341 | 8 |
| K0-RAT-AdamW | 0 | 0 | -0.118847 | -0.066930 | 1.024053 | 9 |
| KCTRL-RandomMatchedProjection | 0 | 0 | -0.046859 | 0.000000 | 1.000000 | 9 |

关键 row：

| dataset | seed | method | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | K-2P1 | 1 | 0.211197 | 0.934317 | -7.682886 | -0.224460 | -0.018661 | 1 |
| MNIST | 2 | K-2P1 | 1 | 0.106932 | 0.956239 | -2.084441 | -0.173862 | -0.017363 | 1 |
| Fashion-MNIST | 2 | K-2P1 | 0 | 0.058117 | 1.222266 | 1.996523 | -0.058117 | -0.014463 | 1 |
| KMNIST | 2 | K-2P1 | 0 | -0.103105 | 1.116455 | 2.066853 | 0.025731 | -0.010889 | 1 |

判断：

```text
1. two-phase action family 没有打开 S5。
2. K-2P1 / K-2P2 低于既有 best 6/9，也低于 extended oracle 7/9。
3. Fashion-MNIST seed2 仍 AUC + CEp99 fail。
4. KMNIST seed2 在 two-phase 下 source 反而失稳。
```

### 14.2 纳入 two-phase 后的 oracle 复核

执行目录：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_twophase_action/
```

route：

```text
route = R2-ActionBankUpperBoundInsufficient
extended_legal_v144_oracle_dataset_seed_pass_count = 7
best_oracle_dataset_seed_pass_count = 7
controller_executed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

oracle summary：

| oracle level | dataset-seed pass | source fail | AUC fail | CEp99 fail | LineC fail | upper bound pass |
|---|---:|---:|---:|---:|---:|---:|
| O1-core-action-bank | 6 / 9 | 1 | 1 | 1 | 0 | 0 |
| O1-extended-legal-v144-bank | 7 / 9 | 0 | 2 | 0 | 0 | 0 |

remaining extended missing states：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | dominant failure |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

判断：

```text
1. 纳入 two-phase 后，extended legal action bank upper bound 仍未提高。
2. 当前 best oracle 仍为 7/9。
3. 按 v14.5 计划，oracle bank < 9/9 时不能执行 controller tuning。
4. 不能写成 S4c 或 S5，也不能 promotion。
```

## 15. 当前最终判断更新

v14.5 仍未达成 S5-OfficialFunctionalSuccess：

```text
route = R2-ActionBankUpperBoundInsufficient
best_oracle_dataset_seed_pass_count = 7 / 9
official_s5_reached = 0
promotion_allowed = 0
controller_executed = 0
```

当前已按计划 oracle-bank-insufficient 方向继续尝试：

```text
1. source-tail-AUC colocation action family:
   K-AUC1 / K-AUC2。
2. final-pulse AUC action family:
   K-AUC3 / K-AUC4。
3. basis-free delayed projection / curvature-safe action family:
   K-BF1 / K-CURV1。
4. two-phase source-tail separation action family:
   K-2P1 / K-2P2。
```

最终 no-go boundary 更新：

```text
1. 当前 action bank upper bound 仍只有 7/9；
   不能执行 controller tuning。
2. 剩余缺口稳定收敛到 Fashion-MNIST seed2 与 KMNIST seed2 的 AUC-only fail。
3. 计划中明确建议的新 action family 示例均已覆盖：
   low-cost curvature-safe FMS、
   two-phase source-tail separation、
   basis-free FMS + delayed projection。
4. 我现在不确定如何在当前 v14.5 runner 内继续安全推进，
   而不把 AUCtime / CEp99 / NLL / ECE / LineC audit metric 用作 direction、
   不做 dataset-name branch、
   不做 seed-specific scaling、
   不拼接不同 run 的局部 positive、
   不把 7/9 diagnostic oracle 写成 S5。
```

## 16. 用户再次要求继续后的 Line W Wavelet bounded hardening

本次继续推进 v14.5 计划中的 Line W。注意：

```text
1. Line W 是 all-basis substrate bounded parallel line。
2. Line W 不阻塞 Line K，也不能替代 Rational real-transfer S5。
3. Wavelet 只有 full 3x3 substrate pass 后，才允许打开 Wavelet-FMS synthetic proof。
4. 本次不使用 LineC / CEp99 / NLL / ECE / AUCtime 作为 direction。
5. 本次不使用 labels / validation / test / future / query batch 生成方向。
```

代码修改：

```text
1. 新增 experiments/run_v145_wavelet_linew_hardening.py。
   用于记录 v14.5 Line W must-record 字段，
   包括 AUCtime_ratio、LineC reservoir audit、support geometry proxy。

2. 修改 experiments/run_v143_nonrat_compact_task_health_probe.py。
   增加 readout gradient gate:
     linear_readout_grad010
     linear_readout_grad005

3. 在 v145 Wavelet runner 中实现 configs:
   W123-train-entropy
   W4-support-entropy-readout
   W5-scale-occupancy
   W6-local-tail-guard
   W7-reservoir-balance
```

合法性说明：

```text
official_fms_proof_executed = 0
open_wavelet_fms_synthetic_proof = 0
promotion_allowed = 0
linec_tail_auc_used_for_direction = 0
labels_used_for_direction = 0
validation_test_future_query_used_for_direction = 0
```

语法检查：

```text
py_compile pass
```

### 16.1 Line W W123/W4/W5/W6 full 3x3

执行目录：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/wavelet_linew_full_v145/
```

route：

```text
route = W0-WaveletSubstrateFail
wavelet_dataset_seed_pass_count = 1 / 9
wavelet_gate_pass_rows = 6
official_fms_proof_executed = 0
open_wavelet_fms_synthetic_proof = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

config summary：

| config | rows | dataset-seed pass | gate pass rows | best AUCtime | best LineC | min NoiseSignalLeak | min RealSignalReservoirRatio |
|---|---:|---:|---:|---:|---:|---:|---:|
| W123-train-entropy | 27 | 1 | 3 | 1.036591 | 1.0 | 0.087610 | 0.651743 |
| W4-support-entropy-readout | 27 | 1 | 3 | 1.035637 | 1.0 | 0.086267 | 0.653829 |
| W5-scale-occupancy | 27 | 0 | 0 | 1.035011 | 1.0 | 0.088432 | 0.656402 |
| W6-local-tail-guard | 27 | 0 | 0 | 1.035011 | 1.0 | 0.088432 | 0.656402 |

通过 rows：

| dataset | seed | config | candidates | mean delta | AUCtime | LineC | NoiseSignalLeak | RealSignalReservoirRatio |
|---|---:|---|---|---:|---:|---:|---:|---:|
| KMNIST | 0 | W123 | D-WAV17/18/19 | -0.046875 | 1.1347-1.1351 | 1.0 | 0.1347-0.1366 | 0.6517-0.6574 |
| KMNIST | 0 | W4 | D-WAV17/18/19 | -0.046875 | 1.1342-1.1347 | 1.0 | 0.1325-0.1347 | 0.6603-0.6660 |

failure counts：

```text
LineC = 96
RealSignalReservoirRatio = 66
NoiseSignalLeak = 60
task = 42
```

best row by dataset-seed：

| dataset | seed | pass | best config | best candidate | mean delta | AUCtime | LineC | NoiseSignalLeak | RSR | failure |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| MNIST | 0 | 0 | W123 | D-WAV19 | -0.093750 | 1.705199 | 0.0 | 0.272144 | 0.667839 | task,LineC,NoiseSignalLeak |
| MNIST | 1 | 0 | W123 | D-WAV19 | -0.062500 | 1.266851 | 0.0 | 0.119641 | 0.753472 | task,LineC,RealSignalReservoirRatio |
| MNIST | 2 | 0 | W123 | D-WAV18 | -0.156250 | 1.358990 | 0.0 | 0.256092 | 0.679193 | task,LineC,NoiseSignalLeak |
| Fashion-MNIST | 0 | 0 | W6 | D-WAV17 | 0.015625 | 1.095563 | 0.0 | 0.292096 | 0.676128 | LineC,NoiseSignalLeak |
| Fashion-MNIST | 1 | 0 | W4 | D-WAV17 | -0.031250 | 1.072189 | 0.0 | 0.226237 | 0.739109 | LineC,NoiseSignalLeak,RealSignalReservoirRatio |
| Fashion-MNIST | 2 | 0 | W4 | D-WAV17 | -0.015625 | 1.057497 | 0.0 | 0.354616 | 0.893309 | LineC,NoiseSignalLeak,RealSignalReservoirRatio |
| KMNIST | 0 | 1 | W4 | D-WAV17 | -0.046875 | 1.134203 | 1.0 | 0.132464 | 0.666046 | - |
| KMNIST | 1 | 0 | W6 | D-WAV19 | 0.015625 | 1.035011 | 0.0 | 0.088432 | 0.963804 | LineC,RealSignalReservoirRatio |
| KMNIST | 2 | 0 | W6 | D-WAV19 | -0.031250 | 1.321511 | 0.0 | 0.091629 | 0.909896 | LineC,RealSignalReservoirRatio |

判断：

```text
1. W123/W4/W5/W6 未达到 full 3x3 substrate pass。
2. 只有 KMNIST seed0 被稳定打开。
3. 主 blocker 不是 workspace / step / AUCtime，而是 LineC reservoir：
   NoiseSignalLeak 与 RealSignalReservoirRatio 不能在多数 dataset-seed 同时满足。
4. 因此不能打开 Wavelet-FMS synthetic proof。
```

### 16.2 W7 reservoir-balance 修复

触发原因：

```text
计划 13.5 要求：
If Wavelet substrate fails:
  classify failure as workspace / task / AUC / LineC / tail;
  if LineC reservoir fail, try train-stream reservoir proxy only;
  if task collapses, repair substrate, not FMS;
  do not enter Wavelet-FMS official proof.
```

局部探针：

```text
freeze_linear_readout 可以让 MNIST seed0 LineC pass_rate = 1.0，
但 task collapse：mean_delta_vs_MLP = -0.5。
因此不能使用 freeze 作为正式 repair。
```

正式 W7：

```text
W7-reservoir-balance =
  quantile_scale050 support
  + linear_readout_grad010
  + no output geometry scaling
```

执行目录：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/wavelet_linew_w7_reservoir_balance_full_v145/
```

route：

```text
route = W0-WaveletSubstrateFail
wavelet_dataset_seed_pass_count = 1 / 9
wavelet_gate_pass_rows = 3
official_fms_proof_executed = 0
open_wavelet_fms_synthetic_proof = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

通过 rows：

| dataset | seed | config | candidates | mean delta | AUCtime | LineC | NoiseSignalLeak | RealSignalReservoirRatio |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | 1 | W7 | D-WAV17/18/19 | -0.031250 | 1.0725-1.0735 | 1.0 | 0.1275-0.1290 | 0.4863-0.4886 |

failure counts：

```text
LineC = 21
RealSignalReservoirRatio = 18
task = 12
NoiseSignalLeak = 12
```

判断：

```text
1. W7 从 LineC reservoir 方向确实能打开 Fashion-MNIST seed1，
   但并没有提高 full coverage。
2. W7 没有超过 W123/W4/W5/W6 的 best 1/9；
   只是通过的 dataset-seed 从 KMNIST seed0 换成 Fashion-MNIST seed1。
3. Wavelet substrate 仍不能进入 Wavelet-FMS proof。
4. 不允许把局部 Wavelet substrate positive 写成 functional S5。
```

## 17. 当前最终判断更新

v14.5 仍未达成 S5-OfficialFunctionalSuccess。

主线状态仍为：

```text
route = R2-ActionBankUpperBoundInsufficient
best_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

Line W 更新：

```text
best Wavelet substrate dataset-seed pass = 1 / 9
best Wavelet gate pass rows = 6
open_wavelet_fms_synthetic_proof = 0
promotion_allowed = 0
```

已新增覆盖的计划方向：

```text
1. Line W W123 D-WAV17/18/19 train-entropy hardening。
2. W4 support + entropy + delayed readout mixing。
3. W5 scale occupancy balanced update。
4. W6 local-tail coverage guard。
5. W7 train-stream reservoir-balance readout gradient gate。
```

当前边界：

```text
1. Line K/O 仍被 action bank upper bound < 9/9 阻断；
   不能进入 controller tuning。
2. Line W 也没有 full 3x3 substrate pass；
   不能进入 Wavelet-FMS official proof。
3. 不能用 LineC / AUCtime / CEp99 / NLL / ECE audit metric 作为 direction。
4. 不能做 dataset-name branch、seed-specific scaling 或拼接不同 run 局部 positive。
5. 不能把 Rational 7/9 diagnostic oracle 或 Wavelet 1/9 substrate positive 写成 S5。
```

我现在仍不确定如何在当前 v14.5 合法约束内继续安全推进到 S5，
而不引入 forbidden direction source 或降低 gate。

## 18. 用户再次要求继续后的 Line D All-basis bounded substrate repair

本次继续覆盖计划文件中的 Line D：

```text
RBF / FastKAN:
  compact center occupancy repair、width condition guard、
  low-k active-center local support、no dense RBF materialization。

Chebyshev:
  recurrence no-materialize lifetime、degree-energy damping、
  low-degree identity residual、high-degree late enable。

Fourier:
  band-limited identity residual、phase-stable low-frequency anchor、
  high-frequency quarantine。
```

本节只做 bounded substrate repair / audit：

```text
official_fms_proof_executed = 0
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

### 18.1 compact workspace 重新测量

新增产物：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_allbasis_workspace_v145/
```

结果：

```text
candidate rows = 32
manual_workspace_gate_pass_rows(route total) = 22
manual_no_materialize rows:
  RBF = 6 / 6 pass
  Chebyshev = 5 / 5 pass
  Fourier = 5 / 5 pass
promotion_allowed = 0
```

判断：

```text
Line D 本轮没有先死在 compact workspace。
真正 blocker 转到 real 3x3 task-health / LineC。
```

### 18.2 full Line D all-basis task-health

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
candidates =
  D-RBF12..D-RBF17
  D-CHE16..D-CHE20
  D-FOU16..D-FOU20
train_size = 128
val_size = 64
epochs = 3
lr = 0.002
rbf_center_repair = quantile_width075
output_geometry_repair = train_entropy_t080_100_else050
```

route：

```text
line_d_dataset_seed_pass_count = 1 / 9
line_d_gate_pass_rows = 5
line_d_total_rows = 144
workspace_manual_gate_pass_rows = 144
promotion_allowed = 0
```

family summary：

| family | rows | dataset-seed pass | gate pass rows | workspace pass rows | best mean delta | best NLL ratio | best LineC |
|---|---:|---:|---:|---:|---:|---:|---:|
| D-RBF | 54 | 0 | 0 | 54 | -0.171875 | 1.344919 | 1.0 |
| D-CHE | 45 | 0 | 0 | 45 | -0.171875 | 1.344722 | 1.0 |
| D-FOU | 45 | 1 | 5 | 45 | 0.015625 | 1.211116 | 1.0 |

failure counts：

```text
LineC = 66
mean_task = 139
worst_task = 137
```

通过 rows：

```text
Fashion-MNIST seed1:
  D-FOU16 / D-FOU17 / D-FOU18 / D-FOU19 / D-FOU20 pass
```

判断：

```text
1. RBF / Chebyshev 虽然 workspace 已过，但 real task-health 仍崩。
2. Fourier 是唯一有局部 substrate pass 的 family。
3. full Line D 不能进入 Non-RAT FMS proof。
```

### 18.3 Fourier low-frequency lr005 epochs6 repair

触发原因：

```text
full Line D 的 pass rows 全来自 Fourier；
最接近失败行也多来自 D-FOU16/17/19/20。
因此尝试统一的 low-frequency Fourier longer/higher-lr substrate repair。
```

执行规模：

```text
candidates = D-FOU16,D-FOU17,D-FOU18,D-FOU19,D-FOU20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 6
lr = 0.005
output_geometry_repair = train_entropy_t080_100_else050
```

route：

```text
line_d_dataset_seed_pass_count = 3 / 9
line_d_gate_pass_rows = 6
line_d_total_rows = 45
workspace_manual_gate_pass_rows = 45
promotion_allowed = 0
```

通过 rows：

| dataset | seed | candidate | mean delta | NLL ratio | LineC |
|---|---:|---|---:|---:|---:|
| MNIST | 2 | D-FOU17 | 0.062500 | 1.058080 | 1.0 |
| MNIST | 2 | D-FOU18 | 0.062500 | 1.058080 | 1.0 |
| MNIST | 2 | D-FOU19 | 0.078125 | 1.056577 | 1.0 |
| MNIST | 2 | D-FOU20 | 0.062500 | 1.058080 | 1.0 |
| Fashion-MNIST | 0 | D-FOU16 | 0.046875 | 1.272914 | 1.0 |
| Fashion-MNIST | 1 | D-FOU16 | 0.093750 | 0.804325 | 1.0 |

failure counts：

```text
LineC = 33
mean_task = 7
worst_task = 6
```

判断：

```text
1. Fourier low-frequency lr005 epochs6 是本节 single-config best。
2. 它把 Line D 从 1/9 推到 3/9。
3. 但仍远不到 full 3x3 substrate pass。
```

### 18.4 Fourier output geometry fixed050 / fixed025 probes

fixed050：

```text
line_d_dataset_seed_pass_count = 3 / 9
line_d_gate_pass_rows = 6
failure counts:
  LineC = 38
  mean_task = 7
  worst_task = 6
```

fixed025：

```text
line_d_dataset_seed_pass_count = 2 / 9
line_d_gate_pass_rows = 5
failure counts:
  LineC = 36
  NLL = 1
  mean_task = 7
  worst_task = 6
```

关键观察：

```text
1. fixed025 打开 MNIST seed1，
   但同时丢失 MNIST seed2 / Fashion-MNIST seed0 的 coverage。
2. fixed050 没有超过 train_entropy 配置，LineC failure 反而更多。
3. 输出尺度不是稳定单调修复，只是在不同 dataset-seed 之间交换 coverage。
4. 因此不能继续把 LineC audit 结果用于阈值扫参。
```

### 18.5 Line D best-audit 汇总

汇总产物：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_allbasis_summary_v145/
```

主要文件：

```text
v145_lined_allbasis_task_health.csv
v145_lined_allbasis_failure_table.csv
v145_lined_allbasis_summary.csv
v145_lined_allbasis_route.json
v145_lined_fourier_lr005_epochs6_task_health.csv
v145_lined_fourier_lr005_epochs6_route.json
v145_lined_fourier_lr005_epochs6_fixed050_route.json
v145_lined_fourier_lr005_epochs6_fixed025_route.json
v145_lined_all_configs_task_health.csv
v145_lined_best_row_by_dataset_seed.csv
v145_lined_all_configs_route.json
```

single-config best：

```text
config = lined_fourier_lowfreq_lr005_epochs6_v145
dataset-seed pass = 3 / 9
gate pass rows = 6
```

best across configs（audit-only）：

```text
dataset-seed pass = 4 / 9
gate pass rows = 22
line_d_total_rows = 279
official_fms_proof_executed = 0
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

best row by dataset-seed：

| dataset | seed | pass | config | candidate | blocker |
|---|---:|---:|---|---|---|
| MNIST | 0 | 0 | lr005_epochs6 | D-FOU17 | mean_task,worst_task |
| MNIST | 1 | 1 | lr005_epochs6_fixed025 | D-FOU17 | - |
| MNIST | 2 | 1 | lr005_epochs6 | D-FOU19 | - |
| Fashion-MNIST | 0 | 1 | lr005_epochs6 | D-FOU16 | - |
| Fashion-MNIST | 1 | 1 | lr005_epochs6 | D-FOU16 | - |
| Fashion-MNIST | 2 | 0 | lr005_epochs6_fixed050 | D-FOU17 | LineC |
| KMNIST | 0 | 0 | lr005_epochs6 | D-FOU19 | LineC |
| KMNIST | 1 | 0 | lr005_epochs6 | D-FOU17 | LineC |
| KMNIST | 2 | 0 | lr005_epochs6 | D-FOU19 | LineC |

判断：

```text
1. Line D 没有达到 full 3x3 substrate pass。
2. RBF / Chebyshev workspace 已打开，但 task-health 没打开。
3. Fourier low-frequency 是当前唯一有效的 Non-RAT substrate 方向，
   但 single-config best 只有 3/9。
4. 跨配置 best 4/9 只是 audit-only；
   不能拼接不同配置写成 substrate success。
5. Non-RAT official FMS proof 仍不能启动。
```

## 19. 当前最终判断更新

v14.5 仍未达成 S5-OfficialFunctionalSuccess。

主线状态仍为：

```text
route = R2-ActionBankUpperBoundInsufficient
best_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

Line W 状态仍为：

```text
best Wavelet substrate dataset-seed pass = 1 / 9
open_wavelet_fms_synthetic_proof = 0
promotion_allowed = 0
```

Line D 新增状态：

```text
manual_no_materialize compact workspace:
  RBF = 6/6
  Chebyshev = 5/5
  Fourier = 5/5

full all-basis task-health:
  dataset-seed pass = 1/9

Fourier low-frequency single-config best:
  dataset-seed pass = 3/9

best across configs audit-only:
  dataset-seed pass = 4/9
```

已新增覆盖的计划方向：

```text
1. Line D RBF D-RBF12..17 compact center / width / local support repair。
2. Line D Chebyshev D-CHE16..20 degree-energy / recurrence / low-degree residual repair。
3. Line D Fourier D-FOU16..20 low-frequency / phase / identity residual repair。
4. Fourier low-frequency lr005 epochs6 longer/higher-lr task-health repair。
5. Fourier fixed050 / fixed025 output geometry probes。
```

当前边界：

```text
1. Line K/O 仍被 action bank upper bound < 9/9 阻断；
   不能进入 controller tuning。
2. Line W 没有 full 3x3 substrate pass；
   不能进入 Wavelet-FMS official proof。
3. Line D 没有 full 3x3 substrate pass；
   不能进入 Non-RAT official FMS proof。
4. Fourier output geometry 显示 coverage 交换而非稳定修复；
   继续扫阈值会变成 LineC audit-tuned scaling，风险不合法。
5. 不能使用 LineC / AUCtime / CEp99 / NLL / ECE audit metric 作为 direction。
6. 不能做 dataset-name branch、seed-specific scaling 或拼接不同 run 局部 positive。
7. 不能把 Rational 7/9 oracle、Wavelet 1/9 substrate、
   或 Line D Fourier 3/9 single-config positive 写成 S5。
```

我现在不确定如何在当前 v14.5 合法约束内继续安全推进到 S5，
而不引入 forbidden direction source、dataset branch、seed-specific scaling、
或降低 gate。

## 20. 用户再次要求继续后的 micro-pulse / curvature low-lambda action probe

用户再次要求未达成则继续。本次先重新检查 v14.5 最新 oracle：

```text
latest route = R2-ActionBankUpperBoundInsufficient
extended legal v144 oracle = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

剩余 two failures 都是 AUC-only：

| dataset | seed | best row | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | official_v144 / K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | basisfree_curvature / K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

判断：

```text
1. 剩余缺口不是 source、tail 或 LineC。
2. 继续增强 tail/source 方向不合适。
3. 本次尝试一个全局、统一参数、train-stream-only 的弱末端 action：
   micro-pulse / curvature-safe action。
4. 该 action 不使用 AUCtime 作为 direction；
   AUCtime 仍只作为最终 audit/gate。
```

### 20.1 micro-pulse curvature action

执行规模：

```text
out_dir = results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_micro_pulse_curvature_action_family
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K-AUC3,K-AUC4,K-CURV1,KCTRL
train_steps = 200
batch_size = 32
lr = 0.005
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.25
linec_mode = exact
compute_budgeted_run = 1
```

route：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 3 / 9
real_short_run_pass_rows = 6
mean_source_vs_best_control_noncontrol = -0.048930605252583824
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|---:|
| K-AUC3-FinalPulseTailTrust | 2 | 2 | -0.051085 | 0.036847 | 1.025813 | 9 |
| K-AUC4-FinalPulseIdentityProjection | 3 | 3 | -0.004408 | 0.013169 | 1.052545 | 9 |
| K-CURV1-CurvatureSafeBasisFreeFMS | 1 | 1 | -0.091298 | -0.016946 | 1.031439 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-AUC4 | 0.050941 | 0.973495 | -3.536077 | -0.162381 | -0.020297 | 1 |
| MNIST | 1 | K-AUC3 | 0.175496 | 0.906495 | -5.737762 | -0.188712 | -0.016337 | 1 |
| MNIST | 1 | K-AUC4 | 0.162172 | 0.855069 | -5.133204 | -0.175389 | -0.014154 | 1 |
| MNIST | 1 | K-CURV1 | 0.272944 | 0.902376 | -5.382935 | -0.286160 | -0.033510 | 1 |
| MNIST | 2 | K-AUC3 | 0.036847 | 0.929429 | -1.094476 | -0.103742 | 0.000018 | 1 |
| MNIST | 2 | K-AUC4 | 0.096762 | 0.997539 | -1.112623 | -0.163657 | -0.024153 | 1 |

failure counts（non-control rows）：

```text
AUCtime = 18
source = 14
CEp99_tail = 11
NLL_tail = 6
ECE_tail = 3
LineC = 0
```

纳入 v14.5 oracle：

```text
out_dir = results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_micro_pulse_curvature_action
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

判断：

```text
micro-pulse curvature action 没有补上 Fashion-MNIST seed2 或 KMNIST seed2；
extended oracle 仍停在 7/9。
```

### 20.2 low-lr micro-pulse curvature action

触发原因：

```text
micro-pulse lr=0.005 没有打开 AUCtime blocker。
此前 low-lr repair 能让 mean source 转正，但没有配合末端/曲率微脉冲。
因此做一个全 3x3 统一 lr=0.003 的合法探针；
不做 dataset-name branch，不做 seed-specific scaling。
```

执行规模：

```text
out_dir = results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_micro_pulse_curvature_lowlr_action_family
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K-AUC3,K-AUC4,K-CURV1,KCTRL
train_steps = 200
batch_size = 32
lr = 0.003
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.25
linec_mode = exact
compute_budgeted_run = 1
```

route：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 3 / 9
real_short_run_pass_rows = 5
mean_source_vs_best_control_noncontrol = 0.03533118963241577
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|---:|
| K-AUC3-FinalPulseTailTrust | 1 | 1 | 0.051515 | 0.001584 | 1.014606 | 8 |
| K-AUC4-FinalPulseIdentityProjection | 2 | 2 | 0.015679 | 0.029783 | 1.055052 | 8 |
| K-CURV1-CurvatureSafeBasisFreeFMS | 2 | 2 | 0.038800 | 0.075406 | 1.020707 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | K-AUC3 | 0.098271 | 0.970760 | -1.928946 | -0.109751 | -0.024200 | 1 |
| MNIST | 1 | K-AUC4 | 0.053912 | 0.932704 | -0.168503 | -0.065392 | -0.004021 | 1 |
| MNIST | 1 | K-CURV1 | 0.091432 | 0.992846 | -0.837555 | -0.102912 | -0.012476 | 1 |
| Fashion-MNIST | 1 | K-AUC4 | 0.107411 | 0.840245 | -0.847453 | -0.107411 | 0.011227 | 1 |
| KMNIST | 1 | K-CURV1 | 0.111598 | 0.953133 | -0.115360 | -0.196210 | -0.015661 | 1 |

failure counts（non-control rows）：

```text
AUCtime = 20
CEp99_tail = 13
source = 9
NLL_tail = 6
LineC = 2
ECE_tail = 1
```

纳入 v14.5 oracle：

```text
out_dir = results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_micro_pulse_curvature_lowlr_action
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

仍未通过的 extended oracle rows：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

判断：

```text
1. low-lr micro-pulse curvature action 也没有打开 S5。
2. 它让 mean source 转正，但 dataset-seed pass 仍为 3/9。
3. 纳入 extended oracle 后仍为 7/9。
4. controller 仍不能合法启动。
5. 不允许 promotion。
```

## 21. 当前最终判断更新

v14.5 仍未达成 S5-OfficialFunctionalSuccess。

主线状态仍为：

```text
route = R2-ActionBankUpperBoundInsufficient
best_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

本次新增覆盖：

```text
1. micro-pulse / curvature-safe low-lambda action：
   K-AUC3 / K-AUC4 / K-CURV1 with rt_lambda_max = 0.25。
2. low-lr micro-pulse / curvature-safe action：
   lr = 0.003, rt_lambda_max = 0.25。
```

新增闭合事实：

```text
1. 剩余 2 个 oracle failure 是 AUC-only：
   Fashion-MNIST seed2 与 KMNIST seed2。
2. 弱化末端 action 到 rt_lambda_max=0.25 没有补上缺口。
3. 低 lr + 弱末端 action 也没有补上缺口。
4. 新 action 纳入 extended oracle 后仍为 7/9。
5. 因此 controller 仍不能启动，promotion_allowed 仍为 0。
```

当前停止边界：

```text
我现在不确定如何在当前 v14.5 合法约束内继续安全推进到 S5。

已尝试的合法方向包括：
1. v14.5 Line O action-bank oracle。
2. source-tail-AUC colocation action。
3. final-pulse AUC action。
4. basis-free / curvature-safe action。
5. two-phase source-tail separation action。
6. micro-pulse / curvature low-lambda action。
7. low-lr micro-pulse / curvature action。
8. Line W Wavelet bounded substrate hardening。
9. Line D RBF / Chebyshev / Fourier all-basis substrate repair。

剩余 failure 不是 source/tail/LineC，而是 AUCtime-only。
继续推进需要新的 train-stream-only mechanism，
能够降低 AUC trajectory cost，同时不使用 AUCtime / LineC / CEp99 / NLL / ECE
作为 direction，不做 dataset-name branch，不做 seed-specific scaling，
也不拼接不同 run 的局部 positive。

在没有新机制之前，不能把 7/9 oracle 写成 S5，
不能启动 controller tuning，不能 promotion。
```

## 22. 用户再次要求继续后的 gradient-aligned trust-region action

本次继续推进一个新的 train-stream-only mechanism，不继续做 audit 阈值扫描。

触发原因：

```text
剩余 oracle failure 是 AUC-only。
已有 micro-pulse / low-lr micro-pulse 没有补上缺口。
因此新增一个 train-gradient trust-region：
只允许 FMS value direction 在当前 train mini-batch 的 AdamW gradient 附近做小幅增量。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py

新增 methods：
  K-TR1-GradientAlignedTrustRegion
  K-TR2-LateGradientAlignedTrustRegion

新增函数：
  gradient_aligned_value_flat

机制：
  value = generic_gradient + lambda * align_gate * clipped_delta

其中：
  generic_gradient 来自当前 train mini-batch。
  fms_delta 来自 FMS state 生成的 train-stream direction。
  align_gate 来自 generic_gradient 与 FMS direction 的 cosine。
  clipped_delta 按当前 train-gradient norm 限制。
```

合法性：

```text
1. 不使用 validation / test / future / query batch。
2. 不使用 AUCtime / LineC / CEp99 / NLL / ECE 作为 direction。
3. 不做 dataset-name branch。
4. 不做 seed-specific scaling。
5. AUCtime 仍只作为 final audit / gate。
```

语法检查：

```text
py_compile pass
```

### 22.1 gradient-aligned trust-region 结果

执行规模：

```text
out_dir = results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_gradient_aligned_trust_region_action_family
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K-TR1,K-TR2,KCTRL
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
route = R2-RealTransferFail
real_dataset_seed_pass_count = 4 / 9
real_short_run_pass_rows = 6
mean_source_vs_best_control_noncontrol = 0.01863147152794732
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|---:|
| K-TR1-GradientAlignedTrustRegion | 2 | 2 | -0.022090 | 0.059831 | 1.058266 | 9 |
| K-TR2-LateGradientAlignedTrustRegion | 4 | 4 | 0.059353 | 0.006297 | 0.968225 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | K-TR1 | 0.223864 | 0.913624 | -4.776531 | -0.237126 | -0.028701 | 1 |
| MNIST | 1 | K-TR2 | 0.252104 | 0.856932 | -5.594723 | -0.265367 | -0.030127 | 1 |
| MNIST | 2 | K-TR2 | 0.214375 | 0.960684 | -2.075740 | -0.281305 | -0.040065 | 1 |
| KMNIST | 0 | K-TR2 | 0.201613 | 0.968225 | -6.414253 | -0.201613 | 0.016415 | 1 |
| KMNIST | 1 | K-TR1 | 0.162151 | 0.972546 | -3.030199 | -0.653782 | -0.041453 | 1 |
| KMNIST | 1 | K-TR2 | 0.006297 | 0.938703 | -1.484419 | -0.497927 | -0.044757 | 1 |

failure counts（non-control rows）：

```text
AUCtime = 8
source = 7
CEp99_tail = 7
ECE_tail = 4
NLL_tail = 2
LineC = 0
```

判断：

```text
1. K-TR2 有真实 AUCtime 改善信号：
   median_AUCtime = 0.968225。
2. 但 dataset-seed pass 只有 4/9。
3. 它没有补上 Fashion-MNIST seed2 或 KMNIST seed2。
4. 不能写成 S5。
```

### 22.2 纳入 v14.5 oracle

纳入后的 route：

```text
out_dir = results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_gradient_aligned_trust_region_action
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
controller_not_executed_reason = oracle_action_bank_upper_bound_lt_9
official_s5_reached = 0
promotion_allowed = 0
```

仍未通过的 extended oracle rows：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

最终判断：

```text
1. Gradient-aligned trust-region action 没有打开 S5。
2. K-TR2 降低了本 run 的 AUCtime failure 数量，
   但没有补上剩余 oracle 缺口。
3. extended oracle 仍为 7/9。
4. controller 仍不能合法启动。
5. promotion_allowed 仍为 0。
```

## 23. 当前最终判断更新

v14.5 仍未达成 S5-OfficialFunctionalSuccess。

当前主线：

```text
route = R2-ActionBankUpperBoundInsufficient
best_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

新增覆盖：

```text
1. K-TR1-GradientAlignedTrustRegion。
2. K-TR2-LateGradientAlignedTrustRegion。
```

新增闭合事实：

```text
1. train-gradient trust-region 能减少 AUCtime failure，
   说明 trajectory-preserving action 是有信号的。
2. 但该信号没有转化为 action-bank upper bound 提升。
3. Fashion-MNIST seed2 与 KMNIST seed2 仍为 AUC-only blocker。
4. 在 action bank upper bound < 9/9 时，controller 不能合法启动。
```

当前停止边界：

```text
我现在不确定如何在当前 v14.5 合法约束内继续安全推进到 S5。

已经尝试的 train-stream-only AUC/trajectory 修复包括：
source-tail-AUC colocation、final pulse、basis-free / curvature-safe、
two-phase source-tail separation、micro-pulse、low-lr micro-pulse、
gradient-aligned trust region。

它们没有补上 Fashion-MNIST seed2 与 KMNIST seed2。
继续推进需要新的机制，而不是继续调已有 action 的阈值；
否则容易变成 AUCtime audit-tuned scaling。

不能使用 AUCtime / LineC / CEp99 / NLL / ECE 作为 direction，
不能 dataset-name branch，不能 seed-specific scaling，
不能拼接不同 run 的局部 positive，
不能把 7/9 oracle 写成 S5。
```

## 24. 用户再次要求继续后的 train-batch descent filter action

本次继续推进一个更直接的 train-stream counterfactual acceptance：

```text
K-CF1-TrainBatchDescentFilter
K-CF2-LateTrainBatchDescentFilter
```

机制：

```text
1. 先生成 FMS candidate gradient。
2. 用当前 train mini-batch 的普通 AdamW gradient 计算一阶下降一致性：
   descent_ratio = dot(generic_grad, candidate_grad) / ||generic_grad||^2。
3. 若 candidate 对当前 train loss 的一阶下降不可靠，则混回 AdamW gradient。
4. 若 candidate 范数过大，则按当前 train-gradient norm 截断。
```

合法性：

```text
1. direction 只使用当前 train mini-batch gradient / FMS state / train proxy。
2. 不使用 validation / test / future / query batch。
3. 不使用 AUCtime / LineC / CEp99 / NLL / ECE 作为 direction。
4. 不做 dataset-name branch。
5. 不做 seed-specific scaling。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py

新增：
  K-CF1-TrainBatchDescentFilter
  K-CF2-LateTrainBatchDescentFilter
  train_batch_descent_filtered_grad
```

语法检查：

```text
py_compile pass
```

### 24.1 train-batch descent filter 结果

执行规模：

```text
out_dir = results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_train_batch_descent_filter_action_family
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K-CF1,K-CF2,KCTRL
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
route = R2-RealTransferFail
real_dataset_seed_pass_count = 2 / 9
real_short_run_pass_rows = 3
mean_source_vs_best_control_noncontrol = -0.0806241167916192
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|---:|
| K-CF1-TrainBatchDescentFilter | 2 | 2 | -0.055043 | -0.033559 | 1.055231 | 8 |
| K-CF2-LateTrainBatchDescentFilter | 1 | 1 | -0.106205 | 0.013662 | 1.012529 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-CF1 | 0.204176 | 0.943531 | -3.758885 | -0.315636 | -0.011722 | 1 |
| MNIST | 1 | K-CF1 | 0.226195 | 0.896773 | -4.564541 | -0.239457 | -0.034714 | 1 |
| MNIST | 1 | K-CF2 | 0.301522 | 0.925878 | -8.713699 | -0.314784 | -0.021352 | 1 |

failure counts（non-control rows）：

```text
AUCtime = 12
CEp99_tail = 10
source = 10
NLL_tail = 8
ECE_tail = 3
LineC = 1
```

判断：

```text
1. K-CF1 / K-CF2 没有打开 S5。
2. 单 run 从 K-TR 的 4/9 退到 2/9。
3. 说明这个 train-batch descent filter 过度保守或破坏了 source/tail 同位。
```

### 24.2 纳入 v14.5 oracle

纳入后的 route：

```text
out_dir = results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_train_batch_descent_filter_action
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
controller_not_executed_reason = oracle_action_bank_upper_bound_lt_9
official_s5_reached = 0
promotion_allowed = 0
```

仍未通过的 extended oracle rows：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

最终判断：

```text
1. Train-batch descent filter 没有打开 S5。
2. 纳入 extended oracle 后仍为 7/9。
3. controller 仍不能合法启动。
4. 不允许 promotion。
```

## 25. 当前最终判断更新

v14.5 仍未达成 S5-OfficialFunctionalSuccess。

当前主线：

```text
route = R2-ActionBankUpperBoundInsufficient
best_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

新增覆盖：

```text
1. K-CF1-TrainBatchDescentFilter。
2. K-CF2-LateTrainBatchDescentFilter。
```

新增闭合事实：

```text
1. 更直接的 train-batch counterfactual descent filter 没有提升 action bank upper bound。
2. 它比 gradient-aligned trust-region 更差，只有 2/9。
3. Fashion-MNIST seed2 与 KMNIST seed2 仍为 AUC-only blocker。
4. action bank upper bound 仍为 7/9，因此 controller 不能启动。
```

当前停止边界：

```text
我现在不确定如何在当前 v14.5 合法约束内继续安全推进到 S5。

已经尝试的 train-stream-only trajectory / AUC 修复包括：
source-tail-AUC colocation、final pulse、basis-free / curvature-safe、
two-phase source-tail separation、micro-pulse、low-lr micro-pulse、
gradient-aligned trust region、train-batch descent filter。

它们没有补上 Fashion-MNIST seed2 与 KMNIST seed2。
继续推进需要新的机制，而不是继续调已有 action 的阈值；
否则容易变成 AUCtime audit-tuned scaling。

不能使用 AUCtime / LineC / CEp99 / NLL / ECE 作为 direction，
不能 dataset-name branch，不能 seed-specific scaling，
不能拼接不同 run 的局部 positive，
不能把 7/9 oracle 写成 S5。
```

## 26. 用户再次要求继续后的 front-loaded trajectory assist action

用户再次要求“未达成则继续”。本次在不使用 AUCtime / LineC / CEp99 / NLL / ECE 作为 direction 的前提下，继续尝试一个新的 train-stream-only 机制：

```text
K-FL1-FrontLoadedTrajectoryAssist
K-FL2-FrontLoadedGradientAlignedAssist
```

触发原因：

```text
1. 当前 extended legal v144 action-bank oracle 仍为 7/9。
2. 剩余失败是 Fashion-MNIST seed2 与 KMNIST seed2 的 AUC-only blocker。
3. final pulse / late ramp / micro-pulse 都没有降低这两个 row 的 AUCtime。
4. 因此尝试相反方向：只在训练前半程给 FMS trajectory assist，后半程回到 AdamW，
   检查是否能减少 late functional action 带来的 AUC overhead。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py

新增：
  K-FL1-FrontLoadedTrajectoryAssist
  K-FL2-FrontLoadedGradientAlignedAssist

主要逻辑：
  K-FL1 使用 train step phase warmup/cooldown + train-stream source/tail trust。
  K-FL2 在 K-FL1 基础上加入 train-gradient alignment 半径限制。
```

合法性：

```text
1. direction 只使用 train step phase / train proxy / FMS state / train gradient。
2. 不使用 AUCtime / LineC / CEp99 / NLL / ECE 作为 direction。
3. 不使用 validation / test / future / query batch。
4. 不做 dataset-name branch。
5. 不做 seed-specific scaling。
6. 不拼接不同 run 的局部 positive。
```

语法检查：

```text
py_compile pass
```

### 26.1 front-loaded trajectory action 结果

执行规模：

```text
out_dir = results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_frontloaded_trajectory_action_family
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0,K-FL1,K-FL2,KCTRL
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
route = R2-RealTransferFail
real_dataset_seed_pass_count = 4 / 9
real_short_run_pass_rows = 6
mean_source_vs_best_control_noncontrol = 0.00047895643446180556
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|---:|
| K-FL1-FrontLoadedTrajectoryAssist | 2 | 2 | -0.027825 | -0.008797 | 1.028604 | 9 |
| K-FL2-FrontLoadedGradientAlignedAssist | 4 | 4 | 0.028783 | 0.073137 | 1.038703 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | K-FL1 | 0.219128 | 0.916936 | -6.472919 | -0.232391 | -0.021093 | 1 |
| MNIST | 1 | K-FL2 | 0.259067 | 0.994922 | -6.384107 | -0.272330 | -0.024174 | 1 |
| MNIST | 2 | K-FL1 | 0.182549 | 0.894147 | -2.439426 | -0.249479 | -0.036731 | 1 |
| MNIST | 2 | K-FL2 | 0.059128 | 0.953285 | -0.997680 | -0.126058 | -0.022846 | 1 |
| Fashion-MNIST | 1 | K-FL2 | 0.140299 | 0.880113 | -0.982815 | -0.160171 | 0.006390 | 1 |
| KMNIST | 0 | K-FL2 | 0.225074 | 0.962043 | -0.032822 | -0.225074 | 0.014608 | 1 |

failure counts（non-control rows）：

```text
AUCtime = 12
CEp99_tail = 7
source = 7
NLL_tail = 3
ECE_tail = 1
LineC = 0
```

判断：

```text
1. Front-loaded trajectory assist 没有打开 S5。
2. K-FL2 单方法 4/9，但没有超过 K-TR，也没有补上剩余 oracle 缺口。
3. K-FL1 / K-FL2 的 LineC 均为 9/9，说明 LineC 不是本轮主要 blocker。
4. 主要 blocker 仍是 AUCtime 与 source/tail 的同位失败。
```

### 26.2 纳入 v14.5 oracle

纳入后的 route：

```text
out_dir = results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_frontloaded_trajectory_action
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
best_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
controller_not_executed_reason = oracle_action_bank_upper_bound_lt_9
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

仍未通过的 extended oracle rows：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

最终判断：

```text
1. K-FL1 / K-FL2 没有打开 S5。
2. 纳入 extended oracle 后仍为 7/9。
3. front-loaded 机制没有补上 Fashion-MNIST seed2 / KMNIST seed2。
4. action bank upper bound 仍不足 9/9，因此 controller 仍不能启动。
5. 不允许 promotion。
```

## 27. 当前最终判断更新

v14.5 仍未达成 S5-OfficialFunctionalSuccess。

当前主线：

```text
route = R2-ActionBankUpperBoundInsufficient
best_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

新增覆盖：

```text
1. K-FL1-FrontLoadedTrajectoryAssist。
2. K-FL2-FrontLoadedGradientAlignedAssist。
```

新增闭合事实：

```text
1. front-loaded trajectory assist 没有提升 action bank upper bound。
2. K-FL2 单 run 为 4/9，不超过 gradient-aligned trust-region。
3. Fashion-MNIST seed2 与 KMNIST seed2 仍为 AUC-only blocker。
4. action bank upper bound 仍为 7/9，因此 controller 不能启动。
```

当前停止边界：

```text
我现在不确定如何在当前 v14.5 合法约束内继续安全推进到 S5。

已经尝试的 train-stream-only trajectory / AUC 修复包括：
source-tail-AUC colocation、final pulse、basis-free / curvature-safe、
two-phase source-tail separation、micro-pulse、low-lr micro-pulse、
gradient-aligned trust region、train-batch descent filter、
front-loaded trajectory assist。

它们没有补上 Fashion-MNIST seed2 与 KMNIST seed2。
继续推进需要新的机制，而不是继续调已有 action 的阈值；
否则容易变成 AUCtime audit-tuned scaling。

不能使用 AUCtime / LineC / CEp99 / NLL / ECE 作为 direction，
不能 dataset-name branch，不能 seed-specific scaling，
不能拼接不同 run 的局部 positive，
不能把 7/9 oracle 写成 S5。
```
