# DG-KAN v13.11 CoverObjectiveValidation CoverPreservingSubstrate 实验结果复盘

生成时间：2026-05-29（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 oracle diagnostic / smoke / focused repeat / Non-RAT vertical slice 写成 promotion。

## 1. 计划理解

v13.11 的第一目标不是继续 v13.10 的 A-RCF/K14-K18 architecture sweep，而是先验证：

```text
当前 cover_purity / cover_churn / cover_specialization objective 是否真能预测 future advantage。
```

计划关键分支：

```text
1. 如果 Line V oracle cover objective invalid：
   route = R2-CoverObjectiveInvalid；
   停止 A-CPF/Line K 大规模 architecture search；
   输出 objective reset recommendation。
2. 如果 Line V valid：
   才允许进入 A-CPF1..A-CPF6 cover-preserving substrate architecture 与 K19/K20 等训练。
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional / optimizer direction 不使用 validation/test/future/query。
5. V1/V2/V3/V4 oracle cover 可以使用 future/LineC/task-family/label，但只能 diagnostic。
6. LineC / CEp99 / NLL / ECE 只能作为 audit/gate，不能生成可 promotion direction。
7. MLP row 只能作为 generic optimizer / cover control，不写成 KAN promotion。
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
```

主要修改：

```text
1. 新增 Line V cover objective diagnostic runner。
2. 实现 V0/V1/V2/V3/V4/V5/V6：
   V0-AdamWControl
   V1-FutureGradientClusterOracle
   V2-LineCSignalReservoirOracle
   V3-TaskFamilyOracle
   V4-LabelClassCentroidOracle
   V5-RandomCoverControl
   V6-PermutedOracleControl
3. oracle rows 全部 promotion_allowed=0，且写入 cover_objective_provenance。
4. 若 Line V invalid，A-CPF1..A-CPF6 与 Line K fail-closed skip，不执行 architecture sweep。
5. 输出 v1311 required artifact surface、figures、no-go boundary 与 next queue。
6. Non-RAT vertical slice 只读取既有 substrate prior，并按 v13.11 gate fail-closed。
7. MLP analog 复用 generic monitor，只作为 control。
```

合法性说明：

```text
1. oracle diagnostic rows 不允许 promotion。
2. V1 使用 future gradient cluster，仅用于 objective upper-bound diagnostic。
3. V2 使用 LineC signal/reservoir，仅用于 objective upper-bound diagnostic。
4. V3/V4 使用 task-family/label，仅用于 objective upper-bound diagnostic。
5. A-CPF architecture search 因 Line V invalid 被计划性停止，不是漏跑。
6. 不降低 S3/S4/promotion gate。
```

## 3. 初始语法检查

执行：

```text
conda run -n kan python -m py_compile experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
```

结果：

```text
py_compile pass
```

## 4. Smoke

执行规模：

```text
task = X1
seed = 0
loss = CE
methods = V0,V1,V5,V6
train_steps = 4
batch_size = 16
```

第一次 smoke 发现 artifact 顺序 blocker：

```text
v1311_required_manifest.csv 被列入 required manifest；
manifest 检查发生在该 manifest 写入前，导致 required_artifact_missing_count=1。
```

修复：

```text
experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
  写出 required_manifest 后再次执行 required_manifest()；
  回填 route.required_artifact_missing_count 并重写 route。
```

重跑 smoke 结果：

```text
route = R2-CoverObjectiveInvalid
minimum_success = S0-ObjectiveDiagnosticExecuted
cover_objective_valid = 0
oracle_rows = 4
oracle_best_task_family_pass_count = 0
oracle_best_source_vs_best_control = 0.360177747701133
random_permuted_control_task_pass_count = 0
cover_preserving_substrate_pass_count = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
nonrat_vertical_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

解释：

```text
smoke 只证明 runner 与 artifact surface 可执行；
不能写成 objective no-go 的 official 证据。
```

## 5. official_v1311 Line V

执行规模：

```text
synthetic_tasks = X1..X7
synthetic_seeds = 0
loss_interfaces = CE,Brier
methods = V0,V1,V2,V3,V4,V5,V6
train_steps = 60
batch_size = 32
datasets = MNIST,Fashion-MNIST,KMNIST
mlp_seeds = 0,1
compute_budgeted_run = 1
```

最终 route：

```text
route = R2-CoverObjectiveInvalid
minimum_success = S0-ObjectiveDiagnosticExecuted
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
cover_objective_valid = 0
oracle_rows = 98
oracle_valid_candidate_count = 0
oracle_best_task_family_pass_count = 1
oracle_best_source_vs_best_control = 0.23338876249709983
random_permuted_control_task_pass_count = 0
cover_preserving_substrate_pass_count = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
nonrat_vertical_pass_count = 0
mlp_cover_dataset_pass_count = 0 / 3
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

Line V summary：

| method | task-family pass | max source vs best control | median source vs best control | median cover purity | median AUC ratio | objective valid |
|---|---:|---:|---:|---:|---:|---:|
| V0-AdamWControl | 0 | 0.0 | -0.011765102855861187 | 0.0 | 1.0076165199279785 | 0 |
| V1-FutureGradientClusterOracle | 0 | 0.015516028988649788 | -0.18799152970314026 | 0.9424925446510315 | 1.1996362209320068 | 0 |
| V2-LineCSignalReservoirOracle | 0 | 0.13602547477413918 | -0.16459329426288605 | 0.9134664535522461 | 1.1178603172302246 | 0 |
| V3-TaskFamilyOracle | 1 | 0.23338876249709983 | -0.19553831219673157 | 0.9213893413543701 | 1.219175100326538 | 0 |
| V4-LabelClassCentroidOracle | 0 | 0.13502636715069394 | -0.11040696501731873 | 0.5518028736114502 | 1.1139154434204102 | 0 |
| V5-RandomCoverControl | 0 | 0.0 | -0.17921516299247742 | 0.8391478061676025 | 1.252002477645874 | 0 |
| V6-PermutedOracleControl | 0 | 0.0 | -0.14925554394721985 | 0.9382239580154419 | 1.131025791168213 | 0 |

唯一 pass row：

| method | task | seed | loss | source_vs_best | AUC ratio | CEp99 delta | NLL delta | ECE delta | Noise delta | Reservoir delta | cover purity |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V3-TaskFamilyOracle | X5 | 0 | CE | 0.23338876249709983 | 0.8546958097786558 | -1.1041901111602783 | -0.2325425148010254 | -0.08042445778846741 | -0.0013203024864196777 | -0.2802622318267822 | 0.4876159736295392 |

判断：

```text
1. V1/V2/V3 没有任何一种达到 >=5/7 task-family pass。
2. 最好的 V3 只有 1/7。
3. 很多 oracle rows 的 cover_purity 很高，但 AUC ratio median > 1，source median 为负。
4. random/permuted control 没过，但 oracle 也没过；因此不是 legal cover formation blocker，而是当前 cover objective upper-bound 不成立。
```

## 6. Line V seed1 repeat

触发原因：

```text
official 有一个 V3/X5/CE 局部 pass row；
为了避免单 seed 偶然性，把 Line V 换 seed=1 重复。
```

执行规模：

```text
synthetic_tasks = X1..X7
synthetic_seeds = 1
loss_interfaces = CE,Brier
methods = V0,V1,V2,V3,V4,V5,V6
train_steps = 40
batch_size = 32
compute_budgeted_run = 1
```

结果：

```text
route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_rows = 98
oracle_valid_candidate_count = 0
oracle_best_task_family_pass_count = 1
oracle_best_source_vs_best_control = 0.14723053126705488
random_permuted_control_task_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

Line V repeat summary：

| method | task-family pass | max source vs best control | median source vs best control | median cover purity | median AUC ratio |
|---|---:|---:|---:|---:|---:|
| V1-FutureGradientClusterOracle | 1 | 0.14723053126705488 | -0.2414291352033615 | 0.9298206567764282 | 1.2692384719848633 |
| V2-LineCSignalReservoirOracle | 0 | 0.09364360944044425 | -0.2757589817047119 | 0.9075677394866943 | 1.2876430749893188 |
| V3-TaskFamilyOracle | 0 | -0.0376215664484183 | -0.3289731740951538 | 0.9347505569458008 | 1.345031976699829 |
| V4-LabelClassCentroidOracle | 0 | 0.04160635789850553 | -0.15294960141181946 | 0.4603429436683655 | 1.181062936782837 |

判断：

```text
seed1 repeat 仍然是 1/7，不支持 objective valid。
```

## 7. A-CPF / Line K 处理

因为 Line V invalid，按计划第 16.1 节：

```text
route = R2-CoverObjectiveInvalid
stop Line A/K large-scale run
next = redefine cover objective
```

因此本轮没有执行 A-CPF1..A-CPF6 大规模 architecture search。artifact 中写入 fail-closed rows：

```text
v1311_cover_preserving_substrate.csv
v1311_cover_identity_telemetry.csv
v1311_signal_to_cover_training.csv
v1311_signal_to_cover_summary.csv
```

审计边界：

```text
这不是漏跑；
这是计划要求的 objective-invalid 分支。
继续 A-CPF sweep 会违反计划，并且会把未验证 objective 当成优化目标。
```

## 8. Non-RAT vertical slice 与 MLP analog

Non-RAT：

```text
v1311_nonrat_vertical_slice rows = 6
nonrat_vertical_pass_count = 0
```

逐候选结果：

| family | candidate | mapped | raw ratio | incremental ratio | step ratio | status | pass |
|---|---|---|---:|---:|---:|---|---:|
| D-FOU | N-FOU-CPF1 | D-FOU20 | 9.0 | 9.0 | 9.0 | WorkspaceFail | 0 |
| D-FOU | N-FOU-CPF2 | D-FOU20 | 9.0 | 9.0 | 9.0 | WorkspaceFail | 0 |
| D-CHE | N-CHE-CPF1 | D-CHE20 | 9.0 | 9.0 | 9.0 | WorkspaceFail | 0 |
| D-CHE | N-CHE-CPF2 | D-CHE17 | 1.4106476848903848 | 13.232558139534884 | 1.878934745131025 | WorkspaceFail | 0 |
| D-RBF | N-RBF-CPF1 | D-RBF17 | 9.0 | 9.0 | 9.0 | WorkspaceFail | 0 |
| D-WAV | N-WAV-CPF1 | D-WAV16 | 9.0 | 9.0 | 9.0 | WorkspaceFail | 0 |

MLP analog：

```text
mlp_cover_dataset_count = 3
mlp_cover_dataset_pass_count = 0
```

解释：

```text
Non-RAT 与 MLP 均不能作为 KAN promotion；
本轮主结论仍由 Line V objective validity 决定。
```

## 9. Required artifacts

official_v1311 required manifest：

```text
manifest_rows = 32
missing_required_rows = 0
```

主要产物：

```text
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_route_decision.json
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_progress_table.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_code_review_manifest.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_architecture_diff_manifest.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_cover_objective_provenance.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_cover_objective_validity.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_oracle_cover_results.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_random_permuted_cover_controls.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_cover_preserving_substrate.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_cover_identity_telemetry.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_signal_to_cover_training.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_signal_to_cover_summary.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_nonrat_vertical_slice.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_mlp_cover_analog.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_linec_audit.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_failure_table.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_no_go_boundary.md
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_next_hypothesis_queue.md
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_required_manifest.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_code_review_packet.zip
```

repair / repeat artifacts：

```text
results/v13_11_cover_objective_validation_cover_preserving_substrate/smoke_v1311/
results/v13_11_cover_objective_validation_cover_preserving_substrate/linev_repeat_seed1_v1311/
```

## 10. 最终科学结论

v13.11 没有达成 S1/S2/S3/S4/S5；最终合法 official route：

```text
R2-CoverObjectiveInvalid
minimum_success = S0-ObjectiveDiagnosticExecuted
promotion_allowed = 0
official_success_reached = 0
```

已闭合事实：

```text
1. Line V oracle cover validity 已执行：V1/V2/V3/V4 + random/permuted controls。
2. official oracle rows = 98。
3. 最好的 oracle 只有 1/7 task-family pass，低于计划要求的 >=5/7。
4. seed1 repeat 仍只有 1/7。
5. 高 cover_purity oracle rows 没有稳定带来 source/AUC/LineC/tail non-harm。
6. random/permuted controls 没过，但 oracle 也没过，所以不能证明当前 cover objective 有价值。
7. A-CPF/Line K 按计划 fail-closed stop；不是漏跑。
8. Non-RAT vertical pass = 0。
9. MLP cover analog dataset pass = 0/3。
10. required artifacts 缺失为 0，forbidden information violation 为 0。
```

no-go boundary：

```text
1. v13.11 证明当前 cover_purity/churn/specialization objective 还没有 oracle upper-bound 支撑。
2. 即使用 future/LineC/task-family/label diagnostic cover，高 cover purity 也没有稳定转化为 future advantage。
3. 因此继续 A-CPF architecture search 会把未验证 objective 当成优化目标，计划禁止。
4. 下一步应重定义 cover objective，例如 value-cover / future-advantage-correlated cover，而不是继续微调 cover-preserving substrate。
```

最终判断：

```text
v13.11 未达成目标；
不允许 promotion；
不允许 real short-run；
允许 final stop，原因是 Line V objective invalid 后计划明确停止 A-CPF/Line K 大规模搜索；
本轮已输出 objective-reset recommendations、Non-RAT vertical slice、MLP analog monitor、no-go boundary 与 next queue。
```

## 11. 用户再次追问后的 objective-reset 诊断

用户再次要求未达成则继续。本次先复核 official route 与计划 stop policy：

```text
official route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_best_task_family_pass_count = 1
promotion_allowed = 0
final_stop_allowed = 1
```

计划第 16.1 节明确：

```text
route = R2-CoverObjectiveInvalid
stop Line A/K large-scale run
next = redefine cover objective
No further A-CPF architecture search is allowed until objective is redefined.
```

因此本次没有继续 A-CPF / Line K substrate search，而是增加一个 post-hoc objective-reset alignment 诊断脚本：

```text
experiments/analyze_v1311_objective_reset_alignment.py
```

该脚本只读取既有 Line V official 与 seed1 repeat artifact，计算当前 cover metrics 与 source_vs_best_control 的 Pearson / Spearman 相关、top/bottom quartile source median、value-positive rows；不新增训练，不生成方向，不允许 promotion。

输出：

```text
results/v13_11_cover_objective_validation_cover_preserving_substrate/objective_reset_diagnostic_v1311/v1311_objective_reset_route.json
results/v13_11_cover_objective_validation_cover_preserving_substrate/objective_reset_diagnostic_v1311/v1311_objective_reset_metric_alignment.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/objective_reset_diagnostic_v1311/v1311_objective_reset_method_summary.csv
```

结果：

```text
diagnostic_route = D1-ObjectiveResetNeeded_CurrentCoverMetricsWeakValueAlignment
official_route_unchanged = R2-CoverObjectiveInvalid
uses_existing_artifacts_only = 1
new_training_executed = 0
line_a_k_architecture_search_executed = 0
promotion_allowed = 0
cover_metric_alignment_pass = 0
combined_rows = 168
combined_oracle_rows = 112
combined_control_rows = 56
combined_task_family_pass_rows = 2
combined_value_positive_rows = 3
official_cover_purity_spearman_oracle = -0.16657552973342446
repeat_cover_purity_spearman_oracle = -0.08448393711551606
combined_cover_purity_spearman_oracle = -0.10525506543205658
combined_signal_to_cover_spearman_oracle = -0.21369631325383537
```

关键对齐表：

| scope | metric | Spearman vs source | top quartile source median | bottom quartile source median |
|---|---|---:|---:|---:|
| official oracle | cover_purity | -0.16657552973342446 | -0.1898157915506301 | -0.10712267434717754 |
| official oracle | signal_to_cover_score | -0.3066985645933014 | -0.21733573263550154 | -0.0301754297464103 |
| seed1 repeat oracle | cover_purity | -0.08448393711551606 | -0.2657414505807829 | -0.19105661718500971 |
| seed1 repeat oracle | signal_to_cover_score | -0.17682843472317156 | -0.2729760249372975 | -0.1192752595294535 |
| combined oracle | cover_purity | -0.10525506543205658 | -0.20345331882506346 | -0.12345115066328455 |
| combined oracle | signal_to_cover_score | -0.21369631325383537 | -0.21068789318157805 | -0.09228270035690683 |

combined method summary：

| method | rows | task-family pass rows | value-positive rows | max source | median source | median cover purity |
|---|---:|---:|---:|---:|---:|---:|
| V1-FutureGradientClusterOracle | 28 | 1 | 1 | 0.14723053126705488 | -0.21068789318157805 | 0.9359859495449906 |
| V2-LineCSignalReservoirOracle | 28 | 0 | 0 | 0.13602547477413918 | -0.17913361560529195 | 0.9112206237368022 |
| V3-TaskFamilyOracle | 28 | 1 | 1 | 0.23338876249709983 | -0.23515813043561434 | 0.9243645158357079 |
| V4-LabelClassCentroidOracle | 28 | 0 | 1 | 0.13502636715069394 | -0.12351184743694793 | 0.4684390423550432 |
| V5-RandomCoverControl | 28 | 0 | 0 | 0.0 | -0.21333231639389194 | 0.8018372211063911 |
| V6-PermutedOracleControl | 28 | 0 | 0 | 0.0 | -0.17466906263729692 | 0.9199462878515988 |

判断：

```text
1. 当前 cover_purity / signal_to_cover_score 与 source_vs_best_control 不但没有稳定正相关，反而在 official、seed1 repeat、combined 中均为负相关。
2. 高 cover_purity quartile 的 source median 比低 cover_purity quartile 更差。
3. combined 168 rows 中只有 3 个 value-positive rows，且 task-family pass rows 只有 2。
4. 这进一步支持 R2-CoverObjectiveInvalid，不支持继续当前 A-CPF / Line K substrate search。
5. 下一步应新开 value-cover / legal precommit train-stream proxy 目标定义，而不是在 v13.11 内把 architecture scout 编造成 success。
```

最终判断仍是：

```text
v13.11 未达成目标；
official route = R2-CoverObjectiveInvalid；
promotion_allowed = 0；
real_short_run_open_allowed = 0；
允许 final stop。
```

## 12. 用户再次追问后的 V7 train-stream value-cover proxy

用户再次要求未达成则继续。本次在不违反第 16.1 节的前提下，尝试将 “redefine cover objective” 具体化为一个新的 objective-reset diagnostic。

代码修改：

```text
experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
  新增 V7-TrainStreamValueCoverProxy。
  新增 train_stream_value_cover_mask()。
  新增 objective_reset_candidate / uses_train_stream_only_value_proxy 审计字段。
```

V7 语义：

```text
1. 使用当前 train batch 的 per-example gradients。
2. 按 group slice 计算 signal norm / gradient noise / per-example agreement。
3. 只用 train-stream parameter/group telemetry 选择 cover mask。
4. 不使用 validation/test/future/query batch 生成方向。
5. 不使用 LineC / CEp99 / NLL / ECE 生成方向。
6. 只作为 post-R2 objective-reset diagnostic，不纳入 v13.11 official oracle validity gate。
```

执行规模：

```text
out_dir = results/v13_11_cover_objective_validation_cover_preserving_substrate/value_proxy_v7_v1311
methods = V0,V5,V6,V7
synthetic_tasks = X1..X7
synthetic_seeds = 0,1
loss_interfaces = CE,Brier
train_steps = 60
batch_size = 32
datasets = MNIST
mlp_seeds = 0
compute_budgeted_run = 1
```

结果：

```text
route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_valid_candidate_count = 0
random_permuted_control_task_pass_count = 0
cover_preserving_substrate_pass_count = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
mlp_cover_dataset_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

validity summary：

| method | rows | task pass | control pass | median source | max source | median cover | median AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0-AdamWControl | 28 | 0 | 0 | 0.0 | 0.0 | 0.0 | 1.0 |
| V5-RandomCoverControl | 28 | 0 | 0 | -0.22668853402137756 | 0.0 | 0.880892813205719 | 1.231992483139038 |
| V6-PermutedOracleControl | 28 | 0 | 0 | -0.17707574367523193 | 0.0 | 0.9499045014381409 | 1.2111748456954956 |
| V7-TrainStreamValueCoverProxy | 28 | 0 | 0 | -0.31016838550567627 | 0.0736189368426392 | 0.9442763328552246 | 1.241366982460022 |

判断：

```text
1. V7 没有打开 task-family pass；task pass = 0。
2. V7 median source = -0.3102，median AUC ratio = 1.2414，比 controls 更差。
3. V7 median cover purity = 0.9443 很高，但没有转化成 source/value advantage。
4. 这进一步支持：高 cover purity 本身不是足够的 value objective。
5. 不允许 promotion，不允许 real short-run。
```

最终判断仍是：

```text
v13.11 没有达成目标；
official route 仍为 R2-CoverObjectiveInvalid；
V7 objective-reset diagnostic 也未打开有效 cover objective；
当前安全可执行的 v13.11 内部继续项已经闭合。
```

## 13. 用户再次追问后的 V8/V9 soft value-cover proxy

用户再次要求未达成则继续。由于 V7 是硬 mask，本次继续排除一个实现层 blocker：

```text
是否只是 hard cover 截断太激进，导致 source 下降？
```

代码修改：

```text
experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
  新增 V8-TrainStreamSoftValueCoverProxy。
  新增 V9-TrainStreamResidualSafeValueCoverProxy。
  train_stream_value_cover_mask() 增加 floor 参数。
```

方法语义：

```text
V8 = train-stream value-cover group score + soft floor 0.25。
V9 = train-stream value-cover group score + residual-safe floor 0.60。
```

合法性：

```text
1. 只使用当前 train batch per-example gradients / group slices。
2. 不使用 validation/test/future/query batch 生成方向。
3. 不使用 LineC / CEp99 / NLL / ECE 生成方向。
4. 只作为 post-R2 objective-reset diagnostic，不允许 promotion。
```

执行规模：

```text
out_dir = results/v13_11_cover_objective_validation_cover_preserving_substrate/value_proxy_v8v9_v1311
methods = V0,V5,V6,V8,V9
synthetic_tasks = X1..X7
synthetic_seeds = 0
loss_interfaces = CE,Brier
train_steps = 40
batch_size = 32
datasets = MNIST
mlp_seeds = 0
compute_budgeted_run = 1
```

结果：

```text
route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_valid_candidate_count = 0
random_permuted_control_task_pass_count = 0
cover_preserving_substrate_pass_count = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

validity summary：

| method | rows | task pass | control pass | median source | max source | median cover | median AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0-AdamWControl | 14 | 0 | 0 | -0.002191375009715557 | 0.0 | 0.0 | 1.002761960029602 |
| V5-RandomCoverControl | 14 | 0 | 0 | -0.17019987106323242 | 0.0 | 0.5056086778640747 | 1.2395715713500977 |
| V6-PermutedOracleControl | 14 | 0 | 0 | -0.14654985070228577 | 0.0 | 0.9624462127685547 | 1.1125260591506958 |
| V8-TrainStreamSoftValueCoverProxy | 14 | 0 | 0 | -0.24296048283576965 | -0.02572410368519984 | 0.9082890152931213 | 1.2377440929412842 |
| V9-TrainStreamResidualSafeValueCoverProxy | 14 | 0 | 0 | -0.2151571810245514 | -0.019695951689679747 | 0.9513776302337646 | 1.1870806217193604 |

判断：

```text
1. V8/V9 仍没有打开 task-family pass。
2. 即使保留 residual-safe soft floor，median source 仍为负，AUC ratio 仍明显 > 1。
3. V8/V9 的 max source 也为负，说明本轮没有局部 positive row。
4. 这排除了“只是 V7 hard mask 太激进”的直接解释。
5. v13.11 仍是 R2-CoverObjectiveInvalid，不允许 promotion。
```

最终判断仍是：

```text
v13.11 没有达成目标；
official route 仍为 R2-CoverObjectiveInvalid；
V7/V8/V9 objective-reset diagnostics 均未打开有效 cover objective；
当前 v13.11 内部可安全推进的 objective-reset diagnostic 已闭合。
```

## 14. 用户再次追问后的 V10 near-AdamW residual value-cover proxy

用户再次要求未达成则继续。V8/V9 仍然有明显 source 伤害；本次继续排除另一个实现层 blocker：

```text
是否只是所有 value-cover proxy 都太多削弱 AdamW 基线？
```

代码修改：

```text
experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
  新增 V10-TrainStreamNearAdamWValueCoverProxy。
  train_stream_value_cover_mask(..., floor=0.85)。
```

V10 语义：

```text
1. 保留 85% 全梯度，只做小幅 group reweight。
2. 只使用当前 train batch per-example gradients / group slices。
3. 不使用 validation/test/future/query batch 生成方向。
4. 不使用 LineC / CEp99 / NLL / ECE 生成方向。
5. 只作为 post-R2 objective-reset diagnostic，不允许 promotion。
```

执行规模：

```text
out_dir = results/v13_11_cover_objective_validation_cover_preserving_substrate/value_proxy_v10_v1311
methods = V0,V5,V6,V10
synthetic_tasks = X1..X7
synthetic_seeds = 0
loss_interfaces = CE,Brier
train_steps = 40
batch_size = 32
datasets = MNIST
mlp_seeds = 0
compute_budgeted_run = 1
```

结果：

```text
route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_valid_candidate_count = 0
random_permuted_control_task_pass_count = 0
cover_preserving_substrate_pass_count = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

validity summary：

| method | rows | task pass | control pass | median source | max source | median cover | median AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0-AdamWControl | 14 | 0 | 0 | -0.0017003349494189024 | 0.0 | 0.0 | 1.0021430253982544 |
| V5-RandomCoverControl | 14 | 0 | 0 | -0.17035727202892303 | 0.0 | 0.5056086778640747 | 1.2395497560501099 |
| V6-PermutedOracleControl | 14 | 0 | 0 | -0.14652642607688904 | 0.0 | 0.9624462127685547 | 1.1120707988739014 |
| V10-TrainStreamNearAdamWValueCoverProxy | 14 | 0 | 0 | -0.10909856855869293 | 0.14819857915185564 | 0.9789875745773315 | 1.1364250183105469 |

判断：

```text
1. V10 相比 V8/V9 减轻了 median source 伤害，但仍为负，median AUC ratio 仍 > 1。
2. V10 有局部 max source positive，但 task pass = 0；不能写成 objective valid。
3. 高 cover purity 仍没有稳定转化为 task-family value advantage。
4. 因此“只是 hard/soft mask 太激进”的解释不足。
5. v13.11 仍是 R2-CoverObjectiveInvalid，不允许 promotion。
```

最终判断仍是：

```text
v13.11 没有达成目标；
official route 仍为 R2-CoverObjectiveInvalid；
V7/V8/V9/V10 objective-reset diagnostics 均未打开有效 cover objective；
当前我已经不确定如何在 v13.11 代码路径内继续安全重定义 cover objective，
同时不把 compute-budgeted diagnostic / local positive row 写成 official success。
```
