# DG-KAN v13.3 TaskFamilyRobustBasisNaturalFunctional 实验结果复盘

生成时间：2026-05-27（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 smoke/diagnostic/fallback row 写成 promotion。

## 1. 计划理解

v13.3 的目标不是继续调 BN4/BN5/BN6/BN7 局部尺度，而是验证：

```text
task-family robust basis metric + true basis-parameter natural update
```

并且并行执行 Non-RAT substrate rescue。

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 teacher / distillation / loss modification / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 可以接收 generic loss-interface output cotangent，但不能 hardcode CE。
5. functional direction 不使用 validation/test/future outcome/query batch。
6. CEp99/NLL/ECE/LineC hard target 只能作为 audit / gate，不能生成方向。
7. blocker 后必须执行 Line G autopsy、Line M low-rank/block metric、Non-RAT rescue 与 MLP analog closure。
```

v13.2 已经证明真实 basis-param writeback 可执行，但只在 X7 打开局部 P3；v13.3 因此重点检查 diagonal/local metric 是否缺少 task-family robustness。

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v133_task_family_robust_basis_natural.py
```

核心修改：

```text
1. 从 v13.2 runner 派生，但 artifact / route / figure 名称改为 v13.3。
2. 新增 BM8-LowRankTangentNatural-r4。
3. 新增 BM9-LowRankTangentNatural-r8。
4. 新增 BM10-BlockGroupNatural。
5. 新增 BM11/BM12/BM13/BM14/BM15 metric fallback hooks。
6. low-rank metric 使用 train-stream per-example basis-parameter gradient 构造 D + U U^T + rho I，并用 Woodbury inverse。
7. 输出 task-family autopsy、lowrank metric rows、synthetic family summary、Non-RAT rescue taxonomy、progress table、no-go boundary。
8. synthetic 5/7 gate 改为 task-family pass：每个 X family 需要 >=2 substrates 或 >=2 seeds pass。
```

合法性说明：

```text
1. BM8/BM9/BM14/BM15 使用的是 train-stream gradient 与 generic loss-interface cotangent。
2. 不读取 validation/test/future outcome/query batch 生成方向。
3. 不使用 CEp99/NLL/ECE/LineC hard target 生成方向。
4. 写回真实 basis parameters，不是 readout-feature proxy。
5. 不改变 promotion gate。
```

## 3. 初始语法检查

执行：

```text
conda run -n kan python -m py_compile experiments/run_v133_task_family_robust_basis_natural.py
```

结果：

```text
py_compile pass
```

## 4. smoke

执行规模：

```text
tasks = X1,X7
seed = 0
selected S1 substrate = 1
loss = CE
updates = BM8,BM10
p3_rows = 4
```

结果：

```text
route = R2-BasisNaturalP3Failed
required_artifact_missing_count = 0
basis_natural_p3_rows = 4
basis_natural_p3_pass_count = 0
lowrank_metric_rows = 4
full_basis_param_update_rows = 4
```

解释：smoke 只证明 v13.3 runner、low-rank/block metric、真实参数写回、manifest 与 code packet 可执行；不能 promotion。

## 5. Non-RAT focused rescue

计划要求 Non-RAT 无 S1 时必须至少执行一次 lifetime repair 与 task-health repair。本轮执行：

```text
D-CHE12-LifetimeRecomputeBackward-K3
dataset = MNIST
seed = 0
train_size = 64
val_size = 32
workspace_warmup_steps = 2
workspace_profile_steps = 5
hardening_epochs = 1
```

结果：

```text
workspace_rows = 1
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 0
linec_executed = 0
skip_reason = workspace_gate_fail
raw_memory_ratio_vs_mlp = 1.1833379351891362
incremental_memory_ratio_vs_mlp = 2.83375104427736
step_ratio_vs_mlp = 1.271381889140255
```

解释：

```text
1. D-CHE12 focused rescue 没有打开 workspace gate。
2. incremental memory ratio 2.8337 > v13.3 S1 gate 的 2.00。
3. hardening/LineC 被显式 skip，不能写成 task-health pass。
4. Non-RAT rescue S1 仍为 0。
```

## 6. official_v133：BM8/BM9/BM10/BM14

执行规模：

```text
selected S1 family = D-RAT
top-3 S1 candidates = D-RAT26,D-RAT27,D-RAT28
synthetic tasks = X1..X7
seed = 0
loss interfaces = CE,Brier
updates = BM8-LowRankTangentNatural-r4,BM9-LowRankTangentNatural-r8,BM10-BlockGroupNatural,BM14-TrustRegionLowRankNatural
p3_rows = 168
```

最终 route：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_s2_healthy_base_count = 0
basis_natural_p3_rows = 168
basis_natural_p3_pass_count = 0
lowrank_metric_rows = 168
lowrank_metric_pass_count = 0
synthetic_rows = 21
synthetic_raw_success_rows = 0
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_rows = 5
nonrat_rescue_s1_count = 0
real_short_run_open_allowed = 0
real_short_run_pass_count = 0
mlp_analog_pass_count = 0
full_basis_param_update_rows = 168
provenance_violation_count = 0
```

最接近但不能 pass 的 rows：

| candidate | task | update | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | P3 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| D-RAT28 | X7 | BM8 | Brier | 0.005442331002339984 | 0.5388603210449219 | 0.025170743465423584 | -0.07901668548583984 | -2.0761308670043945 | 0 |
| D-RAT26 | X7 | BM8 | Brier | 0.005422350787306532 | 0.11803436279296875 | 0.026148438453674316 | -0.053554534912109375 | -2.3919496536254883 | 0 |
| D-RAT27 | X6 | BM8 | CE | 0.004644685275953492 | -40.17839813232422 | 0.044931888580322266 | 2.032320976257324 | -2.4059762954711914 | 0 |
| D-RAT28 | X2 | BM10 | CE | 0.004135613043332917 | 4.639118194580078 | -0.042606353759765625 | -0.5078606605529785 | -1.0987415313720703 | 0 |
| D-RAT26 | X3 | BM14 | CE | 0.0041230647752104055 | -1.1285181045532227 | 0.06325221061706543 | 0.27698826789855957 | -0.40990543365478516 | 0 |

主要 failure pattern：

| count | pattern |
|---:|---|
| 32 | source;coupling;noise;reservoir |
| 26 | source;coupling;noise;reservoir;cep99 |
| 21 | source;coupling;noise;reservoir;cep99;nll;ece |
| 19 | source;noise |
| 11 | source |

metric condition：

```text
min = 1.000007152557373
max = 1.0064374208450317
mean = 1.0010490204606737
```

解释：

```text
1. BM8/BM9/BM10/BM14 没有打开 P3。
2. 最接近的 X7 rows 虽然 source_vs_best >= 0.005，但 NoiseSignalLeak 坏化，因此被 gate 拒绝。
3. metric condition 很健康，失败不能简单归因于数值病态。
4. synthetic task-family pass = 0/7，real short-run gate 未打开。
```

## 7. fallback：BM11/BM12/BM13/BM15

因为 v13.3 no-go 规则写明 BM8-BM15，本轮补跑：

```text
updates = BM11-TaskFamilyBalancedMetric,BM12-LeaveFamilyOutMetric,BM13-KroneckerGroupNatural,BM15-ControlResidualizedNatural
p3_rows = 168
```

结果：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
basis_natural_p3_rows = 168
basis_natural_p3_pass_count = 0
lowrank_metric_pass_count = 0
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_s1_count = 0
real_short_run_open_allowed = 0
full_basis_param_update_rows = 168
required_artifact_missing_count = 0
```

最接近但不能 pass 的 rows：

| candidate | task | update | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | P3 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| D-RAT27 | X1 | BM12 | Brier | 0.007083189464571163 | -3.1536436080932617 | 0.03563427925109863 | 0.47486209869384766 | 0.8770847320556641 | 0 |
| D-RAT27 | X3 | BM12 | Brier | 0.006085850495908096 | -5.999052047729492 | 0.04862922430038452 | 0.7695038318634033 | 0.9988770484924316 | 0 |
| D-RAT28 | X4 | BM13 | CE | 0.004701264841142541 | 1.6782569885253906 | 0.023546040058135986 | -0.026034832000732422 | 0.5170464515686035 | 0 |
| D-RAT28 | X4 | BM12 | CE | 0.004082682960068834 | 5.4921417236328125 | 0.012078046798706055 | -0.43645668029785156 | -1.3066778182983398 | 0 |
| D-RAT26 | X5 | BM15 | CE | 0.003686780866397064 | -120.37969970703125 | 0.02828049659729004 | 4.910923004150391 | -0.2965354919433594 | 0 |

判断：

```text
BM11/BM12/BM13/BM15 也没有打开 P3；
没有达到 low-rank >=4/7 的 continue 条件；
不触发 leave-family-out targeted repair。
```

## 8. Required artifacts

official_v133 required manifest：

```text
manifest_rows = 36
missing_required_rows = 0
```

主要产物：

```text
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_route_decision.json
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_progress_table.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_code_review_manifest.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_basis_param_manifest.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_writeback_trace.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_loss_interface_audit.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_forbidden_info_audit.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_task_family_autopsy.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_gradient_snr_by_task.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_metric_condition_by_task.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_lowrank_metric_rows.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_synthetic_mechanism_v2.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_synthetic_family_summary.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_nonrat_substrate_rescue.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_mlp_analog_closure.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_real_short_run.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_failure_table.csv
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_no_go_boundary.md
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_next_hypothesis_queue.md
results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_code_review_packet.zip
```

## 9. 最终科学结论

v13.3 没有达成 S3/S5；最终合法 route：

```text
R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
promotion_allowed = 0
```

已闭合事实：

```text
1. S1 substrate 仍为 11，全部来自 D-RAT；S2 healthy base = 0。
2. BM8/BM9/BM10/BM14 official 共 168 rows，P3 pass = 0。
3. BM11/BM12/BM13/BM15 fallback 共 168 rows，P3 pass = 0。
4. synthetic task-family pass = 0/7；real short-run gate 未打开。
5. Non-RAT focused D-CHE12 rescue 没有打开 workspace gate；Non-RAT S1 rescue = 0。
6. MLP analog closure 完成，mlp_analog_pass_count = 0。
7. writeback trace 记录 full_basis_param_update_rows = 168，仍是真实 basis-param update，不是 readout proxy。
8. required artifacts 缺失为 0，provenance_violation_count = 0。
```

新增 no-go boundary：

```text
1. v13.3 证明问题不是单纯 BN5 diagonal/local metric 不够；当前 low-rank/block/family-balanced hooks 反而没有保住 v13.2 的 X7 P3。
2. 最接近 rows 的 blocker 主要是 source 不足、CouplingR2 坏化、NoiseSignalLeak 坏化和 Reservoir 坏化。
3. metric condition 很健康，说明当前 failure 不是 Woodbury/condition 数值爆炸。
4. Non-RAT rescue 仍卡在 workspace/lifetime，functional 结论仍被 Rational substrate 绑定。
5. 下一步不应继续在 BM8-BM15 局部参数上扩网格；需要新的机制级计划：真正非对角 task-family metric、Non-RAT/S2 substrate design，或重新定义 synthetic proof 的参数目标。
```

## 10. 用户再次追问后的 stop-contract 复核与 Non-RAT task-health 补跑

用户再次要求确认 v13.3 是否达成目标，若未达成则继续。本次重新读取 final route 与计划 stop rule：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
basis_natural_p3_pass_count = 0
lowrank_metric_pass_count = 0
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_s1_count = 0
required_artifact_missing_count = 0
```

计划 stop 条件复核：

```text
1. low-rank/block/family-balanced metric 全部没有超过 2/7：已满足，实际为 0/7。
2. Non-RAT rescue fails S1：已满足。
3. MLP analog closure complete：已满足，mlp_analog_pass_count = 0。
4. no-go boundary 已输出：已满足。
```

但复核发现一个审计边界：

```text
D-CHE12 focused run 是 lifetime repair candidate；
task-health/LineC probe 因 workspace gate fail 被 skip。
```

为了更严格覆盖计划里的 task-health repair，本次继续补跑 Chebyshev task-health candidate。第一次使用旧 candidate name：

```text
D-CHE6-K3DegreeEnergyCap
```

结果：

```text
KeyError: 'D-CHE6-K3DegreeEnergyCap'
```

原因：该 candidate 不在 v1235 registry 中。查询 v1235 后，实际可用 Chebyshev candidates 为：

```text
D-CHE12-LifetimeRecomputeBackward-K3
D-CHE13-FusedReadoutGradNoMaterialize-K3
D-CHE14-OptimizerStateLifetimeReuse-K3
D-CHE15-FullStepNoMaterialize-K3
D-CHE16-DegreeEnergyDampingSubstrate
D-CHE17-HighDegreeLateEnableSubstrate
D-CHE18-RoleDegreeEnergyCapSubstrate
D-CHE19-ChebyTangentTrustSubstrate
D-CHE20-DegreeNormalizedReadoutHealthSubstrate
```

因此补跑：

```text
D-CHE20-DegreeNormalizedReadoutHealthSubstrate
```

执行规模：

```text
dataset = MNIST
seed = 0
train_size = 64
val_size = 32
workspace_warmup_steps = 2
workspace_profile_steps = 5
hardening_epochs = 1
```

结果：

```text
workspace_rows = 1
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 0
linec_executed = 0
skip_reason = workspace_gate_fail
raw_memory_ratio_vs_mlp = 1.2288712066065914
incremental_memory_ratio_vs_mlp = 2.8111946532999164
step_ratio_vs_mlp = 1.1159531205667197
```

解释：

```text
1. D-CHE20 task-health candidate 仍未打开 workspace gate。
2. incremental_memory_ratio_vs_mlp = 2.811 > S1 gate 2.00。
3. hardening/LineC 按规则被 skip，不能写成 task-health pass。
4. Non-RAT rescue S1 仍为 0。
```

代码修改：

```text
experiments/run_v133_task_family_robust_basis_natural.py
  build_nonrat_rescue() 同时读取 nonrat_rescue_che12 与 nonrat_rescue_che20_taskhealth。
```

刷新后 final route：

```text
official_v133 route = R4-DiagonalAndLowRankMetricNoGo
nonrat_rescue_rows = 6
nonrat_rescue_s1_count = 0
required_artifact_missing_count = 0
```

最终判断仍是：

```text
v13.3 没有达成目标；
没有达到 S3/S5；
只保留 S1-EfficientControllableSubstrate；
BM8-BM15 全部没有 P3 pass；
synthetic task-family pass = 0/7；
Non-RAT focused lifetime 与 task-health candidates 均未打开 S1；
不允许 promotion；
合法 route = R4-DiagonalAndLowRankMetricNoGo。
```

本次继续推进后，stop-contract 更闭合：低秩/块/任务族 metric 已全失败，Non-RAT focused lifetime 与 task-health rescue 都失败，MLP analog 已完成，no-go boundary 已写出。我现在仍不确定如何在当前 v13.3 代码路径内安全地发明新的 Non-RAT/S2 substrate 或真正非对角 task-family metric，同时不把 proxy/diagnostic 编造成成功。

## 11. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v13.3 是否达成目标，若未达成则继续。本次重新读取最终 route、计划 stop rule、no-go boundary、required manifest、Non-RAT rescue、synthetic family summary 与 MLP analog closure。

最终 artifact：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_s2_healthy_base_count = 0
basis_natural_p3_rows = 168
basis_natural_p3_pass_count = 0
lowrank_metric_rows = 168
lowrank_metric_pass_count = 0
synthetic_rows = 21
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_rows = 6
nonrat_rescue_s1_count = 0
real_short_run_open_allowed = 0
real_short_run_pass_count = 0
mlp_analog_pass_count = 0
full_basis_param_update_rows = 168
provenance_violation_count = 0
manifest_rows = 36
missing_required = 0
```

计划 stop rule 对照：

```text
1. low-rank/block/family-balanced metric all fail to exceed 2/7 coverage：满足，实际为 0/7。
2. Non-RAT rescue fails S1：满足，D-CHE12/D-CHE20 focused runs 与 artifact reanalysis 均为 S1=0。
3. MLP analog closure complete：满足，mlp_analog_rows=1, mlp_analog_pass_count=0。
4. no-go boundary explains pivot：满足，v133_no_go_boundary.md 已写明 failed task families 与 Non-RAT blocker taxonomy。
```

最终判断仍是：

```text
v13.3 没有达成 S3/S5；
只达到 S1-EfficientControllableSubstrate；
不允许 promotion；
合法 route = R4-DiagonalAndLowRankMetricNoGo；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因不是轻易放弃，而是计划内继续条件已经闭合：BM8-BM15 未超过 2/7，Non-RAT lifetime/task-health rescue 已失败，MLP analog 已完成。我现在不确定如何在当前 v13.3 代码路径内安全实现新的 Non-RAT/S2 substrate 或真正非对角 task-family metric，同时不把 proxy/diagnostic 编造成成功。

## 12. 用户再次追问后的 stop-contract 复核 2

用户再次要求确认 v13.3 是否达成目标，若未达成则继续。本次复核最终 artifact 与计划 stop rule 后，结论仍不变。

最终 artifact：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
lowrank_metric_pass_count = 0
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_rows = 6
nonrat_rescue_s1_count = 0
mlp_analog_pass_count = 0
required_artifact_missing_count = 0
manifest_rows = 36
missing_required = 0
```

计划 stop rule 对照：

```text
1. low-rank/block/family-balanced metric all fail to exceed 2/7 coverage：满足，实际为 0/7。
2. Non-RAT rescue fails S1：满足，D-CHE12/D-CHE20 focused runs 与 artifact reanalysis 均为 S1=0。
3. MLP analog closure complete：满足，mlp_analog_rows=1, mlp_analog_pass_count=0。
4. no-go boundary explains pivot：满足，v133_no_go_boundary.md 已写明 failed task families 与 Non-RAT blocker taxonomy。
```

最终判断仍是：

```text
v13.3 没有达成 S3/S5；
只达到 S1-EfficientControllableSubstrate；
不允许 promotion；
合法 route = R4-DiagonalAndLowRankMetricNoGo；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因不是轻易放弃，而是当前计划内继续条件已经闭合：BM8-BM15 未超过 2/7，Non-RAT lifetime/task-health rescue 已失败，MLP analog 已完成，no-go boundary 已写明 pivot。我现在不确定如何在当前 v13.3 代码路径内安全实现新的 Non-RAT/S2 substrate 或真正非对角 task-family metric，同时不把 proxy/diagnostic 编造成成功。继续在 BM8-BM15 family 上排列局部超参会成为低价值搜索。

## 13. 用户再次追问后的 stop-contract 复核 3

用户再次要求确认 v13.3 是否达成目标，若未达成则继续。本次重新读取官方 route 与计划 stop rule，结论仍不变。

最终 artifact：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_natural_p3_pass_count = 0
lowrank_metric_pass_count = 0
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_rows = 6
nonrat_rescue_s1_count = 0
mlp_analog_pass_count = 0
```

计划 stop rule 对照：

```text
1. low-rank/block/family-balanced metric all fail to exceed 2/7 coverage：满足，实际为 0/7。
2. Non-RAT rescue fails S1：满足，Non-RAT rescue_s1_count=0。
3. MLP analog closure complete：满足，mlp_analog_pass_count=0 且 closure artifact 已存在。
4. no-go boundary explains pivot：满足，v133_no_go_boundary.md 已写明 failed task families 与 Non-RAT blocker taxonomy。
```

最终判断仍是：

```text
v13.3 没有达成 S3/S5；
只达到 S1-EfficientControllableSubstrate；
不允许 promotion；
合法 route = R4-DiagonalAndLowRankMetricNoGo；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因不是轻易放弃，而是计划内继续条件已经闭合；我现在不确定如何在当前 v13.3 代码路径内安全实现新的 Non-RAT/S2 substrate 或真正非对角 task-family metric，同时不把 proxy/diagnostic 编造成成功。继续在 BM8-BM15 family 上排列局部超参会成为低价值搜索。
