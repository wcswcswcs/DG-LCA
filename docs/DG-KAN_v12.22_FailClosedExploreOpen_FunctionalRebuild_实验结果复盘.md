# DG-KAN v12.22 FailClosedExploreOpen FunctionalRebuild 实验结果复盘

本复盘对应计划文档 `docs/DG-KAN_v12.22_FailClosedExploreOpen_FunctionalRebuild_实验结果分析与下一步计划.md`。结论只引用本轮落盘 artifact 中的实际数据；没有编造或补填缺失指标。

## 1. 本轮目标回读

v12.22 的目标不是放宽 promotion gate，而是修复执行语义：

```text
promotion: fail-closed
exploration: explore-open
```

也就是说，Line A/C/T/I/B/D 任一条线失败时，不能只写 no-go 和 next hypothesis 后停止；必须执行预注册的 exploration queue，或在预算耗尽时明确记录已执行/已延后原因。

本轮最终做到的是 Minimum Success F：不再 fail-fast。没有达到 label-free near-recovered、value survivor、safe actuator survivor 或 functional P3/P4 open。

## 2. 代码修改与审计结论

### 2.1 修改文件

- `experiments/run_v1218_b320_label_free_ablation.py`
  - 新增 A36-A42。
  - 新增 T1B temporal / transition / spectrum logging。
  - 新增 `adaptframeschedp` 训练期 unlabeled covariance residual adaptation。

- `dgkan/models/fc_purekan_primitives.py`
  - 新增 `reslowrankp`、`convexmixp`、`driftcotbankp`、`rolecondp`、`selfcondstopgradp`。
  - 修复 A37 frame 初始化中出现非有限值的 blocker：`_orthogonal_fill()` 和 convex frame normalize 现在会把 NaN/Inf fail-safe 到有限值，残差退化时回退到随机/坐标基向量。
  - 注意：该文件本轮前已有未提交改动，本轮没有回滚无关内容。

- `experiments/run_v1222_failclosed_explore_open_functional_rebuild.py`
  - 新增 v12.22 queue/fallback/budget/final-stop runner。
  - 新增 actuator v3 norm bisection + response-composed diagnostic。
  - 新增 shadow P4 capacity，明确 `promotion_allowed=0`。
  - 新增 Line D `NotExecuted_HypothesisGenerated` + budget tradeoff，不把 hypothesis 当作 executed candidate。

### 2.2 审计面结果

来自 `v1222_route_decision.json`：

```text
code_review_rows=13
line_r_required_missing=0
required_artifact_missing_count=0
fallback_missing_count=0
child_run_executed_count=7
child_run_deferred_with_budget_count=1
exploration_budget_exhausted=1
all_fallback_levels_executed=1
final_stop_allowed=1
next_hypothesis_generator_grants_final_stop=0
```

解释：

- 7 个 child job 真实执行：Line A 三个数据集、Line C、Line T、Line I、Line B shadow。
- 1 个 child job 明确预算延后：Line D classic family smoke 未执行，记录为 `deferred_with_budget_tradeoff=1`。
- `final_stop_allowed=1` 来自 queue/budget/fallback 合约完成，不来自 `next_hypothesis_generator_written`。

## 3. 最终 route

```text
route=R2-LabelFreeSignalFrameMissing
minimum_success=Minimum Success F
primary_blocker=no_label_free_candidate_met_official_promotion_gate
success_f_no_failfast=1
success_g_label_free_near_recovered=0
success_h_value_source_survivor=0
success_i_safe_actuator_survivor=0
success_j_functional_p3_open=0
```

这不是 functional success。它表示本轮完成了 v12.22 的 no-failfast execution contract，但仍没有找到合法 promotion 路径。

## 4. Line A：Label-Free Signal Frame A36-A42

运行规模：

```text
datasets=MNIST,Fashion-MNIST,KMNIST
seeds=0,1,2
epochs=3
train_size=512
val/test=256
batch_size=128
LineC batch=32
LineC sketch_dim=8
linea_rows=99
```

来自 `v1222_label_free_signal_frame.csv`：

| candidate | mean_delta_vs_A0 | worst_delta_vs_A0 | AUC_time_ratio_vs_mlp | LineC_pass_rate | exploration | official |
|---|---:|---:|---:|---:|---:|---:|
| A1-noYForStats | -0.032552083333333336 | -0.0625 | 1.3074867538228092 | 0.1111111111111111 | 0 | 0 |
| A36-ResidualA1LowRankFrame | -0.041666666666666664 | -0.078125 | 1.4762810802593231 | 0.0 | 0 | 0 |
| A37-ConvexMultiFrameMixture | -0.057291666666666664 | -0.09765625 | 1.5837833298421682 | 0.0 | 0 | 0 |
| A38-PersistentDriftCotangentBank | -0.054253472222222224 | -0.10546875 | 2.056094727672557 | 0.0 | 0 | 0 |
| A39-RoleConditionedFrame | -0.06163194444444445 | -0.10546875 | 1.3423676996610772 | 0.0 | 0 | 0 |
| A40-SelfConditionedStopGradFrame | -0.07682291666666667 | -0.15234375 | 1.7299049597481797 | 0.1111111111111111 | 0 | 0 |
| A41-UnlabeledFrameAdaptSchedule | -0.0234375 | -0.05859375 | 1.2943640066010882 | 0.1111111111111111 | 0 | 0 |
| A42-SmallLabelOracleUpperBound diagnostic | -0.038628472222222224 | -0.09375 | 1.2090423831449333 | 0.2222222222222222 | 0 | 0 |

Gate 对照：

```text
official:
  mean_delta_vs_A0 >= -0.003
  worst_delta_vs_A0 >= -0.010
  AUC_time_ratio_vs_mlp <= 1.00
  LineC_all_pass = 1

exploration:
  mean_delta_vs_A0 >= -0.010
  worst_delta_vs_A0 >= -0.030
  AUC_time_ratio_vs_mlp <= 1.15
  LineC_pass_rate >= 5/9
```

结论：

- `label_free_official_pass_count=0`
- `label_free_exploration_pass_count=0`
- best label-free candidate 为 A41，但仍只有：

```text
mean_delta_vs_A0=-0.0234375
worst_delta_vs_A0=-0.05859375
AUC_time_ratio_vs_mlp=1.2943640066010882
LineC_pass_rate=0.1111111111111111
```

A42 使用 small-label oracle upper-bound，只是 diagnostic，不可 promotion；它也没有显示出足以解释差距的 upper-bound 恢复。

## 5. Line C/T：Value Source v4 与 Visibility

来自 `v1222_linec_deployable_targets.csv`、`v1222_visibility_scores.csv`：

```text
linec_deployable_rows=63
T1B_native_logged_rows=63
g_la_official_pass=0
g_la_exploratory=0
```

Visibility 结果：

| tier | auc_joint | precision@k_joint | recall@k_joint | exploration | official | failure |
|---|---:|---:|---:|---:|---:|---|
| T1A | 0.3516624040920716 | 0.14285714285714285 | 0.058823529411764705 | 0 | 0 | AUC / precision-recall / soft-R2 / leaveout stability failed |
| T1B | 0.6508951406649617 | 0.14285714285714285 | 0.058823529411764705 | 0 | 0 | not T1A official; precision/recall below exploration; leaveout/soft-R2 gate failed |

解释：

- T1A 方向明显不可用：AUC 低于 0.5，predeclared sign 不稳定。
- T1B 的 AUC 达到 0.6509，但 precision/recall 只有 0.1429 / 0.0588；按 v12.22 exploration gate 至少 precision 或 recall 要 `>=0.15`，所以仍失败。
- T1B 默认不能作为 official direction source；本轮也没有额外证明它不等价于 CE-specific optimizer direction。

## 6. Line I：Drift-Constrained Actuator v3

运行范围：

```text
datasets=MNIST,Fashion-MNIST,KMNIST
seeds=0,1,2
families=I1-I10
budgets=0.0005,0.001,0.0025,0.005,0.01,0.02
signed_direction=-1,+1
actuator_rows=1080
```

来自 `v1222_actuator_safety.csv`：

```text
actuator_safe_movement_rows=0
actuator_safe_movement_dataset_seed_count=0
actuator_release_audit_rows=131
actuator_release_dataset_seed_count=7
actuator_control_resistant_dataset_seed_count=3
actuator_control_gap_mean=-0.3837713126254683
actuator_exploratory_success=0
actuator_official_gate_pass=0
```

重要观察：

- release rows 很多，但没有任何 row 同时满足 safe movement。
- 典型 release row 的问题是 projector angle 或 logit drift 不满足 safe movement gate。例如 KMNIST seed0 的 `I2-quad-proj-local-rotation` budget 0.001 有强 release，但：

```text
logit_max_abs_drift=0.7476762533187866
projector_angle_deg=3.089081287384033
NoiseSignalLeak_delta=-0.3094484880566597
RealSignalReservoirRatio_delta=-0.16822081804275513
safe_movement_pass=0
```

- 有些 direct/branch actuator drift 很小，但 projector angle 为 0，因此也不能算 safe movement：

```text
KMNIST seed0 I5 budget 0.001:
logit_max_abs_drift=0.0061463117599487305
projector_angle_deg=0.0
NoiseSignalLeak_delta=-0.3060552552342415
RealSignalReservoirRatio_delta=-0.16005128622055054
safe_movement_pass=0
```

结论：

- actuator 仍然能影响 release audit，但不能同时满足 movement / drift / control gap。
- 按计划，下一步应回到 Line C/T target/source，而不是继续增大 actuator amplitude。

## 7. Line B：Functional P3/P4 与 Shadow Capacity

Official P3/P4：

```text
p3_open=0
p3_pass=0
p4_open=0
p4_pass=0
```

原因：

- Line A official pass=0。
- T1A/T1B official value source pass=0。
- actuator official gate pass=0。

Shadow P4：

```text
shadow_p4_capacity_rows=9
shadow_p4_executed=1
promotion_allowed=0
```

代表性 shadow rows：

| dataset | seed | source_actuator | sketch_delta | logit_drift | noise_delta | reservoir_delta |
|---|---:|---|---:|---:|---:|---:|
| Fashion-MNIST | 0 | I5-branch-gain-redistribution | 0.011272983625531197 | 0.041353702545166016 | -0.039294395595788956 | -0.17895293235778809 |
| KMNIST | 1 | I5-branch-gain-redistribution | 0.01104874536395073 | 0.033665359020233154 | -0.16489455103874207 | 0.06636327505111694 |

解释：shadow diagnostic 说明 executor 可以影响 trajectory/audit metrics，但它明确是非 promotion 证据。由于 value source 和 actuator official gate 未开，不能构造 official functional P4。

## 8. Line D：Classic No-BSpline Portfolio

来自 `v1222_classic_family_status.csv`：

```text
classic_status_rows=5
classic_executed_count=0
classic_deferred_with_budget_count=5
```

每个 active family 的状态均为：

```text
status=NotExecuted_HypothesisGenerated
budget_tradeoff_recorded=1
```

这不是 RejectedForThisVersion，也没有把 hypothesis 当作 executed candidate。`v1222_child_run_manifest.csv` 明确记录：

```text
D-classic-status-budget:
  executed_this_version=0
  deferred_with_budget_tradeoff=1
  terminal_for_queue=1
  status=deferred_with_budget_tradeoff
```

说明：本轮 Line A/C/T/I/B 已消耗预注册 v12.22 runtime budget。Classic family 需要下一轮真正执行 family smoke，不能在本轮 claim rejected 或 failed。

## 9. Final-Stop 语义核验

来自 `v1222_final_stop_audit.json`：

```text
official_success=0
exploration_budget_exhausted=1
no_go_complete=1
all_fallback_levels_executed=1
manual_review_required=0
user_stop_flag=0
next_hypothesis_generator_written=1
next_hypothesis_generator_grants_final_stop=0
final_stop_allowed=1
```

解释：

- final stop 是允许的，因为 queue 终态完成、fallback 缺失为 0、预算耗尽。
- next hypothesis generator 只是记录下一轮方向，不再触发 final stop。

## 10. 成功标准对照

| 标准 | 结果 | 依据 |
|---|---:|---|
| Minimum Success F：不再 fail-fast | pass | `success_f_no_failfast=1` |
| Success G：label-free base near-recovered | fail | best A41 mean/worst/AUC/LineC 均未过 |
| Success H：legal value source exploratory survivor | fail | T1A AUC 0.3517；T1B precision/recall 未过 |
| Success I：safe actuator exploratory survivor | fail | safe_movement_dataset_seed_count=0 |
| Success J：functional P3 open | fail | p3_open=0 |
| Official Functional Success | fail | p4_open=0, p4_pass=0 |

## 11. 关键 blocker 与下一步结论

### Blocker A：label-free signal frame 仍显著落后

A36-A42 没有比 A1/no-y 或 v12.21 A31 形成结构性突破。A41 最好，但仍：

```text
mean_delta=-0.0234375
worst_delta=-0.05859375
AUC_time=1.2943640066010882
LineC_pass_rate=1/9
```

下一步应优先降低 residual/adaptation 能量，并把 role-conditioned frame 从“替换/混合 projector”改成更保守的 residual-on-A1 分层注入。

### Blocker B：T1B 有 AUC 信号但 top-k 不可用

T1B AUC=0.6508951406649617，但 precision/recall=0.142857/0.058824，说明排序有弱信号但不能可靠命中 release 支持集。下一步应做 calibration/top-k threshold/support concentration，不应 promotion。

### Blocker C：actuator release 与 safe movement 仍冲突

release rows=131，但 safe rows=0。能 release 的 quad rotation drift 太大；drift 小的 direct/branch actuator projector angle=0，不能满足 movement gate。下一步应回到 target/source，或者重新定义能同时产生 projector angle 与低 drift 的 response-composed actuator，而不是继续增大 norm。

### Blocker D：Line D 仍未执行 family smoke

本轮正确地没有把 hypothesis 当作 executed candidate；但 classic family 仍是未验证支线。下一轮如果要推进 Line D，必须给每个 active family 至少一个真实 smoke 或代码级 repair。

## 12. 审计包

最终 zip：

```text
results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_code_review_packet.zip
sha256=710cab38a4514f2d08612912ea74d9c531abb9ec264fc6fa5acdf4413d5da03e
entries=56
```

zip 内包含 v12.22 runner、Line A runner、primitive 实现、相关 v12.18-v12.21 runner、计划文档、v1222 artifacts 与 figures。
