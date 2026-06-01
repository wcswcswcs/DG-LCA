# DG-KAN v12.23：Fail-Closed Explore-Open 2.0、Label-Free Signal Frame、Loss-Agnostic Functional Value 与 Actuator 机制重建计划

> 基于：v12.22 `FailClosedExploreOpen FunctionalRebuild` 实验复盘、`v1222_code_review_packet.zip` 本地静态审查、v12.19-v12.21 连续结果。  
> 目标：不降低 promotion gate，但彻底修复 Codex “gate fail 后合法停下”的执行问题。下一轮必须从“完成诊断”升级到“持续执行预注册探索队列直到预算真正耗尽”。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；official functional direction 不使用 CE vector、label、permuted-label CE、validation/test/future outcome 或 dataset 名称。CE / ECE / CEp99 / Brier 只能作为审计和坏化约束，不能作为 direction source。

---

# 0. 一句话结论

v12.22 不是没有进展，但它仍不是 functional update 成功。它的真实进展是：执行语义比 v12.19-v12.21 更好，确实执行了 Line A/C/T/I/B 的 fallback queue，并且没有把 no-go 或 next hypothesis 冒充 promotion。

但 v12.22 的实验设置仍然不合理：它把“预算耗尽 + fallback 执行完一层”作为最终停止条件，导致 Codex 合法停止，而不是继续执行下一层机制假设。更严重的是，Line A 使用了更小预算的 3-epoch / train-size 512 scout protocol，Line D classic family 仍被预算延后，actuator 的 safety gate 对不同 role 使用同一 projector-angle 标准，可能把 branch/gain 类 actuator 的有效 release 错判为 unsafe。

因此 v12.23 的核心不是降低 gate，而是改变执行协议：

$$
\boxed{\text{promotion fail-closed, exploration continue-open, mechanism queue must execute until hard budget exhausted.}}
$$

也就是说：失败不能 promotion，但失败后必须继续跑下一层预注册 hypothesis；只写 no-go、写 next hypothesis、或完成一层 fallback，不再允许 final stop。

---

# 1. v12.22 实验结果独立分析

## 1.1 v12.22 的真实进展

v12.22 完成了比 v12.21 更严格的执行合同。路线文件显示：

```text
route = R2-LabelFreeSignalFrameMissing
minimum_success = Minimum Success F
success_f_no_failfast = 1
success_g_label_free_near_recovered = 0
success_h_value_source_survivor = 0
success_i_safe_actuator_survivor = 0
success_j_functional_p3_open = 0
```

审计面显示：

```text
code_review_rows = 13
line_r_required_missing = 0
required_artifact_missing_count = 0
fallback_missing_count = 0
child_run_executed_count = 7
child_run_deferred_with_budget_count = 1
exploration_budget_exhausted = 1
all_fallback_levels_executed = 1
final_stop_allowed = 1
next_hypothesis_generator_grants_final_stop = 0
```

这说明它确实不是 v12.19 那种“第一个 gate fail 后直接停”。它执行了 Line A 的三个数据集 child job、Line C、Line T、Line I、Line B shadow，并把 Line D 记录为 explicit budget tradeoff。这个执行纪律是进步。

但是，**这只是执行层进步，不是科学能力进步**。所有 promotion 仍关闭：没有 label-free base、没有 value source survivor、没有 safe actuator survivor、没有 functional P3/P4 open。

## 1.2 v12.22 最大问题：仍然允许 Codex 合法停在第一层探索

v12.22 的最终停止条件是：

```text
all_fallback_levels_executed = 1
exploration_budget_exhausted = 1
final_stop_allowed = 1
```

但从研究角度看，这个 `budget_exhausted` 太容易满足。它只说明本轮注册的有限 queue 执行或延后了，不说明机制搜索已经充分。尤其是：

```text
Line D classic family smoke 没有执行；
Line A 只是 3-epoch / train-size 512 scout；
Line I 只是 actuator v3 单层 response composition；
Line T 只是 T1A/T1B 两类 visibility；
Line B 只是 shadow P4 capacity，不是 genuine online functional candidate。
```

所以 v12.22 应该解释为：

$$
\boxed{\text{no-failfast contract pass, but exploration depth insufficient.}}
$$

下一版不能让 Codex 在“写出 no-go + next hypothesis + budget exhausted”后停止。必须要求：如果 official gate fail，Codex 要继续执行下一层预注册 fallback，直到达到明确的 `max_wallclock / max_gpu_hours / max_candidate_count`，并且每条主线都有最低执行数量。

---

# 2. Line A：label-free signal frame 的真实状态

## 2.1 当前结果

v12.22 Line A 跑了 A36-A42：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
epochs = 3
train_size = 512
val/test = 256
batch_size = 128
LineC batch = 32
LineC sketch_dim = 8
linea_rows = 99
```

候选表现如下：

| candidate | mean_delta_vs_A0 | worst_delta_vs_A0 | AUC_time_ratio_vs_mlp | LineC_pass_rate | official | exploration |
|---|---:|---:|---:|---:|---:|---:|
| A1-noYForStats | -0.03255 | -0.06250 | 1.30749 | 1/9 | 0 | 0 |
| A36-ResidualA1LowRankFrame | -0.04167 | -0.07813 | 1.47628 | 0/9 | 0 | 0 |
| A37-ConvexMultiFrameMixture | -0.05729 | -0.09766 | 1.58378 | 0/9 | 0 | 0 |
| A38-PersistentDriftCotangentBank | -0.05425 | -0.10547 | 2.05609 | 0/9 | 0 | 0 |
| A39-RoleConditionedFrame | -0.06163 | -0.10547 | 1.34237 | 0/9 | 0 | 0 |
| A40-SelfConditionedStopGradFrame | -0.07682 | -0.15234 | 1/9 | 1.72990 | 0 | 0 |
| A41-UnlabeledFrameAdaptSchedule | -0.02344 | -0.05859 | 1.29436 | 1/9 | 0 | 0 |
| A42-SmallLabelOracleUpperBound diagnostic | -0.03863 | -0.09375 | 1.20904 | 2/9 | 0 | 0 |

官方门和探索门分别是：

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

没有任何 candidate 过 official 或 exploration。

## 2.2 独立解读：不能把这轮 Line A 当作 label-free 彻底失败

v12.22 的 Line A 是一个小预算 scout，不是 hardening。它比 v12.21 的 Line A 更小：v12.21 曾经使用 `train_size=1024`、`val/test=512`、`epochs=3,8`、`LineC batch=64`、`sketch_dim=24`，而 v12.22 退到了 `train_size=512`、`val/test=256`、`epochs=3`、`LineC batch=32`、`sketch_dim=8`。

因此不能从 A36-A42 的失败直接推出：

```text
所有 label-free signal frame 都失败。
```

更准确地说：

$$
\boxed{\text{当前 v12.22 小预算 frame token 没有给出恢复 B320-current 的证据。}}
$$

这仍然是重要负结果，因为 A36-A42 不只是差一点，它们大多同时伤 task、AUC 和 Line C。但它不是最终 no-go。下一版必须对最有希望的 A1/A41/A36 做 **same-budget hardening**，而不是继续小预算 scout 后停止。

## 2.3 这暴露的本质问题

B320-current 强的根源之一是 label-informed trainprobe initialization。去掉 y_for_stats 后，label-free frame 必须自己构造一个能替代 supervised class-mean directions 的 signal frame。

当前 A36-A42 的失败说明：

```text
1. 用 unlabeled covariance / drift / role energy 构造的 frame 不等价于 class-separating frame；
2. 直接替换或混合 projector 会破坏 task trajectory；
3. Line C 不过说明它们不只是 accuracy 慢，而是 signal/reservoir/noise 分配不健康；
4. small-label oracle A42 也没有恢复，说明本轮 small-budget oracle 不足以解释差距，或者 oracle 注入方式错了。
```

下一步不能继续随便加 frame token。应该围绕 A1-noYForStats 做保守 residual-on-A1，而不是重构整个 projector。

---

# 3. Line C/T：value source 与 visibility 的真实状态

## 3.1 当前结果

v12.22 的 visibility 结果：

| tier | auc_joint | precision@k_joint | recall@k_joint | official | exploration | 主要失败 |
|---|---:|---:|---:|---:|---:|---|
| T1A precommit geometry | 0.35166 | 0.14286 | 0.05882 | 0 | 0 | AUC / precision-recall / soft-R2 / leaveout fail |
| T1B optimizer-update opaque | 0.65090 | 0.14286 | 0.05882 | 0 | 0 | 不能 official，precision/recall 和 leaveout/soft-R2 fail |

Soft target regression 也很差：

```text
Delta_NoiseSignalLeak_audit:
  leave_dataset_out_r2_min = -7.032
  leave_seed_out_r2_min = -1.919

Delta_RealSignalReservoirRatio_audit:
  leave_dataset_out_r2_min = -2.604
  leave_seed_out_r2_min = -6.300
```

## 3.2 独立解读：T1B 有弱信号，但不能直接变成 update source

T1A 明显不可用。它的 AUC 低于 0.5，且 leaveout 稳定性很差。

T1B 比 T1A 有意义。它的 joint AUC 达到 0.6509，说明 optimizer-observable update spectrum 不是完全随机。但 top-k precision 和 recall 仍然只有 0.1429 / 0.0588，离可部署 value source 很远。

我对 T1B 的判断是：

$$
\boxed{\text{T1B 是 weak diagnostic signal，不是 promotion signal。}}
$$

它可以用来做下一轮 hypothesis prioritization，但不能直接做 functional direction source。尤其 T1B 不是严格 pre-commit state feature，而是基于 native optimizer-update logging。它更接近“观察 base optimizer 已经怎么动”，而不是“在 commit 前独立判断应该怎么 functional update”。

## 3.3 重要异常：部分 score 的预设符号可能错了

`target_sign_sanity` 显示某些 component 如果反向使用，AUC 反而更高。例如 `G_role` 对 noise 的 predeclared AUC 很低，而 inverted AUC 接近 0.785。这个信号不能直接 promotion，因为它仍然用 audit target 校验方向，但它说明当前 G_LA 组合公式可能存在符号或尺度错配。

下一步必须做：

```text
1. sign-invariant component audit；
2. component-wise leaveout calibration；
3. no-label/no-CE 的 precommit score 与 audit-only target 分离；
4. 不能把 audit-learned sign 直接用于 online direction，除非 sign 在 train-stream history 上预注册并跨 leaveout 稳定。
```

---

# 4. Line I：actuator 的真实状态

## 4.1 当前结果

v12.22 actuator v3 运行：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
families = I1-I10
budgets = 0.0005,0.001,0.0025,0.005,0.01,0.02
signed_direction = -1,+1
actuator_rows = 1080
```

route summary：

```text
actuator_safe_movement_rows = 0
actuator_release_audit_rows = 131
actuator_release_dataset_seed_count = 7
actuator_control_resistant_dataset_seed_count = 3
actuator_control_gap_mean = -0.38377
actuator_official_gate_pass = 0
```

Shadow P4 也显示某些 rows 有明显 release，例如：

```text
Fashion-MNIST seed0, I5 branch-gain redistribution:
  sketch_delta_fro = 0.011273
  projector_angle_deg = 0.0
  logit_max_abs_drift = 0.041354
  NoiseSignalLeak_delta = -0.039294
  RealSignalReservoirRatio_delta = -0.178953

KMNIST seed2, I1 direct-role-scale-shift:
  sketch_delta_fro = 0.007111
  projector_angle_deg = 0.0
  logit_max_abs_drift = 0.018168
  NoiseSignalLeak_delta = -0.283451
  RealSignalReservoirRatio_delta = -0.128927
```

这些都是 shadow diagnostic，不能 promotion。但它们说明 actuator 不是完全没能力。

## 4.2 独立解读：当前 actuator gate 可能错惩罚 branch/gain 类 actuator

v12.22 的 safe movement gate 很可能把不同 role 混在一起了。对于 direct/quad projector 类 actuator，用 `projector_angle_deg` 判断 movement 是合理的；但对 branch/gain redistribution，projector angle 可能天然为 0，因为它不改变 projection 子空间，而是改变 role/gain/readout 分配。

因此 `safe_movement_rows=0` 不能简单解释为：

```text
所有 actuator 都动不了。
```

更准确是：

$$
\boxed{\text{当前 actuator safety gate 与 role-specific movement 语义不匹配。}}
$$

下一步必须改成 role-specific movement gate：

```text
direct / quad projector actuator:
  projector_angle_deg, sketch_delta_fro, logit drift

branch / gain actuator:
  branch_gain_shift_norm, role_energy_transport, logit drift, coupling change

readout / direct actuator:
  readout_cosine_change, classwise logit drift, tail risk

mixed actuator:
  response matrix decomposition + per-role safety constraints
```

这不是降低 gate，而是让 gate 测真正的 movement。

---

# 5. Line B：functional update 当前状态

v12.22 的 functional P3/P4 没有打开：

```text
functional_p3_rows = 6
p3_open = 0
p4_open = 0
p4_pass = 0
shadow_p4_capacity_rows = 9
```

P3 candidates 只是记录为 not opened：

```text
B1-C-T1-geometric-stabilizer
B2-AdamW-orthogonal-projector-stabilizer
B3-signal-frame-occupancy-rebalance
B4-primitive-role-energy-transport
B5-low-frequency-event-geometry-maintenance
B6-T1B-update-spectrum-preconditioner
```

这说明 functional update 线仍然没有 official success。当前最深 blocker 是：

$$
\boxed{\text{没有合法、稳定、precommit、loss-agnostic 的 value source 能指导 actuator。}}
$$

但 v12.22 给了两个可继续追的方向：

```text
1. T1B optimizer-update spectrum 有弱 AUC 信号；
2. actuator / shadow P4 有 release 能力，但 movement/safety/control 没闭合。
```

因此 v12.23 不应该继续只做 T1A/T1B 判定然后停下，而应该进入：

```text
T1B calibration + role-specific actuator response composition + shadow-to-cloned P3 conversion。
```

仍然不能 official promotion，但必须继续探索。

---

# 6. Line D：classic no-BSpline family 状态

v12.22 没有执行 classic family smoke：

```text
classic_executed_count = 0
classic_deferred_with_budget_count = 5
status = NotExecuted_HypothesisGenerated
```

这不能解释成 Rational / Chebyshev / Wavelet / RBF / Fourier 失败。它只能说明本轮预算被 Line A/C/T/I/B 消耗。

v12.23 必须把 Line D 从主 functional budget 中分离出来。否则每次 functional line 消耗预算，classic line 永远 deferred。

Active family：

```text
Rational
Chebyshev
Wavelet
RBF / FastKAN
Fourier
```

B-spline 继续 frozen，不进入 active budget。

---

# 7. 当前项目位置与距离目标

## 7.1 当前完成度估计

| 线 | 完成度 | 当前判断 |
|---|---:|---|
| B320-current diagnostic anchor | 80%-85% | 很强，但 label-informed trainprobe confound 仍限制 claim |
| Label-free B320-like base | 30%-40% | A1/A41 仍未恢复 task + Line C，当前 frame token 失败 |
| Line C audit | 75%-80% | 作为 audit 有价值，但要继续区分 audit-only 与 deployable observable |
| T1 deployable value source | 10%-20% | T1A 不行；T1B 有弱 AUC 但不能 top-k / leaveout |
| Actuator executor | 25%-35% | 有 release 能力，但 safety/movement/control 不闭合；gate 可能 role-mismatch |
| Functional official | 0% | P3/P4 仍关闭 |
| Classic no-BSpline portfolio | 35%-45% | 本轮没执行；历史上多数效率过但 task/expression/LineC 未闭合 |
| Code/provenance discipline | 55%-65% | artifact 合同好转，但 implementation readback 仍太薄，dirty code 未清理 |

## 7.2 离最终目标还差什么

最终目标不是 B320-current 好看，而是：

$$
\text{PureKAN base} + \text{loss-agnostic functional update}
>
\text{PureKAN base} + \text{ordinary optimizer / controls}.
$$

目前缺三件事：

```text
1. 干净 label-free 或明确 scope 的 base anchor；
2. 可部署的 loss-agnostic value source；
3. role-aware、safe、control-resistant actuator / functional event。
```

v12.22 的进展主要在 execution discipline；科学机制仍没过。

---

# 8. v12.23 总体目标

v12.23 的总目标不是“找到一个通过 gate 的 row”。总目标是：

$$
\boxed{
\text{在不降低 promotion gate 的前提下，把 functional update 的失败从单层 no-go 推进为机制级分解，}
\text{并至少打开一个合法的 exploratory path。}
}
$$

更具体地说，v12.23 要回答四个假设：

## H-A：label-free signal frame 是否只是当前注入方式错了？

当前 A36-A42 大多是替换/混合 projector。假设是：更保守的 residual-on-A1、低能量、分阶段注入，可能减少 task/AUC 伤害。

## H-T：T1B 是否可以成为 weak-but-useful exploration signal？

T1B AUC 有信号但 precision/recall 不够。假设是：通过 component calibration、sign-invariant audit、support expansion、continuous weighting，可以用它做 hypothesis prioritization，但不能直接 official。

## H-I：actuator safe movement 失败是否来自 role-gate mismatch？

假设是：branch/gain actuator 不应被 projector angle gate 判死；需要 role-specific movement metrics。

## H-B：functional update 是否需要先做 shadow-to-cloned P3，而不是直接 P4？

假设是：shadow P4 里有 release capacity，但需要转换成 cloned P3 + matched controls 才能判断因果价值。

---

# 9. v12.23 执行协议：Fail-Closed Explore-Open 2.0

## 9.1 新停止规则

Codex 不允许再因为一层 fallback 完成就停。`final_stop_allowed = 1` 只允许以下情况之一：

```text
A. official_success_reached = 1
B. hard_budget_exhausted = 1 AND all mandatory exploration levels executed = 1
C. explicit_user_stop_flag = 1
D. route = R4-MechanismNoGoAfterAllFallbacks, 且 no-go proof 包含至少三层失败证据
```

以下情况不得停止：

```text
next_hypothesis_generator_written = 1
no_go_boundary_written = 1
single fallback layer executed = 1
Line A/C/T/I/B 任一条线得到 fail = 1
Line D 被 deferred = 1
```

如果 Codex 在这些情况下停止，route 必须降级为：

```text
R0-FailFastExploreIncomplete
```

## 9.2 Mandatory exploration levels

每条线至少执行以下层级：

```text
Line A:
  Level 1: scout candidates
  Level 2: top-2 hardening on larger budget
  Level 3: failure-specific repair or no-go proof

Line T:
  Level 1: T1A/T1B scoring
  Level 2: component/sign/leaveout calibration
  Level 3: weak-signal sampler / support expansion diagnostic

Line I:
  Level 1: raw actuator sweep
  Level 2: role-specific movement gate
  Level 3: response composition + matched controls

Line B:
  Level 1: shadow P4 diagnostic
  Level 2: cloned P3 with controls
  Level 3: only if gates pass, short-run P4

Line D:
  Level 1: at least one smoke per active family, unless user explicitly freezes it
```

## 9.3 Budget policy

Budget must be explicit and partitioned:

```text
Line A budget: 35%
Line T/C budget: 20%
Line I/B budget: 30%
Line D budget: 10%
Line R/code audit budget: 5%
```

Line D cannot be starved by Line A/B. If line D is not executed, final route must include:

```text
R0-LineDBudgetStarved
```

unless user explicitly grants a one-version deferral.

---

# 10. Line R：代码语义与 provenance 审查

## 10.1 目标

v12.22 的 implementation readback 太薄，只列了改动文件和 route snapshot。v12.23 必须要求 Codex 对关键实现写出足够细的代码语义说明，尤其是：

```text
1. A36-A42 frame token 如何改变 quad_proj/direct_readout/branch/gain；
2. T1B optimizer-update features 是否是真正 native logged，是否可能混入 response；
3. Line C audit target 是否只用于 scoring，不参与 direction；
4. actuator v3 的 role movement gate 如何定义；
5. shadow P4 是否完全 promotion_allowed=0；
6. dirty file / uncommitted prior changes 是否影响本轮结论。
```

## 10.2 必须新增 artifact

```text
v1223_code_semantics_review.md
v1223_diff_isolation_manifest.csv
v1223_dirty_tree_audit.csv
v1223_core_symbol_map.json
v1223_feature_role_provenance.csv
v1223_actuator_role_gate_map.csv
```

## 10.3 Gate

进入任何 promotion 前必须满足：

```text
code_semantics_review_pass = 1
dirty_tree_unexplained_change_count = 0
feature_provenance_unknown_count = 0
actuator_role_gate_unknown_count = 0
```

如果 `fc_purekan_primitives.py` 存在本轮前未提交改动，Codex 必须列出：

```text
which symbols were already dirty;
which symbols changed this version;
whether dirty code touched B320/A36-A42/LineC/actuator path;
rollback plan.
```

---

# 11. Line A：Label-Free Signal Frame v3

## 11.1 总目标

不再试图一轮替代 B320-current。v12.23 的目标是：

$$
\boxed{\text{在 A1-noYForStats 上恢复 task trajectory 和 Line C，找到至少一个 label-free near-recovered candidate。}}
$$

## 11.2 Candidate design

保留 A1/A41/A36 作为 baseline，然后新增更保守的 residual-on-A1：

```text
A43-ConservativeResidualA1Frame-r005
A44-ConservativeResidualA1Frame-r010
A45-StagedUnlabeledAdapt-warm1-r005
A46-RoleResidualNoReplace-directOnly
A47-QuadProjLowRankResidualFrozenMain
A48-BranchGainOnlyLabelFreeResidual
A49-A1PlusT1BWeakSignalFrame-diagnostic
A50-SmallLabelOracleMatchedBudget-diagnostic
```

设计原则：

```text
1. 不替换 A1 frame，只加 residual；
2. residual energy capped；
3. warmup 后再启用 unlabeled adaptation；
4. role-conditioned frame 先只作用于 residual branch，不改主 projector；
5. A49 只作为 diagnostic，不能 official；
6. A50 仅做 upper bound，不 official。
```

## 11.3 两级预算

### Scout budget

```text
train_size = 512
val/test = 256
epochs = 3
seeds = 0,1,2
```

### Hardening budget

对 scout top-2 必须自动执行：

```text
train_size = 1024
val/test = 512
epochs = 8
LineC batch = 64
sketch_dim = 24
seeds = 0,1,2
```

v12.23 不允许只跑 scout 后停止。

## 11.4 Metrics

必须记录：

```text
candidate_id
uses_label
uses_ce_vector
uses_validation_or_test
uses_dataset_name
mean_delta_vs_A0
worst_delta_vs_A0
mean_delta_vs_MLP
worst_delta_vs_MLP
AUC_step_ratio_vs_mlp
AUC_time_ratio_vs_mlp
ECE_delta_vs_A0
CEp99_delta_vs_A0
LineC_pass_rate
LineC_all_pass
CouplingR2_delta_vs_A0
NoiseSignalLeak_delta_vs_A0
RealSignalReservoirRatio_delta_vs_A0
frame_rank
frame_condition
P_energy_on_frame
residual_frame_energy_ratio
role_energy_by_candidate
quad_feature_std_mean
branch_scale_norm
NaN_count
```

## 11.5 Gate

Official label-free base gate：

$$
mean\_delta\_{vsA0} \ge -0.003,
$$

$$
worst\_delta\_{vsA0} \ge -0.010,
$$

$$
AUCtime\_{vsMLP} \le 1.00,
$$

$$
LineC\_{all\_pass}=1.
$$

Exploratory gate：

$$
mean\_delta\_{vsA0} \ge -0.010,
$$

$$
worst\_delta\_{vsA0} \ge -0.030,
$$

$$
AUCtime\_{vsMLP} \le 1.15,
$$

$$
LineC\_{pass\_rate} \ge 5/9.
$$

If no candidate passes exploratory gate, Codex must execute failure-specific repair, not stop.

## 11.6 Failure-specific actions

```text
If task mean/worst fail but LineC improves:
  reduce residual energy, freeze main P, warm residual after epoch 1.

If AUC_time fail but final acc ok:
  staged activation, lower residual ramp, no optimizer change unless all candidates share same failure.

If LineC fail but task ok:
  reduce noise-sensitive role energy, add branch-only residual, audit signal/reservoir decomposition.

If NaN or condition explosion:
  fallback to coordinate/random orthogonal frame; log NaN as real failure, not zero.

If small-label oracle fails:
  conclude frame injection path is wrong, not simply missing class signal.
```

---

# 12. Line C/T：Loss-Agnostic Value Source v5

## 12.1 总目标

T1A failed; T1B weakly positive. v12.23 must not treat T1B as official, but must not discard it either. The goal is to turn weak visibility into a calibrated exploration signal.

## 12.2 Feature tiers

```text
T1A:
  pure precommit unlabeled state features.
  Eligible for official only.

T1B:
  optimizer-observable but loss-agnostic native update features.
  Eligible for exploratory prioritization only unless separately proven precommit-safe.

T2:
  clone-probe response features.
  diagnostic only.

T3:
  label/CE hard audit targets.
  audit only.
```

## 12.3 New diagnostics

Must run:

```text
T1B component ablation:
  update norm, spectrum entropy, top share, role energy flow, half-life, transition cosine.

Sign-invariant component audit:
  evaluate both signs offline; declare online sign only if stable under leave-dataset/seed.

Support expansion:
  use more windows and seeds before declaring precision no-go.

Calibration:
  isotonic or quantile calibration on train splits only; evaluate heldout dataset/seed.

Soft release ranking:
  not just hard top-k; record pairwise ranking AUC and soft-R2.
```

## 12.4 Metrics

```text
feature_tier
feature_family
feature_name
uses_label
uses_ce_vector
uses_future_outcome
precommit_available
native_logged
clone_probe_only
auc_noise
auc_reservoir
auc_joint
precision_at_k_joint
recall_at_k_joint
bottom_precision_at_k_joint
bottom_recall_at_k_joint
leave_dataset_auc_min
leave_seed_auc_min
leave_window_auc_min
soft_target_r2_noise
soft_target_r2_reservoir
pairwise_rank_auc_noise
pairwise_rank_auc_reservoir
support_count
support_concentration
sign_stability
visibility_exploration_pass
visibility_official_pass
```

## 12.5 Gate

T1A official visibility:

$$
AUC\_{joint,min} \ge 0.70,
$$

$$
precision@k\_{joint,min} \ge 0.25,
$$

$$
recall@k\_{joint,min} \ge 0.20,
$$

$$
leaveout\_auc\_{min} \ge 0.60.
$$

T1B exploration visibility:

$$
AUC\_{joint,min} \ge 0.60,
$$

$$
\max(precision@k, recall@k) \ge 0.15,
$$

$$
sign\_stability=1.
$$

T1B cannot directly open official P4. It can open Line I exploration and shadow/cloned P3.

## 12.6 Failure-specific actions

```text
If T1B AUC > 0.60 but precision/recall low:
  run support expansion, variable-k threshold, and calibration.

If sign-inverted score works:
  do not flip online immediately; require leaveout-stable sign.

If soft R2 strongly negative:
  switch from regression to pairwise ranking objective offline.

If support is concentrated:
  gather more windows/seeds before no-go.

If T1A remains below 0.5:
  deprioritize T1A as source; keep only for audit.
```

---

# 13. Line I：Role-Specific Actuator v4

## 13.1 总目标

v12.22 showed actuator release exists but safe movement gate fails. v12.23 must test whether this is role-gate mismatch.

## 13.2 Role-specific movement metrics

```text
Direct/quad projector actuator:
  sketch_delta_fro
  projector_angle_deg
  quad_proj_energy_delta
  logit_max_abs_drift

Branch/gain actuator:
  branch_gain_shift_norm
  role_energy_transport_l1
  branch_scale_entropy_delta
  classwise_logit_gain_drift
  logit_max_abs_drift

Readout actuator:
  readout_cosine_change
  direct_readout_energy_delta
  classwise_margin_drift
  tail_risk_delta

Mixed actuator:
  response decomposition into role contributions
  max per-role drift
  total logit drift
```

## 13.3 Candidate families

```text
I11-BranchGainRedistributionRoleSafe
I12-DirectRoleScaleShiftRoleSafe
I13-QuadResidualLowDriftRotation
I14-ResponseComposedBranchDirect
I15-NoiseReleaseReservoirReleaseQP
I16-T1BWeightedActuatorSampler-diagnostic
I17-ShadowPositiveReplayMatchedControls
```

## 13.4 Response-composition objective

For actuator coefficients $a$:

$$
\min_a
\quad
\beta \widehat{\Delta NoiseSignalLeak}(a)
+
\gamma \widehat{\Delta RealSignalReservoirRatio}(a)
+
\eta \widehat{TailRisk}(a)
-
\alpha \widehat{\Delta CouplingR^2}(a)
+
\lambda \|a\|_2^2
$$

subject to:

$$
logit\_max\_abs\_drift(a) \le 0.05,
$$

$$
role\_movement(a) \ge \tau_{role},
$$

$$
holdout\_loss\_ratio(a) \le 1.002.
$$

Here `holdout_loss_ratio` is audit only for cloned probe; it must not use validation/test for online commit.

## 13.5 Controls

Every actuator candidate must be compared with:

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
ShuffledActuatorBasis
MatchedRoleEnergyRandomActuator
BranchGainRandomControl
```

## 13.6 Gate

Exploratory actuator pass:

$$
role\_safe\_movement=1,
$$

$$
\Delta NoiseSignalLeak \le -0.01,
$$

$$
\Delta RealSignalReservoirRatio \le -0.01,
$$

$$
control\_gap \ge 0.002.
$$

Official actuator pass:

$$
control\_gap \ge 0.005,
$$

and pass across at least 8/9 dataset-seed rows or bootstrap CI lower bound $\ge 0$.

## 13.7 Failure-specific actions

```text
If release exists but role-safe movement fails:
  inspect role gate; use branch/gain movement metric instead of projector angle.

If movement exists but no release:
  response composition or T1B-weighted sampling.

If release and movement exist but controls also win:
  subtract AdamW-parallel component; add matched role-energy controls.

If logit drift exceeds gate:
  norm bisection and split event into smaller low-frequency maintenance events.
```

---

# 14. Line B：Functional Update Construction

## 14.1 总目标

Functional update remains gated. v12.23 can run shadow/cloned diagnostics but cannot claim official success unless T/I gates pass.

## 14.2 Candidate categories

```text
B1-T1BCalibratedGeometricStabilizer-diagnostic
B2-RoleSafeActuatorComposedMaintenance
B3-AdamWOrthogonalSignalResidual
B4-NoiseLeakVetoedGeometryMaintenance
B5-ReservoirReleaseLowFrequencyEvent
B6-BranchGainRoleEnergyTransport
```

## 14.3 Training form

Functional update is low-frequency maintenance:

$$
\theta_{t+1}=\theta_t+\Delta\theta_{task}+\lambda_t\Delta\theta_{func}.
$$

It must not replace the task optimizer.

## 14.4 P3 cloned gate

P3 candidate must satisfy:

$$
\Delta CouplingR^2 \ge 0.02,
$$

$$
\Delta NoiseSignalLeak \le -0.01,
$$

$$
\Delta RealSignalReservoirRatio \le -0.01,
$$

$$
control\_gap \ge 0.005,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
ECE\_delta \le 0.02.
$$

## 14.5 P4 short-run gate

Only if P3 passes:

$$
AUCtime_{func} \le AUCtime_{base},
$$

$$
Acc_{func} \ge Acc_{base}-0.003,
$$

$$
T_{amortized,func}/T_{base} \le 1.05,
$$

and all Line C metrics non-worse.

## 14.6 Shadow P4 rule

Shadow P4 may run even when official P4 is gated, but:

```text
promotion_allowed = 0
must use cloned weights or explicitly diagnostic path
must include matched controls
must not affect route as success
must write shadow_to_p3_conversion_recommendation
```

---

# 15. Line D：Classic No-BSpline Portfolio

## 15.1 Rule

Line D has independent budget. v12.23 must execute at least one smoke or repair per active family unless user explicitly defers.

Active families:

```text
Rational
Chebyshev
Wavelet
RBF/FastKAN
Fourier
```

Frozen:

```text
B-spline
```

## 15.2 Required smoke per family

```text
Rational:
  denominator-safe coupling repair + group diversity tangent metric.

Chebyshev:
  degree-energy damping + task trajectory repair.

Wavelet:
  local support scale-diversity task-stable repair.

RBF/FastKAN:
  compact capacity expression repair without dense RBF fallback.

Fourier:
  low-frequency expression repair without high-frequency noise path.
```

## 15.3 Metrics

```text
family
candidate_id
executed_this_version
L3_efficiency_pass
A4_expression_pass
A5_task_pass
LineC_pass
mean_delta_vs_mlp
worst_delta_vs_mlp
AUC_time_ratio
ECE_delta
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
blocker
next_candidate
```

## 15.4 Status taxonomy

```text
FamilyPass
FamilyNearPass
KernelBlocked
ExpressionBlocked
TaskBlocked
GeometryBlocked
NotExecutedBudgetViolation
RejectedForThisVersion
```

`NotExecuted_HypothesisGenerated` is no longer acceptable if Line D budget is allocated.

---

# 16. Required artifacts

v12.23 must produce:

```text
v1223_route_decision.json
v1223_final_stop_audit.json
v1223_budget_accounting.csv
v1223_child_run_manifest.csv
v1223_fallback_execution_manifest.csv
v1223_code_semantics_review.md
v1223_diff_isolation_manifest.csv
v1223_dirty_tree_audit.csv
v1223_label_free_signal_frame.csv
v1223_label_free_hardening.csv
v1223_label_free_failure_decomposition.csv
v1223_linec_deployable_targets.csv
v1223_visibility_scores.csv
v1223_visibility_leaveout.csv
v1223_visibility_component_ablation.csv
v1223_t1b_calibration.csv
v1223_target_sign_sanity.csv
v1223_soft_target_regression.csv
v1223_actuator_role_gate_map.csv
v1223_actuator_response_dictionary.csv
v1223_actuator_safety_roleaware.csv
v1223_actuator_controls.csv
v1223_shadow_to_cloned_p3.csv
v1223_functional_p3.csv
v1223_functional_p4_short.csv
v1223_classic_family_status.csv
v1223_no_go_boundary.md
v1223_next_hypothesis_generator.md
v1223_code_review_packet.zip
```

---

# 17. Required visualizations

```text
fig_v1223_queue_progress.svg
fig_v1223_budget_burn_by_line.svg
fig_v1223_label_free_task_linec_pareto.svg
fig_v1223_label_free_hardening_delta.svg
fig_v1223_frame_energy_role_decomposition.svg
fig_v1223_visibility_auc_precision_recall.svg
fig_v1223_t1b_component_ablation.svg
fig_v1223_sign_sanity_by_component.svg
fig_v1223_leaveout_visibility_heatmap.svg
fig_v1223_soft_target_regression_residuals.svg
fig_v1223_actuator_role_movement_pareto.svg
fig_v1223_actuator_release_vs_drift.svg
fig_v1223_actuator_control_gap_by_family.svg
fig_v1223_shadow_to_cloned_p3.svg
fig_v1223_p4_linec_trajectory.svg
fig_v1223_classic_family_status.svg
fig_v1223_no_go_decision_tree.svg
```

---

# 18. Final route definitions

## Success routes

```text
S1-LabelFreeNearRecovered:
  At least one label-free candidate passes exploration hardening.

S2-ValueSourceExploratoryVisible:
  T1B or T1A passes exploration visibility with leaveout stability.

S3-RoleSafeActuatorSurvivor:
  At least one actuator family passes role-specific movement + release + controls exploratory gate.

S4-FunctionalP3Opened:
  Functional cloned P3 opens with matched controls.

S5-OfficialFunctionalSuccess:
  P4 short-run passes all official gates.
```

## Failure routes

```text
R1-LabelFreeFrameTaskBlocked:
  label-free hardening fails task/AUC even after conservative residual repair.

R2-DeployableValueSourceInvisible:
  T1A/T1B/T2 diagnostics fail after support expansion and sign sanity.

R3-ActuatorRoleSafeReleaseConflict:
  release and safe movement cannot be made compatible even with role-specific gates.

R4-FunctionalMechanismNoGo:
  value source and actuator pass individually but no functional P3 candidate beats controls.

R0-FailFastExploreIncomplete:
  Any mandatory fallback/hardening/LineD smoke is skipped without explicit user stop.

R0-CodeSemanticsIncomplete:
  code/provenance audit incomplete or dirty-tree change unexplained.
```

---

# 19. Codex automatic fallback table

| Failure | Codex must try next | Stop allowed? |
|---|---|---|
| Label-free scout fails | Hardening top-2 on 1024/512/e8 | No |
| Hardening task fail | Lower residual energy + staged warmup | No |
| Hardening LineC fail | Role-energy residual + signal/reservoir decomposition | No |
| T1A fail | T1B component / sign / support calibration | No |
| T1B AUC > 0.6 but precision low | Variable-k threshold + support expansion | No |
| T1B all fail | T2 upper-bound diagnostic + no-go proof | Only after T2 |
| Actuator release but no safety | Role-specific movement gate + norm bisection | No |
| Actuator safety but no release | Response composition QP | No |
| Actuator release but controls win | AdamW-orthogonal residual + matched role controls | No |
| Functional P3 fail | Shadow-to-cloned P3 conversion + control audit | No |
| Classic Line D not executed | Run one smoke per active family | No |
| Dirty code found | Diff isolation + rerun minimal affected tests | No |

---

# 20. 最终总结

v12.22 的正确解读是：执行纪律变好了，但探索深度仍然不够。它证明 A36-A42 当前 scout 不行、T1A 不行、T1B 只是弱信号、actuator 有 release 但安全/controls 不闭合、classic family 没执行。它没有证明 functional update 没希望，也没有证明 label-free base 不可能。

v12.23 的关键变化是把实验从“一层 fail-closed continuation”升级成“多层 fail-closed explore-open”。Codex 不能再在 gate fail 后写完 no-go 就停止；必须继续执行预注册 fallback queue，直到真正预算耗尽，或者打开一个合法 exploratory path。

下一步最重要的三件事是：

```text
1. 对 A1/A41/A36 进行 same-budget label-free hardening，而不是只看小预算 scout；
2. 把 T1B 从弱 AUC 信号推进为可校准的 exploration signal，但不直接 official；
3. 用 role-specific movement gate 重新审视 actuator，尤其是 branch/gain release rows。
```

如果这三件事仍失败，并且 Line D 也完成真实 smoke，才可以进入更强 no-go 或重置 functional geometry target。
