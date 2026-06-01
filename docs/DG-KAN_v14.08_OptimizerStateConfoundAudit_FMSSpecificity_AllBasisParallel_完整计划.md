# DG-KAN v14.8：Optimizer-State Confound Audit / FMS Specificity / All-Basis Parallel 完整计划

生成时间：2026-05-29（Asia/Singapore）

> 本计划基于 v14.7 `OptimizerStateTransportFMS AllBasisParallel` 真实复盘、v14.6.1 机制诊断、v14.5 action-bank upper-bound 诊断、v9 系列 action-search 教训，以及 Generalization / Deep Manifold 两篇文档启发。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no label-informed initialization；functional direction 不使用 validation/test/future/query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只做 audit / gate；不把 diagnostic / endpoint-only / control-equivalent / smoke 写成 promotion。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是寻找某个局部好看的 update。项目目标是：

$$
\boxed{
\text{在 label-free strict FC-PureKAN / KAN-basis substrate 上，通过 functional update 改善训练过程，}
\text{得到比 same base + ordinary AdamW / strong controls 更好的模型。}
}
$$

“更好”必须同时满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹更健康，AUC-step / AUC-time 不输；
4. 几何不撕裂：LineC、signal/reservoir/noise、tail/calibration 不坏；
5. functional gain 必须独立于 AdamW、Random、NoOp、generic optimizer-state reset controls；
6. functional update 不能靠 CEp99 / NLL / ECE / LineC hard target 生成方向；
7. 不能按 dataset / seed 调规则。
```

最终成功不是：

$$
\text{FMS 有 endpoint gain}
$$

而是：

$$
\boxed{
\text{FMS mechanism 在 strict gate 下 9/9 real 3x3 通过，}
\text{并且击败最强 optimizer-only / reset / random / MLP analog controls。}
}
$$

---

## 0.2 当前真实进展

到 v14.7 为止，项目状态是：

```text
Rational substrate:
  仍是唯一稳定 functional carrier。

Rational-FMS synthetic:
  已经达成过 S3。

Rational-FMS real-transfer:
  v14.4 / v14.5 打开 S4b，actual best 曾到 6/9；
  v14.6.1 diagnostic zero_moment_reset 曾把 coverage 推到 9/9；
  v14.7 official confirmation 未复现为 S5。

v14.7 official:
  every-step zero moment reset endpoint-vs-AdamW 可到 9/9，
  但 random/full reset controls 也能复现 endpoint gain，
  因此不是 FMS-specific。

v14.7 event-only repair:
  更符合计划语义，但 K2 endpoint-vs-AdamW 只有 4/9，strict 0/9。

v14.7 sparse affected mask:
  delta_top25 把实际 reset fraction 降到约 25%，但 endpoint 退到 2/9。

v14.7 strict:
  所有 run strict gate 均为 0/9，promotion_allowed=0。
```

因此当前不再能说：

```text
zero_moment_reset 已经是 FMS breakthrough。
```

更准确地说：

$$
\boxed{
\text{v14.7 证明了 optimizer-state transport 方向值得研究，}
\text{但也证明当前 zero-moment reset 结果存在严重 generic optimizer-state confound。}
}
$$

---

# 1. 各条线当前进展百分比

这些百分比是基于 gate、机制清晰度、artifact 完整度、离 official success 的距离做出的研究判断，不是 artifact 中的官方字段。

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | 工程闭包强，不是 blocker |
| Historical FHQ / B320-current | 85% frozen | 历史强，但 label-informed init 已禁用，不能 official |
| Line C：几何审计 | 88% | 审计稳定；当前失败基本不是 LineC 主导 |
| PopRisk / FMS 实现面 | 94% | persistent state、per-example gradient、projection trace、real runner、optimizer-state probe 成熟 |
| Generic MLP-FMS | 45% | 有 generic signal，但不稳定，不能写成 KAN success |
| Rational substrate | 85% | 当前最稳定 substrate |
| Rational-FMS synthetic | 70% | 已达 S3，KAN-specific synthetic signal 曾成立 |
| Rational-FMS real-transfer | 55% | 曾到 S4b / 6/9；v14.7 official strict 回到 0/9，机制需重审 |
| Optimizer-state transport diagnostic | 70% diagnostic / 0% official | zero reset diagnostic 有线索，但官方 confirmation 失败 |
| FMS specificity / confound audit | 20% | v14.7 发现 reset controls 复现 endpoint，下一轮主线 |
| Affected-coordinate identification | 15% | 当前 mask median ~0.995，affected-only 近似 all-role reset；top25 稀疏化退化 |
| Step-time / overhead closure | 20% | K2 median step_time_ratio > 1.25，是 hard blocker |
| Train-stream controller | 0% | 不允许启动；v14.5 已证明 action-bank upper bound 不足 |
| Action-search route | 0% / closed | 明确禁止；action bank 只能作为一次性 diagnostic |
| Wavelet substrate | 35%-40% | 有历史线索，但 v14.7 未新训，不能 promotion |
| Fourier substrate | 35%-40% | 低频线有线索，仍不是 strict substrate success |
| RBF / FastKAN substrate | 25% | compact workspace 有进展，但 task-health collapse |
| Chebyshev substrate | 25% | workspace / exact path 有进展，task-health / LineC 不闭合 |
| Non-RAT official FMS | 0%-5% | 没有 full substrate，不允许 official proof |
| Official S5 functional success | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 42%-48% | real S4b 仍是进展，但 v14.7 把 zero reset official route 打回诊断层 |

---

# 2. v14.7 独立分析

## 2.1 v14.7 是否有进展？

有，但不是能力进展。

v14.7 的进展是把 v14.6.1 的 diagnostic positive 做了 official-style confirmation，并发现它不能 promotion：

```text
official_v147:
  K2 strict = 0/9
  K2 endpoint-vs-AdamW = 9/9
  K2 specificity = 0/9
  K2 efficiency = 1/9
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

这说明：

```text
1. every-step reset 的 endpoint 9/9 不是 FMS-specific；
2. event-only reset 才更符合 v14.7 语义，但它不能复现 diagnostic 9/9；
3. 稀疏 affected mask 不是简单修复，top25 反而退化；
4. strict gate 始终 0/9；
5. promotion_allowed 必须保持 0。
```

所以 v14.7 的核心价值是 **阻止我们把 v14.6.1 的 diagnostic positive 错写成 functional breakthrough**。

---

## 2.2 为什么 v14.7 打碎了上一轮乐观？

v14.6.1 的解释是：

```text
FMS event 后 AdamW 一阶动量 stale；
zero_moment_reset 消除 stale momentum；
AUCtime-only debt 被修复。
```

v14.7 发现这个解释还不够，因为：

```text
1. 如果每一步 reset，一些 random/full reset controls 也能 endpoint 成功；
2. 如果只在 FMS event reset，K2 只有 endpoint 4/9；
3. 当前 affected mask 太宽，affected-only 近似 all-FMS-role reset；
4. sparse top25 mask 降低 reset fraction 后性能退化；
5. step_time gate 没过。
```

所以当前真实解释更像：

$$
\boxed{
\text{v14.6.1 捕捉到 optimizer-state 与 FMS 的相互作用，}
\text{但 v14.7 显示当前实现混入了 generic momentum-reset / overhead / mask-width confound。}
}
$$

---

## 2.3 当前真正 blocker

当前 blocker 不是“再找一个动作”，也不是“controller 选不好”。

当前 blocker 是：

$$
\boxed{
\text{FMS 事件、AdamW 状态、affected coordinate、runtime overhead 之间的语义没有闭合。}
}
$$

更细分：

```text
B1. Specificity blocker:
  K2 endpoint gain 可被 random/full reset controls 复现。

B2. Event semantics blocker:
  every-step reset 有 endpoint 9/9，但 event-only reset 只有 4/9。
  说明成功可能来自连续 momentum suppression，而不是 FMS event transport。

B3. Affected mask blocker:
  affected_param_fraction median ≈ 0.995，几乎全参数；
  affected-only 与 all-FMS-role reset 几乎等价。

B4. Sparse mask blocker:
  delta_top25 把 reset fraction 降到约 25%，但 endpoint 退到 2/9。

B5. Efficiency blocker:
  event-only K2 median step_time_ratio = 1.303882 > 1.25。

B6. Strict gate blocker:
  strict success 始终 0/9，不只是 endpoint gate 不足。
```

这些 blocker 指向一个统一判断：

$$
\boxed{
\text{当前 FMS-state-transport route 还没有证明“functional-specific trajectory repair”。}
}
$$

---

# 3. 为什么不能继续旧方向

## 3.1 不能回到 action search

v9 和 v12 之前反复犯过的错误是：

```text
局部 positive -> 扩动作；
P3 positive -> 期待 P4；
oracle frontier -> controller；
actuator 能动 -> 以为有因果价值。
```

这次必须写死：

```text
1. 不新增 K-RT / K-AUC / K-FL action token；
2. 不启动 controller；
3. 不用 action bank 做搜索空间；
4. 不基于局部 positive row 生成新动作；
5. 不用 audit metric 反推方向；
6. 不按 dataset / seed 写规则。
```

v14.8 只允许做 **mechanism disambiguation**：判断 v14.7 失败到底是 generic optimizer reset、event semantics、affected mask、overhead，还是 FMS 本身无效。

---

## 3.2 不能继续调 reset mask / threshold / strength

以下都禁止：

```text
reset fraction 0.2 / 0.3 / 0.5 网格；
delta_topK 继续扫 K；
FMS strength 网格；
refresh interval 网格；
lr 小修；
step_time gate 降低；
endpoint 代替 strict；
拼接不同 run 的局部 positive。
```

原因：v14.7 已经证明 top25 sparse mask 退化，step_time 和 specificity 是 hard blocker。继续调这些会重演“寻找好动作”的老错误。

---

# 4. 下一步核心方案：v14.8

新计划命名为：

```text
DG-KAN v14.8
Optimizer-State Confound Audit
+
FMS Specificity Test
+
All-Basis Parallel Substrate
```

v14.8 的核心不是继续让 zero_moment_reset 过 gate，而是做决定性分解：

$$
\boxed{
\text{v14.7 的 endpoint gain 到底是 FMS-specific，还是 generic optimizer-state reset / momentum suppression？}
}
$$

---

# 5. Line Q：v14.6.1 / v14.7 语义差异审计

## 5.1 目标

解释为什么 v14.6.1 diagnostic 能 9/9，而 v14.7 official / event-only 不能。

## 5.2 必须比较

```text
Q0: v14.6.1 diagnostic replay under old metrics
Q1: v14.6.1 diagnostic replay under v14.7 strict metrics
Q2: v14.7 official every-step reset under v14.7 strict metrics
Q3: v14.7 event-only reset under v14.7 strict metrics
Q4: v14.7 delta_top25 sparse reset under v14.7 strict metrics
```

## 5.3 必须记录字段

```text
run_id
method
reset_timing = every_step / event_only / post_event_window / none
reset_scope = affected / all_fms_roles / random / full / sparse
strict_pass
endpoint_vs_adamw_pass
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
step_time_ratio
memory_ratio
affected_param_fraction
actual_reset_param_fraction
reset_event_count
reset_every_step_count
controls_endpoint_pass_count
```

## 5.4 判断标准

如果 v14.6.1 只在旧 metric 过，v14.7 strict 不过：

```text
R1-DiagnosticMetricInflation
```

如果 every-step 过而 event-only 不过：

```text
R2-ContinuousMomentumSuppressionNotFMSEvent
```

如果 reset controls 也过：

```text
R3-GenericOptimizerStateResetConfound
```

如果 event-only affected 过且 controls fail，才进入 FMS-specific confirmation。

---

# 6. Line G：Generic optimizer-state reset controls

## 6.1 目标

判断 reset gain 是否只是 generic optimizer trick。

## 6.2 必跑 controls

```text
G0-RAT-AdamW
G1-RAT-AdamW-Beta1Zero
G2-RAT-AdamW-Beta1Half
G3-RAT-AdamW-PeriodicMomentReset
G4-RAT-AdamW-EventMatchedRandomReset
G5-RAT-AdamW-FullMomentResetAtFMSIntervalsNoFMS
G6-RAT-AdamW-RMSPropLikeNoMomentum
G7-RAT-AdamW-NoMomentumWarmupThenAdamW
```

说明：

```text
这些不使用 FMS direction。
这些只测试 optimizer-state reset / momentum suppression 是否本身足够。
```

## 6.3 Gate

如果任意 optimizer-only control 达到：

```text
real_dataset_seed_pass_count >= K2_FMS_pass_count
或
endpoint_vs_AdamW = 9/9 且 strict 接近 K2
```

则：

```text
FMS-specific claim = 0
route = R3-GenericOptimizerStateConfound
```

---

# 7. Line S：FMS specificity test

## 7.1 目标

如果 generic controls 不解释 gain，才检查 FMS-specific state transport。

## 7.2 候选

只允许以下预注册候选：

```text
S0-RAT-FMS-NoStateTransport
S1-RAT-FMS-ZeroMomentResetAffectedEventOnly
S2-RAT-FMS-ZeroMomentResetAllFMSRolesEventOnly
S3-RAT-FMS-ProjectedMomentTransportEventOnly
S4-RAT-FMS-RMSRecomputeEventOnly
S5-RAT-FMS-MomentResetWithPostEventRecoveryWindow
```

注意：S5 不是新动作 token。它只测试 bounded recovery window 是否降低 event-only transport 的短期 AUC debt；它不改变 FMS direction。

## 7.3 必须记录

```text
pre_event_moment_cos_with_grad
post_event_moment_cos_with_grad
moment_staleness_delta
moment_norm_over_grad_norm
post_event_recovery_loss_integral
source_vs_best_control
AUCtime_ratio
step_time_ratio
state_transport_cost_ms
specificity_vs_random_reset
specificity_vs_full_reset
```

## 7.4 Gate

S1/S5 只有在下面同时成立时才算 FMS-specific positive：

```text
strict pass count >= 7/9 for exploration；
strict pass count = 9/9 for official；
source_vs_best_control >= 0.005；
AUCtime_ratio <= 1.0；
CEp99/NLL/ECE non-harm；
LineC pass；
step_time_ratio <= 1.25；
random/full/generic reset controls fail。
```

---

# 8. Line A：affected-coordinate identification autopsy

## 8.1 目标

解释为什么 affected mask 过宽，且 top25 稀疏化退化。

## 8.2 不允许做什么

不允许继续扫：

```text
top10 / top25 / top50 / threshold grid；
按 audit metric 选择坐标；
按 dataset/seed 选择坐标；
局部 positive 后改 mask。
```

## 8.3 必须诊断

```text
fms_delta_abs_distribution
fms_delta_norm_by_role
reset_fraction_by_role
source_contribution_by_reset_fraction
moment_staleness_by_role
value_retention_vs_reset_fraction
linec_tail_nonharm_vs_reset_fraction
```

## 8.4 输出结论

只能输出三种结论：

```text
A1-AffectedMaskTooBroadImplementationBug:
  affected set 本应小但实现把几乎所有 role 纳入。

A2-AffectedSetIntrinsicallyBroad:
  FMS 本身影响全局坐标；affected-only reset 不可能稀疏。

A3-SparseAffectedMaskDestroysValue:
  稀疏化本身破坏 FMS value；不能继续 mask sparsification。
```

只有 A1 才允许做实现修复。A2/A3 不允许继续 mask 网格。

---

# 9. Line E：FMS overhead / step-time budget audit

## 9.1 目标

v14.7 的 K2 median step_time_ratio > 1.25。即使机制成立，overhead 不过也不能 official。

## 9.2 必须拆分

```text
per_example_gradient_time
fms_state_update_time
projection_time
state_transport_time
linec_audit_time_excluded
optimizer_update_time
sync_time
artifact_logging_time
```

## 9.3 Gate

不能通过降低 gate。

可接受路径只有：

```text
same mechanism, same direction, same strict results,
step_time_ratio <= 1.25。
```

如果机制不成立，则不做 overhead optimization。

---

# 10. Line M：MLP analog and generic optimizer controls

## 10.1 目标

区分：

```text
FMS/KAN-specific mechanism
vs
MLP/generic optimizer-state reset trick
```

## 10.2 必跑

```text
M0-MLP-AdamW
M1-MLP-FMS-NoStateTransport
M2-MLP-FMS-ZeroMomentResetAffected
M3-MLP-FMS-ZeroMomentResetRandomCoords
M4-MLP-FMS-FullAdamWStateReset
M5-MLP-AdamW-Beta1Zero
M6-MLP-AdamW-PeriodicMomentReset
```

## 10.3 判断

如果 MLP/generic reset 与 Rational-FMS 同样过：

```text
Generic optimizer-state mechanism；
KAN-specific functional claim = 0。
```

如果 Rational-FMS 过而 MLP/generic 不过：

```text
KAN/FMS-specific claim may open。
```

---

# 11. Line D：All-basis substrate parallel

## 11.1 目标

不能因为 Rational/FMS 接近成功就停止其它基函数。RBF/FastKAN、Chebyshev、Fourier、Wavelet 都继续，但不能没过 substrate 就跑 official FMS。

## 11.2 本轮只做 substrate repair / status，不做 Non-RAT official FMS

```text
Rational:
  D-RAT monitor only，防止 regression。

Wavelet:
  support / scale occupancy / local-tail coverage hardening。

Fourier:
  low-frequency identity residual / bandwise SNR / phase-stable mix。

RBF / FastKAN:
  active center occupancy / width guard / identity residual。

Chebyshev:
  low-degree identity residual / high-degree late-enable / degree-energy damping。
```

## 11.3 Substrate gate

```text
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta_vs_MLP >= -0.05
worst_delta_vs_MLP >= -0.10
AUCtime_ratio <= 2.0
LineC_pass_rate >= 0.30
```

过不了 substrate gate：

```text
No official FMS proof。
```

---

# 12. Line C：Manifold-channel / tail audit

Line C 继续作为审计，不作为方向源。

必须记录：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
LineC_pass
CEp99_delta
NLL_delta
ECE_delta
Brier_delta
margin_p10_delta
source_vs_best_control
AUCtime_ratio
```

必须区分：

```text
source fail
tail fail
AUCtime fail
LineC fail
step_time fail
control equivalence fail
specificity fail
```

---

# 13. 必须生成的 artifacts

```text
v148_route_decision.json
v148_progress_table.csv
v148_v1461_v147_semantic_diff.csv
v148_generic_optimizer_reset_controls.csv
v148_fms_specificity_results.csv
v148_affected_mask_autopsy.csv
v148_overhead_breakdown.csv
v148_mlp_analog_controls.csv
v148_all_basis_substrate_status.csv
v148_linec_tail_audit.csv
v148_no_action_search_audit.csv
v148_forbidden_information_audit.csv
v148_required_artifact_manifest.csv
v148_code_review_packet.zip
v148_no_go_boundary.md
v148_next_hypothesis_queue.md
```

---

# 14. 必须生成的可视化

```text
fig_v148_semantic_diff_v1461_v147.svg
fig_v148_specificity_vs_controls.svg
fig_v148_k2_vs_generic_optimizer_reset.svg
fig_v148_affected_fraction_distribution.svg
fig_v148_value_retention_vs_reset_fraction.svg
fig_v148_step_time_waterfall.svg
fig_v148_mlp_vs_rat_reset_controls.svg
fig_v148_all_basis_substrate_status.svg
fig_v148_failure_taxonomy.svg
```

---

# 15. Success / no-go routes

## S5 Official Success

只有以下全部满足才允许：

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
random/full/generic reset controls fail
MLP analog does not explain gain
forbidden audit pass
no-action-search audit pass
code review pass
promotion_allowed = 1
```

## R1-DiagnosticMetricInflation

v14.6.1 只在旧 metric 过，v14.7 strict 不过。

## R2-ContinuousMomentumSuppressionNotFMSEvent

every-step reset 过，event-only reset 不过。

## R3-GenericOptimizerStateResetConfound

optimizer-only / random / full reset controls 解释了 endpoint 或 strict gain。

## R4-AffectedCoordinateIdentificationFail

affected mask 太宽或稀疏化破坏 value，且无合法实现修复。

## R5-OverheadBlocker

mechanism 有 value，但 step_time_ratio > 1.25 且无法低成本实现。

## R6-FMSStateTransportNoGo

specificity、event semantics、affected mask、overhead 都不能闭合。

---

# 16. Codex 执行硬规则

Codex 禁止：

```text
1. 新增 K-RT / K-AUC / K-FL action token；
2. 启动 controller；
3. 调 strength / lambda / lr / refresh 网格；
4. 用 audit metric 生成 direction；
5. 按 dataset / seed 写 branch；
6. 把 endpoint-only 写成 strict；
7. 把 random/full reset controls 也能复现的结果写成 FMS-specific；
8. 把未执行的 Non-RAT 写成成功；
9. 拼接不同 run 的局部 positive；
10. 降低 step_time gate。
```

Codex 必须先执行：

```text
1. v14.6.1 / v14.7 semantic diff；
2. generic optimizer reset controls；
3. FMS specificity test；
4. affected mask autopsy；
5. overhead breakdown；
6. MLP analog controls；
7. all-basis substrate status；
8. final route / no-go boundary / next hypothesis。
```

---

# 17. 最终判断

v14.7 的结果不是“没有意义”，而是及时阻止了一个危险误判：

$$
\boxed{
\text{v14.6.1 的 zero-moment reset diagnostic 不能直接升级为 official FMS success。}
}
$$

下一步不能继续寻找好动作，也不能继续重演 v9/v12 的 action-search 长征。v14.8 必须用决定性实验回答：

$$
\boxed{
\text{FMS-state transport 是否有 KAN/FMS-specific value，}
\text{还是只是 generic optimizer-state reset / momentum suppression confound？}
}
$$

如果是后者，functional update 主线必须从 optimizer-state reset route 撤出，回到 FMS definition / substrate / all-basis parallel repair，而不是继续用 action 或 reset 小修拖延。
