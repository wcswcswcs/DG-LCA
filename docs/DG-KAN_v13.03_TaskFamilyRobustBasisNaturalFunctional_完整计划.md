# DG-KAN v13.3：Task-Family Robust Basis-Natural Functional Update + Non-RAT Substrate Rescue 完整计划

> 版本：v13.3 execution plan  
> 日期：2026-05-27  
> 目标：停止 BN4/BN5/BN6/BN7 局部网格与 readout-feature proxy 路线，转向 **task-family robust basis metric** 与 **真实 basis-parameter update**；并行推进 Non-RAT substrate rescue。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；不使用 label-informed initialization；functional direction 可通过统一 loss interface 接收 output cotangent，但不能 hardcode CE 公式，不能使用 CEp99 / NLL / ECE / LineC hard target / validation / test / future outcome / query batch 生成方向。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的最终目标不是找一个 MNIST-family 上好看的技巧，也不是只证明某个 basis 能跑。项目总目标是：

$$
\boxed{
\text{构造 label-free strict FC-PureKAN substrate/base，}
\text{并通过 loss-interface-generic functional update 获得比 ordinary backprop 更好的训练几何与模型。}
}
$$

最终需要证明：

$$
\boxed{
\text{PureKAN substrate/base + functional update}
>
\text{same substrate/base + ordinary AdamW / matched controls}
}
$$

同时必须满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹不差，AUC-step / AUC-time 不坏；
4. 几何更健康：CouplingR2、NoiseSignalLeak、RealSignalReservoirRatio、tail、calibration 不坏；
5. functional update 的收益必须打过 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / MLP analog controls；
6. functional update 不能是 CE-specific optimizer trick。
```

## 0.2 当前阶段性结论

v13.2 第一次把 functional update 从 readout-feature proxy 推进到真实 basis-parameter update：

```text
full_basis_param_update_rows = 84
readout_feature_proxy_only = 0
```

这是一条真实进展。但 v13.2 的最终可信 route 仍是：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
official_success_reached = 0
promotion_allowed = 0
basis_natural_p3_pass_count = 2
synthetic_task_success_count = 2
synthetic_5of7_pass = 0
real_short_run_pass_count = 0
mlp_analog_pass_count = 0
```

也就是说：

$$
\boxed{
\text{真实参数级 functional 已经能在 X7 局部打开，}
\text{但不能跨 synthetic task family 泛化。}
}
$$

当前不能进入 real short-run，也不能 promotion。

## 0.3 本轮核心判断

v13.2 排除了很多旧解释：

```text
1. 不是 readout-feature proxy 问题：真实 basis-param writeback 已执行。
2. 不是 update norm / clipping 超参传播问题：已修复并重跑。
3. 不是缺少 optimizer-state transport：scale=0.20 / 1.0 都未扩大 coverage。
4. 不是缺少 BN4/BN5/BN6/BN7 局部 metric 变体：全部无法突破 2/21 coverage。
5. 不是单纯 label-free output-Jacobian diagonal metric 缺失：BN7 也失败。
```

因此 v13.3 不再继续 BN4/BN5/BN6/BN7 局部小修，而是重置问题：

$$
\boxed{
\text{当前 diagonal/local basis metric 只能解决一个 task family，}
\text{我们需要 task-family robust 的低秩/非对角 basis metric。}
}
$$

---

# 1. 各条线当前进展百分比

这些百分比是基于 gate 通过情况、artifact 完整性、机制清晰度和离 official success 的距离做出的估计，不是 runner 的官方字段。

| 线 | 上次 v13.0 后估计 | 当前 v13.2 后估计 | 变化 | 判断 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | 99% | 0 | artifact 完整，provenance clean；继续要求 manual review。 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 85% frozen | 0 | 历史强，但 label-informed init 禁用，不能 official。 |
| Label-free FHQ / A-DYN monitor | 23% | 22% | -1 | 本轮未主推，仍无 near-anchor。 |
| Line C：几何审计 | 88% | 88% | 0 | 审计稳定，但仍不能作为方向源。 |
| Rational substrate | 85% | 85% | 0 | 仍是唯一稳定 S1 substrate family。 |
| Non-RAT substrate | 38%-45% | 35%-42% | -3 | 本轮未 rescue；仍需要真实 S1/S2 substrate。 |
| Function-preserving readout-feature proxy | 20% | 10% | -10 | v13.2 已证明重点应转向真实参数级，不再投 proxy。 |
| Full basis-parameter writeback implementation | 0% | 55% | +55 | 真实 basis-param update 已落盘；不是 feature-table proxy。 |
| Basis-natural functional mechanism | 5% | 18% | +13 | 达到 S2：2 个 P3 pass；但只在 X7，未过 synthetic 5/7。 |
| Synthetic mechanism proof | 0% | 25% | +25 | X7 打开，但 X1-X6 不开；还不是可推广机制。 |
| MLP analog control | 20% | 20% | 0 | 已执行，mlp_analog_pass_count=0；没有 generic positive。 |
| Functional official / real short-run | 0%-5% | 0%-5% | 0 | real short-run 不允许打开。 |
| 整体 next-gen MLP claim | 45%-49% | 43%-48% | -1 到 -2 | 真实参数级进展存在，但机制覆盖过窄，官方路线仍远。 |

---

# 2. v13.2 结果独立分析

## 2.1 有进展，但不是 S5

v13.2 最关键的正进展是：

```text
full_basis_param_update_rows = 84
basis_natural_p3_pass_count = 2
synthetic_task_success_count = 2
```

这说明我们终于不是只在 frozen feature table 或 readout proxy 上做功能更新，而是对真实 basis 参数做了写回。这个进展应该承认。

但它不是 official success。P3 pass 只来自：

```text
D-RAT26 / D-RAT27
update = BN5
loss = Brier
task = X7
```

典型 official pass row：

```text
D-RAT26, X7, BN5, Brier:
  source_vs_best = +0.011487
  CouplingR2_delta = +1.973688
  NoiseSignalLeak_delta = -0.004590
  Reservoir_delta = -0.137204
  CEp99_delta = -2.225936

D-RAT27, X7, BN5, Brier:
  source_vs_best = +0.009256
  CouplingR2_delta = +4.897928
  NoiseSignalLeak_delta = -0.003060
  Reservoir_delta = -0.411656
  CEp99_delta = -2.750546
```

这说明 BN5 在某个 synthetic family 上确实能同时改善 task proxy、coupling、reservoir 和 tail；但它没有跨 synthetic family 泛化。

## 2.2 为什么这不是小成功，而是“局部机制”

v13.2 的 gate 要求 synthetic proof 至少达到 5/7，但现在只有 2/21 row 通过，且都集中在 X7。更低 future LR、optimizer-state transport、BN6 consensus clipped metric、BN7 output-Jacobian diagonal metric 都没有扩大 coverage。

所以当前结论应该写成：

$$
\boxed{
\text{BN5 证明存在一个局部 basis-natural functional signal，}
\text{但当前 diagonal/local metric 不是 task-family robust。}
}
$$

## 2.3 这次真正发现的问题

过去我们以为问题是：

```text
functional update 没有真实写回；
或者 metric 太粗；
或者缺少 optimizer-state transport；
或者缺少 output-Jacobian metric。
```

v13.2 之后，更准确的问题是：

$$
\boxed{
\text{当前 functional metric 只捕捉到某一种 synthetic family 的有用 tangent，}
\text{没有学到跨任务族稳定的 basis-natural geometry。}
}
$$

这不是继续调 BN5 scale 可以解决的问题。

---

# 3. 当前核心 blocker

## 3.1 Blocker A：diagonal/local metric 不够

BN1/BN2/BN3 是对角/局部的 natural update；BN4/BN5 增加了 logit-energy orthogonal safety；BN6 做 consensus clipped；BN7 用 output-Jacobian diagonal metric。它们都没有突破 2/21 coverage。

这说明：

$$
M_{basis}=\operatorname{diag}(m)
$$

或近似 diagonal 的 local metric 不够。我们需要：

$$
M_{basis}=D + UU^T
$$

或者 block / low-rank / family-aware metric，让多个 basis channel 的耦合能被表示出来。

## 3.2 Blocker B：synthetic proof 目标可能太单点化

如果 X7 是唯一打开的 family，说明当前 synthetic proof 可能在逼迫 update 过度适配某个局部任务形态。下一步必须做两件事：

```text
1. 对 X1-X7 进行 task-family autopsy；
2. 区分：metric 真的不会泛化，还是某些 synthetic target 与真实目标无关。
```

但不能简单降低 5/7 gate。应该做 leave-family-out：

```text
candidate selection 不看某一类 task；
heldout family 仍必须不崩。
```

## 3.3 Blocker C：Non-RAT substrate 缺席让结论过度 Rational-specific

当前 S1 substrate 仍然只来自 D-RAT。BN5/X7 信号可能是 Rational 的局部结构，而不是 KAN basis 的普遍能力。下一轮必须并行推进 Non-RAT S1/S2 substrate rescue，否则 functional update 只会在 Rational 内部绕圈。

## 3.4 Blocker D：real short-run 还不能打开

因为没有 synthetic 5/7，也没有 MLP analog pass，所以 real short-run 关闭是正确的。现在不能为了“看进度”强开真实任务 P4。

---

# 4. v13.3 核心假设

## H1：task-family robust basis metric 需要非对角 / 低秩结构

假设：BN5 只在 X7 成功，是因为 diagonal metric 只能做 per-parameter local scaling，不能捕捉跨 basis group 的协同方向。使用低秩或 block metric 可以扩大 synthetic coverage。

候选 metric：

$$
M = D + UU^T + \rho I.
$$

其中 $U$ 来自 loss-interface gradient sketch、basis tangent sketch、random cotangent response 或 family-balanced synthetic probes，但不能使用 CEp99 / NLL / ECE / LineC hard target。

## H2：basis-natural update 应该用 task-family balanced objective，而不是 single-row P3

当前 source-vs-control + LineC hard gate 是必要的，但 candidate selection 必须避免被 X7 单一 family 支配。需要记录每个 synthetic family 的 success / failure pattern，并做 family-balanced score。

## H3：Non-RAT substrate rescue 是必要并行线

如果 Non-RAT 仍无 S1/S2，functional 的结论会被 Rational 绑定。下一步必须至少尝试让 Chebyshev / Fourier / RBF / Wavelet 中一个 family 形成 S1 controllable substrate。

## H4：MLP analog 仍是必要控制，但不再扩通用 MLP objective

MLP analog 在 v13.2 没有 pass。下一步只做最小 closure，不再主投 MLP functional。它用于判断：如果低秩 metric 在 MLP analog 也 work，则不是 KAN-specific；如果只在 basis substrate work，才可能是 KAN basis coordinate advantage。

---

# 5. v13.3 实验总结构

```text
Line R：Implementation / provenance / code review
Line S：Substrate map update and Non-RAT rescue
Line G：Task-family gradient / tangent autopsy
Line M：Task-family robust basis metric
Line X：Synthetic mechanism proof v2
Line P：Real short-run gate, only if X passes
Line A：MLP analog minimal closure
Line Z：Finalizer / no-go boundary / next hypothesis queue
```

---

# 6. Line R：Implementation / provenance / code review

## 6.1 目标

确认 v13.3 仍是真实 basis parameter update，不退回 feature proxy。

必须记录：

```text
full_basis_param_update_executed
readout_feature_proxy_only
feature_table_proxy_only
basis_param_writeback_count
basis_param_writeback_norm
basis_param_names
optimizer_state_written
optimizer_state_transport_mode
loss_interface_type
uses_ce_specific_formula
uses_label_for_direction
uses_validation_or_test_for_direction
uses_linec_hard_target_for_direction
uses_dataset_name_branch
```

## 6.2 硬门

```text
full_basis_param_update_executed = 1
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
uses_label_for_direction = 0
uses_validation_or_test_for_direction = 0
uses_linec_hard_target_for_direction = 0
```

若不满足，route：

```text
R0-NotTrueBasisFunctional
```

---

# 7. Line G：Task-family gradient / tangent autopsy

## 7.1 目标

解释为什么 BN5 只在 X7 成功，X1-X6 失败。

## 7.2 实验对象

```text
substrates: D-RAT26, D-RAT27, D-RAT28
updates: BN1, BN2, BN3, BN5, BN7
loss interfaces: CE, Brier
tasks: X1..X7
seeds: 0,1,2 if budget allows
```

## 7.3 必须记录

```text
task_family
substrate_id
update_id
loss_interface
source_vs_best_control
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
update_norm_ratio
basis_metric_condition
basis_metric_diag_min
basis_metric_diag_max
basis_metric_offdiag_energy
per_example_grad_snr_mean
per_example_grad_snr_p10
per_example_grad_snr_p90
cos_update_adamw
cos_update_snr
cos_update_random
cos_update_x7_reference
safety_projection_rejected_fraction
fail_pattern
```

## 7.4 可视化

```text
fig_v133_task_family_success_matrix.svg
fig_v133_gradient_snr_by_task.svg
fig_v133_update_cosine_heatmap.svg
fig_v133_metric_condition_by_task.svg
fig_v133_failure_pattern_sankey.svg
```

## 7.5 判断

如果 X1-X6 failure 与 X7 update cosine 高度相反，说明 BN5 学到的是 X7-specific direction。

如果 X1-X6 failure 与 metric condition / low SNR 相关，优先做 low-rank / block metric。

如果 X1-X6 source_vs_best 正但 LineC fail，优先修 safety projection。

---

# 8. Line M：Task-family robust basis metric

## 8.1 目标

从 diagonal/local metric 升级到 low-rank / block / family-balanced metric。

## 8.2 候选

```text
BM8-LowRankTangentNatural-r4
BM9-LowRankTangentNatural-r8
BM10-BlockGroupNatural
BM11-TaskFamilyBalancedMetric
BM12-LeaveFamilyOutMetric
BM13-KroneckerGroupNatural
BM14-TrustRegionLowRankNatural
BM15-ControlResidualizedNatural
```

## 8.3 公式

低秩 metric：

$$
M = D + UU^T + \rho I.
$$

Woodbury inverse：

$$
M^{-1}g
=
D^{-1}g
-
D^{-1}U(I+U^TD^{-1}U)^{-1}U^TD^{-1}g.
$$

Family-balanced selection score：

$$
S_{fam}
=
\frac{1}{7}\sum_{k=1}^7 I_k
-
\lambda \operatorname{Std}(I_1,...,I_7),
$$

where $I_k$ is the pass indicator or normalized margin for task family $X_k$.

## 8.4 必须记录

```text
metric_id
rank
block_size
rho
woodbury_condition
metric_compute_ms
metric_memory_mb
update_compute_ms
update_memory_mb
family_balanced_score
leave_family_out_pass_rate
synthetic_pass_count
synthetic_task_success_count
```

## 8.5 Gate

Exploratory pass：

```text
synthetic_task_success_count >= 4/7
no single task family contributes > 50% of pass rows
source_vs_best_control mean > 0
NoiseSignalLeak non-worse in >= 5/7
ReservoirRatio non-worse in >= 5/7
```

Official synthetic pass：

```text
synthetic_task_success_count >= 5/7
leave_family_out_pass_rate >= 0.70
real_short_run_open_allowed = 1
```

---

# 9. Line S：Substrate map update and Non-RAT rescue

## 9.1 目标

不要让 Rational 成为唯一 substrate。至少尝试一个 Non-RAT family 进入 S1 controllable substrate。

## 9.2 Candidate focus

```text
Chebyshev:
  degree-energy damped substrate, recurrence lifetime repair

Fourier:
  low-frequency + band guard substrate, no high-frequency noise-heavy path

RBF/FastKAN:
  compact active-center expansion with width condition guard

Wavelet:
  hat/triangle local wavelet with scale occupancy guard
```

## 9.3 Substrate S1 gate

```text
workspace_raw_ratio <= 1.10
workspace_incremental_ratio <= 2.00
step_ratio <= 1.75
mean_delta_vs_MLP >= -0.05
worst_delta_vs_MLP >= -0.10
AUCtime_ratio_vs_MLP <= 2.0
LineC_pass_rate >= 0.30
```

## 9.4 Failure fallback

If workspace passes but task/AUC collapse：

```text
reduce capacity only if LineC improves;
try late-enable high-complexity branch;
add basis energy cap;
rerun short task-health probe.
```

If task passes but workspace fails：

```text
lifetime waterfall;
readout grad recompute;
basis derivative recompute;
optimizer state delayed allocation.
```

If expression fails：

```text
compact capacity repair;
do not go dense;
rerun expression battery before task.
```

---

# 10. Line X：Synthetic mechanism proof v2

## 10.1 目标

验证 basis-natural update 是否跨 task family work。

## 10.2 Tasks

```text
X1 pairwise product
X2 rotated quadratic
X3 random quadratic
X4 low/high frequency mixture
X5 local bump / tail cluster
X6 two-manifold class split
X7 existing pass family / stress family
```

如果当前代码中 X1..X7 名称不同，Codex 必须在 implementation readback 中写出真实含义和生成公式。

## 10.3 Controls

```text
NoOp
RandomMatchedNorm
AdamWParallel
SNROnly
ShuffledCotangent
DiagonalBN5 old reference
MLP analog
```

## 10.4 Gate

Row pass：

```text
source_vs_best_control >= 0.005
CouplingR2_delta >= 0.02
NoiseSignalLeak_delta <= 0
RealSignalReservoirRatio_delta <= 0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
```

Task-family pass：

```text
>= 2 substrates or >= 2 seeds pass for the family
```

Synthetic mechanism pass：

```text
>= 5/7 task-family pass
```

Only then open Line P.

---

# 11. Line P：Real short-run gate

## 11.1 Opening condition

```text
synthetic_5of7_pass = 1
full_basis_param_update_executed = 1
MLP analog not explaining all benefit
no forbidden information violation
```

## 11.2 Real short-run setup

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 3 and 8
base = best S1/S2 substrate
updates = top 2 basis-natural metrics
controls = NoOp, RandomMatchedNorm, AdamWParallel, SNR-only
```

## 11.3 Metrics

```text
accuracy delta vs base
source_vs_best_control
val_loss_auc_step
val_loss_auc_time
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
NLL
ECE
Brier
margin_p10
step_ratio
memory_ratio
functional_overhead
```

## 11.4 Gate

```text
source_vs_best_control >= 0.005 macro
AUCtime_delta <= 0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC majority pass >= 2/3 datasets
no dataset-specific branch
```

---

# 12. Line A：MLP analog minimal closure

## 12.1 目标

只做最小 closure。不要继续扩 MLP functional 小网格。

## 12.2 MLP analogs

```text
MLP-A1 hidden natural diagonal
MLP-A2 hidden low-rank tangent metric
MLP-A3 weight-balanced natural update
```

## 12.3 判断

If MLP analog also passes synthetic 5/7：

```text
KAN-specific claim = 0
generic reparameterization claim possible
```

If MLP analog fails while KAN passes：

```text
KAN-specific basis-coordinate advantage possible
```

If both fail：

```text
current metric family no-go
```

---

# 13. Required artifacts

Codex 必须落盘：

```text
v133_route_decision.json
v133_progress_table.csv
v133_code_review_manifest.csv
v133_basis_param_manifest.csv
v133_writeback_trace.csv
v133_loss_interface_audit.csv
v133_forbidden_info_audit.csv
v133_task_family_autopsy.csv
v133_gradient_snr_by_task.csv
v133_metric_condition_by_task.csv
v133_lowrank_metric_rows.csv
v133_synthetic_mechanism_v2.csv
v133_synthetic_family_summary.csv
v133_nonrat_substrate_rescue.csv
v133_mlp_analog_closure.csv
v133_real_short_run.csv, if opened
v133_failure_table.csv
v133_no_go_boundary.md
v133_next_hypothesis_queue.md
```

Required figures：

```text
fig_v133_progress_vs_previous.svg
fig_v133_task_family_success_matrix.svg
fig_v133_failure_pattern_sankey.svg
fig_v133_lowrank_metric_pareto.svg
fig_v133_nonrat_substrate_status.svg
fig_v133_synthetic_to_real_gate_ladder.svg
fig_v133_control_gap_dashboard.svg
```

---

# 14. Stop / continue rules

## Continue rules

If BN-family still only passes X7：

```text
must run Line G autopsy and Line M low-rank/block metric;
not allowed to stop after restating X7-only pass.
```

If low-rank metric reaches >= 4/7 but not 5/7：

```text
must run leave-family-out analysis;
must identify failing families;
must run one targeted metric repair not based on CE/NLL/ECE/LineC hard target.
```

If Non-RAT has no S1：

```text
must run at least one lifetime repair and one task-health repair for the best Non-RAT family;
not allowed to write Non-RAT rejected without blocker type.
```

## Stop rules

Final stop allowed only if：

```text
1. synthetic 5/7 pass and real short-run completed；or
2. low-rank/block/family-balanced metric all fail to exceed 2/7 coverage；and
3. Non-RAT rescue fails S1；and
4. MLP analog closure is complete；and
5. no-go boundary explains whether to pivot to substrate design or redefine synthetic proof.
```

---

# 15. Expected outcomes

## Success route

```text
S3-TaskFamilyRobustBasisNatural
synthetic_5of7_pass = 1
real_short_run_open = 1
```

## Partial route

```text
R3a-LowRankMetricPartial
synthetic_success_count >= 4/7 but real not opened
```

## No-go route

```text
R4-DiagonalAndLowRankMetricNoGo
coverage <= 2/7 after BN1-BN7 and BM8-BM15
```

## Pivot route

```text
R5-SubstrateDesignRequired
Rational-only substrate remains and Non-RAT rescue fails
```

---

# 16. 最终一句话

v13.2 的意义是：**真实 basis-parameter functional update 已经执行，并在 X7 打开局部 P3；但当前 diagonal/local basis metric 不具备 task-family robustness。**  

v13.3 的任务不是继续调 BN5，而是验证：

$$
\boxed{
\text{低秩/非对角/任务族平衡的 basis metric 能否把局部 X7 信号扩展成跨任务族机制。}
}
$$
