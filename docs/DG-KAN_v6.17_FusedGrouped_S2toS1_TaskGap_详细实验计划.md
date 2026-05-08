# DG-KAN v6.17：Fused Grouped S2 → S1/S0、Efficiency Margin Repair 与 Diagnostic Task Gap 详细实验计划

> 本计划基于 v6.16 final real-only run 制定。v6.16 是一个重要拐点：它首次真实实现 Triton fused grouped kernel，并得到 Reset 系列第一个 S2 near-pass candidate：`GT4-best-fused-no-materialize`。该候选达到 `memory ratio mean = 1.0312`、`step ratio mean = 1.4426`、`backward ratio mean = 0.7281`，one-step loss probe 通过，且 diagnostic task 已真实运行。但它仍不是正式成功：因为 memory 仍高于 `1.00`，step 仍高于 `1.35`，因此没有达到 S0/S1 official task-open gate；diagnostic task accuracy 也明显落后 MLP。因此 v6.17 的目标是：把 fused grouped S2 推到 S1/S0，并解释和修复 diagnostic task gap。

---

## 0. 实验整体目标

v6.17 的整体目标不是重新搜索 DWM2，不是回到 low-rank/piecewise，也不是直接做 optimizer 大扫。v6.17 的主线是围绕 v6.16 的真实 S2 survivor：

```text
GT4-best-fused-no-materialize
```

完成两个问题的闭环。

第一个问题：

$$
\boxed{
\text{能否把 GT4 从 S2 near-pass 推到 S1/S0，从而打开 official task re-entry？}
}
$$

第二个问题：

$$
\boxed{
\text{GT4 diagnostic task accuracy 低于 MLP 的原因是表达力、优化器、训练预算，还是 efficiency margin 不足导致的限制？}
}
$$

v6.16 的关键事实如下：

```text
GT4-best-fused-no-materialize:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.4426
  backward ratio mean = 0.7281
  forward ratio mean  = 1.1425
  grad relerr max     = 1.92e-08
  survivor type       = S2
  one-step probe      = pass
```

S2 的定义是：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

GT4 已经满足 S2，但还没有满足 S1/S0。正式 task-open gate 要求：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

因此 GT4 距离 S1/S0 的工程缺口是：

$$
1-\frac{1.00}{1.0312}\approx3.03\%
$$

的 memory reduction，以及：

$$
1-\frac{1.35}{1.4426}\approx6.42\%
$$

的 step-time speedup。

v6.16 diagnostic task 结果如下：

```text
MLP-autograd-reference:
  val acc mean  = 0.6981
  test acc mean = 0.6523

MLP-manual-linear-reference:
  val acc mean  = 0.6762
  test acc mean = 0.6402

GT4 + ManualAdamW:
  val acc mean  = 0.6224
  test acc mean = 0.5933

GT4 + ManualAdanLite:
  val acc mean  = 0.6302
  test acc mean = 0.5961
```

GT4+ManualAdanLite 相对 MLP-autograd 的 gap 是：

$$
\Delta Acc_{\text{val}} = 0.6302-0.6981=-0.0679,
$$

$$
\Delta Acc_{\text{test}} = 0.5961-0.6523=-0.0562.
$$

因此 v6.17 不能只做 efficiency。它必须同时做：

```text
1. S2 -> S1/S0 efficiency margin repair
2. diagnostic task gap attribution
3. lightweight expressivity/optimization repair under the fused grouped efficiency envelope
```

v6.17 的最低成功目标是得到一个真实 S1 candidate：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35,
$$

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

v6.17 的 task-side最低成功目标是：在 official task re-entry 或 diagnostic task 中，best fused grouped candidate 至少满足：

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.02
$$

on all datasets, and diagnostic trend must improve over v6.16 GT4 baseline.

v6.17 的理想目标是同时满足：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35,
$$

and:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.01.
$$

---

## 1. 当前实验进展与问题判断

### 1.1 已经真正突破的部分

v6.16 首次完成了前几轮没有做到的事情：

```text
real Triton fused grouped kernel implemented
gradient relerr fixed to 1.92e-08 by using input_precision="ieee"
memory/runtime tradeoff 被真实打破
GT4-best-fused-no-materialize 达到 S2
one-step loss probe 通过
diagnostic task trace 真实运行
```

这意味着当前项目已经从：

```text
没有 memory/time survivor
```

推进到：

```text
有 S2 near-pass survivor，但还没有 S1/S0 official task-open survivor
```

因此 v6.17 不应回到 DWM2、low-rank 或 piecewise 大搜索。主线应继续 fused grouped。

### 1.2 当前没有正式成功的原因

v6.16 没有正式成功，有两个原因。

第一，efficiency 还差一小段 margin：

```text
memory ratio mean = 1.0312 > 1.00
step ratio mean   = 1.4426 > 1.35
```

所以 official task re-entry 仍关闭。

第二，diagnostic task accuracy 明显低于 MLP：

```text
GT4+ManualAdanLite val acc  = 0.6302
MLP-autograd val acc        = 0.6981

GT4+ManualAdanLite test acc = 0.5961
MLP-autograd test acc       = 0.6523
```

虽然 GT4 diagnostic task 的 train loss delta 更大：

```text
GT4+ManualAdanLite train loss delta mean = -2.5734
MLP-autograd train loss delta mean       = -2.3124
```

但 validation/test accuracy 低，说明问题可能不是“完全学不动”，而是以下之一：

```text
表达力不足
跨 group mixing 不够
优化器 / regularization 不适配
短 budget 下过拟合或 margin 不好
feature rank / class separation 不足
calibration / logit geometry 不好
```

### 1.3 当前问题定位

当前核心问题已经不是：

```text
fake/proxy
gradient correctness
manual backward
DWM2 attribution
grouped memory/runtime tradeoff
```

这些问题在 v6.16 中已经大幅解决或被清楚冻结。

当前真正问题是：

$$
\boxed{
\text{如何把 GT4 的 S2 efficiency margin 推到 S1/S0，同时让 task performance 接近 MLP。}
}
$$

---

## 2. v6.17 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。所有未实现内容必须写成：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，DWM2 继续冻结。v6.17 不允许新增 DWM2 patch。

第三，不允许回到 Python group loop、torch tile sweep、low-rank/piecewise 大搜索作为主线。它们只能作为 reference 或 fallback，不允许抢占 fused grouped 主线。

第四，不允许把 S2 说成 official task success。只有 S0/S1 或明确 plan 允许的 diagnostic task 可以运行。

第五，不允许把 diagnostic task accuracy 低归因于 optimizer，除非记录了 feature rank、margin、train/val gap、classwise acc、ECE、loss curve 和 update statistics。

第六，不允许只优化 memory 而牺牲 step time。v6.17 的效率目标是双门槛：

$$
r_{\text{mem}}<1.00,\quad r_{\text{step}}\leq1.35.
$$

第七，不允许引入 non-KAN trainable parameters。任何 cross-group correction、low-rank correction、residual enhancer 都必须保持：

```text
nonKAN_param_count = 0
edge-owned parameters only
manual backward available
```

第八，没有 S0/S1 前，P8 official task re-entry 不能打开；但 S2 candidate 可继续 diagnostic task，前提是明确标记 diagnostic-only。

第九，不允许根据 diagnostic task 的 test result 反复调参后再宣称 test success。P5/P6 只能用 train/val/holdout 做选择；test 只能报告，不作为调参依据。

---

## 3. 核心假设

### H1：GT4 离 S1/S0 主要差小幅 memory margin 和 forward/update overhead

GT4 的 backward ratio 已经明显低于 MLP：

$$
r_{\text{backward}}=0.7281.
$$

但 step ratio 仍为：

$$
r_{\text{step}}=1.4426.
$$

同时 forward ratio 为：

$$
r_{\text{forward}}=1.1425.
$$

H1 假设：step 未到 $1.35$ 的主要原因不是 backward，而是 forward fused path、launch overhead、update prep、optimizer update 或 measurement overhead。H1 成立要求 P1/P2 显示：

```text
backward ratio remains <= 0.80
forward/update/launch components explain >= 50% of remaining step gap
```

定义 remaining step gap：

$$
G_{\text{step}}=r_{\text{step}}-1.35.
$$

如果 forward/update/launch component 对 $G_{\text{step}}$ 的解释比例为 $C_{\text{fwd/update}}$，则 H1 成立要求：

$$
\frac{C_{\text{fwd/update}}}{G_{\text{step}}}\geq0.50.
$$

H1 的修复目标是：

$$
r_{\text{step,new}}\leq1.35.
$$

---

### H2：GT4 memory 从 1.0312 到 <1.00 的缺口来自可修的小型 live-set 或 allocator overhead

GT4 memory 已经非常接近：

$$
r_{\text{mem}}=1.0312.
$$

H2 假设：剩余 $3.03\%$ memory gap 不需要换 primitive，可能来自：

```text
materialized grouped output buffer
grad_mix temporary
grad_poly temporary
optimizer/update temporary
manual cache bookkeeping
allocator reserved/allocated mismatch
unreleased scratch buffer
```

H2 成立要求 P1 attribution 能把 KAN-over-MLP residual memory gap 的 top sources 分解清楚，并且至少一个 source 是可修对象。

H2 修复目标：

$$
r_{\text{mem,new}}<1.00.
$$

如果 memory 降到 `<1.00` 但 step 变差超过 $5\%$，该 variant 只能作为 diagnostic，不作为 S1。

---

### H3：v6.16 diagnostic task gap 可能来自 grouped g16 表达力不足或跨 group mixing 不足

GT4 是 grouped g16 fused path。高效来自 group structure，但 group structure 可能限制跨 group mixing。H3 假设 diagnostic task accuracy gap 主要来自 representational bottleneck，而不是 optimizer 完全失败。

H3 成立的证据包括：

```text
feature effective rank 低于 MLP
class centroid separation 低于 MLP
margin_p10 低于 MLP
classwise accuracy 在类似类别上集中失败
residual ablation 显示 residual 有贡献但 cross-group mixing 不足
train loss 下降但 val/test accuracy 不跟
```

H3 的候选修复是轻量 cross-group correction，例如：

```text
edge-owned cross-group rank1 correction
grouped g8/g16 hybrid
periodic cross-group shuffle without trainable non-KAN params
tiny edge-owned low-rank cross-group residual
```

任何修复必须保持 S2，最好保持 S1：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

进入 official task 的修复必须满足：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

---

### H4：GT4 task gap 也可能来自 optimizer / regularization 不适配，而不是 primitive 表达力不足

v6.16 diagnostic task 中 ManualAdanLite 比 ManualAdamW 稍好，但仍低于 MLP。H4 假设：fused grouped path 需要不同的 optimizer hyperparameters 或 regularization，例如：

```text
lower weight decay
group-aware gradient clipping
residual scale warmup
longer warmup
AdanLite beta tuning
learning rate schedule
label smoothing diagnostic
```

H4 成立要求 optimizer-only repair 在不改变 model efficiency 的情况下改善 validation accuracy：

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to v6.16 GT4+ManualAdanLite baseline, and no dataset shows worse generalization gap.

Optimizer repair 不是 v6.17 主线，但可以作为 diagnostic task repair。

---

### H5：S2 diagnostic task 可以继续，但 official task 必须等 S1/S0

H5 是门禁假设。v6.16 已经证明 S2 candidate 可以运行 diagnostic task，但 official task-open 仍应严格要求 S1/S0。

因此 v6.17 的 rule 是：

```text
S2 candidate:
  one-step probe + diagnostic task allowed

S1/S0 candidate:
  official 3-seed task re-entry allowed

No S2:
  no task
```

---

### H6：如果 v6.17 不能把 GT4 推到 S1/S0，但 task diagnostic 大幅改善，则下一轮继续 fused grouped；否则应进入 expressivity redesign

H6 是 route 假设。如果 v6.17 只得到 S2，但 task diagnostic 从 v6.16 的 `val acc 0.6302` 提升到接近 MLP，例如：

$$
\operatorname{Acc}_{val}\geq\operatorname{Acc}_{MLP}-0.02,
$$

则 fused grouped 仍值得继续。如果 efficiency 保持 S2 但 task gap 仍大于 $5\%$，则下一轮必须做 expressivity redesign，而不是只继续 kernel optimization。

---

## 4. 实验阶段总览

v6.17 分为十二个阶段：

```text
P0: v6.16 reproduction and fused grouped contract
P1: GT4 efficiency margin attribution
P2: S2 -> S1/S0 memory and step repair
P3: fused grouped candidate selection
P4: diagnostic task gap attribution
P5: lightweight expressivity repair under efficiency envelope
P6: optimizer / regularization diagnostic under efficiency envelope
P7: one-step probe for selected candidates
P8: official task re-entry gate
P9: diagnostic task or official task execution
P10: optimizer exploration remains gated unless official task passes
P11: route decision
P12: artifact and failure audit
```

P0-P3 是 efficiency margin 主线。P4-P6 是 task gap diagnosis。P7-P9 只在 S2/S1/S0 candidate 存在时运行。

---

## 5. P0：v6.16 reproduction and fused grouped contract

### 5.1 目的

确认 v6.17 与 v6.16 baseline 可比，确认 GT4 fused grouped path 仍是 no-fake、no-proxy、strict PureKAN、manual backward、Triton fused kernel。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-current-frozen-baseline
GT4-best-fused-no-materialize-v616
GT0-best-P2-fused-grouped-v616
FG5-fused-grouped-forward-backward-g16
FG6-fused-grouped-forward-backward-no-materialize
FG9-single-kernel-grouped-mix
```

如果新变体已实现，还包括：

```text
GT4-memory-margin-v1
GT4-step-margin-v1
GT4-s1-candidate
GT4-crossgroup-lite-r1
GT4-crossgroup-lite-r2
```

### 5.3 必须记录字段

```text
variant
family
status
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
uses_triton_kernel
manual_forward_available
manual_backward_available
manual_update_available
nonKAN_param_count
edge_param_count
residual_param_count
mixing_param_count
rollback_max_error
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
v616_memory_ratio_mean
v617_memory_ratio_mean
v616_step_ratio_mean
v617_step_ratio_mean
v616_backward_ratio_mean
v617_backward_ratio_mean
v616_forward_ratio_mean
v617_forward_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
```

### 5.4 判断标准

P0 reproduction pass:

$$
|r_{\text{mem,v617}}-r_{\text{mem,v616}}|\leq0.03.
$$

$$
|r_{\text{step,v617}}-r_{\text{step,v616}}|\leq0.08.
$$

Contract pass:

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0
manual_backward_available = 1
uses_loss_backward = 0
rollback_max_error < 1e-8
grad_relerr < 1e-4
grad_cos > 0.999
```

### 5.5 可视化

```text
p0_reproduction_delta_bar.svg
p0_contract_heatmap.svg
p0_fused_grouped_lineage_table.md
```

---

## 6. P1：GT4 efficiency margin attribution

### 6.1 目的

P1 要回答：GT4 离 S1/S0 的剩余 memory 和 step 缺口到底来自哪里。

### 6.2 测量对象

```text
MLP-manual-linear-reference
MLP-autograd-reference
GT4-best-fused-no-materialize-v616
FG6-fused-grouped-forward-backward-no-materialize
FG9-single-kernel-grouped-mix
GT4-memory-margin-v1, if implemented
GT4-step-margin-v1, if implemented
```

### 6.3 Shape grid

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch sizes:
  128
  256
  512

depths:
  2
  4

hidden_dim:
  64

seed:
  0

warmup / measure:
  50 / 200 update steps
```

### 6.4 必须记录字段

#### Full-step efficiency

```text
variant
dataset
batch_size
depth
memory_ratio
step_ratio
backward_ratio
forward_ratio
update_ratio
peak_allocated_MB
peak_reserved_MB
allocated_vs_reserved_gap_MB
grad_relerr_max
grad_cos_min
```

#### Component timing

```text
forward_fused_grouped_ms
forward_residual_scale_ms
forward_grouped_mix_ms
loss_delta_ms
backward_fused_grouped_ms
backward_dx_ms
backward_grad_mix_ms
backward_grad_poly_ms
update_prep_ms
optimizer_update_ms
kernel_launch_overhead_proxy_ms
layout_conversion_ms
contiguous_copy_ms
python_overhead_ms
```

#### Memory attribution

```text
manual_cache_MB
grouped_activation_MB
grouped_output_MB
grad_mix_MB
grad_poly_MB
update_temp_MB
optimizer_state_MB
triton_scratch_MB
allocator_padding_MB
reserved_unallocated_MB
largest_live_tensor_MB
top1_memory_source
top2_memory_source
top3_memory_source
```

### 6.5 判断标准

P1 attribution pass requires:

```text
top3_memory_sources identified
component timing explains at least 90% of step time
unknown_time_fraction <= 0.10
unknown_memory_fraction <= 0.10
```

P1 must produce a repair target table with:

```text
memory_target_source
estimated_memory_saving_percent
step_target_source
estimated_step_saving_percent
risk_to_gradient
risk_to_task
```

### 6.6 可视化

```text
p1_gt4_memory_gap_waterfall.svg
p1_gt4_step_time_waterfall.svg
p1_forward_backward_update_ratio_bar.svg
p1_memory_source_stacked_bar.svg
p1_batch_depth_efficiency_heatmap.svg
p1_s1_gap_dashboard.svg
```

---

## 7. P2：S2 -> S1/S0 memory and step repair

### 7.1 目的

P2 直接把 GT4 推过 S1/S0 gate。重点不是改变 primitive family，而是做 margin repair。

### 7.2 Candidate packages

```text
E0-GT4-v616-baseline

E1-GT4-no-extra-output-cache:
  remove or shorten grouped output cache lifetime

E2-GT4-grad-buffer-reuse:
  reuse grad_mix / grad_poly buffer

E3-GT4-update-temp-reuse:
  reuse update prep temp and avoid extra update materialization

E4-GT4-forward-fastpath:
  optimize forward grouped mix and residual scale path

E5-GT4-launch-fused-step:
  reduce launch/call overhead between forward/backward/update

E6-GT4-reserved-memory-trim:
  reduce reserved/allocated gap and scratch buffer retention

E7-GT4-memory-margin-combo:
  E1 + E2 + E3 + E6

E8-GT4-step-margin-combo:
  E4 + E5 + update fastpath

E9-GT4-S1-combo:
  E1 + E2 + E3 + E4 + E5 + E6

E10-GT4-S0-aggressive:
  E9 + optional recompute small cache / stricter buffer lifetime
```

### 7.3 必须记录字段

```text
package
components
implementation_status
memory_ratio_min
memory_ratio_mean
memory_ratio_max
step_ratio_min
step_ratio_mean
step_ratio_max
backward_ratio_mean
forward_ratio_mean
update_ratio_mean
memory_improvement_vs_GT4
step_improvement_vs_GT4
backward_improvement_vs_GT4
forward_improvement_vs_GT4
peak_allocated_MB
peak_reserved_MB
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
kernel_count_total
triton_kernel_count
torch_op_count
allocation_proxy_count
```

### 7.4 判断标准

Memory useful:

$$
\frac{M_{\text{new}}}{M_{\text{GT4}}}\leq0.985.
$$

Step useful:

$$
\frac{T_{\text{step,new}}}{T_{\text{GT4}}}\leq0.96.
$$

S2 pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

S1 pass:

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

S0 pass:

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.20.
$$

Gradient correctness:

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

### 7.5 可视化

```text
p2_efficiency_repair_pareto.svg
p2_memory_repair_waterfall.svg
p2_step_repair_waterfall.svg
p2_s0_s1_s2_boundary_plot.svg
p2_gradient_correctness_lollipop.svg
```

---

## 8. P3：fused grouped candidate selection

### 8.1 目的

P3 从 P2 选择进入 task diagnosis 的 candidate。P3 要同时考虑 efficiency、shape stability、gradient correctness。

### 8.2 Survivor 类型

```text
S0:
  r_mem < 1.0 and r_step <= 1.20

S1:
  r_mem < 1.0 and r_step <= 1.35

S2:
  r_mem <= 1.05 and r_step <= 1.50

S3:
  memory pass but step fail

S4:
  step pass but memory fail

S5:
  gradient fail

S6:
  no improvement over GT4
```

### 8.3 必须记录字段

```text
best_candidate
candidate_family
survivor_type
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
shape_stability_memory
shape_stability_step
worst_shape_memory_ratio
worst_shape_step_ratio
memory_improvement_vs_GT4
step_improvement_vs_GT4
open_one_step_probe
open_diagnostic_task
open_official_task
```

Shape stability:

$$
S_{\text{shape,mem}}=1-\frac{\operatorname{std}(r_{\text{mem}})}{\operatorname{mean}(r_{\text{mem}})+\epsilon}.
$$

P3 must prefer candidates with:

$$
S_{\text{shape,mem}}\geq0.85.
$$

### 8.4 打开规则

```text
S0/S1:
  open P7 one-step probe
  open P8 official task re-entry after P7 pass

S2:
  open P7 one-step probe
  open P9 diagnostic task only after P7 pass

S3/S4/S5/S6:
  no task
```

### 8.5 可视化

```text
p3_candidate_scorecard.svg
p3_shape_stability_heatmap.svg
p3_candidate_selection_dashboard.svg
```

---

## 9. P4：diagnostic task gap attribution

### 9.1 目的

P4 解释 v6.16 diagnostic task 中 GT4 accuracy 低于 MLP 的原因。P4 不做大量新训练，而是对 v6.16 diagnostic task 和 v6.17 selected candidates 做深入记录。

### 9.2 必须比较的方法

```text
MLP-autograd-reference
MLP-manual-linear-reference
GT4-v616 + ManualAdamW
GT4-v616 + ManualAdanLite
Best-efficiency-candidate + ManualAdamW
Best-efficiency-candidate + ManualAdanLite
```

### 9.3 训练设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

steps:
  120 diagnostic steps

logging:
  every 20 steps
```

### 9.4 必须记录字段

#### Curves

```text
train_loss_curve
val_loss_curve
train_acc_curve
val_acc_curve
test_acc_at_end
val_acc_at_end
ECE_curve
NLL_curve
```

#### Generalization and representation

```text
train_val_loss_gap
train_val_acc_gap
feature_effective_rank
feature_rank_ratio_vs_MLP
class_centroid_separation
within_class_variance
margin_mean
margin_p10
margin_p50
classwise_acc
hard_class_pairs
logit_norm_mean
logit_norm_std
```

#### Optimization statistics

```text
optimizer_name
learning_rate
weight_decay
beta1
beta2
adan_beta3
grad_norm
update_norm
update_over_param_norm
cos_update_grad
cos_update_prev
role_update_share_input
role_update_share_grouped_mix
role_update_share_residual
role_grad_norm_input
role_grad_norm_grouped_mix
role_grad_norm_residual
```

#### Efficiency during task

```text
task_step_time_ms
task_backward_memory_ratio
task_forward_ratio
task_backward_ratio
samples_per_second
time_to_val_acc_50
time_to_val_acc_60
time_to_val_acc_65
```

### 9.5 判断标准

P4 must classify the task gap into one or more categories:

```text
T1-expressivity_gap:
  feature rank / margin / class separation significantly below MLP

T2-optimization_gap:
  train loss decreases slowly or update statistics unstable

T3-generalization_gap:
  train loss improves but val/test acc lags, train-val gap high

T4-calibration_gap:
  ECE/NLL worse despite similar accuracy

T5-efficiency_constrained_training:
  diagnostic budget too short due step time overhead

T6-no_clear_gap:
  metrics inconclusive
```

Expressivity gap if:

$$
\operatorname{rank}_{KAN} < 0.85\operatorname{rank}_{MLP}
$$

or:

$$
\operatorname{margin}_{p10,KAN}<0.85\operatorname{margin}_{p10,MLP}.
$$

Generalization gap if:

$$
(\operatorname{Acc}_{train,KAN}-\operatorname{Acc}_{val,KAN})
>
(\operatorname{Acc}_{train,MLP}-\operatorname{Acc}_{val,MLP})+0.03.
$$

Optimization gap if:

$$
\operatorname{ValLossAUC}_{step,KAN}
>
\operatorname{ValLossAUC}_{step,MLP}
$$

and update statistics show unstable or too-small updates.

### 9.6 可视化

```text
p4_loss_acc_curves_by_step.svg
p4_loss_acc_curves_by_time.svg
p4_feature_rank_vs_acc.svg
p4_margin_distribution.svg
p4_classwise_acc_heatmap.svg
p4_update_norm_trajectory.svg
p4_task_gap_taxonomy_dashboard.svg
```

---

## 10. P5：lightweight expressivity repair under efficiency envelope

### 10.1 目的

如果 P4 显示 task gap 主要是 expressivity/cross-group gap，则 P5 测试极轻量表达力修复。所有修复必须保持 strict PureKAN 和 S2/S1 efficiency envelope。

### 10.2 Candidate repairs

```text
X0-best-efficiency-candidate-baseline

X1-crossgroup-lite-r1:
  edge-owned rank1 cross-group correction

X2-crossgroup-lite-r2:
  edge-owned rank2 cross-group correction

X3-group-shuffle-static:
  fixed non-trainable group shuffle between layers

X4-group-shuffle-learned-edge-owned:
  learnable but edge-owned small group permutation / scaling

X5-g16-g8-hybrid:
  use g16 for memory-heavy layers, g8 for one selected layer

X6-residual-scale-warmup:
  residual scale schedule, no new params

X7-tiny-calibration-head-forbidden-check:
  diagnostic only; if nonKAN > 0, cannot be mainline
```

`X7` is only a forbidden-control diagnostic. It must not enter route selection if it has nonKAN trainable parameters.

### 10.3 必须记录字段

```text
candidate
repair_type
nonKAN_param_count
edge_param_count_delta
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
memory_overhead_vs_baseline
step_overhead_vs_baseline
grad_relerr_max
grad_cos_min
residual_over_base
feature_rank
margin_p10
val_acc
test_acc
ECE
NLL
```

### 10.4 判断标准

Efficiency envelope:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Official envelope:

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

Expressivity useful if:

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to GT4-v616 diagnostic baseline, and:

$$
\Delta r_{\text{step}}\leq0.05.
$$

If a repair improves accuracy but pushes memory or step outside S2, it is diagnostic only.

### 10.5 可视化

```text
p5_expressivity_repair_pareto.svg
p5_accuracy_vs_efficiency.svg
p5_feature_rank_improvement_bar.svg
p5_margin_improvement_bar.svg
p5_crossgroup_overhead_bar.svg
```

---

## 11. P6：optimizer / regularization diagnostic under efficiency envelope

### 11.1 目的

If P4 indicates optimizer/generalization gap, P6 tests small optimizer/regularization changes. P6 does not replace P2/P5; it is diagnostic unless S1/S0 exists.

### 11.2 Candidate recipes

```text
O0-ManualAdamW-v616
O1-ManualAdanLite-v616
O2-ManualAdanLite-lr-low
O3-ManualAdanLite-lr-high
O4-ManualAdanLite-beta-tuned
O5-ManualAdamW-weightdecay-low
O6-ManualAdamW-weightdecay-high
O7-ManualAdamW-gradclip
O8-residual-scale-warmup
O9-warmup-cosine-short
O10-label-smoothing-diagnostic
```

`label-smoothing-diagnostic` is only used to diagnose generalization/calibration. It should be clearly separated from final claim unless MLP baselines use the same setting.

### 11.3 必须记录字段

```text
recipe
optimizer
lr
weight_decay
betas
grad_clip
warmup_steps
schedule
label_smoothing
val_acc
test_acc
val_loss_auc_by_step
val_loss_auc_by_time
ECE
NLL
train_val_gap
update_norm_mean
update_norm_std
cos_update_grad
grad_norm_mean
bad_step_rate
task_step_time_ms
```

### 11.4 判断标准

Optimizer useful if:

$$
\Delta Acc_{\text{val}}\geq0.02
$$

or:

$$
\operatorname{ValLossAUC}_{time,new}
<
\operatorname{ValLossAUC}_{time,baseline}
$$

with no degradation in memory/step envelope.

Generalization repair if:

$$
\operatorname{train\_val\_gap}_{new}
<
\operatorname{train\_val\_gap}_{baseline}-0.02.
$$

Calibration repair if:

$$
\operatorname{ECE}_{new}
<
\operatorname{ECE}_{baseline}-0.01.
$$

### 11.5 可视化

```text
p6_optimizer_val_acc_bar.svg
p6_val_loss_auc_by_time.svg
p6_train_val_gap_bar.svg
p6_ece_bar.svg
p6_update_statistics_dashboard.svg
```

---

## 12. P7：one-step probe for selected candidates

### 12.1 目的

P7 validates numerical stability and descent before any official or diagnostic task.

### 12.2 运行条件

```text
Run P7 if:
  candidate survivor_type in {S0, S1, S2}
```

### 12.3 方法

```text
clone weights
compute manual gradient
apply one update
measure train loss before/after
measure holdout loss before/after
measure val loss before/after
measure residual ablation before/after
rollback weights
check rollback exactness
```

### 12.4 必须记录

```text
candidate
dataset
train_loss_before
train_loss_after
holdout_loss_before
holdout_loss_after
val_loss_before
val_loss_after
train_loss_delta
holdout_loss_delta
val_loss_delta
bad_step_flag
rollback_error
update_norm
update_over_param
grad_norm
residual_ablation_delta_loss
residual_ablation_delta_logit
```

### 12.5 判断标准

P7 pass requires:

$$
\Delta L_{\text{train}}<0.
$$

$$
\Delta L_{\text{holdout}}\leq0.02L_{\text{holdout,before}}.
$$

$$
\operatorname{BadStepRate}\leq0.05.
$$

$$
\operatorname{rollback\_error}<10^{-8}.
$$

### 12.6 可视化

```text
p7_loss_before_after.svg
p7_bad_step_heatmap.svg
p7_update_norm_vs_loss_delta.svg
p7_residual_ablation_probe.svg
```

---

## 13. P8：official task re-entry gate

### 13.1 打开条件

Official task re-entry opens only if:

```text
survivor_type in {S0, S1}
P7 pass
r_mem < 1.0
r_step <= 1.35
grad pass
no fake/proxy
```

Diagnostic task opens if:

```text
survivor_type == S2
P7 pass
r_mem <= 1.05
r_step <= 1.50
```

### 13.2 Official task methods

If official task opens:

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

methods:
  MLP-autograd-reference
  MLP-manual-linear-reference
  Best-fused-grouped-S1 + ManualAdamW
  Best-fused-grouped-S1 + ManualAdanLite
  Best-fused-grouped-S1 + best diagnostic optimizer
```

### 13.3 Diagnostic task methods

If only S2 opens:

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

steps:
  120 or 240 diagnostic steps

methods:
  MLP-autograd-reference
  MLP-manual-linear-reference
  GT4-v616 + ManualAdanLite
  Best-v617-S2 + ManualAdanLite
  Best-v617-S2 + best diagnostic recipe
```

### 13.4 必须记录

```text
train_loss_curve
val_loss_curve
test_acc
val_acc
train_acc
ECE
NLL
val_loss_auc_by_step
val_loss_auc_by_time
val_acc_auc_by_step
val_acc_auc_by_time
time_to_target_loss
time_to_target_acc
step_time_ms
wall_clock_time_sec
backward_memory_ratio
samples_per_second
feature_rank
margin_mean
margin_p10
classwise_acc
seedwise_failure_reason
```

### 13.5 判断标准

Official task pass:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.01
$$

on all datasets.

At least two datasets:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}.
$$

Memory must hold:

$$
r_{\text{mem}}<1.0.
$$

Wall-clock must hold on at least two datasets:

$$
\operatorname{ValLossAUC}_{time,KAN}
\leq
\operatorname{ValLossAUC}_{time,MLP}.
$$

Diagnostic task improvement pass:

$$
\operatorname{Acc}_{val,new}
\geq
\operatorname{Acc}_{val,GT4-v616}+0.02.
$$

### 13.6 可视化

```text
p8_val_loss_vs_step.svg
p8_val_loss_vs_time.svg
p8_accuracy_vs_time.svg
p8_time_to_target_bar.svg
p8_task_efficiency_pareto.svg
p8_seedwise_delta_plot.svg
p8_feature_rank_margin_dashboard.svg
```

---

## 14. P9：optimizer exploration remains gated

P9 only opens after official P8 task pass. It is not a v6.17 main stage.

If opened, compare:

```text
ManualAdamW
ManualAdanLite
ManualWinLite
Lookahead-ManualAdamW
WarmupCosine-ManualAdamW
Best-diagnostic-recipe-from-P6
```

Pass standard:

$$
T_{\text{target,optimizer}}\leq0.90T_{\text{target,ManualAdamW}},
$$

or:

$$
\operatorname{ValLossAUC}_{time,optimizer}
<
\operatorname{ValLossAUC}_{time,ManualAdamW}.
$$

Accuracy must satisfy:

$$
\operatorname{Acc}_{optimizer}
\geq
\operatorname{Acc}_{ManualAdamW}-0.005.
$$

---

## 15. P10：functional correction remains gated

P10 only opens after P8 official task pass and P9 optimizer stability. Default status is:

```text
not_run
```

If opened, smoke only:

```text
task-only best
task + small residual geometry correction
task + low-frequency LightSmooth
```

Pass standards:

$$
\cos(d_{\text{new}},d_{\text{task}})\geq0.85.
$$

$$
\frac{\operatorname{HoldoutDescent}(d_{\text{new}})}
{\operatorname{HoldoutDescent}(d_{\text{task}})}
\geq0.95.
$$

$$
\operatorname{BadStepRate}\leq0.02.
$$

---

## 16. P11：route decision

### Route cases

```text
R1-FusedGroupedS1Solved:
  GT4-derived candidate reaches S1/S0 and P8 official task improves or matches MLP.
  Continue to 5-seed confirm.

R2-FusedGroupedEfficiencySolvedTaskGap:
  S1/S0 reached, but task still below MLP by >2%.
  Next cycle focuses on expressivity/optimizer.

R3-FusedGroupedS2TaskImproved:
  Still S2 only, but diagnostic task improves by >=2%.
  Continue S2 engineering toward S1.

R4-FusedGroupedS2NoTaskImprovement:
  Still S2 and task gap remains large.
  Need expressivity redesign.

R5-EfficiencyMarginFailed:
  GT4 cannot move beyond S2; memory or step margin stuck.
  Continue kernel margin repair or new primitive.

R6-CrossGroupRepairUseful:
  cross-group-lite improves task while keeping S2/S1.
  Continue expressivity repair.

R7-CrossGroupRepairTooExpensive:
  task improves but memory/step leaves S2.
  Diagnostic only, cannot continue.

R8-NoViableV617:
  no candidate improves efficiency or task over v6.16 GT4.
  Need new grouped expressivity / primitive family.

R9-TaskReentryPass:
  official task passes.

R10-TaskReentryFail:
  official task opens but task fails.
```

### JSON 输出字段

```text
route
best_candidate
best_family
best_memory_ratio
best_step_ratio
best_backward_ratio
best_forward_ratio
memory_improvement_vs_GT4
step_improvement_vs_GT4
survivor_type
one_step_probe_pass
official_task_opened
diagnostic_task_opened
best_val_acc
best_test_acc
val_acc_gap_vs_MLP
test_acc_gap_vs_MLP
task_gap_taxonomy
open_optimizer_exploration
open_functional_correction
primary_blocker
next_required_implementation
no_fake
no_proxy
```

---

## 17. P12：artifact and failure audit

### 17.1 Failure types

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_residual_effect_fail
F5_efficiency_margin_fail
F6_forward_time_fail
F7_update_time_fail
F8_crossgroup_overhead_fail
F9_task_gap_expressivity
F10_task_gap_optimization
F11_task_gap_generalization
F12_task_gap_unclear
F13_official_task_gated
F14_optimizer_gated
F15_functional_gated
F16_fake_or_proxy_violation
F17_artifact_missing
```

### 17.2 Artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p1_gt4_efficiency_margin_attribution.csv
p2_efficiency_margin_repair.csv
p2_efficiency_margin_repair_detail.csv
p3_fused_grouped_candidate_selection.csv
p4_diagnostic_task_gap_attribution.csv
p4_task_trace.csv
p5_expressivity_repair.csv
p5_expressivity_repair_detail.csv
p6_optimizer_regularization_diagnostic.csv
p6_optimizer_regularization_trace.csv
p7_one_step_probe.csv
p8_task_reentry.csv
p8_task_trace.csv
p9_optimizer_exploration.csv
p10_functional_correction_smoke.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### 17.3 Figures

```text
figures/p1_gt4_memory_gap_waterfall.svg
figures/p1_gt4_step_time_waterfall.svg
figures/p2_efficiency_repair_pareto.svg
figures/p2_s0_s1_s2_boundary_plot.svg
figures/p4_loss_acc_curves_by_step.svg
figures/p4_loss_acc_curves_by_time.svg
figures/p4_feature_rank_vs_acc.svg
figures/p4_margin_distribution.svg
figures/p4_classwise_acc_heatmap.svg
figures/p5_expressivity_repair_pareto.svg
figures/p6_optimizer_val_acc_bar.svg
figures/p8_task_efficiency_pareto.svg
figures/p11_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 18. 成功与失败解释规则

### Case A：GT4-derived candidate reaches S1/S0

If a candidate satisfies:

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35,
$$

then official task re-entry opens after P7 one-step pass.

### Case B：S2 remains but diagnostic task improves

If candidate remains S2 but diagnostic validation accuracy improves by at least $2\%$, fused grouped remains mainline, but official claim remains closed.

### Case C：S2 remains and task gap does not improve

If candidate remains S2 and diagnostic task gap remains larger than $5\%$, next cycle must focus on expressivity redesign, not only kernel margin.

### Case D：S1/S0 achieved but task fails

If official task opens but accuracy remains below MLP by more than $2\%$, route is efficiency-solved / task-gap. Next cycle focuses on task expressivity and optimizer, not memory.

### Case E：efficiency margin cannot improve over GT4

If no package improves memory or step over v6.16 GT4 by at least $1\%$, route is margin-stuck. Next cycle must decide whether to accept S2 diagnostic route or redesign primitive.

### Case F：cross-group repair improves task but breaks efficiency

If cross-group repair improves accuracy but violates S2, it is diagnostic only. It cannot enter official route.

---

## 19. 最终建议

v6.17 的一句话策略是：

$$
\boxed{
\text{把 v6.16 的 S2 fused grouped survivor 推到 S1/S0，同时解释 diagnostic task gap。}
}
$$

当前不应再回到 DWM2、Python grouped loop、low-rank 或 piecewise 主线。v6.16 已经证明 Triton fused grouped path 是当前最有希望的 efficient PureKAN route。

如果 v6.17 成功把 GT4 推到 S1/S0，就可以第一次正式打开 official task re-entry。  
如果 v6.17 只能保持 S2，但 diagnostic task 明显改善，也值得继续 fused grouped。  
如果 v6.17 既不能提升效率，也不能改善 task gap，则需要重新设计 grouped expressivity 或新的 primitive family。
