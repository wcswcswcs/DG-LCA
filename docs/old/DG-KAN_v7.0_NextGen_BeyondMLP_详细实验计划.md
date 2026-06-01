# DG-KAN v7.0 Next-Gen Beyond-MLP：更高效、更高表达力 PureKAN 架构路线与详细实验计划

> 本计划基于 v6.20 final real-only run 和当前“全面超越 MLP”的新目标制定。v6.x 的目标主要是让 graph-free PureKAN 接近或替代 MLP；v7.0 的目标更激进：构建一个 **kernel-native、graph-free、strict PureKAN、function-space neural architecture beyond MLP**。  
> v6.20 已经得到真实 S2 candidate：`T3-launch-fused-step`，其 step ratio 已经大幅过线，但 memory ratio 仍为 `1.0312`，还没有达到 official task gate 的 `<1.00`；同时 diagnostic task 中 T3 的 validation/test accuracy 仍低于 MLP，task gap taxonomy 指向 generalization gap。因此 v7.0 不再只做“能否替代 MLP”的实验，而是把目标升级为：在相同或更低 memory / time 预算下，让 PureKAN 在收敛速度、泛化、校准、几何、sample efficiency、noise robustness、scaling 上至少一个层面明确超过 MLP，并逐步进入 patch/token vision scaling。

---

## 0. 重新定义 v7.0 总目标

v6.6/v6.8 以前的核心目标可以概括为：

$$
\boxed{\text{PureKAN 可以成为 MLP 的高效替代。}}
$$

这个目标已经不够。v7.0 的目标应改成：

$$
\boxed{\text{构建一个次世代 graph-free PureKAN 架构，在效率、表达力、泛化和几何上全面挑战并超过 MLP。}}
$$

v7.0 的正式命名目标为：

$$
\boxed{
\text{PureKAN-NG: a graph-free, kernel-native, function-space neural architecture beyond MLP.}
}
$$

它必须满足以下硬条件：

```text
1. zero non-KAN trainable parameters；
2. no PyTorch loss.backward graph；
3. manual forward / backward / update；
4. forward time 接近或优于 MLP；
5. backward time 接近或优于 MLP；
6. backward memory 低于 MLP；
7. convergence 更快；
8. accuracy / AUC / ECE / geometry 至少不弱于 MLP-AdamW，并在至少一个关键维度超过；
9. 在 sample efficiency、noise robustness、patch/token scaling 上显示架构级优势。
```

v7.0 的 efficiency gate 分三档：

```text
S2 diagnostic:
  memory_ratio <= 1.05
  step_ratio <= 1.50

S1 official:
  memory_ratio < 1.00
  step_ratio <= 1.35

S0 strong:
  memory_ratio < 1.00
  step_ratio <= 1.20
```

v7.0 的 beyond-MLP task gate 不再只看 accuracy，而是定义为：

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.01
$$

on all primary datasets, and at least two primary datasets satisfy:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}.
$$

更高一级的 beyond-MLP strong gate 是：

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}+0.005
$$

on at least two datasets, while:

$$
\operatorname{ECE}_{KAN}\leq\operatorname{ECE}_{MLP},
$$

and:

$$
\operatorname{ValLossAUC}_{time,KAN}\leq\operatorname{ValLossAUC}_{time,MLP}.
$$

v7.0 不允许只用单个小数据集 accuracy 宣称成功。必须同时看：

```text
efficiency
task accuracy
validation-loss AUC
wall-clock AUC
ECE / NLL
feature rank
margin distribution
classwise accuracy
sample efficiency
noise robustness
scaling behavior
```

---

## 1. 当前事实基线

v6.20 的关键事实是：

```text
T3-launch-fused-step:
  memory ratio mean   = 1.0312
  step ratio mean     = 0.8436
  backward ratio mean = 0.6822
  forward ratio mean  = 1.0499
  grad relerr max     = 1.92e-08
  survivor type       = S2
  one-step probe      = pass
  diagnostic task     = opened
```

这说明轻量化已经不是主要失败点。T3 的 step ratio 已经远低于 S1 gate：

$$
0.8436 < 1.35.
$$

剩余 official efficiency blocker 是 memory：

$$
r_{\text{mem}}=1.0312>1.00.
$$

也就是说，T3 离 official memory gate 还需要：

$$
1-\frac{1.00}{1.0312}\approx3.03\%
$$

的 memory reduction。

但 v6.20 diagnostic task 暴露出更重要的新问题：

```text
MLP-autograd-reference:
  val acc mean  = 0.6981
  test acc mean = 0.6523

T3-launch-fused-step + ManualAdamW:
  val acc mean  = 0.6387
  test acc mean = 0.6018

T3-launch-fused-step + ManualAdanLite:
  val acc mean  = 0.6280
  test acc mean = 0.5927
```

因此 T3+ManualAdamW 相对 MLP-autograd 的 gap 是：

$$
\Delta Acc_{\text{val}}\approx-0.0595,
$$

$$
\Delta Acc_{\text{test}}\approx-0.0506.
$$

v6.20 的 route 是：

```text
R8-S2RestoredTaskGapPersists
primary_blocker = task_gap
next_required_implementation = task_gap_repair
```

这说明 v7.0 的主线不应该回到 DWM2、low-rank、piecewise 或 Python grouped loop。主线应是：

$$
\boxed{
\text{T3 fused grouped kernel as efficiency base}
+
\text{S1 memory repair}
+
\text{generalization / expressivity / optimizer repair}
+
\text{scaling evaluation}.
}
$$

---

## 2. v7.0 总体策略

v7.0 采用双主线并行、严格 gate 汇合的方式。

第一条主线是 **Efficiency-to-S1**：

```text
目标：
  T3 memory 1.0312 -> < 1.00
  step 保持 <= 1.35，最好保持 <= 1.00

方法：
  non-recompute buffer lifetime repair
  update/foreach temp reuse
  triton scratch trim
  short-lived grouped output free
  grad_mix / grad_poly reuse
```

第二条主线是 **Beyond-MLP Task Repair**：

```text
目标：
  T3 diagnostic val acc 至少提升 2%
  task gap 从 5%-7% 降到 <=2%
  解释 gap 是 generalization、optimizer、cross-group expressivity 还是 budget

方法：
  optimizer / regularization diagnostic
  cross-group-lite edge-owned correction
  group shuffle / g16-g8 hybrid
  longer-budget diagnostic
  sample efficiency / robustness evaluation
```

两条主线必须通过同一个 contract：

```text
nonKAN_param_count = 0
manual_backward_available = 1
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
grad_relerr < 1e-4
```

v7.0 的关键原则是：

$$
\boxed{
\text{任何提升 accuracy 的 repair，如果破坏 S2 efficiency envelope，只能作为 diagnostic，不能作为路线成功。}
}
$$

同样：

$$
\boxed{
\text{任何降低 memory 的 repair，如果 task gap 继续扩大，也不能称为 beyond-MLP 成功。}
}
$$

---

## 3. 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。任何未实现内容必须写为：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，不允许回到 DWM2 主线。DWM2 只保留为历史 baseline，不进入 v7.0 route selection。

第三，不允许把 low-rank、piecewise、Python grouped loop 作为主线。这些分支已经被 fused grouped T3 明显超过，只能作为 reference 或 diagnostic。

第四，不允许把 S2 diagnostic task 写成 official task success。Official task 只能在 S1/S0 后打开：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

第五，不允许只看 accuracy。所有 task result 必须同时报告：

```text
val_loss_auc_by_step
val_loss_auc_by_time
ECE
NLL
train_val_gap
feature_rank
margin_p10
classwise_acc
wall-clock time
memory ratio
step ratio
```

第六，不允许引入 non-KAN trainable parameters。任何 cross-group correction、calibration、temperature scaling、residual repair 都必须明确记录：

```text
nonKAN_param_count
edge_param_count_delta
manual_backward_available
```

如果 `nonKAN_param_count > 0`，该 candidate 只能作为 forbidden-control diagnostic，不能进入 route selection。

第七，不允许用 test accuracy 调参。P3-P8 的 candidate selection 只能依据 train/val/holdout 和 diagnostic metrics。test 只在最终报告中使用。

第八，不允许在没有 S2 candidate 时运行 task/optimizer/functional 阶段；没有 S1/S0 时不能运行 official task；没有 official task pass 时不能运行系统 optimizer exploration 和 functional correction。

第九，不允许把 optimizer 小诊断称为 optimizer exploration。v7.0 允许 small diagnostic optimizer，但系统 optimizer exploration 仍需 official task pass。

第十，不允许只在 MNIST-family 上宣称 beyond-MLP。MNIST-family 是 gate；v7.0 还必须增加至少一个 patch/token scaling diagnostic。

---

## 4. 核心假设

### H1：T3 已经解决 step blocker，v7.0 的效率 blocker 是 memory lifetime

T3 的 step ratio 是：

$$
r_{\text{step}}=0.8436.
$$

这说明 step 已不再是主 blocker。H1 假设：T3 剩余 memory gap 来自短生命周期 buffer 或 update temp，而不是 primitive 本体必须高于 MLP。

候选 memory source 包括：

```text
grad_mix temp
grad_poly temp
update_temp
foreach_update_temp
triton_scratch
grouped_output lifetime
reserved_unallocated gap
allocator padding
```

H1 成立要求 memory attribution 找到至少一个 repairable source，其 estimated saving 满足：

$$
\Delta r_{\text{mem}}\geq0.015.
$$

H1 repair 成立要求：

$$
\frac{M_{\text{repair}}}{M_{\text{T3}}}\leq0.985,
$$

且：

$$
\frac{T_{\text{repair}}}{T_{\text{T3}}}\leq1.02.
$$

组合 repair 成立要求：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

---

### H2：T3 的 diagnostic task gap 是 generalization/regularization 问题，不是完全学不动

v6.20 中 T3 的 train loss delta 明显下降，但 val/test acc 低于 MLP。H2 假设：T3 的 gap 不是“模型不能优化”，而是 generalization / calibration / regularization / margin 结构问题。

H2 成立要求至少出现以下证据之一：

```text
train_val_acc_gap_KAN > train_val_acc_gap_MLP + 0.03
ECE_KAN > ECE_MLP + 0.02
margin_p10_KAN < 0.85 * margin_p10_MLP
classwise_acc 在若干 hard pairs 上集中掉点
```

如果 H2 成立，v7.0 应优先做：

```text
weight decay / grad clip / warmup
label smoothing diagnostic
edge-noise diagnostic
residual scale warmup
cross-group regularization
```

而不是继续只做 kernel repair。

---

### H3：T3 的 task gap 可能来自 grouped g16 跨组表达力不足

T3 的高效来自 grouped g16 fused kernel，但 g16 可能限制 cross-channel information flow。H3 假设：加入极轻量、edge-owned、manual backward 的 cross-group correction 可以提升 task accuracy，并保持 S2/S1 efficiency。

候选包括：

```text
static group shuffle
edge-owned rank1 cross-group correction
edge-owned rank2 cross-group correction
g16/g8 hybrid in one layer
late-phase cross-group residual
groupwise temperature / scaling edge-owned
```

H3 成立要求：

$$
\Delta Acc_{\text{val}}\geq0.02,
$$

且：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

H3 stronger evidence：

$$
\Delta \operatorname{margin}_{p10}\geq0.05\cdot\operatorname{margin}_{p10,baseline},
$$

or hard-class-pair accuracy improves by at least $2\%$.

---

### H4：ManualAdamW 比当前 ManualAdanLite 更适合作为 T3 的 first optimizer baseline

v6.20 中：

```text
T3+ManualAdamW val acc  = 0.6387
T3+ManualAdanLite val acc = 0.6280
```

H4 假设：当前 T3 的 grouped fused path 更适合 AdamW-like dynamics。H4 成立要求：

$$
Acc_{\text{val,AdamW}}-Acc_{\text{val,AdanLite}}\geq0.01,
$$

或：

$$
\operatorname{ValLossAUC}_{time,\text{AdamW}}
<
\operatorname{ValLossAUC}_{time,\text{AdanLite}}.
$$

如果 H4 成立，v7.0 optimizer diagnostic 应优先围绕：

```text
ManualAdamW weight decay
ManualAdamW grad clip
ManualAdamW warmup cosine
ManualAdamW lr low/high
```

而不是继续大幅调整 AdanLite。

---

### H5：Beyond-MLP 不应只在 final accuracy 上定义，而应包含 sample efficiency 和 robustness

如果 PureKAN-NG 是 function-space architecture，它应在数据少、噪声多、需要平滑/几何稳定的任务上显示优势。H5 假设：即使 raw accuracy 暂时接近 MLP，PureKAN-NG 应该在 sample efficiency、noise robustness 或 calibration 上更有机会超过 MLP。

H5 成立标准至少满足一项：

Sample efficiency：

$$
\operatorname{AUC}_{data,KAN}
>
\operatorname{AUC}_{data,MLP}
$$

where data fractions are:

```text
5%, 10%, 25%, 50%, 100%
```

Noise robustness：

$$
\operatorname{AccDrop}_{KAN}
<
\operatorname{AccDrop}_{MLP}
$$

under label noise or input corruption.

Calibration：

$$
\operatorname{ECE}_{KAN}
<
\operatorname{ECE}_{MLP}-0.01.
$$

---

### H6：Patch/token scaling 是 v7.0 的必要 diagnostic，但不能在 MNIST-family gate 失败时作为 final claim

v7.0 的目标包括进入 patch/token vision scaling。H6 假设：T3 fused grouped primitive 可以作为 patch/token MLP block replacement，但需要先通过 MNIST-family 的 S2/S1 + task gap diagnostic。

Patch/token diagnostic 的最低要求是：

```text
CIFAR-10-small or TinyImageNet-subset patch-token classifier
small ViT-like or ConvPatchMLP baseline
same parameter budget
same training budget
same no-fake/no-proxy contract
```

H6 成立要求：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

and:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.02
$$

on patch/token diagnostic.

但如果 MNIST-family task gap 仍 $>5\%$，patch/token result 只能作为 scaling diagnostic，不能作为 final success。

---

## 5. 实验阶段总览

v7.0 分为十五个阶段：

```text
P0: T3 reproduction and v7 contract
P1: S1 memory attribution
P2: non-recompute S1 memory repair
P3: task gap deep attribution
P4: optimizer / regularization diagnostic
P5: cross-group expressivity diagnostic
P6: longer-budget diagnostic
P7: sample efficiency diagnostic
P8: noise robustness and calibration diagnostic
P9: patch/token scaling diagnostic
P10: candidate selection by beyond-MLP score
P11: one-step probe for selected candidates
P12: official task re-entry if S1/S0
P13: diagnostic task if S2 only
P14: route decision
P15: artifact and failure audit
```

P0-P2 是 efficiency S1 主线。  
P3-P8 是 beyond-MLP task / generalization 主线。  
P9 是 scaling diagnostic。  
P10-P13 根据 survivor type 打开。  
P14-P15 生成路线结论。

---

## 6. P0：T3 reproduction and v7 contract

### 6.1 目的

P0 确认 v7.0 与 v6.20 的 T3 baseline 可比，并确认 T3 仍保持 strict PureKAN / graph-free / manual backward / manual update / no-fake contract。

### 6.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
T3-launch-fused-step-v620
T3-launch-fused-step-v700
T3+ManualAdamW-task-baseline
T3+ManualAdanLite-task-baseline
```

如果 v7 新增 candidate 已实现：

```text
T3-S1-memory-repair-v1
T3-crossgroup-lite-r1
T3-static-shuffle
T3-AdamW-warmup
```

也纳入 P0 contract。

### 6.3 必须记录字段

```text
variant
status
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
uses_triton_kernel
uses_foreach_update
manual_forward_available
manual_backward_available
manual_update_available
nonKAN_param_count
edge_param_count
edge_param_count_delta
rollback_max_error
grad_relerr_max
grad_cos_min
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
update_ratio_mean
peak_allocated_MB
peak_reserved_MB
v620_memory_ratio_mean
v700_memory_ratio_mean
v620_step_ratio_mean
v700_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
```

### 6.4 判断标准

P0 reproduction pass：

$$
|r_{\text{mem,v700}}-1.0312|\leq0.02.
$$

$$
|r_{\text{step,v700}}-0.8436|\leq0.08.
$$

T3 S2 reproduction pass：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Contract pass：

```text
fake_data_used = 0
proxy_row_used = 0
uses_loss_backward = 0
uses_torch_autograd_graph = 0
manual_backward_available = 1
nonKAN_param_count = 0
grad_relerr < 1e-4
grad_cos > 0.999
```

### 6.5 可视化

```text
p0_t3_reproduction_bar.svg
p0_contract_heatmap.svg
p0_efficiency_delta_vs_v620.svg
p0_candidate_lineage_table.md
```

---

## 7. P1：S1 memory attribution

### 7.1 目的

P1 专门定位 T3 的 memory ratio `1.0312` 中，超过 MLP 的 $3.03\%$ 来自哪里。P1 不做 repair，只做 attribution。

### 7.2 测量对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
T3-launch-fused-step
M3-update-temp-reuse
T2-update-fastpath
E1-recompute-hidden-cache-diagnostic
```

### 7.3 Shape grid

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

seed:
  0

warmup / measure:
  50 / 200 update steps
```

### 7.4 必须记录字段

```text
variant
dataset
batch_size
depth
peak_allocated_MB
peak_reserved_MB
MLP_peak_allocated_MB
memory_ratio
peak_gap_MB
reserved_minus_allocated_MB
grouped_output_MB
grad_mix_MB
grad_poly_MB
update_temp_MB
foreach_update_temp_MB
optimizer_state_MB
triton_scratch_MB
manual_cache_MB
allocator_padding_MB
short_lived_tensor_peak_MB
largest_live_tensor_MB
top1_memory_source
top1_memory_MB
top2_memory_source
top2_memory_MB
top3_memory_source
top3_memory_MB
unknown_memory_fraction
```

### 7.5 判断标准

Memory attribution pass：

```text
top3_memory_sources identified
unknown_memory_fraction <= 0.10
at least one repairable source >= 1.5% of T3 peak
```

repairable source 包括：

```text
grad_mix
grad_poly
update_temp
foreach_update_temp
triton_scratch
grouped_output_lifetime
reserved_unallocated
allocator_padding
```

P1 必须输出 repair priority table：

```text
source
source_MB
source_fraction_of_peak
estimated_saving_MB
estimated_delta_memory_ratio
repair_risk_to_step
repair_risk_to_gradient
recommended_repair
```

### 7.6 可视化

```text
p1_t3_memory_gap_waterfall.svg
p1_top_memory_sources_bar.svg
p1_peak_gap_stacked_by_shape.svg
p1_repairable_memory_source_heatmap.svg
p1_reserved_vs_allocated_scatter.svg
```

---

## 8. P2：non-recompute S1 memory repair

### 8.1 目的

P2 只做不伤 step 的 memory repair。T3 step 已经很强，不能用 recompute 把 step 拉差。

### 8.2 Candidate packages

```text
M0-T3-baseline

M1-grad-mix-buffer-reuse:
  reuse grad_mix temp after accumulation

M2-grad-poly-buffer-reuse:
  reuse grad_poly temp and avoid duplicate grad temp

M3-update-temp-reuse:
  preserve v6.20 update-temp reuse path

M4-triton-scratch-lifetime-trim:
  reduce or reuse Triton scratch earlier

M5-reserved-memory-trim:
  reduce reserved-unallocated gap

M6-short-lived-grouped-output-free:
  shorten grouped output lifetime after backward consumes it

M7-foreach-update-buffer-reuse:
  reduce foreach update temporary materialization

M8-safe-S1-memory-combo:
  M1 + M2 + M4 + M6 + M7

M9-E1-recompute-hidden-cache:
  diagnostic only
```

### 8.3 必须记录字段

```text
package
components
implementation_status
uses_recompute
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
update_ratio_mean
memory_improvement_vs_T3
step_change_vs_T3
peak_allocated_MB
peak_reserved_MB
reserved_minus_allocated_MB
grad_mix_MB
grad_poly_MB
update_temp_MB
foreach_update_temp_MB
triton_scratch_MB
grouped_output_lifetime_ms
kernel_count_total
torch_op_count
allocation_proxy_count
grad_relerr_max
grad_cos_min
survivor_type
```

### 8.4 判断标准

Single repair useful：

$$
\frac{M_{\text{new}}}{M_{\text{T3}}}\leq0.985.
$$

Step-preserving：

$$
\frac{T_{\text{step,new}}}{T_{\text{T3}}}\leq1.02.
$$

S2：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

S1：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

如果 memory 改善但 step 恶化超过 $2\%$，candidate 是 S3 diagnostic only。

### 8.5 可视化

```text
p2_memory_repair_pareto.svg
p2_s1_memory_boundary_plot.svg
p2_memory_repair_waterfall.svg
p2_recompute_vs_nonrecompute_tradeoff.svg
p2_gradient_correctness_lollipop.svg
```

---

## 9. P3：task gap deep attribution

### 9.1 目的

P3 解释 T3 的 generalization gap。这里不能只看 final val/test acc，要拆 loss curve、rank、margin、classwise、calibration、optimizer dynamics。

### 9.2 必跑方法

```text
MLP-autograd-reference
MLP-manual-linear-reference
T3+ManualAdamW
T3+ManualAdanLite
Best-S1-or-S2-memory-repair+ManualAdamW, if available
Best-S1-or-S2-memory-repair+ManualAdanLite, if available
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

Curves：

```text
train_loss_curve
val_loss_curve
train_acc_curve
val_acc_curve
ECE_curve
NLL_curve
```

Final metrics：

```text
val_acc
test_acc
train_acc
val_loss
test_loss
train_loss
train_loss_delta
val_loss_auc_by_step
val_loss_auc_by_time
val_acc_auc_by_step
val_acc_auc_by_time
time_to_val_acc_50
time_to_val_acc_60
time_to_val_acc_65
```

Generalization metrics：

```text
train_val_acc_gap
train_val_loss_gap
ECE
NLL
logit_norm_mean
logit_norm_std
confidence_mean
confidence_p90
wrong_confidence_mean
```

Representation metrics：

```text
feature_effective_rank
feature_rank_ratio_vs_MLP
class_centroid_separation
within_class_variance
margin_mean
margin_p10
margin_p50
classwise_acc
hard_class_pairs
groupwise_feature_variance
cross_group_correlation
```

Optimization metrics：

```text
optimizer_name
learning_rate
weight_decay
grad_norm
update_norm
update_over_param_norm
cos_update_grad
cos_update_prev
role_update_share_grouped_mix
role_update_share_residual
role_update_share_input
role_grad_norm_grouped_mix
role_grad_norm_residual
bad_step_rate
```

### 9.5 判断标准

Generalization gap：

$$
(\operatorname{Acc}_{train,KAN}-\operatorname{Acc}_{val,KAN})
>
(\operatorname{Acc}_{train,MLP}-\operatorname{Acc}_{val,MLP})+0.03.
$$

Optimization gap：

$$
\operatorname{ValLossAUC}_{step,KAN}
>
\operatorname{ValLossAUC}_{step,MLP}
$$

and update statistics show unstable or too-small updates.

Expressivity / cross-group gap：

$$
\operatorname{rank}_{KAN}<0.85\operatorname{rank}_{MLP}
$$

or:

$$
\operatorname{margin}_{p10,KAN}<0.85\operatorname{margin}_{p10,MLP}.
$$

Calibration gap：

$$
\operatorname{ECE}_{KAN}>\operatorname{ECE}_{MLP}+0.02.
$$

### 9.6 可视化

```text
p3_loss_acc_curves_by_step.svg
p3_loss_acc_curves_by_time.svg
p3_train_val_gap_bar.svg
p3_feature_rank_vs_acc.svg
p3_margin_distribution.svg
p3_classwise_acc_heatmap.svg
p3_cross_group_correlation_heatmap.svg
p3_task_gap_taxonomy_dashboard.svg
```

---

## 10. P4：optimizer / regularization diagnostic

### 10.1 目的

P4 是小范围 diagnostic，不是系统 optimizer exploration。它只在 candidate 保持 S2 时运行。

### 10.2 Candidate recipes

```text
O0-T3-ManualAdamW-v620
O1-T3-ManualAdanLite-v620
O2-ManualAdamW-weightdecay-low
O3-ManualAdamW-weightdecay-high
O4-ManualAdamW-gradclip
O5-ManualAdamW-warmup-cosine
O6-ManualAdamW-lr-low
O7-ManualAdamW-lr-high
O8-ManualAdanLite-beta-tuned
O9-residual-scale-warmup
O10-label-smoothing-diagnostic
O11-edge-noise-diagnostic
```

### 10.3 必须记录字段

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
edge_noise
memory_ratio_mean
step_ratio_mean
val_acc
test_acc
val_loss_auc_by_step
val_loss_auc_by_time
ECE
NLL
train_val_acc_gap
train_val_loss_gap
update_norm_mean
update_norm_std
cos_update_grad
grad_norm_mean
bad_step_rate
```

### 10.4 判断标准

Optimizer/regularization useful：

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to v6.20 T3+ManualAdamW baseline, while:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Generalization repair：

$$
\operatorname{train\_val\_gap}_{new}
<
\operatorname{train\_val\_gap}_{baseline}-0.02.
$$

Calibration repair：

$$
\operatorname{ECE}_{new}
<
\operatorname{ECE}_{baseline}-0.01.
$$

### 10.5 可视化

```text
p4_optimizer_val_acc_bar.svg
p4_val_loss_auc_by_time.svg
p4_train_val_gap_bar.svg
p4_ece_bar.svg
p4_update_statistics_dashboard.svg
p4_accuracy_vs_efficiency.svg
```

---

## 11. P5：cross-group expressivity diagnostic

### 11.1 目的

P5 测试 grouped g16 是否造成跨组表达力不足。所有 candidate 必须 strict PureKAN。

### 11.2 Candidate repairs

```text
X0-T3-baseline

X1-static-group-shuffle:
  fixed non-trainable group shuffle between fused grouped layers

X2-crossgroup-lite-r1-edge-owned:
  edge-owned rank1 cross-group correction

X3-crossgroup-lite-r2-edge-owned:
  edge-owned rank2 cross-group correction

X4-g16-g8-hybrid-one-layer:
  use g8 in one selected layer, g16 elsewhere

X5-late-phase-crossgroup-residual:
  enable cross-group correction after warmup

X6-groupwise-temperature-scaling-edge-owned:
  edge-owned grouped output scaling

X7-forbidden-nonKAN-head-control:
  diagnostic only; cannot enter route if nonKAN > 0
```

### 11.3 必须记录字段

```text
candidate
repair_type
nonKAN_param_count
edge_param_count_delta
manual_backward_available
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
memory_overhead_vs_T3
step_overhead_vs_T3
grad_relerr_max
grad_cos_min
val_acc
test_acc
ECE
NLL
feature_rank
margin_p10
cross_group_correlation
classwise_acc
hard_class_pair_improvement
```

### 11.4 判断标准

Efficiency envelope：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Official envelope：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

Expressivity useful：

$$
\Delta Acc_{\text{val}}\geq0.02,
$$

and:

$$
\Delta r_{\text{step}}\leq0.05.
$$

Cross-group evidence useful if:

$$
\Delta \operatorname{margin}_{p10}\geq0.05\operatorname{margin}_{p10,baseline}
$$

or hard class pair accuracy improves by at least $2\%$.

If repair improves task but violates S2, it is diagnostic only.

### 11.5 可视化

```text
p5_crossgroup_repair_pareto.svg
p5_accuracy_vs_crossgroup_overhead.svg
p5_feature_rank_improvement_bar.svg
p5_margin_improvement_bar.svg
p5_hard_class_pair_heatmap.svg
```

---

## 12. P6：longer-budget diagnostic

### 12.1 目的

判断 T3 是否只是短 budget 下落后。v6.20 task 只有 120 steps，P6 扩到 240 steps。

### 12.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
T3+ManualAdamW
Best-v7-optimizer-repair
Best-v7-crossgroup-repair, if S2
```

### 12.3 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

steps:
  240 diagnostic steps

logging:
  every 20 steps
```

### 12.4 必须记录

```text
val_acc_at_120
val_acc_at_240
test_acc_at_240
val_loss_auc_0_120
val_loss_auc_120_240
time_to_val_acc_60
time_to_val_acc_65
late_slope_val_acc
late_slope_val_loss
overfit_index
```

### 12.5 判断标准

Longer-budget compensation if:

$$
\operatorname{Acc}_{val,240} - \operatorname{Acc}_{val,120} \geq 0.02,
$$

and:

$$
\operatorname{ValLossAUC}_{time,KAN,0:240}
\leq
\operatorname{ValLossAUC}_{time,MLP,0:240} + 0.05.
$$

If T3 gap persists at 240 steps, budget is not the main blocker.

### 12.6 可视化

```text
p6_long_budget_val_acc_curve.svg
p6_late_slope_bar.svg
p6_time_to_target_long_budget.svg
p6_auc_0_120_vs_120_240.svg
```

---

## 13. P7：sample efficiency diagnostic

### 13.1 目的

Beyond-MLP 目标要求 PureKAN-NG 不只在 full-data accuracy 上比较，还要验证少样本是否更强。

### 13.2 数据比例

```text
data fractions:
  5%
  10%
  25%
  50%
  100%
```

### 13.3 必跑方法

```text
MLP-autograd-reference
MLP-manual-linear-reference
T3+ManualAdamW
Best-v7-task-repair
Best-v7-crossgroup-repair, if S2
```

### 13.4 必须记录

```text
data_fraction
train_size
val_acc
test_acc
val_loss
ECE
NLL
time_to_val_acc_50
sample_efficiency_auc
data_to_target_acc
feature_rank
margin_p10
train_val_gap
```

### 13.5 判断标准

Sample efficiency useful if:

$$
\operatorname{AUC}_{data,KAN}
>
\operatorname{AUC}_{data,MLP}
$$

or:

$$
N_{\text{target,KAN}} < N_{\text{target,MLP}},
$$

where $N_{\text{target}}$ is the smallest training size reaching a fixed validation accuracy.

### 13.6 可视化

```text
p7_accuracy_vs_data_fraction.svg
p7_sample_efficiency_auc_bar.svg
p7_data_to_target_acc_bar.svg
p7_margin_vs_data_fraction.svg
```

---

## 14. P8：noise robustness and calibration diagnostic

### 14.1 目的

测试 PureKAN-NG 是否比 MLP 更稳健、更可校准。

### 14.2 Noise settings

```text
label_noise:
  0%
  5%
  10%
  20%

input_noise:
  gaussian sigma 0.05
  gaussian sigma 0.10
  random erasing small
  brightness/contrast perturbation
```

### 14.3 必跑方法

```text
MLP-autograd-reference
MLP-manual-linear-reference
T3+ManualAdamW
Best-v7-task-repair
Best-v7-crossgroup-repair, if S2
```

### 14.4 必须记录

```text
noise_type
noise_level
val_acc
test_acc
accuracy_drop_vs_clean
ECE
NLL
confidence_mean
wrong_confidence_mean
robustness_auc
calibration_auc
margin_mean
margin_p10
```

### 14.5 判断标准

Robustness useful if:

$$
\operatorname{AccDrop}_{KAN}
<
\operatorname{AccDrop}_{MLP}
$$

for at least two noise settings.

Calibration useful if:

$$
\operatorname{ECE}_{KAN}
<
\operatorname{ECE}_{MLP}-0.01.
$$

### 14.6 可视化

```text
p8_accuracy_under_noise.svg
p8_accuracy_drop_bar.svg
p8_ece_under_noise.svg
p8_robustness_auc_bar.svg
p8_confidence_wrong_correct_plot.svg
```

---

## 15. P9：patch/token scaling diagnostic

### 15.1 目的

v7.0 必须开始进入 patch/token vision scaling，但只能作为 diagnostic，除非 MNIST-family gate 已通过。

### 15.2 Task

```text
CIFAR-10-small or TinyImageNet-subset
patch size:
  4x4 or 8x8

architecture:
  ConvPatchMLP baseline
  PatchMLP baseline
  T3-PureKAN patch block
  T3+crossgroup-lite patch block, if S2
```

### 15.3 必须记录

```text
dataset
patch_size
num_tokens
hidden_dim
depth
method
memory_ratio_mean
step_ratio_mean
forward_ratio_mean
backward_ratio_mean
val_acc
test_acc
val_loss_auc_by_time
ECE
NLL
samples_per_second
tokens_per_second
scaling_slope_memory
scaling_slope_step
```

### 15.4 判断标准

Patch/token diagnostic pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

and:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.02.
$$

Scaling advantage if:

$$
\operatorname{slope}_{memory,KAN}<\operatorname{slope}_{memory,MLP}
$$

or:

$$
\operatorname{slope}_{step,KAN}<\operatorname{slope}_{step,MLP}.
$$

### 15.5 可视化

```text
p9_patch_token_accuracy.svg
p9_patch_token_memory_step_pareto.svg
p9_token_scaling_slope.svg
p9_tokens_per_second_bar.svg
p9_patch_task_loss_vs_time.svg
```

---

## 16. P10：candidate selection by beyond-MLP score

### 16.1 目的

P10 汇总 P2-P9，选择是否进入 official task 或下一轮 deeper scaling。

### 16.2 Survivor 类型

```text
B0:
  S1 efficiency + task within 1% of MLP + at least one beyond metric wins

B1:
  S1 efficiency + task gap <=2% + diagnostic beyond metric wins

B2:
  S2 efficiency + task improves by >=2% + memory still >1.0

B3:
  S2 efficiency but task gap persists >5%

B4:
  task improves but efficiency leaves S2

B5:
  no task or efficiency improvement

B6:
  gradient / contract fail
```

### 16.3 Beyond-MLP score

Define:

$$
S_{\text{beyond}}
=
w_1 S_{\text{eff}}
+
w_2 S_{\text{task}}
+
w_3 S_{\text{cal}}
+
w_4 S_{\text{sample}}
+
w_5 S_{\text{robust}}
+
w_6 S_{\text{geom}}.
$$

Default weights:

```text
w_eff = 0.25
w_task = 0.25
w_cal = 0.15
w_sample = 0.15
w_robust = 0.10
w_geom = 0.10
```

Efficiency score:

$$
S_{\text{eff}}
=
\frac{1}{2}
\left(
\frac{1}{r_{\text{mem}}}
+
\frac{1}{r_{\text{step}}}
\right).
$$

Task score:

$$
S_{\text{task}} =
\operatorname{Acc}_{KAN} - \operatorname{Acc}_{MLP}.
$$

Calibration score:

$$
S_{\text{cal}} =
\operatorname{ECE}_{MLP} - \operatorname{ECE}_{KAN}.
$$

Sample score:

$$
S_{\text{sample}} =
\operatorname{AUC}_{data,KAN} - \operatorname{AUC}_{data,MLP}.
$$

Robustness score:

$$
S_{\text{robust}} =
\operatorname{AccDrop}_{MLP} - \operatorname{AccDrop}_{KAN}.
$$

Geometry score:

$$
S_{\text{geom}} =
\operatorname{margin}_{p10,KAN} - \operatorname{margin}_{p10,MLP}.
$$

### 16.4 必须记录字段

```text
candidate
survivor_type
memory_ratio_mean
step_ratio_mean
val_acc
test_acc
ECE
NLL
sample_efficiency_auc
robustness_auc
feature_rank
margin_p10
patch_token_pass
S_eff
S_task
S_cal
S_sample
S_robust
S_geom
S_beyond
open_one_step_probe
open_official_task
open_diagnostic_task
```

### 16.5 可视化

```text
p10_beyond_score_bar.svg
p10_efficiency_task_calibration_pareto.svg
p10_survivor_type_dashboard.svg
p10_metric_radar_chart.svg
```

---

## 17. P11：one-step probe for selected candidates

### 17.1 目的

所有进入 official 或 diagnostic task 的 candidate 必须先通过 one-step descent 与 rollback。

### 17.2 方法

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

### 17.3 必须记录

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

### 17.4 判断标准

P11 pass:

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

### 17.5 可视化

```text
p11_loss_before_after.svg
p11_bad_step_heatmap.svg
p11_update_norm_vs_loss_delta.svg
p11_residual_ablation_probe.svg
```

---

## 18. P12：official task re-entry if S1/S0

### 18.1 打开条件

Official task opens if:

```text
survivor_type in {B0,B1}
efficiency survivor in {S0,S1}
P11 pass
r_mem < 1.0
r_step <= 1.35
grad pass
no fake/proxy
```

### 18.2 Methods

```text
MLP-autograd-reference
MLP-manual-linear-reference
T3-v620+ManualAdamW
Best-v7-S1+ManualAdamW
Best-v7-S1+best-diagnostic-optimizer
Best-v7-S1+best-crossgroup-repair
```

### 18.3 必须记录

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

### 18.4 判断标准

Official task pass:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.01
$$

on all datasets.

At least two datasets must satisfy:

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

### 18.5 可视化

```text
p12_val_loss_vs_step.svg
p12_val_loss_vs_time.svg
p12_accuracy_vs_time.svg
p12_time_to_target_bar.svg
p12_task_efficiency_pareto.svg
p12_seedwise_delta_plot.svg
```

---

## 19. P13：diagnostic task if S2 only

如果没有 S1/S0，但有 B2 candidate，则运行 diagnostic-only task。

### 19.1 Methods

```text
MLP-autograd-reference
MLP-manual-linear-reference
T3-v620+ManualAdamW
Best-v7-S2+best-diagnostic-recipe
```

### 19.2 判断标准

Diagnostic useful:

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to v6.20 T3 baseline.

If diagnostic useful but memory still $>1.0$, next cycle continues S1 memory repair.  
If diagnostic not useful, next cycle prioritizes expressivity redesign.

---

## 20. P14：route decision

### Route cases

```text
R1-BeyondMLP-S1-Pass:
  S1 efficiency, official task pass, at least one beyond metric wins.
  Move to confirm seeds and patch/token expansion.

R2-S1-EfficiencySolved-TaskGap:
  S1 efficiency solved, but task gap remains.
  Focus optimizer / cross-group / generalization.

R3-S2-TaskImproved:
  S2 holds, task improves by >=2%, but memory still >1.0.
  Continue memory repair.

R4-S2-TaskGapPersists:
  S2 holds, task gap remains >5%.
  Redesign grouped expressivity.

R5-MemoryS1Failed:
  memory remains around 1.0312.
  Continue buffer lifetime repair.

R6-TaskRepairTooExpensive:
  task improves but efficiency leaves S2.
  Diagnostic only.

R7-PatchTokenPromising:
  patch/token diagnostic passes while MNIST-family still incomplete.
  Continue scaling diagnostic but no final claim.

R8-NoImprovement:
  no memory or task improvement over v6.20.
  Need architecture redesign.

R9-ContractFail:
  nonKAN / fake / proxy / grad fail.
  Reject candidate.
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
survivor_type
memory_improvement_vs_T3
step_improvement_vs_T3
best_val_acc
best_test_acc
val_acc_gap_vs_MLP
test_acc_gap_vs_MLP
best_ECE
best_NLL
sample_efficiency_auc_delta
robustness_auc_delta
patch_token_pass
S_beyond
one_step_probe_pass
official_task_opened
diagnostic_task_opened
open_optimizer_exploration
open_functional_correction
primary_blocker
next_required_implementation
no_fake
no_proxy
```

---

## 21. P15：artifact and failure audit

### Failure types

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_s1_memory_margin_fail
F5_task_gap_generalization
F6_task_gap_optimizer
F7_task_gap_expressivity
F8_task_gap_calibration
F9_repair_efficiency_regression
F10_repair_nonKAN_violation
F11_sample_efficiency_fail
F12_robustness_fail
F13_patch_token_scaling_fail
F14_official_task_gated
F15_diagnostic_task_gated
F16_optimizer_gated
F17_functional_gated
F18_fake_or_proxy_violation
F19_artifact_missing
```

### Artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p1_s1_memory_attribution.csv
p2_s1_memory_repair.csv
p2_s1_memory_repair_detail.csv
p3_task_gap_attribution.csv
p3_task_trace.csv
p4_optimizer_regularization_diagnostic.csv
p4_optimizer_trace.csv
p5_crossgroup_expressivity_diagnostic.csv
p5_crossgroup_trace.csv
p6_longer_budget_task.csv
p6_longer_budget_trace.csv
p7_sample_efficiency.csv
p7_sample_efficiency_trace.csv
p8_noise_robustness.csv
p8_noise_robustness_trace.csv
p9_patch_token_scaling.csv
p9_patch_token_trace.csv
p10_candidate_selection.csv
p11_one_step_probe.csv
p12_official_task_reentry.csv
p12_task_trace.csv
p13_diagnostic_task.csv
p13_diagnostic_task_trace.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### Figures

```text
figures/p1_t3_memory_gap_waterfall.svg
figures/p2_memory_repair_pareto.svg
figures/p3_task_gap_taxonomy_dashboard.svg
figures/p4_optimizer_val_acc_bar.svg
figures/p5_crossgroup_repair_pareto.svg
figures/p6_long_budget_val_acc_curve.svg
figures/p7_sample_efficiency_auc_bar.svg
figures/p8_robustness_auc_bar.svg
figures/p9_patch_token_memory_step_pareto.svg
figures/p10_beyond_score_bar.svg
figures/p12_task_efficiency_pareto.svg
figures/p14_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 22. 最终成功与失败解释规则

### Case A：S1 + task pass + beyond metric win

如果 candidate 满足：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35,
$$

并且 official task pass，同时至少一个 beyond metric 超过 MLP，则可以称为：

```text
PureKAN-NG v7.0 candidate established.
```

但仍需 5-seed / 10-seed confirm。

### Case B：S1 过，但 task gap 仍大

如果 memory/time official gate 过了，但 task gap 仍大于 $2\%$，说明 efficiency 已基本解决，主 blocker 转为 task expressivity / generalization。下一轮不要继续只修 kernel。

### Case C：S2 过，task improved

如果仍是 S2，但 diagnostic task 提升至少 $2\%$，T3 继续主线。下一轮继续 memory repair toward S1。

### Case D：task improved but efficiency breaks

如果 accuracy 提升但 memory/step 离开 S2，该 repair 只能作为 diagnostic，不能进入 route selection。

### Case E：S2 holds but task gap persists

如果 S2 稳定，但 task gap 仍 $>5\%$，说明 T3 当前 grouped structure 可能表达力不足。下一轮应做 grouped expressivity redesign，而不是只调 optimizer。

### Case F：patch/token promising but MNIST-family incomplete

如果 patch/token diagnostic 有信号，但 MNIST-family 仍不过 task gate，patch/token 只能作为 scaling evidence，不作为 final success。

### Case G：no improvement over v6.20

如果 memory 仍是 `1.0312`，task gap仍是 `5%-7%`，sample/robust/scaling 也无优势，则 v7.0 应承认当前 T3 需要架构级 redesign，例如更强 cross-group function-space primitive。

---

## 23. 最终建议

v7.0 的一句话策略是：

$$
\boxed{
\text{以 T3 为 efficient kernel base，向 S1 memory 和 beyond-MLP task/generalization 双线推进。}
}
$$

本轮不应再问“PureKAN 能不能轻量化”。T3 已经证明它可以接近并在 step 上超过 MLP gate。现在真正的问题是：

```text
1. memory 还差约 3.03% 才进入 official gate；
2. task/generalization 还差约 5%-7% 才接近 MLP；
3. beyond-MLP claim 需要 sample efficiency、robustness、calibration、patch/token scaling 中至少一个明确优势。
```

v7.0 若成功，应不只是“替代 MLP”，而是给出一个更强的 PureKAN-NG 架构证据链。
