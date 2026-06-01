# DG-KAN v9.2.3 Fused Compositional FullEdge P4 Kernel Closure 与 Trainability Re-Entry 完整实验计划

> 本计划基于最新 `v922_fused_compositional_kernel_closure_h512r4_20260509T161500Z` terminal route 制定。  
> 最新结果不是 FullEdge 成功，也不是 compositional route 失败，而是把问题定位到一个更具体的系统-表达力交界处：  
> **fused D2 compiled path 的 gradient correctness 与 synthetic interaction retention 已成立，但 P4 kernel-native gate 仍未闭合，因此 AdamW-only trainability、functional update、external fair validation 都不能打开。**  
> v9.2.3 的目标不是继续换 basis、不是继续调 lr、不是用 functional update 救 P5，也不是回到 transitional route。  
> v9.2.3 只做一件核心事：**把 D2 compositional FullEdge 的 forward / backward / memory path 真正做成 MLP-comparable，然后重新打开 P5。**

---

## 0. 最新状态与独立判断

### 0.1 最新 terminal route

最新 artifact：

```text
results/real_rerun_20260506/v922_fused_compositional_kernel_closure_h512r4_20260509T161500Z/
```

最终 route：

```text
route = R2-CompositionalKernelRequired
success_v922_kernel_closure = false
success_v922_trainability_repair = false
success_v922_functional_opened = false
```

关键结果：

```text
GradPass = 1
GradRelErrMax = 9.247307e-06
GradCosMin = 1.0
synthetic pairwise R2 = 0.9911209941
P4 kernel-native pass = 0
```

P4 子 gate：

```text
forward ratio  = 1.412113  fail
backward ratio = 2.072152  fail
step ratio     = 1.387424  pass
memory ratio   = 1.213611  fail
```

No-fake audit：

```text
rows_checked = 36
fake/proxy/offload = 0
```

这说明当前结果很干净：correctness、interaction retention、no-fake/no-proxy 都成立，但系统 gate 不成立。P3 AdamW trainability 和 functional update 没有打开是正确的。

### 0.2 目标是否达成

没有达到目标。

v9.2.2 的目标链条是：

```text
D2 compositional FullEdge
  -> GradPass
  -> synthetic interaction retention
  -> P4 kernel-native pass
  -> P5 AdamW-only trainability
  -> gated functional update
  -> external fair validation
```

当前只完成到：

```text
GradPass = 1
synthetic interaction retention = 1
P4 kernel-native pass = 0
```

因此不能声明：

```text
FullEdge PureKAN success
P5 trainability repaired
functional advantage
external fair success
```

### 0.3 这次实验真正推进了什么

这次实验比上一轮 generic D2 失败更进一步。上一轮 v9.2.2 已经证明 depth2 在 synthetic pairwise-product 上可以把 R2 从 depth1 的负值提升到约 `0.991`，说明 composition 是真实必要方向。最新 fused D2 compiled path 进一步证明：这种 interaction capacity 没有因为 fused/compiled 实现而丢失，gradient correctness 也成立。

因此当前最重要的正向结论是：

$$
\boxed{
\text{D2 compositional PureKAN 的函数表达机制成立，且当前 fused path 的反传正确。}
}
$$

当前最重要的负向结论是：

$$
\boxed{
\text{D2 compositional PureKAN 的 kernel-native P4 还没有闭合，尤其 backward 和 memory 明显超线。}
}
$$

### 0.4 不要被 step ratio pass 误导

step ratio 是：

$$
1.387424\leq1.50.
$$

它是 pass。但 P4 不是只看 step。forward、backward、memory 三个子 gate 同时失败：

$$
1.412113>1.25,
$$

$$
2.072152>1.50,
$$

$$
1.213611>1.05.
$$

因此不能用 step pass 宣称 P4 pass。更准确地说：**end-to-end step 已经接近可用，但 phase-level P4 仍不合格；尤其 backward 和 live-set memory 是当前主 blocker。**

---

## 1. v9.2.3 整体目标

v9.2.3 的整体目标是：

$$
\boxed{
\text{关闭 D2 compositional FullEdge 的 P4 kernel-native gap，并重新打开 P5 AdamW-only trainability。}
}
$$

v9.2.3 的 minimum success 是：

```text
1. FullEdgeEquivalencePass = 1
2. NoExternalResidualPass = 1
3. GradPass = 1
4. synthetic pairwise R2 >= 0.95
5. P4 kernel-native pass = 1
```

其中 P4 kernel-native pass 定义为：

$$
T_{\text{forward,KAN}}\leq1.25T_{\text{forward,MLP-match}},
$$

$$
T_{\text{backward,KAN}}\leq1.50T_{\text{backward,MLP-match}},
$$

$$
T_{\text{step,KAN}}\leq1.50T_{\text{step,MLP-match}},
$$

$$
M_{\text{peak,KAN}}\leq1.05M_{\text{peak,MLP-match}}.
$$

v9.2.3 的 trainability success 是：

$$
Acc_{\text{KAN-AdamW}}\geq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows，并且：

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

v9.2.3 的 strong success 是：

```text
minimum success
+
trainability success
+
functional_open_allowed = 1
```

本轮仍不以 external fair 为目标。External fair 只能在 P5 near-pass 或 P5 pass 后打开。

---

## 2. 不可违反的约束

### 2.1 Clean training contract

所有 official candidate 必须满足：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
teacher_logits_used = 0
distillation_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
```

训练目标：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

P5 前只允许 AdamW-equivalent update：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{\text{AdamW-equivalent}}.
$$

P5 前不允许：

$$
\Delta\theta_{\text{functional}}.
$$

### 2.2 Pure FullEdge equivalence contract

每一层必须能写成：

$$
h^{\ell+1}_j=\sum_i\phi^{\ell}_{ij}(h^\ell_i).
$$

每条 edge function 必须能写成：

$$
\phi^\ell_{ij}(x)=\sum_k c^\ell_{ij,k}B^\ell_k(\tilde{x}).
$$

允许 identity edge-basis channel：

$$
B_0(x)=x.
$$

允许 nonlinear edge-correction channel：

$$
\sum_{k\geq1}c_{ij,k}B_k(\tilde{x}).
$$

禁止外部 residual shortcut：

$$
h^{\ell+1}=F_{\text{edge}}(h^\ell)+h^\ell W_{\text{skip}},
$$

如果 $W_{\text{skip}}$ 是普通 trainable Linear。

禁止：

```text
ordinary MLP hidden path
ordinary Linear shortcut outside edge function
trainable Conv/Linear preprocessing
KAN output + non-KAN residual branch
node activation MLP disguised as low-rank factorization
```

### 2.3 P4 gating contract

P4 不通过，不得打开：

```text
P5 AdamW-only trainability
P6 functional update
P7 external fair validation
```

如果某个 candidate 有 synthetic interaction 与 GradPass，但 P4 fail，则 route 必须写为：

```text
CompositionalKernelRequired
```

或：

```text
CapacitySystemTradeoff
```

不得写为 trainability failure。

---

## 3. 当前 P4 gap 的定量分析

当前 D2 fused path 的 P4 fail 不是均匀失败。

### 3.1 Forward gap

当前：

$$
r_{\text{forward}}=1.412113.
$$

目标：

$$
r_{\text{forward}}\leq1.25.
$$

相对目标仍需下降：

$$
\frac{1.412113}{1.25}-1\approx0.129690.
$$

也就是约 `13%` 的 forward reduction。Forward 已经接近，但仍需要 fuse basis eval / projection / normalization small kernels。

### 3.2 Backward gap

当前：

$$
r_{\text{backward}}=2.072152.
$$

目标：

$$
r_{\text{backward}}\leq1.50.
$$

相对目标仍需下降：

$$
\frac{2.072152}{1.50}-1\approx0.381435.
$$

也就是约 `38%` 的 backward reduction。Backward 是最大 blocker。

### 3.3 Memory gap

当前：

$$
r_{\text{memory}}=1.213611.
$$

目标：

$$
r_{\text{memory}}\leq1.05.
$$

相对目标仍需下降：

$$
\frac{1.213611}{1.05}-1\approx0.155820.
$$

也就是约 `16%` 的 peak memory reduction。Memory blocker 说明当前仍保存了过多 compositional intermediate、basis derivative 或 compiled temporary。

### 3.4 Step 已经不是主 blocker

当前：

$$
r_{\text{step}}=1.387424.
$$

目标：

$$
r_{\text{step}}\leq1.50.
$$

Step 已经 pass。因此下一步不是优化 optimizer update，也不是调 event/functional cadence，而是聚焦：

```text
backward phase
memory live-set
forward small-kernel overhead
```

---

## 4. 核心假设

### H1：Backward fail 来自 layer1 backward 与 basis-derivative materialization

D2 backward 需要：

```text
layer2 backward -> dh1
layer1 backward -> dparams_layer1
```

当前 backward ratio 为 `2.072152`，远超目标。H1 假设主要瓶颈不是 layer2 head，而是 layer1 backward 中 basis derivative / coefficient gradient / dh propagation 的 materialization 或 kernel fragmentation。

H1 成立标准：

$$
T_{\text{backward-layer1}}\geq0.60T_{\text{backward-total}}.
$$

或者：

$$
M_{\text{layer1-derivative-temp}}\geq0.50(M_{\text{KAN-peak}}-M_{\text{MLP-peak}}).
$$

若 H1 成立，优先实现 layer1 streaming backward 与 fused coeffgrad。

### H2：Memory fail 来自 activation/cache/live-set，而不是 params 或 optimizer state

Memory ratio 当前为 `1.213611`。H2 假设超额主要来自：

```text
h1 activation live-set
layer1 basis cache
layer2 basis cache
basis derivative temp
compiled graph temp
```

而不是 params 或 AdamW optimizer state。

H2 成立标准：

$$
M_{\text{activation/cache/temp}}\geq0.60(M_{\text{KAN-peak}}-M_{\text{MLP-peak}}).
$$

若 H2 成立，优先做 recompute / checkpoint / streaming grad，而不是压参数。

### H3：Forward fail 来自 small-kernel / basis-eval / projection fragmentation

Forward ratio 当前为 `1.412113`，距离目标约 13%。H3 假设 forward 不是理论 FLOPs 过高，而是分散的 small kernels 和 basis/projection 未融合。

H3 成立标准：

```text
small_kernel_count_forward >= 5
```

或：

$$
T_{\text{basis-eval}}+T_{\text{projection}}\geq0.70T_{\text{forward-total}}.
$$

若 H3 成立，优先实现 fused basis eval + projection forward。

### H4：Recompute 可以降低 memory，但可能增加 backward；streaming 可以同时降低 memory 和 backward

H4 预期：

```text
store-h1-only / recompute-basis:
  memory 降，但 backward 可能升；

streaming coefficient grad:
  memory 降，同时 backward 降或持平；

no-input-dx-layer1:
  backward 与 memory 都可能降。
```

H4 成立标准：

至少一个 candidate 满足：

$$
M_{\text{new}}<M_{\text{current}}-0.10(M_{\text{current}}-M_{\text{MLP}}),
$$

同时：

$$
T_{\text{backward,new}}\leq T_{\text{backward,current}}.
$$

### H5：P4 修复不能破坏 synthetic interaction retention

任何 kernel/memory 修复如果把 D2 退化成近似 depth1 additive，不能进入 P5。

H5 成立标准：

$$
R^2_{\text{pairwise}}\geq0.95,
$$

并且：

$$
R^2_{\text{D2}}-R^2_{\text{D1}}\geq0.50.
$$

---

## 5. Candidate 设计

### 5.1 Baselines

```text
B0-MLP-match:
  同参数 MLP-match，CE-only，AdamW。

K0-S3-B2-P4-F0:
  v9.2 single-layer P4-pass candidate，P5 fail reference。

D2-current-fused-compiled:
  最新 D2 fused compiled path，GradPass 和 synthetic R2 pass，P4 fail。
```

### 5.2 Memory repair candidates

```text
M0-current:
  当前 D2 fused path。

M1-store-h1-only:
  forward 只保存 h1，不保存 layer1/layer2 basis。

M2-recompute-layer2-basis:
  backward 重算 layer2 basis 和 derivative。

M3-recompute-layer1-basis:
  backward 重算 layer1 basis 和 derivative。

M4-recompute-both-layers:
  只保存 x 与 h1，两个 layer basis 都重算。

M5-checkpoint-h1-compact:
  h1 使用 compact cache policy，梯度计算保持 FP32。

M6-stream-coeffgrad:
  backward 流式累计 coefficient grad，不保存完整 derivative tensor。

M7-no-input-dx-layer1:
  第一层不计算 raw input dx，只计算参数梯度。

M8-persistent-workspace:
  使用预分配 workspace，避免 compiled temp peak。
```

### 5.3 Backward repair candidates

```text
B1-fused-layer2-backward:
  fuse layer2 basis derivative + coeffgrad + dh1。

B2-fused-layer1-backward:
  fuse layer1 basis derivative + coeffgrad，避免 full dB/dx tensor。

B3-fused-both-layer-backward:
  layer1/layer2 backward 全部 fused。

B4-streaming-grad-layer1:
  layer1 coefficient gradient streaming accumulate。

B5-split-backward-two-stage:
  layer2 backward 完全结束并释放 temp 后再进入 layer1 backward。

B6-no-dx-input:
  第一层不计算 raw input dx。

B7-triton-layer1-backward:
  layer1 使用 Triton fused backward。

B8-triton-layer1-layer2-backward:
  两层都使用 Triton fused backward。
```

### 5.4 Forward repair candidates

```text
F1-fused-basis-projection-forward:
  fuse Chebyshev basis eval + low-rank projection。

F2-fused-source-normalization-basis:
  fuse EdgeAffineNorm + Chebyshev basis eval。

F3-no-materialized-basis-forward:
  直接计算 projected residual，不落盘 basis tensor。

F4-triton-forward-layer1:
  layer1 使用 Triton fused forward。

F5-triton-forward-layer1-layer2:
  两层均使用 Triton fused forward。
```

### 5.5 Combined P4 candidates

```text
C1-M4+B2+F1:
  recompute both layers + fused layer1 backward + fused forward projection。

C2-M6+B2+F3:
  streaming coeffgrad + layer1 fused backward + no-materialized-basis forward。

C3-M7+B3+F3:
  no input dx + fused both backward + no-materialized-basis forward。

C4-M8+B8+F5:
  persistent workspace + full Triton forward/backward。

C5-M4+B6+F2:
  recompute both + no input dx + source/basis fusion。
```

All C candidates must preserve:

```text
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
GradPass = 1
synthetic pairwise R2 >= 0.95
```

---

## 6. 实验阶段

## P0：route recap and measurement stability

### 目标

复现当前 D2 fused path 的 P4 fail 数字，确认 measurement 稳定。

### 必须记录

```text
candidate_id
implementation_id
depth
hidden_dim
rank
basis_channels
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
forward_ratio
backward_ratio
step_ratio
memory_ratio
kernel_count_total
small_kernel_count
unknown_time_fraction
full_edge_equivalence_pass
no_external_residual_pass
```

### 判断标准

Repeat pass：

$$
|r_{\text{forward}}-1.412113|\leq0.10,
$$

$$
|r_{\text{backward}}-2.072152|\leq0.15,
$$

$$
|r_{\text{memory}}-1.213611|\leq0.05.
$$

Timing/profiler sanity：

$$
unknown\_time\_fraction\leq0.10.
$$

### 可视化

```text
p0_repeat_p4_ratio_bar.svg
p0_p4_gate_dashboard.svg
p0_synthetic_r2_repeat.svg
```

---

## P1：P4 failure phase attribution

### 目标

把 P4 failure 拆到 layer 与 phase，判断应该先修 memory、backward 还是 forward。

### 必须记录

```text
candidate_id
layer_id
phase
forward_source_norm_ms
forward_basis_eval_ms
forward_projection_ms
forward_output_ms
backward_dlogit_ms
backward_layer2_basis_deriv_ms
backward_layer2_coeffgrad_ms
backward_dh1_ms
backward_layer1_basis_deriv_ms
backward_layer1_coeffgrad_ms
backward_dx_input_ms
optimizer_update_ms
kernel_count_phase
small_kernel_count_phase
allocated_MB_phase
reserved_MB_phase
largest_temp_tensor_MB_phase
unknown_time_fraction
```

### 判断标准

Attribution pass：

```text
phase_time_sum / total_time >= 0.90
phase_memory_sum / peak_memory >= 0.90
unknown_time_fraction <= 0.10
dominant_phase identified
```

Dominant phase must be one of:

```text
forward_basis_eval
forward_projection
backward_layer1_basis_deriv
backward_layer1_coeffgrad
backward_layer2_basis_deriv
backward_layer2_coeffgrad
dh1_propagation
activation_cache
compiled_temp
kernel_launch_overhead
```

### 可视化

```text
p1_forward_phase_waterfall.svg
p1_backward_phase_waterfall.svg
p1_memory_live_set_stacked.svg
p1_kernel_count_by_phase.svg
p1_largest_temp_tensor_by_phase.svg
```

---

## P2：memory live-set closure

### 目标

将 memory ratio 从 `1.213611` 降到 `<=1.05`。P2 不允许牺牲 synthetic interaction 或 gradient correctness。

### 必跑 candidates

```text
M0-M8
```

### 必须记录

```text
candidate_id
stores_h1
stores_layer1_basis
stores_layer2_basis
stores_layer1_deriv
stores_layer2_deriv
recompute_layer1_basis
recompute_layer2_basis
streams_coeffgrad
computes_input_dx_layer1
uses_persistent_workspace
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
forward_ratio
backward_ratio
step_ratio
memory_ratio
activation_cache_MB
basis_cache_MB
derivative_cache_MB
compiled_temp_MB
largest_temp_tensor_MB
```

### 判断标准

Memory closure pass：

$$
M_{\text{peak,KAN}}\leq1.05M_{\text{MLP-match}}.
$$

No regression：

$$
R^2_{\text{pairwise}}\geq0.95,
$$

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

### 可视化

```text
p2_memory_repair_pareto.svg
p2_cache_policy_memory_bar.svg
p2_memory_vs_backward_tradeoff.svg
p2_synthetic_retention_memory.svg
```

---

## P3：backward time closure

### 目标

将 backward ratio 从 `2.072152` 降到 `<=1.50`。P3 优先修 layer1 backward 和 derivative materialization。

### 必跑 candidates

```text
B1-B8
```

### 必须记录

```text
candidate_id
fused_layer1_backward
fused_layer2_backward
streaming_grad_layer1
split_backward_two_stage
no_dx_input
triton_layer1_backward
triton_layer2_backward
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
backward_ratio
step_ratio
memory_ratio
layer1_backward_ms
layer2_backward_ms
coeffgrad_ms
basis_deriv_ms
dh1_ms
dx_input_ms
```

### 判断标准

Backward closure pass：

$$
T_{\text{backward,KAN}}\leq1.50T_{\text{backward,MLP-match}}.
$$

No correctness regression：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

No interaction regression：

$$
R^2_{\text{pairwise}}\geq0.95.
$$

### 可视化

```text
p3_backward_repair_pareto.svg
p3_layer_backward_breakdown.svg
p3_backward_memory_tradeoff.svg
p3_grad_correctness_by_backward_candidate.svg
```

---

## P4：forward time closure

### 目标

将 forward ratio 从 `1.412113` 降到 `<=1.25`。Forward gap 较小，但必须闭合。

### 必跑 candidates

```text
F1-F5
```

### 必须记录

```text
candidate_id
fused_source_norm
fused_basis_eval
fused_projection
materializes_basis_forward
triton_forward_layer1
triton_forward_layer2
forward_ratio
step_ratio
memory_ratio
kernel_count_forward
small_kernel_count_forward
basis_eval_ms
projection_ms
source_norm_ms
```

### 判断标准

Forward closure pass：

$$
T_{\text{forward,KAN}}\leq1.25T_{\text{forward,MLP-match}}.
$$

No memory regression：

$$
M_{\text{peak,new}}\leq M_{\text{peak,current}}.
$$

### 可视化

```text
p4_forward_repair_pareto.svg
p4_forward_kernel_count.svg
p4_forward_phase_breakdown.svg
```

---

## P5：combined P4 gate closure

### 目标

组合 P2/P3/P4 survivors，形成真正 P4-pass D2 candidate。

### 必跑 candidates

```text
C1-C5
```

### 必须记录

```text
candidate_id
memory_candidate
backward_candidate
forward_candidate
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
forward_ratio
backward_ratio
step_ratio
memory_ratio
kernel_count_total
small_kernel_count_total
full_edge_equivalence_pass
no_external_residual_pass
P4_kernel_native_pass
```

### 判断标准

P4 pass：

$$
T_{\text{forward,KAN}}\leq1.25T_{\text{forward,MLP-match}},
$$

$$
T_{\text{backward,KAN}}\leq1.50T_{\text{backward,MLP-match}},
$$

$$
T_{\text{step,KAN}}\leq1.50T_{\text{step,MLP-match}},
$$

$$
M_{\text{peak,KAN}}\leq1.05M_{\text{MLP-match}}.
$$

Synthetic retention：

$$
R^2_{\text{pairwise}}\geq0.95.
$$

Gradient correctness:

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

### 可视化

```text
p5_combined_p4_gate_dashboard.svg
p5_p4_candidate_pareto.svg
p5_ratio_before_after.svg
```

---

## P6：AdamW-only trainability re-entry

### 目标

只有 P5 出现 P4-pass candidate 后才打开 P6。测试 D2 compositional FullEdge 是否修复 K0 的 trainability gap。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
functional_update = off
baseline = MLP-match
candidate = best P5 P4-pass D2
```

### 必须记录

```text
candidate_id
dataset
seed
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
delta_vs_mlp_match
CE_p50
CE_p90
CE_p99
logit_norm_mean
margin_p10
wrong_confidence_p95
ECE
NLL
interaction_score
layer1_activation_rank
layer2_activation_rank
basis_channel_usage
forward_ratio
backward_ratio
step_ratio
memory_ratio
```

### 判断标准

Trainability gain over K0：

$$
Acc_{\text{D2}}\geq Acc_{\text{K0}}+0.05
$$

on macro mean，或：

$$
Acc_{\text{D2,KMNIST}}\geq Acc_{\text{K0,KMNIST}}+0.05.
$$

P5 near-pass：

$$
Acc_{\text{D2}}\geq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows，并且：

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

P5 pass：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

### 可视化

```text
p6_trainability_gap_bar.svg
p6_train_loss_curve.svg
p6_ce_tail_by_candidate.svg
p6_margin_distribution.svg
p6_task_system_pareto.svg
p6_interaction_score_vs_accuracy.svg
```

---

## P7：functional update open decision

### 目标

决定 functional update 是否允许打开。P7 不追求 functional success，只做 gate decision。

### Functional open criteria

```text
P4_kernel_native_pass = 1
P5_near_pass = 1
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
```

### 必须记录

```text
candidate_id
p4_kernel_native_pass
p5_near_pass
p5_pass
functional_open_allowed
functional_not_open_reason
```

### 判断标准

只有全部满足：

```text
P4_kernel_native_pass = 1
P5_near_pass = 1
```

才允许：

```text
functional_open_allowed = 1
```

### 可视化

```text
p7_functional_open_decision.svg
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_equivalence_audit_v923.csv
p0_route_recap_measurement_stability.csv
p1_p4_failure_phase_attribution.csv
p2_memory_live_set_closure.csv
p3_backward_time_closure.csv
p4_forward_time_closure.csv
p5_combined_p4_gate_closure.csv
p6_adamw_trainability_reentry.csv
p7_functional_open_decision.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_full_edge_equivalence_fail
F3_external_residual_violation
F4_measurement_instability
F5_profiler_incomplete
F6_memory_live_set_fail
F7_backward_time_fail
F8_forward_time_fail
F9_combined_p4_fail
F10_synthetic_interaction_lost
F11_grad_fail
F12_adamw_trainability_fail
F13_functional_not_opened
F14_fake_or_proxy_violation
F15_artifact_missing
```

---

## 8. Route decision

### Route cases

```text
R1-D2P4ClosedP5Opened:
  D2 compositional FullEdge passes P4 and P5 trainability opens.

R2-D2P4ClosedP5NearPass:
  D2 passes P4 and reaches P5 near-pass.

R3-MemoryLiveSetBlocker:
  memory ratio remains > 1.05 after recompute/streaming attempts.

R4-BackwardKernelBlocker:
  backward ratio remains > 1.50 after fused/streaming attempts.

R5-ForwardKernelBlocker:
  forward ratio remains > 1.25 after fusion attempts.

R6-InteractionLostDuringKernelization:
  P4 pass but synthetic pairwise R2 < 0.95.

R7-GradCorrectnessFail:
  fused path cannot keep GradPass.

R8-P4ClosedButTrainabilityFail:
  P4 closes, but AdamW trainability still fails.

R9-ProfilerIncomplete:
  unknown_time_fraction > 0.10 or phase attribution incomplete.

R10-ContractFail:
  any PureKAN / no-teacher / no-loss / no-fake contract violated.
```

### route_decision.json 必须记录

```text
route
best_candidate
best_memory_repair
best_backward_repair
best_forward_repair
best_combined_candidate
full_edge_equivalence_pass
no_external_residual_pass
grad_pass
synthetic_pairwise_R2
forward_ratio
backward_ratio
step_ratio
memory_ratio
p4_kernel_native_pass
p5_trainability_opened
p5_near_pass
functional_open_allowed
primary_blocker
next_required_implementation
success_v923_p4_kernel_closure
success_v923_trainability_reentry
success_v923_functional_opened
```

---

## 9. 第一轮执行顺序

```text
Step 1:
  P0 repeat 当前 fused D2 path，确认 measurement stable。

Step 2:
  P1 做 P4 failure phase attribution。
  unknown_time_fraction > 0.10 时先修 profiler。

Step 3:
  P2 memory live-set closure。
  先把 memory ratio 1.213611 拉到 <= 1.05。

Step 4:
  P3 backward time closure。
  重点把 backward ratio 2.072152 拉到 <= 1.50。

Step 5:
  P4 forward time closure。
  把 forward ratio 1.412113 拉到 <= 1.25。

Step 6:
  P5 combined P4 gate closure。
  组合 memory/backward/forward survivors。

Step 7:
  只有 P5 P4 pass 后，才打开 P6 AdamW-only trainability。

Step 8:
  只有 P6 near-pass 后，才允许 P7 functional update open。
```

---

## 10. 最终解释规则

### Case A：P4 closed and P5 opens

可以声明：

```text
v9.2.3 closed compositional FullEdge kernel-native gate and reopened AdamW-only trainability.
```

但仍不能声明 functional advantage 或 external fair success。

### Case B：memory remains blocker

必须声明：

```text
Compositional FullEdge expression is correct, but current live-set cannot meet MLP-comparable memory. Next work requires deeper streaming / one-buffer backward.
```

### Case C：backward remains blocker

必须声明：

```text
Compositional FullEdge interaction works, but backward kernel is not MLP-comparable. Next work requires custom Triton/CUDA backward.
```

### Case D：P4 pass loses interaction

必须声明：

```text
Kernelization closed system gate by destroying compositional interaction; this candidate is not valid.
```

### Case E：P4 closes but P5 fails

必须声明：

```text
The blocker moved from kernel-native feasibility to AdamW trainability. Functional update still cannot open until near-pass.
```

---

## 11. 最终建议

v9.2.3 的一句话策略是：

$$
\boxed{
\text{不要再证明 D2 有 interaction，也不要继续调 optimizer；先把 D2 的 backward/live-set kernel path 做到 MLP-comparable。}
}
$$

当前最重要的不是 task，而是以下三个数字：

```text
forward ratio  1.412113 -> <= 1.25
backward ratio 2.072152 -> <= 1.50
memory ratio   1.213611 -> <= 1.05
```

只要这三个数字没过，P5/P6/P7 都不应该打开。真正的下一步是 phase attribution、memory live-set closure、backward streaming/fusion、forward fusion，然后才是 AdamW trainability re-entry。
