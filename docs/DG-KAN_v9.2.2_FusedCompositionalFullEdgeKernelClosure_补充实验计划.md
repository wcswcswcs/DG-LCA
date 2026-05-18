# DG-KAN v9.2.2 Fused Compositional FullEdge Kernel Closure 补充实验计划

> 本文件是 v9.2.2 的后续补充计划，仍然使用 **v9.2.2** 版本号，不另起 v9.3。  
> 最新 v9.2.2 执行结果已经把问题定位得更清楚：**compositionality 是必要方向，但当前 generic compositional FullEdge vision candidates 全部卡在 P4 kernel-native gate，导致 P5 trainability 不能打开。**  
> 因此，本补充计划不再继续做单层 active-k/rank 小修，也不直接打开 functional update 或 external fair validation。  
> 本轮唯一主线是：**实现并验证 fused compositional FullEdge kernel-native path，使 depth2/depth3 PureKAN composition 在前向、反传、step time、显存上重新进入同参数 MLP-match envelope，然后再打开 AdamW-only trainability。**

---

## 0. 当前结果的独立判断

### 0.1 v9.2.2 没有达到目标

最新 route 是：

```text
route = R2-CompositionalFullEdgeRequired
best_candidate = S3-B2-P4-F0
best_depth = D2
success_v922_trainability_repair = false
success_v922_functional_opened = false
success_v922_external_fair_opened = false
```

这说明 v9.2.2 没有达到 P5 trainability repair，更没有打开 functional update 或 external fair validation。不能把 synthetic interaction 的成功包装成 FullEdge vision success，也不能把 D2 记为 official survivor。

### 0.2 v9.2.2 的真正进展

v9.2.2 最重要的新证据是 synthetic interaction 诊断。depth1 在 pairwise-product target 上：

```text
R2 = -0.187756
```

而 depth2 达到：

```text
R2 = 0.991121
```

差值为：

```text
+1.178877
```

这说明单层 additive edge classifier 缺少跨特征交互，depth/compositionality 对 FullEdge 是真实必要方向。这个结论比继续增加单层 active-k 或 rank 更本质。

### 0.3 当前不能得出的结论

不能得出：

```text
1. FullEdge PureKAN 已经成功；
2. D2 vision candidate 已经优于 MLP；
3. functional update 可以打开；
4. external fair validation 可以打开；
5. compositional FullEdge 在系统上不可能；
6. 当前失败只是 optimizer 问题。
```

因为本轮 generic compositional vision candidates 全部 P4 kernel-native gate fail：

```text
compositional_p4_kernel_native_pass_count = 0
```

因此 P5 trainability 没有资格打开。本轮最多证明：

$$
\boxed{
\text{Depth/compositionality 是必要方向，但 compositional FullEdge 的 kernel-native implementation 尚未闭合。}
}
$$

### 0.4 当前 blocker 的准确定义

当前 blocker 已经不是 v9.1 的 basis-source conditioning，也不是 v9.2 first-wave 的单层 P4 kernel feasibility。当前 blocker 是：

$$
\boxed{
\text{compositional FullEdge 的 fused forward/backward/memory path 未实现到 MLP-comparable envelope。}
}
$$

更具体地说，`K0 = S3-B2-P4-F0` 仍然保留 v9.2/v9.2.1 已测 P4 pass：

```text
k0_p4_kernel_native_pass = 1
```

但 K0 是单层/浅层 active-k shared edge-correction candidate，在 P5 AdamW-only 下输给 MLP-match。v9.2.2 证明 depth2 可以解决 synthetic interaction，但 D1/D2/D3 generic compositional vision candidates 又全部 P4 fail。因此下一步不是继续单层修补，而是实现：

```text
fused compositional FullEdge kernel
recompute/streaming backward
bounded activation/cache policy
layerwise materialization-free contraction
```

---

## 1. v9.2.2 补充计划的整体目标

本补充计划的整体目标是：

$$
\boxed{
\text{把 synthetic depth2 compositional signal 转化为 kernel-native, graph-free, PureKAN compositional FullEdge candidate。}
}
$$

这包含两个逐级目标。

### 1.1 Minimum success

最低成功是得到至少一个 depth2 compositional FullEdge candidate，同时满足：

```text
FullEdge equivalence pass
No external residual pass
No MLP hidden path pass
Grad correctness pass
P4 kernel-native pass
Synthetic interaction retained
```

形式化为：

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

并且：

$$
GradRelErr_{\max}\leq10^{-4},
$$

$$
GradCos_{\min}\geq0.999.
$$

### 1.2 Trainability success

在 minimum success 后，重新打开 P5 AdamW-only trainability。P5 near-pass 定义为：

$$
Acc_{\text{KAN-AdamW}}\geq Acc_{\text{MLP-match}}-0.01
$$

在至少 6/9 rows 上成立，并且 macro mean gap 满足：

$$
\Delta Acc_{\text{macro}}\geq -0.01.
$$

P5 pass 定义为：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

同时 P4 gate 仍必须保持通过。

### 1.3 本补充计划不追求的目标

本轮不追求：

```text
1. functional update success；
2. external fair success；
3. robustness / continual success；
4. Conv / Transformer extension；
5. 大规模 optimizer sweep；
6. 通过 external residual 或 MLP shortcut 修 task。
```

Functional update 只有在 P5 near-pass 后才允许 gated open。External fair validation 只有在 P5 pass 或 P5 near-pass + functional useful 后才允许 gated open。

---

## 2. 不可违反的 PureKAN contract

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

训练目标是：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

P5 前只允许 AdamW-equivalent update：

$$
\theta_{t+1}
=
\theta_t+\Delta\theta_{\text{AdamW-equivalent}}.
$$

不允许在 P5 前加入：

$$
\Delta\theta_{\text{functional}}.
$$

### 2.2 Compositional FullEdge equivalence contract

每一层必须满足：

$$
h^{\ell+1}_j=\sum_i\phi^{\ell}_{ij}(h^\ell_i).
$$

每条 edge function 必须满足：

$$
\phi^\ell_{ij}(x)=\sum_k c^\ell_{ij,k}B^\ell_k(\tilde{x}).
$$

允许 identity edge-basis channel：

$$
B_0(x)=x.
$$

允许 nonlinear edge-correction channels：

$$
\sum_{k\geq1}c^\ell_{ij,k}B_k(\tilde{x}).
$$

### 2.3 禁止的结构

以下结构不得进入 official route：

$$
h^{\ell+1}=F_{\text{edge}}(h^\ell)+h^\ell W_{\text{skip}},
$$

如果 $W_{\text{skip}}$ 是普通 trainable Linear。

不得使用：

```text
ordinary MLP hidden path
ordinary Linear shortcut outside edge function
trainable Conv/Linear preprocessing
KAN output + non-KAN residual branch
node activation MLP disguised as low-rank factorization
```

### 2.4 合法 low-rank 条件

允许低秩 edge coefficient factorization：

$$
C_{i,j,k}=\sum_{r=1}^{R}u_{i,r}v_{j,r}a_{k,r}.
$$

它必须能等价还原为：

$$
y_j=\sum_i\sum_kC_{i,j,k}B_k(\tilde{x}_i).
$$

不允许变成：

$$
r=\sigma(xU),\quad y=rV,
$$

除非严格证明 $\sigma$ 不是 node activation hidden path，而是 edge-basis contraction 的等价实现。

所有 low-rank candidate 必须记录：

```text
lowrank_edge_factorization_pass
hidden_activation_introduced
ordinary_mlp_hidden_path_used
edge_coefficient_tensor_equivalent
```

---

## 3. 核心假设

### H1：当前 compositional P4 fail 来自 generic implementation，而不是 composition 本身不可行

v9.2.2 中 D2 synthetic interaction 成功说明 depth2 composition 有表达力。但 generic compositional vision candidates P4 全 fail。H1 假设：P4 fail 主要来自高层实现的 materialization、kernel launch、activation cache 或 backward live-set，而不是 compositional FullEdge 理论上必然比 MLP 慢。

H1 成立标准：

通过 fused/recompute compositional implementation 后，至少一个 D2 candidate 满足：

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

若 fused/recompute implementation 仍全部 P4 fail，则应判定：

```text
R5-CompositionalKernelInfeasibleUnderCurrentFC
```

并转向 KANConv-like patch-local primitive 或更低层 fused kernel design。

### H2：compositional FullEdge 必须避免 layerwise dense edge tensor materialization

Depth2 不能构造：

$$
B\times d_{in}\times d_{hidden}\times K
$$

和：

$$
B\times d_{hidden}\times d_{out}\times K
$$

的完整 dense edge tensors。

官方 candidate 必须记录：

```text
materializes_dense_edge_tensor_layer1 = 0
materializes_dense_edge_tensor_layer2 = 0
peak_intermediate_MB_layer1
peak_intermediate_MB_layer2
```

H2 成立标准：

$$
M_{\text{activation/cache,KAN}}\leq1.05M_{\text{activation/cache,MLP-match}}.
$$

### H3：depth2 composition 要保留 synthetic interaction 能力

不能为了 P4 通过把 D2 压回近似 additive。Kernelized D2 必须保留 pairwise-product synthetic fit。

H3 成立标准：

$$
R^2_{\text{pairwise-product,D2}}\geq0.95.
$$

并且相比 depth1：

$$
R^2_{\text{D2}}-R^2_{\text{D1}}\geq0.50.
$$

如果 P4 pass 但 synthetic interaction score 下降到近似 depth1，则说明 kernelization 破坏了 composition 表达力。

### H4：P5 trainability 要在 composition P4 pass 后重新评估

当前 K0 P5 fail 不等于 compositional FullEdge P5 fail。H4 成立标准：

当 D2 P4 pass 后，P5 near-pass 应至少明显优于 K0：

$$
Acc_{\text{D2}}-Acc_{\text{K0}}\geq0.05
$$

on macro mean，或：

$$
Acc_{\text{D2,KMNIST}}-Acc_{\text{K0,KMNIST}}\geq0.05.
$$

若 D2 P4 pass 且 task gain 大，但仍未 near-pass，则继续分析 source / margin / basis organization。若 D2 P4 pass 但无 task gain，说明 synthetic interaction 能力没有转化到 vision task。

### H5：如果 fixed patch source 明显改善，下一步应转 KANConv 而不是继续 FC

v9.2.2 中 patch_source_evidence=0，但该阶段可能没有在 P4-pass compositional candidate 上充分打开。若后续 fixed patch source 对 D2 有明显提升：

$$
Acc_{\text{PatchSource}}-Acc_{\text{FlatSource}}\geq0.03,
$$

则 route 应指向：

```text
KANConv-like patch-local FullEdge primitive
```

而不是继续在 flattened FC FullEdge 上堆 depth。

---

## 4. Candidate 设计

### 4.1 Baselines

```text
B0-MLP-match:
  同参数量 MLP-match，CE-only，AdamW。

K0-S3-B2-P4-F0:
  v9.2/v9.2.1 已通过 P4 的 single-layer active-k FullEdge candidate。

D1-generic-depth1:
  v9.2.2 depth1 generic composition baseline。

D2-generic-depth2:
  v9.2.2 synthetic interaction positive but P4-failed reference。

D3-generic-depth3:
  v9.2.2 P4-failed reference。
```

### 4.2 Fused compositional implementation candidates

```text
FC0-GenericReference:
  当前 generic D2/D3 高层实现，只作为 fail reference。

FC1-LayerwiseTorchCompile:
  每层 FullEdge forward/backward 使用 torch.compile fused path。

FC2-RecomputeBackward:
  forward 只保存 layer input 和必要 normalization stats，backward 重算 edge basis。

FC3-FusedBasisProjection:
  fuse basis eval + low-rank residual projection，不 materialize dense edge tensor。

FC4-TritonLayer1:
  layer1 使用 Triton fused basis/projection，layer2 用 compile path。

FC5-TritonLayer1Layer2:
  layer1/layer2 均使用 Triton fused path。

FC6-CheckpointedComposition:
  depth2 只保存 h1 compact representation，不保存 layer1 basis；backward recompute layer1 basis。

FC7-StreamingGradComposition:
  backward 中流式累积 edge coefficient grad，避免保存 dB/dx dense tensor。
```

### 4.3 Width / rank / depth candidates

只允许 small pre-registered grid：

```text
depth = 2,3
hidden_dim = 128,256,512
rank = 1,2,4
basis_channels = T2, T2T3
```

不得用 test set 选 candidate。先过 P4，再用 validation score 选择 P5 survivor。

### 4.4 Source candidates

```text
S3-EdgeAffineNorm:
  当前 survivor source。

S4-FixedPatchPool:
  fixed patch pooling diagnostic。

S6-FixedPatchPoolEdgeAffineNorm:
  fixed patch source + EdgeAffineNorm。

S7-FixedLocalContrast:
  fixed local contrast source diagnostic。
```

所有 source 必须：

```text
trainable_preprocessing_used = 0
```

### 4.5 Gated functional candidates

默认不打开。只有 P5 near-pass 后才允许：

```text
F0-AdamW-only
F1-CorrectionChannelOnlyFunctional
F2-HighCurvatureCorrectionFunctional
F3-NoOpMatchedOverhead
F4-RandomDirection
```

Functional update 不允许修改 identity edge channel，默认只作用于 nonlinear edge-correction channels：

```text
identity_channel_modified = 0
correction_channels_modified = 1
```

---

## 5. 实验阶段

## P0：Current route audit and failure reproduction

### 目标

确认最新 v9.2.2 结果与 artifact 对齐，复现 D2 synthetic interaction success 与 generic D2/D3 P4 fail。P0 不改模型，只建立后续比较基线。

### 必须记录

```text
candidate
depth
source
basis
implementation
synthetic_pairwise_R2
synthetic_additive_R2
P4_forward_ratio
P4_backward_ratio
P4_step_ratio
P4_memory_ratio
P4_pass
full_edge_equivalence_pass
no_external_residual_pass
materializes_dense_edge_tensor
```

### 判断标准

P0 通过需要复现：

$$
R^2_{\text{D2,pairwise}}\geq0.95,
$$

并且：

```text
generic_D2_P4_pass = 0
generic_D3_P4_pass = 0
K0_P4_pass = 1
```

### 可视化

```text
p0_route_recap_table.md
p0_synthetic_depth_r2_bar.svg
p0_generic_composition_p4_failure.svg
```

---

## P1：Compositional P4 failure attribution

### 目标

定位 D2/D3 P4 fail 的具体来源：forward、backward、activation cache、basis eval、projection、optimizer update、kernel launch、unknown time 哪个占主导。

### 必须记录

```text
candidate
depth
implementation
forward_time_ms
backward_time_ms
step_time_ms
peak_memory_MB
activation_cache_MB
basis_eval_time_layer1
basis_eval_time_layer2
projection_time_layer1
projection_time_layer2
backward_basis_time_layer1
backward_basis_time_layer2
backward_projection_time_layer1
backward_projection_time_layer2
optimizer_update_time_ms
kernel_count_total
small_kernel_count
unknown_time_fraction
peak_tensor_shape_layer1
peak_tensor_shape_layer2
materialized_MB_layer1
materialized_MB_layer2
```

### 判断标准

P1 必须给出 dominant blocker：

```text
dominant_phase = one of:
  forward_basis_eval
  forward_projection
  backward_basis
  backward_projection
  activation_cache
  optimizer_update
  kernel_launch_overhead
  unknown
```

若：

$$
unknown\_time\_fraction>0.10,
$$

则不能进入 P2 kernel selection，必须先修 profiler。

### 可视化

```text
p1_phase_time_breakdown_stacked.svg
p1_memory_live_set_breakdown.svg
p1_kernel_count_by_phase.svg
p1_dominant_blocker_waterfall.svg
```

---

## P2：Fused compositional kernel prototypes

### 目标

实现多种 fused/recompute/streaming compositional path，并验证 correctness、P4 gate、synthetic interaction retention。P2 是本轮核心。

### 必跑 candidates

```text
FC0-GenericReference
FC1-LayerwiseTorchCompile
FC2-RecomputeBackward
FC3-FusedBasisProjection
FC4-TritonLayer1
FC5-TritonLayer1Layer2
FC6-CheckpointedComposition
FC7-StreamingGradComposition
```

### 必须记录

Correctness：

```text
candidate
GradRelErrMax
GradCosMin
OutputAbsDiffMax
DxAbsDiffMax
ParamGradAbsDiffMax
GradPass
```

P4 metrics：

```text
candidate
depth
hidden_dim
rank
basis_channels
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
forward_FLOPs_ratio
backward_FLOPs_ratio
kernel_count_total
small_kernel_count
materializes_dense_edge_tensor_layer1
materializes_dense_edge_tensor_layer2
P4_kernel_native_pass
```

Synthetic retention：

```text
candidate
synthetic_additive_R2
synthetic_pairwise_R2
synthetic_xor_acc
synthetic_composition_R2
```

### 判断标准

GradPass：

$$
GradRelErr_{\max}\leq10^{-4},
$$

$$
GradCos_{\min}\geq0.999.
$$

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
M_{\text{peak,KAN}}\leq1.05M_{\text{peak,MLP-match}}.
$$

Synthetic interaction retention:

$$
R^2_{\text{pairwise}}\geq0.95.
$$

A candidate can enter P3 only if:

```text
GradPass = 1
P4_kernel_native_pass = 1
synthetic_pairwise_R2 >= 0.95
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
```

### 可视化

```text
p2_kernel_candidate_correctness.svg
p2_kernel_candidate_p4_pareto.svg
p2_synthetic_retention_by_kernel.svg
p2_memory_vs_interaction_scatter.svg
```

---

## P3：Compositional AdamW-only trainability

### 目标

对 P2 survivors 重新打开 P5 AdamW-only trainability，判断 composition 是否修复 K0 的 task gap。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
functional_update = off
baseline = MLP-match
```

### 必须记录

```text
candidate
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
step_ratio
memory_ratio
```

### 判断标准

Task gain over K0：

$$
Acc_{\text{candidate}}\geq Acc_{\text{K0}}+0.05
$$

on macro mean，或：

$$
Acc_{\text{candidate,KMNIST}}\geq Acc_{\text{K0,KMNIST}}+0.05.
$$

P5 near-pass：

$$
Acc_{\text{candidate}}\geq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows，并且:

$$
\Delta Acc_{\text{macro}}\geq -0.01.
$$

P5 pass：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

### 可视化

```text
p3_trainability_gap_bar.svg
p3_train_loss_curve.svg
p3_ce_tail_by_candidate.svg
p3_margin_distribution.svg
p3_task_system_pareto.svg
p3_interaction_score_vs_accuracy.svg
```

---

## P4：Depth / width / rank Pareto under fused kernel

### 目标

在 P2/P3 survivors 上做小型 Pareto，不让模型只因过窄而 task fail，也不因过宽而 P4 fail。

### Grid

```text
depth = 2,3
hidden_dim = 128,256,512
rank = 1,2,4
basis_channels = T2,T2T3
```

### 必须记录

```text
candidate
depth
hidden_dim
rank
basis_channels
params_ratio
forward_ratio
backward_ratio
step_ratio
memory_ratio
val_acc
test_acc
delta_vs_mlp_match
CE_p99
margin_p10
P4_pass
P5_near_pass
```

### 判断标准

Pareto survivor：

$$
P4\_pass=1
$$

and:

$$
\Delta Acc_{\text{macro}}\geq -0.01.
$$

If higher capacity improves task but breaks P4, route should be:

```text
R5-CapacitySystemTradeoff
```

### 可视化

```text
p4_depth_width_rank_pareto.svg
p4_capacity_vs_step_memory.svg
p4_capacity_vs_margin.svg
p4_pareto_frontier.svg
```

---

## P5：Fixed patch/local source diagnostic for fused composition

### 目标

只在 fused compositional P4 pass 的 candidate 上测试 fixed patch/local source。若 patch source 显著提升，说明 vision route 应向 KANConv-like primitive 推进。

### Source candidates

```text
S3-EdgeAffineNorm
S4-FixedPatchPool
S6-FixedPatchPoolEdgeAffineNorm
S7-FixedLocalContrast
```

### 必须记录

```text
source_id
fixed_preprocessing_used
trainable_preprocessing_used
dataset
seed
conditioning_number
dominant_basis_fraction
dead_basis_fraction
patch_locality_score
train_acc
test_acc
delta_vs_mlp_match
CE_p99
margin_p10
step_ratio
memory_ratio
P4_pass
P5_near_pass
```

### 判断标准

Patch source evidence：

$$
Acc_{\text{PatchSource}}\geq Acc_{\text{FlatSource}}+0.03.
$$

All official FC candidates must satisfy:

```text
trainable_preprocessing_used = 0
```

### 可视化

```text
p5_source_accuracy_bar.svg
p5_source_condition_vs_accuracy.svg
p5_patch_locality_vs_gap.svg
p5_source_task_system_pareto.svg
```

---

## P6：Margin / scale confirmation for best fused compositional candidate

### 目标

v9.2.1 显示 lr0.0005 显著降低 CE tail，但不足以修复 K0。P6 只在 best fused compositional candidate 上做小型 scale sanity，确认 margin/logit pathology 是否仍存在。

### Allowed matrix

```text
lr = 0.0005, 0.001
correction_init_scale = 0.0, 0.01
output_scale = fanin, current
warmup = none, 5% steps
```

### 必须记录

```text
candidate
lr
correction_init_scale
output_scale
warmup
dataset
seed
early_loss_slope
early_margin_slope
train_acc_final
test_acc_final
train_loss_final
test_loss_final
CE_p99
logit_norm_mean
margin_p10
bad_update_rate
step_ratio
memory_ratio
```

### 判断标准

Scale improvement：

$$
CE_{p99,\text{new}}<0.5CE_{p99,\text{base}}.
$$

Task improvement：

$$
Acc_{\text{new}}\geq Acc_{\text{base}}+0.03.
$$

If CE tail improves but task gap remains:

```text
R4-MarginScaleNecessaryButInsufficient
```

### 可视化

```text
p6_scale_heatmap.svg
p6_ce_tail_by_scale.svg
p6_margin_slope.svg
p6_update_trace.svg
```

---

## P7：Functional update open decision

### 目标

决定 functional update 是否可以打开，不实际追求 external fair。Functional 只能在 P5 near-pass 后打开。

### 必须记录

```text
candidate
p5_near_pass
p5_pass
functional_open_allowed
reason
```

### Functional open criteria

```text
P4_pass = 1
P5_near_pass = 1
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
```

如果打开，则下一阶段才运行：

```text
F0-AdamW-only
F1-CorrectionChannelOnlyFunctional
F2-HighCurvatureCorrectionFunctional
F3-NoOpMatchedOverhead
F4-RandomDirection
```

### 可视化

```text
p7_functional_open_decision.svg
```

---

## 6. Route decision

### Route cases

```text
R1-FusedCompositionalFullEdgeP5Repaired:
  fused compositional FullEdge reaches P5 near-pass/pass and keeps P4 gate.

R2-CompositionalKernelRequired:
  synthetic interaction confirms depth need, but no fused composition candidate passes P4.

R3-CompositionalTrainabilityFail:
  fused composition passes P4 and synthetic interaction, but P5 still fails.

R4-MarginScaleNecessaryButInsufficient:
  CE tail/logit scale improves but task gap remains.

R5-CapacitySystemTradeoff:
  more capacity improves task but breaks P4.

R6-PatchLocalRequired:
  fixed patch/local source significantly improves task; move to KANConv-like primitive.

R7-CompositionSignalNotTaskRelevant:
  synthetic interaction succeeds, but no vision task gain over K0.

R8-ContractFail:
  candidate requires external residual, MLP path, trainable preprocessing, teacher, loss modification, fake/proxy, or loss.backward.

R9-ProfilerIncomplete:
  P4 fail source cannot be attributed due to unknown time/memory fraction.
```

### route_decision.json 必须记录

```text
route
best_candidate
best_kernel_path
best_depth
best_hidden_dim
best_rank
best_basis_channels
full_edge_equivalence_pass
no_external_residual_pass
p4_kernel_native_pass
synthetic_interaction_retained
p5_near_pass
p5_pass
patch_source_evidence
functional_open_allowed
primary_blocker
next_required_implementation
success_v922_kernel_closure
success_v922_trainability_repair
success_v922_functional_opened
success_v922_external_fair_opened
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_equivalence_audit_v922_kernel.csv
p0_route_recap.csv
p1_compositional_p4_failure_attribution.csv
p2_fused_compositional_kernel_correctness.csv
p2_fused_compositional_kernel_p4.csv
p2_synthetic_interaction_retention.csv
p3_compositional_adamw_trainability.csv
p3_compositional_trainability_trace.csv
p4_depth_width_rank_pareto.csv
p5_fixed_patch_source_diagnostic.csv
p6_margin_scale_confirmation.csv
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
F4_grad_fail
F5_p4_kernel_native_fail
F6_synthetic_interaction_lost
F7_p5_trainability_fail
F8_capacity_system_tradeoff
F9_patch_source_no_gain
F10_margin_scale_no_gain
F11_functional_not_opened
F12_profiler_incomplete
F13_fake_or_proxy_violation
F14_artifact_missing
```

---

## 8. 第一轮执行顺序

本补充计划必须按以下顺序执行：

```text
Step 1:
  P0 复现当前 v9.2.2 route 和 D2 synthetic interaction / generic D2 P4 fail。

Step 2:
  P1 做 compositional P4 failure attribution。
  如果 unknown_time_fraction > 0.10，先修 profiler。

Step 3:
  P2 实现 fused compositional kernel prototypes。
  同时验证 grad correctness、P4 gate、synthetic interaction retention。

Step 4:
  只有 P2 有 survivor，才打开 P3 AdamW-only trainability。

Step 5:
  P4 做 depth/width/rank Pareto。
  不允许用 test set 选 candidate。

Step 6:
  P5 只在 fused P4-pass candidate 上做 fixed patch/local source diagnostic。

Step 7:
  P6 只在 best fused candidate 上做小型 margin/scale confirmation。

Step 8:
  P7 判断 functional 是否允许打开。
  不在本轮强行跑 functional 或 external fair。
```

---

## 9. 停止条件

### 成功停止

Minimum success:

```text
at least one fused compositional FullEdge candidate:
  GradPass = 1
  P4_kernel_native_pass = 1
  synthetic_pairwise_R2 >= 0.95
  FullEdgeEquivalencePass = 1
  NoExternalResidualPass = 1
```

Trainability success:

```text
Minimum success
+
P5 near-pass
```

Strong v9.2.2 success:

```text
Trainability success
+
macro mean >= MLP-match
+
functional_open_allowed = 1
```

### 失败停止

```text
1. P0 cannot reproduce D2 synthetic interaction success；
2. P1 unknown_time_fraction > 0.10 and profiler cannot be fixed；
3. all fused compositional kernels fail grad correctness；
4. all fused compositional kernels fail P4；
5. fused kernels pass P4 but lose synthetic interaction；
6. fused kernels pass P4 and synthetic interaction but give no task gain over K0；
7. higher capacity improves task but breaks P4；
8. patch/local fixed source is the only task-improving route；
9. any contract/fake/proxy/offload/loss violation occurs。
```

---

## 10. 最终解释规则

### Case A：fused composition repairs P4 and P5

可以声明：

```text
v9.2.2 repaired FullEdge trainability through fused compositional PureKAN.
```

但仍不能声明 functional advantage 或 external fair success，除非后续 functional/external stages 打开并通过。

### Case B：fused composition repairs P4 but not P5

必须声明：

```text
Compositional kernel-native implementation is feasible, but current FullEdge composition still lacks task trainability.
```

### Case C：fused composition cannot pass P4

必须声明：

```text
Depth/compositionality is necessary but current FC FullEdge compositional kernel is not yet system-feasible.
```

### Case D：patch source is required

必须声明：

```text
Vision route should shift from flattened FC FullEdge to KANConv-like patch-local edge function.
```

### Case E：composition has no task gain

必须声明：

```text
Synthetic interaction capacity does not transfer to current vision task setup; revisit source/basis/task family boundary.
```

---

## 11. 最终建议

v9.2.2 的下一步不是再换 basis，也不是再调 optimizer，而是：

$$
\boxed{
\text{实现 fused compositional FullEdge kernel，验证 depth2 PureKAN 是否能在 P4 gate 内保留 interaction capacity，再重新打开 P5。}
}
$$

当前最重要的问题是：

```text
1. D2 为什么 synthetic interaction 成功但 vision P4 fail？
2. generic composition 的主要系统开销来自哪里？
3. 能否用 recompute / fused basis projection / streaming grad 让 D2 过 P4？
4. 过 P4 的 D2 是否还保留 pairwise interaction R2 >= 0.95？
5. 过 P4 的 D2 是否能把 P5 gap 从 K0 的 -0.028 继续推到 -0.01 内？
```

只有这些问题回答完，v9.2 才应该重新进入 functional update 与 external fair validation。
