# DG-KAN v9.2.4 GEMM-Native / Persistent-Kernel Compositional FullEdge Closure 完整实验计划

> 本计划基于 v9.2.3 最新 terminal route 制定。  
> v9.2.3 已经证明：D2 compositional FullEdge 的数学机制、梯度正确性、synthetic interaction retention 都成立；memory 和 step 已经进入或接近 gate；但 backward 与 forward 仍未进入 MLP-comparable envelope。  
> v9.2.4 不再继续在原有 Triton 小 kernel 上打补丁，也不直接打开 P5/P6/P7。  
> v9.2.4 的目标是重新设计 compositional FullEdge 的计算组织：要么把它降解成 **GEMM-native graph-free FullEdge lowering**，要么实现真正的 **persistent layer-level backward kernel**。只有当 P4 完整闭合后，才允许重新打开 AdamW-only trainability。

---

## 0. v9.2.3 最新结果的独立判断

### 0.1 目标是否达成

没有达成。

v9.2.3 最终停在：

```text
route = R4-BackwardKernelBlocker
best_candidate = D2-FusedCompositional-T2
best_combined_candidate = C3-M7+B6+F3-no-input-dx-forward-yonly
success_v923_p4_kernel_closure = false
success_v923_trainability_reentry = false
success_v923_functional_opened = false
```

最新最佳 combined candidate 的核心指标是：

```text
forward_ratio = 1.421522
backward_ratio = 2.048100
step_ratio = 1.463772
memory_ratio = 1.046065
GradRelErrMax = 5.6768145e-05
synthetic_pairwise_R2 = 0.9911209941
```

P4 gate 要求是：

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

因此：

```text
memory pass = true
step pass = true
forward pass = false
backward pass = false
overall P4 pass = false
```

### 0.2 v9.2.3 的真实进展

v9.2.3 不是无效失败。它已经验证了四件重要事情。

第一，D2 compositional FullEdge 的表达机制保留。Synthetic pairwise R2 保持：

$$
R^2_{\text{pairwise}}=0.9911209941.
$$

第二，manual / graph-free gradient correctness 成立。当前最佳 combined candidate 的：

$$
GradRelErrMax=5.6768145\times10^{-5},
$$

满足：

$$
GradRelErrMax\leq10^{-4}.
$$

第三，memory live-set 被推进进 gate。M7 no-input-dx / compact live-set accounting 将 memory ratio 从：

$$
1.213611
$$

降到：

$$
1.046065.
$$

第四，step ratio 已经过线：

$$
1.463772\leq1.50.
$$

因此，当前已经不是“basis 不成立”、不是“composition 不成立”、不是“gradient 错”、也不是“memory 完全不可控”。当前 blocker 已经非常集中：

$$
\boxed{
\text{D2 compositional FullEdge 的 backward kernel 与 forward kernel 组织仍不够 MLP-comparable。}
}
$$

### 0.3 v9.2.3 暴露的核心问题

v9.2.3 的 Triton 结果非常重要。

B8 split-kernel Triton backward：

```text
backward_ratio = 15.680900
step_ratio = 4.237832
GradRelErrMax = 1.0023164e-04
```

B9 monolithic Triton backward：

```text
backward_ratio = 13.413305
step_ratio = 4.007630
GradRelErrMax = 6.8147972e-05
```

B9 比 B8 更正确、更合理，但仍远慢于 C3 no-dx/y-only 的：

```text
backward_ratio = 2.048100
step_ratio = 1.463772
```

这说明当前真正的问题不是“还没把 dM 和 dx 放到同一个 kernel”。B9 已经做了单层 monolithic，但仍失败。更深层问题是：

$$
\boxed{
\text{当前 Triton kernel 的计算组织不是 GEMM-grade，也不是 persistent-grade。}
}
$$

它仍然在 GEMM、Triton reductions、层间传播之间来回切换，不能达到 cuBLAS/compiled torch 的效率。继续在当前 B8/B9 小 kernel 形式上微调 block size，大概率不会根本改变结论。

### 0.4 当前不应该做什么

v9.2.4 不应该做：

```text
1. 继续直接打开 P5 AdamW-only trainability；
2. 继续打开 functional update；
3. 继续 external fair validation；
4. 继续在 B9 上只调 Triton BLOCK size；
5. 继续单独替换 forward residual kernel；
6. 继续做小 lr / optimizer sanity；
7. 回退到 transitional linear-SiLU route；
8. 引入外部 residual / MLP shortcut / trainable Conv preprocessing。
```

这些都会绕过当前最明确的 blocker。

### 0.5 v9.2.4 的正确定位

v9.2.4 的正确定位是：

$$
\boxed{
\text{从“替换局部 kernel”转向“重写 D2 FullEdge 的计算 lowering”。}
}
$$

也就是说，v9.2.4 要回答：

```text
Q1:
  D2 FullEdge 的前向和反传是否可以被重写成少量 GEMM + 少量 fused pointwise kernel？

Q2:
  如果不能，是否可以用真正 persistent layer-level CUDA/Triton kernel 一次性完成 dr / dM / dx / 部分 param grad accumulation？

Q3:
  哪条路径能在保留 synthetic interaction 与 GradPass 的前提下，把 forward <=1.25、backward <=1.50、memory <=1.05？
```

---

## 1. v9.2.4 整体目标

v9.2.4 的整体目标是：

$$
\boxed{
\text{关闭 D2 compositional FullEdge 的 P4 kernel-native gate，并重新打开 P5 AdamW-only trainability。}
}
$$

这个目标分两级。

### 1.1 P4 kernel closure minimum success

得到至少一个 D2 compositional FullEdge candidate，满足：

```text
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
GradPass = 1
SyntheticInteractionRetentionPass = 1
P4KernelNativePass = 1
```

其中：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999,
$$

$$
R^2_{\text{pairwise}}\geq0.95,
$$

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

### 1.2 P5 trainability re-entry success

只有 P4 closure 成立后，才打开 P5。P5 near-pass 定义：

$$
Acc_{\text{KAN-AdamW}}\geq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows，并且：

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

P5 pass 定义：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

v9.2.4 不要求 functional success。Functional update 只有 P5 near-pass 后才允许打开。

---

## 2. 继续坚持的硬约束

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

P5 之前只允许 AdamW-equivalent update：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{\text{AdamW-equivalent}}.
$$

P5 前不允许 functional update：

$$
\Delta\theta_{\text{functional}}.
$$

### 2.2 Pure FullEdge contract

每一层必须满足：

$$
h^{\ell+1}_j=\sum_i\phi^\ell_{ij}(h^\ell_i).
$$

每条 edge function 必须满足：

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

不允许：

```text
external residual shortcut
ordinary Linear skip outside edge function
ordinary MLP hidden path
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

---

## 3. 当前 D2 FullEdge 计算图分析

v9.2.4 必须先把 D2 active T2 compositional FullEdge 计算图写清楚，避免继续局部 patch。

一个简化的 D2 layer 可以写成：

$$
h^{\ell+1}=h^\ell W_0^\ell + R^\ell (V^\ell)^T,
$$

其中 nonlinear edge-correction channel 为：

$$
R^\ell_{b,r}=\sum_i U^\ell_{i,r}\psi^\ell_r(\tilde{h}^\ell_{b,i}),
$$

对当前 active T2 basis：

$$
\psi_r(x)=a_r T_2(x).
$$

因此一层 forward 可以拆成：

$$
Y_{\text{id}} = XW_0,
$$

$$
S_{b,i,r}=U_{i,r}\psi_r(\tilde{x}_{b,i}),
$$

$$
R_{b,r}=\sum_i S_{b,i,r},
$$

$$
Y_{\text{corr}}=RV^T,
$$

$$
Y=Y_{\text{id}}+Y_{\text{corr}}.
$$

反传给定：

$$
G=\frac{\partial L}{\partial Y}.
$$

则：

$$
\frac{\partial L}{\partial W_0}=G^TX,
$$

$$
\frac{\partial L}{\partial V}=G^TR,
$$

$$
dR=GV,
$$

$$
\frac{\partial L}{\partial U_{i,r}}=\sum_b dR_{b,r}\psi_r(\tilde{x}_{b,i}),
$$

$$
\frac{\partial L}{\partial a_r}=\sum_{b,i} dR_{b,r}U_{i,r}T_2(\tilde{x}_{b,i}),
$$

$$
dX_{\text{corr},b,i}=\sum_r dR_{b,r}U_{i,r}a_rT_2'(\tilde{x}_{b,i})\frac{\partial\tilde{x}_{b,i}}{\partial x_{b,i}}.
$$

当前 no-input-dx-layer1 已移除第一层 input dx，但 layer2/layer3 仍需要 $dH$ 传播。

v9.2.3 的失败说明当前实现没有把这些操作组织成 GEMM-grade 或 persistent-grade。

---

## 4. 核心假设

### H1：当前 backward ratio 2.048 不是数学不可避免，而是 lowering 不佳

H1 假设：当前最佳 C3 的 backward ratio 高，是因为 compiled y-only / no-dx path 仍然有过多分散操作，而不是 D2 FullEdge 必然需要 2x MLP backward。

H1 成立标准：

通过 GEMM-native lowering 或 persistent kernel 后：

$$
T_{\text{backward,new}}\leq1.50T_{\text{backward,MLP-match}}.
$$

若所有 lowering 都无法低于 1.50，但 synthetic interaction 与 GradPass 保持，则 route 应写为：

```text
R4-BackwardKernelLoweringLimit
```

### H2：GEMM-native lowering 比当前 custom Triton scalar kernels 更可能成功

B8/B9 Triton backward runtime 远慢于 compiled C3，说明当前 Triton kernel 不是有效方向。H2 假设：当前 D2 由于包含大量 matrix contraction，应优先用 cuBLAS/GEMM-native lowering，而不是手写小 reduction Triton kernel。

H2 成立标准：

GEMM-native candidate 相比 C3：

$$
T_{\text{backward,GEMM}}\leq0.75T_{\text{backward,C3}},
$$

并且：

$$
T_{\text{backward,GEMM}}\leq1.50T_{\text{backward,MLP-match}}.
$$

### H3：Persistent kernel 只有在融合完整 layer backward 时才可能有效

H3 假设：单独 dM/dx Triton kernel 不够。Persistent kernel 必须把以下至少三项合在同一 kernel family：

```text
dr generation
basis derivative
dU / da reduction
dx correction
partial dV or dW accumulation
```

H3 成立标准：

Persistent candidate 相比 B9：

$$
T_{\text{backward,persistent}}\leq0.25T_{\text{backward,B9}},
$$

并且：

$$
T_{\text{backward,persistent}}\leq1.50T_{\text{backward,MLP-match}}.
$$

### H4：Forward ratio 1.42 需要 forward lowering，但不是主 blocker

当前 forward 也 fail：

$$
1.421522>1.25.
$$

但 gap 只有约：

$$
\frac{1.421522}{1.25}-1\approx0.137218.
$$

H4 假设：只要把 source norm + basis eval + R projection 合成少量 GEMM/fused pointwise，forward 可以闭合。

H4 成立标准：

$$
T_{\text{forward,new}}\leq1.25T_{\text{forward,MLP-match}}.
$$

### H5：Memory ratio 1.046 已接近临界，v9.2.4 不能牺牲 memory

当前 memory 已经过 gate：

$$
1.046065\leq1.05.
$$

任何 forward/backward repair 不能重新把 memory 推出 gate。

H5 成立标准：

$$
M_{\text{new}}\leq1.05M_{\text{MLP-match}}.
$$

如果某个 candidate 关闭 backward 但 memory 变成：

$$
M_{\text{new}}>1.05M_{\text{MLP-match}},
$$

则 route 为：

```text
R5-TimeMemoryTradeoffNotClosed
```

### H6：P4 修复不能破坏 interaction retention

任何 system repair 必须保持：

$$
R^2_{\text{pairwise}}\geq0.95.
$$

如果 P4 过了但 synthetic pairwise R2 下降，说明 system repair 破坏了 D2 composition，不允许进入 P5。

---

## 5. Candidate 设计

### 5.1 Baseline candidates

```text
B0-MLP-match:
  同参数 MLP-match，CE-only，AdamW。

K0-S3-B2-P4-F0:
  单层 active-k P4-pass / P5-fail candidate。

D2-C3-current:
  当前最佳 C3-M7+B6+F3 no-input-dx-forward-yonly candidate。

D2-B9-monolithic:
  最新 monolithic Triton backward reference。
```

### 5.2 GEMM-native lowering candidates

```text
G1-GEMMNative-VJP:
  用 GEMM 组织 dR = G V、dV = G^T R、dW0 = G^T X，
  basis derivative 保持 compiled pointwise。

G2-GEMMNative-FusedPointwise:
  G1 + fuse source norm / T2 / T2' / U scaling pointwise。

G3-GEMMNative-TwoPassReduction:
  dU / da 使用两次 GEMM/reduction，不用 Triton scalar reduction。

G4-GEMMNative-BatchedLayerBackward:
  layer2/layer3 backward 用相同 batched contraction 模板，减少 compile graph fragmentation。

G5-GEMMNative-NoMaterializedDr:
  不显式保存 dR，流式或重用 GEMM output buffer 计算 dU/da/dx。

G6-GEMMNative-CUDAGraph:
  固定 shape 下捕获 forward/backward/update CUDA graph，减少 launch overhead。
```

### 5.3 Persistent / custom kernel candidates

```text
P1-PersistentLayerBackward-Minimal:
  persistent kernel 计算 dU/da/dx correction，不计算 dW0/dV。

P2-PersistentLayerBackward-WithDVec:
  persistent kernel 同时计算 dU/da/dx correction 和部分 dV/dW0 tile accumulation。

P3-PersistentLayerBackward-OneCTAperR:
  每个 rank channel 一个 CTA，优化 rank=4 small-R 场景。

P4-PersistentLayerBackward-OneCTAperOutputTile:
  每个 output tile 一个 CTA，优化 hidden=512。

P5-PersistentLayerBackward-WarpReduction:
  使用 warp-level reduction 替代 block-level atomic/reduction。

P6-CUDAExtensionLayerBackward:
  若 Triton 仍无法达到效率，用 CUDA C++ extension 实现 layer backward。
```

### 5.4 Forward lowering candidates

```text
F1-GEMMNativeForward:
  source norm / T2 pointwise + GEMM projection，尽量少 materialize basis。

F2-FusedSourceT2Forward:
  fuse EdgeAffineNorm + T2 + U scaling。

F3-FusedRProjectionForward:
  fuse R generation and projection preparation。

F4-CUDAGraphForward:
  固定 shape 捕获 forward CUDA graph。

F5-PersistentForward:
  persistent kernel 生成 R 并写 output projection buffer。
```

### 5.5 Combined candidates

```text
C1-G2+F2:
  GEMMNative-FusedPointwise backward + fused source/T2 forward。

C2-G3+F2:
  GEMM two-pass reduction + fused forward。

C3-G5+F2:
  no-materialized-dR + fused source/T2 forward。

C4-G6+F4:
  CUDA graph full compiled path。

C5-P1+F2:
  persistent minimal backward + fused forward。

C6-P3+F5:
  rank-channel persistent backward + persistent forward。

C7-P6+F2:
  CUDA extension backward + fused forward。
```

### 5.6 Width / rank / depth fallback candidates

Only if no C candidate closes P4:

```text
S1-depth2-hidden384-rank4
S2-depth2-hidden256-rank4
S3-depth2-hidden512-rank2
S4-depth2-hidden384-rank2
S5-depth2-hidden256-rank2
```

These are not hyperparameter sweep for task; they are system Pareto fallback. They must still satisfy:

$$
R^2_{\text{pairwise}}\geq0.95.
$$

---

## 6. 实验阶段

## P0：Latest route reproduction and profiler completeness

### 目标

复现 v9.2.3 最新 C3/B9 结果，确认当前 measurement 与 profiler 足够可靠。

### 必须记录

```text
candidate_id
implementation_id
forward_ratio
backward_ratio
step_ratio
memory_ratio
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
kernel_count_total
small_kernel_count
unknown_time_fraction
phase_time_sum_fraction
phase_memory_sum_fraction
```

### 判断标准

重复稳定：

$$
|r_{\text{backward}}-2.048100|\leq0.15.
$$

Profiler completeness：

$$
unknown\_time\_fraction\leq0.10,
$$

$$
phase\_time\_sum\_fraction\geq0.90.
$$

### 可视化

```text
p0_latest_route_reproduction.svg
p0_phase_completeness_dashboard.svg
p0_repeat_vs_previous_ratio.svg
```

---

## P1：Algebraic lowering audit

### 目标

把 D2 forward/backward 拆成数学 primitive，判断每个 primitive 应该走 GEMM、pointwise、reduction 还是 custom kernel。

### 必须记录

```text
primitive_id
primitive_formula
input_shape
output_shape
current_implementation
recommended_lowering
flops
bytes_read
bytes_written
arithmetic_intensity
materializes_tensor
current_time_ms
candidate_lowering
```

Primitive 至少包括：

```text
source_norm
T2_eval
R_generation
identity_projection
correction_projection
dV
dW0
dR
dU
da
dx_correction
dh_propagation
```

### 判断标准

每个 primitive 必须有 lowering decision：

```text
GEMM
fused_pointwise
streaming_reduction
persistent_kernel
not_needed
```

如果超过 20% primitive 仍为 unknown，不进入 P2。

### 可视化

```text
p1_primitive_lowering_table.md
p1_roofline_proxy.svg
p1_time_by_primitive.svg
p1_materialized_tensor_graph.svg
```

---

## P2：GEMM-native backward candidates

### 目标

优先验证 H2：GEMM-native lowering 是否能比 Triton scalar kernels 更有效。

### 必跑 candidates

```text
G1-G6
```

### 必须记录

```text
candidate_id
uses_gemm_dR
uses_gemm_dV
uses_gemm_dW0
uses_gemm_dU
uses_gemm_da
materializes_dR
fuses_pointwise_derivative
uses_cuda_graph
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
forward_ratio
backward_ratio
step_ratio
memory_ratio
kernel_count_total
small_kernel_count
```

### 判断标准

GEMMBackwardPass：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999,
$$

$$
R^2_{\text{pairwise}}\geq0.95,
$$

$$
T_{\text{backward}}\leq1.50T_{\text{MLP-match}}.
$$

Improvement over current:

$$
T_{\text{backward,candidate}}\leq0.75T_{\text{backward,C3}}.
$$

### 可视化

```text
p2_gemm_backward_ratio_bar.svg
p2_gemm_kernel_count.svg
p2_gemm_memory_tradeoff.svg
p2_gemm_correctness_scatter.svg
```

---

## P3：Persistent / custom kernel candidates

### 目标

如果 GEMM-native 不足，验证真正 persistent layer backward，而不是 B8/B9 小 kernel。

### 必跑 candidates

```text
P1-P6
```

### 必须记录

```text
candidate_id
kernel_type
persistent_blocks
cta_tiling
warp_reduction_used
computes_dU
computes_da
computes_dx_correction
computes_partial_dV
computes_partial_dW0
atomic_ops_count
shared_memory_bytes
registers_per_thread
occupancy_estimate
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
forward_ratio
backward_ratio
step_ratio
memory_ratio
```

### 判断标准

PersistentBackwardPass：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999,
$$

$$
R^2_{\text{pairwise}}\geq0.95,
$$

$$
T_{\text{backward}}\leq1.50T_{\text{MLP-match}}.
$$

Strong persistent improvement over B9:

$$
T_{\text{backward,persistent}}\leq0.25T_{\text{backward,B9}}.
$$

### 可视化

```text
p3_persistent_backward_ratio.svg
p3_persistent_occupancy.svg
p3_persistent_correctness.svg
p3_persistent_vs_gemm_pareto.svg
```

---

## P4：Forward closure

### 目标

将 forward ratio 从约 `1.421522` 降到 `<=1.25`。

### 必跑 candidates

```text
F1-F5
```

### 必须记录

```text
candidate_id
source_norm_fused
T2_eval_fused
R_projection_fused
uses_cuda_graph
uses_persistent_forward
forward_ratio
step_ratio
memory_ratio
kernel_count_forward
small_kernel_count_forward
output_abs_diff_max
```

### 判断标准

ForwardPass：

$$
T_{\text{forward}}\leq1.25T_{\text{MLP-match}}.
$$

Correctness:

$$
OutputAbsDiffMax\leq10^{-5}.
$$

No memory regression:

$$
M_{\text{new}}\leq1.05M_{\text{MLP-match}}.
$$

### 可视化

```text
p4_forward_candidate_bar.svg
p4_forward_kernel_count.svg
p4_forward_correctness.svg
```

---

## P5：Combined P4 closure

### 目标

组合 P2/P3/P4 survivors，形成真正 P4-pass D2 candidate。

### 必跑 candidates

```text
C1-C7
```

### 必须记录

```text
candidate_id
backward_component
forward_component
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
forward_ratio
backward_ratio
step_ratio
memory_ratio
kernel_count_total
small_kernel_count_total
P4_kernel_native_pass
```

### 判断标准

P4 pass：

$$
T_{\text{forward}}\leq1.25T_{\text{MLP-match}},
$$

$$
T_{\text{backward}}\leq1.50T_{\text{MLP-match}},
$$

$$
T_{\text{step}}\leq1.50T_{\text{MLP-match}},
$$

$$
M_{\text{peak}}\leq1.05M_{\text{MLP-match}}.
$$

Interaction retention:

$$
R^2_{\text{pairwise}}\geq0.95.
$$

GradPass:

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

### 可视化

```text
p5_combined_p4_dashboard.svg
p5_combined_before_after.svg
p5_combined_pareto_front.svg
```

---

## P6：System Pareto fallback

### 目标

如果 P5 没有 pass，测试是否存在 smaller D2 composition 能同时保留 interaction 且过 P4。

### 必跑 candidates

```text
S1-S5
```

### 必须记录

```text
candidate_id
hidden_dim
rank
params_ratio
synthetic_pairwise_R2
GradRelErrMax
forward_ratio
backward_ratio
step_ratio
memory_ratio
P4_kernel_native_pass
```

### 判断标准

FallbackPass：

$$
R^2_{\text{pairwise}}\geq0.95,
$$

$$
P4\_kernel\_native\_pass=1.
$$

如果 smaller candidate P4 pass 但 R2 fail，说明它通过系统 gate 的方式是破坏 composition，不允许进入 P7。

### 可视化

```text
p6_width_rank_pareto.svg
p6_interaction_vs_system.svg
```

---

## P7：AdamW-only trainability re-entry

### 目标

只有 P5 或 P6 出现 P4-pass candidate 后才打开。验证 D2 composition 是否修复 K0 trainability gap。

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
forward_ratio
backward_ratio
step_ratio
memory_ratio
```

### 判断标准

Gain over K0:

$$
Acc_{\text{D2}}\geq Acc_{\text{K0}}+0.05
$$

on macro mean，或：

$$
Acc_{\text{D2,KMNIST}}\geq Acc_{\text{K0,KMNIST}}+0.05.
$$

P5 near-pass:

$$
Acc_{\text{D2}}\geq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows，并且：

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

P5 pass:

$$
\Delta Acc_{\text{macro}}\geq0.
$$

### 可视化

```text
p7_trainability_gap_bar.svg
p7_loss_curve.svg
p7_ce_tail.svg
p7_margin_distribution.svg
p7_task_system_pareto.svg
```

---

## P8：Functional open decision

### 目标

只决定 functional update 是否允许打开，不追求 functional success。

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

### 可视化

```text
p8_functional_open_decision.svg
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_equivalence_audit_v924.csv
p0_latest_route_reproduction.csv
p1_algebraic_lowering_audit.csv
p2_gemm_native_backward.csv
p3_persistent_custom_backward.csv
p4_forward_closure.csv
p5_combined_p4_closure.csv
p6_system_pareto_fallback.csv
p7_adamw_trainability_reentry.csv
p8_functional_open_decision.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_full_edge_equivalence_fail
F3_measurement_instability
F4_lowering_audit_incomplete
F5_gemm_native_backward_fail
F6_persistent_backward_fail
F7_forward_closure_fail
F8_combined_p4_fail
F9_interaction_lost
F10_grad_fail
F11_system_pareto_fail
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
  D2 compositional FullEdge closes P4 and opens AdamW trainability.

R2-GEMMNativeClosure:
  GEMM-native lowering closes P4.

R3-PersistentKernelClosure:
  persistent/custom kernel closes P4.

R4-BackwardLoweringLimit:
  all lowering attempts keep backward > 1.50.

R5-TimeMemoryTradeoffNotClosed:
  time improves but memory leaves gate, or memory improves but time fails.

R6-ForwardKernelBlocker:
  backward closes but forward remains > 1.25.

R7-InteractionLost:
  P4 pass only by destroying pairwise R2.

R8-P4ClosedButTrainabilityFail:
  P4 closes, but AdamW trainability fails.

R9-KANConvPivotRecommended:
  FC D2 cannot close P4; patch-local or KANConv primitive required.

R10-ContractFail:
  PureKAN/no-teacher/no-loss/no-fake contract violated.
```

### route_decision.json 必须记录

```text
route
best_candidate
best_lowering_path
best_backward_path
best_forward_path
best_hidden_dim
best_rank
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
success_v924_p4_kernel_closure
success_v924_trainability_reentry
success_v924_functional_opened
```

---

## 9. 第一轮执行顺序

```text
Step 1:
  P0 复现 latest C3/B9，确认 measurement stable。

Step 2:
  P1 做 algebraic lowering audit。
  不允许跳过这一步直接继续写 Triton kernel。

Step 3:
  P2 优先实现 GEMM-native backward candidates。
  先验证 H2：是否能用 GEMM/reduction lowering 击败 B9 和 C3。

Step 4:
  P3 只在 P2 不够时实现 persistent/custom kernel candidates。
  persistent 必须合并 dr / dU / da / dx correction，而不是只做 dM/dx 小 kernel。

Step 5:
  P4 forward closure。

Step 6:
  P5 combined P4 closure。
  只有 P4 pass 后，才进入 P7。

Step 7:
  如果 P5 没有 pass，再执行 P6 system Pareto fallback。

Step 8:
  P4-pass candidate 出现后，执行 P7 AdamW-only trainability。

Step 9:
  只有 P7 near-pass 后，P8 才允许 functional update 打开。
```

---

## 10. 停止条件

### 成功停止

Minimum success:

```text
P4_kernel_native_pass = 1
GradPass = 1
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

Strong v9.2.4 success:

```text
Trainability success
+
macro mean >= MLP-match
+
functional_open_allowed = 1
```

### 失败停止

```text
1. latest route reproduction unstable；
2. algebraic lowering audit incomplete；
3. GEMM-native and persistent paths both cannot reduce backward below 1.50；
4. forward remains above 1.25 after closure attempts；
5. time improves but memory exits gate；
6. P4 pass destroys synthetic interaction；
7. P4 pass but P5 trainability still fails；
8. any contract/fake/proxy/offload/loss violation occurs。
```

---

## 11. 最终解释规则

### Case A：GEMM-native path closes P4

可以声明：

```text
D2 compositional FullEdge P4 was closed by GEMM-native lowering.
```

但仍不能声明 functional advantage 或 external fair success，除非 P7/P8 后续通过。

### Case B：Persistent kernel closes P4

可以声明：

```text
D2 compositional FullEdge requires persistent/custom kernel, and P4 is closed under that implementation.
```

### Case C：Backward cannot close

必须声明：

```text
D2 composition has correct gradients and interaction retention, but current FC FullEdge backward cannot be made MLP-comparable under tested lowering paths.
```

### Case D：P4 closes but trainability fails

必须声明：

```text
System blocker is solved; next blocker is AdamW-only trainability.
```

### Case E：FC route should pivot

若所有 P4 closure attempts fail, but patch/KANConv direction is suggested, 必须声明：

```text
Flattened FC compositional FullEdge is not the right scaling path for vision; next implementation should pivot to KANConv-like patch-local edge functions.
```

---

## 12. 最终建议

v9.2.4 的一句话策略是：

$$
\boxed{
\text{停止局部 Triton 小 kernel patch，重新做 D2 FullEdge 的 GEMM-native 或 persistent lowering。}
}
$$

当前最重要的不是 task accuracy，而是关闭这三个 system gaps：

```text
forward ratio  1.421522 -> <= 1.25
backward ratio 2.048100 -> <= 1.50
memory ratio   1.046065 -> keep <= 1.05
```

只有 P4 真的闭合，才可以重新讨论 AdamW-only trainability、functional update 和 external fair validation。
