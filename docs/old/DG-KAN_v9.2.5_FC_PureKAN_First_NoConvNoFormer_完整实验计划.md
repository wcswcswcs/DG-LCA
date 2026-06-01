# DG-KAN v9.2.5 FC-PureKAN First：暂缓 PureKANConv / PureKANFormer 的收紧路线完整实验计划

> 本计划是对上一版 v9.2.5 的路线收紧。  
> 本版明确删除 **PureKANConv pivot** 和 **PureKANFormer / KAN-FFN 扩展** 的主动实验阶段。  
> 原因很简单：**当前 Clean FullEdge / FC-PureKAN 还没有在同等或更低 time / memory 预算下比肩 MLP，更没有形成显著 Beyond-MLP official success。**  
> 在 FC-PureKAN 还没有完成之前，提前研究 PureKANConv 或 PureKANFormer 会把主线带偏：我们会在没有证明基础 PureKAN primitive 的情况下，开始研究更大架构。这不符合当前目标。
>
> v9.2.5 的新目标是：
>
> $$
> \boxed{
> \text{先让 FC-PureKAN 在 strict PureKAN / graph-free / kernel-native / AdamW-only 下比肩 MLP。}
> }
> $$
>
> 只有当 FC-PureKAN 至少达到 P5 near-pass，并且最好达到 P5 pass 后，才考虑 PureKANConv 与 PureKANFormer。它们在本文件中只作为 **deferred future route**，不作为本轮实验内容。

---

## 0. 为什么要收紧路线

### 0.1 当前 PureKAN 还没有完全比过 MLP

必须先把三个层级区分清楚。

第一，dense manual KAN 曾经给出明确的 accuracy signal。v7.0 的 targeted dense D3 candidate 打开过 basic Beyond-MLP task accuracy gate，平均 val gap 约为 `+0.0206`，且 ECE / NLL 优于 MLP。但 dense D3 不是 kernel-native official candidate，full-step efficiency 和 FastOpt 后的 efficiency 都没有通过。因此它只能说明：

$$
\boxed{
\text{KAN-family 有真实表达力信号。}
}
$$

不能说明：

$$
\boxed{
\text{Clean FullEdge PureKAN 已经在同等系统预算下赢 MLP。}
}
$$

第二，v8.7 的 formal selected route 很强，但它是 transitional route。它的实质是：

```text
packed / cached linear-SiLU stack
+
poly2_silu KAN-style head
+
adaptive functional update
```

它在 external fair setting 下取得了很好的 task / timing / geometry 结果，但 v9.0 已经把它正确判定为 `CleanTransitionalOnlySuccess`，不是终极 FullEdge PureKAN success。因此它不能作为 PureKANConv / PureKANFormer 的基础胜利。

第三，v9.x 的 Clean FullEdge / FC-PureKAN 路线仍未成功。v9.1 发现 raw-input basis conditioning 失败；v9.2 找到 `S3-EdgeAffineNorm + B2-Chebyshev3Residual` 并修过单层 P4，但 P5 AdamW-only trainability 失败；v9.2.2 证明 depth2 compositionality 对 synthetic interaction 是必要的；v9.2.4 的 D2 compositional FullEdge 保留了 gradient correctness 与 synthetic interaction，但 P4 仍未关闭。

所以当前真实状态是：

$$
\boxed{
\text{PureKAN 仍处在“表达机制成立，但 FC kernel-native / trainability 未闭合”的阶段。}
}
$$

### 0.2 为什么现在不研究 PureKANConv / PureKANFormer

PureKANConv 与 PureKANFormer 是终极扩展方向，但不是当前阶段的实验对象。原因有三点。

第一，扩展架构会引入新的变量。PureKANConv 会引入 patch locality、channel sharing、kernel layout、im2col / direct conv lowering、spatial reuse 等新问题。PureKANFormer 会引入 token length、FFN expansion ratio、attention coupling、sequence memory、layernorm interaction 等新问题。如果在 FC-PureKAN 还没闭合时进入这些方向，我们无法判断失败来自基础 edge-function primitive，还是来自新架构自身。

第二，当前 blocker 已经足够明确。v9.2.4 的 P4 failure 不是因为不知道该做 Conv，而是因为当前 FC-D2 的 backward / forward lowering 仍未达到 MLP-comparable：

```text
forward_ratio  = 1.4392885671  > 1.25
backward_ratio = 2.0424724744  > 1.50
step_ratio     = 1.3739124629  <= 1.50
memory_ratio   = 1.0460654217  <= 1.05
synthetic_pairwise_R2 = 0.9911209941
grad_pass = 1
```

这说明表达机制、梯度、memory、step 都已经接近或成立，真正需要处理的是 FC-D2 的 lowering 与 P5 re-entry。此时跳到 PureKANConv 等于绕开当前明确问题。

第三，终极目标要求的是 Next-Gen Beyond-MLP，而不是“另起一个更复杂架构也许能过”。因此本轮必须坚持：

$$
\boxed{
\text{FC-PureKAN 先比肩 MLP，再讨论 PureKANConv / PureKANFormer。}
}
$$

### 0.3 本文件对上一版 v9.2.5 的修改

上一版 v9.2.5 写了“FC-D2 最终可行性判定，必要时 pivot PureKANConv”。这在逻辑上是一个可能的远期分支，但现在不应该作为 active plan。因此本版修改为：

```text
1. 删除 PureKANConv active experiments。
2. 删除 PureKANFormer / KAN-FFN active experiments。
3. 保留 FC-D2 final closure。
4. 如果 FC-D2 不闭合，不立即 pivot Conv，而是进入 FC-PureKAN primitive redesign。
5. 只有 FC-PureKAN P5 near-pass / pass 后，才把 Conv / Former 放回路线。
```

---

## 1. v9.2.5 总体目标

v9.2.5 的总体目标是：

$$
\boxed{
\text{在 FC-PureKAN 范围内完成 P4 kernel closure 或明确 FC-PureKAN redesign 方向，并重新打开 P5 AdamW-only trainability。}
}
$$

这个目标分成两个层级。

### 1.1 P4 system success

得到至少一个 FC-PureKAN candidate，满足：

```text
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
NoOrdinaryMLPPathPass = 1
NoTrainablePreprocessorPass = 1
GradPass = 1
SyntheticInteractionRetentionPass = 1
P4KernelNativePass = 1
```

其中 P4KernelNativePass 定义为：

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

同时必须保持 synthetic interaction：

$$
R^2_{\text{pairwise}}\geq0.95.
$$

以及 gradient correctness：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

### 1.2 P5 trainability re-entry success

只有 P4 通过后，才打开 P5 AdamW-only trainability。

P5 near-pass 定义为：

$$
Acc_{\text{KAN-AdamW}}\geq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows，并且：

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

P5 pass 定义为：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

v9.2.5 不追求 functional update success，也不追求 external fair success。Functional update 只有在 P5 near-pass 后才允许打开。

---

## 2. 本轮不可违反的硬约束

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

训练目标必须是：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

P5 前只允许 AdamW-equivalent update：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{\text{AdamW-equivalent}}.
$$

P5 前不允许 functional update：

$$
\Delta\theta_{\text{functional}}.
$$

也不允许把几何写进 loss：

$$
L=CE+\lambda L_{\text{geo}}.
$$

### 2.2 FC FullEdge PureKAN contract

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

允许 nonlinear edge-correction channels：

$$
\sum_{k\geq1}c_{ij,k}B_k(\tilde{x}).
$$

允许 edge coefficient tensor 的低秩分解：

$$
C_{i,j,k}=\sum_r U_{i,r}V_{j,r}A_{k,r}.
$$

但它必须等价于 edge coefficient tensor 的因式分解，而不能变成普通 hidden activation：

$$
r=\sigma(xU),\quad y=rV.
$$

### 2.3 禁止项

以下结构不能进入 official route：

```text
external residual shortcut
ordinary Linear skip outside edge function
ordinary MLP hidden path
trainable Conv / Linear preprocessing
KAN output + non-KAN residual branch
node activation MLP disguised as low-rank factorization
PureKANConv active experiment
PureKANFormer / KAN-FFN active experiment
```

### 2.4 Deferred architecture rule

本轮必须显式记录：

```text
PureKANConv_status = deferred_until_FC_PureKAN_P5_near_pass
PureKANFormer_status = deferred_until_FC_PureKAN_P5_near_pass
```

这两个方向不得产生 active measured candidate，不得参与 route selection，也不得作为 failure 或 success。

---

## 3. 当前 FC-D2 问题的定量分析

v9.2.4 的最佳 candidate 是 `G2-GEMMNative-FusedPointwise-compiled`，其核心指标：

```text
forward_ratio = 1.4392885671
backward_ratio = 2.0424724744
step_ratio = 1.3739124629
memory_ratio = 1.0460654217
synthetic_pairwise_R2 = 0.9911209941
grad_pass = 1
```

当前 gap 是：

$$
\frac{1.4392885671}{1.25}-1\approx0.151431,
$$

也就是 forward 还需要降低约 15%。

$$
\frac{2.0424724744}{1.50}-1\approx0.361648,
$$

也就是 backward 还需要降低约 36%。

memory 已经非常接近 gate：

$$
1.0460654217\leq1.05.
$$

step 已经过 gate：

$$
1.3739124629\leq1.50.
$$

因此，本轮的主要目标不是 memory，也不是 optimizer，而是：

$$
\boxed{
\text{forward / backward lowering closure, especially backward.}
}
$$

---

## 4. v9.2.5 核心假设

### H1：FC-D2 仍有最后一次系统闭合机会，但必须用 CUDA extension / persistent layer kernel，而不是继续小 Triton patch

v9.2.4 已经显示 B9 monolithic Triton backward 仍远慢，G2 GEMM-native compiled 仍约 `2.04x`。H1 假设：如果 FC-D2 还能闭合，必须通过真正的 CUDA extension / persistent layer-level backward 实现，而不是在当前小 kernel 和 compiled graph 上继续修补。

H1 成立标准：

最终 CUDA / persistent candidate 满足：

$$
T_{\text{backward,new}}\leq1.50T_{\text{backward,MLP-match}},
$$

并且相比 G2：

$$
T_{\text{backward,new}}\leq0.75T_{\text{backward,G2}}.
$$

同时：

$$
GradRelErrMax\leq10^{-4},
$$

$$
R^2_{\text{pairwise}}\geq0.95.
$$

### H2：若 FC-D2 final system attempt 仍失败，不代表可以转 PureKANConv；本轮应回到 FC-PureKAN primitive redesign

H2 是路线约束。即使 FC-D2 仍然无法闭合，也不在本轮 pivot Conv / Former，而是判定：

```text
FC-D2 current lowering infeasible
next_required = FC-PureKAN primitive redesign
```

这个 redesign 仍必须在 FC FullEdge 范围内，例如：

```text
1. GEMM-native edge-basis dense lowering；
2. shallower but interaction-retaining FullEdge；
3. different basis with cheaper derivative；
4. coefficient factorization that remains edge-equivalent；
5. trainability-preserving but lower-rank compositional form；
6. full-edge layer normalization/source redesign。
```

H2 成立标准：

如果 FC-D2 final CUDA/persistent attempt 失败，则 route 不写 `KANConvPivotRequired`，而写：

```text
R5-FCD2CurrentLoweringLimit_FCPureKANRedesignRequired
```

### H3：系统闭合不能破坏 synthetic interaction retention

P4 pass 不能通过把 D2 退化成 additive model获得。

H3 成立标准：

$$
R^2_{\text{pairwise}}\geq0.95.
$$

并且相比 depth1：

$$
R^2_{\text{D2}}-R^2_{\text{D1}}\geq0.50.
$$

若 P4 pass 但 R2 下降，则 candidate 无效。

### H4：P4 关闭后，P5 可能仍失败；这时 blocker 才转为 AdamW trainability

H4 约束解释。只有当 P4 关闭后，才允许讨论 P5 trainability。如果 P4 关闭但 P5 失败，则 route 写：

```text
R7-P4ClosedButAdamWTrainabilityFail
```

而不是重新回到系统问题。

### H5：Functional update 不能提前打开

Functional update 只有在 P5 near-pass 后打开。若 P4 失败或 P5 失败，functional update 必须继续 `not_opened`。

---

## 5. Candidate 设计

### 5.1 Baseline candidates

```text
B0-MLP-match:
  同参数 MLP-match，CE-only，AdamW。

K0-S3-B2-P4-F0:
  单层 active-k P4-pass / P5-fail candidate。

D2-G2-current:
  v9.2.4 best G2 GEMMNative-FusedPointwise compiled candidate。

D2-B9-reference:
  v9.2.4 monolithic Triton backward reference。
```

### 5.2 FC-D2 final CUDA / persistent candidates

```text
FC1-CUDAExtensionLayerBackwardMinimal:
  CUDA C++ extension，单层 persistent backward，只计算 dU/dA/dx correction，不计算 dV/dW0。

FC2-CUDAExtensionLayerBackwardFull:
  CUDA C++ extension，计算 dU/dA/dx correction + partial dV/dW0 tile accumulation。

FC3-CUDAExtensionTwoLayerStreaming:
  两层 D2 backward streaming，layer2 temp 释放后再 layer1，workspace 复用。

FC4-CUDAExtensionOneBufferD2:
  D2 全 backward 只使用一个 persistent workspace buffer。

FC5-CUDAExtensionForwardBackwardPair:
  forward 与 backward 都由 CUDA extension 管理 workspace，减少 compiled temp。

FC6-CUDAGraphStaticShapeRetry:
  只复测 CUDA Graph capture issue，不作为主要修复；若 runtime 仍报错则记为 not_run。
```

### 5.3 FC-PureKAN primitive redesign candidates

只有当 FC1-FC5 均不能关闭 P4 时打开。它们仍是 FC FullEdge，不是 Conv / Former。

```text
R1-GEMMNativeEdgeBasisDense-T2:
  Y = XW0 + B2(X)W2。
  这是 full-edge basis channel 的 GEMM-native dense lowering，不是 MLP hidden path。

R2-GEMMNativeEdgeBasisDense-T2T3:
  Y = XW0 + B2(X)W2 + B3(X)W3。

R3-GEMMNativeDepth2-TiedBasis:
  depth2，但两层共享 cheap basis 与 lowering 模板，减少 kernel fragmentation。

R4-ChebyshevT2-CheapDerivative:
  使用更便宜的 T2 derivative path，并记录 derivative FLOPs / bytes。

R5-LegendreP2P3-CheapDerivative:
  用 Legendre P2/P3 替换 Chebyshev，比较 derivative cost 与 conditioning。

R6-EdgeCoefficientBlockFactorized:
  block factorized coefficient tensor，但仍保持 edge coefficient factorization equivalent。

R7-InteractionRetainingLowRankD2:
  depth2 low-rank，但必须保留 synthetic pairwise R2 >= 0.95。
```

### 5.4 Deferred candidates

以下只写入 deferred registry，不执行：

```text
Deferred-PureKANConv
Deferred-PureKANFormer
Deferred-KANFFN
```

它们必须记录：

```text
status = deferred
reason = FC_PureKAN_not_yet_P5_near_pass
measured = 0
eligible_for_route = 0
```

---

## 6. 实验阶段

## P0：route tightening and contract audit

### 目标

确认 v9.2.5 已删除 PureKANConv / PureKANFormer active route，并复现 v9.2.4 边界。

### 必须记录

```text
candidate_id
candidate_family
status
measured
eligible_for_route
loss_type
label_smoothing
external_teacher_used
self_teacher_used
uses_loss_backward
fake_data_used
proxy_row_used
full_edge_equivalence_pass
no_external_residual_pass
ordinary_mlp_hidden_path_used
trainable_preprocessor_used
purekanconv_status
purekanformer_status
deferred_reason
```

### 判断标准

Contract pass：

```text
PureKANConv_status = deferred_until_FC_PureKAN_P5_near_pass
PureKANFormer_status = deferred_until_FC_PureKAN_P5_near_pass
No PureKANConv measured candidate
No PureKANFormer measured candidate
FullEdge candidates satisfy full_edge_equivalence_pass = 1
```

### 可视化

```text
p0_route_tightening_contract_heatmap.svg
p0_deferred_architecture_registry.md
p0_candidate_lattice_fc_only.svg
```

---

## P1：v9.2.4 boundary reproduction

### 目标

复现 v9.2.4 的 `G2` boundary，确保本轮不是在不稳定测量上继续。

### 必须记录

```text
candidate_id
forward_ratio
backward_ratio
step_ratio
memory_ratio
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
unknown_time_fraction
phase_time_sum_fraction
kernel_count_total
small_kernel_count
```

### 判断标准

重复稳定：

$$
|r_{\text{backward}}-2.042472|\leq0.15.
$$

Profiler sanity：

$$
unknown\_time\_fraction\leq0.10.
$$

如果 P1 不稳定，先修 profiler / measurement，不进入 P2。

### 可视化

```text
p1_v924_boundary_repeat.svg
p1_gate_status_dashboard.svg
p1_repeat_vs_previous_ratio.svg
```

---

## P2：FC-D2 final lowering feasibility audit

### 目标

判断 FC-D2 是否值得做最后的 CUDA / persistent attempt。P2 是理论和 profiling 合并审计，不直接声称成功。

### 必须记录

```text
primitive_id
formula
input_shape
output_shape
flops
bytes_read
bytes_written
arithmetic_intensity
current_time_ms
estimated_lower_bound_ms
recommended_lowering
requires_atomic
requires_reduction
requires_workspace
gemm_coverage_fraction
persistent_kernel_needed
estimated_backward_lower_bound_ratio
estimated_forward_lower_bound_ratio
estimated_memory_lower_bound_ratio
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
dA
dx_correction
dh_propagation
```

### 判断标准

FC-D2 final attempt allowed if：

```text
gemm_coverage_fraction + persistent_kernel_coverage_fraction >= 0.80
estimated_backward_lower_bound_ratio <= 1.50
estimated_memory_lower_bound_ratio <= 1.05
```

若 lower-bound audit 已经显示不可行，则跳过 P3，进入 P5 FC-PureKAN primitive redesign，不进入 Conv / Former。

### 可视化

```text
p2_roofline_proxy.svg
p2_primitive_lowering_table.md
p2_backward_lower_bound_waterfall.svg
p2_gemm_persistent_coverage.svg
```

---

## P3：FC-D2 CUDA / persistent final attempt

### 目标

这是 FC-D2 当前 lowering 的最后一次严肃系统尝试。它必须是 CUDA extension 或真正 persistent layer-level kernel，不是小 Triton patch。

### 必跑 candidates

```text
FC1-FC6
```

### 必须记录

```text
candidate_id
cuda_extension_used
persistent_kernel_used
computes_dU
computes_dA
computes_dx_correction
computes_partial_dV
computes_partial_dW0
workspace_bytes
shared_memory_bytes
registers_per_thread
occupancy_estimate
atomic_ops_count
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

Correctness：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

Interaction retention：

$$
R^2_{\text{pairwise}}\geq0.95.
$$

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

Improvement over G2：

$$
T_{\text{backward,FCx}}\leq0.75T_{\text{backward,G2}}.
$$

### 可视化

```text
p3_cuda_persistent_p4_pareto.svg
p3_cuda_correctness.svg
p3_backward_improvement_vs_g2.svg
p3_memory_workspace_breakdown.svg
```

---

## P4：FC-D2 viability decision

### 目标

在 P2/P3 后做明确决策，不允许写模糊的 “more kernel work”。注意：即使 FC-D2 失败，本轮也不 pivot Conv / Former。

### 判定规则

Continue FC-D2 if：

```text
at least one FCx:
  P4_pass = 1
  GradPass = 1
  R2_pairwise >= 0.95
```

Enter FC-PureKAN redesign if：

```text
No FCx closes P4
or
Only P4-pass candidates lose interaction
or
CUDA extension implementation exceeds complexity budget and still >1.50 backward
```

### 必须记录

```text
fc_d2_continue_allowed
fc_purekan_redesign_required
kanconv_pivot_allowed
purekanformer_pivot_allowed
best_fc_candidate
fc_primary_blocker
fc_next_action
```

### 判断标准

本轮必须满足：

```text
kanconv_pivot_allowed = 0
purekanformer_pivot_allowed = 0
```

### 可视化

```text
p4_fc_d2_viability_decision_tree.svg
```

---

## P5：FC-PureKAN primitive redesign gate

### 目标

如果 FC-D2 final attempt 失败，不进入 Conv，而是研究新的 FC-PureKAN primitive 是否能同时保留 interaction 与 P4。

### 必跑 candidates

```text
R1-R7
```

### 必须记录

```text
candidate_id
primitive_family
depth
basis
coefficient_factorization
FullEdgeEquivalencePass
NoExternalResidualPass
ordinary_mlp_hidden_path_used
GradRelErrMax
GradCosMin
synthetic_additive_R2
synthetic_pairwise_R2
forward_ratio
backward_ratio
step_ratio
memory_ratio
basis_condition_number
basis_usage_entropy
dominant_basis_fraction
```

### 判断标准

Redesign candidate promotion pass：

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

If no redesign candidate passes, route becomes:

```text
R8-FCPureKANSystemNotClosed
```

### 可视化

```text
p5_fc_purekan_redesign_pareto.svg
p5_interaction_vs_system.svg
p5_basis_usage_heatmap.svg
p5_redesign_candidate_dashboard.svg
```

---

## P6：AdamW-only trainability re-entry

### 目标

只有 P3 或 P5 出现 P4-pass candidate 后才打开。验证 base learner 是否能训练，不使用 functional update。

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
route_family
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
forward_ratio
backward_ratio
step_ratio
memory_ratio
```

### 判断标准

Trainability near-pass：

$$
Acc_{\text{candidate}}\geq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows，并且：

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

Trainability pass：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

If P4 pass but P6 fails, route is:

```text
R7-P4ClosedButAdamWTrainabilityFail
```

### 可视化

```text
p6_trainability_gap_bar.svg
p6_loss_curves.svg
p6_ce_tail.svg
p6_margin_distribution.svg
p6_task_system_pareto.svg
```

---

## P7：Functional update open decision

### 目标

只决定 functional update 是否允许打开。

### Functional open criteria

```text
P4_pass = 1
P5_near_pass = 1
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
```

### 必须记录

```text
candidate_id
route_family
p4_pass
p5_near_pass
p5_pass
functional_open_allowed
functional_not_open_reason
```

### 判断标准

Functional allowed only if：

```text
p4_pass = 1
p5_near_pass = 1
```

### 可视化

```text
p7_functional_open_decision.svg
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_audit_v925_fc_only.csv
deferred_architecture_registry_v925.csv
p1_v924_boundary_repeat.csv
p2_fc_d2_final_lowering_audit.csv
p3_fc_d2_cuda_persistent_attempt.csv
p4_fc_d2_viability_decision.csv
p5_fc_purekan_redesign_gate.csv
p6_adamw_trainability_reentry.csv
p7_functional_open_decision.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_full_edge_equivalence_fail
F3_deferred_architecture_violation
F4_measurement_instability
F5_lowering_audit_incomplete
F6_cuda_persistent_correctness_fail
F7_fc_d2_backward_lowering_limit
F8_fc_d2_forward_or_memory_fail
F9_fc_d2_interaction_lost
F10_fc_purekan_redesign_no_p4_candidate
F11_adamw_trainability_fail
F12_functional_not_opened
F13_fake_or_proxy_violation
F14_artifact_missing
```

---

## 8. Route decision

### Route cases

```text
R1-FCD2P4Closed:
  FC-D2 closes P4 via CUDA / persistent attempt.

R2-FCD2P4ClosedP5NearPass:
  FC-D2 closes P4 and reaches P5 near-pass.

R3-FCD2BackwardLoweringLimitConfirmed:
  final CUDA / persistent attempt cannot close backward <=1.50.

R4-FCD2InteractionLostDuringClosure:
  FC-D2 P4 closes only by losing synthetic interaction.

R5-FCD2CurrentLoweringLimit_FCPureKANRedesignRequired:
  FC-D2 final attempt fails; enter FC-PureKAN redesign, not Conv / Former.

R6-FCPureKANRedesignP4Closed:
  redesigned FC-PureKAN candidate closes P4.

R7-P4ClosedButAdamWTrainabilityFail:
  P4 closes, but AdamW-only trainability fails.

R8-FCPureKANSystemNotClosed:
  neither FC-D2 final attempt nor FC redesign closes P4.

R9-DeferredArchitectureViolation:
  PureKANConv / PureKANFormer was measured or used in route before FC-PureKAN near-pass.

R10-ContractFail:
  no-teacher/no-loss/no-fake/PureKAN contract violated.
```

### route_decision.json 必须记录

```text
route
best_candidate
route_family
fc_d2_continue_allowed
fc_purekan_redesign_required
kanconv_deferred
purekanformer_deferred
best_fc_candidate
best_redesign_candidate
full_edge_equivalence_pass
no_external_residual_pass
grad_pass
synthetic_pairwise_R2
forward_ratio
backward_ratio
step_ratio
memory_ratio
p4_pass
p5_trainability_opened
p5_near_pass
functional_open_allowed
primary_blocker
next_required_implementation
success_v925_fc_d2_closure
success_v925_fc_purekan_redesign
success_v925_trainability_reentry
success_v925_functional_opened
```

---

## 9. 第一轮执行顺序

```text
Step 1:
  P0 route tightening and contract audit。
  先确认 PureKANConv / PureKANFormer 都是 deferred，不产生 measured candidate。

Step 2:
  P1 复现 v9.2.4 boundary。
  如果测量不稳定，先修 profiler。

Step 3:
  P2 做 FC-D2 final lowering feasibility audit。
  如果理论 lower-bound 已不可行，则跳过 P3，进入 P5 FC-PureKAN redesign。

Step 4:
  P3 执行 FC-D2 CUDA / persistent final attempt。
  这是 FC-D2 当前 lowering 的最后一次严肃系统尝试。

Step 5:
  P4 做 FC-D2 viability decision。
  不允许写模糊的 more kernel work。

Step 6:
  如果 FC-D2 未闭合，执行 P5 FC-PureKAN primitive redesign gate。
  注意这仍是 FC-PureKAN，不是 Conv / Former。

Step 7:
  只有 P3 或 P5 出现 P4-pass candidate，才执行 P6 AdamW-only trainability。

Step 8:
  只有 P6 near-pass 后，才允许 P7 functional update open decision。
```

---

## 10. 停止条件

### 成功停止

FC-D2 route success：

```text
FC-D2 P4 pass
GradPass
SyntheticInteractionPass
NoExternalResidualPass
```

FC-PureKAN redesign success：

```text
Redesigned FC-PureKAN P4 pass
GradPass
SyntheticInteractionPass
NoExternalResidualPass
```

Trainability success：

```text
P4 pass
P5 near-pass
```

Strong v9.2.5 success：

```text
P4 pass
P5 pass
functional_open_allowed = 1
```

### 失败停止

```text
1. v9.2.4 boundary cannot be reproduced；
2. deferred architecture rule is violated；
3. FC-D2 lower-bound audit incomplete；
4. FC-D2 CUDA / persistent correctness fails；
5. FC-D2 final attempt still backward > 1.50；
6. FC-D2 closes P4 only by losing synthetic interaction；
7. FC-PureKAN redesign candidates all fail P4；
8. P4 pass but AdamW trainability fails；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 11. 最终解释规则

### Case A：FC-D2 closes P4

可以声明：

```text
Flattened D2 compositional FullEdge remains system-viable.
```

但仍不能声明 functional 或 external success，除非 P6/P7 后续通过。

### Case B：FC-D2 fails, FC redesign succeeds

可以声明：

```text
Current FC-D2 lowering is limited, but FC-PureKAN remains viable through redesigned FC FullEdge primitive.
```

### Case C：FC-D2 and FC redesign both fail

必须声明：

```text
Current FC-PureKAN routes do not yet have an MLP-comparable kernel-native implementation.
```

这时仍然不直接研究 PureKANConv / PureKANFormer；只能在总路线中记录：

```text
PureKANConv / PureKANFormer remain deferred until FC-PureKAN reaches P5 near-pass.
```

### Case D：P4 closes but P5 fails

必须声明：

```text
System blocker is solved; next blocker is AdamW-only trainability.
```

---

## 12. 最终建议

v9.2.5 的一句话策略是：

$$
\boxed{
\text{先让 FC-PureKAN 比肩 MLP；暂不研究 PureKANConv / PureKANFormer。}
}
$$

当前最关键的问题是：

```text
1. FC-D2 是否能通过最后一次 CUDA / persistent attempt 闭合 P4？
2. 如果不能，是否存在新的 FC-PureKAN primitive 能在保留 interaction 的同时闭合 P4？
3. P4 闭合后，AdamW-only trainability 是否能 near-pass？
4. Functional update 是否有资格重新打开？
```

在这四个问题回答之前，PureKANConv 与 PureKANFormer 都不进入 active route。
