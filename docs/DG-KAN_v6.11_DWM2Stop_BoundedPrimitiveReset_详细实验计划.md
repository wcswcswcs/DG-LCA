# DG-KAN v6.11：DWM2 Stop 后的 Bounded-Workspace PureKAN Primitive Family Reset 与 Terminal Kernel 判定实验计划

> 本计划基于 v6.10 final real-only run 制定。v6.10 已经真实采集 PyTorch CUDA memory history，并输出 `STOP_DWM2_PATCHING`。当前结论不是“DG-KAN 方向失败”，而是“DWM2-poly2 的局部 patch 路线已经不足”。因此 v6.11 不再继续做 DWM2 小修、局部 microkernel 替换、optimizer sweep 或 functional correction，而是执行两条严格路线：  
> **A. 只允许一次 terminal DWM2 true single-call full-step kernel 判定；**  
> **B. 主线切换到 residual-effective bounded-workspace PureKAN primitive family reset。**

---

## 0. 实验整体目标

v6.11 的整体目标是把项目从当前的 DWM2 小修循环中拉出来，完成一次路线级决策：

$$
\boxed{
\text{DWM2-poly2 是否还有必要继续作为主 primitive？}
}
$$

以及：

$$
\boxed{
\text{是否存在一个新的 bounded-workspace PureKAN primitive，可以同时满足 residual 非平凡、memory near-pass、step near-pass？}
}
$$

v6.10 的真实基线如下：

```text
DWM2-current:
  memory ratio mean = 1.2918
  step ratio mean   = 1.8969 in P1
  backward ratio mean = 1.4429 in P1

P3 current:
  memory ratio mean = 1.2918
  step ratio mean   = 1.9693
  backward ratio mean = 1.5433

P1 exact live-set:
  top3 gap fraction mean = 0.4823
  top3 gap fraction max  = 0.5058
  attribution pass       = 0/18

P2 K3 triton-fused-dx-coeffgrad:
  mean time ratio vs current = 0.3110
  grad pass = 8/8

P3/P4/P5:
  single-call full-layer / multi-layer chunked / true one-buffer packages not implemented

P6 reset:
  residual effect pass = true for scale 0.02 / 0.05
  reset memory ratio mean ≈ 1.2981
  reset step ratio mean   ≈ 1.81
  reset near pass = 0
```

v6.10 的 route 是：

```text
STOP_DWM2_PATCHING
next_required_implementation = real_single_call_full_step_or_new_primitive_family
```

因此 v6.11 的最低工程成功标准是找到一个真实 candidate 满足：

$$
r_{\text{mem}}=\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}\leq1.05,
$$

$$
r_{\text{step}}=\frac{T_{\text{step,KAN}}}{T_{\text{step,MLP}}}\leq1.50,
$$

并且：

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

正式 task re-entry 标准是：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

强目标是：

$$
r_{\text{mem}}\leq0.90,
$$

$$
r_{\text{step}}\leq1.20.
$$

终极目标仍是：

$$
r_{\text{mem}}\leq0.80.
$$

v6.11 不把 $0.80$ 作为唯一 gate。v6.11 的现实目标是：确定下一条可推进主线。

---

## 1. 当前问题分析

### 1.1 已经成立的部分

当前已经比较稳的部分是：

```text
strict PureKAN contract
graph-free/manual backward path
manual gradient correctness
no-fake/no-proxy 实验纪律
P7/P8/P9/P10 gate 纪律
```

因此 v6.11 不需要重新证明 “manual adjoint 是否能算对”。所有新 primitive 仍需做 gradient check，但主问题不是公式正确性。

### 1.2 当前失败的真实来源

v6.10 的失败不是 task 失败，也不是 optimizer 失败。失败发生在更底层的 memory/time gate：

```text
F1_memory_fail
F2_step_time_fail
F4_attribution_incomplete
F5_boundary_overhead
F6_live_set_overlap
F10_reset_memory_fail
```

其中最核心的是：

```text
current DWM2 attribution top3 gap fraction 只有约 48%
single-call full-layer / one-buffer full-step 未实现
reset residual effect 已经过，但 workspace model 没有改变
```

### 1.3 为什么不能继续 DWM2 小修

从 v6.5 到 v6.10，DWM2 小修已经连续失败：

```text
noHiddenCache:
  hidden cache 消除，但 total memory 几乎不降，step 变慢

bf16Cache:
  memory 不降，gradient relerr 到 1e-3 量级

bufferReuse-v1:
  actual CUDA peak 不降，memory/time 变差

deltaStreaming-v1:
  memory ratio 与 current 完全相同

Torch-level FlashDWM2:
  gradient 正确，但 memory 不降，time 明显变慢

Triton microkernel K3:
  局部快，但 full-step 不快

L1/L2 partial full-layer:
  memory/time 比 current 更差

one-buffer diagnostics:
  不是 true one-buffer

reset scale:
  residual effect 过，但 memory/time 不过
```

因此 v6.11 只允许 DWM2 保留一个 terminal 判定实验：

```text
DWM2 true single-call full-step kernel
```

如果这个仍然没有 near-pass，则 DWM2-poly2 从主线冻结，不再继续 patch。

---

## 2. v6.11 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。所有未实现内容必须写为：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，不允许继续新增 DWM2 局部 patch，例如：

```text
another coeffgrad-only kernel
another deltaStreaming variant
another bufferReuse wrapper
another Torch-level local reduce
another noHidden variant
```

除非它属于 terminal true single-call full-step kernel 的必要组成部分。

第三，不允许把 microkernel pass 写成 full-step pass。只有 full-step memory/time 过 gate 才允许进入 P7 task re-entry。

第四，不允许在 memory/time survivor 之前打开：

```text
task recipe sweep
optimizer exploration
LightSmooth
functional correction
3-seed / 5-seed / 10-seed confirm
```

第五，不允许把 manual-linear reference 当成 PureKAN success。它只能作为 lower-bound reference。

第六，不允许 reset primitive 只靠 residual effect pass。Reset 必须同时满足：

```text
residual effect pass
memory/time near-pass
strict PureKAN edge ownership
manual gradient correctness
```

第七，不允许只调 residual scale。v6.10 已经证明 residual scale 能到 0.02/0.05；v6.11 reset 必须改变 workspace model。

---

## 3. 核心假设

### H1：DWM2 的唯一可继续条件是真正 single-call full-step kernel

H1 假设：DWM2 当前失败主要是 full-step live-set / boundary / materialization 问题。只有把完整 backward step 的 live-set 重新规划，才可能让局部 Triton microkernel 收益转化为 full-step 收益。

DWM2 terminal kernel 至少要融合：

```text
upstream delta consume
dz or direct delta propagation
dx production or streaming to previous layer
coeffgrad accumulation
update-prep buffer
temporary buffer reuse
```

H1 成立要求：

$$
r_{\text{mem,DWM2-terminal}}\leq1.05,
$$

$$
r_{\text{step,DWM2-terminal}}\leq1.50,
$$

且相对 current：

$$
\frac{M_{\text{terminal}}}{M_{\text{current}}}\leq0.90,
$$

$$
\frac{T_{\text{terminal}}}{T_{\text{current}}}\leq0.90.
$$

如果 H1 不成立，DWM2-poly2 应冻结。

---

### H2：当前 attribution 不足以指导继续 patch，必须做 stable op/tensor taxonomy

v6.10 虽然采集了 PyTorch memory history，但 top3 gap fraction mean 只有约 0.4823，低于 0.70 gate。H2 假设：需要把 allocation stack 映射到稳定 taxonomy，而不是只保留 hash/source file。

定义：

$$
G_{\text{peak}}=M_{\text{peak,DWM2}}-M_{\text{peak,MLP}}.
$$

H2 成立要求：

$$
\frac{M_{\text{top3}}}{G_{\text{peak}}}\geq0.70,
$$

并且 unknown gap：

$$
\frac{M_{\text{unknown}}}{G_{\text{peak}}}\leq0.05.
$$

新增 taxonomy 至少包括：

```text
delta_z_live
delta_x_live
coeffgrad_contribution_live
coeffgrad_partial_live
update_temp_live
forward_transform_live
mixing_output_live
optimizer_state_live
triton_input_materialization
triton_output_materialization
layout_conversion_temp
contiguous_copy_temp
allocator_padding
unknown
```

如果 H2 仍不成立，DWM2 不能继续 patch，只能转 primitive reset。

---

### H3：reset primitive 的核心是 bounded workspace，而不是 residual scale

v6.10 reset 的 residual effect 已经过，但 memory/time 没有 near-pass。H3 假设：新的 reset primitive 必须改变 memory model。

Reset primitive 必须满足：

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02,
$$

并且：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.35.
$$

如果 residual effect pass 但 memory/time fail，则该 reset 不能进入 task。

---

### H4：mixing path 可能是 reset family 的 hidden blocker

ManualLinear reference memory/time 更接近 MLP，但加入 residual 后 reset memory/time 回到 DWM2-like 区间。H4 假设：reset 失败不仅是 residual transform，也可能是 residual 与 mixing path 的 live-set overlap 或 output mixing workspace。

H4 成立的证据是：

```text
residual transform temp 不大
mixing output / chunked mix / update workspace 占 peak top source
chunked-mix 降低 peak 或 step
lowrank-mix 降低 peak 或 step
```

如果 H4 成立，reset 主线应优先研究：

```text
chunked mixing
edge-owned low-rank mixing
block-diagonal mixing
grouped residual + grouped mixing
```

---

### H5：optimizer 探索仍然不是当前主线

H5 是门禁假设。Optimizer exploration 只有在出现 memory/time survivor 后才有意义。

打开 optimizer 的必要条件：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35,
$$

且 one-step probe pass。否则 optimizer 改善 task step 数也无法解决 memory/time gate。

---

### H6：如果 terminal DWM2 和 reset v4 都没有 near-pass，应切换到 new primitive family design

H6 是 Stop/Go 假设。如果本轮完成：

```text
DWM2 terminal kernel measured
bounded reset v4 measured
attribution taxonomy improved or declared insufficient
```

仍没有 S0/S1/S2，则不应继续 patch DWM2。下一轮应设计全新 primitive family，而不是继续当前 DWM2-poly2 branch。

---

## 4. 实验阶段总览

v6.11 分为十二个阶段：

```text
P0: v6.10 reproduction and stop/go contract
P1: stable peak live-set taxonomy attribution
P2: terminal DWM2 true single-call full-step kernel
P3: bounded reset v4 primitive family design
P4: reset microkernel and workspace audit
P5: reset full-step benchmark
P6: one-step numerical and residual-effect probe
P7: limited task re-entry gate
P8: optimizer exploration remains gated
P9: functional correction remains gated
P10: route decision
P11: artifact and failure audit
P12: final stop/go recommendation
```

P1-P5 是核心。P6 只在 P2/P5 有 near-pass 后运行。P7-P9 默认关闭。

---

## 5. P0：v6.10 reproduction and stop/go contract

### 5.1 目的

确认 v6.11 与 v6.10 baseline 可比，并确认 DWM2 不再接受小修 patch。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-current
DWM2-K3-triton-fused-dx-coeffgrad-reference
DWM2-terminal-single-call-full-step, if implemented
ResidualBoundedReset-v3-scale002-current
ResidualBoundedReset-v3-scale005-current
Reset-v4 candidates, if implemented
```

### 5.3 必须记录字段

```text
variant
status
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
uses_triton_kernel
uses_cuda_extension
manual_forward_available
manual_backward_available
manual_update_available
nonKAN_param_count
edge_param_count
rollback_max_error
grad_relerr_max
grad_cos_min
v610_memory_ratio_mean
v611_memory_ratio_mean
v610_step_ratio_mean
v611_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
dwm2_patch_type
is_allowed_dwm2_patch
```

### 5.4 判断标准

Reproduction pass:

$$
|r_{\text{mem,v611}}-r_{\text{mem,v610}}|\leq0.05.
$$

$$
|r_{\text{step,v611}}-r_{\text{step,v610}}|\leq0.15.
$$

Contract pass:

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0 for PureKAN candidates
rollback_max_error < 1e-8
```

DWM2 patch policy pass:

```text
is_allowed_dwm2_patch = 1 only for terminal single-call full-step package
```

### 5.5 可视化

```text
p0_reproduction_delta_bar.svg
p0_contract_heatmap.svg
p0_dwm2_patch_policy_table.md
```

---

## 6. P1：stable peak live-set taxonomy attribution

### 6.1 目的

P1 要完成 v6.10 未完成的 attribution 闭环：把 exact live-set allocation stack 映射到稳定 source taxonomy，使 top-3 source 达到计划阈值，或者明确证明当前 instrumentation 无法解释。

### 6.2 方法

P1 使用：

```text
torch.cuda.memory._record_memory_history
torch.cuda.memory._snapshot
NVTX ranges
custom tensor registry
allocation stack hash
op/tensor taxonomy mapper
```

如果可用，补充：

```text
Nsight Compute
CUPTI allocation callback
```

### 6.3 必须标记的 phase

```text
phase_forward_transform
phase_forward_mix
phase_loss_delta
phase_backward_output_delta
phase_backward_layer_begin
phase_backward_dz
phase_backward_dx_coeffgrad
phase_backward_update_prep
phase_update_params
phase_cleanup
```

### 6.4 必须记录字段

```text
variant
dataset
batch_size
depth
peak_timestamp
peak_phase
peak_allocated_MB
MLP_peak_allocated_MB
peak_gap_MB
exact_live_tensor_count
exact_live_tensor_total_MB
top1_live_tensor_name
top1_live_tensor_MB
top1_source_taxonomy
top1_allocation_stack_hash
top1_lifetime_start_phase
top1_lifetime_end_phase
...
top20_live_tensor_name
top20_live_tensor_MB
top20_source_taxonomy
top20_allocation_stack_hash
top20_lifetime_start_phase
top20_lifetime_end_phase
explained_gap_MB
unexplained_gap_MB
unknown_gap_fraction
top3_gap_fraction
taxonomy_mapper_version
```

### 6.5 Source taxonomy

每个 live tensor 必须映射到：

```text
T0_delta_z_live
T1_delta_x_live
T2_coeffgrad_contribution_live
T3_coeffgrad_partial_live
T4_update_temp_live
T5_forward_transform_live
T6_mixing_output_live
T7_optimizer_state_live
T8_triton_input_materialization
T9_triton_output_materialization
T10_layout_conversion_temp
T11_contiguous_copy_temp
T12_allocator_padding
T13_workspace_pool
T14_manual_cache
T15_unknown
```

### 6.6 判断标准

Attribution pass:

$$
\frac{M_{\text{top3}}}{G_{\text{peak}}}\geq0.70.
$$

Unknown pass:

$$
\frac{M_{\text{unknown}}}{G_{\text{peak}}}\leq0.05.
$$

可行动 blocker pass：

```text
primary_source_taxonomy != T15_unknown
top1_source_fraction >= 0.20
```

如果 P1 失败，route 进入：

```text
R3-AttributionStillIncomplete
```

并且 DWM2 不允许继续 patch。

### 6.7 可视化

```text
p1_exact_live_set_gantt.svg
p1_peak_live_tensor_top20.svg
p1_gap_attribution_stacked_bar.svg
p1_taxonomy_fraction_heatmap.svg
p1_unknown_gap_heatmap.svg
p1_lifetime_overlap_matrix.svg
```

---

## 7. P2：terminal DWM2 true single-call full-step kernel

### 7.1 目的

P2 是 DWM2 的最后一次 Go/Stop 实验。它必须实现真实 terminal package，不再接受 partial integration。

### 7.2 允许的 DWM2 terminal candidates

```text
D0-current

D1-terminal-single-call-depth2:
  for depth=2, fuse complete backward step as much as possible

D2-terminal-single-call-layer:
  single layer backward kernel with dx + coeffgrad + update prep

D3-terminal-single-call-full-step:
  entire forward/backward/update-prep scheduling with bounded live buffers

D4-terminal-onebuffer-full-step:
  enforce live buffer count peak <= 3

D5-terminal-chunked-mix-full-step:
  include chunked mixing to reduce output workspace
```

### 7.3 不允许的 candidates

```text
coeffgrad-only
dx-only
deltaStreaming-only
bufferReuse-only
noHidden-only
bf16Cache-only
torch-level local reduce
```

这些只能作为 internal component，不允许单独进入结果表作为新 DWM2 patch。

### 7.4 必须记录字段

```text
package
status
implementation_scope
calls_per_step
live_buffer_count_peak
live_buffer_total_MB_peak
kernel_count_total
triton_call_count
torch_call_count
boundary_time_ms
materialization_MB
layout_conversion_MB
memory_ratio_min
memory_ratio_mean
memory_ratio_max
step_ratio_min
step_ratio_mean
step_ratio_max
backward_ratio_mean
memory_improvement_vs_current
step_improvement_vs_current
grad_relerr_max
grad_cos_min
rollback_error
```

### 7.5 判断标准

DWM2 terminal useful:

$$
\frac{M_{\text{terminal}}}{M_{\text{current}}}\leq0.90.
$$

$$
\frac{T_{\text{terminal}}}{T_{\text{current}}}\leq0.90.
$$

Near-pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Task-open pass:

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

One-buffer pass:

$$
\operatorname{live\_buffer\_count\_peak}\leq3.
$$

If DWM2 terminal package fails useful gate, DWM2 is frozen.

### 7.6 可视化

```text
p2_dwm2_terminal_pareto.svg
p2_live_buffer_count_timeline.svg
p2_boundary_materialization_bar.svg
p2_current_vs_terminal_waterfall.svg
p2_batch_depth_stability_heatmap.svg
```

---

## 8. P3：bounded reset v4 primitive family design

### 8.1 目的

P3 设计新的 bounded-workspace PureKAN primitive family。它不是继续 DWM2，也不是 manual-linear baseline。每个 primitive 必须有非平凡 residual，并且 memory model 要从设计上 bounded。

### 8.2 Reset-v4 candidate families

#### FAM-A：Inplace Residual + Chunked Mixing

```text
A0-inplace-poly1-chunked-mix
A1-inplace-poly2-chunked-mix
A2-inplace-piecewise2-chunked-mix
```

核心思想：residual transform 直接写回 activation buffer，不 materialize full residual tensor；mixing 按 output chunk 执行，控制 mixing output live set。

#### FAM-B：Edge-owned Low-Rank Mixing

```text
B0-lowrank-r4-poly1
B1-lowrank-r8-poly1
B2-lowrank-r4-piecewise2
B3-lowrank-r8-piecewise2
```

核心思想：将 mixing matrix 约束为 edge-owned low-rank factors：

$$
W = UV^\top,
$$

其中 $U,V$ 均属于 KAN edge system，不引入 non-KAN 参数。目标是减少 mixing workspace 和 backward delta workspace。

#### FAM-C：Grouped Bounded Residual Mixing

```text
C0-grouped-g4-poly1
C1-grouped-g8-poly1
C2-grouped-g4-piecewise2
C3-grouped-g8-piecewise2
```

核心思想：按 channel group 做 residual + mixing，减少 full dense mixing 的 live-set overlap。

#### FAM-D：Streaming Piecewise LocalKAN

```text
D0-piecewise2-forwardOnly-diagnostic
D1-piecewise2-streamingGrad
D2-piecewise4-streamingGrad
D3-piecewise2-chunkedMix-streamingGrad
```

核心思想：每个 channel 只激活 2 或 4 个 local knots，反向使用 streaming gradient，不构造 dense basis。

### 8.3 必须满足的结构约束

所有 reset candidates 必须记录：

```text
nonKAN_param_count = 0
edge_param_count > 0
residual_param_count > 0
manual_backward_available = 1
loss_backward_used = 0
torch_autograd_graph_used = 0
```

### 8.4 必须记录字段

```text
primitive
family
variant
residual_type
mixing_type
chunk_size
rank
group_count
nonKAN_param_count
edge_param_count
residual_param_count
manual_backward_available
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
memory_improvement_vs_current
step_improvement_vs_current
grad_relerr_max
grad_cos_min
residual_over_base
residual_ablation_delta_loss
residual_ablation_delta_logit
live_buffer_count_peak
workspace_temp_MB
mixing_workspace_MB
```

### 8.5 判断标准

Structural pass:

```text
nonKAN_param_count = 0
manual_backward_available = 1
grad_relerr < 1e-4
grad_cos > 0.999
```

Residual effect pass:

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02.
$$

and either:

$$
|\Delta L_{\text{residual ablation}}|>10^{-4},
$$

or:

$$
|\Delta \text{logit}_{\text{residual ablation}}|>10^{-4}.
$$

Reset near-pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Reset task-open pass:

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

### 8.6 可视化

```text
p3_reset_family_memory_time_pareto.svg
p3_residual_effect_vs_memory.svg
p3_mixing_workspace_bar.svg
p3_rank_group_chunk_sweep_heatmap.svg
p3_structural_contract_heatmap.svg
```

---

## 9. P4：reset microkernel and workspace audit

### 9.1 目的

P4 对 P3 中 near-pass 或 close candidate 做 component audit，判断 bottleneck 在 residual transform、mixing、backward delta、coeffgrad 还是 update。

### 9.2 必须记录

```text
primitive
family
component
component_time_ms
component_peak_MB
component_kernel_count
component_allocation_count
component_grad_relerr
component_output_relerr
component_fraction_of_step_time
component_fraction_of_peak_gap
```

Components:

```text
residual_transform
mixing_forward
loss_delta
backward_mixing_delta
backward_residual_dx
backward_residual_params
update_prep
optimizer_update
```

### 9.3 判断标准

A component is repair target if:

$$
\frac{T_{\text{component}}}{T_{\text{step}}}\geq0.25
$$

or:

$$
\frac{M_{\text{component}}}{G_{\text{peak}}}\geq0.25.
$$

If no component dominates but total remains high, primitive design is likely not hardware-friendly.

### 9.4 可视化

```text
p4_component_runtime_waterfall.svg
p4_component_memory_waterfall.svg
p4_component_fraction_heatmap.svg
```

---

## 10. P5：one-step numerical and residual probe

P5 runs only for DWM2 terminal near-pass or reset near-pass candidates.

### 10.1 方法

```text
clone weights
compute manual gradient
apply one update
measure train loss before/after
measure holdout loss before/after
measure val loss before/after
measure residual ablation
rollback weights
check rollback exactness
```

### 10.2 必须记录

```text
candidate
family
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

### 10.3 判断标准

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

### 10.4 可视化

```text
p5_loss_before_after.svg
p5_bad_step_heatmap.svg
p5_update_norm_vs_loss_delta.svg
p5_residual_ablation_probe.svg
```

---

## 11. P6：limited task re-entry gate

### 11.1 打开条件

Open official task if:

```text
candidate_type in {DWM2-terminal, reset-v4}
survivor_type in {S0, S1}
P5 pass
memory_ratio < 1.0
step_ratio <= 1.35
grad pass
residual effect pass for reset
no fake/proxy
```

Open diagnostic-only task if:

```text
survivor_type == S2
memory_ratio <= 1.05
step_ratio <= 1.50
P5 pass
```

Otherwise P6 remains `not_run`.

### 11.2 如果打开 task

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
  Best-DWM2-terminal + ManualAdanLite
  Best-reset-v4 + ManualAdanLite
  Best-reset-v4 + ManualAdamW
```

### 11.3 必须记录

```text
train_loss_curve
val_loss_curve
test_acc
val_acc
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
seedwise_failure_reason
```

### 11.4 判断标准

Task pass:

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

Wall-clock must satisfy on at least two datasets:

$$
\operatorname{ValLossAUC}_{time,KAN}
\leq
\operatorname{ValLossAUC}_{time,MLP}.
$$

### 11.5 可视化

```text
p6_val_loss_vs_step.svg
p6_val_loss_vs_time.svg
p6_accuracy_vs_time.svg
p6_time_to_target_bar.svg
p6_task_efficiency_pareto.svg
p6_seedwise_delta_plot.svg
```

---

## 12. P7：optimizer exploration remains gated

P7 only opens after P6 official task pass. It is not a main stage of v6.11.

If opened, compare only:

```text
ManualAdamW
ManualAdanLite
ManualWinLite
Lookahead-ManualAdamW
WarmupCosine-ManualAdamW
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

## 13. P8：functional correction remains gated

P8 only opens after P6/P7 pass. Default status is `not_run`.

If opened, run only smoke:

```text
task-only best
task + small residual geometry correction
task + low-frequency LightSmooth
```

Pass standard:

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

## 14. P9：route decision

### Route cases

```text
R1-DWM2TerminalSolved:
  Terminal DWM2 package gets S0/S1 and P5 pass.
  DWM2 remains mainline.

R2-DWM2TerminalNearPass:
  Terminal DWM2 gets S2 with clear improvement.
  One more kernel cycle allowed, no task confirm.

R3-DWM2TerminalFail:
  Terminal DWM2 measured and fails useful gate.
  Freeze DWM2-poly2.

R4-AttributionStillIncomplete:
  P1 taxonomy attribution still incomplete.
  Freeze DWM2 patching unless new profiler exists.

R5-ResetV4Candidate:
  Reset-v4 gets S0/S1 or S2 and residual effect pass.
  Switch next cycle to reset primitive.

R6-ResetV4TooHeavy:
  reset residual effect pass but memory/time fail.
  Need new primitive family.

R7-NoViablePrimitive:
  DWM2 terminal fails and reset-v4 fails.
  Start new primitive family design.

R8-TaskReentryPass:
  memory/time survivor passes limited task.

R9-TaskReentryFail:
  memory/time survivor exists but task fails.
```

### JSON 输出字段

```text
route
best_candidate
best_family
best_memory_ratio
best_step_ratio
best_backward_ratio
memory_improvement_vs_current
step_improvement_vs_current
survivor_type
dwm2_terminal_measured
dwm2_terminal_pass
reset_v4_measured
reset_v4_pass
attribution_pass
open_task_reentry
open_optimizer_exploration
open_functional_correction
stop_dwm2_patching
next_required_implementation
no_fake
no_proxy
```

---

## 15. P10：artifact and failure audit

### failure types

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_attribution_incomplete
F5_dwm2_terminal_no_effect
F6_dwm2_terminal_not_implemented
F7_reset_residual_effect_fail
F8_reset_memory_fail
F9_reset_step_fail
F10_task_fail
F11_optimizer_gated
F12_functional_gated
F13_fake_or_proxy_violation
F14_artifact_missing
```

### artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p1_taxonomy_attribution.csv
p2_dwm2_terminal_full_step.csv
p3_reset_v4_family.csv
p4_reset_component_audit.csv
p5_one_step_probe.csv
p6_task_reentry.csv
p6_task_trace.csv
p7_optimizer_exploration.csv
p8_functional_correction_smoke.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### figures

```text
figures/p1_gap_attribution_stacked_bar.svg
figures/p1_live_tensor_top20.svg
figures/p1_taxonomy_fraction_heatmap.svg
figures/p2_dwm2_terminal_pareto.svg
figures/p2_live_buffer_count_timeline.svg
figures/p3_reset_family_memory_time_pareto.svg
figures/p3_residual_effect_vs_memory.svg
figures/p4_component_memory_waterfall.svg
figures/p5_loss_before_after.svg
figures/p9_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 16. 成功与失败解释规则

### Case A：DWM2 terminal pass

如果 terminal DWM2 达到 S0/S1，DWM2 可保留主线，但只允许在该 terminal kernel 上继续，不再回到局部 patch。

### Case B：DWM2 terminal measured but fails

如果 terminal DWM2 实现且 memory/time improvement 不到 $10\%$，冻结 DWM2-poly2。

### Case C：DWM2 terminal not implemented

如果本轮仍未实现 terminal kernel，不允许继续声称 DWM2 还有实验进展。路线必须转 reset primitive 或新 primitive family。

### Case D：reset-v4 pass

如果 reset-v4 同时 residual effect pass 与 memory/time near-pass，切换主线。

### Case E：reset-v4 residual pass but memory/time fail

说明 reset 表达力不是问题，workspace model 仍失败。下一轮不能继续调 scale，必须新 primitive。

### Case F：DWM2 和 reset 都失败

进入 new primitive family design，停止当前分支。

---

## 17. 最终建议

v6.11 的一句话策略是：

$$
\boxed{
\text{停止 DWM2 小修，执行 terminal DWM2 判定，同时主线转向 bounded-workspace primitive family reset。}
}
$$

如果 v6.11 仍无 S0/S1/S2，则应该正式写入路线结论：

```text
DWM2-poly2 is not a viable final efficient PureKAN primitive under current implementation model.
```

之后的主线应改为：

```text
new bounded-workspace KAN primitive family
```

而不是继续围绕 DWM2-poly2 做 patch。
