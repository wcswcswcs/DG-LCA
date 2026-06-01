# DG-KAN v6.9：Full-Step Live-Set Fusion、Attribution 闭环与 Residual-Effective Bounded Primitive 决策实验计划

> 本计划基于 v6.8 final real-only run 的真实结论制定。v6.8 已经证明：`K3-triton-fused-dx-coeffgrad` 在 microkernel 层面有局部加速且梯度正确，但这种局部收益没有转化成 full-step memory/time 收益；`D1/D2/D4` full backward packages 的 memory/time 均比 current 更差；residual-effective reset 已经把 residual/base 提到 `0.02/0.05`，但 memory/time 没有 near-pass。因此 v6.9 的目标不是继续做局部 coeffgrad kernel，也不是打开 task / optimizer / functional correction，而是完成 **full-step live-set fusion** 与 **可行动 attribution 闭环**，并在 DWM2 无法 near-pass 时做一次严格的 bounded primitive branch decision。

---

## 0. 实验整体目标

v6.9 的整体目标是回答两个问题。

第一个问题：

$$
\boxed{
\text{DWM2-poly2 是否能通过 full-step live-set fusion，而不是局部 coeffgrad microkernel，进入 memory/time near-pass？}
}
$$

第二个问题：

$$
\boxed{
\text{如果 DWM2 不能 near-pass，是否存在 residual 非平凡、bounded-workspace、strict PureKAN 的 reset primitive 可以接替？}
}
$$

v6.8 的真实结果给出了当前基线。`DWM2-current` 的 full-step 指标仍然是：

```text
memory ratio vs MLP:
  min  = 1.1441
  mean = 1.2918
  max  = 1.4866

P1 step ratio vs MLP:
  min  = 1.5986
  mean = 1.9698
  max  = 2.3642

P3 current step ratio vs MLP:
  min  = 1.7222
  mean = 2.1140
  max  = 2.7430

P1 backward ratio mean = 1.5417
P3 backward ratio mean = 1.7092
grad relerr max ≈ 1e-7
```

v6.8 的关键新事实是：Triton microkernel 层面已经有局部正信号。`K3-triton-fused-dx-coeffgrad` 的 micro time ratio 约为 current 的 `0.3147`，且梯度正确。但是 full-step package 结果为负：

```text
D2-triton-fused-dx-coeffgrad:
  memory ratio mean = 1.3472
  step ratio mean   = 2.5495
  backward ratio mean = 2.5661

D4-triton-full-backward-light:
  memory ratio mean = 1.3472
  step ratio mean   = 2.5535
  backward ratio mean = 2.5666
```

这说明：

$$
\boxed{
\text{局部 microkernel 加速并不等于 full-step 加速。}
}
$$

当前最可能的问题是：

```text
Triton/PyTorch boundary overhead
full-step live-set overlap
extra materialization at package boundary
forward / mix / update 仍由 Torch op 产生 temp
microkernel 输出与后续 update 的 lifetime 没有融合
allocator peak 没有被改变
```

因此 v6.9 的最低工程成功目标是找到至少一个 candidate 满足：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}\leq1.05,
$$

$$
\frac{T_{\text{step,KAN}}}{T_{\text{step,MLP}}}\leq1.50,
$$

且：

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

v6.9 的正式 task re-entry 标准是：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}<1.00,
$$

$$
\frac{T_{\text{step,KAN}}}{T_{\text{step,MLP}}}\leq1.35.
$$

强目标是：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}\leq0.90,
$$

$$
\frac{T_{\text{step,KAN}}}{T_{\text{step,MLP}}}\leq1.20.
$$

本轮仍不把 $0.80$ 作为唯一成功 gate。$0.80$ 是终极目标，但 v6.9 的现实目标是：完成 DWM2 full-step kernel 是否可修的判别，并产生一个明确的下一路线选择。

---

## 1. 当前问题分析

### 1.1 实验纪律已经足够可信

v6.8 final run 是 real-only run。所有关键 CSV / JSON / log 来自真实落盘结果，没有 fake data、proxy rows、固定占位 ratio 或手填结论。未实现项写为 `not_implemented`，Nsight / lower-level counters 不可用时写为 `metric_unavailable`，P5-P8 因 gate 写为 `not_run`。因此当前判断不是由于实验污染造成，而是候选确实没有通过 memory/time gate。

### 1.2 梯度正确性不是 blocker

P0/P1/P2/P3/P4 中 DWM2 与 reset candidates 的核心梯度 relerr 都在 $10^{-7}$ 量级，当前没有 gradient correctness failure。当前失败集中在：

```text
F1_memory_fail
F2_step_time_fail
F4_attribution_incomplete
F9_reset_no_near_pass
F6_triton_kernel_no_effect
```

因此不要把下一步重点放在重新推导 gradient 公式，而应放在 kernel / live-set / memory attribution。

### 1.3 局部 Triton microkernel 有用，但集成失败

`K3-triton-fused-dx-coeffgrad` 是 v6.8 唯一明确的局部正信号：

```text
micro time ratio vs current mean = 0.3147
micro memory ratio vs current mean = 0.9910
grad relerr max = 9.45e-08
triton pass = 8/8
```

这说明 Triton 写法不是完全无效。但 full-step package 更差，说明 microkernel 的输出和其他 phase 的 tensor lifetime 没有被重新规划。v6.9 必须从“优化某个 kernel”升级成“优化完整 live set”。

### 1.4 current attribution 仍不够

v6.8 P1 中 current DWM2 attribution pass 仍为 `0/18`，Nsight/counter 字段仍未提供可解释 peak gap 的可用数值。即使知道 full-step package 更差，也还不能精确判断是：

```text
Triton boundary copy
delta/dx overlap
coeffgrad partial buffer
update workspace overlap
transform temp
GEMM/mixing dominant
allocator padding/fragmentation
```

哪一个是 primary blocker。v6.9 必须补上这一环，否则继续写 kernel 仍然是盲打。

### 1.5 reset residual effect 已经过，但 memory/time 没过

v6.8 的 residual-effective reset 修复了 v6.7 中 residual 太弱的问题。`scale002/scale005` 的 residual/base 已经达到约 `0.02/0.05`，residual effect pass 为 18/18。但 memory/time 全部没有 near-pass：

```text
ResidualEffectiveTinyKAN-scale002:
  memory ratio mean ≈ 1.2981
  step ratio mean ≈ 1.9014

ResidualEffectiveTinyKAN-scale005:
  memory ratio mean ≈ 1.2981
  step ratio mean ≈ 1.9021
```

这说明 reset route 现在的关键不是 residual 强度，而是 bounded-workspace 实现没有真正 bounded。v6.9 的 reset 必须改变 workspace model，而不是只调 residual scale。

---

## 2. 是否在正确道路上

当前方法论仍然正确，因为它没有把 kernel failure 伪装成 task failure，也没有在 memory/time 未过时打开 optimizer 和 functional correction。项目已经越来越清楚地分离了三件事：

```text
1. graph-free gradient correctness
2. memory/time kernel feasibility
3. task/optimizer/functional utility
```

现在第 1 项基本成立，第 2 项未成立，第 3 项还不能正式评价。

更准确地说：

$$
\boxed{
\text{graph-free PureKAN 方向尚未被证伪，但 DWM2 当前 Python/Torch + partial Triton integration 路线已经不够。}
}
$$

v6.9 仍在正确道路上，但它必须升级问题层级。下一步不应继续：

```text
更多 Torch-level local reduce
更多 optimizer sweep
更多 functional correction smoke
更多 task seed
```

而应该做：

```text
full-step live-set attribution
Triton/PyTorch boundary overhead audit
single-call full-layer backward kernel
fused update-prep/live-set scheduling
residual-effective bounded primitive reset
```

如果这些仍失败，就应承认当前 DWM2-poly2 family 不适合继续做终极 primitive 主线。

---

## 3. 离目标还差多远

### 3.1 Memory 差距

以 current mean memory ratio `1.2918` 计算，要达到正式 gate `<1.0`，需要降低：

$$
1-\frac{1.0}{1.2918}\approx22.6\%.
$$

要达到终极目标 `0.8`，需要降低：

$$
1-\frac{0.8}{1.2918}\approx38.1\%.
$$

以 current best memory ratio `1.1441` 计算，到 `<1.0` 也需要降低：

$$
1-\frac{1.0}{1.1441}\approx12.6\%.
$$

到 `0.8` 需要降低：

$$
1-\frac{0.8}{1.1441}\approx30.1\%.
$$

因此 memory 不是“差一点点”，而是需要 full-step live-set 级别的实质变化。

### 3.2 Step time 差距

以 P3 current mean step ratio `2.1140` 计算，到 `1.35` 需要提速：

$$
1-\frac{1.35}{2.1140}\approx36.1\%.
$$

到强目标 `1.20` 需要提速：

$$
1-\frac{1.20}{2.1140}\approx43.2\%.
$$

以 P1 current mean step ratio `1.9698` 计算，到 `1.35` 需要提速：

$$
1-\frac{1.35}{1.9698}\approx31.5\%.
$$

因此 step time 比 memory 更难，不能只靠 memory buffer compression，需要减少 kernel boundary / materialization / phase overhead。

### 3.3 Reset residual 差距

v6.8 已经让 residual/base 达到 `0.02/0.05`，这项不再是最大问题。现在 reset 的差距是 memory/time：

```text
reset memory ratio mean ≈ 1.2981
reset step ratio mean ≈ 1.90
```

它和 current DWM2 差不多，说明 residual-effective reset 没有改变 memory model。

### 3.4 Task / optimizer / functional 差距

P5-P8 全部 gate，没有 task trace。不能说 task 失败，也不能说 optimizer 不行。准确说：

$$
\boxed{
\text{当前还没有资格系统探索 optimizer / functional correction。}
}
$$

现阶段主要离目标差在 **轻量化 primitive / full-step kernel**，不是 optimizer。

---

## 4. v6.9 核心假设

### H1：microkernel pass 没有转化为 full-step pass，是因为 Triton/PyTorch boundary 和 live-set overlap

v6.8 的 `K3-triton-fused-dx-coeffgrad` microkernel 很快，但 full package 更差。H1 假设：full-step 失败主要来自 boundary overhead 与 live tensor overlap，而不是 Triton kernel 数学本身。

H1 成立的证据包括：

```text
microkernel_time_ms 降低
full_step_time_ms 不降或升高
boundary_copy_time_ms 明显
triton_output_materialization_MB 明显
delta_dx_coeffgrad_update live overlap 明显
```

量化标准：

$$
T_{\text{boundary}} + T_{\text{materialization}}
\geq 0.30 T_{\text{full-step overhead}}.
$$

如果 H1 成立，下一步必须写 single-call full-layer backward，而不是继续替换单个 microkernel。

---

### H2：full-layer backward fusion 比 coeffgrad-only fusion 更关键

H2 假设：必须在同一个 layer-level kernel 或 tightly coupled kernel group 中完成：

```text
consume upstream delta
compute dz or consume dz
compute dx
accumulate residual coeff grad
prepare update temp
release or reuse layer buffers
```

用公式表示，目标是避免同时 materialize：

$$
\delta_z,\quad \delta_x,\quad C_{\theta},\quad U_{\text{update}}.
$$

其中 $C_{\theta}$ 是 per-sample coefficient contribution tensor。

H2 成立标准是 full-layer fused package 相对 current 满足：

$$
\frac{M_{\text{peak,fused-layer}}}{M_{\text{peak,current}}}\leq0.90,
$$

$$
\frac{T_{\text{step,fused-layer}}}{T_{\text{step,current}}}\leq0.90.
$$

如果只在 microkernel 中有收益，但 full package 没收益，则 H2 仍未完成。

---

### H3：current attribution 需要从 phase-level 升级为 live-set-level

v6.8 的 route 仍是 `R3-AttributionIncomplete`。H3 假设：只有记录 peak 时刻 live tensor set、allocation stack、kernel boundary、Triton/PyTorch tensor ownership，才能解释 current DWM2 的 peak gap。

定义：

$$
G_{\text{peak}}=M_{\text{peak,DWM2}}-M_{\text{peak,MLP}}.
$$

Attribution 成立要求：

$$
\frac{M_{\text{explained}}}{G_{\text{peak}}}\geq0.95.
$$

并且 top-3 live-set sources 解释：

$$
\frac{M_{\text{top3}}}{G_{\text{peak}}}\geq0.70.
$$

如果仍然做不到，v6.9 route 应保持 attribution incomplete，而不是继续 repair claim。

---

### H4：reset route 必须把 residual effect 和 bounded workspace 同时做成

v6.8 的 reset 已经让 residual effect 过 gate，但 memory/time 不过。H4 假设：真正的 reset 不是放大 residual，而是改变 residual 的 workspace model。

Reset candidate 必须同时满足：

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02,
$$

$$
\frac{M_{\text{backward}}}{M_{\text{MLP}}}\leq1.05,
$$

$$
\frac{T_{\text{step}}}{T_{\text{MLP}}}\leq1.35.
$$

如果 residual scale 达标后 memory/time 仍像 DWM2 一样失败，则说明 reset 没有改变 memory model。

---

### H5：optimizer exploration 仍应 gate

虽然 optimizer 探索还不够，但现在不是主线。H5 要求：

```text
No optimizer sweep unless memory/time S0/S1 exists.
No functional correction unless task re-entry passes.
No LightSmooth unless residual primitive passes task + memory.
```

否则 wall-clock / memory 结果没有意义。

---

## 5. v6.9 阶段总览

v6.9 分为十一个阶段：

```text
P0: v6.8 reproduction and contract check
P1: full-step live-set attribution
P2: Triton/PyTorch boundary overhead audit
P3: full-layer fused backward kernel
P4: one-buffer full-step DWM2 package
P5: residual-effective bounded reset v2
P6: one-step loss and residual probe
P7: limited task re-entry gate
P8: optimizer exploration remains gated
P9: functional correction remains gated
P10: route decision
P11: failure and artifact audit
```

P1-P5 是核心。P6 只在 P3/P4/P5 出现 near-pass 后运行。P7-P9 默认关闭。

---

## 6. P0：v6.8 reproduction and contract check

### 6.1 目的

确认 v6.9 与 v6.8 可比较，并确认新增 fused layer kernel 或 reset primitive 是真实实现。

### 6.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-current
DWM2-Triton-fused-dx-coeffgrad-v1
DWM2-Triton-full-backward-light-v1
DWM2-FullLayerBackward-v2, if implemented
DWM2-OneBufferFullStep-v1, if implemented
ResidualEffectiveTinyKAN-scale002
ResidualEffectiveTinyKAN-scale005
ResidualBoundedOneBuffer-v2, if implemented
```

### 6.3 必须记录字段

```text
variant
status
implementation_type
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
residual_param_count
mixing_param_count
rollback_max_error
grad_relerr_max
grad_cos_min
v68_memory_ratio_mean
v69_memory_ratio_mean
v68_step_ratio_mean
v69_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
```

### 6.4 通过标准

$$
|r_{\text{memory,v69}}-r_{\text{memory,v68}}|\leq0.05.
$$

$$
|r_{\text{step,v69}}-r_{\text{step,v68}}|\leq0.15.
$$

并且：

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0 for PureKAN candidates
rollback_max_error < 1e-8
```

### 6.5 可视化

```text
p0_reproduction_delta_bar.svg
p0_contract_heatmap.svg
p0_implementation_status.svg
```

---

## 7. P1：full-step live-set attribution

### 7.1 目的

P1 要解决 v6.8 的核心 blocker：current DWM2 attribution incomplete。P1 不只看 phase peak，而是记录 peak 时刻 live tensor set。

### 7.2 必须标记的 phase

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

### 7.3 必须记录字段

#### full-step fields

```text
variant
dataset
batch_size
depth
peak_allocated_MB
peak_reserved_MB
backward_memory_ratio_vs_mlp
step_time_ms
step_ratio_vs_mlp
backward_time_ms
backward_ratio_vs_mlp
grad_relerr_max
grad_cos_min
```

#### live-set fields

```text
peak_timestamp
peak_phase
live_tensor_count
live_tensor_total_MB
top1_live_tensor_name
top1_live_tensor_MB
top1_live_tensor_source
top1_lifetime_start_phase
top1_lifetime_end_phase
...
top20_live_tensor_name
top20_live_tensor_MB
top20_live_tensor_source
top20_lifetime_start_phase
top20_lifetime_end_phase
```

#### boundary fields

```text
triton_input_materialization_MB
triton_output_materialization_MB
torch_to_triton_boundary_time_ms
triton_to_torch_boundary_time_ms
contiguous_copy_MB
dtype_cast_MB
layout_conversion_MB
```

#### attribution fields

```text
peak_gap_MB
explained_gap_MB
unexplained_gap_MB
explain_ratio
top3_gap_fraction
coeffgrad_live_MB
dx_live_MB
dz_live_MB
update_temp_live_MB
forward_transform_live_MB
mixing_live_MB
optimizer_state_live_MB
allocator_padding_MB
fragmentation_ratio
```

### 7.4 判断标准

P1 attribution pass：

$$
\frac{M_{\text{explained}}}{G_{\text{peak}}}\geq0.95.
$$

$$
\frac{M_{\text{top3}}}{G_{\text{peak}}}\geq0.70.
$$

Boundary bottleneck 判定：

$$
T_{\text{boundary}} \geq 0.20 T_{\text{step overhead}},
$$

或：

$$
M_{\text{boundary materialization}} \geq 0.20 G_{\text{peak}}.
$$

Live-set overlap bottleneck 判定：

$$
M_{\delta z}+M_{\delta x}+M_{C_{\theta}}+M_{update}
\geq0.50G_{\text{peak}}.
$$

### 7.5 可视化

```text
p1_live_set_gantt.svg
p1_peak_live_tensor_top20.svg
p1_gap_attribution_stacked_bar.svg
p1_boundary_overhead_bar.svg
p1_live_set_overlap_heatmap.svg
p1_peak_phase_timeline.svg
```

---

## 8. P2：Triton/PyTorch boundary overhead audit

### 8.1 目的

v6.8 的 K3 microkernel 快，但 full package 慢。P2 专门测边界开销。

### 8.2 必跑对象

```text
K0-current torch-only coeffgrad
K3-triton-fused-dx-coeffgrad microkernel
K3-wrapper-only no-op
K3-input-contiguous-only
K3-output-materialize-only
K3-layout-conversion-only
K3-call-overhead-only
K3-batched-layer-call
K3-single-call-multi-layer, if implemented
```

### 8.3 必须记录

```text
kernel_name
call_count_per_step
time_per_call_ms
total_call_time_ms
input_contiguous_time_ms
output_materialize_time_ms
layout_conversion_time_ms
wrapper_overhead_ms
cuda_launch_overhead_ms
peak_allocated_MB
temp_allocated_MB
input_copy_MB
output_copy_MB
grad_relerr_max
```

### 8.4 判断标准

Boundary overhead 是 blocker 如果：

$$
T_{\text{wrapper+copy+layout}} \geq 0.30 T_{\text{triton full package overhead}}.
$$

如果 K3 microkernel 快但 wrapper/copy 占比高，则 P3 必须做 single-call full-layer kernel。

### 8.5 可视化

```text
p2_boundary_time_waterfall.svg
p2_call_count_vs_step_time.svg
p2_materialization_memory_bar.svg
p2_micro_vs_wrapped_runtime.svg
```

---

## 9. P3：full-layer fused backward kernel

### 9.1 目的

P3 是 v6.9 的第一主修复阶段。它验证 single layer backward fusion 是否能把 microkernel benefit 转化为 full-step benefit。

### 9.2 Candidate packages

```text
L0-current
L1-fused-dx-coeffgrad-only
L2-full-layer-backward:
  consume upstream delta, x, params
  produce dx, residual coeff grad

L3-full-layer-backward-no-dx-materialize:
  stream dx to previous layer buffer

L4-full-layer-backward-update-prep:
  also prepare update temp in same kernel

L5-multi-layer-chunked-backward:
  process multiple layers in a chunk, reduce boundary calls

L6-single-call-depth2-backward:
  diagnostic for depth=2, fuse both layers if feasible
```

### 9.3 必须记录

```text
package
implementation_status
layers_fused
calls_per_step
memory_ratio_min
memory_ratio_mean
memory_ratio_max
step_ratio_min
step_ratio_mean
step_ratio_max
backward_ratio_mean
memory_improvement_vs_current
step_improvement_vs_current
boundary_time_reduction
live_set_overlap_reduction
kernel_count_reduction
allocation_count_reduction
grad_relerr_max
grad_cos_min
```

### 9.4 判断标准

Full-layer fusion pass：

$$
\frac{M_{\text{peak,L}}}{M_{\text{peak,current}}}\leq0.90.
$$

$$
\frac{T_{\text{step,L}}}{T_{\text{step,current}}}\leq0.90.
$$

Near-pass:

$$
\frac{M_{\text{backward,L}}}{M_{\text{MLP}}}\leq1.05.
$$

$$
\frac{T_{\text{step,L}}}{T_{\text{MLP}}}\leq1.50.
$$

Task re-entry pass:

$$
\frac{M_{\text{backward,L}}}{M_{\text{MLP}}}<1.0.
$$

$$
\frac{T_{\text{step,L}}}{T_{\text{MLP}}}\leq1.35.
$$

### 9.5 可视化

```text
p3_full_layer_package_pareto.svg
p3_boundary_reduction_bar.svg
p3_live_set_reduction_bar.svg
p3_kernel_count_reduction.svg
p3_batch_depth_stability_heatmap.svg
```

---

## 10. P4：one-buffer full-step DWM2 package

### 10.1 目的

如果 P3 single layer fusion 有收益但 still no near-pass，P4 测试 full-step one-buffer scheduling。核心是全 step 只保留有限个 live buffers。

### 10.2 Candidate packages

```text
O0-current
O1-onebuffer-forward-transform
O2-onebuffer-backward-delta
O3-onebuffer-update-prep
O4-onebuffer-forward-backward
O5-onebuffer-full-step
O6-onebuffer-full-step-chunked-mix
```

### 10.3 必须记录

```text
package
live_buffer_count_peak
live_buffer_total_MB_peak
buffer_reuse_count
buffer_lifetime_conflict_count
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
memory_improvement_vs_current
step_improvement_vs_current
grad_relerr_max
grad_cos_min
```

### 10.4 判断标准

One-buffer package 有效标准：

$$
\operatorname{live\_buffer\_count\_peak}\leq3.
$$

$$
\frac{M_{\text{peak,onebuffer}}}{M_{\text{peak,current}}}\leq0.85.
$$

$$
\frac{T_{\text{step,onebuffer}}}{T_{\text{step,current}}}\leq1.00.
$$

如果 memory 降但 step time 上升超过 $20\%$，只能作为 diagnostic。

### 10.5 可视化

```text
p4_buffer_lifetime_diagram.svg
p4_live_buffer_count_timeline.svg
p4_onebuffer_memory_step_pareto.svg
p4_buffer_conflict_table.md
```

---

## 11. P5：residual-effective bounded reset v2

### 11.1 目的

P5 是 reset 分支。v6.8 已证明 residual strength 可以做到 `0.02/0.05`，但 memory/time 不过。v6.9 要测试真正改变 memory model 的 reset，不再只放大 residual。

### 11.2 Candidate primitives

```text
R0-ManualLinear-reference

R1-ResidualEffectiveTinyKAN-scale002-current

R2-ResidualEffectiveTinyKAN-scale005-current

R3-BoundedResidual-inplace-transform:
  residual applied in-place to activation buffer

R4-BoundedResidual-fused-mix:
  compute residual transform and mixing without separate residual materialization

R5-BoundedResidual-chunked-mix:
  output mixing processed in chunks

R6-BoundedResidual-single-buffer-backward:
  one live residual buffer during backward

R7-BoundedResidual-lowrank-mix:
  reduce mixing workspace through low-rank edge-owned factorization

R8-BoundedResidual-piecewise2:
  local piecewise residual with no dense basis tensor
```

### 11.3 必须记录

```text
primitive
residual_scale_target
residual_over_base
residual_effect_pass
nonKAN_param_count
edge_param_count
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
memory_improvement_vs_current
step_improvement_vs_current
grad_relerr_max
grad_cos_min
residual_ablation_delta_loss
residual_ablation_delta_logit
live_buffer_count_peak
workspace_temp_MB
```

### 11.4 判断标准

Reset near-pass：

$$
\frac{M_{\text{backward}}}{M_{\text{MLP}}}\leq1.05.
$$

$$
\frac{T_{\text{step}}}{T_{\text{MLP}}}\leq1.35.
$$

Residual effect pass：

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

### 11.5 可视化

```text
p5_reset_residual_strength_vs_memory.svg
p5_reset_memory_time_pareto.svg
p5_residual_ablation_delta.svg
p5_reset_workspace_model_comparison.svg
```

---

## 12. P6：one-step loss and residual probe

P6 只在 P3/P4/P5 有 near-pass 后运行。

### 12.1 方法

```text
clone weights
compute manual gradient
apply one update
measure train / holdout / val loss before and after
measure residual ablation before and after
rollback weights
check rollback exactness
```

### 12.2 必须记录

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
residual_ablation_delta_loss
residual_ablation_delta_logit
```

### 12.3 判断标准

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

### 12.4 可视化

```text
p6_loss_before_after.svg
p6_bad_step_heatmap.svg
p6_update_norm_vs_loss_delta.svg
p6_residual_ablation_probe.svg
```

---

## 13. P7：limited task re-entry gate

### 13.1 打开条件

Open limited task if:

```text
survivor_type in {S0, S1}
P6 pass
memory_ratio < 1.0
step_ratio <= 1.35
grad pass
no fake/proxy
```

Open diagnostic task if:

```text
survivor_type == S2
P6 pass
memory_ratio <= 1.05
step_ratio <= 1.50
```

Otherwise P7 remains `not_run`.

### 13.2 如果打开 task

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
  Best-DWM2-fused-live-set-package + ManualAdanLite
  Best-reset-primitive + ManualAdanLite
```

### 13.3 必须记录

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
seedwise_failure_reason
```

### 13.4 判断标准

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
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}<1.0.
$$

Wall-clock must satisfy on at least two datasets:

$$
\operatorname{ValLossAUC}_{time,KAN}
\leq
\operatorname{ValLossAUC}_{time,MLP}.
$$

---

## 14. P8：optimizer exploration remains gated

P8 只有 P7 pass 后才打开。当前不是主线。

如果打开，只比较：

```text
ManualAdamW
ManualAdanLite
ManualWinLite
Lookahead-ManualAdamW
WarmupCosine-ManualAdamW
```

不做大范围 role-wise LR sweep。

### 判断标准

$$
T_{\text{target,optimizer}}
\leq0.90T_{\text{target,ManualAdamW}}
$$

or:

$$
\operatorname{ValLossAUC}_{time,optimizer}
<
\operatorname{ValLossAUC}_{time,ManualAdamW}.
$$

同时：

$$
\operatorname{Acc}_{optimizer}
\geq
\operatorname{Acc}_{ManualAdamW}-0.005.
$$

---

## 15. P9：functional correction remains gated

P9 只有 P7/P8 pass 后才打开。默认 `not_run`。

如果打开，只做 smoke：

```text
task-only best
task + small residual geometry correction
task + low-frequency LightSmooth event
```

判断标准：

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

## 16. P10：route decision

### Route cases

```text
R1-DWM2LiveSetSolved:
  Full-step live-set fused DWM2 gets S0/S1 and P6 pass.
  Open task.

R2-DWM2LiveSetNearPass:
  DWM2 gets S2 with clear memory/time improvement.
  Continue kernel engineering.

R3-AttributionIncomplete:
  live-set attribution still cannot explain peak.
  Improve profiler or switch tools.

R4-DWM2FullFusionNoEffect:
  full-layer/onebuffer fusion improves <5%.
  DWM2 route likely exhausted.

R5-ResetPrimitiveCandidate:
  reset primitive near-pass and residual effect pass.
  Switch next cycle to reset primitive.

R6-ResetStillTooHeavy:
  reset residual effect pass but memory/time fail.
  Need new bounded primitive.

R7-TerminalPrimitiveRedesign:
  no DWM2 near-pass and no reset near-pass.
  Stop DWM2 optimization; design new primitive family.

R8-TaskReentryPass:
  memory/time survivor passes limited task.
  Next cycle can consider confirm.

R9-TaskReentryFail:
  memory/time survivor exists but task fails.
  Need expressivity/optimizer repair.
```

### 必须输出 JSON 字段

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
attribution_pass
boundary_bottleneck_pass
live_set_overlap_pass
reset_residual_effect_pass
open_task_reentry
open_optimizer_exploration
open_functional_correction
primary_blocker
next_required_implementation
no_fake
no_proxy
```

---

## 17. P11：failure and artifact audit

### failure types

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_attribution_incomplete
F5_boundary_overhead
F6_live_set_overlap
F7_full_fusion_no_effect
F8_reset_residual_effect_fail
F9_reset_memory_fail
F10_task_fail
F11_optimizer_gated
F12_functional_gated
F13_fake_or_proxy_violation
F14_artifact_missing
```

### artifacts

必须生成：

```text
p0_contract.csv
p0_reproduction_check.csv
p1_live_set_attribution.csv
p1_boundary_overhead.csv
p2_boundary_audit.csv
p3_full_layer_fusion.csv
p4_onebuffer_full_step.csv
p5_residual_bounded_reset.csv
p6_one_step_probe.csv
p7_task_reentry.csv
p7_task_trace.csv
p8_optimizer_exploration.csv
p9_functional_correction_smoke.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### figures

必须生成：

```text
figures/p1_live_set_gantt.svg
figures/p1_peak_live_tensor_top20.svg
figures/p1_gap_attribution_stacked_bar.svg
figures/p1_boundary_overhead_bar.svg
figures/p2_micro_vs_wrapped_runtime.svg
figures/p3_full_layer_package_pareto.svg
figures/p4_onebuffer_memory_step_pareto.svg
figures/p5_reset_residual_strength_vs_memory.svg
figures/p6_loss_before_after.svg
figures/p10_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 18. 成功与失败解释规则

### Case A：full-layer DWM2 pass

如果：

$$
M_{\text{backward}}/M_{\text{MLP}}<1.0
$$

且：

$$
T_{\text{step}}/T_{\text{MLP}}\leq1.35,
$$

则 DWM2 继续作为主线，允许 task re-entry。

### Case B：full-layer DWM2 near-pass

如果：

$$
M_{\text{backward}}/M_{\text{MLP}}\leq1.05
$$

且：

$$
T_{\text{step}}/T_{\text{MLP}}\leq1.50,
$$

且 memory improvement $\geq10\%$，则继续 kernel engineering，但不做 confirm。

### Case C：microkernel 有效但 full layer 无效

说明 bottleneck 不是局部 kernel，而是 live-set / boundary / update overlap。不能继续只优化 microkernel。

### Case D：reset residual 有效但 memory/time fail

说明 reset expressivity 可以，但 bounded workspace 没做成。下一步重新设计 primitive memory model。

### Case E：DWM2 和 reset 都失败

进入 primitive family redesign，停止当前 DWM2-poly2 小修。

---

## 19. 最终建议

v6.9 的一句话策略是：

$$
\boxed{
\text{把目标从局部 Triton kernel 加速，升级为 full-step live-set 重构。}
}
$$

v6.8 已经证明，局部 `dx+coeffgrad` microkernel 可以很快，但 full-step 不快也不省内存。这意味着下一步要么把整个 layer backward 变成单一 live-set-aware kernel，要么换一个真正 bounded-workspace 的 primitive。

本轮仍然不应该打开 optimizer / functional correction。只有出现 S0/S1 memory/time survivor 后，optimizer 和 functional 才重新成为有意义的问题。
