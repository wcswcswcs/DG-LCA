# DG-KAN v6.10：Peak Live-Set Attribution、Single-Call FullLayer Kernel 与 DWM2 Stop/Go 决策详细实验计划

> 本计划基于 v6.9 final real-only run 制定。v6.9 已经证明：`DWM2-current` baseline 可复现、manual gradient correctness 仍然可靠；但 current DWM2 的 memory/time gate 继续失败，exact peak-time live tensor attribution 仍未取得；`K3-triton-fused-dx-coeffgrad` 这类局部 microkernel 有速度信号，但 measured full-layer packages `L1/L2` 的 memory/time 均比 current 更差；one-buffer diagnostics 没有实现真正 one-buffer，reset primitive 虽然 residual effect 过 gate，但 memory/time 没有 near-pass。因此 v6.10 的目标不是继续堆局部 kernel 或 optimizer，而是完成 **peak live-set attribution 闭环**、验证 **single-call full-layer / full-step kernel 是否能真正改变 live set**，并对 DWM2-poly2 做一次明确的 stop/go 决策。

---

## 0. 实验整体目标

v6.10 的整体目标是回答三个问题。

第一个问题：

$$
\boxed{
\text{current DWM2 的 peak memory gap 是否能被 exact peak-time live tensor attribution 解释？}
}
$$

第二个问题：

$$
\boxed{
\text{single-call full-layer / full-step kernel 是否能把局部 Triton microkernel 收益转化为 full-step memory/time 收益？}
}
$$

第三个问题：

$$
\boxed{
\text{如果 DWM2 仍不能 near-pass，是否应冻结 DWM2-poly2 主线并切换到新的 bounded-workspace primitive family？}
}
$$

v6.9 的真实结果显示：

```text
DWM2-current:
  memory ratio mean = 1.2918
  step ratio mean   = 1.9041 in P1
  backward ratio mean = 1.4447 in P1
  grad relerr max ≈ 9e-08
  attribution pass = 0/18

P3 L0-current:
  memory ratio mean = 1.2918
  step ratio mean   = 1.8776
  backward ratio mean = 1.4261

P3 measured full-layer fusion:
  L1-fused-dx-coeffgrad-only:
    memory ratio mean = 1.3472
    step ratio mean = 2.3699
  L2-full-layer-backward:
    memory ratio mean = 1.3472
    step ratio mean = 2.3577

P4 one-buffer diagnostics:
  O2-onebuffer-backward-delta:
    memory ratio mean = 1.2918
    step ratio mean = 1.9489
  O1/O4 buffer-reuse variants:
    memory worse than current
  true O5/O6 one-buffer full-step:
    not_implemented

P5 reset:
  residual effect pass = true for residual scale 0.02/0.05
  memory ratio mean ≈ 1.2981
  step ratio mean ≈ 1.80
  reset near pass = 0
```

因此 v6.10 的最低工程成功标准是：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}\leq1.05,
$$

$$
\frac{T_{\text{step,KAN}}}{T_{\text{step,MLP}}}\leq1.50,
$$

并且：

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

正式 task re-entry 标准仍然是：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}<1.00,
$$

$$
\frac{T_{\text{step,KAN}}}{T_{\text{step,MLP}}\leq1.35}.
$$

上式为了避免歧义，本计划后文统一写成：

$$
r_{\text{mem}}<1.00,\quad r_{\text{step}}\leq1.35.
$$

强目标是：

$$
r_{\text{mem}}\leq0.90,\quad r_{\text{step}}\leq1.20.
$$

最终终极目标仍然是：

$$
r_{\text{mem}}\leq0.80.
$$

但 v6.10 不把 $0.80$ 作为唯一成功标准。v6.10 的现实目标是：判断 DWM2-poly2 是否还值得继续，还是应该转向新的 primitive family。

---

## 1. 当前数据分析与判断

### 1.1 当前实验进展

v6.9 的最大价值是把“局部 kernel 有用”和“full-step 失败”拆开了。`K3-triton-fused-dx-coeffgrad` 在边界 audit / microkernel 层面有真实可执行信号，且梯度正确；但是 `L1/L2` full-layer package 的 memory/time 比 current 更差。这说明 DWM2 的主要问题不是某个小公式算得慢，而是完整训练步中的 live set、kernel boundary、materialization、layout conversion、update workspace 与 allocator lifetime 没有被一起重构。

v6.9 还说明 reset 分支已经不再是“residual 太弱”的问题。`ResidualEffectiveTinyKAN` 的 residual effect 过了，但 memory/time 没有 near-pass。也就是说，如果要走 reset，必须改变 bounded workspace 模型，而不是继续调 residual scale。

### 1.2 当前问题所在

当前主要 blocker 有四个。

第一，exact peak-time live tensor attribution 没有取得。P1 虽然生成了 phase diagnostic，但没有得到真实 peak timestamp / allocation stack / top live tensor set，因此 `attribution_pass = 0/18`。这使得我们无法知道 full-step peak 到底由哪些 tensor / op / boundary 造成。

第二，局部 Triton microkernel 没有传递到 full-step。`K3` 快，但 `L1/L2` 更慢、更费 memory。这强烈暗示 Triton/PyTorch boundary、layout/contiguous copy、output materialization 或 live-set overlap 抵消了 microkernel 收益。

第三，one-buffer 仍未真正实现。`O1/O4` 的 live buffer count peak 是 31，远不是 one-buffer；`O5/O6 true one-buffer full-step` 仍是 `not_implemented`。因此不能说 one-buffer 路线失败，只能说当前 diagnostics 没有实现真正目标。

第四，reset primitive 仍没有改变 memory model。residual effect 已经达标，但 `memory ratio mean ≈ 1.2981`、`step ratio mean ≈ 1.80`，说明 reset 的 workspace 结构仍像 DWM2 一样重。

### 1.3 是否在正确道路上

实验方法论仍然在正确道路上，因为：

```text
no-fake / no-proxy
gradient correctness 继续保持
没有把 microkernel pass 夸大成 full-step pass
没有把 reset residual effect pass 夸大成 task-ready
task / optimizer / functional correction 被正确 gate
```

但是 DWM2 的当前修复层级已经不够。下一步如果还只是写一个局部 Triton kernel、局部 reduction、局部 delta streaming，大概率继续失败。正确方向必须提升为：

```text
exact peak live-set attribution
single-call full-layer backward
single-call full-step scheduling
true one-buffer live-set constraint
bounded primitive family branch decision
```

### 1.4 离目标还差多远

以 v6.9 current P3 mean 计算：

$$
r_{\text{mem,current}} = 1.2918.
$$

要到 formal gate $r_{\text{mem}}<1.0$，需要降低：

$$
1-\frac{1.0}{1.2918}\approx22.6\%.
$$

要到终极 $0.8$，需要降低：

$$
1-\frac{0.8}{1.2918}\approx38.1\%.
$$

以 v6.9 P3 current step mean 计算：

$$
r_{\text{step,current}}=1.8776.
$$

要到 task re-entry gate $1.35$，需要提速：

$$
1-\frac{1.35}{1.8776}\approx28.1\%.
$$

要到强目标 $1.20$，需要提速：

$$
1-\frac{1.20}{1.8776}\approx36.1\%.
$$

所以目前不是“差一点”。要进入下一阶段，必须出现 full-step 级别的实质改变。

---

## 2. v6.10 的禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。所有未实现内容必须写为：

```text
not_implemented
not_run
metric_unavailable
not_applicable
```

第二，P1 exact live-set attribution 未通过时，不允许宣布 repair 的根因解释成立。

第三，不允许把局部 microkernel pass 当成 full-step pass。只有 full-step memory/time near-pass 才能打开 P6/P7。

第四，不允许继续把 Torch-level wrapper、chunk loop 或局部 op 替换命名为 full-layer kernel。`single-call full-layer kernel` 必须明显减少 call count、boundary time 或 live-set overlap。

第五，不允许在 memory/time survivor 之前打开 optimizer exploration、LightSmooth 或 functional correction。

第六，不允许把 residual effect pass 当成 reset success。reset 必须同时满足 residual effect 与 memory/time near-pass。

第七，不允许把 manual-linear reference 当成 PureKAN success。它只能作为 lower-bound reference。

---

## 3. 核心假设

### H1：current attribution incomplete 是首要 blocker

v6.9 的 route 是 `R3-AttributionIncomplete`。H1 假设：没有 exact peak-time live tensor set，就无法有效决定下一步 kernel 应该融合什么。必须先获得真实 attribution。

定义：

$$
G_{\text{peak}} = M_{\text{peak,DWM2}} - M_{\text{peak,MLP}}.
$$

如果 attribution source 为 $A_i$，则要求：

$$
\frac{\sum_i A_i}{G_{\text{peak}}}\geq0.95.
$$

并且 top-3 source 解释至少：

$$
\frac{A_1+A_2+A_3}{G_{\text{peak}}}\geq0.70.
$$

H1 成立标准：

```text
exact_peak_timestamp_available = 1
top20_live_tensor_available = 1
allocation_stack_available = 1
explain_ratio >= 0.95
top3_gap_fraction >= 0.70
```

如果 H1 不成立，v6.10 不允许继续 claim DWM2 repair success。

---

### H2：microkernel pass 未传递到 full-step，是因为 boundary/materialization/live-set overlap

H2 假设：`K3` 快但 `L1/L2` 慢，是因为 full-step 中出现了以下开销：

```text
torch_to_triton_boundary_time
triton_to_torch_boundary_time
contiguous copy
layout conversion
Triton output materialization
delta/dx/update temp overlap
per-layer kernel call overhead
```

H2 成立的条件之一是：

$$
T_{\text{boundary+materialization}}\geq0.30T_{\text{overhead,full-package}}.
$$

或者：

$$
M_{\text{boundary materialization}}\geq0.20G_{\text{peak}}.
$$

如果 H2 成立，下一步必须做 single-call full-layer 或 multi-layer chunked kernel。

---

### H3：真正的 full-layer kernel 必须减少 live tensors，而不是只替换 coeffgrad

H3 假设：DWM2 backward 必须在同一个 layer-level fused kernel 或紧密 kernel group 中完成：

```text
consume upstream delta
compute dz or consume dz
compute dx
accumulate coeffgrad
prepare update temp
release or reuse layer buffers
```

目标是避免以下对象同时 materialize：

$$
\delta_z,\quad \delta_x,\quad C_{\theta},\quad U_{\text{update}}.
$$

H3 成立标准：

$$
\frac{M_{\text{peak,fused-layer}}}{M_{\text{peak,current}}}\leq0.90,
$$

$$
\frac{T_{\text{step,fused-layer}}}{T_{\text{step,current}}}\leq0.90.
$$

同时要求：

$$
\operatorname{grad\_relerr}<10^{-4}.
$$

---

### H4：true one-buffer 必须把 peak live buffer count 压到 3 以下

v6.9 的 `O1/O4` live buffer count peak 仍为 31，说明它们不是 one-buffer。H4 假设：只有真正限制 live buffer count，memory 才可能显著下降。

H4 成立标准：

$$
\operatorname{live\_buffer\_count\_peak}\leq3.
$$

并且：

$$
\frac{M_{\text{peak,onebuffer}}}{M_{\text{peak,current}}}\leq0.85.
$$

如果 memory 降但 step time 增加超过 $20\%$，该方案只能作为 diagnostic，不能进入 task。

---

### H5：reset primitive 需要改变 workspace model，而不是继续调 residual scale

v6.9 的 reset 已经证明 residual scale 不是 blocker。H5 假设：新的 reset primitive 必须用 one live residual buffer、chunked mix、low-rank mix 或 sparse local residual 改变 workspace model。

H5 成立标准：

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02,
$$

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.35.
$$

若 residual effect pass 但 memory/time fail，则 reset 不能进入 task。

---

### H6：如果 v6.10 仍无 near-pass，则应冻结 DWM2-poly2 主线

H6 是路线决策假设。如果满足：

```text
exact attribution pass
full-layer fusion tested
one-buffer tested or impossible
reset tested
no S0/S1/S2 survivor
```

则不应继续 DWM2-poly2 小修。应进入 new primitive family 或 full custom KAN layer。

---

## 4. 实验阶段总览

v6.10 分为十二个阶段：

```text
P0: v6.9 reproduction and contract check
P1: exact peak live-set attribution
P2: boundary and materialization audit
P3: single-call full-layer backward kernel
P4: multi-layer chunked backward and call-count reduction
P5: true one-buffer full-step DWM2
P6: bounded primitive reset v3
P7: one-step numerical and residual probe
P8: limited task re-entry gate
P9: optimizer exploration remains gated
P10: functional correction remains gated
P11: route decision
P12: artifact and failure audit
```

P1-P6 是核心。P7 只有 near-pass candidate 才运行。P8-P10 默认关闭。

---

## 5. P0：v6.9 reproduction and contract check

### 5.1 目的

确认 v6.10 与 v6.9 baseline 可比，防止 measurement drift。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-current
DWM2-L1-fused-dx-coeffgrad-only
DWM2-L2-full-layer-backward
DWM2-L3-full-layer-backward-no-dx-materialize, if implemented
DWM2-L4-full-layer-backward-update-prep, if implemented
DWM2-L5-multi-layer-chunked-backward, if implemented
DWM2-O5-onebuffer-full-step, if implemented
Reset-v3-bounded-residual, if implemented
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
v69_memory_ratio_mean
v610_memory_ratio_mean
v69_step_ratio_mean
v610_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
```

### 5.4 判断标准

$$
|r_{\text{mem,v610}}-r_{\text{mem,v69}}|\leq0.05.
$$

$$
|r_{\text{step,v610}}-r_{\text{step,v69}}|\leq0.15.
$$

并且：

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0
rollback_max_error < 1e-8
```

### 5.5 可视化

```text
p0_reproduction_delta_bar.svg
p0_contract_heatmap.svg
p0_implementation_status.svg
```

---

## 6. P1：exact peak live-set attribution

### 6.1 目的

P1 是本轮第一硬门禁。它必须获得真实 peak-time live tensors，而不是 phase diagnostic。

### 6.2 必须实现的 instrumentation

至少实现两类：

```text
torch.cuda.memory._record_memory_history / memory_snapshot
NVTX ranges
custom tensor registry for manual buffers
allocation stack hashing
peak timestamp capture
```

如果可能，加入：

```text
Nsight Systems
Nsight Compute
CUPTI allocation callback
```

### 6.3 Phase 标记

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
peak_reserved_MB
MLP_peak_allocated_MB
peak_gap_MB
exact_live_tensor_count
exact_live_tensor_total_MB
top1_live_tensor_name
top1_live_tensor_MB
top1_source_op
top1_allocation_stack_hash
top1_lifetime_start_phase
top1_lifetime_end_phase
...
top20_live_tensor_name
top20_live_tensor_MB
top20_source_op
top20_allocation_stack_hash
top20_lifetime_start_phase
top20_lifetime_end_phase
explained_gap_MB
unexplained_gap_MB
explain_ratio
top3_gap_fraction
```

### 6.5 必须分解的 source class

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

### 6.6 判断标准

Attribution pass:

$$
\frac{M_{\text{explained}}}{G_{\text{peak}}}\geq0.95.
$$

$$
\frac{M_{\text{top3}}}{G_{\text{peak}}}\geq0.70.
$$

Unknown source 要满足：

$$
\frac{M_{\text{unknown}}}{G_{\text{peak}}}\leq0.05.
$$

如果 P1 不过，后续 P3-P6 仍可做 diagnostic，但不得声明根因修复完成。

### 6.7 可视化

```text
p1_exact_live_set_gantt.svg
p1_peak_live_tensor_top20.svg
p1_gap_attribution_stacked_bar.svg
p1_unknown_gap_heatmap.svg
p1_lifetime_overlap_matrix.svg
```

---

## 7. P2：boundary and materialization audit

### 7.1 目的

P2 验证 `K3 microkernel pass -> full-step fail` 的边界开销假设。

### 7.2 必跑对象

```text
K3-microkernel-raw
K3-wrapper-only-noop
K3-input-contiguous-only
K3-output-materialize-only
K3-layout-conversion-only
K3-call-overhead-only
K3-batched-layer-call
K3-single-call-multi-layer, if implemented
```

### 7.3 必须记录

```text
kernel_name
call_count_per_step
time_per_call_ms
total_call_time_ms
wrapper_overhead_ms
layout_conversion_time_ms
contiguous_copy_time_ms
output_materialize_time_ms
input_materialize_MB
output_materialize_MB
layout_conversion_MB
temp_allocated_MB
peak_allocated_MB
grad_relerr_max
```

### 7.4 判断标准

Boundary bottleneck 成立：

$$
T_{\text{wrapper+copy+layout}}\geq0.30T_{\text{full-package overhead}}.
$$

或：

$$
M_{\text{boundary materialization}}\geq0.20G_{\text{peak}}.
$$

如果成立，P3 必须优先 single-call full-layer kernel。

### 7.5 可视化

```text
p2_boundary_time_waterfall.svg
p2_materialization_memory_bar.svg
p2_call_count_vs_step_time.svg
p2_micro_vs_full_gap.svg
```

---

## 8. P3：single-call full-layer backward kernel

### 8.1 目的

P3 真正测试 full-layer kernel，而不是 microkernel 替换。目标是把 backward layer 的 live set 压缩。

### 8.2 Candidate packages

```text
L0-current

L3-full-layer-backward-no-dx-materialize:
  do not materialize dx as standalone tensor if possible

L4-full-layer-backward-update-prep:
  compute dx + coeffgrad + update prep in one kernel group

L4b-full-layer-backward-update-prep-no-partial:
  no coeffgrad partial tensor

L4c-full-layer-backward-update-prep-inplace-buffer:
  write into reusable buffer

L5-multi-layer-chunked-backward:
  fuse multiple layer backward calls into chunks

L6-single-call-depth2-backward:
  diagnostic for depth=2
```

### 8.3 必须记录

```text
package
calls_per_step
layers_fused
dx_materialized
coeffgrad_partial_materialized
update_temp_materialized
memory_ratio_min
memory_ratio_mean
memory_ratio_max
step_ratio_min
step_ratio_mean
step_ratio_max
backward_ratio_mean
boundary_time_reduction
live_set_overlap_reduction
kernel_count_reduction
allocation_count_reduction
grad_relerr_max
grad_cos_min
```

### 8.4 判断标准

Full-layer fusion useful：

$$
\frac{M_{\text{peak,L}}}{M_{\text{peak,current}}}\leq0.90.
$$

$$
\frac{T_{\text{step,L}}}{T_{\text{step,current}}}\leq0.90.
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

### 8.5 可视化

```text
p3_full_layer_package_pareto.svg
p3_live_set_reduction_bar.svg
p3_boundary_reduction_bar.svg
p3_kernel_count_reduction.svg
p3_batch_depth_stability_heatmap.svg
```

---

## 9. P4：multi-layer chunked backward and call-count reduction

### 9.1 目的

如果 P3 减少单层 live set 但 step time 仍高，P4 测试 call-count / launch overhead 是否可以通过多层 chunk 降低。

### 9.2 Candidate packages

```text
M0-current
M1-depth2-single-call
M2-depth4-two-chunks
M3-depth4-single-call, if feasible
M4-chunked-backward-no-update
M5-chunked-backward-with-update-prep
```

### 9.3 必须记录

```text
package
depth
chunk_size
calls_per_step
call_count_reduction
launch_overhead_ms
step_time_ms
step_ratio
memory_ratio
backward_ratio
grad_relerr_max
chunk_boundary_materialization_MB
```

### 9.4 判断标准

Chunking useful:

$$
\frac{\text{calls}_{new}}{\text{calls}_{current}}\leq0.50.
$$

$$
\frac{T_{\text{step,new}}}{T_{\text{step,current}}}\leq0.85.
$$

Memory must not worsen beyond:

$$
\frac{M_{\text{new}}}{M_{\text{current}}}\leq1.05.
$$

### 9.5 可视化

```text
p4_call_count_reduction_bar.svg
p4_chunk_size_vs_step_time.svg
p4_chunk_boundary_memory_bar.svg
```

---

## 10. P5：true one-buffer full-step DWM2

### 10.1 目的

P5 是 DWM2 最后一次 memory model 测试。它要求实现真正 one-buffer，而不是 v6.9 中 live buffer count 仍为 31 的 diagnostic。

### 10.2 Candidate packages

```text
O0-current
O5-onebuffer-full-step
O6-onebuffer-full-step-chunked-mix
O7-onebuffer-full-step-no-dx-materialize
O8-onebuffer-full-step-update-prep
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

True one-buffer implementation pass:

$$
\operatorname{live\_buffer\_count\_peak}\leq3.
$$

One-buffer useful:

$$
\frac{M_{\text{peak,onebuffer}}}{M_{\text{peak,current}}}\leq0.85.
$$

$$
\frac{T_{\text{step,onebuffer}}}{T_{\text{step,current}}}\leq1.00.
$$

If memory decreases but step time increases by more than $20\%$, it remains diagnostic only.

### 10.5 可视化

```text
p5_buffer_lifetime_diagram.svg
p5_live_buffer_count_timeline.svg
p5_onebuffer_memory_step_pareto.svg
p5_buffer_conflict_table.md
```

---

## 11. P6：bounded primitive reset v3

### 11.1 目的

If DWM2 remains blocked, P6 tests a new primitive family. Unlike v6.9 reset, the candidate must change workspace model, not just residual scale.

### 11.2 Candidate primitives

```text
R0-ManualLinear-reference

R1-ResidualEffectiveTinyKAN-scale002-current

R2-ResidualEffectiveTinyKAN-scale005-current

R3-BoundedResidual-inplace-transform-v2:
  residual applied in-place without separate residual tensor

R4-BoundedResidual-fused-mix-v2:
  compute residual transform and mixing in one fused path

R5-BoundedResidual-chunked-mix-v2:
  output mixing processed in chunks

R6-BoundedResidual-lowrank-mix-v2:
  reduce mixing workspace with edge-owned low-rank factors

R7-BoundedResidual-piecewise2-v2:
  local residual with no dense basis tensor

R8-BoundedResidual-streaming-backward-v2:
  no full delta retention
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
workspace_temp_MB
live_buffer_count_peak
```

### 11.4 判断标准

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
r_{\text{step}}\leq1.35.
$$

Task-open reset pass:

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

### 11.5 可视化

```text
p6_reset_memory_time_pareto.svg
p6_residual_strength_vs_memory.svg
p6_residual_ablation_delta.svg
p6_reset_workspace_model_comparison.svg
```

---

## 12. P7：one-step numerical and residual probe

P7 runs only if P3/P4/P5/P6 produces near-pass candidate.

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

---

## 13. P8：limited task re-entry gate

P8 remains closed unless P7 passes and memory/time gates hold.

### 13.1 打开条件

```text
survivor_type in {S0, S1}
P7 pass
memory_ratio < 1.0
step_ratio <= 1.35
grad pass
no fake/proxy
```

Diagnostic-only if:

```text
survivor_type == S2
memory_ratio <= 1.05
step_ratio <= 1.50
P7 pass
```

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
  Best-DWM2-full-step-fused + ManualAdanLite
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
r_{\text{mem}}<1.0.
$$

Wall-clock must satisfy on at least two datasets:

$$
\operatorname{ValLossAUC}_{time,KAN}
\leq
\operatorname{ValLossAUC}_{time,MLP}.
$$

---

## 14. P9：optimizer exploration remains gated

P9 only opens after P8 task pass. It is not a v6.10 main stage.

If opened, compare:

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

## 15. P10：functional correction remains gated

P10 only opens after P8/P9 pass. Default is `not_run`.

Pass standards if opened:

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
R1-DWM2FullStepSolved:
  Full-step fused DWM2 gets S0/S1 and P7 pass.
  Open task.

R2-DWM2NearPass:
  DWM2 gets S2 with clear improvement.
  Continue kernel engineering.

R3-AttributionIncomplete:
  exact live-set attribution still missing.
  Improve profiler or switch instrumentation.

R4-MicroPassFullFail:
  microkernel is useful but full-step cannot benefit.
  Need full-layer redesign or abandon micro-only strategy.

R5-OneBufferSolved:
  true one-buffer gets S0/S1.
  Continue DWM2 route.

R6-OneBufferNoEffect:
  true one-buffer implemented but memory/time no gain.
  DWM2 memory model likely exhausted.

R7-ResetPrimitiveCandidate:
  reset primitive near-pass and residual effect pass.
  Switch to reset primitive.

R8-ResetStillTooHeavy:
  residual effect pass but memory/time fail.
  Need new bounded primitive.

R9-TerminalPrimitiveRedesign:
  DWM2 and reset both fail.
  Stop DWM2-poly2 small repairs and design new primitive family.

R10-TaskReentryPass:
  memory/time survivor passes limited task.

R11-TaskReentryFail:
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
attribution_pass
boundary_bottleneck_pass
onebuffer_pass
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

## 17. P12：artifact and failure audit

### failure types

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_attribution_incomplete
F5_boundary_overhead
F6_live_set_overlap
F7_full_layer_no_effect
F8_onebuffer_no_effect
F9_reset_residual_effect_fail
F10_reset_memory_fail
F11_task_fail
F12_optimizer_gated
F13_functional_gated
F14_fake_or_proxy_violation
F15_artifact_missing
```

### 必须生成 artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p1_exact_live_set_attribution.csv
p1_boundary_materialization.csv
p2_boundary_audit.csv
p3_full_layer_backward.csv
p4_multilayer_chunked_backward.csv
p5_onebuffer_full_step.csv
p6_bounded_reset_v3.csv
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

### 必须生成 figures

```text
figures/p1_exact_live_set_gantt.svg
figures/p1_peak_live_tensor_top20.svg
figures/p1_gap_attribution_stacked_bar.svg
figures/p2_boundary_time_waterfall.svg
figures/p3_full_layer_package_pareto.svg
figures/p4_chunk_size_vs_step_time.svg
figures/p5_onebuffer_memory_step_pareto.svg
figures/p6_reset_memory_time_pareto.svg
figures/p7_loss_before_after.svg
figures/p11_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 18. 成功与失败解释规则

### Case A：exact attribution pass，但 full-layer fusion 无效

说明知道了峰值来源，但 DWM2 当前 fusion 方案不能利用这些信息。应进入 one-buffer 或 reset。

### Case B：microkernel 快，但 full-step 不快

说明局部 kernel 不是核心，boundary/live-set 是核心。不能继续优化 microkernel。

### Case C：true one-buffer 仍无效

说明 DWM2-poly2 memory model 基本耗尽，应停止小修。

### Case D：reset residual effective 且 near-pass

切换到 reset primitive 主线。

### Case E：reset residual effective 但 memory/time fail

说明 reset 表达力可以，但 workspace model 未解决。需要 new bounded primitive。

### Case F：所有 route 都无 near-pass

进入 primitive family redesign，不再继续 DWM2-poly2。

---

## 19. 最终建议

v6.10 的一句话策略是：

$$
\boxed{
\text{停止局部 microkernel 小修，转向 exact live-set attribution 和 full-step memory model 改写。}
}
$$

如果 v6.10 仍然无法产生 S0/S1/S2 candidate，就应该诚实冻结 DWM2-poly2 主线。届时不要再继续 optimizer 或 functional correction，而应开始设计新的 bounded-workspace PureKAN primitive family。
