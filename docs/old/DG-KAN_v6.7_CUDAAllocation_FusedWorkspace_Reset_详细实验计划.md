# DG-KAN v6.7 更新版：FlashDWM2 Coefficient-Gradient Kernel、CUDA Allocation Trace 与 Bounded-Workspace Reset 实验计划

> 本计划是在原 `DG-KAN v6.7 CUDA Allocation Trace、Lower-Level Workspace Kernel 与 Bounded-Workspace Primitive Reset 实验计划` 基础上更新而来。更新的核心原因是：FlashKAT 论文把 KAT / GR-KAN 的训练慢问题定位到 backward pass 中的 memory stalls，尤其是 inefficient gradient accumulation，并通过重构 kernel、减少 atomic adds 与慢速 memory access 实现大幅训练加速。这与 DG-KAN v6.6 的现象高度一致：DWM2-poly2 的 manual gradient correctness 很好，但 memory/time gate 失败；Python/Torch 层 `bufferReuse-v1` 与 `deltaStreaming-v1` 没有降低 actual CUDA peak。因此 v6.7 不再只做泛泛的 allocation trace，而是把 **coefficient-gradient accumulation** 设为第一优先级 hot path。

---

## 0. 实验整体目标

v6.7 更新版的整体目标不是继续调 task recipe，也不是继续打开 LightSmooth 或 functional correction。v6.7 的目标是回答下面三个问题：

$$
\boxed{
\text{当前 graph-free DWM2-poly2 的 memory/time bottleneck 是否主要来自 backward coefficient-gradient accumulation？}
}
$$

$$
\boxed{
\text{FlashKAT-style local reduction / two-stage reduction / fused backward kernel 能否降低 actual CUDA peak 与 step time？}
}
$$

$$
\boxed{
\text{如果 DWM2 的 coefficient-gradient kernel 修复仍无效，是否应该切换到 bounded-workspace primitive reset？}
}
$$

当前事实基线来自 v6.6 final real-only run。`A1-DWM2-current` 的真实结果是：

```text
memory ratio vs MLP:
  min  = 1.1441
  mean = 1.2918
  max  = 1.4866

step ratio vs MLP:
  min  = 1.5289
  mean = 1.9079
  max  = 2.2236

backward ratio vs MLP:
  min  = 1.1100
  mean = 1.4752
  max  = 1.7471

grad relerr max = 8.38e-08
```

v6.6 还证明：

```text
bufferReuse-v1:
  memory ratio mean = 1.3470
  step ratio mean   = 2.1215
  memory reduction vs current mean = -4.17%

deltaStreaming-v1:
  memory ratio mean = 1.2918
  step ratio mean   = 2.0161
  memory reduction vs current mean = 0.00%
```

因此 v6.7 的最低工程成功目标是：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}} \leq 1.05,
$$

$$
\frac{T_{\text{step,KAN}}}{T_{\text{step,MLP}}} \leq 1.50,
$$

且：

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

v6.7 的正式 task re-entry 标准是：

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

最终终极目标仍然是：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}\leq0.80,
$$

但 v6.7 不把 $0.80$ 作为唯一成功标准。v6.7 的科学目标是判断 DWM2-poly2 的 backward kernel 是否可通过 FlashKAT-style restructuring 被修复，还是必须进入 primitive reset 或 Triton/CUDA terminal kernel route。

---

## 1. FlashKAT 给本项目的直接启发

FlashKAT 的核心结论是：KAT / GR-KAN 即使 FLOPs 与 MLP/Transformer 接近，训练仍可能极慢，根因不是 FLOPs，而是 backward pass 中的 memory stalls，尤其是 coefficient-gradient accumulation 低效。FlashKAT 的做法不是简单减少 activation cache，而是重构 backward kernel，减少 atomic adds、减少慢速 global memory access，并用更局部的 reduction 来减少 memory stall。

这对 DG-KAN 当前阶段有三个直接启发。

第一，v6.6 中 `bufferReuse-v1` 失败并不能说明 fused kernel 方向错了。它只说明 Python/Torch 层 workspace pool 没有真正改变 GPU live set。FlashKAT 提示：真正有效的优化应在 backward hot kernel 内部减少 materialization、global memory traffic 和 reduction contention。

第二，v6.6 中 `deltaStreaming-v1` 与 current 几乎相同，说明简单改变 delta tensor lifetime 不足以改变 peak。FlashKAT 提示：应重点检查 coefficient-gradient accumulation 是否产生了 per-sample contribution tensor、临时 reduction tensor、scatter/reduce 中间量或 atomic-like memory contention。

第三，v6.7 的 micro-kernel isolation 必须把 coefficient-gradient path 提到最高优先级，而不是把它当作普通 K6 组件。新的核心实验应是：

$$
\boxed{
\text{FlashDWM2 coefficient-gradient accumulation audit}
}
$$

---

## 2. 当前事实基线与问题定位

### 2.1 DWM2-current 结构干净但 efficiency 不过

DWM2-current 仍满足：

```text
manual backward = 1
nonKAN = 0
fake/proxy = 0/0
grad relerr max = 8.38e-08
```

这说明当前问题不是：

```text
manual gradient 错
graph-free contract 失败
fake data 污染
nonKAN 参数混入
```

当前问题是：

```text
memory ratio > 1
step ratio > gate
P1 attribution incomplete
repair no effect
```

### 2.2 v6.6 repair 没有改变 peak

v6.6 的 P2/P3 真实 measured 结果说明：

```text
bufferReuse-v1:
  actual CUDA peak 没降，反而 memory/time 变差

deltaStreaming-v1:
  gradient 正确，但 memory ratio 与 current 完全相同

bufferReuse+deltaStreaming:
  仍比 current 差
```

因此 v6.7 不再重复 Python/Torch 层 buffer pool，而是改为：

```text
CUDA allocation trace
coefficient-gradient accumulation attribution
FlashDWM2 local-reduction kernel
two-stage reduction kernel
fused dx + coeffgrad kernel
bounded-workspace reset primitive
```

### 2.3 P9 reset signal 不能直接当成功

v6.6 中 `ManualLinear+TinyChannelResidual` 有 11/18 near-pass：

```text
memory ratio mean = 1.1085
step ratio mean   = 1.1853
backward ratio mean = 0.6925
```

但 DWM2-family reset 没 near-pass：

```text
DWM2-poly1-minimal near-pass = 0/18
DWM2-poly2-singleBuffer near-pass = 0/18
```

因此 `ManualLinear+TinyChannelResidual` 只是 diagnostic。它提示 bounded-workspace residual primitive 可能有希望，但必须验证 residual 非平凡、edge-owned、strict PureKAN accounting 与 one-step loss contribution。

---

## 3. v6.7 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio、derived rows 作为结果。任何未真实测量的 package 必须写为：

```text
not_implemented
not_applicable
not_run
```

第二，P1-P4 没有 memory/time survivor 时，不允许打开：

```text
task re-entry
LightSmooth
functional correction
3-seed confirm
5-seed confirm
10-seed confirm
```

第三，不允许把 `manual_cache_MB` 或 `workspace_pool_MB` 下降当成 actual CUDA peak 下降。所有 memory gate 必须以真实：

```text
peak_allocated_MB
peak_reserved_MB
backward_memory_ratio_vs_MLP
```

为准。

第四，不允许把 `ManualLinear+TinyChannelResidual` 的 near-pass 写成 DG-KAN success。它必须重新通过 residual-effect gate。

第五，不允许只看 FLOPs 或 step wall-clock，不看 memory stall。FlashDWM2 必须记录：

```text
global_load_bytes
global_store_bytes
atomic_add_count
L2_read_transactions
L2_write_transactions
stall_long_scoreboard
SM_occupancy
register_spill_count
shared_memory_bytes
```

如果环境不能获取 Nsight 指标，必须写为 `metric_unavailable`，不能填假值。

第六，不允许只实现 forward fused kernel。v6.7 的主瓶颈假设是 backward coefficient-gradient accumulation，任何只优化 forward transform 的 package 只能作为辅助实验。

---

## 4. 核心假设

### H1：v6.6 attribution 失败来自 profiler 粒度不足，需要 allocation stack + tensor lifetime + coefficient-gradient attribution

v6.6 中 phase peak 捕获到了 peak phase，但 `attribution_pass=0/18`，说明不能把 KAN-over-MLP gap 分解到可行动 top-3 source。H1 假设：通过 CUDA allocation trace、PyTorch memory snapshot、NVTX range、Nsight memory-stall counters 和 stack-level attribution，可以解释至少 $95\%$ 的 KAN-over-MLP peak gap。

定义：

$$
G_{\text{peak}} = M_{\text{peak,KAN}} - M_{\text{peak,MLP}}.
$$

如果 attribution 项为 $A_i$，则 H1 成立要求：

$$
\frac{\sum_i A_i}{G_{\text{peak}}}\geq0.95.
$$

并且 top-3 source 至少解释 $70\%$ 的 peak gap：

$$
\frac{A_1+A_2+A_3}{G_{\text{peak}}}\geq0.70.
$$

如果 H1 不成立，则 route 必须保持：

```text
R3-attributionIncomplete
```

不能继续做 repair claim。

---

### H2：DWM2 backward 的主要 memory/time bottleneck 可能来自 coefficient-gradient accumulation

FlashKAT 将 KAT/GR-KAN 的 slowdown 定位到 backward pass 中 inefficient gradient accumulation。DWM2-poly2 也需要计算 residual coefficient gradient：

$$
g_{\theta}
=
\sum_b
\delta_{z,b}
\frac{\partial r(x_b)}{\partial \theta}.
$$

以 poly2 residual 为例：

$$
z = x + s(a_1 x + a_2 x^2),
$$

$$
\frac{\partial L}{\partial a_1}
=
\sum_b \delta_{z,b} s x_b,
$$

$$
\frac{\partial L}{\partial a_2}
=
\sum_b \delta_{z,b} s x_b^2.
$$

如果当前 PyTorch/Torch path 产生完整 per-sample contribution tensor：

$$
C_{b,i,k}
=
\delta_{z,b,i}
\frac{\partial r_i(x_{b,i})}{\partial \theta_{i,k}},
$$

再对 $b$ 做 reduction，则会造成：

```text
large temporary contribution tensor
global memory write/read
reduction temp
allocator peak
memory stall
```

H2 成立的标准是 coefficient-gradient phase 占比满足至少一个条件：

$$
\frac{T_{\text{coeffgrad}}}{T_{\text{backward}}}\geq0.30,
$$

或：

$$
\frac{M_{\text{coeffgrad temp}}}{G_{\text{peak}}}\geq0.30,
$$

或 Nsight 指标显示：

```text
stall_long_scoreboard is among top stall reasons
global_store_bytes / useful_coeffgrad_bytes is high
atomic_add_count or reduction-write count is high
```

如果 H2 成立，P2.5 FlashDWM2 是主线。

---

### H3：FlashDWM2 local-reduction / two-stage reduction 可以降低 coefficient-gradient memory/time

H3 假设：把 coefficient-gradient accumulation 从 PyTorch reduction / temp tensor 改为 local reduction / two-stage reduction，可以降低 memory traffic 和 peak。

新的 FlashDWM2 coeffgrad 结构是：

$$
g_{\theta}^{block}
=
\sum_{b\in block}
\delta_{z,b}
\frac{\partial r(x_b)}{\partial \theta},
$$

然后只把少量 block partials 写回 global memory：

$$
g_{\theta}
=
\sum_{block}
g_{\theta}^{block}.
$$

H3 成立要求：

$$
\frac{T_{\text{coeffgrad,new}}}{T_{\text{coeffgrad,current}}}\leq0.50,
$$

$$
\frac{M_{\text{coeffgrad,new}}}{M_{\text{coeffgrad,current}}}\leq0.70,
$$

且：

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

如果只降 time 不降 peak，可以进入 runtime diagnostic，但不能打开 task。

---

### H4：fused dx + coeffgrad 比单独 coeffgrad 更可能改变 full backward peak

单独优化 coefficient grad 可能不足，因为 full backward 还包含：

$$
\delta_z = \delta_y W^T,
$$

$$
\delta_x = \delta_z \left(1+s(a_1+2a_2x)\right).
$$

如果 $\delta_z$、$\delta_x$、coefficient contribution 分别 materialize，峰值仍然高。H4 假设：fused dx + coeffgrad kernel 可以同时 consume $\delta_z$，写出 $\delta_x$，并局部累积 $g_\theta$，从而减少 live tensor overlap。

H4 成立要求 full backward package 相对 current 满足：

$$
\frac{M_{\text{backward,new}}}{M_{\text{backward,current}}}\leq0.90,
$$

$$
\frac{T_{\text{backward,new}}}{T_{\text{backward,current}}}\leq0.90.
$$

并且 full step 满足 near-pass：

$$
\frac{M_{\text{backward,new}}}{M_{\text{backward,MLP}}}\leq1.05,
$$

$$
\frac{T_{\text{step,new}}}{T_{\text{step,MLP}}}\leq1.50.
$$

---

### H5：FlashDWM2 reduction 顺序可能影响 coefficient-gradient numerical error

FlashKAT 报告其重构 kernel 还能降低 coefficient-gradient rounding errors。DG-KAN 的 FlashDWM2 kernel 也必须记录数值误差，而不能只看速度。

H5 成立标准是 new reduction 至少不比 current 更差：

$$
\operatorname{grad\_relerr}_{new}
\leq
\max(10^{-4}, 1.25\operatorname{grad\_relerr}_{current}),
$$

并且如果和 fp64 reference 比较：

$$
\operatorname{MAE}(g_{\theta,new},g_{\theta,fp64})
\leq
\operatorname{MAE}(g_{\theta,current},g_{\theta,fp64}).
$$

如果 new kernel 明显更快但 rounding error 变差，必须作为 diagnostic，不允许进入 task。

---

### H6：如果 FlashDWM2 也无效，DWM2-poly2 当前 abstraction 应判定为 exhausted

H6 是决策假设。如果 coefficient-gradient accumulation 被优化后仍满足：

$$
\Delta M_{\text{peak}} < 5\%,
$$

且：

$$
\Delta T_{\text{step}} < 5\%,
$$

并且 P4 bounded-workspace reset 也没有 near-pass，则 DWM2-poly2 当前 Python/Torch/manual abstraction 不应再继续 task/functional 尝试，必须进入 terminal lower-level custom kernel 或 primitive family reset。

---

### H7：bounded-workspace reset 必须验证 residual 非平凡，而不是只接近 manual-linear baseline

`ManualLinear+TinyChannelResidual` 的 near-pass 不能直接作为 KAN success。H7 要求 reset primitive 同时满足 memory/time 与 residual effect。

Memory/time gate：

$$
\frac{M_{\text{backward}}}{M_{\text{MLP}}}\leq1.05,
$$

$$
\frac{T_{\text{step}}}{T_{\text{MLP}}}\leq1.35.
$$

Residual 非平凡 gate：

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02,
$$

并且：

$$
|\Delta L_{\text{residual ablation}}|>10^{-4}
$$

或：

$$
|\Delta \text{logit}_{\text{residual ablation}}|>10^{-4}.
$$

---

## 5. 实验阶段总览

v6.7 更新版分为十一个阶段：

```text
P0: v6.6 lineage reproduction and contract check
P1: CUDA allocation trace and tensor lifetime attribution
P2: component micro-kernel isolation
P2.5: FlashDWM2 coefficient-gradient accumulation audit
P3: lower-level fused DWM2 backward package
P4: bounded-workspace reset primitive benchmark
P5: one-step numerical and residual-effect probe
P6: limited task re-entry gate
P7: functional correction remains gated
P8: route decision and next branch
P9: final failure taxonomy and artifact audit
```

其中新增核心阶段是：

```text
P2.5: FlashDWM2 coefficient-gradient accumulation audit
```

P2.5 的结果决定 P3 优先组合什么 fused package。

---

## 6. P0：v6.6 lineage reproduction and contract check

### 6.1 目的

确认 v6.7 与 v6.6 final run 可比较，并确认新增 FlashDWM2 / reset variants 是真实实现而非 proxy。

### 6.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-poly2-current
DWM2-bufferReuse-v1
DWM2-deltaStreaming-v1
DWM2-bufferReuse+deltaStreaming
ManualLinear+TinyChannelResidual
DWM2-poly1-minimal
DWM2-poly2-singleBuffer
FlashDWM2-coeffgrad-local-reduce, if implemented
FlashDWM2-coeffgrad-two-stage-reduce, if implemented
FlashDWM2-fused-dx-coeffgrad, if implemented
```

### 6.3 必须记录字段

```text
run_id
artifact_root
script_path
variant
status
implementation_type
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
manual_forward_available
manual_backward_available
manual_update_available
nonKAN_param_count
edge_param_count
residual_param_count
mixing_param_count
rollback_max_error
v66_current_memory_ratio_mean
v67_current_memory_ratio_mean
v66_current_step_ratio_mean
v67_current_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
```

### 6.4 通过标准

```text
fake_data_used = 0
proxy_row_used = 0
uses_loss_backward = 0 for graph-free candidates
nonKAN_param_count = 0 for PureKAN candidates
rollback_max_error < 1e-8
```

reproduction 可比性要求：

$$
|r_{\text{memory,v67}}-r_{\text{memory,v66}}|\leq0.05,
$$

$$
|r_{\text{step,v67}}-r_{\text{step,v66}}|\leq0.15.
$$

### 6.5 可视化

```text
p0_reproduction_memory_step_delta.svg
p0_contract_heatmap.svg
p0_variant_implementation_status.svg
```

---

## 7. P1：CUDA allocation trace and tensor lifetime attribution

### 7.1 目的

P1 要回答：

```text
peak 到底发生在哪个 phase？
peak 时哪些 tensor alive？
coefficient-gradient accumulation 是否是 top source？
这些 tensor 来自哪些 op / stack？
allocator 是否因为 block reuse / padding / fragmentation 导致 peak？
```

### 7.2 Trace phase

必须加 NVTX range 或等价标签：

```text
phase_forward_transform
phase_forward_mix
phase_loss_delta
phase_backward_output
phase_backward_block_delta
phase_backward_coeff_grad
phase_backward_input
phase_update_params
phase_optimizer_state
phase_cleanup
```

其中 `phase_backward_coeff_grad` 必须单独独立出来，不能再和 general backward block 混在一起。

### 7.3 Trace 工具

至少使用：

```text
torch.profiler(profile_memory=True)
torch.cuda.memory_snapshot or equivalent allocator snapshot
NVTX ranges
```

如果可用，额外使用：

```text
Nsight Systems
Nsight Compute
```

Nsight 指标应包括：

```text
stall_long_scoreboard
SM occupancy
global load/store throughput
L2 read/write transactions
atomic throughput
register spill count
shared memory usage
```

如果不可用，必须写：

```text
metric_unavailable
```

### 7.4 测量对象

```text
A0-MLP-autograd-reference
A1-MLP-manual-linear-reference
A2-DWM2-current
A3-DWM2-bufferReuse-v1
A4-DWM2-deltaStreaming-v1
A5-DWM2-bufferReuse+deltaStreaming
A6-FlashDWM2-coeffgrad-local-reduce, if implemented
A7-FlashDWM2-coeffgrad-two-stage-reduce, if implemented
A8-FlashDWM2-fused-dx-coeffgrad, if implemented
A9-ManualLinear+TinyChannelResidual
```

### 7.5 Grid

核心 trace grid：

```text
datasets:
  Fashion-MNIST
  KMNIST

batch sizes:
  128
  512

depths:
  2
  4

seed:
  0

measure:
  20 warmup + 20 traced update steps
```

完整 profiler grid：

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

measure:
  50 warmup + 200 update steps
```

### 7.6 必须记录字段

#### phase-level fields

```text
variant
dataset
batch_size
depth
phase_name
allocated_before_MB
allocated_peak_MB
allocated_after_MB
reserved_before_MB
reserved_peak_MB
reserved_after_MB
phase_peak_delta_MB
phase_retained_delta_MB
phase_duration_ms
phase_kernel_count
phase_allocation_count
phase_free_count
```

#### allocation-level fields

```text
allocation_id
variant
phase_name
requested_size_MB
allocated_block_size_MB
padding_MB
lifetime_start_phase
lifetime_end_phase
lifetime_duration_ms
allocation_stack_hash
allocation_stack_top_file
allocation_stack_top_function
source_op
tensor_shape
tensor_dtype
is_workspace_pool
is_delta
is_poly_temp
is_coeffgrad_contribution
is_coeffgrad_reduction_temp
is_update_temp
is_optimizer_state
is_manual_cache
is_mlp_baseline_activation
```

#### coeffgrad-specific fields

```text
coeffgrad_phase_time_ms
coeffgrad_phase_peak_MB
coeffgrad_temp_MB
coeffgrad_contribution_tensor_MB
coeffgrad_reduction_temp_MB
coeffgrad_global_load_bytes
coeffgrad_global_store_bytes
coeffgrad_atomic_add_count
coeffgrad_reduction_write_count
coeffgrad_local_reduction_bytes
coeffgrad_block_partial_MB
coeffgrad_stall_long_scoreboard
coeffgrad_L2_read_transactions
coeffgrad_L2_write_transactions
coeffgrad_register_spill_count
coeffgrad_shared_memory_bytes
```

#### attribution summary fields

```text
KAN_peak_MB
MLP_peak_MB
peak_gap_MB
explained_gap_MB
unexplained_gap_MB
explain_ratio
top1_source
top1_MB
top1_fraction_of_gap
top2_source
top2_MB
top2_fraction_of_gap
top3_source
top3_MB
top3_fraction_of_gap
coeffgrad_fraction_of_gap
allocator_padding_total_MB
fragmentation_ratio
new_cuda_block_count
reused_cuda_block_count
largest_live_set_MB
```

### 7.7 判断标准

P1 attribution pass 要求：

$$
\frac{M_{\text{explained}}}{G_{\text{peak}}}\geq0.95,
$$

且：

$$
\frac{M_{\text{top3}}}{G_{\text{peak}}}\geq0.70.
$$

Coeffgrad bottleneck 判定要求至少满足一个：

$$
\frac{M_{\text{coeffgrad}}}{G_{\text{peak}}}\geq0.30,
$$

或：

$$
\frac{T_{\text{coeffgrad}}}{T_{\text{backward}}}\geq0.30.
$$

如果 coeffgrad 既不是 memory top source，也不是 time top source，则 P2.5 仍可跑一次 diagnostic，但 P3 不应优先组合 FlashDWM2。

### 7.8 可视化

```text
p1_phase_peak_waterfall.svg
p1_tensor_lifetime_gantt.svg
p1_kan_over_mlp_gap_stacked_bar.svg
p1_top_allocation_stack_table.md
p1_allocator_padding_scatter.svg
p1_coeffgrad_fraction_heatmap.svg
p1_coeffgrad_memory_stall_dashboard.svg
p1_top3_source_fraction_heatmap.svg
```

---

## 8. P2：component micro-kernel isolation

### 8.1 目的

P2 把 DWM2 hot path 拆到组件级，确认 coefficient-gradient path、delta propagation、transform、update 各自的 memory/time 占比。

### 8.2 Micro-kernels

```text
K0-forward-transform-only:
  z = x + s poly2(x)

K1-forward-mix-only:
  y = Wz

K2-loss-delta-only:
  softmax CE delta

K3-backward-delta-only:
  dz = delta W

K4-poly-derivative-only:
  derivative of poly2 residual

K5-input-grad-only:
  dx = dz * derivative

K6-coeff-grad-only-current:
  current coeff gradient accumulation

K6a-coeff-grad-local-reduce:
  FlashDWM2 local/block reduction

K6b-coeff-grad-two-stage-reduce:
  block partials + second stage global reduction

K6c-coeff-grad-no-atomic-shared-memory:
  shared-memory reduction, minimal global writes

K6d-coeff-grad-warp-register-reduce:
  warp/register-level reduction

K7-update-only:
  optimizer update temp

K8-transform+derivative-fused:
  compute z and derivative in one buffer

K9-delta+inputgrad-fused:
  consume delta and produce dx

K10-coeffgrad+dx-fused:
  compute dx and coeff grad in one fused kernel
```

### 8.3 对照实现

```text
torch-current
torch-out-buffer
torch-compile
triton-fused
cuda-extension, optional
```

### 8.4 必须记录指标

```text
component_name
implementation
input_shape
output_shape
forward_time_ms
backward_time_ms
peak_allocated_MB
peak_reserved_MB
temp_allocated_MB
allocation_count
kernel_count
new_cuda_block_count
largest_temp_MB
bandwidth_estimate_GBps
flops_estimate
global_load_bytes
global_store_bytes
atomic_add_count
shared_memory_bytes
register_spill_count
stall_long_scoreboard
grad_relerr
grad_cos
component_output_relerr
rounding_MAE_vs_fp64
```

### 8.5 判断标准

普通 micro-kernel repair 有价值标准：

$$
\frac{M_{\text{component,repair}}}{M_{\text{component,current}}}\leq0.80
$$

或：

$$
\frac{T_{\text{component,repair}}}{T_{\text{component,current}}}\leq0.80.
$$

Coeffgrad-specific useful pass：

$$
\frac{T_{\text{coeffgrad,new}}}{T_{\text{coeffgrad,current}}}\leq0.50,
$$

$$
\frac{M_{\text{coeffgrad,new}}}{M_{\text{coeffgrad,current}}}\leq0.70.
$$

数值要求：

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

### 8.6 可视化

```text
p2_component_runtime_waterfall.svg
p2_component_memory_waterfall.svg
p2_component_repair_pareto.svg
p2_kernel_count_by_component.svg
p2_temp_allocation_by_component.svg
p2_coeffgrad_reduction_strategy_pareto.svg
p2_coeffgrad_rounding_error_bar.svg
```

---

## 9. P2.5：FlashDWM2 coefficient-gradient accumulation audit

### 9.1 目的

P2.5 是本版新增的核心阶段。它专门验证 FlashKAT-style coefficient-gradient kernel 是否能解决 DWM2 backward 的 memory/time 问题。

### 9.2 FlashDWM2 候选实现

```text
F0-current-coeffgrad:
  当前 PyTorch/Torch coefficient-gradient path

F1-local-reduce:
  每个 block 内局部累积 coeff grad，减少 per-sample contribution tensor

F2-two-stage-reduce:
  第一阶段写 block partials，第二阶段 reduce partials

F3-shared-memory-reduce:
  使用 shared memory 做局部 reduction，减少 global writes

F4-warp-register-reduce:
  warp/register-level reduction，减少 shared/global memory pressure

F5-fused-dx-coeffgrad:
  同时计算 dx 与 coeff grad，consume dz once

F6-fused-dx-coeffgrad-update:
  diagnostic only，把 coeff update temp 也尝试 fused/reused

F7-chunked-batch-reduce:
  按 batch chunk 处理，控制 peak live contribution
```

### 9.3 数学参考

对于 poly2 residual：

$$
r(x)=a_1x+a_2x^2.
$$

如果：

$$
z=x+s r(x),
$$

则：

$$
\frac{\partial L}{\partial a_1}
=
\sum_b
\delta_{z,b} s x_b,
$$

$$
\frac{\partial L}{\partial a_2}
=
\sum_b
\delta_{z,b} s x_b^2,
$$

并且：

$$
\delta_x
=
\delta_z
\left(1+s(a_1+2a_2x)\right).
$$

FlashDWM2 的原则是避免 materialize：

$$
C_{b,i,k}=
\delta_{z,b,i}
\frac{\partial r_i(x_{b,i})}{\partial \theta_{i,k}}.
$$

而是直接做：

$$
g_{\theta}^{block}
=
\sum_{b\in block}
C_{b,i,k}.
$$

### 9.4 必须记录指标

```text
flash_variant
coeffgrad_time_ms
coeffgrad_peak_allocated_MB
coeffgrad_temp_MB
contribution_tensor_MB
block_partial_MB
global_load_bytes
global_store_bytes
atomic_add_count
reduction_write_count
shared_memory_bytes
register_spill_count
stall_long_scoreboard
warp_occupancy
SM_efficiency
L2_read_transactions
L2_write_transactions
rounding_MAE_vs_fp64
rounding_max_abs_vs_fp64
grad_relerr_vs_current
grad_cos_vs_current
grad_relerr_vs_autograd
grad_cos_vs_autograd
```

### 9.5 判断标准

P2.5 FlashDWM2 pass 要求：

$$
\frac{T_{\text{coeffgrad,new}}}{T_{\text{coeffgrad,current}}}\leq0.50,
$$

$$
\frac{M_{\text{coeffgrad,new}}}{M_{\text{coeffgrad,current}}}\leq0.70,
$$

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

如果 Nsight 指标可用，还要求至少一个 stall 指标改善：

$$
\frac{\text{stall\_long\_scoreboard}_{new}}
{\text{stall\_long\_scoreboard}_{current}}
\leq0.70,
$$

或：

$$
\frac{\text{global\_store\_bytes}_{new}}
{\text{global\_store\_bytes}_{current}}
\leq0.70.
$$

### 9.6 P2.5 输出决策

```text
F-pass:
  Flash coeffgrad useful and numerically valid.
  Must be included in P3.

F-time-only:
  time improves but memory not.
  Include only if P1 says coeffgrad is time bottleneck.

F-memory-only:
  memory improves but time worsens.
  Include only as diagnostic.

F-rounding-fail:
  speed/memory improves but gradient error exceeds gate.
  Do not include in P3 task path.

F-no-effect:
  no meaningful time/memory improvement.
  DWM2 bottleneck likely elsewhere.
```

### 9.7 可视化

```text
p25_flash_coeffgrad_time_memory_pareto.svg
p25_reduction_strategy_breakdown.svg
p25_global_memory_traffic_bar.svg
p25_stall_long_scoreboard_bar.svg
p25_rounding_error_vs_speed.svg
p25_grad_correctness_heatmap.svg
```

---

## 10. P3：lower-level fused DWM2 backward package

### 10.1 目的

P3 组合 P2 和 P2.5 中有效的 fused kernel，形成完整 DWM2 backward package。

### 10.2 Candidate packages

```text
D0-current
D1-torch-compile-current
D2-fused-transform
D3-fused-transform-derivative
D4-fused-delta-inputgrad
D5-flash-coeffgrad-local-reduce
D6-flash-coeffgrad-two-stage-reduce
D7-fused-dx-flash-coeffgrad
D8-fused-update-workspace
D9-full-fused-light
D10-full-fused-flash
D11-full-fused-onebuffer
```

其中：

```text
full-fused-light =
  fused transform + fused derivative + fused delta/inputgrad

full-fused-flash =
  full-fused-light + flash coeffgrad

full-fused-onebuffer =
  full-fused-flash + update workspace reuse + single live residual buffer
```

### 10.3 测量 grid

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

### 10.4 必须记录指标

```text
package
components
flash_coeffgrad_variant
implementation_status
memory_ratio_min
memory_ratio_mean
memory_ratio_max
step_ratio_min
step_ratio_mean
step_ratio_max
backward_ratio_min
backward_ratio_mean
backward_ratio_max
memory_improvement_vs_current
step_improvement_vs_current
allocation_count_reduction
kernel_count_reduction
global_store_bytes_reduction
stall_long_scoreboard_reduction
peak_gap_explain_ratio
top_gap_source_after_repair
grad_relerr_max
grad_cos_min
rounding_MAE_vs_fp64
forward_relerr_max
rollback_error
```

### 10.5 Survivor 类型

```text
S0:
  memory_ratio < 1.0 and step_ratio <= 1.20

S1:
  memory_ratio < 1.0 and step_ratio <= 1.35

S2:
  memory_ratio <= 1.05 and step_ratio <= 1.50 and memory improvement >= 10%

S3:
  no near-pass but attribution explains failure

S4:
  no near-pass and attribution incomplete

S5:
  gradient correctness fail

S6:
  Flash coeffgrad helps micro-kernel but not full package
```

S0/S1 可以进入 P5 one-step probe 与 P6 limited task re-entry。  
S2 只能进入 P5 one-step probe，不能 full task。  
S6 表示局部 kernel 有价值，但 full backward 仍有其他 peak source；下一步继续 fusion。

### 10.6 可视化

```text
p3_package_memory_step_pareto.svg
p3_s0_s1_s2_threshold_plot.svg
p3_kernel_count_reduction_bar.svg
p3_allocation_count_reduction_bar.svg
p3_global_store_reduction_bar.svg
p3_stall_reduction_bar.svg
p3_gradient_correctness_plot.svg
p3_batch_depth_stability_heatmap.svg
```

---

## 11. P4：bounded-workspace reset primitive benchmark

### 11.1 目的

如果 DWM2 fused package 无法 near-pass，P4 测试改变 memory model 的 primitive。P4 不做 full task，只做 kernel/profiler 与 residual-effect probe 准备。

### 11.2 Reset candidates

```text
B0-ManualLinear-reference

B1-ManualLinear+TinyChannelResidual-edgeOwned:
  z = x + s r(x)
  y = Wz
  residual parameters are edge-owned
  no nonKAN trainable params

B2-OneBufferPoly1Residual:
  one live residual buffer

B3-OneBufferPiecewiseLinear2:
  two-bin interpolation, no full knot activation tensor

B4-OneBufferFastRational:
  rational residual with bounded temp buffers

B5-LinearResidualGatedKAN:
  z = x + gate(x) r(x), gate must be edge-owned

B6-ChunkedMixingKAN:
  output mixing processed in chunks to bound workspace

B7-SparseInterpForwardOnly:
  forward + input backward only, no knot gradient

B8-SparseInterpStreamingGrad:
  adds streaming knot gradient if B7 near-pass

B9-FlashTinyResidual:
  Tiny residual with Flash-style coeffgrad local reduction
```

### 11.3 必须记录指标

```text
primitive
nonKAN_param_count
edge_param_count
residual_param_count
manual_backward_available
memory_ratio_min
memory_ratio_mean
memory_ratio_max
step_ratio_min
step_ratio_mean
step_ratio_max
backward_ratio_mean
grad_relerr_max
grad_cos_min
residual_over_base
residual_norm
base_norm
residual_ablation_delta_logit
residual_ablation_delta_loss
manual_cache_MB
workspace_temp_MB
allocation_count
kernel_count
global_store_bytes
stall_long_scoreboard
near_pass_count
```

### 11.4 判断标准

P4 near-pass 要求：

$$
\frac{M_{\text{backward}}}{M_{\text{MLP}}}\leq1.05,
$$

$$
\frac{T_{\text{step}}}{T_{\text{MLP}}}\leq1.35,
$$

$$
\operatorname{grad\_relerr}<10^{-4}.
$$

Residual 非平凡要求：

$$
\frac{\|r(x)\|}{\|x\|}\geq0.02,
$$

并且：

$$
|\Delta L_{\text{residual ablation}}|>10^{-4}
$$

或：

$$
|\Delta \text{logit}_{\text{residual ablation}}|>10^{-4}.
$$

### 11.5 可视化

```text
p4_reset_memory_time_pareto.svg
p4_residual_effect_vs_memory.svg
p4_reset_ablation_bar.svg
p4_workspace_model_comparison.svg
p4_flash_reset_comparison.svg
```

---

## 12. P5：one-step numerical and residual-effect probe

P5 只对 P3/P4 的 S0/S1/S2 或 near-pass candidate 运行。它验证 candidate 是否数值稳定、是否有真实 loss descent、residual 是否有贡献。

### 12.1 Probe 方法

每个 candidate 在每个 dataset 上执行：

```text
clone weights
compute manual gradient
apply one update
measure train loss before/after
measure holdout loss before/after
measure validation mini-batch loss before/after
measure residual ablation
rollback weights
check rollback exactness
```

### 12.2 必须记录

```text
candidate
dataset
batch_size
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
param_delta_norm
update_over_param_norm
grad_norm
residual_enabled_loss
residual_disabled_loss
residual_ablation_delta_loss
residual_enabled_logit_norm
residual_disabled_logit_norm
residual_ablation_delta_logit
```

### 12.3 判断标准

$$
\Delta L_{\text{train}}<0,
$$

$$
\Delta L_{\text{holdout}}\leq0.02L_{\text{holdout,before}},
$$

$$
\operatorname{BadStepRate}\leq0.05,
$$

$$
\operatorname{rollback\_error}<10^{-8}.
$$

Residual contribution 要求：

$$
|\Delta L_{\text{residual ablation}}|>10^{-4}
$$

或：

$$
|\Delta \text{logit}_{\text{residual ablation}}|>10^{-4}.
$$

### 12.4 可视化

```text
p5_before_after_loss_plot.svg
p5_bad_step_heatmap.svg
p5_update_norm_vs_loss_delta.svg
p5_residual_ablation_plot.svg
p5_rollback_error_bar.svg
```

---

## 13. P6：limited task re-entry gate

P6 决定是否允许有限 task re-entry。

### 13.1 打开条件

Open limited task re-entry if:

```text
survivor_type in {S0, S1}
P5 pass
grad correctness pass
no fake/proxy
memory_ratio < 1.0
step_ratio <= 1.35
```

Open diagnostic-only task if:

```text
survivor_type == S2
P5 pass
memory_ratio <= 1.05
step_ratio <= 1.50
```

Do not open task if:

```text
S3/S4/S5/S6
P5 fail
residual ablation fail
```

### 13.2 P6 task design

如果正式打开 task：

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
  Best-DWM2-fused-flash-package + ManualAdanLite
  Best-reset-primitive + ManualAdanLite
  Best-reset-primitive + ManualAdamW
```

如果 diagnostic-only：

```text
seed = 0 only
short budget only
no final claim
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
forward_time_ratio
backward_time_ratio
samples_per_second
margin_mean
margin_p10
feature_rank
classwise_acc
seedwise_failure_reason
```

### 13.4 判断标准

正式 task pass 要求：

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.01
$$

on all datasets, and at least two datasets satisfy:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}.
$$

Memory 必须保持：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}<1.0.
$$

Wall-clock 至少满足：

$$
\operatorname{ValLossAUC}_{time,KAN}\leq\operatorname{ValLossAUC}_{time,MLP}
$$

on at least two datasets.

### 13.5 可视化

```text
p6_val_loss_vs_step.svg
p6_val_loss_vs_time.svg
p6_accuracy_vs_time.svg
p6_time_to_target_bar.svg
p6_task_efficiency_pareto.svg
p6_seedwise_delta_plot.svg
```

---

## 14. P7：functional correction remains gated

P7 默认关闭。只有 P6 正式 task pass 后才允许 functional correction smoke。

如果 P7 被打开，只允许 smoke，不允许 final claim。

### 14.1 Minimal P7 design

```text
Task-only best candidate
Task + small residual geometry correction
Task + low-frequency LightSmooth event
```

### 14.2 必须记录

```text
cos_new_task
holdout_descent_ratio
bad_step_rate
phi_change
curvature_change
ECE_change
correction_overhead_ms
fallback_rate
memory_overhead_ratio
step_overhead_ratio
```

### 14.3 判断标准

$$
\cos(d_{\text{new}},d_{\text{task}})\geq0.85,
$$

$$
\frac{\operatorname{HoldoutDescent}(d_{\text{new}})}
{\operatorname{HoldoutDescent}(d_{\text{task}})}
\geq0.95,
$$

$$
\operatorname{BadStepRate}\leq0.02.
$$

Correction overhead:

$$
\frac{T_{\text{corrected}}}{T_{\text{task-only}}}\leq1.10.
$$

---

## 15. P8：route decision and next branch

### Route cases

```text
R1-DWM2FlashSolved:
  FlashDWM2 package gets S0/S1 and P5 pass.
  Open limited task.

R2-DWM2FlashNearPass:
  FlashDWM2 gets S2 with clear memory/time improvement.
  Continue fused kernel engineering, no full task.

R3-AttributionIncomplete:
  CUDA trace still cannot explain peak gap.
  Improve profiler / Nsight trace.

R4-RepairNoEffect:
  lower-level fused repairs do not improve memory/time.
  DWM2 current abstraction likely exhausted.

R5-ResetPrimitiveCandidate:
  bounded-workspace reset gets near-pass and residual effect pass.
  Switch next cycle to reset primitive.

R6-ResetLinearOnly:
  manual-linear-like reset near-passes but residual effect fails.
  It is a baseline, not KAN success.

R7-TerminalCustomKernelNeeded:
  no DWM2 survivor and no reset near-pass.
  Need Triton/CUDA custom kernel or new primitive family.

R8-TaskReentryPass:
  memory/time survivor also passes limited task.
  Next cycle can consider 5-seed confirm.

R9-TaskReentryFail:
  memory/time survivor exists but task fails.
  Need expressivity repair.

R10-FlashCoeffgradLocalOnly:
  coeffgrad micro-kernel improves, but full backward package does not.
  Continue fusing dx/update/lifetime before task.
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
coeffgrad_bottleneck_pass
flash_coeffgrad_pass
flash_coeffgrad_variant
top1_peak_source
top2_peak_source
top3_peak_source
fallback_triggered
fallback_near_pass_count
residual_effect_pass
open_task_reentry
open_functional_correction
no_fake
no_proxy
primary_blocker
next_required_implementation
```

---

## 16. P9：final failure taxonomy and artifact audit

### 16.1 failure types

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_attribution_incomplete
F5_repair_no_effect
F6_repair_breaks_gradient
F7_residual_effect_fail
F8_task_fail
F9_functional_gate_fail
F10_fake_or_proxy_violation
F11_gated_not_run
F12_artifact_missing
F13_coeffgrad_memory_stall
F14_coeffgrad_reduction_no_effect
F15_flash_rounding_error
F16_flash_local_only_no_full_gain
```

### 16.2 Artifact audit

必须生成：

```text
p0_contract.csv
p0_reproduction_check.csv
p1_cuda_allocation_trace.csv
p1_tensor_lifetime_topk.csv
p1_allocator_block_summary.csv
p1_phase_peak_summary.csv
p1_coeffgrad_attribution.csv
p2_component_microkernel.csv
p2_component_repair_summary.csv
p25_flash_coeffgrad_audit.csv
p25_flash_coeffgrad_correctness.csv
p25_flash_coeffgrad_memory_stall.csv
p3_fused_dwm2_packages.csv
p3_fused_dwm2_package_detail.csv
p4_bounded_workspace_reset.csv
p5_one_step_probe.csv
p6_task_reentry.csv
p6_task_trace.csv
p7_functional_correction_smoke.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### 16.3 必须生成可视化总表

```text
figures/p1_phase_peak_waterfall.svg
figures/p1_tensor_lifetime_gantt.svg
figures/p1_gap_attribution_stacked_bar.svg
figures/p1_allocator_padding_scatter.svg
figures/p1_coeffgrad_memory_stall_dashboard.svg
figures/p2_component_runtime_waterfall.svg
figures/p2_component_memory_waterfall.svg
figures/p25_flash_coeffgrad_time_memory_pareto.svg
figures/p25_global_memory_traffic_bar.svg
figures/p25_rounding_error_vs_speed.svg
figures/p3_package_memory_step_pareto.svg
figures/p4_reset_memory_time_pareto.svg
figures/p5_residual_ablation_plot.svg
figures/p8_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 17. 成功与失败解释规则

### Case A：FlashDWM2 full package memory pass

如果：

$$
M_{\text{backward}}/M_{\text{MLP}}<1.0,
$$

且：

$$
T_{\text{step}}/T_{\text{MLP}}\leq1.35,
$$

则 DWM2 继续作为主线，并允许 P5/P6。

### Case B：FlashDWM2 coeffgrad micro-kernel 有效，但 full package 不过

如果 coeffgrad micro-kernel：

$$
T_{\text{coeffgrad,new}}\leq0.5T_{\text{coeffgrad,current}},
$$

但 full package memory/time 不过，则说明瓶颈还在 dx/update/lifetime overlap。下一轮继续 full backward fusion，不开 full task。

### Case C：FlashDWM2 无效

如果 Flash coeffgrad 对 memory/time 改善 $<5\%$，且 P1 显示 coeffgrad 不是 top source，则 DWM2 当前 bottleneck 在其他 phase，应回到 allocation trace 指出的 top source。

### Case D：reset primitive near-pass 且 residual 非平凡

如果 bounded-workspace reset primitive near-pass 且 residual effect pass，则下一轮以 reset primitive 为主线。

### Case E：reset primitive near-pass 但 residual 无效

如果 residual effect fail，则它只是 manual-linear baseline，不能作为 PureKAN success。

### Case F：没有任何 near-pass

如果 DWM2 和 reset primitive 都没有 near-pass，则必须进入 terminal lower-level custom kernel route 或重新设计 primitive family。

---

## 18. 最终建议

v6.7 更新版的一句话策略是：

$$
\boxed{
\text{用 FlashKAT 的诊断思路，把 DWM2 backward 的 coefficient-gradient accumulation 当作第一 hot path 来审计和重构。}
}
$$

不要再只问：

```text
cache 是否保存 hidden activation？
```

而要问：

```text
coefficient-gradient accumulation 是否 materialize 了巨大临时张量？
是否造成 global memory traffic？
是否存在 reduction / atomic-like bottleneck？
是否能用 local reduction / two-stage reduction / fused dx+coeffgrad 降低 peak 与 time？
```

如果 FlashDWM2 成功，DWM2-poly2 仍然有希望成为第一个 memory-pass graph-free PureKAN primitive。

如果 FlashDWM2 失败，而且 bounded-workspace reset 也失败，则 v6.7 应诚实给出结论：

```text
当前 Python/Torch manual PureKAN abstraction 不足以超过 MLP；
需要 Triton/CUDA full custom kernel 或不同 primitive family。
```
