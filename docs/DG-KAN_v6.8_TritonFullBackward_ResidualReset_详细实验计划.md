# DG-KAN v6.8：Nsight-Driven True FlashDWM2 Full Backward Kernel 与 Residual-Effective Bounded Primitive Reset 详细实验计划

> 本计划基于 v6.7 final real-only run 的真实结果制定。v6.7 已经确认：`DWM2-current` 仍保持 strict PureKAN / graph-free / manual gradient correctness，但 memory/time gate 继续失败；Torch-level FlashDWM2 local-reduce / two-stage / fused-dx 版本梯度正确却 memory 基本不降、time 明显变慢；bounded-workspace reset 中 `TinyChannelResidual` 的 residual effect 过弱，不能作为 PureKAN success。因此 v6.8 不允许打开 task、LightSmooth 或 functional correction，而必须进入 **Nsight 级真实 bottleneck 归因**、**Triton/CUDA 级 full backward kernel** 与 **residual-effective bounded primitive reset**。

---

## 0. 实验整体目标

v6.8 的整体目标不是继续跑更多 optimizer，也不是尝试用 task accuracy 掩盖底层失败。v6.8 的目标是完成一次关键判别：

$$
\boxed{
\text{当前 graph-free DWM2-poly2 是否能通过真实底层 kernel 修复进入 memory/time near-pass？}
}
$$

并在 DWM2 不能修复时回答：

$$
\boxed{
\text{是否存在 residual 非平凡且 bounded-workspace 的新 PureKAN primitive 可以接替 DWM2？}
}
$$

v6.7 给出的事实非常清楚。`A2-DWM2-current` 的 memory ratio 是：

```text
min  = 1.1441
mean = 1.2918
max  = 1.4866
```

step ratio 是：

```text
min  = 1.5965
mean = 1.9002
max  = 2.2020
```

backward ratio mean 是 `1.4751`，但 `grad relerr max = 7.30e-08`，说明梯度正确性不是 blocker。当前问题是 memory/time 与 attribution。`A2-DWM2-current` 的 attribution pass 是 `0/18`，最终 route 是 `R3-AttributionIncomplete`。

v6.7 还验证了 FlashKAT 启发的 Torch-level FlashDWM2 路径，但结果是负的。P2.5 中：

```text
F1-local-reduce:
  time ratio vs current mean = 2.8876
  memory ratio vs current mean = 1.0000

F2-two-stage-reduce:
  time ratio vs current mean = 3.3559
  memory ratio vs current mean = 0.9957

F5-fused-dx-coeffgrad:
  time ratio vs current mean = 5.0135
  memory ratio vs current mean = 0.9929
```

它们梯度正确，但没有达到 FlashDWM2 pass。full-step packages 也全部更差，没有 S0/S1/S2 survivor。因此 v6.8 必须区分：

$$
\boxed{
\text{FlashKAT-style idea 未被证伪；Torch-level 模拟 FlashKAT 被证伪。}
}
$$

v6.8 的最低工程成功目标是找到至少一个真实 candidate 满足：

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

v6.8 的正式 task re-entry 标准是：

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

本轮不把 $0.80$ 作为唯一成功 gate。$0.80$ 仍是终极目标，但 v6.8 更现实的目标是：要么产生一个 memory/time near-pass kernel candidate，要么给出 DWM2 当前路线必须切换到 reset primitive 或 terminal custom kernel 的明确证据。

---

## 1. 当前实验进展与问题判断

### 1.1 实验纪律已经足够可信

v6.7 final run 是 real-only run。P1/P2/P2.5/P3/P4 的 `fake_data_used` 与 `proxy_row_used` 都是 0；Nsight / lower-level counters 不可用时写为 `metric_unavailable`，没有填假数；未实现 CUDA/Triton package 写为 `not_implemented`；P5/P6/P7 因 gate 写为 `not_run`。因此当前结论可信，不是 proxy 结果。

### 1.2 DWM2 仍然结构正确，但 efficiency 卡住

DWM2-current 仍然有三个正向属性：

```text
nonKAN = 0
manual backward = 1
grad relerr max = 7.30e-08
```

这说明路线的数学与结构没有坏。失败集中在：

```text
F1_memory_fail
F2_step_time_fail
F4_attribution_incomplete
F14_coeffgrad_reduction_no_effect
```

没有出现 fake/proxy failure，也没有出现 current DWM2 gradient correctness failure。

### 1.3 Torch-level FlashDWM2 负结果说明实现层级不够低

v6.7 中 Torch-level local/two-stage/chunked reduction 梯度正确，但 time 明显变慢，memory 只下降不到 1%。这说明：

$$
\boxed{
\text{用 Python/Torch chunk loop 模拟 FlashKAT 不是有效 kernel restructuring。}
}
$$

FlashKAT 的有效点是减少 global memory traffic、atomic/reduction 写回、memory stalls，而不是把 reduction 拆成更多 PyTorch op。v6.7 的 Torch-level FlashDWM2 反而增加 kernel 数量与 Python/Torch overhead，导致 full-step step ratio 上升到 3 左右。

### 1.4 reset diagnostic 没有形成新主线

v6.7 P4 中：

```text
B0-ManualLinear-reference:
  memory ratio mean = 1.1085
  step ratio mean = 1.1578
  residual_over_base = 0

B1-ManualLinear+TinyChannelResidual-edgeOwned:
  memory ratio mean = 1.2981
  step ratio mean = 1.8838
  residual_over_base mean = 0.0010

B2-OneBufferPoly1Residual:
  memory ratio mean = 1.2981
  step ratio mean = 1.8808
  residual_over_base mean = 0.0010

B9-FlashTinyResidual:
  memory ratio mean = 1.3734
  step ratio mean = 3.0384
  residual_over_base mean = 0.0010
```

reset 没有 near-pass，也没有 residual-effect pass。`residual_over_base = 0.001` 远低于计划要求 $0.02$。因此 reset 不能打开 task，也不能作为 KAN success。

### 1.5 当前是否在正确道路上

当前道路仍然正确，但必须升级层级。更准确地说：

$$
\boxed{
\text{graph-free PureKAN 方向没有被证伪；DWM2 的 Python/Torch manual implementation 已经基本被证伪为不足。}
}
$$

下一步不应该继续在 PyTorch 层做 chunk/reduce，也不应该打开 optimizer exploration。正确道路是：

```text
真实 Nsight / allocator trace
Triton/CUDA full backward kernel
residual-effective bounded reset primitive
```

如果这三者仍然失败，就应承认当前 primitive family 不能在现有层级达到目标。

### 1.6 离目标还差多远

按 v6.7 current mean 看，memory 从 `1.2918` 降到 `<1.0` 需要：

$$
1-\frac{1.0}{1.2918}\approx22.6\%.
$$

从 current mean 降到终极 $0.8$ 需要：

$$
1-\frac{0.8}{1.2918}\approx38.1\%.
$$

step ratio 从 current mean `1.9002` 降到 task re-entry 标准 `1.35` 需要：

$$
1-\frac{1.35}{1.9002}\approx29.0\%.
$$

降到强目标 `1.20` 需要：

$$
1-\frac{1.20}{1.9002}\approx36.8\%.
$$

所以当前不是“小修小补差一点”。DWM2 要继续必须出现实质 kernel breakthrough。reset primitive 则需要同时解决 memory/time 与 residual effect，目前离 residual 非平凡 gate 也很远：

$$
\frac{0.001}{0.02}=0.05.
$$

也就是 residual strength 只有最低要求的约 $5\%$。

---

## 2. v6.8 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio、derived rows 作为结果。任何未真实测量的 package 必须写为：

```text
not_implemented
not_applicable
not_run
metric_unavailable
```

第二，P1-P4 没有 S0/S1/S2 survivor 时，不允许打开：

```text
task re-entry
LightSmooth
functional correction
3-seed confirm
5-seed confirm
10-seed confirm
```

第三，不允许把 Torch-level chunked reduction 继续命名为 FlashDWM2 success。只有 Triton/CUDA 或等价 fused kernel 能进入 FlashDWM2 success path。

第四，不允许只看 memory ratio，不记录 Nsight / allocator / kernel 指标。如果 Nsight 不可用，必须至少用 `torch.cuda.memory_snapshot`、NVTX ranges 和 profiler memory stack 产出可行动 attribution。

第五，不允许把 `ManualLinear+TinyChannelResidual` 当成 PureKAN success，除非它通过 residual-effect gate：

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02.
$$

第六，不允许在 residual effect 不过时进行 task training。否则 task 结果可能只是 manual-linear baseline 的结果。

第七，不允许把 optimizer exploration 作为 v6.8 主线。v6.8 之前的 blocker 是 memory/time/kernel，不是 optimizer。

---

## 3. 核心假设

### H1：current DWM2 attribution incomplete 是因为缺少 Nsight/stack-level trace，而不是 peak 不可解释

v6.7 的 current DWM2 attribution pass 是 `0/18`。H1 假设：通过 Nsight Systems / Nsight Compute、NVTX range、torch memory snapshot、allocation stack hashing，可以解释当前 DWM2-over-MLP peak gap 的至少 $95\%$。

定义：

$$
G_{\text{peak}}=M_{\text{peak,DWM2}}-M_{\text{peak,MLP}}.
$$

如果 attribution 项为 $A_i$，则要求：

$$
\frac{\sum_i A_i}{G_{\text{peak}}}\geq0.95.
$$

并且 top-3 source 要解释至少 $70\%$：

$$
\frac{A_1+A_2+A_3}{G_{\text{peak}}}\geq0.70.
$$

H1 成立还要求输出具体 blocker 类型：

```text
coeffgrad_contribution_tensor
coeffgrad_reduction_temp
delta_lifetime
poly_temp_materialization
update_workspace
allocator_padding
workspace_pool_overlap
kernel_boundary_copy
optimizer_state_overlap
global_memory_stall
```

如果 H1 不成立，route 必须保持 `R3-AttributionIncomplete`，不允许做 repair success claim。

---

### H2：DWM2 的 bottleneck 可能不在 coeffgrad 本身，而在 full backward live-set overlap

v6.7 的 Torch-level coeffgrad variants 在 microkernel 中没有 memory/time 改善，full-step 更差。但这并不排除 coefficient-gradient 是 full backward live-set 的一部分。H2 假设：单独优化 coeffgrad 不够，必须 fusion：

$$
\delta_z \rightarrow \delta_x + g_{\theta} + update\_temp
$$

从而减少 $\delta_z$、$\delta_x$、coefficient contribution、update workspace 同时 alive 的区间。

H2 成立标准是 full fused backward package 相对 current 满足：

$$
\frac{M_{\text{peak,fused}}}{M_{\text{peak,current}}}\leq0.90,
$$

且：

$$
\frac{T_{\text{step,fused}}}{T_{\text{step,current}}}\leq0.90.
$$

如果单独 coeffgrad 改善但 full package 不改善，则说明 bottleneck 在 live-set overlap，不在单组件速度。

---

### H3：真正的 FlashDWM2 必须是 Triton/CUDA fused kernel，而不是 Python/Torch chunk loop

v6.7 的 negative result 是 Torch-level reduction negative，不是 FlashKAT-style kernel negative。H3 假设：用 Triton/CUDA 实现 local reduction、two-stage reduction 或 fused dx+coeffgrad，可以减少 global memory traffic 与 kernel launch overhead。

H3 成立的 microkernel 标准是：

$$
\frac{T_{\text{coeffgrad,triton}}}{T_{\text{coeffgrad,current}}}\leq0.50,
$$

$$
\frac{M_{\text{coeffgrad,triton}}}{M_{\text{coeffgrad,current}}}\leq0.70.
$$

full package 标准是：

$$
\frac{M_{\text{backward,triton}}}{M_{\text{backward,MLP}}}\leq1.05,
$$

$$
\frac{T_{\text{step,triton}}}{T_{\text{step,MLP}}}\leq1.50.
$$

数值标准必须保持：

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

---

### H4：reset primitive 的关键不是更小 residual，而是 residual 非平凡且 bounded workspace

v6.7 的 tiny residual 失败是因为 residual effect 太弱：

$$
\frac{\|r(x)\|}{\|x\|}\approx0.001.
$$

H4 假设：bounded-workspace reset primitive 必须用可控 residual scale、edge-owned residual 参数和 streaming backward，使 residual 强度达到最低有效阈值，同时保持 memory/time near-pass。

H4 成立要求：

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02,
$$

且：

$$
|\Delta L_{\text{residual ablation}}|>10^{-4}
$$

或：

$$
|\Delta \text{logit}_{\text{residual ablation}}|>10^{-4}.
$$

同时满足 near-pass：

$$
\frac{M_{\text{backward}}}{M_{\text{MLP}}}\leq1.05,
$$

$$
\frac{T_{\text{step}}}{T_{\text{MLP}}}\leq1.35.
$$

如果 residual 非平凡后 memory/time 立刻接近 DWM2 失败水平，说明 bounded reset 的 expressivity-cost tradeoff 仍未解决。

---

### H5：current graph-free PureKAN 的 optimizer 探索应继续 gate，直到有 memory/time survivor

H5 是门禁假设。当前 optimizer 探索并非不重要，但在 memory/time 未过线时继续跑 optimizer 会导致 wall-clock 结果不可解释。H5 要求：

```text
No P6 task re-entry unless S0/S1 exists
No P7 functional unless P6 task pass exists
No optimizer sweep beyond one-step probe unless memory/time gate passes
```

### H6：如果 DWM2 Triton full backward 与 residual-effective reset 都失败，项目需要转入 primitive-family redesign

H6 是终局判定。若 DWM2 true fused kernel 失败，reset primitive 也不能同时满足 residual effect 与 memory/time，则当前路线不能靠小修继续。应进入：

```text
new primitive family
single-buffer KAN residual
table-free local polynomial basis
hardware-aware rational primitive
or full custom CUDA KAN layer
```

---

## 4. 实验阶段总览

v6.8 分为十个阶段：

```text
P0: v6.7 reproduction and implementation contract
P1: Nsight / allocation-stack attribution for current DWM2
P2: true Triton/CUDA FlashDWM2 microkernel audit
P3: full backward fused DWM2 package
P4: residual-effective bounded-workspace reset
P5: one-step numerical and residual-effect probe
P6: limited task re-entry gate
P7: optimizer exploration remains gated
P8: functional correction remains gated
P9: route decision
P10: artifact and failure audit
```

P1-P4 是核心。P5 只在 P3/P4 有 S0/S1/S2 或 reset near-pass 时运行。P6/P7/P8 默认关闭。

---

## 5. P0：v6.7 reproduction and implementation contract

### 5.1 目的

确认 v6.8 与 v6.7 baseline 可比较，确认新增 Triton/CUDA kernel 或 reset primitive 是真实实现，不是 proxy。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-current
DWM2-FlashTorch-local-reduce, from v6.7
DWM2-FlashTorch-two-stage, from v6.7
DWM2-FlashTorch-fused-dx, from v6.7
FlashKAT-third-party-smoke
DWM2-Triton-coeffgrad-v1, if implemented
DWM2-Triton-fused-dx-coeffgrad-v1, if implemented
DWM2-Triton-full-backward-v1, if implemented
ResidualEffectiveTinyKAN-v1, if implemented
```

### 5.3 必须记录字段

```text
variant
status
implementation_type
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
uses_custom_autograd_function
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
gradcheck_available
v67_memory_ratio_mean
v68_memory_ratio_mean
v67_step_ratio_mean
v68_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
```

### 5.4 判断标准

P0 pass 要求：

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0 for PureKAN candidates
rollback_max_error < 1e-8
```

reproduction 要求：

$$
|r_{\text{memory,v68}}-r_{\text{memory,v67}}|\leq0.05,
$$

$$
|r_{\text{step,v68}}-r_{\text{step,v67}}|\leq0.15.
$$

如果 P0 reproduction fail，后续结果只能作为 diagnostic，不能和 v6.7 直接比较。

### 5.5 可视化

```text
p0_reproduction_delta_bar.svg
p0_contract_heatmap.svg
p0_kernel_implementation_status.svg
```

---

## 6. P1：Nsight / allocation-stack attribution for current DWM2

### 6.1 目的

P1 是 v6.8 的第一核心。v6.7 route 是 `R3-AttributionIncomplete`，所以 P1 必须给出 actionable attribution。

P1 要回答：

```text
current DWM2 的 peak source 是什么？
coeffgrad 是否真是 top source？
global memory stall 是否存在？
allocator padding / fragmentation 是否是主因？
update workspace / optimizer state 是否与 backward overlap？
```

### 6.2 Trace phase

必须使用 NVTX 或等价 range：

```text
phase_forward_transform
phase_forward_mix
phase_loss_delta
phase_backward_output
phase_backward_delta
phase_backward_dx
phase_backward_coeffgrad
phase_backward_update_prep
phase_update_params
phase_optimizer_state
phase_cleanup
```

### 6.3 工具要求

至少使用：

```text
torch.profiler(profile_memory=True)
torch.cuda.memory_snapshot
NVTX phase ranges
```

如果可用，必须使用：

```text
Nsight Systems
Nsight Compute
```

Nsight 指标至少尝试采集：

```text
stall_long_scoreboard
stall_memory_dependency
dram_read_bytes
dram_write_bytes
l2_read_transactions
l2_write_transactions
sm_occupancy
achieved_occupancy
register_spill_count
shared_memory_bytes
atomic_transactions
global_store_transactions
global_load_transactions
```

不可用时写为 `metric_unavailable`。

### 6.4 测量 grid

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

完整 summary grid：

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

### 6.5 必须记录字段

#### phase fields

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

#### allocation stack fields

```text
allocation_id
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
is_coeffgrad_contribution
is_coeffgrad_reduction_temp
is_delta
is_dx
is_poly_temp
is_update_temp
is_optimizer_state
is_workspace_pool
is_manual_cache
```

#### Nsight / stall fields

```text
phase_dram_read_bytes
phase_dram_write_bytes
phase_l2_read_transactions
phase_l2_write_transactions
phase_atomic_transactions
phase_global_store_transactions
phase_global_load_transactions
phase_stall_long_scoreboard
phase_stall_memory_dependency
phase_sm_occupancy
phase_register_spill_count
phase_shared_memory_bytes
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
top1_fraction
top2_source
top2_MB
top2_fraction
top3_source
top3_MB
top3_fraction
coeffgrad_fraction_of_gap
delta_fraction_of_gap
update_fraction_of_gap
allocator_padding_fraction_of_gap
stall_top_reason
```

### 6.6 判断标准

P1 pass 要求：

$$
\frac{M_{\text{explained}}}{G_{\text{peak}}}\geq0.95,
$$

$$
\frac{M_{\text{top3}}}{G_{\text{peak}}}\geq0.70.
$$

同时必须识别 primary blocker：

```text
coeffgrad_materialization
delta_dx_overlap
update_workspace_overlap
allocator_fragmentation
global_memory_stall
kernel_launch_fragmentation
GEMM_dominant_not_repairable
```

如果 P1 仍不能解释 current DWM2 peak，route 必须保持 `R3-AttributionIncomplete`，并且不能声明任何 repair 方向已经定位。

### 6.7 可视化

必须生成：

```text
p1_phase_peak_waterfall.svg
p1_tensor_lifetime_gantt_top20.svg
p1_allocation_stack_top10.md
p1_gap_attribution_stacked_bar.svg
p1_nsight_stall_dashboard.svg
p1_coeffgrad_vs_delta_gap_fraction_heatmap.svg
p1_allocator_padding_scatter.svg
```

---

## 7. P2：true Triton/CUDA FlashDWM2 microkernel audit

### 7.1 目的

P2 只验证真正的 lower-level FlashDWM2 microkernel。Torch-level chunk loop 不再作为 success candidate。

### 7.2 Microkernels

```text
K0-current-coeffgrad-torch
K1-triton-coeffgrad-local-reduce
K2-triton-coeffgrad-two-stage-reduce
K3-triton-fused-dx-coeffgrad
K4-triton-fused-dx-coeffgrad-update-prep
K5-triton-transform-derivative
K6-triton-delta-dx-only
K7-cuda-extension-coeffgrad, optional
K8-cuda-extension-full-backward, optional
```

### 7.3 数学定义

对于 poly2 residual：

$$
r(x)=a_1x+a_2x^2.
$$

$$
z=x+s r(x).
$$

给定 $\delta_z$：

$$
\frac{\partial L}{\partial a_1}
=
\sum_b \delta_{z,b}s x_b,
$$

$$
\frac{\partial L}{\partial a_2}
=
\sum_b \delta_{z,b}s x_b^2,
$$

$$
\delta_x
=
\delta_z(1+s(a_1+2a_2x)).
$$

Triton/CUDA kernel 必须尽量避免 materialize：

$$
C_{b,i,k}
=
\delta_{z,b,i}
\frac{\partial r_i(x_{b,i})}{\partial \theta_{i,k}}.
$$

### 7.4 必须记录字段

```text
microkernel
implementation
input_shape
dtype
block_size
num_warps
shared_memory_bytes
registers_per_thread
coeffgrad_time_ms
dx_time_ms
combined_time_ms
peak_allocated_MB
peak_reserved_MB
temp_allocated_MB
global_load_bytes
global_store_bytes
atomic_transactions
l2_read_transactions
l2_write_transactions
dram_read_bytes
dram_write_bytes
stall_long_scoreboard
achieved_occupancy
register_spill_count
grad_relerr_vs_torch
grad_cos_vs_torch
grad_relerr_vs_autograd
grad_cos_vs_autograd
rounding_MAE_vs_fp64
rounding_max_abs_vs_fp64
```

### 7.5 判断标准

Microkernel pass 要求：

$$
\frac{T_{\text{micro,new}}}{T_{\text{micro,current}}}\leq0.50,
$$

或：

$$
\frac{M_{\text{micro,new}}}{M_{\text{micro,current}}}\leq0.70.
$$

同时数值必须满足：

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

如果 Nsight 指标可用，memory stall 应改善至少一项：

$$
\frac{\text{dram\_write}_{new}}{\text{dram\_write}_{current}}\leq0.70,
$$

或：

$$
\frac{\text{stall\_long\_scoreboard}_{new}}
{\text{stall\_long\_scoreboard}_{current}}\leq0.70.
$$

### 7.6 可视化

```text
p2_triton_microkernel_pareto.svg
p2_coeffgrad_time_memory_bar.svg
p2_memory_traffic_reduction.svg
p2_stall_reduction_bar.svg
p2_grad_error_vs_speed.svg
p2_tuning_heatmap_blocksize_numwarps.svg
```

---

## 8. P3：full backward fused DWM2 package

### 8.1 目的

P3 将 P2 有效 microkernel 组合成完整 DWM2 backward package，重新测 full-step memory/time。

### 8.2 Candidate packages

```text
D0-current
D1-triton-coeffgrad-only
D2-triton-fused-dx-coeffgrad
D3-triton-transform-derivative+coeffgrad
D4-triton-full-backward-light
D5-triton-full-backward-onebuffer
D6-cuda-full-backward, optional
```

定义：

```text
full-backward-light =
  fused derivative + dx + coeffgrad

full-backward-onebuffer =
  full-backward-light + update prep reuse + one live residual buffer
```

### 8.3 Grid

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

### 8.4 必须记录字段

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
backward_ratio_min
backward_ratio_mean
backward_ratio_max
forward_ratio_mean
memory_improvement_vs_current
step_improvement_vs_current
backward_improvement_vs_current
kernel_count_reduction
allocation_count_reduction
dram_write_reduction
stall_reduction
grad_relerr_max
grad_cos_min
rounding_MAE_vs_fp64
peak_gap_explain_ratio
top_gap_source_after_repair
```

### 8.5 Survivor 类型

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
  microkernel pass but full package no gain
```

S0/S1 可以进入 P5 和 P6。  
S2 只能进入 P5 one-step probe，不能 full task。  
S6 要继续 fusion，不开 task。

### 8.6 可视化

```text
p3_package_memory_step_pareto.svg
p3_s0_s1_s2_threshold_plot.svg
p3_full_package_improvement_bar.svg
p3_batch_depth_stability_heatmap.svg
p3_grad_correctness_bar.svg
p3_memory_traffic_after_repair.svg
```

---

## 9. P4：residual-effective bounded-workspace reset

### 9.1 目的

P4 不是重复 v6.7 的弱 tiny residual。v6.8 的 reset 必须同时满足：

```text
residual 非平凡
bounded workspace
manual gradient correctness
memory/time near-pass
```

### 9.2 Reset candidates

```text
B0-ManualLinear-reference

B1-TinyResidual-scale002:
  residual target ratio around 0.02

B2-TinyResidual-scale005:
  residual target ratio around 0.05

B3-OneBufferPoly1Residual-scale002

B4-OneBufferPoly1Residual-scale005

B5-OneBufferPiecewiseLinear2-scale002

B6-OneBufferFastRational-scale002

B7-ChunkedMixingResidualKAN

B8-ResidualOnlyAblationProbe
```

所有 reset candidate 必须保证：

```text
nonKAN_param_count = 0
edge-owned residual params
manual backward available
no full layer delta retention
one live residual buffer if possible
```

### 9.3 必须记录字段

```text
primitive
scale_target
scale_actual
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
near_pass_count
residual_effect_pass_count
```

### 9.4 判断标准

Reset near-pass：

$$
\frac{M_{\text{backward}}}{M_{\text{MLP}}}\leq1.05,
$$

$$
\frac{T_{\text{step}}}{T_{\text{MLP}}}\leq1.35,
$$

$$
\operatorname{grad\_relerr}<10^{-4}.
$$

Residual effect pass：

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02,
$$

and either:

$$
|\Delta L_{\text{residual ablation}}|>10^{-4},
$$

or:

$$
|\Delta \text{logit}_{\text{residual ablation}}|>10^{-4}.
$$

If residual effect fails, candidate is classified as linear-like and cannot enter task.

### 9.5 可视化

```text
p4_reset_memory_time_pareto.svg
p4_residual_strength_vs_memory.svg
p4_residual_ablation_delta.svg
p4_scale_sweep_pareto.svg
p4_reset_workspace_comparison.svg
```

---

## 10. P5：one-step numerical and residual-effect probe

P5 只对 P3/P4 的 S0/S1/S2 或 reset near-pass 运行。

### 10.1 Probe 方法

```text
clone weights
compute manual gradient
apply one update
measure train loss before/after
measure holdout loss before/after
measure val mini-batch loss before/after
measure residual ablation before/after
rollback weights
check rollback exactness
```

### 10.2 必须记录

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

Residual contribution 要求：

$$
|\Delta L_{\text{residual ablation}}|>10^{-4}
$$

or:

$$
|\Delta \text{logit}_{\text{residual ablation}}|>10^{-4}.
$$

### 10.4 可视化

```text
p5_before_after_loss_plot.svg
p5_bad_step_heatmap.svg
p5_update_norm_vs_loss_delta.svg
p5_residual_ablation_plot.svg
p5_rollback_error_bar.svg
```

---

## 11. P6：limited task re-entry gate

### 11.1 打开条件

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
  Best-DWM2-triton-package + ManualAdanLite
  Best-reset-primitive + ManualAdanLite
  Best-reset-primitive + ManualAdamW
```

如果 diagnostic-only：

```text
seed = 0 only
short budget only
no final claim
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
forward_time_ratio
backward_time_ratio
samples_per_second
margin_mean
margin_p10
feature_rank
classwise_acc
seedwise_failure_reason
```

### 11.4 判断标准

正式 task pass 要求：

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.01
$$

on all datasets, and at least two datasets satisfy:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}.
$$

Memory must hold:

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}<1.0.
$$

Wall-clock must satisfy:

$$
\operatorname{ValLossAUC}_{time,KAN}\leq\operatorname{ValLossAUC}_{time,MLP}
$$

on at least two datasets.

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

P7 不是 functional correction，而是 optimizer exploration。它默认关闭。只有 P6 task re-entry pass 后，才系统比较 optimizer。

如果打开，只允许比较：

```text
ManualAdamW
ManualAdanLite
ManualWinLite
Lookahead-ManualAdamW
WarmupCosine-ManualAdamW
```

不得重新打开大规模 role-wise LR sweep，除非 P6 显示 role update share 明显异常。

### 12.1 必须记录

```text
optimizer_name
train_loss_curve
val_loss_curve
time_to_target_loss
time_to_target_acc
val_loss_auc_by_time
test_acc
ECE
NLL
step_time_ms
optimizer_state_MB
momentum_norm
velocity_norm
update_norm
update_over_param
cos_update_grad
cos_update_prev
```

### 12.2 判断标准

Optimizer pass 要求：

$$
T_{\text{target,optimizer}} \leq 0.90 T_{\text{target,ManualAdamW}},
$$

or:

$$
\operatorname{ValLossAUC}_{time,optimizer}
<
\operatorname{ValLossAUC}_{time,ManualAdamW}.
$$

同时：

$$
\operatorname{Acc}_{optimizer}\geq \operatorname{Acc}_{ManualAdamW}-0.005.
$$

---

## 13. P8：functional correction remains gated

P8 只有在 P6 task pass 和 P7 optimizer stable 后才允许 smoke。默认关闭。

### 13.1 Minimal functional smoke

```text
Task-only best candidate
Task + small residual geometry correction
Task + low-frequency LightSmooth event
```

### 13.2 必须记录

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

### 13.3 判断标准

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

Correction overhead:

$$
\frac{T_{\text{corrected}}}{T_{\text{task-only}}}\leq1.10.
$$

---

## 14. P9：route decision

### Route cases

```text
R1-DWM2TritonSolved:
  Triton/CUDA DWM2 full backward gets S0/S1 and P5 pass.
  Open limited task.

R2-DWM2TritonNearPass:
  DWM2 gets S2 with clear memory/time improvement.
  Continue kernel engineering, no full task.

R3-AttributionIncomplete:
  Nsight/allocator trace still cannot explain peak gap.
  Improve profiler.

R4-DWM2RepairNoEffect:
  Triton/CUDA repairs do not improve memory/time.
  DWM2 current memory model likely exhausted.

R5-ResetPrimitiveCandidate:
  bounded-workspace reset gets near-pass and residual effect pass.
  Switch next cycle to reset primitive.

R6-ResetLinearOnly:
  reset near-passes but residual effect fails.
  It is a baseline, not KAN success.

R7-TerminalCustomKernelNeeded:
  no DWM2 survivor and no reset near-pass.
  Need new primitive family or full custom layer.

R8-TaskReentryPass:
  memory/time survivor passes limited task.
  Next cycle can consider 5-seed confirm.

R9-TaskReentryFail:
  memory/time survivor exists but task fails.
  Need expressivity repair.

R10-OptimizerExplorationOpened:
  task re-entry passes and optimizer exploration is now meaningful.

R11-FunctionalCorrectionOpened:
  optimizer stable and functional correction smoke is now meaningful.
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
triton_kernel_pass
reset_residual_effect_pass
fallback_triggered
fallback_near_pass_count
open_task_reentry
open_optimizer_exploration
open_functional_correction
no_fake
no_proxy
primary_blocker
next_required_implementation
```

---

## 15. P10：artifact and failure audit

### 15.1 failure types

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_attribution_incomplete
F5_triton_compile_fail
F6_triton_kernel_no_effect
F7_triton_breaks_gradient
F8_residual_effect_fail
F9_reset_no_near_pass
F10_task_fail
F11_optimizer_gate_fail
F12_functional_gate_fail
F13_fake_or_proxy_violation
F14_gated_not_run
F15_artifact_missing
```

### 15.2 必须生成 artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p1_nsight_allocation_trace.csv
p1_tensor_lifetime_topk.csv
p1_phase_peak_summary.csv
p1_nsight_stall_summary.csv
p2_triton_microkernel_audit.csv
p2_triton_microkernel_correctness.csv
p3_full_backward_packages.csv
p3_full_backward_package_detail.csv
p4_residual_effective_reset.csv
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

### 15.3 必须生成可视化

```text
figures/p1_phase_peak_waterfall.svg
figures/p1_tensor_lifetime_gantt.svg
figures/p1_nsight_stall_dashboard.svg
figures/p1_gap_attribution_stacked_bar.svg
figures/p2_triton_microkernel_pareto.svg
figures/p2_memory_traffic_reduction.svg
figures/p2_grad_error_vs_speed.svg
figures/p3_package_memory_step_pareto.svg
figures/p3_batch_depth_stability_heatmap.svg
figures/p4_residual_strength_vs_memory.svg
figures/p4_residual_ablation_delta.svg
figures/p5_before_after_loss_plot.svg
figures/p6_task_efficiency_pareto.svg
figures/p9_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 16. 成功与失败解释规则

### Case A：DWM2 Triton full backward pass

如果：

$$
M_{\text{backward}}/M_{\text{MLP}}<1.0
$$

且：

$$
T_{\text{step}}/T_{\text{MLP}}\leq1.35,
$$

则 DWM2 继续作为主线，打开 P5/P6。

### Case B：DWM2 Triton near-pass

如果：

$$
M_{\text{backward}}/M_{\text{MLP}}\leq1.05
$$

且：

$$
T_{\text{step}}/T_{\text{MLP}}\leq1.50,
$$

且 memory improvement $\geq10\%$，则继续 kernel 工程，但不做 full task confirm。

### Case C：Triton microkernel 有效，full package 无效

说明局部 coeffgrad 不是唯一问题，live-set overlap 或 update workspace 仍是 blocker。下一轮应 fusion more of backward, not task.

### Case D：DWM2 Triton 无效

如果 Triton full backward 改善小于 $5\%$，则 DWM2 当前 memory model 基本耗尽，转 P4 reset 或新 primitive family。

### Case E：reset near-pass 且 residual effect pass

下一轮以 reset primitive 为主线。

### Case F：reset near-pass 但 residual effect fail

这是 manual-linear baseline，不是 PureKAN success。

### Case G：无任何 near-pass

进入 terminal route：需要新 primitive family 或 full custom KAN layer，不再继续 DWM2 task/functional。

---

## 17. 最终建议

v6.8 的一句话方向是：

$$
\boxed{
\text{从 Torch-level graph-free prototype 升级到真实 Triton/CUDA full backward kernel，}
}
$$

并同步验证：

$$
\boxed{
\text{bounded-workspace reset 是否能在 residual 非平凡时仍保持 memory/time near-pass。}
}
$$

v6.7 的负结果已经说明：

```text
Torch-level FlashDWM2 reduction 不够；
current DWM2 attribution 仍 incomplete；
tiny residual reset 太弱；
task 和 functional 继续 gate 是正确的。
```

因此 v6.8 只应该相信三类结果：

```text
Nsight/allocator trace 是否解释 peak；
Triton/CUDA kernel 是否降低 actual CUDA peak 与 step time；
reset primitive 是否同时具备 residual effect 和 bounded workspace。
```

如果三者都没有突破，就应停止在当前 DWM2-poly2 路线上继续做 optimizer 或 functional correction，转向新的 primitive family。
