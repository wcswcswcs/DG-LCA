# DG-KAN v6.6：Workspace Kernel Attribution、真实 Buffer Reuse 与 Primitive Reset 决策实验计划

> 本计划基于 v6.5 final real-only run 的真实结论制定。v6.5 已经证明：当前 `DWM2-poly2-compiled-current` 仍满足 strict PureKAN / graph-free / manual backward / no-fake contract，manual gradient correctness 可靠；但 memory/time gate 均未通过。`noHiddenCache` 真实消除了 hidden cache，却几乎没有降低 total backward peak；`bf16Cache` 不仅没有降低 total CUDA peak，还导致 gradient relative error 达到 $10^{-3}$ 量级；P9 fallback primitive 全部没有 near-pass。因此 v6.6 不允许直接重开 task、LightSmooth 或 functional correction，而必须先回答一个更底层的问题：当前峰值到底来自哪个 workspace / allocator / tensor lifetime，真实 buffer reuse 或 delta streaming 能不能把它压下去。

---

## 0. 实验整体目标

v6.6 的整体目标不是继续证明 DG-KAN 可以训练，也不是继续调 optimizer。v6.6 的目标是完成一次 **workspace-kernel 级别的闭环诊断与修复**：

$$
\boxed{\text{定位 DWM2-poly2 graph-free path 的真实 CUDA peak 来源，}}
$$

$$
\boxed{\text{实现真实 buffer reuse / delta streaming / fused workspace policy，}}
$$

$$
\boxed{\text{判断 DWM2-poly2 是否仍值得作为终极 PureKAN primitive 主线。}}
$$

v6.5 的最终 route 是 `R6`，这意味着当前 DWM2-poly2 与 fallback primitive 都没有得到 memory-pass 或 near-pass candidate。下一步如果继续跑 task seed、LightSmooth 或 functional correction，会绕开真正 blocker。因此 v6.6 的任务顺序必须固定为：

```text
先做 phase-local attribution -> 再做真实 workspace repair -> 再做 combined package -> 最后决定 task re-entry 或 primitive reset。
```

v6.6 的最低成功目标是找到一个真实候选满足：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}<1.0,
$$

并且：

$$
\frac{T_{\text{step,KAN}}}{T_{\text{step,MLP}}}\leq1.35.
$$

v6.6 的更强目标是：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}\leq0.90,
$$

$$
\frac{T_{\text{step,KAN}}}{T_{\text{step,MLP}}}\leq1.20.
$$

最终目标仍然是：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}\leq0.80,
$$

但 v6.6 不把 $0.80$ 作为唯一 gate。v6.6 的科学目标是判断这条路线是否还能向 $0.80$ 收敛。

---

## 1. 当前事实基线

v6.5 final real-only run 给出的关键事实如下。

第一，当前 `DWM2-poly2-compiled-current` 仍然是 graph-free strict PureKAN 候选：`manual backward = 1`，`nonKAN = 0`，`fake/proxy = 0`。这说明结构纯度和训练路径纯度没有倒退。

第二，P1 中 `P1-current` 的 memory/time 结果为：

```text
memory ratio vs MLP:
  min  = 1.1441
  mean = 1.2918
  max  = 1.4866

step ratio vs MLP:
  min  = 1.6477
  mean = 1.9295
  max  = 2.1930

backward ratio vs MLP:
  min  = 1.2618
  mean = 1.4867
  max  = 1.7192

grad relerr max = 8.38e-08
grad cos min    = 1.0
```

这说明当前失败不是梯度错误，而是 memory/time 失败。

第三，`noHiddenCache` 的诊断结果为：

```text
manual_cache_MB mean:
  current       = 1.039
  noHiddenCache = 0.893

hidden_cache_MB mean:
  current       = 0.146
  noHiddenCache = 0.000

memory ratio mean:
  current       = 1.2918
  noHiddenCache = 1.2840

step ratio mean:
  current       = 1.9295
  noHiddenCache = 2.1442
```

这说明 hidden cache 被真实消除了，但 total CUDA peak 几乎没降，且 step time 变差。因此 hidden cache 不是主峰值来源。

第四，P2 中 `bf16Cache` 结果为：

```text
memory ratio mean:
  current   = 1.2918
  bf16Cache = 1.3674

grad relerr max:
  bf16Cache = 1.61e-03
```

这说明 naive cache compression 不是可用修复，它既没有降低 total peak，也破坏了 gradient correctness gate。

第五，P9 fallback primitive 没有 near-pass。最低 fallback memory ratio 是 `F0-DWM2-poly3` 的 `1.1673`，仍高于 near-pass 阈值，也高于 memory pass。SparseInterp fused rewrite 的 step/memory 更差，说明现有 fallback 并没有绕开当前 workspace/kernel 问题。

因此 v6.6 的工作假设是：

$$
\boxed{\text{当前 DWM2-family 的主要 blocker 是 workspace / allocator / delta lifetime，而不是 hidden cache。}}
$$

---

## 2. v6.6 的禁止事项

为了避免回到旧问题，本轮必须显式禁止以下行为。

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows 作为结果。任何未真实测量的 package 必须写为：

```text
not_implemented
not_applicable
not_run
```

第二，P1-P3 没有 memory/time survivor 时，不允许打开：

```text
P7 task re-entry
P8 functional correction smoke
LightSmooth integration
3-seed / 5-seed / 10-seed confirm
```

第三，不允许把 `manual_cache_MB` 的下降当作 `CUDA peak memory` 的下降。所有 memory gate 以 actual peak allocated / reserved 的真实 measurement 为准。

第四，不允许把 `bf16Cache` 作为主线继续推进，除非先解决：

$$
\operatorname{grad\_relerr}<10^{-4}
$$

并且证明 total CUDA peak 确实下降。

第五，不允许新增大量 task optimizer 搜索。v6.6 只有在 P1-P3 出现 memory/time survivor 后，才允许 task re-entry。

---

## 3. 核心假设

### H1：当前峰值主要来自 workspace / allocator lifetime，而不是 manual cache

v6.5 中 `manual_cache_MB` 只有约 $1$ MB，但 `workspace/temp/unexplained gap` 是 $20$ MB 级别。`noHiddenCache` 消除了 hidden cache 后 total memory ratio 只下降约 $0.58\%$。因此 H1 假设：真正峰值来自以下来源之一或组合：

```text
upstream delta lifetime
backward workspace temp
poly transform temporary
allocator block reuse failure
optimizer/update phase temporary
MLP baseline 与 KAN measurement phase 不一致
```

H1 成立的标准是：phase-local attribution 可以解释至少 $90\%$ 的 KAN-over-MLP peak gap，并能指出 top-3 tensor/op/lifetime 来源。

用公式表示，定义：

$$
G_{\text{peak}}=M_{\text{peak,KAN}}-M_{\text{peak,MLP}}.
$$

如果 attribution 分解项为 $A_i$，则要求：

$$
\frac{\sum_i A_i}{G_{\text{peak}}}\geq0.90.
$$

如果 attribution 只能说 `unexplained`，H1 不算完成，P1 必须继续加强 profiler。

---

### H2：真实 buffer reuse 可以降低 actual CUDA peak，而不是只降低估算 cache

v6.5 中 `bufferReuse` 未实现。H2 假设：如果使用预分配 workspace pool、固定 lifetime 的 ring buffer、复用 delta / temp / update buffer，可以降低 actual peak。

H2 成立的标准是：相比 current，真实 buffer reuse variant 必须满足：

$$
\Delta M_{\text{peak}} =
\frac{M_{\text{current}}-M_{\text{bufferReuse}}}{M_{\text{current}}}
\geq0.10,
$$

并且 step time 不能恶化超过 $10\%$：

$$
\frac{T_{\text{step,bufferReuse}}}{T_{\text{step,current}}}\leq1.10.
$$

如果 buffer reuse 只降低 `manual_cache_MB`，但 `peak_allocated_MB` 不下降，则 H2 不成立。

---

### H3：delta streaming 是比 noHiddenCache 更有效的 memory repair

`noHiddenCache` 的问题是通过 recompute 换 cache，导致 step time 变差，而且不降 total peak。H3 假设：真正应该处理的是 backward delta 的生命周期，而不是 hidden cache。

delta streaming 的核心是：

$$
\delta_{l-1}=J_l^T\delta_l
$$

在计算完成后立即释放或复用 $
\delta_l$，而不是保留多个 layer 的 delta / workspace。

H3 成立的标准是：deltaStreaming variant 相比 current 满足：

$$
\frac{M_{\text{backward,deltaStreaming}}}{M_{\text{backward,current}}}\leq0.90,
$$

并且：

$$
\frac{T_{\text{step,deltaStreaming}}}{T_{\text{step,current}}}\leq1.15.
$$

如果 memory 下降但 step time 增加到 $>1.25\times$ current，则只能作为 diagnostic，不进入 P3 combined package。

---

### H4：当前 step time 失败来自 elementwise / kernel launch / allocation，而不只是数学复杂度

DWM2-poly2 的数学计算很轻，但 v6.5 的 step ratio 仍然高。H4 假设：当前 runtime 被碎片化 kernel、临时 tensor 分配、sync 或 Python-loop 主导。

H4 成立的标准是：P4 runtime audit 中 top runtime components 可以解释至少 $85\%$ 的 step time，并且存在一个 fused or compiled variant 满足：

$$
\frac{T_{\text{step,fused}}}{T_{\text{step,current}}}\leq0.80.
$$

如果 runtime breakdown 显示主耗时就是 GEMM 且无法下降，则 H4 不成立，DWM2-poly2 的 compute path 需要重新设计。

---

### H5：bf16 cache 失败是因为错误使用了梯度敏感 cache，而不是混合精度完全不可用

v6.5 的 bf16Cache 导致：

$$
\operatorname{grad\_relerr}\approx1.6\times10^{-3},
$$

超过 gate。H5 不把 bf16Cache 作为主线，但允许一个 diagnostic：只压缩非梯度敏感 cache，例如 logging cache、inactive debug cache、或 validation-only cache。

H5 成立的标准很严格：

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\cos(\nabla_{mixed},\nabla_{fp32})>0.999,
$$

且：

$$
M_{\text{peak,mixed}}<M_{\text{peak,current}}.
$$

如果任何一项不满足，bf16 继续冻结。

---

### H6：如果真实 buffer reuse + delta streaming 仍没有 near-pass，则 DWM2-poly2 当前实现不应进入 task 阶段

H6 是 v6.6 的决策假设。当前 DWM2-poly2 已经多次 memory/time blocked。如果本轮实现了真实 bufferReuse 和 deltaStreaming 后仍不能达到：

$$
\frac{M_{\text{backward}}}{M_{\text{MLP}}}\leq1.10,
$$

$$
\frac{T_{\text{step}}}{T_{\text{MLP}}}\leq1.50,
$$

那么不能再继续用 task 或 functional update 掩盖底层失败，必须进入 primitive reset。

---

### H7：fallback primitive 必须改变 memory model，而不是只改变非线性函数

v6.5 的 fallback 结果显示，poly3、poly2+silu-base、RationalKAT、SparseInterp、RBFK2 都没有 near-pass。它们大多仍使用类似 runtime stack 或 workspace model。H7 假设：真正的 fallback 必须改变 memory model，例如：

```text
one-buffer residual transform
single-pass output mixing
chunked backward with bounded workspace
no full-layer delta retention
custom allocator pool
```

因此 v6.6 的 fallback 不再做大范围 task 搜索，而只做 kernel-level primitive reset。

---

## 4. 实验阶段总览

v6.6 分为十个阶段：

```text
P0: real-only contract and v6.5 reproduction check
P1: phase-local attribution v2
P2: true single-factor workspace repair
P3: combined memory package selection
P4: runtime kernel and allocation audit
P5: one-step numerical and loss-descent probe
P6: survivor selection and route gate
P7: limited task re-entry, only if S0/S1 survivor exists
P8: functional correction remains gated, optional only after task survivor
P9: primitive reset kernel benchmark if no survivor
P10: final route decision and next branch
```

v6.6 的核心阶段是 P1-P4。P7/P8 默认是关闭的，只有 P6 产生 S0/S1 survivor 才能打开。

---

## 5. P0：real-only contract and v6.5 reproduction check

### 5.1 目的

P0 的目的不是找新结果，而是确认本轮 runner、数据、baseline、measurement 与 v6.5 final run 对齐。由于 v6.5 有两个废弃中间 run，v6.6 必须从一开始锁定 final run lineage，避免混用旧 artifact。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-poly2-compiled-current
DWM2-poly2-compiled-noHiddenCache
DWM2-poly2-compiled-bf16Cache-diagnostic
DWM2-poly2-bufferReuse-v1, if implemented
DWM2-poly2-deltaStreaming-v1, if implemented
DWM2-poly2-bufferReuse+deltaStreaming, if implemented
```

### 5.3 必须记录字段

```text
run_id
artifact_root
git_commit_or_code_hash
script_path
variant
status
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
manual_forward_available
manual_backward_available
manual_update_available
nonKAN_param_count
edge_param_count
rollback_max_error
v65_reference_memory_ratio_min
v65_reference_step_ratio_min
current_reproduction_memory_ratio_min
current_reproduction_step_ratio_min
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
```

### 5.4 判断标准

P0 必须满足：

```text
fake_data_used = 0
proxy_row_used = 0
uses_loss_backward = 0 for graph-free candidates
nonKAN_param_count = 0 for PureKAN candidates
rollback_max_error < 1e-8
```

对于 reproduction，允许小幅波动，但必须满足：

$$
\left|r_{\text{memory,v66}}-r_{\text{memory,v65}}\right|\leq0.05,
$$

$$
\left|r_{\text{step,v66}}-r_{\text{step,v65}}\right|\leq0.15.
$$

如果 reproduction 偏离太大，则 P1-P4 的结果不能直接和 v6.5 比较，必须先修 profiler 或固定 measurement protocol。

### 5.5 必须生成图

1. `p0_reproduction_delta_bar.svg`：显示 v6.5 与 v6.6 的 min/mean memory ratio 和 step ratio 差异。
2. `p0_contract_heatmap.svg`：显示 no-fake、no-proxy、manual backward、nonKAN 等 contract 是否通过。

---

## 6. P1：phase-local attribution v2

### 6.1 目的

P1 要回答：当前 $M_{\text{backward,KAN}}>M_{\text{backward,MLP}}$ 的具体来源是什么。v6.5 已经说明 hidden cache 不是主因，但还没有把 workspace/temp/unexplained gap 拆到具体 tensor / op / allocator phase。P1-v2 必须把 unexplained gap 降到可行动水平。

### 6.2 测量对象

```text
A0-MLP-manual-linear-reference
A1-DWM2-current
A2-DWM2-noHiddenCache
A3-DWM2-bf16Cache-diagnostic
A4-DWM2-bufferReuse-v1, if implemented
A5-DWM2-deltaStreaming-v1, if implemented
```

### 6.3 数据与 shape grid

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

hidden_dim:
  64

seed:
  0

warmup / measure:
  50 / 200 update steps
```

### 6.4 Phase 定义

P1 必须按 phase 记录 peak：

```text
phase_forward_transform
phase_forward_mix
phase_loss_delta
phase_backward_output
phase_backward_block
phase_backward_input
phase_update_params
phase_optimizer_state
phase_cleanup
```

每个 phase 记录进入前、phase 内峰值、退出后 allocated/reserved：

```text
allocated_before_MB
allocated_peak_MB
allocated_after_MB
reserved_before_MB
reserved_peak_MB
reserved_after_MB
phase_peak_delta_MB
phase_retained_delta_MB
```

### 6.5 Tensor / op attribution 字段

必须记录：

```text
top_tensor_name_1
top_tensor_MB_1
top_tensor_lifetime_phase_1
top_tensor_op_1
...
top_tensor_name_10
top_tensor_MB_10
top_tensor_lifetime_phase_10
top_tensor_op_10

workspace_temp_MB
delta_buffer_MB
poly_temp_MB
mixing_temp_MB
update_temp_MB
allocator_padding_MB
unexplained_gap_MB
phase_explain_ratio
allocation_count_total
allocation_count_forward
allocation_count_backward
allocation_count_update
largest_allocation_MB
new_cuda_block_count
reused_buffer_count
```

### 6.6 判断标准

P1-v2 不要求 memory pass，但要求 attribution pass。通过标准是：

$$
\operatorname{phase\_explain\_ratio}\geq0.90,
$$

并且：

$$
\operatorname{unexplained\_gap\_MB}\leq0.10\cdot G_{\text{peak}},
$$

其中：

$$
G_{\text{peak}}=M_{\text{peak,KAN}}-M_{\text{peak,MLP}}.
$$

如果 P1-v2 没有 attribution pass，不允许进入 P2 修复结论。因为无法知道修复对象是否正确。

### 6.7 可视化

必须生成：

1. **phase-local peak waterfall**：横轴是 phase，纵轴是 allocated peak delta。
2. **tensor lifetime chart**：显示 top-10 tensors 的生命周期覆盖哪些 phase。
3. **allocation count by phase bar**：显示哪个 phase 产生最多 allocation。
4. **unexplained gap heatmap**：dataset × batch × depth。
5. **KAN-over-MLP peak attribution stacked bar**：把 $G_{\text{peak}}$ 分解为 delta、workspace、poly temp、update temp、allocator padding。

---

## 7. P2：true single-factor workspace repair

### 7.1 目的

P2 对每个 repair factor 做真实实现和单因素验证。v6.5 中多个关键 factor 是 `not_implemented`，v6.6 必须至少实现两个真实 factor：

```text
M2-deltaStreaming-v1
M3-bufferReuse-v1
```

如果工程时间有限，优先级是：

$$
\text{bufferReuse-v1} > \text{deltaStreaming-v1} > \text{updateWorkspaceReuse} > \text{safeMixedCache}.
$$

### 7.2 Repair factors

```text
R0-current:
  baseline current implementation

R1-bufferReuse-v1:
  preallocate fixed workspace pool for delta / transform / update temp

R2-deltaStreaming-v1:
  consume upstream delta layer-by-layer and overwrite previous delta buffer

R3-updateWorkspaceReuse-v1:
  reuse update temporary buffers and avoid per-parameter temporary allocation

R4-polyTransformNoAlloc-v1:
  compute poly transform into preallocated buffer, no new allocation inside measured loop

R5-safeMixedCache-diagnostic:
  only compress non-gradient-sensitive cache; disabled if grad relerr > 1e-4

R6-bufferReuse+deltaStreaming:
  combine R1 and R2
```

### 7.3 必须记录指标

```text
variant
repair_factor
status
memory_ratio_min
memory_ratio_mean
memory_ratio_max
step_ratio_min
step_ratio_mean
step_ratio_max
backward_ratio_min
backward_ratio_mean
backward_ratio_max
peak_allocated_MB
peak_reserved_MB
workspace_pool_MB
workspace_pool_used_peak_MB
workspace_pool_fragmentation_MB
new_allocation_count
reused_buffer_count
delta_buffer_count
delta_buffer_MB
poly_temp_MB
update_temp_MB
manual_cache_MB
optimizer_state_MB
kernel_count_forward
kernel_count_backward
kernel_count_update
grad_relerr_max
grad_cos_min
forward_relerr
actual_one_step_loss_delta_probe
```

### 7.4 判断标准

单因素 repair 有三层 gate。

Diagnostic pass:

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

Memory useful pass:

$$
\frac{M_{\text{repair}}}{M_{\text{current}}}\leq0.95.
$$

Near-pass:

$$
\frac{M_{\text{backward,repair}}}{M_{\text{backward,MLP}}}\leq1.10,
$$

$$
\frac{T_{\text{step,repair}}}{T_{\text{step,MLP}}}\leq1.50.
$$

Final single-factor pass:

$$
\frac{M_{\text{backward,repair}}}{M_{\text{backward,MLP}}}<1.0,
$$

$$
\frac{T_{\text{step,repair}}}{T_{\text{step,MLP}}}\leq1.35.
$$

只有满足 diagnostic pass 的 repair 才能进入 P3 组合。

### 7.5 可视化

1. **single-factor memory reduction bar**：显示每个 factor 相对 current 的 memory reduction。
2. **single-factor step overhead bar**：显示每个 factor 的 step overhead。
3. **memory reduction vs step overhead scatter**：理想点在左下方。
4. **gradient correctness lollipop plot**：显示 relerr 和 cos。
5. **allocation count reduction chart**：看 buffer reuse 是否真的减少 new allocation。

---

## 8. P3：combined memory package selection

### 8.1 目的

P3 把 P2 中有效的 factor 组合起来，验证是否存在真正 memory/time survivor。P3 不允许把未实现 factor 组合写成失败或成功；未实现必须保留 `not_implemented`。

### 8.2 Candidate packages

```text
C0-current
C1-bufferReuse
C2-deltaStreaming
C3-bufferReuse+deltaStreaming
C4-bufferReuse+updateWorkspaceReuse
C5-deltaStreaming+updateWorkspaceReuse
C6-bufferReuse+deltaStreaming+polyNoAlloc
C7-allWorkspaceOptimized-light
C8-allWorkspaceOptimized-full
```

其中：

```text
allWorkspaceOptimized-light = bufferReuse + deltaStreaming
allWorkspaceOptimized-full  = bufferReuse + deltaStreaming + updateWorkspaceReuse + polyNoAlloc
```

### 8.3 必须记录指标

除 P2 指标外，P3 还必须记录：

```text
package_components
component_status_all_implemented
component_status_all_grad_pass
memory_pass_count
time_pass_count
grad_pass_count
both_memory_time_pass_count
near_pass_count
best_shape_memory_ratio
worst_shape_memory_ratio
shape_stability_score
batch_scaling_slope_memory
batch_scaling_slope_step
```

shape stability score 定义为：

$$
S_{\text{shape}}=1-\frac{\operatorname{std}(r_{\text{memory}})}{\operatorname{mean}(r_{\text{memory}})+\epsilon}.
$$

如果 $S_{\text{shape}}$ 很低，说明该 package 对 batch/depth 不稳定。

### 8.4 Survivor 类型

P3 输出 survivor type：

```text
S0: memory_ratio < 1.0 and step_ratio <= 1.20
S1: memory_ratio < 1.0 and step_ratio <= 1.35
S2: memory_ratio <= 1.10 and step_ratio <= 1.50 with clear improvement trend
S3: no memory near-pass
S4: gradient correctness fail
```

判断条件：

S0：

$$
\max_{\text{shapes}} r_{\text{memory}}<1.0,
$$

$$
\operatorname{mean}_{\text{shapes}} r_{\text{step}}\leq1.20.
$$

S1：

$$
\max_{\text{shapes}} r_{\text{memory}}<1.0,
$$

$$
\operatorname{mean}_{\text{shapes}} r_{\text{step}}\leq1.35.
$$

S2：

$$
\min_{\text{shapes}} r_{\text{memory}}\leq1.10,
$$

$$
\min_{\text{shapes}} r_{\text{step}}\leq1.50,
$$

并且相对 current 的 memory improvement 至少 $10\%$。

S3：没有 near-pass。

S4：任何核心 package gradient correctness fail。

### 8.5 可视化

1. **combined package scorecard**：显示 memory/time/grad/pass counts。
2. **batch-depth stability heatmap**：dataset × batch × depth 的 memory ratio。
3. **S0/S1/S2 threshold plot**：在 Pareto 图上标出 pass boundary。
4. **component contribution waterfall**：显示每个 component 对 memory reduction 的边际贡献。

---

## 9. P4：runtime kernel and allocation audit

### 9.1 目的

如果 P3 有 S0/S1/S2 candidate，P4 进一步确认 runtime 是否可接受；如果 P3 没有 survivor，但某 repair 明显降低 memory，P4 用于找 time bottleneck。P4 不能打开 task，只做 runtime/kernel audit。

### 9.2 必须记录

```text
variant
package
forward_time_ms
transform_time_ms
mixing_time_ms
loss_delta_time_ms
backward_time_ms
backward_delta_time_ms
backward_coeff_time_ms
update_time_ms
optimizer_state_time_ms
kernel_count_total
kernel_count_elementwise
kernel_count_gemm
kernel_count_custom
kernel_count_allocation_related
cuda_sync_count
python_loop_count
torch_compile_graph_break_count
compiled_region_count
allocation_count_total
allocation_count_inside_measure_loop
largest_temp_allocation_MB
```

### 9.3 判断标准

如果某 candidate memory pass 但 time fail，P4 要判断 time fail 是否可修。

Runtime repairable 的标准：

```text
non-GEMM elementwise/allocation kernels account for >= 35% of step time
or allocation_count_inside_measure_loop remains high
or transform/backward has repeated small kernels
```

如果 runtime repairable，下一轮可以继续 kernel fusion。如果主要耗时是不可避免 GEMM 或数学计算，则 DWM2-poly2 可能不适合作为 final primitive。

### 9.4 可视化

1. **runtime component waterfall**：forward transform、mixing、backward delta、backward coeff、update。
2. **kernel count stacked bar**：GEMM、elementwise、allocation-related、custom。
3. **allocation count vs step time scatter**。
4. **compile graph break heatmap**。

---

## 10. P5：one-step numerical and loss-descent probe

### 10.1 目的

如果 P3 产生 S0/S1/S2，P5 在不做完整 task training 的情况下验证该 package 的数值稳定性。P5 的重点不是 final accuracy，而是证明 memory package 没有破坏 loss descent。

### 10.2 Probe 设置

对每个 survivor，在每个 dataset 上抽取：

```text
train batch
holdout batch
validation mini-batch
```

执行：

```text
clone weights
compute manual gradient
apply one manual optimizer update
measure train/holdout/val loss before and after
rollback weights
check rollback exactness
```

### 10.3 必须记录

```text
train_loss_before
train_loss_after
holdout_loss_before
holdout_loss_after
val_loss_before
val_loss_after
train_loss_delta
holdout_loss_delta
val_loss_delta
actual_descent_train
actual_descent_holdout
actual_descent_val
bad_step_flag
rollback_error_after_probe
param_delta_norm
update_over_param_norm
grad_norm
finite_gradient_rate
```

### 10.4 判断标准

P5 pass 要求：

$$
\Delta L_{\text{train}}<0,
$$

$$
\Delta L_{\text{holdout}}\leq0.02\cdot |L_{\text{holdout,before}}|,
$$

且：

$$
\operatorname{BadStepRate}\leq0.05.
$$

rollback 必须满足：

$$
\operatorname{rollback\_error}<10^{-8}.
$$

如果 P5 失败，candidate 不进入 task re-entry。

### 10.5 可视化

1. **before-after loss slope plot**。
2. **bad-step heatmap**：dataset × package。
3. **update norm vs loss delta scatter**。
4. **rollback error bar**。

---

## 11. P6：survivor selection and route gate

### 11.1 目的

P6 汇总 P1-P5，决定是否打开 task re-entry。

### 11.2 Route gate

```text
Open P7 task re-entry if:
  survivor_type in {S0, S1}
  P5 one-step probe pass
  no fake/proxy
  grad correctness pass

Open limited P7 diagnostic if:
  survivor_type == S2
  memory improved >= 10%
  step_ratio <= 1.50
  P5 pass

Do not open P7 if:
  survivor_type in {S3, S4}
```

### 11.3 输出字段

```text
survivor_type
best_package
best_memory_ratio
best_step_ratio
best_backward_ratio
memory_improvement_vs_current
step_improvement_vs_current
grad_pass
p5_probe_pass
open_task_reentry
open_functional_correction
route
reason
```

---

## 12. P7：limited task re-entry, only if S0/S1 survivor exists

### 12.1 目的

P7 只有在 memory/time survivor 存在时才运行。它验证修复后的 package 是否还能学任务。

### 12.2 方法

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
  DWM2-current-task-only, diagnostic
  DWM2-best-memory-package + ManualAdamW
  DWM2-best-memory-package + ManualAdanLite
```

P7 不运行 functional correction，不运行 LightSmooth。只验证 task learner。

### 12.3 必须记录

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

### 12.4 判断标准

P7 task pass 要求：

$$
\operatorname{Acc}_{\text{KAN}}\geq\operatorname{Acc}_{\text{MLP}}-0.01
$$

on all datasets, and at least two datasets satisfy:

$$
\operatorname{Acc}_{\text{KAN}}\geq\operatorname{Acc}_{\text{MLP}}.
$$

Wall-clock 要求：

$$
\operatorname{ValLossAUC}_{\text{time,KAN}}\leq\operatorname{ValLossAUC}_{\text{time,MLP}}
$$

on at least two datasets.

Memory 要求：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}}<1.0
$$

on all datasets.

### 12.5 可视化

1. **val loss vs step**。
2. **val loss vs wall-clock**。
3. **accuracy vs wall-clock**。
4. **time-to-target bar chart**。
5. **task-efficiency Pareto**。
6. **seedwise paired delta plot**。

---

## 13. P8：functional correction remains gated

P8 只在 P7 task pass 后才允许运行。v6.6 默认不打开 P8。原因是 v6.5 的 blocker 仍是 memory/time，不是 geometry。

如果 P8 被打开，必须只做 smoke，不做 final claim。

P8 的 minimal design 是：

```text
DWM2-best-memory-package + ManualAdanLite task-only
DWM2-best-memory-package + ManualAdanLite + small geometry correction
```

必须记录：

```text
cos_new_task
holdout_descent_ratio
bad_step_rate
phi_change
curvature_change
ECE_change
correction_overhead_ms
fallback_rate
```

P8 pass 要求：

$$
\cos(d_{\text{new}},d_{\text{task}})\geq0.85,
$$

$$
\frac{\operatorname{HoldoutDescent}(d_{\text{new}})}{\operatorname{HoldoutDescent}(d_{\text{task}})}\geq0.95,
$$

$$
\operatorname{BadStepRate}\leq0.02.
$$

---

## 14. P9：primitive reset kernel benchmark if no survivor

### 14.1 目的

如果 P6 输出 S3/S4，P9 进入 primitive reset。v6.5 已经显示旧 fallback 没有 near-pass，因此 v6.6 的 P9 不能重复旧 fallback 表面变体。P9 必须测试改变 memory model 的 primitive。

### 14.2 Reset primitive candidates

```text
R0-ManualLinear+TinyChannelResidual:
  y = W(x + s r(x)), residual bounded and no extra layer delta retention

R1-DWM2-poly1-minimal:
  poly2 降级到 poly1，确认 transform temp 是否主因

R2-DWM2-poly2-singleBuffer:
  强制所有 transform / delta / update 使用单 workspace pool

R3-SparseInterp-gather2-forwardOnly:
  只测 forward + input backward，不更新 knots，验证 sparse path 是否能低 memory

R4-SparseInterp-gather2-streamingGrad:
  在 R3 可行后加入 knot gradient streaming

R5-RationalKAT-oneBuffer-fastpoly:
  rational fastpoly 使用单 workspace buffer
```

### 14.3 P9 不做 task

P9 只做 kernel/profiler，不做 task accuracy。因为 reset primitive 还没证明 memory/time。

### 14.4 判断标准

P9 near-pass 要求：

$$
\frac{M_{\text{backward}}}{M_{\text{MLP}}}\leq1.10,
$$

$$
\frac{T_{\text{step}}}{T_{\text{MLP}}}\leq1.50,
$$

并且 gradient correctness pass。

如果 P9 没有 near-pass，route 是 terminal engineering blocker：

```text
current Python/Torch manual PureKAN primitive family cannot satisfy memory/time target without lower-level custom kernel.
```

### 14.5 可视化

1. **reset primitive memory-time Pareto**。
2. **old fallback vs reset fallback comparison**。
3. **workspace model comparison table**。
4. **single-buffer lifetime diagram**。

---

## 15. P10：final route decision

v6.6 必须输出 route decision，不能只说失败。

### Route cases

```text
R1-memorySolved:
  S0/S1 survivor exists, P5 pass, task re-entry open.

R2-nearPassEngineering:
  S2 survivor exists, memory improved >= 10%, but not full pass.
  Continue engineering, no full task confirm.

R3-workspaceAttributionIncomplete:
  P1 attribution cannot explain peak.
  Improve profiler before repair claim.

R4-repairNoEffect:
  bufferReuse/deltaStreaming implemented but memory reduction < 5%.
  DWM2-poly2 likely not memory-repairable at current abstraction.

R5-gradientBroken:
  repair reduces memory but breaks gradient correctness.
  Fix numerical path or revert.

R6-primitiveResetNeeded:
  no DWM2 survivor, P9 has at least one reset near-pass.
  Switch primitive branch.

R7-terminalKernelNeeded:
  no DWM2 survivor and no reset near-pass.
  Need Triton/CUDA-level custom kernel or different primitive family.
```

### 必须输出 JSON 字段

```text
route
best_package
best_memory_ratio
best_step_ratio
best_backward_ratio
memory_improvement_vs_current
step_improvement_vs_current
survivor_type
fallback_triggered
fallback_near_pass_count
open_task_reentry
open_functional_correction
no_fake
no_proxy
primary_blocker
next_required_implementation
```

---

## 16. 必须生成的 CSV / JSON artifact

```text
p0_contract.csv
p0_reproduction_check.csv
p1_phase_local_attribution.csv
p1_tensor_lifetime_topk.csv
p1_allocation_phase_summary.csv
p2_single_factor_workspace_repair.csv
p2_repair_gradient_correctness.csv
p3_combined_workspace_packages.csv
p3_combined_workspace_packages_detail.csv
p4_runtime_kernel_audit.csv
p5_one_step_probe.csv
p6_survivor_selection.csv
p7_task_reentry.csv
p7_task_trace.csv
p8_functional_correction_smoke.csv
p9_primitive_reset_kernel_benchmark.csv
failure_table.csv
route_decision.json
aggregate_decision.json
```

如果某阶段被 gate，CSV 必须存在一行 `not_run`，并写明：

```text
status
reason
gated_by
```

不得写假数值。

---

## 17. 必须生成的可视化

### 17.1 Phase-local memory dashboard

必须包含：

```text
phase-local peak waterfall
KAN-over-MLP peak attribution stacked bar
tensor lifetime chart
unexplained gap heatmap
```

### 17.2 Workspace repair dashboard

必须包含：

```text
single-factor memory reduction bar
single-factor step overhead bar
memory reduction vs step overhead scatter
allocation count reduction chart
```

### 17.3 Combined package Pareto

必须包含：

```text
memory ratio vs step ratio scatter
S0/S1/S2 threshold boundary
point size = grad relerr
color = package family
```

### 17.4 Runtime kernel audit

必须包含：

```text
runtime component waterfall
kernel count stacked bar
allocation count vs step time scatter
compile graph break heatmap
```

### 17.5 One-step probe dashboard

必须包含：

```text
train/holdout/val before-after loss plot
bad-step heatmap
update norm vs loss delta scatter
rollback error bar
```

### 17.6 Task re-entry dashboard, only if opened

必须包含：

```text
val loss vs step
val loss vs wall-clock
accuracy vs wall-clock
time-to-target bar
task-efficiency Pareto
seedwise paired delta plot
```

### 17.7 Route decision dashboard

必须包含：

```text
route timeline by stage
failure taxonomy heatmap
best candidate scorecard
next required implementation box
```

---

## 18. 成功与失败的解释规则

### Case A：bufferReuse 或 deltaStreaming 让 memory pass

如果出现：

$$
M_{\text{backward}}/M_{\text{MLP}}<1.0,
$$

且 step ratio 不超过 $1.35$，则 DWM2-poly2 继续作为主线，进入 P5/P7。下一步才允许恢复 task/functional。

### Case B：memory 改善明显但没有 pass

如果 memory improvement $\geq10\%$，但 ratio 仍在 $1.0$ 到 $1.10$ 之间，则进入 `R2-nearPassEngineering`。可以继续工程优化，但不做 full task confirm。

### Case C：repair 完全无效

如果真实 bufferReuse / deltaStreaming 都实现后，memory improvement $<5\%$，说明当前峰值可能受 abstraction / allocator / framework 限制。此时应考虑 primitive reset 或更低层 kernel。

### Case D：repair 降 memory 但破坏 gradient

如果 memory 降低但：

$$
\operatorname{grad\_relerr}\geq10^{-4},
$$

则该修复不能进入 task。需要数值修复或放弃。

### Case E：P9 reset 也没有 near-pass

如果 reset primitive 也没有 near-pass，则 v6.6 的诚实结论是：

```text
current graph-free PureKAN implementation level is not enough.
Need lower-level Triton/CUDA custom kernel or different primitive family.
```

---

## 19. 最终建议

v6.6 的一句话策略是：

$$
\boxed{\text{不要再用 task 或 functional correction 掩盖 memory/time failure。}}
$$

本轮只应该相信三类结果：

```text
actual CUDA peak 是否下降
step time 是否下降或可控
gradient correctness 是否保持
```

如果这三者没有同时成立，后续 task / LightSmooth / functional correction 都应该继续 gate。

本轮最应该投入的工程实现不是 bf16 cache，也不是 noHiddenCache，而是：

```text
1. bufferReuse-v1
2. deltaStreaming-v1
3. updateWorkspaceReuse-v1
4. polyTransformNoAlloc-v1
```

如果这些真实实现后仍然没有 memory near-pass，就应接受 DWM2-poly2 当前 abstraction 不足，进入 primitive reset 或 lower-level kernel route。

