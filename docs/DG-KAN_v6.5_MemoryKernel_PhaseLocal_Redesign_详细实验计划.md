# DG-KAN v6.5：Memory Kernel 与 Phase-Local Workspace Redesign 详细实验计划

> 本计划基于 v6.4 real-only run 的真实结果制定。v6.4 已经证明 `DWM2-poly2-compiled-current` 的 graph-free/manual gradient correctness 仍然可靠，但 P1 memory finalization 没有产生 survivor：最佳 backward memory ratio 仍约为 $1.1423$，最佳 step time ratio 仍约为 $1.613$。因此 v6.5 的核心不是继续跑 task recipe、LightSmooth 或 functional correction，而是先把 memory kernel、workspace peak、delta streaming 和 buffer reuse 打穿。只有 P1/P2 出现真实 memory survivor 后，才允许重新打开 task、functional correction 和 confirm 阶段。

---

## 0. 实验整体目标

v6.5 的实验整体目标是把当前 graph-free PureKAN 主线从“梯度正确但 memory/time gate 失败”推进到“至少出现一个真实 memory-pass candidate”。本轮不追求直接完成最终 PureKAN functional training system，而是专注解决 v6.4 暴露出的硬瓶颈：

$$
\frac{M_{\text{backward,DWM2}}}{M_{\text{backward,MLP}}} > 1
$$

和：

$$
\frac{T_{\text{step,DWM2}}}{T_{\text{step,MLP}}} \gg 1.10.
$$

v6.4 的关键数字是：

$$
R_M^{best}=1.1423,
$$

$$
R_T^{best}=1.6130.
$$

其中 $R_M$ 表示 backward memory ratio，$R_T$ 表示 step time ratio。v6.5 的第一目标是把 $R_M$ 压到 $1.0$ 以下，第二目标是在不牺牲 memory 的前提下把 $R_T$ 压回可接受区间。更明确地说，本轮最低成功定义是：

$$
R_M < 1.0,
$$

$$
R_T \leq 1.35,
$$

并且 manual gradient correctness 仍然满足：

$$
\operatorname{relerr}(\nabla_{manual},\nabla_{ref}) < 10^{-4},
$$

$$
\cos(\nabla_{manual},\nabla_{ref}) > 0.999.
$$

如果能进一步达到：

$$
R_M \leq 0.90,
$$

$$
R_T \leq 1.20,
$$

则该 candidate 可以进入 v6.5 的 task re-entry 阶段。如果只达到 $R_M<1.0$ 但 $R_T>1.35$，则仍然视为 memory proof，不进入 task confirm，只继续 runtime repair。

v6.5 的最终输出不应该是“成功/失败”一句话，而要明确回答下面三个问题：

1. 当前 backward peak memory 的主要来源到底是 manual cache、delta buffer、workspace temp、optimizer state，还是 measurement baseline 不公平？
2. 哪一种 memory repair 策略真正降低了 total CUDA peak，而不是只降低了 manual cache estimate？
3. 如果 DWM2-poly2 无法同时通过 memory/time gate，是否应该继续投入 DWM2，还是切换到 fallback primitive？

---

## 1. 当前定位与 v6.4 结论重述

v6.4 的结论不能解读为 task 失败，也不能解读为 functional correction 失败。v6.4 被 P1 efficiency gate 真实阻断，P2-P8 保持 `not_run` 是正确实验纪律。v6.4 已经确认三件事情。

第一，`DWM2-poly2-compiled-current` 仍然保持 strict PureKAN / graph-free contract：候选本身没有 non-KAN 参数，使用 manual backward，fake data 没有进入结果。梯度正确性没有失败，P1 中 current 和 noHiddenCache variant 的 gradient relative error 约为 $8\times10^{-8}$，gradient cosine 约等于 $1$。

第二，`noHiddenCache` 是有诊断价值但不是解法。它把 hidden cache 降到 $0$，manual cache mean 从约 $1.039$ MB 降到约 $0.893$ MB，但 total backward memory ratio 只从 mean $1.292$ 降到 $1.284$，几乎没有改变；同时 step time ratio 从 mean $1.936$ 上升到 $2.142$。这说明 hidden cache 不是主要 memory peak 来源，单纯用 recomputation 换 cache 会显著伤害 runtime。

第三，v6.4 的 route decision 是 R3：DWM2-poly2 task line 仍 memory-blocked，下一步应聚焦 memory kernel，而不是消耗 task/confirm seeds。v6.5 必须继承这个 route，不能越过 P1 去跑 P2/P6。

因此，本轮计划的核心定位是：

$$
\boxed{\text{先做 memory kernel 与 phase-local workspace redesign，再谈 task/functional。}}
$$

---

## 2. 实验纪律与禁止事项

v6.5 继续执行 real-only 规则。任何阶段都不允许使用 proxy rows、derived placeholder rows、mock profiler rows 或 fake data。

旧脚本如果存在历史 proxy 行，必须保持 hard-fail 或只允许 diagnostic-only 输出。所有用于 gate 的 CSV 必须来自真实运行路径。任何未实现的 variant 必须标记为 `not_implemented`，不能写成 failure value，也不能写成 pass value。

本轮禁止以下做法：

```text
1. P1 没有 memory survivor 时继续跑 P2 task recipe。
2. P1 没有 memory survivor 时继续跑 functional correction 或 LightSmooth。
3. 用固定 step_ratio / bmem_ratio 填充未实现 variant。
4. 用 manual cache estimate 代替 actual CUDA peak 作为 pass 依据。
5. 用单一 batch size / 单一 depth 的 near-pass 直接进入 confirm。
6. 只看 step count，不看 wall-clock time。
7. 只看 memory，不看 gradient correctness 与 one-step actual descent。
```

每个阶段都必须输出：

```text
stage_status
run_id
artifact_hash
fake_data_used
proxy_rows_used
not_implemented_count
gated_not_run_count
```

如果有任何 `fake_data_used=1` 或 `proxy_rows_used=1`，该阶段结果不能用于结论。

---

## 3. 核心假设

v6.5 围绕七个假设设计实验。每个假设都必须有明确实验、记录指标和判定标准。

### H1：当前 memory peak 主要来自 workspace / delta / temporary buffer，而不是 hidden cache

v6.4 的 `noHiddenCache` 已经消除了 hidden cache，但 total memory ratio 几乎没有下降。因此最自然的假设是：

$$
M_{peak} \neq M_{hidden\ cache}\ \text{主导},
$$

而更可能是：

$$
M_{peak} \approx M_{workspace}+M_{delta}+M_{temp}+M_{optimizer\ state}+M_{manual\ cache}.
$$

验证方式不是继续猜，而是做 phase-local peak instrumentation，把 forward、loss delta、manual backward、update、optimizer state 的 memory peak 分开记录。若 workspace/delta/temp 阶段解释了 total peak 的主要超额部分，则 H1 成立。

H1 成立标准：

$$
\frac{M_{workspace}+M_{delta}+M_{temp}}{M_{peak}-M_{MLP}} \geq 0.60.
$$

如果这个比例低于 $0.30$，说明 memory peak 不是 workspace/delta/temp 主导，需要重新检查 baseline、optimizer state 或 allocator fragmentation。

### H2：buffer reuse 可以降低 total CUDA peak，而不明显增加 step time

如果当前 peak 是由多个临时 tensor 在同一 phase 内重叠导致，那么复用 buffer 应该能降低 actual peak，而不是只降低 cache estimate。buffer reuse 的目标是让多个 temporary tensor 共享同一段 workspace，特别是 poly transform buffer、delta buffer、backward coefficient accumulation buffer 和 update buffer。

H2 成立标准：

$$
R_M^{bufferReuse} \leq R_M^{current}-0.10,
$$

并且：

$$
R_T^{bufferReuse} \leq R_T^{current}+0.10.
$$

如果 memory 降低小于 $5\%$，H2 不成立。如果 memory 降低但 step time 增加超过 $20\%$，则 buffer reuse 只能作为 diagnostic，不进入 candidate。

### H3：delta streaming 可以降低 backward peak，但必须控制 time penalty

manual backward 阶段如果保留完整 delta 或多层 delta，会增加 peak memory。delta streaming 的策略是逐层 consume upstream delta，并尽量在计算完当前层 gradient 后释放或复用 delta buffer。

H3 成立标准：

$$
M_{delta}^{streaming} \leq 0.5M_{delta}^{current},
$$

$$
R_M^{streaming}<1.05,
$$

并且：

$$
R_T^{streaming} \leq 1.35.
$$

如果 delta memory 降低但 total peak 没降，说明 delta 不是 peak 主因。如果 total peak 降了但 step time 超过 $1.6$，则需要 fused streaming，而不是 Python-level streaming。

### H4：bf16/fp16 cache compression 可以降低 cache memory，但必须保持 gradient 与 one-step descent

把部分 cache 改为 bf16/fp16 是直接降低 memory 的方法。但这个方法风险是：manual backward 的梯度数值可能变差，尤其是在 KMNIST 或 high-curvature geometry 上。v6.5 必须同时记录 gradient correctness 和 actual one-step loss descent。

H4 成立标准：

$$
R_M^{bf16}<1.05,
$$

$$
\operatorname{relerr}(\nabla)<10^{-4},
$$

$$
\cos(\nabla_{bf16},\nabla_{fp32})>0.999,
$$

并且 actual holdout descent ratio 满足：

$$
\frac{\Delta L_{holdout}^{bf16}}{\Delta L_{holdout}^{fp32}} \geq 0.98.
$$

如果 bf16 memory 降低但 holdout descent 明显下降，则只保留为 memory diagnostic，不进入 task。

### H5：noTemp-poly2 只有在 fused recompute 下才可能有用

v6.4 的 noHiddenCache 已经说明，简单 recomputation 会显著增加 step time。noTemp-poly2 也有相同风险：如果不保存 polynomial features，而在 backward 中重新计算，可能省 memory 但拖慢 runtime。只有把 recompute 放进 fused kernel 或少量 kernel 中，才有机会成立。

H5 成立标准：

$$
R_M^{noTemp}<1.05,
$$

$$
R_T^{noTemp}\leq1.35.
$$

如果 $R_T^{noTemp}>1.60$，该路线不进入组合包。

### H6：phase-local instrumentation 必须能解释 total peak，否则当前 profiler 不足以指导优化

v6.4 已经有 `p1_phase_local_memory.csv`，但下一步需要让它更严格：每个 phase 的 peak、allocation count、largest temp tensor、reserved memory change 都要记录，并且要能解释 total peak。

定义 explain ratio：

$$
E = \frac{\max_p M_{phase,p}}{M_{total\ peak}}.
$$

H6 成立标准：

$$
E \geq 0.90.
$$

如果 $E<0.75$，说明 phase-local instrumentation 漏掉了 allocator / overlap / async CUDA 行为，不能用它做 gate，需要先修 profiler。

### H7：如果 DWM2-poly2 多个 memory package 仍无法过线，应进入 fallback primitive decision

如果 bufferReuse、deltaStreaming、bf16 cache、noTemp 和组合包都无法让 $R_M<1.0$，则说明 DWM2-poly2 当前实现不是最终 memory-efficient primitive。此时不能继续在上层 task recipe 上消耗资源，而要进入 fallback primitive decision。

H7 成立条件是所有 memory packages 均失败：

$$
\min_i R_{M,i} \geq 1.0
$$

或即使 memory 过线但 time 无法接受：

$$
\min_{i:R_{M,i}<1.0} R_{T,i} > 1.60.
$$

若 H7 成立，本轮 route decision 应为：

```text
R5-memory-kernel-limit:
  DWM2-poly2 current implementation cannot satisfy P1.
  Move to fallback primitive or lower-level fused CUDA/Triton kernel.
```

---

## 4. v6.5 实验总流程

v6.5 分为九个阶段。前六个阶段围绕 memory/kernel，后续 task/functional 阶段只有在 P1/P2 出现 survivor 后才打开。

```text
P0: real-only implementation contract and profiler sanity
P1: phase-local memory attribution audit
P2: single-factor memory kernel repair
P3: combined memory package interaction matrix
P4: runtime repair and kernel launch audit
P5: one-step gradient/loss correctness probe
P6: P1 survivor selection and route decision
P7: gated task re-entry with ManualAdanLite
P8: gated functional correction smoke
P9: fallback primitive decision and final diagnosis
```

本轮核心 gate 是 P6。P6 之前所有实验都不做 full task claim；P6 没有 survivor，则 P7/P8 必须保持 `not_run`。

---

## 5. P0：real-only implementation contract and profiler sanity

### 5.1 目的

P0 确认所有待测 memory package 是真实实现，不是 proxy row；同时确认 profiler 能够稳定记录 actual CUDA peak、phase-local peak、manual cache estimate、workspace temp 和 runtime decomposition。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-poly2-compiled-current
DWM2-poly2-compiled-noHiddenCache
DWM2-poly2-compiled-bufferReuse
DWM2-poly2-compiled-deltaStreaming
DWM2-poly2-compiled-bf16Cache
DWM2-poly2-compiled-noTempPoly2
DWM2-poly2-compiled-bufferReuse+deltaStreaming
DWM2-poly2-compiled-allMemoryOptimized
```

未实现的 variant 必须写：

```text
status = not_implemented
used_for_gate = 0
```

不能写任何 fake ratio。

### 5.3 必须记录字段

```text
stage
method
variant_id
implementation_status
used_for_gate
fake_data_used
proxy_row_used
loss_backward_used
torch_autograd_graph_used
manual_forward_available
manual_backward_available
manual_update_available
nonKAN_param_count
edge_param_count
coverage_edge
rollback_max_abs_error
device
cuda_available
seed
batch_size
hidden_dim
depth
profiler_actual_cuda_peak_available
phase_local_peak_available
manual_cache_estimate_available
workspace_temp_available
artifact_hash
```

### 5.4 通过标准

P0 contract pass 要求：

```text
implementation_status = measured
fake_data_used = 0
proxy_row_used = 0
loss_backward_used = 0 for graph-free candidates
torch_autograd_graph_used = 0 for graph-free candidates
manual_forward_available = 1
manual_backward_available = 1
manual_update_available = 1
nonKAN_param_count = 0
coverage_edge = 1
rollback_max_abs_error < 1e-8
profiler_actual_cuda_peak_available = 1
```

如果 `profiler_actual_cuda_peak_available=0`，P1 不能开始，因为本轮的核心就是 total memory peak。

### 5.5 可视化

P0 必须生成 implementation contract heatmap。横轴为 variant，纵轴为 contract item，包括 `no_fake`、`no_proxy`、`manual_backward`、`nonKAN=0`、`coverage=1`、`profiler_peak_available`。这张图的目的不是展示性能，而是确保后续数据可信。

---

## 6. P1：phase-local memory attribution audit

### 6.1 目的

P1 先不修模型，只解释 memory peak。它要回答：当前 $R_M>1$ 的超额 memory 到底来自哪个 phase。

### 6.2 测试设置

P1 使用 v6.4 相同 setting，以保证可比较：

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

warmup / measured steps:
  50 / 200

seed:
  0
```

P1 只测：

```text
MLP-manual-linear-reference
DWM2-poly2-compiled-current
DWM2-poly2-compiled-noHiddenCache
```

这一步不测所有 repair 包，先确认 instrumentation 可以解释 current 和 noHiddenCache 的 peak。

### 6.3 必须记录字段

```text
method
dataset
batch_size
depth
hidden_dim
forward_peak_MB
loss_delta_peak_MB
backward_adjoint_peak_MB
update_peak_MB
optimizer_state_peak_MB
total_step_peak_MB
reserved_peak_MB
manual_cache_total_MB
cache_x_MB
cache_hidden_MB
cache_delta_MB
cache_temp_MB
workspace_temp_MB
largest_temp_tensor_MB
num_tensor_allocations_forward
num_tensor_allocations_backward
num_tensor_allocations_update
cuda_allocated_before_step_MB
cuda_allocated_after_forward_MB
cuda_allocated_after_backward_MB
cuda_allocated_after_update_MB
cuda_reserved_before_step_MB
cuda_reserved_after_step_MB
phase_peak_explain_ratio
memory_ratio_vs_MLP
step_time_ratio_vs_MLP
backward_time_ratio_vs_MLP
```

### 6.4 判断标准

P1 instrumentation pass 要求：

$$
E=\frac{\max_p M_{phase,p}}{M_{total\ peak}} \geq 0.90.
$$

同时，manual cache estimate 与 actual peak 不能被混淆。必须报告：

$$
G = M_{total\ peak}-M_{manual\ cache}-M_{parameter}-M_{optimizer}.
$$

其中 $G$ 是 unexplained / workspace-like gap。如果：

$$
G > 0.25M_{total\ peak},
$$

则说明 workspace/temp/allocator 是主问题，P2 应优先 bufferReuse / deltaStreaming。如果：

$$
M_{manual\ cache} > 0.5M_{total\ peak},
$$

才优先 cache compression。

### 6.5 可视化

P1 必须画四类图。

第一，phase-local memory waterfall。对每个 method，把 `forward_peak_MB`、`loss_delta_peak_MB`、`backward_adjoint_peak_MB`、`update_peak_MB`、`optimizer_state_peak_MB` 画成 waterfall，突出 total peak 来自哪个 phase。

第二，actual peak vs manual cache scatter。横轴 `manual_cache_total_MB`，纵轴 `total_step_peak_MB`。如果点远离对角线，说明 manual cache 不是 total peak 的充分解释。

第三，batch/depth scaling curve。横轴 batch size，纵轴 memory ratio，分 depth 画线。若 depth=4 的 peak 超线性增长，说明 per-layer buffer overlap 或 delta retention 有问题。

第四，unexplained gap heatmap。行是 dataset/batch/depth，列是 method，颜色是 $G/M_{total\ peak}$。

---

## 7. P2：single-factor memory kernel repair

### 7.1 目的

P2 对每个 memory repair factor 单独测试，判断哪一种策略真正降低 actual CUDA peak。P2 不允许组合多个因素，否则无法归因。

### 7.2 单因素候选

```text
M0-current:
  当前 DWM2-poly2-compiled-current

M1-bf16Cache:
  x / hidden / selected cache 使用 bf16 保存，backward 中转换回 fp32

M2-deltaStreaming:
  backward 中逐层 consume delta，不保留完整 delta list

M3-bufferReuse:
  poly temp / delta / grad accumulation / update workspace 复用同一 workspace pool

M4-noTempPoly2:
  不保存 poly2 intermediate，backward 重算 x^2 和 derivative

M5-updateInPlaceSafe:
  update 阶段避免额外 clone，使用 safe in-place 或 preallocated update buffer

M6-optimizerStateSplit:
  optimizer state update 与 gradient buffer 生命周期错开，减少峰值重叠
```

### 7.3 测试设置

P2 使用 P1 相同 profiling grid：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
batch sizes = 128,256,512
depths = 2,4
hidden_dim = 64
seed = 0
warmup/measure = 50/200
```

### 7.4 必须记录字段

除 P1 字段外，每个 repair 还要记录：

```text
repair_type
expected_memory_target
actual_memory_reduction_vs_current
actual_step_penalty_vs_current
actual_backward_penalty_vs_current
cache_dtype
buffer_pool_size_MB
buffer_reuse_count
buffer_reuse_hit_rate
delta_streamed_layers
delta_materialized_layers
temp_tensor_saved_count
temp_tensor_recomputed_count
inplace_update_used
optimizer_state_overlap_removed
```

定义 memory reduction：

$$
\Delta_M = \frac{R_M^{current}-R_M^{candidate}}{R_M^{current}}.
$$

定义 step penalty：

$$
P_T = \frac{R_T^{candidate}-R_T^{current}}{R_T^{current}}.
$$

### 7.5 单因素成立标准

一个 single factor 被认为有效，必须满足：

$$
\Delta_M \geq 0.08,
$$

并且：

$$
P_T \leq 0.10.
$$

如果：

$$
\Delta_M \geq 0.15,
$$

即使 $P_T$ 达到 $0.20$，也可进入 P3 组合测试，但不能单独作为 survivor。

如果 gradient correctness 失败：

$$
\operatorname{relerr}(\nabla) \geq 10^{-4}
$$

或：

$$
\cos(\nabla)<0.999,
$$

则该 factor 直接淘汰，无论 memory 是否下降。

### 7.6 可视化

P2 必须画 single-factor memory-time Pareto。横轴是 $
\Delta_M$，纵轴是 $P_T$，右下角代表好候选：memory 降得多，time penalty 小。

还必须画 repair factor waterfall，展示每个 factor 对 `cache_x`、`cache_hidden`、`cache_delta`、`workspace_temp`、`optimizer_state_overlap` 的影响。

---

## 8. P3：combined memory package interaction matrix

### 8.1 目的

P3 测组合包，因为 memory peak 往往来自多个因素叠加。单因素有效不代表组合有效，单因素无效也可能在组合中发挥作用。P3 要找出最小可用组合，而不是堆所有优化。

### 8.2 组合候选

```text
C0-current
C1-bufferReuse+deltaStreaming
C2-bufferReuse+bf16Cache
C3-deltaStreaming+bf16Cache
C4-bufferReuse+noTempPoly2
C5-bufferReuse+deltaStreaming+bf16Cache
C6-bufferReuse+deltaStreaming+updateInPlaceSafe
C7-allMemoryOptimized-light
C8-allMemoryOptimized-full
```

其中：

```text
allMemoryOptimized-light = bufferReuse + deltaStreaming + updateInPlaceSafe
allMemoryOptimized-full = bufferReuse + deltaStreaming + bf16Cache + noTempPoly2 + updateInPlaceSafe + optimizerStateSplit
```

### 8.3 组合包选择标准

P3 有三层 gate。

Memory exploratory gate：

$$
R_M < 1.05.
$$

Memory pass gate：

$$
R_M < 1.00.
$$

Memory target gate：

$$
R_M \leq 0.90.
$$

Time gate：

$$
R_T \leq 1.35
$$

用于 exploratory task re-entry；

$$
R_T \leq 1.20
$$

用于 official task re-entry。

如果某组合满足 $R_M<1.0$ 但 $R_T>1.60$，记为 `memory_only_candidate`，不进入 task，只用于证明 memory route 可行。

### 8.4 必须记录字段

```text
combination_id
included_factors
memory_ratio_min
memory_ratio_mean
memory_ratio_max
step_ratio_min
step_ratio_mean
step_ratio_max
backward_ratio_mean
grad_relerr_max
grad_cos_min
memory_pass_count
step_pass_count
both_pass_count
failure_dataset
failure_batch
failure_depth
interaction_gain_memory
interaction_penalty_time
```

定义 interaction gain：

$$
G_{int}=\Delta_M^{combined}-\sum_i\Delta_M^{factor_i}.
$$

如果 $G_{int}<0$，说明组合优化互相抵消；如果 $G_{int}>0$，说明组合有协同效果。

### 8.5 可视化

P3 必须画组合 heatmap。行是 factor A，列是 factor B，颜色是 memory ratio，格子中写 step ratio。这样能看出哪些组合真正有效。

还要画 pass-count bar，显示每个组合在 $3\times3\times2$ setting 中通过 memory/time gate 的次数。

---

## 9. P4：runtime repair and kernel launch audit

### 9.1 目的

如果 P3 出现 memory near-pass 或 memory pass，但 step time 仍然偏高，P4 专门分析 runtime。v6.4 的 current best step ratio 仍约 $1.613$，说明 time issue 不小于 memory issue。

P4 的目标是把 runtime 分成：

```text
forward transform
mixing GEMM
loss delta
manual backward input-gradient
manual backward coeff-gradient
manual update
optimizer state update
kernel launch overhead
```

### 9.2 必须记录字段

```text
forward_transform_ms
mixing_gemm_ms
loss_delta_ms
backward_input_grad_ms
backward_coeff_grad_ms
manual_update_ms
optimizer_state_update_ms
kernel_launch_count_forward
kernel_launch_count_backward
kernel_launch_count_update
elementwise_kernel_count
small_kernel_count
gemm_kernel_count
avg_kernel_duration_us
largest_kernel_duration_us
python_loop_count
sync_count
cuda_graph_available
compiled_graph_break_count
```

### 9.3 判断标准

如果 small kernel 占比高：

$$
\frac{N_{small\ kernel}}{N_{kernel}} > 0.50,
$$

并且 step ratio 高，则下一步需要 kernel fusion 或 CUDA graph capture。

如果 transform cost 占比高：

$$
\frac{T_{transform}}{T_{step}} > 0.25,
$$

则优先优化 poly2 transform，考虑 fused polynomial transform 和 derivative output。

如果 optimizer state update 占比高：

$$
\frac{T_{optimizer}}{T_{step}} > 0.20,
$$

则 ManualAdanLite / AdamW state update 可能需要 fused update 或 role-wise state simplification。

### 9.4 可视化

P4 必须画 runtime waterfall。每个 candidate 一张 stacked bar，分解 step time。还要画 kernel-count vs step-ratio scatter，判断 step ratio 是否主要由 kernel launch 决定。

---

## 10. P5：one-step gradient/loss correctness probe

### 10.1 目的

P1-P4 主要看 memory/time。P5 只做 one-step correctness，不做 full task。它验证 memory-optimized candidate 是否真实保持 task descent。

### 10.2 方法

对 P3/P4 survivor，在每个 dataset 上抽取：

```text
train batch
holdout batch
validation mini-batch
```

执行：

```text
1. clone weights
2. compute manual gradient
3. apply one candidate update
4. evaluate train/holdout/val loss
5. rollback weights
6. verify rollback exactness
```

### 10.3 必须记录字段

```text
method
dataset
seed
batch_size
train_loss_before
train_loss_after
holdout_loss_before
holdout_loss_after
val_loss_before
val_loss_after
train_descent
holdout_descent
val_descent
train_descent_ratio_vs_current
holdout_descent_ratio_vs_current
val_descent_ratio_vs_current
update_norm
update_over_param_norm
grad_norm
grad_relerr
grad_cos
rollback_error_after_probe
nan_count
inf_count
```

### 10.4 通过标准

One-step pass 要求：

$$
\Delta L_{train}<0,
$$

$$
\Delta L_{holdout}\leq 0.05|\Delta L_{train}|,
$$

或更强：

$$
\Delta L_{holdout}<0.
$$

同时：

$$
\frac{\Delta L_{holdout}^{candidate}}{\Delta L_{holdout}^{current}}\geq0.95
$$

当 current 也有 holdout descent 时。

Rollback 要求：

$$
\operatorname{rollback\_error}<10^{-8}.
$$

### 10.5 可视化

画 actual descent scatter。横轴是 train descent，纵轴是 holdout descent。健康候选应该落在左下区域，即 train 和 holdout 都下降。

还要画 update norm vs holdout descent，检查 memory optimization 是否改变了更新尺度。

---

## 11. P6：P1 survivor selection and route decision

### 11.1 目的

P6 汇总 P1-P5，决定是否打开 task re-entry。

### 11.2 survivor 类型

定义四类 candidate：

```text
S0 official memory-time survivor:
  R_M < 1.0 and R_T <= 1.20 and gradient/loss probe pass

S1 memory survivor:
  R_M < 1.0 and R_T <= 1.35 and gradient/loss probe pass

S2 memory-only diagnostic:
  R_M < 1.0 but R_T > 1.35

S3 no survivor:
  all candidates R_M >= 1.0
```

### 11.3 route decision

如果存在 S0，进入 P7 official task re-entry。

如果只有 S1，进入 P7 exploratory task re-entry，但不能进入 confirm。

如果只有 S2，不进入 task，继续 runtime fusion。

如果是 S3，进入 P9 fallback primitive decision。

### 11.4 输出 artifact

P6 必须输出：

```text
p6_survivor_selection.csv
route_decision.json
aggregate_decision.json
```

`route_decision.json` 必须包含：

```json
{
  "route": "...",
  "best_memory_ratio": 0.0,
  "best_step_ratio": 0.0,
  "survivor_type": "S0/S1/S2/S3",
  "open_task_reentry": true,
  "open_functional_correction": false,
  "no_proxy": true
}
```

---

## 12. P7：gated task re-entry with ManualAdanLite

### 12.1 触发条件

P7 只有在 P6 存在 S0 或 S1 时运行。P7 的目标不是 final confirm，而是验证 memory survivor 是否还能学任务。

### 12.2 方法

固定主线：

```text
DWM2-poly2-memorySurvivor + ManualAdanLite
DWM2-poly2-memorySurvivor + ManualAdamW
MLP-autograd-reference
MLP-manual-linear-reference
```

不再跑 RoleWiseLR InputHeavy / OutputWarmup，因为之前已经明显退化。

### 12.3 数据与训练 budget

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

train/val/test:
  1536/512/512 for re-entry smoke

budget:
  240 steps
```

### 12.4 必须记录字段

```text
train_loss_trace
val_loss_trace
test_acc_trace
val_acc_trace
ECE_trace
NLL_trace
margin_mean_trace
margin_p10_trace
val_loss_auc_by_step
val_loss_auc_by_time
val_acc_auc_by_step
val_acc_auc_by_time
time_to_target_loss
time_to_target_acc
step_time_ms_mean
step_time_ms_p95
backward_memory_ratio_mean
backward_memory_ratio_p95
samples_per_sec
nan_step_count
bad_step_count
```

### 12.5 通过标准

Task re-entry exploratory pass：

$$
\operatorname{Acc}_{candidate}\geq\operatorname{Acc}_{MLP}-0.02
$$

on all datasets, and no NaN/Inf.

Official task pass：

$$
\operatorname{Acc}_{candidate}\geq\operatorname{Acc}_{MLP}-0.01
$$

on all datasets, and at least two datasets satisfy:

$$
T_{target,candidate}\leq T_{target,MLP}.
$$

如果 P7 失败，不能打开 functional correction。原因是 functional correction 不应该拯救一个 task learner 失败的 primitive。

### 12.6 可视化

P7 必须画：

```text
val_loss_vs_step.svg
val_loss_vs_time.svg
acc_vs_time.svg
time_to_target_bar.svg
task_efficiency_pareto.svg
calibration_curve_smoke.svg
```

---

## 13. P8：gated functional correction smoke

### 13.1 触发条件

P8 只有在 P7 official task pass 后运行。P8 不允许 standalone functional update，只允许 task-preserving correction：

$$
d_{new}=d_{task}+\lambda d_{geo}.
$$

### 13.2 候选 correction

```text
C0 task-only baseline
C1 task + curvature correction lambda=0.01
C2 task + curvature correction lambda=0.03
C3 task + residual smooth correction lambda=0.01
C4 task + residual smooth correction lambda=0.03
C5 task + adaptive lambda correction
C6 task + late-phase correction only
```

### 13.3 Direction audit 指标

```text
cos_new_task
holdout_descent_ratio
bad_step_rate
accepted_lambda
lambda_backtrack_count
fallback_to_task_only
geometry_reduction_phi
geometry_reduction_curvature
ECE_delta_proxy
logit_drift
feature_drift
```

### 13.4 通过标准

Correction smoke pass：

$$
\cos(d_{new},d_{task})\geq0.85,
$$

$$
\frac{\operatorname{HoldoutDescent}(d_{new})}{\operatorname{HoldoutDescent}(d_{task})}\geq0.95,
$$

$$
\operatorname{BadStepRate}(d_{new})\leq0.02,
$$

并且至少一个 geometry 指标改善：

$$
\operatorname{Curvature}_{new}<0.95\operatorname{Curvature}_{task}
$$

或：

$$
\phi'_{p95,new}<0.95\phi'_{p95,task}.
$$

如果 correction 失败，route decision 应该写：

```text
memory/task learner available, functional correction not ready.
```

而不是回头否定 memory candidate。

### 13.5 可视化

P8 必须画：

```text
direction_quality_scatter.svg
geometry_gain_vs_task_cost.svg
lambda_acceptance_histogram.svg
bad_step_heatmap.svg
geometry_curve_with_correction_events.svg
```

---

## 14. P9：fallback primitive decision and final diagnosis

### 14.1 触发条件

如果 P6 没有 S0/S1，或者 DWM2-poly2 多轮 memory package 仍然无法过线，则 P9 启动 fallback decision。

### 14.2 fallback options

```text
F0 DWM2-poly3
F1 DWM2-poly2+silu-base
F2 RationalKAT-lite-fastpoly
F3 SparseInterp fused rewrite
F4 DWM2-RBFK2 optimized
```

### 14.3 fallback 初筛指标

fallback 不做 full task，只做 P1-style profiling 和 one-step probe。

必须记录：

```text
memory_ratio
step_ratio
grad_relerr
grad_cos
one_step_holdout_descent
edge_param_count
kernel_count
manual_cache_MB
workspace_temp_MB
```

fallback exploratory pass：

$$
R_M<1.10,
$$

$$
R_T<1.50,
$$

$$
\cos(\nabla)>0.999.
$$

如果 fallback 也没有任何 near-pass，v6.5 route 应明确写：

```text
Current graph-free PureKAN primitive family still fails terminal memory/time envelope.
Need lower-level fused CUDA/Triton implementation or new primitive family.
```

---

## 15. 统一 CSV 字段规范

为了避免后续分析混乱，v6.5 所有 CSV 至少包含下面公共字段。

### 15.1 公共字段

```text
stage
method
variant_id
dataset
seed
batch_size
hidden_dim
depth
device
run_id
wandb_run
artifact_hash
fake_data_used
proxy_row_used
used_for_gate
implementation_status
```

### 15.2 memory 字段

```text
forward_peak_MB
loss_delta_peak_MB
backward_adjoint_peak_MB
update_peak_MB
optimizer_state_peak_MB
total_step_peak_MB
reserved_peak_MB
manual_cache_total_MB
cache_x_MB
cache_hidden_MB
cache_delta_MB
cache_temp_MB
workspace_temp_MB
parameter_MB
optimizer_state_MB
largest_temp_tensor_MB
num_tensor_allocations
memory_ratio_vs_MLP
phase_peak_explain_ratio
unexplained_memory_gap_MB
```

### 15.3 runtime 字段

```text
forward_time_ms
loss_delta_time_ms
manual_backward_time_ms
manual_update_time_ms
optimizer_state_update_time_ms
step_time_ms
forward_time_ratio_vs_MLP
backward_time_ratio_vs_MLP
step_time_ratio_vs_MLP
kernel_count_forward
kernel_count_backward
kernel_count_update
elementwise_kernel_count
gemm_kernel_count
small_kernel_count
python_loop_count
sync_count
```

### 15.4 correctness 字段

```text
grad_relerr
grad_cos
manual_forward_relerr
manual_forward_max_abs
train_descent
holdout_descent
val_descent
rollback_error
nan_count
inf_count
```

### 15.5 task 字段

```text
train_loss
val_loss
test_acc
val_acc
ECE
NLL
margin_mean
margin_p10
val_loss_auc_by_step
val_loss_auc_by_time
val_acc_auc_by_step
val_acc_auc_by_time
time_to_target_loss
time_to_target_acc
samples_per_sec
```

### 15.6 functional correction 字段

```text
cos_new_task
holdout_descent_ratio
bad_step_rate
accepted_lambda
lambda_backtrack_count
fallback_to_task_only
phi_prime_p95
curvature
geometry_reduction_phi
geometry_reduction_curvature
logit_drift
feature_drift
correction_overhead_ms
```

---

## 16. 必须生成的 artifact

v6.5 必须生成如下文件。没有运行的阶段必须输出 `not_run` 状态文件，而不是缺失。

```text
p0_contract.csv
p0_contract_heatmap.svg
p1_phase_local_memory.csv
p1_memory_attribution.csv
p1_memory_waterfall.svg
p1_actual_peak_vs_cache.svg
p1_unexplained_gap_heatmap.svg
p2_single_factor_memory_repair.csv
p2_factor_pareto.svg
p2_memory_reduction_waterfall.svg
p3_combined_memory_packages.csv
p3_combination_heatmap.svg
p3_pass_count_bar.svg
p4_runtime_kernel_audit.csv
p4_runtime_waterfall.svg
p4_kernel_count_scatter.svg
p5_one_step_probe.csv
p5_actual_descent_scatter.svg
p6_survivor_selection.csv
p7_task_reentry.csv
p7_task_trace.csv
p8_functional_correction_smoke.csv
p9_fallback_primitive_decision.csv
failure_table.csv
route_decision.json
aggregate_decision.json
```

---

## 17. 必须可视化的图

### 17.1 Phase-local memory waterfall

每个候选显示 forward、loss delta、backward adjoint、update、optimizer state 的 peak memory。目的是判断 total peak 到底在哪个 phase 出现。

### 17.2 Actual peak vs manual cache scatter

横轴 manual cache，纵轴 actual CUDA peak。若候选点远离对角线，说明 manual cache 不是主要 peak 来源。

### 17.3 Unexplained memory gap heatmap

行是 dataset/batch/depth，列是 variant，颜色为：

$$
\frac{G}{M_{total\ peak}}.
$$

这张图决定下一步是做 workspace/buffer reuse，还是 cache compression。

### 17.4 Single-factor memory-time Pareto

横轴：

$$
\Delta_M.
$$

纵轴：

$$
P_T.
$$

理想点在右下角。

### 17.5 Combination heatmap

行列都是 memory repair factor，颜色是 memory ratio，格子文字是 step ratio。用于判断组合是否协同。

### 17.6 Runtime waterfall

分解 step time，让我们知道 time-blocked 主要来自 transform、backward coeff grad、manual update 还是 kernel launch。

### 17.7 Actual descent scatter

横轴 train descent，纵轴 holdout descent。健康候选应位于 train/holdout 同时下降区域。

### 17.8 Task re-entry curves

如果 P7 运行，必须画 val loss vs step、val loss vs wall-clock、accuracy vs wall-clock 和 time-to-target bar。

### 17.9 Functional correction direction scatter

如果 P8 运行，横轴：

$$
\cos(d_{new},d_{task}),
$$

纵轴：

$$
\frac{\operatorname{HoldoutDescent}(d_{new})}{\operatorname{HoldoutDescent}(d_{task})}.
$$

颜色表示 geometry reduction。

---

## 18. 最终 route decision 规则

v6.5 必须按下面规则输出最终 route，不能只写“失败”。

### R1：memory/time survivor found

条件：存在 S0 candidate。

决策：打开 task re-entry，继续验证 ManualAdanLite / ManualAdamW。

### R2：memory survivor found, time still high

条件：存在 S1/S2 candidate，但 $R_T$ 偏高。

决策：不跑 confirm，继续 runtime fusion / kernel launch repair。

### R3：memory still blocked

条件：所有 DWM2-poly2 package 都满足 $R_M\geq1.0$。

决策：继续 memory kernel 或进入 fallback primitive。

### R4：memory pass but one-step descent fails

条件：$R_M<1.0$，但 P5 one-step probe 失败。

决策：memory package 破坏 task gradient，回退或修 numerical path。

### R5：DWM2-poly2 exhausted

条件：memory/time 多轮失败，fallback primitive 出现 better near-pass。

决策：切换主线到 fallback primitive。

### R6：graph-free primitive family limit

条件：DWM2、Rational、SparseInterp fallback 都没有 memory/time near-pass。

决策：当前 PureKAN primitive family 不满足 terminal efficiency target，需要更底层 fused CUDA/Triton 或新 primitive family。

---

## 19. 本轮预期结论形式

v6.5 最理想结论是：

```text
DWM2-poly2 allMemoryOptimized produces an official memory-time survivor:
R_M < 1.0, R_T <= 1.20, grad correctness pass, one-step descent pass.
Task re-entry can be opened.
```

次优结论是：

```text
A memory-only survivor exists:
R_M < 1.0 but R_T remains high.
This proves memory route is possible, but runtime fusion remains required.
```

失败但有价值的结论是：

```text
Hidden cache is not the blocker; buffer/delta/workspace are quantified;
DWM2-poly2 current implementation remains memory/time blocked;
fallback primitive decision is triggered.
```

最不能接受的结论是：

```text
P1 failed, but P7/P8 still ran and produced task/functional claims.
```

这种结论在 v6.5 中必须被禁止。

---

## 20. 一句话总结

v6.5 的核心不是扩大 task 实验，而是把 v6.4 的 R3 结论彻底闭环：

$$
\boxed{\text{找出 backward memory peak 的真实来源，并通过 buffer reuse / delta streaming / bf16 cache / fused workspace policy 产生真实 memory survivor。}}
$$

如果产生 survivor，再重新打开 ManualAdanLite task re-entry；如果没有 survivor，就停止在 DWM2-poly2 上堆上层实验，进入 fallback primitive 或更底层 fused kernel 设计。
